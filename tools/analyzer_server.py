#!/usr/bin/env python3
"""
Persistent analysis worker that stays alive between jobs.

Loads models once on startup, then reads job requests from stdin as JSON lines
and emits progress/results on stdout. This eliminates the ~10s model reload
overhead between listings.

Protocol:
  Startup:  {"type": "ready"}
  Request:  {"type": "job", "jobId": "...", "propertyKey": "...", "images": [...], ...}
  Response: {"type": "progress", "jobId": "...", ...}
            {"type": "result", "jobId": "...", ...}
            {"type": "job_done", "jobId": "..."}
            {"type": "error", "jobId": "...", "error": "..."}
"""

import asyncio
import json
import logging
import os
import sys
import signal
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import config first
try:
    from tools import pipeline_config as cfg
except Exception as exc:
    print(json.dumps({"type": "error", "error": f"Failed to import pipeline_config: {exc}"}),
          flush=True)
    sys.exit(1)

# Import shared utilities from analyzer_cli
from tools.analyzer_cli import (
    _apply_env_overrides,
    _build_summary,
    _compute_timing_stats,
    _log_timing_stats,
    ImageResult,
    PropertyAnalysisJob,
    install_payload_redactor,
)

# Optional: import pass config if available
try:
    from tools.pass_config import PassToggles, PassModelOverrides, SceneClassifierRunOptions
    PASS_CONFIG_AVAILABLE = True
except ImportError:
    PASS_CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Shutdown flag
_shutdown_requested = False


def _emit(obj: dict) -> None:
    """Write a JSON line to stdout and flush."""
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _setup_signal_handlers():
    """Register signal handlers for graceful shutdown."""
    global _shutdown_requested

    def _handler(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        logger.info(f"Received signal {signum}, will shut down after current job")

    signal.signal(signal.SIGTERM, _handler)
    # On Windows, SIGINT may not work the same way in a subprocess,
    # but we handle it anyway for cross-platform correctness
    signal.signal(signal.SIGINT, _handler)


def main() -> int:
    global _shutdown_requested

    # ─────────────────────────────────────────────────────────────────────────
    # Logging to stderr (stdout is reserved for JSON protocol)
    # ─────────────────────────────────────────────────────────────────────────
    debug = os.environ.get("ANALYZER_SERVER_DEBUG", "").lower() in ("1", "true", "yes")
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    install_payload_redactor()
    _setup_signal_handlers()

    # ─────────────────────────────────────────────────────────────────────────
    # One-time startup: apply env overrides and load all heavy modules
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Persistent analyzer server starting...")
    _apply_env_overrides()

    try:
        from tools.artifact_writers import write_photo_intel, load_issue_catalog
        from tools.scene_classifier_orchestrator import create_orchestrator_from_config
        from tools.vlm_client import get_model_configs_from_pipeline_config, create_vlm_client
    except Exception as exc:
        logger.error(f"Failed to import pipeline components: {exc}", exc_info=True)
        _emit({"type": "error", "error": f"Failed to import pipeline components: {exc}"})
        return 1

    # Load issue catalog
    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)
    logger.info(f"Issue catalog loaded ({len(catalog.get('items', []))} items)")

    # Build embeddings retriever (the expensive SentenceTransformer load)
    candidate_provider = None
    if getattr(cfg, "USE_EMBEDDINGS_CATALOG", False):
        try:
            from tools.catalog_embeddings import CatalogEmbeddingsRetriever, build_guardrails_from_catalog
            from dataclasses import asdict

            retriever = CatalogEmbeddingsRetriever(
                catalog_v2=catalog,
                model_name=getattr(cfg, "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                device=getattr(cfg, "EMBEDDINGS_DEVICE", "cpu"),
                trust_remote_code=getattr(cfg, "EMBEDDINGS_TRUST_REMOTE_CODE", False),
                default_topk=getattr(cfg, "EMBEDDINGS_TOPK", 10),
                guardrails=build_guardrails_from_catalog(catalog),
            )

            def candidate_provider(observation_text: str, context: dict) -> list:
                kind = (context.get("kind") or "").strip().lower()
                topk = context.get("top_k_candidates")
                scene_group = context.get("scene_group")
                allowed_groups = {scene_group} if scene_group else None
                allowed_kinds = {kind} if kind in ("defect", "upgrade") else None
                matches = retriever.retrieve_candidates(
                    observation_text,
                    topk=topk,
                    allowed_kinds=allowed_kinds,
                    allowed_groups=allowed_groups,
                )
                return [asdict(m) for m in matches]

            logger.info(f"Embeddings retriever ready (model={retriever.model_name}, items={len(retriever._items)})")
        except Exception as exc:
            logger.warning(f"Could not initialize embeddings retriever: {exc}. Pass 2d will be skipped.")
            candidate_provider = None

    # Create orchestrator (reused across all jobs)
    orchestrator = create_orchestrator_from_config(
        cfg,
        candidate_provider=candidate_provider,
        catalog_items=catalog.get("items"),
    )

    # Get GPT config and VLM client (reused across all jobs)
    _, gpt5_config = get_model_configs_from_pipeline_config(cfg)
    vlm_client = create_vlm_client()

    logger.info("All models loaded. Server ready.")
    _emit({"type": "ready"})

    # ─────────────────────────────────────────────────────────────────────────
    # Job loop: read JSON requests from stdin, process, emit results
    # ─────────────────────────────────────────────────────────────────────────
    while not _shutdown_requested:
        try:
            line = sys.stdin.readline()
        except EOFError:
            break

        if not line:
            # EOF — parent closed stdin
            logger.info("stdin closed (EOF), shutting down")
            break

        line = line.strip()
        if not line:
            continue

        # Parse job request
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.error(f"Invalid JSON on stdin: {exc}")
            continue

        job_id_from_ts = request.get("jobId", "unknown")

        try:
            _process_job(
                request=request,
                orchestrator=orchestrator,
                catalog=catalog,
                gpt5_config=gpt5_config,
                vlm_client=vlm_client,
                write_photo_intel=write_photo_intel,
            )
        except Exception as exc:
            logger.error(f"Unhandled error in job {job_id_from_ts}: {exc}", exc_info=True)
            _emit({
                "type": "error",
                "jobId": job_id_from_ts,
                "error": traceback.format_exc(),
            })
            _emit({"type": "job_done", "jobId": job_id_from_ts})

    logger.info("Analyzer server exiting")
    return 0


def _process_job(
    request: dict,
    orchestrator: Any,
    catalog: dict,
    gpt5_config: Any,
    vlm_client: Any,
    write_photo_intel: Any,
) -> None:
    """Process a single analysis job."""
    ts_job_id = request.get("jobId", "unknown")
    property_key = request["propertyKey"]
    image_paths = [Path(p).resolve() for p in request["images"]]
    artifacts_root = Path(request["artifactsRoot"]).resolve()
    analysis_profile = request.get("analysisProfile", "standard")
    detection_backend = request.get("detectionBackend", "dinox")
    concurrency = request.get("concurrency", int(os.environ.get("ANALYZER_CONCURRENCY", "3")))

    logger.info(f"[Job {ts_job_id}] Starting: {property_key} ({len(image_paths)} images)")

    # Build run options from profile
    options = SceneClassifierRunOptions.from_analysis_profile(
        analysis_profile=analysis_profile,
    )

    # Generate internal job ID and create artifacts directory
    internal_job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = artifacts_root / property_key / internal_job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-image analysis loop
    # ─────────────────────────────────────────────────────────────────────────
    results: List[ImageResult] = []
    total_images = len(image_paths)
    total_start = time.time()

    async def _analyze_all():
        sem = asyncio.Semaphore(concurrency)
        completed = 0

        async def _analyze_one(idx, image_path):
            nonlocal completed
            async with sem:
                img_start = time.time()
                logger.info(f"  [start] Analyzing: {image_path.name}")

                try:
                    img_options = options.with_meta(
                        run_id=internal_job_id,
                        photo_key=image_path.name,
                        property_key=property_key,
                    )
                    analysis = await orchestrator.analyze_image(
                        image_path=image_path,
                        options=img_options,
                    )
                    elapsed = time.time() - img_start

                    img_result = ImageResult(
                        image_path=str(image_path),
                        scene_data=analysis.to_dict(),
                        scene=analysis.scene or "unknown",
                        processing_time=elapsed,
                    )
                    logger.info(f"    ✅ {image_path.name} → {analysis.scene} ({elapsed:.1f}s)")

                except Exception as exc:
                    elapsed = time.time() - img_start
                    logger.error(f"    ❌ {image_path.name} failed: {exc}", exc_info=True)
                    img_result = ImageResult(
                        image_path=str(image_path),
                        scene="unknown",
                        processing_time=elapsed,
                        error=str(exc),
                    )

                completed += 1
                _emit({
                    "type": "progress",
                    "jobId": ts_job_id,
                    "itemsDone": completed,
                    "itemsTotal": total_images,
                    "progress": round((completed / total_images) * 100),
                })
                return img_result

        tasks = [_analyze_one(i, img) for i, img in enumerate(image_paths)]
        results.extend(await asyncio.gather(*tasks))

    asyncio.run(_analyze_all())

    total_time = time.time() - total_start

    # ─────────────────────────────────────────────────────────────────────────
    # Build job and write artifacts
    # ─────────────────────────────────────────────────────────────────────────
    job = PropertyAnalysisJob(
        property_key=property_key,
        job_id=internal_job_id,
        artifacts_dir=str(job_dir),
        timestamp=datetime.utcnow().isoformat() + "Z",
        results=results,
        total_processing_time=total_time,
    )

    try:
        photo_intel_path = write_photo_intel(
            cfg=cfg,
            job=job,
            detection_backend=detection_backend,
            analysis_profile=analysis_profile,
            use_pass_architecture=True,
            pass_toggles={},
            model_overrides={},
            gpt_config=gpt5_config,
            issue_catalog=catalog,
            vlm_client=vlm_client,
        )
    except Exception as exc:
        logger.error(f"Failed to write photo_intel: {exc}", exc_info=True)
        photo_intel_path = None

    # Compute and log timing statistics
    timing_stats = _compute_timing_stats(results, total_time)
    _log_timing_stats(timing_stats, property_key)

    # Build summary (same format as analyzer_cli)
    summary = _build_summary(
        job,
        photo_intel_path=photo_intel_path,
        detection_backend=detection_backend,
        analysis_profile=analysis_profile,
        used_pass_architecture=True,
        timing_stats=timing_stats,
    )

    logger.info(f"[Job {ts_job_id}] Complete: {property_key} ({total_time:.1f}s)")

    # Emit result and job_done marker
    _emit({"type": "result", "jobId": ts_job_id, **summary})
    _emit({"type": "job_done", "jobId": ts_job_id})


if __name__ == "__main__":
    sys.exit(main())
