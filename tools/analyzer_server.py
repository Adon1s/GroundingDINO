#!/usr/bin/env python3
"""
Persistent analysis worker that stays alive between jobs.

Loads models once on startup, then reads job requests from stdin as JSON lines
and emits progress/results on stdout. This eliminates the ~10s model reload
overhead between listings.

Protocol:
  Startup:  {"type": "ready"}
  Request:  {"type": "job", "jobId": "...", "runId": "...", "propertyKey": "...", "images": [...], ...}
            (runId is the stable checkpoint key across retries; jobId is ephemeral.
            If runId is missing, jobId is used as a fallback with a warning.)
  Response: {"type": "progress", "jobId": "...", ...}
            {"type": "resumed", "jobId": "...", "completed": N, "total": M}
            {"type": "result", "jobId": "...", ...}
            {"type": "job_done", "jobId": "..."}
            {"type": "error", "jobId": "...", "error": "..."}
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import signal
import time
import traceback
import uuid
from dataclasses import asdict, fields
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
            from tools.scene_classifier_passes import prioritize_resolution_candidates
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
                allowed_kinds_ctx = context.get("allowed_kinds")
                if allowed_kinds_ctx:
                    allowed_kinds = {
                        str(k).strip().lower()
                        for k in allowed_kinds_ctx
                        if str(k).strip().lower() in {"defect", "upgrade"}
                    }
                else:
                    allowed_kinds = {kind} if kind in ("defect", "upgrade") else None
                widened_routing = bool(allowed_kinds and len(allowed_kinds) > 1)
                requested_topk = topk
                if widened_routing and topk:
                    requested_topk = max(int(topk), int(topk) * 2)
                matches = retriever.retrieve_candidates(
                    observation_text,
                    topk=requested_topk,
                    allowed_kinds=allowed_kinds,
                    allowed_groups=allowed_groups,
                )
                candidates = [asdict(m) for m in matches]
                return prioritize_resolution_candidates(candidates, widened_routing=widened_routing)

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


def _save_image_checkpoint(ckpt_dir: Path, idx: int, image_result: ImageResult) -> None:
    """Atomically persist one image's result. Best-effort; raises only on programmer error."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / f"image_{idx:04d}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(asdict(image_result), default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, target)


def _load_checkpoint(ckpt_dir: Path, n_images: int) -> Dict[int, ImageResult]:
    """Load any prior per-image checkpoints. Tolerant of corrupt files and schema drift."""
    if not ckpt_dir.is_dir():
        return {}
    out: Dict[int, ImageResult] = {}
    known_fields = {f.name for f in fields(ImageResult)}
    for f in sorted(ckpt_dir.glob("image_*.json")):
        try:
            idx = int(f.stem.split("_")[1])
            if idx >= n_images:
                continue
            raw = json.loads(f.read_text(encoding="utf-8"))
            raw = {k: v for k, v in raw.items() if k in known_fields}
            out[idx] = ImageResult(**raw)
        except (ValueError, json.JSONDecodeError, TypeError, OSError) as exc:
            logger.warning(f"Skipping unreadable checkpoint {f}: {exc}")
    return out


def _clear_checkpoint(ckpt_dir: Path) -> None:
    """Remove the checkpoint dir on successful job completion."""
    if ckpt_dir.is_dir():
        shutil.rmtree(ckpt_dir, ignore_errors=True)


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
    run_id = request.get("runId")
    if not run_id:
        logger.warning(
            f"[Job {ts_job_id}] No runId in request — falling back to jobId as checkpoint key. "
            "Upgrade the TS client to send runId for stable resume across retries."
        )
        run_id = ts_job_id
    property_key = request["propertyKey"]
    image_paths = [Path(p).resolve() for p in request["images"]]
    artifacts_root = Path(request["artifactsRoot"]).resolve()
    analysis_profile = request.get("analysisProfile", "standard")
    detection_backend = request.get("detectionBackend", "dinox")
    concurrency = request.get("concurrency", int(os.environ.get("ANALYZER_CONCURRENCY", "3")))

    logger.info(f"[Job {ts_job_id}] Starting: {property_key} ({len(image_paths)} images)")

    # Reset token counters so this job's TIMING STATS reflect only its own usage.
    try:
        vlm_client.reset_usage_stats()
    except AttributeError:
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Per-job model overrides (top-picks-mini batch + similar experiments)
    #
    # The orchestrator reads `getattr(cfg, "GPT_PASS_{KEY}_MODEL", None)` at
    # runtime per pass, so we can mutate cfg attrs per job and restore them
    # after — this lets the long-running persistent worker honor per-job model
    # swaps without restarting. The family side is handled via PassModelOverrides
    # forced to 'gpt5', so passes that route to Qwen by default still pick up
    # the GPT model name.
    # ─────────────────────────────────────────────────────────────────────────
    raw_overrides = request.get("modelOverrides") or {}
    if not isinstance(raw_overrides, dict):
        raw_overrides = {}

    # Same allowlist as the TS side (lib/analysis/modelOverrides.ts).
    _ALLOWED_OVERRIDE_KEYS = {"1a", "1b", "1c", "2a", "2b", "2c", "2d", "2f"}
    filtered_overrides: Dict[str, str] = {}
    for k, v in raw_overrides.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if k not in _ALLOWED_OVERRIDE_KEYS:
            logger.warning(f"[Job {ts_job_id}] Ignoring unsupported override pass key: {k}")
            continue
        if not v.strip():
            continue
        filtered_overrides[k] = v

    # Family overrides: every overridden pass goes to gpt5 family (mirrors the
    # `--model-{pass} gpt5` flags emitted by the spawn-per-job path).
    family_overrides: Dict[str, str] = {k: "gpt5" for k in filtered_overrides.keys()}

    # Build run options. model_overrides forces the family per pass.
    options = SceneClassifierRunOptions.from_analysis_profile(
        analysis_profile=analysis_profile,
        model_overrides=family_overrides if family_overrides else None,
    )

    # Snapshot current cfg.GPT_PASS_*_MODEL values so we can restore in finally.
    # Using a sentinel so we know whether the attr existed at all (vs. being None).
    _SENTINEL = object()
    cfg_snapshot: Dict[str, Any] = {}
    for pass_key, model_name in filtered_overrides.items():
        attr = f"GPT_PASS_{pass_key.upper()}_MODEL"
        cfg_snapshot[attr] = getattr(cfg, attr, _SENTINEL)
        setattr(cfg, attr, model_name)

    if filtered_overrides:
        logger.info(
            f"[Job {ts_job_id}] modelOverrides applied (cfg mutated): {filtered_overrides}"
        )

    # Generate internal job ID and create artifacts directory
    internal_job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = artifacts_root / property_key / internal_job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Per-run checkpoint dir, parallel to internal_job_id job dirs. The runId is
    # stable across retries so re-attempts of the same AnalysisRun resume from
    # disk; a fresh "Re-analyze" generates a new runId and starts clean.
    ckpt_dir = artifacts_root / property_key / ".checkpoints" / run_id
    cached_results: Dict[int, ImageResult] = _load_checkpoint(ckpt_dir, len(image_paths))
    if cached_results:
        logger.info(
            f"[Job {ts_job_id}] Resuming from checkpoint: "
            f"{len(cached_results)}/{len(image_paths)} images already complete"
        )
        _emit({
            "type": "resumed",
            "jobId": ts_job_id,
            "completed": len(cached_results),
            "total": len(image_paths),
        })

    # Wrap the entire job body in try/finally so cfg.GPT_PASS_*_MODEL is always
    # restored — critical because cfg is module-global and the persistent server
    # processes many jobs back-to-back. Without this, an exception mid-job would
    # leave the next job inheriting this job's overrides.
    try:
        # ─────────────────────────────────────────────────────────────────────
        # Per-image analysis loop
        # ─────────────────────────────────────────────────────────────────────
        results: List[ImageResult] = []
        total_images = len(image_paths)
        total_start = time.time()

        async def _analyze_all():
            sem = asyncio.Semaphore(concurrency)
            completed = 0

            async def _analyze_one(idx, image_path):
                nonlocal completed
                # Skip if a prior attempt already completed this image successfully.
                # Still emit progress so the frontend % advances during resume.
                if idx in cached_results and not cached_results[idx].error:
                    logger.info(f"  [cached] image {idx}: {image_path.name}")
                    completed += 1
                    _emit({
                        "type": "progress",
                        "jobId": ts_job_id,
                        "itemsDone": completed,
                        "itemsTotal": total_images,
                        "progress": round((completed / total_images) * 100),
                    })
                    return cached_results[idx]

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
                        # Aggregate tok/s accounting for concurrency:
                        # (avg tokens per image / this image's wall time) × concurrency.
                        # Uses the avg instead of a per-image delta so concurrent completions
                        # don't pollute one image's bucket.
                        tok_total = int(getattr(vlm_client, "usage_stats", {}).get("total_tokens", 0) or 0)
                        done_so_far = completed + 1  # this image counts; `completed` is incremented below
                        if tok_total > 0 and elapsed > 0 and done_so_far > 0:
                            per_image_rate = (tok_total / done_so_far) / elapsed
                            tps = per_image_rate * concurrency
                            tps_suffix = f", {tps:,.0f} tok/s"
                        else:
                            tps_suffix = ""
                        logger.info(f"    ✅ {image_path.name} → {analysis.scene} ({elapsed:.1f}s{tps_suffix})")

                    except Exception as exc:
                        elapsed = time.time() - img_start
                        logger.error(f"    ❌ {image_path.name} failed: {exc}", exc_info=True)
                        img_result = ImageResult(
                            image_path=str(image_path),
                            scene="unknown",
                            processing_time=elapsed,
                            error=str(exc),
                        )

                    # Persist successful images so a crash doesn't lose work.
                    # Failed images are intentionally not cached — a retry should re-attempt them.
                    if not img_result.error:
                        try:
                            _save_image_checkpoint(ckpt_dir, idx, img_result)
                        except OSError as exc:
                            logger.warning(f"  Checkpoint save failed for image {idx}: {exc}")

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

        # ─────────────────────────────────────────────────────────────────────
        # Build job and write artifacts
        # ─────────────────────────────────────────────────────────────────────
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
                # Pass the resolved per-pass model names so run.model_overrides
                # in photo_intel_debug.json reflects what was actually applied
                # (e.g. {"2a": "gpt-5.4-mini", ...}). Empty dict if no overrides.
                model_overrides=filtered_overrides,
                gpt_config=gpt5_config,
                issue_catalog=catalog,
                vlm_client=vlm_client,
            )
        except Exception as exc:
            logger.error(f"Failed to write photo_intel: {exc}", exc_info=True)
            photo_intel_path = None

        # Compute and log timing statistics
        timing_stats = _compute_timing_stats(
            results, total_time, usage_stats=getattr(vlm_client, "usage_stats", None)
        )
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

        # Only clear checkpoints on full success — leaves them intact on exception
        # or SIGKILL so the next attempt with the same runId can resume. Must stay
        # outside the finally block.
        _clear_checkpoint(ckpt_dir)
    finally:
        # Restore cfg.GPT_PASS_*_MODEL to its pre-job state. Always runs, even
        # on exceptions — guarantees subsequent jobs in the persistent worker
        # see clean defaults (or whatever cfg had before this job started).
        if cfg_snapshot:
            for attr, original in cfg_snapshot.items():
                if original is _SENTINEL:
                    # Attribute didn't exist before; remove it.
                    if hasattr(cfg, attr):
                        delattr(cfg, attr)
                else:
                    setattr(cfg, attr, original)
            logger.info(
                f"[Job {ts_job_id}] modelOverrides restored "
                f"({len(cfg_snapshot)} cfg attr{'s' if len(cfg_snapshot) != 1 else ''})"
            )


if __name__ == "__main__":
    sys.exit(main())
