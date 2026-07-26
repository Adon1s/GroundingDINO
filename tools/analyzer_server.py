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
import hashlib
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
    from tools.pass_config import (
        PassToggles,
        PassModelOverrides,
        SceneClassifierRunOptions,
        normalize_reasoning_efforts,
    )
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

    # Build embeddings retriever (the expensive model load).
    #
    # This is a hard startup dependency. The provider is built once and reused by
    # every job, so tolerating a failure here would silently strip Pass 2d from
    # every property this process ever handles. Refuse to signal ready instead.
    # (Per-job toggles are not known yet; 2d is enabled by default. A job that
    # disables 2d simply leaves the provider unused.)
    from tools.catalog_embeddings import EmbeddingsRuntimeError, build_candidate_provider
    try:
        candidate_provider = build_candidate_provider(catalog)
    except EmbeddingsRuntimeError as exc:
        logger.error(f"Embeddings retriever unavailable: {exc}")
        _emit({"type": "error", "stage": "embeddings_init", "error": str(exc)})
        return 1

    # One shared client supplies both photo-pass and Pass 2f telemetry.
    _, gpt5_config = get_model_configs_from_pipeline_config(cfg)
    vlm_client = create_vlm_client()

    # Create orchestrator (reused across all jobs)
    orchestrator = create_orchestrator_from_config(
        cfg,
        candidate_provider=candidate_provider,
        catalog_items=catalog.get("items"),
        vlm_client=vlm_client,
    )

    logger.info("All models loaded. Server ready.")
    _emit({"type": "ready", "dependency_status": {"embeddings": "ready"}})

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


def _ckpt_key(p: Path) -> tuple:
    """Identity tuple for matching a checkpoint entry to a current input path.

    Compares (parent_dir_name, filename). Basename alone is insufficient when
    multiple property folders use the same naming convention (e.g. photo_001.jpg
    appears in many redfin_<id> folders) — including the parent dir name catches
    cross-property contamination like the 337 → 386 Jackson Point Cir bug.
    """
    return (p.parent.name, p.name)


def _resolved_max_tokens() -> Dict[str, int]:
    """
    Effective OpenAI token cap per pass.

    Caps resolve from env vars and the per-pass defaults table only, so this does
    not depend on run options. Probes with a synthetic OpenAI config because the
    resolver is a no-op for local providers.
    """
    from tools.pass_config import ALL_PASSES, resolve_openai_invocation
    caps: Dict[str, int] = {}
    for pass_key in ALL_PASSES:
        resolved = resolve_openai_invocation(
            pass_key, {"provider": "openai", "model": "gpt-5"}
        )
        cap = resolved.get("max_output_tokens")
        if cap is not None:
            caps[pass_key] = int(cap)
    return caps


def _checkpoint_policy_fingerprint(
    model_overrides: Dict[str, str],
    reasoning_efforts: Dict[str, str],
    pass_toggles: Optional[Dict[str, bool]] = None,
    max_tokens: Optional[Dict[str, int]] = None,
) -> str:
    """
    Hash only non-secret routing inputs that affect reusable image results.

    pass_toggles and max_tokens are part of the policy: checkpoints written with
    Pass 2d disabled must not be reused by a run with 2d enabled (they contain no
    resolved catalog items), and a changed token cap changes truncation, which
    changes content.
    """
    payload = json.dumps(
        {
            "model_overrides": model_overrides,
            "reasoning_efforts": reasoning_efforts,
            "pass_toggles": pass_toggles or {},
            "max_tokens": max_tokens or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepare_checkpoint_dir(ckpt_dir: Path, policy_fingerprint: str) -> None:
    """Reset stale checkpoints, then atomically persist the current policy hash."""
    manifest_path = ckpt_dir / "policy.json"
    existing_fingerprint: Optional[str] = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_fingerprint = manifest.get("policy_fingerprint")
        except (OSError, ValueError, TypeError):
            existing_fingerprint = None

    has_images = ckpt_dir.is_dir() and any(ckpt_dir.glob("image_*.json"))
    if has_images and existing_fingerprint != policy_fingerprint:
        logger.warning(
            "Discarding incompatible image checkpoints: routing policy changed "
            f"({existing_fingerprint or 'legacy/missing'} -> {policy_fingerprint})"
        )
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"policy_fingerprint": policy_fingerprint}, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, manifest_path)


def _load_checkpoint(
    ckpt_dir: Path,
    n_images: int,
    image_paths: List[Path],
) -> Dict[int, ImageResult]:
    """Load any prior per-image checkpoints. Tolerant of corrupt files and schema drift.

    Each loaded entry is validated against `image_paths[idx]` — if the stored
    image_path's (parent, filename) doesn't match the current input, the
    checkpoint is discarded (kept on disk for forensics). This prevents stale
    checkpoints from a prior aborted run (which may have been bound to the
    wrong property's photos) from contaminating a fresh resume.
    """
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
            result = ImageResult(**raw)
            expected = image_paths[idx]
            actual = Path(result.image_path)
            if _ckpt_key(actual) != _ckpt_key(expected):
                logger.warning(
                    f"Discarding checkpoint {f.name}: image_path mismatch "
                    f"(expected {_ckpt_key(expected)}, got {_ckpt_key(actual)}). "
                    f"Likely contamination from a prior aborted run."
                )
                continue
            out[idx] = result
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
    model_routing_profile = request.get("modelRoutingProfile", analysis_profile)
    if model_routing_profile not in {"standard", "premium"}:
        raise ValueError(f"unsupported modelRoutingProfile: {model_routing_profile!r}")
    detection_backend = request.get("detectionBackend", "dinox")
    concurrency = request.get("concurrency", int(os.environ.get("ANALYZER_CONCURRENCY", "4")))

    # Listing facts from the CSV funnel DB (frontend passes the Property row).
    # Feeds cost_factors / estimate_units / estimate_sanity via
    # artifact_writers._resolve_property_metadata (job metadata wins over scrape).
    property_metadata = request.get("propertyMetadata")
    if not isinstance(property_metadata, dict):
        property_metadata = None

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

    # Same allowlist as the TS side (lib/analysis/modelOverrides.ts). Values are
    # concrete OpenAI model names (incl. the allocated Pass 2f model).
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
        filtered_overrides[k] = v.strip()

    raw_reasoning_efforts = request.get("reasoningEfforts") or {}
    filtered_reasoning_efforts = normalize_reasoning_efforts(raw_reasoning_efforts)

    # Per-run model names flow through run options (a supplied name routes that
    # pass to OpenAI). There is NO global pipeline_config mutation: the persistent
    # server processes many jobs back-to-back, so mutating module-global cfg per
    # job (and restoring it) was both unnecessary and unsafe.
    options = SceneClassifierRunOptions.from_analysis_profile(
        analysis_profile=model_routing_profile,
        model_overrides=filtered_overrides if filtered_overrides else None,
        reasoning_efforts=filtered_reasoning_efforts if filtered_reasoning_efforts else None,
    )

    if filtered_overrides:
        logger.info(f"[Job {ts_job_id}] modelOverrides (per-run): {filtered_overrides}")
    if filtered_reasoning_efforts:
        logger.info(
            f"[Job {ts_job_id}] reasoningEfforts (per-run): {filtered_reasoning_efforts}"
        )

    # Generate internal job ID and create artifacts directory
    internal_job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = artifacts_root / property_key / internal_job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Per-run checkpoint dir, parallel to internal_job_id job dirs. The runId is
    # stable across retries so re-attempts of the same AnalysisRun resume from
    # disk; a fresh "Re-analyze" generates a new runId and starts clean.
    ckpt_dir = artifacts_root / property_key / ".checkpoints" / run_id
    policy_fingerprint = _checkpoint_policy_fingerprint(
        filtered_overrides,
        filtered_reasoning_efforts,
        options.toggles.to_dict(),
        _resolved_max_tokens(),
    )
    _prepare_checkpoint_dir(ckpt_dir, policy_fingerprint)
    cached_results: Dict[int, ImageResult] = _load_checkpoint(
        ckpt_dir, len(image_paths), image_paths
    )
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

    # Wrap the job body so checkpoint cleanup only runs on full success. (Model
    # overrides no longer mutate module-global cfg, so there is nothing to restore
    # between jobs.)
    try:
        # ─────────────────────────────────────────────────────────────────────
        # Per-image analysis loop
        # ─────────────────────────────────────────────────────────────────────
        results: List[ImageResult] = []
        attempt_results: List[ImageResult] = []
        total_images = len(image_paths)
        total_start = time.perf_counter()
        job_started = total_start

        async def _analyze_all():
            sem = asyncio.Semaphore(concurrency)
            completed = 0

            async def _analyze_one(idx, image_path):
                nonlocal completed
                # Skip if a prior attempt already completed this image successfully.
                # Still emit progress so the frontend % advances during resume.
                if idx in cached_results and not cached_results[idx].error:
                    logger.info(f"  [cached] image {image_path.name} ({idx + 1}/{total_images})")
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
                    img_start = time.perf_counter()
                    logger.info(f"  [start] Analyzing: {image_path.name} ({idx + 1}/{total_images})")

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
                        elapsed = time.perf_counter() - img_start

                        img_result = ImageResult(
                            image_path=str(image_path),
                            scene_data=analysis.to_dict(),
                            scene=analysis.scene or "unknown",
                            processing_time=elapsed,
                        )
                        logger.info(f"    {image_path.name} -> {analysis.scene} ({elapsed:.1f}s)")

                    except Exception as exc:
                        elapsed = time.perf_counter() - img_start
                        logger.error(f"    ❌ {image_path.name} failed: {exc}", exc_info=True)
                        img_result = ImageResult(
                            image_path=str(image_path),
                            # PassExecutionError carries the partial result so the
                            # failed image's pass_states/pass_errors are retained.
                            scene_data=getattr(exc, "partial_result", None),
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

                    attempt_results.append(img_result)
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

        total_time = time.perf_counter() - total_start
        phase_timings = {"photo_analysis_sec": total_time, "pass_2f_sec": 0.0}
        postprocessing_started = job_started + total_time

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
            property_metadata=property_metadata,
        )

        from tools.artifact_writers import Pass2fModelUnavailable
        fatal_error = None
        failed_phase = None
        photo_intel_path = None

        # A failed image produced no valid analysis. Publishing the rest would
        # present a partial property as a complete one, so fail the job and write
        # nothing. The checkpoint is left intact below so a retry resumes from the
        # images that did succeed.
        failed_images = [r for r in results if getattr(r, "error", None)]
        if failed_images and options.failure_mode == "strict":
            fatal_error = RuntimeError(
                f"{len(failed_images)} of {len(results)} images failed analysis: "
                + "; ".join(f"{Path(r.image_path).name}: {r.error}" for r in failed_images[:5])
            )
            failed_phase = "photo_analysis"
            logger.error(f"[Job {ts_job_id}] {fatal_error}")

        try:
            if fatal_error is None:
                photo_intel_path = write_photo_intel(
                    cfg=cfg,
                    job=job,
                    detection_backend=detection_backend,
                    analysis_profile=analysis_profile,
                    use_pass_architecture=True,
                    # Resolved toggles, not {} — the artifact must record which
                    # passes actually ran.
                    pass_toggles=options.toggles.to_dict(),
                    # Concrete per-pass model names (incl. the allocated Pass 2f model)
                    # so run.model_overrides and model_routing reflect what actually ran.
                    model_overrides=filtered_overrides,
                    gpt_config=gpt5_config,
                    issue_catalog=catalog,
                    vlm_client=vlm_client,
                    reasoning_efforts=filtered_reasoning_efforts,
                    timing_recorder=phase_timings,
                    dependency_status={"embeddings": "ready"},
                )
        except Pass2fModelUnavailable as exc:
            # Pass 2f is OpenAI-only. Fail the job rather than silently degrade to
            # Qwen. Leave the checkpoint intact so a fixed-config retry can resume.
            logger.error(
                f"[Job {ts_job_id}] Pass 2f model unavailable — failing job "
                f"(no Qwen fallback): {exc}"
            )
            photo_intel_path = None
            fatal_error = exc
            failed_phase = "pass_2f"
        except Exception as exc:
            logger.error(f"Failed to write photo_intel: {exc}", exc_info=True)
            photo_intel_path = None
            fatal_error = exc
            failed_phase = "postprocessing"

        if not failed_phase:
            failed_phase = phase_timings.get("failed_phase")

        phase_timings["postprocessing_sec"] = time.perf_counter() - postprocessing_started
        phase_timings["end_to_end_sec"] = time.perf_counter() - job_started

        # Compute and log timing statistics
        timing_stats = _compute_timing_stats(
            results,
            phase_timings["end_to_end_sec"],
            usage_stats=getattr(vlm_client, "usage_stats", None),
            phase_timings=phase_timings,
            attempt_results=attempt_results,
            requested_photo_count=total_images,
            reused_photo_count=len(cached_results),
            configured_concurrency=concurrency,
            status="failed" if fatal_error else ("partial" if failed_phase else "complete"),
            failed_phase=failed_phase,
        )
        _log_timing_stats(timing_stats, property_key)

        if fatal_error is not None:
            _emit({
                "type": "result", "jobId": ts_job_id, "success": False,
                "error": str(fatal_error), "property_key": property_key,
                "timing_stats": timing_stats,
            })
            _emit({"type": "job_done", "jobId": ts_job_id})
            return

        # Build summary (same format as analyzer_cli)
        summary = _build_summary(
            job,
            photo_intel_path=photo_intel_path,
            detection_backend=detection_backend,
            analysis_profile=analysis_profile,
            used_pass_architecture=True,
            timing_stats=timing_stats,
        )

        logger.info(
            f"[Job {ts_job_id}] Complete: {property_key} "
            f"({phase_timings['end_to_end_sec']:.1f}s)"
        )

        # Emit result and job_done marker
        _emit({"type": "result", "jobId": ts_job_id, **summary})
        _emit({"type": "job_done", "jobId": ts_job_id})

        # Only clear checkpoints on full success — leaves them intact on exception
        # or SIGKILL so the next attempt with the same runId can resume. Must stay
        # outside the finally block.
        _clear_checkpoint(ckpt_dir)
    finally:
        # Per-run model overrides flow through run options now, so there is no
        # module-global cfg state to restore between jobs.
        pass


if __name__ == "__main__":
    sys.exit(main())
