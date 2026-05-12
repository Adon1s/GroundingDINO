#!/usr/bin/env python3
"""Audit runner — re-analyze a single scene-group's photos from existing artifacts.

Filters an already-classified `photo_intel.json` to the photos whose Pass 1a
scene group matches `--scene-group` (default: kitchen), runs the full pipeline
on just those photos, and writes a new run dir alongside the source. The new
artifact gets an `audit_meta` block at the top level so the frontend can list
audit runs separately from production runs.

Used for catalog and Pass 2f tuning — produces a per-property focused review
surface that the existing photos UI can render unchanged.

Example:
    python -m tools.audit_runner \
      --artifacts-root C:\\Users\\Steven\\IntelliJProjects\\realtorvision\\artifacts \
      --property redfin_125722422 \
      --scene-group kitchen \
      --concurrency 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import pipeline_config as cfg  # type: ignore
from tools.analyzer_cli import ImageResult, PropertyAnalysisJob, _apply_env_overrides
from tools.artifact_writers import load_issue_catalog, write_photo_intel
from tools.pass_config import SceneClassifierRunOptions
from tools.scene_classifier_orchestrator import create_orchestrator_from_config
from tools.vlm_client import create_vlm_client, get_model_configs_from_pipeline_config

logger = logging.getLogger("audit_runner")

AUDIT_RUNNER_VERSION = "1"
RUN_ID_RE = re.compile(r"^\d{8}_\d{6}_[a-f0-9]+$", re.IGNORECASE)


def _build_candidate_provider(catalog: Dict[str, Any]) -> Any:
    """Mirror analyzer_cli's Pass 2d embeddings retriever setup.

    Without this, Pass 2d cannot match observations to catalog items, so
    no candidates flow into the renovation estimate and Pass 2f is skipped
    with reason="no_package_candidates".
    """
    if not getattr(cfg, "USE_EMBEDDINGS_CATALOG", False):
        return None
    try:
        from dataclasses import asdict

        from tools.catalog_embeddings import (
            CatalogEmbeddingsRetriever,
            build_guardrails_from_catalog,
        )
        from tools.scene_classifier_passes import prioritize_resolution_candidates

        retriever = CatalogEmbeddingsRetriever(
            catalog_v2=catalog,
            model_name=getattr(
                cfg, "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
            ),
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
            cands = [asdict(m) for m in matches]
            return prioritize_resolution_candidates(cands, widened_routing=widened_routing)

        logger.info(
            "Pass 2d candidate_provider ready (model=%s, items=%d)",
            retriever.model_name,
            len(retriever._items),
        )
        return candidate_provider
    except Exception as exc:
        logger.warning(
            "Could not initialize embeddings retriever for Pass 2d: %s. Pass 2d will be skipped.",
            exc,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Source artifact resolution + filtering
# ─────────────────────────────────────────────────────────────────────────────
def _list_property_dirs(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir()] if root.is_dir() else []


def _resolve_source_artifact(property_dir: Path, source_run: str) -> Optional[Path]:
    """Return the source `photo_intel.json` path or None."""
    if source_run and source_run != "latest":
        path = property_dir / source_run / "photo_intel.json"
        return path if path.is_file() else None

    candidates: List[tuple[float, Path]] = []
    for run_dir in property_dir.iterdir():
        if not run_dir.is_dir() or not RUN_ID_RE.match(run_dir.name):
            continue
        photo_intel = run_dir / "photo_intel.json"
        if photo_intel.is_file():
            # Skip prior audit runs so we don't audit our own output.
            if _is_audit_artifact(photo_intel):
                continue
            candidates.append((run_dir.stat().st_mtime, photo_intel))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _is_audit_artifact(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    meta = data.get("audit_meta") if isinstance(data, dict) else None
    return isinstance(meta, dict) and "scene_group" in meta


def _filter_scene_group_photos(
    artifact: Dict[str, Any], scene_group: str
) -> List[Dict[str, str]]:
    photos = artifact.get("photos") or {}
    if not isinstance(photos, dict):
        return []

    matches: List[Dict[str, str]] = []
    for fallback_key, photo_data in photos.items():
        if not isinstance(photo_data, dict):
            continue
        scene = photo_data.get("scene") or {}
        if not isinstance(scene, dict) or scene.get("group") != scene_group:
            continue
        photo = photo_data.get("photo") or {}
        if not isinstance(photo, dict):
            continue
        image_path = photo.get("image_path")
        photo_key = photo.get("photo_key") or fallback_key
        if image_path and photo_key:
            matches.append({"photo_key": str(photo_key), "image_path": str(image_path)})
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Audit_meta patching (atomic via tmp)
# ─────────────────────────────────────────────────────────────────────────────
def _patch_audit_meta(artifact_path: Path, audit_meta: Dict[str, Any]) -> None:
    with artifact_path.open("r", encoding="utf-8") as f:
        artifact = json.load(f)
    artifact["audit_meta"] = audit_meta
    tmp = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    tmp.replace(artifact_path)


def _read_pass_2f_tally(artifact_path: Path) -> Dict[str, int]:
    try:
        with artifact_path.open("r", encoding="utf-8") as f:
            artifact = json.load(f)
    except Exception:
        return {}
    v4 = artifact.get("renovation_estimate_v4") or {}
    trace = v4.get("pass_2f_trace") or {}
    keys = (
        "candidate_count",
        "attempted_count",
        "confirmed_count",
        "rejected_count",
        "uncertain_count",
        "no_image_count",
    )
    return {k: int(trace.get(k) or 0) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Per-property audit run
# ─────────────────────────────────────────────────────────────────────────────
async def _run_for_property(
    *,
    property_dir: Path,
    artifacts_root: Path,
    scene_group: str,
    source_run: str,
    concurrency: int,
    dry_run: bool,
    catalog: Dict[str, Any],
    orchestrator: Any,
    vlm_client: Any,
    gpt5_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    property_key = property_dir.name
    source_path = _resolve_source_artifact(property_dir, source_run)
    if source_path is None:
        return {"property": property_key, "status": "skipped", "reason": "no source artifact"}

    try:
        with source_path.open("r", encoding="utf-8") as f:
            source = json.load(f)
    except Exception as exc:
        return {"property": property_key, "status": "error", "reason": f"read failed: {exc}"}

    matches = _filter_scene_group_photos(source, scene_group)
    if not matches:
        return {
            "property": property_key,
            "status": "skipped",
            "reason": f"no photos with scene.group={scene_group}",
        }

    if dry_run:
        return {
            "property": property_key,
            "status": "dry_run",
            "source_run_id": source_path.parent.name,
            "filtered_count": len(matches),
            "filtered_keys": [m["photo_key"] for m in matches],
        }

    job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = artifacts_root / property_key / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    analysis_profile = getattr(cfg, "ANALYSIS_PROFILE", "standard")
    detection_backend = getattr(cfg, "DETECTION_BACKEND", "dinox")
    options = SceneClassifierRunOptions.from_analysis_profile(
        analysis_profile=analysis_profile,
        toggles=None,
        model_overrides=None,
    )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(matches)
    results: List[ImageResult] = []
    total_start = time.time()

    async def _analyze_one(entry: Dict[str, str]) -> ImageResult:
        nonlocal completed
        async with sem:
            image_path = Path(entry["image_path"])
            t0 = time.time()
            try:
                img_options = options.with_meta(
                    run_id=job_id,
                    photo_key=entry["photo_key"],
                    property_key=property_key,
                )
                analysis = await orchestrator.analyze_image(
                    image_path=image_path,
                    options=img_options,
                )
                elapsed = time.time() - t0
                result = ImageResult(
                    image_path=str(image_path),
                    scene_data=analysis.to_dict(),
                    scene=analysis.scene or "unknown",
                    processing_time=elapsed,
                )
                logger.info(
                    "[%s] [ok] %s -> %s (%.1fs)",
                    property_key,
                    entry["photo_key"],
                    analysis.scene,
                    elapsed,
                )
            except Exception as exc:
                elapsed = time.time() - t0
                logger.error(
                    "[%s] [err] %s failed: %s",
                    property_key,
                    entry["photo_key"],
                    exc,
                )
                result = ImageResult(
                    image_path=str(image_path),
                    scene="unknown",
                    processing_time=elapsed,
                    error=str(exc),
                )
            completed += 1
            logger.info("[%s] [%d/%d] complete", property_key, completed, total)
            return result

    tasks = [_analyze_one(entry) for entry in matches]
    results.extend(await asyncio.gather(*tasks))
    total_time = time.time() - total_start

    job = PropertyAnalysisJob(
        property_key=property_key,
        job_id=job_id,
        artifacts_dir=str(job_dir),
        timestamp=datetime.utcnow().isoformat() + "Z",
        results=results,
        total_processing_time=total_time,
    )

    photo_intel_path: Optional[Path] = None
    write_error: Optional[str] = None
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
        write_error = str(exc)
        logger.exception("[%s] write_photo_intel failed", property_key)

    if photo_intel_path is None:
        return {
            "property": property_key,
            "status": "error",
            "reason": f"write_photo_intel failed: {write_error or 'unknown'}",
            "audit_run_id": job_id,
        }

    audit_meta = {
        "source_run_id": source_path.parent.name,
        "scene_group": scene_group,
        "audit_runner_version": AUDIT_RUNNER_VERSION,
        "filtered_photo_count": len(matches),
        "filtered_photo_keys": [m["photo_key"] for m in matches],
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    try:
        _patch_audit_meta(photo_intel_path, audit_meta)
    except Exception as exc:
        logger.exception("[%s] audit_meta patch failed", property_key)
        return {
            "property": property_key,
            "status": "error",
            "reason": f"audit_meta patch failed: {exc}",
            "audit_run_id": job_id,
        }

    return {
        "property": property_key,
        "status": "ok",
        "source_run_id": source_path.parent.name,
        "audit_run_id": job_id,
        "filtered_count": len(matches),
        "tally": _read_pass_2f_tally(photo_intel_path),
        "duration_s": round(total_time, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────
def _format_summary(result: Dict[str, Any]) -> str:
    status = result["status"]
    if status == "ok":
        t = result.get("tally", {}) or {}
        tally = "{c}/{r}/{u}/{n}".format(
            c=t.get("confirmed_count", 0),
            r=t.get("rejected_count", 0),
            u=t.get("uncertain_count", 0),
            n=t.get("no_image_count", 0),
        )
        return (
            f"[ok] {result['property']} src={result['source_run_id']} "
            f"-> {result['audit_run_id']} photos={result['filtered_count']} "
            f"pass2f(c/r/u/n)={tally} ({result['duration_s']:.1f}s)"
        )
    if status == "dry_run":
        return (
            f"[dry] {result['property']} src={result['source_run_id']} "
            f"photos={result['filtered_count']} keys={result['filtered_keys']}"
        )
    return f"[{status}] {result['property']}: {result.get('reason', '?')}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def _run_all(
    *,
    artifacts_root: Path,
    prop_dirs: List[Path],
    scene_group: str,
    source_run: str,
    concurrency: int,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)
    orchestrator: Any = None
    vlm_client: Any = None
    gpt5_config: Optional[Dict[str, Any]] = None

    if not dry_run:
        candidate_provider = _build_candidate_provider(catalog)
        orchestrator = create_orchestrator_from_config(
            cfg,
            candidate_provider=candidate_provider,
            catalog_items=catalog.get("items"),
        )
        _, gpt5_config = get_model_configs_from_pipeline_config(cfg)
        vlm_client = create_vlm_client()

    summaries: List[Dict[str, Any]] = []
    for prop_dir in prop_dirs:
        result = await _run_for_property(
            property_dir=prop_dir,
            artifacts_root=artifacts_root,
            scene_group=scene_group,
            source_run=source_run,
            concurrency=concurrency,
            dry_run=dry_run,
            catalog=catalog,
            orchestrator=orchestrator,
            vlm_client=vlm_client,
            gpt5_config=gpt5_config,
        )
        summaries.append(result)
        print(_format_summary(result))

    return summaries


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", required=True, help="Path to /artifacts dir")
    parser.add_argument("--scene-group", default="kitchen", help="Scene group filter (e.g. kitchen)")
    parser.add_argument("--property", default=None, help="Single property key (default: all)")
    parser.add_argument("--source-run", default="latest", help="'latest' or a specific run id")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of properties")
    parser.add_argument("--concurrency", type=int, default=4, help="Per-image semaphore")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no VLM calls")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _apply_env_overrides()

    artifacts_root = Path(args.artifacts_root).resolve()
    if not artifacts_root.is_dir():
        logger.error("artifacts root does not exist: %s", artifacts_root)
        return 2

    if args.property:
        prop_dir = artifacts_root / args.property
        if not prop_dir.is_dir():
            logger.error("property dir not found: %s", prop_dir)
            return 2
        prop_dirs = [prop_dir]
    else:
        prop_dirs = _list_property_dirs(artifacts_root)
    if args.limit:
        prop_dirs = prop_dirs[: args.limit]

    if not prop_dirs:
        logger.error("no properties to process")
        return 2

    logger.info(
        "audit_runner: %d propert%s, scene_group=%s, source=%s, dry_run=%s",
        len(prop_dirs),
        "y" if len(prop_dirs) == 1 else "ies",
        args.scene_group,
        args.source_run,
        args.dry_run,
    )

    summaries = asyncio.run(
        _run_all(
            artifacts_root=artifacts_root,
            prop_dirs=prop_dirs,
            scene_group=args.scene_group,
            source_run=args.source_run,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
        )
    )

    ok = sum(1 for r in summaries if r["status"] == "ok")
    skipped = sum(1 for r in summaries if r["status"] == "skipped")
    errors = sum(1 for r in summaries if r["status"] == "error")
    dry = sum(1 for r in summaries if r["status"] == "dry_run")
    print(f"\nDone: {ok} ok, {skipped} skipped, {errors} errors, {dry} dry_run (total: {len(summaries)})")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
