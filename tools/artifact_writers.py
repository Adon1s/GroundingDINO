# tools/artifact_writers.py
"""
Artifact writers for the analysis pipeline.

Contains:
  - load_issue_catalog() / log_catalog_load()
  - write_photo_intel()   (was AutoAnalyzer.save_photo_intel)
  - write_property_summary() (was AutoAnalyzer.generate_property_summary)
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.pipeline_common import (
    SCENE_GROUPS_UI,
    SCENE_TO_GROUP_UI,
    PHOTO_INTEL_SCHEMA_VERSION,
    PROPERTY_SUMMARY_SCHEMA_VERSION,
    NORMALIZATION_POLICY_VERSION,
    make_photo_id,
    make_issue_id,
    safe_list,
)

from tools.scene_classifier_service import scene_classifier_payload

logger = logging.getLogger(__name__)


# Import defect events layer
try:
    from tools.defect_events import build_defect_events, generate_work_items, build_search_index
    DEFECT_EVENTS_AVAILABLE = True
except ImportError:
    DEFECT_EVENTS_AVAILABLE = False


def load_issue_catalog(path: Path) -> dict:
    """
    Load the issue catalog from JSON.

    Returns a dict with canonical keys:
      - 'items': unified list of catalog entries (each has 'id' and 'kind' field)
      - 'trade_buckets': list of trade bucket definitions
      - 'version': catalog version string
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Catalog file loaded: {path.resolve()} raw_keys={list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
    except FileNotFoundError:
        logger.warning(f"Issue catalog not found at {path}; using empty catalog.")
        data = {}
    except Exception as exc:
        logger.error(f"Failed to load issue catalog from {path}: {exc}")
        data = {}

    # Normalize trade_buckets: v3 expects a list, tolerate older dict shapes
    tb = data.get("trade_buckets", [])
    if isinstance(tb, dict):
        tb = list(tb.values())

    # Normalize to canonical "items" array.
    # v3+ catalogs use a single "items" list where each entry has an 'id' and 'kind' field.
    # Older catalogs split into "defects" / "upgrades" (or "defect_issues") -- merge both
    # so upgrades are never silently dropped.
    if "items" in data and data["items"]:
        items = data["items"]
    else:
        # Legacy split format: normalize each side and merge
        raw_defects  = data.get("defects",       []) or data.get("defect_issues", []) or []
        raw_upgrades = data.get("upgrades",       []) or data.get("upgrade_items", []) or []
        # Stamp kind if missing so downstream code has a reliable field
        for d in raw_defects:
            if isinstance(d, dict):
                d.setdefault("kind", "defect")
        for u in raw_upgrades:
            if isinstance(u, dict):
                u.setdefault("kind", "upgrade")
        items = raw_defects + raw_upgrades
        if items:
            logger.info(
                f"load_issue_catalog: merged legacy format -- "
                f"{len(raw_defects)} defects + {len(raw_upgrades)} upgrades -> {len(items)} items"
            )

    return {
        # v3 canonical: single "items" array (each has 'id' and 'kind' field)
        "items": items,
        "trade_buckets": tb,
        "version": data.get("version"),
    }


def log_catalog_load(path: Path, cat: dict):
    """Log catalog load details at INFO so they're always visible."""
    items = cat.get("items", []) or []
    version = cat.get("version")
    kind_counts = {}
    for d in items:
        k = d.get("kind", "defect") if isinstance(d, dict) else "?"
        kind_counts[k] = kind_counts.get(k, 0) + 1
    logger.info(
        f"Catalog load: path={path.resolve()} exists={path.exists()} "
        f"version={version} items={len(items)} "
        f"kinds={kind_counts} "
        f"keys={list(cat.keys())}"
    )
    sample = items[:2]
    if sample:
        logger.info(f"Catalog sample items[0:2]={[{k: v for k, v in s.items() if k in ('id','name','kind','trade_bucket')} for s in sample if isinstance(s, dict)]}")
    else:
        logger.info("Catalog sample: items is EMPTY")


def write_photo_intel(
    *,
    cfg: Any,
    job: Any,  # PropertyAnalysisJob
    detection_backend: str,
    analysis_profile: str,
    use_pass_architecture: bool,
    pass_toggles: Dict[str, bool],
    model_overrides: Dict[str, str],
    gpt_config: Optional[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
    output_path: Optional[Path] = None,
    generate_summary: bool = True,
    # Callables for summary generation (passed by AutoAnalyzer)
    summary_writer: Any = None,  # callable: write_property_summary or similar
) -> Path:
    """Persist per-photo intelligence (including scene classifier fields)."""
    created_at = datetime.utcnow().isoformat() + "Z"

    photos: Dict[str, Any] = {}
    issues_flat: List[Dict[str, Any]] = []  # Flat index of all issues (property-level)

    for photo_index, res in enumerate(job.results, start=1):
        image_key = Path(res.image_path).name
        payload = scene_classifier_payload(res.scene_classifier or res.scene_data)

        # -- Resolve passes dict -----------------------------------------------
        # Use passes from payload (orchestrator). Fall back to flat fields for legacy runs.
        passes = payload.get("passes", None)
        if passes is None:
            feat_notes = payload.get("feature_notes", "") or payload.get("positives_notes", "")
            passes = {
                "1a": {"scene": payload.get("scene", "unknown"), "confidence": None, "reasoning": payload.get("reasoning", "")},
                "1b": {"feature_notes": feat_notes, "positives_notes": feat_notes},
                "1c": {"overall_impression": payload.get("overall_impression", ""), "image_summary": payload.get("image_summary", ""), "notable_features": payload.get("notable_features", []) or []},
                "2a": {"observations_freeform": payload.get("observations_freeform", "")},
                "2b": {"issues_natural_language": payload.get("issues_natural_language", []) or [], "catalog_flags": {}},
                "3":  {"keywords": payload.get("keywords", []) or [], "categories": payload.get("keyword_categories")},
            }

        # -- Core identifiers ---------------------------------------------------
        photo_id     = make_photo_id(job.property_key, job.job_id, image_key)
        scene        = payload.get("scene", res.scene) or "unknown"
        scene_group  = SCENE_TO_GROUP_UI.get(scene, "other")
        scene_conf   = (passes.get("1a", {}) or {}).get("confidence")
        pass_1c      = passes.get("1c", {}) or {}
        pass_1b      = passes.get("1b", {}) or {}

        # -- Resolve final issues -----------------------------------------------
        final_issues = safe_list(payload.get("verified_issues"))
        if not final_issues:
            final_issues = safe_list((passes.get("2c", {}) or {}).get("verified_issues"))
        if not final_issues:
            final_issues = safe_list((passes.get("2b", {}) or {}).get("issues_natural_language"))

        # Matched issues (post-sanity+dedupe, superset of final; for debug/analytics)
        matched_issues = safe_list(payload.get("matched_issues"))
        # Removed issues (explicitly discarded with reasons)
        removed_issues = safe_list((passes.get("2e", {}) or {}).get("removed"))

        # -- Backfill stable issue_id + source linkage on every final issue -----
        issue_sig_counts: Dict[Tuple[str, str, str], int] = {}
        for issue in final_issues:
            if not (isinstance(issue, dict) and issue.get("description")):
                continue
            sig = (issue.get("description", ""), issue.get("location_hint", ""), issue.get("label", ""))
            ordinal = issue_sig_counts.get(sig, 0)
            issue_sig_counts[sig] = ordinal + 1
            if not issue.get("issue_id"):
                issue["issue_id"] = make_issue_id(job.job_id, image_key, sig[0], sig[1], sig[2], ordinal)
            issue.setdefault("source_photo_key", image_key)
            issue.setdefault("source_photo_id",  photo_id)
            issue.setdefault("scene",             scene)
            issue.setdefault("scene_group",       scene_group)

            issues_flat.append({
                "issue_id":    issue["issue_id"],
                "photo_id":    photo_id,
                "photo_key":   image_key,
                "scene":       scene,
                "scene_group": scene_group,
                "description": issue.get("description", ""),
                "label":       issue.get("label", ""),
                "location_hint": issue.get("location_hint", ""),
                "catalog_item_id":  issue.get("catalogItemId") or issue.get("resolved_item_id"),
                "catalog_item_kind": issue.get("catalogItemKind") or issue.get("kind"),
            })

        # -- Build trace (pass internals -- debug/auditing, not read by UI) -----
        pass_2c   = passes.get("2c", {}) or {}
        pass_2d   = passes.get("2d", {}) or {}
        pass_2e_p = passes.get("2e", {}) or {}

        _2c_forward = safe_list(pass_2c.get("labeled_forward") or payload.get("labeled_forward"))
        _2d_resolutions = safe_list(pass_2d.get("resolutions") or payload.get("resolved_items"))

        trace = {
            "passes_run":  payload.get("passes_run", []),
            "models":      payload.get("models_used", {}),
            "timings_sec": payload.get("pass_timings", {}),
            "2c": {
                "forwarded_count": len(_2c_forward),
            },
            "2d": {
                "resolution_count": len(_2d_resolutions),
            },
            "2e": {
                "input_count":   pass_2e_p.get("input_count"),
                "kept_count":    pass_2e_p.get("kept_count"),
                "removed_count": pass_2e_p.get("removed_count"),
            },
        }

        # -- Assemble v3 per-photo entry ----------------------------------------
        photos[image_key] = {
            "photo": {
                "photo_key":   image_key,
                "photo_id":    photo_id,
                "image_path":  res.image_path,
                "index":       photo_index,
            },
            "scene": {
                "id":         scene,
                "group":      scene_group,
                "confidence": scene_conf,
                "reasoning":  (passes.get("1a", {}) or {}).get("reasoning", ""),
            },
            "features": {
                "overall_impression": pass_1c.get("overall_impression", "") or payload.get("overall_impression", ""),
                "image_summary":      pass_1c.get("image_summary",      "") or payload.get("image_summary",      ""),
                "notable_features":   safe_list(pass_1c.get("notable_features") or payload.get("notable_features")),
                "feature_notes":      pass_1b.get("feature_notes",  "") or payload.get("feature_notes",  ""),
                "positives_notes":    pass_1b.get("positives_notes", "") or payload.get("positives_notes", ""),
                "observations_freeform": (passes.get("2a", {}) or {}).get("observations_freeform", "") or payload.get("observations_freeform", ""),
            },
            "issues": {
                "final": final_issues,
                "matched": matched_issues,
                "removed": removed_issues,
            },
            "keywords": {
                "all":        safe_list((passes.get("3", {}) or {}).get("keywords") or payload.get("keywords")),
                "categories": (passes.get("3", {}) or {}).get("categories") or payload.get("keyword_categories"),
                "gdino_prompt": payload.get("groundingdino_prompt", ""),
            },
            "planner_hints": payload.get("planner_hints", {}),
            "is_staged": payload.get("is_staged"),
            "processing_time": res.processing_time,
            "error": res.error,
            "trace": trace,
            "debug": {
                "labeled_debug":   safe_list(payload.get("labeled_debug")),
                "labeled_forward": safe_list(payload.get("labeled_forward")),
                "resolved_items":  safe_list(payload.get("resolved_items")),
                "observations_struct": payload.get("observations_struct", {}),
                "features_struct":    payload.get("features_struct", {}),
                "passes": passes,
            },
            "_compat": {
                "passes":        passes,
                "pass_timings":  payload.get("pass_timings", {}),
                "total_pass_time": payload.get("total_pass_time", 0.0),
                "scene_group":   scene_group,
                "photo_id":      photo_id,
            },
            "_pass_2e_telemetry": {
                "input_count":   pass_2e_p.get("input_count"),
                "deduped_count": pass_2e_p.get("deduped_count"),
                "final_count":   pass_2e_p.get("final_count"),
                "removed_count": pass_2e_p.get("removed_count"),
                "removed_reason_counts": pass_2e_p.get("removed_reason_counts", {}),
                "suppressed_reason_counts": pass_2e_p.get("suppressed_reason_counts", {}),
                "suppressed_samples": pass_2e_p.get("suppressed_samples", []),
            },
        }

    # -- Build room_groups from v3 photo entries --------------------------------
    room_groups: Dict[str, Any] = {}
    for img_key, p in photos.items():
        scene       = p["scene"]["id"]
        group       = p["scene"]["group"]
        feat        = p["features"]
        final_list  = p["issues"]["final"]

        g = room_groups.setdefault(group, {
            "scenes_included": SCENE_GROUPS_UI.get(group, []),
            "image_keys": [],
            "image_count": 0,
            "positives": {"notes": [], "notable_features": []},
            "issues": {"notes": [], "issues_natural_language": []},
        })
        g["image_keys"].append(img_key)
        g["image_count"] += 1

        if feat.get("feature_notes"):
            g["positives"]["notes"].append(feat["feature_notes"])
        for f in safe_list(feat.get("notable_features")):
            s = str(f).strip()
            if s and s not in g["positives"]["notable_features"]:
                g["positives"]["notable_features"].append(s)

        if feat.get("observations_freeform"):
            g["issues"]["notes"].append(feat["observations_freeform"])

        for it in final_list:
            if isinstance(it, dict) and it.get("description"):
                g["issues"]["issues_natural_language"].append({
                    "source_image": img_key,
                    "issue_id":     it.get("issue_id"),
                    "photo_id":     p["photo"]["photo_id"],
                    "photo_key":    img_key,
                    "description":  it.get("description", ""),
                    "label":        it.get("label", ""),
                    "location_hint": it.get("location_hint", ""),
                })

    photo_intel = {
        "schema_version":               PHOTO_INTEL_SCHEMA_VERSION,
        "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
        "run": {
            "run_id":              job.job_id,
            "job_id":              job.job_id,
            "created_at":          created_at,
            "timestamp":           job.timestamp,
            "detection_backend":   detection_backend,
            "analysis_profile":    analysis_profile,
            "used_pass_architecture": use_pass_architecture,
            "pass_toggles":        pass_toggles if pass_toggles else None,
            "model_overrides":     model_overrides if model_overrides else None,
            "model":               getattr(cfg, "LM_STUDIO_MODEL", ""),
            "gpt_model":           gpt_config.get('model') if gpt_config else None,
        },
        "property": {
            "property_key": job.property_key,
            "artifacts_dir": job.artifacts_dir,
        },
        "photos":      photos,
        "room_groups": room_groups,
        "issues_flat": issues_flat,
        "issues_flat_count": len(issues_flat),
    }

    # renovation_needs disabled: severity not reliable at this stage
    photo_intel["renovation_needs"] = None

    # -- Compute estimates (scoring + costing) ---------------------------------
    try:
        from tools.costing import compute_estimates
        estimates = compute_estimates(
            issues_flat=issues_flat,
            issue_catalog=issue_catalog,
            n_photos=len(photos),
            include_optional=False,
        )
        photo_intel["estimates"] = estimates
        logger.info(
            "Estimates: rehab_score=%d cost=$%s-$%s (%d items scored, %d unresolved)",
            estimates["scoring"]["rehab_score"],
            f'{estimates["costs"]["total_low"]:,}',
            f'{estimates["costs"]["total_high"]:,}',
            estimates["meta"]["issues_scored"],
            estimates["meta"]["unresolved_issues"],
        )
    except Exception as exc:
        logger.error(f"Failed to compute estimates: {exc}", exc_info=True)
        photo_intel["estimates"] = None

    # -- Build defect events, work items, and search index ----------------------
    if DEFECT_EVENTS_AVAILABLE:
        try:
            defect_events = build_defect_events(
                photos=photos,
                catalog=issue_catalog,
                run_id=job.job_id,
            )
            work_items = generate_work_items(defect_events, issue_catalog)
            search_index = build_search_index(defect_events, issue_catalog)

            photo_intel["defect_events"] = defect_events
            photo_intel["work_items"] = work_items
            photo_intel["search_index"] = search_index
        except Exception as exc:
            logger.error(f"Failed to build defect events layer: {exc}", exc_info=True)
            photo_intel["defect_events"] = []
            photo_intel["work_items"] = []
            photo_intel["search_index"] = {}
    else:
        photo_intel["defect_events"] = []
        photo_intel["work_items"] = []
        photo_intel["search_index"] = {}

    output_path = output_path or Path(job.artifacts_dir) / "photo_intel.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Write full debug file first (all pass outputs, intermediates, timings) --
    debug_path = output_path.parent / "photo_intel_debug.json"
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(photo_intel, f, indent=2, ensure_ascii=False)
    logger.info(f"Photo intel debug saved to: {debug_path}")

    # -- Build slim version for frontend consumption ----------------------------
    slim = copy.deepcopy(photo_intel)

    # Strip run-level config fields the frontend doesn't read
    _run = slim.get("run")
    if isinstance(_run, dict):
        for _k in ("pass_toggles", "model_overrides", "used_pass_architecture",
                    "model", "gpt_model"):
            _run.pop(_k, None)

    # Strip per-photo debug fields (v3 schema)
    for _img_key, _photo in (slim.get("photos") or {}).items():
        _photo.pop("debug", None)
        _photo.pop("trace", None)
        _photo.pop("_pass_2e_telemetry", None)
        # Strip matched/removed from issues (UI only reads final)
        _issues = _photo.get("issues")
        if isinstance(_issues, dict):
            _issues.pop("matched", None)
            _issues.pop("removed", None)
        _feat = _photo.get("features")
        if isinstance(_feat, dict):
            _feat.pop("observations_freeform", None)
            _feat.pop("feature_notes", None)
            _feat.pop("positives_notes", None)
        _compat = _photo.get("_compat")
        if isinstance(_compat, dict):
            _FRONTEND_PASS_KEYS = {"1a", "2b", "2e"}
            _cp = _compat.get("passes")
            if isinstance(_cp, dict):
                _compat["passes"] = {
                    k: v for k, v in _cp.items()
                    if k in _FRONTEND_PASS_KEYS
                }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    logger.info(f"Photo intel (slim) saved to: {output_path}")

    if generate_summary and summary_writer is not None:
        summary_path = output_path.parent / "property_summary.json"
        summary_writer(job=job, photo_intel_path=output_path, output_path=summary_path)

    return output_path


def write_property_summary(
    *,
    cfg: Any,
    job: Any,  # PropertyAnalysisJob
    analysis_profile: str,
    vlm_client: Any,
    get_model_config_for_pass,  # callable from AutoAnalyzer
    photo_intel_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Generate a property-level summary from the analysis results using Pass 4, 4a, and 4b.

    Always writes property_summary.json and embeds property_pass4 into photo_intel.json,
    even on failure (with error field and empty values) so the UI has consistent structure.
    """
    if not getattr(cfg, "GENERATE_PROPERTY_SUMMARY", True):
        logger.info("Property summary generation disabled in config")
        return None

    logger.info(f"\n{'=' * 60}")
    logger.info("Generating Property Summary...")
    logger.info(f"{'=' * 60}")

    if output_path is None:
        output_path = Path(job.artifacts_dir) / "property_summary.json"

    # Initialize with defaults
    error_msg = None
    model_used = ""
    scene_counts: Dict[str, int] = {}

    # Initialize Pass 4a outputs with defaults
    room_summaries: Dict[str, Any] = {}
    issues_by_category: Dict[str, int] = {}
    total_issues_found = 0

    # Pass 4b fields (renovation intel)
    room_scopes: Dict[str, str] = {}
    room_work_items: Dict[str, List[str]] = {}
    top_work_items: List[str] = []

    # Pass 4c UI card fields
    overall_condition = ""
    overall_summary = ""
    investment_verdict = ""
    investment_rationale = ""
    renovation_scope = ""
    renovation_priorities: List[str] = []
    risk_flags: List[str] = []
    deferred_maintenance: List[str] = []

    errors: Dict[str, str] = {}

    # Check availability of VLM + pass architecture
    try:
        from tools.vlm_client import VLMClient
        VLM_CLIENT_AVAILABLE = True
    except ImportError:
        VLM_CLIENT_AVAILABLE = False

    try:
        from tools.scene_classifier_passes import (
            run_pass_4a_room_summaries,
            run_pass_4b_renovation_intel,
            run_pass_4c_property_card_fields,
            build_grouped_issues,
            derive_property_scope,
        )
        PASS_ARCHITECTURE_AVAILABLE = True
    except ImportError:
        PASS_ARCHITECTURE_AVAILABLE = False

    logger.info(
        f"Pass4 gate: VLM_CLIENT_AVAILABLE={VLM_CLIENT_AVAILABLE} "
        f"vlm_client={bool(vlm_client)} "
        f"PASS_ARCHITECTURE_AVAILABLE={PASS_ARCHITECTURE_AVAILABLE} "
        f"artifact_writers_file={__file__}"
    )

    if VLM_CLIENT_AVAILABLE and vlm_client and PASS_ARCHITECTURE_AVAILABLE:
        logger.info("  Aggregating from job.results for Pass 4a/4b/4c")

        # Build all_results from job (uses already-computed passes, no re-analysis)
        all_results = {}
        for res in job.results:
            image_key = Path(res.image_path).name
            payload = scene_classifier_payload(res.scene_classifier or res.scene_data)
            all_results[image_key] = payload
            # Count scenes
            scene = payload.get("scene", "unknown")
            scene_counts[scene] = scene_counts.get(scene, 0) + 1

        # Compute grouped_issues ONCE from all_results
        grouped_issues, _fallback_count = build_grouped_issues(all_results)

        # Compute deterministic totals from verified issues
        total_images_analyzed = len(all_results)
        total_issues_found = sum(len(issues) for issues in grouped_issues.values())
        issues_by_category = {}
        for group_issues in grouped_issues.values():
            for issue in group_issues:
                cat = issue.get("label", "general") or "general"
                issues_by_category[cat] = issues_by_category.get(cat, 0) + 1

        # --- Pass 4a (room summaries aggregation) ---
        try:
            logger.info("  Using Pass 4a (run_pass_4a_room_summaries)")
            model_config_4a = get_model_config_for_pass('4a')
            model_used = model_config_4a.get('model', '')
            logger.info(f"  Model: {model_used}")

            async def run_pass4a():
                return await run_pass_4a_room_summaries(
                    vlm_client=vlm_client,
                    model_config=model_config_4a,
                    grouped_issues=grouped_issues,
                    scene_counts=scene_counts,
                    total_images_analyzed=total_images_analyzed,
                    total_issues_found=total_issues_found,
                    issues_by_category=issues_by_category,
                )

            loop = asyncio.new_event_loop()
            try:
                pass_4a_result = loop.run_until_complete(run_pass4a())
            finally:
                loop.close()

            room_summaries = pass_4a_result.room_summaries or {}
            logger.info(f"  Pass 4a completed: {len(room_summaries)} room groups")

        except Exception as e:
            errors["pass4a"] = f"Pass 4a failed: {e}"
            logger.error(errors["pass4a"], exc_info=True)

        # --- Pass 4b (renovation intel: scopes + work items) ---
        try:
            logger.info("  Using Pass 4b (run_pass_4b_renovation_intel)")
            model_config_4b = get_model_config_for_pass('4b')
            logger.info(f"  Model: {model_config_4b.get('model', '')}")

            async def run_pass4b():
                return await run_pass_4b_renovation_intel(
                    vlm_client=vlm_client,
                    model_config=model_config_4b,
                    grouped_issues=grouped_issues,
                )

            loop = asyncio.new_event_loop()
            try:
                pass_4b_result = loop.run_until_complete(run_pass4b())
            finally:
                loop.close()

            room_scopes = pass_4b_result.room_scopes or {}
            room_work_items = pass_4b_result.room_work_items or {}
            top_work_items = pass_4b_result.top_work_items or []

            # Derive property scope deterministically from room scopes
            renovation_scope = derive_property_scope(room_scopes)

            logger.info(f"  Pass 4b completed: scope={renovation_scope}, top_items={len(top_work_items)}")

        except Exception as e:
            errors["pass4b"] = f"Pass 4b failed: {e}"
            logger.error(errors["pass4b"], exc_info=True)

        # --- Pass 4c (property card fields for UI) ---
        try:
            logger.info("  Using Pass 4c (run_pass_4c_property_card_fields)")
            model_config_4c = get_model_config_for_pass('4c')
            logger.info(f"  Model: {model_config_4c.get('model', '')}")

            async def run_pass4c():
                return await run_pass_4c_property_card_fields(
                    vlm_client=vlm_client,
                    model_config=model_config_4c,
                    room_summaries=room_summaries,
                    room_scopes=room_scopes,
                    room_work_items=room_work_items,
                    top_work_items=top_work_items,
                    total_issues_found=total_issues_found,
                    total_images_analyzed=total_images_analyzed,
                    issues_by_category=issues_by_category,
                    property_scope=renovation_scope,
                )

            loop = asyncio.new_event_loop()
            try:
                pass_4c_result = loop.run_until_complete(run_pass4c())
            finally:
                loop.close()

            overall_condition = pass_4c_result.overall_condition or ""
            overall_summary = pass_4c_result.overall_summary or ""
            investment_verdict = pass_4c_result.investment_verdict or ""
            investment_rationale = pass_4c_result.investment_rationale or ""
            renovation_priorities = pass_4c_result.renovation_priorities or []
            risk_flags = pass_4c_result.risk_flags or []
            deferred_maintenance = pass_4c_result.deferred_maintenance or []

            logger.info("  Pass 4c completed successfully")

        except Exception as e:
            errors["pass4c"] = f"Pass 4c failed: {e}"
            logger.error(errors["pass4c"], exc_info=True)

    else:
        # VLM client or pass architecture not available
        missing = []
        if not VLM_CLIENT_AVAILABLE:
            missing.append("VLM client")
        if not vlm_client:
            missing.append("vlm_client instance")
        if not PASS_ARCHITECTURE_AVAILABLE:
            missing.append("pass architecture")
        error_msg = f"Passes unavailable: missing {', '.join(missing)}"
        logger.warning(error_msg)

    # Combine errors
    if errors:
        error_msg = "; ".join(errors.values())

    # Build summary dict - always write even on failure
    summary_data = {
        "artifact_schema_version": PROPERTY_SUMMARY_SCHEMA_VERSION,
        "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
        "property_key": job.property_key,
        "run_id": job.job_id,
        "job_id": job.job_id,
        "timestamp": job.timestamp,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "summary_version": "pass4_v3",
        "analysis_profile": analysis_profile,
        "scene_counts": scene_counts,

        # Pass 4a fields
        "room_summaries": room_summaries,
        "issues_by_category": issues_by_category,
        "total_issues_found": total_issues_found,

        # Pass 4b fields
        "room_scopes": room_scopes,
        "room_work_items": room_work_items,
        "top_work_items": top_work_items,
        "renovation_scope": renovation_scope,

        # Pass 4c fields
        "overall_condition": overall_condition,
        "overall_summary": overall_summary,
        "investment_verdict": investment_verdict,
        "investment_rationale": investment_rationale,
        "renovation_priorities": renovation_priorities,
        "risk_flags": risk_flags,
        "deferred_maintenance": deferred_maintenance,

        # Metadata
        "total_images_analyzed": len(job.results),
        "model_used": model_used,
        "error": error_msg,
        "errors": errors if errors else None,
    }

    # renovation_needs disabled: severity not reliable at this stage
    summary_data["renovation_needs"] = None

    # Add defect events layer from photo_intel (already computed in write_photo_intel)
    if DEFECT_EVENTS_AVAILABLE and photo_intel_path and photo_intel_path.exists():
        try:
            with open(photo_intel_path, 'r', encoding='utf-8') as f:
                pi = json.load(f)
            summary_data["defect_events"] = pi.get("defect_events", [])
            summary_data["work_items"] = pi.get("work_items", [])
            summary_data["search_index"] = pi.get("search_index", {})
            summary_data["estimates"] = pi.get("estimates")
        except Exception as exc:
            logger.warning(f"Could not load defect events from photo_intel: {exc}")
            summary_data["defect_events"] = []
            summary_data["work_items"] = []
            summary_data["search_index"] = {}
            summary_data["estimates"] = None
    else:
        summary_data["defect_events"] = []
        summary_data["work_items"] = []
        summary_data["search_index"] = {}
        summary_data["estimates"] = None

    # Write property_summary.json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Property summary saved to: {output_path}")

    # Embed Pass 4a/4b/4c output into photo_intel.json so UI has single payload
    if photo_intel_path and photo_intel_path.exists():
        try:
            with open(photo_intel_path, 'r', encoding='utf-8') as f:
                photo_intel = json.load(f)

            photo_intel["property_pass4a"] = {
                "room_summaries": room_summaries,
                "issues_by_category": issues_by_category,
                "total_issues_found": total_issues_found,
                "scene_counts": scene_counts,
                "total_images_analyzed": len(job.results),
                "error": errors.get("pass4a"),
            }

            photo_intel["property_pass4b"] = {
                "room_scopes": room_scopes,
                "room_work_items": room_work_items,
                "top_work_items": top_work_items,
                "renovation_scope": renovation_scope,
                "error": errors.get("pass4b"),
            }

            photo_intel["property_pass4c"] = {
                "overall_condition": overall_condition,
                "overall_summary": overall_summary,
                "investment_verdict": investment_verdict,
                "investment_rationale": investment_rationale,
                "renovation_priorities": renovation_priorities,
                "risk_flags": risk_flags,
                "deferred_maintenance": deferred_maintenance,
                "error": errors.get("pass4c"),
            }

            # Also embed full summary for convenience
            photo_intel["property_summary"] = summary_data

            with open(photo_intel_path, "w", encoding="utf-8") as f:
                json.dump(photo_intel, f, indent=2, ensure_ascii=False)

            logger.info(f"Pass 4a/4b/4c outputs embedded into: {photo_intel_path}")

            # Also embed into the debug file so it has the complete picture
            debug_path = photo_intel_path.parent / "photo_intel_debug.json"
            if debug_path.exists():
                try:
                    with open(debug_path, 'r', encoding='utf-8') as f:
                        debug_intel = json.load(f)
                    debug_intel["property_pass4a"] = photo_intel["property_pass4a"]
                    debug_intel["property_pass4b"] = photo_intel["property_pass4b"]
                    debug_intel["property_pass4c"] = photo_intel["property_pass4c"]
                    debug_intel["property_summary"] = summary_data
                    with open(debug_path, "w", encoding="utf-8") as f:
                        json.dump(debug_intel, f, indent=2, ensure_ascii=False)
                    logger.info(f"Pass 4a/4b/4c outputs also embedded into: {debug_path}")
                except Exception as exc:
                    logger.warning(f"Could not embed property_pass4 into debug file: {exc}")
        except Exception as exc:
            logger.warning(f"Could not embed property_pass4 into photo_intel.json: {exc}")

    return output_path
