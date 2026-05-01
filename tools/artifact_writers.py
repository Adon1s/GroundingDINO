# tools/artifact_writers.py
"""
Artifact writers for the analysis pipeline.

Contains:
  - load_issue_catalog() / log_catalog_load()
  - write_photo_intel()
"""
from __future__ import annotations

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
    NORMALIZATION_POLICY_VERSION,
    make_photo_id,
    make_issue_id,
    safe_list,
)

from tools.scene_classifier_service import scene_classifier_payload

logger = logging.getLogger(__name__)


def _strip_pass_2f_audit_rationale(renovation_estimate: Any) -> None:
    """Remove debug-only Pass 2f audit fields from frontend-facing payloads."""
    if not isinstance(renovation_estimate, dict):
        return
    audit = renovation_estimate.get("pass_2f_review_audit")
    if not isinstance(audit, dict):
        return
    audit.pop("consistency_flag_counts", None)
    items = audit.get("items")
    if not isinstance(items, list):
        return
    for item in items:
        if isinstance(item, dict):
            item.pop("rationale", None)
            item.pop("review_rationale", None)
            item.pop("consistency_flags", None)


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
    vlm_client: Any = None,
    local_vlm_config: Optional[Dict[str, Any]] = None,
    pass_2f_provider: str = "premium",
) -> Path:
    """Persist per-photo intelligence (including scene classifier fields)."""
    created_at = datetime.utcnow().isoformat() + "Z"

    photos: Dict[str, Any] = {}
    issues_flat: List[Dict[str, Any]] = []  # Flat index of all issues (property-level)
    estimate_issues_flat: List[Dict[str, Any]] = []  # Canonical issues for renovation estimate units
    unmapped_issues: List[Dict[str, Any]] = []  # Issues with no catalog match (debug)
    # Per-pass model routing aggregated across all images. Routing is constant
    # across images for a given run, so we dedupe by pass key (first wins).
    # Each entry: {"pass": "2a", "model_family": "gpt5", "model": "gpt-5.4-mini", "source": "env_override"}
    aggregated_model_routing: List[Dict[str, Any]] = []
    seen_routing_passes: set = set()

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

        # Canonical issues are the truth lane; final/display issues are the UI lane.
        matched_issues = safe_list(payload.get("matched_issues"))
        canonical_issues = safe_list(payload.get("canonical_issues")) or matched_issues
        if canonical_issues and not matched_issues:
            matched_issues = canonical_issues
        estimate_source_issues = canonical_issues or matched_issues or final_issues
        # Removed issues (explicitly discarded with reasons)
        removed_issues = safe_list((passes.get("2e", {}) or {}).get("removed"))

        # -- Backfill stable issue_id + source linkage on issue lanes ------------
        def _stamp_issue(issue: Dict[str, Any], counts: Dict[Tuple[str, str, str], int]) -> bool:
            if not (isinstance(issue, dict) and issue.get("description")):
                return False
            sig = (issue.get("description", ""), issue.get("location_hint", ""), issue.get("label", ""))
            ordinal = counts.get(sig, 0)
            counts[sig] = ordinal + 1
            if not issue.get("issue_id"):
                issue["issue_id"] = make_issue_id(job.job_id, image_key, sig[0], sig[1], sig[2], ordinal)
            issue.setdefault("source_photo_key", image_key)
            issue.setdefault("source_photo_id",  photo_id)
            issue.setdefault("scene",             scene)
            issue.setdefault("scene_group",       scene_group)
            return True

        def _flat_issue(issue: Dict[str, Any], *, source_lane: str) -> Dict[str, Any]:
            return {
                "issue_id":    issue["issue_id"],
                "photo_id":    issue.get("source_photo_id") or photo_id,
                "photo_key":   issue.get("source_photo_key") or image_key,
                "scene":       issue.get("scene") or scene,
                "scene_group": issue.get("scene_group") or scene_group,
                "description": issue.get("description", ""),
                "label":       issue.get("label", ""),
                "location_hint": issue.get("location_hint", ""),
                "room_surrogate_id": issue.get("room_surrogate_id") or issue.get("room_id"),
                "scope_hint": issue.get("scope_hint"),
                "source_lane": source_lane,
                "catalog_item_id":  issue.get("catalogItemId") or issue.get("catalog_item_id") or issue.get("resolved_item_id"),
                "catalog_item_kind": issue.get("catalogItemKind") or issue.get("kind"),
            }

        issue_sig_counts: Dict[Tuple[str, str, str], int] = {}
        for issue in final_issues:
            if _stamp_issue(issue, issue_sig_counts):
                issues_flat.append(_flat_issue(issue, source_lane="display"))

        estimate_sig_counts: Dict[Tuple[str, str, str], int] = {}
        estimate_seen: set = set()
        for issue in estimate_source_issues:
            if not _stamp_issue(issue, estimate_sig_counts):
                continue
            flat = _flat_issue(issue, source_lane="canonical")
            dedupe_key = (
                flat.get("issue_id"),
                flat.get("photo_key"),
                flat.get("catalog_item_id"),
                flat.get("description"),
            )
            if dedupe_key in estimate_seen:
                continue
            estimate_seen.add(dedupe_key)
            estimate_issues_flat.append(flat)

        # -- Build trace (pass internals -- debug/auditing, not read by UI) -----
        pass_2c   = passes.get("2c", {}) or {}
        pass_2d   = passes.get("2d", {}) or {}
        pass_2e_p = passes.get("2e", {}) or {}

        _2c_forward = safe_list(pass_2c.get("labeled_forward") or payload.get("labeled_forward"))
        _2d_resolutions = safe_list(pass_2d.get("resolutions") or payload.get("resolved_items"))

        # Per-pass model routing (populated by orchestrator's _record_model_routing).
        # Carried into the per-photo trace so each photo carries its own copy, and
        # aggregated up to a single property-level list below.
        per_image_routing = payload.get("model_routing", []) or []
        for entry in per_image_routing:
            if not isinstance(entry, dict):
                continue
            pkey = entry.get("pass")
            if not pkey or pkey in seen_routing_passes:
                continue
            seen_routing_passes.add(pkey)
            aggregated_model_routing.append(entry)

        trace = {
            "passes_run":  payload.get("passes_run", []),
            "models":      payload.get("models_used", {}),
            "model_routing": per_image_routing,
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
                "all":        safe_list(payload.get("keywords")),
                "categories": payload.get("keyword_categories"),
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

        # -- Collect unmapped issues from Pass 2d (catalog improvement debug) -----
        resolved = safe_list(payload.get("resolved_items"))
        for item in resolved:
            if item.get("resolved_item_id"):
                continue  # matched — skip
            candidates = item.get("candidates", [])
            top = candidates[0] if candidates else {}
            unmapped_issues.append({
                "text":             item.get("description", ""),
                "scene_group":      scene_group,
                "candidate_bucket": top.get("trade_bucket"),
                "embedding_score":  top.get("score"),
                "kind":             item.get("resolved_kind", ""),
                "photo_key":        image_key,
            })
        # Also capture observations that had zero embedding candidates
        for dbg_row in (payload.get("debug") or {}).get("pass_2d_per_observation", []):
            if dbg_row.get("skipped_reason") == "no_candidates":
                unmapped_issues.append({
                    "text":             dbg_row.get("observation", ""),
                    "scene_group":      dbg_row.get("scene_group") or scene_group,
                    "candidate_bucket": None,
                    "embedding_score":  None,
                    "kind":             dbg_row.get("kind", ""),
                    "photo_key":        image_key,
                })

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
            # Default/base model configuration. With per-pass overrides in play,
            # individual passes may use different models — see top-level
            # `model_routing` array for the per-pass ground truth. These two fields
            # remain as the fallback/default values that non-overridden passes use.
            "default_local_model": getattr(cfg, "LM_STUDIO_MODEL", ""),
            "default_gpt_model":   gpt_config.get('model') if gpt_config else None,
            # Legacy aliases (kept for backward compat with any external readers).
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
        "estimate_issues_flat": estimate_issues_flat,
        "estimate_issues_flat_count": len(estimate_issues_flat),
        # Property-level per-pass routing summary. One entry per LLM pass that ran,
        # deduped across photos. Pass 2f's entry is appended below if 2f executes.
        "model_routing": aggregated_model_routing,
    }

    # renovation_needs disabled: severity not reliable at this stage
    photo_intel["renovation_needs"] = None

    # -- Unmapped issues debug (catalog improvement) ----------------------------
    photo_intel["analysis_debug"] = {
        "unmapped_issues": unmapped_issues,
        "unmapped_count": len(unmapped_issues),
    }
    if unmapped_issues:
        logger.info("Unmapped issues (no catalog match): %d items", len(unmapped_issues))

    # -- Compute scoring (ranking/filtering only; no dollars) -------------------
    try:
        from tools.costing import compute_scoring, CatalogDataError
        scoring = compute_scoring(
            issues_flat=issues_flat,
            issue_catalog=issue_catalog,
            n_photos=len(photos),
        )
        photo_intel["scoring"] = scoring
        logger.info(
            "Scoring: rehab_score=%d raw_points=%s (%d items scored, %d unresolved)",
            scoring["rehab_score"],
            scoring["raw_points"],
            scoring["meta"]["issues_scored"],
            scoring["meta"]["unresolved_issues"],
        )
    except CatalogDataError:
        logger.error("Scoring failed due to faulty catalog data - marking job as failed", exc_info=True)
        raise
    except Exception as exc:
        logger.error(f"Failed to compute scoring: {exc}", exc_info=True)
        photo_intel["scoring"] = None

    # -- Compute deterministic summary v1 ----------------------------------------
    try:
        from tools.property_summary_pass import load_catalog_index, build_property_summary_v1
        catalog_index = load_catalog_index(issue_catalog)
        summary_v1 = build_property_summary_v1(
            property_key=job.property_key,
            run_id=job.job_id,
            issues_flat=issues_flat,
            catalog_index=catalog_index,
        )
        photo_intel["summary_v1"] = summary_v1
        logger.info(
            "Summary V1: %d buckets, top_severity=%d, defects=%d, upgrades=%d",
            len(summary_v1.get("buckets", [])),
            summary_v1.get("listing", {}).get("top_severity", 0),
            summary_v1.get("listing", {}).get("defect_count", 0),
            summary_v1.get("listing", {}).get("upgrade_count", 0),
        )
    except Exception as exc:
        logger.error(f"Failed to compute summary_v1: {exc}", exc_info=True)
        photo_intel["summary_v1"] = None

    # -- Compute renovation estimate (primary cost estimation engine) -------------
    try:
        from tools.renovation_estimate import compute_renovation_estimate
        renovation_issues_flat = estimate_issues_flat or issues_flat

        # Run Pass 2f revisit on eligible candidates if vlm_client available
        import time as _time
        reviewed_candidates = None
        pass_2f_elapsed = 0.0
        pass_2f_model_config = None
        n_attempted = 0
        n_applied = 0
        posture_counts: Dict[str, int] = {}
        fallback_counts: Dict[str, int] = {}
        consistency_flag_counts: Dict[str, int] = {}
        eligible: list = []
        pass_2f_enabled = pass_toggles.get("2f", True) if pass_toggles else True
        if vlm_client and gpt_config and pass_2f_enabled:
            try:
                import asyncio
                from tools.renovation_estimate import (
                    extract_estimate_candidates,
                    resolve_estimate_units,
                    resolve_pass_2f_model_config,
                    run_pass_2f_batch,
                )

                candidates = extract_estimate_candidates(
                    renovation_issues_flat, issue_catalog,
                )
                candidates = resolve_estimate_units(
                    candidates, renovation_issues_flat, issue_catalog,
                )
                eligible = [
                    c for c in candidates
                    if c.estimate_meta.estimate_tier in ("high", "medium")
                    and c.estimate_meta.requires_2f_for_estimate
                ]
                if eligible:
                    # Build photo_key → image_path mapping from job results
                    photo_key_to_path: Dict[str, Path] = {}
                    for res in job.results:
                        pk = Path(res.image_path).name
                        photo_key_to_path[pk] = Path(res.image_path)

                    # Resolve model config for 2f based on provider setting
                    pass_2f_model_config = resolve_pass_2f_model_config(
                        provider=pass_2f_provider,
                        local_config=local_vlm_config,
                        premium_config=gpt_config,
                    )

                    # Record 2f routing on the property-level model_routing list
                    # (the orchestrator only records per-image passes; 2f runs once
                    # per property here, so we add its entry separately).
                    if pass_2f_model_config and "2f" not in {
                        e.get("pass") for e in photo_intel.get("model_routing", [])
                    }:
                        _2f_provider = pass_2f_model_config.get("provider")
                        if _2f_provider is None and pass_2f_model_config.get("api_key"):
                            _2f_provider = "openai"
                        _2f_family = "gpt5" if _2f_provider == "openai" else "qwen"
                        _2f_env_set = bool(os.environ.get("OPENAI_PASS_2F_MODEL"))
                        if _2f_env_set and _2f_provider == "openai":
                            _2f_source = "env_override"
                        elif pass_2f_provider == "premium":
                            _2f_source = "premium_default"
                        else:
                            _2f_source = "standard_default"
                        photo_intel.setdefault("model_routing", []).append({
                            "pass": "2f",
                            "model_family": _2f_family,
                            "model": str(pass_2f_model_config.get("model") or ""),
                            "source": _2f_source,
                        })

                    pass_2f_start = _time.time()
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                loop2 = asyncio.new_event_loop()
                                candidates = pool.submit(
                                    loop2.run_until_complete,
                                    run_pass_2f_batch(
                                        candidates=candidates,
                                        issues_flat=renovation_issues_flat,
                                        issue_catalog=issue_catalog,
                                        vlm_client=vlm_client,
                                        model_config=pass_2f_model_config,
                                        photo_key_to_path=photo_key_to_path,
                                        provider=pass_2f_provider,
                                    ),
                                ).result()
                                loop2.close()
                        else:
                            candidates = loop.run_until_complete(
                                run_pass_2f_batch(
                                    candidates=candidates,
                                    issues_flat=renovation_issues_flat,
                                    issue_catalog=issue_catalog,
                                    vlm_client=vlm_client,
                                    model_config=pass_2f_model_config,
                                    photo_key_to_path=photo_key_to_path,
                                    provider=pass_2f_provider,
                                ),
                            )
                    except RuntimeError:
                        # No event loop exists
                        candidates = asyncio.run(
                            run_pass_2f_batch(
                                candidates=candidates,
                                issues_flat=renovation_issues_flat,
                                issue_catalog=issue_catalog,
                                vlm_client=vlm_client,
                                model_config=pass_2f_model_config,
                                photo_key_to_path=photo_key_to_path,
                                provider=pass_2f_provider,
                            ),
                        )
                    pass_2f_elapsed = _time.time() - pass_2f_start
                    reviewed_candidates = candidates
                    # ── Pass 2f summary (matches 2d/2e style) ──
                    n_attempted = sum(1 for c in candidates if c.pass_2f_attempted)
                    n_applied = sum(1 for c in candidates if c.pass_2f_applied)
                    n_invalidated = sum(1 for c in candidates if c.is_valid_detection is False)
                    for c in candidates:
                        if c.pass_2f_applied and c.review_posture:
                            posture_counts[c.review_posture] = posture_counts.get(c.review_posture, 0) + 1
                    for c in candidates:
                        if c.pass_2f_fallback_reason:
                            fallback_counts[c.pass_2f_fallback_reason] = fallback_counts.get(c.pass_2f_fallback_reason, 0) + 1
                    for c in candidates:
                        for flag in c.review_consistency_flags or []:
                            consistency_flag_counts[flag] = consistency_flag_counts.get(flag, 0) + 1
                    logger.info(
                        "Pass 2f: eligible=%d attempted=%d applied=%d postures=%s fallbacks=%s consistency_flags=%s (%.1fs)",
                        len(eligible),
                        n_attempted,
                        n_applied,
                        posture_counts or {},
                        fallback_counts or {},
                        consistency_flag_counts or {},
                        pass_2f_elapsed,
                    )
            except Exception as exc_2f:
                logger.error(f"Pass 2f batch failed (non-fatal): {exc_2f}", exc_info=True)

        quick_est = compute_renovation_estimate(
            issues_flat=renovation_issues_flat,
            issue_catalog=issue_catalog,
            prebuilt_candidates=reviewed_candidates,
        )
        if reviewed_candidates is not None:
            quick_est["meta"]["pass_2f_ran"] = True
        photo_intel["renovation_estimate"] = quick_est

        # ── Pass 2f trace (top-level, property-scoped) ──
        if reviewed_candidates is not None:
            photo_intel["pass_2f_trace"] = {
                "ran": True,
                "provider": pass_2f_provider,
                "model": (pass_2f_model_config or {}).get("model", "unknown"),
                "elapsed_sec": round(pass_2f_elapsed, 3),
                "eligible_count": len(eligible),
                "attempted_count": n_attempted,
                "applied_count": n_applied,
                "invalidated_count": n_invalidated,
                "posture_counts": posture_counts,
                "fallback_counts": fallback_counts,
                "consistency_flag_counts": consistency_flag_counts,
            }
        elif pass_2f_enabled:
            # Toggle was on but no candidates were reviewed (no eligible or no VLM)
            photo_intel["pass_2f_trace"] = {
                "ran": False,
                "reason": "no_vlm_client" if not vlm_client else "no_eligible_candidates",
            }
        else:
            photo_intel["pass_2f_trace"] = {
                "ran": False,
                "reason": "disabled_by_toggle",
            }
        logger.info(
            "Renovation estimate: $%s-$%s (%d candidates, %d groups, %d high-tier, %d medium-tier)",
            f'{quick_est["raw_totals"]["low"]:,}',
            f'{quick_est["raw_totals"]["high"]:,}',
            quick_est["meta"]["candidate_count"],
            quick_est["meta"]["groups_active"],
            quick_est["meta"]["high_tier_count"],
            quick_est["meta"]["medium_tier_count"],
        )
    except Exception as exc:
        logger.error(f"Failed to compute renovation estimate: {exc}", exc_info=True)
        photo_intel["renovation_estimate"] = None

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
                    "model", "gpt_model", "default_local_model", "default_gpt_model"):
            _run.pop(_k, None)

    # Strip analysis_debug and pass_2f_trace (debug-only, not for UI)
    slim.pop("analysis_debug", None)
    slim.pop("pass_2f_trace", None)

    _strip_pass_2f_audit_rationale(slim.get("renovation_estimate"))

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

    # -- Write summary_v1 to standalone property_summary.json -------------------
    if photo_intel.get("summary_v1") is not None:
        try:
            summary_path = output_path.parent / "property_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(photo_intel["summary_v1"], f, indent=2, ensure_ascii=False)
            logger.info(f"Property summary (summary_v1) saved to: {summary_path}")
        except Exception as exc:
            logger.error(f"Failed to write property_summary.json: {exc}", exc_info=True)

    return output_path
