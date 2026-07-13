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
    PRODUCT_POLICY_VERSION,
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


def _build_ui_priorities_v1(
    *,
    issues_flat: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
    renovation_estimate_v4: Any,
) -> Dict[str, Any]:
    """Build frontend priority lanes from verified/package-first v4 output."""
    try:
        from tools.rehab_packages import (
            ACTIVE_PACKAGE_STATUSES,
            DISPLAY_CLASS_ESTIMATE_DRIVER,
            DISPLAY_CLASS_HIGH_CONCERN,
            DISPLAY_CLASS_MARKETABILITY,
            PACKAGE_CATEGORY_MODERNIZATION,
            PACKAGE_CATEGORY_REPAIR,
            PACKAGE_CATEGORY_TURNOVER,
            catalog_display_class,
        )
    except Exception:
        return {"version": "ui_priorities_v1", "error": "catalog_priority_helpers_unavailable"}

    catalog_lookup = {
        item["id"]: item
        for item in (issue_catalog.get("items") or [])
        if isinstance(item, dict) and item.get("id")
    }

    def _issue_record(issue: Dict[str, Any]) -> Dict[str, Any]:
        item_id = issue.get("catalog_item_id") or issue.get("defect_id") or issue.get("upgrade_id")
        cat = catalog_lookup.get(item_id or "", {})
        return {
            "issue_id": issue.get("issue_id"),
            "catalog_item_id": item_id,
            "display_class": catalog_display_class(cat),
            "photo_key": issue.get("photo_key"),
            "scene": issue.get("scene") or issue.get("scene_group"),
            "description": issue.get("description"),
        }

    v4 = renovation_estimate_v4 if isinstance(renovation_estimate_v4, dict) else {}
    line_items: List[Dict[str, Any]] = []
    for group in v4.get("groups") or []:
        if isinstance(group, dict):
            line_items.extend(group.get("line_items") or [])

    verified_estimate_drivers = []
    high_concern_issues = []
    marketability_signals = []
    for item in line_items:
        cat = catalog_lookup.get(item.get("catalog_item_id") or "", {})
        display_class = catalog_display_class(cat)
        if item.get("is_valid_detection") is False:
            continue
        record = {
            "estimate_unit_id": item.get("estimate_unit_id"),
            "catalog_item_id": item.get("catalog_item_id"),
            "name": item.get("name"),
            "display_class": display_class,
            "cost_low": item.get("cost_low"),
            "cost_high": item.get("cost_high"),
            "visual_verification_status": item.get("visual_verification_status"),
            "review_source": item.get("review_source"),
        }
        if display_class == DISPLAY_CLASS_ESTIMATE_DRIVER:
            verified_estimate_drivers.append(record)
        elif display_class == DISPLAY_CLASS_HIGH_CONCERN:
            high_concern_issues.append(record)
        elif display_class == DISPLAY_CLASS_MARKETABILITY:
            marketability_signals.append(record)

    active_packages = [
        package for package in (v4.get("packages") or [])
        if isinstance(package, dict) and package.get("verification_status") in ACTIVE_PACKAGE_STATUSES
    ]
    audit_only = [
        package for package in (v4.get("package_candidates") or [])
        if isinstance(package, dict) and package.get("verification_status") not in ACTIVE_PACKAGE_STATUSES
    ]
    confirmed_packages_by_category = {
        PACKAGE_CATEGORY_MODERNIZATION: [
            p for p in active_packages if p.get("package_category") == PACKAGE_CATEGORY_MODERNIZATION
        ],
        PACKAGE_CATEGORY_REPAIR: [
            p for p in active_packages if p.get("package_category") == PACKAGE_CATEGORY_REPAIR
        ],
        PACKAGE_CATEGORY_TURNOVER: [
            p for p in active_packages if p.get("package_category") == PACKAGE_CATEGORY_TURNOVER
        ],
    }
    raw_high_concern = [
        _issue_record(issue)
        for issue in (issues_flat or [])
        if catalog_display_class(catalog_lookup.get(
            issue.get("catalog_item_id") or issue.get("defect_id") or issue.get("upgrade_id") or "",
            {},
        )) == DISPLAY_CLASS_HIGH_CONCERN
    ]

    return {
        "version": "ui_priorities_v1",
        "verified_estimate_drivers": verified_estimate_drivers,
        "high_concern_issues": high_concern_issues or raw_high_concern,
        # Legacy lane: sourced from the categorized map so the name stays accurate
        # as repair/turnover categories land alongside modernization.
        "confirmed_modernization_packages": confirmed_packages_by_category[PACKAGE_CATEGORY_MODERNIZATION],
        "marketability_signals": marketability_signals,
        "audit_only_suppressed_or_unverified": audit_only,
        # NEW additive map keyed by package_category.
        "confirmed_packages_by_category": confirmed_packages_by_category,
    }


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


_PROPERTY_METADATA_KEYS = (
    "price", "list_price", "listing_price",
    "beds", "bedrooms", "bed_count", "bedroom_count",
    "baths", "bathrooms", "bath_count", "bathroom_count",
    "full_baths", "full_bathrooms", "half_baths", "half_bathrooms",
    "sqft", "square_feet", "living_area_sqft",
    "year_built", "lot_size", "property_type", "property_type_detail",
    "status", "description", "hoa", "garage", "parking", "price_per_sqft",
    "area_price_per_sqft",
    "days_on_market",
    # Multi-kitchen / multi-unit evidence consumed by
    # tools.estimate_units._has_multi_kitchen_metadata_evidence.
    "kitchen_count", "kitchens",
    "is_multi_unit", "multi_unit",
    "has_adu", "adu",
    "has_second_kitchen", "second_kitchen",
    "number_of_units", "unit_count", "units",
    # Provenance: keep the caller-supplied source label (e.g. "csv_funnel_db")
    # so artifacts record where metadata came from. Scrape path fills this via
    # setdefault below when absent.
    "metadata_source",
)


def _extract_property_metadata_from_mapping(source: Any) -> Dict[str, Any]:
    if not isinstance(source, dict):
        return {}

    metadata: Dict[str, Any] = {}
    nested = source.get("metadata")
    if isinstance(nested, dict):
        metadata.update(_extract_property_metadata_from_mapping(nested))

    for key in _PROPERTY_METADATA_KEYS:
        value = source.get(key)
        if value is not None:
            metadata[key] = value

    # Keep common aliases available to downstream caps/sanity checks even when
    # the scraper only wrote the Redfin-style short names.
    if "price" in metadata:
        metadata.setdefault("list_price", metadata["price"])
    if "beds" in metadata:
        metadata.setdefault("bedrooms", metadata["beds"])
    if "baths" in metadata:
        metadata.setdefault("bath_count", metadata["baths"])
    if "sqft" in metadata:
        metadata.setdefault("square_feet", metadata["sqft"])

    return metadata


def _discover_realtorvision_root_from_images(job: Any) -> Optional[Path]:
    for res in getattr(job, "results", []) or []:
        image_path = getattr(res, "image_path", None)
        if not image_path:
            continue
        try:
            parts = Path(image_path).resolve().parts
        except Exception:
            continue
        lowered = [part.lower() for part in parts]
        for idx in range(0, max(0, len(parts) - 2)):
            if lowered[idx:idx + 3] == ["public", "images", "properties"]:
                if idx == 0:
                    return None
                return Path(*parts[:idx])
    return None


def _load_scrape_metadata_for_job(job: Any) -> Dict[str, Any]:
    property_key = str(getattr(job, "property_key", "") or "").strip()
    if not property_key:
        return {}

    root = _discover_realtorvision_root_from_images(job)
    if root is None:
        return {}

    batches_dir = root / "data" / "scraped" / "batches"
    if not batches_dir.exists():
        return {}

    scrape_paths = []
    try:
        scrape_paths = list(batches_dir.glob(f"*/properties/{property_key}/scrape.json"))
    except Exception:
        return {}
    if not scrape_paths:
        return {}

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    for scrape_path in sorted(scrape_paths, key=_mtime, reverse=True):
        try:
            with open(scrape_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.debug("Failed reading scrape metadata %s: %s", scrape_path, exc)
            continue
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        metadata = _extract_property_metadata_from_mapping(payload)
        if metadata:
            metadata.setdefault("metadata_source", "scrape_json")
            metadata.setdefault("metadata_path", str(scrape_path))
            return metadata
    return {}


def _resolve_property_metadata(job: Any) -> Dict[str, Any]:
    metadata = _extract_property_metadata_from_mapping(
        getattr(job, "property_metadata", None)
    )
    scrape_metadata = _load_scrape_metadata_for_job(job)
    if scrape_metadata:
        scrape_metadata.update(metadata)
        metadata = scrape_metadata
    return metadata


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

    property_metadata = _resolve_property_metadata(job)
    property_block = {
        "property_key": job.property_key,
        "artifacts_dir": job.artifacts_dir,
    }
    if property_metadata:
        property_block.update(property_metadata)

    # Product lanes: raw lanes above are the internal audit record; every
    # product surface (scoring, summary, estimate, priorities) consumes only
    # these quarantine-filtered views.
    from tools.renovation_estimate import filter_product_issues
    product_issues_flat = filter_product_issues(issues_flat, issue_catalog)
    product_estimate_issues_flat = filter_product_issues(
        estimate_issues_flat, issue_catalog,
    )

    photo_intel = {
        "schema_version":               PHOTO_INTEL_SCHEMA_VERSION,
        "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
        "product_policy_version":       PRODUCT_POLICY_VERSION,
        "catalog_version":              str(issue_catalog.get("version") or ""),
        "product_projection_status":    "native",
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
        "property": property_block,
        "property_metadata": property_metadata,
        "photos":      photos,
        "room_groups": room_groups,
        "issues_flat": issues_flat,
        "issues_flat_count": len(issues_flat),
        "estimate_issues_flat": estimate_issues_flat,
        "estimate_issues_flat_count": len(estimate_issues_flat),
        "product_issues_flat": product_issues_flat,
        "product_issues_flat_count": len(product_issues_flat),
        "product_estimate_issues_flat": product_estimate_issues_flat,
        "product_estimate_issues_flat_count": len(product_estimate_issues_flat),
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
            issues_flat=product_issues_flat,
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
            issues_flat=product_issues_flat,
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
        # Lane choice follows the RAW canonical lane's existence; quarantine
        # filtering must never switch which lane feeds the estimate.
        renovation_issues_flat = (
            product_estimate_issues_flat if estimate_issues_flat
            else product_issues_flat
        )

        # Pass 2f is now package-level visual verification in v4.
        reviewed_candidates = None
        pass_2f_model_config = None
        photo_key_to_path: Dict[str, Path] = {}
        pass_2f_enabled = pass_toggles.get("2f", True) if pass_toggles else True
        if vlm_client and gpt_config and pass_2f_enabled:
            try:
                from tools.renovation_estimate import resolve_pass_2f_model_config

                for res in job.results:
                    pk = Path(res.image_path).name
                    photo_key_to_path[pk] = Path(res.image_path)
                pass_2f_model_config = resolve_pass_2f_model_config(
                    provider=pass_2f_provider,
                    local_config=local_vlm_config,
                    premium_config=gpt_config,
                )

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
            except Exception as exc_2f:
                logger.error(f"Pass 2f setup failed (non-fatal): {exc_2f}", exc_info=True)

        from tools.renovation_estimate_v4 import compute_renovation_estimate_v4
        v4_est = compute_renovation_estimate_v4(
            issues_flat=renovation_issues_flat,
            issue_catalog=issue_catalog,
            photos=photos,
            v3_reviewed_candidates=reviewed_candidates,
            property_metadata=property_metadata,
            pass_2f_vlm_client=vlm_client if pass_2f_enabled else None,
            pass_2f_model_config=pass_2f_model_config,
            photo_key_to_path=photo_key_to_path,
            pass_2f_provider=pass_2f_provider,
        )
        photo_intel["renovation_estimate_v4"] = v4_est
        photo_intel["ui_priorities_v1"] = _build_ui_priorities_v1(
            issues_flat=renovation_issues_flat,
            issue_catalog=issue_catalog,
            renovation_estimate_v4=photo_intel["renovation_estimate_v4"],
        )

        v4_pass_2f_trace = (
            photo_intel.get("renovation_estimate_v4", {}) or {}
        ).get("pass_2f_trace")
        if isinstance(v4_pass_2f_trace, dict):
            photo_intel["pass_2f_trace"] = {
                **v4_pass_2f_trace,
                "mode": "package_visual_verification",
            }
        elif pass_2f_enabled:
            photo_intel["pass_2f_trace"] = {
                "ran": False,
                "reason": "no_vlm_client" if not vlm_client else "no_package_candidates",
                "mode": "package_visual_verification",
            }
        else:
            photo_intel["pass_2f_trace"] = {
                "ran": False,
                "reason": "disabled_by_toggle",
                "mode": "package_visual_verification",
            }
        logger.info(
            "Renovation estimate v4: $%s-$%s (%d candidates, %d groups, %d high-tier, %d medium-tier)",
            f'{v4_est["final_rehab"]["low"]:,}',
            f'{v4_est["final_rehab"]["high"]:,}',
            v4_est["meta"]["candidate_count"],
            v4_est["meta"]["groups_active"],
            v4_est["meta"]["high_tier_count"],
            v4_est["meta"]["medium_tier_count"],
        )
    except Exception as exc:
        logger.error(f"Failed to compute renovation estimate: {exc}", exc_info=True)
        photo_intel["renovation_estimate_v4"] = None

    # -- Build defect events, work items, and search index ----------------------
    # NOTE: if this layer is revived, it consumes raw per-photo issues and MUST
    # exclude product-quarantined trades (see filter_product_issues) before
    # producing work_items / search_index — both are product surfaces.
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

    _strip_pass_2f_audit_rationale(slim.get("renovation_estimate_v4"))

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
