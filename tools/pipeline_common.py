# tools/pipeline_common.py
"""
Shared constants, ID generators, and pure utility functions used across
the analysis pipeline (auto_analyzer, artifact_writers, detection_pipeline,
scene_classifier_service).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

# Room grouping map for UI aggregation
SCENE_GROUPS_UI = {
    "kitchen": ["kitchen", "pantry"],
    "bathroom": ["bathroom"],
    "bedroom": ["bedroom", "closet"],
    "living_areas": ["living_room", "dining_room", "home_office", "hallway", "stairway"],
    "utility": ["laundry_room", "basement", "attic", "garage", "hvac"],
    "exterior": ["exterior_front", "exterior_back", "exterior_side", "yard", "patio", "deck", "balcony", "driveway",
                 "pool", "garden"],
    "other": ["roof", "other", "unknown", "floor_plan", "aerial_view", "street_view"],
}

# Reverse lookup: scene -> group
SCENE_TO_GROUP_UI: Dict[str, str] = {}
for _group, _scenes in SCENE_GROUPS_UI.items():
    for _scene in _scenes:
        SCENE_TO_GROUP_UI[_scene] = _group

PHOTO_INTEL_SCHEMA_VERSION = "photo_intel_v3"
PROPERTY_SUMMARY_SCHEMA_VERSION = "property_summary_v3"
NORMALIZATION_POLICY_VERSION = "workitem_v1"


def stable_hash_id(*parts: str, length: int = 12) -> str:
    """Generate a stable, deterministic short hash ID from input parts."""
    combined = "|".join(str(p) if p is not None else "" for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:length]


def make_photo_id(property_key: str, run_id: str, photo_key: str) -> str:
    """Generate deterministic photo ID."""
    return stable_hash_id(property_key, run_id, photo_key, length=16)


def make_issue_id(
    run_id: str,
    photo_key: str,
    description: str,
    location_hint: str,
    label: str,
    ordinal: int = 0
) -> str:
    """Generate deterministic issue ID. Ordinal handles duplicate issues in same photo."""
    return stable_hash_id(run_id, photo_key, description, location_hint, label, str(ordinal), length=16)


def normalize_label_for_hint(label: Optional[str]) -> str:
    if not label:
        return ""
    s = str(label).strip().lower()
    s = s.replace("-", " ").replace("_", " ")
    return " ".join(s.split())


def get_roi_hint_map_for_scene(cfg: Any, scene: str) -> Dict[str, str]:
    """Get ROI hint mapping for a given scene."""
    m = getattr(cfg, "ROI_HINTS_BY_SCENE", None)
    if not isinstance(m, dict):
        return {}

    # Prefer exact scene key, then group key, then default
    keys = [scene]
    group = SCENE_TO_GROUP_UI.get(scene)
    if group:
        keys.append(group)
    keys.append("default")

    scene_map = {}
    for k in keys:
        v = m.get(k)
        if isinstance(v, dict) and v:
            scene_map = v
            break

    out: Dict[str, str] = {}
    for lbl, zone in (scene_map or {}).items():
        nl = normalize_label_for_hint(lbl)
        if nl and zone:
            out[nl] = str(zone).strip().lower()
    return out


def maybe_backfill_planner_hints(cfg: Any, scene: str, planner_hints: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Backfill planner hints from config if not already present."""
    if not getattr(cfg, "ROI_HINTS_ENABLED", False):
        return planner_hints or {}
    existing = dict(planner_hints or {})
    if existing:
        return existing
    return get_roi_hint_map_for_scene(cfg, scene)


def safe_list(x) -> List[Any]:
    """Ensure x is a list."""
    return x if isinstance(x, list) else []
