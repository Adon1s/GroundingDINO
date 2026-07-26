# tools/pipeline_common.py
"""
Shared constants, ID generators, and pure utility functions used across
the analysis pipeline (artifact_writers, detection_pipeline,
scene_classifier_service).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


# =============================================================================
# Canonical scene taxonomy
# -----------------------------------------------------------------------------
# The single source of truth for scene ids across the pipeline. Everything that
# used to hand-maintain its own copy (orchestrator, catalog auditor, model
# comparison, property summarizer, artifact viewer, room surrogates) derives
# from this table instead.
#
# Columns:
#   scene_id       canonical scene label
#   group          UI/retrieval scene group (7 tokens)
#   pass_1a_order  position in the Pass 1a classifier prompt; None = not offered
#                  to the classifier but still recognized downstream (legacy and
#                  nonconforming artifacts, room surrogates, retrieval gating)
#   breaking       opens a new room surrogate (see room_surrogates.py)
#
# Row order is load-bearing: SCENE_GROUPS_UI values are serialized into
# artifacts as `scenes_included`, so keep rows grouped and ordered as-is.
# =============================================================================

@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    group: str
    pass_1a_order: Optional[int]
    breaking: bool


SCENE_SPECS: Tuple[SceneSpec, ...] = tuple(
    SceneSpec(*_row) for _row in (
        # scene_id,         group,           pass_1a_order, breaking
        ("kitchen",         "kitchen",       4,             True),
        ("pantry",          "kitchen",       None,          True),
        ("bathroom",        "bathroom",      7,             True),
        ("bedroom",         "bedroom",       5,             True),
        ("closet",          "bedroom",       6,             False),
        ("living_room",     "living_areas",  3,             True),
        ("dining_room",     "living_areas",  8,             True),
        ("home_office",     "living_areas",  None,          True),
        ("hallway",         "living_areas",  None,          False),
        ("stairway",        "living_areas",  None,          False),
        ("laundry_room",    "utility",       None,          True),
        ("basement",        "utility",       9,             True),
        ("attic",           "utility",       10,            True),
        ("garage",          "utility",       11,            True),
        ("hvac",            "utility",       15,            False),
        ("exterior_front",  "exterior",      0,             False),
        ("exterior_back",   "exterior",      1,             False),
        ("exterior_side",   "exterior",      2,             False),
        ("yard",            "exterior",      12,            False),
        ("patio",           "exterior",      None,          False),
        ("deck",            "exterior",      None,          False),
        ("balcony",         "exterior",      None,          False),
        ("driveway",        "exterior",      None,          False),
        ("pool",            "exterior",      13,            False),
        ("garden",          "exterior",      None,          False),
        ("roof",            "other",         14,            False),
        ("other",           "other",         16,            False),
        ("unknown",         "other",         None,          False),
        ("floor_plan",      "other",         None,          False),
        ("aerial_view",     "other",         None,          False),
        ("street_view",     "other",         None,          False),
    )
)

# Fallback scene id for anything the classifier emits that isn't canonical.
SCENE_OTHER = "other"

# Room grouping map for UI aggregation: group -> [scene ids]
SCENE_GROUPS_UI: Dict[str, List[str]] = {}
for _spec in SCENE_SPECS:
    SCENE_GROUPS_UI.setdefault(_spec.group, []).append(_spec.scene_id)

# Reverse lookup: scene -> group
SCENE_TO_GROUP_UI: Dict[str, str] = {s.scene_id: s.group for s in SCENE_SPECS}

ALL_SCENE_IDS: FrozenSet[str] = frozenset(SCENE_TO_GROUP_UI)

# The scene ids Pass 1a offers the classifier, in prompt order.
PASS_1A_SCENE_IDS: Tuple[str, ...] = tuple(
    s.scene_id
    for s in sorted(
        (s for s in SCENE_SPECS if s.pass_1a_order is not None),
        key=lambda s: s.pass_1a_order,
    )
)

# Scenes that open a room surrogate, and everything else that is still a
# recognized scene id.
BREAKING_SCENES: FrozenSet[str] = frozenset(s.scene_id for s in SCENE_SPECS if s.breaking)
NON_BREAKING_SCENES: FrozenSet[str] = ALL_SCENE_IDS - BREAKING_SCENES


def normalize_scene_id(raw: Any) -> str:
    """Normalize a model/artifact scene label to a canonical scene id.

    Trims and lowercases, then accepts any id in the canonical table — including
    the scenes Pass 1a doesn't offer, since off-contract-but-recognized output
    (``laundry_room``, ``pantry``) still groups and clusters correctly. Anything
    genuinely unknown collapses to ``other``.
    """
    scene = str(raw or "").strip().lower()
    return scene if scene in ALL_SCENE_IDS else SCENE_OTHER


PHOTO_INTEL_SCHEMA_VERSION = "photo_intel_v3"
PROPERTY_SUMMARY_SCHEMA_VERSION = "property_summary_v3"
NORMALIZATION_POLICY_VERSION = "workitem_v1"
# Product-quarantine policy version. Bump when quarantine semantics change.
# Readers must treat a missing version as "legacy / needs projection",
# never as safe.
PRODUCT_POLICY_VERSION = "quarantine_v1"


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
