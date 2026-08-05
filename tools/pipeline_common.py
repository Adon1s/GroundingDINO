# tools/pipeline_common.py
"""
Shared constants, ID generators, and pure utility functions used across
the analysis pipeline (artifact_writers, detection_pipeline,
scene_classifier_service).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Pattern, Tuple


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


def normalize_scene_group(raw: Any) -> str:
    """Normalize a model/artifact scene-group label to a canonical group.

    The group vocabulary is the closed 7-token set in ``SCENE_GROUPS_UI``, so
    anything else — a scene id, a plural, an empty value — collapses to
    ``other``. Callers must use this rather than a generic room-hint filter:
    ``exterior`` is both a scene group and a generic room word, and filters
    written for room hints strip it.
    """
    group = str(raw or "").strip().lower()
    return group if group in SCENE_GROUPS_UI else SCENE_OTHER


PHOTO_INTEL_SCHEMA_VERSION = "photo_intel_v3"
PROPERTY_SUMMARY_SCHEMA_VERSION = "property_summary_v3"
NORMALIZATION_POLICY_VERSION = "workitem_v1"
# Product-quarantine policy version. Bump when quarantine semantics change.
# Readers must treat a missing version as "legacy / needs projection",
# never as safe.
PRODUCT_POLICY_VERSION = "quarantine_v1"

# Observation-kind ontology. Stored artifacts without an ontology_version were
# written under the two-kind (defect | upgrade) contract: read them as
# legacy_v1 and never reinterpret their historical kind values against the
# three-kind (defect | degradation | modernization) ontology.
LEGACY_ONTOLOGY_VERSION = "legacy_v1"


def artifact_ontology_version(artifact: Any) -> str:
    """Ontology version of a stored artifact dict; missing field ⇒ legacy_v1."""
    if isinstance(artifact, dict):
        version = str(artifact.get("ontology_version") or "").strip()
        if version:
            return version
    return LEGACY_ONTOLOGY_VERSION


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


# =============================================================================
# Lexical term matching
# -----------------------------------------------------------------------------
# Single matcher for every catalog keyword list (deny_any / require_any /
# support_any) and for the routing/damage token tables in the pass code.
# Raw substring containment used to fire on cross-word collisions — "ding"
# inside "siding", "aged" inside "damaged", "rat" inside "discoloration" —
# which both blocked and force-resolved items that had nothing to do with the
# observation.
#
# Leading \b alone cannot stop a term matching a longer word it *prefixes*, and
# a global trailing \b is not an option: it drops require_any to zero for ~12
# items whose terms are singular stems, and "stain" alone falls 616 -> 10
# corpus hits. So the trailing boundary is opt-in per term, authored in the
# data as a trailing "$" — the catalog stays authoritative and the matcher
# stays dumb. Used today by "mold$" (else "crown molding" reads as mold in half
# the corpus hits), "wall$" ("wallpaper"), "tub$" ("tube") and "rat$"
# ("rather"). See docs/HANDOFF_catalog_term_hygiene.md.
# =============================================================================

TERM_WHOLE_WORD_MARKER = "$"


def strip_term_marker(term: str) -> str:
    """Return the human-readable term, without its whole-word marker.

    Use anywhere a raw keyword is rendered for a person or a model — prompts,
    reports — so the marker never leaks out of the matching layer.
    """
    text = str(term or "")
    return text[:-1] if text.endswith(TERM_WHOLE_WORD_MARKER) else text


@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> Pattern[str]:
    if term.endswith(TERM_WHOLE_WORD_MARKER):
        return re.compile(r"\b" + re.escape(term[:-1]) + r"\b")
    return re.compile(r"\b" + re.escape(term))


def term_matches(term: str, text_lower: str) -> bool:
    """Word-start anchored containment for a lowercased term against lowered text.

    Leading \\b only, so terms stay authorable as stems: "stain" still covers
    "stained"/"stains" and "deteriorat" covers "deteriorating", while "ding" no
    longer fires inside "siding". A term ending in "$" opts in to a trailing
    \\b as well, for stems whose prefix extensions are a different concept.
    Multi-word phrases work unchanged because re.escape preserves internal
    spaces. Patterns compile once per term.
    """
    if not term or term == TERM_WHOLE_WORD_MARKER:
        return False
    return _term_pattern(term).search(text_lower) is not None
