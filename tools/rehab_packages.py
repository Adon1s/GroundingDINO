"""
tools/rehab_packages.py

Deterministic package inference and reconciliation for renovation_estimate_v4.

Public API:
  - infer_package_candidates(candidates, room_surrogates, issue_catalog) -> list[dict]
        Strength-scored per-room package inference. Returns package candidates
        with audit_only=True / estimate_eligible=False until Pass 2f verification
        promotes them. Sole production inference entrypoint.
  - reconcile_packages_and_estimate_units(groups_out, packages) -> dict
        Normalize line items into per-billable-member child unit records,
        absorb children into matching packages all-or-nothing, recompute
        retained group totals using v3's stack-behavior rule, emit audit
        + explicit estimate buckets. Mutates groups_out and packages in place.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.catalog_cost_model import (
    COST_MODEL_SOURCE_DERIVED_PACKAGE,
    COST_MODEL_SOURCE_INVALID_CATALOG_FALLBACK,
    INSPECTION_ALLOWANCE,
    LINE_ITEM,
    PACKAGE_ALLOWANCE,
    ROOM_ALLOWANCE,
    derive_package_cost_model,
)
from tools import cost_factors
from tools.estimate_scope import (
    INSPECTION_RISK,
    MARKETABILITY_REHAB,
    REQUIRED_REHAB,
    VALID_ESTIMATE_SCOPES,
    add_scope_amount,
    allocate_capped_scope_totals,
    build_scope_headline_tiers,
    classify_package_scope,
    empty_scope_totals,
    normalize_scope_key,
    sum_scope_totals,
)
from tools.pipeline_common import SCENE_TO_GROUP_UI
from tools.renovation_estimate import GROUP_BUDGET_CAPS, EstimateCandidate

logger = logging.getLogger(__name__)


# ── Package level cost ranges (pricing_tier, low, high) ──────────────────────
KITCHEN_MINOR_REPAIR    = ("kitchen_minor_repair",    1_000,  5_000)
KITCHEN_REFRESH         = ("kitchen_refresh",         6_000, 18_000)
KITCHEN_PARTIAL_REHAB   = ("kitchen_partial_rehab",  15_000, 35_000)
KITCHEN_FULL_REHAB      = ("kitchen_full_rehab",     30_000, 70_000)
KITCHEN_REPAIR_LIGHT    = ("kitchen_repair_light",      750,  3_500)
KITCHEN_REPAIR_HEAVY    = ("kitchen_repair_heavy",    3_500, 12_000)
KITCHEN_TURNOVER_LIGHT  = ("kitchen_turnover_light",    600,  2_500)
KITCHEN_TURNOVER_STD    = ("kitchen_turnover_std",    2_500,  6_500)

BATHROOM_MINOR_REPAIR   = ("bathroom_minor_repair",     500,  3_000)
BATHROOM_REFRESH        = ("bathroom_refresh",        3_000, 10_000)
BATHROOM_PARTIAL_REHAB  = ("bathroom_partial_rehab",  8_000, 20_000)
BATHROOM_FULL_REHAB     = ("bathroom_full_rehab",    15_000, 35_000)
BATHROOM_REPAIR_LIGHT   = ("bathroom_repair_light",     400,  2_500)
BATHROOM_REPAIR_HEAVY   = ("bathroom_repair_heavy",   2_500,  8_000)
BATHROOM_TURNOVER_LIGHT = ("bathroom_turnover_light",   500,  2_000)
BATHROOM_TURNOVER_STD   = ("bathroom_turnover_std",   1_800,  4_500)

ROOM_REFRESH            = ("room_refresh",              750,  4_000)
ROOM_REPAIR_HEAVY       = ("room_repair_heavy",       3_000, 10_000)

# Bedroom (cosmetic-led: flooring + paint + lighting + trim/doors/closet).
# Lightweight tier set: modernization uses refresh -> full_rehab (no partial
# middle tier), repair uses light/heavy, turnover uses light/std.
BEDROOM_REFRESH         = ("bedroom_refresh",         1_500,  6_000)
BEDROOM_FULL_REHAB      = ("bedroom_full_rehab",      5_000, 15_000)
BEDROOM_REPAIR_LIGHT    = ("bedroom_repair_light",      500,  2_500)
BEDROOM_REPAIR_HEAVY    = ("bedroom_repair_heavy",    2_500,  8_000)
BEDROOM_TURNOVER_LIGHT  = ("bedroom_turnover_light",    500,  2_000)
BEDROOM_TURNOVER_STD    = ("bedroom_turnover_std",    1_800,  5_000)

# Living room (larger scope than bedroom; often open to dining).
LIVING_REFRESH          = ("living_refresh",          2_000,  8_000)
LIVING_FULL_REHAB       = ("living_full_rehab",       6_000, 18_000)
LIVING_REPAIR_LIGHT     = ("living_repair_light",       600,  3_000)
LIVING_REPAIR_HEAVY     = ("living_repair_heavy",     3_000, 10_000)
LIVING_TURNOVER_LIGHT   = ("living_turnover_light",     600,  2_500)
LIVING_TURNOVER_STD     = ("living_turnover_std",     2_200,  6_000)


_QUALIFIED_POSTURES = frozenset({
    "repair", "replace", "keep_default",                    # Pass 2f outputs
    "repair_only", "replace_only", "repair_or_replace",     # catalog strategy fallbacks
})
# Implicitly excludes: "inspect", "inspect_only", "service_only", None.

_OPTIONAL_TIER = "optional"

CAP_BEHAVIOR_RESPECT_GROUP_CAP = "respect_group_cap"
CAP_BEHAVIOR_ALLOW_ABOVE_GROUP_CAP = "allow_above_group_cap"
CAP_BEHAVIOR_REPLACE_GROUP_CAP = "replace_group_cap"
_VALID_CAP_BEHAVIORS = frozenset({
    CAP_BEHAVIOR_RESPECT_GROUP_CAP,
    CAP_BEHAVIOR_ALLOW_ABOVE_GROUP_CAP,
    CAP_BEHAVIOR_REPLACE_GROUP_CAP,
})


@dataclass
class PackageBuilderInputs:
    """Original builder inputs for a package candidate, kept OUT of the package
    dict (never serialized). Registered per package_id so post-2f re-tiering and
    bathroom expansion can re-run the pure pricing/strength/scope resolvers on a
    confirmed-only candidate subset. The EstimateCandidate lists alias the live
    pipeline objects — never mutate them here."""

    package_type: str
    unit_id: str
    supporting_candidates: List[EstimateCandidate] = field(default_factory=list)
    drivers: List[EstimateCandidate] = field(default_factory=list)
    supports: List[EstimateCandidate] = field(default_factory=list)
    catalog_lookup: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    candidate_catalog_meta: Optional[Dict[int, Dict[str, Any]]] = None
    trigger_reason: str = ""


def _is_estimate_display_only(pkg: Dict[str, Any]) -> bool:
    """True for packages shown in the UI but contributing $0 to estimate totals
    (currently only the whole-home turnover aggregate)."""
    return bool(pkg.get("estimate_display_only"))

DISPLAY_CLASS_ESTIMATE_DRIVER = "estimate_driver"
DISPLAY_CLASS_HIGH_CONCERN = "high_concern"
DISPLAY_CLASS_MARKETABILITY = "marketability"
DISPLAY_CLASS_CLUTTER = "clutter"
DISPLAY_CLASS_HIDDEN = "hidden"
VALID_DISPLAY_CLASSES = frozenset({
    DISPLAY_CLASS_ESTIMATE_DRIVER,
    DISPLAY_CLASS_HIGH_CONCERN,
    DISPLAY_CLASS_MARKETABILITY,
    DISPLAY_CLASS_CLUTTER,
    DISPLAY_CLASS_HIDDEN,
})

PACKAGE_ROLE_STANDALONE = "standalone"
PACKAGE_ROLE_DRIVER = "package_driver"
PACKAGE_ROLE_SUPPORT = "package_support"
PACKAGE_ROLE_IGNORE = "ignore"
VALID_PACKAGE_ROLES = frozenset({
    PACKAGE_ROLE_STANDALONE,
    PACKAGE_ROLE_DRIVER,
    PACKAGE_ROLE_SUPPORT,
    PACKAGE_ROLE_IGNORE,
})

PACKAGE_TYPE_KITCHEN_MODERNIZATION = "kitchen_modernization"
PACKAGE_TYPE_KITCHEN_REPAIR = "kitchen_repair"
PACKAGE_TYPE_KITCHEN_TURNOVER = "kitchen_turnover"
PACKAGE_TYPE_BATHROOM_MODERNIZATION = "bathroom_modernization"
PACKAGE_TYPE_BATHROOM_REPAIR = "bathroom_repair"
PACKAGE_TYPE_BATHROOM_TURNOVER = "bathroom_turnover"
PACKAGE_TYPE_BEDROOM_MODERNIZATION = "bedroom_modernization"
PACKAGE_TYPE_BEDROOM_REPAIR = "bedroom_repair"
PACKAGE_TYPE_BEDROOM_TURNOVER = "bedroom_turnover"
PACKAGE_TYPE_LIVING_MODERNIZATION = "living_modernization"
PACKAGE_TYPE_LIVING_REPAIR = "living_repair"
PACKAGE_TYPE_LIVING_TURNOVER = "living_turnover"
PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH = "interior_paint_flooring_refresh"
# Naming note: the strings above are *package_type* identifiers used as keys to
# _PACKAGE_TYPE_TO_CATEGORY / _PACKAGE_TYPE_TO_ROOM / VALID_PACKAGE_TYPES.
# The *pricing-tier* strings (e.g. "bathroom_turnover_light", "kitchen_full_rehab")
# live in the per-tier constant tuples above and are the keys to
# _PACKAGE_ABSORPTION_SCOPES. The two namespaces are intentionally separate.
VALID_PACKAGE_TYPES = frozenset({
    PACKAGE_TYPE_KITCHEN_MODERNIZATION,
    PACKAGE_TYPE_KITCHEN_REPAIR,
    PACKAGE_TYPE_KITCHEN_TURNOVER,
    PACKAGE_TYPE_BATHROOM_MODERNIZATION,
    PACKAGE_TYPE_BATHROOM_REPAIR,
    PACKAGE_TYPE_BATHROOM_TURNOVER,
    PACKAGE_TYPE_BEDROOM_MODERNIZATION,
    PACKAGE_TYPE_BEDROOM_REPAIR,
    PACKAGE_TYPE_BEDROOM_TURNOVER,
    PACKAGE_TYPE_LIVING_MODERNIZATION,
    PACKAGE_TYPE_LIVING_REPAIR,
    PACKAGE_TYPE_LIVING_TURNOVER,
    PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH,
})

PACKAGE_CATEGORY_MODERNIZATION = "modernization"
PACKAGE_CATEGORY_REPAIR = "repair"
PACKAGE_CATEGORY_TURNOVER = "turnover"
PACKAGE_CATEGORY_INSPECTION_RISK = "inspection_risk"
VALID_PACKAGE_CATEGORIES = frozenset({
    PACKAGE_CATEGORY_MODERNIZATION,
    PACKAGE_CATEGORY_REPAIR,
    PACKAGE_CATEGORY_TURNOVER,
    PACKAGE_CATEGORY_INSPECTION_RISK,
})

ROOM_KITCHEN = "kitchen"
ROOM_BATHROOM = "bathroom"
ROOM_BEDROOM = "bedroom"
ROOM_LIVING = "living"
ROOM_EXTERIOR = "exterior"
ROOM_WHOLE_HOME = "whole_home"
VALID_ROOMS = frozenset({
    ROOM_KITCHEN,
    ROOM_BATHROOM,
    ROOM_BEDROOM,
    ROOM_LIVING,
    ROOM_EXTERIOR,
    ROOM_WHOLE_HOME,
})


# UI group token -> package room constant. Only groups that own a per-room
# package appear here; utility/other deliberately map to no room.
_GROUP_TO_ROOM: Dict[str, str] = {
    "kitchen": ROOM_KITCHEN,
    "bathroom": ROOM_BATHROOM,
    "bedroom": ROOM_BEDROOM,
    "living_areas": ROOM_LIVING,
    "exterior": ROOM_EXTERIOR,
}

# Transitional living-area scenes that must NOT drive a living package even
# though they belong to the living_areas group (product decision).
_SCENE_ROOM_EXCLUSIONS = frozenset({"hallway", "stairway"})


def _normalize_scene_to_room(scene: str) -> str:
    """Map a scene id, UI scene group, or room constant to its package room.

    Three vocabularies feed package routing and must collapse consistently:
    Pass-1a/surrogate scene ids ("living_room", "dining_room"), UI/retrieval
    scene groups ("living_areas"), and package room constants ("living").
    Resolution is group-aware via ``SCENE_TO_GROUP_UI`` so every living-area
    scene (living_room, dining_room, home_office) collapses to ROOM_LIVING and
    "pantry" collapses to ROOM_KITCHEN — keeping both the affinity lookup and the
    scene-mismatch guard in ``infer_package_candidates`` aligned. Bedroom is
    already consistent ("bedroom" == "bedroom").

    ``hallway``/``stairway`` are excluded: as transitional spaces they don't
    drive a living package. They are non-breaking scenes, so they never open a
    surrogate and never reach the primary path as their own scene id; the
    exclusion only bites if a literal scene id is normalized. The fallback path
    sees the group token "living_areas" (no scene-level detail), so a truly
    surrogate-less hallway issue would still map to living — acceptable since the
    pipeline stamps a surrogate on every issue.
    """
    scene_norm = str(scene or "").strip().lower()
    if not scene_norm:
        return scene_norm
    if scene_norm in VALID_ROOMS:
        return scene_norm
    if scene_norm in _SCENE_ROOM_EXCLUSIONS:
        return scene_norm
    if scene_norm in _GROUP_TO_ROOM:
        return _GROUP_TO_ROOM[scene_norm]
    group = SCENE_TO_GROUP_UI.get(scene_norm)
    if group in _GROUP_TO_ROOM:
        return _GROUP_TO_ROOM[group]
    return scene_norm

PACKAGE_LEVEL_PROPERTY = "property"
PACKAGE_LEVEL_ROOM = "room"
PACKAGE_LEVEL_COMPONENT = "component"
PACKAGE_LEVEL_SYSTEM = "system"
VALID_PACKAGE_LEVELS = frozenset({
    PACKAGE_LEVEL_PROPERTY,
    PACKAGE_LEVEL_ROOM,
    PACKAGE_LEVEL_COMPONENT,
    PACKAGE_LEVEL_SYSTEM,
})

PACKAGE_STRENGTH_STRONG = "strong"
PACKAGE_STRENGTH_MODERATE = "moderate"
PACKAGE_STRENGTH_WEAK = "weak"
VALID_PACKAGE_STRENGTHS = frozenset({
    PACKAGE_STRENGTH_STRONG,
    PACKAGE_STRENGTH_MODERATE,
    PACKAGE_STRENGTH_WEAK,
})
VALID_EMITTED_PACKAGE_STRENGTHS = frozenset({
    PACKAGE_STRENGTH_STRONG,
    PACKAGE_STRENGTH_MODERATE,
})

PACKAGE_VERIFICATION_CONFIRMED = "confirmed"
PACKAGE_VERIFICATION_CONFIRMED_BY_RULE = "confirmed_by_rule"
PACKAGE_VERIFICATION_REJECTED = "rejected"
PACKAGE_VERIFICATION_UNCERTAIN = "uncertain"
PACKAGE_VERIFICATION_NOT_RUN = "not_run"
VALID_PACKAGE_VERIFICATION_STATUSES = frozenset({
    PACKAGE_VERIFICATION_CONFIRMED,
    PACKAGE_VERIFICATION_CONFIRMED_BY_RULE,
    PACKAGE_VERIFICATION_REJECTED,
    PACKAGE_VERIFICATION_UNCERTAIN,
    PACKAGE_VERIFICATION_NOT_RUN,
})
ACTIVE_PACKAGE_STATUSES = frozenset({
    PACKAGE_VERIFICATION_CONFIRMED,
    PACKAGE_VERIFICATION_CONFIRMED_BY_RULE,
})

_PACKAGE_ABSORPTION_SCOPES: Dict[str, Dict[str, Any]] = {
    "kitchen_minor_repair": {
        "family": "kitchen",
        "groups": {"kitchen"},
        "trade_buckets": {"kitchen_cabinets_counters", "paint_drywall"},
        "components": {"cabinets", "counter", "kitchen_finish", "appliance", "paint"},
    },
    "kitchen_refresh": {
        "family": "kitchen",
        "groups": {"kitchen", "flooring"},
        "trade_buckets": {"kitchen_cabinets_counters", "flooring", "paint_drywall"},
        "components": {"cabinets", "counter", "kitchen_finish", "appliance", "flooring", "paint"},
    },
    "kitchen_partial_rehab": {
        "family": "kitchen",
        "groups": {"kitchen", "flooring"},
        "trade_buckets": {"kitchen_cabinets_counters", "flooring", "paint_drywall"},
        "components": {"cabinets", "counter", "kitchen_finish", "appliance", "flooring", "paint"},
    },
    "kitchen_full_rehab": {
        "family": "kitchen",
        "groups": {"kitchen", "flooring"},
        "trade_buckets": {"kitchen_cabinets_counters", "flooring", "paint_drywall"},
        "components": {"cabinets", "counter", "kitchen_finish", "appliance", "flooring", "paint"},
    },
    "kitchen_repair_light": {
        "family": "kitchen",
        "groups": {"kitchen"},
        "trade_buckets": {"kitchen_cabinets_counters", "plumbing", "electrical", "moisture_mold"},
        "components": {"cabinets", "appliance", "plumbing", "electrical_heavy", "electrical_light", "moisture"},
    },
    "kitchen_repair_heavy": {
        "family": "kitchen",
        "groups": {"kitchen", "flooring"},
        "trade_buckets": {"kitchen_cabinets_counters", "plumbing", "electrical", "moisture_mold", "flooring"},
        "components": {"cabinets", "appliance", "plumbing", "electrical_heavy", "electrical_light", "moisture", "flooring"},
    },
    "kitchen_turnover_light": {
        "family": "kitchen",
        "groups": {"kitchen"},
        "trade_buckets": {"paint_drywall", "cleaning_turnover"},
        "components": {"paint", "kitchen_finish"},
    },
    "kitchen_turnover_std": {
        "family": "kitchen",
        "groups": {"kitchen", "flooring"},
        "trade_buckets": {"paint_drywall", "flooring", "cleaning_turnover"},
        "components": {"paint", "kitchen_finish", "flooring"},
    },
    "bathroom_minor_repair": {
        "family": "bathroom",
        "groups": {"bathroom"},
        "trade_buckets": {"bathroom_fixtures_tile", "paint_drywall"},
        "components": {"vanity", "tile", "tub_shower", "fixture", "bath_finish", "paint"},
    },
    "bathroom_refresh": {
        "family": "bathroom",
        "groups": {"bathroom", "flooring"},
        "trade_buckets": {"bathroom_fixtures_tile", "flooring", "paint_drywall"},
        "components": {"vanity", "tile", "tub_shower", "fixture", "bath_finish", "flooring", "paint"},
    },
    "bathroom_partial_rehab": {
        "family": "bathroom",
        "groups": {"bathroom", "flooring"},
        "trade_buckets": {"bathroom_fixtures_tile", "flooring", "paint_drywall"},
        "components": {"vanity", "tile", "tub_shower", "fixture", "bath_finish", "flooring", "paint"},
    },
    "bathroom_full_rehab": {
        "family": "bathroom",
        "groups": {"bathroom", "flooring"},
        "trade_buckets": {"bathroom_fixtures_tile", "flooring", "paint_drywall"},
        "components": {"vanity", "tile", "tub_shower", "fixture", "bath_finish", "flooring", "paint"},
    },
    "bathroom_repair_light": {
        "family": "bathroom",
        "groups": {"bathroom"},
        "trade_buckets": {"bathroom_fixtures_tile", "plumbing", "electrical", "moisture_mold"},
        "components": {"vanity", "tile", "tub_shower", "fixture", "plumbing", "electrical_heavy", "electrical_light", "moisture"},
    },
    "bathroom_repair_heavy": {
        "family": "bathroom",
        "groups": {"bathroom", "flooring"},
        "trade_buckets": {"bathroom_fixtures_tile", "plumbing", "electrical", "moisture_mold", "flooring"},
        "components": {"vanity", "tile", "tub_shower", "fixture", "plumbing", "electrical_heavy", "electrical_light", "moisture", "flooring"},
    },
    "bathroom_turnover_light": {
        "family": "bathroom",
        "groups": {"bathroom"},
        "trade_buckets": {"paint_drywall", "cleaning_turnover"},
        "components": {"paint", "bath_finish"},
    },
    "bathroom_turnover_std": {
        "family": "bathroom",
        "groups": {"bathroom", "flooring"},
        "trade_buckets": {"paint_drywall", "flooring", "cleaning_turnover"},
        "components": {"paint", "bath_finish", "flooring"},
    },
    # Bedroom / living reuse the generic component classes (flooring, paint,
    # electrical_light, moisture) — they have no cabinets/tile/vanity analogue.
    # `groups` matches the catalog estimate.group (== room) so reconciliation joins.
    "bedroom_refresh": {
        "family": "bedroom",
        "groups": {"bedroom", "flooring"},
        "trade_buckets": {"flooring", "paint_drywall", "electrical"},
        "components": {"flooring", "paint", "electrical_light"},
    },
    "bedroom_full_rehab": {
        "family": "bedroom",
        "groups": {"bedroom", "flooring"},
        "trade_buckets": {"flooring", "paint_drywall", "electrical", "moisture_mold"},
        "components": {"flooring", "paint", "electrical_light", "moisture"},
    },
    "bedroom_repair_light": {
        "family": "bedroom",
        "groups": {"bedroom"},
        "trade_buckets": {"paint_drywall", "electrical", "moisture_mold"},
        "components": {"paint", "electrical_light", "moisture"},
    },
    "bedroom_repair_heavy": {
        "family": "bedroom",
        "groups": {"bedroom", "flooring"},
        "trade_buckets": {"paint_drywall", "electrical", "moisture_mold", "flooring"},
        "components": {"paint", "electrical_light", "moisture", "flooring"},
    },
    "bedroom_turnover_light": {
        "family": "bedroom",
        "groups": {"bedroom"},
        "trade_buckets": {"paint_drywall", "cleaning_turnover"},
        "components": {"paint"},
    },
    "bedroom_turnover_std": {
        "family": "bedroom",
        "groups": {"bedroom", "flooring"},
        "trade_buckets": {"paint_drywall", "flooring", "cleaning_turnover"},
        "components": {"paint", "flooring"},
    },
    "living_refresh": {
        "family": "living",
        "groups": {"living", "flooring"},
        "trade_buckets": {"flooring", "paint_drywall", "electrical"},
        "components": {"flooring", "paint", "electrical_light"},
    },
    "living_full_rehab": {
        "family": "living",
        "groups": {"living", "flooring"},
        "trade_buckets": {"flooring", "paint_drywall", "electrical", "moisture_mold"},
        "components": {"flooring", "paint", "electrical_light", "moisture"},
    },
    "living_repair_light": {
        "family": "living",
        "groups": {"living"},
        "trade_buckets": {"paint_drywall", "electrical", "moisture_mold"},
        "components": {"paint", "electrical_light", "moisture"},
    },
    "living_repair_heavy": {
        "family": "living",
        "groups": {"living", "flooring"},
        "trade_buckets": {"paint_drywall", "electrical", "moisture_mold", "flooring"},
        "components": {"paint", "electrical_light", "moisture", "flooring"},
    },
    "living_turnover_light": {
        "family": "living",
        "groups": {"living"},
        "trade_buckets": {"paint_drywall", "cleaning_turnover"},
        "components": {"paint"},
    },
    "living_turnover_std": {
        "family": "living",
        "groups": {"living", "flooring"},
        "trade_buckets": {"paint_drywall", "flooring", "cleaning_turnover"},
        "components": {"paint", "flooring"},
    },
    "room_refresh": {
        "family": "room",
        "groups": {"other", "flooring", "paint_drywall"},
        "trade_buckets": {"flooring", "paint_drywall", "cleaning_turnover"},
        "components": {"flooring", "paint"},
    },
    "room_repair_heavy": {
        "family": "room",
        "groups": {"other", "flooring", "paint_drywall"},
        "trade_buckets": {"flooring", "paint_drywall", "cleaning_turnover"},
        "components": {"flooring", "paint"},
    },
}

_BROAD_ABSORPTION_BLOCKED_GROUPS = {
    "roof",
    "roof_gutters",
    "structure",
    "foundation",
    "foundation_structure",
    "pool",
    "hvac",
    "electrical",
    "plumbing",
    "exterior",
    "exterior_siding_trim",
    "masonry_exterior_structure",
}

_BROAD_ABSORPTION_BLOCKED_TRADES = {
    "roof_gutters",
    "foundation_structure",
    "pool",
    "hvac",
    "electrical",
    "plumbing",
    "exterior_siding_trim",
    "masonry_exterior_structure",
}


def _effective_posture(candidate: EstimateCandidate) -> Optional[str]:
    """Return effective_posture if set, else fall back to catalog strategy.

    `compute_renovation_estimate` only populates `effective_posture` on its own
    resolved-candidate list, so candidates passed straight into
    `infer_package_candidates` may still have `effective_posture is None`. Falling
    back to the catalog strategy mirrors v3's `resolve_effective_posture`
    keep-default semantics.
    """
    if candidate.effective_posture:
        return candidate.effective_posture
    return candidate.estimate_meta.strategy


def catalog_display_class(catalog_item: Dict[str, Any]) -> str:
    """Return the UI display class for a catalog item."""
    raw = str(catalog_item.get("display_class") or "").strip().lower()
    if raw in VALID_DISPLAY_CLASSES:
        return raw
    if catalog_item.get("defaultHidden") or catalog_item.get("drop_if_generic"):
        return DISPLAY_CLASS_HIDDEN
    name_id = " ".join([
        str(catalog_item.get("id") or ""),
        str(catalog_item.get("name") or ""),
        str(catalog_item.get("category") or ""),
    ]).lower()
    if any(token in name_id for token in ("clutter", "staging", "furniture")):
        return DISPLAY_CLASS_CLUTTER
    if str(catalog_item.get("kind") or "").lower() == "upgrade":
        return DISPLAY_CLASS_MARKETABILITY
    trade = str(catalog_item.get("trade_bucket") or "")
    category = str(catalog_item.get("category") or "")
    if trade in {
        "foundation_structure",
        "electrical",
        "plumbing",
        "moisture_mold",
        "hvac",
    } or category in {"safety", "structure", "health_safety", "moisture"}:
        return DISPLAY_CLASS_HIGH_CONCERN
    estimate = catalog_item.get("estimate") or {}
    if isinstance(estimate, dict) and estimate.get("estimate_tier") in {"high", "medium"}:
        return DISPLAY_CLASS_ESTIMATE_DRIVER
    return DISPLAY_CLASS_MARKETABILITY


def catalog_package_role(catalog_item: Dict[str, Any]) -> str:
    raw = str(catalog_item.get("package_role") or "").strip().lower()
    if raw in VALID_PACKAGE_ROLES:
        return raw
    return PACKAGE_ROLE_IGNORE


def catalog_package_type(catalog_item: Dict[str, Any]) -> Optional[str]:
    raw = str(catalog_item.get("package_type") or "").strip().lower()
    if raw in VALID_PACKAGE_TYPES:
        return raw
    return None


# Package-type → default category fallback (used until every kitchen entry is
# explicitly re-tagged with `package_category`). Modernization is the legacy
# default for kitchen_modernization data already on disk.
_PACKAGE_TYPE_TO_CATEGORY = {
    PACKAGE_TYPE_KITCHEN_MODERNIZATION: PACKAGE_CATEGORY_MODERNIZATION,
    PACKAGE_TYPE_KITCHEN_REPAIR: PACKAGE_CATEGORY_REPAIR,
    PACKAGE_TYPE_KITCHEN_TURNOVER: PACKAGE_CATEGORY_TURNOVER,
    PACKAGE_TYPE_BATHROOM_MODERNIZATION: PACKAGE_CATEGORY_MODERNIZATION,
    PACKAGE_TYPE_BATHROOM_REPAIR: PACKAGE_CATEGORY_REPAIR,
    PACKAGE_TYPE_BATHROOM_TURNOVER: PACKAGE_CATEGORY_TURNOVER,
    PACKAGE_TYPE_BEDROOM_MODERNIZATION: PACKAGE_CATEGORY_MODERNIZATION,
    PACKAGE_TYPE_BEDROOM_REPAIR: PACKAGE_CATEGORY_REPAIR,
    PACKAGE_TYPE_BEDROOM_TURNOVER: PACKAGE_CATEGORY_TURNOVER,
    PACKAGE_TYPE_LIVING_MODERNIZATION: PACKAGE_CATEGORY_MODERNIZATION,
    PACKAGE_TYPE_LIVING_REPAIR: PACKAGE_CATEGORY_REPAIR,
    PACKAGE_TYPE_LIVING_TURNOVER: PACKAGE_CATEGORY_TURNOVER,
    PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH: PACKAGE_CATEGORY_TURNOVER,
}

_PACKAGE_TYPE_TO_ROOM = {
    PACKAGE_TYPE_KITCHEN_MODERNIZATION: ROOM_KITCHEN,
    PACKAGE_TYPE_KITCHEN_REPAIR: ROOM_KITCHEN,
    PACKAGE_TYPE_KITCHEN_TURNOVER: ROOM_KITCHEN,
    PACKAGE_TYPE_BATHROOM_MODERNIZATION: ROOM_BATHROOM,
    PACKAGE_TYPE_BATHROOM_REPAIR: ROOM_BATHROOM,
    PACKAGE_TYPE_BATHROOM_TURNOVER: ROOM_BATHROOM,
    PACKAGE_TYPE_BEDROOM_MODERNIZATION: ROOM_BEDROOM,
    PACKAGE_TYPE_BEDROOM_REPAIR: ROOM_BEDROOM,
    PACKAGE_TYPE_BEDROOM_TURNOVER: ROOM_BEDROOM,
    PACKAGE_TYPE_LIVING_MODERNIZATION: ROOM_LIVING,
    PACKAGE_TYPE_LIVING_REPAIR: ROOM_LIVING,
    PACKAGE_TYPE_LIVING_TURNOVER: ROOM_LIVING,
    PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH: ROOM_WHOLE_HOME,
}


# ── Catalog-driven package affinity ──────────────────────────────────────────
# Routing ("which issue, seen in which room, feeds which package, in which
# role") lives in the catalog: each routable item carries a `package_affinity`
# object keyed by package room (VALID_ROOMS vocabulary — "living", not the
# "living_areas" scene-group token). Entries store only the non-derivable
# fields (package_type, package_role); package_category and room derive from
# package_type via the vocabulary maps above. Items without a block never
# route to a package.
#
# Generic-issue design note (kitchen/bathroom + cosmetic gaps): generics route
# to a room's MODERNIZATION package as SUPPORT — the true drivers in those
# rooms are cabinets/vanity, so a lone generic stays corroboration-dependent
# and never over-buckets (no generic acts as a kitchen/bathroom driver).
# Bedroom/living carry explicit driver entries where flooring/drywall act as
# drivers. Bathrooms have dedicated items for flooring/paint/wallpaper/lighting
# (dated_bathroom_wallpaper, bathroom_paint_refresh_recommended, …), so
# generics route to a bathroom package ONLY for the "dry construction" gap
# categories that lack a bathroom-specific equivalent. The three
# structural/feature generics (unfinished_interior_wall_osb_exposed,
# stained_glass_or_vintage_light_fixture, layout_modernization_opportunity)
# are intentionally unrouted: they stay line-item / observation only.

def build_package_affinity(
    issue_catalog: Dict[str, Any],
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Flatten per-item ``package_affinity`` blocks into the runtime table.

    Returns {(room, issue_id): {package_type, package_role, package_category,
    room}} with category/room derived from package_type. The catalog is the
    single source of truth for routing, so validation raises instead of
    silently skipping malformed entries.
    """
    table: Dict[Tuple[str, str], Dict[str, str]] = {}
    for item in (issue_catalog or {}).get("items") or []:
        if not isinstance(item, dict):
            continue
        block = item.get("package_affinity")
        if block is None:
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            raise ValueError("package_affinity block on a catalog item without an id")
        if not isinstance(block, dict):
            raise ValueError(
                f"{item_id}: package_affinity must be an object keyed by package room"
            )
        for room_key, entry in block.items():
            room = str(room_key or "").strip().lower()
            if room not in VALID_ROOMS:
                raise ValueError(
                    f"{item_id}: package_affinity room {room_key!r} is not a valid package room"
                )
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{item_id}/{room}: package_affinity entry must be an object"
                )
            package_type = str(entry.get("package_type") or "").strip().lower()
            package_role = str(entry.get("package_role") or "").strip().lower()
            if package_type not in VALID_PACKAGE_TYPES:
                raise ValueError(
                    f"{item_id}/{room}: unknown package_type {entry.get('package_type')!r}"
                )
            if package_type == PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH:
                raise ValueError(
                    f"{item_id}/{room}: {package_type} is derived downstream from "
                    "per-room turnover packages and must never be referenced by a "
                    "catalog item"
                )
            if _PACKAGE_TYPE_TO_ROOM.get(package_type) != room:
                raise ValueError(
                    f"{item_id}/{room}: package_type {package_type!r} belongs to room "
                    f"{_PACKAGE_TYPE_TO_ROOM.get(package_type)!r}, not {room!r}"
                )
            if package_role not in (PACKAGE_ROLE_DRIVER, PACKAGE_ROLE_SUPPORT):
                raise ValueError(
                    f"{item_id}/{room}: package_role must be {PACKAGE_ROLE_DRIVER!r} "
                    f"or {PACKAGE_ROLE_SUPPORT!r}, got {entry.get('package_role')!r}"
                )
            table[(room, item_id)] = {
                "package_type": package_type,
                "package_role": package_role,
                "package_category": _PACKAGE_TYPE_TO_CATEGORY[package_type],
                "room": room,
            }
    return table


_default_affinity_table: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None


def _default_package_affinity() -> Dict[Tuple[str, str], Dict[str, str]]:
    """Affinity table from the shipped catalog, built once on first use."""
    global _default_affinity_table
    if _default_affinity_table is None:
        catalog_path = Path(__file__).resolve().parent / "issue_catalog.json"
        _default_affinity_table = build_package_affinity(
            json.loads(catalog_path.read_text(encoding="utf-8"))
        )
    return _default_affinity_table


def catalog_package_category(catalog_item: Dict[str, Any]) -> Optional[str]:
    raw = str(catalog_item.get("package_category") or "").strip().lower()
    if raw in VALID_PACKAGE_CATEGORIES:
        return raw
    package_type = catalog_package_type(catalog_item)
    if package_type:
        return _PACKAGE_TYPE_TO_CATEGORY.get(package_type)
    return None


def catalog_room(catalog_item: Dict[str, Any]) -> Optional[str]:
    raw = str(catalog_item.get("room") or "").strip().lower()
    if raw in VALID_ROOMS:
        return raw
    package_type = catalog_package_type(catalog_item)
    if package_type:
        room = _PACKAGE_TYPE_TO_ROOM.get(package_type)
        if room:
            return room
    scene_groups = catalog_item.get("scene_groups") or ()
    if isinstance(scene_groups, (list, tuple)):
        for scene in scene_groups:
            scene_norm = str(scene or "").strip().lower()
            if scene_norm in VALID_ROOMS:
                return scene_norm
    return None


def package_affinity_for(
    scene_group: str,
    issue_id: str,
    table: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
) -> Optional[Dict[str, str]]:
    """Resolve the affinity entry for an issue observed in a scene.

    Exact (room, issue_id) hits win. When the scene does not resolve to a
    package room (no surrogate, "other", hallway/stairway) and the issue's
    affinity block routes to exactly one room, that entry is returned —
    preserving the legacy flat-field behavior where a kitchen/bathroom item
    routed without a resolvable scene. Multi-room blocks stay unrouted without
    a scene; the scene-mismatch guard in infer_package_candidates still
    protects this fallback when a contradicting surrogate scene exists.
    """
    if table is None:
        table = _default_package_affinity()
    issue = str(issue_id or "").strip()
    if not issue:
        return None
    scene = _normalize_scene_to_room(scene_group)
    if scene in VALID_ROOMS:
        return table.get((scene, issue))
    rooms = [room for (room, table_issue) in table if table_issue == issue]
    if len(rooms) == 1:
        return table[(rooms[0], issue)]
    return None


def _candidate_scene_room(
    candidate: EstimateCandidate,
    scene_by_room: Dict[str, str],
) -> Optional[str]:
    room_surrogate_id = str(getattr(candidate, "room_surrogate_id", "") or "")
    if room_surrogate_id and room_surrogate_id in scene_by_room:
        return scene_by_room[room_surrogate_id]
    for scene in getattr(candidate, "scene_groups_seen", []) or []:
        scene_norm = _normalize_scene_to_room(str(scene or ""))
        if scene_norm:
            return scene_norm
    return None


def _catalog_item_with_package_affinity(
    catalog_item: Dict[str, Any],
    candidate: EstimateCandidate,
    scene_by_room: Dict[str, str],
    affinity_table: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
) -> Dict[str, Any]:
    scene_room = _candidate_scene_room(candidate, scene_by_room)
    affinity = package_affinity_for(
        scene_room or "",
        getattr(candidate, "catalog_item_id", "") or catalog_item.get("id") or "",
        affinity_table,
    )
    if not affinity:
        return catalog_item
    effective = dict(catalog_item)
    effective.update(affinity)
    effective["_package_affinity_scene"] = (
        scene_room
        if scene_room == affinity["room"]
        else f"single_room_fallback:{affinity['room']}"
    )
    return effective


def is_package_eligible_catalog_item(catalog_item: Dict[str, Any]) -> bool:
    """True when a catalog item should use package-level verification."""
    return (
        catalog_package_role(catalog_item) in {PACKAGE_ROLE_DRIVER, PACKAGE_ROLE_SUPPORT}
        and catalog_package_type(catalog_item) in VALID_PACKAGE_TYPES
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _split_integer(amount: int, n: int) -> List[int]:
    """Split amount into n integers summing exactly to amount.

    Distributes any remainder one unit at a time to the first k entries.
    Example: _split_integer(100, 3) == [34, 33, 33]; sum == 100.
    """
    if n <= 0:
        return []
    base, remainder = divmod(amount, n)
    return [base + 1 if i < remainder else base for i in range(n)]


# Issue 3: split the electrical bucket so a missing GFCI / exhaust fan doesn't
# trigger REPAIR_HEAVY alongside true heavy work (panel, rewire, visible-risk).
# Catalog audit (trade_bucket=electrical):
#   HEAVY:  visible_electrical_risks (sev 4, systems)
#   LIGHT:  GFCI, exhaust fan, lighting fixtures, outlets, ceiling-fan style
# When new panel-work or rewire IDs are added to the catalog, list them here.
_ELECTRICAL_HEAVY_IDS = frozenset({
    "visible_electrical_risks",
})


def classify_component(candidate: EstimateCandidate) -> Optional[str]:
    """Map a candidate to a component class for package inference.

    Returns one of: cabinets, counter, kitchen_finish, appliance, vanity, tile,
    tub_shower, fixture, bath_finish, flooring, plumbing, electrical_heavy,
    electrical_light, moisture, paint — or None when not relevant for any
    package rule.
    """
    if isinstance(candidate, dict):
        cat_id = str(candidate.get("catalog_item_id") or "").lower()
        trade = candidate.get("trade_bucket") or ""
    else:
        cat_id = (candidate.catalog_item_id or "").lower()
        trade = candidate.trade_bucket or ""

    if trade == "kitchen_cabinets_counters":
        if "appliance" in cat_id:
            return "appliance"
        if "cabinet" in cat_id:
            return "cabinets"
        if "counter" in cat_id:
            return "counter"
        return "kitchen_finish"

    if trade == "bathroom_fixtures_tile":
        if "vanity" in cat_id:
            return "vanity"
        if "tile" in cat_id or "grout" in cat_id:
            return "tile"
        if "tub" in cat_id or "shower" in cat_id:
            return "tub_shower"
        if "fixture" in cat_id or "faucet" in cat_id:
            return "fixture"
        return "bath_finish"

    if trade == "flooring":
        return "flooring"
    if trade == "plumbing":
        return "plumbing"
    if trade == "electrical":
        return "electrical_heavy" if cat_id in _ELECTRICAL_HEAVY_IDS else "electrical_light"
    if trade == "moisture_mold":
        return "moisture"
    if trade == "paint_drywall":
        return "paint"
    return None


def _package_absorption_scope(package_type: str) -> Dict[str, Any]:
    raw = _PACKAGE_ABSORPTION_SCOPES.get(package_type) or {}
    return {
        "family": raw.get("family") or "",
        "groups": sorted(raw.get("groups") or []),
        "trade_buckets": sorted(raw.get("trade_buckets") or []),
        "components": sorted(raw.get("components") or []),
    }


# Strong-signal sentinels: explicit override list for catalog items that should
# elevate strength even when they don't meet the generic rule below (e.g. sev-2
# items with visible-substrate severity). Kept as an exception channel.
# Note: the 4 items listed already satisfy the generic rule and could be removed
# from the frozenset without behavior change — they remain here for documentation
# and as a stable contract for sentinel-driven elevation.
_STRONG_SIGNAL_CATALOG_IDS = frozenset({
    "missing_base_cabinets_exposed_subfloor",
    "missing_bathroom_tile_exposed_substrate",
    "active_water_damage_bathroom",
    "missing_vanity_exposed_plumbing",
})


# A support whose catalog id appears in support role across this many distinct
# estimate units is "ambient" — a property-wide trait (e.g. popcorn ceilings)
# rather than a room-scoped finding — and must not be the marginal vote that
# mints a driverless package. Tunable; see HANDOFF_ambient_support_catalog_tagging
# and the N=2 vs N=3 audit. Scope is package-existence only: ambient supports
# still corroborate driver-anchored packages and are still costed.
_AMBIENT_SUPPORT_MIN_UNITS = 3


def _effective_candidate_package_role(
    candidate: EstimateCandidate,
    candidate_catalog_meta: Optional[Dict[int, Dict[str, Any]]] = None,
) -> str:
    cat_item = (candidate_catalog_meta or {}).get(id(candidate))
    if cat_item:
        return catalog_package_role(cat_item)
    raw = str(getattr(candidate, "package_role", None) or "").strip().lower()
    if raw in VALID_PACKAGE_ROLES:
        return raw
    return PACKAGE_ROLE_IGNORE


def _has_strong_signal(
    candidates: List[EstimateCandidate],
    candidate_catalog_meta: Optional[Dict[int, Dict[str, Any]]] = None,
) -> bool:
    """Return True if any candidate signals strong-strength elevation.

    Two paths:
      1. Explicit override: catalog_item_id in _STRONG_SIGNAL_CATALOG_IDS.
      2. Generic rule: severity >= 3 AND package_role == "package_driver".
         Covers outdated_kitchen_finishes / outdated_bathroom_finishes (upgrade-
         kind drivers) and outdated_or_damaged_vanity (defect-kind driver)
         without per-id maintenance. Replaces the previous hardcoded checks.
    """
    for candidate in candidates or []:
        cat_id = candidate.catalog_item_id or ""
        if cat_id in _STRONG_SIGNAL_CATALOG_IDS:
            return True
        if (
            (candidate.severity or 0) >= 3
            and _effective_candidate_package_role(
                candidate, candidate_catalog_meta
            ) == PACKAGE_ROLE_DRIVER
        ):
            return True
    return False


def compute_package_strength(
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
    candidate_catalog_meta: Optional[Dict[int, Dict[str, Any]]] = None,
) -> str:
    """Classify package strength per the user's threshold rules.

    Strong:   >=2 distinct driver catalog IDs, OR 1 driver + >=2 supports,
              OR any candidate matches a strong-signal sentinel.
    Moderate: any single driver (with or without one support), OR
              >=2 supports same estimate_unit.
    Weak:    a single orphan support (or empty). Caller should suppress.
    """
    if (
        _has_strong_signal(drivers, candidate_catalog_meta)
        or _has_strong_signal(supports, candidate_catalog_meta)
    ):
        return PACKAGE_STRENGTH_STRONG

    distinct_driver_ids = {c.catalog_item_id for c in drivers if c.catalog_item_id}
    if len(distinct_driver_ids) >= 2:
        return PACKAGE_STRENGTH_STRONG
    if drivers and len(supports) >= 2:
        return PACKAGE_STRENGTH_STRONG

    if drivers:
        # Any catalog-tagged driver — even alone — meets the moderate bar.
        # Isolated supports do not, to honor the user's "weak signal" rule.
        return PACKAGE_STRENGTH_MODERATE
    if len(supports) >= 2:
        return PACKAGE_STRENGTH_MODERATE

    return PACKAGE_STRENGTH_WEAK


def compute_initial_confidence_score(strength: str) -> float:
    """Prior confidence score before Pass 2f verification."""
    if strength == PACKAGE_STRENGTH_STRONG:
        return 0.30
    if strength == PACKAGE_STRENGTH_MODERATE:
        return 0.15
    return 0.0


def compute_post_verification_score(
    initial_score: float,
    confirmed_count: int,
    rejected_count: int,
    supporting_count: int,
) -> float:
    """Update confidence after Pass 2f returns confirmed/rejected counts."""
    denom = max(int(supporting_count or 0), 1)
    bonus = 0.5 * (int(confirmed_count or 0) / denom)
    penalty = 0.2 * (int(rejected_count or 0) / denom)
    score = float(initial_score) + bonus - penalty
    return max(0.0, min(1.0, score))


def _package_child_absorption_reason(
    pkg: Dict[str, Any],
    child: Dict[str, Any],
    supporting_issue_ids: set,
    *,
    allow_broad: bool,
) -> Optional[str]:
    if child.get("cost_model") == INSPECTION_ALLOWANCE:
        return None

    # Required-scope work is never absorbed into a marketability package — for
    # ANY reason, including an exact supporting-issue match. Absorbing it would
    # move the dollars out of required_rehab into the marketability bundle
    # (the tradeoff_note invariant). It stays a retained line item instead.
    if (
        pkg.get("estimate_scope") == MARKETABILITY_REHAB
        and child.get("estimate_scope") == REQUIRED_REHAB
    ):
        return None

    child_issue_ids = set(child.get("issue_ids") or [])
    if child_issue_ids & supporting_issue_ids:
        return "supporting_issue"

    if not allow_broad:
        return None

    if not _package_scope_covers_child(pkg, child):
        return None

    cost_model = child.get("cost_model") or LINE_ITEM
    if cost_model == ROOM_ALLOWANCE:
        return "same_unit_room_allowance_scope"
    if child.get("estimate_scope") == MARKETABILITY_REHAB:
        return "same_unit_marketability_family"
    if cost_model == LINE_ITEM:
        return "same_unit_line_item_scope"
    return None


def _package_scope_covers_child(pkg: Dict[str, Any], child: Dict[str, Any]) -> bool:
    if _is_broad_absorption_blocked_child(child):
        return False

    # `absorption_scope` is guaranteed to be set by reconcile_packages_and_estimate_units
    # Phase B (raises KeyError if missing). Any package reaching this function has it.
    scope = pkg["absorption_scope"]
    groups = set(scope.get("groups") or [])
    trade_buckets = set(scope.get("trade_buckets") or [])
    components = set(scope.get("components") or [])

    group = child.get("group") or ""
    trade_bucket = child.get("trade_bucket") or ""
    component = classify_component(child)

    return (
        group in groups
        or trade_bucket in trade_buckets
        or (component is not None and component in components)
    )


def _is_broad_absorption_blocked_child(child: Dict[str, Any]) -> bool:
    group = str(child.get("group") or "")
    trade_bucket = str(child.get("trade_bucket") or "")
    return (
        group in _BROAD_ABSORPTION_BLOCKED_GROUPS
        or trade_bucket in _BROAD_ABSORPTION_BLOCKED_TRADES
    )


def _append_invalid_cost_model_warnings(
    children: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> None:
    seen = set()
    for child in children or []:
        if child.get("cost_model_source") != COST_MODEL_SOURCE_INVALID_CATALOG_FALLBACK:
            continue
        key = (
            child.get("catalog_item_id"),
            child.get("parent_line_item_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        warnings.append({
            "code": "invalid_catalog_cost_model",
            "catalog_item_id": child.get("catalog_item_id"),
            "estimate_unit_id": child.get("parent_line_item_id"),
            "defaulted_to": LINE_ITEM,
        })


def is_defect_driver(
    candidate: EstimateCandidate,
    catalog_item: Dict[str, Any],
) -> bool:
    """Real defect that can drive a partial/full rehab package."""
    if candidate.is_valid_detection is False:
        return False
    if catalog_item.get("kind") != "defect":
        return False
    if catalog_item.get("tier") == _OPTIONAL_TIER:
        return False
    if _effective_posture(candidate) not in _QUALIFIED_POSTURES:
        return False
    return True


def is_dated_cosmetic_evidence(
    candidate: EstimateCandidate,
    catalog_item: Dict[str, Any],
) -> bool:
    """Upgrade-kind cosmetic/opportunity item, used ONLY as refresh evidence."""
    if candidate.is_valid_detection is False:
        return False
    if catalog_item.get("tier") == _OPTIONAL_TIER:
        return False
    if _effective_posture(candidate) not in _QUALIFIED_POSTURES:
        return False
    if catalog_item.get("kind") != "upgrade":
        return False
    return catalog_item.get("category") in ("cosmetic", "opportunity")


def _is_defect_driver_role(catalog_item: Dict[str, Any]) -> bool:
    """A package_driver whose evidence is a concrete observable defect."""
    return (
        catalog_package_role(catalog_item) == PACKAGE_ROLE_DRIVER
        and catalog_item.get("kind") == "defect"
    )


def _is_opportunity_driver(catalog_item: Dict[str, Any]) -> bool:
    """A package_driver whose evidence is a subjective marketability judgment."""
    return (
        catalog_package_role(catalog_item) == PACKAGE_ROLE_DRIVER
        and catalog_item.get("kind") == "upgrade"
    )


def _suppress_blocked_opportunity_drivers(
    opportunity_drivers: List[EstimateCandidate],
) -> None:
    """Mark uncorroborated opportunity-driver candidates as invalid so they do
    not flow through as standalone cost lines. Mirrors the Pass 2f rejection
    pattern in apply_package_verifications_to_candidates."""
    for candidate in opportunity_drivers:
        candidate.is_valid_detection = False
        candidate.pass_2f_attempted = False
        candidate.pass_2f_applied = False
        candidate.pass_2f_fallback_reason = "insufficient_corroboration_for_opportunity_driver"
        candidate.review_source = "package_gate:opportunity_driver_uncorroborated"


# ═══════════════════════════════════════════════════════════════════════════
# Package inference
# ═══════════════════════════════════════════════════════════════════════════

def _catalog_lookup(issue_catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        item["id"]: item
        for item in (issue_catalog.get("items") or [])
        if isinstance(item, dict) and item.get("id")
    }


def _candidate_unit_id(candidate: EstimateCandidate) -> str:
    return (
        getattr(candidate, "billable_estimate_unit_id", "") or
        getattr(candidate, "estimate_unit_id", "") or
        getattr(candidate, "room_surrogate_id", "") or
        "property"
    )


def _source_room_ids_for_unit(
    unit_id: str,
    estimate_units: Optional[List[Dict[str, Any]]],
) -> List[str]:
    for unit in estimate_units or []:
        if unit.get("estimate_unit_id") == unit_id:
            return list(unit.get("source_room_surrogate_ids") or [])
    return []


def _candidate_issue_refs(candidate: EstimateCandidate) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    seen = set()
    raw_refs = getattr(candidate, "evidence_refs", []) or []
    if raw_refs:
        for raw in raw_refs:
            if not isinstance(raw, dict):
                continue
            issue_id = str(raw.get("issue_id") or "").strip()
            photo_key = str(raw.get("photo_key") or "").strip()
            if not issue_id:
                continue
            key = (issue_id, photo_key)
            if key in seen:
                continue
            seen.add(key)
            refs.append({
                "issue_id": issue_id,
                "photo_key": photo_key,
                "observation": str(raw.get("observation") or "").strip(),
                "room_surrogate_id": str(
                    raw.get("room_surrogate_id")
                    or getattr(candidate, "room_surrogate_id", "")
                    or ""
                ).strip(),
            })
        return refs

    issue_ids = [
        str(issue_id).strip()
        for issue_id in (candidate.issue_ids or [])
        if str(issue_id or "").strip()
    ]
    photo_keys = [
        str(photo_key).strip()
        for photo_key in (candidate.photo_keys or [])
        if str(photo_key or "").strip()
    ]
    observations = [
        str(obs).strip()
        for obs in (candidate.supporting_observations or [])
        if str(obs or "").strip()
    ]
    pairs: List[Tuple[str, str]] = []
    if len(issue_ids) == len(photo_keys):
        pairs = list(zip(issue_ids, photo_keys))
    elif len(issue_ids) == 1:
        pairs = [(issue_ids[0], key) for key in photo_keys or [""]]
    elif len(photo_keys) == 1:
        pairs = [(issue_id, photo_keys[0]) for issue_id in issue_ids]
    else:
        pairs = [(issue_id, "") for issue_id in issue_ids]
    for idx, (issue_id, photo_key) in enumerate(pairs):
        key = (issue_id, photo_key)
        if not issue_id or key in seen:
            continue
        seen.add(key)
        refs.append({
            "issue_id": issue_id,
            "photo_key": photo_key,
            "observation": observations[idx] if idx < len(observations) else "",
            "room_surrogate_id": str(getattr(candidate, "room_surrogate_id", "") or ""),
        })
    return refs


def _candidate_evidence_item(
    candidate: EstimateCandidate,
    catalog_item: Dict[str, Any],
) -> Dict[str, Any]:
    photo_keys = _distinct_photo_keys([candidate])
    issue_refs = _candidate_issue_refs(candidate)
    return {
        "catalog_item_id": candidate.catalog_item_id,
        "name": candidate.catalog_item_name,
        "issue_ids": list(candidate.issue_ids or []),
        "issue_refs": issue_refs,
        "observations": list(candidate.supporting_observations or []),
        "photo_keys": photo_keys,
        "supporting_photo_count": len(photo_keys),
        "corroboration_basis": (
            "multi_photo_same_issue"
            if len(photo_keys) >= 2
            else "single_photo_or_untracked"
        ),
        "scene_groups_seen": list(candidate.scene_groups_seen or []),
        "estimate_unit_id": _candidate_unit_id(candidate),
        "room_surrogate_id": candidate.room_surrogate_id,
        "display_class": catalog_display_class(catalog_item),
        "package_role": catalog_package_role(catalog_item),
        "package_type": catalog_package_type(catalog_item),
        "package_evidence_only": bool(getattr(candidate, "package_evidence_only", False)),
        "trade_bucket": candidate.trade_bucket,
        "estimate_tier": candidate.estimate_meta.estimate_tier,
    }


def _unique_in_order(values: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for value in values:
        key = str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _distinct_photo_keys(candidates: List[EstimateCandidate]) -> List[str]:
    photo_keys: List[str] = []
    for candidate in candidates or []:
        photo_keys.extend(
            str(photo_key).strip()
            for photo_key in (candidate.photo_keys or [])
            if str(photo_key or "").strip()
        )
    return _unique_in_order(photo_keys)


def _has_multiphoto_opportunity_corroboration(
    opportunity_drivers: List[EstimateCandidate],
) -> bool:
    by_catalog_id: Dict[str, List[EstimateCandidate]] = {}
    for candidate in opportunity_drivers or []:
        catalog_id = candidate.catalog_item_id or ""
        if not catalog_id:
            continue
        by_catalog_id.setdefault(catalog_id, []).append(candidate)
    return any(
        len(_distinct_photo_keys(same_catalog_drivers)) >= 2
        for same_catalog_drivers in by_catalog_id.values()
    )


def _package_corroboration_basis(
    trigger_reason: str,
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> str:
    if trigger_reason == "opportunity_driver_with_multiphoto_corroboration":
        return "multi_photo_same_issue"
    if drivers and supports:
        return "driver_plus_support"
    if len({c.catalog_item_id for c in drivers if c.catalog_item_id}) >= 2:
        return "multiple_distinct_drivers"
    if drivers:
        return "single_driver"
    if len(supports) >= 2:
        return "multiple_supports"
    return "none"


def _resolve_kitchen_modernization_profile(
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    ids = {c.catalog_item_id for c in evidence}
    components = {classify_component(c) for c in evidence}
    components.discard(None)
    notes: List[str] = []

    has_missing_cabinets = "missing_base_cabinets_exposed_subfloor" in ids
    severe_cabinet = any(
        c.catalog_item_id == "outdated_or_damaged_cabinets" and (c.severity or 0) >= 3
        for c in evidence
    )
    if has_missing_cabinets or (
        severe_cabinet
        and {"counter", "flooring"} <= components
        and components.intersection({"appliance", "plumbing"})
    ):
        notes.append("partial_or_full_driver_pattern")
        if len(components.intersection({"cabinets", "counter", "flooring", "appliance", "plumbing"})) >= 4:
            return KITCHEN_FULL_REHAB, "full_rehab", notes + ["full_rehab matched"]
        return KITCHEN_PARTIAL_REHAB, "partial_rehab", notes + ["partial_rehab matched"]

    if drivers and (supports or len(components.intersection({"cabinets", "counter", "flooring"})) >= 2):
        return KITCHEN_PARTIAL_REHAB, "partial_rehab", notes + ["driver_plus_support matched"]

    if drivers:
        return KITCHEN_REFRESH, "refresh", notes + ["single package_driver refresh"]

    return KITCHEN_REFRESH, "refresh", notes + ["multiple package_support refresh"]


def _resolve_kitchen_repair_profile(
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    """Kitchen repair pricing: light vs heavy based on component breadth."""
    components = {classify_component(c) for c in evidence}
    components.discard(None)
    notes: List[str] = []
    heavy_components = components.intersection({"plumbing", "electrical_heavy", "moisture", "flooring"})
    heavy_breadth = (
        any((d.severity or 0) >= 3 for d in drivers)
        or len(components) >= 2
        or len({d.catalog_item_id for d in drivers if d.catalog_item_id}) >= 2
    )
    if drivers and heavy_components and heavy_breadth:
        notes.append("driver_with_heavy_component")
        return KITCHEN_REPAIR_HEAVY, "repair_heavy", notes
    if len(components) >= 3:
        notes.append("multi_component_repair")
        return KITCHEN_REPAIR_HEAVY, "repair_heavy", notes
    notes.append("light_repair_default")
    return KITCHEN_REPAIR_LIGHT, "repair_light", notes


def _resolve_kitchen_turnover_profile(
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    """Kitchen turnover pricing: light (paint-only) vs std (paint + flooring)."""
    components = {classify_component(c) for c in evidence}
    components.discard(None)
    notes: List[str] = []
    if "flooring" in components and "paint" in components:
        notes.append("paint_plus_flooring")
        return KITCHEN_TURNOVER_STD, "turnover_std", notes
    if len(evidence) >= 3:
        notes.append("multi_signal_turnover")
        return KITCHEN_TURNOVER_STD, "turnover_std", notes
    notes.append("light_turnover_default")
    return KITCHEN_TURNOVER_LIGHT, "turnover_light", notes


def _resolve_bathroom_modernization_profile(
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    """Bathroom modernization pricing: refresh / partial / full based on severity + breadth."""
    components = {classify_component(c) for c in evidence}
    components.discard(None)
    notes: List[str] = []

    severe_finish_signal = any(
        c.catalog_item_id == "outdated_bathroom_finishes" and (c.severity or 0) >= 3
        for c in evidence
    )
    has_vanity_or_tile_sev3 = any(
        classify_component(c) in {"vanity", "tile"} and (c.severity or 0) >= 3
        for c in evidence
    )
    full_components = components.intersection(
        {"vanity", "tile", "flooring", "tub_shower", "fixture", "plumbing", "moisture"}
    )
    if (severe_finish_signal or has_vanity_or_tile_sev3) and {"flooring"} <= components and (
        components.intersection({"tub_shower", "fixture", "plumbing", "moisture"})
    ):
        notes.append("severe_driver_plus_flooring_plus_system")
        if len(full_components) >= 4:
            return BATHROOM_FULL_REHAB, "full_rehab", notes + ["full_rehab matched"]
        return BATHROOM_PARTIAL_REHAB, "partial_rehab", notes + ["partial_rehab matched"]

    if drivers and (supports or len(components.intersection({"vanity", "tile", "flooring"})) >= 2):
        return BATHROOM_PARTIAL_REHAB, "partial_rehab", notes + ["driver_plus_support matched"]

    if drivers:
        return BATHROOM_REFRESH, "refresh", notes + ["single package_driver refresh"]

    return BATHROOM_REFRESH, "refresh", notes + ["multiple package_support refresh"]


def _resolve_bathroom_repair_profile(
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    """Bathroom repair pricing: light vs heavy based on component breadth."""
    components = {classify_component(c) for c in evidence}
    components.discard(None)
    notes: List[str] = []
    heavy_components = components.intersection({"plumbing", "electrical_heavy", "moisture", "flooring"})
    heavy_breadth = (
        any((d.severity or 0) >= 3 for d in drivers)
        or len(components) >= 2
        or len({d.catalog_item_id for d in drivers if d.catalog_item_id}) >= 2
    )
    if drivers and heavy_components and heavy_breadth:
        notes.append("driver_with_heavy_component")
        return BATHROOM_REPAIR_HEAVY, "repair_heavy", notes
    if len(components) >= 3:
        notes.append("multi_component_repair")
        return BATHROOM_REPAIR_HEAVY, "repair_heavy", notes
    notes.append("light_repair_default")
    return BATHROOM_REPAIR_LIGHT, "repair_light", notes


def _resolve_bathroom_turnover_profile(
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    """Bathroom turnover pricing: light (paint-only) vs std (paint + flooring)."""
    components = {classify_component(c) for c in evidence}
    components.discard(None)
    notes: List[str] = []
    if "flooring" in components and "paint" in components:
        notes.append("paint_plus_flooring")
        return BATHROOM_TURNOVER_STD, "turnover_std", notes
    if len(evidence) >= 3:
        notes.append("multi_signal_turnover")
        return BATHROOM_TURNOVER_STD, "turnover_std", notes
    notes.append("light_turnover_default")
    return BATHROOM_TURNOVER_LIGHT, "turnover_light", notes


# ─── Generic room resolvers (bedroom / living) ───────────────────────────────
#
# Bedroom and living rooms share the same cosmetic component structure (flooring,
# paint, lighting, moisture) and differ only in pricing tier, so a single set of
# parameterized resolvers serves both — the dispatcher passes the room's tiers.
# Kitchen/bathroom keep their dedicated resolvers (cabinets/tile-specific logic).

def _resolve_room_modernization_profile(
    refresh_tier: Tuple[str, int, int],
    full_tier: Tuple[str, int, int],
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    """Generic room modernization: refresh by default, escalate to full_rehab
    when the cosmetic scope is broad (>=3 distinct component classes, e.g.
    flooring + paint + lighting/moisture)."""
    components = {classify_component(c) for c in evidence}
    components.discard(None)
    notes: List[str] = []
    broad = components.intersection({"flooring", "paint", "electrical_light", "moisture"})
    if drivers and len(broad) >= 3:
        notes.append("broad_cosmetic_scope")
        return full_tier, "full_rehab", notes + ["full_rehab matched"]
    if drivers:
        return refresh_tier, "refresh", notes + ["single package_driver refresh"]
    return refresh_tier, "refresh", notes + ["multiple package_support refresh"]


def _resolve_room_repair_profile(
    light_tier: Tuple[str, int, int],
    heavy_tier: Tuple[str, int, int],
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    """Generic room repair pricing: light vs heavy based on component breadth."""
    components = {classify_component(c) for c in evidence}
    components.discard(None)
    notes: List[str] = []
    heavy_components = components.intersection({"moisture", "electrical_heavy", "flooring"})
    heavy_breadth = (
        any((d.severity or 0) >= 3 for d in drivers)
        or len(components) >= 2
        or len({d.catalog_item_id for d in drivers if d.catalog_item_id}) >= 2
    )
    if drivers and heavy_components and heavy_breadth:
        notes.append("driver_with_heavy_component")
        return heavy_tier, "repair_heavy", notes
    if len(components) >= 3:
        notes.append("multi_component_repair")
        return heavy_tier, "repair_heavy", notes
    notes.append("light_repair_default")
    return light_tier, "repair_light", notes


def _resolve_room_turnover_profile(
    light_tier: Tuple[str, int, int],
    std_tier: Tuple[str, int, int],
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    """Generic room turnover pricing: light (paint-only) vs std (paint + flooring)."""
    components = {classify_component(c) for c in evidence}
    components.discard(None)
    notes: List[str] = []
    if "flooring" in components and "paint" in components:
        notes.append("paint_plus_flooring")
        return std_tier, "turnover_std", notes
    if len(evidence) >= 3:
        notes.append("multi_signal_turnover")
        return std_tier, "turnover_std", notes
    notes.append("light_turnover_default")
    return light_tier, "turnover_light", notes


def _resolve_pricing_profile(
    package_type: str,
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    """Dispatch to the right per-room profile resolver for the package_type."""
    if package_type == PACKAGE_TYPE_KITCHEN_REPAIR:
        return _resolve_kitchen_repair_profile(evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_KITCHEN_TURNOVER:
        return _resolve_kitchen_turnover_profile(evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_BATHROOM_MODERNIZATION:
        return _resolve_bathroom_modernization_profile(evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_BATHROOM_REPAIR:
        return _resolve_bathroom_repair_profile(evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_BATHROOM_TURNOVER:
        return _resolve_bathroom_turnover_profile(evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_BEDROOM_MODERNIZATION:
        return _resolve_room_modernization_profile(
            BEDROOM_REFRESH, BEDROOM_FULL_REHAB, evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_BEDROOM_REPAIR:
        return _resolve_room_repair_profile(
            BEDROOM_REPAIR_LIGHT, BEDROOM_REPAIR_HEAVY, evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_BEDROOM_TURNOVER:
        return _resolve_room_turnover_profile(
            BEDROOM_TURNOVER_LIGHT, BEDROOM_TURNOVER_STD, evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_LIVING_MODERNIZATION:
        return _resolve_room_modernization_profile(
            LIVING_REFRESH, LIVING_FULL_REHAB, evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_LIVING_REPAIR:
        return _resolve_room_repair_profile(
            LIVING_REPAIR_LIGHT, LIVING_REPAIR_HEAVY, evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_LIVING_TURNOVER:
        return _resolve_room_turnover_profile(
            LIVING_TURNOVER_LIGHT, LIVING_TURNOVER_STD, evidence, drivers, supports)
    # Default — kitchen_modernization and any future modernization-family types.
    return _resolve_kitchen_modernization_profile(evidence, drivers, supports)


# ─── Smarter tier dispatch — escalate when absorbed > tier ceiling ───────────
#
# Resolver-time tier selection picks based on component breadth & severity but
# has no view of what catalog cost the package will absorb. A sub-tier package
# absorbing a costly item would silently undercut absorbed cost (Phase C's floor
# catches it post-hoc by raising the package cost). Escalating here moves the
# fix upstream: pick a real tier whose ceiling already covers the absorbed cost.

_PRICING_TIER_ESCALATION: Dict[str, Tuple[str, int, int]] = {
    KITCHEN_REFRESH[0]: KITCHEN_PARTIAL_REHAB,
    KITCHEN_PARTIAL_REHAB[0]: KITCHEN_FULL_REHAB,
    KITCHEN_REPAIR_LIGHT[0]: KITCHEN_REPAIR_HEAVY,
    KITCHEN_TURNOVER_LIGHT[0]: KITCHEN_TURNOVER_STD,
    BATHROOM_REFRESH[0]: BATHROOM_PARTIAL_REHAB,
    BATHROOM_PARTIAL_REHAB[0]: BATHROOM_FULL_REHAB,
    BATHROOM_REPAIR_LIGHT[0]: BATHROOM_REPAIR_HEAVY,
    BATHROOM_TURNOVER_LIGHT[0]: BATHROOM_TURNOVER_STD,
    # Bedroom / living use lightweight tiers: refresh -> full (no partial middle).
    BEDROOM_REFRESH[0]: BEDROOM_FULL_REHAB,
    BEDROOM_REPAIR_LIGHT[0]: BEDROOM_REPAIR_HEAVY,
    BEDROOM_TURNOVER_LIGHT[0]: BEDROOM_TURNOVER_STD,
    LIVING_REFRESH[0]: LIVING_FULL_REHAB,
    LIVING_REPAIR_LIGHT[0]: LIVING_REPAIR_HEAVY,
    LIVING_TURNOVER_LIGHT[0]: LIVING_TURNOVER_STD,
    # Top tiers (full_rehab / heavy / std) have no further escalation; the
    # Phase C floor handles any residual undercount.
}


def _absorbed_cost_estimate(
    drivers: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[int, int]:
    """Sum the catalog base cost range for drivers being absorbed by a package.

    Used by tier-escalation logic. Drivers that lack a catalog entry contribute
    zero rather than blowing up.
    """
    low = 0
    high = 0
    for driver in drivers or []:
        cat = catalog_lookup.get(driver.catalog_item_id or "") or {}
        cost = cat.get("cost") or {}
        try:
            low += int(cost.get("base_low") or 0)
            high += int(cost.get("base_high") or 0)
        except (TypeError, ValueError):
            continue
    return low, high


def _escalate_pricing_tier_if_undercut(
    spec: Tuple[str, int, int],
    drivers: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[Tuple[str, int, int], Optional[str]]:
    """Walk the escalation chain until tier_high >= absorbed_estimate_high.

    Returns the final spec (possibly unchanged) and a note describing the
    escalation if one happened. Phase C floor still applies if the top tier
    is exceeded.
    """
    absorbed_low, absorbed_high = _absorbed_cost_estimate(drivers, catalog_lookup)
    original = spec
    current = spec
    safety = 0  # guard against accidental cycles in the chain
    while current[2] < absorbed_high and safety < len(_PRICING_TIER_ESCALATION):
        next_spec = _PRICING_TIER_ESCALATION.get(current[0])
        if next_spec is None:
            break
        current = next_spec
        safety += 1
    if current is original:
        return current, None
    return current, (
        f"escalated_{original[0]}_to_{current[0]}"
        f"_absorbed_high={absorbed_high}"
    )


def _build_package_candidate(
    *,
    package_type: str,
    unit_id: str,
    room_surrogate_id: str,
    source_room_surrogate_ids: List[str],
    supporting_candidates: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
    trigger_reason: str,
    candidate_catalog_meta: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    spec, _, decision_notes = _resolve_pricing_profile(
        package_type,
        supporting_candidates,
        drivers,
        supports,
    )
    # Smarter dispatch: if the selected tier's ceiling is below what we'll absorb,
    # escalate up the chain (light→heavy, refresh→partial→full) so the package
    # cost reflects what's in scope. Phase C floor remains as the safety net for
    # cases where escalation is exhausted (top tier already).
    escalated_spec, escalation_note = _escalate_pricing_tier_if_undercut(
        spec, drivers, catalog_lookup,
    )
    if escalation_note:
        decision_notes = list(decision_notes) + [escalation_note]
        spec = escalated_spec
    pricing_profile, cost_low, cost_high = spec
    # The generic tier label must come from the spec actually priced — escalation
    # may have replaced the resolver's pick. Profile keys are
    # "<family>_<generic tier>" with single-token families.
    pricing_tier = pricing_profile.split("_", 1)[1]
    cost_model, cost_model_source = derive_package_cost_model()
    package_strength = compute_package_strength(
        drivers, supports, candidate_catalog_meta,
    )
    confidence_score = compute_initial_confidence_score(package_strength)
    package_category, room = _resolve_package_category_and_room(
        package_type, drivers, supports, catalog_lookup, candidate_catalog_meta,
    )
    estimate_scope, estimate_scope_reason = classify_package_scope(
        pricing_profile,
        supporting_candidates,
        trigger_reason,
        package_category=package_category,
        package_strength=package_strength,
    )
    issue_ids: List[str] = []
    cat_ids: List[str] = []
    photo_keys: List[str] = []
    evidence_items: List[Dict[str, Any]] = []
    for candidate in supporting_candidates:
        cat_item = (
            (candidate_catalog_meta or {}).get(id(candidate))
            or catalog_lookup.get(candidate.catalog_item_id or "", {})
        )
        evidence_items.append(_candidate_evidence_item(candidate, cat_item))
        issue_ids.extend(candidate.issue_ids or [])
        photo_keys.extend(candidate.photo_keys or [])
        if candidate.catalog_item_id:
            cat_ids.append(candidate.catalog_item_id)
    issue_ids = _unique_in_order(issue_ids)
    cat_ids = _unique_in_order(cat_ids)
    photo_keys = _unique_in_order(photo_keys)
    corroboration_basis = _package_corroboration_basis(
        trigger_reason,
        drivers,
        supports,
    )
    package = {
        "package_id": f"{package_type}__{unit_id}",
        "package_type": package_type,
        "package_category": package_category,
        "room": room,
        "package_level": PACKAGE_LEVEL_ROOM,
        "package_strength": package_strength,
        "confidence_score": confidence_score,
        "pricing_tier": pricing_tier,
        "pricing_profile": pricing_profile,
        "room_surrogate_id": room_surrogate_id,
        "estimate_unit_id": unit_id,
        "source_room_surrogate_ids": list(source_room_surrogate_ids or []),
        "estimate_group": room or "kitchen",
        "estimate_scope": estimate_scope,
        "estimate_scope_reason": estimate_scope_reason,
        "candidate_cost_low": cost_low,
        "candidate_cost_high": cost_high,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "cost_midpoint": (cost_low + cost_high) // 2,
        "cost_model": cost_model,
        "cost_model_source": cost_model_source,
        "absorption_scope": _package_absorption_scope(pricing_profile),
        "cap_behavior": CAP_BEHAVIOR_RESPECT_GROUP_CAP,
        "absorbed_unit_member_refs": [],
        "absorbed_total_low": 0,
        "absorbed_total_high": 0,
        "replacement_delta_low": 0,
        "replacement_delta_high": 0,
        "supporting_issue_ids": issue_ids,
        "supporting_catalog_item_ids": cat_ids,
        "driver_issue_ids": _unique_in_order([iid for c in drivers for iid in (c.issue_ids or [])]),
        "support_issue_ids": _unique_in_order([iid for c in supports for iid in (c.issue_ids or [])]),
        "review_photo_keys": photo_keys[:3],
        "supporting_photo_count": len(photo_keys),
        "corroboration_basis": corroboration_basis,
        "evidence_items": evidence_items,
        "trigger_reason": trigger_reason,
        "level_decision_notes": decision_notes,
        "verification_status": PACKAGE_VERIFICATION_NOT_RUN,
        "confirmed_issue_ids": [],
        "rejected_issue_ids": [],
        "evidence_summary": "",
        "estimate_eligible": False,
        "ui_eligible": False,
        "audit_only": True,
    }
    return package


def _resolve_package_category_and_room(
    package_type: str,
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
    candidate_catalog_meta: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[str, Optional[str]]:
    """Pick category and room from the first driver (or first support fallback).

    Falls back to the package_type → category/room map when catalog metadata
    is missing on the candidate's catalog entry.
    """
    for candidate in list(drivers or []) + list(supports or []):
        cat_item = (
            (candidate_catalog_meta or {}).get(id(candidate))
            or catalog_lookup.get(candidate.catalog_item_id or "", {})
        )
        category = catalog_package_category(cat_item)
        room = catalog_room(cat_item)
        if category or room:
            return (
                category or _PACKAGE_TYPE_TO_CATEGORY.get(package_type) or PACKAGE_CATEGORY_MODERNIZATION,
                room or _PACKAGE_TYPE_TO_ROOM.get(package_type),
            )
    return (
        _PACKAGE_TYPE_TO_CATEGORY.get(package_type) or PACKAGE_CATEGORY_MODERNIZATION,
        _PACKAGE_TYPE_TO_ROOM.get(package_type),
    )


def infer_package_candidates(
    candidates: List[EstimateCandidate],
    room_surrogates: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
    *,
    estimate_units: Optional[List[Dict[str, Any]]] = None,
    suppressed_out: Optional[List[Dict[str, Any]]] = None,
    builder_inputs_out: Optional[Dict[str, "PackageBuilderInputs"]] = None,
) -> List[Dict[str, Any]]:
    """Build package candidates from catalog-driven scene-aware affinity.

    Candidates are grouped by (estimate_unit_id, package_type) and dispatched
    to the matching pricing profile resolver. Routing comes from the catalog's
    per-item ``package_affinity`` blocks keyed by observed scene (see
    build_package_affinity); single-room blocks fall back to their only room
    when no scene resolves. Weak buckets are suppressed and (optionally)
    reported through ``suppressed_out``.
    """
    catalog_lookup = _catalog_lookup(issue_catalog)
    affinity_table = build_package_affinity(issue_catalog)
    scene_by_room = {
        str(r.get("room_surrogate_id") or ""): _normalize_scene_to_room(str(r.get("scene") or ""))
        for r in (room_surrogates or [])
        if isinstance(r, dict)
    }

    by_unit_type: Dict[Tuple[str, str], List[EstimateCandidate]] = {}
    candidate_catalog_meta: Dict[int, Dict[str, Any]] = {}
    # Cross-room recurrence tally: catalog_item_id -> set of distinct estimate
    # units where it appears in *support* role. Only support-role appearances
    # are counted, so an id that drives elsewhere (e.g. worn_or_stained_carpet
    # drives bedrooms but supports kitchens) never inflates the tally and is
    # never demoted. Uses the same _candidate_unit_id that keys the buckets.
    support_unit_tally: Dict[str, set] = {}
    for candidate in candidates or []:
        cat_item = catalog_lookup.get(candidate.catalog_item_id or "", {})
        effective_cat_item = _catalog_item_with_package_affinity(
            cat_item, candidate, scene_by_room, affinity_table,
        )
        candidate_catalog_meta[id(candidate)] = effective_cat_item
        if not is_package_eligible_catalog_item(effective_cat_item):
            continue
        package_type = catalog_package_type(effective_cat_item)
        if package_type not in VALID_PACKAGE_TYPES:
            continue
        # Per-room package types are room-scoped; the property-wide
        # turnover aggregate (interior_paint_flooring_refresh) is derived
        # downstream from per-room turnover packages, not inferred here.
        if package_type == PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH:
            continue
        expected_room = _PACKAGE_TYPE_TO_ROOM.get(package_type)
        if expected_room in (ROOM_KITCHEN, ROOM_BATHROOM, ROOM_BEDROOM, ROOM_LIVING) and scene_by_room and candidate.room_surrogate_id:
            # scene_by_room is normalized (living_room -> living) so this compares
            # like-for-like against the room constant.
            scene = _candidate_scene_room(candidate, scene_by_room)
            if scene and scene != expected_room:
                continue
        unit_id = _candidate_unit_id(candidate)
        by_unit_type.setdefault((unit_id, package_type), []).append(candidate)
        if (
            catalog_package_role(effective_cat_item) == PACKAGE_ROLE_SUPPORT
            and candidate.catalog_item_id
        ):
            support_unit_tally.setdefault(candidate.catalog_item_id, set()).add(unit_id)

    ambient_support_ids = {
        cat_id for cat_id, units in support_unit_tally.items()
        if len(units) >= _AMBIENT_SUPPORT_MIN_UNITS
    }

    out: List[Dict[str, Any]] = []
    for (unit_id, package_type), unit_candidates in sorted(by_unit_type.items()):
        defect_drivers: List[EstimateCandidate] = []
        opportunity_drivers: List[EstimateCandidate] = []
        supports: List[EstimateCandidate] = []
        for candidate in unit_candidates:
            cat_item = (
                candidate_catalog_meta.get(id(candidate))
                or catalog_lookup.get(candidate.catalog_item_id or "", {})
            )
            if _is_defect_driver_role(cat_item):
                defect_drivers.append(candidate)
            elif _is_opportunity_driver(cat_item):
                opportunity_drivers.append(candidate)
            elif catalog_package_role(cat_item) == PACKAGE_ROLE_SUPPORT:
                supports.append(candidate)

        distinct_opportunity_ids = {
            c.catalog_item_id for c in opportunity_drivers if c.catalog_item_id
        }
        has_multiphoto_opportunity = _has_multiphoto_opportunity_corroboration(
            opportunity_drivers
        )
        has_legacy_flat_opportunity_driver = (
            any(
                bool(getattr(candidate, "estimate_unit_id", ""))
                and (candidate_catalog_meta.get(id(candidate)) or {}).get(
                    "package_affinity"
                ) is None
                for candidate in opportunity_drivers
            )
        )

        # Ambient (recurring cross-room) supports still corroborate and are
        # costed, but do not count toward the driverless emit gate below.
        non_ambient_supports = [
            c for c in supports
            if (c.catalog_item_id or "") not in ambient_support_ids
        ]

        if defect_drivers:
            drivers = defect_drivers + opportunity_drivers
            supporting = drivers + supports
            trigger_reason = "package_driver"
        elif opportunity_drivers and (supports or len(distinct_opportunity_ids) >= 2):
            drivers = list(opportunity_drivers)
            supporting = drivers + supports
            trigger_reason = "opportunity_driver_with_corroboration"
        elif opportunity_drivers and has_multiphoto_opportunity:
            drivers = list(opportunity_drivers)
            supporting = drivers + supports
            trigger_reason = "opportunity_driver_with_multiphoto_corroboration"
        elif opportunity_drivers and has_legacy_flat_opportunity_driver:
            drivers = list(opportunity_drivers)
            supporting = drivers + supports
            trigger_reason = "legacy_flat_package_driver"

        elif len(non_ambient_supports) >= 2:
            drivers = []
            supporting = supports
            trigger_reason = "multiple_package_support_same_estimate_unit"
        else:
            _suppress_blocked_opportunity_drivers(opportunity_drivers)
            if suppressed_out is not None:
                # Distinguish a bucket killed *because* its supports were
                # demoted as ambient (would have emitted under the old
                # len(supports) >= 2 rule) from a genuinely weak bucket.
                demotion_caused = (
                    not opportunity_drivers
                    and len(supports) >= 2
                    and len(non_ambient_supports) < 2
                )
                record = _build_suppressed_candidate_record(
                    unit_id=unit_id,
                    package_type=package_type,
                    drivers=opportunity_drivers,
                    supports=supports,
                    reason=(
                        "weak_after_ambient_support_demotion"
                        if demotion_caused
                        else "weak_no_qualifying_pattern"
                    ),
                )
                if demotion_caused:
                    record["ambient_demoted_catalog_item_ids"] = sorted({
                        c.catalog_item_id for c in supports
                        if (c.catalog_item_id or "") in ambient_support_ids
                    })
                suppressed_out.append(record)
            continue

        strength = compute_package_strength(
            drivers, supports, candidate_catalog_meta,
        )
        if strength not in VALID_EMITTED_PACKAGE_STRENGTHS:
            if suppressed_out is not None:
                suppressed_out.append(_build_suppressed_candidate_record(
                    unit_id=unit_id,
                    package_type=package_type,
                    drivers=drivers,
                    supports=supports,
                    reason="weak_strength_below_emit_threshold",
                ))
            continue

        first = supporting[0]
        source_room_ids = _source_room_ids_for_unit(unit_id, estimate_units)
        if not source_room_ids and first.room_surrogate_id:
            source_room_ids = [first.room_surrogate_id]
        package = _build_package_candidate(
            package_type=package_type,
            unit_id=unit_id,
            room_surrogate_id=first.room_surrogate_id or (source_room_ids[0] if source_room_ids else ""),
            source_room_surrogate_ids=source_room_ids,
            supporting_candidates=supporting,
            drivers=drivers,
            supports=supports,
            catalog_lookup=catalog_lookup,
            trigger_reason=trigger_reason,
            candidate_catalog_meta=candidate_catalog_meta,
        )
        if builder_inputs_out is not None:
            builder_inputs_out[str(package["package_id"])] = PackageBuilderInputs(
                package_type=package_type,
                unit_id=unit_id,
                supporting_candidates=list(supporting),
                drivers=list(drivers),
                supports=list(supports),
                catalog_lookup=catalog_lookup,
                candidate_catalog_meta=candidate_catalog_meta,
                trigger_reason=trigger_reason,
            )
        # Audit note: ambient supports rode along as corroboration/cost on a
        # package that stood on its own anchor (driver or non-ambient supports).
        ambient_corroborating = sorted({
            c.catalog_item_id for c in supports
            if (c.catalog_item_id or "") in ambient_support_ids
        })
        if ambient_corroborating:
            package["level_decision_notes"] = list(
                package.get("level_decision_notes") or []
            ) + ["ambient_supports_corroborating=" + ",".join(ambient_corroborating)]
        out.append(package)
    return out


def _build_suppressed_candidate_record(
    *,
    unit_id: str,
    package_type: str,
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
    reason: str,
) -> Dict[str, Any]:
    """Lightweight debug record for weak-strength inference attempts."""
    return {
        "estimate_unit_id": unit_id,
        "package_type": package_type,
        "package_strength": PACKAGE_STRENGTH_WEAK,
        "suppression_reason": reason,
        "driver_catalog_item_ids": _unique_in_order(
            [c.catalog_item_id for c in drivers if c.catalog_item_id]
        ),
        "support_catalog_item_ids": _unique_in_order(
            [c.catalog_item_id for c in supports if c.catalog_item_id]
        ),
        "supporting_photo_count": len(_distinct_photo_keys(drivers + supports)),
        "corroboration_basis": "insufficient_support",
    }


def _verification_lookup(package_verifications: Any) -> Dict[str, Dict[str, Any]]:
    if not package_verifications:
        return {}
    if isinstance(package_verifications, dict):
        if all(isinstance(v, dict) for v in package_verifications.values()):
            return {str(k): dict(v) for k, v in package_verifications.items()}
        package_id = str(package_verifications.get("package_id") or "")
        return {package_id: dict(package_verifications)} if package_id else {}
    if isinstance(package_verifications, list):
        out: Dict[str, Dict[str, Any]] = {}
        for record in package_verifications:
            if not isinstance(record, dict):
                continue
            package_id = str(record.get("package_id") or "")
            if package_id:
                out[package_id] = dict(record)
        return out
    return {}


def _apply_verification_to_package(
    package: Dict[str, Any],
    verification: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    pkg = dict(package)
    verification = verification or {}
    status = str(verification.get("verification_status") or pkg.get("verification_status") or PACKAGE_VERIFICATION_NOT_RUN).strip().lower()
    if status not in VALID_PACKAGE_VERIFICATION_STATUSES:
        status = PACKAGE_VERIFICATION_UNCERTAIN
    pkg["verification_status"] = status
    reviewed_issue_ids = list(
        verification.get("reviewed_issue_ids")
        or pkg.get("reviewed_issue_ids")
        or []
    )
    reviewed_set = {str(issue_id) for issue_id in reviewed_issue_ids}
    confirmed_issue_ids = list(
        verification.get("confirmed_issue_ids") or pkg.get("confirmed_issue_ids") or []
    )
    rejected_issue_ids = list(
        verification.get("rejected_issue_ids") or pkg.get("rejected_issue_ids") or []
    )
    if reviewed_set:
        confirmed_issue_ids = [
            issue_id for issue_id in confirmed_issue_ids if str(issue_id) in reviewed_set
        ]
        rejected_issue_ids = [
            issue_id for issue_id in rejected_issue_ids if str(issue_id) in reviewed_set
        ]
    pkg["confirmed_issue_ids"] = confirmed_issue_ids
    pkg["rejected_issue_ids"] = rejected_issue_ids
    pkg["reviewed_issue_ids"] = reviewed_issue_ids
    pkg["evidence_summary"] = str(verification.get("evidence_summary") or pkg.get("evidence_summary") or "")
    pkg["raw_pass_2f_response"] = verification.get("raw_response") or pkg.get("raw_pass_2f_response")
    pkg["review_photo_keys"] = list(verification.get("review_photo_keys") or pkg.get("review_photo_keys") or [])
    pkg["review_image_paths"] = list(verification.get("review_image_paths") or pkg.get("review_image_paths") or [])
    if (
        pkg.get("room") == ROOM_BATHROOM
        and ("visible_room_count" in verification or "visible_room_count" in pkg)
    ):
        pkg["visible_room_count"] = str(
            verification.get("visible_room_count")
            or pkg.get("visible_room_count")
            or "unclear"
        )
        pkg["visible_room_count_evidence"] = str(
            verification.get("visible_room_count_evidence")
            or pkg.get("visible_room_count_evidence")
            or ""
        )
    is_active = status in ACTIVE_PACKAGE_STATUSES
    pkg["estimate_eligible"] = is_active
    pkg["ui_eligible"] = is_active
    pkg["audit_only"] = not is_active
    pkg["confidence_score"] = compute_post_verification_score(
        initial_score=float(pkg.get("confidence_score") or 0.0),
        confirmed_count=len(pkg["confirmed_issue_ids"]),
        rejected_count=len(pkg["rejected_issue_ids"]),
        supporting_count=len(pkg.get("supporting_issue_ids") or []),
    )
    return pkg


def _project_package_issue_ids_to_confirmed(pkg: Dict[str, Any]) -> None:
    """Project an active reviewed package's supporting_issue_ids to confirmed-only.

    2f-rejected issue IDs must never remain "supporting": they would still
    index candidates as package-backed and still fire exact supporting-issue
    absorption in reconciliation. The pre-review list is preserved under
    supporting_issue_ids_original (also the idempotence guard).
    """
    if pkg.get("supporting_issue_ids_original") is not None:
        return
    if not (pkg.get("reviewed_issue_ids") or []):
        return
    original = list(pkg.get("supporting_issue_ids") or [])
    confirmed = {str(i) for i in (pkg.get("confirmed_issue_ids") or [])}
    pkg["supporting_issue_ids_original"] = original
    pkg["supporting_issue_ids"] = [i for i in original if str(i) in confirmed]


def _candidate_confirmed(candidate: EstimateCandidate, confirmed: set) -> bool:
    return bool({str(i) for i in (candidate.issue_ids or [])} & confirmed)


def _retier_package_from_confirmed_evidence(
    pkg: Dict[str, Any],
    inputs: Optional[PackageBuilderInputs],
) -> None:
    """Recompute tier, cost, strength, scope, and evidence from confirmed-only
    evidence after Pass 2f. Mutates the finalized package copy in place.

    The finalized dict is a SHALLOW copy of the pre-finalize candidate dict
    (see _apply_verification_to_package), so nested lists are always replaced,
    never mutated in place. An active package left with zero confirmed
    eligible evidence is demoted to audit-only — pre-2f tier and price must
    never survive as truth once 2f has rejected the evidence behind them.
    """
    if str(pkg.get("verification_status") or "") not in ACTIVE_PACKAGE_STATUSES:
        pkg["retier_audit"] = {"applied": False, "reason": "status_not_active"}
        return
    if (pkg.get("retier_audit") or {}).get("applied"):
        return
    if not (pkg.get("reviewed_issue_ids") or []):
        pkg["retier_audit"] = {"applied": False, "reason": "no_reviewed_issue_ids"}
        return
    if inputs is None:
        logger.warning(
            "Re-tier skipped: no builder inputs registered for package_id=%s — "
            "pre-2f tier/cost retained.",
            pkg.get("package_id"),
        )
        pkg["retier_audit"] = {"applied": False, "reason": "missing_builder_inputs"}
        return

    confirmed = {str(i) for i in (pkg.get("confirmed_issue_ids") or [])}
    kept_supporting = [
        c for c in inputs.supporting_candidates if _candidate_confirmed(c, confirmed)
    ]
    kept_drivers = [c for c in inputs.drivers if _candidate_confirmed(c, confirmed)]
    kept_supports = [c for c in inputs.supports if _candidate_confirmed(c, confirmed)]

    previous = {
        "pricing_tier": pkg.get("pricing_tier"),
        "pricing_profile": pkg.get("pricing_profile"),
        "cost_low": pkg.get("cost_low"),
        "cost_high": pkg.get("cost_high"),
        "package_strength": pkg.get("package_strength"),
        "estimate_scope": pkg.get("estimate_scope"),
        "supporting_issue_count": len(pkg.get("supporting_issue_ids") or []),
    }
    dropped_catalog_ids = _unique_in_order([
        c.catalog_item_id
        for c in inputs.supporting_candidates
        if c.catalog_item_id and not _candidate_confirmed(c, confirmed)
    ])

    if not confirmed or not kept_supporting:
        pkg["estimate_eligible"] = False
        pkg["ui_eligible"] = False
        pkg["audit_only"] = True
        pkg["demotion_reason"] = "active_status_without_confirmed_evidence"
        pkg["level_decision_notes"] = list(pkg.get("level_decision_notes") or []) + [
            "demoted_no_confirmed_evidence_after_pass_2f"
        ]
        pkg["retier_audit"] = {
            "applied": True,
            "demoted": True,
            "previous": previous,
            "dropped_candidate_catalog_ids": dropped_catalog_ids,
            "confirmed_candidate_count": 0,
        }
        return

    spec, _, _notes = _resolve_pricing_profile(
        str(pkg.get("package_type") or ""),
        kept_supporting,
        kept_drivers,
        kept_supports,
    )
    escalated_spec, escalation_note = _escalate_pricing_tier_if_undercut(
        spec, kept_drivers, inputs.catalog_lookup,
    )
    if escalation_note:
        spec = escalated_spec
    pricing_profile, cost_low, cost_high = spec

    package_strength = compute_package_strength(
        kept_drivers, kept_supports, inputs.candidate_catalog_meta,
    )
    estimate_scope, estimate_scope_reason = classify_package_scope(
        pricing_profile,
        kept_supporting,
        str(pkg.get("trigger_reason") or inputs.trigger_reason or ""),
        package_category=str(pkg.get("package_category") or ""),
        package_strength=package_strength,
    )

    # Rebuild evidence from the kept candidates, pruned to confirmed refs.
    # The pre-re-tier evidence is preserved verbatim for audit.
    if pkg.get("evidence_items_original") is None:
        pkg["evidence_items_original"] = pkg.get("evidence_items") or []
    evidence_items: List[Dict[str, Any]] = []
    supporting_issue_ids: List[str] = []
    cat_ids: List[str] = []
    photo_keys: List[str] = []
    for candidate in kept_supporting:
        cat_item = (
            (inputs.candidate_catalog_meta or {}).get(id(candidate))
            or inputs.catalog_lookup.get(candidate.catalog_item_id or "", {})
        )
        item = _candidate_evidence_item(candidate, cat_item)
        kept_refs = [
            ref for ref in (item.get("issue_refs") or [])
            if str(ref.get("issue_id") or "") in confirmed
        ]
        kept_ids = [i for i in (item.get("issue_ids") or []) if str(i) in confirmed]
        if not kept_ids:
            continue
        item["issue_refs"] = kept_refs
        item["issue_ids"] = kept_ids
        ref_photo_keys = _unique_in_order([
            str(r.get("photo_key") or "") for r in kept_refs if r.get("photo_key")
        ])
        if ref_photo_keys:
            item["photo_keys"] = ref_photo_keys
            item["supporting_photo_count"] = len(ref_photo_keys)
        evidence_items.append(item)
        supporting_issue_ids.extend(kept_ids)
        photo_keys.extend(item.get("photo_keys") or [])
        if candidate.catalog_item_id:
            cat_ids.append(candidate.catalog_item_id)

    supporting_issue_ids = _unique_in_order(supporting_issue_ids)
    photo_keys = _unique_in_order(photo_keys)

    pkg["pricing_profile"] = pricing_profile
    pkg["pricing_tier"] = pricing_profile.split("_", 1)[1]
    pkg["candidate_cost_low"] = cost_low
    pkg["candidate_cost_high"] = cost_high
    pkg["cost_low"] = cost_low
    pkg["cost_high"] = cost_high
    pkg["cost_midpoint"] = (cost_low + cost_high) // 2
    pkg["package_strength"] = package_strength
    pkg["confidence_score"] = compute_post_verification_score(
        initial_score=compute_initial_confidence_score(package_strength),
        confirmed_count=len(pkg.get("confirmed_issue_ids") or []),
        rejected_count=len(pkg.get("rejected_issue_ids") or []),
        supporting_count=max(len(supporting_issue_ids), 1),
    )
    pkg["absorption_scope"] = _package_absorption_scope(pricing_profile)
    pkg["estimate_scope"] = estimate_scope
    pkg["estimate_scope_reason"] = estimate_scope_reason
    pkg["evidence_items"] = evidence_items
    pkg["supporting_issue_ids"] = supporting_issue_ids
    pkg["supporting_catalog_item_ids"] = _unique_in_order(cat_ids)
    pkg["driver_issue_ids"] = _unique_in_order([
        str(i) for c in kept_drivers for i in (c.issue_ids or []) if str(i) in confirmed
    ])
    pkg["support_issue_ids"] = _unique_in_order([
        str(i) for c in kept_supports for i in (c.issue_ids or []) if str(i) in confirmed
    ])
    pkg["supporting_photo_count"] = len(photo_keys)
    if not (pkg.get("review_photo_keys") or []):
        pkg["review_photo_keys"] = photo_keys[:3]
    notes: List[str] = []
    if escalation_note:
        notes.append(escalation_note)
    if previous["pricing_profile"] != pricing_profile:
        notes.append(
            f"retiered_{previous['pricing_profile']}_to_{pricing_profile}_after_pass_2f"
        )
    if notes:
        pkg["level_decision_notes"] = list(pkg.get("level_decision_notes") or []) + notes
    pkg["retier_audit"] = {
        "applied": True,
        "demoted": False,
        "previous": previous,
        "dropped_candidate_catalog_ids": dropped_catalog_ids,
        "confirmed_candidate_count": len(kept_supporting),
    }


def finalize_package_candidates(
    package_candidates: List[Dict[str, Any]],
    package_verifications: Any = None,
    *,
    require_confirmation: bool = True,
    builder_inputs: Optional[Dict[str, PackageBuilderInputs]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply package verification and return (estimate_packages, audit_packages).

    Active reviewed packages get their supporting_issue_ids projected to
    confirmed-only, and — when `builder_inputs` is provided — are re-tiered
    from confirmed-only evidence (tier, cost, strength, scope, evidence). A
    re-tier that finds zero confirmed evidence demotes the package to
    audit-only, which excludes it from the returned estimate list.
    """
    verification_by_id = _verification_lookup(package_verifications)
    audit_packages: List[Dict[str, Any]] = []
    estimate_packages: List[Dict[str, Any]] = []
    for package in package_candidates or []:
        package_id = str(package.get("package_id") or "")
        finalized = _apply_verification_to_package(
            package,
            verification_by_id.get(package_id),
        )
        if finalized.get("verification_status") in ACTIVE_PACKAGE_STATUSES:
            _project_package_issue_ids_to_confirmed(finalized)
            if builder_inputs is not None:
                _retier_package_from_confirmed_evidence(
                    finalized,
                    builder_inputs.get(package_id),
                )
        audit_packages.append(finalized)
        final_status = finalized.get("verification_status")
        if require_confirmation and final_status not in ACTIVE_PACKAGE_STATUSES:
            continue
        if final_status in ACTIVE_PACKAGE_STATUSES and not finalized.get("audit_only"):
            estimate_packages.append(finalized)
    return estimate_packages, audit_packages


# ─── Package subsumption on physical-room identity ───────────────────────────
#
# infer_package_candidates groups by (estimate_unit_id, package_type), so one
# room can emit a modernization AND a repair AND a turnover package at once.
# Child absorption dedupes their line items, but the package tier ranges are
# independent and all sum into the scope rollups — a bathroom_full_rehab and a
# bathroom_repair_heavy both price, yet a gut rehab subsumes the repair work.
# This pass drops the subsumed bundles after finalization. Dropping (rather
# than transferring supporting_issue_ids to the winner) keeps required-scope
# children out of marketability-scoped packages. Dropped packages' children
# instead get broad-absorbed by the modernization where its absorption_scope
# covers them, or retained as line items in their own scope.
#
# Subsumption is decided on CONFIRMED PHYSICAL ROOMS, not estimate-unit ids:
# estimate units like bathroom_primary can be conservative/default merges of
# several physical bathrooms, so unit-id equality is not proof that two
# packages cover the same room. A repair is only dropped when every confirmed
# room it covers is also confirmed for a subsuming-tier modernization whose
# absorption scope covers the repair's confirmed components; otherwise it is
# retained and the retention is audited. Runs AFTER bathroom expansion so the
# winners are the final per-surrogate packages.

_MODERNIZATION_TIER_RANK = {"refresh": 0, "partial_rehab": 1, "full_rehab": 2}

# Modernization tiers whose scope includes the unit's repair work. A refresh
# is cosmetic-only — it does not fix plumbing — so it never subsumes repair.
_REPAIR_SUBSUMING_MODERNIZATION_TIERS = frozenset({"partial_rehab", "full_rehab"})

# repair_heavy already covers flooring, so a coexisting turnover_std (paint +
# flooring) downgrades to its paint-only light tier. Keyed by pricing profile.
_TURNOVER_STD_TO_LIGHT: Dict[str, Tuple[str, int, int]] = {
    KITCHEN_TURNOVER_STD[0]: KITCHEN_TURNOVER_LIGHT,
    BATHROOM_TURNOVER_STD[0]: BATHROOM_TURNOVER_LIGHT,
    BEDROOM_TURNOVER_STD[0]: BEDROOM_TURNOVER_LIGHT,
    LIVING_TURNOVER_STD[0]: LIVING_TURNOVER_LIGHT,
}

_SUBSUMPTION_TRADEOFF_NOTE = (
    "Dropping a repair package reverts its unit's required_rehab from the "
    "repair bundle range to the sum of retained line items (v3 semantics); "
    "required-scope children are never absorbed into marketability packages."
)


def _drop_subsumed_package(
    loser: Dict[str, Any],
    winner: Dict[str, Any],
    *,
    ref_field: str,
    note: str,
    winner_list_field: str,
) -> None:
    """Mark a package as dropped, mutating the shared dict in place.

    Eligibility flags are the gate — verification_status is never touched, so
    "confirmed" keeps meaning "the VLM looked and said yes" (same pattern as
    _apply_verification_to_package). The mutation is deliberately in place:
    finalize_package_candidates aliases the same dict into the
    package_candidates audit list, which therefore stays consistent for free.
    Idempotent: re-applying the same drop never duplicates notes or lineage.
    """
    winner_id = str(winner.get("package_id") or "")
    loser_id = str(loser.get("package_id") or "")
    if loser.get(ref_field) == winner_id and note in (loser.get("level_decision_notes") or []):
        return
    loser["estimate_eligible"] = False
    loser["ui_eligible"] = False
    loser["audit_only"] = True
    loser[ref_field] = winner_id
    if note not in (loser.get("level_decision_notes") or []):
        loser["level_decision_notes"] = list(loser.get("level_decision_notes") or []) + [note]
    winner_list = list(winner.get(winner_list_field) or [])
    if loser_id not in winner_list:
        winner_list.append(loser_id)
    winner[winner_list_field] = winner_list


def _downgrade_turnover_to_light(
    turnover: Dict[str, Any],
    repair: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Downgrade a turnover_std package to its light tier in place.

    Returns the audit record, or None when the profile has no light mapping
    or the package was already downgraded (idempotence).
    """
    if turnover.get("turnover_downgraded"):
        return None
    from_profile = str(turnover.get("pricing_profile") or "")
    light_spec = _TURNOVER_STD_TO_LIGHT.get(from_profile)
    if light_spec is None:
        return None
    turnover["turnover_downgraded"] = True
    to_profile, cost_low, cost_high = light_spec
    turnover["pricing_profile"] = to_profile
    turnover["pricing_tier"] = "turnover_light"
    turnover["candidate_cost_low"] = cost_low
    turnover["candidate_cost_high"] = cost_high
    turnover["cost_low"] = cost_low
    turnover["cost_high"] = cost_high
    turnover["cost_midpoint"] = (cost_low + cost_high) // 2
    turnover["absorption_scope"] = _package_absorption_scope(to_profile)
    turnover["level_decision_notes"] = list(
        turnover.get("level_decision_notes") or []
    ) + ["downgraded_to_light_by_same_unit_repair_heavy"]
    return {
        "unit": str(turnover.get("estimate_unit_id") or "") or str(turnover.get("room_surrogate_id") or ""),
        "turnover_package_id": str(turnover.get("package_id") or ""),
        "repair_package_id": str(repair.get("package_id") or ""),
        "from_profile": from_profile,
        "to_profile": to_profile,
    }


def dedupe_same_unit_modernizations(
    packages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep one modernization per (estimate unit, room); demote the rest.

    Runs BEFORE bathroom expansion so the expander sees a unique winner.
    Exact-unit-key equality is valid here (unlike subsumption): same unit id +
    same room + same category is the same package claim twice, not a
    physical-identity inference. Winner = highest tier, then confidence, then
    package_id. Returns (active_packages, dedupe_records) with records shaped
    like subsumption entries (rule "modernization_dedupe_highest_tier").
    """
    by_unit: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for pkg in packages or []:
        if not isinstance(pkg, dict):
            continue
        if str(pkg.get("verification_status") or "") not in ACTIVE_PACKAGE_STATUSES:
            continue
        if pkg.get("package_level") != PACKAGE_LEVEL_ROOM:
            continue
        if str(pkg.get("package_category") or "") != PACKAGE_CATEGORY_MODERNIZATION:
            continue
        unit_key = (
            str(pkg.get("estimate_unit_id") or "")
            or str(pkg.get("room_surrogate_id") or "")
        )
        if not unit_key:
            continue
        by_unit.setdefault((unit_key, str(pkg.get("room") or "")), []).append(pkg)

    records: List[Dict[str, Any]] = []
    dropped: set = set()
    for (unit_key, _room), group in by_unit.items():
        if len(group) < 2:
            continue
        ranked = sorted(
            group,
            key=lambda p: (
                -_MODERNIZATION_TIER_RANK.get(str(p.get("pricing_tier") or ""), -1),
                -float(p.get("confidence_score") or 0.0),
                str(p.get("package_id") or ""),
            ),
        )
        winner = ranked[0]
        for loser in ranked[1:]:
            _drop_subsumed_package(
                loser,
                winner,
                ref_field="subsumed_by_package_id",
                note="deduped_same_unit_modernization",
                winner_list_field="subsumed_package_ids",
            )
            dropped.add(id(loser))
            records.append({
                "unit": unit_key,
                "winner_package_id": str(winner.get("package_id") or ""),
                "loser_package_id": str(loser.get("package_id") or ""),
                "rule": "modernization_dedupe_highest_tier",
            })

    active = [pkg for pkg in packages or [] if id(pkg) not in dropped]
    return active, records


def _package_confirmed_surrogates(pkg: Dict[str, Any]) -> Tuple[set, str]:
    """Resolve the set of physical rooms with 2f-confirmed evidence.

    Preferred basis is per-ref surrogate ids on confirmed issue refs. When the
    refs carry no surrogate ids but confirmed evidence exists, fall back to the
    package's source_room_surrogate_ids — data sparsity must not read as "no
    confirmed rooms" and trigger mass retention. The basis string is recorded
    in every audit record so the weaker inference stays visible.
    """
    grouped = _group_confirmed_refs_by_surrogate(pkg)
    if grouped:
        return set(grouped.keys()), "confirmed_refs"
    if pkg.get("confirmed_issue_ids"):
        source = {
            str(s) for s in (pkg.get("source_room_surrogate_ids") or []) if str(s)
        }
        if source:
            return source, "package_source_rooms"
    return set(), "none"


def _confirmed_evidence_components(
    pkg: Dict[str, Any],
    surrogate_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Confirmed evidence items (optionally restricted to one physical room),
    each classified for coverage checks."""
    confirmed = {str(i) for i in (pkg.get("confirmed_issue_ids") or [])}
    out: List[Dict[str, Any]] = []
    for item in pkg.get("evidence_items") or []:
        matched = False
        for ref in item.get("issue_refs") or []:
            if not isinstance(ref, dict):
                continue
            issue_id = str(ref.get("issue_id") or "")
            ref_surrogate = str(ref.get("room_surrogate_id") or "").strip()
            if issue_id not in confirmed:
                continue
            if surrogate_id is not None and ref_surrogate != surrogate_id:
                continue
            matched = True
            break
        if not matched and surrogate_id is None:
            # Refs may be absent on stub/legacy evidence — fall back to item ids.
            if {str(i) for i in (item.get("issue_ids") or [])} & confirmed:
                matched = True
        if matched:
            out.append({
                "component": classify_component(item),
                "trade_bucket": str(item.get("trade_bucket") or ""),
                "catalog_item_id": str(item.get("catalog_item_id") or ""),
            })
    return out


def _winner_scope_covers_components(
    winner: Dict[str, Any],
    components: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """True when the winner's absorption scope covers every component; also
    returns the uncovered labels for the audit."""
    scope = winner.get("absorption_scope") or {}
    scope_components = set(scope.get("components") or [])
    scope_trades = set(scope.get("trade_buckets") or [])
    uncovered: List[str] = []
    for entry in components:
        component = entry.get("component")
        trade = entry.get("trade_bucket") or ""
        if (component is not None and component in scope_components) or (
            trade and trade in scope_trades
        ):
            continue
        label = entry.get("catalog_item_id") or component or trade or "unknown"
        if label not in uncovered:
            uncovered.append(label)
    return (not uncovered), uncovered


def _unit_merge_info(
    pkg: Dict[str, Any],
    estimate_units: Optional[List[Dict[str, Any]]],
    merge_decisions: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    unit_id = str(pkg.get("estimate_unit_id") or "")
    unit = next(
        (
            u for u in (estimate_units or [])
            if isinstance(u, dict) and str(u.get("estimate_unit_id") or "") == unit_id
        ),
        {},
    )
    decision = next(
        (
            d for d in (merge_decisions or [])
            if isinstance(d, dict) and str(d.get("to") or "") == unit_id
        ),
        {},
    )
    return {
        "unit": unit_id,
        "confidence": unit.get("confidence") or decision.get("confidence"),
        "reason": unit.get("merge_reason") or decision.get("reason"),
    }


def _cost_range(pkg: Dict[str, Any]) -> Dict[str, int]:
    return {
        "low": int(pkg.get("cost_low") or 0),
        "high": int(pkg.get("cost_high") or 0),
    }


def apply_physical_package_subsumption(
    packages: List[Dict[str, Any]],
    *,
    estimate_units: Optional[List[Dict[str, Any]]] = None,
    merge_decisions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Drop packages whose confirmed physical rooms a modernization covers.

    Rules (all decided on confirmed surrogate sets, never unit-id equality):
      - a repair is subsumed only when EVERY confirmed repair room is covered
        by some same-room modernization at partial_rehab/full_rehab whose
        absorption scope covers that room's confirmed components; a repair
        with no provable rooms, an uncovered room, or an uncovered component
        is RETAINED and audited;
      - a turnover is suppressed when every confirmed turnover room is covered
        by some same-room modernization at ANY tier (no component check —
        paint/flooring/finish is within any modernization);
      - a retained repair_heavy still downgrades a retained turnover_std with
        overlapping confirmed rooms to its light tier.

    Returns (active_packages, subsumption_audit). Dropped dicts are mutated in
    place for the aliased package_candidates audit list.
    """
    audit: Dict[str, Any] = {
        "applied": False,
        "subsumptions": [],
        "downgrades": [],
        "retained_repairs": [],
        "retained_turnovers": [],
        "tradeoff_note": _SUBSUMPTION_TRADEOFF_NOTE,
    }

    active_room: List[Dict[str, Any]] = [
        pkg for pkg in packages or []
        if isinstance(pkg, dict)
        and str(pkg.get("verification_status") or "") in ACTIVE_PACKAGE_STATUSES
        and pkg.get("package_level") == PACKAGE_LEVEL_ROOM
        and not pkg.get("audit_only")
    ]
    modernizations = [
        p for p in active_room
        if str(p.get("package_category") or "") == PACKAGE_CATEGORY_MODERNIZATION
    ]
    repairs = [
        p for p in active_room
        if str(p.get("package_category") or "") == PACKAGE_CATEGORY_REPAIR
    ]
    turnovers = [
        p for p in active_room
        if str(p.get("package_category") or "") == PACKAGE_CATEGORY_TURNOVER
    ]

    mod_infos: List[Dict[str, Any]] = []
    for mod in modernizations:
        surrogates, basis = _package_confirmed_surrogates(mod)
        mod_infos.append({
            "pkg": mod,
            "surrogates": surrogates,
            "basis": basis,
            "tier": str(mod.get("pricing_tier") or ""),
            "room": str(mod.get("room") or ""),
        })

    def _legacy_unit(pkg: Dict[str, Any]) -> str:
        return (
            str(pkg.get("estimate_unit_id") or "")
            or str(pkg.get("room_surrogate_id") or "")
        )

    dropped: set = set()

    for repair in repairs:
        loser_surrogates, basis = _package_confirmed_surrogates(repair)
        base_record = {
            "unit": _legacy_unit(repair),
            "loser_confirmed_surrogates": sorted(loser_surrogates),
            "surrogate_basis": basis,
            "loser_cost": _cost_range(repair),
            "unit_merge": _unit_merge_info(repair, estimate_units, merge_decisions),
        }
        if not loser_surrogates:
            audit["retained_repairs"].append({
                **base_record,
                "retained_package_id": str(repair.get("package_id") or ""),
                "retention_reason": "loser_no_confirmed_surrogates",
            })
            continue

        winners_by_surrogate: Dict[str, Dict[str, Any]] = {}
        uncovered_surrogates: List[str] = []
        uncovered_components: List[str] = []
        coverage_failed_somewhere = False
        for surrogate in sorted(loser_surrogates):
            # With per-ref surrogate data, coverage is checked against only
            # that room's confirmed components; the source-rooms fallback has
            # no per-room refs, so it checks all confirmed components.
            component_surrogate = surrogate if basis == "confirmed_refs" else None
            loser_components = _confirmed_evidence_components(
                repair, component_surrogate,
            )
            winner_pkg: Optional[Dict[str, Any]] = None
            surrogate_had_tier_match = False
            surrogate_uncovered: List[str] = []
            for info in mod_infos:
                if info["room"] != str(repair.get("room") or ""):
                    continue
                if info["tier"] not in _REPAIR_SUBSUMING_MODERNIZATION_TIERS:
                    continue
                if surrogate not in info["surrogates"]:
                    continue
                surrogate_had_tier_match = True
                covered, uncovered = _winner_scope_covers_components(
                    info["pkg"], loser_components,
                )
                if covered:
                    winner_pkg = info["pkg"]
                    break
                surrogate_uncovered = uncovered
            if winner_pkg is not None:
                winners_by_surrogate[surrogate] = winner_pkg
            else:
                uncovered_surrogates.append(surrogate)
                if surrogate_had_tier_match:
                    coverage_failed_somewhere = True
                    uncovered_components.extend(
                        u for u in surrogate_uncovered
                        if u not in uncovered_components
                    )

        if uncovered_surrogates:
            audit["retained_repairs"].append({
                **base_record,
                "retained_package_id": str(repair.get("package_id") or ""),
                "retention_reason": (
                    "components_not_covered"
                    if coverage_failed_somewhere
                    else "loser_surrogates_not_covered"
                ),
                "uncovered_surrogates": uncovered_surrogates,
                "uncovered_components": uncovered_components,
            })
            continue

        winner_ids = _unique_in_order([
            str(winners_by_surrogate[s].get("package_id") or "")
            for s in sorted(winners_by_surrogate)
        ])
        primary = winners_by_surrogate[sorted(winners_by_surrogate)[0]]
        for winner in {id(w): w for w in winners_by_surrogate.values()}.values():
            _drop_subsumed_package(
                repair,
                winner,
                ref_field="subsumed_by_package_id",
                note="subsumed_by_same_unit_modernization",
                winner_list_field="subsumed_package_ids",
            )
        # _drop_subsumed_package leaves the single-valued pointer at the LAST
        # winner applied; pin it to the primary and record the full list.
        repair["subsumed_by_package_id"] = str(primary.get("package_id") or "")
        repair["subsumed_by_package_ids"] = winner_ids
        dropped.add(id(repair))
        audit["subsumptions"].append({
            **base_record,
            "winner_package_id": str(primary.get("package_id") or ""),
            "winner_package_ids": winner_ids,
            "loser_package_id": str(repair.get("package_id") or ""),
            "rule": "modernization_subsumes_repair",
            "surrogate_winners": {
                s: str(w.get("package_id") or "")
                for s, w in winners_by_surrogate.items()
            },
            "winner_confirmed_surrogates": sorted({
                s
                for info in mod_infos
                if info["pkg"] in winners_by_surrogate.values()
                for s in info["surrogates"]
            }),
            "component_coverage": "covered",
            "winner_cost": _cost_range(primary),
        })

    for turnover in turnovers:
        t_surrogates, t_basis = _package_confirmed_surrogates(turnover)
        base_record = {
            "unit": _legacy_unit(turnover),
            "loser_confirmed_surrogates": sorted(t_surrogates),
            "surrogate_basis": t_basis,
            "loser_cost": _cost_range(turnover),
            "unit_merge": _unit_merge_info(turnover, estimate_units, merge_decisions),
        }
        if not t_surrogates:
            audit["retained_turnovers"].append({
                **base_record,
                "retained_package_id": str(turnover.get("package_id") or ""),
                "retention_reason": "loser_no_confirmed_surrogates",
            })
            continue
        winners_by_surrogate = {}
        uncovered_surrogates = []
        for surrogate in sorted(t_surrogates):
            winner_pkg = next(
                (
                    info["pkg"] for info in mod_infos
                    if info["room"] == str(turnover.get("room") or "")
                    and surrogate in info["surrogates"]
                ),
                None,
            )
            if winner_pkg is not None:
                winners_by_surrogate[surrogate] = winner_pkg
            else:
                uncovered_surrogates.append(surrogate)
        if uncovered_surrogates:
            audit["retained_turnovers"].append({
                **base_record,
                "retained_package_id": str(turnover.get("package_id") or ""),
                "retention_reason": "loser_surrogates_not_covered",
                "uncovered_surrogates": uncovered_surrogates,
            })
            continue
        winner_ids = _unique_in_order([
            str(winners_by_surrogate[s].get("package_id") or "")
            for s in sorted(winners_by_surrogate)
        ])
        primary = winners_by_surrogate[sorted(winners_by_surrogate)[0]]
        for winner in {id(w): w for w in winners_by_surrogate.values()}.values():
            _drop_subsumed_package(
                turnover,
                winner,
                ref_field="suppressed_by_package_id",
                note="suppressed_by_same_unit_modernization",
                winner_list_field="suppressed_package_ids",
            )
        turnover["suppressed_by_package_id"] = str(primary.get("package_id") or "")
        turnover["suppressed_by_package_ids"] = winner_ids
        dropped.add(id(turnover))
        audit["subsumptions"].append({
            **base_record,
            "winner_package_id": str(primary.get("package_id") or ""),
            "winner_package_ids": winner_ids,
            "loser_package_id": str(turnover.get("package_id") or ""),
            "rule": "modernization_suppresses_turnover",
            "surrogate_winners": {
                s: str(w.get("package_id") or "")
                for s, w in winners_by_surrogate.items()
            },
            "winner_confirmed_surrogates": sorted({
                s
                for info in mod_infos
                if info["pkg"] in winners_by_surrogate.values()
                for s in info["surrogates"]
            }),
            "winner_cost": _cost_range(primary),
        })

    # Retained repair_heavy downgrades a retained, room-overlapping
    # turnover_std — the heavy repair already covers flooring.
    retained_heavy = [
        r for r in repairs
        if id(r) not in dropped and str(r.get("pricing_tier") or "") == "repair_heavy"
    ]
    retained_std_turnovers = [
        t for t in turnovers
        if id(t) not in dropped and str(t.get("pricing_tier") or "") == "turnover_std"
    ]
    for turnover in retained_std_turnovers:
        t_surrogates, _basis = _package_confirmed_surrogates(turnover)
        for repair in retained_heavy:
            if str(repair.get("room") or "") != str(turnover.get("room") or ""):
                continue
            r_surrogates, _r_basis = _package_confirmed_surrogates(repair)
            if t_surrogates and r_surrogates and not (t_surrogates & r_surrogates):
                continue
            record = _downgrade_turnover_to_light(turnover, repair)
            if record is not None:
                audit["downgrades"].append(record)
            break

    audit["applied"] = bool(audit["subsumptions"] or audit["downgrades"])
    active = [
        pkg for pkg in packages or []
        if id(pkg) not in dropped
        and not (
            isinstance(pkg, dict)
            and pkg.get("audit_only")
            and (
                pkg.get("subsumed_by_package_id")
                or pkg.get("suppressed_by_package_id")
            )
        )
    ]
    return active, audit


def apply_same_unit_package_subsumption(
    packages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deprecated alias — subsumption is now decided on confirmed physical
    rooms. Kept for callers/tests importing the old name."""
    return apply_physical_package_subsumption(packages)


def _group_refs_by_surrogate(
    package: Dict[str, Any],
    *,
    only_confirmed: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group a package's issue refs by their room_surrogate_id.

    With only_confirmed=True, refs whose issue_id is not in the package's
    confirmed_issue_ids are filtered out (reviewed-but-not-confirmed, rejected,
    and unreviewed refs excluded). Refs with an empty surrogate id are always
    excluded.
    """
    confirmed_ids = {str(i) for i in (package.get("confirmed_issue_ids") or [])}
    by_surrogate: Dict[str, List[Dict[str, Any]]] = {}
    for evidence in package.get("evidence_items") or []:
        for ref in evidence.get("issue_refs") or []:
            if not isinstance(ref, dict):
                continue
            issue_id = str(ref.get("issue_id") or "")
            surrogate_id = str(ref.get("room_surrogate_id") or "").strip()
            if not issue_id or not surrogate_id:
                continue
            if only_confirmed and issue_id not in confirmed_ids:
                continue
            by_surrogate.setdefault(surrogate_id, []).append(ref)
    return by_surrogate


def _group_confirmed_refs_by_surrogate(
    package: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Confirmed-only view of _group_refs_by_surrogate."""
    return _group_refs_by_surrogate(package, only_confirmed=True)


def _build_expanded_bathroom_package(
    original: Dict[str, Any],
    surrogate_id: str,
    confirmed_refs_for_surrogate: List[Dict[str, Any]],
    *,
    all_refs_for_surrogate: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build one per-surrogate copy of a confirmed bathroom_modernization package.

    Restricts evidence and issue lists to the given surrogate's confirmed refs,
    clears estimate_unit_id so reconciliation absorbs by room_surrogate_id, and
    rewrites package_id to {package_type}__{estimate_unit_id}__{surrogate_id}.

    Every evidence-derived field is rebuilt from THIS surrogate's refs —
    reviewed/rejected/driver/support ids, catalog ids, review photos, photo
    count — so a clone never displays another bathroom's evidence. Lineage
    lists start empty (expansion runs before subsumption) and re-tier state is
    cleared so the per-clone re-price can run fresh.
    """
    pkg = copy.deepcopy(original)
    confirmed_ids = {str(i) for i in (original.get("confirmed_issue_ids") or [])}
    surrogate_confirmed_ids = {str(r.get("issue_id") or "") for r in confirmed_refs_for_surrogate}
    surrogate_confirmed_ids.discard("")
    surrogate_all_ids = {
        str(r.get("issue_id") or "")
        for r in (all_refs_for_surrogate or confirmed_refs_for_surrogate)
    }
    surrogate_all_ids.discard("")

    new_evidence_items: List[Dict[str, Any]] = []
    kept_photo_keys: List[str] = []
    kept_catalog_ids: List[str] = []
    for evidence in pkg.get("evidence_items") or []:
        kept_refs = [
            ref for ref in (evidence.get("issue_refs") or [])
            if isinstance(ref, dict)
            and str(ref.get("room_surrogate_id") or "").strip() == surrogate_id
            and str(ref.get("issue_id") or "") in confirmed_ids
        ]
        if not kept_refs:
            continue
        item = dict(evidence)
        item["issue_refs"] = kept_refs
        item["issue_ids"] = [str(r.get("issue_id") or "") for r in kept_refs if r.get("issue_id")]
        item["photo_keys"] = _unique_in_order([
            str(r.get("photo_key") or "") for r in kept_refs if r.get("photo_key")
        ])
        item["supporting_photo_count"] = len(item["photo_keys"])
        item["room_surrogate_id"] = surrogate_id
        new_evidence_items.append(item)
        kept_photo_keys.extend(item["photo_keys"])
        if item.get("catalog_item_id"):
            kept_catalog_ids.append(str(item["catalog_item_id"]))
    pkg["evidence_items"] = new_evidence_items
    kept_photo_keys = _unique_in_order(kept_photo_keys)

    pkg["room_surrogate_id"] = surrogate_id
    original_estimate_unit_id = str(original.get("estimate_unit_id") or "")
    pkg["estimate_unit_id"] = ""
    pkg["source_room_surrogate_ids"] = [surrogate_id]
    pkg["confirmed_issue_ids"] = [
        i for i in (original.get("confirmed_issue_ids") or [])
        if str(i) in surrogate_confirmed_ids
    ]
    pkg["supporting_issue_ids"] = [
        i for i in (original.get("supporting_issue_ids") or [])
        if str(i) in surrogate_confirmed_ids
    ]
    # Per-surrogate projections of the property-wide review/driver metadata —
    # inheriting them verbatim was the clone-integrity defect.
    pkg["supporting_issue_ids_original"] = [
        i
        for i in (
            original.get("supporting_issue_ids_original")
            or original.get("supporting_issue_ids")
            or []
        )
        if str(i) in surrogate_all_ids
    ]
    pkg["reviewed_issue_ids"] = [
        i for i in (original.get("reviewed_issue_ids") or [])
        if str(i) in surrogate_all_ids
    ]
    pkg["rejected_issue_ids"] = [
        i for i in (original.get("rejected_issue_ids") or [])
        if str(i) in surrogate_all_ids
    ]
    pkg["driver_issue_ids"] = [
        i for i in (original.get("driver_issue_ids") or [])
        if str(i) in surrogate_confirmed_ids
    ]
    pkg["support_issue_ids"] = [
        i for i in (original.get("support_issue_ids") or [])
        if str(i) in surrogate_confirmed_ids
    ]
    pkg["supporting_catalog_item_ids"] = _unique_in_order(kept_catalog_ids)
    pkg["review_photo_keys"] = kept_photo_keys[:3]
    pkg["review_image_paths"] = []
    pkg["supporting_photo_count"] = len(kept_photo_keys)
    pkg["subsumed_package_ids"] = []
    pkg["suppressed_package_ids"] = []
    pkg.pop("retier_audit", None)
    pkg.pop("evidence_items_original", None)
    pkg.pop("visible_room_count", None)
    pkg.pop("visible_room_count_evidence", None)

    package_type = str(original.get("package_type") or "")
    pkg["package_id"] = (
        f"{package_type}__{original_estimate_unit_id}__{surrogate_id}"
        if original_estimate_unit_id
        else f"{package_type}__{surrogate_id}"
    )
    pkg["expansion_source_package_id"] = str(original.get("package_id") or "")

    original_trigger = str(original.get("trigger_reason") or "").strip()
    pkg["trigger_reason"] = (
        f"{original_trigger}+bathroom_surrogate_expansion"
        if original_trigger
        else "bathroom_surrogate_expansion"
    )
    return pkg


def _derive_surrogate_builder_inputs(
    source: PackageBuilderInputs,
    surrogate_id: str,
    surrogate_confirmed_ids: set,
    trigger_reason: str,
) -> PackageBuilderInputs:
    """Restrict a merged package's builder inputs to one physical bathroom."""

    def _matches(candidate: EstimateCandidate) -> bool:
        if str(getattr(candidate, "room_surrogate_id", "") or "") == surrogate_id:
            return True
        return bool(
            {str(i) for i in (candidate.issue_ids or [])} & surrogate_confirmed_ids
        )

    return PackageBuilderInputs(
        package_type=source.package_type,
        unit_id=surrogate_id,
        supporting_candidates=[c for c in source.supporting_candidates if _matches(c)],
        drivers=[c for c in source.drivers if _matches(c)],
        supports=[c for c in source.supports if _matches(c)],
        catalog_lookup=source.catalog_lookup,
        candidate_catalog_meta=source.candidate_catalog_meta,
        trigger_reason=trigger_reason,
    )


def expand_bathroom_modernization_packages(
    packages: List[Dict[str, Any]],
    *,
    bathroom_room_count_signal: Optional[Dict[str, Any]],
    bathroom_metadata_cap: Optional[int],
    builder_inputs: Optional[Dict[str, PackageBuilderInputs]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Split a merged bathroom_modernization package into one-per-surrogate copies.

    Expansion fires when the package is already confirmed, the room-count signal
    reports `likely_multiple_visible_bathrooms`, the scraped metadata cap allows
    >=2 bathrooms, and confirmed issue refs map to multiple distinct surrogates.
    When expansion fires, the original merged package is REMOVED from the
    returned list (and demoted in place for the aliased audit list, with
    expanded_into_package_ids lineage) so totals are not double-counted.

    Each clone is re-tiered/re-priced from only its bathroom's confirmed
    evidence via the registered builder inputs. When `builder_inputs` is
    provided but has no entry for the source package, expansion is SKIPPED
    entirely — emitting clones that each carry the merged package's full price
    is the defect this rework removes. `builder_inputs=None` (unit tests)
    keeps the inherited pricing.

    Non-bathroom_modernization packages pass through untouched. Returns
    (new_packages, expansion_audit).
    """
    audit: Dict[str, Any] = {
        "expanded": False,
        "source_package_id": None,
        "produced_package_ids": [],
        "qualifying_surrogate_ids": [],
        "skipped_surrogate_ids_no_confirmed": [],
        "skipped_surrogate_ids_no_candidates": [],
        "per_surrogate": [],
        "bathroom_metadata_cap": bathroom_metadata_cap,
        "cap_applied": False,
        "signal_likely_multiple_visible_bathrooms": bool(
            (bathroom_room_count_signal or {}).get("likely_multiple_visible_bathrooms")
        ),
        "fallback_reason": None,
    }

    if not packages:
        audit["fallback_reason"] = "no_packages"
        return list(packages or []), audit

    candidates = [
        p for p in packages
        if isinstance(p, dict)
        and str(p.get("package_type") or "") == PACKAGE_TYPE_BATHROOM_MODERNIZATION
        and str(p.get("room") or "") == ROOM_BATHROOM
        and str(p.get("verification_status") or "") in ACTIVE_PACKAGE_STATUSES
        and not p.get("superseded_by_expansion")
        and not p.get("expansion_source_package_id")
    ]
    if not candidates:
        audit["fallback_reason"] = "no_confirmed_bathroom_modernization_package"
        return list(packages), audit

    signal_hot = audit["signal_likely_multiple_visible_bathrooms"]
    cap_ok = bathroom_metadata_cap is not None and bathroom_metadata_cap >= 2
    if not signal_hot or not cap_ok:
        audit["fallback_reason"] = (
            "signal_cold" if not signal_hot else "metadata_cap_below_two"
        )
        return list(packages), audit

    # Only expand the first confirmed bathroom_modernization package — there is
    # at most one per house because infer_package_candidates groups by
    # (estimate_unit_id, package_type), bathroom_primary is one unit, and
    # dedupe_same_unit_modernizations has already enforced uniqueness.
    original = candidates[0]
    original_id = str(original.get("package_id") or "")
    audit["source_package_id"] = original_id

    source_inputs: Optional[PackageBuilderInputs] = None
    if builder_inputs is not None:
        source_inputs = builder_inputs.get(original_id)
        if source_inputs is None:
            logger.warning(
                "Bathroom expansion skipped: no builder inputs registered for "
                "package_id=%s — clones cannot be re-priced per surrogate.",
                original_id,
            )
            audit["fallback_reason"] = "missing_builder_inputs"
            return list(packages), audit

    by_surrogate = _group_confirmed_refs_by_surrogate(original)
    if not by_surrogate:
        audit["fallback_reason"] = "no_confirmed_refs_with_surrogate"
        return list(packages), audit

    all_refs_by_surrogate = _group_refs_by_surrogate(original, only_confirmed=False)

    sorted_surrogates = sorted(by_surrogate.keys())
    audit["qualifying_surrogate_ids"] = sorted_surrogates

    if len(sorted_surrogates) < 2:
        audit["fallback_reason"] = "single_qualifying_surrogate"
        return list(packages), audit

    capped_surrogates = sorted_surrogates[: int(bathroom_metadata_cap)]
    audit["cap_applied"] = len(capped_surrogates) < len(sorted_surrogates)

    expanded: List[Dict[str, Any]] = []
    for surrogate_id in capped_surrogates:
        clone = _build_expanded_bathroom_package(
            original,
            surrogate_id,
            by_surrogate[surrogate_id],
            all_refs_for_surrogate=all_refs_by_surrogate.get(surrogate_id),
        )
        if source_inputs is not None:
            surrogate_confirmed_ids = {
                str(i) for i in (clone.get("confirmed_issue_ids") or [])
            }
            derived = _derive_surrogate_builder_inputs(
                source_inputs,
                surrogate_id,
                surrogate_confirmed_ids,
                str(clone.get("trigger_reason") or ""),
            )
            if not derived.supporting_candidates:
                audit["skipped_surrogate_ids_no_candidates"].append(surrogate_id)
                continue
            builder_inputs[str(clone["package_id"])] = derived
            _retier_package_from_confirmed_evidence(clone, derived)
            if clone.get("audit_only"):
                # Re-tier demoted the clone (no confirmed evidence survived) —
                # do not emit it as an active package.
                audit["skipped_surrogate_ids_no_candidates"].append(surrogate_id)
                continue
        expanded.append(clone)
        audit["per_surrogate"].append({
            "surrogate_id": surrogate_id,
            "package_id": str(clone.get("package_id") or ""),
            "pricing_tier": clone.get("pricing_tier"),
            "cost_low": int(clone.get("cost_low") or 0),
            "cost_high": int(clone.get("cost_high") or 0),
            "confirmed_issue_count": len(clone.get("confirmed_issue_ids") or []),
        })

    if len(expanded) < 2:
        # Fewer than two viable per-room packages — keep the merged original.
        audit["per_surrogate"] = []
        audit["fallback_reason"] = "insufficient_viable_expanded_packages"
        return list(packages), audit

    out: List[Dict[str, Any]] = []
    for pkg in packages:
        if pkg is original:
            out.extend(expanded)
            continue
        out.append(pkg)

    # Demote the original in place: it aliases into package_candidates_audit,
    # so the audit list shows the superseded merged package with its lineage.
    original["expanded_into_package_ids"] = [p["package_id"] for p in expanded]
    original["superseded_by_expansion"] = True
    original["estimate_eligible"] = False
    original["ui_eligible"] = False
    original["audit_only"] = True
    original["level_decision_notes"] = list(
        original.get("level_decision_notes") or []
    ) + ["expanded_into_per_surrogate_packages"]

    audit["expanded"] = True
    audit["produced_package_ids"] = [p["package_id"] for p in expanded]
    return out, audit


_STRENGTH_RANK = {
    PACKAGE_STRENGTH_STRONG: 2,
    PACKAGE_STRENGTH_MODERATE: 1,
    PACKAGE_STRENGTH_WEAK: 0,
}


def aggregate_whole_home_turnover(
    active_packages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Roll per-room turnover packages into a single property-wide bundle.

    Returns None when fewer than two distinct rooms contributed (avoids a
    noisy "whole-home turnover" derived from a single room's signals).
    """
    turnover = [
        p for p in (active_packages or [])
        if isinstance(p, dict) and p.get("package_category") == PACKAGE_CATEGORY_TURNOVER
        and p.get("package_level") != PACKAGE_LEVEL_PROPERTY
    ]
    distinct_rooms = {p.get("room") for p in turnover if p.get("room")}
    if len(distinct_rooms) < 2:
        return None

    cost_low = sum(int(p.get("cost_low") or 0) for p in turnover)
    cost_high = sum(int(p.get("cost_high") or 0) for p in turnover)
    strongest = max(
        (p.get("package_strength") for p in turnover),
        key=lambda s: _STRENGTH_RANK.get(s or "", -1),
        default=PACKAGE_STRENGTH_MODERATE,
    )
    confidence = max(
        (float(p.get("confidence_score") or 0.0) for p in turnover),
        default=0.0,
    )
    supporting_issue_ids: List[str] = []
    supporting_catalog_item_ids: List[str] = []
    contributing_package_ids: List[str] = []
    for p in turnover:
        supporting_issue_ids.extend(p.get("supporting_issue_ids") or [])
        supporting_catalog_item_ids.extend(p.get("supporting_catalog_item_ids") or [])
        if p.get("package_id"):
            contributing_package_ids.append(str(p["package_id"]))

    return {
        "package_id": "interior_paint_flooring_refresh__whole_home",
        "package_type": PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH,
        "package_category": PACKAGE_CATEGORY_TURNOVER,
        "room": ROOM_WHOLE_HOME,
        "package_level": PACKAGE_LEVEL_PROPERTY,
        "package_strength": strongest,
        "confidence_score": confidence,
        "pricing_tier": "property_turnover_aggregate",
        "pricing_profile": "interior_paint_flooring_refresh",
        "room_surrogate_id": "",
        "estimate_unit_id": "whole_home",
        "source_room_surrogate_ids": [],
        "estimate_group": ROOM_WHOLE_HOME,
        "estimate_scope": MARKETABILITY_REHAB,
        "estimate_scope_reason": "whole_home_turnover_aggregate",
        "candidate_cost_low": cost_low,
        "candidate_cost_high": cost_high,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "cost_midpoint": (cost_low + cost_high) // 2,
        "cost_model": "aggregate",
        "cost_model_source": "whole_home_turnover_sum",
        "absorption_scope": {
            "family": "whole_home",
            "groups": [],
            "trade_buckets": [],
            "components": [],
        },
        "cap_behavior": CAP_BEHAVIOR_RESPECT_GROUP_CAP,
        "absorbed_unit_member_refs": [],
        "absorbed_total_low": 0,
        "absorbed_total_high": 0,
        "replacement_delta_low": 0,
        "replacement_delta_high": 0,
        "supporting_issue_ids": _unique_in_order(supporting_issue_ids),
        "supporting_catalog_item_ids": _unique_in_order(supporting_catalog_item_ids),
        "driver_issue_ids": [],
        "support_issue_ids": _unique_in_order(supporting_issue_ids),
        "review_photo_keys": [],
        "evidence_items": [],
        "trigger_reason": "whole_home_turnover_aggregate",
        "level_decision_notes": [
            f"aggregated_from_{len(distinct_rooms)}_rooms",
            f"contributing_packages={len(turnover)}",
        ],
        "contributing_package_ids": contributing_package_ids,
        "verification_status": PACKAGE_VERIFICATION_CONFIRMED_BY_RULE,
        "confirmed_issue_ids": _unique_in_order(supporting_issue_ids),
        "rejected_issue_ids": [],
        "evidence_summary": "Aggregated from per-room turnover packages.",
        # Display-only: the aggregate's range is the SUM of its still-active
        # contributing packages, so it must contribute $0 to estimate totals —
        # reconciliation and scope rollups skip it via estimate_display_only.
        "estimate_display_only": True,
        "estimate_eligible": False,
        "ui_eligible": True,
        "audit_only": False,
    }


def apply_package_verifications_to_candidates(
    candidates: List[EstimateCandidate],
    package_candidates: List[Dict[str, Any]],
    package_verifications: Any = None,
    *,
    provider: str = "premium",
) -> None:
    """Gate package-eligible line items based on package verification status."""
    _estimate_packages, audit_packages = finalize_package_candidates(
        package_candidates,
        package_verifications,
        require_confirmation=False,
    )
    by_issue_id: Dict[str, Dict[str, Any]] = {}
    for package in audit_packages:
        # Index the PRE-projection list so candidates backed only by rejected
        # issue ids still find their parent package (and get invalidated below)
        # instead of silently keeping their pre-2f validity.
        issue_ids = (
            package.get("supporting_issue_ids_original")
            or package.get("supporting_issue_ids")
            or []
        )
        for issue_id in issue_ids:
            by_issue_id[str(issue_id)] = package

    for candidate in candidates or []:
        package = None
        for issue_id in candidate.issue_ids or []:
            package = by_issue_id.get(str(issue_id))
            if package:
                break
        if not package:
            continue
        status = str(package.get("verification_status") or PACKAGE_VERIFICATION_NOT_RUN)
        candidate.package_id = package.get("package_id")
        candidate.package_type = package.get("package_type")
        candidate.package_role = ""
        candidate.visual_verification_status = status
        candidate.package_verification_source = f"pass_2f:{provider}" if status != PACKAGE_VERIFICATION_NOT_RUN else "pass_2f:not_run"
        candidate.review_source = candidate.package_verification_source
        if status in ACTIVE_PACKAGE_STATUSES:
            rejected_ids = {str(i) for i in (package.get("rejected_issue_ids") or [])}
            candidate_ids = {str(i) for i in (candidate.issue_ids or [])}
            if candidate_ids and rejected_ids and candidate_ids <= rejected_ids:
                # Package survived review, but every issue behind THIS candidate
                # was rejected — the candidate must not stay a valid detection.
                candidate.is_valid_detection = False
                candidate.pass_2f_attempted = True
                candidate.pass_2f_applied = False
                candidate.pass_2f_fallback_reason = "issues_rejected_in_package_review"
                continue
            candidate.is_valid_detection = True
            candidate.pass_2f_attempted = status == PACKAGE_VERIFICATION_CONFIRMED
            candidate.pass_2f_applied = status == PACKAGE_VERIFICATION_CONFIRMED
            candidate.pass_2f_fallback_reason = (
                None if status == PACKAGE_VERIFICATION_CONFIRMED
                else f"package_{status}"
            )
        else:
            candidate.is_valid_detection = False
            candidate.pass_2f_attempted = status != PACKAGE_VERIFICATION_NOT_RUN
            candidate.pass_2f_applied = False
            candidate.pass_2f_fallback_reason = f"package_{status}"


def select_package_review_image_paths(
    package: Dict[str, Any],
    photo_key_to_path: Dict[str, Path],
    *,
    max_images: int = 3,
) -> Tuple[List[str], List[Path]]:
    keys = list(package.get("review_photo_keys") or [])
    for evidence in package.get("evidence_items") or []:
        keys.extend(evidence.get("photo_keys") or [])
    selected_keys: List[str] = []
    selected_paths: List[Path] = []
    for key in _unique_in_order([str(k) for k in keys if str(k or "").strip()]):
        path = photo_key_to_path.get(key)
        if path is None:
            continue
        path = Path(path)
        if not path.is_file():
            continue
        selected_keys.append(key)
        selected_paths.append(path)
        if len(selected_paths) >= max_images:
            break
    return selected_keys, selected_paths


def _filter_evidence_items_for_review(
    evidence_items: List[Dict[str, Any]],
    review_photo_keys: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    selected = {
        str(key).strip()
        for key in (review_photo_keys or [])
        if str(key or "").strip()
    }
    if not selected:
        return [], []

    filtered_items: List[Dict[str, Any]] = []
    reviewed_issue_ids: List[str] = []
    for item in evidence_items or []:
        if not isinstance(item, dict):
            continue
        issue_refs = [
            dict(ref)
            for ref in (item.get("issue_refs") or [])
            if isinstance(ref, dict)
            and str(ref.get("issue_id") or "").strip()
            and str(ref.get("photo_key") or "").strip() in selected
        ]
        if issue_refs:
            kept_issue_ids = _unique_in_order([
                str(ref.get("issue_id") or "").strip()
                for ref in issue_refs
                if str(ref.get("issue_id") or "").strip()
            ])
            kept_photo_keys = _unique_in_order([
                str(ref.get("photo_key") or "").strip()
                for ref in issue_refs
                if str(ref.get("photo_key") or "").strip()
            ])
            kept_observations = _unique_in_order([
                str(ref.get("observation") or "").strip()
                for ref in issue_refs
                if str(ref.get("observation") or "").strip()
            ])
        else:
            item_photo_keys = [
                str(key).strip()
                for key in (item.get("photo_keys") or [])
                if str(key or "").strip()
            ]
            if not item_photo_keys:
                continue
            kept_photo_keys = [key for key in item_photo_keys if key in selected]
            if item_photo_keys and not kept_photo_keys:
                continue
            kept_issue_ids = _unique_in_order([
                str(issue_id).strip()
                for issue_id in (item.get("issue_ids") or [])
                if str(issue_id or "").strip()
            ])
            kept_observations = list(item.get("observations") or [])
            issue_refs = []
        if not kept_issue_ids:
            continue
        next_item = dict(item)
        next_item["issue_refs"] = issue_refs
        next_item["issue_ids"] = kept_issue_ids
        next_item["photo_keys"] = kept_photo_keys
        next_item["observations"] = kept_observations
        next_item["supporting_photo_count"] = len(kept_photo_keys)
        next_item["corroboration_basis"] = (
            "multi_photo_same_issue"
            if len(kept_photo_keys) >= 2
            else "single_photo_or_untracked"
        )
        filtered_items.append(next_item)
        reviewed_issue_ids.extend(kept_issue_ids)
    return filtered_items, _unique_in_order(reviewed_issue_ids)


_PACKAGE_TYPE_VLM_LABELS = {
    PACKAGE_TYPE_KITCHEN_MODERNIZATION: "Kitchen modernization",
    PACKAGE_TYPE_KITCHEN_REPAIR: "Kitchen repair",
    PACKAGE_TYPE_BATHROOM_MODERNIZATION: "Bathroom modernization",
    PACKAGE_TYPE_BATHROOM_REPAIR: "Bathroom repair",
    PACKAGE_TYPE_BEDROOM_MODERNIZATION: "Bedroom modernization",
    PACKAGE_TYPE_BEDROOM_REPAIR: "Bedroom repair",
    PACKAGE_TYPE_LIVING_MODERNIZATION: "Living room modernization",
    PACKAGE_TYPE_LIVING_REPAIR: "Living room repair",
}


def _package_vlm_label(package: Dict[str, Any]) -> str:
    pt = str(package.get("package_type") or "")
    return _PACKAGE_TYPE_VLM_LABELS.get(pt, "Renovation package")

def prepare_pass_2f_cases(
    package_candidates: List[Dict[str, Any]],
    *,
    photo_key_to_path: Dict[str, Path],
    max_images: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Freeze package-level Pass 2f inputs once for one or more model runs."""
    from tools.scene_classifier_passes import (
        PASS_2F_PROMPT_SHA256,
        PASS_2F_PROMPT_VERSION,
        PASS_2F_ROOM_PROMPTS,
    )

    prepared_cases: List[Dict[str, Any]] = []
    trace: Dict[str, Any] = {
        "candidate_count": len(package_candidates or []),
        "vlm_eligible_count": 0,
        "rule_confirmed_count": 0,
        "not_evaluated_count": 0,
        "not_evaluated_reasons": {},
    }

    for source_package in package_candidates or []:
        package = copy.deepcopy(source_package)
        package_id = str(package.get("package_id") or "")
        package_type = str(package.get("package_type") or "")
        room = str(package.get("room") or "").lower()
        case: Dict[str, Any] = {
            "package_id": package_id,
            "package_type": package_type,
            "package_label": _package_vlm_label(package),
            "room": room,
            "source_package": package,
            "evaluation_kind": "vlm_eligible",
            "not_evaluated_reason": None,
            "rule_confirmation": None,
            "prepared_input": {
                "evidence_items": [],
                "review_photo_keys": [],
                "review_image_paths": [],
                "reviewed_issue_ids": [],
                "prompt_template_version": PASS_2F_PROMPT_VERSION,
                "prompt_template_sha256": PASS_2F_PROMPT_SHA256,
                "max_images_per_package": max_images,
            },
        }

        if package.get("package_category") == PACKAGE_CATEGORY_TURNOVER:
            supporting_issue_ids = list(package.get("supporting_issue_ids") or [])
            case["evaluation_kind"] = "rule_confirmed"
            case["rule_confirmation"] = {
                "verification_status": PACKAGE_VERIFICATION_CONFIRMED_BY_RULE,
                "confirmed_issue_ids": supporting_issue_ids,
                "rejected_issue_ids": [],
                "evidence_summary": (
                    "Turnover bundling rule fired; deterministic confirmation "
                    "without VLM review."
                ),
                "review_source": "turnover_rule_based",
            }
            case["prepared_input"]["evidence_items"] = copy.deepcopy(
                package.get("evidence_items") or []
            )
            case["prepared_input"]["review_photo_keys"] = list(
                package.get("review_photo_keys") or []
            )
            case["prepared_input"]["reviewed_issue_ids"] = supporting_issue_ids
            trace["rule_confirmed_count"] += 1
            prepared_cases.append(case)
            continue

        if room not in PASS_2F_ROOM_PROMPTS:
            case["evaluation_kind"] = "not_evaluated"
            case["not_evaluated_reason"] = "unsupported_room"
            case["prepared_input"]["evidence_items"] = copy.deepcopy(
                package.get("evidence_items") or []
            )
        else:
            image_keys, image_paths = select_package_review_image_paths(
                package,
                photo_key_to_path,
                max_images=max_images,
            )
            evidence_items, reviewed_issue_ids = _filter_evidence_items_for_review(
                list(package.get("evidence_items") or []),
                image_keys,
            )
            case["prepared_input"].update({
                "evidence_items": evidence_items,
                "review_photo_keys": image_keys,
                "review_image_paths": [str(path) for path in image_paths],
                "reviewed_issue_ids": reviewed_issue_ids,
            })
            if not image_paths:
                case["evaluation_kind"] = "not_evaluated"
                case["not_evaluated_reason"] = "no_review_images"
            elif not reviewed_issue_ids:
                case["evaluation_kind"] = "not_evaluated"
                case["not_evaluated_reason"] = "no_reviewed_issue_ids"

        if case["evaluation_kind"] == "vlm_eligible":
            trace["vlm_eligible_count"] += 1
        else:
            reason = str(case.get("not_evaluated_reason") or "unknown")
            trace["not_evaluated_count"] += 1
            reasons = trace["not_evaluated_reasons"]
            reasons[reason] = int(reasons.get(reason) or 0) + 1
        prepared_cases.append(case)

    return prepared_cases, trace


async def run_pass_2f_batch(
    package_candidates: List[Dict[str, Any]],
    *,
    vlm_client: Any,
    model_config: Dict[str, Any],
    photo_key_to_path: Dict[str, Path],
    provider: str = "premium",
    max_images: int = 3,
    strict: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Run Pass 2f once per frozen package case.

    Production remains tolerant by default. Comparison callers can opt into
    strict provider and response failures without changing production behavior.
    """
    from tools.scene_classifier_passes import run_pass_2f

    prepared_cases, preparation_trace = prepare_pass_2f_cases(
        package_candidates,
        photo_key_to_path=photo_key_to_path,
        max_images=max_images,
    )
    verifications: Dict[str, Dict[str, Any]] = {}
    trace = {
        "ran": False,
        "provider": provider,
        "model": str((model_config or {}).get("model") or "unknown"),
        "candidate_count": len(package_candidates or []),
        "attempted_count": 0,
        "confirmed_count": 0,
        "confirmed_by_rule_count": 0,
        "rejected_count": 0,
        "uncertain_count": 0,
        "no_image_count": 0,
        "preparation": preparation_trace,
    }

    for case in prepared_cases:
        package = case["source_package"]
        package_id = str(case.get("package_id") or "")
        package_type = str(case.get("package_type") or "")
        prepared_input = case["prepared_input"]
        evaluation_kind = case.get("evaluation_kind")

        if evaluation_kind == "rule_confirmed":
            rule = dict(case.get("rule_confirmation") or {})
            verifications[package_id] = {
                "package_id": package_id,
                "package_type": package_type,
                **rule,
                "review_photo_keys": list(
                    prepared_input.get("review_photo_keys") or []
                ),
                "review_image_paths": [],
            }
            trace["confirmed_by_rule_count"] += 1
            continue

        if evaluation_kind == "not_evaluated":
            reason = str(case.get("not_evaluated_reason") or "")
            room = str(case.get("room") or "")
            image_keys = list(prepared_input.get("review_photo_keys") or [])
            image_paths = list(prepared_input.get("review_image_paths") or [])
            if reason == "no_review_images":
                trace["no_image_count"] += 1
                summary = (
                    f"No representative {room} images were available for "
                    "package verification."
                )
                image_keys = []
                image_paths = []
            elif reason == "no_reviewed_issue_ids":
                trace["uncertain_count"] += 1
                summary = (
                    "Pass 2f skipped: selected review photos had no matching "
                    "package evidence."
                )
            else:
                trace["uncertain_count"] += 1
                summary = (
                    "Pass 2f skipped: no prompt template registered for "
                    f"room={room or 'unknown'!r}."
                )
                image_keys = []
                image_paths = []
            verifications[package_id] = {
                "package_id": package_id,
                "package_type": package_type,
                "verification_status": PACKAGE_VERIFICATION_UNCERTAIN,
                "confirmed_issue_ids": [],
                "rejected_issue_ids": [],
                "evidence_summary": summary,
                "review_photo_keys": image_keys,
                "review_image_paths": image_paths,
                "reviewed_issue_ids": [],
            }
            continue

        trace["ran"] = True
        trace["attempted_count"] += 1
        room = str(case.get("room") or "")
        image_keys = list(prepared_input.get("review_photo_keys") or [])
        image_paths = [
            Path(path) for path in (prepared_input.get("review_image_paths") or [])
        ]
        evidence_items = list(prepared_input.get("evidence_items") or [])
        reviewed_issue_ids = list(
            prepared_input.get("reviewed_issue_ids") or []
        )
        result = await run_pass_2f(
            image_paths=image_paths,
            vlm_client=vlm_client,
            model_config=model_config,
            room=room,
            package_id=package_id,
            package_type=package_type,
            evidence_items=evidence_items,
            package_label=str(case.get("package_label") or "Renovation package"),
            strict=strict,
        )
        record = {
            "package_id": result.package_id,
            "package_type": result.package_type,
            "verification_status": result.verification_status,
            "confirmed_issue_ids": list(result.confirmed_issue_ids or []),
            "rejected_issue_ids": list(result.rejected_issue_ids or []),
            "evidence_summary": result.evidence_summary,
            "raw_response": result.raw_response,
            "review_photo_keys": image_keys,
            "review_image_paths": [str(path) for path in image_paths],
            "reviewed_issue_ids": reviewed_issue_ids,
        }
        if room == ROOM_BATHROOM:
            record["visible_room_count"] = result.visible_room_count
            record["visible_room_count_evidence"] = (
                result.visible_room_count_evidence
            )
        verifications[package_id] = record
        key = f"{result.verification_status}_count"
        if key in trace:
            trace[key] += 1
    return verifications, trace


# ═══════════════════════════════════════════════════════════════════════════
# Reconciliation
# ═══════════════════════════════════════════════════════════════════════════

# Deterministic Phase B priority: room-level packages absorb before the
# property level; within a level, repairs own their evidence before
# modernizations broad-absorb it and turnovers come last; higher tiers first;
# package_id is the stable tie-break. This is only the tie-break policy —
# Pass 1's evidence-ownership rule is what makes assignment order-independent.
_ABSORPTION_CATEGORY_PRIORITY = {
    PACKAGE_CATEGORY_REPAIR: 0,
    PACKAGE_CATEGORY_MODERNIZATION: 1,
    PACKAGE_CATEGORY_TURNOVER: 2,
}


def _absorption_priority_key(pkg: Dict[str, Any]) -> Tuple[int, int, int, str]:
    level = 1 if pkg.get("package_level") == PACKAGE_LEVEL_PROPERTY else 0
    category = _ABSORPTION_CATEGORY_PRIORITY.get(
        str(pkg.get("package_category") or ""), 3
    )
    tier_rank = _MODERNIZATION_TIER_RANK.get(str(pkg.get("pricing_tier") or ""), -1)
    return (level, category, -tier_rank, str(pkg.get("package_id") or ""))


def _package_child_join(pkg: Dict[str, Any], child: Dict[str, Any]) -> bool:
    """True when the package and child refer to the same billable scope.

    Packages with an estimate_unit_id join on unit-id equality. Expanded
    per-surrogate packages (empty unit id) join on room_surrogate_id — plus a
    verified-membership fallback for merged children that still carry an empty
    surrogate with the package's room among their source_room_surrogate_ids.
    """
    pkg_estimate_unit_id = pkg.get("estimate_unit_id")
    if pkg_estimate_unit_id:
        return child.get("estimate_unit_id") == pkg_estimate_unit_id
    pkg_room_id = pkg.get("room_surrogate_id")
    if child.get("room_surrogate_id") == pkg_room_id:
        return True
    return bool(
        pkg_room_id
        and not child.get("room_surrogate_id")
        and pkg_room_id in (child.get("source_room_surrogate_ids") or [])
    )


def _absorb_child(
    pkg: Dict[str, Any],
    child: Dict[str, Any],
    absorption_reason: str,
    groups_out: List[Dict[str, Any]],
) -> None:
    pkg_id = pkg.get("package_id")
    child["absorbed_by_package_id"] = pkg_id
    child["absorption_reason"] = absorption_reason
    pkg["absorbed_unit_member_refs"].append({
        "child_id": child["child_id"],
        "parent_line_item_id": child["parent_line_item_id"],
        "group": child.get("group"),
        "catalog_item_id": child.get("catalog_item_id"),
        "cost_model": child.get("cost_model"),
        "cost_model_source": child.get("cost_model_source"),
        "absorption_reason": absorption_reason,
        "estimate_scope": child.get("estimate_scope"),
        "estimate_scope_reason": child.get("estimate_scope_reason"),
        "allocated_low": child["allocated_low"],
        "allocated_high": child["allocated_high"],
    })
    pkg["absorbed_total_low"] += child["allocated_low"]
    pkg["absorbed_total_high"] += child["allocated_high"]
    _mark_parent_member_absorbed(groups_out, child, pkg_id)


def _split_merged_children_for_expanded_packages(
    children: List[Dict[str, Any]],
    expanded_surrogates: set,
) -> List[Dict[str, Any]]:
    """Split multi-room merged children into per-surrogate sub-children.

    A merged bathroom child (empty room_surrogate_id, several
    source_room_surrogate_ids) can only ever join ONE expanded package via the
    membership fallback, handing that bathroom the whole merged cost. Splitting
    keeps each physical bathroom's absorbed/retained arithmetic balanced:
    allocations split exactly via _split_integer (retained + absorbed still
    equals the original), each sub-child is absorbed at most once, and each
    carries its surrogate as a distinct physical_unit_key for per-unit caps.
    issue_ids stay on every sub-child — the surrogate join gates first, so a
    sibling package never sees another bathroom's sub-child.
    """
    out: List[Dict[str, Any]] = []
    for child in children:
        source_ids = [
            str(s) for s in (child.get("source_room_surrogate_ids") or []) if str(s)
        ]
        if (
            child.get("room_surrogate_id")
            or len(source_ids) < 2
            or not (set(source_ids) & expanded_surrogates)
        ):
            out.append(child)
            continue
        n = len(source_ids)
        alloc_low = _split_integer(int(child.get("allocated_low") or 0), n)
        alloc_high = _split_integer(int(child.get("allocated_high") or 0), n)
        orig_low = _split_integer(int(child.get("original_low") or 0), n)
        orig_high = _split_integer(int(child.get("original_high") or 0), n)
        for i, surrogate_id in enumerate(source_ids):
            sub = dict(child)
            sub["child_id"] = f"{child['child_id']}::{surrogate_id}"
            sub["split_from_child_id"] = child["child_id"]
            sub["room_surrogate_id"] = surrogate_id
            sub["source_room_surrogate_ids"] = [surrogate_id]
            sub["physical_unit_key"] = surrogate_id
            sub["allocated_low"] = alloc_low[i]
            sub["allocated_high"] = alloc_high[i]
            sub["original_low"] = orig_low[i]
            sub["original_high"] = orig_high[i]
            out.append(sub)
    return out


def reconcile_packages_and_estimate_units(
    groups_out: List[Dict[str, Any]],
    packages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reconcile packages against group line items.

    Mutations (in place):
      - Each line_item gets `unit_member_allocations` (list of child records).
      - Absorbed children get `absorbed_by_package_id` set.
      - Parent unit_members of absorbed children get `absorbed_by_package_id` mirrored.
      - Each package gets `absorbed_unit_member_refs`, `absorbed_total_low/high`,
        `replacement_delta_low/high` populated.

    Returns audit dict with totals, warnings, and explicit estimate buckets.
    """
    # Phase A: normalize line items into child records, indexed by group.
    # Children spanning several physical rooms that expanded per-surrogate
    # packages target are split into per-surrogate sub-children first, so each
    # bathroom's package absorbs exactly its own share and the per-unit cap
    # keys stay distinct.
    expanded_surrogates = {
        str(p.get("room_surrogate_id") or "")
        for p in packages or []
        if isinstance(p, dict)
        and p.get("expansion_source_package_id")
        and not p.get("estimate_unit_id")
    }
    expanded_surrogates.discard("")
    children_by_group: Dict[str, List[Dict[str, Any]]] = {}
    for group in groups_out or []:
        group_name = group.get("group", "other")
        group_children: List[Dict[str, Any]] = []
        for line_item in group.get("line_items", []) or []:
            children = _normalize_line_item(line_item, group_name)
            if expanded_surrogates:
                children = _split_merged_children_for_expanded_packages(
                    children, expanded_surrogates,
                )
            line_item["unit_member_allocations"] = children
            group_children.extend(children)
        children_by_group.setdefault(group_name, []).extend(group_children)

    all_children = [c for lst in children_by_group.values() for c in lst]
    warnings: List[Dict[str, Any]] = []
    _append_invalid_cost_model_warnings(all_children, warnings)

    # Phase B0: validate packages and reset absorption bookkeeping.
    for pkg in packages or []:
        pkg.setdefault("cost_model", PACKAGE_ALLOWANCE)
        pkg.setdefault("cost_model_source", COST_MODEL_SOURCE_DERIVED_PACKAGE)
        # NOTE: `absorption_scope` MUST be set by the package builder using the
        # pricing-tier key (e.g. "bathroom_repair_light"), not the package_type
        # ("bathroom_repair"). _PACKAGE_ABSORPTION_SCOPES is keyed by pricing
        # tier — a fallback using package_type would silently lookup to {} and
        # break absorption entirely. Refuse to proceed if it's missing.
        if "absorption_scope" not in pkg:
            raise KeyError(
                "Package missing absorption_scope at reconciliation: "
                f"package_id={pkg.get('package_id', '?')} "
                f"package_type={pkg.get('package_type')}"
            )
        if pkg.get("estimate_scope") not in VALID_ESTIMATE_SCOPES:
            scope, reason = classify_package_scope(
                str(pkg.get("package_type") or ""),
                [],
                str(pkg.get("trigger_reason") or ""),
            )
            pkg["estimate_scope"] = scope
            pkg["estimate_scope_reason"] = reason
        pkg.setdefault("absorbed_unit_member_refs", [])
        pkg["absorbed_total_low"] = 0
        pkg["absorbed_total_high"] = 0

    # Phase B: absorb children into packages in two passes. Pass 1 gives each
    # child to the package that holds its issue as CONFIRMED direct evidence —
    # ownership by evidence, not by package list order. Pass 2 lets packages
    # broad-absorb the remainder by absorption scope. Both passes iterate
    # packages in an explicit deterministic priority order (room-level before
    # property-level, repair < modernization < turnover, higher tier first,
    # package_id as the stable tie-break), so shuffling the input list can
    # never change an assignment. Display-only packages absorb nothing.
    billable_packages = [
        pkg for pkg in packages or [] if not _is_estimate_display_only(pkg)
    ]
    ordered_packages = sorted(billable_packages, key=_absorption_priority_key)
    supporting_by_pkg: Dict[int, set] = {}
    rejected_by_pkg: Dict[int, set] = {}
    for pkg in ordered_packages:
        rejected = {str(i) for i in (pkg.get("rejected_issue_ids") or [])}
        rejected_by_pkg[id(pkg)] = rejected
        # Rejected ids must never provide an absorption reason, even for
        # stub/test packages that bypassed the finalize-time projection.
        supporting_by_pkg[id(pkg)] = {
            str(i) for i in (pkg.get("supporting_issue_ids") or [])
        } - rejected

    for child in all_children:
        if child.get("absorbed_by_package_id") is not None:
            continue
        child_issue_ids = {str(i) for i in (child.get("issue_ids") or [])}
        if not child_issue_ids:
            continue
        for pkg in ordered_packages:
            if not _package_child_join(pkg, child):
                continue
            rejected = rejected_by_pkg[id(pkg)]
            if rejected and child_issue_ids <= rejected:
                continue
            if not (child_issue_ids & supporting_by_pkg[id(pkg)]):
                continue
            absorption_reason = _package_child_absorption_reason(
                pkg,
                child,
                supporting_by_pkg[id(pkg)],
                allow_broad=False,
            )
            if absorption_reason:
                _absorb_child(pkg, child, absorption_reason, groups_out)
                break

    for pkg in ordered_packages:
        allow_broad = bool(pkg.get("estimate_unit_id"))
        if not allow_broad:
            continue
        rejected = rejected_by_pkg[id(pkg)]
        for child in all_children:
            if child.get("absorbed_by_package_id") is not None:
                continue
            if not _package_child_join(pkg, child):
                continue
            child_issue_ids = {str(i) for i in (child.get("issue_ids") or [])}
            if rejected and child_issue_ids and child_issue_ids <= rejected:
                continue
            absorption_reason = _package_child_absorption_reason(
                pkg,
                child,
                supporting_by_pkg[id(pkg)],
                allow_broad=True,
            )
            if absorption_reason:
                _absorb_child(pkg, child, absorption_reason, groups_out)

    # Phase C: per-package replacement_delta + warnings
    for pkg in packages or []:
        raw_cap_behavior = pkg.get("cap_behavior") or CAP_BEHAVIOR_RESPECT_GROUP_CAP
        if raw_cap_behavior not in _VALID_CAP_BEHAVIORS:
            warnings.append({
                "code": "invalid_package_cap_behavior",
                "package_id": pkg.get("package_id"),
                "cap_behavior": raw_cap_behavior,
                "defaulted_to": CAP_BEHAVIOR_RESPECT_GROUP_CAP,
            })
            raw_cap_behavior = CAP_BEHAVIOR_RESPECT_GROUP_CAP
        pkg["cap_behavior"] = raw_cap_behavior
        if _is_estimate_display_only(pkg):
            # Display-only packages carry no estimate cost: no floor, no delta.
            pkg["replacement_delta_low"] = 0
            pkg["replacement_delta_high"] = 0
            continue
        # Floor: a package's cost can never undercut what it absorbed. Tier dispatch
        # picks a cost range without seeing absorbed totals (those are only known
        # here in Phase C). If a sub-tier package absorbed a costly catalog item,
        # raise the package cost to match — otherwise the difference would be
        # silently lost when retained group totals clip to 0 in Phase D.
        # Smarter dispatch in Step 6 (see _resolve_*_profile) reduces how often
        # this floor engages; this is the safety net.
        absorbed_low = pkg["absorbed_total_low"]
        absorbed_high = pkg["absorbed_total_high"]
        if pkg["cost_low"] < absorbed_low or pkg["cost_high"] < absorbed_high:
            logger.warning(
                "Package cost undercut by absorbed total — applying floor. "
                "package_id=%s cost_low=%s -> %s cost_high=%s -> %s",
                pkg.get("package_id"),
                pkg["cost_low"], max(pkg["cost_low"], absorbed_low),
                pkg["cost_high"], max(pkg["cost_high"], absorbed_high),
            )
            pkg["cost_floor_applied"] = True
            pkg["cost_low"] = max(pkg["cost_low"], absorbed_low)
            pkg["cost_high"] = max(pkg["cost_high"], absorbed_high)
            # Keep cost_midpoint consistent with the floored low/high.
            pkg["cost_midpoint"] = (pkg["cost_low"] + pkg["cost_high"]) // 2
        pkg["replacement_delta_low"] = pkg["cost_low"] - absorbed_low
        pkg["replacement_delta_high"] = pkg["cost_high"] - absorbed_high

    # Phase D: recompute retained group totals using v3's stack rule
    retained_group_totals: List[Dict[str, Any]] = []
    retained_total_low = 0
    retained_total_high = 0
    for group in groups_out or []:
        group_name = group.get("group", "other")
        group_children = children_by_group.get(group_name, [])
        retained_children = [
            c for c in group_children
            if c.get("absorbed_by_package_id") is None and _is_visible_rehab_child(c)
        ]
        rt_low, rt_high, rt_per_unit = _recompute_retained_group_detailed(
            group_name, retained_children,
        )
        retained_group_totals.append({
            "group": group_name,
            "low": rt_low,
            "high": rt_high,
            "unit_count": len(rt_per_unit),
            "per_unit": rt_per_unit,
        })
        retained_total_low += rt_low
        retained_total_high += rt_high

    # Phase E: assemble explicit rehab/exposure buckets. Display-only packages
    # (whole-home turnover aggregate) are excluded from every dollar total.
    package_total_low = sum(int(pkg.get("cost_low") or 0) for pkg in billable_packages)
    package_total_high = sum(int(pkg.get("cost_high") or 0) for pkg in billable_packages)
    absorbed_total_low = sum(int(pkg.get("absorbed_total_low") or 0) for pkg in billable_packages)
    absorbed_total_high = sum(int(pkg.get("absorbed_total_high") or 0) for pkg in billable_packages)
    absorbed_member_count = sum(len(pkg.get("absorbed_unit_member_refs") or []) for pkg in billable_packages)
    package_group_reconciliation = _build_package_group_reconciliation(
        groups_out or [],
        children_by_group,
        packages or [],
        warnings,
    )
    _populate_package_absorption_audits(packages or [], all_children)
    estimate_members = _build_estimate_members(groups_out or [], children_by_group)
    reconciliation_audit = {
        "groups": _build_reconciliation_audit_groups(
            package_group_reconciliation,
            retained_group_totals,
        )
    }
    scope_rollups = _build_scope_rollups(
        groups_out or [],
        children_by_group,
        packages or [],
        package_group_reconciliation,
    )
    totals_by_scope_raw = scope_rollups["totals_by_scope_raw"]
    totals_by_scope_capped = scope_rollups["totals_by_scope_capped"]
    (
        final_rehab_required,
        final_rehab_resale_ready,
        final_rehab_full_renewal,
    ) = build_scope_headline_tiers(
        totals_by_scope_capped,
        scope_rollups["scope_width_stats"],
        rho=cost_factors.ROLLUP_CORRELATION_RHO,
    )
    package_net_delta_low = sum(
        int(a["package_net_delta"]["low"]) for a in package_group_reconciliation
    )
    package_net_delta_high = sum(
        int(a["package_net_delta"]["high"]) for a in package_group_reconciliation
    )

    visible_low = sum(
        int(a["original_group_capped"]["low"]) for a in package_group_reconciliation
    )
    visible_high = sum(
        int(a["original_group_capped"]["high"]) for a in package_group_reconciliation
    )
    existing_risk_exposure_high = sum(
        int(group.get("risk_exposure_high") or 0) for group in (groups_out or [])
    )

    package_adjusted_low = sum(
        int(a["post_cap_package_adjusted"]["low"]) for a in package_group_reconciliation
    )
    package_adjusted_high = sum(
        int(a["post_cap_package_adjusted"]["high"]) for a in package_group_reconciliation
    )

    visible_rehab = {
        "low": visible_low,
        "high": visible_high,
        "midpoint": (visible_low + visible_high) // 2,
        "basis": "verified_visible_line_items_before_package_replacement",
    }
    package_adjusted_rehab = {
        "low": package_adjusted_low,
        "high": package_adjusted_high,
        "midpoint": (package_adjusted_low + package_adjusted_high) // 2,
        "basis": "visible_work_after_package_reconciliation",
    }
    latent_risk_exposure = {
        "low": 0,
        "high": existing_risk_exposure_high,
        "midpoint": None,
        "basis": "inspect_posture_items_and_hidden_condition_exposure",
    }
    worst_case_exposure = {
        "low": package_adjusted_low,
        "high": package_adjusted_high + existing_risk_exposure_high,
        "midpoint": None,
        "basis": "package_adjusted_rehab_plus_latent_risk_exposure",
    }
    final_rehab = {
        **package_adjusted_rehab,
        "basis": "package_adjusted_rehab",
        "source": "renovation_estimate_v4",
    }

    return {
        "absorbed_total_low": absorbed_total_low,
        "absorbed_total_high": absorbed_total_high,
        "package_total_low": package_total_low,
        "package_total_high": package_total_high,
        "net_delta_low": package_net_delta_low,
        "net_delta_high": package_net_delta_high,
        "absorbed_member_count": absorbed_member_count,
        "package_count": len(packages or []),
        "retained_group_totals": retained_group_totals,
        "package_group_reconciliation": package_group_reconciliation,
        "estimate_members": estimate_members,
        "reconciliation_audit": reconciliation_audit,
        "reconciliation_warnings": warnings,
        "warnings": [w["code"] for w in warnings if w.get("code")],
        "visible_rehab": visible_rehab,
        "package_adjusted_rehab": package_adjusted_rehab,
        "latent_risk_exposure": latent_risk_exposure,
        "worst_case_exposure": worst_case_exposure,
        "final_rehab": final_rehab,
        "totals_by_scope_raw": totals_by_scope_raw,
        "totals_by_scope_capped": totals_by_scope_capped,
        "final_rehab_required": final_rehab_required,
        "final_rehab_resale_ready": final_rehab_resale_ready,
        "final_rehab_full_renewal": final_rehab_full_renewal,
    }


def _populate_package_absorption_audits(
    packages: List[Dict[str, Any]],
    all_children: List[Dict[str, Any]],
) -> None:
    children_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for child in all_children or []:
        children_by_parent.setdefault(child.get("parent_line_item_id") or "", []).append(child)

    for pkg in packages or []:
        pkg_id = pkg.get("package_id")
        absorbed_children = [
            child for child in all_children or []
            if child.get("absorbed_by_package_id") == pkg_id
        ]
        absorbed_parent_ids = {
            child.get("parent_line_item_id") for child in absorbed_children
        }
        retained_partial_children: List[Dict[str, Any]] = []
        for parent_id in absorbed_parent_ids:
            siblings = children_by_parent.get(parent_id or "", [])
            if len(siblings) <= 1:
                continue
            retained_partial_children.extend(
                sibling for sibling in siblings
                if sibling.get("absorbed_by_package_id") is None
            )

        absorbed = {
            "line_items": [],
            "room_allowances": [],
            "partial_allocations": [],
            "totals": _sum_children_amount(absorbed_children),
        }
        retained = {
            "partial_allocations": [],
            "totals": _sum_children_amount(retained_partial_children),
        }

        for child in absorbed_children:
            siblings = children_by_parent.get(child.get("parent_line_item_id") or "", [])
            if len(siblings) > 1:
                _append_unique(absorbed["partial_allocations"], _child_ref(child, include_unit=True))
            elif child.get("cost_model") == ROOM_ALLOWANCE:
                _append_unique(absorbed["room_allowances"], _child_ref(child))
            else:
                _append_unique(absorbed["line_items"], _child_ref(child))

        for child in retained_partial_children:
            _append_unique(retained["partial_allocations"], _child_ref(child, include_unit=True))

        net_low = max(0, int(pkg.get("cost_low") or 0) - int(pkg.get("absorbed_total_low") or 0))
        net_high = max(0, int(pkg.get("cost_high") or 0) - int(pkg.get("absorbed_total_high") or 0))
        pkg["absorption_audit"] = {
            "absorbed": absorbed,
            "retained": retained,
            "package_net_delta": _amount(net_low, net_high),
        }


def _build_estimate_members(
    groups_out: List[Dict[str, Any]],
    children_by_group: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    children_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for children in (children_by_group or {}).values():
        for child in children or []:
            children_by_parent.setdefault(child.get("parent_line_item_id") or "", []).append(child)

    members: List[Dict[str, Any]] = []
    for group in groups_out or []:
        group_name = group.get("group") or "other"
        for line_item in group.get("line_items", []) or []:
            parent_id = line_item.get("estimate_unit_id") or ""
            children = children_by_parent.get(parent_id, [])
            absorbed_children = [
                child for child in children
                if child.get("absorbed_by_package_id") is not None
            ]
            retained_children = [
                child for child in children
                if child.get("absorbed_by_package_id") is None
                and _is_visible_rehab_child(child)
            ]
            package_ids = sorted({
                str(child.get("absorbed_by_package_id"))
                for child in absorbed_children
                if child.get("absorbed_by_package_id")
            })
            status = _estimate_member_status(line_item, children, absorbed_children, retained_children)
            record = {
                "estimate_unit_id": parent_id,
                "catalog_item_id": line_item.get("catalog_item_id"),
                "name": line_item.get("name"),
                "group": group_name,
                "trade_bucket": line_item.get("trade_bucket"),
                "status": status,
                "cost_model": line_item.get("cost_model") or LINE_ITEM,
                "cost_model_source": line_item.get("cost_model_source"),
                "original_amount": _sum_children_amount(children),
                "absorbed_amount": _sum_children_amount(absorbed_children),
                "retained_amount": _sum_children_amount(retained_children),
                "unit_allocations": [
                    _unit_allocation_record(child, status)
                    for child in children
                ],
            }
            if len(package_ids) == 1 and status == "absorbed":
                record["absorbed_by_package_id"] = package_ids[0]
            if package_ids:
                record["absorbed_by_package_ids"] = package_ids
            members.append(record)
    return members


def _estimate_member_status(
    line_item: Dict[str, Any],
    children: List[Dict[str, Any]],
    absorbed_children: List[Dict[str, Any]],
    retained_children: List[Dict[str, Any]],
) -> str:
    if line_item.get("is_valid_detection") is False:
        return "suppressed"
    if line_item.get("cost_model") == INSPECTION_ALLOWANCE or any(
        child.get("cost_model") == INSPECTION_ALLOWANCE for child in children
    ):
        return "risk_only"
    if absorbed_children and not retained_children:
        return "absorbed"
    if absorbed_children:
        return "partially_absorbed"
    return "retained"


def _unit_allocation_record(child: Dict[str, Any], parent_status: str) -> Dict[str, Any]:
    absorbed_by = child.get("absorbed_by_package_id")
    is_visible_retained = absorbed_by is None and _is_visible_rehab_child(child)
    status = parent_status
    if parent_status not in {"risk_only", "suppressed"}:
        status = "absorbed" if absorbed_by else "retained"
    return {
        "child_id": child.get("child_id"),
        "catalog_item_id": child.get("catalog_item_id"),
        "estimate_unit_id": child.get("estimate_unit_id"),
        "room_surrogate_id": child.get("room_surrogate_id"),
        "status": status,
        "cost_model": child.get("cost_model") or LINE_ITEM,
        "cost_model_source": child.get("cost_model_source"),
        "original_amount": _amount(
            int(child.get("allocated_low") or 0),
            int(child.get("allocated_high") or 0),
        ),
        "absorbed_amount": _amount(
            int(child.get("allocated_low") or 0) if absorbed_by else 0,
            int(child.get("allocated_high") or 0) if absorbed_by else 0,
        ),
        "retained_amount": _amount(
            int(child.get("allocated_low") or 0) if is_visible_retained else 0,
            int(child.get("allocated_high") or 0) if is_visible_retained else 0,
        ),
        "absorbed_by_package_id": absorbed_by,
    }


def _build_reconciliation_audit_groups(
    package_group_reconciliation: List[Dict[str, Any]],
    retained_group_totals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    retained_by_group = {
        item.get("group"): {
            "low": int(item.get("low") or 0),
            "high": int(item.get("high") or 0),
            "midpoint": (int(item.get("low") or 0) + int(item.get("high") or 0)) // 2,
        }
        for item in retained_group_totals or []
    }
    out: List[Dict[str, Any]] = []
    for audit in package_group_reconciliation or []:
        group_name = audit.get("group") or "other"
        out.append({
            "group": group_name,
            "original_group_raw": audit.get("original_group_raw") or {"low": 0, "high": 0},
            "original_group_capped": audit.get("original_group_capped") or {"low": 0, "high": 0},
            "retained_group": retained_by_group.get(group_name, _amount(0, 0)),
            "absorbed_total": audit.get("absorbed_total") or {"low": 0, "high": 0},
            "package_total": audit.get("package_total") or {"low": 0, "high": 0},
            "package_net_delta": audit.get("package_net_delta") or {"low": 0, "high": 0},
            "pre_cap_package_adjusted": audit.get("pre_cap_package_adjusted") or {"low": 0, "high": 0},
            "post_cap_package_adjusted": audit.get("post_cap_package_adjusted") or {"low": 0, "high": 0},
            "cap_applied_after_packages": bool(audit.get("cap_applied_after_packages")),
            "cap_override": bool(audit.get("cap_override")),
        })
    return out


def _sum_children_amount(children: List[Dict[str, Any]]) -> Dict[str, int]:
    low = sum(int(child.get("allocated_low") or 0) for child in children or [])
    high = sum(int(child.get("allocated_high") or 0) for child in children or [])
    return _amount(low, high)


def _amount(low: int, high: int) -> Dict[str, int]:
    return {
        "low": int(low),
        "high": int(high),
        "midpoint": (int(low) + int(high)) // 2,
    }


def _child_ref(child: Dict[str, Any], *, include_unit: bool = False) -> str:
    catalog_id = str(child.get("catalog_item_id") or child.get("parent_line_item_id") or "")
    if include_unit:
        unit_id = child.get("estimate_unit_id") or child.get("room_surrogate_id") or ""
        return f"{catalog_id}:{unit_id}" if unit_id else catalog_id
    return catalog_id


def _append_unique(items: List[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _build_scope_rollups(
    groups_out: List[Dict[str, Any]],
    children_by_group: Dict[str, List[Dict[str, Any]]],
    packages: List[Dict[str, Any]],
    package_group_reconciliation: List[Dict[str, Any]],
) -> Dict[str, Any]:
    raw_by_group: Dict[str, Dict[str, Dict[str, int]]] = {}
    # Half-width spread stats (Σw² per scope, across groups) accumulate from
    # the same RAW contributions summed below. Group caps truncate scopes
    # rather than scale contributors, so the tier builder clamps its blended
    # width to the capped half-width instead of re-deriving widths post-cap.
    sum_w2_by_scope: Dict[str, float] = {}

    def _add_width(scope: Any, low: Any, high: Any) -> None:
        w = (int(high or 0) - int(low or 0)) / 2.0
        key = normalize_scope_key(scope)
        sum_w2_by_scope[key] = sum_w2_by_scope.get(key, 0.0) + w * w

    for audit in package_group_reconciliation or []:
        group_name = audit.get("group") or "other"
        raw_by_group.setdefault(group_name, empty_scope_totals())

    for group_name, children in (children_by_group or {}).items():
        group_totals = raw_by_group.setdefault(group_name, empty_scope_totals())
        for child in children or []:
            scope = child.get("estimate_scope")
            if scope == INSPECTION_RISK or child.get("cost_model") == INSPECTION_ALLOWANCE:
                continue
            add_scope_amount(
                group_totals,
                scope,
                child.get("allocated_low", 0),
                child.get("allocated_high", 0),
            )
            _add_width(scope, child.get("allocated_low", 0), child.get("allocated_high", 0))

    for pkg in packages or []:
        scope = pkg.get("estimate_scope")
        if scope == INSPECTION_RISK:
            continue
        if _is_estimate_display_only(pkg):
            # The whole-home aggregate mirrors its contributors' summed range —
            # adding its net here is exactly the double count being prevented.
            continue
        group_name = pkg.get("estimate_group") or "other"
        group_totals = raw_by_group.setdefault(group_name, empty_scope_totals())
        net_low = max(
            0,
            int(pkg.get("cost_low") or 0) - int(pkg.get("absorbed_total_low") or 0),
        )
        net_high = max(
            0,
            int(pkg.get("cost_high") or 0) - int(pkg.get("absorbed_total_high") or 0),
        )
        add_scope_amount(group_totals, scope, net_low, net_high)
        _add_width(scope, net_low, net_high)

    totals_by_scope_raw = sum_scope_totals(raw_by_group)
    inspection_risk = {
        "low": 0,
        "high": sum(int(group.get("risk_exposure_high") or 0) for group in groups_out),
    }
    totals_by_scope_raw[INSPECTION_RISK] = dict(inspection_risk)

    caps_by_group = {
        (audit.get("group") or "other"): {
            "low": int((audit.get("post_cap_package_adjusted") or {}).get("low") or 0),
            "high": int((audit.get("post_cap_package_adjusted") or {}).get("high") or 0),
        }
        for audit in package_group_reconciliation or []
    }
    totals_by_scope_capped = allocate_capped_scope_totals(
        raw_by_group,
        caps_by_group,
        inspection_risk=inspection_risk,
    )
    # Σw needs no accumulator of its own: the raw scope totals are straight
    # sums of the same contributions, so Σw == (raw_high − raw_low) / 2.
    scope_width_stats = {
        scope: {
            "sum_w": (int(bucket.get("high") or 0) - int(bucket.get("low") or 0)) / 2.0,
            "sum_w2": sum_w2_by_scope.get(scope, 0.0),
        }
        for scope, bucket in totals_by_scope_raw.items()
        if scope != INSPECTION_RISK
    }
    return {
        "totals_by_scope_raw": totals_by_scope_raw,
        "totals_by_scope_capped": totals_by_scope_capped,
        "scope_width_stats": scope_width_stats,
    }


def _build_package_group_reconciliation(
    groups_out: List[Dict[str, Any]],
    children_by_group: Dict[str, List[Dict[str, Any]]],
    packages: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    group_records: Dict[str, Dict[str, Any]] = {}
    group_names: List[str] = []
    for group in groups_out or []:
        group_name = group.get("group", "other")
        if group_name not in group_records:
            group_records[group_name] = group
        if group_name not in group_names:
            group_names.append(group_name)
    packages_by_group: Dict[str, List[Dict[str, Any]]] = {}
    display_only_by_group: Dict[str, List[str]] = {}
    for pkg in packages or []:
        group_name = pkg.get("estimate_group") or "other"
        if _is_estimate_display_only(pkg):
            # Display-only packages contribute $0 — kept out of every sum and
            # out of cap-behavior resolution, listed on the record for audit.
            display_only_by_group.setdefault(group_name, []).append(
                str(pkg.get("package_id") or "")
            )
        else:
            packages_by_group.setdefault(group_name, []).append(pkg)
        if group_name not in group_names:
            group_names.append(group_name)

    out: List[Dict[str, Any]] = []
    for group_name in group_names:
        group = group_records.get(group_name)
        group_children = children_by_group.get(group_name, [])
        original_raw_low, original_raw_high = _original_group_raw(group, group_children)
        original_capped_low, original_capped_high = _original_group_capped(
            group_name,
            group,
            group_children,
        )
        group_packages = packages_by_group.get(group_name, [])
        package_low = sum(int(pkg.get("cost_low") or 0) for pkg in group_packages)
        package_high = sum(int(pkg.get("cost_high") or 0) for pkg in group_packages)
        absorbed_low = sum(int(pkg.get("absorbed_total_low") or 0) for pkg in group_packages)
        absorbed_high = sum(int(pkg.get("absorbed_total_high") or 0) for pkg in group_packages)
        net_low = max(0, package_low - absorbed_low)
        net_high = max(0, package_high - absorbed_high)
        pre_low = original_capped_low + net_low
        pre_high = original_capped_high + net_high

        # Selected-package floor: the group's final contribution must never
        # fall below the sum of its selected (estimate-eligible) packages'
        # verified, re-tiered ranges — neither via the group cap nor via
        # net-delta arithmetic when absorbed exceeds the capped originals.
        # Dollars a package absorbed from OTHER groups are already priced in
        # those groups' totals, so they are netted out of this group's floor
        # (an absolute floor would double-count cross-group absorption).
        selected_packages = [
            pkg for pkg in group_packages if pkg.get("estimate_eligible", True)
        ]
        selected_package_low = sum(
            int(pkg.get("cost_low") or 0) for pkg in selected_packages
        )
        selected_package_high = sum(
            int(pkg.get("cost_high") or 0) for pkg in selected_packages
        )
        absorbed_out_low = 0
        absorbed_out_high = 0
        for pkg in selected_packages:
            for ref in pkg.get("absorbed_unit_member_refs") or []:
                if str(ref.get("group") or group_name) != group_name:
                    absorbed_out_low += int(ref.get("allocated_low") or 0)
                    absorbed_out_high += int(ref.get("allocated_high") or 0)
        package_floor_low = max(0, selected_package_low - absorbed_out_low)
        package_floor_high = max(0, selected_package_high - absorbed_out_high)
        unit_count = len({
            (
                str(pkg.get("estimate_unit_id") or "")
                or str(pkg.get("room_surrogate_id") or "")
            )
            for pkg in group_packages
        }) if group_packages else 0

        cap_behavior = _dominant_cap_behavior(group_packages)
        cap_override = cap_behavior != CAP_BEHAVIOR_RESPECT_GROUP_CAP
        has_group_cap = _has_group_cap(group_name, group, group_children, group_packages)
        post_low, post_high, cap_applied, package_floor_applied = (
            _apply_post_package_group_cap(
                group_name=group_name,
                pre_low=pre_low,
                pre_high=pre_high,
                original_capped_high=original_capped_high,
                package_high=package_high,
                cap_behavior=cap_behavior,
                has_group_cap=has_group_cap,
                package_floor_low=package_floor_low,
                package_floor_high=package_floor_high,
            )
        )

        if cap_behavior == CAP_BEHAVIOR_ALLOW_ABOVE_GROUP_CAP and group_packages:
            warnings.append({
                "code": "cap_override_used",
                "group": group_name,
                "cap_behavior": cap_behavior,
                "package_ids": [
                    pkg.get("package_id") for pkg in group_packages if pkg.get("package_id")
                ],
                "pre_cap_high": pre_high,
                "group_cap_high": _group_cap_high(group_name),
            })

        record = {
            "group": group_name,
            "original_group_raw": {"low": original_raw_low, "high": original_raw_high},
            "original_group_capped": {
                "low": original_capped_low,
                "high": original_capped_high,
            },
            "absorbed_total": {"low": absorbed_low, "high": absorbed_high},
            "package_total": {"low": package_low, "high": package_high},
            "package_net_delta": {"low": net_low, "high": net_high},
            "pre_cap_package_adjusted": {"low": pre_low, "high": pre_high},
            "post_cap_package_adjusted": {"low": post_low, "high": post_high},
            "cap_applied_after_packages": cap_applied,
            "cap_override": cap_override,
            "selected_package_floor": {
                "low": package_floor_low,
                "high": package_floor_high,
            },
            "selected_package_total": {
                "low": selected_package_low,
                "high": selected_package_high,
            },
            "absorbed_out_of_group": {
                "low": absorbed_out_low,
                "high": absorbed_out_high,
            },
            "package_floor_applied": package_floor_applied,
            "unit_count": unit_count,
        }
        if display_only_by_group.get(group_name):
            record["display_only_package_ids"] = display_only_by_group[group_name]
        out.append(record)
    return out


def _original_group_raw(
    group: Optional[Dict[str, Any]],
    group_children: List[Dict[str, Any]],
) -> Tuple[int, int]:
    visible_children = _visible_rehab_children(group_children)
    if len(visible_children) != len(group_children):
        return (
            sum(int(c.get("allocated_low") or 0) for c in visible_children),
            sum(int(c.get("allocated_high") or 0) for c in visible_children),
        )
    if group and "raw_sum_low" in group and "raw_sum_high" in group:
        return (int(group.get("raw_sum_low") or 0), int(group.get("raw_sum_high") or 0))
    return (
        sum(int(c.get("allocated_low") or 0) for c in group_children),
        sum(int(c.get("allocated_high") or 0) for c in group_children),
    )


def _original_group_capped(
    group_name: str,
    group: Optional[Dict[str, Any]],
    group_children: List[Dict[str, Any]],
) -> Tuple[int, int]:
    visible_children = _visible_rehab_children(group_children)
    if len(visible_children) != len(group_children):
        return _recompute_retained_group(group_name, visible_children)
    if group and "low" in group and "high" in group:
        return (int(group.get("low") or 0), int(group.get("high") or 0))
    return _recompute_retained_group(group_name, group_children)


def _visible_rehab_children(children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [child for child in children or [] if _is_visible_rehab_child(child)]


def _is_visible_rehab_child(child: Dict[str, Any]) -> bool:
    return (
        child.get("estimate_scope") != INSPECTION_RISK
        and child.get("cost_model") != INSPECTION_ALLOWANCE
    )


def _dominant_cap_behavior(group_packages: List[Dict[str, Any]]) -> str:
    behaviors = {
        pkg.get("cap_behavior") or CAP_BEHAVIOR_RESPECT_GROUP_CAP
        for pkg in group_packages
    }
    if CAP_BEHAVIOR_ALLOW_ABOVE_GROUP_CAP in behaviors:
        return CAP_BEHAVIOR_ALLOW_ABOVE_GROUP_CAP
    if CAP_BEHAVIOR_REPLACE_GROUP_CAP in behaviors:
        return CAP_BEHAVIOR_REPLACE_GROUP_CAP
    return CAP_BEHAVIOR_RESPECT_GROUP_CAP


def _has_group_cap(
    group_name: str,
    group: Optional[Dict[str, Any]],
    group_children: List[Dict[str, Any]],
    group_packages: List[Dict[str, Any]],
) -> bool:
    if group and group.get("stack_behavior") == "group_cap":
        return True
    if any((c.get("stack_behavior") or "sum") == "group_cap" for c in group_children):
        return True
    if group_packages and group_name in GROUP_BUDGET_CAPS:
        return True
    return False


def _group_cap_high(group_name: str) -> Optional[int]:
    cap = GROUP_BUDGET_CAPS.get(group_name)
    if not cap:
        return None
    return int(cap[1])


def _apply_post_package_group_cap(
    *,
    group_name: str,
    pre_low: int,
    pre_high: int,
    original_capped_high: int,
    package_high: int,
    cap_behavior: str,
    has_group_cap: bool,
    package_floor_low: int = 0,
    package_floor_high: int = 0,
) -> Tuple[int, int, bool, bool]:
    """Cap the package-adjusted group total, then apply the selected-package
    floor so the result can never fall below the selected packages' verified
    range sum (net of what they absorbed from other groups — those dollars
    are priced there). The floor is applied AFTER the clamp on EVERY path,
    including the no-cap/allow-above early paths: net-delta arithmetic alone
    can push pre_high below the package sum when in-group absorbed exceeds
    the capped originals. Returns (post_low, post_high, cap_applied,
    floor_applied).
    """

    def _with_floor(low: int, high: int, cap_applied: bool) -> Tuple[int, int, bool, bool]:
        floored_high = max(high, package_floor_high)
        floored_low = max(low, min(package_floor_low, floored_high))
        return (
            floored_low,
            floored_high,
            cap_applied and floored_high < pre_high,
            floored_high > high or floored_low > low,
        )

    if not has_group_cap or cap_behavior == CAP_BEHAVIOR_ALLOW_ABOVE_GROUP_CAP:
        return _with_floor(min(pre_low, pre_high), pre_high, False)

    group_cap_high = _group_cap_high(group_name)
    if group_cap_high is None:
        return _with_floor(min(pre_low, pre_high), pre_high, False)

    cap_high = max(group_cap_high, original_capped_high)
    if cap_behavior == CAP_BEHAVIOR_REPLACE_GROUP_CAP:
        cap_high = max(cap_high, package_high)

    post_high = min(pre_high, cap_high)
    post_low = min(pre_low, post_high)
    return _with_floor(post_low, post_high, post_high < pre_high)


def _normalize_line_item(line_item: Dict[str, Any], group_name: str) -> List[Dict[str, Any]]:
    """Build child unit records for a single line item.

    One child per billable unit_member; cost is split exactly via _split_integer.
    Line items with no billable members get one synthesized child carrying the
    full parent cost so reconciliation never loses a line item.
    """
    parent_id = line_item.get("estimate_unit_id") or ""
    cost_low = int(line_item.get("cost_low") or 0)
    cost_high = int(line_item.get("cost_high") or 0)
    stack_behavior = line_item.get("stack_behavior") or "sum"
    unit_members = line_item.get("unit_members") or []
    billable = [m for m in unit_members if m.get("counts_toward_estimate")]

    if not billable:
        return [{
            "child_id": f"{parent_id}::__synthetic__",
            "parent_line_item_id": parent_id,
            "group": group_name,
            "catalog_item_id": line_item.get("catalog_item_id"),
            "name": line_item.get("name"),
            "trade_bucket": line_item.get("trade_bucket"),
            "cost_model": line_item.get("cost_model") or LINE_ITEM,
            "cost_model_source": line_item.get("cost_model_source"),
            "estimate_unit_id": line_item.get("billable_estimate_unit_id") or line_item.get("estimate_unit_id") or "",
            "room_surrogate_id": line_item.get("room_surrogate_id") or "",
            "source_room_surrogate_ids": list(line_item.get("source_room_surrogate_ids") or []),
            "physical_unit_key": (
                str(line_item.get("room_surrogate_id") or "")
                or str(
                    line_item.get("billable_estimate_unit_id")
                    or line_item.get("estimate_unit_id")
                    or ""
                )
            ),
            "issue_ids": list(line_item.get("source_issue_ids") or []),
            "scope_keys": [],
            "estimate_scope": line_item.get("estimate_scope"),
            "estimate_scope_reason": line_item.get("estimate_scope_reason"),
            "baseline_scope_before_posture": line_item.get("baseline_scope_before_posture"),
            "visible_required_with_inspect_posture": line_item.get(
                "visible_required_with_inspect_posture", False,
            ),
            "required_baseline_included": line_item.get(
                "required_baseline_included", False,
            ),
            "inspection_risk_added": line_item.get("inspection_risk_added", False),
            "stack_behavior": stack_behavior,
            "allocated_low": cost_low,
            "allocated_high": cost_high,
            "original_low": cost_low,
            "original_high": cost_high,
            "absorbed_by_package_id": None,
        }]

    n = len(billable)
    alloc_low = _split_integer(cost_low, n)
    alloc_high = _split_integer(cost_high, n)
    children: List[Dict[str, Any]] = []
    for i, member in enumerate(billable):
        unit_key = member.get("unit_key") or f"member_{i}"
        children.append({
            "child_id": f"{parent_id}::{unit_key}",
            "parent_line_item_id": parent_id,
            "group": group_name,
            "catalog_item_id": line_item.get("catalog_item_id"),
            "name": line_item.get("name"),
            "trade_bucket": line_item.get("trade_bucket"),
            "cost_model": (
                member.get("cost_model")
                or line_item.get("cost_model")
                or LINE_ITEM
            ),
            "cost_model_source": (
                member.get("cost_model_source")
                or line_item.get("cost_model_source")
            ),
            "estimate_unit_id": member.get("estimate_unit_id") or unit_key,
            "room_surrogate_id": member.get("room_surrogate_id") or "",
            "source_room_surrogate_ids": list(
                member.get("source_room_surrogate_ids")
                or member.get("room_surrogate_ids")
                or []
            ),
            "physical_unit_key": (
                str(member.get("room_surrogate_id") or "")
                or str(member.get("estimate_unit_id") or unit_key or "")
            ),
            "issue_ids": list(member.get("issue_ids") or []),
            "scope_keys": list(member.get("estimate_scope_keys") or []),
            "estimate_scope": (
                member.get("estimate_scope") or line_item.get("estimate_scope")
            ),
            "estimate_scope_reason": (
                member.get("estimate_scope_reason")
                or line_item.get("estimate_scope_reason")
            ),
            "baseline_scope_before_posture": (
                member.get("baseline_scope_before_posture")
                or line_item.get("baseline_scope_before_posture")
            ),
            "visible_required_with_inspect_posture": (
                member.get("visible_required_with_inspect_posture")
                or line_item.get("visible_required_with_inspect_posture", False)
            ),
            "required_baseline_included": (
                member.get("required_baseline_included")
                or line_item.get("required_baseline_included", False)
            ),
            "inspection_risk_added": (
                member.get("inspection_risk_added")
                or line_item.get("inspection_risk_added", False)
            ),
            "stack_behavior": stack_behavior,
            "allocated_low": alloc_low[i],
            "allocated_high": alloc_high[i],
            "original_low": alloc_low[i],
            "original_high": alloc_high[i],
            "absorbed_by_package_id": None,
        })
    return children


def _mark_parent_member_absorbed(
    groups_out: List[Dict[str, Any]],
    child: Dict[str, Any],
    package_id: str,
) -> None:
    """Mirror absorbed_by_package_id onto the matching parent unit_member.

    Split sub-children (see _split_merged_children_for_expanded_packages) are
    partial absorptions of one member: different siblings may go to different
    packages, so the single-valued member field would misrepresent them. Those
    append to the additive absorbed_by_package_ids list instead.
    """
    parent_id = child["parent_line_item_id"]
    split_from = child.get("split_from_child_id")
    base_child_id = str(split_from or child["child_id"])
    member_unit_key = (
        base_child_id.split("::", 1)[-1] if "::" in base_child_id else None
    )
    if not member_unit_key or member_unit_key == "__synthetic__":
        return
    for group in groups_out or []:
        for li in group.get("line_items", []) or []:
            if li.get("estimate_unit_id") != parent_id:
                continue
            for m in li.get("unit_members", []) or []:
                if m.get("unit_key") != member_unit_key:
                    continue
                if split_from:
                    absorbed_ids = list(m.get("absorbed_by_package_ids") or [])
                    if package_id not in absorbed_ids:
                        absorbed_ids.append(package_id)
                    m["absorbed_by_package_ids"] = absorbed_ids
                    continue
                m["absorbed_by_package_id"] = package_id
                m["cost_model"] = child.get("cost_model")
                m["cost_model_source"] = child.get("cost_model_source")
                m.setdefault("estimate_scope", child.get("estimate_scope"))
                m.setdefault(
                    "estimate_scope_reason",
                    child.get("estimate_scope_reason"),
                )
            return


def _apply_group_stack_rule(
    group_name: str,
    children: List[Dict[str, Any]],
) -> Tuple[int, int, bool]:
    """v3's stack-behavior rule over one set of children.

    Returns (low, high, cap_applied).
    """
    if not children:
        return (0, 0, False)
    raw_low = sum(c["allocated_low"] for c in children)
    raw_high = sum(c["allocated_high"] for c in children)
    behaviors = {c.get("stack_behavior") or "sum" for c in children}
    if "max_only" in behaviors:
        best = max(children, key=lambda c: c["allocated_high"])
        return (best["allocated_low"], best["allocated_high"], False)
    if "group_cap" in behaviors:
        cap_low, cap_high = GROUP_BUDGET_CAPS.get(group_name, (200, 5_000))
        capped_low = min(raw_low, cap_low)
        capped_high = min(raw_high, cap_high)
        best = max(children, key=lambda c: c["allocated_high"])
        capped_low = max(capped_low, best["allocated_low"])
        capped_high = max(capped_high, best["allocated_high"])
        return (capped_low, capped_high, capped_high < raw_high)
    return (raw_low, raw_high, False)


def _recompute_retained_group_detailed(
    group_name: str,
    retained_children: List[Dict[str, Any]],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Apply v3's stack-behavior rule PER PHYSICAL UNIT and sum the units.

    The legacy flat group cap treated three physical bathrooms as one budget,
    erasing multiplicity. Partitioning by physical_unit_key applies the cap to
    each real room separately; groups with a single unit (the common case)
    behave exactly as before. Returns (low, high, per_unit records).
    """
    if not retained_children:
        return (0, 0, [])
    partitions: Dict[str, List[Dict[str, Any]]] = {}
    for child in retained_children:
        key = str(
            child.get("physical_unit_key")
            or child.get("room_surrogate_id")
            or child.get("estimate_unit_id")
            or ""
        )
        partitions.setdefault(key, []).append(child)
    per_unit: List[Dict[str, Any]] = []
    total_low = 0
    total_high = 0
    for key in sorted(partitions):
        low, high, cap_applied = _apply_group_stack_rule(group_name, partitions[key])
        per_unit.append({
            "unit_key": key,
            "low": low,
            "high": high,
            "cap_applied": cap_applied,
        })
        total_low += low
        total_high += high
    return (total_low, total_high, per_unit)


def _recompute_retained_group(
    group_name: str,
    retained_children: List[Dict[str, Any]],
) -> Tuple[int, int]:
    low, high, _per_unit = _recompute_retained_group_detailed(
        group_name, retained_children,
    )
    return (low, high)


# ─── Issue-level final disposition audit ─────────────────────────────────────

def build_issue_disposition_audit(
    package_candidates_audit: List[Dict[str, Any]],
    packages: List[Dict[str, Any]],
    groups_out: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Map every package-reviewed issue id to its final priced representation.

    Runs after reconciliation. For each issue id seen in any package's
    reviewed/confirmed/rejected/supporting sets, resolves where its dollars
    ended up: priced inside a surviving package, absorbed as a line-item
    child, retained as a line item, covered by a subsuming winner, or —
    the defect this audit exists to catch — nowhere ("unpriced"). Confirmed
    issues resolving to "unpriced" are listed in
    confirmed_issues_without_priced_representation for a pipeline warning.
    """
    confirmed_issues: set = set()
    rejected_issues: set = set()
    all_issue_ids: set = set()
    for pkg in package_candidates_audit or []:
        if not isinstance(pkg, dict):
            continue
        confirmed_issues.update(str(i) for i in (pkg.get("confirmed_issue_ids") or []))
        rejected_issues.update(str(i) for i in (pkg.get("rejected_issue_ids") or []))
        for key in (
            "reviewed_issue_ids",
            "confirmed_issue_ids",
            "rejected_issue_ids",
            "supporting_issue_ids_original",
            "supporting_issue_ids",
        ):
            all_issue_ids.update(str(i) for i in (pkg.get(key) or []))

    final_packages = [
        pkg for pkg in packages or []
        if isinstance(pkg, dict)
        and pkg.get("estimate_eligible")
        and not _is_estimate_display_only(pkg)
    ]

    children: List[Dict[str, Any]] = []
    for group in groups_out or []:
        for line_item in group.get("line_items", []) or []:
            children.extend(line_item.get("unit_member_allocations") or [])

    def _confirmed_in_package(issue_id: str) -> Optional[Dict[str, Any]]:
        for pkg in final_packages:
            if issue_id in {str(i) for i in (pkg.get("confirmed_issue_ids") or [])}:
                return pkg
        return None

    def _supporting_in_package(issue_id: str) -> Optional[Dict[str, Any]]:
        for pkg in final_packages:
            if issue_id in {str(i) for i in (pkg.get("supporting_issue_ids") or [])}:
                return pkg
        return None

    def _child_holding(issue_id: str, *, absorbed: bool) -> Optional[Dict[str, Any]]:
        for child in children:
            if issue_id not in {str(i) for i in (child.get("issue_ids") or [])}:
                continue
            child_absorbed = child.get("absorbed_by_package_id") is not None
            if child_absorbed != absorbed:
                continue
            if not absorbed and not _is_visible_rehab_child(child):
                continue
            return child
        return None

    def _dropped_package_holding(issue_id: str) -> Optional[Dict[str, Any]]:
        for pkg in package_candidates_audit or []:
            if not isinstance(pkg, dict) or not pkg.get("audit_only"):
                continue
            if issue_id not in {str(i) for i in (pkg.get("confirmed_issue_ids") or [])}:
                continue
            if (
                pkg.get("subsumed_by_package_id")
                or pkg.get("suppressed_by_package_id")
                or pkg.get("superseded_by_expansion")
                or pkg.get("demotion_reason")
            ):
                return pkg
        return None

    issues: List[Dict[str, Any]] = []
    unpriced_confirmed: List[str] = []
    for issue_id in sorted(all_issue_ids):
        if issue_id in confirmed_issues:
            review_status = "confirmed"
        elif issue_id in rejected_issues:
            review_status = "rejected"
        else:
            review_status = "unreviewed"

        record: Dict[str, Any] = {
            "issue_id": issue_id,
            "review_status": review_status,
            "final_package_id": None,
            "disposition": "unpriced",
        }
        if review_status == "rejected":
            record["disposition"] = "rejected"
            issues.append(record)
            continue

        pkg = (
            _confirmed_in_package(issue_id)
            if review_status == "confirmed"
            else _supporting_in_package(issue_id)
        )
        if pkg is not None:
            record["disposition"] = "priced_in_package"
            record["final_package_id"] = str(pkg.get("package_id") or "")
            issues.append(record)
            continue
        absorbed_child = _child_holding(issue_id, absorbed=True)
        if absorbed_child is not None:
            record["disposition"] = "absorbed_child"
            record["final_package_id"] = str(
                absorbed_child.get("absorbed_by_package_id") or ""
            )
            record["child_id"] = absorbed_child.get("child_id")
            issues.append(record)
            continue
        retained_child = _child_holding(issue_id, absorbed=False)
        if retained_child is not None:
            record["disposition"] = "retained_line_item"
            record["child_id"] = retained_child.get("child_id")
            issues.append(record)
            continue
        dropped = _dropped_package_holding(issue_id)
        if dropped is not None:
            winner_id = str(
                dropped.get("subsumed_by_package_id")
                or dropped.get("suppressed_by_package_id")
                or ""
            )
            if winner_id:
                record["disposition"] = "dropped_with_subsumed_package"
                record["final_package_id"] = winner_id
            else:
                record["disposition"] = "dropped_with_demoted_package"
            issues.append(record)
            if not winner_id and review_status == "confirmed":
                # Demoted with no covering winner and no surviving line item —
                # the confirmed issue has genuinely lost its priced form.
                unpriced_confirmed.append(issue_id)
            continue
        issues.append(record)
        if review_status == "confirmed":
            unpriced_confirmed.append(issue_id)

    return {
        "issues": issues,
        "confirmed_issues_without_priced_representation": unpriced_confirmed,
    }
