"""
tools/rehab_packages.py

Deterministic package inference and reconciliation for renovation_estimate_v4.

Public API:
  - infer_packages(candidates, room_surrogates, issue_catalog) -> list[dict]
        Walk per-room candidates and emit kitchen / bathroom / room packages.
  - reconcile_packages_and_estimate_units(groups_out, packages) -> dict
        Normalize line items into per-billable-member child unit records,
        absorb children into matching packages all-or-nothing, recompute
        retained group totals using v3's stack-behavior rule, emit audit
        + explicit estimate buckets. Mutates groups_out and packages in place.
"""
from __future__ import annotations

import logging
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
from tools.estimate_scope import (
    INSPECTION_RISK,
    MARKETABILITY_REHAB,
    VALID_ESTIMATE_SCOPES,
    add_scope_amount,
    allocate_capped_scope_totals,
    build_required_and_resale_ready,
    classify_package_scope,
    empty_scope_totals,
    sum_scope_totals,
)
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

ROOM_REFRESH            = ("room_refresh",              750,  4_000)
ROOM_REPAIR_HEAVY       = ("room_repair_heavy",       3_000, 10_000)


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
PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH = "interior_paint_flooring_refresh"
VALID_PACKAGE_TYPES = frozenset({
    PACKAGE_TYPE_KITCHEN_MODERNIZATION,
    PACKAGE_TYPE_KITCHEN_REPAIR,
    PACKAGE_TYPE_KITCHEN_TURNOVER,
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
        "components": {"cabinets", "appliance", "plumbing", "electrical", "moisture"},
    },
    "kitchen_repair_heavy": {
        "family": "kitchen",
        "groups": {"kitchen", "flooring"},
        "trade_buckets": {"kitchen_cabinets_counters", "plumbing", "electrical", "moisture_mold", "flooring"},
        "components": {"cabinets", "appliance", "plumbing", "electrical", "moisture", "flooring"},
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
    resolved-candidate list, so candidates passed straight into `infer_packages`
    may still have `effective_posture is None`. Falling back to the catalog
    strategy mirrors v3's `resolve_effective_posture` keep-default semantics.
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
    PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH: PACKAGE_CATEGORY_TURNOVER,
}

_PACKAGE_TYPE_TO_ROOM = {
    PACKAGE_TYPE_KITCHEN_MODERNIZATION: ROOM_KITCHEN,
    PACKAGE_TYPE_KITCHEN_REPAIR: ROOM_KITCHEN,
    PACKAGE_TYPE_KITCHEN_TURNOVER: ROOM_KITCHEN,
    PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH: ROOM_WHOLE_HOME,
}


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


def is_package_eligible_catalog_item(catalog_item: Dict[str, Any]) -> bool:
    """True when a catalog item should use package-level verification."""
    return (
        catalog_package_role(catalog_item) in {PACKAGE_ROLE_DRIVER, PACKAGE_ROLE_SUPPORT}
        and catalog_package_type(catalog_item) in VALID_PACKAGE_TYPES
    )


def is_candidate_package_eligible(
    candidate: EstimateCandidate,
    catalog_lookup: Dict[str, Dict[str, Any]],
) -> bool:
    return is_package_eligible_catalog_item(catalog_lookup.get(candidate.catalog_item_id or "", {}))


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


def classify_component(candidate: EstimateCandidate) -> Optional[str]:
    """Map a candidate to a component class for package inference.

    Returns one of: cabinets, counter, kitchen_finish, appliance, vanity, tile,
    tub_shower, fixture, bath_finish, flooring, plumbing, electrical, moisture,
    paint — or None when not relevant for any package rule.
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
        return "electrical"
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


# Strong-signal sentinels: a single occurrence of these catalog IDs is enough
# to elevate a package to strong strength regardless of corroboration count.
# Extended as the kitchen catalog is re-tagged; bathroom/etc. added in later passes.
_STRONG_SIGNAL_CATALOG_IDS = frozenset({
    "missing_base_cabinets_exposed_subfloor",
})


def _has_strong_signal(candidates: List[EstimateCandidate]) -> bool:
    for candidate in candidates or []:
        cat_id = candidate.catalog_item_id or ""
        if cat_id in _STRONG_SIGNAL_CATALOG_IDS:
            return True
        if (candidate.severity or 0) >= 3 and cat_id == "outdated_kitchen_finishes":
            return True
    return False


def compute_package_strength(
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> str:
    """Classify package strength per the user's threshold rules.

    Strong:   >=2 distinct driver catalog IDs, OR 1 driver + >=2 supports,
              OR any candidate matches a strong-signal sentinel.
    Moderate: any single driver (with or without one support), OR
              >=2 supports same estimate_unit.
    Weak:    a single orphan support (or empty). Caller should suppress.
    """
    if _has_strong_signal(drivers) or _has_strong_signal(supports):
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

    scope = pkg.get("absorption_scope") or _package_absorption_scope(
        str(pkg.get("package_type") or "")
    )
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


def _candidate_evidence_item(
    candidate: EstimateCandidate,
    catalog_item: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "catalog_item_id": candidate.catalog_item_id,
        "name": candidate.catalog_item_name,
        "issue_ids": list(candidate.issue_ids or []),
        "observations": list(candidate.supporting_observations or []),
        "photo_keys": list(candidate.photo_keys or []),
        "scene_groups_seen": list(candidate.scene_groups_seen or []),
        "estimate_unit_id": _candidate_unit_id(candidate),
        "room_surrogate_id": candidate.room_surrogate_id,
        "display_class": catalog_display_class(catalog_item),
        "package_role": catalog_package_role(catalog_item),
        "package_type": catalog_package_type(catalog_item),
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
        and components.intersection({"appliance", "plumbing", "electrical"})
    ):
        notes.append("partial_or_full_driver_pattern")
        if len(components.intersection({"cabinets", "counter", "flooring", "appliance", "plumbing", "electrical"})) >= 4:
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
    heavy_components = components.intersection({"plumbing", "electrical", "moisture", "flooring"})
    if drivers and heavy_components:
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


def _resolve_pricing_profile(
    package_type: str,
    evidence: List[EstimateCandidate],
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
) -> Tuple[Tuple[str, int, int], str, List[str]]:
    """Dispatch to the right kitchen profile resolver for the package_type."""
    if package_type == PACKAGE_TYPE_KITCHEN_REPAIR:
        return _resolve_kitchen_repair_profile(evidence, drivers, supports)
    if package_type == PACKAGE_TYPE_KITCHEN_TURNOVER:
        return _resolve_kitchen_turnover_profile(evidence, drivers, supports)
    # Default — kitchen_modernization and any future modernization-family types.
    return _resolve_kitchen_modernization_profile(evidence, drivers, supports)


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
) -> Dict[str, Any]:
    spec, pricing_tier, decision_notes = _resolve_pricing_profile(
        package_type,
        supporting_candidates,
        drivers,
        supports,
    )
    pricing_profile, cost_low, cost_high = spec
    cost_model, cost_model_source = derive_package_cost_model()
    package_strength = compute_package_strength(drivers, supports)
    confidence_score = compute_initial_confidence_score(package_strength)
    package_category, room = _resolve_package_category_and_room(
        package_type, drivers, supports, catalog_lookup,
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
        cat_item = catalog_lookup.get(candidate.catalog_item_id or "", {})
        evidence_items.append(_candidate_evidence_item(candidate, cat_item))
        issue_ids.extend(candidate.issue_ids or [])
        photo_keys.extend(candidate.photo_keys or [])
        if candidate.catalog_item_id:
            cat_ids.append(candidate.catalog_item_id)
    issue_ids = _unique_in_order(issue_ids)
    cat_ids = _unique_in_order(cat_ids)
    photo_keys = _unique_in_order(photo_keys)
    return {
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


def _resolve_package_category_and_room(
    package_type: str,
    drivers: List[EstimateCandidate],
    supports: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[str, Optional[str]]:
    """Pick category and room from the first driver (or first support fallback).

    Falls back to the package_type → category/room map when catalog metadata
    is missing on the candidate's catalog entry.
    """
    for candidate in list(drivers or []) + list(supports or []):
        cat_item = catalog_lookup.get(candidate.catalog_item_id or "", {})
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
) -> List[Dict[str, Any]]:
    """Build package candidates from catalog package metadata.

    Candidates are grouped by (estimate_unit_id, package_type) and dispatched
    to the matching pricing profile resolver. Weak-strength buckets are
    suppressed and (optionally) reported through ``suppressed_out`` for audit.
    """
    catalog_lookup = _catalog_lookup(issue_catalog)
    scene_by_room = {
        str(r.get("room_surrogate_id") or ""): str(r.get("scene") or "")
        for r in (room_surrogates or [])
        if isinstance(r, dict)
    }

    by_unit_type: Dict[Tuple[str, str], List[EstimateCandidate]] = {}
    for candidate in candidates or []:
        cat_item = catalog_lookup.get(candidate.catalog_item_id or "", {})
        if not is_package_eligible_catalog_item(cat_item):
            continue
        package_type = catalog_package_type(cat_item)
        if package_type not in VALID_PACKAGE_TYPES:
            continue
        # Kitchen package types are kitchen-scoped; the property-wide
        # turnover aggregate (interior_paint_flooring_refresh) is derived
        # downstream from per-room turnover packages, not inferred here.
        if package_type == PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH:
            continue
        expected_room = _PACKAGE_TYPE_TO_ROOM.get(package_type)
        if expected_room == ROOM_KITCHEN and scene_by_room and candidate.room_surrogate_id:
            scene = scene_by_room.get(candidate.room_surrogate_id)
            if scene and scene != "kitchen":
                continue
        unit_id = _candidate_unit_id(candidate)
        by_unit_type.setdefault((unit_id, package_type), []).append(candidate)

    out: List[Dict[str, Any]] = []
    for (unit_id, package_type), unit_candidates in sorted(by_unit_type.items()):
        defect_drivers: List[EstimateCandidate] = []
        opportunity_drivers: List[EstimateCandidate] = []
        supports: List[EstimateCandidate] = []
        for candidate in unit_candidates:
            cat_item = catalog_lookup.get(candidate.catalog_item_id or "", {})
            if _is_defect_driver_role(cat_item):
                defect_drivers.append(candidate)
            elif _is_opportunity_driver(cat_item):
                opportunity_drivers.append(candidate)
            elif catalog_package_role(cat_item) == PACKAGE_ROLE_SUPPORT:
                supports.append(candidate)

        distinct_opportunity_ids = {
            c.catalog_item_id for c in opportunity_drivers if c.catalog_item_id
        }

        if defect_drivers:
            drivers = defect_drivers + opportunity_drivers
            supporting = drivers + supports
            trigger_reason = "package_driver"
        elif opportunity_drivers and (supports or len(distinct_opportunity_ids) >= 2):
            drivers = list(opportunity_drivers)
            supporting = drivers + supports
            trigger_reason = "opportunity_driver_with_corroboration"
        elif len(supports) >= 2:
            drivers = []
            supporting = supports
            trigger_reason = "multiple_package_support_same_estimate_unit"
        else:
            _suppress_blocked_opportunity_drivers(opportunity_drivers)
            if suppressed_out is not None:
                suppressed_out.append(_build_suppressed_candidate_record(
                    unit_id=unit_id,
                    package_type=package_type,
                    drivers=opportunity_drivers,
                    supports=supports,
                    reason="weak_no_qualifying_pattern",
                ))
            continue

        strength = compute_package_strength(drivers, supports)
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
        out.append(_build_package_candidate(
            package_type=package_type,
            unit_id=unit_id,
            room_surrogate_id=first.room_surrogate_id or (source_room_ids[0] if source_room_ids else ""),
            source_room_surrogate_ids=source_room_ids,
            supporting_candidates=supporting,
            drivers=drivers,
            supports=supports,
            catalog_lookup=catalog_lookup,
            trigger_reason=trigger_reason,
        ))
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
    pkg["confirmed_issue_ids"] = list(verification.get("confirmed_issue_ids") or pkg.get("confirmed_issue_ids") or [])
    pkg["rejected_issue_ids"] = list(verification.get("rejected_issue_ids") or pkg.get("rejected_issue_ids") or [])
    pkg["evidence_summary"] = str(verification.get("evidence_summary") or pkg.get("evidence_summary") or "")
    pkg["raw_pass_2f_response"] = verification.get("raw_response") or pkg.get("raw_pass_2f_response")
    pkg["review_photo_keys"] = list(verification.get("review_photo_keys") or pkg.get("review_photo_keys") or [])
    pkg["review_image_paths"] = list(verification.get("review_image_paths") or pkg.get("review_image_paths") or [])
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


def finalize_package_candidates(
    package_candidates: List[Dict[str, Any]],
    package_verifications: Any = None,
    *,
    require_confirmation: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply package verification and return (estimate_packages, audit_packages)."""
    verification_by_id = _verification_lookup(package_verifications)
    audit_packages: List[Dict[str, Any]] = []
    estimate_packages: List[Dict[str, Any]] = []
    for package in package_candidates or []:
        package_id = str(package.get("package_id") or "")
        finalized = _apply_verification_to_package(
            package,
            verification_by_id.get(package_id),
        )
        audit_packages.append(finalized)
        final_status = finalized.get("verification_status")
        if require_confirmation and final_status not in ACTIVE_PACKAGE_STATUSES:
            continue
        if final_status in ACTIVE_PACKAGE_STATUSES:
            estimate_packages.append(finalized)
    return estimate_packages, audit_packages


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
        "estimate_eligible": True,
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
        for issue_id in package.get("supporting_issue_ids") or []:
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


_PACKAGE_TYPE_VLM_LABELS = {
    PACKAGE_TYPE_KITCHEN_MODERNIZATION: "Kitchen modernization",
    PACKAGE_TYPE_KITCHEN_REPAIR: "Kitchen repair",
}


def _package_vlm_label(package: Dict[str, Any]) -> str:
    pt = str(package.get("package_type") or "")
    return _PACKAGE_TYPE_VLM_LABELS.get(pt, "Kitchen modernization")


async def run_pass_2f_batch(
    package_candidates: List[Dict[str, Any]],
    *,
    vlm_client: Any,
    model_config: Dict[str, Any],
    photo_key_to_path: Dict[str, Path],
    provider: str = "premium",
    max_images: int = 3,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Run Pass 2f once per package candidate.

    Turnover packages short-circuit: they are confirmed by deterministic
    bundling, not by VLM image verification. They receive
    ``verification_status = "confirmed_by_rule"`` to preserve trust semantics
    (``"confirmed"`` continues to mean "VLM looked and said yes").
    """
    from tools.scene_classifier_passes import run_pass_2f

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
    }
    for package in package_candidates or []:
        package_id = str(package.get("package_id") or "")

        if package.get("package_category") == PACKAGE_CATEGORY_TURNOVER:
            supporting_issue_ids = list(package.get("supporting_issue_ids") or [])
            verifications[package_id] = {
                "package_id": package_id,
                "package_type": package.get("package_type"),
                "verification_status": PACKAGE_VERIFICATION_CONFIRMED_BY_RULE,
                "confirmed_issue_ids": supporting_issue_ids,
                "rejected_issue_ids": [],
                "evidence_summary": (
                    "Turnover bundling rule fired; deterministic confirmation without VLM review."
                ),
                "review_source": "turnover_rule_based",
                "review_photo_keys": list(package.get("review_photo_keys") or []),
                "review_image_paths": [],
            }
            trace["confirmed_by_rule_count"] += 1
            continue

        image_keys, image_paths = select_package_review_image_paths(
            package,
            photo_key_to_path,
            max_images=max_images,
        )
        if not image_paths:
            trace["no_image_count"] += 1
            verifications[package_id] = {
                "package_id": package_id,
                "package_type": package.get("package_type"),
                "verification_status": PACKAGE_VERIFICATION_UNCERTAIN,
                "confirmed_issue_ids": [],
                "rejected_issue_ids": [],
                "evidence_summary": "No representative kitchen images were available for package verification.",
                "review_photo_keys": [],
                "review_image_paths": [],
            }
            continue
        trace["ran"] = True
        trace["attempted_count"] += 1
        result = await run_pass_2f(
            image_paths=image_paths,
            vlm_client=vlm_client,
            model_config=model_config,
            package_id=package_id,
            package_type=str(package.get("package_type") or ""),
            evidence_items=list(package.get("evidence_items") or []),
            package_label=_package_vlm_label(package),
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
        }
        verifications[package_id] = record
        key = f"{result.verification_status}_count"
        if key in trace:
            trace[key] += 1
    return verifications, trace


def infer_packages(
    candidates: List[EstimateCandidate],
    room_surrogates: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
    *,
    estimate_units: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Walk per-room candidates, emit kitchen/bathroom/room packages."""
    catalog_lookup: Dict[str, Dict[str, Any]] = {}
    for item in (issue_catalog.get("items") or []):
        if isinstance(item, dict) and item.get("id"):
            catalog_lookup[item["id"]] = item

    by_surrogate: Dict[str, List[EstimateCandidate]] = {}
    by_estimate_unit: Dict[str, List[EstimateCandidate]] = {}
    for c in candidates:
        if c.room_surrogate_id:
            by_surrogate.setdefault(c.room_surrogate_id, []).append(c)
        unit_id = getattr(c, "billable_estimate_unit_id", "") or c.room_surrogate_id
        if unit_id:
            by_estimate_unit.setdefault(unit_id, []).append(c)

    packages: List[Dict[str, Any]] = []

    if estimate_units is not None:
        for unit in estimate_units or []:
            unit_id = unit.get("estimate_unit_id")
            if not unit_id:
                continue
            scene = unit.get("unit_type") or ""
            room_candidates = by_estimate_unit.get(unit_id, [])
            if not room_candidates:
                continue
            source_room_ids = list(unit.get("source_room_surrogate_ids") or [])

            if scene == "kitchen":
                pkg = _infer_kitchen_package(
                    unit_id, room_candidates, catalog_lookup,
                    estimate_unit_id=unit_id,
                    source_room_surrogate_ids=source_room_ids,
                )
            elif scene == "bathroom":
                pkg = _infer_bathroom_package(
                    unit_id, room_candidates, catalog_lookup,
                    estimate_unit_id=unit_id,
                    source_room_surrogate_ids=source_room_ids,
                )
            else:
                pkg = _infer_room_refresh(
                    unit_id, scene, room_candidates, catalog_lookup,
                    estimate_unit_id=unit_id,
                    source_room_surrogate_ids=source_room_ids,
                )

            if pkg is not None:
                packages.append(pkg)
        return packages

    for surrogate in room_surrogates or []:
        sid = surrogate.get("room_surrogate_id")
        if not sid:
            continue
        scene = surrogate.get("scene") or ""
        room_candidates = by_surrogate.get(sid, [])
        if not room_candidates:
            continue

        if scene == "kitchen":
            pkg = _infer_kitchen_package(sid, room_candidates, catalog_lookup)
        elif scene == "bathroom":
            pkg = _infer_bathroom_package(sid, room_candidates, catalog_lookup)
        else:
            pkg = _infer_room_refresh(sid, scene, room_candidates, catalog_lookup)

        if pkg is not None:
            packages.append(pkg)

    return packages


def _build_package(
    spec: Tuple[str, int, int],
    *,
    room_surrogate_id: str,
    estimate_group: str,
    supporting_candidates: List[EstimateCandidate],
    trigger_reason: str,
    decision_notes: List[str],
    estimate_unit_id: Optional[str] = None,
    source_room_surrogate_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    pkg_type, cost_low, cost_high = spec
    package_unit_id = estimate_unit_id or room_surrogate_id
    cost_model, cost_model_source = derive_package_cost_model()
    estimate_scope, estimate_scope_reason = classify_package_scope(
        pkg_type,
        supporting_candidates,
        trigger_reason,
    )
    issue_ids: List[str] = []
    cat_ids: List[str] = []
    for c in supporting_candidates:
        for iid in c.issue_ids:
            if iid not in issue_ids:
                issue_ids.append(iid)
        if c.catalog_item_id and c.catalog_item_id not in cat_ids:
            cat_ids.append(c.catalog_item_id)
    return {
        "package_id": f"{pkg_type}__{package_unit_id}",
        "package_type": pkg_type,
        "room_surrogate_id": room_surrogate_id,
        "estimate_unit_id": package_unit_id,
        "source_room_surrogate_ids": list(source_room_surrogate_ids or []),
        "estimate_group": estimate_group,
        "estimate_scope": estimate_scope,
        "estimate_scope_reason": estimate_scope_reason,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "cost_midpoint": (cost_low + cost_high) // 2,
        "cost_model": cost_model,
        "cost_model_source": cost_model_source,
        "absorption_scope": _package_absorption_scope(pkg_type),
        "cap_behavior": CAP_BEHAVIOR_RESPECT_GROUP_CAP,
        "absorbed_unit_member_refs": [],
        "absorbed_total_low": 0,
        "absorbed_total_high": 0,
        "replacement_delta_low": 0,
        "replacement_delta_high": 0,
        "supporting_issue_ids": issue_ids,
        "supporting_catalog_item_ids": cat_ids,
        "trigger_reason": trigger_reason,
        "level_decision_notes": list(decision_notes),
    }


def _bucket_by_component(
    candidates: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, List[EstimateCandidate]], List[EstimateCandidate], List[EstimateCandidate]]:
    """Return (defect_drivers_by_component, dated_evidence, all_defect_drivers)."""
    drivers: Dict[str, List[EstimateCandidate]] = {}
    dated: List[EstimateCandidate] = []
    all_drivers: List[EstimateCandidate] = []
    for c in candidates:
        cat = catalog_lookup.get(c.catalog_item_id or "", {})
        if is_defect_driver(c, cat):
            all_drivers.append(c)
            comp = classify_component(c)
            if comp:
                drivers.setdefault(comp, []).append(c)
        elif is_dated_cosmetic_evidence(c, cat):
            dated.append(c)
    return drivers, dated, all_drivers


def _infer_kitchen_package(
    room_surrogate_id: str,
    candidates: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
    *,
    estimate_unit_id: Optional[str] = None,
    source_room_surrogate_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    drivers, dated, all_drivers = _bucket_by_component(candidates, catalog_lookup)
    notes: List[str] = []

    cabinets   = drivers.get("cabinets",   [])
    counter    = drivers.get("counter",    [])
    flooring   = drivers.get("flooring",   [])
    appliance  = drivers.get("appliance",  [])
    plumbing   = drivers.get("plumbing",   [])
    electrical = drivers.get("electrical", [])

    cabinets_sev3 = [c for c in cabinets if (c.severity or 0) >= 3]
    if cabinets_sev3 and counter and flooring and (appliance or plumbing or electrical):
        return _build_package(
            KITCHEN_FULL_REHAB,
            room_surrogate_id=room_surrogate_id,
            estimate_group="kitchen",
            supporting_candidates=cabinets_sev3 + counter + flooring + appliance + plumbing + electrical,
            trigger_reason="cabinets_sev3 + counter + flooring + (appliance|plumbing|electrical)",
            decision_notes=["full_rehab matched"],
            estimate_unit_id=estimate_unit_id,
            source_room_surrogate_ids=source_room_surrogate_ids,
        )
    notes.append("full_rehab not met")

    if cabinets and (counter or flooring):
        return _build_package(
            KITCHEN_PARTIAL_REHAB,
            room_surrogate_id=room_surrogate_id,
            estimate_group="kitchen",
            supporting_candidates=cabinets + counter + flooring,
            trigger_reason="cabinets + (counter|flooring)",
            decision_notes=notes + ["partial_rehab matched"],
            estimate_unit_id=estimate_unit_id,
            source_room_surrogate_ids=source_room_surrogate_ids,
        )
    notes.append("partial_rehab not met")

    if len(dated) >= 2 and len(all_drivers) >= 1:
        return _build_package(
            KITCHEN_REFRESH,
            room_surrogate_id=room_surrogate_id,
            estimate_group="kitchen",
            supporting_candidates=dated + all_drivers,
            trigger_reason="2+ dated_evidence + 1+ defect_driver",
            decision_notes=notes + ["refresh matched"],
            estimate_unit_id=estimate_unit_id,
            source_room_surrogate_ids=source_room_surrogate_ids,
        )
    notes.append("refresh not met")

    if 1 <= len(all_drivers) <= 2:
        return _build_package(
            KITCHEN_MINOR_REPAIR,
            room_surrogate_id=room_surrogate_id,
            estimate_group="kitchen",
            supporting_candidates=all_drivers,
            trigger_reason="1-2 isolated defect drivers",
            decision_notes=notes + ["minor_repair matched"],
            estimate_unit_id=estimate_unit_id,
            source_room_surrogate_ids=source_room_surrogate_ids,
        )
    return None


def _infer_bathroom_package(
    room_surrogate_id: str,
    candidates: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
    *,
    estimate_unit_id: Optional[str] = None,
    source_room_surrogate_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    drivers, dated, all_drivers = _bucket_by_component(candidates, catalog_lookup)
    notes: List[str] = []

    vanity     = drivers.get("vanity",     [])
    tile       = drivers.get("tile",       [])
    flooring   = drivers.get("flooring",   [])
    tub_shower = drivers.get("tub_shower", [])
    fixture    = drivers.get("fixture",    [])
    moisture   = drivers.get("moisture",   [])
    plumbing   = drivers.get("plumbing",   [])

    if vanity and tile and flooring and (tub_shower or fixture or moisture or plumbing):
        return _build_package(
            BATHROOM_FULL_REHAB,
            room_surrogate_id=room_surrogate_id,
            estimate_group="bathroom",
            supporting_candidates=vanity + tile + flooring + tub_shower + fixture + moisture + plumbing,
            trigger_reason="vanity + tile + flooring + (tub_shower|fixture|moisture|plumbing)",
            decision_notes=["full_rehab matched"],
            estimate_unit_id=estimate_unit_id,
            source_room_surrogate_ids=source_room_surrogate_ids,
        )
    notes.append("full_rehab not met")

    if (vanity or tile) and (flooring or tub_shower):
        return _build_package(
            BATHROOM_PARTIAL_REHAB,
            room_surrogate_id=room_surrogate_id,
            estimate_group="bathroom",
            supporting_candidates=vanity + tile + flooring + tub_shower,
            trigger_reason="(vanity|tile) + (flooring|tub_shower)",
            decision_notes=notes + ["partial_rehab matched"],
            estimate_unit_id=estimate_unit_id,
            source_room_surrogate_ids=source_room_surrogate_ids,
        )
    notes.append("partial_rehab not met")

    if len(dated) >= 2 and len(all_drivers) >= 1:
        return _build_package(
            BATHROOM_REFRESH,
            room_surrogate_id=room_surrogate_id,
            estimate_group="bathroom",
            supporting_candidates=dated + all_drivers,
            trigger_reason="2+ dated_evidence + 1+ defect_driver",
            decision_notes=notes + ["refresh matched"],
            estimate_unit_id=estimate_unit_id,
            source_room_surrogate_ids=source_room_surrogate_ids,
        )
    notes.append("refresh not met")

    if 1 <= len(all_drivers) <= 2:
        return _build_package(
            BATHROOM_MINOR_REPAIR,
            room_surrogate_id=room_surrogate_id,
            estimate_group="bathroom",
            supporting_candidates=all_drivers,
            trigger_reason="1-2 isolated defect drivers",
            decision_notes=notes + ["minor_repair matched"],
            estimate_unit_id=estimate_unit_id,
            source_room_surrogate_ids=source_room_surrogate_ids,
        )
    return None


def _infer_room_refresh(
    room_surrogate_id: str,
    scene: str,
    candidates: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
    *,
    estimate_unit_id: Optional[str] = None,
    source_room_surrogate_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Room refresh / repair-heavy for non-kitchen, non-bathroom surrogates."""
    _drivers, _dated, all_drivers = _bucket_by_component(candidates, catalog_lookup)

    minor_sev = [c for c in all_drivers if (c.severity or 0) <= 2]
    medium_sev = [c for c in all_drivers if (c.severity or 0) == 3]

    if len(minor_sev) >= 2 and len(medium_sev) >= 1:
        return _build_package(
            ROOM_REPAIR_HEAVY,
            room_surrogate_id=room_surrogate_id,
            estimate_group="other",
            supporting_candidates=minor_sev + medium_sev,
            trigger_reason="2+ minor_severity + 1+ medium_severity defect drivers",
            decision_notes=["room_repair_heavy matched"],
            estimate_unit_id=estimate_unit_id,
            source_room_surrogate_ids=source_room_surrogate_ids,
        )

    qualified_for_refresh: List[EstimateCandidate] = []
    for c in all_drivers:
        cat = catalog_lookup.get(c.catalog_item_id or "", {})
        if (c.severity or 0) <= 2 or cat.get("category") == "cosmetic":
            qualified_for_refresh.append(c)
    if len(qualified_for_refresh) >= 3:
        return _build_package(
            ROOM_REFRESH,
            room_surrogate_id=room_surrogate_id,
            estimate_group="other",
            supporting_candidates=qualified_for_refresh,
            trigger_reason="3+ qualified non-optional cosmetic-or-minor-severity defect drivers",
            decision_notes=["room_refresh matched"],
            estimate_unit_id=estimate_unit_id,
            source_room_surrogate_ids=source_room_surrogate_ids,
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Reconciliation
# ═══════════════════════════════════════════════════════════════════════════

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
    # Phase A: normalize line items into child records, indexed by group
    children_by_group: Dict[str, List[Dict[str, Any]]] = {}
    for group in groups_out or []:
        group_name = group.get("group", "other")
        group_children: List[Dict[str, Any]] = []
        for line_item in group.get("line_items", []) or []:
            children = _normalize_line_item(line_item, group_name)
            line_item["unit_member_allocations"] = children
            group_children.extend(children)
        children_by_group.setdefault(group_name, []).extend(group_children)

    all_children = [c for lst in children_by_group.values() for c in lst]
    warnings: List[Dict[str, Any]] = []
    _append_invalid_cost_model_warnings(all_children, warnings)

    # Phase B: absorb children into packages (first-come-first-absorbed)
    for pkg in packages or []:
        pkg.setdefault("cost_model", PACKAGE_ALLOWANCE)
        pkg.setdefault("cost_model_source", COST_MODEL_SOURCE_DERIVED_PACKAGE)
        pkg.setdefault(
            "absorption_scope",
            _package_absorption_scope(str(pkg.get("package_type") or "")),
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
        pkg_supporting_issue_ids = set(pkg.get("supporting_issue_ids") or [])
        pkg_room_id = pkg.get("room_surrogate_id")
        pkg_estimate_unit_id = pkg.get("estimate_unit_id")
        pkg_id = pkg.get("package_id")

        for child in all_children:
            if child.get("absorbed_by_package_id") is not None:
                continue
            if pkg_estimate_unit_id:
                if child.get("estimate_unit_id") != pkg_estimate_unit_id:
                    continue
            elif child.get("room_surrogate_id") != pkg_room_id:
                continue
            absorption_reason = _package_child_absorption_reason(
                pkg,
                child,
                pkg_supporting_issue_ids,
                allow_broad=bool(pkg_estimate_unit_id),
            )
            if not absorption_reason:
                continue
            child["absorbed_by_package_id"] = pkg_id
            child["absorption_reason"] = absorption_reason
            pkg["absorbed_unit_member_refs"].append({
                "child_id": child["child_id"],
                "parent_line_item_id": child["parent_line_item_id"],
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
        delta_low = pkg["cost_low"] - pkg["absorbed_total_low"]
        delta_high = pkg["cost_high"] - pkg["absorbed_total_high"]
        pkg["replacement_delta_low"] = delta_low
        pkg["replacement_delta_high"] = delta_high
        if delta_low < 0:
            warnings.append({
                "code": "package_total_below_absorbed_total_low",
                "package_id": pkg["package_id"],
                "absorbed_total_low": pkg["absorbed_total_low"],
                "package_cost_low": pkg["cost_low"],
            })
        if delta_high < 0:
            warnings.append({
                "code": "package_total_below_absorbed_total_high",
                "package_id": pkg["package_id"],
                "absorbed_total_high": pkg["absorbed_total_high"],
                "package_cost_high": pkg["cost_high"],
            })

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
        rt_low, rt_high = _recompute_retained_group(group_name, retained_children)
        retained_group_totals.append({"group": group_name, "low": rt_low, "high": rt_high})
        retained_total_low += rt_low
        retained_total_high += rt_high

    # Phase E: assemble explicit rehab/exposure buckets
    package_total_low = sum(int(pkg.get("cost_low") or 0) for pkg in packages or [])
    package_total_high = sum(int(pkg.get("cost_high") or 0) for pkg in packages or [])
    absorbed_total_low = sum(int(pkg.get("absorbed_total_low") or 0) for pkg in packages or [])
    absorbed_total_high = sum(int(pkg.get("absorbed_total_high") or 0) for pkg in packages or [])
    absorbed_member_count = sum(len(pkg.get("absorbed_unit_member_refs") or []) for pkg in packages or [])
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
    final_rehab_required, final_rehab_resale_ready = build_required_and_resale_ready(
        totals_by_scope_capped,
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
) -> Dict[str, Dict[str, Dict[str, int]]]:
    raw_by_group: Dict[str, Dict[str, Dict[str, int]]] = {}

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

    for pkg in packages or []:
        scope = pkg.get("estimate_scope")
        if scope == INSPECTION_RISK:
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
    return {
        "totals_by_scope_raw": totals_by_scope_raw,
        "totals_by_scope_capped": totals_by_scope_capped,
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
    for pkg in packages or []:
        group_name = pkg.get("estimate_group") or "other"
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

        cap_behavior = _dominant_cap_behavior(group_packages)
        cap_override = cap_behavior != CAP_BEHAVIOR_RESPECT_GROUP_CAP
        has_group_cap = _has_group_cap(group_name, group, group_children, group_packages)
        post_low, post_high, cap_applied = _apply_post_package_group_cap(
            group_name=group_name,
            pre_low=pre_low,
            pre_high=pre_high,
            original_capped_high=original_capped_high,
            package_high=package_high,
            cap_behavior=cap_behavior,
            has_group_cap=has_group_cap,
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

        out.append({
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
        })
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
) -> Tuple[int, int, bool]:
    if not has_group_cap or cap_behavior == CAP_BEHAVIOR_ALLOW_ABOVE_GROUP_CAP:
        return (min(pre_low, pre_high), pre_high, False)

    group_cap_high = _group_cap_high(group_name)
    if group_cap_high is None:
        return (min(pre_low, pre_high), pre_high, False)

    cap_high = max(group_cap_high, original_capped_high)
    if cap_behavior == CAP_BEHAVIOR_REPLACE_GROUP_CAP:
        cap_high = max(cap_high, package_high)

    post_high = min(pre_high, cap_high)
    post_low = min(pre_low, post_high)
    return (post_low, post_high, post_high < pre_high)


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
    """Mirror absorbed_by_package_id onto the matching parent unit_member."""
    parent_id = child["parent_line_item_id"]
    child_id = child["child_id"]
    member_unit_key = child_id.split("::", 1)[-1] if "::" in child_id else None
    if not member_unit_key or member_unit_key == "__synthetic__":
        return
    for group in groups_out or []:
        for li in group.get("line_items", []) or []:
            if li.get("estimate_unit_id") != parent_id:
                continue
            for m in li.get("unit_members", []) or []:
                if m.get("unit_key") == member_unit_key:
                    m["absorbed_by_package_id"] = package_id
                    m["cost_model"] = child.get("cost_model")
                    m["cost_model_source"] = child.get("cost_model_source")
                    m.setdefault("estimate_scope", child.get("estimate_scope"))
                    m.setdefault(
                        "estimate_scope_reason",
                        child.get("estimate_scope_reason"),
                    )
            return


def _recompute_retained_group(
    group_name: str,
    retained_children: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """Apply v3's stack-behavior rule to retained children only."""
    if not retained_children:
        return (0, 0)
    raw_low = sum(c["allocated_low"] for c in retained_children)
    raw_high = sum(c["allocated_high"] for c in retained_children)
    behaviors = {c.get("stack_behavior") or "sum" for c in retained_children}
    if "max_only" in behaviors:
        best = max(retained_children, key=lambda c: c["allocated_high"])
        return (best["allocated_low"], best["allocated_high"])
    if "group_cap" in behaviors:
        cap_low, cap_high = GROUP_BUDGET_CAPS.get(group_name, (200, 5_000))
        capped_low = min(raw_low, cap_low)
        capped_high = min(raw_high, cap_high)
        best = max(retained_children, key=lambda c: c["allocated_high"])
        capped_low = max(capped_low, best["allocated_low"])
        capped_high = max(capped_high, best["allocated_high"])
        return (capped_low, capped_high)
    return (raw_low, raw_high)
