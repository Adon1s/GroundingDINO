"""
Catalog cost model classification for renovation estimate items.

The cost model describes what a priced estimate candidate represents, which is
separate from estimate scope (required/marketability/risk) and reconciliation
status (retained/absorbed/risk-only).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


LINE_ITEM = "line_item"
ROOM_ALLOWANCE = "room_allowance"
PACKAGE_ALLOWANCE = "package_allowance"
INSPECTION_ALLOWANCE = "inspection_allowance"

VALID_COST_MODELS = frozenset({
    LINE_ITEM,
    ROOM_ALLOWANCE,
    PACKAGE_ALLOWANCE,
    INSPECTION_ALLOWANCE,
})

COST_MODEL_SOURCE_CATALOG = "catalog"
COST_MODEL_SOURCE_POSTURE_OVERRIDE = "posture_override"
COST_MODEL_SOURCE_DERIVED_UPGRADE_ROOM_ALLOWANCE = "derived_upgrade_room_allowance"
COST_MODEL_SOURCE_DERIVED_INSPECTION_STRATEGY = "derived_inspection_strategy"
COST_MODEL_SOURCE_DERIVED_PACKAGE = "derived_package"
COST_MODEL_SOURCE_LEGACY_DEFAULT = "legacy_default"
COST_MODEL_SOURCE_INVALID_CATALOG_FALLBACK = "invalid_catalog_fallback"

VALID_COST_MODEL_SOURCES = frozenset({
    COST_MODEL_SOURCE_CATALOG,
    COST_MODEL_SOURCE_POSTURE_OVERRIDE,
    COST_MODEL_SOURCE_DERIVED_UPGRADE_ROOM_ALLOWANCE,
    COST_MODEL_SOURCE_DERIVED_INSPECTION_STRATEGY,
    COST_MODEL_SOURCE_DERIVED_PACKAGE,
    COST_MODEL_SOURCE_LEGACY_DEFAULT,
    COST_MODEL_SOURCE_INVALID_CATALOG_FALLBACK,
})

_INSPECT_POSTURES = {"inspect", "inspect_only"}
_ROOM_ALLOWANCE_POLICIES = {"per_kitchen", "per_bathroom", "per_room"}


def derive_cost_model(
    catalog_item: Dict[str, Any],
    candidate: Any = None,
) -> Tuple[str, str]:
    """Return ``(cost_model, cost_model_source)`` for an item/candidate pair."""
    catalog_item = catalog_item or {}

    if _candidate_has_inspect_posture(candidate):
        return INSPECTION_ALLOWANCE, COST_MODEL_SOURCE_POSTURE_OVERRIDE

    estimate = catalog_item.get("estimate") or {}
    if isinstance(estimate, dict) and estimate.get("strategy") == "inspect_only":
        return INSPECTION_ALLOWANCE, COST_MODEL_SOURCE_DERIVED_INSPECTION_STRATEGY

    explicit = catalog_item.get("cost_model")
    if explicit is None and isinstance(estimate, dict):
        explicit = estimate.get("cost_model")
    if explicit is not None:
        if explicit in VALID_COST_MODELS:
            return str(explicit), COST_MODEL_SOURCE_CATALOG
        return LINE_ITEM, COST_MODEL_SOURCE_INVALID_CATALOG_FALLBACK

    if catalog_item.get("kind") == "upgrade":
        unit_policy = estimate.get("unit_policy") if isinstance(estimate, dict) else None
        if unit_policy in _ROOM_ALLOWANCE_POLICIES:
            return ROOM_ALLOWANCE, COST_MODEL_SOURCE_DERIVED_UPGRADE_ROOM_ALLOWANCE

    catalog_id = str(catalog_item.get("id") or "")
    if catalog_id.endswith("_rehab") or "package" in catalog_item:
        return PACKAGE_ALLOWANCE, COST_MODEL_SOURCE_DERIVED_PACKAGE

    return LINE_ITEM, COST_MODEL_SOURCE_LEGACY_DEFAULT


def derive_package_cost_model() -> Tuple[str, str]:
    return PACKAGE_ALLOWANCE, COST_MODEL_SOURCE_DERIVED_PACKAGE


def _candidate_has_inspect_posture(candidate: Any) -> bool:
    if candidate is None:
        return False
    for field_name in (
        "pricing_posture",
        "effective_posture",
        "review_posture",
        "strategy",
        "default_posture",
    ):
        value = _get(candidate, field_name)
        if value in _INSPECT_POSTURES:
            return True
    estimate_meta = _get(candidate, "estimate_meta")
    if estimate_meta is not None and _get(estimate_meta, "strategy") in _INSPECT_POSTURES:
        return True
    return False


def _get(obj: Any, field_name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(field_name)
    return getattr(obj, field_name, None)
