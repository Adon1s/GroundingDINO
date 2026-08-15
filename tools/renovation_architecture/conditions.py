"""Lane -> ObservedCondition drafts for the renovation architecture.

Consumes exactly the product-filtered canonical lane the v4 estimator uses
and reuses the existing physical-identity resolvers (room surrogates,
estimate units, scope keys). One draft per (catalog item, resolved estimate
unit): repeats on one unit collapse, the same condition in distinct units
stays distinct. Stale catalog ids and issues without a photo reference are
operational failures, never `cannot_assess` — the newest catalog is the only
catalog, and a condition that cannot cite a photo cannot be visually
reviewed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tools.estimate_units import (
    _OPENING_INSTANCE_FIELDS,
    _meaningful_unit_hint,
    build_estimate_units,
)
from tools.pipeline_common import normalize_scene_group
from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    ObservedCondition,
)
from tools.renovation_architecture.ids import make_condition_id
from tools.renovation_estimate import (
    _clean_scope_component,
    _estimate_scope_key_for_issue,
)
from tools.room_surrogates import build_room_surrogates
from tools.scene_classifier_passes import PassExecutionError

FALLBACK_RESOLUTION_REASON = "no_room_surrogate_scope_room_fallback"
# build_estimate_units confidence values that leave the physical identity
# assumed rather than evidenced.
_AMBIGUOUS_UNIT_CONFIDENCE = frozenset(
    {"default_assumption", "conservative_assumption"}
)


@dataclass(frozen=True)
class ConditionDraft:
    """An ObservedCondition plus the per-issue refs the evidence stage needs."""
    condition: ObservedCondition
    evidence_refs: Tuple[Mapping[str, str], ...]


def _conditions_failure(message: str, *, code: str) -> PassExecutionError:
    return PassExecutionError("terra_conditions", "dependency", message, code=code)


def _opening_instance_hint(issue: Mapping[str, Any]) -> str:
    """First explicit opening-instance identifier on an issue, as
    "field:value" — byte-compatible with the legacy tier-1 resolution
    (tools/estimate_units._explicit_opening_hints). Empty when the issue
    carries none."""
    for field_name in _OPENING_INSTANCE_FIELDS:
        hint = _meaningful_unit_hint(issue.get(field_name))
        if hint:
            return f"{field_name}:{hint}"
    return ""


def build_observed_conditions(
    *,
    issues_flat: List[Dict[str, Any]],
    photos: Mapping[str, Any],
    property_metadata: Optional[Dict[str, Any]],
    projection: Mapping[str, Any],
    estimate_id: str,
) -> List[ConditionDraft]:
    """Group the estimate lane into one condition per (catalog item, unit)."""
    surrogates = build_room_surrogates(dict(photos or {}))
    resolution = build_estimate_units(
        photos or {},
        surrogates.get("room_surrogates", []),
        property_metadata=property_metadata,
    )
    photo_to_surrogate = surrogates.get("photo_key_to_room_surrogate_id", {}) or {}
    photo_to_unit = resolution.get("photo_to_estimate_unit_id", {}) or {}
    units_by_id = {
        unit["estimate_unit_id"]: unit
        for unit in resolution.get("estimate_units", [])
        if isinstance(unit, dict) and unit.get("estimate_unit_id")
    }
    observables = projection["observables"]

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for issue in issues_flat or []:
        catalog_item_id = issue.get("catalog_item_id")
        if not catalog_item_id:
            continue  # never catalog-matched; the lane keeps these for display
        observable = observables.get(catalog_item_id)
        if observable is None:
            raise _conditions_failure(
                f"issue {issue.get('issue_id')!r} resolves to unknown/stale "
                f"catalog id {catalog_item_id!r} — the projection accepts only "
                "the newest catalog",
                code="StaleCatalogItemId",
            )
        photo_key = str(issue.get("photo_key") or "")
        if not photo_key:
            raise _conditions_failure(
                f"issue {issue.get('issue_id')!r} ({catalog_item_id}) carries "
                "no photo reference and cannot be visually reviewed",
                code="MissingPhotoReference",
            )
        scope_key, scope_room = _estimate_scope_key_for_issue(catalog_item_id, issue)
        unit_id = photo_to_unit.get(photo_key)
        if unit_id:
            unit = units_by_id.get(unit_id) or {}
            source = "photo_estimate_unit"
            reason = str(unit.get("merge_reason") or "unrecorded_merge_reason")
            ambiguous = unit.get("confidence") in _AMBIGUOUS_UNIT_CONFIDENCE
        else:
            unit_id = scope_room
            source = "scope_room_fallback"
            reason = FALLBACK_RESOLUTION_REASON
            ambiguous = True

        entry = grouped.setdefault(
            (catalog_item_id, unit_id),
            {
                "kind": observable.get("kind"),
                "issue_ids": set(),
                "photo_keys": set(),
                "scope_keys": set(),
                "surrogate_ids": set(),
                "scene_groups": set(),
                "sources": set(),
                "reasons": set(),
                "ambiguous": False,
                "opening_hints": set(),
                "refs": {},
            },
        )
        issue_id = str(issue.get("issue_id") or f"unlabeled:{photo_key}")
        surrogate_id = str(photo_to_surrogate.get(photo_key) or "")
        scene_group = normalize_scene_group(
            _clean_scope_component(issue.get("scene_group"))
        )
        entry["issue_ids"].add(issue_id)
        entry["photo_keys"].add(photo_key)
        entry["scope_keys"].add(scope_key)
        if surrogate_id:
            entry["surrogate_ids"].add(surrogate_id)
        entry["scene_groups"].add(scene_group)
        entry["sources"].add(source)
        entry["reasons"].add(reason)
        entry["ambiguous"] = entry["ambiguous"] or ambiguous
        opening_hint = _opening_instance_hint(issue)
        if opening_hint:
            entry["opening_hints"].add(opening_hint)
        entry["refs"][(issue_id, photo_key)] = {
            "issue_id": issue_id,
            "photo_key": photo_key,
            "observation": str(
                issue.get("description") or issue.get("label") or ""
            ),
            "room_surrogate_id": surrogate_id,
        }

    drafts: List[ConditionDraft] = []
    for (catalog_item_id, unit_id), entry in sorted(grouped.items()):
        # A unit resolved through any fallback contributor keeps the honest
        # weaker source; its reason wins so the audit trail names the gap.
        if "scope_room_fallback" in entry["sources"]:
            source = "scope_room_fallback"
            reason = FALLBACK_RESOLUTION_REASON
        else:
            source = "photo_estimate_unit"
            reason = sorted(entry["reasons"])[0]
        surrogate_ids = tuple(sorted(entry["surrogate_ids"]))
        condition = ObservedCondition(
            condition_id=make_condition_id(
                estimate_id=estimate_id,
                catalog_item_id=catalog_item_id,
                estimate_unit_id=unit_id,
            ),
            schema_version=CONTRACTS_SCHEMA_VERSION,
            catalog_item_id=catalog_item_id,
            catalog_kind=str(entry["kind"]),
            scope_key=sorted(entry["scope_keys"])[0],
            estimate_unit_id=unit_id,
            room_surrogate_id=surrogate_ids[0] if surrogate_ids else "",
            scene_group=sorted(entry["scene_groups"])[0],
            issue_ids=tuple(sorted(entry["issue_ids"])),
            identity_ambiguous=bool(entry["ambiguous"]),
            source_room_surrogate_ids=surrogate_ids,
            source_scope_keys=tuple(sorted(entry["scope_keys"])),
            unit_resolution_source=source,
            unit_resolution_reason=reason,
            opening_instance_hints=tuple(sorted(entry["opening_hints"])),
        )
        refs = tuple(
            entry["refs"][key] for key in sorted(entry["refs"])
        )
        drafts.append(ConditionDraft(condition=condition, evidence_refs=refs))
    return drafts
