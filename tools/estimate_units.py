"""
estimate_units.py
-----------------
Conservative billable-unit resolution for renovation estimate candidates.

Extraction groups observations into evidence units. This module turns those
evidence units into pricing units without pretending to know geometry that the
vision pipeline has not actually measured.
"""

from __future__ import annotations

import re
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.estimate_scope import choose_scope
from tools.renovation_estimate import (
    EstimateCandidate,
    _clean_scope_component,
    _meaningful_scope_hint,
)


_GENERIC_UNIT_HINTS = {
    "", "unknown", "other", "n/a", "na", "none", "property", "whole_property",
    "room", "area", "space", "interior", "exterior", "general", "visible",
}

_KITCHEN_HINTS = {"kitchen", "kitchenette", "galley_kitchen"}
_BATHROOM_TOKENS = ("bath", "powder", "toilet", "wc", "ensuite", "en_suite")

_OPENING_INSTANCE_FIELDS = (
    "opening_id", "opening_key", "window_id", "window_key", "door_id",
    "door_key", "instance_id", "object_id", "detection_id",
)

_OPENING_SIDE_WORDS = {
    "front", "rear", "back", "left", "right", "north", "south", "east", "west",
    "garage", "entry", "patio", "sliding", "bay", "side",
}


def build_estimate_units(
    photos: list[dict],
    room_surrogates: list[dict],
    property_metadata: dict | None = None,
) -> dict:
    """
    Build billable pricing identities from deterministic room surrogates.

    ``room_surrogate_id`` remains the photo/scene grouping. ``estimate_unit_id``
    is the billable identity used by per_kitchen/per_bathroom pricing.
    """
    photo_lookup = _normalize_photo_lookup(photos)
    surrogates = [
        dict(s)
        for s in (room_surrogates or [])
        if isinstance(s, dict) and s.get("room_surrogate_id")
    ]

    estimate_units: List[Dict[str, Any]] = []
    room_to_unit: Dict[str, str] = {}
    photo_to_unit: Dict[str, str] = {}
    merge_decisions: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def add_unit(
        estimate_unit_id: str,
        unit_type: str,
        sources: List[Dict[str, Any]],
        *,
        confidence: str,
        merge_reason: str,
        decision_reason: Optional[str] = None,
        decision_confidence: str = "medium",
        multi_kitchen_evidence: bool = False,
    ) -> None:
        source_ids = [s["room_surrogate_id"] for s in sources]
        photo_ids = _unique_sorted(_surrogate_photo_ids(s) for s in sources)
        unit = {
            "estimate_unit_id": estimate_unit_id,
            "unit_type": unit_type,
            "source_room_surrogate_ids": source_ids,
            "photo_ids": photo_ids,
            "confidence": confidence,
            "merge_reason": merge_reason,
        }
        if multi_kitchen_evidence and unit_type == "kitchen":
            unit["multi_kitchen_evidence"] = True
        estimate_units.append(unit)
        for sid in source_ids:
            room_to_unit[sid] = estimate_unit_id
        for photo_id in photo_ids:
            photo_to_unit[photo_id] = estimate_unit_id
        if len(source_ids) > 1:
            merge_decisions.append({
                "from": source_ids,
                "to": estimate_unit_id,
                "reason": decision_reason or merge_reason,
                "confidence": decision_confidence,
            })

    kitchens = [s for s in surrogates if _surrogate_unit_type(s) == "kitchen"]
    bathrooms = [s for s in surrogates if _surrogate_unit_type(s) == "bathroom"]
    bedrooms = [s for s in surrogates if _surrogate_unit_type(s) == "bedroom"]
    handled = {s["room_surrogate_id"] for s in kitchens + bathrooms + bedrooms}

    if kitchens:
        multi_kitchen_evidence = _has_multi_kitchen_metadata_evidence(
            property_metadata or {}
        )
        if _should_keep_kitchens_distinct(kitchens, photo_lookup, property_metadata or {}):
            for idx, surrogate in enumerate(kitchens, start=1):
                unit_id = "kitchen_primary" if idx == 1 else f"kitchen_secondary_{idx - 1}"
                add_unit(
                    unit_id,
                    "kitchen",
                    [surrogate],
                    confidence="explicit_evidence",
                    merge_reason="multiple_kitchens_allowed_by_metadata_or_evidence",
                    multi_kitchen_evidence=multi_kitchen_evidence,
                )
        else:
            add_unit(
                "kitchen_primary",
                "kitchen",
                kitchens,
                confidence="default_assumption",
                merge_reason="single_family_default_one_kitchen",
                decision_reason="repeated_kitchen_surrogates_merged_by_default",
            )

    if bathrooms:
        bath_cap = bathroom_metadata_cap(property_metadata or {})
        if bath_cap == 1:
            add_unit(
                "bathroom_primary",
                "bathroom",
                bathrooms,
                confidence="metadata_cap",
                merge_reason="metadata_single_bath_cap",
                decision_reason="single_bath_metadata_caps_repeated_bathroom_surrogates",
                decision_confidence="high",
            )
        elif _should_keep_bathrooms_distinct(bathrooms, photo_lookup, property_metadata or {}):
            max_units = min(len(bathrooms), bath_cap) if bath_cap else len(bathrooms)
            distinct_count = max_units if max_units == len(bathrooms) else max(1, max_units - 1)
            for idx, surrogate in enumerate(bathrooms[:distinct_count], start=1):
                unit_id = "bathroom_primary" if idx == 1 else f"bathroom_{idx}"
                add_unit(
                    unit_id,
                    "bathroom",
                    [surrogate],
                    confidence="explicit_evidence",
                    merge_reason="multiple_bathrooms_allowed_by_metadata_or_evidence",
                )
            if distinct_count < len(bathrooms):
                overflow = bathrooms[distinct_count:]
                capped_unit_id = (
                    "bathroom_primary"
                    if distinct_count == 0
                    else f"bathroom_{distinct_count + 1}"
                )
                add_unit(
                    capped_unit_id,
                    "bathroom",
                    overflow,
                    confidence="metadata_cap",
                    merge_reason="bathroom_metadata_cap_applied",
                    decision_reason="bathroom_surrogates_capped_by_metadata",
                )
        else:
            reason = (
                "bathroom_metadata_allows_two_but_weak_evidence_merged"
                if bath_cap and bath_cap >= 2
                else "conservative_repeated_bathroom_merge"
            )
            add_unit(
                "bathroom_primary",
                "bathroom",
                bathrooms,
                confidence="conservative_assumption",
                merge_reason=reason,
                decision_reason="repeated_bathroom_surrogates_merged_conservatively",
            )

    if bedrooms:
        bed_cap = _bedroom_metadata_cap(property_metadata or {})
        if not bed_cap or bed_cap >= len(bedrooms):
            for surrogate in bedrooms:
                add_unit(
                    surrogate["room_surrogate_id"],
                    "bedroom",
                    [surrogate],
                    confidence="surrogate_identity",
                    merge_reason="room_surrogate_identity_default",
                )
        elif bed_cap == 1:
            add_unit(
                "bedroom_primary",
                "bedroom",
                bedrooms,
                confidence="metadata_cap",
                merge_reason="metadata_single_bedroom_cap",
                decision_reason="single_bed_metadata_caps_repeated_bedroom_surrogates",
                decision_confidence="high",
            )
        else:
            distinct_count = max(1, bed_cap - 1)
            for idx, surrogate in enumerate(bedrooms[:distinct_count], start=1):
                unit_id = "bedroom_primary" if idx == 1 else f"bedroom_{idx}"
                add_unit(
                    unit_id,
                    "bedroom",
                    [surrogate],
                    confidence="surrogate_identity",
                    merge_reason="bedroom_surrogate_identity_default",
                )
            if distinct_count < len(bedrooms):
                overflow = bedrooms[distinct_count:]
                capped_unit_id = f"bedroom_{distinct_count + 1}"
                add_unit(
                    capped_unit_id,
                    "bedroom",
                    overflow,
                    confidence="metadata_cap",
                    merge_reason="bedroom_metadata_cap_applied",
                    decision_reason="bedroom_surrogates_capped_by_metadata",
                    decision_confidence="high",
                )

    for surrogate in surrogates:
        sid = surrogate["room_surrogate_id"]
        if sid in handled:
            continue
        unit_type = _surrogate_unit_type(surrogate)
        add_unit(
            sid,
            unit_type,
            [surrogate],
            confidence="surrogate_identity",
            merge_reason="room_surrogate_identity_default",
        )

    return {
        "estimate_units": estimate_units,
        "photo_to_estimate_unit_id": photo_to_unit,
        "room_surrogate_to_estimate_unit_id": room_to_unit,
        "merge_decisions": merge_decisions,
        "warnings": warnings,
    }


def _normalize_photo_lookup(photos: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(photos, dict):
        items = photos.items()
    else:
        items = []
        for idx, photo in enumerate(photos or [], start=1):
            if not isinstance(photo, dict):
                continue
            photo_id = (
                photo.get("photo_id")
                or photo.get("photo_key")
                or ((photo.get("photo") or {}).get("photo_key"))
                or f"photo_{idx:03d}"
            )
            items.append((photo_id, photo))

    out: Dict[str, Dict[str, Any]] = {}
    for key, photo in items:
        if not isinstance(photo, dict):
            continue
        photo_id = (
            photo.get("photo_id")
            or photo.get("photo_key")
            or ((photo.get("photo") or {}).get("photo_key"))
            or key
        )
        out[str(photo_id)] = photo
    return out


def _surrogate_unit_type(surrogate: Dict[str, Any]) -> str:
    scene = str(surrogate.get("scene") or surrogate.get("scene_group") or "room")
    return "kitchen" if scene == "pantry" else scene


def _surrogate_photo_ids(surrogate: Dict[str, Any]) -> List[str]:
    return [
        str(photo_id)
        for photo_id in (
            surrogate.get("photo_ids")
            or surrogate.get("photo_keys")
            or surrogate.get("photos")
            or []
        )
        if str(photo_id or "").strip()
    ]


def _metadata_text(property_metadata: Dict[str, Any]) -> str:
    chunks: List[str] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            chunks.append(value.lower())
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, (list, tuple, set)):
            for child in value:
                visit(child)

    visit(property_metadata or {})
    return " ".join(chunks)


def _metadata_number(property_metadata: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = (property_metadata or {}).get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            parsed = float(value)
            if math.isfinite(parsed):
                return parsed
            continue
        if isinstance(value, str):
            cleaned = value.strip().replace("$", "").replace(",", "")
            if not cleaned:
                continue
            try:
                parsed = float(cleaned)
            except ValueError:
                continue
            if math.isfinite(parsed):
                return parsed
    return None


def _metadata_truthy(property_metadata: Dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = (property_metadata or {}).get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and value > 0:
            return True
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "y"}:
            return True
    return False


def _photo_text_for_surrogate(
    surrogate: Dict[str, Any],
    photo_lookup: Dict[str, Dict[str, Any]],
) -> str:
    text_fields = (
        "caption", "description", "summary", "image_summary",
        "overall_impression", "notable_features", "positives", "notes",
    )
    chunks: List[str] = []
    for photo_id in _surrogate_photo_ids(surrogate):
        photo = photo_lookup.get(photo_id) or {}
        for field_name in text_fields:
            value = photo.get(field_name)
            if isinstance(value, str):
                chunks.append(value.lower())
            elif isinstance(value, (list, tuple)):
                chunks.extend(str(item).lower() for item in value)
        scene_classifier = photo.get("scene_classifier") or {}
        if isinstance(scene_classifier, dict):
            for field_name in text_fields:
                value = scene_classifier.get(field_name)
                if isinstance(value, str):
                    chunks.append(value.lower())
                elif isinstance(value, (list, tuple)):
                    chunks.extend(str(item).lower() for item in value)
    return " ".join(chunks)


def _should_keep_kitchens_distinct(
    kitchens: List[Dict[str, Any]],
    photo_lookup: Dict[str, Dict[str, Any]],
    property_metadata: Dict[str, Any],
) -> bool:
    if len(kitchens) <= 1:
        return False
    if _has_multi_kitchen_metadata_evidence(property_metadata):
        return True
    metadata_text = _metadata_text(property_metadata)
    explicit_terms = (
        "duplex", "triplex", "multi-unit", "multi unit", "adu", "in-law",
        "mother-in-law", "second kitchen", "basement kitchen", "kitchenette",
    )
    if any(term in metadata_text for term in explicit_terms):
        return True
    photo_texts = [_photo_text_for_surrogate(s, photo_lookup) for s in kitchens]
    strong_terms = ("second kitchen", "basement kitchen", "kitchenette", "in-law", "adu")
    return sum(any(term in text for term in strong_terms) for text in photo_texts) > 1


def _has_multi_kitchen_metadata_evidence(property_metadata: Dict[str, Any]) -> bool:
    if _metadata_truthy(
        property_metadata,
        "is_multi_unit", "multi_unit", "has_adu", "adu", "has_second_kitchen",
        "second_kitchen",
    ):
        return True
    if (_metadata_number(property_metadata, "kitchen_count", "kitchens") or 0) > 1:
        return True
    return (
        _metadata_number(property_metadata, "number_of_units", "unit_count", "units") or 0
    ) > 1


def bathroom_metadata_cap(property_metadata: Dict[str, Any]) -> Optional[int]:
    full = _metadata_number(property_metadata, "full_baths", "full_bathrooms")
    half = _metadata_number(property_metadata, "half_baths", "half_bathrooms")
    if full is not None or half is not None:
        total = math.ceil((full or 0) + (half or 0))
        return total if total > 0 else None
    baths = _metadata_number(
        property_metadata,
        "bath_count", "bathrooms", "baths", "bathroom_count",
    )
    if baths is None:
        return None
    total = math.ceil(baths)
    return total if total > 0 else None


def _bedroom_metadata_cap(property_metadata: Dict[str, Any]) -> Optional[int]:
    beds = _metadata_number(
        property_metadata,
        "bed_count", "bedrooms", "beds", "bedroom_count",
    )
    if beds is None:
        return None
    total = math.ceil(beds)
    return total if total > 0 else None


def _should_keep_bathrooms_distinct(
    bathrooms: List[Dict[str, Any]],
    photo_lookup: Dict[str, Dict[str, Any]],
    property_metadata: Dict[str, Any],
) -> bool:
    if len(bathrooms) <= 1:
        return False
    if bathroom_metadata_cap(property_metadata) == 1:
        return False
    photo_texts = [_photo_text_for_surrogate(s, photo_lookup) for s in bathrooms]
    strong_terms = (
        "primary bath", "primary bathroom", "ensuite", "en suite",
        "powder room", "half bath", "guest bath", "guest bathroom",
    )
    distinct_evidence = sum(any(term in text for term in strong_terms) for text in photo_texts)
    return distinct_evidence > 1


def resolve_estimate_units(
    candidates: List[EstimateCandidate],
    issues_flat: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
) -> List[EstimateCandidate]:
    """Resolve extracted estimate candidates into conservative pricing units."""
    if not candidates:
        return []
    if all(c.unit_resolution_method for c in candidates):
        return candidates

    issues_by_id = _issues_by_id(issues_flat)

    clusters: Dict[Tuple[str, str], List[EstimateCandidate]] = {}
    for candidate in candidates:
        policy = _unit_policy(candidate)
        cluster_key = _cluster_key(candidate, policy)
        clusters.setdefault((policy, cluster_key), []).append(candidate)

    resolved: List[EstimateCandidate] = []
    for (policy, cluster_key), cluster_candidates in sorted(clusters.items()):
        handler = {
            "per_property": _resolve_per_property,
            "per_room": _resolve_per_room,
            "per_kitchen": _resolve_per_kitchen,
            "per_bathroom": _resolve_per_bathroom,
            "per_opening": _resolve_per_opening,
            "per_system": _resolve_per_system,
            "per_area": _resolve_per_area,
        }.get(policy, _resolve_per_scope)
        resolved.extend(handler(cluster_candidates, cluster_key, issues_by_id))

    resolved.sort(key=_candidate_sort_key)
    return resolved


def _unit_policy(candidate: EstimateCandidate) -> str:
    policy = getattr(candidate.estimate_meta, "unit_policy", "per_scope") or "per_scope"
    return str(policy)


def _cluster_key(candidate: EstimateCandidate, policy: str) -> str:
    if policy == "per_scope":
        return candidate.estimate_scope_key or candidate.estimate_unit_id
    return f"catalog:{candidate.catalog_item_id}|unit_policy:{policy}"


def _resolve_per_scope(
    candidates: List[EstimateCandidate],
    cluster_key: str,
    issues_by_id: Dict[str, Dict[str, Any]],
) -> List[EstimateCandidate]:
    resolved: List[EstimateCandidate] = []
    for candidate in sorted(candidates, key=_candidate_sort_key):
        member = _member_for_candidates(
            "scope",
            candidate.room_surrogate_id or "scope",
            [candidate],
            issues_by_id,
            counts_toward_estimate=True,
        )
        resolved.append(_build_resolved_candidate(
            [candidate],
            resolved_cluster_key=cluster_key,
            unit_count=max(1, candidate.estimate_unit_count),
            unit_label="scope",
            method="extracted_scope",
            confidence="high",
            notes=[candidate.estimate_scope_key] if candidate.estimate_scope_key else [],
            members=[member],
            preserve_scope_identity=True,
        ))
    return resolved


def _resolve_per_property(
    candidates: List[EstimateCandidate],
    cluster_key: str,
    issues_by_id: Dict[str, Dict[str, Any]],
) -> List[EstimateCandidate]:
    member = _member_for_candidates(
        "property", "property", candidates, issues_by_id,
        counts_toward_estimate=True,
    )
    return [_build_resolved_candidate(
        candidates,
        resolved_cluster_key=cluster_key,
        unit_count=1,
        unit_label="property",
        method="property_single_unit",
        confidence="high",
        notes=["property"],
        members=[member],
    )]


def _resolve_per_room(
    candidates: List[EstimateCandidate],
    cluster_key: str,
    issues_by_id: Dict[str, Dict[str, Any]],
) -> List[EstimateCandidate]:
    return [_resolve_distinct_room_like(
        candidates,
        cluster_key,
        issues_by_id,
        method="distinct_room_surrogates",
        fallback_method="generic_room_fallback",
        unit_name="room",
        matcher=_is_meaningful_room,
        confidence="medium",
        identity_getter=_billable_or_room_hint,
    )]


def _resolve_per_kitchen(
    candidates: List[EstimateCandidate],
    cluster_key: str,
    issues_by_id: Dict[str, Dict[str, Any]],
) -> List[EstimateCandidate]:
    return [_resolve_distinct_room_like(
        candidates,
        cluster_key,
        issues_by_id,
        method="distinct_kitchen_surrogates",
        fallback_method="single_kitchen_fallback",
        unit_name="kitchen",
        matcher=_is_kitchen_like,
        confidence="medium",
        identity_getter=_billable_or_room_hint,
    )]


def _resolve_per_bathroom(
    candidates: List[EstimateCandidate],
    cluster_key: str,
    issues_by_id: Dict[str, Dict[str, Any]],
) -> List[EstimateCandidate]:
    return [_resolve_distinct_room_like(
        candidates,
        cluster_key,
        issues_by_id,
        method="distinct_bathroom_surrogates",
        fallback_method="single_bathroom_fallback",
        unit_name="bathroom",
        matcher=_is_bathroom_like,
        confidence="medium",
        identity_getter=_billable_or_room_hint,
    )]


def _resolve_distinct_room_like(
    candidates: List[EstimateCandidate],
    cluster_key: str,
    issues_by_id: Dict[str, Dict[str, Any]],
    *,
    method: str,
    fallback_method: str,
    unit_name: str,
    matcher,
    confidence: str,
    identity_getter=None,
) -> EstimateCandidate:
    identity_getter = identity_getter or _room_hint
    groups: Dict[str, List[EstimateCandidate]] = {}
    for candidate in sorted(candidates, key=_candidate_sort_key):
        hint = identity_getter(candidate)
        member_key = hint if hint else "unspecified"
        groups.setdefault(member_key, []).append(candidate)

    billable_keys = sorted(key for key in groups if matcher(key))
    if not billable_keys:
        billable_keys = ["unspecified"]

    members = [
        _member_for_candidates(
            unit_name,
            key,
            groups[key],
            issues_by_id,
            counts_toward_estimate=key in billable_keys,
        )
        for key in sorted(groups)
    ]
    unit_count = max(1, len(billable_keys))
    notes = billable_keys if billable_keys != ["unspecified"] else ["fallback:1"]
    label = _plural_label(unit_count, unit_name)
    actual_method = method if billable_keys != ["unspecified"] else fallback_method
    actual_confidence = confidence if billable_keys != ["unspecified"] else "low"

    return _build_resolved_candidate(
        candidates,
        resolved_cluster_key=cluster_key,
        unit_count=unit_count,
        unit_label=label,
        method=actual_method,
        confidence=actual_confidence,
        notes=notes,
        members=members,
    )


def _resolve_per_opening(
    candidates: List[EstimateCandidate],
    cluster_key: str,
    issues_by_id: Dict[str, Dict[str, Any]],
) -> List[EstimateCandidate]:
    explicit_groups: Dict[str, List[EstimateCandidate]] = {}
    weak_groups: Dict[str, List[EstimateCandidate]] = {}
    for candidate in sorted(candidates, key=_candidate_sort_key):
        explicit_hints = _explicit_opening_hints(candidate, issues_by_id)
        if explicit_hints:
            for explicit_hint in explicit_hints:
                explicit_groups.setdefault(explicit_hint, []).append(candidate)
            continue
        weak_hint = _weak_opening_hint(candidate)
        weak_groups.setdefault(weak_hint or "unspecified", []).append(candidate)

    if explicit_groups:
        groups = explicit_groups
        method = "explicit_opening_ids"
        confidence = "medium"
    else:
        distinct_weak = {
            key for key in weak_groups
            if key != "unspecified" and _has_distinct_opening_language(key)
        }
        if len(distinct_weak) > 1:
            groups = {key: weak_groups[key] for key in sorted(distinct_weak)}
            method = "distinct_opening_hints"
            confidence = "low"
        else:
            groups = {"unspecified": candidates}
            method = "conservative_single_opening"
            confidence = "low"

    members = [
        _member_for_candidates(
            "opening", key, members, issues_by_id,
            counts_toward_estimate=True,
        )
        for key, members in sorted(groups.items())
    ]
    unit_count = max(1, len(groups))
    return [_build_resolved_candidate(
        candidates,
        resolved_cluster_key=cluster_key,
        unit_count=unit_count,
        unit_label=_plural_label(unit_count, "opening"),
        method=method,
        confidence=confidence,
        notes=sorted(groups.keys()),
        members=members,
    )]


def _resolve_per_system(
    candidates: List[EstimateCandidate],
    cluster_key: str,
    issues_by_id: Dict[str, Dict[str, Any]],
) -> List[EstimateCandidate]:
    member = _member_for_candidates(
        "system", "system", candidates, issues_by_id,
        counts_toward_estimate=True,
    )
    return [_build_resolved_candidate(
        candidates,
        resolved_cluster_key=cluster_key,
        unit_count=1,
        unit_label="system",
        method="system_default_single_unit",
        confidence="low",
        notes=["fallback:1"],
        members=[member],
    )]


def _resolve_per_area(
    candidates: List[EstimateCandidate],
    cluster_key: str,
    issues_by_id: Dict[str, Dict[str, Any]],
) -> List[EstimateCandidate]:
    member = _member_for_candidates(
        "area", "area", candidates, issues_by_id,
        counts_toward_estimate=True,
    )
    return [_build_resolved_candidate(
        candidates,
        resolved_cluster_key=cluster_key,
        unit_count=1,
        unit_label="area",
        method="area_default_single_unit",
        confidence="low",
        notes=["fallback:1"],
        members=[member],
    )]


def _build_resolved_candidate(
    candidates: List[EstimateCandidate],
    *,
    resolved_cluster_key: str,
    unit_count: int,
    unit_label: str,
    method: str,
    confidence: str,
    notes: List[str],
    members: List[Dict[str, Any]],
    preserve_scope_identity: bool = False,
) -> EstimateCandidate:
    sources = sorted(candidates, key=_candidate_sort_key)
    canonical = sources[0]
    source_estimate_unit_ids = _unique_sorted(
        _candidate_source_unit_ids(source) for source in sources
    )
    source_issue_ids = _unique_sorted(
        _candidate_source_issue_ids(source) for source in sources
    )
    issue_ids = _unique_sorted(source.issue_ids for source in sources)
    photo_keys = _unique_sorted(source.photo_keys for source in sources)
    scene_groups = _unique_sorted(source.scene_groups_seen for source in sources)
    observations = _unique_sorted(source.supporting_observations for source in sources)
    evidence_refs = _unique_evidence_refs(source.evidence_refs for source in sources)
    estimate_scope, estimate_scope_reason = choose_scope(
        (
            source.estimate_scope,
            source.estimate_scope_reason,
        )
        for source in sources
    )
    baseline_scope_before_posture, _baseline_reason = choose_scope(
        (
            source.baseline_scope_before_posture,
            "",
        )
        for source in sources
    )

    estimate_scope_key = canonical.estimate_scope_key
    estimate_unit_id = canonical.estimate_unit_id
    if not preserve_scope_identity:
        clean_cluster = _clean_scope_component(resolved_cluster_key)
        estimate_unit_id = f"{canonical.catalog_item_id}:{clean_cluster}"

    resolved = EstimateCandidate(
        catalog_item_id=canonical.catalog_item_id,
        catalog_item_name=canonical.catalog_item_name,
        estimate_meta=canonical.estimate_meta,
        kind=canonical.kind,
        severity=max(source.severity for source in sources),
        scope=canonical.scope,
        trade_bucket=canonical.trade_bucket,
        cost_obj=canonical.cost_obj,
        occurrences=sum(max(0, source.occurrences) for source in sources),
        estimate_unit_count=max(1, unit_count),
        scene_groups_seen=scene_groups,
        photo_keys=photo_keys,
        issue_ids=issue_ids,
        estimate_unit_id=estimate_unit_id,
        billable_estimate_unit_id=_resolved_billable_estimate_unit(sources, members),
        estimate_scope_key=estimate_scope_key,
        resolved_cluster_key=resolved_cluster_key,
        room_surrogate_id=_resolved_room_surrogate(sources),
        source_room_surrogate_ids=_unique_sorted(
            _room_hint(source) for source in sources if _room_hint(source)
        ),
        estimate_scope=estimate_scope,
        estimate_scope_reason=estimate_scope_reason,
        baseline_scope_before_posture=baseline_scope_before_posture,
        visible_required_with_inspect_posture=any(
            source.visible_required_with_inspect_posture for source in sources
        ),
        required_baseline_included=any(
            source.required_baseline_included for source in sources
        ),
        inspection_risk_added=any(
            source.inspection_risk_added for source in sources
        ),
        cost_model=canonical.cost_model,
        cost_model_source=canonical.cost_model_source,
        supporting_observations=observations,
        distinct_photo_count=len(photo_keys),
        distinct_scene_group_count=len(scene_groups),
        representative_issue_id=issue_ids[0] if issue_ids else None,
        estimate_unit_label=unit_label,
        unit_resolution_method=method,
        unit_resolution_confidence=confidence,
        unit_resolution_notes=notes,
        unit_members=members,
        source_estimate_unit_ids=source_estimate_unit_ids,
        source_issue_ids=source_issue_ids or issue_ids,
        evidence_refs=evidence_refs,
        package_evidence_only=all(
            getattr(source, "package_evidence_only", False) for source in sources
        ),
    )
    _copy_common_review_fields(resolved, sources)
    return resolved


def _copy_common_review_fields(
    target: EstimateCandidate,
    sources: List[EstimateCandidate],
) -> None:
    attrs = (
        "is_valid_detection", "review_posture", "review_visible_scope", "review_rationale",
        "review_source", "effective_posture", "review_image_path",
        "pass_2f_attempted", "pass_2f_applied", "pass_2f_fallback_reason",
        "package_id", "package_type", "package_role",
        "visual_verification_status", "package_verification_source",
        "estimate_scope", "estimate_scope_reason",
        "baseline_scope_before_posture", "visible_required_with_inspect_posture",
        "required_baseline_included", "inspection_risk_added",
        "cost_model", "cost_model_source",
    )
    for attr in attrs:
        values = [getattr(source, attr) for source in sources]
        if len(sources) == 1 or all(value == values[0] for value in values):
            setattr(target, attr, values[0])
    flags = {
        str(flag).strip()
        for source in sources
        for flag in (getattr(source, "review_consistency_flags", []) or [])
        if str(flag or "").strip()
    }
    target.review_consistency_flags = sorted(flags)


def _member_for_candidates(
    unit_type: str,
    unit_key: str,
    candidates: List[EstimateCandidate],
    issues_by_id: Dict[str, Dict[str, Any]],
    *,
    counts_toward_estimate: bool,
) -> Dict[str, Any]:
    sources = sorted(candidates, key=_candidate_sort_key)
    issue_ids = _unique_sorted(source.issue_ids for source in sources)
    issue_records = [issues_by_id[iid] for iid in issue_ids if iid in issues_by_id]
    observations = _unique_sorted(
        [source.supporting_observations for source in sources]
        + [[issue.get("description", "") for issue in issue_records]]
    )
    photo_keys = _unique_sorted(
        [source.photo_keys for source in sources]
        + [[issue.get("photo_key", "") for issue in issue_records]]
    )
    scene_groups = _unique_sorted(
        [source.scene_groups_seen for source in sources]
        + [[issue.get("scene_group", "") for issue in issue_records]]
    )
    scope_keys = _unique_sorted(source.estimate_scope_key for source in sources)
    room_keys = _unique_sorted(
        _room_hint(source) for source in sources if _room_hint(source)
    )
    estimate_unit_keys = _unique_sorted(
        _billable_estimate_unit_hint(source)
        for source in sources
        if _billable_estimate_unit_hint(source)
    )
    estimate_scope, estimate_scope_reason = choose_scope(
        (
            source.estimate_scope,
            source.estimate_scope_reason,
        )
        for source in sources
    )
    baseline_scope_before_posture, _baseline_reason = choose_scope(
        (
            source.baseline_scope_before_posture,
            "",
        )
        for source in sources
    )

    return {
        "unit_type": unit_type,
        "unit_key": unit_key,
        "unit_label": unit_key,
        "counts_toward_estimate": counts_toward_estimate,
        "estimate_unit_id": estimate_unit_keys[0] if len(estimate_unit_keys) == 1 else unit_key,
        "estimate_unit_ids": estimate_unit_keys,
        "room_surrogate_id": room_keys[0] if len(room_keys) == 1 else "",
        "room_surrogate_ids": room_keys,
        "source_room_surrogate_ids": room_keys,
        "estimate_scope_keys": scope_keys,
        "estimate_scope": estimate_scope,
        "estimate_scope_reason": estimate_scope_reason,
        "baseline_scope_before_posture": baseline_scope_before_posture,
        "visible_required_with_inspect_posture": any(
            source.visible_required_with_inspect_posture for source in sources
        ),
        "required_baseline_included": any(
            source.required_baseline_included for source in sources
        ),
        "inspection_risk_added": any(
            source.inspection_risk_added for source in sources
        ),
        "cost_model": sources[0].cost_model if sources else "",
        "cost_model_source": sources[0].cost_model_source if sources else "",
        "source_estimate_unit_ids": _unique_sorted(
            _candidate_source_unit_ids(source) for source in sources
        ),
        "issue_ids": issue_ids,
        "photo_keys": photo_keys,
        "scene_groups": scene_groups,
        "observations": observations,
    }


def _issues_by_id(issues_flat: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        issue["issue_id"]: issue
        for issue in (issues_flat or [])
        if isinstance(issue, dict) and issue.get("issue_id")
    }


def _candidate_sort_key(candidate: EstimateCandidate) -> Tuple[str, str, str]:
    return (
        candidate.catalog_item_id or "",
        candidate.estimate_scope_key or "",
        candidate.estimate_unit_id or "",
    )


def _candidate_source_unit_ids(candidate: EstimateCandidate) -> List[str]:
    if candidate.source_estimate_unit_ids:
        return candidate.source_estimate_unit_ids
    return [candidate.estimate_unit_id] if candidate.estimate_unit_id else []


def _candidate_source_issue_ids(candidate: EstimateCandidate) -> List[str]:
    if candidate.source_issue_ids:
        return candidate.source_issue_ids
    return candidate.issue_ids


def _resolved_room_surrogate(candidates: List[EstimateCandidate]) -> str:
    rooms = _unique_sorted(
        _room_hint(source) for source in candidates if _room_hint(source)
    )
    if len(rooms) == 1:
        return rooms[0]
    if len(rooms) > 1:
        return "multiple_rooms"
    return ""


def _room_hint(candidate: EstimateCandidate) -> str:
    return _meaningful_unit_hint(candidate.room_surrogate_id)


def _billable_estimate_unit_hint(candidate: EstimateCandidate) -> str:
    return _meaningful_unit_hint(getattr(candidate, "billable_estimate_unit_id", ""))


def _billable_or_room_hint(candidate: EstimateCandidate) -> str:
    return _billable_estimate_unit_hint(candidate) or _room_hint(candidate)


def _resolved_billable_estimate_unit(
    candidates: List[EstimateCandidate],
    members: List[Dict[str, Any]],
) -> str:
    unit_ids = _unique_sorted(
        _billable_estimate_unit_hint(source)
        for source in candidates
        if _billable_estimate_unit_hint(source)
    )
    if len(unit_ids) == 1:
        return unit_ids[0]
    member_ids = _unique_sorted(
        member.get("estimate_unit_id")
        for member in members
        if member.get("estimate_unit_id")
    )
    if len(member_ids) == 1:
        return member_ids[0]
    if len(member_ids) > 1:
        return "multiple_estimate_units"
    return ""


def _meaningful_unit_hint(value: Any) -> str:
    hint = _meaningful_scope_hint(value)
    if hint in _GENERIC_UNIT_HINTS:
        return ""
    return hint


def _is_meaningful_room(value: str) -> bool:
    return bool(_meaningful_unit_hint(value))


def _is_kitchen_like(value: str) -> bool:
    hint = _meaningful_unit_hint(value)
    return hint in _KITCHEN_HINTS or "kitchen" in hint


def _is_bathroom_like(value: str) -> bool:
    hint = _meaningful_unit_hint(value)
    return any(token in hint for token in _BATHROOM_TOKENS)


def _explicit_opening_hints(
    candidate: EstimateCandidate,
    issues_by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    hints: List[str] = []
    for issue_id in candidate.issue_ids:
        issue = issues_by_id.get(issue_id, {})
        for field_name in _OPENING_INSTANCE_FIELDS:
            hint = _meaningful_unit_hint(issue.get(field_name))
            if hint:
                hints.append(f"{field_name}:{hint}")
                break
    return sorted(set(hints))


def _weak_opening_hint(candidate: EstimateCandidate) -> str:
    room = _room_hint(candidate) or "room"
    description = " ".join(candidate.supporting_observations)
    normalized = _normalize_description(description)
    return f"{room}|{normalized}" if normalized else room


def _has_distinct_opening_language(value: str) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", value.lower()))
    return bool(tokens & _OPENING_SIDE_WORDS)


def _normalize_description(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token]
    return " ".join(tokens[:10])


def _plural_label(count: int, unit_name: str) -> str:
    if count == 1:
        return f"1 {unit_name}"
    if unit_name.endswith("y"):
        return f"{count} {unit_name[:-1]}ies"
    return f"{count} {unit_name}s"


def _unique_sorted(values: Iterable[Any]) -> List[str]:
    result = set()

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                result.add(text)
            return
        try:
            iterator = iter(value)
        except TypeError:
            text = str(value or "").strip()
            if text:
                result.add(text)
            return
        else:
            if isinstance(value, dict):
                text = str(value or "").strip()
                if text:
                    result.add(text)
                return
            for item in iterator:
                visit(item)

    visit(values)
    return sorted(result)


def _unique_evidence_refs(values: Iterable[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            issue_id = str(value.get("issue_id") or "").strip()
            photo_key = str(value.get("photo_key") or "").strip()
            if not issue_id:
                return
            key = (issue_id, photo_key)
            if key in seen:
                return
            seen.add(key)
            result.append({
                "issue_id": issue_id,
                "photo_key": photo_key,
                "observation": str(value.get("observation") or "").strip(),
                "room_surrogate_id": str(value.get("room_surrogate_id") or "").strip(),
            })
            return
        if isinstance(value, str):
            return
        try:
            iterator = iter(value)
        except TypeError:
            return
        for item in iterator:
            visit(item)

    visit(values)
    return result
