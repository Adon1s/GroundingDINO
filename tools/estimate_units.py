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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
) -> EstimateCandidate:
    groups: Dict[str, List[EstimateCandidate]] = {}
    for candidate in sorted(candidates, key=_candidate_sort_key):
        hint = _room_hint(candidate)
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
        estimate_scope_key=estimate_scope_key,
        resolved_cluster_key=resolved_cluster_key,
        room_surrogate_id=_resolved_room_surrogate(sources),
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

    return {
        "unit_type": unit_type,
        "unit_key": unit_key,
        "unit_label": unit_key,
        "counts_toward_estimate": counts_toward_estimate,
        "room_surrogate_id": room_keys[0] if len(room_keys) == 1 else "",
        "room_surrogate_ids": room_keys,
        "estimate_scope_keys": scope_keys,
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
