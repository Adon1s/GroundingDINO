"""Deterministic work derivation and the standalone estimate (Session 3).

Consumes the frozen, validated Session 2 condition-review result and the
newest-catalog projection; produces priced work items, max-envelope dedup
collisions, and a package-independent standalone estimate. Only
``accepted_for_work`` dispositions create work — every other terminal outcome
(excluded, inspection, withheld, no_action) stays visible in its lane with
zero work dollars, exactly as Session 2 recorded it.

Pricing reuses the legacy costing core verbatim (tools/costing.py:
heuristic fallback, base + (N-1)*per_occurrence, caps, kind/scope/trade
multipliers applied once, manual allowances exempt). The property
market/size factor is applied exactly once per source aggregate, before any
split, and the room-like split is the sum-preserving integer allocation, so
scope and headline totals reconcile exactly with no epsilon.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tools.cost_factors import resolve_property_cost_factor
from tools.costing import compute_item_cost_range
from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    DEDUP_SUPPRESSION_REASON,
    ESTIMATE_SCOPE_MERGE_PRIORITY,
    ESTIMATE_SCOPES,
    MoneyRange,
    STANDALONE_PRICING_POLICY_VERSION,
    StandaloneEstimate,
    WORK_DEDUP_POLICY_VERSION,
    WorkDedupCollision,
    WorkItem,
)
from tools.renovation_architecture.ids import (
    make_merged_work_item_id,
    make_work_dedup_collision_id,
    make_work_item_id,
)
from tools.renovation_architecture.validators import (
    validate_condition_review_result,
    validate_standalone_estimate_result,
)
from tools.scene_classifier_passes import PassExecutionError

# Collapse unit policies bill one fixed synthetic unit regardless of how many
# physical units contributed conditions.
_COLLAPSE_BILLABLE_UNITS = {
    "per_property": "property", "per_system": "system", "per_area": "area"
}
# Room-like policies keep one work item per distinct accepted physical unit
# and split the legacy aggregate allowance across them deterministically.
_DISTINCT_UNIT_POLICIES = frozenset({"per_room", "per_kitchen", "per_bathroom"})
# The deterministic projection of legacy POSTURE_TO_SCOPE
# (tools/renovation_estimate.py) onto catalog strategy: repair_or_replace and
# absent fall through to the catalog scope; inspect_only never reaches
# accepted_for_work (the inspection route dispositions to inspection).
_STRATEGY_COSTING_SCOPE = {
    "repair_only": "repair", "replace_only": "replace", "service_only": "service"
}


def _derivation_failure(
    stage: str, message: str, *, code: str
) -> PassExecutionError:
    return PassExecutionError("work_derivation", stage, message, code=code)


def _split_integer(amount: int, n: int) -> List[int]:
    """Split amount into n integers summing exactly to amount, remainder one
    unit at a time to the first k entries — byte-compatible with the legacy
    reconciliation split (tools/rehab_packages._split_integer), copied so the
    derivation module never imports the legacy package monolith."""
    if n <= 0:
        return []
    base, remainder = divmod(amount, n)
    return [base + 1 if i < remainder else base for i in range(n)]


def _priced_range(
    policy: Mapping[str, Any],
    observable: Mapping[str, Any],
    n_price: int,
    factor: float,
) -> Tuple[int, int]:
    """Legacy costing core, then the property factor exactly once. round() is
    monotone, so low <= high survives the scaling."""
    severity = observable.get("severity")
    low, high = compute_item_cost_range(
        cost_obj=policy.get("cost"),
        n_occurrences=n_price,
        kind=observable.get("kind"),
        scope=(
            _STRATEGY_COSTING_SCOPE.get(policy.get("strategy"))
            or observable.get("scope")
        ),
        trade_bucket=policy.get("trade_bucket"),
        severity=(
            severity
            if isinstance(severity, int) and not isinstance(severity, bool)
            else 2
        ),
    )
    return int(round(low * factor)), int(round(high * factor))


def _source_item(
    *,
    estimate_id: str,
    catalog_item_id: str,
    policy: Mapping[str, Any],
    conditions: List[Dict[str, Any]],
    billable_unit_id: str,
    unit_count: int,
    low: int,
    high: int,
) -> WorkItem:
    return WorkItem(
        work_item_id=make_work_item_id(
            estimate_id=estimate_id,
            catalog_item_id=catalog_item_id,
            billable_unit_id=billable_unit_id,
            action_code=policy["action_code"],
        ),
        schema_version=CONTRACTS_SCHEMA_VERSION,
        condition_ids=tuple(sorted(c["condition_id"] for c in conditions)),
        catalog_item_ids=(catalog_item_id,),
        source_estimate_unit_ids=tuple(
            sorted({c["estimate_unit_id"] for c in conditions})
        ),
        billable_unit_id=billable_unit_id,
        action_code=policy["action_code"],
        action_sources=(policy["action_source"],),
        trade_bucket=policy["trade_bucket"],
        unit_policy=policy["unit_policy"],
        unit_count=unit_count,
        pricing_modes=(policy["pricing_mode"],),
        identity_ambiguous=any(c["identity_ambiguous"] for c in conditions),
        estimate_scope=policy["estimate_scope"],
        estimate_scope_reason=policy["estimate_scope_reason"],
        low=low,
        high=high,
        status="active",
        reason_code=None,
    )


def _derive_source_items(
    *,
    accepted_by_item: Dict[str, List[Dict[str, Any]]],
    projection: Mapping[str, Any],
    estimate_id: str,
    factor: float,
) -> List[WorkItem]:
    work_policy = projection["work_policy"]
    observables = projection["observables"]
    items: List[WorkItem] = []
    for catalog_item_id in sorted(accepted_by_item):
        conditions = sorted(
            accepted_by_item[catalog_item_id], key=lambda c: c["condition_id"]
        )
        policy = work_policy.get(catalog_item_id)
        if policy is None:
            # decide_disposition only accepts work-route conditions, and the
            # projection carries a work policy for every work-route item, so
            # this is an operational contradiction, never a pricing gap.
            raise _derivation_failure(
                "dependency",
                f"accepted condition cites catalog item {catalog_item_id!r} "
                "with no work policy in the projection",
                code="WorkPolicyMissing",
            )
        observable = observables[catalog_item_id]
        unit_policy = policy["unit_policy"]
        if unit_policy in _COLLAPSE_BILLABLE_UNITS:
            low, high = _priced_range(policy, observable, 1, factor)
            items.append(_source_item(
                estimate_id=estimate_id,
                catalog_item_id=catalog_item_id,
                policy=policy,
                conditions=conditions,
                billable_unit_id=_COLLAPSE_BILLABLE_UNITS[unit_policy],
                unit_count=1,
                low=low,
                high=high,
            ))
        elif unit_policy in _DISTINCT_UNIT_POLICIES:
            # One aggregate allowance across the distinct accepted units
            # (base + (N-1)*per_occurrence, capped, factored), then the
            # sum-preserving split — legacy economics, per-unit records.
            by_unit: Dict[str, List[Dict[str, Any]]] = {}
            for condition in conditions:
                by_unit.setdefault(condition["estimate_unit_id"], []).append(condition)
            units = sorted(by_unit)
            aggregate_low, aggregate_high = _priced_range(
                policy, observable, len(units), factor
            )
            lows = _split_integer(aggregate_low, len(units))
            highs = _split_integer(aggregate_high, len(units))
            for index, unit_id in enumerate(units):
                items.append(_source_item(
                    estimate_id=estimate_id,
                    catalog_item_id=catalog_item_id,
                    policy=policy,
                    conditions=by_unit[unit_id],
                    billable_unit_id=unit_id,
                    unit_count=1,
                    low=lows[index],
                    high=highs[index],
                ))
        else:
            # per_scope and per_opening: one work item per condition, priced
            # unit-locally. per_opening's occurrence count is the explicit
            # instance-hint count within the unit (conservative 1 without
            # hints); the legacy weak-language middle tier is dropped.
            for condition in conditions:
                unit_count = (
                    max(1, len(condition["opening_instance_hints"]))
                    if unit_policy == "per_opening"
                    else 1
                )
                low, high = _priced_range(policy, observable, unit_count, factor)
                items.append(_source_item(
                    estimate_id=estimate_id,
                    catalog_item_id=catalog_item_id,
                    policy=policy,
                    conditions=[condition],
                    billable_unit_id=condition["estimate_unit_id"],
                    unit_count=unit_count,
                    low=low,
                    high=high,
                ))
    return items


def _dedup_work_items(
    source_items: List[WorkItem], *, estimate_id: str
) -> Tuple[List[WorkItem], List[WorkDedupCollision]]:
    """Max-envelope dedup on (action_code, trade_bucket, unit_policy,
    billable unit). Colliding sources are suppressed as audit records and one
    merged active carries max(lows)/max(highs) — never a sum."""
    groups: Dict[Tuple[str, str, str, str], List[WorkItem]] = {}
    for item in source_items:
        key = (
            item.action_code, item.trade_bucket,
            item.unit_policy, item.billable_unit_id,
        )
        groups.setdefault(key, []).append(item)

    final_items: List[WorkItem] = []
    collisions: List[WorkDedupCollision] = []
    for key in sorted(groups):
        members = groups[key]
        if len(members) == 1:
            final_items.append(members[0])
            continue
        action_code, trade_bucket, unit_policy, billable_unit_id = key
        winning_scope = next(
            scope for scope in ESTIMATE_SCOPE_MERGE_PRIORITY
            if scope in {member.estimate_scope for member in members}
        )
        winner = min(
            (m for m in members if m.estimate_scope == winning_scope),
            key=lambda member: member.work_item_id,
        )
        merged_id = make_merged_work_item_id(
            estimate_id=estimate_id,
            action_code=action_code,
            trade_bucket=trade_bucket,
            unit_policy=unit_policy,
            billable_unit_id=billable_unit_id,
        )
        final_items.append(WorkItem(
            work_item_id=merged_id,
            schema_version=CONTRACTS_SCHEMA_VERSION,
            condition_ids=tuple(sorted(
                {cid for member in members for cid in member.condition_ids}
            )),
            catalog_item_ids=tuple(sorted(
                {item for member in members for item in member.catalog_item_ids}
            )),
            source_estimate_unit_ids=tuple(sorted(
                {unit for member in members for unit in member.source_estimate_unit_ids}
            )),
            billable_unit_id=billable_unit_id,
            action_code=action_code,
            action_sources=tuple(sorted(
                {source for member in members for source in member.action_sources}
            )),
            trade_bucket=trade_bucket,
            unit_policy=unit_policy,
            unit_count=max(member.unit_count for member in members),
            pricing_modes=tuple(sorted(
                {mode for member in members for mode in member.pricing_modes}
            )),
            identity_ambiguous=any(member.identity_ambiguous for member in members),
            estimate_scope=winning_scope,
            estimate_scope_reason=winner.estimate_scope_reason,
            low=max(member.low for member in members),
            high=max(member.high for member in members),
            status="active",
            reason_code=None,
        ))
        final_items.extend(
            replace(
                member,
                status="suppressed",
                reason_code=DEDUP_SUPPRESSION_REASON,
            )
            for member in members
        )
        collisions.append(WorkDedupCollision(
            collision_id=make_work_dedup_collision_id(
                estimate_id=estimate_id, active_work_item_id=merged_id
            ),
            schema_version=CONTRACTS_SCHEMA_VERSION,
            action_code=action_code,
            trade_bucket=trade_bucket,
            unit_policy=unit_policy,
            billable_unit_id=billable_unit_id,
            active_work_item_id=merged_id,
            suppressed_work_item_ids=tuple(sorted(
                member.work_item_id for member in members
            )),
            policy_version=WORK_DEDUP_POLICY_VERSION,
        ))
    return final_items, collisions


def derive_standalone_estimate(
    *,
    review_result: Mapping[str, Any],
    projection: Mapping[str, Any],
    property_metadata: Optional[Dict[str, Any]],
    estimate_id: str,
) -> Dict[str, Any]:
    """The Session 3 step: frozen Session 2 result in, superset result out.

    The input is validated first and never mutated. The output carries the
    Session 2 keys unchanged plus work_items, work_dedup_collisions, and
    standalone_estimate, and is self-validated before it is returned."""
    validation = validate_condition_review_result(
        dict(review_result), estimate_id=estimate_id
    )
    if not validation.ok:
        raise _derivation_failure(
            "dependency",
            "condition-review result failed validation before work "
            "derivation: " + "; ".join(validation.errors[:5]),
            code="ReviewResultInvalid",
        )

    conditions_by_id = {
        condition["condition_id"]: condition
        for condition in review_result["observed_conditions"]
    }
    accepted_by_item: Dict[str, List[Dict[str, Any]]] = {}
    for disposition in review_result["condition_dispositions"]:
        if disposition["disposition"] != "accepted_for_work":
            continue
        condition = conditions_by_id[disposition["condition_id"]]
        accepted_by_item.setdefault(
            condition["catalog_item_id"], []
        ).append(condition)

    factor, factor_audit = resolve_property_cost_factor(property_metadata)
    source_items = _derive_source_items(
        accepted_by_item=accepted_by_item,
        projection=projection,
        estimate_id=estimate_id,
        factor=factor,
    )
    work_items, collisions = _dedup_work_items(
        source_items, estimate_id=estimate_id
    )

    scope_totals = {scope: [0, 0] for scope in ESTIMATE_SCOPES}
    for item in work_items:
        if item.status != "active":
            continue
        scope_totals[item.estimate_scope][0] += item.low
        scope_totals[item.estimate_scope][1] += item.high
    standalone = StandaloneEstimate(
        schema_version=CONTRACTS_SCHEMA_VERSION,
        currency="USD",
        pricing_policy_version=STANDALONE_PRICING_POLICY_VERSION,
        property_cost_factor=factor,
        property_cost_factor_audit=factor_audit,
        totals_by_estimate_scope={
            scope: MoneyRange(low=scope_totals[scope][0], high=scope_totals[scope][1])
            for scope in sorted(ESTIMATE_SCOPES)
        },
        headline=MoneyRange(
            low=sum(low for low, _ in scope_totals.values()),
            high=sum(high for _, high in scope_totals.values()),
        ),
    )

    result: Dict[str, Any] = {
        **review_result,
        "work_items": [
            item.to_dict()
            for item in sorted(work_items, key=lambda item: item.work_item_id)
        ],
        "work_dedup_collisions": [
            collision.to_dict()
            for collision in sorted(
                collisions, key=lambda collision: collision.collision_id
            )
        ],
        "standalone_estimate": standalone.to_dict(),
    }
    validation = validate_standalone_estimate_result(result, estimate_id=estimate_id)
    if not validation.ok:
        raise _derivation_failure(
            "parse",
            "derived standalone-estimate result failed self-validation: "
            + "; ".join(validation.errors[:5]),
            code="StandaloneResultInvalid",
        )
    return result
