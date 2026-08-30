"""Deterministic package candidates from ACTIVE work items (Session 4).

Consumes the frozen, validated Session 3 standalone-estimate result and
produces PackageCandidate records by feeding the immutable work/condition
lineage through the legacy inference primitives
(tools/rehab_packages.infer_package_candidates: catalog affinities, scene
guard, ambient-support demotion, driver/support lanes, strength thresholds,
tier resolution, tier escalation) via throwaway adapter objects. The legacy
entrypoint's one side effect — mutating weak opportunity drivers — lands on
the adapters, so weak opportunities produce no package and leave their work
items untouched.

Only ACTIVE work items contribute (suppressed dedup sources are audit-only);
lineage collapses back to stable work-item IDs with driver precedence. The
legacy primitives price tiers unscaled, so the standalone property cost
factor is applied to each tier spec once, on adoption, which puts it in the
same dollars as the work items it will absorb. The legacy Phase C cost-floor
concept is then applied deterministically against the children's standalone
range; the unfloored (but factored) tier spec is retained for audit. The whole-home turnover aggregate is emitted as a display-only
candidate with NO children — its lineage flows through
contributing_candidate_ids, so it can never absorb work.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from tools.rehab_packages import infer_package_candidates
from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    PACKAGE_CATEGORIES,
    PACKAGE_LEVELS,
    PACKAGE_ROOMS,
    PACKAGE_STRENGTHS,
    PACKAGE_TREATMENTS,
    PACKAGE_TYPES,
    PackageCandidate,
    WHOLE_HOME_PACKAGE_TYPE,
    WHOLE_HOME_PRICING_PROFILE,
    WHOLE_HOME_PRICING_TIER,
    WHOLE_HOME_UNIT_ID,
)
from tools.renovation_architecture.ids import make_package_candidate_id
from tools.renovation_architecture.validators import (
    validate_standalone_estimate_result,
)
from tools.scene_classifier_passes import PassExecutionError

# Strength ordering for the whole-home rollup (strongest contributor wins).
_STRENGTH_RANK = {"strong": 2, "moderate": 1}


def _candidate_failure(stage: str, message: str, *, code: str) -> PassExecutionError:
    return PassExecutionError("package_candidates", stage, message, code=code)


class _EstimateMeta:
    def __init__(self, strategy: Optional[str]):
        self.strategy = strategy
        self.estimate_tier = None


class _AdapterCandidate:
    """Throwaway duck-typed stand-in for renovation_estimate.EstimateCandidate.

    One instance per (ACTIVE work item, contributing condition): the condition
    carries the room/scene/evidence identity the legacy inference routes on,
    while work_item_id is what the emitted package collapses back to. The
    instance is mutable on purpose — the legacy weak-opportunity suppression
    writes to it, and the writes must land here, never on the WorkItem."""

    def __init__(
        self,
        *,
        work_item_id: str,
        condition: Mapping[str, Any],
        evidence: Mapping[str, Any],
        observable: Mapping[str, Any],
        policy: Mapping[str, Any],
        work_item: Mapping[str, Any],
        catalog_name: str,
    ):
        severity = observable.get("severity")
        self.work_item_id = work_item_id
        self.condition_id = condition["condition_id"]
        self.catalog_item_id = condition["catalog_item_id"]
        self.catalog_item_name = catalog_name
        self.severity = (
            severity
            if isinstance(severity, int) and not isinstance(severity, bool)
            else 2
        )
        # The condition's own catalog trade bucket (not the merged work item's
        # dedup-key bucket) — component classification is per catalog item.
        self.trade_bucket = policy["trade_bucket"]
        self.issue_ids = list(condition["issue_ids"])
        # Deduped representatives only: duplicate and near-duplicate photos
        # are not independent corroboration, so they must not feed the
        # legacy multi-photo corroboration counts.
        self.photo_keys = list(evidence["representative_photo_keys"])
        self.evidence_refs = [dict(ref) for ref in evidence["evidence_refs"]]
        self.supporting_observations = sorted({
            str(ref.get("observation") or "")
            for ref in evidence["evidence_refs"]
            if ref.get("observation")
        })
        self.scene_groups_seen = [condition["scene_group"]]
        self.room_surrogate_id = condition["room_surrogate_id"]
        self.estimate_unit_id = condition["estimate_unit_id"]
        self.billable_estimate_unit_id = condition["estimate_unit_id"]
        self.estimate_meta = _EstimateMeta(policy.get("strategy"))
        self.estimate_scope = work_item["estimate_scope"]
        self.estimate_scope_reason = work_item["estimate_scope_reason"]
        self.package_evidence_only = False
        self.package_role = None
        # Written by _suppress_blocked_opportunity_drivers on weak buckets.
        self.is_valid_detection = True
        self.pass_2f_attempted = False
        self.pass_2f_applied = False
        self.pass_2f_fallback_reason = None
        self.review_source = None


def _build_adapters(
    result: Mapping[str, Any],
    projection: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> Tuple[List[_AdapterCandidate], Dict[str, str]]:
    """Adapter records plus the issue_id -> work_item_id collapse index."""
    conditions = {
        condition["condition_id"]: condition
        for condition in result["observed_conditions"]
    }
    evidence_by_condition = {
        evidence["condition_id"]: evidence
        for evidence in result["evidence_facts"]
    }
    observables = projection["observables"]
    work_policy = projection["work_policy"]
    names = {
        str(item.get("id")): str(item.get("name") or item.get("id"))
        for item in catalog.get("items") or []
        if isinstance(item, dict) and item.get("id")
    }

    adapters: List[_AdapterCandidate] = []
    issue_to_work: Dict[str, str] = {}
    active_items = sorted(
        (item for item in result["work_items"] if item["status"] == "active"),
        key=lambda item: item["work_item_id"],
    )
    for work_item in active_items:
        for condition_id in work_item["condition_ids"]:
            condition = conditions[condition_id]
            catalog_item_id = condition["catalog_item_id"]
            policy = work_policy.get(catalog_item_id)
            if policy is None:
                # The S3 gate only accepts work items whose conditions cite
                # work-route catalog items, so this is an operational
                # contradiction, never a routing gap.
                raise _candidate_failure(
                    "dependency",
                    f"active work item {work_item['work_item_id']!r} cites "
                    f"catalog item {catalog_item_id!r} with no work policy "
                    "in the projection",
                    code="WorkPolicyMissing",
                )
            adapters.append(_AdapterCandidate(
                work_item_id=work_item["work_item_id"],
                condition=condition,
                evidence=evidence_by_condition[condition_id],
                observable=observables[catalog_item_id],
                policy=policy,
                work_item=work_item,
                catalog_name=names.get(catalog_item_id, catalog_item_id),
            ))
            for issue_id in condition["issue_ids"]:
                issue_to_work[issue_id] = work_item["work_item_id"]
    return adapters, issue_to_work


def _collapse_issue_ids(
    issue_ids: List[str],
    issue_to_work: Mapping[str, str],
    *,
    package_id: str,
) -> List[str]:
    work_ids = set()
    for issue_id in issue_ids or []:
        work_id = issue_to_work.get(str(issue_id))
        if work_id is None:
            raise _candidate_failure(
                "parse",
                f"legacy package {package_id!r} cites issue {issue_id!r} "
                "outside the supplied work lineage",
                code="PackageLineageUnknownIssue",
            )
        work_ids.add(work_id)
    return sorted(work_ids)


def _vocab_check(package: Mapping[str, Any]) -> None:
    """The legacy dict must land inside the closed v4 vocabularies; anything
    else is a contract drift to surface, not to coerce."""
    checks = (
        ("package_type", PACKAGE_TYPES),
        ("package_category", PACKAGE_CATEGORIES),
        ("package_level", PACKAGE_LEVELS),
        ("room", PACKAGE_ROOMS),
        ("package_strength", PACKAGE_STRENGTHS),
        ("trigger_reason", PACKAGE_TREATMENTS),
    )
    for field, vocabulary in checks:
        if package.get(field) not in vocabulary:
            raise _candidate_failure(
                "parse",
                f"legacy package {package.get('package_id')!r} carries "
                f"{field}={package.get(field)!r} outside the closed v4 "
                "vocabulary",
                code="PackageVocabularyDrift",
            )


def build_package_candidates(
    *,
    standalone_result: Mapping[str, Any],
    projection: Mapping[str, Any],
    catalog: Mapping[str, Any],
    estimate_id: str,
) -> List[Dict[str, Any]]:
    """The Session 4 deterministic step: frozen Session 3 result in,
    PackageCandidate dicts out (sorted by candidate id). The input is
    validated first and never mutated."""
    result = dict(standalone_result)
    validation = validate_standalone_estimate_result(result, estimate_id=estimate_id)
    if not validation.ok:
        raise _candidate_failure(
            "dependency",
            "standalone-estimate result failed validation before package "
            "candidate construction: " + "; ".join(validation.errors[:5]),
            code="StandaloneResultInvalid",
        )

    # The legacy inference primitives price tier specs off the unscaled
    # catalog, but work items are already factored (Session 3 applies it once,
    # pre-split). Adopt the same factor here so the floor below compares like
    # for like.
    factor = float(result["standalone_estimate"]["property_cost_factor"])
    adapters, issue_to_work = _build_adapters(result, projection, catalog)
    work_dollars = {
        item["work_item_id"]: (item["low"], item["high"])
        for item in result["work_items"]
        if item["status"] == "active"
    }

    # Room surrogates and estimate units synthesized from the condition
    # layer's identity audit trail (first-wins over the deterministic
    # condition order; a surrogate has one scene group by construction).
    scene_by_surrogate: Dict[str, str] = {}
    units: Dict[str, set] = {}
    for condition in sorted(
        result["observed_conditions"], key=lambda c: c["condition_id"]
    ):
        surrogate = condition["room_surrogate_id"]
        if surrogate and surrogate not in scene_by_surrogate:
            scene_by_surrogate[surrogate] = condition["scene_group"]
        units.setdefault(condition["estimate_unit_id"], set()).update(
            condition["source_room_surrogate_ids"]
        )
    room_surrogates = [
        {"room_surrogate_id": surrogate, "scene": scene}
        for surrogate, scene in sorted(scene_by_surrogate.items())
    ]
    estimate_units = [
        {"estimate_unit_id": unit_id, "source_room_surrogate_ids": sorted(ids)}
        for unit_id, ids in sorted(units.items())
    ]

    try:
        legacy_packages = infer_package_candidates(
            adapters,
            room_surrogates,
            dict(catalog),
            estimate_units=estimate_units,
            # Catalog 3.2. v5 only: infer_package_candidates is shared with the
            # v4 estimator, which keeps its pre-3.2 routing (the flag defaults
            # off). See docs/FINDINGS_catalog_3_2_deferred_issues.md.
            contextual_repair_support=True,
        )
    except ValueError as exc:
        raise _candidate_failure(
            "dependency",
            f"legacy package inference rejected the supplied lineage: {exc}",
            code="PackageInferenceInvalid",
        ) from exc

    candidates: List[PackageCandidate] = []
    for package in legacy_packages:
        _vocab_check(package)
        package_id = str(package["package_id"])
        driver_ids = _collapse_issue_ids(
            package.get("driver_issue_ids") or [], issue_to_work,
            package_id=package_id,
        )
        support_owner_ids = _collapse_issue_ids(
            package.get("support_issue_ids") or [], issue_to_work,
            package_id=package_id,
        )
        # Driver precedence: a merged work item supplying both roles is a
        # driver. children = drivers ∪ supports (disjoint by construction
        # after the subtraction).
        support_ids = sorted(set(support_owner_ids) - set(driver_ids))
        child_ids = sorted(set(driver_ids) | set(support_ids))
        if not child_ids:
            raise _candidate_failure(
                "parse",
                f"legacy package {package_id!r} produced no work-item "
                "children — packages may only group active accepted work",
                code="PackageWithoutChildren",
            )
        unfloored_low = int(round(int(package["candidate_cost_low"]) * factor))
        unfloored_high = int(round(int(package["candidate_cost_high"]) * factor))
        child_low = sum(work_dollars[work_id][0] for work_id in child_ids)
        child_high = sum(work_dollars[work_id][1] for work_id in child_ids)
        low = max(unfloored_low, child_low)
        high = max(unfloored_high, child_high)
        candidates.append(PackageCandidate(
            package_candidate_id=make_package_candidate_id(
                estimate_id=estimate_id,
                package_type=package["package_type"],
                estimate_unit_id=package["estimate_unit_id"],
            ),
            schema_version=CONTRACTS_SCHEMA_VERSION,
            package_type=package["package_type"],
            package_category=package["package_category"],
            package_level=package["package_level"],
            room=package["room"],
            estimate_unit_id=package["estimate_unit_id"],
            child_work_item_ids=tuple(child_ids),
            driver_work_item_ids=tuple(driver_ids),
            support_work_item_ids=tuple(support_ids),
            strength=package["package_strength"],
            pricing_profile=package["pricing_profile"],
            pricing_tier=package["pricing_tier"],
            absorption_scope=dict(package["absorption_scope"]),
            proposed_treatment=package["trigger_reason"],
            unfloored_low=unfloored_low,
            unfloored_high=unfloored_high,
            cost_floor_applied=(low, high) != (unfloored_low, unfloored_high),
            low=low,
            high=high,
            display_only=False,
            contributing_candidate_ids=(),
        ))

    whole_home = _aggregate_whole_home(candidates, estimate_id=estimate_id)
    if whole_home is not None:
        candidates.append(whole_home)

    return [
        candidate.to_dict()
        for candidate in sorted(
            candidates, key=lambda candidate: candidate.package_candidate_id
        )
    ]


def _aggregate_whole_home(
    candidates: List[PackageCandidate], *, estimate_id: str
) -> Optional[PackageCandidate]:
    """The legacy whole-home turnover rollup as a reviewable, display-only
    candidate: >= 2 distinct turnover rooms, range = sum of the contributors'
    floored ranges, no children (contributor refs carry the lineage)."""
    contributors = [
        candidate for candidate in candidates
        if candidate.package_category == "turnover"
        and candidate.package_level == "room"
    ]
    if len({candidate.room for candidate in contributors}) < 2:
        return None
    strongest = max(
        (candidate.strength for candidate in contributors),
        key=lambda strength: _STRENGTH_RANK[strength],
    )
    low = sum(candidate.low for candidate in contributors)
    high = sum(candidate.high for candidate in contributors)
    return PackageCandidate(
        package_candidate_id=make_package_candidate_id(
            estimate_id=estimate_id,
            package_type=WHOLE_HOME_PACKAGE_TYPE,
            estimate_unit_id=WHOLE_HOME_UNIT_ID,
        ),
        schema_version=CONTRACTS_SCHEMA_VERSION,
        package_type=WHOLE_HOME_PACKAGE_TYPE,
        package_category="turnover",
        package_level="property",
        room="whole_home",
        estimate_unit_id=WHOLE_HOME_UNIT_ID,
        child_work_item_ids=(),
        driver_work_item_ids=(),
        support_work_item_ids=(),
        strength=strongest,
        pricing_profile=WHOLE_HOME_PRICING_PROFILE,
        pricing_tier=WHOLE_HOME_PRICING_TIER,
        absorption_scope={
            "family": "whole_home", "groups": [], "trade_buckets": [],
            "components": [],
        },
        proposed_treatment="whole_home_turnover_aggregate",
        unfloored_low=low,
        unfloored_high=high,
        cost_floor_applied=False,
        low=low,
        high=high,
        display_only=True,
        contributing_candidate_ids=tuple(sorted(
            candidate.package_candidate_id for candidate in contributors
        )),
    )
