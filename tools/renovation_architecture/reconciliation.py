"""Deterministic coverage reconciliation — Session 5.

Applies Sol package decisions to the frozen Session 4 result without touching
it: absorption eligibility (approved, non-display, no split recommendation),
at-most-once child ownership under the legacy absorption priority ordering,
effective ranges recomputed from actually-owned children, exactly one
reason-coded ledger entry per ACTIVE work item, ledger-derived totals,
recomputed audits (all must be empty), and observability. The conservative
combine/split policy applies: combine recommendations become non-economic
group metadata while every member keeps its own deterministic pricing, and a
split recommendation makes the candidate non-billable with its children
falling back to their exact standalone allowances unless another approved
candidate independently absorbs them. No derived-package pricing is invented.

compute_reconciliation is the single deterministic computation, shared
verbatim by the producer (build_complete_result) and the artifact gate
(validators.validate_complete_result, via a function-level import that keeps
the module graph acyclic), so the two can never drift.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tools.rehab_packages import _MODERNIZATION_TIER_RANK
from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    FUNNEL_KEYS,
    OBSERVABILITY_PHASES,
)
from tools.renovation_architecture.ids import (
    make_combine_group_id,
    make_ledger_entry_id,
    make_package_application_id,
)
from tools.renovation_architecture.validators import (
    validate_complete_result,
    validate_package_review_result,
)
from tools.scene_classifier_passes import PassExecutionError

# The upstream phases the runtime measures around its stage calls; the
# reconciliation phase is measured here and total is the exact sum.
_UPSTREAM_PHASES = tuple(
    phase for phase in OBSERVABILITY_PHASES
    if phase not in ("reconciliation", "total")
)


def _reconciliation_failure(
    stage: str, message: str, *, code: str
) -> PassExecutionError:
    return PassExecutionError("reconciliation", stage, message, code=code)


# The legacy Phase B tie-break policy (rehab_packages._absorption_priority_key)
# adapted to v5 candidate dicts: room level before property, repair owns its
# evidence before modernization broad-absorbs it and turnover comes last,
# higher tier first, stable candidate ID last.
_ABSORPTION_CATEGORY_PRIORITY = {"repair": 0, "modernization": 1, "turnover": 2}


def absorption_priority_key(candidate: Mapping[str, Any]) -> Tuple[int, int, int, str]:
    level = 1 if candidate["package_level"] == "property" else 0
    category = _ABSORPTION_CATEGORY_PRIORITY.get(
        str(candidate["package_category"]), 3
    )
    tier_rank = _MODERNIZATION_TIER_RANK.get(str(candidate["pricing_tier"]), -1)
    return (level, category, -tier_rank, str(candidate["package_candidate_id"]))


def _combine_groups(
    decisions_by_candidate: Mapping[str, Mapping[str, Any]], *, estimate_id: str
) -> Dict[str, str]:
    """candidate_id -> deterministic combine-group ID over the transitive
    closure of the undirected combine edges. Membership only — combine is
    non-economic by policy."""
    adjacency: Dict[str, set] = {}
    for candidate_id, decision in decisions_by_candidate.items():
        for other_id in decision["combine_with"]:
            adjacency.setdefault(candidate_id, set()).add(other_id)
            adjacency.setdefault(other_id, set()).add(candidate_id)
    group_by_candidate: Dict[str, str] = {}
    seen: set = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        component: List[str] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(adjacency.get(node, ()))
        group_id = make_combine_group_id(
            estimate_id=estimate_id, member_candidate_ids=component
        )
        for member in component:
            group_by_candidate[member] = group_id
    return group_by_candidate


def compute_reconciliation(
    result: Mapping[str, Any], *, estimate_id: str
) -> Dict[str, Any]:
    """The deterministic reconciliation over a valid package-review result.

    Input-order independent: eligible candidates are walked in the absorption
    priority order and children in sorted order, so a shared child is owned by
    exactly one package regardless of list ordering. Returns plain dict
    sections: package_applications, coverage_ledger, totals."""
    candidates = {
        candidate["package_candidate_id"]: candidate
        for candidate in result["package_candidates"]
    }
    decisions_by_candidate = {
        decision["package_candidate_id"]: decision
        for decision in result["package_decisions"]
    }
    active_work = {
        item["work_item_id"]: item
        for item in result["work_items"]
        if item["status"] == "active"
    }
    group_by_candidate = _combine_groups(
        decisions_by_candidate, estimate_id=estimate_id
    )

    def _eligible(candidate_id: str) -> bool:
        decision = decisions_by_candidate[candidate_id]
        return (
            decision["decision"] == "approve"
            and not candidates[candidate_id]["display_only"]
            and not decision["split_groups"]
        )

    # At-most-once ownership: the first eligible candidate in priority order
    # that lists a child owns it.
    owner_by_child: Dict[str, str] = {}
    ordered = sorted(
        (candidates[cid] for cid in candidates if _eligible(cid)),
        key=absorption_priority_key,
    )
    for candidate in ordered:
        for child_id in sorted(candidate["child_work_item_ids"]):
            owner_by_child.setdefault(child_id, candidate["package_candidate_id"])

    applications: List[Dict[str, Any]] = []
    for candidate_id in sorted(candidates):
        candidate = candidates[candidate_id]
        decision = decisions_by_candidate[candidate_id]
        children = sorted(candidate["child_work_item_ids"])
        owned = [c for c in children if owner_by_child.get(c) == candidate_id]
        unowned = [c for c in children if owner_by_child.get(c) != candidate_id]
        if candidate["display_only"]:
            status, reason = "display_only", "display_only_aggregate"
        elif decision["decision"] == "reject":
            status, reason = "not_applied", "decision_rejected"
        elif decision["decision"] == "uncertain":
            status, reason = "not_applied", "decision_uncertain"
        elif decision["split_groups"]:
            status, reason = "not_applied", "split_recommended"
        elif not owned:
            # Every child went to a higher-priority approved package; billing
            # the tier floor with no owned work would double-count.
            status, reason = "not_applied", "no_owned_children"
        else:
            status, reason = "applied", "approved_absorbs_children"
        if status == "applied":
            effective_low = max(
                candidate["unfloored_low"],
                sum(active_work[c]["low"] for c in owned),
            )
            effective_high = max(
                candidate["unfloored_high"],
                sum(active_work[c]["high"] for c in owned),
            )
        else:
            effective_low = effective_high = 0
        applications.append({
            "application_id": make_package_application_id(
                estimate_id=estimate_id, package_candidate_id=candidate_id
            ),
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "package_candidate_id": candidate_id,
            "decision_id": decision["decision_id"],
            "status": status,
            "reason_code": reason,
            "absorbed_work_item_ids": owned if status == "applied" else [],
            "unabsorbed_child_work_item_ids": (
                unowned if status == "applied" else children
            ),
            "combine_group_id": group_by_candidate.get(candidate_id),
            "effective_low": effective_low,
            "effective_high": effective_high,
        })

    ledger: List[Dict[str, Any]] = []
    for work_id in sorted(active_work):
        owner_id = owner_by_child.get(work_id)
        work = active_work[work_id]
        if owner_id is not None:
            representation = "absorbed_by_package"
            package_id: Optional[str] = owner_id
            reason = "absorbed_by_approved_package"
            low = high = 0
        else:
            # No eligible candidate covers this work item (an eligible one
            # would own it), so the reason comes from the highest-priority
            # non-eligible covering candidate, if any.
            representation = "standalone"
            package_id = None
            low, high = work["low"], work["high"]
            covering = [
                candidates[cid]
                for cid in candidates
                if work_id in candidates[cid]["child_work_item_ids"]
            ]
            if not covering:
                reason = "no_covering_package"
            else:
                first = min(covering, key=absorption_priority_key)
                decision = decisions_by_candidate[first["package_candidate_id"]]
                if decision["decision"] == "reject":
                    reason = "package_rejected"
                elif decision["decision"] == "uncertain":
                    reason = "package_uncertain"
                else:
                    reason = "package_split"
        ledger.append({
            "entry_id": make_ledger_entry_id(
                estimate_id=estimate_id, work_item_id=work_id
            ),
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "work_item_id": work_id,
            "representation": representation,
            "package_id": package_id,
            "reason_code": reason,
            "low": low,
            "high": high,
        })

    applied = [app for app in applications if app["status"] == "applied"]
    standalone_entries = [
        entry for entry in ledger if entry["representation"] == "standalone"
    ]
    inspection_entries = [
        entry for entry in ledger if entry["representation"] == "inspection"
    ]
    lanes = {
        "standalone": (
            sum(entry["low"] for entry in standalone_entries),
            sum(entry["high"] for entry in standalone_entries),
        ),
        "packaged": (
            sum(app["effective_low"] for app in applied),
            sum(app["effective_high"] for app in applied),
        ),
        "inspection": (
            sum(entry["low"] for entry in inspection_entries),
            sum(entry["high"] for entry in inspection_entries),
        ),
    }
    headline = (
        sum(low for low, _ in lanes.values()),
        sum(high for _, high in lanes.values()),
    )
    totals = {
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "currency": "USD",
        **{
            lane: {"low": low, "high": high}
            for lane, (low, high) in lanes.items()
        },
        "headline": {"low": headline[0], "high": headline[1]},
    }
    return {
        "package_applications": applications,
        "coverage_ledger": ledger,
        "totals": totals,
    }


def recompute_reconciliation_audit(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Independent defect scan over the finished sections (never a re-run of
    compute_reconciliation, so a bug there cannot vouch for itself). Every
    list must come back empty for a complete result."""
    candidates = {
        candidate["package_candidate_id"]: candidate
        for candidate in result["package_candidates"]
    }
    decisions_by_candidate = {
        decision["package_candidate_id"]: decision
        for decision in result["package_decisions"]
    }
    active_work = {
        item["work_item_id"]: item
        for item in result["work_items"]
        if item["status"] == "active"
    }
    applications = result["package_applications"]
    ledger = result["coverage_ledger"]

    lost = sorted(
        set(active_work) - {entry["work_item_id"] for entry in ledger}
    )

    duplicate: List[str] = []
    owners_by_child: Dict[str, List[str]] = {}
    for app in applications:
        for child_id in app["absorbed_work_item_ids"]:
            owners_by_child.setdefault(child_id, []).append(
                app["package_candidate_id"]
            )
    for child_id in sorted(owners_by_child):
        owners = owners_by_child[child_id]
        if len(owners) > 1:
            duplicate.append(f"{child_id} absorbed by {sorted(owners)}")
    entry_work_ids = [entry["work_item_id"] for entry in ledger]
    for work_id in sorted(set(entry_work_ids)):
        if entry_work_ids.count(work_id) > 1:
            duplicate.append(f"{work_id} has multiple ledger entries")

    unsupported: List[str] = []
    orphans: List[str] = []
    applications_by_candidate = {
        app["package_candidate_id"]: app for app in applications
    }
    for app in applications:
        candidate_id = app["package_candidate_id"]
        candidate = candidates.get(candidate_id)
        decision = decisions_by_candidate.get(candidate_id)
        if candidate is None or decision is None:
            orphans.append(f"{app['application_id']} references unknown candidate")
            continue
        children = set(candidate["child_work_item_ids"])
        for child_id in app["absorbed_work_item_ids"]:
            if child_id not in children:
                orphans.append(
                    f"{candidate_id} absorbs {child_id} outside its children"
                )
            elif child_id not in active_work:
                orphans.append(f"{candidate_id} absorbs non-active {child_id}")
        for child_id in app["unabsorbed_child_work_item_ids"]:
            if child_id not in children:
                orphans.append(
                    f"{candidate_id} lists foreign unabsorbed child {child_id}"
                )
        if app["status"] == "applied":
            if (
                decision["decision"] != "approve"
                or decision["split_groups"]
                or candidate["display_only"]
            ):
                unsupported.append(f"{candidate_id} billed without approval basis")
            if not app["absorbed_work_item_ids"]:
                unsupported.append(f"{candidate_id} billed with no owned children")
            owned = [
                c for c in app["absorbed_work_item_ids"] if c in active_work
            ]
            expected = (
                max(
                    candidate["unfloored_low"],
                    sum(active_work[c]["low"] for c in owned),
                ),
                max(
                    candidate["unfloored_high"],
                    sum(active_work[c]["high"] for c in owned),
                ),
            )
            if (app["effective_low"], app["effective_high"]) != expected:
                unsupported.append(
                    f"{candidate_id} effective range is not "
                    "max(unfloored tier, owned child sum)"
                )
        elif (app["effective_low"], app["effective_high"]) != (0, 0):
            unsupported.append(f"{candidate_id} non-applied but carries dollars")
    for entry in ledger:
        work = active_work.get(entry["work_item_id"])
        if work is None:
            orphans.append(
                f"{entry['entry_id']} covers unknown or suppressed work "
                f"{entry['work_item_id']}"
            )
            continue
        if entry["representation"] == "standalone":
            if (entry["low"], entry["high"]) != (work["low"], work["high"]):
                unsupported.append(
                    f"{entry['work_item_id']} standalone entry does not equal "
                    "its work item allowance"
                )
        elif entry["representation"] == "absorbed_by_package":
            if (entry["low"], entry["high"]) != (0, 0):
                unsupported.append(
                    f"{entry['work_item_id']} absorbed entry carries dollars"
                )
            owner_app = applications_by_candidate.get(entry["package_id"])
            if (
                owner_app is None
                or owner_app["status"] != "applied"
                or entry["work_item_id"] not in owner_app["absorbed_work_item_ids"]
            ):
                unsupported.append(
                    f"{entry['work_item_id']} absorbed by a package that does "
                    "not bill it"
                )

    mismatches: List[str] = []
    applied = [app for app in applications if app["status"] == "applied"]
    expected_lanes = {
        "standalone": (
            sum(e["low"] for e in ledger if e["representation"] == "standalone"),
            sum(e["high"] for e in ledger if e["representation"] == "standalone"),
        ),
        "packaged": (
            sum(app["effective_low"] for app in applied),
            sum(app["effective_high"] for app in applied),
        ),
        "inspection": (
            sum(e["low"] for e in ledger if e["representation"] == "inspection"),
            sum(e["high"] for e in ledger if e["representation"] == "inspection"),
        ),
    }
    totals = result["totals"]
    for lane, (low, high) in expected_lanes.items():
        recorded = totals[lane]
        if (recorded["low"], recorded["high"]) != (low, high):
            mismatches.append(f"{lane} must be {low}/{high}")
    headline = (
        sum(low for low, _ in expected_lanes.values()),
        sum(high for _, high in expected_lanes.values()),
    )
    recorded_headline = totals["headline"]
    if (recorded_headline["low"], recorded_headline["high"]) != headline:
        mismatches.append(f"headline must be {headline[0]}/{headline[1]}")

    return {
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "lost_work_item_ids": lost,
        "duplicate_absorption": duplicate,
        "unsupported_billing": unsupported,
        "orphan_children": orphans,
        "arithmetic_mismatches": mismatches,
    }


def build_observability(
    result: Mapping[str, Any], *, phase_timings_ms: Mapping[str, int]
) -> Dict[str, Any]:
    """Assemble the observability block. phase_timings_ms must carry exactly
    the six OBSERVABILITY_PHASES; tokens and funnel are recomputed from the
    result sections (and re-verified by the artifact gate)."""
    dispositions = [d["disposition"] for d in result["condition_dispositions"]]
    applications = result["package_applications"]
    ledger = result["coverage_ledger"]
    statuses = [app["status"] for app in applications]
    representations = [entry["representation"] for entry in ledger]
    funnel = {
        "observations": sum(
            len(condition["issue_ids"])
            for condition in result["observed_conditions"]
        ),
        "conditions": len(result["observed_conditions"]),
        "condition_reviews": len(result["condition_reviews"]),
        "dispositions_accepted_for_work": dispositions.count("accepted_for_work"),
        "dispositions_excluded": dispositions.count("excluded"),
        "dispositions_inspection": dispositions.count("inspection"),
        "dispositions_withheld": dispositions.count("withheld"),
        "dispositions_no_action": dispositions.count("no_action"),
        "work_items_active": sum(
            1 for item in result["work_items"] if item["status"] == "active"
        ),
        "work_items_suppressed": sum(
            1 for item in result["work_items"] if item["status"] == "suppressed"
        ),
        "package_candidates": len(result["package_candidates"]),
        "package_decisions": len(result["package_decisions"]),
        "applications_applied": statuses.count("applied"),
        "applications_not_applied": statuses.count("not_applied"),
        "applications_display_only": statuses.count("display_only"),
        "ledger_standalone": representations.count("standalone"),
        "ledger_absorbed": representations.count("absorbed_by_package"),
        "ledger_entries": len(ledger),
    }
    assert set(funnel) == set(FUNNEL_KEYS)
    terra_tokens = result["terra_listing_usage"]["total_tokens"]
    sol_tokens = result["sol_listing_usage"]["total_tokens"]
    return {
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "phase_timings_ms": dict(phase_timings_ms),
        "terra_total_tokens": terra_tokens,
        "sol_total_tokens": sol_tokens,
        "combined_total_tokens": terra_tokens + sol_tokens,
        "funnel": funnel,
    }


def build_complete_result(
    package_review_result: Mapping[str, Any],
    *,
    estimate_id: str,
    phase_timings_ms: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Produce the validated complete result dict.

    The input is re-validated by the frozen Session 4 gate and never mutated;
    every Session 4 key (including the snapshot fingerprints, which cover
    sections this stage does not touch) carries through unchanged. Dirty
    audits and self-validation failures raise typed operational failures —
    a complete result with defects must never exist."""
    started = time.monotonic()
    validation = validate_package_review_result(
        package_review_result, estimate_id=estimate_id
    )
    if not validation.ok:
        raise _reconciliation_failure(
            "dependency",
            "reconciliation requires a valid package-review result: "
            + "; ".join(validation.errors[:5]),
            code="PackageReviewResultInvalid",
        )

    upstream = dict(phase_timings_ms or {})
    unknown = set(upstream) - set(_UPSTREAM_PHASES)
    if unknown:
        raise _reconciliation_failure(
            "dependency",
            f"unknown phase timings {sorted(unknown)}; expected a subset of "
            f"{list(_UPSTREAM_PHASES)}",
            code="PhaseTimingsInvalid",
        )

    sections = compute_reconciliation(
        package_review_result, estimate_id=estimate_id
    )
    result: Dict[str, Any] = {**package_review_result, **sections}

    audit = recompute_reconciliation_audit(result)
    defects = [
        f"{name}: {values[:3]}"
        for name, values in audit.items()
        if name != "schema_version" and values
    ]
    if defects:
        # stage "parse" -> failure category "parse", the S4 self-validation
        # precedent for an internally assembled contract violation.
        raise _reconciliation_failure(
            "parse",
            "reconciliation audit found defects: " + "; ".join(defects),
            code="ReconciliationAuditFailure",
        )
    result["reconciliation_audit"] = audit

    timings = {phase: int(upstream.get(phase, 0)) for phase in _UPSTREAM_PHASES}
    timings["reconciliation"] = int((time.monotonic() - started) * 1000)
    timings["total"] = sum(timings.values())
    result["observability"] = build_observability(
        result, phase_timings_ms=timings
    )

    self_check = validate_complete_result(result, estimate_id=estimate_id)
    if not self_check.ok:
        raise _reconciliation_failure(
            "parse",
            "assembled complete result failed self-validation: "
            + "; ".join(self_check.errors[:5]),
            code="CompleteResultInvalid",
        )
    return result
