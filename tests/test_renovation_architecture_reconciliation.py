"""Session 5 reconciliation tests: deterministic decision application,
at-most-once ownership under the legacy absorption priority, exact standalone
fallback, non-economic combine groups, ledger/totals arithmetic, audits, and
observability.

Fixture builders are shared with tests/test_renovation_architecture_contracts.py
(the established cross-test-module reuse pattern); the multi-unit builders here
compose them into valid Session 4 package-review results.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_reconciliation.py -q
"""
import copy

import pytest

from tests.test_renovation_architecture_contracts import (
    EST_ID,
    _candidate,
    _condition,
    _decision,
    _lattice,
    _sol_call,
    _snapshots,
    _standalone_estimate,
    _terra_call,
    _TOKEN_FIELDS,
    _unit_usage,
    _work_item,
)
from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    WHOLE_HOME_PACKAGE_TYPE,
    WHOLE_HOME_PRICING_PROFILE,
    WHOLE_HOME_PRICING_TIER,
    WHOLE_HOME_UNIT_ID,
)
from tools.renovation_architecture.ids import (
    make_combine_group_id,
    make_package_candidate_id,
)
from tools.renovation_architecture.reconciliation import build_complete_result
from tools.renovation_architecture.validators import validate_complete_result
from tools.scene_classifier_passes import PassExecutionError

_TIMINGS = {
    "condition_review": 5, "standalone_estimate": 7,
    "package_candidates": 11, "sol_review": 13,
}


# ── fixture builders ─────────────────────────────────────────────────────────

def _base_result(units):
    """A valid Session 3 standalone result with one condition + one active
    work item per (catalog_item_id, unit_id, action, low, high) spec. Units
    must be distinct (the Terra call id keys on the estimate unit)."""
    conditions, evidence, reviews, dispositions = [], [], [], []
    calls, unit_usages, items = [], [], []
    for catalog_item_id, unit_id, action, low, high in units:
        condition = _condition(catalog_item_id, unit_id)
        ev, rv, dp = _lattice(condition)
        call = _terra_call(condition, input_tokens=500, output_tokens=80)
        conditions.append(condition)
        evidence.append(ev)
        reviews.append(rv)
        dispositions.append(dp)
        calls.append(call)
        unit_usages.append(_unit_usage(call))
        items.append(_work_item(condition, action, low, high))
    items.sort(key=lambda item: item["work_item_id"])
    listing = {
        "schema_version": CONTRACTS_SCHEMA_VERSION, "call_count": len(calls),
    }
    for name in _TOKEN_FIELDS:
        listing[name] = sum(call[name] for call in calls)
    return {
        "observed_conditions": conditions,
        "evidence_facts": evidence,
        "condition_reviews": reviews,
        "condition_dispositions": dispositions,
        "terra_calls": calls,
        "terra_unit_usage": unit_usages,
        "terra_listing_usage": listing,
        "work_items": items,
        "work_dedup_collisions": [],
        "standalone_estimate": _standalone_estimate(items),
    }


def _item(base, unit_id):
    """The work item derived from the condition in the given unit."""
    return next(
        item for item in base["work_items"]
        if unit_id in item["source_estimate_unit_ids"]
    )


def _with_packages(base, candidates, decisions):
    """The Session 4 result: base + candidates/decisions + one Sol call (when
    candidates exist) + reconciled usage + snapshots."""
    if candidates:
        call = _sol_call(candidates)
        sol_calls = [call]
        listing = {
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "call_count": 1,
            "input_tokens": call["input_tokens"],
            "cached_input_tokens": call["cached_input_tokens"],
            "output_tokens": call["output_tokens"],
            "total_tokens": call["total_tokens"],
        }
    else:
        sol_calls = []
        listing = {
            "schema_version": CONTRACTS_SCHEMA_VERSION, "call_count": 0,
            "input_tokens": 0, "cached_input_tokens": 0,
            "output_tokens": 0, "total_tokens": 0,
        }
    return {
        **base,
        "package_candidates": candidates,
        "package_decisions": sorted(
            decisions, key=lambda decision: decision["decision_id"]
        ),
        "sol_calls": sol_calls,
        "sol_listing_usage": listing,
        "package_review_snapshots": _snapshots(base, candidates),
    }


def _whole_home(contributors):
    low = sum(candidate["low"] for candidate in contributors)
    high = sum(candidate["high"] for candidate in contributors)
    return {
        "package_candidate_id": make_package_candidate_id(
            estimate_id=EST_ID, package_type=WHOLE_HOME_PACKAGE_TYPE,
            estimate_unit_id=WHOLE_HOME_UNIT_ID,
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "package_type": WHOLE_HOME_PACKAGE_TYPE,
        "package_category": "turnover",
        "package_level": "property",
        "room": "whole_home",
        "estimate_unit_id": WHOLE_HOME_UNIT_ID,
        "child_work_item_ids": [],
        "driver_work_item_ids": [],
        "support_work_item_ids": [],
        "strength": "strong",
        "pricing_profile": WHOLE_HOME_PRICING_PROFILE,
        "pricing_tier": WHOLE_HOME_PRICING_TIER,
        "absorption_scope": {
            "family": "whole_home", "groups": [], "trade_buckets": [],
            "components": [],
        },
        "proposed_treatment": "whole_home_turnover_aggregate",
        "unfloored_low": low,
        "unfloored_high": high,
        "cost_floor_applied": False,
        "low": low,
        "high": high,
        "display_only": True,
        "contributing_candidate_ids": sorted(
            candidate["package_candidate_id"] for candidate in contributors
        ),
    }


def _complete(review_result):
    return build_complete_result(
        review_result, estimate_id=EST_ID, phase_timings_ms=dict(_TIMINGS)
    )


def _ledger_by_work(result):
    return {
        entry["work_item_id"]: entry for entry in result["coverage_ledger"]
    }


def _apps_by_candidate(result):
    return {
        app["package_candidate_id"]: app
        for app in result["package_applications"]
    }


_THREE_UNITS = [
    ("outdated_or_damaged_cabinets", "kitchen_primary", "CABINETS_REPLACE", 1000, 3000),
    ("bathroom_vanity_worn", "bathroom_1", "VANITY_REPLACE", 500, 1500),
    ("worn_bedroom_floor", "bedroom_1", "FLOORING_REPAIR", 300, 900),
]


# ── parity: no billable package reproduces the exact standalone totals ───────

class TestStandaloneParity:
    def test_zero_candidates(self):
        base = _base_result(_THREE_UNITS[:2])
        result = _complete(_with_packages(base, [], []))
        assert result["package_applications"] == []
        headline = base["standalone_estimate"]["headline"]
        assert result["totals"]["standalone"] == headline
        assert result["totals"]["packaged"] == {"low": 0, "high": 0}
        assert result["totals"]["headline"] == headline
        ledger = _ledger_by_work(result)
        assert all(
            entry["reason_code"] == "no_covering_package"
            for entry in ledger.values()
        )

    @pytest.mark.parametrize("decision,reason", [
        ("reject", "package_rejected"), ("uncertain", "package_uncertain"),
    ])
    def test_all_non_approved(self, decision, reason):
        base = _base_result(_THREE_UNITS)
        p1 = _candidate(
            "kitchen_modernization", "kitchen_primary",
            [_item(base, "kitchen_primary"), _item(base, "bedroom_1")],
        )
        p2 = _candidate(
            "bathroom_repair", "bathroom_1", [_item(base, "bathroom_1")]
        )
        result = _complete(_with_packages(
            base, [p1, p2], [_decision(p1, decision), _decision(p2, decision)]
        ))
        apps = _apps_by_candidate(result)
        for app in apps.values():
            assert app["status"] == "not_applied"
            assert app["reason_code"] == f"decision_{'rejected' if decision == 'reject' else 'uncertain'}"
            assert (app["effective_low"], app["effective_high"]) == (0, 0)
            assert app["absorbed_work_item_ids"] == []
        ledger = _ledger_by_work(result)
        for item in base["work_items"]:
            entry = ledger[item["work_item_id"]]
            assert entry["representation"] == "standalone"
            assert entry["reason_code"] == reason
            assert (entry["low"], entry["high"]) == (item["low"], item["high"])
        headline = base["standalone_estimate"]["headline"]
        assert result["totals"]["standalone"] == headline
        assert result["totals"]["packaged"] == {"low": 0, "high": 0}
        assert result["totals"]["headline"] == headline

    def test_approved_split_recommendation_is_non_billable(self):
        """The conservative policy: a split recommendation on an approved
        candidate removes the package treatment; children keep their exact
        standalone allowances. No derived-package pricing is invented."""
        base = _base_result(_THREE_UNITS[:2])
        w_kitchen = _item(base, "kitchen_primary")
        w_bath = _item(base, "bathroom_1")
        p1 = _candidate(
            "kitchen_modernization", "kitchen_primary", [w_kitchen, w_bath]
        )
        d1 = _decision(p1, "approve", split_groups=sorted(
            ([w_kitchen["work_item_id"]], [w_bath["work_item_id"]]), key=tuple
        ))
        result = _complete(_with_packages(base, [p1], [d1]))
        (app,) = result["package_applications"]
        assert app["status"] == "not_applied"
        assert app["reason_code"] == "split_recommended"
        ledger = _ledger_by_work(result)
        for item in (w_kitchen, w_bath):
            entry = ledger[item["work_item_id"]]
            assert entry["representation"] == "standalone"
            assert entry["reason_code"] == "package_split"
            assert (entry["low"], entry["high"]) == (item["low"], item["high"])
        headline = base["standalone_estimate"]["headline"]
        assert result["totals"]["headline"] == headline


# ── QP3: opportunity-only interior modernization stays standalone ───────────

class TestOpportunityOnlyInteriorModernizationPolicy:
    @pytest.mark.parametrize("package_type,unit_id", [
        ("bedroom_modernization", "bedroom_1"),
        ("living_modernization", "living_room_primary"),
    ])
    @pytest.mark.parametrize("treatment", [
        "opportunity_driver_with_corroboration",
        "opportunity_driver_with_multiphoto_corroboration",
    ])
    def test_approved_candidate_is_retained_but_not_applied(
        self, package_type, unit_id, treatment
    ):
        base = _base_result([
            ("dated_interior_trim", unit_id, "TRIM_REPLACE", 300, 900),
        ])
        work = _item(base, unit_id)
        candidate = _candidate(
            package_type,
            unit_id,
            [work],
            proposed_treatment=treatment,
            unfloored_low=1000,
            unfloored_high=3000,
        )
        result = _complete(_with_packages(
            base, [candidate], [_decision(candidate, "approve")]
        ))

        # QP3 is application-side: Sol's reviewed candidate and decision stay
        # intact, while its child keeps the exact standalone allowance.
        assert result["package_candidates"] == [candidate]
        assert result["package_decisions"][0]["decision"] == "approve"
        (app,) = result["package_applications"]
        assert app["status"] == "not_applied"
        assert app["reason_code"] == "opportunity_only_interior_modernization"
        assert app["absorbed_work_item_ids"] == []
        assert app["unabsorbed_child_work_item_ids"] == [work["work_item_id"]]
        assert (app["effective_low"], app["effective_high"]) == (0, 0)

        entry = _ledger_by_work(result)[work["work_item_id"]]
        assert entry["representation"] == "standalone"
        assert entry["reason_code"] == "opportunity_only_interior_modernization"
        assert (entry["low"], entry["high"]) == (work["low"], work["high"])
        assert result["totals"]["standalone"] == {"low": 300, "high": 900}
        assert result["totals"]["packaged"] == {"low": 0, "high": 0}
        assert result["totals"]["headline"] == {"low": 300, "high": 900}

    @pytest.mark.parametrize("treatment", [
        "package_driver",
        "multiple_package_support_same_estimate_unit",
    ])
    def test_non_opportunity_only_treatments_remain_eligible(self, treatment):
        unit_id = "bedroom_1"
        base = _base_result([
            ("dated_interior_trim", unit_id, "TRIM_REPLACE", 300, 900),
        ])
        work = _item(base, unit_id)
        overrides = {"proposed_treatment": treatment}
        if treatment == "multiple_package_support_same_estimate_unit":
            overrides.update({
                "driver_work_item_ids": [],
                "support_work_item_ids": [work["work_item_id"]],
            })
        candidate = _candidate(
            "bedroom_modernization", unit_id, [work], **overrides
        )
        result = _complete(_with_packages(
            base, [candidate], [_decision(candidate, "approve")]
        ))

        (app,) = result["package_applications"]
        assert app["status"] == "applied"
        assert app["reason_code"] == "approved_absorbs_children"

    @pytest.mark.parametrize("package_type,unit_id", [
        ("kitchen_modernization", "kitchen_primary"),
        ("bathroom_modernization", "bathroom_primary"),
    ])
    def test_other_modernization_rooms_remain_eligible(
        self, package_type, unit_id
    ):
        base = _base_result([
            ("dated_finishes", unit_id, "FINISH_REPLACE", 300, 900),
        ])
        work = _item(base, unit_id)
        candidate = _candidate(
            package_type,
            unit_id,
            [work],
            proposed_treatment="opportunity_driver_with_corroboration",
        )
        result = _complete(_with_packages(
            base, [candidate], [_decision(candidate, "approve")]
        ))

        (app,) = result["package_applications"]
        assert app["status"] == "applied"
        assert app["reason_code"] == "approved_absorbs_children"


# ── combine: non-economic grouping metadata ──────────────────────────────────

class TestCombineGroups:
    def test_members_share_a_group_id_and_keep_their_own_pricing(self):
        base = _base_result(_THREE_UNITS)
        p1 = _candidate(
            "kitchen_modernization", "kitchen_primary",
            [_item(base, "kitchen_primary")],
        )
        p2 = _candidate(
            "bathroom_modernization", "bathroom_1",
            [_item(base, "bathroom_1")],
        )
        d1 = _decision(p1, "approve", combine_with=[p2["package_candidate_id"]])
        d2 = _decision(p2, "approve", combine_with=[p1["package_candidate_id"]])
        result = _complete(_with_packages(base, [p1, p2], [d1, d2]))
        apps = _apps_by_candidate(result)
        expected_group = make_combine_group_id(
            estimate_id=EST_ID,
            member_candidate_ids=[
                p1["package_candidate_id"], p2["package_candidate_id"],
            ],
        )
        for candidate in (p1, p2):
            app = apps[candidate["package_candidate_id"]]
            assert app["status"] == "applied"
            assert app["combine_group_id"] == expected_group
            # Non-economic: each member bills its own deterministic range.
            assert (app["effective_low"], app["effective_high"]) == (
                candidate["low"], candidate["high"],
            )
        assert result["totals"]["packaged"] == {
            "low": p1["low"] + p2["low"], "high": p1["high"] + p2["high"],
        }
        # The bedroom item is uncovered and stays standalone.
        entry = _ledger_by_work(result)[_item(base, "bedroom_1")["work_item_id"]]
        assert entry["representation"] == "standalone"
        assert entry["reason_code"] == "no_covering_package"

    def test_group_id_is_order_independent(self):
        assert make_combine_group_id(
            estimate_id=EST_ID, member_candidate_ids=["pk1_a", "pk1_b"]
        ) == make_combine_group_id(
            estimate_id=EST_ID, member_candidate_ids=["pk1_b", "pk1_a"]
        )


# ── shared children: priority, order independence, partial ownership ─────────

class TestSharedChildren:
    def _priority_fixture(self):
        base = _base_result(_THREE_UNITS)
        shared = _item(base, "bedroom_1")
        # The tier floor sits below the kitchen child's own range, so the
        # partial-ownership test can observe the owned-children sum winning.
        p_mod = _candidate(
            "kitchen_modernization", "kitchen_primary",
            [_item(base, "kitchen_primary"), shared],
            unfloored_low=800, unfloored_high=2500,
        )
        p_rep = _candidate(
            "bathroom_repair", "bathroom_1",
            [_item(base, "bathroom_1"), shared],
        )
        return base, shared, p_mod, p_rep

    def test_repair_owns_the_shared_child_before_modernization(self):
        base, shared, p_mod, p_rep = self._priority_fixture()
        result = _complete(_with_packages(
            base, [p_mod, p_rep], [_decision(p_mod), _decision(p_rep)]
        ))
        apps = _apps_by_candidate(result)
        rep_app = apps[p_rep["package_candidate_id"]]
        mod_app = apps[p_mod["package_candidate_id"]]
        assert shared["work_item_id"] in rep_app["absorbed_work_item_ids"]
        assert shared["work_item_id"] in mod_app["unabsorbed_child_work_item_ids"]
        entry = _ledger_by_work(result)[shared["work_item_id"]]
        assert entry["package_id"] == p_rep["package_candidate_id"]
        assert entry["reason_code"] == "absorbed_by_approved_package"
        assert (entry["low"], entry["high"]) == (0, 0)

    def test_assignment_is_input_order_independent(self):
        base, _, p_mod, p_rep = self._priority_fixture()
        forward = _complete(_with_packages(
            base, [p_mod, p_rep], [_decision(p_mod), _decision(p_rep)]
        ))
        reversed_input = _complete(_with_packages(
            base, [p_rep, p_mod], [_decision(p_rep), _decision(p_mod)]
        ))
        for section in ("package_applications", "coverage_ledger", "totals"):
            assert forward[section] == reversed_input[section]

    def test_higher_tier_owns_the_shared_child_within_a_category(self):
        base = _base_result(_THREE_UNITS)
        shared = _item(base, "bedroom_1")
        p_full = _candidate(
            "kitchen_modernization", "kitchen_primary",
            [_item(base, "kitchen_primary"), shared],
            pricing_tier="full_rehab",
        )
        p_refresh = _candidate(
            "bathroom_modernization", "bathroom_1",
            [_item(base, "bathroom_1"), shared],
            pricing_tier="refresh",
        )
        result = _complete(_with_packages(
            base, [p_full, p_refresh], [_decision(p_full), _decision(p_refresh)]
        ))
        entry = _ledger_by_work(result)[shared["work_item_id"]]
        assert entry["package_id"] == p_full["package_candidate_id"]

    def test_partial_owner_bills_owned_children_only(self):
        """The applied effective range floors on actually-owned children —
        billing the stored candidate floor would double-count the shared
        child's dollars through two packages."""
        base, shared, p_mod, p_rep = self._priority_fixture()
        result = _complete(_with_packages(
            base, [p_mod, p_rep], [_decision(p_mod), _decision(p_rep)]
        ))
        apps = _apps_by_candidate(result)
        mod_app = apps[p_mod["package_candidate_id"]]
        rep_app = apps[p_rep["package_candidate_id"]]
        w_kitchen = _item(base, "kitchen_primary")
        w_bath = _item(base, "bathroom_1")
        # p_mod lost the shared child: effective = its remaining child only,
        # strictly below the stored floored range (which includes the child).
        assert (mod_app["effective_low"], mod_app["effective_high"]) == (
            w_kitchen["low"], w_kitchen["high"],
        )
        assert mod_app["effective_high"] < p_mod["high"]
        # p_rep owns both of its children: effective == stored range.
        assert (rep_app["effective_low"], rep_app["effective_high"]) == (
            w_bath["low"] + shared["low"], w_bath["high"] + shared["high"],
        )
        # Exact totals: every dollar exactly once.
        assert result["totals"]["standalone"] == {"low": 0, "high": 0}
        assert result["totals"]["headline"] == {
            "low": w_kitchen["low"] + w_bath["low"] + shared["low"],
            "high": w_kitchen["high"] + w_bath["high"] + shared["high"],
        }

    def test_partial_owner_keeps_its_tier_floor(self):
        """When the unfloored tier spec exceeds the owned-children sum, the
        partial owner still bills the tier floor (the floor is legitimate
        package pricing; only lost-child dollars are excluded)."""
        base = _base_result(_THREE_UNITS)
        shared = _item(base, "bedroom_1")
        w_kitchen = _item(base, "kitchen_primary")
        p_mod = _candidate(
            "kitchen_modernization", "kitchen_primary", [w_kitchen, shared],
            unfloored_low=5000, unfloored_high=9000,
        )
        p_rep = _candidate(
            "bathroom_repair", "bathroom_1",
            [_item(base, "bathroom_1"), shared],
        )
        result = _complete(_with_packages(
            base, [p_mod, p_rep], [_decision(p_mod), _decision(p_rep)]
        ))
        mod_app = _apps_by_candidate(result)[p_mod["package_candidate_id"]]
        assert (mod_app["effective_low"], mod_app["effective_high"]) == (5000, 9000)

    def test_zero_owned_children_is_non_billable(self):
        base = _base_result(_THREE_UNITS[:2])
        shared = _item(base, "kitchen_primary")
        p_rep = _candidate("kitchen_repair", "kitchen_primary", [shared])
        p_mod = _candidate("kitchen_modernization", "kitchen_primary", [shared])
        result = _complete(_with_packages(
            base, [p_rep, p_mod], [_decision(p_rep), _decision(p_mod)]
        ))
        apps = _apps_by_candidate(result)
        mod_app = apps[p_mod["package_candidate_id"]]
        assert mod_app["status"] == "not_applied"
        assert mod_app["reason_code"] == "no_owned_children"
        assert (mod_app["effective_low"], mod_app["effective_high"]) == (0, 0)
        entry = _ledger_by_work(result)[shared["work_item_id"]]
        assert entry["package_id"] == p_rep["package_candidate_id"]
        rep_app = apps[p_rep["package_candidate_id"]]
        assert result["totals"]["packaged"] == {
            "low": rep_app["effective_low"], "high": rep_app["effective_high"],
        }


# ── display-only whole-home aggregate ────────────────────────────────────────

class TestDisplayOnly:
    def test_aggregate_never_bills(self):
        base = _base_result(_THREE_UNITS[:2])
        p_k = _candidate(
            "kitchen_turnover", "kitchen_primary",
            [_item(base, "kitchen_primary")],
        )
        p_b = _candidate(
            "bathroom_turnover", "bathroom_1", [_item(base, "bathroom_1")]
        )
        aggregate = _whole_home([p_k, p_b])
        result = _complete(_with_packages(
            base, [p_k, p_b, aggregate],
            [_decision(p_k), _decision(p_b), _decision(aggregate)],
        ))
        apps = _apps_by_candidate(result)
        agg_app = apps[aggregate["package_candidate_id"]]
        assert agg_app["status"] == "display_only"
        assert agg_app["reason_code"] == "display_only_aggregate"
        assert (agg_app["effective_low"], agg_app["effective_high"]) == (0, 0)
        assert agg_app["absorbed_work_item_ids"] == []
        # Totals carry only the room packages; the aggregate adds nothing.
        assert result["totals"]["packaged"] == {
            "low": p_k["low"] + p_b["low"], "high": p_k["high"] + p_b["high"],
        }
        assert result["totals"]["standalone"] == {"low": 0, "high": 0}


# ── construction discipline ──────────────────────────────────────────────────

class TestConstruction:
    def test_input_result_is_not_mutated(self):
        base = _base_result(_THREE_UNITS[:2])
        p1 = _candidate(
            "kitchen_modernization", "kitchen_primary",
            [_item(base, "kitchen_primary")],
        )
        review = _with_packages(base, [p1], [_decision(p1)])
        frozen = copy.deepcopy(review)
        _complete(review)
        assert review == frozen

    def test_result_passes_the_complete_gate(self):
        base = _base_result(_THREE_UNITS[:2])
        p1 = _candidate(
            "kitchen_modernization", "kitchen_primary",
            [_item(base, "kitchen_primary")],
        )
        result = _complete(_with_packages(base, [p1], [_decision(p1)]))
        res = validate_complete_result(result, estimate_id=EST_ID)
        assert res.ok, res.errors

    def test_phase_timings_pass_through_and_total_is_the_sum(self):
        base = _base_result(_THREE_UNITS[:2])
        result = _complete(_with_packages(base, [], []))
        timings = result["observability"]["phase_timings_ms"]
        for phase, value in _TIMINGS.items():
            assert timings[phase] == value
        assert timings["total"] == sum(
            value for phase, value in timings.items() if phase != "total"
        )

    def test_invalid_input_is_a_typed_dependency_failure(self):
        with pytest.raises(PassExecutionError) as excinfo:
            build_complete_result({}, estimate_id=EST_ID)
        assert excinfo.value.code == "PackageReviewResultInvalid"
        assert excinfo.value.stage == "dependency"

    def test_unknown_phase_timings_are_rejected(self):
        base = _base_result(_THREE_UNITS[:2])
        review = _with_packages(base, [], [])
        with pytest.raises(PassExecutionError) as excinfo:
            build_complete_result(
                review, estimate_id=EST_ID, phase_timings_ms={"bogus": 1}
            )
        assert excinfo.value.code == "PhaseTimingsInvalid"

    def test_dirty_audit_fails_closed(self, monkeypatch):
        """A reconciliation-internal defect must raise, never publish: the
        audit recompute is independent of compute_reconciliation, so a broken
        computation cannot vouch for itself."""
        import tools.renovation_architecture.reconciliation as reconciliation

        real = reconciliation.compute_reconciliation

        def _broken(result, *, estimate_id):
            sections = real(result, estimate_id=estimate_id)
            sections["coverage_ledger"] = []  # lose every active work item
            return sections

        monkeypatch.setattr(
            reconciliation, "compute_reconciliation", _broken
        )
        base = _base_result(_THREE_UNITS[:2])
        review = _with_packages(base, [], [])
        with pytest.raises(PassExecutionError) as excinfo:
            build_complete_result(
                review, estimate_id=EST_ID, phase_timings_ms=dict(_TIMINGS)
            )
        assert excinfo.value.code == "ReconciliationAuditFailure"

    def test_hand_tampered_double_absorption_fails_the_gate(self):
        base = _base_result(_THREE_UNITS)
        shared = _item(base, "bedroom_1")
        p_mod = _candidate(
            "kitchen_modernization", "kitchen_primary",
            [_item(base, "kitchen_primary"), shared],
        )
        p_rep = _candidate(
            "bathroom_repair", "bathroom_1",
            [_item(base, "bathroom_1"), shared],
        )
        result = _complete(_with_packages(
            base, [p_mod, p_rep], [_decision(p_mod), _decision(p_rep)]
        ))
        mod_app = _apps_by_candidate(result)[p_mod["package_candidate_id"]]
        mod_app["absorbed_work_item_ids"] = sorted(
            mod_app["absorbed_work_item_ids"] + [shared["work_item_id"]]
        )
        mod_app["unabsorbed_child_work_item_ids"] = [
            child for child in mod_app["unabsorbed_child_work_item_ids"]
            if child != shared["work_item_id"]
        ]
        res = validate_complete_result(result, estimate_id=EST_ID)
        assert not res.ok
        assert any("absorbed by both" in error for error in res.errors)
