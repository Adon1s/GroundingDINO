"""Deterministic work-derivation and standalone-estimate tests (Session 3).

Pure-deterministic: no FakeOpenAI, no Terra. Synthetic Session 2 results come
from the tests/test_renovation_architecture_contracts builders; projections
come from real build_renovation_catalog_projection runs over synthetic v3.1
catalogs (tests/test_renovation_architecture_catalog builders), so every
pricing pin exercises the actual legacy costing core.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_work_items.py -q
"""
import copy
import json

import pytest

from tools.cost_factors import resolve_property_cost_factor
from tools.costing import compute_item_cost_range
from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    DEDUP_SUPPRESSION_REASON,
    WORK_DEDUP_POLICY_VERSION,
)
from tools.renovation_architecture.validators import (
    validate_standalone_estimate_result,
)
from tools.renovation_architecture.work_items import (
    _split_integer,
    derive_standalone_estimate,
)
from tools.scene_classifier_passes import PassExecutionError
from tests.test_renovation_architecture_catalog import (
    _build,
    _v31_catalog,
    _v31_item,
)
from tests.test_renovation_architecture_contracts import (
    _TOKEN_FIELDS,
    EST_ID,
    _assert_error,
    _condition,
    _lattice,
    _terra_call,
    _unit_usage,
)


def _manual_cost(base_low, base_high, per_low, per_high, cap_low=None, cap_high=None):
    return {
        "mode": "allowance",
        "cost_source": "manual",
        "base_low": base_low,
        "base_high": base_high,
        "per_occurrence_low": per_low,
        "per_occurrence_high": per_high,
        "cap_low": cap_low if cap_low is not None else base_high * 4,
        "cap_high": cap_high if cap_high is not None else base_high * 8,
    }


def _review_for(specs):
    """A valid Session 2 result from (item, unit[, disposition, hints,
    ambiguous]) specs: one lattice per condition, one Terra call per unit."""
    conditions, evidence, reviews, dispositions = [], [], [], []
    for spec in specs:
        condition = _condition(spec["item"], spec["unit"])
        if "hints" in spec:
            condition["opening_instance_hints"] = sorted(spec["hints"])
        if spec.get("ambiguous"):
            condition["identity_ambiguous"] = True
        disposition_value = spec.get("disposition", "accepted_for_work")
        ev, rv, dp = _lattice(condition, disposition=disposition_value)
        if disposition_value == "withheld":
            # 2 distinct views < 3 required -> the recompute gate agrees.
            ev["min_photo_evidence_required"] = 3
        conditions.append(condition)
        evidence.append(ev)
        reviews.append(rv)
        dispositions.append(dp)

    by_unit = {}
    for condition in conditions:
        by_unit.setdefault(condition["estimate_unit_id"], []).append(condition)
    calls, usages = [], []
    for unit in sorted(by_unit):
        call = _terra_call(by_unit[unit][0], input_tokens=1000, output_tokens=200)
        call["condition_ids"] = [c["condition_id"] for c in by_unit[unit]]
        calls.append(call)
        usages.append(_unit_usage(call))
    listing = {"schema_version": CONTRACTS_SCHEMA_VERSION, "call_count": len(calls)}
    for name in _TOKEN_FIELDS:
        listing[name] = sum(usage[name] for usage in usages)
    return {
        "observed_conditions": conditions,
        "evidence_facts": evidence,
        "condition_reviews": reviews,
        "condition_dispositions": dispositions,
        "terra_calls": calls,
        "terra_unit_usage": usages,
        "terra_listing_usage": listing,
    }


def _derive(tmp_path, specs, *items, metadata=None):
    projection = _build(_v31_catalog(*items), tmp_path)
    review = _review_for(specs)
    result = derive_standalone_estimate(
        review_result=review,
        projection=projection,
        property_metadata=metadata,
        estimate_id=EST_ID,
    )
    return result, review, projection


def _active(result):
    return [w for w in result["work_items"] if w["status"] == "active"]


def _suppressed(result):
    return [w for w in result["work_items"] if w["status"] == "suppressed"]


# ── unit-policy semantics ────────────────────────────────────────────────────

class TestUnitSemantics:
    @pytest.mark.parametrize(
        "policy,token",
        [("per_property", "property"), ("per_system", "system"), ("per_area", "area")],
    )
    def test_collapse_policies_bill_one_unit(self, tmp_path, policy, token):
        """Conditions in two physical units collapse to ONE billable unit
        priced at N=1, with both source units in the lineage."""
        result, _, _ = _derive(
            tmp_path,
            [{"item": "roof_probe", "unit": "area_front"},
             {"item": "roof_probe", "unit": "area_back"}],
            _v31_item(
                "roof_probe",
                cost=_manual_cost(1000, 3000, 500, 1000),
                estimate={"estimate_tier": "high", "unit_policy": policy},
            ),
        )
        (item,) = _active(result)
        assert item["billable_unit_id"] == token
        assert item["unit_count"] == 1
        assert item["source_estimate_unit_ids"] == ["area_back", "area_front"]
        assert len(item["condition_ids"]) == 2
        # N=1: base only, manual allowance -> no multipliers, neutral factor.
        assert (item["low"], item["high"]) == (1000, 3000)

    def test_room_like_splits_the_aggregate_across_units(self, tmp_path):
        """Two bathrooms: one aggregate base + 1 x per_occurrence allowance,
        split deterministically (odd remainder to the first sorted unit)."""
        result, _, _ = _derive(
            tmp_path,
            [{"item": "vanity_probe", "unit": "bathroom_1"},
             {"item": "vanity_probe", "unit": "bathroom_2"}],
            _v31_item(
                "vanity_probe",
                cost=_manual_cost(1001, 3000, 500, 1000),
                estimate={"estimate_tier": "minor", "unit_policy": "per_bathroom"},
            ),
        )
        items = _active(result)
        assert [w["billable_unit_id"] for w in items] == ["bathroom_1", "bathroom_2"]
        assert all(w["unit_count"] == 1 for w in items)
        # Aggregate: (1001 + 500, 3000 + 1000) = (1501, 4000).
        assert [w["low"] for w in items] == [751, 750]
        assert [w["high"] for w in items] == [2000, 2000]
        assert sum(w["low"] for w in items) == 1501
        assert sum(w["high"] for w in items) == 4000

    def test_room_like_single_unit_prices_base_only(self, tmp_path):
        result, _, _ = _derive(
            tmp_path,
            [{"item": "vanity_probe", "unit": "bathroom_1"}],
            _v31_item(
                "vanity_probe",
                cost=_manual_cost(1000, 3000, 500, 1000),
                estimate={"estimate_tier": "minor", "unit_policy": "per_bathroom"},
            ),
        )
        (item,) = _active(result)
        assert (item["low"], item["high"]) == (1000, 3000)

    def test_aggregate_caps_bound_the_split(self, tmp_path):
        """Three rooms would price 1000 + 2x500 = 2000, but cap_low 1800
        truncates the aggregate BEFORE the split — legacy cap semantics."""
        result, _, _ = _derive(
            tmp_path,
            [{"item": "wall_probe", "unit": "room_a"},
             {"item": "wall_probe", "unit": "room_b"},
             {"item": "wall_probe", "unit": "room_c"}],
            _v31_item(
                "wall_probe",
                cost=_manual_cost(1000, 3000, 500, 1000, cap_low=1800, cap_high=4500),
                estimate={"estimate_tier": "minor", "unit_policy": "per_room"},
            ),
        )
        items = _active(result)
        assert [w["low"] for w in items] == [600, 600, 600]
        # High: 3000 + 2x1000 = 5000 -> capped at 4500 -> [1500, 1500, 1500].
        assert [w["high"] for w in items] == [1500, 1500, 1500]

    def test_per_opening_counts_explicit_hints_within_the_unit(self, tmp_path):
        """Hints price base + (count-1) x per_occurrence inside the unit; a
        hintless condition falls back to the conservative single opening.
        Openings are unit-local: the two conditions stay separate items."""
        result, _, _ = _derive(
            tmp_path,
            [{"item": "window_probe", "unit": "room_a",
              "hints": ["window_id:w1", "window_id:w2", "window_id:w3"]},
             {"item": "window_probe", "unit": "room_b"}],
            _v31_item(
                "window_probe",
                cost=_manual_cost(400, 900, 200, 500),
                estimate={"estimate_tier": "minor", "unit_policy": "per_opening"},
            ),
        )
        items = _active(result)
        by_unit = {w["billable_unit_id"]: w for w in items}
        assert by_unit["room_a"]["unit_count"] == 3
        assert (by_unit["room_a"]["low"], by_unit["room_a"]["high"]) == (800, 1900)
        assert by_unit["room_b"]["unit_count"] == 1
        assert (by_unit["room_b"]["low"], by_unit["room_b"]["high"]) == (400, 900)

    def test_per_scope_one_item_per_condition_no_split(self, tmp_path):
        """per_scope conditions each price a full base — no aggregate, no
        occurrence discount across units."""
        result, _, _ = _derive(
            tmp_path,
            [{"item": "scope_probe", "unit": "room_a"},
             {"item": "scope_probe", "unit": "room_b"}],
            _v31_item("scope_probe", cost=_manual_cost(500, 1200, 100, 300)),
        )
        items = _active(result)
        assert all((w["low"], w["high"]) == (500, 1200) for w in items)
        assert all(w["unit_count"] == 1 for w in items)
        assert {w["billable_unit_id"] for w in items} == {"room_a", "room_b"}


# ── pricing ──────────────────────────────────────────────────────────────────

class TestPricing:
    def test_manual_allowance_skips_multipliers(self, tmp_path):
        """cost_source=manual means the catalog dollars ARE the price — no
        kind/scope/trade multipliers (78/128 shipped items behave this way)."""
        result, _, projection = _derive(
            tmp_path,
            [{"item": "manual_probe", "unit": "room_a"}],
            _v31_item("manual_probe", cost=_manual_cost(1000, 3000, 100, 200)),
        )
        (item,) = _active(result)
        assert (item["low"], item["high"]) == (1000, 3000)
        assert item["pricing_modes"] == ["catalog_allowance"]

    def test_heuristic_pricing_hand_pin(self, tmp_path):
        """The default synthetic item (degradation, severity 2, repair,
        flooring, heuristic): base (300, 1500), kind 1.0 x scope 1.0 x trade
        flooring 0.9 -> 270/1350."""
        result, _, _ = _derive(
            tmp_path,
            [{"item": "heuristic_probe", "unit": "room_a"}],
            _v31_item("heuristic_probe"),
        )
        (item,) = _active(result)
        assert (item["low"], item["high"]) == (270, 1350)
        assert item["pricing_modes"] == ["heuristic"]

    def test_pricing_matches_the_legacy_core_exactly(self, tmp_path):
        """The derived dollars are compute_item_cost_range verbatim (neutral
        factor), for both the heuristic and manual paths."""
        cost = _manual_cost(1000, 3000, 500, 1000)
        result, _, projection = _derive(
            tmp_path,
            [{"item": "manual_probe", "unit": "bathroom_1"},
             {"item": "manual_probe", "unit": "bathroom_2"},
             {"item": "heuristic_probe", "unit": "room_a"}],
            _v31_item(
                "manual_probe",
                cost=cost,
                estimate={"estimate_tier": "minor", "unit_policy": "per_bathroom"},
            ),
            _v31_item("heuristic_probe"),
        )
        manual_expected = compute_item_cost_range(
            cost_obj=cost, n_occurrences=2, kind="degradation",
            scope="repair", trade_bucket="flooring", severity=2,
        )
        manual_items = [
            w for w in _active(result) if w["catalog_item_ids"] == ["manual_probe"]
        ]
        assert sum(w["low"] for w in manual_items) == manual_expected[0]
        assert sum(w["high"] for w in manual_items) == manual_expected[1]
        heuristic_expected = compute_item_cost_range(
            cost_obj=None, n_occurrences=1, kind="degradation",
            scope="repair", trade_bucket="flooring", severity=2,
        )
        (heuristic_item,) = [
            w for w in _active(result) if w["catalog_item_ids"] == ["heuristic_probe"]
        ]
        assert (heuristic_item["low"], heuristic_item["high"]) == heuristic_expected

    def test_strategy_maps_to_costing_scope(self, tmp_path):
        """replace_only prices with scope 'replace' (SCOPE_MULT 1.3), exactly
        like legacy POSTURE_TO_SCOPE, even when the catalog scope is repair."""
        result, _, _ = _derive(
            tmp_path,
            [{"item": "strategy_probe", "unit": "room_a"}],
            _v31_item(
                "strategy_probe",
                estimate={"estimate_tier": "minor", "strategy": "replace_only"},
            ),
        )
        (item,) = _active(result)
        expected = compute_item_cost_range(
            cost_obj=None, n_occurrences=1, kind="degradation",
            scope="replace", trade_bucket="flooring", severity=2,
        )
        assert (item["low"], item["high"]) == expected

    def test_property_factor_applied_once_before_split(self, tmp_path):
        """A non-neutral factor scales the AGGREGATE once, then the split
        preserves the scaled sum exactly — no per-part rounding drift."""
        metadata = {"area_price_per_sqft": 460, "sqft": 3600}
        factor, _ = resolve_property_cost_factor(metadata)
        assert factor != 1.0
        cost = _manual_cost(1001, 3000, 500, 1000)
        result, _, _ = _derive(
            tmp_path,
            [{"item": "vanity_probe", "unit": "bathroom_1"},
             {"item": "vanity_probe", "unit": "bathroom_2"}],
            _v31_item(
                "vanity_probe",
                cost=cost,
                estimate={"estimate_tier": "minor", "unit_policy": "per_bathroom"},
            ),
            metadata=metadata,
        )
        items = _active(result)
        assert sum(w["low"] for w in items) == int(round(1501 * factor))
        assert sum(w["high"] for w in items) == int(round(4000 * factor))
        standalone = result["standalone_estimate"]
        assert standalone["property_cost_factor"] == factor
        assert standalone["property_cost_factor_audit"]["ppsf"] == 460
        assert standalone["headline"]["low"] == int(round(1501 * factor))

    def test_missing_metadata_is_neutral_with_audit_reasons(self, tmp_path):
        result, _, _ = _derive(
            tmp_path,
            [{"item": "manual_probe", "unit": "room_a"}],
            _v31_item("manual_probe", cost=_manual_cost(1000, 3000, 100, 200)),
        )
        standalone = result["standalone_estimate"]
        assert standalone["property_cost_factor"] == 1.0
        assert set(standalone["property_cost_factor_audit"]["reasons"]) == {
            "no_ppsf_signal_market_factor_neutral",
            "no_sqft_size_factor_neutral",
        }

    def test_split_integer_matches_legacy_reconciliation_split(self):
        from tools.rehab_packages import _split_integer as legacy_split

        for amount, parts in ((100, 3), (101, 3), (0, 2), (7, 7), (5, 3)):
            allocation = _split_integer(amount, parts)
            assert allocation == legacy_split(amount, parts)
            assert sum(allocation) == amount


# ── max-envelope dedup ───────────────────────────────────────────────────────

def _colliding_items(**scope_over):
    """Two catalog items sharing action/trade/unit-policy whose conditions
    land in the same physical unit -> one dedup collision."""
    first = _v31_item(
        "probe_alpha",
        work_item_code="SHARED_FIX",
        cost=_manual_cost(1000, 3000, 100, 200),
        **scope_over.get("alpha", {}),
    )
    second = _v31_item(
        "probe_beta",
        work_item_code="SHARED_FIX",
        cost=_manual_cost(800, 4000, 100, 200),
        **scope_over.get("beta", {}),
    )
    return first, second


class TestDedup:
    SPECS = [
        {"item": "probe_alpha", "unit": "kitchen_primary"},
        {"item": "probe_beta", "unit": "kitchen_primary"},
    ]

    def test_collision_takes_the_max_envelope_never_the_sum(self, tmp_path):
        result, _, _ = _derive(tmp_path, self.SPECS, *_colliding_items())
        (active,) = _active(result)
        # max(1000, 800) / max(3000, 4000) — NOT 1800/7000.
        assert (active["low"], active["high"]) == (1000, 4000)
        assert active["catalog_item_ids"] == ["probe_alpha", "probe_beta"]
        assert len(active["condition_ids"]) == 2
        suppressed = _suppressed(result)
        assert len(suppressed) == 2
        assert all(w["reason_code"] == DEDUP_SUPPRESSION_REASON for w in suppressed)
        # Suppressed audit records keep their original dollars.
        assert sorted((w["low"], w["high"]) for w in suppressed) == [
            (800, 4000), (1000, 3000),
        ]
        (collision,) = result["work_dedup_collisions"]
        assert collision["active_work_item_id"] == active["work_item_id"]
        assert sorted(collision["suppressed_work_item_ids"]) == sorted(
            w["work_item_id"] for w in suppressed
        )
        assert collision["policy_version"] == WORK_DEDUP_POLICY_VERSION
        # Totals count the merged active exactly once.
        assert result["standalone_estimate"]["headline"] == {
            "low": 1000, "high": 4000,
        }

    def test_collision_lands_in_the_most_required_scope_lane(self, tmp_path):
        items = _colliding_items(alpha={
            "estimate_scope": "required_rehab",
            "estimate_scope_reason": "safety_item",
        })
        result, _, _ = _derive(tmp_path, self.SPECS, *items)
        (active,) = _active(result)
        assert active["estimate_scope"] == "required_rehab"
        assert active["estimate_scope_reason"] == "safety_item"
        buckets = result["standalone_estimate"]["totals_by_estimate_scope"]
        assert buckets["required_rehab"] == {"low": 1000, "high": 4000}
        assert buckets["marketability_rehab"] == {"low": 0, "high": 0}

    def test_no_collision_across_distinct_physical_units(self, tmp_path):
        specs = [
            {"item": "probe_alpha", "unit": "kitchen_primary"},
            {"item": "probe_beta", "unit": "bathroom_1"},
        ]
        result, _, _ = _derive(tmp_path, specs, *_colliding_items())
        assert len(_active(result)) == 2
        assert result["work_dedup_collisions"] == []
        assert _suppressed(result) == []

    def test_merged_ambiguity_is_any_of_sources(self, tmp_path):
        specs = [
            {"item": "probe_alpha", "unit": "kitchen_primary", "ambiguous": True},
            {"item": "probe_beta", "unit": "kitchen_primary"},
        ]
        result, _, _ = _derive(tmp_path, specs, *_colliding_items())
        (active,) = _active(result)
        assert active["identity_ambiguous"] is True


# ── lanes and totals ─────────────────────────────────────────────────────────

class TestLanes:
    def test_non_accepted_dispositions_create_no_work(self, tmp_path):
        """excluded / inspection / withheld / no_action conditions survive in
        the Session 2 lanes untouched and contribute zero work dollars."""
        specs = [
            {"item": "accepted_probe", "unit": "room_a"},
            {"item": "excluded_probe", "unit": "room_b", "disposition": "excluded"},
            {"item": "inspect_probe", "unit": "room_c", "disposition": "inspection"},
            {"item": "withheld_probe", "unit": "room_d", "disposition": "withheld"},
            {"item": "noaction_probe", "unit": "room_e", "disposition": "no_action"},
        ]
        items = [
            _v31_item("accepted_probe", cost=_manual_cost(1000, 3000, 100, 200)),
            _v31_item("excluded_probe", drop_if_generic=True),
            _v31_item(
                "inspect_probe",
                estimate={"estimate_tier": "minor", "strategy": "inspect_only"},
            ),
            _v31_item("withheld_probe", cost=_manual_cost(500, 900, 100, 200)),
            _v31_item("noaction_probe", work_item_code=..., cost=...),
        ]
        result, review, _ = _derive(tmp_path, specs, *items)
        (item,) = _active(result)
        assert item["catalog_item_ids"] == ["accepted_probe"]
        assert result["standalone_estimate"]["headline"] == {
            "low": 1000, "high": 3000,
        }
        # The non-accepted dispositions are still visible, verbatim.
        preserved = {
            d["disposition"] for d in result["condition_dispositions"]
        }
        assert preserved == {
            "accepted_for_work", "excluded", "inspection", "withheld", "no_action",
        }

    def test_empty_accepted_set_yields_zero_buckets(self, tmp_path):
        specs = [
            {"item": "excluded_probe", "unit": "room_b", "disposition": "excluded"},
        ]
        result, _, _ = _derive(
            tmp_path, specs, _v31_item("excluded_probe", drop_if_generic=True)
        )
        assert result["work_items"] == []
        assert result["work_dedup_collisions"] == []
        standalone = result["standalone_estimate"]
        assert standalone["headline"] == {"low": 0, "high": 0}
        assert standalone["property_cost_factor"] == 1.0

    def test_totals_reconcile_exactly_across_scopes(self, tmp_path):
        specs = [
            {"item": "required_probe", "unit": "room_a"},
            {"item": "market_probe", "unit": "room_b"},
        ]
        result, _, _ = _derive(
            tmp_path,
            specs,
            _v31_item(
                "required_probe",
                cost=_manual_cost(2000, 5000, 100, 200),
                estimate_scope="required_rehab",
                estimate_scope_reason="safety_item",
            ),
            _v31_item("market_probe", cost=_manual_cost(300, 800, 100, 200)),
        )
        buckets = result["standalone_estimate"]["totals_by_estimate_scope"]
        assert buckets["required_rehab"] == {"low": 2000, "high": 5000}
        assert buckets["marketability_rehab"] == {"low": 300, "high": 800}
        assert buckets["optional_value_add"] == {"low": 0, "high": 0}
        assert buckets["inspection_risk"] == {"low": 0, "high": 0}
        assert result["standalone_estimate"]["headline"] == {
            "low": 2300, "high": 5800,
        }


# ── failure and hygiene ──────────────────────────────────────────────────────

class TestFailures:
    def test_invalid_review_result_is_a_dependency_failure(self, tmp_path):
        projection = _build(_v31_catalog(_v31_item("probe")), tmp_path)
        with pytest.raises(PassExecutionError) as excinfo:
            derive_standalone_estimate(
                review_result={"observed_conditions": []},
                projection=projection,
                property_metadata=None,
                estimate_id=EST_ID,
            )
        assert excinfo.value.code == "ReviewResultInvalid"
        assert excinfo.value.stage == "dependency"

    def test_accepted_condition_without_work_policy_fails(self, tmp_path):
        """An accepted condition citing a catalog item absent from the
        projection is an operational contradiction, not a pricing gap."""
        projection = _build(_v31_catalog(_v31_item("known_probe")), tmp_path)
        review = _review_for([{"item": "ghost_item", "unit": "room_a"}])
        with pytest.raises(PassExecutionError) as excinfo:
            derive_standalone_estimate(
                review_result=review,
                projection=projection,
                property_metadata=None,
                estimate_id=EST_ID,
            )
        assert excinfo.value.code == "WorkPolicyMissing"

    def test_input_review_result_is_never_mutated(self, tmp_path):
        projection = _build(
            _v31_catalog(
                _v31_item("probe", cost=_manual_cost(1000, 3000, 100, 200))
            ),
            tmp_path,
        )
        review = _review_for([{"item": "probe", "unit": "room_a"}])
        snapshot = copy.deepcopy(review)
        derive_standalone_estimate(
            review_result=review,
            projection=projection,
            property_metadata=None,
            estimate_id=EST_ID,
        )
        assert review == snapshot

    def test_derivation_is_byte_deterministic(self, tmp_path):
        specs = [
            {"item": "probe_alpha", "unit": "kitchen_primary"},
            {"item": "probe_beta", "unit": "kitchen_primary"},
        ]
        first, _, _ = _derive(tmp_path, specs, *_colliding_items())
        second, _, _ = _derive(tmp_path, specs, *_colliding_items())
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# ── standalone-result validator invariants ───────────────────────────────────

@pytest.fixture
def collided(tmp_path):
    """A validated result containing a collision plus a plain active item."""
    specs = [
        {"item": "probe_alpha", "unit": "kitchen_primary"},
        {"item": "probe_beta", "unit": "kitchen_primary"},
        {"item": "solo_probe", "unit": "room_z",
         "hints": ["window_id:w1", "window_id:w2"]},
    ]
    result, _, _ = _derive(
        tmp_path,
        specs,
        *_colliding_items(),
        _v31_item(
            "solo_probe",
            cost=_manual_cost(400, 900, 200, 500),
            estimate={"estimate_tier": "minor", "unit_policy": "per_opening"},
        ),
    )
    return result


def _revalidate(result):
    return validate_standalone_estimate_result(result, estimate_id=EST_ID)


class TestStandaloneValidatorInvariants:
    def test_the_derived_result_validates(self, collided):
        assert _revalidate(collided).ok

    def test_active_range_must_be_the_exact_max_envelope(self, collided):
        mutated = copy.deepcopy(collided)
        (active,) = [
            w for w in mutated["work_items"]
            if w["status"] == "active" and len(w["catalog_item_ids"]) == 2
        ]
        active["low"] += 1
        _assert_error(_revalidate(mutated), "max envelope")

    def test_suppressed_item_requires_a_collision_audit(self, collided):
        mutated = copy.deepcopy(collided)
        mutated["work_dedup_collisions"] = []
        _assert_error(_revalidate(mutated), "no collision audit")

    def test_active_work_groups_must_be_unique(self, collided):
        mutated = copy.deepcopy(collided)
        for item in mutated["work_items"]:
            if item["status"] == "suppressed":
                item["status"] = "active"
                item["reason_code"] = None
        _assert_error(_revalidate(mutated), "active groups must be unique")

    def test_accepted_scope_cannot_vanish(self, collided):
        mutated = copy.deepcopy(collided)
        mutated["work_items"] = [
            w for w in mutated["work_items"]
            if w["catalog_item_ids"] != ["solo_probe"]
        ]
        _assert_error(_revalidate(mutated), "accepted scope cannot vanish")

    def test_accepted_condition_in_exactly_one_active_item(self, collided):
        mutated = copy.deepcopy(collided)
        (solo,) = [
            w for w in mutated["work_items"]
            if w["catalog_item_ids"] == ["solo_probe"]
        ]
        clone = copy.deepcopy(solo)
        clone["work_item_id"] = "wk1_" + "e" * 16
        clone["action_code"] = "OTHER_FIX"
        mutated["work_items"].append(clone)
        _assert_error(_revalidate(mutated), "multiple active work items")

    def test_non_accepted_conditions_create_no_work(self, tmp_path):
        specs = [
            {"item": "accepted_probe", "unit": "room_a"},
            {"item": "inspect_probe", "unit": "room_b", "disposition": "inspection"},
        ]
        result, _, _ = _derive(
            tmp_path,
            specs,
            _v31_item("accepted_probe", cost=_manual_cost(1000, 3000, 100, 200)),
            _v31_item(
                "inspect_probe",
                estimate={"estimate_tier": "minor", "strategy": "inspect_only"},
            ),
        )
        mutated = copy.deepcopy(result)
        inspection_condition = next(
            c for c in mutated["observed_conditions"]
            if c["catalog_item_id"] == "inspect_probe"
        )
        (active,) = [w for w in mutated["work_items"] if w["status"] == "active"]
        active["condition_ids"] = sorted(
            active["condition_ids"] + [inspection_condition["condition_id"]]
        )
        _assert_error(_revalidate(mutated), "non-accepted conditions create no work")

    def test_lineage_must_be_the_exact_condition_sets(self, collided):
        mutated = copy.deepcopy(collided)
        (solo,) = [
            w for w in mutated["work_items"]
            if w["catalog_item_ids"] == ["solo_probe"]
        ]
        solo["catalog_item_ids"] = ["other_item", "solo_probe"]
        _assert_error(_revalidate(mutated), "catalog_item_ids must be exactly")

    def test_billable_unit_must_match_the_source_unit(self, collided):
        mutated = copy.deepcopy(collided)
        (solo,) = [
            w for w in mutated["work_items"]
            if w["catalog_item_ids"] == ["solo_probe"]
        ]
        solo["billable_unit_id"] = "room_other"
        _assert_error(_revalidate(mutated), "must bill exactly")

    def test_per_opening_unit_count_is_recomputed(self, collided):
        mutated = copy.deepcopy(collided)
        (solo,) = [
            w for w in mutated["work_items"]
            if w["catalog_item_ids"] == ["solo_probe"]
        ]
        solo["unit_count"] = 5
        _assert_error(_revalidate(mutated), "unit_count must be 2")

    def test_scope_totals_must_reconcile(self, collided):
        mutated = copy.deepcopy(collided)
        buckets = mutated["standalone_estimate"]["totals_by_estimate_scope"]
        buckets["marketability_rehab"]["high"] += 1
        _assert_error(_revalidate(mutated), "must be exactly")

    def test_headline_must_be_the_bucket_sum(self, collided):
        mutated = copy.deepcopy(collided)
        mutated["standalone_estimate"]["headline"]["low"] += 1
        _assert_error(_revalidate(mutated), "headline must be exactly")

    def test_collision_key_fields_must_match_the_active(self, collided):
        mutated = copy.deepcopy(collided)
        mutated["work_dedup_collisions"][0]["trade_bucket"] = "hvac"
        _assert_error(_revalidate(mutated), "does not match")
