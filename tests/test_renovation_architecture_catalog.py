"""Catalog projection tests: strict v3.1 preconditions, terminal-route
precedence, and the shipped-catalog pins.

The builder enforces structural completeness; the exact route distribution
(12/4/5/9/98 over 128 items) is pinned HERE, not in the builder, so a future
catalog regeneration updates these expectations instead of breaking
shadow-mode worker startup (same philosophy as the byte-parity pin in
tests/test_catalog_kind_v2.py).

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_catalog.py -q
"""
import json
from pathlib import Path
from typing import get_args

import pytest

from tools.catalog_validation import load_shipped_catalog_v2
from tools.comparison_common import sha256_file
from tools.renovation_architecture.catalog_projection import (
    RenovationCatalogError,
    build_renovation_catalog_projection,
    resolve_terminal_route,
)
from tools.renovation_architecture.contracts import (
    STRATEGIES,
    TERMINAL_ROUTES,
    UNIT_POLICIES,
)

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_V2_PATH = ROOT / "tools" / "issue_catalog_kind_v2.json"

EXPECTED_ROUTE_COUNTS = {
    "excluded_quarantine": 12,
    "excluded_generic": 4,
    "inspection": 5,
    "no_action": 9,
    "work": 98,
}
# The user-approved no-economics gaps (inferred no_action).
EXPECTED_NO_ECONOMICS_GAP_IDS = {
    "dirty_or_grimy_window_screens",
    "dated_or_older_windows",
    "door_hardware_worn",
    "door_hardware_dated_style",
}
# The Session 8 opportunity/presence triage: otherwise-billable items routed
# out of billing by an explicit catalog route_override.
EXPECTED_ROUTE_OVERRIDE_IDS = {
    "unfinished_basement_present",
    "staging_or_decluttering_opportunity",
    "mismatched_or_inconsistent_furniture_staging",
    "curb_appeal_upgrade",
    "landscaping_enhancement_opportunity",
}
EXPECTED_NO_ACTION_IDS = EXPECTED_NO_ECONOMICS_GAP_IDS | EXPECTED_ROUTE_OVERRIDE_IDS
EXPECTED_INSPECTION_IDS = {
    "major_foundation_or_settlement_signs",
    "roofline_water_damage_suspected",
    "retaining_wall_failure_or_missing_section",
    "inground_pool_empty_or_unserviceable",
    "inground_pool_finishes_deteriorated",
}
EXPECTED_COST_WITHOUT_CODE_IDS = {
    "soffit_or_porch_ceiling_failed",
    "soffit_or_porch_ceiling_weathered",
    "fence_broken_or_leaning",
    "fence_weathered",
    "bare_or_missing_finish_flooring",
}
EXPECTED_MIN_PHOTO_EVIDENCE = {
    "roof_shingles_missing_or_curling": 2,
    "roof_shingles_aged_or_worn": 2,
    "paving_heaved_or_trip_hazard": 2,
    "paving_cracked_or_settled": 2,
    "clogged_or_damaged_gutters": 2,
    "roofline_water_damage_suspected": 2,
}


# ── synthetic catalog builders ───────────────────────────────────────────────

def _v31_item(item_id, **over):
    base = {
        "id": item_id,
        "name": item_id.replace("_", " ").title(),
        "kind": "degradation",
        "severity": 2,
        "trade_bucket": "flooring",
        "scope": "repair",
        "tier": "work",
        "defaultHidden": False,
        "description": "synthetic item",
        "embed_text": "synthetic item",
        "scene_groups": ["kitchen"],
        "atomic_claim": {
            "subject": "surface",
            "state": "worn",
            "ontology_basis": "visible_deterioration",
        },
        "drop_if_generic": False,
        "work_item_code": "FLOORING_REPAIR",
        "cost": {"mode": "heuristic"},
    }
    base.update(over)
    return {key: value for key, value in base.items() if value is not ...}


def _v31_catalog(*items, **root_over):
    catalog = {
        "version": "3.2",
        "ontology_version": "observation-kind-v2",
        "publication_status": "publishable",
        "trade_buckets": [
            {"id": "flooring", "name": "Flooring"},
            {"id": "electrical", "name": "Electrical", "product_quarantined": True},
        ],
        "items": list(items) or [_v31_item("synthetic_default")],
    }
    catalog.update(root_over)
    return catalog


def _build(catalog, tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return build_renovation_catalog_projection(catalog, catalog_path=path)


def _build_error(catalog, tmp_path):
    with pytest.raises(RenovationCatalogError) as excinfo:
        _build(catalog, tmp_path)
    return excinfo.value


# ── build preconditions ──────────────────────────────────────────────────────

class TestBuildPreconditions:
    def test_happy_synthetic_catalog_builds(self, tmp_path):
        projection = _build(_v31_catalog(), tmp_path)
        assert projection["version"] == "renovation_catalog_projection_v3"
        assert projection["route_counts"]["work"] == 1

    def test_wrong_version_fails(self, tmp_path):
        error = _build_error(_v31_catalog(version="3.0"), tmp_path)
        assert any("version" in message for message in error.errors)

    def test_underscore_ontology_spelling_fails(self, tmp_path):
        """The catalog stamp is hyphenated; the underscore string is the env
        selector value and must never be accepted here."""
        error = _build_error(
            _v31_catalog(ontology_version="observation_kind_v2"), tmp_path
        )
        assert any("observation-kind-v2" in message for message in error.errors)

    def test_non_publishable_fails(self, tmp_path):
        error = _build_error(
            _v31_catalog(publication_status="blocked_pending_pricing"), tmp_path
        )
        assert any("publication_status" in message for message in error.errors)

    def test_catalog_validation_error_fails(self, tmp_path):
        error = _build_error(
            _v31_catalog(_v31_item("bad_severity", severity=9)), tmp_path
        )
        assert any("severity" in message for message in error.errors)

    def test_malformed_affinity_fails(self, tmp_path):
        error = _build_error(
            _v31_catalog(
                _v31_item(
                    "bad_affinity",
                    package_affinity={"bathroom": {"package_type": "bogus_package", "package_role": "package_support"}},
                )
            ),
            tmp_path,
        )
        assert any("package" in message for message in error.errors)

    def test_unknown_strategy_fails(self, tmp_path):
        error = _build_error(
            _v31_catalog(
                _v31_item(
                    "bad_strategy",
                    estimate={"estimate_tier": "minor", "strategy": "bulldoze"},
                )
            ),
            tmp_path,
        )
        assert any("unknown estimate.strategy" in message for message in error.errors)

    def test_absent_estimate_block_is_legal(self, tmp_path):
        projection = _build(_v31_catalog(_v31_item("no_estimate")), tmp_path)
        assert projection["work_policy"]["no_estimate"]["unit_policy"] == "per_scope"
        assert projection["work_policy"]["no_estimate"]["strategy"] is None

    def test_unknown_route_override_fails(self, tmp_path):
        error = _build_error(
            _v31_catalog(_v31_item("bad_override", route_override="maybe")),
            tmp_path,
        )
        assert any("route_override" in message for message in error.errors)

    def test_route_override_item_builds_without_work_policy(self, tmp_path):
        projection = _build(
            _v31_catalog(
                _v31_item("overridden", route_override="no_action"),
                _v31_item("kept"),
            ),
            tmp_path,
        )
        assert projection["route_counts"]["no_action"] == 1
        assert "overridden" not in projection["work_policy"]
        assert "kept" in projection["work_policy"]

    def test_all_errors_reported_together(self, tmp_path):
        error = _build_error(
            _v31_catalog(
                _v31_item("bad_severity", severity=9),
                version="3.0",
                publication_status="blocked_pending_pricing",
            ),
            tmp_path,
        )
        assert len(error.errors) >= 3


# ── terminal route precedence ────────────────────────────────────────────────

class TestRoutePrecedence:
    QUARANTINED = frozenset({"electrical"})

    def _route(self, **over):
        return resolve_terminal_route(
            _v31_item("probe", **over), quarantined_buckets=self.QUARANTINED
        )

    def test_quarantine_beats_generic(self):
        route, reason = self._route(trade_bucket="electrical", drop_if_generic=True)
        assert (route, reason) == ("excluded_quarantine", "product_quarantined_trade")

    def test_generic_beats_inspect_only(self):
        route, _ = self._route(
            drop_if_generic=True,
            estimate={"estimate_tier": "minor", "strategy": "inspect_only"},
        )
        assert route == "excluded_generic"

    def test_inspect_only_beats_missing_economics(self):
        route, reason = self._route(
            estimate={"estimate_tier": "minor", "strategy": "inspect_only"},
            cost=...,
            work_item_code=...,
        )
        assert (route, reason) == ("inspection", "strategy_inspect_only")

    def test_missing_economics_routes_no_action(self):
        route, reason = self._route(cost=..., work_item_code=...)
        assert (route, reason) == ("no_action", "no_economics_approved_gap")

    def test_route_override_forces_no_action_on_billable_item(self):
        route, reason = self._route(route_override="no_action")
        assert (route, reason) == ("no_action", "route_override_no_action")

    def test_route_override_beats_missing_economics_reason(self):
        """Precedence puts the explicit override ahead of the inferred gap so
        the reason code states intent (validation separately rejects this
        redundant authoring on the shipped catalogs)."""
        route, reason = self._route(
            route_override="no_action", cost=..., work_item_code=...
        )
        assert (route, reason) == ("no_action", "route_override_no_action")

    def test_inspect_only_beats_route_override(self):
        route, reason = self._route(
            route_override="no_action",
            estimate={"estimate_tier": "minor", "strategy": "inspect_only"},
        )
        assert (route, reason) == ("inspection", "strategy_inspect_only")

    def test_generic_beats_route_override(self):
        route, _ = self._route(route_override="no_action", drop_if_generic=True)
        assert route == "excluded_generic"

    def test_quarantine_beats_route_override(self):
        route, _ = self._route(route_override="no_action", trade_bucket="electrical")
        assert route == "excluded_quarantine"

    def test_absent_drop_if_generic_means_false(self):
        item = _v31_item("probe")
        del item["drop_if_generic"]
        route, _ = resolve_terminal_route(item, quarantined_buckets=self.QUARANTINED)
        assert route == "work"

    def test_everything_else_is_work(self):
        assert self._route() == ("work", "work_default")

    def test_cost_without_code_projects_catalog_scope_action(self, tmp_path):
        projection = _build(
            _v31_catalog(
                _v31_item(
                    "cost_no_code",
                    work_item_code=...,
                    cost={"mode": "allowance", "base_low": 100, "base_high": 300},
                    scope="replace",
                )
            ),
            tmp_path,
        )
        policy = projection["work_policy"]["cost_no_code"]
        assert policy["action_source"] == "catalog_scope"
        assert policy["action_code"] == "replace"
        assert policy["pricing_mode"] == "catalog_allowance"

    def test_code_without_cost_projects_heuristic(self, tmp_path):
        projection = _build(_v31_catalog(_v31_item("code_no_cost", cost=...)), tmp_path)
        policy = projection["work_policy"]["code_no_cost"]
        assert policy["action_source"] == "work_item_code"
        assert policy["pricing_mode"] == "heuristic"
        assert policy["cost"] is None

    def test_catalog_estimate_scope_override_wins(self, tmp_path):
        """Projection v2: an explicit catalog estimate_scope beats the term
        classification, and the catalog's own reason survives."""
        projection = _build(
            _v31_catalog(
                _v31_item(
                    "value_add_probe",
                    estimate_scope="optional_value_add",
                    estimate_scope_reason="buyer_taste_upgrade",
                )
            ),
            tmp_path,
        )
        policy = projection["work_policy"]["value_add_probe"]
        assert policy["estimate_scope"] == "optional_value_add"
        assert policy["estimate_scope_reason"] == "buyer_taste_upgrade"

    def test_classified_scope_without_override(self, tmp_path):
        """Without an override, the existing estimate-scope policy classifies
        from catalog fields alone (degradation + worn -> marketability)."""
        projection = _build(_v31_catalog(_v31_item("classified_probe")), tmp_path)
        policy = projection["work_policy"]["classified_probe"]
        assert policy["estimate_scope"] == "marketability_rehab"

    def test_inspection_route_scope_metadata(self, tmp_path):
        projection = _build(
            _v31_catalog(
                _v31_item(
                    "inspect_probe",
                    estimate={"estimate_tier": "minor", "strategy": "inspect_only"},
                )
            ),
            tmp_path,
        )
        policy = projection["work_policy"]["inspect_probe"]
        assert policy["estimate_scope"] == "inspection_risk"
        assert policy["estimate_scope_reason"] == "terminal_route_inspection"


# ── shipped catalog pins ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def shipped_projection():
    catalog = load_shipped_catalog_v2()
    return build_renovation_catalog_projection(catalog, catalog_path=SHIPPED_V2_PATH)


class TestShippedCatalogPins:
    def test_route_distribution(self, shipped_projection):
        assert shipped_projection["route_counts"] == EXPECTED_ROUTE_COUNTS
        assert sum(shipped_projection["route_counts"].values()) == 128

    def test_every_item_has_exactly_one_terminal_route(self, shipped_projection):
        routes = shipped_projection["terminal_routes"]
        assert len(routes) == len(shipped_projection["observables"]) == 128
        assert all(entry["route"] in TERMINAL_ROUTES for entry in routes.values())

    def test_no_action_ids_and_reason_codes(self, shipped_projection):
        no_action = {
            item_id
            for item_id, entry in shipped_projection["terminal_routes"].items()
            if entry["route"] == "no_action"
        }
        assert no_action == EXPECTED_NO_ACTION_IDS
        for item_id in EXPECTED_NO_ECONOMICS_GAP_IDS:
            assert (
                shipped_projection["terminal_routes"][item_id]["reason_code"]
                == "no_economics_approved_gap"
            ), item_id
        for item_id in EXPECTED_ROUTE_OVERRIDE_IDS:
            assert (
                shipped_projection["terminal_routes"][item_id]["reason_code"]
                == "route_override_no_action"
            ), item_id

    def test_triage_siblings_still_bill(self, shipped_projection):
        """The route_override on landscaping_enhancement_opportunity is
        authored on the split successor only — its degradation sibling (and
        the explicitly kept paint item) must keep the work route."""
        for item_id in (
            "landscaping_overgrown_or_neglected", "paint_refresh_recommended"
        ):
            entry = shipped_projection["terminal_routes"][item_id]
            assert (entry["route"], entry["reason_code"]) == (
                "work", "work_default"
            ), item_id

    def test_route_override_items_have_no_work_policy(self, shipped_projection):
        for item_id in EXPECTED_ROUTE_OVERRIDE_IDS:
            assert item_id not in shipped_projection["work_policy"], item_id

    def test_inspection_ids(self, shipped_projection):
        inspection = {
            item_id
            for item_id, entry in shipped_projection["terminal_routes"].items()
            if entry["route"] == "inspection"
        }
        assert inspection == EXPECTED_INSPECTION_IDS

    def test_quarantine_wins_over_generic(self, shipped_projection):
        """dated_electrical_outlets_switches carries drop_if_generic AND sits
        in the quarantined electrical bucket — precedence is load-bearing."""
        entry = shipped_projection["terminal_routes"]["dated_electrical_outlets_switches"]
        assert entry["route"] == "excluded_quarantine"

    def test_worn_or_stained_flooring_is_heuristic_work(self, shipped_projection):
        assert (
            shipped_projection["terminal_routes"]["worn_or_stained_flooring"]["route"]
            == "work"
        )
        policy = shipped_projection["work_policy"]["worn_or_stained_flooring"]
        assert policy["action_code"] == "FLOORING_REPLACE"
        assert policy["action_source"] == "work_item_code"
        assert policy["pricing_mode"] == "heuristic"

    def test_cost_without_code_items_use_catalog_scope(self, shipped_projection):
        for item_id in EXPECTED_COST_WITHOUT_CODE_IDS:
            policy = shipped_projection["work_policy"][item_id]
            assert policy["action_source"] == "catalog_scope", item_id
            assert (
                policy["action_code"]
                == shipped_projection["observables"][item_id]["scope"]
            ), item_id

    def test_min_photo_evidence_projection(self, shipped_projection):
        gated = {
            item_id: entry["min_photo_evidence"]
            for item_id, entry in shipped_projection["observables"].items()
            if entry["min_photo_evidence"] is not None
        }
        assert gated == EXPECTED_MIN_PHOTO_EVIDENCE

    def test_estimate_scope_distribution(self, shipped_projection):
        """Projection v2 risk-lane metadata: 98 work-route items classify
        35/58/5 through the existing estimate-scope policy; the 5
        inspection-route items are the inspection_risk lane by routing."""
        scopes = {"required_rehab": 0, "marketability_rehab": 0,
                  "optional_value_add": 0, "inspection_risk": 0}
        for item_id, policy in shipped_projection["work_policy"].items():
            scopes[policy["estimate_scope"]] += 1
        assert scopes == {
            "required_rehab": 35,
            "marketability_rehab": 58,
            "optional_value_add": 5,
            "inspection_risk": 5,
        }

    def test_inspection_route_scope_is_a_routing_fact(self, shipped_projection):
        for item_id in EXPECTED_INSPECTION_IDS:
            policy = shipped_projection["work_policy"][item_id]
            assert policy["estimate_scope"] == "inspection_risk", item_id
            assert policy["estimate_scope_reason"] == "terminal_route_inspection", item_id

    def test_every_work_policy_entry_carries_scope_metadata(self, shipped_projection):
        for item_id, policy in shipped_projection["work_policy"].items():
            assert policy["estimate_scope"] in {
                "required_rehab", "marketability_rehab", "optional_value_add",
                "inspection_risk",
            }, item_id
            assert isinstance(policy["estimate_scope_reason"], str), item_id
            assert policy["estimate_scope_reason"], item_id

    def test_package_affinity_counts(self, shipped_projection):
        affinities = shipped_projection["package_policy"]["affinities"]
        assert len(affinities) == 70
        assert sum(len(rooms) for rooms in affinities.values()) == 117

    def test_flat_roles(self, shipped_projection):
        flat_roles = shipped_projection["package_policy"]["flat_roles"]
        assert len(flat_roles) == 47
        assert set(flat_roles.values()) <= {"standalone", "ignore"}

    def test_catalog_sha256_matches_file(self, shipped_projection):
        assert shipped_projection["catalog_sha256"] == sha256_file(SHIPPED_V2_PATH)

    def test_projection_is_json_serializable(self, shipped_projection):
        assert json.loads(json.dumps(shipped_projection))


class TestFingerprint:
    def test_fingerprint_is_deterministic(self, shipped_projection):
        rebuilt = build_renovation_catalog_projection(
            load_shipped_catalog_v2(), catalog_path=SHIPPED_V2_PATH
        )
        assert rebuilt["fingerprint"] == shipped_projection["fingerprint"]

    def test_fingerprint_changes_with_content(self, tmp_path):
        first = _build(_v31_catalog(_v31_item("probe", severity=2)), tmp_path)
        second = _build(_v31_catalog(_v31_item("probe", severity=3)), tmp_path)
        assert first["fingerprint"] != second["fingerprint"]


class TestVocabularyPins:
    def test_strategies_match_estimator_literal(self):
        from tools.renovation_estimate import EstimateStrategy

        assert STRATEGIES == frozenset(get_args(EstimateStrategy))

    def test_unit_policies_match_estimator(self):
        from tools.renovation_estimate import VALID_UNIT_POLICIES

        assert UNIT_POLICIES == frozenset(VALID_UNIT_POLICIES)
