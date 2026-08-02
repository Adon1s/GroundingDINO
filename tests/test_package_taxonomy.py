"""Tests covering the package taxonomy rework.

Spine: Observation -> IssueCandidate / Issue -> RenovationPackage -> ProjectScope.

Covers:
  - package_category / room / package_level / package_strength / confidence_score emission
  - package_level (the new structural axis) vs pricing_tier (the renamed depth field)
  - ACTIVE_PACKAGE_STATUSES gating across both 'confirmed' and 'confirmed_by_rule'
  - Turnover Pass 2f short-circuit -> confirmed_by_rule, never confirmed
  - Whole-home turnover aggregator threshold (>=2 distinct rooms)
  - Catalog consistency: every kitchen package-eligible entry carries package_category + room
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tools.rehab_packages import (
    ACTIVE_PACKAGE_STATUSES,
    PACKAGE_CATEGORY_MODERNIZATION,
    PACKAGE_CATEGORY_REPAIR,
    PACKAGE_CATEGORY_TURNOVER,
    PACKAGE_LEVEL_PROPERTY,
    PACKAGE_LEVEL_ROOM,
    PACKAGE_STRENGTH_MODERATE,
    PACKAGE_STRENGTH_STRONG,
    PACKAGE_STRENGTH_WEAK,
    PACKAGE_TYPE_KITCHEN_MODERNIZATION,
    PACKAGE_TYPE_KITCHEN_REPAIR,
    PACKAGE_TYPE_KITCHEN_TURNOVER,
    PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH,
    PACKAGE_VERIFICATION_CONFIRMED,
    PACKAGE_VERIFICATION_CONFIRMED_BY_RULE,
    PACKAGE_VERIFICATION_NOT_RUN,
    PACKAGE_VERIFICATION_REJECTED,
    PACKAGE_VERIFICATION_UNCERTAIN,
    ROOM_KITCHEN,
    ROOM_WHOLE_HOME,
    VALID_EMITTED_PACKAGE_STRENGTHS,
    VALID_PACKAGE_CATEGORIES,
    VALID_PACKAGE_TYPES,
    VALID_ROOMS,
    _apply_verification_to_package,
    aggregate_whole_home_turnover,
    build_package_affinity,
    catalog_package_category,
    catalog_package_type,
    catalog_room,
    compute_initial_confidence_score,
    compute_package_strength,
    compute_post_verification_score,
)
from tools.renovation_estimate import CatalogEstimateMeta, EstimateCandidate
from tools.estimate_scope import (
    INSPECTION_RISK,
    MARKETABILITY_REHAB,
    OPTIONAL_VALUE_ADD,
    REQUIRED_REHAB,
    classify_package_scope,
)
import tools.rehab_packages as rp


# ─── Builders ────────────────────────────────────────────────────────────────

def _candidate(*, catalog_item_id, severity=2, trade_bucket="kitchen_cabinets_counters",
               issue_ids=None, room_surrogate_id="kitchen_1"):
    meta = CatalogEstimateMeta(
        estimate_tier="medium",
        strategy="replace_only",
        group="kitchen",
        stack_behavior="sum",
        unit_policy="per_kitchen",
        affects_estimate=True,
        requires_2f_for_estimate=False,
    )
    c = EstimateCandidate(
        catalog_item_id=catalog_item_id,
        catalog_item_name=catalog_item_id,
        estimate_meta=meta,
        kind="defect",
        severity=severity,
        scope="replace",
        trade_bucket=trade_bucket,
    )
    c.room_surrogate_id = room_surrogate_id
    c.issue_ids = list(issue_ids or [f"{catalog_item_id}__{room_surrogate_id}"])
    c.effective_posture = "replace"
    return c


def _stub_turnover_package(room: str, package_id: str, *, cost_low: int = 1000,
                            cost_high: int = 3000, strength: str = PACKAGE_STRENGTH_MODERATE,
                            confidence: float = 0.4) -> Dict[str, Any]:
    return {
        "package_id": package_id,
        "package_type": PACKAGE_TYPE_KITCHEN_TURNOVER if room == ROOM_KITCHEN else f"{room}_turnover",
        "package_category": PACKAGE_CATEGORY_TURNOVER,
        "room": room,
        "package_level": PACKAGE_LEVEL_ROOM,
        "package_strength": strength,
        "confidence_score": confidence,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "supporting_issue_ids": [f"{package_id}__issue_1"],
        "supporting_catalog_item_ids": [],
        "verification_status": PACKAGE_VERIFICATION_CONFIRMED_BY_RULE,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Strength classification
# ═══════════════════════════════════════════════════════════════════════════

class TestPackageStrength:

    def test_strong_sentinel_alone_is_strong(self):
        # missing_base_cabinets_exposed_subfloor is a strong-signal sentinel.
        drivers = [_candidate(catalog_item_id="missing_base_cabinets_exposed_subfloor")]
        assert compute_package_strength(drivers, []) == PACKAGE_STRENGTH_STRONG

    def test_two_distinct_drivers_is_strong(self):
        drivers = [
            _candidate(catalog_item_id="outdated_kitchen_finishes"),
            _candidate(catalog_item_id="outdated_or_damaged_cabinets"),
        ]
        assert compute_package_strength(drivers, []) == PACKAGE_STRENGTH_STRONG

    def test_driver_plus_two_supports_is_strong(self):
        drivers = [_candidate(catalog_item_id="outdated_kitchen_finishes")]
        supports = [
            _candidate(catalog_item_id="countertop_damage"),
            _candidate(catalog_item_id="appliance_damage_or_missing"),
        ]
        assert compute_package_strength(drivers, supports) == PACKAGE_STRENGTH_STRONG

    def test_driver_plus_one_support_is_moderate(self):
        drivers = [_candidate(catalog_item_id="outdated_kitchen_finishes")]
        supports = [_candidate(catalog_item_id="countertop_damage")]
        assert compute_package_strength(drivers, supports) == PACKAGE_STRENGTH_MODERATE

    def test_single_driver_alone_is_moderate(self):
        # A catalog-tagged driver alone is enough for a moderate package.
        drivers = [_candidate(catalog_item_id="outdated_kitchen_finishes")]
        assert compute_package_strength(drivers, []) == PACKAGE_STRENGTH_MODERATE

    def test_two_supports_only_is_moderate(self):
        supports = [
            _candidate(catalog_item_id="countertop_damage"),
            _candidate(catalog_item_id="appliance_damage_or_missing"),
        ]
        assert compute_package_strength([], supports) == PACKAGE_STRENGTH_MODERATE

    def test_orphan_support_is_weak(self):
        supports = [_candidate(catalog_item_id="countertop_damage")]
        assert compute_package_strength([], supports) == PACKAGE_STRENGTH_WEAK

    def test_emitted_set_excludes_weak(self):
        assert PACKAGE_STRENGTH_WEAK not in VALID_EMITTED_PACKAGE_STRENGTHS
        assert PACKAGE_STRENGTH_STRONG in VALID_EMITTED_PACKAGE_STRENGTHS
        assert PACKAGE_STRENGTH_MODERATE in VALID_EMITTED_PACKAGE_STRENGTHS


# ═══════════════════════════════════════════════════════════════════════════
# Confidence score
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceScore:

    def test_initial_prior_strong(self):
        assert compute_initial_confidence_score(PACKAGE_STRENGTH_STRONG) == pytest.approx(0.30)

    def test_initial_prior_moderate(self):
        assert compute_initial_confidence_score(PACKAGE_STRENGTH_MODERATE) == pytest.approx(0.15)

    def test_initial_prior_weak_is_zero(self):
        assert compute_initial_confidence_score(PACKAGE_STRENGTH_WEAK) == 0.0

    def test_full_confirmation_boosts_score(self):
        score = compute_post_verification_score(
            initial_score=0.30,
            confirmed_count=3,
            rejected_count=0,
            supporting_count=3,
        )
        # 0.30 + 0.50 * (3/3) - 0 = 0.80
        assert score == pytest.approx(0.80)

    def test_full_rejection_penalizes_score(self):
        score = compute_post_verification_score(
            initial_score=0.30,
            confirmed_count=0,
            rejected_count=3,
            supporting_count=3,
        )
        # 0.30 + 0 - 0.20 * (3/3) = 0.10
        assert score == pytest.approx(0.10)

    def test_score_clamped_to_unit_interval(self):
        assert compute_post_verification_score(0.95, 10, 0, 1) == 1.0
        assert compute_post_verification_score(-1.0, 0, 10, 1) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Active package status helper
# ═══════════════════════════════════════════════════════════════════════════

class TestActivePackageStatuses:

    def test_confirmed_is_active(self):
        assert PACKAGE_VERIFICATION_CONFIRMED in ACTIVE_PACKAGE_STATUSES

    def test_confirmed_by_rule_is_active(self):
        assert PACKAGE_VERIFICATION_CONFIRMED_BY_RULE in ACTIVE_PACKAGE_STATUSES

    def test_other_statuses_not_active(self):
        assert PACKAGE_VERIFICATION_REJECTED not in ACTIVE_PACKAGE_STATUSES
        assert PACKAGE_VERIFICATION_UNCERTAIN not in ACTIVE_PACKAGE_STATUSES
        assert PACKAGE_VERIFICATION_NOT_RUN not in ACTIVE_PACKAGE_STATUSES

    def test_apply_verification_marks_both_active_states_eligible(self):
        base = {
            "package_id": "p1",
            "supporting_issue_ids": ["i1"],
            "confidence_score": 0.30,
        }
        for status in (PACKAGE_VERIFICATION_CONFIRMED, PACKAGE_VERIFICATION_CONFIRMED_BY_RULE):
            result = _apply_verification_to_package(
                dict(base),
                {"verification_status": status, "confirmed_issue_ids": ["i1"]},
            )
            assert result["estimate_eligible"] is True, status
            assert result["ui_eligible"] is True, status
            assert result["audit_only"] is False, status

    def test_apply_verification_rejected_is_audit_only(self):
        result = _apply_verification_to_package(
            {"package_id": "p1", "supporting_issue_ids": ["i1"]},
            {"verification_status": PACKAGE_VERIFICATION_REJECTED},
        )
        assert result["estimate_eligible"] is False
        assert result["ui_eligible"] is False
        assert result["audit_only"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Whole-home turnover aggregator
# ═══════════════════════════════════════════════════════════════════════════

class TestWholeHomeTurnoverAggregator:

    def test_single_room_does_not_aggregate(self):
        packages = [_stub_turnover_package(ROOM_KITCHEN, "kt_1", cost_low=1000, cost_high=3000)]
        assert aggregate_whole_home_turnover(packages) is None

    def test_two_rooms_aggregate_sum_and_property_level(self):
        packages = [
            _stub_turnover_package(ROOM_KITCHEN, "kt_1", cost_low=1000, cost_high=3000),
            _stub_turnover_package("bathroom", "bt_1", cost_low=500, cost_high=2000),
        ]
        agg = aggregate_whole_home_turnover(packages)
        assert agg is not None
        assert agg["package_level"] == PACKAGE_LEVEL_PROPERTY
        assert agg["room"] == ROOM_WHOLE_HOME
        assert agg["package_type"] == PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH
        assert agg["package_category"] == PACKAGE_CATEGORY_TURNOVER
        assert agg["cost_low"] == 1500
        assert agg["cost_high"] == 5000
        assert agg["verification_status"] == PACKAGE_VERIFICATION_CONFIRMED_BY_RULE

    def test_aggregate_takes_strongest_strength(self):
        packages = [
            _stub_turnover_package(ROOM_KITCHEN, "kt_1", strength=PACKAGE_STRENGTH_MODERATE),
            _stub_turnover_package("bathroom", "bt_1", strength=PACKAGE_STRENGTH_STRONG),
        ]
        agg = aggregate_whole_home_turnover(packages)
        assert agg["package_strength"] == PACKAGE_STRENGTH_STRONG

    def test_aggregate_ignores_non_turnover_packages(self):
        packages = [
            _stub_turnover_package(ROOM_KITCHEN, "kt_1"),
            {
                "package_id": "mod_1",
                "package_category": PACKAGE_CATEGORY_MODERNIZATION,
                "room": ROOM_KITCHEN,
                "package_level": PACKAGE_LEVEL_ROOM,
                "cost_low": 5000,
                "cost_high": 10000,
                "supporting_issue_ids": [],
                "supporting_catalog_item_ids": [],
                "package_strength": PACKAGE_STRENGTH_STRONG,
                "confidence_score": 0.6,
            },
        ]
        # Only one room contributes turnover; aggregate should be None.
        assert aggregate_whole_home_turnover(packages) is None

    def test_aggregate_skips_already_aggregated(self):
        # An aggregate masquerading as a property-level turnover must not be re-aggregated.
        existing_aggregate = _stub_turnover_package(ROOM_WHOLE_HOME, "agg_1")
        existing_aggregate["package_level"] = PACKAGE_LEVEL_PROPERTY
        per_room = _stub_turnover_package(ROOM_KITCHEN, "kt_1")
        assert aggregate_whole_home_turnover([existing_aggregate, per_room]) is None


# ═══════════════════════════════════════════════════════════════════════════
# Category-aware scope classification
# ═══════════════════════════════════════════════════════════════════════════

class TestCategoryAwareScope:

    def test_repair_category_maps_to_required(self):
        scope, reason = classify_package_scope(
            "kitchen_repair_light", [], "",
            package_category=PACKAGE_CATEGORY_REPAIR,
            package_strength=PACKAGE_STRENGTH_STRONG,
        )
        assert scope == REQUIRED_REHAB
        assert "repair" in reason

    def test_modernization_strong_maps_to_marketability(self):
        scope, reason = classify_package_scope(
            "kitchen_partial_rehab", [], "",
            package_category=PACKAGE_CATEGORY_MODERNIZATION,
            package_strength=PACKAGE_STRENGTH_STRONG,
        )
        assert scope == MARKETABILITY_REHAB

    def test_modernization_moderate_maps_to_optional(self):
        scope, reason = classify_package_scope(
            "kitchen_refresh", [], "",
            package_category=PACKAGE_CATEGORY_MODERNIZATION,
            package_strength=PACKAGE_STRENGTH_MODERATE,
        )
        assert scope == OPTIONAL_VALUE_ADD

    def test_turnover_category_maps_to_marketability(self):
        scope, _ = classify_package_scope(
            "kitchen_turnover_light", [], "",
            package_category=PACKAGE_CATEGORY_TURNOVER,
            package_strength=PACKAGE_STRENGTH_MODERATE,
        )
        assert scope == MARKETABILITY_REHAB


# ═══════════════════════════════════════════════════════════════════════════
# Catalog consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestCatalogConsistency:

    @pytest.fixture(scope="class")
    def catalog(self) -> Dict[str, Any]:
        catalog_path = Path(__file__).resolve().parent.parent / "tools" / "issue_catalog.json"
        with open(catalog_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def test_every_package_affinity_entry_has_category_and_room(self, catalog):
        # Routing is catalog-driven (per-item package_affinity blocks); the
        # loader derives category/room from package_type, so every flattened
        # entry must land in the valid vocabularies.
        table = build_package_affinity(catalog)
        assert table, "shipped catalog must route at least one item"
        offenders = []
        for (room, issue_id), meta in table.items():
            if catalog_package_category(meta) not in VALID_PACKAGE_CATEGORIES:
                offenders.append(((room, issue_id), "missing/invalid category"))
            if catalog_room(meta) not in VALID_ROOMS:
                offenders.append(((room, issue_id), "missing/invalid room"))
        assert not offenders, f"Catalog tagging holes: {offenders}"

    def test_kitchen_modernization_items_resolve_to_modernization_category(self, catalog):
        table = build_package_affinity(catalog)
        kitchen_modernization = {
            key: meta for key, meta in table.items()
            if catalog_package_type(meta) == PACKAGE_TYPE_KITCHEN_MODERNIZATION
        }
        assert kitchen_modernization
        for key, meta in kitchen_modernization.items():
            assert catalog_package_category(meta) == PACKAGE_CATEGORY_MODERNIZATION, key
            assert catalog_room(meta) == ROOM_KITCHEN, key


# ═══════════════════════════════════════════════════════════════════════════
# Pricing tier rename — make sure historical name is gone in fresh output
# ═══════════════════════════════════════════════════════════════════════════

class TestPricingTierRename:

    def test_build_package_candidate_emits_pricing_tier_not_package_level_for_depth(self):
        # The new package_level holds "room"/"property" etc., not "refresh"/"partial_rehab".
        # The depth field is renamed to pricing_tier.
        from tools.rehab_packages import _build_package_candidate

        drivers = [_candidate(catalog_item_id="outdated_kitchen_finishes")]
        supports = [_candidate(catalog_item_id="countertop_damage")]
        pkg = _build_package_candidate(
            package_type=PACKAGE_TYPE_KITCHEN_MODERNIZATION,
            unit_id="kitchen_primary",
            room_surrogate_id="kitchen_1",
            source_room_surrogate_ids=["kitchen_1"],
            supporting_candidates=drivers + supports,
            drivers=drivers,
            supports=supports,
            catalog_lookup={
                "outdated_kitchen_finishes": {
                    "id": "outdated_kitchen_finishes",
                    "package_role": "package_driver",
                    "package_type": PACKAGE_TYPE_KITCHEN_MODERNIZATION,
                    "package_category": PACKAGE_CATEGORY_MODERNIZATION,
                    "room": ROOM_KITCHEN,
                },
                "countertop_damage": {
                    "id": "countertop_damage",
                    "package_role": "package_support",
                    "package_type": PACKAGE_TYPE_KITCHEN_MODERNIZATION,
                    "package_category": PACKAGE_CATEGORY_MODERNIZATION,
                    "room": ROOM_KITCHEN,
                },
            },
            trigger_reason="package_driver",
        )
        # pricing_tier carries the depth value ("refresh"/"partial_rehab"/etc.)
        assert pkg["pricing_tier"] in {"refresh", "partial_rehab", "full_rehab"}
        # package_level carries the new structural axis (always "room" here).
        assert pkg["package_level"] == PACKAGE_LEVEL_ROOM
        # Other new fields must be present.
        assert pkg["package_category"] == PACKAGE_CATEGORY_MODERNIZATION
        assert pkg["room"] == ROOM_KITCHEN
        assert pkg["package_strength"] in VALID_EMITTED_PACKAGE_STRENGTHS
        assert 0.0 <= pkg["confidence_score"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Bedroom / living taxonomy wiring (full parity: modernization/repair/turnover)
# ═══════════════════════════════════════════════════════════════════════════

class TestBedroomLivingTaxonomy:

    _NEW_TYPES = (
        "bedroom_modernization", "bedroom_repair", "bedroom_turnover",
        "living_modernization", "living_repair", "living_turnover",
    )

    def test_new_types_are_valid_and_mapped(self):
        for t in self._NEW_TYPES:
            assert t in VALID_PACKAGE_TYPES, t
            assert t in rp._PACKAGE_TYPE_TO_CATEGORY, t
            assert t in rp._PACKAGE_TYPE_TO_ROOM, t
        assert rp._PACKAGE_TYPE_TO_ROOM["bedroom_modernization"] == rp.ROOM_BEDROOM
        assert rp._PACKAGE_TYPE_TO_ROOM["living_turnover"] == rp.ROOM_LIVING
        assert rp._PACKAGE_TYPE_TO_CATEGORY["bedroom_repair"] == PACKAGE_CATEGORY_REPAIR
        assert rp._PACKAGE_TYPE_TO_CATEGORY["living_turnover"] == PACKAGE_CATEGORY_TURNOVER

    def test_absorption_scopes_present_for_all_new_tiers(self):
        tiers = [
            rp.BEDROOM_REFRESH, rp.BEDROOM_FULL_REHAB, rp.BEDROOM_REPAIR_LIGHT,
            rp.BEDROOM_REPAIR_HEAVY, rp.BEDROOM_TURNOVER_LIGHT, rp.BEDROOM_TURNOVER_STD,
            rp.LIVING_REFRESH, rp.LIVING_FULL_REHAB, rp.LIVING_REPAIR_LIGHT,
            rp.LIVING_REPAIR_HEAVY, rp.LIVING_TURNOVER_LIGHT, rp.LIVING_TURNOVER_STD,
        ]
        for spec in tiers:
            scope = rp._PACKAGE_ABSORPTION_SCOPES.get(spec[0])
            assert scope is not None, spec[0]
            # groups must equal the room family so reconciliation joins by estimate_group.
            assert scope["family"] in {"bedroom", "living"}
            assert scope["family"] in scope["groups"]

    def test_dispatcher_routes_each_new_type_to_its_room_tier(self):
        cases = {
            "bedroom_modernization": "bedroom_refresh",
            "bedroom_repair": "bedroom_repair_light",
            "bedroom_turnover": "bedroom_turnover_light",
            "living_modernization": "living_refresh",
            "living_repair": "living_repair_light",
            "living_turnover": "living_turnover_light",
        }
        for package_type, expected_tier in cases.items():
            spec, label, _notes = rp._resolve_pricing_profile(package_type, [], [], [])
            assert spec[0] == expected_tier, (package_type, spec[0])

    def test_bedroom_and_living_turnover_roll_up_to_whole_home(self):
        packages = [
            _stub_turnover_package("bedroom", "br_1", cost_low=800, cost_high=2500),
            _stub_turnover_package("living", "lv_1", cost_low=1200, cost_high=4000),
        ]
        agg = aggregate_whole_home_turnover(packages)
        assert agg is not None
        assert agg["room"] == ROOM_WHOLE_HOME
        assert agg["package_level"] == PACKAGE_LEVEL_PROPERTY
        assert agg["cost_low"] == 2000
        assert agg["cost_high"] == 6500
