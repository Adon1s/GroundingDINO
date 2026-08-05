"""Catalog regression tests for exterior estimate coverage.

Locks the population decision in place: which exterior items carry a
line-item estimate, which are deliberately omitted and why, and which of the
populated ones are guarded behind Pass 2f confirmation.

Runs against the real catalog, like tests/test_project_scopes.py.

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_exterior_estimate_coverage.py -q`
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.renovation_estimate import (
    GROUP_BUDGET_CAPS,
    VALID_UNIT_POLICIES,
    resolve_catalog_estimate_meta,
)

CATALOG_PATH = Path(__file__).resolve().parent.parent / "tools" / "issue_catalog.json"

# The nine items populated behind the guard, with their intended routing.
GUARDED_EXTERIOR_ESTIMATES = {
    "standing_water_or_poor_grading": ("high", "repair_or_replace", "landscaping", "group_cap", "per_area"),
    "driveway_or_walkway_cracking": ("medium", "repair_or_replace", "landscaping", "group_cap", "per_area"),
    "clogged_or_damaged_gutters": ("medium", "service_only", "exterior", "group_cap", "per_property"),
    "trees_or_vegetation_too_close": ("medium", "service_only", "landscaping", "group_cap", "per_property"),
    "shed_exterior_paint_failure": ("medium", "repair_only", "exterior", "group_cap", "per_property"),
    "metal_carport_rust_corrosion": ("medium", "repair_only", "exterior", "group_cap", "per_property"),
    "uneven_gravel_drive_or_parking": ("medium", "repair_only", "landscaping", "group_cap", "per_area"),
    "tree_stump_present_in_yard": ("medium", "service_only", "landscaping", "group_cap", "per_property"),
    "exterior_door_paint_failure": ("medium", "repair_only", "windows_doors", "group_cap", "per_opening"),
}

# Populated but deliberately NOT guarded: inspect-only items already price at
# the flat INSPECT_ALLOWANCE rather than making a repair claim, so guarding
# would suppress a latent-risk signal precisely because it is unverified.
UNGUARDED_EXTERIOR_ESTIMATES = {
    "roofline_water_damage_suspected": ("high", "inspect_only", "roof", "max_only", "per_property"),
}

NEWLY_POPULATED = {**GUARDED_EXTERIOR_ESTIMATES, **UNGUARDED_EXTERIOR_ESTIMATES}

# Omitted because Pass 2e suppresses tier:optional upstream, so an estimate
# block would never fire.
OPTIONAL_OMISSIONS = {
    "landscape_improvement_needed",
    "curb_appeal_upgrade",
    "yard_debris_overgrown_leaves",
    "gutter_maintenance_needed",
    "concrete_driveway_surface_wear",
}

# Omitted because a package already prices them; a line item would double-count.
PACKAGE_AFFINITY_OMISSIONS = {
    "damaged_or_unsafe_deck_or_porch",
    "exterior_siding_discoloration_fading",
    "deck_surface_weathering",
    "brick_weathering_or_mortar_deterioration",
    "patio_or_porch_surface_wear",
}


@pytest.fixture(scope="module")
def catalog() -> dict:
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def exterior_items(catalog) -> dict:
    return {
        item["id"]: item
        for item in catalog["items"]
        if item.get("category") == "exterior"
    }


class TestCoveragePredicate:
    def test_exterior_population_is_fully_partitioned(self, exterior_items):
        """25 exterior items: 15 priced, 10 omitted for two stated reasons."""
        assert len(exterior_items) == 25

        covered, missing = set(), set()
        for item_id, item in exterior_items.items():
            target = covered if resolve_catalog_estimate_meta(item).affects_estimate else missing
            target.add(item_id)

        assert len(covered) == 15
        assert missing == OPTIONAL_OMISSIONS | PACKAGE_AFFINITY_OMISSIONS
        assert NEWLY_POPULATED.keys() <= covered

    def test_omission_reasons_are_disjoint_and_accurate(self, exterior_items):
        assert not (OPTIONAL_OMISSIONS & PACKAGE_AFFINITY_OMISSIONS)
        for item_id in OPTIONAL_OMISSIONS:
            assert exterior_items[item_id].get("tier") == "optional", item_id
        for item_id in PACKAGE_AFFINITY_OMISSIONS:
            assert exterior_items[item_id].get("package_affinity"), item_id


class TestPopulatedEstimates:
    @pytest.mark.parametrize("item_id", sorted(NEWLY_POPULATED))
    def test_routing_matches_intent(self, exterior_items, item_id):
        tier, strategy, group, stack, unit = NEWLY_POPULATED[item_id]
        estimate = exterior_items[item_id]["estimate"]
        assert estimate["estimate_tier"] == tier
        assert estimate["strategy"] == strategy
        assert estimate["group"] == group
        assert estimate["stack_behavior"] == stack
        assert estimate["unit_policy"] == unit

    @pytest.mark.parametrize("item_id", sorted(NEWLY_POPULATED))
    def test_guard_flag_is_explicit(self, exterior_items, item_id):
        """Never rely on the default. The guard is opt-in, so an unset flag
        silently prices the item — say it out loud in the catalog."""
        estimate = exterior_items[item_id]["estimate"]
        assert "requires_2f_for_estimate" in estimate, item_id
        expected = item_id in GUARDED_EXTERIOR_ESTIMATES
        assert estimate["requires_2f_for_estimate"] is expected

    @pytest.mark.parametrize("item_id", sorted(NEWLY_POPULATED))
    def test_routing_values_are_valid(self, exterior_items, item_id):
        estimate = exterior_items[item_id]["estimate"]
        # group is validated against the budget-cap table, not the Literal
        assert estimate["group"] in GROUP_BUDGET_CAPS
        assert estimate["unit_policy"] in VALID_UNIT_POLICIES
        assert estimate["stack_behavior"] in ("sum", "group_cap", "max_only")
        # high/medium tiers require a cost block (catalog_validation.py)
        assert exterior_items[item_id].get("cost"), item_id

    @pytest.mark.parametrize("item_id", sorted(NEWLY_POPULATED))
    def test_no_package_affinity(self, exterior_items, item_id):
        """These price as standalone line items; package affinity would route
        them into a bundle and double-count."""
        assert not exterior_items[item_id].get("package_affinity"), item_id

    def test_gutters_avoid_the_roof_group(self, exterior_items):
        """_resolve_dominant_stack promotes a whole group to max_only if any
        member has it, and damaged_or_aged_roof_shingles is max_only. Gutters
        in `roof` would price $0 on any property with shingle damage.
        """
        assert exterior_items["clogged_or_damaged_gutters"]["estimate"]["group"] == "exterior"
        assert exterior_items["damaged_or_aged_roof_shingles"]["estimate"]["stack_behavior"] == "max_only"
        # The trade lane is what drives project scope, and it stays roof/gutters.
        assert exterior_items["clogged_or_damaged_gutters"]["trade_bucket"] == "roof_gutters"


class TestFence:
    def test_fence_has_cost_but_no_estimate(self, catalog):
        """A cost block alone is inert — it makes the item eligible for a
        future high/medium estimate without pricing anything today."""
        fence = next(
            item for item in catalog["items"]
            if item["id"] == "fence_damaged_or_weathered"
        )
        assert fence["cost"] == {"mode": "heuristic"}
        assert "estimate" not in fence
        assert "category" not in fence
        assert resolve_catalog_estimate_meta(fence).affects_estimate is False
