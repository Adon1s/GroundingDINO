"""Gate tests for tools/catalog_validation.py.

``test_shipped_catalog_has_no_errors`` is the enforcement point for every
catalog invariant: the shipped issue_catalog.json must validate clean on
every pytest run. The warnings snapshot makes new advisory findings surface
in review instead of accumulating silently — update it deliberately when
warnings legitimately change.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.catalog_validation import (
    CatalogValidationResult,
    validate_issue_catalog,
)


def _load_shipped_catalog():
    return json.loads(Path("tools/issue_catalog.json").read_text(encoding="utf-8"))


def _valid_item(**overrides):
    item = {
        "id": "synthetic_item",
        "name": "Synthetic Item",
        "category": "cosmetic",
        "severity": 2,
        "trade_bucket": "flooring",
        "description": "Synthetic description.",
        "embed_text": "Synthetic embed text.",
        "kind": "defect",
        "scope": "repair",
        "tier": "work",
        "defaultHidden": False,
        "scene_groups": ["kitchen"],
        "display_class": "marketability",
    }
    item.update(overrides)
    return item


def _catalog(*items):
    return {
        "version": "synthetic",
        "trade_buckets": [{"id": "flooring"}],
        "items": list(items),
    }


def _errors_for(*items) -> list:
    return validate_issue_catalog(_catalog(*items)).errors


# ─── The gate ────────────────────────────────────────────────────────────────

def test_shipped_catalog_has_no_errors():
    result = validate_issue_catalog(_load_shipped_catalog())
    assert result.errors == []


# Items knowingly missing display_class (advisory: they fall back to keyword
# sniffing in catalog_display_class). Shrink this list as items get classified;
# any NEW entry here should be a deliberate decision, not drift.
_MISSING_DISPLAY_CLASS_IDS = [
    "brick_weathering_or_mortar_deterioration",
    "ceiling_fan_mismatched_blades",
    "clogged_or_damaged_gutters",
    "concrete_driveway_surface_wear",
    "curb_appeal_upgrade",
    "damaged_or_aged_roof_shingles",
    "damaged_or_rotted_siding_or_trim",
    "damaged_or_unsafe_deck_or_porch",
    "damaged_soffit_or_porch_ceiling",
    "dated_exterior_finishes",
    "deck_surface_weathering",
    "driveway_or_walkway_cracking",
    "empty_or_deteriorated_inground_pool",
    "exposed_sheathing_or_missing_siding",
    "exterior_door_paint_failure",
    "exterior_siding_discoloration_fading",
    "fence_damaged_or_weathered",
    "garage_or_basement_damage",
    "gutter_maintenance_needed",
    "landscape_improvement_needed",
    "major_foundation_or_settlement_signs",
    "metal_carport_rust_corrosion",
    "older_ceiling_fan_style",
    "patio_or_porch_surface_wear",
    "retaining_wall_failure_or_missing_section",
    "roofline_water_damage_suspected",
    "shed_exterior_paint_failure",
    "standing_water_or_poor_grading",
    "tree_stump_present_in_yard",
    "trees_or_vegetation_too_close",
    "uneven_gravel_drive_or_parking",
    "unfinished_basement_present",
    "yard_debris_overgrown_leaves",
]


def test_shipped_catalog_warnings_snapshot():
    result = validate_issue_catalog(_load_shipped_catalog())
    expected = sorted(
        [
            f"{item_id}: missing display_class "
            "(falls back to keyword sniffing in catalog_display_class)"
            for item_id in _MISSING_DISPLAY_CLASS_IDS
        ]
        + ["<catalog>: trade bucket 'hvac' declared but used by no item"]
    )
    assert sorted(result.warnings) == expected


# ─── Sanity ──────────────────────────────────────────────────────────────────

def test_minimal_valid_item_has_no_errors():
    assert _errors_for(_valid_item()) == []


def test_result_ok_property():
    assert CatalogValidationResult().ok
    assert not CatalogValidationResult(errors=["x: boom"]).ok


# ─── Rule 1: id ──────────────────────────────────────────────────────────────

def test_missing_id():
    item = _valid_item()
    del item["id"]
    errors = _errors_for(item)
    assert any("id missing" in e for e in errors)


def test_id_bad_pattern():
    errors = _errors_for(_valid_item(id="Bad-Id"))
    assert any(e.startswith("Bad-Id:") and "^[a-z0-9_]+$" in e for e in errors)


def test_duplicate_id():
    errors = _errors_for(_valid_item(), _valid_item())
    assert any(e.startswith("synthetic_item:") and "duplicate id" in e for e in errors)


# ─── Rule 2: core enums ──────────────────────────────────────────────────────

def test_bad_kind():
    errors = _errors_for(_valid_item(kind="fixture"))
    assert any(e.startswith("synthetic_item:") and "kind" in e for e in errors)


def test_bad_scope():
    errors = _errors_for(_valid_item(scope="demolish"))
    assert any(e.startswith("synthetic_item:") and "scope" in e for e in errors)


def test_bad_item_tier():
    errors = _errors_for(_valid_item(tier="luxury"))
    assert any(e.startswith("synthetic_item:") and "tier" in e for e in errors)


@pytest.mark.parametrize("severity", [0, 6, "3", 2.5, True, None])
def test_bad_severity(severity):
    errors = _errors_for(_valid_item(severity=severity))
    assert any(e.startswith("synthetic_item:") and "severity" in e for e in errors)


def test_severity_5_is_legal():
    assert _errors_for(_valid_item(severity=5)) == []


# ─── Rule 3: trade_bucket ────────────────────────────────────────────────────

def test_undeclared_trade_bucket():
    errors = _errors_for(_valid_item(trade_bucket="masonry"))
    assert any(
        e.startswith("synthetic_item:") and "trade_bucket" in e for e in errors
    )


def test_missing_trade_buckets_section_is_one_catalog_error():
    catalog = _catalog(_valid_item())
    catalog["trade_buckets"] = []
    errors = validate_issue_catalog(catalog).errors
    assert errors == ["<catalog>: trade_buckets section missing or empty"]


# ─── Rule 4: scene_groups ────────────────────────────────────────────────────

def test_scene_groups_living_is_rejected():
    # The classic mistake: "living" instead of "living_areas" silently drops
    # the item from retrieval.
    errors = _errors_for(_valid_item(scene_groups=["living"]))
    assert any(
        e.startswith("synthetic_item:") and "'living'" in e for e in errors
    )


def test_scene_groups_pool_is_accepted():
    assert _errors_for(_valid_item(scene_groups=["exterior", "pool"])) == []


@pytest.mark.parametrize("scene_groups", [[], None, "kitchen"])
def test_scene_groups_must_be_non_empty_list(scene_groups):
    errors = _errors_for(_valid_item(scene_groups=scene_groups))
    assert any(
        e.startswith("synthetic_item:") and "scene_groups" in e for e in errors
    )


# ─── Rule 5: estimate block ──────────────────────────────────────────────────

def _estimate(**overrides):
    block = {
        "estimate_tier": "medium",
        "strategy": "repair_only",
        "group": "flooring",
        "stack_behavior": "group_cap",
        "unit_policy": "per_room",
    }
    block.update(overrides)
    return block


def _cost(**overrides):
    block = {
        "mode": "allowance",
        "base_low": 100,
        "base_high": 1000,
        "per_occurrence_low": 90,
        "per_occurrence_high": 300,
        "cap_low": 900,
        "cap_high": 2000,
        "cost_source": "manual",
    }
    block.update(overrides)
    return block


def test_estimate_tier_low_is_rejected():
    # The rule that would have caught the shipped "low" bug.
    errors = _errors_for(
        _valid_item(estimate=_estimate(estimate_tier="low"), cost=_cost())
    )
    assert any(
        e.startswith("synthetic_item:") and "estimate_tier 'low'" in e
        for e in errors
    )


def test_estimate_tier_required_in_authored_block():
    block = _estimate()
    del block["estimate_tier"]
    errors = _errors_for(_valid_item(estimate=block, cost=_cost()))
    assert any(
        e.startswith("synthetic_item:") and "estimate_tier" in e for e in errors
    )


def test_bad_estimate_strategy():
    errors = _errors_for(
        _valid_item(estimate=_estimate(strategy="demolish_only"), cost=_cost())
    )
    assert any(e.startswith("synthetic_item:") and "strategy" in e for e in errors)


def test_bad_stack_behavior():
    errors = _errors_for(
        _valid_item(estimate=_estimate(stack_behavior="average"), cost=_cost())
    )
    assert any(
        e.startswith("synthetic_item:") and "stack_behavior" in e for e in errors
    )


def test_bad_unit_policy():
    errors = _errors_for(
        _valid_item(estimate=_estimate(unit_policy="per_wall"), cost=_cost())
    )
    assert any(
        e.startswith("synthetic_item:") and "unit_policy" in e for e in errors
    )


def test_estimate_group_paint_is_rejected():
    # The rule that would have caught the shipped "paint" bug.
    errors = _errors_for(
        _valid_item(estimate=_estimate(group="paint"), cost=_cost())
    )
    assert any(
        e.startswith("synthetic_item:") and "group 'paint'" in e for e in errors
    )


# ─── Rules 6-7: cost block ───────────────────────────────────────────────────

def test_bad_cost_mode():
    errors = _errors_for(_valid_item(cost=_cost(mode="exact")))
    assert any(e.startswith("synthetic_item:") and "mode" in e for e in errors)


def test_heuristic_cost_with_only_mode_is_valid():
    assert _errors_for(_valid_item(cost={"mode": "heuristic"})) == []


def test_negative_cost_amount():
    errors = _errors_for(_valid_item(cost=_cost(base_low=-5)))
    assert any(e.startswith("synthetic_item:") and "base_low" in e for e in errors)


def test_non_numeric_cost_amount():
    errors = _errors_for(_valid_item(cost=_cost(cap_high="2000")))
    assert any(e.startswith("synthetic_item:") and "cap_high" in e for e in errors)


@pytest.mark.parametrize("overrides", [
    {"base_low": 2000, "base_high": 1000},
    {"cap_low": 5000, "cap_high": 2000},
    {"per_occurrence_low": 500, "per_occurrence_high": 300},
    {"base_high": 9000, "cap_high": 2000},
])
def test_cost_pair_ordering(overrides):
    errors = _errors_for(_valid_item(cost=_cost(**overrides)))
    assert any(e.startswith("synthetic_item:") and ">" in e for e in errors)


@pytest.mark.parametrize("tier", ["high", "medium"])
def test_priced_tier_requires_cost_block(tier):
    errors = _errors_for(_valid_item(estimate=_estimate(estimate_tier=tier)))
    assert any(
        e.startswith("synthetic_item:") and "requires a cost block" in e
        for e in errors
    )


def test_minor_tier_does_not_require_cost_block():
    assert _errors_for(_valid_item(estimate=_estimate(estimate_tier="minor"))) == []


# ─── Rule 8: static classifications ──────────────────────────────────────────

def test_bad_estimate_scope():
    errors = _errors_for(_valid_item(estimate_scope="cosmetic_rehab"))
    assert any(
        e.startswith("synthetic_item:") and "estimate_scope" in e for e in errors
    )


def test_bad_display_class():
    errors = _errors_for(_valid_item(display_class="banner"))
    assert any(
        e.startswith("synthetic_item:") and "display_class" in e for e in errors
    )


# ─── Rule 9: cost_model ──────────────────────────────────────────────────────

@pytest.mark.parametrize("cost_model", ["package_allowance", "inspection_allowance"])
def test_derived_cost_models_must_not_be_authored(cost_model):
    errors = _errors_for(_valid_item(cost_model=cost_model))
    assert any(
        e.startswith("synthetic_item:") and "cost_model" in e for e in errors
    )


def test_derived_cost_model_in_estimate_block_rejected():
    errors = _errors_for(
        _valid_item(
            estimate=_estimate(cost_model="package_allowance"), cost=_cost()
        )
    )
    assert any(
        e.startswith("synthetic_item:") and "estimate.cost_model" in e
        for e in errors
    )


def test_authorable_cost_models_accepted():
    assert _errors_for(_valid_item(cost_model="room_allowance")) == []
    assert _errors_for(_valid_item(cost_model="line_item")) == []


# ─── Rule 10: package_affinity (delegated to build_package_affinity) ─────────

def test_package_affinity_wrong_room_for_package_type():
    errors = _errors_for(_valid_item(package_affinity={
        "bathroom": {
            "package_type": "kitchen_modernization",
            "package_role": "package_driver",
        },
    }))
    assert any("synthetic_item" in e and "kitchen_modernization" in e
               for e in errors)


def test_package_affinity_bad_role():
    errors = _errors_for(_valid_item(package_affinity={
        "kitchen": {
            "package_type": "kitchen_modernization",
            "package_role": "driver",
        },
    }))
    assert any("synthetic_item" in e and "package_role" in e for e in errors)


def test_valid_package_affinity_accepted():
    assert _errors_for(_valid_item(package_affinity={
        "kitchen": {
            "package_type": "kitchen_modernization",
            "package_role": "package_support",
        },
    })) == []


# ─── Rule 11: flat routing fields are forbidden ──────────────────────────────

@pytest.mark.parametrize("field_name", ["package_type", "package_category", "room"])
def test_flat_routing_fields_forbidden(field_name):
    errors = _errors_for(_valid_item(**{field_name: "kitchen"}))
    assert any(
        e.startswith("synthetic_item:") and field_name in e and "forbidden" in e
        for e in errors
    )


def test_flat_package_role_driver_rejected():
    errors = _errors_for(_valid_item(package_role="package_driver"))
    assert any(
        e.startswith("synthetic_item:") and "package_role" in e for e in errors
    )


@pytest.mark.parametrize("role", ["standalone", "ignore"])
def test_flat_package_role_standalone_and_ignore_accepted(role):
    assert _errors_for(_valid_item(package_role=role)) == []


# ─── Rule 12: field types ────────────────────────────────────────────────────

@pytest.mark.parametrize("field_name", ["defaultHidden", "drop_if_generic"])
def test_bool_fields(field_name):
    errors = _errors_for(_valid_item(**{field_name: "yes"}))
    assert any(
        e.startswith("synthetic_item:") and field_name in e for e in errors
    )


@pytest.mark.parametrize("field_name", ["deny_any", "support_any", "require_any"])
def test_list_fields(field_name):
    errors = _errors_for(_valid_item(**{field_name: "carpet"}))
    assert any(
        e.startswith("synthetic_item:") and field_name in e for e in errors
    )


@pytest.mark.parametrize("field_name", ["deny_any", "support_any", "require_any"])
@pytest.mark.parametrize("term", ["mo$ld", "$mold", "$"])
def test_whole_word_marker_only_allowed_as_a_suffix(field_name, term):
    """"$" opts a term in to a trailing word boundary; anywhere but the end it
    is an authoring typo that would silently never match."""
    errors = _errors_for(_valid_item(**{field_name: [term]}))
    assert any(
        e.startswith("synthetic_item:") and "whole-word marker" in e for e in errors
    ), errors


@pytest.mark.parametrize("field_name", ["deny_any", "support_any", "require_any"])
def test_whole_word_marker_suffix_is_valid(field_name):
    assert _errors_for(_valid_item(**{field_name: ["mold$", "mildew"]})) == []


# ─── Warnings ────────────────────────────────────────────────────────────────

def test_warning_driver_without_cost():
    item = _valid_item(package_affinity={
        "kitchen": {
            "package_type": "kitchen_modernization",
            "package_role": "package_driver",
        },
    })
    result = validate_issue_catalog(_catalog(item))
    assert result.errors == []
    assert any(
        w.startswith("synthetic_item:") and "driver without a cost block" in w
        for w in result.warnings
    )


def test_no_driver_warning_when_cost_present():
    item = _valid_item(
        cost=_cost(),
        package_affinity={
            "kitchen": {
                "package_type": "kitchen_modernization",
                "package_role": "package_driver",
            },
        },
    )
    result = validate_issue_catalog(_catalog(item))
    assert not any("driver without a cost block" in w for w in result.warnings)


def test_warning_missing_display_class():
    item = _valid_item()
    del item["display_class"]
    result = validate_issue_catalog(_catalog(item))
    assert result.errors == []
    assert any(
        w.startswith("synthetic_item:") and "display_class" in w
        for w in result.warnings
    )


def test_warning_unused_trade_bucket():
    catalog = _catalog(_valid_item())
    catalog["trade_buckets"].append({"id": "hvac"})
    result = validate_issue_catalog(catalog)
    assert result.errors == []
    assert any(
        w.startswith("<catalog>:") and "'hvac'" in w for w in result.warnings
    )


# ─── absence must never resolve onto a gutter item ───────────────────────────
# absent / damaged / maintenance are three different claims with three different
# cost implications, and "not visible in this photo" is not "not present on the
# building". The deny lists on the two gutter items are what enforces that; this
# is a guardrail-level test, so it needs no embeddings server.
# See docs/HANDOFF_pass2c_exterior_recall.md.

GUTTER_ITEM_IDS = ("clogged_or_damaged_gutters", "gutter_maintenance_needed")

# Verbatim from the production corpus: absence claims that reached a gutter item
# before the deny terms landed.
ABSENCE_OBSERVATIONS = (
    "No visible gutters or downspouts direct water away from the foundation.",
    "There is no visible gutter system despite the roof overhang.",
    "The roofline has no visible gutter system on the front face of the house.",
    "No downspout extension is visible for rainwater drainage.",
    "Missing gutters and downspouts pose a risk for water damage.",
    "Roofline appears uneven with no visible gutters.",
)

# Also verbatim: real damage and maintenance claims that must keep resolving.
LEGITIMATE_OBSERVATIONS = (
    "A downspout is disconnected or poorly routed near the porch and steps.",
    "Rusted downspout and gutter system with visible rust streaks running down the side of the house.",
    "Gutters may be clogged or detached from the structure.",
    "Downspouts discharge near the foundation and stoop area.",
    "The downspout ends too close to the foundation, discharging directly onto soil.",
    "Gutters and downspouts are visible but show signs of lacking clear maintenance.",
)


def _denied_by(item_id: str, text: str) -> bool:
    from tools.catalog_embeddings import build_guardrails_from_catalog
    from tools.pipeline_common import term_matches

    guardrails = build_guardrails_from_catalog(_load_shipped_catalog())
    deny = (guardrails.get(item_id) or {}).get("deny_any", [])
    return any(term_matches(term, text.lower()) for term in deny)


@pytest.mark.parametrize("item_id", GUTTER_ITEM_IDS)
@pytest.mark.parametrize("description", ABSENCE_OBSERVATIONS)
def test_absence_claim_is_denied_by_both_gutter_items(item_id, description):
    assert _denied_by(item_id, description), (
        f"{description!r} would resolve onto {item_id}"
    )


@pytest.mark.parametrize("item_id", GUTTER_ITEM_IDS)
@pytest.mark.parametrize("description", LEGITIMATE_OBSERVATIONS)
def test_visible_gutter_condition_is_not_denied(item_id, description):
    if item_id == "gutter_maintenance_needed" and any(
        t in description.lower() for t in ("disconnected", "sagging")
    ):
        pytest.skip("pre-existing damage-routing deny, not an absence deny")
    assert not _denied_by(item_id, description), (
        f"{description!r} is a visible condition and must stay resolvable"
    )


def test_gutter_deny_lists_stay_narrow():
    """A bare 'missing' or 'appears limited' on the damage item would swallow
    mixed claims like 'gutters appear clogged or missing downspouts'."""
    catalog = _load_shipped_catalog()
    item = next(
        i for i in catalog["items"] if i.get("id") == "clogged_or_damaged_gutters"
    )

    assert "missing" not in item["deny_any"]
    assert "appears limited" not in item["deny_any"]
