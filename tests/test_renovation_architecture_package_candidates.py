"""Deterministic package-candidate tests (Session 4): ACTIVE-only membership,
merged lineage with driver precedence, scene-aware affinity, opportunity
corroboration over deduped representatives, strength/tier/floor economics,
and the display-only whole-home aggregate.

Candidates are built from REAL derive_standalone_estimate output over
synthetic v3.1 catalogs carrying package_affinity blocks, so every fixture
crosses the frozen Session 3 gate before it reaches the builder.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_package_candidates.py -q
"""
import copy
import json

import pytest

from tests.test_renovation_architecture_catalog import _v31_catalog, _v31_item
from tests.test_renovation_architecture_contracts import (
    EST_ID,
    FP,
    _condition,
    _lattice,
)
from tools.cost_factors import resolve_property_cost_factor
from tools.rehab_packages import _package_absorption_scope
from tools.renovation_architecture.catalog_projection import (
    build_renovation_catalog_projection,
)
from tools.renovation_architecture.contracts import CONTRACTS_SCHEMA_VERSION
from tools.renovation_architecture.ids import (
    make_package_candidate_id,
    make_terra_call_id,
)
from tools.renovation_architecture.package_candidates import (
    build_package_candidates,
)
from tools.renovation_architecture.validators import (
    validate_standalone_estimate_result,
)
from tools.renovation_architecture.work_items import derive_standalone_estimate
from tools.scene_classifier_passes import PassExecutionError

_TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
    "budget_debited_tokens",
)


def _pkg_item(item_id, room, package_type, role, **over):
    """A synthetic v3.1 item routed to a package via its affinity block."""
    scene_groups = over.pop(
        "scene_groups", ["living_areas" if room == "living" else room]
    )
    return _v31_item(
        item_id,
        scene_groups=scene_groups,
        package_affinity={
            room: {"package_type": package_type, "package_role": role}
        },
        **over,
    )


def _projection_and_catalog(tmp_path, *items):
    catalog = _v31_catalog(
        *items,
        trade_buckets=[
            {"id": "flooring", "name": "Flooring"},
            {"id": "paint_drywall", "name": "Paint"},
            {"id": "kitchen_cabinets_counters", "name": "Kitchen"},
        ],
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return build_renovation_catalog_projection(catalog, catalog_path=path), catalog


def _cond(catalog_item_id, unit_id, scene_group):
    condition = _condition(catalog_item_id, unit_id)
    condition["scene_group"] = scene_group
    return condition


def _with_photos(evidence, photo_keys, exact_groups=()):
    """Rewrite an evidence record for a photo set with optional exact-dup
    groups, keeping the recompute rules of the Session 2 gate satisfied."""
    keys = sorted(set(photo_keys))
    grouped = {key for group in exact_groups for key in group}
    classes = sorted(
        [sorted(group) for group in exact_groups]
        + [[key] for key in keys if key not in grouped]
    )
    evidence.update({
        "photo_keys": keys,
        "distinct_photo_count": len(keys),
        "distinct_view_count": len(classes),
        "duplicate_groups": [group for group in classes if len(group) > 1],
        "exact_duplicate_groups": [sorted(group) for group in exact_groups],
        "near_duplicate_groups": [],
        "representative_photo_keys": sorted(group[0] for group in classes),
        "evidence_refs": [
            {**dict(ref), "photo_key": keys[0]}
            for ref in evidence["evidence_refs"]
        ],
    })
    return evidence


def _review_result(lattices):
    """A valid Session 2 result from (condition, evidence, review,
    disposition) tuples: one Terra call per estimate unit."""
    conditions = [entry[0] for entry in lattices]
    by_unit = {}
    for condition in conditions:
        by_unit.setdefault(condition["estimate_unit_id"], []).append(condition)
    calls, units = [], []
    for unit_id in sorted(by_unit):
        call = {
            "call_id": make_terra_call_id(
                estimate_id=EST_ID, estimate_unit_id=unit_id,
                request_fingerprint=FP,
            ),
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "estimate_unit_id": unit_id,
            "condition_ids": sorted(
                c["condition_id"] for c in by_unit[unit_id]
            ),
            "request_fingerprint": FP,
            "provider": "openai",
            "model": "terra-test",
            "prompt_version": "v1",
            "usage_source": "provider",
            "input_tokens": 1000,
            "cached_input_tokens": 0,
            "output_tokens": 200,
            "total_tokens": 1200,
            "budget_debited_tokens": 1200,
        }
        calls.append(call)
        units.append({
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "estimate_unit_id": unit_id,
            "call_ids": [call["call_id"]],
            **{name: call[name] for name in _TOKEN_FIELDS},
        })
    listing = {
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "call_count": len(calls),
        **{
            name: sum(unit[name] for unit in units)
            for name in _TOKEN_FIELDS
        },
    }
    return {
        "observed_conditions": conditions,
        "evidence_facts": [entry[1] for entry in lattices],
        "condition_reviews": [entry[2] for entry in lattices],
        "condition_dispositions": [entry[3] for entry in lattices],
        "terra_calls": calls,
        "terra_unit_usage": units,
        "terra_listing_usage": listing,
    }


def _standalone(projection, lattices, property_metadata=None):
    result = derive_standalone_estimate(
        review_result=_review_result(lattices),
        projection=projection,
        property_metadata=property_metadata,
        estimate_id=EST_ID,
    )
    assert validate_standalone_estimate_result(result, estimate_id=EST_ID).ok
    return result


def _build(tmp_path, items, conditions_spec, property_metadata=None):
    """items -> projection/catalog; conditions_spec: (item_id, unit, scene,
    evidence_tweak|None)."""
    projection, catalog = _projection_and_catalog(tmp_path, *items)
    lattices = []
    for item_id, unit_id, scene, tweak in conditions_spec:
        condition = _cond(item_id, unit_id, scene)
        evidence, review, disposition = _lattice(condition)
        if tweak is not None:
            tweak(evidence)
        lattices.append((condition, evidence, review, disposition))
    standalone = _standalone(projection, lattices, property_metadata)
    candidates = build_package_candidates(
        standalone_result=standalone,
        projection=projection,
        catalog=catalog,
        estimate_id=EST_ID,
    )
    return standalone, candidates


def _work_by_condition(standalone):
    index = {}
    for item in standalone["work_items"]:
        if item["status"] != "active":
            continue
        for condition_id in item["condition_ids"]:
            index[condition_id] = item
    return index


# ── driver packages over accepted work ───────────────────────────────────────

class TestDriverPackages:
    def _kitchen(self, tmp_path, property_metadata=None):
        items = [
            _pkg_item(
                "worn_cabinets", "kitchen", "kitchen_modernization",
                "package_driver", trade_bucket="kitchen_cabinets_counters",
                work_item_code="CABINETS_REPLACE",
            ),
            _pkg_item(
                "worn_kitchen_paint", "kitchen", "kitchen_modernization",
                "package_support", trade_bucket="paint_drywall",
                work_item_code="PAINT_REFRESH",
            ),
        ]
        spec = [
            ("worn_cabinets", "kitchen_primary", "kitchen", None),
            ("worn_kitchen_paint", "kitchen_primary", "kitchen", None),
        ]
        return _build(tmp_path, items, spec, property_metadata)

    def test_driver_plus_support_emits_one_candidate(self, tmp_path):
        standalone, candidates = self._kitchen(tmp_path)
        (candidate,) = candidates
        by_condition = _work_by_condition(standalone)
        driver_work = next(
            item for cid, item in by_condition.items()
            if "worn_cabinets" in item["catalog_item_ids"]
        )
        support_work = next(
            item for cid, item in by_condition.items()
            if "worn_kitchen_paint" in item["catalog_item_ids"]
        )
        assert candidate["package_type"] == "kitchen_modernization"
        assert candidate["package_category"] == "modernization"
        assert candidate["package_level"] == "room"
        assert candidate["room"] == "kitchen"
        assert candidate["estimate_unit_id"] == "kitchen_primary"
        assert candidate["proposed_treatment"] == "package_driver"
        assert candidate["strength"] == "moderate"
        assert candidate["driver_work_item_ids"] == [driver_work["work_item_id"]]
        assert candidate["support_work_item_ids"] == [support_work["work_item_id"]]
        assert candidate["child_work_item_ids"] == sorted(
            [driver_work["work_item_id"], support_work["work_item_id"]]
        )
        assert candidate["display_only"] is False
        assert candidate["contributing_candidate_ids"] == []
        assert candidate["package_candidate_id"] == make_package_candidate_id(
            estimate_id=EST_ID,
            package_type="kitchen_modernization",
            estimate_unit_id="kitchen_primary",
        )

    def test_tier_and_absorption_scope_come_from_the_legacy_resolvers(self, tmp_path):
        _, candidates = self._kitchen(tmp_path)
        (candidate,) = candidates
        # driver + support -> kitchen partial rehab (the legacy resolver),
        # with its absorption scope keyed by the pricing profile.
        assert candidate["pricing_profile"] == "kitchen_partial_rehab"
        assert candidate["pricing_tier"] == "partial_rehab"
        assert (candidate["unfloored_low"], candidate["unfloored_high"]) == (
            15_000, 35_000,
        )
        assert candidate["absorption_scope"] == _package_absorption_scope(
            "kitchen_partial_rehab"
        )

    def test_tier_spec_carries_the_property_cost_factor(self, tmp_path):
        """The legacy resolvers price tiers unscaled, so the candidate must
        adopt the same factor the work items already carry — otherwise a
        package is billed in different dollars than the children it absorbs."""
        metadata = {"price_per_sqft": 84, "sqft": 1632}
        factor, _ = resolve_property_cost_factor(metadata)
        assert factor != 1.0
        standalone, candidates = self._kitchen(tmp_path, metadata)
        (candidate,) = candidates
        assert standalone["standalone_estimate"]["property_cost_factor"] == factor
        # Same tier as the neutral build, priced through the factor once.
        assert candidate["pricing_tier"] == "partial_rehab"
        assert (candidate["unfloored_low"], candidate["unfloored_high"]) == (
            int(round(15_000 * factor)), int(round(35_000 * factor)),
        )

    def test_factor_applies_to_the_tier_exactly_once(self, tmp_path):
        """Scaling happens on adoption, not per child and not twice: the
        factored candidate is the neutral candidate through one factor."""
        metadata = {"price_per_sqft": 84, "sqft": 1632}
        factor, _ = resolve_property_cost_factor(metadata)
        _, neutral = self._kitchen(tmp_path / "neutral")
        _, factored = self._kitchen(tmp_path / "factored", metadata)
        (neutral_candidate,) = neutral
        (factored_candidate,) = factored
        for bound in ("unfloored_low", "unfloored_high"):
            assert factored_candidate[bound] == int(
                round(neutral_candidate[bound] * factor)
            )

    def test_small_children_leave_the_tier_unfloored(self, tmp_path):
        standalone, candidates = self._kitchen(tmp_path)
        (candidate,) = candidates
        child_low = sum(
            item["low"] for item in standalone["work_items"]
            if item["work_item_id"] in candidate["child_work_item_ids"]
        )
        assert child_low < candidate["unfloored_low"]
        assert candidate["cost_floor_applied"] is False
        assert (candidate["low"], candidate["high"]) == (
            candidate["unfloored_low"], candidate["unfloored_high"],
        )

    def test_output_is_deterministic(self, tmp_path):
        standalone, first = self._kitchen(tmp_path)
        projection, catalog = _projection_and_catalog(
            tmp_path,
            _pkg_item(
                "worn_cabinets", "kitchen", "kitchen_modernization",
                "package_driver", trade_bucket="kitchen_cabinets_counters",
                work_item_code="CABINETS_REPLACE",
            ),
            _pkg_item(
                "worn_kitchen_paint", "kitchen", "kitchen_modernization",
                "package_support", trade_bucket="paint_drywall",
                work_item_code="PAINT_REFRESH",
            ),
        )
        again = build_package_candidates(
            standalone_result=standalone,
            projection=projection,
            catalog=catalog,
            estimate_id=EST_ID,
        )
        assert first == again


# ── cost floor against the children's standalone range ──────────────────────

class TestCostFloor:
    def test_floor_raises_to_the_child_standalone_sum(self, tmp_path):
        items = [
            _pkg_item(
                "gutted_bedroom_floor", "bedroom", "bedroom_turnover",
                "package_driver", work_item_code="FLOORING_REPLACE",
                scene_groups=["bedroom"],
                cost={"mode": "allowance", "base_low": 50_000, "base_high": 80_000},
            ),
        ]
        spec = [("gutted_bedroom_floor", "bedroom_1", "bedroom", None)]
        standalone, candidates = _build(tmp_path, items, spec)
        (candidate,) = candidates
        (work,) = [
            item for item in standalone["work_items"]
            if item["status"] == "active"
        ]
        # Escalation runs first (inside the legacy entrypoint) and lands on
        # the top turnover tier; the floor then raises to the child sum.
        assert candidate["pricing_tier"] == "turnover_std"
        assert candidate["low"] == max(candidate["unfloored_low"], work["low"])
        assert candidate["high"] == max(candidate["unfloored_high"], work["high"])
        assert candidate["low"] > candidate["unfloored_low"]
        assert candidate["cost_floor_applied"] is True

    def test_floor_compares_factored_tier_against_factored_children(self, tmp_path):
        """Both sides of the floor carry the same factor, so a non-neutral
        property changes the dollars but not which side wins."""
        items = [
            _pkg_item(
                "gutted_bedroom_floor", "bedroom", "bedroom_turnover",
                "package_driver", work_item_code="FLOORING_REPLACE",
                scene_groups=["bedroom"],
                cost={"mode": "allowance", "base_low": 50_000, "base_high": 80_000},
            ),
        ]
        spec = [("gutted_bedroom_floor", "bedroom_1", "bedroom", None)]
        metadata = {"price_per_sqft": 84, "sqft": 1632}
        factor, _ = resolve_property_cost_factor(metadata)
        standalone, candidates = _build(
            tmp_path / "factored", items, spec, metadata
        )
        (candidate,) = candidates
        (work,) = [
            item for item in standalone["work_items"]
            if item["status"] == "active"
        ]
        assert candidate["cost_floor_applied"] is True
        assert candidate["low"] == max(candidate["unfloored_low"], work["low"])
        assert candidate["high"] == max(candidate["unfloored_high"], work["high"])
        # The neutral build makes the same floor decision at unscaled dollars.
        _, neutral = _build(tmp_path / "neutral", items, spec)
        (neutral_candidate,) = neutral
        assert neutral_candidate["cost_floor_applied"] is True
        assert candidate["unfloored_low"] == int(
            round(neutral_candidate["unfloored_low"] * factor)
        )

    def test_invalid_standalone_input_is_a_dependency_failure(self, tmp_path):
        items = [
            _pkg_item(
                "worn_cabinets", "kitchen", "kitchen_modernization",
                "package_driver", trade_bucket="kitchen_cabinets_counters",
                work_item_code="CABINETS_REPLACE",
            ),
        ]
        projection, catalog = _projection_and_catalog(tmp_path, *items)
        condition = _cond("worn_cabinets", "kitchen_primary", "kitchen")
        standalone = _standalone(projection, [(condition, *_lattice(condition))])
        tampered = copy.deepcopy(standalone)
        tampered["standalone_estimate"]["headline"]["high"] += 1
        with pytest.raises(PassExecutionError) as excinfo:
            build_package_candidates(
                standalone_result=tampered,
                projection=projection,
                catalog=catalog,
                estimate_id=EST_ID,
            )
        assert excinfo.value.code == "StandaloneResultInvalid"
        assert excinfo.value.stage == "dependency"


# ── merged lineage and driver precedence ─────────────────────────────────────

class TestMergedLineage:
    def test_merged_active_is_one_child_with_driver_precedence(self, tmp_path):
        # Same action/trade/unit-policy/unit -> the two source items collide
        # into ONE merged active spanning both conditions. Its conditions
        # land in both the driver and support lanes; the collapsed work item
        # must appear once, as a driver.
        items = [
            _pkg_item(
                "cracked_kitchen_floor", "kitchen", "kitchen_modernization",
                "package_driver",
            ),
            _pkg_item(
                "worn_kitchen_floor", "kitchen", "kitchen_modernization",
                "package_support",
            ),
        ]
        spec = [
            ("cracked_kitchen_floor", "kitchen_primary", "kitchen", None),
            ("worn_kitchen_floor", "kitchen_primary", "kitchen", None),
        ]
        standalone, candidates = _build(tmp_path, items, spec)
        (collision,) = standalone["work_dedup_collisions"]
        merged_id = collision["active_work_item_id"]
        (candidate,) = candidates
        assert candidate["child_work_item_ids"] == [merged_id]
        assert candidate["driver_work_item_ids"] == [merged_id]
        assert candidate["support_work_item_ids"] == []
        # Suppressed dedup sources are audit-only and never appear.
        for suppressed_id in collision["suppressed_work_item_ids"]:
            assert suppressed_id not in candidate["child_work_item_ids"]


# ── scene-aware affinity ─────────────────────────────────────────────────────

class TestSceneAffinity:
    def test_affinity_routes_only_in_its_room(self, tmp_path):
        item = _pkg_item(
            "worn_cabinets", "kitchen", "kitchen_modernization",
            "package_driver", trade_bucket="kitchen_cabinets_counters",
            work_item_code="CABINETS_REPLACE",
            scene_groups=["kitchen", "bathroom"],
        )
        # Observed in its affinity room -> candidate.
        _, in_kitchen = _build(
            tmp_path / "a", [item],
            [("worn_cabinets", "kitchen_primary", "kitchen", None)],
        )
        assert len(in_kitchen) == 1
        # The same catalog item observed in a bathroom scene has no bathroom
        # affinity entry -> no candidate, work stays standalone.
        standalone, in_bathroom = _build(
            tmp_path / "b", [item],
            [("worn_cabinets", "bathroom_1", "bathroom", None)],
        )
        assert in_bathroom == []
        assert len(standalone["work_items"]) == 1


# ── opportunity corroboration over deduped representatives ──────────────────

class TestOpportunityCorroboration:
    def _item(self):
        return _pkg_item(
            "dated_kitchen_finishes", "kitchen", "kitchen_modernization",
            "package_driver", kind="modernization",
            trade_bucket="kitchen_cabinets_counters",
            work_item_code="FINISHES_REFRESH",
        )

    def test_lone_opportunity_driver_is_suppressed_and_work_untouched(self, tmp_path):
        def one_view(evidence):
            _with_photos(
                evidence,
                ["img_001.jpg", "img_002.jpg", "img_003.jpg"],
                exact_groups=[["img_001.jpg", "img_002.jpg", "img_003.jpg"]],
            )

        standalone, candidates = _build(
            tmp_path, [self._item()],
            [("dated_kitchen_finishes", "kitchen_primary", "kitchen", one_view)],
        )
        before = copy.deepcopy(standalone)
        assert candidates == []
        # Weak opportunities produce no package but leave their work
        # untouched — the accepted work item stays active and standalone.
        assert standalone == before
        (work,) = standalone["work_items"]
        assert work["status"] == "active"

    def test_duplicate_photos_do_not_corroborate(self, tmp_path):
        """Three photos collapsing to one representative are one view; two
        genuinely distinct views corroborate."""
        def two_views(evidence):
            _with_photos(evidence, ["img_001.jpg", "img_002.jpg"])

        _, corroborated = _build(
            tmp_path / "distinct", [self._item()],
            [("dated_kitchen_finishes", "kitchen_primary", "kitchen", two_views)],
        )
        (candidate,) = corroborated
        assert candidate["proposed_treatment"] == (
            "opportunity_driver_with_multiphoto_corroboration"
        )

    def test_support_corroborates_an_opportunity_driver(self, tmp_path):
        def one_view(evidence):
            _with_photos(evidence, ["img_001.jpg"])

        items = [
            self._item(),
            _pkg_item(
                "worn_kitchen_paint", "kitchen", "kitchen_modernization",
                "package_support", trade_bucket="paint_drywall",
                work_item_code="PAINT_REFRESH",
            ),
        ]
        spec = [
            ("dated_kitchen_finishes", "kitchen_primary", "kitchen", one_view),
            ("worn_kitchen_paint", "kitchen_primary", "kitchen", one_view),
        ]
        _, candidates = _build(tmp_path, items, spec)
        (candidate,) = candidates
        assert candidate["proposed_treatment"] == (
            "opportunity_driver_with_corroboration"
        )


# ── whole-home display-only aggregate ────────────────────────────────────────

class TestWholeHome:
    def _turnover_items(self):
        return [
            _pkg_item(
                "worn_bedroom_floor", "bedroom", "bedroom_turnover",
                "package_driver", scene_groups=["bedroom"],
                work_item_code="BEDROOM_FLOOR_REPAIR",
            ),
            _pkg_item(
                "worn_living_floor", "living", "living_turnover",
                "package_driver", scene_groups=["living_areas"],
                work_item_code="LIVING_FLOOR_REPAIR",
            ),
        ]

    def test_two_turnover_rooms_emit_the_display_only_aggregate(self, tmp_path):
        spec = [
            ("worn_bedroom_floor", "bedroom_1", "bedroom", None),
            ("worn_living_floor", "living_1", "living_areas", None),
        ]
        _, candidates = _build(tmp_path, self._turnover_items(), spec)
        rooms = [c for c in candidates if not c["display_only"]]
        aggregates = [c for c in candidates if c["display_only"]]
        assert len(rooms) == 2
        (aggregate,) = aggregates
        assert aggregate["package_type"] == "interior_paint_flooring_refresh"
        assert aggregate["package_level"] == "property"
        assert aggregate["room"] == "whole_home"
        assert aggregate["estimate_unit_id"] == "whole_home"
        assert aggregate["proposed_treatment"] == "whole_home_turnover_aggregate"
        assert aggregate["child_work_item_ids"] == []
        assert aggregate["driver_work_item_ids"] == []
        assert aggregate["support_work_item_ids"] == []
        assert aggregate["contributing_candidate_ids"] == sorted(
            room["package_candidate_id"] for room in rooms
        )
        assert aggregate["low"] == sum(room["low"] for room in rooms)
        assert aggregate["high"] == sum(room["high"] for room in rooms)
        assert aggregate["cost_floor_applied"] is False

    def test_one_turnover_room_emits_no_aggregate(self, tmp_path):
        spec = [("worn_bedroom_floor", "bedroom_1", "bedroom", None)]
        _, candidates = _build(tmp_path, [self._turnover_items()[0]], spec)
        assert len(candidates) == 1
        assert not any(c["display_only"] for c in candidates)
