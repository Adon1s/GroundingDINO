"""
Tests for tools.rehab_packages — package inference + reconciliation for v4.

Phase 1: reconciliation tests with hand-crafted stub packages (built first
to validate no-double-counting invariants before inference enters the picture).
Phase 2: package inference rule tests.
Phase 3: light integration tests against the wired v4 pipeline.
"""

import asyncio
import json
from pathlib import Path

import pytest

from tools.rehab_packages import (
    BATHROOM_FULL_REHAB,
    BATHROOM_PARTIAL_REHAB,
    BATHROOM_REFRESH,
    BATHROOM_REPAIR_HEAVY,
    BATHROOM_REPAIR_LIGHT,
    EXTERIOR_REPAIR_HEAVY,
    EXTERIOR_REPAIR_LIGHT,
    PACKAGE_CATEGORY_REPAIR,
    PACKAGE_ROLE_DRIVER,
    PACKAGE_ROLE_SUPPORT,
    PACKAGE_TYPE_EXTERIOR_REPAIR,
    PACKAGE_VERIFICATION_NOT_RUN,
    REPAIR_SUPPORT_MARKER,
    ROOM_EXTERIOR,
    _escalate_pricing_tier_if_undercut,
    _package_absorption_scope,
    _split_integer,
    aggregate_whole_home_turnover,
    build_package_affinity,
    classify_component,
    compute_package_strength,
    finalize_package_candidates,
    infer_package_candidates,
    package_affinity_for,
    paired_repair_package_type,
    reconcile_packages_and_estimate_units,
    run_pass_2f_batch,
)
from tools.renovation_estimate import (
    GROUP_BUDGET_CAPS,
    CatalogEstimateMeta,
    EstimateCandidate,
    _meaningful_scope_hint,
    compute_renovation_estimate,
)
from tools.scene_classifier_passes import PASS_2F_ROOM_PROMPTS
from tools.renovation_estimate_v4 import compute_renovation_estimate_v4
import tools.rehab_packages as rp


# ─── Builders ────────────────────────────────────────────────────────────────

def _member(unit_key, room_id, issue_ids, *, counts=True, estimate_unit_id=None):
    return {
        "unit_key": unit_key,
        "counts_toward_estimate": counts,
        "estimate_unit_id": estimate_unit_id or unit_key,
        "room_surrogate_id": room_id,
        "issue_ids": list(issue_ids),
        "estimate_scope_keys": [],
    }


def _line_item(estimate_unit_id, cost_low, cost_high, *, stack_behavior="sum",
               room_surrogate_id="", source_issue_ids=None, unit_members=None,
               catalog_item_id=None, trade_bucket=None, cost_model="line_item",
               cost_model_source="legacy_default", estimate_scope="required_rehab",
               is_valid_detection=True):
    return {
        "estimate_unit_id": estimate_unit_id,
        "catalog_item_id": catalog_item_id or estimate_unit_id,
        "trade_bucket": trade_bucket or "",
        "cost_model": cost_model,
        "cost_model_source": cost_model_source,
        "estimate_scope": estimate_scope,
        "is_valid_detection": is_valid_detection,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "stack_behavior": stack_behavior,
        "room_surrogate_id": room_surrogate_id,
        "source_issue_ids": list(source_issue_ids or []),
        "unit_members": unit_members or [],
    }


def _group(name, line_items, *, risk_exposure_high=0):
    return {
        "group": name,
        "line_items": line_items,
        "risk_exposure_high": risk_exposure_high,
    }


def _stub_package(package_id, room_surrogate_id, *, supporting_issue_ids,
                  cost_low, cost_high, package_type="stub", estimate_group="stub",
                  cap_behavior="respect_group_cap", estimate_unit_id=None,
                  absorption_scope=None):
    return {
        "package_id": package_id,
        "package_type": package_type,
        "room_surrogate_id": room_surrogate_id,
        "estimate_unit_id": estimate_unit_id or "",
        "estimate_group": estimate_group,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "cost_midpoint": (cost_low + cost_high) // 2,
        "cap_behavior": cap_behavior,
        # Production builders set absorption_scope via _package_absorption_scope(pricing_profile).
        # Test stubs default to deriving from package_type — mirrors the previous
        # reconciliation Phase B setdefault behavior. Tests passing a pricing-tier name
        # (e.g. "kitchen_partial_rehab") as package_type will get a real scope; tests
        # using "stub" get an empty scope and rely on supporting_issue matching.
        "absorption_scope": absorption_scope or _package_absorption_scope(package_type),
        "absorbed_unit_member_refs": [],
        "absorbed_total_low": 0,
        "absorbed_total_high": 0,
        "replacement_delta_low": 0,
        "replacement_delta_high": 0,
        "supporting_issue_ids": list(supporting_issue_ids),
        "supporting_catalog_item_ids": [],
        "trigger_reason": "stub",
        "level_decision_notes": [],
    }


def _candidate(*, catalog_item_id, name=None, kind="defect", severity=2,
               scope="repair", trade_bucket="flooring",
               group="other", strategy="repair_only",
               estimate_tier="medium", stack_behavior="sum",
               room_surrogate_id="", issue_ids=None,
               photo_keys=None,
               effective_posture="repair", is_valid_detection=True,
               package_role=None):
    meta = CatalogEstimateMeta(
        estimate_tier=estimate_tier,
        strategy=strategy,
        group=group,
        stack_behavior=stack_behavior,
        unit_policy="per_scope",
        affects_estimate=True,
        requires_2f_for_estimate=False,
    )
    c = EstimateCandidate(
        catalog_item_id=catalog_item_id,
        catalog_item_name=name or catalog_item_id,
        estimate_meta=meta,
        kind=kind,
        severity=severity,
        scope=scope,
        trade_bucket=trade_bucket,
    )
    c.room_surrogate_id = room_surrogate_id
    c.issue_ids = list(issue_ids or [])
    c.photo_keys = list(photo_keys or [])
    c.effective_posture = effective_posture
    c.is_valid_detection = is_valid_detection
    c.package_role = package_role
    return c


def _cat_item(catalog_id, *, kind="defect", tier="work",
              category="cosmetic", severity=2):
    return {
        "id": catalog_id,
        "kind": kind,
        "tier": tier,
        "category": category,
        "severity": severity,
    }


def _affinity(*entries):
    """Build a `package_affinity` block from (room, package_type, package_role)
    tuples, splattable into a synthetic catalog item."""
    return {
        "package_affinity": {
            room: {"package_type": package_type, "package_role": package_role}
            for room, package_type, package_role in entries
        }
    }


def _surrogate(sid, scene):
    return {"room_surrogate_id": sid, "scene": scene}


class _FilteringPackageVLM:
    def __init__(self):
        self.calls = []

    async def analyze_images(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({
            "verification_status": "confirmed",
            "confirmed_issue_ids": ["issue_1", "issue_2", "unreviewed"],
            "rejected_issue_ids": ["issue_2"],
            "evidence_summary": "Only the reviewed photo supports the package.",
            "visible_room_count": "one_room",
            "visible_room_count_evidence": "Same vanity and tile layout.",
        })


def test_pass_2f_batch_filters_evidence_to_selected_review_photos():
    tmp_dir = Path("tests") / "_tmp_rehab_packages"
    tmp_dir.mkdir(exist_ok=True)
    img_1 = tmp_dir / "bath_1.jpg"
    img_2 = tmp_dir / "bath_2.jpg"
    img_1.write_bytes(b"image-one")
    img_2.write_bytes(b"image-two")
    vlm = _FilteringPackageVLM()
    package = {
        "package_id": "bathroom_modernization__bathroom_primary",
        "package_type": "bathroom_modernization",
        "package_category": "modernization",
        "room": "bathroom",
        "review_photo_keys": ["bath_1.jpg", "bath_2.jpg"],
        "supporting_issue_ids": ["issue_1", "issue_2"],
        "evidence_items": [{
            "catalog_item_id": "outdated_bathroom_finishes",
            "issue_ids": ["issue_1", "issue_2"],
            "issue_refs": [
                {
                    "issue_id": "issue_1",
                    "photo_key": "bath_1.jpg",
                    "observation": "dated vanity",
                    "room_surrogate_id": "bathroom_1",
                },
                {
                    "issue_id": "issue_2",
                    "photo_key": "bath_2.jpg",
                    "observation": "dated tile",
                    "room_surrogate_id": "bathroom_2",
                },
            ],
            "photo_keys": ["bath_1.jpg", "bath_2.jpg"],
            "observations": ["dated vanity", "dated tile"],
        }],
    }

    try:
        verifications, trace = asyncio.run(run_pass_2f_batch(
            [package],
            vlm_client=vlm,
            model_config={"model": "test", "provider": "test"},
            photo_key_to_path={"bath_1.jpg": img_1, "bath_2.jpg": img_2},
            max_images=1,
        ))
    finally:
        for path in (img_1, img_2):
            if path.exists():
                path.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    record = verifications["bathroom_modernization__bathroom_primary"]
    assert trace["attempted_count"] == 1
    assert record["review_photo_keys"] == ["bath_1.jpg"]
    assert record["reviewed_issue_ids"] == ["issue_1"]
    assert record["confirmed_issue_ids"] == ["issue_1"]
    assert record["rejected_issue_ids"] == []
    assert record["visible_room_count"] == "one_room"
    prompt = vlm.calls[0]["user_prompt"]
    assert "issue_1" in prompt
    assert "issue_2" not in prompt


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: micro-unit tests on helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitInteger:

    def test_even_split(self):
        assert _split_integer(100, 4) == [25, 25, 25, 25]

    def test_remainder_distributed_to_first_entries(self):
        assert _split_integer(101, 3) == [34, 34, 33]
        assert _split_integer(1003, 3) == [335, 334, 334]

    def test_zero(self):
        assert _split_integer(0, 3) == [0, 0, 0]

    def test_n_zero(self):
        assert _split_integer(100, 0) == []


class TestClassifyComponent:

    def test_kitchen_cabinets(self):
        c = _candidate(catalog_item_id="outdated_or_damaged_cabinets",
                       trade_bucket="kitchen_cabinets_counters")
        assert classify_component(c) == "cabinets"

    def test_kitchen_counter(self):
        c = _candidate(catalog_item_id="countertop_damage",
                       trade_bucket="kitchen_cabinets_counters")
        assert classify_component(c) == "counter"

    def test_kitchen_appliance(self):
        c = _candidate(catalog_item_id="appliance_damage_or_missing",
                       trade_bucket="kitchen_cabinets_counters")
        assert classify_component(c) == "appliance"

    def test_bathroom_tile(self):
        c = _candidate(catalog_item_id="tile_or_grout_damage",
                       trade_bucket="bathroom_fixtures_tile")
        assert classify_component(c) == "tile"

    def test_bathroom_vanity(self):
        c = _candidate(catalog_item_id="dated_bathroom_vanity_light",
                       trade_bucket="bathroom_fixtures_tile")
        assert classify_component(c) == "vanity"

    def test_flooring_trade_bucket(self):
        c = _candidate(catalog_item_id="bare_or_missing_finish_flooring",
                       trade_bucket="flooring")
        assert classify_component(c) == "flooring"

    def test_unclassified_returns_none(self):
        c = _candidate(catalog_item_id="something_else",
                       trade_bucket="cleaning_turnover")
        assert classify_component(c) is None

    # Issue 3: electrical bucket split — distinguish panel-work/visible-risk
    # from low-cost items (GFCI, exhaust fan, dated outlets) so the latter
    # don't trigger REPAIR_HEAVY tiers.
    def test_electrical_visible_risks_is_heavy(self):
        c = _candidate(catalog_item_id="visible_electrical_risks",
                       trade_bucket="electrical")
        assert classify_component(c) == "electrical_heavy"

    def test_electrical_gfci_is_light(self):
        c = _candidate(catalog_item_id="bathroom_gfci_missing_or_damaged",
                       trade_bucket="electrical")
        assert classify_component(c) == "electrical_light"

    def test_electrical_exhaust_fan_is_light(self):
        c = _candidate(catalog_item_id="exhaust_fan_missing_or_damaged",
                       trade_bucket="electrical")
        assert classify_component(c) == "electrical_light"

    def test_electrical_dated_outlets_is_light(self):
        c = _candidate(catalog_item_id="dated_electrical_outlets_switches",
                       trade_bucket="electrical")
        assert classify_component(c) == "electrical_light"


class TestThreeKindDriverLanes:
    """Driver dispatch under observation-kind-v2: degradation drives in the
    defect lane (concrete deterioration evidence), modernization in the
    opportunity lane (subjective datedness, corroboration-gated), and a
    driver-role item with an unroutable kind fails loud instead of silently
    vanishing from both lanes."""

    def _catalog(self, driver_kind):
        return {
            "items": [{
                **_cat_item("worn_or_stained_carpet", kind=driver_kind,
                            category="cosmetic"),
                "package_role": "package_driver",
                "package_type": "bedroom_modernization",
                "estimate": {"estimate_tier": "high", "group": "flooring"},
            }]
        }

    def _carpet_candidate(self, kind):
        return _candidate(
            catalog_item_id="worn_or_stained_carpet",
            kind=kind,
            trade_bucket="flooring",
            group="bedroom",
            room_surrogate_id="bed_1",
            issue_ids=["i_carpet"],
            photo_keys=["b1.jpg"],
        )

    @pytest.mark.parametrize("kind", ["defect", "degradation"])
    def test_degradation_drives_like_defect(self, kind):
        candidates = infer_package_candidates(
            [self._carpet_candidate(kind)],
            [_surrogate("bed_1", "bedroom")],
            self._catalog(kind),
        )
        assert len(candidates) == 1
        assert candidates[0]["trigger_reason"] == "package_driver"

    @pytest.mark.parametrize("kind", ["upgrade", "modernization"])
    def test_modernization_gated_like_upgrade(self, kind):
        c = self._carpet_candidate(kind)
        candidates = infer_package_candidates(
            [c], [_surrogate("bed_1", "bedroom")], self._catalog(kind),
        )
        # lone opportunity driver, single photo: suppressed pending corroboration
        assert candidates == []
        assert c.is_valid_detection is False
        assert c.pass_2f_fallback_reason == (
            "insufficient_corroboration_for_opportunity_driver"
        )

    def test_unknown_driver_kind_raises(self):
        with pytest.raises(ValueError, match="unroutable"):
            infer_package_candidates(
                [self._carpet_candidate("flooble")],
                [_surrogate("bed_1", "bedroom")],
                self._catalog("flooble"),
            )

    def test_modernization_display_class_is_marketability(self):
        item = _cat_item("dated_vanity_style", kind="modernization",
                         category="cosmetic")
        assert rp.catalog_display_class(item) == rp.DISPLAY_CLASS_MARKETABILITY


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: reconciliation tests with stub packages
# ═══════════════════════════════════════════════════════════════════════════

class TestKitchenPackageCandidates:

    def _catalog(self):
        return {
            "items": [
                {
                    **_cat_item("outdated_kitchen_finishes", kind="upgrade", category="opportunity"),
                    "display_class": "marketability",
                    "package_role": "package_driver",
                    "package_type": "kitchen_modernization",
                    "estimate": {"estimate_tier": "high", "group": "kitchen"},
                },
                {
                    **_cat_item("countertop_damage", kind="defect"),
                    "display_class": "estimate_driver",
                    "package_role": "package_support",
                    "package_type": "kitchen_modernization",
                    "estimate": {"estimate_tier": "medium", "group": "kitchen"},
                },
                {
                    **_cat_item("dated_lighting_fixtures", kind="upgrade", category="opportunity"),
                    "display_class": "marketability",
                    "package_role": "package_support",
                    "package_type": "kitchen_modernization",
                    "estimate": {"estimate_tier": "medium", "group": "kitchen"},
                },
            ]
        }

    def test_single_photo_outdated_only_is_suppressed(self):
        c = _candidate(
            catalog_item_id="outdated_kitchen_finishes",
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id="kitchen_1",
            issue_ids=["i_outdated"],
            photo_keys=["k1.jpg"],
        )
        suppressed = []
        candidates = infer_package_candidates(
            [c],
            [_surrogate("kitchen_1", "kitchen")],
            self._catalog(),
            suppressed_out=suppressed,
        )

        assert candidates == []
        assert c.is_valid_detection is False
        assert c.pass_2f_fallback_reason == "insufficient_corroboration_for_opportunity_driver"
        assert len(suppressed) == 1
        assert suppressed[0]["suppression_reason"] == "weak_no_qualifying_pattern"
        assert suppressed[0]["supporting_photo_count"] == 1

    def test_multiphoto_outdated_only_creates_audit_candidate_without_confirmation(self):
        c = _candidate(
            catalog_item_id="outdated_kitchen_finishes",
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id="kitchen_1",
            issue_ids=["i_outdated"],
            photo_keys=["k1.jpg", "k2.jpg", "k2.jpg"],
        )
        candidates = infer_package_candidates(
            [c],
            [_surrogate("kitchen_1", "kitchen")],
            self._catalog(),
        )

        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "kitchen_modernization"
        assert candidates[0]["trigger_reason"] == "opportunity_driver_with_multiphoto_corroboration"
        assert candidates[0]["supporting_photo_count"] == 2
        assert candidates[0]["corroboration_basis"] == "multi_photo_same_issue"
        assert candidates[0]["review_photo_keys"] == ["k1.jpg", "k2.jpg"]
        assert candidates[0]["evidence_items"][0]["supporting_photo_count"] == 2

        packages, audit = finalize_package_candidates(candidates)
        assert packages == []
        assert audit[0]["verification_status"] == "not_run"
        assert audit[0]["audit_only"] is True

    def test_multiphoto_outdated_can_corroborate_across_same_estimate_unit(self):
        c1 = _candidate(
            catalog_item_id="outdated_kitchen_finishes",
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id="kitchen_1",
            issue_ids=["i_outdated_1"],
            photo_keys=["k1.jpg"],
        )
        c2 = _candidate(
            catalog_item_id="outdated_kitchen_finishes",
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id="kitchen_2",
            issue_ids=["i_outdated_2"],
            photo_keys=["k2.jpg"],
        )
        c1.billable_estimate_unit_id = "kitchen_primary"
        c2.billable_estimate_unit_id = "kitchen_primary"

        candidates = infer_package_candidates(
            [c1, c2],
            [_surrogate("kitchen_1", "kitchen"), _surrogate("kitchen_2", "kitchen")],
            self._catalog(),
        )

        assert len(candidates) == 1
        assert candidates[0]["trigger_reason"] == "opportunity_driver_with_multiphoto_corroboration"
        assert candidates[0]["supporting_photo_count"] == 2
        assert candidates[0]["review_photo_keys"] == ["k1.jpg", "k2.jpg"]

    def test_confirmed_multiphoto_candidate_becomes_estimate_package(self):
        c = _candidate(
            catalog_item_id="outdated_kitchen_finishes",
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id="kitchen_1",
            issue_ids=["i_outdated"],
            photo_keys=["k1.jpg", "k2.jpg"],
        )
        candidates = infer_package_candidates(
            [c],
            [_surrogate("kitchen_1", "kitchen")],
            self._catalog(),
        )
        packages, audit = finalize_package_candidates(
            candidates,
            {
                candidates[0]["package_id"]: {
                    "verification_status": "confirmed",
                    "confirmed_issue_ids": ["i_outdated"],
                    "evidence_summary": "Visible dated kitchen finishes.",
                }
            },
        )

        assert len(packages) == 1
        assert packages[0]["estimate_eligible"] is True
        assert packages[0]["pricing_profile"] == "kitchen_refresh"
        assert audit[0]["ui_eligible"] is True

    def test_opportunity_driver_plus_support_still_creates_candidate(self):
        outdated = _candidate(
            catalog_item_id="outdated_kitchen_finishes",
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id="kitchen_1",
            issue_ids=["i_outdated"],
            photo_keys=["k1.jpg"],
        )
        counter = _candidate(
            catalog_item_id="countertop_damage",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id="kitchen_1",
            issue_ids=["i_counter"],
            photo_keys=["k1.jpg"],
        )
        candidates = infer_package_candidates(
            [outdated, counter],
            [_surrogate("kitchen_1", "kitchen")],
            self._catalog(),
        )

        assert len(candidates) == 1
        assert candidates[0]["trigger_reason"] == "opportunity_driver_with_corroboration"
        assert candidates[0]["corroboration_basis"] == "driver_plus_support"

    def test_multiple_supports_same_unit_create_candidate(self):
        counter = _candidate(
            catalog_item_id="countertop_damage",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id="kitchen_1",
            issue_ids=["i_counter"],
        )
        lighting = _candidate(
            catalog_item_id="dated_lighting_fixtures",
            kind="upgrade",
            trade_bucket="electrical",
            group="electrical",
            room_surrogate_id="kitchen_1",
            issue_ids=["i_light"],
        )
        candidates = infer_package_candidates(
            [counter, lighting],
            [_surrogate("kitchen_1", "kitchen")],
            self._catalog(),
        )

        assert len(candidates) == 1
        assert candidates[0]["trigger_reason"] == "multiple_package_support_same_estimate_unit"
        assert set(candidates[0]["supporting_issue_ids"]) == {"i_counter", "i_light"}

    def test_supports_in_different_units_do_not_create_candidate(self):
        counter = _candidate(
            catalog_item_id="countertop_damage",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id="kitchen_1",
            issue_ids=["i_counter"],
        )
        counter.billable_estimate_unit_id = "kitchen_primary"
        lighting = _candidate(
            catalog_item_id="dated_lighting_fixtures",
            kind="upgrade",
            trade_bucket="electrical",
            group="electrical",
            room_surrogate_id="kitchen_2",
            issue_ids=["i_light"],
        )
        lighting.billable_estimate_unit_id = "kitchen_secondary"

        candidates = infer_package_candidates(
            [counter, lighting],
            [_surrogate("kitchen_1", "kitchen"), _surrogate("kitchen_2", "kitchen")],
            self._catalog(),
        )

        assert candidates == []


class TestAmbientSupportDemotion:
    """Mechanism #1: a support recurring across >= _AMBIENT_SUPPORT_MIN_UNITS
    distinct units is 'ambient' (a property-wide trait) and must not be the
    marginal vote that mints a driverless package. It still corroborates a
    driver-anchored package and is still costed; drivers are never demoted.
    """

    def _catalog(self):
        return {
            "items": [
                {
                    **_cat_item("recurring_cosmetic_a", kind="upgrade", category="cosmetic"),
                    "display_class": "marketability",
                    "package_role": "package_support",
                    "package_type": "kitchen_modernization",
                    "estimate": {"estimate_tier": "medium", "group": "kitchen"},
                },
                {
                    **_cat_item("recurring_cosmetic_b", kind="upgrade", category="cosmetic"),
                    "display_class": "marketability",
                    "package_role": "package_support",
                    "package_type": "kitchen_modernization",
                    "estimate": {"estimate_tier": "medium", "group": "kitchen"},
                },
                {
                    **_cat_item("local_kitchen_driver", kind="defect"),
                    "display_class": "estimate_driver",
                    "package_role": "package_driver",
                    "package_type": "kitchen_modernization",
                    "estimate": {"estimate_tier": "high", "group": "kitchen"},
                },
            ]
        }

    @staticmethod
    def _support(cat_id, unit, issue):
        return _candidate(
            catalog_item_id=cat_id,
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id=unit,
            issue_ids=[issue],
            photo_keys=[f"{unit}.jpg"],
        )

    @staticmethod
    def _driver(unit, issue):
        return _candidate(
            catalog_item_id="local_kitchen_driver",
            kind="defect",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id=unit,
            issue_ids=[issue],
            photo_keys=[f"{unit}.jpg"],
            effective_posture="repair",
        )

    def test_recurring_support_across_three_units_does_not_mint_packages(self):
        # Two cosmetic supports recur across 3 driverless units -> both ambient
        # -> no non-ambient supports remain -> no package minted anywhere.
        cands = []
        for i in (1, 2, 3):
            cands.append(self._support("recurring_cosmetic_a", f"room_{i}", f"a_{i}"))
            cands.append(self._support("recurring_cosmetic_b", f"room_{i}", f"b_{i}"))
        suppressed = []
        candidates = infer_package_candidates(
            cands, [], self._catalog(), suppressed_out=suppressed,
        )

        assert candidates == []
        assert {s["suppression_reason"] for s in suppressed} == {
            "weak_after_ambient_support_demotion"
        }
        demoted = suppressed[0]["ambient_demoted_catalog_item_ids"]
        assert demoted == ["recurring_cosmetic_a", "recurring_cosmetic_b"]

    def test_recurring_support_below_threshold_still_mints(self):
        # Same two supports but only across 2 units (< N=3) -> not ambient ->
        # each driverless room still mints a package (current behavior).
        assert rp._AMBIENT_SUPPORT_MIN_UNITS == 3
        cands = []
        for i in (1, 2):
            cands.append(self._support("recurring_cosmetic_a", f"room_{i}", f"a_{i}"))
            cands.append(self._support("recurring_cosmetic_b", f"room_{i}", f"b_{i}"))
        candidates = infer_package_candidates(cands, [], self._catalog())

        assert len(candidates) == 2
        assert all(
            c["trigger_reason"] == "multiple_package_support_same_estimate_unit"
            for c in candidates
        )

    def test_ambient_support_still_corroborates_driver_anchored_packages(self):
        # Each of 3 rooms has its own defect driver + the recurring ambient
        # support. The driver anchors the package, so all 3 still emit and the
        # ambient support rides along as corroboration (the vanity-analogue:
        # legitimate per-room driver work is never suppressed).
        cands = []
        for i in (1, 2, 3):
            cands.append(self._driver(f"room_{i}", f"d_{i}"))
            cands.append(self._support("recurring_cosmetic_a", f"room_{i}", f"a_{i}"))
        candidates = infer_package_candidates(cands, [], self._catalog())

        assert len(candidates) == 3
        for c in candidates:
            assert "recurring_cosmetic_a" in c["supporting_catalog_item_ids"]
            assert any(
                note.startswith("ambient_supports_corroborating=")
                for note in c["level_decision_notes"]
            )

    def test_repeated_defect_driver_is_never_demoted(self):
        # A defect driver repeated across 3 units (two bad vanities in two
        # bathrooms, generalized) is never tallied as a support, so all rooms
        # keep their package.
        cands = [self._driver(f"room_{i}", f"d_{i}") for i in (1, 2, 3)]
        candidates = infer_package_candidates(cands, [], self._catalog())

        assert len(candidates) == 3


def test_real_catalog_has_kitchen_metadata_coverage():
    path = Path(__file__).resolve().parents[1] / "tools" / "issue_catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [item for item in data.get("items", []) if isinstance(item, dict)]
    assert items

    # package_affinity is the single source of truth for routing: the flat
    # routing fields must never reappear on catalog items. package_role may
    # remain only as an inert standalone/ignore annotation.
    for item in items:
        assert "package_type" not in item, item.get("id")
        assert "package_category" not in item, item.get("id")
        assert "room" not in item, item.get("id")
        assert item.get("package_role") in {None, "standalone", "ignore"}, item.get("id")

    # build_package_affinity validates room keys, type/role enums, and
    # type<->room consistency; it must accept the shipped catalog.
    table = build_package_affinity(data)
    kitchen_entries = {
        issue_id: meta for (room, issue_id), meta in table.items() if room == "kitchen"
    }
    assert kitchen_entries
    for issue_id, meta in kitchen_entries.items():
        assert meta["package_type"] == "kitchen_modernization", issue_id
        assert meta["package_role"] in {"package_driver", "package_support"}, issue_id
        assert meta["package_category"] == "modernization", issue_id
        assert meta["room"] == "kitchen", issue_id

    kitchen_items = [
        item for item in items
        if "kitchen" in (item.get("scene_groups") or [])
    ]
    assert kitchen_items
    for item in kitchen_items:
        assert item.get("display_class") in {
            "estimate_driver",
            "high_concern",
            "marketability",
            "clutter",
            "hidden",
        }
    outdated = next(item for item in kitchen_items if item["id"] == "outdated_kitchen_finishes")
    assert outdated["display_class"] == "marketability"
    assert outdated["package_affinity"] == {
        "kitchen": {
            "package_type": "kitchen_modernization",
            "package_role": "package_driver",
        }
    }


def test_package_affinity_preserves_legacy_snapshot():
    # Keystone regression guard: every entry in the snapshot of the deleted
    # in-code PACKAGE_AFFINITY + flat-field-derived routing
    # (tests/fixtures/package_affinity_snapshot.json) must still be produced,
    # unchanged, by the catalog-driven table.
    #
    # The fixture is a FROZEN LEGACY SUBSET, not a current inventory. Its
    # generator (scripts/migrate_package_affinity.py) was a one-off that can no
    # longer run — it imports the since-deleted PACKAGE_AFFINITY constant — and
    # authoring new routing in the catalog is now the supported mechanism. So
    # this asserts containment, and families added after the migration get
    # their own explicit coverage tests (see the exterior test below).
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "tools" / "issue_catalog.json").read_text(encoding="utf-8"))
    snapshot = json.loads(
        (root / "tests" / "fixtures" / "package_affinity_snapshot.json").read_text(encoding="utf-8")
    )
    table = build_package_affinity(catalog)
    rebuilt = {f"{room}|{issue_id}": meta for (room, issue_id), meta in table.items()}
    for key, meta in snapshot.items():
        assert key in rebuilt, f"legacy routing dropped: {key}"
        assert rebuilt[key] == meta, f"legacy routing changed: {key}"


def test_real_catalog_has_exterior_repair_routing():
    # Explicit coverage for the exterior family (added after the affinity
    # migration, so it is deliberately absent from the legacy snapshot).
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "tools" / "issue_catalog.json").read_text(encoding="utf-8"))
    catalog_by_id = {item["id"]: item for item in catalog["items"] if item.get("id")}
    table = build_package_affinity(catalog)
    exterior = {
        issue_id: meta for (room, issue_id), meta in table.items()
        if room == ROOM_EXTERIOR
    }

    expected_drivers = {
        "damaged_or_unsafe_deck_or_porch",
        "damaged_or_rotted_siding_or_trim",
        "exposed_sheathing_or_missing_siding",
        "damaged_soffit_or_porch_ceiling",
    }
    expected_supports = {
        "exterior_siding_discoloration_fading",
        "deck_surface_weathering",
        "patio_or_porch_surface_wear",
        "brick_weathering_or_mortar_deterioration",
    }
    assert set(exterior) == expected_drivers | expected_supports

    for issue_id, meta in exterior.items():
        assert meta["package_type"] == PACKAGE_TYPE_EXTERIOR_REPAIR
        assert meta["package_category"] == PACKAGE_CATEGORY_REPAIR, (
            "exterior must not be turnover — turnover short-circuits to "
            "confirmed_by_rule and would skip Pass 2f verification"
        )
        assert meta["room"] == ROOM_EXTERIOR
        expected_role = (
            PACKAGE_ROLE_DRIVER if issue_id in expected_drivers
            else PACKAGE_ROLE_SUPPORT
        )
        assert meta["package_role"] == expected_role

        item = catalog_by_id[issue_id]
        # Flat routing fields are validator errors; routing lives in the block.
        for forbidden in ("package_type", "package_category", "room"):
            assert forbidden not in item
        if expected_role == PACKAGE_ROLE_DRIVER:
            # A driver with no cost block makes tier escalation see $0.
            assert item.get("cost"), f"{issue_id} is a driver and needs a cost block"

    # Roof and landscaping stay out of the package on purpose: different trades,
    # and ground-level roof assessment is the pipeline's worst hallucination
    # surface. They remain standalone line items.
    for excluded in (
        "damaged_or_aged_roof_shingles",
        "roofline_water_damage_suspected",
        "clogged_or_damaged_gutters",
        "landscape_improvement_needed",
        "driveway_or_walkway_cracking",
    ):
        assert excluded not in exterior


class TestBuildPackageAffinity:

    @staticmethod
    def _catalog(block):
        return {"items": [{**_cat_item("item_x"), "package_affinity": block}]}

    def test_derives_category_and_room(self):
        table = build_package_affinity(self._catalog({
            "bathroom": {"package_type": "bathroom_repair", "package_role": "package_driver"},
        }))
        assert table == {
            ("bathroom", "item_x"): {
                "package_type": "bathroom_repair",
                "package_role": "package_driver",
                "package_category": "repair",
                "room": "bathroom",
            }
        }

    def test_raises_on_unknown_room_key(self):
        with pytest.raises(ValueError, match="not a valid package room"):
            build_package_affinity(self._catalog({
                "garage": {"package_type": "kitchen_modernization", "package_role": "package_support"},
            }))

    def test_raises_on_type_room_mismatch(self):
        with pytest.raises(ValueError, match="belongs to room"):
            build_package_affinity(self._catalog({
                "bedroom": {"package_type": "kitchen_modernization", "package_role": "package_support"},
            }))

    def test_raises_on_bad_role(self):
        with pytest.raises(ValueError, match="package_role must be"):
            build_package_affinity(self._catalog({
                "kitchen": {"package_type": "kitchen_modernization", "package_role": "standalone"},
            }))

    def test_raises_on_whole_home_refresh_reference(self):
        with pytest.raises(ValueError, match="derived downstream"):
            build_package_affinity(self._catalog({
                "whole_home": {
                    "package_type": "interior_paint_flooring_refresh",
                    "package_role": "package_support",
                },
            }))


class TestSingleRoomSceneFallback:

    _SINGLE = {
        "items": [{
            **_cat_item("bath_only_item"),
            "package_affinity": {
                "bathroom": {"package_type": "bathroom_repair", "package_role": "package_driver"},
            },
        }]
    }
    _MULTI = {
        "items": [{
            **_cat_item("multi_room_item"),
            "package_affinity": {
                "bedroom": {"package_type": "bedroom_repair", "package_role": "package_driver"},
                "living": {"package_type": "living_repair", "package_role": "package_driver"},
            },
        }]
    }

    def test_single_room_block_routes_without_scene(self):
        table = build_package_affinity(self._SINGLE)
        for scene in ("", "other", "hallway"):
            entry = package_affinity_for(scene, "bath_only_item", table)
            assert entry is not None, scene
            assert entry["package_type"] == "bathroom_repair"

    def test_single_room_block_does_not_override_resolvable_scene(self):
        table = build_package_affinity(self._SINGLE)
        assert package_affinity_for("kitchen", "bath_only_item", table) is None

    def test_multi_room_block_stays_unrouted_without_scene(self):
        table = build_package_affinity(self._MULTI)
        assert package_affinity_for("", "multi_room_item", table) is None
        assert package_affinity_for("other", "multi_room_item", table) is None

    def test_infer_routes_single_room_item_without_scene(self):
        # End-to-end: legacy flat-field parity — a bathroom item with no
        # surrogate and no scene_groups_seen still drives its bathroom package.
        catalog = {
            "items": [{
                **_cat_item("bath_only_item", kind="defect", severity=3),
                "display_class": "estimate_driver",
                "package_affinity": {
                    "bathroom": {"package_type": "bathroom_repair", "package_role": "package_driver"},
                },
                "estimate": {"estimate_tier": "medium", "group": "bathroom"},
            }]
        }
        candidate = _candidate(
            catalog_item_id="bath_only_item", kind="defect",
            trade_bucket="bathroom_fixtures_tile", group="bathroom",
            room_surrogate_id="", issue_ids=["i1"], photo_keys=["b1.jpg"],
        )
        candidates = infer_package_candidates([candidate], [], catalog)
        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "bathroom_repair"
        assert candidates[0]["room"] == "bathroom"


class TestReconcileBasic:

    def test_empty_packages(self):
        groups = [
            _group("kitchen", [
                _line_item("li1", 1000, 3000, room_surrogate_id="kitchen_1",
                           unit_members=[_member("k1", "kitchen_1", ["i1"])]),
            ]),
        ]
        result = reconcile_packages_and_estimate_units(groups, [])
        assert result["absorbed_member_count"] == 0
        assert result["package_count"] == 0
        assert result["package_total_low"] == 0
        assert result["absorbed_total_low"] == 0
        assert result["net_delta_low"] == 0
        assert result["net_delta_high"] == 0
        assert result["final_rehab"]["low"] == 1000
        assert result["final_rehab"]["high"] == 3000
        assert result["final_rehab"]["source"] == "renovation_estimate_v4"
        assert result["final_rehab"]["basis"] == "package_adjusted_rehab"
        assert result["visible_rehab"]["high"] == 3000
        assert result["package_adjusted_rehab"]["high"] == 3000
        assert result["latent_risk_exposure"]["high"] == 0
        assert result["worst_case_exposure"]["high"] == 3000
        assert result["reconciliation_warnings"] == []

    def test_single_full_absorption(self):
        groups = [
            _group("kitchen", [
                _line_item("li1", 1000, 3000, room_surrogate_id="kitchen_1",
                           unit_members=[_member("k1", "kitchen_1", ["i1"])]),
            ]),
        ]
        pkg = _stub_package("p1", "kitchen_1",
                            supporting_issue_ids=["i1"],
                            cost_low=15000, cost_high=35000)
        result = reconcile_packages_and_estimate_units(groups, [pkg])
        assert pkg["absorbed_total_low"] == 1000
        assert pkg["absorbed_total_high"] == 3000
        assert pkg["replacement_delta_low"] == 14000
        assert pkg["replacement_delta_high"] == 32000
        assert len(pkg["absorbed_unit_member_refs"]) == 1
        assert pkg["absorbed_unit_member_refs"][0]["child_id"] == "li1::k1"

        # Parent member also marked
        member = groups[0]["line_items"][0]["unit_members"][0]
        assert member["absorbed_by_package_id"] == "p1"

        retained = next(g for g in result["retained_group_totals"] if g["group"] == "kitchen")
        assert retained["low"] == 0
        assert retained["high"] == 0
        assert result["final_rehab"]["low"] == 15000
        assert result["final_rehab"]["high"] == 35000

    def test_multi_room_partial_absorption(self):
        line = _line_item(
            "li_flooring", 800, 2000,
            unit_members=[
                _member("kitchen_member", "kitchen_1", ["iss_kitchen_floor"]),
                _member("living_member", "living_room_1", ["iss_living_floor"]),
            ],
        )
        groups = [_group("flooring", [line])]
        pkg = _stub_package("k1_pkg", "kitchen_1",
                            supporting_issue_ids=["iss_kitchen_floor"],
                            cost_low=15000, cost_high=35000)
        result = reconcile_packages_and_estimate_units(groups, [pkg])

        children = line["unit_member_allocations"]
        assert len(children) == 2
        # Sum invariant exact
        assert sum(c["allocated_low"] for c in children) == 800
        assert sum(c["allocated_high"] for c in children) == 2000

        kitchen_child = next(c for c in children if c["room_surrogate_id"] == "kitchen_1")
        living_child = next(c for c in children if c["room_surrogate_id"] == "living_room_1")
        assert kitchen_child["absorbed_by_package_id"] == "k1_pkg"
        assert living_child["absorbed_by_package_id"] is None

        retained = next(g for g in result["retained_group_totals"] if g["group"] == "flooring")
        assert retained["low"] == 400
        assert retained["high"] == 1000

        assert pkg["absorbed_total_low"] == 400
        assert pkg["absorbed_total_high"] == 1000
        assert result["final_rehab"]["low"] == 400 + 15000
        assert result["final_rehab"]["high"] == 1000 + 35000

    def test_kitchen_package_absorbs_outdated_kitchen_room_allowance(self):
        groups = [
            _group("kitchen", [
                _line_item(
                    "missing_base_cabinets_exposed_subfloor:kitchen_primary",
                    5000,
                    12000,
                    catalog_item_id="missing_base_cabinets_exposed_subfloor",
                    trade_bucket="kitchen_cabinets_counters",
                    room_surrogate_id="kitchen_1",
                    unit_members=[
                        _member("cabs", "kitchen_1", ["iss_cabs"], estimate_unit_id="kitchen_primary"),
                    ],
                ),
                _line_item(
                    "outdated_kitchen_finishes:kitchen_primary",
                    2000,
                    20000,
                    catalog_item_id="outdated_kitchen_finishes",
                    trade_bucket="kitchen_cabinets_counters",
                    cost_model="room_allowance",
                    cost_model_source="catalog",
                    estimate_scope="marketability_rehab",
                    room_surrogate_id="kitchen_1",
                    unit_members=[
                        _member("dated", "kitchen_1", ["iss_dated"], estimate_unit_id="kitchen_primary"),
                    ],
                ),
            ]),
        ]
        pkg = _stub_package(
            "kitchen_partial_rehab__kitchen_primary",
            "kitchen_1",
            estimate_unit_id="kitchen_primary",
            supporting_issue_ids=["iss_cabs"],
            cost_low=15000,
            cost_high=35000,
            package_type="kitchen_partial_rehab",
            estimate_group="kitchen",
        )

        result = reconcile_packages_and_estimate_units(groups, [pkg])

        audit = pkg["absorption_audit"]
        assert "outdated_kitchen_finishes" in audit["absorbed"]["room_allowances"]
        assert audit["absorbed"]["totals"]["high"] == 32000
        assert pkg["absorbed_total_high"] == 32000
        member = next(
            m for m in result["estimate_members"]
            if m["catalog_item_id"] == "outdated_kitchen_finishes"
        )
        assert member["status"] == "absorbed"
        assert member["cost_model"] == "room_allowance"

    def test_kitchen_package_does_not_absorb_living_room_flooring_portion(self):
        line = _line_item(
            "scratched_or_damaged_flooring:multi_room",
            1000,
            4000,
            catalog_item_id="scratched_or_damaged_flooring",
            trade_bucket="flooring",
            unit_members=[
                _member("kitchen_floor", "kitchen_1", ["iss_kitchen_floor"], estimate_unit_id="kitchen_primary"),
                _member("living_floor", "living_room_1", ["iss_living_floor"], estimate_unit_id="living_room_1"),
            ],
        )
        groups = [_group("flooring", [line])]
        pkg = _stub_package(
            "kitchen_partial_rehab__kitchen_primary",
            "kitchen_1",
            estimate_unit_id="kitchen_primary",
            supporting_issue_ids=["iss_cabs"],
            cost_low=15000,
            cost_high=35000,
            package_type="kitchen_partial_rehab",
            estimate_group="kitchen",
        )

        result = reconcile_packages_and_estimate_units(groups, [pkg])

        children = line["unit_member_allocations"]
        kitchen_child = next(c for c in children if c["estimate_unit_id"] == "kitchen_primary")
        living_child = next(c for c in children if c["estimate_unit_id"] == "living_room_1")
        assert kitchen_child["absorbed_by_package_id"] == "kitchen_partial_rehab__kitchen_primary"
        assert kitchen_child["absorption_reason"] == "same_unit_line_item_scope"
        assert living_child["absorbed_by_package_id"] is None
        assert "scratched_or_damaged_flooring:kitchen_primary" in (
            pkg["absorption_audit"]["absorbed"]["partial_allocations"]
        )
        assert "scratched_or_damaged_flooring:living_room_1" in (
            pkg["absorption_audit"]["retained"]["partial_allocations"]
        )
        retained_unit_ids = [
            alloc["estimate_unit_id"]
            for member in result["estimate_members"]
            for alloc in member["unit_allocations"]
            if alloc["status"] == "retained"
        ]
        assert "living_room_1" in retained_unit_ids

    def test_same_unit_unrelated_system_item_is_not_absorbed_by_kitchen_package(self):
        groups = [
            _group("kitchen", [
                _line_item(
                    "missing_base_cabinets_exposed_subfloor:kitchen_primary",
                    5000,
                    12000,
                    catalog_item_id="missing_base_cabinets_exposed_subfloor",
                    trade_bucket="kitchen_cabinets_counters",
                    room_surrogate_id="kitchen_1",
                    unit_members=[
                        _member("cabs", "kitchen_1", ["iss_cabs"], estimate_unit_id="kitchen_primary"),
                    ],
                ),
            ]),
            _group("electrical", [
                _line_item(
                    "electrical_panel_issue:kitchen_primary",
                    2500,
                    9000,
                    catalog_item_id="electrical_panel_issue",
                    trade_bucket="electrical",
                    room_surrogate_id="kitchen_1",
                    unit_members=[
                        _member("panel", "kitchen_1", ["iss_panel"], estimate_unit_id="kitchen_primary"),
                    ],
                ),
            ]),
        ]
        pkg = _stub_package(
            "kitchen_partial_rehab__kitchen_primary",
            "kitchen_1",
            estimate_unit_id="kitchen_primary",
            supporting_issue_ids=["iss_cabs"],
            cost_low=15000,
            cost_high=35000,
            package_type="kitchen_partial_rehab",
            estimate_group="kitchen",
        )

        result = reconcile_packages_and_estimate_units(groups, [pkg])

        panel_member = next(
            m for m in result["estimate_members"]
            if m["catalog_item_id"] == "electrical_panel_issue"
        )
        assert panel_member["status"] == "retained"
        assert panel_member["retained_amount"]["high"] == 9000
        assert "electrical_panel_issue" not in pkg["absorption_audit"]["absorbed"]["line_items"]

    def test_inspection_allowance_routes_to_risk_only_not_final_rehab(self):
        groups = [
            _group("pool", [
                _line_item(
                    "empty_or_deteriorated_inground_pool:property",
                    200,
                    800,
                    catalog_item_id="empty_or_deteriorated_inground_pool",
                    trade_bucket="masonry_exterior_structure",
                    cost_model="inspection_allowance",
                    cost_model_source="derived_inspection_strategy",
                    estimate_scope="inspection_risk",
                    unit_members=[
                        _member("property", "", ["iss_pool"], estimate_unit_id="property"),
                    ],
                ),
            ], risk_exposure_high=40000),
        ]

        result = reconcile_packages_and_estimate_units(groups, [])

        item = result["estimate_members"][0]
        assert item["status"] == "risk_only"
        assert item["cost_model"] == "inspection_allowance"
        assert result["final_rehab"]["high"] == 0
        assert result["latent_risk_exposure"]["high"] == 40000

    def test_invalid_explicit_catalog_cost_model_falls_back_with_warning(self):
        groups = [
            _group("other", [
                _line_item(
                    "some_old_item:scope",
                    100,
                    500,
                    catalog_item_id="some_old_item",
                    cost_model="line_item",
                    cost_model_source="invalid_catalog_fallback",
                    unit_members=[
                        _member("scope", "", ["iss_old"], estimate_unit_id="scope"),
                    ],
                ),
            ]),
        ]

        result = reconcile_packages_and_estimate_units(groups, [])

        item = result["estimate_members"][0]
        assert item["cost_model"] == "line_item"
        assert item["cost_model_source"] == "invalid_catalog_fallback"
        assert "invalid_catalog_cost_model" in result["warnings"]

    def test_group_cap_preserved_on_retained(self):
        groups = [
            _group("kitchen", [
                _line_item("li1", 5000, 10000, stack_behavior="group_cap",
                           room_surrogate_id="kitchen_1",
                           unit_members=[_member("m1", "kitchen_1", ["i1"])]),
                _line_item("li2", 8000, 20000, stack_behavior="group_cap",
                           room_surrogate_id="kitchen_1",
                           unit_members=[_member("m2", "kitchen_1", ["i2"])]),
                _line_item("li3", 12000, 30000, stack_behavior="group_cap",
                           room_surrogate_id="kitchen_1",
                           unit_members=[_member("m3", "kitchen_1", ["i3"])]),
            ]),
        ]
        pkg = _stub_package("pkg", "kitchen_1",
                            supporting_issue_ids=["i3"],
                            cost_low=15000, cost_high=35000)
        result = reconcile_packages_and_estimate_units(groups, [pkg])
        retained = next(g for g in result["retained_group_totals"] if g["group"] == "kitchen")
        # raw_low=13000 capped at 3000, floored at li2 (8000)
        # raw_high=30000 capped at 35000 (no change), floored at li2 (20000)
        assert retained["low"] == 8000
        assert retained["high"] == 30000

    def test_package_reconciliation_respects_group_cap_by_default(self):
        groups = [
            _group("kitchen", [
                _line_item("li_retained", 12000, 28000, stack_behavior="group_cap",
                           room_surrogate_id="kitchen_1",
                           unit_members=[_member("m_retained", "kitchen_1", ["i_retained"])]),
                _line_item("li_absorbed", 8000, 27000, stack_behavior="group_cap",
                           room_surrogate_id="kitchen_1",
                           unit_members=[_member("m_absorbed", "kitchen_1", ["i_absorbed"])]),
            ]),
        ]
        pkg = _stub_package(
            "kitchen_partial_rehab__kitchen_1",
            "kitchen_1",
            supporting_issue_ids=["i_absorbed"],
            cost_low=15000,
            cost_high=45000,
            package_type="kitchen_partial_rehab",
            estimate_group="kitchen",
        )

        result = reconcile_packages_and_estimate_units(groups, [pkg])
        audit = next(
            g for g in result["package_group_reconciliation"]
            if g["group"] == "kitchen"
        )

        assert audit["original_group_raw"] == {"low": 20000, "high": 55000}
        assert audit["original_group_capped"] == {"low": 12000, "high": 35000}
        assert audit["absorbed_total"] == {"low": 8000, "high": 27000}
        assert audit["package_total"] == {"low": 15000, "high": 45000}
        assert audit["package_net_delta"] == {"low": 7000, "high": 18000}
        assert audit["pre_cap_package_adjusted"] == {"low": 19000, "high": 53000}
        # The cap clips the pre-cap total (53000), but never below the selected
        # package's own verified range (45000) — the selected-package floor.
        assert audit["post_cap_package_adjusted"]["high"] == 45000
        assert audit["cap_applied_after_packages"] is True
        assert audit["cap_override"] is False
        assert audit["package_floor_applied"] is True
        assert audit["selected_package_floor"] == {"low": 15000, "high": 45000}
        assert result["package_adjusted_rehab"]["high"] == 45000

    def test_package_net_delta_is_used_instead_of_full_package_stack(self):
        groups = [
            _group("kitchen", [
                _line_item("li_retained", 12000, 28000, stack_behavior="group_cap",
                           room_surrogate_id="kitchen_1",
                           unit_members=[_member("m_retained", "kitchen_1", ["i_retained"])]),
                _line_item("li_absorbed", 8000, 27000, stack_behavior="group_cap",
                           room_surrogate_id="kitchen_1",
                           unit_members=[_member("m_absorbed", "kitchen_1", ["i_absorbed"])]),
            ]),
        ]
        pkg = _stub_package(
            "kitchen_partial_rehab__kitchen_1",
            "kitchen_1",
            supporting_issue_ids=["i_absorbed"],
            cost_low=15000,
            cost_high=45000,
            package_type="kitchen_partial_rehab",
            estimate_group="kitchen",
        )

        result = reconcile_packages_and_estimate_units(groups, [pkg])
        audit = next(
            g for g in result["package_group_reconciliation"]
            if g["group"] == "kitchen"
        )

        assert audit["package_net_delta"]["high"] == max(
            0,
            audit["package_total"]["high"] - audit["absorbed_total"]["high"],
        )
        assert audit["pre_cap_package_adjusted"]["high"] != 28000 + 45000

    def test_explicit_full_rehab_override_can_exceed_group_cap_with_warning(self):
        groups = [
            _group("kitchen", [
                _line_item("li_retained", 12000, 28000, stack_behavior="group_cap",
                           room_surrogate_id="kitchen_1",
                           unit_members=[_member("m_retained", "kitchen_1", ["i_retained"])]),
                _line_item("li_absorbed", 8000, 27000, stack_behavior="group_cap",
                           room_surrogate_id="kitchen_1",
                           unit_members=[_member("m_absorbed", "kitchen_1", ["i_absorbed"])]),
            ]),
        ]
        pkg = _stub_package(
            "kitchen_full_rehab__kitchen_1",
            "kitchen_1",
            supporting_issue_ids=["i_absorbed"],
            cost_low=15000,
            cost_high=45000,
            package_type="kitchen_full_rehab",
            estimate_group="kitchen",
            cap_behavior="allow_above_group_cap",
        )

        result = reconcile_packages_and_estimate_units(groups, [pkg])
        audit = next(
            g for g in result["package_group_reconciliation"]
            if g["group"] == "kitchen"
        )

        assert audit["cap_override"] is True
        assert audit["cap_applied_after_packages"] is False
        assert audit["post_cap_package_adjusted"]["high"] == 53000
        assert result["package_adjusted_rehab"]["high"] > 35000
        assert "cap_override_used" in result["warnings"]

    def test_audit_invariant_holds(self):
        line = _line_item(
            "li_floor", 800, 2000,
            unit_members=[
                _member("k", "kitchen_1", ["iss_k"]),
                _member("l", "living_1", ["iss_l"]),
            ],
        )
        groups = [_group("flooring", [line])]
        pkg = _stub_package("p", "kitchen_1",
                            supporting_issue_ids=["iss_k"],
                            cost_low=15000, cost_high=35000)
        result = reconcile_packages_and_estimate_units(groups, [pkg])
        assert (result["package_total_low"] - result["absorbed_total_low"]
                == result["net_delta_low"])
        assert (result["package_total_high"] - result["absorbed_total_high"]
                == result["net_delta_high"])

    def test_allocation_sum_invariant_with_rounding(self):
        line = _line_item(
            "li", 101, 1003,
            unit_members=[
                _member(f"m{i}", "kitchen_1", [f"i{i}"]) for i in range(3)
            ],
        )
        groups = [_group("kitchen", [line])]
        reconcile_packages_and_estimate_units(groups, [])
        children = line["unit_member_allocations"]
        assert len(children) == 3
        assert sum(c["allocated_low"] for c in children) == 101
        assert sum(c["allocated_high"] for c in children) == 1003
        assert [c["allocated_low"] for c in children] == [34, 34, 33]
        assert [c["allocated_high"] for c in children] == [335, 334, 334]

    def test_risk_exposure_high_only(self):
        groups = [
            _group("structure", [
                _line_item("li1", 100, 500, room_surrogate_id="",
                           unit_members=[_member("m1", "", ["i1"])]),
            ], risk_exposure_high=500),
        ]
        result = reconcile_packages_and_estimate_units(groups, [])
        assert result["final_rehab"]["low"] == 100
        assert result["final_rehab"]["high"] == 500
        assert result["package_adjusted_rehab"]["high"] == 500
        assert result["latent_risk_exposure"] == {
            "low": 0,
            "high": 500,
            "midpoint": None,
            "basis": "inspect_posture_items_and_hidden_condition_exposure",
        }
        assert result["worst_case_exposure"]["low"] == result["package_adjusted_rehab"]["low"]
        assert result["worst_case_exposure"]["high"] == (
            result["package_adjusted_rehab"]["high"]
            + result["latent_risk_exposure"]["high"]
        )

    def test_cost_floor_applied_when_absorbed_exceeds_package_cost(self):
        """If a package's tier cost is lower than what it absorbed, the cost is
        floored up to match. Replacement delta is clipped to 0 (no negative).
        Replaces the prior `package_total_below_absorbed_total_*` warning path.
        """
        groups = [
            _group("kitchen", [
                _line_item("li1", 10000, 15000,
                           room_surrogate_id="kitchen_1",
                           unit_members=[_member("m1", "kitchen_1", ["i1"])]),
            ]),
        ]
        pkg = _stub_package("p", "kitchen_1",
                            supporting_issue_ids=["i1"],
                            cost_low=8000, cost_high=10000)
        result = reconcile_packages_and_estimate_units(groups, [pkg])
        # Cost floored up to match absorbed totals
        assert pkg["cost_low"] == 10000
        assert pkg["cost_high"] == 15000
        assert pkg.get("cost_floor_applied") is True
        # Replacement delta is now 0 (no silent loss)
        assert pkg["replacement_delta_low"] == 0
        assert pkg["replacement_delta_high"] == 0
        # The legacy warning is no longer emitted (floor replaces it)
        codes = [w["code"] for w in result["reconciliation_warnings"]]
        assert "package_total_below_absorbed_total_low" not in codes
        assert "package_total_below_absorbed_total_high" not in codes

    def test_synthetic_child_for_no_billable_members(self):
        # Line item with no billable members → one synthetic child carrying full cost
        groups = [
            _group("other", [
                _line_item("li1", 200, 800, room_surrogate_id="",
                           source_issue_ids=["i_orphan"],
                           unit_members=[]),
            ]),
        ]
        result = reconcile_packages_and_estimate_units(groups, [])
        children = groups[0]["line_items"][0]["unit_member_allocations"]
        assert len(children) == 1
        assert children[0]["allocated_low"] == 200
        assert children[0]["allocated_high"] == 800
        assert result["final_rehab"]["low"] == 200
        assert result["final_rehab"]["high"] == 800

    def test_reconcile_raises_when_absorption_scope_missing(self):
        """Phase B's loud guard: a package without absorption_scope must raise KeyError.

        Prior to the fix, a missing absorption_scope was silently filled via
        _package_absorption_scope(package_type) — wrong namespace, returned an
        empty scope dict, and broke broad absorption invisibly. Now we refuse
        to proceed.
        """
        groups = [
            _group("kitchen", [
                _line_item("li1", 1000, 3000, room_surrogate_id="kitchen_1",
                           unit_members=[_member("k1", "kitchen_1", ["i1"])]),
            ]),
        ]
        pkg = _stub_package("p_bad", "kitchen_1",
                            supporting_issue_ids=["i1"],
                            cost_low=15000, cost_high=35000)
        del pkg["absorption_scope"]  # Simulate upstream omission
        raised = False
        try:
            reconcile_packages_and_estimate_units(groups, [pkg])
        except KeyError as exc:
            raised = True
            msg = str(exc)
            assert "absorption_scope" in msg
            assert "p_bad" in msg
        assert raised, "Expected KeyError when absorption_scope is missing"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: package strength + whole-home aggregation
# ═══════════════════════════════════════════════════════════════════════════

class TestComputePackageStrength:

    def test_bathroom_strong_signal_elevation(self):
        # outdated_bathroom_finishes at severity 3 → strong on the elevation branch
        sev3 = _candidate(catalog_item_id="outdated_bathroom_finishes",
                          kind="upgrade", trade_bucket="bathroom_fixtures_tile",
                          room_surrogate_id="bathroom_1",
                          issue_ids=["i1"], severity=3,
                          package_role="package_driver")
        assert compute_package_strength([sev3], []) == "strong"

    def test_generic_rule_catches_sev3_package_driver_defect(self):
        # outdated_or_damaged_vanity (kind=defect, sev=3, package_role=package_driver)
        # is NOT in _STRONG_SIGNAL_CATALOG_IDS but the generic rule catches it.
        sev3 = _candidate(catalog_item_id="outdated_or_damaged_vanity",
                          kind="defect", trade_bucket="bathroom_fixtures_tile",
                          room_surrogate_id="bathroom_1",
                          issue_ids=["i1"], severity=3,
                          package_role="package_driver")
        assert compute_package_strength([sev3], []) == "strong"

    def test_generic_rule_skips_sev2_package_driver(self):
        # Sev-2 + package_driver does NOT elevate (severity gate).
        sev2 = _candidate(catalog_item_id="outdated_or_damaged_vanity",
                          kind="defect", trade_bucket="bathroom_fixtures_tile",
                          room_surrogate_id="bathroom_1",
                          issue_ids=["i1"], severity=2,
                          package_role="package_driver")
        # One driver alone with no supports → at most "moderate"; never "strong".
        assert compute_package_strength([sev2], []) != "strong"

    def test_generic_rule_skips_sev3_package_support(self):
        # Sev-3 + package_support does NOT elevate (role gate).
        sev3 = _candidate(catalog_item_id="some_support_item",
                          kind="defect", trade_bucket="bathroom_fixtures_tile",
                          room_surrogate_id="bathroom_1",
                          issue_ids=["i1"], severity=3,
                          package_role="package_support")
        assert compute_package_strength([sev3], []) != "strong"

    def test_strength_uses_effective_affinity_metadata(self):
        # Inference can promote a standalone catalog item to package_driver via
        # scene affinity; strength must read that effective metadata, not the
        # stale/unset role on the candidate object.
        sev3 = _candidate(catalog_item_id="water_stain_ceiling",
                          kind="defect", trade_bucket="paint_drywall",
                          room_surrogate_id="bedroom_1",
                          issue_ids=["i1"], severity=3,
                          package_role="package_support")
        effective_meta = {
            id(sev3): {
                "id": "water_stain_ceiling",
                "package_role": "package_driver",
                "package_type": "bedroom_repair",
                "package_category": "repair",
                "room": "bedroom",
            }
        }
        assert compute_package_strength([sev3], [], effective_meta) == "strong"


class TestTierEscalation:
    """Issue 2B: when the resolver picks a sub-tier whose ceiling is below the
    absorbed catalog cost, escalation bumps the tier up the chain.
    """

    def test_repair_light_escalates_to_heavy_when_vanity_exceeds_ceiling(self):
        # outdated_or_damaged_vanity catalog: base_high=6000
        # BATHROOM_REPAIR_LIGHT ceiling: 2500 → must escalate to REPAIR_HEAVY (8000)
        driver = _candidate(catalog_item_id="outdated_or_damaged_vanity",
                            kind="defect", trade_bucket="bathroom_fixtures_tile",
                            room_surrogate_id="bathroom_1",
                            issue_ids=["i_vanity"], severity=3,
                            package_role="package_driver")
        catalog_lookup = {
            "outdated_or_damaged_vanity": {
                "id": "outdated_or_damaged_vanity",
                "cost": {"base_low": 500, "base_high": 6000},
            },
        }
        spec, note = _escalate_pricing_tier_if_undercut(
            BATHROOM_REPAIR_LIGHT, [driver], catalog_lookup,
        )
        assert spec == BATHROOM_REPAIR_HEAVY
        assert note is not None
        assert "bathroom_repair_light" in note
        assert "bathroom_repair_heavy" in note

    def test_no_escalation_when_tier_ceiling_already_covers_absorbed(self):
        # Cheap driver fits well within BATHROOM_REPAIR_LIGHT — no escalation.
        driver = _candidate(catalog_item_id="cheap_thing",
                            kind="defect", trade_bucket="paint_drywall",
                            room_surrogate_id="bathroom_1",
                            issue_ids=["i1"], severity=2,
                            package_role="package_driver")
        catalog_lookup = {
            "cheap_thing": {
                "id": "cheap_thing",
                "cost": {"base_low": 100, "base_high": 800},
            },
        }
        spec, note = _escalate_pricing_tier_if_undercut(
            BATHROOM_REPAIR_LIGHT, [driver], catalog_lookup,
        )
        assert spec == BATHROOM_REPAIR_LIGHT
        assert note is None

    def test_chain_walks_refresh_to_full_when_absorbed_huge(self):
        # Huge driver forces refresh → partial → full escalation in one go.
        driver = _candidate(catalog_item_id="huge_replace",
                            kind="upgrade", trade_bucket="bathroom_fixtures_tile",
                            room_surrogate_id="bathroom_1",
                            issue_ids=["i1"], severity=3,
                            package_role="package_driver")
        catalog_lookup = {
            "huge_replace": {
                "id": "huge_replace",
                # base_high=30000 exceeds REFRESH (10000) and PARTIAL (20000);
                # only FULL (35000) covers it.
                "cost": {"base_low": 5000, "base_high": 30000},
            },
        }
        spec, note = _escalate_pricing_tier_if_undercut(
            BATHROOM_REFRESH, [driver], catalog_lookup,
        )
        assert spec == BATHROOM_FULL_REHAB
        assert "bathroom_refresh" in note
        assert "bathroom_full_rehab" in note

    def test_no_escalation_at_top_tier_even_when_absorbed_exceeds(self):
        # At FULL_REHAB already; no further tier. Phase C floor handles residual.
        driver = _candidate(catalog_item_id="enormous",
                            kind="defect", trade_bucket="bathroom_fixtures_tile",
                            room_surrogate_id="bathroom_1",
                            issue_ids=["i1"], severity=3,
                            package_role="package_driver")
        catalog_lookup = {
            "enormous": {
                "id": "enormous",
                "cost": {"base_low": 40000, "base_high": 80000},
            },
        }
        spec, note = _escalate_pricing_tier_if_undercut(
            BATHROOM_FULL_REHAB, [driver], catalog_lookup,
        )
        # Stays at FULL_REHAB (no chain entry); floor will compensate.
        assert spec == BATHROOM_FULL_REHAB
        assert note is None


class TestEscalatedTierLabel:
    """pricing_tier must reflect the spec actually priced. A package escalated
    to full_rehab shipping labeled "refresh" dodges same-unit subsumption and
    double-counts its repair sibling."""

    def _catalog(self):
        return {
            "items": [
                {
                    **_cat_item("outdated_bathroom_finishes", kind="upgrade", category="opportunity"),
                    "display_class": "marketability",
                    "package_role": "package_driver",
                    "package_type": "bathroom_modernization",
                    "estimate": {"estimate_tier": "high", "group": "bathroom"},
                    # base_high above the partial_rehab ceiling (20000) forces
                    # refresh -> partial -> full escalation.
                    "cost": {"base_low": 5_000, "base_high": 30_000},
                },
                {
                    **_cat_item("water_damaged_flooring", kind="defect", category="structural"),
                    "package_role": "package_driver",
                    "package_type": "bathroom_repair",
                    "estimate": {"estimate_tier": "medium", "group": "bathroom"},
                    "cost": {"base_low": 200, "base_high": 800},
                },
            ]
        }

    def _packages_by_type(self):
        modern_driver = _candidate(
            catalog_item_id="outdated_bathroom_finishes",
            kind="upgrade",
            trade_bucket="bathroom_fixtures_tile",
            group="bathroom",
            room_surrogate_id="bathroom_1",
            issue_ids=["i_finishes"],
            photo_keys=["b1.jpg", "b2.jpg"],
        )
        repair_driver = _candidate(
            catalog_item_id="water_damaged_flooring",
            kind="defect",
            severity=3,
            trade_bucket="flooring",
            group="bathroom",
            room_surrogate_id="bathroom_1",
            issue_ids=["i_floor"],
        )
        packages = infer_package_candidates(
            [modern_driver, repair_driver],
            [_surrogate("bathroom_1", "bathroom")],
            self._catalog(),
        )
        return {p["package_type"]: p for p in packages}

    def test_escalated_package_label_matches_priced_spec(self):
        modern = self._packages_by_type()["bathroom_modernization"]
        assert modern["pricing_profile"] == "bathroom_full_rehab"
        assert modern["pricing_tier"] == "full_rehab"
        assert modern["cost_low"] == BATHROOM_FULL_REHAB[1]
        assert modern["cost_high"] == BATHROOM_FULL_REHAB[2]
        assert any(
            n.startswith("escalated_bathroom_refresh_to_bathroom_full_rehab")
            for n in modern["level_decision_notes"]
        )

    def test_escalated_modernization_subsumes_same_unit_repair(self):
        by_type = self._packages_by_type()
        modern = by_type["bathroom_modernization"]
        repair = by_type["bathroom_repair"]
        for pkg in (modern, repair):
            pkg["verification_status"] = "confirmed"
            # Physical subsumption gates on confirmed evidence rooms.
            pkg["confirmed_issue_ids"] = list(pkg["supporting_issue_ids"])
            pkg["estimate_eligible"] = True
            pkg["ui_eligible"] = True
            pkg["audit_only"] = False
        active, audit = apply_same_unit_package_subsumption([modern, repair])
        assert active == [modern]
        assert repair["audit_only"] is True
        assert repair["subsumed_by_package_id"] == modern["package_id"]
        assert audit["subsumptions"][0]["rule"] == "modernization_subsumes_repair"


class TestWholeHomeTurnoverAggregation:

    def _turnover_pkg(self, room, cost_low, cost_high, issue_ids):
        return {
            "package_id": f"{room}_turnover__{room}_1",
            "package_type": f"{room}_turnover",
            "package_category": "turnover",
            "room": room,
            "package_level": "room",
            "package_strength": "moderate",
            "confidence_score": 0.5,
            "cost_low": cost_low,
            "cost_high": cost_high,
            "supporting_issue_ids": list(issue_ids),
            "supporting_catalog_item_ids": [],
        }

    def test_aggregates_kitchen_plus_bathroom_turnover(self):
        kitchen = self._turnover_pkg("kitchen", 1_000, 3_000, ["k1"])
        bathroom = self._turnover_pkg("bathroom", 500, 2_000, ["b1"])
        agg = aggregate_whole_home_turnover([kitchen, bathroom])
        assert agg is not None
        assert agg["package_id"] == "interior_paint_flooring_refresh__whole_home"
        assert agg["cost_low"] == 1_500
        assert agg["cost_high"] == 5_000
        assert agg["room"] == "whole_home"
        assert agg["package_category"] == "turnover"

    def test_single_room_turnover_does_not_aggregate(self):
        # Only one distinct room → return None (no whole-home aggregate).
        only_bathroom = self._turnover_pkg("bathroom", 500, 2_000, ["b1"])
        assert aggregate_whole_home_turnover([only_bathroom]) is None


class TestV4Integration:

    def test_v3_output_shape_unchanged_when_v4_runs(self):
        """v3 must remain stable. Calling v3 directly returns its full schema."""
        result = compute_renovation_estimate([], {"items": []})
        assert "groups" in result
        assert "totals" in result
        assert "primary_estimate" in result
        # No v4-only fields
        assert "packages" not in result
        assert "reconciliation" not in result
        assert "final_rehab" not in result

    def test_v4_emits_packages_and_reconciliation(self):
        """Realistic kitchen-rehab fixture → v4 emits packages + reconciliation."""
        catalog = {
            "items": [
                {
                    "id": "outdated_or_damaged_cabinets",
                    "name": "Outdated Cabinets",
                    "kind": "defect",
                    "tier": "work",
                    "category": "cosmetic",
                    "severity": 2,
                    "trade_bucket": "kitchen_cabinets_counters",
                    "scope": "replace",
                    "estimate": {
                        "estimate_tier": "high",
                        "strategy": "replace_only",
                        "group": "kitchen",
                        "stack_behavior": "group_cap",
                    },
                    "cost": {"mode": "heuristic"},
                    "display_class": "estimate_driver",
                    "package_role": "package_driver",
                    "package_type": "kitchen_modernization",
                },
                {
                    "id": "countertop_damage",
                    "name": "Counter Damage",
                    "kind": "defect",
                    "tier": "work",
                    "category": "cosmetic",
                    "severity": 2,
                    "trade_bucket": "kitchen_cabinets_counters",
                    "scope": "repair",
                    "estimate": {
                        "estimate_tier": "medium",
                        "strategy": "repair_only",
                        "group": "kitchen",
                        "stack_behavior": "group_cap",
                    },
                    "cost": {"mode": "heuristic"},
                    "display_class": "estimate_driver",
                    "package_role": "package_support",
                    "package_type": "kitchen_modernization",
                },
            ],
        }
        issues = [
            {"issue_id": "i_cab", "catalog_item_id": "outdated_or_damaged_cabinets",
             "catalog_item_kind": "defect", "scene_group": "kitchen",
             "photo_key": "kitchen_1.jpg", "description": "x", "label": "defect_or_damage"},
            {"issue_id": "i_cnt", "catalog_item_id": "countertop_damage",
             "catalog_item_kind": "defect", "scene_group": "kitchen",
             "photo_key": "kitchen_2.jpg", "description": "x", "label": "defect_or_damage"},
        ]
        photos = {
            "kitchen_1.jpg": {"scene": {"id": "kitchen"}, "photo": {"index": 1}},
            "kitchen_2.jpg": {"scene": {"id": "kitchen"}, "photo": {"index": 2}},
        }
        v4 = compute_renovation_estimate_v4(
            issues_flat=issues,
            issue_catalog=catalog,
            photos=photos,
            package_verifications={
                "kitchen_modernization__kitchen_primary": {
                    "verification_status": "confirmed",
                    "confirmed_issue_ids": ["i_cab", "i_cnt"],
                    "evidence_summary": "Visible cabinet and counter modernization evidence.",
                }
            },
        )
        assert v4 is not None
        assert v4["packages"], "expected at least one package on a kitchen rehab fixture"
        assert v4["packages"][0]["package_type"] == "kitchen_modernization"
        assert v4["packages"][0]["pricing_profile"] == "kitchen_partial_rehab"

        rec = v4["reconciliation"]
        assert (rec["package_total_low"] - rec["absorbed_total_low"]
                == rec["net_delta_low"])
        assert (rec["package_total_high"] - rec["absorbed_total_high"]
                == rec["net_delta_high"])

        assert "final_rehab" in v4
        assert v4["final_rehab"]["source"] == "renovation_estimate_v4"
        assert v4["final_rehab"]["midpoint"] == (
            v4["final_rehab"]["low"] + v4["final_rehab"]["high"]
        ) // 2
        assert v4["provenance"]["packages_enabled"] is True
        assert v4["provenance"]["reconciliation_enabled"] is True

    def test_v4_with_empty_inputs(self):
        v4 = compute_renovation_estimate_v4(
            issues_flat=[],
            issue_catalog={"items": []},
            photos={},
        )
        assert v4 is not None
        assert v4["packages"] == []
        assert v4["final_rehab"]["low"] == 0
        assert v4["final_rehab"]["high"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: bathroom_modernization expansion (multi-bathroom safe costing)
# ═══════════════════════════════════════════════════════════════════════════

from tools.rehab_packages import expand_bathroom_modernization_packages
from tools.estimate_units import bathroom_metadata_cap


def _ref(issue_id, surrogate_id, photo_key=None):
    return {
        "issue_id": issue_id,
        "photo_key": photo_key or f"{surrogate_id}_p1.jpg",
        "observation": "dated finishes",
        "room_surrogate_id": surrogate_id,
    }


def _bathroom_modernization_package(
    *,
    confirmed_issue_ids,
    refs,
    reviewed_issue_ids=None,
    rejected_issue_ids=None,
    estimate_unit_id="bathroom_primary",
    verification_status="confirmed",
    supporting_issue_ids=None,
):
    """Build a confirmed bathroom_modernization package dict for expander tests.

    Mirrors a post-finalize package: supporting_issue_ids is the confirmed-only
    projection, supporting_issue_ids_original the pre-review list (confirmed +
    rejected). The old fixture hardcoded supporting == confirmed even when
    rejected ids were passed, which masked the rejected-ID absorption bug.
    """
    supporting_original = list(dict.fromkeys([
        *confirmed_issue_ids,
        *(rejected_issue_ids or []),
    ]))
    return {
        "package_id": f"bathroom_modernization__{estimate_unit_id}",
        "package_type": "bathroom_modernization",
        "package_category": "modernization",
        "room": "bathroom",
        "room_surrogate_id": "bathroom_primary",
        "estimate_unit_id": estimate_unit_id,
        "estimate_group": "bathroom",
        "verification_status": verification_status,
        "confirmed_issue_ids": list(confirmed_issue_ids),
        "reviewed_issue_ids": list(reviewed_issue_ids or supporting_original),
        "rejected_issue_ids": list(rejected_issue_ids or []),
        "supporting_issue_ids": list(
            supporting_issue_ids
            if supporting_issue_ids is not None
            else confirmed_issue_ids
        ),
        "supporting_issue_ids_original": supporting_original,
        "cost_low": 12000,
        "cost_high": 18000,
        "package_level": "room",
        "pricing_tier": "partial_rehab",
        "pricing_profile": "bathroom_partial_rehab",
        "absorption_scope": _package_absorption_scope("bathroom_partial_rehab"),
        "trigger_reason": "package_driver",
        "evidence_items": [{
            "catalog_item_id": "outdated_bathroom_finishes",
            "issue_ids": [r["issue_id"] for r in refs],
            "issue_refs": list(refs),
            "photo_keys": [r["photo_key"] for r in refs],
            "observations": ["dated finishes" for _ in refs],
            "supporting_photo_count": len({r["photo_key"] for r in refs}),
        }],
    }


def _kitchen_package():
    return {
        "package_id": "kitchen_modernization__kitchen_primary",
        "package_type": "kitchen_modernization",
        "package_category": "modernization",
        "room": "kitchen",
        "room_surrogate_id": "kitchen_1",
        "estimate_unit_id": "kitchen_primary",
        "verification_status": "confirmed",
        "confirmed_issue_ids": ["k1", "k2"],
        "reviewed_issue_ids": ["k1", "k2"],
        "rejected_issue_ids": [],
        "supporting_issue_ids": ["k1", "k2"],
        "cost_low": 30000,
        "cost_high": 50000,
        "trigger_reason": "package_driver",
        "evidence_items": [{
            "catalog_item_id": "outdated_kitchen_finishes",
            "issue_ids": ["k1", "k2"],
            "issue_refs": [
                _ref("k1", "kitchen_1"),
                _ref("k2", "kitchen_1"),
            ],
            "photo_keys": ["kitchen_1_p1.jpg", "kitchen_1_p2.jpg"],
            "observations": ["dated cabinets", "dated counters"],
            "supporting_photo_count": 2,
        }],
    }


_HOT_SIGNAL = {"likely_multiple_visible_bathrooms": True}
_COLD_SIGNAL = {"likely_multiple_visible_bathrooms": False}


class TestExpandBathroomModernizationPackages:

    def test_chandler_three_bathroom_expansion(self):
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i3", "i4"],
            refs=[
                _ref("i1", "bathroom_1"),
                _ref("i3", "bathroom_3"),
                _ref("i4", "bathroom_4"),
            ],
        )
        out, audit = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=4,
        )
        assert audit["expanded"] is True
        assert len(out) == 3
        ids = [p["package_id"] for p in out]
        assert ids == [
            "bathroom_modernization__bathroom_primary__bathroom_1",
            "bathroom_modernization__bathroom_primary__bathroom_3",
            "bathroom_modernization__bathroom_primary__bathroom_4",
        ]
        # Each expanded package isolates its surrogate's refs.
        for expanded, surrogate, issue_id in zip(out, ["bathroom_1", "bathroom_3", "bathroom_4"], ["i1", "i3", "i4"]):
            assert expanded["room_surrogate_id"] == surrogate
            assert expanded["estimate_unit_id"] == ""
            assert expanded["confirmed_issue_ids"] == [issue_id]
            assert len(expanded["evidence_items"]) == 1
            refs = expanded["evidence_items"][0]["issue_refs"]
            assert {r["room_surrogate_id"] for r in refs} == {surrogate}
            assert {r["issue_id"] for r in refs} == {issue_id}
            assert expanded["expansion_source_package_id"] == "bathroom_modernization__bathroom_primary"

    def test_reviewed_but_not_confirmed_is_ignored(self):
        # bathroom_2 has a reviewed ref but it's NOT in confirmed_issue_ids → no package for bathroom_2.
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i3"],
            refs=[
                _ref("i1", "bathroom_1"),
                _ref("i2", "bathroom_2"),  # reviewed, not confirmed
                _ref("i3", "bathroom_3"),
            ],
            reviewed_issue_ids=["i1", "i2", "i3"],
        )
        out, audit = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=4,
        )
        assert audit["expanded"] is True
        surrogates = {p["room_surrogate_id"] for p in out}
        assert surrogates == {"bathroom_1", "bathroom_3"}
        assert "bathroom_2" not in surrogates

    def test_rejected_refs_ignored(self):
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i3"],
            refs=[
                _ref("i1", "bathroom_1"),
                _ref("i2", "bathroom_2"),
                _ref("i3", "bathroom_3"),
            ],
            reviewed_issue_ids=["i1", "i2", "i3"],
            rejected_issue_ids=["i2"],
        )
        out, _ = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=4,
        )
        surrogates = {p["room_surrogate_id"] for p in out}
        assert surrogates == {"bathroom_1", "bathroom_3"}

    def test_single_confirmed_surrogate_no_expansion(self):
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i2"],
            refs=[
                _ref("i1", "bathroom_1"),
                _ref("i2", "bathroom_1"),
            ],
        )
        out, audit = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=4,
        )
        assert audit["expanded"] is False
        assert audit["fallback_reason"] == "single_qualifying_surrogate"
        assert out == [pkg]

    def test_signal_cold_no_expansion(self):
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i3"],
            refs=[_ref("i1", "bathroom_1"), _ref("i3", "bathroom_3")],
        )
        out, audit = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_COLD_SIGNAL,
            bathroom_metadata_cap=4,
        )
        assert audit["expanded"] is False
        assert audit["fallback_reason"] == "signal_cold"
        assert out == [pkg]

    def test_metadata_cap_two_limits_to_two(self):
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i3", "i4"],
            refs=[
                _ref("i1", "bathroom_1"),
                _ref("i3", "bathroom_3"),
                _ref("i4", "bathroom_4"),
            ],
        )
        out, audit = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=2,
        )
        assert audit["expanded"] is True
        assert audit["cap_applied"] is True
        surrogates = [p["room_surrogate_id"] for p in out]
        # Deterministic sort kept the first two.
        assert surrogates == ["bathroom_1", "bathroom_3"]

    def test_metadata_missing_no_expansion(self):
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i3"],
            refs=[_ref("i1", "bathroom_1"), _ref("i3", "bathroom_3")],
        )
        out, audit = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=None,
        )
        assert audit["expanded"] is False
        assert audit["fallback_reason"] == "metadata_cap_below_two"
        assert out == [pkg]

    def test_metadata_cap_helper_handles_full_plus_half(self):
        # full=2 half=1 → ceil(3) = 3
        assert bathroom_metadata_cap({"full_baths": 2, "half_baths": 1}) == 3
        # bath_count=3.5 → ceil(3.5) = 4
        assert bathroom_metadata_cap({"bath_count": 3.5}) == 4
        # missing → None
        assert bathroom_metadata_cap({}) is None

    def test_non_bathroom_modernization_packages_pass_through(self):
        bathroom = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i3"],
            refs=[_ref("i1", "bathroom_1"), _ref("i3", "bathroom_3")],
        )
        kitchen = _kitchen_package()
        # bathroom_repair shape: same structure but different package_type, out of scope.
        repair = dict(bathroom)
        repair["package_id"] = "bathroom_repair__bathroom_primary"
        repair["package_type"] = "bathroom_repair"
        out, audit = expand_bathroom_modernization_packages(
            [kitchen, bathroom, repair],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=4,
        )
        assert audit["expanded"] is True
        # Kitchen and bathroom_repair survive untouched.
        kitchen_out = next(p for p in out if p["package_type"] == "kitchen_modernization")
        repair_out = next(p for p in out if p["package_type"] == "bathroom_repair")
        assert kitchen_out is kitchen
        assert repair_out is repair

    def test_original_package_removed_when_expanded(self):
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i3"],
            refs=[_ref("i1", "bathroom_1"), _ref("i3", "bathroom_3")],
        )
        original_id = pkg["package_id"]
        out, _ = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=4,
        )
        out_ids = [p["package_id"] for p in out]
        assert original_id not in out_ids
        assert all(pid.startswith("bathroom_modernization__bathroom_primary__bathroom_") for pid in out_ids)

    def test_empty_surrogate_id_on_confirmed_ref_skipped(self):
        # Confirmed ref with empty room_surrogate_id can't anchor a costed package.
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i_blank"],
            refs=[
                _ref("i1", "bathroom_1"),
                {"issue_id": "i_blank", "photo_key": "x.jpg", "observation": "", "room_surrogate_id": ""},
            ],
        )
        out, audit = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=4,
        )
        # Only one surrogate qualifies → no expansion (1 package out).
        assert audit["expanded"] is False
        assert audit["fallback_reason"] == "single_qualifying_surrogate"
        assert out == [pkg]

    def test_unconfirmed_package_status_skipped(self):
        # An uncertain bathroom_modernization (not active) is not eligible.
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i3"],
            refs=[_ref("i1", "bathroom_1"), _ref("i3", "bathroom_3")],
            verification_status="uncertain",
        )
        out, audit = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=4,
        )
        assert audit["expanded"] is False
        assert audit["fallback_reason"] == "no_confirmed_bathroom_modernization_package"
        assert out == [pkg]


# ═══════════════════════════════════════════════════════════════════════════
# Bedroom / living package inference (parity with kitchen/bathroom).
#
# These rooms ship dormant (no catalog items today), so the path is exercised
# with inline synthetic catalogs — the same technique the kitchen tests use.
# ═══════════════════════════════════════════════════════════════════════════

class TestBedroomLivingPackageCandidates:

    def _bedroom_catalog(self):
        return {
            "items": [
                {
                    **_cat_item("worn_bedroom_carpet", kind="upgrade", category="opportunity"),
                    "display_class": "marketability",
                    **_affinity(("bedroom", "bedroom_modernization", "package_driver")),
                    "estimate": {"estimate_tier": "high", "group": "bedroom"},
                },
                {
                    **_cat_item("dated_bedroom_light_fixture", kind="upgrade", category="opportunity"),
                    "display_class": "marketability",
                    **_affinity(("bedroom", "bedroom_modernization", "package_support")),
                    "estimate": {"estimate_tier": "medium", "group": "bedroom"},
                },
            ]
        }

    def _living_repair_catalog(self):
        return {
            "items": [
                {
                    **_cat_item("damaged_living_flooring", kind="defect"),
                    "display_class": "estimate_driver",
                    **_affinity(("living", "living_repair", "package_driver")),
                    "estimate": {"estimate_tier": "medium", "group": "living"},
                },
                {
                    **_cat_item("living_wall_damage", kind="defect"),
                    "display_class": "estimate_driver",
                    **_affinity(("living", "living_repair", "package_support")),
                    "estimate": {"estimate_tier": "medium", "group": "living"},
                },
            ]
        }

    def test_bedroom_modernization_package_inferred(self):
        driver = _candidate(
            catalog_item_id="worn_bedroom_carpet", kind="upgrade",
            trade_bucket="flooring", group="bedroom",
            room_surrogate_id="bedroom_1", issue_ids=["i1"], photo_keys=["b1.jpg"],
        )
        support = _candidate(
            catalog_item_id="dated_bedroom_light_fixture", kind="upgrade",
            trade_bucket="electrical", group="bedroom",
            room_surrogate_id="bedroom_1", issue_ids=["i2"], photo_keys=["b2.jpg"],
        )
        candidates = infer_package_candidates(
            [driver, support],
            [_surrogate("bedroom_1", "bedroom")],
            self._bedroom_catalog(),
        )
        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "bedroom_modernization"
        assert candidates[0]["room"] == "bedroom"
        assert candidates[0]["estimate_group"] == "bedroom"
        assert candidates[0]["pricing_profile"] in {"bedroom_refresh", "bedroom_full_rehab"}

    def test_living_repair_package_inferred_with_living_room_scene(self):
        # Regression for the living vs living_room naming gotcha: the surrogate
        # scene id is "living_room" but the package room is "living". The
        # scene-mismatch guard must normalize and KEEP the candidates.
        driver = _candidate(
            catalog_item_id="damaged_living_flooring", kind="defect",
            trade_bucket="flooring", group="living",
            room_surrogate_id="living_room_1", issue_ids=["i1"], photo_keys=["l1.jpg"],
        )
        support = _candidate(
            catalog_item_id="living_wall_damage", kind="defect",
            trade_bucket="paint_drywall", group="living",
            room_surrogate_id="living_room_1", issue_ids=["i2"], photo_keys=["l2.jpg"],
        )
        candidates = infer_package_candidates(
            [driver, support],
            [_surrogate("living_room_1", "living_room")],
            self._living_repair_catalog(),
        )
        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "living_repair"
        assert candidates[0]["room"] == "living"
        assert candidates[0]["pricing_profile"] in {"living_repair_light", "living_repair_heavy"}

    def test_living_candidates_dropped_when_scene_truly_mismatched(self):
        # Same living_repair candidates, but the surrogate classified as a kitchen
        # scene -> no kitchen affinity entry and no fallback (the scene resolved
        # to a real room) -> ineligible.
        driver = _candidate(
            catalog_item_id="damaged_living_flooring", kind="defect",
            trade_bucket="flooring", group="living",
            room_surrogate_id="r1", issue_ids=["i1"], photo_keys=["l1.jpg"],
        )
        support = _candidate(
            catalog_item_id="living_wall_damage", kind="defect",
            trade_bucket="paint_drywall", group="living",
            room_surrogate_id="r1", issue_ids=["i2"], photo_keys=["l2.jpg"],
        )
        candidates = infer_package_candidates(
            [driver, support],
            [_surrogate("r1", "kitchen")],
            self._living_repair_catalog(),
        )
        assert candidates == []

    def test_generic_carpet_routes_by_bedroom_scene_not_stale_catalog_room(self):
        # Stale flat fields (kitchen) are inert; the package_affinity block keyed
        # by the observed scene is the router.
        catalog = {
            "items": [{
                **_cat_item("worn_or_stained_carpet", kind="defect"),
                "display_class": "estimate_driver",
                "package_role": "package_support",
                "package_type": "kitchen_modernization",
                "package_category": "modernization",
                "room": "kitchen",
                **_affinity(
                    ("bedroom", "bedroom_modernization", "package_driver"),
                    ("living", "living_modernization", "package_driver"),
                    ("kitchen", "kitchen_modernization", "package_support"),
                ),
                "estimate": {"estimate_tier": "medium", "group": "flooring"},
            }]
        }
        carpet = _candidate(
            catalog_item_id="worn_or_stained_carpet", kind="defect",
            trade_bucket="flooring", group="bedroom",
            room_surrogate_id="bedroom_1", issue_ids=["i_carpet"], photo_keys=["b1.jpg"],
        )

        candidates = infer_package_candidates(
            [carpet],
            [_surrogate("bedroom_1", "bedroom")],
            catalog,
        )

        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "bedroom_modernization"
        assert candidates[0]["room"] == "bedroom"
        assert candidates[0]["supporting_catalog_item_ids"] == ["worn_or_stained_carpet"]
        assert candidates[0]["evidence_items"][0]["package_type"] == "bedroom_modernization"

    def test_generic_carpet_routes_by_living_scene_not_stale_catalog_room(self):
        catalog = {
            "items": [{
                **_cat_item("worn_or_stained_carpet", kind="defect"),
                "display_class": "estimate_driver",
                "package_role": "package_support",
                "package_type": "kitchen_modernization",
                "package_category": "modernization",
                "room": "kitchen",
                **_affinity(
                    ("bedroom", "bedroom_modernization", "package_driver"),
                    ("living", "living_modernization", "package_driver"),
                    ("kitchen", "kitchen_modernization", "package_support"),
                ),
                "estimate": {"estimate_tier": "medium", "group": "flooring"},
            }]
        }
        carpet = _candidate(
            catalog_item_id="worn_or_stained_carpet", kind="defect",
            trade_bucket="flooring", group="living",
            room_surrogate_id="living_room_1", issue_ids=["i_carpet"], photo_keys=["l1.jpg"],
        )

        candidates = infer_package_candidates(
            [carpet],
            [_surrogate("living_room_1", "living_room")],
            catalog,
        )

        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "living_modernization"
        assert candidates[0]["room"] == "living"

    def test_generic_carpet_keeps_kitchen_fallback_behavior(self):
        catalog = {
            "items": [
                {
                    **_cat_item("outdated_kitchen_finishes", kind="upgrade", category="opportunity", severity=3),
                    "display_class": "marketability",
                    **_affinity(("kitchen", "kitchen_modernization", "package_driver")),
                    "estimate": {"estimate_tier": "high", "group": "kitchen"},
                },
                {
                    **_cat_item("worn_or_stained_carpet", kind="defect"),
                    "display_class": "estimate_driver",
                    **_affinity(
                        ("bedroom", "bedroom_modernization", "package_driver"),
                        ("living", "living_modernization", "package_driver"),
                        ("kitchen", "kitchen_modernization", "package_support"),
                    ),
                    "estimate": {"estimate_tier": "medium", "group": "flooring"},
                },
            ]
        }
        outdated = _candidate(
            catalog_item_id="outdated_kitchen_finishes", kind="upgrade",
            trade_bucket="kitchen_cabinets_counters", group="kitchen",
            room_surrogate_id="kitchen_1", issue_ids=["i_outdated"], photo_keys=["k1.jpg"],
            severity=3,
        )
        carpet = _candidate(
            catalog_item_id="worn_or_stained_carpet", kind="defect",
            trade_bucket="flooring", group="kitchen",
            room_surrogate_id="kitchen_1", issue_ids=["i_carpet"], photo_keys=["k1.jpg"],
        )

        candidates = infer_package_candidates(
            [outdated, carpet],
            [_surrogate("kitchen_1", "kitchen")],
            catalog,
        )

        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "kitchen_modernization"
        assert candidates[0]["room"] == "kitchen"
        assert candidates[0]["evidence_items"][1]["package_type"] == "kitchen_modernization"

    def test_bedroom_water_stain_routes_to_repair_affinity(self):
        catalog = {
            "items": [{
                **_cat_item("water_stain_ceiling", kind="defect"),
                "display_class": "estimate_driver",
                "package_role": "standalone",
                **_affinity(
                    ("bedroom", "bedroom_repair", "package_driver"),
                    ("living", "living_repair", "package_driver"),
                ),
                "estimate": {"estimate_tier": "medium", "group": "paint_drywall"},
            }]
        }
        stain = _candidate(
            catalog_item_id="water_stain_ceiling", kind="defect",
            trade_bucket="paint_drywall", group="bedroom",
            room_surrogate_id="bedroom_1", issue_ids=["i_stain"], photo_keys=["b1.jpg"],
        )

        candidates = infer_package_candidates(
            [stain],
            [_surrogate("bedroom_1", "bedroom")],
            catalog,
        )

        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "bedroom_repair"
        assert candidates[0]["room"] == "bedroom"

    def test_bedroom_water_stain_strength_uses_affinity_driver_role(self):
        catalog = {
            "items": [{
                **_cat_item("water_stain_ceiling", kind="defect", severity=3),
                "display_class": "estimate_driver",
                "package_role": "standalone",
                **_affinity(
                    ("bedroom", "bedroom_repair", "package_driver"),
                    ("living", "living_repair", "package_driver"),
                ),
                "estimate": {"estimate_tier": "medium", "group": "paint_drywall"},
            }]
        }
        stain = _candidate(
            catalog_item_id="water_stain_ceiling", kind="defect",
            trade_bucket="paint_drywall", group="bedroom",
            room_surrogate_id="bedroom_1", issue_ids=["i_stain"], photo_keys=["b1.jpg"],
            severity=3,
        )

        candidates = infer_package_candidates(
            [stain],
            [_surrogate("bedroom_1", "bedroom")],
            catalog,
        )

        assert len(candidates) == 1
        assert candidates[0]["package_strength"] == "strong"

    def test_bedroom_wallpaper_routes_as_modernization_support(self):
        catalog = {
            "items": [
                {
                    **_cat_item("dated_wallpaper_present", kind="upgrade", category="opportunity"),
                    "display_class": "marketability",
                    "package_role": "package_support",
                    "package_type": "kitchen_modernization",
                    "package_category": "modernization",
                    "room": "kitchen",
                    "estimate": {"estimate_tier": "medium", "group": "paint_drywall"},
                },
                {
                    **_cat_item("dated_interior_trim", kind="upgrade", category="opportunity"),
                    "display_class": "marketability",
                    "package_role": "package_support",
                    "package_type": "kitchen_modernization",
                    "package_category": "modernization",
                    "room": "kitchen",
                    "estimate": {"estimate_tier": "medium", "group": "trim_doors_windows"},
                },
            ]
        }
        for item in catalog["items"]:
            item.update(_affinity(
                ("bedroom", "bedroom_modernization", "package_support"),
                ("living", "living_modernization", "package_support"),
                ("kitchen", "kitchen_modernization", "package_support"),
            ))
        wallpaper = _candidate(
            catalog_item_id="dated_wallpaper_present", kind="upgrade",
            trade_bucket="interior_finishes", group="bedroom",
            room_surrogate_id="bedroom_1", issue_ids=["i_wallpaper"], photo_keys=["b1.jpg"],
        )
        trim = _candidate(
            catalog_item_id="dated_interior_trim", kind="upgrade",
            trade_bucket="trim_doors_windows", group="bedroom",
            room_surrogate_id="bedroom_1", issue_ids=["i_trim"], photo_keys=["b1.jpg"],
        )

        candidates = infer_package_candidates(
            [wallpaper, trim],
            [_surrogate("bedroom_1", "bedroom")],
            catalog,
        )

        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "bedroom_modernization"
        roles = {
            item["catalog_item_id"]: item["package_role"]
            for item in candidates[0]["evidence_items"]
        }
        assert roles["dated_wallpaper_present"] == "package_support"
        assert roles["dated_interior_trim"] == "package_support"

    def test_living_trim_and_doors_route_as_modernization_support(self):
        catalog = {
            "items": [
                {
                    **_cat_item("dated_interior_trim", kind="upgrade", category="opportunity"),
                    "display_class": "marketability",
                    "package_role": "package_support",
                    "package_type": "kitchen_modernization",
                    "package_category": "modernization",
                    "room": "kitchen",
                    "estimate": {"estimate_tier": "medium", "group": "trim_doors_windows"},
                },
                {
                    **_cat_item("dated_interior_doors", kind="upgrade", category="opportunity"),
                    "display_class": "marketability",
                    "package_role": "package_support",
                    "package_type": "kitchen_modernization",
                    "package_category": "modernization",
                    "room": "kitchen",
                    "estimate": {"estimate_tier": "medium", "group": "trim_doors_windows"},
                },
            ]
        }
        for item in catalog["items"]:
            item.update(_affinity(
                ("bedroom", "bedroom_modernization", "package_support"),
                ("living", "living_modernization", "package_support"),
                ("kitchen", "kitchen_modernization", "package_support"),
                ("bathroom", "bathroom_modernization", "package_support"),
            ))
        trim = _candidate(
            catalog_item_id="dated_interior_trim", kind="upgrade",
            trade_bucket="trim_doors_windows", group="living",
            room_surrogate_id="living_room_1", issue_ids=["i_trim"], photo_keys=["l1.jpg"],
        )
        doors = _candidate(
            catalog_item_id="dated_interior_doors", kind="upgrade",
            trade_bucket="trim_doors_windows", group="living",
            room_surrogate_id="living_room_1", issue_ids=["i_doors"], photo_keys=["l1.jpg"],
        )

        candidates = infer_package_candidates(
            [trim, doors],
            [_surrogate("living_room_1", "living_room")],
            catalog,
        )

        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "living_modernization"

    def test_living_fireplace_surround_is_real_living_component(self):
        catalog_path = Path(__file__).resolve().parents[1] / "tools" / "issue_catalog.json"
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        fireplace = next(
            item for item in data.get("items", [])
            if item.get("id") == "dated_fireplace_surround"
        )
        assert fireplace["scene_groups"] == ["living_areas"]
        assert fireplace["package_affinity"] == {
            "living": {
                "package_type": "living_modernization",
                "package_role": "package_support",
            }
        }
        assert "package_type" not in fireplace
        assert "room" not in fireplace

        catalog = {"items": [
            fireplace,
            {
                **_cat_item("dated_wood_paneling", kind="upgrade", category="opportunity"),
                "display_class": "marketability",
                **_affinity(
                    ("bedroom", "bedroom_modernization", "package_support"),
                    ("living", "living_modernization", "package_support"),
                    ("kitchen", "kitchen_modernization", "package_support"),
                    ("bathroom", "bathroom_modernization", "package_support"),
                ),
                "estimate": {"estimate_tier": "medium", "group": "paint_drywall"},
            },
        ]}
        surround = _candidate(
            catalog_item_id="dated_fireplace_surround", kind="upgrade",
            trade_bucket="interior_finishes", group="living",
            room_surrogate_id="living_room_1", issue_ids=["i_fireplace"], photo_keys=["l1.jpg"],
        )
        paneling = _candidate(
            catalog_item_id="dated_wood_paneling", kind="upgrade",
            trade_bucket="paint_drywall", group="living",
            room_surrogate_id="living_room_1", issue_ids=["i_paneling"], photo_keys=["l1.jpg"],
        )

        candidates = infer_package_candidates(
            [surround, paneling],
            [_surrogate("living_room_1", "living_room")],
            catalog,
        )

        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "living_modernization"
        assert {
            item["catalog_item_id"]: item["package_type"]
            for item in candidates[0]["evidence_items"]
        }["dated_fireplace_surround"] == "living_modernization"

    def test_bathroom_generic_wallpaper_is_left_to_bathroom_specific_item(self):
        catalog_path = Path(__file__).resolve().parents[1] / "tools" / "issue_catalog.json"
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        shipped_table = build_package_affinity(data)
        assert package_affinity_for(
            "bathroom", "dated_wallpaper_present", shipped_table) is None

        ids = {item.get("id") for item in data.get("items", []) if isinstance(item, dict)}
        assert "dated_bathroom_wallpaper" in ids
        assert "bathroom_paint_refresh_recommended" in ids

        catalog = {
            "items": [{
                **_cat_item("dated_wallpaper_present", kind="upgrade", category="opportunity"),
                "display_class": "marketability",
                **_affinity(
                    ("bedroom", "bedroom_modernization", "package_support"),
                    ("living", "living_modernization", "package_support"),
                    ("kitchen", "kitchen_modernization", "package_support"),
                ),
                "estimate": {"estimate_tier": "medium", "group": "paint_drywall"},
            }]
        }
        generic_wallpaper = _candidate(
            catalog_item_id="dated_wallpaper_present", kind="upgrade",
            trade_bucket="interior_finishes", group="bathroom",
            room_surrogate_id="bathroom_1", issue_ids=["i_wallpaper"], photo_keys=["ba1.jpg"],
        )

        assert infer_package_candidates(
            [generic_wallpaper],
            [_surrogate("bathroom_1", "bathroom")],
            catalog,
        ) == []

    def test_package_affinity_normalizes_living_area_aliases(self):
        catalog_path = Path(__file__).resolve().parents[1] / "tools" / "issue_catalog.json"
        shipped_table = build_package_affinity(
            json.loads(catalog_path.read_text(encoding="utf-8"))
        )
        assert package_affinity_for(
            "living_areas", "popcorn_or_acoustic_ceiling_texture", shipped_table
        )["package_type"] == "living_modernization"
        assert package_affinity_for(
            "living_room", "popcorn_or_acoustic_ceiling_texture", shipped_table
        )["package_type"] == "living_modernization"

    # ── Issue 1: group-aware scene -> room normalization ──────────────────────

    def test_normalize_scene_to_room_is_group_aware_with_exclusions(self):
        assert rp._normalize_scene_to_room("dining_room") == "living"
        assert rp._normalize_scene_to_room("home_office") == "living"
        assert rp._normalize_scene_to_room("living_areas") == "living"
        assert rp._normalize_scene_to_room("pantry") == "kitchen"
        # hallway/stairway are transitional and must NOT drive a living package.
        assert rp._normalize_scene_to_room("hallway") != "living"
        assert rp._normalize_scene_to_room("stairway") != "living"

    @staticmethod
    def _carpet_catalog():
        return {"items": [{
            **_cat_item("worn_or_stained_carpet", kind="defect"),
            "display_class": "estimate_driver",
            **_affinity(
                ("bedroom", "bedroom_modernization", "package_driver"),
                ("living", "living_modernization", "package_driver"),
                ("kitchen", "kitchen_modernization", "package_support"),
            ),
            "estimate": {"estimate_tier": "medium", "group": "flooring"},
        }]}

    @pytest.mark.parametrize("surrogate_scene", ["living_room", "dining_room", "home_office"])
    def test_generic_carpet_routes_living_for_living_area_scenes(self, surrogate_scene):
        # Primary path: a breaking living-area surrogate scene collapses to living.
        carpet = _candidate(
            catalog_item_id="worn_or_stained_carpet", kind="defect",
            trade_bucket="flooring", group="living",
            room_surrogate_id="r1", issue_ids=["i_carpet"], photo_keys=["l1.jpg"],
        )
        candidates = infer_package_candidates(
            [carpet], [_surrogate("r1", surrogate_scene)], self._carpet_catalog(),
        )
        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "living_modernization"
        assert candidates[0]["room"] == "living"

    def test_generic_carpet_routes_living_via_scene_group_fallback(self):
        # Fallback path: no room_surrogate_id -> _candidate_scene_room uses the
        # scene_groups_seen UI group ("living_areas"), which must map to living.
        carpet = _candidate(
            catalog_item_id="worn_or_stained_carpet", kind="defect",
            trade_bucket="flooring", group="living",
            room_surrogate_id="", issue_ids=["i_carpet"], photo_keys=["l1.jpg"],
        )
        carpet.scene_groups_seen = ["living_areas"]
        candidates = infer_package_candidates([carpet], [], self._carpet_catalog())
        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "living_modernization"

    # ── Issue 3: generics route via affinity using the stripped real catalog ──

    @staticmethod
    def _real_catalog():
        path = Path(__file__).resolve().parents[1] / "tools" / "issue_catalog.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_generic_supports_route_to_kitchen_modernization_real_catalog(self):
        # Real catalog carries no room package metadata on generics; two supports
        # in a kitchen scene still form a kitchen_modernization package via affinity.
        catalog = self._real_catalog()
        lighting = _candidate(
            catalog_item_id="dated_lighting_fixtures", kind="upgrade",
            trade_bucket="electrical", group="kitchen",
            room_surrogate_id="kitchen_1", issue_ids=["i_light"], photo_keys=["k1.jpg"],
        )
        doors = _candidate(
            catalog_item_id="dated_interior_doors", kind="upgrade",
            trade_bucket="trim_doors_windows", group="kitchen",
            room_surrogate_id="kitchen_1", issue_ids=["i_doors"], photo_keys=["k1.jpg"],
        )
        candidates = infer_package_candidates(
            [lighting, doors], [_surrogate("kitchen_1", "kitchen")], catalog,
        )
        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "kitchen_modernization"
        roles = {
            it["catalog_item_id"]: it["package_role"]
            for it in candidates[0]["evidence_items"]
        }
        # over-bucketing guard: borderline cosmetics are supports, never drivers.
        assert roles["dated_interior_doors"] == "package_support"
        assert roles["dated_lighting_fixtures"] == "package_support"

    def test_generic_gap_supports_route_to_bathroom_modernization_real_catalog(self):
        # Bathroom "gap" generics (no bathroom-specific equivalent) route to
        # bathroom_modernization; flooring/paint/wallpaper defer to specific items.
        catalog = self._real_catalog()
        drywall = _candidate(
            catalog_item_id="damaged_drywall_or_cracks", kind="defect",
            trade_bucket="paint_drywall", group="bathroom",
            room_surrogate_id="bathroom_1", issue_ids=["i_dw"], photo_keys=["b1.jpg"],
        )
        scuffs = _candidate(
            catalog_item_id="wall_scuffs_marks_or_dents", kind="defect",
            trade_bucket="paint_drywall", group="bathroom",
            room_surrogate_id="bathroom_1", issue_ids=["i_sc"], photo_keys=["b1.jpg"],
        )
        candidates = infer_package_candidates(
            [drywall, scuffs], [_surrogate("bathroom_1", "bathroom")], catalog,
        )
        assert len(candidates) == 1
        assert candidates[0]["package_type"] == "bathroom_modernization"


class TestExteriorRepairPackageCandidates:
    """The exterior family. Unlike every interior room, exterior scenes are
    non-breaking and open no per-scene surrogate, so the property carries a
    single `exterior_primary` identity (see room_surrogates)."""

    def _exterior_catalog(self):
        return {
            "items": [
                {
                    **_cat_item("rotted_siding", kind="defect", severity=3),
                    "display_class": "estimate_driver",
                    **_affinity(("exterior", "exterior_repair", "package_driver")),
                    "estimate": {"estimate_tier": "medium", "group": "exterior"},
                },
                {
                    **_cat_item("deck_damage", kind="defect", severity=3),
                    "display_class": "estimate_driver",
                    **_affinity(("exterior", "exterior_repair", "package_driver")),
                    "estimate": {"estimate_tier": "medium", "group": "exterior"},
                },
                {
                    **_cat_item("siding_discoloration", kind="defect", severity=1),
                    "display_class": "marketability",
                    **_affinity(("exterior", "exterior_repair", "package_support")),
                    "estimate": {"estimate_tier": "minor", "group": "exterior"},
                },
            ]
        }

    @staticmethod
    def _ext(cid, *, trade="exterior_siding_trim", sev=3, issue_ids, photo_keys):
        return _candidate(
            catalog_item_id=cid, kind="defect", severity=sev,
            trade_bucket=trade, group="exterior",
            room_surrogate_id="exterior_primary", issue_ids=issue_ids,
            photo_keys=photo_keys,
        )

    def test_exterior_repair_package_inferred(self):
        driver = self._ext("rotted_siding", issue_ids=["i1"], photo_keys=["e1.jpg"])
        support = self._ext("siding_discoloration", sev=1,
                            issue_ids=["i2"], photo_keys=["e2.jpg"])
        candidates = infer_package_candidates(
            [driver, support],
            [_surrogate("exterior_primary", "exterior")],
            self._exterior_catalog(),
        )
        assert len(candidates) == 1
        pkg = candidates[0]
        assert pkg["package_type"] == PACKAGE_TYPE_EXTERIOR_REPAIR
        assert pkg["room"] == ROOM_EXTERIOR
        assert pkg["estimate_group"] == "exterior"
        assert pkg["package_id"] == "exterior_repair__exterior_primary"

    def test_exterior_package_is_repair_not_turnover(self):
        # Turnover packages short-circuit to confirmed_by_rule BEFORE the 2f
        # room-prompt check, which would skip visual verification entirely.
        driver = self._ext("rotted_siding", issue_ids=["i1"], photo_keys=["e1.jpg"])
        candidates = infer_package_candidates(
            [driver], [_surrogate("exterior_primary", "exterior")],
            self._exterior_catalog(),
        )
        assert candidates[0]["package_category"] == PACKAGE_CATEGORY_REPAIR

    def test_drivers_across_exterior_scenes_share_one_package(self):
        # The regression that motivated the property-level surrogate: exterior
        # evidence spread across front/back/side elevations is ONE exterior, so
        # it must corroborate into a single package rather than fragmenting.
        siding = self._ext("rotted_siding", issue_ids=["i1"], photo_keys=["front.jpg"])
        deck = self._ext("deck_damage", issue_ids=["i2"], photo_keys=["back.jpg"])
        stain = self._ext("siding_discoloration", sev=1,
                          issue_ids=["i3"], photo_keys=["side.jpg"])
        candidates = infer_package_candidates(
            [siding, deck, stain],
            [_surrogate("exterior_primary", "exterior")],
            self._exterior_catalog(),
        )
        assert len(candidates) == 1
        assert set(candidates[0]["supporting_issue_ids"]) == {"i1", "i2", "i3"}

    def test_envelope_defect_prices_heavy_and_porch_alone_prices_light(self):
        catalog = self._exterior_catalog()
        surrogates = [_surrogate("exterior_primary", "exterior")]

        siding = self._ext("rotted_siding", issue_ids=["i1"], photo_keys=["e1.jpg"])
        heavy = infer_package_candidates([siding], surrogates, catalog)
        assert heavy[0]["pricing_profile"] == EXTERIOR_REPAIR_HEAVY[0]
        assert (heavy[0]["cost_low"], heavy[0]["cost_high"]) == EXTERIOR_REPAIR_HEAVY[1:]

        # deck/porch is not an envelope component, so it stays light.
        deck = self._ext("deck_damage", issue_ids=["i2"], photo_keys=["e2.jpg"])
        light = infer_package_candidates([deck], surrogates, catalog)
        assert light[0]["pricing_profile"] == EXTERIOR_REPAIR_LIGHT[0]

    def test_exterior_tiers_stay_within_group_budget_cap(self):
        low, high = GROUP_BUDGET_CAPS["exterior"]
        for tier in (EXTERIOR_REPAIR_LIGHT, EXTERIOR_REPAIR_HEAVY):
            assert tier[1] >= low
            assert tier[2] <= high

    def test_exterior_absorption_scopes_registered(self):
        # Missing absorption scopes raise KeyError in reconciliation Phase B0.
        for tier in (EXTERIOR_REPAIR_LIGHT, EXTERIOR_REPAIR_HEAVY):
            scope = _package_absorption_scope(tier[0])
            assert scope["family"] == "exterior"
            # Shared with interior openings — must not be absorbable here.
            assert "trim_doors_windows" not in scope["trade_buckets"]

    def test_classify_component_maps_exterior_trades(self):
        assert classify_component(
            self._ext("rotted_siding", issue_ids=["i"], photo_keys=["p"])) == "siding"
        assert classify_component(
            self._ext("damaged_soffit_or_porch_ceiling",
                      issue_ids=["i"], photo_keys=["p"])) == "deck_porch"
        assert classify_component(
            self._ext("brick_weathering", trade="masonry_exterior_structure",
                      issue_ids=["i"], photo_keys=["p"])) == "masonry"


class TestExteriorRepairV4Integration:
    """Full-pipeline coverage for the exterior family.

    This must go through `compute_renovation_estimate_v4`, not
    `infer_package_candidates` directly. The seam being guarded lives in the
    stamping loop: exterior issues only share a bucket because the property-level
    surrogate gives them an `estimate_unit_id`, and that id has to survive
    `renovation_estimate._meaningful_scope_hint`. A direct call bypasses both and
    would pass even if the id were silently discarded.
    """

    CATALOG = {
        "items": [
            {
                **_cat_item("rotted_siding", kind="defect", severity=3),
                "trade_bucket": "exterior_siding_trim",
                "display_class": "estimate_driver",
                **_affinity(("exterior", "exterior_repair", "package_driver")),
                "estimate": {"estimate_tier": "medium", "strategy": "repair_or_replace",
                             "group": "exterior", "stack_behavior": "sum"},
                "cost": {"mode": "heuristic"},
            },
            {
                **_cat_item("deck_damage", kind="defect", severity=3),
                "trade_bucket": "exterior_siding_trim",
                "display_class": "estimate_driver",
                **_affinity(("exterior", "exterior_repair", "package_driver")),
                "cost": {"mode": "heuristic"},
            },
            {
                **_cat_item("siding_discoloration", kind="defect", severity=1),
                "trade_bucket": "exterior_siding_trim",
                "display_class": "marketability",
                **_affinity(("exterior", "exterior_repair", "package_support")),
                "cost": {"mode": "heuristic"},
            },
        ]
    }

    ISSUES = [
        {"issue_id": "i_siding", "catalog_item_id": "rotted_siding",
         "catalog_item_kind": "defect", "scene_group": "exterior",
         "photo_key": "front.jpg", "description": "x", "label": "defect_or_damage"},
        {"issue_id": "i_deck", "catalog_item_id": "deck_damage",
         "catalog_item_kind": "defect", "scene_group": "exterior",
         "photo_key": "back.jpg", "description": "x", "label": "defect_or_damage"},
        {"issue_id": "i_stain", "catalog_item_id": "siding_discoloration",
         "catalog_item_kind": "defect", "scene_group": "exterior",
         "photo_key": "side.jpg", "description": "x", "label": "defect_or_damage"},
    ]

    PHOTOS = {
        "front.jpg": {"scene": {"id": "exterior_front"}, "photo": {"index": 1}},
        "back.jpg": {"scene": {"id": "exterior_back"}, "photo": {"index": 2}},
        "side.jpg": {"scene": {"id": "exterior_side"}, "photo": {"index": 3}},
    }

    def _v4(self):
        return compute_renovation_estimate_v4(
            issues_flat=self.ISSUES,
            issue_catalog=self.CATALOG,
            photos=self.PHOTOS,
        )

    def test_emits_exactly_one_exterior_package_candidate(self):
        v4 = self._v4()
        exterior = [
            p for p in v4["package_candidates"]
            if p["package_type"] == PACKAGE_TYPE_EXTERIOR_REPAIR
        ]
        assert len(exterior) == 1, (
            "exterior evidence fragmented into separate packages — the "
            "estimate_unit_id stamp did not survive"
        )
        assert exterior[0]["package_id"] == "exterior_repair__exterior_primary"
        assert exterior[0]["room"] == ROOM_EXTERIOR
        assert set(exterior[0]["supporting_issue_ids"]) == {"i_siding", "i_deck", "i_stain"}

    def test_exterior_estimate_unit_id_survives_scope_hint_filter(self):
        # "exterior" is a member of renovation_estimate._GENERIC_SCOPE_HINTS and
        # would be stripped to ""; the surrogate id must not be that token.
        v4 = self._v4()
        assert "exterior_primary" in [u["estimate_unit_id"] for u in v4["estimate_units"]]
        assert _meaningful_scope_hint("exterior_primary") == "exterior_primary"
        assert _meaningful_scope_hint("exterior") == ""

    def test_all_exterior_photos_map_to_the_one_unit(self):
        v4 = self._v4()
        mapping = v4["photo_to_estimate_unit_id"]
        assert set(mapping) == set(self.PHOTOS)
        assert set(mapping.values()) == {"exterior_primary"}

    def test_package_is_2f_eligible_and_not_rule_confirmed(self):
        # No verifications supplied, so the package must sit at not_run and be
        # withheld from the estimate — never auto-confirmed like turnover.
        v4 = self._v4()
        pkg = next(p for p in v4["package_candidates"]
                   if p["package_type"] == PACKAGE_TYPE_EXTERIOR_REPAIR)
        assert pkg["verification_status"] == PACKAGE_VERIFICATION_NOT_RUN
        assert pkg["estimate_eligible"] is False
        assert v4["packages"] == []
        assert pkg["room"] in PASS_2F_ROOM_PROMPTS, "no Pass 2f prompt registered"

    def test_confirmed_exterior_package_reaches_the_estimate(self):
        v4 = compute_renovation_estimate_v4(
            issues_flat=self.ISSUES,
            issue_catalog=self.CATALOG,
            photos=self.PHOTOS,
            package_verifications={
                "exterior_repair__exterior_primary": {
                    "verification_status": "confirmed",
                    "confirmed_issue_ids": ["i_siding", "i_deck", "i_stain"],
                    "evidence_summary": "Visible rot and deck damage.",
                }
            },
        )
        pkg = next(p for p in v4["packages"]
                   if p["package_type"] == PACKAGE_TYPE_EXTERIOR_REPAIR)
        assert pkg["verification_status"] == "confirmed"
        assert pkg["estimate_eligible"] is True
        assert pkg["audit_only"] is False
        assert pkg["cost_high"] > 0

    def test_review_photos_cover_both_drivers(self):
        v4 = self._v4()
        pkg = next(p for p in v4["package_candidates"]
                   if p["package_type"] == PACKAGE_TYPE_EXTERIOR_REPAIR)
        # Only 3 evidence items and 3 slots, so every one gets reviewed — the
        # point is that no single item monopolizes the sample.
        assert set(pkg["review_photo_keys"]) == {"front.jpg", "back.jpg", "side.jpg"}


class TestReviewPhotoCoverage:
    """Review photos must span the package's distinct evidence, not just the
    first three frames — an unreviewed driver can never be confirmed."""

    @staticmethod
    def _c(cid, issue_ids, photo_keys):
        return _candidate(
            catalog_item_id=cid, kind="defect", severity=3,
            trade_bucket="exterior_siding_trim", group="exterior",
            room_surrogate_id="exterior_primary",
            issue_ids=issue_ids, photo_keys=photo_keys,
        )

    def test_selects_one_photo_per_distinct_evidence_item(self):
        # The dominant item owns the first three photos in listing order; the
        # rarer driver still has to make the cut.
        common = self._c("rotted_siding", ["i1", "i2", "i3"],
                         ["p1.jpg", "p2.jpg", "p3.jpg"])
        rare = self._c("deck_damage", ["i4"], ["p9.jpg"])
        picked = rp._select_review_photo_keys(
            [common, rare], [], ["p1.jpg", "p2.jpg", "p3.jpg", "p9.jpg"],
        )
        assert "p9.jpg" in picked
        assert picked[0] == "p1.jpg"
        assert len(picked) == 3

    def test_backfills_when_fewer_evidence_items_than_slots(self):
        only = self._c("rotted_siding", ["i1", "i2"], ["p1.jpg", "p2.jpg"])
        picked = rp._select_review_photo_keys([only], [], ["p1.jpg", "p2.jpg"])
        assert picked == ["p1.jpg", "p2.jpg"]

    def test_is_deterministic_and_respects_limit(self):
        drivers = [self._c(f"d{i}", [f"i{i}"], [f"p{i}.jpg"]) for i in range(5)]
        photo_keys = [f"p{i}.jpg" for i in range(5)]
        first = rp._select_review_photo_keys(drivers, [], photo_keys)
        assert first == rp._select_review_photo_keys(drivers, [], photo_keys)
        assert len(first) == rp.PACKAGE_REVIEW_IMAGE_LIMIT

    def test_drivers_are_offered_a_slot_before_supports(self):
        support = self._c("siding_discoloration", ["s1"], ["s1.jpg"])
        driver = self._c("deck_damage", ["d1"], ["d1.jpg"])
        picked = rp._select_review_photo_keys(
            [driver], [support], ["s1.jpg", "d1.jpg"],
        )
        assert picked[0] == "d1.jpg"


class TestRepairProfileSeverityGate:
    """Issue 2: a single low-severity defect must not spawn a heavy repair tier."""

    @staticmethod
    def _drv(cid, trade, sev):
        return _candidate(
            catalog_item_id=cid, kind="defect", trade_bucket=trade, severity=sev,
            room_surrogate_id="bedroom_1", issue_ids=[f"i_{cid}"],
        )

    def test_single_low_sev_flooring_driver_stays_light(self):
        d = self._drv("scratched_or_damaged_flooring", "flooring", 1)
        spec, label, _notes = rp._resolve_room_repair_profile(
            rp.BEDROOM_REPAIR_LIGHT, rp.BEDROOM_REPAIR_HEAVY, [d], [d], [])
        assert label == "repair_light"
        assert spec == rp.BEDROOM_REPAIR_LIGHT

    def test_severe_moisture_driver_goes_heavy(self):
        d = self._drv("water_stain_ceiling", "moisture_mold", 3)
        spec, label, _notes = rp._resolve_room_repair_profile(
            rp.BEDROOM_REPAIR_LIGHT, rp.BEDROOM_REPAIR_HEAVY, [d], [d], [])
        assert label == "repair_heavy"
        assert spec == rp.BEDROOM_REPAIR_HEAVY

    def test_two_distinct_heavy_components_go_heavy(self):
        d1 = self._drv("scratched_or_damaged_flooring", "flooring", 1)
        d2 = self._drv("water_stain_ceiling", "moisture_mold", 1)
        _spec, label, _notes = rp._resolve_room_repair_profile(
            rp.BEDROOM_REPAIR_LIGHT, rp.BEDROOM_REPAIR_HEAVY, [d1, d2], [d1, d2], [])
        assert label == "repair_heavy"

    def test_two_distinct_low_sev_drivers_go_heavy(self):
        d1 = self._drv("scratched_or_damaged_flooring", "flooring", 1)
        d2 = self._drv("worn_or_stained_carpet", "flooring", 1)
        _spec, label, _notes = rp._resolve_room_repair_profile(
            rp.BEDROOM_REPAIR_LIGHT, rp.BEDROOM_REPAIR_HEAVY, [d1, d2], [d1, d2], [])
        assert label == "repair_heavy"

    def test_kitchen_single_low_sev_flooring_driver_stays_light(self):
        d = self._drv("scratched_or_damaged_flooring", "flooring", 1)
        spec, label, _notes = rp._resolve_kitchen_repair_profile([d], [d], [])
        assert label == "repair_light"
        assert spec == rp.KITCHEN_REPAIR_LIGHT

    def test_bathroom_single_low_sev_flooring_driver_stays_light(self):
        d = self._drv("scratched_or_damaged_flooring", "flooring", 1)
        spec, label, _notes = rp._resolve_bathroom_repair_profile([d], [d], [])
        assert label == "repair_light"
        assert spec == rp.BATHROOM_REPAIR_LIGHT


# ═══════════════════════════════════════════════════════════════════════════
# Merged units price as ONE room — repeated surrogates are re-shoots of the
# same physical room, never a multi-room multiplier
# ═══════════════════════════════════════════════════════════════════════════


class TestMergedUnitPricing:

    def _catalog(self):
        return {
            "items": [
                {
                    **_cat_item("outdated_kitchen_finishes", kind="upgrade", category="opportunity"),
                    "display_class": "marketability",
                    "package_role": "package_driver",
                    "package_type": "kitchen_modernization",
                    "estimate": {"estimate_tier": "high", "group": "kitchen"},
                },
            ]
        }

    def _infer(self, estimate_units=None):
        c = _candidate(
            catalog_item_id="outdated_kitchen_finishes",
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            group="kitchen",
            room_surrogate_id="kitchen_1",
            issue_ids=["i_outdated"],
            photo_keys=["k1.jpg", "k2.jpg"],
        )
        candidates = infer_package_candidates(
            [c],
            [_surrogate("kitchen_1", "kitchen")],
            self._catalog(),
            estimate_units=estimate_units,
        )
        assert len(candidates) == 1
        return candidates[0]

    def _merged_unit(self, n):
        return [{
            "estimate_unit_id": "kitchen_1",
            "source_room_surrogate_ids": [f"kitchen_{i}" for i in range(1, n + 1)],
        }]

    def test_single_surrogate_prices_at_tier_spec(self):
        pkg = self._infer()
        assert "unit_count_factor" not in pkg
        assert "pre_unit_factor_cost_low" not in pkg
        assert not any("unit_count_factor" in n for n in pkg["level_decision_notes"])

    def test_merged_surrogates_price_as_one_room(self):
        base = self._infer()
        pkg = self._infer(estimate_units=self._merged_unit(3))
        assert pkg["cost_low"] == base["cost_low"]
        assert pkg["cost_high"] == base["cost_high"]
        assert pkg["cost_midpoint"] == base["cost_midpoint"]
        assert pkg["candidate_cost_low"] == base["cost_low"]
        assert pkg["candidate_cost_high"] == base["cost_high"]
        assert "unit_count_factor" not in pkg
        assert "pre_unit_factor_cost_low" not in pkg
        assert not any("unit_count_factor" in n for n in pkg["level_decision_notes"])


class TestExpansionKeepsTierCost:

    def test_expanded_copies_keep_tier_cost(self):
        pkg = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i2"],
            refs=[
                _ref("i1", "bathroom_1"),
                _ref("i2", "bathroom_2"),
            ],
        )
        out, audit = expand_bathroom_modernization_packages(
            [pkg],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=2,
        )
        assert audit["expanded"] is True
        assert len(out) == 2
        for expanded in out:
            assert expanded["cost_low"] == 12000
            assert expanded["cost_high"] == 18000


# ═══════════════════════════════════════════════════════════════════════════
# Same-unit package subsumption
# ═══════════════════════════════════════════════════════════════════════════

from tools.rehab_packages import (
    apply_physical_package_subsumption,
    apply_same_unit_package_subsumption,
    dedupe_same_unit_modernizations,
)


_DEFAULT_EVIDENCE_TRADE_BY_ROOM = {
    "bathroom": "bathroom_fixtures_tile",
    "kitchen": "kitchen_cabinets_counters",
}


def _unit_pkg(package_type, category, tier, *, room="bathroom",
              unit="bathroom_primary", surrogate="bathroom_1",
              cost_low=2_000, cost_high=8_000, status="confirmed",
              level="room", supporting_issue_ids=None, package_id=None,
              confirmed_issue_ids=None, confirmed_surrogates=None,
              rejected_issue_ids=None, evidence_trade_bucket=None):
    """Active room-level package dict carrying the fields the subsumption
    pass and reconciliation Phase B read. pricing_profile derives as
    f"{room}_{tier}" (e.g. bathroom + partial_rehab → bathroom_partial_rehab).

    Physical subsumption decides on confirmed surrogate sets, so the fixture
    carries evidence_items with per-surrogate issue_refs: confirmed_surrogates
    defaults to [surrogate], confirmed_issue_ids defaults to the supporting
    ids (or one generated id), and the evidence trade bucket defaults to one
    the room's modernization absorption scope covers.
    """
    profile = f"{room}_{tier}"
    pid = package_id or f"{package_type}__{unit}"
    supporting = list(supporting_issue_ids or [])
    if confirmed_issue_ids is None:
        confirmed_issue_ids = supporting or [f"{pid}_i1"]
    confirmed_issue_ids = list(confirmed_issue_ids)
    if not supporting:
        supporting = list(confirmed_issue_ids)
    surrogates = list(confirmed_surrogates or [surrogate])
    trade = evidence_trade_bucket or _DEFAULT_EVIDENCE_TRADE_BY_ROOM.get(
        room, "paint_drywall"
    )
    refs = [
        {
            "issue_id": issue_id,
            "photo_key": "",
            "observation": "",
            "room_surrogate_id": sid,
        }
        for issue_id in confirmed_issue_ids
        for sid in surrogates
    ]
    evidence_items = [{
        "catalog_item_id": f"{pid}_evidence",
        "name": "stub evidence",
        "issue_ids": list(confirmed_issue_ids),
        "issue_refs": refs,
        "trade_bucket": trade,
        "photo_keys": [],
        "supporting_photo_count": 0,
        "room_surrogate_id": surrogate,
        "estimate_unit_id": unit,
    }] if refs else []
    return {
        "package_id": pid,
        "package_type": package_type,
        "package_category": category,
        "room": room,
        "package_level": level,
        "pricing_tier": tier,
        "pricing_profile": profile,
        "room_surrogate_id": surrogate,
        "estimate_unit_id": unit,
        "source_room_surrogate_ids": surrogates,
        "estimate_group": room,
        "estimate_scope": "required_rehab" if category == "repair" else "marketability_rehab",
        "cost_low": cost_low,
        "cost_high": cost_high,
        "cost_midpoint": (cost_low + cost_high) // 2,
        "candidate_cost_low": cost_low,
        "candidate_cost_high": cost_high,
        "absorption_scope": _package_absorption_scope(profile),
        "cap_behavior": "respect_group_cap",
        "absorbed_unit_member_refs": [],
        "absorbed_total_low": 0,
        "absorbed_total_high": 0,
        "replacement_delta_low": 0,
        "replacement_delta_high": 0,
        "supporting_issue_ids": supporting,
        "supporting_catalog_item_ids": [],
        "confirmed_issue_ids": confirmed_issue_ids,
        "rejected_issue_ids": list(rejected_issue_ids or []),
        "evidence_items": evidence_items,
        "trigger_reason": "stub",
        "level_decision_notes": [],
        "verification_status": status,
        "estimate_eligible": True,
        "ui_eligible": True,
        "audit_only": False,
    }


class TestSameUnitSubsumption:

    @pytest.mark.parametrize("tier", ["partial_rehab", "full_rehab"])
    def test_modernization_partial_or_full_subsumes_repair(self, tier):
        modern = _unit_pkg("bathroom_modernization", "modernization", tier)
        repair = _unit_pkg("bathroom_repair", "repair", "repair_heavy")
        active, audit = apply_same_unit_package_subsumption([modern, repair])
        assert active == [modern]
        assert repair["estimate_eligible"] is False
        assert repair["ui_eligible"] is False
        assert repair["audit_only"] is True
        assert repair["subsumed_by_package_id"] == modern["package_id"]
        # Eligibility flags are the gate — "confirmed" still means the VLM said yes.
        assert repair["verification_status"] == "confirmed"
        assert "subsumed_by_same_unit_modernization" in repair["level_decision_notes"]
        assert modern["subsumed_package_ids"] == [repair["package_id"]]
        assert repair["subsumed_by_package_ids"] == [modern["package_id"]]
        assert audit["applied"] is True
        record = audit["subsumptions"][0]
        assert record["unit"] == "bathroom_primary"
        assert record["winner_package_id"] == modern["package_id"]
        assert record["loser_package_id"] == repair["package_id"]
        assert record["rule"] == "modernization_subsumes_repair"
        assert record["loser_confirmed_surrogates"] == ["bathroom_1"]
        assert record["winner_confirmed_surrogates"] == ["bathroom_1"]
        assert record["surrogate_basis"] == "confirmed_refs"
        assert record["component_coverage"] == "covered"
        assert record["winner_cost"] == {
            "low": modern["cost_low"], "high": modern["cost_high"],
        }
        assert record["loser_cost"] == {
            "low": repair["cost_low"], "high": repair["cost_high"],
        }

    def test_refresh_modernization_does_not_subsume_repair(self):
        modern = _unit_pkg("bathroom_modernization", "modernization", "refresh")
        repair = _unit_pkg("bathroom_repair", "repair", "repair_light")
        active, audit = apply_same_unit_package_subsumption([modern, repair])
        assert active == [modern, repair]
        assert repair["estimate_eligible"] is True
        assert "subsumed_by_package_id" not in repair
        assert audit["applied"] is False
        assert audit["subsumptions"] == []

    @pytest.mark.parametrize("tier", ["refresh", "partial_rehab", "full_rehab"])
    def test_modernization_any_tier_suppresses_turnover(self, tier):
        modern = _unit_pkg("bathroom_modernization", "modernization", tier)
        turnover = _unit_pkg("bathroom_turnover", "turnover", "turnover_std",
                             status="confirmed_by_rule")
        active, audit = apply_same_unit_package_subsumption([modern, turnover])
        assert active == [modern]
        assert turnover["audit_only"] is True
        assert turnover["suppressed_by_package_id"] == modern["package_id"]
        assert turnover["verification_status"] == "confirmed_by_rule"
        assert "suppressed_by_same_unit_modernization" in turnover["level_decision_notes"]
        assert modern["suppressed_package_ids"] == [turnover["package_id"]]
        assert audit["subsumptions"][0]["rule"] == "modernization_suppresses_turnover"

    def test_repair_light_plus_turnover_both_survive_unchanged(self):
        repair = _unit_pkg("bathroom_repair", "repair", "repair_light")
        turnover = _unit_pkg("bathroom_turnover", "turnover", "turnover_std",
                             status="confirmed_by_rule")
        active, audit = apply_same_unit_package_subsumption([repair, turnover])
        assert active == [repair, turnover]
        assert turnover["pricing_profile"] == "bathroom_turnover_std"
        assert audit["applied"] is False

    def test_repair_heavy_downgrades_turnover_std_to_light(self):
        repair = _unit_pkg("bathroom_repair", "repair", "repair_heavy")
        turnover = _unit_pkg("bathroom_turnover", "turnover", "turnover_std",
                             status="confirmed_by_rule",
                             cost_low=1_800, cost_high=4_500)
        active, audit = apply_same_unit_package_subsumption([repair, turnover])
        assert active == [repair, turnover]
        assert turnover["estimate_eligible"] is True
        assert turnover["pricing_profile"] == "bathroom_turnover_light"
        assert turnover["pricing_tier"] == "turnover_light"
        assert turnover["cost_low"] == rp.BATHROOM_TURNOVER_LIGHT[1]
        assert turnover["cost_high"] == rp.BATHROOM_TURNOVER_LIGHT[2]
        assert turnover["cost_midpoint"] == (
            rp.BATHROOM_TURNOVER_LIGHT[1] + rp.BATHROOM_TURNOVER_LIGHT[2]
        ) // 2
        assert turnover["absorption_scope"] == _package_absorption_scope("bathroom_turnover_light")
        assert "downgraded_to_light_by_same_unit_repair_heavy" in turnover["level_decision_notes"]
        assert audit["applied"] is True
        assert audit["subsumptions"] == []
        assert audit["downgrades"] == [{
            "unit": "bathroom_primary",
            "turnover_package_id": turnover["package_id"],
            "repair_package_id": repair["package_id"],
            "from_profile": "bathroom_turnover_std",
            "to_profile": "bathroom_turnover_light",
        }]

    def test_repair_heavy_leaves_turnover_light_alone(self):
        repair = _unit_pkg("bathroom_repair", "repair", "repair_heavy")
        turnover = _unit_pkg("bathroom_turnover", "turnover", "turnover_light",
                             status="confirmed_by_rule")
        active, audit = apply_same_unit_package_subsumption([repair, turnover])
        assert active == [repair, turnover]
        assert turnover["pricing_profile"] == "bathroom_turnover_light"
        assert audit["applied"] is False

    def test_cross_unit_packages_untouched(self):
        modern = _unit_pkg("bathroom_modernization", "modernization", "full_rehab")
        repair = _unit_pkg("bathroom_repair", "repair", "repair_heavy",
                           unit="bathroom_secondary", surrogate="bathroom_2")
        turnover = _unit_pkg("kitchen_turnover", "turnover", "turnover_std",
                             room="kitchen", unit="kitchen_primary",
                             surrogate="kitchen_1", status="confirmed_by_rule")
        active, audit = apply_same_unit_package_subsumption([modern, repair, turnover])
        assert active == [modern, repair, turnover]
        assert audit["applied"] is False

    def test_surrogate_fallback_when_unit_id_empty(self):
        modern = _unit_pkg("bathroom_modernization", "modernization", "full_rehab",
                           unit="", surrogate="bathroom_1",
                           package_id="bm_surrogate")
        repair = _unit_pkg("bathroom_repair", "repair", "repair_heavy",
                           unit="", surrogate="bathroom_1",
                           package_id="br_surrogate")
        active, audit = apply_same_unit_package_subsumption([modern, repair])
        assert active == [modern]
        assert audit["subsumptions"][0]["unit"] == "bathroom_1"

    def test_inactive_modernization_does_not_subsume(self):
        modern = _unit_pkg("bathroom_modernization", "modernization", "full_rehab",
                           status="rejected")
        repair = _unit_pkg("bathroom_repair", "repair", "repair_heavy")
        active, audit = apply_same_unit_package_subsumption([modern, repair])
        assert active == [modern, repair]
        assert audit["applied"] is False

    def test_multiple_modernizations_deduped_then_repairs_subsumed(self):
        # Duplicate same-unit modernizations are no longer both billable: the
        # dedupe pre-pass keeps the highest tier, then physical subsumption
        # drops the covered repairs.
        refresh = _unit_pkg("bathroom_modernization", "modernization", "refresh",
                            package_id="bm_refresh")
        full = _unit_pkg("bathroom_modernization", "modernization", "full_rehab",
                         package_id="bm_full")
        repair_a = _unit_pkg("bathroom_repair", "repair", "repair_light",
                             package_id="br_a")
        repair_b = _unit_pkg("bathroom_repair", "repair", "repair_heavy",
                             package_id="br_b")
        deduped, dedupe_records = dedupe_same_unit_modernizations(
            [refresh, full, repair_a, repair_b]
        )
        assert [p["package_id"] for p in deduped] == ["bm_full", "br_a", "br_b"]
        assert refresh["audit_only"] is True
        assert refresh["subsumed_by_package_id"] == "bm_full"
        assert "deduped_same_unit_modernization" in refresh["level_decision_notes"]
        assert dedupe_records == [{
            "unit": "bathroom_primary",
            "winner_package_id": "bm_full",
            "loser_package_id": "bm_refresh",
            "rule": "modernization_dedupe_highest_tier",
        }]
        active, audit = apply_physical_package_subsumption(deduped)
        assert [p["package_id"] for p in active] == ["bm_full"]
        assert repair_a["subsumed_by_package_id"] == "bm_full"
        assert repair_b["subsumed_by_package_id"] == "bm_full"
        assert full["subsumed_package_ids"] == ["bm_refresh", "br_a", "br_b"]
        assert len(audit["subsumptions"]) == 2
        # Idempotence: reapplying over the aliased list (dropped dicts still
        # present) changes nothing and duplicates no lineage.
        active_again, _ = apply_physical_package_subsumption(deduped)
        assert [p["package_id"] for p in active_again] == ["bm_full"]
        assert full["subsumed_package_ids"] == ["bm_refresh", "br_a", "br_b"]
        assert repair_a["level_decision_notes"].count(
            "subsumed_by_same_unit_modernization"
        ) == 1

    def test_suppressed_turnover_breaks_whole_home_aggregation(self):
        modern = _unit_pkg("bathroom_modernization", "modernization", "refresh")
        bath_turn = _unit_pkg("bathroom_turnover", "turnover", "turnover_std",
                              status="confirmed_by_rule")
        kitchen_turn = _unit_pkg("kitchen_turnover", "turnover", "turnover_std",
                                 room="kitchen", unit="kitchen_primary",
                                 surrogate="kitchen_1", status="confirmed_by_rule")
        # Without subsumption two distinct rooms would aggregate.
        assert aggregate_whole_home_turnover([modern, bath_turn, kitchen_turn]) is not None
        active, _ = apply_same_unit_package_subsumption([modern, bath_turn, kitchen_turn])
        # Bathroom turnover suppressed → only kitchen remains → no aggregate.
        assert aggregate_whole_home_turnover(active) is None

    def test_uncovered_component_retains_repair_package(self):
        # The repair's confirmed evidence includes plumbing, which no
        # modernization scope covers — the whole repair bundle is retained
        # (conservative gate), its children absorb into IT, and required_rehab
        # keeps the full repair bundle range.
        modern = _unit_pkg("bathroom_modernization", "modernization", "partial_rehab",
                           cost_low=8_000, cost_high=20_000,
                           supporting_issue_ids=["m1"])
        repair = _unit_pkg("bathroom_repair", "repair", "repair_heavy",
                           supporting_issue_ids=["r_plumb", "r_tile"])
        repair["evidence_items"] = [
            {
                "catalog_item_id": "leaking_supply_line",
                "issue_ids": ["r_plumb"],
                "issue_refs": [_ref("r_plumb", "bathroom_1")],
                "trade_bucket": "plumbing",
            },
            {
                "catalog_item_id": "cracked_tile",
                "issue_ids": ["r_tile"],
                "issue_refs": [_ref("r_tile", "bathroom_1")],
                "trade_bucket": "bathroom_fixtures_tile",
            },
        ]
        groups = [
            _group("plumbing", [
                _line_item("leaking_supply_line:bathroom_primary", 600, 1_800,
                           catalog_item_id="leaking_supply_line",
                           trade_bucket="plumbing",
                           room_surrogate_id="bathroom_1",
                           unit_members=[_member("plumb", "bathroom_1", ["r_plumb"],
                                                 estimate_unit_id="bathroom_primary")]),
            ]),
            _group("bathroom", [
                _line_item("cracked_tile:bathroom_primary", 400, 1_200,
                           catalog_item_id="cracked_tile",
                           trade_bucket="bathroom_fixtures_tile",
                           room_surrogate_id="bathroom_1",
                           unit_members=[_member("tile", "bathroom_1", ["r_tile"],
                                                 estimate_unit_id="bathroom_primary")]),
            ]),
        ]
        active, audit = apply_physical_package_subsumption([modern, repair])
        assert [p["package_id"] for p in active] == [
            modern["package_id"], repair["package_id"],
        ]
        retained = audit["retained_repairs"][0]
        assert retained["retained_package_id"] == repair["package_id"]
        assert retained["retention_reason"] == "components_not_covered"
        assert "leaking_supply_line" in retained["uncovered_components"]
        assert repair["estimate_eligible"] is True

        result = reconcile_packages_and_estimate_units(groups, active)
        plumb_child = groups[0]["line_items"][0]["unit_member_allocations"][0]
        tile_child = groups[1]["line_items"][0]["unit_member_allocations"][0]
        # Direct confirmed evidence: both children belong to the repair.
        assert plumb_child["absorbed_by_package_id"] == repair["package_id"]
        assert tile_child["absorbed_by_package_id"] == repair["package_id"]
        # required_rehab carries the retained repair bundle's full range.
        required_raw = result["totals_by_scope_raw"]["required_rehab"]
        assert required_raw["low"] == repair["cost_low"]
        assert required_raw["high"] == repair["cost_high"]
        assert result["package_total_low"] == 8_000 + repair["cost_low"]
        assert result["package_total_high"] == 20_000 + repair["cost_high"]

    def test_covered_repair_subsumed_and_required_work_survives(self):
        # Fully covered repair (tile evidence only) IS subsumed, but required
        # work stays priced: the unrelated plumbing line item is retained, and
        # the required-scope tile child is never absorbed into the
        # marketability modernization (required-scope guard).
        modern = _unit_pkg("bathroom_modernization", "modernization", "partial_rehab",
                           cost_low=8_000, cost_high=20_000,
                           supporting_issue_ids=["m1"])
        repair = _unit_pkg("bathroom_repair", "repair", "repair_heavy",
                           supporting_issue_ids=["r_tile"])
        groups = [
            _group("plumbing", [
                _line_item("leaking_supply_line:bathroom_primary", 600, 1_800,
                           catalog_item_id="leaking_supply_line",
                           trade_bucket="plumbing",
                           room_surrogate_id="bathroom_1",
                           unit_members=[_member("plumb", "bathroom_1", ["u_plumb"],
                                                 estimate_unit_id="bathroom_primary")]),
            ]),
            _group("bathroom", [
                _line_item("cracked_tile:bathroom_primary", 400, 1_200,
                           catalog_item_id="cracked_tile",
                           trade_bucket="bathroom_fixtures_tile",
                           room_surrogate_id="bathroom_1",
                           unit_members=[_member("tile", "bathroom_1", ["r_tile"],
                                                 estimate_unit_id="bathroom_primary")]),
            ]),
        ]
        active, audit = apply_physical_package_subsumption([modern, repair])
        assert [p["package_id"] for p in active] == [modern["package_id"]]
        assert audit["subsumptions"][0]["rule"] == "modernization_subsumes_repair"
        result = reconcile_packages_and_estimate_units(groups, active)

        plumb_child = groups[0]["line_items"][0]["unit_member_allocations"][0]
        tile_child = groups[1]["line_items"][0]["unit_member_allocations"][0]
        # Required-scope children are never absorbed into the marketability
        # modernization — both survive as retained line items in required_rehab.
        assert plumb_child["absorbed_by_package_id"] is None
        assert tile_child["absorbed_by_package_id"] is None
        required_raw = result["totals_by_scope_raw"]["required_rehab"]
        assert required_raw["low"] == 600 + 400
        assert required_raw["high"] == 1_800 + 1_200
        # Package totals exclude the dropped repair range.
        assert result["package_total_low"] == 8_000
        assert result["package_total_high"] == 20_000

    def test_expansion_then_physical_subsumption(self):
        # New pipeline order: expand first, then subsume on physical rooms.
        # The repair confirmed only in bathroom_1 is subsumed by exactly the
        # bathroom_1 clone; a repair confirmed in an unexpanded room would be
        # retained (see test_repair_in_uncovered_room_is_retained).
        modern = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i2"],
            refs=[_ref("i1", "bathroom_1"), _ref("i2", "bathroom_2")],
        )
        modern["package_level"] = "room"
        modern["pricing_tier"] = "partial_rehab"
        modern["pricing_profile"] = "bathroom_partial_rehab"
        repair = _unit_pkg("bathroom_repair", "repair", "repair_heavy",
                           confirmed_surrogates=["bathroom_1"])
        out, exp_audit = expand_bathroom_modernization_packages(
            [modern, repair],
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=2,
        )
        assert exp_audit["expanded"] is True
        assert {
            p["room_surrogate_id"] for p in out
            if p.get("expansion_source_package_id")
        } == {"bathroom_1", "bathroom_2"}
        active, audit = apply_physical_package_subsumption(out)
        clone_b1 = next(
            p for p in out
            if p.get("expansion_source_package_id")
            and p["room_surrogate_id"] == "bathroom_1"
        )
        assert repair["subsumed_by_package_id"] == clone_b1["package_id"]
        assert repair not in active
        assert audit["subsumptions"][0]["surrogate_winners"] == {
            "bathroom_1": clone_b1["package_id"],
        }

    def test_repair_in_uncovered_room_is_retained(self):
        # redfin_126514011 regression: repair confirmed in bathroom_5, a room
        # no modernization covers — the repair must survive subsumption.
        modern = _bathroom_modernization_package(
            confirmed_issue_ids=["i1", "i3"],
            refs=[_ref("i1", "bathroom_1"), _ref("i3", "bathroom_3")],
        )
        modern["package_level"] = "room"
        modern["pricing_tier"] = "full_rehab"
        modern["pricing_profile"] = "bathroom_full_rehab"
        repair = _unit_pkg("bathroom_repair", "repair", "repair_heavy",
                           supporting_issue_ids=["r3", "r5"],
                           confirmed_surrogates=["bathroom_3", "bathroom_5"])
        active, audit = apply_physical_package_subsumption([modern, repair])
        assert repair in active
        assert repair["estimate_eligible"] is True
        retained = audit["retained_repairs"][0]
        assert retained["retention_reason"] == "loser_surrogates_not_covered"
        assert retained["uncovered_surrogates"] == ["bathroom_5"]
        assert retained["loser_confirmed_surrogates"] == ["bathroom_3", "bathroom_5"]

    def test_v4_pipeline_emits_subsumption_audit(self):
        v4 = compute_renovation_estimate_v4(
            issues_flat=[],
            issue_catalog={"items": []},
            photos={},
            package_verifications={},
        )
        audit = v4["package_subsumption_audit"]
        assert audit["applied"] is False
        assert audit["subsumptions"] == []
        assert audit["downgrades"] == []
        assert "tradeoff_note" in audit
        assert "package_subsumption" in v4["provenance"]["v4_phases_applied"]


# ─── Post-2f re-tier, hygiene, expansion re-price, caps, disposition ─────────

from tools.rehab_packages import (
    _apply_post_package_group_cap,
    _recompute_retained_group_detailed,
    apply_package_verifications_to_candidates,
    build_issue_disposition_audit,
)


_BATHROOM_AFFINITY_CATALOG = {
    "items": [
        {
            **_cat_item("outdated_bathroom_finishes", kind="upgrade",
                        category="opportunity", severity=3),
            "display_class": "marketability",
            "package_role": "package_driver",
            "package_type": "bathroom_modernization",
            "estimate": {"estimate_tier": "high", "group": "bathroom"},
        },
        {
            **_cat_item("vintage_tile_pattern_style", kind="upgrade",
                        category="opportunity"),
            "display_class": "marketability",
            "package_role": "package_support",
            "package_type": "bathroom_modernization",
            "estimate": {"estimate_tier": "medium", "group": "bathroom"},
        },
    ]
}


def _bathroom_evidence_candidate(catalog_item_id, refs, *, severity=3):
    """refs: list of (issue_id, surrogate_id) pairs, all on one candidate."""
    c = _candidate(
        catalog_item_id=catalog_item_id,
        kind="upgrade",
        severity=severity,
        trade_bucket="bathroom_fixtures_tile",
        group="bathroom",
        room_surrogate_id=refs[0][1],
        issue_ids=[i for i, _ in refs],
        photo_keys=[f"{s}_{i}.jpg" for i, s in refs],
    )
    c.estimate_unit_id = "bathroom_primary"
    c.evidence_refs = [
        {
            "issue_id": issue_id,
            "photo_key": f"{surrogate}_{issue_id}.jpg",
            "observation": "dated finishes",
            "room_surrogate_id": surrogate,
        }
        for issue_id, surrogate in refs
    ]
    return c


def _infer_bathroom_modernization(driver_refs, support_refs):
    """Infer a bathroom_modernization package + builder inputs registry."""
    driver = _bathroom_evidence_candidate("outdated_bathroom_finishes", driver_refs)
    support = _bathroom_evidence_candidate(
        "vintage_tile_pattern_style", support_refs, severity=2,
    )
    builder_inputs = {}
    candidates = infer_package_candidates(
        [driver, support],
        [_surrogate("bathroom_1", "bathroom"), _surrogate("bathroom_2", "bathroom")],
        _BATHROOM_AFFINITY_CATALOG,
        builder_inputs_out=builder_inputs,
    )
    assert len(candidates) == 1
    return candidates, builder_inputs


class TestPostVerificationRetier:

    def test_retier_downgrades_after_support_rejected(self):
        candidates, builder_inputs = _infer_bathroom_modernization(
            driver_refs=[("d1", "bathroom_1")],
            support_refs=[("s1", "bathroom_1")],
        )
        assert candidates[0]["pricing_profile"] == "bathroom_partial_rehab"
        packages, _audit = finalize_package_candidates(
            candidates,
            {
                candidates[0]["package_id"]: {
                    "verification_status": "confirmed",
                    "reviewed_issue_ids": ["d1", "s1"],
                    "confirmed_issue_ids": ["d1"],
                    "rejected_issue_ids": ["s1"],
                }
            },
            builder_inputs=builder_inputs,
        )
        assert len(packages) == 1
        pkg = packages[0]
        # Driver alone resolves to refresh — the pre-2f partial_rehab tier
        # must not survive as post-2f truth.
        assert pkg["pricing_profile"] == "bathroom_refresh"
        assert pkg["pricing_tier"] == "refresh"
        assert pkg["cost_low"] == rp.BATHROOM_REFRESH[1]
        assert pkg["cost_high"] == rp.BATHROOM_REFRESH[2]
        assert pkg["absorption_scope"] == _package_absorption_scope("bathroom_refresh")
        assert pkg["retier_audit"]["applied"] is True
        assert pkg["retier_audit"]["previous"]["pricing_profile"] == "bathroom_partial_rehab"
        assert pkg["retier_audit"]["previous"]["cost_high"] == rp.BATHROOM_PARTIAL_REHAB[2]
        assert pkg["retier_audit"]["dropped_candidate_catalog_ids"] == [
            "vintage_tile_pattern_style",
        ]
        # Confirmed-only projections with originals preserved.
        assert pkg["supporting_issue_ids"] == ["d1"]
        assert pkg["supporting_issue_ids_original"] == ["d1", "s1"]
        assert [e["catalog_item_id"] for e in pkg["evidence_items"]] == [
            "outdated_bathroom_finishes",
        ]
        assert len(pkg["evidence_items_original"]) == 2
        assert pkg["driver_issue_ids"] == ["d1"]
        assert pkg["support_issue_ids"] == []

    def test_retier_demotes_active_package_with_zero_confirmed_evidence(self):
        candidates, builder_inputs = _infer_bathroom_modernization(
            driver_refs=[("d1", "bathroom_1")],
            support_refs=[("s1", "bathroom_1")],
        )
        packages, audit = finalize_package_candidates(
            candidates,
            {
                candidates[0]["package_id"]: {
                    "verification_status": "confirmed",
                    "reviewed_issue_ids": ["d1", "s1"],
                    "confirmed_issue_ids": [],
                    "rejected_issue_ids": ["d1", "s1"],
                }
            },
            builder_inputs=builder_inputs,
        )
        # A "confirmed" package with no confirmed evidence must not survive
        # as an active priced package.
        assert packages == []
        demoted = audit[0]
        assert demoted["audit_only"] is True
        assert demoted["estimate_eligible"] is False
        assert demoted["demotion_reason"] == "active_status_without_confirmed_evidence"
        assert demoted["verification_status"] == "confirmed"
        assert demoted["retier_audit"]["demoted"] is True

    def test_retier_missing_builder_inputs_keeps_tier_with_audit(self):
        candidates, _builder_inputs = _infer_bathroom_modernization(
            driver_refs=[("d1", "bathroom_1")],
            support_refs=[("s1", "bathroom_1")],
        )
        packages, _ = finalize_package_candidates(
            candidates,
            {
                candidates[0]["package_id"]: {
                    "verification_status": "confirmed",
                    "reviewed_issue_ids": ["d1", "s1"],
                    "confirmed_issue_ids": ["d1"],
                    "rejected_issue_ids": ["s1"],
                }
            },
            builder_inputs={},  # registry provided but entry missing
        )
        pkg = packages[0]
        assert pkg["pricing_profile"] == "bathroom_partial_rehab"
        assert pkg["retier_audit"] == {
            "applied": False,
            "reason": "missing_builder_inputs",
        }
        # Projection still runs even when re-tier cannot.
        assert pkg["supporting_issue_ids"] == ["d1"]
        assert pkg["supporting_issue_ids_original"] == ["d1", "s1"]

    def test_finalize_without_builder_inputs_still_projects_supporting(self):
        candidates, _builder_inputs = _infer_bathroom_modernization(
            driver_refs=[("d1", "bathroom_1")],
            support_refs=[("s1", "bathroom_1")],
        )
        packages, _ = finalize_package_candidates(
            candidates,
            {
                candidates[0]["package_id"]: {
                    "verification_status": "confirmed",
                    "reviewed_issue_ids": ["d1", "s1"],
                    "confirmed_issue_ids": ["d1"],
                    "rejected_issue_ids": ["s1"],
                }
            },
        )
        pkg = packages[0]
        assert "retier_audit" not in pkg
        assert pkg["supporting_issue_ids"] == ["d1"]
        assert pkg["supporting_issue_ids_original"] == ["d1", "s1"]


class TestRejectedIdHygiene:

    def test_fully_rejected_candidate_invalidated_under_active_package(self):
        candidates, _builder = _infer_bathroom_modernization(
            driver_refs=[("d1", "bathroom_1")],
            support_refs=[("s1", "bathroom_1")],
        )
        c_confirmed = _candidate(catalog_item_id="a", issue_ids=["d1"])
        c_rejected = _candidate(catalog_item_id="b", issue_ids=["s1"])
        c_mixed = _candidate(catalog_item_id="c", issue_ids=["d1", "s1"])
        apply_package_verifications_to_candidates(
            [c_confirmed, c_rejected, c_mixed],
            candidates,
            {
                candidates[0]["package_id"]: {
                    "verification_status": "confirmed",
                    "reviewed_issue_ids": ["d1", "s1"],
                    "confirmed_issue_ids": ["d1"],
                    "rejected_issue_ids": ["s1"],
                }
            },
        )
        assert c_confirmed.is_valid_detection is True
        assert c_rejected.is_valid_detection is False
        assert c_rejected.pass_2f_fallback_reason == "issues_rejected_in_package_review"
        assert c_mixed.is_valid_detection is True


# ─── Package sufficiency vs issue visibility are independent verdicts ────────
#
# 2f can reject the BUNDLE while confirming that a specific issue is visibly
# there. The confirmed issue keeps its own line item at catalog price; only the
# package's tier allowance dies. Gated on the prompt version, because verdicts
# from before pass_2f_package_v2 only ever ranked package sufficiency.

def _apply_package_verdict(candidates, verification):
    """Run the candidate gate for one bathroom_modernization verdict."""
    packages, _builder = _infer_bathroom_modernization(
        driver_refs=[("d1", "bathroom_1")],
        support_refs=[("s1", "bathroom_1")],
    )
    apply_package_verifications_to_candidates(
        candidates, packages, {packages[0]["package_id"]: verification},
    )
    return packages


class TestConfirmedIssueSurvivesRejectedPackage:

    _V2 = "pass_2f_package_v2"

    def _verdict(self, status, *, version, confirmed=("d1",), rejected=("s1",)):
        return {
            "verification_status": status,
            "reviewed_issue_ids": ["d1", "s1"],
            "confirmed_issue_ids": list(confirmed),
            "rejected_issue_ids": list(rejected),
            "prompt_template_version": version,
        }

    @pytest.mark.parametrize("status", ["rejected", "uncertain"])
    def test_confirmed_issue_survives_a_non_active_v2_verdict(self, status):
        confirmed = _candidate(catalog_item_id="a", issue_ids=["d1"])
        rejected = _candidate(catalog_item_id="b", issue_ids=["s1"])
        _apply_package_verdict(
            [confirmed, rejected], self._verdict(status, version=self._V2),
        )
        assert confirmed.is_valid_detection is True
        assert confirmed.pass_2f_applied is True
        # "None when applied" is the model contract; the package verdict lives
        # in visual_verification_status and the pairing in the source string.
        assert confirmed.pass_2f_fallback_reason is None
        assert confirmed.visual_verification_status == status
        assert confirmed.package_verification_source == (
            f"pass_2f:premium:issue_confirmed_package_{status}"
        )
        # Cleared so nothing tries to absorb it into a package that is gone.
        assert confirmed.package_id is None
        # The issue 2f could NOT see stays invalidated.
        assert rejected.is_valid_detection is False
        assert rejected.pass_2f_fallback_reason == f"package_{status}"

    @pytest.mark.parametrize("version", ["pass_2f_package_v1", "", None])
    def test_pre_v2_verdicts_do_not_revive(self, version):
        confirmed = _candidate(catalog_item_id="a", issue_ids=["d1"])
        _apply_package_verdict(
            [confirmed], self._verdict("rejected", version=version),
        )
        # Older prompts never asked for an independent per-issue judgment, so
        # their confirmed ids cannot resurrect a line item.
        assert confirmed.is_valid_detection is False
        assert confirmed.pass_2f_applied is False
        assert confirmed.pass_2f_fallback_reason == "package_rejected"

    def test_provided_verification_collision_lets_rejection_win(self):
        # A cached/provided verification bypasses _coerce_pass_2f entirely, so
        # the collision has to be resolved at the package ingestion boundary.
        colliding = _candidate(catalog_item_id="a", issue_ids=["d1"])
        packages = _apply_package_verdict(
            [colliding],
            self._verdict(
                "rejected", version=self._V2,
                confirmed=("d1",), rejected=("d1", "s1"),
            ),
        )
        assert colliding.is_valid_detection is False
        assert colliding.pass_2f_fallback_reason == "package_rejected"
        _estimate, audit = finalize_package_candidates(
            packages,
            {packages[0]["package_id"]: self._verdict(
                "rejected", version=self._V2,
                confirmed=("d1",), rejected=("d1", "s1"),
            )},
            require_confirmation=False,
        )
        assert audit[0]["confirmed_issue_ids"] == []
        assert audit[0]["rejected_issue_ids"] == ["d1", "s1"]

    def test_revived_issue_is_priced_as_a_retained_line_item(self):
        pkg = _unit_pkg(
            "bathroom_modernization", "modernization", "refresh",
            supporting_issue_ids=["a", "b"],
            confirmed_issue_ids=["a"],
            rejected_issue_ids=["b"],
            status="rejected",
        )
        pkg["reviewed_issue_ids"] = ["a", "b"]
        pkg["supporting_issue_ids_original"] = ["a", "b"]
        pkg["estimate_eligible"] = False
        pkg["audit_only"] = True
        groups = [{
            "line_items": [{
                "unit_member_allocations": [{
                    "child_id": "li::a",
                    "issue_ids": ["a"],
                    "absorbed_by_package_id": None,
                    "estimate_scope": "marketability_rehab",
                    "cost_model": "line_item",
                }],
            }],
        }]
        audit = build_issue_disposition_audit([pkg], [], groups)
        by_id = {rec["issue_id"]: rec for rec in audit["issues"]}
        assert by_id["a"]["disposition"] == "retained_line_item"
        assert by_id["b"]["disposition"] == "rejected"
        assert audit["confirmed_issues_without_priced_representation"] == []

    def test_reconcile_never_absorbs_fully_rejected_child(self):
        pkg = _unit_pkg(
            "bathroom_repair", "repair", "repair_heavy",
            supporting_issue_ids=["a", "b"],
            confirmed_issue_ids=["a"],
            rejected_issue_ids=["b"],
        )
        groups = [_group("bathroom", [
            _line_item("item_a:bathroom_primary", 400, 900,
                       room_surrogate_id="bathroom_1",
                       unit_members=[_member("ma", "bathroom_1", ["a"],
                                             estimate_unit_id="bathroom_primary")]),
            _line_item("item_b:bathroom_primary", 300, 700,
                       room_surrogate_id="bathroom_1",
                       unit_members=[_member("mb", "bathroom_1", ["b"],
                                             estimate_unit_id="bathroom_primary")]),
        ])]
        reconcile_packages_and_estimate_units(groups, [pkg])
        child_a = groups[0]["line_items"][0]["unit_member_allocations"][0]
        child_b = groups[0]["line_items"][1]["unit_member_allocations"][0]
        assert child_a["absorbed_by_package_id"] == pkg["package_id"]
        assert child_a["absorption_reason"] == "supporting_issue"
        # The rejected id stays in supporting on this stub, but reconciliation
        # subtracts rejected ids — no "supporting_issue" absorption fires.
        assert child_b["absorbed_by_package_id"] != pkg["package_id"] or (
            child_b.get("absorption_reason") != "supporting_issue"
        )

    def test_required_scope_child_never_absorbed_by_marketability_package(self):
        modern = _unit_pkg(
            "bathroom_modernization", "modernization", "full_rehab",
            supporting_issue_ids=["r1"],
        )
        groups = [_group("bathroom", [
            _line_item("cracked_tile:bathroom_primary", 400, 1_200,
                       catalog_item_id="cracked_tile",
                       trade_bucket="bathroom_fixtures_tile",
                       room_surrogate_id="bathroom_1",
                       estimate_scope="required_rehab",
                       unit_members=[_member("tile", "bathroom_1", ["r1"],
                                             estimate_unit_id="bathroom_primary")]),
        ])]
        result = reconcile_packages_and_estimate_units(groups, [modern])
        child = groups[0]["line_items"][0]["unit_member_allocations"][0]
        # Exact supporting-issue match AND covering scope — still refused:
        # required work never moves into a marketability bundle.
        assert child["absorbed_by_package_id"] is None
        required_raw = result["totals_by_scope_raw"]["required_rehab"]
        assert required_raw["low"] == 400
        assert required_raw["high"] == 1_200


class TestExpansionRetier:

    def _finalized_two_bath_package(self):
        candidates, builder_inputs = _infer_bathroom_modernization(
            driver_refs=[("d1", "bathroom_1"), ("d2", "bathroom_2")],
            support_refs=[("s1", "bathroom_1")],
        )
        packages, _ = finalize_package_candidates(
            candidates,
            {
                candidates[0]["package_id"]: {
                    "verification_status": "confirmed",
                    "reviewed_issue_ids": ["d1", "d2", "s1"],
                    "confirmed_issue_ids": ["d1", "d2", "s1"],
                    "rejected_issue_ids": [],
                }
            },
            builder_inputs=builder_inputs,
        )
        assert len(packages) == 1
        return packages, builder_inputs

    def test_expanded_clones_repriced_from_their_own_confirmed_evidence(self):
        packages, builder_inputs = self._finalized_two_bath_package()
        out, audit = expand_bathroom_modernization_packages(
            packages,
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=2,
            builder_inputs=builder_inputs,
        )
        assert audit["expanded"] is True
        by_surrogate = {p["room_surrogate_id"]: p for p in out}
        b1 = by_surrogate["bathroom_1"]
        b2 = by_surrogate["bathroom_2"]
        # bathroom_1 keeps driver + support → partial_rehab; bathroom_2 has
        # only the driver → refresh. One merged tier cloned onto every
        # bathroom was the defect.
        assert b1["pricing_profile"] == "bathroom_partial_rehab"
        assert b1["cost_high"] == rp.BATHROOM_PARTIAL_REHAB[2]
        assert b2["pricing_profile"] == "bathroom_refresh"
        assert b2["cost_high"] == rp.BATHROOM_REFRESH[2]
        assert b1["retier_audit"]["applied"] is True
        assert b2["retier_audit"]["applied"] is True
        # Per-surrogate audit mirrors the re-priced tiers.
        per_surrogate = {e["surrogate_id"]: e for e in audit["per_surrogate"]}
        assert per_surrogate["bathroom_1"]["pricing_tier"] == "partial_rehab"
        assert per_surrogate["bathroom_2"]["pricing_tier"] == "refresh"
        # Clone evidence/review metadata is per-bathroom, lineage starts empty.
        assert set(b1["confirmed_issue_ids"]) == {"d1", "s1"}
        assert b2["confirmed_issue_ids"] == ["d2"]
        assert b2["supporting_catalog_item_ids"] == ["outdated_bathroom_finishes"]
        assert b1["subsumed_package_ids"] == []
        assert b2["subsumed_package_ids"] == []
        # Clones registered for any later re-tier need.
        assert b1["package_id"] in builder_inputs
        assert b2["package_id"] in builder_inputs

    def test_expansion_original_demoted_with_lineage(self):
        packages, builder_inputs = self._finalized_two_bath_package()
        original = packages[0]
        out, audit = expand_bathroom_modernization_packages(
            packages,
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=2,
            builder_inputs=builder_inputs,
        )
        assert original not in out
        assert original["superseded_by_expansion"] is True
        assert original["audit_only"] is True
        assert original["estimate_eligible"] is False
        assert original["expanded_into_package_ids"] == audit["produced_package_ids"]
        assert original["verification_status"] == "confirmed"

    def test_expansion_skipped_when_builder_inputs_entry_missing(self):
        packages, _builder_inputs = self._finalized_two_bath_package()
        out, audit = expand_bathroom_modernization_packages(
            packages,
            bathroom_room_count_signal=_HOT_SIGNAL,
            bathroom_metadata_cap=2,
            builder_inputs={},  # registry provided, entry missing
        )
        assert audit["expanded"] is False
        assert audit["fallback_reason"] == "missing_builder_inputs"
        assert out == packages


class TestExpandedPackageAbsorption:

    def _clone(self, package_id, surrogate, issue_id):
        pkg = _unit_pkg(
            "bathroom_modernization", "modernization", "partial_rehab",
            unit="", surrogate=surrogate,
            supporting_issue_ids=[issue_id],
            package_id=package_id,
        )
        pkg["expansion_source_package_id"] = "bathroom_modernization__bathroom_primary"
        return pkg

    def test_merged_child_splits_and_each_sibling_absorbs_its_share(self):
        clone_1 = self._clone("cl_b1", "bathroom_1", "d1")
        clone_2 = self._clone("cl_b2", "bathroom_2", "d2")
        merged_member = {
            **_member("finishes", "", ["d1", "d2"],
                      estimate_unit_id="bathroom_primary"),
            "room_surrogate_id": "",
            "source_room_surrogate_ids": ["bathroom_1", "bathroom_2"],
        }
        line = _line_item(
            "outdated_bathroom_finishes:bathroom_primary", 901, 1_501,
            catalog_item_id="outdated_bathroom_finishes",
            trade_bucket="bathroom_fixtures_tile",
            estimate_scope="marketability_rehab",
            unit_members=[merged_member],
        )
        groups = [_group("bathroom", [line])]
        result = reconcile_packages_and_estimate_units(groups, [clone_1, clone_2])

        children = line["unit_member_allocations"]
        assert len(children) == 2
        assert [c["room_surrogate_id"] for c in children] == [
            "bathroom_1", "bathroom_2",
        ]
        assert all(c["split_from_child_id"] for c in children)
        assert [c["physical_unit_key"] for c in children] == [
            "bathroom_1", "bathroom_2",
        ]
        # Exact-sum arithmetic across the split.
        assert sum(c["allocated_low"] for c in children) == 901
        assert sum(c["allocated_high"] for c in children) == 1_501
        # Each sibling absorbs exactly its own bathroom's share — the zero-
        # absorption defect is gone and no child is absorbed twice.
        assert children[0]["absorbed_by_package_id"] == "cl_b1"
        assert children[1]["absorbed_by_package_id"] == "cl_b2"
        assert clone_1["absorbed_total_high"] + clone_2["absorbed_total_high"] == 1_501
        member_absorbers = merged_member["absorbed_by_package_ids"]
        assert set(member_absorbers) == {"cl_b1", "cl_b2"}
        assert result["absorbed_member_count"] == 2


class TestPerUnitCapsAndFloor:

    def _cap_child(self, unit_key, low, high):
        return {
            "allocated_low": low,
            "allocated_high": high,
            "stack_behavior": "group_cap",
            "physical_unit_key": unit_key,
        }

    def test_retained_group_cap_applies_per_physical_unit(self):
        children = [
            self._cap_child("bathroom_1", 10_000, 20_000),
            self._cap_child("bathroom_1", 8_000, 12_000),
            self._cap_child("bathroom_2", 3_000, 9_000),
        ]
        low, high, per_unit = _recompute_retained_group_detailed("bathroom", children)
        # bathroom_1 raw 32000 clips to the 25000 cap; bathroom_2 stays raw.
        # A flat cap would have clipped the whole group to 25000.
        assert high == 25_000 + 9_000
        # v3 low rule per unit: min(raw_low, cap_low) floored at best item low.
        assert low == 10_000 + 3_000
        by_unit = {u["unit_key"]: u for u in per_unit}
        assert by_unit["bathroom_1"]["cap_applied"] is True
        assert by_unit["bathroom_1"]["high"] == 25_000
        assert by_unit["bathroom_2"]["cap_applied"] is False

    def test_single_unit_group_behaves_as_before(self):
        children = [
            self._cap_child("bathroom_1", 10_000, 20_000),
            self._cap_child("bathroom_1", 8_000, 12_000),
        ]
        low, high, per_unit = _recompute_retained_group_detailed("bathroom", children)
        assert (low, high) == (10_000, 25_000)
        assert len(per_unit) == 1

    def test_post_cap_floor_cannot_clip_selected_package_range(self):
        post_low, post_high, cap_applied, floor_applied = _apply_post_package_group_cap(
            group_name="bathroom",
            pre_low=20_000,
            pre_high=53_000,
            original_capped_high=25_000,
            package_high=45_000,
            cap_behavior="respect_group_cap",
            has_group_cap=True,
            package_floor_low=15_000,
            package_floor_high=45_000,
        )
        assert post_high == 45_000
        assert floor_applied is True
        assert cap_applied is True  # still clipped relative to pre (53000)

    def test_floor_applies_even_when_pre_is_below_package_sum(self):
        # Net-delta arithmetic can push pre below the package sum when
        # in-group absorbed exceeds the capped originals.
        post_low, post_high, cap_applied, floor_applied = _apply_post_package_group_cap(
            group_name="bathroom",
            pre_low=12_000,
            pre_high=20_000,
            original_capped_high=25_000,
            package_high=45_000,
            cap_behavior="respect_group_cap",
            has_group_cap=True,
            package_floor_low=15_000,
            package_floor_high=45_000,
        )
        assert post_high == 45_000
        assert post_low == 15_000
        assert floor_applied is True
        assert cap_applied is False


class TestDisplayOnlyAggregate:

    def _turnovers(self):
        kitchen = _unit_pkg(
            "kitchen_turnover", "turnover", "turnover_std",
            room="kitchen", unit="kitchen_primary", surrogate="kitchen_1",
            cost_low=1_000, cost_high=3_000, status="confirmed_by_rule",
        )
        living = _unit_pkg(
            "living_turnover", "turnover", "turnover_std",
            room="living_room", unit="living_room_primary", surrogate="living_1",
            cost_low=500, cost_high=2_000, status="confirmed_by_rule",
        )
        return kitchen, living

    def test_aggregate_is_display_only(self):
        kitchen, living = self._turnovers()
        agg = aggregate_whole_home_turnover([kitchen, living])
        assert agg["estimate_display_only"] is True
        assert agg["estimate_eligible"] is False
        assert agg["ui_eligible"] is True
        assert agg["verification_status"] == "confirmed_by_rule"
        assert agg["cost_low"] == 1_500
        assert agg["cost_high"] == 5_000

    def test_reconciliation_totals_identical_with_and_without_aggregate(self):
        kitchen_a, living_a = self._turnovers()
        without = reconcile_packages_and_estimate_units([], [kitchen_a, living_a])

        kitchen_b, living_b = self._turnovers()
        agg = aggregate_whole_home_turnover([kitchen_b, living_b])
        with_agg = reconcile_packages_and_estimate_units([], [kitchen_b, living_b, agg])

        # The aggregate mirrors its contributors' summed range — retaining
        # both used to double the turnover total exactly.
        for key in (
            "package_total_low", "package_total_high",
            "net_delta_low", "net_delta_high",
        ):
            assert with_agg[key] == without[key], key
        assert with_agg["totals_by_scope_raw"] == without["totals_by_scope_raw"]
        assert agg["replacement_delta_low"] == 0
        assert agg["replacement_delta_high"] == 0
        whole_home = next(
            (g for g in with_agg["package_group_reconciliation"]
             if g["group"] == "whole_home"),
            None,
        )
        assert whole_home is not None
        assert whole_home["package_total"] == {"low": 0, "high": 0}
        assert whole_home["display_only_package_ids"] == [agg["package_id"]]


class TestTwoPassAbsorption:

    def _fixture(self):
        owner = _unit_pkg(
            "bathroom_repair", "repair", "repair_heavy",
            supporting_issue_ids=["i1"], package_id="owner",
        )
        broad = _unit_pkg(
            "bathroom_modernization", "modernization", "full_rehab",
            supporting_issue_ids=["m1"], package_id="broad",
        )
        line = _line_item(
            "worn_finish:bathroom_primary", 500, 1_500,
            catalog_item_id="worn_finish",
            trade_bucket="bathroom_fixtures_tile",
            estimate_scope="marketability_rehab",
            unit_members=[_member("m", "bathroom_1", ["i1"],
                                  estimate_unit_id="bathroom_primary")],
        )
        return owner, broad, [_group("bathroom", [line])], line

    def test_direct_evidence_owner_wins_regardless_of_list_order(self):
        for order in ("broad_first", "owner_first"):
            owner, broad, groups, line = self._fixture()
            packages = [broad, owner] if order == "broad_first" else [owner, broad]
            reconcile_packages_and_estimate_units(groups, packages)
            child = line["unit_member_allocations"][0]
            # The broad modernization joins the same unit and its scope covers
            # the trade — but the repair holds the issue as confirmed direct
            # evidence, so ownership never depends on list order.
            assert child["absorbed_by_package_id"] == "owner", order
            assert child["absorption_reason"] == "supporting_issue", order


class TestIssueDispositionAudit:

    def test_dispositions_and_unpriced_detector(self):
        active = _unit_pkg(
            "bathroom_modernization", "modernization", "partial_rehab",
            supporting_issue_ids=["a", "b", "c"],
            confirmed_issue_ids=["a", "b", "c"],
            package_id="active_pkg",
        )
        active["rejected_issue_ids"] = ["r"]
        active["reviewed_issue_ids"] = ["a", "b", "c", "r"]
        active["supporting_issue_ids_original"] = ["a", "b", "c", "r"]
        demoted = {
            "package_id": "demoted_pkg",
            "audit_only": True,
            "estimate_eligible": False,
            "demotion_reason": "active_status_without_confirmed_evidence",
            "confirmed_issue_ids": ["z"],
            "reviewed_issue_ids": ["z"],
            "supporting_issue_ids": [],
            "supporting_issue_ids_original": ["z"],
            "rejected_issue_ids": [],
        }
        groups = [{
            "line_items": [{
                "unit_member_allocations": [
                    {
                        "child_id": "li::b",
                        "issue_ids": ["b"],
                        "absorbed_by_package_id": "active_pkg",
                        "estimate_scope": "marketability_rehab",
                        "cost_model": "line_item",
                    },
                    {
                        "child_id": "li::c",
                        "issue_ids": ["c"],
                        "absorbed_by_package_id": None,
                        "estimate_scope": "required_rehab",
                        "cost_model": "line_item",
                    },
                ],
            }],
        }]
        audit = build_issue_disposition_audit([active, demoted], [active], groups)
        by_id = {rec["issue_id"]: rec for rec in audit["issues"]}
        assert by_id["a"]["disposition"] == "priced_in_package"
        assert by_id["a"]["final_package_id"] == "active_pkg"
        assert by_id["b"]["disposition"] == "priced_in_package"  # confirmed in active pkg
        assert by_id["c"]["disposition"] == "priced_in_package"
        assert by_id["r"]["disposition"] == "rejected"
        assert by_id["z"]["disposition"] == "dropped_with_demoted_package"
        # z is confirmed but has no priced representation anywhere — the
        # "confirmed issue disappeared" detector fires.
        assert audit["confirmed_issues_without_priced_representation"] == ["z"]

    def test_absorbed_and_retained_children_resolve_when_not_packaged(self):
        pkg = _unit_pkg(
            "bathroom_repair", "repair", "repair_heavy",
            supporting_issue_ids=["a"],
            confirmed_issue_ids=["a"],
            package_id="pkg_r",
        )
        pkg["reviewed_issue_ids"] = ["a", "b", "c"]
        pkg["rejected_issue_ids"] = []
        pkg["supporting_issue_ids_original"] = ["a", "b", "c"]
        groups = [{
            "line_items": [{
                "unit_member_allocations": [
                    {
                        "child_id": "li::b",
                        "issue_ids": ["b"],
                        "absorbed_by_package_id": "pkg_r",
                        "estimate_scope": "required_rehab",
                        "cost_model": "line_item",
                    },
                    {
                        "child_id": "li::c",
                        "issue_ids": ["c"],
                        "absorbed_by_package_id": None,
                        "estimate_scope": "required_rehab",
                        "cost_model": "line_item",
                    },
                ],
            }],
        }]
        audit = build_issue_disposition_audit([pkg], [pkg], groups)
        by_id = {rec["issue_id"]: rec for rec in audit["issues"]}
        # Only "a" is confirmed; b/c are package-adjacent line-item issues and
        # resolve through their children.
        assert by_id["b"]["disposition"] == "absorbed_child"
        assert by_id["b"]["final_package_id"] == "pkg_r"
        assert by_id["c"]["disposition"] == "retained_line_item"
        assert audit["confirmed_issues_without_priced_representation"] == []


# ─── Repeated evidence is not repeated cost or scope ─────────────────────────
#
# Candidates are keyed (catalog_item_id, scene_group, room_surrogate) while
# package buckets key on billable_estimate_unit_id, so one condition seen across
# several surrogates of ONE billable room arrives as several candidates. Every
# breadth and cost count in package inference must collapse them, or a more
# observant model gets "rewarded" with a higher tier.

_KITCHEN_DEDUPE_CATALOG = {
    "items": [
        {
            **_cat_item("outdated_or_damaged_cabinets", severity=2),
            "display_class": "marketability",
            "trade_bucket": "kitchen_cabinets_counters",
            "estimate": {"estimate_tier": "high", "group": "kitchen"},
            "cost": {"base_low": 500, "base_high": 10_000},
            **_affinity(("kitchen", "kitchen_modernization", "package_driver")),
        },
        {
            **_cat_item("outdated_kitchen_finishes", kind="upgrade",
                        category="opportunity", severity=3),
            "display_class": "marketability",
            "trade_bucket": "kitchen_cabinets_counters",
            "estimate": {"estimate_tier": "high", "group": "kitchen"},
            "cost": {"base_low": 2_000, "base_high": 20_000},
            **_affinity(("kitchen", "kitchen_modernization", "package_driver")),
        },
    ]
}

_LIVING_SUPPORT_CATALOG = {
    "items": [
        {
            **_cat_item(cat_id, kind="upgrade", category="opportunity", severity=1),
            "display_class": "marketability",
            "trade_bucket": "electrical",
            "estimate": {"estimate_tier": "medium", "group": "living"},
            "cost": {"base_low": 500, "base_high": 5_000},
            **_affinity(("living", "living_modernization", "package_support")),
        }
        for cat_id in ("dated_lighting_fixtures", "dated_wood_paneling")
    ]
}


def _dedupe_candidate(catalog_item_id, surrogate, issue_id, *, severity=2,
                      kind="defect", trade_bucket="kitchen_cabinets_counters",
                      group="kitchen", billable="kitchen"):
    """A candidate as extract_estimate_candidates would emit it: one per
    (catalog item, surrogate), all pointing at the same billable unit."""
    c = _candidate(
        catalog_item_id=catalog_item_id,
        kind=kind,
        severity=severity,
        trade_bucket=trade_bucket,
        group=group,
        room_surrogate_id=surrogate,
        issue_ids=[issue_id],
        photo_keys=[f"{surrogate}_{issue_id}.jpg"],
    )
    c.billable_estimate_unit_id = billable
    c.evidence_refs = [{
        "issue_id": issue_id,
        "photo_key": f"{surrogate}_{issue_id}.jpg",
        "observation": "observed condition",
        "room_surrogate_id": surrogate,
    }]
    return c


def _living_support(catalog_item_id, surrogate, issue_id, *, billable="living_1"):
    return _dedupe_candidate(
        catalog_item_id, surrogate, issue_id,
        severity=1, kind="upgrade", trade_bucket="electrical",
        group="living", billable=billable,
    )


class TestRepeatedEvidenceIsNotRepeatedScope:

    _KITCHEN_SURROGATES = [
        _surrogate(f"kitchen_{n}", "kitchen") for n in (1, 2, 3)
    ]
    _LIVING_SURROGATES = [
        _surrogate(f"living_{n}", "living_room") for n in (1, 2, 3)
    ]

    def test_repeated_kitchen_evidence_does_not_escalate_the_tier(self):
        once = infer_package_candidates(
            [
                _dedupe_candidate("outdated_or_damaged_cabinets", "kitchen_1", "cab1"),
                _dedupe_candidate("outdated_kitchen_finishes", "kitchen_1", "fin1",
                                  severity=3, kind="upgrade"),
            ],
            self._KITCHEN_SURROGATES, _KITCHEN_DEDUPE_CATALOG,
        )
        repeated = infer_package_candidates(
            [
                _dedupe_candidate("outdated_or_damaged_cabinets", "kitchen_1", "cab1"),
                _dedupe_candidate("outdated_or_damaged_cabinets", "kitchen_2", "cab2"),
                _dedupe_candidate("outdated_kitchen_finishes", "kitchen_1", "fin1",
                                  severity=3, kind="upgrade"),
                _dedupe_candidate("outdated_kitchen_finishes", "kitchen_2", "fin2",
                                  severity=3, kind="upgrade"),
                _dedupe_candidate("outdated_kitchen_finishes", "kitchen_3", "fin3",
                                  severity=3, kind="upgrade"),
            ],
            self._KITCHEN_SURROGATES, _KITCHEN_DEDUPE_CATALOG,
        )
        assert len(once) == 1 and len(repeated) == 1
        # Absorbed high is 10k + 20k either way. Summing every observation gave
        # 2x10k + 3x20k = 80k, which escalated refresh all the way to full_rehab.
        assert once[0]["pricing_tier"] == "partial_rehab"
        assert repeated[0]["pricing_tier"] == once[0]["pricing_tier"]
        assert repeated[0]["cost_high"] == once[0]["cost_high"]
        assert any(
            "absorbed_high=30000" in note
            for note in repeated[0]["level_decision_notes"]
        )

    def test_same_item_in_two_billable_units_stays_two_items(self):
        # The guard against over-dedup: two physical kitchens are two work items.
        main = _dedupe_candidate("outdated_or_damaged_cabinets", "kitchen_1", "c1",
                                 billable="kitchen_main")
        adu = _dedupe_candidate("outdated_or_damaged_cabinets", "kitchen_2", "c2",
                                billable="kitchen_adu")
        repeat_of_main = _dedupe_candidate("outdated_or_damaged_cabinets",
                                           "kitchen_3", "c3", billable="kitchen_main")
        assert len(rp._distinct_billable_items([main, adu, repeat_of_main])) == 2

    def test_repeated_single_support_no_longer_emits_a_package(self):
        suppressed = []
        emitted = infer_package_candidates(
            [
                _living_support("dated_lighting_fixtures", "living_1", "a"),
                _living_support("dated_lighting_fixtures", "living_2", "b"),
            ],
            self._LIVING_SURROGATES, _LIVING_SUPPORT_CATALOG,
            suppressed_out=suppressed,
        )
        # Two photos of ONE dated fixture used to conjure a whole package.
        assert emitted == []
        assert suppressed[0]["suppression_reason"] == (
            "weak_after_duplicate_evidence_collapse"
        )

    def test_two_distinct_supports_still_emit_a_package(self):
        emitted = infer_package_candidates(
            [
                _living_support("dated_lighting_fixtures", "living_1", "a"),
                _living_support("dated_wood_paneling", "living_1", "b"),
            ],
            self._LIVING_SURROGATES, _LIVING_SUPPORT_CATALOG,
        )
        assert len(emitted) == 1
        assert emitted[0]["trigger_reason"] == (
            "multiple_package_support_same_estimate_unit"
        )

    def test_strength_counts_distinct_supports_not_observations(self):
        driver = _dedupe_candidate("some_driver", "living_1", "d", severity=2)
        one_support_twice = [
            _living_support("dated_lighting_fixtures", "living_1", "s1"),
            _living_support("dated_lighting_fixtures", "living_2", "s2"),
        ]
        two_supports = [
            _living_support("dated_lighting_fixtures", "living_1", "s1"),
            _living_support("dated_wood_paneling", "living_1", "s3"),
        ]
        assert compute_package_strength([driver], one_support_twice) == "moderate"
        assert compute_package_strength([driver], two_supports) == "strong"

    def test_turnover_breadth_counts_distinct_items_not_observations(self):
        def _ev(cat_id, surrogate, issue_id):
            return _dedupe_candidate(
                cat_id, surrogate, issue_id, severity=1, kind="upgrade",
                trade_bucket="cleaning_turnover", group="living",
                billable="living_1",
            )

        # classify_component returns None for cleaning_turnover, so the
        # paint+flooring branch cannot fire and only breadth decides the tier.
        repeated = [_ev("dated_thing", f"living_{n}", f"i{n}") for n in (1, 2, 3)]
        spec, _tier, _notes = rp._resolve_room_turnover_profile(
            rp.LIVING_TURNOVER_LIGHT, rp.LIVING_TURNOVER_STD,
            repeated, [], repeated,
        )
        assert spec == rp.LIVING_TURNOVER_LIGHT

        distinct = [_ev(f"dated_{k}", "living_1", f"j{k}") for k in ("a", "b", "c")]
        spec, _tier, _notes = rp._resolve_room_turnover_profile(
            rp.LIVING_TURNOVER_LIGHT, rp.LIVING_TURNOVER_STD,
            distinct, [], distinct,
        )
        assert spec == rp.LIVING_TURNOVER_STD

    def test_multiphoto_opportunity_driver_still_emits(self):
        # The intentionally-untouched path: repeated photos of ONE opportunity
        # driver corroborate it. Dedup must not break this, only stop it from
        # being priced twice.
        emitted = infer_package_candidates(
            [
                _dedupe_candidate("outdated_kitchen_finishes", "kitchen_1", "f1",
                                  severity=3, kind="upgrade"),
                _dedupe_candidate("outdated_kitchen_finishes", "kitchen_2", "f2",
                                  severity=3, kind="upgrade"),
            ],
            self._KITCHEN_SURROGATES, _KITCHEN_DEDUPE_CATALOG,
        )
        assert len(emitted) == 1
        assert emitted[0]["trigger_reason"] == (
            "opportunity_driver_with_multiphoto_corroboration"
        )
        # Priced on one $20k item, not two.
        assert any(
            "absorbed_high=20000" in note
            for note in emitted[0]["level_decision_notes"]
        )


# ── Catalog 3.2: contextual repair support in the shared inference core ──────

class TestContextualRepairSupportFlag:
    """infer_package_candidates is shared with the v4 estimator, which must keep
    its pre-3.2 routing. The flag is the seam; these tests pin both sides."""

    @staticmethod
    def _catalog():
        def item(cid, kind, room, package_type, role, marked=False):
            entry = {"package_type": package_type, "package_role": role}
            if marked:
                entry[REPAIR_SUPPORT_MARKER] = True
            return {**_cat_item(cid, kind=kind),
                    "package_affinity": {room: entry}}

        return {"items": [
            item("drywall_crack", "defect", "bedroom",
                 "bedroom_repair", PACKAGE_ROLE_DRIVER),
            item("carpet_worn", "degradation", "bedroom",
                 "bedroom_modernization", PACKAGE_ROLE_DRIVER, marked=True),
            item("paint_worn", "degradation", "bedroom",
                 "bedroom_modernization", PACKAGE_ROLE_SUPPORT, marked=True),
        ]}

    @staticmethod
    def _inputs():
        surrogates = [_surrogate("bedroom_1", "bedroom")]
        units = [{"estimate_unit_id": "bedroom_primary",
                  "source_room_surrogate_ids": ["bedroom_1"]}]
        candidates = [
            _candidate(catalog_item_id="drywall_crack", kind="defect",
                       room_surrogate_id="bedroom_1", issue_ids=["i_drywall"]),
            _candidate(catalog_item_id="carpet_worn", kind="degradation",
                       room_surrogate_id="bedroom_1", issue_ids=["i_carpet"]),
            _candidate(catalog_item_id="paint_worn", kind="degradation",
                       room_surrogate_id="bedroom_1", issue_ids=["i_paint"]),
        ]
        return candidates, surrogates, units

    def _run(self, flag):
        candidates, surrogates, units = self._inputs()
        packages = infer_package_candidates(
            candidates, surrogates, self._catalog(),
            estimate_units=units, contextual_repair_support=flag,
        )
        return {p["package_type"]: p for p in packages}

    def test_default_is_off_so_v4_routing_is_unchanged(self):
        """The v4 guarantee: a marker in the catalog changes nothing unless the
        caller opts in."""
        candidates, surrogates, units = self._inputs()
        default = infer_package_candidates(
            candidates, surrogates, self._catalog(), estimate_units=units,
        )
        explicit_off = self._run(False)
        assert {p["package_type"] for p in default} == set(explicit_off)
        assert set(explicit_off) == {"bedroom_modernization", "bedroom_repair"}
        assert explicit_off["bedroom_modernization"]["driver_issue_ids"] == ["i_carpet"]

    def test_flag_on_moves_marked_wear_into_the_paired_repair_family(self):
        on = self._run(True)
        assert set(on) == {"bedroom_repair"}
        repair = on["bedroom_repair"]
        assert repair["driver_issue_ids"] == ["i_drywall"]
        supports = set(repair["support_issue_ids"]) - set(repair["driver_issue_ids"])
        assert supports == {"i_carpet", "i_paint"}

    def test_no_issue_is_emitted_under_two_families(self):
        on = self._run(True)
        seen = [i for p in on.values() for i in p["support_issue_ids"]]
        assert len(seen) == len(set(seen))


def test_real_catalog_marked_routes_have_a_wired_repair_family():
    """Catalog 3.2 coverage for the marked routes: each one is a modernization
    route whose paired {room}_repair family is a real package type. Rooms whose
    repair family has no catalog driver are recorded in
    docs/FINDINGS_catalog_3_2_deferred_issues.md — the marker is inert there,
    deliberately, until a driver exists."""
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads(
        (root / "tools" / "issue_catalog_kind_v2.json").read_text(encoding="utf-8")
    )
    table = build_package_affinity(catalog)
    marked = {
        (room, issue_id): meta
        for (room, issue_id), meta in table.items()
        if meta.get(REPAIR_SUPPORT_MARKER)
    }
    assert len(marked) == 24

    repair_drivers_by_room = {
        room for (room, _), meta in table.items()
        if meta["package_category"] == PACKAGE_CATEGORY_REPAIR
        and meta["package_role"] == PACKAGE_ROLE_DRIVER
    }
    for (room, issue_id), meta in marked.items():
        assert meta["package_type"] == f"{room}_modernization", (room, issue_id)
        assert paired_repair_package_type(meta["package_type"]) == f"{room}_repair"

    # The recorded gap: kitchen has a kitchen_repair package type and tier
    # specs, but no catalog item drives it, so its 8 marked routes cannot fire.
    assert "kitchen" not in repair_drivers_by_room
    assert sum(1 for room, _ in marked if room == "kitchen") == 8
    assert {"bathroom", "bedroom", "living"} <= repair_drivers_by_room
