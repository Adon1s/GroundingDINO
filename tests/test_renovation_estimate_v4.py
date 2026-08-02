"""
Tests for tools.renovation_estimate_v4 -- room-aware orchestrator.

PR 3A scope:
  - Stamps room_surrogate_id from build_room_surrogates onto deep-copied
    v4 issues; original issues_flat is never mutated.
  - Re-extracts candidates so the same catalog item splits per room.
  - Reuses Pass 2f review fields from v3-reviewed candidates with auditable
    casework (exact / subset / collapse-with-agreeing-postures / ambiguous /
    unmatched). Stamps pass_2f_reuse_method on
    each v4 candidate that received fields.
  - Runs the existing v3 group-estimate machinery against the v4-stamped
    issues. v3 output is left untouched.

PR 3B+ behaviors (allocation, packages, reconciliation, property bounds)
are explicitly out of scope here.
"""

import copy
import json
from pathlib import Path

import pytest

from tools.catalog_cost_model import (
    COST_MODEL_SOURCE_DERIVED_UPGRADE_ROOM_ALLOWANCE,
    COST_MODEL_SOURCE_LEGACY_DEFAULT,
    LINE_ITEM,
    ROOM_ALLOWANCE,
    derive_cost_model,
)
from tools.estimate_scope import (
    INSPECTION_RISK,
    MARKETABILITY_REHAB,
    OPTIONAL_VALUE_ADD,
    REQUIRED_REHAB,
    apply_estimate_scope,
    build_scope_headline_tiers,
    classify_estimate_scope,
    classify_estimate_scope_with_reason,
)
from tools.renovation_estimate import (
    compute_renovation_estimate,
    extract_estimate_candidates,
    resolve_estimate_units,
)
from tools.renovation_estimate_v4 import (
    _build_bathroom_room_count_signal,
    _build_project_scope_breakdown,
    _extract_package_only_candidates,
    _reuse_pass_2f_fields,
    compute_renovation_estimate_v4,
)
from tools.rehab_packages import is_package_eligible_catalog_item

from tests.test_renovation_estimate import (
    HIGH_ESTIMATE,
    MEDIUM_ESTIMATE,
    _make_catalog,
    _make_issue,
    _make_item,
)


# ─── Local fixtures ──────────────────────────────────────────────────────────

BATHROOM_ESTIMATE = {
    "estimate_tier": "high",
    "strategy": "replace_only",
    "group": "bathroom",
    "stack_behavior": "group_cap",
}


def _make_photos(*entries):
    """Build a photos dict from a sequence of scenes.

    Each entry may be a bare scene string or a (scene, photo_key) tuple.
    Indices are assigned 1..N in order. Modeled on tests/test_room_surrogates.py.
    """
    photos = {}
    for i, entry in enumerate(entries, start=1):
        if isinstance(entry, tuple):
            scene, photo_key = entry
        else:
            scene = entry
            photo_key = f"p{i:03d}.jpg"
        photos[photo_key] = {
            "photo": {"photo_key": photo_key, "index": i},
            "scene": {"id": scene, "group": "irrelevant"},
        }
    return photos


def _assert_nested_scope_tiers(v4):
    """Assert the three headline tiers are nested, banded, and scope-decomposed.

    ``required <= resale_ready <= full_renewal`` componentwise; each tier's
    correlated band sits inside its straight-sum band (``sum_low``/``sum_high``);
    and ``full_renewal``'s straight sums equal the sum of the three visible
    capped scopes — the scope-decomposed view of the existing ``final_rehab``
    headline, so the two agree when no inspection-risk packages are present.
    """
    required = v4["final_rehab_required"]
    resale = v4["final_rehab_resale_ready"]
    full = v4["final_rehab_full_renewal"]
    capped = v4["totals_by_scope_capped"]
    for key in ("low", "high"):
        assert required[key] <= resale[key] <= full[key]
    for tier in (required, resale, full):
        assert 0 <= tier["sum_low"] <= tier["low"] <= tier["high"] <= tier["sum_high"]
        assert tier["midpoint"] == (tier["low"] + tier["high"]) // 2
    for key, sum_key in (("low", "sum_low"), ("high", "sum_high")):
        assert full[sum_key] == (
            capped[REQUIRED_REHAB][key]
            + capped[MARKETABILITY_REHAB][key]
            + capped[OPTIONAL_VALUE_ADD][key]
        )
        assert full[sum_key] == v4["final_rehab"][key]


def _build_reviewed_candidates(
    catalog,
    issues,
    *,
    posture="replace",
    is_valid=True,
):
    """Extract + resolve candidates and stamp them as Pass-2f-reviewed."""
    candidates = extract_estimate_candidates(issues, catalog)
    candidates = resolve_estimate_units(candidates, issues, catalog)
    for c in candidates:
        c.is_valid_detection = is_valid
        c.review_posture = posture
        c.effective_posture = posture
        c.review_source = "pass_2f"
        c.pass_2f_attempted = True
        c.pass_2f_applied = True
        c.review_visible_scope = "partial"
    return candidates


def _line_items(estimate):
    out = []
    for group in estimate.get("groups", []):
        out.extend(group.get("line_items", []))
    return out


def _line_item_by_room(estimate, room_surrogate_id):
    for li in _line_items(estimate):
        if li.get("room_surrogate_id") == room_surrogate_id:
            return li
    return None


def _build_simple_v3():
    """A small v3 setup with two issues in different scene groups."""
    catalog = _make_catalog(
        _make_item(
            "cabinets",
            estimate=HIGH_ESTIMATE,
            trade_bucket="kitchen_cabinets_counters",
            scope="replace",
        ),
        _make_item(
            "paint",
            estimate=MEDIUM_ESTIMATE,
            trade_bucket="paint",
            scope="repair",
        ),
    )
    issues = [
        _make_issue("cabinets", scene_group="kitchen", photo_key="k1.jpg",
                    issue_id="iss_cab"),
        _make_issue("paint", scene_group="living_areas", photo_key="l1.jpg",
                    issue_id="iss_paint"),
    ]
    photos = _make_photos(
        ("kitchen", "k1.jpg"),
        ("living_room", "l1.jpg"),
    )
    return catalog, issues, photos


class _UncertainPackageVLM:
    def __init__(self):
        self.calls = []

    async def analyze_images(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({
            "verification_status": "uncertain",
            "confirmed_issue_ids": [],
            "rejected_issue_ids": [],
            "evidence_summary": "Images are not sufficient to confirm.",
        })


def test_bathroom_room_count_signal_accepts_single_active_multiple_vote_and_ignores_kitchen():
    assert _build_bathroom_room_count_signal([
        {
            "room": "bathroom",
            "package_id": "bath_1",
            "verification_status": "confirmed",
            "visible_room_count": "multiple_rooms",
        },
        {
            "room": "kitchen",
            "package_id": "kitchen_1",
            "verification_status": "confirmed",
            "visible_room_count": "multiple_rooms",
        },
    ]) == {
        "likely_multiple_visible_bathrooms": True,
        "multiple_rooms_vote_count": 1,
        "one_room_vote_count": 0,
        "active_one_room_vote_count": 0,
        "unclear_vote_count": 0,
        "supporting_package_ids": ["bath_1"],
        "one_room_package_ids": [],
        "conflicting_active_votes": False,
        "warnings": [],
        "basis": "requires_active_multiple_room_vote_no_one_room_votes",
    }
    uncertain = _build_bathroom_room_count_signal([
        {
            "room": "bathroom",
            "package_id": "bath_1",
            "verification_status": "uncertain",
            "visible_room_count": "multiple_rooms",
        },
    ])
    assert uncertain["likely_multiple_visible_bathrooms"] is False
    assert uncertain["multiple_rooms_vote_count"] == 0
    mixed = _build_bathroom_room_count_signal([
        {
            "room": "bathroom",
            "package_id": "bath_1",
            "verification_status": "confirmed",
            "visible_room_count": "multiple_rooms",
        },
        {
            "room": "bathroom",
            "package_id": "bath_2",
            "verification_status": "confirmed",
            "visible_room_count": "one_room",
        },
    ])
    # Two ACTIVE packages disagreeing is explicit ambiguity, not a silent cold
    # signal: no expansion, and the conflict is flagged.
    assert mixed["likely_multiple_visible_bathrooms"] is False
    assert mixed["one_room_vote_count"] == 1
    assert mixed["conflicting_active_votes"] is True
    assert mixed["warnings"] == ["conflicting_active_room_count_votes"]
    assert mixed["basis"] == "conflicting_active_votes_ambiguous_no_expansion"


def test_bathroom_room_count_rejected_one_room_vote_does_not_veto():
    # A rejected package's one_room observation must not veto expansion any
    # more than a rejected multiple_rooms vote may trigger it.
    signal = _build_bathroom_room_count_signal([
        {
            "room": "bathroom",
            "package_id": "bath_1",
            "verification_status": "confirmed",
            "visible_room_count": "multiple_rooms",
        },
        {
            "room": "bathroom",
            "package_id": "bath_2",
            "verification_status": "rejected",
            "visible_room_count": "one_room",
        },
    ])
    assert signal["likely_multiple_visible_bathrooms"] is True
    assert signal["one_room_vote_count"] == 0
    assert signal["conflicting_active_votes"] is False


# ─── Surface-level provenance / version / null-input behavior ────────────────


class TestSurface:

    def test_v4_version_field(self):
        catalog, issues, photos = _build_simple_v3()
        v3 = compute_renovation_estimate(issues, catalog)
        v4 = compute_renovation_estimate_v4(
            issues, catalog, photos,
        )
        assert v4["version"] == "renovation_estimate_v4"
        assert v3["version"] == "renovation_estimate_v3"

    def test_v4_returns_estimate_without_base_estimate_input(self):
        catalog, issues, photos = _build_simple_v3()
        v4 = compute_renovation_estimate_v4(issues, catalog, photos)
        assert v4["version"] == "renovation_estimate_v4"

    def test_v4_provenance_metadata(self):
        catalog, issues, photos = _build_simple_v3()
        v4 = compute_renovation_estimate_v4(
            issues, catalog, photos,
        )
        assert v4["provenance"] == {
            "mode": "room_aware_line_item_estimate",
            "derived_from": "renovation_estimate_v3",
            "v3_pass_2f_reused": False,
            "v4_phases_applied": [
                "surrogates",
                "estimate_units",
                "package_candidates",
                "pass_2f",
                "package_finalization",
                "post_verification_retier",
                "modernization_dedupe",
                "bathroom_expansion",
                "package_subsumption",
                "reconciliation",
                "issue_disposition_audit",
                "cost_adjustment",
                "evidence_projection",
            ],
            "packages_enabled": True,
            "reconciliation_enabled": True,
            "package_confirmation_required": True,
        }

    def test_v4_provenance_marks_pass_2f_reused_when_provided(self):
        catalog, issues, photos = _build_simple_v3()
        reviewed = _build_reviewed_candidates(catalog, issues)
        v4 = compute_renovation_estimate_v4(
            issues, catalog, photos,
            v3_reviewed_candidates=reviewed,
        )
        assert v4["provenance"]["v3_pass_2f_reused"] is True

    def test_v4_empty_candidates_fallback(self):
        catalog = _make_catalog(_make_item("a", estimate=HIGH_ESTIMATE))
        v4 = compute_renovation_estimate_v4(
            [], catalog, {},
        )
        assert v4["version"] == "renovation_estimate_v4"
        assert v4["raw_totals"] == {"low": 0, "high": 0}
        assert v4["groups"] == []
        assert v4["room_surrogates"] == []
        assert v4["packages"] == []
        assert v4["reconciliation"] == {
            "absorbed_total_low": 0,
            "absorbed_total_high": 0,
            "package_total_low": 0,
            "package_total_high": 0,
            "net_delta_low": 0,
            "net_delta_high": 0,
            "absorbed_member_count": 0,
            "package_count": 0,
            "retained_group_totals": [],
            "package_group_reconciliation": [],
            "estimate_members": [],
            "reconciliation_audit": {"groups": []},
            "reconciliation_warnings": [],
            "warnings": [],
            "visible_rehab": {
                "low": 0,
                "high": 0,
                "midpoint": 0,
                "basis": "verified_visible_line_items_before_package_replacement",
            },
            "package_adjusted_rehab": {
                "low": 0,
                "high": 0,
                "midpoint": 0,
                "basis": "visible_work_after_package_reconciliation",
            },
            "latent_risk_exposure": {
                "low": 0,
                "high": 0,
                "midpoint": None,
                "basis": "inspect_posture_items_and_hidden_condition_exposure",
            },
            "worst_case_exposure": {
                "low": 0,
                "high": 0,
                "midpoint": None,
                "basis": "package_adjusted_rehab_plus_latent_risk_exposure",
            },
            "final_rehab": {
                "low": 0,
                "high": 0,
                "midpoint": 0,
                "basis": "package_adjusted_rehab",
                "source": "renovation_estimate_v4",
            },
            "totals_by_scope_raw": {
                "required_rehab": {"low": 0, "high": 0},
                "marketability_rehab": {"low": 0, "high": 0},
                "optional_value_add": {"low": 0, "high": 0},
                "inspection_risk": {"low": 0, "high": 0},
            },
            "totals_by_scope_capped": {
                "required_rehab": {"low": 0, "high": 0},
                "marketability_rehab": {"low": 0, "high": 0},
                "optional_value_add": {"low": 0, "high": 0},
                "inspection_risk": {"low": 0, "high": 0},
            },
            "final_rehab_required": {
                "low": 0,
                "high": 0,
                "midpoint": 0,
                "basis": "totals_by_scope_capped.required_rehab",
                "source": "renovation_estimate_v4",
                "sum_low": 0,
                "sum_high": 0,
            },
            "final_rehab_resale_ready": {
                "low": 0,
                "high": 0,
                "midpoint": 0,
                "basis": "totals_by_scope_capped.required_rehab_plus_marketability_rehab",
                "source": "renovation_estimate_v4",
                "sum_low": 0,
                "sum_high": 0,
            },
            "final_rehab_full_renewal": {
                "low": 0,
                "high": 0,
                "midpoint": 0,
                "basis": "totals_by_scope_capped.required_rehab_plus_marketability_rehab_plus_optional_value_add",
                "source": "renovation_estimate_v4",
                "sum_low": 0,
                "sum_high": 0,
            },
        }
        for bucket_name in (
            "visible_rehab",
            "package_adjusted_rehab",
            "latent_risk_exposure",
            "worst_case_exposure",
            "final_rehab",
            "totals_by_scope_raw",
            "totals_by_scope_capped",
            "final_rehab_required",
            "final_rehab_resale_ready",
            "final_rehab_full_renewal",
        ):
            assert v4[bucket_name] == v4["reconciliation"][bucket_name]
        assert v4["pass_2f_reuse_audit"] == {
            "matched_count": 0,
            "unmatched_v4_count": 0,
            "ambiguous_count": 0,
            "dominant_posture_collapses": 0,
        }
        assert v4["warnings"] == []
        assert v4["provenance"]["mode"] == "room_aware_line_item_estimate"

    def test_single_photo_kitchen_opportunity_is_suppressed_before_2f(self):
        item = _make_item(
            "outdated_kitchen_finishes",
            estimate=HIGH_ESTIMATE,
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            scope="replace",
        )
        item.update({
            "display_class": "marketability",
            "package_role": "package_driver",
            "package_type": "kitchen_modernization",
            "scene_groups": ["kitchen"],
        })
        catalog = _make_catalog(item)
        issues = [
            _make_issue(
                "outdated_kitchen_finishes",
                scene_group="kitchen",
                photo_key="k1.jpg",
                issue_id="iss_outdated_kitchen",
                catalog_item_kind="upgrade",
            )
        ]
        photos = _make_photos(("kitchen", "k1.jpg"))
        v4 = compute_renovation_estimate_v4(
            issues,
            catalog,
            photos,
        )

        assert v4["packages"] == []
        assert v4["package_candidates"] == []
        assert len(v4["suppressed_package_candidates"]) == 1
        assert v4["suppressed_package_candidates"][0]["supporting_photo_count"] == 1
        assert v4["final_rehab"]["low"] == 0
        assert v4["final_rehab"]["high"] == 0
        line_items = _line_items(v4)
        assert line_items[0]["is_valid_detection"] is False
        assert line_items[0]["pass_2f_fallback_reason"] == "insufficient_corroboration_for_opportunity_driver"

    def test_multiphoto_kitchen_opportunity_reaches_candidate_but_stays_zero_without_confirmation(self):
        item = _make_item(
            "outdated_kitchen_finishes",
            estimate=HIGH_ESTIMATE,
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            scope="replace",
        )
        item.update({
            "display_class": "marketability",
            "package_role": "package_driver",
            "package_type": "kitchen_modernization",
            "scene_groups": ["kitchen"],
        })
        catalog = _make_catalog(item)
        issues = [
            _make_issue(
                "outdated_kitchen_finishes",
                scene_group="kitchen",
                photo_key="k1.jpg",
                issue_id="iss_outdated_kitchen_1",
                catalog_item_kind="upgrade",
            ),
            _make_issue(
                "outdated_kitchen_finishes",
                scene_group="kitchen",
                photo_key="k2.jpg",
                issue_id="iss_outdated_kitchen_2",
                catalog_item_kind="upgrade",
            ),
        ]
        photos = _make_photos(("kitchen", "k1.jpg"), ("kitchen", "k2.jpg"))
        v4 = compute_renovation_estimate_v4(
            issues,
            catalog,
            photos,
        )

        assert v4["packages"] == []
        assert len(v4["package_candidates"]) == 1
        candidate = v4["package_candidates"][0]
        assert candidate["trigger_reason"] == "opportunity_driver_with_multiphoto_corroboration"
        assert candidate["supporting_photo_count"] == 2
        assert candidate["corroboration_basis"] == "multi_photo_same_issue"
        assert candidate["review_photo_keys"] == ["k1.jpg", "k2.jpg"]
        assert candidate["verification_status"] == "not_run"
        assert v4["final_rehab"] == {
            "low": 0,
            "high": 0,
            "midpoint": 0,
            "basis": "package_adjusted_rehab",
            "source": "renovation_estimate_v4",
        }
        line_items = _line_items(v4)
        assert line_items[0]["is_valid_detection"] is False
        assert line_items[0]["visual_verification_status"] == "not_run"

    def test_multiphoto_kitchen_opportunity_runs_2f_and_stays_zero_when_uncertain(self):
        item = _make_item(
            "outdated_kitchen_finishes",
            estimate=HIGH_ESTIMATE,
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            scope="replace",
        )
        item.update({
            "display_class": "marketability",
            "package_role": "package_driver",
            "package_type": "kitchen_modernization",
            "scene_groups": ["kitchen"],
        })
        catalog = _make_catalog(item)
        issues = [
            _make_issue(
                "outdated_kitchen_finishes",
                scene_group="kitchen",
                photo_key="k1.jpg",
                issue_id="iss_outdated_kitchen_1",
                catalog_item_kind="upgrade",
            ),
            _make_issue(
                "outdated_kitchen_finishes",
                scene_group="kitchen",
                photo_key="k2.jpg",
                issue_id="iss_outdated_kitchen_2",
                catalog_item_kind="upgrade",
            ),
        ]
        photos = _make_photos(("kitchen", "k1.jpg"), ("kitchen", "k2.jpg"))
        tmp_dir = Path("tests") / "_tmp_v4_package_pass_2f"
        tmp_dir.mkdir(exist_ok=True)
        image_1 = tmp_dir / "k1.jpg"
        image_2 = tmp_dir / "k2.jpg"
        image_1.write_bytes(b"image-one")
        image_2.write_bytes(b"image-two")
        vlm = _UncertainPackageVLM()
        timing = {}
        try:
            v4 = compute_renovation_estimate_v4(
                issues,
                catalog,
                photos,
                pass_2f_vlm_client=vlm,
                pass_2f_model_config={"model": "test-model", "provider": "test"},
                photo_key_to_path={"k1.jpg": image_1, "k2.jpg": image_2},
                timing_recorder=timing,
            )
        finally:
            for path in (image_1, image_2):
                if path.exists():
                    path.unlink()
            try:
                tmp_dir.rmdir()
            except OSError:
                pass

        assert len(vlm.calls) == 1
        assert vlm.calls[0]["image_paths"] == [image_1, image_2]
        assert v4["pass_2f_trace"]["attempted_count"] == 1
        assert timing["pass_2f_sec"] > 0
        assert v4["pass_2f_trace"]["wall_sec"] == pytest.approx(timing["pass_2f_sec"])
        assert v4["pass_2f_trace"]["uncertain_count"] == 1
        assert v4["package_candidates"][0]["verification_status"] == "uncertain"
        assert v4["package_candidates"][0]["review_image_paths"] == [str(image_1), str(image_2)]
        assert v4["packages"] == []
        assert v4["final_rehab"]["low"] == 0
        assert v4["final_rehab"]["high"] == 0

    def test_confirmed_multiphoto_kitchen_opportunity_produces_refresh_package(self):
        item = _make_item(
            "outdated_kitchen_finishes",
            estimate=HIGH_ESTIMATE,
            kind="upgrade",
            trade_bucket="kitchen_cabinets_counters",
            scope="replace",
        )
        item.update({
            "display_class": "marketability",
            "package_role": "package_driver",
            "package_type": "kitchen_modernization",
            "scene_groups": ["kitchen"],
        })
        catalog = _make_catalog(item)
        issues = [
            _make_issue(
                "outdated_kitchen_finishes",
                scene_group="kitchen",
                photo_key="k1.jpg",
                issue_id="iss_outdated_kitchen_1",
                catalog_item_kind="upgrade",
            ),
            _make_issue(
                "outdated_kitchen_finishes",
                scene_group="kitchen",
                photo_key="k2.jpg",
                issue_id="iss_outdated_kitchen_2",
                catalog_item_kind="upgrade",
            ),
        ]
        photos = _make_photos(("kitchen", "k1.jpg"), ("kitchen", "k2.jpg"))
        v4 = compute_renovation_estimate_v4(
            issues,
            catalog,
            photos,
            package_verifications={
                "kitchen_modernization__kitchen_primary": {
                    "verification_status": "confirmed",
                    "confirmed_issue_ids": [
                        "iss_outdated_kitchen_1",
                        "iss_outdated_kitchen_2",
                    ],
                    "evidence_summary": "Visible dated kitchen finishes across two photos.",
                }
            },
        )

        assert len(v4["packages"]) == 1
        assert v4["packages"][0]["pricing_profile"] == "kitchen_refresh"
        assert v4["packages"][0]["estimate_eligible"] is True
        assert v4["package_candidates"][0]["verification_status"] == "confirmed"
        assert v4["final_rehab"]["low"] > 0
        assert v4["final_rehab"]["high"] > 0

    def test_bathroom_package_only_supports_feed_package_without_line_items(self):
        driver = _make_item(
            "outdated_bathroom_finishes",
            estimate={**BATHROOM_ESTIMATE, "group": "bathroom"},
            kind="upgrade",
            severity=3,
            trade_bucket="bathroom_fixtures_tile",
            scope="replace",
        )
        driver.update({
            "display_class": "marketability",
            "package_affinity": {
                "bathroom": {
                    "package_type": "bathroom_modernization",
                    "package_role": "package_driver",
                },
            },
            "scene_groups": ["bathroom"],
        })
        vintage_tile = _make_item(
            "vintage_tile_pattern_style",
            kind="upgrade",
            severity=2,
            trade_bucket="bathroom_fixtures_tile",
            scope="replace",
        )
        vintage_tile.update({
            "display_class": "marketability",
            "package_affinity": {
                "bathroom": {
                    "package_type": "bathroom_modernization",
                    "package_role": "package_support",
                },
            },
            "scene_groups": ["bathroom"],
        })
        vanity_light = _make_item(
            "dated_bathroom_vanity_light",
            kind="upgrade",
            severity=2,
            trade_bucket="electrical",
            scope="replace",
        )
        vanity_light.update({
            "display_class": "marketability",
            "package_affinity": {
                "bathroom": {
                    "package_type": "bathroom_modernization",
                    "package_role": "package_support",
                },
            },
            "scene_groups": ["bathroom"],
        })
        catalog = _make_catalog(driver, vintage_tile, vanity_light)
        issues = [
            _make_issue(
                "outdated_bathroom_finishes",
                scene_group="bathroom",
                photo_key="b1.jpg",
                issue_id="iss_bath_driver",
                catalog_item_kind="upgrade",
            ),
            _make_issue(
                "vintage_tile_pattern_style",
                scene_group="bathroom",
                photo_key="b1.jpg",
                issue_id="iss_vintage_tile",
                catalog_item_kind="upgrade",
            ),
            _make_issue(
                "dated_bathroom_vanity_light",
                scene_group="bathroom",
                photo_key="b2.jpg",
                issue_id="iss_vanity_light",
                catalog_item_kind="upgrade",
            ),
        ]
        photos = _make_photos(("bathroom", "b1.jpg"), ("bathroom", "b2.jpg"))
        v4 = compute_renovation_estimate_v4(
            issues,
            catalog,
            photos,
            package_verifications={
                "bathroom_modernization__bathroom_primary": {
                    "verification_status": "confirmed",
                    "confirmed_issue_ids": [
                        "iss_bath_driver",
                        "iss_vintage_tile",
                        "iss_vanity_light",
                    ],
                    "evidence_summary": "Visible outdated bathroom package.",
                    "visible_room_count": "one_room",
                    "visible_room_count_evidence": "Consistent tile and vanity layout.",
                }
            },
        )

        line_catalog_ids = {
            line.get("catalog_item_id")
            for line in _line_items(v4)
        }
        assert "outdated_bathroom_finishes" in line_catalog_ids
        assert "vintage_tile_pattern_style" not in line_catalog_ids
        assert "dated_bathroom_vanity_light" not in line_catalog_ids
        evidence_catalog_ids = {
            item["catalog_item_id"]
            for item in v4["package_candidates"][0]["evidence_items"]
        }
        assert evidence_catalog_ids == {
            "outdated_bathroom_finishes",
            "vintage_tile_pattern_style",
            "dated_bathroom_vanity_light",
        }
        package_only_items = [
            item for item in v4["package_candidates"][0]["evidence_items"]
            if item["catalog_item_id"] != "outdated_bathroom_finishes"
        ]
        assert all(item["package_evidence_only"] is True for item in package_only_items)
        assert v4["bathroom_room_count_signal"] == {
            "likely_multiple_visible_bathrooms": False,
            "multiple_rooms_vote_count": 0,
            "one_room_vote_count": 1,
            "active_one_room_vote_count": 1,
            "unclear_vote_count": 0,
            "supporting_package_ids": [],
            "one_room_package_ids": ["bathroom_modernization__bathroom_primary"],
            "conflicting_active_votes": False,
            "warnings": [],
            "basis": "requires_active_multiple_room_vote_no_one_room_votes",
        }


def _build_inspect_risk_v4():
    est = {
        "estimate_tier": "high",
        "strategy": "inspect_only",
        "group": "structure",
        "stack_behavior": "max_only",
        "unit_policy": "per_property",
    }
    manual_cost = {
        "mode": "allowance",
        "base_low": 2000,
        "base_high": 40000,
        "per_occurrence_low": 3900,
        "per_occurrence_high": 7500,
        "cap_low": 40000,
        "cap_high": 100000,
        "cost_source": "manual",
    }
    catalog = _make_catalog(
        _make_item(
            "foundation",
            estimate=est,
            trade_bucket="foundation_structure",
            scope="repair",
            cost=manual_cost,
        ),
    )
    issues = [
        _make_issue(
            "foundation",
            scene_group="exterior",
            photo_key="ext.jpg",
            issue_id="iss_foundation",
        ),
    ]
    photos = _make_photos(("exterior_front", "ext.jpg"))
    return compute_renovation_estimate_v4(
        issues,
        catalog,
        photos,
    )


def _real_catalog_item(item_id):
    catalog = json.loads(Path("tools/issue_catalog.json").read_text(encoding="utf-8"))
    for item in catalog.get("items", []):
        if item.get("id") == item_id:
            return item
    raise AssertionError(f"missing catalog item {item_id}")


class TestExplicitEstimateBuckets:

    def test_inspect_risk_does_not_inflate_final_rehab_high(self):
        result = _build_inspect_risk_v4()
        assert result["final_rehab"]["high"] == result["package_adjusted_rehab"]["high"]

    def test_worst_case_includes_latent_risk_exposure(self):
        result = _build_inspect_risk_v4()
        assert result["worst_case_exposure"]["high"] == (
            result["package_adjusted_rehab"]["high"]
            + result["latent_risk_exposure"]["high"]
        )

    def test_latent_risk_low_defaults_to_zero(self):
        result = _build_inspect_risk_v4()
        assert result["latent_risk_exposure"]["low"] == 0

    def test_inspection_risk_scope_is_not_required_or_resale_ready(self):
        result = _build_inspect_risk_v4()
        assert result["totals_by_scope_capped"][INSPECTION_RISK]["high"] == (
            result["latent_risk_exposure"]["high"]
        )
        assert result["final_rehab_required"]["high"] == 0
        assert result["final_rehab_resale_ready"]["high"] == 0

    def test_does_not_hard_cap_estimate_by_price(self):
        catalog, issues, photos = _build_simple_v3()
        baseline = compute_renovation_estimate_v4(
            issues,
            catalog,
            photos,
        )
        computed_high = baseline["package_adjusted_rehab"]["high"]

        result = compute_renovation_estimate_v4(
            issues,
            catalog,
            photos,
            property_metadata={"list_price": 5000, "sqft": 1800},
        )

        # The bargain list price drags the market factor to its clamp floor,
        # but the estimate is scaled by the factor — never capped at price.
        factor = result["cost_adjustment"]["factor"]
        assert factor == 0.75
        assert result["package_adjusted_rehab"]["high"] == round(computed_high * factor)
        assert result["final_rehab"]["high"] == result["package_adjusted_rehab"]["high"]
        assert computed_high > 5000
        assert result["package_adjusted_rehab"]["high"] > 5000
        codes = {flag["code"] for flag in result["sanity_flags"]}
        assert "package_adjusted_high_gt_50pct_price" in codes
        assert "worst_case_high_gt_80pct_price" in codes


# ─── Market/size cost adjustment ─────────────────────────────────────────────

class TestCostAdjustment:

    def test_no_metadata_is_factor_neutral(self):
        catalog, issues, photos = _build_simple_v3()
        v4 = compute_renovation_estimate_v4(issues, catalog, photos)
        adjustment = v4["cost_adjustment"]
        assert adjustment["factor"] == 1.0
        assert "cost_adjustment" in v4["provenance"]["v4_phases_applied"]

    def test_baseline_metadata_output_matches_no_metadata(self):
        # Identity keystone: factor exactly 1.0 leaves every dollar untouched.
        catalog, issues, photos = _build_simple_v3()
        plain = compute_renovation_estimate_v4(issues, catalog, photos)
        at_baseline = compute_renovation_estimate_v4(
            issues, catalog, photos,
            property_metadata={"price_per_sqft": 230, "sqft": 1800},
        )
        assert at_baseline["cost_adjustment"]["factor"] == 1.0
        plain.pop("cost_adjustment")
        at_baseline.pop("cost_adjustment")
        assert plain == at_baseline

    def test_end_to_end_scaling(self):
        catalog, issues, photos = _build_simple_v3()
        baseline = compute_renovation_estimate_v4(issues, catalog, photos)
        scaled = compute_renovation_estimate_v4(
            issues, catalog, photos,
            property_metadata={"price_per_sqft": 460, "sqft": 2400},
        )

        adjustment = scaled["cost_adjustment"]
        expected_factor = (460 / 230) ** 0.4 * (2400 / 1800) ** 0.3
        assert adjustment["factor"] == pytest.approx(expected_factor)
        assert adjustment["ppsf_source_key"] == "price_per_sqft"

        factor = adjustment["factor"]
        for bucket_name in (
            "visible_rehab",
            "package_adjusted_rehab",
            "final_rehab",
            "final_rehab_required",
            "final_rehab_resale_ready",
            "final_rehab_full_renewal",
        ):
            for key in ("low", "high"):
                assert scaled[bucket_name][key] == round(
                    baseline[bucket_name][key] * factor
                )

        final = scaled["final_rehab"]
        assert final["midpoint"] == (final["low"] + final["high"]) // 2

    def test_sanity_flags_see_scaled_totals(self):
        # High market factor pushes the package-adjusted high over 50% of a
        # list price the unscaled estimate would have stayed under.
        catalog, issues, photos = _build_simple_v3()
        baseline = compute_renovation_estimate_v4(issues, catalog, photos)
        unscaled_high = baseline["package_adjusted_rehab"]["high"]
        # List price just above 2x the unscaled high: unscaled ratio < 0.5,
        # scaled-by-1.5 ratio > 0.5. sqft chosen so ppsf clamps at 1.5 and the
        # derived list_price/sqft chain is exercised.
        list_price = int(unscaled_high * 2) + 100
        sqft = max(int(list_price / 800), 1)
        scaled = compute_renovation_estimate_v4(
            issues, catalog, photos,
            property_metadata={"list_price": list_price, "sqft": sqft},
        )
        assert scaled["cost_adjustment"]["market_factor"] == 1.5
        codes = {flag["code"] for flag in scaled["sanity_flags"]}
        assert "package_adjusted_high_gt_50pct_price" in codes


# ─── Non-mutation guarantees ─────────────────────────────────────────────────

# Estimate scope modes.
class TestEstimateScopeModes:
    def _inspect_candidate(self, item, item_id=None):
        catalog_item_id = item_id or item["id"]
        catalog = _make_catalog(item)
        candidate = extract_estimate_candidates([
            _make_issue(catalog_item_id, issue_id=f"iss_{catalog_item_id}"),
        ], catalog)[0]
        candidate.is_valid_detection = True
        candidate.review_posture = "inspect"
        candidate.effective_posture = "inspect"
        apply_estimate_scope(candidate, item)
        return candidate

    def test_outdated_kitchen_finishes_are_marketability_not_required(self):
        item = _real_catalog_item("outdated_kitchen_finishes")
        catalog = _make_catalog(item)
        issues = [
            _make_issue(
                "outdated_kitchen_finishes",
                scene_group="kitchen",
                photo_key="k1.jpg",
                issue_id="iss_outdated_kitchen",
            ),
        ]
        candidate = extract_estimate_candidates(issues, catalog)[0]
        assert classify_estimate_scope(candidate, item, None) == MARKETABILITY_REHAB
        assert candidate.estimate_scope == MARKETABILITY_REHAB

    def test_outdated_bathroom_finishes_are_marketability_not_required(self):
        item = _real_catalog_item("outdated_bathroom_finishes")
        catalog = _make_catalog(item)
        issues = [
            _make_issue(
                "outdated_bathroom_finishes",
                scene_group="bathroom",
                photo_key="b1.jpg",
                issue_id="iss_outdated_bathroom",
            ),
        ]
        candidate = extract_estimate_candidates(issues, catalog)[0]
        assert classify_estimate_scope(candidate, item, None) == MARKETABILITY_REHAB
        assert candidate.estimate_scope == MARKETABILITY_REHAB

    def test_untagged_room_upgrade_derives_room_allowance_and_marketability(self):
        cases = [
            ("per_room", "other"),
            ("per_kitchen", "kitchen"),
            ("per_bathroom", "bathroom"),
        ]
        for unit_policy, group in cases:
            item_id = f"untagged_{unit_policy}_dated_upgrade"
            item = _make_item(
                item_id,
                name=f"Untagged {unit_policy} dated finishes",
                kind="upgrade",
                scope="replace",
                trade_bucket="kitchen_cabinets_counters",
                estimate={
                    "estimate_tier": "medium",
                    "strategy": "replace_only",
                    "group": group,
                    "stack_behavior": "sum",
                    "unit_policy": unit_policy,
                },
            )
            catalog = _make_catalog(item)
            candidate = extract_estimate_candidates([
                _make_issue(item_id, issue_id=f"iss_{unit_policy}"),
            ], catalog)[0]

            assert candidate.cost_model == ROOM_ALLOWANCE
            assert (
                candidate.cost_model_source
                == COST_MODEL_SOURCE_DERIVED_UPGRADE_ROOM_ALLOWANCE
            )
            assert classify_estimate_scope(candidate, item, None) == MARKETABILITY_REHAB
            assert candidate.estimate_scope == MARKETABILITY_REHAB

    def test_untagged_and_unknown_cost_models_default_to_line_item(self):
        unknown_model, unknown_source = derive_cost_model({}, None)
        assert unknown_model == LINE_ITEM
        assert unknown_source == COST_MODEL_SOURCE_LEGACY_DEFAULT

        item = _make_item(
            "untagged_physical_defect",
            kind="defect",
            estimate={
                "estimate_tier": "medium",
                "strategy": "repair_only",
                "group": "other",
                "stack_behavior": "sum",
                "unit_policy": "per_scope",
            },
        )
        catalog = _make_catalog(item)
        candidate = extract_estimate_candidates([
            _make_issue("untagged_physical_defect", issue_id="iss_defect"),
        ], catalog)[0]

        assert candidate.cost_model == LINE_ITEM
        assert candidate.cost_model_source == COST_MODEL_SOURCE_LEGACY_DEFAULT

    def test_missing_base_cabinets_are_required_rehab(self):
        item = _real_catalog_item("missing_base_cabinets_exposed_subfloor")
        catalog = _make_catalog(item)
        issues = [
            _make_issue(
                "missing_base_cabinets_exposed_subfloor",
                scene_group="kitchen",
                photo_key="k1.jpg",
                issue_id="iss_missing_cabs",
            ),
        ]
        candidate = extract_estimate_candidates(issues, catalog)[0]
        assert classify_estimate_scope(candidate, item, None) == REQUIRED_REHAB
        assert candidate.estimate_scope == REQUIRED_REHAB

    def test_missing_base_cabinets_with_inspect_preserves_required_rehab(self):
        item = _real_catalog_item("missing_base_cabinets_exposed_subfloor")
        candidate = self._inspect_candidate(item)
        assert candidate.estimate_scope == REQUIRED_REHAB
        assert candidate.baseline_scope_before_posture == REQUIRED_REHAB
        assert candidate.effective_posture == "inspect"
        assert candidate.visible_required_with_inspect_posture is True
        assert candidate.required_baseline_included is True
        assert candidate.inspection_risk_added is True

    def test_interior_wall_stripped_to_studs_with_inspect_preserves_required_rehab(self):
        item = _real_catalog_item("interior_wall_stripped_to_studs")
        candidate = self._inspect_candidate(item)
        assert candidate.estimate_scope == REQUIRED_REHAB
        assert candidate.baseline_scope_before_posture == REQUIRED_REHAB
        assert candidate.effective_posture == "inspect"
        assert candidate.visible_required_with_inspect_posture is True
        assert candidate.required_baseline_included is True
        assert candidate.inspection_risk_added is True

    def test_pass_2f_inspect_routes_possible_foundation_to_inspection_risk(self):
        item = _make_item(
            "possible_foundation_issue",
            name="Possible foundation issue",
            estimate={
                "estimate_tier": "high",
                "strategy": "repair_only",
                "group": "structure",
                "stack_behavior": "max_only",
            },
            trade_bucket="foundation_structure",
            severity=3,
        )
        candidate = self._inspect_candidate(item)
        assert candidate.estimate_scope == INSPECTION_RISK
        assert candidate.baseline_scope_before_posture == REQUIRED_REHAB
        assert candidate.required_baseline_included is False
        assert candidate.inspection_risk_added is True

    def test_visible_required_condition_helper_preserves_required_under_inspect(self):
        estimate = {
            "estimate_tier": "high",
            "strategy": "repair_only",
            "group": "other",
            "stack_behavior": "sum",
        }
        cases = [
            ("bare_floor", "Bare or missing finish flooring exposing subfloor."),
            ("boarded_opening", "Door or window opening is boarded over and damaged."),
            ("water_damage", "Visible water damage and moisture intrusion on wall."),
            ("structure_damage", "Visible structural damage with compromised framing."),
            ("unsafe_system", "Visible electrical risk with exposed wiring."),
            ("nonfunctional_component", "Major component is removed and nonfunctional."),
        ]
        for item_id, description in cases:
            item = {
                "id": item_id,
                "name": item_id.replace("_", " ").title(),
                "kind": "defect",
                "severity": 3,
                "scope": "repair",
                "trade_bucket": "electrical" if item_id == "unsafe_system" else "other",
                "description": description,
                "cost": {"mode": "heuristic"},
                "estimate": estimate,
            }
            candidate = self._inspect_candidate(item)
            assert candidate.estimate_scope == REQUIRED_REHAB, item_id
            assert candidate.baseline_scope_before_posture == REQUIRED_REHAB
            assert candidate.visible_required_with_inspect_posture is True
            assert candidate.inspection_risk_added is True

    def test_outdated_kitchen_finishes_with_inspect_is_not_required(self):
        item = _real_catalog_item("outdated_kitchen_finishes")
        candidate = self._inspect_candidate(item)
        assert candidate.estimate_scope != REQUIRED_REHAB
        assert candidate.baseline_scope_before_posture == MARKETABILITY_REHAB
        assert candidate.estimate_scope == INSPECTION_RISK
        assert candidate.inspection_risk_added is True

    def test_optional_modernization_routes_to_optional_value_add(self):
        item = {
            "id": "layout_modernization_opportunity",
            "name": "Layout Modernization Opportunity",
            "kind": "upgrade",
            "tier": "optional",
            "category": "opportunity",
            "severity": 2,
            "scope": "replace",
            "trade_bucket": "paint_drywall",
            "description": "Layout modernization opportunity to reconfigure rooms.",
            "cost": {"mode": "heuristic"},
            "estimate": {
                "estimate_tier": "medium",
                "strategy": "replace_only",
                "group": "other",
                "stack_behavior": "sum",
            },
        }
        catalog = _make_catalog(item)
        candidate = extract_estimate_candidates([
            _make_issue("layout_modernization_opportunity", issue_id="iss_layout"),
        ], catalog)[0]
        assert classify_estimate_scope(candidate, item, None) == OPTIONAL_VALUE_ADD

    def test_dated_only_kitchen_bath_with_multiphoto_corroboration_is_resale_ready_not_required(self):
        # Multi-photo corroboration is required for opportunity-only kitchen/bathroom
        # signals to emit. A single isolated dated photo per room is suppressed by
        # the corroboration gate (kitchen and bathroom are symmetric on this).
        kitchen = copy.deepcopy(_real_catalog_item("outdated_kitchen_finishes"))
        bathroom = copy.deepcopy(_real_catalog_item("outdated_bathroom_finishes"))
        catalog = _make_catalog(kitchen, bathroom)
        issues = [
            _make_issue(
                "outdated_kitchen_finishes",
                scene_group="kitchen",
                photo_key="k1.jpg",
                issue_id="iss_kitchen_dated_1",
            ),
            _make_issue(
                "outdated_kitchen_finishes",
                scene_group="kitchen",
                photo_key="k2.jpg",
                issue_id="iss_kitchen_dated_2",
            ),
            _make_issue(
                "outdated_bathroom_finishes",
                scene_group="bathroom",
                photo_key="b1.jpg",
                issue_id="iss_bath_dated_1",
            ),
            _make_issue(
                "outdated_bathroom_finishes",
                scene_group="bathroom",
                photo_key="b2.jpg",
                issue_id="iss_bath_dated_2",
            ),
        ]
        photos = _make_photos(
            ("kitchen", "k1.jpg"),
            ("kitchen", "k2.jpg"),
            ("bathroom", "b1.jpg"),
            ("bathroom", "b2.jpg"),
        )

        v4 = compute_renovation_estimate_v4(
            issues,
            catalog,
            photos,
        )

        # No defects → no required rehab. Dated finishes route to resale-ready.
        assert v4["final_rehab_required"]["high"] == 0
        assert v4["final_rehab_resale_ready"]["high"] >= v4["final_rehab_required"]["high"]
        assert v4["final_rehab"]["low"] == v4["package_adjusted_rehab"]["low"]
        assert v4["final_rehab"]["high"] == v4["package_adjusted_rehab"]["high"]
        assert v4["final_rehab"]["basis"] == "package_adjusted_rehab"
        _assert_nested_scope_tiers(v4)

    def test_required_damage_listing_has_required_rehab_total(self):
        missing_cabs = copy.deepcopy(_real_catalog_item("missing_base_cabinets_exposed_subfloor"))
        bare_floor = copy.deepcopy(_real_catalog_item("bare_or_missing_finish_flooring"))
        catalog = _make_catalog(missing_cabs, bare_floor)
        issues = [
            _make_issue(
                "missing_base_cabinets_exposed_subfloor",
                scene_group="kitchen",
                photo_key="k1.jpg",
                issue_id="iss_missing_cabs",
            ),
            _make_issue(
                "bare_or_missing_finish_flooring",
                scene_group="kitchen",
                photo_key="k2.jpg",
                issue_id="iss_bare_floor",
            ),
        ]
        photos = _make_photos(("kitchen", "k1.jpg"), ("kitchen", "k2.jpg"))

        v4 = compute_renovation_estimate_v4(
            issues,
            catalog,
            photos,
        )

        assert v4["final_rehab_required"]["high"] > 0
        assert v4["totals_by_scope_capped"][REQUIRED_REHAB]["high"] > 0
        assert v4["final_rehab_resale_ready"]["high"] >= v4["final_rehab_required"]["high"]
        assert v4["final_rehab"]["low"] == v4["package_adjusted_rehab"]["low"]
        assert v4["final_rehab"]["high"] == v4["package_adjusted_rehab"]["high"]
        assert v4["final_rehab"]["basis"] == "package_adjusted_rehab"
        _assert_nested_scope_tiers(v4)


# Representative catalog estimate-scope regression. Protects the renewal toggle
# from silent catalog drift (see docs/catalog_estimate_scope_audit.md). Uses an
# empty `{}` candidate so the catalog values are the sole source of truth (pure
# baseline, no inspect posture) -- the same call the audit enumeration makes.
class TestRepresentativeCatalogScopes:
    # (catalog_id, expected_scope, expect_catalog_override)
    CASES = [
        # required repair (heuristic) -- incl. substring traps that must NOT demote:
        ("missing_base_cabinets_exposed_subfloor", REQUIRED_REHAB, False),
        ("plumbing_fixture_leaking_stained", REQUIRED_REHAB, False),   # "fixture" trap stays required
        ("unfinished_interior_wall_osb_exposed", REQUIRED_REHAB, False),  # "unfinished"/"finish" trap stays required
        ("tile_or_grout_damage", REQUIRED_REHAB, False),              # package-repair driver
        # marketability renewal:
        ("outdated_kitchen_finishes", MARKETABILITY_REHAB, True),     # override; package-modernization driver
        # optional value-add (incl. "open wall" value-add substring):
        ("layout_modernization_opportunity", OPTIONAL_VALUE_ADD, False),
        # corrected by this audit's overrides:
        ("roofline_water_damage_suspected", INSPECTION_RISK, True),   # latent -> inspection_risk
        ("deck_surface_weathering", MARKETABILITY_REHAB, True),       # "structural/failure" via negation phrase
        ("scratched_or_damaged_flooring", MARKETABILITY_REHAB, True),  # sev-1 cosmetic flooring
        ("bathroom_layout_modernization_opportunity", OPTIONAL_VALUE_ADD, True),  # layout redesign
        # curb-appeal / conflated entries pinned by the term_matches conversion pass:
        ("fence_damaged_or_weathered", MARKETABILITY_REHAB, True),     # "rotted" fired on weathered boards
        ("appliance_damage_or_missing", MARKETABILITY_REHAB, True),    # category:systems forced required
        ("gutter_maintenance_needed", MARKETABILITY_REHAB, True),      # minor service, not a must-fix
        ("empty_or_deteriorated_inground_pool", MARKETABILITY_REHAB, True),  # restore-or-fill is discretionary
    ]

    def test_representative_catalog_items_classify_as_intended(self):
        for item_id, expected_scope, expect_override in self.CASES:
            item = _real_catalog_item(item_id)
            scope, reason = classify_estimate_scope_with_reason({}, item, None)
            assert scope == expected_scope, (
                f"{item_id}: expected {expected_scope}, got {scope} (reason={reason})"
            )
            if expect_override:
                assert reason.startswith("catalog_override"), (
                    f"{item_id}: expected a catalog_override reason, got {reason!r}"
                )
            else:
                assert not reason.startswith("catalog_override"), (
                    f"{item_id}: expected heuristic classification, got override {reason!r}"
                )

    # `_catalog_text` concatenates the candidate's supporting_observations, so
    # free-form VLM text reaches the scope terms. `mold` used to prefix-match
    # "crown molding", promoting any cosmetic item to required_rehab whenever a
    # photo mentioned trim. The `mold$` marker fixes it; assert both directions
    # so the real-mold path can't be traded away for the fix.
    def test_crown_molding_does_not_promote_a_cosmetic_item_to_required(self):
        item = _real_catalog_item("wall_scuffs_marks_or_dents")
        candidate = {
            "supporting_observations": [
                "crown molding runs along the ceiling and the walls "
                "show scuffs and minor dents.",
            ],
        }
        scope, reason = classify_estimate_scope_with_reason(candidate, item, None)
        assert scope == MARKETABILITY_REHAB, (
            f"crown molding promoted a cosmetic item to {scope} (reason={reason})"
        )

    def test_real_mold_observation_still_classifies_required(self):
        item = _real_catalog_item("wall_scuffs_marks_or_dents")
        for observation in (
            "dark mold growth is spreading across the lower wall surface.",
            "the wall has mold-like patches near the baseboard.",
        ):
            candidate = {"supporting_observations": [observation]}
            scope, reason = classify_estimate_scope_with_reason(candidate, item, None)
            assert scope == REQUIRED_REHAB, (
                f"{observation!r}: expected required_rehab, got {scope} (reason={reason})"
            )


# Scope headline tiers.
class TestScopeHeadlineTiers:
    """Direct unit coverage of build_scope_headline_tiers decomposition."""

    def test_three_tiers_nested_and_decomposed(self):
        capped = {
            REQUIRED_REHAB: {"low": 1000, "high": 2000},
            MARKETABILITY_REHAB: {"low": 3000, "high": 5000},
            OPTIONAL_VALUE_ADD: {"low": 4000, "high": 9000},
            INSPECTION_RISK: {"low": 0, "high": 7000},
        }
        required, resale, full = build_scope_headline_tiers(capped)

        assert required["low"] == 1000
        assert required["high"] == 2000
        assert resale["low"] == 1000 + 3000
        assert resale["high"] == 2000 + 5000
        assert full["low"] == 1000 + 3000 + 4000
        assert full["high"] == 2000 + 5000 + 9000

        # Strictly nested: required <= resale_ready <= full_renewal.
        for key in ("low", "high"):
            assert required[key] <= resale[key] <= full[key]

        # inspection_risk is never folded into any headline tier.
        assert full["high"] == 2000 + 5000 + 9000  # not + 7000
        assert full["basis"] == (
            "totals_by_scope_capped."
            "required_rehab_plus_marketability_rehab_plus_optional_value_add"
        )
        assert full["source"] == "renovation_estimate_v4"
        assert full["midpoint"] == (full["low"] + full["high"]) // 2

    def test_missing_scopes_default_to_zero(self):
        required, resale, full = build_scope_headline_tiers({})
        for bucket in (required, resale, full):
            assert bucket["low"] == 0
            assert bucket["high"] == 0

    def test_no_width_stats_keeps_straight_sums_without_band_fields(self):
        capped = {REQUIRED_REHAB: {"low": 1000, "high": 2000}}
        required, _resale, _full = build_scope_headline_tiers(capped)
        assert required["low"] == 1000
        assert required["high"] == 2000
        assert "sum_low" not in required
        assert "sum_high" not in required


class TestCorrelatedHeadlineBands:
    """Width blend W = √Σw² + ρ(Σw − √Σw²): tiers become midpoint ± W instead
    of summed extremes, without ever leaving the straight-sum band."""

    # Four contributions of (0, 2000) in required_rehab: Σw = 4000, Σw² = 4M.
    _CAPPED = {REQUIRED_REHAB: {"low": 0, "high": 8000}}
    _STATS = {REQUIRED_REHAB: {"sum_w": 4000.0, "sum_w2": 4_000_000.0}}

    def test_rho_zero_is_pure_quadrature(self):
        required, _, _ = build_scope_headline_tiers(self._CAPPED, self._STATS, rho=0.0)
        # W = √(4·1000²·4) = 2000 around midpoint 4000.
        assert required["low"] == 2000
        assert required["high"] == 6000
        assert required["sum_low"] == 0
        assert required["sum_high"] == 8000
        assert required["midpoint"] == 4000

    def test_rho_one_reproduces_straight_sums(self):
        required, resale, full = build_scope_headline_tiers(self._CAPPED, self._STATS, rho=1.0)
        for bucket in (required, resale, full):
            assert bucket["low"] == bucket["sum_low"]
            assert bucket["high"] == bucket["sum_high"]

    def test_single_contributor_keeps_full_width(self):
        capped = {REQUIRED_REHAB: {"low": 0, "high": 2000}}
        stats = {REQUIRED_REHAB: {"sum_w": 1000.0, "sum_w2": 1_000_000.0}}
        required, _, _ = build_scope_headline_tiers(capped, stats, rho=0.4)
        assert required["low"] == 0
        assert required["high"] == 2000

    def test_raw_widths_clamp_to_capped_half_width(self):
        # Caps truncated the scope to 0-1000 but raw widths say Σw = 5000:
        # the band must not exceed the capped band.
        capped = {REQUIRED_REHAB: {"low": 0, "high": 1000}}
        stats = {REQUIRED_REHAB: {"sum_w": 5000.0, "sum_w2": 25_000_000.0}}
        required, _, _ = build_scope_headline_tiers(capped, stats, rho=0.4)
        assert required["low"] == 0
        assert required["high"] == 1000

    def test_containment_monotonicity_and_zero_width_scopes(self):
        capped = {
            REQUIRED_REHAB: {"low": 1000, "high": 9000},
            MARKETABILITY_REHAB: {"low": 4000, "high": 4000},  # zero width
            OPTIONAL_VALUE_ADD: {"low": 2000, "high": 12000},
        }
        stats = {
            REQUIRED_REHAB: {"sum_w": 4000.0, "sum_w2": 4_000_000.0},
            MARKETABILITY_REHAB: {"sum_w": 0.0, "sum_w2": 0.0},
            OPTIONAL_VALUE_ADD: {"sum_w": 5000.0, "sum_w2": 13_000_000.0},
        }
        required, resale, full = build_scope_headline_tiers(capped, stats, rho=0.4)
        for key in ("low", "high"):
            assert required[key] <= resale[key] <= full[key]
        for bucket in (required, resale, full):
            assert 0 <= bucket["sum_low"] <= bucket["low"]
            assert bucket["low"] <= bucket["high"] <= bucket["sum_high"]
            assert bucket["midpoint"] == (bucket["low"] + bucket["high"]) // 2
        # Zero-width marketability shifts the resale band without widening it.
        assert resale["high"] - resale["low"] == required["high"] - required["low"]


# Non-mutation guarantees.
class TestNonMutation:

    def test_original_issues_not_mutated(self):
        catalog, issues, photos = _build_simple_v3()
        # Use a kitchen photo so room_surrogate stamping would have something
        # to do for at least one issue.
        photos = _make_photos(("kitchen", "k1.jpg"), ("living_room", "l1.jpg"))
        issues_snapshot = copy.deepcopy(issues)

        compute_renovation_estimate_v4(
            issues, catalog, photos,
        )

        assert issues == issues_snapshot, (
            "compute_renovation_estimate_v4 must not mutate the caller's "
            "issues_flat (no room_surrogate_id stamping in place)"
        )
        for original in issues:
            assert "room_surrogate_id" not in original or (
                original.get("room_surrogate_id")
                == issues_snapshot[issues.index(original)].get("room_surrogate_id")
            )


# ─── Room-surrogate stamping ─────────────────────────────────────────────────


class TestRoomSurrogateStamping:

    def test_v4_issues_receive_room_surrogate_id(self):
        """A kitchen photo causes the v4 line item to carry kitchen_1."""
        catalog = _make_catalog(
            _make_item("cabinets", estimate=HIGH_ESTIMATE,
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
        )
        issues = [
            _make_issue("cabinets", scene_group="kitchen", photo_key="p001.jpg",
                        issue_id="iss_cab"),
        ]
        photos = _make_photos("kitchen")  # p001.jpg → kitchen_1

        v4 = compute_renovation_estimate_v4(
            issues, catalog, photos,
        )

        v4_lis = _line_items(v4)
        assert len(v4_lis) == 1
        assert v4_lis[0]["room_surrogate_id"] == "kitchen_1"

    def test_same_catalog_item_splits_by_room_surrogate(self):
        """Two bathrooms separated by a bedroom split into bathroom_1/_2 in v4
        but remain merged into a single candidate in v3."""
        catalog = _make_catalog(
            _make_item(
                "bath_tile",
                estimate=BATHROOM_ESTIMATE,
                trade_bucket="bath_tile_walls",
                scope="replace",
            ),
        )
        issues = [
            _make_issue("bath_tile", scene_group="bathroom",
                        photo_key="p001.jpg", issue_id="iss_b1"),
            _make_issue("bath_tile", scene_group="bathroom",
                        photo_key="p003.jpg", issue_id="iss_b2"),
        ]
        photos = _make_photos("bathroom", "bedroom", "bathroom")
        # p001 → bathroom_1, p002 → bedroom_1, p003 → bathroom_2

        v3 = compute_renovation_estimate(issues, catalog)
        v3_lis = _line_items(v3)
        assert len(v3_lis) == 1, (
            "v3 lumps both bathrooms into a single candidate"
        )

        v4 = compute_renovation_estimate_v4(
            issues, catalog, photos,
        )
        v4_lis = _line_items(v4)
        rooms = sorted(li["room_surrogate_id"] for li in v4_lis)
        assert rooms == ["bathroom_1", "bathroom_2"], (
            f"expected v4 to split per bathroom surrogate; got {rooms}"
        )

    def test_non_breaking_scenes_only_falls_back_safely(self):
        """Photos with no breaking scenes open no room surrogate and v4 totals
        match v3. Exterior is the one non-breaking group that still gets an
        identity (the property-level one), which must not disturb the fallback."""
        catalog = _make_catalog(
            _make_item("paint", estimate=MEDIUM_ESTIMATE,
                       trade_bucket="paint", scope="repair"),
        )
        issues = [
            _make_issue("paint", scene_group="hallway", photo_key="h1.jpg",
                        issue_id="iss_p1"),
        ]
        photos = _make_photos(("hallway", "h1.jpg"), ("exterior_front", "e1.jpg"))

        v3 = compute_renovation_estimate(issues, catalog)
        v4 = compute_renovation_estimate_v4(
            issues, catalog, photos,
        )

        assert [s["room_surrogate_id"] for s in v4["room_surrogates"]] == ["exterior_primary"]
        assert v4["totals"] == v3["totals"]
        assert v4["raw_totals"] == v3["raw_totals"]

    def test_hallway_only_photos_open_no_surrogate(self):
        catalog = _make_catalog(
            _make_item("paint", estimate=MEDIUM_ESTIMATE,
                       trade_bucket="paint", scope="repair"),
        )
        issues = [
            _make_issue("paint", scene_group="hallway", photo_key="h1.jpg",
                        issue_id="iss_p1"),
        ]
        photos = _make_photos(("hallway", "h1.jpg"), ("closet", "c1.jpg"))

        v4 = compute_renovation_estimate_v4(issues, catalog, photos)

        assert v4["room_surrogates"] == []


# ─── Pass 2f reuse: integration via line items / audit ───────────────────────


class TestPass2fReuseIntegration:

    def _bath_setup(self):
        catalog = _make_catalog(
            _make_item(
                "bath_tile",
                estimate=BATHROOM_ESTIMATE,
                trade_bucket="bath_tile_walls",
                scope="replace",
            ),
        )
        issues = [
            _make_issue("bath_tile", scene_group="bathroom",
                        photo_key="p001.jpg", issue_id="iss_b1"),
            _make_issue("bath_tile", scene_group="bathroom",
                        photo_key="p003.jpg", issue_id="iss_b2"),
        ]
        photos = _make_photos("bathroom", "bedroom", "bathroom")
        return catalog, issues, photos

    def test_pass_2f_reuse_disabled_when_no_v3_reviewed(self):
        catalog, issues, photos = self._bath_setup()
        v4 = compute_renovation_estimate_v4(
            issues, catalog, photos,
            v3_reviewed_candidates=None,
        )
        audit = v4["pass_2f_reuse_audit"]
        n_lis = len(_line_items(v4))
        assert audit["matched_count"] == 0
        assert audit["ambiguous_count"] == 0
        assert audit["dominant_posture_collapses"] == 0
        assert audit["unmatched_v4_count"] == n_lis
        assert v4["provenance"]["v3_pass_2f_reused"] is False

    def test_pass_2f_subset_match_propagates_to_split_line_items(self):
        """v3 lumped (one candidate, two issues), v4 split (two candidates).
        Both v4 candidates inherit the v3 review fields via subset match."""
        catalog, issues, photos = self._bath_setup()

        # v3 reviewed: extract WITHOUT room surrogates so both bath issues
        # land in a single v3 candidate. _build_reviewed_candidates uses the
        # raw issues (no v4 stamping).
        reviewed = _build_reviewed_candidates(
            catalog, issues, posture="replace",
        )
        # Sanity: v3 lumped to one candidate covering both issue_ids.
        assert len(reviewed) == 1
        assert set(reviewed[0].issue_ids) == {"iss_b1", "iss_b2"}

        v4 = compute_renovation_estimate_v4(
            issues, catalog, photos,
            v3_reviewed_candidates=reviewed,
        )

        audit = v4["pass_2f_reuse_audit"]
        assert audit["matched_count"] == 2
        assert audit["ambiguous_count"] == 0
        assert audit["unmatched_v4_count"] == 0

        for room in ("bathroom_1", "bathroom_2"):
            li = _line_item_by_room(v4, room)
            assert li is not None, f"missing line item for {room}"
            assert li["review_posture"] == "replace"
            assert li["effective_posture"] == "replace"
            assert li["pass_2f_attempted"] is True
            assert li["pass_2f_applied"] is True
            assert li["review_source"] == "pass_2f"

    def test_pass_2f_exact_match(self):
        """Same issue set on v3 and v4 (no room split) → exact match."""
        catalog = _make_catalog(
            _make_item("paint", estimate=MEDIUM_ESTIMATE,
                       trade_bucket="paint", scope="repair"),
        )
        issues = [
            _make_issue("paint", scene_group="hallway", photo_key="h1.jpg",
                        issue_id="iss_p1"),
        ]
        photos = _make_photos(("hallway", "h1.jpg"))

        reviewed = _build_reviewed_candidates(catalog, issues, posture="repair")
        v4 = compute_renovation_estimate_v4(
            issues, catalog, photos,
            v3_reviewed_candidates=reviewed,
        )

        audit = v4["pass_2f_reuse_audit"]
        assert audit["matched_count"] == 1
        assert audit["ambiguous_count"] == 0
        assert audit["unmatched_v4_count"] == 0


# ─── Pass 2f reuse: unit-level helper tests ──────────────────────────────────


class TestPass2fReuseHelper:

    def _make_pair(self):
        """Build a (catalog, issues) pair with a single catalog item
        spanning two issues so candidate extraction can produce a lumped or
        split shape depending on the room_surrogate_id stamping."""
        catalog = _make_catalog(
            _make_item(
                "bath_tile",
                estimate=BATHROOM_ESTIMATE,
                trade_bucket="bath_tile_walls",
                scope="replace",
            ),
        )
        issues = [
            _make_issue("bath_tile", scene_group="bathroom",
                        photo_key="p001.jpg", issue_id="iss_b1"),
            _make_issue("bath_tile", scene_group="bathroom",
                        photo_key="p003.jpg", issue_id="iss_b2"),
        ]
        return catalog, issues

    def test_exact_match_marks_reuse_method(self):
        catalog, issues = self._make_pair()
        v3_reviewed = _build_reviewed_candidates(catalog, issues, posture="replace")
        # v3_reviewed has 1 candidate covering {iss_b1, iss_b2}.

        # v4 candidates with the same issue_ids → exact match.
        v4_candidates = extract_estimate_candidates(issues, catalog)
        v4_candidates = resolve_estimate_units(v4_candidates, issues, catalog)
        assert len(v4_candidates) == 1
        assert set(v4_candidates[0].issue_ids) == {"iss_b1", "iss_b2"}

        audit = _reuse_pass_2f_fields(v4_candidates, v3_reviewed)
        assert audit == {
            "matched_count": 1,
            "unmatched_v4_count": 0,
            "ambiguous_count": 0,
            "dominant_posture_collapses": 0,
        }
        c = v4_candidates[0]
        assert c.review_posture == "replace"
        assert c.pass_2f_reuse_method == "exact"

    def test_subset_match_marks_reuse_method(self):
        catalog, issues = self._make_pair()
        v3_reviewed = _build_reviewed_candidates(catalog, issues, posture="replace")
        # v4: stamp each issue with its own room_surrogate_id so v4 splits
        # into two candidates, each subset of v3's lumped one.
        v4_issues = copy.deepcopy(issues)
        v4_issues[0]["room_surrogate_id"] = "bathroom_1"
        v4_issues[1]["room_surrogate_id"] = "bathroom_2"
        v4_candidates = extract_estimate_candidates(v4_issues, catalog)
        v4_candidates = resolve_estimate_units(v4_candidates, v4_issues, catalog)
        assert len(v4_candidates) == 2

        audit = _reuse_pass_2f_fields(v4_candidates, v3_reviewed)
        assert audit["matched_count"] == 2
        assert audit["ambiguous_count"] == 0
        assert audit["unmatched_v4_count"] == 0

        for c in v4_candidates:
            assert c.review_posture == "replace"
            assert c.pass_2f_reuse_method == "subset"

    def test_ambiguous_when_multiple_v3_supersets(self):
        """Two synthetic v3 candidates (same catalog_item_id) whose issue_ids
        each fully contain a v4 candidate's set → ambiguous, no copy."""
        catalog, issues = self._make_pair()
        # Build v4 candidates from base issues (one v4 candidate covering
        # {iss_b1, iss_b2}).
        v4_candidates = extract_estimate_candidates(issues, catalog)
        v4_candidates = resolve_estimate_units(v4_candidates, issues, catalog)

        # Build two v3 candidates that both contain {iss_b1, iss_b2} but
        # have extra issue_ids unique to each → both are strict supersets.
        v3_a = _build_reviewed_candidates(catalog, issues, posture="replace")[0]
        v3_b = _build_reviewed_candidates(catalog, issues, posture="repair")[0]
        v3_a.issue_ids = ["iss_b1", "iss_b2", "extra_a"]
        v3_b.issue_ids = ["iss_b1", "iss_b2", "extra_b"]

        audit = _reuse_pass_2f_fields(v4_candidates, [v3_a, v3_b])
        assert audit["matched_count"] == 0
        assert audit["ambiguous_count"] == 1
        assert audit["unmatched_v4_count"] == 0

        c = v4_candidates[0]
        assert c.review_posture is None
        assert not hasattr(c, "pass_2f_reuse_method") or c.pass_2f_reuse_method is None

    def test_collapse_with_agreeing_postures_copies(self):
        """Multiple disjoint v3 subsets union to v4's set; postures agree →
        copy with method='collapse_agreeing_postures'."""
        catalog, issues = self._make_pair()
        v4_candidates = extract_estimate_candidates(issues, catalog)
        v4_candidates = resolve_estimate_units(v4_candidates, issues, catalog)
        assert set(v4_candidates[0].issue_ids) == {"iss_b1", "iss_b2"}

        # Two v3 candidates with disjoint issue_ids that union to v4's set.
        v3_a = _build_reviewed_candidates(catalog, issues, posture="replace")[0]
        v3_b = _build_reviewed_candidates(catalog, issues, posture="replace")[0]
        v3_a.issue_ids = ["iss_b1"]
        v3_b.issue_ids = ["iss_b2"]

        audit = _reuse_pass_2f_fields(v4_candidates, [v3_a, v3_b])
        assert audit["matched_count"] == 1
        assert audit["ambiguous_count"] == 0
        assert audit["dominant_posture_collapses"] == 0  # not used in PR 3A
        assert audit["unmatched_v4_count"] == 0

        c = v4_candidates[0]
        assert c.review_posture == "replace"
        assert c.pass_2f_reuse_method == "collapse_agreeing_postures"

    def test_collapse_with_disagreeing_postures_is_ambiguous(self):
        catalog, issues = self._make_pair()
        v4_candidates = extract_estimate_candidates(issues, catalog)
        v4_candidates = resolve_estimate_units(v4_candidates, issues, catalog)

        v3_a = _build_reviewed_candidates(catalog, issues, posture="replace")[0]
        v3_b = _build_reviewed_candidates(catalog, issues, posture="repair")[0]
        v3_a.issue_ids = ["iss_b1"]
        v3_b.issue_ids = ["iss_b2"]

        audit = _reuse_pass_2f_fields(v4_candidates, [v3_a, v3_b])
        assert audit["matched_count"] == 0
        assert audit["ambiguous_count"] == 1
        assert audit["unmatched_v4_count"] == 0

        c = v4_candidates[0]
        assert c.review_posture is None

    def test_unmatched_when_no_v3_for_catalog_item(self):
        catalog, issues = self._make_pair()
        v4_candidates = extract_estimate_candidates(issues, catalog)
        v4_candidates = resolve_estimate_units(v4_candidates, issues, catalog)

        audit = _reuse_pass_2f_fields(v4_candidates, [])
        assert audit["matched_count"] == 0
        assert audit["unmatched_v4_count"] == 1
        assert audit["ambiguous_count"] == 0


def test_stripped_generic_support_still_extracted_via_affinity():
    """Asymmetry guard (Issue 3): a generic whose catalog package tags are
    stripped (and affects_estimate=False) must still reach packaging through
    scene affinity via _extract_package_only_candidates, not silently vanish."""
    catalog = json.loads(Path("tools/issue_catalog.json").read_text(encoding="utf-8"))
    # worn_or_stained_vinyl_linoleum is affects_estimate=False and stripped to
    # package_role="ignore", so it is no longer package-eligible on its own ...
    raw = _real_catalog_item("worn_or_stained_vinyl_linoleum")
    assert not is_package_eligible_catalog_item(raw)
    # ... yet a kitchen-scene occurrence is still extracted as package evidence
    # because (kitchen, worn_or_stained_vinyl_linoleum) has an affinity entry.
    issues = [_make_issue(
        "worn_or_stained_vinyl_linoleum",
        scene_group="kitchen",
        photo_key="k1.jpg",
        issue_id="iss_vinyl",
        room_surrogate_id="kitchen_1",
        estimate_unit_id="kitchen_primary",
    )]
    candidates = _extract_package_only_candidates(issues, catalog)
    assert any(c.catalog_item_id == "worn_or_stained_vinyl_linoleum" for c in candidates)


def _ambient_support_item(item_id):
    """A costed cosmetic support that routes to bedroom_modernization."""
    item = _make_item(
        item_id,
        estimate={
            "estimate_tier": "medium",
            "strategy": "replace_only",
            "group": "bedroom",
            "stack_behavior": "sum",
            "unit_policy": "per_room",
        },
        kind="upgrade",
        trade_bucket="paint_drywall",
        scope="replace",
        cost={
            "mode": "allowance",
            "cost_source": "manual",
            "base_low": 500,
            "base_high": 1500,
            "per_occurrence_low": 500,
            "per_occurrence_high": 1500,
            "cap_low": 5000,
            "cap_high": 15000,
        },
    )
    item.update({
        "category": "cosmetic",
        "display_class": "marketability",
        "package_role": "package_support",
        "package_type": "bedroom_modernization",
        "scene_groups": ["bedroom"],
        "defaultHidden": False,
        "tier": "work",
    })
    return item


def test_demoted_ambient_support_stays_visible_and_costed():
    """MANDATORY safeguard #4: when recurring cross-room supports are demoted
    (no driverless package minted), the underlying findings must still appear as
    costed line items. Mechanism #1 removes the package *vote*, never the *work*.
    """
    # Two cosmetic supports recur across 3 distinct bedroom units (bedrooms are
    # not collapsed by Mechanism #2), so both become ambient and the driverless
    # bedroom packages are suppressed.
    catalog = _make_catalog(
        _ambient_support_item("recurring_ceiling_texture"),
        _ambient_support_item("recurring_wall_trim"),
    )
    issues = []
    for n, key in ((1, "b1.jpg"), (2, "b2.jpg"), (3, "b3.jpg")):
        issues.append(_make_issue(
            "recurring_ceiling_texture", scene_group="bedroom",
            photo_key=key, issue_id=f"rt_{n}", catalog_item_kind="upgrade",
        ))
        issues.append(_make_issue(
            "recurring_wall_trim", scene_group="bedroom",
            photo_key=key, issue_id=f"wt_{n}", catalog_item_kind="upgrade",
        ))
    photos = _make_photos(
        ("bedroom", "b1.jpg"),
        ("bathroom", "x1.jpg"),
        ("bedroom", "b2.jpg"),
        ("bathroom", "x2.jpg"),
        ("bedroom", "b3.jpg"),
    )
    v4 = compute_renovation_estimate_v4(issues, catalog, photos)

    # Mechanism #1 fired: no bedroom package minted from the recurring supports.
    assert not any(
        p.get("package_type") == "bedroom_modernization" for p in v4["packages"]
    )
    assert any(
        s.get("suppression_reason") == "weak_after_ambient_support_demotion"
        for s in v4["suppressed_package_candidates"]
    )

    # Safeguard #4: the demoted findings are still VISIBLE and COSTED.
    demoted_items = [
        li for li in _line_items(v4)
        if li.get("catalog_item_id") in {"recurring_ceiling_texture", "recurring_wall_trim"}
    ]
    assert demoted_items, "demoted supports must still appear as line items"
    assert all(li.get("is_valid_detection") is not False for li in demoted_items)
    assert sum(int(li.get("cost_high") or 0) for li in demoted_items) > 0


def test_project_scope_breakdown_aggregates_line_items_and_packages():
    """Costs come from group line items (not the group dict itself), and a
    scope with line items but no package still appears in the breakdown."""
    groups = [
        {
            "group": "kitchen", "low": 8000, "high": 16000,
            "line_items": [{
                "trade_bucket": "kitchen_cabinets_counters",
                "cost_low": 8000, "cost_high": 16000,
            }],
        },
        {
            "group": "flooring", "low": 1200, "high": 3400,
            "line_items": [{
                "trade_bucket": "flooring",
                "cost_low": 1200, "cost_high": 3400,
            }],
        },
    ]
    packages = [{
        "package_id": "kitchen_modernization__kitchen_primary",
        "absorption_scope": {"trade_buckets": ["kitchen_cabinets_counters"]},
    }]

    breakdown = _build_project_scope_breakdown(groups=groups, packages=packages)

    by_id = {entry["scope_id"]: entry for entry in breakdown}
    kb = by_id["kitchen_bath"]
    assert kb["cost_low"] == 8000
    assert kb["cost_high"] == 16000
    assert kb["item_count"] == 1
    assert kb["trade_buckets"] == ["kitchen_cabinets_counters"]
    assert kb["contributing_package_ids"] == ["kitchen_modernization__kitchen_primary"]

    # scope with line items but NO package still appears
    ig = by_id["interior_generalist"]
    assert ig["cost_low"] == 1200
    assert ig["cost_high"] == 3400
    assert ig["item_count"] == 1
    assert ig["trade_buckets"] == ["flooring"]
    assert ig["contributing_package_ids"] == []


# ─── End-to-end: partial 2f confirmation through final scope totals ──────────


class TestPartialConfirmationEndToEnd:
    """Full-ordering coverage: partial 2f confirmation → confirmed-only
    re-tier → bathroom expansion (per-room re-price) → physical subsumption →
    display-only whole-home aggregate → merged-child splitting + per-unit caps
    + selected-package floor → final scope totals → disposition audit."""

    def _catalog(self):
        bath_driver = _make_item(
            "outdated_bathroom_finishes",
            estimate=BATHROOM_ESTIMATE,
            kind="upgrade",
            severity=3,
            trade_bucket="bathroom_fixtures_tile",
            scope="replace",
        )
        bath_driver.update({
            "display_class": "marketability",
            "package_affinity": {
                "bathroom": {
                    "package_type": "bathroom_modernization",
                    "package_role": "package_driver",
                },
            },
            "scene_groups": ["bathroom"],
        })
        bath_support = _make_item(
            "vintage_tile_pattern_style",
            kind="upgrade",
            severity=2,
            trade_bucket="bathroom_fixtures_tile",
            scope="replace",
        )
        bath_support.update({
            "display_class": "marketability",
            "package_affinity": {
                "bathroom": {
                    "package_type": "bathroom_modernization",
                    "package_role": "package_support",
                },
            },
            "scene_groups": ["bathroom"],
        })
        tile_repair = _make_item(
            "cracked_tile",
            estimate=BATHROOM_ESTIMATE,
            kind="defect",
            severity=3,
            trade_bucket="bathroom_fixtures_tile",
            scope="repair",
        )
        tile_repair.update({
            "package_affinity": {
                "bathroom": {
                    "package_type": "bathroom_repair",
                    "package_role": "package_driver",
                },
            },
            "scene_groups": ["bathroom"],
        })
        kitchen_paint = _make_item(
            "scuffed_kitchen_walls",
            estimate={**MEDIUM_ESTIMATE, "group": "kitchen"},
            kind="defect",
            severity=2,
            trade_bucket="paint_drywall",
            scope="repair",
        )
        kitchen_paint.update({
            "package_affinity": {
                "kitchen": {
                    "package_type": "kitchen_turnover",
                    "package_role": "package_driver",
                },
            },
            "scene_groups": ["kitchen"],
        })
        living_paint = _make_item(
            "scuffed_living_walls",
            estimate=MEDIUM_ESTIMATE,
            kind="defect",
            severity=2,
            trade_bucket="paint_drywall",
            scope="repair",
        )
        living_paint.update({
            "package_affinity": {
                "living": {
                    "package_type": "living_turnover",
                    "package_role": "package_driver",
                },
            },
            "scene_groups": ["living_room"],
        })
        return _make_catalog(
            bath_driver, bath_support, tile_repair, kitchen_paint, living_paint,
        )

    def _inputs(self):
        catalog = self._catalog()
        issues = [
            _make_issue("outdated_bathroom_finishes", scene_group="bathroom",
                        photo_key="b1.jpg", issue_id="iss_bd1",
                        catalog_item_kind="upgrade"),
            _make_issue("vintage_tile_pattern_style", scene_group="bathroom",
                        photo_key="b1.jpg", issue_id="iss_bs1",
                        catalog_item_kind="upgrade"),
            _make_issue("cracked_tile", scene_group="bathroom",
                        photo_key="b1.jpg", issue_id="iss_crack"),
            _make_issue("outdated_bathroom_finishes", scene_group="bathroom",
                        photo_key="b2.jpg", issue_id="iss_bd2",
                        catalog_item_kind="upgrade"),
            _make_issue("vintage_tile_pattern_style", scene_group="bathroom",
                        photo_key="b2.jpg", issue_id="iss_rej",
                        catalog_item_kind="upgrade"),
            _make_issue("scuffed_kitchen_walls", scene_group="kitchen",
                        photo_key="k1.jpg", issue_id="iss_pk"),
            _make_issue("scuffed_living_walls", scene_group="living_areas",
                        photo_key="l1.jpg", issue_id="iss_pl"),
        ]
        # Non-consecutive bathroom photos → two distinct bathroom surrogates.
        photos = _make_photos(
            ("bathroom", "b1.jpg"),
            ("kitchen", "k1.jpg"),
            ("bathroom", "b2.jpg"),
            ("living_room", "l1.jpg"),
        )
        metadata = {"baths": 2, "beds": 3}
        return catalog, issues, photos, metadata

    def _verifications_for(self, package_ids):
        verifications = {}
        for package_id in package_ids:
            if package_id.startswith("bathroom_modernization"):
                verifications[package_id] = {
                    "verification_status": "confirmed",
                    "reviewed_issue_ids": ["iss_bd1", "iss_bs1", "iss_bd2", "iss_rej"],
                    "confirmed_issue_ids": ["iss_bd1", "iss_bs1", "iss_bd2"],
                    "rejected_issue_ids": ["iss_rej"],
                    "visible_room_count": "multiple_rooms",
                    "visible_room_count_evidence": "Two distinct vanities.",
                }
            elif package_id.startswith("bathroom_repair"):
                verifications[package_id] = {
                    "verification_status": "confirmed",
                    "reviewed_issue_ids": ["iss_crack"],
                    "confirmed_issue_ids": ["iss_crack"],
                    "rejected_issue_ids": [],
                }
            elif package_id.startswith("kitchen_turnover"):
                verifications[package_id] = {
                    "verification_status": "confirmed",
                    "reviewed_issue_ids": ["iss_pk"],
                    "confirmed_issue_ids": ["iss_pk"],
                    "rejected_issue_ids": [],
                }
            elif package_id.startswith("living_turnover"):
                verifications[package_id] = {
                    "verification_status": "confirmed",
                    "reviewed_issue_ids": ["iss_pl"],
                    "confirmed_issue_ids": ["iss_pl"],
                    "rejected_issue_ids": [],
                }
        return verifications

    def test_partial_confirmation_through_final_totals(self):
        catalog, issues, photos, metadata = self._inputs()
        # Discovery pass: learn the candidate package ids this fixture forms.
        discovery = compute_renovation_estimate_v4(
            issues, catalog, photos,
            property_metadata=metadata,
            package_verifications={},
        )
        candidate_ids = [
            p["package_id"] for p in discovery["package_candidates"]
        ]
        assert any(i.startswith("bathroom_modernization") for i in candidate_ids)
        assert any(i.startswith("bathroom_repair") for i in candidate_ids)
        assert any(i.startswith("kitchen_turnover") for i in candidate_ids)
        assert any(i.startswith("living_turnover") for i in candidate_ids)

        v4 = compute_renovation_estimate_v4(
            issues, catalog, photos,
            property_metadata=metadata,
            package_verifications=self._verifications_for(candidate_ids),
        )

        packages = {p["package_id"]: p for p in v4["packages"]}

        # 1. Expansion fired and produced per-room, re-priced clones:
        #    bathroom_1 (driver + support) → partial_rehab; bathroom_2
        #    (driver only, its support was rejected) → refresh.
        expansion = v4["bathroom_expansion_audit"]
        assert expansion["expanded"] is True
        clones = [
            p for p in v4["packages"] if p.get("expansion_source_package_id")
        ]
        assert len(clones) == 2
        by_surrogate = {p["room_surrogate_id"]: p for p in clones}
        b1_clone = by_surrogate["bathroom_1"]
        b2_clone = by_surrogate["bathroom_2"]
        assert b1_clone["pricing_tier"] == "partial_rehab"
        assert b2_clone["pricing_tier"] == "refresh"
        assert b1_clone["cost_high"] > b2_clone["cost_high"]
        assert b1_clone["retier_audit"]["applied"] is True
        assert b2_clone["retier_audit"]["applied"] is True
        # The rejected support id never leaks into a clone's supporting set.
        for clone in clones:
            assert "iss_rej" not in clone["supporting_issue_ids"]

        # 2. Physical subsumption: the tile repair (confirmed only in
        #    bathroom_1, tile component covered) is subsumed by exactly the
        #    bathroom_1 clone.
        subsumptions = v4["package_subsumption_audit"]["subsumptions"]
        repair_records = [
            r for r in subsumptions if r["rule"] == "modernization_subsumes_repair"
        ]
        assert len(repair_records) == 1
        assert repair_records[0]["winner_package_id"] == b1_clone["package_id"]
        assert repair_records[0]["loser_confirmed_surrogates"] == ["bathroom_1"]
        assert not any(
            p["package_type"] == "bathroom_repair" for p in v4["packages"]
        )

        # 3. Whole-home turnover aggregate: appended, visible, display-only —
        #    package totals equal the billable packages exactly (no double
        #    count of the aggregate's mirrored range).
        aggregate = packages.get("interior_paint_flooring_refresh__whole_home")
        assert aggregate is not None
        assert aggregate["estimate_display_only"] is True
        assert aggregate["ui_eligible"] is True
        billable = [
            p for p in v4["packages"] if not p.get("estimate_display_only")
        ]
        reconciliation = v4["reconciliation"]
        assert reconciliation["package_total_low"] == sum(
            p["cost_low"] for p in billable
        )
        assert reconciliation["package_total_high"] == sum(
            p["cost_high"] for p in billable
        )

        # 4. The dropped repair's required-scope tile child survives as a
        #    retained line item — required work is never absorbed into the
        #    marketability clones.
        tile_children = [
            child
            for group in v4["groups"]
            for li in group.get("line_items", [])
            for child in li.get("unit_member_allocations", [])
            if "iss_crack" in (child.get("issue_ids") or [])
        ]
        assert tile_children
        assert all(c["absorbed_by_package_id"] is None for c in tile_children)
        assert v4["totals_by_scope_raw"]["required_rehab"]["high"] > 0

        # 5. Bathroom group caps honor the selected-package floor: the two
        #    clones' verified ranges are never clipped away.
        bathroom_group = next(
            g for g in reconciliation["package_group_reconciliation"]
            if g["group"] == "bathroom"
        )
        clone_floor_high = (
            b1_clone["cost_high"] + b2_clone["cost_high"]
            - bathroom_group["absorbed_out_of_group"]["high"]
        )
        assert bathroom_group["post_cap_package_adjusted"]["high"] >= clone_floor_high
        assert bathroom_group["unit_count"] == 2

        # 6. Disposition audit: every confirmed issue kept a priced
        #    representation; the rejected id is recorded as rejected.
        disposition = v4["issue_disposition_audit"]
        assert disposition["confirmed_issues_without_priced_representation"] == []
        by_issue = {r["issue_id"]: r for r in disposition["issues"]}
        assert by_issue["iss_rej"]["disposition"] == "rejected"
        assert by_issue["iss_bd1"]["disposition"] == "priced_in_package"
        assert (
            "confirmed_issues_without_priced_representation"
            not in v4["warnings"]
        )

        # 7. Legacy artifact surface stays intact.
        for key in (
            "packages", "package_candidates", "package_subsumption_audit",
            "bathroom_expansion_audit", "bathroom_room_count_signal",
            "reconciliation", "totals_by_scope_raw", "totals_by_scope_capped",
            "final_rehab", "final_rehab_required", "final_rehab_resale_ready",
            "final_rehab_full_renewal", "project_scope_breakdown",
        ):
            assert key in v4, key
        _assert_nested_scope_tiers(v4)
