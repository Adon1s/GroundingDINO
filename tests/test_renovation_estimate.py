"""
Tests for tools.renovation_estimate — primary renovation cost estimation engine.
"""

import asyncio
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import json

import pytest


def _run_async(coro):
    """Run a coroutine without clearing the legacy default event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


from tools.renovation_estimate import (
    ESTIMATE_DEFAULTS,
    INSPECT_ALLOWANCE,
    PASS_2F_FALLBACK_RETIRED,
    CatalogEstimateMeta,
    EstimateCandidate,
    _select_representative_image,
    compute_group_estimate,
    compute_renovation_estimate,
    extract_estimate_candidates,
    resolve_effective_posture,
    resolve_estimate_meta,
    resolve_estimate_units,
    resolve_pass_2f_model_config,
    resolve_pricing_band,
    run_pass_2f_batch,
)
from tools.scene_classifier_passes import (
    PASS_2F_KITCHEN_SYSTEM_PROMPT,
    PASS_2F_KITCHEN_USER_PROMPT,
    _coerce_pass_2f,
    run_pass_2e,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_catalog(*items):
    """Wrap items into a minimal catalog dict."""
    return {"items": list(items), "trade_buckets": []}


def _make_item(item_id, *, estimate=None, kind="defect", severity=3,
               scope="repair", trade_bucket="flooring", name=None,
               cost=None):
    """Build a minimal catalog item dict."""
    item = {
        "id": item_id,
        "name": name or item_id.replace("_", " ").title(),
        "kind": kind,
        "severity": severity,
        "scope": scope,
        "trade_bucket": trade_bucket,
        "cost": cost or {"mode": "heuristic"},
    }
    if estimate is not None:
        item["estimate"] = estimate
    return item


def _make_issue(catalog_item_id, *, scene_group="kitchen", photo_key="img_001.jpg",
                issue_id=None, **extra):
    """Build a minimal issues_flat entry."""
    issue = {
        "issue_id": issue_id or f"iss_{catalog_item_id}_{photo_key}",
        "catalog_item_id": catalog_item_id,
        "catalog_item_kind": "defect",
        "scene_group": scene_group,
        "photo_key": photo_key,
        "description": "test",
        "label": "defect_or_damage",
    }
    issue.update(extra)
    return issue


HIGH_ESTIMATE = {
    "estimate_tier": "high",
    "strategy": "replace_only",
    "group": "kitchen",
    "stack_behavior": "group_cap",
}

MEDIUM_ESTIMATE = {
    "estimate_tier": "medium",
    "strategy": "repair_only",
    "group": "other",
    "stack_behavior": "sum",
}

MINOR_ESTIMATE = {
    "estimate_tier": "minor",
}


# ─── Phase 1: resolve_estimate_meta ──────────────────────────────────────────

class TestResolveEstimateMeta:

    def test_defaults_from_none(self):
        meta = resolve_estimate_meta(None)
        assert meta == ESTIMATE_DEFAULTS
        assert meta.estimate_tier == "minor"
        assert meta.group == "other"

    def test_defaults_from_empty_dict(self):
        assert resolve_estimate_meta({}) == ESTIMATE_DEFAULTS

    def test_full_parse(self):
        raw = {
            "estimate_tier": "high",
            "strategy": "repair_or_replace",
            "group": "kitchen",
            "stack_behavior": "group_cap",
        }
        meta = resolve_estimate_meta(raw)
        assert meta.estimate_tier == "high"
        assert meta.strategy == "repair_or_replace"
        assert meta.group == "kitchen"
        assert meta.stack_behavior == "group_cap"

    def test_unit_policy_defaults_to_per_scope(self):
        meta = resolve_estimate_meta({"estimate_tier": "medium"})
        assert meta.unit_policy == "per_scope"

    def test_unit_policy_parses_known_policy(self):
        meta = resolve_estimate_meta({
            "estimate_tier": "medium",
            "unit_policy": "per_room",
        })
        assert meta.unit_policy == "per_room"

    def test_invalid_unit_policy_defaults_to_per_scope(self):
        meta = resolve_estimate_meta({
            "estimate_tier": "medium",
            "unit_policy": "per_photo",
        })
        assert meta.unit_policy == "per_scope"

    def test_partial_parse_fills_defaults(self):
        meta = resolve_estimate_meta({"estimate_tier": "medium"})
        assert meta.estimate_tier == "medium"
        assert meta.strategy == "repair_only"

    def test_invalid_tier_defaults_to_minor(self):
        meta = resolve_estimate_meta({"estimate_tier": "ultra"})
        assert meta.estimate_tier == "minor"


# ─── Phase 4: extract_estimate_candidates ────────────────────────────────────

class TestExtractCandidates:

    def test_empty_issues(self):
        catalog = _make_catalog(_make_item("a", estimate=HIGH_ESTIMATE))
        assert extract_estimate_candidates([], catalog) == []

    def test_filters_items_without_estimate(self):
        """Items without estimate block should be excluded (defaults to minor)."""
        catalog = _make_catalog(_make_item("a"))  # no estimate
        issues = [_make_issue("a")]
        assert extract_estimate_candidates(issues, catalog) == []

    def test_filters_minor_tier(self):
        """Minor-tier items are excluded from quick estimate."""
        catalog = _make_catalog(_make_item("a", estimate=MINOR_ESTIMATE))
        issues = [_make_issue("a")]
        assert extract_estimate_candidates(issues, catalog) == []

    def test_includes_high_tier(self):
        catalog = _make_catalog(_make_item("a", estimate=HIGH_ESTIMATE))
        issues = [_make_issue("a")]
        result = extract_estimate_candidates(issues, catalog)
        assert len(result) == 1
        assert result[0].catalog_item_id == "a"
        assert result[0].occurrences == 1
        assert result[0].estimate_meta.estimate_tier == "high"

    def test_includes_medium_tier(self):
        catalog = _make_catalog(_make_item("a", estimate=MEDIUM_ESTIMATE))
        issues = [_make_issue("a")]
        result = extract_estimate_candidates(issues, catalog)
        assert len(result) == 1
        assert result[0].estimate_meta.estimate_tier == "medium"

    def test_agreed_catalog_items_are_estimate_eligible(self):
        catalog = json.loads(Path("tools/issue_catalog.json").read_text())
        agreed_ids = [
            "visible_mold_or_mildew",
            "ceiling_cracks_or_sagging",
            "retaining_wall_failure_or_missing_section",
            "water_stain_ceiling",
            "bare_or_missing_finish_flooring",
        ]
        issues = [
            _make_issue(item_id, scene_group="exterior" if "retaining" in item_id else "living_areas")
            for item_id in agreed_ids
        ]
        candidates = extract_estimate_candidates(issues, catalog)
        by_id = {c.catalog_item_id: c for c in candidates}

        assert set(agreed_ids).issubset(by_id)
        assert by_id["visible_mold_or_mildew"].estimate_meta.group == "remediation"
        assert by_id["water_stain_ceiling"].estimate_meta.group == "remediation"
        assert by_id["retaining_wall_failure_or_missing_section"].estimate_meta.strategy == "inspect_only"
        assert by_id["bare_or_missing_finish_flooring"].estimate_meta.unit_policy == "per_room"

    def test_occurrence_counting(self):
        catalog = _make_catalog(_make_item("a", estimate=HIGH_ESTIMATE))
        issues = [
            _make_issue("a", photo_key="img_001.jpg"),
            _make_issue("a", photo_key="img_002.jpg"),
            _make_issue("a", photo_key="img_003.jpg"),
        ]
        result = extract_estimate_candidates(issues, catalog)
        assert result[0].occurrences == 3
        assert result[0].estimate_unit_count == 1
        assert len(result[0].photo_keys) == 3

    def test_same_scene_photos_are_one_estimate_unit(self):
        catalog = _make_catalog(_make_item("flooring", estimate={
            "estimate_tier": "medium",
            "strategy": "repair_only",
            "group": "flooring",
            "stack_behavior": "sum",
        }))
        issues = [
            _make_issue("flooring", scene_group="living_areas", photo_key="living_1.jpg"),
            _make_issue("flooring", scene_group="living_areas", photo_key="living_2.jpg"),
            _make_issue("flooring", scene_group="living_areas", photo_key="living_3.jpg"),
        ]

        candidates = extract_estimate_candidates(issues, catalog)

        assert len(candidates) == 1
        assert candidates[0].occurrences == 3
        assert candidates[0].estimate_unit_count == 1
        assert candidates[0].distinct_photo_count == 3

    def test_distinct_scene_groups_create_estimate_units(self):
        catalog = _make_catalog(_make_item("flooring", estimate={
            "estimate_tier": "medium",
            "strategy": "repair_only",
            "group": "flooring",
            "stack_behavior": "sum",
        }))
        issues = [
            _make_issue("flooring", scene_group="kitchen", photo_key="kitchen.jpg"),
            _make_issue("flooring", scene_group="bathroom", photo_key="bath.jpg"),
            _make_issue("flooring", scene_group="bedrooms", photo_key="bed.jpg"),
        ]

        candidates = extract_estimate_candidates(issues, catalog)

        assert len(candidates) == 3
        assert sum(c.estimate_unit_count for c in candidates) == 3

    def test_same_scene_photo_evidence_does_not_multiply_cost(self):
        estimate = {
            "estimate_tier": "medium",
            "strategy": "repair_only",
            "group": "flooring",
            "stack_behavior": "sum",
        }
        cost = {
            "mode": "allowance",
            "base_low": 1000,
            "base_high": 2000,
            "per_occurrence_low": 500,
            "per_occurrence_high": 1000,
            "cap_low": 5000,
            "cap_high": 10000,
        }
        catalog = _make_catalog(_make_item("flooring", estimate=estimate, cost=cost))
        one_photo = [_make_issue("flooring", scene_group="living_areas", photo_key="living_1.jpg")]
        three_photos = [
            _make_issue("flooring", scene_group="living_areas", photo_key="living_1.jpg"),
            _make_issue("flooring", scene_group="living_areas", photo_key="living_2.jpg"),
            _make_issue("flooring", scene_group="living_areas", photo_key="living_3.jpg"),
        ]

        one = compute_renovation_estimate(one_photo, catalog)
        three = compute_renovation_estimate(three_photos, catalog)

        assert one["raw_totals"] == three["raw_totals"]
        assert three["groups"][0]["line_items"][0]["occurrences"] == 3
        assert three["groups"][0]["line_items"][0]["estimate_unit_count"] == 1

    def test_sort_high_tier_first(self):
        """High-tier items sort before medium even if medium has higher severity."""
        est_high = {**HIGH_ESTIMATE}
        est_medium = {**MEDIUM_ESTIMATE, "group": "kitchen", "stack_behavior": "group_cap"}
        catalog = _make_catalog(
            _make_item("medium_item", estimate=est_medium, severity=4),
            _make_item("high_item", estimate=est_high, severity=1),
        )
        issues = [_make_issue("medium_item"), _make_issue("high_item")]
        result = extract_estimate_candidates(issues, catalog)
        assert result[0].catalog_item_id == "high_item"


class TestResolveEstimateUnits:

    def _resolve(self, item_id, estimate, issues, *, trade_bucket="flooring", cost=None):
        catalog = _make_catalog(
            _make_item(item_id, estimate=estimate, trade_bucket=trade_bucket, cost=cost),
        )
        candidates = extract_estimate_candidates(issues, catalog)
        return resolve_estimate_units(candidates, issues, catalog), catalog

    def test_per_property_merges_candidates_with_evidence(self):
        estimate = {
            "estimate_tier": "high",
            "strategy": "repair_or_replace",
            "group": "roof",
            "stack_behavior": "max_only",
            "unit_policy": "per_property",
        }
        issues = [
            _make_issue("roof", scene_group="exterior", photo_key="front.jpg", issue_id="roof_1"),
            _make_issue("roof", scene_group="other", photo_key="rear.jpg", issue_id="roof_2"),
        ]

        resolved, _ = self._resolve("roof", estimate, issues, trade_bucket="roof_gutters")

        assert len(resolved) == 1
        c = resolved[0]
        assert c.estimate_unit_count == 1
        assert c.occurrences == 2
        assert c.estimate_meta.unit_policy == "per_property"
        assert c.unit_resolution_method == "property_single_unit"
        assert c.resolved_cluster_key == "catalog:roof|unit_policy:per_property"
        assert c.source_issue_ids == ["roof_1", "roof_2"]
        assert sorted(c.photo_keys) == ["front.jpg", "rear.jpg"]

    def test_per_room_counts_distinct_rooms_and_keeps_members(self):
        estimate = {
            "estimate_tier": "medium",
            "strategy": "repair_only",
            "group": "flooring",
            "stack_behavior": "sum",
            "unit_policy": "per_room",
        }
        issues = [
            _make_issue("flooring", scene_group="kitchen", photo_key="k1.jpg",
                        room_surrogate_id="kitchen", issue_id="floor_1"),
            _make_issue("flooring", scene_group="bedroom", photo_key="b1.jpg",
                        room_surrogate_id="bedroom", issue_id="floor_2"),
            _make_issue("flooring", scene_group="bedroom", photo_key="b2.jpg",
                        room_surrogate_id="bedroom", issue_id="floor_3"),
        ]

        resolved, _ = self._resolve("flooring", estimate, issues)

        assert len(resolved) == 1
        c = resolved[0]
        assert c.estimate_unit_count == 2
        assert c.estimate_unit_label == "2 rooms"
        assert c.unit_resolution_method == "distinct_room_surrogates"
        assert c.unit_resolution_confidence == "medium"
        assert c.room_surrogate_id == "multiple_rooms"
        assert [m["unit_key"] for m in c.unit_members] == ["bedroom", "kitchen"]
        assert all(m["counts_toward_estimate"] for m in c.unit_members)

    def test_grouping_preserves_consistency_flag_union(self):
        estimate = {
            "estimate_tier": "medium",
            "strategy": "repair_only",
            "group": "flooring",
            "stack_behavior": "sum",
            "unit_policy": "per_room",
        }
        catalog = _make_catalog(_make_item("flooring", estimate=estimate))
        issues = [
            _make_issue("flooring", scene_group="kitchen", photo_key="k1.jpg",
                        room_surrogate_id="kitchen", issue_id="floor_1"),
            _make_issue("flooring", scene_group="bedroom", photo_key="b1.jpg",
                        room_surrogate_id="bedroom", issue_id="floor_2"),
        ]
        candidates = extract_estimate_candidates(issues, catalog)
        candidates[0].review_consistency_flags = ["unknown_replace"]
        candidates[1].review_consistency_flags = [
            "room_wide_repair",
            "unknown_replace",
        ]

        resolved = resolve_estimate_units(candidates, issues, catalog)

        assert len(resolved) == 1
        assert resolved[0].review_consistency_flags == [
            "room_wide_repair",
            "unknown_replace",
        ]

    def test_per_room_generic_surrogates_fallback_to_one(self):
        estimate = {
            "estimate_tier": "medium",
            "strategy": "repair_only",
            "group": "flooring",
            "stack_behavior": "sum",
            "unit_policy": "per_room",
        }
        issues = [
            _make_issue("flooring", scene_group="other", photo_key="o1.jpg",
                        room_surrogate_id="other", issue_id="floor_1"),
            _make_issue("flooring", scene_group="other", photo_key="o2.jpg",
                        room_surrogate_id="property", issue_id="floor_2"),
        ]

        resolved, _ = self._resolve("flooring", estimate, issues)

        assert len(resolved) == 1
        assert resolved[0].estimate_unit_count == 1
        assert resolved[0].unit_resolution_method == "generic_room_fallback"
        assert resolved[0].unit_resolution_confidence == "low"

    def test_per_kitchen_and_per_bathroom_count_matching_surrogates(self):
        kitchen_estimate = {
            "estimate_tier": "high",
            "strategy": "replace_only",
            "group": "kitchen",
            "stack_behavior": "group_cap",
            "unit_policy": "per_kitchen",
        }
        kitchen_issues = [
            _make_issue("cabinets", scene_group="kitchen", photo_key="k1.jpg",
                        room_surrogate_id="kitchen", issue_id="kit_1"),
            _make_issue("cabinets", scene_group="kitchen", photo_key="k2.jpg",
                        room_surrogate_id="basement_kitchen", issue_id="kit_2"),
        ]
        kitchens, _ = self._resolve(
            "cabinets", kitchen_estimate, kitchen_issues,
            trade_bucket="kitchen_cabinets_counters",
        )

        bath_estimate = {
            "estimate_tier": "medium",
            "strategy": "replace_only",
            "group": "bathroom",
            "stack_behavior": "group_cap",
            "unit_policy": "per_bathroom",
        }
        bath_issues = [
            _make_issue("bath", scene_group="other", photo_key="x1.jpg",
                        room_surrogate_id="other", issue_id="bath_1"),
        ]
        baths, _ = self._resolve(
            "bath", bath_estimate, bath_issues,
            trade_bucket="bathroom_fixtures_tile",
        )

        assert kitchens[0].estimate_unit_count == 2
        assert kitchens[0].unit_resolution_method == "distinct_kitchen_surrogates"
        assert baths[0].estimate_unit_count == 1
        assert baths[0].unit_resolution_method == "single_bathroom_fallback"

    def test_per_opening_avoids_photo_or_description_multipliers(self):
        estimate = {
            "estimate_tier": "medium",
            "strategy": "replace_only",
            "group": "windows_doors",
            "stack_behavior": "sum",
            "unit_policy": "per_opening",
        }
        issues = [
            _make_issue("window", scene_group="living_areas", photo_key="w1.jpg",
                        room_surrogate_id="living_room", issue_id="win_1",
                        description="Fogged window in living room."),
            _make_issue("window", scene_group="living_areas", photo_key="w2.jpg",
                        room_surrogate_id="living_room", issue_id="win_2",
                        description="Broken glass on living room window."),
        ]

        resolved, _ = self._resolve(
            "window", estimate, issues, trade_bucket="trim_doors_windows",
        )

        assert len(resolved) == 1
        assert resolved[0].estimate_unit_count == 1
        assert resolved[0].unit_resolution_method == "conservative_single_opening"
        assert resolved[0].distinct_photo_count == 2

    def test_per_opening_allows_explicit_instance_ids(self):
        estimate = {
            "estimate_tier": "medium",
            "strategy": "replace_only",
            "group": "windows_doors",
            "stack_behavior": "sum",
            "unit_policy": "per_opening",
        }
        issues = [
            _make_issue("window", scene_group="living_areas", photo_key="w1.jpg",
                        room_surrogate_id="living_room", issue_id="win_1",
                        window_id="front_window"),
            _make_issue("window", scene_group="living_areas", photo_key="w2.jpg",
                        room_surrogate_id="living_room", issue_id="win_2",
                        window_id="side_window"),
        ]

        resolved, _ = self._resolve(
            "window", estimate, issues, trade_bucket="trim_doors_windows",
        )

        assert resolved[0].estimate_unit_count == 2
        assert resolved[0].unit_resolution_method == "explicit_opening_ids"

    def test_per_system_and_per_area_are_conservative_single_units(self):
        system_estimate = {
            "estimate_tier": "high",
            "strategy": "inspect_only",
            "group": "electrical",
            "stack_behavior": "max_only",
            "unit_policy": "per_system",
        }
        system_issues = [
            _make_issue("service", scene_group="utility", photo_key="s1.jpg"),
            _make_issue("service", scene_group="utility", photo_key="s2.jpg"),
        ]
        systems, _ = self._resolve("service", system_estimate, system_issues,
                                   trade_bucket="electrical")

        area_estimate = {
            "estimate_tier": "medium",
            "strategy": "repair_or_replace",
            "group": "exterior",
            "stack_behavior": "sum",
            "unit_policy": "per_area",
        }
        area_issues = [
            _make_issue("siding", scene_group="exterior", photo_key="a1.jpg"),
            _make_issue("siding", scene_group="other", photo_key="a2.jpg"),
        ]
        areas, _ = self._resolve("siding", area_estimate, area_issues,
                                 trade_bucket="exterior_siding_trim")

        assert systems[0].estimate_unit_count == 1
        assert systems[0].unit_resolution_method == "system_default_single_unit"
        assert areas[0].estimate_unit_count == 1
        assert areas[0].unit_resolution_method == "area_default_single_unit"

    def test_per_scope_preserves_extracted_units(self):
        estimate = {
            "estimate_tier": "medium",
            "strategy": "repair_only",
            "group": "flooring",
            "stack_behavior": "sum",
        }
        issues = [
            _make_issue("flooring", scene_group="kitchen", photo_key="k.jpg"),
            _make_issue("flooring", scene_group="bathroom", photo_key="b.jpg"),
            _make_issue("flooring", scene_group="bedroom", photo_key="r.jpg"),
        ]

        resolved, _ = self._resolve("flooring", estimate, issues)

        assert len(resolved) == 3
        assert sum(c.estimate_unit_count for c in resolved) == 3
        assert {c.unit_resolution_method for c in resolved} == {"extracted_scope"}

    def test_per_room_pricing_uses_base_plus_additional_units(self):
        estimate = {
            "estimate_tier": "medium",
            "strategy": "repair_only",
            "group": "flooring",
            "stack_behavior": "sum",
            "unit_policy": "per_room",
        }
        cost = {
            "mode": "allowance",
            "base_low": 1000,
            "base_high": 2000,
            "per_occurrence_low": 500,
            "per_occurrence_high": 1000,
            "cap_low": 5000,
            "cap_high": 10000,
        }
        catalog = _make_catalog(_make_item(
            "flooring", estimate=estimate, cost=cost,
            trade_bucket="safety_general",
        ))
        issues = [
            _make_issue("flooring", scene_group="kitchen", photo_key="k.jpg",
                        room_surrogate_id="kitchen"),
            _make_issue("flooring", scene_group="bedroom", photo_key="b.jpg",
                        room_surrogate_id="bedroom"),
            _make_issue("flooring", scene_group="living_areas", photo_key="l.jpg",
                        room_surrogate_id="living_room"),
        ]

        result = compute_renovation_estimate(issues, catalog)

        assert result["meta"]["candidate_count"] == 1
        assert result["meta"]["estimate_unit_count"] == 3
        li = result["groups"][0]["line_items"][0]
        assert li["estimate_unit_count"] == 3
        assert li["unit_policy"] == "per_room"
        assert li["unit_resolution_method"] == "distinct_room_surrogates"
        assert len(li["unit_members"]) == 3
        assert li["cost_high"] == 4000
        summary_item = result["tier_summary"]["unreviewed_items"][0]
        assert summary_item["unit_policy"] == "per_room"
        assert summary_item["resolved_cluster_key"] == "catalog:flooring|unit_policy:per_room"
        audit_item = result["pass_2f_review_audit"]["items"][0]
        assert audit_item["unit_members"] == li["unit_members"]
        assert audit_item["source_issue_ids"] == li["source_issue_ids"]


class TestPass2eCanonicalDisplaySplit:

    def test_optional_suppression_keeps_canonical_issue(self):
        issues = [{
            "issue_id": "iss_1",
            "kind": "upgrade",
            "description": "Dated but functional light fixture.",
            "catalogItemId": "dated_lighting",
        }]
        context = {
            "catalog_meta_by_id": {
                "dated_lighting": {"tier": "optional"},
            },
            "policy": {"include_optional": False},
            "deny_phrases": [],
        }

        result = _run_async(
            run_pass_2e(
                vlm_client=None,
                model_config={},
                verified_issues=issues,
                context=context,
            )
        )

        assert len(result.canonical_issues) == 1
        assert result.display_issues == []
        assert result.display_suppressed_issues[0]["suppressed_reason"] == "tier_optional_suppressed"

    def test_catalog_id_dedupe_keeps_distinct_observations(self):
        issues = [
            {
                "issue_id": "iss_floor_1",
                "kind": "defect",
                "description": "Scratched wood flooring near the living room entry.",
                "catalogItemId": "scratched_flooring",
            },
            {
                "issue_id": "iss_floor_2",
                "kind": "defect",
                "description": "Worn flooring visible in the kitchen walkway.",
                "catalogItemId": "scratched_flooring",
            },
        ]

        result = _run_async(
            run_pass_2e(
                vlm_client=None,
                model_config={},
                verified_issues=issues,
                context={"deny_phrases": [], "policy": {"include_optional": False}},
            )
        )

        assert len(result.canonical_issues) == 2
        assert len(result.display_issues) == 2


# ─── Phase 5: group estimate engine ──────────────────────────────────────────

class TestGroupEstimate:

    def _candidate(self, item_id="a", *, strategy="replace_only",
                   stack_behavior="sum", estimate_tier="medium", group="kitchen",
                   cost_obj=None, occurrences=1, severity=3, scope="replace",
                   trade_bucket="flooring"):
        meta = CatalogEstimateMeta(
            estimate_tier=estimate_tier,
            strategy=strategy,
            group=group,
            stack_behavior=stack_behavior,
        )
        return EstimateCandidate(
            catalog_item_id=item_id,
            catalog_item_name=item_id.replace("_", " ").title(),
            estimate_meta=meta,
            kind="defect",
            severity=severity,
            scope=scope,
            trade_bucket=trade_bucket,
            cost_obj=cost_obj or {
                "mode": "allowance",
                "base_low": 500,
                "base_high": 5000,
                "per_occurrence_low": 300,
                "per_occurrence_high": 900,
                "cap_low": 5000,
                "cap_high": 12000,
            },
            occurrences=occurrences,
            scene_groups_seen=["kitchen"],
            photo_keys=["img_001.jpg"],
            issue_ids=["iss_1"],
        )

    def test_empty_group(self):
        result = compute_group_estimate("kitchen", [])
        assert result["low"] == 0
        assert result["high"] == 0
        assert result["item_count"] == 0

    def test_sum_behavior(self):
        """Two sum items should add together."""
        c1 = self._candidate("a", stack_behavior="sum")
        c2 = self._candidate("b", stack_behavior="sum")
        result = compute_group_estimate("kitchen", [c1, c2])
        assert result["stack_behavior"] == "sum"
        assert result["low"] == result["raw_sum_low"]
        assert result["high"] == result["raw_sum_high"]
        assert result["item_count"] == 2

    def test_group_cap_behavior(self):
        """group_cap should cap at GROUP_BUDGET_CAPS, floor at highest item."""
        # Create items whose raw sum would exceed kitchen cap ($35k)
        big_cost = {
            "mode": "allowance",
            "base_low": 5000, "base_high": 20000,
            "per_occurrence_low": 1000, "per_occurrence_high": 5000,
            "cap_low": 20000, "cap_high": 50000,
        }
        c1 = self._candidate("a", stack_behavior="group_cap", cost_obj=big_cost,
                             trade_bucket="kitchen_cabinets_counters", scope="replace")
        c2 = self._candidate("b", stack_behavior="group_cap", cost_obj=big_cost,
                             trade_bucket="kitchen_cabinets_counters", scope="replace")
        result = compute_group_estimate("kitchen", [c1, c2])
        assert result["stack_behavior"] == "group_cap"
        # Cap should be applied: high should not exceed 35000
        assert result["high"] <= 35000
        # But should be at least the single highest item
        assert result["high"] >= result["line_items"][0]["cost_high"]

    def test_max_only_behavior(self):
        """max_only should take only the single highest-cost item."""
        small_cost = {
            "mode": "allowance",
            "base_low": 100, "base_high": 500,
            "per_occurrence_low": 50, "per_occurrence_high": 100,
            "cap_low": 500, "cap_high": 1000,
        }
        big_cost = {
            "mode": "allowance",
            "base_low": 2000, "base_high": 10000,
            "per_occurrence_low": 1000, "per_occurrence_high": 3000,
            "cap_low": 10000, "cap_high": 25000,
        }
        c_small = self._candidate("small", stack_behavior="max_only",
                                  cost_obj=small_cost, group="roof",
                                  trade_bucket="roof_gutters")
        c_big = self._candidate("big", stack_behavior="max_only",
                                cost_obj=big_cost, group="roof",
                                trade_bucket="roof_gutters")
        result = compute_group_estimate("roof", [c_small, c_big])
        assert result["stack_behavior"] == "max_only"
        # Should use the big item only
        big_line = next(li for li in result["line_items"] if li["catalog_item_id"] == "big")
        assert result["high"] == big_line["cost_high"]
        assert result["low"] == big_line["cost_low"]

    def test_inspect_only_produces_allowance(self):
        """inspect_only items should produce INSPECT_ALLOWANCE, not real costs."""
        c = self._candidate("foundation", strategy="inspect_only",
                            stack_behavior="max_only", group="structure",
                            trade_bucket="foundation_structure")
        result = compute_group_estimate("structure", [c])
        assert result["inspection_only"] is True
        assert result["low"] == INSPECT_ALLOWANCE[0]
        assert result["high"] == INSPECT_ALLOWANCE[1]

    def test_dominant_stack_priority(self):
        """max_only should dominate over group_cap over sum."""
        c1 = self._candidate("a", stack_behavior="sum")
        c2 = self._candidate("b", stack_behavior="group_cap")
        result = compute_group_estimate("kitchen", [c1, c2])
        assert result["stack_behavior"] == "group_cap"

        c3 = self._candidate("c", stack_behavior="max_only")
        result2 = compute_group_estimate("kitchen", [c1, c2, c3])
        assert result2["stack_behavior"] == "max_only"


# ─── Phase 6: end-to-end compute_renovation_estimate ─────────────────────────────

class TestComputeQuickEstimate:

    def test_end_to_end(self):
        kitchen_est = {
            "estimate_tier": "high",
            "strategy": "replace_only",
            "group": "kitchen",
            "stack_behavior": "group_cap",
        }
        roof_est = {
            "estimate_tier": "high",
            "strategy": "repair_or_replace",
            "group": "roof",
            "stack_behavior": "max_only",
        }
        catalog = _make_catalog(
            _make_item("cabinets", estimate=kitchen_est,
                       trade_bucket="kitchen_cabinets_counters", scope="replace",
                       cost={"mode": "allowance", "base_low": 500, "base_high": 10000,
                             "per_occurrence_low": 750, "per_occurrence_high": 2100,
                             "cap_low": 10000, "cap_high": 30000, "cost_source": "manual"}),
            _make_item("roof", estimate=roof_est,
                       trade_bucket="roof_gutters", scope="repair",
                       cost={"mode": "allowance", "base_low": 500, "base_high": 10000,
                             "per_occurrence_low": 600, "per_occurrence_high": 2250,
                             "cap_low": 10000, "cap_high": 25000, "cost_source": "manual"}),
            _make_item("paint", kind="defect"),  # no estimate — should be excluded
        )
        issues = [
            _make_issue("cabinets", scene_group="kitchen", photo_key="k1.jpg"),
            _make_issue("cabinets", scene_group="kitchen", photo_key="k2.jpg"),
            _make_issue("roof", scene_group="exterior", photo_key="e1.jpg"),
            _make_issue("paint", scene_group="living_areas", photo_key="l1.jpg"),
        ]

        result = compute_renovation_estimate(issues, catalog)

        assert result["version"] == "renovation_estimate_v3"
        assert result["meta"]["candidate_count"] == 2
        assert result["meta"]["high_tier_count"] == 2
        assert result["meta"]["groups_active"] == 2
        assert result["raw_totals"]["low"] > 0
        assert result["raw_totals"]["high"] > result["raw_totals"]["low"]

        group_names = [g["group"] for g in result["groups"]]
        assert "kitchen" in group_names
        assert "roof" in group_names

        assert "primary_estimate" in result
        assert "tier_summary" in result
        assert result["primary_estimate"]["source"] == "probable_total"
        assert result["primary_estimate"]["low"] == result["totals"]["probable_total"]["low"]
        assert result["primary_estimate"]["high"] == result["totals"]["probable_total"]["high"]
        assert result["tier_summary"]["version"] == "tier_summary_v1"

    def test_backward_compat_no_estimate_blocks(self):
        """Catalog without any estimate blocks → empty quick estimate."""
        catalog = _make_catalog(
            _make_item("a"),
            _make_item("b"),
        )
        issues = [_make_issue("a"), _make_issue("b")]
        result = compute_renovation_estimate(issues, catalog)
        assert result["meta"]["candidate_count"] == 0
        assert result["raw_totals"]["low"] == 0
        assert result["raw_totals"]["high"] == 0
        assert result["groups"] == []

    def test_empty_issues(self):
        catalog = _make_catalog(_make_item("a", estimate=HIGH_ESTIMATE))
        result = compute_renovation_estimate([], catalog)
        assert result["meta"]["candidate_count"] == 0

    def test_inspect_only_separates_inspection_allowance_and_risk_exposure(self):
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
            _make_item("foundation", estimate=est, trade_bucket="foundation_structure",
                       scope="repair", cost=manual_cost),
        )
        issues = [_make_issue("foundation", scene_group="exterior")]
        result = compute_renovation_estimate(issues, catalog)
        li = result["groups"][0]["line_items"][0]

        assert (li["cost_low"], li["cost_high"]) == INSPECT_ALLOWANCE
        assert (li["risk_exposure_low"], li["risk_exposure_high"]) == (2000, 40000)
        assert result["totals"]["probable_total"] == {
            "low": INSPECT_ALLOWANCE[0],
            "high": INSPECT_ALLOWANCE[1],
        }
        assert result["totals"]["risk_exposure_total"] == {"low": 2000, "high": 40000}


# ─── Phase A/B: 2f readiness tests ──────────────────────────────────────────

class TestRevisitPassReadiness:

    def test_line_items_include_posture_audit_trail(self):
        """Line items should expose estimate_tier and posture audit trail."""
        catalog = _make_catalog(_make_item("cab", estimate=HIGH_ESTIMATE,
                                           trade_bucket="kitchen_cabinets_counters",
                                           scope="replace"))
        issues = [_make_issue("cab")]
        result = compute_renovation_estimate(issues, catalog)
        li = result["groups"][0]["line_items"][0]
        assert li["estimate_tier"] == "high"
        assert li["default_posture"] == "replace_only"
        assert li["review_posture"] is None
        assert li["pricing_posture"] is None
        assert li["effective_posture"] == "replace_only"
        assert li["review_source"] is None
        assert li["baseline_scope_before_posture"] == li["estimate_scope"]
        assert li["visible_required_with_inspect_posture"] is False
        assert li["required_baseline_included"] is True
        assert li["inspection_risk_added"] is False
        assert "review_rationale" not in li
        summary_item = result["tier_summary"]["unreviewed_items"][0]
        audit_item = result["pass_2f_review_audit"]["items"][0]
        for record in (summary_item, audit_item):
            assert "baseline_scope_before_posture" in record
            assert "visible_required_with_inspect_posture" in record
            assert "required_baseline_included" in record
            assert "inspection_risk_added" in record
            assert "pricing_posture" in record

    def test_review_override_flows_to_output(self):
        """Review posture flows to output while rationale stays debug-only."""
        catalog = _make_catalog(_make_item("cab", estimate=HIGH_ESTIMATE,
                                           trade_bucket="kitchen_cabinets_counters",
                                           scope="replace"))
        issues = [_make_issue("cab")]

        # Extract candidates, then manually set review data (simulating 2f)
        candidates = extract_estimate_candidates(issues, catalog)
        assert len(candidates) == 1
        candidates[0].review_posture = "repair"
        candidates[0].review_rationale = "Minor surface damage only"
        candidates[0].review_source = "pass_2f"
        candidates[0].effective_posture = resolve_effective_posture(candidates[0])

        # Feed into group estimate
        result = compute_group_estimate("kitchen", candidates)
        li = result["line_items"][0]
        assert li["review_posture"] == "repair"
        assert li["pricing_posture"] == "repair"
        assert "review_rationale" not in li
        assert li["review_source"] == "pass_2f"
        assert li["effective_posture"] == "repair"

    def test_candidate_review_fields_default_none(self):
        catalog = _make_catalog(_make_item("a", estimate=HIGH_ESTIMATE))
        issues = [_make_issue("a")]
        candidates = extract_estimate_candidates(issues, catalog)
        c = candidates[0]
        assert c.review_posture is None
        assert c.review_rationale is None
        assert c.review_source is None


# ─── Phase C: _coerce_pass_2f ────────────────────────────────────────────────

class TestCoercePass2f:

    def test_valid_package_response(self):
        raw = {
            "verification_status": "confirmed",
            "confirmed_issue_ids": ["issue_1"],
            "rejected_issue_ids": [],
            "evidence_summary": "Dated cabinets and counters are visible.",
        }
        result = _coerce_pass_2f(
            raw,
            package_id="kitchen_modernization__kitchen_1",
            package_type="kitchen_modernization",
            valid_issue_ids={"issue_1", "issue_2"},
        )
        assert result.package_id == "kitchen_modernization__kitchen_1"
        assert result.package_type == "kitchen_modernization"
        assert result.verification_status == "confirmed"
        assert result.confirmed_issue_ids == ["issue_1"]
        assert result.rejected_issue_ids == []
        assert result.evidence_summary == "Dated cabinets and counters are visible."

    def test_invalid_status_fallback_to_uncertain(self):
        raw = {
            "verification_status": "galaxy_wide",
        }
        result = _coerce_pass_2f(
            raw,
            package_id="pkg",
            package_type="kitchen_modernization",
            valid_issue_ids={"issue_1"},
        )
        assert result.verification_status == "uncertain"

    def test_empty_dict(self):
        result = _coerce_pass_2f(
            {},
            package_id="pkg",
            package_type="kitchen_modernization",
            valid_issue_ids={"issue_1"},
        )
        assert result.verification_status == "uncertain"

    def test_evidence_summary_truncated_at_400(self):
        raw = {"evidence_summary": "x" * 500}
        result = _coerce_pass_2f(
            raw,
            package_id="pkg",
            package_type="kitchen_modernization",
            valid_issue_ids={"issue_1"},
        )
        assert len(result.evidence_summary) == 400

    def test_case_insensitive(self):
        raw = {
            "verification_status": "Confirmed",
        }
        result = _coerce_pass_2f(
            raw,
            package_id="pkg",
            package_type="kitchen_modernization",
            valid_issue_ids={"issue_1"},
        )
        assert result.verification_status == "confirmed"

    def test_rejected_defaults_to_all_issue_ids_when_model_omits_rejections(self):
        raw = {
            "verification_status": "rejected",
        }
        result = _coerce_pass_2f(
            raw,
            package_id="pkg",
            package_type="kitchen_modernization",
            valid_issue_ids={"issue_b", "issue_a"},
        )
        assert result.rejected_issue_ids == ["issue_a", "issue_b"]

    def test_issue_id_lists_filter_unknowns_and_duplicates(self):
        raw = {
            "verification_status": "confirmed",
            "confirmed_issue_ids": ["issue_1", "unknown", "issue_1", "issue_2"],
            "rejected_issue_ids": ["unknown", "issue_2"],
        }
        result = _coerce_pass_2f(
            raw,
            package_id="pkg",
            package_type="kitchen_modernization",
            valid_issue_ids={"issue_1", "issue_2"},
        )
        assert result.confirmed_issue_ids == ["issue_1", "issue_2"]
        assert result.rejected_issue_ids == ["issue_2"]

    def test_bathroom_room_count_values_are_coerced(self):
        result = _coerce_pass_2f(
            {
                "visible_room_count": "two_bathrooms",
                "visible_room_count_evidence": "x" * 300,
            },
            package_id="pkg",
            package_type="bathroom_modernization",
            valid_issue_ids={"issue_1"},
            room="bathroom",
        )
        assert result.visible_room_count == "unclear"
        assert len(result.visible_room_count_evidence) == 240

    def test_kitchen_room_count_fields_are_ignored(self):
        result = _coerce_pass_2f(
            {
                "visible_room_count": "multiple_rooms",
                "visible_room_count_evidence": "Conflicting layouts.",
            },
            package_id="pkg",
            package_type="kitchen_modernization",
            valid_issue_ids={"issue_1"},
            room="kitchen",
        )
        assert result.visible_room_count == "unclear"
        assert result.visible_room_count_evidence == ""

    def test_prompt_shape_is_visual_truth_only(self):
        assert "review_outcome" not in PASS_2F_KITCHEN_SYSTEM_PROMPT
        assert "review_outcome" not in PASS_2F_KITCHEN_USER_PROMPT
        assert '"verification_status"' in PASS_2F_KITCHEN_USER_PROMPT
        assert '"confirmed_issue_ids"' in PASS_2F_KITCHEN_USER_PROMPT
        assert '"rejected_issue_ids"' in PASS_2F_KITCHEN_USER_PROMPT
        assert "pricing_posture" not in PASS_2F_KITCHEN_USER_PROMPT
        assert "visible_scope" not in PASS_2F_KITCHEN_USER_PROMPT
        assert "do not estimate prices" in PASS_2F_KITCHEN_SYSTEM_PROMPT.lower()


class TestRunPass2f:

    def test_successful_multi_image_call(self):
        from tools.scene_classifier_passes import run_pass_2f

        vlm_response = json.dumps({
            "verification_status": "confirmed",
            "confirmed_issue_ids": ["issue_1"],
            "rejected_issue_ids": [],
            "evidence_summary": "Dated cabinets and counters visible.",
        })

        mock_vlm = MagicMock()
        mock_vlm.analyze_images = AsyncMock(return_value=vlm_response)

        result = asyncio.get_event_loop().run_until_complete(
            run_pass_2f(
                image_paths=[Path("/fake/kitchen_1.jpg"), Path("/fake/kitchen_2.jpg")],
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                room="kitchen",
                package_id="kitchen_modernization__kitchen_1",
                package_type="kitchen_modernization",
                evidence_items=[{
                    "catalog_item_id": "outdated_kitchen_finishes",
                    "issue_ids": ["issue_1"],
                    "observations": ["dated cabinets"],
                }],
            )
        )

        assert result.package_id == "kitchen_modernization__kitchen_1"
        assert result.verification_status == "confirmed"
        assert result.confirmed_issue_ids == ["issue_1"]
        assert result.raw_response == vlm_response
        mock_vlm.analyze_images.assert_called_once()
        call = mock_vlm.analyze_images.call_args.kwargs
        assert call["system_prompt"] == PASS_2F_KITCHEN_SYSTEM_PROMPT
        assert "pricing_posture" not in call["user_prompt"]
        assert len(call["image_paths"]) == 2

    def test_single_image_client_fallback_for_compatibility(self):
        from tools.scene_classifier_passes import run_pass_2f

        vlm_response = json.dumps({
            "verification_status": "rejected",
            "confirmed_issue_ids": [],
            "rejected_issue_ids": ["issue_1"],
            "evidence_summary": "The proposed package is not visible.",
        })
        class SingleImageVLM:
            def __init__(self):
                self.analyze_image = AsyncMock(return_value=vlm_response)

        mock_vlm = SingleImageVLM()

        result = asyncio.get_event_loop().run_until_complete(
            run_pass_2f(
                image_paths=[Path("/fake/kitchen_1.jpg")],
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                room="kitchen",
                package_id="pkg",
                package_type="kitchen_modernization",
                evidence_items=[{"issue_ids": ["issue_1"]}],
            )
        )

        assert result.verification_status == "rejected"
        assert result.rejected_issue_ids == ["issue_1"]
        mock_vlm.analyze_image.assert_called_once()

    def test_vlm_error_returns_uncertain_without_raising(self):
        from tools.scene_classifier_passes import run_pass_2f

        mock_vlm = MagicMock()
        mock_vlm.analyze_images = AsyncMock(side_effect=RuntimeError("API down"))

        result = asyncio.get_event_loop().run_until_complete(
            run_pass_2f(
                image_paths=[Path("/fake/kitchen_1.jpg")],
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                room="kitchen",
                package_id="pkg",
                package_type="kitchen_modernization",
                evidence_items=[{"issue_ids": ["issue_1"]}],
            )
        )

        assert result.verification_status == "uncertain"
        assert "verification failed" in result.evidence_summary.lower()

    def test_missing_json_returns_uncertain_without_raising(self):
        from tools.scene_classifier_passes import run_pass_2f

        mock_vlm = MagicMock()
        mock_vlm.analyze_images = AsyncMock(return_value="not json")

        result = asyncio.get_event_loop().run_until_complete(
            run_pass_2f(
                image_paths=[Path("/fake/kitchen_1.jpg")],
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                room="kitchen",
                package_id="pkg",
                package_type="kitchen_modernization",
                evidence_items=[{"issue_ids": ["issue_1"]}],
            )
        )

        assert result.verification_status == "uncertain"


# Phase C: retired legacy run_pass_2f_batch compatibility shim

REVISIT_ESTIMATE = {
    **HIGH_ESTIMATE,
}

NO_REVISIT_ESTIMATE = {
    **MINOR_ESTIMATE,
}


class TestRunPass2fBatch:

    def _make_candidates_and_context(self, *, estimate_tier="high", strategy="replace_only"):
        """Create candidates, issues, catalog, and photo mapping for batch tests."""
        est = {
            "estimate_tier": estimate_tier,
            "strategy": strategy,
            "group": "kitchen",
            "stack_behavior": "group_cap",
        }
        catalog = _make_catalog(
            _make_item("cab", estimate=est, trade_bucket="kitchen_cabinets_counters",
                       scope="replace"),
        )
        issues = [_make_issue("cab", photo_key="kitchen_1.jpg")]
        candidates = extract_estimate_candidates(issues, catalog)
        photo_map = {"kitchen_1.jpg": Path("/fake/kitchen_1.jpg")}
        return candidates, issues, catalog, photo_map

    def test_legacy_batch_is_noop_and_marks_retired(self):
        candidates, issues, catalog, photo_map = self._make_candidates_and_context()

        mock_vlm = MagicMock()
        mock_vlm.analyze_image = AsyncMock()
        mock_vlm.analyze_images = AsyncMock()

        result = _run_async(
            run_pass_2f_batch(
                candidates=candidates,
                issues_flat=issues,
                issue_catalog=catalog,
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                photo_key_to_path=photo_map,
            )
        )

        assert len(result) == 1
        c = result[0]
        assert c.pass_2f_attempted is False
        assert c.pass_2f_applied is False
        assert c.pass_2f_fallback_reason == PASS_2F_FALLBACK_RETIRED
        assert c.review_posture is None
        assert c.review_source is None
        assert c.effective_posture == "replace_only"
        mock_vlm.analyze_image.assert_not_called()
        mock_vlm.analyze_images.assert_not_called()

    def test_legacy_batch_ignores_tier_provider_and_image_availability(self):
        candidates, issues, catalog, _ = self._make_candidates_and_context(
            estimate_tier="medium",
            strategy="repair_only",
        )

        mock_vlm = MagicMock()
        mock_vlm.analyze_image = AsyncMock()
        mock_vlm.analyze_images = AsyncMock()

        result = _run_async(
            run_pass_2f_batch(
                candidates=candidates,
                issues_flat=issues,
                issue_catalog=catalog,
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                photo_key_to_path={},
                provider="local",
                pass_2f_tiers=frozenset({"high"}),
            )
        )

        c = result[0]
        assert c.pass_2f_fallback_reason == PASS_2F_FALLBACK_RETIRED
        assert c.effective_posture == "repair_only"
        mock_vlm.analyze_image.assert_not_called()
        mock_vlm.analyze_images.assert_not_called()

    def test_retired_batch_results_flow_to_renovation_estimate_without_review_override(self):
        candidates, issues, catalog, photo_map = self._make_candidates_and_context()

        mock_vlm = MagicMock()
        mock_vlm.analyze_image = AsyncMock()
        mock_vlm.analyze_images = AsyncMock()

        reviewed = _run_async(
            run_pass_2f_batch(
                candidates=candidates,
                issues_flat=issues,
                issue_catalog=catalog,
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                photo_key_to_path=photo_map,
            )
        )

        result = compute_renovation_estimate(
            issues,
            catalog,
            prebuilt_candidates=reviewed,
        )

        assert result["meta"]["candidate_count"] == 1
        li = result["groups"][0]["line_items"][0]
        assert li["review_posture"] is None
        assert li["review_source"] is None
        assert li["effective_posture"] == "replace_only"
        assert li["pass_2f_attempted"] is False
        assert li["pass_2f_applied"] is False
        assert li["pass_2f_fallback_reason"] == PASS_2F_FALLBACK_RETIRED
        assert result["pass_2f_review_audit"]["ran"] is False


# Phase D: effective_posture resolution

class TestEffectivePosture:

    def _candidate(self, *, strategy="replace_only", review_posture=None):
        meta = CatalogEstimateMeta(
            estimate_tier="high",
            strategy=strategy, group="kitchen",
            stack_behavior="group_cap",
        )
        c = EstimateCandidate(
            catalog_item_id="test_item",
            catalog_item_name="Test Item",
            estimate_meta=meta,
            kind="defect", severity=3, scope="replace",
            trade_bucket="kitchen_cabinets_counters",
            cost_obj={"mode": "heuristic"},
            occurrences=1,
            scene_groups_seen=["kitchen"],
            photo_keys=["img_001.jpg"],
            issue_ids=["iss_1"],
        )
        if review_posture is not None:
            c.review_posture = review_posture
            c.review_source = "pass_2f"
        return c

    def test_review_repair_overrides_replace_strategy(self):
        c = self._candidate(strategy="replace_only", review_posture="repair")
        assert resolve_effective_posture(c) == "repair"

    def test_review_replace_is_authoritative(self):
        c = self._candidate(strategy="repair_only", review_posture="replace")
        assert resolve_effective_posture(c) == "replace"

    def test_review_inspect_is_authoritative(self):
        c = self._candidate(strategy="replace_only", review_posture="inspect")
        assert resolve_effective_posture(c) == "inspect"

    def test_keep_default_falls_to_catalog(self):
        c = self._candidate(strategy="replace_only", review_posture="keep_default")
        assert resolve_effective_posture(c) == "replace_only"

    def test_none_review_falls_to_catalog(self):
        c = self._candidate(strategy="repair_only", review_posture=None)
        assert resolve_effective_posture(c) == "repair_only"

    def test_effective_posture_in_line_items(self):
        c = self._candidate(strategy="replace_only", review_posture="repair")
        c.effective_posture = resolve_effective_posture(c)
        result = compute_group_estimate("kitchen", [c])
        li = result["line_items"][0]
        assert li["effective_posture"] == "repair"
        assert li["default_posture"] == "replace_only"

    def test_effective_posture_defaults_when_no_review(self):
        c = self._candidate(strategy="replace_only")
        c.effective_posture = resolve_effective_posture(c)
        result = compute_group_estimate("kitchen", [c])
        li = result["line_items"][0]
        assert li["effective_posture"] == "replace_only"


# ─── Phase H: representative image selection ──────────────────────────────

class TestImageSelection:

    def _candidate(self, photo_keys, scene_groups=None):
        meta = CatalogEstimateMeta(
            estimate_tier="high",
            strategy="replace_only", group="kitchen",
            stack_behavior="group_cap",
        )
        return EstimateCandidate(
            catalog_item_id="cab",
            catalog_item_name="Cabinets",
            estimate_meta=meta,
            kind="defect", severity=3, scope="replace",
            trade_bucket="kitchen_cabinets_counters",
            cost_obj={"mode": "heuristic"},
            occurrences=1,
            scene_groups_seen=scene_groups or ["kitchen"],
            photo_keys=photo_keys,
            issue_ids=["iss_1"],
        )

    def test_prefers_issue_matched_photo(self):
        """Photo with a direct issue match for this item should be preferred."""
        c = self._candidate(["img_a.jpg", "img_b.jpg"])
        issues = [
            _make_issue("other_item", photo_key="img_a.jpg", scene_group="kitchen"),
            _make_issue("cab", photo_key="img_b.jpg", scene_group="kitchen"),
        ]
        paths = {
            "img_a.jpg": Path("/fake/img_a.jpg"),
            "img_b.jpg": Path("/fake/img_b.jpg"),
        }
        with patch("pathlib.Path.exists", return_value=True):
            result = _select_representative_image(c, issues, paths)
        assert result == Path("/fake/img_b.jpg")

    def test_prefers_scene_matched_photo(self):
        """Photo with matching scene_group should be preferred over unmatched."""
        c = self._candidate(["img_a.jpg", "img_b.jpg"], scene_groups=["kitchen"])
        issues = [
            _make_issue("other", photo_key="img_a.jpg", scene_group="bathroom"),
            _make_issue("other", photo_key="img_b.jpg", scene_group="kitchen"),
        ]
        paths = {
            "img_a.jpg": Path("/fake/img_a.jpg"),
            "img_b.jpg": Path("/fake/img_b.jpg"),
        }
        with patch("pathlib.Path.exists", return_value=True):
            result = _select_representative_image(c, issues, paths)
        assert result == Path("/fake/img_b.jpg")

    def test_fallback_to_first_valid(self):
        """When no issue or scene match, fall back to first valid photo (sorted)."""
        c = self._candidate(["img_c.jpg", "img_a.jpg"])
        issues = []
        paths = {
            "img_c.jpg": Path("/fake/img_c.jpg"),
            "img_a.jpg": Path("/fake/img_a.jpg"),
        }
        with patch("pathlib.Path.exists", return_value=True):
            result = _select_representative_image(c, issues, paths)
        # Both rank (3, key), sorted: img_a < img_c
        assert result == Path("/fake/img_a.jpg")

    def test_returns_none_when_no_valid_paths(self):
        c = self._candidate(["img_a.jpg"])
        result = _select_representative_image(c, [], {})
        assert result is None

    def test_retired_batch_does_not_select_review_image(self):
        """The retired per-item batch shim should not prepare a VLM review image."""
        est = {**REVISIT_ESTIMATE}
        catalog = _make_catalog(
            _make_item("cab", estimate=est,
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
        )
        issues = [_make_issue("cab", photo_key="k1.jpg")]
        candidates = extract_estimate_candidates(issues, catalog)
        photo_map = {"k1.jpg": Path("/fake/k1.jpg")}

        mock_vlm = MagicMock()
        mock_vlm.analyze_image = AsyncMock()
        mock_vlm.analyze_images = AsyncMock()

        result = _run_async(
            run_pass_2f_batch(
                candidates=candidates,
                issues_flat=issues,
                issue_catalog=catalog,
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                photo_key_to_path=photo_map,
            )
        )

        assert result[0].review_image_path is None
        assert result[0].pass_2f_fallback_reason == PASS_2F_FALLBACK_RETIRED
        mock_vlm.analyze_image.assert_not_called()
        mock_vlm.analyze_images.assert_not_called()


# ─── Phase I: posture-to-pricing mapping ──────────────────────────────────

class TestPostureToPricing:

    ALLOWANCE_COST = {
        "mode": "allowance",
        "base_low": 1000, "base_high": 5000,
        "per_occurrence_low": 500, "per_occurrence_high": 1500,
        "cap_low": 5000, "cap_high": 15000,
        "cost_source": "estimated",
    }

    def _candidate(self, *, scope="replace", effective_posture=None):
        meta = CatalogEstimateMeta(
            estimate_tier="high",
            strategy="replace_only", group="kitchen",
            stack_behavior="group_cap",
        )
        c = EstimateCandidate(
            catalog_item_id="test_item",
            catalog_item_name="Test Item",
            estimate_meta=meta,
            kind="defect", severity=3, scope=scope,
            trade_bucket="kitchen_cabinets_counters",
            cost_obj=self.ALLOWANCE_COST.copy(),
            occurrences=1,
            scene_groups_seen=["kitchen"],
            photo_keys=["img_001.jpg"],
            issue_ids=["iss_1"],
        )
        c.effective_posture = effective_posture
        return c

    def test_repair_posture_uses_repair_scope(self):
        """Repair posture should use SCOPE_MULT 1.0 (lower than replace 1.3)."""
        c_repair = self._candidate(scope="replace", effective_posture="repair")
        c_replace = self._candidate(scope="replace", effective_posture="replace")
        low_repair, high_repair = resolve_pricing_band(c_repair, "repair")
        low_replace, high_replace = resolve_pricing_band(c_replace, "replace")
        # repair (SCOPE_MULT 1.0) should be cheaper than replace (SCOPE_MULT 1.3)
        assert low_repair < low_replace
        assert high_repair < high_replace

    def test_replace_posture_uses_replace_scope(self):
        c = self._candidate(effective_posture="replace")
        low, high = resolve_pricing_band(c, "replace")
        assert low > 0
        assert high > low

    def test_inspect_posture_returns_allowance(self):
        c = self._candidate(effective_posture="inspect")
        result = resolve_pricing_band(c, "inspect")
        assert result == INSPECT_ALLOWANCE

    def test_inspect_only_strategy_returns_allowance(self):
        c = self._candidate(effective_posture="inspect_only")
        result = resolve_pricing_band(c, "inspect_only")
        assert result == INSPECT_ALLOWANCE

    def test_manual_allowance_is_not_post_multiplied(self):
        manual_cost = {
            "mode": "allowance",
            "base_low": 1000, "base_high": 5000,
            "per_occurrence_low": 500, "per_occurrence_high": 1500,
            "cap_low": 5000, "cap_high": 15000,
            "cost_source": "manual",
        }
        c = self._candidate(scope="replace", effective_posture="replace")
        c.cost_obj = manual_cost
        assert resolve_pricing_band(c, "replace") == (1000, 5000)

    def test_ambiguous_falls_to_catalog_scope(self):
        """repair_or_replace is ambiguous → falls back to candidate.scope."""
        c = self._candidate(scope="replace", effective_posture="repair_or_replace")
        low, high = resolve_pricing_band(c, "repair_or_replace")
        # Should use candidate.scope="replace" (SCOPE_MULT 1.3)
        c_explicit = self._candidate(scope="replace", effective_posture="replace")
        low_explicit, high_explicit = resolve_pricing_band(c_explicit, "replace")
        assert low == low_explicit
        assert high == high_explicit

    def test_pass_2f_repair_overrides_replace_catalog(self):
        """End-to-end: 2f returns repair for a catalog item with scope=replace."""
        est = {**REVISIT_ESTIMATE, "strategy": "replace_only", "group": "kitchen", "stack_behavior": "group_cap"}
        catalog = _make_catalog(
            _make_item("cab", estimate=est,
                       trade_bucket="kitchen_cabinets_counters", scope="replace",
                       cost=self.ALLOWANCE_COST.copy()),
        )
        issues = [_make_issue("cab")]

        # Without 2f: default pricing uses scope=replace (SCOPE_MULT 1.3)
        result_default = compute_renovation_estimate(issues, catalog)
        default_high = result_default["groups"][0]["line_items"][0]["cost_high"]

        # With 2f review: posture=repair overrides scope to repair (SCOPE_MULT 1.0)
        candidates = extract_estimate_candidates(issues, catalog)
        candidates[0].review_posture = "repair"
        candidates[0].review_source = "pass_2f"
        candidates[0].effective_posture = resolve_effective_posture(candidates[0])
        result_review = compute_renovation_estimate(
            issues, catalog, prebuilt_candidates=candidates,
        )
        review_high = result_review["groups"][0]["line_items"][0]["cost_high"]

        # repair pricing must be cheaper than replace pricing
        assert review_high < default_high
        # Verify the posture audit trail
        li = result_review["groups"][0]["line_items"][0]
        assert li["effective_posture"] == "repair"
        assert li["default_posture"] == "replace_only"
        assert li["review_posture"] == "repair"


# ─── Phase K: fallback behavior ────────────────────────────────────────────

class TestFallbackBehavior:

    def _make_batch_context(self, *, estimate_tier="high"):
        """Create candidates, issues, catalog, and photo mapping for batch tests."""
        est = {
            "estimate_tier": estimate_tier,
            "strategy": "replace_only",
            "group": "kitchen",
            "stack_behavior": "group_cap",
        }
        catalog = _make_catalog(
            _make_item("cab", estimate=est, trade_bucket="kitchen_cabinets_counters",
                       scope="replace"),
        )
        issues = [_make_issue("cab", photo_key="kitchen_1.jpg")]
        candidates = extract_estimate_candidates(issues, catalog)
        photo_map = {"kitchen_1.jpg": Path("/fake/kitchen_1.jpg")}
        return candidates, issues, catalog, photo_map

    def test_retired_batch_uses_single_fallback_reason(self):
        candidates, issues, catalog, photo_map = self._make_batch_context(estimate_tier="high")

        mock_vlm = MagicMock()
        mock_vlm.analyze_image = AsyncMock()
        mock_vlm.analyze_images = AsyncMock()

        result = _run_async(
            run_pass_2f_batch(
                candidates=candidates,
                issues_flat=issues,
                issue_catalog=catalog,
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                photo_key_to_path=photo_map,
            )
        )

        c = result[0]
        assert c.pass_2f_attempted is False
        assert c.pass_2f_applied is False
        assert c.pass_2f_fallback_reason == PASS_2F_FALLBACK_RETIRED
        assert c.review_posture is None
        assert c.effective_posture == "replace_only"
        mock_vlm.analyze_image.assert_not_called()
        mock_vlm.analyze_images.assert_not_called()

    def test_retired_batch_does_not_call_vlm_when_image_exists(self):
        candidates, issues, catalog, photo_map = self._make_batch_context(estimate_tier="high")

        mock_vlm = MagicMock()
        mock_vlm.analyze_image = AsyncMock(side_effect=AssertionError("legacy call should not run"))
        mock_vlm.analyze_images = AsyncMock(side_effect=AssertionError("legacy call should not run"))

        with patch("pathlib.Path.exists", return_value=True):
            result = _run_async(
                run_pass_2f_batch(
                    candidates=candidates,
                    issues_flat=issues,
                    issue_catalog=catalog,
                    vlm_client=mock_vlm,
                    model_config={"model": "test"},
                    photo_key_to_path=photo_map,
                )
            )

        assert result[0].pass_2f_fallback_reason == PASS_2F_FALLBACK_RETIRED
        mock_vlm.analyze_image.assert_not_called()
        mock_vlm.analyze_images.assert_not_called()

    def test_fallback_fields_in_line_items(self):
        """Verify all 3 fallback fields appear in compute_group_estimate output."""
        est = {**REVISIT_ESTIMATE}
        catalog = _make_catalog(
            _make_item("cab", estimate=est,
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
        )
        issues = [_make_issue("cab")]
        candidates = extract_estimate_candidates(issues, catalog)
        candidates[0].pass_2f_attempted = False
        candidates[0].pass_2f_applied = False
        candidates[0].pass_2f_fallback_reason = PASS_2F_FALLBACK_RETIRED
        candidates[0].effective_posture = "replace_only"

        result = compute_group_estimate("kitchen", candidates)
        li = result["line_items"][0]
        assert "pass_2f_attempted" in li
        assert "pass_2f_applied" in li
        assert "pass_2f_fallback_reason" in li
        assert li["pass_2f_attempted"] is False
        assert li["pass_2f_applied"] is False
        assert li["pass_2f_fallback_reason"] == PASS_2F_FALLBACK_RETIRED


# Phase L: audit block

class TestAuditBlock:

    def _make_candidate(self, item_id, *, estimate_tier="high", strategy="replace_only",
                        group="kitchen"):
        est = {
            "estimate_tier": estimate_tier,
            "strategy": strategy,
            "group": group,
            "stack_behavior": "group_cap",
        }
        catalog = _make_catalog(
            _make_item(item_id, estimate=est,
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
        )
        issues = [_make_issue(item_id)]
        candidates = extract_estimate_candidates(issues, catalog)
        return candidates[0], issues, catalog

    def test_audit_block_present_in_output(self):
        """pass_2f_review_audit key should exist in compute_renovation_estimate output."""
        c, issues, catalog = self._make_candidate("cab")
        result = compute_renovation_estimate(issues, catalog, prebuilt_candidates=[c])
        assert "pass_2f_review_audit" in result
        audit = result["pass_2f_review_audit"]
        assert "ran" in audit
        assert "reviewed_count" in audit
        assert "applied_count" in audit
        assert "fallback_count" in audit
        assert "items" in audit

    def test_audit_counts_correct(self):
        """3 candidates: 1 applied, 1 keep_default, 1 not_eligible → correct counts."""
        # Candidate 1: eligible, applied (repair)
        c1, _, _ = self._make_candidate("cab1")
        c1.pass_2f_attempted = True
        c1.pass_2f_applied = True
        c1.review_posture = "repair"
        c1.review_source = "pass_2f"
        c1.effective_posture = "repair"

        # Candidate 2: eligible, keep_default (attempted but not applied)
        c2, _, _ = self._make_candidate("cab2")
        c2.pass_2f_attempted = True
        c2.pass_2f_applied = False
        c2.pass_2f_fallback_reason = "keep_default"
        c2.review_posture = "keep_default"
        c2.review_source = "pass_2f"
        c2.effective_posture = "replace_only"

        # Candidate 3: not eligible
        c3, _, _ = self._make_candidate("cab3", estimate_tier="medium")
        c3.pass_2f_fallback_reason = "not_eligible"
        c3.effective_posture = "replace_only"

        # Build catalog with all items
        est_high = {**HIGH_ESTIMATE}
        est_medium = {**MEDIUM_ESTIMATE, "group": "kitchen", "stack_behavior": "group_cap"}
        catalog = _make_catalog(
            _make_item("cab1", estimate=est_high,
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
            _make_item("cab2", estimate=est_high,
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
            _make_item("cab3", estimate=est_medium,
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
        )
        issues = [_make_issue("cab1"), _make_issue("cab2"), _make_issue("cab3")]

        result = compute_renovation_estimate(
            issues, catalog, prebuilt_candidates=[c1, c2, c3],
        )
        audit = result["pass_2f_review_audit"]
        assert audit["ran"] is True
        assert audit["reviewed_count"] == 2   # c1 + c2 attempted
        assert audit["applied_count"] == 1    # only c1 applied
        assert audit["fallback_count"] == 1   # c2 attempted but not applied

    def test_audit_items_includes_all_tiers(self):
        """Both high and medium candidates appear in audit items list."""
        c1, _, _ = self._make_candidate("cab1", estimate_tier="high")
        c1.effective_posture = "replace_only"

        c2, _, _ = self._make_candidate("cab2", estimate_tier="medium")
        c2.effective_posture = "replace_only"

        catalog = _make_catalog(
            _make_item("cab1", estimate=HIGH_ESTIMATE,
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
            _make_item("cab2", estimate={**MEDIUM_ESTIMATE, "group": "kitchen", "stack_behavior": "group_cap"},
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
        )
        issues = [_make_issue("cab1"), _make_issue("cab2")]

        result = compute_renovation_estimate(
            issues, catalog, prebuilt_candidates=[c1, c2],
        )
        audit = result["pass_2f_review_audit"]
        item_ids = [it["catalog_item_id"] for it in audit["items"]]
        assert "cab1" in item_ids
        assert "cab2" in item_ids

    def test_audit_ran_false_when_no_review(self):
        """No 2f review happened → ran=False, all counts=0."""
        c, issues, catalog = self._make_candidate("cab")
        c.effective_posture = "replace_only"
        # No 2f fields set — defaults are False/None

        result = compute_renovation_estimate(issues, catalog, prebuilt_candidates=[c])
        audit = result["pass_2f_review_audit"]
        assert audit["ran"] is False
        assert audit["reviewed_count"] == 0
        assert audit["applied_count"] == 0
        assert audit["fallback_count"] == 0

    def test_meta_includes_2f_counts(self):
        """meta dict should have pass_2f_reviewed_count and pass_2f_applied_count."""
        c, issues, catalog = self._make_candidate("cab")
        c.pass_2f_attempted = True
        c.pass_2f_applied = True
        c.review_posture = "repair"
        c.review_source = "pass_2f"
        c.effective_posture = "repair"

        result = compute_renovation_estimate(issues, catalog, prebuilt_candidates=[c])
        meta = result["meta"]
        assert "pass_2f_reviewed_count" in meta
        assert "pass_2f_applied_count" in meta
        assert meta["pass_2f_reviewed_count"] == 1
        assert meta["pass_2f_applied_count"] == 1

    def test_audit_emits_consistency_flags_and_counts(self):
        c, issues, catalog = self._make_candidate("cab")
        c.pass_2f_attempted = True
        c.pass_2f_applied = True
        c.review_posture = "repair"
        c.review_visible_scope = "room_wide"
        c.review_consistency_flags = ["room_wide_repair"]
        c.review_source = "pass_2f"
        c.effective_posture = "repair"

        result = compute_renovation_estimate(issues, catalog, prebuilt_candidates=[c])
        audit = result["pass_2f_review_audit"]

        assert audit["consistency_flag_counts"] == {"room_wide_repair": 1}
        assert audit["items"][0]["consistency_flags"] == ["room_wide_repair"]

    def test_empty_candidates_has_audit_block(self):
        """Even with no candidates, audit block should be present."""
        catalog = _make_catalog(_make_item("a"))
        result = compute_renovation_estimate([], catalog)
        assert "pass_2f_review_audit" in result
        audit = result["pass_2f_review_audit"]
        assert audit["ran"] is False
        assert audit["items"] == []
        assert audit["consistency_flag_counts"] == {}
        assert result["meta"]["pass_2f_reviewed_count"] == 0
        assert result["meta"]["pass_2f_applied_count"] == 0

    def test_slim_artifact_scrubber_removes_debug_only_pass_2f_fields(self):
        from tools.artifact_writers import _strip_pass_2f_audit_rationale

        payload = {
            "pass_2f_review_audit": {
                "consistency_flag_counts": {"room_wide_repair": 1},
                "items": [
                    {
                        "catalog_item_id": "cab",
                        "visible_scope": "localized",
                        "pricing_posture": "repair",
                        "rationale": "Visible minor damage.",
                        "review_rationale": "Legacy visible minor damage.",
                        "consistency_flags": ["room_wide_repair"],
                    }
                ]
            },
            "groups": [
                {
                    "line_items": [
                        {
                            "catalog_item_id": "cab",
                            "review_rationale": "Visible minor damage.",
                            "rationale": "Future user-facing explanation.",
                        }
                    ]
                }
            ],
        }

        _strip_pass_2f_audit_rationale(payload)

        audit = payload["pass_2f_review_audit"]
        audit_item = payload["pass_2f_review_audit"]["items"][0]
        line_item = payload["groups"][0]["line_items"][0]
        assert "consistency_flag_counts" not in audit
        assert "rationale" not in audit_item
        assert "review_rationale" not in audit_item
        assert "consistency_flags" not in audit_item
        assert line_item["review_rationale"] == "Visible minor damage."
        assert line_item["rationale"] == "Future user-facing explanation."
        assert audit_item["visible_scope"] == "localized"
        assert audit_item["pricing_posture"] == "repair"

    def test_write_photo_intel_keeps_package_pass_2f_trace_debug_only(self):
        from tools.artifact_writers import write_photo_intel

        tmp_path = Path.cwd() / ".pytest_cache" / f"photo_intel_debug_{uuid.uuid4().hex}"
        tmp_path.mkdir(parents=True)
        image_path = tmp_path / "kitchen_1.jpg"
        try:
            image_path.write_bytes(b"fake image")
            kitchen_item = _make_item(
                "outdated_kitchen_finishes",
                estimate={
                    "estimate_tier": "medium",
                    "strategy": "replace_only",
                    "group": "kitchen",
                    "stack_behavior": "group_cap",
                    "unit_policy": "per_kitchen",
                },
                trade_bucket="kitchen_cabinets_counters",
                scope="replace",
            )
            kitchen_item.update({
                "display_class": "marketability",
                "package_role": "package_driver",
                "package_type": "kitchen_modernization",
            })
            issue_catalog = _make_catalog(
                kitchen_item,
            )
            result = SimpleNamespace(
                image_path=str(image_path),
                scene="kitchen",
                scene_classifier={
                    "scene": "kitchen",
                    "canonical_issues": [
                        {
                            "description": "Kitchen finishes are visibly dated.",
                            "catalogItemId": "outdated_kitchen_finishes",
                            "label": "marketability",
                        }
                    ],
                    "verified_issues": [],
                    "matched_issues": [],
                    "passes": {
                        "1a": {"scene": "kitchen", "confidence": 0.9, "reasoning": ""},
                        "1c": {"overall_impression": "", "image_summary": "", "notable_features": []},
                        "2e": {},
                    },
                },
                scene_data=None,
                processing_time=0.1,
                error=None,
            )
            job = SimpleNamespace(
                property_key="prop",
                job_id="job",
                timestamp="2026-04-19T00:00:00Z",
                artifacts_dir=str(tmp_path),
                results=[result],
            )
            mock_vlm = MagicMock()
            mock_vlm.analyze_images = AsyncMock(return_value=json.dumps({
                "verification_status": "confirmed",
                "confirmed_issue_ids": [],
                "rejected_issue_ids": [],
                "evidence_summary": "Dated kitchen finishes are visible.",
            }))
            mock_vlm.analyze_image = AsyncMock()

            output_path = write_photo_intel(
                cfg=SimpleNamespace(LM_STUDIO_MODEL="test-model"),
                job=job,
                detection_backend="test",
                analysis_profile="test",
                use_pass_architecture=True,
                pass_toggles={"2f": True},
                model_overrides={},
                gpt_config={"model": "test-model"},
                issue_catalog=issue_catalog,
                output_path=tmp_path / "photo_intel.json",
                vlm_client=mock_vlm,
            )

            slim = json.loads(output_path.read_text(encoding="utf-8"))
            debug = json.loads((tmp_path / "photo_intel_debug.json").read_text(encoding="utf-8"))
            slim_audit = slim["renovation_estimate"]["pass_2f_review_audit"]

            assert debug["pass_2f_trace"]["mode"] == "package_visual_verification"
            assert debug["pass_2f_trace"]["attempted_count"] == 1
            assert debug["pass_2f_trace"]["confirmed_count"] == 1
            assert debug["renovation_estimate_v4"]["pass_2f_trace"]["attempted_count"] == 1
            assert mock_vlm.analyze_images.await_count == 1
            mock_vlm.analyze_image.assert_not_called()
            assert "consistency_flag_counts" not in slim_audit
            assert "pass_2f_trace" not in slim
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


# ─── Phase M: provider selection ────────────────────────────────────────────

class TestProviderSelection:

    def test_resolve_premium_config(self):
        """provider='premium' should return premium_config."""
        local = {"model": "qwen-local"}
        premium = {"model": "gpt-5"}
        result = resolve_pass_2f_model_config(
            provider="premium", local_config=local, premium_config=premium,
        )
        assert result == premium

    def test_resolve_local_config(self):
        """provider='local' should return local_config."""
        local = {"model": "qwen-local"}
        premium = {"model": "gpt-5"}
        result = resolve_pass_2f_model_config(
            provider="local", local_config=local, premium_config=premium,
        )
        assert result == local

    def test_default_provider_is_premium(self):
        """Unknown provider string should default to premium."""
        local = {"model": "qwen-local"}
        premium = {"model": "gpt-5"}
        result = resolve_pass_2f_model_config(
            provider="unknown", local_config=local, premium_config=premium,
        )
        assert result == premium

    def test_retired_batch_ignores_provider_premium_without_vlm_call(self):
        """The old per-item batch shim should not call VLM for provider='premium'."""
        est = {**REVISIT_ESTIMATE}
        catalog = _make_catalog(
            _make_item("cab", estimate=est,
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
        )
        issues = [_make_issue("cab", photo_key="kitchen_1.jpg")]
        candidates = extract_estimate_candidates(issues, catalog)
        photo_map = {"kitchen_1.jpg": Path("/fake/kitchen_1.jpg")}

        mock_vlm = MagicMock()
        mock_vlm.analyze_image = AsyncMock()
        mock_vlm.analyze_images = AsyncMock()

        result = asyncio.get_event_loop().run_until_complete(
            run_pass_2f_batch(
                candidates=candidates,
                issues_flat=issues,
                issue_catalog=catalog,
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                photo_key_to_path=photo_map,
                provider="premium",
            )
        )

        assert result[0].review_source is None
        assert result[0].pass_2f_fallback_reason == PASS_2F_FALLBACK_RETIRED
        mock_vlm.analyze_image.assert_not_called()
        mock_vlm.analyze_images.assert_not_called()

    def test_retired_batch_ignores_provider_local_without_vlm_call(self):
        """The old per-item batch shim should not call VLM for provider='local'."""
        est = {**REVISIT_ESTIMATE}
        catalog = _make_catalog(
            _make_item("cab", estimate=est,
                       trade_bucket="kitchen_cabinets_counters", scope="replace"),
        )
        issues = [_make_issue("cab", photo_key="kitchen_1.jpg")]
        candidates = extract_estimate_candidates(issues, catalog)
        photo_map = {"kitchen_1.jpg": Path("/fake/kitchen_1.jpg")}

        mock_vlm = MagicMock()
        mock_vlm.analyze_image = AsyncMock()
        mock_vlm.analyze_images = AsyncMock()

        result = asyncio.get_event_loop().run_until_complete(
            run_pass_2f_batch(
                candidates=candidates,
                issues_flat=issues,
                issue_catalog=catalog,
                vlm_client=mock_vlm,
                model_config={"model": "test"},
                photo_key_to_path=photo_map,
                provider="local",
            )
        )

        assert result[0].review_source is None
        assert result[0].pass_2f_fallback_reason == PASS_2F_FALLBACK_RETIRED
        mock_vlm.analyze_image.assert_not_called()
        mock_vlm.analyze_images.assert_not_called()
