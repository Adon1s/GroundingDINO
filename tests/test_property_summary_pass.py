"""Tests for tools.property_summary_pass — deterministic property summary builder."""

import pytest

from tools.property_summary_pass import (
    CatalogIndex,
    CatalogItem,
    MAX_MODERNIZATION_SEVERITY,
    build_property_summary_v1,
    load_catalog_index,
    _evidence_boost,
    _clamp,
    _max_scope,
    _norm_scope,
    _norm_kind,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_CATALOG = {
    "version": "2.0",
    "items": [
        {
            "id": "water_stain_ceiling",
            "name": "Water Stain on Ceiling",
            "kind": "defect",
            "severity": 3,
            "trade_bucket": "moisture_mold",
            "scope": "repair",
        },
        {
            "id": "peeling_or_discolored_paint",
            "name": "Peeling or Discolored Paint",
            "kind": "defect",
            "severity": 1,
            "trade_bucket": "paint_drywall",
            "scope": "cosmetic",
        },
        {
            "id": "outdated_kitchen_finishes",
            "name": "Outdated Kitchen Finishes",
            "kind": "upgrade",
            "severity": 1,
            "trade_bucket": "kitchen_cabinets_counters",
            "scope": "replace",
        },
        {
            "id": "visible_mold_or_mildew",
            "name": "Visible Mold or Mildew",
            "kind": "defect",
            "severity": 4,
            "trade_bucket": "moisture_mold",
            "scope": "repair",
        },
        {
            "id": "scratched_flooring",
            "name": "Scratched or Worn Flooring",
            "kind": "defect",
            "severity": 2,
            "trade_bucket": "flooring",
            "scope": "cosmetic",
        },
        {
            "id": "outdated_flooring_style",
            "name": "Outdated Flooring Style",
            "kind": "upgrade",
            "severity": 2,
            "trade_bucket": "flooring",
            "scope": "replace",
        },
        {
            "id": "electrical_hazard",
            "name": "Exposed or Unsafe Wiring",
            "kind": "defect",
            "severity": 4,
            "trade_bucket": "electrical",
            "scope": "repair",
        },
    ],
    "trade_buckets": [
        {"id": "moisture_mold", "name": "Moisture & Mold"},
        {"id": "paint_drywall", "name": "Paint & Drywall"},
        {"id": "kitchen_cabinets_counters", "name": "Kitchen Cabinets & Counters"},
        {"id": "flooring", "name": "Flooring"},
        {"id": "electrical", "name": "Electrical"},
    ],
}


@pytest.fixture
def catalog_index():
    return load_catalog_index(SAMPLE_CATALOG)


def _make_issue(
    issue_id: str,
    catalog_item_id: str,
    photo_key: str = "photo_001.jpg",
    scene: str = "kitchen",
    scene_group: str = "kitchen",
    catalog_item_kind: str = "defect",
) -> dict:
    return {
        "issue_id": issue_id,
        "photo_id": "abc123",
        "photo_key": photo_key,
        "scene": scene,
        "scene_group": scene_group,
        "description": f"Test issue {issue_id}",
        "label": "defect_or_damage",
        "location_hint": "",
        "catalog_item_id": catalog_item_id,
        "catalog_item_kind": catalog_item_kind,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CatalogIndex loader tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadCatalogIndex:

    def test_loads_items(self, catalog_index):
        assert len(catalog_index.items_by_id) == 7
        assert "water_stain_ceiling" in catalog_index.items_by_id

    def test_loads_trade_buckets(self, catalog_index):
        assert catalog_index.trade_bucket_name_by_id["moisture_mold"] == "Moisture & Mold"
        assert catalog_index.trade_bucket_name_by_id["paint_drywall"] == "Paint & Drywall"

    def test_item_fields(self, catalog_index):
        item = catalog_index.items_by_id["water_stain_ceiling"]
        assert item.name == "Water Stain on Ceiling"
        assert item.severity == 3
        assert item.trade_bucket == "moisture_mold"
        assert item.scope == "repair"
        assert item.kind == "defect"

    def test_missing_severity_defaults_to_1(self):
        cat = {"items": [{"id": "test", "name": "Test"}], "trade_buckets": []}
        idx = load_catalog_index(cat)
        assert idx.items_by_id["test"].severity == 1

    def test_missing_scope_defaults_to_unknown(self):
        cat = {"items": [{"id": "test", "name": "Test"}], "trade_buckets": []}
        idx = load_catalog_index(cat)
        assert idx.items_by_id["test"].scope == "unknown"

    def test_missing_kind_defaults_to_defect(self):
        cat = {"items": [{"id": "test", "name": "Test"}], "trade_buckets": []}
        idx = load_catalog_index(cat)
        assert idx.items_by_id["test"].kind == "defect"

    def test_empty_catalog(self):
        idx = load_catalog_index({})
        assert len(idx.items_by_id) == 0
        assert len(idx.trade_bucket_name_by_id) == 0

    def test_legacy_defect_id_field(self):
        cat = {"items": [{"defect_id": "legacy_item", "name": "Legacy"}], "trade_buckets": []}
        idx = load_catalog_index(cat)
        assert "legacy_item" in idx.items_by_id


# ═══════════════════════════════════════════════════════════════════════════════
# Helper function tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_evidence_boost_thresholds(self):
        assert _evidence_boost(0) == 0
        assert _evidence_boost(1) == 0
        assert _evidence_boost(2) == 1
        assert _evidence_boost(3) == 2
        assert _evidence_boost(4) == 2
        assert _evidence_boost(5) == 3
        assert _evidence_boost(10) == 3

    def test_clamp(self):
        assert _clamp(1, 5, 0) == 1
        assert _clamp(1, 5, 3) == 3
        assert _clamp(1, 5, 7) == 5
        assert _clamp(1, 5, -5) == 1

    def test_max_scope(self):
        assert _max_scope("cosmetic", "repair") == "repair"
        assert _max_scope("replace", "repair") == "replace"
        assert _max_scope("unknown", "cosmetic") == "cosmetic"
        assert _max_scope("service", "cosmetic") == "service"

    def test_norm_scope(self):
        assert _norm_scope("repair") == "repair"
        assert _norm_scope("REPLACE") == "replace"
        assert _norm_scope(None) == "unknown"
        assert _norm_scope("") == "unknown"
        assert _norm_scope("bogus") == "unknown"

    def test_norm_kind(self):
        assert _norm_kind("defect") == "defect"
        assert _norm_kind("degradation") == "degradation"
        assert _norm_kind("modernization") == "modernization"
        # legacy upgrade (v1 catalogs, historical artifacts) maps to its v2 role
        assert _norm_kind("upgrade") == "modernization"
        # junk stays tolerant on this artifact-reading path
        assert _norm_kind(None) == "defect"
        assert _norm_kind("") == "defect"
        assert _norm_kind("bogus") == "defect"


# ═══════════════════════════════════════════════════════════════════════════════
# Empty / minimal input tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmptyInput:

    def test_empty_issues_flat(self, catalog_index):
        result = build_property_summary_v1(
            property_key="test_prop",
            run_id="run_001",
            issues_flat=[],
            catalog_index=catalog_index,
        )
        assert result["version"] == "2.0"
        assert result["property_key"] == "test_prop"
        assert result["listing"]["top_severity"] == 0
        assert result["listing"]["kind_counts"] == {
            "defect": 0, "degradation": 0, "modernization": 0,
        }
        assert result["listing"]["one_liner"] == ""
        assert result["buckets"] == []

    def test_issues_without_catalog_id_skipped(self, catalog_index):
        issues = [
            {
                "issue_id": "i1",
                "photo_key": "p1.jpg",
                "scene": "kitchen",
                "scene_group": "kitchen",
                "description": "No catalog match",
                "label": "defect_or_damage",
                "catalog_item_id": None,
                "catalog_item_kind": "defect",
            }
        ]
        result = build_property_summary_v1(
            property_key="test_prop",
            run_id="run_001",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        assert result["buckets"] == []

    def test_issues_with_unknown_catalog_id_skipped(self, catalog_index):
        issues = [_make_issue("i1", "nonexistent_item")]
        result = build_property_summary_v1(
            property_key="test_prop",
            run_id="run_001",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        assert result["buckets"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# Single bucket / single scene tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingleBucketSingleScene:

    def test_single_issue(self, catalog_index):
        issues = [_make_issue("i1", "water_stain_ceiling")]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        assert len(result["buckets"]) == 1
        bucket = result["buckets"][0]
        assert bucket["bucket_id"] == "moisture_mold"
        assert bucket["bucket_name"] == "Moisture & Mold"
        assert bucket["issue_count"] == 1
        assert bucket["kind_counts"] == {
            "defect": 1, "degradation": 0, "modernization": 0,
        }
        assert len(bucket["scenes"]) == 1
        assert bucket["scenes"][0]["scene_group"] == "kitchen"
        assert len(bucket["scenes"][0]["blocks"]) == 1
        block = bucket["scenes"][0]["blocks"][0]
        assert block["title"] == "Water Stain on Ceiling"
        assert block["base_severity"] == 3
        assert block["kind"] == "defect"
        assert block["scope_max"] == "repair"
        assert block["evidence_count"] == 1

    def test_multiple_photos_evidence_boost(self, catalog_index):
        """Same item in 3 different photos → evidence_count=3 → boost +2."""
        issues = [
            _make_issue("i1", "water_stain_ceiling", photo_key="photo_001.jpg"),
            _make_issue("i2", "water_stain_ceiling", photo_key="photo_002.jpg"),
            _make_issue("i3", "water_stain_ceiling", photo_key="photo_003.jpg"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        block = result["buckets"][0]["scenes"][0]["blocks"][0]
        assert block["evidence_count"] == 3
        # base=3, evidence=+2, scope(repair)=+1, kind(defect)=0 → raw=6 → clamped=5
        assert block["display_severity"] == 5

    def test_display_severity_formula_basic(self, catalog_index):
        """Peeling paint: base=1, cosmetic(0), defect(0), 1 photo(0) → display=1."""
        issues = [_make_issue("i1", "peeling_or_discolored_paint")]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        block = result["buckets"][0]["scenes"][0]["blocks"][0]
        assert block["base_severity"] == 1
        assert block["display_severity"] == 1  # 1 + 0 + 0 + 0 = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-scene boost tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiSceneBoost:

    def test_same_item_across_scenes_gets_boost(self, catalog_index):
        """Water stain in kitchen AND bathroom → multi-scene boost +2."""
        issues = [
            _make_issue("i1", "water_stain_ceiling", scene_group="kitchen"),
            _make_issue("i2", "water_stain_ceiling", scene_group="bathroom"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        # Should have 1 bucket (moisture_mold) with 2 scene_groups
        bucket = result["buckets"][0]
        assert len(bucket["scenes"]) == 2
        # Each block gets multi-scene boost
        for scene in bucket["scenes"]:
            block = scene["blocks"][0]
            # base=3, evidence=0(1 photo each), scope(repair)=+1, kind(defect)=0, multi=+2 → 6 → clamped=5
            assert block["display_severity"] == 5

    def test_different_items_no_cross_boost(self, catalog_index):
        """Different catalog items in different scenes → no multi-scene boost."""
        issues = [
            _make_issue("i1", "water_stain_ceiling", scene_group="kitchen"),
            _make_issue("i2", "peeling_or_discolored_paint", scene_group="bathroom"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        # Should have 2 buckets (moisture_mold, paint_drywall)
        assert len(result["buckets"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Kind boost tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKindBoost:

    def test_upgrade_ranks_below_defect(self, catalog_index):
        """Upgrade gets kind_boost=-1, defect gets 0 → upgrade ranks lower."""
        issues = [
            _make_issue("i1", "outdated_kitchen_finishes", catalog_item_kind="upgrade"),
            _make_issue("i2", "peeling_or_discolored_paint"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        # Paint (defect): base=1, cosmetic=0, defect=0 → display=1
        # Kitchen (upgrade→modernization): base=1, replace=+2, modernization=-1 → display=2
        # Kitchen has higher display, but Paint (defect) gets rank_score bonus
        assert len(result["buckets"]) == 2
        # First bucket has higher defect count
        assert (
            result["buckets"][0]["kind_counts"]["defect"]
            >= result["buckets"][1]["kind_counts"]["defect"]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Display severity clamping
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeverityClamping:

    def test_never_below_1(self, catalog_index):
        """Upgrade with low base → still at least 1."""
        issues = [_make_issue("i1", "outdated_kitchen_finishes", catalog_item_kind="upgrade")]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        block = result["buckets"][0]["scenes"][0]["blocks"][0]
        # base=1, replace=+2, upgrade=-1 → raw=2 → display=2 (still above 1)
        assert block["display_severity"] >= 1

    def test_never_above_5(self, catalog_index):
        """High severity with all boosts → clamped to 5."""
        issues = [
            _make_issue("i1", "visible_mold_or_mildew", photo_key="p1.jpg", scene_group="kitchen"),
            _make_issue("i2", "visible_mold_or_mildew", photo_key="p2.jpg", scene_group="kitchen"),
            _make_issue("i3", "visible_mold_or_mildew", photo_key="p3.jpg", scene_group="kitchen"),
            _make_issue("i4", "visible_mold_or_mildew", photo_key="p4.jpg", scene_group="kitchen"),
            _make_issue("i5", "visible_mold_or_mildew", photo_key="p5.jpg", scene_group="kitchen"),
            _make_issue("i6", "visible_mold_or_mildew", photo_key="p6.jpg", scene_group="bathroom"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        for bucket in result["buckets"]:
            for scene in bucket["scenes"]:
                for block in scene["blocks"]:
                    assert block["display_severity"] <= 5


# ═══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeduplication:

    def test_duplicate_issue_ids_deduped(self, catalog_index):
        issues = [
            _make_issue("i1", "water_stain_ceiling", photo_key="p1.jpg"),
            _make_issue("i1", "water_stain_ceiling", photo_key="p2.jpg"),  # same issue_id
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        block = result["buckets"][0]["scenes"][0]["blocks"][0]
        assert len(block["issue_ids"]) == 1
        assert block["evidence_count"] == 1  # only 1 unique photo from first seen

    def test_same_catalog_item_same_scene_photo_stays_one_block(self, catalog_index):
        issues = [
            _make_issue("i1", "peeling_or_discolored_paint", photo_key="photo_008.jpg", scene_group="living_areas"),
            _make_issue("i2", "peeling_or_discolored_paint", photo_key="photo_008.jpg", scene_group="living_areas"),
        ]

        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )

        blocks = result["buckets"][0]["scenes"][0]["blocks"]
        assert len(blocks) == 1
        assert blocks[0]["title"] == "Peeling or Discolored Paint"
        assert blocks[0]["issue_ids"] == ["i1", "i2"]
        assert blocks[0]["photo_keys"] == ["photo_008.jpg"]
        assert blocks[0]["evidence_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Summary text generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummaryText:

    def test_one_bucket_one_liner(self, catalog_index):
        issues = [_make_issue("i1", "water_stain_ceiling")]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        assert result["listing"]["one_liner"] == "Most notable: Moisture & Mold."

    def test_two_bucket_one_liner(self, catalog_index):
        issues = [
            _make_issue("i1", "water_stain_ceiling"),
            _make_issue("i2", "peeling_or_discolored_paint"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        one_liner = result["listing"]["one_liner"]
        assert one_liner.startswith("Most notable:")
        assert "and" in one_liner
        # Exactly 2 buckets, so no "additional issues" clause
        assert "additional" not in one_liner

    def test_three_bucket_one_liner(self, catalog_index):
        issues = [
            _make_issue("i1", "water_stain_ceiling"),
            _make_issue("i2", "peeling_or_discolored_paint"),
            _make_issue("i3", "outdated_kitchen_finishes", catalog_item_kind="upgrade"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        one_liner = result["listing"]["one_liner"]
        assert "additional issues noted in 1 other area" in one_liner

    def test_bucket_summary_line_single_block(self, catalog_index):
        issues = [_make_issue("i1", "water_stain_ceiling")]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        bucket = result["buckets"][0]
        assert bucket["summary_line"] == "Top concern: Water Stain on Ceiling."

    def test_bucket_summary_line_two_blocks(self, catalog_index):
        issues = [
            _make_issue("i1", "water_stain_ceiling", scene_group="kitchen"),
            _make_issue("i2", "visible_mold_or_mildew", scene_group="bathroom"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        bucket = result["buckets"][0]
        line = bucket["summary_line"]
        assert line.startswith("Top concerns:")
        assert ";" in line


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism / stability
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:

    def test_same_input_same_output(self, catalog_index):
        issues = [
            _make_issue("i1", "water_stain_ceiling", photo_key="p1.jpg", scene_group="kitchen"),
            _make_issue("i2", "peeling_or_discolored_paint", photo_key="p2.jpg", scene_group="bedroom"),
            _make_issue("i3", "outdated_kitchen_finishes", photo_key="p3.jpg", scene_group="kitchen", catalog_item_kind="upgrade"),
            _make_issue("i4", "water_stain_ceiling", photo_key="p4.jpg", scene_group="bathroom"),
        ]

        r1 = build_property_summary_v1(
            property_key="prop1", run_id="run1",
            issues_flat=issues, catalog_index=catalog_index,
        )
        r2 = build_property_summary_v1(
            property_key="prop1", run_id="run1",
            issues_flat=issues, catalog_index=catalog_index,
        )

        # Strip generated_at since it'll differ by microseconds
        r1.pop("generated_at")
        r2.pop("generated_at")
        assert r1 == r2

    def test_ordering_stable_across_shuffled_input(self, catalog_index):
        """Even if issues_flat order changes, output bucket/scene order is deterministic."""
        issues_a = [
            _make_issue("i1", "water_stain_ceiling", scene_group="kitchen"),
            _make_issue("i2", "peeling_or_discolored_paint", scene_group="bedroom"),
        ]
        issues_b = list(reversed(issues_a))

        r1 = build_property_summary_v1(
            property_key="prop1", run_id="run1",
            issues_flat=issues_a, catalog_index=catalog_index,
        )
        r2 = build_property_summary_v1(
            property_key="prop1", run_id="run1",
            issues_flat=issues_b, catalog_index=catalog_index,
        )

        r1.pop("generated_at")
        r2.pop("generated_at")
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════════════════
# Listing-level stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestListingStats:

    def test_counts_and_buckets_touched(self, catalog_index):
        issues = [
            _make_issue("i1", "water_stain_ceiling"),
            _make_issue("i2", "peeling_or_discolored_paint"),
            _make_issue("i3", "outdated_kitchen_finishes", catalog_item_kind="upgrade"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        listing = result["listing"]
        assert listing["kind_counts"] == {
            "defect": 2, "degradation": 0, "modernization": 1,
        }
        assert len(listing["buckets_touched"]) == 3
        assert listing["top_severity"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Refinement: Scene-aware summary_line (Fix 1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSceneAwareSummaryLine:

    def test_summary_line_scene_aware(self, catalog_index):
        """Same item in bathroom AND bedroom → summary includes scene context."""
        issues = [
            _make_issue("i1", "scratched_flooring", scene_group="bathroom"),
            _make_issue("i2", "scratched_flooring", scene_group="bedroom"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        bucket = result["buckets"][0]
        line = bucket["summary_line"]
        # Should mention both scenes in parentheses
        assert "(bathroom, bedroom)" in line or "(bedroom, bathroom)" in line
        # Should be singular "Top concern:" since there's only 1 unique title
        assert line.startswith("Top concern:")
        assert "Scratched or Worn Flooring" in line

    def test_summary_line_no_duplicate_titles(self, catalog_index):
        """Same item in 2 scenes → single mention with scene context, not duplicated."""
        issues = [
            _make_issue("i1", "outdated_flooring_style", scene_group="bathroom",
                        catalog_item_kind="upgrade"),
            _make_issue("i2", "outdated_flooring_style", scene_group="bedroom",
                        catalog_item_kind="upgrade"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        bucket = result["buckets"][0]
        line = bucket["summary_line"]
        # Title should appear exactly once
        assert line.count("Outdated Flooring Style") == 1
        # Should NOT have the old duplicate pattern
        assert "outdated Flooring Style" not in line.split(";")[-1] if ";" in line else True


# ═══════════════════════════════════════════════════════════════════════════════
# Refinement: Upgrade severity cap (Fix 2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestModernizationSeverityCap:

    def test_modernization_capped_at_4(self, catalog_index):
        """Modernization with multi-scene boost → capped at MAX_MODERNIZATION_SEVERITY."""
        # outdated_flooring_style: base=2, scope(replace)=+2, kind(modernization)=-1, multi_scene=+2
        # raw = 2 + 2 - 1 + 2 = 5 → should be capped to 4
        issues = [
            _make_issue("i1", "outdated_flooring_style", scene_group="bathroom",
                        catalog_item_kind="upgrade"),
            _make_issue("i2", "outdated_flooring_style", scene_group="bedroom",
                        catalog_item_kind="upgrade"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        capped_blocks = 0
        for bucket in result["buckets"]:
            for scene in bucket["scenes"]:
                for block in scene["blocks"]:
                    if block["kind"] == "modernization":
                        capped_blocks += 1
                        assert block["display_severity"] <= MAX_MODERNIZATION_SEVERITY
                        # Verify the cap was applied (severity_calc shows capped=True)
                        assert block["severity_calc"]["raw_total"] == 5
                        assert block["severity_calc"]["capped"] is True
        assert capped_blocks == 2

    def test_defect_not_capped_at_4(self, catalog_index):
        """Defects can reach severity 5 — cap only applies to upgrades."""
        # visible_mold: base=4, scope(repair)=+1, kind(defect)=0, multi_scene=+2
        # raw = 4 + 1 + 0 + 2 = 7 → clamped to 5 (not capped at 4)
        issues = [
            _make_issue("i1", "visible_mold_or_mildew", scene_group="kitchen"),
            _make_issue("i2", "visible_mold_or_mildew", scene_group="bathroom"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        for bucket in result["buckets"]:
            for scene in bucket["scenes"]:
                for block in scene["blocks"]:
                    if block["kind"] == "defect":
                        assert block["display_severity"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# Refinement: Defect-aware bucket ranking (Fix 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDefectAwareBucketRanking:

    def test_defect_bucket_outranks_upgrade_bucket(self, catalog_index):
        """Defect bucket at sev=4 ranks above pure-upgrade bucket at sev=4 (capped from 5)."""
        issues = [
            # Electrical: defect, base=4, repair(+1), defect(0) → display=5
            # base=4, 1 photo(0), repair(+1), defect(0) = 5
            _make_issue("i1", "electrical_hazard", scene_group="kitchen"),
            # Flooring upgrade: base=2, replace(+2), upgrade(-1), multi_scene(+2) → 5 → capped to 4
            _make_issue("i2", "outdated_flooring_style", scene_group="bathroom",
                        catalog_item_kind="upgrade"),
            _make_issue("i3", "outdated_flooring_style", scene_group="bedroom",
                        catalog_item_kind="upgrade"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        buckets = result["buckets"]
        assert len(buckets) == 2
        # Electrical (defect, rank_score = 5+1=6) should be first
        assert buckets[0]["bucket_id"] == "electrical"
        assert buckets[0]["kind_counts"]["defect"] > 0
        # Flooring (modernization, rank_score = 4+0=4) should be second
        assert buckets[1]["bucket_id"] == "flooring"
        assert buckets[1]["kind_counts"]["defect"] == 0

    def test_max_base_severity_defect_field(self, catalog_index):
        """Bucket should include max_base_severity_defect for ranking."""
        issues = [
            _make_issue("i1", "electrical_hazard", scene_group="kitchen"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        bucket = result["buckets"][0]
        assert bucket["max_base_severity_defect"] == 4  # electrical_hazard base=4


# ═══════════════════════════════════════════════════════════════════════════════
# Refinement: Debug severity_calc field (Fix 4)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeverityCalcDebug:

    def test_severity_calc_debug_fields(self, catalog_index):
        """Each block has severity_calc dict with correct breakdown."""
        # water_stain_ceiling: base=3, 1 photo(0), repair(+1), defect(0) → raw=4
        issues = [_make_issue("i1", "water_stain_ceiling")]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        block = result["buckets"][0]["scenes"][0]["blocks"][0]
        calc = block["severity_calc"]
        assert calc["base"] == 3
        assert calc["evidence_boost"] == 0  # 1 photo → no boost
        assert calc["scope_boost"] == 1     # repair → +1
        assert calc["kind_boost"] == 0      # defect → 0 (neutral)
        assert calc["multi_scene_boost"] == 0  # single scene
        assert calc["raw_total"] == 4       # 3 + 0 + 1 + 0 + 0
        assert calc["capped"] is False      # raw == display, no clamping
        assert block["display_severity"] == 4

    def test_severity_calc_with_modernization_cap(self, catalog_index):
        """severity_calc.capped is True when the modernization cap is applied."""
        # outdated_flooring_style in 2 scenes: base=2, replace(+2), modernization(-1), multi(+2) → raw=5
        issues = [
            _make_issue("i1", "outdated_flooring_style", scene_group="bathroom",
                        catalog_item_kind="upgrade"),
            _make_issue("i2", "outdated_flooring_style", scene_group="bedroom",
                        catalog_item_kind="upgrade"),
        ]
        result = build_property_summary_v1(
            property_key="prop1",
            run_id="run1",
            issues_flat=issues,
            catalog_index=catalog_index,
        )
        for bucket in result["buckets"]:
            for scene in bucket["scenes"]:
                for block in scene["blocks"]:
                    calc = block["severity_calc"]
                    assert calc["raw_total"] == 5
                    assert calc["capped"] is True
                    assert block["display_severity"] == MAX_MODERNIZATION_SEVERITY


# ═══════════════════════════════════════════════════════════════════════════════
# Three-kind ontology (observation-kind-v2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestThreeKindOntology:

    def _v2_catalog_index(self):
        return load_catalog_index({
            "items": [
                {"id": "worn_carpet", "name": "Worn Carpet", "kind": "degradation",
                 "severity": 2, "trade_bucket": "flooring", "scope": "replace"},
                {"id": "dated_kitchen", "name": "Dated Kitchen", "kind": "modernization",
                 "severity": 2, "trade_bucket": "kitchen_cabinets_counters",
                 "scope": "replace"},
                {"id": "roof_leak", "name": "Roof Leak", "kind": "defect",
                 "severity": 4, "trade_bucket": "roof_gutters", "scope": "repair"},
            ],
            "trade_buckets": [
                {"id": "flooring", "name": "Flooring"},
                {"id": "kitchen_cabinets_counters", "name": "Kitchen"},
                {"id": "roof_gutters", "name": "Roof & Gutters"},
            ],
        })

    def test_degradation_not_penalized_and_not_capped(self):
        idx = self._v2_catalog_index()
        issues = [
            _make_issue("i1", "worn_carpet", scene_group="bedroom",
                        catalog_item_kind="degradation"),
            _make_issue("i2", "worn_carpet", scene_group="living_areas",
                        catalog_item_kind="degradation"),
        ]
        result = build_property_summary_v1(
            property_key="p", run_id="r", issues_flat=issues, catalog_index=idx)
        block = result["buckets"][0]["scenes"][0]["blocks"][0]
        assert block["kind"] == "degradation"
        # base=2, replace(+2), degradation(0), multi_scene(+2) → raw=6 → clamp 5, no cap
        assert block["severity_calc"]["kind_boost"] == 0
        assert block["display_severity"] == 5

    def test_kind_counts_cover_all_three_kinds(self):
        idx = self._v2_catalog_index()
        issues = [
            _make_issue("i1", "roof_leak", catalog_item_kind="defect"),
            _make_issue("i2", "worn_carpet", catalog_item_kind="degradation"),
            _make_issue("i3", "dated_kitchen", catalog_item_kind="modernization"),
        ]
        result = build_property_summary_v1(
            property_key="p", run_id="r", issues_flat=issues, catalog_index=idx)
        assert result["listing"]["kind_counts"] == {
            "defect": 1, "degradation": 1, "modernization": 1,
        }

    def test_mixed_kind_block_keeps_most_condition_like(self):
        """defect < degradation < modernization: min rank wins on merge."""
        idx = self._v2_catalog_index()
        issues = [
            _make_issue("i1", "worn_carpet", scene_group="bedroom",
                        catalog_item_kind="modernization"),
            _make_issue("i2", "worn_carpet", scene_group="bedroom",
                        catalog_item_kind="degradation"),
        ]
        result = build_property_summary_v1(
            property_key="p", run_id="r", issues_flat=issues, catalog_index=idx)
        block = result["buckets"][0]["scenes"][0]["blocks"][0]
        assert block["kind"] == "degradation"
