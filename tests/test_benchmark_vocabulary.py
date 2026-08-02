"""Frozen vocabulary: snapshot, fingerprint integrity, and drift detection.

The point of freezing is that a sealed gold set keeps validating after live code
changes. These tests assert against the *real* catalog on purpose — a snapshot
built from a synthetic fixture would not catch drift in the one that ships.
"""
import copy

import pytest

from tools.benchmarking import vocabulary as V
from tools.comparison_common import ComparisonError, atomic_json


class TestSnapshot:
    def test_captures_every_vocabulary_the_reference_depends_on(self, frozen_vocabulary):
        for key in (
            "scene_ids", "ui_scene_groups", "catalog_scene_group_tokens", "catalog_items",
            "actionability_by_catalog_item", "package_types", "package_categories",
            "package_rooms", "package_levels", "estimate_scopes", "rehab_scope_bands",
            "actionability", "catalog_version", "fingerprint",
        ):
            assert key in frozen_vocabulary, f"snapshot is missing {key}"

    def test_records_real_catalog_content(self, frozen_vocabulary):
        assert frozen_vocabulary["catalog_version"] == "2.1"
        assert len(frozen_vocabulary["catalog_items"]) > 50
        assert "damaged_drywall_or_cracks" in frozen_vocabulary["catalog_items"]

    def test_preserves_the_pool_scene_group_skew(self, frozen_vocabulary):
        """`pool` is a catalog retrieval token but not a UI group (it is a scene
        whose group is `exterior`). Recording both separately keeps the skew
        visible instead of quietly reconciling it."""
        assert "pool" in frozen_vocabulary["catalog_scene_group_tokens"]
        assert "pool" not in frozen_vocabulary["ui_scene_groups"]
        assert frozen_vocabulary["scene_ids"]["pool"] == "exterior"

    def test_empty_catalog_fails_closed(self):
        """load_issue_catalog swallows a missing file and returns no items. A
        snapshot built from that would accept any catalog id at all."""
        with pytest.raises(ComparisonError, match="no items"):
            V.snapshot({"items": []})

    def test_catalog_without_ids_fails_closed(self):
        with pytest.raises(ComparisonError, match="no items with an 'id'"):
            V.snapshot({"items": [{"name": "nameless"}]})

    def test_fingerprint_is_order_independent(self, issue_catalog):
        shuffled = dict(issue_catalog)
        shuffled["items"] = list(reversed(issue_catalog["items"]))
        assert V.snapshot(shuffled)["fingerprint"] == V.snapshot(issue_catalog)["fingerprint"]

    def test_fingerprint_excludes_itself(self, frozen_vocabulary):
        """Otherwise recomputing on load could never reproduce the stored value."""
        assert V.fingerprint(frozen_vocabulary) == frozen_vocabulary["fingerprint"]

    def test_fingerprint_changes_when_a_catalog_item_changes(self, issue_catalog):
        mutated = copy.deepcopy(issue_catalog)
        mutated["items"][0]["name"] = "Renamed For Test"
        assert V.snapshot(mutated)["fingerprint"] != V.snapshot(issue_catalog)["fingerprint"]


class TestDefaultActionability:
    @pytest.mark.parametrize("item,expected", [
        ({"trade_bucket": "cleaning_turnover", "kind": "defect", "scope": "repair"}, "turnover"),
        ({"kind": "defect", "scope": "service"}, "inspection_risk"),
        ({"kind": "upgrade", "scope": "cosmetic"}, "modernization"),
        ({"kind": "defect", "scope": "repair"}, "repair"),
        ({"kind": "defect", "scope": "replace"}, "repair"),
    ])
    def test_four_ordered_rules(self, item, expected):
        assert V.default_actionability(item) == expected

    def test_turnover_trade_wins_over_kind(self):
        """Rule order matters: a cleaning_turnover upgrade is turnover work."""
        assert V.default_actionability(
            {"trade_bucket": "cleaning_turnover", "kind": "upgrade", "scope": "cosmetic"}
        ) == "turnover"

    def test_every_derived_value_is_in_the_declared_vocabulary(self, frozen_vocabulary):
        declared = set(frozen_vocabulary["actionability"])
        assert set(frozen_vocabulary["actionability_by_catalog_item"].values()) <= declared


class TestLoad:
    def test_round_trips(self, tmp_path, frozen_vocabulary):
        path = tmp_path / "reference_vocabulary.json"
        atomic_json(path, frozen_vocabulary)
        assert V.load(path)["fingerprint"] == frozen_vocabulary["fingerprint"]

    def test_rejects_a_hand_edited_snapshot(self, tmp_path, frozen_vocabulary):
        """Editing the frozen vocabulary is the one way a sealed reference could
        start accepting values nobody reviewed."""
        path = tmp_path / "reference_vocabulary.json"
        tampered = copy.deepcopy(frozen_vocabulary)
        tampered["catalog_items"]["made_up_item"] = {"name": "Fake", "kind": "defect"}
        atomic_json(path, tampered)
        with pytest.raises(ComparisonError, match="fingerprint mismatch"):
            V.load(path)

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(ComparisonError, match="not found"):
            V.load(tmp_path / "nope.json")

    def test_rejects_unsupported_schema_version(self, tmp_path, frozen_vocabulary):
        path = tmp_path / "reference_vocabulary.json"
        future = copy.deepcopy(frozen_vocabulary)
        future["vocabulary_schema_version"] = 99
        future["fingerprint"] = V.fingerprint(future)
        atomic_json(path, future)
        with pytest.raises(ComparisonError, match="not supported"):
            V.load(path)


class TestDiff:
    def test_self_diff_is_clean(self, frozen_vocabulary, issue_catalog):
        changes = V.diff(frozen_vocabulary, issue_catalog=issue_catalog)
        assert changes["compatible"] is True
        assert V.summarize_diff(changes) == []

    def test_added_catalog_item_stays_compatible(self, frozen_vocabulary, issue_catalog):
        """New items mean the pipeline can say something the reference never
        reviewed — reported as unreviewed, not as incompatible."""
        extended = copy.deepcopy(issue_catalog)
        extended["items"].append({
            "id": "brand_new_item", "name": "Brand New", "kind": "defect",
            "scope": "repair", "tier": "work", "trade_bucket": "flooring",
        })
        changes = V.diff(frozen_vocabulary, issue_catalog=extended)
        assert changes["compatible"] is True
        assert changes["catalog_items"]["added"] == ["brand_new_item"]

    def test_removed_catalog_item_is_incompatible(self, frozen_vocabulary, issue_catalog):
        reduced = copy.deepcopy(issue_catalog)
        removed_id = reduced["items"].pop(0)["id"]
        changes = V.diff(frozen_vocabulary, issue_catalog=reduced)
        assert changes["compatible"] is False
        assert removed_id in changes["catalog_items"]["removed"]
        assert any("REMOVED" in line for line in V.summarize_diff(changes))

    def test_changed_catalog_item_is_reported(self, frozen_vocabulary, issue_catalog):
        mutated = copy.deepcopy(issue_catalog)
        mutated["items"][0]["kind"] = "upgrade" if mutated["items"][0]["kind"] == "defect" else "defect"
        changes = V.diff(frozen_vocabulary, issue_catalog=mutated)
        assert [c["id"] for c in changes["catalog_items"]["changed"]] == [mutated["items"][0]["id"]]

    def test_regrouped_scene_is_incompatible(self, frozen_vocabulary):
        """A scene changing groups silently remaps human room truth."""
        live = copy.deepcopy(frozen_vocabulary)
        live["scene_ids"]["pantry"] = "utility"
        changes = V.diff(frozen_vocabulary, live)
        assert changes["compatible"] is False
        assert changes["scene_ids"]["regrouped"] == [
            {"scene_id": "pantry", "frozen_group": "kitchen", "live_group": "utility"}
        ]

    def test_removed_package_type_is_incompatible(self, frozen_vocabulary):
        live = copy.deepcopy(frozen_vocabulary)
        live["package_types"] = [t for t in live["package_types"] if t != "kitchen_repair"]
        changes = V.diff(frozen_vocabulary, live)
        assert changes["compatible"] is False
        assert changes["sets"]["package_types"]["removed"] == ["kitchen_repair"]

    def test_catalog_version_bump_is_reported_without_being_fatal(self, frozen_vocabulary,
                                                                 issue_catalog):
        """A catalog change is a treatment dimension, not an incompatibility."""
        bumped = copy.deepcopy(issue_catalog)
        bumped["version"] = "2.2"
        changes = V.diff(frozen_vocabulary, issue_catalog=bumped)
        assert changes["compatible"] is True
        assert any("catalog version" in line for line in V.summarize_diff(changes))

    def test_requires_a_live_snapshot_or_catalog(self, frozen_vocabulary):
        with pytest.raises(ComparisonError, match="live snapshot or an issue_catalog"):
            V.diff(frozen_vocabulary)
