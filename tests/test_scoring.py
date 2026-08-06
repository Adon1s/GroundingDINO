import pytest

from tools.costing import (
    CatalogDataError,
    compute_item_cost_range,
    compute_scoring,
    kind_multiplier,
)


def _catalog():
    return {
        "items": [
            {
                "id": "minor_paint",
                "name": "Minor Paint Wear",
                "kind": "defect",
                "severity": 1,
                "scope": "cosmetic",
                "trade_bucket": "paint_drywall",
            },
            {
                "id": "major_roof",
                "name": "Roof Damage",
                "kind": "defect",
                "severity": 4,
                "scope": "repair",
                "trade_bucket": "roof_gutters",
            },
        ]
    }


def _issue(item_id):
    return {
        "issue_id": f"iss_{item_id}",
        "catalog_item_id": item_id,
        "status": "confirmed",
        "scene_group": "living_areas",
        "photo_key": "img.jpg",
    }


def test_minor_items_affect_rehab_score():
    empty = compute_scoring([], _catalog(), n_photos=20)
    with_minor = compute_scoring([_issue("minor_paint")], _catalog(), n_photos=20)

    assert with_minor["rehab_score"] > empty["rehab_score"]
    assert with_minor["raw_points"] > empty["raw_points"]


def test_scoring_contains_no_dollar_totals_or_per_item_costs():
    scoring = compute_scoring([_issue("major_roof")], _catalog(), n_photos=20)

    assert scoring["version"] == "scoring_v1"
    assert "costs" not in scoring
    assert "trade_breakdown" not in scoring
    assert "project_scope_breakdown" not in scoring
    assert "total_low" not in scoring
    assert "total_high" not in scoring
    assert all("cost_low" not in item and "cost_high" not in item for item in scoring["per_item"])


class TestKindMultiplierFailLoud:
    """Unknown kinds must never silently price at 1.0 (Task 3, deferred to
    the pricing phase). defect/upgrade keep their v1 values."""

    def test_v1_kinds_unchanged(self):
        assert kind_multiplier("defect", phase="costing") == 1.0
        assert kind_multiplier("upgrade", phase="costing") == 0.6

    @pytest.mark.parametrize("kind", ["degradation", "modernization", "", None, "junk"])
    def test_unpriced_kind_raises(self, kind):
        with pytest.raises(CatalogDataError) as exc:
            kind_multiplier(kind, phase="scoring", item_id="worn_roof_shingles")
        msg = str(exc.value)
        assert "scoring" in msg
        assert "worn_roof_shingles" in msg
        assert "pricing" in msg

    def test_cost_range_raises_for_unpriced_kind(self):
        cost = {"mode": "allowance", "cost_source": "catalog", "base_low": 100, "base_high": 400}
        with pytest.raises(CatalogDataError):
            compute_item_cost_range(cost, 1, "modernization", "repair", "paint_drywall")

    def test_manual_allowance_stays_exempt_from_kind_multiplier(self):
        cost = {"mode": "allowance", "cost_source": "manual", "base_low": 100, "base_high": 400}
        low, high = compute_item_cost_range(cost, 1, "modernization", "repair", "paint_drywall")
        assert (low, high) == (100, 400)

    def test_upgrade_cost_range_byte_identical_to_v1(self):
        cost = {"mode": "allowance", "cost_source": "catalog", "base_low": 100, "base_high": 400}
        low, high = compute_item_cost_range(cost, 1, "upgrade", "repair", "safety_general")
        assert (low, high) == (60, 240)  # kind 0.6 * repair 1.0 * safety_general 1.0

    def test_compute_scoring_raises_naming_the_item(self):
        catalog = {
            "items": [{
                "id": "worn_or_stained_carpet",
                "kind": "degradation",
                "severity": 2,
                "scope": "replace",
                "trade_bucket": "flooring",
            }]
        }
        with pytest.raises(CatalogDataError, match="worn_or_stained_carpet"):
            compute_scoring([_issue("worn_or_stained_carpet")], catalog, n_photos=20)
