from tools.costing import compute_scoring


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
