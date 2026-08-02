from tools.compare_reno_estimates import compare, format_report
from tools.estimate_units import build_estimate_units
from tools.renovation_estimate import compute_renovation_estimate
from tools.renovation_estimate_v4 import compute_renovation_estimate_v4
from tools.room_surrogates import build_room_surrogates


def _make_photos(*entries):
    photos = {}
    for i, entry in enumerate(entries, start=1):
        if isinstance(entry, tuple):
            scene, photo_key = entry
        else:
            scene = entry
            photo_key = f"photo_{i:03d}.jpg"
        photos[photo_key] = {
            "photo": {"photo_key": photo_key, "index": i},
            "scene": {"id": scene, "group": "irrelevant"},
        }
    return photos


def _make_catalog(*items):
    return {"items": list(items), "trade_buckets": []}


def _make_item(item_id, *, estimate, trade_bucket, kind="defect", cost=None):
    return {
        "id": item_id,
        "name": item_id.replace("_", " ").title(),
        "kind": kind,
        "severity": 3,
        "scope": "replace",
        "trade_bucket": trade_bucket,
        "estimate": estimate,
        "cost": cost or {"mode": "heuristic"},
    }


def _make_issue(catalog_item_id, *, photo_key, issue_id, scene_group="kitchen"):
    return {
        "issue_id": issue_id,
        "catalog_item_id": catalog_item_id,
        "catalog_item_kind": "upgrade",
        "scene_group": scene_group,
        "photo_key": photo_key,
        "description": f"Dated {scene_group} finishes.",
        "label": "defect_or_damage",
    }


def _line_items(estimate):
    return [
        line_item
        for group in estimate.get("groups", [])
        for line_item in group.get("line_items", [])
    ]


def test_repeated_nonconsecutive_kitchen_surrogates_merge_to_one_billable_unit():
    photos = _make_photos(
        ("kitchen", "photo_002.jpg"),
        ("living_room", "photo_005.jpg"),
        ("kitchen", "photo_012.jpg"),
        ("dining_room", "photo_015.jpg"),
        ("kitchen", "photo_019.jpg"),
    )
    surrogates = build_room_surrogates(photos)["room_surrogates"]

    result = build_estimate_units(photos, surrogates)

    kitchen_units = [u for u in result["estimate_units"] if u["unit_type"] == "kitchen"]
    assert len(kitchen_units) == 1
    assert kitchen_units[0] == {
        "estimate_unit_id": "kitchen_primary",
        "unit_type": "kitchen",
        "source_room_surrogate_ids": ["kitchen_1", "kitchen_2", "kitchen_3"],
        "photo_ids": ["photo_002.jpg", "photo_012.jpg", "photo_019.jpg"],
        "confidence": "default_assumption",
        "merge_reason": "single_family_default_one_kitchen",
    }
    assert result["room_surrogate_to_estimate_unit_id"]["kitchen_1"] == "kitchen_primary"
    assert result["room_surrogate_to_estimate_unit_id"]["kitchen_2"] == "kitchen_primary"
    assert result["room_surrogate_to_estimate_unit_id"]["kitchen_3"] == "kitchen_primary"
    assert result["merge_decisions"] == [{
        "from": ["kitchen_1", "kitchen_2", "kitchen_3"],
        "to": "kitchen_primary",
        "reason": "repeated_kitchen_surrogates_merged_by_default",
        "confidence": "medium",
    }]


def test_metadata_multi_kitchen_evidence_is_marked_on_distinct_kitchen_units():
    photos = _make_photos(
        ("kitchen", "photo_002.jpg"),
        ("living_room", "photo_005.jpg"),
        ("kitchen", "photo_012.jpg"),
    )
    surrogates = build_room_surrogates(photos)["room_surrogates"]

    result = build_estimate_units(
        photos,
        surrogates,
        property_metadata={"kitchen_count": 2},
    )

    kitchen_units = [u for u in result["estimate_units"] if u["unit_type"] == "kitchen"]
    assert len(kitchen_units) == 2
    assert all(u["multi_kitchen_evidence"] is True for u in kitchen_units)


def test_single_bath_metadata_caps_repeated_bathroom_surrogates_to_one_unit():
    photos = _make_photos("bathroom", "bedroom", "bathroom")
    surrogates = build_room_surrogates(photos)["room_surrogates"]

    result = build_estimate_units(
        photos,
        surrogates,
        property_metadata={"full_baths": 1, "half_baths": 0},
    )

    assert result["room_surrogate_to_estimate_unit_id"]["bathroom_1"] == "bathroom_primary"
    assert result["room_surrogate_to_estimate_unit_id"]["bathroom_2"] == "bathroom_primary"
    bathroom_units = [u for u in result["estimate_units"] if u["unit_type"] == "bathroom"]
    assert len(bathroom_units) == 1
    assert bathroom_units[0]["source_room_surrogate_ids"] == ["bathroom_1", "bathroom_2"]
    assert bathroom_units[0]["merge_reason"] == "metadata_single_bath_cap"


def test_two_bath_metadata_allows_two_units_but_does_not_force_split_without_evidence():
    photos = _make_photos("bathroom", "bedroom", "bathroom")
    surrogates = build_room_surrogates(photos)["room_surrogates"]

    result = build_estimate_units(
        photos,
        surrogates,
        property_metadata={"bath_count": 2},
    )

    bathroom_units = [u for u in result["estimate_units"] if u["unit_type"] == "bathroom"]
    assert len(bathroom_units) == 1
    assert bathroom_units[0]["estimate_unit_id"] == "bathroom_primary"
    assert bathroom_units[0]["source_room_surrogate_ids"] == ["bathroom_1", "bathroom_2"]
    assert bathroom_units[0]["merge_reason"] == (
        "bathroom_metadata_allows_two_but_weak_evidence_merged"
    )


def test_fractional_bath_metadata_is_ceiled_for_unit_cap():
    photos = _make_photos(
        ("bathroom", "photo_001.jpg"),
        ("bedroom", "photo_002.jpg"),
        ("bathroom", "photo_003.jpg"),
        ("kitchen", "photo_004.jpg"),
        ("bathroom", "photo_005.jpg"),
    )
    photos["photo_001.jpg"]["caption"] = "Primary bathroom."
    photos["photo_003.jpg"]["caption"] = "Guest bathroom."
    photos["photo_005.jpg"]["caption"] = "Powder room."
    surrogates = build_room_surrogates(photos)["room_surrogates"]

    result = build_estimate_units(
        photos,
        surrogates,
        property_metadata={"baths": 1.5},
    )

    bathroom_units = [u for u in result["estimate_units"] if u["unit_type"] == "bathroom"]
    assert len(bathroom_units) == 2
    assert result["room_surrogate_to_estimate_unit_id"]["bathroom_3"] == "bathroom_2"
    assert bathroom_units[1]["merge_reason"] == "bathroom_metadata_cap_applied"


def test_bed_metadata_caps_four_bedroom_surrogates_to_three_units():
    photos = _make_photos(
        ("bedroom", "photo_001.jpg"),
        ("bathroom", "photo_002.jpg"),
        ("bedroom", "photo_003.jpg"),
        ("kitchen", "photo_004.jpg"),
        ("bedroom", "photo_005.jpg"),
        ("living_room", "photo_006.jpg"),
        ("bedroom", "photo_007.jpg"),
    )
    surrogates = build_room_surrogates(photos)["room_surrogates"]

    result = build_estimate_units(
        photos,
        surrogates,
        property_metadata={"beds": 3},
    )

    bedroom_units = [u for u in result["estimate_units"] if u["unit_type"] == "bedroom"]
    assert len(bedroom_units) == 3
    assert result["room_surrogate_to_estimate_unit_id"]["bedroom_1"] == "bedroom_primary"
    assert result["room_surrogate_to_estimate_unit_id"]["bedroom_2"] == "bedroom_2"
    assert result["room_surrogate_to_estimate_unit_id"]["bedroom_3"] == "bedroom_3"
    assert result["room_surrogate_to_estimate_unit_id"]["bedroom_4"] == "bedroom_3"
    assert bedroom_units[-1]["source_room_surrogate_ids"] == ["bedroom_3", "bedroom_4"]
    assert bedroom_units[-1]["merge_reason"] == "bedroom_metadata_cap_applied"


def test_repeated_living_room_surrogates_collapse_to_one_unit():
    # living -> kitchen -> living -> kitchen -> living splits into 3 living
    # surrogates; single-instance collapse folds them into one billable unit.
    photos = _make_photos(
        ("living_room", "photo_001.jpg"),
        ("kitchen", "photo_002.jpg"),
        ("living_room", "photo_003.jpg"),
        ("kitchen", "photo_004.jpg"),
        ("living_room", "photo_005.jpg"),
    )
    surrogates = build_room_surrogates(photos)["room_surrogates"]

    result = build_estimate_units(photos, surrogates)

    living_units = [u for u in result["estimate_units"] if u["unit_type"] == "living_room"]
    assert len(living_units) == 1
    assert living_units[0]["estimate_unit_id"] == "living_room_primary"
    assert living_units[0]["merge_reason"] == "single_instance_scene_collapsed"
    assert living_units[0]["source_room_surrogate_ids"] == [
        "living_room_1", "living_room_2", "living_room_3",
    ]
    assert result["room_surrogate_to_estimate_unit_id"]["living_room_1"] == "living_room_primary"
    assert result["room_surrogate_to_estimate_unit_id"]["living_room_3"] == "living_room_primary"
    assert any(
        d["to"] == "living_room_primary"
        and d["reason"] == "repeated_living_room_surrogates_merged_single_instance_default"
        for d in result["merge_decisions"]
    )


def test_dining_and_garage_each_collapse_to_one_unit():
    photos = _make_photos(
        ("dining_room", "photo_001.jpg"),
        ("kitchen", "photo_002.jpg"),
        ("dining_room", "photo_003.jpg"),
        ("garage", "photo_004.jpg"),
        ("kitchen", "photo_005.jpg"),
        ("garage", "photo_006.jpg"),
    )
    surrogates = build_room_surrogates(photos)["room_surrogates"]

    result = build_estimate_units(photos, surrogates)

    by_type = {}
    for unit in result["estimate_units"]:
        by_type.setdefault(unit["unit_type"], []).append(unit["estimate_unit_id"])
    assert by_type["dining_room"] == ["dining_room_primary"]
    assert by_type["garage"] == ["garage_primary"]


def test_bedrooms_without_metadata_are_not_collapsed_as_single_instance():
    # Bedrooms are deliberately excluded from single-instance collapse: with no
    # metadata cap they stay distinct billable units (multiple real bedrooms).
    photos = _make_photos(
        ("bedroom", "photo_001.jpg"),
        ("bathroom", "photo_002.jpg"),
        ("bedroom", "photo_003.jpg"),
        ("kitchen", "photo_004.jpg"),
        ("bedroom", "photo_005.jpg"),
    )
    surrogates = build_room_surrogates(photos)["room_surrogates"]

    result = build_estimate_units(photos, surrogates)

    bedroom_units = [u for u in result["estimate_units"] if u["unit_type"] == "bedroom"]
    assert len(bedroom_units) == 3
    assert {u["estimate_unit_id"] for u in bedroom_units} == {
        "bedroom_1", "bedroom_2", "bedroom_3",
    }
    assert all(
        u["merge_reason"] != "single_instance_scene_collapsed" for u in bedroom_units
    )


def test_v4_candidates_use_estimate_unit_id_for_per_kitchen_costs():
    estimate = {
        "estimate_tier": "high",
        "strategy": "replace_only",
        "group": "kitchen",
        "stack_behavior": "sum",
        "unit_policy": "per_kitchen",
    }
    cost = {
        "mode": "allowance",
        "cost_source": "manual",
        "base_low": 1000,
        "base_high": 2000,
        "per_occurrence_low": 1000,
        "per_occurrence_high": 2000,
        "cap_low": 10000,
        "cap_high": 20000,
    }
    catalog = _make_catalog(_make_item(
        "outdated_kitchen_finishes",
        estimate=estimate,
        trade_bucket="kitchen_cabinets_counters",
        kind="upgrade",
        cost=cost,
    ))
    photos = _make_photos(
        ("kitchen", "photo_002.jpg"),
        ("living_room", "photo_005.jpg"),
        ("kitchen", "photo_012.jpg"),
        ("dining_room", "photo_015.jpg"),
        ("kitchen", "photo_019.jpg"),
    )
    issues = [
        _make_issue("outdated_kitchen_finishes", photo_key="photo_002.jpg", issue_id="kit_1"),
        _make_issue("outdated_kitchen_finishes", photo_key="photo_012.jpg", issue_id="kit_2"),
        _make_issue("outdated_kitchen_finishes", photo_key="photo_019.jpg", issue_id="kit_3"),
    ]

    v3 = compute_renovation_estimate(issues, catalog)
    v4 = compute_renovation_estimate_v4(issues, catalog, photos)

    line_items = _line_items(v4)
    assert len(line_items) == 1
    line_item = line_items[0]
    assert line_item["catalog_item_id"] == "outdated_kitchen_finishes"
    assert line_item["estimate_unit_count"] == 1
    assert line_item["billable_estimate_unit_id"] == "kitchen_primary"
    assert line_item["cost_high"] == 2000
    assert line_item["source_room_surrogate_ids"] == [
        "kitchen_1",
        "kitchen_2",
        "kitchen_3",
    ]
    assert line_item["unit_members"][0]["estimate_unit_id"] == "kitchen_primary"
    assert line_item["unit_members"][0]["source_room_surrogate_ids"] == [
        "kitchen_1",
        "kitchen_2",
        "kitchen_3",
    ]
    assert v4["estimate_units"][0]["estimate_unit_id"] == "kitchen_primary"

    report = format_report(compare({
        "property": {"property_key": "test_property"},
        "run": {"run_id": "test_run"},
        "renovation_estimate": v3,
        "renovation_estimate_v4": v4,
    }))
    assert "billable kitchens: 1" in report
    assert "kitchen_primary <- kitchen_1, kitchen_2, kitchen_3" in report


def test_v4_per_room_bedroom_costs_use_bed_metadata_cap():
    estimate = {
        "estimate_tier": "medium",
        "strategy": "repair_only",
        "group": "flooring",
        "stack_behavior": "sum",
        "unit_policy": "per_room",
    }
    cost = {
        "mode": "allowance",
        "cost_source": "manual",
        "base_low": 1000,
        "base_high": 2000,
        "per_occurrence_low": 500,
        "per_occurrence_high": 1000,
        "cap_low": 10000,
        "cap_high": 20000,
    }
    catalog = _make_catalog(_make_item(
        "worn_bedroom_flooring",
        estimate=estimate,
        trade_bucket="flooring",
        kind="defect",
        cost=cost,
    ))
    photos = _make_photos(
        ("bedroom", "photo_001.jpg"),
        ("bathroom", "photo_002.jpg"),
        ("bedroom", "photo_003.jpg"),
        ("kitchen", "photo_004.jpg"),
        ("bedroom", "photo_005.jpg"),
        ("living_room", "photo_006.jpg"),
        ("bedroom", "photo_007.jpg"),
    )
    issues = [
        _make_issue(
            "worn_bedroom_flooring",
            photo_key=photo_key,
            issue_id=f"bed_floor_{idx}",
            scene_group="bedroom",
        )
        for idx, photo_key in enumerate(
            ("photo_001.jpg", "photo_003.jpg", "photo_005.jpg", "photo_007.jpg"),
            start=1,
        )
    ]

    v4 = compute_renovation_estimate_v4(
        issues,
        catalog,
        photos,
        property_metadata={"beds": 3},
    )

    line_items = _line_items(v4)
    assert len(line_items) == 1
    line_item = line_items[0]
    assert line_item["estimate_unit_count"] == 3
    assert line_item["estimate_unit_label"] == "3 rooms"
    assert line_item["cost_high"] == 4000
    merged_member = next(
        member
        for member in line_item["unit_members"]
        if member["unit_key"] == "bedroom_3"
    )
    assert merged_member["source_room_surrogate_ids"] == [
        "bedroom_3",
        "bedroom_4",
    ]


def test_exterior_photos_collapse_to_one_property_level_unit():
    # Exterior scenes open no per-scene surrogate, so without the property-level
    # exterior identity these photos would carry no estimate_unit_id at all and
    # every exterior issue would price in its own bucket.
    photos = _make_photos(
        ("exterior_front", "photo_001.jpg"),
        ("kitchen", "photo_002.jpg"),
        ("exterior_back", "photo_003.jpg"),
        ("yard", "photo_004.jpg"),
    )
    surrogates = build_room_surrogates(photos)["room_surrogates"]

    result = build_estimate_units(photos, surrogates)

    exterior_units = [u for u in result["estimate_units"] if u["unit_type"] == "exterior"]
    assert len(exterior_units) == 1
    assert exterior_units[0]["estimate_unit_id"] == "exterior_primary"
    for photo_key in ("photo_001.jpg", "photo_003.jpg", "photo_004.jpg"):
        assert result["photo_to_estimate_unit_id"][photo_key] == "exterior_primary"
    # The interior photo is untouched.
    assert result["photo_to_estimate_unit_id"]["photo_002.jpg"] == "kitchen_primary"


def test_interior_units_unchanged_when_exterior_photos_present():
    interior = _make_photos(
        ("kitchen", "photo_001.jpg"),
        ("bathroom", "photo_002.jpg"),
        ("bedroom", "photo_003.jpg"),
    )
    with_exterior = dict(interior)
    with_exterior["photo_004.jpg"] = {
        "photo": {"photo_key": "photo_004.jpg", "index": 4},
        "scene": {"id": "exterior_front", "group": "irrelevant"},
    }

    before = build_estimate_units(
        interior, build_room_surrogates(interior)["room_surrogates"])
    after = build_estimate_units(
        with_exterior, build_room_surrogates(with_exterior)["room_surrogates"])

    interior_ids = [u["estimate_unit_id"] for u in before["estimate_units"]]
    after_ids = [u["estimate_unit_id"] for u in after["estimate_units"]]
    assert after_ids == interior_ids + ["exterior_primary"]
    for photo_key in interior:
        assert (after["photo_to_estimate_unit_id"][photo_key]
                == before["photo_to_estimate_unit_id"][photo_key])
