from tools.estimate_sanity import build_estimate_sanity_flags


def _estimate(package_high=0, worst_high=0, final_high=None):
    if final_high is None:
        final_high = package_high
    return {
        "package_adjusted_rehab": {"low": 0, "high": package_high, "midpoint": package_high // 2},
        "worst_case_exposure": {"low": 0, "high": worst_high, "midpoint": None},
        "final_rehab": {"low": 0, "high": final_high, "midpoint": final_high // 2},
    }


def _codes(flags):
    return {flag["code"] for flag in flags}


def test_warns_when_package_adjusted_high_exceeds_50_percent_of_price():
    flags = build_estimate_sanity_flags(
        _estimate(package_high=120000, worst_high=120000),
        {"list_price": 199700},
        [],
    )

    flag = next(f for f in flags if f["code"] == "package_adjusted_high_gt_50pct_price")
    assert flag["severity"] == "warning"
    assert flag["value"] == 0.6
    assert flag["numerator"] == 120000
    assert flag["denominator"] == 199700
    assert flag["threshold"] == 0.50
    assert flag["compared_field"] == "package_adjusted_rehab.high/list_price"


def test_strong_warning_when_worst_case_exceeds_80_percent_of_price():
    flags = build_estimate_sanity_flags(
        _estimate(package_high=120000, worst_high=188767),
        {"list_price": 199700},
        [],
    )

    flag = next(f for f in flags if f["code"] == "worst_case_high_gt_80pct_price")
    assert flag["severity"] == "strong_warning"
    assert flag["value"] == 0.95
    assert "inspection-dependent" in flag["message"]
    assert flag["compared_field"] == "worst_case_exposure.high/list_price"


def test_warns_when_package_adjusted_high_exceeds_100_per_sqft():
    flags = build_estimate_sanity_flags(
        _estimate(package_high=185000, worst_high=185000),
        {"sqft": 1800},
        [],
    )

    flag = next(f for f in flags if f["code"] == "package_adjusted_high_gt_100_per_sqft")
    assert flag["severity"] == "warning"
    assert flag["value"] == 102.78
    assert flag["threshold"] == 100.0
    assert flag["compared_field"] == "package_adjusted_rehab.high/sqft"


def test_strong_warning_when_worst_case_exceeds_150_per_sqft():
    flags = build_estimate_sanity_flags(
        _estimate(package_high=100000, worst_high=280000),
        {"sqft": 1800},
        [],
    )

    flag = next(f for f in flags if f["code"] == "worst_case_high_gt_150_per_sqft")
    assert flag["severity"] == "strong_warning"
    assert flag["value"] == 155.56
    assert flag["compared_field"] == "worst_case_exposure.high/sqft"


def test_sqft_flags_do_not_use_final_rehab_high():
    flags = build_estimate_sanity_flags(
        _estimate(package_high=10000, worst_high=10000, final_high=500000),
        {"sqft": 1800},
        [],
    )

    assert "package_adjusted_high_gt_100_per_sqft" not in _codes(flags)
    assert "worst_case_high_gt_150_per_sqft" not in _codes(flags)


def test_warns_when_billable_kitchens_gt_one_for_single_family():
    flags = build_estimate_sanity_flags(
        _estimate(),
        {"property_type": "single_family"},
        [
            {"estimate_unit_id": "kitchen_primary", "unit_type": "kitchen"},
            {"estimate_unit_id": "kitchen_secondary_1", "unit_type": "kitchen"},
        ],
    )

    flag = next(f for f in flags if f["code"] == "multiple_billable_kitchens_single_family")
    assert flag["severity"] == "warning"
    assert flag["value"] == 2


def test_multi_kitchen_evidence_suppresses_single_family_kitchen_warning():
    flags = build_estimate_sanity_flags(
        _estimate(),
        {"property_type": "single-family"},
        [
            {
                "estimate_unit_id": "kitchen_primary",
                "unit_type": "kitchen",
                "multi_kitchen_evidence": True,
            },
            {"estimate_unit_id": "kitchen_secondary_1", "unit_type": "kitchen"},
        ],
    )

    assert "multiple_billable_kitchens_single_family" not in _codes(flags)


def test_fractional_metadata_baths_are_ceiled_before_comparison():
    flags = build_estimate_sanity_flags(
        _estimate(),
        {"baths": "1.5"},
        [
            {"estimate_unit_id": "bathroom_primary", "unit_type": "bathroom"},
            {"estimate_unit_id": "bathroom_2", "unit_type": "bathroom"},
        ],
    )

    assert "billable_bathrooms_gt_metadata_baths" not in _codes(flags)


def test_warns_when_billable_bathrooms_exceed_ceiled_metadata_baths():
    flags = build_estimate_sanity_flags(
        _estimate(),
        {"baths": 1.5},
        [
            {"estimate_unit_id": "bathroom_primary", "unit_type": "bathroom"},
            {"estimate_unit_id": "bathroom_2", "unit_type": "bathroom"},
            {"estimate_unit_id": "bathroom_3", "unit_type": "bathroom"},
        ],
    )

    flag = next(f for f in flags if f["code"] == "billable_bathrooms_gt_metadata_baths")
    assert flag["value"] == 3
    assert flag["threshold"] == 2


def test_missing_price_sqft_and_bath_metadata_produce_no_flags():
    flags = build_estimate_sanity_flags(
        _estimate(package_high=500000, worst_high=700000),
        {"property_type": "townhouse"},
        [{"estimate_unit_id": "bathroom_primary", "unit_type": "bathroom"}],
    )

    assert flags == []


def test_warns_when_billable_bedrooms_exceed_metadata_beds():
    flags = build_estimate_sanity_flags(
        _estimate(),
        {"beds": 2},
        [
            {"estimate_unit_id": "bedroom_primary", "unit_type": "bedroom"},
            {"estimate_unit_id": "bedroom_2", "unit_type": "bedroom"},
            {"estimate_unit_id": "bedroom_3", "unit_type": "bedroom"},
        ],
    )

    flag = next(f for f in flags if f["code"] == "billable_bedrooms_gt_metadata_beds")
    assert flag["severity"] == "warning"
    assert flag["value"] == 3
    assert flag["threshold"] == 2
    assert flag["numerator"] == 3
    assert flag["denominator"] == 2
    assert flag["compared_field"] == "billable_bedroom_count"


def test_fractional_metadata_beds_are_ceiled_before_comparison():
    flags = build_estimate_sanity_flags(
        _estimate(),
        {"beds": "2.5"},
        [
            {"estimate_unit_id": "bedroom_primary", "unit_type": "bedroom"},
            {"estimate_unit_id": "bedroom_2", "unit_type": "bedroom"},
            {"estimate_unit_id": "bedroom_3", "unit_type": "bedroom"},
        ],
    )

    assert "billable_bedrooms_gt_metadata_beds" not in _codes(flags)


def test_bedrooms_under_metadata_produce_no_flag():
    flags = build_estimate_sanity_flags(
        _estimate(),
        {"beds": 3},
        [{"estimate_unit_id": "bedroom_primary", "unit_type": "bedroom"}],
    )

    assert "billable_bedrooms_gt_metadata_beds" not in _codes(flags)


def test_missing_bedroom_metadata_produces_no_bedroom_flag():
    flags = build_estimate_sanity_flags(
        _estimate(),
        {},
        [
            {"estimate_unit_id": "bedroom_primary", "unit_type": "bedroom"},
            {"estimate_unit_id": "bedroom_2", "unit_type": "bedroom"},
        ],
    )

    assert "billable_bedrooms_gt_metadata_beds" not in _codes(flags)
