"""
Tests for tools.cost_factors — property market/size factor resolution and the
uniform post-hoc dollar scaler applied to the v4 estimate.
"""

import copy

import pytest

from tools.cost_factors import (
    PPSF_BASELINE,
    PPSF_CLAMP,
    PPSF_EXPONENT,
    SQFT_BASELINE,
    SQFT_CLAMP,
    SQFT_EXPONENT,
    resolve_property_cost_factor,
    scale_estimate_dollars,
)


# ─── resolve_property_cost_factor ────────────────────────────────────────────

class TestResolvePropertyCostFactor:

    def test_neutral_on_missing_metadata(self):
        for metadata in (None, {}):
            factor, audit = resolve_property_cost_factor(metadata)
            assert factor == 1.0
            assert audit["market_factor"] == 1.0
            assert audit["size_factor"] == 1.0
            assert "no_ppsf_signal_market_factor_neutral" in audit["reasons"]
            assert "no_sqft_size_factor_neutral" in audit["reasons"]

    def test_known_ppsf_and_sqft(self):
        factor, audit = resolve_property_cost_factor(
            {"price_per_sqft": 460, "sqft": 2400},
        )
        expected_market = (460 / PPSF_BASELINE) ** PPSF_EXPONENT
        expected_size = (2400 / SQFT_BASELINE) ** SQFT_EXPONENT
        assert audit["market_factor"] == pytest.approx(expected_market)
        assert audit["size_factor"] == pytest.approx(expected_size)
        assert factor == pytest.approx(expected_market * expected_size)

    def test_baseline_inputs_give_exactly_one(self):
        factor, _ = resolve_property_cost_factor(
            {"price_per_sqft": PPSF_BASELINE, "sqft": SQFT_BASELINE},
        )
        assert factor == 1.0

    def test_market_factor_clamps(self):
        low, _ = resolve_property_cost_factor({"price_per_sqft": 100, "sqft": 1800})
        high, _ = resolve_property_cost_factor({"price_per_sqft": 800, "sqft": 1800})
        assert low == PPSF_CLAMP[0]
        assert high == PPSF_CLAMP[1]
        mid, _ = resolve_property_cost_factor({"price_per_sqft": 500, "sqft": 1800})
        assert mid == pytest.approx((500 / PPSF_BASELINE) ** PPSF_EXPONENT)
        assert PPSF_CLAMP[0] < mid < PPSF_CLAMP[1]

    def test_size_factor_clamps(self):
        small, _ = resolve_property_cost_factor({"price_per_sqft": 230, "sqft": 400})
        large, _ = resolve_property_cost_factor({"price_per_sqft": 230, "sqft": 20000})
        assert small == SQFT_CLAMP[0]
        assert large == SQFT_CLAMP[1]

    def test_missing_ppsf_keeps_size_factor(self):
        factor, audit = resolve_property_cost_factor({"sqft": 2400})
        assert audit["market_factor"] == 1.0
        assert audit["ppsf"] is None
        assert "no_ppsf_signal_market_factor_neutral" in audit["reasons"]
        assert factor == pytest.approx((2400 / SQFT_BASELINE) ** SQFT_EXPONENT)

    def test_lookup_chain_area_beats_subject_beats_derived(self):
        metadata = {
            "area_price_per_sqft": 300,
            "price_per_sqft": 200,
            "list_price": 100_000,
            "sqft": 1000,
        }
        _, audit = resolve_property_cost_factor(metadata)
        assert audit["ppsf"] == 300
        assert audit["ppsf_source_key"] == "area_price_per_sqft"

        metadata.pop("area_price_per_sqft")
        _, audit = resolve_property_cost_factor(metadata)
        assert audit["ppsf"] == 200
        assert audit["ppsf_source_key"] == "price_per_sqft"

        metadata.pop("price_per_sqft")
        _, audit = resolve_property_cost_factor(metadata)
        assert audit["ppsf"] == 100.0
        assert audit["ppsf_source_key"] == "list_price/sqft"

    def test_string_metadata_values_are_coerced(self):
        _, audit = resolve_property_cost_factor(
            {"price_per_sqft": "$460", "sqft": "2,400"},
        )
        assert audit["ppsf"] == 460.0
        assert audit["sqft"] == 2400.0


# ─── scale_estimate_dollars ──────────────────────────────────────────────────

def _sample_estimate():
    package = {
        "package_id": "kitchen_modernization__kitchen_1",
        "cost_low": 1000,
        "cost_high": 2000,
        "cost_midpoint": 1500,
        "candidate_cost_low": 1000,
        "candidate_cost_high": 2000,
        "absorbed_total_low": 100,
        "absorbed_total_high": 200,
        "confidence_score": 0.62,
        "supporting_photo_count": 3,
    }
    return package, {
        "final_rehab": {"low": 100, "high": 301, "midpoint": 200, "basis": "x"},
        "latent_risk_exposure": {"low": 0, "high": 500, "midpoint": None},
        "groups": [{
            "group": "kitchen",
            "allocated_low": 50,
            "allocated_high": 90,
            "risk_exposure_high": 10,
            "item_count": 4,
            "severity": 3,
        }],
        "packages": [package],
        "package_candidates": [package],
    }


class TestScaleEstimateDollars:

    def test_factor_one_is_identity(self):
        _, estimate = _sample_estimate()
        snapshot = copy.deepcopy(estimate)
        scale_estimate_dollars(estimate, 1.0)
        assert estimate == snapshot

    def test_scales_suffix_keys_and_nested_buckets(self):
        _, estimate = _sample_estimate()
        scale_estimate_dollars(estimate, 1.5)
        assert estimate["final_rehab"]["low"] == 150
        assert estimate["final_rehab"]["high"] == 452  # round(301 * 1.5)
        assert estimate["groups"][0]["allocated_low"] == 75
        assert estimate["groups"][0]["allocated_high"] == 135
        assert estimate["groups"][0]["risk_exposure_high"] == 15
        assert estimate["latent_risk_exposure"]["high"] == 750

    def test_midpoints_recomputed_not_scaled(self):
        _, estimate = _sample_estimate()
        scale_estimate_dollars(estimate, 1.5)
        bucket = estimate["final_rehab"]
        assert bucket["midpoint"] == (bucket["low"] + bucket["high"]) // 2
        package = estimate["packages"][0]
        assert package["cost_midpoint"] == (
            (package["cost_low"] + package["cost_high"]) // 2
        )

    def test_none_midpoint_left_alone(self):
        _, estimate = _sample_estimate()
        scale_estimate_dollars(estimate, 1.5)
        assert estimate["latent_risk_exposure"]["midpoint"] is None

    def test_non_dollar_fields_untouched(self):
        _, estimate = _sample_estimate()
        scale_estimate_dollars(estimate, 1.5)
        group = estimate["groups"][0]
        assert group["item_count"] == 4
        assert group["severity"] == 3
        assert estimate["packages"][0]["confidence_score"] == 0.62
        assert estimate["packages"][0]["supporting_photo_count"] == 3
        assert estimate["final_rehab"]["basis"] == "x"

    def test_ints_stay_ints(self):
        _, estimate = _sample_estimate()
        scale_estimate_dollars(estimate, 0.75)
        assert isinstance(estimate["final_rehab"]["low"], int)
        assert isinstance(estimate["packages"][0]["cost_low"], int)
        # round(), not truncation: 301 * 0.75 = 225.75 → 226
        assert estimate["final_rehab"]["high"] == 226

    def test_aliased_package_dict_scaled_exactly_once(self):
        package, estimate = _sample_estimate()
        assert estimate["packages"][0] is estimate["package_candidates"][0]
        scale_estimate_dollars(estimate, 1.5)
        assert package["cost_low"] == 1500
        assert package["cost_high"] == 3000

    def test_aliased_bucket_dict_scaled_exactly_once(self):
        # Mirrors v4: reconciliation buckets are aliased to top-level keys.
        bucket = {"low": 100, "high": 200, "midpoint": 150}
        estimate = {
            "final_rehab": bucket,
            "reconciliation": {"final_rehab": bucket},
        }
        scale_estimate_dollars(estimate, 1.5)
        assert bucket["low"] == 150
        assert bucket["high"] == 300

    def test_headline_tier_band_scales_coherently(self):
        # Correlated tier buckets: band, straight sums, and midpoint all move
        # by the same factor — the blend is positively homogeneous, so scaling
        # after reconciliation is order-safe.
        bucket = {
            "low": 45_000,
            "high": 67_000,
            "midpoint": 56_000,
            "sum_low": 30_000,
            "sum_high": 80_000,
            "basis": "totals_by_scope_capped.required_rehab",
            "source": "renovation_estimate_v4",
        }
        estimate = {"final_rehab_resale_ready": bucket}
        scale_estimate_dollars(estimate, 1.5)
        assert bucket["low"] == 67_500
        assert bucket["high"] == 100_500
        assert bucket["sum_low"] == 45_000
        assert bucket["sum_high"] == 120_000
        assert bucket["midpoint"] == (bucket["low"] + bucket["high"]) // 2
        assert bucket["sum_low"] <= bucket["low"] <= bucket["high"] <= bucket["sum_high"]
