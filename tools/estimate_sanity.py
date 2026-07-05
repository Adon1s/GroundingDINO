"""Metadata sanity flags for renovation estimates.

These checks are intentionally additive: they annotate suspicious estimates
without capping or changing the estimate buckets.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


PACKAGE_PRICE_THRESHOLD = 0.50
WORST_CASE_PRICE_THRESHOLD = 0.80
PACKAGE_SQFT_THRESHOLD = 100.0
WORST_CASE_SQFT_THRESHOLD = 150.0


def build_estimate_sanity_flags(
    estimate: dict,
    property_metadata: dict | None,
    estimate_units: dict | list | None,
) -> list[dict]:
    """Build non-capping sanity flags from estimate buckets and metadata."""
    estimate = estimate or {}
    metadata = property_metadata or {}
    flags: List[Dict[str, Any]] = []

    list_price = _metadata_number(metadata, "list_price", "price", "listing_price")
    if list_price and list_price > 0:
        package_high = _bucket_high(estimate, "package_adjusted_rehab")
        if package_high is not None:
            ratio = package_high / list_price
            if ratio > PACKAGE_PRICE_THRESHOLD:
                flags.append(_ratio_flag(
                    code="package_adjusted_high_gt_50pct_price",
                    severity="warning",
                    message="Package-adjusted rehab high exceeds 50% of list price.",
                    value=ratio,
                    numerator=package_high,
                    denominator=list_price,
                    threshold=PACKAGE_PRICE_THRESHOLD,
                    compared_field="package_adjusted_rehab.high/list_price",
                ))

        worst_case_high = _bucket_high(estimate, "worst_case_exposure")
        if worst_case_high is not None:
            ratio = worst_case_high / list_price
            if ratio > WORST_CASE_PRICE_THRESHOLD:
                flags.append(_ratio_flag(
                    code="worst_case_high_gt_80pct_price",
                    severity="strong_warning",
                    message=(
                        "Worst-case exposure exceeds 80% of list price; "
                        "treat as inspection-dependent."
                    ),
                    value=ratio,
                    numerator=worst_case_high,
                    denominator=list_price,
                    threshold=WORST_CASE_PRICE_THRESHOLD,
                    compared_field="worst_case_exposure.high/list_price",
                ))

    sqft = _metadata_number(metadata, "sqft", "square_feet", "living_area_sqft")
    if sqft and sqft > 0:
        package_high = _bucket_high(estimate, "package_adjusted_rehab")
        if package_high is not None:
            dollars_per_sqft = package_high / sqft
            if dollars_per_sqft > PACKAGE_SQFT_THRESHOLD:
                flags.append(_ratio_flag(
                    code="package_adjusted_high_gt_100_per_sqft",
                    severity="warning",
                    message="Package-adjusted rehab high exceeds $100 per sqft.",
                    value=dollars_per_sqft,
                    numerator=package_high,
                    denominator=sqft,
                    threshold=PACKAGE_SQFT_THRESHOLD,
                    compared_field="package_adjusted_rehab.high/sqft",
                ))

        worst_case_high = _bucket_high(estimate, "worst_case_exposure")
        if worst_case_high is not None:
            dollars_per_sqft = worst_case_high / sqft
            if dollars_per_sqft > WORST_CASE_SQFT_THRESHOLD:
                flags.append(_ratio_flag(
                    code="worst_case_high_gt_150_per_sqft",
                    severity="strong_warning",
                    message="Worst-case exposure exceeds $150 per sqft.",
                    value=dollars_per_sqft,
                    numerator=worst_case_high,
                    denominator=sqft,
                    threshold=WORST_CASE_SQFT_THRESHOLD,
                    compared_field="worst_case_exposure.high/sqft",
                ))

    units = _coerce_estimate_units(estimate_units)
    kitchen_count = sum(1 for unit in units if unit.get("unit_type") == "kitchen")
    if (
        kitchen_count > 1
        and _is_single_family(metadata)
        and not _has_multi_kitchen_evidence(units)
    ):
        flags.append({
            "code": "multiple_billable_kitchens_single_family",
            "severity": "warning",
            "message": (
                "Multiple billable kitchens inferred for a single-family property "
                "without explicit multi-kitchen evidence."
            ),
            "value": kitchen_count,
            "numerator": kitchen_count,
            "denominator": 1,
            "threshold": 1,
            "compared_field": "billable_kitchen_count",
        })

    metadata_baths = _metadata_bath_count(metadata)
    if metadata_baths is not None:
        bathroom_count = sum(1 for unit in units if unit.get("unit_type") == "bathroom")
        if bathroom_count > metadata_baths:
            flags.append({
                "code": "billable_bathrooms_gt_metadata_baths",
                "severity": "warning",
                "message": "Billable bathroom count exceeds metadata bath count.",
                "value": bathroom_count,
                "numerator": bathroom_count,
                "denominator": metadata_baths,
                "threshold": metadata_baths,
                "compared_field": "billable_bathroom_count",
            })

    metadata_beds = _metadata_bedroom_count(metadata)
    if metadata_beds is not None:
        bedroom_count = sum(1 for unit in units if unit.get("unit_type") == "bedroom")
        if bedroom_count > metadata_beds:
            flags.append({
                "code": "billable_bedrooms_gt_metadata_beds",
                "severity": "warning",
                "message": "Billable bedroom count exceeds metadata bed count.",
                "value": bedroom_count,
                "numerator": bedroom_count,
                "denominator": metadata_beds,
                "threshold": metadata_beds,
                "compared_field": "billable_bedroom_count",
            })

    return flags


def _ratio_flag(
    *,
    code: str,
    severity: str,
    message: str,
    value: float,
    numerator: float,
    denominator: float,
    threshold: float,
    compared_field: str,
) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "value": round(value, 2),
        "numerator": _clean_number(numerator),
        "denominator": _clean_number(denominator),
        "threshold": threshold,
        "compared_field": compared_field,
    }


def _bucket_high(estimate: Dict[str, Any], bucket_name: str) -> Optional[float]:
    bucket = estimate.get(bucket_name) or (estimate.get("reconciliation") or {}).get(bucket_name)
    if not isinstance(bucket, dict):
        return None
    value = _coerce_number(bucket.get("high"))
    if value is None:
        return None
    return value


def _coerce_estimate_units(estimate_units: dict | list | None) -> List[Dict[str, Any]]:
    if isinstance(estimate_units, dict):
        estimate_units = estimate_units.get("estimate_units")
    return [
        unit
        for unit in (estimate_units or [])
        if isinstance(unit, dict)
    ]


def _has_multi_kitchen_evidence(units: List[Dict[str, Any]]) -> bool:
    return any(
        unit.get("unit_type") == "kitchen" and unit.get("multi_kitchen_evidence") is True
        for unit in units
    )


def _metadata_bath_count(metadata: Dict[str, Any]) -> Optional[int]:
    full = _metadata_number(metadata, "full_baths", "full_bathrooms")
    half = _metadata_number(metadata, "half_baths", "half_bathrooms")
    if full is not None or half is not None:
        return math.ceil((full or 0) + (half or 0))

    baths = _metadata_number(metadata, "baths", "bath_count", "bathrooms", "bathroom_count")
    if baths is None:
        return None
    return math.ceil(baths)


def _metadata_bedroom_count(metadata: Dict[str, Any]) -> Optional[int]:
    beds = _metadata_number(metadata, "beds", "bed_count", "bedrooms", "bedroom_count")
    if beds is None:
        return None
    return math.ceil(beds)


def _is_single_family(metadata: Dict[str, Any]) -> bool:
    property_type = str(metadata.get("property_type") or "").strip().lower()
    normalized = property_type.replace("-", "_").replace(" ", "_")
    # CSV funnel emits "Single Family Residential" → single_family_residential.
    return normalized.startswith("single_family") or normalized == "sfh"


def _metadata_number(metadata: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _coerce_number(metadata.get(key))
        if value is not None:
            return value
    return None


def _coerce_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "")
        if not cleaned:
            return None
        try:
            parsed = float(cleaned)
        except ValueError:
            return None
        if math.isfinite(parsed):
            return parsed
    return None


def _clean_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else round(value, 2)
