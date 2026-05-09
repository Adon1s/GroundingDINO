"""Read-only audit tool comparing v3 and v4 renovation estimates.

Loads a saved photo_intel.json artifact, extracts both
``photo_intel["renovation_estimate"]`` (v3) and
``photo_intel["renovation_estimate_v4"]`` (v4), and prints a side-by-side
comparison covering totals, inferred packages, reconciliation, and any
warnings.

Usage:
    python tools/compare_reno_estimates.py --run path/to/photo_intel.json
    python tools/compare_reno_estimates.py --run path/to/artifact_dir/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.estimate_sanity import build_estimate_sanity_flags


_BORDER = "=" * 78
_THIN = "-" * 78


def compare(photo_intel: Dict[str, Any]) -> Dict[str, Any]:
    """Build a comparison dict from a photo_intel artifact. Pure read-only."""
    property_key = (
        (photo_intel.get("property") or {}).get("property_key")
        or photo_intel.get("property_key")
        or "<unknown>"
    )
    run_id = (
        (photo_intel.get("run") or {}).get("run_id")
        or photo_intel.get("run_id")
        or "<unknown>"
    )

    v3 = photo_intel.get("renovation_estimate")
    v4 = photo_intel.get("renovation_estimate_v4")

    warnings: List[Dict[str, str]] = []
    diagnostics: List[Dict[str, str]] = []

    if v3 is None:
        warnings.append({
            "code": "v3_key_missing",
            "detail": "photo_intel has no 'renovation_estimate' key",
        })

    if v4 is None:
        warnings.append({
            "code": "missing_v4",
            "detail": "photo_intel has no 'renovation_estimate_v4' key",
        })

    v3_totals = _build_v3_totals(v3)
    v4_totals = _build_v4_totals(v4)
    v4_buckets = _build_v4_buckets(v4)

    if v4 is not None and v4_totals is None:
        warnings.append({
            "code": "missing_final_rehab",
            "detail": "renovation_estimate_v4 has no 'final_rehab' (likely scaffold mode)",
        })

    delta = _build_delta(v3_totals, v4_totals)
    estimate_units = _build_estimate_units(v4, diagnostics)
    billable_units = estimate_units
    packages = _build_packages(v4, diagnostics)
    reconciliation = _build_reconciliation(v3, v4, v3_totals, v4_totals, diagnostics)
    sanity_flags, sanity_flags_source = _build_sanity_flags(photo_intel, v4)

    warnings.extend(diagnostics)
    warnings.extend(_check_reconciliation_invariants(v4))

    return {
        "header": {
            "property_key": property_key,
            "run_id": run_id,
            "source_path": None,
        },
        "v3_totals": v3_totals,
        "v4_totals": v4_totals,
        "v4_buckets": v4_buckets,
        "delta": delta,
        "estimate_units": estimate_units,
        "billable_units": billable_units,
        "packages": packages,
        "reconciliation": reconciliation,
        "sanity_flags": sanity_flags,
        "sanity_flags_source": sanity_flags_source,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def _build_v3_totals(v3: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not v3:
        return None
    totals = v3.get("totals") or {}
    probable = totals.get("probable_total") or {}
    if "low" not in probable or "high" not in probable:
        return None
    low = int(probable["low"])
    high = int(probable["high"])
    risk_high = int((totals.get("risk_exposure_total") or {}).get("high") or 0)
    return {
        "low": low,
        "high": high,
        "midpoint": (low + high) // 2,
        "inspection_exposure_high": risk_high,
    }


def _build_v4_totals(v4: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not v4:
        return None
    final_rehab = v4.get("final_rehab") or {}
    if "low" not in final_rehab or "high" not in final_rehab:
        return None
    low = int(final_rehab["low"])
    high = int(final_rehab["high"])
    midpoint = int(final_rehab.get("midpoint", (low + high) // 2))
    latent = _coerce_bucket(
        v4.get("latent_risk_exposure")
        or (v4.get("reconciliation") or {}).get("latent_risk_exposure")
    )
    if latent is not None:
        applied_high = int(latent["high"])
    else:
        applied_high = sum(
            int(g.get("risk_exposure_high") or 0) for g in (v4.get("groups") or [])
        )
    return {
        "low": low,
        "high": high,
        "midpoint": midpoint,
        "inspection_exposure_high_applied": applied_high,
    }


def _coerce_bucket(bucket: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not bucket:
        return None
    if "low" not in bucket or "high" not in bucket:
        return None
    midpoint = bucket.get("midpoint")
    out: Dict[str, Any] = {
        "low": int(bucket["low"]),
        "high": int(bucket["high"]),
        "midpoint": None if midpoint is None else int(midpoint),
    }
    if "basis" in bucket:
        out["basis"] = bucket.get("basis")
    if "source" in bucket:
        out["source"] = bucket.get("source")
    return out


def _build_v4_buckets(v4: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not v4:
        return {}
    rec = v4.get("reconciliation") or {}
    buckets: Dict[str, Dict[str, Any]] = {}
    for name in (
        "visible_rehab",
        "package_adjusted_rehab",
        "latent_risk_exposure",
        "worst_case_exposure",
        "final_rehab",
    ):
        bucket = _coerce_bucket(v4.get(name) or rec.get(name))
        if bucket is not None:
            buckets[name] = bucket
    return buckets


def _build_delta(
    v3_totals: Optional[Dict[str, Any]],
    v4_totals: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if v3_totals is None or v4_totals is None:
        return None
    return {
        "low": v4_totals["low"] - v3_totals["low"],
        "high": v4_totals["high"] - v3_totals["high"],
        "midpoint": v4_totals["midpoint"] - v3_totals["midpoint"],
    }


def _build_packages(
    v4: Optional[Dict[str, Any]],
    diagnostics: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    if not v4:
        return []
    out: List[Dict[str, Any]] = []
    for pkg in v4.get("packages") or []:
        package_id = pkg.get("package_id", "<unknown>")
        cost_low = int(pkg.get("cost_low") or 0)
        cost_high = int(pkg.get("cost_high") or 0)
        cost_mid = int(pkg.get("cost_midpoint", (cost_low + cost_high) // 2))
        if "estimate_scope" not in pkg:
            diagnostics.append({
                "code": "package_estimate_scope_missing",
                "detail": f"package {package_id} has no 'estimate_scope' field",
            })
        if "absorption_audit" not in pkg:
            diagnostics.append({
                "code": "package_absorption_audit_missing",
                "detail": f"package {package_id} has no 'absorption_audit' field",
            })
        out.append({
            "package_id": package_id,
            "package_type": pkg.get("package_type", "<unknown>"),
            "room_surrogate_id": pkg.get("room_surrogate_id", "<unknown>"),
            "estimate_unit_id": pkg.get("estimate_unit_id", ""),
            "source_room_surrogate_ids": list(pkg.get("source_room_surrogate_ids") or []),
            "estimate_group": pkg.get("estimate_group", "<unknown>"),
            "estimate_scope": pkg.get("estimate_scope", ""),
            "absorption_scope": _copy_jsonish_dict(pkg.get("absorption_scope")),
            "absorption_audit": _build_absorption_audit(pkg),
            "cost_low": cost_low,
            "cost_high": cost_high,
            "cost_midpoint": cost_mid,
            "cap_behavior": pkg.get("cap_behavior", "respect_group_cap"),
            "supporting_issue_ids": list(pkg.get("supporting_issue_ids") or []),
            "absorbed_member_count": len(pkg.get("absorbed_unit_member_refs") or []),
        })
    return out


def _build_absorption_audit(pkg: Dict[str, Any]) -> Dict[str, Any]:
    audit = pkg.get("absorption_audit") or {}
    absorbed = audit.get("absorbed") or {}
    retained = audit.get("retained") or {}
    package_net_delta = audit.get("package_net_delta") or {}
    return {
        "absorbed": {
            "line_items": _copy_string_list(absorbed.get("line_items")),
            "room_allowances": _copy_string_list(absorbed.get("room_allowances")),
            "partial_allocations": _copy_string_list(
                absorbed.get("partial_allocations")
            ),
            "totals": _copy_jsonish_dict(absorbed.get("totals")),
        },
        "retained": {
            "partial_allocations": _copy_string_list(
                retained.get("partial_allocations")
            ),
            "totals": _copy_jsonish_dict(retained.get("totals")),
        },
        "package_net_delta": _copy_jsonish_dict(package_net_delta),
    }


def _copy_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _copy_jsonish_dict(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return json.loads(json.dumps(value))


def _build_estimate_units(
    v4: Optional[Dict[str, Any]],
    diagnostics: List[Dict[str, str]],
) -> Dict[str, Any]:
    raw_units = None if not v4 else v4.get("estimate_units")
    if v4 is not None and "estimate_units" not in v4:
        diagnostics.append({
            "code": "estimate_units_missing",
            "detail": "renovation_estimate_v4 has no 'estimate_units' field",
        })
    units = [dict(u) for u in (raw_units or []) if isinstance(u, dict)]
    for unit in units:
        unit_id = unit.get("estimate_unit_id") or "<unknown>"
        if "source_room_surrogate_ids" not in unit:
            diagnostics.append({
                "code": "estimate_unit_source_room_surrogate_ids_missing",
                "detail": (
                    f"estimate unit {unit_id} has no "
                    "'source_room_surrogate_ids' field"
                ),
            })
        elif not unit.get("source_room_surrogate_ids"):
            diagnostics.append({
                "code": "estimate_unit_source_room_surrogate_ids_empty",
                "detail": (
                    f"estimate unit {unit_id} has an empty "
                    "'source_room_surrogate_ids' field"
                ),
            })
    kitchens = [u for u in units if u.get("unit_type") == "kitchen"]
    bathrooms = [u for u in units if u.get("unit_type") == "bathroom"]
    return {
        "billable_kitchens": len(kitchens),
        "billable_bathrooms": len(bathrooms),
        "units": units,
        "kitchens": kitchens,
        "bathrooms": bathrooms,
    }


def _build_reconciliation(
    v3: Optional[Dict[str, Any]],
    v4: Optional[Dict[str, Any]],
    v3_totals: Optional[Dict[str, int]],
    v4_totals: Optional[Dict[str, int]],
    diagnostics: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    if not v4:
        return None
    rec = v4.get("reconciliation")
    if not rec:
        return None

    retained = rec.get("retained_group_totals") or []
    retained_low = sum(int(g.get("low") or 0) for g in retained)
    retained_high = sum(int(g.get("high") or 0) for g in retained)

    v3_insp_high = v3_totals["inspection_exposure_high"] if v3_totals else None
    v4_insp_high = v4_totals["inspection_exposure_high_applied"] if v4_totals else 0

    group_reconciliation = list(rec.get("package_group_reconciliation") or [])
    for audit in group_reconciliation:
        group = str(audit.get("group", "<unknown>"))
        for field in (
            "original_group_capped",
            "absorbed_total",
            "package_total",
            "package_net_delta",
            "post_cap_package_adjusted",
        ):
            if field not in audit:
                diagnostics.append({
                    "code": "reconciliation_group_field_missing",
                    "detail": f"group {group} has no '{field}' field",
                })

    return {
        "absorbed_total_low": int(rec.get("absorbed_total_low") or 0),
        "absorbed_total_high": int(rec.get("absorbed_total_high") or 0),
        "package_total_low": int(rec.get("package_total_low") or 0),
        "package_total_high": int(rec.get("package_total_high") or 0),
        "net_delta_low": int(rec.get("net_delta_low") or 0),
        "net_delta_high": int(rec.get("net_delta_high") or 0),
        "retained_low": retained_low,
        "retained_high": retained_high,
        "package_count": int(rec.get("package_count") or 0),
        "absorbed_member_count": int(rec.get("absorbed_member_count") or 0),
        "v3_inspection_exposure_high": v3_insp_high,
        "v4_inspection_exposure_high_applied": v4_insp_high,
        "package_group_reconciliation": group_reconciliation,
        "buckets": _build_v4_buckets(v4),
        "v4_internal_warnings": list(rec.get("reconciliation_warnings") or []),
    }


def _check_reconciliation_invariants(
    v4: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    if not v4:
        return []
    warnings: List[Dict[str, str]] = []
    rec = v4.get("reconciliation") or {}
    final_rehab = _coerce_bucket(v4.get("final_rehab") or rec.get("final_rehab"))
    if final_rehab is None:
        return []

    buckets = _build_v4_buckets(v4)
    package_adjusted = buckets.get("package_adjusted_rehab")
    latent = buckets.get("latent_risk_exposure")
    worst_case = buckets.get("worst_case_exposure")

    if package_adjusted is None:
        warnings.append({
            "code": "package_adjusted_rehab_missing",
            "detail": (
                "cannot check final_rehab invariant because "
                "'package_adjusted_rehab' is missing"
            ),
        })
        return warnings

    observed_low = int(final_rehab["low"])
    observed_high = int(final_rehab["high"])
    expected_low = int(package_adjusted["low"])
    expected_high = int(package_adjusted["high"])
    if expected_low != observed_low or expected_high != observed_high:
        warnings.append({
            "code": "reconciliation_invariant_failed",
            "detail": (
                f"final_rehab should equal package_adjusted_rehab; "
                f"expected low={expected_low}, observed low={observed_low}; "
                f"expected high={expected_high}, observed high={observed_high}"
            ),
        })

    missing_worst_case_inputs = []
    if latent is None:
        missing_worst_case_inputs.append("latent_risk_exposure")
    if worst_case is None:
        missing_worst_case_inputs.append("worst_case_exposure")
    if missing_worst_case_inputs:
        warnings.append({
            "code": "worst_case_exposure_inputs_missing",
            "detail": (
                "cannot check worst_case_exposure invariant because "
                f"{', '.join(missing_worst_case_inputs)} is missing"
            ),
        })
        return warnings

    observed_worst_low = int(worst_case["low"])
    observed_worst_high = int(worst_case["high"])
    expected_worst_low = expected_low + int(latent["low"])
    expected_worst_high = expected_high + int(latent["high"])
    if (
        expected_worst_low != observed_worst_low
        or expected_worst_high != observed_worst_high
    ):
        warnings.append({
            "code": "reconciliation_invariant_failed",
            "detail": (
                "worst_case_exposure should equal package_adjusted_rehab "
                "plus latent_risk_exposure; "
                f"expected low={expected_worst_low}, "
                f"observed low={observed_worst_low}; "
                f"expected high={expected_worst_high}, "
                f"observed high={observed_worst_high}"
            ),
        })
    return warnings


def _build_sanity_flags(
    photo_intel: Dict[str, Any],
    v4: Optional[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], str]:
    if not v4:
        return [], "none"
    if "sanity_flags" in v4:
        return [dict(flag) for flag in (v4.get("sanity_flags") or [])], "persisted"

    metadata = _extract_property_metadata(photo_intel)
    flags = build_estimate_sanity_flags(
        v4,
        metadata,
        v4.get("estimate_units") or [],
    )
    return flags, "derived"


def _extract_property_metadata(photo_intel: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    prop = photo_intel.get("property") or {}
    if isinstance(prop, dict):
        metadata.update(prop)
        nested = prop.get("metadata") or {}
        if isinstance(nested, dict):
            metadata.update(nested)
    top_level = photo_intel.get("property_metadata") or {}
    if isinstance(top_level, dict):
        metadata.update(top_level)
    return metadata


def format_report(comparison: Dict[str, Any]) -> str:
    """Format a comparison dict as a human-readable string."""
    lines: List[str] = []
    lines.extend(_format_header(comparison["header"]))
    lines.extend(_format_totals(
        comparison.get("v3_totals"),
        comparison.get("v4_totals"),
        comparison.get("v4_buckets") or {},
        comparison.get("delta"),
    ))
    lines.extend(_format_estimate_units(
        comparison.get("estimate_units") or comparison.get("billable_units") or {}
    ))
    lines.extend(_format_packages(comparison.get("packages") or []))
    lines.extend(_format_reconciliation(comparison.get("reconciliation")))
    lines.extend(_format_sanity_flags(
        comparison.get("sanity_flags") or [],
        comparison.get("sanity_flags_source") or "none",
    ))
    lines.extend(_format_warnings(comparison.get("warnings") or []))
    return "\n".join(lines) + "\n"


def _format_header(header: Dict[str, Any]) -> List[str]:
    lines = [
        "",
        _BORDER,
        "  RENOVATION ESTIMATE COMPARISON (v3 vs v4)",
        _BORDER,
        f"  Property: {header.get('property_key', '<unknown>')}",
        f"  Run ID:   {header.get('run_id', '<unknown>')}",
    ]
    src = header.get("source_path")
    if src:
        lines.append(f"  Source:   {src}")
    return lines


def _money(amount: Optional[int]) -> str:
    if amount is None:
        return "        --"
    return f"${amount:>10,}"


def _signed_money(amount: int) -> str:
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):>9,}"


def _format_totals(
    v3: Optional[Dict[str, int]],
    v4: Optional[Dict[str, int]],
    buckets: Dict[str, Dict[str, Any]],
    delta: Optional[Dict[str, int]],
) -> List[str]:
    lines = ["", _BORDER, "  TOTALS", _BORDER]
    header = f"  {'':<30} {'low':>11} {'high':>11} {'midpoint':>11}"
    lines.append(header)
    lines.append(f"  {'-' * 30} {'-' * 11} {'-' * 11} {'-' * 11}")
    if v3 is not None:
        lines.append(_format_bucket_row("v3 probable", v3))
    else:
        lines.append(_format_missing_bucket_row("v3 probable"))

    for key in (
        "visible_rehab",
        "package_adjusted_rehab",
        "latent_risk_exposure",
        "worst_case_exposure",
        "final_rehab",
    ):
        label = f"v4 {key}"
        bucket = buckets.get(key)
        if bucket is not None:
            lines.append(_format_bucket_row(label, bucket))
        elif key == "final_rehab" and v4 is not None:
            lines.append(_format_bucket_row(label, v4))
        else:
            lines.append(_format_missing_bucket_row(label))

    if delta is not None:
        lines.append(
            f"  {'delta final_rehab - v3':<30} "
            f"{_signed_money(delta['low']):>11} "
            f"{_signed_money(delta['high']):>11} "
            f"{_signed_money(delta['midpoint']):>11}"
        )
    final_bucket = buckets.get("final_rehab") or {}
    lines.append("")
    lines.append(f"  final_rehab basis: {final_bucket.get('basis') or '<unknown>'}")
    lines.append("")
    if v3 is not None:
        lines.append(
            f"  v3 inspection exposure (high):           "
            f"{_money(v3['inspection_exposure_high'])}"
        )
    if v4 is not None:
        lines.append(
            f"  v4 latent risk exposure (high):          "
            f"{_money(v4['inspection_exposure_high_applied'])}"
        )
    return lines


def _format_bucket_row(label: str, bucket: Dict[str, Any]) -> str:
    return (
        f"  {label:<30} "
        f"{_money(int(bucket['low']))} "
        f"{_money(int(bucket['high']))} "
        f"{_money(bucket.get('midpoint'))}"
    )


def _format_missing_bucket_row(label: str) -> str:
    return f"  {label:<30} {'(missing)':>11}"


def _format_v4_buckets(buckets: Dict[str, Dict[str, Any]]) -> List[str]:
    if not buckets:
        return []

    sections = [
        ("visible_rehab", "VISIBLE REHAB"),
        ("package_adjusted_rehab", "PACKAGE-ADJUSTED REHAB"),
        ("latent_risk_exposure", "LATENT RISK EXPOSURE"),
        ("worst_case_exposure", "WORST-CASE EXPOSURE"),
        ("final_rehab", "FINAL REHAB"),
    ]
    lines: List[str] = []
    for key, title in sections:
        bucket = buckets.get(key)
        if bucket is None:
            continue
        lines.extend(["", _BORDER, f"  {title}", _BORDER])
        lines.append(f"  {'low':>11} {'high':>11} {'midpoint':>11}")
        lines.append(f"  {'-' * 11} {'-' * 11} {'-' * 11}")
        lines.append(
            f"  {_money(bucket['low'])} "
            f"{_money(bucket['high'])} "
            f"{_money(bucket.get('midpoint'))}"
        )
        basis = bucket.get("basis")
        if key == "final_rehab":
            lines.append(f"  FINAL REHAB BASIS: {basis or '<unknown>'}")
        elif basis:
            lines.append(f"  basis: {basis}")
    return lines


def _format_estimate_units(summary: Dict[str, Any]) -> List[str]:
    lines = ["", _BORDER, "  ESTIMATE UNITS", _BORDER]
    lines.append(f"  billable kitchens: {summary.get('billable_kitchens', 0)}")
    lines.append(f"  billable bathrooms: {summary.get('billable_bathrooms', 0)}")
    units = summary.get("units") or []
    if not units:
        lines.append("  No estimate units present.")
        return lines

    for unit in units:
        unit_id = unit.get("estimate_unit_id") or "<unknown>"
        if "source_room_surrogate_ids" not in unit:
            source_rooms = "<missing source_room_surrogate_ids>"
        else:
            source_ids = [
                str(source_id)
                for source_id in (unit.get("source_room_surrogate_ids") or [])
            ]
            source_rooms = ", ".join(source_ids) if source_ids else "(none)"
        unit_type = unit.get("unit_type") or "<unknown>"
        confidence = unit.get("confidence")
        merge_reason = unit.get("merge_reason")
        suffix = f" [{unit_type}]"
        details = []
        if confidence:
            details.append(f"confidence={confidence}")
        if merge_reason:
            details.append(f"merge_reason={merge_reason}")
        if details:
            suffix += f" ({', '.join(details)})"
        lines.append(f"  {unit_id} <- {source_rooms}{suffix}")
    return lines


def _format_billable_units(summary: Dict[str, Any]) -> List[str]:
    return _format_estimate_units(summary)


def _format_packages_legacy(packages: List[Dict[str, Any]]) -> List[str]:
    lines = ["", _BORDER, f"  PACKAGES ({len(packages)})", _BORDER]
    if not packages:
        lines.append("  No packages inferred.")
        return lines
    for pkg in packages:
        lines.append("")
        lines.append(f"  {pkg['package_id']}")
        lines.append(f"    type:   {pkg['package_type']}")
        lines.append(f"    room:   {pkg['room_surrogate_id']}")
        if pkg.get("estimate_unit_id"):
            lines.append(f"    unit:   {pkg['estimate_unit_id']}")
        if pkg.get("source_room_surrogate_ids"):
            lines.append(
                "    source room surrogates: "
                f"{', '.join(pkg['source_room_surrogate_ids'])}"
            )
        lines.append(f"    group:  {pkg['estimate_group']}")
        lines.append(f"    cap:    {pkg['cap_behavior']}")
        lines.append(
            f"    cost:   ${pkg['cost_low']:,} – ${pkg['cost_high']:,} "
            f"(mid ${pkg['cost_midpoint']:,})"
        )
        lines.append(
            f"    supporting issues: {_format_issue_ids(pkg['supporting_issue_ids'])}"
        )
        lines.append(f"    absorbed members:  {pkg['absorbed_member_count']}")
    return lines


def _format_issue_ids(issue_ids: List[str], cap: int = 6) -> str:
    if not issue_ids:
        return "0"
    n = len(issue_ids)
    shown = issue_ids[:cap]
    suffix = f" ... +{n - cap} more" if n > cap else ""
    return f"{n} ({', '.join(shown)}{suffix})"


def _format_packages(packages: List[Dict[str, Any]]) -> List[str]:
    lines = ["", _BORDER, f"  PACKAGES ({len(packages)})", _BORDER]
    if not packages:
        lines.append("  No packages inferred.")
        return lines
    for pkg in packages:
        lines.append("")
        lines.append(f"  {pkg['package_id']}")
        lines.append(f"    package_type:      {pkg['package_type']}")
        lines.append(f"    room_surrogate_id: {pkg['room_surrogate_id']}")
        lines.append(f"    estimate_unit_id:  {pkg.get('estimate_unit_id') or '<missing>'}")
        if pkg.get("source_room_surrogate_ids"):
            lines.append(
                "    source room surrogates: "
                f"{', '.join(pkg['source_room_surrogate_ids'])}"
            )
        lines.append(f"    estimate_group:    {pkg['estimate_group']}")
        lines.append(f"    estimate_scope:    {pkg.get('estimate_scope') or '<missing>'}")
        lines.append(
            f"    absorption_scope:  "
            f"{_format_absorption_scope(pkg.get('absorption_scope') or {})}"
        )
        lines.append(f"    cap_behavior:      {pkg['cap_behavior']}")
        lines.append(
            f"    cost:              ${pkg['cost_low']:,} - ${pkg['cost_high']:,} "
            f"(mid ${pkg['cost_midpoint']:,})"
        )
        lines.append(
            f"    supporting issues: {_format_issue_ids(pkg['supporting_issue_ids'])}"
        )
        lines.append(f"    absorbed members:  {pkg['absorbed_member_count']}")
        lines.extend(_format_absorption_audit(pkg.get("absorption_audit") or {}))
    return lines


def _format_absorption_scope(scope: Dict[str, Any]) -> str:
    if not scope:
        return "(none)"
    parts = []
    for key in ("family", "groups", "trade_buckets", "components"):
        value = scope.get(key)
        if isinstance(value, list):
            formatted = ", ".join(str(item) for item in value) or "(none)"
        else:
            formatted = str(value) if value else "(none)"
        parts.append(f"{key}={formatted}")
    return "; ".join(parts)


def _format_absorption_audit(audit: Dict[str, Any]) -> List[str]:
    absorbed = audit.get("absorbed") or {}
    retained = audit.get("retained") or {}
    lines = [
        f"    absorbed line items: {_format_string_list(absorbed.get('line_items') or [])}",
        (
            "    absorbed room allowances: "
            f"{_format_string_list(absorbed.get('room_allowances') or [])}"
        ),
        (
            "    absorbed partial allocations: "
            f"{_format_string_list(absorbed.get('partial_allocations') or [])}"
        ),
        (
            "    retained partial allocations: "
            f"{_format_string_list(retained.get('partial_allocations') or [])}"
        ),
    ]
    net_delta = audit.get("package_net_delta") or {}
    if net_delta:
        lines.append(f"    package net delta: {_format_amount_pair(net_delta)}")
    return lines


def _format_string_list(items: List[str]) -> str:
    if not items:
        return "(none)"
    return ", ".join(str(item) for item in items)


def _format_amount_pair(amount: Dict[str, Any]) -> str:
    if not amount:
        return "(missing)"
    return (
        f"low={_money(int(amount.get('low') or 0))}, "
        f"high={_money(int(amount.get('high') or 0))}"
    )


def _format_signed_amount_pair(amount: Dict[str, Any]) -> str:
    if not amount:
        return "(missing)"
    return (
        f"low={_signed_money(int(amount.get('low') or 0))}, "
        f"high={_signed_money(int(amount.get('high') or 0))}"
    )


def _format_reconciliation(rec: Optional[Dict[str, Any]]) -> List[str]:
    lines = ["", _BORDER, "  RECONCILIATION", _BORDER]
    if rec is None:
        lines.append("  Not present (v4 missing or scaffold mode).")
        return lines
    lines.append(f"  {'':<22} {'low':>11} {'high':>11}")
    lines.append(f"  {'-' * 22} {'-' * 11} {'-' * 11}")
    lines.append(
        f"  {'absorbed total':<22} "
        f"{_money(rec['absorbed_total_low'])} {_money(rec['absorbed_total_high'])}"
    )
    lines.append(
        f"  {'package total':<22} "
        f"{_money(rec['package_total_low'])} {_money(rec['package_total_high'])}"
    )
    lines.append(
        f"  {'net delta':<22} "
        f"{_signed_money(rec['net_delta_low']):>11} "
        f"{_signed_money(rec['net_delta_high']):>11}"
    )
    lines.append(
        f"  {'retained standalone':<22} "
        f"{_money(rec['retained_low'])} {_money(rec['retained_high'])}"
    )
    lines.append("")
    lines.append(f"  package count:        {rec['package_count']}")
    lines.append(f"  absorbed members:     {rec['absorbed_member_count']}")
    lines.append("")
    lines.append(
        f"  v3 inspection exposure (high):              "
        f"{_money(rec['v3_inspection_exposure_high'])}"
    )
    lines.append(
        f"  v4 latent risk exposure (high):             "
        f"{_money(rec['v4_inspection_exposure_high_applied'])}"
    )
    lines.append("")
    lines.extend(_format_group_cap_reconciliation(
        rec.get("package_group_reconciliation") or []
    ))
    if rec.get("package_group_reconciliation"):
        lines.append("")
    internal = rec.get("v4_internal_warnings") or []
    if not internal:
        lines.append("  v4 internal warnings: None.")
    else:
        lines.append(f"  v4 internal warnings ({len(internal)}):")
        for w in internal:
            code = w.get("code", "<unknown>")
            details = ", ".join(
                f"{k}={v}" for k, v in w.items() if k != "code"
            )
            lines.append(f"    - {code}: {details}" if details else f"    - {code}")
    return lines


def _format_group_cap_reconciliation(group_audits: List[Dict[str, Any]]) -> List[str]:
    if not group_audits:
        return []
    lines: List[str] = ["  group reconciliation:"]
    for audit in group_audits:
        original = audit.get("original_group_capped") or {}
        absorbed = audit.get("absorbed_total") or {}
        package_total = audit.get("package_total") or {}
        net = audit.get("package_net_delta") or {}
        post = audit.get("post_cap_package_adjusted") or {}
        cap = "yes" if audit.get("cap_applied_after_packages") else "no"
        final_group_total = post or audit.get("pre_cap_package_adjusted") or {}
        lines.append(f"    group: {audit.get('group', '<unknown>')}")
        lines.append(
            f"      original_group_capped:     {_format_amount_pair(original)}"
        )
        lines.append(f"      absorbed_total:            {_format_amount_pair(absorbed)}")
        lines.append(f"      package_total:             {_format_amount_pair(package_total)}")
        lines.append(
            f"      package_net_delta:         {_format_signed_amount_pair(net)}"
        )
        lines.append(
            f"      post_cap_package_adjusted: {_format_amount_pair(post)}"
        )
        lines.append(f"      cap_applied_after_packages: {cap}")
        lines.append(
            f"      final group total:         {_format_amount_pair(final_group_total)}"
        )
    return lines


def _format_sanity_flags(flags: List[Dict[str, Any]], source: str) -> List[str]:
    title = "  SANITY FLAGS"
    if source == "derived":
        title += " (derived; not persisted)"
    lines = ["", _BORDER, title, _BORDER]
    if not flags:
        lines.append("  None.")
        return lines

    for flag in flags:
        severity = flag.get("severity", "warning")
        message = flag.get("message", "")
        lines.append(f"  - {flag.get('code', '<unknown>')} [{severity}]: {message}")
        basis = _format_flag_basis(flag)
        if basis:
            lines.append(f"    {basis}")

    if any(flag.get("code") == "worst_case_high_gt_80pct_price" for flag in flags):
        lines.append(
            "  This is extreme relative to list price; interpret as "
            "worst-case/inspection-dependent."
        )
    return lines


def _format_flag_basis(flag: Dict[str, Any]) -> str:
    compared_field = flag.get("compared_field")
    numerator = flag.get("numerator")
    denominator = flag.get("denominator")
    threshold = flag.get("threshold")
    value = flag.get("value")
    parts = []
    if compared_field:
        parts.append(f"field={compared_field}")
    if numerator is not None:
        parts.append(f"numerator={numerator}")
    if denominator is not None:
        parts.append(f"denominator={denominator}")
    if value is not None:
        parts.append(f"value={value}")
    if threshold is not None:
        parts.append(f"threshold={threshold}")
    return ", ".join(parts)


def _format_warnings(warnings: List[Dict[str, str]]) -> List[str]:
    lines = ["", _BORDER, "  WARNINGS", _BORDER]
    if not warnings:
        lines.append("  None.")
        return lines
    for w in warnings:
        code = w.get("code", "<unknown>")
        detail = w.get("detail")
        if detail is None:
            detail = ", ".join(f"{k}={v}" for k, v in w.items() if k != "code")
        lines.append(f"  - {code}: {detail}" if detail else f"  - {code}")
    return lines


def _resolve_artifact_path(run_path: Path) -> Path:
    if run_path.is_file():
        return run_path
    if run_path.is_dir():
        candidate = run_path / "photo_intel.json"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"directory does not contain photo_intel.json: {run_path}"
        )
    raise FileNotFoundError(f"path does not exist: {run_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare v3 and v4 renovation estimates from a photo_intel artifact.",
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to a photo_intel.json file or a directory containing one.",
    )
    args = parser.parse_args(argv)

    run_path = Path(args.run)
    try:
        artifact_path = _resolve_artifact_path(run_path)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        with artifact_path.open("r", encoding="utf-8") as f:
            photo_intel = json.load(f)
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON in {artifact_path}: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: cannot read {artifact_path}: {e}", file=sys.stderr)
        return 1

    comparison = compare(photo_intel)
    comparison["header"]["source_path"] = str(artifact_path)
    print(format_report(comparison))
    return 0


if __name__ == "__main__":
    sys.exit(main())
