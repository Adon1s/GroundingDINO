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
    billable_units = _build_billable_units(v4)
    packages = _build_packages(v4)
    reconciliation = _build_reconciliation(v3, v4, v3_totals, v4_totals)
    sanity_flags, sanity_flags_source = _build_sanity_flags(photo_intel, v4)

    invariant_warning = _check_reconciliation_invariant(v4)
    if invariant_warning is not None:
        warnings.append(invariant_warning)

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
        "billable_units": billable_units,
        "packages": packages,
        "reconciliation": reconciliation,
        "sanity_flags": sanity_flags,
        "sanity_flags_source": sanity_flags_source,
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


def _build_packages(v4: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not v4:
        return []
    out: List[Dict[str, Any]] = []
    for pkg in v4.get("packages") or []:
        cost_low = int(pkg.get("cost_low") or 0)
        cost_high = int(pkg.get("cost_high") or 0)
        cost_mid = int(pkg.get("cost_midpoint", (cost_low + cost_high) // 2))
        out.append({
            "package_id": pkg.get("package_id", "<unknown>"),
            "package_type": pkg.get("package_type", "<unknown>"),
            "room_surrogate_id": pkg.get("room_surrogate_id", "<unknown>"),
            "estimate_unit_id": pkg.get("estimate_unit_id", ""),
            "source_room_surrogate_ids": list(pkg.get("source_room_surrogate_ids") or []),
            "estimate_group": pkg.get("estimate_group", "<unknown>"),
            "cost_low": cost_low,
            "cost_high": cost_high,
            "cost_midpoint": cost_mid,
            "cap_behavior": pkg.get("cap_behavior", "respect_group_cap"),
            "supporting_issue_ids": list(pkg.get("supporting_issue_ids") or []),
            "absorbed_member_count": len(pkg.get("absorbed_unit_member_refs") or []),
        })
    return out


def _build_billable_units(v4: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    units = list((v4 or {}).get("estimate_units") or [])
    kitchens = [u for u in units if u.get("unit_type") == "kitchen"]
    bathrooms = [u for u in units if u.get("unit_type") == "bathroom"]
    return {
        "billable_kitchens": len(kitchens),
        "billable_bathrooms": len(bathrooms),
        "kitchens": kitchens,
        "bathrooms": bathrooms,
    }


def _build_reconciliation(
    v3: Optional[Dict[str, Any]],
    v4: Optional[Dict[str, Any]],
    v3_totals: Optional[Dict[str, int]],
    v4_totals: Optional[Dict[str, int]],
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
        "package_group_reconciliation": list(
            rec.get("package_group_reconciliation") or []
        ),
        "buckets": _build_v4_buckets(v4),
        "v4_internal_warnings": list(rec.get("reconciliation_warnings") or []),
    }


def _check_reconciliation_invariant(
    v4: Optional[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    if not v4:
        return None
    final_rehab = v4.get("final_rehab") or {}
    rec = v4.get("reconciliation") or {}
    if not final_rehab or not rec:
        return None
    if "low" not in final_rehab or "high" not in final_rehab:
        return None

    buckets = _build_v4_buckets(v4)
    package_adjusted = buckets.get("package_adjusted_rehab")
    latent = buckets.get("latent_risk_exposure")
    worst_case = buckets.get("worst_case_exposure")
    if package_adjusted is not None:
        observed_low = int(final_rehab["low"])
        observed_high = int(final_rehab["high"])
        expected_low = int(package_adjusted["low"])
        expected_high = int(package_adjusted["high"])
        if expected_low != observed_low or expected_high != observed_high:
            return {
                "code": "reconciliation_invariant_failed",
                "detail": (
                    f"final_rehab should equal package_adjusted_rehab; "
                    f"expected low={expected_low}, observed low={observed_low}; "
                    f"expected high={expected_high}, observed high={observed_high}"
                ),
            }

        if latent is not None and worst_case is not None:
            observed_worst_low = int(worst_case["low"])
            observed_worst_high = int(worst_case["high"])
            expected_worst_low = expected_low
            expected_worst_high = expected_high + int(latent["high"])
            if (
                expected_worst_low != observed_worst_low
                or expected_worst_high != observed_worst_high
            ):
                return {
                    "code": "reconciliation_invariant_failed",
                    "detail": (
                        "worst_case_exposure should equal "
                        "package_adjusted_rehab plus latent_risk_exposure; "
                        f"expected low={expected_worst_low}, "
                        f"observed low={observed_worst_low}; "
                        f"expected high={expected_worst_high}, "
                        f"observed high={observed_worst_high}"
                    ),
                }
        return None

    retained = rec.get("retained_group_totals") or []
    retained_low = sum(int(g.get("low") or 0) for g in retained)
    retained_high = sum(int(g.get("high") or 0) for g in retained)
    pkg_low = int(rec.get("package_total_low") or 0)
    pkg_high = int(rec.get("package_total_high") or 0)
    risk_high = sum(
        int(g.get("risk_exposure_high") or 0) for g in (v4.get("groups") or [])
    )

    expected_low = retained_low + pkg_low
    expected_high = retained_high + pkg_high + risk_high
    observed_low = int(final_rehab["low"])
    observed_high = int(final_rehab["high"])

    if expected_low == observed_low and expected_high == observed_high:
        return None

    return {
        "code": "reconciliation_invariant_failed",
        "detail": (
            f"expected low={expected_low}, observed low={observed_low}; "
            f"expected high={expected_high}, observed high={observed_high}"
        ),
    }


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
        comparison.get("delta"),
    ))
    lines.extend(_format_v4_buckets(comparison.get("v4_buckets") or {}))
    lines.extend(_format_billable_units(comparison.get("billable_units") or {}))
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
        return "      —"
    return f"${amount:>10,}"


def _signed_money(amount: int) -> str:
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):>9,}"


def _format_totals(
    v3: Optional[Dict[str, int]],
    v4: Optional[Dict[str, int]],
    delta: Optional[Dict[str, int]],
) -> List[str]:
    lines = ["", _BORDER, "  TOTALS", _BORDER]
    header = f"  {'':<18} {'low':>11} {'high':>11} {'midpoint':>11}"
    lines.append(header)
    lines.append(f"  {'-' * 18} {'-' * 11} {'-' * 11} {'-' * 11}")
    if v3 is not None:
        lines.append(
            f"  {'v3 probable':<18} "
            f"{_money(v3['low'])} {_money(v3['high'])} {_money(v3['midpoint'])}"
        )
    else:
        lines.append(f"  {'v3 probable':<18} {'(missing)':>11}")
    if v4 is not None:
        lines.append(
            f"  {'v4 final_rehab':<18} "
            f"{_money(v4['low'])} {_money(v4['high'])} {_money(v4['midpoint'])}"
        )
    else:
        lines.append(f"  {'v4 final_rehab':<18} {'(missing)':>11}")
    if delta is not None:
        lines.append(
            f"  {'delta (v4 - v3)':<18} "
            f"{_signed_money(delta['low']):>11} "
            f"{_signed_money(delta['high']):>11} "
            f"{_signed_money(delta['midpoint']):>11}"
        )
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


def _format_billable_units(summary: Dict[str, Any]) -> List[str]:
    lines = ["", _BORDER, "  BILLABLE UNITS", _BORDER]
    lines.append(f"  billable kitchens: {summary.get('billable_kitchens', 0)}")
    for unit in summary.get("kitchens") or []:
        source_rooms = ", ".join(unit.get("source_room_surrogate_ids") or [])
        lines.append(f"    {unit.get('estimate_unit_id', '<unknown>')}")
        lines.append(f"      source room surrogates: {source_rooms or 'none'}")
    lines.append(f"  billable bathrooms: {summary.get('billable_bathrooms', 0)}")
    for unit in summary.get("bathrooms") or []:
        source_rooms = ", ".join(unit.get("source_room_surrogate_ids") or [])
        lines.append(f"    {unit.get('estimate_unit_id', '<unknown>')}")
        lines.append(f"      source room surrogates: {source_rooms or 'none'}")
    return lines


def _format_packages(packages: List[Dict[str, Any]]) -> List[str]:
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
    lines: List[str] = ["  group cap reconciliation:"]
    lines.append(
        f"    {'group':<16} {'orig high':>11} {'net high':>11} "
        f"{'pre high':>11} {'post high':>11} {'cap':>7} {'override':>9}"
    )
    for audit in group_audits:
        original = audit.get("original_group_capped") or {}
        net = audit.get("package_net_delta") or {}
        pre = audit.get("pre_cap_package_adjusted") or {}
        post = audit.get("post_cap_package_adjusted") or {}
        cap = "yes" if audit.get("cap_applied_after_packages") else "no"
        override = "yes" if audit.get("cap_override") else "no"
        lines.append(
            f"    {str(audit.get('group', '<unknown>')):<16} "
            f"{_money(int(original.get('high') or 0))} "
            f"{_signed_money(int(net.get('high') or 0)):>11} "
            f"{_money(int(pre.get('high') or 0))} "
            f"{_money(int(post.get('high') or 0))} "
            f"{cap:>7} {override:>9}"
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
        lines.append(f"  - {w['code']}: {w['detail']}")
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
