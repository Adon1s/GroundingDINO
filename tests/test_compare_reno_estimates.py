"""Tests for tools/compare_reno_estimates.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.compare_reno_estimates import compare, format_report, main


# ── Fixture builders ────────────────────────────────────────────────────────

def _make_v3(
    *,
    probable_low: int = 10000,
    probable_high: int = 25000,
    risk_exposure_high: int = 0,
) -> Dict[str, Any]:
    return {
        "version": "renovation_estimate_v3",
        "totals": {
            "probable_total": {"low": probable_low, "high": probable_high},
            "risk_exposure_total": {"low": 0, "high": risk_exposure_high},
        },
        "groups": [],
    }


def _make_package(
    *,
    package_id: str = "kitchen_full__rs1",
    package_type: str = "kitchen_full_rehab",
    room_surrogate_id: str = "rs_kitchen_001",
    estimate_unit_id: str = "kitchen_primary",
    source_room_surrogate_ids: Optional[List[str]] = None,
    estimate_group: str = "kitchen",
    estimate_scope: str = "marketability_rehab",
    absorption_scope: Optional[Dict[str, Any]] = None,
    absorption_audit: Optional[Dict[str, Any]] = None,
    cost_low: int = 30000,
    cost_high: int = 70000,
    cost_midpoint: Optional[int] = None,
    supporting_issue_ids: Optional[List[str]] = None,
    absorbed_unit_member_refs: Optional[List[Dict[str, Any]]] = None,
    absorbed_total_low: int = 0,
    absorbed_total_high: int = 0,
    cap_behavior: str = "respect_group_cap",
) -> Dict[str, Any]:
    if cost_midpoint is None:
        cost_midpoint = (cost_low + cost_high) // 2
    if source_room_surrogate_ids is None:
        source_room_surrogate_ids = ["kitchen_1", "kitchen_2", "kitchen_3"]
    if absorption_scope is None:
        absorption_scope = {
            "family": "kitchen",
            "groups": ["kitchen", "flooring"],
            "trade_buckets": ["kitchen_cabinets_counters", "flooring"],
            "components": ["cabinets", "counter", "flooring"],
        }
    if absorption_audit is None:
        absorption_audit = {
            "absorbed": {
                "line_items": [],
                "room_allowances": [],
                "partial_allocations": [],
                "totals": {"low": 0, "high": 0, "midpoint": 0},
            },
            "retained": {
                "partial_allocations": [],
                "totals": {"low": 0, "high": 0, "midpoint": 0},
            },
            "package_net_delta": {
                "low": max(0, cost_low - absorbed_total_low),
                "high": max(0, cost_high - absorbed_total_high),
                "midpoint": (
                    max(0, cost_low - absorbed_total_low)
                    + max(0, cost_high - absorbed_total_high)
                ) // 2,
            },
        }
    return {
        "package_id": package_id,
        "package_type": package_type,
        "room_surrogate_id": room_surrogate_id,
        "estimate_unit_id": estimate_unit_id,
        "source_room_surrogate_ids": list(source_room_surrogate_ids),
        "estimate_group": estimate_group,
        "estimate_scope": estimate_scope,
        "absorption_scope": dict(absorption_scope),
        "absorption_audit": dict(absorption_audit),
        "cost_low": cost_low,
        "cost_high": cost_high,
        "cost_midpoint": cost_midpoint,
        "cap_behavior": cap_behavior,
        "supporting_issue_ids": list(supporting_issue_ids or []),
        "absorbed_unit_member_refs": list(absorbed_unit_member_refs or []),
        "absorbed_total_low": absorbed_total_low,
        "absorbed_total_high": absorbed_total_high,
    }


def _make_reconciliation(
    *,
    package_total_low: int,
    package_total_high: int,
    retained_low: int,
    retained_high: int,
    absorbed_total_low: int = 0,
    absorbed_total_high: int = 0,
    package_count: int = 1,
    absorbed_member_count: int = 0,
    risk_exposure_high: int = 0,
    warnings: Optional[List[Dict[str, Any]]] = None,
    package_group_reconciliation: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    visible_low = retained_low + absorbed_total_low
    visible_high = retained_high + absorbed_total_high
    package_adjusted_low = retained_low + package_total_low
    package_adjusted_high = retained_high + package_total_high
    visible_rehab = {
        "low": visible_low,
        "high": visible_high,
        "midpoint": (visible_low + visible_high) // 2,
        "basis": "verified_visible_line_items_before_package_replacement",
    }
    package_adjusted_rehab = {
        "low": package_adjusted_low,
        "high": package_adjusted_high,
        "midpoint": (package_adjusted_low + package_adjusted_high) // 2,
        "basis": "visible_work_after_package_reconciliation",
    }
    latent_risk_exposure = {
        "low": 0,
        "high": risk_exposure_high,
        "midpoint": None,
        "basis": "inspect_posture_items_and_hidden_condition_exposure",
    }
    worst_case_exposure = {
        "low": package_adjusted_low,
        "high": package_adjusted_high + risk_exposure_high,
        "midpoint": None,
        "basis": "package_adjusted_rehab_plus_latent_risk_exposure",
    }
    final_rehab = {
        **package_adjusted_rehab,
        "basis": "package_adjusted_rehab",
        "source": "renovation_estimate_v4",
    }
    return {
        "absorbed_total_low": absorbed_total_low,
        "absorbed_total_high": absorbed_total_high,
        "package_total_low": package_total_low,
        "package_total_high": package_total_high,
        "net_delta_low": package_total_low - absorbed_total_low,
        "net_delta_high": package_total_high - absorbed_total_high,
        "absorbed_member_count": absorbed_member_count,
        "package_count": package_count,
        "retained_group_totals": [
            {"group": "other", "low": retained_low, "high": retained_high},
        ],
        "package_group_reconciliation": list(package_group_reconciliation or []),
        "reconciliation_warnings": list(warnings or []),
        "warnings": [
            w.get("code") for w in list(warnings or []) if w.get("code")
        ],
        "visible_rehab": visible_rehab,
        "package_adjusted_rehab": package_adjusted_rehab,
        "latent_risk_exposure": latent_risk_exposure,
        "worst_case_exposure": worst_case_exposure,
        "final_rehab": final_rehab,
    }


def _make_v4(
    *,
    final_rehab: Optional[Dict[str, Any]] = None,
    packages: Optional[List[Dict[str, Any]]] = None,
    reconciliation: Optional[Dict[str, Any]] = None,
    groups_risk_high: int = 0,
    packages_enabled: bool = True,
    sanity_flags: Optional[List[Dict[str, Any]]] = None,
    estimate_units: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    groups = (
        [{"group": "other", "risk_exposure_high": groups_risk_high}]
        if groups_risk_high
        else []
    )
    v4: Dict[str, Any] = {
        "version": "renovation_estimate_v4",
        "groups": groups,
        "packages": list(packages or []),
        "estimate_units": list(estimate_units or []),
        "provenance": {
            "packages_enabled": packages_enabled,
            "reconciliation_enabled": packages_enabled,
        },
    }
    if reconciliation is not None:
        v4["reconciliation"] = reconciliation
        for bucket_name in (
            "visible_rehab",
            "package_adjusted_rehab",
            "latent_risk_exposure",
            "worst_case_exposure",
            "final_rehab",
        ):
            if bucket_name in reconciliation:
                v4[bucket_name] = reconciliation[bucket_name]
    if final_rehab is not None:
        v4["final_rehab"] = final_rehab
    elif reconciliation is not None and reconciliation.get("final_rehab"):
        v4["final_rehab"] = reconciliation["final_rehab"]
    if sanity_flags is not None:
        v4["sanity_flags"] = list(sanity_flags)
    return v4


def _make_photo_intel(
    *,
    v3: Optional[Dict[str, Any]] = None,
    v4: Optional[Dict[str, Any]] = None,
    property_key: str = "redfin_test",
    run_id: str = "test_run",
    property_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "property_key": property_key,
        "run_id": run_id,
    }
    if v3 is not None:
        out["renovation_estimate"] = v3
    if v4 is not None:
        out["renovation_estimate_v4"] = v4
    if property_metadata is not None:
        out["property_metadata"] = dict(property_metadata)
    return out


def _make_estimate_units() -> List[Dict[str, Any]]:
    return [
        {
            "estimate_unit_id": "kitchen_primary",
            "unit_type": "kitchen",
            "source_room_surrogate_ids": ["kitchen_1", "kitchen_2", "kitchen_3"],
            "confidence": "default_assumption",
            "merge_reason": "single_family_default_one_kitchen",
        },
        {
            "estimate_unit_id": "bathroom_primary",
            "unit_type": "bathroom",
            "source_room_surrogate_ids": ["bathroom_1", "bathroom_2"],
            "confidence": "conservative_assumption",
            "merge_reason": "conservative_repeated_bathroom_merge",
        },
    ]


# ── Tests ───────────────────────────────────────────────────────────────────

def test_v3_only_warns_missing_v4():
    pi = _make_photo_intel(v3=_make_v3())
    result = compare(pi)

    assert result["v3_totals"] is not None
    assert result["v3_totals"]["low"] == 10000
    assert result["v3_totals"]["midpoint"] == 17500
    assert result["v4_totals"] is None
    assert result["delta"] is None
    assert result["packages"] == []
    assert result["reconciliation"] is None

    codes = {w["code"] for w in result["warnings"]}
    assert "missing_v4" in codes
    assert "v3_key_missing" not in codes


def test_v3_plus_v4_scaffold_no_packages():
    v3 = _make_v3()
    v4 = _make_v4(packages=[], reconciliation=None, packages_enabled=False)
    pi = _make_photo_intel(v3=v3, v4=v4)
    result = compare(pi)
    report = format_report(result)

    assert result["v3_totals"] is not None
    assert result["v4_totals"] is None
    assert result["delta"] is None
    assert result["packages"] == []
    assert result["reconciliation"] is None

    codes = {w["code"] for w in result["warnings"]}
    assert "missing_final_rehab" in codes
    assert "missing_v4" not in codes
    assert "reconciliation_invariant_failed" not in codes
    assert "ESTIMATE UNITS" in report
    assert "No estimate units present." in report


def test_partial_v4_artifact_reports_missing_field_diagnostics():
    final_rehab = {
        "low": 1000,
        "high": 2000,
        "midpoint": 1500,
        "basis": "legacy_final_only",
    }
    v4 = _make_v4(
        final_rehab=final_rehab,
        packages=[{
            "package_id": "legacy_pkg",
            "package_type": "legacy_package",
            "room_surrogate_id": "kitchen_1",
            "estimate_group": "kitchen",
            "cost_low": 1000,
            "cost_high": 2000,
        }],
        reconciliation={"final_rehab": final_rehab},
    )
    del v4["estimate_units"]
    pi = _make_photo_intel(v3=_make_v3(), v4=v4)

    result = compare(pi)
    report = format_report(result)
    codes = {w["code"] for w in result["warnings"]}

    assert "estimate_units_missing" in codes
    assert "package_estimate_scope_missing" in codes
    assert "package_absorption_audit_missing" in codes
    assert "package_adjusted_rehab_missing" in codes
    assert "estimate_unit_id:  <missing>" in report
    assert "absorbed line items: (none)" in report


def test_v3_plus_v4_with_packages():
    v3 = _make_v3(probable_low=10000, probable_high=25000, risk_exposure_high=2000)
    rec = _make_reconciliation(
        package_total_low=30000,
        package_total_high=70000,
        retained_low=5000,
        retained_high=10000,
        package_count=1,
        absorbed_member_count=2,
    )
    pkg = _make_package(supporting_issue_ids=["i1", "i2"])
    v4 = _make_v4(packages=[pkg], reconciliation=rec)
    pi = _make_photo_intel(v3=v3, v4=v4)
    result = compare(pi)

    assert result["v3_totals"]["low"] == 10000
    assert result["v3_totals"]["high"] == 25000
    assert result["v3_totals"]["inspection_exposure_high"] == 2000

    assert result["v4_totals"]["low"] == 35000
    assert result["v4_totals"]["high"] == 80000
    assert result["v4_totals"]["midpoint"] == 57500
    assert result["v4_buckets"]["package_adjusted_rehab"]["high"] == 80000
    assert result["v4_buckets"]["latent_risk_exposure"]["high"] == 0
    assert result["v4_buckets"]["final_rehab"]["basis"] == "package_adjusted_rehab"

    assert result["delta"] == {"low": 25000, "high": 55000, "midpoint": 40000}

    assert len(result["packages"]) == 1
    assert result["packages"][0]["package_id"] == "kitchen_full__rs1"
    assert result["packages"][0]["supporting_issue_ids"] == ["i1", "i2"]

    rec_out = result["reconciliation"]
    assert rec_out is not None
    assert rec_out["package_total_low"] == 30000
    assert rec_out["retained_low"] == 5000
    assert rec_out["v3_inspection_exposure_high"] == 2000
    assert rec_out["v4_inspection_exposure_high_applied"] == 0

    codes = {w["code"] for w in result["warnings"]}
    assert "reconciliation_invariant_failed" not in codes
    assert "missing_v4" not in codes
    assert "missing_final_rehab" not in codes


def test_format_report_shows_full_v4_accounting_audit():
    group_audit = {
        "group": "kitchen",
        "original_group_capped": {"low": 12000, "high": 35000},
        "absorbed_total": {"low": 8000, "high": 30000},
        "package_total": {"low": 15000, "high": 45000},
        "package_net_delta": {"low": 7000, "high": 15000},
        "post_cap_package_adjusted": {"low": 19000, "high": 35000},
        "cap_applied_after_packages": True,
    }
    absorption_audit = {
        "absorbed": {
            "line_items": ["outdated_kitchen_finishes"],
            "room_allowances": ["kitchen_room_allowance"],
            "partial_allocations": ["scratched_flooring:kitchen_primary"],
            "totals": {"low": 8000, "high": 30000, "midpoint": 19000},
        },
        "retained": {
            "partial_allocations": ["scratched_flooring:living_room_1"],
            "totals": {"low": 1000, "high": 4000, "midpoint": 2500},
        },
        "package_net_delta": {"low": 7000, "high": 15000, "midpoint": 11000},
    }
    rec = _make_reconciliation(
        package_total_low=15000,
        package_total_high=45000,
        retained_low=4000,
        retained_high=10000,
        absorbed_total_low=8000,
        absorbed_total_high=30000,
        package_group_reconciliation=[group_audit],
        risk_exposure_high=5000,
    )
    sanity_flags = [
        {"code": "ordinary_flag", "severity": "warning", "message": "Check it."},
        {
            "code": "serious_flag",
            "severity": "strong_warning",
            "message": "Really check it.",
        },
    ]
    pi = _make_photo_intel(
        v3=_make_v3(),
        v4=_make_v4(
            packages=[_make_package(
                package_id="kitchen_partial_rehab__kitchen_primary",
                package_type="kitchen_partial_rehab",
                cost_low=15000,
                cost_high=45000,
                absorbed_total_low=8000,
                absorbed_total_high=30000,
                absorption_audit=absorption_audit,
            )],
            reconciliation=rec,
            estimate_units=_make_estimate_units(),
            sanity_flags=sanity_flags,
        ),
    )

    report = format_report(compare(pi))

    assert "v4 visible_rehab" in report
    assert "v4 package_adjusted_rehab" in report
    assert "v4 latent_risk_exposure" in report
    assert "v4 worst_case_exposure" in report
    assert "v4 final_rehab" in report
    assert "final_rehab basis: package_adjusted_rehab" in report
    assert "kitchen_primary <- kitchen_1, kitchen_2, kitchen_3" in report
    assert "bathroom_primary <- bathroom_1, bathroom_2" in report
    assert "estimate_scope:    marketability_rehab" in report
    assert "absorption_scope:" in report
    assert "absorbed line items: outdated_kitchen_finishes" in report
    assert "absorbed room allowances: kitchen_room_allowance" in report
    assert "absorbed partial allocations: scratched_flooring:kitchen_primary" in report
    assert "retained partial allocations: scratched_flooring:living_room_1" in report
    assert "original_group_capped:" in report
    assert "post_cap_package_adjusted:" in report
    assert "final group total:" in report
    assert "ordinary_flag [warning]" in report
    assert "serious_flag [strong_warning]" in report


def test_v3_plus_v4_with_groups_risk_high_keeps_final_no_risk():
    v3 = _make_v3()
    rec = _make_reconciliation(
        package_total_low=30000,
        package_total_high=70000,
        retained_low=5000,
        retained_high=10000,
        risk_exposure_high=3000,
    )
    v4 = _make_v4(
        packages=[_make_package()],
        reconciliation=rec,
        groups_risk_high=3000,
    )
    pi = _make_photo_intel(v3=v3, v4=v4)
    result = compare(pi)
    report = format_report(result)

    assert result["v4_totals"]["high"] == 80000
    assert result["v4_buckets"]["latent_risk_exposure"]["high"] == 3000
    assert result["v4_buckets"]["worst_case_exposure"]["high"] == 83000
    assert "v4 final_rehab" in report
    assert "$    80,000" in report
    assert "v4 worst_case_exposure" in report
    assert "$    83,000" in report
    codes = {w["code"] for w in result["warnings"]}
    assert "reconciliation_invariant_failed" not in codes


def test_reconciliation_invariant_failure_warning():
    v3 = _make_v3()
    rec = _make_reconciliation(
        package_total_low=30000,
        package_total_high=70000,
        retained_low=5000,
        retained_high=10000,
    )
    rec["final_rehab"]["low"] = 99999
    v4 = _make_v4(packages=[_make_package()], reconciliation=rec)
    pi = _make_photo_intel(v3=v3, v4=v4)
    result = compare(pi)

    invariant_warnings = [
        w for w in result["warnings"] if w["code"] == "reconciliation_invariant_failed"
    ]
    assert len(invariant_warnings) == 1
    detail = invariant_warnings[0]["detail"]
    assert "expected low=35000" in detail
    assert "observed low=99999" in detail


def test_package_audit_empty_lists_use_none_fallback():
    rec = _make_reconciliation(
        package_total_low=30000,
        package_total_high=70000,
        retained_low=5000,
        retained_high=10000,
    )
    pi = _make_photo_intel(
        v3=_make_v3(),
        v4=_make_v4(packages=[_make_package()], reconciliation=rec),
    )

    report = format_report(compare(pi))

    assert "absorbed line items: (none)" in report
    assert "absorbed room allowances: (none)" in report
    assert "absorbed partial allocations: (none)" in report
    assert "retained partial allocations: (none)" in report


def test_estimate_unit_missing_source_rooms_is_explicit():
    rec = _make_reconciliation(
        package_total_low=30000,
        package_total_high=70000,
        retained_low=5000,
        retained_high=10000,
    )
    pi = _make_photo_intel(
        v3=_make_v3(),
        v4=_make_v4(
            packages=[_make_package()],
            reconciliation=rec,
            estimate_units=[{
                "estimate_unit_id": "kitchen_primary",
                "unit_type": "kitchen",
            }],
        ),
    )

    result = compare(pi)
    report = format_report(result)
    codes = {w["code"] for w in result["warnings"]}

    assert "estimate_unit_source_room_surrogate_ids_missing" in codes
    assert "kitchen_primary <- <missing source_room_surrogate_ids>" in report


def test_v3_key_missing():
    pi = _make_photo_intel(v3=None, v4=None)
    result = compare(pi)

    assert result["v3_totals"] is None
    assert result["v4_totals"] is None
    assert result["delta"] is None

    codes = {w["code"] for w in result["warnings"]}
    assert "v3_key_missing" in codes
    assert "missing_v4" in codes


def test_missing_file_errors(tmp_path: Path, capsys):
    nonexistent = tmp_path / "nope.json"
    rc = main(["--run", str(nonexistent)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_invalid_json_errors(tmp_path: Path, capsys):
    bad = tmp_path / "photo_intel.json"
    bad.write_text("{not valid json", encoding="utf-8")
    rc = main(["--run", str(bad)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "invalid json" in captured.err.lower()


def test_directory_resolves_photo_intel(tmp_path: Path, capsys):
    pi = _make_photo_intel(v3=_make_v3())
    artifact = tmp_path / "photo_intel.json"
    artifact.write_text(json.dumps(pi), encoding="utf-8")
    rc = main(["--run", str(tmp_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "RENOVATION ESTIMATE COMPARISON" in captured.out
    assert str(artifact) in captured.out


def test_directory_without_photo_intel_errors(tmp_path: Path, capsys):
    rc = main(["--run", str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "photo_intel.json" in captured.err


def test_does_not_mutate_artifact(tmp_path: Path):
    pi = _make_photo_intel(
        v3=_make_v3(),
        v4=_make_v4(
            packages=[_make_package()],
            reconciliation=_make_reconciliation(
                package_total_low=30000,
                package_total_high=70000,
                retained_low=5000,
                retained_high=10000,
            ),
        ),
    )
    pi_copy = json.loads(json.dumps(pi))
    compare(pi)
    assert pi == pi_copy


def test_passes_through_v4_reconciliation_warnings():
    v3 = _make_v3()
    internal_warning = {
        "code": "package_total_below_absorbed_total_low",
        "package_id": "kitchen_full__rs1",
        "absorbed_total_low": 50000,
        "package_cost_low": 30000,
    }
    rec = _make_reconciliation(
        package_total_low=30000,
        package_total_high=70000,
        retained_low=5000,
        retained_high=10000,
        warnings=[internal_warning],
    )
    v4 = _make_v4(packages=[_make_package()], reconciliation=rec)
    pi = _make_photo_intel(v3=v3, v4=v4)
    result = compare(pi)

    assert result["reconciliation"]["v4_internal_warnings"] == [internal_warning]
    report = format_report(result)
    assert "package_total_below_absorbed_total_low" in report


def test_compare_uses_persisted_sanity_flags_when_present():
    rec = _make_reconciliation(
        package_total_low=30000,
        package_total_high=70000,
        retained_low=5000,
        retained_high=10000,
        risk_exposure_high=10000,
    )
    persisted = [{
        "code": "persisted_manual_flag",
        "severity": "warning",
        "message": "Persisted flag wins.",
        "value": 1,
    }]
    pi = _make_photo_intel(
        v3=_make_v3(),
        v4=_make_v4(
            packages=[_make_package()],
            reconciliation=rec,
            sanity_flags=persisted,
        ),
        property_metadata={"list_price": 100000},
    )

    result = compare(pi)
    report = format_report(result)

    assert result["sanity_flags_source"] == "persisted"
    assert result["sanity_flags"] == persisted
    assert "persisted_manual_flag" in report
    assert "derived; not persisted" not in report


def test_compare_derives_sanity_flags_when_missing_from_v4():
    rec = _make_reconciliation(
        package_total_low=30000,
        package_total_high=70000,
        retained_low=5000,
        retained_high=10000,
        risk_exposure_high=10000,
    )
    pi = _make_photo_intel(
        v3=_make_v3(),
        v4=_make_v4(
            packages=[_make_package()],
            reconciliation=rec,
            groups_risk_high=10000,
        ),
        property_metadata={"list_price": 100000},
    )

    result = compare(pi)
    report = format_report(result)
    codes = {flag["code"] for flag in result["sanity_flags"]}

    assert result["sanity_flags_source"] == "derived"
    assert "package_adjusted_high_gt_50pct_price" in codes
    assert "worst_case_high_gt_80pct_price" in codes
    assert "SANITY FLAGS (derived; not persisted)" in report
    assert (
        "This is extreme relative to list price; interpret as "
        "worst-case/inspection-dependent."
    ) in report


def test_format_report_shows_explicit_v4_bucket_sections():
    rec = _make_reconciliation(
        package_total_low=30000,
        package_total_high=70000,
        retained_low=5000,
        retained_high=10000,
        risk_exposure_high=3000,
    )
    pi = _make_photo_intel(
        v3=_make_v3(),
        v4=_make_v4(packages=[_make_package()], reconciliation=rec, groups_risk_high=3000),
    )

    report = format_report(compare(pi))

    assert "v4 visible_rehab" in report
    assert "v4 package_adjusted_rehab" in report
    assert "v4 latent_risk_exposure" in report
    assert "v4 worst_case_exposure" in report
    assert "v4 final_rehab" in report
    assert "final_rehab basis: package_adjusted_rehab" in report


def test_format_report_shows_group_cap_reconciliation():
    group_audit = {
        "group": "kitchen",
        "original_group_raw": {"low": 20000, "high": 58000},
        "original_group_capped": {"low": 12000, "high": 35000},
        "absorbed_total": {"low": 8000, "high": 30000},
        "package_total": {"low": 15000, "high": 45000},
        "package_net_delta": {"low": 7000, "high": 15000},
        "pre_cap_package_adjusted": {"low": 19000, "high": 50000},
        "post_cap_package_adjusted": {"low": 19000, "high": 35000},
        "cap_applied_after_packages": True,
        "cap_override": False,
    }
    rec = _make_reconciliation(
        package_total_low=15000,
        package_total_high=45000,
        retained_low=12000,
        retained_high=28000,
        absorbed_total_low=8000,
        absorbed_total_high=30000,
        package_group_reconciliation=[group_audit],
    )
    pi = _make_photo_intel(
        v3=_make_v3(),
        v4=_make_v4(packages=[_make_package()], reconciliation=rec),
    )

    result = compare(pi)
    report = format_report(result)

    assert result["reconciliation"]["package_group_reconciliation"] == [group_audit]
    assert "group reconciliation" in report
    assert "kitchen" in report
    assert "post_cap_package_adjusted: low=$    19,000, high=$    35,000" in report
    assert "final group total:         low=$    19,000, high=$    35,000" in report
    assert "$    35,000" in report


def test_format_report_runs_for_all_states():
    """Smoke test: format_report should not raise for any of the typical shapes."""
    for pi in [
        _make_photo_intel(),
        _make_photo_intel(v3=_make_v3()),
        _make_photo_intel(
            v3=_make_v3(),
            v4=_make_v4(packages=[], reconciliation=None, packages_enabled=False),
        ),
        _make_photo_intel(
            v3=_make_v3(),
            v4=_make_v4(
                packages=[_make_package()],
                reconciliation=_make_reconciliation(
                    package_total_low=30000,
                    package_total_high=70000,
                    retained_low=5000,
                    retained_high=10000,
                ),
            ),
        ),
    ]:
        report = format_report(compare(pi))
        assert "RENOVATION ESTIMATE COMPARISON" in report
