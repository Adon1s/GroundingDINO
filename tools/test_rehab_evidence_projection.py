"""Regression fixtures for the rehab evidence allocation projection.

Plain-assert script (no pytest dependency), matching this repo's
script-oriented conventions:

    .venv\\Scripts\\python.exe -m tools.test_rehab_evidence_projection

Covers: exact endpoint reconciliation, photo-supported-only and
inspection-only estimates, excluded risk with and without a headline,
malformed-range fail-closed, deterministic projection ids, and the
2100 Aftonbrae Dr (redfin_125800935) golden numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.rehab_evidence_projection import (
    EVIDENCE_PROJECTION_POLICY_VERSION,
    EVIDENCE_PROJECTION_VERSION,
    build_rehab_evidence_projection,
    stamp_evidence_projection_provenance,
)

_CHECKS = 0


def check(condition: bool, label: str) -> None:
    global _CHECKS
    _CHECKS += 1
    if not condition:
        raise AssertionError(f"FAILED: {label}")


def estimate(
    *,
    resale_ready=None,
    inspection=None,
    risk=None,
) -> dict:
    totals = {}
    if inspection is not None:
        totals["inspection_allowance_total"] = inspection
    if risk is not None:
        totals["risk_exposure_total"] = risk
    est = {"version": "renovation_estimate_v4", "totals": totals}
    if resale_ready is not None:
        est["final_rehab_resale_ready"] = resale_ready
    return est


def main() -> int:
    # -- Aftonbrae golden case (redfin_125800935) ---------------------------
    aftonbrae = build_rehab_evidence_projection(estimate(
        resale_ready={"low": 4567, "high": 16125},
        inspection={"low": 158, "high": 633},
        risk={"low": 1187, "high": 19788},
    ))
    check(aftonbrae is not None, "aftonbrae: projection built")
    check(aftonbrae["photo_supported"] == {"low": 4567, "high": 16125},
          "aftonbrae: photo_supported")
    check(aftonbrae["needs_inspection"] == {"low": 158, "high": 633},
          "aftonbrae: needs_inspection")
    check(aftonbrae["headline"] == {"low": 4725, "high": 16758},
          "aftonbrae: headline is the endpoint-wise sum")
    check(aftonbrae["risk_exposure"] == {"low": 1187, "high": 19788},
          "aftonbrae: risk_exposure carried")
    check(aftonbrae["risk_exposure_included"] is False,
          "aftonbrae: risk excluded")
    check(aftonbrae["reconciliation"]["raw_exact"] is True,
          "aftonbrae: raw reconciliation asserted")
    check(aftonbrae["version"] == EVIDENCE_PROJECTION_VERSION,
          "aftonbrae: version")
    check(aftonbrae["policy_version"] == EVIDENCE_PROJECTION_POLICY_VERSION,
          "aftonbrae: policy version")
    check(aftonbrae["rounding_policy"] == "compact_currency_v1",
          "aftonbrae: rounding policy declared")

    # -- Exact reconciliation invariant on arbitrary inputs ------------------
    for photo, insp in (
        ({"low": 0, "high": 0}, {"low": 0, "high": 0}),
        ({"low": 999, "high": 999}, {"low": 1, "high": 1}),
        ({"low": 1_200_000, "high": 3_400_000}, {"low": 50_000, "high": 90_000}),
    ):
        p = build_rehab_evidence_projection(estimate(resale_ready=photo, inspection=insp))
        check(p is not None, "reconcile: projection built")
        check(
            p["headline"]["low"] == p["photo_supported"]["low"] + p["needs_inspection"]["low"]
            and p["headline"]["high"] == p["photo_supported"]["high"] + p["needs_inspection"]["high"],
            "reconcile: photo + inspection == headline at both endpoints",
        )

    # -- Photo-supported-only (no inspection allowance) ----------------------
    photo_only = build_rehab_evidence_projection(estimate(
        resale_ready={"low": 5000, "high": 9000},
    ))
    check(photo_only["needs_inspection"] == {"low": 0, "high": 0},
          "photo-only: inspection defaults to zero")
    check(photo_only["headline"] == {"low": 5000, "high": 9000},
          "photo-only: headline equals photo_supported")

    # -- Inspection-only ------------------------------------------------------
    inspection_only = build_rehab_evidence_projection(estimate(
        resale_ready={"low": 0, "high": 0},
        inspection={"low": 400, "high": 900},
    ))
    check(inspection_only["photo_supported"] == {"low": 0, "high": 0},
          "inspection-only: photo_supported zero")
    check(inspection_only["headline"] == {"low": 400, "high": 900},
          "inspection-only: headline equals needs_inspection")

    # -- Excluded risk without a headline ------------------------------------
    risk_only = build_rehab_evidence_projection(estimate(
        risk={"low": 1000, "high": 5000},
    ))
    check(risk_only is not None, "risk-only: projection built")
    check("headline" not in risk_only and "photo_supported" not in risk_only
          and "needs_inspection" not in risk_only and "reconciliation" not in risk_only,
          "risk-only: no fabricated allocation or assertion")
    check(risk_only["risk_exposure"] == {"low": 1000, "high": 5000}
          and risk_only["risk_exposure_included"] is False,
          "risk-only: risk carried and excluded")
    check(build_rehab_evidence_projection(estimate()) is None,
          "risk-only: nothing to project -> None")

    # -- Fail closed on malformed ranges --------------------------------------
    for bad in (
        estimate(resale_ready={"low": 100}, inspection={"low": 0, "high": 0}),
        estimate(resale_ready={"low": 900, "high": 100}),
        estimate(resale_ready={"low": -5, "high": 100}),
        estimate(resale_ready={"low": "x", "high": 100}),
        estimate(resale_ready={"low": 100, "high": 200},
                 inspection={"low": 700, "high": 20}),
        estimate(resale_ready={"low": 100, "high": 200},
                 risk={"low": float("nan"), "high": 3}),
        "not_a_dict",
        None,
    ):
        check(build_rehab_evidence_projection(bad) is None,
              f"fail-closed: malformed input rejected ({bad!r:.60})")

    # -- Deterministic, content-based projection id ----------------------------
    a = build_rehab_evidence_projection(estimate(
        resale_ready={"low": 4567, "high": 16125},
        inspection={"low": 158, "high": 633},
        risk={"low": 1187, "high": 19788},
    ))
    b = build_rehab_evidence_projection(estimate(
        resale_ready={"low": 4567, "high": 16125},
        inspection={"low": 158, "high": 633},
        risk={"low": 1187, "high": 19788},
    ))
    for proj, status in ((a, "native"), (b, "reprojected")):
        stamp_evidence_projection_provenance(
            proj,
            run_id="20260528_031858_6d1c59a0",
            completed_at="2026-05-28T08:27:50.449531Z",
            source_artifact="redfin_125800935/20260528_031858_6d1c59a0/photo_intel.json",
            projection_status=status,
            product_policy_version="quarantine_v1",
        )
    check(a["projection_id"] == b["projection_id"],
          "projection_id: stable across rebuilds of the same content/run")
    check(a["projection_id"].startswith("rep1_"), "projection_id: namespaced")
    check(a["provenance"]["source_run_id"] == "20260528_031858_6d1c59a0"
          and a["provenance"]["completed_at"] == "2026-05-28T08:27:50.449531Z"
          and a["provenance"]["product_policy_version"] == "quarantine_v1",
          "provenance: run id, completion time, and product policy stamped")
    check("\\" not in a["provenance"]["source_artifact"]
          and ":" not in a["provenance"]["source_artifact"],
          "provenance: source artifact is a relative identity, not a path")

    c = build_rehab_evidence_projection(estimate(
        resale_ready={"low": 4567, "high": 16126},
        inspection={"low": 158, "high": 633},
    ))
    stamp_evidence_projection_provenance(
        c,
        run_id="20260528_031858_6d1c59a0",
        completed_at="2026-05-28T08:27:50.449531Z",
        source_artifact="redfin_125800935/20260528_031858_6d1c59a0/photo_intel.json",
        projection_status="native",
        product_policy_version="quarantine_v1",
    )
    check(c["projection_id"] != a["projection_id"],
          "projection_id: changes when range content changes")

    print(f"OK: {_CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
