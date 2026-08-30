"""Unit tests for the read-only legacy-v2 rescore (no live models).

Covers: offline replay with zero VLM invocations, output confined to
runs/legacy_v2_rescore/, the catalog-sha guard, and the
not_available_legacy_projection markers.
"""
import json

import pytest

from tools import benchmark_pass2a as bench
from tools import benchmark_pass2a_legacy as legacy
from tools import benchmark_pass2a_packages as pkg

from tests.test_benchmark_pass2a_packages import (  # shared fixtures/helpers
    CATALOG, PROP, VALID_GOLD, _catalog_stub, _config, _manifest, _write_gold,
    pkg_tree,
)


def _write_legacy_tree(tmp_path, catalog_sha="cat-sha"):
    """A minimal frozen v2 lineage: one variant artifact per cell + saved
    verifications + the resolution fingerprint recording the catalog sha."""
    runs = bench.RUNS_DIR
    (pkg.LEGACY_TAIL_DIR).mkdir(parents=True, exist_ok=True)
    (pkg.LEGACY_TAIL_DIR / "fingerprint.json").write_text(
        json.dumps({"layer": "package_resolution",
                    "catalog_sha256": catalog_sha}), encoding="utf-8")
    artifact = {"photos": {}, "product_issues_flat": [{"issue_id": "x"}],
                "property_metadata": {}}
    for cell, (variant, mode) in pkg.cells_for_round(
            "baseline", "checklist").items():
        stage = f"variant_{variant}"
        if mode == "cap25":
            a_dir = runs / stage / "rep1" / PROP / f"{stage}_rep1"
        else:
            a_dir = (pkg.LEGACY_CELLS_DIR / cell / "rep1" / PROP
                     / f"pkgcell_{cell}_rep1")
        a_dir.mkdir(parents=True, exist_ok=True)
        (a_dir / "photo_intel.json").write_text(json.dumps(artifact),
                                                encoding="utf-8")
        v_dir = pkg.LEGACY_PASS2F_DIR / cell / "rep1" / PROP
        v_dir.mkdir(parents=True, exist_ok=True)
        (v_dir / "verifications.json").write_text(json.dumps({
            "verifications": {"kitchen_modernization__kitchen_primary": {
                "package_id": "kitchen_modernization__kitchen_primary",
                "verification_status": "confirmed"}}}), encoding="utf-8")


@pytest.fixture
def legacy_stubs(monkeypatch):
    """Offline v4 stub + a catalog sha pin + tripwires on every VLM path."""
    import tools.comparison_common as cc
    import tools.renovation_estimate_v4 as rev4
    import tools.vlm_client as vc

    monkeypatch.setattr(cc, "sha256_file", lambda path: "cat-sha")
    monkeypatch.setattr(
        vc, "create_vlm_client",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError(
            "legacy rescore must never build a VLM client")))

    calls = []

    def fake_v4(issues_flat, catalog, photos, property_metadata=None,
                package_verifications=None, **kwargs):
        calls.append({"verifications": package_verifications})
        assert "pass_2f_vlm_client" not in kwargs or \
            kwargs["pass_2f_vlm_client"] is None
        pkg_row = {
            "package_id": "kitchen_modernization__kitchen_primary",
            "package_type": "kitchen_modernization",
            "estimate_unit_id": "kitchen_primary",
            "pricing_profile": "kitchen_full_rehab",
            "verification_status": "confirmed" if package_verifications
            else "not_run",
            "estimate_eligible": bool(package_verifications),
            "estimate_display_only": False, "audit_only": False,
            "cost_low": 30000, "cost_high": 70000, "cost_midpoint": 50000,
            "supporting_catalog_item_ids": ["cab_dated"],
            "review_photo_keys": ["photo_001.jpg"],
            "evidence_summary": "dated",
        }
        return {
            "package_candidates": [dict(pkg_row)],
            "packages": [pkg_row],
            "groups": [{"line_items": [
                {"catalog_item_id": "brick_weathered", "name": "Brick",
                 "billable_estimate_unit_id": "bedroom_1",
                 "cost_low": 500, "cost_high": 900, "package_id": None,
                 "trade_bucket": "masonry_exterior_structure",
                 "is_valid_detection": None}]}],
            "estimate_units": [
                {"estimate_unit_id": "kitchen_primary",
                 "photo_ids": ["photo_001.jpg"]},
                {"estimate_unit_id": "bedroom_1",
                 "photo_ids": ["photo_002.jpg"]}],
            "final_rehab": {"low": 30500, "high": 70900, "midpoint": 50700},
            "pass_2f_trace": {"ran": False,
                              "reason": "provided_verifications"}
            if package_verifications else {"ran": False, "reason": "no_2f"},
        }

    monkeypatch.setattr(rev4, "compute_renovation_estimate_v4", fake_v4)
    return calls


def test_legacy_rescore_offline_and_confined(pkg_tree, legacy_stubs):
    _write_legacy_tree(pkg_tree)
    _write_gold(pkg_tree)
    out = legacy.stage_legacy_rescore(_config(), _manifest(pkg_tree, photos=2),
                                      "baseline_vs_checklist")
    assert out == bench.RUNS_DIR / "legacy_v2_rescore"
    scored = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    assert scored["lineage"] == "legacy_v2_rescore"
    # The repaired projection retained the unreviewed standalone line and the
    # canonical-room alignment credited both gold targets.
    rep = scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"]
    assert rep["required_work"]["brick_weathered@bedroom_A"][
        "status"] == "satisfied"
    assert rep["expected_packages"]["kitchen_modernization__kitchen"][
        "status"] == "matched_exact"
    # Unrecoverable v2 fields are explicit nulls with the reason.
    for field in ("scene_capture_sha", "pass_1a_routing_provenance",
                  "frozen_scene_consistency"):
        assert scored["legacy_limits"][field] == {
            "value": None, "reason": "not_available_legacy_projection"}
    # Replay passed the STORED verifications through (never re-verified).
    replays = [c for c in legacy_stubs if c["verifications"]]
    assert len(replays) == 3      # one per cell
    assert all("kitchen_modernization__kitchen_primary" in c["verifications"]
               for c in replays)
    # Output stays out of the frozen v2 dirs and the v3 lineage.
    assert not (pkg.LEGACY_EVAL_DIR).exists()
    assert not (bench.V3_DIR / "package_2f").exists()
    assert (out / "report.md").read_text(encoding="utf-8").startswith(
        "# Legacy v2 rescore")
    # Scorer state was restored after the rebind.
    assert pkg.PASS2F_DIR == bench.V3_DIR / "package_2f"


def test_legacy_rescore_aborts_on_catalog_drift(pkg_tree, legacy_stubs):
    _write_legacy_tree(pkg_tree, catalog_sha="an-older-catalog")
    _write_gold(pkg_tree)
    with pytest.raises(SystemExit, match="catalog changed"):
        legacy.stage_legacy_rescore(_config(), _manifest(pkg_tree, photos=2),
                                    "baseline_vs_checklist")
    assert not (bench.RUNS_DIR / "legacy_v2_rescore").exists()


def test_legacy_rescore_is_resumable_without_recompute(pkg_tree, legacy_stubs):
    _write_legacy_tree(pkg_tree)
    _write_gold(pkg_tree)
    legacy.stage_legacy_rescore(_config(), _manifest(pkg_tree, photos=2),
                                "baseline_vs_checklist")
    before = len(legacy_stubs)
    legacy.stage_legacy_rescore(_config(), _manifest(pkg_tree, photos=2),
                                "baseline_vs_checklist")
    # Second run only rescoreds (probe+final per cell skipped).
    assert len(legacy_stubs) == before
