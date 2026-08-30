"""Read-only rescore of the frozen legacy-v2 package artifacts.

Replays the saved Pass 2f verifications through compute_renovation_estimate_v4
offline (zero VLM calls — 2f is skipped whenever verifications are supplied),
reprojects with the repaired tri-state line-item projection, and scores with
the canonical-room v3 scorer. Nothing under the flat runs/* v2 lineage is
modified; all output goes to runs/legacy_v2_rescore/.

These are developmental diagnostics over the old artifacts, NOT authoritative
results — the fresh v3 run supplies those. Fields whose inputs the v2
projection already discarded are reported as null with the reason
"not_available_legacy_projection".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from tools import benchmark_pass2a as bench
from tools import benchmark_pass2a_packages as pkg
from tools.benchmark_pass2a import _load_json, _utcnow, _write_json, rep_job_id

_UNRECOVERABLE = {"value": None, "reason": "not_available_legacy_projection"}


def legacy_rescore_dir() -> Path:
    return bench.RUNS_DIR / "legacy_v2_rescore"


def _legacy_cell_artifact_path(cell: str, rep: int, prop: str,
                               variant_a: str, variant_b: str) -> Path:
    """The v2 lineage's cell-artifact layout (flat runs/*)."""
    cells = pkg.cells_for_round(variant_a, variant_b)
    variant, mode = cells[cell]
    stage = f"variant_{variant}"
    if mode == "cap25":
        return (bench.RUNS_DIR / stage / f"rep{rep}" / prop
                / rep_job_id(stage, rep) / "photo_intel.json")
    return (pkg.LEGACY_CELLS_DIR / cell / f"rep{rep}" / prop
            / f"pkgcell_{cell}_rep{rep}" / "photo_intel.json")


def _load_legacy_catalog() -> tuple:
    """The catalog the v2 artifacts were priced under, verified by the
    recorded sha. When the working-tree catalog has since changed, the exact
    v2-era file is recovered from the git commit recorded in info.json (data
    recovery, not a compatibility layer); any mismatch aborts."""
    import hashlib
    import subprocess

    from tools import pipeline_config as cfg
    from tools.artifact_writers import load_issue_catalog
    from tools.comparison_common import sha256_file

    fp_path = pkg.LEGACY_TAIL_DIR / "fingerprint.json"
    if not fp_path.is_file():
        raise SystemExit(f"legacy resolution fingerprint missing: {fp_path}")
    recorded = _load_json(fp_path).get("catalog_sha256")
    current_path = Path(cfg.ISSUE_CATALOG_PATH)
    current = sha256_file(current_path)
    if recorded == current:
        return load_issue_catalog(current_path), recorded, "working_tree"

    info_path = pkg.LEGACY_TAIL_DIR / "info.json"
    head = (_load_json(info_path).get("git_head")
            if info_path.is_file() else None)
    blob = None
    if head:
        try:
            blob = subprocess.run(
                ["git", "show", f"{head}:tools/issue_catalog_kind_v2.json"],
                cwd=str(bench.ROOT), capture_output=True, check=True,
            ).stdout
        except Exception:
            blob = None
    if blob:
        # The recorded sha hashed the on-disk file, which may carry either
        # line ending; the git blob is the LF form.
        for candidate in (blob, blob.replace(b"\n", b"\r\n")):
            if hashlib.sha256(candidate).hexdigest() == recorded:
                snapshot = legacy_rescore_dir() / "catalog_v2_snapshot.json"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(blob)
                return (load_issue_catalog(snapshot), recorded,
                        f"git:{head[:12]}")
    raise SystemExit(
        "legacy rescore refused: the catalog changed since the v2 run "
        f"(recorded {str(recorded)[:12]}..., current {current[:12]}...) and "
        "the v2-era catalog could not be recovered from git — an offline "
        "replay would price against different definitions.")


def stage_legacy_rescore(config: Dict[str, Any], manifest: Dict[str, Any],
                         round_label: str) -> Path:
    from tools.benchmark_pass2a_scoring import score_package_round
    from tools.renovation_estimate_v4 import compute_renovation_estimate_v4

    variant_a, variant_b = pkg.package_round_variants(config, round_label)
    catalog, catalog_sha, catalog_source = _load_legacy_catalog()
    out_root = legacy_rescore_dir()
    cells_root = out_root / "package_2f"

    rebuilt = missing = 0
    for cell in pkg.cells_for_round(variant_a, variant_b):
        for rep in range(1, int(config["repeats"]) + 1):
            for prop in sorted(manifest["properties"]):
                artifact_path = _legacy_cell_artifact_path(
                    cell, rep, prop, variant_a, variant_b)
                vers_path = (pkg.LEGACY_PASS2F_DIR / cell / f"rep{rep}" / prop
                             / "verifications.json")
                if not (artifact_path.is_file() and vers_path.is_file()):
                    missing += 1
                    continue
                out_dir = cells_root / cell / f"rep{rep}" / prop
                if (out_dir / "final_estimate.json").is_file():
                    rebuilt += 1
                    continue
                artifact = _load_json(artifact_path)
                verifications = _load_json(vers_path)["verifications"]
                issues_flat = pkg.artifact_issues_flat(artifact)
                photos = artifact.get("photos") or {}
                metadata = artifact.get("property_metadata") or None
                probe = compute_renovation_estimate_v4(
                    issues_flat, catalog, photos, property_metadata=metadata)
                final = compute_renovation_estimate_v4(
                    issues_flat, catalog, photos, property_metadata=metadata,
                    package_verifications=verifications,
                )
                trace = final.get("pass_2f_trace") or {}
                if trace.get("ran"):
                    raise SystemExit(
                        f"legacy rescore invariant broken: 2f ran for {cell} "
                        f"rep{rep} {prop} — replay must stay offline")
                _write_json(out_dir / "candidates.json", {
                    "cell": cell, "rep": rep, "property_key": prop,
                    "candidates": [
                        pkg.package_projection(p)
                        for p in (probe.get("package_candidates") or [])
                        if isinstance(p, dict)],
                    "estimate_units": pkg.estimate_units_projection(probe),
                })
                _write_json(out_dir / "verifications.json", {
                    "cell": cell, "rep": rep, "property_key": prop,
                    "verifications": verifications,
                    "source": str(vers_path),
                })
                _write_json(out_dir / "final_estimate.json", {
                    "cell": cell, "rep": rep, "property_key": prop,
                    "packages": [pkg.package_projection(p)
                                 for p in (final.get("packages") or [])],
                    "line_items": pkg.v4_line_items(final),
                    "estimate_units": pkg.estimate_units_projection(final),
                    "final_rehab": final.get("final_rehab") or {},
                    "pass_2f_trace": trace,
                })
                rebuilt += 1

    gold = pkg.load_package_gold()
    # Score the rebuilt tree with the v3 scorer. PASS2F_DIR/V3_DIR are
    # rebound for the duration so cell reads and cap-bound checkpoint reads
    # hit the legacy lineage; the v2 stage dirs themselves are never written.
    saved_pass2f, saved_v3 = pkg.PASS2F_DIR, bench.V3_DIR
    pkg.PASS2F_DIR, bench.V3_DIR = cells_root, bench.RUNS_DIR
    try:
        scored = score_package_round(config, manifest, round_label, gold, {})
    finally:
        pkg.PASS2F_DIR, bench.V3_DIR = saved_pass2f, saved_v3
    scored.pop("_review_rows", None)
    scored["lineage"] = "legacy_v2_rescore"
    scored["note"] = ("Diagnostics over frozen v2 artifacts (Qwen Pass 1a, "
                      "uncontrolled scenes). NOT authoritative — the v3 run "
                      "supplies corrected results.")
    scored["catalog_sha256"] = catalog_sha
    scored["catalog_source"] = catalog_source
    scored["rescored_at"] = _utcnow()
    scored["legacy_limits"] = {
        "scene_capture_sha": dict(_UNRECOVERABLE),
        "pass_1a_routing_provenance": dict(_UNRECOVERABLE),
        "frozen_scene_consistency": dict(_UNRECOVERABLE),
    }
    _write_json(out_root / "scores.json", scored)

    lines = ["# Legacy v2 rescore (diagnostics only)", "",
             scored["note"], ""]
    lines += pkg.package_report_md_lines(scored)
    (out_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"legacy rescore: {rebuilt} cells rebuilt offline, {missing} "
          f"missing -> {out_root}")
    return out_root
