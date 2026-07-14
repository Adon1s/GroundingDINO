"""Reproject product views of saved photo_intel artifacts under the current
product-quarantine policy.

Legacy artifacts were written before quarantined trades (see
``product_quarantined`` on catalog trade_buckets) were excluded from product
surfaces, so their scoring, summary_v1, renovation_estimate_v4, and
ui_priorities_v1 may carry quarantined influence. This tool recomputes all of
them from quarantine-filtered product lanes using the canonical Python engine.

Deterministic: the VLM is never called. Package verification is reused from
the stored v4 output, but ONLY for packages whose original evidence was
entirely non-quarantined. Tainted packages lose their verification, and since
package finalization requires confirmation, they are dropped even when the
same package id is re-inferred from the filtered issues.

Artifacts missing usable issue lanes are stamped
``product_projection_status: "needs_reanalysis"`` with product fields nulled —
never stale or zero totals.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import pipeline_config as cfg
from tools.artifact_writers import (
    _build_ui_priorities_v1,
    _strip_pass_2f_audit_rationale,
    load_issue_catalog,
)
from tools.backfill_reno_v4 import _resolve_property_metadata_from_artifact
from tools.costing import compute_scoring
from tools.pipeline_common import PRODUCT_POLICY_VERSION
from tools.property_summary_pass import build_property_summary_v1, load_catalog_index
from tools.rehab_packages import PACKAGE_VERIFICATION_NOT_RUN
from tools.renovation_estimate import (
    filter_product_issues,
    product_quarantined_trade_buckets,
)
from tools.renovation_estimate_v4 import compute_renovation_estimate_v4

# Product surfaces nulled when an artifact cannot be reprojected.
_PRODUCT_FIELDS = (
    "scoring",
    "summary_v1",
    "renovation_estimate_v4",
    "ui_priorities_v1",
    "product_issues_flat",
    "product_estimate_issues_flat",
)

_VERIFICATION_CARRY_FIELDS = (
    "package_type",
    "verification_status",
    "confirmed_issue_ids",
    "rejected_issue_ids",
    "reviewed_issue_ids",
    "evidence_summary",
    "review_photo_keys",
    "review_image_paths",
    "review_source",
    "visible_room_count",
    "visible_room_count_evidence",
)


def _quarantined_catalog_item_ids(issue_catalog: Dict[str, Any]) -> frozenset:
    quarantined_trades = product_quarantined_trade_buckets(issue_catalog)
    return frozenset(
        item["id"]
        for item in (issue_catalog.get("items") or [])
        if isinstance(item, dict) and item.get("id")
        and str(item.get("trade_bucket") or "") in quarantined_trades
    )


def _stored_package_evidence_ids(package: Dict[str, Any]) -> set:
    ids = {
        str(item.get("catalog_item_id") or "")
        for item in (package.get("evidence_items") or [])
        if isinstance(item, dict)
    }
    ids.update(str(v) for v in (package.get("supporting_catalog_item_ids") or []))
    ids.discard("")
    return ids


def collect_stored_verifications(
    artifact: Dict[str, Any],
    quarantined_item_ids: frozenset,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Rebuild a package_verifications dict from a stored v4 output.

    Returns (verifications, tainted). A stored package contributes a
    verification record only when its verification actually ran AND none of
    its original evidence resolves to a quarantined catalog item. Tainted
    packages are returned for the audit trail; withholding their verification
    is the blacklist — confirmation-required finalization drops them even if
    re-inferred under the same package id.
    """
    v4 = artifact.get("renovation_estimate_v4")
    if not isinstance(v4, dict):
        return {}, []
    stored = v4.get("package_candidates")
    if not isinstance(stored, list) or not stored:
        stored = v4.get("packages") if isinstance(v4.get("packages"), list) else []

    verifications: Dict[str, Dict[str, Any]] = {}
    tainted: List[Dict[str, Any]] = []
    for package in stored:
        if not isinstance(package, dict):
            continue
        package_id = str(package.get("package_id") or "")
        status = str(package.get("verification_status") or "").strip().lower()
        if not package_id or not status or status == PACKAGE_VERIFICATION_NOT_RUN:
            continue
        evidence_ids = _stored_package_evidence_ids(package)
        quarantined_evidence = sorted(evidence_ids & quarantined_item_ids)
        if quarantined_evidence:
            tainted.append({
                "package_id": package_id,
                "package_type": package.get("package_type"),
                "verification_status": status,
                "cost_low": int(package.get("cost_low") or 0),
                "cost_high": int(package.get("cost_high") or 0),
                "quarantined_catalog_item_ids": quarantined_evidence,
            })
            continue
        record: Dict[str, Any] = {"package_id": package_id}
        for field in _VERIFICATION_CARRY_FIELDS:
            if field in package:
                record[field] = package[field]
        verifications[package_id] = record
    return verifications, tainted


def _headline_total(estimate: Any) -> Dict[str, int]:
    if not isinstance(estimate, dict):
        return {"low": 0, "high": 0}
    for key in ("final_rehab", "raw_totals", "primary_estimate"):
        value = estimate.get(key)
        if isinstance(value, dict):
            return {
                "low": int(value.get("low") or 0),
                "high": int(value.get("high") or 0),
            }
    return {"low": 0, "high": 0}


def _write_with_backup(artifact_path: Path, artifact: Dict[str, Any]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = artifact_path.parent / (
        f"{artifact_path.stem}.pre_reprojection_{stamp}{artifact_path.suffix}"
    )
    shutil.copy2(artifact_path, backup_path)
    tmp_path = artifact_path.parent / (artifact_path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)
        tmp_path.replace(artifact_path)
    except OSError:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return backup_path


def reproject_artifact(
    artifact_path: Path,
    issue_catalog: Dict[str, Any],
    *,
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Reproject one photo_intel.json. Returns a status dict, never raises
    for expected user-facing failures."""
    try:
        with artifact_path.open("r", encoding="utf-8") as f:
            artifact = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"status": "error", "reason": f"cannot load artifact: {e}"}
    if not isinstance(artifact, dict):
        return {"status": "error", "reason": "artifact root must be a JSON object"}

    already_current = (
        artifact.get("product_policy_version") == PRODUCT_POLICY_VERSION
        and artifact.get("product_projection_status") in ("native", "reprojected")
    )
    if already_current and not force:
        return {
            "status": "skip_current",
            "projection_status": artifact.get("product_projection_status"),
        }

    raw_display = artifact.get("issues_flat")
    raw_estimate = artifact.get("estimate_issues_flat")
    if not isinstance(raw_display, list) or not raw_display:
        # No usable inputs: fail closed, never leave stale product fields.
        if dry_run:
            return {"status": "dry_run_needs_reanalysis"}
        for field in _PRODUCT_FIELDS:
            artifact[field] = None
        artifact["product_policy_version"] = PRODUCT_POLICY_VERSION
        artifact["catalog_version"] = str(issue_catalog.get("version") or "")
        artifact["product_projection_status"] = "needs_reanalysis"
        try:
            backup_path = _write_with_backup(artifact_path, artifact)
        except OSError as e:
            return {"status": "error", "reason": f"write failed: {e}"}
        return {"status": "needs_reanalysis", "backup_path": str(backup_path)}

    quarantined_item_ids = _quarantined_catalog_item_ids(issue_catalog)
    product_display = filter_product_issues(raw_display, issue_catalog)
    product_estimate = filter_product_issues(
        raw_estimate if isinstance(raw_estimate, list) else [],
        issue_catalog,
    )
    photos = artifact.get("photos") or {}
    property_key = str(
        (artifact.get("property") or {}).get("property_key")
        or artifact.get("property_key") or ""
    )
    run_id = str((artifact.get("run") or {}).get("run_id") or "")

    old_total = _headline_total(artifact.get("renovation_estimate_v4"))
    verifications, tainted = collect_stored_verifications(
        artifact, quarantined_item_ids,
    )

    try:
        scoring = compute_scoring(
            issues_flat=product_display,
            issue_catalog=issue_catalog,
            n_photos=len(photos),
        )
        summary_v1 = build_property_summary_v1(
            property_key=property_key,
            run_id=run_id,
            issues_flat=product_display,
            catalog_index=load_catalog_index(issue_catalog),
        )
        # Lane choice follows the RAW canonical lane's existence, mirroring
        # the live writer.
        renovation_issues = (
            product_estimate if (isinstance(raw_estimate, list) and raw_estimate)
            else product_display
        )
        property_metadata = _resolve_property_metadata_from_artifact(artifact)
        v4_est = compute_renovation_estimate_v4(
            issues_flat=renovation_issues,
            issue_catalog=issue_catalog,
            photos=photos,
            property_metadata=property_metadata or None,
            package_verifications=verifications,
            pass_2f_vlm_client=None,
        )
        _strip_pass_2f_audit_rationale(v4_est)
        ui_priorities = _build_ui_priorities_v1(
            issues_flat=renovation_issues,
            issue_catalog=issue_catalog,
            renovation_estimate_v4=v4_est,
        )
    except Exception as e:
        return {"status": "error", "reason": f"reprojection failed: {e}"}

    new_total = _headline_total(v4_est)
    result = {
        "status": "dry_run" if dry_run else "reprojected",
        "quarantined_display_issues": len(raw_display) - len(product_display),
        "quarantined_estimate_issues": (
            len(raw_estimate) - len(product_estimate)
            if isinstance(raw_estimate, list) else 0
        ),
        "verifications_reused": len(verifications),
        "tainted_packages": tainted,
        "old_total": old_total,
        "new_total": new_total,
    }
    if dry_run:
        return result

    artifact["product_issues_flat"] = product_display
    artifact["product_issues_flat_count"] = len(product_display)
    artifact["product_estimate_issues_flat"] = product_estimate
    artifact["product_estimate_issues_flat_count"] = len(product_estimate)
    artifact["scoring"] = scoring
    artifact["summary_v1"] = summary_v1
    artifact["renovation_estimate_v4"] = v4_est
    artifact["ui_priorities_v1"] = ui_priorities
    artifact["product_policy_version"] = PRODUCT_POLICY_VERSION
    artifact["catalog_version"] = str(issue_catalog.get("version") or "")
    artifact["product_projection_status"] = "reprojected"
    artifact["product_reprojection"] = {
        "reprojected_at": datetime.now().isoformat(),
        "quarantined_display_issues": result["quarantined_display_issues"],
        "quarantined_estimate_issues": result["quarantined_estimate_issues"],
        "verifications_reused": len(verifications),
        "tainted_packages": tainted,
        "old_total": old_total,
        "new_total": new_total,
    }

    try:
        backup_path = _write_with_backup(artifact_path, artifact)
    except OSError as e:
        return {"status": "error", "reason": f"write failed: {e}"}
    result["backup_path"] = str(backup_path)

    # Keep the standalone summary file in sync when present.
    summary_path = artifact_path.parent / "property_summary.json"
    if summary_path.is_file() and summary_v1 is not None:
        try:
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary_v1, f, indent=2, ensure_ascii=False)
        except OSError as e:
            result["summary_write_error"] = str(e)

    return result


def _iter_artifact_paths(root: Path) -> List[Path]:
    return sorted(root.rglob("photo_intel.json"))


def _fmt_range(total: Dict[str, int]) -> str:
    return f"${total.get('low', 0):,}-${total.get('high', 0):,}"


def _format_result(path: Path, result: Dict[str, Any]) -> str:
    status = result.get("status")
    if status in ("reprojected", "dry_run"):
        tainted = result.get("tainted_packages") or []
        tainted_note = (
            " tainted=" + ",".join(t["package_id"] for t in tainted)
            if tainted else ""
        )
        return (
            f"{status}: {path} | quarantined_issues="
            f"{result['quarantined_display_issues']}d/"
            f"{result['quarantined_estimate_issues']}e "
            f"verifications_reused={result['verifications_reused']}"
            f"{tainted_note} "
            f"total={_fmt_range(result['old_total'])} -> {_fmt_range(result['new_total'])}"
        )
    if status == "skip_current":
        return f"skip_current: {path} ({result.get('projection_status')})"
    if status in ("needs_reanalysis", "dry_run_needs_reanalysis"):
        return f"{status}: {path} | no usable issue lanes; product fields nulled"
    return f"error: {path} - {result.get('reason', 'unknown')}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproject product views of saved photo_intel artifacts "
                    "under the current product-quarantine policy.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", help="Path to photo_intel.json or its run directory.")
    target.add_argument("--all", metavar="ROOT",
                        help="Reproject every photo_intel.json under ROOT.")
    parser.add_argument("--catalog",
                        help="Issue catalog path. Defaults to pipeline_config.ISSUE_CATALOG_PATH.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    parser.add_argument("--force", action="store_true",
                        help="Reproject even artifacts already at the current policy version.")
    args = parser.parse_args(argv)

    catalog_path = Path(args.catalog) if args.catalog else Path(cfg.ISSUE_CATALOG_PATH)
    if not catalog_path.is_file():
        print(f"error: catalog not found: {catalog_path}", file=sys.stderr)
        return 1
    issue_catalog = load_issue_catalog(catalog_path)

    if args.run:
        run_path = Path(args.run)
        artifact_path = run_path if run_path.is_file() else run_path / "photo_intel.json"
        if not artifact_path.is_file():
            print(f"error: artifact not found: {artifact_path}", file=sys.stderr)
            return 1
        paths = [artifact_path]
    else:
        root = Path(args.all)
        if not root.is_dir():
            print(f"error: root not found: {root}", file=sys.stderr)
            return 1
        paths = _iter_artifact_paths(root)
        if not paths:
            print(f"error: no photo_intel.json found under {root}", file=sys.stderr)
            return 1

    exit_code = 0
    for path in paths:
        result = reproject_artifact(
            path, issue_catalog, dry_run=args.dry_run, force=args.force,
        )
        line = _format_result(path, result)
        if result.get("status") == "error":
            print(line, file=sys.stderr)
            exit_code = 1
        else:
            print(line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
