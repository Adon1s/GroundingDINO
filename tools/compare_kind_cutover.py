"""Compare legacy and v2 canary artifacts and enforce ontology cutover gates."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifact_writers import load_issue_catalog

CURRENT_KINDS = frozenset({"defect", "degradation", "modernization"})


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _load_artifacts(root: Path) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
    paths = [root] if root.is_file() else sorted(root.rglob("photo_intel.json"))
    result: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
    for path in paths:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(artifact, dict):
            continue
        property_key = str(
            (artifact.get("property") or {}).get("property_key")
            or artifact.get("property_key")
            or path.parent.parent.name
        )
        created = str((artifact.get("run") or {}).get("created_at") or artifact.get("created_at") or "")
        previous = result.get(property_key)
        if previous is None:
            result[property_key] = (path, artifact)
        else:
            previous_created = str((previous[1].get("run") or {}).get("created_at") or previous[1].get("created_at") or "")
            if created >= previous_created:
                result[property_key] = (path, artifact)
    return result


def _issues(artifact: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for lane in ("product_issues_flat", "issues_flat"):
        value = artifact.get(lane)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _estimate_issues(artifact: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for lane in ("product_estimate_issues_flat", "estimate_issues_flat"):
        value = artifact.get(lane)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _kind(issue: Mapping[str, Any]) -> str:
    return str(
        issue.get("canonical_kind")
        or issue.get("catalog_item_kind")
        or issue.get("kind")
        or ""
    ).strip().lower()


def _catalog_id(issue: Mapping[str, Any]) -> str:
    return str(issue.get("catalog_item_id") or issue.get("catalogItemId") or "")


def _headline(artifact: Mapping[str, Any]) -> Dict[str, int]:
    estimate = artifact.get("renovation_estimate_v4")
    if not isinstance(estimate, dict):
        estimate = artifact.get("renovation_estimate")
    if not isinstance(estimate, dict):
        return {"low": 0, "high": 0}
    for key in ("final_rehab", "raw_totals", "primary_estimate", "total_range"):
        value = estimate.get(key)
        if isinstance(value, dict):
            return {"low": int(value.get("low") or 0), "high": int(value.get("high") or 0)}
    return {"low": 0, "high": 0}


def _packages(artifact: Mapping[str, Any]) -> List[str]:
    estimate = artifact.get("renovation_estimate_v4")
    if not isinstance(estimate, dict):
        return []
    return sorted({
        str(row.get("package_id") or row.get("package_type") or "")
        for row in (estimate.get("packages") or [])
        if isinstance(row, dict) and (row.get("package_id") or row.get("package_type"))
    })


def _summary_digest(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    summary = artifact.get("summary_v1")
    if not isinstance(summary, dict):
        return {}
    return {
        "version": summary.get("version"),
        "kind_counts": summary.get("kind_counts"),
        "bucket_count": len(summary.get("buckets") or []),
        "listing_kind_counts": (summary.get("listing") or {}).get("kind_counts")
        if isinstance(summary.get("listing"), dict) else None,
    }


def _pct_delta(old: int, new: int) -> Optional[float]:
    if old == 0:
        return 0.0 if new == 0 else None
    return (new - old) / old


def _distribution(rows: Iterable[Mapping[str, Any]], value_fn) -> Dict[str, int]:
    return dict(sorted(Counter(value_fn(row) or "<missing>" for row in rows).items()))


def compare_property(
    property_key: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    valid_catalog_ids: frozenset[str],
) -> Dict[str, Any]:
    old_issues = _issues(baseline)
    new_issues = _issues(candidate)
    old_estimate = _estimate_issues(baseline)
    new_estimate = _estimate_issues(candidate)
    old_by_description = {_norm(row.get("description")): _kind(row) for row in old_issues if _norm(row.get("description"))}
    new_by_description = {_norm(row.get("description")): _kind(row) for row in new_issues if _norm(row.get("description"))}
    shared = sorted(set(old_by_description) & set(new_by_description))
    classification_changes = [
        {"description": text, "baseline": old_by_description[text], "candidate": new_by_description[text]}
        for text in shared if old_by_description[text] != new_by_description[text]
    ]
    old_unresolved = sum(not _catalog_id(row) for row in old_issues)
    new_unresolved = sum(not _catalog_id(row) for row in new_issues)
    new_ids = {_catalog_id(row) for row in new_issues if _catalog_id(row)}
    stale_ids = sorted(new_ids - valid_catalog_ids)
    stale_kinds = sorted({_kind(row) for row in new_issues if _kind(row) not in CURRENT_KINDS})
    old_headline = _headline(baseline)
    new_headline = _headline(candidate)
    return {
        "property_key": property_key,
        "pass_2c_classification": {
            "shared_descriptions": len(shared),
            "changes": classification_changes,
        },
        "unresolved": {
            "baseline_count": old_unresolved,
            "candidate_count": new_unresolved,
            "baseline_rate": old_unresolved / max(1, len(old_issues)),
            "candidate_rate": new_unresolved / max(1, len(new_issues)),
        },
        "catalog_id_distribution": {
            "baseline": _distribution(old_issues, _catalog_id),
            "candidate": _distribution(new_issues, _catalog_id),
        },
        "kind_distribution": {
            "baseline": _distribution(old_issues, _kind),
            "candidate": _distribution(new_issues, _kind),
        },
        "estimate_coverage": {
            "baseline_issue_count": len(old_estimate),
            "candidate_issue_count": len(new_estimate),
            "delta_rate": (len(new_estimate) - len(old_estimate)) / max(1, len(old_estimate)),
        },
        "final_rehab": {
            "baseline": old_headline,
            "candidate": new_headline,
            "low_delta_pct": _pct_delta(old_headline["low"], new_headline["low"]),
            "high_delta_pct": _pct_delta(old_headline["high"], new_headline["high"]),
        },
        "package_selection": {
            "baseline": _packages(baseline),
            "candidate": _packages(candidate),
        },
        "display_summary": {
            "baseline": _summary_digest(baseline),
            "candidate": _summary_digest(candidate),
        },
        "stale_kinds": stale_kinds,
        "unresolved_successor_ids": stale_ids,
    }


def evaluate_gates(rows: List[Dict[str, Any]], config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    thresholds = config.get("thresholds") if isinstance(config.get("thresholds"), dict) else {}
    approvals = config.get("approved_headline_deltas") if isinstance(config.get("approved_headline_deltas"), dict) else {}
    failures: List[Dict[str, Any]] = []
    minimum = int(config.get("minimum_canary_properties") or 1)
    if len(rows) < minimum:
        failures.append({"gate": "representative_property_count", "actual": len(rows), "minimum": minimum})
    unresolved_limit = float(thresholds.get("max_unresolved_rate_increase", 0.02))
    coverage_drop_limit = float(thresholds.get("max_estimate_coverage_drop", 0.05))
    headline_limit = float(thresholds.get("max_unapproved_headline_delta", 0.15))
    for row in rows:
        key = row["property_key"]
        if row["stale_kinds"]:
            failures.append({"gate": "no_stale_kinds", "property_key": key, "values": row["stale_kinds"]})
        if row["unresolved_successor_ids"]:
            failures.append({"gate": "no_unresolved_successor_ids", "property_key": key, "values": row["unresolved_successor_ids"]})
        unresolved = row["unresolved"]
        if unresolved["candidate_rate"] - unresolved["baseline_rate"] > unresolved_limit:
            failures.append({"gate": "unresolved_rate", "property_key": key, "detail": unresolved})
        coverage_delta = float(row["estimate_coverage"]["delta_rate"])
        if coverage_delta < -coverage_drop_limit:
            failures.append({"gate": "estimate_coverage", "property_key": key, "detail": row["estimate_coverage"]})
        if key not in approvals:
            for bound in ("low_delta_pct", "high_delta_pct"):
                delta = row["final_rehab"][bound]
                if delta is None or abs(float(delta)) > headline_limit:
                    failures.append({"gate": "unapproved_headline_delta", "property_key": key, "bound": bound, "detail": row["final_rehab"]})
    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "kind_ontology_cutover.json")
    parser.add_argument("--catalog", type=Path, default=ROOT / "tools" / "issue_catalog_kind_v2.json")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    catalog = load_issue_catalog(args.catalog)
    valid_ids = frozenset(str(row["id"]) for row in catalog.get("items") or [] if isinstance(row, dict) and row.get("id"))
    baseline = _load_artifacts(args.baseline)
    candidate = _load_artifacts(args.candidate)
    property_keys = sorted(set(baseline) & set(candidate))
    rows = [compare_property(key, baseline[key][1], candidate[key][1], valid_catalog_ids=valid_ids) for key in property_keys]
    failures = evaluate_gates(rows, config)
    report = {
        "gate_status": "pass" if not failures else "fail",
        "compared_properties": len(rows),
        "missing_from_candidate": sorted(set(baseline) - set(candidate)),
        "missing_from_baseline": sorted(set(candidate) - set(baseline)),
        "failures": failures,
        "properties": rows,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())