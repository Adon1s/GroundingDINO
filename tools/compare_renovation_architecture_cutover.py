"""Compare two frozen Session 6 shadow replicas and enforce cutover gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.renovation_architecture.contracts import SHADOW_DEBUG_KEY
from tools.renovation_architecture.validators import validate_envelope


class CanaryComparisonError(RuntimeError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any, prefix: str = "") -> str:
    return prefix + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:16]


def _full_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load_latest_artifacts(root: Path) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
    result: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
    for path in sorted(root.rglob("photo_intel_debug.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(artifact, dict):
            continue
        key = str(
            ((artifact.get("property") or {}).get("property_key"))
            or artifact.get("property_key")
            or path.parent.parent.name
        )
        created = str((artifact.get("run") or {}).get("created_at") or "")
        previous = result.get(key)
        if previous is None or created >= str(
            (previous[1].get("run") or {}).get("created_at") or ""
        ):
            result[key] = (path, artifact)
    return result


def _money(value: Any) -> Dict[str, int]:
    row = value if isinstance(value, Mapping) else {}
    return {"low": int(row.get("low") or 0), "high": int(row.get("high") or 0)}


def _pct_delta(old: int, new: int) -> Optional[float]:
    if old == 0:
        return 0.0 if new == 0 else None
    return (new - old) / old


def _v4_headline(artifact: Mapping[str, Any]) -> Dict[str, int]:
    estimate = artifact.get("renovation_estimate_v4")
    if not isinstance(estimate, Mapping):
        raise CanaryComparisonError("artifact has no renovation_estimate_v4 object")
    return _money(estimate.get("final_rehab"))


def _v4_scope(artifact: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    estimate = artifact.get("renovation_estimate_v4") or {}
    package_types = {
        str(row.get("package_id")): str(row.get("package_type") or row.get("package_id") or "")
        for lane in ("packages", "package_candidates")
        for row in estimate.get(lane) or []
        if isinstance(row, Mapping) and row.get("package_id")
    }
    rows: Dict[str, List[Dict[str, Any]]] = {}
    for group in estimate.get("groups") or []:
        if not isinstance(group, Mapping):
            continue
        for item in group.get("line_items") or []:
            if not isinstance(item, Mapping):
                continue
            allocations = item.get("unit_member_allocations") or [item]
            for allocation in allocations:
                if not isinstance(allocation, Mapping):
                    continue
                catalog_id = str(
                    allocation.get("catalog_item_id")
                    or item.get("catalog_item_id")
                    or ""
                )
                unit_id = str(
                    allocation.get("estimate_unit_id")
                    or item.get("billable_estimate_unit_id")
                    or item.get("estimate_unit_id")
                    or ""
                )
                if not catalog_id:
                    continue
                key = f"{catalog_id}|{unit_id}"
                package_id = allocation.get("absorbed_by_package_id") or item.get(
                    "package_id"
                )
                package_type = package_types.get(str(package_id), str(package_id or ""))
                rows.setdefault(key, []).append(
                    {
                        "catalog_item_id": catalog_id,
                        "billable_unit_id": unit_id,
                        "estimate_scope": allocation.get("estimate_scope")
                        or item.get("estimate_scope"),
                        "low": int(
                            allocation.get("allocated_low")
                            if allocation.get("allocated_low") is not None
                            else item.get("cost_low")
                            or 0
                        ),
                        "high": int(
                            allocation.get("allocated_high")
                            if allocation.get("allocated_high") is not None
                            else item.get("cost_high")
                            or 0
                        ),
                        "representation": (
                            f"absorbed_by:{package_type}" if package_id else "standalone"
                        ),
                    }
                )
    return {key: sorted(value, key=_canonical) for key, value in sorted(rows.items())}


def _v5_scope(result: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    ledger = {
        str(row.get("work_item_id")): row
        for row in result.get("coverage_ledger") or []
        if isinstance(row, Mapping)
    }
    candidates = {
        str(row.get("package_candidate_id")): row
        for row in result.get("package_candidates") or []
        if isinstance(row, Mapping)
    }
    rows: Dict[str, List[Dict[str, Any]]] = {}
    for work in result.get("work_items") or []:
        if not isinstance(work, Mapping) or work.get("status") != "active":
            continue
        entry = ledger.get(str(work.get("work_item_id")), {})
        package_id = str(entry.get("package_id") or "")
        package_type = str((candidates.get(package_id) or {}).get("package_type") or package_id)
        representation = str(entry.get("representation") or "")
        if representation == "absorbed_by":
            representation = f"absorbed_by:{package_type}"
        for catalog_id in work.get("catalog_item_ids") or []:
            key = f"{catalog_id}|{work.get('billable_unit_id') or ''}"
            rows.setdefault(key, []).append(
                {
                    "catalog_item_id": catalog_id,
                    "billable_unit_id": work.get("billable_unit_id"),
                    "action_code": work.get("action_code"),
                    "estimate_scope": work.get("estimate_scope"),
                    "low": int(work.get("low") or 0),
                    "high": int(work.get("high") or 0),
                    "representation": representation,
                    "merged_catalog_item_ids": list(work.get("catalog_item_ids") or []),
                }
            )
    return {key: sorted(value, key=_canonical) for key, value in sorted(rows.items())}


def _v4_packages(artifact: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    estimate = artifact.get("renovation_estimate_v4") or {}
    rows = {}
    for package in estimate.get("packages") or []:
        if not isinstance(package, Mapping):
            continue
        key = f"{package.get('package_type') or ''}|{package.get('estimate_unit_id') or ''}"
        rows[key] = {
            "package_type": package.get("package_type"),
            "estimate_unit_id": package.get("estimate_unit_id"),
            "pricing_tier": package.get("pricing_tier"),
            "status": "applied",
        }
    return dict(sorted(rows.items()))


def _v5_packages(result: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    decisions = {
        str(row.get("package_candidate_id")): row
        for row in result.get("package_decisions") or []
        if isinstance(row, Mapping)
    }
    applications = {
        str(row.get("package_candidate_id")): row
        for row in result.get("package_applications") or []
        if isinstance(row, Mapping)
    }
    rows = {}
    for candidate in result.get("package_candidates") or []:
        if not isinstance(candidate, Mapping) or candidate.get("display_only"):
            continue
        candidate_id = str(candidate.get("package_candidate_id") or "")
        key = f"{candidate.get('package_type') or ''}|{candidate.get('estimate_unit_id') or ''}"
        rows[key] = {
            "package_type": candidate.get("package_type"),
            "estimate_unit_id": candidate.get("estimate_unit_id"),
            "pricing_tier": candidate.get("pricing_tier"),
            "decision": (decisions.get(candidate_id) or {}).get("decision"),
            "status": (applications.get(candidate_id) or {}).get("status"),
            "reason_code": (applications.get(candidate_id) or {}).get("reason_code"),
        }
    return dict(sorted(rows.items()))


def _review_item(
    *, property_key: str, category: str, key: str, baseline: Any, candidate: Any
) -> Dict[str, Any]:
    body = {
        "property_key": property_key,
        "category": category,
        "key": key,
        "baseline": baseline,
        "candidate": candidate,
    }
    return {"review_id": _digest(body, "rar1_"), **body}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {key: 0 for key in ("median", "p90", "p95", "p99", "worst")}
    return {
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "worst": max(values),
    }


def _capacity(distribution: Mapping[str, float], ceiling: int) -> Dict[str, int]:
    return {
        key: math.floor(ceiling / value) if value > 0 else 0
        for key, value in distribution.items()
    }


def _v5_digest(result: Mapping[str, Any]) -> Dict[str, Any]:
    condition_keys = {
        str(row.get("condition_id")): {
            "catalog_item_id": row.get("catalog_item_id"),
            "estimate_unit_id": row.get("estimate_unit_id"),
        }
        for row in result.get("observed_conditions") or []
        if isinstance(row, Mapping)
    }
    return {
        "conditions": sorted(condition_keys.values(), key=_canonical),
        "dispositions": sorted(
            [
                {
                    **condition_keys.get(str(row.get("condition_id")), {}),
                    "disposition": row.get("disposition"),
                    "reason_code": row.get("reason_code"),
                }
                for row in result.get("condition_dispositions") or []
                if isinstance(row, Mapping)
            ],
            key=_canonical,
        ),
        "work": _v5_scope(result),
        "packages": _v5_packages(result),
        "totals": result.get("totals") or {},
    }


def _reviewed(item: Mapping[str, Any], reviews: Mapping[str, Any]) -> bool:
    review = reviews.get(str(item.get("review_id")))
    return (
        isinstance(review, Mapping)
        and review.get("decision") in ("approved", "accepted")
        and bool(str(review.get("explanation") or "").strip())
    )


def compare_canary(
    *,
    run_roots: Sequence[Path],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    freeze: Mapping[str, Any],
    reviews_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    expected = [str(row["property_key"]) for row in manifest.get("properties") or []]
    failures: List[Dict[str, Any]] = []
    required_replicates = int(config.get("required_replicates") or 2)
    if len(run_roots) != required_replicates:
        failures.append({"gate": "replicate_count", "actual": len(run_roots)})
    if manifest.get("status") != "frozen":
        failures.append({"gate": "manifest_frozen"})
    frozen_payload = dict(freeze)
    frozen_sha = frozen_payload.pop("freeze_sha256", None)
    if not frozen_sha or frozen_sha != _full_digest(frozen_payload):
        failures.append({"gate": "freeze_integrity"})
    minimum = int(config.get("minimum_canary_properties") or 18)
    if len(expected) < minimum:
        failures.append({"gate": "property_count", "actual": len(expected), "minimum": minimum})

    runs = [_load_latest_artifacts(Path(root)) for root in run_roots]
    envelopes: List[Dict[str, Dict[str, Any]]] = []
    for run_index, artifacts in enumerate(runs, start=1):
        missing = sorted(set(expected) - set(artifacts))
        extra = sorted(set(artifacts) - set(expected))
        if missing or extra:
            failures.append(
                {"gate": "artifact_set", "run": run_index, "missing": missing, "extra": extra}
            )
        run_envelopes: Dict[str, Dict[str, Any]] = {}
        for key in expected:
            if key not in artifacts:
                continue
            _, artifact = artifacts[key]
            private = (artifact.get("analysis_debug") or {}).get(SHADOW_DEBUG_KEY)
            validation = validate_envelope(private)
            if not validation.ok:
                failures.append(
                    {"gate": "schema_valid", "run": run_index, "property_key": key,
                     "errors": validation.errors[:5]}
                )
                continue
            if private.get("state") != "complete":
                failures.append(
                    {"gate": "complete_artifact", "run": run_index,
                     "property_key": key, "state": private.get("state"),
                     "reason": private.get("reason")}
                )
                continue
            if (private.get("provenance") or {}).get("architecture_mode") != "shadow":
                failures.append(
                    {"gate": "shadow_provenance", "run": run_index, "property_key": key}
                )
                continue
            result = private["result"]
            audit = result.get("reconciliation_audit") or {}
            dirty_audit = {
                name: value for name, value in audit.items()
                if name != "schema_version" and value
            }
            if dirty_audit:
                failures.append(
                    {"gate": "reconciliation", "run": run_index,
                     "property_key": key, "audit": dirty_audit}
                )
            accepted = {
                str(row.get("condition_id"))
                for row in result.get("condition_dispositions") or []
                if isinstance(row, Mapping)
                and row.get("disposition") == "accepted_for_work"
            }
            covered = {
                str(condition_id)
                for row in result.get("work_items") or []
                if isinstance(row, Mapping) and row.get("status") == "active"
                for condition_id in row.get("condition_ids") or []
            }
            if accepted - covered:
                failures.append(
                    {"gate": "accepted_condition_loss", "run": run_index,
                     "property_key": key,
                     "condition_ids": sorted(accepted - covered)}
                )
            run_envelopes[key] = private
        envelopes.append(run_envelopes)

    threshold = float(config.get("headline_delta_review_threshold") or 0.15)
    review_items: List[Dict[str, Any]] = []
    property_rows: List[Dict[str, Any]] = []
    if runs and envelopes:
        for key in expected:
            if key not in runs[0] or key not in envelopes[0]:
                continue
            artifact = runs[0][key][1]
            result = envelopes[0][key]["result"]
            try:
                v4_headline = _v4_headline(artifact)
            except CanaryComparisonError as exc:
                failures.append(
                    {"gate": "legacy_v4_available", "property_key": key,
                     "detail": str(exc)}
                )
                continue
            v5_headline = _money((result.get("totals") or {}).get("headline"))
            deltas = {
                "low": _pct_delta(v4_headline["low"], v5_headline["low"]),
                "high": _pct_delta(v4_headline["high"], v5_headline["high"]),
            }
            scope_v4, scope_v5 = _v4_scope(artifact), _v5_scope(result)
            scope_changes = []
            for scope_key in sorted(set(scope_v4) | set(scope_v5)):
                if scope_v4.get(scope_key) != scope_v5.get(scope_key):
                    item = _review_item(
                        property_key=key, category="scope", key=scope_key,
                        baseline=scope_v4.get(scope_key), candidate=scope_v5.get(scope_key),
                    )
                    scope_changes.append(item["review_id"])
                    review_items.append(item)
            packages_v4, packages_v5 = _v4_packages(artifact), _v5_packages(result)
            package_changes = []
            for package_key in sorted(set(packages_v4) | set(packages_v5)):
                if packages_v4.get(package_key) != packages_v5.get(package_key):
                    item = _review_item(
                        property_key=key, category="package", key=package_key,
                        baseline=packages_v4.get(package_key),
                        candidate=packages_v5.get(package_key),
                    )
                    package_changes.append(item["review_id"])
                    review_items.append(item)
            if any(delta is None or abs(delta) > threshold for delta in deltas.values()):
                review_items.append(
                    _review_item(
                        property_key=key, category="headline_delta", key="headline",
                        baseline=v4_headline,
                        candidate={"headline": v5_headline, "delta_pct": deltas},
                    )
                )
            dispositions = result.get("condition_dispositions") or []
            evidence = result.get("evidence_facts") or []
            property_rows.append(
                {
                    "property_key": key,
                    "v4_headline": v4_headline,
                    "v5_headline": v5_headline,
                    "headline_delta_pct": deltas,
                    "condition_counts": {
                        "observed": len(result.get("observed_conditions") or []),
                        "accepted": sum(row.get("disposition") == "accepted_for_work" for row in dispositions),
                        "unsupported": sum(row.get("disposition") == "excluded" for row in dispositions),
                        "cannot_assess_or_withheld": sum(
                            row.get("disposition") in ("inspection", "withheld")
                            for row in dispositions
                        ),
                    },
                    "duplicate_handling": {
                        "duplicate_group_count": sum(len(row.get("duplicate_groups") or []) for row in evidence),
                        "distinct_photos": sum(int(row.get("distinct_photo_count") or 0) for row in evidence),
                        "distinct_views": sum(int(row.get("distinct_view_count") or 0) for row in evidence),
                    },
                    "scope_review_ids": scope_changes,
                    "package_review_ids": package_changes,
                }
            )

    if len(envelopes) >= 2:
        for key in expected:
            if key not in envelopes[0] or key not in envelopes[1]:
                continue
            first = _v5_digest(envelopes[0][key]["result"])
            second = _v5_digest(envelopes[1][key]["result"])
            if first != second:
                review_items.append(
                    _review_item(
                        property_key=key, category="run_to_run_stability", key="v5",
                        baseline=first, candidate=second,
                    )
                )

    terra_listing: List[float] = []
    terra_calls: List[Dict[str, Any]] = []
    terra_units: List[Dict[str, Any]] = []
    condition_tokens: List[Dict[str, Any]] = []
    phase_values: Dict[str, List[float]] = {}
    budget_debited = 0
    for run_index, run in enumerate(envelopes, start=1):
        for key, envelope in sorted(run.items()):
            result = envelope["result"]
            listing = result.get("terra_listing_usage") or {}
            terra_listing.append(float(listing.get("total_tokens") or 0))
            budget_debited += int(listing.get("budget_debited_tokens") or 0)
            for call in result.get("terra_calls") or []:
                call_row = {"run": run_index, "property_key": key, **call}
                terra_calls.append(call_row)
                if call.get("usage_source") != "provider" or int(call.get("total_tokens") or 0) <= 0:
                    failures.append(
                        {"gate": "terra_usage_measured", "run": run_index,
                         "property_key": key, "call_id": call.get("call_id")}
                    )
                condition_ids = list(call.get("condition_ids") or [])
                allocation = (
                    float(call.get("total_tokens") or 0) / len(condition_ids)
                    if condition_ids else 0.0
                )
                for condition_id in condition_ids:
                    condition_tokens.append(
                        {"run": run_index, "property_key": key,
                         "condition_id": condition_id,
                         "estimate_unit_id": call.get("estimate_unit_id"),
                         "allocated_total_tokens": allocation}
                    )
            terra_units.extend(
                {"run": run_index, "property_key": key, **row}
                for row in result.get("terra_unit_usage") or []
            )
            for phase, value in ((result.get("observability") or {}).get("phase_timings_ms") or {}).items():
                phase_values.setdefault(str(phase), []).append(float(value or 0))

    ceiling = int(config.get("terra_daily_token_ceiling") or 2_500_000)
    if budget_debited > ceiling:
        failures.append(
            {"gate": "terra_daily_ceiling", "budget_debited_tokens": budget_debited,
             "ceiling": ceiling}
        )
    listing_distribution = _distribution(terra_listing)
    condition_distribution = _distribution(
        [row["allocated_total_tokens"] for row in condition_tokens]
    )

    reviews = (reviews_payload or {}).get("reviews") or {}
    # A review file authorizes exactly one freeze. An absent freeze_sha256 is
    # a mismatch too — an unbound review file could otherwise be carried over
    # from an earlier canary and approve deltas nobody looked at.
    if reviews_payload and reviews_payload.get("freeze_sha256") != freeze.get(
        "freeze_sha256"
    ):
        failures.append(
            {"gate": "review_freeze_mismatch",
             "expected": freeze.get("freeze_sha256"),
             "found": reviews_payload.get("freeze_sha256")}
        )
    unique_items = {item["review_id"]: item for item in review_items}
    unreviewed = [
        item for item in unique_items.values() if not _reviewed(item, reviews)
    ]
    if unreviewed:
        failures.append(
            {"gate": "manual_review", "unreviewed_count": len(unreviewed),
             "review_ids": [item["review_id"] for item in unreviewed]}
        )

    return {
        "schema_version": 1,
        "freeze_sha256": freeze.get("freeze_sha256"),
        "release_ready": not failures,
        "gates": {"passed": not failures, "failures": failures},
        "property_count": len(expected),
        "replicate_count": len(run_roots),
        "properties": property_rows,
        "review_items": list(unique_items.values()),
        "unreviewed_items": unreviewed,
        "terra_usage": {
            "daily_ceiling": ceiling,
            "canary_budget_debited_tokens": budget_debited,
            "listing_total_tokens": listing_distribution,
            "condition_allocated_tokens": condition_distribution,
            "projected_listings_per_day": _capacity(listing_distribution, ceiling),
            "call_rows": terra_calls,
            "estimate_unit_rows": terra_units,
            "condition_rows": condition_tokens,
            "condition_allocation_policy": config.get(
                "terra_condition_token_allocation_policy"
            ),
        },
        "phase_latency_ms": {
            phase: _distribution(values) for phase, values in sorted(phase_values.items())
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Renovation architecture Session 6 canary",
        "",
        f"Release ready: **{'YES' if report.get('release_ready') else 'NO'}**",
        f"Properties: {report.get('property_count')}; replicas: {report.get('replicate_count')}",
        f"Freeze: `{report.get('freeze_sha256')}`",
        "",
        "## Gates",
        "",
    ]
    failures = (report.get("gates") or {}).get("failures") or []
    if failures:
        lines.extend(f"- FAIL `{row.get('gate')}`: `{_canonical(row)}`" for row in failures)
    else:
        lines.append("- All automated and manual-review gates passed.")
    usage = report.get("terra_usage") or {}
    lines.extend(
        [
            "",
            "## Terra capacity",
            "",
            f"- Listing tokens: `{_canonical(usage.get('listing_total_tokens') or {})}`",
            f"- Projected listings/day: `{_canonical(usage.get('projected_listings_per_day') or {})}`",
            f"- Canary budget debit: `{usage.get('canary_budget_debited_tokens')}` / `{usage.get('daily_ceiling')}`",
            "",
            "## Manual review",
            "",
            f"- Review items: {len(report.get('review_items') or [])}",
            f"- Unreviewed: {len(report.get('unreviewed_items') or [])}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True,
                        help="replicate candidate root; pass exactly twice")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "configs" / "kind_ontology_canary_manifest.json")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "renovation_architecture_cutover.json")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review-template", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    reviews = (
        json.loads(args.reviews.read_text(encoding="utf-8"))
        if args.reviews and args.reviews.is_file() else None
    )
    report = compare_canary(
        run_roots=args.run,
        manifest=manifest,
        config=config,
        freeze=freeze,
        reviews_payload=reviews,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.report.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    if args.review_template:
        template = {
            "schema_version": 1,
            "freeze_sha256": report.get("freeze_sha256"),
            "reviews": {
                item["review_id"]: {"decision": "", "explanation": ""}
                for item in report.get("review_items") or []
            },
        }
        args.review_template.parent.mkdir(parents=True, exist_ok=True)
        args.review_template.write_text(
            json.dumps(template, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {"release_ready": report["release_ready"],
             "failures": report["gates"]["failures"],
             "report": str(args.report)},
            indent=2,
        )
    )
    return 0 if report["release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
