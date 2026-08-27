"""Offline replay of the deterministic v5 estimate chain (Session 8).

Replays stored canary artifacts through the CURRENT working-tree code and
catalog with zero provider calls: stored Terra verdicts and Sol decisions are
reused verbatim, while dispositions, work items, package candidates, and
reconciliation are recomputed deterministically. The stored estimate_id is
held fixed so every derived ID stays joinable to the stored Sol decisions.

Dispositions are recomputed from stored VERDICTS via decide_disposition —
never reused — because stored disposition records embed the terminal routes
of the code that produced them. Stored Sol decisions are mapped onto rebuilt
candidates by package_candidate_id; candidates the stored run never judged
are dropped (Sol truth is never fabricated) and every such sanitization is
counted in the report.

This module imports only the deterministic renovation-architecture modules,
so it is structurally incapable of spending provider tokens.

Usage:
  .venv\\Scripts\\python.exe scripts\\replay_renovation_architecture.py \
      --root artifacts_canary\\renovation_session6_20260816_02\\run_1 \
      --label baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.renovation_architecture.catalog_projection import (  # noqa: E402
    build_renovation_catalog_projection,
)
from tools.renovation_architecture.contracts import (  # noqa: E402
    CONDITION_DISPOSITION_POLICY_VERSION,
    CONTRACTS_SCHEMA_VERSION,
)
from tools.renovation_architecture.disposition import decide_disposition  # noqa: E402
from tools.renovation_architecture.ids import make_disposition_id  # noqa: E402
from tools.renovation_architecture.package_candidates import (  # noqa: E402
    build_package_candidates,
)
from tools.renovation_architecture.reconciliation import (  # noqa: E402
    build_complete_result,
)
from tools.renovation_architecture.validators import (  # noqa: E402
    package_review_snapshot_hashes,
    validate_package_review_result,
)
from tools.renovation_architecture.work_items import (  # noqa: E402
    derive_standalone_estimate,
)

CATALOG_PATH = REPO_ROOT / "tools" / "issue_catalog_kind_v2.json"
ENVELOPE_KEY = "renovation_estimate_v5"
_REVIEW_KEYS = (
    "observed_conditions", "evidence_facts", "condition_reviews",
    "terra_calls", "terra_unit_usage", "terra_listing_usage",
)
_SOL_TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
)
_SANITIZATION_KEYS = (
    "stored_decisions_unused", "candidates_dropped_no_stored_decision",
    "split_degraded_to_no_split", "combine_edges_filtered",
    "sol_call_candidate_ids_rewritten", "sol_calls_dropped",
    "disposition_diffs_vs_stored",
)


def _money(obj: Optional[Mapping[str, Any]]) -> Tuple[int, int]:
    if not isinstance(obj, Mapping):
        return (0, 0)
    return (int(obj.get("low") or 0), int(obj.get("high") or 0))


def _newest_run_dir(candidate_dir: Path) -> Optional[Path]:
    runs = sorted(
        run for run in candidate_dir.iterdir()
        if run.is_dir() and (run / "photo_intel_debug.json").is_file()
    )
    return runs[-1] if runs else None


def _rebuild_dispositions(
    stored: Mapping[str, Any],
    projection: Mapping[str, Any],
    *,
    estimate_id: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Recompute every disposition from stored verdict x current route x
    stored evidence. Returns (records, count differing from stored)."""
    reviews = {r["condition_id"]: r for r in stored["condition_reviews"]}
    evidence = {e["condition_id"]: e for e in stored["evidence_facts"]}
    stored_by_condition = {
        d["condition_id"]: d for d in stored["condition_dispositions"]
    }
    routes = projection["terminal_routes"]
    records: List[Dict[str, Any]] = []
    diffs = 0
    for condition in sorted(
        stored["observed_conditions"], key=lambda c: c["condition_id"]
    ):
        condition_id = condition["condition_id"]
        review = reviews[condition_id]
        fact = evidence[condition_id]
        route = routes[condition["catalog_item_id"]]["route"]
        disposition, reason_code = decide_disposition(
            review["verdict"],
            route,
            fact["distinct_view_count"],
            fact["min_photo_evidence_required"],
        )
        record = {
            "disposition_id": make_disposition_id(
                estimate_id=estimate_id, condition_id=condition_id
            ),
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "condition_id": condition_id,
            "review_id": review["review_id"],
            "evidence_id": fact["evidence_id"],
            "disposition": disposition,
            "reason_code": reason_code,
            "terminal_route": route,
            "policy_version": CONDITION_DISPOSITION_POLICY_VERSION,
        }
        previous = stored_by_condition.get(condition_id)
        if previous is None or (
            previous["disposition"], previous["reason_code"],
            previous["terminal_route"],
        ) != (disposition, reason_code, route):
            diffs += 1
        records.append(record)
    return records, diffs


def _map_stored_decisions(
    candidates: List[Dict[str, Any]],
    stored: Mapping[str, Any],
    counters: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Join stored Sol decisions onto rebuilt candidates by candidate ID.

    Candidates the stored run never judged are dropped (never fabricated);
    stored split/combine references to vanished work items or candidates are
    filtered so the result satisfies the Session 4 gate. Every sanitization
    increments a counter."""
    stored_by_candidate = {
        d["package_candidate_id"]: d for d in stored["package_decisions"]
    }
    kept: List[Dict[str, Any]] = []
    for candidate in candidates:
        if candidate["package_candidate_id"] in stored_by_candidate:
            kept.append(candidate)
        else:
            counters["candidates_dropped_no_stored_decision"] += 1
    kept_ids = {c["package_candidate_id"] for c in kept}
    children_by_candidate = {
        c["package_candidate_id"]: set(c["child_work_item_ids"]) for c in kept
    }
    counters["stored_decisions_unused"] += len(
        set(stored_by_candidate) - kept_ids
    )

    decisions: List[Dict[str, Any]] = []
    for candidate in kept:
        candidate_id = candidate["package_candidate_id"]
        decision = dict(stored_by_candidate[candidate_id])
        children = children_by_candidate[candidate_id]
        if decision["split_groups"]:
            groups = [
                [m for m in group if m in children]
                for group in decision["split_groups"]
            ]
            groups = [group for group in groups if group]
            covered = {m for group in groups for m in group}
            if len(groups) >= 2 and covered == children:
                decision["split_groups"] = groups
            else:
                decision["split_groups"] = []
                counters["split_degraded_to_no_split"] += 1
        if decision["combine_with"]:
            edges = [c for c in decision["combine_with"] if c in kept_ids]
            counters["combine_edges_filtered"] += (
                len(decision["combine_with"]) - len(edges)
            )
            decision["combine_with"] = edges
        decisions.append(decision)

    calls: List[Dict[str, Any]] = []
    for stored_call in stored["sol_calls"]:
        if not kept:
            counters["sol_calls_dropped"] += 1
            continue
        call = dict(stored_call)
        expected_ids = sorted(kept_ids)
        if list(call["package_candidate_ids"]) != expected_ids:
            call["package_candidate_ids"] = expected_ids
            counters["sol_call_candidate_ids_rewritten"] += 1
        calls.append(call)
    return kept, decisions, calls


def _assemble_package_review_result(
    standalone_result: Mapping[str, Any],
    candidates: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    sol_calls: List[Dict[str, Any]],
    *,
    estimate_id: str,
) -> Dict[str, Any]:
    """Mirror sol_review._assemble_result without importing the Sol client
    module: listing usage summed from the kept calls, snapshot fingerprints
    recomputed over the rebuilt sections, then the Session 4 gate."""
    listing_usage = {
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "call_count": len(sol_calls),
        **{
            name: sum(call[name] for call in sol_calls)
            for name in _SOL_TOKEN_FIELDS
        },
    }
    snapshots = package_review_snapshot_hashes(
        {**standalone_result, "package_candidates": candidates}
    )
    result: Dict[str, Any] = {
        **standalone_result,
        "package_candidates": candidates,
        "package_decisions": sorted(
            decisions, key=lambda decision: decision["decision_id"]
        ),
        "sol_calls": sol_calls,
        "sol_listing_usage": listing_usage,
        "package_review_snapshots": {
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            **snapshots,
        },
    }
    validation = validate_package_review_result(result, estimate_id=estimate_id)
    if not validation.ok:
        raise ValueError(
            "replayed package-review result failed the Session 4 gate: "
            + "; ".join(validation.errors[:5])
        )
    return result


def replay_property(
    run_dir: Path,
    projection: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> Dict[str, Any]:
    debug = json.loads(
        (run_dir / "photo_intel_debug.json").read_text(encoding="utf-8")
    )
    envelope = debug["analysis_debug"][ENVELOPE_KEY]
    if envelope["state"] != "complete":
        return {"state": "skipped", "reason": f"envelope state {envelope['state']!r}"}
    stored = envelope["result"]
    estimate_id = envelope["estimate_id"]
    counters = {key: 0 for key in _SANITIZATION_KEYS}

    dispositions, disposition_diffs = _rebuild_dispositions(
        stored, projection, estimate_id=estimate_id
    )
    counters["disposition_diffs_vs_stored"] = disposition_diffs
    review_result = {
        **{key: stored[key] for key in _REVIEW_KEYS},
        "condition_dispositions": dispositions,
    }
    standalone_result = derive_standalone_estimate(
        review_result=review_result,
        projection=projection,
        property_metadata=debug.get("property_metadata"),
        estimate_id=estimate_id,
    )
    candidates = build_package_candidates(
        standalone_result=standalone_result,
        projection=projection,
        catalog=catalog,
        estimate_id=estimate_id,
    )
    kept, decisions, sol_calls = _map_stored_decisions(
        candidates, stored, counters
    )
    package_review_result = _assemble_package_review_result(
        standalone_result, kept, decisions, sol_calls, estimate_id=estimate_id
    )
    complete = build_complete_result(package_review_result, estimate_id=estimate_id)

    v4_low = v4_high = 0
    intel_path = run_dir / "photo_intel.json"
    if intel_path.is_file():
        intel = json.loads(intel_path.read_text(encoding="utf-8"))
        v4_low, v4_high = _money(
            (intel.get("renovation_estimate_v4") or {}).get("final_rehab")
        )

    return {
        "state": "complete",
        "estimate_id": estimate_id,
        "sanitization": counters,
        "stored_factor": stored["standalone_estimate"]["property_cost_factor"],
        "replayed_factor": complete["standalone_estimate"]["property_cost_factor"],
        "v4_final_rehab": {"low": v4_low, "high": v4_high},
        "stored_totals": stored["totals"],
        "replayed_totals": complete["totals"],
        "stored_standalone_headline": stored["standalone_estimate"]["headline"],
        "replayed_standalone_headline": complete["standalone_estimate"]["headline"],
        "funnel": complete["observability"]["funnel"],
        "result": complete,
    }


def _fmt_delta(new: Tuple[int, int], old: Tuple[int, int]) -> str:
    return f"{new[0] - old[0]:+,} / {new[1] - old[1]:+,}"


def write_report(
    out_dir: Path, label: str, rows: List[Dict[str, Any]]
) -> Path:
    lines = [
        f"# Replay report — label `{label}`",
        "",
        "All dollars are headline low/high. `replayed` is the current "
        "working-tree code and catalog over stored replica-1 verdicts and "
        "Sol decisions.",
        "",
        "| property | v4 | stored v5 | replayed v5 | replayed − stored | replayed − v4 |",
        "|---|---|---|---|---|---|",
    ]
    sums = {name: [0, 0] for name in ("v4", "stored", "replayed")}
    sanitization = {key: 0 for key in _SANITIZATION_KEYS}
    failures: List[str] = []
    for row in rows:
        if row["state"] != "complete":
            failures.append(f"{row['property_key']}: {row.get('reason')}")
            continue
        v4 = _money(row["v4_final_rehab"])
        stored = _money(row["stored_totals"]["headline"])
        replayed = _money(row["replayed_totals"]["headline"])
        for name, value in (("v4", v4), ("stored", stored), ("replayed", replayed)):
            sums[name][0] += value[0]
            sums[name][1] += value[1]
        for key in _SANITIZATION_KEYS:
            sanitization[key] += row["sanitization"][key]
        lines.append(
            f"| {row['property_key']} | {v4[0]:,} / {v4[1]:,} "
            f"| {stored[0]:,} / {stored[1]:,} "
            f"| {replayed[0]:,} / {replayed[1]:,} "
            f"| {_fmt_delta(replayed, stored)} | {_fmt_delta(replayed, v4)} |"
        )
    v4_sum, stored_sum, replayed_sum = (
        tuple(sums["v4"]), tuple(sums["stored"]), tuple(sums["replayed"])
    )
    lines.append(
        f"| **corpus** | {v4_sum[0]:,} / {v4_sum[1]:,} "
        f"| {stored_sum[0]:,} / {stored_sum[1]:,} "
        f"| {replayed_sum[0]:,} / {replayed_sum[1]:,} "
        f"| {_fmt_delta(replayed_sum, stored_sum)} "
        f"| {_fmt_delta(replayed_sum, v4_sum)} |"
    )
    if v4_sum != (0, 0):
        lines += [
            "",
            f"Corpus replayed-vs-v4 residual: "
            f"{(replayed_sum[0] - v4_sum[0]) / v4_sum[0]:+.1%} low / "
            f"{(replayed_sum[1] - v4_sum[1]) / v4_sum[1]:+.1%} high.",
        ]
    lines += ["", "## Sanitization counters (corpus)", ""]
    lines += [f"- {key}: {value}" for key, value in sanitization.items()]
    if failures:
        lines += ["", "## Skipped/failed properties", ""]
        lines += [f"- {failure}" for failure in failures]
    lines.append("")
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("artifacts_canary")
        / "renovation_session6_20260816_02" / "run_1",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--properties", nargs="*", default=None)
    args = parser.parse_args()

    candidate_root = args.root / "candidate"
    if not candidate_root.is_dir():
        print(f"no candidate directory under {args.root}", file=sys.stderr)
        return 2
    out_dir = args.out or (
        args.root.parent / "analysis_session8" / args.label
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    projection = build_renovation_catalog_projection(
        catalog, catalog_path=CATALOG_PATH
    )

    keys = args.properties or sorted(
        entry.name for entry in candidate_root.iterdir() if entry.is_dir()
    )
    rows: List[Dict[str, Any]] = []
    for property_key in keys:
        run_dir = _newest_run_dir(candidate_root / property_key)
        if run_dir is None:
            rows.append({
                "property_key": property_key, "state": "skipped",
                "reason": "no run dir with photo_intel_debug.json",
            })
            continue
        try:
            row = replay_property(run_dir, projection, catalog)
        except Exception as exc:  # surface, never silently drop a property
            rows.append({
                "property_key": property_key, "state": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            print(f"FAILED {property_key}: {exc}", file=sys.stderr)
            continue
        row["property_key"] = property_key
        rows.append(row)
        full = dict(row)
        result = full.pop("result")
        (out_dir / f"{property_key}.replay.json").write_text(
            json.dumps({**full, "result": result}, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        headline = _money(row["replayed_totals"]["headline"])
        print(
            f"{property_key}: replayed headline {headline[0]:,}/{headline[1]:,}"
            f" (dispositions changed: {row['sanitization']['disposition_diffs_vs_stored']})"
        )

    report_path = write_report(out_dir, args.label, rows)
    print(f"report: {report_path}")
    failed = [row for row in rows if row["state"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
