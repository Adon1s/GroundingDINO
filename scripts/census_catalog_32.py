"""Provider-free census of a Catalog 3.2 replay against a baseline replay.

Reads two `scripts/replay_renovation_architecture.py` output roots and reports
what contextual repair support actually did to the frozen 18-property canary:

  - candidates whose full Sol payload changed, and which child roles flipped
  - candidate / pricing-tier / allowance-range deltas
  - total and per-family package counts on both sides
  - bedroom/living single-child repair counts (the QP6 calibration input)

Zero provider calls and no imports from the Sol or Terra client modules — it
reads JSON off disk and reuses the same payload hash the replay join uses, so
"changed" here means exactly what "must be re-reviewed by Sol" means there.

Dollars are deliberately secondary. Replay cannot score a candidate the stored
run never judged, so a routing change shows up as dropped/mismatched counters
plus new candidates — those counts are the signal, not the headline totals.

Run:
  .venv\\Scripts\\python.exe scripts\\census_catalog_32.py \\
      --baseline reports_scratch\\wave1_qp3_final \\
      --candidate reports_scratch\\cat32_candidate \\
      --out reports_scratch\\cat32_census
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.renovation_architecture.candidate_payload import (  # noqa: E402
    candidate_payload_hashes,
)

_SANITIZATION_KEYS = (
    "stored_decisions_unused",
    "candidates_dropped_no_stored_decision",
    "stored_decisions_payload_mismatch",
    "split_degraded_to_no_split",
    "combine_edges_filtered",
    "sol_call_candidate_ids_rewritten",
    "sol_calls_dropped",
    "disposition_diffs_vs_stored",
)


def _load_side(root: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.glob("*.replay.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("state") != "complete":
            continue
        out[payload["property_key"]] = payload
    if not out:
        raise SystemExit(f"census_catalog_32: no complete replays under {root}")
    return out


def _candidates(replay: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The candidates as REBUILT, before stored-decision sanitization.

    result["package_candidates"] holds only the subset whose stored Sol
    decision survived the join, so a candidate whose payload changed has
    already been dropped from it. Reading that would report every changed
    candidate as "removed" and report zero payload changes — the exact
    opposite of the census's job. Older replay outputs predate the
    rebuilt_package_candidates key; they are only sound to read when the run
    dropped nothing, which _assert_baseline_is_lossless checks.
    """
    source = replay.get("rebuilt_package_candidates")
    if source is None:
        source = replay["result"]["package_candidates"]
    return {c["package_candidate_id"]: c for c in source}


def _assert_baseline_is_lossless(baseline: Mapping[str, Dict[str, Any]]) -> None:
    """A baseline without rebuilt_package_candidates is only usable if its
    replay kept everything it rebuilt."""
    lossy = []
    for key, replay in sorted(baseline.items()):
        if replay.get("rebuilt_package_candidates") is not None:
            continue
        dropped = sum(
            replay["sanitization"].get(counter, 0)
            for counter in ("candidates_dropped_no_stored_decision",
                            "stored_decisions_payload_mismatch")
        )
        if dropped:
            lossy.append(f"{key} (dropped {dropped})")
    if lossy:
        raise SystemExit(
            "census_catalog_32: the baseline predates rebuilt_package_candidates "
            "and dropped candidates, so its rebuilt set cannot be recovered: "
            + ", ".join(lossy)
        )


def _child_roles(candidate: Mapping[str, Any]) -> Dict[str, str]:
    drivers = set(candidate["driver_work_item_ids"])
    return {
        work_id: ("driver" if work_id in drivers else "support")
        for work_id in candidate["child_work_item_ids"]
    }


def _single_child_repairs(candidates: Mapping[str, Dict[str, Any]]) -> Counter:
    """Bedroom/living repair packages with exactly one child — the population
    QP6's single-child rule is scoped against."""
    counts: Counter = Counter()
    for candidate in candidates.values():
        package_type = candidate["package_type"]
        if package_type not in ("bedroom_repair", "living_repair"):
            continue
        if len(candidate["child_work_item_ids"]) == 1:
            counts[package_type] += 1
    return counts


def _family_counts(candidates: Mapping[str, Dict[str, Any]]) -> Counter:
    return Counter(c["package_type"] for c in candidates.values())


def compare_property(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Dict[str, Any]:
    base_c, cand_c = _candidates(baseline), _candidates(candidate)
    base_hashes = candidate_payload_hashes(
        baseline["result"], list(base_c.values())
    )
    cand_hashes = candidate_payload_hashes(
        candidate["result"], list(cand_c.values())
    )

    added = sorted(set(cand_c) - set(base_c))
    removed = sorted(set(base_c) - set(cand_c))
    shared = sorted(set(base_c) & set(cand_c))
    changed: List[Dict[str, Any]] = []
    for candidate_id in shared:
        if base_hashes[candidate_id] == cand_hashes[candidate_id]:
            continue
        before, after = base_c[candidate_id], cand_c[candidate_id]
        before_roles, after_roles = _child_roles(before), _child_roles(after)
        role_flips = sorted(
            work_id for work_id in set(before_roles) & set(after_roles)
            if before_roles[work_id] != after_roles[work_id]
        )
        changed.append({
            "package_candidate_id": candidate_id,
            "package_type": after["package_type"],
            "estimate_unit_id": after["estimate_unit_id"],
            "children_added": sorted(set(after_roles) - set(before_roles)),
            "children_removed": sorted(set(before_roles) - set(after_roles)),
            "child_role_flips": [
                {"work_item_id": w, "from": before_roles[w], "to": after_roles[w]}
                for w in role_flips
            ],
            "pricing_tier": _delta(before["pricing_tier"], after["pricing_tier"]),
            "strength": _delta(before["strength"], after["strength"]),
            "range_low": _delta(before["low"], after["low"]),
            "range_high": _delta(before["high"], after["high"]),
        })

    return {
        "property_key": candidate["property_key"],
        "candidate_count": {"baseline": len(base_c), "candidate": len(cand_c)},
        "added_candidates": [
            {"package_candidate_id": cid,
             "package_type": cand_c[cid]["package_type"],
             "estimate_unit_id": cand_c[cid]["estimate_unit_id"]}
            for cid in added
        ],
        "removed_candidates": [
            {"package_candidate_id": cid,
             "package_type": base_c[cid]["package_type"],
             "estimate_unit_id": base_c[cid]["estimate_unit_id"]}
            for cid in removed
        ],
        "changed_candidates": changed,
        "family_counts": {
            "baseline": dict(sorted(_family_counts(base_c).items())),
            "candidate": dict(sorted(_family_counts(cand_c).items())),
        },
        "single_child_repairs": {
            "baseline": dict(sorted(_single_child_repairs(base_c).items())),
            "candidate": dict(sorted(_single_child_repairs(cand_c).items())),
        },
        "headline": {
            "baseline": baseline["replayed_totals"],
            "candidate": candidate["replayed_totals"],
        },
        "sanitization": {
            key: candidate["sanitization"].get(key, 0)
            for key in _SANITIZATION_KEYS
        },
    }


def _delta(before: Any, after: Any) -> Optional[Dict[str, Any]]:
    return None if before == after else {"from": before, "to": after}


def _money(totals: Mapping[str, Any]) -> Tuple[int, int]:
    block = totals.get("final_rehab") or totals
    return int(block.get("low") or 0), int(block.get("high") or 0)


def render(rows: List[Dict[str, Any]]) -> str:
    touched = [r for r in rows if r["added_candidates"]
               or r["removed_candidates"] or r["changed_candidates"]]
    fam_base: Counter = Counter()
    fam_cand: Counter = Counter()
    single_base: Counter = Counter()
    single_cand: Counter = Counter()
    for row in rows:
        fam_base.update(row["family_counts"]["baseline"])
        fam_cand.update(row["family_counts"]["candidate"])
        single_base.update(row["single_child_repairs"]["baseline"])
        single_cand.update(row["single_child_repairs"]["candidate"])

    lines = [
        "# Catalog 3.2 canary census — contextual repair support",
        "",
        "Provider-free. Baseline and candidate are both offline replays over the",
        "frozen 18-property canary; no Sol or Terra call was made to produce this.",
        "",
        "## Scope of change",
        "",
        f"- Properties replayed: {len(rows)}",
        f"- Properties with any candidate change: **{len(touched)}**",
        f"- Candidates added: {sum(len(r['added_candidates']) for r in rows)}",
        f"- Candidates removed: {sum(len(r['removed_candidates']) for r in rows)}",
        f"- Candidates with a changed Sol payload: "
        f"**{sum(len(r['changed_candidates']) for r in rows)}**",
        "",
        "Counts are over the REBUILT candidate set, before stored-decision",
        "sanitization. Every added candidate and every changed payload needs a",
        "fresh Sol review: replay reuses a stored decision only on an exact",
        "payload hash match, and never fabricates one. The mismatch counter in",
        "the per-property table is what the replay dropped for that reason.",
        "",
        "Headline dollars are deliberately not reported. The replay totals are",
        "computed over the KEPT set, so they exclude every candidate whose",
        "payload changed — an undercount, not a comparable headline. The real",
        "dollar effect is only measurable after the fresh Sol reviews.",
        "",
        "One work item can legitimately appear under two families in the same",
        "unit: v5 collapses conditions into work items by action code, so a",
        "marked occurrence that moves and an unmarked one that stays can share",
        "a parent work item. The occurrence itself is never in both.",
        "",
        "## Package counts by family",
        "",
        "| family | baseline | candidate | delta |",
        "|---|---:|---:|---:|",
    ]
    for family in sorted(set(fam_base) | set(fam_cand)):
        base, cand = fam_base[family], fam_cand[family]
        lines.append(f"| {family} | {base} | {cand} | {cand - base:+d} |")
    lines += [
        f"| **total** | **{sum(fam_base.values())}** | "
        f"**{sum(fam_cand.values())}** | {sum(fam_cand.values()) - sum(fam_base.values()):+d} |",
        "",
        "## Single-child bedroom/living repair packages (QP6 input)",
        "",
        "| family | baseline | candidate | delta |",
        "|---|---:|---:|---:|",
    ]
    for family in sorted(set(single_base) | set(single_cand)):
        base, cand = single_base[family], single_cand[family]
        lines.append(f"| {family} | {base} | {cand} | {cand - base:+d} |")
    if not (single_base or single_cand):
        lines.append("| _none_ | 0 | 0 | 0 |")

    lines += [
        "",
        "## Per property",
        "",
        "| property | candidates b→c | added | removed | payload-changed | mismatch counter |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        counts = row["candidate_count"]
        lines.append(
            f"| {row['property_key']} | {counts['baseline']}→{counts['candidate']} | "
            f"{len(row['added_candidates'])} | {len(row['removed_candidates'])} | "
            f"{len(row['changed_candidates'])} | "
            f"{row['sanitization']['stored_decisions_payload_mismatch']} |"
        )

    if touched:
        lines += ["", "## Changed candidates in detail", ""]
        for row in touched:
            lines.append(f"### {row['property_key']}")
            lines.append("")
            for entry in row["added_candidates"]:
                lines.append(
                    f"- **added** `{entry['package_type']}` @ "
                    f"{entry['estimate_unit_id']} — needs a fresh Sol decision"
                )
            for entry in row["removed_candidates"]:
                lines.append(
                    f"- **removed** `{entry['package_type']}` @ "
                    f"{entry['estimate_unit_id']}"
                )
            for entry in row["changed_candidates"]:
                bits = []
                if entry["children_added"]:
                    bits.append(f"+{len(entry['children_added'])} children")
                if entry["children_removed"]:
                    bits.append(f"-{len(entry['children_removed'])} children")
                for flip in entry["child_role_flips"]:
                    bits.append(f"{flip['work_item_id']} {flip['from']}→{flip['to']}")
                for field in ("pricing_tier", "strength", "range_low", "range_high"):
                    delta = entry[field]
                    if delta:
                        bits.append(f"{field} {delta['from']}→{delta['to']}")
                lines.append(
                    f"- **changed** `{entry['package_type']}` @ "
                    f"{entry['estimate_unit_id']}: " + "; ".join(bits)
                )
            lines.append("")

    lines += [
        "",
        "## Sanitization counters (candidate side)",
        "",
        "| counter | total |",
        "|---|---:|",
    ]
    totals: Counter = Counter()
    for row in rows:
        totals.update(row["sanitization"])
    for key in _SANITIZATION_KEYS:
        lines.append(f"| {key} | {totals[key]} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path,
                        help="replay output root for the pre-3.2 baseline")
    parser.add_argument("--candidate", required=True, type=Path,
                        help="replay output root for the Catalog 3.2 tree")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports_scratch" / "cat32_census")
    args = parser.parse_args()

    baseline = _load_side(args.baseline)
    candidate = _load_side(args.candidate)
    _assert_baseline_is_lossless(baseline)
    missing = sorted(set(baseline) ^ set(candidate))
    if missing:
        raise SystemExit(
            "census_catalog_32: the two roots cover different properties: "
            f"{missing}"
        )

    rows = [
        compare_property(baseline[key], candidate[key])
        for key in sorted(baseline)
    ]
    report = render(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "census.json").write_text(
        json.dumps({"schema_version": 1, "properties": rows},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "census.md").write_text(report + "\n", encoding="utf-8")

    touched = sum(
        1 for r in rows
        if r["added_candidates"] or r["removed_candidates"] or r["changed_candidates"]
    )
    print(f"properties: {len(rows)} | changed: {touched}")
    print(f"wrote {args.out / 'census.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
