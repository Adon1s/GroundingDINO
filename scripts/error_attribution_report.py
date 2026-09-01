"""Scores the Pass 2a vs. downstream error attribution audit from its ledger.

Reads only the two ledger files -- reports/error_attribution_queue.json (the
frozen cases) and reports/error_attribution_verdicts.jsonl (the reviewer's
attributions) -- and never touches an artifact. Every total in the report is
recomputed from the ledger, so the numbers can be re-derived from two files and
a disagreement is a disagreement about the ledger, not about a walk.

  reports/error_attribution.md     human-readable report
  reports/error_attribution.json   machine-readable mirror

Reconciliation is enforced, not reported: a case with no verdict, a downstream
verdict with no stage, or a lane count that drifted from the queue fails the
run. `--allow-incomplete` reports partial progress during a review session.

The gold lane is review output, not extraction output, so its cases cannot live
in the frozen queue: `--gold-cases` merges the review-produced ga_/gx_ cases
into the same reconciliation and tally. Without the flag a gold verdict in the
ledger is an orphan and fails the run.

Run:
  .venv\\Scripts\\python.exe scripts\\error_attribution_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.comparison_common import atomic_json, sha256_file  # noqa: E402

QUEUE = ROOT / "reports" / "error_attribution_queue.json"
VERDICTS = ROOT / "reports" / "error_attribution_verdicts.jsonl"
OUT_MD = ROOT / "reports" / "error_attribution.md"
OUT_JSON = ROOT / "reports" / "error_attribution.json"

ATTRIBUTIONS = ("pass_2a", "downstream", "unclear", "untraceable", "excluded")
STAGES = ("2b", "2c", "2d", "2e", "condition_projection", "terra")
MISS_LANES = ("miss_label", "miss_v1only", "miss_gold")
HALLUC_LANES = ("halluc_label", "halluc_v1only", "halluc_gold_extra")
APPENDIX_LANES = ("appendix_misnamed", "appendix_trivial", "appendix_inconclusive")
GOLD_LANES = ("miss_gold", "halluc_gold_extra")

# A v1-only case carries one collapsed label instead of the two v1.1 axes, so its
# truth is weaker than the adjudicated cohort's; the report refuses to record it
# as high confidence rather than silently pooling the two bases.
V1_ONLY_MAX_CONFIDENCE = "medium"


class ReconciliationError(Exception):
    """A ledger that cannot be reported on without misleading someone."""


def latest_verdicts(path: Path) -> Dict[str, Dict[str, Any]]:
    """case_id -> latest record; a record with attribution null undoes the case.

    Same append-only, latest-wins contract as reports/review_verdicts.jsonl
    (tools/review_cards.py:573), so a re-review is one appended line."""
    out: Dict[str, Dict[str, Any]] = {}
    path = Path(path)
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        case_id = rec.get("case_id") if isinstance(rec, dict) else None
        if not case_id:
            continue
        if rec.get("attribution") is None:
            out.pop(case_id, None)
        else:
            out[case_id] = rec
    return out


def reconcile(queue: Dict[str, Any], verdicts: Dict[str, Dict[str, Any]],
              *, allow_incomplete: bool = False,
              gold: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Every invariant that must hold before a number in this report means anything."""
    problems: List[str] = []

    ids = [c["case_id"] for c in queue.get("cases") or []]
    dupes = [cid for cid, n in Counter(ids).items() if n > 1]
    if dupes:
        problems.append(f"duplicate case_id in queue: {sorted(dupes)[:5]}")

    # Gold cases are review output, so they merge in after the frozen-queue
    # header checks and must satisfy their own contract first.
    gold_cases = (gold or {}).get("cases") or []
    gold_ids = [c.get("case_id") for c in gold_cases]
    gold_dupes = [cid for cid, n in Counter(gold_ids).items() if n > 1]
    if gold_dupes:
        problems.append(f"duplicate case_id in gold cases: {sorted(gold_dupes)[:5]}")
    collisions = sorted(set(gold_ids) & set(ids))
    if collisions:
        problems.append(f"gold case ids collide with the queue: {collisions[:5]}")
    for case in gold_cases:
        cid = case.get("case_id") or "<gold case without id>"
        if not str(cid).startswith(("ga_", "gx_")):
            problems.append(f"{cid}: gold case ids must start with ga_ or gx_")
        if case.get("lane") not in GOLD_LANES:
            problems.append(f"{cid}: gold lane {case.get('lane')!r} outside {GOLD_LANES}")
        if not case.get("attribute"):
            problems.append(f"{cid}: gold cases must be attributable")
        if (case.get("human_truth") or {}).get("basis") != "gold":
            problems.append(f"{cid}: gold truth basis must be 'gold'")

    all_cases = (queue.get("cases") or []) + gold_cases
    cases = {c.get("case_id"): c for c in all_cases}

    # condition_id is run-scoped, so the same condition twice in one run is a real
    # collision -- including a gold case anchored on a condition an rc_ case carries.
    seen: Dict[Tuple[Any, Any, Any], str] = {}
    for case in all_cases:
        claim = case.get("v5_claim") or {}
        if not claim.get("condition_id"):
            continue
        run_ref = case.get("run_ref") or {}
        key = (run_ref.get("source"), run_ref.get("run_id"), claim["condition_id"])
        if key in seen:
            problems.append(f"condition {key[2]} appears in {seen[key]} and {case['case_id']}")
        seen[key] = case["case_id"]

    orphan_verdicts = sorted(set(verdicts) - set(cases))
    if orphan_verdicts:
        problems.append(f"verdicts with no queue case: {orphan_verdicts[:5]}")

    expected = {lane: n for lane, n in (queue.get("lane_counts") or {}).items()}
    actual = Counter(c["lane"] for c in queue.get("cases") or [])
    drift = {lane: (expected.get(lane), actual.get(lane)) for lane in set(expected) | set(actual)
             if expected.get(lane) != actual.get(lane)}
    if drift:
        problems.append(f"lane counts drifted from the queue header: {drift}")

    pending: List[str] = []
    for case_id, case in sorted(cases.items()):
        if not case.get("attribute"):
            continue
        rec = verdicts.get(case_id)
        if rec is None:
            if case.get("status") != "untraceable":
                pending.append(case_id)
            continue
        attribution = rec.get("attribution")
        if attribution not in ATTRIBUTIONS:
            problems.append(f"{case_id}: attribution {attribution!r} outside {ATTRIBUTIONS}")
        stage = rec.get("first_responsible_stage")
        if attribution == "downstream" and stage not in STAGES:
            problems.append(f"{case_id}: downstream needs first_responsible_stage, got {stage!r}")
        if attribution != "downstream" and stage:
            problems.append(f"{case_id}: first_responsible_stage set on a {attribution} verdict")
        basis = (case.get("human_truth") or {}).get("basis")
        if basis == "v1_only" and rec.get("confidence") == "high":
            problems.append(f"{case_id}: v1-only truth cannot carry high confidence")
        if not str(rec.get("rationale") or "").strip():
            problems.append(f"{case_id}: no rationale")
        joins = ((case.get("mechanical_hints") or {}).get("join_methods") or [])
        if "none" in joins and "join" not in str(rec.get("rationale") or "").lower():
            problems.append(f"{case_id}: 2b join failed; the rationale must address it")

    if pending and not allow_incomplete:
        problems.append(f"{len(pending)} attributable case(s) have no verdict: {pending[:5]}")
    if problems:
        raise ReconciliationError("; ".join(problems))
    return {"ok": True, "cases": len(cases), "verdicts": len(verdicts),
            "pending": pending,
            "attributable": sum(1 for c in cases.values() if c.get("attribute")),
            "gold_cases": len(gold_cases)}


def tally(queue: Dict[str, Any], verdicts: Dict[str, Dict[str, Any]],
          *, gold: Optional[Dict[str, Any]] = None,
          gold_path: Optional[Path] = None) -> Dict[str, Any]:
    """The whole report as one dict; the markdown is a rendering of exactly this."""
    cases = {c["case_id"]: c
             for c in (queue.get("cases") or []) + ((gold or {}).get("cases") or [])}
    by_lane: Dict[str, Counter] = defaultdict(Counter)
    stages_by_lane: Dict[str, Counter] = defaultdict(Counter)
    confidence: Counter = Counter()
    by_basis: Dict[str, Counter] = defaultdict(Counter)
    rows: List[Dict[str, Any]] = []

    for case_id, case in sorted(cases.items()):
        rec = verdicts.get(case_id)
        lane = case["lane"]
        if not case.get("attribute"):
            by_lane[lane]["counted_only"] += 1
            continue
        attribution = (rec or {}).get("attribution") or (
            "untraceable" if case.get("status") == "untraceable" else "pending")
        by_lane[lane][attribution] += 1
        basis = (case.get("human_truth") or {}).get("basis")
        by_basis[basis][attribution] += 1
        stage = (rec or {}).get("first_responsible_stage")
        if stage:
            stages_by_lane[lane][stage] += 1
        if rec:
            confidence[rec.get("confidence") or "unspecified"] += 1
        claim = case.get("v5_claim") or {}
        run_ref = case.get("run_ref") or {}
        rows.append({
            "case_id": case_id, "lane": lane, "basis": basis,
            "source": run_ref.get("source"),
            "property_key": run_ref.get("property_key"),
            "condition_id": claim.get("condition_id"),
            "catalog_item_id": claim.get("catalog_item_id"),
            "human_class": (case.get("human_truth") or {}).get("class_v1_1")
                           or (case.get("human_truth") or {}).get("v1_verdict"),
            "terra_verdict": claim.get("terra_verdict"),
            "attribution": attribution,
            "first_responsible_stage": stage,
            "confidence": (rec or {}).get("confidence"),
            "rationale": (rec or {}).get("rationale"),
        })

    def headline(lanes: Tuple[str, ...]) -> Dict[str, Any]:
        total = Counter()
        for lane in lanes:
            total.update(by_lane.get(lane, Counter()))
        judged = sum(v for k, v in total.items() if k in ("pass_2a", "downstream", "unclear"))
        return {"counts": dict(total), "judged": judged,
                "pass_2a": total.get("pass_2a", 0),
                "downstream": total.get("downstream", 0),
                "unclear": total.get("unclear", 0),
                "pass_2a_share": (total.get("pass_2a", 0) / judged) if judged else None,
                "downstream_share": (total.get("downstream", 0) / judged) if judged else None}

    stage_total = Counter()
    for lane, counter in stages_by_lane.items():
        stage_total.update(counter)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"queue": {"path": str(QUEUE.relative_to(ROOT)),
                             "sha256": sha256_file(QUEUE) if QUEUE.is_file() else None,
                             "generated_at": queue.get("generated_at")},
                   "verdicts": {"path": str(VERDICTS.relative_to(ROOT)),
                                "sha256": sha256_file(VERDICTS) if VERDICTS.is_file() else None},
                   "gold_cases": None if gold is None else {
                       "path": str(Path(gold_path)) if gold_path else None,
                       "sha256": (sha256_file(Path(gold_path))
                                  if gold_path and Path(gold_path).is_file() else None),
                       "generated_at": gold.get("generated_at")},
                   "queue_inputs": queue.get("inputs")},
        "misses": headline(MISS_LANES),
        "hallucinations": headline(HALLUC_LANES),
        "appendix": headline(APPENDIX_LANES),
        "by_lane": {lane: dict(counter) for lane, counter in sorted(by_lane.items())},
        "downstream_stages": dict(stage_total),
        "downstream_stages_by_lane": {lane: dict(c) for lane, c in sorted(stages_by_lane.items())},
        "by_basis": {basis: dict(c) for basis, c in sorted(by_basis.items(), key=lambda x: str(x[0]))},
        "confidence": dict(confidence),
        "p2b_join_health": queue.get("p2b_join_health"),
        "gold": {"photos": queue.get("gold_photo_count"),
                 "findings": queue.get("gold_finding_count"),
                 # Counted by whatever decisions the review actually recorded, so a
                 # new decision kind shows up instead of being silently dropped.
                 "matching": dict(Counter(str(row.get("decision"))
                                          for row in ((gold or {}).get("matching_table") or []))),
                 "cases": len((gold or {}).get("cases") or [])},
        "notes": queue.get("notes"),
        "cases": rows,
    }


def render(report: Dict[str, Any]) -> str:
    """Markdown rendering of the tally dict; every number comes from that dict."""
    out: List[str] = []
    add = out.append
    add("# Pass 2a vs. downstream error attribution")
    add("")
    add(f"Generated {report['generated_at']} from "
        f"`{report['inputs']['queue']['path']}` and `{report['inputs']['verdicts']['path']}`.")
    add("")
    add("Descriptive only: this is the reviewed discrepancy cohort, not a system-wide")
    add("miss or hallucination rate. Canary and production are not pooled into a rate.")
    add("")

    for title, key in (("Misses", "misses"), ("Hallucinations", "hallucinations"),
                       ("Appendix", "appendix")):
        block = report[key]
        add(f"## {title}")
        add("")
        add("| attribution | n | share of judged |")
        add("| --- | ---: | ---: |")
        for name in ("pass_2a", "downstream", "unclear"):
            n = block["counts"].get(name, 0)
            share = f"{n / block['judged']:.1%}" if block["judged"] else "-"
            add(f"| {name} | {n} | {share} |")
        for name, n in sorted(block["counts"].items()):
            if name not in ("pass_2a", "downstream", "unclear"):
                add(f"| {name} | {n} | - |")
        add(f"| **judged** | **{block['judged']}** | |")
        add("")

    add("## Downstream stages")
    add("")
    stages = report["downstream_stages"]
    if stages:
        add("| first responsible stage | n |")
        add("| --- | ---: |")
        for stage in STAGES:
            if stages.get(stage):
                add(f"| {stage} | {stages[stage]} |")
    else:
        add("No downstream attributions recorded.")
    add("")

    add("## Cases by lane")
    add("")
    add("| lane | " + " | ".join(ATTRIBUTIONS) + " | counted_only | pending |")
    add("| --- | " + " | ".join("---:" for _ in ATTRIBUTIONS) + " | ---: | ---: |")
    for lane, counts in report["by_lane"].items():
        cells = [str(counts.get(a, 0)) for a in ATTRIBUTIONS]
        add(f"| {lane} | " + " | ".join(cells)
            + f" | {counts.get('counted_only', 0)} | {counts.get('pending', 0)} |")
    add("")

    gold = report.get("gold") or {}
    if gold.get("matching") or gold.get("cases"):
        add("## Gold lane")
        add("")
        add(f"{gold.get('photos')} photos, {gold.get('findings')} frozen findings, "
            f"{gold.get('cases')} case(s) materialised from the review.")
        add("")
        add("Gold is photo-observation truth, not billable-condition truth: an unmatched")
        add("finding is only a miss when the v5 catalog could have carried it.")
        add("")
        if gold.get("matching"):
            add("| decision | n |")
            add("| --- | ---: |")
            for decision, n in sorted(gold["matching"].items()):
                add(f"| {decision} | {n} |")
            add("")

    add("## Truth basis")
    add("")
    add("`v1_1` carries both adjudicated axes; `v1_only` is one collapsed v1 label and")
    add(f"is capped at {V1_ONLY_MAX_CONFIDENCE} confidence.")
    add("")
    for basis, counts in report["by_basis"].items():
        add(f"- **{basis}**: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    add("")

    add("## Confidence")
    add("")
    add(", ".join(f"{k} {v}" for k, v in sorted(report["confidence"].items())) or "none recorded")
    add("")

    add("## Evidence table")
    add("")
    add("| case | lane | basis | property | catalog item | human | terra | attribution | stage | conf |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report["cases"]:
        add(f"| {row['case_id']} | {row['lane']} | {row['basis']} | {row['property_key']} "
            f"| {row['catalog_item_id']} | {row['human_class']} | {row['terra_verdict']} "
            f"| {row['attribution']} | {row['first_responsible_stage'] or ''} "
            f"| {row['confidence'] or ''} |")
    add("")

    add("## Limitations")
    add("")
    for note in report.get("notes") or []:
        add(f"- {note}")
    add(f"- 2b join health across the queue: {report.get('p2b_join_health')}.")
    add("")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--verdicts", type=Path, default=VERDICTS)
    parser.add_argument("--gold-cases", type=Path, default=None,
                        help="review-produced gold lane cases (ga_/gx_) to merge into the tally")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="report partial progress instead of failing on unreviewed cases")
    args = parser.parse_args(argv)

    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    verdicts = latest_verdicts(Path(args.verdicts))
    gold = (json.loads(Path(args.gold_cases).read_text(encoding="utf-8"))
            if args.gold_cases else None)
    try:
        status = reconcile(queue, verdicts, allow_incomplete=args.allow_incomplete, gold=gold)
    except ReconciliationError as exc:
        print(f"RECONCILIATION FAILED: {exc}")
        return 1
    report = tally(queue, verdicts, gold=gold, gold_path=args.gold_cases)
    report["reconciliation"] = status
    atomic_json(OUT_JSON, report)
    OUT_MD.write_text(render(report), encoding="utf-8")

    print(f"cases {status['cases']} | attributable {status['attributable']} "
          f"| gold cases {status['gold_cases']} "
          f"| verdicts {status['verdicts']} | pending {len(status['pending'])}")
    print(f"misses: {report['misses']['counts']}")
    print(f"hallucinations: {report['hallucinations']['counts']}")
    print(f"wrote {OUT_MD} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
