"""Score re-decide harness arms against the frozen human review labels.

Joins each arm's re-decided Terra verdicts (scripts/
redecide_renovation_architecture.py output roots) to the frozen review queue
and verdict log by (property_key, condition_id). Scoreable population: the
125 canary condition cards. Classes and scoring rule (QP1, corrected):

  hard_false_billed  (11)  flip away from supported      -> WIN
  supported_billed   (55)  flip away from supported      -> LOSS
  overstated_billed   (6)  flip away from supported      -> COST (the right
                           fix is wording, not exclusion)
  dirB_recovery      (16)  flip to supported             -> WIN (goal 2)

The Terra replica noise floor is 6.4% (16/250) and humans matched the two
replicas 8:6, so every arm is reported next to the same-prompt control arm —
never against zero. Dry-run roots score as not_redecided (coverage check
only); the frozen inputs are asserted byte-for-byte (125 cards, 11/55/6/16)
so a drifted queue fails loudly instead of mis-scoring.

Usage:
  .venv\\Scripts\\python.exe scripts\\score_redecide_variants.py \
      --control-root artifacts_canary\\redecide_control_<date> \
      --variant-root artifacts_canary\\redecide_<label>_<date> \
      --out reports_scratch\\redecide_scorecard
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.review_cards import latest_verdicts  # noqa: E402

CLASSES = ("hard_false_billed", "supported_billed", "overstated_billed",
           "dirB_recovery")
EXPECTED_CLASS_N = {"hard_false_billed": 11, "supported_billed": 55,
                    "overstated_billed": 6, "dirB_recovery": 16}
EXPECTED_CANARY_CONDITION_CARDS = 125
NOISE_FLOOR_NOTE = (
    "Noise floor: Terra replica flips ran 6.4% (16/250) and humans matched "
    "run_1/run_2 8:6 — judge every arm against the control column, never "
    "against zero."
)


def classify(card: Mapping[str, Any], human_verdict: Optional[str]) -> str:
    strata = set(card.get("strata") or [])
    billed = bool((card.get("meta") or {}).get("accepted"))
    if "dirB" in strata:
        return ("dirB_recovery" if human_verdict == "terra_claim_supported"
                else "dirB_other")
    if billed and (strata & {"dirA", "uniform"}):
        return {
            "terra_claim_unsupported": "hard_false_billed",
            "terra_claim_supported": "supported_billed",
            "terra_claim_overstated": "overstated_billed",
        }.get(human_verdict or "", "other_billed")
    return "out_of_scope"


def load_population(
    queue_path: Path, verdicts_path: Path
) -> List[Dict[str, Any]]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    cards = [c for c in queue["cards"]
             if c.get("kind") == "condition" and c.get("source") == "canary"]
    if len(cards) != EXPECTED_CANARY_CONDITION_CARDS:
        raise SystemExit(
            f"queue has {len(cards)} canary condition cards, expected "
            f"{EXPECTED_CANARY_CONDITION_CARDS} — the frozen queue drifted"
        )
    # Orphan verdicts drop by construction: only card_ids in the queue join.
    labels = latest_verdicts(verdicts_path)
    rows = []
    for card in cards:
        human = (labels.get(card["card_id"]) or {}).get("verdict")
        rows.append({
            "card_id": card["card_id"],
            "property_key": card["property_key"],
            "condition_id": (card.get("meta") or {}).get("condition_id"),
            "stored_verdict": (card.get("meta") or {}).get("terra_verdict"),
            "human_verdict": human,
            "klass": classify(card, human),
        })
    counts = Counter(row["klass"] for row in rows)
    drift = {name: (counts.get(name, 0), expected)
             for name, expected in EXPECTED_CLASS_N.items()
             if counts.get(name, 0) != expected}
    if drift:
        raise SystemExit(
            f"class counts drifted from the frozen labels: {drift}"
        )
    return rows


def load_arm(root: Path) -> Dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    verdicts: Dict[Tuple[str, str], str] = {}
    statuses: Dict[Tuple[str, str], str] = {}
    for unit_file in sorted(root.glob("*/units/terra_unit_*.json")):
        record = json.loads(unit_file.read_text(encoding="utf-8"))
        prop = str(record.get("property_key") or "")
        for condition_id in record.get("condition_ids") or []:
            join = (prop, str(condition_id))
            statuses[join] = str(record.get("status") or "")
            verdict = (
                (record.get("reviews") or {}).get(condition_id) or {}
            ).get("verdict")
            if verdict:
                verdicts[join] = str(verdict)
    return {
        "label": str((manifest.get("variant") or {}).get("label") or root.name),
        "dry_run": bool(manifest.get("dry_run")),
        "verdicts": verdicts,
        "statuses": statuses,
    }


def score_arm(
    population: List[Dict[str, Any]], arm: Mapping[str, Any]
) -> Dict[str, Any]:
    per_class: Dict[str, Dict[str, Any]] = {
        name: {"n": 0, "redecided": 0, "not_redecided": 0, "changed": 0,
               "wins": 0, "losses": 0, "costs": 0, "transitions": Counter()}
        for name in CLASSES
    }
    buckets = Counter()
    for row in population:
        klass = row["klass"]
        if row["human_verdict"] is None:
            buckets["no_human_verdict"] += 1
            continue
        if klass not in per_class:
            buckets[klass] += 1
            continue
        stats = per_class[klass]
        stats["n"] += 1
        join = (row["property_key"], str(row["condition_id"]))
        new_verdict = arm["verdicts"].get(join)
        if new_verdict is None:
            stats["not_redecided"] += 1
            continue
        stats["redecided"] += 1
        stored = str(row["stored_verdict"])
        changed = new_verdict != stored
        if changed:
            stats["changed"] += 1
            stats["transitions"][f"{stored}->{new_verdict}"] += 1
        if klass == "dirB_recovery":
            stats["wins"] += int(new_verdict == "supported")
        elif klass == "hard_false_billed":
            stats["wins"] += int(new_verdict != "supported")
        elif klass == "supported_billed":
            stats["losses"] += int(new_verdict != "supported")
        elif klass == "overstated_billed":
            stats["costs"] += int(new_verdict != "supported")
    for stats in per_class.values():
        stats["transitions"] = dict(sorted(stats["transitions"].items()))
    return {"label": arm["label"], "dry_run": arm["dry_run"],
            "classes": per_class, "out_of_scope": dict(buckets)}


def _rate(stats: Mapping[str, Any], field: str) -> str:
    if not stats["redecided"]:
        return "—"
    return f"{stats[field]}/{stats['redecided']}"


def render(scores: List[Dict[str, Any]]) -> str:
    control = scores[0]
    lines = ["# Re-decide scorecard", "", NOISE_FLOOR_NOTE, ""]
    for arm in scores:
        lines += [
            f"## {arm['label']}"
            + (" (control)" if arm is control else "")
            + (" — DRY RUN, nothing re-decided" if arm["dry_run"] else ""),
            "",
            "| class | n | redecided | changed | wins | losses | costs "
            "| control changed |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name in CLASSES:
            stats = arm["classes"][name]
            control_stats = control["classes"][name]
            lines.append(
                f"| {name} | {stats['n']} | {stats['redecided']} "
                f"| {_rate(stats, 'changed')} | {_rate(stats, 'wins')} "
                f"| {_rate(stats, 'losses')} | {_rate(stats, 'costs')} "
                f"| {_rate(control_stats, 'changed')} |"
            )
        transitions = {
            name: stats["transitions"]
            for name, stats in arm["classes"].items()
            if stats["transitions"]
        }
        if transitions:
            lines += ["", f"Transitions: `{json.dumps(transitions, sort_keys=True)}`"]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path,
                        default=REPO_ROOT / "reports" / "review_queue.json")
    parser.add_argument("--verdicts", type=Path,
                        default=REPO_ROOT / "reports" / "review_verdicts.jsonl")
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, action="append",
                        default=[])
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "reports_scratch" / "redecide_scorecard")
    args = parser.parse_args(argv)

    population = load_population(args.queue, args.verdicts)
    scores = [
        score_arm(population, load_arm(root))
        for root in [args.control_root, *args.variant_root]
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scorecard.json").write_text(
        json.dumps({"schema_version": 1, "noise_floor": "16/250",
                    "arms": scores}, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    (args.out / "scorecard.md").write_text(render(scores), encoding="utf-8")
    print(f"scorecard: {args.out / 'scorecard.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
