"""Score re-decide harness arms against the human review labels.

Joins each arm's re-decided Terra verdicts (scripts/
redecide_renovation_architecture.py output roots) to the review labels by
(property_key, condition_id). Scoreable population: the canary condition cards.

Two label versions are supported, and the version is always printed:

  v1 (default, no --labels)   the frozen 2026-08-26 review verdicts.
      hard_false_billed  (11)  flip away from supported      -> WIN
      supported_billed   (55)  flip away from supported      -> LOSS
      overstated_billed   (6)  flip away from supported      -> COST (the right
                               fix is wording, not exclusion)
      dirB_recovery      (16)  flip to supported             -> WIN (goal 2)
      The frozen inputs are asserted byte-for-byte (125 cards, 11/55/6/16) so a
      drifted queue fails loudly instead of mis-scoring.

  v1.1 (--labels reports/labels_v1_1.json)   the adjudicated overlay from
      docs/DESIGN_label_v1_1_adjudication.md. Classes and outcomes come from
      tools/label_schema.py; class counts are read from the label file rather
      than hard-coded, because the whole point of the repair is that they moved.
      Two outcomes are arm-dependent: a `misnamed` claim is literally wrong but
      sits on a real condition, so dropping it is a COST for an arm that leaves
      the wording alone and NEUTRAL for one that coarsens the claim — declare
      coarsening arms with --coarsened-variant-root, not --variant-root.

The Terra replica noise floor is 6.4% (16/250) and humans matched the two
replicas 8:6, so every arm is reported next to the same-prompt control arm —
never against zero. Dry-run roots score as not_redecided (coverage check only).

Usage:
  .venv\\Scripts\\python.exe scripts\\score_redecide_variants.py \
      --control-root artifacts_canary\\redecide_control_<date> \
      --variant-root artifacts_canary\\redecide_rubric_<date> \
      --coarsened-variant-root artifacts_canary\\redecide_coarsened_<date> \
      --labels reports\\labels_v1_1.json \
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

from tools import label_schema as ls  # noqa: E402
from tools.review_cards import latest_verdicts  # noqa: E402

EXPECTED_CLASS_N = {"hard_false_billed": 11, "supported_billed": 55,
                    "overstated_billed": 6, "dirB_recovery": 16}
EXPECTED_CANARY_CONDITION_CARDS = 125
OUTCOMES = ("wins", "losses", "costs", "neutrals")
OUTCOME_FIELD = {"win": "wins", "loss": "losses", "cost": "costs",
                 "neutral": "neutrals"}
NOISE_FLOOR_NOTE = (
    "Noise floor: Terra replica flips ran 6.4% (16/250) and humans matched "
    "run_1/run_2 8:6 — judge every arm against the control column, never "
    "against zero."
)


def classify(card: Mapping[str, Any], human_verdict: Optional[str]) -> str:
    return ls.classify_v1(card.get("strata") or [],
                          (card.get("meta") or {}).get("accepted"),
                          human_verdict)


def load_population(queue_path: Path, verdicts_path: Path) -> List[Dict[str, Any]]:
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


def load_population_v1_1(labels_path: Path, *, allow_partial: bool = False
                         ) -> List[Dict[str, Any]]:
    """Population and classes come from the adjudicated overlay, not constants."""
    doc = json.loads(labels_path.read_text(encoding="utf-8"))
    if doc.get("label_version") != ls.LABEL_VERSION:
        raise SystemExit(
            f"{labels_path} is label_version {doc.get('label_version')!r}, "
            f"this scorer speaks {ls.LABEL_VERSION!r}"
        )
    rows = []
    pending = []
    for origin, row in sorted(doc["labels"].items()):
        if row.get("source") != "canary":
            continue
        if not row.get("slug"):
            pending.append(origin)
            continue
        rows.append({
            "card_id": origin,
            "property_key": row["property_key"],
            "condition_id": row["condition_id"],
            "stored_verdict": row["stored_terra_verdict"],
            "human_verdict": row["v1_1_projected_verdict"],
            "klass": row["class_v1_1"],
        })
    if pending and not allow_partial:
        raise SystemExit(
            f"{len(pending)} canary cards are not adjudicated yet "
            f"(e.g. {pending[:3]}). Finish phases 1+2, or pass --allow-partial "
            "and read the scorecard as provisional."
        )
    return rows


def load_arm(root: Path, *, claim_text_changed: bool = False) -> Dict[str, Any]:
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
        "claim_text_changed": claim_text_changed,
        "verdicts": verdicts,
        "statuses": statuses,
    }


def score_arm(population: List[Dict[str, Any]], arm: Mapping[str, Any],
              policy: Optional[Mapping[str, Tuple[str, str, str]]] = None
              ) -> Dict[str, Any]:
    policy = policy or ls.SCORING_V1
    per_class: Dict[str, Dict[str, Any]] = {
        name: {"n": 0, "redecided": 0, "not_redecided": 0, "changed": 0,
               "wins": 0, "losses": 0, "costs": 0, "neutrals": 0,
               "transitions": Counter()}
        for name in policy
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
        if new_verdict != stored:
            stats["changed"] += 1
            stats["transitions"][f"{stored}->{new_verdict}"] += 1
        outcome = ls.outcome_for(
            policy, klass, new_verdict,
            claim_text_changed=bool(arm.get("claim_text_changed")))
        if outcome:
            stats[OUTCOME_FIELD[outcome]] += 1
    for stats in per_class.values():
        stats["transitions"] = dict(sorted(stats["transitions"].items()))
    return {"label": arm["label"], "dry_run": arm["dry_run"],
            "claim_text_changed": bool(arm.get("claim_text_changed")),
            "classes": per_class, "out_of_scope": dict(buckets)}


def _rate(stats: Mapping[str, Any], field: str) -> str:
    if not stats["redecided"]:
        return "—"
    return f"{stats[field]}/{stats['redecided']}"


def render(scores: List[Dict[str, Any]], label_version: str,
           policy: Mapping[str, Tuple[str, str, str]]) -> str:
    control = scores[0]
    lines = [f"# Re-decide scorecard — labels {label_version}", "",
             NOISE_FLOOR_NOTE, ""]
    if label_version == ls.LABEL_VERSION:
        lines += [
            "`trivial_billed` and `dirB_trivial` are recorded but never scored: "
            "they are severity/threshold failures, and no wording change makes a "
            "model reject a true claim about a mildly dated finish.", ""]
    for arm in scores:
        tags = []
        if arm is control:
            tags.append("control")
        if arm["claim_text_changed"]:
            tags.append("coarsened claims")
        if arm["dry_run"]:
            tags.append("DRY RUN, nothing re-decided")
        lines += [
            f"## {arm['label']}" + (f" ({'; '.join(tags)})" if tags else ""),
            "",
            "| class | n | redecided | changed | win | loss | cost | neutral "
            "| control changed |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name in policy:
            stats = arm["classes"][name]
            control_stats = control["classes"][name]
            lines.append(
                f"| {name} | {stats['n']} | {stats['redecided']} "
                f"| {_rate(stats, 'changed')} | {_rate(stats, 'wins')} "
                f"| {_rate(stats, 'losses')} | {_rate(stats, 'costs')} "
                f"| {_rate(stats, 'neutrals')} "
                f"| {_rate(control_stats, 'changed')} |"
            )
        totals = {o: sum(c[o] for c in arm["classes"].values()) for o in OUTCOMES}
        lines += ["", "Totals: " + " · ".join(f"{o} {totals[o]}" for o in OUTCOMES)]
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", type=Path,
                        default=REPO_ROOT / "reports" / "review_queue.json")
    parser.add_argument("--verdicts", type=Path,
                        default=REPO_ROOT / "reports" / "review_verdicts.jsonl")
    parser.add_argument("--labels", type=Path, default=None,
                        help="reports/labels_v1_1.json — score against the "
                             "adjudicated overlay instead of the frozen v1 labels")
    parser.add_argument("--allow-partial", action="store_true",
                        help="score a v1.1 label set that is not finished yet")
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, action="append", default=[])
    parser.add_argument("--coarsened-variant-root", type=Path, action="append",
                        default=[], help="a variant that rewrites the claim text; "
                                         "changes the two arm-dependent outcomes")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "reports_scratch" / "redecide_scorecard")
    args = parser.parse_args(argv)

    if args.labels:
        population = load_population_v1_1(args.labels,
                                          allow_partial=args.allow_partial)
        policy, label_version = ls.SCORING, ls.LABEL_VERSION
    else:
        population = load_population(args.queue, args.verdicts)
        policy, label_version = ls.SCORING_V1, "v1"

    arms = [load_arm(args.control_root)]
    arms += [load_arm(r) for r in args.variant_root]
    arms += [load_arm(r, claim_text_changed=True)
             for r in args.coarsened_variant_root]
    scores = [score_arm(population, arm, policy) for arm in arms]

    counts = Counter(row["klass"] for row in population)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scorecard.json").write_text(
        json.dumps({"schema_version": 2, "label_version": label_version,
                    "noise_floor": "16/250",
                    "population": dict(sorted(counts.items())),
                    "arms": scores}, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    (args.out / "scorecard.md").write_text(
        render(scores, label_version, policy), encoding="utf-8")
    print(f"labels: {label_version} ({len(population)} canary cards)")
    for name in policy:
        print(f"  {name:24s} {counts.get(name, 0)}")
    print(f"scorecard: {args.out / 'scorecard.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
