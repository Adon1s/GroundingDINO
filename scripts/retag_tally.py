"""Tally reports/retag_verdicts.jsonl -> reports/retag_tally.md.

Group A is reported in **weighted error mass**, not card counts: a uniform-arm
card stands for accepted_non_dirA/n_uniform of the population (canary 17.775,
production 13.143) while a dirA card is a census record weighting 1 — the same
identity scripts/review_analysis.py uses for gate_exposure, so the masses here
reconcile with reports/review_analysis.md §9 (canary 100.875, production 21.143).
Counting cards instead would let two uniform cards look like two errors when
they stand for ~36.

Group B (dirB recoveries) is a targeted stratum: raw counts only, no weighting.

Run:
  .venv\\Scripts\\python.exe scripts\\retag_tally.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import review_cards as rc  # noqa: E402

RETAG_QUEUE = ROOT / "reports" / "retag_queue.json"
RETAG_VERDICTS = ROOT / "reports" / "retag_verdicts.jsonl"
REVIEW_QUEUE = ROOT / "reports" / "review_queue.json"
OUT_MD = ROOT / "reports" / "retag_tally.md"

FIXABLE = {"A": {"mechanism_only"}, "B": {"wording_blocked"}}  # the "wording, not perception" answers


def slug(verdict: str | None) -> str:
    return str(verdict or "").split(":", 1)[0].strip() or "unanswered"


def uniform_weights(review_queue: Path) -> Dict[str, float]:
    """accepted_non_dirA / n_uniform per source — the uniform arm's population weight."""
    meta = json.loads(review_queue.read_text(encoding="utf-8"))["meta"]
    out = {}
    for src, m in meta.items():
        n_uniform = (m.get("strata") or {}).get("uniform") or 0
        out[src] = (m["accepted_non_dirA"] / n_uniform) if n_uniform else 0.0
    return out


def card_weight(card: Dict[str, Any], weights: Dict[str, float]) -> float:
    strata = card.get("strata") or []
    if "dirA" in strata:
        return 1.0
    return weights.get(card["source"], 0.0) if "uniform" in strata else 0.0


def table(rows: List[List[str]], head: List[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=RETAG_QUEUE)
    ap.add_argument("--verdicts", type=Path, default=RETAG_VERDICTS)
    ap.add_argument("--review-queue", type=Path, default=REVIEW_QUEUE)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    args = ap.parse_args(argv)

    q = json.loads(args.queue.read_text(encoding="utf-8"))
    cards = {c["card_id"]: c for c in q["cards"]}
    done = rc.latest_verdicts(args.verdicts)
    weights = uniform_weights(args.review_queue)

    md = ["# Re-tag tally — mechanism (language) vs perception",
          "",
          f"{len(done)} of {len(cards)} cards answered. Group A is weighted error mass "
          "(dirA census = 1, uniform = accepted_non_dirA/n_uniform), reconciling with "
          "`reports/review_analysis.md` §9; group B is raw targeted counts.", ""]

    # ---- group A: weighted mass by answer, per source
    mass: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total: Dict[str, float] = defaultdict(float)
    counts_a: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cid, card in cards.items():
        if card["meta"]["retag_group"] != "A":
            continue
        w = card_weight(card, weights)
        src = card["source"]
        total[src] += w
        s = slug((done.get(cid) or {}).get("verdict"))
        mass[src][s] += w
        counts_a[src][s] += 1

    md += ["## Group A — the billed errors: is a real problem there?", ""]
    answers = sorted({s for m in mass.values() for s in m})
    rows = []
    for src in sorted(mass):
        for s in answers:
            if not counts_a[src][s]:
                continue
            share = mass[src][s] / total[src] * 100 if total[src] else 0.0
            rows.append([src, s, counts_a[src][s], f"{mass[src][s]:.2f}", f"{share:.1f}%"])
    md += [table(rows, ["source", "answer", "cards", "error mass", "share of that source's error mass"]), ""]

    for src in sorted(mass):
        fixable = sum(m for s, m in mass[src].items() if s in FIXABLE["A"])
        unanswered = mass[src].get("unanswered", 0.0)
        if total[src] and not unanswered:
            md += [f"**{src}: {fixable / total[src] * 100:.0f}% of measured billed-error mass is "
                   f"mechanism-only** — a correctly-worded (or coarser) claim would have been true. "
                   f"The remaining {100 - fixable / total[src] * 100:.0f}% is not a wording problem.", ""]
        elif total[src]:
            md += [f"*{src}: {unanswered / total[src] * 100:.0f}% of error mass still unanswered.*", ""]

    # ---- group B: raw counts
    counts_b: Dict[str, int] = defaultdict(int)
    per_item: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cid, card in cards.items():
        if card["meta"]["retag_group"] != "B":
            continue
        s = slug((done.get(cid) or {}).get("verdict"))
        counts_b[s] += 1
        per_item[card["meta"].get("catalog_item_id") or "?"][s] += 1

    md += ["## Group B — the dropped real conditions: why was it rejected?", "",
           table([[s, n] for s, n in sorted(counts_b.items(), key=lambda kv: -kv[1])], ["answer", "cards"]), "",
           "Targeted stratum (Terra-rejected, 2f-confirmed) — counts only, not a population rate.", ""]
    rows_b = [[item, ", ".join(f"{s} {n}" for s, n in sorted(d.items(), key=lambda kv: -kv[1]))]
              for item, d in sorted(per_item.items()) if sum(d.values()) > 1]
    if rows_b:
        md += ["### by catalog item (2+ cards)", "", table(rows_b, ["catalog item", "answers"]), ""]

    # ---- what it means
    md += ["## Reading", "",
           "- High `mechanism_only` / `wording_blocked` share => the system SEES correctly and NAMES badly. "
           "Fixable by the Terra rubric (QP2) and/or by coarsening the degradation claim text so the claim "
           "stops asserting a distinction the photo cannot settle. Both ride one canary.",
           "- High `wholly_false` / `terra_miss` share => a perception limit. Wording work will not move it; "
           "the levers are corroboration and the escalation/inspection lane.",
           "- High `too_trivial` => neither: a severity/threshold problem, not truth.", ""]

    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.out}  ({len(done)}/{len(cards)} answered)")
    for src in sorted(mass):
        fixable = sum(m for s, m in mass[src].items() if s in FIXABLE["A"])
        if total[src]:
            print(f"  group A {src}: mechanism_only = {fixable / total[src] * 100:.0f}% of error mass "
                  f"({total[src]:.1f} total mass)")
    if counts_b:
        print(f"  group B: {dict(counts_b)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
