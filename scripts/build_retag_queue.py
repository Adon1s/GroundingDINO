"""Build reports/retag_queue.json — the 46-card "mechanism or perception?" re-tag.

Selects, from the FROZEN review evidence (read-only, never modified):

  group A (26 cards) the weighted billed-error records — dirA-accepted or uniform
          cards whose human verdict was `unsupported` or `overstated`. These are
          the errors behind the 13.6% / 10.4% broad rates.
  group B (20 cards) the dirB recoveries — Terra-rejected claims the human judged
          `supported`. These are the real conditions v5 dropped.

and re-asks ONE question per card: was the *underlying problem* real (so a
differently-worded claim would have been right), or was nothing there at all?
The answer decides whether the degradation-mechanism granularity is a language
problem (fixable by rubric or by coarsening the catalog claims) or a perception
limit (not fixable by wording).

Deliberately NOT blind: each card shows the reviewer's own earlier verdict and
note, because the question being asked is a different one and the prior note
often already contains the answer. Zero provider calls; writes one new file.

Run:
  .venv\\Scripts\\python.exe scripts\\build_retag_queue.py [--check]
  .venv\\Scripts\\python.exe scripts\\review_server.py --queue reports\\retag_queue.json --verdicts reports\\retag_verdicts.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import review_cards as rc  # noqa: E402

QUEUE = ROOT / "reports" / "review_queue.json"
VERDICTS = ROOT / "reports" / "review_verdicts.jsonl"
OUT = ROOT / "reports" / "retag_queue.json"

# The frozen evidence this exercise is defined against (reports/review_analysis.md §1.1).
FROZEN_QUEUE_SHA = "8512b87c2af18d1104069c15a5eda977d5fe79eb736f9aedd976d174f89d318a"
FROZEN_VERDICTS_SHA = "0c7ca6a8b55d6dac69021a00e81315be1b15cef68ffd59fbe2973cb895a8d70f"

V_SUP, V_UNS, V_OVER, _V_INC = rc.CONDITION_VERDICTS
BROAD = (V_UNS, V_OVER)

GROUPS = {
    "A": {
        "phase": 1,
        "question": "was a real problem there, just named wrong?",
        "expected": 26,
        "keys": {
            "1": "mechanism_only: a real problem is there, the claim named it wrong",
            "2": "wholly_false: nothing of this kind is there at all",
            "3": "too_trivial: something is there but not worth billing",
            "4": "cannot_tell: evidence too poor to judge",
        },
    },
    "B": {
        "phase": 2,
        "question": "why did Terra reject something you could see?",
        "expected": 20,
        "keys": {
            "1": "wording_blocked: the claim's specific wording is what failed",
            "2": "terra_miss: claim was fair as worded, Terra missed it",
            "3": "borderline: visible but a close call, rejection defensible",
            "4": "cannot_tell: evidence too poor to judge",
        },
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(cards: List[Dict[str, Any]], done: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group A / group B card lists, mirroring scripts/review_analysis.py:arm_records."""
    out: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}
    for card in cards:
        if card.get("kind") != "condition":
            continue
        rec = done.get(card["card_id"])
        if not rec:
            continue
        verdict, strata = rec.get("verdict"), card.get("strata") or []
        in_arm = ("dirA" in strata and (card.get("meta") or {}).get("accepted")) or "uniform" in strata
        if in_arm and verdict in BROAD:
            out["A"].append(card)
        elif "dirB" in strata and verdict == V_SUP:
            out["B"].append(card)
    return out


def retag_card(card: Dict[str, Any], group: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    """The same condition card, re-pointed at the re-tag question (kind 'retag' hides the a-e tags)."""
    g = GROUPS[group]
    meta = dict(card.get("meta") or {})
    prior = str(rec.get("verdict") or "").replace("terra_claim_", "").replace("terra_evidence_", "")
    note = (rec.get("notes") or "").strip()
    out = dict(card)
    out.update({
        "kind": "retag",
        "phase": g["phase"],
        "title": f"[{group}] {g['question']}   |   {card.get('title')}",
        "claim": {
            "catalog_claim": (card.get("claim") or {}).get("catalog_claim"),
            "observations": [f">> your earlier call: {prior}" + (f' - "{note}"' if note else " (no note)")]
                            + list((card.get("claim") or {}).get("observations") or []),
        },
        "meta": {**meta, "retag_group": group, "prior_verdict": rec.get("verdict"), "prior_note": note or None},
        "reveal": [{"label": "your earlier verdict", "text": f"{rec.get('verdict')}" + (f" — {note}" if note else "")}]
                  + list(card.get("reveal") or []),
        "verdict_options": list(g["keys"].values()),
        "verdict_keys": dict(g["keys"]),
        "tags": {},
    })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=QUEUE)
    ap.add_argument("--verdicts", type=Path, default=VERDICTS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="fail unless the group counts are 26 / 20")
    args = ap.parse_args(argv)

    q_sha, v_sha = sha256(args.queue), sha256(args.verdicts)
    print(f"queue    {args.queue.name}  {q_sha[:12]}  {'== frozen' if q_sha == FROZEN_QUEUE_SHA else '!! DRIFTED from the frozen evidence'}")
    print(f"verdicts {args.verdicts.name}  {v_sha[:12]}  {'== frozen' if v_sha == FROZEN_VERDICTS_SHA else '!! DRIFTED from the frozen evidence'}")

    cards = json.loads(args.queue.read_text(encoding="utf-8"))["cards"]
    done = rc.latest_verdicts(args.verdicts)
    groups = select(cards, done)

    out_cards: List[Dict[str, Any]] = []
    for group in ("A", "B"):
        picked = sorted(groups[group], key=lambda c: (c["source"], c["property_key"],
                                                      (c.get("meta") or {}).get("catalog_item_id") or "", c["card_id"]))
        out_cards += [retag_card(c, group, done[c["card_id"]]) for c in picked]

    payload = {
        "generated_for": "mechanism-vs-perception re-tag of the frozen manual review",
        "source_queue": {"path": str(args.queue), "sha256": q_sha},
        "source_verdicts": {"path": str(args.verdicts), "sha256": v_sha},
        "groups": {g: {"question": GROUPS[g]["question"], "keys": GROUPS[g]["keys"],
                       "n": sum(1 for c in out_cards if c["meta"]["retag_group"] == g)} for g in ("A", "B")},
        "cards": out_cards,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {args.out}  cards={len(out_cards)}")
    ok = True
    for group in ("A", "B"):
        n, exp = payload["groups"][group]["n"], GROUPS[group]["expected"]
        flag = "ok" if n == exp else "FAIL"
        ok &= n == exp
        print(f"  [{flag}] group {group}: {n} cards (expected {exp}) - {GROUPS[group]['question']}")
        by_src: Dict[str, int] = {}
        for c in out_cards:
            if c["meta"]["retag_group"] == group:
                by_src[c["source"]] = by_src.get(c["source"], 0) + 1
        print(f"         by source: {by_src}   listings: {len({c['property_key'] for c in out_cards if c['meta']['retag_group'] == group})}")
    print("\nnext:\n  .venv\\Scripts\\python.exe scripts\\review_server.py --queue reports\\retag_queue.json --verdicts reports\\retag_verdicts.jsonl")
    if args.check and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
