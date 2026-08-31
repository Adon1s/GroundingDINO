"""Emit reports/labels_v1_1.json — the corrected label overlay, and its report.

Design: docs/DESIGN_label_v1_1_adjudication.md. This script owns no judgment;
it applies the pre-committed tables in tools/label_schema.py to the recorded
adjudication answers and reports what moved.

v1 is never rewritten. The overlay is keyed by the ORIGINAL `card_id` so it
joins straight onto the frozen queue, `scripts/score_redecide_variants.py`
(`--labels`) and, later, the population-rate recomputation.

Pre-committed tie-break: when a repeat disagrees with its primary, the
**primary** answer is the label. The repeat exists to measure consistency, not
to re-litigate a card — letting the second answer win would make the measured
consistency rate meaningless and would quietly re-introduce the anchoring the
blind arm was built to remove. Disagreements are reported, never silently
resolved.

Run:
  .venv\\Scripts\\python.exe scripts\\build_labels_v1_1.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import label_schema as ls  # noqa: E402
from tools import review_cards as rc  # noqa: E402

ADJ_QUEUE = ROOT / "reports" / "adjudication_queue.json"
ADJ_VERDICTS = ROOT / "reports" / "adjudication_verdicts.jsonl"
OUT_JSON = ROOT / "reports" / "labels_v1_1.json"
OUT_MD = ROOT / "reports" / "labels_v1_1.md"

VALID_SLUGS = set(ls.ADJUDICATION_KEYS.values())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def build(queue: Dict[str, Any], answers: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    provenance: Dict[str, Dict[str, Any]] = queue["provenance"]

    bad = {aid: rec.get("verdict") for aid, rec in answers.items()
           if aid in provenance
           and ls.slug_of(rec.get("verdict")) not in VALID_SLUGS}
    if bad:
        raise SystemExit(
            "answers outside the v1.1 vocabulary — is this the right verdict "
            f"log? {dict(list(bad.items())[:5])}"
        )

    # primary answers first; repeats are consistency evidence only
    primary_of: Dict[str, str] = {}      # origin_card_id -> adjudication id
    repeat_of: Dict[str, str] = {}
    for aid, prov in provenance.items():
        (primary_of if prov["role"] == "primary" else repeat_of)[
            prov["origin_card_id"]] = aid

    labels: Dict[str, Dict[str, Any]] = {}
    repeat_rows: List[Dict[str, Any]] = []
    for origin, aid in sorted(primary_of.items()):
        prov = provenance[aid]
        rec = answers.get(aid)
        slug = ls.slug_of(rec.get("verdict")) if rec else None
        claim = ls.CLAIM_AXIS.get(slug or "")
        work = ls.WORK_AXIS.get(slug or "")
        klass = ls.classify_v1_1(prov["strata"], prov["accepted"], slug)

        retag = prov.get("retag_answer")
        expectation = ls.RETAG_EXPECTATION.get(retag or "", {}) if retag else {}
        met: Optional[bool] = None
        if slug and expectation:
            axes = {"claim": claim, "work": work}
            met = all(axes[axis] == want for axis, want in expectation.items())

        row = {
            "adjudication_card_id": aid,
            "arm": prov["arm"],
            "source": prov["source"],
            "property_key": prov["property_key"],
            "condition_id": prov["condition_id"],
            "catalog_item_id": prov["catalog_item_id"],
            "catalog_kind": prov["catalog_kind"],
            "strata": prov["strata"],
            "accepted": prov["accepted"],
            "stored_terra_verdict": prov["stored_terra_verdict"],
            "v1_verdict": prov["v1_verdict"],
            "v1_class": prov["v1_class"],
            "slug": slug,
            "claim": claim,
            "work": work,
            "class_v1_1": klass,
            "v1_1_projected_verdict": ls.V1_PROJECTION.get(claim or ""),
            "note": (rec or {}).get("notes") or None,
            "retag_answer": retag,
            "retag_expectation_met": met,
            "repeat": None,
        }

        rid = repeat_of.get(origin)
        if rid:
            rrec = answers.get(rid)
            rslug = ls.slug_of(rrec.get("verdict")) if rrec else None
            row["repeat"] = {
                "adjudication_card_id": rid,
                "slug": rslug,
                "agrees": None if (rslug is None or slug is None) else rslug == slug,
                "claim_agrees": None if (rslug is None or slug is None)
                else ls.CLAIM_AXIS.get(rslug) == claim,
            }
            repeat_rows.append(row)
        labels[origin] = row

    return {
        "schema_version": 1,
        "label_version": ls.LABEL_VERSION,
        "generated_for": "v1.1 label repair — docs/DESIGN_label_v1_1_adjudication.md",
        "tie_break": "primary answer wins; repeats measure consistency only",
        "sources": queue.get("sources"),
        "coverage": coverage(labels, queue),
        "movement": movement(labels),
        "repeat_consistency": repeat_consistency(repeat_rows),
        "retag_agreement": retag_agreement(labels),
        "labels": labels,
    }


def coverage(labels: Dict[str, Dict[str, Any]], queue: Dict[str, Any]) -> Dict[str, Any]:
    per_arm: Dict[str, Counter] = defaultdict(Counter)
    for row in labels.values():
        per_arm[row["arm"]]["total"] += 1
        per_arm[row["arm"]]["answered" if row["slug"] else "pending"] += 1
    answered = sum(1 for r in labels.values() if r["slug"])
    canary = [r for r in labels.values() if r["source"] == "canary"]
    return {
        "cards": len(labels),
        "answered": answered,
        "pending": len(labels) - answered,
        "canary_complete": all(r["slug"] for r in canary),
        "per_arm": {arm: dict(c) for arm, c in sorted(per_arm.items())},
    }


def movement(labels: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    matrix: Dict[str, Counter] = defaultdict(Counter)
    for row in labels.values():
        if row["slug"]:
            matrix[row["v1_class"]][row["class_v1_1"]] += 1
    v1_counts = Counter(r["v1_class"] for r in labels.values() if r["slug"])
    v1_1_counts = Counter(r["class_v1_1"] for r in labels.values() if r["slug"])
    canary = [r for r in labels.values() if r["source"] == "canary" and r["slug"]]
    return {
        "matrix": {k: dict(v) for k, v in sorted(matrix.items())},
        "v1_class_counts": dict(v1_counts),
        "v1_1_class_counts": dict(v1_1_counts),
        "canary_v1_class_counts": dict(Counter(r["v1_class"] for r in canary)),
        "canary_v1_1_class_counts": dict(Counter(r["class_v1_1"] for r in canary)),
        "changed": sum(1 for r in labels.values()
                       if r["slug"] and r["v1_class"] != r["class_v1_1"]),
    }


def repeat_consistency(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    judged = [r for r in rows if r["repeat"] and r["repeat"]["agrees"] is not None]
    exact = sum(1 for r in judged if r["repeat"]["agrees"])
    claim_same = sum(1 for r in judged if r["repeat"]["claim_agrees"])
    return {
        "pairs": len(rows),
        "both_answered": len(judged),
        "exact_agreement": exact,
        "claim_axis_agreement": claim_same,
        "exact_rate": (exact / len(judged)) if judged else None,
        "claim_axis_rate": (claim_same / len(judged)) if judged else None,
        "disagreements": [
            {"card": r["adjudication_card_id"], "item": r["catalog_item_id"],
             "primary": r["slug"], "repeat": r["repeat"]["slug"]}
            for r in judged if not r["repeat"]["agrees"]
        ],
    }


def retag_agreement(labels: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_answer: Dict[str, Counter] = defaultdict(Counter)
    for row in labels.values():
        if not (row["retag_answer"] and row["slug"]):
            continue
        met = row["retag_expectation_met"]
        per_answer[row["retag_answer"]][
            "no_expectation" if met is None else ("met" if met else "diverged")] += 1
        per_answer[row["retag_answer"]][f"-> {row['claim']}/{row['work']}"] += 1
    return {k: dict(v) for k, v in sorted(per_answer.items())}


# ------------------------------------------------------------------- rendering

def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def render(a: Dict[str, Any]) -> str:
    cov, mov = a["coverage"], a["movement"]
    L = [f"# Labels {a['label_version']} — corrected label overlay", "",
         "v1 is unchanged and remains the published result. This overlay is the",
         "development label source; every number below is printed against its v1",
         "original.", "",
         "## Coverage", ""]
    L += _table(["arm", "cards", "answered", "pending"],
                [[arm, c.get("total", 0), c.get("answered", 0), c.get("pending", 0)]
                 for arm, c in cov["per_arm"].items()])
    L += ["", f"Canary label set complete: **{cov['canary_complete']}** "
              f"({cov['answered']}/{cov['cards']} cards answered overall).", ""]

    if not cov["answered"]:
        L += ["No adjudication answers recorded yet — the scaffolding above is",
              "the queue's shape, not a result.", ""]
        return "\n".join(L)

    L += ["## Class movement (v1 -> v1.1)", ""]
    L += _table(["v1 class", "-> v1.1 class", "cards"],
                [[v1, v11, n] for v1, row in mov["matrix"].items()
                 for v11, n in sorted(row.items(), key=lambda x: -x[1])])
    L += ["", f"{mov['changed']} of {cov['answered']} adjudicated cards changed class.", ""]

    L += ["## Canary scoreable population — what Session F will tune against", ""]
    old, new = mov["canary_v1_class_counts"], mov["canary_v1_1_class_counts"]
    names = sorted(set(old) | set(new))
    L += _table(["class", "v1", "v1.1"], [[n, old.get(n, "—"), new.get(n, "—")]
                                          for n in names])
    hard_old, hard_new = old.get("hard_false_billed", 0), new.get("hard_false_billed", 0)
    L += ["", f"**Hard-false canary cards: {hard_old} -> {hard_new}.** This is the "
              "win-opportunity count the variant scorecard depends on. Redo Session "
              "B's control-arm sample-size note against it before spending Terra "
              "tokens: against a 6.4% replica-flip floor, a smaller class means a "
              "smaller separation between arm and control.", ""]

    rc_ = a["repeat_consistency"]
    L += ["## Reviewer self-consistency (the repeat arm)", ""]
    if rc_["both_answered"]:
        L += [f"- exact agreement: **{rc_['exact_agreement']}/{rc_['both_answered']}** "
              f"({rc_['exact_rate']:.0%})",
              f"- claim-axis agreement: **{rc_['claim_axis_agreement']}/"
              f"{rc_['both_answered']}** ({rc_['claim_axis_rate']:.0%})",
              "",
              "This is the human noise floor. Every rate in this program — and the "
              "6.4% Terra replica floor it is compared against — should be read "
              "next to it.", ""]
        if rc_["disagreements"]:
            L += _table(["card", "item", "primary", "repeat"],
                        [[d["card"], d["item"], d["primary"], d["repeat"]]
                         for d in rc_["disagreements"]]) + [""]
    else:
        L += ["Not yet measurable — no repeat pair has both answers.", ""]

    L += ["## Re-tag agreement (validation, not a label source)", ""]
    rows = []
    for answer, counts in a["retag_agreement"].items():
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())
                             if k.startswith("-> "))
        rows.append([answer, counts.get("met", 0), counts.get("diverged", 0),
                     counts.get("no_expectation", 0), outcomes])
    L += _table(["re-tag answer", "met", "diverged", "n/a", "v1.1 claim/work"], rows)
    L += ["", "Divergence is expected on `mechanism_only`: its own notes split "
              "between claims that were actually right and claims that were "
              "misnamed. That split is why the re-tag answers were re-asked "
              "rather than translated.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=ADJ_QUEUE)
    ap.add_argument("--verdicts", type=Path, default=ADJ_VERDICTS)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args(argv)

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    answers = rc.latest_verdicts(args.verdicts)
    payload = build(queue, answers)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                             encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")

    cov = payload["coverage"]
    print(f"labels: {args.out_json}")
    print(f"report: {args.out_md}")
    print(f"  {cov['answered']}/{cov['cards']} adjudicated, "
          f"canary complete={cov['canary_complete']}")
    if cov["answered"]:
        print(f"  class changes: {payload['movement']['changed']}")
        rcs = payload["repeat_consistency"]
        if rcs["both_answered"]:
            print(f"  self-consistency: {rcs['exact_agreement']}/{rcs['both_answered']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
