"""Tally the manual review verdicts: reports/review_queue.json x reports/review_verdicts.jsonl
-> reports/review_tally.md (+ optional neutral-label fill of a COPY of the Session 9 worksheet).

Only condition-level comparisons are derived (human label vs Terra's own
verdict). Nothing is derived about Pass 2f; direction A/B are strata.

Run:
  .venv\\Scripts\\python.exe scripts\\review_tally.py [--export-legacy]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import review_cards as rc  # noqa: E402

QUEUE = ROOT / "reports" / "review_queue.json"
VERDICTS = ROOT / "reports" / "review_verdicts.jsonl"
OUT = ROOT / "reports" / "review_tally.md"
WORKSHEET = ROOT / "reports" / "session9_decision_worksheet.json"
WORKSHEET_FILLED = ROOT / "reports" / "session9_decision_worksheet.filled.json"

V_SUP, V_UNS, V_OVER, V_INC = rc.CONDITION_VERDICTS


def terra_vs_photo(human: str, terra: Optional[str]) -> str:
    """Condition-level only: the human label against Terra's verdict on the same claim."""
    if human == V_SUP:
        return "agree" if terra == "supported" else "miss"
    if human == V_UNS:
        return "false_positive" if terra == "supported" else "agree"
    return "overstated" if human == V_OVER else "inconclusive"


def table(headers: List[str], rows: Iterable[Iterable[Any]]) -> str:
    rows = [[str(x) for x in r] for r in rows]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.0f}%" if d else "—"


def verdict_row(label: str, recs: List[Dict[str, Any]]) -> List[Any]:
    c = Counter(r["verdict"] for r in recs)
    n = len(recs)
    return [label, n, c[V_SUP], c[V_UNS], c[V_OVER], c[V_INC], pct(c[V_UNS], n), pct(c[V_UNS] + c[V_OVER], n)]


VH = ["group", "n", "supported", "unsupported", "overstated", "inconclusive", "unsupported %", "unsup+over %"]


def weighted(meta_src: Dict[str, Any], dirA: List[Dict[str, Any]], uni: List[Dict[str, Any]], which: tuple) -> List[Any]:
    nA, nU = len(dirA), len(uni)
    rA = sum(1 for r in dirA if r["verdict"] in which) / nA if nA else None
    rU = sum(1 for r in uni if r["verdict"] in which) / nU if nU else None
    NA, NN, NT = meta_src.get("accepted_dirA", 0), meta_src.get("accepted_non_dirA", 0), meta_src.get("accepted", 0)
    overall = (NA * rA + NN * rU) / NT if (rA is not None and rU is not None and NT) else None
    f = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    return [NT, NA, f"{nA}", f(rA), NN, f"{nU}", f(rU), f(overall)]


def build_report(queue: Dict[str, Any], done: Dict[str, Dict[str, Any]]) -> str:
    cards = {c["card_id"]: c for c in queue["cards"]}
    recs = []  # joined records
    for cid, v in done.items():
        c = cards.get(cid)
        if c:
            recs.append({**v, "card": c})
    cond = [r for r in recs if r["card"]["kind"] == "condition"]
    L = [f"# Review tally — {len(recs)} of {len(cards)} cards reviewed\n"]
    L.append("Human labels judge the claim against the photo. Only Terra is compared at condition level; "
             "nothing is derived about Pass 2f (direction A/B are sampling strata). Package warrant is judged on its own cards.\n")
    orphans = sum(1 for cid in done if cid not in cards)
    if orphans:
        L.append(f"{orphans} verdict(s) refer to cards no longer in the queue (e.g. low-res exclusions); they are ignored above.\n")
    lr = {src: m.get("low_res_excluded") for src, m in (queue.get("meta") or {}).items() if m.get("low_res_excluded")}
    if lr:
        L.append(f"Excluded from the queue and from all denominators — evidence entirely below {rc.MIN_EVIDENCE_PX}px short side: {lr}\n")

    # 1. progress per stratum
    rows = []
    for s in rc.STRATUM_ORDER:
        N = sum(1 for c in cards.values() if s in c["strata"])
        n = sum(1 for r in recs if s in r["card"]["strata"])
        bysrc = Counter(c["source"] for c in cards.values() if s in c["strata"])
        rows.append([s, f"{n} of {N}", pct(n, N), dict(bysrc)])
    L.append("## 1. Progress per stratum\n\n" + table(["stratum", "reviewed", "%", "cards by source"], rows))

    # 2. condition verdict mix per stratum
    rows = [verdict_row(s, [r for r in cond if s in r["card"]["strata"]]) for s in ("dirA", "terra_flip", "uniform", "dirB", "p6_forced_single")]
    L.append("## 2. Condition verdicts by stratum\n\n" + table(VH, rows))

    # 3. terra_vs_photo per stratum
    rows = []
    for s in ("dirA", "terra_flip", "uniform", "dirB", "p6_forced_single"):
        rs = [r for r in cond if s in r["card"]["strata"]]
        c = Counter(terra_vs_photo(r["verdict"], r["card"]["meta"].get("terra_verdict")) for r in rs)
        rows.append([s, len(rs), c["agree"], c["false_positive"], c["miss"], c["overstated"], c["inconclusive"]])
    L.append("## 3. Terra verdict vs photo (condition level)\n\n" + table(["stratum", "n", "agree", "Terra false positive", "Terra miss", "overstated", "inconclusive"], rows))

    # 4. weighted overall false-billed rate per source
    rows = []
    for src, m in sorted((queue.get("meta") or {}).items()):
        dA = [r for r in cond if r["card"]["source"] == src and "dirA" in r["card"]["strata"] and r["card"]["meta"].get("accepted")]
        un = [r for r in cond if r["card"]["source"] == src and "uniform" in r["card"]["strata"]]
        rows.append([src, "unsupported"] + weighted(m, dA, un, (V_UNS,)))
        rows.append([src, "unsup+over"] + weighted(m, dA, un, (V_UNS, V_OVER)))
    L.append("## 4. Weighted overall rate on billed (accepted) conditions\n\n"
             "overall = (N_A·r_A + N_nonA·r_nonA) / N_accepted — dirA at 100 %, the rest via the uniform sample.\n\n"
             + table(["source", "measure", "N_accepted", "N_A", "n_A reviewed", "r_A", "N_nonA", "n_uniform reviewed", "r_nonA", "overall"], rows))

    # 5. breakdowns over Terra-supported claims (the false-positive question)
    sup = [r for r in cond if r["card"]["meta"].get("terra_verdict") == "supported"]

    def bucket_rows(keyf, title):
        g = defaultdict(list)
        for r in sup:
            g[keyf(r["card"]["meta"])].append(r)
        return f"### {title}\n\n" + table(VH, [verdict_row(str(k), v) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))])

    def batch_bucket(n, edges):
        if n is None:
            return "unknown"
        for lo, hi, lab in edges:
            if lo <= n <= hi:
                return lab
        return "unknown"

    cb = [(1, 1, "1"), (2, 4, "2-4"), (5, 8, "5-8"), (9, 999, "9+")]
    ib = [(1, 1, "1"), (2, 3, "2-3"), (4, 999, "4+")]
    L.append("## 5. Breakdowns — Terra-supported claims only\n\n"
             + bucket_rows(lambda m: m.get("catalog_kind"), "by catalog kind")
             + bucket_rows(lambda m: "1 photo" if (m.get("photo_count") or 0) <= 1 else "2+ photos", "by photo count")
             + bucket_rows(lambda m: m.get("second_opinion"), "by second opinion (Pass 2f)")
             + bucket_rows(lambda m: m.get("terra_batch_conditions") and batch_bucket(m.get("terra_batch_conditions"), cb), "by Terra batch size (conditions per call)")
             + bucket_rows(lambda m: m.get("terra_batch_images") and batch_bucket(m.get("terra_batch_images"), ib), "by Terra batch images (per call)"))
    g = defaultdict(list)
    for r in sup:
        g[r["card"]["meta"].get("catalog_item_id")].append(r)
    items = [(k, v) for k, v in g.items() if len(v) >= 3]
    items.sort(key=lambda kv: (-sum(1 for r in kv[1] if r["verdict"] in (V_UNS, V_OVER)) / len(kv[1]), -len(kv[1])))
    L.append("### by catalog item (n ≥ 3, worst first)\n\n" + table(VH, [verdict_row(k, v) for k, v in items]))
    g = defaultdict(list)
    for r in sup:
        g[r["card"]["source"]].append(r)
    L.append("### by source\n\n" + table(VH, [verdict_row(k, v) for k, v in sorted(g.items())]))

    # 6. error tags, peeks, flips
    tags = Counter(r.get("tag") for r in cond if r.get("tag"))
    L.append("## 6. Error types (tags)\n\n" + table(["tag", "n"], sorted(tags.items(), key=lambda kv: -kv[1])))
    peeks = sum(1 for r in recs if r.get("peeked"))
    L.append(f"Revealed before verdict (peeked): {peeks} of {len(recs)}.\n")
    fl = [r for r in cond if "terra_flip" in r["card"]["strata"]]
    rows = []
    for r in fl:
        m = r["card"]["meta"]
        human_sup = r["verdict"] == V_SUP
        match = "run_1" if (m.get("terra_verdict") == "supported") == human_sup else "run_2" if (m.get("replica_terra_verdict") == "supported") == human_sup else "neither"
        if r["verdict"] in (V_OVER, V_INC):
            match = "n/a"
        rows.append([r["card"]["property_key"], m.get("catalog_item_id"), m.get("terra_verdict"), m.get("replica_terra_verdict"), r["verdict"], match])
    L.append("## 7. Terra self-flips — which replica matched the photo\n\n" + table(["property", "item", "run_1", "run_2", "human", "matched"], rows)
             + f"\n{Counter(r[-1] for r in rows)}\n")

    # 8. P1 / P3
    p1 = [r for r in recs if r["card"]["kind"] == "package"]
    L.append("## 8. P1 packages\n\n" + table(["property", "package", "Sol", "2f", "human", "tag/notes", "legacy"],
             [[r["card"]["property_key"], r["card"]["title"], r["card"]["meta"].get("v5_sol_decision"), r["card"]["meta"].get("v4_2f_status"),
               r["verdict"], r.get("notes") or "", r["card"].get("legacy_item_id") or ""] for r in p1])
             + f"\n{Counter(r['verdict'] for r in p1)}\n")
    p3 = [r for r in recs if r["card"]["kind"] == "bathroom"]
    L.append("## 9. P3 bathrooms\n\n" + table(["property", "listing baths", "v4 surrogates", "v5 units", "human distinct", "billing", "notes", "legacy"],
             [[r["card"]["property_key"], r["card"]["meta"].get("listing_baths"), r["card"]["meta"].get("surrogates"),
               ", ".join(r["card"]["meta"].get("v5_bath_units") or []), (r.get("extra") or {}).get("distinct_bathrooms"), r["verdict"],
               r.get("notes") or "", r["card"].get("legacy_item_id") or ""] for r in p3]))
    return "\n".join(L)


def export_legacy(queue: Dict[str, Any], done: Dict[str, Dict[str, Any]], worksheet: Path, out: Path) -> int:
    """Fill a COPY of the Session 9 worksheet with the neutral labels (original untouched)."""
    ws = json.loads(Path(worksheet).read_text(encoding="utf-8"))
    ws.setdefault("instructions", {})
    ws["instructions"]["p2_conditions"] = ("verdict: " + " | ".join(rc.CONDITION_VERDICTS)
                                           + " (judged against the photo; no Pass 2f correctness implied)")
    ws["instructions"]["filled_from"] = "scripts/review_tally.py --export-legacy; notes = [tag] notes"
    cards = {c["card_id"]: c for c in queue["cards"]}
    n = 0
    for cid, v in done.items():
        c = cards.get(cid)
        lid = c and c.get("legacy_item_id")
        if not lid:
            continue
        notes = " ".join(x for x in (f"[{v['tag']}]" if v.get("tag") else "", v.get("notes") or "") if x)
        if lid.startswith("C") and lid in ws.get("p2_conditions", {}):
            ws["p2_conditions"][lid].update({"verdict": v["verdict"], "notes": notes})
        elif lid.startswith("P") and lid in ws.get("p1_packages", {}):
            ws["p1_packages"][lid].update({"verdict": v["verdict"], "notes": notes})
        elif lid.startswith("B") and lid in ws.get("p3_bathrooms", {}):
            ws["p3_bathrooms"][lid].update({"distinct_bathrooms": (v.get("extra") or {}).get("distinct_bathrooms", ""),
                                            "bill": v["verdict"], "notes": notes})
        else:
            continue
        n += 1
    Path(out).write_text(json.dumps(ws, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=QUEUE)
    ap.add_argument("--verdicts", type=Path, default=VERDICTS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--export-legacy", action="store_true")
    ap.add_argument("--worksheet", type=Path, default=WORKSHEET)
    ap.add_argument("--worksheet-out", type=Path, default=WORKSHEET_FILLED)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    done = rc.latest_verdicts(args.verdicts)
    report = build_report(queue, done)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {args.out}")
    if args.export_legacy:
        n = export_legacy(queue, done, args.worksheet, args.worksheet_out)
        print(f"legacy export: {n} slots filled -> {args.worksheet_out} (original untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
