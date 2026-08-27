"""Descriptive analysis of the completed manual v5 output-quality review.

Reads the frozen review inputs (reports/review_queue.json, reports/review_verdicts.jsonl)
plus the canary and production artifacts they were built from, and writes

  reports/review_analysis.md    human-readable report
  reports/review_analysis.json  machine-readable mirror (raw floats, no formatting)

The report is descriptive only: canary and production are never pooled into one
headline, nothing about Pass 2f correctness is derived from condition verdicts,
and every rate carries its numerator, denominator, unique-listing count and an
evidence-strength flag. docs/DESIGN_review_method.md stays the semantic
authority. All inputs are read-only; regenerating reports/review_tally.md is a
separate manual step (scripts/review_tally.py).

Run:
  .venv\\Scripts\\python.exe scripts\\review_analysis.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.review_tally import table, terra_vs_photo  # noqa: E402
from tools import review_cards as rc  # noqa: E402
from tools.pass_2f_artifact_inputs import photo_key_to_path  # noqa: E402
from tools.quant_artifact_comparison import wilson  # noqa: E402

QUEUE = ROOT / "reports" / "review_queue.json"
VERDICTS = ROOT / "reports" / "review_verdicts.jsonl"
OUT_MD = ROOT / "reports" / "review_analysis.md"
OUT_JSON = ROOT / "reports" / "review_analysis.json"

V_SUP, V_UNS, V_OVER, V_INC = rc.CONDITION_VERDICTS
MEASURES = {"hard_false": (V_UNS,), "broad": (V_UNS, V_OVER), "inconclusive": (V_INC,)}
COND_BATCH_BUCKETS = ((1, 1, "1"), (2, 4, "2-4"), (5, 8, "5-8"), (9, 999, "9+"))  # data mirror of review_tally.py:140
IMG_BATCH_BUCKETS = ((1, 1, "1"), (2, 3, "2-3"), (4, 999, "4+"))  # data mirror of review_tally.py:141
EXPLORATORY_MIN_JUDGED = 10
EXPLORATORY_MIN_LISTINGS = 3
CI_METHOD = ("95% Wilson interval on the uniform-arm sample proportion, propagated linearly through the "
             "weighting identity with the dirA arm held fixed (complete census); not a full-design interval.")

THEMES = ("claim_wording", "evidence_ingest", "object_room_mismatch", "package_coherence", "room_identity", "other")
# Auditable coding of every free-text note (latest-wins record per card) into one theme.
# Coded 2026-08-26 by the analysis session from the note text alone; the raw note is
# echoed next to the theme in the report so the coding can be re-derived or disputed.
NOTE_THEMES: Dict[str, str] = {
    "rc_025ca614449b": "claim_wording",        # ceiling discoloration is reflected light; casing part true
    "rc_128caa6212b8": "object_room_mismatch", # hardwood, not vinyl/linoleum; wear itself real
    "rc_42f54b0e00fd": "other",                # claim false but hallucination judged understandable
    "rc_45b9d2f6923c": "object_room_mismatch", # backsplash read as wallpaper; dated look real
    "rc_48a619175251": "claim_wording",        # intentional yellow paint read as yellowed/aged
    "rc_61b0f2981ca9": "claim_wording",        # open vanity read as unfinished
    "rc_6285e9a79e45": "claim_wording",        # wear overstated; outdated rather than worn
    "rc_62d0dc3b9df8": "object_room_mismatch", # drop ceiling read as popcorn texture
    "rc_63fac87a14e4": "evidence_ingest",      # photo too dark to judge
    "rc_65f169bbe3b4": "object_room_mismatch", # wood cabinets read as wall paneling
    "rc_6c4e3194c4c9": "claim_wording",        # paint/discoloration read as broken/warped flooring
    "rc_6f8278034b05": "claim_wording",        # true, but should be a generic-presence claim
    "rc_7057c3c171e5": "claim_wording",        # trim missing, not scuffed
    "rc_7b7343ba4892": "package_coherence",    # indoor photo inside an exterior package
    "rc_aa7dae68eb01": "object_room_mismatch", # not carpet; replacement still needed
    "rc_ceeb17f8e242": "claim_wording",        # cabinets dated but replacement overstated
    "rc_cf0c3ee00fcb": "claim_wording",        # worn/uneven false; dated colour is the real issue
    "rc_d36f87e74356": "room_identity",        # one surrogate is a utility room, not a bathroom
    "rc_d58b1f530fd9": "claim_wording",        # dents, not peeling paint; damage real
    "rc_df5b348c6ea6": "claim_wording",        # claim understates the door condition
    "rc_e2ad788f0845": "object_room_mismatch", # photo shows the shed, not the house
    "rc_e9d27cf6e9aa": "claim_wording",        # outdated fits better than worn
    "rc_eef1daf73e2a": "claim_wording",        # stains read as scuffs
    "rc_f468f4066e3f": "claim_wording",        # true but minor as described
    "rc_f48a8d9f18d3": "room_identity",        # claim about a bathroom barely visible in a living-room photo
    "rc_f8baa9e1dd45": "other",                # confirms three bathrooms
    "rc_fb876c9b1b2d": "claim_wording",        # overgrowth real but plausibly intentional
}

NOT_SHOWN = [
    "Overall Terra recall (miss rate on rejected claims): rejected claims were sampled only through the "
    "targeted dirB stratum (2f-confirmed cases); no uniform sample of the rejected population exists.",
    "Overall Sol or package-pipeline accuracy: the P1 cards are a census of one disagreement slice "
    "(2f-rejected, v5 Sol-decided), not a population sample of packages.",
    "Pass 2f correctness: condition verdicts judge Terra's condition-level claim only; direction A/B are "
    "sampling strata, and no code path here joins a condition verdict to a 2f correctness label.",
    "Causal effects of Terra batch size: batch buckets are observational and confounded with listing "
    "photo volume; the breakdowns are associations only.",
    "Pricing accuracy or renovation cost: prices are provisional and were hidden during review.",
    "Standalone-observation usefulness: it was not directly labeled; the packaging-flow table is a "
    "utilization proxy, not a human judgment of usefulness.",
]

GROUPINGS = {
    "catalog_kind": lambda m: str(m.get("catalog_kind")),
    "catalog_item": lambda m: str(m.get("catalog_item_id")),
    "photo_bucket": lambda m: "1 photo" if (m.get("photo_count") or 0) <= 1 else "2+ photos",
    "second_opinion": lambda m: str(m.get("second_opinion")),
    "batch_conditions": lambda m: bucket(m.get("terra_batch_conditions"), COND_BATCH_BUCKETS),
    "batch_images": lambda m: bucket(m.get("terra_batch_images"), IMG_BATCH_BUCKETS),
}


# --------------------------------------------------------------------------- freeze / audit

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_verdicts(path: Path) -> Dict[str, Any]:
    """Raw pass over the JSONL, independent of rc.latest_verdicts (then cross-checked)."""
    lines = invalid = undos = peeked = 0
    per_card: Dict[str, List[Dict[str, Any]]] = {}
    ts_min = ts_max = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        lines += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(rec, dict) or not rec.get("card_id"):
            invalid += 1
            continue
        if rec.get("verdict") is None:
            undos += 1
        if rec.get("peeked"):
            peeked += 1
        ts = rec.get("ts")
        if ts:
            ts_min = ts if ts_min is None or ts < ts_min else ts_min
            ts_max = ts if ts_max is None or ts > ts_max else ts_max
        per_card.setdefault(rec["card_id"], []).append(rec)
    identical = changed = 0
    changed_cards = []
    for cid, rs in sorted(per_card.items()):
        eff = [r for r in rs if r.get("verdict") is not None]
        if len(eff) <= 1:
            continue
        keys = {(r.get("verdict"), r.get("tag"), r.get("notes"), json.dumps(r.get("extra"), sort_keys=True)) for r in eff}
        if len(keys) == 1:
            identical += 1
        else:
            changed += 1
            changed_cards.append({"card_id": cid,
                                  "records": [{"verdict": r.get("verdict"), "tag": r.get("tag"), "ts": r.get("ts")} for r in eff]})
    # cross-check latest-wins against rc.latest_verdicts
    expect: Dict[str, Dict[str, Any]] = {}
    for cid, rs in per_card.items():
        cur = None
        for r in rs:
            cur = None if r.get("verdict") is None else r
        if cur is not None:
            expect[cid] = cur
    latest = rc.latest_verdicts(path)
    return {"lines": lines, "invalid_lines": invalid, "undos": undos, "peeked_records": peeked,
            "distinct_ids": len(per_card), "net_verdicts": len(latest),
            "overwrites": {"identical": identical, "changed": changed, "changed_cards": changed_cards},
            "ts_range": [ts_min, ts_max], "latest_crosscheck_ok": latest == expect, "latest": latest}


def load_listings(canary_root: Optional[Path] = rc.CANARY_ROOT, prod_root: Optional[Path] = rc.PROD_ROOT,
                  since: Optional[str] = rc.DEFAULT_SINCE) -> List[Dict[str, Any]]:
    """Mirror of rc.build_queue's listing assembly, keeping artifact paths for the manifest."""
    listings: List[Dict[str, Any]] = []
    if canary_root and Path(canary_root).is_dir():
        run1, run2 = rc.load_canary(Path(canary_root))
        for prop, (path, art) in sorted(run1.items()):
            rpath, rart = run2.get(prop) or (None, None)
            listings.append({"source": "canary", "property_key": prop, "run_id": path.parent.name,
                             "artifact": art, "replica_artifact": rart,
                             "path": str(path), "replica_path": str(rpath) if rpath else None})
    if prod_root and Path(prod_root).is_dir():
        for prop, run_id, path, art in rc.iter_runs(Path(prod_root), since):
            listings.append({"source": "production", "property_key": prop, "run_id": run_id,
                             "artifact": art, "replica_artifact": None, "path": str(path), "replica_path": None})
    return listings


def listing_manifest(listings: List[Dict[str, Any]], carded_props: set) -> List[Dict[str, Any]]:
    out = []
    for L in listings:
        row = {"source": L["source"], "property_key": L["property_key"], "run_id": L["run_id"],
               "artifact_path": L.get("path"), "artifact_sha256": None, "replica_sha256": None,
               "carded": L["property_key"] in carded_props}
        for field, key in (("artifact_sha256", "path"), ("replica_sha256", "replica_path")):
            p = L.get(key)
            if p and Path(p).is_file():
                row[field] = sha256_file(Path(p))
        out.append(row)
    return out


# --------------------------------------------------------------------------- population

def enumerate_population(listings: List[Dict[str, Any]], uniform_rate: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]:
    """In-memory 100%-accepted-condition view, mirroring rc.build_queue_from_listings'
    condition loop (including the MIN_EVIDENCE_PX exclusion) without building or
    writing any queue. Returns (rows, per_source_meta, ctx); ctx caches each parsed
    artifact for the lineage / package / bathroom / spot-check steps."""
    try:
        import PIL  # noqa: F401
    except ImportError:  # unknown dims would silently un-exclude low-res evidence
        raise RuntimeError("Pillow is required: low-res exclusion parity with the queue builder needs photo dimensions")
    rows: List[Dict[str, Any]] = []
    per_source: Dict[str, Dict[str, Any]] = {}
    ctx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for L in listings:
        source, prop, run_id, art = L["source"], L["property_key"], L["run_id"], L["artifact"]
        res = rc.v5_result(art)
        if res is None:
            continue
        st = per_source.setdefault(source, {"listings": 0, "accepted": 0, "accepted_dirA": 0,
                                            "uniform": 0, "low_res": Counter()})
        st["listings"] += 1
        idx = rc.index_result(res)
        v4 = art.get("renovation_estimate_v4") or {}
        f2 = rc.join_2f(v4, idx)
        paths = photo_key_to_path(art)
        needed = set()
        for ev in res.get("evidence_facts") or []:
            needed.update(ev.get("photo_keys") or [])
        for pk in v4.get("package_candidates") or []:
            needed.update(pk.get("review_photo_keys") or [])
        for s in rc._bath_surrogates(v4):
            needed.update(s.get("photo_keys") or [])
        dims = rc.photo_dims({k: p for k, p in paths.items() if k in needed})
        res2 = rc.v5_result(L.get("replica_artifact"))
        flips = {cid: (v1, v2) for cid, v1, v2 in rc.terra_flips(res, res2)}
        ctx[(source, prop)] = {"idx": idx, "res": res, "res2": res2, "v4": v4, "art": art, "run_id": run_id,
                               "paths": paths, "dims": dims, "f2": f2, "flips": flips,
                               "amap": absorbed_map(res)}
        for cid, c in idx["conds"].items():
            r = idx["revs"].get(cid) or {}
            d = idx["disps"].get(cid) or {}
            dir_ = rc.direction(r.get("verdict"), {row[1] for row in (f2.get(cid) or {}).get("rows", [])})
            accepted = r.get("verdict") == "supported" and d.get("disposition") == "accepted_for_work"
            ev = idx["evs"].get(cid) or {}
            if rc.all_low_res(dims, ev.get("photo_keys") or []):
                st["low_res"]["conditions"] += 1
                if accepted:
                    st["low_res"]["accepted"] += 1
                continue
            if not accepted:
                continue
            st["accepted"] += 1
            is_dirA = dir_ in ("A", "mixed")
            cd = rc.card_id(source, prop, run_id, "condition", cid)
            uniform_pick = False
            if is_dirA:
                st["accepted_dirA"] += 1
            elif rc.uniform_included(cd, uniform_rate):
                uniform_pick = True
                st["uniform"] += 1
            n_batch, n_imgs = rc.terra_batch(idx, cid)
            rows.append({"source": source, "property_key": prop, "run_id": run_id, "condition_id": cid,
                         "card_id": cd, "catalog_item_id": c.get("catalog_item_id"),
                         "catalog_kind": c.get("catalog_kind"),
                         "photo_count": int(ev.get("distinct_photo_count") or len(ev.get("photo_keys") or []) or 0),
                         "second_opinion": rc.SECOND_OPINION[dir_],
                         "terra_batch_conditions": n_batch, "terra_batch_images": n_imgs,
                         "is_dirA": is_dirA, "uniform_pick": uniform_pick})
        # package / bathroom low-res parity, using the builder's own card construction
        for pk in v4.get("package_candidates") or []:
            if pk.get("verification_status") != "rejected":
                continue
            ptype, unit = pk.get("package_type"), pk.get("estimate_unit_id") or ""
            cands = [x for x in res.get("package_candidates") or []
                     if x.get("package_type") == ptype and x.get("estimate_unit_id") == unit]
            if not cands:
                continue
            card = rc.package_card(source=source, prop=prop, run_id=run_id, idx=idx, v4_pk=pk,
                                   cand=cands[0], res=res, paths=paths, dims=dims)
            if rc.all_low_res(dims, [p["key"] for s in card["strips"] for p in s["photos"]]):
                st["low_res"]["packages"] += 1
        surr = rc._bath_surrogates(v4)
        if len(surr) >= 2:
            card = rc.bathroom_card(source=source, prop=prop, run_id=run_id, art=art, idx=idx,
                                    res=res, surr=surr, paths=paths, dims=dims)
            if rc.all_low_res(dims, [p["key"] for s in card["strips"] for p in s["photos"]]):
                st["low_res"]["bathrooms"] += 1
    meta = {src: {"listings": v["listings"], "accepted": v["accepted"], "accepted_dirA": v["accepted_dirA"],
                  "accepted_non_dirA": v["accepted"] - v["accepted_dirA"], "uniform": v["uniform"],
                  "low_res_excluded": dict(v["low_res"])}
            for src, v in per_source.items()}
    return rows, meta, ctx


def reconcile(enum_meta: Dict[str, Any], queue_meta: Dict[str, Any]) -> Dict[str, Any]:
    fields = ("listings", "accepted", "accepted_dirA", "accepted_non_dirA", "low_res_excluded")
    out: Dict[str, Any] = {"ok": True, "sources": {}}
    for src in sorted(set(enum_meta) | set(queue_meta)):
        exp = {f: (queue_meta.get(src) or {}).get(f) for f in fields}
        exp["uniform"] = ((queue_meta.get(src) or {}).get("strata") or {}).get("uniform", 0)
        got = {f: (enum_meta.get(src) or {}).get(f) for f in fields}
        got["uniform"] = (enum_meta.get(src) or {}).get("uniform", 0)
        ok = exp == got
        out["sources"][src] = {"expected": exp, "enumerated": got, "ok": ok}
        out["ok"] = out["ok"] and ok
    return out


# --------------------------------------------------------------------------- estimation

def bucket(n: Optional[int], edges: Iterable[Tuple[int, int, str]]) -> str:
    if n is None:
        return "unknown"
    for lo, hi, lab in edges:
        if lo <= n <= hi:
            return lab
    return "unknown"


def count_in(recs: List[Dict[str, Any]], which: Tuple[str, ...]) -> int:
    return sum(1 for r in recs if r["verdict"] in which)


def weighted_estimate(N_A: int, N_nonA: int, x_A: int, n_A: int, x_U: int, n_U: int) -> Dict[str, Any]:
    """The weighting identity of record: overall = (N_A*r_A + N_nonA*r_U) / (N_A + N_nonA).
    dirA is a census arm (n_A must equal N_A); the uniform arm carries the sampling error,
    so the reported band propagates the uniform arm's Wilson interval only (CI_METHOD)."""
    est: Dict[str, Any] = {"N": N_A + N_nonA, "N_A": N_A, "N_nonA": N_nonA,
                           "x_A": x_A, "n_A": n_A, "x_U": x_U, "n_U": n_U,
                           "r_A": x_A / n_A if n_A else None, "r_U": x_U / n_U if n_U else None,
                           "ci_A": wilson(x_A, n_A), "ci_U": wilson(x_U, n_U),
                           "point": None, "low": None, "high": None, "flags": []}
    if n_A != N_A:
        est["flags"].append("census_gap")
    if not est["N"]:
        return est
    if N_A and est["r_A"] is None:
        est["flags"].append("no_dirA_coverage")
        return est
    a_mass = N_A * (est["r_A"] or 0.0)
    if N_nonA == 0:
        est["point"] = est["low"] = est["high"] = a_mass / est["N"]
    elif est["r_U"] is None:
        est["flags"].append("no_uniform_coverage")
    else:
        est["point"] = (a_mass + N_nonA * est["r_U"]) / est["N"]
        est["low"] = (a_mass + N_nonA * est["ci_U"]["low"]) / est["N"]
        est["high"] = (a_mass + N_nonA * est["ci_U"]["high"]) / est["N"]
    return est


def arm_records(cond: List[Dict[str, Any]], source: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dirA = [r for r in cond if r["card"]["source"] == source and "dirA" in r["card"]["strata"]
            and r["card"]["meta"].get("accepted")]
    uni = [r for r in cond if r["card"]["source"] == source and "uniform" in r["card"]["strata"]]
    return dirA, uni


def label_mix(recs: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter(r["verdict"] for r in recs)
    return {"n": len(recs), "supported": c[V_SUP], "unsupported": c[V_UNS],
            "overstated": c[V_OVER], "inconclusive": c[V_INC]}


def headline(cond: List[Dict[str, Any]], queue_meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for src, m in sorted(queue_meta.items()):
        dirA, uni = arm_records(cond, src)
        listings = len({r["card"]["property_key"] for r in dirA + uni})
        weighted = {}
        for name, which in MEASURES.items():
            weighted[name] = weighted_estimate(m.get("accepted_dirA", 0), m.get("accepted_non_dirA", 0),
                                               count_in(dirA, which), len(dirA), count_in(uni, which), len(uni))
        out[src] = {"label_mix": {"dirA_accepted": label_mix(dirA), "uniform": label_mix(uni)},
                    "weighted": weighted, "listings": listings}
    return out


def subgroups(pop_rows: List[Dict[str, Any]], cond: List[Dict[str, Any]], queue_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Denominator-weighted subgroup rates + error-contribution shares, per source and grouping."""
    out: Dict[str, Any] = {}
    for src in sorted(queue_meta):
        pop_src = [p for p in pop_rows if p["source"] == src]
        dirA, uni = arm_records(cond, src)
        src_out: Dict[str, Any] = {}
        for gname, keyf in GROUPINGS.items():
            pop_g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for p in pop_src:
                pop_g[keyf(p)].append(p)
            dA_g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in dirA:
                dA_g[keyf(r["card"]["meta"])].append(r)
            un_g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in uni:
                un_g[keyf(r["card"]["meta"])].append(r)
            total_mass = {name: 0.0 for name in ("hard_false", "broad")}
            labels: Dict[str, Any] = {}
            for lab in sorted(set(pop_g) | set(dA_g) | set(un_g)):
                pop_l = pop_g.get(lab, [])
                N_A_g = sum(1 for p in pop_l if p["is_dirA"])
                N_non_g = len(pop_l) - N_A_g
                dA, un = dA_g.get(lab, []), un_g.get(lab, [])
                judged = len(dA) + len(un)
                listings = len({r["card"]["property_key"] for r in dA + un})
                ests = {name: weighted_estimate(N_A_g, N_non_g, count_in(dA, MEASURES[name]), len(dA),
                                                count_in(un, MEASURES[name]), len(un))
                        for name in ("hard_false", "broad")}
                flags = sorted(set(ests["broad"]["flags"])
                               | ({"exploratory_small_n"} if judged < EXPLORATORY_MIN_JUDGED else set())
                               | ({"few_listings"} if listings < EXPLORATORY_MIN_LISTINGS else set())
                               | ({"not_in_population"} if not pop_l and judged else set()))
                mass = {}
                for name in ("hard_false", "broad"):
                    e = ests[name]
                    if e["point"] is not None:
                        mass[name] = e["N"] * e["point"]
                        total_mass[name] += mass[name]
                    else:
                        mass[name] = None
                labels[lab] = {"N": len(pop_l), "N_A": N_A_g, "N_nonA": N_non_g, "judged": judged,
                               "listings": listings, "hard_false": ests["hard_false"], "broad": ests["broad"],
                               "error_mass": mass, "flags": flags}
            complete = all(v["error_mass"]["broad"] is not None for v in labels.values())
            for v in labels.values():
                v["contribution_share"] = (v["error_mass"]["broad"] / total_mass["broad"]
                                           if v["error_mass"]["broad"] is not None and total_mass["broad"] else None)
            src_out[gname] = {"labels": labels,
                              "partition": {"sum_N_A": sum(v["N_A"] for v in labels.values()),
                                            "sum_N_nonA": sum(v["N_nonA"] for v in labels.values()),
                                            "expected_N_A": queue_meta[src].get("accepted_dirA", 0),
                                            "expected_N_nonA": queue_meta[src].get("accepted_non_dirA", 0)},
                              "contribution_complete": complete}
            p = src_out[gname]["partition"]
            p["ok"] = p["sum_N_A"] == p["expected_N_A"] and p["sum_N_nonA"] == p["expected_N_nonA"]
        out[src] = src_out
    return out


# --------------------------------------------------------------------------- dirB / stability

def dirb_recovery(cond: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    recs = [r for r in cond if "dirB" in r["card"]["strata"]]
    for src in sorted({r["card"]["source"] for r in recs}) + ["all"]:
        rs = recs if src == "all" else [r for r in recs if r["card"]["source"] == src]
        by_tv: Dict[str, Any] = {}
        for tv in ("unsupported", "cannot_assess"):
            sub = [r for r in rs if r["card"]["meta"].get("terra_verdict") == tv]
            by_tv[tv] = dict(label_mix(sub), listings=len({r["card"]["property_key"] for r in sub}))
        out[src] = by_tv
    return out


def pair_stats(res1: Dict[str, Any], res2: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exact-comparable (item, unit, photo-set) pairs across the two canary replicas —
    the denominator rc.terra_flips does not expose. Mirrors its by_key closure."""
    if not res2:
        return []

    def by_key(res):
        evs = {e["condition_id"]: e for e in res.get("evidence_facts") or []}
        revs = {r["condition_id"]: r for r in res.get("condition_reviews") or []}
        cnt, val = Counter(), {}
        for c in res.get("observed_conditions") or []:
            k = (c["catalog_item_id"], c["estimate_unit_id"],
                 tuple(sorted(evs.get(c["condition_id"], {}).get("photo_keys") or [])))
            cnt[k] += 1
            val[k] = (c["condition_id"], revs.get(c["condition_id"], {}).get("verdict"))
        return {k: v for k, v in val.items() if cnt[k] == 1}

    k1, k2 = by_key(res1), by_key(res2)
    return [{"condition_id": k1[k][0], "photos": len(k[2]), "v1": k1[k][1], "v2": k2[k][1],
             "flipped": k1[k][1] != k2[k][1]}
            for k in sorted(set(k1) & set(k2))]


def flip_match(verdict: str, meta: Dict[str, Any]) -> str:
    if verdict in (V_OVER, V_INC):
        return "n/a"
    human_sup = verdict == V_SUP
    if (meta.get("terra_verdict") == "supported") == human_sup:
        return "run_1"
    if (meta.get("replica_terra_verdict") == "supported") == human_sup:
        return "run_2"
    return "neither"


def stability(cond: List[Dict[str, Any]], ctx: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    pairs: List[Dict[str, Any]] = []
    flips_rc = 0
    for (source, _prop), c in sorted(ctx.items()):
        if source != "canary" or not c.get("res2"):
            continue
        pairs.extend(pair_stats(c["res"], c["res2"]))
        flips_rc += len(rc.terra_flips(c["res"], c["res2"]))
    n_pairs = len(pairs)
    n_flips = sum(1 for p in pairs if p["flipped"])
    directions = Counter()
    for p in pairs:
        if not p["flipped"]:
            continue
        if p["v1"] == "supported" and p["v2"] == "unsupported":
            directions["supported->unsupported"] += 1
        elif p["v1"] == "unsupported" and p["v2"] == "supported":
            directions["unsupported->supported"] += 1
        elif "cannot_assess" in (p["v1"], p["v2"]):
            directions[f"{p['v1']}->{p['v2']}"] += 1
        else:
            directions[f"{p['v1']}->{p['v2']}"] += 1
    by_photos = {}
    for lab, pred in (("1 photo", lambda n: n <= 1), ("2+ photos", lambda n: n >= 2)):
        sub = [p for p in pairs if pred(p["photos"])]
        by_photos[lab] = {"pairs": len(sub), "flips": sum(1 for p in sub if p["flipped"]),
                          "wilson": wilson(sum(1 for p in sub if p["flipped"]), len(sub))}
    flip_cards = [r for r in cond if "terra_flip" in r["card"]["strata"]]
    match = Counter(flip_match(r["verdict"], r["card"]["meta"]) for r in flip_cards)
    per_flip = [{"property_key": r["card"]["property_key"],
                 "catalog_item_id": r["card"]["meta"].get("catalog_item_id"),
                 "run_1": r["card"]["meta"].get("terra_verdict"),
                 "run_2": r["card"]["meta"].get("replica_terra_verdict"),
                 "human": r["verdict"], "match": flip_match(r["verdict"], r["card"]["meta"]),
                 "photo_count": r["card"]["meta"].get("photo_count")}
                for r in sorted(flip_cards, key=lambda r: r["card"]["card_id"])]
    return {"comparable_pairs": n_pairs, "flips": n_flips, "flips_rc_crosscheck": flips_rc,
            "rate": n_flips / n_pairs if n_pairs else None, "wilson": wilson(n_flips, n_pairs),
            "directions": dict(directions), "human_match": dict(match), "by_photos": by_photos,
            "flip_cards_reviewed": len(flip_cards), "per_flip": per_flip,
            "listings": len({r["card"]["property_key"] for r in flip_cards})}


# --------------------------------------------------------------------------- lineage / packages

def absorbed_map(res: Dict[str, Any]) -> Dict[str, str]:
    """work_item_id -> package_driver|package_support, from APPLIED package applications only
    (absorbed work items keep status active, so status is never used for this)."""
    cands = {c.get("package_candidate_id"): c for c in res.get("package_candidates") or []}
    out: Dict[str, str] = {}
    for a in res.get("package_applications") or []:
        if a.get("status") != "applied":
            continue
        cand = cands.get(a.get("package_candidate_id")) or {}
        drivers = set(cand.get("driver_work_item_ids") or [])
        for wid in a.get("absorbed_work_item_ids") or []:
            out[wid] = "package_driver" if wid in drivers else "package_support"
    return out


def billing_path(cid: str, idx: Dict[str, Any], amap: Dict[str, str]) -> str:
    w = idx["work_by_cond"].get(cid)
    if not w:
        return "no_active_work_item"
    return amap.get(w["work_item_id"], "standalone")


BILLING_PATHS = ("standalone", "package_driver", "package_support", "no_active_work_item")


def packaging_flow(cond: List[Dict[str, Any]], ctx: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """Raw composition (not weighted): reviewed accepted conditions x human label x billing path."""
    out: Dict[str, Any] = {}
    accepted = [r for r in cond if r["card"]["meta"].get("accepted")]
    for src in sorted({r["card"]["source"] for r in accepted}):
        rs = [r for r in accepted if r["card"]["source"] == src]
        grid = {v: Counter() for v in rc.CONDITION_VERDICTS}
        for r in rs:
            c = ctx[(src, r["card"]["property_key"])]
            grid[r["verdict"]][billing_path(r["card"]["meta"]["condition_id"], c["idx"], c["amap"])] += 1
        out[src] = {"n": len(rs), "listings": len({r["card"]["property_key"] for r in rs}),
                    "grid": {v: dict(grid[v]) for v in grid}}
    return out


def gate_exposure(cond: List[Dict[str, Any]], ctx: Dict[Tuple[str, str], Dict[str, Any]],
                  queue_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Weighted share of broad-error mass (unsupported+overstated accepted conditions) billing
    standalone vs inside an applied package. Uses the estimator arms only: dirA weight 1
    (census), uniform weight N_nonA / n_uniform."""
    out: Dict[str, Any] = {}
    for src, m in sorted(queue_meta.items()):
        dirA, uni = arm_records(cond, src)
        w_uni = (m.get("accepted_non_dirA", 0) / len(uni)) if uni else None
        mass = Counter()
        n_err = 0
        for weight, rs in ((1.0, dirA), (w_uni, uni)):
            if weight is None:
                continue
            for r in rs:
                if r["verdict"] not in MEASURES["broad"]:
                    continue
                n_err += 1
                c = ctx[(src, r["card"]["property_key"])]
                mass[billing_path(r["card"]["meta"]["condition_id"], c["idx"], c["amap"])] += weight
        total = sum(mass.values())
        packaged = mass["package_driver"] + mass["package_support"]
        flags = []
        if n_err < EXPLORATORY_MIN_JUDGED:
            flags.append("exploratory_small_n")
        if uni == [] and m.get("accepted_non_dirA", 0):
            flags.append("no_uniform_coverage")
        out[src] = {"mass": {k: mass.get(k, 0.0) for k in BILLING_PATHS},
                    "total_mass": total, "share_packaged": packaged / total if total else None,
                    "n_error_records": n_err, "uniform_weight": w_uni, "flags": flags}
    return out


def catalog_kinds(path: Path = rc.CATALOG_PATH) -> Dict[str, str]:
    cat = json.loads(Path(path).read_text(encoding="utf-8"))
    items = cat.get("items") if isinstance(cat, dict) else cat
    return {it["id"]: it.get("kind") for it in items or [] if isinstance(it, dict) and it.get("id")}


def driver_class(card: Dict[str, Any], ctx: Dict[Tuple[str, str], Dict[str, Any]],
                 kind_of: Dict[str, str]) -> Dict[str, Any]:
    c = ctx[(card["source"], card["property_key"])]
    ptype, unit = card["meta"].get("package_type"), card["meta"].get("estimate_unit_id")
    cands = [x for x in c["res"].get("package_candidates") or []
             if x.get("package_type") == ptype and x.get("estimate_unit_id") == unit]
    cand = cands[0] if cands else {}
    items: List[str] = []
    for wid in cand.get("driver_work_item_ids") or []:
        w = c["idx"]["works"].get(wid) or {}
        items.extend(w.get("catalog_item_ids") or [])
    kinds = sorted({kind_of.get(i) for i in items if kind_of.get(i)})
    unknown = sorted({i for i in items if not kind_of.get(i)})
    if not items or unknown:
        cls = "unknown"
    elif kinds == ["modernization"]:
        cls = "style_only"
    else:
        cls = "includes_damage_or_degradation"
    return {"class": cls, "driver_items": sorted(set(items)), "driver_kinds": kinds, "unknown_items": unknown}


def packages_block(recs: List[Dict[str, Any]], ctx: Dict[Tuple[str, str], Dict[str, Any]],
                   kind_of: Dict[str, str]) -> Dict[str, Any]:
    p1 = [r for r in recs if r["card"]["kind"] == "package"]
    cards = []
    confusion: Dict[str, Counter] = defaultdict(Counter)
    by_type: Dict[str, Counter] = defaultdict(Counter)
    by_children: Dict[str, Counter] = defaultdict(Counter)
    by_class: Dict[str, Counter] = defaultdict(Counter)
    for r in sorted(p1, key=lambda r: r["card"]["card_id"]):
        m = r["card"]["meta"]
        dc = driver_class(r["card"], ctx, kind_of)
        confusion[str(m.get("v5_sol_decision"))][r["verdict"]] += 1
        by_type[str(m.get("package_type"))][r["verdict"]] += 1
        by_children[str(m.get("child_count"))][r["verdict"]] += 1
        by_class[dc["class"]][r["verdict"]] += 1
        cards.append({"card_id": r["card"]["card_id"], "source": r["card"]["source"],
                      "property_key": r["card"]["property_key"], "package_type": m.get("package_type"),
                      "estimate_unit_id": m.get("estimate_unit_id"), "child_count": m.get("child_count"),
                      "sol": m.get("v5_sol_decision"), "v4_2f": m.get("v4_2f_status"), "human": r["verdict"],
                      "driver_class": dc["class"], "driver_kinds": dc["driver_kinds"],
                      "unknown_items": dc["unknown_items"], "legacy_item_id": r["card"].get("legacy_item_id"),
                      "note": r.get("notes")})
    return {"n": len(p1), "listings": len({r["card"]["property_key"] for r in p1}),
            "by_source": dict(Counter(r["card"]["source"] for r in p1)),
            "verdicts": dict(Counter(r["verdict"] for r in p1)),
            "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
            "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
            "by_child_count": {k: dict(v) for k, v in sorted(by_children.items())},
            "by_driver_class": {k: dict(v) for k, v in sorted(by_class.items())},
            "cards": cards}


def bathrooms_block(recs: List[Dict[str, Any]], ctx: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    p3 = [r for r in recs if r["card"]["kind"] == "bathroom"]
    cards = []
    agg = Counter()
    gap_v5 = gap_v4 = 0
    for r in sorted(p3, key=lambda r: r["card"]["card_id"]):
        m = r["card"]["meta"]
        c = ctx[(r["card"]["source"], r["card"]["property_key"])]
        v5bm = [x for x in c["res"].get("package_candidates") or [] if x.get("package_type") == "bathroom_modernization"]
        v5_ids = {x.get("package_candidate_id") for x in v5bm}
        v5_applied = sum(1 for a in c["res"].get("package_applications") or []
                         if a.get("status") == "applied" and a.get("package_candidate_id") in v5_ids)
        v4 = c["v4"]
        v4_count = (len([p for p in v4.get("packages") or [] if p.get("package_type") == "bathroom_modernization"])
                    if v4 else None)
        human = (r.get("extra") or {}).get("distinct_bathrooms")
        target = human if r["verdict"] == "per_bathroom" else 1 if r["verdict"] == "once" else None
        if target is None:
            rel_v5 = rel_v4 = "excluded_unsure"
            agg["excluded_unsure"] += 1
        else:
            rel_v5 = "exact" if v5_applied == target else "under" if v5_applied < target else "over"
            agg[f"v5_{rel_v5}"] += 1
            gap_v5 += target - v5_applied
            if v4_count is None:
                rel_v4 = "v4_unrecoverable"
                agg["v4_unrecoverable"] += 1
            else:
                rel_v4 = "exact" if v4_count == target else "under" if v4_count < target else "over"
                agg[f"v4_{rel_v4}"] += 1
                gap_v4 += target - v4_count
        cards.append({"card_id": r["card"]["card_id"], "source": r["card"]["source"],
                      "property_key": r["card"]["property_key"], "listing_baths": m.get("listing_baths"),
                      "surrogates": m.get("surrogates"), "v5_bath_units": len(m.get("v5_bath_units") or []),
                      "human_distinct": human, "billing": r["verdict"], "target": target,
                      "v5_candidates": len(v5bm), "v5_applied": v5_applied, "v4_count": v4_count,
                      "v4_expanded": m.get("v4_expanded"), "relation_v5": rel_v5, "relation_v4": rel_v4,
                      "note": r.get("notes")})
    return {"n": len(p3), "listings": len({r["card"]["property_key"] for r in p3}),
            "by_source": dict(Counter(r["card"]["source"] for r in p3)),
            "billing": dict(Counter(r["verdict"] for r in p3)),
            "aggregate": dict(agg), "unit_gap_v5": gap_v5, "unit_gap_v4": gap_v4, "cards": cards}


# --------------------------------------------------------------------------- qualitative / orphans / checks

def qualitative(latest: Dict[str, Dict[str, Any]], cards: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    in_queue = {cid: v for cid, v in latest.items() if cid in cards}
    tags = Counter(v.get("tag") for v in in_queue.values() if v.get("tag"))
    notes = []
    themes = Counter()
    for cid, v in sorted(in_queue.items()):
        if not v.get("notes"):
            continue
        theme = NOTE_THEMES.get(cid)
        if theme is None:
            raise ValueError(f"note on {cid} has no NOTE_THEMES coding — code it before publishing")
        if theme not in THEMES:
            raise ValueError(f"NOTE_THEMES[{cid}] = {theme!r} is not a defined theme")
        themes[theme] += 1
        notes.append({"card_id": cid, "kind": cards[cid]["kind"], "title": cards[cid].get("title"),
                      "verdict": v["verdict"], "tag": v.get("tag"), "note": v["notes"], "theme": theme})
    return {"tags": dict(tags), "themes": dict(themes), "notes": notes,
            "unique_noted_cards": len(notes), "coding_table": dict(NOTE_THEMES)}


def iter_all_runs(root: Path, since: Optional[str]) -> List[Tuple[str, str, Path, Dict[str, Any]]]:
    """Every complete-v5 run >= since (rc.iter_runs keeps only the newest per property;
    orphan resolution must also see superseded runs)."""
    out = []
    root = Path(root)
    if not root.is_dir():
        return out
    for prop_dir in sorted(root.iterdir()):
        if not prop_dir.is_dir() or prop_dir.name.startswith("."):
            continue
        for run_dir in sorted(prop_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not rc.RUN_DIR_RE.match(run_dir.name):
                continue
            if since and run_dir.name < since:
                continue
            path = run_dir / "photo_intel_debug.json"
            if not path.is_file():
                continue
            try:
                art = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if rc.v5_result(art) is None:
                continue
            prop = str(((art.get("property") or {}).get("property_key")) or prop_dir.name)
            out.append((prop, run_dir.name, path, art))
    return out


def _candidate_ids(source: str, prop: str, run_id: str, art: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every card id this artifact could have produced (the card_id preimage set)."""
    res = rc.v5_result(art)
    if res is None:
        return {}
    out: Dict[str, Dict[str, Any]] = {}

    def put(kind, key):
        out[rc.card_id(source, prop, run_id, kind, key)] = {
            "source": source, "property_key": prop, "run_id": run_id, "kind": kind, "key": key}

    for c in res.get("observed_conditions") or []:
        put("condition", c.get("condition_id"))
    v4 = art.get("renovation_estimate_v4") or {}
    for pk in v4.get("package_candidates") or []:
        if pk.get("verification_status") != "rejected":
            continue
        ptype, unit = pk.get("package_type"), pk.get("estimate_unit_id") or ""
        if any(x.get("package_type") == ptype and x.get("estimate_unit_id") == unit
               for x in res.get("package_candidates") or []):
            put("package", f"{ptype}|{unit}")
    if len(rc._bath_surrogates(v4)) >= 2:
        put("bathroom", "bathrooms")
    return out


def resolve_orphans(orphan_ids: Iterable[str], listings: List[Dict[str, Any]],
                    latest: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    orphan_ids = sorted(orphan_ids)
    current = {(L["source"], L["property_key"], L["run_id"]) for L in listings}
    candidates: Dict[str, Dict[str, Any]] = {}
    for L in listings:  # zero extra IO: artifacts already in memory
        candidates.update(_candidate_ids(L["source"], L["property_key"], L["run_id"], L["artifact"]))
    unresolved = [o for o in orphan_ids if o not in candidates]
    if unresolved:  # only then walk superseded production runs
        for prop, run_id, _path, art in iter_all_runs(rc.PROD_ROOT, rc.DEFAULT_SINCE):
            candidates.update(_candidate_ids("production", prop, run_id, art))
    out = []
    for oid in orphan_ids:
        ident = candidates.get(oid)
        v = latest.get(oid) or {}
        row = {"card_id": oid, "verdict": v.get("verdict"), "tag": v.get("tag"),
               "has_note": bool(v.get("notes")), "ts": v.get("ts")}
        if not ident:
            row.update({"source": None, "property_key": None, "run_id": None, "kind": None, "key": None,
                        "classification": "unresolved"})
        else:
            row.update(ident)
            row["classification"] = ("excluded_low_res"
                                     if (ident["source"], ident["property_key"], ident["run_id"]) in current
                                     else "run_superseded")
        out.append(row)
    return out


def spot_checks(queue: Dict[str, Any], ctx: Dict[Tuple[str, str], Dict[str, Any]],
                claims: Dict[str, str]) -> List[Dict[str, Any]]:
    """Deterministic rebuild of one card per stratum (lexicographically smallest card_id)
    straight from the source artifact, field-compared on meta / claim / strip photo keys."""
    out = []
    for stratum in rc.STRATUM_ORDER:
        picks = sorted(c["card_id"] for c in queue["cards"] if stratum in c["strata"])
        if not picks:  # a stratum with no cards in this queue is absent, not broken
            out.append({"stratum": stratum, "card_id": None, "kind": None, "ok": True,
                        "mismatches": [], "empty": True})
            continue
        card = next(c for c in queue["cards"] if c["card_id"] == picks[0])
        c = ctx.get((card["source"], card["property_key"]))
        if not c or c["run_id"] != card["run_id"]:
            out.append({"stratum": stratum, "card_id": picks[0], "ok": False, "mismatches": ["artifact not loaded"]})
            continue
        try:
            if card["kind"] == "condition":
                rebuilt = rc.condition_card(source=card["source"], prop=card["property_key"], run_id=card["run_id"],
                                            idx=c["idx"], cid=card["meta"]["condition_id"], claims=claims,
                                            paths=c["paths"], f2=c["f2"], dims=c["dims"],
                                            replica=c["flips"].get(card["meta"]["condition_id"]))
            elif card["kind"] == "package":
                ptype, unit = card["meta"]["package_type"], card["meta"]["estimate_unit_id"]
                v4_pk = next(p for p in c["v4"].get("package_candidates") or []
                             if p.get("verification_status") == "rejected" and p.get("package_type") == ptype
                             and (p.get("estimate_unit_id") or "") == unit)
                cand = next(x for x in c["res"].get("package_candidates") or []
                            if x.get("package_type") == ptype and x.get("estimate_unit_id") == unit)
                rebuilt = rc.package_card(source=card["source"], prop=card["property_key"], run_id=card["run_id"],
                                          idx=c["idx"], v4_pk=v4_pk, cand=cand, res=c["res"],
                                          paths=c["paths"], dims=c["dims"])
            else:
                rebuilt = rc.bathroom_card(source=card["source"], prop=card["property_key"], run_id=card["run_id"],
                                           art=c["art"], idx=c["idx"], res=c["res"],
                                           surr=rc._bath_surrogates(c["v4"]), paths=c["paths"], dims=c["dims"])
        except StopIteration:
            out.append({"stratum": stratum, "card_id": picks[0], "ok": False, "mismatches": ["source rows not found"]})
            continue
        mismatches = [f for f in ("meta", "claim") if rebuilt[f] != card[f]]
        if ([[p["key"] for p in s["photos"]] for s in rebuilt["strips"]]
                != [[p["key"] for p in s["photos"]] for s in card["strips"]]):
            mismatches.append("strips")
        out.append({"stratum": stratum, "card_id": picks[0], "kind": card["kind"],
                    "ok": not mismatches, "mismatches": mismatches})
    return out


# --------------------------------------------------------------------------- orchestration

def join_records(queue: Dict[str, Any], latest: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    cards = {c["card_id"]: c for c in queue["cards"]}
    recs = [{**v, "card": cards[cid]} for cid, v in latest.items() if cid in cards]
    orphans = sorted(cid for cid in latest if cid not in cards)
    return recs, orphans


def evidence_statement(result: str, denominator: str, concentration: str, basis: str, domain: str) -> str:
    return (f"*Evidence:* {result} ({denominator}). Concentration: {concentration}. Basis: {basis}. "
            f"Decision domain: {domain}. No action is prescribed here.")


def fmt_pct(x: Optional[float], digits: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def fmt_band(e: Dict[str, Any]) -> str:
    if e.get("low") is None or e.get("high") is None:
        return "—"
    return f"[{fmt_pct(e['low'])}, {fmt_pct(e['high'])}]"


def analyze(queue: Dict[str, Any], verdicts_path: Path, listings: List[Dict[str, Any]],
            claims: Optional[Dict[str, str]] = None, kind_of: Optional[Dict[str, str]] = None,
            queue_path: Optional[Path] = None) -> Dict[str, Any]:
    queue_meta = queue.get("meta") or {}
    uniform_rate = queue.get("uniform_rate", rc.DEFAULT_UNIFORM_RATE)
    audit = audit_verdicts(verdicts_path)
    latest = audit.pop("latest")
    recs, orphan_ids = join_records(queue, latest)
    cond = [r for r in recs if r["card"]["kind"] == "condition"]
    cards_by_id = {c["card_id"]: c for c in queue["cards"]}

    pop_rows, enum_meta, ctx = enumerate_population(listings, uniform_rate)
    recon = reconcile(enum_meta, queue_meta)
    claims = claims if claims is not None else rc.claim_texts()
    kind_of = kind_of if kind_of is not None else catalog_kinds()

    completion = {}
    for s in rc.STRATUM_ORDER:
        total = sum(1 for c in queue["cards"] if s in c["strata"])
        done = sum(1 for r in recs if s in r["card"]["strata"])
        completion[s] = {"reviewed": done, "total": total}
    completion_ok = len(recs) == len(queue["cards"])

    head = headline(cond, queue_meta)
    census = {src: {"n_A": head[src]["weighted"]["broad"]["n_A"], "N_A": head[src]["weighted"]["broad"]["N_A"],
                    "ok": head[src]["weighted"]["broad"]["n_A"] == head[src]["weighted"]["broad"]["N_A"]}
              for src in head}
    sub = subgroups(pop_rows, cond, queue_meta)
    partition_ok = all(g["partition"]["ok"] for s in sub.values() for g in s.values())
    orphans = resolve_orphans(orphan_ids, listings, latest)
    checks = spot_checks(queue, ctx, claims)
    stab = stability(cond, ctx)
    pkgs = packages_block(recs, ctx, kind_of)
    baths = bathrooms_block(recs, ctx)
    flow = packaging_flow(cond, ctx)
    gate = gate_exposure(cond, ctx, queue_meta)
    dirb = dirb_recovery(cond)
    qual = qualitative(latest, cards_by_id)
    p6_recs = sorted((r for r in cond if "p6_forced_single" in r["card"]["strata"]),
                     key=lambda r: r["card"]["card_id"])
    p6 = {"n": len(p6_recs), "listings": len({r["card"]["property_key"] for r in p6_recs}),
          "verdicts": dict(Counter(r["verdict"] for r in p6_recs)),
          "cards": [{"property_key": r["card"]["property_key"],
                     "catalog_item_id": r["card"]["meta"].get("catalog_item_id"),
                     "photo_count": r["card"]["meta"].get("photo_count"),
                     "verdict": r["verdict"], "legacy_item_id": r["card"].get("legacy_item_id")}
                    for r in p6_recs]}

    integrity_ok = all([recon["ok"], completion_ok, audit["latest_crosscheck_ok"], partition_ok,
                        all(o["classification"] != "unresolved" for o in orphans),
                        all(c["ok"] for c in checks), all(v["ok"] for v in census.values()),
                        stab["flips"] == stab["flips_rc_crosscheck"]])

    statements = {}
    for src, h in head.items():
        b, hf, inc = h["weighted"]["broad"], h["weighted"]["hard_false"], h["weighted"]["inconclusive"]
        top = None
        kinds = sub.get(src, {}).get("catalog_kind", {}).get("labels", {})
        shares = [(lab, v["contribution_share"]) for lab, v in kinds.items() if v["contribution_share"] is not None]
        if shares:
            top = max(shares, key=lambda kv: kv[1])
        statements[f"condition_truth_{src}"] = evidence_statement(
            f"weighted broad-problem rate {fmt_pct(b['point'])} (95% band {fmt_band(b)}), "
            f"hard-false {fmt_pct(hf['point'])} ({fmt_band(hf)}), inconclusive {fmt_pct(inc['point'])}",
            f"N_accepted={b['N']} ({b['N_A']} dirA census + {b['N_nonA']} non-dirA via {b['n_U']} uniform judgments), "
            f"{h['listings']} listings",
            (f"largest catalog-kind contribution: {top[0]} at {fmt_pct(top[1], 0)} of weighted error mass" if top
             else "see §4"),
            "population-estimating (stratified census + uniform sample)",
            "condition prompting and evidence handling")
    n_dirb = sum(v["n"] for v in dirb.get("all", {}).values())
    dirb_l = len({r["card"]["property_key"] for r in cond if "dirB" in r["card"]["strata"]})
    statements["dirb"] = evidence_statement(
        "human labels on Terra-rejected, 2f-confirmed claims as tallied in §5",
        f"n={n_dirb} targeted dirB cards, {dirb_l} listings",
        "split by Terra unsupported vs cannot_assess",
        "targeted (a recovery yield, not Terra recall)",
        "pass boundaries and condition prompting")
    statements["stability"] = evidence_statement(
        f"Terra flip rate {fmt_pct(stab['rate'])} on exact-comparable replica pairs"
        + (f" (95% Wilson [{fmt_pct((stab['wilson'] or {}).get('low'))}, {fmt_pct((stab['wilson'] or {}).get('high'))}])"
           if stab["wilson"] else ""),
        f"{stab['flips']} flips / {stab['comparable_pairs']} pairs, canary only, {stab['listings']} flip-card listings",
        f"human matched run_1 {stab['human_match'].get('run_1', 0)}, run_2 {stab['human_match'].get('run_2', 0)}, "
        f"neither {stab['human_match'].get('neither', 0)}, n/a {stab['human_match'].get('n/a', 0)}",
        "population-estimating for the canary replicas; does not attribute flips to batching",
        "model stability")
    statements["packages"] = evidence_statement(
        "human package-warrant labels and Sol-vs-human confusion as tallied in §7",
        f"n={pkgs['n']} P1 cards (census of the 2f-rejected, Sol-decided slice), {pkgs['listings']} listings",
        "split by package type, child count and driver class",
        "targeted census of one contested slice — not package-pipeline accuracy",
        "package formation")
    statements["flow"] = evidence_statement(
        "billing-path composition of reviewed accepted conditions as tallied in §8",
        f"n={sum(v['n'] for v in flow.values())} reviewed accepted conditions across "
        f"{sum(v['listings'] for v in flow.values())} listing-source rows",
        "shown per human label and billing path",
        "raw composition (mixes strata; a packaging-utilization proxy, not weighted and not a usefulness judgment)",
        "package formation")
    statements["gate"] = evidence_statement(
        "weighted broad-error mass split standalone vs packaged as tallied in §9",
        f"error records: " + ", ".join(f"{src} n={v['n_error_records']}" for src, v in gate.items()),
        "share billed inside an applied package: "
        + ", ".join(f"{src} {fmt_pct(v['share_packaged'])}" for src, v in gate.items()),
        "population-estimating weights on a small error count — read with the flags",
        "package formation and pass boundaries (where defects survive downstream)")
    statements["bathrooms"] = evidence_statement(
        f"exact/under/over vs the human bathroom-count target as tallied in §10; "
        f"aggregate unit gap v5 {baths['unit_gap_v5']:+d}, v4 {baths['unit_gap_v4']:+d}",
        f"n={baths['n']} multi-surrogate bathroom cards, {baths['listings']} listings",
        "per-card table in §10",
        "targeted (the selected multi-surrogate cohort)",
        "room multiplicity")
    statements["qualitative"] = evidence_statement(
        "theme counts over the coded free-text notes as tallied in §11",
        f"{qual['unique_noted_cards']} noted cards, {len(qual['tags'])} distinct tags",
        "themes coded by the analysis session; the full note text is shown so the coding is auditable",
        "exploratory (absence of a note is not evidence a theme is absent)",
        "condition prompting, evidence handling, package formation and room identity")

    qp = Path(queue_path) if queue_path else QUEUE
    inputs = {"queue": {"path": str(qp), "sha256": None, "bytes": None},
              "verdicts": {"path": str(verdicts_path), "sha256": None, "bytes": None},
              "catalog": {"path": str(rc.CATALOG_PATH), "sha256": None, "bytes": None},
              "uniform_rate": uniform_rate, "min_evidence_px": rc.MIN_EVIDENCE_PX,
              "generated_from": queue.get("generated_from")}
    for key, p in (("queue", qp), ("verdicts", Path(verdicts_path)), ("catalog", rc.CATALOG_PATH)):
        if Path(p).is_file():
            inputs[key]["sha256"] = sha256_file(Path(p))
            inputs[key]["bytes"] = Path(p).stat().st_size
    freeze = rc.CANARY_ROOT / "input_freeze.json"
    if freeze.is_file():
        inputs["canary_input_freeze"] = {"path": str(freeze), "sha256": sha256_file(freeze),
                                         "bytes": freeze.stat().st_size}

    per_card = [{"card_id": c["card_id"], "kind": c["kind"], "source": c["source"], "strata": c["strata"],
                 "property_key": c["property_key"], "verdict": latest.get(c["card_id"], {}).get("verdict"),
                 "tag": latest.get(c["card_id"], {}).get("tag"),
                 "has_note": bool(latest.get(c["card_id"], {}).get("notes"))}
                for c in queue["cards"]]

    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ci_method": CI_METHOD,
        "inputs": inputs,
        "integrity": {"ok": integrity_ok, "verdict_audit": audit, "orphans": orphans,
                      "manifest": listing_manifest(listings, {c["property_key"] for c in queue["cards"]}),
                      "reconciliation": recon, "census": census,
                      "partition_ok": partition_ok, "completion_ok": completion_ok,
                      "spot_checks": checks,
                      "no_eligible_evidence_listings": sorted({L["property_key"] for L in listings}
                                                              - {c["property_key"] for c in queue["cards"]})},
        "population": {src: enum_meta.get(src) for src in sorted(enum_meta)},
        "completion": completion,
        "condition_truth": head,
        "error_concentration": sub,
        "p6_forced_single": p6,
        "dirb_recovery": dirb,
        "stability": stab,
        "packages": pkgs,
        "packaging_flow": flow,
        "gate_exposure": gate,
        "bathrooms": baths,
        "qualitative": qual,
        "evidence_statements": statements,
        "not_shown": NOT_SHOWN,
        "per_card": per_card,
    }


# --------------------------------------------------------------------------- rendering

def _weighted_rows(head: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    for src, h in sorted(head.items()):
        for name in ("hard_false", "broad", "inconclusive"):
            e = h["weighted"][name]
            rows.append([src, name, e["N"], e["N_A"], e["x_A"], fmt_pct(e["r_A"]), e["N_nonA"], e["n_U"], e["x_U"],
                         fmt_pct(e["r_U"]), fmt_pct(e["point"]), fmt_band(e), h["listings"],
                         ",".join(e["flags"]) or "—"])
    return rows


def render_md(a: Dict[str, Any]) -> str:
    integ = a["integrity"]
    L: List[str] = []
    L.append("# Review analysis — v5 manual review (frozen cohort)\n")
    L.append(f"Generated {a['generated_at']} · queue sha `{(a['inputs']['queue']['sha256'] or '')[:12]}` · "
             f"verdicts sha `{(a['inputs']['verdicts']['sha256'] or '')[:12]}` · "
             f"integrity **{'OK' if integ['ok'] else 'FAILED'}**\n")

    L.append("## 0. Scope and framing\n")
    L.append("- Condition, package and bathroom cards are separate analysis units; nothing about Pass 2f "
             "correctness is derived from a condition verdict (direction A/B are sampling strata).\n"
             "- Canary and production are analyzed separately; **no pooled headline rate is published**.\n"
             "- Every percentage carries its numerator, denominator and unique-listing count; slices with "
             f"fewer than {EXPLORATORY_MIN_JUDGED} judgments or fewer than {EXPLORATORY_MIN_LISTINGS} listings "
             "are flagged exploratory.\n"
             f"- Interval method: {a['ci_method']}\n"
             "- Semantic authority: docs/DESIGN_review_method.md. The 11 orphaned verdicts (§1.4) are excluded "
             "from every metric.\n")

    L.append("## 1. Freeze and integrity audit\n")
    L.append("### 1.1 Inputs\n")
    rows = [[k, v.get("path"), v.get("sha256"), v.get("bytes")]
            for k, v in a["inputs"].items() if isinstance(v, dict) and "path" in v]
    L.append(table(["input", "path", "sha256", "bytes"], rows))
    L.append(f"\nuniform_rate {a['inputs']['uniform_rate']} · MIN_EVIDENCE_PX {a['inputs']['min_evidence_px']} · "
             f"generated_from {a['inputs'].get('generated_from')}\n")
    L.append("### 1.2 Listing manifest\n")
    L.append(table(["source", "property", "run", "artifact sha256", "replica sha256", "carded"],
                   [[m["source"], m["property_key"], m["run_id"], (m["artifact_sha256"] or "")[:12],
                     (m["replica_sha256"] or "")[:12] or "—", "yes" if m["carded"] else "NO — no eligible evidence"]
                    for m in integ["manifest"]]))
    L.append("### 1.3 Verdict stream audit\n")
    au = integ["verdict_audit"]
    L.append(table(["metric", "value"], [
        ["raw lines", au["lines"]], ["invalid lines", au["invalid_lines"]], ["undo records", au["undos"]],
        ["peeked records", au["peeked_records"]], ["distinct card_ids", au["distinct_ids"]],
        ["net verdicts (latest wins)", au["net_verdicts"]],
        ["overwrites — identical re-saves", au["overwrites"]["identical"]],
        ["overwrites — changed verdict", au["overwrites"]["changed"]],
        ["timestamp range", " → ".join(x or "" for x in au["ts_range"])],
        ["latest-wins cross-check vs rc.latest_verdicts", "ok" if au["latest_crosscheck_ok"] else "MISMATCH"]]))
    for ch in au["overwrites"]["changed_cards"]:
        seq = " → ".join(f"{r['verdict']} ({r['ts']})" for r in ch["records"])
        L.append(f"\nChanged-verdict overwrite on `{ch['card_id']}`: {seq}. The latest record is used everywhere.\n")
    L.append("### 1.4 Orphaned verdicts (excluded from every metric)\n")
    L.append(table(["card_id", "source", "property", "run", "kind", "key", "classification", "verdict"],
                   [[o["card_id"], o["source"], o["property_key"], o["run_id"], o["kind"], o["key"],
                     o["classification"], o["verdict"]] for o in integ["orphans"]]))
    L.append("### 1.5 Population reconciliation (artifacts vs queue.meta)\n")
    rows = []
    for src, rcn in sorted(integ["reconciliation"]["sources"].items()):
        for f in sorted(set(rcn["expected"]) | set(rcn["enumerated"])):
            rows.append([src, f, json.dumps(rcn["expected"].get(f)), json.dumps(rcn["enumerated"].get(f)),
                         "ok" if rcn["expected"].get(f) == rcn["enumerated"].get(f) else "MISMATCH"])
    L.append(table(["source", "field", "queue.meta", "enumerated", "ok"], rows))
    L.append("### 1.6 Census and spot checks\n")
    L.append(table(["source", "dirA judged", "dirA population", "census ok"],
                   [[src, v["n_A"], v["N_A"], "ok" if v["ok"] else "GAP"] for src, v in sorted(integ["census"].items())]))
    L.append("\n" + table(["stratum", "card_id", "kind", "rebuild vs queue"],
                          [[c["stratum"], c["card_id"], c.get("kind"), "ok" if c["ok"] else "; ".join(c["mismatches"])]
                           for c in integ["spot_checks"]]))
    L.append(f"\nSubgroup partition checks: {'all ok' if integ['partition_ok'] else 'FAILED'} · "
             f"completion {'212-complete' if integ['completion_ok'] else 'INCOMPLETE'} · "
             f"flip cross-check vs rc.terra_flips: {a['stability']['flips']} == {a['stability']['flips_rc_crosscheck']}\n")

    L.append("## 2. Population and completion\n")
    L.append(table(["source", "listings", "accepted", "dirA", "non-dirA", "uniform picks", "low-res excluded"],
                   [[src, p["listings"], p["accepted"], p["accepted_dirA"], p["accepted_non_dirA"], p["uniform"],
                     json.dumps(p["low_res_excluded"])] for src, p in sorted(a["population"].items())]))
    L.append("\n" + table(["stratum", "reviewed", "total"],
                          [[s, c["reviewed"], c["total"]] for s, c in a["completion"].items()]))

    L.append("## 3. Accepted-condition truth (per source)\n")
    rows = []
    for src, h in sorted(a["condition_truth"].items()):
        for arm, mix in h["label_mix"].items():
            rows.append([src, arm, mix["n"], mix["supported"], mix["unsupported"], mix["overstated"],
                         mix["inconclusive"]])
    L.append("### 3.1 Label mix per estimator arm\n")
    L.append(table(["source", "arm", "n", "supported", "unsupported", "overstated", "inconclusive"], rows))
    L.append("### 3.2 Weighted rates on accepted (billed) conditions\n")
    L.append(table(["source", "measure", "N", "N_A", "x_A", "r_A", "N_nonA", "n_U", "x_U", "r_U",
                    "weighted", "95% band", "listings", "flags"], _weighted_rows(a["condition_truth"])))
    for src in sorted(a["condition_truth"]):
        L.append("\n" + a["evidence_statements"][f"condition_truth_{src}"] + "\n")

    L.append("## 4. Error concentration (weighted subgroups)\n")
    L.append("Weighted subgroup rates use the same census + uniform identity within the subgroup; "
             "`contribution` is the subgroup's share of the source's total weighted broad-error mass.\n")
    for src, groups in sorted(a["error_concentration"].items()):
        L.append(f"\n### {src}\n")
        for gname, g in groups.items():
            labels = g["labels"]
            if gname == "catalog_item":
                shown = {k: v for k, v in labels.items() if v["judged"] >= 3}
                title = f"by {gname} (judged ≥ 3; full set in the JSON)"
            else:
                shown = labels
                title = f"by {gname}"
            rows = []
            for lab, v in sorted(shown.items(), key=lambda kv: (-(kv[1]["contribution_share"] or -1), kv[0])):
                e = v["broad"]
                rows.append([lab, v["N"], v["N_A"], v["N_nonA"], v["judged"], v["listings"],
                             fmt_pct(e["point"]), fmt_band(e), fmt_pct(v["contribution_share"], 0),
                             ",".join(v["flags"]) or "—"])
            L.append(f"\n#### {title}\n" + table(["group", "N", "N_A", "N_nonA", "judged", "listings",
                                                  "weighted broad", "95% band", "contribution", "flags"], rows))
            if not g["partition"]["ok"]:
                L.append(f"\nPARTITION MISMATCH: {g['partition']}\n")
    p6 = a["p6_forced_single"]
    if p6["n"]:
        L.append(f"\n### p6_forced_single (targeted raw counts — the accepted-provisionally single-photo items)\n")
        L.append(table(["property", "item", "photos", "human verdict", "legacy"],
                       [[c["property_key"], c["catalog_item_id"], c["photo_count"], c["verdict"],
                         c["legacy_item_id"] or "—"] for c in p6["cards"]]))
        L.append(f"\nVerdict counts: {json.dumps(p6['verdicts'], sort_keys=True)} over n={p6['n']} cards, "
                 f"{p6['listings']} listings — targeted, not a population estimate.\n")
    L.append("\n*Evidence:* subgroup rates are population-estimating only where the uniform arm covers the "
            "subgroup; rows flagged `no_uniform_coverage`, `exploratory_small_n` or `few_listings` are targeted "
            "or exploratory. Decision domain: condition prompting, evidence handling and pass boundaries. "
            "No action is prescribed here.\n")

    L.append("## 5. Terra-rejected claims (dirB — targeted recovery yield)\n")
    rows = []
    for src, by_tv in sorted(a["dirb_recovery"].items()):
        for tv, mix in sorted(by_tv.items()):
            rows.append([src, tv, mix["n"], mix["listings"], mix["supported"], mix["unsupported"],
                         mix["overstated"], mix["inconclusive"]])
    L.append(table(["source", "terra_verdict", "n", "listings", "human supported", "human unsupported",
                    "human overstated", "human inconclusive"], rows))
    L.append("\n" + a["evidence_statements"]["dirb"] + "\n")

    st = a["stability"]
    L.append("## 6. Terra stability (canary run_1 × run_2)\n")
    L.append(f"Exact-comparable pairs: **{st['comparable_pairs']}** · flips: **{st['flips']}** · "
             f"flip rate {fmt_pct(st['rate'])}"
             + (f" (95% Wilson [{fmt_pct((st['wilson'] or {}).get('low'))}, {fmt_pct((st['wilson'] or {}).get('high'))}])"
                if st["wilson"] else "") + "\n")
    L.append("\n" + table(["flip direction", "n"], sorted(st["directions"].items())))
    L.append("\n" + table(["human matched", "n"], sorted(st["human_match"].items())))
    L.append("\n" + table(["evidence load", "pairs", "flips", "flip rate", "95% Wilson"],
                          [[lab, v["pairs"], v["flips"], fmt_pct(v["flips"] / v["pairs"] if v["pairs"] else None),
                            f"[{fmt_pct((v['wilson'] or {}).get('low'))}, {fmt_pct((v['wilson'] or {}).get('high'))}]"
                            if v["wilson"] else "—"]
                           for lab, v in a["stability"]["by_photos"].items()]))
    L.append("\n" + table(["property", "item", "run_1", "run_2", "human", "matched", "photos"],
                          [[p["property_key"], p["catalog_item_id"], p["run_1"], p["run_2"], p["human"],
                            p["match"], p["photo_count"]] for p in st["per_flip"]]))
    L.append("\n" + a["evidence_statements"]["stability"] + "\n")

    pk = a["packages"]
    L.append("## 7. Package warrant (P1 — census of the 2f-rejected, Sol-decided slice)\n")
    L.append(table(["human verdict", "n"], sorted(pk["verdicts"].items())))
    L.append("\n### Sol decision × human verdict\n")
    hlabels = rc.PACKAGE_VERDICTS
    L.append(table(["Sol decision"] + list(hlabels),
                   [[sol] + [v.get(h, 0) for h in hlabels] for sol, v in sorted(pk["confusion"].items())]))
    for name, block in (("package type", pk["by_type"]), ("child count", pk["by_child_count"]),
                        ("driver class", pk["by_driver_class"])):
        L.append(f"\n### by {name}\n" + table([name] + list(hlabels),
                 [[k] + [v.get(h, 0) for h in hlabels] for k, v in sorted(block.items())]))
    L.append("\n### Per-card detail\n")
    L.append(table(["source", "property", "package", "children", "Sol", "human", "driver class", "legacy", "note"],
                   [[c["source"], c["property_key"], f"{c['package_type']} @ {c['estimate_unit_id']}",
                     c["child_count"], c["sol"], c["human"], c["driver_class"], c["legacy_item_id"] or "—",
                     (c["note"] or "")[:80]] for c in pk["cards"]]))
    L.append("\n" + a["evidence_statements"]["packages"] + "\n")

    L.append("## 8. Packaging flow of reviewed accepted conditions\n")
    for src, f in sorted(a["packaging_flow"].items()):
        L.append(f"\n### {src} (n={f['n']}, {f['listings']} listings)\n")
        L.append(table(["human label"] + list(BILLING_PATHS),
                       [[v] + [f["grid"][v].get(bp, 0) for bp in BILLING_PATHS] for v in rc.CONDITION_VERDICTS]))
    L.append("\n" + a["evidence_statements"]["flow"] + "\n")

    L.append("## 9. Package-gate exposure (weighted broad-error mass)\n")
    L.append(table(["source", "standalone", "package driver", "package support", "no active work item",
                    "share packaged", "error records", "uniform weight", "flags"],
                   [[src, f"{v['mass']['standalone']:.1f}", f"{v['mass']['package_driver']:.1f}",
                     f"{v['mass']['package_support']:.1f}", f"{v['mass']['no_active_work_item']:.1f}",
                     fmt_pct(v["share_packaged"]), v["n_error_records"],
                     "—" if v["uniform_weight"] is None else f"{v['uniform_weight']:.2f}",
                     ",".join(v["flags"]) or "—"] for src, v in sorted(a["gate_exposure"].items())]))
    L.append("\n" + a["evidence_statements"]["gate"] + "\n")

    b = a["bathrooms"]
    L.append("## 10. Multi-bathroom billing (P3 — the selected multi-surrogate cohort)\n")
    L.append(table(["source", "property", "listing baths", "surrogates", "human distinct", "billing", "target",
                    "v5 applied", "v5 candidates", "v4 packages", "v4 expanded", "vs v5", "vs v4"],
                   [[c["source"], c["property_key"], c["listing_baths"], c["surrogates"], c["human_distinct"],
                     c["billing"], c["target"] if c["target"] is not None else "—", c["v5_applied"],
                     c["v5_candidates"], c["v4_count"] if c["v4_count"] is not None else "—",
                     c["v4_expanded"], c["relation_v5"], c["relation_v4"]] for c in b["cards"]]))
    L.append(f"\nAggregate (decisive cards): {json.dumps(b['aggregate'], sort_keys=True)} · "
             f"unit gap (human target − billed packages): v5 {b['unit_gap_v5']:+d}, v4 {b['unit_gap_v4']:+d}. "
             "Counts cover bathroom_modernization packages only; bathroom work billed through other package "
             "types or standalone lines is outside this comparison.\n")
    L.append("\n" + a["evidence_statements"]["bathrooms"] + "\n")

    q = a["qualitative"]
    L.append("## 11. Qualitative themes (tags and coded notes)\n")
    L.append(table(["error tag", "n"], sorted(q["tags"].items(), key=lambda kv: -kv[1]) or [["—", 0]]))
    L.append("\n" + table(["theme", "n"], sorted(q["themes"].items(), key=lambda kv: -kv[1])))
    L.append("\n### Coded notes (full text, auditable)\n")
    L.append(table(["card_id", "kind", "verdict", "theme", "note"],
                   [[n["card_id"], n["kind"], n["verdict"], n["theme"], n["note"]] for n in q["notes"]]))
    L.append("\n" + a["evidence_statements"]["qualitative"] + "\n")

    L.append("## 12. What this analysis does not show\n")
    for item in a["not_shown"]:
        L.append(f"- {item}\n")

    L.append("\n## Appendix A. Per-card reconciliation\n")
    L.append(f"{len(a['per_card'])} in-queue cards (all verdicted) + {len(integ['orphans'])} orphaned verdicts "
             "(§1.4). Full per-card rows are in reports/review_analysis.json → `per_card`.\n")
    counts = Counter((c["kind"], c["source"]) for c in a["per_card"])
    verdicted = Counter((c["kind"], c["source"]) for c in a["per_card"] if c["verdict"])
    L.append(table(["kind", "source", "cards", "verdicted"],
                   [[k, s, n, verdicted.get((k, s), 0)] for (k, s), n in sorted(counts.items())]))
    return "\n".join(L)


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=QUEUE)
    ap.add_argument("--verdicts", type=Path, default=VERDICTS)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    listings = load_listings()
    analysis = analyze(queue, args.verdicts, listings, queue_path=args.queue)
    args.out_md.write_text(render_md(analysis), encoding="utf-8")
    args.out_json.write_text(json.dumps(analysis, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    ok = analysis["integrity"]["ok"]
    print(f"integrity: {'OK' if ok else 'FAILED'}")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")
    for src, h in sorted(analysis["condition_truth"].items()):
        e = h["weighted"]["broad"]
        print(f"{src}: weighted broad {fmt_pct(e['point'])} band {fmt_band(e)} on N={e['N']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
