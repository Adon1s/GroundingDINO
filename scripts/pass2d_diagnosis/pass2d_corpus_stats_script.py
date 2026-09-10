r"""
Pass 2d corpus-wide statistics, computed offline from stored artifacts only.

READ-ONLY. No model / provider / network calls. Run with:
  C:/Users/Steven/PycharmProjects/realtorvision-backend/.venv/Scripts/python.exe <this file>

Populations
-----------
ALL      : every resolved_items row in the 25 readable pinned artifacts listed in
           reports/catalog_audit_replay_corpus.json["artifacts"]  (expected 4,107)
PRODUCT  : the subset whose (run_id, issue_id) appears in
           reports/catalog_audit_replay_corpus.json["rows"]        (expected 1,969)

Sources
-------
- reports/catalog_audit_replay_corpus.json              : artifact pin list + product-lane rows
- <artifact>/photo_intel_debug.json
    photos[<k>].scene.group                             : scene group
    photos[<k>].debug.resolved_items[]                  : the Pass 2d trace
    analysis_debug.renovation_estimate_v5.result        : Terra conditions / reviews / dispositions
- reports/catalog_audit_replay_snapshot_checkpoint_after.json : pre/post guardrail top-8 + shortcut margins
- git show 58c073e:tools/issue_catalog_kind_v2.json     : the 3.1-era catalog the artifacts ran under
- tools/pipeline_common.term_matches                    : the real term matcher
- tools/scene_classifier_passes._resolve_candidate_via_lexical_shortcut : the real shortcut predicate
"""

import collections
import json
import os
import subprocess
import sys

REPO = r"C:\Users\Steven\PycharmProjects\realtorvision-backend"
SCRATCH = (r"C:\Users\Steven\AppData\Local\Temp\claude"
           r"\C--Users-Steven-PycharmProjects-realtorvision-backend"
           r"\6f751f7e-4d60-4702-a546-01b841cf1207\scratchpad")
sys.path.insert(0, REPO)

from tools.pipeline_common import term_matches                      # noqa: E402
from tools.scene_classifier_passes import (                         # noqa: E402
    _normalize_signal_text,
    _resolve_candidate_via_lexical_shortcut,
)

OUT = []


def say(*a):
    line = " ".join(str(x) for x in a)
    OUT.append(line)
    print(line)


def jload(path):
    # Artifacts contain non-cp1252 bytes; the shell default encoding must be overridden.
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Load corpus pin list and the product-lane membership key
# --------------------------------------------------------------------------
corpus = jload(os.path.join(REPO, "reports", "catalog_audit_replay_corpus.json"))
artifacts = [a for a in corpus["artifacts"] if a.get("readable")]
product_key = {(r["run_id"], r["issue_id"]) for r in corpus["rows"]}
say(f"readable artifacts={len(artifacts)}  product-lane (run_id,issue_id) keys={len(product_key)}")

# --------------------------------------------------------------------------
# Flatten every resolved_items row, and build the Terra join tables per artifact
# --------------------------------------------------------------------------
ROWS = []          # one dict per resolved_items row
for art in artifacts:
    doc = jload(art["artifact_path"])
    run_id = art["run_id"]

    # Terra join tables for this artifact ------------------------------------
    res = ((doc.get("analysis_debug") or {}).get("renovation_estimate_v5") or {}).get("result") or {}
    issue_to_cond = {}
    for oc in res.get("observed_conditions", []) or []:
        for iid in oc.get("issue_ids") or []:
            issue_to_cond[iid] = oc.get("condition_id")
    cond_to_verdict = {}
    for cr in res.get("condition_reviews", []) or []:
        cond_to_verdict[cr.get("condition_id")] = cr.get("verdict")
    cond_to_disp = {}
    for cd in res.get("condition_dispositions", []) or []:
        cond_to_disp[cd.get("condition_id")] = (cd.get("disposition"), cd.get("reason_code"),
                                                cd.get("terminal_route"))

    for pkey, photo in (doc.get("photos") or {}).items():
        scene_group = ((photo.get("scene") or {}).get("group")) or "other"
        for ri in ((photo.get("debug") or {}).get("resolved_items") or []):
            iid = ri.get("issue_id")
            cands = ri.get("candidates") or []
            sel = ri.get("resolved_item_id")
            rank = None
            for i, c in enumerate(cands):
                if c.get("item_id") == sel:
                    rank = i + 1
                    break
            cond = issue_to_cond.get(iid)
            ROWS.append({
                "run_id": run_id,
                "issue_id": iid,
                "photo_key": pkey,
                "scene_group": scene_group,
                "observation": ri.get("description") or "",
                "original_kind": ri.get("original_kind"),
                "resolved_kind": ri.get("resolved_kind"),
                "routing_reason": ri.get("routing_reason"),
                "path": ri.get("resolution_path"),
                "shortcut_reason": ri.get("shortcut_reason"),
                "selected": sel,
                "rank": rank,
                "candidates": cands,
                "raw_response": ri.get("raw_response"),
                "condition_id": cond,
                "verdict": cond_to_verdict.get(cond),
                "disposition": cond_to_disp.get(cond, (None, None, None))[0],
                "reason_code": cond_to_disp.get(cond, (None, None, None))[1],
                "product": (run_id, iid) in product_key,
            })

ALL = ROWS
PROD = [r for r in ROWS if r["product"]]
POPS = [("ALL", ALL), ("PRODUCT", PROD)]
say(f"ALL rows={len(ALL)}  PRODUCT rows={len(PROD)}")


def pct(n, d):
    return "n/a" if not d else f"{100.0 * n / d:.1f}%"


# ==========================================================================
# 1. Path mix and shortcut_reason breakdown
# ==========================================================================
say("\n=== 1. PATH MIX ===")
for name, pop in POPS:
    paths = collections.Counter(r["path"] for r in pop)
    reasons = collections.Counter(r["shortcut_reason"] for r in pop if r["path"] == "lexical_shortcut")
    say(f"{name}: {dict(paths)}   shortcut_reason={dict(reasons)}")

# ==========================================================================
# 2. Null resolutions
# ==========================================================================
say("\n=== 2. NULL RESOLUTIONS ===")
for name, pop in POPS:
    nulls = [r for r in pop if r["selected"] is None]
    say(f"{name}: nulls={len(nulls)}/{len(pop)} ({pct(len(nulls), len(pop))})")
    say(f"  by path: {dict(collections.Counter(r['path'] for r in nulls))}"
        f"   (shortcut-path nulls = {sum(1 for r in nulls if r['path'] == 'lexical_shortcut')})")
    bk = collections.Counter(r["original_kind"] for r in nulls)
    tk = collections.Counter(r["original_kind"] for r in pop)
    say("  by original_kind: " + ", ".join(
        f"{k}={bk[k]}/{tk[k]} ({pct(bk[k], tk[k])})" for k in sorted(tk)))
    bs = collections.Counter(r["scene_group"] for r in nulls)
    ts = collections.Counter(r["scene_group"] for r in pop)
    say("  by scene_group: " + ", ".join(
        f"{k}={bs[k]}/{ts[k]} ({pct(bs[k], ts[k])})" for k in sorted(ts)))

# ==========================================================================
# 3. Selected-rank distribution on the llm path
# ==========================================================================
say("\n=== 3. SELECTED RANK (llm path, non-null) ===")
for name, pop in POPS:
    sel = [r for r in pop if r["path"] == "llm" and r["selected"] is not None]
    d = collections.Counter(r["rank"] for r in sel)
    say(f"{name}: n={len(sel)}  " + "  ".join(f"rank{k}={d[k]}" for k in sorted(d, key=lambda x: (x is None, x))))
    say(f"  rank1 share={pct(d[1], len(sel))}   rank>=4={sum(v for k, v in d.items() if k and k >= 4)} "
        f"({pct(sum(v for k, v in d.items() if k and k >= 4), len(sel))})")
    off = [r for r in sel if r["rank"] is None]
    say(f"  selections NOT present in the rendered candidate list (hallucinated ids): {len(off)}"
        + ("  e.g. " + ", ".join(sorted({r['selected'] for r in off})[:8]) if off else ""))

# ==========================================================================
# 4. Shortcut token census
# ==========================================================================
# RULE (auditable):
#   For each lexical_shortcut row, re-derive the FIRING support_any terms of the
#   rank-1 candidate using the pipeline's own matcher, against the pipeline's own
#   normalisation of the observation (_normalize_signal_text, i.e. the exact string
#   the shortcut compares).  A firing term is CONDITION-BEARING iff any whitespace
#   token of the term starts with one of CONDITION_STEMS below (prefix test, the
#   same stem semantics term_matches itself uses).  Otherwise the term is BARE:
#   it names only a subject, component, material, spec or texture.
#   A row is "bare-only" iff it has >=1 firing term and EVERY firing term is bare.
CONDITION_STEMS = (
    # age / style currency
    "dated", "outdated", "old", "aging", "aged", "obsolete",
    # wear and surface degradation
    "worn", "wear", "stain", "discolor", "discolour", "peel", "flak", "fad",
    "scratch", "scuff", "chip", "crack", "dirty", "grim", "soil", "dingy",
    "weathered", "patchy", "patch", "uneven", "loose", "sag", "lean", "warp",
    # damage / failure
    "damag", "broken", "break", "hole", "gap", "separation", "missing", "rot",
    "rust", "corrosion", "leak", "mold", "mould", "moss", "algae", "unfinished",
    "overgrown", "weed", "deteriorat", "crumbl", "buckl", "torn", "tear",
)


def is_condition_term(term: str) -> bool:
    t = term[:-1] if term.endswith("$") else term
    return any(w.startswith(s) for w in t.lower().split() for s in CONDITION_STEMS)


say("\n=== 4. SHORTCUT TOKEN CENSUS ===")
say("CONDITION_STEMS used: " + ", ".join(CONDITION_STEMS))
for name, pop in POPS:
    sc = [r for r in pop if r["path"] == "lexical_shortcut"]
    bare_rows, cond_rows, nofire_rows = [], [], []
    bare_tok = collections.Counter()
    cond_tok = collections.Counter()
    bare_items = collections.Counter()
    for r in sc:
        cands = r["candidates"]
        if not cands:
            nofire_rows.append(r)
            continue
        obs = _normalize_signal_text(r["observation"])
        hits = [t for t in (cands[0].get("support_any") or []) if term_matches(t, obs)]
        r["fired"] = hits
        if not hits:
            nofire_rows.append(r)
            continue
        if all(not is_condition_term(t) for t in hits):
            bare_rows.append(r)
            r["tokclass"] = "bare"
            for t in hits:
                bare_tok[t] += 1
            bare_items[r["selected"]] += 1
        else:
            cond_rows.append(r)
            r["tokclass"] = "condition"
            for t in hits:
                cond_tok[t] += 1
    say(f"{name}: shortcut rows={len(sc)}  with>=1 firing term={len(sc) - len(nofire_rows)}  "
        f"bare-only={len(bare_rows)}  condition-bearing={len(cond_rows)}  no-firing-term={len(nofire_rows)}")
    say(f"  bare-only share of firing rows = {pct(len(bare_rows), len(sc) - len(nofire_rows))}")
    say("  top bare firing tokens: " + ", ".join(f"{t}({c})" for t, c in bare_tok.most_common(15)))
    say("  items most often reached bare: " + ", ".join(f"{t}({c})" for t, c in bare_items.most_common(12)))
    if name == "ALL":
        say("  rows with NO firing term (these are the component_condition_overlap rows): "
            + ", ".join(f"{r['shortcut_reason']}/{r['selected']}" for r in nofire_rows))
        say("  top condition-bearing firing tokens: "
            + ", ".join(f"{t}({c})" for t, c in cond_tok.most_common(12)))

# ==========================================================================
# 5. Candidate-list sizes vs the 3.1-era retrieval pool
# ==========================================================================
say("\n=== 5. CANDIDATE LIST SIZES / GUARDRAIL SLOT LOSS ===")
cat31_path = os.path.join(SCRATCH, "catalog_31.json")
if not os.path.exists(cat31_path):
    blob = subprocess.run(["git", "show", "58c073e:tools/issue_catalog_kind_v2.json"],
                          cwd=REPO, capture_output=True)
    open(cat31_path, "wb").write(blob.stdout)
cat31 = jload(cat31_path)
items31 = cat31["items"]
say(f"3.1-era catalog (git 58c073e) items={len(items31)} version={cat31.get('version')}")

# pool(scene_group, kind) = items of that kind whose scene_groups contain the group.
pool = collections.Counter()
for it in items31:
    for g in it.get("scene_groups") or []:
        pool[(g, it.get("kind"))] += 1

for name, pop in POPS:
    sizes = collections.Counter(len(r["candidates"]) for r in pop)
    say(f"{name}: len(candidates) distribution = "
        + ", ".join(f"{k}:{sizes[k]}" for k in sorted(sizes)))
    lost = collections.Counter()
    by_cell = collections.Counter()
    short = 0
    for r in pop:
        exp = min(8, pool.get((r["scene_group"], r["resolved_kind"]), 0))
        got = len(r["candidates"])
        if got < exp:
            short += 1
            lost[exp - got] += 1
            by_cell[(r["scene_group"], r["resolved_kind"], exp - got)] += 1
    say(f"  rows receiving fewer than min(8,pool): {short}/{len(pop)} ({pct(short, len(pop))})")
    say("  slots lost: " + ", ".join(f"{k}:{lost[k]}" for k in sorted(lost)))
    if name == "ALL":
        say("  by (scene_group, kind, slots_lost):")
        for (g, k, n), c in sorted(by_cell.items(), key=lambda x: -x[1]):
            say(f"    {g:12s} {k:14s} lost={n}  rows={c}   pool={pool.get((g, k), 0)}")

# ==========================================================================
# 6. Guardrail attribution from the checkpoint-after snapshot
# ==========================================================================
say("\n=== 6. GUARDRAIL ATTRIBUTION (reports/catalog_audit_replay_snapshot_checkpoint_after.json) ===")
snap = jload(os.path.join(REPO, "reports", "catalog_audit_replay_snapshot_checkpoint_after.json"))
snap_rows = snap["rows"]
say(f"snapshot rows={len(snap_rows)}  arm={snap.get('arm')}  catalog={snap['catalog'].get('version')} "
    f"git_head={snap.get('git_head')}")
drop_rank = collections.Counter()
drop_item = collections.Counter()
drop_role = collections.Counter()
drop_item_role_terms = collections.Counter()
rows_with_drop = 0
for r in snap_rows:
    gd = r.get("guardrail_drops") or {}
    if not gd:
        continue
    rows_with_drop += 1
    pre_rank = {c["item_id"]: c["rank"] for c in r.get("pre_guardrail") or []}
    for item, info in gd.items():
        if not info.get("dropped"):
            continue
        rk = pre_rank.get(item)
        drop_rank[rk] += 1
        drop_item[item] += 1
        drop_role[info.get("role")] += 1
        drop_item_role_terms[(item, info.get("role"), tuple(info.get("terms") or []))] += 1
say(f"rows with >=1 guardrail drop: {rows_with_drop}/{len(snap_rows)} ({pct(rows_with_drop, len(snap_rows))})")
say(f"total drops={sum(drop_rank.values())}  by pre-guardrail rank: "
    + ", ".join(f"r{k}:{drop_rank[k]}" for k in sorted(drop_rank, key=lambda x: (x is None, x))))
top3 = sum(v for k, v in drop_rank.items() if k in (1, 2, 3))
say(f"drops that were at pre-guardrail rank 1/2/3: {top3} ({pct(top3, sum(drop_rank.values()))} of drops)")
say(f"by role: {dict(drop_role)}")
say("most-dropped items:")
for it, c in drop_item.most_common(15):
    say(f"  {it:44s} {c}")
say("item / role / terms detail (top 12):")
for (it, role, terms), c in drop_item_role_terms.most_common(12):
    say(f"  {c:5d}  {it:40s} {role:12s} {list(terms)}")

# ==========================================================================
# 7. Near-threshold shortcut census
# ==========================================================================
say("\n=== 7. NEAR-THRESHOLD SHORTCUT CENSUS ===")
MARGIN_GATE, SCORE_GATE = 0.03, 0.72
near_margin, near_score = [], []
for r in snap_rows:
    sc = r.get("shortcut") or {}
    m, ts = sc.get("margin"), sc.get("top_score")
    top = (r.get("post_guardrail") or [{}])[0].get("item_id")
    if m is not None and abs(m - MARGIN_GATE) <= 0.002:
        near_margin.append((r["row_id"], top, m))
    if ts is not None and abs(ts - SCORE_GATE) <= 0.01:
        near_score.append((r["row_id"], top, ts))
say(f"rows with |margin - 0.03| <= 0.002 : {len(near_margin)}   (prior session recorded 91)")
say("  items involved: " + ", ".join(f"{it}({c})" for it, c in
                                     collections.Counter(x[1] for x in near_margin).most_common()))
say(f"    of these, margin just BELOW gate (<0.03): {sum(1 for x in near_margin if x[2] < MARGIN_GATE)}"
    f"   just AT/ABOVE: {sum(1 for x in near_margin if x[2] >= MARGIN_GATE)}")
say(f"rows with |top_score - 0.72| <= 0.01 : {len(near_score)}")
say("  items involved: " + ", ".join(f"{it}({c})" for it, c in
                                     collections.Counter(x[1] for x in near_score).most_common()))
say(f"    below 0.72: {sum(1 for x in near_score if x[2] < SCORE_GATE)}   "
    f"at/above: {sum(1 for x in near_score if x[2] >= SCORE_GATE)}")
both = [x for x in near_margin if x[0] in {y[0] for y in near_score}]
say(f"rows near BOTH gates: {len(both)}")

# ==========================================================================
# 8. Terra verdict joins
# ==========================================================================
say("\n=== 8. TERRA VERDICT JOIN ===")


def unsup(rows):
    j = [r for r in rows if r["verdict"] is not None]
    u = sum(1 for r in j if r["verdict"] == "unsupported")
    return u, len(j), pct(u, len(j))


for name, pop in POPS:
    say(f"--- {name} ---")
    joined = [r for r in pop if r["verdict"] is not None]
    say(f"rows joined to a Terra verdict: {len(joined)}/{len(pop)}  "
        f"verdicts={dict(collections.Counter(r['verdict'] for r in joined))}")
    for p in ("lexical_shortcut", "llm"):
        u, n, s = unsup([r for r in pop if r["path"] == p])
        say(f"  path={p:16s} unsupported {u}/{n} = {s}")
    for cls in ("bare", "condition"):
        u, n, s = unsup([r for r in pop if r["path"] == "lexical_shortcut" and r.get("tokclass") == cls])
        say(f"  shortcut tokclass={cls:10s} unsupported {u}/{n} = {s}")
    for k in ("defect", "degradation", "modernization"):
        u, n, s = unsup([r for r in pop if r["original_kind"] == k])
        say(f"  kind={k:14s} unsupported {u}/{n} = {s}")
        u2, n2, s2 = unsup([r for r in pop if r["original_kind"] == k and r["path"] == "llm"])
        say(f"      llm-path only: {u2}/{n2} = {s2}")
    bands = {"rank1": lambda x: x == 1, "rank2-3": lambda x: x in (2, 3), "rank4+": lambda x: x is not None and x >= 4}
    for bname, f in bands.items():
        u, n, s = unsup([r for r in pop if r["path"] == "llm" and r["rank"] is not None and f(r["rank"])])
        say(f"  llm {bname:8s} unsupported {u}/{n} = {s}")
    if name == "ALL":
        say("  per-item unsupported rate (n>=12):")
        byitem = collections.defaultdict(list)
        for r in pop:
            if r["verdict"] is not None and r["selected"]:
                byitem[r["selected"]].append(r)
        stats = []
        for it, rs in byitem.items():
            if len(rs) >= 12:
                u = sum(1 for r in rs if r["verdict"] == "unsupported")
                stats.append((u / len(rs), u, len(rs), it))
        for rate, u, n, it in sorted(stats, reverse=True):
            say(f"    {it:46s} {u:4d}/{n:4d}  {100 * rate:5.1f}%")

# ==========================================================================
# 9. Re-run the frozen shortcut self-test
# ==========================================================================
# Replay the REAL shortcut predicate over every stored candidate block and check
# it reproduces the recorded resolution_path / resolved id / shortcut_reason.
say("\n=== 9. FROZEN SHORTCUT SELF-TEST REPLAY ===")
say("recorded in corpus: " + json.dumps(corpus["frozen_shortcut_selftest"])[:200])
mismatch = []
for r in ALL:
    rid, rkind, reason = _resolve_candidate_via_lexical_shortcut(
        r["observation"], r["candidates"], kind=r["original_kind"] or "")
    got_path = "lexical_shortcut" if reason else "llm"
    if got_path != r["path"] or (reason and (rid != r["selected"] or reason != r["shortcut_reason"])):
        mismatch.append((r["run_id"], r["issue_id"], r["path"], got_path, r["shortcut_reason"], reason))
say(f"rows considered={len(ALL)}  mismatches={len(mismatch)}  "
    f"result={'pass' if not mismatch else 'FAIL'}")
for m in mismatch[:10]:
    say("  " + str(m))

# ==========================================================================
# 10. Extras the prior numbers did not cover
# ==========================================================================
say("\n=== 10. EXTRAS ===")
# 10a. null rate by kind on the llm path only (nulls can only happen there)
for name, pop in POPS:
    llm = [r for r in pop if r["path"] == "llm"]
    n = sum(1 for r in llm if r["selected"] is None)
    say(f"{name}: llm-path null rate {n}/{len(llm)} = {pct(n, len(llm))}")
# 10b. how often the shortcut's rank-1 pick would also have been rank 1 for the llm
sc_all = [r for r in ALL if r["path"] == "lexical_shortcut"]
say(f"shortcut rows where the returned id is not candidates[0]: "
    f"{sum(1 for r in sc_all if r['candidates'] and r['selected'] != r['candidates'][0]['item_id'])}")
# 10c. most selected items overall + their path split
sel_counter = collections.Counter(r["selected"] for r in ALL if r["selected"])
say("top selected items (ALL) with shortcut share:")
for it, c in sel_counter.most_common(15):
    s = sum(1 for r in ALL if r["selected"] == it and r["path"] == "lexical_shortcut")
    say(f"  {it:46s} n={c:4d}  shortcut={s:4d} ({pct(s, c)})")
# 10d. raw_response non-null-but-null-id sanity, and malformed responses
llm_all = [r for r in ALL if r["path"] == "llm"]
say(f"llm rows with raw_response missing: {sum(1 for r in llm_all if not r['raw_response'])}")
# 10e. disposition mix for shortcut vs llm
for p in ("lexical_shortcut", "llm"):
    d = collections.Counter(r["disposition"] for r in ALL if r["path"] == p and r["disposition"])
    say(f"dispositions ({p}): {dict(d)}")

with open(os.path.join(SCRATCH, "pass2d_corpus_stats_output.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT))
