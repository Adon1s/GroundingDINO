"""B0: re-score every Terra-derived number on CONDITIONS (not replicated rows), split by source,
fold in run_2 as a replication arm, and recompute the null/window and text-reproducibility facts.
Offline, deterministic, no provider/sidecar calls."""
import json, os, sys, glob
from collections import Counter, defaultdict
sys.path.insert(0, os.getcwd())
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
corpus = json.load(open("reports/catalog_audit_replay_corpus.json", encoding="utf-8"))
pinned = [a for a in corpus["artifacts"] if a.get("readable") and a.get("artifact_path") and os.path.exists(a["artifact_path"])]
run2 = sorted(glob.glob("artifacts_canary/renovation_session9_20260818/run_2/candidate/*/*/photo_intel_debug.json"))
draft = json.load(open(f"{SP}/pass2d_diagnosis_20260909.draft.json", encoding="utf-8"))
strong7 = [i["item_id"] for i in sorted(draft["catalog_divergence"]["items"], key=lambda i: i["severity_rank"])[:7]]

def load_arm(paths, label):
    rows, conds, photos = [], [], {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        prop = d.get("property", {}).get("property_key") or d.get("property_key") or p.split(os.sep)[-3]
        run = d.get("run", {}).get("run_id") or p.split(os.sep)[-2]
        i2r = {}
        for pk, ph in (d.get("photos") or {}).items():
            descs = []
            for r in (ph.get("debug") or {}).get("resolved_items") or []:
                ids = [c["item_id"] for c in r.get("candidates") or []]
                sel = r.get("resolved_item_id")
                rec = dict(prop=prop, run=run, photo=r.get("source_photo_key") or pk, issue=r.get("issue_id"), obs=r.get("description"),
                           kind=r.get("original_kind"), path=r.get("resolution_path"), sel=sel,
                           rank=(ids.index(sel) + 1) if sel in ids else None, n_cands=len(ids))
                rows.append(rec); i2r[rec["issue"]] = rec; descs.append(rec["obs"])
            photos[(prop, pk)] = descs
        v5 = ((d.get("analysis_debug") or {}).get("renovation_estimate_v5") or {}).get("result") or {}
        revs = {}
        for r in v5.get("condition_reviews") or []:
            revs.setdefault(r["condition_id"], r)
        for c in v5.get("observed_conditions") or []:
            rv = revs.get(c["condition_id"])
            if not rv:
                continue
            issues = [i2r[i] for i in (c.get("issue_ids") or []) if i in i2r]
            paths = {i["path"] for i in issues}
            conds.append(dict(prop=prop, cond=c["condition_id"], item=c.get("catalog_item_id"), kind=c.get("catalog_kind"),
                              verdict=rv.get("verdict"), n_issues=len(c.get("issue_ids") or []), n_joined=len(issues),
                              pathclass=("shortcut_only" if paths == {"lexical_shortcut"} else "llm_only" if paths == {"llm"} else "mixed" if paths else "unjoined"),
                              ranks=[i["rank"] for i in issues if i["path"] == "llm" and i["rank"]]))
    return rows, conds, photos

def unsup(cs):
    n = len(cs); u = sum(1 for c in cs if c["verdict"] == "unsupported"); ca = sum(1 for c in cs if c["verdict"] == "cannot_assess")
    return dict(n=n, unsupported=u, cannot_assess=ca, rate=(u / n) if n else None)

out = {}
r1_rows, r1_conds, r1_photos = load_arm([a["artifact_path"] for a in pinned], "pinned")
r2_rows, r2_conds, r2_photos = load_arm(run2, "run_2")
src_of = {a["artifact_path"]: a["source"] for a in pinned}
out["populations"] = dict(pinned_artifacts=len(pinned), pinned_rows=len(r1_rows), pinned_conditions_with_verdict=len(r1_conds),
                          pinned_condition_source=dict(Counter("canary" if any(c["prop"] == a["property_key"] and a["source"] == "canary" for a in pinned) else "production" for c in r1_conds)),
                          run2_artifacts=len(run2), run2_rows=len(r2_rows), run2_conditions_with_verdict=len(r2_conds),
                          issues_per_condition_pinned=round(sum(c["n_issues"] for c in r1_conds) / len(r1_conds), 2),
                          note="all verdict-carrying artifacts are canary; the 8 production artifacts are v4-era with no v5 block")

def tables(conds, label):
    t = {}
    t["overall"] = unsup(conds)
    t["by_pathclass"] = {k: unsup([c for c in conds if c["pathclass"] == k]) for k in ("shortcut_only", "llm_only", "mixed", "unjoined")}
    t["by_kind"] = {k: unsup([c for c in conds if c["kind"] == k]) for k in ("defect", "degradation", "modernization")}
    single_llm = [c for c in conds if c["pathclass"] == "llm_only" and len(c["ranks"]) == 1]
    t["by_rank_band_single_issue_llm_conditions"] = {
        "rank1": unsup([c for c in single_llm if c["ranks"][0] == 1]),
        "rank2-3": unsup([c for c in single_llm if c["ranks"][0] in (2, 3)]),
        "rank4+": unsup([c for c in single_llm if c["ranks"][0] >= 4]),
        "note": f"{len(single_llm)} single-issue llm conditions of {len(conds)}",
    }
    per = defaultdict(list)
    for c in conds: per[c["item"]].append(c)
    t["per_item"] = {k: unsup(v) for k, v in per.items()}
    t["divergence_strong7_vs_rest"] = {"strong7": unsup([c for c in conds if c["item"] in strong7]),
                                        "rest": unsup([c for c in conds if c["item"] not in strong7]), "strong7_items": strong7}
    return t
out["pinned_run1"] = tables(r1_conds, "pinned")
out["run2_replication"] = tables(r2_conds, "run_2")

# per-item comparison table for items with n>=12 in either arm
pi1, pi2 = out["pinned_run1"]["per_item"], out["run2_replication"]["per_item"]
cmp = []
for k in set(pi1) | set(pi2):
    a, b = pi1.get(k, dict(n=0, unsupported=0, rate=None)), pi2.get(k, dict(n=0, unsupported=0, rate=None))
    if max(a["n"], b["n"]) >= 12:
        cmp.append(dict(item=k, pinned_n=a["n"], pinned_rate=a["rate"], run2_n=b["n"], run2_rate=b["rate"],
                        unstable=(a["n"] < 15 or b["n"] < 15), delta=(None if a["rate"] is None or b["rate"] is None else round(b["rate"] - a["rate"], 3))))
cmp.sort(key=lambda x: -(x["pinned_rate"] or 0))
out["per_item_replication"] = cmp

# nulls: window share over ALL nulls, degradation/defect nulls, product-lane nulls
prod = {(r["run_id"], r["issue_id"]) for r in corpus["rows"]}
nulls = [r for r in r1_rows if r["path"] == "llm" and r["sel"] is None]
dd = [r for r in nulls if r["kind"] in ("degradation", "defect")]
pl = [r for r in nulls if (r["run"], r["issue"]) in prod]
pldd = [r for r in pl if r["kind"] in ("degradation", "defect")]
w = lambda rs: sum(1 for r in rs if "window" in (r["obs"] or "").lower())
out["nulls_window_share"] = dict(all_nulls=dict(n=len(nulls), window=w(nulls)), degradation_defect_nulls=dict(n=len(dd), window=w(dd)),
                                 product_lane_nulls=dict(n=len(pl), window=w(pl)), product_lane_degradation_defect_nulls=dict(n=len(pldd), window=w(pldd)),
                                 note="the product lane admits non-modernization rows only via REPLAY_STEMS, which contains 'window'")

# text reproducibility run_1 vs run_2 on shared (property, photo)
shared = set(r1_photos) & set(r2_photos)
ident = sum(1 for k in shared if sorted(r1_photos[k]) == sorted(r2_photos[k]))
r1_descs = [(k, dsc) for k in shared for dsc in r1_photos[k]]
verbatim = sum(1 for k, dsc in r1_descs if dsc in r2_photos[k])
out["text_reproducibility_run1_vs_run2"] = dict(shared_photos=len(shared), photos_with_identical_observation_sets=ident,
                                                run1_observations_on_shared_photos=len(r1_descs), reproduced_verbatim_in_run2=verbatim)

# Terra ledger untouched?
led = "artifacts_canary/catalog_audit_session6_20260907/.renovation_architecture/terra_usage.sqlite3"
import datetime
out["terra_ledger"] = dict(path=led, exists=os.path.exists(led),
                           mtime=datetime.datetime.fromtimestamp(os.path.getmtime(led)).isoformat() if os.path.exists(led) else None,
                           newer_ledgers=[p for p in glob.glob("artifacts_canary/**/terra_usage.sqlite3", recursive=True) if os.path.getmtime(p) > datetime.datetime(2026, 9, 8).timestamp()])

json.dump(out, open(f"{SP}/b0_rescore.json", "w", encoding="utf-8"), indent=1)
P = out["pinned_run1"]; R = out["run2_replication"]
f = lambda d: f"{d['unsupported']}/{d['n']} = {d['rate']:.1%}" if d["n"] else "n/a"
print("POPULATIONS:", json.dumps(out["populations"]))
print("\nOVERALL     pinned:", f(P["overall"]), "| run_2:", f(R["overall"]))
print("BY PATHCLASS (conditions):")
for k in ("shortcut_only", "llm_only", "mixed"):
    print(f"  {k:14s} pinned {f(P['by_pathclass'][k]):22s} run_2 {f(R['by_pathclass'][k])}")
print("BY KIND:")
for k in ("defect", "degradation", "modernization"):
    print(f"  {k:14s} pinned {f(P['by_kind'][k]):22s} run_2 {f(R['by_kind'][k])}")
print("BY RANK BAND (single-issue llm conditions):", P["by_rank_band_single_issue_llm_conditions"]["note"], "/", R["by_rank_band_single_issue_llm_conditions"]["note"])
for k in ("rank1", "rank2-3", "rank4+"):
    print(f"  {k:8s} pinned {f(P['by_rank_band_single_issue_llm_conditions'][k]):22s} run_2 {f(R['by_rank_band_single_issue_llm_conditions'][k])}")
print("DIVERGENCE strong7 vs rest:  pinned", f(P["divergence_strong7_vs_rest"]["strong7"]), "vs", f(P["divergence_strong7_vs_rest"]["rest"]),
      "| run_2", f(R["divergence_strong7_vs_rest"]["strong7"]), "vs", f(R["divergence_strong7_vs_rest"]["rest"]))
print("PER-ITEM (n>=12 in either arm), pinned -> run_2 [unstable = n<15 in either]:")
for x in cmp[:16]:
    pr = f"{x['pinned_rate']:.0%}" if x["pinned_rate"] is not None else "n/a"; rr = f"{x['run2_rate']:.0%}" if x["run2_rate"] is not None else "n/a"
    print(f"  {x['item']:42s} {pr:>5s} (n={x['pinned_n']:3d}) -> {rr:>5s} (n={x['run2_n']:3d}) {'UNSTABLE' if x['unstable'] else ''}")
print("NULL WINDOW SHARE:", json.dumps(out["nulls_window_share"]))
print("TEXT REPRODUCIBILITY:", json.dumps(out["text_reproducibility_run1_vs_run2"]))
print("TERRA LEDGER:", json.dumps(out["terra_ledger"]))
