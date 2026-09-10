import json, os, sys, glob
from collections import Counter, defaultdict
sys.path.insert(0, os.getcwd())

corpus = json.load(open("reports/catalog_audit_replay_corpus.json", encoding="utf-8"))
arts = [a for a in corpus["artifacts"] if a.get("readable") and a.get("artifact_path") and os.path.exists(a["artifact_path"])]
print("=" * 96)
print("C1. SOURCE OF EVERY TERRA VERDICT: canary vs production")
src = Counter(a["source"] for a in arts)
print("  readable artifacts by source:", dict(src))
rows_by_src = Counter(); v5_by_src = Counter()
cond_rows = []
for a in arts:
    d = json.load(open(a["artifact_path"], encoding="utf-8"))
    n = sum(len((ph.get("debug") or {}).get("resolved_items") or []) for ph in (d.get("photos") or {}).values())
    rows_by_src[a["source"]] += n
    v5 = ((d.get("analysis_debug") or {}).get("renovation_estimate_v5") or {}).get("result") or {}
    has = bool(v5.get("condition_reviews"))
    v5_by_src[(a["source"], has)] += 1
    if has:
        revs = {}
        for r in v5.get("condition_reviews") or []:
            revs.setdefault(r["condition_id"], r)
        for c in v5.get("observed_conditions") or []:
            rv = revs.get(c["condition_id"])
            if rv:
                cond_rows.append(dict(source=a["source"], cond=c["condition_id"], item=c.get("catalog_item_id"),
                                      verdict=rv.get("verdict"), n_issues=len(c.get("issue_ids") or [])))
print("  resolved_items rows by source:", dict(rows_by_src))
print("  artifacts with/without a v5 condition_reviews block:", {f"{k[0]}/{'v5' if k[1] else 'NO-v5'}": v for k, v in v5_by_src.items()})
print(f"  -> distinct Terra-reviewed CONDITIONS: {len(cond_rows)}; by source: {dict(Counter(c['source'] for c in cond_rows))}")
vc = Counter(c["verdict"] for c in cond_rows)
print(f"  -> verdicts over CONDITIONS: {dict(vc)}; unsupported rate = {vc['unsupported']/len(cond_rows):.1%}")
print(f"  -> issues per condition: mean {sum(c['n_issues'] for c in cond_rows)/len(cond_rows):.2f}, "
      f"conditions with >1 issue: {sum(1 for c in cond_rows if c['n_issues']>1)}")

print()
print("=" * 96)
print("C2. TERRA BY PATH, scored on CONDITIONS instead of rows")
path_by_cond = defaultdict(set)
for a in arts:
    d = json.load(open(a["artifact_path"], encoding="utf-8"))
    v5 = ((d.get("analysis_debug") or {}).get("renovation_estimate_v5") or {}).get("result") or {}
    if not v5.get("condition_reviews"):
        continue
    i2p = {}
    for ph in (d.get("photos") or {}).values():
        for r in (ph.get("debug") or {}).get("resolved_items") or []:
            i2p[r.get("issue_id")] = r.get("resolution_path")
    for c in v5.get("observed_conditions") or []:
        for iid in c.get("issue_ids") or []:
            if iid in i2p:
                path_by_cond[c["condition_id"]].add(i2p[iid])
byp = defaultdict(Counter)
for c in cond_rows:
    p = path_by_cond.get(c["cond"], set())
    key = "shortcut_only" if p == {"lexical_shortcut"} else ("llm_only" if p == {"llm"} else ("mixed" if p else "unknown"))
    byp[key][c["verdict"]] += 1
for k in sorted(byp):
    t = sum(byp[k].values())
    print(f"  {k:14s} n={t:4d} unsupported={byp[k]['unsupported']/t:5.1%}  ({dict(byp[k])})")

print()
print("=" * 96)
print("C3. run_2 EXISTS AND WAS NEVER OPENED?")
r1 = glob.glob("artifacts_canary/renovation_session9_20260818/run_1/candidate/*/*/photo_intel_debug.json")
r2 = glob.glob("artifacts_canary/renovation_session9_20260818/run_2/candidate/*/*/photo_intel_debug.json")
print(f"  run_1 candidate artifacts: {len(r1)} | run_2 candidate artifacts: {len(r2)}")
pinned = {a["artifact_path"].replace("\\", "/") for a in arts}
print(f"  run_2 artifacts inside the pinned corpus: {sum(1 for p in r2 if p.replace(chr(92),'/') in pinned)}")
if r2:
    tot = 0; per_item = defaultdict(Counter)
    for p in r2:
        d = json.load(open(p, encoding="utf-8"))
        tot += sum(len((ph.get("debug") or {}).get("resolved_items") or []) for ph in (d.get("photos") or {}).values())
        v5 = ((d.get("analysis_debug") or {}).get("renovation_estimate_v5") or {}).get("result") or {}
        revs = {}
        for r in v5.get("condition_reviews") or []:
            revs.setdefault(r["condition_id"], r)
        for c in v5.get("observed_conditions") or []:
            rv = revs.get(c["condition_id"])
            if rv: per_item[c.get("catalog_item_id")][rv.get("verdict")] += 1
    print(f"  run_2 resolved_items rows: {tot}")
    print("  per-item unsupported, run_2 (conditions), for the items the draft named:")
    r1_item = defaultdict(Counter)
    for a in arts:
        d = json.load(open(a["artifact_path"], encoding="utf-8"))
        v5 = ((d.get("analysis_debug") or {}).get("renovation_estimate_v5") or {}).get("result") or {}
        revs = {}
        for r in v5.get("condition_reviews") or []:
            revs.setdefault(r["condition_id"], r)
        for c in v5.get("observed_conditions") or []:
            rv = revs.get(c["condition_id"])
            if rv: r1_item[c.get("catalog_item_id")][rv.get("verdict")] += 1
    for k in ("brick_weathered_or_discolored", "paint_refresh_recommended", "concrete_driveway_surface_wear",
              "exterior_siding_discoloration_fading", "peeling_or_discolored_paint", "wall_scuffs_marks_or_dents"):
        a1 = r1_item.get(k, Counter()); a2 = per_item.get(k, Counter())
        n1, n2 = sum(a1.values()), sum(a2.values())
        f = lambda c, n: f"{c['unsupported']}/{n} = {c['unsupported']/n:.0%}" if n else "n/a"
        print(f"    {k:40s} pinned(run_1): {f(a1,n1):14s}  run_2: {f(a2,n2)}")

print()
print("=" * 96)
print("C4. ROW 5 - is the bath_fixtures billing actually live?")
for a in arts:
    if a["property_key"] != "redfin_125779232":
        continue
    d = json.load(open(a["artifact_path"], encoding="utf-8"))
    v5 = ((d.get("analysis_debug") or {}).get("renovation_estimate_v5") or {}).get("result") or {}
    for c in v5.get("observed_conditions") or []:
        if c.get("catalog_item_id") == "bath_fixtures_stained_or_worn":
            print(f"  condition {c['condition_id']} issues={c.get('issue_ids')}")
    for w in (v5.get("work_items") or d.get("work_items") or []):
        if "bath_fixtures_stained_or_worn" in json.dumps(w):
            print("  work_item:", {k: w.get(k) for k in ("work_item_id", "status", "reason_code", "action_code", "catalog_item_ids", "condition_ids")})
    for w in (v5.get("work_items") or d.get("work_items") or []):
        if w.get("action_code") == "FIXTURES_UPDATE":
            print("  FIXTURES_UPDATE line:", {k: w.get(k) for k in ("work_item_id", "status", "reason_code", "catalog_item_ids", "cost_low", "cost_high")})

print()
print("=" * 96)
print("C5. STEVEN'S RETAG ON ROW 4 (rc_9c702893a6ce) AND ROW 3 (rc_eef1daf73e2a)")
import re
txt = open("docs/RESULT_retag_mechanism_vs_perception_20260828.md", encoding="utf-8").read()
for cid in ("rc_9c702893a6ce", "rc_eef1daf73e2a", "rc_3a19f759f2d9"):
    for line in txt.splitlines():
        if cid in line:
            print("  " + line.strip()[:260])

print()
print("=" * 96)
print("C6. WHAT IS THE 2.5M FIGURE IN THE REPO?")
for p in ("docs/analysis/session8_terra_budget_denominator.md",):
    if os.path.exists(p):
        for i, line in enumerate(open(p, encoding="utf-8").read().splitlines(), 1):
            if "2,500,000" in line or "2.5" in line or "free" in line.lower():
                print(f"  {p}:{i}: {line.strip()[:200]}")
m = json.load(open("reports/catalog_audit_live_experiment_manifest_v2.json", encoding="utf-8"))
print("  manifest_v2 budget:", json.dumps(m.get("budget"))[:400])
print("  manifest_v2 status:", m.get("status"))

print()
print("=" * 96)
print("C7. CAP-016 REFERRAL AND THE BLINDS DESCRIPTION")
wl = open("reports/catalog_checkpoint_pass2d_worklist.json", encoding="utf-8").read()
print("  'CAP-016' occurrences in the shipped worklist:", wl.count("CAP-016"))
print("  'ceiling texture' occurrences in the shipped worklist:", wl.count("ceiling texture"))
cat = json.load(open("tools/issue_catalog_kind_v2.json", encoding="utf-8"))
b = [i for i in cat["items"] if i["id"] == "window_blinds_basic_or_plain"][0]
print("  window_blinds_basic_or_plain.description:", b["description"])
