"""Analyse arms S, N and the Terra resolver comparison. Judgement is on whether an answer represents
the supplied observation; Terra verdicts are not used as proof of correct selection."""
import json, os, sys
from collections import Counter, defaultdict
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
res = defaultdict(dict)
for line in open(f"{SP}/sn_results.jsonl", encoding="utf-8"):
    r = json.loads(line)
    res[(r["arm"], r["replica"])][r["issue"]] = r
pop = json.load(open(f"{SP}/sn_populations.json", encoding="utf-8"))
S = {r["issue"]: r for r in pop["S"]}
N = {r["issue"]: r for r in pop["N"]}
ANCH = {"979a8dd3f7eab927": "C6 range", "280f2d14bdd2b082": "C7 faucet", "0bb902dec263ed01": "C8 kitchen"}
NULLC = {"6dce1550f7c7a08e": "D2 swirl-plaster (must stay null)", "32fe017ce31ed9b4": "D3 laundry cabinets (must stay null)"}
C8_OK = {"dated_overall_decor_style", "appliances_dated_or_basic", "cabinets_dated_style", "older_flooring_style", "outdated_kitchen_finishes"}
err = [r for k in res for r in res[k].values() if r.get("error")]
print("ERRORS:", len(err), [f"{r['arm']}:{r['issue']}:{r['error'][:60]}" for r in err[:5]])

print("\n" + "=" * 104)
print("ARM S — Rule E: the 25 shortcut rows routed to the model instead (candidate lists held fixed)")
sq = res.get(("S_qwen_base", 1), {}); st = res.get(("S_terra_base", 1), {})
if sq:
    same = chg = nul = 0
    print(f"{'':2s} {'head/fired':30s} {'shortcut gave':32s} {'qwen':30s} {'terra':30s}")
    for iid, row in S.items():
        q = sq.get(iid, {}); t = st.get(iid, {})
        qa, ta = q.get("resolved"), t.get("resolved")
        flag = "**" if row.get("human_confirmed_wrong_subject") else "  "
        if qa == row["stored_sel"]: same += 1
        elif qa is None: nul += 1
        else: chg += 1
        print(f"{flag} {row['note'][:30]:30s} {str(row['stored_sel'])[:32]:32s} {str(qa)[:30]:30s} {str(ta)[:30]:30s}")
        print(f"   {row['obs'][:100]}")
    print(f"\n  qwen vs shortcut: same {same} | different item {chg} | null {nul}  (of {len(S)})")
    print(f"  terra vs shortcut: same {sum(1 for i,r in S.items() if st.get(i,{}).get('resolved')==r['stored_sel'])} | "
          f"null {sum(1 for i in S if st.get(i,{}).get('resolved') is None)}")

print("\n" + "=" * 104)
print("ARM N — base vs modified prompt on local Qwen, plus Terra with the UNCHANGED prompt")
nb, nm, nt = res.get(("N_qwen_base", 1), {}), res.get(("N_qwen_mod", 1), {}), res.get(("N_terra_base", 1), {})
if nb and nm:
    print("\n-- anchors (Steven: a listed candidate fit the observation) --")
    for iid, lab in ANCH.items():
        b, m, t = nb.get(iid, {}).get("resolved"), nm.get(iid, {}).get("resolved"), nt.get(iid, {}).get("resolved")
        ok = (lambda v: (v in C8_OK) if iid == "0bb902dec263ed01" else bool(v))
        print(f"  {lab:14s} base={str(b)[:30]:30s} mod={str(m)[:30]:30s} terra={str(t)[:30]:30s} | mod_ok={ok(m)} terra_ok={ok(t)}")
    print("\n-- must-stay-null controls --")
    for iid, lab in NULLC.items():
        b, m, t = nb.get(iid, {}).get("resolved"), nm.get(iid, {}).get("resolved"), nt.get(iid, {}).get("resolved")
        print(f"  {lab:38s} base={str(b)[:26]:26s} mod={str(m)[:26]:26s} terra={str(t)[:26]:26s} | mod_held={m is None} terra_held={t is None}")
    for role, limit in (("control_decline", 4), ("control_nonnull", 2)):
        rows = [i for i, r in N.items() if r["role"] == role]
        bm = [(i, nb.get(i, {}).get("resolved"), nm.get(i, {}).get("resolved"), nt.get(i, {}).get("resolved")) for i in rows]
        mod_moved = [(i, b, m) for i, b, m, t in bm if b != m]
        terra_moved = [(i, b, t) for i, b, m, t in bm if b != t]
        print(f"\n-- {role} controls (n={len(rows)}, limit {limit}) --")
        print(f"   modified prompt changed {len(mod_moved)} | terra changed {len(terra_moved)}")
        for i, b, m in mod_moved[:14]:
            print(f"     MOD  {str(b)[:28]:28s} -> {str(m)[:28]:28s} | {N[i]['obs'][:66]}")
        for i, b, t in terra_moved[:14]:
            print(f"     TERRA {str(b)[:28]:28s} -> {str(t)[:28]:28s} | {N[i]['obs'][:66]}")
    print("\n-- base-vs-stored fidelity (did the harness reproduce the stored run?) --")
    nn = [i for i, r in N.items() if r["role"] == "control_nonnull"]
    agree = sum(1 for i in nn if nb.get(i, {}).get("resolved") == N[i]["stored_sel"])
    dec = [i for i, r in N.items() if r["role"] in ("control_decline", "control_correct_null", "control_correct_null_item_absent")]
    dagree = sum(1 for i in dec if nb.get(i, {}).get("resolved") is None)
    print(f"   non-null rows reproducing the stored selection: {agree}/{len(nn)}")
    print(f"   decline rows still declining under the base prompt: {dagree}/{len(dec)}")
    print(f"   terra on non-null rows agreeing with the stored selection: {sum(1 for i in nn if nt.get(i,{}).get('resolved')==N[i]['stored_sel'])}/{len(nn)}")
    print(f"   terra on decline rows still declining: {sum(1 for i in dec if nt.get(i,{}).get('resolved') is None)}/{len(dec)}")

# token accounting
meta = json.load(open(f"{SP}/sn_run_meta.json", encoding="utf-8")) if os.path.exists(f"{SP}/sn_run_meta.json") else {}
print("\n" + "=" * 104)
print("USAGE:", json.dumps(meta.get("usage"), default=str)[:1500])
calls = Counter(k[0] for k in res for _ in res[k])
print("calls by arm:", {k[0]: len(v) for k, v in res.items()})
