"""Analyse the candidate validation: base vs candidate prompt, both at Qwen temperature 0.1."""
import json, os
from collections import Counter, defaultdict
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
by = defaultdict(lambda: defaultdict(list))
meta = {}
for l in open(f"{SP}/v_results.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if not r.get("error"):
        by[r["arm"]][r["issue"]].append(r["resolved"])
        meta[r["issue"]] = r
def m(v):
    if not v: return (None, 0, 0)
    c = Counter(v).most_common(1)[0]; return (c[0], c[1], len(v))
def f(v):
    a, k, n = m(v); return f"{str(a)[:30]:30s} {k}/{n}"
ANCH = {"979a8dd3f7eab927": "C6 range (IN prompt example)", "280f2d14bdd2b082": "C7 faucet (IN prompt example)",
        "0bb902dec263ed01": "C8 kitchen generic"}
NULLC = {"6dce1550f7c7a08e": "D2 swirl plaster", "32fe017ce31ed9b4": "D3 laundry cabinets"}
PAVED = None
for i, r in meta.items():
    if "pavement is stained and aged" in (r.get("obs") or ""):
        PAVED = i

print("=" * 100); print("REVIEWED RECOVERIES AND CORRECT NULLS (Qwen temperature 0.1, 3 replicas)")
for i, lab in {**ANCH, **NULLC}.items():
    print(f"  {lab:30s} base {f(by['base_t01'].get(i, []))} | cand {f(by['cand_t01'].get(i, []))}")
if PAVED:
    print(f"  {'paved-surface known regression':30s} base {f(by['base_t01'].get(PAVED, []))} | cand {f(by['cand_t01'].get(PAVED, []))}")

print("\n" + "=" * 100); print("GENERALIZATION — instance relations OUTSIDE the prompt's own examples")
gen = [i for i, r in meta.items() if r["role"].startswith("gen_")]
fired = 0
for i in sorted(gen, key=lambda x: meta[x]["role"]):
    b, c = m(by["base_t01"].get(i, []))[0], m(by["cand_t01"].get(i, []))[0]
    moved = (b is None and c is not None)
    fired += moved
    print(f"  {meta[i]['role']:22s} base {f(by['base_t01'].get(i, [])):38s} cand {f(by['cand_t01'].get(i, [])):38s} {'FIRED' if moved else ''}")
    print(f"     {meta[i]['obs'][:92]}")
print(f"  generalization rows where the rule fired: {fired} of {len(gen)}")

print("\n" + "=" * 100); print("CONTROLS")
for role, limit in (("control_decline", 4), ("control_nonnull", 2)):
    rows = [i for i, r in meta.items() if r["role"] == role]
    moved = []
    for i in rows:
        b, c = m(by["base_t01"].get(i, []))[0], m(by["cand_t01"].get(i, []))[0]
        if b != c: moved.append((i, b, c))
    print(f"  {role:16s} n={len(rows):3d} moved {len(moved):2d} / limit {limit}  {'WITHIN' if len(moved) <= limit else 'BREACH'}")
    for i, b, c in moved:
        print(f"     {str(b)[:28]:28s} -> {str(c)[:28]:28s} | {meta[i]['obs'][:60]}")
print("\n-- base-arm fidelity at t=0.1 vs the stored run --")
nn = [i for i, r in meta.items() if r["role"] == "control_nonnull"]
dec = [i for i, r in meta.items() if r["role"] in ("control_decline", "control_correct_null", "control_correct_null_item_absent")]
print(f"   non-null rows reproducing the stored selection: {sum(1 for i in nn if m(by['base_t01'].get(i, []))[0] == meta[i]['stored_sel'])}/{len(nn)}")
print(f"   decline rows still declining: {sum(1 for i in dec if m(by['base_t01'].get(i, []))[0] is None)}/{len(dec)}")
print("\n-- stability at t=0.1 (rows whose 3 replicas disagree) --")
for arm in ("base_t01", "cand_t01"):
    uns = [i for i in by[arm] if len(set(by[arm][i])) > 1]
    print(f"   {arm}: {len(uns)} unstable of {len(by[arm])}")
u = json.load(open(f"{SP}/v_meta.json", encoding="utf-8")) if os.path.exists(f"{SP}/v_meta.json") else {}
print("\nprompt:", u.get("prompt_version"), u.get("prompt_sha256", "")[:16], "| temp", u.get("temperature"))
mdl = (u.get("usage") or {}).get("per_pass", {}).get("2d", {}).get("models", {})
for k, v in mdl.items():
    print(f"  {k}: {v.get('metered_calls')} calls, {v.get('total_tokens'):,} tokens")
