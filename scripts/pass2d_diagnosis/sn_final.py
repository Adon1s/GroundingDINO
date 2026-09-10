"""Final stability analysis across all replicas + token accounting."""
import json, os
from collections import Counter, defaultdict
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
by = defaultdict(lambda: defaultdict(list))          # arm -> issue -> [resolved per replica]
for line in open(f"{SP}/sn_results.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if not r.get("error"):
        by[r["arm"]][r["issue"]].append(r["resolved"])
pop = json.load(open(f"{SP}/sn_populations.json", encoding="utf-8"))
S = {r["issue"]: r for r in pop["S"]}
N = {r["issue"]: r for r in pop["N"]}
ANCH = {"979a8dd3f7eab927": "C6 range", "280f2d14bdd2b082": "C7 faucet", "0bb902dec263ed01": "C8 kitchen"}
NULLC = {"6dce1550f7c7a08e": "D2 swirl plaster", "32fe017ce31ed9b4": "D3 laundry cabinets"}
def mode(vals):
    c = Counter(vals); return c.most_common(1)[0] if c else (None, 0)
def fmt(vals):
    m, k = mode(vals); return f"{str(m)[:30]:30s} {k}/{len(vals)}"

print("=" * 100); print("ARM S (Rule E) — stability over replicas")
stable = unstable = 0
for iid, row in S.items():
    v = by["S_qwen_base"].get(iid, [])
    if len(v) < 2: continue
    m, k = mode(v)
    tag = "STABLE " if k == len(v) else "UNSTABLE"
    (stable, unstable) = (stable + (k == len(v)), unstable + (k != len(v)))
    hm = "**" if row.get("human_confirmed_wrong_subject") else "  "
    print(f"{hm} {tag} qwen {fmt(v)} | terra {fmt(by['S_terra_base'].get(iid, []))} | was {str(row['stored_sel'])[:26]:26s} | {row['obs'][:56]}")
print(f"  consequential rows repeated: stable {stable}, unstable {unstable}")

print("\n" + "=" * 100); print("ARM N — anchors and must-stay-null controls over replicas")
for iid, lab in {**ANCH, **NULLC}.items():
    print(f"  {lab:20s} base {fmt(by['N_qwen_base'].get(iid, []))} | mod {fmt(by['N_qwen_mod'].get(iid, []))} | terra {fmt(by['N_terra_base'].get(iid, []))}")

print("\n" + "=" * 100); print("CONTROL BREACHES over replicas (mode answer per row)")
for arm, base_arm, label in (("N_qwen_mod", "N_qwen_base", "modified prompt"), ("N_terra_base", "N_qwen_base", "terra resolver")):
    for role, limit in (("control_decline", 4), ("control_nonnull", 2)):
        rows = [i for i, r in N.items() if r["role"] == role]
        moved = []
        for i in rows:
            b = mode(by[base_arm].get(i, [None]))[0]; a = mode(by[arm].get(i, [None]))[0]
            if a != b: moved.append((i, b, a))
        flag = "WITHIN" if len(moved) <= limit else "BREACH"
        print(f"  {label:16s} {role:16s} moved {len(moved):2d} / limit {limit}  {flag}")

usage_files = [f"{SP}/sn_run_meta.json", f"{SP}/sn_replica_usage.json"]
tot = Counter()
for f in usage_files:
    if not os.path.exists(f): continue
    u = json.load(open(f, encoding="utf-8"))
    u = u.get("usage") if isinstance(u, dict) and "usage" in u else u
    if not u: continue
    for mdl, s in (u.get("per_pass", {}).get("2d", {}).get("models", {}) or {}).items():
        tot[(mdl, "calls")] += s.get("metered_calls", 0); tot[(mdl, "tokens")] += s.get("total_tokens", 0)
print("\n" + "=" * 100); print("TOKENS (resolver; no Terra verification was run)")
for (mdl, k), v in sorted(tot.items()):
    print(f"  {mdl:34s} {k:7s} {v:,}")
terra = tot[("openai/gpt-5.6-terra", "tokens")]
print(f"  TERRA TOTAL {terra:,} of 2,500,000 authorized ({terra/2_500_000:.2%}); remaining {2_500_000-terra:,}")
