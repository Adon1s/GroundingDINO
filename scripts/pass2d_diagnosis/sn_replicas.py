"""Repeat the consequential rows (any row whose answer changed, plus every anchor and must-stay-null
control) 4 more times per arm, so a winner is never declared on a single sample."""
import asyncio, json, os, sys
sys.path.insert(0, os.getcwd())
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
src = open(f"{SP}/sn_run.py", encoding="utf-8").read().replace("asyncio.run(main())", "")
exec(compile(src, "sn_run.py", "exec"))  # brings in resolve, run_arm, QWEN, TERRA, templates, pop

res = {}
for line in open(f"{SP}/sn_results.jsonl", encoding="utf-8"):
    r = json.loads(line)
    res.setdefault((r["arm"], r["replica"]), {})[r["issue"]] = r
S = {r["issue"]: r for r in pop["S"]}
N = {r["issue"]: r for r in pop["N"]}
ANCH = {"979a8dd3f7eab927", "280f2d14bdd2b082", "0bb902dec263ed01", "6dce1550f7c7a08e", "32fe017ce31ed9b4"}

def consequential(arm_key, universe, baseline_field):
    got = res.get(arm_key, {})
    out = []
    for iid, row in universe.items():
        a = got.get(iid, {}).get("resolved")
        base = row["stored_sel"] if baseline_field == "stored" else res.get(("N_qwen_base", 1), {}).get(iid, {}).get("resolved")
        if iid in ANCH or a != base:
            out.append(row)
    return out

PLAN = [
    ("S_qwen_base", consequential(("S_qwen_base", 1), S, "stored"), QWEN, PASS_2D_USER_PROMPT_TEMPLATE, 1),
    ("S_terra_base", consequential(("S_terra_base", 1), S, "stored"), TERRA, PASS_2D_USER_PROMPT_TEMPLATE, 3),
    ("N_qwen_base", [N[i] for i in N if i in ANCH], QWEN, PASS_2D_USER_PROMPT_TEMPLATE, 1),
    ("N_qwen_mod", consequential(("N_qwen_mod", 1), N, "base"), QWEN, MODIFIED_TEMPLATE, 1),
    ("N_terra_base", consequential(("N_terra_base", 1), N, "base"), TERRA, PASS_2D_USER_PROMPT_TEMPLATE, 3),
]
for arm, rows, mc, tpl, conc in PLAN:
    print(f"{arm}: {len(rows)} consequential rows x4 replicas")

async def main():
    vlm = create_vlm_client(timeout=360)
    for arm, rows, mc, tpl, conc in PLAN:
        for rep in (2, 3, 4, 5):
            await run_arm(vlm, arm, rows, mc, tpl, replica=rep, concurrency=conc)
    stats = getattr(vlm, "usage_stats", None)
    json.dump(stats, open(f"{SP}/sn_replica_usage.json", "w", encoding="utf-8"), indent=1, default=str)
    print("USAGE:", json.dumps(stats, default=str)[:1200])

asyncio.run(main())
