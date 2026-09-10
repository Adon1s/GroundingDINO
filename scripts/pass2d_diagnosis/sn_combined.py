"""Combined check: the 25 Rule E rows (arm S) resolved with the arm-N modified prompt, 5 replicas.
Rule E only changes these rows; N's effect on every other row was measured separately."""
import asyncio, json, os, sys
sys.path.insert(0, os.getcwd())
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
exec(compile(open(f"{SP}/sn_run.py", encoding="utf-8").read().replace("asyncio.run(main())", ""), "sn_run.py", "exec"))

async def main():
    vlm = create_vlm_client(timeout=360)
    for rep in (1, 2, 3, 4, 5):
        await run_arm(vlm, "SN_qwen_mod", pop["S"], QWEN, MODIFIED_TEMPLATE, replica=rep, concurrency=1)
    json.dump(getattr(vlm, "usage_stats", None), open(f"{SP}/sn_combined_usage.json", "w", encoding="utf-8"), indent=1, default=str)
    print("USAGE:", json.dumps(getattr(vlm, "usage_stats", None), default=str)[:900])

asyncio.run(main())
