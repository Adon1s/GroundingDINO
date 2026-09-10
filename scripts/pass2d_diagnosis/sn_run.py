"""Arms S and N plus the Terra resolver comparison.
Mirrors tools/scene_classifier_passes.run_pass_2d exactly except: the lexical shortcut is bypassed
(that is what arm S tests) and the user template is swappable (that is what arm N tests).
Observation text and candidate lists are the stored ones, held fixed. Retrieval is never re-run.
Results append to sn_results.jsonl and the script is resumable."""
import asyncio, json, os, sys, time
sys.path.insert(0, os.getcwd())
from tools.scene_classifier_passes import (PASS_2D_USER_PROMPT_TEMPLATE, PASS_2D_SYSTEM_PROMPT, PASS_2D_PROMPT_VERSION,
                                           format_candidates_text, safe_format_prompt, _with_analysis_pass,
                                           _candidate_item_id, _resolved_kind_for_candidate)
from tools.llm_json import extract_json_object
from tools.vlm_client import create_vlm_client
from tools import pipeline_config as cfg

SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
OUT = f"{SP}/sn_results.jsonl"
pop = json.load(open(f"{SP}/sn_populations.json", encoding="utf-8"))

# The arm-N prompt change: one added rule line. Everything else byte-identical.
MODIFIED_TEMPLATE = PASS_2D_USER_PROMPT_TEMPLATE.replace(
    "- If none fit, return null.",
    "- If the observation names a specific instance of a candidate's subject, select that candidate\n"
    "  (a faucet is a bathroom fixture; a range is an appliance; a cabinet door is a cabinet).\n"
    "- If none fit, return null.",
)
assert MODIFIED_TEMPLATE != PASS_2D_USER_PROMPT_TEMPLATE

QWEN = {"provider": "lmstudio", "model": cfg.LM_STUDIO_MODEL, "url": "http://127.0.0.1:1234"}
TERRA = {"provider": "openai", "model": "gpt-5.6-terra", "reasoning_effort": "low", "verbosity": "low",
         "api_key": os.environ.get("OPENAI_API_KEY") or cfg.OPENAI_API_KEY}

async def resolve(vlm, model_config, obs, candidates, kind, template):
    """Byte-for-byte the run_pass_2d tail, with the shortcut skipped and the template injected."""
    candidates_text = format_candidates_text(candidates)
    user_prompt = safe_format_prompt(template, observation=obs, candidates_text=candidates_text, kind=kind)
    t0 = time.time()
    response = await vlm.analyze_text(system_prompt=PASS_2D_SYSTEM_PROMPT, user_prompt=user_prompt,
                                      **_with_analysis_pass(model_config, "Pass 2d (catalog resolution)"))
    dt = round(time.time() - t0, 2)
    result = extract_json_object(response) or {}
    rid = None
    if isinstance(result, dict):
        rid = result.get("resolved_item_id") or result.get("resolved_defect_id") or result.get("resolved_upgrade_id")
        rid = str(rid).strip() if rid else None
    hallucinated = False
    if rid and rid not in {_candidate_item_id(c) for c in candidates}:
        hallucinated = True; rid = None
    return dict(resolved=rid, raw=response, seconds=dt, hallucinated=hallucinated,
                prompt_chars=len(user_prompt), candidate_count=len(candidates))

def done_keys():
    if not os.path.exists(OUT):
        return set()
    ks = set()
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line); ks.add((r["arm"], r["issue"], r["replica"]))
        except Exception:
            pass
    return ks

async def run_arm(vlm, arm, rows, model_config, template, replica=1, concurrency=1):
    have = done_keys(); todo = [r for r in rows if (arm, r["issue"], replica) not in have]
    print(f"[{arm} r{replica}] {len(todo)} to run of {len(rows)}", flush=True)
    sem = asyncio.Semaphore(concurrency)
    fh = open(OUT, "a", encoding="utf-8")
    n_ok = n_err = 0
    async def one(r):
        nonlocal n_ok, n_err
        async with sem:
            rec = dict(arm=arm, replica=replica, issue=r["issue"], prop=r["prop"], photo=r["photo"], role=r["role"],
                       obs=r["obs"], kind=r["kind"], stored_path=r["stored_path"], stored_sel=r["stored_sel"],
                       candidate_ids=[_candidate_item_id(c) for c in r["candidates"]], note=r.get("note", ""))
            for attempt in (1, 2, 3):
                try:
                    out = await resolve(vlm, model_config, r["obs"], r["candidates"], r["kind"], template)
                    rec.update(out); rec["error"] = None; n_ok += 1
                    break
                except Exception as e:
                    if attempt == 3:
                        rec.update(resolved=None, raw=None, error=f"{type(e).__name__}: {e}"); n_err += 1
                    else:
                        await asyncio.sleep(5)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
    await asyncio.gather(*[one(r) for r in todo])
    fh.close()
    print(f"[{arm} r{replica}] ok={n_ok} err={n_err}", flush=True)

async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    vlm = create_vlm_client(timeout=360)
    plan = []
    if which in ("all", "qwen", "S"):
        plan.append(("S_qwen_base", pop["S"], QWEN, PASS_2D_USER_PROMPT_TEMPLATE, 1, 1))
    if which in ("all", "qwen", "N"):
        plan.append(("N_qwen_base", pop["N"], QWEN, PASS_2D_USER_PROMPT_TEMPLATE, 1, 1))
        plan.append(("N_qwen_mod", pop["N"], QWEN, MODIFIED_TEMPLATE, 1, 1))
    if which in ("all", "terra"):
        plan.append(("S_terra_base", pop["S"], TERRA, PASS_2D_USER_PROMPT_TEMPLATE, 1, 3))
        plan.append(("N_terra_base", pop["N"], TERRA, PASS_2D_USER_PROMPT_TEMPLATE, 1, 3))
    for arm, rows, mc, tpl, rep, conc in plan:
        await run_arm(vlm, arm, rows, mc, tpl, replica=rep, concurrency=conc)
    stats = getattr(vlm, "usage_stats", None) or getattr(vlm, "_usage", None)
    print("USAGE:", json.dumps(stats, default=str)[:2000] if stats else "no usage attribute on client")
    json.dump(dict(prompt_version=PASS_2D_PROMPT_VERSION, modified_template=MODIFIED_TEMPLATE,
                   qwen=QWEN, terra={k: v for k, v in TERRA.items() if k != "api_key"}, usage=stats),
              open(f"{SP}/sn_run_meta.json", "w", encoding="utf-8"), indent=1, default=str)

asyncio.run(main())
