"""Candidate validation for the N prompt change (pass_2d_exact_kind_v3).
Unchanged prompt vs candidate prompt, BOTH at Qwen temperature 0.1, 3 replicas, stored candidate
lists and observation text held fixed. Generalization rows are deliberately outside the prompt's
own examples (faucet / range / cabinet)."""
import asyncio, json, os, sys
sys.path.insert(0, os.getcwd())
from tools.scene_classifier_passes import (PASS_2D_USER_PROMPT_TEMPLATE, PASS_2D_SYSTEM_PROMPT, PASS_2D_PROMPT_VERSION,
                                           PASS_2D_PROMPT_SHA256, format_candidates_text, safe_format_prompt,
                                           _with_analysis_pass, _candidate_item_id)
from tools.llm_json import extract_json_object
from tools.vlm_client import create_vlm_client
from tools import pipeline_config as cfg
import subprocess

SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
OUT = f"{SP}/v_results.jsonl"
TEMP = 0.1
QWEN = {"provider": "lmstudio", "model": cfg.LM_STUDIO_MODEL, "url": "http://127.0.0.1:1234", "temperature": TEMP}
# the shipped v2 template, recovered from git so the comparison is exact
BASE_TEMPLATE = subprocess.run(["git", "show", "master:tools/scene_classifier_passes.py"], capture_output=True, text=True, encoding="utf-8").stdout
BASE_TEMPLATE = BASE_TEMPLATE.split('PASS_2D_USER_PROMPT_TEMPLATE = """', 1)[1].split('"""', 1)[0]
CAND_TEMPLATE = PASS_2D_USER_PROMPT_TEMPLATE
assert "specific instance" not in BASE_TEMPLATE and "specific instance" in CAND_TEMPLATE

pop = json.load(open(f"{SP}/sn_populations.json", encoding="utf-8"))
corpus = json.load(open("reports/catalog_audit_replay_corpus.json", encoding="utf-8"))
idx = {}
for a in corpus["artifacts"]:
    p = a.get("artifact_path")
    if not a.get("readable") or not p or not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    for pk, ph in (d.get("photos") or {}).items():
        for r in (ph.get("debug") or {}).get("resolved_items") or []:
            idx[r["issue_id"]] = dict(prop=a["property_key"], photo=r.get("source_photo_key") or pk, issue=r["issue_id"],
                                      obs=r["description"], kind=r.get("original_kind"), stored_path=r.get("resolution_path"),
                                      stored_sel=r.get("resolved_item_id"), candidates=r.get("candidates") or [])
# generalization rows: instance relations whose subject is NOT in the prompt examples
GEN = {
    "d1d8cd9809162b59": ("gen_tile_surround", "tile surround -> bathroom finishes"),
    "429ac7089e34e369": ("gen_backsplash", "backsplash -> kitchen finishes"),
    "ac006414d7b2cb59": ("gen_rear_door", "rear door -> entry/patio door"),
    "7f2dc04b630a242c": ("gen_ext_trim", "window and door trim -> dated exterior finishes"),
    "0e82c9c772ac47ea": ("gen_basement", "exposed joists/piping -> unfinished basement"),
    "3a4aa709a161e09a": ("gen_porch", "porch appearance -> curb appeal (weak instance relation)"),
    "0b2f8278945e08e9": ("gen_screens", "window screens -> dirty/grimy window screens"),
    "51d2489aa75e7505": ("gen_window_trim_paint", "window trim paint -> peeling/discoloured paint"),
}
rows = []
for r in pop["N"]:
    rows.append(dict(r))
for iid, (role, note) in GEN.items():
    if iid in idx:
        rows.append(dict(idx[iid], role=role, note=note))
seen = set(); uniq = []
for r in rows:
    if r["issue"] not in seen:
        seen.add(r["issue"]); uniq.append(r)
rows = uniq
print(f"validation population: {len(rows)} rows; generalization rows: {sum(1 for r in rows if r['role'].startswith('gen_'))}")

async def resolve(vlm, obs, candidates, kind, template):
    user_prompt = safe_format_prompt(template, observation=obs, candidates_text=format_candidates_text(candidates), kind=kind)
    resp = await vlm.analyze_text(system_prompt=PASS_2D_SYSTEM_PROMPT, user_prompt=user_prompt,
                                  **_with_analysis_pass(QWEN, "Pass 2d (catalog resolution)"))
    res = extract_json_object(resp) or {}
    rid = res.get("resolved_item_id") if isinstance(res, dict) else None
    rid = str(rid).strip() if rid else None
    if rid and rid not in {_candidate_item_id(c) for c in candidates}:
        rid = None
    return rid, resp

def done():
    s = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(l); s.add((r["arm"], r["issue"], r["replica"]))
            except Exception:
                pass
    return s

async def main():
    vlm = create_vlm_client(timeout=360)
    have = done(); fh = open(OUT, "a", encoding="utf-8")
    for arm, tpl in (("base_t01", BASE_TEMPLATE), ("cand_t01", CAND_TEMPLATE)):
        for rep in (1, 2, 3):
            todo = [r for r in rows if (arm, r["issue"], rep) not in have]
            print(f"[{arm} r{rep}] {len(todo)}", flush=True)
            for r in todo:
                try:
                    rid, raw = await resolve(vlm, r["obs"], r["candidates"], r["kind"], tpl)
                    rec = dict(arm=arm, replica=rep, issue=r["issue"], prop=r["prop"], photo=r["photo"], role=r["role"],
                               obs=r["obs"], kind=r["kind"], stored_sel=r["stored_sel"], resolved=rid, raw=raw, error=None)
                except Exception as e:
                    rec = dict(arm=arm, replica=rep, issue=r["issue"], role=r["role"], obs=r["obs"], resolved=None,
                               error=f"{type(e).__name__}: {e}")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
    fh.close()
    u = getattr(vlm, "usage_stats", None)
    json.dump(dict(temperature=TEMP, prompt_version=PASS_2D_PROMPT_VERSION, prompt_sha256=PASS_2D_PROMPT_SHA256,
                   base_template_source="git show master:tools/scene_classifier_passes.py", usage=u),
              open(f"{SP}/v_meta.json", "w", encoding="utf-8"), indent=1, default=str)
    print("USAGE:", json.dumps(u, default=str)[:600])

asyncio.run(main())
