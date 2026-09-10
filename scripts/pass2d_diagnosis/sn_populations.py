"""Build the fixed populations for arms S and N. Deterministic, offline.
Candidate lists and observation text are taken verbatim from the stored artifacts and never re-retrieved."""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.getcwd())
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
corpus = json.load(open("reports/catalog_audit_replay_corpus.json", encoding="utf-8"))
ruleE = json.load(open(f"{SP}/b2_rule_e_flips.json", encoding="utf-8"))
triage = json.load(open(f"{SP}/pass2d_diagnosis_20260909.v3.json", encoding="utf-8"))["null_triage"]

idx = {}
for a in corpus["artifacts"]:
    p = a.get("artifact_path")
    if not a.get("readable") or not p or not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    for pk, ph in (d.get("photos") or {}).items():
        for r in (ph.get("debug") or {}).get("resolved_items") or []:
            idx[r["issue_id"]] = dict(prop=a["property_key"], run=a["run_id"], photo=r.get("source_photo_key") or pk,
                                      scene=(ph.get("scene") or {}).get("id"), group=(ph.get("scene") or {}).get("group"),
                                      issue=r["issue_id"], obs=r["description"], kind=r.get("original_kind"),
                                      stored_path=r.get("resolution_path"), stored_shortcut=r.get("shortcut_reason"),
                                      stored_sel=r.get("resolved_item_id"), candidates=r.get("candidates") or [])

def row(issue, role, note=""):
    r = dict(idx[issue]); r["role"] = role; r["note"] = note; return r

ANCHORS = {
    "C6": ("979a8dd3f7eab927", "anchor_missed_match", "Steven: candidate is the best option, clearly outdated but functional -> expect appliances_dated_or_basic"),
    "C7": ("280f2d14bdd2b082", "anchor_missed_match", "Steven: candidate is good, clearly outdated; unsure about costing -> expect outdated_bathroom_finishes (observation valid; spend is a separate axis)"),
    "C8": ("0bb902dec263ed01", "anchor_missed_match", "Steven: floor/appliances/cabinets all outdated; general decor is acceptable HERE -> any of decor/appliances/cabinets/flooring counts as representing the observation"),
}
# D2 and D3 by property+photo+text
def find(prop, photo, frag):
    for i, r in idx.items():
        if r["prop"] == prop and r["photo"] == photo and frag.lower() in r["obs"].lower():
            return i
    raise SystemExit(f"not found: {prop} {photo} {frag}")
D2 = find("redfin_126418713", "photo_025.jpg", "swirl plaster")
D3 = find("redfin_11185681", "photo_027.jpg", "Painted cabinets appear older")

pop = {}
# ---- Arm S: the 25 Rule E rows ----
s_rows = []
for e in ruleE:
    r = row(e["issue"], "ruleE_flip", f"fired {e['firing']} outside head [{e['head']}]")
    r["human_confirmed_wrong_subject"] = e["issue"] in ("c2785d1cb9cf1e59",) or (r["prop"] == "redfin_10952874" and r["photo"] == "photo_031.jpg")
    r["stored_terra_verdict"] = e.get("verdict")
    s_rows.append(r)
pop["S"] = s_rows

# ---- Arm N ----
n_rows = [row(iid, role, note) for _, (iid, role, note) in ANCHORS.items()]
n_rows.append(row(D2, "control_correct_null", "Steven: swirl plaster, NOT popcorn; would not cost it. MUST STAY NULL."))
n_rows.append(row(D3, "control_correct_null_item_absent", "Steven: laundry room, cabinets outdated, but cabinets_dated_style is kitchen-only eligible so the right item is ABSENT. MUST STAY NULL."))
seen = {r["issue"] for r in n_rows}
# 40 correct-decline controls: triage no_fit rows, deterministic
nofit = []
for k in ("degradation_defect", "modernization_sample"):
    for t in triage[k if k == "degradation_defect" else "modernization_sample"]["rows"]:
        if t["bucket"] == "no_fit" and t["issue_id"] in idx and t["issue_id"] not in seen:
            nofit.append(t["issue_id"])
nofit = sorted(set(nofit), key=lambda i: (idx[i]["prop"], idx[i]["photo"], i))
step = max(1, len(nofit) // 40)
pick = nofit[::step][:40]
for i in pick:
    n_rows.append(row(i, "control_decline", "triage judged this decline correct; must not start selecting"))
    seen.add(i)
# 40 non-null controls: llm-path resolved rows, deterministic, spread across kinds
resolved = sorted([i for i, r in idx.items() if r["stored_path"] == "llm" and r["stored_sel"] and i not in seen],
                  key=lambda i: (idx[i]["prop"], idx[i]["photo"], i))
step2 = max(1, len(resolved) // 40)
for i in resolved[::step2][:40]:
    n_rows.append(row(i, "control_nonnull", f"stored selection {idx[i]['stored_sel']}; must not change"))
pop["N"] = n_rows

for k, v in pop.items():
    print(f"{k}: {len(v)} rows | roles {dict(Counter(r['role'] for r in v))} | kinds {dict(Counter(r['kind'] for r in v))}")
print("D2 issue:", D2, "|", idx[D2]["obs"])
print("D3 issue:", D3, "|", idx[D3]["obs"])
print("S rows with human-confirmed wrong subject:", [r["issue"] for r in pop["S"] if r["human_confirmed_wrong_subject"]])
json.dump(pop, open(f"{SP}/sn_populations.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("total model calls planned: S", len(pop["S"]), "x2 (qwen,terra) + N", len(pop["N"]), "x3 (qwen-base,qwen-mod,terra) =",
      len(pop["S"]) * 2 + len(pop["N"]) * 3)
