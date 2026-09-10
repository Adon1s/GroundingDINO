"""B2: offline enumeration of shortcut-rule variants over the 640 stored shortcut rows, plus the
D3 scene-eligibility count. Stored candidates + shipped matcher only; no model or sidecar."""
import json, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.getcwd())
from tools.pipeline_common import term_matches
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
corpus = json.load(open("reports/catalog_audit_replay_corpus.json", encoding="utf-8"))
cat = {i["id"]: i for i in json.load(open("tools/issue_catalog_kind_v2.json", encoding="utf-8"))["items"]}
COND = re.compile(r"worn|stain|scuff|crack|peel|fad|dated|old|outdated|damag|broken|torn|sag|loose|missing|dirty|rust|rot|discolor|chip|warp|dull|weather|leak|mold|mildew|grime|grimy|soil|bubbl|flak|gray|grey|chalk|deteriorat|patch|hole|dent|mark|ding|uneven|basic|plain|aged|aging|tired|fail|detached|clog|drip|frayed|mismatch|clutter|overgrown|dead|bare|exposed|unfinished|water|spall|erod|settle|lean|bow|corro|debris|filth|smudg|scratch|lift|delamin|cupping|swell|swollen|efflor|seep|moist|drain|neglect|weeds|thin lawn")
VERBS = {"has", "have", "is", "are", "appears", "appear", "shows", "show", "looks", "look", "seems", "seem", "with", "feature", "features", "remains", "remain", "needs", "need"}

rows, verdict_of = [], {}
for a in corpus["artifacts"]:
    p = a.get("artifact_path")
    if not a.get("readable") or not p or not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    v5 = ((d.get("analysis_debug") or {}).get("renovation_estimate_v5") or {}).get("result") or {}
    revs = {}
    for r in v5.get("condition_reviews") or []:
        revs.setdefault(r["condition_id"], r["verdict"])
    for c in v5.get("observed_conditions") or []:
        for iid in c.get("issue_ids") or []:
            verdict_of[iid] = revs.get(c["condition_id"])
    for pk, ph in (d.get("photos") or {}).items():
        sg = (ph.get("scene") or {}).get("group"); sid = (ph.get("scene") or {}).get("id")
        for r in (ph.get("debug") or {}).get("resolved_items") or []:
            rows.append(dict(prop=a["property_key"], photo=r.get("source_photo_key") or pk, scene=sid, group=sg, issue=r["issue_id"],
                             obs=r["description"], kind=r.get("original_kind"), path=r.get("resolution_path"), sel=r.get("resolved_item_id"),
                             cands=r.get("candidates") or [], verdict=verdict_of.get(r["issue_id"])))

def subject_prefix(obs):
    toks = re.findall(r"[a-z\-']+|[\$]", obs.lower())
    out = []
    for t in toks:
        if t in VERBS: break
        out.append(t)
    return " ".join(out) if out else obs.lower()

sc = [r for r in rows if r["path"] == "lexical_shortcut" and r["cands"]]
analysis = []
for r in sc:
    top = r["cands"][0]; obs = re.sub(r"\s+", " ", r["obs"]).strip().lower()
    terms = top.get("support_any") or []
    firing = [t for t in terms if term_matches(re.sub(r"\s+", " ", t).strip().lower(), obs)]
    pre = subject_prefix(r["obs"])
    in_subject = [t for t in firing if term_matches(t.strip().lower(), pre)]
    cond = [t for t in firing if COND.search(t)]
    r.update(firing=firing, in_subject=in_subject, cond=cond, bare_only=bool(firing) and not cond)
    analysis.append(r)

def stats(sub, label):
    n = len(sub); u = sum(1 for r in sub if r["verdict"] == "unsupported"); v = sum(1 for r in sub if r["verdict"])
    return dict(label=label, rows=n, with_verdict=v, unsupported=u, unsupported_rate=(u / v) if v else None,
                by_kind=dict(Counter(r["kind"] for r in sub)), top_items=Counter(r["sel"] for r in sub).most_common(8))
RULES = {
    "A_subject_aware": lambda r: not r["in_subject"],          # flips rows where no firing term sits in the sentence subject
    "B_condition_term": lambda r: not r["cond"],               # flips rows with no condition-bearing firing term
    "C_subject_or_condition": lambda r: not r["in_subject"] and not r["cond"],  # flips only rows failing both
}
out = dict(shortcut_rows=len(sc), baseline=stats(sc, "all shortcut rows"))
for name, flips in RULES.items():
    f = [r for r in sc if flips(r)]; k = [r for r in sc if not flips(r)]
    out[name] = dict(flipped=stats(f, "flipped to LLM"), kept=stats(k, "kept as shortcut"),
                     flipped_examples=[dict(prop=r["prop"], photo=r["photo"], obs=r["obs"], sel=r["sel"], firing=r["firing"], verdict=r["verdict"]) for r in f[:400]],
                     c5_flips=any(r["issue"] == "c2785d1cb9cf1e59" for r in f), d1_flips=any(r["prop"] == "redfin_10952874" and r["photo"] == "photo_031.jpg" for r in f))
# bath fixtures item specifically
bf = [r for r in sc if r["sel"] == "bath_fixtures_stained_or_worn"]
out["bath_fixtures_shortcuts"] = [dict(obs=r["obs"], firing=r["firing"], in_subject=r["in_subject"], verdict=r["verdict"]) for r in bf]
# D3: cabinets mentioned outside kitchen scenes
cab = [r for r in rows if "cabinet" in (r["obs"] or "").lower() and r["group"] != "kitchen"]
out["cabinets_outside_kitchen"] = dict(rows=len(cab), by_group=dict(Counter(r["group"] for r in cab)), by_scene=dict(Counter(r["scene"] for r in cab)),
                                       nulls=sum(1 for r in cab if r["sel"] is None), by_kind=dict(Counter(r["kind"] for r in cab)),
                                       resolved_to=Counter(r["sel"] for r in cab if r["sel"]).most_common(6),
                                       cabinets_dated_style_scene_groups=cat["cabinets_dated_style"]["scene_groups"],
                                       cabinets_worn_finish_scene_groups=cat.get("cabinets_worn_finish", {}).get("scene_groups"),
                                       examples=[dict(prop=r["prop"], photo=r["photo"], scene=r["scene"], obs=r["obs"], sel=r["sel"]) for r in cab if r["sel"] is None][:12])
json.dump(out, open(f"{SP}/b2_shortcut_analysis.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
def show(s): return f"{s['rows']} rows; Terra unsupported {s['unsupported']}/{s['with_verdict']}" + (f" = {s['unsupported_rate']:.1%}" if s['unsupported_rate'] is not None else "")
print("SHORTCUT ROWS:", len(sc), "|", show(out["baseline"]))
for name in RULES:
    o = out[name]
    print(f"\nRULE {name}: flipped {show(o['flipped'])} | kept {show(o['kept'])} | flips C5={o['c5_flips']} D1={o['d1_flips']}")
    print("   flipped by kind:", o["flipped"]["by_kind"], "| top flipped items:", o["flipped"]["top_items"][:6])
print("\nBATH FIXTURES SHORTCUTS:", len(bf))
for x in out["bath_fixtures_shortcuts"]: print("  ", x)
c = out["cabinets_outside_kitchen"]
print("\nCABINETS OUTSIDE KITCHEN:", {k: c[k] for k in ("rows", "by_group", "nulls", "by_kind", "resolved_to", "cabinets_dated_style_scene_groups", "cabinets_worn_finish_scene_groups")})
for e in c["examples"][:8]: print("   null:", e)
