"""Rule D: the shortcut may fire only if a firing support term matches the HEAD of the sentence subject,
i.e. the subject prefix cut at the first verb OR the first preposition. Offline over the 640 stored shortcut rows."""
import json, os, re, sys
from collections import Counter
sys.path.insert(0, os.getcwd())
from tools.pipeline_common import term_matches
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
corpus = json.load(open("reports/catalog_audit_replay_corpus.json", encoding="utf-8"))
VERBS = {"has", "have", "is", "are", "appears", "appear", "shows", "show", "looks", "look", "seems", "seem", "with", "feature", "features", "remains", "remain", "needs", "need"}
PREPS = {"of", "above", "below", "near", "around", "behind", "at", "on", "in", "along", "beside", "under", "between", "by", "next", "over", "from", "beyond", "to", "inside", "outside"}
rows = []
for a in corpus["artifacts"]:
    p = a.get("artifact_path")
    if not a.get("readable") or not p or not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    v5 = ((d.get("analysis_debug") or {}).get("renovation_estimate_v5") or {}).get("result") or {}
    revs = {r["condition_id"]: r["verdict"] for r in reversed(v5.get("condition_reviews") or [])}
    vo = {i: revs.get(c["condition_id"]) for c in v5.get("observed_conditions") or [] for i in c.get("issue_ids") or []}
    for pk, ph in (d.get("photos") or {}).items():
        for r in (ph.get("debug") or {}).get("resolved_items") or []:
            if r.get("resolution_path") == "lexical_shortcut" and r.get("candidates"):
                rows.append(dict(prop=a["property_key"], photo=r.get("source_photo_key") or pk, issue=r["issue_id"], obs=r["description"], kind=r.get("original_kind"),
                                 sel=r["resolved_item_id"], top=r["candidates"][0], verdict=vo.get(r["issue_id"])))
def head(obs):
    toks = re.findall(r"[a-z\-']+", obs.lower()); out = []
    for t in toks:
        if t in VERBS or t in PREPS: break
        out.append(t)
    return " ".join(out) if out else obs.lower()
flip, keep = [], []
for r in rows:
    obs = re.sub(r"\s+", " ", r["obs"]).strip().lower(); h = head(r["obs"])
    firing = [t for t in r["top"].get("support_any") or [] if term_matches(t.strip().lower(), obs)]
    r["firing"] = firing; r["head"] = h; r["in_head"] = [t for t in firing if term_matches(t.strip().lower(), h)]
    (keep if r["in_head"] else flip).append(r)
def st(s):
    v = [r for r in s if r["verdict"]]; u = sum(1 for r in v if r["verdict"] == "unsupported")
    return f"{len(s)} rows; unsupported {u}/{len(v)}" + (f" = {u/len(v):.1%}" if v else "")
print("RULE D head-noun: flipped", st(flip), "| kept", st(keep))
print("flipped by kind:", dict(Counter(r["kind"] for r in flip)), "| top flipped items:", Counter(r["sel"] for r in flip).most_common(8))
print("C5 flips:", any(r["issue"] == "c2785d1cb9cf1e59" for r in flip), "| D1 flips:", any(r["prop"] == "redfin_10952874" and r["photo"] == "photo_031.jpg" for r in flip),
      "| row6 flips:", any(r["issue"] == "9b611208c1823c13" for r in flip))
print("\nALL FLIPPED ROWS (head | firing | item | verdict | sentence):")
for r in flip:
    print(f"  [{r['head'][:28]:28s}] {r['firing']} -> {r['sel'][:34]:34s} {str(r['verdict'])[:11]:11s} | {r['obs'][:90]}")
json.dump(dict(flipped=[{k: r[k] for k in ("prop", "photo", "issue", "obs", "kind", "sel", "firing", "head", "verdict")} for r in flip], kept_count=len(keep)),
          open(f"{SP}/b2_rule_d.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
