"""Build the Session 9 decision packets: evidence JSON, photo-review HTML, worksheet, doc tables.

Reads the frozen Session 9 canary artifacts (run_1 is the comparison baseline,
run_2 only feeds the Sol run-to-run probe) and produces:

  reports/session9_decision_packets.json      every number/list the design doc cites
  reports/session9_photo_review_20260821.html  photos inline next to each model's verdict
  reports/session9_decision_worksheet.json     blank per-item verdict fields for Steven
  docs/DESIGN_renovation_architecture_decision_packets.md  tables regenerated in place
                                               between <!-- GEN:name --> ... <!-- /GEN:name -->

Zero provider calls. Re-runnable; the doc's narrative outside the GEN markers is
never touched.

Run:
  .venv\\Scripts\\python.exe scripts\\build_session9_decision_packets.py
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_renovation_architecture_cutover import _load_latest_artifacts  # noqa: E402
from tools.renovation_architecture.contracts import (  # noqa: E402
    SOL_REVIEW_REASONING_EFFORT,
    TERRA_REVIEW_REASONING_EFFORT,
)

CANARY = ROOT / "artifacts_canary" / "renovation_session9_20260818"
FALLBACK_PHOTO_ROOT = Path(r"C:/Users/Steven/IntelliJProjects/renointel-prod/public/images/properties")

# The 5 Tier-2 "v4 $0 -> v5 priced" review ids (digest section 3).
TIER2_ZERO_IDS = [
    "rar1_32834e11fec24cbe", "rar1_de0247255b9e220e", "rar1_7a69e9ae5fbabd7f",
    "rar1_afda9927a9027fb6", "rar1_69d1a08831644ff0",
]


def v5(art):
    return art["analysis_debug"]["renovation_estimate_v5"]["result"]


def money(lo, hi):
    return f"${lo:,}/{hi:,}"


def photo_path(art, prop, key):
    p = ((art.get("photos") or {}).get(key) or {}).get("photo") or {}
    if p.get("image_path"):
        return str(p["image_path"])
    return str(FALLBACK_PHOTO_ROOT / prop / key)


def file_url(path):
    return "file:///" + str(path).replace("\\", "/")


def short(text, n=220):
    text = str(text or "")
    return text if len(text) <= n else text[: n - 1] + "…"


# ----------------------------------------------------------------------------
def build(run1, run2, reviews):
    out = {}

    # ---- facts ---------------------------------------------------------
    v4lo = sum(a["renovation_estimate_v4"]["final_rehab"]["low"] for _, a in run1.values())
    v4hi = sum(a["renovation_estimate_v4"]["final_rehab"]["high"] for _, a in run1.values())
    v5lo = sum(v5(a)["totals"]["headline"]["low"] for _, a in run1.values())
    v5hi = sum(v5(a)["totals"]["headline"]["high"] for _, a in run1.values())
    vc, dc, sc = Counter(), Counter(), Counter()
    models = {"terra": Counter(), "sol": Counter()}
    for run in (run1, run2):
        for _, a in run.values():
            r = v5(a)
            vc.update(x["verdict"] for x in r["condition_reviews"])
            dc.update(x["disposition"] for x in r["condition_dispositions"])
            sc.update(x["decision"] for x in r["package_decisions"])
            models["terra"].update(f"{x.get('provider')}/{x.get('model')}/{x.get('prompt_version')}" for x in r["condition_reviews"])
            models["sol"].update(f"{x.get('provider')}/{x.get('model')}/{x.get('prompt_version')}" for x in r["package_decisions"])
    routing = {}
    for _, a in run1.values():
        for row in a.get("model_routing") or []:
            routing[row.get("pass")] = {k: row.get(k) for k in ("model", "reasoning_effort", "max_output_tokens")}
    out["facts"] = {
        "corpus_v4": [v4lo, v4hi], "corpus_v5": [v5lo, v5hi],
        "corpus_delta_pct": [round((v5lo - v4lo) / v4lo, 4), round((v5hi - v4hi) / v4hi, 4)],
        "terra_verdicts_both_runs": dict(vc), "dispositions_both_runs": dict(dc), "sol_decisions_both_runs": dict(sc),
        "models": {k: dict(v) for k, v in models.items()},
        "upstream_routing": routing,
        "terra_review_reasoning_effort": TERRA_REVIEW_REASONING_EFFORT,
        "sol_review_reasoning_effort": SOL_REVIEW_REASONING_EFFORT,
        "properties": sorted(run1),
    }

    # ---- churn + Terra condition-level consistency proxy (run_1 vs run_2) ----
    churn_absent, churn_symdiff = [], []
    terra_same = terra_flip = 0
    terra_flip_kinds = Counter()
    for prop in sorted(run1):
        if prop not in run2:
            continue
        r1, r2 = v5(run1[prop][1]), v5(run2[prop][1])
        c1 = {(c["catalog_item_id"], c["estimate_unit_id"]) for c in r1["observed_conditions"]}
        c2 = {(c["catalog_item_id"], c["estimate_unit_id"]) for c in r2["observed_conditions"]}
        if c1:
            churn_absent.append(len(c1 - c2) / len(c1))
        if c1 | c2:
            churn_symdiff.append(len(c1 ^ c2) / len(c1 | c2))

        def verdict_by_key(res):
            evs = {e["condition_id"]: e for e in res["evidence_facts"]}
            revs = {r["condition_id"]: r for r in res["condition_reviews"]}
            cnt, val = Counter(), {}
            for c in res["observed_conditions"]:
                k = (c["catalog_item_id"], c["estimate_unit_id"],
                     tuple(sorted(evs.get(c["condition_id"], {}).get("photo_keys") or [])))
                cnt[k] += 1
                val[k] = revs.get(c["condition_id"], {}).get("verdict")
            return {k: val[k] for k, n in cnt.items() if n == 1}
        k1, k2 = verdict_by_key(r1), verdict_by_key(r2)
        for k in set(k1) & set(k2):
            if k1[k] == k2[k]:
                terra_same += 1
            else:
                terra_flip += 1
                terra_flip_kinds[f"{k1[k]}->{k2[k]}"] += 1
    sol_tok = [int((v5(a)["sol_listing_usage"] or {}).get("total_tokens") or 0) for run in (run1, run2) for _, a in run.values()]
    terra_tok = [int((v5(a)["terra_listing_usage"] or {}).get("total_tokens") or 0) for run in (run1, run2) for _, a in run.values()]
    out["facts"]["churn"] = {
        "mean_share_of_run1_conditions_absent_from_run2": round(sum(churn_absent) / len(churn_absent), 3) if churn_absent else None,
        "mean_symdiff_over_union": round(sum(churn_symdiff) / len(churn_symdiff), 3) if churn_symdiff else None,
        "key": "(catalog_item_id, estimate_unit_id)",
    }
    out["facts"]["terra_condition_proxy"] = {
        "identical_item_unit_photoset_conditions_in_both_replicas": terra_same + terra_flip,
        "same_verdict": terra_same, "flipped": terra_flip, "flip_kinds": dict(terra_flip_kinds),
        "note": "batch context (other conditions in the same estimate-unit call) differed, so this is a proxy, not a strict identical-input probe",
    }
    out["facts"]["tokens_per_listing_both_replicas"] = {
        "sol_mean": round(sum(sol_tok) / len(sol_tok)) if sol_tok else None,
        "terra_mean": round(sum(terra_tok) / len(terra_tok)) if terra_tok else None,
        "n_artifacts": len(sol_tok),
    }

    # ---- P2: condition-level 2f vs Terra ------------------------------
    # v4's 2f verdicts are per upstream issue; several issues can merge into one v5
    # condition. Rows are keyed by v5 condition (one photo set / one Terra verdict)
    # and carry every v4 issue that maps to it.
    contra = []
    agree = Counter()
    by_cond = {}
    for prop, (_, art) in sorted(run1.items()):
        res = v5(art)
        v4 = art["renovation_estimate_v4"]
        issue_meta = {}
        for lane in ("issues_flat", "estimate_issues_flat", "product_issues_flat"):
            for it in art.get(lane) or []:
                if isinstance(it, dict) and it.get("issue_id"):
                    issue_meta.setdefault(it["issue_id"], it)
        cond_by_issue = {}
        for c in res["observed_conditions"]:
            for iid in c.get("issue_ids") or []:
                cond_by_issue[iid] = c
        revs = {r["condition_id"]: r for r in res["condition_reviews"]}
        disps = {d["condition_id"]: d for d in res["condition_dispositions"]}
        evs = {e["condition_id"]: e for e in res["evidence_facts"]}
        cond_work = {}
        for w in res["work_items"]:
            if w.get("status") == "active":
                for cid in w.get("condition_ids") or []:
                    cond_work[cid] = w
        seen = set()
        for pk in (v4.get("package_candidates") or []) + (v4.get("packages") or []):
            raw = pk.get("raw_pass_2f_response")
            try:
                j = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                j = {}
            for verdict, ids in (("confirmed", pk.get("confirmed_issue_ids") or []),
                                 ("rejected", pk.get("rejected_issue_ids") or [])):
                for iid in ids:
                    if (iid, pk.get("package_id")) in seen:
                        continue
                    seen.add((iid, pk.get("package_id")))
                    c = cond_by_issue.get(iid)
                    if not c:
                        agree[("2f_" + verdict, "no_v5_condition")] += 1
                        continue
                    r = revs.get(c["condition_id"], {})
                    t = r.get("verdict")
                    agree[("2f_" + verdict, "terra_" + str(t))] += 1
                    if (verdict == "rejected" and t == "supported") or (
                            verdict == "confirmed" and t in ("unsupported", "cannot_assess")):
                        ck = (prop, c["condition_id"])
                        if ck not in by_cond:
                            ev = evs.get(c["condition_id"], {})
                            w = cond_work.get(c["condition_id"])
                            keys = list(ev.get("photo_keys") or [])
                            by_cond[ck] = {
                                "property": prop, "catalog_item_id": c["catalog_item_id"],
                                "estimate_unit_id": c["estimate_unit_id"], "condition_id": c["condition_id"],
                                "v4_issues": [],
                                "direction": ("terra_supported_2f_rejected" if verdict == "rejected"
                                              else "2f_confirmed_terra_not"),
                                "v4_2f_verdicts": set(),
                                "v4_2f_package": pk.get("package_id"),
                                "v4_2f_evidence_summary": j.get("evidence_summary"),
                                "v4_2f_review_photos": pk.get("review_photo_keys"),
                                "terra_verdict": t, "terra_rationale": r.get("rationale"),
                                "terra_photo_keys": keys, "distinct_photo_count": ev.get("distinct_photo_count"),
                                "v5_disposition": (disps.get(c["condition_id"]) or {}).get("disposition"),
                                "v5_work": ({"action": w["action_code"], "low": w["low"], "high": w["high"]} if w else None),
                                "photo_paths": [photo_path(art, prop, k) for k in keys],
                            }
                        im = issue_meta.get(iid, {})
                        row = by_cond[ck]
                        row["v4_issues"].append({"issue_id": iid, "v4_2f_verdict": verdict,
                                                 "observation": im.get("observation") or im.get("description") or im.get("label")})
                        row["v4_2f_verdicts"].add(verdict)
    contra = list(by_cond.values())
    for c in contra:
        c["v4_2f_verdicts"] = sorted(c["v4_2f_verdicts"])
        c["v4_issue_count"] = len(c["v4_issues"])
        if len(c["v4_2f_verdicts"]) > 1:
            c["direction"] = "mixed_2f_verdicts"
    contra.sort(key=lambda x: (x["direction"], x["property"], x["catalog_item_id"], x["estimate_unit_id"]))
    for i, c in enumerate(contra, 1):
        c["item_id"] = f"C{i:03d}"
    out["p2_agreement_issue_level"] = {"|".join(k): n for k, n in sorted(agree.items())}
    out["p2_issue_rows"] = sum(c["v4_issue_count"] for c in contra)
    out["p2_contradictions"] = contra
    by_item = defaultdict(Counter)
    for c in contra:
        by_item[c["catalog_item_id"]][c["direction"]] += 1
    out["p2_by_catalog_item"] = {k: dict(v) for k, v in sorted(by_item.items(), key=lambda kv: -sum(kv[1].values()))}

    # single-photo share of v5 billed scope (run_1)
    one = Counter(); one_money = [0, 0]; all_money = [0, 0]
    for prop, (_, art) in run1.items():
        res = v5(art)
        evs = {e["condition_id"]: e for e in res["evidence_facts"]}
        acc = {d["condition_id"] for d in res["condition_dispositions"] if d["disposition"] == "accepted_for_work"}
        for w in res["work_items"]:
            if w.get("status") != "active":
                continue
            cids = [c for c in (w.get("condition_ids") or []) if c in acc]
            if not cids:
                continue
            n = max(int(evs.get(c, {}).get("distinct_photo_count") or 0) for c in cids)
            one["one_photo" if n <= 1 else "multi_photo"] += 1
            all_money[0] += w["low"]; all_money[1] += w["high"]
            if n <= 1:
                one_money[0] += w["low"]; one_money[1] += w["high"]
    out["single_photo_work"] = {"work_items": dict(one), "one_photo_dollars": one_money, "all_active_dollars": all_money}

    # ---- P1: package disputes (v4 2f rejected the package) --------------
    pkgs = []
    for prop, (_, art) in sorted(run1.items()):
        res = v5(art)
        v4 = art["renovation_estimate_v4"]
        decs = {d["package_candidate_id"]: d for d in res["package_decisions"]}
        apps = {a["package_candidate_id"]: a for a in res["package_applications"]}
        works = {w["work_item_id"]: w for w in res["work_items"]}
        conds = {c["condition_id"]: c for c in res["observed_conditions"]}
        revs = {r["condition_id"]: r for r in res["condition_reviews"]}
        evs = {e["condition_id"]: e for e in res["evidence_facts"]}
        for pk in v4.get("package_candidates") or []:
            if pk.get("verification_status") != "rejected":
                continue
            ptype, unit = pk.get("package_type"), pk.get("estimate_unit_id") or ""
            cands = [c for c in res["package_candidates"] if c["package_type"] == ptype and c["estimate_unit_id"] == unit]
            if not cands:
                continue
            c = cands[0]
            d = decs.get(c["package_candidate_id"], {})
            a = apps.get(c["package_candidate_id"], {})
            raw = pk.get("raw_pass_2f_response")
            try:
                j = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                j = {}
            children = []
            for wid in c["child_work_item_ids"]:
                w = works[wid]
                cl = []
                for cid in w.get("condition_ids") or []:
                    cc = conds.get(cid, {}); r = revs.get(cid, {}); ev = evs.get(cid, {})
                    cl.append({"catalog_item_id": cc.get("catalog_item_id"), "terra": r.get("verdict"),
                               "rationale": r.get("rationale"), "photo_keys": ev.get("photo_keys"),
                               "photo_paths": [photo_path(art, prop, k) for k in (ev.get("photo_keys") or [])]})
                children.append({"action": w["action_code"], "catalog_item_ids": w["catalog_item_ids"],
                                 "low": w["low"], "high": w["high"], "conditions": cl,
                                 "role": "driver" if wid in (c.get("driver_work_item_ids") or []) else "support"})
            ch_lo = sum(works[w]["low"] for w in a.get("absorbed_work_item_ids") or [])
            ch_hi = sum(works[w]["high"] for w in a.get("absorbed_work_item_ids") or [])
            pkgs.append({
                "property": prop, "package": f"{ptype}|{unit}", "v5_tier": c["pricing_tier"],
                "v5_price": [c["low"], c["high"]], "v5_sol_decision": d.get("decision"),
                "v5_sol_rationale": d.get("rationale"), "v5_status": a.get("status"),
                "v5_effective": [a.get("effective_low") or 0, a.get("effective_high") or 0],
                "v5_children_standalone": [ch_lo, ch_hi],
                "v5_net_lift": [(a.get("effective_low") or 0) - ch_lo, (a.get("effective_high") or 0) - ch_hi],
                "v4_tier": pk.get("pricing_tier"), "v4_price": [pk.get("cost_low"), pk.get("cost_high")],
                "v4_2f_status": pk.get("verification_status"), "v4_2f_evidence_summary": j.get("evidence_summary"),
                "v4_2f_confirmed": len(pk.get("confirmed_issue_ids") or []),
                "v4_2f_rejected": len(pk.get("rejected_issue_ids") or []),
                "v4_2f_review_photos": pk.get("review_photo_keys") or [],
                "v4_2f_review_photo_paths": pk.get("review_image_paths") or [],
                "children": children,
            })
    for i, p in enumerate(pkgs, 1):
        p["item_id"] = f"P{i:02d}"
    out["p1_packages"] = pkgs
    appr = [p for p in pkgs if p["v5_sol_decision"] == "approve"]
    out["p1_totals"] = {
        "count_2f_rejected": len(pkgs), "count_sol_approved": len(appr),
        "sol_approved_effective": [sum(p["v5_effective"][0] for p in appr), sum(p["v5_effective"][1] for p in appr)],
        "sol_approved_net_lift": [sum(p["v5_net_lift"][0] for p in appr), sum(p["v5_net_lift"][1] for p in appr)],
    }

    # ---- P3: multi-bathroom -------------------------------------------
    baths = []
    for prop, (_, art) in sorted(run1.items()):
        v4 = art["renovation_estimate_v4"]
        res = v5(art)
        surr = [s for s in v4.get("room_surrogates") or []
                if str(s.get("scene_group")) == "bathroom" or "bath" in str(s.get("room_surrogate_id"))]
        if len(surr) < 2:
            continue
        exp = v4.get("bathroom_expansion_audit") or {}
        sig = v4.get("bathroom_room_count_signal") or {}
        meta = art.get("property") or {}
        bathpk = [p for p in v4.get("packages") or [] if p.get("package_type") == "bathroom_modernization"]
        evs = {e["condition_id"]: e for e in res["evidence_facts"]}
        revs = {r["condition_id"]: r for r in res["condition_reviews"]}
        v5conds = []
        for c in res["observed_conditions"]:
            if "bath" in (c.get("estimate_unit_id") or ""):
                v5conds.append({"catalog_item_id": c["catalog_item_id"], "unit": c["estimate_unit_id"],
                                "source_surrogates": c.get("source_room_surrogate_ids"),
                                "identity_ambiguous": c.get("identity_ambiguous"),
                                "terra": revs.get(c["condition_id"], {}).get("verdict"),
                                "photo_keys": evs.get(c["condition_id"], {}).get("photo_keys")})
        v5bm = []
        apps = {a["package_candidate_id"]: a for a in res["package_applications"]}
        for c in res["package_candidates"]:
            if c["package_type"] != "bathroom_modernization":
                continue
            a = apps.get(c["package_candidate_id"], {})
            # effective (applied) dollars, so the v4-v5 column matches the ledger, not the candidate price
            v5bm.append({**c, "low": a.get("effective_low") or 0, "high": a.get("effective_high") or 0,
                         "status": a.get("status")})
        baths.append({
            "property": prop, "listing_baths": meta.get("baths") or meta.get("bath_count"),
            "bathroom_metadata_cap": exp.get("bathroom_metadata_cap"),
            "surrogates": [{"id": s["room_surrogate_id"], "photo_keys": s.get("photo_keys") or [],
                            "photo_paths": [photo_path(art, prop, k) for k in s.get("photo_keys") or []]} for s in surr],
            "v4_signal_likely_multiple": sig.get("likely_multiple_visible_bathrooms"),
            "v4_expanded": exp.get("expanded"), "v4_qualifying_surrogates": exp.get("qualifying_surrogate_ids"),
            "v4_fallback_reason": exp.get("fallback_reason"), "v4_per_surrogate": exp.get("per_surrogate"),
            "v4_bath_mod_packages": [{"id": p["package_id"], "surrogate": p.get("room_surrogate_id"),
                                      "cost": [p["cost_low"], p["cost_high"]],
                                      "confirmed_issue_count": len(p.get("confirmed_issue_ids") or [])} for p in bathpk],
            "v5_bath_units": sorted({c["unit"] for c in v5conds}),
            "v5_bath_mod": [[c["estimate_unit_id"], c["pricing_tier"], c["low"], c["high"]] for c in v5bm],
            "v5_conditions": v5conds,
            "v4_minus_v5_bathmod": [sum(p["cost_low"] for p in bathpk) - sum(c["low"] for c in v5bm),
                                    sum(p["cost_high"] for p in bathpk) - sum(c["high"] for c in v5bm)],
        })
    for i, b in enumerate(baths, 1):
        b["item_id"] = f"B{i:02d}"
    out["p3_multi_bath"] = baths

    # ---- P4: Sol consistency ------------------------------------------
    def cands(res):
        works = {w["work_item_id"]: w for w in res["work_items"]}
        decs = {d["package_candidate_id"]: d for d in res["package_decisions"]}
        m = {}
        for c in res["package_candidates"]:
            if c.get("display_only"):
                continue
            kids = tuple(sorted(tuple(sorted(works[w]["catalog_item_ids"])) for w in c["child_work_item_ids"]))
            d = decs[c["package_candidate_id"]]
            m[(c["package_type"], c["estimate_unit_id"])] = (kids, c["pricing_tier"], d["decision"], d["rationale"])
        return m
    flips, same, diff = [], 0, 0
    for prop in sorted(run1):
        if prop not in run2:
            continue
        m1, m2 = cands(v5(run1[prop][1])), cands(v5(run2[prop][1]))
        for k in set(m1) & set(m2):
            k1, t1, d1, ra1 = m1[k]; k2, t2, d2, ra2 = m2[k]
            if k1 != k2:
                diff += 1
                continue
            if d1 == d2:
                same += 1
            else:
                flips.append({"property": prop, "package": "|".join(k), "children": [list(x) for x in k1],
                              "tier": t1, "sol_run1": d1, "sol_run2": d2, "rationale_run1": ra1, "rationale_run2": ra2})
    singles, ext = [], []
    for prop, (_, art) in sorted(run1.items()):
        res = v5(art)
        works = {w["work_item_id"]: w for w in res["work_items"]}
        decs = {d["package_candidate_id"]: d for d in res["package_decisions"]}
        for c in res["package_candidates"]:
            if c.get("display_only"):
                continue
            d = decs[c["package_candidate_id"]]
            kids = [{"action": works[w]["action_code"], "catalog_item_ids": works[w]["catalog_item_ids"],
                     "low": works[w]["low"], "high": works[w]["high"]} for w in c["child_work_item_ids"]]
            row = {"property": prop, "package": f"{c['package_type']}|{c['estimate_unit_id']}", "tier": c["pricing_tier"],
                   "price": [c["low"], c["high"]], "children": kids, "sol": d["decision"], "rationale": d["rationale"],
                   "split_groups": d.get("split_groups") or [], "combine_with": d.get("combine_with") or []}
            if len(kids) == 1:
                singles.append(row)
            if c["package_type"] == "exterior_repair":
                ext.append(row)
    out["p4_sol"] = {"identical_children_same": same, "identical_children_flips": flips,
                     "same_key_different_children": diff, "single_child": singles, "exterior_repair": ext}

    # ---- P6: Tier-2 $0 -> priced -------------------------------------
    t2 = []
    items = {i["review_id"]: i for i in reviews.get("report_items", [])}
    for rid in TIER2_ZERO_IDS:
        it = items.get(rid)
        if not it:
            continue
        prop = it["property_key"]; cid, _, unit = it["key"].partition("|")
        art = run1[prop][1]; v4 = art["renovation_estimate_v4"]; res = v5(art)
        reason = None
        for g in v4.get("groups") or []:
            for li in g.get("line_items") or []:
                if li.get("catalog_item_id") == cid:
                    u = li.get("billable_estimate_unit_id") or li.get("estimate_unit_id")
                    if u == unit or any(a.get("estimate_unit_id") == unit for a in (li.get("unit_member_allocations") or [])):
                        reason = li.get("pass_2f_fallback_reason") or li.get("estimate_verification_status")
        cond = next((c for c in res["observed_conditions"] if c["catalog_item_id"] == cid and c["estimate_unit_id"] == unit), None)
        ev = next((e for e in res["evidence_facts"] if cond and e["condition_id"] == cond["condition_id"]), {})
        rv = next((r for r in res["condition_reviews"] if cond and r["condition_id"] == cond["condition_id"]), {})
        v5rows = it.get("candidate") or []
        t2.append({"review_id": rid, "property": prop, "item": it["key"], "v4_zero_reason": reason,
                   "v5_price": [sum(r["low"] for r in v5rows), sum(r["high"] for r in v5rows)],
                   "distinct_photo_count": ev.get("distinct_photo_count"), "terra": rv.get("verdict"),
                   "terra_rationale": rv.get("rationale"), "photo_keys": ev.get("photo_keys"),
                   "photo_paths": [photo_path(art, prop, k) for k in (ev.get("photo_keys") or [])]})
    out["p6_tier2_zero_items"] = t2
    return out


# ----------------------------------------------------------------------------
def esc(x):
    return html.escape(str(x if x is not None else ""))


def img_strip(paths, keys=None):
    keys = keys or [Path(p).name for p in paths]
    cells = []
    for p, k in zip(paths, keys):
        u = file_url(p)
        cells.append(f'<a href="{u}" target="_blank"><img src="{u}" loading="lazy" alt="{esc(k)}"><span>{esc(k)}</span></a>')
    return '<div class="strip">' + "".join(cells) + "</div>"


def render_html(d, date_tag):
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:0;padding:16px 24px;background:#fafafa;color:#222}
    h1{font-size:22px} h2{font-size:18px;border-bottom:2px solid #444;padding-top:18px} h3{font-size:15px;margin:14px 0 6px}
    .card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px 14px;margin:10px 0}
    .strip{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0}
    .strip a{display:flex;flex-direction:column;align-items:center;text-decoration:none;color:#555;font-size:11px}
    .strip img{max-height:230px;max-width:340px;border:1px solid #bbb;border-radius:4px}
    .v{display:grid;grid-template-columns:140px 1fr;gap:4px 10px;font-size:13px}
    .k{color:#666} .tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:12px;margin-right:4px}
    .sup{background:#d9f2d9} .uns{background:#f7d9d9} .rej{background:#f7d9d9} .con{background:#d9f2d9} .can{background:#eee}
    .appr{background:#d9f2d9} .mono{font-family:Consolas,monospace;font-size:12px}
    details summary{cursor:pointer;font-weight:600} .money{font-family:Consolas,monospace}
    nav a{margin-right:14px} .small{font-size:12px;color:#555} table{border-collapse:collapse;font-size:13px}
    td,th{border:1px solid #ccc;padding:3px 6px;vertical-align:top}
    """
    L = [f"<!doctype html><html><head><meta charset='utf-8'><title>Session 9 photo review</title><style>{css}</style></head><body>",
         "<h1>Session 9 canary — photo review sheet</h1>",
         f"<p class='small'>Generated {date_tag} from artifacts_canary/renovation_session9_20260818 (run_1). Photos open from the local renointel-prod image folder. "
         "Record verdicts in reports/session9_decision_worksheet.json using the item ids shown (C###, P##, B##).</p>",
         "<nav><a href='#p2'>P2 conditions (2f vs Terra)</a><a href='#p1'>P1 packages (Sol vs 2f)</a><a href='#p3'>P3 bathrooms</a><a href='#p4'>P4 Sol consistency</a><a href='#p6'>P6 single-photo items</a></nav>"]

    # P2
    L.append("<h2 id='p2'>P2 — Condition-level disagreements: v4 Pass 2f vs v5 Terra (same photos)</h2>")
    L.append("<p class='small'>Vocabulary for the worksheet: <b>terra_correct</b> | <b>2f_correct</b> | <b>both_partly</b> | <b>cannot_tell</b>. "
             "Direction A = Terra supported / 2f rejected (v5 may be hallucinating). Direction B = 2f confirmed / Terra unsupported-or-cannot_assess (v5 may be over-rejecting).</p>")
    L.append(f"<p class='small'>{d['p2_issue_rows']} v4 issue-level verdicts collapse to {len(d['p2_contradictions'])} distinct v5 conditions; each card below is one condition (one photo set, one Terra verdict) and lists every v4 issue that mapped to it.</p>")
    for direction, title in (("terra_supported_2f_rejected", "Direction A — Terra SUPPORTED, 2f REJECTED"),
                             ("2f_confirmed_terra_not", "Direction B — 2f CONFIRMED, Terra UNSUPPORTED / CANNOT_ASSESS"),
                             ("mixed_2f_verdicts", "Mixed — 2f confirmed some and rejected other issues of the same condition")):
        rows = [c for c in d["p2_contradictions"] if c["direction"] == direction]
        if not rows:
            continue
        L.append(f"<h3>{title} ({len(rows)})</h3>")
        byprop = defaultdict(list)
        for c in rows:
            byprop[c["property"]].append(c)
        for prop, cs in byprop.items():
            L.append(f"<details open><summary>{esc(prop)} — {len(cs)} condition(s)</summary>")
            for c in cs:
                w = c["v5_work"]
                L.append("<div class='card'>")
                L.append(f"<div><b>{c['item_id']}</b> &nbsp; <span class='mono'>{esc(c['catalog_item_id'])} @ {esc(c['estimate_unit_id'])}</span> "
                         f"&nbsp; photos: {esc(', '.join(c['terra_photo_keys'] or []))} (distinct {esc(c['distinct_photo_count'])})</div>")
                L.append(img_strip(c["photo_paths"], c["terra_photo_keys"]))
                L.append("<div class='v'>")
                issues = "".join(f"<div><span class='tag {'con' if i['v4_2f_verdict']=='confirmed' else 'rej'}'>2f {esc(i['v4_2f_verdict'])}</span> {esc(i['observation'])} <span class='small mono'>{esc(i['issue_id'])}</span></div>" for i in c["v4_issues"])
                L.append(f"<div class='k'>upstream issue(s) → v4 2f per-issue verdict</div><div>{issues}</div>")
                L.append(f"<div class='k'>v4 2f package view ({esc(c['v4_2f_package'])})</div><div>{esc(c['v4_2f_evidence_summary'])}</div>")
                L.append(f"<div class='k'>v5 Terra</div><div><span class='tag {'sup' if c['terra_verdict']=='supported' else ('can' if c['terra_verdict']=='cannot_assess' else 'uns')}'>{esc(c['terra_verdict'])}</span> {esc(c['terra_rationale'])}</div>")
                L.append(f"<div class='k'>v5 outcome</div><div>{esc(c['v5_disposition'])}" + (f" → {esc(w['action'])} <span class='money'>{money(w['low'], w['high'])}</span>" if w else "") + "</div>")
                L.append("</div></div>")
            L.append("</details>")

    # P1
    L.append("<h2 id='p1'>P1 — Package-level disagreements: v4 2f rejected the package, v5 Sol judged it</h2>")
    L.append("<p class='small'>Worksheet vocabulary: <b>package_warranted</b> | <b>not_warranted</b> | <b>unsure</b>. The 2f review photos are what v4's verifier saw for the whole package; each child shows the photo(s) Terra reviewed for that condition.</p>")
    for p in d["p1_packages"]:
        L.append("<div class='card'>")
        L.append(f"<div><b>{p['item_id']}</b> &nbsp; <b>{esc(p['property'])}</b> &nbsp; <span class='mono'>{esc(p['package'])}</span> &nbsp; v5 tier {esc(p['v5_tier'])} "
                 f"<span class='money'>{money(*p['v5_price'])}</span> &nbsp; Sol: <span class='tag {'appr' if p['v5_sol_decision']=='approve' else 'rej'}'>{esc(p['v5_sol_decision'])}</span> "
                 f"status {esc(p['v5_status'])} effective <span class='money'>{money(*p['v5_effective'])}</span>; children standalone <span class='money'>{money(*p['v5_children_standalone'])}</span>; net lift <span class='money'>{money(*p['v5_net_lift'])}</span></div>")
        L.append(f"<div class='v'><div class='k'>v4 2f verdict</div><div><span class='tag rej'>{esc(p['v4_2f_status'])}</span> (issues confirmed {p['v4_2f_confirmed']} / rejected {p['v4_2f_rejected']}; v4 tier {esc(p['v4_tier'])} {money(*p['v4_price'])})<br>{esc(p['v4_2f_evidence_summary'])}</div>"
                 f"<div class='k'>Sol rationale</div><div>{esc(p['v5_sol_rationale'])}</div></div>")
        L.append(f"<div class='small'>2f review photos: {esc(', '.join(p['v4_2f_review_photos']))}</div>")
        L.append(img_strip(p["v4_2f_review_photo_paths"], p["v4_2f_review_photos"]))
        L.append("<div><b>v5 children</b></div>")
        for ch in p["children"]:
            L.append(f"<div class='small'><b>[{esc(ch['role'])}] {esc(ch['action'])}</b> {esc(', '.join(ch['catalog_item_ids']))} <span class='money'>{money(ch['low'], ch['high'])}</span></div>")
            for cc in ch["conditions"]:
                L.append(f"<div class='small' style='margin-left:16px'>{esc(cc['catalog_item_id'])}: <span class='tag {'sup' if cc['terra']=='supported' else 'uns'}'>{esc(cc['terra'])}</span> {esc(cc['rationale'])} <i>({esc(', '.join(cc['photo_keys'] or []))})</i></div>")
                if cc["photo_paths"]:
                    L.append(img_strip(cc["photo_paths"], cc["photo_keys"]))
        L.append("</div>")

    # P3
    L.append("<h2 id='p3'>P3 — Multi-bathroom properties: are the surrogates distinct bathrooms?</h2>")
    L.append("<p class='small'>Worksheet vocabulary: <b>distinct_bathrooms</b> = how many different bathrooms you see across the strips; <b>bill</b> = per_bathroom | once | unsure. "
             "Both engines merged all bathroom surrogates into one estimate unit; v4 then re-expanded its modernization package per qualifying surrogate, v5 did not.</p>")
    for b in d["p3_multi_bath"]:
        L.append("<div class='card'>")
        qual = ", ".join(b["v4_qualifying_surrogates"] or []) or "—"
        v4p = "; ".join(f"{p['surrogate']} {money(*p['cost'])} ({p['confirmed_issue_count']} confirmed issues)" for p in b["v4_bath_mod_packages"]) or "none applied"
        v5p = "; ".join(f"{u} {t} {money(lo, hi)}" for u, t, lo, hi in b["v5_bath_mod"]) or "none"
        L.append(f"<div><b>{b['item_id']}</b> &nbsp; <b>{esc(b['property'])}</b> &nbsp; listing baths {esc(b['listing_baths'])} &nbsp; surrogates {len(b['surrogates'])} &nbsp; "
                 f"v4 signal likely_multiple: {esc(b['v4_signal_likely_multiple'])}; expanded: {esc(b['v4_expanded'])}; qualifying: {esc(qual)}; fallback: {esc(b['v4_fallback_reason'] or '—')}</div>")
        L.append(f"<div class='small'>v4 bathroom_modernization packages: {esc(v4p)} &nbsp; | &nbsp; v5 (effective): {esc(v5p)} &nbsp; | &nbsp; v4−v5 bath-mod dollars <span class='money'>{money(*b['v4_minus_v5_bathmod'])}</span></div>")
        for s in b["surrogates"]:
            L.append(f"<div class='small'><b>{esc(s['id'])}</b> photos {esc(', '.join(s['photo_keys']))}</div>")
            L.append(img_strip(s["photo_paths"], s["photo_keys"]))
        L.append("<details><summary>v5 merged bathroom conditions (all at bathroom_primary, identity_ambiguous)</summary><table><tr><th>catalog item</th><th>source surrogates</th><th>Terra</th><th>photos</th></tr>")
        for c in b["v5_conditions"]:
            L.append(f"<tr><td>{esc(c['catalog_item_id'])}</td><td>{esc(c['source_surrogates'])}</td><td>{esc(c['terra'])}</td><td>{esc(c['photo_keys'])}</td></tr>")
        L.append("</table></details></div>")

    # P4
    s = d["p4_sol"]
    L.append("<h2 id='p4'>P4 — Sol consistency (no photos needed: grouping coherence)</h2>")
    L.append(f"<p>Identical-children candidates present in both replicas: same decision {s['identical_children_same']}, flipped {len(s['identical_children_flips'])}; "
             f"{s['same_key_different_children']} same-key candidates had different children (upstream churn, not comparable).</p>")
    L.append("<table><tr><th>property</th><th>package</th><th>children</th><th>run 1</th><th>run 2</th></tr>")
    for f in s["identical_children_flips"]:
        L.append(f"<tr><td>{esc(f['property'])}</td><td>{esc(f['package'])}</td><td>{esc(f['children'])}</td><td>{esc(f['sol_run1'])}: {esc(f['rationale_run1'])}</td><td>{esc(f['sol_run2'])}: {esc(f['rationale_run2'])}</td></tr>")
    L.append("</table>")
    for title, rows in (("Single-child candidates (run 1)", s["single_child"]), ("exterior_repair candidates (run 1)", s["exterior_repair"])):
        L.append(f"<h3>{title}</h3><table><tr><th>property</th><th>package</th><th>tier / price</th><th>children</th><th>Sol</th><th>rationale</th></tr>")
        for r in rows:
            kids = "; ".join(f"{k['action']} {','.join(k['catalog_item_ids'])} {money(k['low'], k['high'])}" for k in r["children"])
            L.append(f"<tr><td>{esc(r['property'])}</td><td>{esc(r['package'])}</td><td>{esc(r['tier'])} {money(*r['price'])}</td><td>{esc(kids)}</td><td><span class='tag {'appr' if r['sol']=='approve' else 'rej'}'>{esc(r['sol'])}</span></td><td>{esc(r['rationale'])}{(' split=' + esc(r['split_groups'])) if r['split_groups'] else ''}</td></tr>")
        L.append("</table>")

    # P6
    L.append("<h2 id='p6'>P6 — Tier-2 '$0 → priced' items (single-photo opportunity drivers)</h2>")
    for t in d["p6_tier2_zero_items"]:
        L.append("<div class='card'>")
        L.append(f"<div><b>{esc(t['review_id'])}</b> {esc(t['property'])} <span class='mono'>{esc(t['item'])}</span> v4 $0 reason: <i>{esc(t['v4_zero_reason'])}</i> → v5 <span class='money'>{money(*t['v5_price'])}</span>; photos {esc(t['distinct_photo_count'])}; Terra {esc(t['terra'])}: {esc(t['terra_rationale'])}</div>")
        L.append(img_strip(t["photo_paths"], t["photo_keys"]))
        L.append("</div>")
    L.append("</body></html>")
    return "\n".join(L)


def worksheet(d):
    ws = {"schema_version": 1, "instructions": {
        "p2_conditions": "verdict: terra_correct | 2f_correct | both_partly | cannot_tell",
        "p1_packages": "verdict: package_warranted | not_warranted | unsure",
        "p3_bathrooms": "distinct_bathrooms: integer you see; bill: per_bathroom | once | unsure",
    }}
    ws["p2_conditions"] = {c["item_id"]: {"ref": f"{c['property']} {c['catalog_item_id']}@{c['estimate_unit_id']}", "verdict": "", "notes": ""} for c in d["p2_contradictions"]}
    ws["p1_packages"] = {p["item_id"]: {"ref": f"{p['property']} {p['package']}", "verdict": "", "notes": ""} for p in d["p1_packages"]}
    ws["p3_bathrooms"] = {b["item_id"]: {"ref": b["property"], "distinct_bathrooms": "", "bill": "", "notes": ""} for b in d["p3_multi_bath"]}
    return ws


def md_tables(d):
    T = {}
    f = d["facts"]
    ch = f.get("churn") or {}
    tp = f.get("terra_condition_proxy") or {}
    tk = f.get("tokens_per_listing_both_replicas") or {}
    T["facts"] = "\n".join([
        f"- Corpus (run_1, 18 properties): v4 {money(*f['corpus_v4'])} → v5 {money(*f['corpus_v5'])} ({f['corpus_delta_pct'][0]:+.1%} / {f['corpus_delta_pct'][1]:+.1%}).",
        f"- Terra verdicts (both replicas): {f['terra_verdicts_both_runs']}; dispositions: {f['dispositions_both_runs']}; Sol decisions: {f['sol_decisions_both_runs']}.",
        f"- Models: Terra {list(f['models']['terra'])}, Sol {list(f['models']['sol'])}; reasoning_effort Terra={f['terra_review_reasoning_effort']}, Sol={f['sol_review_reasoning_effort']}; upstream routing {f['upstream_routing']}.",
        f"- Replica churn, keyed {ch.get('key')}: on average {ch.get('mean_share_of_run1_conditions_absent_from_run2', 0):.0%} of a property's replica-1 conditions are absent from replica 2; {ch.get('mean_symdiff_over_union', 0):.0%} of the union differs.",
        f"- Terra condition-level consistency proxy: {tp.get('identical_item_unit_photoset_conditions_in_both_replicas')} conditions with the same item/unit/photo set in both replicas → same verdict {tp.get('same_verdict')}, flipped {tp.get('flipped')} ({tp.get('flip_kinds')}); {tp.get('note')}.",
        f"- Measured tokens per listing (both replicas, {tk.get('n_artifacts')} artifacts): Terra condition review mean {tk.get('terra_mean'):,}, Sol package review mean {tk.get('sol_mean'):,}.",
    ])
    p = d["p1_totals"]
    rows = ["| id | property | package | v4 tier / price | v5 tier | Sol | v5 effective | children standalone | net lift | v4 2f | 2f says |", "|---|---|---|---|---|---|---:|---:|---:|---|---|"]
    for x in d["p1_packages"]:
        rows.append(f"| {x['item_id']} | {x['property']} | `{x['package']}` | {x['v4_tier']} {money(*x['v4_price'])} | {x['v5_tier']} | {x['v5_sol_decision']} | {money(*x['v5_effective'])} | {money(*x['v5_children_standalone'])} | {money(*x['v5_net_lift'])} | {x['v4_2f_status']} ({x['v4_2f_confirmed']}✓/{x['v4_2f_rejected']}✗ issues) | {short(x['v4_2f_evidence_summary'], 200)} |")
    rows.append(f"\n{p['count_2f_rejected']} candidates 2f rejected; Sol approved {p['count_sol_approved']}; those bill {money(*p['sol_approved_effective'])} effective, net lift over children {money(*p['sol_approved_net_lift'])}. (v4 tier/price = what v4 would have billed had 2f confirmed; two of the Sol-approved cases are also tier demotions.)")
    T["p1_packages"] = "\n".join(rows)
    ag = d["p2_agreement_issue_level"]
    rows = ["Agreement matrix at the **v4 issue level** (each of v4's 2f per-issue verdicts mapped to the v5 condition that absorbed that issue; one condition can carry several issues):", "",
            "| | Terra supported | Terra unsupported | Terra cannot_assess | no v5 condition |", "|---|---:|---:|---:|---:|"]
    for v in ("confirmed", "rejected"):
        rows.append(f"| 2f {v} | {ag.get(f'2f_{v}|terra_supported', 0)} | {ag.get(f'2f_{v}|terra_unsupported', 0)} | {ag.get(f'2f_{v}|terra_cannot_assess', 0)} | {ag.get(f'2f_{v}|no_v5_condition', 0)} |")
    A = sum(1 for c in d["p2_contradictions"] if c["direction"] == "terra_supported_2f_rejected")
    B = sum(1 for c in d["p2_contradictions"] if c["direction"] == "2f_confirmed_terra_not")
    M = len(d["p2_contradictions"]) - A - B
    rows.append(f"\nDisagreements: **{d['p2_issue_rows']} v4 issue-level verdicts on {len(d['p2_contradictions'])} distinct v5 conditions** — **{A}** Terra-supported / 2f-rejected (direction A), **{B}** 2f-confirmed / Terra-not (direction B)" + (f", {M} with mixed 2f verdicts" if M else "") + ". The sheet and the worksheet are keyed per condition (one photo set, one Terra verdict); the v4 issues are listed inside each card.")
    rows.append("\nBy catalog item, counted per condition (A / B):\n")
    rows.append("| catalog item | A: Terra sup, 2f rej | B: 2f con, Terra not |")
    rows.append("|---|---:|---:|")
    for k, v in d["p2_by_catalog_item"].items():
        rows.append(f"| `{k}` | {v.get('terra_supported_2f_rejected', 0)} | {v.get('2f_confirmed_terra_not', 0)} |")
    sp = d["single_photo_work"]
    rows.append(f"\nSingle-photo share of v5 work (run_1): {sp['work_items']} active work items; one-photo items carry {money(*sp['one_photo_dollars'])} of {money(*sp['all_active_dollars'])} of active work-item dollars (pre-package, standalone-priced — not the headline).")
    T["p2_summary"] = "\n".join(rows)
    rows = ["| id | property | item @ unit | photos | v4 issues (2f) | Terra | v5 outcome |", "|---|---|---|---|---|---|---|"]
    for c in d["p2_contradictions"]:
        w = c["v5_work"]
        rows.append(f"| {c['item_id']} | {c['property']} | `{c['catalog_item_id']}` @ {c['estimate_unit_id']} | {', '.join(c['terra_photo_keys'] or [])} | {c['v4_issue_count']} {'/'.join(c['v4_2f_verdicts'])} | {c['terra_verdict']} | {c['v5_disposition']}{(' ' + money(w['low'], w['high'])) if w else ''} |")
    T["p2_list"] = "\n".join(rows)
    rows = ["| id | property | listing baths | surrogates (photos) | v4 signal / expanded | v4 bath-mod pkgs | v5 bath-mod | v4−v5 bath-mod $ |", "|---|---|---:|---|---|---|---|---:|"]
    for b in d["p3_multi_bath"]:
        surr = "; ".join(f"{s['id']}({len(s['photo_keys'])})" for s in b["surrogates"])
        v4p = ", ".join(f"{p['surrogate']}" for p in b["v4_bath_mod_packages"]) or "none applied"
        rows.append(f"| {b['item_id']} | {b['property']} | {b['listing_baths']} | {surr} | {b['v4_signal_likely_multiple']} / {b['v4_expanded']} ({b['v4_fallback_reason'] or 'expanded'}) | {len(b['v4_bath_mod_packages'])}: {v4p} | {len(b['v5_bath_mod'])} | {money(*b['v4_minus_v5_bathmod'])} |")
    T["p3_multibath"] = "\n".join(rows)
    s = d["p4_sol"]
    rows = [f"Identical-children candidates in both replicas: **{s['identical_children_same']} same decision, {len(s['identical_children_flips'])} flipped**; {s['same_key_different_children']} same-key candidates had different children (upstream churn; not comparable).", "",
            "| property | package | children | run 1 | run 2 |", "|---|---|---|---|---|"]
    for x in s["identical_children_flips"]:
        rows.append(f"| {x['property']} | `{x['package']}` | {x['children']} | {x['sol_run1']} — {x['rationale_run1']} | {x['sol_run2']} — {x['rationale_run2']} |")
    sc = Counter(r["sol"] for r in s["single_child"]); ec = Counter(r["sol"] for r in s["exterior_repair"])
    rows.append(f"\nSingle-child candidates (run_1): {dict(sc)}. exterior_repair candidates (run_1): {dict(ec)}.\n")
    rows.append("| property | package | children | Sol | rationale |"); rows.append("|---|---|---|---|---|")
    for r in s["single_child"] + [e for e in s["exterior_repair"] if len(e["children"]) > 1]:
        kids = "; ".join(f"{k['action']}[{','.join(k['catalog_item_ids'])}]" for k in r["children"])
        rows.append(f"| {r['property']} | `{r['package']}` | {kids} | {r['sol']} | {short(r['rationale'], 160)}{(' split=' + str(r['split_groups'])) if r['split_groups'] else ''} |")
    T["p4_sol"] = "\n".join(rows)
    rows = ["| review id | property | item | v4 $0 reason | v5 price | photos | Terra |", "|---|---|---|---|---:|---:|---|"]
    for t in d["p6_tier2_zero_items"]:
        rows.append(f"| {t['review_id']} | {t['property']} | `{t['item']}` | {t['v4_zero_reason']} | {money(*t['v5_price'])} | {t['distinct_photo_count']} | {t['terra']}: {short(t['terra_rationale'], 120)} |")
    T["p6_tier2"] = "\n".join(rows)
    return T


def splice(doc_path, tables):
    text = doc_path.read_text(encoding="utf-8")
    for name, body in tables.items():
        pat = re.compile(rf"(<!-- GEN:{name} -->)(.*?)(<!-- /GEN:{name} -->)", re.S)
        if not pat.search(text):
            print(f"  warning: no GEN:{name} marker in {doc_path.name}")
            continue
        text = pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
    doc_path.write_text(text, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canary-root", type=Path, default=CANARY)
    ap.add_argument("--report", type=Path, default=ROOT / "reports/renovation_architecture_session9_canary_20260821.json")
    ap.add_argument("--out-json", type=Path, default=ROOT / "reports/session9_decision_packets.json")
    ap.add_argument("--out-html", type=Path, default=ROOT / "reports/session9_photo_review_20260821.html")
    ap.add_argument("--out-worksheet", type=Path, default=ROOT / "reports/session9_decision_worksheet.json")
    ap.add_argument("--doc", type=Path, default=ROOT / "docs/DESIGN_renovation_architecture_decision_packets.md")
    ap.add_argument("--date-tag", default="2026-08-21")
    ap.add_argument("--no-worksheet", action="store_true", help="do not (re)write the worksheet (preserves filled verdicts)")
    args = ap.parse_args(argv)
    run1 = _load_latest_artifacts(args.canary_root / "run_1/candidate")
    run2 = _load_latest_artifacts(args.canary_root / "run_2/candidate")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    d = build(run1, run2, {"report_items": report.get("review_items") or []})
    args.out_json.write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
    args.out_html.write_text(render_html(d, args.date_tag), encoding="utf-8")
    if not args.no_worksheet:
        if args.out_worksheet.exists():
            print(f"  worksheet exists, not overwritten: {args.out_worksheet} (pass nothing; delete it to regenerate)")
        else:
            args.out_worksheet.write_text(json.dumps(worksheet(d), indent=1), encoding="utf-8")
    if args.doc.exists():
        splice(args.doc, md_tables(d))
    print(f"packets: P1={len(d['p1_packages'])} P2={len(d['p2_contradictions'])} P3={len(d['p3_multi_bath'])} "
          f"sol flips={len(d['p4_sol']['identical_children_flips'])}/{d['p4_sol']['identical_children_same'] + len(d['p4_sol']['identical_children_flips'])}")
    print("json ->", args.out_json); print("html ->", args.out_html); print("doc  ->", args.doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
