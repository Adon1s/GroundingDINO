import json, os, sys, hashlib
from collections import Counter
sys.path.insert(0, os.getcwd())
from tools.pipeline_common import term_matches

SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
wf = json.load(open(f"{SP}/pass2d_phaseA_workflow1_result.json", encoding="utf-8"))
res = wf["result"]

# ---------- coordinator verifications (d), (e), (f) ----------
corpus = json.load(open("reports/catalog_audit_replay_corpus.json", encoding="utf-8"))
prod = {(r["run_id"], r["issue_id"]) for r in corpus["rows"]}
rows = []
for a in corpus["artifacts"]:
    p = a.get("artifact_path")
    if not a.get("readable") or not p or not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    for pk, ph in (d.get("photos") or {}).items():
        for r in (ph.get("debug") or {}).get("resolved_items") or []:
            rows.append((a["run_id"], r))
ver = {}
for run, r in rows:
    if r.get("issue_id") == "d8fc3b6364d55cc0":
        c = r["candidates"]; top = c[0]; obs = r["description"].lower()
        firing = [t for t in top["support_any"] if term_matches(t.lower(), obs)]
        ver["row3b_d8fc3b6364d55cc0"] = dict(
            top=top["item_id"], top_score=round(c[0]["score"], 5), margin=round(c[0]["score"] - c[1]["score"], 5),
            top_defaultHidden=top.get("defaultHidden"), firing_terms=firing, resolved=r["resolved_item_id"], path=r["resolution_path"],
            would_shortcut_but_for_generic_gate=bool(firing) and c[0]["score"] >= 0.72 and (c[0]["score"] - c[1]["score"]) >= 0.03)
nulls_dd = [r for run, r in rows if r.get("resolved_item_id") is None and r.get("resolution_path") == "llm"
            and (run, r.get("issue_id")) in prod and r.get("original_kind") in ("degradation", "defect")]
ver["degradation_defect_nulls"] = dict(total=len(nulls_dd), containing_window=sum(1 for r in nulls_dd if "window" in r["description"].lower()))
off = sum(1 for run, r in rows if r.get("resolved_item_id") and r["resolved_item_id"] not in [c["item_id"] for c in r.get("candidates") or []])
ver["off_list_selections"] = dict(off_list=off, llm_nonnull_rows=sum(1 for run, r in rows if r.get("resolved_item_id") and r.get("resolution_path") == "llm"), all_rows=len(rows))
print("VERIFICATIONS:", json.dumps(ver, indent=1))

# ---------- reconciliation ----------
prov = {1: "C", 2: "B", 3: "C", 4: "A", 5: "A", 6: "A", 7: "A", 8: "B", 9: "B", 10: "C", 11: "B", 12: "B", 13: "G", 14: "B", 15: "B",
        16: "D", 17: "D", 18: "B", 19: "D", 20: "D", 21: "B", 22: "D", 23: "B", 24: "B", 25: "B", 26: "C",
        101: "F", 102: "E", 103: "F", 104: "E", 105: "E", 106: "F", 107: "F", 108: "E", 109: "F", 110: "F"}
R = {
 1: ("B", None, False, "Text tie; Steven's 2026-08-28 retag on rc_4cab6e7be02b: 'perhaps this one is not problematic'. Naming only; no economics delta."),
 2: ("B", None, False, "LEAD-ONLY model_judge unit; text favours the selected generic damage item; the 2a text never names vinyl or tearing."),
 3: ("C", None, True, "Rank-1 worn_or_stained_flooring passed over twice for rank 3. Steven's retag on rc_eef1daf73e2a ('I see stains not scuffs') supports the reviewer. Worklist note misstates the matcher ('scuff' does fire) and omits that the generic gate on the default-hidden rank-1 item, not a failed term, is why no shortcut fired; on photo_029 every other gate passed and the shortcut would have returned the reviewer's item."),
 4: ("A", "D", True, "Shortcut fired on the single bare token 'vinyl'; the ledger's 'worn' claim is wrong (the worklist row already says so). The style item was kind-gated (secondary D). A lost billing."),
 5: ("A", None, True, "Shortcut fired on 'tub$' used as a location word; margin 0.03012, 0.00012 above the gate. A wrong-subject billing still live."),
 6: ("A", None, False, "Same 'tub$' locative mechanism; no incremental cost because four bullets collapsed into one already-billed condition. Rank-3 wall_scuffs fits 'look rough' as well as the nominated item."),
 7: ("A", None, False, "Shortcut fired on bare 'window' (only 'window' fires, not 'windows'); outcome Terra-supported and correct; costs nothing. Mechanism present, no failure. The worklist better_item still names the removed parent: decision-record R9 is false about the shipped file."),
 8: ("B", None, False, "Text favours the selected porch item. Steven's retag on rc_3a19f759f2d9: 'I would reverse my decision here. I think it is in need of work so the LLM is right'. Not a Pass 2d failure on the human record."),
 9: ("B", None, False, "labels_v1_1: claim=exact, work=warranted, retag_answer=terra_miss on the soffit item; the human record ratifies 2d's choice and names Terra. Filed as a 2d failure anyway; it is not one."),
 10: ("C", None, True, "'unfinished and patchy' resolved to the OSB item over rank-3 drywall damage; plausible model error on the text. Human evidence conflicts: v1 verdict terra_claim_supported (reviewer saw the OSB claim as supported) versus the later photo review (patchy paint, no OSB). Lead with conflicting human evidence; the worklist note argues from embed_text the model never sees."),
 11: ("B", None, False, "Text tie; the gold finding says 'peeling paint and staining' but the 2a text says 'discoloration and patching' - the selecting word was deleted upstream. This is the single HUMAN unit CAP-014 holds under D5; the note calls it lead-graded."),
 12: ("B", None, False, "'scraped paint' appears in no rendered clause of the peeling item; the note's discriminator (work_item_code) is not rendered."),
 13: ("G", None, False, "Graffiti: no item; the nearest home is a naming choice. Shares one condition with row 14."),
 14: ("B", None, False, "Same condition as row 13 (oc1_f5a8a54a0117c5b3); one billing, not two."),
 15: ("B", None, False, "The note itself says the retrieval query was weak upstream; axis and body disagree."),
 16: ("B", "D", False, "Rank 2 over rank 1 at margin 0.0015; text favours the scuffs item (names stains, minor damage) over 'Paint that is peeling'. The true home (deteriorated wallboard) is kind defect and was kind-gated. Axis 'shortcut_or_rank1_selection' is wrong on both halves."),
 17: ("D", None, False, "damaged_drywall_or_cracks is kind defect against a degradation bullet; never in the searched matrix. The note's 'kind-open' premise is false and 'route via embed_text' cannot work."),
 18: ("B", None, False, "Rank 3 over rank 1; the rank-1 bathroom paint item's rendered description says 'without moisture evidence' and never names staining; text favours the scuffs item. Axis wrong on both halves. Shares a condition with row 19."),
 19: ("B", "D", False, "water_stain_ceiling is kind defect; kind-gated, not 'kind-open'. Same condition as row 18."),
 20: ("D", None, False, "damaged_drywall_or_cracks kind defect against degradation; the note's 'gate-open' is false. Both items carry DRYWALL_PATCH."),
 21: ("B", None, False, "The note body says the 2a text mislabels peeling as patching - an upstream text error filed as selection. Full 8-item list, no guardrail drop."),
 22: ("B", "D", False, "Drywall item kind-gated; the companion issue 928a53b83f7dc9ec is present in the artifact and resolved."),
 23: ("B", None, False, "Reproduces exactly but is no longer reachable under D4 (item scene-excluded from bathrooms); Terra supported and billed; now an instance of D4's disclosed 'wallcovering' residual and the CCF-13 double-charge reproduction case."),
 24: ("B", None, False, "The siding item's rendered description names 'or trim'; no exterior painted-trim item exists; the nominated door item's claim is narrower than the defect. A lost billing, but a coverage and wording loss, not a selection one."),
 25: ("B", None, False, "Same pattern; costs nothing (the condition was carried by two true siding bullets)."),
 26: ("C", None, True, "'The shower trim is old.' - rank-1 outdated_bathroom_finishes passed over for rank-3 trim; homograph. The checkpoint_note is one-sided: the selected side costs nothing under D1 but the foregone side is a severity-3 bathroom_modernization driver. Near-miss: 'old' fires on the rank-1 item and only the 0.72 score floor (0.703) blocked a shortcut onto it."),
 101: ("F", None, False, "Blinds successor at rank 2 (list of 7 after a require_any drop); model declined a two-subject bullet. Accepted under D3."),
 102: ("E", None, False, "Neither successor retrieved; Terra's rationale in both runs describes brown fabric valances; the nominated blinds item deny-lists 'valance'. compare.json buckets it surviving_on_fabric with reaches_successor false. A fabric-successor retrieval miss of a $92-$460 baseline billing. Accepted under D3; evidence for a human."),
 103: ("F", None, False, "Blinds successor at rank 1; the shortcut was lost to a sub-threshold margin (0.0228) as a side effect of the split; the model declined a two-subject bullet. Accepted under D3."),
 104: ("E", None, False, "Neither successor retrieved for 'Window treatments are dated.' in a flat 0.57-0.63 band. Accepted under D3."),
 105: ("E", None, False, "Neither successor retrieved; Terra's rationale describes curtains; the blinds item deny-lists 'curtain'. Accepted under D3."),
 106: ("F", None, False, "Blinds successor at rank 2 for 'The blinds appear dated and mismatched.'; the cleanest decline of a present item. Accepted under D3."),
 107: ("F", None, False, "Blinds successor at rank 2; two-subject bullet. Accepted under D3."),
 108: ("E", None, False, "Both successors retrieved at ranks 1-2 then removed by their own guardrails (makeshift, curtain), correctly; compare.json bucket reaching_neither. The shared note contradicts the bundle it was written from. Accepted under D3."),
 109: ("F", None, False, "Blinds successor at rank 5; Terra's rationale describes sheer curtains; the blinds item deny-lists 'sheer'. Accepted under D3."),
 110: ("F", None, False, "Blinds successor at rank 4 for 'window treatments and trim are dated stylistically'; R9 re-pointed this row to a naming question. Accepted under D3."),
}
inv = {}
for k in ("case_rows1_10", "case_cap014", "case_rows23_26_and_d3"):
    for r in res[k]["rows"]:
        inv[r["worklist_index"]] = r
out_rows = []
KEEP = ("issue_id", "property_key", "photo_key", "artifact_path", "observation", "kind", "scene_group", "resolution_path", "shortcut_firing_terms",
        "candidate_ids_in_rank_order", "selected_item", "selected_rank", "better_item_claimed", "better_item_rank_observed", "better_item_kind",
        "better_item_reachable", "successor_in_list", "successor_rank", "terra_verdict", "disposition", "reason_code", "cost_now", "worklist_discrepancy", "citations")
for idx in sorted(R):
    r = inv[idx]; cat, sec, fail, note = R[idx]
    row = {"worklist_index": idx, "cap": r["cap"], "case_ref": r["case_ref"]}
    row.update({k: r.get(k) for k in KEEP})
    row.update(dict(investigator_category=r["category"], coordinator_provisional_category=prov[idx], reconciled_category=cat,
                    reconciled_secondary=sec, is_pass2d_failure_on_evidence=fail, text_favours=r["text_favours"],
                    evidence_grade=r["decision_status"], reconciliation_note=note))
    out_rows.append(row)
print(f"investigator vs reconciled agreement: {sum(1 for i in R if inv[i]['category'] == R[i][0])}/{len(R)}; "
      f"investigator vs coordinator-provisional: {sum(1 for i in R if inv[i]['category'] == prov[i])}/{len(R)}")
print("reconciled categories:", dict(Counter(R[i][0] for i in R)), "| pass2d failures on evidence:", [i for i in R if R[i][2]])

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
record = dict(
    schema_version="pass2d-diagnosis-v1-draft", generated="2026-09-09",
    status="DRAFT - coordinated; awaiting Workflow 2 adversarial verification and Steven's Stop 2 review",
    inputs=dict(worklist_sha256=sha("reports/catalog_checkpoint_pass2d_worklist.json"), corpus_sha256=sha("reports/catalog_audit_replay_corpus.json"),
                snapshot_after_sha256=sha("reports/catalog_audit_replay_snapshot_checkpoint_after.json"), catalog_sha256=sha("tools/issue_catalog_kind_v2.json"),
                decisions_sha256=sha("tools/catalog_migrations/kind_v2_decisions.json"), workflow1_run_id="wf_7a3702b3-779",
                workflow1_agent_tokens=wf["totalTokens"],
                artifacts=[{k: a.get(k) for k in ("property_key", "run_id", "artifact_path", "sha256", "readable")} for a in corpus["artifacts"]]),
    pins_check=dict(catalog_matches_checkpoint=sha("tools/issue_catalog_kind_v2.json") == "787964013d083368403524c610df54cc9858a8f6c1a1e5c9535c7204c1e67a2f",
                    decisions_matches_checkpoint=sha("tools/catalog_migrations/kind_v2_decisions.json") == "4cd0bc073f6fcdab3bde93993ed05a6199645e756e0df4b816230bd047a3b005"),
    coordinator_verifications=ver, rows=out_rows,
    corpus_statistics=res["corpus_stats"]["statistics"], reproduction_check=res["corpus_stats"]["reproduction_check"],
    corpus_new_findings=res["corpus_stats"]["new_findings"],
    null_triage=dict(
        degradation_defect=dict(aggregate=res["nulls_degradation_defect"]["aggregate"], population_note=res["nulls_degradation_defect"]["population_note"],
                                coverage_gap_subjects=res["nulls_degradation_defect"].get("coverage_gap_subjects"), rows=res["nulls_degradation_defect"]["rows"]),
        modernization_sample=dict(aggregate=res["nulls_modernization"]["aggregate"], population_note=res["nulls_modernization"]["population_note"],
                                  coverage_gap_subjects=res["nulls_modernization"].get("coverage_gap_subjects"), rows=res["nulls_modernization"]["rows"])),
    catalog_divergence=dict(items=sorted(res["catalog_divergence"]["items"], key=lambda i: i["severity_rank"]),
                            correlation_note=res["catalog_divergence"]["correlation_note"], summary=res["catalog_divergence"]["summary"]),
)
json.dump(record, open(f"{SP}/pass2d_diagnosis_20260909.draft.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(f"{SP}/pass2d_corpus_stats_script.py", "w", encoding="utf-8").write(res["corpus_stats"]["script_text"])
fit = [r for k in ("nulls_degradation_defect", "nulls_modernization") for r in res[k]["rows"] if r["bucket"] == "fitting_candidate_present"]
print("fitting_candidate_present rows:", len(fit))
for r in fit:
    print(f"  {r['kind'][:4]}/{r['scene_group'][:8]:8s} {r['confidence'][:3]} -> {r.get('named_candidate')}@{r.get('named_candidate_rank')} | {r['observation'][:72]}")
print("pins:", record["pins_check"])
print("written:", f"{SP}/pass2d_diagnosis_20260909.draft.json")
