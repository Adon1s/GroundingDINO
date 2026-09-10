"""Reconcile the 26+10 row Pass 2d worklist against everything established in this investigation.
Four buckets, with naming errors, missing observations and billing consequences kept distinct."""
import json
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
v3 = json.load(open(f"{SP}/pass2d_diagnosis_20260909.v3.json", encoding="utf-8"))

# status: fixed_by_N | accepted_limitation | unresolved_2d | outside_2d
# defect_class: naming (wrong item, work still recorded) | missing_observation (no condition recorded)
#             | none (model was right / nothing lost)
# billing: none | wrong_line | lost | unmeasured
R = {
 1:  ("outside_2d", "naming", "none", "Text tie; Steven's retag: 'perhaps this one is not problematic'. Both items live, same kind/severity/tier. Upstream wording, not selection."),
 2:  ("outside_2d", "naming", "none", "LEAD-ONLY model_judge unit; the 2a text never names vinyl or tearing. Text favours the selected item."),
 3:  ("unresolved_2d", "naming", "none", "Rank-1 passed over for rank 3 twice. Steven (C2 photo): no scuffing at all, paint on the floor, 'discoloration/patchy' was window light. The sentence is partly wrong AND the pick is unsupported. N does not reach it (no instance relation); Rule E rejected. Both items bill; no dollar delta measured."),
 4:  ("outside_2d", "naming", "none", "Steven's retag withdraws the wear claim ('appears fine from the photos'). Style item kind-gated. Bare-noun shortcut mechanism recorded; no selection error left."),
 5:  ("unresolved_2d", "naming", "none", "Human-confirmed wrong subject (C5: floor stained, fixtures not). Billing was already suppressed by dedup_collision, so no dollar consequence. Rule E fixes it but is rejected on net cost; deferred as the narrow wrong-subject follow-up."),
 6:  ("accepted_limitation", "naming", "none", "Same 'tub$' locative mechanism; four bullets collapsed into one already-billed condition, so nothing incremental."),
 7:  ("accepted_limitation", "naming", "none", "Bare 'window' fired; outcome Terra-supported and correct; costs nothing. Only stale pointer in the shipped worklist (better_item names the CAP-007 parent, flagged ABSENT)."),
 8:  ("outside_2d", "naming", "none", "Steven's recorded answer is mechanism_only / claim=misnamed, which SUPPORTS the worklist's naming complaint. Text favours the selected item; the porch item's claim is a concrete slab. Catalog wording, not selection."),
 9:  ("outside_2d", "none", "lost", "labels_v1_1 claim=exact ratifies the selected item's claim. The billing was lost at Terra's cannot_assess, which the worklist already prices. Not a 2d defect."),
 10: ("outside_2d", "none", "none", "The only human verdict (terra_claim_supported) ratifies the OSB selection; the contrary reading is a model. Withdrawn as a 2d failure."),
 11: ("outside_2d", "naming", "wrong_line", "Gold says 'peeling paint and staining'; the 2a text says 'discoloration and patching' - the selecting word was deleted upstream. CAP-014's single human unit under D5. Pricing mode differs (heuristic+hidden vs allowance); dollar delta unmeasured."),
 12: ("outside_2d", "naming", "wrong_line", "'scraped paint' appears in no rendered clause of the peeling item. The note's discriminator (work_item_code) is not rendered to the model."),
 13: ("outside_2d", "none", "none", "Graffiti: no catalog item names it. Coverage, not selection. Shares one condition with row 14."),
 14: ("outside_2d", "naming", "none", "Same condition as row 13; one billing, not two."),
 15: ("outside_2d", "naming", "none", "The note's own body says the retrieval query was weak upstream; axis and body disagree."),
 16: ("unresolved_2d", "naming", "wrong_line", "Steven (C3 photo): staining, deteriorated paint and deteriorated wallboard all true; he judged the SELECTED item better than the worklist's nominee. The item his answer most implies (damaged_drywall_or_cracks, kind defect) was kind-gated out. Worklist axis 'shortcut_or_rank1' is wrong on both halves. Open as a Pass 2c kind-boundary question, not a 2d selection one."),
 17: ("outside_2d", "none", "none", "Nominated item is kind defect against a degradation bullet: never in the searched matrix. The note's 'kind-open' premise is false and its embed_text remedy could not work."),
 18: ("unresolved_2d", "naming", "none", "Steven (C4 photo): staining not clearly visible, no peeling paint, the bathroom is barely in frame; he called the pick 'pretty bad' and the nominee also bad. Two problems: the sentence is unsupported AND no listed candidate fits. Axis wrong on both halves. Unresolved but not fixable inside 2d."),
 19: ("outside_2d", "none", "none", "water_stain_ceiling is kind defect; kind-gated, not 'kind-open'. Same condition as row 18."),
 20: ("outside_2d", "none", "none", "Nominated item kind defect against degradation; 'gate-open' is false. Both items carry DRYWALL_PATCH."),
 21: ("outside_2d", "none", "none", "Steven (D4 photo): the claim is correct, the wall is patched and missing paint. The worklist's 'the 2a text mislabels peeling as patching' is not supported - patching is real. Model was right."),
 22: ("outside_2d", "none", "none", "Drywall item kind-gated; the companion issue is present in the artifact and resolved."),
 23: ("accepted_limitation", "naming", "none", "Reproduces exactly but is unreachable under D4 (item scene-excluded from bathrooms). Now an instance of D4's disclosed 'wallcovering' residual and the CCF-13 double-charge reproduction case."),
 24: ("outside_2d", "naming", "lost", "No exterior painted-trim item exists; the siding item's rendered description names 'or trim' while its claim does not. Catalog wording and coverage. The condition was excluded at Terra."),
 25: ("outside_2d", "naming", "none", "Same pattern; the condition was carried by two true siding bullets and refuted as a whole."),
 26: ("unresolved_2d", "naming", "unmeasured", "Steven (C1 photo): bathroom finishes fits, sink and bathtub are the main dated items, trim is minor; and a section of base trim is MISSING between tub and toilet. The pick named a real but minor thing and missed the primary subject. N does not fire (no instance relation named). Dollar effect unmeasured: work items dedup on max envelope. Also the CAP-013 third property (evidence only)."),
}
D3ROWS = {
 101: ("accepted_limitation", "missing_observation", "none", "Blinds successor at rank 2 (list of 7 after a require_any drop); model declined a two-subject bullet."),
 102: ("accepted_limitation", "missing_observation", "lost", "Neither successor retrieved; Terra's rationale describes brown fabric valances; the nominated blinds item deny-lists 'valance'. A $92-$460 baseline billing lost to a top-8 retrieval miss, not to selection."),
 103: ("accepted_limitation", "missing_observation", "none", "Successor at rank 1; the CAP-007 split collapsed the margin below the gate so the row went to the model, which declined."),
 104: ("accepted_limitation", "missing_observation", "none", "Neither successor retrieved in a flat 0.57-0.63 band."),
 105: ("accepted_limitation", "missing_observation", "none", "Neither successor retrieved; Terra describes curtains; the blinds item deny-lists 'curtain'."),
 106: ("accepted_limitation", "missing_observation", "none", "Successor at rank 2; the cleanest decline of a present item. Its rendered description says 'blinds are not billed as renovation work' - an untested cause."),
 107: ("accepted_limitation", "missing_observation", "none", "Successor at rank 2; two-subject bullet."),
 108: ("accepted_limitation", "missing_observation", "none", "Both successors retrieved at ranks 1-2 and removed by their own deny terms; compare.json already buckets this reaching_neither."),
 109: ("accepted_limitation", "missing_observation", "none", "Successor at rank 5; Terra describes sheer curtains; the blinds item deny-lists 'sheer'."),
 110: ("accepted_limitation", "missing_observation", "none", "Successor at rank 4; R9 re-pointed this row to a naming question."),
}
rows = []
for r in v3["rows"]:
    i = r["worklist_index"]
    st, dc, bill, note = (R.get(i) or D3ROWS.get(i))
    rows.append(dict(worklist_index=i, cap=r["cap"], case_ref=r["case_ref"], observation=r.get("observation"),
                     status=st, defect_class=dc, billing_consequence=bill, note=note,
                     photo_reviewed=r["worklist_index"] in (26, 3, 16, 18, 5) or r["worklist_index"] in (21,),
                     evidence_grade=r["evidence_grade"], reconciled_category=r["reconciled_category"]))
from collections import Counter
out = dict(schema_version="pass2d-worklist-reconciliation-v1", generated="2026-09-10",
           what_this_is="The 26+10 row worklist reconciled against the Phase A record, Steven's twelve photo judgments, and the S/N experiments. Statuses are dispositions of the WORKLIST ROW, not of any catalog decision; every recorded decision is preserved.",
           legend=dict(status=dict(fixed_by_N="the candidate prompt change resolves it",
                                   accepted_limitation="already accepted by a recorded decision (D3, D4, CAP-007) or costs nothing",
                                   unresolved_2d="a real defect that no authorized change resolves",
                                   outside_2d="the defect is upstream wording, catalog wording/coverage, or the Pass 2c kind boundary"),
                       defect_class=dict(naming="a wrong or imprecise item where work was still recorded",
                                         missing_observation="no condition was recorded at all",
                                         none="nothing was lost; the model's pick stands"),
                       billing_consequence=dict(none="no dollar effect", wrong_line="billed under a different line or pricing mode",
                                                lost="a billing that did not happen", unmeasured="not measurable offline")),
           totals=dict(status=dict(Counter(r["status"] for r in rows)),
                       defect_class=dict(Counter(r["defect_class"] for r in rows)),
                       billing=dict(Counter(r["billing_consequence"] for r in rows))),
           rows=rows)
json.dump(out, open(f"{SP}/pass2d_worklist_reconciliation.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(out["totals"], indent=1))
for st in ("fixed_by_N", "unresolved_2d", "accepted_limitation", "outside_2d"):
    ids = [r["worklist_index"] for r in rows if r["status"] == st]
    print(f"{st:22s} {len(ids):2d}  rows {ids}")
