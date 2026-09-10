"""Assemble the consolidated Pass 2d diagnosis JSON (v3) from the v1 reconciled rows, the Workflow 2
surviving cores, the B0 re-score, the budget reconciliation, and the photo-review cards. Offline."""
import json, hashlib, os
SP = r"C:/Users/Steven/AppData/Local/Temp/claude/C--Users-Steven-PycharmProjects-realtorvision-backend/6f751f7e-4d60-4702-a546-01b841cf1207/scratchpad"
v1 = json.load(open(f"{SP}/pass2d_diagnosis_20260909.draft.json", encoding="utf-8"))
b0 = json.load(open(f"{SP}/b0_rescore.json", encoding="utf-8"))
wf2 = json.load(open(f"{SP}/wf2_cores.json", encoding="utf-8"))
PHOTOS = "C:/Users/Steven/IntelliJProjects/renointel-prod/public/images/properties"

STATUS = {3: "shortlist_weak", 26: "shortlist_clean", 4: "withdrawn", 5: "withdrawn", 10: "withdrawn"}
NOTES = {
 3: "SHORTLIST (weak). Rank-1 worn_or_stained_flooring passed over for rank 3 on both issues. Steven's retag on rc_eef1daf73e2a is recorded answer mechanism_only with note 'I see stains not scuffs' (prior verdict overstated), which supports the reviewer's mechanism but is a claim-level judgment, not a candidate comparison. Verification: the generic-gate counterfactual fires only on d8fc3b6364d55cc0; the sibling fails the margin gate independently, so lifting the gate would split the condition, not fix it. Goes to photo card C2.",
 4: "WITHDRAWN as a selection error. Steven's retag on rc_9c702893a6ce: answer wording_blocked, note 'The linoleum needs replacing but i was wrong that it appears worn. it appears fine from the photos.' The human withdraws the wear claim; text favours the selected item; the style item was kind-gated. The bare-noun shortcut mechanism ('vinyl') stands as a mechanism fact only.",
 5: "WITHDRAWN as a wrong billing. Work item wk1_0a432a2287bfbd24 is status=suppressed, reason_code=dedup_collision; the active FIXTURES_UPDATE line carries both this item and outdated_bathroom_finishes and the condition aggregates four issues, two of them genuine fixture observations. The 'tub$' locative-noun mechanism stands (margin 0.03012). Goes to photo card C5 on the mechanism question only.",
 7: "Bare 'window' fired (only 'window', not 'windows'); outcome Terra-supported and correct; costs nothing. R9 is satisfied by the shipped file (the successor id appears ten times in the D3 block); the only stale pointer is this row's better_item field, flagged ABSENT and billing_relevant_now false. Cosmetic.",
 8: "NOT a contradiction of the worklist. v1 inverted the source: Steven's recorded answer on rc_3a19f759f2d9 is mechanism_only ('a real problem is there, the claim named it wrong') and his v1.1 re-adjudication is claim=misnamed, work=warranted. That SUPPORTS the worklist's naming complaint. The quoted note reversed only his earlier 'unsupported' work call. Human-confirmed misnaming; stage undetermined (the observation says 'porch', the porch item's claim is a concrete slab, the text favours the selected item).",
 9: "Narrowed. labels_v1_1 claim=exact ratifies the truth of the selected item's claim on a non-comparative axis from the non-blind reask arm; the reviewer never saw the rank-3 alternative. The billing was lost at Terra's cannot_assess, which the worklist already records and prices. Not a contradiction of a decision.",
 10: "WITHDRAWN. The only human verdict on rc_f227f639eb82 (reports/review_verdicts.jsonl) is terra_claim_supported: the human judged the exposed-OSB claim true of the photo. The contrary reading is claude_photo_review_session_20260901, a model. Human evidence ratifies the selection.",
 26: "SHORTLIST (clean). 'The shower trim is old.' - the chosen rank-3 item renders as a millwork package (baseboards, window casing, door casing), a scope that excludes shower hardware; the passed-over rank-1 renders as older tile, vanity or fixtures. No human label on this card (v1 verdict terra_evidence_inconclusive). The economic claim is untested: work items dedup on max envelope, so a correct selection may have merged rather than added. Near-miss: 'old' fires on the rank-1 item and only the 0.72 score floor (0.703) blocked a shortcut. Goes to photo card C1.",
}
N3 = " N-3: window_blinds_basic_or_plain's rendered description ends 'blinds are not billed as renovation work', a plausible untested cause of the decline."
rows = []
for r in v1["rows"]:
    idx = r["worklist_index"]
    r = dict(r)
    r["v1_is_pass2d_failure_on_evidence"] = r.pop("is_pass2d_failure_on_evidence")
    if idx >= 101:
        r["selection_error_status"] = "accepted_limitation_D3"
        if r["reconciled_category"] == "F":
            r["reconciliation_note"] = r["reconciliation_note"] + N3
    else:
        r["selection_error_status"] = STATUS.get(idx, "not_a_selection_error_on_evidence")
    if idx in NOTES:
        r["reconciliation_note"] = NOTES[idx]
    if idx == 4:
        r["reconciled_category"] = "A"; r["reconciled_secondary"] = "D"
    if idx == 10:
        r["reconciled_category"] = "B"
    rows.append(r)

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

budget = dict(
    steven_instruction_verbatim="2.5 million gpt 5.6 terra tokens which i am currently seeing 0 tokens spent. so we are still clear to spend 2.5 million terra tokens if necessary. there is no cap other than that.",
    instruction_date="2026-09-09",
    reconciliation="Steven's instruction is the human authorization that reports/catalog_audit_live_experiment_manifest_v2.json's rule ('No provider call may be made until a human fills these in') requires. It governs. docs/analysis/session8_terra_budget_denominator.md records that 2.5M/day is also the provider's free daily allowance; that is operational context (a single run needing more than 2.5M in one day would throttle and should be split across days), not an authority that overrides the instruction.",
    proposed_manifest_v2_budget_block={"stage_a_authorized_tokens": 2500000, "stage_b_authorized_tokens": None, "authorized_by": "Steven (chat, 2026-09-09)", "authorized_at": "2026-09-09",
                                       "scope": "Pass 2d diagnosis Phase B live experiments, ledgered per run against this figure; split any run that would exceed the provider's 2.5M/day allowance",
                                       "rule": "No provider call may be made until a human fills these in."},
    apply_at="Stop 2, together with the record (a repo edit)",
    terra_spent_to_date=0, ledger=b0["terra_ledger"],
    claude_agent_tokens=dict(workflow1=1256890, workflow2=4792555, note="not counted against the Terra figure"),
)

CARDS = [
 dict(card="C1", worklist_row=26, purpose="Is there a human-adjudicated Pass 2d selection error? (clean shortlist row)",
      property="redfin_11079485", photo="photo_005.jpg", observation="The shower trim is old.", kind="modernization", scene="bathroom",
      pass2d_selected=dict(item="dated_interior_trim", rank=3, path="llm", claim="interior trim: plain, thin, or builder-grade trim package"),
      alternative=dict(item="outdated_bathroom_finishes", rank=1, claim="bathroom finishes: dated tile, vanity, or fixtures"),
      questions=["Q1. In the photo, what is the 'trim' that looks old? (a) plumbing trim: escutcheon, handles, spout, shower arm (b) millwork: baseboard, casing (c) both (d) cannot tell",
                 "Q2. Is 'dated tile, vanity, or fixtures' true of this photo? (yes / no / cannot tell)"],
      decides="Q1=a with Q2=yes establishes the first human-adjudicated Pass 2d selection error in the record and makes row 26 the anchor case for any rendering/prompt experiment. Q1=b or Q2=no closes the row."),
 dict(card="C2", worklist_row=3, purpose="Is there a human-adjudicated Pass 2d selection error? (weak shortlist row; your earlier retag said 'stains not scuffs')",
      property="redfin_10952874", photo="photo_029.jpg", observation="Wood-look plank flooring has discoloration, patchy areas, and scuffs.", kind="degradation", scene="bedroom",
      pass2d_selected=dict(item="hard_flooring_scratched_or_worn", rank=3, path="llm", claim="hard flooring finish: scratches, scuffs, or worn finish on intact flooring"),
      alternative=dict(item="worn_or_stained_flooring", rank=1, claim="flooring: wear, staining, or discoloration without structural concern"),
      questions=["Q1. Which claim is true of this photo? (a) scratches/scuffs/worn finish (b) wear/staining/discoloration (c) both equally (d) neither",
                 "Q2. Is the observation text itself faithful to the photo? (yes / overstated / wrong)"],
      decides="Q1=b confirms the misnaming at human grade and adds a second anchor case; Q1=c means the two items overlap and the lever is catalog wording, not selection."),
 dict(card="C3", worklist_row=16, purpose="CAP-014: selection error, upstream text, or Pass 2c kind? (one of the four 'promotion photos')",
      property="redfin_125970550", photo="photo_018.jpg", observation="Wall surfaces have heavy staining and deteriorated wallboard or paneling.", kind="degradation", scene="utility",
      pass2d_selected=dict(item="wall_scuffs_marks_or_dents", rank=2, path="llm", claim="walls: scuffs, marks, or minor dents from use"),
      alternative=dict(item="peeling_or_discolored_paint", rank=1, claim="interior paint: peeling, bubbling, or visibly aged finish"),
      also_consider=dict(item="damaged_drywall_or_cracks (kind defect, NOT in the list because Pass 2c said degradation)", claim="drywall: cracks, holes, or impact damage"),
      questions=["Q1. Which claim is true of this photo? (a) scuffs/marks/dents from use (b) peeling or aged paint (c) damaged or deteriorated wallboard, i.e. a defect (d) water damage (e) none",
                 "Q2. Is the observation text faithful? (yes / understated / wrong)"],
      decides="Q1=c or d means the right home was kind-gated by Pass 2c and no Pass 2d change could reach it; Q1=b makes it a selection case; Q1=a clears the model. Feeds the CAP-014 reopening-trigger judgment, which is yours."),
 dict(card="C4", worklist_row=18, purpose="CAP-014: the second lower-rank case",
      property="redfin_166147710", photo="photo_038.jpg", observation="Bathroom walls and ceiling are stained.", kind="degradation", scene="bathroom",
      pass2d_selected=dict(item="wall_scuffs_marks_or_dents", rank=3, path="llm", claim="walls: scuffs, marks, or minor dents from use"),
      alternative=dict(item="bathroom_paint_peeling_or_worn", rank=1, claim="bathroom paint: peeling or worn finish WITHOUT moisture evidence"),
      also_consider=dict(item="water_stain_ceiling (kind defect, NOT in the list)", claim="ceiling: water staining indicating moisture intrusion"),
      questions=["Q1. Which claim is true? (a) scuffs/marks/dents (b) peeling/worn paint without moisture (c) moisture staining, i.e. a defect (d) none",
                 "Q2. Is 'stained' the right word for what the photo shows? (yes / no)"],
      decides="Same as C3. If C3 and C4 both return c/d, CAP-014's rows are a Pass 2c kind-boundary question and the trigger is not met by a Pass 2d mechanism."),
 dict(card="C5", worklist_row=5, purpose="Does the bare-noun lexical shortcut assert a wrong subject? (billing already suppressed; mechanism question only)",
      property="redfin_125779232", photo="photo_007.jpg", observation="The floor has discoloration and staining around the tub and vanity.", kind="degradation", scene="bathroom",
      pass2d_selected=dict(item="bath_fixtures_stained_or_worn", rank=1, path="lexical_shortcut on the term 'tub$'", claim="bath fixtures: heavy staining or worn finish, undamaged"),
      alternative=dict(item="worn_or_stained_flooring (rank 2) / vinyl_linoleum_worn_or_stained (rank 4)", rank=2, claim="flooring: wear, staining, or discoloration"),
      questions=["Q1. Which component is deteriorated in the photo? (a) the floor (b) the tub/fixtures (c) both (d) cannot tell",
                 "Q2. Is 'heavy staining or worn finish on the tub/shower/toilet' true of this photo? (yes / no)"],
      decides="Q1=a with Q2=no is a human-confirmed wrong-subject shortcut and is the go signal for B2 (a subject-aware shortcut rule, evaluated offline first). Q1=b/c clears the shortcut here."),
 dict(card="C6", worklist_row=None, purpose="Does the strict null lose a real condition? (strongest 'fitting candidate' decline, high confidence)",
      property="redfin_11185681", photo="photo_024.jpg", observation="The range appears older.", kind="modernization", scene="kitchen",
      pass2d_selected=dict(item=None, rank=None, path="llm (declined)", claim=None),
      alternative=dict(item="appliances_dated_or_basic", rank=4, claim="kitchen appliances: dated style or basic grade, functional"),
      questions=["Q1. Is the range visibly dated or basic in the photo? (yes / no / cannot tell)", "Q2. Should the model have selected the appliances item? (yes / no)"],
      decides="Yes/yes means the 'if none fit, return null' rule lost a real modernization condition; three such answers across C6-C8 justify a prompt-rule arm in B3."),
 dict(card="C7", worklist_row=None, purpose="Does the strict null lose a real condition? (rank-1 candidate declined)",
      property="redfin_25809814", photo="photo_027.jpg", observation="The faucet has an older style.", kind="modernization", scene="bathroom",
      pass2d_selected=dict(item=None, rank=None, path="llm (declined)", claim=None),
      alternative=dict(item="outdated_bathroom_finishes", rank=1, claim="bathroom finishes: dated tile, vanity, or fixtures"),
      questions=["Q1. Is the faucet visibly dated in the photo? (yes / no / cannot tell)", "Q2. Does 'dated tile, vanity, or fixtures' fairly describe what you see? (yes / no)"],
      decides="As C6."),
 dict(card="C8", worklist_row=None, purpose="Does the strict null lose a real condition? (generic bullet; the rank-1 decor item routes excluded_generic at $0)",
      property="redfin_80925528", photo="photo_045.jpg", observation="Kitchen is functional and very dated.", kind="modernization", scene="kitchen",
      pass2d_selected=dict(item=None, rank=None, path="llm (declined)", claim=None),
      alternative=dict(item="dated_overall_decor_style (rank 1) / appliances_dated_or_basic (rank 2) / cabinets_dated_style (rank 7)", rank=1, claim="overall decor: dated finish selections (drop_if_generic=true, routes excluded_generic at $0) / kitchen appliances: dated style or basic grade / cabinets: dated style or basic grade"),
      questions=["Q1. Is there a specific dated element in the photo? (a) cabinets (b) appliances (c) counters/finishes (d) only general decor (e) none",
                 "Q2. Should the model have selected an item, and which? (free text)"],
      decides="Tests whether generic 'very dated' bullets should map to a specific billable item or correctly decline; informs the prompt rule and the drop_if_generic policy, not a catalog edit."),
]
for c in CARDS:
    c["photo_path"] = f"{PHOTOS}/{c['property']}/{c['photo']}"
    c["photo_exists"] = os.path.exists(c["photo_path"])

record = dict(
    schema_version="pass2d-diagnosis-v3", generated="2026-09-09",
    status="Phase A complete: investigated (Workflow 1), adversarially verified (Workflow 2, all 12 findings refuted as stated), corrected offline (B0). Awaiting Stop 2: Steven's photo review (8 cards) and his decision on writing this record to the repo.",
    supersedes=["pass2d_diagnosis_20260909.draft.json (v1)", "pass2d_diagnosis_20260909.draft.md (v1)", "pass2d_diagnosis_20260909.v2.md"],
    inputs=v1["inputs"], pins_check=dict(catalog_matches_checkpoint=sha("tools/issue_catalog_kind_v2.json") == "787964013d083368403524c610df54cc9858a8f6c1a1e5c9535c7204c1e67a2f",
                                        decisions_matches_checkpoint=sha("tools/catalog_migrations/kind_v2_decisions.json") == "4cd0bc073f6fcdab3bde93993ed05a6199645e756e0df4b816230bd047a3b005"),
    rows=rows,
    selection_error_summary={k: sum(1 for r in rows if r["selection_error_status"] == k) for k in ("shortlist_clean", "shortlist_weak", "withdrawn", "not_a_selection_error_on_evidence", "accepted_limitation_D3")},
    b0_condition_level_rescore=b0,
    verification=wf2,
    corpus_statistics_row_level=v1["corpus_statistics"], reproduction_check=v1["reproduction_check"],
    null_triage=v1["null_triage"], catalog_divergence=v1["catalog_divergence"],
    budget_reconciliation=budget, photo_review_cards=CARDS,
)
json.dump(record, open(f"{SP}/pass2d_diagnosis_20260909.v3.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("selection_error_summary:", record["selection_error_summary"])
print("cards:", [(c["card"], c["photo_exists"]) for c in CARDS])
print("written v3 json")
