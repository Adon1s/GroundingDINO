# Catalog audit — follow-up register

Every open item the program has produced, in one index, generated from the pinned artifacts. Each row says where its
evidence lives so it can be picked up without re-deriving anything. **This is an index, not a decision record:**
nothing here is approved, and nothing here authorizes work.

**Status: open through the human gate (2026-09-06).** Sessions 4, 5 and 6 append their own items as they complete;
regenerate with `scratchpad/gate_work/lead/build_follow_up_register.py` or extend by hand.

Suggested order of attack, per Steven (2026-09-06): finish Sessions 4-6, then work sections 2-6 of this register,
then section 1 (the Pass 2d worklist) as the next optimization target.

## Source artifacts (hash-pinned)

| Artifact | sha256 |
|---|---|
| `reports/catalog_audit_proposals.json` | `1411352c42e9d3a555c3a1561470323da2234447badc2869c9874c3e8fa3ef99` |
| `reports/catalog_audit_adversarial_review.json` | `d1149875399a9f30c6cd56e924731c7e207d90b603cb248f402c692c3438b5c4` |
| `reports/catalog_audit_gate_decisions.json` | `d89b2bf1149738d9b3e2d5a9222a5b4e163e2cf97dbb0c9f6396d595b849c980` |
| `reports/catalog_audit_redraft.json` | `449f86170dd4b01916988b2000b900e560403bda12fb93cc7e7d7d27bb31811b` |
| `reports/catalog_audit_renewed_review.json` | `aa12e42a0205c0ede8c01362e15f3c16e47cffd9b37de8dd468e70400e8b538b` |
| `reports/catalog_audit_live_check_results.json` | `de47d13f1dfa5802e885027520ae318ab6a6166e8b609115eb2bdb60eaec74e1` |
| `reports/catalog_audit_approvals.json` | `a725c89af131e94b04867f4f5a226bda495278a8ba28fdd7162ed6188d76c871` |

Rendered companions: `docs/PROPOSAL_catalog_audit_20260901.md`, `docs/REVIEW_catalog_audit_adversarial_20260905.md`.
Program docs: `docs/catalog_audit_program/` (00 context, session briefs, handoffs, and the errata appended to
`HANDOFF_SESSION_3.md`).

## 1. Pass 2d selection worklist — the last thing you plan to work on

26 recorded cases in 16 bounded confusion families: an adequate item was reachable at retrieval and Pass 2d
selected another. The program's standing direction is to review these in families, never as one omnibus prompt change.
Full rows with rank, note and the frozen candidate context are in `judgment.pass_2d_follow_ups` per cluster
(`reports/catalog_audit_proposals.json`), rendered as Appendix G of the proposal document.

| Family | Cases | From | Selected -> better item |
|---|---:|---|---|
| `bath_fixture_vs_room_finish` | 2 | CAP-006 | `bath_fixtures_stained_or_worn` -> `vinyl_linoleum_worn_or_stained` (rank 4) |
| `bathroom_dated_finish_subject` | 1 | CAP-017 | `dated_wallpaper_present` -> `outdated_bathroom_finishes` (rank 6) |
| `exterior_painted_trim_vs_siding_field` | 2 | CAP-019 | `exterior_siding_discoloration_fading` -> `exterior_door_paint_failure` (rank 2) |
| `exterior_walking_surface_deck_vs_patio` | 1 | CAP-009 | `patio_or_porch_surface_wear` -> `deck_surface_weathering` (rank 4) |
| `finished_wall_damage_vs_unfinished_assembly` | 1 | CAP-013 | `unfinished_interior_wall_osb_exposed` -> `damaged_drywall_or_cracks` (rank 3) |
| `flooring_dated_style_vs_wear` | 1 | CAP-005 | `vinyl_linoleum_worn_or_stained` -> `older_flooring_style` (rank None) |
| `flooring_material_hard_vs_resilient` | 1 | CAP-004 | `hard_flooring_broken_or_warped` -> `vinyl_linoleum_torn_or_lifted` (rank 2) |
| `flooring_wear_vs_contamination` | 1 | CAP-005 | `hard_flooring_scratched_or_worn` -> `worn_or_stained_flooring` (rank 1) |
| `graffiti_unnamed_in_catalog` | 1 | CAP-014 | `wall_scuffs_marks_or_dents` -> `peeling_or_discolored_paint` (rank 2) |
| `paint_film_vs_wall_surface_wear` | 1 | CAP-001 | `peeling_or_discolored_paint` -> `wall_scuffs_marks_or_dents` (rank 6) |
| `porch_assembly_soffit_vs_structure` | 1 | CAP-011 | `soffit_or_porch_ceiling_failed` -> `damaged_or_unsafe_deck_or_porch` (rank 3) |
| `retrieval_reachability_not_selection` | 2 | CAP-014 | `wall_scuffs_marks_or_dents` -> `damaged_drywall_or_cracks` (rank None) |
| `trim_homograph_plumbing_vs_millwork` | 1 | CAP-021 | `dated_interior_trim` -> `outdated_bathroom_finishes` (rank 1) |
| `unfinished_wall_vs_wall_wear` | 1 | CAP-014 | `wall_scuffs_marks_or_dents` -> `peeling_or_discolored_paint` (rank 2) |
| `wall_surface_scuffs_vs_paint_discoloration` | 8 | CAP-014 | `wall_scuffs_marks_or_dents` -> `peeling_or_discolored_paint` (rank 2) |
| `window_unit_vs_treatment` | 1 | CAP-007 | `dated_or_older_windows` -> `window_blinds_basic_or_plain` (rank 2 at the 3.1 baseline as the removed parent `dated_window_treatment_valance`; re-pointed 2026-09-08 per S4-3, now a naming question: the successor is `no_action`) |

**Re-pointed 2026-09-08 (checkpoint Phase 0, S4-3 closed):** the `window_unit_vs_treatment` row originally named `dated_window_treatment_valance` as the
better item, and the approved CAP-007 split removes that id. It now points at `window_blinds_basic_or_plain` (no_action,
so the follow-up is a naming question rather than a billing one). The rank shown is the frozen 3.1 rank of the parent. Three further rows name
items the ruling-3 sweep may reword; re-read the worklist against the catalog before starting section 1.

Related retrieval hazards recorded elsewhere in this register and worth reading before starting: the bare-token
lexical shortcut (section 4, CCF-12 and the CAP-005/006/021 notes), and the Tier-2 measurements owed by the approved
CAP-007 split (section 3).

## 2. Deferred clusters — evidence or instrument missing

Each carries a promotion recipe in its cluster record. Nothing is implemented for any of them.

| Id | Subject | Why deferred | What would unblock it |
|---|---|---|---|
| CAP-008 | dated_interior_trim, plain-presence billing | wording instrument failed the live check; deferred into the catalog-wide ruling-3 sweep (section 6) | a non-wording instrument, or a removal-only state under a fresh live-check authorization |
| CAP-010 | fence_weathered, leaning fence | only unit is a model lead; diagnosis corrected (the lean WAS captured and billed under the sibling successor) | a human unit on a different property where a leaning fence was lost or double-billed across the two successors |
| CAP-013 | missing or incomplete interior base trim | migration-system gap: the catalog 3.2 marker section names the split parent and is unreachable from the ops surface; estimate_scope is inherited and unauthorable. Q-6 declined schema work | adjudicate the candidate cards, then authorize schema work: rc_ceeb17f8e242, rc_7057c3c171e5, rc_201ccb0c2313, plus rc_57c6faf79c7e added at the gate (A-2) |
| CAP-014 | wall_scuffs_marks_or_dents over-reach | ten model leads, no human unit; mechanism corrected to the Pass 2c kind gate, not embed_text reachability | two human reviews on different properties; and the CCF-14 precedence ruling, which decides whether the cluster is lead-only at all |
| CAP-016 | popcorn_or_acoustic_ceiling_texture, swirled plaster | lead-only; the swirl condition was in fact Terra-supported and billed under a shared work code, so the open question is naming, not a lost condition | a ruling on whether a decorative troweled swirl is an era tell or mere presence (ruling 3/4 question) |
| CAP-018 | brick_weathered_or_discolored, stone billed as brick | one human unit on one property; Terra also enforced the 'intact mortar' qualifier, so a subject widening alone would not recover it | a second property, a ruling on whether staining and biological growth on stone near grade is billable, and a qualifier decision |

## 3. Owed by the one approved change (CAP-007 window-treatment split)

Conditions attached to the approval. Sessions 5 and 6 own most of these.

1. Session 4 implements exactly these seven ops and nothing else on this entry; Tier 1 must show the generated diff as added [dated_window_valance_or_curtains, window_blinds_basic_or_plain], removed [dated_window_treatment_valance], no other item changed, zero validator errors, byte-identical double generation.
2. Display (A-4): the no_action blinds successor stays visible as drafted (inherited defaultHidden false, display_class marketability); no defaultHidden override. Session 6's baseline arm should report how many no_action blinds conditions render at zero dollars (about 28 evidence-era conditions move).
3. Curtain hardware (A-5): 'The curtain rods are dated.' (oc1_649033275db36ca2) keeps billing on the fabric successor for now; its Terra verdict under the new claim is a Session 6 observation and the hardware-subject question is recorded for the catalog-wide sweep.
4. Improvised coverings (A-6): the one currently billed makeshift-covering condition (oc1_c839d6e1118a7006, redfin_81000709 photo_029) reaches neither successor by design under Q-2; accepted, undetected rather than no_action.
5. Claim texts (A-7): kept as drafted; tighter states ('basic or plain style, functional'; 'dated style, functional') are an optional future wording change that would need its own check.
6. Session 5 Tier 2 measures: the lexical-shortcut fire rate on blinds observations against the new successor (scene_classifier_passes.py:1096-1104 thresholds); post-split ranks against dated_or_older_windows on 'windows and blinds' bullets; the new shortcut surface on the fabric successor's bare stems ('curtain', 'drape', 'fabric'), several of whose corpus hits are currently owned by other items; the embedding neighbourhood of both successors (no offline store exists).
7. Session 5/6 plan a full replay and an explicit cutover for stored 3.2 artifacts: no runtime alias for the deprecated parent; resumed runs carrying the stale id fail hard; backfill_kind_v2 is a no-op at the target version; both new atomic_claims change the projection fingerprint, so no stored Terra checkpoint survives (CCF-8).
8. The three predicted Terra verdict flips to supported on the presence-only claim (rc_4f4b93c40ef3, rc_ffad088fa5fe, rc_a22a241b3bf0) stay unmeasured; they are unbillable on every path.
9. Withdrawn billings accepted by Q-1 and recorded: rc_7c5b9f9bb3aa (v1.1 claim exact / work warranted) and gold g3 -> oc1_12e3660bf3923583, plus 26 further evidence-era accepted conditions moving to no_action and one reaching neither successor.

Open questions recorded against the split itself are in `reports/catalog_audit_redraft.json` `cap007.open_questions`; the Tier-2 measurement list is
in `cap007.tier2_questions`.

## 4. Cross-cutting findings (Session 3)

14 findings that span clusters. Full detail and references in `reports/catalog_audit_adversarial_review.json`
`cross_cutting_findings`, rendered in the review document.

| Id | Finding | Affects | Status at the gate |
|---|---|---|---|
| CCF-1 | The Phase C brief the proposals cite is not in the repository | CAP-003, CAP-007, CAP-008, CAP-013 | resolved: citations corrected in the redraft; errata appended to HANDOFF_SESSION_3.md |
| CCF-2 | Cited 'scan' artifacts exist only as prose; Session 3 re-derived the load-bearing ones | CAP-007, CAP-008, CAP-013, CAP-019, CAP-021 | standing constraint: Sessions 4/5 must not cite the scans without a committed derivation |
| CCF-3 | Production Terra has no literal-reading or negation rule, and rarely engages qualifiers | CAP-003, CAP-007, CAP-008, CAP-010, CAP-012, CAP-018 | CONFIRMED AND EXTENDED by the live check: Terra did not engage either tested predicate; see section 5 |
| CCF-4 | Ruling 4 post-dates the Phase B adjudications and two clusters' claims use the phrase it excludes | CAP-003, CAP-008, CAP-016 | resolved: ruling-4 re-adjudication is the precondition recorded on CAP-003 (Q-8) |
| CCF-5 | An unallocated two-property coverage group was left for lead adjudication and never adjudicated in the repo | program-wide | OPEN COVERAGE QUESTION: 42 railing/handrail/baluster bullets across the pinned runs resolved to no item; strongest unadjudicated coverage candidate in the corpus |
| CCF-6 | Bundle records carried by no cluster | CAP-007, CAP-008, CAP-018, CAP-019 | resolved for the two native proposals (rc_470fb7e0624f added; gold g3 located and ledgered); rc_c5afc5b26357 and rc_4c62bf04b442 still carried by no cluster |
| CCF-7 | Splits of the ten marker-listed items are blocked by the catalog 3.2 marker section; other generator hazards | CAP-013 | standing constraint: splits of the ten marker-listed items need schema work first (blocks CAP-013) |
| CCF-8 | Cutover consequences of any split or claim edit are program-wide, not proposal-specific | CAP-007, CAP-008 | carried into the CAP-007 approval conditions: full replay and explicit cutover for stored 3.2 artifacts |
| CCF-10 | Coverage-triage class inconsistencies (no outcome changes) | program-wide | OPEN: a dozen coverage-triage class inconsistencies, plus a second two-property documented question (HVAC ductwork, no item and no trade bucket) |
| CCF-11 | no_change and non_catalog_action are used interchangeably | CAP-001, CAP-002, CAP-003, CAP-004, CAP-005, CAP-006, CAP-009, CAP-011, CAP-015, CAP-017, CAP-019, CAP-021 | resolved: the gate reads no_change and non_catalog_action as one class; `follow_up` distinguishes |
| CCF-12 | Lexical guardrails are brittle to hyphenation and word order, and require_any is unauthorable on carryovers | CAP-014, CAP-016, CAP-017 | OPEN and relevant to section 1: lexical guardrails are brittle to hyphenation and word order; the gate hit this again on the CAP-007 shower guard |
| CCF-13 | The generic and bathroom wallpaper items double-billed one wall | CAP-017 | OPEN: the generic and bathroom wallpaper items double-billed one wall; recipe recorded on CAP-017 |
| CCF-14 | Evidence-class precedence for matched gold annotations needs one program ruling | CAP-014 | OPEN, RULING OWED: whether a matched gold annotation counts as human evidence; decides whether CAP-014 stays lead-only, and applies to seven units |
| CCF-9 | Frozen-attribution discrepancies for the error-attribution record | CAP-006, CAP-011, CAP-012, CAP-019 | OPEN, other owner: four frozen-attribution discrepancies for the error-attribution record, not this program |

## 5. New at the human gate (2026-09-05/06)

Findings this stage produced that no earlier artifact contains.

### The live Terra check

23 single-condition calls, 72,756 tokens, model `gpt-5.6-terra`. Results in `reports/catalog_audit_live_check_results.json`;
the reading that matters is in `reports/catalog_audit_renewed_review.json` `live_check`.

| arm | rc_73d15ca14405 (kill) | rc_9e889631af67 (plain trim) | rc_b642fe69b86a (recovery) |
|---|---|---|---|
| control | supported 2/2 (control_mismatch) | supported 1/1 (control_match) | supported 2/2 (control_mismatch) |
| A | supported 3/3 (blocks) | supported 1/3 (blocks) | supported 3/3 (pass) |
| B | supported 3/3 (blocks) | supported 2/3 (blocks) | supported 3/3 (pass) |

- **Wording has weak purchase on Terra.** two candidate states sharing no content word but 'dated' produced identical verdicts on every card; no accept in twelve calls turned on either predicate. Terra reported that trim was visible.
- **The harness does not reproduce batched production.** the control arm, using the UNCHANGED current claim, returned supported on two of three cards where production recorded unsupported. Any future live check on these cards should consider a batched replay reproducing the stored payload shape.
- **Subject drift is real.** candidate B returned supported on a dark WALL colour in a room whose trim is white, one rationale saying so explicitly.

### Item-level items with no owner yet

- **rc_57c6faf79c7e** (redfin_125779232 photo_003): a fourth claim-exact/work-warranted billing that any future CAP-008 op withdraws, unruled; also a candidate instance of the CAP-013 missing-base-trim concept (its pinned photo-review row records no visible baseboard plus a damaged base board, on the same property as CAP-013 candidate rc_201ccb0c2313). Recorded at the gate under A-2, not adjudicated.
- **paint_refresh_recommended overlap**: its claim is `interior paint color | dated or highly personalized colors`, same kind, same scene groups and the same package-support rooms as dated_interior_trim. Any trim-colour wording duplicates it at the Terra surface. Added to the ruling-3 sweep list.
- **baseboard_wear_scuffs overlap**: rc_6f4590af65a3 is a baseboard_wear_scuffs condition on redfin_80990371 photo_028, the same photo and estimate unit as the CAP-008 recovery card. Terra's rationales mix 'aged' and 'dated' under a grade-only claim, so the style/wear distinction the catalog draws is one the verifier does not.
- **Curtain hardware** (A-5, kept billing for now): `The curtain rods are dated.` (oc1_649033275db36ca2) passes the approved fabric successor's gate while its claim names valances and curtains only. Decide the subject or the term list in the sweep.
- **Billable-side shortcut surface**: the approved fabric successor's bare stems ('curtain', 'drape', 'fabric') create lexical-shortcut triggers the parent never had, on roughly a dozen evidence-era observations, several currently owned by other items. Measure in Session 5 (section 3).

## 6. Rulings owed and policy sweeps

### Catalog-wide ruling-3 sweep (Q-5, A-1) — outside this program

Ruling 3: plain, basic or builder-grade presence is not billable by itself. Items carrying such a branch:

- dated_interior_trim (CAP-008, deferred here)
- cabinets_dated_style
- appliances_dated_or_basic
- dated_bathroom_vanity_light
- landscaping_enhancement_opportunity (already route_override no_action)
- older_ceiling_fan_style
- dated_interior_doors
- dated_window_valance_or_curtains ('The room has basic curtains.' keeps billing on the approved fabric successor)
- paint_refresh_recommended (overlap with any trim-colour wording)

Inputs the sweep must take with it:
- `reports/catalog_audit_live_check_results.json`
- `reports/catalog_audit_renewed_review.json (CAP-008 disposition and live_check block)`
- `reports/catalog_audit_redraft.json cap008 (ledger with four human-approved withdrawals; corrections; overlaps)`

### Other rulings owed

- **CCF-14 evidence precedence** (program-level): does a matched gold annotation count as human evidence? Seven runtime units are affected and it decides CAP-014's classification.
- **Terra calibration for dated paint** (Q-8, declined for now): precondition recorded on CAP-003 is a human re-adjudication of its four units under ruling 4, because two of them are bare neutrals Terra rejected correctly. Calibrating against the old labels could make the system worse.
- **Outbuilding billing** (Q-7, answered yes for now): shed damage bills under the dwelling siding item. The catalog is internally inconsistent about sheds (one exterior item denies 'shed', another exists only for shed paint). Revisit with a second unit.
- **Exterior casing subject** (Q-9, answered casing included): no catalog claim names exterior casing although three items advertise it in retrieval text. A native name/description clarification is a one-property candidate below the bar.
- **Swirled ceiling texture** (CAP-016): era tell or mere presence?
- **Stone near grade** (CAP-018): is staining and biological growth billable?

## 7. Coverage questions (no catalog owner today)

- **Railing and handrail finish wear** (CCF-5): 42 bullets across the 25 pinned runs (interior and exterior, wood and metal, mostly degradation-kinded finish wear) resolved to no item, because the only railing item is the safety defect. None is human-adjudicated, so the bar is none, but this is the strongest unadjudicated coverage candidate in the corpus. Recipe: two adjudicated supports on different properties where a railing finish-wear observation reached Pass 2d and found no home.
- **HVAC ductwork** (CCF-10): deteriorating wrap in an unfinished basement and a bare surface-run duct through a finished room. No catalog item and no trade bucket. Neither observation reached Pass 2d, so the frozen-2a rule means no catalog item could have recovered them.
- **Exterior painted trim, casing, surround or jamb finish failure** (CAP-019): one property; recipe recorded.
- **Boarded gable, attic or vent opening** (CAP-012): one property; a new lower-economics concept, not a field change; severity and scope for the true subject are unknown.
- **Missing or incomplete interior base trim** (CAP-013): see section 2; blocked on schema work as well as evidence.
- **Coverage-triage class inconsistencies** (CCF-10): about a dozen rows where `covered_by_existing_item` points at an item whose claim does not reach the finding, plus three product-quarantined rows that are plain-presence observations that would not bill anyway, so the quarantine's cost is overstated. Correct before quoting the triage counts.

## 8. Other registers to carry forward

- **Pass 2a regression register** (Appendix H of the proposal document, `reports/catalog_audit_proposals.json`): 28 rows. Pass 2a is frozen; these are
  negative controls and regression constraints for any catalog broadening, split or retrieval change, not a remediation target.
- **Per-cluster unresolved questions**: 83 across the 20 clusters, in `judgment.unresolved_questions` (`reports/catalog_audit_proposals.json`).
  Section 6 lifts the ones needing a ruling; the rest are cluster-local and are read when that cluster is picked up.
- **Coverage triage**: 63 rows in `reports/catalog_audit_proposals.json` `coverage_triage`, with the CCF-10 caveats above.
- **Error-attribution corrections** (CCF-9): four discrepancies for `docs/RESULT_error_attribution_20260831.md`; different owner.

## 9. Appended by later sessions

_Session 4 (implementation), Session 5 (deterministic and retrieval validation) and Session 6 (live) append their own
follow-ups here, with the same pointer discipline: what, where the evidence is, what would unblock it._

### Session 4 — approved implementation (2026-09-06)

Candidate `6e67eaa` on `catalog_audit_session4`, baseline `9afe0fa`. CAP-007 implemented exactly; generated diff equals
the manifest's `expected_generated_changes`; double generation byte-identical; validators zero errors. Full detail and
every hash: `docs/catalog_audit_program/HANDOFF_SESSION_4.md`.

| # | Item | Where the evidence is | What would unblock it |
|---|---|---|---|
| S4-1 | The catalog-resolution benchmark scores neither new successor. `test_every_split_successor_is_gold_somewhere` fails: `never scored: [dated_window_valance_or_curtains, window_blinds_basic_or_plain]`. No gold case referenced the removed parent, so nothing regressed — the concept was never covered and now owes two cases. | `tests/test_catalog_resolution_benchmark.py:120-132`; slices under `benchmarks/catalog-resolution-v2/` | Author one gold case per successor. Left to Session 5: its Tier 2 replay produces the observation text, kind and scene evidence to author them from. `benchmarks/` was not in Session 4's authorized files. |
| S4-2 | A stored canary artifact cannot be recomputed offline against the candidate catalog: `ValueError: ... unknown/stale catalog id 'dated_window_treatment_valance'` (`tools/renovation_estimate.py:501`). This is CCF-8 behaving as approved, not a defect. | `tests/test_benchmark_pass2a.py::test_compute_pre2f_totals_force_confirms_packages`; artifact `artifacts_canary/candidate/redfin_11000447/20260808_085243_19c4a4d0` | A replay (Sessions 5/6). Not an alias, and not an edit to a frozen artifact. Any stored 3.2 artifact naming this item is affected, so treat the test as a cutover canary. |
| S4-3 | Section 1's `window_unit_vs_treatment` row still names the removed parent as the better item. Not re-pointed by Session 4 (the register is not an authorized file for it). | Section 1 of this register | Re-point at `window_blinds_basic_or_plain`, which turns the follow-up into a naming question rather than a billing one. |
| S4-4 | `00_OVERALL_CONTEXT.md` §3's decision table is stale against the candidate: it reads 50 reclassified / 19 split; the candidate is 49 / 20, with 44 inherited-economics successors and 129 items. | `00_OVERALL_CONTEXT.md` §3, which already instructs a re-count on decisions-hash drift | A one-line correction whenever that document is next edited. |
| S4-5 | The three audit-renderer tests and both renderers' `--check` fail in the candidate arm by design, because they pin the baseline decisions and catalog hashes. They pass at baseline. | `HANDOFF_SESSION_4.md`, *Validation gate* | Nothing. Do not "fix" them; verify them on the baseline arm. |

Register hash note: this file was `268cd370d3652695f211ce568aed8a28116920ce76e3abe11d8d866e4d453203` when the gate
handoff pinned it. Section 9 invites later sessions to append, so that pin is a snapshot of the gate, not a frozen
artifact; the current value is recorded in `HANDOFF_SESSION_4.md`.

### Session 5 — deterministic and retrieval validation (2026-09-07)

Tier 1 pass (36/36), Tier 2 pass (14/14), no provider call. The implementation reconciles exactly against the approved
diff, reproduced independently from the baseline decisions. The live experiment manifest
(`reports/catalog_audit_live_experiment_manifest.json`, `e1f3b7f1...a42f`) is written with both budget fields null.
Full detail and every hash: `docs/catalog_audit_program/HANDOFF_SESSION_5.md` and
`docs/RESULT_catalog_audit_validation_tier12_20260907.md`.

| # | Item | Where the evidence is | What would unblock it |
|---|---|---|---|
| S5-1 | **The approval's surviving-billings count is optimistic by one condition.** `oc1_93a2e3f3d299a7a1` is recorded among the ten surviving on the fabric successor, but that item is not in the candidate top-8 for either of its observations: on "The blind, floral valance, and plain trim make the room feel older." generic decor language dominates the embedding, and on its other observation the fabric successor reaches rank 2 and is denied by `require_any`. Not an implementation defect — the diff matches the approval exactly. | `reports/catalog_audit_validation_tier12.json` finding `F-REACH-oc1_93a2e3f3d299a7a1`; reachability audit 45/48 | **A ruling from Steven**: accept the accounting as approximate, or reopen the fabric successor's wording. Session 6 measures the live outcome either way. |
| S5-2 | **One blinds observation now resolves deterministically to a billable neighbour.** "The mini blinds feel dated compared with the updated kitchen." shortcuts onto `outdated_kitchen_finishes` at margin 0.03004, four hundred-thousandths above the 0.03 gate; at baseline the parent held rank 1 below the gate so the row reached the LLM. Ruling 5 intends blinds conditions to become no_action; this one bills under a kitchen item. | finding `F-SHORTCUT-fee5271eaff08de5`; the row is a pinned Stage A case in the manifest | Session 6's live Stage A settles what Terra and the disposition actually do with it. If it must not bill, the lever is a deny stem or a margin change, both outside this program. |
| S5-3 | **The blinds successor's presence-only wording collapses the shortcut margin.** Nine of the 46 blinds rows fall below the 0.03 margin gate and now reach the LLM instead of resolving deterministically; 91 rows corpus-wide decide within 0.002 of the threshold. The successor carries no datedness language, so on dated-blinds bullets it sits closer to `dated_or_older_windows` than the parent did. | `tier2_metrics.blinds_margin_collapse`; fire rate 56.5% -> 50.0% (n=46) | Nothing required. Recorded because any future embedding, wording or sidecar-build change flips these rows first; a regression pin would be cheap if shortcut behaviour is ever treated as stable. |
| S5-4 | **Benchmark gold (S4-1) drafted, not applied.** Two schema-valid cases, each grounded in an observation that resolves to the intended successor at rank 1 in the replay, are in the manifest under `benchmark_gold_drafts`. The frozen slices were not edited: they are fingerprinted, committed results pin those fingerprints, and `--gates` refuses a non-frozen slice. | manifest `benchmark_gold_drafts`; `tests/test_catalog_resolution_benchmark.py:120-132` | Session 6 adds both cases to a slice and re-freezes alongside its live benchmark run. `test_every_split_successor_is_gold_somewhere` stays red until then. |
| S5-5 | **CCF-8 inventory (S4-2) complete.** 97 stored canary artifacts across six roots still name the deprecated parent, including all 25 pinned evidence-era runs. Each fails offline recompute against the candidate catalog. No alias was added and no artifact was edited. | manifest `stale_id_inventory` | A replay from frozen Pass 2c, which the manifest requires as a stage and enforces as a stop condition. |

Two arm-isolation facts a later session will otherwise rediscover: 92 tracked files differ between the arms on disk by
line endings only (identical git blobs, none in the retrieval path), so code identity is compared by git blob; and both
arms stamp catalog version `3.2`, so live runs must be identified by `catalog_sha256` and the projection fingerprint,
never by the version string.

### Session 6 — cost-gated live validation, Stage A (2026-09-07)

Evidence: `reports/catalog_audit_live_validation.json` (Part A froze it; Part B set `status: evaluated` and appended
the recommendation). Readable result: `docs/RESULT_catalog_audit_live_validation_20260907.md`. Recommendation:
`inconclusive_repeat_authorization_needed`, scoped to one rubric clause.

| # | Item | Where the evidence is | What would unblock it |
|---|---|---|---|
| S6-1 | **Error migration onto `dated_interior_trim`, billable.** With the parent gone, subject-generic treatment bullets that cannot say "blind" cannot reach the presence-only blinds successor; the LLM picks the nearest billable modernization neighbour at margins under 0.01, and Terra supports "plain trim" when asked about trim. Two candidate-attributable conditions ($76 low each), one from a bullet that never named trim — the redraft's own leak exemplar. The baseline arm's single Pass 2d drift landed on the same item, so 2-vs-1 on n=66 is inside instrument noise. | bundle `aggregates.hijacking`, `per_case` rows for rc_a22a241b3bf0 and oc1_414b89a8d035a952; RESULT §"Error migration" | Any one of: a ruling from Steven that the rate is acceptable (then `publish_candidate_supported` with no further spend); a targeted repeat — Pass 2d ×5 on the 34 LLM-path rows in both arms, local and free, Terra only on new billable landings, under 40k tokens; or a `deny_any` stem on `dated_interior_trim` / a wording change admitting "window treatment" to the blinds successor, both outside this program. |
| S6-2 | **Six moving conditions end undetected rather than displayed at no_action.** Ten candidate rows resolved to no item (LLM path, full candidate list, model chose none): the "window treatment(s)" and "windows and blinds" bullets whose shortcut margin collapsed under S5-3, plus the A-6 makeshift row. Same dollars as no_action; a different frontend envelope; a departure from A-4's "stays visible" intent. | bundle `per_case` where `candidate.pass_2d_status == "unresolved"`; RESULT §"Ten candidate rows" | A ruling on whether undetected is acceptable for presentation-only blinds conditions. If not, the lever is the successor's `support_any` (add "treatment"), which the redraft rejected deliberately. |
| S6-3 | **The fabric successor's tighter claim rejects 3 of the 10 surviving conditions.** Terra unsupported on "dated valance or curtains" for the A-5 curtain-rods condition ("simple dark metal rods, not dated"), a "basic curtains" condition, and a hardware+curtains condition already unsupported at baseline. With S5-1's unreachable condition, the approval's surviving-billings count was optimistic by four in this set. A-5 is answered: hardware does not bill under the fabric claim. | bundle `aggregates.material_findings`, `per_case` for oc1_649033275db36ca2, oc1_72e5dcfe1ef8d6b0, oc1_af9f0d3783467a7c | Nothing required for the candidate; these read as overstated billings correctly withdrawn. Feed A-5's live answer to the catalog-wide hardware-subject sweep. |
| S6-4 | **Terra is unstable on the parent-claim population.** Stored 3.1 run → live baseline, identical claim and photos: 11/66 verdict flips (16.7%), against the 6.4% floor of record; neighbours with identical claims reproduce the floor (20/251, 8.0%). Two controls flipped in the baseline arm alone (rc_64f996e3a886 to supported; rc_ffad088fa5fe to supported). | RESULT §"The instrument's own noise"; bundle `per_case` baseline vs stored verdicts | Any future gate scoring Terra verdicts on this family needs its own same-prompt control arm and a threshold set from it, not from the 6.4% prior. |
| S6-5 | **S5-2 measured: the kitchen shortcut did not fire.** Offline margin 0.03004; live 0.029982. The row reached the LLM, resolved to the blinds successor, supported, no_action, $0. Fragility confirmed at 6×10⁻⁵; outcome benign this run. | bundle `aggregates.material_findings["F-SHORTCUT-fee5271eaff08de5"]` | A regression pin on this row's margin if shortcut behaviour is ever treated as stable (S5-3's suggestion, now with a concrete case). |
| S6-6 | **Operational drift, recorded not fixed.** The manifest's pinned `LM_STUDIO_URL` (`169.254.83.107`) is a stale link-local address; the model server is local and `127.0.0.1:1234` was substituted with the model identity verified. The embeddings sidecar's `ErrorDeviceLost` false-healthy failure recurred and was caught by the real-POST gate. The harness's first smoke attempt recorded transport failures as resolved-to-nothing outcomes; fixed (retry, no cache, fail loud) before any kept measurement. | bundle `deviations`, `invariants.lm_studio`; `PAUSE_SESSION_6_PART_A.md` | Re-pin `LM_STUDIO_URL` in the next manifest; keep the real-POST probe as the readiness check; keep the fail-loud rule for any replay harness. |
| S6-7 | **Catalog 3.2 acceptance still has no fresh Sol data.** Stage A reviews conditions only; the 33 changed candidates sit on 10 canary properties, 3 outside this set. The baseline arm's per-condition verdicts, dispositions and standalone dollars under the 3.2 projection are preserved at `artifacts_canary/catalog_audit_session6_20260907/baseline/`. | RESULT §"Catalog 3.2 acceptance thread" | A whole-property Stage B baseline arm, separately authorized. Stage B is not recommended on the Stage A evidence as it stands (S6-1). |
| S6-8 | **Adjacent, not acted on.** Pass 2d runs on the local Qwen model in production, not the Terra-tier GPT (Steven, 2026-09-07); the manifest and every frozen artifact route it there, so Stage A kept it. Recorded for the Pass 2d follow-up thread. S5-4 (benchmark gold drafts) remains deferred to its own task. | `PAUSE_SESSION_6_PART_A.md`; manifest `benchmark_gold_drafts` | Out of scope here. |

Instrument facts a later session will otherwise rediscover: the manifest's condition ids are 3.1-era and reproduce in
neither 3.2 arm (join by `case_id`/`issue_id`; the bundle carries a crosswalk); production feeds the product-filtered
issue lane, not `estimate_issues_flat`; Terra reviews whole estimate units, so a subset payload measures a different
question; and Pass 2d's LLM path runs at temperature 0.2 with no seed.

#### Session 6 addendum: repeat stage, same day (2026-09-07)

Steven authorized 800,000 further tokens in chat; 572,308 were spent on Pass 2d x5 replicas per row and one
same-prompt Terra replica per Stage A unit, both arms. Evidence appended to the bundle (`repeat`,
`recommendation_after_repeat`); Stage A rows untouched. **Recommendation revised to `publish_candidate_supported`**,
with the strict reading (`candidate_rejected` pending the trim lever) stated for Steven's choice.

| # | Item | Where the evidence is | What would unblock it |
|---|---|---|---|
| S6-1 (updated) | **Resolved as a measurement, open as a decision.** The trim migration is structural: 18/396 candidate resolutions vs 5/396 baseline (the baseline five are one trim-naming bullet, identical in both arms); the two candidate-attributable rows land on trim 6/6 and 5/6, and under the baseline 0/12. No other billable item was reached. Terra supported the trim claim 4/4 on identical prompts. Two conditions, one item, about $76 low each. | bundle `repeat.s6_1`, `repeat.trim_conditions_terra_replica`; RESULT "Repeat stage" | Steven's ruling: publish on the standing priority and carry the trim lever into the catalog-wide sweep, or fix first (`deny_any` `treatment`/`treatments` on `dated_interior_trim`, or admit "window treatment" to the blinds successor). |
| S6-4 (updated) | **Terra same-prompt instability measured directly.** Both arms about 8% on reviewed units (floor reproduced); parent-claim targets 11.4%, candidate targets 7.3%. rc_64f996e3a886 (correct rejection) supported in 3 of 4 identical-prompt calls across arms. | bundle `repeat.terra_same_prompt`, `repeat.controls_terra_replica` | Nothing for the candidate; a control arm remains mandatory for any verdict-scored gate on this family. |
| S6-9 | **A redraft leak-test prediction was wrong.** "The window and blinds are dated." (rc_429ba6851d09) was predicted to reach the blinds successor by shortcut; live it never does (6/6: no item x4, windows x2) because the collapsed margin (S5-3) hands it to the LLM. The hallucination control ends undetected, $0. | bundle `repeat.pass_2d_rows`; RESULT "Controls under replication" | Nothing economically; the same mechanism as S6-2 and it shares that lever. |
| S6-10 | **The candidate's claims are easier for Terra.** Presence-only blinds wording flips at 7.3% vs 11.4% for the parent's "dated valance or treatments" on identical prompts. | bundle `repeat.terra_same_prompt` | Record as a point in the split's favour; nothing required. |

#### Session 6 rulings (2026-09-07, Steven in chat)

| # | Item | Where the evidence is | What would unblock it |
|---|---|---|---|
| S5-1 (closed) | **Accept the missed fabric condition.** The fabric successor's wording stays as approved; `oc1_93a2e3f3d299a7a1` is an accepted miss at $0 (undetected in the candidate arm); the approval's surviving-billings count is accepted as approximate (optimistic by four in this set, with S6-3). | bundle `rulings.s5_1` | Nothing; closed. |
| S6-1 (ruled) | **Fix first.** The strict reading governs: the program's final recommendation on candidate `6e67eaa` is `candidate_rejected` pending the trim lever. The lever was evaluated offline the same day on all 1,969 replay-corpus rows (sidecar only, no provider call) and drafted as **CAP-022**, a `require_any` subject gate on `dated_interior_trim` (`["baseboard", "trim", "casing", "millwork", "molding", "moulding", "wainscot", "chair rail", "crown"]`): 0 shortcut changes, 4 trim-owned rows lost (all subject-mismatched), the S6-1 row's top candidate becomes the generic decor item. | `PROPOSAL_CAP-022_trim_subject_gate.md`; bundle `rulings.s6_1`; `artifacts_canary/catalog_audit_session6_20260907/provenance/cap022_lever_eval/` | Steven's approval of the exact op at the gate; Session 7 lands it (generator gap closure + op + regeneration proof), runs Tier 1/2, writes a new manifest and re-runs Stage A's candidate arm. |
| S6-11 | **The authorable `deny_any` form has a side effect.** `deny_any "window treatment"` on the trim item fixes the S6-1 row but opens a new lexical shortcut onto billable `dated_or_older_windows` for "Window trim and window treatment are dated." (support_phrase_hit, margin 0.0738), a row that names trim; that row is `98f322926d11c81e`, one of the four unpinned parent-owned rows excluded from Stage A, so the Stage A set would not have caught it. Same collapsed-margin mechanism as S5-3/S6-2. | provenance `lever_eval_report.json` `per_lever.deny_window_treatment` | Nothing if CAP-022 takes the `require_any` form; if Steven prefers the deny form, this side effect must be accepted knowingly or the windows item's bare `window` support term revisited (S5-3 lever). |
| S6-12 | **`require_any` is not authorable for a carryover item.** The generator's carryover override set is wording-only (`name`, `description`, `embed_text`, `support_any`, `deny_any`); the renderer classifies a `require_any` op on a carryover as a system gap. Twenty-four catalog items already carry `require_any`, all inherited from v1 or authored on split successors. | `scripts/migrate_catalog_kind_v2.py` `WORDING_OVERRIDE_FIELDS`; `scripts/render_catalog_audit_proposals.py` `classify_op` | A one-line generator change (admit `require_any` to the carryover override set, renamed) with a test, in Session 7; the alternative, authoring it in the v1 catalog, is recorded and not recommended. |

### Session 7 — bounded catalog policy checkpoint, Phase 1 (2026-09-08)

Charter `08_SESSION_BOUNDED_POLICY_CHECKPOINT.md`; packet `docs/DECISION_PACKET_catalog_policy_checkpoint.md` /
`reports/catalog_policy_checkpoint_packet.json` (`a7416ac6…238c`). No semantic change, no provider call; nothing
approved. Housekeeping done: S4-3 re-pointed (section 1), S4-4 corrected in `00_OVERALL_CONTEXT.md`.

| # | Item | Where the evidence is | What would unblock it |
|---|---|---|---|
| S7-1 | **The presence-stem `deny_any` eligibility lever on `dated_interior_trim` is rejected.** It removes trim from 245 rows and 54 trim-owned rows, but 20 of those get a billable neighbour first (wood paneling 8, interior doors 6; one by lexical shortcut at margin 0.096). Plain-trim billing would become plain-paneling and plain-door billing. | packet `trim.eligibility_lever_evaluation_2026_09_08`; `artifacts_canary/catalog_checkpoint_20260908/provenance/trim_eligibility_lever_eval/` | Nothing; recorded so nobody re-proposes it. |
| S7-2 | **Trim decision owed:** `route_override no_action` (recommended; needs carryover `route_override` generator support) vs the CAP-007-style split (native; 97-artifact cutover; residual plain-as-dated billing that Terra cannot separate: kill card rc_73d15ca14405, 3 of 4 sampled photos). CAP-022 alone is not a policy; CAP-008 resolves with the choice. | packet §2; photo readings in the packet JSON | Steven's choice at the gate. |
| S7-3 | **Both validation harnesses are pinned to CAP-007** (`scripts/catalog_audit_validation.py:41-48`, `scripts/catalog_audit_live_validation.py:49-55`, `tests/test_catalog_audit_validation.py:40`). | packet `dispositions.support_request` | Manifest parameterization with the Session 5/6 pins as historical defaults, authorized as support work. |
| S7-4 | **Two families are already non-billing** through the electrical product quarantine (`dated_bathroom_vanity_light`, `older_ceiling_fan_style`: route `excluded_quarantine`, 0 conditions from 69 corpus rows). | packet `families` | Re-enter the sweep only if the quarantine is lifted. |
| S7-5 | **Wallpaper double charge (CCF-13) is deterministic:** dedup keys on (action_code, trade_bucket, unit_policy, unit) and the generic item has no bathroom `package_affinity`; reproduced under 3.2 offline. | packet `wallpaper_check` | Carryover `scene_groups` support (then a one-line exclusion op), or a dedup-key decision outside this program. |
| S7-6 | **Blinds visibility (S6-2) itemized:** ten unresolved rows; five already excluded at baseline; five accepted at baseline ($79–92) now display nothing. Recommendation: accept as presentation-only. | packet `blinds_visibility` | Steven's acceptance, or a frontend need for the blinds no_action envelope. |
| S7-7 | **Railing coverage question re-read:** 14 no-item bullets on 8 properties are dated-railing *style* (modernization), not finish wear; none is a safety defect. | packet `railing_check` | A human-adjudicated finish-wear or unsafe-railing observation with no home, or a product decision on a presence-only railing concept. |
| S7-8 | **Instrument note (rule 3):** this run's embeddings reproduce the Session 5 candidate snapshot on 1,919/1,969 lists; the one shortcut mismatch is the S5-2/S6-5 kitchen row (margin 0.02998 vs 0.03004); first candidate differs on two near-tie rows. Listed, not tolerated by threshold. | packet `trim.eligibility_lever_evaluation_2026_09_08.instrument_self_check_vs_session5_candidate_snapshot` | Nothing; any future snapshot compare lists its boundary crossings the same way. |

#### Session 7 addendum: Steven's trim photo review and the with/without replays (2026-09-08)

Review record `REVIEW_trim_photos_steven_20260908.md`; packet JSON now `99a40d74…1637`; replays frozen under
`artifacts_canary/catalog_checkpoint_20260908/provenance/trim_with_without_replay/`.

| # | Item | Where the evidence is | What would unblock it |
|---|---|---|---|
| S7-9 | **Trim's package role is measured.** Over the 17 replayable pinned artifacts, 44 active trim work items: 27 bill standalone ($2,242–11,212), 17 are absorbed as room-package context. With trim at `no_action`: 77 candidates identical, 15 change only their support list (stored Sol decision payload-incompatible), 9 change shape (7 bedroom packages strong→moderate at unchanged tier/price; 2 living highs −$394/−$526; 1 support-only bedroom package no longer forms). Replay headline deltas are dominated by incompatible stored decisions, not removed work. | packet `trim.corpus_replay_as_is_vs_no_action` | Steven's decision 7: whether a `no_action` condition should still count as package support (new code, own gate). |
| S7-10 | **Kitchen rc_ceeb17f8e242 (missing trim, ruled real work):** represented in both variants only through the kitchen package's `CABINET_REPLACE` driver at $0 standalone; `no_action` removes support membership and the `accepted_for_work` label; missing trim still has no direct-repair owner (CAP-013). Steven has not accepted that loss. | packet `trim.with_without_replay_two_units.kitchen…`; review record | Decision 8: accept driver-only representation, or authorize the CAP-013 authoring extension. |
| S7-11 | **Bathroom rc_57c6faf79c7e (trim claim not confirmed):** the bathroom modernization candidate rebuilds identical without trim (strong, partial_rehab, $5,826–14,566, same driver); the repair package is unchanged; trim's contribution to the costed bathroom is nil; the stored modernization decision is payload-incompatible and needs a fresh Sol decision. | packet `trim.with_without_replay_two_units.bathroom…` | Nothing for the catalog; the cutover replay re-decides the package. |
| S7-12 | **Eight production artifacts cannot be replayed** (no `renovation_estimate_v5` envelope; they predate v5), so every offline package comparison is canary-weighted (17 of 26 pinned runs). | packet `trim.corpus_replay_as_is_vs_no_action.scope` | A production re-run under v5, outside this checkpoint. |

### Session 7 — bounded policy checkpoint, Phases 2 and 3 (2026-09-08)

Steven decided D1-D8 and adjudicated a second CAP-013 card. Two approved ops implemented on
`637a7c0`, `ea34be5`, `7ddd52a` over `6e67eaa`; candidate now `7ddd52a`. Tier 1 24/24, Tier 2 12/12,
no provider call, nothing published. Records: `docs/DECISION_RECORD_catalog_checkpoint_20260908.md`,
`reports/catalog_audit_approvals_v2.json`, `docs/catalog_audit_program/HANDOFF_SESSION_7.md`.

| # | Item | Where the evidence is | What would unblock it |
|---|---|---|---|
| S7-13 | **Trim routes to `no_action` (D1).** 27 standalone billings stop ($2,242-$11,212 over 17 artifacts); trim stays retrievable and displays at $0; the S6-1 migration now costs nothing. CAP-022 declined as unnecessary, not deferred: its gate would move "The built-ins are dated." onto billable `dated_wood_paneling`. | `reports/catalog_checkpoint_tier1_proof.json`, `catalog_checkpoint_package_effect.json` | Nothing for the catalog; live confirmation is owed (S7-17). |
| S7-14 | **Wallpaper bathroom scene exclusion (D4, CCF-13 closed).** One wall, two charges, fixed at retrieval. Measured: the generic item leaves the top-8 of 33 of 267 bathroom rows, 7 first-candidate changes, **0 shortcut changes**, no package candidate changes. | `reports/catalog_checkpoint_tier2.json`, `catalog_checkpoint_per_op_isolation.json` | The 33-row LLM-path residual is accepted unmeasured under D4; a local Pass 2d x5 on those rows would settle it. |
| S7-15 | **The wallpaper fix cannot show up in a replay.** `replay_property` reuses stored `observed_conditions` and never re-runs Pass 2d, so the double charge persists in replayed 3.1-era artifacts by construction. Do not read this as the fix failing. | `reports/catalog_checkpoint_package_effect.json` `wallpaper_finding` | The cutover replay from frozen Pass 2c. |
| S7-16 | **The real combined candidate cannot be package-replayed on the pinned artifacts**, because CAP-007 removed the parent every one of them names (CCF-8/S4-2). Pruning those conditions to force a replay orphans `terra_calls` and contaminates support lists; it was tried and rejected. | handoff finding 2 | The cutover replay; not authorized here. |
| S7-17 | **No live confirmation.** Trim's $0 display is derived from Session 6 stored verdicts against the new route, not observed. | `reports/catalog_audit_live_experiment_manifest_v2.json` (budgets null, three live questions) | Steven authorizes a budget. |
| S7-18 | **CAP-013 evidence bar MET, still schema-blocked.** Steven adjudicated `rc_7057c3c171e5` (redfin_10952874 photos 018/027) as missing base trim and real work, a second property. Q-6 stands; nothing implemented. | decision record, CAP-013 section | A subject-breadth decision (wall base trim only vs interior trim including cabinet casework) plus authorization for the marker/scope schema work. |
| S7-19 | **Generator carryover set widened to `route_override` and `scene_groups` only.** `require_any` deliberately NOT admitted, so S6-12 stays a recorded gap rather than a speculatively closed one. | `637a7c0`; `tests/test_catalog_kind_v2.py` | A future approved op that actually needs it. |
| S7-20 | **The renderer now has a current authoring surface** for classifying new ops; the frozen bundle's historical surface is untouched and the bundle was not rebuilt. The candidate-arm renderer parity test now names the generator as a third drifted input, which is the bundle correctly refusing a widened surface. | `reports/catalog_authoring_surface_v2.json` | Nothing; by design. |
| S7-21 | **Harness pins are parameterized** by `--manifest`, with the Session 5/6 CAP-007 values as defaults (S7-3 closed). The CAP-007 `tier1` reconciliation is split-specific and was NOT forced onto this candidate; Tier 1 here is a purpose-built 24-check proof. | `scripts/catalog_audit_validation.py`, `reports/catalog_checkpoint_validation_pins.json` | Nothing. |
| S7-22 | **Benchmark gold landed (S4-1/S5-4 closed).** The two CAP-007 successor cases are in a new frozen slice with lineage; dev and holdout are untouched; slice discovery is now dynamic. The slice has never been scored. | `benchmarks/catalog-resolution-v2/cases_checkpoint_20260908.json` | An authorized Pass 2d run to score it. |
| S7-23 | **`package_affinity` on a `no_action` item is dead configuration** and no validator forbids it. Retained deliberately under D7 as the landing pad for a zero-dollar-support mechanism. Do not remove as unused. | `tests/test_catalog_checkpoint_trim_no_action.py` | Decision D7's trigger: acceptance shows package-strength regressions that change a decision. |
| S7-24 | **Cutover planned, not executed.** 97 stored artifacts, six roots, 18 properties. The packet's "20 properties" counts trim-owning properties in the corpus, which includes production runs outside `artifacts_canary`; the 97 file count agrees exactly. | `reports/catalog_checkpoint_affected_artifacts.json` | Authorization for the replay and fresh Sol decisions. |
