# Catalog audit — Session 3 adversarial review (2026-09-05)

Reviewer: `claude_adversarial_review_session_20260904`. Every Session 2 proposal id receives exactly one disposition; corrections are recommendations tied to proposal ids and nothing in the proposals, evidence, decisions, or catalogs was modified. Rendered from `reports/catalog_audit_adversarial_review.json`.

## Summary

| Disposition | Count |
|---|---|
| insufficient_evidence | 1 |
| sustained | 11 |
| sustained_with_conditions | 8 |

| Session 2 outcome | Dispositions |
|---|---|
| deferred_insufficient_evidence | sustained 1, sustained_with_conditions 3 |
| migration_system_gap | sustained_with_conditions 1 |
| native_decisions_proposal | insufficient_evidence 1, sustained_with_conditions 1 |
| no_change | sustained 6, sustained_with_conditions 1 |
| non_catalog_action | sustained 4, sustained_with_conditions 2 |

| Risk category | Dispositions touching it |
|---|---|
| economics | 4 |
| evidence | 10 |
| expressibility | 2 |
| ownership | 12 |
| policy | 10 |
| provenance | 2 |
| retrieval | 7 |
| semantic | 8 |

Reopened clusters: 0. Cross-cutting findings: 14. Questions for the human gate: 9. Validation ok: `True`.

## Dispositions at a glance

| Id | Session 2 outcome | Disposition | Risk | Reopen | Challenge |
|---|---|---|---|---|---|
| CAP-001 | non_catalog_action | sustained | ownership, semantic | False | Tested whether the four Terra refusals of a claim branch Terra was given (visibly aged finish; general wear) are really a catalog wording defect (a disjunctive claim conflating peeling with aged finish) rather than a Terra error, and whether the declined baseboard_wear_scuffs claim-widening op hides a lost condition. |
| CAP-002 | no_change | sustained | ownership, policy | False | Tested whether the shed-siding unit hides a catalog scope defect (dwelling-only subject) rather than a Terra mechanism error, and whether a no_change carrying an open outbuilding-billing ruling should instead be deferred. |
| CAP-003 | non_catalog_action | sustained_with_conditions | evidence, policy, provenance | False | Re-read the four supporting units under ruling 4 (dated style bills on concrete tells only; a bare neutral a buyer would repaint does not), which post-dates the Phase B adjudications and is the exact phrase two of those adjudications use as their reason; tested whether Terra's rejections are the error or the ruling. |
| CAP-004 | no_change | sustained | semantic, evidence | False | Tested Steven's own v1.1 note on rc_128caa6212b8 (collapse the material split to 'hard flooring') against the adjudications, and whether hard_flooring_broken_or_warped's description and embed_text promising laminate/vinyl/tile while its claim says 'hard flooring' is a ruling-1 defect. |
| CAP-005 | no_change | sustained | retrieval, ownership | False | Tested whether the redfin_80916010 miss, resolved by the lexical shortcut on the bare support_any token 'vinyl', is a natively editable catalog retrieval-metadata defect rather than a resolver-design question, and whether no_change is the right label for a cluster whose primary cause is Pass 2d. |
| CAP-006 | no_change | sustained | retrieval, ownership | False | Tested whether bath_fixtures_stained_or_worn's support_any (the location noun 'tub$' is the term that fired the lexical shortcut on the misresolved bullets in rc_6285e9a79e45; 'stained' does not match 'staining' under leading-boundary matching) makes the misresolution catalog-owned, and whether the rc_e9d27cf6e9aa kind-gate discrepancy changes the outcome. |
| CAP-007 | native_decisions_proposal | insufficient_evidence | evidence, semantic, policy, retrieval, economics | False | Attacked on four fronts: whether ruling 5 (blinds must be a no_action condition) extends to visibly deteriorated blinds; whether the two counted gold units evidence the defect the split repairs; whether the drafted successors are coherent under the migration policy and the production Terra prompt; and whether the split's regression accounting is complete. |
| CAP-008 | native_decisions_proposal | sustained_with_conditions | semantic, evidence, economics, policy, provenance | False | Attacked the kill condition (whether the rewritten claim newly accepts heavy period trim in clean condition), the exclusion clause's house-style claim, the withdrawal of three human-approved billings under ruling 3, the legality of the structural-exception bar, and the package and retrieval statements. |
| CAP-009 | no_change | sustained | ownership, semantic | False | Tested whether either porch/patio unit hides a catalog defect (the item's concrete-specific embed_text against a generic claim; the weathered railings and stair stringers that resolved to nothing) and whether the deck-versus-patio choice is a wrong selection. |
| CAP-010 | deferred_insufficient_evidence | sustained_with_conditions | evidence, ownership | False | Tested whether the lead-only fence cluster hides human evidence (any human or gold unit with leaning or failing fence sections), whether its diagnosis (Pass 2a never reported the lean) survives the pinned run artifact, and whether the promotion-recipe correction is right. |
| CAP-011 | non_catalog_action | sustained | ownership, policy | False | Tested the truth conflict (frozen attribution Terra versus Phase B 'panels not visible'), whether damaged_or_unsafe_deck_or_porch's claim covers a porch-roof framing sag seen from the street, and whether rotted_subfloor_or_structural_framing's exterior scene exclusion is a migration-system gap. |
| CAP-012 | non_catalog_action | sustained | expressibility, economics, policy | False | Tested whether the finding (a nailed board over a gable-level opening priced as a severity-4 DOOR_WINDOW_REPLACE assembly) is, by the program's own definition, a migration-system gap (an unauthorable field on an 'unchanged' carryover) rather than a downstream action. |
| CAP-013 | migration_system_gap | sustained_with_conditions | expressibility, evidence, economics | False | Tested the migration-system-gap classification from both sides: whether a legitimate split parent exists (baseboard_wear_scuffs, whose legacy retrieval text conflates wear with missing and incomplete base trim and whose legacy kind was defect), whether the stated economics blocker is right, and whether the coverage-gap evidence is one property or more. |
| CAP-014 | deferred_insufficient_evidence | sustained_with_conditions | evidence, retrieval, ownership | False | Tested whether any human or gold unit (not a lead) shows staining or damage billed under wall_scuffs_marks_or_dents or lost because of it, whether the cluster's reachability claims for the defect-kind wall items survive a pool re-derivation, and whether the deferral's promotion recipe is right. |
| CAP-015 | no_change | sustained | ownership | False | Tested whether the missing drawer front is covered by the defect claim ('broken components or water staining') and whether cabinets_worn_finish's 'on intact cabinets' qualifier hides a claim defect. |
| CAP-016 | deferred_insufficient_evidence | sustained_with_conditions | evidence, policy, retrieval | False | Tested whether the swirled-plaster question has any human or gold unit, whether the expressibility and recipe statements are right, and whether the cluster's statements about Terra and about the drop-ceiling guardrails survive the pinned artifact and the real matcher. |
| CAP-017 | no_change | sustained_with_conditions | ownership, retrieval, policy, semantic | False | Tested billability of the stick-mosaic tile accent wall under rulings 3 and 4, whether vintage_tile_pattern_style's require_any tile gate made the exact item unreachable (a retrieval-metadata cause), and whether the 2d follow-up target is right. |
| CAP-018 | deferred_insufficient_evidence | sustained | evidence, policy, semantic | False | Tested whether the stone-billed-as-brick finding has a second property, whether Terra enforced the 'with intact mortar' qualifier, and whether the deferral's billability question is decisive. |
| CAP-019 | non_catalog_action | sustained_with_conditions | ownership, evidence, semantic, policy | False | Tested whether the door-surround unit is 'right item reachable' (a Pass 2d follow-up) or 'right item unavailable' (a catalog question): the proposal names exterior_door_paint_failure at rank 2 as the better item, but its claim is 'exterior door paint \| worn, chipped, or peeling finish' and the door leaf is sound. |
| CAP-021 | non_catalog_action | sustained | ownership, retrieval | False | Tested whether the millwork item's presence in a bathroom candidate list (bare 'trim' support token plus a bathroom scene group) is a catalog retrieval-metadata defect rather than a selection failure, and whether the shortcut could have fired on it. |

## Dispositions

### CAP-001 — `sustained` (Session 2: `non_catalog_action`)

**Challenge.** Tested whether the four Terra refusals of a claim branch Terra was given (visibly aged finish; general wear) are really a catalog wording defect (a disjunctive claim conflating peeling with aged finish) rather than a Terra error, and whether the declined baseboard_wear_scuffs claim-widening op hides a lost condition.

**Strongest counterargument.** Ruling 2 says a true branch does not rescue a differently named mechanism, so 'visibly aged finish' inside a peeling claim could be read as the catalog packing two mechanisms into one Terra test, which is the shape the 19 existing splits were made to remove.

**Resolution.** Sustained. The characteristic mechanism on the refused units (tired, uneven, aged finish) is a named branch of the claim Terra was shown, so the mapping is broad-but-true and ruling 2's own last clause makes it a Pass 2d specificity question, which Session 2 filed. The one incompleteness-flavoured unit (rc_48ea825e33f8) is adjudicated refuted because paint loss, the item's own mechanism, is visible, so it is a Terra miss and gives no second property to the missing-trim concept. The truth conflict on control rc_924163d5a9c3 is correctly flagged and not leaned on.

**Residual risk.** The two-item disjunction (peeling vs aged finish) remains a claim that Terra applies inconsistently; if a live Terra calibration is commissioned for CAP-003 it should include these two items, otherwise the four lost conditions stay lost. baseboard_wear_scuffs still advertises missing and incomplete trim in retrieval text its claim does not cover (cross-link CAP-013).

Risk categories: ownership, semantic. Photos opened: —.

Evidence inspected: `rc_4cab6e7be02b`; `rc_48ea825e33f8`; `rc_924163d5a9c3`; `runtime:canary:redfin_10952874:20260818_231243_cb71fce9:oc1_b10d0b1346b9e17a`; `item:peeling_or_discolored_paint`; `item:baseboard_wear_scuffs`; `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `docs/catalog_audit_program/RECONCILIATION_SESSION_2_PHASE_B.md:147-157`; `tools/renovation_architecture/terra_review.py:33-51`

### CAP-002 — `sustained` (Session 2: `no_change`)

**Challenge.** Tested whether the shed-siding unit hides a catalog scope defect (dwelling-only subject) rather than a Terra mechanism error, and whether a no_change carrying an open outbuilding-billing ruling should instead be deferred.

**Strongest counterargument.** The claim's subject 'exterior siding or trim' silently absorbs outbuilding damage into a severity-3 exterior_repair package driver, and only a ruling can say whether that is intended; a cluster that depends on an unmade ruling looks like a deferral, not a no_change.

**Resolution.** Sustained. The only unit is adjudicated refuted on the mechanism (bent, creased, missing panels satisfy the claim as written), so there is no catalog-owned loss to act on and no_change is the correct label for the catalog. The outbuilding question is a product-policy ruling, recorded as a gate question; it does not change the catalog outcome because neither answer is natively authorable on this narrowed carryover (severity, scope and package_affinity are economic or structural fields).

**Residual risk.** If Steven rules outbuilding damage non-billable under the dwelling item, the coherent native fix is a deny_any 'shed' override on damaged_or_rotted_siding_or_trim: exterior_door_paint_failure already carries deny_any ['shed'] and shed_exterior_paint_failure exists, so the catalog is internally inconsistent about sheds, and the sibling bullet 'The outbuilding door is missing or removed.' on the same photo resolved to nothing. Neither is evidenced by a supporting human unit, so the cluster would need reopening on the ruling plus a second unit.

Risk categories: ownership, policy. Photos opened: —.

Evidence inspected: `runtime:canary:redfin_125970550:20260817_235322_005806b2:oc1_1650c027b1d144a2`; `item:damaged_or_rotted_siding_or_trim`; `item:shed_exterior_paint_failure`; `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`

### CAP-003 — `sustained_with_conditions` (Session 2: `non_catalog_action`)

**Challenge.** Re-read the four supporting units under ruling 4 (dated style bills on concrete tells only; a bare neutral a buyer would repaint does not), which post-dates the Phase B adjudications and is the exact phrase two of those adjudications use as their reason; tested whether Terra's rejections are the error or the ruling.

**Strongest counterargument.** All four units carry v1.1 human truth claim exact / work warranted on four different properties, the claim already says 'dated', and Terra answered on the 'highly personalized' limb without ever testing 'dated'; on that record a Terra calibration is the smallest lever and the catalog is rightly untouched.

**Resolution.** Sustained as a non-catalog outcome, with conditions on the recommended action. A blind photo reader asked the ruling-4 question with no proposal context called redfin_10952874 photo_015 and redfin_10735912 photo_003 bare intact neutrals with no era tell, redfin_80877597 photo_002 an identifiable era finish (builder tan with a popcorn ceiling), and redfin_80990371 intact cream walls with taupe trim; the lead's own reading of photos 016/017 agrees that the paint is largely intact, so that unit rests on an era-palette reading, not on the 'worn paint and patches' the Phase B reason records. The qualifier census over the frozen cases shows Terra rejected 6 of 8 paint_refresh_recommended claims with 'neutral, not clearly dated', which is ruling 4 applied. So at most two of the four units carry a ruling-4 tell, the claim_under_test ('a dated neutral a buyer would repaint') is the phrase ruling 4 excludes, and a Terra calibration built on the four units as adjudicated would push Terra toward billing bare neutrals.

**Conditions.**
- Before any Terra calibration is commissioned, the four units are re-adjudicated by a human against ruling 4 as recorded (era-specific finish or visible deterioration), and only units that carry a tell count as calibration targets.
- The non-catalog action is restated as 'calibrate Terra to ruling-4 tells', not 'make Terra accept dated neutrals'; the two bare-neutral rejections (redfin_10952874 photo_015, redfin_10735912 photo_003) are treated as correct under ruling 4.
- The truth conflict on control rc_4abeff09724f (v1.1 supported vs Phase B refuted) is resolved before the calibration is scored.

**Residual risk.** The two era-palette readings (builder tan; cream with taupe trim) are convention-sensitive and sit inside the 80% human self-consistency band; a calibration that fires on them will also fire on many ordinary beige rooms. The parenthetical example the proposals attribute to ruling 4 ('builder beige with taupe trim and worn paint, parquet, popcorn ceiling') is not in the recorded ruling and should not be treated as Steven's words.

Risk categories: evidence, policy, provenance. Photos opened: —.

Evidence inspected: `rc_bcb50df30b68`; `rc_8df38cb68fc0`; `rc_c19a185f9547`; `rc_d857c0db9013`; `item:paint_refresh_recommended`; `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `tools/renovation_architecture/terra_review.py:33-51`; `photo:redfin_10952874/photo_015.jpg`; `photo:redfin_80877597/photo_002.jpg`; `photo:redfin_80990371/photo_016.jpg`; `photo:redfin_80990371/photo_017.jpg`; `photo:redfin_10735912/photo_003.jpg`

### CAP-004 — `sustained` (Session 2: `no_change`)

**Challenge.** Tested Steven's own v1.1 note on rc_128caa6212b8 (collapse the material split to 'hard flooring') against the adjudications, and whether hard_flooring_broken_or_warped's description and embed_text promising laminate/vinyl/tile while its claim says 'hard flooring' is a ruling-1 defect.

**Strongest counterargument.** The human who labelled the flooring cases said the catalog should stop distinguishing hardwood, LVP and linoleum, which is direct human input that the material commitments are the wrong axis.

**Resolution.** Sustained. On the decidable cases the material commitment was correct and Terra, not the catalog, failed (rc_d60c090abd2c: sheet vinyl worn through, Terra denied it); the ambiguous LVP-versus-laminate case is adjudicated unclear and cannot ground anything. The description/embed divergence on hard_flooring_broken_or_warped is real but benign on this record (its one firing was supported and human-agreed). Steven's note is a policy preference recorded as a gate question, not evidence that a material commitment lost a billing.

**Residual risk.** Both vinyl successors have zero reviewed positive uses or correct rejections, so any future edit to them starts with no control family; Terra's inconsistency about what 'hard' excludes remains untested.

Risk categories: semantic, evidence. Photos opened: —.

Evidence inspected: `rc_128caa6212b8`; `rc_d60c090abd2c`; `runtime:canary:redfin_11000447:20260817_224602_d422d7ae:oc1_476cd4f80565d162`; `item:vinyl_linoleum_torn_or_lifted`; `item:hard_flooring_broken_or_warped`; `docs/catalog_audit_program/RECONCILIATION_SESSION_2_PHASE_B.md:77-87`

### CAP-005 — `sustained` (Session 2: `no_change`)

**Challenge.** Tested whether the redfin_80916010 miss, resolved by the lexical shortcut on the bare support_any token 'vinyl', is a natively editable catalog retrieval-metadata defect rather than a resolver-design question, and whether no_change is the right label for a cluster whose primary cause is Pass 2d.

**Strongest counterargument.** support_any is a catalog field and the shortcut fired on it, so the catalog authored the token that bypassed the LLM; removing 'vinyl' from support_any is native on a split successor and would have sent the bullet to the LLM.

**Resolution.** Sustained. The bullet ('Vinyl flooring is worn-looking and visually dated.') was kind-filtered into an all-degradation pool, so the style item the human preferred (older_flooring_style, modernization) was unreachable regardless of the token; removing 'vinyl' would only have moved the choice to an LLM over the same wrong-kind pool. The token hazard is catalog-wide (125 of 128 items author bare subject nouns) and is correctly filed as a resolver-design question in the Pass 2d worklist rather than as a per-item edit.

**Residual risk.** The shortcut's contract mismatch (support_any documented as a non-gating hint, used as a resolver) stays live for every item; the kind gate on compound wear-plus-dated bullets is a Pass 2c question nobody owns yet.

Risk categories: retrieval, ownership. Photos opened: —.

Evidence inspected: `rc_9c702893a6ce`; `rc_eef1daf73e2a`; `runtime:production:redfin_80916010:20260825_045644_dda08b6b:oc1_2c8aac3461bdb5b6`; `item:vinyl_linoleum_worn_or_stained`; `item:older_flooring_style`; `tools/scene_classifier_passes.py:1080-1118`; `tools/pipeline_common.py:265-284`

### CAP-006 — `sustained` (Session 2: `no_change`)

**Challenge.** Tested whether bath_fixtures_stained_or_worn's support_any (the location noun 'tub$' is the term that fired the lexical shortcut on the misresolved bullets in rc_6285e9a79e45; 'stained' does not match 'staining' under leading-boundary matching) makes the misresolution catalog-owned, and whether the rc_e9d27cf6e9aa kind-gate discrepancy changes the outcome.

**Strongest counterargument.** Bullets bypassed the LLM on a catalog-authored location noun ('tub$'), and a replay of the shortcut function with that term removed sends both misresolved bullets to the LLM while every reviewed shortcut resolution on the item's positive-use controls still fires; that is a natively authorable narrowing with zero regression on the reviewed record.

**Resolution.** Sustained. No human-approved billing was lost: the exact item billed from a sibling bullet on the same photo in both human units, so the mismatch is naming, not billability, and under the 2026-09-03 direction a reachable-right-item case is a Pass 2d follow-up. The 'tub$' narrowing is evidenced on one property only and is recorded for the Pass 2d worklist; support_any has no retrieval effect (retrieval ranks on embed_text and gates on deny/require only), so the item's rank-1 concentration on bathroom wear bullets is an embedding effect no offline store can test. The rc_e9d27cf6e9aa discrepancy (frozen attribution 2d; pool shows the vanity item kind-gated out) is a correction for the error-attribution record, not this cluster.

**Residual risk.** bath_fixtures_stained_or_worn will keep taking rank 1 on generic bathroom wear bullets until retrieval is measured live; the frozen attribution of rc_e9d27cf6e9aa overstates Pass 2d.

Risk categories: retrieval, ownership. Photos opened: —.

Evidence inspected: `rc_6285e9a79e45`; `rc_e9d27cf6e9aa`; `item:bath_fixtures_stained_or_worn`; `item:vanity_damaged_or_water_stained`; `tools/scene_classifier_passes.py:1080-1118`

### CAP-007 — `insufficient_evidence` (Session 2: `native_decisions_proposal`)

**Challenge.** Attacked on four fronts: whether ruling 5 (blinds must be a no_action condition) extends to visibly deteriorated blinds; whether the two counted gold units evidence the defect the split repairs; whether the drafted successors are coherent under the migration policy and the production Terra prompt; and whether the split's regression accounting is complete.

**Strongest counterargument.** The conflation is real and natively expressible: the parent's retrieval text advertises basic, aged, bent and damaged mini blinds while its Terra claim says 'dated valance or treatments', the shortcut demonstrably recruited blinds-only bullets onto that claim on four properties, Terra split 3-1 on identical facts and billed one, and a split with route_override no_action on the blinds successor is the only native shape that satisfies ruling 5 while leaving valances and curtains billable, with landscaping_enhancement_opportunity as exact precedent.

**Resolution.** Not sustained as drafted, on five findings. (1) Ruling 5 as recorded names the object, not a state, and its prompting case was ordinary intact mini-blinds (rc_a22a241b3bf0); whether it reaches yellowed, crooked, collapsed blinds is Steven's call, and the human/gold record weighs the other way: rc_7c5b9f9bb3aa (v1.1 exact/warranted; the pinned run artifact, hash-verified against the bundle, shows the bullet 'The blinds are old.' resolved by the lexical shortcut on 'blinds' and Terra supporting on yellowed blinds plus an improvised covering), gold g3 (redfin_11000447 photo_003, 'Venetian blinds are damaged/bent', matched to an accepted billing of this item the cluster never lists), and the g4/g7 miss findings all treat deteriorated blinds as billable. The proposal's premise that the split withdraws 'the item's only demonstrated correct billing' is therefore wrong: it withdraws at least two human/gold-backed deteriorated-blinds billings, and an evidence-era census across the 25 pinned runs shows roughly 26 of the item's 39 accepted conditions moving to no_action. (2) The evidence bar is met by units the ops cannot repair: g7 is a Pass 2a loss (no bullet about the valance or sheeting ever existed) and g4 is a Pass 2c kind loss (its bullet was kinded degradation and no treatment item exists at that kind); the over-bill lane the ops act on is grounded only by refutations and a Pass-2a-attributed hallucination card, which under the program's own policy may ground non-native outcomes only. (3) The blinds successor packs wear and damage words ('aged, bent, or unevenly hung'; 'yellowed, bent, damaged') into a modernization-kind claim with ontology_basis dated_style, which would be the first same-kind split in the decisions file and contradicts the policy that wear and damage are degradation or defect; every existing no_action item is presentation_only or discretionary_improvement. Its 'basic' branch is satisfied by any blind, so under the production prompt Terra will support it on nearly every interior photo with blinds. (4) The fabric successor keeps the bare token 'window treatment' in support_any and has no require_any, so blinds-only bullets such as 'Window treatments are dated and basic' (three Terra-supported examples exist in the evidence era) can still reach the billing successor by shortcut or LLM choice; its 'improvised' prong is mostly unreachable (Pass 2c dropped five of six improvised-covering bullets in the pinned corpus, including the one on rc_7c5b9f9bb3aa's own photo) and, by the catalog's own precedents, an occupant's draped cloth is presentation-only. (5) The regression register cannot express 'supported but unbilled', so the expectations for rc_4f4b93c40ef3 and rc_ffad088fa5fe are internally inconsistent with the still_supported reasoning used for rc_429ba6851d09. Ordinary blinds (the four over-bill photos) and deteriorated blinds (redfin_81000709 photo_003: yellow-orange, crooked, partly collapsed; redfin_11000447 photo_005: crooked hanging, yellowing not resolvable) are visually separable, which is what an R2 redraft would need.

**Required modification.** Do not implement the ops as drafted. First obtain Steven's answer to the R1/R2 question with the enlarged stakes stated (at least two human/gold-backed deteriorated-blinds billings plus roughly two dozen presence billings are withdrawn). Under R1 (all blinds no_action): redraft the blinds successor as a presence-only item (claim of the shape 'basic or plain window blinds', ontology_basis presentation_only, no damage words on any surface, support_any and route_override kept); give the fabric successor a require_any (valance, cornice, swag, curtain, drape, drapery, sheer, cloth, sheet, blanket, fabric, window treatment, covering) and make its support_any and the blinds successor's deny_any mirror images, dropping the bare 'window treatment' token from support_any; decide 'improvised' and 'heavy' explicitly (drop them, or record the broadening and its single-rationale basis). Under R2 (deteriorated blinds bill): add a third, degradation-kind successor for yellowed/bent/broken-slat blinds with a work route, put its tells in the basic-blinds successor's deny_any, and note that its recovery of rc_7c5b9f9bb3aa depends on Pass 2c kinding. In both cases: add gold g3 and rc_7c5b9f9bb3aa to the regression register as acknowledged withdrawn billings; restate expected_after for rc_4f4b93c40ef3 and rc_ffad088fa5fe consistently or extend CONTROL_EXPECTATIONS with a route-level value; fix the human_inputs citations (rulings at HANDOFF_SESSION_2.md lines 74-93; no PHASE_C_BRIEF.md); re-run the dry run and return the redraft for adversarial check before the gate.

**Residual risk.** Under either answer the family's real losses stay upstream of the catalog: g7 never produced a treatment bullet, g4 and five redfin_81000709 damage-worded blinds bullets are degradation/defect-kinded and reach no treatment item, and Pass 2c drops most improvised-covering bullets. A redrafted split changes stored 3.2 artifacts' resolutions with no alias table (resumed server checkpoints are not keyed on the catalog and stale ids fail hard; backfill skips artifacts already at the target version). The billed hallucination it removes is a single Terra verdict inside the 6.4% replica noise.

Risk categories: evidence, semantic, policy, retrieval, economics. Photos opened: photo:redfin_81000709/photo_003.jpg, photo:redfin_11000447/photo_006.jpg, photo:redfin_11000447/photo_008.jpg.

Evidence inspected: `rc_7c5b9f9bb3aa`; `rc_429ba6851d09`; `rc_a22a241b3bf0`; `rc_4f4b93c40ef3`; `rc_ffad088fa5fe`; `rc_64f996e3a886`; `gold:redfin_11000447:photo_004.jpg:g7`; `gold:redfin_11000447:photo_005.jpg:g4`; `gold:redfin_11000447:photo_003.jpg:g3`; `item:dated_window_treatment_valance`; `item:dated_or_older_windows`; `item:landscaping_enhancement_opportunity`; `item:mismatched_or_inconsistent_furniture_staging`; `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `docs/catalog_audit_program/02_SESSION_SEMANTIC_AUDIT.md:9-16`; `tools/renovation_architecture/terra_review.py:33-51`; `tools/renovation_architecture/terra_review.py:76-90`; `tools/scene_classifier_passes.py:1080-1118`; `tools/catalog_embeddings.py:439-450`; `tools/catalog_embeddings.py:521-527`; `scripts/migrate_catalog_kind_v2.py:96-129`; `scripts/migrate_catalog_kind_v2.py:68-85`; `tools/renovation_architecture/catalog_projection.py:46-72`; `tools/catalog_validation.py:485-493`; `tools/renovation_architecture/disposition.py:17-46`; `scripts/render_catalog_audit_proposals.py:98`; `tools/publication_gate.py:141-162`; `tools/analyzer_server.py:277-329`; `tools/renovation_architecture/conditions.py:88-102`; `tools/backfill_kind_v2.py:258-276`; `tools/costing.py:210-249`; `artifacts_canary/renovation_session9_20260818/run_1/candidate/redfin_81000709/20260817_225858_d13bd83c/photo_intel_debug.json`; `photo:redfin_81000709/photo_003.jpg`; `photo:redfin_11000447/photo_004.jpg`; `photo:redfin_11000447/photo_005.jpg`; `photo:redfin_10803207/photo_005.jpg`; `photo:redfin_11185681/photo_007.jpg`; `photo:redfin_25809814/photo_023.jpg`; `photo:redfin_80925528/photo_005.jpg`

### CAP-008 — `sustained_with_conditions` (Session 2: `native_decisions_proposal`)

**Challenge.** Attacked the kill condition (whether the rewritten claim newly accepts heavy period trim in clean condition), the exclusion clause's house-style claim, the withdrawal of three human-approved billings under ruling 3, the legality of the structural-exception bar, and the package and retrieval statements.

**Strongest counterargument.** The defect is visible without any evidence unit: an atomic claim whose state is 'plain, thin, or builder-grade trim package' bills plain presence by construction, Steven ruled plain presence non-billable, Terra reads the claim verbatim, and the frozen record shows both failure directions on real photos (a lost dated-trim condition and two hard_false_billed plain-trim billings). Any dated-finish rewrite improves on that, and the proposal already discloses the kill condition, the house-style deviation and the withdrawn billings.

**Resolution.** Sustained in direction, not as drafted, on six findings. (1) Kill condition: it cannot be settled offline and the drafted text is not a reliable guard. Terra receives subject and state verbatim inside a payload that also carries the 2b observations, and its production prompt has no rule for negations or literal reading; for rc_73d15ca14405 those observations are 'The trim is basic and dated.' and 'The chair rail and wainscot-style division are dated.', so three signals point to supported and one clause points away. The qualifier census over 33 frozen cases found Terra engaging a qualifier in 10 rationales and rejecting because of one in only 2, and echoing 'dated' observations as supported 15 of 18 times when the claim's state is a style word, and the observation corpus contains no negation to learn from. Photos 006/044 show a period trim package in a clean, uniform, multi-coat white finish, which the exclusion clause describes exactly and which the drafted 'heavily painted-over' tell also describes, so the state is self-contradictory on the very photos that decide it. (2) House style: the claim that no atomic_claim uses negation is imprecise (five states use 'without' or 'non-' exclusions), but the form is unprecedented (longest state 9 words, no semicolon, no trailing sentence; the drafted state is 33 words), nothing validates it, and 'era-specific OR aged finish' makes age an independent trigger with 'worn painted' and 'heavily painted-over' as wear language on a modernization item, overlapping baseboard_wear_scuffs (degradation, TRIM_REPAIR) at the Terra surface. (3) Withdrawn billings: rc_d0ffde500a92 and rc_9e889631af67 are clean plain-presence billings that ruling 3 supersedes, and the in-program precedent (rc_a22a241b3bf0) supports supersession, so withdrawing them is acceptable if Steven confirms it; but rc_9e889631af67's frames carry heavy multi-coat build-up the drafted text names as a tell, so its predicted flip is unreliable, and rc_ceeb17f8e242 is not a plain-presence billing at all: its v1.1 label note reads 'I see the missing trim now. Technically this is true then', photo_025 shows a wall meeting the floor with no baseboard, and incompleteness is a qualifying need under ruling 3, so that withdrawal is a lost real condition mis-described as a supersession. (4) Bar: the structural exception rests on an item-specific surface divergence, while the program reserves it for a universal structural or schema fact; the renderer checks shape only. The proposal survives instead as an application of ruling 3 to a claim that certifies plain presence, which is a policy ground, and Session 3 records that seven catalog claims carry a sufficient plain/basic/builder-grade branch and only this one is being held to the ruling. (5) The regression ledger omits rc_470fb7e0624f, a currently billed plain-trim record the new claim would flip. (6) The package statement is false in code: support acceptance can create or destroy a package (a lone opportunity driver emits only with a support; two non-ambient supports mint a driverless package); on this evidence the effect is a strength downgrade in two redfin_25809814 bedrooms, not existence.

**Conditions.**
- Replace the drafted state with a positive-only state of at most nine words carrying no age or wear disjunct and no exclusion clause, of the shape {subject 'interior trim', state 'era-specific dated finish on casing, jambs, or baseboards', ontology_basis dated_style}; if an exclusion is wanted, use the in-house 'without ...' or 'intact but ...' form.
- Author name and description overrides in the same decisions entry so no surface still advertises plain, thin or builder-grade trim as the condition (zero retrieval effect because embed_text is the entire embedding input); record the embed_text decision either way.
- Live Terra checks under the final wording, with the frozen observations in the payload, on rc_73d15ca14405 (photos 006/044), rc_b642fe69b86a (photos 022/028/029) and rc_9e889631af67 (photo_007) before implementation; a 'supported' on the first or third, or an 'unsupported' on the second, blocks the op.
- Steven confirms at the gate that rc_9e889631af67 and rc_d0ffde500a92 are withdrawn under ruling 3, and rules separately on rc_ceeb17f8e242 as a missing-trim (incompleteness) loss rather than a supersession.
- Add rc_470fb7e0624f to regression_controls with an expected_after; correct package_route_effects; fix the human_inputs citations (rulings at HANDOFF_SESSION_2.md lines 74-93; no PHASE_C_BRIEF.md; the convention-4 example parenthetical is not in the recorded ruling).
- Re-ground the evidence bar explicitly as a ruling-3 policy application (the divergence fact is item-specific, not universal) and ask Steven whether ruling 3 is to be applied to the other six plain/basic/builder-grade claims; that question is recorded, not acted on, in this program.
- Re-run the in-memory dry run on the final wording (item_diff must remain exactly {dated_interior_trim: [atomic_claim]} plus the name/description overrides, zero validator errors) and return the redraft for adversarial check before the gate.

**Residual risk.** Even with a positive-only state, 'era-specific' sits one word from 'period millwork' under a prompt with no literal-reading rule, so the kill condition can only be bounded by a live check and should be re-checked in the Session 6 baseline arm; one live call is a weak guarantee at 6.4% replica disagreement. Plain-trim bullets keep routing to the item through the untouched embed_text (support_any has no retrieval effect; it feeds only the lexical shortcut, which fired on 'basic builder-grade finishes', and the 2d prompt), each costing a Terra call. Any atomic_claim edit changes the projection fingerprint and invalidates every stored Terra checkpoint, so Session 5/6 must replay rather than reuse. Missing or incomplete interior trim has no catalog owner after the change (cross-link CAP-013).

Risk categories: semantic, evidence, economics, policy, provenance. Photos opened: photo:redfin_11185681/photo_006.jpg, photo:redfin_11185681/photo_044.jpg.

Evidence inspected: `rc_b642fe69b86a`; `rc_73d15ca14405`; `rc_9e889631af67`; `rc_ceeb17f8e242`; `rc_d0ffde500a92`; `rc_57c6faf79c7e`; `rc_022497229bd3`; `rc_a0e75a18620e`; `rc_470fb7e0624f`; `rc_eccd12ababc5`; `runtime:canary:redfin_80990371:20260817_221632_3d4a8269:oc1_c6c4c4017b1ed9dd`; `item:dated_interior_trim`; `item:baseboard_wear_scuffs`; `item:peeling_or_discolored_paint`; `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `docs/catalog_audit_program/00_OVERALL_CONTEXT.md:240-247`; `docs/catalog_audit_program/RECONCILIATION_SESSION_2_PHASE_B.md:124-145`; `tools/renovation_architecture/terra_review.py:33-51`; `tools/renovation_architecture/terra_review.py:76-90`; `tools/renovation_architecture/terra_review.py:143-165`; `scripts/migrate_catalog_kind_v2.py:132-141`; `scripts/migrate_catalog_kind_v2.py:83`; `tools/catalog_validation.py:541-552`; `tools/catalog_embeddings.py:357-382`; `tools/scene_classifier_passes.py:1080-1118`; `scripts/render_catalog_audit_proposals.py:736-745`; `tools/rehab_packages.py:2222-2233`; `tools/costing.py:210-249`; `tools/renovation_architecture/catalog_projection.py:46-72`; `tools/label_schema.py:170-179`; `photo:redfin_11185681/photo_006.jpg`; `photo:redfin_11185681/photo_044.jpg`; `photo:redfin_10949071/photo_007.jpg`; `photo:redfin_11185681/photo_024.jpg`; `photo:redfin_11185681/photo_025.jpg`; `photo:redfin_25809814/photo_020.jpg`; `photo:redfin_80990371/photo_022.jpg`; `photo:redfin_80990371/photo_028.jpg`; `photo:redfin_80990371/photo_029.jpg`

### CAP-009 — `sustained` (Session 2: `no_change`)

**Challenge.** Tested whether either porch/patio unit hides a catalog defect (the item's concrete-specific embed_text against a generic claim; the weathered railings and stair stringers that resolved to nothing) and whether the deck-versus-patio choice is a wrong selection.

**Strongest counterargument.** A materially more specific reachable item (deck_surface_weathering, rank 4) existed on rc_3a19f759f2d9, and the item's embed_text is concrete-specific while its claim is generic, which is the ruling-1 divergence pattern.

**Resolution.** Sustained. Both units are adjudicated refuted: the billed claim ('surface aging, surface wear, or coating breakdown') is literally true of bare grey weathered decking and of a blackened paver patio, Steven's reconciled ruling calls the deck case a Pass 2d specificity question, and the only false element ('uneven') originates in frozen Pass 2a prose. The railings and stringer null resolutions are a one-property coverage note with no adjudication of their own.

**Residual risk.** No degradation-kind item covers weathered exterior wood railings and stairs; a second property would make that a coverage question. The three-photo condition absorbed a Pass 2a observation with no referent (photo_001), which the min_photo_evidence gate cannot catch because it is an economic field.

Risk categories: ownership, semantic. Photos opened: —.

Evidence inspected: `rc_3deb5aaef0f0`; `rc_3a19f759f2d9`; `item:patio_or_porch_surface_wear`; `item:deck_surface_weathering`; `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `docs/catalog_audit_program/RECONCILIATION_SESSION_2_PHASE_B.md:101-111`

### CAP-010 — `sustained_with_conditions` (Session 2: `deferred_insufficient_evidence`)

**Challenge.** Tested whether the lead-only fence cluster hides human evidence (any human or gold unit with leaning or failing fence sections), whether its diagnosis (Pass 2a never reported the lean) survives the pinned run artifact, and whether the promotion-recipe correction is right.

**Strongest counterargument.** The lead reports a lean the frozen queue digests never mention, and fence_weathered's 'structurally sound' qualifier would be contradicted if the lean is real, so a human look at photo_017 could turn this into a claim-qualifier failure.

**Resolution.** The deferral label stands (the only unit is a model lead and no human or gold unit records a leaning fence), but the diagnosis is wrong on the facts, and an Opus refuter that tried to break the hunter's correction could not. The hash-pinned run artifact shows Pass 2a did report the lean ('leaning/poorly supported in places'; 'leaning or stressed by vines'), Pass 2b emitted separate lean bullets on both photos, Pass 2d resolved them to fence_broken_or_leaning, Terra supported that condition ('visibly leaning and bowed') and it was accepted for work beside the fence_weathered condition, whose 'structurally sound' qualifier Terra affirmed in the same call. So the lean was captured and billed under the sibling successor; the factorized 'misnamed' lead is an artefact of reviewing one condition in isolation, the primary cause upstream_pass_2a_observation is unsupported, and the only genuine observation is that Terra affirmed contradictory qualifiers on the same fence. The recipe correction (same photo and condition is method corroboration) is right.

**Conditions.**
- Rewrite the diagnosis: primary cause becomes 'no catalog defect; lean captured by the sibling successor' with the artifact lineage cited, and the promotion recipe becomes a human unit on a different property where a leaning fence was lost or double-billed across the two successors.

**Residual risk.** Terra affirming 'structurally sound' and 'leaning' for one fence is one more instance of qualifier non-enforcement; work_dedup merged the two conditions into one work item, so the exposure is semantic, not dollars.

Risk categories: evidence, ownership. Photos opened: —.

Evidence inspected: `runtime:canary:redfin_10952874:20260818_231243_cb71fce9:oc1_3c1a032f6a249ccb`; `item:fence_weathered`; `item:fence_broken_or_leaning`; `docs/catalog_audit_program/00_OVERALL_CONTEXT.md:240-247`; `artifacts_canary/renovation_session9_20260818/run_1/candidate/redfin_10952874/20260818_231243_cb71fce9/photo_intel_debug.json`

### CAP-011 — `sustained` (Session 2: `non_catalog_action`)

**Challenge.** Tested the truth conflict (frozen attribution Terra versus Phase B 'panels not visible'), whether damaged_or_unsafe_deck_or_porch's claim covers a porch-roof framing sag seen from the street, and whether rotted_subfloor_or_structural_framing's exterior scene exclusion is a migration-system gap.

**Strongest counterargument.** If a porch-roof framing deflection is a walking-surface item's claim only by accident of wording, then no item claims it, and the right label is a one-property coverage question rather than a Pass 2d follow-up.

**Resolution.** Sustained. The selected item's claim is precise and divergence-free, Terra's cannot_assess produced an inspection disposition that surfaced the condition rather than losing it, and a claim-level fit was reachable (rank-2 damaged_or_rotted_siding_or_trim, 'exterior siding or trim \| rot, warping, cracks, or missing sections', fits a sagging fascia line with ragged out-of-plane boards at least as well as the rank-3 deck item the proposal names), which makes this a bounded 2d follow-up under the 2026-09-03 rule. The scene-group exclusion on the structural-framing item is real and unauthorable on a carryover, but that item is the wrong concept here, so it is a gate note rather than a reopening ground. The stage-attribution conflict is a correction for the error-attribution record.

**Residual risk.** Whether a street-view framing sag should be billable as a deck/porch item at all is a ruling only Steven can give; until then cannot_assess-to-inspection is the de facto policy for this class.

Risk categories: ownership, policy. Photos opened: —.

Evidence inspected: `rc_905575ed8a3d`; `runtime:canary:redfin_81000709:20260817_225858_d13bd83c:oc1_69263601050587ce`; `item:soffit_or_porch_ceiling_failed`; `item:damaged_or_unsafe_deck_or_porch`; `item:rotted_subfloor_or_structural_framing`

### CAP-012 — `sustained` (Session 2: `non_catalog_action`)

**Challenge.** Tested whether the finding (a nailed board over a gable-level opening priced as a severity-4 DOOR_WINDOW_REPLACE assembly) is, by the program's own definition, a migration-system gap (an unauthorable field on an 'unchanged' carryover) rather than a downstream action.

**Strongest counterargument.** The overall context says 'fix this item's field X on a carryover' is a migration-system gap under the current schema; severity, scope and route_override are exactly such fields here, so non_catalog_action mislabels an expressibility gap and lets the gate believe no schema work is implicated.

**Resolution.** Sustained. The migration-system-gap reading fails on the program's own definition: severity 4, scope replace and DOOR_WINDOW_REPLACE are correct for the item's own subject (a boarded door or window), so the evidenced fix is not a field change on this carryover but a new lower-economics concept for a boarded gable, attic or vent opening, which is a coverage promotion needing a met bar (one human unit on one property is none) before expressibility even matters. The correct severity and scope for the true subject are unknown, so it is a pricing and policy question and non_catalog_action is right. One nit worth a native fix some day: the claim renders to Terra as 'entry or window opening opening boarded over' because subject and state both end in 'opening'.

**Residual risk.** One property; the item lists all seven scene groups and reaches interior stair-header and electrical-cutout observations, a breadth nobody adjudicated. Heuristic pricing keys on severity and scope, so the mis-billing is a dollar effect only (context, not a scored result).

Risk categories: expressibility, economics, policy. Photos opened: —.

Evidence inspected: `rc_6f8278034b05`; `runtime:canary:redfin_80990371:20260817_221632_3d4a8269:oc1_32771feb31dced8e`; `item:boarded_up_entry_or_window`; `scripts/migrate_catalog_kind_v2.py:83`; `scripts/migrate_catalog_kind_v2.py:68-85`; `tools/costing.py:210-249`

### CAP-013 — `sustained_with_conditions` (Session 2: `migration_system_gap`)

**Challenge.** Tested the migration-system-gap classification from both sides: whether a legitimate split parent exists (baseboard_wear_scuffs, whose legacy retrieval text conflates wear with missing and incomplete base trim and whose legacy kind was defect), whether the stated economics blocker is right, and whether the coverage-gap evidence is one property or more.

**Strongest counterargument.** Every one of the 19 existing splits was justified on the legacy item's semantics (v1 items carry no atomic claim), baseboard_wear_scuffs' legacy text conflates wear with 'missing baseboards, detached trim, visible gaps at wall base, incomplete base trim installation' exactly as damaged_soffit_or_porch_ceiling conflated failure with finish, its legacy kind was defect, and severity, tier, scope and defaultHidden are authorable on a split successor while heuristic pricing keys on severity rather than the inherited work code; on that reading a native split exists and the gap label hides it.

**Resolution.** Sustained, on corrected grounds. The lead's own read-only in-memory probe of that split (two successors, minimal fields) fails in the generator exactly as the mechanical agent found: catalog 3.2's package_routing_decisions.repair_support_when_driven section names baseboard_wear_scuffs and is applied after successor construction, so the deprecated parent is an unknown item, and the Session 2 ops surface cannot edit that section. The same block applies to all ten marker-listed items. With the marker edited out of band the split generates and validates, but the defect successor inherits estimate_scope 'marketability_rehab' with reason 'catalog_override_cosmetic_trim_marketability', economic fields that are never authorable, so a missing-base-trim defect would be scoped as marketability rehab. The gap is therefore real, but its mechanism is the marker section and the inherited estimate scope, not the 'atomic claim must conflate' rule or the severity argument the proposal gives (severity is authorable and drives heuristic pricing). On evidence: the coverage-gap sub-claim is adjudicated on one property (g5 and g10 are both redfin_11000447; photo_006's baseboard is partly uncertain at the base but its raw wall-to-floor edge is not, and photo_008 has no base trim anywhere); branch B is a Pass 2d case; CAP-001's incompleteness-flavoured unit is refuted (paint loss visible). Three further human-annotated cards point at the same concept under other items and have not been adjudicated for it: rc_ceeb17f8e242 (redfin_11185681, label note 'I see the missing trim now'), rc_7057c3c171e5 (redfin_10952874, review note 'Trim is missing in some photos, its not visibly scuffed', Phase B unclear) and rc_201ccb0c2313 (redfin_125779232, Phase B 'the base trim is bare unfinished wood'). The runtime unit's own human label (v1 verdict terra_claim_supported on rc_f227f639eb82) conflicts with the cluster claim and should be disclosed.

**Conditions.**
- Rewrite gap_description to name the two mechanical blockers actually demonstrated (the repair_support_when_driven marker section names the parent and is unreachable from the ops surface; estimate_scope and its reason are inherited and unauthorable), and drop the severity and 'atomic claim must conflate' arguments, so any schema work the gate authorizes is aimed at the right surface.
- State that the coverage-gap sub-claim is adjudicated on one property, and list rc_ceeb17f8e242, rc_7057c3c171e5 and rc_201ccb0c2313 as candidate second properties whose photos need a human adjudication for the missing-or-incomplete base-trim concept; two adjudicated supports on different properties would meet the bar and make a split of baseboard_wear_scuffs the native lever once the marker block is lifted.
- Disclose the v1 label conflict on rc_f227f639eb82 (human verdict terra_claim_supported versus the cluster's 'no OSB visible') in the conflict_note; it does not change the count.

**Residual risk.** The declined OSB retrieval narrowing is native on an 'unchanged' carryover and, because embed_text is the whole embedding input, a name/description/support_any override would have zero retrieval effect while changing what the 2d LLM reads; the abstention survives only because that effect is live-only, not for the reasons stated. Any future split of a marker-listed item needs schema work first, which the gate must authorize separately.

Risk categories: expressibility, evidence, economics. Photos opened: photo:redfin_11000447/photo_006.jpg, photo:redfin_11000447/photo_008.jpg.

Evidence inspected: `gold:redfin_11000447:photo_006.jpg:g5`; `gold:redfin_11000447:photo_008.jpg:g10`; `rc_f227f639eb82`; `rc_48ea825e33f8`; `rc_ceeb17f8e242`; `rc_7057c3c171e5`; `rc_201ccb0c2313`; `item:unfinished_interior_wall_osb_exposed`; `item:baseboard_wear_scuffs`; `item:damaged_drywall_or_cracks`; `scripts/migrate_catalog_kind_v2.py:144-168`; `scripts/migrate_catalog_kind_v2.py:243`; `tools/catalog_migrations/kind_v2_decisions.json:45-53`; `scripts/migrate_catalog_kind_v2.py:96-129`; `scripts/migrate_catalog_kind_v2.py:68-85`; `tools/catalog_validation.py:72-83`; `tools/costing.py:210-249`; `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `photo:redfin_11000447/photo_006.jpg`; `photo:redfin_11000447/photo_008.jpg`

### CAP-014 — `sustained_with_conditions` (Session 2: `deferred_insufficient_evidence`)

**Challenge.** Tested whether any human or gold unit (not a lead) shows staining or damage billed under wall_scuffs_marks_or_dents or lost because of it, whether the cluster's reachability claims for the defect-kind wall items survive a pool re-derivation, and whether the deferral's promotion recipe is right.

**Strongest counterargument.** Ten model leads on nine properties all point at the same over-reach, the item's description, embed_text and support_any promise stains and damage its claim does not test, and its own deny_any shows the author intended water staining out; that is as close as a lead-only cluster gets to a catalog finding.

**Resolution.** The deferral stands: the only human/gold unit is refuted, so the bar is none and the policy forbids a non-deferred outcome on leads, and the seven positive-use controls include stain-flavoured marks the humans approved under this item, so the severity threshold the cluster needs is not located anywhere in the reviewed evidence. But the cluster's mechanism is wrong, and the lead re-derived it: every one of the nine pools for the issues that resolved to wall_scuffs_marks_or_dents is degradation-only, while damaged_drywall_or_cracks, water_stain_ceiling and visible_mold_or_mildew are defect kind, and Pass 2d retrieves exactly the observation's 2c kind with a fail-closed filter. The judgment's statements that those items were 'kind_ok true' and that the absence is an embed_text reachability question are false; the three 'retrieval_reachability_not_selection' follow-ups are Pass 2c kind-assignment cases. The hunter's second proposal, to count matched gold annotation rows as human evidence, was refuted: that is a program-level precedence ruling (recorded as a cross-cutting finding), not a fact this cluster can settle.

**Conditions.**
- Correct alternative_stage_analysis.upstream_kind_wrong, unresolved question 3 and the three 'retrieval_reachability_not_selection' follow-ups to name the Pass 2c kind gate (defect homes unreachable from degradation-kinded bullets) instead of an embed_text reachability question; the recipe (two human reviews on different properties) is unchanged.

**Residual risk.** This is the highest-volume lead-only cluster in the packet and the first candidate for human review if the program buys more adjudication; the deny_any guardrail is lexical and every paraphrase of 'water stain' bypasses it.

Risk categories: evidence, retrieval, ownership. Photos opened: —.

Evidence inspected: `gold:redfin_10806500:photo_005.jpg:g4`; `rc_9e57092c89dd`; `rc_a2c0f92f2c30`; `item:wall_scuffs_marks_or_dents`; `item:damaged_drywall_or_cracks`; `item:water_stain_ceiling`; `item:visible_mold_or_mildew`; `tools/catalog_embeddings.py:439-450`; `tools/scene_classifier_passes.py:485-507`; `tools/catalog_embeddings.py:486-497`

### CAP-015 — `sustained` (Session 2: `no_change`)

**Challenge.** Tested whether the missing drawer front is covered by the defect claim ('broken components or water staining') and whether cabinets_worn_finish's 'on intact cabinets' qualifier hides a claim defect.

**Strongest counterargument.** A missing drawer front is arguably 'missing', not 'broken', and the claim names only broken components or water staining, so Terra's 'intact' reading could be a claim-wording gap.

**Resolution.** Sustained. A detached or missing drawer front leaving an open cavity is a broken cabinet component on any ordinary reading; the human truth and the Phase B adjudication agree the claim was satisfied, so the failure is verification of a satisfied claim and is already attributed to Terra in the frozen record. The 'intact cabinets' qualifier question is lead-only.

**Residual risk.** Both implicated items have zero reviewed positive uses, so no_change is not positive assurance about them.

Risk categories: ownership. Photos opened: —.

Evidence inspected: `rc_eace19540d2d`; `runtime:canary:redfin_125779232:20260817_222351_c069aab3:oc1_b7dbbbb05b827f66`; `item:cabinets_damaged_or_water_stained`; `item:cabinets_worn_finish`

### CAP-016 — `sustained_with_conditions` (Session 2: `deferred_insufficient_evidence`)

**Challenge.** Tested whether the swirled-plaster question has any human or gold unit, whether the expressibility and recipe statements are right, and whether the cluster's statements about Terra and about the drop-ceiling guardrails survive the pinned artifact and the real matcher.

**Strongest counterargument.** The item's embed_text widens it to any heavy applied texture while its claim commits to popcorn/stipple, the divergence pattern the program treats as catalog-owned, and the widening op is dry-run native.

**Resolution.** The deferral stands (lead-only; the recipe correction is right; expressibility is confirmed), but two statements in the cluster are wrong on the record. First, the swirl condition is not unverified: the pinned artifact shows Terra supported it ('prominent heavy swirled stipple texture') and it was accepted for work under CEILING_TEXTURE_UPDATE, so the case is misnamed-but-billed under a shared work code, which is operationally equivalent, not a lost condition. Second, the cluster's account of rc_62d0dc3b9df8 is misleading: re-running the real matcher on 'The room has a suspended acoustic-tile ceiling.' gives zero hits on the popcorn item's deny terms and zero hits on suspended_drop_ceiling's require terms (hyphenation and word order defeat the word-start substring match), so the guardrail never applied and the correct item was hard-excluded by its own require_any. A hunter proposed reopening on that defect; an Opus refuter killed it because no condition was lost (four sibling suspended_drop_ceiling conditions on the same property were Terra-supported and billed DROP_CEILING_TO_DRYWALL). The brittleness is recorded as a cross-cutting finding.

**Conditions.**
- Correct the cluster's Terra statement (the swirl condition was Terra-supported and billed) and reframe the deferral as a naming question on an operationally equivalent, shared-work-code billing; correct the rc_62d0dc3b9df8 account to say the guardrails never fired and the right item was require_any-excluded.

**Residual risk.** The bare support token 'ceiling texture' routes any ceiling-texture bullet to this item by shortcut; unmeasured and unowned. Whether a decorative troweled swirl is an era tell or mere presence is a ruling-3/4 question for Steven that precedes any wording change, and it now decides only naming, not billing.

Risk categories: evidence, policy, retrieval. Photos opened: —.

Evidence inspected: `runtime:canary:redfin_126418713:20260818_221718_ebc48de0:oc1_e6be634a3ab8e0f6`; `rc_62d0dc3b9df8`; `item:popcorn_or_acoustic_ceiling_texture`; `item:suspended_drop_ceiling`; `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `tools/pipeline_common.py:265-284`; `tools/catalog_embeddings.py:439-450`; `artifacts_canary/renovation_session9_20260818/run_1/candidate/redfin_126418713/20260818_221718_ebc48de0/photo_intel_debug.json`

### CAP-017 — `sustained_with_conditions` (Session 2: `no_change`)

**Challenge.** Tested billability of the stick-mosaic tile accent wall under rulings 3 and 4, whether vintage_tile_pattern_style's require_any tile gate made the exact item unreachable (a retrieval-metadata cause), and whether the 2d follow-up target is right.

**Strongest counterargument.** The material-neutral observation could never reach the exact item because of a catalog-authored require_any gate, which is retrieval metadata and therefore catalog-owned under ruling 1.

**Resolution.** Sustained on its own claim: the wallpaper item behaved as a precise object claim should (Terra confirmed 'dated' and the human refuted the object), the reachable alternative (outdated_bathroom_finishes, rank 6) makes it a 2d follow-up, and the require_any gate on the tile item is a design choice that keeps material-specific items from firing on neutral text. One thing the cluster states is wrong: it says dated_bathroom_wallpaper 'correctly stayed out'. The pinned artifact shows the same wall on the same photo was billed twice, by dated_wallpaper_present (this unit) and by dated_bathroom_wallpaper (oc1_5314b32efbbd352e), both Terra-supported and accepted for work; the two items share the work code, the generic item includes the bathroom scene, and neither denies the other. A hunter proposed reopening on that overlap and an Opus refuter killed it: the second condition was never adjudicated, so it cannot ground a non-deferred outcome, and the overlap is evidenced on one property. It is recorded as a cross-cutting finding with a recipe. Whether the accent wall is billable at all is unruled.

**Conditions.**
- Correct the statement that dated_bathroom_wallpaper stayed out, record the co-firing condition oc1_5314b32efbbd352e on the same photo, and carry the overlap to the human gate as a one-property overlapping-ontology observation whose promotion needs the second condition adjudicated and a second property.

**Residual risk.** Terra confirming the state limb and not the subject limb is the same shape as CAP-003; two instances are not yet a calibration finding. The generic and bathroom wallpaper items can double-bill any bathroom wall that both reach.

Risk categories: ownership, retrieval, policy, semantic. Photos opened: —.

Evidence inspected: `rc_45b9d2f6923c`; `runtime:canary:redfin_25809814:20260817_223307_7b336e61:oc1_7a4fa741ab03da30`; `item:dated_wallpaper_present`; `item:dated_bathroom_wallpaper`; `item:vintage_tile_pattern_style`; `item:outdated_bathroom_finishes`; `tools/catalog_embeddings.py:439-450`; `artifacts_canary/renovation_session9_20260818/run_1/candidate/redfin_25809814/20260817_223307_7b336e61/photo_intel_debug.json`; `photo:redfin_25809814/photo_019.jpg`

### CAP-018 — `sustained` (Session 2: `deferred_insufficient_evidence`)

**Challenge.** Tested whether the stone-billed-as-brick finding has a second property, whether Terra enforced the 'with intact mortar' qualifier, and whether the deferral's billability question is decisive.

**Strongest counterargument.** Terra's rationale concedes the substance (heavily stained, weathered masonry) and rejects only on the material and mortar commitments the claim adds; that is the signature of a catalog commitment losing a real condition, and one human unit plus Steven's reconciliation note could be read as enough.

**Resolution.** Sustained. One supporting human unit on one property (the second unit is refuted), and the qualifier census confirms Terra did engage 'intact mortar' on rc_5063066c1a57 as a secondary rejection reason, so widening only the subject would not recover the unit. Whether staining and biological growth on stone near grade is billable under ruling 3 decides the promotion and is Steven's; the recipe (second property, ruling, qualifier decision, control for rc_dff4c3209df1) is complete.

**Residual risk.** The parged and painted masonry issue on the support unit is a third material no brick-or-stone widening reaches; the cornice row at redfin_11000447 is not a masonry-field unit and was rightly not promoted.

Risk categories: evidence, policy, semantic. Photos opened: —.

Evidence inspected: `rc_5063066c1a57`; `rc_dff4c3209df1`; `runtime:canary:redfin_80925528:20260818_220543_1af5ac93:oc1_fd62a1d71743be6b`; `runtime:canary:redfin_11077450:20260818_223200_a4168368:oc1_261de865e0dd299d`; `item:brick_weathered_or_discolored`; `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `docs/catalog_audit_program/RECONCILIATION_SESSION_2_PHASE_B.md:159-169`

### CAP-019 — `sustained_with_conditions` (Session 2: `non_catalog_action`)

**Challenge.** Tested whether the door-surround unit is 'right item reachable' (a Pass 2d follow-up) or 'right item unavailable' (a catalog question): the proposal names exterior_door_paint_failure at rank 2 as the better item, but its claim is 'exterior door paint \| worn, chipped, or peeling finish' and the door leaf is sound.

**Strongest counterargument.** Pass 2d skipped rank 1 and rank 2 to land on a rank-5 siding item on a limestone facade with no siding, on two properties, and the scope direction says never broaden the catalog to compensate for a selection failure; a non-catalog 2d follow-up is the direction's own answer.

**Resolution.** Sustained, with the reasoning corrected and a deferred coverage twin recorded. The lead's first reading, shared by the exterior family hunter, was that the right item was unavailable: the blind reader and the Phase B reviewer agree the surround and jambs are rough, patchy and paint-smeared while the door leaf is sound and there is no siding anywhere, so the rank-2 item's claim ('exterior door paint \| worn, chipped, or peeling finish') names the door's paint, not the casing, and a correct 2d selection would still have handed Terra a claim narrower than the defect. An Opus refuter refuted the relabel on the policy as written, and the lead accepts it: the evidence bar explicitly permits non_catalog_action for a refutation-backed cluster and CAP-019 carries a reviewed refutation with photos (redfin_166147710, a Terra miss on real siding discoloration); the scope direction defines 'reachable' at retrieval (kind and scene), not at claim fit, so treating claim fit as the test would be a policy amendment; Session 2 named this 'the one soft call' and wrote the deferral wording out; and rc_91bb03de4429 (carried by CAP-002) is a casing observation Terra correctly rejected because the casing was intact, which a subject widening would have to clear. What survives is the fact the cluster under-states: no item's claim names exterior casing, surround or jamb finish while three advertise it on retrieval surfaces, and the casing concept has one human unit on one property. That is a coverage question that belongs beside the 2d follow-up, not instead of it.

**Conditions.**
- State in the cluster that exterior_door_paint_failure's claim does not fit the casing subject (the door leaf is sound), so the Pass 2d follow-ups cannot recover the condition, and record the concept 'exterior painted trim, casing, surround or jamb finish failure' as a one-property coverage question with its promotion recipe (a second independent human unit on a different property where casing or surround finish failure was lost for want of a claim).
- Add rc_8afb476fa69f, the only bundle record on exterior_door_paint_failure and a human-approved billing, as a positive-use control.
- Put the subject-scope question to Steven (gate question Q-9): does exterior_door_paint_failure's subject include the casing, surround and jambs, or the door leaf only? A 'leaf only' ruling makes the right item definitionally unavailable and moves the cluster to deferral.

**Residual risk.** One property; three exterior items advertise door frame, casing or trim on their retrieval surfaces while none claims it, so the family-level ontology question the proposal raises remains open and is the shape a second property would promote. The two 2d follow-ups should not be read as the fix: a correct 2d choice would still have reached Terra with a door-paint claim on a sound door.

Risk categories: ownership, evidence, semantic, policy. Photos opened: —.

Evidence inspected: `rc_7b23faa0bd8f`; `rc_a35f1b3a231e`; `rc_8afb476fa69f`; `runtime:production:redfin_10740044:20260821_233648_4cd603bb:oc1_30375086ad2fd9f6`; `runtime:canary:redfin_166147710:20260817_224909_2f979a60:oc1_b2e96c2b1020d48b`; `item:exterior_siding_discoloration_fading`; `item:exterior_door_paint_failure`; `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `docs/catalog_audit_program/RECONCILIATION_SESSION_2_PHASE_B.md:113-122`; `photo:redfin_10740044/photo_001.jpg`; `photo:redfin_166147710/photo_006.jpg`

### CAP-021 — `sustained` (Session 2: `non_catalog_action`)

**Challenge.** Tested whether the millwork item's presence in a bathroom candidate list (bare 'trim' support token plus a bathroom scene group) is a catalog retrieval-metadata defect rather than a selection failure, and whether the shortcut could have fired on it.

**Strongest counterargument.** The homograph ('shower valve trim' versus millwork 'trim') is authored into dated_interior_trim's support_any and scene_groups, both catalog fields, so the catalog put the wrong item in reach.

**Resolution.** Sustained. The exact item ranked first with kind and scene compatible and the LLM chose rank 3 with no shortcut, which is the reachable-right-item case by definition; Terra then rejected the millwork claim correctly, so nothing billed wrongly. The homograph is recorded for the Pass 2d worklist and matters to CAP-008 (whose op leaves the retrieval surface untouched).

**Residual risk.** If the property supplied no other bathroom_modernization driver, the substitution cost a package, which is a dollar effect only.

Risk categories: ownership, retrieval. Photos opened: —.

Evidence inspected: `rc_eccd12ababc5`; `runtime:canary:redfin_11079485:20260817_224118_67317092:oc1_81d91b622cb7287d`; `item:outdated_bathroom_finishes`; `item:dated_interior_trim`; `tools/scene_classifier_passes.py:1080-1118`

## Reopened clusters

None.

## Cross-cutting findings

### CCF-1 — The Phase C brief the proposals cite is not in the repository

About forty judgment fields cite 'PHASE_C_BRIEF.md section 2/3' and several cite 'HANDOFF_SESSION_2.md lines 55-63' for Steven's rulings. No such file exists in the repository and lines 55-63 of the handoff are its hash table; the rulings are recorded at lines 74-93 and the scope direction in the Session 2 brief. Session 3 verified every ruling it relied on against those two files. One consequence: the example parenthetical the proposals attribute to ruling 4 ('builder beige with taupe trim and worn paint, parquet, popcorn ceiling') is not part of the recorded ruling and must not be treated as Steven's words at the gate.

Affects: CAP-003, CAP-007, CAP-008, CAP-013. Refs: `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `docs/catalog_audit_program/02_SESSION_SEMANTIC_AUDIT.md:9-16`; `reports/catalog_audit_proposals.json`

### CCF-2 — Cited 'scan' artifacts exist only as prose; Session 3 re-derived the load-bearing ones

The divergence_scan, reachability_scan, promised_not_claimed and claimed_not_promised values, kind_ok and scene_ok flags and digest/items.json the proposals cite as machine artifacts do not exist in the repository; the bundle's candidate lists carry item_id, rank and score only. A read-only re-derivation from candidate_enrichment and the catalog confirmed that every pool the review depends on is kind-homogeneous with exact_kind routing and in-scene candidates (CAP-013 ranks 3 and 6, CAP-019 ranks 2 and 1, CAP-008 ranks 3/1/7, CAP-007 valance at rank 2 behind a support_phrase_hit shortcut and absent from the degradation pool of 'Window blinds are worn.'), and confirmed by reading the catalog that the surface divergences named for dated_interior_trim, dated_window_treatment_valance and unfinished_interior_wall_osb_exposed are real. Sessions 4 and 5 should not cite the scans without a committed derivation.

Affects: CAP-007, CAP-008, CAP-013, CAP-019, CAP-021. Refs: `reports/catalog_audit_evidence.json`; `reports/catalog_audit_proposals.json`

### CCF-3 — Production Terra has no literal-reading or negation rule, and rarely engages qualifiers

Terra receives subject + ' ' + state verbatim inside a payload that also carries the 2b observations; the production system prompt defines supported as 'the claimed condition is clearly visible in at least one photo' and says nothing about negations, exclusions or complete-claim reading (the literal-reading prompt exists only in the unimported factorized_review.py). A census of the 33 frozen cases whose claim carries a qualifier (intact, functional, clean, without) found only 10 rationales mentioning the qualifier and 2 rejections resting on one; on 35 cases where a 'dated' or 'basic' observation met a dated-style item, style-word claims were echoed as supported 15 of 18 times while concrete positive predicates were rejected 13 of 17 times when absent; the observation corpus contains no negation at all. Two artifact facts from the sweep confirm it from the other side: in one Terra call the model affirmed fence_weathered ('structurally sound') and fence_broken_or_leaning ('leaning') for the same fence on the same two photos, and it affirmed 'entry or window opening' while its rationale described a gable opening. Any claim that relies on an exclusion clause or on a positive qualifier to discriminate split successors, and any modernization claim whose state is a bare style word, should be assumed permissive until a live check says otherwise.

Affects: CAP-003, CAP-007, CAP-008, CAP-010, CAP-012, CAP-018. Refs: `tools/renovation_architecture/terra_review.py:33-51`; `tools/renovation_architecture/terra_review.py:76-90`; `tools/renovation_architecture/terra_review.py:143-165`; `tools/renovation_architecture/factorized_review.py:105-113`; `reports/catalog_audit_evidence.json`; `rc_6f8278034b05`

### CCF-4 — Ruling 4 post-dates the Phase B adjudications and two clusters' claims use the phrase it excludes

The photo review was adjudicated on 2026-09-01/02 against cluster claims written before Steven's 2026-09-02 rulings; CAP-003's claim_under_test literally asks for 'a neutral but visibly dated palette that a buyer would repaint', which ruling 4 says does not bill, and two of its four adjudication reasons repeat that phrase. A blind re-read under the ruling found two of the four units to be bare intact neutrals. The same ordering affects CAP-008's tells (ruling 4 names era-specific finish or visible deterioration; 'heavily painted-over' is neither).

Affects: CAP-003, CAP-008, CAP-016. Refs: `docs/catalog_audit_program/HANDOFF_SESSION_2.md:74-93`; `reports/catalog_audit_photo_review.json`

### CCF-5 — An unallocated two-property coverage group was left for lead adjudication and never adjudicated in the repo

Two coverage rows (gold:redfin_10806500:photo_001.jpg:g7, a worn-paint iron stair rail, and gold:redfin_11000447:photo_001.jpg:g5, rusted and bent basement window grates), both adjudicated supports_claim with no covering item, record that they were 'reported in promotion_candidates' for the lead with 2 gold units on 2 properties; that record lived in Session 2's scratchpad and no adjudication reached the repository. Session 3 adjudicates it as a documented coverage question, not a promotable gap: neither observation reached Pass 2d (2a described the rail as 'dated', not worn, and 2c dropped the rail half of its bullet; 2a described the grilles as security hardware, 2b's bullet was dropped at 2c, and the gold matching note itself says 'normal-for-context ... not a miss'), so under the frozen-2a rule no catalog item could have recovered either, and the two mechanisms (paint wear on a rail; rust and deformation on a grate) are not obviously one concept. The pinned run artifacts nonetheless show the wider railing concept reaching Pass 2d with no home: 42 railing, handrail or baluster bullets across the 25 canary runs (interior and exterior, wood and metal, mostly degradation-kinded finish wear such as 'Handrail finish is peeling and chipped' at redfin_10952874 and 'The painted wood guardrail is worn and scuffed' at redfin_125970550) resolved to no item, because the only railing item is the safety defect missing_or_damaged_handrails. None is human-adjudicated, so the bar is none, but this is the strongest unadjudicated coverage candidate in the corpus and the first thing to adjudicate if the program buys more review. Promotion recipe: two adjudicated supports on different properties where a railing finish-wear observation reached Pass 2d and found no home.

Affects: —. Refs: `gold:redfin_10806500:photo_001.jpg:g7`; `gold:redfin_11000447:photo_001.jpg:g5`; `reports/catalog_audit_proposals.json`

### CCF-6 — Bundle records carried by no cluster

rc_8afb476fa69f (redfin_11077450) is the only bundle record on exterior_door_paint_failure, a human-approved billing, and is carried by no cluster although CAP-019 names that item as the better selection; rc_470fb7e0624f (dated_interior_trim, redfin_10949071, Terra supported, human inconclusive) is a currently billed plain-trim record absent from CAP-008's regression ledger; gold:redfin_11000447:photo_003.jpg:g3 (bent or damaged venetian blinds matched to an accepted billing of dated_window_treatment_valance) is absent from CAP-007; rc_c5afc5b26357 and rc_4c62bf04b442 are inconclusive appendix cases absent from CAP-019 and CAP-018. None changes an outcome, but the first three change the regression accounting of the two native proposals.

Affects: CAP-007, CAP-008, CAP-018, CAP-019. Refs: `rc_8afb476fa69f`; `rc_470fb7e0624f`; `gold:redfin_11000447:photo_003.jpg:g3`; `rc_c5afc5b26357`; `rc_4c62bf04b442`

### CCF-7 — Splits of the ten marker-listed items are blocked by the catalog 3.2 marker section; other generator hazards

The generator applies package_routing_decisions.repair_support_when_driven after successor construction and fails on a rule naming a deprecated parent; the Session 2 ops surface cannot edit that section. So a split of any of the ten marker-listed items (peeling_or_discolored_paint, worn_or_stained_carpet, vinyl_linoleum_worn_or_stained, worn_or_stained_flooring, baseboard_wear_scuffs, wall_scuffs_marks_or_dents, cabinets_worn_finish, appliances_worn_or_neglected, vanity_countertop_worn, vanity_countertop_dated) needs schema or out-of-band work first; carryover claim edits on those items are unaffected. Two tooling notes for Session 4: a successor without an atomic_claim raises an uncaught KeyError rather than a clean failure (the renderer's dry run would crash instead of reporting ok=false), and severity 0 or an empty support_any is treated as 'missing' by the falsy check.

Affects: CAP-013. Refs: `scripts/migrate_catalog_kind_v2.py:144-168`; `scripts/migrate_catalog_kind_v2.py:243`; `tools/catalog_migrations/kind_v2_decisions.json:45-53`; `scripts/migrate_catalog_kind_v2.py:96-117`; `scripts/render_catalog_audit_proposals.py:646-649`

### CCF-8 — Cutover consequences of any split or claim edit are program-wide, not proposal-specific

requires_re_resolution and deprecated are manifest-only; the only runtime consumer is the publication gate, which rejects deprecated or unknown ids. There is no alias table: server per-image checkpoints are not keyed on the catalog, so a run resumed after regeneration carries stale ids that fail hard (StaleCatalogItemId); backfill_kind_v2 skips artifacts already at the target version and re-resolves by support_any hits only. Any atomic_claim edit changes the projection fingerprint that is hashed into every Terra request, so no stored Terra checkpoint survives a one-field change and Sessions 5 and 6 must replay rather than reuse. A Terra-supported condition on a route_override no_action item stays in observed_conditions and the FE envelope, is counted in dispositions_no_action and carries zero dollars, so 'detectable but unbilled' is real; the regression-register vocabulary cannot express that state.

Affects: CAP-007, CAP-008. Refs: `tools/publication_gate.py:141-162`; `tools/analyzer_server.py:277-329`; `tools/renovation_architecture/conditions.py:88-102`; `tools/backfill_kind_v2.py:258-276`; `tools/renovation_architecture/disposition.py:17-46`; `tools/renovation_architecture/catalog_projection.py:240-253`; `scripts/render_catalog_audit_proposals.py:98`

### CCF-10 — Coverage-triage class inconsistencies (no outcome changes)

An independent pass over the 63 coverage rows found no promotable gap (every row is a gold row on two properties, and no candidate concept both meets the bar and reached Pass 2d with no home) but a dozen class inconsistencies worth correcting before the counts are quoted: several covered_by_existing_item rows point at items whose atomic_claim does not reach the finding (gold:redfin_10806500:photo_001.jpg:g4 gutter extension vs 'clogged, sagging, or disconnected'; gold:redfin_11000447:photo_008.jpg:g3 fitted access panels vs 'cracks, holes, or impact damage'; gold:redfin_11000447:photo_001.jpg:g3 security grille vs a discretionary curb-appeal claim; gold:redfin_10806500:photo_007.jpg:g7 a stained concrete slab under worn_or_stained_flooring, contradicting the entry-steps ruling the same triage applies elsewhere); three product_quarantined rows are plain-presence observations that would not bill anyway (gold:redfin_10806500:photo_007.jpg:g8 bare lampholders in an unfinished basement, gold:redfin_11000447:photo_003.jpg:g8 a plain white fan, gold:redfin_10806500:photo_003.jpg:g5 a surface-mounted panel), so the quarantine's cost is still overstated; gold:redfin_11000447:photo_001.jpg:g6 (a soiled Formstone cornice band, paint wear explicitly not supported) is classed coverage_gap_candidate against the same ruling; and the same range is classed unclear on one photo and not_billable on another. A second two-property documented question surfaced: HVAC ductwork (deteriorating wrap in an unfinished basement; a bare surface-run duct through a finished room) has no catalog item and no trade bucket, but neither observation reached Pass 2d.

Affects: —. Refs: `gold:redfin_10806500:photo_001.jpg:g4`; `gold:redfin_11000447:photo_008.jpg:g3`; `gold:redfin_11000447:photo_001.jpg:g3`; `gold:redfin_10806500:photo_007.jpg:g7`; `gold:redfin_10806500:photo_007.jpg:g8`; `gold:redfin_11000447:photo_003.jpg:g8`; `gold:redfin_10806500:photo_003.jpg:g5`; `gold:redfin_11000447:photo_001.jpg:g6`; `gold:redfin_10806500:photo_003.jpg:g2`; `gold:redfin_11000447:photo_005.jpg:g1`; `reports/catalog_audit_proposals.json`

### CCF-11 — no_change and non_catalog_action are used interchangeably

Clusters whose primary cause is Pass 2d selection are labelled no_change in four cases (CAP-005, CAP-006, CAP-009, CAP-017) and non_catalog_action in three (CAP-011, CAP-019, CAP-021); Terra-primary clusters are likewise split (CAP-002, CAP-004, CAP-015 no_change; CAP-001, CAP-003 non_catalog_action). Neither the policy block nor the renderer defines a difference: both are refutation-backed, both carry no ops. The human gate should read the two labels as one class and, if it wants the distinction, define it (for example non_catalog_action only where a concrete non-catalog action is commissioned) before Session 4 records reclassifications.

Affects: CAP-001, CAP-002, CAP-003, CAP-004, CAP-005, CAP-006, CAP-009, CAP-011, CAP-015, CAP-017, CAP-019, CAP-021. Refs: `scripts/render_catalog_audit_proposals.py:62`; `reports/catalog_audit_proposals.json`

### CCF-12 — Lexical guardrails are brittle to hyphenation and word order, and require_any is unauthorable on carryovers

term_matches is a word-start substring containment on whitespace-normalised text, so deny_any and require_any behave lexically where the proposals reason about them semantically. Verified with the real matcher: on 'The room has a suspended acoustic-tile ceiling.' none of popcorn_or_acoustic_ceiling_texture's ten deny terms and none of suspended_drop_ceiling's require terms match, so the correct item was hard-excluded from retrieval by its own gate and the popcorn item captured the bullet; wall_scuffs_marks_or_dents' deny term 'water stain' never fires on 'heavy staining' or 'stained'. On the pinned record these misses cost nothing (sibling bullets billed the drop ceiling four times on the same property), so no cluster reopens, but require_any is not authorable on any reclassified carryover, which makes a require_any repair a migration-system gap wherever it is needed.

Affects: CAP-014, CAP-016, CAP-017. Refs: `tools/pipeline_common.py:265-284`; `tools/catalog_embeddings.py:439-450`; `tools/catalog_embeddings.py:201-202`; `rc_62d0dc3b9df8`; `item:suspended_drop_ceiling`; `item:popcorn_or_acoustic_ceiling_texture`; `scripts/migrate_catalog_kind_v2.py:83`

### CCF-13 — The generic and bathroom wallpaper items double-billed one wall

On redfin_25809814 photo_019 the same vanity wall produced two accepted conditions: dated_wallpaper_present (the CAP-017 unit) and dated_bathroom_wallpaper (oc1_5314b32efbbd352e), both Terra-supported, both billed under WALLPAPER_REMOVE_PAINT. The overlap is structural: the generic item's scene_groups include bathroom, neither item denies the other, and only the bathroom item carries an allowance cost. It appears once in the 25 pinned runs and the second condition was never adjudicated, so under the program's rules it grounds nothing today; recipe: adjudicate oc1_5314b32efbbd352e and find a second property where both items fire on one wall, after which a deny_any on the generic item (native on a reclassified carryover) or a bathroom exclusion from its scene_groups (a migration-system gap) would be the lever.

Affects: CAP-017. Refs: `item:dated_wallpaper_present`; `item:dated_bathroom_wallpaper`; `photo:redfin_25809814/photo_019.jpg`; `artifacts_canary/renovation_session9_20260818/run_1/candidate/redfin_25809814/20260817_223307_7b336e61/photo_intel_debug.json`

### CCF-14 — Evidence-class precedence for matched gold annotations needs one program ruling

The policy defines a gold photo finding as human evidence, but Session 1's precedence ranks a matched gold annotation row below a factorized lead, so seven runtime units that carry a frozen human gold finding matched to the same condition are counted as model-only (for example CAP-014's redfin_10806500 photo_003 g3 'Painted masonry walls show peeling paint and staining' on a wall_scuffs_marks_or_dents condition). The hunter who raised this proposed relabelling CAP-014 on that basis; the refuter and the lead declined, because it is a program-level rule, not a cluster fact. Whichever way it is ruled, it should be stated once and applied to all seven units, since it decides whether CAP-014 stays lead-only.

Affects: CAP-014. Refs: `docs/catalog_audit_program/HANDOFF_SESSION_1.md:76`; `docs/catalog_audit_program/00_OVERALL_CONTEXT.md:240-247`; `reports/catalog_audit_evidence.json`

### CCF-9 — Frozen-attribution discrepancies for the error-attribution record

Not this program's to change, but the review confirmed four: rc_e9d27cf6e9aa is attributed to 2d while the vanity item was kind-gated out of the pool; rc_7b23faa0bd8f is attributed to Terra while no candidate claimed the casing; rc_905575ed8a3d is attributed to Terra while the pinned review says the soffit panels are not visible; rc_6f8278034b05 is attributed to 2d with high confidence while no candidate names a boarded gable opening. Each is a note for the error-attribution result document, not a catalog finding.

Affects: CAP-006, CAP-011, CAP-012, CAP-019. Refs: `rc_e9d27cf6e9aa`; `rc_7b23faa0bd8f`; `rc_905575ed8a3d`; `rc_6f8278034b05`; `docs/RESULT_error_attribution_20260831.md`

## Questions for the human gate

### Q-1 (CAP-007)

Does ruling 5 ('blinds stay detectable but must be a no_action catalog condition') cover visibly deteriorated blinds (yellowed, bent, crooked, collapsed, as at redfin_81000709 photo_003 and redfin_11000447 photos 003/005), or only ordinary intact blinds like the case that prompted it? The answer withdraws or keeps at least two human/gold-backed billings and roughly two dozen presence billings across the evidence era.

- **All blinds, any condition** → The split is the right shape but must be redrafted as a presence-only, presentation_only blinds successor with a require_any-guarded fabric successor before implementation; rc_7c5b9f9bb3aa and gold g3 are withdrawn as the ruling applied.
- **Only ordinary blinds; deteriorated blinds bill** → A third, degradation-kind successor for yellowed, bent or broken-slat blinds with a work route is needed; the drafted two-way split is wrong and the current item's billing of those cases is correct on the merits.

### Q-2 (CAP-007)

Is an occupant's improvised sheet, cloth or blanket over a window a billable treatment condition or presentation-only (the catalog treats all occupant-belongings items as presentation_only and non-billable), and are heavy or dated fitted curtains and valances billable at all?

- **Presentation-only** → Drop 'improvised' from the fabric successor's claim and embed text; rc_7c5b9f9bb3aa's only surviving billing basis is its deteriorated blind (see Q-1).
- **Billable** → Keep the prong but record that it rests on a single Terra rationale and that Pass 2c drops most improvised-covering bullets, so the recovery is mostly unreachable.

### Q-3 (CAP-008)

Are the two clean plain-trim billings rc_9e889631af67 and rc_d0ffde500a92 to be withdrawn under ruling 3, and is rc_ceeb17f8e242 (human note 'I see the missing trim now') a missing-trim condition that should stay billable somewhere rather than a plain-presence billing to withdraw?

- **Withdraw the two; treat the third as missing trim** → CAP-008 proceeds under its conditions; the missing-trim concept gains a candidate second property for CAP-013's gap.
- **Keep all three billings** → CAP-008 falls to no_change plus documentation of the claim-versus-name divergence.

### Q-4 (CAP-008)

Which claim form is acceptable: a house-style positive-only state of at most nine words (recommended), or the drafted 33-word state with a semicolon and a 'does not qualify' exclusion (not recommended, because production Terra has no exclusion semantics and the clause names the kill-condition trim)? Either form requires the live Terra checks in the conditions before implementation.

- **Positive-only house style** → Session 4 authors the short state plus name and description overrides and returns the redraft for check; the live check decides implementation.
- **Drafted exclusion form** → Accept a catalog-wide first (negation, semicolon, 3.7 times the longest state) and rely entirely on the live check; the review advises against it.

### Q-5 (CAP-008)

Ruling 3 (plain, basic or builder-grade presence is not billable by itself) is being applied to dated_interior_trim alone; six other catalog claims carry a sufficient plain, basic, builder-grade or hollow-core branch (cabinets_dated_style, appliances_dated_or_basic, dated_bathroom_vanity_light, landscaping_enhancement_opportunity, older_ceiling_fan_style, dated_interior_doors). Should the ruling be applied catalog-wide in a later session, or only where evidence implicates an item?

- **Catalog-wide later** → Record a policy-sweep backlog item outside this program; CAP-008 proceeds as the first instance.
- **Evidence-driven only** → CAP-008 stands alone; the other six claims keep billing plain presence until evidence implicates them.

### Q-6 (CAP-013)

Should schema or generator work be authorized to make a defect-kind 'missing or incomplete interior base trim' item expressible (the blockers are the catalog 3.2 marker section that names split parents and the unauthorable inherited estimate scope), or should the gap stay documented until a second property is adjudicated?

- **Authorize schema work** → A separately scoped task edits the marker handling and successor economics authoring; not part of Session 4's approved catalog changes.
- **Document and wait** → CAP-013 stays a recorded gap; the three candidate cards (rc_ceeb17f8e242, rc_7057c3c171e5, rc_201ccb0c2313) are the first things to adjudicate.

### Q-7 (CAP-002)

Does damage to a detached outbuilding (shed, barn, garage) bill under the dwelling siding item damaged_or_rotted_siding_or_trim, a severity-3 exterior_repair package driver?

- **Yes** → CAP-002 stays no_change; the shed case is a Terra miss.
- **No** → A dwelling-only scope needs a migration-system or policy change (subject, severity and package fields are not wording overrides); reopen CAP-002 on that basis.

### Q-9 (CAP-019)

Does exterior_door_paint_failure's subject ('exterior door paint') include the door casing, surround and jambs, or the door leaf only? On redfin_10740044 the surround is rough and paint-smeared while the leaf is sound, and no catalog claim names exterior casing although three items advertise it in retrieval text.

- **Leaf only** → The right item was definitionally unavailable; CAP-019 moves to deferred_insufficient_evidence as a one-property coverage question with the recipe already recorded.
- **Casing included** → The Pass 2d follow-ups stand as the primary record, and a future name/description override on the door item (native) could make the claim's scope explicit.

### Q-8 (CAP-003, CAP-001)

Given that Terra's 'neutral, not clearly dated' rejections match ruling 4 on the bare-neutral units, should a Terra calibration still be commissioned for dated paint, and if so only for ruling-4 tells (era-specific finish, visible deterioration)?

- **Commission for ruling-4 tells only** → Re-adjudicate the four CAP-003 units under the ruling first and include CAP-001's peeling/aged-finish disjunction in the same calibration.
- **Do not commission** → CAP-003 and CAP-001 stay non-catalog with the losses recorded as Terra behaviour under current policy.

## Appendix — pins and reconciliation

| Input | sha256 |
|---|---|
| docs/PROPOSAL_catalog_audit_20260901.md | 771572e7c1d095281756b6baf9f28e65cffedc7f9b04cabfd559117768b17591 |
| reports/catalog_audit_evidence.json | 43354104cc109c768efbe2d2f8b9a825f33e0882210b81e9f12b847ca67e29b7 |
| reports/catalog_audit_photo_review.json | 2225db9caae573b52b4270f846c314749d39aff468fab4e1a8e2012c49919150 |
| reports/catalog_audit_photo_review_packet.json | 7b4621dfe6c72aa42369c2f5ef7051dfecbec05b6333a80651b1b57a5b37bc80 |
| reports/catalog_audit_proposals.json | 1411352c42e9d3a555c3a1561470323da2234447badc2869c9874c3e8fa3ef99 |
| scripts/migrate_catalog_kind_v2.py | b732022fcea8cd22109c7781eb0f93d0688d557230c759ccb6216d667250a053 |
| tools/catalog_migrations/kind_v2_decisions.json | 47614d822bdd6b672d8596050b46839d04a8a6df0a04c18e052f4a63bd1903e8 |
| tools/catalog_validation.py | 041637a3684bacd3bc06dd92955b845088b4fedda8bb7e2e31222ac9ec75bee2 |
| tools/issue_catalog.json | 4ba046a1a78337f1c8e47701a011ec16e700c296782ddedbe7bf52cf888314f2 |
| tools/issue_catalog_kind_v2.json | 51bf7e263ff98109d657ef730b0ce1a705598101f09843eb288b1bdc3e0eaa54 |

git HEAD: `9afe0fa5a0490a856abed507dccc4a02ed1de24c`

Expected ids: 20; reviewed: 20; missing: —; orphans: —; duplicates: —.

References resolved by kind: `{"card":59,"file":15,"file_line":96,"item":50,"photo":27,"unit":35}`; unresolved: —.

Validation: ok=`True`
