# Decision packet — bounded catalog policy checkpoint

Date: 2026-09-08. Phase 1 of `docs/catalog_audit_program/08_SESSION_BOUNDED_POLICY_CHECKPOINT.md`.
Status: **for Steven's decision. No catalog, decisions, prompt, route or threshold was changed; no provider
call was made; nothing is approved by this document.** Machine record:
`reports/catalog_policy_checkpoint_packet.json` (`99a40d74af9a9d36f5b4e644f4bba61fb2c055181268a8a491553c7f63791637`),
built from `reports/catalog_checkpoint_source_manifest.json` and the extraction scripts, lever
evaluation and with/without replays frozen under `artifacts_canary/catalog_checkpoint_20260908/provenance/`
(hashes in the packet's `provenance_sha256`). Revised the same day after Steven's trim photo review (section 2a).

Inputs: the ruled Session 6 bundle (`candidate_rejected` on `6e67eaa` pending the trim lever), the CAP-022
proposal and its frozen offline evaluation, the approvals manifest `a725c89a…`, the 1,969-row replay corpus
(frozen Pass 2c bullets from the 25 pinned runs), the generated catalogs of both arms, the benchmark slices,
the pinned photo review, and the listing photos under the frontend repo. Evidence units are counted by property
and by human adjudication; repeated model outputs and same-property cards are not independent corroboration.

## 1. What was measured

Offline, sidecar only, one embedding run, every delta exact (instrument identical to the Session 6 CAP-022
evaluation and reproducing its numbers: `require_any` set B removes trim from 699 top-8 lists, loses 4
trim-owned rows, changes 0 shortcuts):

| Lever on `dated_interior_trim` | Rows losing trim from top-8 | Trim-owned rows lost | of which a **billable** neighbour becomes first | Shortcut changes (to a billable item) | Consequential changes (rule 3) |
|---|---:|---:|---:|---:|---:|
| A. `require_any` set B (CAP-022) | 699 | 4 | 1 (`dated_wood_paneling`, "The built-ins are dated.") | 0 | 26 |
| B. `deny_any` presence stems (basic, plain, builder, builder-grade, simple) | 245 | 54 | **20** (paneling 8, interior doors 6, flooring 3, paint 2, kitchen/vanity 1) | 11 (1: `dated_interior_doors` by shortcut, m=0.096) | 76 |
| C. A + B combined | 764 | 57 | **21** | 11 (1) | 80 |
| D. retirement (first-cut from frozen lists) | n/a | 108 | 55 have a billable next candidate | n/a | n/a |

Instrument self-check against the Session 5 candidate snapshot: 1,919 of 1,969 top-8 lists identical
(45 order-only swaps, 5 eighth-slot substitutions, first candidate differs on 2 rows: "Finishes are mismatched
and dated." and "The backsplash is basic."), maximum score delta on identical lists 0.0013, and **one shortcut
mismatch**: "The mini blinds feel dated compared with the updated kitchen." sits at margin 0.02998 in this run
against 0.03004 in the snapshot, so it takes the LLM path here and the shortcut there. That is the S5-2/S6-5
row; the live run also took the LLM path. Listed per rule 3, not waved through.

Photo re-reads (this pass): the S6-1 row "Window treatments are dated and basic." (redfin_10803207 photo_017)
shows plain, intact, freshly painted casings and baseboards; CAP-022's "declared expected landing" R1 "Window
blind and trim appear dated." (photo_005) shows ordinary white painted bathroom casing. Neither is dated or
damaged trim. The four CAP-022 losses are a renovated bedroom with new plain trim, an older house whose subject
is aged doors, an unfinished basement, and painted built-ins: none is trim. Four "dated"-worded trim bullets
sampled across properties: three show plain, often newly installed trim; one shows worn casing (a repair
condition, not a modernization one).

## 2. Trim: the decision

The item bills "plain, thin, or builder-grade trim package" at `TRIM_REPLACE`. It owns 108 corpus bullets on
20 properties: 48 use plain-presence wording, 46 say dated or older, 6 say both, 7 say modest or minimal, 1 says
incomplete. Ruling 3 says plain presence does not bill. Interior trim coverage elsewhere is one item,
`baseboard_wear_scuffs` (degradation, repair); there is no item for missing trim (CAP-013) and none for
dated-style trim other than this one.

**Rejected mechanisms.** CAP-022 alone: fixes the subject leak, leaves every trim-naming plain-grade billing in
place, including R1. Presence-stem deny (B, C): the plain bullets do not disappear, they migrate; 20 of the 54
rows it removes get a billable neighbour first, chiefly wood paneling and interior doors in rooms that have
neither, two of them by lexical shortcut. That trades a wrong $76 trim billing for a wrong paneling or door
billing. Retirement: the same migration, worse (55 of 108 rows), plus loss of visibility and the stale-id cutover.

**Option (b), recommended: `route_override: no_action` on `dated_interior_trim`.** Every trim bullet keeps its
trim-shaped landing and displays at $0; nothing leaves the retrieval pool, so nothing migrates; S6-1's two
conditions display instead of billing; CAP-022 becomes unnecessary (a subject-generic bullet on a no_action item
is a display question). Ruling 3 is satisfied completely. Costs: genuinely dated trim stops billing (the
recovery card rc_b642fe69b86a and, on the corpus and photo sample, perhaps 10 to 15 of the 46 dated-worded
rows); the two CAP-013 candidate billings on this item (rc_ceeb17f8e242, rc_57c6faf79c7e) stop, which Q-3 and
A-2 already treat as withdrawals, and the missing-trim loss is disclosed for acceptance; trim leaves the
package support lists (its per-scene affinity is `package_support`; no package driver is lost, but section 2a
measures what the lost support does to package strength). The projection fingerprint moves
(route change) but `atomic_claim` does not, so stored Terra verdicts stay claim-compatible and the item id
survives (no new stale ids). Needs the generator to admit `route_override` on a carryover override: one change
with a test.

**Option (a2), the preservation alternative: split by the CAP-007 pattern.** `interior_trim_basic_or_plain`
(presence_only, `route_override no_action`, visible, `support_any` basic/plain/builder/thin/narrow vocabulary)
and a dated-style successor with a `require_any` trim-family subject gate (set B) and a claim on style or
period, economics inherited. Both successors are split successors, so no generator change is needed for this
item. Plain bullets keep a trim-shaped no_action landing (no migration), dated trim keeps billing, the subject
leak is closed on both successors. Costs: new ids, so the same 97-artifact stale-id cutover CAP-007 already
owes applies to trim as well; and a **residual that this packet cannot remove**: Terra supports "dated" on
plain trim (the CAP-008 kill card, 3 of 4 sampled photos), so roughly half of the dated-worded plain-trim
bullets would still bill on the dated successor. Preservation therefore reduces plain-grade billing by about
half to two-thirds; it does not eliminate it. Measurable exactly offline before approval (two new embeddings).

Under either option CAP-008 is resolved and CAP-022 is subsumed. The choice is Steven's: zero false trim
billing at the price of the dated-trim minority (b), or dated-trim billing preserved at the price of a disclosed
residual and the trim cutover (a2). The standing priority (fewer false billings over dollars) favours (b).
Section 2a adds what Steven's photo review and the deterministic with/without replays change about this.

Cohort if either lands: 128 trim-owned or trim-first rows in 58 estimate units on 21 properties, 19 of them
already in Session 6's Stage A set (two new: redfin_11079485, redfin_11216660). Under (b) retrieval does not
change, so the offline instrument settles routing and the live question is only display and disposition.

## 2a. Steven's trim photo review and the with/without checks (2026-09-08)

Steven's notes are recorded verbatim in `docs/catalog_audit_program/REVIEW_trim_photos_steven_20260908.md`.
Standing intent: trim observations are primarily context for broader room work; real missing trim is valuable
work in its own right; an unsupported trim claim is never evidence of other defects nor the sole justification
for a package. Kitchen card rc_ceeb17f8e242 (redfin_11185681): trim is missing, a valuable detection, part of
the broader kitchen work. Bathroom card rc_57c6faf79c7e (redfin_125779232): trim not particularly bad, claim
not necessarily true, the bathroom overall is the concern and must be costed without relying on trim.

**Deterministic check, two units** (provider-free replay of the stored artifacts through the current chain,
as-is versus trim at `no_action` patched in memory; stored Terra verdicts and Sol decisions reused):

| Unit | As-is | Trim at no_action |
|---|---|---|
| Kitchen `kitchen_primary` | trim `TRIM_REPLACE` $92–460 is 1 of 4 supports of the approved full-rehab kitchen modernization package ($26,234–61,213; driver `CABINET_REPLACE`); absorbed at $0 standalone | trim displays at $0, visible; the package candidate rebuilds **identical** in strength, tier, price and driver with 3 supports; the changed support list changes the candidate payload hash, so the stored Sol decision is incompatible and the replay drops the package |
| Bathroom `bathroom_primary` | trim $76–383 is 1 of 2 supports of the approved partial-rehab bathroom modernization package ($5,826–14,566; driver `vanity_dated_style`); a separate approved bathroom repair package ($1,821–15,294) | modernization candidate rebuilds **identical** (same driver, tier, price) with 1 support; repair package unchanged, its decision kept; the modernization decision is payload-incompatible and dropped |

Reading: neither unit depends on trim for its warranted work or its package price. Both depend on trim only
through the support list, and any support-list change invalidates a stored Sol decision. Replay headline drops
in the no_action variant are that incompatibility, not removed work, and are reported as such. The kitchen's
missing trim is cabinet casework trim (crown, filler, edge) and is represented in both variants by the
cabinet-replacement driver inside the package, at $0 standalone in both; what `no_action` removes is the
condition's support membership and its `accepted_for_work` label. Missing trim still has no catalog owner as
direct repair work (CAP-013): if the cabinet drivers were ever rejected, the work would have no representation,
exactly as today. The bathroom is costed without trim under both variants.

**Deterministic check, corpus** (the 17 of 26 pinned artifacts that carry a v5 envelope; the 8 production
artifacts predate v5 and cannot be replayed):

- 44 active trim work items as-is: **27 bill standalone** ($2,242–11,212 in total), the ruling-3 exposure;
  **17 are absorbed** into room packages at $0, the "context" Steven values. By scene: bedroom 21, living 8,
  kitchen 5, bathroom 5, basement 4, closet 1.
- With trim at `no_action`, of 102 package candidates 77 are identical, 15 change only their support list
  (stored decision incompatible, package otherwise the same), 9 change shape: seven bedroom modernization
  packages fall from strong to moderate at unchanged tier and price because trim was one of two supports; two
  living packages keep strength and tier and lose $394 and $526 from their high; one driverless bedroom
  package (redfin_166147710, formed from supports only, $1,147–4,588) no longer forms because trim was its
  only modernization-kind support, and its three repair items ($197–983) stay standalone. That last case is
  the pattern the standing intent forbids (a package resting on a trim claim) and its disappearance is correct.

**Three classes, made explicit.** Direct repair work: missing or damaged trim; no catalog owner today except
`baseboard_wear_scuffs` (repair); the kitchen card is this class and is carried only by the kitchen package's
cabinet driver; CAP-013 is the gap. Supporting room context: the 17 absorbed items and the support membership
on 25 candidates; every catalog-only mechanism (no_action or split) removes plain-trim support membership,
because a no_action condition forms no work item; letting a no_action condition still count as package
context would be new code and is not approved. Unsupported or plain observation: the 27 standalone billings;
the bathroom card is this class.

**What this changes in the recommendation.** Option (b) still removes exactly the 27 standalone billings and
keeps every trim observation visible. Its cost under Steven's intent is now measured: plain-trim context stops
counting toward package strength (seven bedroom packages strong→moderate; two living highs a few hundred
dollars lower; one support-only package gone), and every affected package needs a fresh Sol decision, which
the cutover replay requires anyway. Option (a2) keeps context and billing for dated-worded trim only, with the
residual disclosed in section 2. Neither option gives missing trim a direct-repair owner.

## 3. The other eight families

| Family | Authoring | Corpus exposure | Disposition | Reopening trigger |
|---|---|---|---|---|
| `cabinets_dated_style` | split successor (levers native) | 52 rows; 3 plain-presence ("Lower cabinets are basic white", "Kitchen has basic white cabinets", "Cabinet hardware is basic") | tranche 2: no change now | a human-adjudicated basic-grade cabinet billing, or the trim mechanism proven and applied catalog-wide (split pattern, not a deny) |
| `appliances_dated_or_basic` | split successor | 41 rows; 7 plain-presence (basic range, hood, refrigerator) | tranche 2: no change now | same; keep modernization separate from the damage sibling |
| `dated_bathroom_vanity_light` | carryover | 15 rows, **0 conditions**: electrical bucket is product-quarantined, route `excluded_quarantine` | no change; already non-billing | the electrical quarantine is lifted |
| `landscaping_enhancement_opportunity` | split successor | 4 rows at $0 | no change; `route_override_no_action` verified in v5 | none |
| `older_ceiling_fan_style` | carryover | 54 rows, **0 conditions**: quarantined | no change; already non-billing | quarantine lifted |
| `dated_interior_doors` | carryover | 36 rows; 7 plain-presence ("six-panel builder-grade door", "builder-grade bifold doors"), 6 "basic and dated"; the claim equates hollow-core with warranted replacement; also the destination of the trim deny lever's worst shortcut | tranche 2, strongest case; no change now | trim mechanism proven; or a human-adjudicated builder-grade door billing |
| `dated_window_valance_or_curtains` | split successor (candidate) | approved wording stands; "basic curtains" and curtain rods were Terra-rejected live under the tighter claim (S6-3) | no change; S5-1 closed, A-5 preserved | a Terra-supported basic-curtain billing in the cohort |
| `paint_refresh_recommended` | carryover | 40 rows, 38 dated-worded, no plain-neutral bullet | no change; Q-8 re-adjudication stays the precondition | Q-8 showing plain neutrals billing |

Observation-level note, not a loss: "The brown valance appears dated." (kitchen) never reaches the fabric
successor in the candidate arm (not in its eight candidates, unresolved 6 of 6), but its condition is shared
with two sibling observations that do reach it, so the condition survives. Short valance bullets are a
fragility of the fabric successor's embedding to watch in the cohort.

Benchmark gold on these families (`res-cab-003/005`, `res-app-003`, `res-pnt-004`, `res-lnd-002`,
`res-mod-006`) records resolution identity only and is preserved as is. Option (b) leaves all of it reachable;
option (a2) makes no trim gold unreachable because there is none.

## 4. The six deferred CAPs

| CAP | Disposition | Reopening trigger |
|---|---|---|
| CAP-008 | resolved inside the trim decision; the four human-approved claim-exact billings withdraw under either option (Q-3, A-2), rc_57c6faf79c7e's loss disclosed | n/a |
| CAP-010 | deferral retained; the lean was captured and billed on `fence_broken_or_leaning`; exposure semantic, not dollars | a human unit on a different property where a leaning fence was lost or double-billed across the two successors |
| CAP-013 | no schema work (Q-6). The four cards re-read: rc_ceeb17f8e242 (redfin_11185681) is the only one on a new property and the photo review supports missing/plain trim; rc_7057c3c171e5 and rc_57c6faf79c7e are photo-review "unclear"; rc_201ccb0c2313 and rc_57c6faf79c7e share a property. No catalog owner exists for missing trim; under the trim decision the two billed on `dated_interior_trim` become explicit missing-trim losses | Steven adjudicates rc_ceeb17f8e242 (and rc_57c6faf79c7e) as missing-trim work **and** approves the marker/scope authoring extension |
| CAP-014 | deferral retained; the eight scuffs-vs-paint follow-ups are Pass 2d selection, the two reachability rows are Pass 2c kind exclusion, graffiti is naming; no catalog-owned defect on the evidence. Only one of the ten leads carries matched gold (g3, redfin_10806500 photo_003) | two human reviews on different properties plus a demonstrated catalog mechanism |
| CAP-016 | expansion deferred; the swirl example was Terra-supported and billed under `CEILING_TEXTURE_UPDATE`, so it is a naming question; the bare `ceiling texture` shortcut token goes to the Pass 2d worklist | a concrete era/style ruling case or a consequential subject correction |
| CAP-018 | deferral retained; the corpus supplies no second property (the only other stone/brick unit is the refuted one) | second-property human unit; the stone-near-grade billability ruling; the intact-mortar qualifier; an `expected_after` for rc_dff4c3209df1 that is not flips_to_supported |

**CCF-14 (matched gold) recommendation:** rule once that an explicit matched human gold annotation is evidence
of the observation it states, so the unit counts as human on the observation axis; billability stays a
separate axis; automated `gold_incomplete` matches are not evidence. Under that ruling CAP-014 gains one human
unit on one property and still does not meet two-property independence.

## 5. Two one-pass checks

**Wallpaper (CCF-13).** Confirmed one wall, two charges, and the mechanism is deterministic: on redfin_25809814
photo_019 `dated_wallpaper_present` ($140–699, standalone, `no_covering_package`) and
`dated_bathroom_wallpaper` ($345–1,726, absorbed by the approved bathroom package) both bill
`WALLPAPER_REMOVE_PAINT` on `bathroom_primary`; the work-item dedup keys on trade bucket and unit policy, which
differ (`interior_finishes`/`per_scope` vs `paint_drywall`/`per_bathroom`), and the generic item has no bathroom
`package_affinity`, so nothing absorbs it. Provider-free replays under the current 3.2 catalog reproduce it
byte-identically; the 3.1→3.2 migration did not change it. The register's "second property" requirement
establishes frequency, not existence. Levers: a bathroom exclusion from the generic item's `scene_groups`
(carryover `scene_groups` is not authorable today: same generator-support question), a `package_affinity` key
(economic, never authorable), or the dedup key (code, outside this program). Disposition: confirmed; if the
generator support admits carryover `scene_groups`, the exclusion is a one-line op measurable offline; otherwise
parked with that trigger. Not fixed through the dedup key here.

**Railing.** 43 railing bullets in the corpus; 20 resolved to no item; 14 of those are modernization-kind
"dated railing" bullets on 8 properties (one property contributes 7). None is finish wear (peeling, rust,
scuffs) and none is a safety defect. The coverage question is dated railing *style*, which under ruling 3 is the
presence/style billing this checkpoint is removing elsewhere. Deferred; trigger: a human-adjudicated railing
finish-wear or unsafe-railing observation that reached Pass 2d with no home, or a product decision that dated
railing is a visible no_action concept.

## 6. Blinds visibility (S6-2)

Ten candidate rows resolved to no item six times out of six. Five were already `excluded` at baseline ($0, no
visibility change). Five were `accepted_for_work` at baseline ($79–92) and now display nothing; one of the
five is the A-6 makeshift row. The only preservation lever is adding the bare token "treatment" to the blinds
successor's `support_any`, rejected at the redraft because it opens shortcut surface. **Recommendation:
accept the omissions** as presentation-only, record the ten rows as the accepted-limitations list, and revisit
only if the frontend needs the no_action envelope for blinds specifically. Until Steven accepts, the A-4
visibility requirement stays active and the conditions count as open.

## 7. Pass 2d worklist, reconciled

26 rows: 17 incorrect selection with the better item reachable at rank 2–6; 5 shortcut or rank-1 selection;
3 candidate absence or Pass 2c kind exclusion (two CAP-014 rows, the CAP-005 `older_flooring_style` row);
1 naming question after the split (the CAP-007 row now points at `window_blinds_basic_or_plain`). The trim
homograph row (CAP-021, "trim" as plumbing) displays at $0 under option (b). Per-row axis is in the packet JSON.

## 8. Support request and approval shape

One generator change: `WORDING_OVERRIDE_FIELDS` → `CARRYOVER_OVERRIDE_FIELDS` admitting `require_any` and
`route_override` (option (b) needs `route_override`; (a2) needs nothing for trim), `scene_groups` only if the
wallpaper exclusion is chosen; economic and unknown keys keep failing; validator dead/redundant-override checks
unchanged. Renderer: a current authoring surface for new ops, historical classification untouched, versioned
surface record. Harnesses: manifest-parameterized Tier 1/2 and live harness with the Session 5/6 pins as
historical defaults. Order: Steven authorizes support → it lands with tests → exact ops are approved in a new
`reports/catalog_audit_approvals_v2.json` (template in the packet JSON, `approved_against` pinning `6e67eaa`,
the post-support generator blob, the CAP-022 proposal hash, this packet's hash and the source manifest hash) →
implementation. The existing approvals manifest is never edited.

Affected artifacts: 97 stored canary artifacts name `dated_interior_trim` (the same 97 that name the removed
valance parent; per-family counts in the packet). Any claim-text change moves the catalog-wide projection
fingerprint regardless of the count.

## 9. Decisions requested

1. **Trim:** (b) no_action, or (a2) split with the disclosed residual and cutover. (Not CAP-022 alone; not the deny lever; not retirement.)
2. **Generator support scope:** `require_any` + `route_override` on carryovers; add `scene_groups` if the wallpaper exclusion is wanted.
3. **Blinds visibility:** accept the ten presentation-only omissions, or keep the visibility requirement open.
4. **CCF-14:** matched human gold annotation counts as evidence of the stated observation, billability separate.
5. **Tranche 2:** confirm "no change, recorded with trigger" for cabinets, appliances, doors, paint; or name the family to carry into this checkpoint.
6. **CAP-013 cards:** answered in part by the photo review (kitchen: missing trim, real work; bathroom: not confirmed). Remaining: accept that the kitchen's missing trim is represented only through the cabinet driver inside the package, or authorize the CAP-013 authoring extension so missing trim has an owner.
7. **Supporting-context mechanism:** should a `no_action` trim condition still count as package support? That is new code with its own gate, outside this catalog checkpoint; a yes changes the cost of option (b), a no leaves the measured context loss in place.
