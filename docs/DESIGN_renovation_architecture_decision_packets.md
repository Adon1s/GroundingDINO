# Design document — renovation architecture decision packets (post–Session 9 canary)

Date: 2026-08-21. Branch: `renovation_architecture_rework`. Status: v5 runs in
**shadow**; production selector is still `current` (v4). The Session 9 canary
report is mechanically green (`release_ready: true`, 967/967 reviews accepted,
comparator reproducible byte-for-byte), and every review explanation was
independently audited against the artifacts. This document exists because the
audit surfaced decisions that the per-item review could not see and that only
Steven can make. It is a reference to come back to, not a handoff.

Companion files (all regenerable by
`scripts/build_session9_decision_packets.py`, zero provider calls):

- `reports/session9_photo_review_20260821.html` — every disputed condition,
  package and bathroom with the **photos inline** next to what each model said.
  Open the file directly from disk in a browser (double-click / `file:///…`);
  the images are `file:///` links into the local renointel-prod image folder
  (`…\renointel-prod\public\images\properties\<property>\photo_NNN.jpg`), so
  an http preview or a snapshot viewer will show the text but not the photos.
- `reports/session9_decision_worksheet.json` — blank verdict fields keyed by
  the item ids in the sheet (C### conditions, P## packages, B## bathrooms).
- `reports/session9_decision_packets.json` — every number and list cited here.

Tables inside `<!-- GEN:… -->` markers are generated; the prose is not.

---

## 0. How to use this document

Each packet is one decision. Each has: the question; why it matters in
Steven's terms (fewer hallucinations, more useful packages; prices are
provisional); the evidence; exactly what to look at; the options with their
consequences; a recommendation (mine, labelled); what closes the packet; and
dependencies. Work them in the order in §9 — the three photo passes
(conditions, packages, bathrooms) come first because their results feed the
architecture calls. Record decisions in the log in §9 (and verdicts per item
in the worksheet).

Ground rules carried over from the plan (`docs/IMPLEMENTATION_PLAN_renovation_scope_estimate_architecture.md`):
no price calibration in this migration; Terra judges only supplied conditions;
Sol judges only supplied candidates and cannot change condition/work truth;
rejected/uncertain packages keep children standalone; v4 stays the rollback
authority.

## 1. One-page context

**How v4 verified packages.** v4 built package candidates from issues and then
ran **Pass 2f**, a VLM call that looked at the room's photos and answered at
the *package* level ("does this room warrant a kitchen modernization?"),
confirming/rejecting individual issues along the way. It also carried a
multi-bathroom expansion heuristic (§P3) and subsumed a same-unit repair into
the modernization package.

**How v5 verifies.** Terra returns a separate verdict per *condition*
(`supported` / `unsupported` / `cannot_assess`) against that condition's
photos — the calls themselves are batched per estimate unit (~7 conditions per
call in the canary); deterministic policy
turns verdicts into dispositions, work items and package candidates; **Sol**
judges only whether a candidate's grouping is *coherent* and is instructed to
"never reassess whether a condition is real". There is no package-level photo
check in v5 — that is the design, not an oversight, and it is what P1 is about.

**The canary.** 18 properties × 2 replicas (separate days), frozen inputs.
Mechanical gates: schema, publication, reconciliation, accepted-work, freeze
binding, manual review — all pass; the audit session re-ran the comparator
into a scratch path and got a byte-identical report. Stability: `headline ==
standalone + packaged` holds on all 36 artifacts; replicas differ mainly
through upstream detection churn — on average about a third of a property's
replica-1 (item, unit) conditions are absent from replica 2 and about half of
the union differs (measured figures in the facts block below; the earlier
"~25%" shorthand understated it).

<!-- GEN:facts -->
- Corpus (run_1, 18 properties): v4 $551,717/1,848,871 → v5 $617,377/2,220,253 (+11.9% / +20.1%).
- Terra verdicts (both replicas): {'supported': 1746, 'unsupported': 282, 'cannot_assess': 65}; dispositions: {'accepted_for_work': 1483, 'excluded': 382, 'no_action': 146, 'inspection': 69, 'withheld': 13}; Sol decisions: {'approve': 226, 'reject': 23}.
- Models: Terra ['openai/gpt-5.6-terra/terra_condition_review_v1'], Sol ['openai/gpt-5.6-sol/sol_package_review_v1']; reasoning_effort Terra=medium, Sol=medium; upstream routing {'1a': {'model': 'unsloth/qwen3.6-27b@q6_k', 'reasoning_effort': None, 'max_output_tokens': None}, '2a': {'model': 'gpt-5.6-terra', 'reasoning_effort': 'low', 'max_output_tokens': 2000}, '2b': {'model': 'gpt-5.6-terra', 'reasoning_effort': 'low', 'max_output_tokens': 2000}, '2c': {'model': 'gpt-5.6-terra', 'reasoning_effort': 'low', 'max_output_tokens': 2000}, '2d': {'model': 'unsloth/qwen3.6-27b@q6_k', 'reasoning_effort': None, 'max_output_tokens': None}, '2f': {'model': 'gpt-5.6-terra', 'reasoning_effort': 'medium', 'max_output_tokens': 4096}}.
- Replica churn, keyed (catalog_item_id, estimate_unit_id): on average 35% of a property's replica-1 conditions are absent from replica 2; 51% of the union differs.
- Terra condition-level consistency proxy: 250 conditions with the same item/unit/photo set in both replicas → same verdict 234, flipped 16 ({'unsupported->supported': 6, 'unsupported->cannot_assess': 1, 'supported->unsupported': 9}); batch context (other conditions in the same estimate-unit call) differed, so this is a proxy, not a strict identical-input probe.
- Measured tokens per listing (both replicas, 36 artifacts): Terra condition review mean 51,008, Sol package review mean 8,545.
<!-- /GEN:facts -->

**Steven's evaluation frame (2026-08-21).** Not a renovator; prices are
provisional. Judge the rework on (1) fewer hallucinations and (2) more
observations that build useful packages. Dollar deflation from dedup is
accepted. Price calibration is a later thread.

---

## 2. The variance question (read before P1/P4)

Steven's worry: Sol is inconsistent, and the model family no longer exposes a
temperature knob. Facts from the code and the canary:

- Terra and Sol are `gpt-5.6-terra` / `gpt-5.6-sol` via the OpenAI Responses
  API. `tools/vlm_client.py` never sends `temperature` on the OpenAI path (the
  0.2 default only reaches the LM Studio path); for the gpt-5 family it sends
  `reasoning.effort` (and `text.verbosity` only when a caller supplies one —
  Terra and Sol do not, so effort is the only knob in play). Terra condition
  review and Sol package review both run at `reasoning_effort="medium"`
  (`tools/renovation_architecture/contracts.py`); the canary's 2f also ran at
  medium. On the provider side: for reasoning requests OpenAI/Azure document
  `temperature`/`top_p` as unsupported (community reports: accepted only with
  `reasoning.effort="none"`, i.e. by turning reasoning off — not verified from
  current OpenAI pages), and the Responses API has no `seed` parameter. So
  temperature is not a usable lever for Terra/Sol as configured; the levers are
  reasoning effort (ladder `none/low/medium/high/xhigh/max`, plus gpt-5.6's
  `reasoning.mode="pro"`), prompt/rubric structure, deterministic rules, and
  multi-sample voting.
- Measured Sol self-consistency on **identical candidates** (same package,
  same child catalog-ids, present in both replicas — inside listing-level Sol
  calls whose *other* candidates differed): agreement on 29 of 31, 2 flips,
  both single-child packages (P4 table). Cross-property consistency (mine: it
  looks no better) — single-child candidates 23 approve / 2 reject;
  exterior_repair 7 / 3, though those child sets are not identical.
- Terra self-consistency on strictly identical calls is unmeasured (no Terra
  request fingerprint repeats across replicas — calls are batched per estimate
  unit and churn changes the batch). A free **condition-level proxy** exists:
  250 conditions have the same item/unit/photo set in both replicas and Terra
  gave the same verdict on 234, flipped on 16 (6.4%; 9 supported→unsupported,
  6 unsupported→supported, 1 →cannot_assess). Batch context differed, so it is
  a proxy, but it sizes the term before spending a Terra day (below).

Options for reducing variance (not mutually exclusive):

| option | what it does | cost | what it fixes |
|---|---|---|---|
| A. Deterministic rules for the unstable classes | e.g. "single-child candidate ⇒ never a package (bill standalone)" or "⇒ always a package"; exterior_repair coherence by an affinity table; split → sub-package | code only; removes Sol calls for those candidates | removes the observed flips entirely; cross-property consistency by construction |
| B. Structured rubric | ask Sol 2–3 yes/no sub-questions (same room? same trade? one mobilization?) and derive the decision deterministically from the answers | prompt change + validator | moves the judgment into checkable parts; reduces free-text drift |
| C. Self-consistency voting | call Sol N=3 per listing, majority per candidate | Sol measured ~8.5k tokens/listing in this canary; ×3 ≈ +17k/listing, and Sol's 250k/day ceiling drops from ~29 to ~10 listings/day | reduces random flips; does not fix systematic cross-property inconsistency |
| D. Raise Sol `reasoning_effort` (high → xhigh → max) or use gpt-5.6 `reasoning.mode="pro"` | more deliberation per call | more tokens/latency; effect on variance undocumented (OpenAI's guides claim consistency at medium/low effort, nothing about variance vs effort) | measurable only by probe |
| E. Decision cache keyed by candidate fingerprint | same candidate ⇒ same stored decision on re-analysis | code only | stability across re-runs of the same listing, not correctness |
| F. Accept | LLM variance is bounded and fails safe (children stay standalone) | none | nothing |

**How to measure before choosing (cheap, offline-ish):** a *Sol consistency
probe* that re-submits the stored run_1 package candidates to Sol N times and
counts flips (18 properties × 3 samples × ~8.5k ≈ 460k Sol tokens ≈ two
Sol-days at the 250k ceiling, or one day on 9 properties). A *Terra consistency
probe* re-asks Terra the stored run_1 conditions with the same photos once more
(≈ 51k Terra tokens/listing ≈ 0.9M for 18 ⇒ one Terra day) — worth doing only
if the 6.4% proxy flip rate above is judged material. Neither needs upstream
passes or a new freeze; both need a small "re-decide" mode added to
`scripts/replay_renovation_architecture.py` (today it replays stored verdicts
and decisions verbatim with zero provider calls).

**Recommendation (mine):** A for the single-child class and split→sub-package,
B for the rest, then run the Sol probe to confirm the flip rate drops; keep C
in reserve. Terra probe once, to know whether Terra variance is a real term or
a rounding error before designing anything around it.

---

## 3. Packet P1 — Package-level photo verification (Sol approves what v4's 2f rejected)

**Question.** When Terra confirms the individual conditions but v4's
package-level verifier said the room does not warrant the package, who is
right — and does v5 need a package-level photo check?

**Why it matters.** This is the dominant driver of v5's positive delta and it
is exactly the hallucination question. The 11 Sol-approved packages below bill
more net lift than the entire corpus low-side delta; without them v5 would sit
about 5% *below* v4 at the low end.

**Evidence.** 13 v4 candidates had `verification_status: rejected` from 2f and
still exist as v5 candidates; Sol approved 11, rejected 2 (both exteriors).
For kitchens 2f typically *confirmed* most individual issues (worn cabinet
faces, basic white range, wall marks) but rejected the *package* because the
room reads as updated ("current white shaker cabinetry, stone-look/granite
counters, gray plank flooring"); Terra read the same photos as "basic white
cabinetry, laminate-style counters". Two VLM passes contradict each other on
the same images.

<!-- GEN:p1_packages -->
| id | property | package | v4 tier / price | v5 tier | Sol | v5 effective | children standalone | net lift | v4 2f | 2f says |
|---|---|---|---|---|---|---:|---:|---:|---|---|
| P01 | redfin_10803207 | `bathroom_modernization|bathroom_primary` | partial_rehab $5,792/14,481 | partial_rehab | approve | $5,792/14,666 | $1,557/14,666 | $4,235/0 | rejected (6✓/1✗ issues) | Some older-style elements are visible, including a textured framed shower enclosure, traditional faucet/vanity styling, square beige floor tile, a pedestal sink, and a visually busy brown tile-and-mo… |
| P02 | redfin_10806500 | `exterior_repair|exterior_primary` | repair_light $693/3,463 | repair_light | approve | $693/3,463 | $277/2,424 | $416/1,039 | rejected (1✓/1✗ issues) | The porch soffit/fascia shows visible weathering and discoloration, but no clear peeling, rot, detachment, or other material failure is visible. The porch-front finish appears intact, and the overall… |
| P03 | redfin_10806500 | `kitchen_modernization|kitchen_primary` | full_rehab $20,780/48,486 | full_rehab | approve | $20,780/48,486 | $2,450/25,415 | $18,330/23,071 | rejected (6✓/3✗ issues) | The kitchen has predominantly updated-looking white cabinetry, stone-look counters, a mosaic-tile backsplash, and a stainless range, which does not support a full modernization package. Visible suppo… |
| P04 | redfin_11077450 | `kitchen_modernization|kitchen_primary` | full_rehab $25,415/59,303 | full_rehab | approve | $25,415/59,303 | $2,614/26,632 | $22,801/32,671 | rejected (8✓/6✗ issues) | The kitchen predominantly shows current white shaker cabinetry, stone-look/granite counters, and gray plank flooring, which does not support a kitchen-wide modernization package. Visible issues inclu… |
| P05 | redfin_11185681 | `exterior_repair|exterior_primary` | repair_heavy $4,372/15,740 | repair_heavy | approve | $4,372/15,740 | $476/3,691 | $3,896/12,049 | rejected (5✓/0✗ issues) | Porch ceiling/soffit, siding, porch boards, and rail top show visible age, discoloration, and peeling/weathered finish. However, the visible siding, trim, porch structure, railings, and decking appea… |
| P06 | redfin_11185681 | `kitchen_modernization|kitchen_primary` | full_rehab $26,234/61,213 | full_rehab | approve | $26,234/61,213 | $2,661/26,862 | $23,573/34,351 | rejected (11✓/5✗ issues) | The photos show generally updated gray shaker cabinetry, mosaic backsplash, and wood-look flooring, which contradict a kitchen-wide modernization package. Visible issues include wear and scuffing on … |
| P07 | redfin_125970550 | `exterior_repair|exterior_primary` | repair_heavy $4,427/15,937 | repair_heavy | reject | $0/0 | $0/0 | $0/0 | rejected (6✓/1✗ issues) | The house siding and visible trim appear generally intact, with no clearly visible rot, missing sections, or material failure. The rear deck/rails, siding, eaves, lower wall, brick steps, and concret… |
| P08 | redfin_126418713 | `exterior_repair|exterior_primary` | repair_light $733/3,664 | repair_light | reject | $0/0 | $0/0 | $0/0 | rejected (4✓/0✗ issues) | Soffit/fascia finish appears discolored and the concrete entry steps/landing appear stained and worn, but the visible exterior envelope is intact: continuous soffit/fascia, sound painted brick, and n… |
| P09 | redfin_127468088 | `exterior_repair|exterior_primary` | repair_heavy $3,743/13,476 | repair_heavy | approve | $3,743/13,476 | $371/2,980 | $3,372/10,496 | rejected (8✓/2✗ issues) | Porch ceilings visibly appear yellowed/stained, and the concrete porch/walk shows wear and a patched or roughened area. However, visible siding is intact without clear material failure, and the paint… |
| P10 | redfin_25809814 | `bedroom_modernization|bedroom_2` | full_rehab $4,316/12,948 | full_rehab | approve | $4,316/12,948 | $782/3,605 | $3,534/9,343 | rejected (0✓/4✗ issues) | The bedroom shows clean neutral paint, smooth ceiling, a modern light fixture, intact white trim and paneled doors, and clean contemporary carpet. No visibly dated or worn flooring, doors, or trim is… |
| P11 | redfin_25809814 | `bedroom_modernization|bedroom_3` | full_rehab $4,316/12,948 | full_rehab | approve | $4,316/12,948 | $782/3,605 | $3,534/9,343 | rejected (1✓/4✗ issues) | The bedroom shows clean neutral paint, smooth ceilings, intact white trim, and clean carpet without visible wear or staining. Mirrored sliding closet doors are visible, but this isolated dated elemen… |
| P12 | redfin_25809814 | `kitchen_modernization|kitchen_primary` | full_rehab $25,896/60,425 | partial_rehab | approve | $12,948/30,212 | $2,158/22,443 | $10,790/7,769 | rejected (8✓/1✗ issues) | White raised-panel cabinetry and dark laminate-style counters are visible, along with an unfinished open upper-cabinet area. However, the kitchen otherwise presents predominantly updated finishes, in… |
| P13 | redfin_80990371 | `exterior_repair|exterior_primary` | repair_heavy $3,739/13,462 | repair_light | approve | $748/3,739 | $299/2,618 | $449/1,121 | rejected (3✓/1✗ issues) | Minor staining is visible at the right eave/fascia area, and the brick entry steps appear weathered/soiled. However, the visible siding, soffits, fascia, trim, railings, and brickwork appear generall… |

13 candidates 2f rejected; Sol approved 11; those bill $109,357/276,194 effective, net lift over children $94,930/141,253. (v4 tier/price = what v4 would have billed had 2f confirmed; two of the Sol-approved cases are also tier demotions.)
<!-- /GEN:p1_packages -->

**What to look at.** Photo sheet §P1, items P01–P13: the 2f review photos
(what the package-level verifier saw) and each child's Terra photos and
rationale. Start with redfin_11077450 kitchen (the +156% headline),
redfin_25809814 bedrooms 2/3 and kitchen (+195%), redfin_10806500 and
redfin_11185681 kitchens. For each, record in the worksheet:
`package_warranted` / `not_warranted` / `unsure`.

**Options.**
- (a) Accept v5 as designed: Terra's per-condition truth is the only photo
  gate; Sol is coherence-only. Consequence: v5 forms full kitchen/bedroom
  packages whenever enough subjective "dated/style" conditions pass Terra,
  even in rooms that look updated. More packages (Steven's goal 2) at the risk
  of package-level hallucination (goal 1).
- (b) Add a bounded **package-warrant check** inside v5: one Terra-style call
  per candidate with the room's photos and a closed answer
  (`warranted` / `not_warranted` / `cannot_assess`, one sentence), used only
  as a veto on package application (children stay standalone). This reinstates
  what 2f did, under v5's invariants (no new work, no prices). Cost ≈ the
  former 2f budget (~37k Terra tokens/listing).
- (c) Tighten Terra for subjective style items: require ≥2 distinct photos or
  explicit "clearly dated relative to current finishes" wording for
  `*_dated_style` / `outdated_*_finishes` / `dated_interior_trim`; P2's by-item
  table shows whether the disagreements concentrate there. Cost: prompt +
  rubric change; its effect can only be measured by re-running Terra on the
  stored conditions (~51k Terra tokens/listing), not by the offline replay.
- (d) Deterministic driver rules: full-rehab tiers only when a *damage* or
  *replacement* driver is present, not style alone. Cost: code only; effect
  measurable by the existing offline replay.

**Recommendation (mine).** Decide after the P1/P2 photo passes. If Steven
judges most of P01–P13 as "not warranted", build (b) before cutover — it is
the smallest change that restores parity with v4's behavior; its effect is
measurable once the §2 "re-decide" replay mode exists, with live Terra calls
for the veto (~37k tokens/listing ≈ 0.66M for the 18-listing canary, inside
one 2.5M Terra day). Otherwise ship (a) and put (c)/(d) on the calibration
thread. Either way it is not a mechanical-gate failure and v4 remains the
rollback authority.

**What closes it.** Worksheet P01–P13 filled; a one-line decision in §9;
if (b): a design note + a re-decide replay (live veto calls) showing the
veto's effect on the stored canary.

**Dependencies.** P2 (condition-level truth) informs this; P4 variance work
touches the same Sol call sites.

---

## 4. Packet P2 — Condition-level truth: which model was right (Terra vs 2f)

**Question.** On the conditions where v4's 2f and v5's Terra disagree about
the *same photo*, which one is correct — and does that concentrate on a
particular kind of condition?

**Why it matters.** This is the cleanest per-condition hallucination
measurement available without new provider calls: both models looked at the
same listing photos. Direction A (Terra supported, 2f rejected) is v5's
hallucination exposure; direction B (2f confirmed, Terra unsupported /
cannot_assess) is v5's over-rejection exposure.

<!-- GEN:p2_summary -->
Agreement matrix at the **v4 issue level** (each of v4's 2f per-issue verdicts mapped to the v5 condition that absorbed that issue; one condition can carry several issues):

| | Terra supported | Terra unsupported | Terra cannot_assess | no v5 condition |
|---|---:|---:|---:|---:|
| 2f confirmed | 766 | 38 | 4 | 0 |
| 2f rejected | 43 | 38 | 2 | 0 |

Disagreements: **85 v4 issue-level verdicts on 68 distinct v5 conditions** — **32** Terra-supported / 2f-rejected (direction A), **36** 2f-confirmed / Terra-not (direction B). The sheet and the worksheet are keyed per condition (one photo set, one Terra verdict); the v4 issues are listed inside each card.

By catalog item, counted per condition (A / B):

| catalog item | A: Terra sup, 2f rej | B: 2f con, Terra not |
|---|---:|---:|
| `paint_refresh_recommended` | 1 | 6 |
| `dated_interior_trim` | 4 | 3 |
| `peeling_or_discolored_paint` | 4 | 1 |
| `brick_weathered_or_discolored` | 1 | 3 |
| `older_flooring_style` | 2 | 2 |
| `hard_flooring_scratched_or_worn` | 1 | 2 |
| `baseboard_wear_scuffs` | 0 | 3 |
| `cabinets_worn_finish` | 0 | 2 |
| `vanity_worn_finish` | 1 | 1 |
| `vinyl_linoleum_torn_or_lifted` | 0 | 2 |
| `damaged_or_rotted_siding_or_trim` | 1 | 1 |
| `exterior_siding_discoloration_fading` | 1 | 1 |
| `soffit_or_porch_ceiling_weathered` | 1 | 1 |
| `outdated_kitchen_finishes` | 2 | 0 |
| `worn_or_stained_flooring` | 2 | 0 |
| `cabinets_dated_style` | 2 | 0 |
| `patio_or_porch_surface_wear` | 2 | 0 |
| `bath_fixtures_stained_or_worn` | 2 | 0 |
| `outdated_bathroom_finishes` | 2 | 0 |
| `hard_flooring_broken_or_warped` | 0 | 1 |
| `vanity_countertop_worn` | 0 | 1 |
| `cabinets_damaged_or_water_stained` | 0 | 1 |
| `tub_surround_or_shower_pan_damage` | 0 | 1 |
| `dated_wood_paneling` | 0 | 1 |
| `popcorn_or_acoustic_ceiling_texture` | 0 | 1 |
| `soffit_or_porch_ceiling_failed` | 0 | 1 |
| `worn_or_stained_carpet` | 0 | 1 |
| `vintage_tile_pattern_style` | 1 | 0 |
| `appliances_dated_or_basic` | 1 | 0 |
| `dated_interior_doors` | 1 | 0 |

Single-photo share of v5 work (run_1): {'one_photo': 357, 'multi_photo': 300} active work items; one-photo items carry $82,637/713,757 of $171,031/1,594,286 of active work-item dollars (pre-package, standalone-priced — not the headline).
<!-- /GEN:p2_summary -->

**What to look at.** Photo sheet §P2 — each card is one v5 condition: its
photo(s), the upstream issue sentence(s) with 2f's per-issue verdict, 2f's
package-level summary, Terra's sentence and the v5 outcome. Record
`terra_correct` / `2f_correct` / `both_partly` / `cannot_tell` per card. All
68 cards is roughly 45 minutes; if sampling, do all of direction A (it is what
v5 bills) and half of direction B.

**What the answer feeds.** A per-model precision estimate; whether Terra's
prompt v1 needs a stricter rubric for "dated/style" items versus "damage"
items (P1 option c); whether the `route_excluded_generic` / `no_action`
routes are catching the right things.

**Options after the session.**
- If Terra is mostly right: keep Terra prompt v1; P1 leans to (a).
- If 2f is mostly right on direction A: tighten Terra for the offending
  catalog items (c) and/or add the package-warrant veto (b).
- If 2f is mostly right on direction B: Terra is over-rejecting real
  conditions — review the `cannot_assess` routing and evidence-view thresholds.

**What closes it.** Worksheet C-items filled; a short tally by direction and
by catalog item written into §9.

<!-- GEN:p2_list -->
| id | property | item @ unit | photos | v4 issues (2f) | Terra | v5 outcome |
|---|---|---|---|---|---|---|
| C001 | redfin_10803207 | `hard_flooring_scratched_or_worn` @ living_room_primary | photo_007.jpg | 1 confirmed | unsupported | excluded |
| C002 | redfin_10952874 | `baseboard_wear_scuffs` @ bathroom_primary | photo_032.jpg | 1 confirmed | unsupported | excluded |
| C003 | redfin_10952874 | `baseboard_wear_scuffs` @ bedroom_4 | photo_022.jpg, photo_025.jpg, photo_030.jpg, photo_034.jpg | 2 confirmed | unsupported | excluded |
| C004 | redfin_10952874 | `cabinets_worn_finish` @ kitchen_primary | photo_015.jpg | 1 confirmed | unsupported | excluded |
| C005 | redfin_10952874 | `hard_flooring_broken_or_warped` @ bedroom_3 | photo_005.jpg | 1 confirmed | unsupported | excluded |
| C006 | redfin_10952874 | `paint_refresh_recommended` @ bedroom_primary | photo_001.jpg | 1 confirmed | unsupported | excluded |
| C007 | redfin_10952874 | `paint_refresh_recommended` @ kitchen_primary | photo_015.jpg | 1 confirmed | unsupported | excluded |
| C008 | redfin_10952874 | `vanity_countertop_worn` @ bathroom_primary | photo_013.jpg | 1 confirmed | unsupported | excluded |
| C009 | redfin_10952874 | `vanity_worn_finish` @ bathroom_primary | photo_032.jpg | 1 confirmed | unsupported | excluded |
| C010 | redfin_11000447 | `vinyl_linoleum_torn_or_lifted` @ bedroom_1 | photo_003.jpg | 1 confirmed | unsupported | excluded |
| C011 | redfin_11000447 | `vinyl_linoleum_torn_or_lifted` @ kitchen_primary | photo_004.jpg | 1 confirmed | unsupported | excluded |
| C012 | redfin_11077450 | `brick_weathered_or_discolored` @ exterior_primary | photo_001.jpg, photo_008.jpg | 2 confirmed | unsupported | excluded |
| C013 | redfin_11077450 | `damaged_or_rotted_siding_or_trim` @ exterior_primary | photo_008.jpg | 1 confirmed | unsupported | excluded |
| C014 | redfin_11079485 | `dated_interior_trim` @ bathroom_primary | photo_005.jpg | 1 confirmed | unsupported | excluded |
| C015 | redfin_11185681 | `dated_interior_trim` @ living_room_primary | photo_006.jpg, photo_044.jpg | 2 confirmed | unsupported | excluded |
| C016 | redfin_11185681 | `older_flooring_style` @ bedroom_5 | photo_046.jpg | 1 confirmed | unsupported | excluded |
| C017 | redfin_125779232 | `cabinets_damaged_or_water_stained` @ kitchen_primary | photo_002.jpg | 1 confirmed | unsupported | excluded |
| C018 | redfin_125970550 | `tub_surround_or_shower_pan_damage` @ bathroom_primary | photo_017.jpg | 1 confirmed | unsupported | excluded |
| C019 | redfin_126418713 | `hard_flooring_scratched_or_worn` @ bedroom_4 | photo_044.jpg, photo_048.jpg | 1 confirmed | unsupported | excluded |
| C020 | redfin_126418713 | `paint_refresh_recommended` @ bedroom_2 | photo_028.jpg | 1 confirmed | cannot_assess | inspection |
| C021 | redfin_126418713 | `paint_refresh_recommended` @ bedroom_4 | photo_043.jpg, photo_046.jpg, photo_047.jpg | 1 confirmed | unsupported | excluded |
| C022 | redfin_166147710 | `dated_wood_paneling` @ kitchen_primary | photo_011.jpg | 1 confirmed | unsupported | excluded |
| C023 | redfin_166147710 | `exterior_siding_discoloration_fading` @ exterior_primary | photo_006.jpg, photo_019.jpg | 3 confirmed | unsupported | excluded |
| C024 | redfin_166147710 | `popcorn_or_acoustic_ceiling_texture` @ bedroom_3 | photo_040.jpg | 1 confirmed | unsupported | excluded |
| C025 | redfin_25809814 | `cabinets_worn_finish` @ kitchen_primary | photo_011.jpg | 1 confirmed | unsupported | excluded |
| C026 | redfin_80877597 | `paint_refresh_recommended` @ bedroom_1 | photo_002.jpg | 1 confirmed | unsupported | excluded |
| C027 | redfin_80925528 | `brick_weathered_or_discolored` @ exterior_primary | photo_002.jpg, photo_014.jpg, photo_022.jpg, photo_023.jpg, photo_034.jpg, photo_035.jpg | 1 confirmed | unsupported | excluded |
| C028 | redfin_80925528 | `soffit_or_porch_ceiling_weathered` @ exterior_primary | photo_035.jpg | 1 confirmed | cannot_assess | inspection |
| C029 | redfin_80990371 | `baseboard_wear_scuffs` @ living_room_primary | photo_028.jpg | 1 confirmed | unsupported | excluded |
| C030 | redfin_80990371 | `brick_weathered_or_discolored` @ exterior_primary | photo_006.jpg, photo_010.jpg | 2 confirmed | unsupported | excluded |
| C031 | redfin_80990371 | `dated_interior_trim` @ living_room_primary | photo_022.jpg, photo_028.jpg, photo_029.jpg | 1 confirmed | unsupported | excluded |
| C032 | redfin_80990371 | `paint_refresh_recommended` @ living_room_primary | photo_004.jpg, photo_016.jpg, photo_017.jpg | 1 confirmed | unsupported | excluded |
| C033 | redfin_80990371 | `peeling_or_discolored_paint` @ bedroom_1 | photo_003.jpg | 1 confirmed | unsupported | excluded |
| C034 | redfin_81000709 | `older_flooring_style` @ bedroom_3 | photo_024.jpg | 1 confirmed | cannot_assess | inspection |
| C035 | redfin_81000709 | `soffit_or_porch_ceiling_failed` @ exterior_primary | photo_001.jpg | 1 confirmed | cannot_assess | inspection |
| C036 | redfin_81000709 | `worn_or_stained_carpet` @ bedroom_3 | photo_022.jpg | 1 confirmed | unsupported | excluded |
| C037 | redfin_10806500 | `outdated_kitchen_finishes` @ kitchen_primary | photo_004.jpg | 1 rejected | supported | accepted_for_work $1,385/13,853 |
| C038 | redfin_10806500 | `peeling_or_discolored_paint` @ kitchen_primary | photo_004.jpg | 1 rejected | supported | accepted_for_work $139/1,732 |
| C039 | redfin_10806500 | `soffit_or_porch_ceiling_weathered` @ exterior_primary | photo_001.jpg | 1 rejected | supported | accepted_for_work $277/2,424 |
| C040 | redfin_10952874 | `hard_flooring_scratched_or_worn` @ bedroom_4 | photo_025.jpg, photo_029.jpg | 1 rejected | supported | accepted_for_work $202/3,375 |
| C041 | redfin_10952874 | `worn_or_stained_flooring` @ bedroom_3 | photo_005.jpg, photo_007.jpg | 2 rejected | supported | accepted_for_work $119/593 |
| C042 | redfin_10952874 | `worn_or_stained_flooring` @ bedroom_4 | photo_021.jpg, photo_022.jpg, photo_024.jpg, photo_025.jpg, photo_028.jpg, photo_034.jpg | 1 rejected | supported | accepted_for_work $119/593 |
| C043 | redfin_11000447 | `peeling_or_discolored_paint` @ living_room_primary | photo_002.jpg, photo_005.jpg | 1 rejected | supported | accepted_for_work $152/1,901 |
| C044 | redfin_11000447 | `vanity_worn_finish` @ bathroom_primary | photo_008.jpg | 1 rejected | supported | accepted_for_work $380/4,563 |
| C045 | redfin_11077450 | `cabinets_dated_style` @ kitchen_primary | photo_016.jpg | 1 rejected | supported | accepted_for_work $1,694/16,944 |
| C046 | redfin_11077450 | `outdated_kitchen_finishes` @ kitchen_primary | photo_014.jpg, photo_016.jpg, photo_017.jpg | 2 rejected | supported | accepted_for_work $1,694/16,944 |
| C047 | redfin_11077450 | `patio_or_porch_surface_wear` @ exterior_primary | photo_008.jpg, photo_033.jpg | 1 rejected | supported | accepted_for_work $41/203 |
| C048 | redfin_11077450 | `peeling_or_discolored_paint` @ kitchen_primary | photo_015.jpg | 1 rejected | supported | accepted_for_work $169/2,118 |
| C049 | redfin_11079485 | `vintage_tile_pattern_style` @ bathroom_primary | photo_005.jpg | 1 rejected | supported | accepted_for_work $256/1,278 |
| C050 | redfin_11185681 | `cabinets_dated_style` @ kitchen_primary | photo_002.jpg, photo_024.jpg | 2 rejected | supported | accepted_for_work $1,749/17,489 |
| C051 | redfin_11185681 | `dated_interior_trim` @ kitchen_primary | photo_024.jpg, photo_025.jpg | 1 rejected | supported | accepted_for_work $92/460 |
| C052 | redfin_125779232 | `bath_fixtures_stained_or_worn` @ bathroom_primary | photo_003.jpg, photo_007.jpg | 1 rejected | supported | accepted_for_work $1,092/10,924 |
| C053 | redfin_125779232 | `damaged_or_rotted_siding_or_trim` @ exterior_primary | photo_015.jpg, photo_018.jpg | 1 rejected | supported | accepted_for_work $364/5,826 |
| C054 | redfin_125779232 | `dated_interior_trim` @ bathroom_primary | photo_003.jpg | 1 rejected | supported | accepted_for_work $76/383 |
| C055 | redfin_125779232 | `outdated_bathroom_finishes` @ bathroom_primary | photo_003.jpg, photo_007.jpg, photo_008.jpg | 4 rejected | supported | accepted_for_work $1,092/10,924 |
| C056 | redfin_125779232 | `patio_or_porch_surface_wear` @ exterior_primary | photo_016.jpg | 1 rejected | supported | accepted_for_work $35/175 |
| C057 | redfin_126224899 | `bath_fixtures_stained_or_worn` @ bathroom_primary | photo_003.jpg, photo_047.jpg | 1 rejected | supported | accepted_for_work $1,182/11,816 |
| C058 | redfin_126224899 | `brick_weathered_or_discolored` @ exterior_primary | photo_010.jpg | 1 rejected | supported | accepted_for_work $87/433 |
| C059 | redfin_126418713 | `outdated_bathroom_finishes` @ bathroom_primary | photo_003.jpg, photo_039.jpg, photo_040.jpg, photo_041.jpg, photo_042.jpg | 3 rejected | supported | accepted_for_work $1,099/10,993 |
| C060 | redfin_127468088 | `exterior_siding_discoloration_fading` @ exterior_primary | photo_019.jpg, photo_020.jpg, photo_022.jpg, photo_023.jpg, photo_024.jpg, photo_025.jpg, photo_026.jpg | 1 rejected | supported | accepted_for_work $36/180 |
| C061 | redfin_127468088 | `paint_refresh_recommended` @ bedroom_3 | photo_035.jpg, photo_039.jpg | 1 rejected | supported | accepted_for_work $374/5,241 |
| C062 | redfin_25809814 | `appliances_dated_or_basic` @ kitchen_primary | photo_011.jpg | 1 rejected | supported | accepted_for_work $432/5,179 |
| C063 | redfin_25809814 | `dated_interior_doors` @ bedroom_2 | photo_017.jpg | 1 rejected | supported | accepted_for_work $91/454 |
| C064 | redfin_25809814 | `dated_interior_trim` @ bedroom_2 | photo_017.jpg, photo_018.jpg | 2 rejected | supported | accepted_for_work $91/454 |
| C065 | redfin_25809814 | `dated_interior_trim` @ bedroom_3 | photo_021.jpg, photo_023.jpg | 2 rejected | supported | accepted_for_work $91/454 |
| C066 | redfin_25809814 | `older_flooring_style` @ bedroom_2 | photo_017.jpg | 1 rejected | supported | accepted_for_work $600/2,697 |
| C067 | redfin_25809814 | `older_flooring_style` @ bedroom_3 | photo_021.jpg, photo_025.jpg | 2 rejected | supported | accepted_for_work $600/2,697 |
| C068 | redfin_81000709 | `peeling_or_discolored_paint` @ bedroom_3 | photo_022.jpg, photo_030.jpg | 1 rejected | supported | accepted_for_work $170/2,124 |
<!-- /GEN:p2_list -->

---

## 5. Packet P3 — Multi-bathroom expansion

**Question.** When a listing says 2–3 bathrooms and the photos cluster into
several bathroom "surrogates", should the estimate bill a bathroom
modernization once (v5) or once per bathroom that shows dated conditions
(v4's expansion)?

**Why it matters.** Both engines merge every bathroom surrogate into one
estimate unit (`repeated_bathroom_surrogates_merged_conservatively`; v5 marks
each bath condition `identity_ambiguous`, reason
`bathroom_metadata_allows_two_but_weak_evidence_merged`). v4 then re-expands
its confirmed modernization package per qualifying surrogate; v5 does not.
Corpus effect −$58,438 / −$146,095 across six properties. The Session 9
comparator keyed v4 packages by `type|unit`, so the copies collapsed into one
key and the review auto-accepted all six pairs as Tier-1 "key migration, no
package change". The expansion is mentioned only in two headline explanations
(125779232, 80877597 — as "duplicated v4 rows", which mischaracterizes a
deliberate v4 feature); 125970550 and 126224899 stayed under the 15% headline
threshold, so nothing a reviewer reads describes it for them. It is therefore
an undecided question, not an accepted one.

<!-- GEN:p3_multibath -->
| id | property | listing baths | surrogates (photos) | v4 signal / expanded | v4 bath-mod pkgs | v5 bath-mod | v4−v5 bath-mod $ |
|---|---|---:|---|---|---|---|---:|
| B01 | redfin_10803207 | 2.5 | bathroom_1(2); bathroom_2(1); bathroom_3(2); bathroom_4(1) | False / False (no_confirmed_bathroom_modernization_package) | 0: none applied | 1 | $-5,792/-14,666 |
| B02 | redfin_10952874 | 1.5 | bathroom_1(1); bathroom_2(2) | False / False (signal_cold) | 1: bathroom_2 | 1 | $0/12,842 |
| B03 | redfin_11185681 | 3 | bathroom_1(2); bathroom_2(3); bathroom_3(2); bathroom_4(2) | True / True (expanded) | 3: bathroom_1, bathroom_2, bathroom_3 | 1 | $13,992/34,978 |
| B04 | redfin_125779232 | 2.0 | bathroom_1(1); bathroom_2(2) | True / True (expanded) | 2: bathroom_1, bathroom_2 | 1 | $5,826/14,566 |
| B05 | redfin_125970550 | 3.0 | bathroom_1(1); bathroom_2(1); bathroom_3(1); bathroom_4(1) | True / True (expanded) | 3: bathroom_2, bathroom_3, bathroom_4 | 1 | $14,166/35,416 |
| B06 | redfin_126224899 | 2.5 | bathroom_1(1); bathroom_2(1); bathroom_3(1); bathroom_4(2) | True / True (expanded) | 3: bathroom_1, bathroom_3, bathroom_4 | 1 | $12,604/31,510 |
| B07 | redfin_126418713 | 2.0 | bathroom_1(1); bathroom_2(4); bathroom_3(1) | True / True (expanded) | 2: bathroom_1, bathroom_2 | 1 | $5,863/14,658 |
| B08 | redfin_166147710 | 2 | bathroom_1(1); bathroom_2(1); bathroom_3(1) | False / False (signal_cold) | 1: bathroom_2 | 1 | $0/0 |
| B09 | redfin_25809814 | 2 | bathroom_1(1); bathroom_2(2); bathroom_3(1) | False / False (signal_cold) | 1: bathroom_1 | 1 | $0/3,561 |
| B10 | redfin_80877597 | 2 | bathroom_1(1); bathroom_2(1); bathroom_3(1) | True / True (expanded) | 2: bathroom_1, bathroom_2 | 1 | $5,987/14,967 |
<!-- /GEN:p3_multibath -->

**What to look at.** Photo sheet §P3 — one strip per surrogate. For each
property write how many *distinct* bathrooms you see (tile, vanity, layout,
window tell them apart) and whether each one looks dated, then `bill:
per_bathroom | once | unsure`. The four non-expanded properties (10803207,
10952874, 166147710, 25809814) are useful controls: v4 saw multiple
surrogates there too but did not expand.

**Options.**
- (a) Keep v5: one unit per listing bathroom group, ambiguity explicit.
  Under-counts genuinely multi-bath homes; consistent with the plan's
  "ambiguous identity is represented explicitly, no new room-understanding
  system in this migration".
- (b) Port the *shape* of v4's expansion deterministically into v5 **package
  application**: if listing baths ≥ k and k surrogates each carry
  Terra-supported modernization conditions, apply the bathroom package k times
  (conditions stay merged; ledger records the expansion). Auditable and small.
  Note it cannot reuse v4's `bathroom_room_count_signal` — that signal is a
  vote over Pass 2f's `visible_room_count` answer, which v5 never produces; the
  v5 driver would be surrogate count × listing bath cap × Terra-supported
  conditions per surrogate.
- (c) Resolve distinct bathrooms *before* Terra (surrogate clustering capped
  by listing bath count ⇒ per-bathroom estimate units ⇒ per-unit conditions
  and packages). Correct in principle (mine); it is the "new room-understanding
  system" the plan deferred.
- (d) Crude: scale one package by listing bath count. Not recommended (mine).

**Recommendation (mine).** Decide from the strips. If the surrogates are
mostly real, distinct, dated bathrooms (as v4's per-surrogate confirmed-issue
counts suggest for 11185681 and 125970550), do (b) — it is a one-session
deterministic change with a replay-able effect; otherwise (a) with the
decision recorded. Either way, fix the comparator so duplicate v4 package keys
are never collapsed again (P5).

**What closes it.** Worksheet B-items filled; decision in §9; if (b): a
design note for the application rule and a replay on the stored canary.

---

## 6. Packet P4 — Sol consistency and split handling

**Question.** Which package-coherence decisions should stay with Sol, and
which should become rules?

**Why it matters.** A Sol flip changes a property's dollars silently (a
rejected package drops to its children's standalone prices — fails safe on the
low side, but it is variance the estimate carries into production), and the
same grouping being approved on one property and rejected on the next is
indefensible to a user comparing listings.

**Evidence.** See §2 for the measured flip rate. The concrete cases:

<!-- GEN:p4_sol -->
Identical-children candidates in both replicas: **29 same decision, 2 flipped**; 76 same-key candidates had different children (upstream churn; not comparable).

| property | package | children | run 1 | run 2 |
|---|---|---|---|---|
| redfin_127468088 | `bedroom_repair|bedroom_3` | [['damaged_drywall_or_cracks', 'wall_scuffs_marks_or_dents']] | approve — The single drywall patch item is coherent as a focused bedroom repair package. | reject — A single drywall-patching item does not constitute a grouped renovation package and should remain standalone. |
| redfin_166147710 | `bedroom_repair|bedroom_2` | [['hard_flooring_scratched_or_worn']] | approve — The flooring replacement is a coherent bedroom repair scope. | reject — A single flooring replacement should remain a standalone work item rather than a room package. |

Single-child candidates (run_1): {'approve': 23, 'reject': 2}. exterior_repair candidates (run_1): {'approve': 7, 'reject': 3}.

| property | package | children | Sol | rationale |
|---|---|---|---|---|
| redfin_10803207 | `bedroom_repair|bedroom_1` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | approve | The single bedroom flooring replacement is a coherent room-level repair package. |
| redfin_10806500 | `exterior_repair|exterior_primary` | repair[soffit_or_porch_ceiling_weathered] | approve | The weathered porch ceiling and fascia repair is a coherent exterior repair scope. |
| redfin_10952874 | `bedroom_repair|bedroom_3` | DRYWALL_PATCH[damaged_drywall_or_cracks,wall_scuffs_marks_or_dents] | approve | The bedroom drywall repair coherently belongs with the broader Bedroom 3 renovation scope. |
| redfin_11000447 | `living_repair|living_room_primary` | FLOORING_REPLACE[hard_flooring_broken_or_warped,hard_flooring_scratched_or_worn,older_flooring_style] | approve | The living-room flooring replacement is coherent with the broader living-room renovation scope. |
| redfin_11000447 | `bathroom_repair|bathroom_primary` | VANITY_REPLACE[vanity_damaged_or_water_stained,vanity_worn_finish] | approve | The vanity replacement is part of the same primary-bathroom renovation scope as the fixture modernization. |
| redfin_11000447 | `bedroom_repair|bedroom_2` | DRYWALL_PATCH[damaged_drywall_or_cracks,wall_scuffs_marks_or_dents] | approve | The Bedroom 2 drywall repairs form a coherent room-level repair package. |
| redfin_11077450 | `living_repair|living_room_primary` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | reject | A single flooring replacement does not form a broader living-room renovation package and should remain standalone. |
| redfin_11185681 | `bedroom_repair|bedroom_2` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | approve | This hardwood flooring work coherently joins the other room-level hardwood replacements as one property-wide flooring scope. |
| redfin_11185681 | `bedroom_repair|bedroom_3` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | approve | This hardwood flooring work coherently joins the other room-level hardwood replacements as one property-wide flooring scope. |
| redfin_11185681 | `bedroom_repair|bedroom_4` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | approve | This hardwood flooring work coherently joins the other room-level hardwood replacements as one property-wide flooring scope. |
| redfin_11185681 | `bedroom_repair|bedroom_1` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | approve | This hardwood flooring work coherently joins the other room-level hardwood replacements as one property-wide flooring scope. |
| redfin_11185681 | `living_repair|living_room_primary` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | approve | This hardwood flooring work coherently joins the bedroom-level hardwood replacements as one property-wide flooring scope. |
| redfin_11185681 | `bedroom_repair|bedroom_5` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | approve | This hardwood flooring work coherently joins the other room-level hardwood replacements as one property-wide flooring scope. |
| redfin_125970550 | `bedroom_repair|bedroom_3` | DRYWALL_PATCH[damaged_drywall_or_cracks,wall_scuffs_marks_or_dents] | reject | A single drywall patch item should remain standalone rather than constitute a bedroom package. |
| redfin_126418713 | `bedroom_repair|bedroom_2` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | approve | The bedroom flooring replacement coheres with the living-room flooring replacement as one multi-room flooring scope. |
| redfin_126418713 | `living_repair|living_room_primary` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | approve | The living-room flooring replacement coheres with the bedroom flooring replacement as one multi-room flooring scope. |
| redfin_127468088 | `bedroom_repair|bedroom_3` | DRYWALL_PATCH[damaged_drywall_or_cracks,wall_scuffs_marks_or_dents] | approve | The single drywall patch item is coherent as a focused bedroom repair package. |
| redfin_166147710 | `bedroom_repair|bedroom_3` | DRYWALL_PATCH[damaged_drywall_or_cracks,wall_scuffs_marks_or_dents] | approve | The drywall patch work forms a coherent bedroom repair scope. |
| redfin_166147710 | `bedroom_repair|bedroom_2` | FLOORING_REPLACE[hard_flooring_scratched_or_worn] | approve | The flooring replacement is a coherent bedroom repair scope. |
| redfin_25809814 | `bathroom_repair|bathroom_primary` | VANITY_REPLACE[vanity_dated_style] | approve | The primary-bathroom vanity replacement belongs with the broader modernization of the same bathroom. |
| redfin_80877597 | `bathroom_repair|bathroom_primary` | VANITY_REPLACE[vanity_dated_style] | approve | The vanity replacement clearly belongs with the other primary-bathroom modernization work. |
| redfin_80990371 | `bedroom_repair|bedroom_1` | DRYWALL_PATCH[damaged_drywall_or_cracks] | approve | The bedroom drywall repair coheres with the same-room flooring modernization as one renovation scope. |
| redfin_80990371 | `living_repair|dining_room_primary` | DRYWALL_PATCH[damaged_drywall_or_cracks,wall_scuffs_marks_or_dents] | approve | The dining-room wall patching is a coherent localized room repair package. |
| redfin_80990371 | `exterior_repair|exterior_primary` | repair[soffit_or_porch_ceiling_weathered] | approve | The weathered porch and eave repair is a coherent exterior repair scope. |
| redfin_81000709 | `bedroom_repair|bedroom_1` | MOISTURE_REMEDIATION[water_stain_ceiling] | approve | The bedroom moisture remediation is coherent as a focused room-level repair package. |
| redfin_10952874 | `exterior_repair|exterior_primary` | DECK_REFINISH[deck_surface_weathering]; DECK_REPAIR[damaged_or_unsafe_deck_or_porch] | approve | Deck repair and refinishing are directly related parts of one exterior repair scope. |
| redfin_11185681 | `exterior_repair|exterior_primary` | DECK_REFINISH[deck_surface_weathering]; repair[soffit_or_porch_ceiling_weathered]; PAINT_EXTERIOR[exterior_siding_discoloration_fading]; SURFACE_RESURFACE[patio_or_porch_surface_wear] | approve | The porch-surface, ceiling, trim, and siding work forms a coherent exterior weathering repair package. |
| redfin_125779232 | `exterior_repair|exterior_primary` | SURFACE_RESURFACE[patio_or_porch_surface_wear]; MASONRY_REPAIR[brick_weathered_or_discolored]; EXTERIOR_REPAIR[damaged_or_rotted_siding_or_trim]; PAINT_EXTERIOR[exterior_siding_discoloration_fading]; repair[soffit_or_porch_ceiling_weathered] | approve | The siding, eave, paint, masonry, and surface work coherently comprise a broad exterior repair package. |
| redfin_125970550 | `exterior_repair|exterior_primary` | SURFACE_RESURFACE[patio_or_porch_surface_wear]; MASONRY_REPAIR[brick_weathered_or_discolored]; PAINT_EXTERIOR[exterior_siding_discoloration_fading]; DECK_REFINISH[deck_surface_weathering] | reject | Patio, masonry, siding-paint, and deck work are independent exterior scopes rather than one coordinated repair package. split=[['wk1_0ac9cf03ab83c910'], ['wk1_6d09815977f12779'], ['wk1_ef07dbb159c3f59d'], ['wk1_f5256a3d045d8915']] |
| redfin_126224899 | `exterior_repair|exterior_primary` | repair[soffit_or_porch_ceiling_failed,soffit_or_porch_ceiling_weathered]; MASONRY_REPAIR[brick_weathered_or_discolored]; PAINT_EXTERIOR[exterior_siding_discoloration_fading]; DECK_REFINISH[deck_surface_weathering]; SURFACE_RESURFACE[patio_or_porch_surface_wear] | reject | The building-envelope finishes, masonry, deck, and concrete-surface work are independent exterior scopes. split=[['wk1_08298bb506e6fa2d', 'wk1_7702ad6314753c46'], ['wk1_52f3809f8d9c03ca'], ['wk1_eaed4845214f3553'], ['wk1_f625c6201df8bea3']] |
| redfin_126418713 | `exterior_repair|exterior_primary` | SURFACE_RESURFACE[patio_or_porch_surface_wear]; repair[soffit_or_porch_ceiling_weathered] | reject | Patio resurfacing and soffit or fascia repair are independent exterior scopes and should remain separate. split=[['wk1_9e09a4da3a46b1ed'], ['wk1_c18fb2be442da913']] |
| redfin_127468088 | `exterior_repair|exterior_primary` | PAINT_EXTERIOR[exterior_siding_discoloration_fading]; SURFACE_RESURFACE[patio_or_porch_surface_wear]; repair[soffit_or_porch_ceiling_weathered] | approve | The siding, concrete-surface, and porch or soffit work form a coherent exterior repair package. |
| redfin_80925528 | `exterior_repair|exterior_primary` | MASONRY_REPAIR[mortar_joints_deteriorated]; EXTERIOR_REPAIR[damaged_or_rotted_siding_or_trim]; SURFACE_RESURFACE[patio_or_porch_surface_wear]; DECK_REPAIR[damaged_or_unsafe_deck_or_porch] | approve | The masonry, trim, porch-surface, and step repairs form a coherent exterior repair scope. |
<!-- /GEN:p4_sol -->

The split case: on redfin_11079485 Sol rejected the kitchen candidate because
stair carpet had been unit-resolved into `kitchen_primary`, and its own
`split_groups` isolated a four-item kitchen core (cabinets, appliances,
countertop, floor). Sol's decision was `reject` with a split attached, so the
stored reason is `decision_rejected`; and splits are non-economic by design
anyway (`reconciliation.py`: an approve-with-split is `not_applied /
split_recommended`). Either way the whole $10,920/25,480 kitchen package fell
to $1,514/21,045 of standalone work — the −52% low headline on that property.

**Options.** §2 A–F, plus: make a Sol split produce sub-candidates that are
re-judged (one more Sol round, bounded) or applied deterministically when each
group has a driver.

**Recommendation (mine).** Rule for single-child candidates (the flips are
entirely there); rubric-structured Sol for multi-child; split → sub-package
with a driver check; then the Sol probe. Not a cutover blocker — every
rejection fails safe.

**What closes it.** A written rule set + a replay showing package counts and
dollars before/after on the stored canary; probe flip rate recorded in §9.

---

## 7. Packet P5 — Review-tooling gaps to fix before the next canary

These are deterministic and cheap; they are what let the findings above hide.

1. `tools/compare_renovation_architecture_cutover.py::_v4_packages` keys by
   `type|unit` and overwrites duplicates — key by `package_id` so expanded
   bathroom packages survive as separate review items.
2. The package review item carries type/unit/tier/status but **no dollars**:
   21 of the 78 Tier-1 "identical package" rows differ in effective price
   (net −$834/−$40,837; single rows up to ±$12.8k). Add effective low/high to
   the v4/v5 package rows so same-tier repricing is a review item.
3. Add v4's 2f `verification_status` (and evidence summary) to package review
   items so a reviewer sees model disagreement without re-deriving it.
4. Add a Sol flip detector to the stability item (identical children, different
   decision) — it is the only direct variance signal the canary yields for free.
5. `scripts/cluster_session9_reviews.py`: do not fold multi-surrogate packages
   into `package_key_migration`; add a "disagreement" tier.
6. **ERRATUM (landed 2026-08-27, Session B — this supersedes the original
   item 6 and is the correction of record).** The two Tier-2 headline
   explanations for redfin_125779232 (`rar1_56aa0653723ad643`) and
   redfin_80877597 (`rar1_907cf475274a84b9`) in
   `reports/renovation_architecture_session9_reviews_20260821.prefilled.json`
   describe the removed v4 bathroom-modernization allowance as "duplicated
   v4 rows" being canonicalized. That characterization is wrong. The rows
   are v4's per-surrogate expansion clones
   (`bathroom_modernization__bathroom_primary__bathroom_1` / `__bathroom_2`,
   both 2f-confirmed, identically priced by v4's naive cloning) — two
   distinct bathrooms, not redundancy — and the allowance v5 dropped is the
   P3/QP4 under-billing (v5 16/17 under vs the §10 human targets, gap +25),
   not cleanup. The dollar arithmetic in both explanations is correct and
   self-consistent; only the characterization is wrong, so the cutover
   decision does not reopen: dedup-driven deflation was already accepted,
   and the P3 option call already commissions QP4. These two properties
   re-file as QP4 gate evidence (Session E scoring vs the §10 targets)
   instead of benign-cleanup examples. Root cause of the mis-wording: the
   pre-P5 comparator keyed packages by type|unit, so both clones (same type,
   empty unit) collapsed into one review row — item 1 above, fixed in
   Session B — and a reviewer seeing one matching bathroom row per side
   reasonably inferred a duplicate. The frozen review file stays
   byte-identical: its sha256 is an integrity anchor in the session9
   handoffs, and the original suggestion of re-editing it is no longer
   viable — a post-P5 comparator re-run produces a different, unreviewed
   item set (967→1065 items, every package review id changed), so the
   "deterministic and offline re-run" this item originally described no
   longer re-blesses anything.

**Recommendation (mine).** Do 1–5 before the next canary; they change no
estimate output, only what the reviewer sees. **What closes it:** the five
changes landed with tests, and the next comparator run shows price-level
package items, un-collapsed v4 packages, 2f verdicts and a Sol-flip section.

---

## 8. Packet P6 — Tier-2 sign-off and single-photo opportunity drivers

The three Tier-2 policies still need Steven's explicit acceptance:

1. **Merge-dedup prices shared scope lower** (56 items, −$13,616 / −$231,270):
   one merged work item instead of stacked per-observation prices. Matches
   Steven's stated acceptance of dedup deflation.
2. **Single-item repricing, wider high** (16 items, −$7 / +$18,712):
   area-priced surfaces; provisional prices.
3. **v4 $0 → v5 priced** (5 items, +$1,751 / +$20,946). What is actually being
   approved here: v4 held these at $0 because a single photo is
   `insufficient_corroboration_for_opportunity_driver` (or the parent package
   was 2f-rejected); v5 prices them on Terra's verdict with one photo / one view.

<!-- GEN:p6_tier2 -->
| review id | property | item | v4 $0 reason | v5 price | photos | Terra |
|---|---|---|---|---:|---:|---|
| rar1_32834e11fec24cbe | redfin_10803207 | `dated_bathroom_flooring_style|bathroom_primary` | package_rejected | $217/2,534 | 1 | supported: Beige square bathroom floor tile is visible and appears intact with a dated style. |
| rar1_de0247255b9e220e | redfin_10803207 | `vanity_dated_style|bathroom_primary` | insufficient_corroboration_for_opportunity_driver | $362/4,344 | 1 | supported: A white paneled vanity with a molded sink top is visible and appears functional with a dated builder-grade style. |
| rar1_7a69e9ae5fbabd7f | redfin_126418713 | `vanity_dated_style|bathroom_primary` | insufficient_corroboration_for_opportunity_driver | $366/4,397 | 1 | supported: A wood bathroom vanity with an older cabinet style is clearly visible. |
| rar1_afda9927a9027fb6 | redfin_127468088 | `vanity_dated_style|bathroom_primary` | insufficient_corroboration_for_opportunity_driver | $374/4,492 | 1 | supported: An older-style wood vanity cabinet is clearly visible. |
| rar1_69d1a08831644ff0 | redfin_25809814 | `appliances_dated_or_basic|kitchen_primary` | package_rejected | $432/5,179 | 1 | supported: The kitchen shows a mixed set of basic-looking appliances, including an older-style microwave and range. |
<!-- /GEN:p6_tier2 -->

The broader version of (3) is the single-photo share of v5's work in the P2
summary above. **Question:** is Terra's verdict on one photo enough for an
opportunity-driver item, or should v5 keep v4's corroboration rule for that
class (withhold / inspection until a second distinct view)? **What to look
at:** photo sheet §P6 — the five items, one photo each, with Terra's sentence.
**Recommendation (mine):** accept 1 and 2 now; decide 3 together with P2 (the
photo session shows whether one-photo "dated" calls hold up). **What closes
it:** three lines in the §9 log (one per policy), and — if 3 is rejected — a
rule note for the corroboration requirement.

---

## 9. Order of work and decision log

Suggested sequence (each step is closable on its own):

1. **Photo pass A — P2 conditions** (sheet §P2, 68 cards, ~45 min). No code.
2. **Photo pass A′ — P1 packages** (sheet §P1, P01–P13, ~20–30 min). No code.
3. **Photo pass B — P3 bathrooms** (sheet §P3, 10 properties, ~30 min). No
   code.
4. **P1 decision** from A + A′; if (b), spec the package-warrant veto.
5. **P3 decision** from B; if (b), spec the expansion rule.
6. **P4**: write the single-child/split rules; add Sol rubric; build the
   "re-decide" replay mode; run the Sol probe (and the one-day Terra probe if
   the 6.4% proxy is judged material).
7. **P5** tooling fixes.
8. **P6** sign-off (1, 2 now; 3 with P2).
9. Cutover call. The mechanical gates are green; what gates cutover is
   whether P1/P3 outcomes are regressions Steven will not ship behind the v4
   rollback. Mechanics (plan §"Session 6" / "Rollback"): set
   `RENOVATION_ARCHITECTURE_MODE=new` (with `KIND_ONTOLOGY_VERSION=
   observation_kind_v2`), restart the persistent worker, smoke one listing;
   rollback = selector back to `current` + restart, no artifact rewrites.
   Which decisions force what: P4 rules, P3(b) and P1(d) are replayable
   offline on the stored canary (no new freeze); P1(b), P1(c) and any prompt
   change need live Terra calls and therefore a fresh freeze + new canary
   replicas (`docs/RUNBOOK_renovation_architecture_session_6.md` §1a, two
   separate days). Production throughput stays bounded by the Sol 250k/day
   ceiling (~29 listings/day at ~8.5k) and the Terra 2.5M/day ceiling shared
   with upstream passes (~10–12 listings/day all-service; see
   `docs/HANDOFF_renovation_architecture_token_budget.md` and
   `docs/analysis/session8_terra_budget_denominator.md`).

| packet | decision | date | by | notes |
|---|---|---|---|---|
| P1 package-warrant | evidence recorded — option call pending (Steven) | 2026-08-26 | analysis session | 22 P1 cards (census of the 2f-rejected, Sol-decided slice; 13 canary + 9 production): 11 `package_warranted` / 11 `not_warranted`. Sol approve (19): 9 warranted / 10 not; Sol reject (3): 2 / 1. Style-only drivers: 2 warranted / 3 not. `reports/review_analysis.md` §7 |
| P2 condition truth tally | evidence recorded — descriptive only | 2026-08-26 | analysis session | weighted broad-problem (unsupported+overstated) rate on billed conditions: canary 13.6% (95% band 6.8–26.6%, N=743), production 10.4% (5.1–32.5%, N=203); hard-false 10.5% / 10.4%; dirA census 12/32 and 8/19 broad; dirB targeted recovery: 20 of 51 Terra-rejected, 2f-confirmed claims judged supported by the human. Error mass concentrates in `degradation`-kind claims (66% canary / 83% production). `reports/review_analysis.md` §§3–5 |
| P3 multi-bathroom | evidence recorded — option call pending (Steven) | 2026-08-26 | analysis session | 17 multi-surrogate cards: billing judged 14 `per_bathroom` / 3 `once`; vs the human bathroom-count target v5 under-bills 16/17 (1 exact), aggregate gap +25 bathroom_modernization units; v4 on the same cards: 6 exact / 10 under / 1 over, gap +16. `reports/review_analysis.md` §10 |
| P4 Sol rules / probe | | | | |
| P5 tooling | | | | |
| P6 Tier-2 sign-off (1) | merge-dedup prices shared scope lower — 56 items, −$13,616 / −$231,270: **accepted** | 2026-08-21 | Steven | one merged work item instead of stacked per-observation prices; matches the stated acceptance of dedup deflation |
| P6 Tier-2 sign-off (2) | single-item repricing, wider high — 16 items, −$7 / +$18,712: **accepted** | 2026-08-21 | Steven | area-priced surfaces; prices provisional |
| P6 Tier-2 sign-off (3) | v4 $0 → v5 priced — 5 single-photo items, +$1,751 / +$20,946: **accepted provisionally** | 2026-08-21 | Steven | revisit with the P2 photo pass (single-photo opportunity drivers); the review file already records `accepted` — this line closes the governance gap |
| P6 Tier-2 sign-off (3) — P2-pass revisit | evidence recorded — the provisional acceptance is Steven's to confirm | 2026-08-26 | analysis session | the 5 single-photo items were reviewed blind in the P2 photo pass: 4 `terra_claim_supported`, 1 `terra_claim_unsupported` (appliances_dated_or_basic @ redfin_25809814, C062). `reports/review_analysis.md` §4 (p6 table) |
| Cutover | **Conditional GO** (independent review, `docs/DECISION_renovation_architecture_session9_cutover_20260821.md`); conditions C1–C5 closed 2026-08-21/22 per that record's §9 execution log | 2026-08-21 | Steven / review session | the flip itself is Steven's `npm run worker -- --premium` after C1–C5; production stays staged until then |
| Cutover — Task 4A (C1) | **Waived in writing**: the architecture cutover knowingly carries the Task 4A v4-on-v2 deltas (23 headline rows on 14 properties under catalog 3.1, `reports/kind_cutover_canary_catalog31_20260808.json`) as the transitional display until the FE adopts v5; unresolved-rate gate closed as option (a) | 2026-08-21 | Steven | `approved_headline_deltas` deliberately left `{}` — a waiver, not per-property approval; sign-off block appended to `reports/kind_cutover_canary_final_20260807.md` |
| Cutover — observation (C5) | 7 calendar days from enablement; operational invariants only; rollback after three consecutive ontology/resolver/writer failures, v5 job-fatal failures counted by code/category | 2026-08-21 | Steven | envelope: ~10–12 listings/day all-service (median 11, worst 6 under the 2.5M free quota); v5 failure = listing failure (no artifact); +~51k Terra / ~8.5k Sol per listing dark; budget guard ON |
| P1 package-warrant — option call | blanket package-level photo veto (option b) **rejected**; adopted = interior-modernization application gate (opportunity-only drivers → children standalone; option-d family) + Terra rubric v2 (option-c family); high-consequence VLM pilot reserved as fallback; FE adoption held until the wave-2 canary passes | 2026-08-26 | Steven (via proposals review) | evidence rows above + `docs/PROPOSALS_output_quality_improvements_20260826.md` (S1/S2/S6); P1 census: veto-class judges ~50% on the contested slice, interior-modernization 0/5 warranted |
| P3 multi-bathroom — option call | target = option (c) pre-Terra per-surrogate resolution; interim = option (b) constrained deterministic expansion (per-surrogate evidence + partitioned, re-priced children — never naive cloning), shipping **only** on the replay gate: exactness ≥ v4's 6/17 with over-billing ≤ v4's 1/17 vs the §10 human targets | 2026-08-26 | Steven (via proposals review) | `docs/PROPOSALS_output_quality_improvements_20260826.md` QP4 (S3/S9); §10 evidence: human per_bathroom 14/17, v5 under 16/17 gap +25 |
| P4 Sol rules — option call | scoped deterministic rules adopted: single-child standalone rule (blanket vs interior-repair scoping picked at implementation, one predicate either way) + Sol splits materialized as deterministically applied sub-packages; **no** multi-sample Sol voting; Sol probe rides the shared re-decide scaffolding later | 2026-08-26 | Steven (via proposals review) | `docs/PROPOSALS_output_quality_improvements_20260826.md` QP5/QP6 (S4/S5); flips 2/31 both single-child bedroom_repair; split case −52% headline, P07 human-warranted zeroed |
| P6 Tier-2 sign-off (3) — final | provisional single-photo acceptance **confirmed**; no blanket two-photo gate; consequence-triggered corroboration stays available via the design's escalation lane | 2026-08-26 | Steven (via proposals review) | p6 evidence 4/5 held (targeted, n=5); photo-bucket evidence contradicts across sources (`reports/review_analysis.md` §4) |

---

## Appendix — provenance

- Canary root `artifacts_canary/renovation_session9_20260818` (run_1 / run_2,
  `input_freeze.json` sha `f32f6812…4a973`).
- Report `reports/renovation_architecture_session9_canary_20260821.json`
  (sha `37694749…161b`), reviews
  `reports/renovation_architecture_session9_reviews_20260821.prefilled.json`
  (sha `1381fba2…45f8`), digest `reports/session9_review_digest_20260821.md`.
- Audit handoff: `docs/HANDOFF_renovation_architecture_session9_manual_review_complete.md`.
- Session 8 analyses: `docs/analysis/session8_floor_vs_sum_trace.md`,
  `…group_caps_measurement.md`, `…terra_budget_denominator.md`.
- Regenerate everything: `.venv\Scripts\python.exe scripts\build_session9_decision_packets.py`
  (the worksheet is not overwritten once it exists).
