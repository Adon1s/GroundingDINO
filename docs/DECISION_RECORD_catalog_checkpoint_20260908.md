# Decision record — bounded catalog policy checkpoint

Date: 2026-09-08. Status: **Steven's decisions, recorded. This document records human decisions and their
evidence; it is not itself an implementation authorization.** The exact operations these decisions authorize
are in `reports/catalog_audit_approvals_v2.json`; `reports/catalog_audit_approvals.json` (CAP-007,
`a725c89a…c871`) is preserved unedited and remains the authorization for CAP-007 only.

Lineage. Charter: `docs/catalog_audit_program/08_SESSION_BOUNDED_POLICY_CHECKPOINT.md`. Packet:
`docs/DECISION_PACKET_catalog_policy_checkpoint.md` (`befb086e…1b76e`) and
`reports/catalog_policy_checkpoint_packet.json` (`99a40d74…1637`). Human photo review:
`docs/catalog_audit_program/REVIEW_trim_photos_steven_20260908.md` (`399f376f…d26f33`). Ruled Session 6
bundle: `reports/catalog_audit_live_validation.json` (`ee1bcf62…4f5c`). Source manifest:
`reports/catalog_checkpoint_source_manifest.json` (`90d67401…28a6c`, see section 0). Implementation base:
`6e67eaa83887cf4d218100dcd060c07d3bd30522` on `catalog_audit_session4`.

Decisions D1 to D8 and the CAP-013 card adjudication were given by Steven in chat on 2026-09-08, in the
session that produced this record, in answer to explicitly evidenced alternatives. They are recorded here as
answers to stated questions, not as reconstructions of earlier rulings.

## 0. Hash drift, explained rather than silently reconciled

The packet pins `reports/catalog_checkpoint_source_manifest.json` at `bd4003f8…3b03`; on disk it is
`90d67401c9277b76d0af5d06d6684a18b59fda98f272c36755fcd3566cd28a6c`. Cause: the packet was generated at
`2026-09-08T16:40:19Z` and the source manifest was regenerated at `16:41:03Z`, the charter's "regenerated
last" step, after which it pins the final packet hash `99a40d74…1637` — which matches disk exactly. The two
artifacts pin each other; the manifest is the later one, so the packet's pin of it is stale by construction
while the manifest's pin of the packet is authoritative. This record and the approvals manifest pin the
current on-disk manifest hash. No artifact was edited to make the discrepancy disappear.

All other hashes in `docs/HANDOFF_catalog_completion_20260908.md` section 10 were re-verified unchanged.

## 1. Corrections to the packet (superseded on these points, not rewritten)

| # | Packet statement | What the record says | Correction carried forward |
|---|---|---|---|
| R1 | Section 2: the two CAP-013 candidate billings stop, "which Q-3 and A-2 already treat as withdrawals" | Q-3 (2026-09-05) withdrew `rc_9e889631af67` and `rc_d0ffde500a92`, and ruled `rc_ceeb17f8e242` a missing-trim **loss**, "not a supersession". A-2 recorded `rc_57c6faf79c7e` with CAP-013's cards, "not adjudicated". Steven 2026-09-08: kitchen loss **not accepted**, bathroom claim **not confirmed** | The four claim-exact billings resolve as two ruled withdrawals (Q-3), one loss now represented under D8 (kitchen), and one withdrawn as a claim the human declined to confirm (bathroom). Q-3 and A-2 never accepted the kitchen loss |
| R2 | Missing-trim representation is "decision 8" in the JSON and folded into decision 6 in the readable list | — | Stable ids D1 to D8 in section 2; missing-trim representation is **D8** |
| R3 | Charter standing rule 5 states CCF-14 as settled; packet section 4 asks it as a decision | The charter was approved "as a plan … not an approved semantic diff" | CCF-14 was **unruled** until D5 below. No approval was manufactured from the charter text |
| R4 | "Under either option CAP-008 is resolved" | — | CAP-008 is resolved when the D1 operation lands **and** Tier 1/2 pass, with the section 3 losses disclosed. Approving a mechanism is not a completed disposition |
| R5 | Three conceptual classes; "44 active trim work items … 27 standalone, 17 absorbed" | — | Accounting categories, not 44 human billability judgments. The human-adjudicated trim cards are exactly six: `rc_ceeb17f8e242`, `rc_57c6faf79c7e`, `rc_9e889631af67`, `rc_d0ffde500a92`, `rc_b642fe69b86a`, and `rc_7057c3c171e5` (adjudicated today) |
| R6 | "The kitchen's missing trim is cabinet casework trim … represented by the cabinet-replacement driver" | Stated as a hypothesis | Corroborated three independent ways and **accepted by Steven under D8**: the card's own observation is "Cabinet trim is mismatched and dated." (photo_024); the stored Pass 2f rationale names "counters with a visibly damaged or unfinished front edge"; the photographs show raw unfinished wood along the underside of the upper cabinets, an exposed raw countertop front edge on the peninsula, and a chipped base-cabinet end, while the door casing and baseboard are present and plain (the pinned photo review agrees on the casing) |
| R7 | "25 pinned runs" in places | The inventory has 26 entries: 17 replayable, 8 without a v5 envelope, 1 unpinned | Use 26/17/8/1. Every package measurement in this checkpoint is canary-weighted |
| R8 | `rc_b642fe69b86a` (heavy dated trim) listed among the costs of `no_action` | Its stored run records Terra **unsupported**, disposition `excluded`, $0 | The loss is of a future recovery path, not of a current billing |
| R9 | Packet `pass_2d_worklist` row 7 still names the removed parent `dated_window_treatment_valance` | Register section 1 re-pointed it on 2026-09-08 (S4-3) | The final worklist names `window_blinds_basic_or_plain`, making it a naming question |

## 2. Decisions

### D1 — Trim mechanism: `route_override: no_action` on `dated_interior_trim`, without CAP-022

Steven's answer: *"route_override no_action, no CAP-022."*

Reasoning of record. Ruling 3 (plain, basic or builder-grade presence does not bill by itself) is satisfied
completely: nothing on this item bills, and the two treatment bullets that migrated onto it under CAP-007
(S6-1) display at $0, which closes the fix-first condition. The item stays in the retrieval pool, so no
observation migrates to a billable neighbour — the measured failure of every removal-shaped lever. The
presence-stem `deny_any` lever moved 20 of the 54 rows it removed onto billable wood paneling and interior
doors in rooms that have neither; retirement is worse, at 55 of 108.

CAP-022's subject gate was **declined as unnecessary**: it would push four subject-mismatched rows out of the
item, one of which ("The built-ins are dated.") lands first on billable `dated_wood_paneling`, trading a
zero-dollar display imprecision for a new false billing, against the standing priority. CAP-022 is therefore
**subsumed**, recorded as an optional display-precision follow-on with that measured cost.

The preservation split (option a2) was declined because its only benefit is standalone billing for dated
trim, which the standing intent says trim is not for, and because Terra cannot separate plain from dated
(kill card `rc_73d15ca14405`: heavy period trim, refuted as plain; three of four sampled "dated"-worded
photographs show plain trim), so roughly half the dated-worded rows would still bill plain trim as dated.

Mechanically the claim text is unchanged, so `atomic_claim` does not move and the item id survives: no new
stale ids and no trim cutover. The projection fingerprint moves regardless, as it already does under CAP-007.

### D2 — Generator support scope (consequence of D1 and D4)

Admit exactly `route_override` and `scene_groups` to the carryover override set. `require_any` is **not**
admitted: no approved operation needs it once CAP-022 is declined, so S6-12 remains a recorded gap rather
than being closed speculatively. Economic and unknown keys continue to fail closed.

### D3 — Blinds visibility (S6-2): accept the omissions as presentation-only

Steven's answer: *"Accept as presentation-only; re-home to the Pass 2d worklist."*

Ten candidate rows resolved to no item. Five were already `excluded` at baseline; five were
`accepted_for_work` at baseline ($79 to $92) and now display nothing. The dollars are identical to
`no_action` and only the frontend envelope differs. All ten took the LLM path with a full candidate list and
the model chose none, so this is Pass 2d selection behaviour rather than catalog reachability: the ten rows
are recorded as accepted limitations **and** added to the Pass 2d worklist as a distinct category. The only
catalog lever, adding the bare token "treatment" to the blinds successor's `support_any`, stays rejected for
opening shortcut surface. Reopening trigger: the frontend requires the v5 blinds `no_action` envelope
specifically. A-4's visibility intent is satisfied by acceptance rather than left open.

### D4 — Wallpaper double charge (CCF-13): approve the bathroom scene exclusion

Steven's answer: *"Approve the scene exclusion."*

`dated_wallpaper_present` carries `bathroom` in `scene_groups` but has no bathroom `package_affinity`, so in
a bathroom it bills `WALLPAPER_REMOVE_PAINT` standalone while `dated_bathroom_wallpaper` bills the same code
absorbed into the bathroom package: one wall, two charges, reproduced byte-identically under 3.2. Because
`scene_groups` is a retrieval pre-filter, removing `bathroom` makes the generic item unreachable there, and
nothing is stranded because it has no bathroom affinity route to lose.

Disclosed and accepted effect. Across the frozen corpus the generic item sits in the post-guardrail top-8 of
**33 of 267** bathroom-scene rows and is first candidate on **7** of them (wall-tile and wall-colour bullets
such as "Small white wall tile is heavily dated." and "The pastel green walls date the space."), where it is
a wrong-subject distractor the model did not select. Removing it changes **no lexical shortcut**, changes
seven first candidates onto tile, paneling and paint items, and lets an unseen ninth candidate into 33
lists. The LLM-path consequence on those 33 rows is **not measurable offline** and is accepted unmeasured; a
local Pass 2d replay would settle it and was not authorized. Residual risk recorded: a future bathroom bullet
phrased "wallcovering" without "wallpaper" has no owner, because it fails the bathroom item's `require_any`.

### D5 — CCF-14 evidence precedence: matched human gold counts as evidence of the observation

Steven's answer: *"Yes, rule it that way."*

An explicit matched **human** gold annotation is evidence of the observation it states, so the unit counts as
human on the observation axis. Billability remains a separate axis. Automated `gold_incomplete` matches are
not evidence. Consequence: CAP-014 gains one human unit (g3, redfin_10806500 photo_003) and still fails
two-property independence, so it stays deferred. This ruling now stands on its own; the charter's standing
rule 5 is confirmed rather than assumed.

### D6 — Tranche 2: no change, recorded with triggers

Steven's answer: *"Confirm no change for all four."*

| Family | Disposition | Reopening trigger |
|---|---|---|
| `cabinets_dated_style` | no change | a human-adjudicated basic-grade cabinet billing, or a product ruling extending the trim treatment |
| `appliances_dated_or_basic` | no change | same; the damage sibling stays separate |
| `dated_interior_doors` | no change; the strongest remaining case (7 plain-presence rows, 6 "basic and dated", and a claim that equates hollow-core with warranted replacement) | a human-adjudicated builder-grade door billing, or a product ruling that dated doors are context like trim |
| `paint_refresh_recommended` | no change | Q-8 re-adjudication showing plain neutrals billing |
| `dated_bathroom_vanity_light`, `older_ceiling_fan_style` | no change; already non-billing through the electrical product quarantine (`excluded_quarantine`, 0 conditions from 69 corpus rows) | the quarantine is lifted |
| `landscaping_enhancement_opportunity` | no change; already `route_override no_action` | none |
| `dated_window_valance_or_curtains` | no change; S5-1 closed, A-5 preserved | a Terra-supported basic-curtain billing in the cohort |
| Railing coverage (CCF-5) | deferred; the 14 unresolved bullets on 8 properties are dated railing *style*, not finish wear and not a safety defect | a human-adjudicated railing finish-wear or unsafe-railing observation with no home, or a product decision on a presence-only railing concept |

### D7 — Supporting-context mechanism: no new code now, gated design recorded

Steven's answer: *"No new code now; record a gated design."*

v5 has no context lane: a `no_action` condition forms no work item, so it cannot be a package support, and
Sol sees only the Terra rationales of accepted children. The measured cost of accepting this is a strength
label on 7 bedroom packages at unchanged tier and price, roughly $920 of high end across 2 living packages,
and one support-only bedroom package that the standing intent says should not exist. Every affected package
needs a fresh Sol decision at cutover in any case.

Gated design recorded for later and **not authorized**: allow a `no_action` condition that carries a
`package_affinity` block to count as a zero-dollar support for strength purposes only, never for dollars and
never for package formation. `dated_interior_trim` keeps its `package_affinity` block as the landing pad;
under D1 that block becomes dead configuration, which no validator forbids and which is documented here
rather than removed. Trigger: whole-property acceptance shows package-strength regressions that change a
decision.

### D8 — Kitchen missing trim (`rc_ceeb17f8e242`): accept cabinet and counter-scope representation

Steven's answer: *"Accept cabinet/counter-scope representation."*

The physical subject is cabinet casework and countertop edge (R6), which the kitchen package's
`CABINET_REPLACE` driver and counter scope already cover; the deterministic replay shows the package rebuilds
identical in driver, strength, tier and price with three supports instead of four. What `no_action` removes
is the condition's support membership and its `accepted_for_work` label, not the work. Disclosed residual,
accepted: if the kitchen package were ever rejected, the missing-trim work would have no owner, unchanged
from today. Steven's earlier position that this loss was not accepted is resolved by this decision, taken on
the evidence above.

### CAP-013 — second property adjudicated: `rc_7057c3c171e5`

Steven's judgment on the two evidence photographs of redfin_10952874 (photo_018 and photo_027, canary run
`20260818_231243_cb71fce9`): **"Missing base trim — real work on a second property."**

Consequence, stated precisely. CAP-013 now has two human-adjudicated units on two different properties,
redfin_11185681 (kitchen cabinet casework and counter edge) and redfin_10952874 (stairwell wall base trim),
so the program's two-property evidence bar is **met** and the concept is no longer evidence-blocked. It
remains **schema-blocked and unauthorized**: Q-6 ("document and wait") stands, no authoring or schema work is
approved here, and nothing is implemented in this checkpoint. What changes is eligibility — an authoring
extension may now be *proposed* at a later gate. Such a proposal must decide the subject breadth the two
units span (wall base trim only, versus interior trim including cabinet casework), because they are
different physical subjects, and must account for the overlap with `baseboard_wear_scuffs`, whose
`embed_text` already advertises "Missing baseboards, detached trim … incomplete base trim installation"
while its `atomic_claim` states wear only.

This does not supersede the stored records: the condition is recorded under `baseboard_wear_scuffs` with a
Terra `supported` verdict, and the pinned 2026-09-02 photo review called it "unclear". Steven's adjudication
is a new human judgment on the same photographs and is recorded alongside them.

## 3. Losses and limitations accepted by these decisions

| Loss | Evidence | Size |
|---|---|---|
| 27 standalone trim billings stop | `trim_corpus_replay.json` | $2,242 to $11,212 across the 17 replayable artifacts; the ruling-3 exposure, and the intended effect |
| The dated and heavy-trim recovery path closes | `rc_b642fe69b86a` (currently unsupported, $0) and roughly 10 to 15 of the 46 dated-worded rows | future recoveries only; no current billing is lost (R8) |
| Plain-trim package context leaves support lists | corpus replay: 77 candidates identical, 15 support-list-only, 9 shape changes, 1 disappears | 7 bedroom packages strong to moderate at unchanged tier and price; 2 living highs down $394 and $526; 1 driverless bedroom package (redfin_166147710) no longer forms, its 3 repair children staying standalone, which is the pattern the standing intent forbids |
| Kitchen missing trim loses its `accepted_for_work` label and support membership | `rc_ceeb17f8e242`; two-unit replay | represented by the cabinet driver at $0 standalone in both variants (D8) |
| Every stored Terra and Sol checkpoint invalidates | the projection fingerprint is listing-global | already true under CAP-007; 97 stored artifacts across six roots name the trim item |
| `package_affinity` on a `no_action` item becomes dead configuration | no validator forbids it; the five existing overrides carry no affinity block | retained deliberately under D7 |
| Wallpaper: 33 bathroom rows lose a distractor, 7 change first candidate; the LLM-path effect is unmeasured | Session 5 candidate snapshot | accepted unmeasured under D4 |
| Ten blinds rows display nothing rather than a zero-dollar item | Session 6 bundle | accepted as presentation-only under D3 |

## 4. Deferred, with triggers

CAP-008 resolves with D1 once landed and validated. CAP-010 deferred: the lean was captured and billed on
the sibling successor, so the exposure is semantic. CAP-013 deferred but no longer evidence-blocked, as
above. CAP-014 deferred: one human unit after D5, needing a second property. CAP-016 deferred, with the bare
`ceiling texture` shortcut token going to the Pass 2d worklist. CAP-018 deferred: no second property.
CAP-022 subsumed by D1, with its measured cost recorded.

## 5. What these decisions do not authorize

Publication or production mutation; any provider or local-model spend, including the Pass 2d replay that
would measure D4's residual; Stage B; frontend or pricing work; a dedup-key change; CAP-013 schema or
authoring work; admitting `require_any` to carryover overrides; claim-text edits (A-7); removal of inherited
economics from any `no_action` item; and any edit to a frozen artifact, to
`reports/catalog_audit_approvals.json`, or to the evidence bundle.
