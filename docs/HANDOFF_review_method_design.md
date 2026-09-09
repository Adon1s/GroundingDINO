# Handoff — design the manual review method and its priorities

Written 2026-08-22 · backend `renovation_architecture_rework` @ `df35f23` · frontend `renointel-prod` @ `kind_ontology_v2_compat` (`411312e6`)

**Purpose.** A new session is to design *how* Steven reviews the output of the
new renovation architecture (v5) and *what he should review first*, so that
the open quality questions get answered with the least of his time. This
document is context only: the situation, the data that exists, the measured
facts, the constraints, and the questions. Where a view was expressed in
conversation it is labelled as such. **Method, tooling, sampling rules, and
order of work are for the new session to decide.**

> **Designed 2026-08-22 → `docs/DESIGN_review_method.md`** (neutral verdict
> labels, strata not contest, dirA at 100 % + uniform sample of accepted
> conditions, blind-then-reveal queue served by `scripts/review_server.py`).
> **Correction to §5 below:** in the artifact, the photo_023 flooring
> condition (`oc1_38b0a263fd54d0f4`, bedroom_3) has Terra `unsupported`
> ("…intact without clearly visible scratches, scuffs, or worn finish") and v5
> disposition `excluded`; the "visible wear, uneven sheen" text is the upstream
> 2a/2c observation, and it was **Pass 2f (v4)** that confirmed it and v4 that
> billed $242/$4,041 — which is what the frontend (still rendering v4) showed.
> It *is* in the disagreement set (direction B); v5 got it right.

---

## 1. Situation

- Production flipped to the new architecture on 2026-08-21 (~23:20 CT):
  `RENOVATION_ARCHITECTURE_MODE=new` + `KIND_ONTOLOGY_VERSION=observation_kind_v2`
  (catalog 3.1). The estimator of record written to the artifact root is
  `renovation_estimate_v5`; `renovation_estimate_v4` is still computed and is
  what the frontend renders. Frontend adoption of v5 is a separate later task
  of Steven's.
- A **7-day operational observation window is running** (day 1 = 2026-08-22).
  Its rule is operational invariants only; it does not judge estimate quality.
  Estimate behaviour (prompts, catalog, package policy) is not to change during
  the window. Reading and adjudicating is unconstrained.
- Throughput is bounded by the OpenAI free quota with the budget guard on:
  roughly 5–6 listings/day while Pass 2f rides the Sol tier (the 250k Sol
  ledger binds first), ~11/day Terra-wise. Production artifacts accrue at that
  rate through the week.
- Record of the cutover, the verified gates and the execution log:
  `docs/DECISION_renovation_architecture_session9_cutover_20260821.md`.
  (Its §9 final paragraph "staged, not flipped" is stale.)

## 2. Steven's stated priorities and decisions (verbatim intent)

- Evaluation frame (2026-08-21): he is not a renovator; prices are provisional.
  The rework is judged on **(1) fewer hallucinations** and **(2) more useful
  observations that build packages**. Dollar deflation from dedup is accepted.
- 2026-08-22, on the first live listing: the flooring claim "visible wear,
  uneven sheen, possible staining or scratching" on photo_023 of
  `redfin_10965375` is **a hallucination — there is no scratched hardwood**;
  his read is that the model confuses stain/colour with wear, and that it
  "often confuses staining with worn."
- His explanation of why v4 required packages to build from issues: it
  **silently removed singular hallucinations** like that one. (v5 by design
  has no package-level photo check; a lone Terra-confirmed condition prices
  standalone.) He called this "an issue for another time" and agreed the
  7-day window comes first.
- **Decision: review where the two models disagree first.** He agreed this is
  the first priority for any review method.
- He wants the new session to design the method and the priorities itself; he
  asked that this handoff carry findings, not recommendations.

## 3. What "the two models" means, and where their verdicts live

Every artifact (canary and production) carries two independent photo
judgments on the same listing:

- **Terra (v5)** — one verdict per *condition* (`supported` / `unsupported` /
  `cannot_assess`), against that condition's photos, batched per estimate
  unit. Path: `renovation_estimate_v5.result.condition_reviews[*]` with
  `condition_id`, `verdict`, `rationale`, `terra_call_id`; join to
  `renovation_estimate_v5.result.observed_conditions[*]` on `condition_id` for
  `catalog_item_id`, `catalog_kind`, `scene_group`, `estimate_unit_id`,
  `issue_ids`; join `issue_ids` to `issues_flat[*].issue_id` /
  `photos[<key>].issues.final[*].issue_id` for `photo_key`, `description`.
- **Pass 2f (v4)** — a *package-level* VLM check that confirms/rejects the
  individual issues inside a package candidate and the package as a whole.
  Path: `renovation_estimate_v4.packages[*]` (and candidates) with
  `verification_status`, `confirmed_issue_ids`, `rejected_issue_ids`,
  `driver_issue_ids`, `evidence_summary`; run stats in
  `renovation_estimate_v4.pass_2f_trace` (`confirmed_count`,
  `rejected_count`, `uncertain_count`, `no_image_count`). Note 2f only sees
  issues that reached a package candidate — standalone conditions have **no**
  second opinion.
- Both live in `photo_intel_debug.json` (full) and `photo_intel.json` (slim;
  `pass_2f_trace` survives slimming, `analysis_debug` does not).
- The canary mapping of 2f issue-level verdicts onto v5 conditions is already
  implemented in `scripts/build_session9_decision_packets.py` (it produced the
  P2 cards) — the reference for how the join is done.
- Images: `renointel-prod/public/images/properties/<property_key>/photo_NNN.jpg`;
  artifact photo keys are `photo_NNN.jpg` in order (`photos[key].photo.index`).

## 4. Data that exists today

| Set | Size | What it offers | Where |
|---|---|---|---|
| Session 9 canary | 18 properties × 2 replicas = 36 artifacts, ~2,100 Terra verdicts (supported 1746 / unsupported 282 / cannot_assess 65), Sol approve 226 / reject 23 | the only set with **two replicas of the same inputs** (stability) and with the 2f-vs-Terra join already built | `artifacts_canary/renovation_session9_20260818/run_{1,2}/candidate/` |
| Canary disagreement cards (P2) | 85 v4 issue-level verdicts on **68 distinct v5 conditions** where 2f and Terra disagree on the same photo: 32 Terra-supported/2f-rejected (direction A — what v5 bills), 36 2f-confirmed/Terra-not (direction B); agreement on 766 + 38 + 4 + 38 of ~891 shared | photos inline, both models' text, v5 outcome | sheet `reports/session9_photo_review_20260821.html` (images are `file:///` links — works from disk, not over http); list at `docs/DESIGN_renovation_architecture_decision_packets.md` §P2; blank verdict slots `reports/session9_decision_worksheet.json` (`p2_conditions` C001–C068, `p1_packages` P01–P13, `p3_bathrooms` B01–B10; **0 of 81 filled**) |
| P1 packages | 13 v4 package candidates that 2f **rejected** at package level and v5/Sol **approved** (11 of 13); net lift $94,930 / $141,253 over their children — larger than the corpus low-side delta | the package-level version of the hallucination question | design doc §P1, sheet §P1 |
| P3 bathrooms | 10 properties, per-surrogate photo strips; v4 expanded bathroom packages per surrogate, v5 bills once (−$58,438 / −$146,095 over six) | room-identity judgment | design doc §P3, sheet §P3 |
| Production, new mode | **5 complete artifacts** as of 2026-08-22 morning: 229 Terra verdicts (162 / 55 / 12), 138 work items, 24 packages applied; growing ~5–6/day | production routing (1a/2a/2b/2c/2d on Terra at effort none, 2f on Sol), real queue distribution, no replica | `renointel-prod/artifacts/<key>/<run>/` (newest runs, `catalog_version 3.1`) |
| Pre-cutover corpus | ~1,264 artifacts, catalog 2.1, v4 only (2f verdicts present, no Terra) | historical baseline for display/severity behaviour | same tree, runs dated ≤ 2026-08-03 |
| Single-photo share | in canary run_1, 357 of 657 active work items rest on one photo, carrying $82,637 / $713,757 of $171,031 / $1,594,286 | relevant to "one photo is enough?" | design doc §P2 summary |
| Per-item disagreement table | direction A/B counts by catalog item (e.g. `paint_refresh_recommended` 1/6, `dated_interior_trim` 4/3, `peeling_or_discolored_paint` 4/1, `hard_flooring_scratched_or_worn` 1/2, `worn_or_stained_flooring` 2/0, `outdated_kitchen_finishes` 2/0, `cabinets_dated_style` 2/0 …) | where disagreement concentrates | design doc §P2 table |

Note on the P2 worksheet: the design doc estimated ~45 minutes for all 68
cards and suggested, if sampling, all of direction A and half of direction B.

## 5. Observed facts from the first live listing (redfin_10965375, run 20260821_233007_c5b989ee)

- Verifier PASS 16/16 (mechanically clean). 34 photos; 35 conditions reviewed
  (24 supported / 9 unsupported / 2 cannot_assess); 19 work items; 4 packages,
  Sol approved 4/4; v5 headline $24,645 / $69,481 vs v4 $20,611 / $50,922.
- The photo_023 claim: `hard_flooring_scratched_or_worn` @ bedroom, Terra
  `supported` — "The wood floors show visible wear, uneven sheen, and possible
  staining or scratching." Steven: not true of the photo (renovated room,
  clean hardwood). It priced standalone ($242 / $4,041, `verified_estimate_drivers`,
  `review_source pass_2f:premium`, `visual_verification_status confirmed`) —
  i.e. in this instance **2f also confirmed it**, so it is *not* in the
  disagreement set. This is a data point the new session should weigh: the
  disagreement filter does not catch a hallucination both models share.
- Separately, the *What we noticed* "Flooring · Severe" card over a "Low" photo
  row was traced to two different quantities on one word-scale (bucket max of
  boosted `display_severity` vs raw catalog severity); pre-existing (91% of
  pre-cutover buckets show the pattern), one notch moved by the
  `upgrade`→`degradation` reclassification. Documented, with the mechanism and
  numbers, in `docs/HANDOFF_output_quality_manual_review.md` §2. It is a
  display matter, not a review-method matter, but the same listing is the
  worked example.

## 6. Tooling that exists (and its friction)

- `scripts/build_session9_decision_packets.py` — regenerates the photo sheet,
  the packets JSON (`reports/session9_decision_packets.json`) and the worksheet
  (worksheet is not overwritten once present). Zero provider calls.
- `reports/session9_photo_review_20260821.html` — static page, `file:///`
  images (no http serving; not phone-viewable).
- `reports/session9_decision_worksheet.json` — verdicts are recorded by
  **hand-editing JSON**; instructions in its `instructions` block.
- `scripts/verify_renovation_artifact.py` — mechanical artifact check (new).
- `scripts/replay_renovation_architecture.py` — offline replay of stored
  verdicts/decisions (no provider calls); a "re-decide" mode (re-ask Terra/Sol)
  does **not** exist yet — design doc §2 notes it would be needed for a
  consistency probe.
- `scripts/show_daily_token_spend.py --root <FE artifacts>` — ledger.
- Frontend DB: SQLite `renointel-prod/prisma/dev.db`, `Property` table maps
  `propertyKey` → street address; property page `http://localhost:3000/property/<propertyKey>`.
- Observed friction (conversation): two files to juggle, JSON by hand, prices
  on the cards, no progress/resume, no way to review away from the desk.

## 7. Constraints the design must respect

- No change to estimate behaviour, prompts, or catalog before the observation
  window closes; any Terra prompt/rubric change also invalidates the canary
  deltas and requires a fresh freeze + new two-day canary.
- Adding another model's opinion as a pre-screen was raised; Steven's core
  concern is model hallucination, so the trade-off is his to weigh.
- Prices are provisional — dollar-shaped findings are "needs a renovator."
- Do not re-open the 967 Session 9 review items (audited, reproducible).
- `RV_ROOT` is the live working tree: don't `git checkout` other branches in
  `realtorvision-backend` during the window; use a worktree.
- Terra/Sol run at `reasoning_effort=medium`; temperature is not a lever on
  the Responses API (design doc §2 records the available levers).

## 8. Open quality questions this review program is meant to answer

1. How often does Terra confirm a condition that is not true of the photo
   (direction A), and does it concentrate on particular catalog items (style/
   "dated" items; wear/stain items on flooring) or on single-photo conditions?
2. How often does Terra reject a condition that is real (direction B)?
3. When Terra confirms the children but 2f rejected the package, is the
   package warranted? (P1: 11 packages, $94.9k / $141.3k lift.)
4. Are multi-bathroom listings under-billed by v5's single bathroom unit? (P3)
5. What does the loss of the package-level photo gate cost in single
   hallucinations that now price standalone — and is the case where both models
   agree on a false claim (the photo_023 instance) common?
6. Is one photo sufficient evidence for opportunity-driver items? (P6 item 3,
   five items, accepted provisionally.)
7. Does the 20-condition / 12-image Terra batching dilute verdict quality?
   (no gate can see this; measured mean 7.0 conditions / 2.9 images per call)
8. Acceptance rates overall — 71.4% of conditions accepted for work, 93.7% of
   package candidates applied — correct judgment or over-acceptance?

## 9. Things raised in conversation, undecided (listed for completeness, not as recommendations)

- Extract disagreement cards from each day's production artifacts
  automatically (the canary join already exists in the packets builder).
- A one-card-at-a-time local review page served over http (keyboard verdicts,
  writes the worksheet, resumable, phone-viewable); order cards direction A
  first and grouped by catalog item; show running tallies; hide prices.
- Sampling rules (e.g. all of direction A, half of B; disagreements plus a
  small random control per production listing).
- A model pre-screen to sort obvious vs contested (with the caveat above).
- Steven's own framing: "do I have enough data?" — the canary is the
  disagreement dataset; production adds routing validity and volume at
  ~5–6/day; the binding resource is adjudication time, not listings.

## 10. Pointers

`docs/DESIGN_renovation_architecture_decision_packets.md` (P1–P6, §2 variance
levers, §9 decision log — P1–P5 rows blank) ·
`docs/DECISION_renovation_architecture_session9_cutover_20260821.md` ·
`docs/HANDOFF_output_quality_manual_review.md` (display finding + index of
the other open review tasks) · `reports/session9_review_digest_20260821.md` ·
memory notes `steven-priorities-hallucinations-over-pricing`,
`ui-severity-display-defect`, `renovation-architecture-session9-cutover-decision`.
