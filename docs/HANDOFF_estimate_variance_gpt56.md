# HANDOFF — Run-to-run estimate variance under gpt-5.6-terra (no temperature control)

2026-08-08. Discovered during the catalog 3.1 canary re-measure (layout-item
retirement, see `docs/HANDOFF_layout_item_audit.md` and commits 58c073e /
a8fcac7 on `pass_2c_redesign`). This handoff is investigation-scoping only —
no fix has been designed or attempted.

## The problem

Re-running the identical pipeline on identical inputs produces materially
different renovation estimates. The only intended change between the run pairs
below was catalog 3.0 → 3.1 (removal of two layout items worth a known
−$5,000–$25,000 per property); everything else — photos, code, prompts, model
map, metadata — was byte-identical.

Steven's context: this was never observed with the local Qwen models, which
ran with temperature turned well down. The canary model map routes Pass
2a/2b/2c (reasoning `low`) and 2f (`medium`) to `gpt-5.6-terra`
(`benchmarks/configs/kind_canary_model_map.json`), and the gpt-5.6 API
exposes **no temperature parameter** — there is no dial to turn down.

## Measured evidence (all artifacts preserved)

Candidate-vs-candidate pairs, old run in
`artifacts_canary/candidate_archive_pre_v31/<key>/`, new run in
`artifacts_canary/candidate/<key>/` (both are `photo_intel.json`; the
archived runs used catalog 3.0, the new ones 3.1):

| Property | Old candidate headline | New candidate headline | Gross swing after backing out the −$25k retired item |
|---|---|---|---|
| redfin_11000447 | $30,100–$120,675 | $49,500–$140,500 | ≈ +$45k of high-end line-item churn |
| redfin_10806500 | $38,500–$134,000 | $19,650–$69,200 | ≈ −$40k beyond the retirement |
| redfin_25809814 | $25,752–$98,500 | $13,252–$67,500 | low bound swung +8% → −44% vs baseline |

Worked attribution for redfin_11000447 (7 photos): the new run added
`outdated_kitchen_finishes` (+$20k high), `cabinets_worn_finish` (+$10k),
`cabinets_damaged_or_water_stained` (+$10k), `older_flooring_style` (+$8k)
and dropped `appliances_dated_or_basic` (−$6k), `dated_bathroom_flooring_style`
(−$3.5k). Every one of the appearing items traces to observation text that
did not exist in the old run ("Countertops are laminate", "Lower cabinetry
shows staining and discoloration", "Cabinet doors and drawers appear
damaged", "The floor appears to be old wood or painted plank flooring") —
the old run's kitchen observations were sparser ("Cabinets are old and
mismatched"). This is **generation variance in the Terra passes**, not
resolution or costing drift.

## Why the dollar impact is so large per flipped observation

Costing amplifies observation-set diffs into step functions: most of these
items are flat room allowances ($500–$10,000, $2,000–$20,000, …), so a single
observation appearing or vanishing flips an entire allowance on or off.
Occurrence counts do not scale cost (verified: 1 vs 19 observations price
identically). The estimate is therefore highly sensitive to *which items
clear the one-observation threshold*, and insensitive to everything after
that.

Deterministic-by-construction stages (not suspects): 2d resolution is
embedding retrieval (jina, fixed vectors), 2e is rule-based, costing v4 is
pure arithmetic. Pass 1a runs on local Qwen with controlled temperature.
The variance enters at 2a/2b/2c (and possibly 2f verification).

## Consequences

- Canary sign-off deltas in the ±20–40% band carry substantial noise;
  per-property BREACH labels from single runs partly measure the dice roll,
  not v1-vs-v2. redfin_11000447's +93%/+79% breach in
  `reports/kind_cutover_delta_review_20260808.md` should be read this way.
- Production concern once cutover lands: the same listing re-analyzed twice
  can hand a user a rehab range differing by tens of thousands of dollars.

## What the investigating session should establish

1. **Quantify it.** N≥3 repeat candidate runs on 2–3 manifest properties
   (`scripts/run_kind_canary.py --side candidate --build-root . --only <key>`;
   archive each run dir before re-running — the script skips properties that
   already have an artifact). Report headline spread and per-item flip rates.
   Terra API cost is the constraint; pick small-photo properties
   (redfin_11000447 has 7 photos, redfin_10806500 has 7).
2. **Localize the pass.** Hold upstream outputs fixed and replay downstream
   (the checkpoint files the analyzer writes per run should allow replaying
   2c on frozen 2a/2b output, and costing on frozen 2c output) to split
   variance between 2a/2b scene+condition description, 2c issue extraction,
   and 2f verification.
3. **API-side determinism options for gpt-5.6:** seed / deterministic
   sampling support (if any), whether reasoning-effort changes variance,
   structured-output modes, and whether the `original photo quality` setting
   (commit 2e1b9ed on origin/pass_2c_redesign — NOT in the cutover build)
   changes observation stability.
4. **Damping options if the model can't be pinned down**, in rough order of
   invasiveness: prompt changes that force per-surface enumeration (bounded
   checklists produce more stable sets than open-ended description);
   median/consensus over k runs for the passes that matter (cost ×k);
   costing-side smoothing so one marginal observation can't flip a full
   $20k allowance (e.g. confidence-weighted allowances) — this last one
   touches the estimate contract, treat as a separate project.
5. **Decide what this means for the 4A sign-off** currently in review: either
   accept the noise band explicitly in `approved_headline_deltas`, or
   stabilize the 5 re-measured properties with repeat runs before approval.

## Constraints

- Branch discipline: measure on `pass_2c_redesign` (cutover build). Do not
  mix in `pass_2c_kind_v3` prompt changes or origin's 2e1b9ed — both alter
  2c behavior and would confound the measurement.
- Do not modify prompts, catalog, or costing while quantifying (steps 1–2);
  those are candidate fixes, not measurement.
- Embeddings sidecar gotcha: :8081 `/health` lies; the canary script's real-
  POST preflight handles this, heed it if driving the analyzer directly.
