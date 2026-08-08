# HANDOFF — Pass 2a prompt ablation (variance reduction)

2026-08-08. Follow-up to `docs/HANDOFF_estimate_variance_gpt56.md` (read it
first — it has the measured evidence and the amplification mechanism).
Steven's decision: 2c work is frozen; attention moves to Pass 2a. This is a
small, tightly-scoped measurement task — resist expanding it.

## Hypothesis

`PASS_2A_USER_PROMPT = "What stands out here to a renovator"`
(`tools/scene_classifier_passes.py:608`) is salience-framed: it asks the
model to *sample* what's notable, which is inherently unstable run-to-run
under gpt-5.6-terra (no temperature control). Inventory framing should force
enumeration toward a repeatable observation set.

## Variants (staged — do not run 3 unless 2 forces it)

1. **Current prompt** (baseline).
2. **Whole-image inventory wording** — replace only the "stands out" framing
   with an enumerate-the-visible-condition instruction. One wording change,
   nothing else.
3. **Variant 2 + visible-evidence boundary** — ONLY if variant 2 produces
   unsupported claims (claims with no visible basis in the photo). Otherwise
   skip.

## Design

- **Branch**: `pass_2c_redesign` @ cdefc2b (cutover build). Not
  `pass_2c_kind_v3`, not origin's 2e1b9ed — both alter 2c/photo handling and
  confound the measurement. 2c prompts stay frozen exactly as they are here.
- **Fixed inputs**: redfin_11000447 and redfin_10806500 (7 photos each; both
  already have archived run pairs in
  `artifacts_canary/candidate_archive_pre_v31/` + `artifacts_canary/candidate/`
  demonstrating the problem, giving free historical comparison).
- **Repeats**: k=3 per variant per property.
- **Attribution leg (variant 1 only, do this first)**: also replay the
  downstream pipeline (2b→2c→2d→costing) k=3 times from ONE frozen 2a
  capture. This splits variance into "2a's own instability" vs
  "everything-after-2a instability". **Decision gate: if frozen-2a replays
  still flip cost-bearing issues heavily, the variance lives in 2c extraction
  and prompt-tuning 2a won't fix it — stop and report rather than running
  variants 2/3.**
- **Gold reference**: hand-gold the ~14 photos once (list of visibly
  supported condition observations per photo); reuse across all variants.

## Metrics (per variant, per property)

1. **Supported recall** vs the hand-gold set.
2. **Unsupported observations** — claims with no visible basis (drives the
   variant-3 decision).
3. **Run-to-run agreement** — Jaccard overlap of resolved catalog-id sets
   across the k repeats (reuses the existing 2d machinery; canonicalizes
   free-text drift).
4. **Cost-bearing issue flips** — line items present in some repeats but not
   others, and the headline low/high spread across repeats.
5. **Bias, not just stability** — headline center per variant vs variant 1's
   center. Inventory wording will likely emit MORE observations, and flat
   allowances mean each marginal observation can flip a $10–20k line ON. A
   variant that wins on agreement but shifts headlines systematically is not
   a free win; report the shift, don't judge it here.

## Scope guards

- Measurement only. No shipping prompt change comes out of this handoff; the
  winning wording goes to Steven with the numbers.
- Do not touch 2c prompts, the catalog, costing, or the canary artifacts of
  the other 16 manifest properties.
- A later shipped 2a change invalidates the current 18-property canary
  deltas as-measured — the 4A sign-off record must state which 2a prompt its
  approved numbers were measured under. Note it; don't act on it.
- Budget: ~2 properties × 7 photos × 3 repeats × (≤3 variants) 2a calls plus
  k downstream replays — Terra reasoning `low`; keep it to this set.
- Embeddings sidecar gotcha: :8081 `/health` lies; probe with a real POST
  (`scripts/run_kind_canary.py` `_preflight_embeddings` shows how).
- Rerun mechanics if driving via the canary script: it skips properties whose
  candidate dir has an artifact — archive, don't delete.
