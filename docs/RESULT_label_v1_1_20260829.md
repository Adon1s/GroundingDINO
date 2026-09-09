# Result — v1.1 label repair

Written 2026-08-29 · 135/135 cards adjudicated · 125 condition labels emitted ·
zero provider calls · frozen v1 evidence preserved byte-for-byte

## Decision

**The repair succeeded as a labelling exercise and failed to produce a
publishable rate.** Adopt v1.1 as the development label source; do **not**
publish 2.66% as a corrected population rate; do **not** run Session F as
chartered.

- Canary hard-false falls **10.51% → 2.66%** and production **10.42% → 8.94%**,
  but every v1.1 interval overlaps its v1 interval. No v1↔v1.1 difference is
  separable, in either cohort.
- **Nothing about Terra changed.** The scored v5 outputs are byte-identical
  across v1 and v1.1. This is a re-measurement of *labels*, by the same
  reviewer, on the same frozen artifacts. "11 → 3" is a reclassification, not a
  safety improvement.
- The canary scoreable population no longer supports the planned wording arms:
  `hard_false_billed` = 3 (a paired test tops out at p = 0.125, unreachable at
  any outcome) and `dirB_wording_recovery` = 0.
- The structural finding *is* real and is the part worth keeping: v1's single
  `unsupported` label was collapsing **absent** / **named wrong** / **not worth
  billing** into one bucket. They split **24.1% / 49.4% / 26.5%** of canary
  billed-error mass. Roughly half the "errors" v1 counted are real conditions
  Terra found and described imperfectly.

## Integrity and reproducibility

| file | sha256 | state |
|---|---|---|
| `reports/review_queue.json` | `8512b87c2af18d1104069c15a5eda977d5fe79eb736f9aedd976d174f89d318a` | frozen, unchanged |
| `reports/review_verdicts.jsonl` | `0c7ca6a8b55d6dac69021a00e81315be1b15cef68ffd59fbe2973cb895a8d70f` | frozen, unchanged |
| `reports/retag_queue.json` | `aac934b72102fe91be1d89a249474ea05d1fba54994cba532bb748e230c4f78a` | frozen, unchanged |
| `reports/retag_verdicts.jsonl` | `a038029811c9fafe78f6e2cd3d2456d26ad6499018a217ff66ecb4c13901a4a7` | frozen, unchanged |
| `reports/adjudication_queue.json` | `42443aa2ea37cb6d150502f54b11917ff93181be336729eec451f24dc0efc078` | generated, deterministic on rebuild |
| `reports/adjudication_verdicts.jsonl` | `3de687790ea1faefbfffa33e6092a28cb3fdcf691d167b45dc87a0eec9461086` | append-only |
| `reports/labels_v1_1.json` | `7f9b03017195144195550074b49dbb0488ed3ad33fdde905ba4e5867dfdbe3bd` | the overlay |
| `reports/labels_v1_1.md` | `6ed28f6ecb6cbcdbfcc883d64428f2298288fb4fc845c1f5fc1adde95032c84d` | the report |

| integrity check | result |
|---|---:|
| queue cards / net verdicts | 135 / 135 |
| orphan verdicts / unanswered cards | 0 / 0 |
| answers outside the v1.1 vocabulary | 0 |
| peeked / explicit undos | 0 / 0 |
| blind cards / leaking any v1 outcome | 89 / **0** |
| repeat pairs byte-identical / min spacing | 10 / 15 |
| estimator-arm coverage, canary dirA + uniform | 32/32 + 40/40 |
| estimator-arm coverage, production dirA + uniform | 19/19 + 14/14 |
| min photo short side across served cards | 651 px (gate 500) |

**The 35 "overwrites" are not deliberation.** All 35 are duplicate POSTs from
the Enter handler, max gap 1.0 s, median 0.0 s, with **0 slug changes and 0 note
changes**. There are zero genuine revisions in the log.

Validity gate: an independent re-implementation of the weighting identity
reproduces the frozen v1 estimator exactly — canary hard-false 10.5114%
[4.73, 23.00] from dirA 7/32 + uniform 4/40, canary broad 13.5767%, production
10.4152% [5.09, 32.46] from dirA 8/19 + uniform 1/14 — matching
`reports/review_analysis.md` lines 165-169. A five-way independent rebuild of
the v1.1 axis labels from the raw verdict log agrees with `labels_v1_1.json` on
all 125 cards with zero disagreements.

## Quantitative result

Weighted; dirA weight 1 (census), uniform 17.775 canary / 13.143 production;
N_accepted = 743 / 203. Bands are 95% Wilson on the uniform arm propagated
through the weighting identity (`CI_METHOD`), and ignore listing clustering on
both sides.

| cohort | measure | v1 | v1.1 |
|---|---|---|---|
| canary | hard-false (`claim = absent`) | **10.51%** [4.73, 23.00] | **2.66%** [0.69, 12.60] |
| canary | like-for-like broad (`absent + misnamed`) | **13.58%** [6.84, 26.60] | **8.12%** [3.41, 19.95] |
| canary | wrongly billed (`absent + trivial`) | — | 5.59% [2.13, 16.60] |
| canary | broadest (`absent + misnamed + trivial`) | — | 11.05% [5.27, 23.54] |
| production | hard-false | **10.42%** [5.09, 32.46] | **8.94%** [3.62, 30.99] |
| production | like-for-like broad | **10.42%** | **9.43%** [4.11, 31.48] |
| production | broadest | — | 9.92% |

`docs/DESIGN_label_v1_1_adjudication.md` §3 maps `absent`→unsupported and
`misnamed`→overstated, so **absent + misnamed** is the only defensible analogue
of v1's "broad". Folding in `trivial` departs from §4 fn2, which designates
trivial neutral and bars it from arm selection; 11.05% is reported, not the
comparison figure. The two production `inconclusive` cards are held in the
denominator and out of the numerator, matching `review_analysis.py:379-381`;
four defensible exclusion policies span 8.94–9.92%. Canary is exactly
policy-invariant (zero inconclusive in any canary arm).

### Canary scoreable population

| class | v1 | v1.1 |
|---|---:|---:|
| `hard_false_billed` | 11 | **3** |
| `supported_billed` | 55 | 57 |
| `overstated_billed` | 6 | *(retired)* |
| `misnamed_billed` | — | 7 |
| `trivial_billed` | — | 5 |
| `dirB_recovery` | 16 | 16 |
| `dirB_wording_recovery` | — | **0** |

Production: hard-false 9 → 6, plus 2 `excluded`, 1 `trivial`, 1 `misnamed`,
1 `dirB_wording_recovery`. 23 of 125 cards changed class.

## Why the drop is not a measured safety improvement

Three structural facts, each independently reproduced.

**1. The hard-false class was never re-examined blind — by construction.** All
79 blind cards were v1 `supported_billed`; all 20 v1 hard-false cards sat in the
phase-2 re-ask arm, which prints the reviewer's own v1 verdict and re-tag note
above the photos. Movement was **18/46 (39.1%) non-blind vs 5/79 (6.3%)
blind**, and the non-blind arm moved **10 cards toward "the claim was exact"
against 1 away**. A hard-false card could not have been re-judged blind under
any selection, because the blind arms filter to `supported`.

**2. Phase 2 is nearly recoverable from the re-tag it was meant to
independently re-ask.** A rule reading only the re-tag answer predicts **44/46
(95.7%)** of the non-blind answers and predicts the canary 11→3 outcome **11 of
11, card for card**. The pre-committed §5 expectation table scores only 26/46 =
56.5%; 11 cards moved against it, all toward `exact`. §5 declared the re-tag
"validation, never a label source" — at the present/absent level phase 2
reproduced it almost exactly.

**3. Both hard-false rates are effectively single-observation estimates.** One
canary uniform card = **2.392 pp**; one production uniform card = **6.474 pp**.
89.9% of canary's 2.66% and 72% of production's 8.94% rest on one uniform card
each. Setting the canary uniform count to 0 gives 0.27%; to 2 gives 5.05%.

Further limits on the same numbers:

- **One listing carries the canary story.** `redfin_25809814` supplied 7 of the
  11 canary v1 hard-falses and 5 of the 8 departures. Excluding it, the move is
  7.72% → 2.52%, not 10.51% → 2.66%.
- **The canary/production divergence is not a finding.** The bootstrapped
  difference is 6.57 pp, 95% CI [−4.01, +21.91]; Fisher on the two uniform arms
  (1/40 vs 1/14) gives p = 0.455. The disagreement that made the re-tag
  inconclusive has changed form, not resolved.
- McNemar on the canary billed arms (11/72 → 3/72, 8 discordant out, 0 in) gives
  **p = 0.0078**: the revision is systematic and one-directional. That is
  equally the signature of a genuine correction and of motivated reasoning, and
  nothing in this exercise separates them.

**The one blind result robust to any noise floor:** **0 of 79** blind
`supported` cards were relabelled `absent` (95% Wilson upper bound 4.6% pooled;
6.5% canary, 13.8% production). That null supports the repair's direction.

## Session F — do not run as chartered

| leg | canary n | status |
|---|---:|---|
| `hard_false_billed` | **3** | paired exact McNemar best attainable **p = 0.125** — cannot reach α = 0.05 under *any* outcome, including a perfect 3/3. Unpaired vs the 6.4% floor needs ≥2/3 (p = 0.0118), power 0.50 at a true 50% flip rate. Effective independent n ≈ 2: two of the three are the same catalog item on the same property, decided in one batched call |
| `dirB_recovery` | 16 | the only win leg with a pulse — needs ≥6/16 recovered (p = 0.042) |
| `supported_billed` | 57 | the only well-powered column; detects loss inflation above ~12/57 (p = 0.028) |
| `misnamed_billed` | 7 | needs a ~37–42% true rate for 80% power |
| `dirB_wording_recovery` | **0** | the single instance is production, and `score_redecide_variants.py:118` filters to canary. **No test exists** |

**The 6.4% noise floor is a blend of two significantly different directional
rates.** Recomputed from the frozen run_1/run_2 pair (250 comparable conditions,
16 flips — reproduces exactly): supported→not-supported **9/222 = 4.05%**
[2.15, 7.52]; unsupported→supported **6/26 = 23.1%** [11.0, 42.1]; Fisher
p = 0.0019. 6.4% remains the floor of record and is conservative on the billed
side, but the dirB gate is genuinely miscalibrated against it.

Scorer defects to fix before any arm runs:

- There is no dirB **loss** population. Only cards v1 already called recoveries
  were re-asked, so `dirB_recovery` could fall but never rise — "16 → 16" means
  "could only fall and didn't".
- `load_population_v1_1` silently excludes production, which makes every
  production comparison in this program non-scoreable.
- The `SCORING` policy gives a coarsening arm no way to earn credit on
  `misnamed_billed` — it scores neutral whether Terra keeps or drops the card.

## Failure modes by measured mass

Canary billed-error mass = 82.100 population-conditions = 11.05% of accepted.

| rank | mode | canary mass | share | robustness | lever |
|---|---|---:|---:|---|---|
| 1 | **misnamed** — real condition, wrong name | 40.550 (5.46%) | **49.4%** | Weak. 87.7% of mass on 2 uniform cards; 43.8% on `boarded_up_entry_or_window`, which the reviewer answered **both ways** in the invisible repeat | catalog wording / coarsening — untested, not refuted |
| 2 | **trivial** — real, named right, not worth billing | 21.775 (2.93%) | 26.5% | Moderate. 6/6 `modernization`; 6/14 on `older_flooring_style`, `cabinets_dated_style`, `dated_interior_doors` vs 0/89 elsewhere. But the *kind* claim fails within-listing permutation (p = 0.29) and the *item* claim is p = 0.050 once post-hoc set selection is costed. 4 of 6 came from the non-blind arm | severity/threshold follow-up on those three items — an item-level gate removes 6 `supported_billed` alongside 6 `trivial` (1:1 collateral) |
| 3 | **absent** — hallucination | 19.775 (2.66%) | 24.1% | Very weak. 89.9% one uniform card, n = 3, effective n ≈ 2, zero blind and zero repeat coverage | nothing actionable at this n — the lever is more review, not more tuning |

Production is a different shape: hard-false is ~90% of its 9.92% error mass
(misnamed and trivial are one card each).

## Discard list

Claims that did not survive adversarial verification and must not be carried
forward:

- **"12 of 16 Terra rejections hedge on visibility."** Not a signal: 39/54 =
  72.2% of *all* Terra `unsupported` texts contain "clear/clearly", and pooled
  across cohorts recovered cards are marginally *less* hedged (70.0% vs 77.4%,
  OR 0.68, p = 0.74).
- **Any kappa on the repeat arm.** With 9/1 marginals only 8 or 10 agreements
  are attainable, so the statistic is forced, not measured.
- **The canary/production "3.5×" split.** The difference CI spans zero.
- **`trivial` concentration as a significant test.** It is a defensible
  descriptive observation; the significance claim does not survive permutation
  or post-hoc correction.

## Limits

- **One reviewer, non-independent, throughout.** There is no inter-rater figure
  anywhere in this program. Do not write "verified", "confirmed" or "validated"
  of any v1.1 result. Per `DESIGN_label_v1_1_adjudication.md` §6, the honest
  claim is "re-adjudicated under a schema that separates the axes".
- **The 8/10 repeat figure licenses very little.** 95% Wilson [49.0%, 94.3%] —
  it cannot distinguish a 50% reviewer from a 95% one. All 20 repeat answers
  fell inside `{exact_and_warranted, misnamed_but_warranted}`, and **both
  disagreements sit on the exact-vs-misnamed boundary** — precisely the boundary
  separating `supported_billed` from `misnamed_billed`. Weighted to the canary
  population, consistency is ~74.6%. The repeat arm has **zero** coverage of
  `absent`, `trivial`, `inconclusive` and dirB, structurally: repeats were drawn
  only from blind arms, which are 100% `supported_billed`.
- **Review pace.** Median inter-answer gap **8 s** (7 s excluding breaks over 10
  minutes, n = 160) across a 23.8 h span, against the design's ~45 s budget,
  with 5 free-text notes on 135 cards. This is a signal about the input, not the
  tooling.
- **Coverage.** 125 of 173 condition cards; **0 of 22 package** and **0 of 17
  bathroom** cards; 32 unreviewed production conditions; two production listings
  excluded wholesale for thumbnail resolution (not an artifact — min short side
  across served cards is 651 px).
- **Denominators are inherited, not re-derived.** An error in the frozen
  population enumeration propagates identically into v1 and v1.1.
- v1.1 re-measures **labels**, not Terra. No claim about model behaviour,
  package accuracy, Pass-2f correctness, Terra recall, or pricing follows from
  it.

## Recommendation

The bottleneck is **review coverage, not tokens**. In priority order:

1. **Adjudicate the 20 unasked canary dirB cards** (17 v1-`unsupported`,
   3 `inconclusive`) — **~15 min**. This is the only place the coarsening
   hypothesis can be tested, and it supplies the missing dirB *loss* population.
   Exclude the 17 `terra_flip`/`p6_forced_single` cards (they return
   `out_of_scope`) and the 11 production dirB cards (the scorer filters to
   canary).
2. **A blind repeat set covering the `absent` boundary** — ~30 fresh-id,
   provenance-withheld cards enriched with v1 error-class cards, **~25 min**.
   The current repeat arm cannot speak to the boundary the headline rests on.
   It cannot reuse phase 2, which is non-blind by construction.
3. **Expand the canary uniform arm** — 671 unreviewed canary accepted
   conditions; a 200-card enriched draw is **~2.5 h**, a full census ~8.4 h at
   45 s/card. This is the only action that fixes the 2.392 pp/card leverage
   problem. Expected yield ≈ 17 further canary hard-false cards → n ≈ 20, enough
   for ~80% paired power at a 40% true flip rate. Zero tokens.
4. **A second reviewer on a 40–60 card overlap sample**, weighted toward the
   exact/misnamed and absent boundaries — **~45 min of a second person's time**.
   Per §6 this is the only thing that converts "re-adjudicated" into "verified".
5. **Re-derive Session B's control-arm thresholds** against the directional
   floors (4.05% billed / 23.1% dirB) and the corrected class sizes; the n = 11
   and n = 6 thresholds are void, n = 16 survives. **~1 h analyst, 0 tokens.**
6. **Then** run the same-prompt control arm alone (~1.8M tokens) and size the
   variant arms only after its floors land.

The severity/threshold follow-up the re-tag opened is independently actionable
today at zero token cost, on `older_flooring_style`, `cabinets_dated_style` and
`dated_interior_doors` — subject to the 1:1 collateral noted above. Catalog
changes go through `tools/catalog_migrations/kind_v2_decisions.json` +
regeneration, never hand-edits.
