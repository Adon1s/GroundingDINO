# Re-tag tally — mechanism (language) vs perception

46 of 46 cards answered. Group A is weighted error mass (dirA census = 1, uniform = accepted_non_dirA/n_uniform), reconciling with `reports/review_analysis.md` §9; group B is raw targeted counts.

## Group A — the billed errors: is a real problem there?

| source | answer | cards | error mass | share of that source's error mass |
|---|---|---|---|---|
| canary | mechanism_only | 10 | 60.32 | 59.8% |
| canary | too_trivial | 4 | 20.77 | 20.6% |
| canary | wholly_false | 3 | 19.77 | 19.6% |
| production | cannot_tell | 1 | 1.00 | 4.7% |
| production | mechanism_only | 1 | 1.00 | 4.7% |
| production | too_trivial | 1 | 1.00 | 4.7% |
| production | wholly_false | 6 | 18.14 | 85.8% |

**canary: 60% of measured billed-error mass is mechanism-only** — a correctly-worded (or coarser) claim would have been true. The remaining 40% is not a wording problem.

**production: 5% of measured billed-error mass is mechanism-only** — a correctly-worded (or coarser) claim would have been true. The remaining 95% is not a wording problem.

## Group B — the dropped real conditions: why was it rejected?

| answer | cards |
|---|---|
| borderline | 7 |
| terra_miss | 7 |
| wording_blocked | 6 |

Targeted stratum (Terra-rejected, 2f-confirmed) — counts only, not a population rate.

### by catalog item (2+ cards)

| catalog item | answers |
|---|---|
| baseboard_wear_scuffs | wording_blocked 1, borderline 1 |
| brick_weathered_or_discolored | wording_blocked 2 |
| exterior_siding_discoloration_fading | terra_miss 1, borderline 1 |
| paint_refresh_recommended | borderline 4 |
| peeling_or_discolored_paint | terra_miss 1, borderline 1 |
| vinyl_linoleum_torn_or_lifted | wording_blocked 2 |

## Reading

- High `mechanism_only` / `wording_blocked` share => the system SEES correctly and NAMES badly. Fixable by the Terra rubric (QP2) and/or by coarsening the degradation claim text so the claim stops asserting a distinction the photo cannot settle. Both ride one canary.
- High `wholly_false` / `terra_miss` share => a perception limit. Wording work will not move it; the levers are corroboration and the escalation/inspection lane.
- High `too_trivial` => neither: a severity/threshold problem, not truth.

