# Result — mechanism vs perception re-tag

Written 2026-08-28 · 46/46 cards complete · zero provider calls · frozen
evidence preserved

## Decision

The formal result is **inconclusive** under the pre-committed rule.

- The primary canary weighted view is **mixed**: `mechanism_only` accounts for
  60.325 / 100.875 = **59.8%** of measured billed-error mass.
- Canary raw cards and the combined 26-card raw view are also mixed, but both
  production views are **perception-limited**. That disagreement alone makes
  the overall result inconclusive.
- Group A contains one `cannot_tell`; the rule independently makes any
  `cannot_tell` in either group disqualifying. It is retained in the
  denominator and is not counted as perception.

Session F must therefore run all three pre-specified wording arms:
**rubric-only, coarsened-only, and combined**. Fine mechanism wording is not
retired on this evidence. The existing structured-fields and
remove-observations hypotheses remain live and are not decided by this
re-tag.

A separate severity/threshold follow-up also opens: `too_trivial` appears on
two listings and represents **20.6%** of canary Group A mass, above the
pre-committed 10% gate.

## Integrity and reproducibility

The hashes below were recorded before the tally was generated and verified
again after the review. Start and end values are identical.

| file | starting sha256 | ending sha256 |
|---|---|---|
| `reports/retag_queue.json` | `aac934b72102fe91be1d89a249474ea05d1fba54994cba532bb748e230c4f78a` | `aac934b72102fe91be1d89a249474ea05d1fba54994cba532bb748e230c4f78a` |
| `reports/retag_verdicts.jsonl` | `a038029811c9fafe78f6e2cd3d2456d26ad6499018a217ff66ecb4c13901a4a7` | `a038029811c9fafe78f6e2cd3d2456d26ad6499018a217ff66ecb4c13901a4a7` |
| `reports/review_queue.json` | `8512b87c2af18d1104069c15a5eda977d5fe79eb736f9aedd976d174f89d318a` | `8512b87c2af18d1104069c15a5eda977d5fe79eb736f9aedd976d174f89d318a` |
| `reports/review_verdicts.jsonl` | `0c7ca6a8b55d6dac69021a00e81315be1b15cef68ffd59fbe2973cb895a8d70f` | `0c7ca6a8b55d6dac69021a00e81315be1b15cef68ffd59fbe2973cb895a8d70f` |
| `reports/review_analysis.json` | `3652ba3c3feb15800c064ffafd076d15cd9773de29c3d77c7b383fe26a259c9a` | `3652ba3c3feb15800c064ffafd076d15cd9773de29c3d77c7b383fe26a259c9a` |
| `reports/review_analysis.md` | `6b025d47482ee0159d0103a919d6bbf6c12fd8fa7164059c95b82567ba967213` | `6b025d47482ee0159d0103a919d6bbf6c12fd8fa7164059c95b82567ba967213` |

The re-tag queue embeds the same full hashes for the frozen source queue and
source verdicts. An independent replay of the builder's selection logic,
performed without writing a queue, reproduced the exact ID sets: Group A
26/26 with no missing or extra IDs and Group B 20/20 with no missing or extra
IDs.

| integrity check | result |
|---|---:|
| queue cards / unique queue IDs | 46 / 46 |
| raw verdict lines / unique completed verdicts | 46 / 46 |
| Group A membership | 26 (17 canary, 9 production) |
| Group B membership | 20 (16 canary, 4 production) |
| missing verdicts / orphan verdict IDs | 0 / 0 |
| duplicate IDs or overwrite evidence | 0 |
| undo markers / peeked cards | 0 / 0 |
| verdicts outside the group vocabulary | 0 |
| cards with free-text notes | 13 |

`scripts/retag_tally.py` was run twice. Both runs produced the same 2,321-byte
`reports/retag_tally.md`, sha256
`f214685adadd1474b71962542e74314f942b7bd73d267af24a510d26c768ea29`.
The queue builder was not run, and none of the frozen inputs was rebuilt.

An independent computation from the JSON/JSONL inputs, without importing the
tally's weighting functions, reproduced:

- uniform weights: canary **17.775**, production **13.142857** (reported as
  **13.143**);
- Group A: canary **17 cards / 100.875 mass**, production **9 cards / 21.143
  mass**;
- Group B: exactly **20 raw cards**.

## Quantitative result

### Group A — known billed errors

| source | answer | cards | weighted mass | source-mass share |
|---|---|---:|---:|---:|
| canary | `mechanism_only` | 10 | 60.325 | 59.8% |
| canary | `wholly_false` | 3 | 19.775 | 19.6% |
| canary | `too_trivial` | 4 | 20.775 | 20.6% |
| canary | `cannot_tell` | 0 | 0.000 | 0.0% |
| production | `mechanism_only` | 1 | 1.000 | 4.7% |
| production | `wholly_false` | 6 | 18.143 | 85.8% |
| production | `too_trivial` | 1 | 1.000 | 4.7% |
| production | `cannot_tell` | 1 | 1.000 | 4.7% |

The classification check uses `mechanism_only` as the numerator and keeps all
other answers in the denominator.

| required view | numerator / denominator | share | band |
|---|---:|---:|---|
| canary weighted (primary) | 60.325 / 100.875 | 59.8% | mixed |
| canary raw | 10 / 17 | 58.8% | mixed |
| production weighted | 1.000 / 21.143 | 4.7% | perception-limited |
| production raw | 1 / 9 | 11.1% | perception-limited |
| combined raw | 11 / 26 | 42.3% | mixed |

The primary view does not cross the language-dominant threshold. The
production disagreement and the retained `cannot_tell` each force the formal
overall classification to **inconclusive**, irrespective of the qualitative
notes.

The `too_trivial` gate is also positive: five cards on two listings, with
20.775 / 100.875 = **20.6%** of canary mass and 1.000 / 21.143 = **4.7%** of
production mass. This opens a separate threshold/severity investigation; it
does not change the wording classification.

### Group B — targeted dropped conditions

| answer | raw cards | share of targeted 20 |
|---|---:|---:|
| `wording_blocked` | 6 | 30% |
| `terra_miss` | 7 | 35% |
| `borderline` | 7 | 35% |
| `cannot_tell` | 0 | 0% |

These are targeted counts only. The six `wording_blocked` answers are
supporting evidence that coarsening may recover some real conditions, but
they are not a recall estimate and cannot override Group A's gate.

| cut | cards | `wording_blocked` | `terra_miss` | `borderline` |
|---|---:|---:|---:|---:|
| canary | 16 | 5 | 7 | 4 |
| production | 4 | 1 | 0 | 3 |
| defect | 5 | 2 | 3 | 0 |
| degradation | 10 | 4 | 3 | 3 |
| modernization | 5 | 0 | 1 | 4 |

## Pre-specified explanatory cuts

All cuts in this section are exploratory small-number descriptions. They do
not support causal claims, and no inference is made from photo-count or batch
dimensions.

### Group A by source and sampling stratum

| source / stratum | cards | total mass | answer counts | answer mass |
|---|---:|---:|---|---|
| canary / dirA | 12 | 12.000 | mechanism 7; false 2; trivial 3 | mechanism 7.000; false 2.000; trivial 3.000 |
| canary / uniform | 5 | 88.875 | mechanism 3; false 1; trivial 1 | mechanism 53.325; false 17.775; trivial 17.775 |
| production / dirA | 8 | 8.000 | mechanism 1; false 5; trivial 1; cannot tell 1 | mechanism 1.000; false 5.000; trivial 1.000; cannot tell 1.000 |
| production / uniform | 1 | 13.143 | false 1 | false 13.143 |

### Group A by prior verdict and catalog kind

| cut | cards | total mass | answer counts | answer mass |
|---|---:|---:|---|---|
| prior `overstated` | 6 | 22.775 | mechanism 6 | mechanism 22.775 |
| prior `unsupported` | 20 | 99.243 | mechanism 5; false 9; trivial 5; cannot tell 1 | mechanism 38.550; false 37.918; trivial 21.775; cannot tell 1.000 |
| degradation | 14 | 93.243 | mechanism 9; false 5 | mechanism 59.325; false 33.918 |
| modernization | 12 | 28.775 | mechanism 2; false 4; trivial 5; cannot tell 1 | mechanism 2.000; false 4.000; trivial 21.775; cannot tell 1.000 |

All six prior `overstated` cards became `mechanism_only`, which is consistent
with the semantic difference between an overstated real condition and an
unsupported claim. The modernization cut contains all five `too_trivial`
answers, but its weighted result is driven by one canary uniform card and
must not be generalized.

### Repeated catalog items only

Group A:

| catalog item | cards | answer counts | answer mass |
|---|---:|---|---|
| `cabinets_dated_style` | 2 | trivial 1; false 1 | trivial 17.775; false 1.000 |
| `dated_interior_trim` | 4 | mechanism 1; false 2; cannot tell 1 | mechanism 1.000; false 2.000; cannot tell 1.000 |
| `hard_flooring_scratched_or_worn` | 2 | mechanism 1; false 1 | mechanism 1.000; false 1.000 |
| `older_flooring_style` | 3 | trivial 3 | trivial 3.000 |
| `peeling_or_discolored_paint` | 4 | mechanism 2; false 2 | mechanism 2.000; false 14.143 |
| `worn_or_stained_carpet` | 2 | false 2 | false 18.775 |
| `worn_or_stained_flooring` | 2 | mechanism 2 | mechanism 18.775 |

Group B:

| catalog item | cards | raw answers |
|---|---:|---|
| `baseboard_wear_scuffs` | 2 | wording blocked 1; borderline 1 |
| `brick_weathered_or_discolored` | 2 | wording blocked 2 |
| `exterior_siding_discoloration_fading` | 2 | Terra miss 1; borderline 1 |
| `paint_refresh_recommended` | 4 | borderline 4 |
| `peeling_or_discolored_paint` | 2 | Terra miss 1; borderline 1 |
| `vinyl_linoleum_torn_or_lifted` | 2 | wording blocked 2 |

No item-level action is warranted from these repeated but still small cells.

## Qualitative note audit

The theme column is deliberately restrained. It summarizes the note without
changing the recorded answer or the formal classification.

| card ID | group | answer | prior verdict / note | new note | restrained theme |
|---|---|---|---|---|---|
| `rc_4cab6e7be02b` | A | `mechanism_only` | unsupported / none | There is a patch on the drywall. Which I suppose is understandably interpreted as discolored paint. perhaps this one is not problematic | plausible alternate visible cue; reviewer uncertainty |
| `rc_f3d19e171c6d` | A | `mechanism_only` | unsupported / none | Crap this one I messed up. The floor is soiled in multiple photos | reviewer correction: condition visible |
| `rc_eef1daf73e2a` | A | `mechanism_only` | overstated / I see stains not scuffs | I see stains not scuffs | mechanism mismatch: stain versus scuff |
| `rc_6f59ce859fb7` | A | `mechanism_only` | overstated / none | I messed up again. Flooring staining is fair | reviewer correction: claim considered fair |
| `rc_3da266b97f62` | A | `mechanism_only` | unsupported / none | This is also another one of my mistakes. The LLM claim is true they are dated | reviewer correction: dated claim considered true |
| `rc_165093fb25c2` | A | `too_trivial` | unsupported / none | They might be slightly dated but not worth replacing in my opinion | severity / replacement threshold |
| `rc_022497229bd3` | A | `wholly_false` | unsupported / none | I think the trim looks fine | no visible problem judged |
| `rc_3a19f759f2d9` | A | `mechanism_only` | unsupported / none | I would reverse my decision here. I think it is in need of work so the LLM is right | reviewer correction: work considered warranted |
| `rc_470fb7e0624f` | A | `cannot_tell` | unsupported / none | Cannot see trim in the photo | insufficient visible evidence |
| `rc_48ea825e33f8` | B | `wording_blocked` | supported / none | Another one where I messed up. The LLM wording isnt ideal but the claim is legit | wording mismatch with a real condition |
| `rc_5b3b3cb9e726` | B | `wording_blocked` | supported / none | This is my miss, terra is correct | reviewer correction; answer/note tension |
| `rc_9c702893a6ce` | B | `wording_blocked` | supported / none | The linoleum needs replacing but i was wrong that it appears worn. it appears fine from the photos | replacement rationale differs from visible mechanism |
| `rc_8907b19a4a2f` | B | `borderline` | supported / none | Hard to say if the fireplace paint is intentional like that or not | intentional-versus-defect ambiguity |

Several notes explicitly revise the reviewer's earlier call. That is useful
context for why the split looks the way it does, but it also reinforces the
non-blind limitation and cannot repair the cross-source disagreement.

## Recommendation and limits

1. Session F runs rubric-only, coarsened-only, and combined wording arms, each
   against the same-prompt control. The scorecard, not this re-tag, selects the
   mechanism/rubric/coarsening shape.
2. Keep structured-fields and remove-observations as independent harness
   hypotheses. This exercise did not test either one.
3. Do not retire fine mechanism wording yet. Production's perception-limited
   result keeps corroboration, escalation/inspection, and retirement
   candidates relevant if the live arms fail.
4. Open a separate severity/threshold follow-up for the `too_trivial` result;
   do not fold it into the wording arms.
5. Use Group B only as targeted evidence that coarsening has a plausible
   recovery path. Do not report its 6/20 as recall or as a population rate.

Known limitations are material: weighted mass is directional and has
one-card sensitivity; the reclassification was non-blind; this is not an
error-rate re-measurement; Group B was targeted; and there is still no
evidence that coarse claims would pass Terra. The small cuts are exploratory,
and no causal interpretation is assigned to photo or batch dimensions.
