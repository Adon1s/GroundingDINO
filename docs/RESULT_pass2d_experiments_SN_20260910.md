# Pass 2d — arms S, N and the Terra resolver comparison: result (2026-09-10)

Authorized by Steven 2026-09-10 within the 2.5M Terra budget; local Qwen authorized. Observation text and candidate lists held fixed from the stored artifacts; retrieval never re-run. The harness mirrors `run_pass_2d` exactly except that the lexical shortcut is bypassed (arm S) and the user template is swappable (arm N). Consequential rows repeated 5x. 525 resolver calls, 0 failures.

**Harness fidelity.** Under the unchanged prompt, local Qwen reproduced the stored selection on 39 of 40 non-null control rows and declined on 42 of 42 stored declines. The comparison baseline is sound.

## Arm S — Rule E (bare subject noun must match the sentence head)

25 rows, routed to the model instead of shortcutting. 14 unchanged, 4 changed item, 7 to null. Of the 11 consequential rows, 7 stable across 5 replicas and 4 unstable.

| Stable gains | | |
|---|---|---|
| "The floor has discoloration and staining around the tub and vanity." | fixtures → `worn_or_stained_flooring` | 5/5. **Human-confirmed** (C5): exactly the item Steven preferred |
| "Lower cabinets around the sink … are stained." | plumbing fixture → `cabinets_worn_finish` | 5/5 |
| "Wall outlets and details near the window appear old-looking." | windows → `dated_electrical_outlets_switches` | 5/5 |

| Costs | | |
|---|---|---|
| "Caulk at the tub-to-wall and tub-to-floor joints appears tired." | fixtures → null | 5/5 stable loss |
| "Dark staining is visible along the roof edge." | roof shingles → null | 3/5, unstable |
| "Visible seams or gaps … along the foundation area." | foundation → null | 4/5, unstable |
| "Ceiling above the shower has discoloration and staining." (D1) | fixtures → null 3/5, paint 2/5 | **unstable**; the item Steven's answer implies (`water_stain_ceiling`) is kind-gated out of the list, so Rule E cannot reach it |

Three further nulls are defensible removals of a wrong subject (wall material near the sink, wiring near the window, countertops near the sink): the shortcut's item was wrong and nothing listed was right.

## Arm N — one added prompt rule, local Qwen, base vs modified

| Row | Base | Modified | Result |
|---|---|---|---|
| C6 "The range appears older." | null 5/5 | `appliances_dated_or_basic` 5/5 | anchor moved, stable |
| C7 "The faucet has an older style." | null 4/5 | `outdated_bathroom_finishes` 5/5 | anchor moved, stable |
| C8 "Kitchen is functional and very dated." | null 4/5 | null 3/5 | anchor did not move |
| D2 swirl plaster (must stay null) | null | **null 5/5** | held |
| D3 laundry cabinets, right item absent (must stay null) | null | **null 5/5** | held |
| 40 correct-decline controls (limit 4) | — | **0 moved** | within |
| 40 non-null controls (limit 2) | — | **2 moved** | at limit: one gain (shower enclosure), one loss (paved surface wear → null) |

C8 not moving is coherent rather than a failure: the added rule fires only when the sentence names a specific instance of a candidate's subject, and "functional and very dated" names none. The rule declined to overreach, which is the property that keeps D2 and D3 null.

## Terra as resolver (unchanged prompt, identical candidate lists)

| | Result |
|---|---|
| Anchors | all three: C6 5/5, C7 5/5, C8 3/5 (unstable) |
| D2 must stay null | **`popcorn_or_acoustic_ceiling_texture` 5/5** — the claim Steven called "straight up false" |
| D3 must stay null | **`dated_overall_decor_style` 5/5** — selects a generic item where the right item is absent |
| 40 correct-decline controls (limit 4) | **11 moved — breach** |
| 40 non-null controls (limit 2) | **4 moved — breach** |
| Stored declines still declining | 29/42, against Qwen's 42/42 |

Terra buys the third anchor and pays with a much looser null, including a stable false claim on the one row where Steven pre-judged that exact item false. Terra verification was not run: it verifies the claim of whatever item was selected and cannot distinguish a wrong-but-plausible selection, so it could not have adjudicated any of this.

## Tokens

| Model | Calls | Tokens |
|---|---|---|
| Local Qwen 3.6 27B | 287 | 198,702 |
| GPT-5.6 Terra (resolver only) | 238 | 162,893 |

Terra: 162,893 of 2,500,000 authorized (6.5%); 2,337,107 remaining. No Terra verification calls. No production artifact touched, no catalog change, no arm R.

## Recommendation

**Implement N.** Two of three anchors fixed stably, zero movement on 40 correct declines, non-null controls at the stated limit, and both must-stay-null controls held at 5/5 including the one Steven judged false. It is a two-line addition to one prompt template with a measured cost of one lost selection in forty.

**Do not implement S yet.** It fixes the human-confirmed wrong-subject case stably and two others, but sends three plausible rows to null and leaves four of eleven consequential rows unstable. Net it is close to a wash. The reason to revisit rather than drop it: S's failure mode *is* the null, and N reduces nulls. The two changes interact and were tested independently. Re-running S on top of N is the cheap next step and needs no new photo review.

**Do not change the resolver.** It breaches both control limits and produces a stable false claim on D2.

## The one case that stays open

D1, "Ceiling above the shower has discoloration and staining." Steven confirmed the ceiling is stained and the fixtures are not, so the shortcut was wrong; but the item his answer implies, `water_stain_ceiling`, is kind `defect` while Pass 2c called the sentence `degradation`, so it was never in the list. Neither S nor N can reach it. This is the Pass 2c kind boundary, which no instrument in the program has ever measured, and it is outside the authorized scope.
