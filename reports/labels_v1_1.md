# Labels v1.1 — corrected label overlay

v1 is unchanged and remains the published result. This overlay is the
development label source; every number below is printed against its v1
original.

## Coverage

| arm | cards | answered | pending |
|---|---|---|---|
| blind_canary | 55 | 55 | 0 |
| blind_production | 24 | 24 | 0 |
| reask | 46 | 46 | 0 |

Canary label set complete: **True** (125/125 cards answered overall).

## Class movement (v1 -> v1.1)

| v1 class | -> v1.1 class | cards |
|---|---|---|
| dirB_recovery | dirB_recovery | 19 |
| dirB_recovery | dirB_wording_recovery | 1 |
| hard_false_billed | hard_false_billed | 9 |
| hard_false_billed | trivial_billed | 4 |
| hard_false_billed | supported_billed | 3 |
| hard_false_billed | misnamed_billed | 2 |
| hard_false_billed | excluded | 2 |
| overstated_billed | supported_billed | 3 |
| overstated_billed | misnamed_billed | 3 |
| supported_billed | supported_billed | 74 |
| supported_billed | misnamed_billed | 3 |
| supported_billed | trivial_billed | 2 |

23 of 125 adjudicated cards changed class.

## Canary scoreable population — what Session F will tune against

| class | v1 | v1.1 |
|---|---|---|
| dirB_recovery | 16 | 16 |
| hard_false_billed | 11 | 3 |
| misnamed_billed | — | 7 |
| overstated_billed | 6 | — |
| supported_billed | 55 | 57 |
| trivial_billed | — | 5 |

**Hard-false canary cards: 11 -> 3.** This is the win-opportunity count the variant scorecard depends on. Redo Session B's control-arm sample-size note against it before spending Terra tokens: against a 6.4% replica-flip floor, a smaller class means a smaller separation between arm and control.

## Reviewer self-consistency (the repeat arm)

- exact agreement: **8/10** (80%)
- claim-axis agreement: **8/10** (80%)

This is the human noise floor. Every rate in this program — and the 6.4% Terra replica floor it is compared against — should be read next to it.

| card | item | primary | repeat |
|---|---|---|---|
| adj_d008895ad9f7 | boarded_up_entry_or_window | misnamed_but_warranted | exact_and_warranted |
| adj_1994db949edf | peeling_or_discolored_paint | exact_and_warranted | misnamed_but_warranted |

## Re-tag agreement (validation, not a label source)

| re-tag answer | met | diverged | n/a | v1.1 claim/work |
|---|---|---|---|---|
| borderline | 0 | 0 | 7 | -> exact/warranted 7 |
| cannot_tell | 1 | 0 | 0 | -> inconclusive/inconclusive 1 |
| mechanism_only | 5 | 6 | 0 | -> exact/warranted 6, -> misnamed/warranted 5 |
| terra_miss | 7 | 0 | 0 | -> exact/warranted 7 |
| too_trivial | 4 | 1 | 0 | -> absent/none 1, -> exact/trivial 4 |
| wholly_false | 8 | 1 | 0 | -> absent/none 8, -> inconclusive/inconclusive 1 |
| wording_blocked | 1 | 5 | 0 | -> exact/warranted 5, -> misnamed/warranted 1 |

Divergence is expected on `mechanism_only`: its own notes split between claims that were actually right and claims that were misnamed. That split is why the re-tag answers were re-asked rather than translated.
