# Session B — control-arm design note (QP1 harness)

Written 2026-08-27 · companion to `scripts/redecide_renovation_architecture.py`
and `scripts/score_redecide_variants.py`. How many cards decide, at what
threshold, given the measured 6.4% replica noise floor.

## The measurement problem

The scoreable population is 125 canary condition cards, but the decisive
subsets are small: **11** hard-false billed (win side), **16** dirB
human-supported (win side, goal 2), **55** human-supported billed (loss
side), **6** overstated (cost side, reported but never netted). Terra's
replica-to-replica flip rate on identical inputs is **6.4% (16/250)**, and
humans judged the two replicas' disagreements 8:6 — a coin toss. Any
variant-vs-stored comparison therefore contains ~1–2 expected flips of pure
noise on the win-side subsets alone, which is why the harness requires a
**same-prompt control arm** re-run on the same 18 listings and the scorecard
prints every rate next to the control column.

## What the control arm is

`--control`: the rebuilt request, byte-identical prompts (asserted), own
fingerprint namespace, re-submitted live. Its measured per-class flip counts
are the empirical noise floor for that day/model — the 6.4% prior is the
planning number, the control column is the evidence.

## Decision thresholds (planning numbers at the 6.4% prior, α ≈ 0.05)

- **Hard-false wins (n=11):** noise expectation 0.064 × 11 ≈ 0.7 flips;
  P(≥2) ≈ 0.15, P(≥3) ≈ 0.03 under the noise-only null. Threshold: a
  variant must flip **≥ 3/11** hard-false cards to unsupported, and clear
  the control arm's observed count by ≥ 2, before its win-side is treated
  as signal.
- **dirB recoveries (n=16):** noise expectation ≈ 1.0; P(≥4) ≈ 0.02.
  Threshold: **≥ 4/16** recoveries to supported, and ≥ 3 over control.
- **Supported losses (n=55):** noise expectation ≈ 3.5; the SD of the
  variant-minus-control difference is ≈ √(2·55·0.064·0.936) ≈ 2.6.
  Threshold: losses may not exceed the control arm's losses by **more than
  2**; a variant that wins on hard-false but exceeds this band is not
  shippable as-is.
- **Overstated deletions (n=6):** each deletion is a cost line in the
  scorecard. No netting against wins — the right fix for these cards is
  wording (QP2's catalog half), not exclusion.

## What n=11 cannot resolve in one pass

A true 30% hard-false flip rate yields ~3.3 expected wins vs ~0.7 noise —
detectable in one pass. A true 15% (~1.7 expected) is not distinguishable
from noise at n=11. So: a variant that lands between the noise floor and
the ≥3/11 threshold is **inconclusive, not dead** — the resolution path is
a second independent pass (~1.8M further Terra tokens for variant+control),
not a lowered threshold. Two passes double the effective n and let flips be
demanded to *repeat*, which is a much stronger filter than one-pass counts.

## Ship rule

A variant is a candidate winner only when all three hold against the same
control arm: hard-false wins ≥ 3/11 (and ≥ control + 2), supported losses
≤ control + 2, and dirB recoveries reported (≥ 4/16 where goal 2 is the
variant's purpose). Anything marginal → repeat pass before wave-2 canary
commitment. The gate after Session F (roadmap gate 3) consumes this
scorecard shape directly.
