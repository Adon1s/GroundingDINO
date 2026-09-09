# Session 8 — floor-vs-sum residual trace (delta-review input)

Date: 2026-08-17. Branch: `renovation_architecture_rework`.
Method: `scripts/analyze_session8_canary.py` over the Session 8 baseline
replay (current code incl. the b77b388 factor fix, replica-1 verdicts/Sol
decisions, all 18 properties). Sidecar:
`artifacts_canary/renovation_session6_20260816_02/analysis_session8/floor_trace.json`.

## What was measured

Per property, the exact decomposition identity

```
headline_v5 − v4_final_rehab =
      class_standalone            (48-class work billed standalone in the ledger)
    + class_absorbed              (48-class owned-child allowances inside applied packages)
    + package_floor_lift          (Σ max(0, effective − Σ owned-child allowances) per applied package
                                   — dollars the tier-spec floor bills above child truth,
                                   reconciliation.py:171-179)
    + residual                    (everything else: v5-vs-v4 pricing of shared scope,
                                   dedup differences, v4's own package/cap behavior)
```

"48-class" = v4-non-estimable but v5-billable catalog items; a work item
counts as class only if ALL its catalog items are in the class (zero mixed
merged items existed in this corpus, so attribution is exact).

## Corpus result (pre-triage baseline)

| component | low | high |
|---|---|---|
| delta (v5 − v4) | +52,789 | +469,719 |
| class_standalone | +38,028 | +419,870 |
| class_absorbed | +29,077 | +239,752 |
| package_floor_lift | +463,193 | +640,044 |
| residual | −477,509 | −829,947 |

(class_standalone and class_absorbed match the 67-item handoff's measured
split to the dollar — independent validation of the replay chain.)

## Reading

1. **The class explains the headline residual.** Class scope contributes
   +$659.6k high across both lanes against a +$469.7k total delta — without
   it, v5 would price this corpus BELOW v4 (residual + floor lift net to
   −$190k high on non-class scope).
2. **The floor lift is large but not the story.** Applied packages bill
   +$463k/+$640k above their owned-child truth — but v4's own
   package-adjusted pricing does something comparable, and the negative
   residual (v5 pricing shared scope lower: max-envelope dedup, per-unit
   splits, no v4 stacking) absorbs it. Floor lift and residual are two halves
   of "packages price differently than child sums in both engines"; neither
   endpoint of the pair is independently actionable without price
   calibration, which is out of migration scope.
3. **Post-triage state.** The Session 8 five-item triage removes the
   opportunity/presence subset ($6.8k low / $129.8k high of headline),
   leaving the corpus at **+7.7% low / +17.6% high vs v4** — of which the
   surviving class scope (severity-3 defects, wear items — deliberately
   accepted as intended v5 surfacing) is effectively the entire high side.

## Per-property decomposition (baseline, low / high)

| property | delta | class_standalone | class_absorbed | floor_lift | residual |
|---|---|---|---|---|---|
| redfin_10803207 | +8,400 / +23,448 | 721 / 3,605 | 533 / 2,669 | 18,738 / 26,315 | −11,592 / −9,141 |
| redfin_10806500 | +11,296 / +39,856 | 906 / 7,994 | 26 / 133 | 18,816 / 24,803 | −8,452 / +6,926 |
| redfin_10952874 | −7,563 / −21,588 | 1,695 / 14,879 | 1,642 / 15,297 | 23,183 / 24,669 | −34,083 / −76,433 |
| redfin_11000447 | +2,681 / +11,219 | 781 / 6,183 | 1,398 / 13,834 | 18,849 / 13,983 | −18,347 / −22,781 |
| redfin_11077450 | +26,371 / +87,164 | 3,620 / 53,279 | 1,047 / 7,802 | 32,344 / 45,490 | −10,640 / −19,407 |
| redfin_11079485 | −4,510 / +49,156 | 2,552 / 44,789 | 701 / 6,051 | 6,943 / 14,697 | −14,706 / −16,381 |
| redfin_11185681 | +5,553 / +16,270 | 1,882 / 17,288 | 1,896 / 15,601 | 50,254 / 96,623 | −48,479 / −113,242 |
| redfin_125779232 | +9,893 / +39,520 | 1,402 / 13,568 | 1,514 / 14,132 | 32,832 / 40,288 | −25,855 / −28,468 |
| redfin_125970550 | −4,653 / +17,421 | 2,675 / 27,094 | 3,842 / 32,947 | 30,442 / 30,497 | −41,612 / −73,117 |
| redfin_126224899 | −5,654 / +418 | 2,573 / 25,450 | 3,672 / 29,396 | 41,100 / 71,325 | −52,999 / −125,753 |
| redfin_126418713 | −1,557 / +12,151 | 2,650 / 21,671 | 652 / 3,256 | 18,616 / 14,845 | −23,475 / −27,621 |
| redfin_127468088 | +9,852 / +50,953 | 1,424 / 17,232 | 2,365 / 23,805 | 35,387 / 47,151 | −29,324 / −37,235 |
| redfin_166147710 | −4,690 / +12,071 | 4,218 / 34,464 | 719 / 4,356 | 16,105 / 21,642 | −25,732 / −48,391 |
| redfin_25809814 | +1,509 / +45,367 | 3,637 / 59,610 | 1,022 / 5,108 | 21,223 / 22,731 | −24,373 / −42,082 |
| redfin_80877597 | −5,829 / −12,557 | 1,014 / 8,432 | 1,394 / 10,328 | 15,223 / 9,748 | −23,460 / −41,065 |
| redfin_80925528 | +4,380 / +47,902 | 2,400 / 30,002 | 3,225 / 24,626 | 37,155 / 52,540 | −38,400 / −59,266 |
| redfin_80990371 | +5,597 / +50,614 | 2,954 / 29,714 | 2,235 / 19,767 | 36,377 / 62,346 | −35,969 / −61,213 |
| redfin_81000709 | +1,713 / +334 | 924 / 4,616 | 1,194 / 10,644 | 9,606 / 20,351 | −10,011 / −35,277 |

Worst-delta properties (redfin_11077450, redfin_10806500): the delta is
dominated by class scope (11077450: $53.3k of $87.2k high is class
standalone — largely `unfinished_basement_present`, removed by the triage)
plus floor lift partially offset by negative residual. No property shows an
unexplained component.

## How this feeds the canary delta review

For each >15% headline delta in the rerun, the accepted explanation is one
of: (a) surviving class scope (intended v5 surfacing — the Session 8
decision), (b) floor-vs-residual package pricing difference (systemic,
symmetric with v4's own package pricing, deferred to price calibration), or
(c) something this table does not explain — which is a defect to trace, not
to accept. Components (a)/(b) are recomputable per property from
`floor_trace.json`.
