# Pass 2a prompt ablation — benchmark report

Generated 2026-08-09T21:02:06.905145+00:00. Dollar figures are **pre-2f, all packages assumed confirmed** benchmark totals, not production headlines.

## Attribution gate: PASSED
- redfin_10806500: median midpoint $72,925, spread $20,500, 5 partial priced ids
- redfin_11000447: median midpoint $88,750, spread $5,660, 4 partial priced ids

## Stability — baseline
- mean resolved-ID Jaccard: 0.53
- median midpoint spread: $23,230.0
  - redfin_10806500: Jaccard 0.524, spread $28,875, 4 partial priced ids
  - redfin_11000447: Jaccard 0.535, spread $17,585, 8 partial priced ids

## Stability — checklist
- mean resolved-ID Jaccard: 0.635
- median midpoint spread: $22,495.5
  - redfin_10806500: Jaccard 0.669, spread $7,950, 8 partial priced ids
  - redfin_11000447: Jaccard 0.601, spread $37,041, 5 partial priced ids

## Stability — __frozen_2a_attribution__
- mean resolved-ID Jaccard: 0.77
- median midpoint spread: $13,080.0
  - redfin_10806500: Jaccard 0.715, spread $20,500, 5 partial priced ids
  - redfin_11000447: Jaccard 0.826, spread $5,660, 4 partial priced ids

## Judge round — baseline_vs_checklist
- baseline: 630 claims, unsupported 6.03%, uncertain 13.02%, recall vs gold 52.6%, critical(2+ reps) 1
- checklist: 2514 claims, unsupported 3.98%, uncertain 18.77%, recall vs gold 88.7%, critical(2+ reps) 3
- **verdict: reject** (dJaccard 0.105, spread -3.2%, midpoint shift 3.5%)
  - REJECT: new critical hallucination in >=2/3 repeats: ['redfin_10806500/photo_003.jpg', 'redfin_10806500/photo_006.jpg', 'redfin_11000447/photo_001.jpg']
- human review queue: 692 judge-flagged claims + 245 supported-audit samples (see report.json)
