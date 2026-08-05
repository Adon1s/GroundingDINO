# kind-ontology-v2 — v2 — GPT-5.6 Terra — holdout

- run at: 2026-08-05T14:43:19
- model: `{"provider": "openai", "model": "gpt-5.6-terra", "reasoning_effort": "low", "verbosity": "low", "api_key": "<redacted>"}`
- repeats: 5
- case fingerprint: `e56998c7c51995fdf8bde678e5affd15cce799174318c3ae603bf56e4d3e8708`
- ontology: observation-kind-v2
- 2b prompt: pass_2b_atomic_v2 `243ae92a37fd`
- 2c prompt: pass_2c_kind_v2 `e074c606e071`

## Metrics

- schema-failure calls: **0**
- atomic kind accuracy (mean per-repeat): **1.000**
- defect: recall 1.000, precision 1.000
- degradation: recall 1.000, precision 1.000
- modernization: recall 1.000, precision 1.000
- excluded false-classification rate: **0.000**
- excluded reason accuracy: 0.833
- unanimity: **1.000**, pairwise agreement: **1.000**
- mixed full-case success: **0.967**, cross-kind bundling: **0.000**

## Confusion matrix (gold × predicted, decisions pooled over repeats)

| gold \ predicted | defect | degradation | excluded | modernization |
|---|---|---|---|---|
| defect | 45 | 0 | 0 | 0 |
| degradation | 0 | 45 | 0 | 0 |
| excluded | 0 | 0 | 30 | 0 |
| modernization | 0 | 0 | 0 | 45 |

## Gates

- [PASS] 100% valid/complete response partitions
- [PASS] >=95% kind accuracy (mean per-repeat, atomic cases)
- [PASS] >=90% recall for every kind
- [PASS] >=95% of atomic cases unanimous across repeats
- [PASS] >=98% pairwise agreement across repeats
- [PASS] <=5% of excluded-gold text classified with a kind
- [PASS] >=90% mixed-case full success
- [PASS] <=5% mixed-case cross-kind bundling
