# kind-ontology-v2 — v2 — GPT-5.6 Terra — dev

- run at: 2026-08-05T14:39:40
- model: `{"provider": "openai", "model": "gpt-5.6-terra", "reasoning_effort": "low", "verbosity": "low", "api_key": "<redacted>"}`
- repeats: 5
- case fingerprint: `ff95e151eb684039ff42e2569e0829e8e8e68970c86cc1341b44ab3e74560f7b`
- ontology: observation-kind-v2
- 2b prompt: pass_2b_atomic_v2 `243ae92a37fd`
- 2c prompt: pass_2c_kind_v2 `e074c606e071`

## Metrics

- schema-failure calls: **0**
- atomic kind accuracy (mean per-repeat): **0.997**
- defect: recall 1.000, precision 0.990
- degradation: recall 0.991, precision 1.000
- modernization: recall 1.000, precision 1.000
- excluded false-classification rate: **0.000**
- excluded reason accuracy: 0.917
- unanimity: **0.987**, pairwise agreement: **0.995**
- mixed full-case success: **1.000**, cross-kind bundling: **0.000**

## Confusion matrix (gold × predicted, decisions pooled over repeats)

| gold \ predicted | defect | degradation | excluded | modernization |
|---|---|---|---|---|
| defect | 100 | 0 | 0 | 0 |
| degradation | 1 | 109 | 0 | 0 |
| excluded | 0 | 0 | 60 | 0 |
| modernization | 0 | 0 | 0 | 105 |
