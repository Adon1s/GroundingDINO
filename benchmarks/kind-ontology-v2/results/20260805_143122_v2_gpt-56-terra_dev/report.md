# kind-ontology-v2 — v2 — GPT-5.6 Terra — dev

- run at: 2026-08-05T14:31:22
- model: `{"provider": "openai", "model": "gpt-5.6-terra", "reasoning_effort": "low", "verbosity": "low", "api_key": "<redacted>"}`
- repeats: 5
- case fingerprint: `ff95e151eb684039ff42e2569e0829e8e8e68970c86cc1341b44ab3e74560f7b`
- ontology: observation-kind-v2
- 2b prompt: pass_2b_atomic_v2 `164d3a4bc60c`
- 2c prompt: pass_2c_kind_v2 `3d414ec5fa17`

## Metrics

- schema-failure calls: **0**
- atomic kind accuracy (mean per-repeat): **0.927**
- defect: recall 0.880, precision 0.936
- degradation: recall 0.945, precision 0.897
- modernization: recall 0.952, precision 1.000
- excluded false-classification rate: **0.000**
- excluded reason accuracy: 0.917
- unanimity: **0.933**, pairwise agreement: **0.971**
- mixed full-case success: **0.889**, cross-kind bundling: **0.022**

## Confusion matrix (gold × predicted, decisions pooled over repeats)

| gold \ predicted | defect | degradation | excluded | modernization |
|---|---|---|---|---|
| defect | 88 | 12 | 0 | 0 |
| degradation | 6 | 104 | 0 | 0 |
| excluded | 0 | 0 | 60 | 0 |
| modernization | 0 | 0 | 5 | 100 |
