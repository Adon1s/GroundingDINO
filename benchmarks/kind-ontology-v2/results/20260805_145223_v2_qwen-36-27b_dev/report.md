# kind-ontology-v2 — v2 — Qwen 3.6 27B — dev

- run at: 2026-08-05T14:52:23
- model: `{"provider": "lmstudio", "model": "unsloth/qwen3.6-27b@q6_k", "url": "http://100.102.92.1:1234"}`
- repeats: 5
- case fingerprint: `ff95e151eb684039ff42e2569e0829e8e8e68970c86cc1341b44ab3e74560f7b`
- ontology: observation-kind-v2
- 2b prompt: pass_2b_atomic_v2 `243ae92a37fd`
- 2c prompt: pass_2c_kind_v2 `e074c606e071`

## Metrics

- schema-failure calls: **22**
- atomic kind accuracy (mean per-repeat): **0.937**
- defect: recall 1.000, precision 0.952
- degradation: recall 0.955, precision 1.000
- modernization: recall 0.857, precision 1.000
- excluded false-classification rate: **0.000**
- excluded reason accuracy: 0.767
- unanimity: **0.880**, pairwise agreement: **0.944**
- mixed full-case success: **1.000**, cross-kind bundling: **0.000**

## Confusion matrix (gold × predicted, decisions pooled over repeats)

| gold \ predicted | defect | degradation | excluded | modernization | schema_failure |
|---|---|---|---|---|---|
| defect | 100 | 0 | 0 | 0 | 0 |
| degradation | 5 | 105 | 0 | 0 | 0 |
| excluded | 0 | 0 | 51 | 0 | 9 |
| modernization | 0 | 0 | 2 | 90 | 13 |
