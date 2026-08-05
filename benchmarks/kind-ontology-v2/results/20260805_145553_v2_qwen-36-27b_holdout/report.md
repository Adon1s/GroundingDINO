# kind-ontology-v2 — v2 — Qwen 3.6 27B — holdout

- run at: 2026-08-05T14:55:53
- model: `{"provider": "lmstudio", "model": "unsloth/qwen3.6-27b@q6_k", "url": "http://100.102.92.1:1234"}`
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
- mixed full-case success: **0.833**, cross-kind bundling: **0.000**

## Confusion matrix (gold × predicted, decisions pooled over repeats)

| gold \ predicted | defect | degradation | excluded | modernization |
|---|---|---|---|---|
| defect | 45 | 0 | 0 | 0 |
| degradation | 0 | 45 | 0 | 0 |
| excluded | 0 | 0 | 30 | 0 |
| modernization | 0 | 0 | 0 | 45 |
