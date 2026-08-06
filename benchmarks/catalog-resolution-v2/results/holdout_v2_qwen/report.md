# catalog-resolution-v2 — v2 / holdout

- model: `{"provider": "lmstudio", "model": "unsloth/qwen3.6-27b@q6_k", "url": "http://100.102.92.1:1234"}`
- embeddings: `{"backend": "openai_compatible", "base_url": "http://127.0.0.1:8081/v1", "model_name": "jinaai/jina-embeddings-v5-text-small-text-matching-GGUF:Q8_0", "st_model_name": "jinaai/jina-embeddings-v3", "dimension": 1024, "device": "cpu", "trust_remote_code": true, "topk": 5, "top_k_candidates": 8, "note": "Explicit, never read from env. Mirrors tools/pipeline_config defaults as of Task 2. Constructing the retriever embeds the whole catalog, which is the real-POST sidecar probe: /health has been observed returning 200 while every embeddings call fails."}`
- repeats: 5
- case fingerprint: `1a8b99145881bcaf99aca48d818868848c9dd27907859c650ed43ee24dc2910e`
- catalog fingerprint: `1198b33f9c8f4e07634c6dd05c0d6f6dc64ecc11c8346fc9910d75d81e6c30f4`
- 2d prompt: pass_2d_exact_kind_v2 `c779b2ac5b43ba70…`

## Headline

- candidate recall@5: **1.000** (by kind: {"defect": 1.0, "degradation": 1.0, "modernization": 1.0})
- final resolved-ID accuracy: **1.000** (by kind: {"defect": 1.0, "degradation": 1.0, "modernization": 1.0})
- candidate-kind purity: **1.000**
- mean gold rank: 1.04 (top-1 0.963)
- no-match false positives: 0.000 (n=10)
- paired cases: recall 1.000, accuracy 1.000 (n=90)
- filter probes: 0 violations of 15
- split-family confusions: 0
- failures: 0
