# catalog-resolution-v2 — v2 / dev

- model: `{"provider": "openai", "model": "gpt-5.6-terra", "reasoning_effort": "low", "verbosity": "low", "api_key": "<redacted>"}`
- embeddings: `{"backend": "openai_compatible", "base_url": "http://127.0.0.1:8081/v1", "model_name": "jinaai/jina-embeddings-v5-text-small-text-matching-GGUF:Q8_0", "st_model_name": "jinaai/jina-embeddings-v3", "dimension": 1024, "device": "cpu", "trust_remote_code": true, "topk": 5, "top_k_candidates": 8, "note": "Explicit, never read from env. Mirrors tools/pipeline_config defaults as of Task 2. Constructing the retriever embeds the whole catalog, which is the real-POST sidecar probe: /health has been observed returning 200 while every embeddings call fails."}`
- repeats: 3
- case fingerprint: `d355313936b87c34f341c82f6ff249c30c95cf0d91160a7390675bd4f2822667`
- catalog fingerprint: `1198b33f9c8f4e07634c6dd05c0d6f6dc64ecc11c8346fc9910d75d81e6c30f4`
- 2d prompt: pass_2d_exact_kind_v2 `c779b2ac5b43ba70…`

## Headline

- candidate recall@5: **1.000** (by kind: {"defect": 1.0, "degradation": 1.0, "modernization": 1.0})
- final resolved-ID accuracy: **0.983** (by kind: {"defect": 0.962, "degradation": 1.0, "modernization": 1.0})
- candidate-kind purity: **1.000**
- mean gold rank: 1.07 (top-1 0.966)
- no-match false positives: 0.000 (n=6)
- paired cases: recall 1.000, accuracy 1.000 (n=87)
- filter probes: 0 violations of 6
- split-family confusions: 3
- failures: 0

## Misses

| case | kind | gold | resolved | recall hit |
|---|---|---|---|---|
| res-moi-002 | defect | visible_mold_or_mildew | mold_or_mildew_visible_bathroom | yes |
| res-moi-002 | defect | visible_mold_or_mildew | mold_or_mildew_visible_bathroom | yes |
| res-moi-002 | defect | visible_mold_or_mildew | mold_or_mildew_visible_bathroom | yes |
