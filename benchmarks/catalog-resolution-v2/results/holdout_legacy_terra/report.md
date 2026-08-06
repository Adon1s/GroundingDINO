# catalog-resolution-v2 — legacy / holdout

- model: `{"provider": "openai", "model": "gpt-5.6-terra", "reasoning_effort": "low", "verbosity": "low", "api_key": "<redacted>"}`
- embeddings: `{"backend": "openai_compatible", "base_url": "http://127.0.0.1:8081/v1", "model_name": "jinaai/jina-embeddings-v5-text-small-text-matching-GGUF:Q8_0", "st_model_name": "jinaai/jina-embeddings-v3", "dimension": 1024, "device": "cpu", "trust_remote_code": true, "topk": 5, "top_k_candidates": 8, "note": "Explicit, never read from env. Mirrors tools/pipeline_config defaults as of Task 2. Constructing the retriever embeds the whole catalog, which is the real-POST sidecar probe: /health has been observed returning 200 while every embeddings call fails."}`
- repeats: 5
- case fingerprint: `1a8b99145881bcaf99aca48d818868848c9dd27907859c650ed43ee24dc2910e`
- catalog fingerprint: `fec54467bc839780ec141ab09265ef01f193e78e30b76dea3a0989fd421731cb`
- 2d prompt: pass_2d_exact_kind_v2 `c779b2ac5b43ba70…`

## Headline

- candidate recall@5: **0.870** (by kind: {"defect": 0.932, "upgrade": 0.6})
- final resolved-ID accuracy: **0.870** (by kind: {"defect": 0.932, "upgrade": 0.6})
- candidate-kind purity: **1.000**
- mean gold rank: 1.09 (top-1 0.915)
- no-match false positives: 0.000 (n=10)
- paired cases: recall 0.833, accuracy 0.833 (n=90)
- filter probes: 15 violations of 15
- split-family confusions: 20
- failures: 0

## Misses

| case | kind | gold | resolved | recall hit |
|---|---|---|---|---|
| res-cab-005 | upgrade | outdated_or_damaged_cabinets | outdated_kitchen_finishes | NO |
| res-cab-005 | upgrade | outdated_or_damaged_cabinets | outdated_kitchen_finishes | NO |
| res-cab-005 | upgrade | outdated_or_damaged_cabinets | outdated_kitchen_finishes | NO |
| res-cab-005 | upgrade | outdated_or_damaged_cabinets | outdated_kitchen_finishes | NO |
| res-cab-005 | upgrade | outdated_or_damaged_cabinets | outdated_kitchen_finishes | NO |
| res-van-003 | upgrade | outdated_or_damaged_vanity | outdated_bathroom_finishes | NO |
| res-van-003 | upgrade | outdated_or_damaged_vanity | outdated_bathroom_finishes | NO |
| res-van-003 | upgrade | outdated_or_damaged_vanity | outdated_bathroom_finishes | NO |
| res-van-003 | upgrade | outdated_or_damaged_vanity | outdated_bathroom_finishes | NO |
| res-van-003 | upgrade | outdated_or_damaged_vanity | outdated_bathroom_finishes | NO |
| res-fan-003 | upgrade | exhaust_fan_missing_or_damaged | None | NO |
| res-fan-003 | upgrade | exhaust_fan_missing_or_damaged | None | NO |
| res-fan-003 | upgrade | exhaust_fan_missing_or_damaged | None | NO |
| res-fan-003 | upgrade | exhaust_fan_missing_or_damaged | None | NO |
| res-fan-003 | upgrade | exhaust_fan_missing_or_damaged | None | NO |
| res-lnd-003 | defect | yard_debris_overgrown_leaves | None | NO |
| res-lnd-003 | defect | yard_debris_overgrown_leaves | None | NO |
| res-lnd-003 | defect | yard_debris_overgrown_leaves | None | NO |
| res-lnd-003 | defect | yard_debris_overgrown_leaves | None | NO |
| res-lnd-003 | defect | yard_debris_overgrown_leaves | None | NO |
| res-lnd-004 | upgrade | tree_stump_present_in_yard | landscape_improvement_needed | NO |
| res-lnd-004 | upgrade | tree_stump_present_in_yard | landscape_improvement_needed | NO |
| res-lnd-004 | upgrade | tree_stump_present_in_yard | landscape_improvement_needed | NO |
| res-lnd-004 | upgrade | tree_stump_present_in_yard | yard_debris_overgrown_leaves | NO |
| res-lnd-004 | upgrade | tree_stump_present_in_yard | yard_debris_overgrown_leaves | NO |
| res-hdw-001 | defect | dated_door_hardware | None | NO |
| res-hdw-001 | defect | dated_door_hardware | None | NO |
| res-hdw-001 | defect | dated_door_hardware | None | NO |
| res-hdw-001 | defect | dated_door_hardware | None | NO |
| res-hdw-001 | defect | dated_door_hardware | None | NO |
| res-deg-008 | defect | gutter_maintenance_needed | clogged_or_damaged_gutters | NO |
| res-deg-008 | defect | gutter_maintenance_needed | clogged_or_damaged_gutters | NO |
| res-deg-008 | defect | gutter_maintenance_needed | clogged_or_damaged_gutters | NO |
| res-deg-008 | defect | gutter_maintenance_needed | clogged_or_damaged_gutters | NO |
| res-deg-008 | defect | gutter_maintenance_needed | clogged_or_damaged_gutters | NO |
