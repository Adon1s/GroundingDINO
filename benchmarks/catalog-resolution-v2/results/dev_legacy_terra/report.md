# catalog-resolution-v2 — legacy / dev

- model: `{"provider": "openai", "model": "gpt-5.6-terra", "reasoning_effort": "low", "verbosity": "low", "api_key": "<redacted>"}`
- embeddings: `{"backend": "openai_compatible", "base_url": "http://127.0.0.1:8081/v1", "model_name": "jinaai/jina-embeddings-v5-text-small-text-matching-GGUF:Q8_0", "st_model_name": "jinaai/jina-embeddings-v3", "dimension": 1024, "device": "cpu", "trust_remote_code": true, "topk": 5, "top_k_candidates": 8, "note": "Explicit, never read from env. Mirrors tools/pipeline_config defaults as of Task 2. Constructing the retriever embeds the whole catalog, which is the real-POST sidecar probe: /health has been observed returning 200 while every embeddings call fails."}`
- repeats: 3
- case fingerprint: `d355313936b87c34f341c82f6ff249c30c95cf0d91160a7390675bd4f2822667`
- catalog fingerprint: `fec54467bc839780ec141ab09265ef01f193e78e30b76dea3a0989fd421731cb`
- 2d prompt: pass_2d_exact_kind_v2 `c779b2ac5b43ba70…`

## Headline

- candidate recall@5: **0.897** (by kind: {"defect": 0.935, "upgrade": 0.75})
- final resolved-ID accuracy: **0.862** (by kind: {"defect": 0.891, "upgrade": 0.75})
- candidate-kind purity: **1.000**
- mean gold rank: 1.21 (top-1 0.865)
- no-match false positives: 0.000 (n=6)
- paired cases: recall 0.828, accuracy 0.793 (n=87)
- filter probes: 6 violations of 6
- split-family confusions: 21
- failures: 0

## Misses

| case | kind | gold | resolved | recall hit |
|---|---|---|---|---|
| res-cab-003 | upgrade | outdated_or_damaged_cabinets | outdated_kitchen_finishes | NO |
| res-cab-003 | upgrade | outdated_or_damaged_cabinets | outdated_kitchen_finishes | NO |
| res-cab-003 | upgrade | outdated_or_damaged_cabinets | outdated_kitchen_finishes | NO |
| res-pnt-002 | defect | peeling_or_damaged_bathroom_paint | peeling_or_discolored_paint | yes |
| res-pnt-002 | defect | peeling_or_damaged_bathroom_paint | peeling_or_discolored_paint | yes |
| res-pnt-002 | defect | peeling_or_damaged_bathroom_paint | peeling_or_discolored_paint | yes |
| res-vct-001 | defect | dated_or_worn_vanity_countertop | vanity_countertop_damage | NO |
| res-vct-001 | defect | dated_or_worn_vanity_countertop | vanity_countertop_damage | NO |
| res-vct-001 | defect | dated_or_worn_vanity_countertop | vanity_countertop_damage | NO |
| res-app-003 | upgrade | appliance_damage_or_missing | outdated_kitchen_finishes | NO |
| res-app-003 | upgrade | appliance_damage_or_missing | outdated_kitchen_finishes | NO |
| res-app-003 | upgrade | appliance_damage_or_missing | outdated_kitchen_finishes | NO |
| res-cfn-002 | upgrade | ceiling_fan_mismatched_blades | older_ceiling_fan_style | NO |
| res-cfn-002 | upgrade | ceiling_fan_mismatched_blades | older_ceiling_fan_style | NO |
| res-cfn-002 | upgrade | ceiling_fan_mismatched_blades | older_ceiling_fan_style | NO |
| res-pav-003 | defect | concrete_driveway_surface_wear | driveway_or_walkway_cracking | NO |
| res-pav-003 | defect | concrete_driveway_surface_wear | driveway_or_walkway_cracking | NO |
| res-pav-003 | defect | concrete_driveway_surface_wear | driveway_or_walkway_cracking | NO |
| res-lnd-001 | defect | landscape_improvement_needed | None | NO |
| res-lnd-001 | defect | landscape_improvement_needed | None | NO |
| res-lnd-001 | defect | landscape_improvement_needed | None | NO |
| res-moi-002 | defect | visible_mold_or_mildew | mold_or_mildew_visible_bathroom | yes |
| res-moi-002 | defect | visible_mold_or_mildew | mold_or_mildew_visible_bathroom | yes |
| res-moi-002 | defect | visible_mold_or_mildew | mold_or_mildew_visible_bathroom | yes |
