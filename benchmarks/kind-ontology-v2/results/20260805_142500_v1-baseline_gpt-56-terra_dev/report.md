# kind-ontology-v2 — v1-baseline — GPT-5.6 Terra — dev

- run at: 2026-08-05T14:25:00
- model: `{"provider": "openai", "model": "gpt-5.6-terra", "reasoning_effort": "low", "verbosity": "low", "api_key": "<redacted>"}`
- repeats: 5
- case fingerprint: `ff95e151eb684039ff42e2569e0829e8e8e68970c86cc1341b44ab3e74560f7b`
- ontology: observation-kind-v2
- 2b prompt: pass_2b_atomic_v2 `164d3a4bc60c`
- 2c prompt: pass_2c_kind_v2 `3d414ec5fa17`

## v1 baseline (two-kind contract replay)

- degradation-gold split: {"decisions": 110, "defect_or_damage": 110, "upgrade_candidate": 0, "not_forwarded": 0, "unanimous_cases": 22, "n_degradation_cases": 22}
- mapped accuracy: {"defect_as_defect_or_damage": 1.0, "modernization_as_upgrade_candidate": 0.9142857142857143}
- excluded-gold forwarded rate: 0.250

### Label distribution by gold kind

- defect: {"defect_or_damage": 100}
- degradation: {"defect_or_damage": 110}
- modernization: {"upgrade_candidate": 96, "generic_presence": 9}
