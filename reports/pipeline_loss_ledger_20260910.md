# Pipeline loss investigation — historical baseline

The table counts observation endpoints, not useful observations lost. Several observations can represent one condition; package absorption preserves representation.

| Endpoint | canary | production |
| --- | --- | --- |
| 2d_null_selection | 1059 | 159 |
| product_filter | 350 | 92 |
| represented_absorbed_by_package | 996 | 209 |
| represented_standalone | 737 | 152 |
| routing_excluded_generic | 136 | 51 |
| routing_inspection | 4 | 0 |
| routing_no_action | 174 | 35 |
| routing_withheld | 7 | 2 |
| terra_cannot_assess | 30 | 14 |
| terra_unsupported | 175 | 99 |

## Reviewed causes

| First responsible stage from prior review | rc_ miss cases |
| --- | --- |
| 2d | 3 |
| terra | 20 |

The final mechanical endpoint and the first semantic error are separate fields in the ledger. A mismatched catalog claim can fail Terra correctly while the useful observation was lost at 2d.

## Coverage and limits

{'labels_v1_1': 125, 'rc_misses': 23, 'rc_hallucinations': 12, 'six_closeout': 6, 'gold': 40, 'worklist': 36}

- Unlabelled mechanical filtering is not a useful-loss rate.
- 2a prose has no observation ids; first-loss causes use existing reviewed evidence, not lexical guesses.
- Old artifacts do not persist every 2c exclusion; unmatched 2b bullets are not proven deletions.
- Condition-based reviewed cards cannot measure observations that never reached conditions.
- Gold review covers two properties; cohort counts are not population recall.

Detailed observation traces: `artifacts_canary/backend_acceptance_20260910/investigation/observation_traces.json`.

Fresh replica dispositions remain unresolved until the replay is scored. No launch conclusion follows from this baseline census.
