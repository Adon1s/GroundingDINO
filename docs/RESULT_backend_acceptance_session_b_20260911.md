# Frozen replay and observation-loss investigation

Steven approved the Session A recommendations and explicitly renewed permission
to use Terra, Sol, and local tokens. Pipeline investigation is the primary goal;
launch readiness remains a separate conclusion. No new task/session was created.

## Verified implementation

- All 18 stored properties reproduce all four output lanes under frozen 1a–2d.
  Canonical JSON bytes, v5 observed conditions, evidence facts, and every Terra
  request fingerprint agree. No model calls were needed for this proof.
- The full 2c lane is recoverable, including 1,059 null resolutions. Contrary to
  the draft plan, `resolved_items` includes LLM null selections. The independent
  2e input count and retained/removed/suppressed rows establish completeness.
- Original catalog 3.1 and its historical publication validator run privately
  for Tier 0. Current production validators are never relaxed.
- The production orchestrator accepts explicit frozen 2b/2c/2d inputs. Debug
  artifacts now preserve 2c observations, exclusions, and per-observation 2d
  records. An empty frozen lane does not fall through to model calls.
- Both approved reliability fixes are implemented: malformed lone surrogates
  are normalized at atomic artifact writing; `new` mode requires explicit Sol.
- The SQLite reservation transaction enforces one 2M Terra batch across both
  replicas and UTC days. Unknown usage retains its reservation. Both replicas
  use one usage root; Sol remains capped at 250k per UTC day.

Commands are `prove`, `verify`, `plan`, `prepare --replica N`, `run --replica N`,
and `score` in `scripts/replay_frozen_upstream_acceptance.py`. `prepare` calls
only local Qwen; `run` consumes verified local checkpoints through the production
writer and v5 runtime. Source/image/code drift fails before live execution.
Outputs live only in `artifacts_canary/backend_acceptance_20260910`.

The initial forecast is 982,176 Terra and 173,973 Sol per replica: each property's
larger stored replica plus 5%. Newly prepared unit counts are checked before
paid calls, and the transactional budget guard remains authoritative. Forecasts
are estimates, not a promise that both replicas fit without a budget stop.

## Validation

The full suite produced **2,878 passed, 9 failed, 6 skipped**. The failures are
the same nine documented catalog-drift failures, recorded in
`reports/backend_acceptance_full_pytest_20260910.txt`. Targeted replay/usage/pass
tests passed 86/86; four additional scorer tests pass. Tier 0 was rerun after
the checkpoint refactor and remains 18/18.

A local-only dry run of `redfin_10806500` completed all 7 photos and 36 frozen
observations, then wrote product lanes and recomputed v5 requests. Initial
attempts encountered the old remote default URL and an unloaded/overcommitted
local model. The successful setup explicitly uses localhost, the approved Qwen
Q6_K model, temperature 0.1, 8,192 context, one inference slot, and 90% GPU
offload. Failed attempts made no Terra/Sol calls. The superseded preflight
manifest is preserved; no paid batch boundary was reset.

## Investigation evidence

`reports/pipeline_loss_ledger_20260910.json` preserves all 125 labels, 23 reviewed
misses, 12 hallucinations, 36 worklist rows, six closeout cases, and 40 gold cases.
Populations overlap intentionally. Every population row remains in the scorer,
including out-of-cohort production cases and gold cases with no condition anchor.

The detailed trace joins 2b text, the entire 2c lane, 2d candidates/selection,
2e suppression, product filtering, condition projection, Terra, routing, work,
and package coverage. Original semantic attributions remain separate from the
mechanical point where an observation stopped. Exact ids and explicit equivalent
representations are used; no nearest-text match silently declares recovery.

Initial narrowing: all 350 canary observations removed by product filtering are
electrical items covered by the existing quarantine. Display suppression is not
an estimator loss when the canonical lane still carries the condition. Package
absorption is representation elsewhere, not deletion. The 1,059 null selections
still require separating scope gaps, generic text, and actual resolver misses.

The human-reviewed condition miss cohort retains its prior 20 Terra / 3 Pass 2d
attributions. The gold cohort instead contains prior attributions of 4 at 2b,
5 at 2c, 8 at 2d, 2 at projection, and 1 at Terra; 10 at 2a, 5 unclear, and 5
excluded. Gold covers only two properties. These populations cannot be combined
into a pipeline-wide loss rate.

Fresh replicas, per-case dollar effects, final tuning priorities, and production
smoke remain in progress. This checkpoint is not a launch approval.

## Resumption correction during replica 1

Two properties completed with 49,801 Terra / 5,950 Sol tokens, exactly matching
the shared ledgers. Inspection of per-property resumes found that recomputing
`daily ceiling = remaining batch` would count earlier same-day calls twice.
The effective ceiling is now `min(2.5M, spent_today + remaining_batch)`, while
the separate transaction still enforces the unchanged 2M batch cap. A regression
test spans two UTC days and resumes partway through day two; 36 targeted tests
pass. This corrects the draft formula for the per-property execution strategy.

The manifest's original timestamp, shared ledger root, model settings, and
budget are unchanged. Driver-only source changes are recorded in
`reports/backend_acceptance_implementation_amendments_20260911.json`, with
before/after hashes chained to the original manifest. The full manifest must
also equal the committed authorization snapshot; a new namespace or altered
batch start cannot silently reset the budget. Amendments cannot change model,
prompt, catalog, or pipeline files.

## Completed first paid replica and investigation checkpoint

All 18 replica-1 properties are complete, with clean production verifiers and
catalog invariants. Exact artifact/ledger usage is 912,131 Terra tokens across
148 calls and 136,606 Sol tokens across 18 calls; every reservation is settled.
The original 2M Terra batch retains 1,087,869 tokens. No batch boundary changed.

The detailed findings are in `docs/RESULT_pipeline_loss_investigation_20260911.md`.
The archived first-replica packet and exact price-component reconciliations are
`reports/backend_acceptance_replica1_packet_20260911.json` and its Markdown peer.
The scorer records the 33-row bathroom subset, 55 trim conditions, 72 dedup
collisions, and the CCF-13 case separately. CCF-13 has zero wallpaper charges,
not the expected single surviving charge; this remains a semantic review item.

The latest focused suite passes **20/20**, covering replay guards, resume budgets,
policy-aware dispositions, package-support changes, and applied-price accounting.
The full-suite result above remains the broader baseline; it was not rerun for
offline report changes. The production smoke helper now supports `--local-2d`,
matching the existing frontend worker flag while preserving production routing
for the other stages. The smoke itself remains pending.

Replica 2's local preparation is running with verified per-photo checkpoints.
Paid replica 2 must start on or after 2026-09-12 00:00 UTC (September 11, 7 p.m.
Central), because replica 1 used September 11 UTC. Resume with the same driver,
output root, manifest and ledgers; do not regenerate the authorization snapshot.
The two analysis commands below are read-only with respect to the runtime:

```powershell
.venv/Scripts/python.exe scripts/analysis/backend_acceptance_local_stability.py
.venv/Scripts/python.exe scripts/replay_frozen_upstream_acceptance.py score
```

`score` intentionally returns nonzero while fewer than 36 artifacts are complete.
The first-replica packet generator must run before paid replica 2 and refuses to
overwrite its ledger snapshot once a replica-2 property is complete. Final
readiness still requires the second paid replica, production smoke and Steven's
disposition of material dollar changes; this checkpoint is not a GO decision.
