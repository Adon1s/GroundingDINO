# Plan: bounded backend acceptance after the catalog checkpoint and Pass 2d closeout

Prepared 2026-09-10 against `master` @ `a656152`. Revised after Steven's review of the first draft. Read-only so far: no model calls, no edits.

## 1. Context

Steven wants the backend reliable enough to start up and promote RenoIntel. Frontend contract adoption and pricing stay deferred. The catalog checkpoint (Session 7) and the Pass 2d investigation are closed; neither shipped a pipeline change. Another session recommended: (1) diagnose six persistent Terra-rejected misses, then (2) run whole-property acceptance ending in a go/no-go, with "retain the current implementation" a valid outcome. Steven asked whether I agree and for my own implementation plan under a 4M-token GPT-5.6 Terra limit split into batches of at most 2M each. Local Qwen is unlimited.

Decisions Steven made in this session:

| Decision | Answer |
|---|---|
| Pass 2d temperature for acceptance | Pin **0.1** explicitly in the production call path before the run |
| Hallucination leg | **Measure** the corroboration gate offline first; Steven decides adoption before Batch 1 |
| Sol | 250k/day ceiling; multiple days fine; **break the work into sessions** |
| Fresh-observation arm | Replace with a production-path smoke on **1 property** (2 acceptable) |

Corrections Steven made to the first draft, all adopted below: the replay must consume every frozen Pass 2c observation including prior 2d nulls and reproduce production's 2e normalization and product filtering; the six-case closeout must not pre-decide its conclusion; the v1 "~10% hallucination" premise is superseded by the v1.1 label repair; every original reviewed case needs an explicit disposition in the acceptance; noise bands contextualize, they do not excuse; approved fixes land before the smoke; the temperature pin must be 2d-specific; the batch needs one shared cap.

## 2. Verdict on the other session's path

**Agree on the shape and endpoint.** Stop general 2d tuning; whole-property acceptance is the owed obligation (Session 7 S7-17, S7-24, manifest_v2 LQ-1..3, D7, the `cutover_plan` in `reports/catalog_checkpoint_affected_artifacts.json`); go/no-go with "retain" allowed; all five corrections verified true; no reopening of D1–D8, CAP-022, the electrical quarantine, or the FE/pricing deferrals.

**Amend in four places:**

**A1. The six-case closeout stays, bounded and offline, without a predetermined conclusion.** What the record already establishes: the photos Terra saw were full-resolution originals (1200–1280 px) and byte-identical between production and the factorized replay (fingerprint match 6/6); the Terra payload carries the 2a observation text next to the catalog claim (`tools/renovation_architecture/terra_review.py:138-148`); claim wording was examined by the catalog program (CAP-004 `no_change`; CAP-008 two wordings live-tested, both failed; claim-text edits `not_authorized_by_this_record` in `reports/catalog_audit_approvals_v2.json`); the human labels are a canned menu option with null notes and `skeptic_checked: false`; the factorized design is not to be repeated (`docs/RESULT_factorized_replay_20260831.md` §7); and "Terra hedges on visibility" is on the v1.1 discard list (72% of all unsupported rationales contain "clear", recovered cards are not more hedged; `docs/RESULT_label_v1_1_20260829.md` "Discard list"). What the record does **not** establish: why Terra rejects these six, or whether any alternative recovery mechanism would be unsafe. Session A quantifies the candidate mechanisms that remain and concludes either "no justified experiment" or "one supported hypothesis with a control design and cost". Steven can accept unresolved misses without accepting any explanation of them.

**A2. Add the missing false-claim leg, on the repaired evidence.** The v1 figures (canary 10.5%, production 10.4%) mixed absent, misnamed and trivial into one bucket. Under v1.1 (`docs/RESULT_label_v1_1_20260829.md`): canary hard-false 2.66% [0.69, 12.60], like-for-like broad 8.12% [3.41, 19.95], production hard-false 8.94% [3.62, 30.99]; canary billed-error mass splits absent 24.1% / misnamed 49.4% / trivial 26.5%; canary `hard_false_billed` n=3, one uniform card moves the rate 2.4 pp; 2.66% is not publishable as a population rate; 0 of 79 blind supported cards relabelled absent. What survives: every adjudicated hallucination was Terra-supported and reached a work item, and the 2a freeze was scoped "during catalog evaluation" (`docs/catalog_audit_program/00_OVERALL_CONTEXT.md:32,45`), now closed. Per Steven's answer, Session A measures the `min_photo_evidence=2` corroboration gate on the reviewed cohort as outcomes, not population precision/recall, reporting false claims withheld against legitimate work withheld, with misnamed and trivial reported separately. Two distinct views establish evidence coverage, not corroboration of the condition; the tradeoff must be in hand before any adoption.

**A3. The fresh-observation arm becomes a one-property production-path smoke.** Fresh 2a variance is already measured (resolved-ID Jaccard 0.53 on identical inputs; 8 of 523 photos reproduce their observation set); one 18-property fresh replica ≈ 3.9M Terra. The smoke establishes operational execution under the final configuration only, not end-to-end quality stability; the decision record says so.

**A4. Production-reliability items from the C5 window join the go/no-go.** Lone-surrogate `UnicodeEncodeError` killed 2 of 8 attempts on 2026-08-25 and has no fix in `tools/`; Sol silently resolves to the Terra model when `RENOVATION_SOL_MODEL` is unset (`tools/pipeline_config.py:252-253`); Sol's 250k/day binds at ~5 listings/day, mostly v4's Pass 2f feeding a FE that renders v4. Two are small fixes for Steven to approve; the third is a launch-throughput decision.

One production-config ambiguity to settle in Session A: the C4 smoke recorded Pass 2d on `gpt-5.6-terra` under `--premium`, but the 2026-08-25 production artifact records 2d on local Qwen (`standard_default`). Every 2d result in the record is on Qwen. The acceptance must pin whatever production will actually run.

## 3. Configuration under test

- Catalog `tools/issue_catalog_kind_v2.json` v3.2, sha `787964013d…` (unchanged).
- Pass 2d: `pass_2d_exact_kind_v2` prompt (unchanged), local Qwen `unsloth/qwen3.6-27b@q6_k`, **temperature 0.1 pinned in code, 2d-specific** (Session A). `LM_STUDIO_URL` `http://127.0.0.1:1234`, not the stale `.env` address (S6-6). Embeddings sidecar probed with a real POST.
- Terra: `gpt-5.6-terra`, `terra_condition_review_v1`, effort medium, max_output 8192 (unchanged).
- Sol: `gpt-5.6-sol`, `sol_package_review_v1`, effort medium (unchanged).
- Corroboration gate: **off unless Steven adopts it after Session A.** If adopted, it is applied before Batch 1 (generator change + regeneration + Tier 1 proof), so Sol decisions are made under the final configuration. The gate runs after Terra (`tools/renovation_architecture/disposition.py:38-42`), so Terra verdicts are unaffected either way; package candidates and Sol decisions are not.
- Reliability fixes (if approved): applied in Session B, before any run.
- Budget guard on with one usage root shared by both replicas and a **batch cap** (see §8).

## 4. Sessions

### Session A — offline closeout and decision packet (Terra 0, Sol 0, local Qwen small)

1. **Six-case closeout** → `docs/RESULT_terra_miss_closeout_<date>.md`. Record the established facts (A1) per case from `reports/error_attribution_queue.json` (lines 5750/7025/7724/8337/10529/11716), `reports/retag_verdicts.jsonl`, `reports/adjudication_verdicts.jsonl`, and the factorized units under `artifacts_canary/factorized_v1_20260831/<property>/units/`. Keep neutral-paint, material-dispute, trim, subject-mismatch (`rc_ace7463d6837`: shower valve/trim observation resolved to a tub-surround item), and weak-label rows separately identified. No additional manual review upfront.
2. **Candidate-mechanism quantification** (one new script under `scripts/analysis/`, zero model calls, over all 1,047 run_1 `condition_reviews[]` joined to `observed_conditions[]`, `evidence_facts[]`, `terra_calls[]` and catalog observables). Candidates from the record, each scored against the 47 dirB human-labelled cards (`reports/review_verdicts.jsonl` + `reports/review_queue.json`) and the correctly-rejected controls: (a) adjacent-mechanism refutation, where the rationale negates a claim term absent from the observation text (`docs/RESULT_error_attribution_20260831.md` §2); (b) unit batch size, conditions and photos per Terra call (the six units carried 4–19 conditions and 1–13 photos; an open unowned thread in the Session 9 decision §6; `redecide_renovation_architecture.py` already supports `drop_condition_keys` for a controlled split); (c) any pattern the closeout surfaces. Excluded by prior result: visibility hedging (discarded), same-pixels re-ask with a factorized rubric (failed). Conclusion is one of two: **no justified experiment**, or **one hypothesis** with its control population, replica design against the directional floors (supported→not 4.05%, unsupported→supported 23.1%), and token cost, for Steven to authorize or decline separately. A hypothesis carried by the six alone does not qualify.
3. **Corroboration-gate tradeoff on the reviewed cohort** (second script): for every `labels_v1_1.json` card and every rc_ hallucination case, join `property_key` + `condition_id` to `evidence_facts[].distinct_view_count` in the source artifact; tabulate per catalog item, under `min_photo_evidence=2`: absent withheld, misnamed withheld, trivial withheld, supported-and-warranted withheld, and the cards not withheld in each class. Also list the unlabelled single-view billed conditions per item in run_1 so the unreviewed exposure is visible. Family list parametrized; default = the §9.4 families plus the `reports/review_analysis.md` §3 by-item table, minus `dated_interior_trim` (already `no_action`) and the CAP-007 parent (split). The v1.1 note that an item-level gate on three items removes 6 supported alongside 6 trivial (1:1 collateral) is the shape of tradeoff to reproduce. Adoption note: `min_photo_evidence` is not authorable in `tools/catalog_migrations/kind_v2_decisions.json` (`scripts/migrate_catalog_kind_v2.py:82-92`); adoption needs that generator change, regeneration, and a Tier 1 proof.
4. **Temperature pin, 2d-specific.** In `run_pass_2d` (`tools/scene_classifier_passes.py:1179-1183`) apply `setdefault("temperature", cfg.PASS_2D_TEMPERATURE)` to the kwargs built by `_with_analysis_pass`, with the setting resolved in `tools/pipeline_config.py` (default 0.1). This is the only 2d call site: both the orchestrator (`tools/scene_classifier_orchestrator.py:989` → `:255`) and the audit harness reach it through `resolve_observation_against_catalog`. It cannot touch 1a or `property_summarizer.py` because it is not in the shared `qwen_config` (`tools/vlm_client.py:1418-1422`), and the OpenAI branch of `analyze_text` does not forward temperature (`tools/vlm_client.py:1338-1348`), so it is inert if 2d is ever routed to Terra. Test in `tests/test_scene_classifier_passes.py`; update the sampling note at `scripts/catalog_audit_live_validation.py:1372-1375`. Regression: score the catalog-resolution benchmark slices (`scripts/benchmark_catalog_resolution.py`) at 0.1 against stored results; this also closes S7-22.
5. **Production 2d model check.** Read `model_routing` on the newest artifacts under `C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts/` and `lib/analysis/workerMode.ts`; record which model production runs for 2d.
6. **Decision packet**: (a) six cases: accept as unresolved misses, or authorize the one hypothesis if Session A produced one; (b) gate adoption per item from the tradeoff table; (c) confirm the 2d model; (d) approve the two reliability fixes.

Finished means: RESULT doc committed, both tables delivered, temperature pin merged with green tests and benchmark scores, Steven's four rulings recorded.

### Session B — build the frozen-upstream replay, prove it for free, land approved fixes (Terra 0, Sol 0)

**Design principle: reuse production's own 2d, 2e, lane construction, product filtering and v5 seam; freeze only the vision and text stages above 2d.** Nothing is reimplemented.

1. **Frozen-upstream orchestrator mode.** Two new `options.meta` hooks following the existing `pass_1a_frozen_scene` (`scene_classifier_orchestrator.py:692-704`) and `pass_2a_frozen_freeform` (`:769-780`) pattern: `pass_2b_frozen_struct` (sets `result.observations_struct`, source `photos[k].debug.observations_struct`) and `pass_2c_frozen_observations` (sets `result.observations` with `issue_id` and `source_photo_key` preserved so the stamping at `:872` is a no-op). Each records `models_used = "frozen_replay"`, zero calls. From there `_run_passes` runs 2d live (`:986-1002`), 2e live (`:1032-1130`, rule-based), and `write_photo_intel` builds `issues_flat`, `estimate_issues_flat`, the product lanes, and the v5 seam exactly as production does.
2. **Reconstructing the frozen 2c lane.** `debug.resolved_items[]` is downstream of 2d and drops prior nulls (`:1001`), so it is **not** the input. The full 2c set is 2e's input: every observation, resolved or not (`:1044-1059`). Reconstruct it per photo from the persisted 2e outputs, `issues.matched` ∪ `issues.final` ∪ `issues.removed` ∪ `_pass_2e_telemetry.suppressed_samples`, deduped by `issue_id`, stripped of `catalogItemId`/`catalogItemKind` so 2d decides afresh. Whether this union is complete is **proven, not assumed**, in step 4.
3. **Driver** `scripts/replay_frozen_upstream_acceptance.py`, modelled on `tools/benchmark_pass2a.py:449-556` (builds a `PropertyAnalysisJob` with a caller-chosen `job_id`/`artifacts_dir` and calls `write_photo_intel`), with `RENOVATION_ARCHITECTURE_MODE=new`, `KIND_ONTOLOGY_VERSION=observation_kind_v2`, and the `_guard_out_root` pattern (`scripts/catalog_audit_live_validation.py:632-643`) refusing the session9 and production roots. Photo paths via `tools/pass_2f_artifact_inputs.py:40-61` with the session9 `input_freeze.json` SHA-256 check. Subcommands: `verify` (invariants, catalog sha, model pins, LM Studio + embeddings real-POST, batch headroom) → `plan` (dry run: per-property units, request bytes, Terra reservation forecast, Sol forecast) → `run --replica N` (resumable per property; skips a property with a real `photo_intel.json`; stops cleanly at the batch cap) → `score`. Records per property: the artifact, the v5 envelope, and a `resolution_record.json` with every 2d row (candidates, scores, path, shortcut reason, null or item).
4. **Tier-0 proof, zero cost.** A third hook `pass_2d_frozen_resolutions` re-applies the stored resolutions; with all four upstream hooks, catalog 3.1 (`git show 07ee112:tools/issue_catalog_kind_v2.json`), and mode `current`, the driver must reproduce the stored `issues_flat`, `estimate_issues_flat`, `product_issues_flat`, `product_estimate_issues_flat` byte-for-byte on all 18 run_1 artifacts, and the v5 `observed_conditions`, `evidence_facts`, and Terra request fingerprints when recomputed from the written lanes (the pattern at `catalog_audit_live_validation.py:809-841`). Any diff names exactly which 2c observations the artifact does not preserve. If the lane is not fully recoverable, stop and report the gap and the cost of the fallback (re-running 2c from the persisted 2b struct, which is then fresh not frozen); Steven decides.
5. **Batch cap.** Add `spent_since(created_at)` to `TerraUsageLedger` (table has `created_at` and `utc_day`, `tools/renovation_architecture/usage_guard.py:43-56`). The driver reads it against the batch start recorded in the manifest and stops before any property whose forecast would cross the cap. Both replicas share one `RENOVATION_TERRA_USAGE_ROOT`; each session's daily ceiling is set to `min(2.5M, cap − spent_since)`. This closes the two-independent-caps hole.
6. **Scorer** (`score` subcommand): implements §5. Joins the replay to labels on (`property_key`, `catalog_item_id` or CAP-007 successor, sorted evidence photo keys) because `make_estimate_id` includes the catalog sha (`catalog_audit_live_validation.py:810-812`) so condition ids differ under 3.2; Session 6 `compare` is the precedent. Reuses gate logic from `tools/compare_renovation_architecture_cutover.py` where it fits.
7. **Approved reliability fixes** (from A6d): (a) normalize lone surrogates at the postprocessing artifact write, regression test with `\udc8f` in a description; (b) under mode `new`, refuse startup when `RENOVATION_SOL_MODEL` is unset instead of the fallback at `tools/pipeline_config.py:252-253`, with a test. Both engine-agnostic and small. Landing them here means Sessions C–E all run the final configuration.

Finished means: Tier-0 reproduction passes 18/18 (or the gap is reported and ruled on), `verify` and `plan` exit 0, forecast ≤ 1.0M Terra per replica, pinned run manifest written with budgets set (this fills the `reports/catalog_audit_live_experiment_manifest_v2.json` authorization surface), fixes merged with tests.

### Session C — Batch 1a: replica 1 (Terra ≈ 0.92M forecast; Sol ≈ 150k; one UTC day)

All 18 canary properties from `artifacts_canary/renovation_session9_20260818/run_1/candidate/`. Production worker idle that day (shared Sol ledger). Score against stored run_1/run_2 and the reviewed cases. Interim report, no decisions.

### Session D — Batch 1b: replica 2 (same forecast; next UTC day)

Same inputs, new run namespace, same usage root. Replica 1 vs 2 gives the under-3.2 noise context (2d + Terra + Sol). Full scorer. Draft go/no-go.

Batch 1 shared cap: **2,000,000 Terra**, enforced by the driver. Sol ≈ 300k across two days.

### Session E — Batch 2: production-path smoke of the final configuration, decision record (Terra ≤ 0.25M; Sol ≤ 40k)

1. With the temperature pin, any adopted gate, and the approved fixes all merged: `scripts/run_worker_smoke.py --mode new --routing production --dotenv <renointel-prod/.env> --property redfin_10806500` (7 photos, ~95k Terra), then `scripts/verify_renovation_artifact.py --expect-mode new --expect-terra-model gpt-5.6-terra --expect-sol-model gpt-5.6-sol`; confirm 2d model, `model_routing`, guard, Sol model, and 0.1 in the effective config. Second property optional.
2. `docs/DECISION_backend_acceptance_<date>.md`: GO / NO-GO / RETAIN, every material delta fixed, accepted, or named as a blocker; the smoke's limitation stated explicitly (operational execution under production routing, not end-to-end quality stability).

Batch 2 remainder (~1.75M) is **reserved and unspent by default**. It is spent only on the one hypothesis Session A may produce, and only after Steven authorizes that specific design.

## 5. Go/no-go criteria (pre-committed)

**Reviewed-case disposition ledger.** Every original reviewed case gets exactly one disposition in the replay: **retained** (same condition, same verdict and route), **represented elsewhere** (successor item or merged condition, named), **policy-superseded** (D1 trim `no_action`, D2 wallpaper exclusion, CAP-007 split, an adopted gate; the decision cited), **lost** (no equivalent condition, or verdict/route changed against the human label), or **unresolved** (cannot be joined). Populations: 125 `labels_v1_1.json` cards, 23 rc_ miss cases, 12 rc_ hallucination cases, the 36 worklist rows, the 6 closeout cases, and the 40 gold cases where a condition existed. No case is excluded because its condition "did not reproduce"; that is a `lost` or `unresolved` row.

| Check | Pass condition | Threshold source |
|---|---|---|
| Invariants | 18/18 × 2 envelopes `state: complete`, validator clean; zero occurrences of the CAP-007 parent id; every `dated_interior_trim` condition routes `no_action` at $0; no `dated_wallpaper_present` in bathroom scenes | D1, D2, CAP-007, S7-15 |
| 2d stability | Replica 1 vs 2 selection agreement reported per row; the 0.1 validation showed 1 unstable of 93 | `docs/RESULT_pass2d_closeout_20260910.md` §2 |
| Terra replica context | Verdict flip rate replica 1 vs 2 reported against the directional floors (4.05% supported→not, 23.1% unsupported→supported). **Context only**: an individual lost case is explained on its own, never excused by the aggregate | `docs/RESULT_label_v1_1_20260829.md` |
| Legitimate work | Every `lost` disposition on a supported-and-warranted card listed with cause and dollar effect | A2 obligation |
| False claims | If the gate is adopted: each hallucination case's disposition; if not: hallucination exposure recorded as an accepted limitation with the v1.1 figures and their bands | Steven's priority |
| Packages | Every changed candidate (15 support-list-only + 9 shape + 1 disappearing on the replayable set) has a fresh Sol decision; each change classified; Sol agreement across replicas reported | D7, LQ-3 |
| Duplicate billing | Row 23 / CCF-13 case resolves to one billing; `work_dedup_collisions` reviewed | S7-15 |
| Headline | Per-property replay-vs-stored delta above 15% reviewed by Steven with its package explanation; replica spread reported alongside as context, not as an automatic excuse | `configs/renovation_architecture_cutover.json` |
| Six cases | Disposition recorded like any other case; a verdict change is reported as observed, with the replica-2 result beside it, and classified by Steven | A1 |
| LQ-1 (33 bathroom rows) and LQ-2 (trim $0) | Answered by the same run; reported as named subsets | manifest_v2 |
| Production path | Session E smoke passes the verifier under production routing with the final configuration; limitation stated | C3/C4 precedent |

Outcome vocabulary: **GO** = all rows pass or are explicitly accepted by Steven; **NO-GO** = a named blocker; **RETAIN** = no configuration change adopted beyond the 0.1 pin and the reliability fixes, current implementation re-confirmed. All three are successful endpoints.

## 6. What Steven must decide, and when

| When | Decision |
|---|---|
| After Session A | Six cases: accept as unresolved misses, or authorize the one hypothesis (if any) as a separate, costed experiment |
| After Session A | Corroboration gate: adopt per item from the reviewed-cohort tradeoff, or not |
| After Session A | If production 2d runs on Terra: keep that, or pin production 2d to Qwen@0.1 (all 2d evidence is on Qwen) |
| After Session A | Approve the two reliability fixes |
| After Session B | If the 2c lane is not fully recoverable from the artifacts: accept the fallback (fresh 2c from the persisted 2b struct, Terra cost reported) or stop |
| At go/no-go | Accept fresh-2a variance (Jaccard 0.53; v5 per-property replica swings to +100% low) as a stated limitation of the launched product |
| At go/no-go | Launch throughput: Sol binds at ~5 listings/day while v4's Pass 2f runs for a FE that renders v4; the lever is FE v5 adoption / 2f sunset (S8), deferred by Steven |

## 7. Preserved, not reopened

D1–D8; CAP-022 declined; CAP-013 evidence-eligible but schema-blocked and unauthorized; S5-1 closed; electrical quarantine; FE contract and pricing deferrals; Pass 2a prompt unchanged (the gate is a catalog policy field, not a 2a change; the freeze's stated scope has lapsed but no prompt change is proposed). Historical migration of the 97 stored artifacts across six roots stays separately scoped; Sessions C/D execute cutover steps 1–5 on the session9 run_1 root only. The `pass2d_prompt_v3` branch stays unmerged.

## 8. Budget

| Session | Terra | Sol | Local Qwen |
|---|---|---|---|
| A | 0 | 0 | benchmark slices at 0.1 |
| B | 0 | 0 | Tier-0 proof (frozen 2d, no calls); optional 1-property live-2d dry run |
| C (Batch 1a) | ≈ 0.92M | ≈ 150k | 18 properties × every 2c observation |
| D (Batch 1b) | ≈ 0.92M | ≈ 150k | same |
| Batch 1 cap | **2.0M shared**, driver-enforced via `spent_since` | 250k/day | — |
| E (Batch 2) | ≤ 0.25M | ≤ 40k | smoke |
| Reserved | ≈ 1.75M unspent by default | — | — |
| **Committed** | **≈ 2.1M of 4M** | ≈ 340k over 3+ days | unlimited |

Forecast basis: Session 9 `canary_budget_debited_tokens` 1,836,273 for both replicas of the condition-review call across 18 properties (`docs/DECISION_renovation_architecture_session9_cutover_20260821.md` F7); Sol ≈ 8.3k per property. Session B's `plan` output replaces these before any spend. If `plan` forecasts more than 1.0M per replica, replica 2 is trimmed to the properties whose replica-1 result is outside the noise context, and the trim is recorded.

## 9. Verification

- Session A: `.venv\Scripts\python.exe -m pytest tests/test_scene_classifier_passes.py tests/test_error_attribution.py -q`; benchmark slice scores at 0.1 vs stored; both analysis scripts re-run to byte-identical output; the temperature pin shown absent from 1a and summarizer requests (log the effective request payload once each).
- Session B: Tier-0 reproduction 18/18 with byte-identical lanes and equal Terra fingerprints; `verify` and `plan` exit 0; `_guard_out_root` refuses the session9 and production roots; a synthetic ledger test shows the driver stops at the batch cap across two UTC days; tests for both fixes; full suite with the nine known catalog-drift failures unchanged from `master`.
- Sessions C/D: `scripts/show_daily_token_spend.py --root <usage-root>` equals the sum of per-property `token_usage`; `scripts/verify_renovation_artifact.py` on every written artifact; disposition ledger has one row per original reviewed case with no blanks; scorer output committed under `reports/`.
- Session E: smoke exit 0 and verifier pass under `--routing production`; effective config printed and archived with the decision record.