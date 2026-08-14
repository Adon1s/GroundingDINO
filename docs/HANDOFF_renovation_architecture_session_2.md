# Handoff — Renovation Architecture Session 2

Date: 2026-08-14. Branch: `renovation_architecture_rework` (all Session 1 and
Session 2 work uncommitted by design; nothing staged or pushed). Scope
executed: Session 2 of
`docs/IMPLEMENTATION_PLAN_renovation_scope_estimate_architecture.md` —
condition aggregation, evidence facts, bounded Terra review, deterministic
disposition, per-unit checkpoints, and the daily Terra usage guard. Output
remains private shadow data under
`analysis_debug.renovation_architecture_shadow_v1`; v4 output and frontend
contracts are unchanged; `current` mode stays a no-op.

## Files added

- `tools/renovation_architecture/conditions.py` — product-filtered lane →
  `ConditionDraft` (ObservedCondition + per-issue evidence refs). Reuses
  `build_room_surrogates`, `build_estimate_units`,
  `_estimate_scope_key_for_issue`; `EstimateCandidate` was not revived.
  Stale catalog ids and photo-less issues raise typed operational failures.
- `tools/renovation_architecture/evidence.py` — `evidence_dedup_v1` identity:
  exact = sha256 over `{w}x{h}|` + normalized RGB pixel bytes; near = 64-bit
  8×8 average-hash, Hamming ≤ 6 AND max-channel mean-RGB delta ≤ 16;
  union-find closure; lexicographically-first representative per view class;
  `distinct_view_count` is what `min_photo_evidence` gates.
- `tools/renovation_architecture/disposition.py` — one pure
  `decide_disposition(verdict, terminal_route, distinct_view_count,
  min_photo_evidence_required)`; the table is the whole policy
  (`condition_disposition_v1`).
- `tools/renovation_architecture/terra_review.py` — versioned prompt
  (`terra_condition_review_v1`), strict per-call JSON schema (condition-id
  enum), request fingerprint (`sha256_canonical` over schema version,
  projection fingerprint, prompt/model/effort/cap, dedup policy, condition
  payload, sent-image hashes), closed-contract response parsing.
- `tools/renovation_architecture/checkpoints.py` — atomic per-estimate-unit
  checkpoints under
  `<artifacts_root>/<property_key>/.checkpoints/<stable_run_id>/renovation_architecture/`.
- `tools/renovation_architecture/usage_guard.py` — `TerraUsageLedger`
  (SQLite, `BEGIN IMMEDIATE`, UTC-day rows, reserve → settle), 2,500,000/day
  ceiling, reservation `max(25000, cap + prompt-bytes + 5000×images)`,
  `TerraDailyBudgetExceeded`.
- `tools/renovation_architecture/review_pipeline.py` — orchestration:
  conditions → evidence → per-unit (checkpoint reuse | reserve → Terra →
  settle → checkpoint) → dispositions → self-validated result.
- `tests/test_renovation_architecture_conditions.py` (16 tests)
- `tests/test_renovation_architecture_terra.py` (22 tests; also the shared
  fake-provider harness the other new files import)
- `tests/test_renovation_architecture_disposition.py` (31 incl. matrix)
- `tests/test_renovation_architecture_usage_guard.py` (16 tests)

## Files changed

- `tools/renovation_architecture/contracts.py` — schema v2 (contracts AND
  envelope), `condition_review_complete` state, additive fields
  (ObservedCondition lineage quartet; EvidenceFacts representatives +
  exact/near groups + dedup version; ConditionReview terra_call_id/
  fingerprint/provider; ConditionDisposition evidence_id/terminal_route),
  new `TerraCall`/`TerraUnitUsage`/`TerraListingUsage` records,
  `POLICY_VERSIONS` map (4 keys), `DISPOSITION_REASON_CODES`,
  `REVIEW_RATIONALE_MAX_CHARS = 400`.
- `tools/renovation_architecture/ids.py` — `make_terra_call_id` (`tc1`).
- `tools/renovation_architecture/validators.py` — extended field sets and
  record validators, `tc1` id pattern, `condition_review_complete` envelope
  branch, `validate_condition_review_result()` (closed keys; evidence
  view-class recomputation; disposition recompute-and-compare against
  `decide_disposition`; exact call → unit → listing token reconciliation
  incl. `cached_input_tokens <= input_tokens`), shared condition-lattice
  helper factored out of `validate_complete_result` (whose invariant code is
  otherwise untouched — it stays the Session 5 gate).
- `tools/renovation_architecture/runtime.py` — runtime carries
  `terra_model`/`terra_max_output_tokens` (shadow init rejects an empty model
  and a non-positive cap: fail-closed startup); `build_shadow_envelope`
  widened with the Session-2 inputs and now emits `condition_review_complete`
  or a `failed` envelope whose reason is the `classify_failure` category.
- `tools/pipeline_config.py` — `resolve_renovation_terra_model` (falls back
  to `OPENAI_MODEL`) and `resolve_renovation_terra_max_output_tokens`
  (default 8192, positive int or `ValueError`), both invoked at module level.
- `tools/analyzer_cli.py` — `PropertyAnalysisJob.source_run_id` (default
  None); typed Terra overrides in `_apply_env_overrides` re-running the pure
  resolvers; init call passes the Terra settings.
- `tools/analyzer_server.py` — job gains `source_run_id=run_id` (the API
  runId with jobId fallback); init call passes the Terra settings.
- `tools/artifact_writers.py` — `photo_key_to_path` hoisted out of the
  Pass-2f guard (built unconditionally; v4 gates 2f on client/config, never
  map presence); seam widened with lane/photos/paths/metadata/vlm_client/
  artifacts_root and `run_id = getattr(job, "source_run_id", None) or
  job.job_id`; last-resort envelope literal bumped to schema 2. The seam
  still never raises.
- `tools/vlm_client.py` — additive `cached_input_tokens` in both usage-stat
  shapes, `_record_usage(..., cached_input_tokens=None)`, the three OpenAI
  paths read `usage.input_tokens_details.cached_tokens`.
- `tools/failure_taxonomy.py` — one `_CODE_CATEGORY` entry:
  `TerraDailyBudgetExceeded → quota`. Category list unchanged (TS parity safe).
- `tests/test_renovation_architecture_contracts.py` — builders carry the v2
  fields; new `TestConditionReviewEnvelope` (+ `_review_result()` shared with
  the disposition tests). `tests/test_renovation_architecture_runtime.py` —
  scaffold pins replaced by `condition_review_complete` expectations, Terra
  config-resolver and fail-closed init tests, `source_run_id` plumbing pins.

## Decisions and deviations

- **`POLICY_VERSIONS` is a dict constant; `POLICY_VERSION_KEYS` is derived.**
  Every envelope — including `failed`/`runtime_not_initialized` — must carry
  exactly the four policy keys (terminal_route, condition_disposition,
  evidence_dedup, terra_review_prompt). The two can never drift.
- **Terra env vars use a typed override, not `STRING_OVERRIDE_KEYS`** (a
  deviation from the session plan's wording, same intent):
  `_apply_env_overrides` re-runs the pure resolvers so the `OPENAI_MODEL`
  fallback tracks other overrides and an invalid cap raises before any job. A
  runtime test pins that the keys are NOT in `STRING_OVERRIDE_KEYS`.
- **`scaffold` stays in `ENVELOPE_STATES`** as the minimal-valid-envelope
  vocabulary for tests; the runtime never emits it again. Removal is
  Session 6 cleanup.
- **Near-duplicate = average-hash, not DCT pHash** (Pillow-only, no numpy).
  Uniform images share the degenerate all-zero hash; the AND'ed mean-RGB gate
  (≤ 16 max-channel delta) carries those cases — pinned by test. The near
  family excludes exact-equal pairs (they belong to the exact family); the
  merged `duplicate_groups` closure reunites both. Any rule change must bump
  `EVIDENCE_DEDUP_POLICY_VERSION` (invalidates fingerprints + checkpoints).
- **The disposition table is executable, not advisory**:
  `validate_condition_review_result` recomputes `decide_disposition` per
  condition and rejects any result whose recorded
  `(disposition, reason_code)` disagrees.
- **Checkpoint reuse keeps the original provider token numbers** on the
  republished `TerraCall` (`usage_source="checkpoint"`,
  `budget_debited_tokens=0`) so per-listing usage stays meaningful while the
  ledger never double-debits. Reuse never touches the ledger.
- **Settle before parse.** Provider tokens are debited to truth even when the
  response then violates the review contract; a provider/timeout failure
  settles as unknown and the conservative reservation stands. Unknown usage
  puts zeros on the call record (nothing is invented) with the reservation as
  the debit.
- **Terra calls are unattributed in timing-stats pass telemetry** — no fake
  pass number was invented for `_canonical_pass_key`. The authoritative
  accounting is the envelope's call/unit/listing records (snapshot/delta
  around each sequential call).
- **Unit-resolution honesty**: a condition whose contributors mixed
  photo-mapped and scope-room-fallback resolution keeps the weaker
  `scope_room_fallback` source; `identity_ambiguous` is also true whenever
  the estimate-unit confidence is `default_assumption` or
  `conservative_assumption` — note this flags the common single-kitchen
  default merge as ambiguous, which is honest, not a bug.
- **Lane rows without a `catalog_item_id` are skipped** (display-only,
  mirroring `extract_estimate_candidates`); rows with an unknown id, and rows
  with no `photo_key`, raise typed dependency failures — operational
  problems never become `cannot_assess`.
- **Server estimate identity changed vs Session 1** by design: the
  architecture run id is now the stable API runId (`job.source_run_id`), not
  the per-attempt internal job id. Photo/issue ids and everything else keep
  using `job.job_id`. Nothing pinned the old estimate ids.
- **`audit_runner.py` stays unwired** (Session 1 posture); its job object
  lacks `source_run_id` and the seam's `getattr` fallback covers it.

## Commands run and results

All on 2026-08-14, all green:

```
.venv\Scripts\python.exe -m pytest tests\test_renovation_architecture_contracts.py tests\test_renovation_architecture_catalog.py tests\test_renovation_architecture_runtime.py tests\test_renovation_architecture_conditions.py tests\test_renovation_architecture_terra.py tests\test_renovation_architecture_disposition.py tests\test_renovation_architecture_usage_guard.py -q --basetemp=C:\tmp\rv_pytest_session2_focused_20260814
  -> 228 passed, 3.35s

.venv\Scripts\python.exe -m pytest tests\test_estimate_units.py tests\test_room_surrogates.py tests\test_renovation_estimate.py tests\test_min_photo_evidence_gate.py tests\test_product_quarantine.py tests\test_artifact_writers.py tests\test_analyzer_server_failclosed.py tests\test_pass_failures.py tests\test_failure_taxonomy.py tests\test_analyzer_timing_stats.py tests\test_package_pass_2f_and_vlm.py -q --basetemp=C:\tmp\rv_pytest_session2_regression_20260814
  -> 292 passed, 2.05s

.venv\Scripts\python.exe -m pytest tests -q --basetemp=C:\tmp\rv_pytest_session2_full_20260814
  -> 2067 passed, 6 skipped, 21.45s   (Session 1 baseline: 1966 passed, 6 skipped; +101 new tests)
```

All warnings are pre-existing (utcnow deprecation, google-genai). Fake
providers only; no live Terra call was made at any point.

## Session 2 acceptance gates

- Every input condition ends in exactly one explicit terminal disposition —
  validator-enforced (lattice: one evidence/review/disposition per condition,
  no orphans).
- Duplicate/near-duplicate images never inflate corroboration —
  `distinct_view_count` is recomputed by the validator from the disjoint
  closure and is the only thing `min_photo_evidence` gates.
- Supported-but-under-threshold findings are withheld on every route,
  including exclusion routes.
- Unsupported findings produce no work or dollars — the Session-2 result has
  no work/package/price lanes at all (their keys are rejected), and
  `unsupported → excluded`.
- `cannot_assess → inspection`, private/debug-only, outside visible scope.
- Provider/timeout/parse/schema/cardinality failures fail closed as taxonomy
  categories on a `failed` envelope; completed unit checkpoints survive; no
  partial Session-2 result is ever published.
- Token accounting reconciles exactly call → estimate unit → listing
  (validator-enforced), cached input measured separately, budget debits never
  double-count provider tokens.
- Terra can never emit work, package, quantity, price, or confidence fields —
  strict response schema, closed-contract parser, and the validator's layer
  boundary all reject them independently.

## Risks / notes for the reviewer

- **Terra checkpoints ride the server's run-checkpoint lifecycle**:
  `_clear_checkpoint` on full job success and `_prepare_checkpoint_dir`'s
  image-policy rmtree remove them along with the image checkpoints. They are
  retry-only artifacts; a shadow failure inside an otherwise successful job
  loses its Terra checkpoints. Accepted Session-2 posture.
- **Seam durability posture unchanged**: the publication gate runs after the
  seam, so a gate failure still loses the shadow envelope (Session 1
  accepted; checkpoints make the retry cheap).
- **`photo_key_to_path` is keyed by basename** — duplicate basenames across
  directories collide (pre-existing Pass-2f behavior, deliberately not
  "fixed").
- **PassExecutionError-wrapped provider 429s classify as `provider`** (the
  taxonomy's body-sniffing quota refinement only runs on live exceptions) —
  pre-existing Pass-2f parity. The ledger denial is the spec'd `quota` path.
- **The ledger admits, then the provider call runs outside the transaction**:
  between reserve and settle the day's sum includes the conservative
  reservation, so concurrent workers see an honest upper bound. WAL is
  deliberately not used (artifacts roots may be network shares).
- **Shadow mode now spends real Terra tokens per listing** when enabled with
  a live key — that is its purpose; the guard caps it at 2.5M/day per
  artifacts root. `current` mode remains the production default.
- In shadow mode the envelope is still written even when v4 itself failed —
  the seam runs after and independently of the v4 try/except.

## Session 3 prerequisites

- Consume ONLY `accepted_for_work` dispositions from
  `result["condition_dispositions"]` when deriving work items. Every other
  terminal outcome (`excluded`, `inspection`, `withheld`, `no_action`) must
  be preserved visibly in its lane — never dropped, never silently promoted.
- The input contract is `validate_condition_review_result` — treat it as
  frozen; Session 3 adds its own state/keys additively (with a schema bump if
  fields change) and leaves `validate_complete_result` as the Session 5 gate.
- Work derivation reads the projection's `work_policy` (action_code,
  action_source, trade_bucket, unit_policy, pricing_mode, cost) keyed by the
  condition's `catalog_item_id`; `terminal_route` is already recorded on each
  disposition.
- New policies extend `POLICY_VERSIONS` (validators pin exact key equality —
  the intended friction, same as the state change was for this session).
- The per-unit checkpoint fingerprint recipe covers the Terra call inputs
  only; Session 3's deterministic derivation needs no checkpoints of its own.
