# Benchmark Session 2 handoff — pipeline instrumentation

Session 1 landed the dataset and reference-authoring layer. Session 2 instruments the
analysis pipeline so a run's routing, prompts, and provenance are verifiable.

Plan: `C:\Users\Steven\.claude\plans\renointel-terra-benchmark-system-humming-whale.md`
(sessions 1–4A are committed; 4B/5/6/7 are a backlog gated on findings from 4A).

---

## What Session 1 landed

| File | Purpose |
|---|---|
| `tools/comparison_common.py` | **NEW** — `canonical_json`, `sha256_bytes`, `sha256_file`, `sha256_canonical`, `atomic_json`, `ComparisonError`. Extracted from `quant_artifact_comparison.py`, which re-exports them unchanged. |
| `tools/benchmarking/vocabulary.py` | Frozen vocabulary snapshot / load / diff, `default_actionability`, benchmark-owned enums. |
| `tools/benchmarking/schemas.py` | Strict multi-error validation for manifest / listing / reference / config, plus `reporting_category`. |
| `tools/benchmarking/dataset.py` | Photo integrity, containment, ordering, `dataset_fingerprint`, `import_listing`, `seal`, `require_unsealed`. |
| `tools/benchmarking/reference_authoring.py` | YAML draft ↔ canonical JSON; `template` / `check` / `compile_draft`. |
| `tools/benchmarking/catalog_index.py` | `search` / `format_results` for blind annotation. |
| `tools/benchmark.py` | CLI: `validate`, `import`, `reference`, `catalog`, `vocabulary diff`, `seal`. |
| `benchmarks/README.md` | The four annotation phases and the schema rationale. Steven's working doc. |
| `benchmarks/configs/terra-upstream-terra-2f.json` | First config profile; validated by a test. |
| `.gitattributes` | **NEW FILE** — LFS for `benchmarks/datasets/**/photos/**`. |
| `.gitignore` | `benchmark-results/` appended. |

**Scope change from the plan:** `comparison_common.py` was extracted in Session 1 rather than
Session 2, because dataset fingerprinting needed `canonical_json`/`sha256_file` immediately and
duplicating them would have let the two copies drift. Session 2 only needs to add the *remaining*
helpers (`ratio`, `wilson`, `jaccard`, `normalize_text`, `SentenceEncoder`, `range_delta`).

Tests: 190 added, all green. Full suite **1167 passed / 1 skipped / 0 failed**. The one skip
(`test_openai_compatible_encoder.py:195`) needs the llama-server sidecar.

---

## Frozen contracts — do not change without a version bump

These are consumed by Sessions 3 and 4A. A handoff may not silently supersede them; changing one
requires an explicit schema-version bump and a migration note.

- **`REFERENCE_SCHEMA_VERSION = 1`** and the reference shape in `benchmarks/README.md`.
  Session 4A's metrics read `findings[].presence/reporting_expectation/visual_sufficiency/
  catalog_status/catalog_item_ids/critical/evidence[]`, `photo_expectations[]`,
  `billable_components[].expected_units`, and `packages[].decision`.
- **`reporting_category(finding)` is the only derivation point.** Do not reimplement it. It is
  disjoint *because* validation rejects `indeterminate + must_not_report`; if you relax that rule
  the function needs precedence logic.
- **`VOCABULARY_SCHEMA_VERSION = 1`** and `reference_vocabulary.json`. Sealed references validate
  against their own frozen snapshot, never live code.
- **`dataset_fingerprint`** hashes JSON *canonically* and photos *by bytes*. Reindenting
  `metadata.json` must not read as "the dataset changed".
- **`BENCHMARK_MODEL_MAP_PASSES = ("1a","2a","2b","2c","2d","2f")`** — the routable passes.

---

## Session 2 work

Additive only. No analysis behavior may change.

### 1. `tools/comparison_common.py` — finish the extraction

Move the rest out of `quant_artifact_comparison.py` and re-export: `ratio`, `wilson`, `jaccard`
(L232–252), `normalize_text`, `SentenceEncoder` (L361), `range_delta` (L540).

**Do not port `match_issues` (L379).** Session 4A writes an occurrence-aware replacement. The
existing one uses bipartite assignment only for its embedding tier; the exact tier takes
`choices[0]` greedily and is therefore prediction-order dependent.

Keep `tests/test_quant_artifact_comparison.py` green (16 tests).

### 2. `tools/pass_config.py` — verbosity

Add `verbosity` to `resolve_openai_invocation` (L83), gated on
`model_supports_reasoning(resolved["model"])` exactly as `reasoning_effort` is (L147–151). Add
`normalize_verbosity_map` mirroring `normalize_reasoning_efforts` (L156). Add `verbosities` to
`SceneClassifierRunOptions` (L257) and `from_analysis_profile`.

This is the only missing link: passes already splat the resolved config into the client
(`scene_classifier_passes.py:77`) and `analyze_image` already reads `verbosity` from kwargs
(`vlm_client.py:1017-1020`), pre-gated at each call site (L466/553/616).

### 3. `tools/vlm_client.py` — capture the served model

⚠️ **Today no API-reported model identity is captured anywhere.** Every model string in every
artifact is the *requested* name (`model_routing[*].model`, `run.default_gpt_model`,
`pass_2f_trace.model`). So "verify every route was Terra" is currently unverifiable, and Session 3's
routing enforcement depends on fixing it.

Add `served_models` to `_empty_usage_stats` (L151) and `_empty_pass_usage` (L139), plus
`_record_served_model(name)` called alongside each `_record_call()` — six sites: L513/592/658
(OpenAI), L722/757 (Gemini), L838/906/969 (LM Studio). Read `getattr(response, "model", None)` for
the SDK paths and `data["model"]` for LM Studio. Attribution is already task-local via the
`_active_pass_key` ContextVar, so no new plumbing is needed.

### 4. `tools/analyzer_cli.py` — surface it

Add `served_models` to the per-pass block in `_compute_timing_stats` (L392) so it reaches
`timing_stats.passes.<key>.served_models` in the artifact. Additive;
`tests/test_analyzer_timing_stats.py` must stay green.

### 5. `tools/analyzer_server.py` — three changes

1. **Honor `passToggles`.** It is read by nothing today (L357–419), so the server always runs all
   nine passes at `failure_mode='strict'` regardless of the request, while still recording the
   requested toggles into the artifact via `options.toggles.to_dict()` (L608) — a config-vs-reality
   mismatch that looks like a clean run. Build `PassToggles.from_dict` into
   `SceneClassifierRunOptions.from_analysis_profile` at L425. Accept `verbosities` too.
   `_checkpoint_policy_fingerprint` (L247) already hashes `pass_toggles`, so invalidation is free.
2. **Enforce benchmark mode.** Accept `datasetRoot` and `resultsRoot`; when
   `RENOINTEL_BENCHMARK_MODE=1`, reject the job unless every resolved image path is beneath
   `datasetRoot` and `artifactsRoot` is beneath `resultsRoot`. Setting an env var creates no
   isolation on its own. (Lowest-value item in this session — it defends against a bug in the
   benchmark's own runner. Drop it if the session runs long.)
3. **Add `"requestJobId": ts_job_id` after the `**summary` splat at L681.** The success `result`
   message reports the *internal* job id there, because `_build_summary` sets `jobId` to
   `job.job_id` (`analyzer_cli.py:651`), while `progress`/`job_done`/`error` use the caller's id.
   Add a field rather than changing `jobId`, so no production consumer breaks.

### 6. Prompt versions and 2f provenance

Add `PASS_<K>_PROMPT_VERSION` constants for 1a/2a/2b/2c/2d in `scene_classifier_passes.py`,
following the existing `PASS_2F_PROMPT_VERSION` / `PASS_2F_PROMPT_SHA256` pattern (L1851–1865).

Stamp `prompt_template_version` / `prompt_template_sha256` onto the `run_pass_2f_batch` trace
(`rehab_packages.py:3803-3815`). `prepare_pass_2f_cases` already computes them (L3705–3706) but they
never reach `photo_intel["pass_2f_trace"]`, so production 2f output has no prompt provenance today.

### 7. `tools/benchmarking/provenance.py`

`prompt_manifest()` (one entry per executing pass: logical version + sha256 over system template,
user template, output schema, renderer version — fail if an executing pass has no entry, so a new
pass cannot land unversioned), `catalog_fingerprint()`, `cost_model_fingerprint()`,
`aggregation_fingerprint()`, `source_provenance()` (git commit, dirty flag, relevant-tree hash).

⚠️ **None of these gate baseline compatibility.** They are treatment and provenance dimensions. A
catalog change is something to benchmark, not something that should invalidate a baseline.
Compatibility keys only on dataset + reference fingerprint, reference schema version,
normalized-prediction schema version, and `EVALUATION_VERSION`.

### 8. Tests

Verbosity reaches the wire and is dropped for non-gpt-5 models. Served-model capture — **note
`conftest.openai_response` does not set `.model`; add a `model=` kwarg.** `passToggles` genuinely
disables a pass and shifts the checkpoint fingerprint. Benchmark-mode containment rejects an
out-of-root photo and an out-of-root `artifactsRoot`. Prompt manifest covers every executing pass
and its sha moves when a prompt changes. Extracted helpers behave identically.

No custom markers — there is no `pytest.ini` and `tests/conftest.py` documents that nothing may rely
on markers or plugins. Use builtin `skipif` (see `test_openai_compatible_encoder.py:195`).

---

## Gotchas found in Session 1

- **`load_issue_catalog` swallows a missing file** and returns `{"items": []}` rather than raising,
  so a bad `ISSUE_CATALOG_PATH` yields an empty catalog. `vocabulary.snapshot` fails closed on that;
  anything else reading the catalog for provenance should too.
- **The catalog's `scene_groups` includes `pool`**, which is not a UI scene group (`pool` is a scene
  whose group is `exterior`). Already documented at `catalog_validation.py:65-67` and exported as
  `VALID_SCENE_GROUP_TOKENS`. The snapshot records the two vocabularies separately; do not reconcile
  them.
- **Rehab scope bands have no backend owner.** They are frontend-only:
  `'light' | 'moderate' | 'heavy'` in `lib/types/topPicks.ts:54`, derived in
  `lib/topPicks/scopeBand.ts`. Mirrored by hand in `vocabulary.REHAB_SCOPE_BANDS`; if the frontend
  union changes, that constant must change with it. (The plan's claim that they live in
  `tools/estimate_scope.py` was wrong — that module owns `VALID_ESTIMATE_SCOPES`, a different thing.)
- **`actionability` reuses `VALID_PACKAGE_CATEGORIES`** plus `nonactionable`, rather than inventing a
  parallel vocabulary. Derived from `kind` + `scope` + `trade_bucket` by four ordered rules, and the
  derivation is frozen into the snapshot.
- **Production photo ordering uses two different comparators** — bare `.sort()` at
  `lib/property/propertyImages.ts:29` versus `localeCompare` at `lib/property/imageManifest.ts:52`.
  Order is therefore resolved once at import and read back verbatim by `dataset.photo_paths`.
- **A draft manifest legitimately has zero listings.** `import` is what adds the first one, so
  "at least one listing" applies only to a sealed dataset.

---

## Verify Session 2

```bash
.venv\Scripts\python.exe -m pytest tests -q
```

Baseline to beat: **1167 passed / 1 skipped / 0 failed**. Then confirm a real run records the served
model and honors toggles:

```bash
.venv\Scripts\python.exe -m tools.analyzer_cli --property-key smoke --images <one.jpg> --artifacts-root <scratch> --disable-2d --model-map "{\"1a\":\"gpt-5.6-terra\"}"
```

Check the artifact's `timing_stats.passes.1a.served_models` is populated and
`pass_states["2d"] == "skipped"`.

Session 2 exit: write `docs/HANDOFF_benchmark_s3.md`.
