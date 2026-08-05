# HANDOFF — Kind ontology Task 1: atomic observations + three-kind classification

2026-08-05. Task 1 of 3 in the `defect | upgrade` → `defect | degradation | modernization`
rework. This document is the contract Task 2 (catalog + resolution migration) builds on.

## Status

- **Done and verified**: observation-kind-v2 semantic contract implemented as a hard
  replace of the v1 Pass 2b/2c contract; pipeline ends after classification
  (non-publishable); benchmark built, human-reviewed, frozen, and run — **all 8 holdout
  gates pass on GPT-5.6 Terra**; full test suite green (1515 passed / 23 skipped, all
  skips enumerated below).
- **Frozen until Task 3 cutover**: no active-listing analysis may run. End-to-end
  analyzer invocations fail by design at `write_photo_intel` with the
  classification-only guard message. Task 3 must reanalyze active listings after cutover.
- **Not migrated (deliberately)**: catalog items/IDs/kinds, Pass 2d retrieval +
  resolution, Pass 2e, shadow lane, pricing, scoring, packages, display, API, historical
  artifacts, reprojection. `tools/issue_catalog.json` still carries v1 kinds
  (defect 67 / upgrade 40) — that migration is Task 2.

## The ontology (approved by Steven, benchmark-validated)

- **defect** — expected function, safety, integrity, or protection has FAILED. Broken or
  missing required parts, active leaks, rot, mold/mildew, structural damage, unsafe
  conditions, failed weather protection, rusted-through metal.
- **degradation** — the item works and nothing has failed, but it has visibly
  deteriorated: wear, fading, cosmetic staining, scuffing, surface rust/corrosion,
  peeling finish, aging, weathering.
- **modernization** — functional and acceptably maintained, but dated, basic, low-grade,
  or a discretionary improvement opportunity.

Precedence and boundary rules (all encoded in `PASS_2C_SYSTEM_PROMPT`, each added rule
traceable to a measured dev miss):

1. Defect requires stated/visible failure; ambiguous deterioration defaults to degradation.
2. Visible deterioration beats dated/style language; purely dated/basic is modernization.
3. Within one atomic claim: defect > degradation > modernization.
4. **Paving only** (driveways, walkways, patios): cracking, surface wear, or minor
   unevenness (incl. uneven joints) = degradation; heaving, raised trip edges, or
   crumbling = defect. *(Steven's review call on def-009: flatwork cracks are not
   automatically defects.)*
5. **Everywhere except paving, cracks are defects** (walls, ceilings, tiles, panes,
   basins, fixtures).
6. Mold/mildew growth = defect.
7. **Explicit water stains / water damage = defect** (moisture intrusion evidence);
   ordinary dirt or cosmetic staining = degradation.
8. An item described only by low-grade/dated material or grade (laminate, hollow-core,
   builder-grade, basic) = modernization, not neutral presence.
9. Asserted missing required/protective component = defect; absent optional feature =
   modernization; absence inferred only from "not visible" = excluded
   `unsupported_or_speculative` (bakes in the gutter absence-safety learnings).

**Atomicity (Pass 2b)**: one condition claim per observation. Mixed text splits when
claims differ in type ("weathered and rotted deck boards" → two observations;
"worn and dated cabinets" → two). Same-type detail stays together ("faded and stained
siding" = one degradation). Splitting is deliberately preferred over defect-precedence
merging because it is lossless — downstream dedupe can merge splits, but a merged claim
can never be re-split. 2b's damage list explicitly includes water-stained/water damage
so moisture claims separate from wear claims.

**Groundedness boundary (decided in review, mix-009)**: 2c classifies what the text
*asserts*. Unverifiable claims ("leaking faucet" in a photo) are a 2a/2b grounding
problem — and eventually per-item 2F visual verification — not a 2c classification
problem. Do not teach 2c to discount asserted failures.

## The v2 contract

- `ONTOLOGY_VERSION = "observation-kind-v2"`; `OBSERVATION_KINDS = {defect, degradation,
  modernization}`; `EXCLUSION_REASONS = {good_condition, neutral_presence,
  advice_or_process, unsupported_or_speculative, measurement_overlay,
  not_renovation_related}` — `tools/scene_classifier_passes.py`.
- Pass 2c receives a numbered observation list and returns **indexed decisions only**
  (it cannot rewrite descriptions):
  ```json
  {"decisions": [{"index": 1, "kind": "degradation"}, {"index": 2, "exclude": "neutral_presence"}]}
  ```
- **Fail-closed validation** (`_validate_pass_2c_decisions`): every index exactly once;
  exactly one of kind/exclude; closed enums; violations →
  `PassExecutionError('2c','parse')`. The v1 unknown-label→"other" silent coercion is
  gone on purpose — that trap hid prompt/contract drift.
- **Deterministic overlay pre-filter** (`partition_dimension_overlays`): dimension
  overlays are excluded with `measurement_overlay` before the LLM call (reuses `_DIM_RE`
  + the high-signal-damage escape hatch). `measurement_overlay` is still accepted from
  the model for regex misses.
- `Pass2cResult`: `observations` (`[{description, kind}]`), `excluded`
  (`[{description, reason}]`), `raw_response`, `ontology_version`.
  `labeled_debug`/`labeled_forward` no longer exist.
- Prompt provenance (finally exists for 2b/2c, following the 2f pattern):
  - `PASS_2B_PROMPT_VERSION = "pass_2b_atomic_v2"`,
    SHA256 `243ae92a37fd300f517e6997dcd1d2dcd8e87c36b68e7d17606df55a99c94f02`
  - `PASS_2C_PROMPT_VERSION = "pass_2c_kind_v2"`,
    SHA256 `e074c606e0719b85e72b20a8b68532318f8ac0ed6a6df125edd3b6c9b782a0bd`
    (recorded before the holdout run was opened)

## Files / interfaces changed

| File | Change |
|---|---|
| `tools/scene_classifier_passes.py` | v2 2b/2c prompts, ontology constants, prompt versions+SHAs, indexed validation, overlay pre-filter, new `Pass2cResult`; removed `VALID_LABELS`, `_coerce_labeled_2c`, `force_other_if_dimensions` |
| `tools/scene_classifier_orchestrator.py` | classification-only stop after 2c; result fields `observations` / `excluded_observations` / `classification_only` / `ontology_version`; issue_id stamping uses kind; shadow/2d/2e code left in place but **dormant/unreachable** |
| `tools/scene_classifier_service.py` | payload carries new fields + `classification_only` marker |
| `tools/artifact_writers.py` | `write_photo_intel` raises `RuntimeError` on any `classification_only` payload (the freeze-enforcement point) |
| `tools/pipeline_common.py` | `LEGACY_ONTOLOGY_VERSION = "legacy_v1"` + `artifact_ontology_version()` — stored artifacts without `ontology_version` are legacy v1 and must never be reinterpreted |
| `tools/model_comparison.py`, `tools/bias_check.py`, `tools/catalog_auditor.py` | fail-fast `RuntimeError` at `main()` — they consume the retired v1 label vocabulary; migration is Task 3 measurement work |
| `scripts/benchmark_kind_ontology.py` (+`benchmarks/kind-ontology-v2/`) | new standalone benchmark runner, case sets, v1 prompt snapshot, model configs, reports |
| tests | see below |

Unchanged and still valid for legacy artifacts: `tools/quant_artifact_comparison.py`,
`scripts/audit_pass2c_funnel.py` (both read stored v1 artifacts, not live passes).

## Benchmark: kind-ontology-v2 (all numbers are real, GPT-5.6 Terra low reasoning, 5 repeats)

132 human-reviewed cases (Steven reviewed all 132; amendments: def-009→degradation +
flatwork rule, def-005/mix-004 sharpened with displacement language, def-014/mix-009
unverifiable-leak claims replaced). Frozen fingerprints: dev (93) `ff95e151eb684039…`,
holdout (39) `e56998c7c51995fd…`. Full protocol in `benchmarks/kind-ontology-v2/README.md`.

### v1 baseline (retired two-kind prompts replayed; the motivating stat)

- **Degradation-gold decisions: 110/110 (100%) landed on `defect_or_damage`**, 0 on
  `upgrade_candidate`. The entire wear band inflated the defect lane under v1.
- Excluded-gold text forwarded 25% of the time.
- Modernization-gold → `upgrade_candidate` only 91.4% (rest `generic_presence`).
- Report: `benchmarks/kind-ontology-v2/results/20260805_142500_v1-baseline_gpt-56-terra_dev/`

### v2 development (after one tuning iteration; misses → rules 4-8 above)

- Pre-tuning: 92.7% atomic accuracy (patterned misses: cracks-as-degradation,
  water-stains-as-cosmetic, mold, laminate-as-presence).
  Report: `results/20260805_143122_v2_gpt-56-terra_dev/`
- **Post-tuning: 99.7% atomic accuracy** (per-repeat [0.984, 1.0, 1.0, 1.0, 1.0]);
  recalls defect 1.000 / degradation 0.991 / modernization 1.000; precision ≥0.990 all
  kinds; excluded false-classification 0%; unanimity 98.7%; pairwise 99.5%;
  **mixed-case full success 100%, cross-kind bundling 0%**; 0 schema failures.
  Residual: `deg-007` ("worn and uneven hardwood") → defect in 1/5 repeats.
  Report: `results/20260805_143940_v2_gpt-56-terra_dev/`

### v2 holdout gates (Terra) — ALL 8 GATES PASS

- **Atomic kind accuracy 100%** in all 5 repeats; recall and precision 1.000 for every
  kind; unanimity 100%; pairwise agreement 100%; 0 schema failures; excluded
  false-classification 0%.
- Excluded reason accuracy 83.3% (5/30 decisions picked a different *reason* while
  still correctly excluding — lane-correct, reason-mismatch; not gated).
- Mixed-case full success 96.7% (29/30), cross-kind bundling 0%. The single miss:
  `mix-022`, "garage door is dented at the corner" → degradation in 1/5 repeats
  (splitting was perfect all 5). Analysis, recorded per holdout discipline instead of
  retuned: the gold (defect) is arguably over-strict under the ontology's own
  evidence-of-failure rule — a dent with no stated functional impact reads as visible
  deterioration. Task 2 should treat dents-without-functional-failure as degradation
  when mapping catalog concepts (contrast def-019, where "does not close fully" makes
  the dent a defect).
- Report: `results/20260805_144319_v2_gpt-56-terra_holdout/`

### Qwen 3.6 27B transfer report (current production model; informational, not gated)

- Dev: atomic accuracy 93.7% (per-repeat [0.889, 0.905, 0.968, 0.952, 0.968]); recalls
  defect 1.000 / degradation 0.955 / modernization 0.857; excluded false-classification
  0%; unanimity 88%; pairwise 94.4%; mixed full success 100%, bundling 0%.
  **22 schema-failure case-decisions (4.7% of 465)** — Qwen intermittently emits
  decision rows violating exactly-one-of kind/exclude, concentrated in
  modernization-heavy batches; each failure aborts the whole call under the fail-closed
  no-retry contract. Also `deg-013` (roof moss/discoloration) → defect 5/5.
  Report: `results/20260805_145223_v2_qwen-36-27b_dev/`
- Holdout: atomic 100% all repeats, 0 schema failures, unanimity 100%; mixed 83.3% —
  entirely the `mix-022` dent-claim boundary above (5/5, split itself perfect).
  Report: `results/20260805_145553_v2_qwen-36-27b_holdout/`
- **Transfer conclusion**: the three-kind semantics transfer well (the wear band lands
  in degradation on both models; defect recall 1.0), but Qwen's *contract compliance*
  is not production-grade under fail-closed/no-retry — ~5% of 2c calls would abort the
  photo. Task 3 decision, recorded not made here: route 2c to Terra (consistent with
  `benchmarks/configs/terra-upstream-terra-2f.json`), add a bounded schema-retry, or
  harden the prompt for Qwen specifically.

## Tests

Full suite: **1515 passed, 23 skipped** (`.venv\Scripts\python.exe -m pytest tests -q
--basetemp=C:/Users/Steven/AppData/Local/Temp/claude/pytest`; the basetemp is the known
Windows temp-ACL workaround, an environment issue, not a code failure).

New coverage: `tests/test_scene_classifier_passes.py` (v2 contract: fail-closed
validation matrix, overlay pre-filter, prompt invariants, prompt-SHA pins,
classification-only orchestrator stop), `tests/test_kind_ontology_guards.py` (publish
guard + legacy helper), `tests/test_kind_ontology_benchmark.py` (16 harness tests).

Enumerated skips (dormant, not deleted — revive with Task 2/3):
- `tests/test_pass_2e_fail_closed.py`: 7 skipped (`dormant_2e`) — 2e failure path; the
  active invariant is pinned by `test_classification_only_stop_keeps_2e_dormant`.
- `tests/test_candidate_provider.py`: 10 skipped (`dormant_2d`) — 2d retrieval
  failure-policy and debug rows; pure signature/adapter tests still run.
- `tests/test_model_comparison.py`: 4 skipped (`blocked_live_2c`) — live-2c cells;
  fixture-replay/judge tests still run.
- 1 pre-existing (embeddings sidecar).
- Superseded (removed with reasons in-file): v1 label-enum prompt tests, safety-label
  coercion tests, interior-finish-anchor prompt test (anchors replaced by measured
  benchmark gates), weathered-exterior assertions (already reverted in working tree
  before this task; stay retired — degradation owns exterior wear now).

## Task 2 starting point

- Consume `Pass2cResult.observations[].kind` (or orchestrator
  `result.observations[]`) directly — kind is now assigned *by the contract*, not by the
  orchestrator's retired `_label_to_kind` mapping (dormant code below the stop).
- **Do not redefine the ontology.** The rules above are benchmark-validated and
  Steven-approved; if a catalog migration decision seems to require reinterpreting a
  kind, that's a blocking contradiction to raise, not silently patch.
- Migrate/split catalog concepts to the three kinds. Known conflations to resolve are
  documented in `docs/HANDOFF_catalog_kind_consistency.md` (12-item wear band, 6+
  damage+wear/style conflations). The wear band largely becomes `degradation`; note
  rule 4/5 (paving vs everything-else cracking) and rule 7 (water stains = defect) when
  deciding successor concepts, and Steven's review stance that flatwork cracking is not
  automatically a defect.
- Retrieval/2d rework notes: `evaluate_kind_routing` and `catalog_embeddings`
  `allowed_kinds` are hard filters over `{defect, upgrade}` — landmine: an unknown kind
  currently yields an *empty* `allowed_kinds`, which `retrieve_candidates` treats as
  **no filter at all** (silent whole-catalog search). The shadow lane hard-codes
  `{defect, upgrade}` too.
- Keep v2 non-publishable. Remove the classification-only stop **only** inside
  catalog-resolution benchmarking, never for artifact writes — the `write_photo_intel`
  guard must stay until Task 3 cutover.
- The kind-ontology benchmark harness is reusable: point it at resolution the same way
  (frozen cases, deterministic scoring, dev/holdout discipline).
- Ambiguities Task 2 must own: maintenance/cleaning concepts (dirt/soiling is
  degradation-adjacent but may be "cleaning", not renovation); asserted vs inferred
  absence (rule 9) against the gutter deny-list work; where dual-claim observations
  resolve when both claims map to one catalog concept (dedupe should collapse them).

## Task 3 landmines (recorded now so semantic cleanup doesn't silently change pricing)

- `"modernization"` string collision: `estimate_scope._VALUE_ADD_TERMS` contains the
  literal token and `_catalog_text` concatenates the kind string into scope matching
  (`estimate_scope.py:571-576`) → every modernization item would auto-flag
  `has_value_add`. Also collides with `PACKAGE_CATEGORY_MODERNIZATION` vocabulary.
- `costing.KIND_MULT = {defect: 1.0, upgrade: 0.6}` with silent `.get(kind, 1.0)` —
  a new kind gets 1.0 (≈ +67% vs upgrade) with no warning.
- `property_summary_pass._norm_kind` coerces unknown → `"defect"` (severity boost + cap
  changes); reprojection prefers *stored* `catalog_item_kind`, so reprojected legacy
  artifacts mix ontologies.
- Pass 2e hard-drops `invalid_kind:<x>` — any new kind reaching live 2e deletes the issue.
- `rehab_packages` driver predicates (`kind == "defect"` / `== "upgrade"`) drop
  three-kind items out of the package partition entirely.
- `catalog_cost_model` routes `upgrade` → ROOM_ALLOWANCE; renamed kinds fall to LINE_ITEM.
- `catalog_auditor` gives every non-defect kind the upgrade threshold silently.
- Blocked harnesses (model_comparison, bias_check, catalog_auditor) need migration to
  the v2 result shape before Task 3 measurement runs.
- v1 baseline evidence for "degradation ≠ defect pricing by default": under v1 the wear
  band was priced as defect (110/110); Task 3 decides degradation's economic behavior
  deliberately, with corpus-wide measurement.

## Gotchas

- The classification-only stop is an early `return` in `_run_passes` after the 2c block;
  everything below (shadow lane, 2d, 2e, `_label_to_kind`) is dormant and still
  references removed `Pass2cResult` fields — it will AttributeError if reached without
  Task 2 rewiring. That is intentional: loud, not silent.
- `benchmarks/kind-ontology-v2/v1_baseline_prompts.json` is the frozen v1 prompt
  snapshot — the live constants are gone, so baselines replay from this file only.
- Benchmark model configs are explicit; `.env` has no `OPENAI_MODEL` (stale `GPT_MODEL`
  alias only). Never rely on env model resolution. LM Studio serves
  `unsloth/qwen3.6-27b@q6_k` at `pipeline_config.LM_STUDIO_URL`.
- Working tree preserved: the weathered-exterior 2c revert (do NOT resurrect the two
  HEAD prompt lines from `7f7fa04`) and the unrelated gutter `deny_any` absence-safety
  work (catalog + tests + audit docstrings) are intact and uncommitted.
- Case gold amendments from review are recorded in each case's `rationale` field and in
  `benchmarks/kind-ontology-v2/REVIEW_cases.md`.

## Verification commands

```bash
.venv/Scripts/python.exe -m pytest tests -q --basetemp=C:/Users/Steven/AppData/Local/Temp/claude/pytest
```
```bash
.venv/Scripts/python.exe scripts/benchmark_kind_ontology.py --contract v2 --cases holdout --model-config benchmarks/kind-ontology-v2/models/terra.json --repeats 5 --gates
```
