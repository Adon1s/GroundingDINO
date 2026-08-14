# Handoff — Renovation Architecture Session 1

Date: 2026-08-14. Branch: `pass_2a_benchmarking` (HEAD f55d59e at start; nothing
committed — all Session 1 work is uncommitted by design). Scope executed:
Session 1 of `docs/IMPLEMENTATION_PLAN_renovation_scope_estimate_architecture.md`
— contracts, newest-catalog projection, startup selector, shadow scaffolding.
No Terra/Sol calls, no price changes, no frontend contract, no authoritative
new estimator.

## Files added

- `tools/renovation_architecture/__init__.py` — re-exports; deliberately does
  NOT import `catalog_projection` (current-mode startup never pays for its
  dependency chain).
- `tools/renovation_architecture/contracts.py` — version constants, enum
  frozensets, 11 frozen dataclasses + `to_dict()` (plain JSON-safe dicts).
- `tools/renovation_architecture/ids.py` — `make_*_id` constructors over
  `pipeline_common.stable_hash_id(length=16)`, prefix + namespace literal per
  record type. Estimate ID = property key + source run + catalog sha256 +
  projection fingerprint.
- `tools/renovation_architecture/catalog_projection.py` —
  `build_renovation_catalog_projection()` (strict v3.1-only, collect-all-then-
  raise `RenovationCatalogError`) and `resolve_terminal_route()`.
- `tools/renovation_architecture/validators.py` — `validate_envelope()` per
  state, `validate_complete_result()` invariants (frozen now; production use
  begins Session 5), local `ValidationResult` (do not "deduplicate" it into
  `tools.benchmarking` — per-module result classes are the repo precedent and
  benchmarking drags pass_config coupling).
- `tools/renovation_architecture/runtime.py` — `initialize_renovation_architecture()`
  (no-op in current mode), `get_runtime()`, `reset_runtime_for_tests()`,
  `build_shadow_envelope()` (scaffold, or a valid private `failed` envelope
  with reason `runtime_not_initialized` when init never ran).
- `tests/test_renovation_architecture_contracts.py` (54 tests)
- `tests/test_renovation_architecture_catalog.py` (31 tests)
- `tests/test_renovation_architecture_runtime.py` (22 tests)

## Files changed

- `tools/pipeline_config.py` — `RENOVATION_ARCHITECTURE_MODE=current|shadow|new`
  selector block after the kind-ontology block; pure resolver
  `resolve_renovation_architecture(raw, *, kind_ontology_version)`; module-level
  invocation so invalid values raise at import (= worker startup). `shadow`
  requires `KIND_ONTOLOGY_VERSION=observation_kind_v2`; `new` is recognized but
  raises until Session 6. It derives nothing else — `ISSUE_CATALOG_PATH` /
  `PIPELINE_MODE` stay owned by the kind-ontology selector, and there is
  deliberately no env-override path (`analyzer_cli` override sets untouched).
- `tools/analyzer_server.py` — init call between catalog load and the
  embeddings build; failure emits `{"type":"error","stage":"renovation_architecture_init"}`
  and returns 1 before "ready" (embeddings fail-closed shape).
- `tools/analyzer_cli.py` — same init after catalog load; failure prints the
  CLI JSON summary `{"success": false, "error": "renovation_architecture_init: ..."}`.
- `tools/artifact_writers.py` — `_write_renovation_architecture_shadow()` helper
  plus one call after output-path resolution and before
  `validate_publication_payload`. Never raises; current mode adds zero keys;
  shadow mode writes only `analysis_debug["renovation_architecture_shadow_v1"]`
  (stripped from slim `photo_intel.json`; lands in `photo_intel_debug.json`).

## The explicit four-item no_action decision

`dirty_or_grimy_window_screens`, `dated_or_older_windows`, `door_hardware_worn`,
`door_hardware_dated_style` — exactly the four catalog items lacking BOTH a
`cost` block and a `work_item_code` — route to zero-dollar `no_action` with
reason code `no_economics_approved_gap` (user-approved optional gaps). The two
door-hardware items claim `pricing_status: "inherited_from_split_parent"` while
carrying no real economics (known validator blind spot: `package_role` counts
as an economic field); they were intentionally NOT "fixed" in the catalog.

Terminal route precedence (load-bearing, in `resolve_terminal_route`):
quarantined trade → `excluded_quarantine`; `drop_if_generic` →
`excluded_generic`; `estimate.strategy == "inspect_only"` → `inspection`;
no cost + no code → `no_action`; else `work`. `dated_electrical_outlets_switches`
is in both quarantine and generic and must land in quarantine. Shipped
distribution over 128 items: 12 / 4 / 5 / 4 / 103.

## Decisions and deviations

- **Route distribution pinned in tests, not the builder.** The builder enforces
  structural completeness (every item exactly one route; present-but-unknown
  `estimate.strategy`/`unit_policy` fails the build); the exact 12/4/5/4/103
  counts, item IDs, and 70-item/117-relationship affinity counts are pinned in
  `tests/test_renovation_architecture_catalog.py`, so a catalog regeneration
  updates test expectations instead of breaking shadow-mode worker startup.
- **`audit_runner.py` left unwired** (third entrypoint). The seam is defensive:
  shadow mode with an uninitialized runtime produces a valid private `failed`
  envelope (`runtime_not_initialized`), never an exception. Revisit in Session 5.
- **Failed-envelope provenance nullability.** The catalog identity group
  (catalog_version, catalog_ontology_version, catalog_sha256,
  projection_fingerprint) AND `kind_ontology_selector` are nullable only in the
  `failed` state — a deviation from the plan's "quartet" wording: an
  uninitialized runtime honestly knows neither the catalog nor the selector.
  Scaffold/complete envelopes require all of them, pinned to exact values.
- **Hyphen/underscore ontology distinction is contract-level.** Provenance
  carries `catalog_ontology_version == "observation-kind-v2"` (catalog stamp)
  and `kind_ontology_selector == "observation_kind_v2"` (env selector) as
  separate validated fields; tests pin that each spelling is rejected in the
  other's position.
- **`worn_or_stained_flooring` stays a work route** with
  `pricing_mode="heuristic"`: the catalog has no cost block for it; missing
  cost and `mode=="heuristic"` are identical downstream (`tools/costing.py`
  heuristic fallback), and the projection records the merged fact. The five
  cost-without-code items (`soffit_or_porch_ceiling_failed`,
  `soffit_or_porch_ceiling_weathered`, `fence_broken_or_leaning`,
  `fence_weathered`, `bare_or_missing_finish_flooring`) use catalog `scope` as
  `action_code` with `action_source="catalog_scope"`; no codes or prices were
  invented.
- **Scaffold reason string is load-bearing.** The validator pins
  `session_1_not_implemented`. Session 2 changing the emitted envelope requires
  the paired validator/test change — intended friction.
- **Deferred contract fields** (additive with a `CONTRACTS_SCHEMA_VERSION`
  bump): Terra/Sol token telemetry, request fingerprints, prompt/model routing
  in provenance, unit-resolution audit trail on ObservedCondition.

## Commands run and results

All on 2026-08-14, all green:

```
.venv\Scripts\python.exe -m pytest tests\test_renovation_architecture_contracts.py tests\test_renovation_architecture_catalog.py tests\test_renovation_architecture_runtime.py tests\test_catalog_kind_v2.py tests\test_catalog_validation.py tests\test_kind_ontology_cutover.py tests\test_kind_ontology_guards.py tests\test_artifact_writers.py -q --basetemp=C:\tmp\rv_pytest_session1_focused_20260814
  -> 318 passed, 1 skipped, 7.31s

.venv\Scripts\python.exe -m pytest tests\test_renovation_estimate.py tests\test_renovation_estimate_v4.py tests\test_rehab_packages.py tests\test_product_quarantine.py tests\test_min_photo_evidence_gate.py -q --basetemp=C:\tmp\rv_pytest_session1_regression_20260814
  -> 414 passed, 2.18s

.venv\Scripts\python.exe -m pytest tests -q --basetemp=C:\tmp\rv_pytest_session1_full_20260814
  -> 1966 passed, 6 skipped, 20.35s   (pre-change baseline: 1839 passed, 6 skipped; +127 new tests)
```

`tests/test_min_photo_evidence_gate.py` (staged at session start) was not
touched and passes; Steven committed it separately mid-session (650a447).
All warnings in the runs are pre-existing (utcnow deprecation, google-genai).

## Risks / notes for the reviewer

- `write_photo_intel` reads the mode via `getattr(cfg, "RENOVATION_ARCHITECTURE_MODE", "current")`
  — a test passing a bare `SimpleNamespace()` silently gets current mode. This
  mirrors the existing `KIND_ONTOLOGY_VERSION` read in the same function.
- The projection fingerprint includes `catalog_sha256`, so it changes when the
  catalog file changes byte-wise even if the projected content is identical.
  Estimate IDs include both values by design.
- Shadow output durability: the publication gate runs before both writes, so a
  canonical payload that fails the gate also loses the shadow envelope. That is
  the intended Session 1 posture (shadow output is strictly subordinate).
- In shadow mode the envelope is written even when v4 itself failed
  (`renovation_estimate_v4: null`) — the seam runs after and independently of
  the v4 try/except, and never mutates it.

## Session 2 prerequisites

- Consume `ObservedCondition` / `EvidenceFacts` / `ConditionReview` /
  `ConditionDisposition` from `tools/renovation_architecture/contracts.py`.
  Build conditions from the product-filtered issue lane using the existing
  room-surrogate/estimate-unit identity (`tools/renovation_estimate.py`
  `_estimate_scope_key_for_issue`, `tools/estimate_units.py`). Do NOT revive
  the ~55-field `EstimateCandidate` as the new boundary.
- `validate_complete_result()` is the target gate; its invariants are already
  frozen and test-covered against synthetic payloads.
- Terra review contract: `supported` / `unsupported` / `cannot_assess` only;
  the closed field set enforces the layer boundary. Operational failures are
  failures, not `cannot_assess`.
- Token telemetry and the daily-budget guard belong to Session 2 per the
  migration plan and are not scaffolded here.
