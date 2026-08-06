# HANDOFF — Kind ontology Task 3: downstream consumers go three-kind

2026-08-05. Task 3 of 3 in the `defect | upgrade` → `defect | degradation | modernization`
rework. Builds on [HANDOFF_kind_ontology_task2.md](HANDOFF_kind_ontology_task2.md); this
document is the contract the pricing/cutover phase ("part 4") builds on.

## Status

- **Done and verified**: every runtime consumer in the Task 2 inventory handles three
  kinds — costing (fail-loud), estimate_scope, property_summary_pass, rehab_packages,
  catalog_cost_model, Pass 2e, the orchestrator's 2d→2e chain, and the benchmarking
  vocabulary/index. The dead two-kind code (B5, B10, B12, shadow-lane config) is deleted,
  not migrated.
- **Deliberately unchanged**: `ISSUE_CATALOG_PATH` still points at the v1 catalog; both
  `write_photo_intel` guards stand; production `pipeline_mode` is still
  `classification_only` and cannot be changed via `from_analysis_profile`. **Task 3 makes
  no pricing change and no cutover.**
- **Still frozen**: no active-listing analysis may run. Production stops after Pass 2c.
- **Not done (part 4)**: multiplier values, the 42 successor prices, the catalog flip,
  and the blocked harnesses (deliberately deferred — see below).

## What shipped

| Commit | Contents |
|---|---|
| `0de9332` | C1 — costing fails loud on unpriced kinds (`kind_multiplier`) |
| `05ff16e` | C2 — estimate_scope label-leak fix (kind string out of matched text) |
| `252ae12` | C3 — three-kind scope routing branches |
| `d128610` | C4 — property_summary_pass three-kind, summary_v1 v2.0 (`kind_counts`) |
| `0a23736` | C5 — package driver lanes, required affinity table (W1), cost model |
| `f1cf1e6` | C6 — Pass 2e revived; benchmark mode runs 2c→2d→2e |
| `5dacb78` | C7 — benchmarking vocabulary versioned sealing + annotator index |
| (this)    | C8 — dead-code deletions, cosmetics, this handoff |

### The semantic mapping (Steven's decisions, applied)

| Consumer | defect | degradation | modernization |
|---|---|---|---|
| `costing.KIND_MULT` | 1.0 | **raises** (part 4) | **raises** (part 4) |
| `estimate_scope` baseline | defect gates (severity/structural/required-terms → required) | same gates; past them → `marketability_rehab` (`degradation_default`) | old upgrade branch (marketability / optional value-add) |
| `_is_visible_required_condition` | eligible | eligible | ineligible |
| `property_summary_pass` KIND_BOOST / cap | 0 / uncapped | 0 / uncapped | −1 / capped at 4 (`MAX_MODERNIZATION_SEVERITY`) |
| `rehab_packages` driver lane | defect lane | defect lane (concrete deterioration evidence) | opportunity lane (corroboration-gated) |
| `catalog_display_class` | trade/tier ladder | trade/tier ladder | `marketability` |
| `catalog_cost_model` | — | — | ROOM_ALLOWANCE derivation (like upgrade) |
| Pass 2e speculation gate | ungated | ungated (evidence-anchored; 2c handles speculation upstream) | gated → `speculative_modernization` |
| `default_actionability` (benchmark) | repair | repair (you pay to fix deterioration) | modernization category |

Legacy `"upgrade"` remains valid only where v1 inputs still flow: KIND_MULT (0.6), the
estimate_scope/`rehab_packages`/`catalog_cost_model` branch sets (`in ("upgrade",
"modernization")`), and `property_summary_pass._norm_kind` (aliases it to modernization).
Pass 2e rejects it (`invalid_kind:upgrade`) — 2e only ever consumes fresh 2c output.

### Fail-loud vs tolerant, the boundary

Catalog-driven runtime paths fail loud on unknown kinds:
- `costing.kind_multiplier` raises `CatalogDataError` naming the phase + item
  (manual allowances remain exempt; the two `compute_scoring` writer call sites are
  already wrapped in try/except, so the frozen v1 path cannot be broken by the raise);
- `estimate_scope` raises `ValueError` on a *non-empty* unknown kind (empty keeps the v1
  text fallthrough for kindless synthetic candidates — `:181` coerces a *missing* kind to
  defect before the tail);
- `rehab_packages` raises on a driver-role item with an unroutable kind (a driver
  silently vanishing from both lanes is the exact bug being fixed).

Artifact-reading paths stay tolerant: `property_summary_pass._norm_kind` warns and
defaults junk to defect (a raise would nuke the whole summary for one bad kind), and
`quant_artifact_comparison` still passes historical kinds through untouched.

### Pass 2e / orchestrator (the cutover blocker, closed)

`run_pass_2e` validates kinds against `OBSERVATION_KINDS`. The orchestrator's
unconditional post-2d `return` is gone: `catalog_resolution_benchmark` mode now runs
2c→2d→2e, with the 2d gate falling through so 2e runs even when 2d is off or resolves
nothing (v1 semantics — the fail-closed tests depend on it). The 2e input block iterates
`result.observations` joined with `result.resolved_items` by `issue_id`; the
`_label_to_kind` fallbacks and `labeled_forward` reads are deleted (their input contract
no longer exists). The disabled-2e path promotes observations verbatim. **No third
pipeline mode was added** — 2e is deterministic (no LLM), `PassToggles.pass_2e` already
gates per-run, and the resolution benchmark calls `resolve_observation_against_catalog`
directly so it is unaffected either way.

`_2e_dedupe_key` folding kind is now load-bearing: paired split observations (same
photo/subject, different kinds) must never collapse. Pinned in
`tests/test_renovation_estimate.py::TestPass2eThreeKindSanity`.

### summary_v1 v2.0 (published shape change)

`defect_count`/`upgrade_count` are replaced by `kind_counts` per kind at bucket and
listing level; `version` is `"2.0"` so the frontend can key its cutover. Mixed-kind
blocks keep the most condition-like kind (`defect < degradation < modernization` —
identical to v1's defect-wins for two kinds). Bucket rank still boosts only
`kind_counts["defect"]`. Production is frozen, so no live artifact changes shape until
part 4 — **but see the reprojection flag below.**

### Deletions

- `is_defect_driver` / `is_dated_cosmetic_evidence` (zero runtime callers, test-only).
- `_default_package_affinity` + module memo — `package_affinity_for` now **requires** the
  table; the memoized default silently pinned the shipped v1 catalog's ids for the
  process lifetime (W1).
- `embeddings_retrieve_defect_candidates` / `embeddings_retrieve_upgrade_candidates`
  (zero production callers; tests converted to `retrieve_candidates`).
- The pre-v3 split-format ("defects"/"upgrades") merge in `load_issue_catalog`; the
  load-log now counts a missing kind as `"?"`, never as defect.
- `SHADOW_LANE_*` config (the lane itself was deleted in Task 2 C4).
- `artifact_viewer.build_catalog_from_issue_catalog` now reads `items` (it read pre-v3
  keys and always returned an empty catalog).

## Blocked harnesses — inventoried, deliberately NOT migrated

All three stay `RuntimeError`-blocked at `main()` and were left untouched (Steven's
call: no payoff until unblocked). Their two-kind internals, for whoever migrates or
deletes them:

- `tools/catalog_auditor.py` — `MATCH_THRESHOLD_DEFECT`/`MATCH_THRESHOLD_UPGRADE`
  (`:107-108`) with the else-ternary at `:289-290` giving any non-defect kind the
  upgrade threshold; private `_label_to_kind` (`:282-286`); duplicated routing call
  blocks (`:312-314`, `:421-434`).
- `tools/model_comparison.py` — `_label_to_kind` (`:259-260`); forward-set filters
  (`:410-414`, `:1060-1065`); judge prompts speaking v1 labels (`:770-832`); kind
  default at `:1466`.
- `tools/bias_check.py` — blocked at `:428`; no kind logic of its own beyond the above
  imports.
- Related naming debt left on purpose: `quant_artifact_comparison`'s
  `defect_upgrade_disagreements` report key (logic is kind-agnostic; renaming breaks
  diffs against historical reports), `COST_MODEL_SOURCE_DERIVED_UPGRADE_ROOM_ALLOWANCE`
  (frozen v1 artifact provenance string), `audit_pass2c_funnel.FORWARDED_LABELS`
  (deliberate mirror with its own rationale comment).

## Part 4 checklist (pricing + cutover)

1. **KIND_MULT values** for degradation and modernization (`tools/costing.py`) — the
   structural tension is measured: degradation = 26 ex-defect items priced ×1.0 + 7
   ex-upgrade ×0.6; modernization = 6 ex-defect ×1.0 + 23 ex-upgrade ×0.6 (manual
   allowances exempt). No single per-kind value preserves both lineages. Drop the dead
   `upgrade` entry at cutover.
2. **Author pricing/estimate/package metadata for the 42 split successors**
   (`pricing_status: deferred_post_task3`) via `kind_v2_decisions.json` + the generator.
3. **Cutover**: flip `ISSUE_CATALOG_PATH`, set the v2 catalog's `publication_status` to
   publishable, remove both `write_photo_intel` guards, expose `pipeline_mode` through
   `from_analysis_profile`. Historical artifacts re-resolve per the manifest's
   `requires_re_resolution` flags (audit record, not a lookup table — never alias).
4. **Product review** (display changes that only manifest at cutover): KIND_BOOST
   degradation=0 (7 ex-upgrade→degradation items gain +1 display severity vs v1),
   defect-only bucket rank boost (degradation buckets rank like upgrade buckets did),
   modernization cap unchanged at 4.
5. **Frontend alignment**: summary_v1 v2.0 `kind_counts`, `speculative_modernization`
   suppression reason, display-class shifts.
6. **Catalog content flags carried from Task 2**: `trees_or_vegetation_too_close`
   (likely not worth costing), `staging_or_decluttering_opportunity` vs
   `indoor_storage_clutter_heavy` overlap, `visible_mold_or_mildew` vs
   `mold_or_mildew_visible_bathroom` overlap.
7. **2d model routing**: Qwen matched Terra exactly on the 2d contract in Task 2's
   holdout — decide whether 2d needs Terra at all.
8. Blocked harnesses (above): migrate or delete.

## Wiring hazard for the pending legacy-reprojection rollout

`tools/reproject_product_views.py` (a consumer the Task 2 inventory missed) recomputes
`compute_scoring` **and** `build_property_summary_v1` over historical artifacts with the
current code and catalog, writing `summary_v1` back (`:258`, `:263`, `:339`, `:371`).
After C4, any reprojection run emits the v2.0 summary shape (`kind_counts`) while the
frontend still reads `defect_count`/`upgrade_count`. The pending electrical-quarantine
legacy reprojection should either run from a pre-C4 checkout or wait for frontend
`kind_counts` support. (Scoring is safe: v1 artifacts carry v1 kinds, and the writer
wraps `compute_scoring` in try/except besides.)

## Tests

**Full suite: 1681 passed, 6 skipped** (Task 2 ended at 1623 / 13; all 7 `dormant_2e`
skips are revived, net +58 passing).

```bash
.venv/Scripts/python.exe -m pytest tests -q --basetemp=C:/Users/Steven/AppData/Local/Temp/claude/pytest
```

Remaining skips: 4 × `blocked_live_2c` (`model_comparison` live cells), 1 embeddings
integration (`RUN_EMBEDDINGS_INTEGRATION=1`), 1 pre-existing runtime skip in
`test_catalog_validation.py`.

New/rewritten coverage: `test_scoring.py::TestKindMultiplierFailLoud`,
`test_estimate_scope.py` (new file: label-leak pin + full branch table, v1 reason
strings byte-identical), `test_property_summary_pass.py` (v2.0 shape + three-kind
ontology class), `test_rehab_packages.py::TestThreeKindDriverLanes`,
`test_renovation_estimate.py::TestPass2eThreeKindSanity`,
`test_pass_2e_fail_closed.py` (all revived, benchmark-mode options),
`test_candidate_provider.py::test_benchmark_mode_runs_deterministic_2e` (inverted pin),
`test_benchmark_vocabulary.py` (sealed-snapshot invariance: a v1 snapshot still records
exactly `["defect","upgrade"]`), `test_benchmark_catalog_index.py` (column alignment,
v2 kind filters).

## Gotchas

- **The v1 parity contract**: every v1-reachable `estimate_scope_reason` string is
  byte-identical; `optional_modernization` is a shared literal for both discretionary
  kinds (it was upgrade's v1 tier-optional-with-value-add reason).
- **`package_affinity_for` is now 3-arg** — any new caller must build the table with
  `build_package_affinity(catalog)`; there is no default.
- **A driver-role item with an unknown kind raises** in package inference. If part 4
  adds a kind, the driver lane sets (`_DEFECT_DRIVER_KINDS` / `_OPPORTUNITY_DRIVER_KINDS`
  in `rehab_packages.py`) must be extended deliberately.
- **`snapshot()` seals `catalog_kinds` off the catalog's own `ontology_version` stamp.**
  Do not "simplify" it to a constant — a snapshot must describe the catalog it froze.
- **Kind-vs-category name collision**: "modernization" is both an observation kind and a
  package category. `default_actionability` maps the kind to the category identically,
  but they are distinct vocabularies — degradation has no category namesake.
- Task 1's ontology remains fixed. If a part-4 decision seems to require reinterpreting
  a kind, that is a blocking contradiction to raise, not to patch.
