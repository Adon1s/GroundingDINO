# HANDOFF — Kind ontology Task 3 → Task 4 (pricing + cutover)

2026-08-05. Task 3 landed on `pass_2c_redesign` (`0de9332`..`342dd95`, C1–C8). Suite:
**1681 passed / 6 skipped**. What each commit did is in its message; this document is
only what Task 4 needs.

## State you inherit

Every downstream consumer is three-kind-native. Nothing is published and nothing is
priced for the new kinds. The freeze is intact and **triple-gated**:

- `ISSUE_CATALOG_PATH` still points at `tools/issue_catalog.json` (v1);
- both `write_photo_intel` guards stand (`classification_only` payloads, and any catalog
  whose `publication_status` is not `publishable`);
- production `pipeline_mode` is `classification_only` and is deliberately not settable
  through `from_analysis_profile`, so production still stops after Pass 2c.

`tools/issue_catalog_kind_v2.json` (130 items, `blocked_pending_pricing`) is reachable
only from the generator, the validator, and the benchmark. Pass 2e is alive again but
runs only in `catalog_resolution_benchmark` mode (2c→2d→2e).

## How each kind behaves (author prices against this)

| Consumer | defect | degradation | modernization |
|---|---|---|---|
| `costing.KIND_MULT` | 1.0 | **raises** | **raises** |
| `estimate_scope` baseline | severity/structural/required-term gates → `required_rehab` | same gates; past them → `marketability_rehab` | marketability / optional value-add, never required |
| visible-required-condition | eligible | eligible | ineligible |
| `property_summary_pass` boost / cap | 0 / uncapped | 0 / uncapped | −1 / capped at 4 |
| `rehab_packages` driver lane | defect lane | defect lane | opportunity lane (corroboration-gated) |
| `catalog_display_class` | trade/tier ladder | trade/tier ladder | `marketability` |
| `catalog_cost_model` | — | — | ROOM_ALLOWANCE derivation |
| Pass 2e speculation gate | ungated | ungated | gated → `speculative_modernization` |
| `default_actionability` | repair | repair | modernization category |

Legacy `"upgrade"` still resolves (0.6 multiplier, branch sets, summary alias) so the v1
catalog keeps working until cutover. Pass 2e is the exception: it rejects `upgrade` as
`invalid_kind` because it only ever sees fresh 2c output.

## Task 4 checklist

1. **KIND_MULT values for degradation and modernization.** The tension is measured and
   unavoidable: degradation = **26 ex-defect items currently priced ×1.0 + 7 ex-upgrade
   ×0.6**; modernization = **6 ex-defect ×1.0 + 23 ex-upgrade ×0.6** (manual allowances
   exempt on top of these). No single per-kind value preserves both lineages — pick the
   values deliberately, or add a per-item mechanism. Drop the dead `upgrade` entry at
   cutover. Same call feeds `compute_scoring` (impact points, ranked issue order).
2. **Author pricing/estimate/package metadata for the 42 split successors**
   (`pricing_status: deferred_post_task3`). Edit
   `tools/catalog_migrations/kind_v2_decisions.json` and re-run
   `scripts/migrate_catalog_kind_v2.py` — never hand-edit the v2 catalog, manifest, or
   audit report (a parity test pins them). Successors currently have **no** cost,
   estimate, work-item, or package fields, so today they hit heuristic pricing,
   `LINE_ITEM`/`legacy_default`, and are invisible to package inference.
3. **Cutover**: flip `ISSUE_CATALOG_PATH`, set `publication_status: publishable`, remove
   both writer guards, expose `pipeline_mode` through `from_analysis_profile`.
4. **Re-resolve historical artifacts** per the manifest's `requires_re_resolution` flags
   (71 of 107 legacy ids). It is an audit record, not a runtime alias table — v1
   artifacts are `legacy_v1` and must never be reinterpreted in place.
5. **Product review of display shifts** that only appear at cutover: the 7
   ex-upgrade→degradation items gain +1 display severity (KIND_BOOST 0, no cap), and
   degradation buckets rank like upgrade buckets did (the bucket rank boost still counts
   only defects).
6. **Frontend alignment**: `summary_v1` is now version **2.0** with per-kind
   `kind_counts` replacing `defect_count`/`upgrade_count` at bucket and listing level;
   plus the new `speculative_modernization` suppression reason and the modernization
   display-class change.
7. **Catalog content questions carried from Task 2**: `trees_or_vegetation_too_close`
   (likely not worth costing), `staging_or_decluttering_opportunity` vs
   `indoor_storage_clutter_heavy`, `visible_mold_or_mildew` vs
   `mold_or_mildew_visible_bathroom` (both overlaps inherited from v1, not introduced).
8. **2d model routing**: Qwen matched Terra exactly on 2d in Task 2's holdout (0 schema
   failures) — decide whether 2d needs Terra.
9. **Blocked harnesses** — migrate or delete; all still `RuntimeError` at `main()` and
   deliberately untouched by Task 3. Two-kind internals: `catalog_auditor`
   `MATCH_THRESHOLD_DEFECT/UPGRADE` (`:107-108`) with the `:289-290` ternary handing every
   non-defect kind the upgrade threshold, plus `_label_to_kind` (`:282`) and two
   duplicated routing call blocks; `model_comparison` `_label_to_kind` (`:259`),
   forward-set filters (`:410-414`, `:1060`), judge prompts (`:770-832`), kind default
   (`:1466`); `bias_check` (`:428`) has no kind logic of its own.

## Live hazard before you start

`tools/reproject_product_views.py` recomputes `compute_scoring` **and**
`build_property_summary_v1` over historical artifacts with current code, and writes
`summary_v1` back. Since C4 it emits the v2.0 `kind_counts` shape while the frontend
still reads `defect_count`/`upgrade_count`. **The pending electrical-quarantine legacy
reprojection must run from a pre-C4 checkout or wait for frontend support.** Scoring
itself is safe (v1 artifacts carry v1 kinds).

## Gotchas

- **Unknown kinds fail loud in three places** — `costing.kind_multiplier`,
  `estimate_scope` (non-empty unknown kind only; empty keeps the v1 text fallthrough for
  kindless synthetic candidates), and package driver dispatch. Adding a kind means
  extending `_DEFECT_DRIVER_KINDS`/`_OPPORTUNITY_DRIVER_KINDS` in `rehab_packages.py`
  deliberately. Artifact-*reading* paths stay tolerant on purpose:
  `property_summary_pass._norm_kind` warns and defaults to defect rather than nulling a
  whole summary.
- **A full-chain v2 estimate dry-run is impossible until step 1 is done.** Costing raises
  on the new kinds by design; that is the verified behavior, not a gap to work around.
- **`package_affinity_for` is 3-arg now** — build the table with
  `build_package_affinity(catalog)`. The old default silently pinned v1 ids for the
  process lifetime.
- **`snapshot()` seals `catalog_kinds` off the catalog's own `ontology_version`.** Do not
  reduce it to a constant; a sealed benchmark snapshot must describe the catalog it froze,
  and sealing is terminal per dataset (no re-seal path).
- **Name collision**: "modernization" is both an observation kind and a package category.
  Distinct vocabularies that happen to map 1:1; degradation has no category namesake.
- Frozen-on-purpose v1 strings that look like debt:
  `COST_MODEL_SOURCE_DERIVED_UPGRADE_ROOM_ALLOWANCE` (artifact provenance),
  `defect_upgrade_disagreements` (report key, kind-agnostic logic),
  `benchmarks/catalog-resolution-v2/legacy_baseline/` (preserves v1 bugs as the baseline).
- Task 1's ontology is fixed. If a Task 4 decision seems to need reinterpreting a kind,
  that is a blocking contradiction to raise, not to patch.

## Verify

```bash
.venv/Scripts/python.exe -m pytest tests -q --basetemp=C:/Users/Steven/AppData/Local/Temp/claude/pytest
```

Remaining skips are all pre-existing: 4 × `blocked_live_2c`, 1 embeddings integration
(`RUN_EMBEDDINGS_INTEGRATION=1`), 1 runtime skip in `test_catalog_validation.py`.
