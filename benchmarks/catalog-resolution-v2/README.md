# catalog-resolution-v2

Measures Pass 2d catalog resolution under `observation-kind-v2`: does a
three-kind observation retrieve the right candidates from the v2 catalog, and
does the resolver pick the right item?

Task 2 deliverable. Companion to `benchmarks/kind-ontology-v2`, which measures
the *classification* contract; this one measures everything downstream of it up
to a resolved catalog ID.

## What is being compared

| lane | catalog | retrieval |
|---|---|---|
| `v2` | `tools/issue_catalog_kind_v2.json` (130 items, 3 kinds) | strict exact-kind, production closure + production resolution step |
| `legacy` | `tools/issue_catalog.json` (107 items, 2 kinds) | v1 routing frozen in `legacy_baseline/routing_snapshot.py` |

The v2 lane calls the real production code —
`catalog_embeddings.make_candidate_provider` and
`scene_classifier_orchestrator.resolve_observation_against_catalog` — so the
benchmark measures the shipped path, not a copy of it.

Both lanes resolve with the **current** Pass 2d prompt. Only retrieval semantics
differ, so the comparison isolates catalog structure + routing rather than
confounding it with a prompt change.

The legacy lane exists to answer one question: *did splitting concepts and
filtering strictly cost us recall?* Its snapshot deliberately preserves the v1
bugs (exterior-only upgrade widening; unknown kind → empty `allowed_kinds` →
whole-catalog search), because those bugs are the baseline.

## Discipline

- **dev** (`cases_dev.json`) is for iteration. When a case misses, fix the
  catalog's `embed_text` / `support_any` through
  `tools/catalog_migrations/kind_v2_decisions.json`, regenerate with
  `scripts/migrate_catalog_kind_v2.py`, and rerun dev.
- **holdout** (`cases_holdout.json`) decides the gates. Run it once, at the end.
  Do not tune against it; record misses instead (the Task 1 precedent).
- Case files carry `status` and a sha256 fingerprint over the whole case list.
  `--gates` refuses to run against a `draft` slice. Any gold edit changes the
  fingerprint, so a report can always be traced to the exact cases it scored.
- Model configuration is file-driven (`models/*.json`) and never resolved from
  env; the same rule applies to `embeddings.json`.

## Coverage

- Every one of the 42 split successors is gold in at least one case.
- 20 paired groups exercise same-subject kind boundaries — cabinets, hard
  flooring, vinyl, siding, roofing, paint, vanity, vanity top, appliances,
  plumbing, bath fixtures, exhaust fan, ceiling fan, soffit, paving, fence,
  masonry, pool, landscaping, door hardware.
- `no_match` cases: real observation text with no catalog concept behind it.
- `invalid_kind` probes: a retired (`upgrade`) or unknown kind must fail closed
  **before** retrieval in the v2 lane. Under legacy this is the whole-catalog
  bug, so the probe scores the difference.
- `empty_filter` probes: an explicitly empty `allowed_kinds`, and an unknown
  one, must both return zero candidates — never the whole catalog.

## Metrics

candidate recall@5 (overall / per kind / per family), gold rank, final
resolved-ID accuracy, candidate-kind purity, no-match false positives,
split-family confusion, paired-group recall and accuracy, filter-probe
violations, and legacy-vs-v2 deltas over the comparable population.

## Holdout gates

| gate | threshold |
|---|---|
| `no_failures` | zero schema/provider failures |
| `kind_purity` | 100% |
| `recall_overall` | recall@5 ≥ 0.98 |
| `recall_per_kind` | recall@5 ≥ 0.95 for every kind |
| `recall_no_regression` | ≥ comparable legacy baseline |
| `final_accuracy_overall` | ≥ 0.95 |
| `final_accuracy_per_kind` | ≥ 0.90 for every kind |
| `paired_recall` | 100% |
| `paired_accuracy` | ≥ 0.95 |
| `filter_probes` | zero violations |

## Running

Needs the embeddings sidecar (`scripts/start-embeddings-server.ps1`). Retriever
construction embeds the catalog, so a dead sidecar fails loudly at startup
rather than silently returning zero candidates.

Dev, both lanes:

```bash
.venv/Scripts/python.exe scripts/benchmark_catalog_resolution.py --lane legacy --cases dev --model-config benchmarks/catalog-resolution-v2/models/terra.json --repeats 3
```

```bash
.venv/Scripts/python.exe scripts/benchmark_catalog_resolution.py --lane v2 --cases dev --model-config benchmarks/catalog-resolution-v2/models/terra.json --repeats 3
```

Gated holdout run (after freezing the cases), comparing against the legacy
holdout report:

```bash
.venv/Scripts/python.exe scripts/benchmark_catalog_resolution.py --lane v2 --cases holdout --model-config benchmarks/catalog-resolution-v2/models/terra.json --repeats 5 --compare-to benchmarks/catalog-resolution-v2/results/<legacy-holdout>/report.json --gates
```

Qwen transfer report (informational, ungated) — swap in `models/qwen.json`.

Reports land in `results/<stamp>_<lane>_<model>_<slice>/report.{json,md}`.

## Publication safety

Nothing here can publish. The v2 catalog carries
`publication_status: blocked_pending_pricing`, `write_photo_intel` rejects both
that and any `classification_only` result, and the benchmark never calls a
writer. Split successors have no cost, estimate, work-item, or package metadata
at all — resolution accuracy is being measured, not economics.
