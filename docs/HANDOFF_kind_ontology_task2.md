# HANDOFF — Kind ontology Task 2: catalog + Pass 2d migration

2026-08-05. Task 2 of 3 in the `defect | upgrade` → `defect | degradation | modernization`
rework. Builds on [HANDOFF_kind_ontology_task1.md](HANDOFF_kind_ontology_task1.md); this
document is the contract Task 3 (downstream consumer migration + cutover) builds on.

## Status

- **Done and verified**: the catalog is migrated to the three-kind ontology as a *separate,
  non-publishable* file; Pass 2d retrieval and resolution are strict exact-kind; the
  whole-catalog-search bug is closed; a catalog-resolution benchmark exists with frozen
  reviewed cases and gates.
- **Deliberately unchanged**: `tools/issue_catalog.json`, `ISSUE_CATALOG_PATH`, and every
  economic field on every unsplit item. **Task 2 makes no pricing or product change.**
- **Still frozen**: no active-listing analysis may run. Production stops after Pass 2c; both
  `write_photo_intel` guards remain. Pass 2e is dormant.
- **Not migrated (Task 3)**: every consumer in the inventory below.

## What shipped

| Commit | Contents |
|---|---|
| `0098d7f` | Housekeeping: Task 1 deliverables tracked (docs/, benchmarks/, `tools/__init__.py`) |
| `8bcd6f4` | C1 — shared ontology module `tools/observation_kinds.py` |
| `b6c7560` | C2 — strict exact-kind route + retrieval filter-semantics fix |
| `ccb6b08` | C3 — v2 catalog, migration manifest, audit report, versioned validator |
| `c11e9eb` | C4 — `pipeline_mode` + strict Pass 2d path, dormant v1 block deleted |
| `893b9db` | C5 — writer guard rejects non-publishable catalogs |
| `81777fa` | C6 — catalog-resolution-v2 benchmark harness + cases |
| `a2cc9d9` | Pause B — cases frozen, staging/clutter overlap flagged |

### The ontology enum now has one home

`tools/observation_kinds.py` — `ONTOLOGY_VERSION`, `OBSERVATION_KINDS`, `EXCLUSION_REASONS`,
`LEGACY_CATALOG_KINDS`. Dependency-free; `scene_classifier_passes` re-exports it so every
existing importer is unchanged. Identity (not equality) is asserted in
`tests/test_kind_ontology_guards.py`, so the two names cannot drift.

### Retrieval is strict exact-kind

`evaluate_kind_routing` keeps its name and signature (the blocked harnesses import it at
module level) but is now a singleton route: a valid kind returns `(kind,)` with
`reason="exact_kind"`; anything else returns `()` with `reason="invalid_kind"`. The
asymmetric upgrade→defect component-term widening is gone, along with the widening prompt
addendum and the `kind_routing` plumbing through `run_pass_2d`.

**The whole-catalog-search bug is closed** (`catalog_embeddings.py`):
`retrieve_candidates` now tests `if allowed_kinds is not None:`, so an *empty* set fails
closed to `[]` instead of falling through as "no filter". The provider closure is extracted
as `make_candidate_provider(retriever)` and decides kind membership from the index rather
than a hardcoded `{"defect","upgrade"}` literal — the same closure therefore serves any
catalog. `_build_items` no longer coerces a missing kind to `"defect"`.

Pass 2d gained prompt provenance: `PASS_2D_PROMPT_VERSION = "pass_2d_exact_kind_v2"`,
SHA256 `c779b2ac5b43ba703020f40ae17b9778b53af96198d0cd50528c89380ff7a670`.

### The catalog

`tools/issue_catalog_kind_v2.json` — version `3.0`, `ontology_version: observation-kind-v2`,
`publication_status: blocked_pending_pricing`. **130 items: 52 defect / 36 degradation /
42 modernization**, from 107 legacy items (67 defect / 40 upgrade).

Generated deterministically (byte-stable, pinned by a parity test) by
`scripts/migrate_catalog_kind_v2.py` from `tools/catalog_migrations/kind_v2_decisions.json`
— the single reviewable authoring source. Dispositions: **32 unchanged, 52 reclassified,
4 narrowed, 19 split into 42 successors**.

Inheritance is enforced in code, not by authoring discipline:
- unchanged / reclassified / narrowed items copy the legacy item verbatim, apply the new
  kind + `atomic_claim` + optional wording overrides, and **keep every economic field
  byte-identical**;
- split successors get new evidence-based ids, inherit only structural fields, are stamped
  `pricing_status: deferred_post_task3`, and the generator *strips* every economic field
  (`cost`, `estimate`, `work_item_code`, `cost_model`, `package_affinity`, `package_role`,
  `estimate_scope`, `estimate_scope_reason`). A successor cannot accidentally inherit a price.

`tools/catalog_migrations/2.1_to_3.0.json` is the manifest: one entry per legacy id (all
107), recording successors + kinds, change type, deprecation, re-resolution requirement,
atomicity rationale, and expected routing/scope/pricing/package effects. It carries
`audit_only: true` and **is not a runtime alias table** — nothing imports it outside
validation and the benchmark. No analysis runs until Task 3, so nothing needs legacy-id
resolution at runtime; historical artifacts are re-resolved at cutover, never aliased.

`tools/catalog_migrations/2.1_to_3.0_audit.md` is the generated audit report (dispositions,
splits, kind counts, deferred-pricing successors, flagged items, unresolved effects).

### Validation

`tools/catalog_validation.py` versions its kind vocabulary off the catalog's own root
metadata: a catalog stamped `observation-kind-v2` validates against the three kinds,
anything else against the legacy two. The shipped v1 catalog and its warnings snapshot are
untouched and still pass. v2-only rules cover root metadata, `atomic_claim` completeness,
and the deferred-pricing restriction. New `validate_migration_manifest()` checks all 107
entries, successor existence + kind agreement, orphan v2 items, split-parent deprecation,
and re-resolution consistency.

### Pipeline mode

`SceneClassifierRunOptions.pipeline_mode` — `classification_only` (default) or
`catalog_resolution_benchmark`. **Deliberately not settable via `from_analysis_profile`**,
which is how every production entry point builds options, so production cannot reach Pass
2d. Both modes set `classification_only=True` on the result, and Pass 2e never runs in
either.

The dormant v1 block was **deleted**, not left to rot: the shadow lane, `_label_to_kind`,
the unknown-kind→`defect` coercion, the two-kind 2d gate, and the
`resolved_defects`/`resolved_upgrades` summary are gone, along with the orphaned shadow
helpers in `scene_classifier_passes`. Net −150 lines.

`resolve_observation_against_catalog()` (module-level in the orchestrator) holds the whole
resolution step so the benchmark measures the production path rather than a copy. It fails
closed twice:
- an out-of-ontology kind raises `PassExecutionError('2d','parse', code="invalid_kind")`
  **before any retrieval** — the empty-`allowed_kinds` input is exactly what used to search
  the whole catalog;
- an off-kind candidate raises `code="kind_purity_violation"` rather than being quietly
  reranked. Exact-kind routing means the pool is pure; a leak is a broken filter, and must
  be loud.

### Publication guards (Task 3 must remove neither)

1. `write_photo_intel` refuses any `classification_only` payload (Task 1).
2. `write_photo_intel` refuses any catalog whose `publication_status` is not `"publishable"`
   (Task 2). Absent status = publishable, so the v1 catalog is unaffected;
   `load_issue_catalog` now passes `publication_status` and `ontology_version` through its
   normalization, which previously dropped all unknown root keys.

Pinned end-to-end: the real v2 catalog, loaded the way production loads a catalog, is
rejected by the writer.

## Fingerprints

| Artifact | SHA256 |
|---|---|
| `tools/issue_catalog.json` (unchanged) | `fec54467bc839780ec141ab09265ef01f193e78e30b76dea3a0989fd421731cb` |
| `tools/issue_catalog_kind_v2.json` | `1198b33f9c8f4e07634c6dd05c0d6f6dc64ecc11c8346fc9910d75d81e6c30f4` |
| `tools/catalog_migrations/2.1_to_3.0.json` | `1f5088265ee93b649aafd997f1d1ae12554cee27aba95ad5bc3d3664cf938572` |
| `tools/catalog_migrations/kind_v2_decisions.json` | `7e51720be3dd99e970034d7bfd257591be5753f6339eb2369aeaccf485256267` |
| `benchmarks/catalog-resolution-v2/embeddings.json` | `9d5523f95cb4921ae86224ee7b15432bf4fdf94b6048eae0ea1c818da114c3f2` |
| `legacy_baseline/routing_snapshot.py` | `fcbf0b8150803c40ca58cc3ecf480fda0b3445078bdea4f9b6c6371c081db0b2` |
| cases_dev (62, frozen-2026-08-05) | `d355313936b87c34f341c82f6ff249c30c95cf0d91160a7390675bd4f2822667` |
| cases_holdout (59, frozen-2026-08-05) | `1a8b99145881bcaf99aca48d818868848c9dd27907859c650ed43ee24dc2910e` |
| Pass 2d prompt `pass_2d_exact_kind_v2` | `c779b2ac5b43ba703020f40ae17b9778b53af96198d0cd50528c89380ff7a670` |

## Benchmark: catalog-resolution-v2

All numbers are real runs on GPT-5.6 Terra against the frozen case slices, embeddings
sidecar live (jina v5 Q8_0, 1024-dim). Protocol in
`benchmarks/catalog-resolution-v2/README.md`.

Two lanes over the same cases and the same encoder. The **v2 lane calls the production
code** (`make_candidate_provider` + `resolve_observation_against_catalog`), so this measures
the shipped path. The **legacy lane** replays v1 retrieval from a frozen code snapshot,
preserving the exterior-only widening and the unknown-kind whole-catalog search — those bugs
*are* the baseline. Both lanes resolve with the current 2d prompt, so the comparison
isolates catalog structure + retrieval rather than confounding it with a prompt change.

### Holdout (59 cases, 5 repeats) — ALL 10 GATES PASS

| metric | legacy | **v2** |
|---|---|---|
| candidate recall@5 | 0.870 (defect .932 / upgrade .600) | **1.000** (defect 1.000 / degradation 1.000 / modernization 1.000) |
| final resolved-ID accuracy | 0.870 | **1.000** |
| candidate-kind purity | 1.000 | **1.000** |
| paired-group recall / accuracy | 0.833 / 0.833 | **1.000 / 1.000** |
| gold rank top-1 | 0.915 | **0.963** |
| no-match false positives | 0.000 | **0.000** |
| filter-probe violations | **15 of 15** | **0 of 15** |
| split-family confusions | 20 | **0** |
| schema/provider failures | 0 | **0** |

Comparable population: recall **+0.130**, accuracy **+0.130**. Zero misses in the v2 lane
across all 5 repeats.

### Dev (62 cases, 3 repeats)

| metric | legacy | v2 |
|---|---|---|
| candidate recall@5 | 0.897 | 1.000 |
| final accuracy | 0.862 | 0.983 |
| paired recall / accuracy | 0.828 / 0.793 | 1.000 / 1.000 |
| filter-probe violations | 6 of 6 | 0 of 6 |
| split-family confusions | 21 | 3 |

### What these numbers do and do not say

- **Strict exact-kind filtering costs no recall — it gains it.** That was the main risk in
  the plan (the older scope note suggested a degradation fallback lane "until corpus accuracy
  proves strict filtering safe"). It is now measured: v2 beats legacy on every axis, on both
  slices, so no fallback lane was built.
- **The paired-group jump (0.833 → 1.000) is the split evidence.** Those are exactly the
  same-subject defect/degradation/modernization boundaries that v1 collapsed into one item
  and one price.
- **The probe columns quantify the retired bug.** Legacy violated every single filter probe
  (15/15, 6/6): an unknown kind produced an empty `allowed_kinds`, which its retriever read
  as "no filter" and answered with a whole-catalog search. v2 violates none.
- **Holdout modernization recall reads 1.000 partly because `res-mod-007` was dropped** at
  Pause B (see below). Before dropping it, retrieval-only measurement put holdout
  modernization at 0.909. The drop was a case-authoring decision about a genuine v1 catalog
  overlap, not a tuning move against the gate — recorded here so the 1.000 is not over-read.
- **Legacy's `upgrade` row is small** (recall 0.600 on holdout): the v1 catalog simply had
  fewer reachable upgrade concepts, which is the recall hole
  `docs/HANDOFF_catalog_kind_consistency.md` first measured.

### Qwen 3.6 27B transfer (holdout, v2 lane, informational — not gated)

Recall@5 **1.000**, final accuracy **1.000**, purity 1.000, paired 1.000/1.000, 0 no-match
false positives, 0 probe violations, 0 split-family confusions, **0 schema failures**.
Identical to Terra on every metric.

This is a meaningful contrast with Task 1, where Qwen produced schema failures on ~4.7% of
2c decisions under the fail-closed contract and was judged not production-grade for
classification. Pass 2d is a far simpler contract — pick zero or one id from a supplied list
— and Qwen handles it cleanly. **Task 3 signal, recorded not decided:** the model-routing
problem is specific to 2c; 2d does not appear to need Terra.

### Reports

`benchmarks/catalog-resolution-v2/results/` — `dev_legacy_terra`, `dev_v2_terra`,
`holdout_legacy_terra`, `holdout_v2_terra` (gated, with `comparison` block), and
`holdout_v2_qwen` (transfer, ungated).

## Steven's review decisions

**Pause A (catalog dispositions):**
- `tree_stump_present_in_yard` — a stump is *not* a trip hazard. The proposed
  defect/modernization split was dropped; it stays one generic discretionary observation
  (defect → modernization), economics untouched.
- `trees_or_vegetation_too_close` — a common model hallucination and probably not worth
  costing. Classification is correct under the ontology (overgrowth = degradation), so the
  kind changed and the economics did not. Recorded as a `task3_flags` entry rather than
  acted on: Task 2 does not make pricing decisions, and the grounding half belongs to
  2a/2b evidence work.
- All other dispositions approved as generated.

**Pause B (benchmark cases):**
- `res-mod-007` dropped: `staging_or_decluttering_opportunity` and
  `indoor_storage_clutter_heavy` are not separable from observation text. The overlap is
  inherited from v1, not introduced by this migration, so both items were left untouched and
  the ambiguous case removed rather than inventing a boundary. Flagged for Task 3 on both
  entries.
- Gate on Terra, with a Qwen transfer report (informational).

**Second overlap found during the dev run** (recorded the same way, not acted on):
`visible_mold_or_mildew` vs `mold_or_mildew_visible_bathroom`. Dev case `res-moi-002`
("black fuzzy growth… near the tub", bathroom scene) resolved to the bathroom-specific item
over the general one — arguably the better answer, but the two are not distinguishable by
rule. Also inherited from v1. The dev case was left as-is rather than re-frozen; it is the
only dev miss and dev is not gated.

**Standing rule recorded:** if Pass 2f ever enters scope, it must run on Sol. Task 2 never
invokes 2f.

## Deferred-pricing successors (42)

Every split successor carries `pricing_status: deferred_post_task3` and **no** cost,
estimate, work-item, or package metadata. Authoring them is the follow-up *after* Task 3;
publication stays blocked until it is done. Full list in
`tools/catalog_migrations/2.1_to_3.0_audit.md` under "Deferred-pricing successors".

The economic weight of this is real and was measured before the migration: under v1,
defects were 67% priced and upgrades 20%, so a kind flip without an estimate block is a
silent price deletion. That is exactly why unsplit items kept their fields byte-identical
and split successors were left unpriced rather than inheriting a parent price.

## Task 3: every consumer still assuming two kinds

Verified against the working tree at `a2cc9d9`, not against older docs.

**The number that sizes the problem:** no v2 item carries the string `upgrade`. Every
`== "upgrade"` predicate below is dead-always-false; every `== "defect"` predicate silently
loses 35 items.

| legacy → v2 | count |
|---|---|
| defect → defect | 52 |
| defect → degradation | 29 |
| defect → modernization | 6 |
| upgrade → modernization | 36 |
| upgrade → degradation | 7 |

### (A) Silently changes pricing or product output

| # | Location | What happens with a three-kind value |
|---|---|---|
| A1 | `costing.py:48-51` `KIND_MULT` (+ call sites `:152`, `:279`) | `.get("modernization", 1.0)` → **1.0 instead of 0.6, i.e. +66.7% on every ex-upgrade line item**, silently. Manual allowances are exempt, so ~30 of the 43 ex-upgrade successors are exposed. Money path: `renovation_estimate.py:796-803` / `:818-825`. **Highest-value single finding.** |
| A2 | `costing.py:374`, `:385` | Same silent default in `compute_scoring` → ex-upgrade issues gain 67% impact points, reordering the published ranked issue list (`artifact_writers.py:819-820`). |
| A3 | `estimate_scope.py:571-576` + `:83-87` | `_catalog_text` concatenates the **kind string** into the matched text, and `_VALUE_ADD_TERMS` contains the literal `"modernization"`. Every modernization item therefore sets `has_value_add=True` **from its own label**, flipping scope routing at `:195` and `:220`. Fix this one-line field list *before* rewriting any branch, or the rewrite gets tuned against poisoned text. |
| A4 | `estimate_scope.py:198-224` | `kind == "upgrade"` never fires; `kind == "defect"` loses 35 items. A severity-4 degradation skips the severity/structural/required-term checks entirely and lands on `marketability_signal`. |
| A5 | `estimate_scope.py:534-536` | `_is_visible_required_condition` returns False for non-defect kinds → visible damage gets downgraded into `inspection_risk`. |
| A6 | `rehab_packages.py:1213-1226` (consumed `:2100-2105`) | A `package_driver` whose kind is degradation/modernization matches **no branch** — dropped from drivers *and* supports, so packages silently fail to form. 6 driver entries in the v2 affinity table vanish. |
| A7 | `rehab_packages.py:591-592` | `kind == "upgrade"` → `DISPLAY_CLASS_MARKETABILITY` is dead; 41 v2 items without an explicit `display_class` can fall through to `HIGH_CONCERN`, misfiling them in `ui_priorities_v1`. |
| A8 | `property_summary_pass.py:44`, `:75-77` | Private two-kind `VALID_KINDS` + `_norm_kind` coerce **all 78 degradation+modernization items to `defect`**. Cascade: `KIND_BOOST` penalty lost, `MAX_UPGRADE_SEVERITY` cap never applies (cosmetic items can display severity 5), `upgrade_count` always 0, bucket ordering collapses. Published via `artifact_writers.py:848-857`. |

### (B) Drops or mangles data

| # | Location | What happens |
|---|---|---|
| B1 | `scene_classifier_passes.py:1496-1502` | Pass 2e sanity check deletes **100% of issues** under v2 (auditable via `removed_reason_counts`, but total loss). Currently unreachable behind the orchestrator `return` at `:992`. **This is the blocker gating cutover.** |
| B2 | `scene_classifier_passes.py:1374-1379` | Speculation suppression gated on `kind == "upgrade"` → speculative modernization/degradation text now leaks into display issues (under-filtering). |
| B3 | `scene_classifier_orchestrator.py:1025-1027` | `resolved_kind in {"defect","upgrade"}` never true → `catalogItemKind` never stamped. Unreachable today. |
| B4 | `scene_classifier_orchestrator.py:1140` | 2e-skipped path calls `_label_to_kind`, which no longer exists → NameError if reached. Deliberate: loud, not silent. |
| B5 | `rehab_packages.py:1181-1210` | `is_dated_cosmetic_evidence` always False; `is_defect_driver` loses 35 items. Exported API + test contract to rewrite. |
| B6 | `catalog_cost_model.py:69-72` | `upgrade` → ROOM_ALLOWANCE derivation dead; 6 v2 items silently fall to `LINE_ITEM`/`legacy_default`, changing emitted cost-model provenance and package absorption labels. |
| B7 | `benchmarking/vocabulary.py:100-121` | `default_actionability` buckets all 78 degradation+modernization items as `repair`, and that value is **frozen into the sealed snapshot** and shown to human annotators as the default — silently wrong gold data. |
| B8 | `benchmarking/vocabulary.py:29`, `:177` | Seals `catalog_kinds` from `catalog_validation.VALID_KINDS` (the *legacy* alias), so a v2-derived snapshot records `["defect","upgrade"]` while its items carry three kinds. Corrupts the fingerprint. |
| B9 | `benchmarking/catalog_index.py:38,43,52,68,95` | Annotator search/display keyed on two kinds; `{row['kind']:<8}` breaks on 13-char `modernization`. |
| B10 | `catalog_embeddings.py:544-568` | `embeddings_retrieve_upgrade_candidates` returns `[]` against a v2 index (correct under strict filtering, but dead API). |
| B11 | `artifact_writers.py:686`, `:697` | `unmapped_issues` kind passthrough — unvalidated, low risk, listed for completeness. |
| B12 | `artifact_writers.py:223-243` | Legacy split-format loader still stamps the retired `upgrade` string; `:261` log counter defaults to `defect`. |

### (C) Cosmetic, logging, or already fail-closed

- **C1 — blocked harnesses** (`catalog_auditor.py:2003`, `model_comparison.py:2571`,
  `bias_check.py:428`): still `RuntimeError` at `main()`. Internals needing rewrite:
  `catalog_auditor` `MATCH_THRESHOLD_DEFECT/UPGRADE` (`:107-108`) — **thresholds keyed by
  kind with no third value**, and the `else` at `:289-290` silently gives degradation and
  modernization the *upgrade* threshold; plus its private copy of the routing/retrieval
  block (`:312-315`, `:421-435`), which now fails closed to `[]` for every observation.
  `model_comparison` `_label_to_kind` (`:259`), forward-set filters (`:410-414`, `:1060`),
  prompt text (`:774-832`).
- **C2** `quant_artifact_comparison.py:341,504,564,607` — v1 label vocabulary in reports.
- **C3** `benchmark.py:247` — `--kind` argparse `choices=("defect","upgrade")` rejects the new kinds.
- **C4** `pipeline_config.py:137-161` — `SHADOW_LANE_*` config now has **no consumer at all**
  (the lane was deleted in C4); dead config to remove.
- **C5-C13** — docstrings (`pass_config.py:13,200,431`), legacy key aliases
  (`defect_id`/`upgrade_id`), `orchestrator.py:484` kind default, audit-script report lines
  (`audit_issue_catalog.py:480-481` prints "0 upgrades"), `audit_pass2c_funnel.py:74`,
  `artifact_viewer/app.py:198-202` (dead code reading keys in neither catalog), and
  `catalog_validation.py:57` (`VALID_KINDS` is correctly the legacy alias for the validator,
  but it is the name other modules import — see B8; rename or re-export deliberately).

### Two wiring hazards found outside the requested scope

- **W1** `rehab_packages.py:754-762` — `_default_package_affinity()` hardcodes
  `tools/issue_catalog.json`. The migration renames 19 v1 ids out of existence and adds 42
  new ones, so any caller passing `table=None` would route against dead ids with **zero
  error**. Production entry points pass the catalog explicitly, so this is latent.
- **W2** `pipeline_config.py:64` — `ISSUE_CATALOG_PATH` still points at the v1 catalog,
  confirmed. The v2 catalog is referenced only by the generator, the benchmark, and the
  validator. Cutover is currently **triple-gated**: the writer guards, `pipeline_mode`, and
  the 2e `return`.

### Suggested Task 3 order (risk per line)

1. `costing.KIND_MULT` (A1/A2) — two lines, 30 items of direct dollars.
2. `estimate_scope` A3 first (one-line field-list fix), then A4/A5.
3. `property_summary_pass` (A8) — self-contained, but changes the published `summary_v1`
   shape (`upgrade_count` needs a successor name).
4. `rehab_packages` drivers (A6) + display class (A7).
5. Pass 2e sanity drop + speculation gate (B1/B2), then lift the orchestrator `return` and
   rewire 2e's input onto `result.observations` (B3/B4).
6. `catalog_cost_model` (B6) and the benchmarking vocabulary/index (B7-B9) — the snapshot
   must be re-sealed before any new benchmark reference is authored.
7. Blocked harnesses (C1) last.

## Task 3 constraints

- **Do not switch `ISSUE_CATALOG_PATH`** to the v2 catalog until every consumer below is
  migrated *and* pricing/package metadata is authored for the 42 deferred successors.
- **Do not remove either `write_photo_intel` guard.** Publication stays blocked until the
  post-Task-3 pricing work completes.
- Historical artifacts are `legacy_v1` and must never be reinterpreted. The manifest's
  `requires_re_resolution` flag says which legacy ids need re-resolution at cutover; it is
  an audit record, not a lookup table.
- Task 1's ontology is fixed. If a Task 3 decision seems to require reinterpreting a kind,
  that is a blocking contradiction to raise, not to patch.

## Tests

**Full suite: 1623 passed, 13 skipped** (Task 1 ended at 1515 passed / 23 skipped; the 10
dormant 2d tests are now live).

```bash
.venv/Scripts/python.exe -m pytest tests -q --basetemp=C:/Users/Steven/AppData/Local/Temp/claude/pytest
```

New: `tests/test_catalog_kind_v2.py` (v2 catalog + manifest + byte-parity of the generator),
`tests/test_catalog_resolution_benchmark.py` (40 harness tests — no sidecar, no LLM, no
network). Extended: `tests/test_kind_ontology_guards.py` (both publish guards + the shared
ontology identity assertions), `tests/test_catalog_embeddings.py` (strict filter semantics,
including "an unknown kind never searches the whole catalog"),
`tests/test_scene_classifier_passes.py` (strict routing table, 2d prompt pins),
`tests/test_candidate_provider.py` (10 revived 2d tests + the pipeline_mode contract).

Every skip is enumerated:
- 7 × `dormant_2e` — reason names Task 3 and the specific 2e sanity check that blocks
  revival.
- 4 × `blocked_live_2c` (`model_comparison` live cells).
- 1 embeddings integration (`RUN_EMBEDDINGS_INTEGRATION=1`).
- 1 pre-existing runtime skip in `test_catalog_validation.py`.

Two harness tests earned their keep during authoring: one caught gold items that were not
retrievable in their own scene group, another caught a paired group split across dev and
holdout (which would have silently broken the paired metric).

### Verification commands

```bash
.venv/Scripts/python.exe scripts/migrate_catalog_kind_v2.py
```
Run twice — outputs are byte-identical, and the parity test pins them against the committed
files.

```bash
.venv/Scripts/python.exe scripts/benchmark_catalog_resolution.py --lane v2 --cases holdout --model-config benchmarks/catalog-resolution-v2/models/terra.json --repeats 5 --compare-to benchmarks/catalog-resolution-v2/results/holdout_legacy_terra/report.json --gates
```

## Gotchas

- **The generator is the authoring path.** Edit `kind_v2_decisions.json` and re-run
  `scripts/migrate_catalog_kind_v2.py`; never hand-edit the v2 catalog, manifest, or audit
  report — the parity test will fail and the audit report will drift.
- **Two independent kind enums remain** outside the shared module:
  `catalog_validation.VALID_KINDS` (aliased to `LEGACY_CATALOG_KINDS`, correct — it is the
  v1 vocabulary) and `property_summary_pass.VALID_KINDS` (Task 3 work).
- **The embeddings sidecar reports false-healthy**: `/health` has returned 200 while every
  embeddings call failed. Retriever construction embeds the whole catalog, which is the real
  probe — the benchmark relies on that and fails loudly.
- **`benchmarks/catalog-resolution-v2/legacy_baseline/routing_snapshot.py` preserves v1 bugs
  on purpose.** Do not "fix" it; its bugs are the baseline.
- Both blocked harnesses (`catalog_auditor`, `model_comparison`) still import
  `evaluate_kind_routing` at module level. Its *semantics* changed under them — they remain
  hard-blocked at `main()` and are inventoried below.
