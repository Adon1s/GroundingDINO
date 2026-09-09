# Handoff: repair the package-outcome benchmark

## Objective

Repair the existing two-property package-outcome benchmark so it can reliably compare Pass 2a prompts, the Pass 2d observation cap, and later model configurations.

Keep this task limited to benchmark correctness, diagnostic scoring, provenance, and tests. Do not redesign the production package ontology, catalog, prompts, or package builder unless a very small compatibility change is indispensable. Leave detailed implementation choices to the implementing session.

Benchmark worktree:

`C:\Users\Steven\PycharmProjects\rv-pass2a-bench`

Branch:

`pass2a_prompt_bench`

Current main workspace, which contains the newer Pass 2c implementation:

`C:\Users\Steven\PycharmProjects\realtorvision-backend`

The benchmark worktree is dirty and contains uncommitted benchmark code, gold, reports, and tests. Preserve those changes. Do not reset or overwrite them.

## Required experiment routing

The repaired benchmark must run **Pass 1a scene classification on Terra**, not local Qwen.

Passes 1a, 2a, 2b, and 2c should use the explicitly configured OpenAI model routing for the experiment. Pass 2d may remain local Qwen unless the experiment explicitly changes it.

Do not assume that adding `1a` to a configuration dictionary is sufficient. Verify the actual runtime route and record it in the saved provenance. Add a regression test or runtime assertion proving that Pass 1a used the configured Terra model.

For Pass 2a prompt comparisons, scene classification must also be controlled across variants. The implementation may run Terra Pass 1a once and replay/freeze its result for all Pass 2a variants, or use another sound design that guarantees identical scene and physical-room assignments across the compared variants. Separate Pass 1a model-quality experiments from Pass 2a prompt experiments.

## Why the benchmark needs repair

### 1. Scene and room identity are currently uncontrolled

The intended experiment changes Pass 2a, but each variant/repeat reruns the entire image pipeline, including Pass 1a. In the saved run, Pass 1a used local Qwen.

Carroll `photo_005.jpg` is gold bedroom 2, but it was classified as `living_room` in every checklist repeat and baseline repeat 2. This changed the downstream unit allocation:

- `photo_003` became `bedroom_1`;
- `photo_005` became `living_room_primary`;
- `photo_007` became `bedroom_2`;
- `bedroom_3` ceased to exist.

The scorer consequently credits and penalizes packages using unstable ordinal room IDs. A predicted `bedroom_2` package supported by `photo_007` actually corresponds to gold bedroom 3.

Relevant files:

- `tools/benchmark_pass2a.py`, especially `run_property_once`
- `tools/room_surrogates.py`
- `benchmarks/pass2a-prompt/gold/package_reference.json`

Required result:

- Terra is the actual Pass 1a runtime model.
- Pass 2a variants use identical frozen/replayed scene assignments.
- Benchmark gold defines canonical physical rooms/photo groups independently of predicted ordinal unit IDs.
- Predictions are aligned to canonical rooms by supporting photo membership before package scoring.

Do not modify production scene-classification behavior merely to make this benchmark pass.

### 2. Standalone work is dropped before scoring

`v4_line_items(..., valid_only=True)` removes line items whose `is_valid_detection` is not truthy. Standalone items do not normally receive package Pass 2f review, so a correct standalone item may retain `None` and disappear.

Carroll `ceiling_cracks_or_sagging` was observed and resolved upstream, but the benchmark projection removed it. The scorer separately claims that valid standalone work should satisfy required-work gold, so projection and scoring disagree.

Relevant code:

- `tools/benchmark_pass2a_packages.py`, `v4_line_items`
- required-work satisfaction in `score_package_round`
- `tools/issue_catalog_kind_v2.json`, `ceiling_cracks_or_sagging`

Required result: correctly eligible standalone work survives benchmark projection and can satisfy required work, while genuinely invalid or rejected detections remain excluded.

### 3. Gold needs a reachability audit

Some required IDs are unreachable under current catalog/assembly policy or do not match the catalog's visible-condition definition.

Known examples:

- `dated_entry_or_patio_door` is required by gold but is optional, has `package_role: ignore`, and lacks a usable estimate/package path.
- Some `countertop_damage` rationales describe datedness, wear, staining, or unfinished caulk rather than the catalog's physical-defect definition.
- Carroll gold acknowledges that `damaged_or_rotted_siding_or_trim` is more severe than the observed weathering.
- Carroll `tile_or_grout_damage` gold acknowledges staining/residue rather than broken, loose, or missing tile.
- Frisby package gold says broken tile while the frozen observation reference describes darkened/stained grout.

Relevant files:

- `benchmarks/pass2a-prompt/gold/package_reference.json`
- `benchmarks/pass2a-prompt/gold/reference.json`
- `tools/issue_catalog_kind_v2.json`

Required result: add a benchmark-side validation/report that determines whether each required work item can reach a scoreable output through its kind, scene eligibility, catalog policy, pricing or package affinity, and family-subsumpion rules. Correct, remap, or explicitly mark known incompatible targets so they do not appear as model failures.

Keep exact catalog-ID accuracy as a diagnostic. Do not broadly rewrite the production catalog in this task.

### 4. Scoring needs useful diagnostics in addition to strict completion

The current repeat is complete only if every expected package, every exact required-work ID, and every extra adjudication passes. One catalog cousin or unreachable label fails the whole property repeat.

Retain strict completion as a secondary release-style gate, but report at least:

- package-family recall;
- tier result or distance from the target tier;
- required component/scope coverage;
- exact catalog-ID coverage;
- unreachable or incompatible gold targets;
- extras;
- final strict completion.

Where accepted sibling IDs represent the same intended component/scope, report concept satisfaction separately from exact-ID satisfaction. The implementing session should choose the simplest maintainable schema.

Relevant code:

- `tools/benchmark_pass2a_packages.py`, `score_package_round`
- strict completion around the `complete` calculation
- candidate diagnostics and cap experiment reporting

### 5. Cap comparisons must be paired and must account for Pass 2f variance

`missing_for_gold_reconsideration` is currently the union of anything missing in any run; it is not a list of uniform misses. The cap verdict compresses all outcomes into a binary two-of-three result, so unrelated failures can yield `both_wrong` despite meaningful package improvements.

Capped and uncapped cells also receive separate stochastic Pass 2f calls. A disagreement must not automatically be attributed to the observation cap.

Required result:

- compare capped and all-retained results by property and repeat;
- report deltas for package recall, tiers, required-work/component coverage, extras, and cost;
- identify Pass 2f disagreement when comparable candidate evidence receives different decisions;
- rename or restructure union-of-misses output so its semantics are clear.

Prefer rescoring the saved artifacts. Do not rerun the 123 paid Pass 2f calls unless the implementation first demonstrates why that is unavoidable.

## Existing artifacts and verified facts

Important files/directories:

- `benchmarks/pass2a-prompt/runs/package_eval/scores.json`
- `benchmarks/pass2a-prompt/runs/package_review/review.csv`
- `benchmarks/pass2a-prompt/runs/package_2f/**`
- `benchmarks/pass2a-prompt/runs/package_cells/**`
- `benchmarks/pass2a-prompt/runs/package_tail_resolution/**`

Verified counts:

- 2 properties and 15 photos;
- 3 configurations x 3 repeats x 2 properties = 18 correlated evaluation instances;
- 11 expected packages and 28 required work items;
- 123 saved Pass 2f calls: 96 confirmed, 20 rejected, 7 uncertain;
- 5 pending review rows;
- official verdicts are blocked because pending review globally changes `passed` to `null`.

These are repeated measurements of two houses, not 18 independent samples. Treat both houses as a development set.

Previous claims that must not be carried forward:

- Pass 2f's three-image cap did not bind in these artifacts. Of 123 candidates, none had more than three evidence photos.
- Carroll kitchen used only `photo_004.jpg` in all nine evaluations, so the image cap did not cause its tier behavior.
- Carroll's capped kitchen was already `kitchen_refresh` before Pass 2f; it was not demoted from full rehab by Pass 2f.
- The existing `missing_for_gold_reconsideration` list is not evidence of uniform upstream misses.

## Incorporating the newer Pass 2c branch

The benchmark branch forked before the latest Pass 2c work. The current main workspace and `pass_2c_redesign` point at commit `ae3630a`, while the benchmark branch and its saved upstream artifacts were produced from the older Pass 2c v2 lineage.

The repaired benchmark should ultimately run against the current Pass 2c implementation and current explicit OpenAI image detail behavior. However, do not silently reinterpret or regenerate old artifacts as if they came from the new code.

Recommended integration order:

1. Preserve/commit the dirty benchmark work on a dedicated branch.
2. Merge or rebase the current `pass_2c_redesign` work into that benchmark-repair branch, resolving conflicts deliberately.
3. Keep existing paid artifacts labeled as legacy v2 artifacts and allow them to be rescored only where the repair is purely scoring/alignment logic.
4. Give any new v3 benchmark run a new fingerprint/stage directory. Do not resume v2 checkpoints under v3 code.
5. Ensure fingerprints include the actual Pass 1a model/provider, Pass 2c prompt/version, image-detail setting, and other runtime routing needed to distinguish the datasets.

Current-code differences already verified:

- benchmark snapshot: `pass_2c_kind_v2`;
- current workspace: `pass_2c_kind_v3`;
- benchmark snapshot omits explicit OpenAI image detail;
- current workspace sends `detail: "original"`.

## Scope boundaries

In scope:

- Terra routing and provenance for benchmark Pass 1a;
- freezing/replaying Pass 1a for Pass 2a comparisons;
- canonical room/photo alignment;
- benchmark scoring and report corrections;
- standalone-work projection;
- gold reachability validation;
- paired cap diagnostics;
- provenance/fingerprint improvements;
- tests;
- rescoring saved artifacts where valid.

Out of scope:

- production prompt redesign;
- production catalog overhaul;
- redesigning package-builder or Pass 2f policy;
- new model-quality experiments;
- adding more properties;
- automatically rerunning paid stages.

## Acceptance criteria

The repair is complete when:

1. Pass 1a is demonstrably routed to Terra and the actual route is recorded in provenance.
2. Pass 2a variants cannot receive different scene assignments from incidental Pass 1a reruns.
3. Packages are evaluated against stable physical-room identities.
4. Correct standalone work is not lost solely because it did not receive package Pass 2f review.
5. Required gold targets receive a reachability/incompatibility result.
6. Exact-ID performance and component/scope performance are reported separately.
7. Cap comparisons are paired by property/repeat and do not reduce all evidence to only `both_wrong`.
8. Existing paid Pass 2f artifacts can be rescored without being regenerated.
9. New v3 runs cannot accidentally reuse or overwrite legacy v2 artifacts.
10. Tests cover at least:
    - Terra Pass 1a routing/provenance;
    - frozen scene assignments across Pass 2a variants;
    - Carroll room-number drift;
    - standalone ceiling-crack credit;
    - unreachable ignored/optional required work;
    - union-of-misses semantics;
    - paired cap reporting;
    - version/fingerprint separation between legacy v2 and current v3 runs.
11. Existing benchmark tests continue to pass.

Baseline test command from the benchmark worktree:

```powershell
C:\Users\Steven\PycharmProjects\realtorvision-backend\.venv\Scripts\python.exe -m pytest tests\test_benchmark_pass2a_packages.py -q
```

The suite was independently rerun and currently passes 35 tests.
