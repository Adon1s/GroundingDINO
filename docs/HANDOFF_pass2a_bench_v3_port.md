# Handoff — port the Pass 2a benchmark v3 work into the backend

Written 2026-08-30 · backend `renovation_architecture_rework` @ `c833da4` · commissioned by
Steven

**Purpose.** Consolidate the Pass 2a prompt/package benchmark into one authoritative lineage
in `realtorvision-backend`, so that later work has a single place to run it from. This is the
declared prerequisite for the Prompt Lab (Terra verifier + Pass 2a tabs); that build starts
once this lands.

**Token cost of this task: zero.** It is a code transplant plus rescoring of already-paid
artifacts. Re-running any live stage is out of scope and needs its own decision.

---

## 1. Correct the premise first: there is nothing to merge

The work was described as "merge the v3 lineage from `rv-pass2a-bench`". At the commit level
that is a no-op, and a session that starts with `git merge` will conclude, wrongly, that
there is nothing to do.

```bash
git merge-base codex/benchmark-repair renovation_architecture_rework
# -> f55d59e1aece58df218261d0a0f7220ccdc6b3a8   == codex/benchmark-repair's own HEAD
```

| | commits |
|---|---:|
| `f55d59e..codex/benchmark-repair` | **0** |
| `f55d59e..renovation_architecture_rework` | **20** |

`git merge-base --is-ancestor codex/benchmark-repair renovation_architecture_rework` returns
true. `git branch --contains f55d59e` lists `renovation_architecture_rework` among others.
Every committed bench artefact — the harness, the frozen 112-condition gold, the judge round,
the package harness — **is already at backend HEAD**. `pass2a_prompt_bench` (`8db59db`) and
`benchmark_system` (`a1972cf`) are likewise plain ancestors with nothing unique.

**The v3 work exists only as the uncommitted working tree in
`C:\Users\Steven\PycharmProjects\rv-pass2a-bench`.** This task is a **15-file transplant**,
not a branch merge.

---

## 2. The payload

8 modified tracked files (+1176 / −654) and 7 untracked files (1917 lines):

| Modified | Δ lines | | Untracked | lines |
|---|---:|---|---|---:|
| `tools/benchmark_pass2a_packages.py` | 699 | | `tools/benchmark_pass2a_scoring.py` | 808 |
| `tests/test_benchmark_pass2a_packages.py` | 511 | | `tools/benchmark_pass2a_scenes.py` | 272 |
| `benchmarks/pass2a-prompt/gold/package_reference.json` | 262 | | `tools/benchmark_pass2a_reachability.py` | 190 |
| `tests/test_benchmark_pass2a.py` | 116 | | `tools/benchmark_pass2a_legacy.py` | 189 |
| `…/gold/package_reference.template.json` | 96 | | `tests/test_benchmark_pass2a_scenes.py` | 201 |
| `tools/benchmark_pass2a.py` | 92 | | `tests/test_benchmark_pass2a_legacy.py` | 160 |
| `tools/scene_classifier_orchestrator.py` | 50 | | `tests/test_benchmark_pass2a_scoring.py` | 97 |
| `benchmarks/pass2a-prompt/config.json` | 4 | | | |

Tests go **86 → 120**.

### Conflict risk is effectively nil — this was verified, not assumed

The 20 backend commits touched **no benchmark file at all**: `tools/benchmark_pass2a*.py`,
`tests/test_benchmark_pass2a*.py` and everything under `benchmarks/pass2a-prompt/` are
byte-identical between `f55d59e` and backend HEAD. `tests/conftest.py` is byte-identical
between the two worktrees.

Of the benchmark's dependencies, only three moved, and none breaks it:
`artifact_writers.py` (+187/−8, v5 envelope writing; `load_issue_catalog` untouched),
`pipeline_config.py` (+203, the `RENOVATION_ARCHITECTURE_MODE` selector and the Terra
condition-review block), `issue_catalog_kind_v2.json` (+5, `route_override` additions).
`scene_classifier_passes.py`, `scene_classifier_orchestrator.py`, `rehab_packages.py`,
`renovation_estimate_v4.py`, `room_surrogates.py` and `comparison_common.py` are identical.

Every symbol the four new modules import exists at backend HEAD with the same definition,
including the private ones (`rehab_packages._REPAIR_SUBSUMING_MODERNIZATION_TIERS`,
`_PRICING_TIER_ESCALATION`, `_normalize_scene_to_room`, `classify_component`,
`build_package_affinity`, `catalog_package_role`;
`renovation_estimate.product_quarantined_trade_buckets`; `room_surrogates.build_room_surrogates`;
`artifact_writers.load_issue_catalog`; `vlm_client.create_vlm_client` /
`get_model_configs_from_pipeline_config`). **Nothing imports `tools/renovation_architecture/`**
— the v5 engine is not a dependency of the benchmark.

---

## 3. What the v3 restructure is

- Run dirs move to `V3_DIR = RUNS_DIR / "v3"`, with `LEGACY_*` constants preserving the flat
  v2 paths as read-only evidence.
- `compute_fingerprint()` gains three pinned keys — `scene_capture_sha`,
  `pass_2c_prompt_version` (from `PASS_2C_PROMPT_VERSION`, backend
  `tools/scene_classifier_passes.py:829`, `"pass_2c_kind_v3"`), and a literal
  `image_detail: "original"` — which is what makes v2 and v3 artifacts mutually
  non-resumable. `guard_fingerprint()` itself is unchanged.
- Two new subcommands: `scene-capture [--check]` and `legacy-rescore --round`.
- Tri-state 2f line filter: `if not li.get("is_valid_detection")` becomes
  `if li.get("is_valid_detection") is False`, so a standalone line 2f never reviewed
  (`None`) survives projection. This is the "Carroll regression".
- `PACKAGE_GOLD_SCHEMA_VERSION` `package_reference_v1 → v2`; `GOLD_POLICIES =
  ("strict", "diagnostic_only")`; gold moves from ordinal room ids (`bedroom_1`) to canonical
  ones (`bedroom_A`) with explicit `photo_keys`.
- Scoring, scene capture, legacy rescore and reachability are extracted out of
  `benchmark_pass2a_packages.py` into four new modules.

| New module | What it does |
|---|---|
| `benchmark_pass2a_scenes.py` | Freezes Terra Pass 1a once per photo into `runs/v3/scene_capture/scenes.json`, proves the route was the explicit OpenAI override (never a silent local fallback), and replays those scenes into every variant/repeat. `load_scene_capture` re-validates photo set, per-image sha, self-hash and routing on every load. |
| `benchmark_pass2a_scoring.py` | The heart of the repair. Aligns predictions to canonical physical rooms by supporting-photo overlap *before* matching, so ordinal drift can no longer flip credit between rooms. Adds signed tier distance, family recall, component/exact-ID coverage, paired per-repeat cap deltas and 2f-noise attribution; demotes strict completion to a secondary verdict. |
| `benchmark_pass2a_reachability.py` | Per gold target, asks whether the catalog + assembly policy can produce a scoreable output at all (catalog existence, trade quarantine, scene eligibility, priced-standalone path, package-affinity path incl. the repair→modernization bridge, tier suppression). Read-only, no I/O — the safest file to copy. |
| `benchmark_pass2a_legacy.py` | Offline read-only rescore of the frozen v2 artifacts under the repaired projection + canonical-room scorer, so old evidence stays comparable. Zero VLM calls; asserts hard if `pass_2f_trace.ran` is truthy. |

`config.json` changes are three real lines — `"1a": "gpt-5.6-terra"` in `model_overrides`,
`"1a": "low"` in `reasoning_efforts`, plus a rewritten `note`. **No gate values changed**;
`gates`, `package_eval.gates`, `tail_resolution`, `openai_max_output_tokens`, `judge` and
`matcher` are byte-identical.

---

## 4. The one production file, and why it is safe

`tools/scene_classifier_orchestrator.py` (+50) adds a `pass_1a_frozen_scene` benchmark meta
hook that mirrors the existing `pass_2a_frozen_freeform` / `pass_2a_user_prompt` hooks in the
same file (lines 741-782). It re-indents the live Pass 1a block into an `else:` branch, so
production behaviour is unchanged whenever `meta` is absent, and widens one import to
`from tools.pipeline_common import SCENE_TO_GROUP_UI, normalize_scene_id`.

Verified mergeable: the file is identical between `f55d59e` and backend HEAD, backend's line
38 import and its 1a block at line 685 match the patch's `-` side character for character,
and `normalize_scene_id` exists at `tools/pipeline_common.py:109` (`Pass1aResult` at
`tools/scene_classifier_passes.py:318`).

Note the deliberate semantics the tests pin: the replay path does **not** call
`_record_model_routing`, so `result.model_routing` has no `1a` entry and
`models_used["1a"] == "frozen_replay"`.

**Without this file the benchmark's `scene-capture` and `run` stages are inert.** It must
travel with the port, and it is the one hunk that deserves review on its own.

---

## 5. Required scope: finish the v3 layout split

The v3 run-directory migration is **half-done**, and the unfinished half is exactly what the
Prompt Lab depends on.

`run` now writes to `runs/v3/variant_*`, and the package pipeline was fully migrated (it
introduced `variant_dir()` and rewrote all its own call sites). But four call sites still
resolve artifacts through the flat v2 layout:

- `find_stage_artifacts` — `tools/benchmark_pass2a.py:714`
- `load_photo_repeat_records` — `:933`
- `match_fingerprint` (reads `variant_{v}/fingerprint.json`) — `:1357`
- report / stability aggregation — `:1897`

and `attribution`, `judge_*`, `match_*`, `review_*` and `report.json|md` all still write to
flat `runs/` (`:814, :1193, :1445, :1518, :1627, :2301`).

**Consequence: after a v3 run, `match` finds nothing.** The Prompt Lab's Pass 2a tab is built
on the match + `decisions.json` review loop, so completing this migration is **in scope for
this port**, not an optional tidy-up. If there is a reason to leave `judge` on the legacy
layout (it is the archived Sol round), say so explicitly in the code and the report.

---

## 6. Order of work

1. **Clear the decks.** The backend worktree currently carries ~131 dirty entries — 8 tracked
   modifications (incl. `scripts/score_redecide_variants.py` and 6 docs) plus untracked
   `artifacts/`, `.claude/` and loose root `*.json` audit dumps. Commit or park the 8 tracked
   modifications so the port lands as a reviewable change.
2. **Capture the baseline** (§9) before touching anything.
3. **Copy the 7 untracked modules and tests.** Import order matters:
   `reachability → scenes → scoring → packages → benchmark_pass2a → legacy`
   (`packages.package_gold_check` imports reachability; `packages.score_package_round`
   delegates into scoring).
4. **Apply the 8 modified files**, taking `tests/test_benchmark_pass2a_packages.py` and
   `tests/test_benchmark_pass2a_legacy.py` **together** — the legacy test cross-imports
   `CATALOG, PROP, VALID_GOLD, _catalog_stub, _config, _manifest, _write_gold, pkg_tree`
   from the *bench* version of the packages test, where `_catalog_stub` is a new autouse
   fixture and `pkg_tree` gained the `V3_DIR` + five `LEGACY_*` monkeypatches.
5. **Review `scene_classifier_orchestrator.py` on its own** (§4).
6. **Finish the v3 layout split** (§5).
7. **Copy the paid artifacts** (§8).
8. **Verify** (§9), then commit as one change.

---

## 7. Pin the runtime — two hazards that post-date the bench snapshot

1. **`RENOVATION_ARCHITECTURE_MODE` did not exist at `f55d59e`.** It defaults to `current`
   (v4), but production runs `new`, and `tools/artifact_writers.py:432` branches on it — an
   inherited environment would silently add v5 work to every benchmark photo. The benchmark's
   gold and its 123 paid Pass 2f artifacts are v4/2f-bound
   (`benchmark_pass2a_packages.py` calls `compute_renovation_estimate_v4` / `v4_line_items`),
   so pin it explicitly rather than relying on the default. Pin `KIND_ONTOLOGY_VERSION`
   alongside it — `resolve_renovation_architecture()` raises at import if `shadow`/`new` is
   used without v2.
2. **The Terra/Sol usage guard and daily ledgers also post-date the snapshot.** Benchmark
   runs now debit the same 2.5M/day Terra and 250k/day Sol ledgers as production and Session
   F unless `RENOVATION_TERRA_USAGE_ROOT` is set deliberately. Decide and record which.

---

## 8. Preserve the paid artifacts — git will not do it for you

`benchmarks/pass2a-prompt/.gitignore` contains just `runs/`, and only three files were
force-added (`runs/judge_baseline_vs_checklist/judgments.json`, `runs/report.json`,
`runs/report.md`). Everything else exists **only** in the bench worktree, ~110 MB:

| dir | size | | dir | size |
|---|---:|---|---|---:|
| `package_cells` | 37 MB | | `package_tail_resolution` | 6.4 MB |
| `variant_checklist` | 34 MB | | `legacy_v2_rescore` | 1.2 MB |
| `attribution` | 13 MB | | `review_baseline_vs_checklist` | 880 KB |
| `variant_baseline` | 13 MB | | `match_baseline_vs_checklist` | 796 KB |
| `judge_baseline_vs_checklist` | 2.1 MB | | `package_2f` | 759 KB |
| | | | `v3` | 51 KB |

`runs/v3/` holds the completed `scene_capture` (all 15 photos, both properties, plus
`scenes.json`) and `package_eval/gold_reachability.json`. No v3 variant or match round has
been run.

Copy `runs/` into the backend's `benchmarks/pass2a-prompt/runs/` (gitignored there too), and
**do not remove the `rv-pass2a-bench` worktree until that copy is verified.**
`rv-legacy-a1972cf` is a clean ancestor snapshot with nothing to salvage.

---

## 9. Verification

Baseline **before** any change, from the backend:

```bash
.venv/Scripts/python.exe -m pytest tests/test_benchmark_pass2a.py tests/test_benchmark_pass2a_packages.py -q
```

Bench-side target — the bench worktree has no `.venv`, so use the backend interpreter with
cwd set to the worktree. This is the 120-test result the port must reproduce:

```bash
C:\Users\Steven\PycharmProjects\realtorvision-backend\.venv\Scripts\python.exe -m pytest tests\test_benchmark_pass2a.py tests\test_benchmark_pass2a_packages.py tests\test_benchmark_pass2a_scenes.py tests\test_benchmark_pass2a_scoring.py tests\test_benchmark_pass2a_legacy.py -q
```

The port is done when:

1. All five benchmark test modules pass in the backend (120 tests).
2. The full suite holds at its baseline: `.venv\Scripts\python.exe -m pytest tests -q`
   → 2433 passed / 6 skipped, plus the new tests.
3. `package-eval --stage score` reproduces `runs/legacy_v2_rescore/scores.json` from saved
   artifacts with **no new model calls**.
4. `scene-capture --check` validates the copied capture offline.
5. `match` and `report` resolve v3 run artifacts (§5).
6. `git status` shows no production-file change beyond the reviewed orchestrator hook.
7. The 11 acceptance criteria in `docs/HANDOFF_package_benchmark_repair.md` §"Acceptance
   criteria" are each demonstrably met — that document is the spec this work implements.

---

## 10. Decisions owed by Steven — surface, do not resolve

**a. Four gold targets demoted to `diagnostic_only`.** The v3 gold demotes these with
audit-derived reasons (`runs/v3/package_eval/gold_reachability.json`), all on
`redfin_11000447`:

| target | blocker |
|---|---|
| `peeling_or_discolored_paint@bedroom_A` | `affinity_package_mismatch: bedroom_modernization vs expected ['bedroom_repair']` |
| `peeling_or_discolored_paint@bedroom_C` | same |
| `vanity_countertop_dated@bathroom` | `affinity_package_mismatch: bathroom_modernization vs expected ['bathroom_repair']` |
| `dated_entry_or_patio_door@kitchen` | `no_package_affinity_for_room`; tier=optional, `package_role: ignore`, no estimate block |

The three `affinity_package_mismatch` rows are the ones recorded as awaiting review — a
demotion silently converts a model failure into a non-failure, so they need ratification.
The fourth was already named in `HANDOFF_package_benchmark_repair.md` §3 as unreachable by
construction.

**b. One gold target deleted outright.** `damaged_drywall_or_cracks@bathroom_primary`
(`redfin_11000447`) was removed as a required work item and folded into the `bathroom_repair`
package rationale, justified in-file as "no bathroom-repair-family wall item exists in the
catalog (Steven 2026-08-13)". Confirm that attribution stands.

**c. Five un-adjudicated package-review rows.** `runs/package_review/review.csv` has 5 rows
with an empty `human_decision` and no `decisions.json`; they hold `scores.json` at
`status=blocked`. Four are repair↔modernization family swaps, one is an add-on. **They were
written under ordinal room ids** (`bedroom_1`, `bedroom_2`, `bedroom_3`), so v3's
canonical-room rework changes their `row_id`s — state whether they can be imported or must be
re-exported.

**d. Provenance hash change.** `package_gold_sha(gold)` now uses `sha256_canonical(gold)`
instead of `sha256_file(PACKAGE_GOLD_PATH)`, so identical gold hashes the same under CRLF and
LF. This invalidates every previously recorded `gold_sha256` in stored review decisions —
affected rows will surface as `needs_rereview`. Expected, but say so in the report rather
than letting it look like data loss.

---

## 11. Known warts to flag while porting (do not silently fix)

- `assert_frozen_scene_consistency` and `artifact_scene_map` in `benchmark_pass2a_scenes.py`
  are documented as a package-evaluation preflight but are called from nothing except their
  tests. Wire them in or drop the docstring claim.
- `benchmark_pass2a_legacy.stage_legacy_rescore` rebinds module globals at runtime
  (`pkg.PASS2F_DIR`, `bench.V3_DIR`) inside a `try/finally`. Correct single-threaded, unsafe
  under `pytest-xdist` or any concurrent invocation.
- `_load_legacy_catalog` shells out to `git show <head>:tools/issue_catalog_kind_v2.json` to
  recover the v2-era catalog, trying both LF and CRLF forms against the recorded sha. It
  hard-codes that path independently of `cfg.ISSUE_CATALOG_PATH` and runs `git` with
  `cwd=bench.ROOT` — confirm `ROOT` resolves to the repo root in the backend.

---

## 12. Scope boundaries

**In scope:** the 15-file transplant; completing the v3 layout split; runtime pinning;
artifact preservation; tests; rescoring saved artifacts.

**Out of scope:** production prompt, catalog or package-builder changes; rerunning the 123
paid Pass 2f calls or any other live stage; regenerating v2 artifacts under v3 code; resuming
v2 checkpoints under v3 fingerprints; adding properties; and the Prompt Lab itself, which is
a separate build that starts after this lands.

Catalog changes, if any prove unavoidable, go through
`tools/catalog_migrations/kind_v2_decisions.json` + regeneration — never hand-edits; a parity
test pins this.

## 13. Reading order

1. `docs/HANDOFF_package_benchmark_repair.md` — the spec this work implements, and the source
   of the 11 acceptance criteria. **Start here.**
2. `docs/HANDOFF_package_benchmark_findings.md` — the measured failures that motivated it.
3. `docs/HANDOFF_benchmark_token_economics.md` — why the text-only match stage replaced the
   image-based Sol judge.
4. `docs/HANDOFF_pass2a_variance_ablation.md` — the original ablation design and its
   attribution stop-gate.
5. `benchmarks/pass2a-prompt/runs/report.md` — the last completed round, and the 2650
   un-adjudicated match rows that stalled it.
