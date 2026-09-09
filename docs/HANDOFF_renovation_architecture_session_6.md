# Handoff — Renovation Architecture Session 6

Date: 2026-08-16. Branch: `renovation_architecture_rework` (HEAD `eff304f`;
Sessions 1–4 committed, Session 5 and Session 6 uncommitted — nothing staged
or pushed). Scope executed: Session 6 code of
`docs/IMPLEMENTATION_PLAN_renovation_scope_estimate_architecture.md` — frozen
shadow canary tooling, shared Terra budget, `new` selector activation,
authoritative root-key writer behavior, and publication-gate hardening — plus
**canary replica 1 (18/18 complete)**.

**SESSION OUTCOME: Session 6 is NOT complete.** After replica 1, Steven
stopped the canary: the measured API usage and the size of the v4→v5 headline
deltas require adjustments before further benchmarking. Replica 2, the
comparator/review step, the `new` cutover smoke, and the rollback smoke were
not executed; no release gate beyond replica-1 artifact validity has passed.
The backend remains on `current`. All findings are recorded — deliberately
without conclusions — in
`docs/HANDOFF_renovation_architecture_session_6_canary_cutover.md` for a
follow-up session to interpret and plan against.

The catalog file is byte-identical: SHA-256
`5BE11ABB6DAD9921C6F1A6851C972DA2DCE57F95CCB1FEB8577742A59349E1EC`
verified before and after. Schema v5 and all policy versions are unchanged —
Session 6 changes runtime selection, artifact placement, and release tooling,
not condition/work/package contracts.

## What was implemented

Session 6 completed the five untracked Session 6 drafts (they were written but
never executed — see Draft defects below) and opened the authoritative path.

### 1. Frozen-input canary (`scripts/run_kind_canary.py`)

Added optional `--input-freeze`. When supplied, each property runs from the
freeze's recorded image paths — every file SHA-256-verified before the
analyzer subprocess is invoked — and the freeze's stored property metadata,
instead of discovering both from disk. A changed image, a missing file, or a
manifest property absent from the freeze fails that property and is reported
in the failure list (`main` returns 1) without spending anything. Omitting the
flag preserves the legacy discovery behavior byte-for-byte.

### 2. Shared daily Terra budget (`RENOVATION_TERRA_USAGE_ROOT`)

The ledger path was previously derived solely from each run's `artifacts_root`,
so the canary's two isolated replica roots would each have received a full
2.5M-token daily allowance. Added `resolve_ledger_root()` plus an optional
`usage_root_override` on `TerraUsageLedger`, resolved from the new
`RENOVATION_TERRA_USAGE_ROOT` env var at the single construction site in
`review_pipeline.py`. Default behavior (ledger under the run's artifacts root)
is unchanged. The canary coordinator sets the override to the shared output
root, so both replicas debit one UTC-day ledger and the ceiling is enforced
pre-call across the whole canary.

### 3. `new` selector activation

Two independent gates blocked `new` and both are now open, with the
authoritative mode subject to *identical* validation to shadow:

- `pipeline_config.resolve_renovation_architecture` — `new` resolves only
  under `KIND_ONTOLOGY_VERSION=observation_kind_v2` (the same catalog coupling
  shadow already had); the "unavailable until the Session 6 cutover" raise is
  gone.
- `runtime.initialize_renovation_architecture` — accepts `shadow` and `new`;
  the ontology check, Terra/Sol model checks, token-cap checks, and catalog
  projection build now gate both modes. Previously those checks sat *after*
  the shadow-only raise, so naively opening `new` would have skipped every one
  of them.

`build_shadow_envelope` was generalized to `build_estimate_envelope` (the
builder was never shadow-specific), with `build_shadow_envelope` retained as a
module-level alias and both exported. The uninitialized-runtime provenance
path no longer hardcodes `architecture_mode="shadow"`; it stamps the
configured mode via `_configured_mode()`.

### 4. Writer behavior per mode (`tools/artifact_writers.py`)

`_write_renovation_architecture_shadow` became
`_write_renovation_architecture_estimate` with three distinct contracts:

| Mode | v5 placement | Failure behavior |
|---|---|---|
| `current` | none | n/a (nothing computed) |
| `shadow` | private `analysis_debug.renovation_estimate_v5` | swallowed; key omitted |
| `new` | root `renovation_estimate_v5` | **job-fatal** |

In `new` mode the envelope must be `state="complete"`; a failed or partial
envelope raises `PassExecutionError("renovation_architecture", "publish",
code="AuthoritativeEstimateIncomplete")` rather than publishing an artifact
with no estimate of record, and any private copy is removed so the two
placements can never coexist. `renovation_estimate_v4` is still computed and
attached independently in every mode — it remains the comparison and rollback
path. The authoritative root key is canonical output and therefore reaches the
slim artifact (only `analysis_debug` is stripped there); this was an explicit
decision, see below.

### 5. Publication-gate hardening (`tools/publication_gate.py`)

`_validate_renovation_architecture_keys` gained two rules:

- private and root v5 keys present simultaneously → reject (two estimates of
  record with no tiebreak rule);
- a root envelope must carry `provenance.architecture_mode == "new"` → a
  shadow-stamped envelope can no longer occupy the authoritative slot.

Existing private/root state rules are unchanged.

### 6. Comparator review binding (`tools/compare_renovation_architecture_cutover.py`)

A reviews payload omitting `freeze_sha256` previously passed the binding
check. It now fails the `review_freeze_mismatch` gate (with expected/found
reported), so an unbound review file cannot be carried over from an earlier
canary to authorize deltas nobody looked at.

## Draft defects found and fixed

The five Session 6 draft files existed but had never been run. Two defects
would each have invalidated the canary:

1. **Property metadata was never loaded.** Both `_latest_metadata`
   (coordinator) and `_stored_property_metadata` (kind driver) read
   `artifact["property"]["metadata"]`, which is `null` in current artifacts —
   the listing facts (price/beds/baths/sqft/`area_price_per_sqft`) live at the
   artifact **root** key `property_metadata`. Every canary listing would have
   been priced with no metadata at all. Both helpers now read the root key,
   with `property.metadata` kept as a legacy fallback. The freeze now carries
   metadata for 16 of 18 properties; `redfin_125970550` and `redfin_80925528`
   have none stored in any run (pre-metadata-wiring artifacts). Both engines
   receive identical inputs either way, so the v4-vs-v5 comparison stays
   valid for those two — they simply exercise the no-metadata path.
2. **The coordinator could not import `tools`.** `run_renovation_architecture_canary.py`
   never inserted the repo root into `sys.path`, so a direct
   `python scripts/...` invocation died in `build_freeze` with
   `ModuleNotFoundError: No module named 'tools'`. Fixed with a module-level
   `sys.path` insert.

Additionally, `SAFE_MODEL_ENV_KEYS` now records `LM_STUDIO_URL` and
`LM_STUDIO_MODEL` in the freeze: pass 1a runs on LM Studio, so local routing
changes upstream observations and belongs in the frozen record.

## Files changed

- `scripts/run_kind_canary.py` — `--input-freeze`, `_sha256_file`,
  `_frozen_images` (pre-subprocess hash verification), root-key metadata fix.
- `scripts/run_renovation_architecture_canary.py` — `sys.path` insert,
  root-key metadata fix, `LM_STUDIO_*` in the freeze, shared
  `RENOVATION_TERRA_USAGE_ROOT`, `--replicate {1,2}` for staged execution,
  richer success summary.
- `tools/pipeline_config.py` — `new` resolves under the v2 ontology;
  `resolve_renovation_terra_usage_root` + `RENOVATION_TERRA_USAGE_ROOT`.
- `tools/renovation_architecture/runtime.py` — `new` initializes with
  shadow-identical validation; `build_estimate_envelope` (+ alias);
  `_configured_mode()` replaces the hardcoded shadow provenance.
- `tools/renovation_architecture/usage_guard.py` — `resolve_ledger_root`,
  `usage_root_override`.
- `tools/renovation_architecture/review_pipeline.py` — ledger construction
  passes the override.
- `tools/renovation_architecture/__init__.py` — exports
  `build_estimate_envelope`.
- `tools/artifact_writers.py` — three-mode seam (see above).
- `tools/publication_gate.py` — mutual exclusion + authoritative-mode check.
- `tools/compare_renovation_architecture_cutover.py` — review binding.
- `.env` — `RENOVATION_TERRA_MODEL=gpt-5.6-terra`,
  `RENOVATION_SOL_MODEL=gpt-5.6-sol`, both 8192 caps, `LM_STUDIO_URL`,
  `LM_STUDIO_MODEL`.
- Tests: `tests/test_renovation_architecture_runtime.py`,
  `tests/test_kind_ontology_guards.py`,
  `tests/test_renovation_architecture_usage_guard.py`,
  `tests/test_renovation_architecture_canary.py`.

## Decisions

- **Root v5 stays in the slim artifact** (Steven's decision). The
  authoritative envelope is the canonical versioned output the future frontend
  adopts; the current frontend ignores unknown keys. Only `analysis_debug` is
  stripped from slim.
- **No Sol budget guard** (Steven's decision). Sol makes exactly one
  listing-level call per property (max 37 across the canary, both smokes
  included) against a 250k/day Sol budget that nothing in the codebase
  enforces — the usage guard is Terra-only. Mitigation is staged execution
  plus measurement between replicas, not code. `--replicate {1,2}` exists for
  that staging.
- **Model env lives in `.env`** (Steven's decision), so worker restarts keep
  the routing. `OPENAI_MODEL` stays unset — legacy pass routing is untouched.
- **`build_shadow_envelope` kept as an alias** rather than updating every
  caller: the rename is cosmetic and the alias keeps Session 1–5 code and
  tests working.
- **Identical validation for `shadow` and `new`** rather than a lighter path
  for the authoritative mode — the runbook requires startup to refuse
  readiness on an invalid catalog or model, and the checks previously lived
  inside the shadow-only branch.

## Deviations from the implementation plan

- **Local model host changed, model preserved.** The reference 1a/2d routing
  is `unsloth/qwen3.6-27b@q6_k`, configured at `100.102.92.1:1234`. That host
  was unreachable at execution time; the canary runs the identical model at
  `http://169.254.83.107:1234` instead. The model — the part that affects
  upstream observations — matches the reference exactly, and both host and
  model are recorded in `input_freeze.json`.
- Everything else follows the plan's Session 6 procedure.

## Tests and exact commands

All on 2026-08-16, all green (fake providers; no live call in any test):

```
.venv\Scripts\python.exe -m pytest tests\test_renovation_architecture_reconciliation.py tests\test_renovation_architecture_contracts.py tests\test_renovation_architecture_runtime.py tests\test_renovation_architecture_package_candidates.py tests\test_renovation_architecture_sol.py tests\test_renovation_architecture_work_items.py tests\test_renovation_architecture_terra.py tests\test_renovation_architecture_conditions.py tests\test_renovation_architecture_disposition.py tests\test_renovation_architecture_usage_guard.py tests\test_renovation_architecture_catalog.py tests\test_renovation_architecture_canary.py tests\test_kind_ontology_guards.py tests\test_artifact_writers.py -q --basetemp=C:\tmp\rv_pytest_session6_focused_20260816
  -> 437 passed

.venv\Scripts\python.exe -m pytest tests -q --basetemp=C:\tmp\rv_pytest_session6_full_20260816
  -> 2248 passed, 6 skipped, 25.44s   (Session 5 baseline: 2228 passed, 6 skipped; +20 new tests)
```

`git diff --check` clean (pre-existing CRLF warnings only); catalog SHA-256
reproduced before/after; tracked modifications are exactly the 11 intended
files (plus `.env` and the untracked drafts/new files). New tests: `new`-mode
selector/init (4), new-mode writer contract (3), publication-gate mutual
exclusion + authoritative-mode (2, +3 updated), shared usage root (4),
comparator review binding (1), coordinator freeze drift (1), plus the 2
pre-existing draft driver tests turned green by `--input-freeze`.

Live-run evidence, token accounting, and the v4→v5 comparison are in
`docs/HANDOFF_renovation_architecture_session_6_canary_cutover.md`.

## Session 7 / follow-on prerequisites

There is no Session 7 in this migration. Follow-on work, each separately
approved:

- Frontend adoption of the v5 contract (the backend now emits it at the
  artifact root in `new` mode).
- Historical backfill / reprojection (`needs_reanalysis` rules unchanged).
- Catalog price calibration.
- Legacy estimator retirement — **not yet**: v4 remains the rollback path and
  must stay until the frontend redesign and an observation period are done.
- A Sol daily budget guard, if Sol spend is to be enforced rather than
  observed.
- Pass-table token attribution for the new stages: the per-job timing table
  reports Terra condition review + Sol package review as "unattributed"
  (the v5 envelope's own accounting is exact — this is display only).
  Decision: reuse the existing `terra_review` / `sol_review` keys rather
  than inventing pass codes like 3a/3b, so the per-listing stages don't
  masquerade as `--model-map`-routable photo passes. Deferred from this
  session because the canary freeze hashes the relevant files.
- **Terra batch-size quality review** (raised by Steven during the canary):
  condition review batches by estimate unit and reaches 20 conditions / 12
  images / 18k input tokens in one call. No Session 6 gate detects this — the
  verdicts are structurally valid and reconcile exactly — but a single call
  asked for 20 independent judgments over 12 photos may be diluting
  per-condition quality. Evidence and the capacity tradeoff are in
  `docs/HANDOFF_renovation_architecture_session_6_canary_cutover.md`
  ("Observations for review").
- **Terra capacity denominator**: the usage guard meters only the
  architecture's condition review (~41k/listing), not upstream 2a/2b/2c/2f
  (~124k/listing) on the same Terra service. Expected throughput is ~60 or
  ~16 listings/day depending on which the 2.5M ceiling covers. Needs a
  decision, and possibly a guard that sees total service usage.
- Carried forward from S4/S5: the missing `_extract_package_only_candidates`
  evidence lane, stricter multi-photo corroboration, S3 pricing deltas, and
  whether the conservative combine/split policy diverges materially from a
  priced transformed package.
