# Handoff: Terra-vs-Sol Pass 2f comparison with optional Qwen

## Goal

Implement a cost-conscious comparison that reuses one property analysis produced
by the local model, runs only Pass 2f with Terra and Sol by default, and can
optionally add local Qwen as an auxiliary concurrent verifier. It preserves
complete package-level results in JSON. The report should highlight packages
approved by
exactly one contestant, but it must also retain both-approved, neither-approved,
rule-confirmed, skipped, and failed cases.

The intended workflow is:

1. The normal property pipeline runs once with local
   `unsloth/qwen3.6-27b@q6_k` for every model-routed upstream pass. Pass 2e is
   deterministic/rule-based. Do not rerun those passes for either premium model.
2. Freeze the package candidates, evidence items, selected review photos, and
   other Pass 2f inputs from that local artifact.
3. Run Pass 2f with `gpt-5.6-terra` for every VLM-eligible package.
4. Run Pass 2f with `gpt-5.6-sol` against the exact same prepared cases.
   Terra and Sol are independent concurrent streams.
5. When `--include-qwen` is requested, start a third concurrent stream using
   local `unsloth/qwen3.6-27b@q6_k` against the same frozen cases. Qwen is
   auxiliary and never changes the Terra/Sol approval bucket.
6. Classify packages by approval-set membership. The normal comparison does not
   make a separate judge call.
7. Optionally run a fresh `gpt-5.6-sol` executive review only for asymmetric
   approvals when explicitly requested.
8. Print a compact terminal summary and save a full JSON report.

There must be no GPT-5.5 role or call anywhere in this workflow. If a top-level
judgment is requested, the model must be `gpt-5.6-sol`.

## Important current-state findings

- `tools/model_comparison.py` compares five per-image cells (`2a`, `2b`, `2c`,
  `2d_isolated`, and `2c+2d_coupled`). It does not run the real Pass 2f and
  should not be stretched into this package-level comparison.
- The real multi-image Pass 2f implementation is `run_pass_2f()` in
  `tools/scene_classifier_passes.py`.
- Package-level orchestration already exists in `run_pass_2f_batch()` in
  `tools/rehab_packages.py`.
- `tools/rerun_pass_2f_artifact.py` already resolves a saved
  `photo_intel.json`, reconstructs package candidates, selects photo paths, and
  recomputes v4. Reuse or extract its read-only preparation helpers rather than
  duplicating that logic.
- `run_pass_2f()` currently catches provider/parse errors and returns
  `verification_status="uncertain"`. Comparison mode needs strict failures so
  an API failure is never treated as a model decision.
- The comparison config and analyzer fallback both use
  `unsloth/qwen3.6-27b@q6_k`. The repo `.env` currently does not set
  `LM_STUDIO_MODEL`, so the pipeline fallback remains the source of truth.
- Existing model-comparison work and tests are uncommitted/untracked. Preserve
  all existing changes and avoid unrelated cleanup.

## Approval comparison semantics

For a premium Pass 2f result, define:

```python
approved = verification_status == "confirmed"
```

Classify every VLM-evaluated package into exactly one bucket:

| Terra approved | Sol approved | Bucket |
|---|---|---|
| yes | yes | `both_approved` |
| no | no | `neither_approved` |
| yes | no | `terra_only_approved` |
| no | yes | `sol_only_approved` |

Only `terra_only_approved` and `sol_only_approved` are focus/disagreement
packages. They are the packages the user wants to examine.

Do not use byte equality of raw responses. For the primary comparison, only the
production approval gate matters. Therefore:

- both `confirmed` is agreement even when rationales or issue-id subsets differ;
- both non-confirmed is agreement even when one says `rejected` and the other
  says `uncertain`;
- retain those differing details in JSON even though the package is not in the
  focus set;
- a provider exception, timeout, quota error, or invalid/unparseable response is
  an execution failure, not `neither_approved` and not `uncertain`.

`confirmed_by_rule` is reserved for deterministic turnover packages. No
contestant model
should be called for those packages. Include them in the report as
`rule_confirmed`, but exclude them from contestant agreement/win denominators.
Packages with unsupported rooms, no review images, or no reviewed issue IDs must
also remain in the report under explicit `not_evaluated` reasons.

## Recommended implementation layout

Create a separate package-level tool instead of modifying the per-image engine:

```text
tools/pass_2f_model_comparison.py
tools/pass_2f_comparison_config.py       # only if configuration cannot cleanly reuse ModelSpec
configs/pass_2f_comparison/terra_vs_sol.json
tests/test_pass_2f_model_comparison.py
```

Use `tools/model_comparison_config.ModelSpec` where practical, but do not force
Pass 2f package semantics into the existing `RunSettings.skills` validation.
Keeping a small dedicated profile loader is preferable to destabilizing the
existing comparison CLI.

Suggested profile:

```json
{
  "schema_version": 1,
  "name": "terra_vs_sol_2f",
  "source": {
    "required_local_model": "unsloth/qwen3.6-27b@q6_k",
    "require_local_upstream_routing": true
  },
  "models": {
    "terra": {
      "label": "Terra",
      "provider": "openai",
      "model": "gpt-5.6-terra",
      "reasoning_effort": "low",
      "verbosity": "low"
    },
    "sol": {
      "label": "Sol",
      "provider": "openai",
      "model": "gpt-5.6-sol",
      "reasoning_effort": "low",
      "verbosity": "low"
    },
    "qwen": {
      "label": "Local Qwen",
      "provider": "lmstudio",
      "model": "unsloth/qwen3.6-27b@q6_k",
      "timeout_seconds": 1800
    }
  },
  "executive_review": {
    "enabled": false,
    "provider": "openai",
    "model": "gpt-5.6-sol",
    "only": "asymmetric_approvals",
    "reasoning_effort": "low",
    "verbosity": "low"
  },
  "run": {
    "max_images_per_package": 3,
    "contestant_order": ["terra", "sol"],
    "include_qwen": false,
    "fail_fast": true
  }
}
```

Credentials and endpoints stay environment-only. Do not store `OPENAI_API_KEY`,
base URLs, or resolved secrets in profiles, checkpoints, logs, or reports.

## CLI contract

The first implementation should consume a completed local artifact instead of
duplicating the analyzer. A normal local run should be performed with Pass 2f
disabled, then its artifact supplied to the comparator.

```powershell
# Redacted validation and package-count preview; no provider calls.
.\.venv\Scripts\python.exe tools\pass_2f_model_comparison.py `
  --profile terra_vs_sol_2f `
  --run PATH_TO_RUN_OR_PHOTO_INTEL_JSON `
  --validate-config

# Execute Terra and Sol Pass 2f and write the full report.
.\.venv\Scripts\python.exe tools\pass_2f_model_comparison.py `
  --profile terra_vs_sol_2f `
  --run PATH_TO_RUN_OR_PHOTO_INTEL_JSON

# Add local Qwen as an auxiliary verifier. All three streams start together.
.\.venv\Scripts\python.exe tools\pass_2f_model_comparison.py `
  --profile terra_vs_sol_2f `
  --run PATH_TO_RUN_OR_PHOTO_INTEL_JSON `
  --include-qwen

# Resume without repeating completed model calls.
.\.venv\Scripts\python.exe tools\pass_2f_model_comparison.py `
  --profile terra_vs_sol_2f `
  --run PATH_TO_RUN_OR_PHOTO_INTEL_JSON `
  --resume

# Optional Sol executive review of asymmetric approvals only.
.\.venv\Scripts\python.exe tools\pass_2f_model_comparison.py `
  --profile terra_vs_sol_2f `
  --run PATH_TO_RUN_OR_PHOTO_INTEL_JSON `
  --resume --executive-review
```

Support at minimum:

- `--profile NAME_OR_PATH`
- `--run RUN_DIR_OR_PHOTO_INTEL_JSON`
- `--output PATH`
- `--validate-config`
- `--resume`
- `--executive-review`
- `--include-qwen` (off by default)
- `--max-images-per-package`
- `--terra-model` and `--sol-model` for explicit one-off overrides

Do not add a GPT-5.5 option. Do not add automatic fallback models. A
one-command wrapper that launches `analyzer_cli.py` can be considered later;
artifact-first comparison is safer and guarantees upstream work is not rerun.

## Source-artifact validation and frozen cases

Resolve either a run directory or `photo_intel.json` using the existing replay
tool's conventions. Before any paid call:

1. Hash the source artifact and issue catalog.
2. Verify that `run.default_local_model` is the required Qwen model.
3. Inspect `model_routing` and reject artifacts where an upstream model-routed
   pass used a premium provider/model. Ignore an existing `2f` routing entry;
   comparison recomputes 2f and never trusts or patches that result.
4. Reconstruct estimate candidates and package candidates once.
5. Select review photo keys/paths and filter evidence items once per package.
6. Persist these prepared cases in the checkpoint before calling either model.

Both contestants must receive identical:

- package id/type/label and room;
- evidence items and valid issue IDs;
- ordered review photo keys and paths;
- prompt-template version and maximum image count.

Refactor `run_pass_2f_batch()` to accept prepared cases, or extract a reusable
`prepare_pass_2f_cases()` helper and call `run_pass_2f()` for each contestant.
Do not independently select evidence/photos inside the Terra and Sol phases.

The source `photo_intel.json` must remain read-only. Unlike
`rerun_pass_2f_artifact.py`, the comparison command must not patch the production
artifact or create a replay backup.

## Strict Pass 2f errors

Add an opt-in strict mode to `run_pass_2f()` and the relevant batch wrapper,
while preserving current production behavior by default:

```python
async def run_pass_2f(..., strict: bool = False) -> Pass2fResult:
    ...
```

In strict mode:

- provider exceptions and timeouts propagate;
- missing or invalid JSON raises `Pass2fInvalidResponseError`;
- quota/rate-limit failures are not retried with another model;
- a model-returned, schema-valid `verification_status="uncertain"` remains a
  legitimate model decision;
- the comparator saves the checkpoint/error context and exits non-zero.

Never map an execution error to a non-approval bucket. Never run Qwen, Sol, or
another premium model as a fallback.

## Optional Sol executive review

There is no routine judge phase. When `--executive-review` is enabled, call a
fresh `gpt-5.6-sol` request only for packages in `terra_only_approved` or
`sol_only_approved`.

The executive review should:

- receive the original prepared images and Qwen-derived package evidence;
- receive both contestant decisions as deterministically randomized
  `Decision A` and `Decision B`;
- never receive Terra/Sol labels, model IDs, profile names, or contestant order;
- return a final operational decision of `approve`, `reject`, or
  `needs_manual_review`, plus concise reasoning and supported/rejected issue IDs;
- be canonicalized only after the response;
- be recorded explicitly as a non-independent review because the executive
  model is also one contestant.

The executive result decides neither the comparison bucket nor the historical
contestant output. It is an additional recommendation stored on the disputed
package. If the Sol executive call fails, fail without fallback.

## Full JSON report

The terminal can emphasize disagreement counts, but the JSON must include every
package candidate and complete details. Suggested shape:

```json
{
  "schema_version": 2,
  "config": {},
  "config_fingerprint": "...",
  "source": {
    "artifact_path": "...",
    "artifact_sha256": "...",
    "catalog_sha256": "...",
    "property_id": "...",
    "local_model": "unsloth/qwen3.6-27b@q6_k",
    "model_routing": []
  },
  "aggregate": {
    "package_candidate_count": 0,
    "vlm_evaluated_count": 0,
    "qwen_evaluated_count": 0,
    "qwen_approved_count": 0,
    "both_approved": 0,
    "neither_approved": 0,
    "terra_only_approved": 0,
    "sol_only_approved": 0,
    "rule_confirmed": 0,
    "not_evaluated": 0,
    "focus_package_count": 0
  },
  "phase_stats": {
    "terra": {},
    "sol": {},
    "qwen": {},
    "executive_review": {}
  },
  "packages": [
    {
      "package_id": "...",
      "package_type": "...",
      "package_label": "...",
      "room": "...",
      "source_package": {},
      "prepared_input": {
        "evidence_items": [],
        "review_photo_keys": [],
        "review_image_paths": [],
        "reviewed_issue_ids": []
      },
      "terra": {
        "verification_status": "confirmed",
        "approved": true,
        "confirmed_issue_ids": [],
        "rejected_issue_ids": [],
        "evidence_summary": "...",
        "visible_room_count": "unclear",
        "visible_room_count_evidence": "...",
        "raw_response": "...",
        "wall_time_seconds": 0.0,
        "usage": {}
      },
      "sol": {},
      "qwen": {},
      "comparison": {
        "bucket": "terra_only_approved",
        "is_focus": true
      },
      "executive_review": null,
      "not_evaluated_reason": null
    }
  ],
  "errors": []
}
```

Preserve full parsed and raw contestant output even for `both_approved` and
`neither_approved`. Include usage deltas and wall time per package/model when
available, plus aggregate calls and input/output/total tokens. Do not estimate
dollar cost from a hard-coded pricing table.

The terminal summary should print the aggregate counts, then list only the
asymmetric package IDs/types with Terra and Sol statuses. Always print the full
report path.

## Checkpoint and resume

Premium calls make resume important. Save atomically after every prepared case,
contestant result, and executive result. The fingerprint must cover:

- redacted profile and CLI overrides;
- source artifact hash and issue-catalog hash;
- prepared-case input hashes;
- Pass 2f prompt/template version;
- model IDs and generation controls;
- maximum images per package and strict-mode setting;
- the optional-Qwen flag and fixed local Qwen model.

Resume must refuse any fingerprint mismatch. If Terra completed and Sol failed,
resume should reuse Terra results and continue with Sol. A failed provider call
must not be marked complete.

## Local-model default correction

Keep the real analyzer fallback in `tools/pipeline_config.py` aligned:

```python
LM_STUDIO_MODEL = os.environ.get(
    "LM_STUDIO_MODEL",
    "unsloth/qwen3.6-27b@q6_k",
)
```

Keep environment precedence. Update relevant documentation/tests and verify the
artifact records this exact value in `run.default_local_model` and upstream
`model_routing`. Do not modify `.env` or store endpoint/credential values.

## Tests

Add focused tests covering:

- profile loading, CLI precedence, validation, and secret redaction;
- actual analyzer fallback is Qwen when `LM_STUDIO_MODEL` is absent;
- source artifact must have local-Qwen upstream routing;
- source artifact and catalog are never modified;
- package candidates/evidence/photo order are prepared once and are byte-for-byte
  identical for Terra, Sol, and optional Qwen calls;
- enabling Qwen starts all three independent streams before any gated call completes;
- LM Studio receives all frozen package images in one request;
- only Pass 2f is invoked for contestants; no 1a-2e contestant calls;
- all four approval buckets;
- `confirmed_by_rule` and not-evaluated cases are preserved but excluded from
  contestant denominators;
- every package remains in JSON, including both-approved and neither-approved;
- raw responses, parsed fields, image references, timings, and usage are retained;
- provider/timeout/parse errors fail and never become `uncertain` or a
  non-approval bucket;
- a schema-valid model-returned `uncertain` remains a valid non-approval;
- no GPT-5.5 model or environment variable is consulted;
- executive review is disabled by default and runs only for asymmetric approvals
  when requested;
- executive prompts contain no Terra/Sol/model/profile identity and order is
  deterministic;
- executive model is exactly `gpt-5.6-sol` and has no fallback;
- checkpoint fingerprint mismatch, partial resume, and atomic writes;
- mocked end-to-end output retains all packages while the terminal focus list
  contains only asymmetric approvals.

Run at least:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_pass_2f_model_comparison.py `
  tests\test_package_pass_2f_and_vlm.py `
  tests\test_rehab_packages.py `
  tests\test_rerun_pass_2f_artifact.py -q
```

Use a writable `--basetemp` because this Windows workspace may deny pytest's
default temporary directory inside the sandbox.

## Acceptance criteria

Implementation is complete when:

1. Redacted validation against a local Qwen artifact makes no provider calls and
   reports the exact number of VLM-eligible, rule-confirmed, and not-evaluated
   packages.
2. A default mocked end-to-end run calls Terra and Sol only through Pass 2f;
   an opt-in run starts Terra, Sol, and Qwen concurrently with identical prepared
   inputs while preserving Terra/Sol comparison semantics.
3. The report contains every package and complete contestant details, even when
   all packages are both-approved or neither-approved.
4. The terminal highlights only Terra-only and Sol-only approvals.
5. No normal judge calls occur. Optional executive review calls only
   `gpt-5.6-sol` and only for asymmetric approvals.
6. Any contestant or executive provider failure produces a non-zero exit and a
   resumable checkpoint, with no fallback and no false `uncertain` result.
7. The source artifact is unchanged.
8. The real pipeline fallback local model is
   `unsloth/qwen3.6-27b@q6_k`.

## Out of scope

- Running Sol or Terra on passes other than 2f.
- A GPT-5.5 judge or any GPT-5.5 request.
- Automatically applying a contestant or executive result to the production
  property artifact.
- A GUI. The JSON schema should be stable enough for the existing GUI handoff to
  add a dedicated 2f report view later.
- Dollar-cost estimates based on hard-coded model pricing.
