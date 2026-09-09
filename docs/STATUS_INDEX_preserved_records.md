# Status index for preserved records

Written as part of the repository preservation commit (2026-09-08). This file
describes the **status** of records that were previously untracked. It does not
alter any of them: historical documents keep their original wording, including
conclusions later superseded. Read a document's own date and this table
together — do not assume a later decision existed when an earlier doc was written.

## Why these files were untracked until now

The catalog audit program (Sessions 1-7) deliberately deferred the tracking
decision to `docs/HANDOFF_git_repository_audit.md`, so its outputs accumulated
outside version control. That audit is itself included here.

## Categories

| Status | Meaning |
|---|---|
| **approved** | A decision Steven ruled on. Binding until explicitly revisited. |
| **proposed** | Put forward, not ruled on. Carries no authority. |
| **superseded** | Correct when written; a later record replaces it. |
| **historical** | A record of what happened. Not a statement of current state. |

## Key records

| Path | Status | Note |
|---|---|---|
| `reports/catalog_audit_approvals.json` | approved | Session 4 approval set. |
| `reports/catalog_audit_approvals_v2.json` | approved | Supersedes v1 for the checkpoint. |
| `docs/DECISION_RECORD_catalog_checkpoint_20260908.md` | approved | D1-D8, Session 7. |
| `docs/catalog_audit_program/PROPOSAL_CAP-022_trim_subject_gate.md` | proposed | Declined; retained for the reasoning. |
| `reports/catalog_audit_live_experiment_manifest.json` | historical | `status: awaiting_human_cost_authorization`. Never executed. See drift note below. |
| `reports/catalog_audit_live_experiment_manifest_v2.json` | superseded | Different, smaller shape; no case-level artifact pins. |
| `docs/HANDOFF_git_repository_audit.md` | historical | Facts-only audit that prompted this commit. Its measurements are as-of 2026-09-06/08. |
| `docs/catalog_audit_program/HANDOFF_SESSION_*.md` | historical | One per session; later sessions supersede earlier conclusions. |

## Two known drifts, recorded rather than repaired

**1. `scripts/catalog_audit_validation.py` no longer matches its own pin.**
`reports/catalog_audit_live_experiment_manifest.json` pins
`inputs_pinned.validation_script.sha256 = dbacf929...`. The file's current
sha256 is `7b9d1e54...`. Cause: the S7-3 change that made the harness pins
rebindable via `--manifest` (see the comment at `scripts/catalog_audit_validation.py`
above `DEFAULT_PINS`), made after the manifest was frozen. The manifest's own
`frozen_inputs_rule` says such a change invalidates it and requires a new one.
Not repaired here: reverting S7-3 would remove a real improvement, and editing a
frozen manifest would destroy the record. The other seven pinned inputs still
verify.

**2. The 22 frozen input artifacts are not in this repository.**
Every `cases[].artifact_path` in the live manifest is an absolute path into
`C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts/` — a separate
repository — totalling 65.8 MiB. They are not committed here: they belong to
another repo, and importing them would add permanent object-store weight to the
repository this cleanup is shrinking. All 22 verified against their pinned
sha256 on 2026-09-08 and are archived, with a restoration manifest, at
`D:\realtorvision-recovery\20260908\frozen-inputs\`.

Run directories are likewise not committed, consistent with the existing
`.gitignore` policy: reports are versioned, runs are not.

## Line-ending handling

`.gitattributes` now pins byte behaviour for evidence records, because their
hashes are of exact bytes:

- Newly tracked `reports/**/*.json` and `*.jsonl` are `-text` — stored verbatim,
  reproduced identically on any platform.
- Ten files that pre-date this commit are `text eol=crlf`. They are stored as LF
  blobs and their pinned hashes are of the CRLF bytes `core.autocrlf=true`
  produces on checkout; forcing CRLF keeps those pins valid off-Windows.
  **Do not change these to `-text`** — that flips the working tree to LF and
  breaks all ten.
