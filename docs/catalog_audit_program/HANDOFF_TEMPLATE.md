# Catalog audit stage handoff template

Copy this file into a stage-specific handoff when a session finishes. Keep it concise and point to machine-readable artifacts instead of copying large datasets into Markdown.

Suggested name: `docs/catalog_audit_program/HANDOFF_SESSION_<N>.md`.

---

# Handoff — catalog audit session <N>

Date:

Session/task title:

## Outcome

State what was completed and whether the session met its exit criteria.

## Repository state

- Starting commit:
- Ending commit:
- Branch/worktree:
- Dirty state at start:
- Dirty state at end:
- Pre-existing changes preserved:

## Inputs verified

| Input | Expected identity/hash | Actual identity/hash | Result |
|---|---|---|---|
| | | | |

Record every required input and explain mismatches. Do not silently update a frozen baseline.

## Files created or changed

| Path | Status | Purpose |
|---|---|---|
| | | |

Separate this session's changes from pre-existing user changes.

## Work completed

- TBD

## Decisions and invariants

Record only decisions the next session must preserve.

- TBD

## Deviations from the session brief

| Deviation | Reason | Impact |
|---|---|---|
| | | |

Write `None` if there were no deviations.

## Commands and verification

| Command/check | Result | Notes |
|---|---|---|
| | | |

Include exact pass/fail/skip counts where applicable. Distinguish assertion failures from environment or sandbox failures.

## Outputs for the next session

| Artifact | Hash/commit | Contract |
|---|---|---|
| | | |

## Unresolved questions and risks

- TBD

## Required next task

Objective:

Required inputs:

Authorized changes:

Prohibited changes:

Exit criteria:

## Do not redo

- Do not regenerate frozen evidence unless a new audit baseline is explicitly approved.
- Do not repeat completed joins or semantic reviews whose output hashes match this handoff.
- Add task-specific prohibitions here.

## Suggested opening prompt for the next task

> Continue the catalog-audit program from this handoff. Read `docs/catalog_audit_program/00_OVERALL_CONTEXT.md`, the next session brief, and the machine-readable artifacts listed above. First verify the expected commit and hashes, then draft a task-specific implementation plan. Do not repeat completed work or expand beyond the next session's authority. End by producing the next stage handoff.
