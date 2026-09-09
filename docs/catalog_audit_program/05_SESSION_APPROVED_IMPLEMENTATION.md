# Session 4 brief — approved surgical implementation

## Mission

Implement only the catalog changes explicitly authorized by the completed human approval manifest. Regenerate the deterministic artifacts and produce a clean candidate handoff for independent validation.

This session is not allowed to reinterpret proposals or make opportunistic improvements.

## Hard prerequisites

Do not begin implementation until all exist and reconcile:

1. `docs/catalog_audit_program/00_OVERALL_CONTEXT.md`
2. This brief.
3. `docs/catalog_audit_program/HANDOFF_SESSION_3.md`
4. `reports/catalog_audit_proposals.json`
5. `reports/catalog_audit_adversarial_review.json`
6. `reports/catalog_audit_approvals.json`
7. Exact proposal-baseline Git commit and source hashes.

If no proposal is approved, stop with a no-change handoff. If an approved record requires unsupported `add`, `merge`, gate, economic, generator-schema, or product-policy work that was not separately authorized, stop and report the blocked approval rather than improvising.

## First action

Verify the approval manifest and checkout. Produce a task-specific implementation plan listing each approved proposal ID, exact intended file diff, generator command, tests, and rollback boundary. The plan must distinguish pre-existing user changes from this session's files.

## Worktree/branch posture

Use a clean candidate branch or worktree based on the exact approved proposal baseline. Preserve a clean baseline reference for Session 5. Do not implement directly on an unrelated dirty checkout.

## Authorized changes

Normally, only these production artifacts may change:

- `tools/catalog_migrations/kind_v2_decisions.json`
- `tools/issue_catalog_kind_v2.json` through normal regeneration.
- `tools/catalog_migrations/2.1_to_3.0.json` through normal regeneration.
- `tools/catalog_migrations/2.1_to_3.0_audit.md` through normal regeneration.
- Focused tests necessary to pin an approved semantic or migration invariant.
- `docs/catalog_audit_program/HANDOFF_SESSION_4.md`

Changes to the generator or validator are prohibited unless the human manifest explicitly authorizes a separately reviewed migration-system proposal with exact scope.

## Prohibited changes

- No unapproved wording cleanup.
- No new catalog proposal.
- No Pass 2a–2e, Terra, projection, costing, packaging, quarantine, or prompt change.
- No price calibration.
- No live provider run.
- No manual edit to generated catalog/manifest/audit files.
- No publication or default-selector change.

## Implementation procedure

For each approved proposal:

1. Confirm its approved diff still applies to the exact baseline.
2. Apply the smallest decisions-file change.
3. Regenerate with the repository's normal migration command.
4. Compare the produced semantic/economic/retrieval diff with the approved proposal.
5. Add or update a focused test only where it pins the approved invariant.
6. Record the proposal-to-file mapping for validation.

Do not partially implement an approved cluster if its semantic coherence depends on the full approved set. If proposals conflict when combined, stop for renewed human review.

## Required safeguards

- The generated file must equal a fresh generator run byte-for-byte.
- No item ID/order change may occur unless explicitly approved.
- No economic field change may occur unless separately authorized.
- Split successors must retain and report their approved inherited economics.
- Retirements must match the approved lost-route/package/economic analysis.
- Package routing changes must use the existing decisions section when applicable.
- The current v2 selector must remain the selection path; do not add `ISSUE_CATALOG_PATH` testing behavior.

## Verification

Run at least:

- Generator twice, comparing outputs.
- `tests/test_catalog_kind_v2.py`.
- Catalog validation tests relevant to the approved fields.
- Retrieval/scene tests relevant to affected items.
- Focused new tests.

Session 5 owns broad independent validation, but Session 4 must not hand off an internally inconsistent candidate.

## Required handoff contents

`HANDOFF_SESSION_4.md` must include:

- Baseline and candidate commits.
- Approval, proposal, and evidence hashes.
- Approved proposal IDs implemented.
- Exact changed files.
- Per-proposal semantic and economic diff summary.
- Generator/parity/test results.
- Any approved proposal not implemented and why.
- Baseline/candidate worktree instructions for Session 5.

## Exit criteria

- Every approved proposal is implemented exactly, explicitly blocked, or returned for review.
- Every rejected/deferred/non-catalog proposal remains untouched.
- Generated artifacts are deterministic and in parity.
- No unrelated runtime behavior changed.
- Candidate commit is clean and identified.
- `HANDOFF_SESSION_4.md` is complete.

## Suggested opening prompt

> Execute Session 4 of the catalog-audit program. Read the overall context, this brief, the Session 3 handoff, and the human approval manifest. Verify all proposal/review/approval hashes and begin with an exact implementation plan. Implement only approved native decisions-file changes, regenerate the catalog normally, pin focused invariants, and perform local parity checks. Do not reinterpret approvals, change unrelated runtime behavior, run live providers, or publish. Finish with a clean candidate commit and `HANDOFF_SESSION_4.md`.

