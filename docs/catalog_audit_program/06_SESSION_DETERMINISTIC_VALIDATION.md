# Session 5 brief — deterministic and retrieval validation

## Mission

Independently validate the approved candidate using clean baseline and candidate arms. This session owns baseline/candidate isolation plus Tier 1 and Tier 2 validation. It must produce the pinned experiment manifest required for any live evaluation.

No live LLM/provider calls are authorized by this brief.

## Required context

Read:

1. `docs/catalog_audit_program/00_OVERALL_CONTEXT.md`
2. This brief.
3. `docs/catalog_audit_program/HANDOFF_SESSION_4.md`
4. `reports/catalog_audit_approvals.json`
5. `reports/catalog_audit_proposals.json`
6. `reports/catalog_audit_evidence.json`
7. Candidate commit and clean baseline commit.

Read Session 4's conversational history only if necessary; validation should be grounded in the approved artifacts and actual diff.

## First action

Verify clean baseline/candidate worktrees and all required hashes. Produce a validation plan that identifies affected items, supporting cases, success/counterexample controls, commands, sidecar requirements, expected output schemas, and pass/fail criteria.

## Authorized changes

This session may create minimal validation tooling and reports, expected to include:

- `reports/catalog_audit_validation_tier12.json`
- `docs/RESULT_catalog_audit_validation_tier12_<date>.md`
- `reports/catalog_audit_live_experiment_manifest.json`
- Focused replay/comparison script and tests only if existing tooling cannot express the required arm comparison.
- `docs/catalog_audit_program/HANDOFF_SESSION_5.md`

Prefer extending or wrapping existing canary/retrieval comparators narrowly. Do not rewrite the broader evaluation harness.

## Prohibited changes

- Do not change approved decisions or proposal semantics.
- Do not change prompts, model routes, Pass 2a–2e behavior, Terra logic, product policy, pricing, or packages.
- Do not run provider-backed live evaluation.
- Do not publish or cut over.

If validation finds a semantic implementation defect, fail the candidate and return it to human/Session 4 review. Do not fix it inside the validation task.

## Arm isolation

Build two clean arms from the approved baseline:

- Baseline: unmodified as-generated 3.2.
- Candidate: approved decisions plus regenerated catalog artifacts.

Record per arm:

- Git commit.
- Decisions hash.
- Catalog hash/version/ontology.
- Generator hash.
- Prompt and relevant runtime hashes.
- Selector settings.
- Python/dependency/test environment.

Do not use `ISSUE_CATALOG_PATH`. Use the normal observation-kind-v2 selector and catalog generation path.

## Tier 1 — deterministic validation

Perform:

- Fresh regeneration twice in the candidate arm.
- Byte parity against committed generated artifacts.
- Explicit v2 catalog validation; do not rely on `python -m tools.catalog_validation` if that entry point validates v1 by default.
- Targeted migration/catalog/retrieval/scene tests.
- Full test suite using a known writable base temp.
- Approved-versus-actual semantic diff.
- Economic/package/route diff.
- Item count, identity, and ordering comparison.
- In-memory reviewed diff versus committed implementation comparison.

The only expected production diffs are the approved decisions and corresponding generated artifacts, plus focused tests explicitly added by Session 4.

## Tier 2 — retrieval replay

Before replay, probe the embeddings sidecar with a real request. A successful `/health` response is not sufficient.

Replay through baseline and candidate for:

- Every proposal-supporting observation.
- Every successful-use control for affected items.
- Every correct-rejection/negative control selected in the proposal.
- Neighbor items identified as overlap risks.

Use the exact frozen observation text, kind, scene group, and relevant metadata. Capture:

- Candidate IDs and ordering.
- Ranks and scores.
- Kind/scene filtering.
- Guardrail outcomes.
- Support/deny effects.
- Lexical-shortcut decision and inputs where available.
- Selected resolution only if the replay remains provider-free; do not invoke live pass 2d.

Evaluate:

- Intended reachability improvement or preservation.
- Successful-use preservation.
- Correct-rejection preservation.
- Neighbor hijacking.
- New ambiguity.
- Negative-control regressions.

The saturated catalog-resolution benchmark may run as a gross regression check but must not replace affected-observation replay.

## Live experiment manifest

If Tiers 1 and 2 pass, produce a manifest pinning:

- Baseline and candidate identities.
- Approved proposal IDs.
- Targeted runtime cases.
- Success/counterexample controls.
- Frozen pass-2c observations, kinds, scenes, photo references, and metadata.
- Model map and prompt/code hashes.
- Required stages to rerun.
- Metrics and comparison schema.
- Targeted-first ordering.
- Full-canary authorization requirement.
- Cost estimate/budget field left for human authorization.
- Stop conditions.

The manifest must make it impossible for Session 6 to silently change Pass 2a, 2b, 2c, prompts, model routing, or the case set.

## Pass/fail rules

Fail the candidate for:

- Any unapproved production diff.
- Generator/parity/validation failure.
- Unexpected item identity/order/economic change.
- Loss of intended reachability.
- Reviewed success or correct-rejection regression.
- Material neighbor hijacking or negative-control regression.
- Manifest inputs that cannot be pinned reproducibly.

Environment failures should be distinguished from candidate failures, repaired only within validation tooling/environment scope, and rerun.

## Exit criteria

- Tier 1 and Tier 2 have explicit pass/fail results.
- Actual diff matches the human-approved diff.
- All affected/support/control cases are reconciled.
- No live provider call occurred.
- A complete live manifest exists only if the candidate passed.
- `HANDOFF_SESSION_5.md` contains exact commands, results, hashes, and the live cost gate.

## Suggested opening prompt

> Execute Session 5 of the catalog-audit program. Read the overall context, this brief, `HANDOFF_SESSION_4.md`, the approval manifest, and the evidence/proposal artifacts. Verify clean baseline and candidate worktrees, then write a task-specific validation plan. Perform only Tier 1 deterministic and Tier 2 provider-free retrieval validation. Do not alter approved semantics or run live LLM evaluation. If the candidate passes, produce a fully pinned live experiment manifest; otherwise return a failure report. Finish with `HANDOFF_SESSION_5.md`.

