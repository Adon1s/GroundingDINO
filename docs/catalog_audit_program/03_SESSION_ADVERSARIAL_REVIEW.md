# Session 3 brief — independent adversarial review

## Mission

Independently challenge every Session 2 proposal, no-change conclusion, and migration-system-gap classification. The goal is to catch incorrect fix ownership, weak evidence, regression risk, and unexpressible changes before human approval.

This should be a fresh Codex task. Do not fork or import Session 2's full conversational reasoning if it can be avoided. The structured evidence, proposal artifact, generated proposal document, and handoff are sufficient context.

## Required context

Read:

1. `docs/catalog_audit_program/00_OVERALL_CONTEXT.md`
2. This brief.
3. `docs/catalog_audit_program/HANDOFF_SESSION_2.md`
4. `reports/catalog_audit_evidence.json`
5. `reports/catalog_audit_proposals.json`
6. The generated proposal Markdown.
7. Source cases/photos only where needed to verify or refute a proposal.
8. Generator, validator, retrieval, and runtime source relevant to challenged claims.

Do not treat Session 2's narrative as authoritative when machine evidence or current code disagrees.

## First action

Verify all proposal and evidence hashes and produce a concise review plan. The plan should group review work by risk while still guaranteeing that every proposal ID receives a disposition.

## Authorized changes

Create only independent review artifacts, expected to include:

- `reports/catalog_audit_adversarial_review.json`
- `docs/REVIEW_catalog_audit_adversarial_<date>.md`
- Focused tests only if deterministic review reconciliation tooling is added.
- `docs/catalog_audit_program/HANDOFF_SESSION_3.md`

Do not silently modify the proposal source. Corrections should be recommendations tied to proposal IDs.

## Prohibited changes

Do not modify:

- `reports/catalog_audit_proposals.json`
- The generated proposal document.
- Migration decisions or generated catalogs.
- Runtime behavior, prompts, pricing, packages, or product policy.
- Session 1 evidence.

## Challenge rubric

For every proposal, try to disprove at least the following:

### Existing coverage

- Is an adequate existing item already present?
- Was it present in the frozen candidate set?
- Is the issue retrieval reachability or pass-2d selection rather than missing/incorrect catalog semantics?

### Upstream and downstream ownership

- Is the upstream kind incorrect?
- Did pass 2c distort or drop the observation?
- Did pass 2d select incorrectly despite adequate candidates?
- Did Terra rule on the wrong detail despite an adequate claim?
- Does projection, tiering, routing, packaging, or costing own the behavior?

### Evidence quality

- Are claimed independent cases actually the same property, image, or condition?
- Is factorized/gold evidence genuinely independent?
- Were supporting photos inspected?
- Are success cases and correct rejections fairly represented?
- Is the result larger than known reviewer/model noise?
- Does a single-case exception demonstrate a universal structural fact?

### Semantic regression

- Does proposed wording introduce unsupported material, object, mechanism, severity, location, or scope commitments?
- Does broadening create ambiguity or allow irrelevant observations?
- Does narrowing exclude current valid successes?
- Does a split create overlapping successors?
- Could changed embedding/support/deny/scene fields steal mappings from neighbors?
- Could lexical shortcut behavior change unexpectedly?

### Migration and economics

- Is the operation actually supported by the current generator/schema?
- Is an `add`, `merge`, gate, or economic change mislabeled as native?
- Are split-inherited economics semantically defensible?
- Does retirement remove work-item, package, route, or pricing behavior not acknowledged in the proposal?
- Does the change cross product-quarantine or package-policy boundaries?

## Required disposition

Every proposal ID receives one of:

- `sustained`
- `sustained_with_conditions`
- `reassign_non_catalog`
- `insufficient_evidence`
- `reject`

Each disposition must include:

- Challenge summary.
- Evidence inspected.
- Strongest counterargument.
- Resolution.
- Required modification or condition, if any.
- Residual risk.

Also review no-change/deferred clusters for false negatives. A reviewer may recommend reopening one, but must use the same evidence standard.

## Verification

- Reconcile exactly one review disposition per proposal ID.
- Flag orphan/missing IDs.
- Verify every review claim points to an evidence or source reference.
- Confirm that no proposal/catalog/decision artifact changed.
- Summarize counts by disposition and risk category.

## Exit criteria

Session 3 is complete when:

- Every proposal has one adversarial disposition.
- All high-risk semantic/economic/retrieval questions were explicitly challenged.
- Reopened no-change/deferred records are separately identified.
- The proposal and implementation artifacts remain untouched.
- The review artifact is sufficient for the human gate.
- `HANDOFF_SESSION_3.md` is complete.

## Suggested opening prompt

> Perform Session 3, the independent adversarial review of the catalog-audit proposals. Start fresh from the overall context, this brief, `HANDOFF_SESSION_2.md`, and the hash-verified evidence/proposal artifacts. Draft a review plan, then try to disprove catalog ownership, evidence sufficiency, semantic safety, migration expressibility, and economic/package assumptions for every proposal. Do not edit the proposals, catalog, decisions, prompts, or runtime code. Produce one disposition per proposal and finish with `HANDOFF_SESSION_3.md`.

