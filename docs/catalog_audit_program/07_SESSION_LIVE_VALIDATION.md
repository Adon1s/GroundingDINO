# Session 6 brief — cost-gated live validation

## Mission

Run the approved live causal evaluation from frozen pass-2c outputs and determine whether the catalog candidate should proceed, remain inconclusive, or be rejected. This session owns Tier 3 only.

It is an evaluation and recommendation task, not an implementation or publication task.

## Hard prerequisites

Do not make provider calls until all are true:

1. `docs/catalog_audit_program/00_OVERALL_CONTEXT.md` has been read.
2. This brief has been read.
3. `docs/catalog_audit_program/HANDOFF_SESSION_5.md` reports Tier 1/2 pass.
4. `reports/catalog_audit_live_experiment_manifest.json` is complete and hash-pinned.
5. Baseline and candidate commits still match the manifest.
6. The human has explicitly authorized the targeted run budget.
7. The human has separately authorized a full canary budget if the targeted-first gate reaches that point.

If any prerequisite fails, stop without calls and report what is missing.

## First action

Verify the manifest and authorizations. Produce a run plan that lists the exact targeted cases, stages, model routes, repetitions, metrics, stopping conditions, and maximum authorized spend/tokens. Do not broaden the case set or change prompts to improve results.

## Authorized changes

Create evaluation artifacts and reports only, expected to include:

- `reports/catalog_audit_live_validation.json`
- `docs/RESULT_catalog_audit_live_validation_<date>.md`
- Raw run artifacts in the repository's established artifact location.
- `docs/catalog_audit_program/HANDOFF_SESSION_6.md`

Minimal run-wrapper fixes are allowed only for an operational defect that prevents the pinned experiment from executing and does not change experiment semantics. Record and validate such a fix before resuming calls.

## Prohibited changes

- Do not edit decisions, catalog semantics, proposals, or approvals.
- Do not change Pass 2a, 2b, or 2c.
- Do not change unrelated prompts, model routes, thresholds, product policy, pricing, or packages.
- Do not select new cases after seeing outcomes.
- Do not publish or change production defaults.
- Do not rerun automatically beyond the authorized repetition/cost limit.
- Do not evaluate or decide Catalog 3.2 acceptance inside this session; the baseline arm's package outputs are that thread's data and are handed off, not judged here.

## Experimental posture

The primary causal replay begins from frozen pass-2c outputs and reruns:

- Pass 2d resolution.
- Condition projection as required.
- Terra verification.
- Relevant disposition/routing/package/cost projection needed to evaluate downstream consequences.

Pass 2a, 2b, and 2c remain frozen. Baseline and candidate use the same inputs, prompts, models, and orchestration code except for the approved catalog decisions/generated artifacts.

Stored 3.1 replicas may inform historical variance. They do not substitute for a fresh 3.2 control when package/economic behavior is part of the conclusion.

## Stage A — targeted cases

Run the manifest's affected cases and controls first. Compare:

- Resolved catalog item.
- Candidate neighborhood if recorded.
- Terra verdict and rationale.
- Final disposition.
- Accepted/rejected condition state.
- Package/route/cost consequences where relevant.
- Error migration into neighboring items or stages.

Use the manifest's repetition rule for borderline cases only. Do not repeat clear failures until they pass.

Stop before a full canary if:

- Targeted behavior does not improve as predicted.
- Reviewed successes or correct rejections regress.
- A new false positive or mapping hijack appears.
- The result is within the defined noise band without an authorized repeat.
- An experiment invariant changes.
- The budget boundary is reached.

## Stage B — full frozen canary

Proceed only if targeted cases pass and full-canary cost is explicitly authorized.

Batch all approved changes in one 18-property frozen canary rather than running one canary per proposal. Compare baseline and candidate on:

- Resolved item changes.
- Terra verdict/rationale changes.
- Disposition and accepted-set changes.
- New misses and false positives.
- Package and total changes.
- Neighbor/error migration.
- Run failures and incomplete cases.

A second full replica is not automatic. Request or use it only when already authorized and the first result is near the known noise floor or otherwise inconclusive.

## Decision rubric

Return one program recommendation:

- `publish_candidate_supported`
- `candidate_rejected`
- `inconclusive_repeat_authorization_needed`
- `experiment_invalid`

`publish_candidate_supported` requires:

- Targeted improvement on reviewed evidence.
- No material loss of successful uses/correct rejections.
- No new false-positive behavior.
- No unexpected mapping theft or error migration.
- Tier 1/2 invariants still hold.
- A behavioral effect distinguishable from expected variance.

This recommendation does not itself authorize publication. Package and total deltas are reported for context; the standing priority is fewer hallucinations and fewer lost real conditions (overall context, Section 1).

## Reporting

The report should include:

- Exact manifest and source hashes.
- Provider/model identifiers and call counts.
- Token/cost totals versus authorization.
- Targeted-case results.
- Canary results if run.
- Variance and repetition analysis.
- Regressions and error migration.
- Final recommendation and confidence.
- Location of the preserved baseline-arm package outputs (Sol decisions, candidates, v4/v5 totals) for the Catalog 3.2 acceptance thread.
- Remaining risks and any separately scoped next work.

## Exit criteria

- No run exceeded its authorization.
- Every executed case reconciles to the pinned manifest.
- Baseline/candidate isolation is demonstrated.
- The recommendation follows the predeclared rubric.
- No implementation or publication change occurred.
- `HANDOFF_SESSION_6.md` is complete and states whether an optional publication task is justified.

## Suggested opening prompt

> Execute Session 6, the cost-gated live validation for the catalog-audit candidate. Read the overall context, this brief, `HANDOFF_SESSION_5.md`, and the pinned live experiment manifest. Verify the baseline/candidate identities and explicit budget authorization, then write an exact run plan. Start from frozen pass-2c outputs, keep all unrelated stages/prompts/models fixed, and run targeted cases before any separately authorized full canary. Do not modify the candidate or publish it. Produce a variance-aware recommendation and finish with `HANDOFF_SESSION_6.md`.

