# Implementation plan - renovation scope and estimate architecture

Status: approved direction; implementation not started.

Source architecture: [Revised_Renovation_Scope_and_Estimate_Architecture.docx](Revised_Renovation_Scope_and_Estimate_Architecture.docx).

This document is the implementation handoff for migrating the current renovation
estimate pipeline to the following ownership model:

```text
existing photo analysis
    -> observed conditions
    -> Terra condition review
    -> deterministic condition disposition
    -> deterministic work items and standalone allowances
    -> deterministic package candidates
    -> Sol package review
    -> deterministic coverage and cost reconciliation
    -> versioned estimate artifact
```

The migration is intentionally evolutionary. Preserve the existing upstream photo
analysis, room/estimate-unit identity, catalog economics, and useful reconciliation
rules. Add the new architecture beside `renovation_estimate_v4`, validate it in
shadow mode, and only then make it the backend's authoritative estimator.

The core migration is divided into six medium-to-large implementation sessions.
Each session should be independently reviewable and leave the repository in a
working state. Later sessions may refine implementation details, but they must not
change the ownership boundaries and invariants in this document without an explicit
architecture decision.

## Decisions already made

These are settled inputs to the implementation, not questions for later sessions:

1. The new architecture supports only the newest catalog. It does not need a
   compatibility layer for older catalog schemas. Session 1 must identify the
   authoritative newest catalog form, validate it, and pin its version in all new
   artifacts. Legacy v4 may remain readable during migration, but the new engine
   must not dynamically fall back to an older catalog.
2. Physical-unit identity initially reuses the current room-surrogate and
   estimate-unit resolver. Ambiguous identity is represented explicitly; building a
   new room-understanding system is not part of this migration.
3. `cannot_assess` findings and high-consequence findings with inadequate evidence
   do not enter visible rehabilitation dollars. They remain in an inspection,
   uncertainty, or withheld lane.
4. Sol may make bounded combine/split recommendations among supplied package
   candidates. It may not change condition truth, work actions, quantities, or
   prices.
5. Frontend contract work is deferred. The backend should emit a clean, versioned
   new artifact while retaining existing v4 output during migration. A later
   frontend redesign will adopt the new contract; this project does not build a v4
   compatibility projection or redesign the UI.
6. Terra has a hard service budget of 2.5 million tokens per day. The rough current
   operating reference is 100 listings per day, or approximately 25,000 Terra
   tokens per listing. Maximize decision quality first, measure actual use throughout
   shadowing, and optimize only when measurements show where tokens can be removed
   without reducing performance.

## Non-negotiable invariants

The following rules must be executable validations, not just prompt instructions:

- There is at most one instance of a catalog condition per physical estimate unit.
- Duplicate and near-duplicate photos are not independent corroborating evidence.
- Objective evidence facts, Terra's review verdict, and the deterministic policy
  disposition are separate records.
- Terra can judge only supplied conditions. It cannot create work, packages, prices,
  confidence percentages, or new catalog concepts.
- Unsupported conditions cannot create work or dollars.
- `cannot_assess` is not a positive finding and cannot silently become visible
  rehabilitation scope.
- Every accepted condition maps to work, inspection, or an explicit no-action
  disposition.
- Package candidates are created only from accepted work items.
- Sol can judge only supplied package candidates and can never mutate condition or
  work truth.
- Rejecting or being uncertain about a package preserves every accepted child work
  item as standalone work.
- An approved package absorbs each child work item at most once.
- Final estimate totals contain no overlapping standalone and packaged allowances.
- Every condition and work item has an explicit final disposition and complete
  lineage back to its evidence.

## Existing integration boundaries

The migration should use the following current seams rather than refactoring the
entire pipeline first:

- `tools/artifact_writers.py` assembles the canonical/product issue lanes and calls
  the current v4 estimator. Add the new orchestration beside that call.
- `tools/renovation_estimate.py` already groups repeated condition evidence by
  catalog identity and room/estimate unit. Reuse the valuable identity and evidence
  semantics, but replace the mixed candidate object with the explicit layers in this
  plan.
- `tools/renovation_estimate_v4.py` remains the legacy comparison path during
  migration.
- `tools/rehab_packages.py` contains reusable package profiles, absorption rules,
  cost floors, and reconciliation behavior. Reuse proven primitives where their
  semantics match. Do not make a wholesale cleanup of this large module a
  prerequisite.
- `tools/pipeline_config.py` provides the precedent for one atomic runtime selector.
- `tools/publication_gate.py` is the final validation boundary before canonical
  publication.
- `tools/analyzer_server.py` loads long-lived dependencies once. Architecture,
  catalog, policy, prompt, and model selection must therefore be startup-safe and a
  configuration change requires a worker restart.

## Shared execution rules for all sessions

Every implementation session must:

1. Read this document and the source architecture before changing code.
2. Inspect the current worktree and preserve unrelated or staged user changes.
3. Identify the exact production caller before changing a legacy helper.
4. Prefer new, focused modules with typed/versioned boundaries over expanding the
   legacy package monolith.
5. Add focused tests for the session's contract and run the relevant existing
   estimate/package tests.
6. Run the full repository pytest suite before handoff. There is no tracked pytest
   configuration, so the handoff must record the exact command and an explicit
   writable `--basetemp` location.
7. Record schema versions and policy provenance in fixtures and artifacts rather
   than depending on implicit runtime defaults.
8. Leave a concise handoff documenting files changed, tests run, decisions made,
   and any remaining risks.

Sessions are sequential. A later session may begin only after the prior session's
acceptance gates pass, unless its work is limited to non-conflicting investigation.

## Session 1 - contracts, newest-catalog projection, and shadow scaffolding

### Objective

Define the durable data boundaries for the new engine and make it possible to run
the new path without affecting current output.

### Implementation procedure

1. Identify the authoritative newest catalog and its generated/source-of-truth
   workflow. The new engine will accept only this catalog version.
2. Define versioned contracts for:
   - `ObservedCondition`
   - `EvidenceFacts`
   - `ConditionReview`
   - `ConditionDisposition`
   - `WorkItem`
   - `PackageCandidate`
   - `PackageDecision`
   - coverage/reconciliation ledger entries
   - estimate-level provenance and totals
3. Define stable identifiers and lineage relationships. A model verdict, policy
   decision, work item, and package must refer to immutable IDs rather than copied
   display text.
4. Expose three logical projections from the newest catalog:
   - observable condition definition and evidence requirements;
   - deterministic work action, unit, pricing, and trade policy;
   - permitted package relationships and roles.
5. Require every catalog condition to have an explicit terminal route. Missing
   work/package data must fail validation or map intentionally to `inspection`,
   `no_action`, or `excluded`; it must not disappear through a default.
6. Add one startup-time architecture selector with modes equivalent to `current`,
   `shadow`, and `new`. Default to `current` while the migration is incomplete.
7. Add an empty shadow orchestration seam beside the current v4 calculation. In
   `current` mode it must not change v4 output. In `shadow` mode, incomplete new
   results must remain private/debug-only and must not become canonical output.
8. Stamp architecture, schema, catalog, policy, prompt, and model-routing versions
   into the new result envelope.

### Deliverables

- Versioned schemas/types and validators.
- Validated newest-catalog projection.
- Atomic architecture selector.
- Shadow result envelope and artifact-writer seam.
- Contract and selector tests.

### Acceptance gates

- The full existing test suite passes.
- Current mode preserves the existing v4 shape and behavior.
- Validators reject mixed, partial, or incorrectly versioned payloads.
- Every newest-catalog condition has an explicit terminal route.
- Every new envelope contains complete provenance.
- Invalid selector/catalog combinations fail at worker startup rather than midway
  through a listing.

### Explicitly out of scope

- Terra or Sol calls.
- Catalog price calibration.
- Support for older catalog versions.
- Catalog ontology redesign beyond what is required to make the newest catalog
  internally valid.
- Frontend changes.

## Session 2 - condition aggregation, evidence facts, and Terra review

### Objective

Create one auditable condition layer per physical estimate unit, review each
condition with Terra, and apply deterministic disposition policy.

### Implementation procedure

1. Consume the existing canonical/product-filtered issue lane after upstream photo
   analysis. Do not modify passes 1a-2e as part of this session.
2. Reuse the current room-surrogate and estimate-unit resolver to aggregate repeated
   observations into one `(catalog condition, physical estimate unit)` record.
3. Preserve separate records for the same condition in distinct rooms/components.
4. Persist objective evidence facts, including relevant image references, distinct
   photo count, distinct-view count where available, duplicate/near-duplicate
   groups, and any deterministic evidence thresholds.
5. Build a bounded Terra request containing only the supplied catalog condition,
   relevant evidence, and the allowed output schema.
6. Permit only `supported`, `unsupported`, or `cannot_assess` as semantic verdicts.
   Operational provider, timeout, schema, and parsing failures remain failures and
   must not be converted to `cannot_assess`.
7. Apply deterministic policy after the model verdict. The policy selects accepted
   for work, excluded, inspection/uncertain, withheld/escalated, or no action.
8. Persist the original evidence facts, Terra verdict, deterministic disposition,
   model/prompt provenance, and reason codes separately.
9. Preserve worker checkpoints so a failed Terra call can be retried without
   repeating successful upstream processing.
10. Instrument Terra input/output tokens at call, estimate-unit, and listing level.
    Batch related conditions by estimate unit when that reduces repeated image
    context without obscuring per-condition verdicts. Do not truncate evidence or
    downgrade the model merely to hit an unmeasured per-listing target.
11. Add a daily usage guard at the scheduler/orchestrator boundary. Approaching the
    2.5-million-token ceiling must queue or stop new Terra work explicitly; it must
    not silently skip condition review or publish a partial estimate.

### Deliverables

- Condition aggregation and evidence-fact modules.
- Terra review client/schema and checkpoint integration.
- Deterministic disposition policy.
- Token telemetry and daily-budget guard.
- Focused condition/evidence/model-failure tests.

### Acceptance gates

- Every input condition instance ends in one explicit terminal disposition.
- Duplicate images do not inflate condition or corroboration counts.
- High-consequence findings with insufficient evidence are withheld/escalated.
- Unsupported findings produce no work or dollars.
- `cannot_assess` findings remain outside visible rehabilitation scope.
- Provider and parse failures fail closed and preserve completed checkpoints.
- Token accounting reconciles from calls to estimate-unit and listing totals.
- Terra never emits work, package, quantity, price, or model-confidence fields.

### Explicitly out of scope

- Upstream observation recall or prompt redesign.
- A new room/computer-vision identity system.
- Work-item pricing or package generation.
- Premature token optimization before real shadow measurements exist.

## Session 3 - deterministic work-item derivation and standalone estimate

### Objective

Translate reviewed conditions into complete, deterministic work scope and a valid
standalone estimate that does not depend on packages.

### Implementation procedure

1. Map accepted condition dispositions to estimator-recognizable actions using the
   newest-catalog work projection.
2. Represent actions such as repair, replace, remediate, refinish, inspect, and no
   action explicitly rather than inferring them from display text.
3. Reuse existing catalog costs, units, market/cost factors, trade buckets,
   estimate-unit resolution, allowance tiers, and risk lanes. Do not recalibrate
   prices in this migration.
4. Merge duplicate billable work for the same physical scope while retaining all
   contributing condition IDs and evidence lineage.
5. Ensure formerly package-only catalog concepts have an explicit work,
   inspection, no-action, or exclusion route. Package membership is not a condition
   disposition.
6. Produce standalone low/high allowances and deterministic totals before any
   package candidate exists.
7. Persist suppressed and audit-only work with reason codes rather than dropping it.

### Deliverables

- Condition-to-work mapper.
- Work deduplication and lineage rules.
- Standalone estimate calculation.
- Work coverage and exact-total tests.

### Acceptance gates

- Every accepted condition maps to work, inspection, or explicit no action.
- Unsupported conditions contribute zero dollars.
- `cannot_assess` remains in the inspection/uncertainty lane.
- No physical work is billed twice because it appeared in multiple photos or
  conditions.
- Every work item identifies all contributing condition IDs.
- Standalone low/high totals reconcile exactly without packages.
- The current pricing corpus remains unchanged unless a pre-existing catalog error
  makes the new contract impossible; such a contradiction must be raised rather
  than silently patched.

### Explicitly out of scope

- Package creation or Sol calls.
- Price calibration and contractor-level estimating assemblies.
- Frontend presentation.

## Session 4 - deterministic package candidates and bounded Sol review

### Objective

Construct package opportunities from accepted work and let Sol judge only whether
those proposed groupings are coherent.

### Implementation procedure

1. Generate package candidates deterministically from accepted work-item IDs only.
2. Initially reuse current package families, affinities, driver/support roles,
   strength rules, room identity, allowance tiers, cost floors, absorption scopes,
   and whole-home rollup concepts where they remain compatible with the new
   contracts.
3. Make the complete candidate membership and proposed treatment explicit before
   invoking Sol.
4. Give Sol only supplied package candidates, accepted child work, relevant
   evidence summaries, and a bounded output schema.
5. Allow `approve`, `reject`, or `uncertain`, plus bounded combine/split
   recommendations among supplied candidates.
6. Validate combine/split output so it cannot add an unknown work item, remove a
   child, alter an action/quantity, invent a package family, or set prices.
7. Store `PackageDecision` separately. Applying the decision may change only
   packaging treatment; the condition and work snapshots remain immutable.
8. Treat model/provider failures as operational failures, not as package
   uncertainty.

### Deliverables

- Deterministic package-candidate builder.
- Bounded Sol request/response contract.
- Combine/split validator.
- Immutable condition/work snapshot checks.
- Package decision and failure tests.

### Acceptance gates

- Every package child is an existing accepted work-item ID.
- Condition and work truth are byte-for-byte or structurally unchanged by Sol.
- Rejected or uncertain packages preserve all accepted child work as standalone.
- Combine/split cannot alter actions, quantities, prices, or total child coverage.
- Sol cannot create unsupplied work or package families.
- Provider failure is distinguishable from an `uncertain` decision.

### Explicitly out of scope

- A general-purpose LLM work planner.
- LLM-selected prices or quantities.
- Package pricing redesign.
- Broad cleanup of `tools/rehab_packages.py` unrelated to extracted/reused logic.

## Session 5 - coverage ledger, reconciliation, artifact output, and observability

### Objective

Apply package decisions without losing or double-counting accepted scope, and emit
a complete versioned backend artifact.

### Implementation procedure

1. Create an explicit coverage ledger in which every accepted work item has exactly
   one final representation:
   - `standalone`;
   - `absorbed_by(package_id)`;
   - `inspection`;
   - `no_action`.
2. Apply approved package decisions. A package may replace child standalone
   allowances, but each child can be absorbed only once.
3. For rejected or uncertain packages, retain the exact original standalone child
   allowances.
4. Reuse current deterministic cost floors, caps, scaling, allowance tiers, and
   evidence projection only where their behavior respects the new ledger.
5. Calculate totals from the ledger rather than attempting to infer coverage from
   display groups.
6. Add audits for lost accepted work, duplicate absorption, unsupported billed
   work, orphan package children, and arithmetic mismatch.
7. Emit the new estimate under its own versioned artifact key. During migration,
   continue emitting existing v4 output independently; do not build a frontend
   compatibility adapter.
8. Extend publication validation so malformed new artifacts cannot be written as
   successful canonical results when the new engine becomes authoritative.
9. Add architecture/schema/catalog/policy/model provenance, checkpoint
   fingerprints, failure categories, phase timing, token totals, and funnel counts:
   observations -> conditions -> dispositions -> work -> package candidates ->
   package decisions -> final ledger entries.

### Deliverables

- Coverage and reconciliation ledger.
- Deterministic package application.
- New versioned estimate artifact.
- Publication and checkpoint integration.
- Reconciliation audits and telemetry.

### Acceptance gates

- Exact low/high arithmetic reconciliation.
- Rejecting or marking every package uncertain reproduces the exact standalone
  total.
- No accepted work disappears from the ledger.
- No work item is absorbed by more than one package.
- Unsupported and `cannot_assess` conditions cannot enter visible rehab totals.
- Current mode continues to emit an unaffected v4 result.
- Mixed, partial, or invalid new artifacts fail before canonical publication.
- Every suppressed/audit-only record has a reason code.

### Explicitly out of scope

- Frontend adapters or UI changes.
- Historical artifact regeneration.
- Deleting the legacy estimator.
- Pricing calibration.

## Session 6 - shadow canary, token measurement, and backend cutover

### Objective

Demonstrate correctness, quality, stability, and acceptable Terra usage on
production-shaped listings, then make the new estimator selectable as the backend
authority.

### Implementation procedure

1. Freeze the code, newest catalog, policy versions, prompt versions, model routing,
   images, and listing metadata used for the comparison.
2. Reuse the existing focused package golden cases and the established stratified
   18-property canary. Do not make creation of a new benchmark platform a release
   dependency.
3. Run the legacy v4 and new engine from the same upstream inputs. Keep new output
   private/shadow-only during comparison.
4. Compare condition survival/dispositions, duplicate handling, work coverage,
   package family/tier, absorbed-versus-standalone representation, headline
   low/high totals, run-to-run stability, phase latency, and tokens.
5. Manually review every changed scope/package and every headline low/high delta
   greater than 15 percent. Record an explanation or fix for each accepted delta.
6. Measure Terra tokens per call, estimate unit, condition, and listing. Report
   median, upper percentiles, worst case, and projected listings/day under the
   2.5-million-token ceiling.
7. If optimization is needed, first remove repeated context and improve batching.
   Any prompt or evidence reduction must rerun the same canary and show no material
   loss in condition-review quality.
8. Enable the `new` backend selector only after all gates pass. Restart the
   persistent worker and run end-to-end smoke validation of the resulting artifact.
9. Keep the legacy v4 computation/output available for comparison and rollback
   until the later frontend redesign and an observation period are complete.
10. Roll back by returning the selector to `current` and restarting the worker. Do
    not rewrite already-valid artifacts as part of rollback.

### Release gates

- All 18 canary properties produce schema-valid new artifacts.
- Zero publication failures.
- Zero arithmetic reconciliation failures or double counts.
- Zero unexplained accepted-condition loss.
- Every material package/scope change has written review.
- Every headline delta greater than 15 percent is reviewed and approved or fixed.
- Run-to-run instability is measured and has no unexplained material scope loss or
  gain.
- Terra token usage is fully measured and respects the hard 2.5-million-token daily
  scheduler ceiling.
- The report states expected listings/day from observed usage; approximately
  25,000 Terra tokens/listing is the initial capacity reference, not a quality
  target that overrides correctness.
- Selector rollback and worker restart are successfully smoke-tested.

### Explicitly out of scope

- Frontend adoption or redesign.
- A new human-truth benchmark platform.
- Bulk historical backfill.
- Legacy estimator deletion.
- Catalog price calibration or unrelated upstream prompt tuning.

## Historical artifacts

Existing v4 artifacts remain readable and immutable. Do not synthesize new
condition-review verdicts from old summarized estimate output.

A historical artifact may be reprojected into the new architecture only if all
required condition evidence, estimate-unit identity, review inputs, and model
verdicts are available. Otherwise mark it `needs_reanalysis` and regenerate it from
source images through the normal pipeline when needed.

Bulk history migration and legacy-code retirement are optional follow-on projects
after backend stabilization and frontend adoption. They are not a seventh hidden
stage of this migration.

## Terra budget policy

The 2.5-million-token daily ceiling is operationally hard, but per-listing token use
is initially an observation rather than a hard rejection threshold.

- Measure input, cached input where reported, and output tokens separately.
- Attribute tokens to listing, estimate unit, and condition review.
- Avoid sending irrelevant images or the full catalog to Terra.
- Prefer sharing relevant image context across conditions in one estimate unit when
  the structured response still yields one verdict per condition.
- Do not substitute confidence summaries for evidence or silently skip review to
  save tokens.
- When the daily ceiling would be exceeded, queue or stop new analysis and expose
  the reason explicitly.
- Optimize after the shadow report identifies repeated or low-value context. Every
  optimization must be compared against the same frozen inputs.

## Rollback and stop conditions

Immediate rollback or release stop is required for:

- a mixed or invalid canonical artifact;
- accepted work disappearing without an explicit terminal disposition;
- reproducible double counting or reconciliation failure;
- Sol altering condition/work truth;
- Terra failure being published as a semantic verdict;
- silent continuation after the daily Terra ceiling is reached;
- repeated estimator/writer failures after the selector is enabled.

Rollback is a configuration change to `current` plus a persistent-worker restart.
Artifact deletion or database rollback is not part of the normal procedure.

## Completion definition

The backend migration is complete when:

- all six sessions and their gates are complete;
- the new estimator can run as the selected backend authority;
- the newest catalog is the only catalog accepted by the new engine;
- every condition, work item, package decision, and final dollar has auditable
  lineage;
- the 2.5-million-token daily Terra ceiling is measured and enforced without silent
  quality degradation;
- the legacy path remains available for rollback; and
- the new artifact is stable and documented for the later frontend redesign.

Frontend adoption, historical backfill, catalog price calibration, and legacy-code
retirement begin only as separately approved follow-on projects.
