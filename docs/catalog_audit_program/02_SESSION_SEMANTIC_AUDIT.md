# Session 2 brief — semantic audit and proposal drafting

## Mission

Use Session 1's verified evidence bundle to perform the semantic catalog audit and produce a reviewable proposal artifact.

This session may propose changes, deferrals, or non-catalog outcomes. It must not edit the migration decisions, generated catalog, prompts, or runtime code.

### Phase C scope direction (Steven, 2026-09-03)

- Keep the audit broad enough to diagnose ownership, but make the intervention catalog-first and targeted. Prioritize recovery of real billable conditions, catalog claim coherence, missing coverage, overlapping ontology, and operationally consequential misresolution.
- Do not optimize Pass 2a in this session. Pass-2a-owned hallucinations remain negative controls and regression constraints; they do not independently justify catalog proposals.
- Record cases where an adequate item was reachable but Pass 2d selected another as a bounded post-catalog Pass 2d follow-up. Do not propose Pass 2d prompt or runtime changes in Session 2.
- Treat catalog 3.2 as the stable proposal baseline, not as a completed semantic redesign of 3.1. The audit should address root catalog semantics while preserving the pinned baseline distinction.
- Separate internal observation specificity from repair/billing normalization and UI granularity. Precise catalog observations may share a work item, route, package, or concise display label.
- Revise any pre-existing Phase C task plan to reflect this direction before authoring final cluster judgments. Work cluster by cluster and avoid asking one reviewer to resolve unrelated semantic families or pipeline questions in a single omnibus judgment.

## Required context

Read:

1. `docs/catalog_audit_program/00_OVERALL_CONTEXT.md`
2. This brief.
3. `docs/catalog_audit_program/HANDOFF_SESSION_1.md`
4. `reports/catalog_audit_evidence.json`
5. Only the source evidence and photos referenced by records promoted for semantic review.
6. `scripts/migrate_catalog_kind_v2.py` and `tools/catalog_validation.py` for expressibility checks.

Do not repeat Session 1's full mechanical joins when its artifact hashes and reconciliation checks pass.

## First action

Verify the Session 1 commit, handoff, and evidence hash. Inspect a representative sample against source artifacts. Then write a session-specific implementation/review plan covering semantic clustering, photo review, evidence scoring, proposal schema, rendering, and verification.

## Authorized changes

Expected outputs are deliberately small:

- `reports/catalog_audit_proposals.json`
- `scripts/render_catalog_audit_proposals.py`
- `tests/test_catalog_audit_proposals.py`
- `docs/PROPOSAL_catalog_audit_<date>.md`
- `docs/catalog_audit_program/HANDOFF_SESSION_2.md`

If an existing generic renderer can satisfy the contract cleanly, reuse it. Do not create an interactive review application.

## Prohibited changes

Do not modify:

- Migration decisions or either catalog.
- Generated migration artifacts.
- Runtime retrieval, pass 2d, Terra, projection, pricing, or package behavior.
- Frozen evidence or Session 1's evidence bundle except through a separately documented evidence-builder correction.

## Workstream A — semantic clustering

Review unresolved observations by:

- Physical subject.
- Visible state/mechanism.
- Upstream kind.
- Scene.
- Unsupported material/object/mechanism/severity/location/scope commitments.

Allow one item to appear in multiple clusters where failure mechanisms differ. Consolidate one semantic issue spanning several items into one cluster. Record merge/split reasoning for the cluster structure.

Execute the review in bounded semantic/confusion families. Do not pool unrelated questions merely for throughput; the cluster boundary should make the claim, controls, and alternative owners understandable in isolation.

## Workstream B — full item-family review

For each cluster:

- Inspect current and evidence-era item semantics.
- Inspect migration parent/siblings.
- Inspect frozen and current candidate neighbors when available.
- Search the full current catalog before classifying missing coverage.
- Add or remove family neighbors only with a written rationale.
- Review kind and scene reachability, embedding text, lexical shortcut exposure, and negative-control neighbors.

The question is not merely whether current wording is imperfect. Determine whether a catalog change is the smallest appropriate lever.

For each imperfect mapping, classify whether it is an exact semantic match, an operationally equivalent mismatch, or an operationally consequential mismatch. Operational equivalence requires the same physical subject, repair, approximate scope, billability, trade/route, and safety implications. Do not create catalog granularity solely to make the UI more specific, and do not treat a shared broad repair as proof of equivalence when those dimensions differ.

## Workstream C — evidence and photo review

For every case used as proposal support:

- Inspect the actual evidence photo(s).
- Read the full relevant lineage, not just the case summary.
- Distinguish what the image supports from what catalog language commits to.
- Record the independent evidence-unit key and correlation flags.
- Identify whether gold/factorized support is independent or method corroboration.

For every implicated item/family, review:

- Successful uses.
- Correct rejections.
- Failure cases.
- Relevant reviewer notes.
- Selected negative controls.

If no human-reviewed successful example exists, state that absence without inferring that the item never succeeds.

Pass-2a hallucination cases are negative controls in this session, not proposal seeds. Use them to test whether broader wording, retrieval text, or a split would make a known neutral or false observation easier to select or bill.

## Workstream D — diagnosis

Assign one primary cause and optional contributing causes using the taxonomy in the overall context.

Explicitly answer:

- Why does the issue belong or not belong to the catalog?
- If pass 2d or Terra was the attributed stage, why is catalog ownership still or no longer justified?
- Does an adequate item exist but fail retrieval/selection?
- Does the evidence instead point upstream, downstream, product policy, or uncertainty?
- Is the mismatch operationally consequential, or would the same subject, repair, scope, billability, route, and safety treatment make it a low-priority naming difference?
- If the correct item was reachable but not selected, should the catalog remain unchanged while the case is carried into the post-catalog Pass 2d worklist?

Non-catalog and no-change results must receive the same evidence discipline as catalog proposals.

## Workstream E — evidence-bar enforcement

Calculate evidence counts from unique evidence units, not raw rows. For each cluster record:

- Independent support count.
- Distinct property count.
- Correlated support count.
- Gold/factorized corroboration type.
- Success/counterexample count.
- Evidence-bar result and justification.

Single-case structural exceptions require a mechanically demonstrable universal fact and an explicit written justification. All other below-threshold clusters are deferred or no-change.

## Workstream F — proposal expressibility

Classify each result as one of:

- Native decisions-file proposal.
- Non-catalog action.
- Deferred/insufficient evidence.
- Migration-system gap.

For native proposals, produce an exact or near-exact decisions-file diff and a field-level semantic diff. Validate it against a deep in-memory copy where practical. Do not write it to the real decisions file or generated outputs.

Enforce these constraints:

- No standalone `add` or `merge` fiction.
- No unrelated-parent split used to add an item.
- No economic override hidden in a wording proposal.
- Every split includes inherited economic fields and a semantic-fit assessment.
- Every retirement includes lost route, pricing, package, and work-item behavior.
- Evidence-gate changes are migration-system/policy gaps under the current schema unless separately approved.
- Retrieval-metadata (`scene_groups`, `require_any`) and `route_override` changes on a non-split carryover are migration-system gaps; they are natively expressible only on split successors.
- Product-quarantined coverage remains a product-policy issue.

## Required proposal schema

Each proposal/cluster record should include:

- Stable ID, title, and status.
- Source and baseline hashes.
- Implicated items and family.
- Evidence-era and proposal-baseline semantics.
- Supporting, success, correct-rejection, and counterexample records.
- Photo-review status.
- Independence and correlation assessment.
- Primary diagnosis and contributing causes.
- Rejected alternative owners.
- Proposed outcome type.
- Near-exact decisions diff when native.
- Semantic before/after summary.
- Retrieval and reachability effects.
- Economic, package, route, and product-policy implications.
- Expected benefit and regression risk.
- Evidence-bar result, confidence, and unresolved questions.
- Blank human disposition.

The generated Markdown should contain a concise main body plus appendices for no-change clusters, deferred clusters, source fingerprints, and the full item-family/validation matrices.

## Verification

Test:

- Proposal schema validation.
- Deterministic ordering and rendering.
- Complete evidence references back to Session 1.
- Unique proposal IDs.
- Required success/counterexample disclosure.
- Evidence-bar calculation.
- Native-operation validation and migration-system-gap classification.
- No write to decisions/generated catalog during in-memory checks.

## Exit criteria

Session 2 is complete when:

- Every worklist cluster has exactly one primary outcome.
- Every proposal cites concrete, photo-reviewed evidence.
- Successes, correct rejections, and counterexamples are present or explicitly absent.
- Every result has an evidence-bar decision and alternative-stage analysis.
- Native proposals are shown to be expressible in memory.
- Unsupported changes are clearly separated as system/policy gaps.
- Pass-2a hallucinations are retained as regression controls without becoming a remediation workstream.
- Reachable-right-item Pass 2d cases are identified for a bounded post-catalog follow-up rather than being disguised as catalog gaps.
- Semantic precision, operational equivalence, and user-facing granularity are distinguished in every proposal where that distinction affects the outcome.
- The real decisions and generated catalogs are unchanged.
- `HANDOFF_SESSION_2.md` is complete.

## Suggested opening prompt

> Execute Session 2 of the catalog-audit program. Read the overall context, this session brief, `HANDOFF_SESSION_1.md`, and the hash-verified `reports/catalog_audit_evidence.json`. Verify the handoff, then draft a task-specific implementation/review plan. Perform the semantic audit, inspect all photos used as proposal evidence, compare failures with successes and correct rejections, and generate the structured proposal artifact and deterministic review document. Do not edit migration decisions, generated catalogs, prompts, or runtime code. End with `HANDOFF_SESSION_2.md`.
