# Session 1 brief — evidence foundation and worklist

## Mission

Implement the deterministic foundation for the catalog audit and produce a complete, deduplicated candidate/family worklist.

This is an offline evidence-engineering task. It must not decide catalog wording, propose actual fixes, or edit catalog/migration/runtime artifacts.

## Required context

Read before planning:

1. `docs/catalog_audit_program/00_OVERALL_CONTEXT.md`
2. This brief.
3. `docs/RESULT_error_attribution_20260831.md`
4. `docs/PLAN_error_attribution_review_20260831.md`
5. `scripts/migrate_catalog_kind_v2.py`
6. `tools/catalog_validation.py`
7. `scripts/build_error_attribution_queue.py`
8. `scripts/error_attribution_report.py`
9. The evidence files listed in the overall context.

Also inspect `docs/CONTEXT_catalog_audit_inputs_20260901.md` as historical source material, but use the corrections in the overall context where they disagree.

## First action

Read the README baseline rule, then inspect the current checkout, verify the expected architecture and frozen inputs, and produce a concise implementation plan for this session. The plan must identify exact files to create or change, tests, output schemas, and stopping conditions. Treat any baseline hash mismatch as drift per the overall context, Section 5. Do not start coding if the queue or its pinned inputs have unexplained hash drift.

## Authorized changes

This session may create or modify only narrowly scoped offline audit assets, expected to include:

- `scripts/build_catalog_audit_evidence.py`
- `tests/test_catalog_audit_evidence.py`
- `reports/catalog_audit_evidence.json`
- An optional compact generated Markdown summary if it materially improves reviewability.
- `docs/catalog_audit_program/HANDOFF_SESSION_1.md`

Names may be adjusted if repository conventions demand it, but keep one evidence builder and one principal machine-readable output. Do not build a UI, database, service, or new pipeline tracer.

## Prohibited changes

Do not modify:

- `tools/catalog_migrations/kind_v2_decisions.json`
- `tools/issue_catalog.json`
- `tools/issue_catalog_kind_v2.json`
- Generated migration manifests/audits.
- Pass 2a–2e prompts or runtime behavior.
- Terra behavior.
- Pricing, package, quarantine, or product policy.
- The frozen attribution queue or ledger.

## Workstream A — provenance freeze

Record in the output manifest:

- Starting Git commit, branch, and dirty state.
- SHA-256 for every consumed evidence, catalog, decision, generator, and relevant schema file.
- Queue-embedded input hashes and whether the current files match.
- Every referenced run artifact and its recorded hash, when artifact enrichment is used.
- Evidence-era catalog identity (version, resolved git commit, on-disk and blob SHA-256, and the artifact `catalog_sha256` run count per source) and proposal-baseline catalog identity.

Preserve unrelated untracked/user-owned files. A mismatch in an expected frozen source should fail closed unless it is explicitly reconciled in the handoff.

## Workstream B — 3.1 versus 3.2 baseline reconciliation

Materialize the evidence-era 3.1 catalog from the git state whose checkout bytes hash to the artifact-recorded `catalog_sha256` (expected `07ee112`; overall context, Section 5), and compare against that exact state rather than the 3.1 version string.

Build a deterministic comparison covering, at minimum:

- Item IDs and order.
- `kind`
- `name`
- `atomic_claim`
- `description`
- `embed_text`
- `support_any`, `deny_any`, and `require_any`
- `scene_groups`
- `drop_if_generic` and `defaultHidden`
- Route/package fields.
- Economic fields.

The output should state which runtime semantics in frozen evidence differ from the current proposal baseline. Explicitly test the historical finding that `atomic_claim` is unchanged rather than hardcoding it.

## Workstream C — deterministic indexes and joins

Reuse existing loaders where available. Build indexes for:

- Queue case and latest attribution by `case_id`.
- Property, source, run, photo, condition, projected condition, and final resolved item.
- Migration parent, successor, and sibling relationships.
- Evidence-era and current catalog item semantics.
- Labels joined to exact review cards.
- Qualitative notes joined by card ID.
- Factorized leads joined by `(property_key, condition_id)`.
- Gold cases and matching-table rows.

Do not join `condition_id` across runs without the run/source identity. Do not use `p2c_surviving[].catalog_item_id` as the final mapping.

## Workstream D — optional frozen candidate enrichment

The queue contains candidate counts but not every candidate ID/rank. Candidate enrichment is allowed only from the exact run artifact named in the frozen evidence after its hash is verified.

Capture:

- Original candidate IDs and order.
- Scores if present.
- Selected `resolved_item_id`.
- Artifact provenance.

If the artifact is missing or mismatched, record candidate detail as unavailable. Do not rerun pass 2d to reconstruct historical candidates.

## Workstream E — candidate seeding

Seed a neutral worklist from:

- Latest pass-2d-attributed cases.
- Appendix-misnamed cases.
- Terra-attributed miss cases potentially involving catalog-introduced identity or specificity.
- Factorized misnamed leads.
- Gold miss candidates and explicitly open coverage questions.
- Deferred catalog findings as backlog/constraint references.

Product-quarantined findings go to a product-policy/no-change lane. Pass-2a hallucinations may annotate an already implicated family but do not independently seed a catalog change.

The builder should explain every inclusion rule mechanically. It must not label a seed as a confirmed catalog problem.

## Workstream F — deduplication and evidence units

Use the independence keys from the overall context. Preserve:

- Source identifiers.
- Same-property correlation.
- Same-photo/condition correlation.
- Whether a record is independent evidence, method corroboration, or annotation.

Reconcile every seed exactly once after deduplication. Emit counts before and after deduplication by source and lane.

## Workstream G — item families and semantic-cluster inputs

Build the mechanical part of item families:

- Migration parent and all successors.
- Same-kind/same-trade items with overlapping scenes.
- Shared work-item, estimate, or package relationships.
- Frozen candidate neighbors when available.
- Current baseline retrieval neighbors only if they can be obtained without live provider calls and with deterministic provenance.

Do not force semantic clustering mechanically. Emit normalized observation text and structured dimensions that Session 2 can review, plus a field for reviewer-added/removed neighbors and rationale.

## Required output contract

`reports/catalog_audit_evidence.json` should contain at least:

- Schema version.
- Generator timestamp only if necessary; deterministic content must otherwise be stable.
- Git and source manifest.
- Baseline comparison.
- Case/evidence index.
- Candidate seeds with source reasons.
- Independence/correlation records.
- Item families and neighbor provenance.
- Positive-use and correct-rejection records.
- Gold and factorized lead annotations.
- Deferred/product-policy lanes.
- Reconciliation counts and validation results.
- Explicit unavailable-data records.

The builder should sort all collections deterministically. Tests should generate into a temporary location or compare in memory, not overwrite frozen files.

## Verification

At minimum, test:

- Latest-wins ledger behavior, including undo records.
- Frozen hash mismatch failure.
- Run-scoped join identity.
- Correct authoritative 2d field selection.
- Deduplication and independence classification.
- Label/review-card and note/card joins.
- Factorized join behavior where no item ID is present.
- Migration parent/sibling family construction.
- Stable ordering and repeat generation.
- Reconciliation of all seeded inputs.

Run focused tests first, then relevant existing catalog/evidence tests. Invoke pytest as `.venv\Scripts\python.exe -m pytest`; bare `python` does not resolve the project venv. Use an explicitly writable pytest base temp if the repository's stale temp directories cause permission errors.

## Exit criteria

Session 1 is complete when:

- All required source hashes are recorded and reconciled.
- Evidence and proposal baselines are explicit.
- The 3.1-to-3.2 comparison is stored.
- All configured seeds reconcile exactly once after deduplication.
- Positive uses and correct rejections are available by item/family.
- No semantic proposal or production artifact was changed.
- The evidence output is deterministic and tested.
- `HANDOFF_SESSION_1.md` is complete.

## Suggested opening prompt

> Implement Session 1 of the catalog-audit program. Read `docs/catalog_audit_program/00_OVERALL_CONTEXT.md` and `docs/catalog_audit_program/01_SESSION_EVIDENCE_FOUNDATION.md`, then inspect and fact-check the current repository. Begin with a task-specific implementation plan. Build only the offline provenance/evidence/worklist foundation described in the brief. Do not propose catalog fixes or edit catalog, migration, prompt, or runtime artifacts. Preserve frozen evidence and unrelated user changes. Verify determinism and end with `docs/catalog_audit_program/HANDOFF_SESSION_1.md` using the shared handoff template.

