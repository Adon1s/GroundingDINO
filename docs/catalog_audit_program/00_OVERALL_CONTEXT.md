# Catalog audit program — verified overall context

Last verified: 2026-09-01 against repository HEAD `e9a7dc3a54439fb341c1ae8cd665e5e572fcfaf6` on branch `terra_factorized_verifier`. Coherence review applied the same day: Sections 1, 3, 4, 5, and 10 were corrected or extended, and the README gained a baseline rule covering the unvalidated 3.2 change.

This document is the stable context spine for all catalog-audit sessions. A future task must verify the expected checkout and relevant hashes before relying on it. If the repository has materially changed, update the task's handoff rather than silently applying stale assumptions. A later commit that only adds or edits this packet is not a material change; the pinned file hashes are what matter, and HEAD moving past `e9a7dc3` for that reason needs only a one-line note in the handoff.

## 1. Objective

Build an evidence-driven process that can determine whether catalog-shaped failures represent:

- Catalog wording or specificity problems.
- Missing catalog coverage.
- Overlapping or ambiguous ontology.
- Retrieval metadata or reachability problems.
- Pass 2d selection behavior.
- Terra verification behavior.
- Upstream kind/pass-2c behavior.
- Downstream tier, projection, routing, package, or costing behavior.
- Product-policy/quarantine behavior.
- Insufficient evidence or no change.

`No catalog change / fix another stage` is a successful first-class outcome. Attribution stage is evidence about where behavior failed; it is not automatic proof of where the fix belongs.

Standing priority (Steven, 2026-08-21): judge candidates on fewer hallucinations and fewer lost real conditions, not on dollar effects. Prices are provisional and price calibration is a separate future thread, so package and total deltas are reported for context, never scored as the primary result.

The program should produce surgical, reviewable catalog proposals, obtain human approval, implement only approved native changes, and validate them in increasingly expensive tiers.

### Current intervention priority (Steven, 2026-09-03)

The audit remains system-wide as a diagnostic framework, but the current intervention is deliberately catalog-first. Session 2 should prioritize catalog-owned coverage gaps, incoherent claims, overlapping concepts, and consequential misresolution of real billable conditions. It should not turn into a general effort to improve every pipeline stage at once.

Pass-2a-owned hallucinations are not a current remediation target because Pass 2a is intentionally frozen. Retain those cases as negative controls and regression constraints around any catalog broadening, split, or retrieval change; do not let a hallucination-only cluster independently drive a catalog proposal. Record adjacent Pass 2a findings as deferred without proposing prompt or runtime changes.

Pass 2d is the intended next optimization target after approved catalog changes are implemented and validated. Until then, Session 2 should distinguish candidate unavailability from wrong selection, record reachable-right-item cases for that follow-up, and avoid compensating for catalog defects with Pass 2d prompt changes. The later Pass 2d review should be performed in bounded semantic/confusion families rather than one omnibus prompt.

Internal catalog precision, repair/billing normalization, and user-facing presentation are separate concerns. Atomic claims should be specific enough for reliable retrieval, selection, and verification; rich synonyms and visual language belong primarily in retrieval fields. Multiple precise observations may share the same work item, route, package, or concise UI description. A semantic mismatch is operationally equivalent only when it preserves the physical subject, repair, approximate scope, billability, trade/route, and safety implications. Do not proliferate items for harmless wording differences, but do not excuse a wrong subject, mechanism, severity, or route merely because two cases could share a broad repair label.

## 2. Non-goals

The program does not authorize:

- Rebuilding the existing error-attribution trace.
- Editing the generated catalog directly.
- Proposing or applying catalog changes before evidence review.
- Changing Pass 2a or unrelated prompts during catalog evaluation.
- Broad pass-2d, Terra, costing, packaging, or product-policy redesign.
- General price calibration.
- Publishing the catalog to production.

Adjacent findings should be recorded and classified, not absorbed into the current task.

## 3. Catalog architecture and edit surface

The current generated catalog is:

- `tools/issue_catalog_kind_v2.json`
- Version `3.2`
- Ontology `observation-kind-v2`
- 128 generated items

It is generated deterministically by `scripts/migrate_catalog_kind_v2.py` from:

- `tools/issue_catalog.json` — v1 version `2.1`, 107 items.
- `tools/catalog_migrations/kind_v2_decisions.json` — reviewable migration decisions.

Generation also produces:

- `tools/catalog_migrations/2.1_to_3.0.json`
- `tools/catalog_migrations/2.1_to_3.0_audit.md`

Never hand-edit any generated output. Eventual catalog implementation changes go through the decisions file and normal regeneration.

At the proposal baseline (`e9a7dc3`) the 107 decision entries are (re-count whenever the decisions-file hash has drifted):

| Decision | Count |
|---|---:|
| `unchanged` | 32 |
| `reclassified` | 50 |
| `narrowed` | 4 |
| `split` | 19 |
| `retired` | 2 |

Correction (2026-09-08, S4-4): at the CAP-007 candidate `6e67eaa` (decisions `2e49a82f…9c76`) the table reads `reclassified` 49 / `split` 20 (other rows unchanged), with 44 inherited-economics successors and 129 generated items; the 128-item / 50 / 19 figures above describe the proposal baseline only.

The exact supported decision vocabulary is `unchanged`, `reclassified`, `narrowed`, `split`, and `retired`. There is no native standalone `add` or `merge` operation. A genuinely new item can currently be introduced only as a successor of a legitimate split. Do not misuse an unrelated parent to simulate an addition.

### Non-split and split authoring rules

For a non-split carryover, the migration system permits wording/retrieval overrides only for:

- `name`
- `description`
- `embed_text`
- `support_any`
- `deny_any`

The successor decision separately supplies its v2 `kind` and `atomic_claim`.

Split successors are authored fresh for their core identity and semantics (`id`, `kind`, `severity`, `name`, `description`, `embed_text`, and `support_any` are required) and may author any field on the generator's inherited-structural list — `trade_bucket`, `scope`, `tier`, `defaultHidden`, `drop_if_generic`, `category`, `display_class`, `require_any`, `deny_any`, `scene_groups`, `route_override` — otherwise inheriting the parent's value. Economic fields cannot be overridden.

Two consequences matter for expressibility:

- Retrieval metadata (`scene_groups`, `require_any`) and `route_override` are natively authorable only on split successors. For a non-split carryover they are outside the wording allowlist, so "fix this item's `scene_groups`" is a migration-system gap under the current schema, not a native proposal.
- `route_override` has a closed vocabulary of `no_action` (`VALID_ROUTE_OVERRIDES` in `tools/catalog_validation.py`) and is deliberately non-economic: it can only force an otherwise-billable successor out of billing under the v5 projection, never create a work route, and legacy consumers ignore it. It is the native lever for a true-but-should-not-bill successor.

### Economics

The generator treats the following as economic fields:

- `cost`
- `estimate`
- `work_item_code`
- `cost_model`
- `package_affinity`
- `package_role`
- `estimate_scope`
- `estimate_scope_reason`

Split successors inherit their parent's economic fields and receive `pricing_status=inherited_from_split_parent`. Every split proposal must assess whether those inherited economics remain semantically appropriate.

Missing economics do not have one universal runtime outcome. High/medium estimate tiers are more constrained, minor work may use heuristic behavior, and the v5 projection can route items with neither cost nor work-item code to no-action. Proposals must describe the actual route and cost implications rather than assuming either automatic failure or automatic safety.

`min_photo_evidence` is nested inside `estimate`, so a gate change is an economic override under the current migration rules. The generated catalog currently has six items with a two-photo gate, originating from four legacy parents. A proposed gate change is therefore not a normal decisions-file wording change; it must be classified as a migration-system/policy gap unless separate schema work is approved.

`package_strength` is computed at runtime, and `package_category` is derived or supplied by runtime fallback behavior. They are not catalog authoring fields.

## 4. Runtime selection and retrieval behavior

Under `KIND_ONTOLOGY_VERSION=observation_kind_v2`, `ISSUE_CATALOG_PATH` is a hard configuration error. Baseline/candidate validation must use normal decisions-file generation in isolated worktrees rather than pointing the pipeline at an alternate catalog copy.

Pass 2d retrieval has several material behaviors:

- Kind filtering is strict and fail-closed. Changing `kind` changes which observations can see an item.
- `scene_groups` filters candidate eligibility. Scene tokens are exact; the living-area token is `living_areas`, not `living`. An item with no `scene_groups` field is eligible in every scene group (`tools/catalog_embeddings.py`), so giving such an item a list narrows it.
- A nonempty `embed_text` replaces the other text as the full embedding source.
- Candidate scoring can also enter a lexical shortcut based on score/margin and support/component-condition overlap. Candidate changes can therefore affect both the LLM candidate set and the pre-LLM shortcut.
- `lineage.per_issue[].p2d.resolved_item_id` is the authoritative final 2d resolution in the frozen evidence.

The historical context pack said every `p2c_surviving[].catalogItemId` was `None`. That key does not exist; the field is `lineage.per_photo[*].p2c_surviving[].catalog_item_id` (snake_case), and the `None` reading was an artifact of probing the wrong name. In the frozen queue it is non-null in 683 of the 859 case-lineage rows (176 null; gold-photo lineage holds further rows). It remains non-authoritative for the final 2d choice.

Trades marked `product_quarantined` should be routed to a product-policy/no-catalog-change lane unless a separate product-policy change is approved. Apply this rule from metadata rather than hardcoding a particular trade.

## 5. Frozen evidence and provenance

Do not regenerate `reports/error_attribution_queue.json`. It is the frozen lineage authority for this audit.

At the verified baseline it contains:

- 104 queue cases.
- 55 attributable human-review cases.
- 15 gold photos with 112 gold findings.
- Lane counts: 6 appendix inconclusive, 8 appendix misnamed, 6 appendix trivial, 7 counted agreement, 31 counted correct rejection, 11 counted orphan, 9 hallucination-label, 3 hallucination-v1-only, 20 miss-label, and 3 miss-v1-only.
- SHA-256 `b58dcca7c3779f2b0539f988077f865c6c1ca040d23a44646d89b04a398940df`.

The append-only attribution ledger has 107 lines but 95 latest effective verdicts; later records supersede earlier records by file order. A record whose `attribution` is missing or null removes the effective verdict for that case. Reuse the existing loader rather than implementing another latest-wins interpretation.

### Evidence source map

| Source | Role | Important constraint |
|---|---|---|
| `reports/error_attribution_queue.json` | Frozen lineage and case cohort | Never regenerate; final 2d mapping is `p2d.resolved_item_id` |
| `reports/error_attribution_verdicts.jsonl` | Latest human stage attribution and rationale | Latest wins by file order; null/missing attribution acts as undo |
| `reports/error_attribution_gold_cases.json` | Materialized gold misses and full matching table | Open coverage notes are questions, not proven catalog gaps |
| `reports/error_attribution.json` and `.md` | Reconciled attribution totals | Summary only; do not substitute for case evidence |
| `reports/labels_v1_1.json` | Human truth spine, including successful uses | Currently untracked but hash-pinned by the frozen queue; stop on mismatch |
| `reports/review_queue.json` | Exact review cards, catalog IDs, claims, photos, dispositions | Join labels here for item-level positive evidence |
| `reports/review_analysis.json` | Review summaries and qualitative notes | Join qualitative notes by card ID |
| `reports/factorized_review_scorecard.json` | Fifty retained misnamed leads | Leads are model outputs, not truth; join by property and condition |
| `reports/review_verdicts.jsonl` | Raw review-verdict ledger behind the review-queue dispositions | Pinned by the frozen queue; stop on mismatch |
| `benchmarks/pass2a-prompt/gold/reference.json` | Gold photo-observation reference | Pinned by the frozen queue; gold is observation truth, not billable-condition truth |
| `docs/FINDINGS_catalog_3_2_deferred_issues.md` | Previously parked issues and constraints | Backlog input, not automatic evidence for a change |

At the verified baseline, the principal frozen source hashes are:

| Source | SHA-256 |
|---|---|
| Attribution verdicts | `166e8bec20641c0a8fb0ca5ddbd6c042f31b828447304ccbbbd4d585cd2a894e` |
| Gold cases | `77a7b515c594eb862bfc9ffbd0073556bc8a44f2fed644610f2412cb3bd0954a` |
| Labels v1.1 | `7f9b03017195144195550074b49dbb0488ed3ad33fdde905ba4e5867dfdbe3bd` |
| Review queue | `8512b87c2af18d1104069c15a5eda977d5fe79eb736f9aedd976d174f89d318a` |
| Review analysis | `3652ba3c3feb15800c064ffafd076d15cd9773de29c3d77c7b383fe26a259c9a` |
| Factorized scorecard | `60300ec5e957e42d2e443ab9493d0b9820b66498a2f57e3ac34a4479fcd9d103` |
| Review verdicts ledger | `0c7ca6a8b55d6dac69021a00e81315be1b15cef68ffd59fbe2973cb895a8d70f` |
| Gold reference | `253075132988ac2a1abbdb6e0815c1bd75e89d9ad982e745ddb9ce5a9f424543` |
| Generated 3.2 catalog | `51bf7e263ff98109d657ef730b0ce1a705598101f09843eb288b1bdc3e0eaa54` |
| Migration decisions | `47614d822bdd6b672d8596050b46839d04a8a6df0a04c18e052f4a63bd1903e8` |

These are baseline fingerprints, not permanent constants. Session 1 must record actual hashes and explain every mismatch.

### Claim-text provenance correction

The queue's `claim_text` was reconstructed by the queue builder from catalog review-card claim rendering. The run artifacts store Terra call fingerprints and IDs, not the exact request payload claim. Therefore the queue claim must not be described as a run-stamped Terra payload.

Current repository comparison found no `atomic_claim` changes between the evidence-era 3.1 catalog and the current 3.2 catalog, so the claim text is presently equivalent. Session 1 must prove and record that equivalence against the selected baseline instead of relying on this historical observation.

### Evidence baseline versus proposal baseline

The referenced frozen run artifacts use catalog 3.1, but the version string is not a unique identity: three distinct generated-catalog states in git history all carry `version: 3.1` (`58c073e`, `ae3630a`, `07ee112`). Every run the queue references — 26 run IDs, 17 canary and 9 production — records `catalog_sha256 = d1cef743f379918fa17f0684c245779180d7d692ebc8fd3e66b7c69bebf68347` in its `photo_intel_debug.json`. That value is written by `tools/renovation_architecture/catalog_projection.py` as a raw-bytes hash of the on-disk catalog, and it is the `07ee112` (2026-08-17) state of `tools/issue_catalog_kind_v2.json` as checked out on Windows with CRLF line endings; the LF git blob of the same state hashes differently (`20adb6e3…`), and the same CRLF rule reproduces the on-disk 3.2 hash from `a9ed7ad`. Session 1 must re-derive this mapping mechanically (artifact `catalog_sha256` values against the generated catalog's git history under the checkout's line-ending convention) and record the resolved commit, both hashes, and the run count per source rather than reusing these numbers.

The current generated catalog is 3.2, introduced by ancestor commit `a9ed7ad`. It is implemented, generated, and marked `publishable`, and v5 already applies its contextual-repair routing, but it has not been accepted: no live run has produced Sol decisions for the 33 package candidates the re-routing changed, and the manual package review and 15% headline-delta gate are still owed. Decided by Steven on 2026-09-01: that is not a blocker for this program (see the README baseline rule). Use:

- The `07ee112` catalog 3.1 state as the evidence/runtime baseline.
- The as-generated 3.2 catalog at `e9a7dc3` as the proposal baseline.

Because 3.2 is unvalidated, two cautions apply. Offline replay totals from `scripts/replay_renovation_architecture.py` exclude those 33 candidates and are an undercount on the 10 affected canary properties (`docs/FINDINGS_catalog_3_2_deferred_issues.md` §11), so no session may quote replayed dollar totals as a headline. And if 3.2 acceptance edits the decisions file during the program, that is ordinary baseline drift: stop, record the new hashes, re-count the decision table in Section 3, and re-confirm every approved diff before continuing.

Produce a field-level 3.1-to-3.2 semantic/retrieval/economic comparison before audit conclusions.

## 6. Candidate evidence population

The worklist should consider:

- All latest cases attributed to pass 2d.
- All appendix-misnamed cases.
- Terra-attributed misses whose rejection may involve catalog-introduced material, object, name, mechanism, or specificity commitments.
- The 50 retained factorized-verifier misnamed leads.
- Gold miss candidates and explicitly unresolved coverage rows.
- Previously deferred catalog findings as constraints or backlog references.

Pass-2a hallucinations should not independently seed catalog proposals. They may serve as counterexamples for a family already implicated by stronger catalog-shaped evidence.

Known exemplar case IDs in the historical input include `rc_128caa6212b8`, `rc_3a19f759f2d9`, `rc_45b9d2f6923c`, `rc_cf0c3ee00fcb`, and `rc_a22a241b3bf0`. They are starting points for inspection, not pre-approved diagnoses or changes.

## 7. Success evidence and evidence independence

The discrepancy cohort alone is insufficient. For every implicated item or family, compare failures with:

- The 80 v1.1 `supported_billed` labels joined to their exact review cards.
- The seven counted-agreement cases.
- The 31 counted-correct-rejection cases.
- Relevant v1.1 failure classes.
- The 27 qualitative review notes.

Do not infer that an item never succeeds because no successful example appears in the sample.

Define an independent evidence unit as:

- Runtime case: `(source, property_key, run_id, condition_id)`.
- Gold case: `(property_key, photo_id, gold_id)`.

Multiple model outputs, labels, or reviewer notes about the same underlying observation are annotations, not additional independent cases. Same-property or near-duplicate cases must carry a correlation flag.

Default evidence bar:

- At least two independent cases, preferably on different properties; or
- One case plus independent gold/factorized corroboration; or
- A documented single-case exception for a universal structural/schema fact that can be demonstrated mechanically and survives adversarial review.

If corroboration points to the same image or condition, call it method corroboration rather than a second independent case.

Historical review context indicates approximately 80% human self-consistency on the repeated v1.1 arm and about 6.4% Terra replica disagreement. A single verdict flip or a small live delta near this noise is not proof of improvement.

## 8. Item-family and semantic clustering method

For an implicated item, assemble a review family from:

1. Its migration parent and every successor/sibling.
2. Items with the same trade bucket and kind with overlapping scenes.
3. Items sharing work-item, estimate, or package-routing relationships where relevant.
4. Original hash-verified top-K candidates from the frozen artifact, when available.
5. Current proposal-baseline top-K candidates.
6. Reviewer-added or removed semantic neighbors with a written rationale.

Separately cluster observations by:

- Physical subject.
- Visible state or mechanism.
- Upstream kind.
- Scene.
- Unsupported commitment type: material, object, mechanism, severity, location, or scope.

One item may participate in more than one problem cluster. One semantic issue spanning several items should remain one proposal cluster.

Perform adjudication and diagnosis cluster by cluster so a reviewer is not asked to resolve unrelated ontology, stage-ownership, billability, and policy questions in one undifferentiated pass. For each imperfect mapping, explicitly distinguish an exact semantic match, an operationally equivalent mismatch, and an operationally consequential mismatch. Only the last category normally justifies structural catalog growth unless stronger retrieval or verification evidence demonstrates a need.

## 9. Diagnosis and adversarial review

Each cluster receives one primary cause and optional contributing causes. Before calling anything missing coverage, search the full current catalog, migration siblings, frozen candidate neighbors, and current candidate neighbors for an adequate existing item.

The independent reviewer should attempt to disprove catalog ownership by asking:

- Does a suitable current item already exist?
- Is the upstream kind wrong?
- Is scene filtering or retrieval metadata the cause?
- Was the right item retrieved but pass 2d selected another?
- Did Terra adjudicate the wrong semantic detail despite a suitable claim?
- Would a proposed wording create unsupported commitments?
- Would a change steal mappings from neighbors or damage reviewed successes?
- Does the change alter product eligibility, package behavior, or economics?

Every proposal, deferral, and no-change conclusion must survive this challenge or clearly record why it did not.

## 10. Proposal and approval contract

The Session 2 proposal artifact should be machine-readable and deterministically rendered. Each cluster should include:

- Stable proposal ID and status.
- Baseline commit and source hashes.
- Implicated items and item family.
- Evidence-era 3.1 semantics and proposal-baseline 3.2 semantics.
- Supporting cases, successful uses, correct rejections, and counterexamples.
- Independence/correlation assessment.
- Primary diagnosis and rejected alternative owners.
- One outcome: native catalog decision, non-catalog action, defer, or migration-system gap.
- Near-exact decisions-file diff when natively expressible.
- Semantic before/after summary.
- Retrieval, scene, kind, lexical-shortcut, economic, package, and product-policy implications.
- Evidence-bar result, confidence, adversarial result, and unresolved questions.
- Blank human disposition.

No decision-file or generated-catalog edit occurs during proposal generation. An in-memory dry run against a deep copy is allowed to prove that a proposed native diff is expressible and produces only the described output.

Human review may approve, modify, reject, defer, or reclassify each proposal. Implementation begins only from a locked approval manifest tied to exact proposal IDs, diffs, baselines, and hashes.

## 11. Validation tiers

### Tier 1 — deterministic

- Generate twice and require byte-identical outputs.
- Run v2 catalog validation explicitly.
- Run catalog, migration, retrieval, and scene tests.
- Run the broader test suite with a writable temporary root.
- Require only approved decisions and generated artifacts to differ.
- Compare semantic and economic fields and require the implementation diff to match the reviewed diff.

### Tier 2 — retrieval

- Probe the embeddings sidecar with a real POST; health status alone is insufficient.
- Replay every affected observation and selected counterexample with exact frozen kind and scene.
- Capture top-K ranks/scores, filtering, guardrails, and lexical-shortcut behavior for baseline and candidate.
- Reject unintended neighbor hijacking, negative-control regression, or lost reachability.

The existing catalog-resolution benchmark remains useful as a gross regression test but is not authoritative for affected observations because it supplies gold kinds and is saturated.

### Tier 3 — live and cost-gated

- Start from frozen pass-2c outputs and rerun pass 2d, projection, Terra, and relevant downstream behavior.
- Keep Pass 2a, 2b, 2c, unrelated prompts, and model routing frozen.
- Run targeted clusters first.
- Run the full frozen 18-property canary only if targeted cases pass and cost is explicitly authorized.
- Compare resolutions, Terra verdicts/rationales, dispositions, accepted conditions, packages/totals, and error migration.

Stored 3.1 replicas provide variance context. Package/economic conclusions require an appropriate fresh 3.2 control because stored evidence predates 3.2. That fresh control is also the Catalog 3.2 acceptance data owed to the separate 3.2 thread; preserve and hand off its package outputs, but do not decide 3.2 acceptance inside this program. A full replica has historically cost roughly 3.98 million tokens, so batch approved changes into one run and repeat only when a result is near the noise floor or otherwise inconclusive.

## 12. Session chain and authority boundaries

| Session | May change | Must not change |
|---|---|---|
| 1 | Offline audit scripts, tests, evidence reports, handoff | Decisions, catalogs, prompts, runtime behavior |
| 2 | Proposal data/document, proposal tooling/tests, handoff | Decisions, catalogs, runtime behavior |
| 3 | Adversarial review artifacts and handoff | Proposals, decisions, catalogs, runtime behavior |
| Human gate | Approval manifest | Repository implementation |
| 4 | Approved decisions, generated outputs, focused tests, handoff | Unapproved proposals, unrelated pipeline behavior |
| 5 | Minimal validation tooling/reports, handoff | Approved semantic decisions, live provider behavior |
| 6 | Live evaluation artifacts/reports, handoff | Decisions, prompts, unrelated runtime code, publication |

## 13. Source reading map

All sessions should use only the subset relevant to their task, but the principal sources are:

- `scripts/migrate_catalog_kind_v2.py`
- `tools/catalog_migrations/kind_v2_decisions.json`
- `tools/catalog_validation.py`
- `tools/catalog_embeddings.py`
- `tools/scene_classifier_orchestrator.py`
- `tools/pipeline_config.py`
- `scripts/build_error_attribution_queue.py`
- `scripts/error_attribution_report.py`
- `docs/RESULT_error_attribution_20260831.md`
- `docs/RESULT_factorized_replay_20260831.md`
- `docs/FINDINGS_catalog_3_2_deferred_issues.md`
- `docs/STATE_kind_ontology_program.md`
- The evidence sources listed in Section 5.
