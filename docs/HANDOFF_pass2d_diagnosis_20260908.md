# Pass 2d diagnosis — context handoff

Prepared 2026-09-08. **Context only: this is not an experiment plan, an implementation proposal, or authorization to spend tokens.**

## Steven's request and direction

Steven wants a fresh session to diagnose Pass 2d after the catalog checkpoint. His instruction for this handoff is: “Let the new session do all of the planning and assumption making. You just need to provide the relevant files, context, etc.”

The broader intent is to get the backend into a useful, sufficiently reliable state before starting up and promoting RenoIntel. Frontend contract adoption and pricing are deferred until after the backend work. The recorded sequence was to finish the catalog work and related findings, then address Pass 2d. The bounded catalog checkpoint has now finished. That does not mean every catalog idea was implemented or every release acceptance obligation is complete.

This is **diagnosis**, not a predetermined prompt-tuning exercise. The next session owns the investigation plan, interpretation of evidence, assumptions, possible interventions, validation design, and stopping criteria. Steven wants fewer unnecessary handoffs and repeated validation cycles, without combining unrelated projects. Backlog entries are not promises to fix everything. Preserve explicit human decisions; explain any recommendation to reopen one.

No implementation, provider experiments, publication, or new task creation is authorized by this handoff. Read-only investigation and planning can begin from these records; any subsequent action needs authority from the new conversation rather than an old session's budget or proposed plan.

## Repository state: historical paths no longer identify historical code

Backend: `C:/Users/Steven/PycharmProjects/realtorvision-backend`.

At preparation, main is on `master` at `fb945b02350709af65ea285c1aa859c24c2da442`. It contains the catalog checkpoint. The tracked tree was clean before this handoff was added; there are pre-existing untracked artifacts. Nothing in those artifacts was changed for this handoff.

Relevant history:

| Commit | Meaning |
|---|---|
| `9afe0fa` | Historical main/baseline used by the catalog program |
| `6e67eaa` | CAP-007 blinds/fabric split |
| `637a7c0` | Generator carryover overrides admit `route_override` and `scene_groups` |
| `ea34be5` | Approved trim and wallpaper operations |
| `7ddd52a` | Two CAP-007 successor gold cases in a new benchmark slice; checkpoint candidate |
| `617144d` | Preserves previously untracked catalog audit records, tooling, and tests in version control |
| `29a498f` | Repository tracking policy |
| `fb945b0` | Test-only frozen-label path fix |

The candidate worktree also exists at `C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4`, branch `catalog_audit_session4`, at `7ddd52a`. Older instructions calling the main directory the unchanged `9afe0fa` baseline are stale. So are instructions saying the audit records and program tests must remain untracked. Consult current Git state before relying on either description.

Current catalog bytes were verified against the checkpoint pins:

- `tools/issue_catalog_kind_v2.json`: SHA-256 `787964013d083368403524c610df54cc9858a8f6c1a1e5c9535c7204c1e67a2f`.
- `tools/catalog_migrations/kind_v2_decisions.json`: SHA-256 `4cd0bc073f6fcdab3bde93993ed05a6199645e756e0df4b816230bd047a3b005`.

The frontend path `C:/Users/Steven/IntelliJProjects/renointel-prod` exists. Its frozen artifacts are relevant evidence even though frontend work is deferred. Older documents name other frontend locations. No frontend code or deployment state was audited for this handoff.

All following paths are relative to the backend root unless stated otherwise.

## Primary context and authority

| File | What it supplies |
|---|---|
| `docs/catalog_audit_program/HANDOFF_SESSION_7.md` | Completed checkpoint, validation, remaining live questions, and replay limitations; repository-state section is historical |
| `docs/DECISION_RECORD_catalog_checkpoint_20260908.md` | Steven's D1–D8 decisions, accepted losses, corrections to earlier claims, and CAP-013 adjudication |
| `reports/catalog_audit_approvals_v2.json` | Exact authorized checkpoint operations and dispositions |
| `reports/catalog_audit_approvals.json` | Earlier CAP-007 authorization, preserved separately |
| `reports/catalog_checkpoint_pass2d_worklist.json` | Consolidated diagnostic leads reread against the checkpoint catalog |
| `docs/catalog_audit_program/FOLLOW_UP_REGISTER.md` | Source lineage, confusion families, human rulings, and follow-up dispositions |
| `docs/RESULT_catalog_audit_live_validation_20260907.md` and `reports/catalog_audit_live_validation.json` | Session 6 live evidence and subsequent rulings |

For disputed details, follow the referenced case and human decision. Earlier packets and proposals are retained history, not competing current instructions. `docs/DECISION_PACKET_catalog_policy_checkpoint.md` contains the evidence/options before adjudication; the decision record explicitly corrects parts of it.

## Accepted catalog state relevant to diagnosis

The original audit was six sessions. Session 6 finished; its candidate was rejected under Steven's “fix first” ruling on S6-1. The later bounded checkpoint, called Session 7, resolved that condition with approved operations and deterministic validation. It was not an originally planned seventh audit session.

- **Trim:** `dated_interior_trim` remains retrievable but routes to `no_action`. Plain/basic/builder-grade presence must not bill by itself. CAP-008 is resolved by this operation and validation. CAP-022's proposed trim subject gate was **declined**, not left awaiting implementation; removal/gating experiments exposed migration toward billable neighbours, including wood paneling for built-ins. Those retrieval findings do not by themselves establish future live model choices.
- **Generator:** `route_override` and `scene_groups` are now supported carryover overrides. `require_any` was deliberately not made authorable there. The associated S6-12 gap remains intentionally open.
- **Wallpaper:** the generic wallpaper item excludes bathroom scenes. This is prospective: existing stored conditions are not reselected merely because catalog eligibility changes. Steven accepted the 33 affected bathroom rows' future LLM-selection residual unmeasured. “Wallcovering” without “wallpaper” remains a documented reachability concern for the bathroom-specific item.
- **Blinds/fabric:** the CAP-007 split remains. Ten observation omissions were accepted as presentation-only and transferred to the Pass 2d worklist; they are not ten newly promised fixes. S5-1, the missed fabric condition, remains accepted and closed.
- **Package context:** nonbillable trim does not create a work item and therefore does not count as package support. Steven declined new supporting-context code at this checkpoint and accepted the measured effects. Trim's retained affinity configuration is a possible future design landing point, not evidence that the current behavior is accidental.
- **Missing trim:** D8 accepts the reviewed kitchen work as represented by cabinet/counter scope. Earlier statements that this loss was “not accepted” are superseded. CAP-013 now meets its two-property evidence bar, but remains schema-blocked and unauthorized under Q-6; the kitchen casework and stairwell wall-base subjects are different, so a later proposal still needs a breadth decision.
- **Other catalog families:** the decision record leaves cabinets, appliances, doors, and paint unchanged with evidence triggers. Electrical lights/fans remain quarantined from billing; landscaping remains `no_action`. Deferred ideas are not automatic prerequisites to this diagnosis.
- **Evidence policy:** matched human gold supports the observation it actually states; it does not automatically establish billability. Automated `gold_incomplete` tags are not equivalent evidence.

## The Pass 2d evidence map

`reports/catalog_checkpoint_pass2d_worklist.json` contains 26 diagnostic rows plus ten accepted blinds omissions. Its recorded classification is 17 `incorrect_selection_reachable`, five `shortcut_or_rank1_selection`, three `candidate_absence_or_upstream`, and one `naming_question_after_split`. These are worklist classifications, not freshly verified causal findings or 26 promised fixes.

Rows carry case references, selected and suggested items, confusion-family descriptions, historical ranks, route information, and source notes. They mix human judgments, model-judge leads, naming concerns, and possible economic consequences. `billing_relevant_now` is not a measured price delta. A worklist axis is not a substitute for the actual trace's `resolution_path`.

There are reasons to examine the underlying evidence rather than treating annotations as gold: the accepted-omissions section assigns a blinds `better_item` even to a brown-valance description and makeshift coverings, whose source decisions have additional qualifications. Historical ranks and possible mechanisms likewise need their original context. The handoff does not resolve those interpretations for the new session.

Supporting evidence:

| File or directory | Use and limitation |
|---|---|
| `reports/catalog_audit_proposals.json` | Original proposal clusters, case-level reasoning, counterexamples, and Pass 2d follow-ups |
| `reports/error_attribution_queue.json` | Frozen review lineage and source references |
| `reports/error_attribution_verdicts.jsonl` | Human attribution judgments and appended revisions; use the existing loading semantics, including null undo/revisions |
| `reports/error_attribution_gold_cases.json` | Materialized gold cases |
| `docs/RESULT_error_attribution_20260831.md` | Historical separation of upstream false claims, downstream misses, and billability failures; not a current post-checkpoint error rate |
| `docs/RESULT_label_v1_1_20260829.md`, `reports/labels_v1_1.json` | Label repair, judgment axes, and limits of the evidence |
| `reports/catalog_audit_replay_corpus.json` | 1,969 product-filtered stored observations used for deterministic catalog analysis |
| `reports/catalog_audit_replay_snapshot_checkpoint_base.json`, `reports/catalog_audit_replay_snapshot_checkpoint_after.json` | Checkpoint retrieval comparisons, not new live Pass 2d outcomes |
| `reports/catalog_checkpoint_tier2.json` | Twelve validation rules and all 33 changed bathroom rows in `changed_rows_in_full` |
| `artifacts_canary/catalog_audit_session6_20260907/` | Local live-run evidence and repeat/provenance outputs; run directories are not versioned |
| `artifacts_canary/catalog_checkpoint_20260908/provenance/checkpoint_implementation/` | Local checkpoint instrumentation and provenance |

`docs/HANDOFF_quality_program_direction_20260829.md`, `docs/RESULT_factorized_replay_20260831.md`, and `docs/FINDINGS_catalog_3_2_deferred_issues.md` supply broader context. The quality handoff explains the emphasis on fewer hallucinations and more useful observations that support packages, rather than price calibration. Its historical Session F/G/H instructions do not define the next session's plan. The deferred inventory `docs/REPORT_deferred_work_inventory_20260907.md` is navigation, not a list of outstanding promises.

## Code and benchmark entry points

Pass 2d consumes text observations and an upstream kind, not the original photo. Retrieval, deterministic resolution, and model selection are distinct parts of the path; downstream billing and package behavior are additional stages.

| Location | Role |
|---|---|
| `tools/scene_classifier_orchestrator.py::resolve_observation_against_catalog` | Shared production/benchmark resolver, kind validation, candidate retrieval, result and debug traces |
| `tools/catalog_embeddings.py` | Candidate provider, embeddings, kind/scene eligibility, guardrails, retrieval and ranking |
| `tools/pipeline_common.py::term_matches` | Term-matching behavior used by routing logic |
| `tools/scene_classifier_passes.py` | `PASS_2D_USER_PROMPT_TEMPLATE`, prompt version/hash, `format_candidates_text`, `_resolve_candidate_via_lexical_shortcut`, and `run_pass_2d`; upstream 2b/2c prompts are here too |
| `tools/pipeline_config.py`, `tools/vlm_client.py` | Runtime configuration and model request behavior |
| `tools/issue_catalog_kind_v2.json`, `tools/catalog_migrations/kind_v2_decisions.json`, `scripts/migrate_catalog_kind_v2.py` | Generated catalog, editable decision source, and generator |
| `tools/renovation_architecture/` | Projection, conditions/evidence, disposition, work items, packages and reconciliation downstream of selection |
| `scripts/benchmark_catalog_resolution.py` | Existing benchmark through the shared resolver, with model configuration and named case slices |
| `benchmarks/catalog-resolution-v2/` | README, dev/holdout cases, model/embedding configs, frozen legacy retrieval, and historical results |
| `benchmarks/catalog-resolution-v2/cases_checkpoint_20260908.json` | New frozen two-case CAP-007 successor slice; landed without live scoring at checkpoint completion |
| `tests/test_catalog_resolution_benchmark.py`, `tests/test_catalog_checkpoint_trim_no_action.py`, `tests/test_catalog_checkpoint_wallpaper_scene.py` | Existing deterministic coverage |
| `scripts/catalog_audit_validation.py`, `scripts/catalog_audit_live_validation.py` | Program-specific instruments; their old manifests and commands are not an automatically valid new experiment |

At preparation the prompt version is `pass_2d_exact_kind_v2`. Candidate rendering includes ID, name, trade, kind, description, selected support phrases, and generic/hidden flags; it is not the entire catalog record. The shortcut can bypass the LLM based on top-candidate score/margin and lexical evidence. Empty inputs/candidates can also return without a model call. Consequently, neither a selected ID nor a null result alone establishes what the model did.

The historical live setup used local Qwen for Pass 2d, separate from Terra's verification role. Benchmark configs are file-driven and may differ from runtime configuration. Old documents mention different endpoints and defaults; no endpoint was contacted or model readiness verified for this handoff. The ontology configuration also matters: legacy defaults must not be mistaken for the intended v2 path.

The benchmark README includes historical catalog counts and an earlier catalog-tuning workflow. Treat it as documentation of that benchmark and its dev/holdout distinction, not an instruction to predetermine this diagnosis or rewrite frozen cases. Current catalog and prompt fingerprints matter to comparisons.

## What the checkpoint validation does and does not establish

Recorded results: 24/24 Tier 1 checks and 12/12 Tier 2 rules passed. Trim alone changed zero retrieval rows. Wallpaper changed 33 rows, all bathroom, with zero lexical-shortcut changes. No provider tokens were spent in Session 7. See `reports/catalog_checkpoint_tier1_proof.json`, `reports/catalog_checkpoint_tier2.json`, and `reports/catalog_checkpoint_validation_pins.json`.

Package instrumentation reproduced 77 identical candidates, 15 support-list-only changes, nine shape changes, and one disappearance under the trim operation. The package corpus is canary-weighted: its inventory has 26 entries, of which 17 are v5-replayable, eight predate v5, and one is unpinned. Those are not representative production-wide rates.

Several limits are already known:

- Stored package replay reuses conditions; it does not rerun Pass 2d. It cannot measure a new prompt's choices or demonstrate that prospective wallpaper eligibility has repaired historical selections.
- The true combined catalog cannot be package-replayed on those pinned artifacts because they name CAP-007's removed parent `dated_window_treatment_valance`. Pruning conditions to force compatibility was tried and rejected: it orphaned `terra_calls` and contaminated the meaning of support lists. No runtime alias was approved.
- Changed candidate payloads can invalidate stored Sol decisions. A package disappearing after an incompatible replay is not evidence that the work is unwarranted.
- An earlier isolation measurement accidentally disabled the kind filter and was rejected. Its numbers are not accepted results.
- The 33 wallpaper retrieval rows, the earlier 33 changed package candidates from catalog 3.2, and the trim package-effect counts are different populations.

Fresh live confirmation, compatible Sol decisions, and the 97-artifact historical cutover remain separate unfinished obligations. Their necessity and timing for a later release must be distinguished from what is necessary to diagnose Pass 2d. This handoff does not authorize or prescribe doing them together.

## Frozen inputs, manifests, and conflicting status summaries

`reports/catalog_audit_live_experiment_manifest_v2.json` records checkpoint live questions with **null budgets** and an awaiting-authorization status. Its suggested instruments are proposals, not this handoff's plan. Its baseline arm still names the main directory at historical `9afe0fa`; that directory now contains later code. Neither it nor the original manifest should be assumed executable unchanged.

`docs/STATUS_INDEX_preserved_records.md` explains the preservation commit, line-ending policy, and known pin drift. In particular, `scripts/catalog_audit_validation.py` changed after the original manifest was frozen to support manifest parameterization. Preserve the original evidence rather than changing a frozen manifest to conceal drift.

There is a status-summary conflict worth knowing: the preservation index calls the original live manifest “Never executed,” while the Session 6 result and ruled live-validation report record executed live experiments. Do not use that shorthand to erase the live evidence; consult the original results and provenance for what actually ran.

The preservation index records 22 external frozen input artifacts, with their hashes verified during preservation, archived under `D:/realtorvision-recovery/20260908/frozen-inputs/`. That directory and the frontend repository exist as of this handoff; this session did not independently rehash those 22 artifacts. `.gitattributes` deliberately preserves byte behavior for pinned evidence. Run directories remain local and unversioned, so a fresh checkout alone may not contain every linked artifact.

## Suggested opening message for the new session

> Read `docs/HANDOFF_pass2d_diagnosis_20260908.md` in `C:/Users/Steven/PycharmProjects/realtorvision-backend` and follow the relevant source links. Help me diagnose Pass 2d now that the catalog checkpoint is complete. This handoff is context, not a prescribed solution: make your own investigation plan, check the assumptions and current state, and determine what the evidence actually supports. Preserve my recorded catalog decisions and distinguish confirmed failures from diagnostic leads and accepted limitations. Start with investigation and planning; do not run model/provider experiments, implement changes, publish, or create new tasks without further authorization.
