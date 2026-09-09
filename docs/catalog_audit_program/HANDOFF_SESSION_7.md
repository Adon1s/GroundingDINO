# Handoff — catalog audit session 7 (bounded policy checkpoint)

Date: 2026-09-08

Session/task title: Bounded catalog policy checkpoint
(`08_SESSION_BOUNDED_POLICY_CHECKPOINT.md`), Phases 2 and 3 — decisions taken, two operations
implemented and validated.

**Status: COMPLETE. Steven decided all eight open questions and adjudicated a second CAP-013 card.
Two approved operations are implemented on three commits over `6e67eaa`, with 24/24 Tier 1 checks
and 12/12 Tier 2 rules passing. No provider call was made and no token was spent. Nothing is
published, no production artifact was mutated, and every frozen artifact is byte-identical.**

## Outcome

| Exit criterion (charter) | Result |
|---|---|
| Every in-scope finding has a change, an accepted limitation, or a deferral with a trigger | yes; 14 dispositions in `reports/catalog_audit_approvals_v2.json` |
| All nine families dispositioned | yes (D1 for trim, D6 for the other eight) |
| All six deferred CAPs dispositioned with triggers | yes; CAP-008 resolves with D1, CAP-022 subsumed |
| Legitimate-work losses resolved or explicitly accepted | yes; losses are enumerated in the decision record section 3 |
| Generated outputs, contracts and regression checks pass | 24/24 Tier 1, 12/12 Tier 2, suite green against the base's known failures |
| No new wrong-neighbour billing hidden in aggregates | every changed row is enumerated individually in the Tier 2 report |
| Candidate has hashes, fingerprint, inventory and a cutover plan | yes, below |
| Pass 2d worklist references current ids and separates axes | yes, `reports/catalog_checkpoint_pass2d_worklist.json` |

## Repository state

- Baseline arm: main checkout, `9afe0fa`, `terra_factorized_verifier`, tracked-clean, unchanged.
- Candidate arm: worktree `C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4`, branch
  `catalog_audit_session4`, now at **`7ddd52ab1fb198f3242d857e700cab23527327af`**, tracked-clean.
- Three new commits on `6e67eaa`:

| Commit | What |
|---|---|
| `637a7c0` | generator support: `route_override` and `scene_groups` admitted on carryover overrides |
| `ea34be5` | the two approved ops, regenerated catalog, focused tests, pin updates |
| `7ddd52a` | the two CAP-007 successor gold cases in a new frozen benchmark slice |

- Nothing is pushed. The branch is local-only, as the whole program is.
- Program artifacts remain untracked in the main checkout, matching the program's posture. The
  renderer, the validation harness and their tests are untracked in both locations; a first commit
  attempt swept three of those copied test files in and was corrected before the commit stood.

## The decisions

Steven answered all eight in chat on 2026-09-08, against evidenced alternatives with their measured
losses. Verbatim, with reasoning and the accepted losses:
**`docs/DECISION_RECORD_catalog_checkpoint_20260908.md`** (`916cff09…e567`).

| Id | Decision |
|---|---|
| D1 | trim → `route_override: no_action`, **without** CAP-022 (declined as unnecessary, not deferred) |
| D2 | generator admits `route_override` and `scene_groups` on carryovers; `require_any` deliberately not admitted |
| D3 | the ten blinds omissions accepted as presentation-only and re-homed to the Pass 2d worklist |
| D4 | wallpaper: approve the bathroom scene exclusion, with the 33-row LLM-path residual accepted unmeasured |
| D5 | CCF-14 ruled: matched human gold is evidence of the observation it states |
| D6 | tranche 2: no change for cabinets, appliances, doors and paint, with triggers |
| D7 | no supporting-context code; a gated design is recorded and trim keeps its affinity block as its landing pad |
| D8 | kitchen missing trim accepted as represented by the cabinet and counter scope |

**CAP-013 moved on evidence.** Steven adjudicated `rc_7057c3c171e5` (redfin_10952874, photos 018 and
027) as *"missing base trim — real work on a second property."* CAP-013 now has two human-adjudicated
units on two properties, so its evidence bar is **met** and it is no longer evidence-blocked. It stays
schema-blocked and unauthorized: Q-6 stands and nothing is implemented. A later proposal must decide
the subject breadth the two units span, because they are different physical subjects (kitchen cabinet
casework and counter edge; stairwell wall base trim), and must account for the overlap with
`baseboard_wear_scuffs`, whose `embed_text` already advertises missing trim while its claim states wear.

## The implemented change

Two ops, applied with the renderer's `apply_ops` straight from the approval:

```
dated_interior_trim      /successors/0/overrides/route_override -> "no_action"
dated_wallpaper_present  /successors/0/overrides/scene_groups   -> [kitchen, bedroom, living_areas, utility]
```

The generated catalog diff is **two lines**: one `route_override` added, one `"bathroom"` removed.
The migration manifest does not change at all.

| Artifact | sha256 |
|---|---|
| `tools/catalog_migrations/kind_v2_decisions.json` | `4cd0bc073f6fcdab3bde93993ed05a6199645e756e0df4b816230bd047a3b005` |
| `tools/issue_catalog_kind_v2.json` | `787964013d083368403524c610df54cc9858a8f6c1a1e5c9535c7204c1e67a2f` |
| projection fingerprint | `185a3923ad0acc0e66417f90c4395d24ec00557ff1d2626a7cb5578b3d8be28b` (verified independently against the on-disk catalog) |
| route counts | `no_action` 10 → **11**, `work` 98 → **97**, 129 items |

## Validation, all provider-free

**Tier 1 — 24/24** (`reports/catalog_checkpoint_tier1_proof.json`). Double generation byte-identical;
the on-disk catalog equals a fresh generation; the decisions file equals `apply_ops(base, approved
ops)` rebuilt independently from `6e67eaa`; the generated diff equals the approval's
`expected_generated_changes`; no item added or removed and order preserved; exactly two items changed
in exactly one field each; both validators zero errors with the warning count unchanged;
`atomic_claim` unchanged for all 129 items; exactly one terminal route and one observable changed;
trim has no work policy, which is what stops it billing.

**Tier 2 — 12/12** (`reports/catalog_checkpoint_tier2.json`). Before and after snapshots over the
1,969-row corpus sharing one vector cache; the after-run needed **zero new embeddings**, because
neither op touches embedded text.

| Measurement | Value |
|---|---|
| rows whose candidate list or shortcut changed | 33 |
| first-candidate changes | 7 |
| lexical shortcut changes | **0** |
| maximum score movement | 1.79e-07 |
| trim-owned corpus rows still able to reach trim | 108 of 108 |

**Per-op isolation** (`reports/catalog_checkpoint_per_op_isolation.json`). The combined arm cannot
attribute effects on its own, because removing the wallpaper item from a bathroom list frees a top-8
slot. Measured separately, with the base variant self-checked against the recorded snapshot
(0 mismatches): **the trim op changes 0 rows; the wallpaper op changes exactly the 33.** All five
rows where trim's list membership moved are inside those 33, and trim is `no_action`, so a promotion
cannot create a billing.

**Package effect** (`reports/catalog_checkpoint_package_effect.json`), 17 replayable artifacts. The
trim-only arm reproduces the frozen 2026-09-08 measurement exactly — 44 trim work items (27
standalone at $2,242–$11,212, 17 absorbed) and candidate classes 77 identical / 15 support-list-only
/ 9 shape changes / 1 disappearing — which is this run's instrument check. **The wallpaper op changes
no package candidate at all.**

## Three findings a later session would otherwise rediscover

1. **The wallpaper fix is prospective only.** The replay still shows both charges on
   redfin_25809814 `bathroom_primary`, and that is correct: `replay_property` reuses stored
   `observed_conditions` verbatim and never re-runs Pass 2d, so a retrieval-time fix cannot appear
   in a replay of a 3.1-era artifact. Tier 2 is where the fix is demonstrated. Stored artifacts keep
   the duplicate until the cutover re-resolves them.
2. **The real combined candidate cannot be package-replayed on the pinned artifacts at all**, because
   CAP-007 removed `dated_window_treatment_valance`, which every one of them names. That is
   CCF-8/S4-2 behaving as approved. Pruning those conditions to force a replay was tried and
   rejected: it orphans `terra_calls` references and would contaminate support lists. The package
   effect above is therefore measured with the frozen instrument, which isolates this checkpoint
   from CAP-007's cutover.
3. **`route_override` on a `no_action` item leaves `package_affinity` as dead configuration**, and no
   validator forbids the combination. Retained deliberately under D7 as the landing pad. Do not
   remove it as unused.

## Files created or changed

Untracked in the main checkout, matching the program's posture:
`docs/DECISION_RECORD_catalog_checkpoint_20260908.md`, `reports/catalog_audit_approvals_v2.json`
(`3c939bf5…`), `reports/catalog_authoring_surface_v2.json`, `reports/catalog_checkpoint_tier1_proof.json`,
`catalog_checkpoint_tier2.json`, `catalog_checkpoint_per_op_isolation.json`,
`catalog_checkpoint_package_effect.json`, `catalog_checkpoint_affected_artifacts.json`,
`catalog_checkpoint_pass2d_worklist.json`, `catalog_checkpoint_validation_pins.json`,
`reports/catalog_audit_live_experiment_manifest_v2.json`, both retrieval snapshots, this handoff, and
the implementation scripts under
`artifacts_canary/catalog_checkpoint_20260908/provenance/checkpoint_implementation/`.

Modified untracked tooling: `scripts/render_catalog_audit_proposals.py` (current-surface loader and
`--write-current-surface`), `scripts/catalog_audit_validation.py` (pins parameterized by
`--manifest`, Session 5/6 values as defaults), and their tests. Copied to the worktree so both
locations agree.

## Integrity

Every frozen artifact re-hashed unchanged at the end: approvals, evidence, proposals, adversarial
review, redraft, renewed review, gate decisions, live-check results, the ruled Session 6 bundle, the
photo review, the packet, the CAP-022 proposal, the trim photo review and the replay corpus. **Zero
drift.** `reports/catalog_audit_approvals.json` was never edited. The evidence bundle was not
rebuilt. The dev and holdout benchmark slices are untouched in git. The Session 6 Terra ledger is
unchanged, confirming no provider call.

One hash drift was found and explained rather than reconciled: the packet pins the source manifest at
`bd4003f8…` while disk has `90d67401…`, because the manifest was regenerated 44 seconds after the
packet and now pins the final packet hash, which matches disk. Decision record section 0.

## Test state

Full suite in the candidate arm: **2804 passed, 21 skipped, 6 failed** — the same six by-design
failures as the base commit, no new ones. One of the base's six is now **fixed**:
`test_every_split_successor_is_gold_somewhere` passes, having been red since Session 4.

Remaining five, all pre-existing and deliberate: the stored-artifact stale-id recompute (S4-2), the
three audit-renderer parity tests and both renderers' `--check` in the candidate arm (S4-5), and
`test_review_analysis::test_full_run_against_frozen_inputs` (worktree environment gap). The renderer
parity failure now names the generator as a third drifted input, which is the frozen bundle
correctly refusing a widened surface.

## Unresolved and owed

1. **No live confirmation.** Trim's zero-dollar display is derived from Session 6's stored verdicts
   against the new route, not observed. `reports/catalog_audit_live_experiment_manifest_v2.json`
   carries the candidate identity and three live questions with budgets **null**.
2. **The D4 residual is accepted unmeasured.** The 33 bathroom rows lose a wrong-subject distractor;
   what Pass 2d then selects is not measurable offline. The cheapest instrument is a local Pass 2d
   ×5 on exactly those rows, which was not authorized.
3. **Fresh Sol decisions are owed** for 15 support-list-only plus 9 shape-changed plus 1
   disappearing candidate. This is also the Catalog 3.2 acceptance data (S6-7).
4. **The cutover is planned, not executed.** 97 stored artifacts across six roots and 18 properties;
   plan in `reports/catalog_checkpoint_affected_artifacts.json`.
5. **The new benchmark slice has never been scored** — scoring needs a Pass 2d call.
6. **CAP-013 is now evidence-eligible** and needs a subject-breadth decision before any proposal.

## Required next task

Pass 2d diagnosis against this catalog, per the agreed sequence: catalog → Pass 2d → combined
whole-property backend acceptance. Start from
`reports/catalog_checkpoint_pass2d_worklist.json` (26 rows re-read against the final catalog, plus
the ten blinds rows accepted under D3 as a new category), and read the axes as diagnostic categories
rather than promised fixes. No prompt, model or threshold tuning is authorized by this handoff.

## Do not redo

- Do not re-run Tier 1 or Tier 2 to confirm them; both are deterministic, reported check by check,
  and their scripts are preserved under the checkpoint provenance directory.
- Do not try to package-replay the real combined candidate on the pinned artifacts, and do not prune
  stale conditions to force it. See finding 2.
- Do not read the replayed wallpaper double charge as the fix failing. See finding 1.
- Do not remove trim's `package_affinity`, add `require_any` to it, or edit its claim.
- Do not admit `require_any` to carryover overrides: CAP-022 was declined, and S6-12 stays a
  recorded gap on purpose.
- Do not edit a frozen slice, `reports/catalog_audit_approvals.json`, or the evidence bundle.
- Do not add a runtime alias for the removed CAP-007 parent.
