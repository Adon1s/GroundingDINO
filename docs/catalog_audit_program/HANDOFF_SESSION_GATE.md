# Handoff — catalog audit human gate (between Session 3 and Session 4)

Date: 2026-09-06

Session/task title: Human review gate per `04_HUMAN_REVIEW_GATE.md`, run as a three-phase task: interim decision record →
redraft and renewed review with a live Terra check → explicit approval and the approvals manifest.

**Status: COMPLETE.** Every proposal id carries exactly one disposition in `reports/catalog_audit_approvals.json`; one
native change is approved with an exact, dry-run-proven diff; the proposal and review hashes match; no approved record
depends on a migration-system gap; no schema, generator, Terra-calibration or product-policy work was authorized.

## Outcome

| Disposition | Count | Ids |
|---|---:|---|
| `approved_with_modification` | 1 | CAP-007 (window-treatment split) |
| `deferred` | 6 | CAP-008, 010, 013, 014, 016, 018 |
| `reclassified_non_catalog` | 13 | CAP-001–006, 009, 011, 012, 015, 017, 019, 021 |

Two results reverse what the Session 3 handoff led one to expect. CAP-008 (dated interior trim), which the review had
sustained in direction, is **not** implementable as a wording change: both positive-only candidate states were tested
live against the three decisive cards and both failed the rule, decisively on the one card where the harness reproduced
production. CAP-007 (window treatments), which the review had marked insufficient as drafted, **is** approved once
redrafted under Steven's ruling that all blinds are `no_action`. The live check also showed that the single-condition
harness does not reproduce the stored batched verdicts on two of three cards, so the check establishes what the wordings
do in that harness, not a general property of Terra. The reading is in `reports/catalog_audit_renewed_review.json`
`live_check`.

## Repository state

- Starting commit: `9afe0fa5a0490a856abed507dccc4a02ed1de24c`, branch `terra_factorized_verifier`
- Ending commit: unchanged, nothing committed (Steven decides)
- Dirty state at start and end: no tracked file modified; every session output is an untracked working-tree file
- Pre-existing changes preserved: yes. A 17-file protected-hash snapshot (Session 1–3 artifacts, decisions, both catalogs,
  generator, validator, Terra prompt file, packet handoffs) was taken before any agent ran and re-verified after every
  stage; all 17 were byte-identical throughout. Both renderers' `--check` still exit 0.
- The decisions file is identical at the proposal baseline `e9a7dc3` and at HEAD (git blob `521279b2…`; on-disk CRLF
  hash `47614d82…03e8`), so a Session 4 candidate worktree may branch from HEAD.

## Inputs verified

| Input | Expected (HANDOFF_SESSION_3) | Actual | Result |
|---|---|---|---|
| `reports/catalog_audit_proposals.json` | `1411352c…3ef99` | `1411352c42e9d3a555c3a1561470323da2234447badc2869c9874c3e8fa3ef99` | match |
| `reports/catalog_audit_adversarial_review.json` | `d1149875…b5c4` | `d1149875399a9f30c6cd56e924731c7e207d90b603cb248f402c692c3438b5c4` | match |
| `reports/catalog_audit_evidence.json` | `43354104…29b7` | `43354104cc109c768efbe2d2f8b9a825f33e0882210b81e9f12b847ca67e29b7` | match |
| `tools/catalog_migrations/kind_v2_decisions.json` | `47614d82…03e8` | `47614d822bdd6b672d8596050b46839d04a8a6df0a04c18e052f4a63bd1903e8` | match |
| `tools/issue_catalog_kind_v2.json` | `51bf7e26…aa54` | `51bf7e263ff98109d657ef730b0ce1a705598101f09843eb288b1bdc3e0eaa54` | match |
| both rendered documents, photo review, packet, v1 catalog, generator, validator, HANDOFF_SESSION_2 | as pinned | as pinned | match |

The handoff's suggested prompt said "eight gate questions"; the artifact carries nine. All nine were ruled on
(Steven's correction #2, recorded in the errata appended to `HANDOFF_SESSION_3.md`).

## Files created or changed

| Path | Status | sha256 | Purpose |
|---|---|---|---|
| `reports/catalog_audit_gate_decisions.json` | new | `d89b2bf1149738d9b3e2d5a9222a5b4e163e2cf97dbb0c9f6396d595b849c980` | interim decision record: nine rulings, twenty dispositions, redraft brief, review concerns, live-check rule and ceiling, authorizations |
| `reports/catalog_audit_live_check_results.json` | new | `de47d13f1dfa5802e885027520ae318ab6a6166e8b609115eb2bdb60eaec74e1` | 23 live Terra calls (control, wording A, wording B) with rationales, tokens and the rule evaluation |
| `reports/catalog_audit_redraft.json` | new | `449f86170dd4b01916988b2000b900e560403bda12fb93cc7e7d7d27bb31811b` | lead-authored redraft of both native proposals: exact ops, dry runs, leak test and census recomputed from the pinned artifacts, corrected ledgers, lead corrections with evidence |
| `reports/catalog_audit_renewed_review.json` | new | `aa12e42a0205c0ede8c01362e15f3c16e47cffd9b37de8dd468e70400e8b538b` | one disposition per redraft on the Session 3 standard, the live check applied by the rule with decisiveness qualifications, eligibility per id, approval questions |
| `reports/catalog_audit_approvals.json` | new | `a725c89af131e94b04867f4f5a226bda495278a8ba28fdd7162ed6188d76c871` | the approvals manifest (04 schema); pins every stage artifact; Session 4 starts here |
| `docs/catalog_audit_program/FOLLOW_UP_REGISTER.md` | new | `268cd370d3652695f211ce568aed8a28116920ce76e3abe11d8d866e4d453203` | one index of every open item with pointers to its evidence; living document for Sessions 4–6 |
| `docs/catalog_audit_program/HANDOFF_SESSION_3.md` | errata appended | `fbaa1bb41e477fb7cb12c1669f78a3b24376e79b4ff86c6811e29be6bc437db5` | Steven's six corrections, append-only after the original last line |
| `docs/catalog_audit_program/HANDOFF_SESSION_GATE.md` | new | — | this file |

Not in the repository: the agent briefs, schemas, workflow scripts, the live-check harness (`terra_live_check.py`), the
read-only dry-run CLI, the six agent outputs and the run journals live in the session scratchpad
(`…\Temp\claude\…\33db7eb1-2348-4147-aa2d-c18fb197bbd1\scratchpad\gate_work\`). Everything load-bearing was carried into
the artifacts above with its evidence citations; the artifacts cite cases, units, items, photos and `file:line`, never an
agent.

## Work completed

- **Phase A (Fable 5.1).** Verified pins; opened the photos named in every disputed disposition; ran an in-memory dry run
  proving the CAP-008 edit shape expressible; put the nine gate questions to Steven and recorded his rulings; wrote the
  decision record; appended the errata; staged six agent briefs with isolation rules, six output schemas, two workflow
  scripts, an output validator, a spec filler and the live-check harness (single-condition production payload shape,
  unchanged system prompt, direct `call_terra_review`, results checkpointed per call).
- **Phase B (Opus 5).** Ran the two drafters in fresh contexts (2 agents, 501k subagent tokens), the live check
  (23 calls, 72,756 tokens, zero errors, zero `cannot_assess`), then the three refuters and the mechanical verifier
  (4 agents, 853k tokens). Every output passed schema validation twice (workflow and independent). One staging fix:
  the workflow scripts were normalised from CRLF to LF, which the approval dialog requires.
- **Phase C (Fable 5.1).** Read all six outputs; re-verified every load-bearing fact against the pinned artifacts, code
  and photos (gold g3 located in `gold.matching_rows`; labels; pinned photo-review rows; the term matcher's real
  semantics and the gate at `catalog_embeddings.py:439-450`; exact-kind facts from the run artifacts; an independent
  census of 48 conditions / 39 accepted); re-ran every dry run on the final ops; authored the redraft and the renewed
  review; reported eligibility; took Steven's seven approval answers; wrote the manifest; built the follow-up register.

## Decisions and invariants

- Steven's nine rulings (2026-09-05), verbatim in the decision record: Q-1 all blinds in any condition are `no_action`;
  Q-2 improvised coverings are presentation-only; Q-3 withdraw rc_9e889631af67 and rc_d0ffde500a92 under ruling 3 and
  treat rc_ceeb17f8e242 as a missing-trim loss; Q-4 positive-only house-style form, no wording endorsed; Q-5 ruling 3 is
  applied catalog-wide later, outside the program; Q-6 document CAP-013 and wait; Q-7 outbuilding damage keeps billing
  under the dwelling item for now; Q-8 no Terra calibration now, ruling-4 re-adjudication first; Q-9 the door item's
  subject includes the casing.
- Steven's seven approval answers (2026-09-06), in the manifest: A-1 CAP-008 deferred into the catalog-wide ruling-3
  sweep; A-2 rc_57c6faf79c7e recorded with CAP-013's candidate cards, unadjudicated; A-3 the split is approved with the
  lead-added fabric deny stems `shower` and `tub`; A-4 the blinds successor stays visible, no `defaultHidden`; A-5 curtain
  hardware keeps billing on the fabric successor; A-6 the makeshift-covering condition reaching neither successor is
  accepted; A-7 claim texts as drafted.
- Disposition convention (CCF-11 answered): `no_change` and `non_catalog_action` are one class,
  `reclassified_non_catalog`; only `approved*` is implementable; `approved_diff` is an ops list in the shape
  `apply_ops` consumes, null elsewhere.
- The approvals manifest is the only authorization. Session 4 implements CAP-007's seven ops and nothing else.
- Live-check rule as applied: any `supported` on a kill card blocks; recovery needs a majority; errored or
  `cannot_assess` calls are non-decisive and re-run once; the control arm is a sanity check whose mismatch flags but
  does not block; a card whose control already returns the candidate's required verdict is uninformative.
- Evidence discipline unchanged: adjudications are never rewritten; the redraft discloses where a drafter's photo
  reading conflicted with a pinned photo-review row and rests the classification on the ruling and the label note.

## Deviations from the session brief

| Deviation | Reason | Impact |
|---|---|---|
| Three stages instead of one manifest write | Steven's correction #3: answering the questions could not make either draft implementable; redrafting and renewed review had no assigned place | An interim decision record precedes the manifest; both are pinned |
| 23 live Terra calls inside the gate | The CAP-008 kill condition could not be settled offline; Steven authorized a 30-call ceiling in the decision record | The wording lever was disproved by measurement rather than argued |
| Multi-agent, multi-model execution | Steven's instruction to stage as Fable, run agents as Opus, review as Fable | Fresh-context drafters and refuters; every adopted finding re-verified by the lead |
| Errata append and follow-up register beyond "manifest only" | Authorized by Steven (W3; request of 2026-09-06) | Corrections recorded once; open items indexed |
| One lead amendment after the refuters ran | The drafted `shower curtain` guard matched only the literal phrase; the two added stems are narrowing and were re-tested with the real matcher | Disclosed in the redraft, the review and the manifest; approved under A-3 |

## Commands and verification

| Command/check | Result | Notes |
|---|---|---|
| `scripts\render_catalog_audit_proposals.py --check` | matches; validation ok, 0 errors, exit 0 | run after every stage |
| `scripts\render_catalog_audit_adversarial_review.py --check` | matches; validation ok, 0 errors, exit 0 | run after every stage |
| 17-file protected-hash snapshot vs disk | all unchanged | before, between and after every stage |
| `validate_outputs.py --stage drafters` / `--stage review` | PASS 2/2, PASS 4/4 | independent of the workflow's own validation |
| `terra_live_check.py` | exit 0; 23 calls, 72,756 tokens, ceiling 30 | evaluation recomputed independently by the mechanical verifier and by the lead: identical |
| dry runs on the final ops (CAP-008 A, CAP-008 B, CAP-007 split, split with `defaultHidden`) | all ok; all ops native; zero validator errors; expected item diffs | lead re-runs; the approval builder re-proves the CAP-007 diff again before writing |
| manifest verification | 20/20 ids once; vocabulary valid; nine pins equal fresh hashes; one approved record with seven ops | script in the scratchpad; results recorded in the manifest's `gate_outcome_checks` |
| `git status --porcelain -uno` | empty | at start and end |

## Outputs for the next session

| Artifact | Hash | Contract |
|---|---|---|
| `reports/catalog_audit_approvals.json` | `a725c89a…c871` | the authorization; `dispositions[CAP-007].approved_diff.ops` is the exact change; `gate_chain` names every stage artifact |
| `reports/catalog_audit_redraft.json` | `449f8617…811b` | successors, leak test, census and ledger behind the approved diff |
| `reports/catalog_audit_renewed_review.json` | `aa12e42a…538b` | why CAP-007 is eligible and CAP-008 is not; the conditions Sessions 5 and 6 own |
| `reports/catalog_audit_gate_decisions.json` | `d89b2bf1…c980` | rulings and dispositions; the live-check rule |
| `reports/catalog_audit_live_check_results.json` | `de47d13f…74e1` | raw calls and rationales |
| `docs/catalog_audit_program/FOLLOW_UP_REGISTER.md` | `268cd370…3203` | every open item; Sessions 4–6 append to section 9 |

## Unresolved questions and risks

1. **Nothing is committed, and 46 program artifacts are untracked** (every `reports/catalog_audit_*` file, the packet
   docs, the two rendered documents, the renderers and their tests). A clean candidate worktree cut from git will not
   contain any of them, and Session 4's brief requires one. Decide before starting Session 4: commit the artifacts
   (recommended; the pins are what matter, not the commit id), or have Session 4 read them from the main checkout by
   absolute path and record that in its handoff.
2. **Session 6 timing.** One approved change is a thin payload for a four-million-token replica. The overall context
   says to batch; Session 6's unmodified baseline arm is also the Catalog 3.2 acceptance data. Sessions 4 and 5 are
   cheap and can proceed now; hold Session 6 until more changes accumulate or the acceptance data is wanted.
3. **The live-check harness is not in the repository.** If Session 5 or 6 wants the single-condition instrument again,
   copy `terra_live_check.py` and the dry-run CLI out of the scratchpad before it is lost; the results file records the
   method (payload shape, prompt hash, model).
4. **Harness versus production.** The control arm returned `supported` on two cards production had recorded
   `unsupported`. Any future live check on those cards should consider a batched replay reproducing the stored payload
   shape before crediting a wording with anything.
5. **Stale Pass 2d worklist row.** `window_unit_vs_treatment` names `dated_window_treatment_valance`, which the approved
   split removes; re-point it after Session 4 (register, section 1).
6. **Rulings still owed** (register, section 6): CCF-14 evidence precedence (decides whether CAP-014 is lead-only);
   swirl texture; stone near grade; and the ruling-3 sweep itself, which now carries CAP-008 and its live-check inputs.
7. **CCF-8.** The split changes both new atomic claims, so every stored Terra checkpoint is invalid: Sessions 5 and 6
   must replay, never reuse, and plan an explicit cutover (no runtime alias; stale ids fail hard; backfill is a no-op at
   the target version).

## Required next task

Objective: Session 4 per `docs/catalog_audit_program/05_SESSION_APPROVED_IMPLEMENTATION.md` — implement CAP-007's
approved diff, regenerate, pin the invariant with focused tests, hand off for Session 5.

Required inputs: `00_OVERALL_CONTEXT.md`, the Session 4 brief, this handoff, `reports/catalog_audit_approvals.json`
(hash above), `reports/catalog_audit_redraft.json` for the successor detail, and the pinned decisions file and catalog.

Authorized changes: `tools/catalog_migrations/kind_v2_decisions.json` (the `dated_window_treatment_valance` entry only,
via the seven ops), the regenerated `tools/issue_catalog_kind_v2.json`, `2.1_to_3.0.json` and `2.1_to_3.0_audit.md`,
focused tests, `HANDOFF_SESSION_4.md`.

Prohibited changes: any other decisions entry; any proposal, review, gate or redraft artifact; prompts; runtime code;
publication; any "opportunistic" tidy-up, including the tighter claim texts recorded as optional.

Exit criteria: generated diff equals the manifest's `expected_generated_changes` (added
`dated_window_valance_or_curtains` and `window_blinds_basic_or_plain`, removed `dated_window_treatment_valance`, no
other item changed, order preserved); double generation byte-identical; validators zero errors; the generated blinds
successor carries `route_override: no_action`, `defaultHidden` false and the fabric successor's `deny_any` ends with
`shower`, `tub`; `HANDOFF_SESSION_4.md` pins the manifest, redraft, renewed-review, live-check and decision-record hashes.

## Do not redo

- Do not re-run the live check, the drafters or the reviewers; their outputs are in the scratchpad and the artifacts
  cite the underlying evidence.
- Do not reopen Q-1 to Q-9 or A-1 to A-7; they are recorded rulings. In particular, gold g3 exists (the drafter that
  denied it searched the wrong half of the bundle), so Q-1's stakes were accurate.
- Do not edit any Session 1–3 or gate artifact; corrections go in the next handoff.
- Do not treat the single-condition harness as reproducing batched production behaviour.
- Do not implement anything but CAP-007's seven ops; CAP-008 is deferred and its wordings were measured to fail.

## Suggested opening prompt for the next task

> Execute Session 4 of the catalog-audit program, the approved surgical implementation, per
> `docs/catalog_audit_program/05_SESSION_APPROVED_IMPLEMENTATION.md`. Read `00_OVERALL_CONTEXT.md`, the brief,
> `HANDOFF_SESSION_GATE.md`, and the hash-verified `reports/catalog_audit_approvals.json`; take the successor detail from
> `reports/catalog_audit_redraft.json` `cap007`. First verify HEAD `9afe0fa`, every pin in the manifest's
> `approved_against`, and that the decisions file still hashes `47614d82…03e8`; resolve the untracked-artifacts question
> in this handoff's risk 1 before cutting a worktree. Then write an implementation plan naming the single decisions
> entry, the seven ops, the generator command, the tests and the rollback boundary, and implement exactly CAP-007's
> `approved_diff` and nothing else. Regenerate twice, prove the generated diff equals the manifest's
> `expected_generated_changes`, add focused tests, and finish with `HANDOFF_SESSION_4.md` pinning every gate artifact.
> Do not touch CAP-008 or any other entry, and do not tidy the claim texts.
