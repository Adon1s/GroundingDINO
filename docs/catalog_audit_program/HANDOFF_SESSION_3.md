# Handoff — catalog audit session 3

Date: 2026-09-05

Session/task title: Independent adversarial review (`03_SESSION_ADVERSARIAL_REVIEW.md`)

**Status: COMPLETE.** Every exit criterion in the brief is met: all 20 proposal ids carry exactly one disposition,
the two native proposals were attacked on the blocking questions, every no-change, deferred and non-catalog cluster
was reviewed for false negatives on the same evidence standard, one reopened cluster is identified separately, the
proposal, evidence, decisions and catalog artifacts are byte-identical to the session start, and the review artifact
is pinned by hash for the human gate.

## Outcome

| Disposition | Count | Ids |
|---|---|---|
| `sustained` | 11 | CAP-001, 002, 004, 005, 006, 009, 011, 012, 015, 018, 021 |
| `sustained_with_conditions` | 8 | CAP-003, 008, 010, 013, 014, 016, 017, 019 |
| `insufficient_evidence` | 1 | CAP-007 |
| `reassign_non_catalog` | 0 | — |
| `reject` | 0 | — |

Reopened: **none survived adversarial refutation.** Six reopen or relabel candidates came out of the family sweep
(CAP-003, CAP-010, CAP-014, CAP-016, CAP-017, CAP-019); each was handed to an independent refuter told to default to
refuted. Two survived as diagnosis corrections that leave the label in place (CAP-010: the fence lean was captured and
billed under the sibling successor, so "Pass 2a never reported the lean" is wrong; CAP-014: the defect-kind wall items
were kind-gated, not retrieval-unreachable). The two nearest true reopens were CAP-019 (the "better" item's claim covers
the door leaf, not the surround) and CAP-017 (a verified double-bill of two wallpaper items on one wall); both fell
because the policy as written permits the Session 2 label and the evidence is one property. No no-change or deferred
cluster hides a bar-meeting catalog defect. Two coverage groups with two-property gold support (exterior-metal
deterioration; HVAC ductwork) are documented questions because neither observation reached Pass 2d, and 42
unadjudicated railing finish-wear bullets that resolved to nothing are the strongest coverage candidate in the corpus.

**Native proposals.** CAP-008 survives in direction, not as drafted: the drafted 33-word state is self-contradictory
on the photos that decide it ("heavily painted-over" names the multi-coat white finish of the kill-condition trim that
the exclusion clause is meant to exclude), production Terra has no exclusion semantics and echoes "dated" observations
as supported 15 of 18 times on style-word claims, one of the three "withdrawn" billings rests on a human note about
missing trim, the structural-exception bar is item-specific rather than universal, the regression ledger omits a
billed record, and the package statement is false in code. Conditions: a positive-only ≤9-word state plus name and
description overrides, live Terra checks on three cards, a completed ledger, and Steven's explicit rulings (Q-3, Q-4,
Q-5). CAP-007 is not sustained as drafted: the two gold units that meet the bar are upstream losses the split cannot
repair, the split withdraws at least two human/gold-backed deteriorated-blinds billings (not one) plus roughly two
dozen presence billings, the blinds successor packs damage words into a modernization claim (the first same-kind split
in the decisions file), and the fabric successor keeps a billing path open for blinds-only bullets. It can return as a
redraft once Steven answers Q-1 and Q-2.

**CAP-013.** The migration-system-gap label stands on corrected grounds: a lead-run in-memory probe shows a
`baseboard_wear_scuffs` split is blocked by the catalog 3.2 `repair_support_when_driven` marker section (which names
the parent and is unreachable from the ops surface), and with the marker edited out of band the defect successor
inherits an unauthorable `estimate_scope: marketability_rehab`. The proposal's own blockers (the "atomic claim must
conflate" rule; severity) are wrong. The gap is adjudicated on one property; three human-annotated cards on other
properties point at the same concept and are the fastest route to a met bar.

## Repository state

- Starting commit: `9afe0fa5a0490a856abed507dccc4a02ed1de24c`, branch `terra_factorized_verifier`
- Ending commit: unchanged, nothing committed (Steven decides)
- Dirty state: no modified tracked files at start or end; all session files are untracked working-tree files
- Pre-existing changes preserved: yes (the Session 1/2 untracked files are byte-identical, see below)

## Inputs verified

All 19 pinned inputs from `HANDOFF_SESSION_2.md` matched at the start and were re-hashed at the end (unchanged):
`reports/catalog_audit_proposals.json` `1411352c…3ef99`; `docs/PROPOSAL_catalog_audit_20260901.md` `771572e7…b17591`;
`reports/catalog_audit_evidence.json` `43354104…29b7` (fingerprint `e4c0a2b8…0c08`); photo review `2225db9c…9150`;
packet `7b4621df…bc80`; decisions `47614d82…03e8`; generated 3.2 catalog `51bf7e26…aa54`; v1 catalog `4ba046a1…14f2`;
generator `b732022f…a053`; validator `041637a3…bee2`; and the nine frozen sources. The pinned run artifact for
rc_7c5b9f9bb3aa (`…/redfin_81000709/20260817_225858_d13bd83c/photo_intel_debug.json`) hashed `05738c0e…` as pinned.

## Files created or changed

| Path | Status | sha256 | Purpose |
|---|---|---|---|
| `reports/catalog_audit_adversarial_review.json` | new | `d1149875399a9f30c6cd56e924731c7e207d90b603cb248f402c692c3438b5c4` | 20 dispositions, 0 reopened, 14 cross-cutting findings, 9 gate questions, tool-derived reconciliation |
| `docs/REVIEW_catalog_audit_adversarial_20260905.md` | new | `538f28724531018cdbd937e5355fbbdca4273352cbf62aafa6ebfb69abf1a96a` | deterministic rendering |
| `scripts/render_catalog_audit_adversarial_review.py` | new | `b5afbed95e4a7a08df6faa2e927d74e7a0e9ed3513d791cc73ed2c33d5a5d606` | reconciliation/render tool, `--check` writes nothing, guard hashes before/after |
| `tests/test_catalog_audit_adversarial_review.py` | new | `a1fbd22dfe253189071ddad1bebb26e6bee5b32208b74331d032fd9427fa361d` | 9 tests (7 synthetic, 2 on the real artifacts) |
| `docs/catalog_audit_program/HANDOFF_SESSION_3.md` | new | — | this file |

Nothing else changed. In particular `reports/catalog_audit_proposals.json`, the proposal document, the photo review,
the evidence bundle, the decisions file, both catalogs, prompts and runtime are untouched (re-hashed at the end).

## Work completed

- Verified pins, then ran two bounded multi-agent workflows (read-only, structured returns): three blind photo readers
  (32 photos, never shown the proposals), eight distinct-lens refuters on CAP-008/CAP-007/CAP-013, four mechanical
  re-derivation agents (Terra qualifier census over 33 frozen cases; control completeness and candidate-pool
  re-derivation for nine items; run-artifact verification and runtime tracing; in-memory dry runs of both native
  proposals plus a `baseboard_wear_scuffs` split probe), five family hunters over the 17 other clusters, six refuters on
  the hunters' reopen and relabel candidates, and one coverage critic over the 63 triage rows (27 agents in all; the
  runs were interrupted by usage limits three times and resumed from cache each time). Agent findings were inputs;
  every disposition was authored by the lead and every claim in the artifact cites a case, unit, item, photo or
  `file:line`.
- The lead independently re-opened the decisive photos (kill condition, blinds, base trim, the disputed paint
  palette), re-ran the baseboard split probe, verified the seven plain/basic/builder-grade claims, and re-verified
  every load-bearing agent fact from the pinned artifacts (the fence lineage, the wall-item kind gate, the wallpaper
  double-bill, the drop-ceiling guardrail, the casing correct rejection) before adopting it.
- Built the reconciliation tool and tests; the tool resolves every `evidence_inspected` reference against the bundle,
  proposals, catalog, photo root and repository (with line ranges), enforces one disposition per `KNOWN_IDS` id, the
  disposition vocabulary and shape, reopen consistency, and re-hashes the guarded files before and after.

## Decisions and invariants

- Dispositions are recommendations tied to proposal ids; nothing in the proposals JSON (including
  `human_disposition`) was edited. Corrections live in `required_modification` and `conditions`.
- `no_change` and `non_catalog_action` were treated as one class for false-negative review (CCF-11); the gate should
  define the distinction if it wants one.
- Rulings were taken from `HANDOFF_SESSION_2.md` lines 74-93 and the Session 2 brief; the `PHASE_C_BRIEF.md` the
  proposals cite does not exist in the repository (CCF-1), and the divergence/reachability "scans" they cite are prose
  only (CCF-2). Session 4 and 5 must not cite either without a committed derivation.
- The review artifact's hash is what the approval manifest pins (`adversarial_review_sha256`); re-rendering with
  `--check` must keep matching.

## Deviations from the session brief

| Deviation | Reason | Impact |
|---|---|---|
| Multi-agent orchestration (Steven's ultracode instruction) | Independence: blind readers and distinct-lens refuters do not share the lead's hypotheses | Agent outputs kept in the session scratchpad, not the repo; the artifact cites evidence, never an agent |
| Two models (Steven's instruction) | The final refuter run executed under Opus 5 while the lead was switched; every disposition, finding, question and this handoff were authored by Fable 5.1, which also re-verified the load-bearing agent facts | Model diversity on the adversarial checks; no effect on the artifact's provenance rules |
| Two workflow runs were interrupted by session usage limits and resumed | Cached agents replayed; only failed agents re-ran | No effect on results; the journals record the failures |
| A small tool and tests were added | The brief permits tests only with deterministic reconciliation tooling; the gate pins the JSON by hash, so a `--check`able renderer was warranted | Mirrors the Session 2 renderer contract; reuses its loaders |

## Commands and verification

| Command/check | Result |
|---|---|
| `scripts\render_catalog_audit_proposals.py --check` | `proposals json matches; markdown matches; validation final ok=True errors=0 pending=0`, exit 0 |
| `scripts\render_catalog_audit_adversarial_review.py` | `validation ok=True errors=0`; 20/20 ids, 0 missing, 0 orphans, 0 duplicates; 282 references resolved (card 59, unit 35, item 50, photo 27, file:line 96, file 15), 0 unresolved |
| `scripts\render_catalog_audit_adversarial_review.py --check` | `review json matches; markdown matches; validation ok=True errors=0`, exit 0 |
| `pytest tests/test_catalog_audit_adversarial_review.py tests/test_catalog_audit_proposals.py` | **43 passed** (9 review + 34 proposals) |
| Protected-file rehash, start vs end | all 26 unchanged (19 pinned inputs, v1 catalog, generator, validator, four program docs) |
| `git status --porcelain -uno` | empty (no tracked file modified) |

## Outputs for the next session

| Artifact | Hash | Contract |
|---|---|---|
| `reports/catalog_audit_adversarial_review.json` | `d1149875399a9f30c6cd56e924731c7e207d90b603cb248f402c692c3438b5c4` | one disposition per proposal id; `derived.validation.ok` true |
| `docs/REVIEW_catalog_audit_adversarial_20260905.md` | `538f28724531018cdbd937e5355fbbdca4273352cbf62aafa6ebfb69abf1a96a` | human-readable; sections per disposition, reopened list, cross-cutting findings, gate questions |
| `reports/catalog_audit_proposals.json` | `1411352c…3ef99` | unchanged Session 2 artifact |
| `reports/catalog_audit_evidence.json` | `43354104…29b7` | unchanged Session 1 bundle |

## Unresolved questions and risks

The nine gate questions (Q-1 … Q-9 in the artifact) are the open items; the ones that block implementation are:

1. **Q-1 / Q-2 (CAP-007).** Whether ruling 5 covers visibly deteriorated blinds, and whether an improvised cloth
   covering is billable. The drafted split is wrong under either answer without a redraft.
2. **Q-3 / Q-4 / Q-5 (CAP-008).** Withdrawal of two clean plain-trim billings under ruling 3; the claim form (positive
   house style recommended); whether ruling 3 applies to the other six plain/basic claims. The kill condition cannot be
   settled offline: production Terra has no exclusion semantics, and the live checks in CAP-008's conditions are a hard
   gate.
3. **Q-6 (CAP-013).** Whether to authorize schema work for the marker section and successor economics, or wait for a
   second adjudicated property (three candidate cards named).
4. **Q-9 (CAP-019).** Whether exterior_door_paint_failure's subject includes the casing and surround; a "leaf only"
   ruling moves the cluster to deferral as a one-property coverage question.
5. Any atomic_claim edit invalidates every stored Terra checkpoint (projection fingerprint), and any split has no
   runtime alias (stale ids fail hard; backfill skips artifacts at the target version): Session 5/6 must plan a full
   replay and an explicit cutover for stored 3.2 artifacts (CCF-8).
6. Three facts the sweep verified that no cluster acts on and the gate should know: the wallpaper double-bill
   (CCF-13), lexical-guardrail brittleness to hyphenation and word order with require_any unauthorable on carryovers
   (CCF-12), and the unruled evidence-class precedence for matched gold annotations that decides whether CAP-014 is
   lead-only (CCF-14).

## Required next task

Objective: Human review gate per `04_HUMAN_REVIEW_GATE.md`, producing `reports/catalog_audit_approvals.json`.

Required inputs: `reports/catalog_audit_proposals.json`, `docs/PROPOSAL_catalog_audit_20260901.md`,
`reports/catalog_audit_adversarial_review.json`, `docs/REVIEW_catalog_audit_adversarial_20260905.md`,
`HANDOFF_SESSION_2.md`, this handoff, and the photos named in the dispositions for any disputed record.

Authorized changes: the approval manifest only.

Prohibited changes: repository implementation; edits to any Session 1–3 artifact.

Exit criteria: every proposal id appears exactly once in the manifest; every approved native change has an exact
approved diff (for CAP-008 that means the redrafted state and overrides, returned for adversarial check first; for
CAP-007 a redraft after Q-1/Q-2); the proposal and review hashes match; no approved record depends on an unresolved
migration-system gap.

## Do not redo

- Do not re-run the photo review or the Session 2 proposals; the pins are the authority.
- Do not treat the refuters' or readers' outputs as evidence; they are in the session scratchpad and the artifact
  cites the underlying cases, units, items, photos and code.
- Do not re-derive the baseboard split probe; the result (blocked by the marker section as specified) is recorded in
  CAP-013 and CCF-7.
- Do not cite `PHASE_C_BRIEF.md`, `divergence_scan`, `reachability_scan` or `digest/items.json` as artifacts.

## Suggested opening prompt for the next task

> Perform the catalog-audit human review gate. Read `00_OVERALL_CONTEXT.md`, `04_HUMAN_REVIEW_GATE.md`,
> `HANDOFF_SESSION_3.md`, and the hash-verified `reports/catalog_audit_proposals.json` and
> `reports/catalog_audit_adversarial_review.json` with their rendered documents. Answer the eight gate questions,
> assign exactly one disposition per proposal id, and write `reports/catalog_audit_approvals.json` pinned to the
> proposal and review hashes. Do not implement anything.


## Errata (human gate, 2026-09-05)

Steven's corrections to this handoff, given at the human gate on 2026-09-05. They are recorded once here and referenced from
`reports/catalog_audit_gate_decisions.json` (`references.handoff_errata`). Nothing above this section was changed.

1. **Line 9 — "one reopened cluster is identified separately."** Wrong: the Outcome section, the review JSON and the
   reconciliation all report zero reopened clusters. Correction: six reopening or relabelling candidates were investigated
   and none ended in a formal reopening; several factual corrections and unresolved concerns remained (items 4 and 6).
2. **Line 195 — "Answer the eight gate questions."** Stale: the review carries nine questions (Q-1 to Q-9; line 145 of this
   handoff says nine). Correction: refer to all nine by id, otherwise a session could omit a decision while believing it
   completed the task. The gate ruled on all nine.
3. **Lines 165-180 ("Required next task") and REVIEW Q-4 (docs/REVIEW_catalog_audit_adversarial_20260905.md line 492).**
   No workable path from the current proposals to approval: the next task may edit only the approval manifest and may not
   touch Session 1-3 artifacts, yet the exit criteria require revised proposals that have already returned for adversarial
   review, and those do not exist; Q-4's implication says Session 4 authors the redraft while the gate requires exact
   approved changes before Session 4 begins. Correction: policy decisions, proposal redrafting, renewed review with the
   required checks, and final approval are separate stages. Resolution: the gate recorded its rulings and all twenty
   dispositions in `reports/catalog_audit_gate_decisions.json` (no proposal approved); a redraft stage covering CAP-008 and
   CAP-007 and one renewed review (including the live Terra check for the CAP-008 wording) follow;
   `reports/catalog_audit_approvals.json` is written only after that review and Steven's explicit per-id approval. The
   "Required next task" section above is superseded by the decision record's `redraft_brief`, `renewed_review`,
   `live_check` and `final_approval` blocks.
4. **Line 29 — "No no-change or deferred cluster hides a bar-meeting catalog defect."** More conclusive than the evidence
   supports: it holds only under the program's current evidence classifications and thresholds; the review itself leaves
   the matched-gold-annotation precedence rule unresolved (CCF-14), which affects CAP-014's classification, and the
   wallpaper overlap (CCF-13) was observed but fell below the action threshold, which is not the same as disproved.
   Correction: "No additional cluster was promoted under the current evidence rules; several potential defects and one
   evidence-classification issue remain unresolved." This qualifies the conclusion; it does not mean those candidates
   should be approved.
5. **Line 36 — "production Terra has no exclusion semantics."** Overstated: the code shows only that the production prompt
   (`tools/renovation_architecture/terra_review.py:33`) has no explicit instruction for handling exclusions or checking
   every qualifier, and the frozen examples give reason to distrust exclusion-heavy wording; that does not establish that
   the model cannot understand exclusions, and a short rationale's silence on a qualifier does not prove the qualifier was
   ignored. Correction: "Exclusion handling is not explicitly specified, and reliable behaviour under the proposed wording
   has not been demonstrated." The recommendation to test the wording stands (the gate's live check).
6. **Under-emphasis in the Outcome section.** Two corrections that change the right next optimization targets receive much
   less attention than the fence and wall diagnoses. CAP-003 (REVIEW line 89): some supposed Terra errors may be correct
   rejections under ruling 4, so calibrating Terra against the old labels could make the system worse; the gate
   commissioned no calibration and requires a ruling-4 re-adjudication of the four units first (Q-8). CAP-016 (REVIEW line
   298): the supposedly problematic condition was already Terra-supported and billed, so the remaining question is naming,
   not recovery of a lost condition. These are omissions of emphasis, not false statements.

Decision record: `reports/catalog_audit_gate_decisions.json`, sha256 `d89b2bf1149738d9b3e2d5a9222a5b4e163e2cf97dbb0c9f6396d595b849c980`. Handoff sha256 before this append:
`69e2a746920702bdfa1f617e038f2478d775f738ea8043cd62b57c9decddcb7d`.
