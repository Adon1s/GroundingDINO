# Handoff — analyse the mechanism-vs-perception re-tag

Written 2026-08-28 · backend `renovation_architecture_rework` · re-tag
completed by Steven 2026-08-28 (46/46) · **the result has not been computed by
anyone yet** — `reports/retag_tally.md` does not exist and no session has seen
the answer distribution.

**Purpose.** A fresh session is to analyse the completed re-tag and report what
it says. The re-tag exists to split the measured billed-error mass into a
*language* failure mode (a real problem was there; the claim named it wrong)
and a *perception* failure mode (nothing of the kind was there), because that
split decides the next unit of work on output quality.

**This document is context only.** It contains the question, the inputs, the
mechanics, the pre-committed reading rule, and the known limits. It contains
**no results, no distribution, no interpretation** — this session computes and
interprets them. Deliverable format is this session's choice.

---

## 1. The question, and why the answer matters

The manual review measured *that* ~10–14% of billed conditions are wrong. It
never separated *how* they are wrong. The reviewer's own notes suggested both
modes exist — "I see stains not scuffs"; "paint isn't peeling, there are dents,
but the damage is substantiated"; "it is a drop ceiling not popcorn — but it
does need replacing" — but nothing quantified the split.

The consequences are opposite:

| if the error mass is mostly… | the lever is | and the granularity question |
|---|---|---|
| **language** | Terra rubric v2 (QP2) and/or coarsening the degradation claim text so claims stop asserting distinctions MLS photos cannot settle | **retiring the fine mechanism granularity is the right call** — the coarse claim is true, the fine one is what fails |
| **perception** | corroboration and the escalation/inspection lane; wording work will not move it | granularity is not the problem; coarsening would not help either |
| **too trivial** | a severity/threshold question | unaffected |

Group B asks the mirror question about conditions v5 *dropped*: a high
`wording_blocked` share means coarsening pays twice — fewer false bills **and**
fewer missed real conditions.

## 2. The pre-committed reading rule — fixed before any answer existed

Recorded in `docs/HANDOFF_retag_mechanism_vs_perception.md` §4 on 2026-08-26,
deliberately in advance so the conclusion could not be fitted to the data.
**Honor it, or state explicitly and with reasons that you are departing from
it** — do not quietly substitute a different threshold after seeing the
numbers.

- **`mechanism_only` ≥ ~⅔ of weighted error mass** → language dominates. Ship
  the QP2 rubric *and* coarsen the degradation claim family; the fine
  mechanism granularity is retired for that family.
- **~⅓–⅔** → mixed. Run all three Session F arms (rubric / coarsened / both)
  and let the harness pick.
- **≤ ~⅓** → perception limit. Wording is not the lever; go to corroboration
  and the escalation/inspection lane, and treat the affected items as
  candidates for retirement rather than re-wording.
- **`too_trivial` materially present** → a separate severity/threshold
  question, not part of the wording decision.
- **Tie-breaker, also pre-committed:** the tally prints card counts beside the
  weighted mass. **If the two views disagree, the result is inconclusive and
  all three Session F arms run** — do not pick a winner from this exercise
  alone.

## 3. What was actually asked, per card

46 cards, two groups, one question each. Cards were **not** shown blind: each
displayed the reviewer's own earlier verdict and note, because the question is
a different one and the prior note often already contains the answer.

**Group A — 26 cards** (17 canary + 9 production, 13 listings). These *are* the
weighted billed-error records behind the 13.6% / 10.4% rates: dirA-accepted or
uniform-arm cards whose original human verdict was `unsupported` or
`overstated`. Question: *was a real problem there, just named wrong?*

| key | recorded verdict string (prefix before `:` is the slug) |
|---|---|
| 1 | `mechanism_only: a real problem is there, the claim named it wrong` |
| 2 | `wholly_false: nothing of this kind is there at all` |
| 3 | `too_trivial: something is there but not worth billing` |
| 4 | `cannot_tell: evidence too poor to judge` |

**Group B — 20 cards** (16 canary + 4 production, 15 listings). The dirB
recoveries: Terra-rejected claims the human judged supported. Question: *why
did Terra reject something you could see?*

| key | recorded verdict string |
|---|---|
| 1 | `wording_blocked: the claim's specific wording is what failed` |
| 2 | `terra_miss: claim was fair as worded, Terra missed it` |
| 3 | `borderline: visible but a close call, rejection defensible` |
| 4 | `cannot_tell: evidence too poor to judge` |

Selection was derived programmatically from the frozen evidence by
`scripts/build_retag_queue.py`, mirroring `scripts/review_analysis.py`'s
`arm_records`; the builder asserts the 26/20 counts and prints the source
hashes.

## 4. Inputs and their integrity state (verified 2026-08-28)

| file | state |
|---|---|
| `reports/retag_verdicts.jsonl` | 46 raw lines → **46 net verdicts**, sha256 `a038029811c9`; **zero** overwrites, undos, peeks or orphans; 13 cards carry free-text notes; recorded 10:54–11:13 on 2026-08-28 |
| `reports/retag_queue.json` | the 46 cards, generated 2026-08-26; carries `groups` metadata with each group's question and key map |
| `reports/review_queue.json` | **frozen**, sha256 `8512b87c2af1` — supplies the population denominators used for weighting |
| `reports/review_verdicts.jsonl` | **frozen**, sha256 `0c7ca6a8b55d` — the original 223 verdicts |
| `reports/retag_tally.md` | **does not exist yet** — produced by running the tally |

Do not modify or rebuild any of the four existing files. A rebuild of the
frozen review queue would absorb listings analysed after 2026-08-26.

## 5. Mechanics you must get right

**Report group A in weighted error mass, not card counts.** A uniform-arm card
stands for `accepted_non_dirA / n_uniform` population conditions — **17.775**
(canary) and **13.143** (production) — while a dirA card is a census record
weighting 1. Counting cards would let two uniform cards look like two errors
when they stand for ~36.

`scripts/retag_tally.py` already implements this and writes
`reports/retag_tally.md`:

```bash
.venv/Scripts/python.exe scripts/retag_tally.py
```

**Validity check with a known answer:** the group-A total masses must
reproduce `reports/review_analysis.md` §9 exactly — **canary 100.875,
production 21.143**. The tally was verified to do so on an empty verdict set;
if your numbers drift from those totals, the weighting is wrong, not the data.
Group B is a targeted stratum — raw counts only, never weighted.

**Available cut dimensions** (each retag card carries the original card's
`meta`, plus `retag_group`, `prior_verdict`, `prior_note`): `catalog_item_id`,
`catalog_kind`, `scene_group`, `estimate_unit_id`, `photo_count`,
`terra_batch_conditions`, `terra_batch_images`, `second_opinion` (the v4 Pass 2f
status), `direction`, plus card-level `source`, `property_key`, `strata`.
A cross-tab of retag answer against `prior_verdict` separates the 20
`unsupported` from the 6 `overstated` group-A cards, which are different
things. Which cuts are worth making is this session's judgment; small-n
discipline from the source analysis applies.

The 13 free-text notes are qualitative evidence. The source analysis coded its
notes into auditable themes with the raw text echoed beside each
(`reports/review_analysis.md` §11) — that is the precedent for handling them,
not a requirement.

## 6. Known limits — state them, do not discover them late

- **The mass view is directional, not precise.** Canary error mass is carried
  by 5 uniform cards (88.9 of 100.9) and production by 1 (13.1 of 21.1). One
  uniform card's answer swings the canary share by ~17.6 points. This is why
  the card-count tie-breaker in §2 exists.
- **Not blind.** The exercise re-classifies the reviewer's own prior judgments
  and inherits whatever bias those notes carry. It is not an independent
  re-review.
- **Not a re-measurement.** The error rates are frozen and unchanged; this
  only re-classifies known errors by cause.
- **Group B is targeted** (Terra-rejected *and* 2f-confirmed). Its counts are
  not a recall estimate; the source analysis's §12 bar on deriving Terra
  recall stands, along with its other bounded inferences (no overall Sol or
  package-pipeline accuracy, no Pass-2f correctness from condition verdicts,
  batch-size associations are confounded, no pricing claims).
- **Nothing here shows whether a coarser claim would actually pass Terra.**
  That is what the Session F harness arms test with live calls.

## 7. Where the answer goes

- **Session F variant design** (`docs/ROADMAP_quality_program_sessions_20260826.md`)
  — it decides which arms are worth ~1 Terra day each: rubric-only,
  coarsened-claims, remove-observations, structured-fields, or combinations.
  Session F is currently blocked on this result.
- **The granularity call.** If language dominates, this is the evidence for
  retiring fine mechanism distinctions from the catalog claim text — a catalog
  edit that rides QP2's single canary at zero marginal quota. Catalog changes
  go through `tools/catalog_migrations/kind_v2_decisions.json` + regeneration,
  never hand-edits.
- **`docs/PROPOSALS_output_quality_improvements_20260826.md` §3** — record the
  outcome as the basis for the S1 shape (S1 is already adopted; this refines
  *which* prompt/claim shape ships).

## 8. Supporting documents — what each is for

| document | why you need it |
|---|---|
| `docs/HANDOFF_retag_mechanism_vs_perception.md` | **Start here.** The exercise's design, the vocabularies, the pre-committed reading rule, the stated limits |
| `reports/review_analysis.md` §§3, 4, 9, 11, 12 | the frozen source analysis: the rates being decomposed, the subgroup structure, the §9 gate-exposure masses your totals must reconcile with, the coded notes precedent, and the bounded inferences |
| `reports/review_analysis.json` | machine-readable mirror; carries per-item rows the md filters out at judged<3 |
| `docs/DESIGN_review_method.md` | semantic authority for what the *original* verdicts mean (supported / unsupported / overstated / inconclusive) |
| `scripts/build_retag_queue.py` | how the 46 cards were selected; the authority on group membership |
| `scripts/retag_tally.py` | the weighting implementation and its docstring rationale |
| `scripts/review_analysis.py` | reference implementation of the weighting identity (`weighted_estimate`, `arm_records`, `gate_exposure`) |
| `docs/PROPOSALS_output_quality_improvements_20260826.md` QP1/QP2 | what the answer feeds; the harness that will test it |
| `docs/ROADMAP_quality_program_sessions_20260826.md` | Session F's charter and its dependency on this |

Deliberately out of scope for this session: the v4→v5 migration state, the
frontend contract work, the package-policy backlog, and price calibration.
None bear on this question.

## 9. Working rules

- `RV_ROOT` is the live production working tree — no `git checkout` of other
  branches; use a worktree if needed.
- Frozen inputs are read-only: `reports/review_queue.json`,
  `reports/review_verdicts.jsonl`, `reports/review_analysis.*`, all
  `reports/session9_*`. Write new outputs to new paths.
- Zero provider calls are needed or wanted for this work.
- Tests: `.venv\Scripts\python.exe -m pytest` (bare `python` does not resolve
  the venv). The re-tag tooling has no test suite of its own; the review
  tooling suites are `tests/test_review_cards.py` and
  `tests/test_review_analysis.py` (33 tests).
- The re-tag tooling (`scripts/build_retag_queue.py`, `scripts/retag_tally.py`,
  `reports/retag_queue.json`) may be untracked — check `git status` before
  assuming a commit is needed.

## 10. Suggested reading order (procedural, not a steer)

1. `docs/HANDOFF_retag_mechanism_vs_perception.md` end to end.
2. `reports/review_analysis.md` §§3–4 (what the rates are), §9 (the masses),
   §11 (the notes), §12 (what may not be inferred).
3. `scripts/build_retag_queue.py` then `scripts/retag_tally.py` — selection
   and weighting.
4. Run the tally; verify the masses reconcile with §9 before reading anything
   into the shares.
5. The 13 notes in `reports/retag_verdicts.jsonl`, against each card's
   `prior_note` in `reports/retag_queue.json`.
6. Only then form a view, and check it against §2's pre-committed rule.
