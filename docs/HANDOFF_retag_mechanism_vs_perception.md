# Re-tag — is the measured error a *language* problem or a *perception* problem?

Written 2026-08-26 · backend `renovation_architecture_rework` · tooling:
`scripts/build_retag_queue.py`, `scripts/review_server.py` (reused unchanged),
`scripts/retag_tally.py` · reads the frozen review evidence read-only, writes
only new `reports/retag_*` files. Zero provider calls.

## 1. The question

The manual review measured *that* ~10–14% of billed conditions are wrong. It
did not separate the two ways a claim can be wrong:

- **Language failure** — a real problem is in the photo, but the claim names it
  wrong ("I see stains not scuffs"; "paint isn't peeling, there are dents, but
  the damage is substantiated"; "it is a drop ceiling not popcorn — but it does
  need replacing"). The system **saw** correctly and **named** badly.
- **Perception failure** — nothing of the kind is there at all. No wording
  change fixes this.

The split decides the next move, and it is currently unknown:

| if the mass is mostly… | then | and the granularity question |
|---|---|---|
| **language** | fix by rubric (QP2) and/or by coarsening the degradation claim text so the claim stops asserting a distinction MLS photos cannot settle | **dropping the fine granularity is the right call** — the coarse claim is true, the fine one is what fails |
| **perception** | wording work will not move it; the levers are corroboration and the escalation/inspection lane | granularity is not the problem; coarsening would not help either |
| **too trivial** | a severity/threshold problem, not truth | unaffected |

Group B answers the mirror question: of the real conditions v5 *dropped*, how
many were rejected because of the claim's specific wording? Every one of those
is work that coarser claims would recover — so a high `wording_blocked` share
means coarsening pays twice (fewer false bills *and* fewer missed conditions).

## 2. The 46 cards

Derived from the frozen inputs by `scripts/build_retag_queue.py`, mirroring
`scripts/review_analysis.py:arm_records` (the builder asserts the counts and
prints the source hashes; it refuses nothing but flags any drift):

- **Group A — 26 cards** (17 canary + 9 production, 13 listings): the weighted
  billed-error records — dirA-accepted or uniform cards whose verdict was
  `unsupported` or `overstated`. These *are* the 13.6% / 10.4%.
- **Group B — 20 cards** (16 canary + 4 production, 15 listings): dirB
  recoveries — Terra-rejected claims the human judged supported.

**Deliberately not blind.** Each card shows your own earlier verdict and note,
because the question being asked is a different one and the note often already
contains the answer. That is what makes this ~30–45 minutes rather than a
second review.

## 3. Run it

```bash
.venv\Scripts\python.exe scripts\build_retag_queue.py --check
```

```bash
.venv\Scripts\python.exe scripts\review_server.py --queue reports\retag_queue.json --verdicts reports\retag_verdicts.jsonl
```

Then `http://localhost:8765` — same UI and keys as the review program (`1`–`4`
answer, `n` notes, `Enter` save, `j`/`k` move, `z` undo, `r` reveal). The `a`–`e`
error tags and the bathroom digits are hidden (cards are `kind: "retag"`).
Phase 1 is group A, phase 2 is group B; the header tracks progress per phase.
Resumable — verdicts append to `reports/retag_verdicts.jsonl`, latest wins.

```bash
.venv\Scripts\python.exe scripts\retag_tally.py
```

→ `reports/retag_tally.md`.

### The answer vocabularies

**Group A** — *was a real problem there, just named wrong?*
`1 mechanism_only` (real problem, claim named it wrong) · `2 wholly_false`
(nothing of the kind is there) · `3 too_trivial` (there, but not worth billing)
· `4 cannot_tell`.

**Group B** — *why did Terra reject something you could see?*
`1 wording_blocked` (the claim's specific wording is what failed) ·
`2 terra_miss` (fair as worded, Terra missed it) · `3 borderline` (visible, but
the rejection is defensible) · `4 cannot_tell`.

## 4. How to read the result — decided in advance

Group A is reported in **weighted error mass**, not card counts: a uniform-arm
card stands for 17.775 (canary) or 13.143 (production) population conditions, a
dirA card for 1. The tally reproduces `reports/review_analysis.md` §9 exactly
(canary 100.875, production 21.143), so the shares are in the same units as the
headline rates. Card counts are printed alongside.

Pre-committed reading (fixed before the answers exist, so the conclusion is not
fitted to them):

- **`mechanism_only` ≥ ~⅔ of error mass** → language dominates. Ship the QP2
  rubric *and* coarsen the degradation claim family; expect most of the
  measured error to move. Fine mechanism granularity is retired for that family.
- **~⅓–⅔** → mixed. Run all three Session F arms (rubric / coarsened / both)
  and let the harness pick; expect partial improvement.
- **≤ ~⅓** → perception limit. Wording work is not the lever; go to
  corroboration and the escalation/inspection lane, and treat the affected
  items as candidates for retirement rather than re-wording.
- **`too_trivial` materially present** → open a separate severity/threshold
  question; it is not part of the wording decision.

**Honest limit, stated up front:** canary error mass is carried by 5 uniform
cards (88.9 of 100.9) and production by 1 (13.1 of 21.1) — a single uniform
card's answer swings the canary share by ~17.6 points. The mass view is
therefore *directional*, not precise. The tally prints card counts next to it;
**if the two views disagree, treat the result as inconclusive and run all three
Session F arms** rather than picking a winner from this exercise alone.

## 5. What this does not show

- Not a re-measurement of the error rate — the rates are frozen and unchanged;
  this only re-classifies known errors by cause.
- Not blind, so it inherits whatever bias the original notes carry. It is a
  classification of your own prior judgments, not an independent re-review.
- Group B is a targeted stratum (Terra-rejected *and* 2f-confirmed); its counts
  are not a recall estimate, and §12's bar on deriving Terra recall stands.
- Nothing here measures whether a coarser claim would actually pass Terra —
  that is what the Session F harness arms test with live calls.

## 6. Where the result goes

Into the **Session F** variant design
(`docs/ROADMAP_quality_program_sessions_20260826.md`): it sets which arms are
worth their ~1 Terra day each — rubric-only, coarsened-claims, or both — and,
if language dominates, it is the evidence for retiring the fine mechanism
distinctions from the catalog claim text (a catalog edit that rides QP2's
single canary at zero marginal quota). Record the outcome in
`docs/PROPOSALS_output_quality_improvements_20260826.md` §3 as the basis for
the S1 shape.

Tooling status: untracked, like the rest of the review tooling — commit it with
the review tooling in Session B.
