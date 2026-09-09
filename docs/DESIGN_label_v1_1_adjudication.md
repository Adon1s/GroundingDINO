# Design — v1.1 label adjudication (the label repair)

Written 2026-08-28 · backend `renovation_architecture_rework` · commissioned by
Steven after `docs/RESULT_retag_mechanism_vs_perception_20260828.md` ·
tooling: `scripts/build_adjudication_queue.py`, `scripts/review_server.py`
(unchanged), `scripts/build_labels_v1_1.py`.

**Pre-commitment.** Every mapping table in §§3–5 was written and committed
**before any adjudication verdict existed** — verified 2026-08-28: none of
`reports/adjudication_queue.json`, `reports/adjudication_verdicts.jsonl`,
`reports/labels_v1_1.json` was present when this document was written. This is
the same discipline the re-tag used (`HANDOFF_retag_mechanism_vs_perception.md`
§4) and it exists for the same reason: the label repair re-judges cards whose
answers the reviewer already has opinions about, so the rule that turns answers
into labels must not be selectable after the answers are visible. Departing
from a table below is allowed; doing it silently is not.

## 1. The problem this fixes

The v1 review asked one question — *"does the photo support the claim?"* — and
recorded one label. That single label is carrying three different judgments at
once:

| judgment | example from the frozen notes |
|---|---|
| is the underlying condition **there** | "The floor is soiled in multiple photos" (recorded `unsupported`) |
| is the **exact claim** accurate | "I see stains not scuffs" (recorded `overstated`) |
| is the work **worth doing** | "slightly dated but not worth replacing" (recorded `unsupported`) |

Three different failures are collapsed into one word, so every rate computed
from it is a blend, and Session F would tune against a blend. Worse, the
re-tag's free-text notes show the reviewer explicitly reversing his own call on
at least six cards ("Crap this one I messed up", "This is my miss, terra is
correct"). Those are label errors, not model errors, and the F scorer currently
treats them as ground truth.

**v1.1 records two axes instead of one**, so truth, wording, and renovation
judgment stop competing for the same field.

## 2. Scope, and what stays frozen

**v1 is immutable.** `reports/review_queue.json`,
`reports/review_verdicts.jsonl`, `reports/review_analysis.*`,
`reports/retag_*` and all `reports/session9_*` are read-only inputs. v1.1 is an
**overlay** — one new file, `reports/labels_v1_1.json`, keyed by the original
`card_id`. Nothing rewrites history; every downstream report prints original
and corrected side by side.

**In scope: condition cards only.** 173 cards carry v1 condition verdicts.

| arm | cards | why | phase |
|---|---|---|---|
| **blind canary** | 55 `supported_billed` | never re-examined by anything; a wrong `supported` here makes Session F score a *correct* flip as a loss | 1 |
| **re-ask** | 46 (the whole re-tag set) | the error classes; the re-tag asked a different question and its answers do not determine both axes | 2 |
| **blind production** | 24 `supported_billed` | completes the corrected population rates | 3 |
| repeats | ~12% of the blind arms | measures the reviewer's own consistency — the only noise floor in this program that has never been measured | 1, 3 |

Phase order is deliberate. The blind arm must run before the reviewer re-opens
cards he has strong opinions about. **After phases 1+2 the canary label set is
complete and Session F is unblocked**; phase 3 only refines the published
population rates. Stopping between phases always leaves a coherent artifact.

**Out of scope:** package cards (22), bathroom cards (17), the 32 unreviewed
production cards, the two thumbnail-excluded listings, prices, and the
`review_analysis.py` rate recomputation (a follow-up; see §7).

## 3. The adjudication schema (pre-committed)

One keypress per card, encoding two axes. Seven options, no free combination —
the reviewer never fills six fields, which is what keeps this at ~45 s/card.

| key | slug | claim axis | work axis |
|---|---|---|---|
| 1 | `exact_and_warranted` | `exact` | `warranted` |
| 2 | `misnamed_but_warranted` | `misnamed` | `warranted` |
| 3 | `exact_but_trivial` | `exact` | `trivial` |
| 4 | `misnamed_and_trivial` | `misnamed` | `trivial` |
| 5 | `wrong_object_or_place` | `absent` | `none` |
| 6 | `absent` | `absent` | `none` |
| 7 | `inconclusive` | `inconclusive` | `inconclusive` |

- **claim axis** — is the catalog claim, *as worded*, true of this photo?
  `exact` true as written · `misnamed` a real condition is there but this claim
  names it wrong (stains called scuffs) · `absent` nothing of the kind is there
  · `inconclusive` the evidence cannot settle it.
- **work axis** — would a renovator do work here? `warranted` yes ·
  `trivial` something is there, not worth billing · `none` nothing to do.
- Key 5 (`wrong_object_or_place`) is `absent` on both axes; it exists only to
  keep the "right problem, wrong room" diagnostic visible, since it implies a
  different upstream fix than a plain hallucination.

Projection back onto the v1 vocabulary, for side-by-side reporting only:

| v1.1 claim | v1 verdict |
|---|---|
| `exact` | `terra_claim_supported` |
| `misnamed` | `terra_claim_overstated` |
| `absent` | `terra_claim_unsupported` |
| `inconclusive` | `terra_evidence_inconclusive` |

The projection is lossy by construction — it is the compression this repair
exists to undo. Never feed it back into a v1.1 computation.

## 4. Scorer classes (pre-committed)

A card is **billed** when `meta.accepted` and it carries `dirA` or `uniform`.

Two outcome columns, because two of the rows depend on what the arm changed
(footnote ¹). "wording arm" = the claim text is untouched (control,
rubric-only); "coarsening arm" = the arm rewrites the claim text.

| billed card | class | flip away from `supported`: wording arm | coarsening arm |
|---|---|---|---|
| `absent` | `hard_false_billed` | **win** — the primary safety metric | win |
| `exact` + `warranted` | `supported_billed` | **loss** | loss |
| `misnamed` + `warranted` | `misnamed_billed` | **cost**¹ | neutral¹ |
| any + `trivial` | `trivial_billed` | **neutral**² | neutral |
| `inconclusive` | *excluded* | — | — |

| dirB card | class | flip to `supported`: wording arm | coarsening arm |
|---|---|---|---|
| `exact` + `warranted` | `dirB_recovery` | **win** | win |
| `misnamed` + `warranted` | `dirB_wording_recovery` | **neutral**¹ | **win**¹ |
| any + `trivial` | `dirB_trivial` | **neutral**² | neutral |
| `absent` | `dirB_terra_correct` | **loss**³ | loss |
| `inconclusive` | *excluded* | — | — |

¹ **Arm-dependent, and this is the subtle one.** `misnamed` means the exact
claim is wrong but a real condition is behind it. For an arm that leaves the
claim text alone (control, rubric-only), Terra *keeping* a literally-wrong
claim is not a success and dropping it is a cost — the right fix is wording.
For an arm that coarsens the claim text, Terra is being asked about a
*different, now-true* claim, so acceptance is the intended behaviour and the
cost does not apply. The scorer therefore takes coarsening arms on a separate
flag (`--coarsened-variant-root`, not `--variant-root`) and swaps these two
rows; it does not hard-wire one reading. Getting this wrong would let the
coarsening arm be penalised for doing exactly what it was built to do. The
re-decide variant spec has no field for this — coarsening is a catalog change,
not a payload override — so it has to be declared at scoring time.

² `trivial` is a severity/threshold failure, not a wording failure. No prompt
change makes a model reject a *true* claim about a mildly dated cabinet. These
are recorded and reported but never drive arm selection — they belong to the
severity follow-up the re-tag opened (`too_trivial` crossed its 10% gate).

³ A dirB card the reviewer now judges `absent` means **Terra was right to
reject it**. Recovering it would manufacture a new false positive, so it
inverts from a recovery target to a loss. Three re-tag notes already point this
way ("This is my miss, terra is correct").

## 5. The re-tag cross-check (validation, never a label source)

The 46 re-tag answers are **not** translated into v1.1 labels — every one is
re-asked under §3. But each answer does imply an expectation, and the emitter
reports agreement so the two exercises can be compared:

| re-tag answer | expected v1.1 |
|---|---|
| `wholly_false` | claim `absent` |
| `mechanism_only` | claim `misnamed` |
| `too_trivial` | work `trivial` |
| `cannot_tell` | `inconclusive` |
| `terra_miss` | claim `exact` |
| `wording_blocked` | claim `misnamed` |
| `borderline` | *(no expectation)* |

Disagreement is a finding, not an error — `mechanism_only` in particular is
expected to split, because four of its notes say the claim was actually right
("The LLM claim is true they are dated") while others say it was misnamed ("I
see stains not scuffs"). That split is the whole reason a mechanical
translation was rejected.

## 6. Blindness, and its limits

- Phase 1 and 3 cards carry **no** prior verdict, no re-tag answer, no Terra /
  2f / Sol text, no strata, no prices, no disposition. Those fields are held in
  a top-level `provenance` block that `review_server.py` never sends to the
  browser (it serves `blind_card()` and `reveal_payload()`, both card-scoped).
- Adjudication cards get **fresh minted ids** (`adj_<12 hex>`), so pointing the
  server at the wrong verdict log yields zero matches instead of a queue that
  silently looks half-done.
- Repeat cards are byte-identical in content, placed ≥ 15 positions after their
  origin, and indistinguishable in the UI (the page never renders `card_id`).
  Only cards early enough in a phase to leave room for the full gap are
  eligible; the order is a hash, so that restriction does not correlate the
  repeat sample with anything on the card.
- **Pre-committed tie-break:** when a repeat disagrees with its primary, the
  **primary** answer is the label. The repeat measures consistency; letting the
  second answer win would make the measured rate meaningless and would
  re-introduce the anchoring the blind arm removes. Disagreements are listed in
  the report, never silently resolved.
- **This is not independence.** It is the same reviewer, on cards he has seen
  before, in a program whose conclusions he already knows. It removes anchoring
  on the recorded label; it does not remove memory or motivated reasoning. The
  honest claim v1.1 can make is *"re-adjudicated under a schema that separates
  the axes"* — not *"independently verified"*. An independent second reviewer
  remains the only thing that would support the stronger claim, and the fresh
  v2 cohort is where that question belongs.

## 7. Consequences to expect, and what to do about them

**The hard-false class will probably shrink.** Five of the twenty
`hard_false_billed` cards drew `mechanism_only` on the re-tag and three of
those carry notes that read as outright reversals; five more drew
`too_trivial`, which §4 routes to `trivial_billed`. The canary count of 11
could plausibly land nearer 5–7.

That is a **power problem for Session F**, and it must be faced before tokens
are spent: fewer win-opportunities against a 6.4% replica-flip floor means a
smaller separation between arm and control. **Required before Session F runs
live:** redo Session B's control-arm sample-size note against the v1.1 class
counts and state plainly whether the scorecard can still distinguish an arm
from control. If it cannot, the answer is a larger scoreable population, not a
looser threshold.

This is also the argument for doing the repair *now*: running three arms at
~1.8M Terra tokens each against labels known to contain reversals would buy an
expensive answer to the wrong question.

**Follow-up, not this session:** `scripts/review_analysis.py` recomputing the
weighted population rates under v1.1 for a side-by-side against the frozen
13.6% / 10.4%. Its single join seam is `join_records()` (`:874`). It is audited
tooling that produced frozen reports, so it gets its own session and its own
validation, and the frozen `review_analysis.*` outputs are never overwritten.

## 8. Files

| file | state |
|---|---|
| `reports/adjudication_queue.json` | built by `scripts/build_adjudication_queue.py`; pure function of the frozen inputs |
| `reports/adjudication_verdicts.jsonl` | append-only, latest-wins, written by `review_server.py` |
| `reports/labels_v1_1.json` | the overlay: `card_id` → both axes, class, provenance, agreement with the re-tag |
| `reports/labels_v1_1.md` | human-readable side-by-side: v1 vs v1.1 class movement, repeat consistency, re-tag agreement |

Run:

```bash
.venv/Scripts/python.exe scripts/build_adjudication_queue.py --check
```

```bash
.venv/Scripts/python.exe scripts/review_server.py --queue reports/adjudication_queue.json --verdicts reports/adjudication_verdicts.jsonl
```

```bash
.venv/Scripts/python.exe scripts/build_labels_v1_1.py
```

`scripts/score_redecide_variants.py` takes `--labels reports/labels_v1_1.json`
to score against the corrected classes; with no flag it keeps asserting the
frozen v1 counts (125 cards, 11/55/6/16) and behaves exactly as before.
