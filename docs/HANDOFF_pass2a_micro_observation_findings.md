# Handoff: micro-observation surplus found by the human-gold Pass 2a benchmark

Findings only. This document records what was measured and observed on
2026-08-10; it deliberately contains no recommendations. The decision on what
to do — about the benchmark and/or the wider system — belongs to a later
session.

## Where this happened

- Worktree `C:\Users\Steven\PycharmProjects\rv-pass2a-bench`, branch
  `pass2a_prompt_bench`. All artifacts referenced below live under
  `benchmarks/pass2a-prompt/`.
- The benchmark had just been reworked (uncommitted) from image-based Sol
  judging to: frozen human gold (112 conditions, 15 photos, 2 properties) + a
  blinded text-only semantic matcher (gpt-5.6-terra, low reasoning) + a human
  review CSV + net-observation scoring (gold matches minus human-confirmed
  unsupported additions). 75 unit tests pass; the matcher stage ran to
  completion (90/90 calls) on the existing k=3 generation runs for the
  `baseline` and `checklist` prompt variants.
- Scoring decisions already made by Steven before the run: keep the 25-cap;
  count unsupported additions across all lineages; non-blocking 10% audit
  flag; first round = re-score checklist from existing artifacts.

## The observation that triggered this handoff

Of 3,144 generated atomic claims (both variants × 3 repeats), the matcher
returned 494 match / 392 ambiguous / 2,258 no_match. Joining every pending row
to the archived Sol judgments (join key `(photo, variant, repeat,
claim_index)`; 0 misses — the claim sets are identical):

| pending bucket | Sol supported | Sol uncertain | Sol unsupported |
|---|---|---|---|
| no_match (2,258) | 1,704 | 431 | 123 |
| ambiguous (392) | 302 | 82 | 8 |

30 pending rows carry Sol critical-category unsupported labels.

The 1,704 `no_match + Sol-supported` rows were screened by token overlap
against the gold conditions on their own photo:

- **7 rows** overlap ≥0.4 — plausible matcher near-misses (e.g. obs "The
  ceiling fan appears basic" vs gold "Ceiling fan is dated").
- **208 rows** overlap 0.2–0.4 — predominantly presence-statements where the
  gold states a defect (e.g. obs "Masonry walls are painted" vs gold "Painted
  masonry walls show peeling paint and staining"). The matcher's
  same-component-AND-same-condition contract rejects these by design.
- **1,489 rows** overlap <0.2 — no corresponding gold condition at all.
  Sampled texts: "Laminate countertops are older", "The concrete approach is
  basic", "Some visible wiring appears tangled or unsupported", "The porch
  canopy paint appears aged", "Lower cabinets are white painted … in a dated
  basic style".

These 1,489 are true statements (Sol graded them supported against the image)
describing conditions the human gold deliberately does not contain. Steven's
read, quoted from the session: "1704 rows is insane and there are definitely
not that many issues in the photos. observations sure".

## Scale of the surplus, per variant

From the match-stage dry run (`runs/match_baseline_vs_checklist/dry_run.json`)
over 45 photo-repeats per variant, 15 photos, gold = 112 conditions (7–8 per
photo):

| | baseline | checklist |
|---|---|---|
| total claims | 630 | 2,514 |
| max claims in one photo-repeat | 25 | 98 |
| photo-repeats exceeding the 25-observation 2d cap | 0 | 42 of 45 |
| claims never offered to catalog resolution (`resolution_not_attempted`) | 0 | 1,103 |
| claims filtered by 2c | 174 | 300 |
| resolved to a catalog id | 280 | 669 |
| retained but unresolved by 2d | 176 | 442 |

So for the checklist prompt, ~44% of generated claims never reach catalog
resolution (production runs the same 25 cap), and the majority of the
remainder describe conditions below the gold's bar.

## What the coverage signal says (matcher is not the flaw)

Matcher gold coverage, distinct `(photo, gold_id)` matched in any repeat:
baseline 62/112 (55%), checklist 88/112 (79%). Per-repeat: baseline 31/39/41,
checklist 68/70/68. The archived, fully independent image-based Sol round
measured 52.6% / 88.7%. Two different judges with different modalities agree
on the shape; the matcher runs ~10pp stricter on checklist. The near-miss
count (7 of 1,704) is small. The surplus is a property of what Pass 2a
generates, not of the measurement.

## Facts about the system this exposes (as observed, no interpretation)

1. The checklist prompt yields up to 98 atomic claims for a single photo whose
   human-authored condition list has at most 10 entries; the surplus is
   dominated by true, minor, style/age/presence observations.
2. Pass 2c's exclusion lanes (good_condition, neutral_presence, etc.) removed
   only 300 of checklist's 2,514 claims; the micro-observations above passed
   2c as kept degradation/modernization observations.
3. The production `max_resolve_per_image=25` cap is the only mechanism that
   currently limits how many of these reach catalog resolution, and it
   truncates positionally (first 25 kept observations), not by importance —
   verified positional in all 90 checkpoints.
4. The frozen gold's own note already flagged a related issue: two basement
   photos contain "technically-true conditions that are normal-for-context",
   kept intact, context weighting deferred.
5. The prior Sol round's reject verdict for checklist was driven by critical
   hallucinations on 3 photos; its per-claim grading nevertheless scored 96%
   of checklist claims supported-or-uncertain. Neither judging scheme has a
   category for "true but immaterial" — the new benchmark surfaces them as
   no_match, the old one as supported.

## Open questions the next session must decide (listed, not answered)

- Whether "true but not in gold" claims should cost, earn, or be ignored by
  the benchmark score, and what that implies for the review-burden contract
  (spec currently requires all 2,650 no_match/ambiguous rows adjudicated).
- Whether the gold's 7–8-conditions-per-photo bar is the intended definition
  of "what matters" for the pipeline as a whole — and if so, where in the
  pipeline (2a prompt, 2c filter, 2d cap, or elsewhere) the surplus should be
  addressed, given that today it is only cut by a positional 25-item cap.
- Whether the 1,103 checklist claims never offered to resolution invalidate
  checklist's pre-2f dollar/stability comparisons.

## State as of this handoff (nothing committed, nothing decided)

- Uncommitted working-tree changes: `tools/benchmark_pass2a.py` (+~1,000
  lines: match/review/repin-gold/report stages), `tests/test_benchmark_pass2a.py`
  (+50 tests, 75 total green), `benchmarks/pass2a-prompt/config.json`
  (matcher block), regenerated `runs/report.json` / `report.md` (timestamp,
  legacy retitle, empty `match_rounds` — archived Sol data intact).
- Paid artifacts on disk: `runs/match_baseline_vs_checklist/photos/` (15
  files, 90 calls) and the exported `runs/review_baseline_vs_checklist/review.csv`
  (3,144 rows, 2,650 pending, 49 audit flags). No decisions imported; no gold
  edits; no repin.
- Two AskUserQuestion rounds on review scope / gold expansion were dismissed
  pending this handoff.

## Reproduce the numbers

```
cd C:\Users\Steven\PycharmProjects\rv-pass2a-bench
C:\Users\Steven\PycharmProjects\realtorvision-backend\.venv\Scripts\python.exe tools/benchmark_pass2a.py match --round baseline_vs_checklist --dry-run
```

The Sol-join and overlap screens were ad-hoc scripts over
`runs/review_baseline_vs_checklist/review.csv`,
`runs/judge_baseline_vs_checklist/judgments.json`, and
`gold/reference.json`; the tables above record their outputs.
