# HANDOFF — Long-term plan needed: prompt benchmarking vs token limits

2026-08-09. Context for designing a sustainable benchmarking strategy. The
immediate experiment this grew out of lives in `tools/benchmark_pass2a.py` +
`benchmarks/pass2a-prompt/` (worktree `rv-pass2a-bench`, branch
`pass2a_prompt_bench`); read `runs/report.md` there for the current verdict.

## The situation in one paragraph

The renovation-estimate pipeline runs on gpt-5.6-terra, which exposes no
temperature control, so identical inputs produce materially different
estimates run to run. Measuring any prompt change therefore requires k
repeats per photo plus an LLM judge to grade claim-level hallucinations —
and the judge (gpt-5.6-sol, chosen for independence from Terra) turned out
to be the dominant cost. One 15-photo, 2-variant, 3-repeat comparison
consumed over half of Steven's Sol token budget in a single round. Steven
wants to iterate on prompts roughly daily; the current design supports
roughly one comparison per budget period. That mismatch is the problem to
solve.

## Why judging costs what it costs

- Every atomic claim from every repeat of every variant is graded
  individually against the image (~3,100 claims for one round; the
  inventory-style variant emits 4x the claims of the baseline).
- Repeats overlap heavily, so most of that grading is re-grading
  near-duplicates.
- Each photo costs an image call (claim grading) plus a text call
  (gold-coverage/recall vs a frozen 112-condition human gold).

## Constraints the plan must respect

1. **Terra cannot judge its own claims.** Unsupported-claim detection is
   the metric under test; asking the generator to grade itself points
   self-preference bias directly at the reading. Terra IS acceptable for
   gold-coverage (recall is anchored to the frozen human gold) and any
   metric that is pure text-matching.
2. **Absolute counts matter, not rates.** Costing flips full allowances on
   single observations, so "more claims, lower unsupported %" can still
   mean more bad dollars. Any cheap metric must preserve absolute
   hallucination counting.
3. **Half the decision needs no judge at all.** Jaccard of resolved-ID
   sets, run-to-run cost spread, midpoint shift, claim volume, and
   cost-bearing flips are computed deterministically from artifacts.
   Generation-only runs (Terra, comparatively cheap) already kill bad
   variants.
4. Local models (Qwen on LM Studio) are free but the machine is
   resource-contended (24GB VRAM, shared with everything else) and the
   27B VLM is the same family class as the passes being measured — vendor
   independence is worth preserving somewhere in the loop.

## Measured facts to build on (from the completed round)

- A ~30-term keyword screen (moisture/structural/hidden/hedging language)
  flags 7.6% of claims and catches 65% of known critical hallucinations →
  ~92% judging reduction at ranking-grade (not certification-grade)
  fidelity. Screen-then-judge is viable for iteration; full judging only
  for a release candidate.
- Critical hallucinations concentrate heavily (9 of 31 on one basement
  photo). Judging a small photo subset where failures concentrate is
  nearly as informative as judging everything, for iteration purposes.
- 15 photos of full Sol judgments now exist and are committed — a free
  calibration set for measuring ANY cheaper judge (Terra, a local model,
  screen+judge hybrids) by label agreement against Sol before trusting it.

## Directions worth evaluating (not yet decided)

- Screen-then-judge: keyword/embedding pre-filter, judge only flagged
  claims; full Sol pass reserved for finalists.
- Dedup before judging: grade the per-variant claim union instead of
  per-repeat duplicates (loses per-repeat attribution; the ">=2 of 3
  repeats" gate would need reformulating).
- Judge tiering: calibrate a cheap judge (Terra on non-self-generated
  text, or local Qwen/Gemma) against the existing Sol judgments; use Sol
  only where the cheap judge is proven weak.
- Cadence: generation-only runs daily (deterministic metrics), judged runs
  weekly/on-demand for surviving candidates.
- Photo-subset judging: maintain a small "hard set" (basements, dense
  kitchens) where hallucinations concentrate.

## What NOT to redo

The harness (resumable stages, fingerprints, blinding, frozen gold,
pre-2f force-confirmed costing) works and survived a full live round.
The problem is purely the judging token economics, not the benchmark
design.
