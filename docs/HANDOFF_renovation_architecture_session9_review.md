# Handoff — Session 9 canary review (helping Steven clear 154 review items)

Date: 2026-08-21. Branch: `renovation_architecture_rework` (Session 9 code
UNCOMMITTED, suite 2337 green). This is a **review-assistance** session:
no code changes, no provider calls, no reruns. The job is to help Steven
make the remaining review decisions and flip the cutover report to
`release_ready: true`.

## State

- Both canary replicas are complete and verified: 36 artifacts, all with
  complete v5 envelopes, under `artifacts_canary/renovation_session9_20260818/`
  (`run_1/candidate`, `run_2/candidate`, `input_freeze.json`).
- The comparator has been run. All mechanical gates pass (schema, publication,
  reconciliation, accepted-work). Only `manual_review` fails: 967 review
  items need a decision + non-empty explanation.
  - Report: `reports/renovation_architecture_session9_canary_20260821.json`
  - Corpus: v5 vs v4 = +11.9%/+20.1% (run 1), +12.6%/+21.1% (run 2)
    (Session 8 replay anchor was +7.7%/+17.6%); replica 2 vs replica 1 =
    +0.6%/+0.9% corpus-wide, but individual properties swing up to ±50%
    (upstream detection churn ~25% of conditions per property).
- `scripts/cluster_session9_reviews.py` sorted the 967 into three tiers and
  wrote:
  - **Digest (read this first):** `reports/session9_review_digest_20260821.md`
  - **Working review file:**
    `reports/renovation_architecture_session9_reviews_20260821.prefilled.json`
    (813 of 967 already filled; the pristine blank template is the
    non-`.prefilled` file beside it — leave it alone).

## The three tiers

- **Tier 1 — 736 auto-accepted.** Machine-verified no material change
  (identical dollars, ≤$2 rounding, catalog id absent from every v4 row in
  the corpus = the accepted 48-item v5 class, package key migration,
  package metadata-only). Steven can skim the criteria; no per-item work.
- **Tier 2 — 77 items in 4 policy groups, prefilled as DRAFTS.** Each is one
  real decision; the digest states the corpus dollar effect and lists each
  merge group once. Steven should read each group and keep/edit/reject the
  drafted explanation. The big one: *merge dedup prices shared scope lower*
  (56 items, −$13,616 low / −$231,270 high — the "v5 prices shared scope
  lower" term from `docs/analysis/session8_floor_vs_sum_trace.md`).
  Rerun the tool with `--blank-drafts` if he wants them emptied.
- **Tier 3 — 154 individual, blank.** merge groups that increased (8 items /
  3 groups), 47 packages, 27 dropped-by-Terra-disposition, 28 dropped-other,
  12 unit reattributions, 14 headline deltas >15%, 18 run-to-run stability
  (one per property). Each line in the digest carries its evidence.

## Gotcha that matters for judging scope items

The comparator's scope view lists a merged v5 work item's **full price under
each constituent catalog id**. A 3-item merge therefore looks like three
separate increases even when the group's total fell. Judge merges at the
group level (the digest already does; the 8 "merge_increased" items are 3
groups). Scope items compare v4 vs v5 **within run_1 only**.

## How to finish

1. Walk Steven through the 4 Tier-2 groups; record his decision per group.
2. Work the 154 Tier-3 items with him, writing `decision` (`accepted` or
   `approved`) + a non-empty `explanation` into the `.prefilled.json` file.
   Useful context per category: headline deltas should mostly decompose into
   v5-class additions + package floor lift minus merge dedup (Session 8
   trace); dropped-by-disposition items show the v5 disposition inline;
   stability items reflect upstream nondeterminism v4 shares.
3. Re-run the comparator (exit 0 + `release_ready: true` is the gate):

```powershell
.\.venv\Scripts\python.exe tools\compare_renovation_architecture_cutover.py `
  --run artifacts_canary\renovation_session9_20260818\run_1\candidate `
  --run artifacts_canary\renovation_session9_20260818\run_2\candidate `
  --freeze artifacts_canary\renovation_session9_20260818\input_freeze.json `
  --reviews reports\renovation_architecture_session9_reviews_20260821.prefilled.json `
  --report reports\renovation_architecture_session9_canary_20260821.json
```

Review ids are bound to this freeze's sha; do not regenerate the template.
Then Steven returns to the Session 9 session with the result (release_ready
or the list of things he rejected) for the cutover decision.
