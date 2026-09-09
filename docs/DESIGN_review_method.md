# Manual review method for v5 output quality — design and priorities

Written 2026-08-22 · backend `renovation_architecture_rework` · designed from
`docs/HANDOFF_review_method_design.md` · tooling: `tools/review_cards.py`,
`scripts/build_review_queue.py`, `scripts/review_server.py`, `scripts/review_tally.py`.

## 1. What is being measured, and by whom

Steven judges the rework on **(1) fewer hallucinations, (2) more useful
observations that build packages**. Prices are provisional and hidden during
review. The binding resource is his adjudication time, so the queue is ordered
so that stopping at any point still leaves an unbiased, usable tally.

Two independent photo judgments exist per artifact: **Terra** (v5, one verdict
per *condition*) and **Pass 2f** (v4, one decision per *package*, with the
package's issues listed as confirmed/rejected). They are not opposites: a
condition can be visibly present while the package is unwarranted, and a
package can be warranted while one condition claim is wrong. So:

- The human judges **the claim against the photo**, with neutral labels.
- Only Terra is compared at condition level (its verdict is condition-level).
- **Nothing about Pass 2f correctness is derived** from a condition verdict.
  Direction A/B are sampling strata, not a contest. Package warrant is judged
  only on package cards.

## 2. The card and its vocabulary

Condition card — *"Does the photo evidence support the condition claim Terra
was asked to verify?"* The card shows exactly what Terra saw: the catalog
`atomic_claim` text, the evidence observations, the `representative_photo_keys`
strip (near-duplicates not sent are shown muted).

| key | stored verdict | meaning |
|---|---|---|
| 1 | `terra_claim_supported` | true of the photo(s) as stated |
| 2 | `terra_claim_unsupported` | not there — a hallucination |
| 3 | `terra_claim_overstated` | something is there, the claim overstates / mis-describes it |
| 4 | `terra_evidence_inconclusive` | photo too small / angle / dark |

Optional error type on 2/3 (tests "it confuses staining with worn"): `a`
stain/colour read as wear · `b` style/"dated" claim on an ordinary or
renovated finish · `c` wrong object or room · `d` real but overstated · `e`
other. Optional note.

Derived per card (tally): `terra_vs_photo` ∈ agree / false_positive (Terra
supported, human unsupported) / miss (Terra not supported, human supported) /
overstated / inconclusive.

Package card (P1): `package_warranted | not_warranted | unsure`.
Bathroom card (P3): distinct bathrooms (digit) + `per_bathroom | once | unsure`.

**Blind-then-reveal.** Terra/2f/Sol text, strata, replica-2 verdict and prices
are served only after the verdict (or on `r`, which is recorded as a peek).

## 3. Strata and phases

"Accepted" = Terra `supported` + disposition `accepted_for_work` (what v5 bills).

| stratum | definition | answers (handoff §8) |
|---|---|---|
| `dirA` | Terra supported / 2f rejected the issue (same photo); mixed 2f verdicts count here — **100 %** | Q1 Terra false positives where a second model objected; concentration |
| `terra_flip` | same (item, unit, photo-set) in canary run_1/run_2, different Terra verdict | Q7 which way Terra errs when unstable |
| `uniform` | accepted conditions not in dirA, included iff `sha256(card_id) mod 10000 < rate·10000` (default 7 %) — stateless, idempotent as listings accrue, monotone in rate | with dirA at 100 %: the **overall false-billed rate** = (N_A·r_A + N_nonA·r_nonA)/N_accepted (Q5, Q8); kind / item / photo-count / 2f-status breakdowns (Q1, Q5 shared-false case, Q6) |
| `dirB` | 2f confirmed / Terra unsupported or cannot_assess | Q2 Terra misses (priority 2) |
| `p1_package` | v4 candidate 2f-rejected, v5 candidate Sol-decided | Q3 package warrant |
| `p3_bathroom` | ≥ 2 bathroom surrogates | Q4 multi-bath billing |
| `p6_forced_single` | the 5 accepted-provisionally single-photo items | Q6 |

One card per condition; a card may carry several strata (reviewed once,
tallied in each). Q6/Q7 are covariates on every card (`photo_count`, Terra
batch conditions/images), not extra cards.

Phases (queue order): **1** dirA + terra_flip + uniform interleaved in hash
order (all Terra-supported claims, so the blind hides whether 2f objected —
stopping part-way still yields an A-rate *and* a uniform rate; `--order tier`
restores A-first) · **2** dirB · **3** packages, bathrooms, P6.

Canary sizes: dirA 32, dirB 36, flips 16, uniform ≈ 50, P1 13, P3 10, P6 5 ≈
160 cards (~2.5 h total; phase 1 ≈ 65 min). Production: ~3–4 phase-1 cards per
listing plus ~2 dirB and any package/bathroom cards. **Daily floor ≈ 15–20
min** = the day's production phase-1 cards. `--uniform-rate` is the volume knob.

Milestones: canary phase 1 → design doc §9 "P2 condition truth tally" row and
the Q5 evidence; phase 3 → P1 / P3 rows and the P6(3) revisit.

Not done by design: no model pre-screen (zero provider calls; it would put the
thing under test into the sampler), no Terra "re-decide" probe (Terra quota;
`terra_flip` is the free proxy). Only run_1 canary conditions are reviewed.

Evidence-resolution rule (added 2026-08-26, Steven's decision): cards whose
every known-size photo has a short side under `MIN_EVIDENCE_PX = 500` are
excluded at build time — from the queue *and* from the accepted denominators
(`meta.<source>.low_res_excluded`); photo dimensions are shown on the strips
and low-res photos outlined. These are failed photo fetches, not judgeable
evidence; root cause + pipeline fix: `docs/HANDOFF_thumbnail_photo_ingest_fix.md`.

## 4. Daily loop

```
.venv\Scripts\python.exe scripts\build_review_queue.py [--check]      # idempotent; prints strata counts
.venv\Scripts\python.exe scripts\review_server.py [--host 0.0.0.0]     # http://localhost:8765 ; phone over LAN
.venv\Scripts\python.exe scripts\review_tally.py [--export-legacy]      # reports/review_tally.md
```

Files: `reports/review_queue.json` (pure function of artifacts; no done-state),
`reports/review_verdicts.jsonl` (append-only, latest wins, undo = null verdict),
`reports/review_tally.md`, `reports/session9_decision_worksheet.filled.json`
(a copy with neutral labels; the audited original is never written).

Keys: `1–4` verdict (bathroom: digits = count, `p/o/u` = billing) · `a–e` tag ·
`n` notes · `Enter` save, then `Enter`/`j` next · `j/k` · `s` skip · `z` undo ·
`r` reveal early · `$` prices after reveal · `?` help.

## 5. Correction to the handoff's photo_023 anecdote

`docs/HANDOFF_review_method_design.md` §5 says Terra confirmed the photo_023
flooring claim on `redfin_10965375` and that 2f agreed, so "the disagreement
filter does not catch a hallucination both models share". The artifact says
otherwise: condition `oc1_38b0a263fd54d0f4` (`hard_flooring_scratched_or_worn`
@ bedroom_3) has Terra **`unsupported`** and v5 disposition **`excluded`**; the
"visible wear, uneven sheen" sentence is the upstream 2a/2c *observation*
(`issue cb5c882e37a718f7`), and it was **Pass 2f (v4)** that confirmed it
(`bedroom_repair__bedroom_3`, "uneven sheen, discoloration, and worn-looking
areas") and v4 that billed $242/$4,041 — which the frontend, still rendering
v4, displayed. So this case is a **direction-B** card where v5 was right. The
general concern (shared false claims are invisible to a disagreement filter)
stands and is why the uniform stratum exists; this anecdote just is not an
instance of it. First-listing cards (builder output): dirA 4, dirB 3, uniform
1, P1 2, P3 1 — out of 35 reviewed conditions and 19 work items.

## 6. Constraints honoured

Zero provider calls; no estimate/prompt/catalog change; artifacts read-only
(production root filtered by run-dir name before any JSON is loaded); writes
only under `reports/`; the Session 9 builder and its outputs are read for
legacy ids (C###/P##/B##) and never modified; no `git checkout` in `RV_ROOT`.
