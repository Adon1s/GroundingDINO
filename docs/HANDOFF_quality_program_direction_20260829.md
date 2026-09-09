# Handoff — decide the quality program's direction after Session F was blocked

Written 2026-08-29 · backend `renovation_architecture_rework` · commissioned by
Steven

**Purpose.** Session F (Terra v2 tuning) is blocked and the measurement
programme that was supposed to feed it has just been re-measured and found too
thin to decide anything. A fresh session is to determine the **best path
forward** — including the larger architectural and design calls that the last
year of sessions deliberately deferred.

**This document is context only.** It states the situation, the locked
decisions, the measured facts with their limits, the constraints, and the open
questions. It contains **no recommendation about which path to take** — that is
this session's to produce. Where the docs already record a direction, it is
marked as such and cited, so it can be followed *or* deliberately reopened, but
not accidentally relitigated.

---

## 1. The decision frame (the record, not this author)

Steven's stated frame, unchanged since 2026-08-21 and recorded in
`PROPOSALS_output_quality_improvements_20260826.md` §0:

- He is **not a renovator**. Prices are provisional and were hidden during
  review. Price calibration is a separate thread that needs a renovator.
- The v5 rework is judged on **(1) fewer hallucinations** and **(2) more useful
  observations that build packages**. Dollar deflation from dedup is accepted.
- Memory note: `steven-priorities-hallucinations-over-pricing`.

Against that frame the programme decomposed "quality" into four measured defect
classes (D1 false billed conditions, D2 lost real conditions, D3 package-level
failures, D4 multi-bath under-billing). **Section 4 below revises what is
actually known about D1 and D2.**

---

## 2. Where the programme actually stands

| thread | state |
|---|---|
| Session 9 cutover | **Live.** `RENOVATION_ARCHITECTURE_MODE=new` + `KIND_ONTOLOGY_VERSION=observation_kind_v2` (catalog 3.1) since 2026-08-21 |
| 7-day observation window | **Closed early 2026-08-27**, day 6 of 7, no rollback trigger fired. Freeze on estimate behaviour / prompts / catalog is **lifted** |
| Wave 1 | **QP3 only.** QP6, QP5, QP4 were *not* implemented (see §6) |
| Session B (harness + tooling) | **Done.** `scripts/redecide_renovation_architecture.py` + `scripts/score_redecide_variants.py` exist and are dry-run tested; zero tokens spent |
| Mechanism-vs-perception re-tag | Complete 2026-08-28, formally **inconclusive** |
| v1.1 label repair | Complete 2026-08-29 — `docs/RESULT_label_v1_1_20260829.md` |
| **Session F (Terra v2 tuning)** | **BLOCKED.** Does not run as chartered |
| Sessions G (wave-2 canary), H (round-2 review) | downstream of F; both stalled |
| FE v5 adoption | **Held on v4** — and the hold now has no expiry (§5) |

---

## 3. What is locked, and what "locked" means

Steven adopted S1–S9 on 2026-08-26 (`PROPOSALS…` §3; decision log in
`DESIGN_renovation_architecture_decision_packets.md` §9). These are *decisions
of record*. This session may recommend reopening any of them, but should say so
explicitly rather than quietly assuming a different answer.

| # | decision | call |
|---|---|---|
| S1 | QP2 rubric v2 + catalog wording; goal-2 trade accepted | Adopted; **shape unresolved** — F must compare rubric-only / coarsened-only / combined. No 4th verdict lane. Fine mechanism wording not retired |
| S2 | QP3 scope | Adopted — interior-modernization application gate, not tier demotion. **Shipped** |
| S3 | QP4 multi-bath expansion | Adopted to build; ships only on the replay gate (exactness ≥ v4's 6/17, over ≤ 1/17). **Deferred, unbuilt** |
| S4 | QP5 Sol split → sub-packages | Adopted to build. **Deferred, unbuilt** |
| S5 | QP6 single-child rule | Adopted. **Deferred, unbuilt** |
| S6 | Package-warrant photo veto | Adopted as **deferred**; high-consequence pilot reserved as fallback |
| S7 | Round-2 manual review after wave 2 | Committed, ~2.5–3 h |
| S8 | v4 / Pass-2f sunset linkage | **OPEN by design** — decide at FE-adoption time |
| S9 | P3 target architecture | Adopted — **pre-Terra per-surrogate bathroom resolution is the target**; QP4 was the gated interim |

Standing do-nots (QP10, evidence-based, not doctrine): don't shrink Terra
batches for quality; don't add a photo-count gate; don't target individual
catalog items off the concentration tables (item *wording* fixes are different);
don't duplicate the thumbnail thread; don't restructure Sol into a warrant judge
this cycle.

---

## 4. What the measurements actually support now

Three exercises ran against the same frozen evidence. Read them in order:
`reports/review_analysis.md` (v1, frozen) → `RESULT_retag_mechanism_vs_perception_20260828.md`
→ `RESULT_label_v1_1_20260829.md`.

**The headline shift.** v1's single `unsupported` label was collapsing three
different judgments — *the condition is absent*, *it is real but named wrong*,
*it is real and named right but not worth billing*. Splitting them changes the
picture materially:

| canary | v1 | v1.1 |
|---|---|---|
| hard-false (claim absent) | 10.51% [4.73, 23.00] | **2.66%** [0.69, 12.60] |
| like-for-like broad (absent + misnamed) | 13.58% [6.84, 26.60] | **8.12%** [3.41, 19.95] |
| production hard-false | 10.42% [5.09, 32.46] | **8.94%** [3.62, 30.99] |

Canary billed-error mass now splits **misnamed 49.4% / trivial 26.5% / absent
24.1%**. Roughly half the "errors" v1 counted are real conditions Terra found
and described imperfectly.

**What that does not license.** Every v1.1 band overlaps its v1 band; each
hard-false rate rests on **one uniform card** (canary 2.392 pp/card, production
6.474 pp/card); one listing (`redfin_25809814`) supplied 7 of 11 canary
hard-falses; the canary/production divergence is not statistically separable;
and the whole exercise is one non-independent reviewer re-judging his own
labels, with the hard-false class re-examined **only in the non-blind arm**.
`RESULT_label_v1_1_20260829.md` §"Why the drop is not a measured safety
improvement" is the authority; its discard list names four claims that failed
adversarial verification and must not be carried forward.

**The one blind result robust to any noise floor:** 0 of 79 blind `supported`
cards were relabelled `absent` (95% Wilson upper 4.6%).

**Nothing about Terra changed.** v1 and v1.1 score byte-identical v5 outputs.
No claim about model behaviour, package accuracy, Pass-2f correctness, Terra
recall, or pricing follows from either.

---

## 5. Why Session F is blocked, and the consequence that is easy to miss

**The arithmetic.** The canary scoreable population no longer supports the
chartered arms:

| leg | canary n | status |
|---|---:|---|
| `hard_false_billed` | **3** | paired exact McNemar best attainable **p = 0.125** — cannot reach α = 0.05 under *any* outcome, including a perfect 3/3. Effective independent n ≈ 2 |
| `dirB_recovery` | 16 | the only win leg with a pulse (needs ≥ 6/16) |
| `supported_billed` | 57 | the only well-powered column (detects loss inflation above ~12/57) |
| `misnamed_billed` | 7 | needs a ~37–42% true rate |
| `dirB_wording_recovery` | **0** | the coarsening arm's distinctive win condition has no instances |

Three scorer defects must also be fixed before any arm runs: there is no dirB
*loss* population (only cards v1 already called recoveries were re-asked, so
`dirB_recovery` could fall but never rise); `load_population_v1_1` silently
excludes production; and the `SCORING` policy gives a coarsening arm no way to
earn credit on `misnamed_billed`.

Also: the **6.4% noise floor of record is a blend of two significantly
different directional rates** — supported→not-supported 4.05% [2.15, 7.52] vs
unsupported→supported 23.1% [11.0, 42.1], Fisher p = 0.0019. The dirB gate is
calibrated against the wrong one.

**The consequence.** The FE hold was explicitly scoped: hold v4 "until wave 1 +
wave 2 land and the fresh canary passes" (`PROPOSALS…` §0). Wave 1 is complete;
wave 2 runs through Session F; Session F is blocked. **The FE hold therefore has
no expiry.** Its cost is not hypothetical: v4's Pass 2f keeps burning the
*binding* Sol quota, which is why production caps at ~5–6 premium listings/day.
Sunsetting 2f after FE adoption roughly triples Sol-side headroom, and that is
the forcing function behind the still-open S8. Whether the FE hold should
survive a blocked Session F is a live architectural question, not a settled one.

---

## 6. The architectural questions actually on the table

These are the "larger decisions" the programme deferred. Each has a recorded
reason; none has been decided.

1. **Where the effort belongs: Terra, or upstream of Terra.** `STATE_kind_ontology_program.md`
   records that cutover was blocked not by the ontology but by **run-to-run
   estimate variance under gpt-5.6-terra**, because observation generation
   (2a/2b/2c) is unstable and flat room allowances turn one flipped observation
   into a $10–20k line. Upstream replica churn is **35%/51%**. The Pass 2a
   prompt ablation is listed there as the **next task** and has never run. The
   v1.1 finding that ~half the billed-error mass is *mis-description of real
   scenes* is consistent with a wording problem that originates upstream of
   Terra. Whether the programme should be tuning the verifier at all before the
   generator is stabilised is an open call. Note the stop-gate in
   `HANDOFF_pass2a_variance_ablation.md`: run the attribution leg first; if
   issues still flip when replaying downstream from a frozen 2a capture, the
   variance is in 2c and tuning 2a will not fix it.
2. **Work-item identity (blocks S3/QP4).** Wave 1 dropped QP4 because the
   per-surrogate child partition is **unrepresentable with current merged
   work-item ids** (`RESULT_quality_wave1_qp3_20260827.md`). S9 already names
   the target architecture — option (c), resolving distinct bathrooms *before*
   Terra, one estimate unit and one independently reviewed condition set per
   bathroom. QP4 was the cheap experiment meant to tell you whether (c) is worth
   building; it never ran. Known risk of (c): it makes surrogate clustering
   load-bearing earlier, and the review showed surrogates **over-count** real
   bathrooms (a utility room counted as a bathroom; humans counted fewer baths
   than surrogates on 11216660 3→1, 10866780 4→2, 80917686 6→4).
3. **Split authority (blocks S4/QP5).** Deferred because the stored
   reject-with-split authority is **ambiguous** — a contract question, not an
   implementation one. The measured cost of leaving it: a −52% headline on
   redfin_11079485 and three zeroed exterior packages including P07, which the
   human judged warranted.
4. **The benchmark architecture itself.** The current benchmark is
   **closed-world**: it only judges claims the pipeline already generated, so it
   cannot see renovation work the upstream system never proposed. Steven has
   sketched a three-part structure (adjudicated v1.1 regression set, a fresh
   blind v2 cohort for ship decisions, and a small open-world listing
   benchmark). Session H is currently chartered as the fresh-cohort measurement
   (QP-M's two uniform arms: rejected-condition recall, package-warrant
   precision) but is stalled behind G. Whether measurement should be re-sequenced
   *ahead* of tuning is open.
5. **S8 — v4/2f sunset.** Retiring v4 kills **both** dirA and dirB strata, i.e.
   the entire disagreement telemetry this programme's sampling design rests on.
   The recorded options are: accept the telemetry loss, or ship the per-item
   Pass-2F revival (dormant scaffolding kept for exactly this). Do not bump
   `PASS_2F_PROMPT_VERSION` without extending
   `PASS_2F_ISSUE_INDEPENDENT_PROMPT_VERSIONS`.
6. **Package-level warrant (S6).** Deferred on evidence that no measured judge
   is reliable on the contested slice (P1 census: 11 warranted / 11 not; Sol
   approve 9/10 carries no warrant signal). The high-consequence pilot —
   veto-to-standalone only, never deletion, scoped to full modernizations or
   high-uplift packages — remains the sanctioned fallback.
7. **Pricing.** The 42 split successors have **no authored prices**; they
   inherit their v1 parent's economics verbatim (`inherited_from_split_parent`),
   and the `KIND_MULT` values are temporary. Do not read current estimates as
   intentional. Needs a renovator; it is not this session's to solve, but it
   bounds what any dollar-denominated claim can mean.
8. **Task 4B / 4C.** ~1,264 historical artifacts are still catalog 2.1; 4C
   (legacy retirement) is blocked on 4B. Not started.

---

## 7. Constraints that bind any plan

- **Quotas.** Sol binds production throughput (250k/day ≈ 5–6 premium
  listings/day today, ~29 at ~8.5k without 2f). Terra is 2.5M/day shared with
  upstream (~10–12 listings/day all-service). One re-decide pass ≈ 0.9M Terra
  tokens; a variant-vs-control comparison ≈ 1.8M ≈ one free Terra day. A fresh
  freeze + two-replica canary ≈ 8M ≈ ~4 free-paced days.
- **The human labels do not transfer.** They join only to the *stored* canary
  run_1 conditions; 35%/51% upstream churn breaks the join on any fresh canary.
  This is why the re-decide harness exists and why "compare variants only
  through fresh canaries" was explicitly **rejected** (`PROPOSALS…` §4).
- **Frozen and untouchable:** `reports/review_queue.json`,
  `reports/review_verdicts.jsonl`, `reports/review_analysis.*`,
  `reports/retag_*`, all `reports/session9_*`, and the canary root
  `artifacts_canary/renovation_session9_20260818`. New work uses new paths.
- **`RV_ROOT` is the live production working tree** — no `git checkout` of other
  branches; use a worktree.
- `scripts/replay_renovation_architecture.py` stays structurally provider-free.
- Catalog changes go through `tools/catalog_migrations/kind_v2_decisions.json` +
  regeneration, **never** hand-edits. A parity test pins this.
- Tests: `.venv\Scripts\python.exe -m pytest` (bare `python` does not resolve
  the venv). Suite currently 2433 passed / 6 skipped.
- Two production listings (`redfin_10965375`, `redfin_10922002`) are thumbnail
  evidence end-to-end and excluded from all measurement. The thumbnail ingest
  fix is its own thread (`HANDOFF_thumbnail_photo_ingest_fix.md`), non-blocking,
  scheduled 2026-08-29+.

---

## 8. Coverage gaps a plan has to reckon with

- **Review coverage, not tokens, is the current bottleneck.** 671 canary
  accepted conditions are unreviewed. 125 of 173 condition cards carry v1.1
  labels; **0 of 22 package** and **0 of 17 bathroom** cards do. 32 production
  conditions are unreviewed. 20 canary dirB cards were never adjudicated under
  the two-axis schema — that is the only place the coarsening hypothesis can be
  tested and the only source of a dirB *loss* population.
- **The `absent` boundary has zero blind and zero repeat coverage**, and the
  headline rests on it. The repeat arm's 8/10 (95% Wilson [49.0%, 94.3%]) cannot
  distinguish a 50% reviewer from a 95% one, and both its disagreements sit on
  the exact-vs-misnamed boundary — the same boundary separating
  `supported_billed` from `misnamed_billed`.
- **There is no inter-rater figure anywhere in this programme.** Per
  `DESIGN_label_v1_1_adjudication.md` §6, a second reviewer is the only thing
  that converts "re-adjudicated" into "verified".
- **Review pace was median 8 s/card** against the design's ~45 s budget, with 5
  free-text notes on 135 cards. Treat as a signal about label quality.
- Terra **recall** remains unmeasurable (dirB is targeted and 2f-confirmed-only);
  package-warrant precision has no denominator. QP-M's two uniform arms were
  designed to fix exactly these and have not run.

---

## 9. Tooling and data inventory

| asset | what it is |
|---|---|
| `scripts/redecide_renovation_architecture.py` | live re-decide harness, variant/ledger/fingerprint-safe, dry-run tested, never run live |
| `scripts/score_redecide_variants.py` | variant scorer; `--labels` selects v1 or v1.1, `--coarsened-variant-root` declares a claim-text-changing arm |
| `tools/label_schema.py` | the two-axis schema, classes and scoring policy (pre-committed) |
| `scripts/build_adjudication_queue.py` | queue builder with an enforced blindness check |
| `scripts/build_labels_v1_1.py` | overlay emitter + side-by-side report |
| `tools/review_cards.py`, `scripts/build_review_queue.py`, `review_server.py`, `review_tally.py`, `review_analysis.py` | the review stack; the server is queue-driven and takes any vocabulary |
| `reports/labels_v1_1.json` | the corrected overlay, keyed by original `card_id` |
| `reports/review_analysis.json` | frozen v1 analysis; carries per-item rows the md filters out at judged < 3 |
| `reports/session9_decision_packets.json` | agreement matrix, per-item disagreements, P1/P3/P4/P6 packets |

**Does not exist:** any measurement of Terra recall, package-warrant precision,
inter-rater agreement, open-world (missing-work) performance, or price accuracy.

---

## 10. What this session is asked to produce

A recommended direction, with reasoning, covering at minimum:

1. Whether to fix Session F's population and run it, re-scope it, or drop the
   wording programme for now — and what replaces it if dropped.
2. Whether tuning effort belongs on Terra at all before the upstream 2a/2c
   variance is characterised (§6.1).
3. What to do about the FE hold now that its stated expiry cannot be reached
   (§5), including whether S8 must be decided earlier than planned.
4. Which of the deferred architectural items (work-item identity / S9, split
   authority, package warrant) should be picked up, and in what order.
5. What the benchmark architecture should be going forward, given that the
   current one is closed-world and the labels do not survive a fresh canary.
6. A sequenced plan with review-time and token costs, and explicit go/no-go
   gates in the style the roadmap already uses.

Say plainly which locked decisions (S1–S9) your plan reopens, and why.

**What this session must not do:** treat v1.1 as a verified re-measurement;
publish 2.66% as a corrected population rate; derive Terra recall from dirB;
derive Pass-2f correctness from condition verdicts; treat card counts as
population rates; rebuild any frozen input; or spend Terra tokens.

---

## 11. Reading order

1. `docs/RESULT_label_v1_1_20260829.md` — the current state of the evidence and
   why Session F is blocked. **Start here.**
2. `docs/ROADMAP_quality_program_sessions_20260826.md` — the programme plan and
   its amendments; the sessions table is the current state of play.
3. `docs/PROPOSALS_output_quality_improvements_20260826.md` — the QP catalog,
   the §3 locked decisions S1–S9, and QP10's do-nots.
4. `docs/DESIGN_renovation_architecture_decision_packets.md` §9 — the decision
   log of record, including every option call and its evidence.
5. `docs/RESULT_retag_mechanism_vs_perception_20260828.md` — the inconclusive
   result that motivated the label repair.
6. `docs/DESIGN_label_v1_1_adjudication.md` — the two-axis schema, its
   pre-committed mappings, and §6's honest statement of what it can claim.
7. `docs/STATE_kind_ontology_program.md` — the upstream variance problem, the
   unrun 2a ablation, 4B/4C, and the pricing facts that "cost money if
   forgotten".
8. `docs/DECISION_renovation_architecture_session9_cutover_20260821.md` §10 —
   the observation-window outcome and its carry-forwards.
9. `reports/review_analysis.md` §§3–5, 7, 9–12 — the frozen v1 evidence,
   especially §12's bounded-inference rules, which still bind.
