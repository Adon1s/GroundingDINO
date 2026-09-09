# Handoff — analyse the completed manual review and derive conclusions

Written 2026-08-26 · backend `renovation_architecture_rework` (working tree `df35f23` + untracked review tooling) · verdict data frozen as of 2026-08-26 11:13

**Purpose.** Steven completed the manual photo review on 2026-08-26. A fresh session is to
audit the verdict data, run and check the aggregation, and derive conclusions against his
stated priorities. **This document is context only — the situation, the data, its exact
semantics, measured integrity facts, constraints, and where conclusions are expected to
land. It contains no analysis, no interpretation of the verdicts, and no recommendations;
those are the new session's to produce.** The rates themselves have deliberately not been
computed or read in the session that wrote this handoff.

## 1. Steven's stated priorities and decision frame (the record, not this author)

- Evaluation frame (2026-08-21): he is not a renovator; prices are provisional and were
  hidden during review. The v5 rework is judged on **(1) fewer hallucinations** and
  **(2) more useful observations that build packages**. Dollar deflation from dedup is
  accepted. (memory `steven-priorities-hallucinations-over-pricing`.)
- Review-method decisions (2026-08-22, recorded in `docs/DESIGN_review_method.md`):
  disagreements first; **neutral labels** — the human judges the claim against the photo;
  only Terra is compared at condition level; **nothing about Pass 2f correctness may be
  derived from a condition verdict** (Terra is condition-level, 2f is package-level);
  direction A/B are sampling strata, not a contest; package warrant is judged only on P1
  cards; the control is a **uniform** sample over accepted conditions so an overall
  false-billed rate is estimable (dirA at 100 % + uniform non-A, weighted).
- The questions this review was built to answer: Q1–Q8 in
  `docs/HANDOFF_review_method_design.md` §8.
- Where conclusions are expected to land: the blank rows of
  `docs/DESIGN_renovation_architecture_decision_packets.md` §9 (**P2 condition truth
  tally**, **P1 package-warrant**, **P3 multi-bathroom**; P6(3) is `accepted provisionally —
  revisit with the P2 photo pass`), and the findings log of
  `docs/HANDOFF_output_quality_manual_review.md`. Per the cutover record, the P1–P4
  packets gate **frontend adoption of v5**, not the production flip.

## 2. The data

Two files, joined on `card_id`:

- **`reports/review_queue.json`** — 212 cards (173 condition / 22 package / 17 bathroom),
  built by `scripts/build_review_queue.py` from the Session 9 canary (run_1; run_2 only
  feeds `terra_flip`) and the production new-mode artifacts (`--since 20260821_230000`).
  `order: hash`, `uniform_rate: 0.07`. Each card carries: the claim Terra judged
  (`claim.catalog_claim` + `claim.observations`, the exact Terra payload), photo strips
  with absolute paths and `wh` dimensions, `meta` (catalog item/kind, scene, unit,
  condition_id, photo_count, Terra batch conditions/images, `terra_verdict`,
  `replica_terra_verdict` for flips, `direction`, `second_opinion`, `accepted`,
  disposition), `reveal` (Terra rationale, 2f verdict/evidence, Sol decision, v5
  disposition), `hidden` (prices), `strata`, `phase`, `source`, `legacy_item_id`
  (C###/P##/B## of the Session 9 worksheet, plus P6 review ids).
- **`reports/review_verdicts.jsonl`** — append-only; **latest record per card wins**; a
  record with `verdict: null` is an undo. Record: `{card_id, verdict, tag, notes, peeked,
  extra, ts}`.

Verdict vocabularies (defined in `docs/DESIGN_review_method.md` §2): condition —
`terra_claim_supported | terra_claim_unsupported | terra_claim_overstated |
terra_evidence_inconclusive`; package — `package_warranted | not_warranted | unsure`;
bathroom — `per_bathroom | once | unsure` + `extra.distinct_bathrooms` (integer). Optional
condition error tags: `stain_or_colour_read_as_wear`, `style_claim_on_ordinary_finish`,
`wrong_object_or_room`, `real_but_overstated`, `other`. `peeked: true` means the model
text was revealed before the verdict was entered (the review was blind-then-reveal).

Strata definitions, phases, the uniform hash-threshold sampling, the weighted-overall
formula ((N_A·r_A + N_nonA·r_nonA) / N_accepted, denominators in `queue.meta`), and the
`terra_vs_photo` derivation rule are all specified in `docs/DESIGN_review_method.md` §§2–3.

## 3. Measured completion and integrity facts (verified 2026-08-26; counts only, no rates read)

- Log: **205 raw records → 191 net verdicts** (14 same-card overwrites, latest wins;
  0 explicit undos). Timestamps 2026-08-23 20:46 → 2026-08-26 11:13.
- **180 of 212 in-queue cards verdicted.** Per stratum (reviewed of total): dirA 39/51 ·
  terra_flip 16/16 · uniform 47/54 · dirB 45/51 · p1_package 18/22 · p3_bathroom 14/17 ·
  p6_forced_single 5/5.
- **32 cards are unreviewed**, all from the four production listings analysed 2026-08-25
  that entered the queue on the 2026-08-26 rebuild: `redfin_10949071` (15),
  `redfin_10735912` (8), `redfin_10866780` (5), `redfin_80916010` (4). Steven declared the
  review finished on 2026-08-26; whether these 32 are to be reviewed was not stated.
- **11 verdicts are orphaned** — all on `redfin_10965375` cards that were later excluded
  by the thumbnail rule (below). They remain in the JSONL; the tally ignores them.
- All verdict values are within the vocabularies; all 14 bathroom verdicts carry
  `distinct_bathrooms`; **peeked = 0 of 191** (the blind held); 5 verdicts carry tags
  (2 `real_but_overstated`, 3 `other`); **26 carry free-text notes** (unread by this
  session — raw data in the JSONL).
- Denominators (`queue.meta`): canary 18 listings, accepted 743 (dirA 32 / non-A 711);
  production 10 listings, accepted 203 (dirA 19 / non-A 184). dirA condition cards split
  `second_opinion`: 34 `2f_objected` / 17 `2f_mixed` (2f confirmed some issues of the
  condition and rejected others — a sub-class the Session 9 packets did not separate).
- **Thumbnail exclusion (2026-08-26, Steven's decision):** cards whose every known-size
  photo has short side < 500 px are excluded from the queue *and* from the accepted
  denominators (`meta.production.low_res_excluded`: 64 conditions / 36 accepted /
  2 packages / 1 bathroom). Two production listings are entirely excluded
  (`redfin_10965375`, `redfin_10922002` — analysed on CDN thumbnail renditions; root
  cause and pipeline fix: `docs/HANDOFF_thumbnail_photo_ingest_fix.md`).
- `reports/session9_decision_worksheet.filled.json` does **not** exist yet
  (`--export-legacy` has not been run against the real verdicts).

## 4. Corrections on record that bear on reading the data

- The originating handoff's photo_023 anecdote was misattributed: on
  `redfin_10965375` the flooring claim was **rejected by Terra** (`unsupported`, v5
  `excluded`) and **confirmed by Pass 2f** (v4 billed $242/$4,041; the FE renders v4) — a
  direction-B case. Recorded in `docs/HANDOFF_review_method_design.md` (correction block)
  and `docs/DESIGN_review_method.md` §5. That listing's condition cards were later
  excluded wholesale by the thumbnail rule.
- Both excluded listings' verdicts (the 11 orphans) were given before the exclusion
  existed, on thumbnail evidence.

## 5. Tooling available (all zero provider calls; read-only on artifacts)

- **`scripts/review_tally.py`** — the aggregation of record: progress per stratum; verdict
  mix per stratum; `terra_vs_photo` per stratum; weighted overall rate per source (both
  `unsupported` and `unsupported+overstated` measures); breakdowns over Terra-supported
  claims by catalog kind, photo count, second opinion, Terra batch buckets, catalog item
  (n≥3), source; tag distribution; flips (which replica matched the human); P1/P3 tables;
  orphan and exclusion notes. `--export-legacy` fills a **copy**
  (`session9_decision_worksheet.filled.json`) with neutral labels — the audited original
  is never written. Writes `reports/review_tally.md`.
- `tools/review_cards.py` — loaders (`v5_result` both placements, `iter_runs`,
  `load_canary`), the 2f→condition join (`join_2f`, `direction`), `terra_flips`,
  `latest_verdicts`, denominator logic; tests in `tests/test_review_cards.py`
  (14, incl. a canary integration test reproducing the Session 9 counts 85/68/32/36/16).
- `scripts/review_server.py` — re-view any card with photos
  (`.venv\Scripts\python.exe scripts\review_server.py`, http://localhost:8765); resumable;
  re-verdicting a card just appends (latest wins).
- Cross-reference data: `reports/session9_decision_packets.json` (Session 9 evidence:
  agreement matrix, per-item disagreement table, P1/P3/P4/P6 packets, single-photo-share);
  canary artifacts `artifacts_canary/renovation_session9_20260818/run_{1,2}/candidate/`;
  production artifacts `C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts/<key>/<run>/`;
  addresses in FE SQLite `renointel-prod/prisma/dev.db` (`Property`, read-only `mode=ro`).
- What does **not** exist: a Terra "re-decide" mode (re-asking Terra costs quota; scoped in
  `docs/DESIGN_renovation_architecture_decision_packets.md` §2), confidence intervals in
  the tally, and a flat per-card gold export (both were discussed with Steven on
  2026-08-23 as possible additions, not built).

## 6. Constraints

- The 7-day observation window (day 1 = 2026-08-22) is still running as of this writing:
  **no changes to estimate behaviour, prompts, or catalog** until it closes; reading,
  tallying and adjudicating are unconstrained. Any Terra prompt/rubric change invalidates
  the canary deltas (fresh freeze + new two-day canary).
- Prices are provisional — dollar-shaped findings are "needs a renovator".
- Do not re-open the 967 Session 9 review items; do not modify
  `scripts/build_session9_decision_packets.py` or the `reports/session9_*` outputs
  (audited; the worksheet export writes a copy).
- `RV_ROOT` is the live working tree — no `git checkout` of other branches here; use a
  worktree. All review tooling is **uncommitted** (untracked files on
  `renovation_architecture_rework`).
- Rebuilding the queue (`scripts/build_review_queue.py`) is idempotent for existing cards
  but will pick up any production listings analysed after 2026-08-26.

## 7. Pointers

`docs/DESIGN_review_method.md` (method of record + §5 corrections) ·
`docs/HANDOFF_review_method_design.md` (original commission: situation, Q1–Q8, data
inventory, §2 priorities verbatim) ·
`docs/DESIGN_renovation_architecture_decision_packets.md` (§9 decision log to fill; §2
variance levers) · `docs/HANDOFF_output_quality_manual_review.md` (findings log; §7
thumbnail finding) · `docs/HANDOFF_thumbnail_photo_ingest_fix.md` (queued pipeline fix) ·
`docs/DECISION_renovation_architecture_session9_cutover_20260821.md` (window rules; what
gates FE adoption) · memory notes `review-program-tooling-built`,
`review-method-design-handoff`, `thumbnail-evidence-photos-defect`,
`steven-priorities-hallucinations-over-pricing`.
