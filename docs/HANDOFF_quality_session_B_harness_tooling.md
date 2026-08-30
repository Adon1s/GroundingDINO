# Handoff — Session B: re-decide harness + review tooling (window-safe)

Written 2026-08-26 · backend `renovation_architecture_rework` · commissioned by
`docs/ROADMAP_quality_program_sessions_20260826.md` (Session B) · proposals =
`docs/PROPOSALS_output_quality_improvements_20260826.md` (QP1, QP8, QP9/P5,
QP7-instrumentation). **This session builds and measures; it ships no
estimate-behaviour change and makes NO live provider calls** (live harness
runs belong to Session F). *Update 2026-08-27: the observation window closed
early (`DECISION_…cutover_20260821.md` §10), so the in-window timing caveats
below no longer bind — but B's charter is unchanged: build + dry-run only.*
Inherit the roadmap's hygiene + simplicity guardrail blocks in full.

## 1. Deliverable 1 — the re-decide harness (QP1)

A **new sibling script** `scripts/redecide_renovation_architecture.py` plus a
scoring script. Do NOT add live calls to
`scripts/replay_renovation_architecture.py` — its documented invariant is that
it imports only deterministic modules and is structurally incapable of
spending provider tokens.

What it does (per listing, canary run_1 root
`artifacts_canary/renovation_session9_20260818`):

1. Load the stored v5 envelope (`analysis_debug.renovation_estimate_v5` in
   `photo_intel_debug.json` — shadow placement; `tools/review_cards.py:v5_result`
   is the reference reader for both placements).
2. Rebuild each estimate unit's Terra request from stored
   `observed_conditions` + `evidence_facts`, reusing
   `tools/renovation_architecture/terra_review.py:build_unit_request` /
   `parse_unit_reviews` and the payload dict at `terra_review.py:143-150`.
   **Integrity check first:** recompute the request fingerprint
   (`terra_review.py:166-179`, includes per-photo `exact_sha256`) and compare
   to the stored `terra_calls[].request_fingerprint` — a mismatch means photo
   bytes or payload drifted (images live in the mutable renointel-prod tree);
   refuse that unit rather than measure garbage.
3. Apply a **variant override** (harness-side): candidate system-prompt text
   and/or extra payload keys (e.g. `verification_guidance`), or payload
   ablations (the remove-`observations` arm). Every variant gets its own
   fingerprint namespace / label — the system-prompt text is NOT part of the
   stored fingerprint (only `TERRA_REVIEW_PROMPT_VERSION` is), so the harness
   must never collide with production checkpoints: **never read or write the
   canary properties' `.checkpoints` dirs** (same-fingerprint reuse would
   silently return stored verdicts and measure nothing).
4. Call Terra live under the budget guard: reserve → call → settle via the
   shared ledger (`tools/renovation_architecture/usage_guard.py`; follow the
   `review_pipeline.py:86-125` TerraUsageLedger + `external_reservation()`
   pattern so nothing double-debits). Write results to a harness-owned output
   root (e.g. `artifacts_canary/redecide_<variant>_<date>/`), never into the
   frozen canary.

Scoring script: join re-decided verdicts to `reports/review_verdicts.jsonl`
(latest-wins per card; the 11 orphaned verdicts are excluded — mirror
`scripts/review_analysis.py`'s loader). Scoreable population = the **125
canary condition cards** (32 dirA, 36 dirB, 40 uniform, 13 flip-only, 4 p6).
Score per variant, split by class:

- 11 hard-false billed cards → flip to unsupported = **win**
- 55 human-supported billed cards → leave supported = required (each loss
  counts against the variant)
- 6 overstated-but-real billed cards → deletion is a **cost**, not a win
  (right fix is wording, not exclusion)
- 16 dirB human-supported cards → recovery to supported = goal-2 **win**

Noise floor: Terra replica flips ran 6.4% (16/250) and humans matched
run_1/run_2 8:6 — so the harness must support a **same-prompt control arm**
and report every variant against it. Cost ~0.9M Terra tokens per full
18-listing pass (mean ~51k/listing); a variant-vs-control comparison ≈ 1.8M ≈
one free Terra day. (Session B only *builds and tests* this — a dry-run mode
that stops before the provider call is the in-window test path.)

Design the variant/label/ledger scaffolding so Session F can also plug the
**Sol** consistency probe into it later (text-only listing calls,
`sol_review.py` path) — but do not build the Sol probe now.

## 2. Deliverable 2 — P5 review-tooling fixes (packet §7, items 1–5)

1. `tools/compare_renovation_architecture_cutover.py:179-185` — `_v4_packages`
   keys by `type|unit` and silently overwrites duplicates; key by
   `package_id` so expanded bathroom packages survive. Fix the v5 side at
   `:211` symmetrically (benign today, load-bearing once QP4 lands).
2. Add effective low/high dollars to v4/v5 package review rows (21 of 78
   Tier-1 "identical package" rows differed in price, invisible to review).
3. Add v4 2f `verification_status` (+ evidence summary) to package review
   items.
4. Add a Sol flip detector (identical children, different decision) to the
   stability item.
5. `scripts/cluster_session9_reviews.py` (`:92`, `:382`) — stop folding
   multi-surrogate packages into `package_key_migration`; add a
   "disagreement" tier.

Track packet item 6 (the two mis-worded headline explanations) in the session
notes; it is outside the close condition. Do NOT re-run or modify the audited
`reports/session9_*` outputs.

## 3. Deliverable 3 — commit the review tooling (QP7 instrumentation)

`tools/review_cards.py`, `scripts/build_review_queue.py`,
`scripts/review_server.py`, `scripts/review_tally.py`,
`scripts/review_analysis.py`, `tests/test_review_cards.py` (14),
`tests/test_review_analysis.py` (19) are all untracked working-tree code —
commit them as-is (suite green first). Do not "improve" them in passing; the
frozen reports they produced are audited.

Commit the re-tag tooling in the same pass: `scripts/build_retag_queue.py`,
`scripts/retag_tally.py`, `reports/retag_queue.json` (generated 2026-08-26,
verified 26+20 cards, masses reconcile with `review_analysis.md` §9) and
`reports/retag_verdicts.jsonl` if Steven has started answering — see
`docs/HANDOFF_retag_mechanism_vs_perception.md`. It reuses
`scripts/review_server.py` unchanged via `--queue` / `--verdicts`, so nothing
about the main review path moves.

## 4. Deliverable 4 — two read-only audits (short notes, e.g. `docs/analysis/`)

- **QP8 inspection-lane visibility:** v5 routes `cannot_assess` → `inspection`
  (69 canary dispositions); the inspection totals lane is contractually 0/0
  (`contracts.py:157-161`) and inspection conditions never become work. The v4
  split-dollar contract has a `needs_inspection` lane the FE renders. Answer:
  does anything the FE/user sees consume v5's inspection conditions? (FE repo
  = `C:\Users\Steven\IntelliJProjects\renointel-prod`, read-only.) Output: a
  short note routed to the FE-contract task list.
- **Scene-identity mismatch count (QP9 amendment):** across stored canary +
  post-cutover production artifacts, count how often a package candidate's
  evidence photos carry scene assignments conflicting with the package room
  (the P05 "indoor photo in an exterior package" mode; also the utility-room
  bathroom surrogate). One-off script, counts only — NOT a new pipeline
  stage; a rule is proposed later only if the volume warrants it.

## 5. Constraints (beyond the roadmap blocks)

- No live provider calls this session, period. No estimate-behaviour change
  ships. Artifacts are read-only.
- The frozen review inputs and `reports/session9_*` are untouchable; new
  outputs go to new paths.
- Production root filtering: post-cutover run dirs ≥ `20260821_230000`
  (`tools/review_cards.py:34` PROD_ROOT pattern is the reference).
- Two production listings (redfin_10965375, redfin_10922002) are
  thumbnail-based end to end — exclude them from any measurement.

## 5a. Repo/git gotchas (added 2026-08-26)

- The repo root is littered with untracked scratch (`bias_check_*.json`,
  `catalog_audit_*.json`, `artifacts/`, `.codex_tmp/`, `.asset/`, …). Never
  `git add .` / `git add -A` — stage the named files explicitly.
- **Recommended, Steven to confirm at session start:** also commit the frozen
  evidence files — `reports/review_queue.json`, `reports/review_verdicts.jsonl`,
  `reports/review_analysis.md`/`.json`, `reports/review_tally.md`. They are the
  ground truth the whole program scores against and currently exist only as
  untracked working-tree files; committing them protects them. (Committed
  as-is, never regenerated — a queue rebuild absorbs new listings.)
- Harness model config: resolve the Terra model exactly as production does
  (`RENOVATION_TERRA_MODEL` → `OPENAI_MODEL` fallback; canary + production ran
  `gpt-5.6-terra`; `scripts/verify_renovation_artifact.py` prints the routing)
  so re-decided verdicts come from the same model the stored verdicts did —
  a different model would make every comparison meaningless.

## 6. Exit criteria

1. Harness + scoring committed with tests (dry-run mode exercised; no tokens
   spent); a short control-arm design note (how many cards decide, at what
   threshold, given the 6.4% floor).
2. P5 items 1–5 landed with tests; comparator output shows package_id-keyed
   v4 packages, dollars, 2f verdicts, and a Sol-flip section on a scratch
   re-run (scratch path, not the audited reports).
3. Review tooling committed; suite green
   (`.venv\Scripts\python.exe -m pytest`).
4. The two audit notes written.
5. **Session C's handoff written** (`docs/HANDOFF_quality_session_C_package_policy_1.md`):
   QP3 interior-modernization application gate + QP6 single-child rule with
   the blanket-vs-scoped choice presented to Steven, code sites from the
   proposals doc, per-change replay + comparator validation plan, post-window
   ship timing.
