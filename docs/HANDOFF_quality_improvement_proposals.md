# Handoff — review the completed quality evidence and the v4→v5 architecture change, then propose quality improvements

Written 2026-08-26 · backend `renovation_architecture_rework` (working tree `df35f23` + untracked files — see §6) · review inputs frozen (hashes in §2)

**Purpose.** A fresh session is to (1) review the completed manual-review analysis
deliverables, (2) understand the current renovation architecture (v5) and how it differs
from the previous Pass-2f architecture (v4), and (3) propose changes to the system to
improve output quality. "Quality" is deliberately left open — the session decides what it
means, which evidence matters, and what to propose. **This document is context only: file
locations, semantics, mechanics, and constraints. It contains no findings, no numbers, no
interpretation of the review results, and no candidate proposals — those are the new
session's to derive and produce.** The measured results live in the reports listed in §2;
read them directly.

## 1. The decision frame on record (Steven's, not this author's)

- Evaluation frame (2026-08-21): Steven is not a renovator; prices are provisional and
  were hidden during review. The v5 rework is judged on **(1) fewer hallucinations** and
  **(2) more useful observations that build packages**. Dollar deflation from dedup is
  accepted. Price calibration is a separate future thread that needs a renovator.
- Review-method semantics (2026-08-22, `docs/DESIGN_review_method.md` — the semantic
  authority): the human judged *the claim against the photo* with neutral labels; only
  Terra is compared at condition level; **nothing about Pass 2f correctness may be derived
  from a condition verdict** (Terra is condition-level, 2f is package-level; direction A/B
  are sampling strata, not a contest); package warrant is judged only on P1 package cards.
- The analysis report's §12 lists further bounded/prohibited inferences (Terra recall,
  overall Sol accuracy, batch causality, pricing, standalone-observation usefulness).
  Proposals must not rest on inferences that list rules out.

## 2. The deliverables to review (all descriptive; decisions were left open)

Frozen inputs (do not modify; rebuilds absorb new production listings):

- `reports/review_queue.json` — 212 cards, sha256 `8512b87c2af1…`
- `reports/review_verdicts.jsonl` — 238 records, sha256 `0c7ca6a8b55d…`

Analysis outputs (2026-08-26, independently verified by a six-agent adversarial
recomputation with zero material findings):

- **`reports/review_analysis.md`** — the primary report. §0 scope/framing, §1 freeze +
  integrity audit (incl. the 11 orphaned verdicts, excluded from every metric), §2
  population/completion, §3 accepted-condition truth (weighted, per source, never
  pooled), §4 error concentration by subgroup (+ p6 forced-single table), §5 dirB
  (Terra-rejected) recovery, §6 Terra replica stability, §7 P1 package warrant, §8
  packaging flow, §9 package-gate exposure, §10 multi-bathroom billing, §11 qualitative
  themes with full note text, §12 what the analysis does not show, Appendix A
  reconciliation. Every rate carries n / denominator / unique listings / evidence-strength
  flags; read the flags before leaning on any slice.
- `reports/review_analysis.json` — machine-readable mirror (raw floats; per-card appendix;
  full subgroup tables including rows the md filters).
- Generator + tests: `scripts/review_analysis.py`, `tests/test_review_analysis.py`
  (run with `.venv\Scripts\python.exe -m pytest tests\test_review_analysis.py -q`).
- `reports/review_tally.md` — the simpler tally of record (its §5 pooled subgroup tables
  are raw diagnostics, not population estimates).
- `reports/session9_decision_worksheet.filled.json` — neutral-label fill of a COPY of the
  Session 9 worksheet (the audited original is never written).
- `docs/DESIGN_renovation_architecture_decision_packets.md` **§9 decision log** — the
  P1 / P2 / P3 / P6(3)-revisit rows now carry the recorded evidence; the option calls are
  open. §§2–8 of that doc also hold the previously-drafted option menus per packet (P1
  options a–d, P3, P4, P6) — the new session may adopt, extend, or discard them.
- `docs/HANDOFF_output_quality_manual_review.md` — findings log (§4, entries 1–3) and the
  index of other open review tasks (§5).
- Method + provenance: `docs/DESIGN_review_method.md` (authority),
  `docs/HANDOFF_review_verdict_analysis.md` (the analysis commission; its §3 counts are
  superseded by the report's §1), `docs/HANDOFF_review_method_design.md` (original
  commission; §8 holds questions Q1–Q8 the review was built to answer).

## 3. The current architecture (v5) — where it lives

Production backend has run v5 ("new" mode) since the 2026-08-21 ~23:20 CT flip.
`RENOVATION_ARCHITECTURE_MODE` selects the mode (`tools/renovation_architecture/runtime.py`,
consumed via `tools/pipeline_config.py`, `tools/analyzer_cli.py`, `tools/analyzer_server.py`,
`tools/artifact_writers.py`).

`tools/renovation_architecture/` (the v5 package):

- `contracts.py` — schemas, ids, prompt versions (`TERRA_REVIEW_PROMPT_VERSION =
  terra_condition_review_v1` :73, `SOL_REVIEW_PROMPT_VERSION = sol_package_review_v1`
  :54), `SHADOW_DEBUG_KEY`, provenance block.
- `catalog_projection.py` / `conditions.py` / `evidence.py` — catalog projection,
  observed-condition construction, evidence facts (photo keys, dedup,
  representative-photo selection, `min_photo_evidence_required`).
- `terra_review.py` + `review_pipeline.py` — **Terra**: one verdict per *condition*
  (`supported | unsupported | cannot_assess`), batched calls (conditions + images per
  call recorded on `terra_calls[]`), claim text from the catalog `atomic_claim`.
- `disposition.py` — accepted_for_work / excluded / inspection routing ("accepted" =
  Terra `supported` + disposition `accepted_for_work` — what v5 bills).
- `work_items.py` — conditions → work items (billing lines).
- `package_candidates.py` — deterministic candidate construction (drivers / supports /
  children).
- `sol_review.py` — **Sol**: one decision per *package candidate* (approve / reject /
  combine / split), bounded review.
- `reconciliation.py`, `usage_guard.py`, `checkpoints.py`, `validators.py` — totals
  reconciliation/ledger, token-budget choke-point guard + telemetry, replay checkpoints.

Artifact shape: production writes `renovation_estimate_v5` at the `photo_intel.json` root
(debug copy in `photo_intel_debug.json`); the Session 9 canary ran shadow mode
(`analysis_debug.renovation_estimate_v5` in `photo_intel_debug.json`). The `result` lists
(`observed_conditions`, `evidence_facts`, `condition_reviews`, `condition_dispositions`,
`terra_calls`, `work_items`, `package_candidates`, `package_decisions`,
`package_applications`, `coverage_ledger`, `totals`, …) are condition-id-joined;
`tools/review_cards.py` (`v5_result`, `index_result`, `join_2f`) is a working example of
reading both placements.

Design intent: `docs/Revised_Renovation_Scope_and_Estimate_Architecture.docx` (note: a
LibreOffice lock file `.~lock…#` sits next to it — it may be open on the desktop).
Build history: session handoffs `docs/HANDOFF_renovation_architecture_session_{1..9}*.md`
and commits `f808e48` (S1) → `e00698d` (S2) → `893f09c` (S3) → `f33c35e` (S4) → S5/S6 →
`b77b388` (property_cost_factor fix) → `07ee112` (S8) → `6640ebb`/`df35f23` (S9 + cutover
conditions). Cutover record: `docs/DECISION_renovation_architecture_session9_cutover_20260821.md`.

## 4. The previous architecture (v4 / Pass 2f) — where it lives

Still what the **frontend renders today**; v5 adoption by the FE is the open decision the
P1–P4 packets gate.

- `tools/renovation_estimate_v4.py` + `tools/rehab_packages.py` — the v4 estimate and
  package construction; **Pass 2f** is the package-level visual verification: one VLM call
  per package over `review_photo_keys`, returning confirmed/rejected member *issue ids*
  (`packages[].confirmed_issue_ids` / `rejected_issue_ids`, `raw_pass_2f_response`,
  `pass_2f_review_audit`, `pass_2f_trace` in the artifact). Issue-level verdicts fold onto
  v5 conditions only for sampling purposes (`tools/review_cards.py:join_2f`).
- Bathroom multiplicity in v4: `bathroom_room_count_signal` + `bathroom_expansion_audit`
  (expansion of a source bathroom package across qualifying surrogates). v5 has no
  equivalent expansion — how each version shapes bathroom billing is visible in the
  artifacts and in report §10's per-card columns.
- Standing facts about 2f state (recorded in memory/docs, relevant to any proposal that
  touches it): evidence-dedup + confirmed-issue revival landed 2026-07-27 gated on 2f
  prompt version v2 (inert until 2F re-runs; bumping the prompt version again silently
  disables it); a per-item/per-photo premium 2f pass was planned — the dormant per-item
  Pass 2f scaffolding is the landing pad and should not be deleted as a side effect.
- Estimate-side context that predates the rework and still applies: the
  `min_photo_evidence` gate (4 items gated at 2 photos, policy `estimate_guard_v2`),
  `requires_2f_for_estimate` defaults False by measurement, electrical trade is
  product-quarantined.

## 5. Change mechanics — what a proposal costs (facts, not preferences)

- **Offline-replayable on the stored canary (no new freeze):** package-formation rules,
  Sol single-child/split rules (P4), P3(b)-style expansion rules, P1(d)-style gates —
  anything that recomputes from stored artifacts. Replay tooling:
  `scripts/replay_renovation_architecture.py`; canary root
  `artifacts_canary/renovation_session9_20260818` (run_1/run_2, 18 listings,
  `input_freeze.json` sha `f32f6812…4a973`).
- **Requires live model calls → fresh freeze + a new two-day canary:** any Terra or Sol
  prompt/rubric change, batching change, or evidence-selection change that alters what the
  model sees (`docs/RUNBOOK_renovation_architecture_session_6.md` §1a). A Terra prompt
  change invalidates the existing canary deltas.
- **Quota ceilings** (`docs/HANDOFF_renovation_architecture_token_budget.md`,
  `docs/analysis/session8_terra_budget_denominator.md`): Terra ~2.5M tokens/day free
  quota shared with upstream passes (~10–12 listings/day all-service), Sol ~250k/day
  (~29 listings/day); a canary replica ≈ 3.98M tokens > one free day. Budget guard is ON.
  Note the tension recorded in the findings-log index: smaller Terra batches re-send
  images and raise the binding token cost.
- **Observation window:** 7 calendar days from 2026-08-22 (closes end of 2026-08-28) — no
  changes to estimate behaviour, prompts, or catalog while it runs. Check the date; if the
  window is still open, proposals are written, not shipped.
- **No Terra "re-decide" mode exists** (re-asking Terra costs quota; `terra_flip` was the
  free stability proxy). Do not assume one.
- The thumbnail-ingest defect has an already-scoped separate fix thread:
  `docs/HANDOFF_thumbnail_photo_ingest_fix.md` (start 2026-08-29+).

## 6. Working rules

- Everything review- and analysis-related is **untracked/uncommitted** on
  `renovation_architecture_rework`; the only tracked-file edits pending are
  `docs/DECISION_…cutover_20260821.md` (pre-existing) and
  `docs/DESIGN_renovation_architecture_decision_packets.md` (§9 evidence rows).
- `RV_ROOT` is the live production working tree — no `git checkout` of other branches
  here; use a worktree.
- Artifacts are read-only. Do not modify `scripts/build_session9_decision_packets.py` or
  any `reports/session9_*` output (audited, byte-reproducible). Do not rebuild
  `reports/review_queue.json` (a rebuild absorbs listings analysed after 2026-08-26).
- Tests: `.venv\Scripts\python.exe -m pytest` (bare `python` does not resolve the venv).
  Review tooling suites: `tests/test_review_cards.py` (14) +
  `tests/test_review_analysis.py` (19).
- Artifact gate: `scripts/verify_renovation_artifact.py --artifact <run dir>
  --expect-mode new --expect-terra-model gpt-5.6-terra --expect-sol-model gpt-5.6-sol`.
- Production artifacts: `C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts/<key>/<run>/`
  (post-cutover = run dirs ≥ `20260821_230000`); FE repo = `renointel-prod`; addresses in
  its `prisma/dev.db` (read-only `mode=ro`).

## 7. Suggested reading order (procedural, not a steer)

1. `docs/DESIGN_review_method.md` — what each verdict and stratum means.
2. `reports/review_analysis.md` end to end (the JSON for anything the md filters).
3. `docs/DESIGN_renovation_architecture_decision_packets.md` §§1–9 — architecture
   context, existing option menus, and the now-filled evidence rows.
4. The architecture docx + `tools/renovation_architecture/` source; then
   `tools/rehab_packages.py` / `renovation_estimate_v4.py` for the 2f side.
5. Form your own view of what "improve quality" should mean against §1's recorded frame,
   and produce proposals with their change-cost class (§5) attached.

The deliverable format is the new session's choice; Steven asked that it come to its own
conclusions and make its own calls rather than inherit any from prior sessions.
