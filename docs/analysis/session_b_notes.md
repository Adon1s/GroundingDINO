# Session B — session notes (2026-08-27)

Charter: `docs/HANDOFF_quality_session_B_harness_tooling.md`. All four
deliverables closed + Session C handoff written
(`docs/HANDOFF_quality_session_C_package_policy_1.md`). No live provider
calls were made; no estimate behaviour changed; frozen inputs untouched.

## Tracked outside the close condition

- **Packet P5 item 6** (still open, deliberately not done this session):
  two headline explanations in the audited review record (properties
  125779232 and 80877597) describe the bathroom expansion as "duplicated v4
  rows"; the correct description is v4's per-surrogate expansion clones.
  Fixing the wording in the review file changes its hash and needs a
  deterministic offline comparator re-run; fixing only the audit record
  does not. Owner: Steven's call on which record to correct, any wave-1
  session can carry it.

## Handoff corrections found by fact-check (do not propagate)

- `tools/review_cards.py:34` is `PROD_ROOT` (a path); the post-cutover
  filter is `DEFAULT_SINCE` (:38) + `RUN_DIR_RE` (:40) applied in
  `iter_runs`.
- The canary's 18 listing dirs live under `run_N/candidate/`, not directly
  under the canary root.
- The "69 canary inspection dispositions" figure is run_1 (31) + run_2 (38)
  combined.

## Session F inheritance

- Full 18-listing dry-run control pass at
  `artifacts_canary/redecide_control_20260827/`: 148/148 units
  fingerprint-VERIFIED — the rebuildable-set record to check against before
  any live pass (photos are mutable; the thumbnail-ingest thread starts
  2026-08-29 and may rewrite images).
- Variant mechanics smoke at
  `artifacts_canary/redecide_remove_observations_v1_20260827/` (2 units,
  payload ablated, distinct fingerprint namespace).
- Decision thresholds: `docs/analysis/session_b_control_arm_design.md`.
