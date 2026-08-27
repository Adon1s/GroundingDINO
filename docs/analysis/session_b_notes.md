# Session B — session notes (2026-08-27)

Charter: `docs/HANDOFF_quality_session_B_harness_tooling.md`. All four
deliverables closed + Session C handoff written
(`docs/HANDOFF_quality_session_C_package_policy_1.md`). No live provider
calls were made; no estimate behaviour changed; frozen inputs untouched.

## Tracked outside the close condition

- **Packet P5 item 6 — RESOLVED 2026-08-27 (Steven's call: correct the
  audit record).** Erratum landed in
  `docs/DESIGN_renovation_architecture_decision_packets.md` §7 item 6: the
  "duplicated v4 rows" in the two headline explanations (125779232,
  80877597) were per-surrogate bathroom expansion clones — distinct
  bathrooms, i.e. the P3/QP4 under-billing, not redundancy. The frozen
  review file stays byte-identical (sha256 is an integrity anchor in the
  session9 handoffs; a post-P5 comparator re-run yields a different,
  unreviewed item set, so the file-edit option died with the P5 keying
  fix). Both properties re-filed as QP4 gate evidence — pointer added to
  the Session C handoff for Session E.

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
