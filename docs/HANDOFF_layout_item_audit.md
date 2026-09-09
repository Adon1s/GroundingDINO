# HANDOFF — Layout-priced catalog item audit (pre-cutover blocker)

2026-08-08. The Task 4A canary delta review
(`reports/kind_cutover_delta_review_20260808.md`) found that
`bathroom_layout_modernization_opportunity` — a v2 catalog item — prices
*layout* observations ("tight layout", fixture arrangement) at $5,000–$25,000.
It was newly added on 4 canary properties ($100k combined high-end) and is the
largest single driver of the unapproved headline-delta breaches. Steven's
ruling: layout claims must not price. This is a catalog defect (contradicts
the Pass 2c v3 boundary rubric and "the catalog refuses non-renovation
concepts"); the v3 prompt starving it upstream is mitigation, not a fix.

## Task

1. **Audit** the v2 catalog (v2 catalog file — see `ISSUE_CATALOG_PATH` /
   Task 2 handoff `docs/HANDOFF_kind_ontology_task2.md`; v3.0 ids in
   `tools/catalog_migrations/2.1_to_3.0.json`) for the whole family:
   layout / arrangement / spatial "opportunity"-shaped priced items, not just
   the bathroom one (check for kitchen/bedroom siblings).
2. **Identify affected canary properties**: which of the 18 in
   `artifacts_canary/candidate/*/*/photo_intel.json` resolved any flagged item
   (search `renovation_estimate_v4.groups[].line_items[].catalog_item_id` and
   `issues_flat[].catalog_item_id`). Expect ~4–6.
3. **Deprecate** the flagged items (remove from pricing; follow the existing
   catalog migration/deprecation pattern in `tools/catalog_migrations/`).
4. **Re-run the candidate canary side for affected properties only**
   (`scripts/run_kind_canary.py --side candidate`), re-run
   `tools/compare_kind_cutover.py`, and update the delta review sheet.

## Context / constraints

- Branches: `pass_2c_redesign` @ a8558ef = canary-validated v2 cutover build
  (deploy target — the catalog fix lands here and re-measures under it).
  `pass_2c_kind_v3` = post-cutover prompt v3 + 2e Gate 5 (do NOT ship with
  cutover).
- Do not touch the 2c prompts on either branch; this is catalog-only.
- Stamp the new catalog version in the sign-off record
  (`configs/kind_ontology_cutover.json`) so approval provenance is clean.
- Non-affected properties' deltas stand as measured; Steven reviews all deltas
  after the re-measure (sheet regenerates via the same line-item comparison).
- Split-successor stacking (worn+dated both pricing, pattern 2 in the review
  sheet) is a SEPARATE post-cutover costing decision — out of scope here.
