# Handoff: package-outcome benchmark results — pipeline scope gaps to plan around

Findings only, for a planning session. 2026-08-12. No remedies proposed here.

## What ran

The package-outcome benchmark (tools/benchmark_pass2a_packages.py, uncommitted,
worktree C:\Users\Steven\PycharmProjects\rv-pass2a-bench branch
pass2a_prompt_bench) asks: do the photos produce the human-defined renovation
packages and material work? Human gold: benchmarks/pass2a-prompt/gold/
package_reference.json (sha a2ba0165, authored by Steven 2026-08-11 — Frisby St
from a live photo walkthrough, Carroll St derived from the frozen 112-condition
observation gold with Steven verifying tiers). 18 cells = {baseline_cap25,
checklist_cap25, checklist_all_retained} × 3 reps × 2 properties. Real Pass 2f
ran on gpt-5.6-sol: 123 VLM calls, all complete and saved (paid artifacts under
benchmarks/pass2a-prompt/runs/package_2f/{cell}/rep{n}/{prop}/ — candidates,
verifications, pass_2f_trace, final_estimate per cell). Verification totals:
96 confirmed / 20 rejected / 7 uncertain.

## Status: verdicts BLOCKED on 5 pending review rows

runs/package_review/review.csv has 5 unmatched approved outcomes (all Carroll
St packages; 4 are repair↔modernization family-swaps vs the gold, 1 add-on).
Steven has not adjudicated them. Until decisions are imported
(package-review import), scores.json reports status=blocked and no cap
verdicts. Preview shape regardless of those 5: **every rep of every cell is
incomplete** — the misses below dominate, so the cap experiment is heading
for both_wrong everywhere unless the underlying issues or the gold change.

## The three finding classes (the things to plan fixes for)

1. **Uniform upstream scope misses — not a cap phenomenon.** Core material
   items are missing 3/3 reps in ALL THREE cells (baseline, capped checklist,
   uncapped checklist): countertop_damage@kitchen_primary,
   cabinets_damaged_or_water_stained@kitchen_primary,
   appliances_worn_or_neglected@kitchen_primary,
   tile_or_grout_damage@bathroom_primary, everything exterior
   (patio_or_porch_surface_wear, soffit_or_porch_ceiling_weathered,
   damaged_or_rotted_siding_or_trim), ceiling_cracks_or_sagging@bedroom_1
   (Carroll), popcorn_or_acoustic_ceiling_texture@bedroom_1 (Frisby).
   Concrete example: Carroll St's kitchen (missing cabinet door fronts,
   makeshift counter, older worn range — all in the frozen observation gold)
   produces kitchen packages whose evidence is ONLY flooring/paint/drywall/trim
   ids in every rep of every cell. No cabinet, countertop, appliance, or
   backsplash id ever enters any package or line item. Since the observation
   gold proves the conditions were *observed* at 2a-level in at least some
   variants, the loss is somewhere in 2c classification → 2d resolution →
   evidence assembly — the composed 2c×2d error the saturated
   catalog-resolution benchmark never measured (docs/STATE_kind_ontology_program.md).
   Full missing list: runs/package_eval/scores.json →
   missing_for_gold_reconsideration (36 distinct) and per-rep
   outcomes[cell][prop].reps[n].missing.

2. **Pass 2f rejections and demotions of real scope.** 20 of 123 verifications
   rejected. Exterior packages almost never survive 2f on either property
   (candidates exist pre-2f; final_estimate lacks them). Sol demoted Carroll's
   approved kitchen_modernization to pricing_profile kitchen_refresh — two
   tiers below the gold's kitchen_full_rehab — because its ≤3 review photos
   only confirmed cosmetic evidence (retier-from-confirmed-evidence,
   rehab_packages.py). 2f's max_images=3 cap on review photos is the binding
   constraint here.

3. **Catalog-id cousins.** Some "missing" work items exist under sibling ids:
   gold vanity_countertop_dated@bathroom_primary vs pipeline evidence
   vanity_worn_finish; gold worn/older flooring ids vs pipeline
   hard_flooring_scratched_or_worn. Work-item matching is id-exact by design;
   the review 'equivalent' mechanism only covers extras (unmatched approved
   outcomes), not evidence-level cousins. This is the catalog-ID-quality
   problem explicitly deferred at benchmark inception.

Also observed (context, not necessarily a problem): with real 2f, only ~3
valid line items survive per property (vs 19 force-confirmed pre-2f) because
reviewed_issue_ids projection invalidates unconfirmed package members —
scoring already credits absorbed evidence via package supporting_catalog_item_ids
(verified un-pruned pre→post 2f).

## Where everything lives

- Harness: tools/benchmark_pass2a_packages.py (scoring: score_package_round;
  work-item satisfaction incl. absorbed-evidence rule), tools/benchmark_pass2a.py
  (CLI). Tests: tests/test_benchmark_pass2a_packages.py (35 green).
- Scores: benchmarks/pass2a-prompt/runs/package_eval/scores.json (blocked).
  Dry run (pre-2f extras scale + cap diff): runs/package_eval/dry_run_report.{json,md}.
- Review: runs/package_review/review.csv (5 rows, undecided).
- Paid 2f artifacts: runs/package_2f/** (fingerprinted Layer F — do NOT edit
  config package_eval.pass_2f or catalog/embeddings/Qwen settings, that
  orphans 123 Sol calls; gold edits + review imports invalidate scoring only).
- Gold: benchmarks/pass2a-prompt/gold/package_reference.json (+ template with
  vocabulary). Observation gold (proves conditions were observable):
  gold/reference.json.
- Tail resolutions (paid local-Qwen, Layer R): runs/package_tail_resolution/**;
  rebuilt cells: runs/package_cells/**.
- Prior context: docs/HANDOFF_pass2a_micro_observation_findings.md (why this
  benchmark exists), docs/STATE_kind_ontology_program.md.

## Pipeline code + trace data the issues live in (investigation surface)

Finding 1 (scope misses) — trace an observation end-to-end using the frozen
per-photo checkpoints at runs/variant_{baseline,checklist}/rep{n}/{prop}/.photos/{photo}.json:
scene_data.observations_struct.observations = 2b atomic claims;
scene_data.observations (kept, with kind + issue_id) vs
scene_data.excluded_observations (+ reason) = 2c's verdict;
scene_data.resolved_items (+ debug.pass_2d_per_observation, skipped_reason,
candidates with scores) = 2d's verdict; tail rows for beyond-cap observations
in runs/package_tail_resolution/. Property-level "resolver returned nothing"
rollup: photo_intel_debug.json → analysis_debug.unmapped_issues. Code:
- tools/scene_classifier_passes.py — run_pass_2c (~:892), run_pass_2d (:1109,
  incl. lexical shortcut gates PASS_2D_SHORTCUT_MIN_SCORE/MARGIN in
  tools/pipeline_config.py:180), run_pass_2e (:1434).
- tools/scene_classifier_orchestrator.py — resolve_observation_against_catalog
  (:140, exact-kind retrieval + kind-purity), max_resolve_per_image cap (:910).
- tools/catalog_embeddings.py — retrieval, kind/scene-group pools, per-item
  deny_any/require_any guardrails applied post-topk (:526) — a guardrail or
  scene-group filter silently shrinking candidate pools is a candidate cause.
- tools/issue_catalog_kind_v2.json — THE catalog: item ids (id-cousin issue),
  scene_groups (living needs "living_areas" not "living" — known gotcha),
  package_affinity blocks (which item feeds which package in which role —
  rehab_packages.py:669-750). A cabinet id that resolves but lacks/mismatches
  package_affinity for the room would price standalone or vanish from
  packages: check BOTH resolution and affinity.

Finding 2 (2f rejections/demotions) — tools/rehab_packages.py:
_select_review_photo_keys (:1366) + PACKAGE_REVIEW_IMAGE_LIMIT=3 (:1363),
prepare_pass_2f_cases (:3813) + select_package_review_image_paths (:3695),
_filter_evidence_items_for_review (:3720) → reviewed_issue_ids,
_project_package_issue_ids_to_confirmed / _retier_package_from_confirmed_evidence
(:2320/:2350, the tier demotion), apply_package_verifications_to_candidates
(:3603, revival gate :3663 requires prompt_template_version pass_2f_package_v2).
2f prompt/schema: tools/scene_classifier_passes.py:1610-1889 + run_pass_2f
(:1954). Raw Sol responses per package: saved in
runs/package_2f/**/verifications.json (raw_response field) and evidence
summaries in final_estimate.json packages.

Finding 3 (id-cousins) — the catalog item vocabulary itself plus 2d candidate
lists (debug.pass_2d_per_observation shows the top candidates the resolver
chose between, with scores — check whether the gold's id was even in the pool).

## Constraints on any fix plan

- Production runs the same 25-cap, same 2c/2d, same 2f max_images=3 — these
  findings describe production behavior, not benchmark artifacts.
- Electrical is product_quarantined (deliberate; separate parked tangent).
- Changing 2c/2d/2f code or prompts and re-measuring requires regenerating the
  affected layers: tail resolution is $0 (local Qwen), 2f re-run is ~123 Sol
  calls (~a day-plus of Steven's token budget, resumable via --budget N).
- Gold is Steven-authored and closed-world; the 5 pending family-swap reviews
  and the id-cousin cases may partly be gold-vocabulary choices rather than
  pipeline errors — adjudication pending.

## To finish the current round (independent of fixes)

Steven adjudicates 5 rows → package-review import --csv ... → report.
Commands run from the worktree with the main repo venv
(C:\Users\Steven\PycharmProjects\realtorvision-backend\.venv\Scripts\python.exe
tools/benchmark_pass2a.py ...).
