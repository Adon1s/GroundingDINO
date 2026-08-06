# Handoff — Task 4A → 4B: Historical-Artifact Migration ("active reanalysis")

4B brings the stored corpus (~1,264 `photo_intel.json` under
`renointel-prod/artifacts/`, all `catalog_version: "2.1"`, ontology legacy_v1)
into the observation-kind-v2 world. 4A migrated **nothing**: stored v1
artifacts are immutable and the FE reads them through a version-aware adapter.

## State shipped by 4A (branch pass_2c_redesign)

| Commit | What |
|---|---|
| 352160c | v2 catalog publishable; 42 split successors inherit parent economics verbatim (`pricing_status: inherited_from_split_parent`); KIND_MULT degradation=1.0 / modernization=0.6 (temporary bridge), upgrade=0.6 retained for legacy reads |
| 846407d | `KIND_ONTOLOGY_VERSION=legacy_v1\|observation_kind_v2` selector (default legacy_v1) derives catalog + pipeline depth; `pipeline_mode=publish` runs 2c→2d→2e with `classification_only=False` |
| 8c302d5 | `tools/publication_gate.py` — permanent write-boundary validation (stamps, ids, kinds, no mixed payloads); root `ontology_version` stamp (v1 writes stamp `legacy_v1` explicitly) |
| 9c5990d | cutover config (18-property canary, 7-day relative window), comparator tracked, `compute_scoring`/`extract_estimate_candidates` fail loud on stale resolved ids |
| renointel-prod `411312e6` (branch kind_ontology_v2_compat) | FE version-aware adapter (`lib/analysis/kindOntology.ts`), per-artifact catalog selection, shape-discriminated summary counts |

Canary/observation status at handoff time: **manifest proposed, pending
Steven's approval** (`configs/kind_ontology_canary_manifest.json`). Update this
line when the canary report and the 7-day observation outcome exist.

Temporary pricing behavior 4B must not disturb: successor economics are v1
inheritances, not authored prices; the dedicated pricing project replaces them.

## Corpus inventory (refresh before running)

Survey of 2026-08-06: 849 property dirs, 840 with runs, 821 with both a
completed run and images on disk (`renointel-prod/public/images/properties/<key>/`).
Projection status mix: 888 reprojected / 349 native / 27 needs_reanalysis.
NOTE: stored `photos[*].photo.image_path` values point at the dead
`IntelliJProjects\realtorvision\...` path — resolve images via the current
repo name, not the stored absolute path.

## The three migration paths (per artifact, per issue)

1. **Deterministic same-id reprojection** — 88 of 107 legacy ids are
   unchanged/reclassified/narrowed: same id, possibly new kind. Metadata-only
   restamp + derived-view recompute. No model calls.
2. **Stored-description re-resolution for splits** — issues resolving to the
   19 deprecated split parents (manifest `2.1_to_3.0.json`, `deprecated: true`)
   must be re-resolved to one of their 42 successors from the stored
   description. Deterministic; ambiguity fails closed.
3. **Image reanalysis, only where descriptions are insufficient** — emit the
   exact photo set needing reanalysis; never reanalyze wholesale.

Manifest `requires_re_resolution` (71 of 107 true) is the per-id worklist.

## The untracked prototype: `tools/backfill_kind_v2.py`

Deliberately excluded from 4A commits. What it already does well: the three
paths above, per-issue `support_any` scoring with deterministic tie-rejection,
`--resolution-map` manual overrides, dry-run default, atomic backup-backed
writes, nulls derived fields + stamps `kind_migration` provenance, chains into
`reproject_product_views.reproject_artifact`.

Known limitations to fix before trusting it:

- **Reprojection chain is the risky half.** It calls the current
  `reproject_product_views` which recomputes scoring + summary with current
  code; on a migrated artifact this is correct only under the v2 catalog —
  never run it under `legacy_v1`. It does NOT recompute
  `renovation_estimate_v4` (it nulls it); recomputation without replaying
  stored `package_verifications` zeroes all packages (the not_run gate — same
  hazard as `backfill_reno_v4`, see memory/offline-v4-recompute-needs-2f-replay).
  4B must define how estimates come back: 2F replay from stored
  verifications, or explicit `needs_reanalysis`.
- No checkpoint/resume for a 1,264-artifact walk (per-artifact atomicity only);
  no idempotency marker beyond skip-if-current; no batch audit report contract;
  no DB coordination (renointel-prod `AnalysisRun` rows / completion contract
  are untouched by file edits).
- Its `_stamp_issue` rewrites issues in **all** lanes including
  `photos.*.issues.removed` — decide whether removed/suppressed lanes should be
  migrated or left as historical record.
- The removed 4A draft tests (below) are its only test surface; restore and
  extend them in 4B.

### Removed draft tests (verbatim reference)

The 4A test rework deleted these from `tests/test_kind_ontology_cutover.py`;
they exercised the prototype and should seed 4B's suite: 
`test_backfill_dry_run_is_non_mutating_and_resolves_splits`,
`test_backfill_apply_stamps_provenance_and_clears_stale_views`,
`test_backfill_fails_closed_when_split_description_is_insufficient`,
`test_split_resolver_rejects_ties`, plus the `_legacy_artifact` fixture
(splits `damaged_soffit_or_porch_ceiling` → `soffit_or_porch_ceiling_failed`
by description support). Recover them from git history of this branch
(`git log --all -- tests/test_kind_ontology_cutover.py`, pre-9c5990d working
tree) or rewrite from the prototype's docstrings.

## 4B requirements

- **Idempotent + resumable**: checkpoint file, deterministic ordering,
  re-running never double-migrates (skip-if-current already exists; keep it).
- **Fail closed**: any unresolved split → `needs_reanalysis` with the exact
  photo list; no partial stamps.
- **Audit output**: one report per batch (counts by status, per-artifact
  decisions, unresolved reasons) committed or archived.
- **DB/artifact coordination**: decide how migrated artifacts interact with
  the frontend completion contract and any cached rows; the FE reads files at
  request time, so file-level migration is visible immediately — coordinate
  the reprojection + estimate story before flipping anything.
- **Frontend dependency**: adapter (renointel-prod `411312e6`) must be
  deployed first — it already is a 4A release precondition. Migrated
  artifacts render via the v2 path (catalog selection by stamp).
- **Electrical-quarantine legacy reprojection** (pending since 2026-07-12) is
  unblocked by the FE adapter: `reproject_product_views` may now emit
  `kind_counts` summaries for legacy artifacts. Fold it into 4B's reprojection
  pass rather than running it separately.

## 4B acceptance gates

- Zero artifacts left carrying a deprecated split-parent id.
- Every migrated artifact passes `tools/publication_gate.validate_publication_payload`
  against the v2 catalog.
- `needs_reanalysis` set is enumerated, bounded, and either reanalyzed or
  explicitly parked with reasons.
- Spot-check sample (per stratum of the 4A canary manifest) reviewed by Steven.
- Full backend suite green; FE renders migrated artifacts.

Out of scope for 4B: the pricing redesign (multipliers + successor prices stay
as the 4A bridge) and all legacy deletion (4C).
