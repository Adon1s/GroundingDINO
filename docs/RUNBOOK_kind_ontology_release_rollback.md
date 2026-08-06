# Runbook — Kind Ontology v2 Release / Rollback (Task 4A)

## Release order (each step gates the next)

1. **Frontend first.** Merge `kind_ontology_v2_compat` (renointel-prod) and
   deploy/restart the frontend. Verify against **stored v1 artifacts**: one
   property page (counts + badges render), the Issue Explorer, and one
   reprojected artifact (its summary already carries `kind_counts` — the
   adapter is shape-discriminated, so this must render correctly today).
2. **Canary green.** All gates in docs/RUNBOOK_kind_ontology_canary.md passed;
   changed-claim review done; approvals recorded.
3. **Pin rollback.** Legacy build = `a1972cf` (worktree `../rv-legacy-a1972cf`).
   Rollback configuration = unset/`KIND_ONTOLOGY_VERSION=legacy_v1` on the
   current build (publication kill switch), or run the pinned build for true
   legacy publication.
4. **Enable v2.** In the production environment for the backend analyzer
   (CLI/server invocation):
   - `KIND_ONTOLOGY_VERSION=observation_kind_v2`
   - `ISSUE_CATALOG_PATH` must NOT be set (startup fails otherwise, by design).
   - Model map routes 2c explicitly (see
     `benchmarks/configs/kind_canary_model_map.json`); publish mode without a
     2c override logs a loud warning and runs 2c on local Qwen — treat the
     warning as a misconfiguration.
   - Embeddings sidecar healthy (real-POST probe) — Pass 2d needs it.
5. **Smoke the first artifact end to end.** Root stamps
   `ontology_version=observation-kind-v2`, `catalog_version=3.0`,
   `run.kind_ontology_version`, `run.pipeline_mode=publish`; every resolved
   issue has a v2 `catalog_item_id` + kind; estimates/packages/summary_v1
   (`kind_counts`) present; frontend renders the property.

## Observation window — 7 calendar days from enablement

Monitor **operational invariants only** (no cross-property semantic
comparisons):

- publication-gate rejections (`write_photo_intel: refusing to publish`),
- unresolved/stale-id failures (`unknown/stale catalog id`),
- ontology/resolver/writer run failures,
- frontend rendering failures on new artifacts.

## Rollback

Flip `KIND_ONTOLOGY_VERSION=legacy_v1` (or unset) and restart the analyzer.
Published v2 artifacts stay on disk and stay readable through the FE adapter —
no data rollback. The legacy selector on this build is a publication kill
switch (classification-only, nothing publishes); for continued v1 publication
run the pinned `a1972cf` build.

Roll back:

- **immediately** for any published invalid/mixed artifact or reproducible
  frontend schema failure;
- after **three consecutive** ontology/resolver/writer failures.

The window ending does NOT authorize legacy deletion — that is Task 4C, gated
on Task 4B acceptance (docs/HANDOFF_kind_ontology_4A_to_4C_history_retirement.md).
Keep v1 read compatibility after the seven days.
