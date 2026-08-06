# Handoff — Task 4A → 4C: Legacy Retirement

4C deletes the two-kind compatibility surface. **4C cannot start until 4B is
accepted** (historical migration done and approved) and until every item below
is proven reader-free. Deleting earlier than that breaks stored-artifact reads
or the rollback path.

Status inputs (update before starting 4C):

- 7-day observation outcome: **pending** — canary not yet run at handoff time;
  this document must be updated with the observation result before 4C begins.
- 4B acceptance: **pending**.

## Deletion ledger — every remaining legacy surface

Backend (realtorvision-backend):

- `tools/costing.py` — `KIND_MULT["upgrade"]` (kept solely for scoring/
  reprojecting legacy_v1 artifacts; deletable when no v1 artifact is ever
  rescored).
- `tools/pipeline_config.py` — the `legacy_v1` selector branch, the
  `ISSUE_CATALOG_PATH` override path, and eventually the selector itself once
  v2 is the only behavior.
- `tools/issue_catalog.json` — the v1 catalog (still read by the FE for v1
  artifacts and by the legacy selector).
- `tools/catalog_validation.py` — `LEGACY_CATALOG_KINDS` vocabulary branch,
  `blocked_pending_pricing` enum value, and `inherited_from_split_parent`
  (superseded when the pricing project authors real successor prices — only
  then).
- `tools/publication_gate.py` — the legacy (non-v2) validation branch.
- `tools/observation_kinds.py` — `LEGACY_CATALOG_KINDS`.
- `tools/pipeline_common.py` — `LEGACY_ONTOLOGY_VERSION` handling stays until
  no stored artifact lacks a v2 stamp (i.e., after 4B completes AND archives).
- Legacy summary fallbacks: `property_summary_pass._norm_kind` unknown→defect
  coercion; frozen v1 strings (`COST_MODEL_SOURCE_DERIVED_UPGRADE_ROOM_ALLOWANCE`,
  `defect_upgrade_disagreements`) — rename only with a migration story.
- Blocked harnesses (still `RuntimeError` at `main()`): `tools/catalog_auditor.py`
  (`MATCH_THRESHOLD_DEFECT/UPGRADE` :107-108, two-kind ternary :289-290,
  `_label_to_kind` :282), `tools/model_comparison.py` (`_label_to_kind` :259,
  forward-set filters :410-414/:1060, judge prompts :770-832, kind default
  :1466), `tools/bias_check.py` (:428) — migrate to three-kind or delete.
- Legacy benchmark baseline: `benchmarks/catalog-resolution-v2/legacy_baseline/`
  (frozen v1 routing snapshot — archival candidate, not silent deletion).
- The pinned legacy worktree `../rv-legacy-a1972cf` (rollback boundary; remove
  only after 4C ships and no rollback to two-kind publication is possible).
- Untracked 4B prototype `tools/backfill_kind_v2.py` — deleted or promoted by
  4B; must not survive 4C as dead code.

Frontend (renointel-prod):

- `lib/analysis/kindOntology.ts` — the legacy branch of
  `readArtifactOntologyVersion`, the `upgrade` member of `ObservationKind`,
  and the legacy pair branch of `normalizeSummaryCounts`.
- `lib/artifactReader.ts` — v1 catalog path + `loadCatalog()` legacy default;
  `interior_paint_drywall → paint_drywall` remap.
- Types: `defect_count`/`upgrade_count` on `SummaryV1Bucket`/`SummaryV1Listing`;
  `catalog.defects[]`/`upgrades[]` shapes in `IssueCatalog`.
- `speculative_upgrade` suppression label (FiltersRail/ActiveFilterChips).
- Tests fixtures built on the legacy pair (`tests/property/conditionSummary.test.ts`).

## Order of operations

1. 4B accepted; historical migration inventory approved by Steven.
2. Proof of no v1 artifacts in active use: corpus scan showing every served
   artifact carries `ontology_version=observation-kind-v2` (or is archived
   out of the serving root); DB/completion contract consistent.
3. Archive before delete: v1 catalog, migration manifest + audit, legacy
   baseline snapshot, and a corpus backup manifest (paths + sha256) recorded.
4. Remove compatibility in dependency order: FE legacy branches first (they
   only read), then backend readers (`LEGACY_*` constants, tolerant reads),
   then writers/selector, then the v1 catalog file, then the harnesses.
5. Rollback boundary: after step 4 begins, rollback to two-kind publication is
   no longer supported; the pinned worktree covers only pre-4C incidents.
6. Deletion tests: suite green with the v1 catalog file removed; grep-clean for
   `upgrade` as an observation kind (excluding history/docs); FE build green
   with two-kind types removed; publication gate rejects any legacy_v1 payload.

Pricing redesign stays a separate project unless it has independently completed
before 4C; do not couple deletions to it.
