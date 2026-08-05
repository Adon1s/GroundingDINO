# Handoff: single source of truth for package routing (catalog-driven affinity)

## Status / framing

Package routing — "which issue, seen in which room, feeds which rehab package, in which role" — currently lives in **two places**:

1. **Flat catalog fields** on 30 items in [tools/issue_catalog.json](../tools/issue_catalog.json): `package_type`, `package_role`, `package_category`, `room` (5 kitchen items, 24 bathroom items, 1 living item — `dated_fireplace_surround`).
2. **Hardcoded Python**: the `PACKAGE_AFFINITY` dict ([tools/rehab_packages.py:595](../tools/rehab_packages.py#L595)) with explicit bedroom/living entries, plus three generic-support tuples (`_GENERIC_KITCHEN_SUPPORTS`, `_GENERIC_BATHROOM_SUPPORTS`, `_GENERIC_BEDROOM_LIVING_GAP_SUPPORTS`, [tools/rehab_packages.py:785-825](../tools/rehab_packages.py#L785)) registered via `_register_generic_support` ([tools/rehab_packages.py:828](../tools/rehab_packages.py#L828)).

The code itself calls the catalog path "a fallback for kitchen/bathroom compatibility during the transition" (`infer_package_candidates` docstring). This handoff **finishes that transition in the affinity direction**: all routing becomes data in the catalog, keyed by room; the flat fields and the Python tables are removed. Scene-conditional routing (same issue id → different package per room) is strictly more expressive than the flat fields, and every flat-field item is trivially expressible as an affinity entry with a single room.

**This is a refactor with parity as the acceptance bar.** Estimate output on existing artifacts must be identical before/after (one narrow, documented exception — see "Scene-missing fallback" below).

## Background — how routing works today

`infer_package_candidates` ([tools/rehab_packages.py:1962](../tools/rehab_packages.py#L1962)) resolves each candidate's room via `_candidate_scene_room` ([tools/rehab_packages.py:884](../tools/rehab_packages.py#L884)) → `_normalize_scene_to_room` ([tools/rehab_packages.py:208](../tools/rehab_packages.py#L208)), then overlays `PACKAGE_AFFINITY[(room, issue_id)]` onto the catalog item via `_catalog_item_with_package_affinity` ([tools/rehab_packages.py:898](../tools/rehab_packages.py#L898)). If no affinity entry matches, the **raw catalog item's flat fields** decide eligibility (`is_package_eligible_catalog_item`, [tools/rehab_packages.py:915](../tools/rehab_packages.py#L915)). A scene-mismatch guard ([tools/rehab_packages.py:2009-2015](../tools/rehab_packages.py#L2009)) then skips flat-field candidates whose observed scene contradicts the package room.

The same dual check appears in `_extract_package_only_candidates` ([tools/renovation_estimate_v4.py:316-323](../tools/renovation_estimate_v4.py#L316)): `is_package_eligible_catalog_item(cat) or package_affinity_for(scene_group, cat_id) is not None`.

Derivation maps `_PACKAGE_TYPE_TO_CATEGORY` ([tools/rehab_packages.py:559](../tools/rehab_packages.py#L559)) and `_PACKAGE_TYPE_TO_ROOM` ([tools/rehab_packages.py:575](../tools/rehab_packages.py#L575)) already make `package_category` and `room` fully derivable from `package_type`. **These maps stay in code** — they are vocabulary, not routing.

## The change

### New catalog format

Add an optional per-item `package_affinity` object to issue_catalog.json, keyed by **package room** (`kitchen | bathroom | bedroom | living` — the `VALID_ROOMS` vocabulary, NOT scene_groups tokens like `living_areas`). Each entry carries only the two non-derivable fields:

```json
"package_affinity": {
  "bedroom":  {"package_type": "bedroom_modernization",  "package_role": "package_driver"},
  "living":   {"package_type": "living_modernization",   "package_role": "package_driver"},
  "kitchen":  {"package_type": "kitchen_modernization",  "package_role": "package_support"}
}
```

(Example above is `worn_or_stained_carpet`: drives bedroom/living modernization, supports kitchen modernization — exactly its current routing.)

`package_category` and `room` are **not stored**; they derive from `package_type` and the affinity key at load time. Items with no `package_affinity` block never route to a package (today's `package_role: "ignore"` / absent behavior).

### Loader

At catalog load (a small `build_package_affinity(issue_catalog) -> Dict[Tuple[str, str], Dict[str, str]]` in rehab_packages.py), flatten the per-item blocks into the same `{(room, issue_id): {package_type, package_role, package_category, room}}` shape the runtime already consumes, filling `package_category`/`room` from the derivation maps. Validate strictly while flattening:

- room key ∈ `VALID_ROOMS`
- `package_type` ∈ `VALID_PACKAGE_TYPES` and `_PACKAGE_TYPE_TO_ROOM[package_type] == room key` (a bedroom entry must reference a bedroom package)
- `package_role` ∈ `{package_driver, package_support}`

Raise on violation — do not silently skip; this is the single source of truth now.

`infer_package_candidates` and `package_affinity_for` ([tools/rehab_packages.py:876](../tools/rehab_packages.py#L876)) should take/use the loaded table instead of the module constant. `infer_package_candidates` already receives `issue_catalog`; build the table there (or accept a prebuilt one). `package_affinity_for` is also called from renovation_estimate_v4.py with no catalog in scope — thread the table through (it's called inside `_extract_package_only_candidates`, which has `issue_catalog`).

### Migration of existing data (write a one-off script)

Do NOT hand-author the JSON. Write a one-off script (e.g. `scripts/migrate_package_affinity.py`) that:

1. Imports the current `PACKAGE_AFFINITY` (post-`_register_generic_support` registration) and emits per-item blocks from it.
2. For each of the 30 flat-field items, emits `{item.room: {package_type, package_role}}` from the flat fields. None of these ids collide with a PACKAGE_AFFINITY key for the same room today, but assert that anyway.
3. Writes the updated catalog, then **asserts round-trip equality**: `build_package_affinity(new_catalog)` == old in-code dict (including derived category/room), AND for each of the 30 items the derived (type, role, category, room) equals the old flat fields.

Keep the script committed; delete-or-keep at the user's discretion afterward.

### Deletions (after parity is proven)

- `PACKAGE_AFFINITY` literal, `_MODERNIZATION_PACKAGE_BY_ROOM`, the three `_GENERIC_*_SUPPORTS` tuples, `_register_generic_support`, and the registration loops ([tools/rehab_packages.py:595-845](../tools/rehab_packages.py#L595)).
- Flat `package_type`, `package_role`, `package_category`, `room` fields from all 30 catalog items.
- The flat-field fallback leg: `is_package_eligible_catalog_item` raw-item checks in `_extract_package_only_candidates` collapse to the affinity check alone. `is_package_eligible_catalog_item` itself stays but is now only meaningful on **overlay-applied effective items** inside `infer_package_candidates` (the overlay still writes `package_type`/`package_role` keys into the effective dict — that mechanism is unchanged).
- The accessors `catalog_package_type/role/category` ([tools/rehab_packages.py:542-855](../tools/rehab_packages.py#L542)) stay (they read effective/overlay dicts and package dicts); `catalog_room`'s scene_groups fallback ([tools/rehab_packages.py:867-873](../tools/rehab_packages.py#L867)) stays for Pass 2f room resolution.

### Scene-missing fallback (the one behavior decision)

Today a flat-field bathroom item routes to its package even when the candidate has **no resolvable scene** (no surrogate match, empty `scene_groups_seen`) — the affinity overlay no-ops and the flat fields win. After migration that candidate would route nowhere. Preserve parity: when `_candidate_scene_room` returns `None` and the item's `package_affinity` block has **exactly one room key**, use that entry. Multi-room blocks with no scene stay unrouted (today only generics are multi-room, and generics already required scene resolution). Implement this inside the overlay/lookup so both call sites get it. Record the fallback in the effective item (e.g. `_package_affinity_scene: "single_room_fallback:bathroom"`) for auditability.

## Where to change

| File | Change |
|---|---|
| [tools/issue_catalog.json](../tools/issue_catalog.json) | Add `package_affinity` blocks (script-generated); strip flat `package_type`/`package_role`/`package_category`/`room` from the 30 items |
| [tools/rehab_packages.py](../tools/rehab_packages.py) | Add `build_package_affinity` loader + validation; rewire `package_affinity_for` / `_catalog_item_with_package_affinity` / `infer_package_candidates`; single-room scene-missing fallback; delete Python routing tables |
| [tools/renovation_estimate_v4.py](../tools/renovation_estimate_v4.py) | `_extract_package_only_candidates` eligibility becomes affinity-table lookup only |
| [scripts/audit_issue_catalog.py](../scripts/audit_issue_catalog.py) | `PACKAGE_FIELDS` ([line 44](../scripts/audit_issue_catalog.py#L44)) and the report columns ([lines 280-292, 419-420, 504-505](../scripts/audit_issue_catalog.py#L280)) must read the new `package_affinity` block (report one row per (item, room) entry, or join rooms into one cell — implementer's choice) |
| `scripts/migrate_package_affinity.py` | New one-off migration + round-trip assertion |

## Tests

Run with `.venv\Scripts\python.exe -m pytest` (bare `python`/`py` don't resolve the venv). Scope to touched files: `tests/test_rehab_packages.py tests/test_package_taxonomy.py tests/test_renovation_estimate_v4.py tests/test_package_pass_2f_and_vlm.py`. Note: **7 tests fail on this branch pre-existing and unrelated** — establish the baseline failure set before your change and diff against it; don't chase failures you didn't cause. `tests/` is untracked on this branch — never `git stash` test paths.

- **Parity test (the keystone):** a test that loads the migrated catalog and asserts `build_package_affinity(catalog)` equals a snapshot of the old in-code `PACKAGE_AFFINITY` + flat-field-derived entries (take the snapshot before deleting the constants, commit it as a fixture).
- Catalog metadata coverage guard (`test_real_catalog_has_kitchen_metadata_coverage`, [tests/test_rehab_packages.py:730](../tests/test_rehab_packages.py#L730)) — update to validate `package_affinity` blocks (room keys, type/role enums, type↔room consistency) and to assert the flat fields are **gone** (prevents regression/re-introduction).
- Existing tests call `package_affinity_for(scene, issue_id)` directly ([tests/test_rehab_packages.py:2338, 2369-2373](../tests/test_rehab_packages.py#L2338)). If you thread the loaded table through as a parameter, give it a default (lazily built from the shipped catalog) or update these call sites — don't let the signature change silently break the alias-normalization tests.
- `test_package_taxonomy.py` catalog-driven tests (`test_every_package_eligible_kitchen_item_has_category`, [tests/test_package_taxonomy.py:362](../tests/test_package_taxonomy.py#L362), and `test_kitchen_modernization_items_resolve_to_modernization_category`, [line 381](../tests/test_package_taxonomy.py#L381)) — rewrite against derived metadata.
- New unit tests: single-room scene-missing fallback fires for a 1-room block; does NOT fire for a multi-room block; loader raises on bad room key / type-room mismatch / bad role.
- End-to-end parity: if a stored v4 estimate artifact + fixture inputs exist in the test suite, re-run and diff `packages` + `suppressed_package_candidates`; otherwise rely on the existing `infer_package_candidates` behavior tests in test_rehab_packages.py passing unchanged.

## Gotchas

- **Affinity room keys are package rooms, not scene groups.** `living`, not `living_areas`; `_normalize_scene_to_room` does the collapsing at lookup time. Separately, if you touch any living-area item's `scene_groups`, it must say `["living_areas"]` (not `["living"]`) or retrieval drops the item entirely.
- The three structural/feature generics (`unfinished_interior_wall_osb_exposed`, `stained_glass_or_vintage_light_fixture`, `layout_modernization_opportunity`) are **intentionally unrouted** — they must NOT gain `package_affinity` blocks (see comment at [tools/rehab_packages.py:767-776](../tools/rehab_packages.py#L767)).
- `hallway`/`stairway` exclusion (`_SCENE_ROOM_EXCLUSIONS`, [tools/rehab_packages.py:205](../tools/rehab_packages.py#L205)) is scene-normalization logic, not routing — leave it in code, untouched.
- `interior_paint_flooring_refresh` is derived downstream (whole-home aggregate), never inferred — no catalog item should reference it, and the guard at [tools/rehab_packages.py:2007](../tools/rehab_packages.py#L2007) stays.
- The scene-mismatch guard ([tools/rehab_packages.py:2009-2015](../tools/rehab_packages.py#L2009)) becomes mostly redundant once routing is scene-keyed, but it still protects the single-room fallback path — keep it.
- `_STRONG_SIGNAL_CATALOG_IDS`, `compute_package_strength`, ambient-support demotion, and absorption scopes are **out of scope** — routing only. Same for the Pass 2f scaffolding (a per-item premium verification revival is planned; don't remove anything around it).
- `_effective_candidate_package_role` ([tools/rehab_packages.py:1031](../tools/rehab_packages.py#L1031)) prefers `candidate_catalog_meta` (effective items) and falls back to `candidate.package_role` — unchanged, but verify it never reads raw catalog flat fields after the strip.
- docs that describe the flat fields ([docs/HANDOFF_rehab_packages_architecture.md](HANDOFF_rehab_packages_architecture.md), [docs/frontend_contract_packages.md](frontend_contract_packages.md)) — add a short "superseded by package_affinity" note where they describe catalog routing fields; the frontend contract is about estimate output (package dicts keep `package_type` etc.) and should need no functional change.

## Out of scope (separate handoffs exist/planned)

- Same-unit package subsumption (modernization vs repair vs turnover stacking).
- Region/size cost factors; `estimate_tier: "low"` coercion; `estimate.group: "paint"` cap vocabulary; catalog JSON-Schema validator. Don't fix these en passant even if you notice them.
