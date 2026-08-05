# Handoff: catalog `support_scope` tagging (Option 2 — refinement of ambient-support demotion)

## Status / framing

This is **Option 2** of a three-part fix for "a recurring cross-room finding mints multiple rehab packages." **Option 1 already shipped** (ambient-support demotion in `tools/rehab_packages.py`). This handoff is an **optional override/refinement layered on top of Option 1 — not a replacement for it.** The runtime recurrence heuristic stays the primary, general mechanism that needs no catalog curation; the catalog tag only adjusts it at the edges.

Do not redesign the demotion. Only add a catalog-driven override.

## Background (what Option 1 does today)

`infer_package_candidates` ([tools/rehab_packages.py:1952](../tools/rehab_packages.py#L1952)) tallies, per `catalog_item_id`, the number of distinct estimate units where it appears in **support** role. Ids appearing in `>= _AMBIENT_SUPPORT_MIN_UNITS` (currently 3, [tools/rehab_packages.py](../tools/rehab_packages.py) constant) distinct units become `ambient_support_ids`. Ambient supports are excluded from the **driverless** emit gate (`elif len(non_ambient_supports) >= 2`) but still corroborate driver-anchored packages and are still costed. Scope is **package-existence only** — `compute_package_strength` is untouched.

Two limitations this handoff addresses:
1. A support that is *inherently* a property-wide trait (popcorn, dated trim) only gets demoted once it has actually been detected in ≥3 rooms. Detected in 2 rooms, it still mints packages.
2. A support that is *genuinely room-scoped* (a real per-room object) is demoted if it happens to recur in ≥3 rooms, even though each instance is independent work.

## The change

Add an optional catalog field `support_scope: "ambient" | "room"` to items in [tools/issue_catalog.json](../tools/issue_catalog.json). Default (absent) = `"room"` with **no behavioral effect** — i.e. recurrence-only, exactly as today.

Two override directions:

- **`"room"` — the primary use.** Force-*excludes* a support from recurrence demotion. Even if the id recurs across ≥N units, it is never treated as ambient, so genuinely per-room supports keep minting their per-room packages. This is the safety valve for false positives from the runtime heuristic.
- **`"ambient"` — secondary.** Demotes a known property-wide trait even below the N threshold (e.g. detected in only 2 rooms). Candidate ids: `popcorn_or_acoustic_ceiling_texture`, `dated_interior_trim`, `dated_interior_doors`, `dated_wallpaper_present`, `dated_wood_paneling`, `baseboard_wear_scuffs`, `wall_scuffs_marks_or_dents`. (These are catalog `package_role: "ignore"` items promoted to support per-room by the affinity layer — see `_register_generic_support`, [tools/rehab_packages.py:827](../tools/rehab_packages.py#L827).)

### Predicate after this refinement

Replace the ambient membership check (today: `cat_id in ambient_support_ids`) with a per-candidate predicate threaded through `infer_package_candidates`:

```
is_ambient(cat_id) =
    support_scope(cat_id) == "ambient"
    OR (support_scope(cat_id) != "room" AND cat_id in ambient_support_ids)
```

i.e. `"ambient"` always demotes, `"room"` never demotes, unset falls back to the recurrence tally. Read `support_scope` off the catalog item (a `catalog_support_scope(catalog_item)` accessor mirroring `catalog_package_role`, [tools/rehab_packages.py:541](../tools/rehab_packages.py#L541), keeps it tidy and validated against `{"ambient", "room"}`).

## Where to change

- `tools/rehab_packages.py`: add `catalog_support_scope()` accessor; in `infer_package_candidates`, compute the ambient set using the predicate above instead of the bare `ambient_support_ids` membership (both at the driverless gate and the `non_ambient_supports`/observability lines).
- `tools/issue_catalog.json`: add `support_scope` to the ambient cosmetic ids listed above (and to any genuinely-per-room support you want to protect).

## Tests

- Extend the catalog metadata-coverage guard ([tests/test_rehab_packages.py:607](../tests/test_rehab_packages.py#L607)) to assert `support_scope`, when present, is one of `{"ambient", "room"}`.
- New unit tests in `TestAmbientSupportDemotion` ([tests/test_rehab_packages.py](../tests/test_rehab_packages.py)):
  - `support_scope: "ambient"` demotes at 2 units (below N).
  - `support_scope: "room"` is never demoted even at ≥3 units.
  - unset still follows the N-unit recurrence rule (regression of current behavior).

## Gotchas

- Living-area catalog items must use `scene_groups: ["living_areas"]` (not `["living"]`) or retrieval drops them — applies if you add/edit any living items here.
- Keep the change "package-existence only" (don't touch `compute_package_strength`) unless the user revisits the scope decision.
- `_AMBIENT_SUPPORT_MIN_UNITS` is the shared knob; this refinement does not change it.
