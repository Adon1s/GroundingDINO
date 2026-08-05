# Handoff: Bathroom catalog second-pass (gap analysis vs kitchen)

## Why

The first bathroom catalog overhaul (`HANDOFF_bathroom_catalog_overhaul.md`) brought the bathroom catalog from 6 items to 21 — close to kitchen's 24. That work focused on **defect drivers** (vanity, tile, tub/shower, plumbing, electrical, moisture) and a couple of upgrade items (`dated_or_worn_vanity_countertop`, `dated_bathroom_flooring_style`).

The gap that remains is in the **upgrade / modernization / opportunity** space — cosmetic / dated-finish items that kitchen has but bathroom doesn't yet. Without these, `_resolve_bathroom_modernization_profile` has fewer component classes to work with, so the resolver routes more aggressively to `bathroom_refresh` and may miss `bathroom_partial_rehab` opportunities for properties with broad cosmetic-dated bathroom signals.

This handoff is a focused second pass: identify kitchen→bathroom upgrade-kind analogues that should exist, and add them.

## Current catalog state (as of 2026-05-21)

- **Kitchen-tagged**: 24 items (room=kitchen OR package_type starts with `kitchen_`)
- **Bathroom-tagged**: 21 items
- Audit script: `py -c "import json; data=json.load(open('tools/issue_catalog.json')); items=data.get('items') if isinstance(data, dict) else data; print('K:', len([i for i in items if (i.get('package_type') or '').startswith('kitchen_')]), 'B:', len([i for i in items if (i.get('package_type') or '').startswith('bathroom_')]))"`

## Gap analysis: kitchen items without a bathroom analogue

Run this exact mapping table against the catalog before adding anything — some "missing" items may already be covered by a cross-room generic.

| Kitchen item | Status | Suggested bathroom analogue | Priority |
|---|---|---|---|
| `paint_refresh_recommended` (sev 1, upgrade, support) | **GAP** | `bathroom_paint_refresh_recommended` — bathroom-specific cosmetic paint upgrade signal. Distinct from `peeling_or_damaged_bathroom_paint` (defect-kind). | Medium |
| `dated_wallpaper_present` (sev 2, upgrade, support) | **GAP** | `dated_bathroom_wallpaper` — common signal in dated bathrooms (floral/border wallpaper, vinyl wallpaper). | High |
| `dated_wood_paneling` (sev 2, upgrade, support) | **GAP** | `dated_bathroom_wood_paneling` — less common (most bathrooms don't have paneling) but happens in older builds. | Low |
| `layout_modernization_opportunity` (sev 3, upgrade, support) | **GAP** | `bathroom_layout_modernization_opportunity` — awkward layouts (corner toilet, narrow wedge, no vanity counter, etc.). | Medium |
| `stained_glass_or_vintage_light_fixture` (sev 1, upgrade, support) | Partial coverage | Already covered by `dated_bathroom_vanity_light` (sev 1, upgrade, support) in most cases. Skip unless the team wants a separate "vintage style" signal distinct from "dated vanity light". | Low / skip |
| `outdated_or_damaged_cabinets` (sev 2, defect, driver) | ✓ Covered | `outdated_or_damaged_vanity` | — |
| `countertop_damage` (sev 2, defect, support) | ✓ Covered | `vanity_countertop_damage` | — |
| `older_flooring_style` (sev 2, upgrade, support) | ✓ Covered | `dated_bathroom_flooring_style` | — |
| `missing_base_cabinets_exposed_subfloor` (sev 4, defect, driver) | ✓ Covered | `missing_vanity_exposed_plumbing` + `missing_bathroom_tile_exposed_substrate` | — |
| `dated_lighting_fixtures` (sev 1, upgrade, support) | ✓ Covered | `dated_bathroom_vanity_light` | — |
| `outdated_kitchen_finishes` (sev 3, upgrade, driver) | ✓ Covered | `outdated_bathroom_finishes` | — |

**Cross-room items — DO NOT tag with `room: bathroom`** (per `HANDOFF_bathroom_catalog_overhaul.md:105-116`). These intentionally span multiple rooms; tagging them as bathroom-specific would regress kitchen/bedroom flows:

- `damaged_drywall_or_cracks`
- `worn_or_stained_carpet` (N/A in bathrooms anyway)
- `baseboard_wear_scuffs`
- `wall_scuffs_marks_or_dents`
- `popcorn_or_acoustic_ceiling_texture`
- `unfinished_interior_wall_osb_exposed`
- `dated_interior_trim`
- `dated_interior_doors`
- `worn_or_stained_flooring`
- `worn_or_stained_vinyl_linoleum` (covered for bathroom via `worn_or_damaged_bathroom_flooring` defect-kind)
- `scratched_or_damaged_flooring`
- `appliance_damage_or_missing` (N/A in bathrooms)

If a cross-room signal is significantly different in bathrooms (e.g., moisture-related peeling paint near a tub vs general kitchen peeling), create a NEW bathroom-specific item rather than retagging.

## Items to add (3 high/medium priority)

Add each at the bottom of `tools/issue_catalog.json` mirroring the shape of `dated_or_worn_vanity_countertop`. Pricing baselines come from `BATHROOM_*` constants in `tools/rehab_packages.py` and from `outdated_bathroom_finishes`' cost block:

### 1. `dated_bathroom_wallpaper` (Priority: HIGH)

Common defect signal in older bathrooms (pre-2000s wallpaper, floral/vinyl borders). Drives `bathroom_modernization` packages alongside `outdated_bathroom_finishes`.

```json
{
  "name": "Dated Bathroom Wallpaper",
  "category": "opportunity",
  "severity": 2,
  "trade_bucket": "paint_drywall",
  "scope": "replace",
  "kind": "upgrade",
  "id": "dated_bathroom_wallpaper",
  "defaultHidden": false,
  "scene_groups": ["bathroom"],
  "require_any": ["wallpaper", "border", "pattern", "vinyl wall"],
  "deny_any": [],
  "embed_text": "Dated bathroom wallpaper. Floral, geometric, or vinyl-print wallpaper covering bathroom walls. Wallpaper borders along ceiling or above tile. Peeling or curling wallpaper edges. Vintage 80s/90s wallpaper patterns. Wallpaper removal and bathroom wall paint refresh.",
  "tier": "work",
  "drop_if_generic": false,
  "cost": {
    "mode": "allowance",
    "base_low": 400,
    "base_high": 2000,
    "per_occurrence_low": 360,
    "per_occurrence_high": 900,
    "cap_low": 2500,
    "cap_high": 6000,
    "cost_source": "manual"
  },
  "estimate": {
    "estimate_tier": "low",
    "strategy": "replace_only",
    "group": "paint",
    "stack_behavior": "group_cap",
    "unit_policy": "per_bathroom"
  },
  "display_class": "marketability",
  "package_role": "package_support",
  "package_type": "bathroom_modernization",
  "package_category": "modernization",
  "room": "bathroom"
}
```

### 2. `bathroom_layout_modernization_opportunity` (Priority: MEDIUM)

Awkward bathroom layouts (corner toilets, narrow wedges, no vanity counter, fixture clearance issues) that signal a layout-rework opportunity. Mirrors `layout_modernization_opportunity` for kitchen.

```json
{
  "name": "Bathroom Layout Modernization Opportunity",
  "category": "opportunity",
  "severity": 3,
  "trade_bucket": "bathroom_fixtures_tile",
  "scope": "replace",
  "kind": "upgrade",
  "id": "bathroom_layout_modernization_opportunity",
  "defaultHidden": false,
  "scene_groups": ["bathroom"],
  "require_any": ["awkward", "narrow", "corner toilet", "no vanity", "small", "tight", "cramped", "wedge"],
  "deny_any": [],
  "embed_text": "Bathroom layout modernization opportunity. Awkward or constrained bathroom layout: corner toilet placement, narrow wedge bathroom, no vanity counter or only a pedestal sink in master, fixture clearance issues, tight shower-toilet sandwich. Layout reconfiguration or full bathroom redesign.",
  "tier": "work",
  "drop_if_generic": false,
  "cost": {
    "mode": "allowance",
    "base_low": 5000,
    "base_high": 25000,
    "per_occurrence_low": 4500,
    "per_occurrence_high": 9000,
    "cap_low": 15000,
    "cap_high": 35000,
    "cost_source": "manual"
  },
  "estimate": {
    "estimate_tier": "high",
    "strategy": "replace_only",
    "group": "bathroom",
    "stack_behavior": "group_cap",
    "unit_policy": "per_bathroom"
  },
  "display_class": "marketability",
  "package_role": "package_support",
  "package_type": "bathroom_modernization",
  "package_category": "modernization",
  "room": "bathroom"
}
```

### 3. `bathroom_paint_refresh_recommended` (Priority: MEDIUM)

Bathroom-specific cosmetic paint refresh signal. Distinct from `peeling_or_damaged_bathroom_paint` (defect, sev 2) — this is an opportunity-kind signal for dated/dull paint without active damage.

```json
{
  "name": "Bathroom Paint Refresh Recommended",
  "category": "opportunity",
  "severity": 1,
  "trade_bucket": "paint_drywall",
  "scope": "cosmetic",
  "kind": "upgrade",
  "id": "bathroom_paint_refresh_recommended",
  "defaultHidden": false,
  "scene_groups": ["bathroom"],
  "require_any": ["paint", "wall", "ceiling"],
  "deny_any": ["peeling", "damage", "stained"],
  "embed_text": "Bathroom paint refresh recommended. Dated paint colors, dull or worn finish, beige or tan walls in need of a contemporary refresh. Wall paint refresh without underlying damage. Bathroom paint update.",
  "tier": "work",
  "drop_if_generic": false,
  "cost": {
    "mode": "allowance",
    "base_low": 200,
    "base_high": 800,
    "per_occurrence_low": 180,
    "per_occurrence_high": 360,
    "cap_low": 1000,
    "cap_high": 2500,
    "cost_source": "manual"
  },
  "estimate": {
    "estimate_tier": "low",
    "strategy": "replace_only",
    "group": "paint",
    "stack_behavior": "group_cap",
    "unit_policy": "per_bathroom"
  },
  "display_class": "marketability",
  "package_role": "package_support",
  "package_type": "bathroom_modernization",
  "package_category": "modernization",
  "room": "bathroom"
}
```

## Items NOT to add (and why)

- **`dated_bathroom_wood_paneling`** — rare in modern listings; the `dated_wood_paneling` cross-room item already covers the few bathrooms that have paneling. Adding a bathroom-specific entry would split signal between two items without enough volume to justify it.
- **`bathroom_vintage_light_fixture`** (mirror of `stained_glass_or_vintage_light_fixture`) — `dated_bathroom_vanity_light` already catches this in practice. Mirror adds noise without lift.
- **`bathroom_appliance_damage_or_missing`** — bathrooms don't have appliances. Kitchen-only concept.
- **`bathroom_worn_or_stained_carpet`** — bathrooms with carpet exist but are rare; the cross-room `worn_or_stained_carpet` covers them adequately.
- **Anything that duplicates existing bathroom items**: `vanity_countertop_damage`, `dated_or_worn_vanity_countertop`, `dated_bathroom_flooring_style` already cover their respective spaces.

## Verification (after additions)

1. **JSON lint**: `py -c "import json; json.load(open('tools/issue_catalog.json'))"`
2. **Test suite**: `py -m pytest tests/test_rehab_packages.py tests/test_renovation_estimate_v4.py tests/test_renovation_estimate.py tests/test_package_taxonomy.py --tb=short -q` — must remain green.
3. **Catalog counts**:
   ```bash
   py -c "import json; data=json.load(open('tools/issue_catalog.json')); items=data.get('items') if isinstance(data, dict) else data; print('K:', len([i for i in items if (i.get('package_type') or '').startswith('kitchen_')]), 'B:', len([i for i in items if (i.get('package_type') or '').startswith('bathroom_')]))"
   ```
   Expected after all 3 additions: K=24, B=24 (parity).
4. **Live audit**: pick a bathroom-heavy property with dated wallpaper or awkward layout. Run `py -m tools.audit_runner --artifacts-root C:\Users\Steven\IntelliJProjects\realtorvision\artifacts --scene-group bathroom --property <id>`. Confirm `photo_intel.json` `renovation_estimate_v4.package_candidates` shows new items in `supporting_catalog_item_ids` when matched.
5. **Resolver routing**: confirm that adding `dated_bathroom_wallpaper` + `outdated_bathroom_finishes` co-fire produces a `bathroom_partial_rehab` or stronger tier rather than `bathroom_refresh`.

## Conventions reminder

- Set `scene_groups: ["bathroom"]` (single-room).
- Set `tier: "work"` (NOT `"optional"` — optional items are filtered by `is_defect_driver` / `is_dated_cosmetic_evidence`).
- Set `trade_bucket: "paint_drywall"` for paint/wallpaper, `bathroom_fixtures_tile` for layout/vanity, `plumbing` / `electrical` / `moisture_mold` for those domains.
- For new upgrade-kind drivers (severity 3 + `package_role: package_driver`), they will automatically elevate package strength via the generic rule in `_has_strong_signal` ([tools/rehab_packages.py:549-572](../tools/rehab_packages.py#L549)) — no `_STRONG_SIGNAL_CATALOG_IDS` edit needed unless you want explicit override behavior.
- `drop_if_generic` MUST be `false` — global kill-switch semantics make `true` unsafe until the scene-conditional follow-up lands (see `HANDOFF_drop_if_generic_scene_conditional.md`).
- Insert order: append at the end of the items array, mirroring the placement of recently-added items.

## Out of scope

- Frontend changes — see `HANDOFF_bathroom_frontend.md`.
- New pricing tiers beyond what's already defined in `BATHROOM_*` constants.
- Pass 2f prompt changes — kitchen + bathroom prompts already shipped.
- Scene-conditional `drop_if_generic` implementation — see its own handoff.
- Catalog audit for new electrical / plumbing heavy items — see comment in `_ELECTRICAL_HEAVY_IDS` (tools/rehab_packages.py:482-490) for when to extend that frozenset.

## Reference

- [tools/issue_catalog.json](../tools/issue_catalog.json) — the catalog
- [tools/rehab_packages.py](../tools/rehab_packages.py) lines 58-65 — `BATHROOM_*` pricing tier constants
- [tools/rehab_packages.py](../tools/rehab_packages.py) `_resolve_bathroom_modernization_profile` (lines 962-997) — what new modernization items feed into
- [tools/rehab_packages.py](../tools/rehab_packages.py) `_has_strong_signal` (lines 549-572) — generic strength elevation rule
- `HANDOFF_bathroom_catalog_overhaul.md` — first-pass overhaul context
- `HANDOFF_drop_if_generic_scene_conditional.md` — why `drop_if_generic` is `false` here
