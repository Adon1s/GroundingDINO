# Handoff: Full bathroom catalog overhaul

This is a self-contained brief for a follow-up session. The backend bathroom-parity *machinery* has landed (per-room Pass 2f prompts, bathroom profile resolvers, bathroom turnover tiers, strong-signal sentinel for `outdated_bathroom_finishes`). What's left is the **catalog work**: bring bathroom items up to kitchen-level breadth so the pipeline has enough variety to detect, gate, and emit bathroom packages reliably.

You'll be editing `C:\Users\Steven\PycharmProjects\realtorvision-backend\tools\issue_catalog.json` and, in a few places, `tools\rehab_packages.py`. **Do not touch** `_infer_bathroom_package` rule logic, `_PACKAGE_ABSORPTION_SCOPES` for existing tiers, Pass 2f prompts, or the frontend — those have all been built.

## Current state (post-machinery work)

The catalog contains exactly 6 bathroom items with full verbose metadata (`display_class`, `package_role`, `package_type`, `package_category`, `room`):

| id | line | display_class | package_role | package_type | package_category |
|---|---|---|---|---|---|
| `tile_or_grout_damage` | ~905 | `estimate_driver` | `package_driver` | `bathroom_repair` | `repair` |
| `stained_or_damaged_bath_fixtures` | ~1044 | `estimate_driver` | `package_support` | `bathroom_repair` | `repair` |
| `outdated_bathroom_finishes` | ~1544 | `marketability` | `package_driver` | `bathroom_modernization` | `modernization` |
| `dated_bathroom_vanity_light` | ~1686 | `marketability` | `package_support` | `bathroom_modernization` | `modernization` |
| `vintage_tile_pattern_style` | ~2214 | `marketability` | `package_support` | `bathroom_modernization` | `modernization` |
| `ornate_vintage_mirror` | ~2275 | `marketability` | `package_support` | `bathroom_modernization` | `modernization` |

Kitchen has **~25 items** with `package_type` starting with `kitchen_`. Bathroom needs comparable depth so the bathroom resolver functions (`_resolve_bathroom_modernization_profile`, `_resolve_bathroom_repair_profile`, `_resolve_bathroom_turnover_profile` in `tools/rehab_packages.py`) have enough component variety to land in the right tier.

## Audit queries — run these first

Get the complete picture of where kitchen is and where bathroom needs to be:

```bash
# Every kitchen-tagged item
grep -nE '"package_type":\s*"kitchen_[^"]+"' tools/issue_catalog.json

# Every existing bathroom-tagged item (should be 6 today)
grep -nE '"package_type":\s*"bathroom_[^"]+"' tools/issue_catalog.json

# Every item that already mentions bathroom in scene_groups (cross-room candidates)
# Use a multiline ripgrep:
rg -U --multiline-dotall '"scene_groups":\s*\[[^\]]*"bathroom"' tools/issue_catalog.json
```

Also enumerate items by trade bucket:
```bash
grep -nE '"trade_bucket":\s*"bathroom_fixtures_tile"' tools/issue_catalog.json
grep -nE '"trade_bucket":\s*"plumbing"' tools/issue_catalog.json
grep -nE '"trade_bucket":\s*"electrical"' tools/issue_catalog.json
```

## Kitchen → bathroom mapping table (seed)

For every kitchen catalog item with `room: kitchen`, decide whether it has a sensible bathroom analogue. Fill out this table from the audit-query results. Add new bathroom entries (or retag existing un-tagged ones) to match. Some seed rows:

| Kitchen item id | Bathroom analogue (status) | package_type | package_role | display_class |
|---|---|---|---|---|
| `outdated_kitchen_finishes` | `outdated_bathroom_finishes` (exists, tagged) | `bathroom_modernization` | `package_driver` | `marketability` |
| `missing_base_cabinets_exposed_subfloor` | `missing_bathroom_tile_exposed_substrate` **(NEW)** | `bathroom_repair` | `package_driver` | `high_concern` |
| `outdated_or_damaged_cabinets` | `outdated_or_damaged_vanity` **(NEW)** | `bathroom_modernization` (if dated) / `bathroom_repair` (if damaged at sev≥3) | `package_driver` | `estimate_driver` |
| `countertop_damage` | `vanity_top_damage` **(NEW)** | `bathroom_repair` | `package_support` | `estimate_driver` |
| `peeling_or_discolored_paint` | (shared cross-room item — do NOT add `room: bathroom`; see below) | n/a | n/a | n/a |
| `dated_lighting_fixtures` | `dated_bathroom_vanity_light` (exists, tagged) | `bathroom_modernization` | `package_support` | `marketability` |
| `appliance_damage_or_missing` | n/a — bathrooms don't have analogous fixtures | — | — | — |
| `worn_kitchen_flooring` | (typically routes via `flooring` trade — shared, not bathroom-specific) | n/a | n/a | n/a |
| `kitchen_plumbing_visible_issue` | `bathroom_plumbing_visible_issue` **(NEW)** | `bathroom_repair` | `package_driver` | `high_concern` |
| `kitchen_electrical_concern` | `bathroom_gfci_missing_or_damaged` **(NEW)** | `bathroom_repair` | `package_driver` | `high_concern` |

Suggested **new** bathroom items beyond the kitchen mapping (no kitchen analogue):

- `missing_or_damaged_caulk_at_tub_or_shower` (sev 1-2, `bathroom_repair`, `package_support`)
- `active_water_damage_bathroom` (sev 3, `bathroom_repair`, `package_driver`, **add to `_STRONG_SIGNAL_CATALOG_IDS`**)
- `mold_or_mildew_visible_bathroom` (sev 2-3, `bathroom_repair`, `package_driver`, `display_class: high_concern`)
- `exhaust_fan_missing_or_damaged` (sev 1-2, `bathroom_repair`, `package_support`)
- `tub_surround_or_shower_pan_damage` (sev 2-3, `bathroom_repair`, `package_driver`)
- `dated_or_worn_vanity_countertop` (sev 1-2, `bathroom_modernization`, `package_support`)

The full table is your output — fill it in by going row-by-row through every `package_type: kitchen_*` item and asking "is there a bathroom version of this defect/opportunity?" If yes and it's missing, add it. If yes and it exists un-tagged, tag it. If no, skip and document why.

## Pricing baselines

Use these as anchors when setting cost ranges for new bathroom items. They come from `BATHROOM_*` constants in `tools/rehab_packages.py` and from `outdated_bathroom_finishes`' existing cost block (per_occurrence_low: 1350, per_occurrence_high: 2700, cap_low: 15000, cap_high: 35000).

Per component family (per-occurrence ranges):
- **Vanity** (cosmetic damage / dated): $300-$1,500 per_occurrence; cap $3,000-$6,000
- **Vanity** (full replacement, severe): $800-$2,500 per_occurrence; cap $3,500-$8,000
- **Tile** (grout / minor repair): $200-$800 per_occurrence; cap $2,000-$5,000
- **Tile** (full re-tile shower surround): $1,500-$6,000 per_occurrence; cap $6,000-$15,000
- **Tub/Shower** (repair): $400-$2,000 per_occurrence; cap $4,000-$10,000
- **Tub/Shower** (replace surround / pan): $1,500-$6,000 per_occurrence; cap $6,000-$15,000
- **Fixture** (faucet, drain, stopper): $150-$600 per_occurrence; cap $1,000-$2,500
- **Bath finish** (paint, caulk, mildew remediation): $200-$1,000 per_occurrence; cap $1,500-$3,000

These are rough guideposts; calibrate against actual quotes you have.

## Strong-signal sentinels to add

After adding the candidate IDs to the catalog, extend `_STRONG_SIGNAL_CATALOG_IDS` in `tools/rehab_packages.py` (currently around line 494):

```python
_STRONG_SIGNAL_CATALOG_IDS = frozenset({
    "missing_base_cabinets_exposed_subfloor",
    # Bathroom sentinels — add only after creating the catalog entries:
    "missing_bathroom_tile_exposed_substrate",
    "active_water_damage_bathroom",
    "missing_vanity_exposed_plumbing",
})
```

These are catalog IDs that, when matched at any severity, elevate a bathroom package to `strong` strength even without supports. They represent unambiguously severe, photo-evidenced defects.

## Cross-room items — DO NOT tag with `room: bathroom`

These items have `scene_groups` containing `"bathroom"` along with kitchen, bedroom, etc. They are intentionally room-agnostic and flow through `_infer_room_refresh` / system-level packages. **Leaving them un-tagged is correct** — assigning `room: bathroom` would lock them out of kitchen/bedroom paths and silently regress those flows.

Examples (verify list via the audit query above):
- `water_stain_ceiling`
- `peeling_or_discolored_paint`
- `older_flooring_style`
- `bare_or_missing_finish_flooring`
- Any "visible electrical risk" / "visible plumbing concern" items that span multiple room types

If you find a cross-room item that you believe should be split into a bathroom-specific entry, **create a new bathroom-specific item** rather than retagging the existing cross-room one. The cross-room one stays untouched.

## Edit conventions

When tagging an existing bathroom item, insert the five fields **after** the `estimate` block (or after `require_any` if there's no `estimate` block — see how `dated_bathroom_vanity_light` was tagged at line ~1686):

```json
"estimate": {
  ...
},
"display_class": "<class>",
"package_role": "<role>",
"package_type": "<type>",
"package_category": "<category>",
"room": "bathroom"
```

When adding a new bathroom item, mirror the shape of an existing tagged bathroom item (e.g., copy `outdated_bathroom_finishes` and modify). Make sure to:
- Set `scene_groups: ["bathroom"]` (single-room).
- Set `tier: "work"` (NOT `"optional"` — optional items are filtered out by `is_defect_driver` / `is_dated_cosmetic_evidence`).
- Set `kind: "defect"` for damage-type items, `kind: "upgrade"` for dated/opportunity-type items.
- Set `trade_bucket` appropriately: `bathroom_fixtures_tile` for vanity/tile/tub/shower/fixture/finish, `plumbing` for visible plumbing concerns, `electrical` for GFCI / vanity-light electrical, `moisture_mold` for active water damage.
- Set `embed_text` with concrete, photo-detectable vocabulary (this is what the embeddings retriever matches against Qwen's observation text).

## QC checklist

After your edits:

1. **Lint the catalog JSON** — `python -c "import json; json.load(open('tools/issue_catalog.json'))"` must succeed.

2. **Test suite** — `pytest tests/test_rehab_packages.py tests/test_renovation_estimate_v4.py` must still be green.

3. **Live run** — `python -m tools.audit_runner --artifacts-root C:\Users\Steven\IntelliJProjects\realtorvision\artifacts --scene-group bathroom --property <known_bathroom_heavy_property>`. Inspect the produced `photo_intel.json`:
   - `renovation_estimate_v4.package_candidates` should include entries with `package_type` starting with `bathroom_` and `room: bathroom`.
   - At least one bathroom package should hit `bathroom_partial_rehab` or `bathroom_full_rehab` if the property has severity-2+ defects — proves the new catalog items are reaching the resolver.
   - `pass_2f_trace.attempted_count > 0` — Pass 2f ran with bathroom-aware prompts.

4. **Frontend check** (assumes Phase H1 frontend handoff is done) — `npm run dev` in the frontend repo, open `/audits/bathroom` on the test property, confirm:
   - Bathroom packages render in the right category badges (Modernization / Repair / Turnover).
   - Pass 2f tally chips are non-zero.
   - For properties with both kitchen and bathroom turnover signals, an `interior_paint_flooring_refresh__whole_home` aggregated package appears at the property level.

5. **Hallucination filter** — find a property where Qwen produced a borderline bathroom finding (e.g., one photo with "outdated vanity" but the bathroom is otherwise updated). In `photo_intel.json`:
   - The single-photo opportunity_driver should be **suppressed** before reaching Pass 2f (corroboration gate).
   - Confirm `suppressed_out` records show `weak_no_qualifying_pattern` or `insufficient_corroboration_for_opportunity_driver`.

## Out of scope

- Frontend changes — handled in `HANDOFF_bathroom_frontend.md`.
- `_infer_bathroom_package` rule changes — already done.
- `_resolve_bathroom_*_profile` function changes — already done.
- Pass 2f prompt changes — already done (`PASS_2F_BATHROOM_SYSTEM_PROMPT` / `PASS_2F_BATHROOM_USER_PROMPT`).
- New pricing tiers beyond what's defined — the `BATHROOM_*` constants in `rehab_packages.py` are the canonical set; if you need a new tier, raise with the user first.

## Reference

- `tools/issue_catalog.json` — the catalog
- `tools/rehab_packages.py` lines 56-65 — `BATHROOM_*` pricing tier constants
- `tools/rehab_packages.py` lines 106-130 — `PACKAGE_TYPE_BATHROOM_*` constants
- `tools/rehab_packages.py` `_resolve_bathroom_modernization_profile`, `_resolve_bathroom_repair_profile`, `_resolve_bathroom_turnover_profile` — what your new catalog items feed into
- `tools/rehab_packages.py` `_has_strong_signal` — where to extend sentinels
- `tools/catalog_embeddings.py` `CatalogEmbeddingsRetriever` — how `embed_text` and `support_any` / `deny_any` get used at matching time
