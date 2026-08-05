# Handoff: Bedroom & living-room catalog population

This is a self-contained brief for a follow-up session. The backend **machinery** to make
`bedroom` and `living` package-aware has landed — full parity with kitchen/bathroom
(modernization + repair + turnover for each room): pricing tiers, package types, absorption
scopes, profile resolvers, Pass 2f prompts, the scene-mismatch guard fix, and the whole-home
turnover roll-up. All of it is **dormant** because no catalog items carry the new
`package_type`s yet — exactly like `kitchen_turnover` was before its catalog work.

What's left is the **catalog work**: add new bedroom-specific and living-specific items to
`tools/issue_catalog.json` so the pipeline has enough variety to detect, gate, and emit
bedroom/living packages. You'll edit **only** `tools/issue_catalog.json` (and optionally add a
few sentinels to `tools/rehab_packages.py`). **Do not touch** the resolvers, absorption scopes,
Pass 2f prompts, or the scene-mismatch guard — those are built and tested.

## What already landed (do not redo)

In `tools/rehab_packages.py`:
- **Pricing tiers** (lightweight, near the `KITCHEN_*`/`BATHROOM_*`/`ROOM_*` block): `BEDROOM_REFRESH`,
  `BEDROOM_FULL_REHAB`, `BEDROOM_REPAIR_LIGHT/HEAVY`, `BEDROOM_TURNOVER_LIGHT/STD`, and the six
  `LIVING_*` equivalents.
- **Package types**: `PACKAGE_TYPE_BEDROOM_MODERNIZATION/_REPAIR/_TURNOVER` and `PACKAGE_TYPE_LIVING_*`,
  all in `VALID_PACKAGE_TYPES`, mapped in `_PACKAGE_TYPE_TO_CATEGORY` and `_PACKAGE_TYPE_TO_ROOM`.
- **Absorption scopes**: 12 entries keyed by pricing tier (`bedroom_refresh`, … `living_turnover_std`),
  using the generic component classes (`flooring`, `paint`, `electrical_light`, `moisture`).
- **Resolvers + dispatcher**: generic `_resolve_room_modernization/repair/turnover_profile`, wired into
  `_resolve_pricing_profile` for all six new types.
- **Scene-mismatch guard**: `_normalize_scene_to_room` maps the `living_room` scene id → the `living`
  room constant; the guard now includes `ROOM_BEDROOM`/`ROOM_LIVING`.
- **VLM labels** for the four VLM-verified types (turnover is confirmed-by-rule, no label).

In `tools/scene_classifier_passes.py`:
- `PASS_2F_BEDROOM_SYSTEM/USER_PROMPT` and `PASS_2F_LIVING_SYSTEM/USER_PROMPT`, registered in
  `PASS_2F_ROOM_PROMPTS` under keys **`"bedroom"` and `"living"`** (standard output schema — no
  `visible_room_count` telemetry).

Tests covering all of the above live in `tests/test_package_taxonomy.py`
(`TestBedroomLivingTaxonomy`), `tests/test_rehab_packages.py`
(`TestBedroomLivingPackageCandidates`), and `tests/test_package_pass_2f_and_vlm.py`.

## Current catalog state

**Zero** items carry `package_type: bedroom_*` or `living_*`. Several bedroom/living-relevant
signals already exist as **cross-room** items (e.g. `water_stain_ceiling`,
`popcorn_or_acoustic_ceiling_texture`, `peeling_or_discolored_paint`,
`older_ceiling_fan_style`) — those stay untouched (see "Cross-room items" below). Your job is to
add **new room-specific** items.

## Audit queries — run these first

```bash
# Every kitchen-tagged item (the breadth target to mirror)
grep -nE '"package_type":\s*"kitchen_[^"]+"' tools/issue_catalog.json

# Every bedroom / living tagged item (should be 0 today, then your additions)
grep -nE '"package_type":\s*"(bedroom|living)_[^"]+"' tools/issue_catalog.json

# Items already mentioning bedroom / living_areas in scene_groups (cross-room candidates)
rg -U --multiline-dotall '"scene_groups":\s*\[[^\]]*"bedroom"' tools/issue_catalog.json
rg -U --multiline-dotall '"scene_groups":\s*\[[^\]]*"living_areas"' tools/issue_catalog.json
```

## Item seed lists

Bedroom and living rooms have a smaller surface area than kitchen/bathroom (no
cabinets/counters/tile/vanity) — they're cosmetic-led: flooring, paint, ceiling, lighting/fans,
trim, doors, closet, drywall, water-stain, and (living only) fireplace/built-ins. Aim for ~8–12
items per room. Each maps to one of three package types.

### Bedroom (`room: "bedroom"`, `scene_groups: ["bedroom"]`, `estimate.group: "bedroom"`)

| id (NEW) | kind | trade_bucket | package_type | package_role | display_class |
|---|---|---|---|---|---|
| `worn_or_dated_bedroom_carpet` | upgrade | `flooring` | `bedroom_modernization` | `package_driver` | `marketability` |
| `dated_bedroom_flooring_style` | upgrade | `flooring` | `bedroom_modernization` | `package_support` | `marketability` |
| `dated_bedroom_light_or_fan` | upgrade | `electrical` | `bedroom_modernization` | `package_support` | `marketability` |
| `bedroom_wood_paneling_or_wallpaper` | upgrade | `paint_drywall` | `bedroom_modernization` | `package_support` | `marketability` |
| `bedroom_popcorn_ceiling` | upgrade | `paint_drywall` | `bedroom_modernization` | `package_support` | `marketability` |
| `damaged_bedroom_flooring` | defect | `flooring` | `bedroom_repair` | `package_driver` | `estimate_driver` |
| `bedroom_drywall_damage_or_holes` | defect | `paint_drywall` | `bedroom_repair` | `package_driver` | `estimate_driver` |
| `bedroom_water_stain` | defect | `moisture_mold` | `bedroom_repair` | `package_driver` | `high_concern` |
| `damaged_or_missing_closet_doors` | defect | `paint_drywall` | `bedroom_repair` | `package_support` | `estimate_driver` |
| `bedroom_paint_refresh_recommended` | upgrade | `paint_drywall` | `bedroom_turnover` | `package_support` | `marketability` |

### Living room (`room: "living"`, `scene_groups: ["living"]`, `estimate.group: "living"`)

> **Naming rule (critical):** the room value and `estimate.group` are **`"living"`**, never
> `"living_room"` or `"living_areas"`. `"living"` is the only value in `VALID_ROOMS`; the
> `living_room` scene id is normalized to `living` inside the pipeline. `scene_groups` is a separate
> namespace — use `["living"]` to mirror the single-room convention kitchen/bathroom use (the
> retriever matches on `embed_text`, not `scene_groups`).

| id (NEW) | kind | trade_bucket | package_type | package_role | display_class |
|---|---|---|---|---|---|
| `worn_or_dated_living_carpet` | upgrade | `flooring` | `living_modernization` | `package_driver` | `marketability` |
| `dated_living_flooring_style` | upgrade | `flooring` | `living_modernization` | `package_support` | `marketability` |
| `dated_living_light_fixture` | upgrade | `electrical` | `living_modernization` | `package_support` | `marketability` |
| `living_wood_paneling_or_wallpaper` | upgrade | `paint_drywall` | `living_modernization` | `package_support` | `marketability` |
| `living_popcorn_ceiling` | upgrade | `paint_drywall` | `living_modernization` | `package_support` | `marketability` |
| `dated_fireplace_surround` | upgrade | `paint_drywall` | `living_modernization` | `package_support` | `marketability` |
| `damaged_living_flooring` | defect | `flooring` | `living_repair` | `package_driver` | `estimate_driver` |
| `living_drywall_damage_or_holes` | defect | `paint_drywall` | `living_repair` | `package_driver` | `estimate_driver` |
| `living_water_stain` | defect | `moisture_mold` | `living_repair` | `package_driver` | `high_concern` |
| `living_paint_refresh_recommended` | upgrade | `paint_drywall` | `living_turnover` | `package_support` | `marketability` |

These are seeds — go row-by-row through the kitchen items and ask "is there a bedroom / living
version?" Add what's missing; skip and document what doesn't apply (appliances, plumbing fixtures,
cabinets, tile).

## Required fields per item

Mirror the shape of an existing tagged item (copy `outdated_kitchen_finishes` and modify). After
the `estimate` block, set the five package fields. Full required set:

```json
"scene_groups": ["bedroom"],          // or ["living"] — single-room
"tier": "work",                         // NOT "optional" (optional items are filtered out)
"kind": "defect",                       // damage; use "upgrade" for dated/opportunity items
"trade_bucket": "flooring",             // existing buckets only — see below
"drop_if_generic": false,               // keep false (global kill-switch unsafe; see drop_if_generic handoff)
"embed_text": "concrete, photo-detectable vocabulary the retriever matches against Qwen text",
"cost": { "mode": "allowance", "base_low": ..., "base_high": ...,
          "per_occurrence_low": ..., "per_occurrence_high": ...,
          "cap_low": ..., "cap_high": ..., "cost_source": "manual" },
"estimate": {
  "estimate_tier": "high",              // or "medium"
  "strategy": "replace_only",           // or "repair_only" for defects
  "group": "bedroom",                   // MUST equal room (see reconciliation note)
  "stack_behavior": "group_cap",
  "unit_policy": "per_room"             // generic per-room counting (no per_bedroom code needed)
},
"display_class": "marketability",
"package_role": "package_driver",       // package_driver | package_support
"package_type": "bedroom_modernization",
"package_category": "modernization",    // modernization | repair | turnover (match the type)
"room": "bedroom"                       // bedroom | living — must be in VALID_ROOMS
```

**Trade buckets** — use the existing set only (bedroom/living have no dedicated bucket):
`flooring`, `paint_drywall`, `electrical` (lighting/fans), `moisture_mold` (water stains),
`cleaning_turnover` (turnover paint/clean). `classify_component` already routes these to
`flooring` / `paint` / `electrical_light` / `moisture` — do **not** invent a new bucket.

**Two reconciliation-critical rules:**
1. `estimate.group` **must equal** `room` (`"bedroom"` or `"living"`). Packages bucket by
   `estimate_group` (= room) and estimate line items bucket by catalog `estimate.group`; if they
   differ, the package won't absorb its own line items (it won't crash, but costs double-count).
2. `room` and `estimate.group` use **`"living"`**, never `"living_room"`/`"living_areas"`.

## Pricing baselines

Anchor cost ranges to the new tier constants in `tools/rehab_packages.py` (tune against real
quotes). Per-occurrence guideposts:

- **Flooring** (carpet/vinyl refresh): $500–$2,500 per_occurrence; cap $3,000–$8,000
- **Flooring** (full room re-floor, hardwood/LVP): $1,500–$5,000 per_occurrence; cap $6,000–$15,000
- **Paint / drywall** (patch + repaint a room): $200–$1,200 per_occurrence; cap $1,500–$4,000
- **Ceiling** (popcorn removal / smooth): $400–$1,500 per_occurrence; cap $2,000–$5,000
- **Lighting / fan** (fixture swap): $150–$600 per_occurrence; cap $1,000–$2,500
- **Closet / doors / trim**: $150–$800 per_occurrence; cap $1,000–$3,000
- **Water-stain / moisture** (cosmetic, source already addressed): $200–$1,500 per_occurrence; cap $2,000–$6,000
- **Fireplace surround** (living, cosmetic reface): $800–$4,000 per_occurrence; cap $5,000–$12,000

Tier envelopes (from the constants): bedroom refresh $1.5k–$6k / full $5k–$15k; living refresh
$2k–$8k / full $6k–$18k; repair light/heavy and turnover light/std per the `*_REPAIR_*` /
`*_TURNOVER_*` constants.

## Strong-signal sentinels (optional)

If you add an unambiguously severe, photo-evidenced defect that should elevate a package to
`strong` strength even without supports, add its id to `_STRONG_SIGNAL_CATALOG_IDS` in
`tools/rehab_packages.py` (~line 686) **after** creating the catalog entry. Candidates:
`bedroom_water_stain`, `living_water_stain` (only if you intend active-damage elevation). Leave it
alone otherwise — most bedroom/living signals are cosmetic and should rely on normal
driver/support corroboration.

## Cross-room items — DO NOT tag with `room: bedroom` / `room: living`

These span multiple rooms and intentionally flow through generic/room-refresh paths. Tagging them
to a specific room would lock them out of the other rooms and silently regress those flows. Leave
them untouched; if you want a room-specific version, **create a new item** instead.

- `water_stain_ceiling`
- `popcorn_or_acoustic_ceiling_texture`
- `peeling_or_discolored_paint`
- `older_ceiling_fan_style`
- any multi-room drywall/baseboard/trim/door items

## QC checklist

1. **Lint** — `py -c "import json; json.load(open('tools/issue_catalog.json'))"` succeeds.
2. **Tests** — `py -m pytest tests/test_package_taxonomy.py tests/test_rehab_packages.py tests/test_renovation_estimate_v4.py -q` green.
3. **Live run** — run an audit on a property with worn bedrooms / a dated living room and inspect
   `renovation_estimate_v4.package_candidates`:
   - entries with `package_type` starting `bedroom_`/`living_` and `room: bedroom`/`living`;
   - at least one reaches `*_full_rehab` or `*_repair_heavy` on a broad/severe property;
   - `pass_2f_trace.attempted_count > 0` (modernization/repair packages ran the new prompts);
   - if two+ rooms produce turnover packages, an `interior_paint_flooring_refresh__whole_home`
     aggregate appears at the property level.
4. **Hallucination filter** — on a property with one borderline "dated bedroom" photo against an
   otherwise updated bedroom, the single-photo opportunity driver is **suppressed** before Pass 2f
   (`suppressed_out` shows `weak_no_qualifying_pattern` /
   `insufficient_corroboration_for_opportunity_driver`).

## Out of scope

- Frontend changes (separate repo). New `package_type`/room values may need a badge/label update —
  see `HANDOFF_bathroom_frontend.md` / `frontend_contract_packages.md`.
- Resolver / absorption-scope / Pass 2f prompt / scene-guard changes — already done and tested.
- Bathroom-style multi-room expansion (`expand_bathroom_modernization_packages`) — intentionally
  not built for bedroom/living; distinct bedrooms already separate via per-surrogate
  `estimate_unit_id`.
- New pricing tiers beyond the `BEDROOM_*` / `LIVING_*` constants — raise with the user first.

## Reference

- `tools/issue_catalog.json` — the catalog
- `tools/rehab_packages.py` — `BEDROOM_*` / `LIVING_*` tier constants; `PACKAGE_TYPE_BEDROOM_*` /
  `_LIVING_*`; `_PACKAGE_ABSORPTION_SCOPES`; `_resolve_room_modernization/repair/turnover_profile`;
  `_normalize_scene_to_room`; `_STRONG_SIGNAL_CATALOG_IDS`
- `tools/scene_classifier_passes.py` — `PASS_2F_BEDROOM_*` / `PASS_2F_LIVING_*`, `PASS_2F_ROOM_PROMPTS`
- `tools/catalog_embeddings.py` `CatalogEmbeddingsRetriever` — how `embed_text` / `support_any` /
  `deny_any` get used at matching time
- `docs/HANDOFF_bathroom_catalog_overhaul.md` / `HANDOFF_bathroom_catalog_second_pass.md` — the
  bathroom precedent this mirrors
