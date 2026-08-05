# Catalog Audit Report

Catalog audited: `tools/issue_catalog.json`

## 1. Summary of Current Catalog Issues

The catalog currently has 107 items: 67 defects and 40 upgrades. Missing-field counts are: id=0, embed_text=0, kind=0, scene_groups=0.

### Kind Counts

| kind | count |
| --- | --- |
| defect | 67 |
| upgrade | 40 |

### Scene Group Counts

| scene_group | count |
| --- | --- |
| bathroom | 66 |
| bedroom | 44 |
| exterior | 44 |
| kitchen | 49 |
| living_areas | 46 |
| other | 53 |
| pool | 14 |
| utility | 46 |

### Missing Fields

| field | count |
| --- | --- |
| embed_text | 0 |
| id | 0 |
| kind | 0 |
| scene_groups | 0 |

## Resolver Assumptions

- `embed_text` fully replaces the fallback embedding text in `tools/catalog_embeddings.py`; it is not shown in the Pass 2d prompt.
- `support_any` is passed into candidates, shown to Pass 2d, and used by lexical shortcuts; it is not a hard retrieval gate.
- `require_any` and `deny_any` are hard lexical guardrails during embedding candidate retrieval.
- `scene_groups` hard-filters candidate retrieval by observed scene group; missing `scene_groups` defaults to broad reach.
- `drop_if_generic` and `defaultHidden` make candidates generic for prioritization and shortcuts; `drop_if_generic` also suppresses final output in Pass 2e.
- `package_affinity` (room-keyed routing blocks; superseded the flat `package_type`/`package_role`/`package_category`/`room` fields) is not a Pass 2d matching field; it belongs to later package construction / Pass 2f behavior.

## 2. Top 25 Longest `embed_text` Items

| id | name | words | severity | embed_text preview |
| --- | --- | --- | --- | --- |
| missing_bathroom_tile_exposed_substrate | Missing Bathroom Tile Exposed Substrate | 45 | review | Missing bathroom tile exposed substrate. Large area of missing tile exposing cement board, drywall, or wall framing behind shower or tub. Section of tile tor... |
| active_water_damage_bathroom | Active Water Damage in Bathroom | 42 | review | Active water damage in bathroom. Standing water on bathroom floor, soft or buckled drywall near tub or shower, swollen or rotted baseboard, soggy or sunken s... |
| bathroom_plumbing_visible_issue | Bathroom Plumbing Visible Issue | 41 | review | Bathroom plumbing visible issue. Leaking supply line under vanity, corroded shutoff valves, failing or improvised drain assembly, missing P-trap, rusted or s... |
| exterior_door_paint_failure | Exterior Door Paint Failure | 41 | review | Exterior door paint failure or lower door deterioration. Worn, chipped, peeling, or flaking paint on entry door. Moisture-damaged, swollen, or rotting wood a... |
| dated_interior_doors | Dated or Basic Interior Doors | 40 | review | Dated or basic interior doors. Hollow-core six-panel doors, basic builder-grade interior doors, dated wood interior doors, older painted interior doors, louv... |
| mold_or_mildew_visible_bathroom | Mold or Mildew Visible in Bathroom | 40 | review | Mold or mildew visible in bathroom. Black or dark organic staining on grout lines, tile corners, shower caulk, bathroom ceiling, or near exhaust fan. Microbi... |
| missing_base_cabinets_exposed_subfloor | Missing Base Cabinets with Exposed Subfloor/Framing | 39 | review | Missing base cabinets with exposed subfloor or framing. Open or damaged under-sink cabinet area with exposed substrate, plumbing, or framing. Missing lower c... |
| tub_surround_or_shower_pan_damage | Tub Surround or Shower Pan Damage | 39 | review | Tub surround or shower pan damage. Cracked fiberglass tub, chipped enamel surface, separated tub surround panels, damaged shower base or pan, deteriorated sh... |
| concrete_driveway_surface_wear | Concrete Driveway Surface Wear | 38 | review | Concrete driveway or pad surface wear. Visible spalling, pitting, or surface breakdown on driveway concrete. Stained, aged, or discolored concrete pad or dri... |
| damaged_or_aged_roof_shingles | Damaged or Aged Roof Shingles | 38 | review | Damaged or aged roof shingles. Missing, curling, or worn shingles. Roof surface discoloration, debris-covered or dirty shingles, moss or algae growth on roof... |
| missing_vanity_exposed_plumbing | Missing Vanity Exposed Plumbing | 38 | review | Missing vanity with exposed plumbing. Bathroom vanity removed exposing supply lines and P-trap, no sink cabinet present, stripped vanity area showing rough p... |
| peeling_or_damaged_bathroom_paint | Peeling or Damaged Bathroom Paint | 38 | review | Peeling or damaged bathroom paint. Paint failure near bathroom tub or shower, bubbling or flaking paint on bathroom walls or ceiling, moisture-related discol... |
| suspended_drop_ceiling | Suspended/Drop Ceiling with Acoustic Tiles | 38 | review | Suspended or drop ceiling made from removable acoustic tiles set in a visible metal grid. Also called a lay-in tile or T-bar ceiling. Modernization may invol... |
| bathroom_layout_modernization_opportunity | Bathroom Layout Modernization Opportunity | 37 | review | Bathroom layout modernization opportunity. Awkward or constrained bathroom layout: corner toilet placement, narrow wedge bathroom, no vanity counter or only... |
| dated_interior_trim | Dated or Basic Interior Trim | 37 | review | Dated or builder-grade interior trim. Plain or basic baseboards, thin baseboards, older painted baseboards, dated window casing, basic window trim, plain doo... |
| dated_window_treatment_valance | Dated Window Treatment/Valance | 37 | review | Basic or builder-grade mini blinds, horizontal blinds, older blinds, worn window coverings. Dated window treatment or valance. Vintage window valance or curt... |
| worn_or_damaged_bathroom_flooring | Worn or Damaged Bathroom Flooring | 37 | review | Worn or damaged bathroom flooring. Lifted or peeling vinyl tile near toilet base, soft or spongy subfloor at vanity or tub, cracked or missing bathroom floor... |
| landscape_improvement_needed | Landscape Improvement | 36 | review | Landscape improvement. Yard cleanup, trimming, debris removal, refresh for curb appeal. Dirt or debris buildup around exterior areas, general site cleanup ne... |
| outdated_or_damaged_vanity | Outdated or Damaged Vanity | 36 | review | Outdated or damaged vanity. Worn bathroom vanity with peeling laminate, water-damaged base, swollen particle board, broken hinges or drawer fronts. Dated van... |
| yard_debris_overgrown_leaves | Yard Debris / Overgrown with Leaves | 36 | review | Yard debris or overgrown with leaves. Leaf accumulation on ground, debris-covered yard, leaf litter along foundation, dead leaves and organic debris on walkw... |

## 3. Likely Duplicate Room-Specific Clusters

### `dated_wallpaper`

| id | name | kind | scene_groups |
| --- | --- | --- | --- |
| dated_bathroom_wallpaper | Dated Bathroom Wallpaper | upgrade | bathroom |
| dated_wallpaper_present | Dated Wallpaper | upgrade | kitchen, bathroom, bedroom, living_areas, utility |

### `layout_modernization_opportunity`

| id | name | kind | scene_groups |
| --- | --- | --- | --- |
| bathroom_layout_modernization_opportunity | Bathroom Layout Modernization Opportunity | upgrade | bathroom |
| layout_modernization_opportunity | Layout Modernization Opportunity | upgrade | kitchen, bathroom, bedroom, living_areas, utility, exterior, other |

### `outdated_finishes`

| id | name | kind | scene_groups |
| --- | --- | --- | --- |
| outdated_bathroom_finishes | Outdated Bathroom Finishes | upgrade | bathroom |
| outdated_kitchen_finishes | Outdated Kitchen Finishes | upgrade | kitchen |

### `paint_refresh_recommended`

| id | name | kind | scene_groups |
| --- | --- | --- | --- |
| bathroom_paint_refresh_recommended | Bathroom Paint Refresh Recommended | upgrade | bathroom |
| paint_refresh_recommended | Paint Refresh Opportunity | upgrade | kitchen, bathroom, bedroom, living_areas, utility |

## 4. Generic Items With One-Room Package Metadata

| id | name | scene_groups | package_affinity | classification |
| --- | --- | --- | --- | --- |
| baseboard_wear_scuffs | Baseboard Wear or Scuffs | kitchen, bathroom, bedroom, living_areas, utility, exterior, other | bathroom:bathroom_modernization/package_support; bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| damaged_drywall_or_cracks | Drywall Damage or Cracks | kitchen, bathroom, bedroom, living_areas, utility | bathroom:bathroom_modernization/package_support; bedroom:bedroom_repair/package_driver; kitchen:kitchen_modernization/package_support; living:living_repair/package_driver | multi-room affinity |
| dated_interior_doors | Dated or Basic Interior Doors | kitchen, bathroom, bedroom, living_areas, utility | bathroom:bathroom_modernization/package_support; bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| dated_interior_trim | Dated or Basic Interior Trim | kitchen, bathroom, bedroom, living_areas, utility | bathroom:bathroom_modernization/package_support; bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| dated_lighting_fixtures | Outdated Lighting Fixtures | kitchen, bedroom, living_areas, utility, exterior, other | bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| dated_wallpaper_present | Dated Wallpaper | kitchen, bathroom, bedroom, living_areas, utility | bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| dated_wood_paneling | Dated Wood Paneling | kitchen, bathroom, bedroom, living_areas, utility | bathroom:bathroom_modernization/package_support; bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| older_flooring_style | Outdated Flooring Style | kitchen, bathroom, bedroom, living_areas, utility | bedroom:bedroom_modernization/package_driver; kitchen:kitchen_modernization/package_support; living:living_modernization/package_driver | multi-room affinity |
| paint_refresh_recommended | Paint Refresh Opportunity | kitchen, bathroom, bedroom, living_areas, utility | bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| peeling_or_discolored_paint | Peeling or Discolored Paint | kitchen, bathroom, bedroom, living_areas, utility | bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| popcorn_or_acoustic_ceiling_texture | Popcorn Ceiling Texture | kitchen, bathroom, bedroom, living_areas, utility, other | bathroom:bathroom_modernization/package_support; bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| scratched_or_damaged_flooring | Scratched or Damaged Hard Flooring | kitchen, bathroom, bedroom, living_areas, utility | bedroom:bedroom_repair/package_driver; kitchen:kitchen_modernization/package_support; living:living_repair/package_driver | multi-room affinity |
| suspended_drop_ceiling | Suspended/Drop Ceiling with Acoustic Tiles | kitchen, bathroom, bedroom, living_areas, utility, other | bathroom:bathroom_modernization/package_support; bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| wall_scuffs_marks_or_dents | Wall Scuffs, Marks, or Dents | kitchen, bathroom, bedroom, living_areas, utility | bathroom:bathroom_modernization/package_support; bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| water_stain_ceiling | Water Stain on Ceiling | kitchen, bathroom, bedroom, living_areas, utility, exterior, other | bedroom:bedroom_repair/package_driver; living:living_repair/package_driver | multi-room affinity |
| worn_or_stained_carpet | Worn or Stained Carpet | kitchen, bathroom, bedroom, living_areas, utility | bedroom:bedroom_modernization/package_driver; kitchen:kitchen_modernization/package_support; living:living_modernization/package_driver | multi-room affinity |
| worn_or_stained_flooring | Worn or Stained Flooring | kitchen, bathroom, bedroom, living_areas, utility, other | bedroom:bedroom_modernization/package_support; kitchen:kitchen_modernization/package_support; living:living_modernization/package_support | multi-room affinity |
| worn_or_stained_vinyl_linoleum | Worn or Stained Vinyl / Linoleum Flooring | kitchen, bathroom, bedroom, living_areas, utility | bedroom:bedroom_modernization/package_driver; kitchen:kitchen_modernization/package_support; living:living_modernization/package_driver | multi-room affinity |
| bare_or_missing_finish_flooring | Bare or Missing Finish Flooring | kitchen, bathroom, bedroom, living_areas, utility, other |  | role-only metadata |
| boarded_up_entry_or_window | Boarded-Up Entry or Window | kitchen, bathroom, bedroom, living_areas, utility, exterior, other |  | role-only metadata |
| broken_or_fogged_windows | Broken or Fogged Windows | kitchen, bathroom, bedroom, living_areas, utility, exterior, other |  | role-only metadata |
| ceiling_cracks_or_sagging | Ceiling Cracks or Sagging | kitchen, bathroom, bedroom, living_areas, utility, other |  | role-only metadata |
| dated_door_hardware | Dated or Worn Door Hardware | kitchen, bathroom, bedroom, living_areas, utility, exterior, other |  | role-only metadata |
| dated_electrical_outlets_switches | Dated Electrical Outlets or Switches | kitchen, bathroom, bedroom, living_areas, utility |  | role-only metadata |
| dated_entry_or_patio_door | Dated Entry or Patio Door | exterior, living_areas, kitchen, other |  | role-only metadata |
| dated_or_older_windows | Dated or Older Windows | kitchen, bathroom, bedroom, living_areas, utility, exterior, other |  | role-only metadata |
| dated_overall_decor_style | Overall Decor Style Appears Dated | kitchen, bathroom, bedroom, living_areas, utility |  | role-only metadata |
| dated_window_treatment_valance | Dated Window Treatment/Valance | kitchen, bathroom, bedroom, living_areas, utility |  | role-only metadata |
| dirty_or_grimy_window_screens | Dirty or Grimy Window Screens | kitchen, bathroom, bedroom, living_areas, utility, exterior, other |  | role-only metadata |
| floor_dirty_or_heavily_soiled | Dirty or Heavily Soiled Floor | kitchen, bathroom, bedroom, living_areas, utility |  | role-only metadata |
| indoor_storage_clutter_heavy | Heavy Indoor Storage/Clutter | kitchen, bathroom, bedroom, living_areas, utility |  | role-only metadata |
| interior_wall_stripped_to_studs | Interior Wall Stripped to Studs / Exposed Insulation | kitchen, bathroom, bedroom, living_areas, utility |  | role-only metadata |
| layout_modernization_opportunity | Layout Modernization Opportunity | kitchen, bathroom, bedroom, living_areas, utility, exterior, other |  | role-only metadata |
| mismatched_or_inconsistent_furniture_staging | Mismatched Furniture/Staging Opportunity | kitchen, bathroom, bedroom, living_areas, utility |  | role-only metadata |
| missing_or_damaged_handrails | Missing or Damaged Handrails/Guardrails | kitchen, bathroom, bedroom, living_areas, utility, exterior, other, pool |  | role-only metadata |
| pest_or_rodent_evidence | Visible Pest or Rodent Evidence | kitchen, bathroom, bedroom, living_areas, utility, exterior, other |  | role-only metadata |
| plumbing_fixture_leaking_stained | Leaking or Stained Plumbing Fixtures | kitchen, bathroom, utility, pool |  | role-only metadata |
| rotted_subfloor_or_structural_framing | Rotted Subfloor or Structural Framing | kitchen, bathroom, bedroom, living_areas, utility, other |  | role-only metadata |
| staging_or_decluttering_opportunity | Staging or Decluttering Opportunity | kitchen, bathroom, bedroom, living_areas, utility |  | role-only metadata |
| stained_glass_or_vintage_light_fixture | Stained-Glass or Vintage Light Fixture | kitchen, bathroom, bedroom, living_areas, utility |  | role-only metadata |
| trip_hazard_or_unlevel_floor | Trip Hazard or Unlevel Floor | kitchen, bathroom, bedroom, living_areas, utility, exterior, other, pool |  | role-only metadata |
| unfinished_interior_wall_osb_exposed | Unfinished Interior Wall (OSB Exposed) | kitchen, bathroom, bedroom, living_areas, utility |  | role-only metadata |
| visible_electrical_risks | Visible Electrical Risks | kitchen, bathroom, bedroom, living_areas, utility, exterior, other, pool |  | role-only metadata |
| visible_mold_or_mildew | Visible Mold or Mildew | kitchen, bathroom, bedroom, living_areas, utility, exterior, other, pool |  | role-only metadata |

## 5. Latest Tail Items and Recommended Action

The requested bedroom/living-room tail IDs are not present in this checkout. They are still listed below as a safety checklist for the next catalog rewrite.

| id | present | line | after_4694 | recommended_action | name |
| --- | --- | --- | --- | --- | --- |
| worn_or_dated_bedroom_carpet | no |  |  | collapse |  |
| dated_bedroom_flooring_style | no |  |  | collapse |  |
| dated_bedroom_light_or_fan | no |  |  | collapse |  |
| bedroom_wood_paneling_or_wallpaper | no |  |  | collapse |  |
| bedroom_popcorn_ceiling | no |  |  | collapse |  |
| damaged_bedroom_flooring | no |  |  | collapse |  |
| bedroom_drywall_damage_or_holes | no |  |  | collapse |  |
| bedroom_water_stain | no |  |  | collapse |  |
| bedroom_paint_refresh_recommended | no |  |  | collapse |  |
| damaged_or_missing_closet_doors | no |  |  | review |  |
| worn_or_dated_living_carpet | no |  |  | collapse |  |
| dated_living_flooring_style | no |  |  | collapse |  |
| dated_living_light_fixture | no |  |  | collapse |  |
| living_wood_paneling_or_wallpaper | no |  |  | collapse |  |
| living_popcorn_ceiling | no |  |  | collapse |  |
| damaged_living_flooring | no |  |  | collapse |  |
| living_drywall_damage_or_holes | no |  |  | collapse |  |
| living_water_stain | no |  |  | collapse |  |
| living_paint_refresh_recommended | no |  |  | collapse |  |
| dated_fireplace_surround | yes | 3717 | no | review | Dated Fireplace Surround |

## 6. Proposed Generic Replacement IDs for Next Session

- `worn_or_dated_carpet`
- `dated_flooring_style`
- `scratched_or_damaged_flooring`
- `damaged_drywall_or_cracks`
- `water_stain_ceiling`
- `water_stain_wall_or_ceiling`
- `popcorn_or_acoustic_ceiling_texture`
- `dated_wood_paneling_or_wallpaper`
- `dated_lighting_fixtures`
- `interior_paint_refresh_recommended`
- `damaged_or_missing_closet_doors`
- `dated_fireplace_surround`
- `dated_or_scuffed_trim_baseboards`

## Room-Specific Manual Review Appendix

| id | name | kind | scene_groups | embed_words | matched_terms |
| --- | --- | --- | --- | --- | --- |
| active_water_damage_bathroom | Active Water Damage in Bathroom | defect | bathroom | 42 | bathroom |
| bathroom_gfci_missing_or_damaged | Bathroom GFCI Missing or Damaged | defect | bathroom | 33 | bathroom |
| bathroom_layout_modernization_opportunity | Bathroom Layout Modernization Opportunity | upgrade | bathroom | 37 | bathroom |
| bathroom_paint_refresh_recommended | Bathroom Paint Refresh Recommended | upgrade | bathroom | 30 | bathroom |
| bathroom_plumbing_visible_issue | Bathroom Plumbing Visible Issue | defect | bathroom | 41 | bathroom |
| dated_bathroom_flooring_style | Dated Bathroom Flooring Style | upgrade | bathroom | 29 | bathroom |
| dated_bathroom_vanity_light | Dated Bathroom Vanity Light | upgrade | bathroom | 32 | bathroom |
| dated_bathroom_wallpaper | Dated Bathroom Wallpaper | upgrade | bathroom | 35 | bathroom |
| dated_or_worn_vanity_countertop | Dated or Worn Vanity Countertop | upgrade | bathroom | 25 | bathroom |
| exhaust_fan_missing_or_damaged | Bathroom Exhaust Fan Missing or Damaged | defect | bathroom | 33 | bathroom |
| missing_bathroom_tile_exposed_substrate | Missing Bathroom Tile Exposed Substrate | defect | bathroom | 45 | bathroom |
| missing_vanity_exposed_plumbing | Missing Vanity Exposed Plumbing | defect | bathroom | 38 | bathroom |
| mold_or_mildew_visible_bathroom | Mold or Mildew Visible in Bathroom | defect | bathroom | 40 | bathroom |
| outdated_bathroom_finishes | Outdated Bathroom Finishes | upgrade | bathroom | 32 | bathroom |
| outdated_or_damaged_vanity | Outdated or Damaged Vanity | defect | bathroom | 36 | bathroom |
| peeling_or_damaged_bathroom_paint | Peeling or Damaged Bathroom Paint | defect | bathroom | 38 | bathroom |
| vanity_countertop_damage | Vanity Countertop Damage | defect | bathroom | 32 | bathroom |
| vintage_tile_pattern_style | Vintage Tile Pattern/Style | upgrade | bathroom | 14 | bathroom |
| worn_or_damaged_bathroom_flooring | Worn or Damaged Bathroom Flooring | defect | bathroom | 37 | bathroom |
| garage_or_basement_damage | Basement or Garage Damage | defect | utility, other | 25 | garage, basement |
| appliance_damage_or_missing | Appliance Damage or Missing | defect | kitchen | 11 | kitchen |
| outdated_kitchen_finishes | Outdated Kitchen Finishes | upgrade | kitchen | 18 | kitchen |
| dated_fireplace_surround | Dated Fireplace Surround | upgrade | living_areas | 23 | living |
| unfinished_basement_present | Unfinished Basement | upgrade | utility | 24 | living, basement |
