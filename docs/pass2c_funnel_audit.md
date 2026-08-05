# Pass 2c funnel audit

- Artifacts root: `C:\Users\Steven\IntelliJProjects\renointel-prod\artifacts`
- Selection: latest_run_per_property
- Scanned 824, audited 822, parse errors 2, skipped 0
- 216,299 observations → 66,666 forwarded (30.82%)

## Forward rate by scene group

| Scene group | Observations | Forwarded | Rate | Δ rate |
| --- | --- | --- | --- | --- |
| kitchen | 29,312 | 10,246 | 34.95% | +0.0 |
| bathroom | 24,109 | 7,837 | 32.51% | +0.0 |
| bedroom | 43,128 | 12,505 | 29.0% | +0.0 |
| living_areas | 40,248 | 11,020 | 27.38% | +0.0 |
| utility | 13,630 | 4,625 | 33.93% | +0.0 |
| exterior | 52,446 | 17,470 | 33.31% | +0.0 |
| other | 13,426 | 2,963 | 22.07% | +0.0 |

## Labels

| Label | Forwarded | Dropped |
| --- | --- | --- |
| generic_presence | 0 | 104,642 |
| upgrade_candidate | 41,236 | 0 |
| other | 0 | 28,296 |
| defect_or_damage | 25,430 | 0 |
| good_condition | 0 | 16,437 |
| safety | 0 | 258 |

## Gutter-absence cohort

_Review queue, not human truth._ Regex families over model-authored prose; see `patterns` in the JSON snapshot for the exact set.

276 absence-shaped observations across 200 properties; 179 forwarded, 131 of those resolved to nothing. 115 properties also carry non-absence gutter evidence.

**Stored resolutions onto a gutter item: 26** (`clogged_or_damaged_gutters`, `gutter_maintenance_needed`) — absence must never map here. Of those, **2** would still resolve under the current catalog (`2.1`); the rest are now blocked by `deny_any`. Stored artifacts record the catalog as it was when they were analysed, so only the replayed number can move without a corpus-wide re-analysis.

| Resolved catalog item | Occurrences | Forbidden |
| --- | --- | --- |
| clogged_or_damaged_gutters | 22 | **yes** |
| roofline_water_damage_suspected | 13 | no |
| standing_water_or_poor_grading | 5 | no |
| gutter_maintenance_needed | 4 | **yes** |
| damaged_or_aged_roof_shingles | 3 | no |
| trees_or_vegetation_too_close | 1 | no |

| Absence pattern | Hits |
| --- | --- |
| no_before_component | 220 |
| component_not_visible | 30 |
| missing_component | 12 |
| lack_or_absence_of | 7 |
| without_component | 4 |
| appears_limited | 3 |

### Absence claims resolving onto a gutter item

| Property | Photo | Resolved to | Now blocked | Description |
| --- | --- | --- | --- | --- |
| redfin_10817528 | photo_025.jpg | clogged_or_damaged_gutters | yes | Gutter management on the right-side structure appears limited or absent. |
| redfin_10995792 | photo_008.jpg | clogged_or_damaged_gutters | yes | No visible gutter downspouts direct water away from the porch floor area. |
| redfin_11169120 | photo_035.jpg | clogged_or_damaged_gutters | **no** | The roofline appears uneven with missing or poorly installed gutters. |
| redfin_11177262 | photo_008.jpg | clogged_or_damaged_gutters | yes | Gutters and downspouts appear limited or poorly detailed around the front projection. |
| redfin_11217765 | photo_008.jpg | clogged_or_damaged_gutters | yes | No visible gutters or downspouts are seen in the photo, requiring verification of proper drainage away from foundation. |
| redfin_125969347 | photo_020.jpg | clogged_or_damaged_gutters | **no** | Gutters appear clogged or missing downspouts along the roofline. |
| redfin_125970550 | photo_026.jpg | clogged_or_damaged_gutters | yes | Roof overhangs appear minimal, with missing gutters or downspouts visible. |
| redfin_125970550 | photo_030.jpg | clogged_or_damaged_gutters | yes | Roofline appears uneven with no visible gutters. |
| redfin_126224899 | photo_014.jpg | clogged_or_damaged_gutters | yes | No visible gutters or downspouts are directing water away from the structure. |
| redfin_126224899 | photo_020.jpg | clogged_or_damaged_gutters | yes | No visible gutters or downspouts are functioning properly. |
| redfin_126505816 | photo_026.jpg | gutter_maintenance_needed | yes | No downspout extension is visible for rainwater drainage. |
| redfin_126599097 | photo_014.jpg | clogged_or_damaged_gutters | yes | Lack of visible, well-maintained gutter systems in certain areas. |
| redfin_126879617 | photo_029.jpg | clogged_or_damaged_gutters | yes | No well-maintained gutter systems are visible. |
| redfin_127009168 | photo_007.jpg | clogged_or_damaged_gutters | yes | No visible gutters or downspouts are properly directing water away from the structure. |
| redfin_127060964 | photo_029.jpg | clogged_or_damaged_gutters | yes | No visible gutters or downspouts direct water away from the foundation. |
| redfin_127141331 | photo_043.jpg | clogged_or_damaged_gutters | yes | No visible gutter downspout extension is present to direct water runoff away from the house. |
| redfin_127518587 | photo_001.jpg | clogged_or_damaged_gutters | yes | There is a lack of visible, well-maintained gutter systems. |
| redfin_127665608 | photo_006.jpg | gutter_maintenance_needed | yes | No visible gutters or downspouts are present near the roofline. |
| redfin_127686162 | photo_012.jpg | clogged_or_damaged_gutters | yes | There is no visible gutter system despite the roof overhang. |
| redfin_128706194 | photo_004.jpg | gutter_maintenance_needed | yes | The roofline has no visible gutter system on the front face of the house. |
| redfin_80910681 | photo_002.jpg | clogged_or_damaged_gutters | yes | Missing gutters and downspouts pose a risk for water damage. |
| redfin_80918379 | photo_002.jpg | clogged_or_damaged_gutters | yes | No visible gutters or effective downspout routing are noted. |
| redfin_80930868 | photo_033.jpg | clogged_or_damaged_gutters | yes | No visible gutters or downspouts direct water away from the house foundation. |
| redfin_80963304 | photo_021.jpg | gutter_maintenance_needed | yes | Grass grows close to the foundation with no visible gutters or downspouts. |
| redfin_80964975 | photo_003.jpg | clogged_or_damaged_gutters | yes | Gutters appear missing or non-functional, with no visible downspouts along the roofline. |
| redfin_80990371 | photo_011.jpg | clogged_or_damaged_gutters | yes | No obvious functioning gutters or downspouts are visible. |

## Exterior-finish cohort

25,593 observations mention an exterior finish term; 9,161 forwarded (35.79%), 16,432 dropped.

## Parse errors

| Artifact | Reason |
| --- | --- |
| redfin_10937111/20260719_181703_46ebb62b/photo_intel_debug.json | JSONDecodeError: Expecting value: line 35 column 20 (char 1139) |
| redfin_10993751/20260724_025951_39f7b444/photo_intel_debug.json | JSONDecodeError: Expecting value: line 35 column 20 (char 1134) |
