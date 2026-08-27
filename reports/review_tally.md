# Review tally — 212 of 212 cards reviewed

Human labels judge the claim against the photo. Only Terra is compared at condition level; nothing is derived about Pass 2f (direction A/B are sampling strata). Package warrant is judged on its own cards.

11 verdict(s) refer to cards no longer in the queue (e.g. low-res exclusions); they are ignored above.

Excluded from the queue and from all denominators — evidence entirely below 500px short side: {'production': {'conditions': 64, 'accepted': 36, 'packages': 2, 'bathrooms': 1}}

## 1. Progress per stratum

| stratum | reviewed | % | cards by source |
|---|---|---|---|
| dirA | 51 of 51 | 100% | {'production': 19, 'canary': 32} |
| terra_flip | 16 of 16 | 100% | {'canary': 16} |
| uniform | 54 of 54 | 100% | {'canary': 40, 'production': 14} |
| dirB | 51 of 51 | 100% | {'canary': 36, 'production': 15} |
| p1_package | 22 of 22 | 100% | {'canary': 13, 'production': 9} |
| p3_bathroom | 17 of 17 | 100% | {'production': 7, 'canary': 10} |
| p6_forced_single | 5 of 5 | 100% | {'canary': 5} |

## 2. Condition verdicts by stratum

| group | n | supported | unsupported | overstated | inconclusive | unsupported % | unsup+over % |
|---|---|---|---|---|---|---|---|
| dirA | 51 | 31 | 15 | 5 | 0 | 29% | 39% |
| terra_flip | 16 | 7 | 8 | 1 | 0 | 50% | 56% |
| uniform | 54 | 48 | 5 | 1 | 0 | 9% | 11% |
| dirB | 51 | 20 | 27 | 0 | 4 | 53% | 53% |
| p6_forced_single | 5 | 4 | 1 | 0 | 0 | 20% | 20% |

## 3. Terra verdict vs photo (condition level)

| stratum | n | agree | Terra false positive | Terra miss | overstated | inconclusive |
|---|---|---|---|---|---|---|
| dirA | 51 | 31 | 15 | 0 | 5 | 0 |
| terra_flip | 16 | 8 | 4 | 3 | 1 | 0 |
| uniform | 54 | 48 | 5 | 0 | 1 | 0 |
| dirB | 51 | 27 | 0 | 20 | 0 | 4 |
| p6_forced_single | 5 | 4 | 1 | 0 | 0 | 0 |

## 4. Weighted overall rate on billed (accepted) conditions

overall = (N_A·r_A + N_nonA·r_nonA) / N_accepted — dirA at 100 %, the rest via the uniform sample.

| source | measure | N_accepted | N_A | n_A reviewed | r_A | N_nonA | n_uniform reviewed | r_nonA | overall |
|---|---|---|---|---|---|---|---|---|---|
| canary | unsupported | 743 | 32 | 32 | 21.9% | 711 | 40 | 10.0% | 10.5% |
| canary | unsup+over | 743 | 32 | 32 | 37.5% | 711 | 40 | 12.5% | 13.6% |
| production | unsupported | 203 | 19 | 19 | 42.1% | 184 | 14 | 7.1% | 10.4% |
| production | unsup+over | 203 | 19 | 19 | 42.1% | 184 | 14 | 7.1% | 10.4% |

## 5. Breakdowns — Terra-supported claims only

### by catalog kind

| group | n | supported | unsupported | overstated | inconclusive | unsupported % | unsup+over % |
|---|---|---|---|---|---|---|---|
| defect | 11 | 11 | 0 | 0 | 0 | 0% | 0% |
| degradation | 51 | 36 | 10 | 5 | 0 | 20% | 29% |
| modernization | 53 | 39 | 13 | 1 | 0 | 25% | 26% |
### by photo count

| group | n | supported | unsupported | overstated | inconclusive | unsupported % | unsup+over % |
|---|---|---|---|---|---|---|---|
| 1 photo | 62 | 48 | 14 | 0 | 0 | 23% | 23% |
| 2+ photos | 53 | 38 | 9 | 6 | 0 | 17% | 28% |
### by second opinion (Pass 2f)

| group | n | supported | unsupported | overstated | inconclusive | unsupported % | unsup+over % |
|---|---|---|---|---|---|---|---|
| 2f_agreed | 33 | 29 | 3 | 1 | 0 | 9% | 12% |
| 2f_mixed | 17 | 13 | 1 | 3 | 0 | 6% | 24% |
| 2f_objected | 34 | 18 | 14 | 2 | 0 | 41% | 47% |
| none | 31 | 26 | 5 | 0 | 0 | 16% | 16% |
### by Terra batch size (conditions per call)

| group | n | supported | unsupported | overstated | inconclusive | unsupported % | unsup+over % |
|---|---|---|---|---|---|---|---|
| 2-4 | 23 | 12 | 11 | 0 | 0 | 48% | 48% |
| 5-8 | 36 | 27 | 9 | 0 | 0 | 25% | 25% |
| 9+ | 56 | 47 | 3 | 6 | 0 | 5% | 16% |
### by Terra batch images (per call)

| group | n | supported | unsupported | overstated | inconclusive | unsupported % | unsup+over % |
|---|---|---|---|---|---|---|---|
| 1 | 18 | 12 | 6 | 0 | 0 | 33% | 33% |
| 2-3 | 35 | 27 | 7 | 1 | 0 | 20% | 23% |
| 4+ | 62 | 47 | 10 | 5 | 0 | 16% | 24% |

### by catalog item (n ≥ 3, worst first)

| group | n | supported | unsupported | overstated | inconclusive | unsupported % | unsup+over % |
|---|---|---|---|---|---|---|---|
| peeling_or_discolored_paint | 6 | 2 | 3 | 1 | 0 | 50% | 67% |
| dated_interior_trim | 7 | 3 | 3 | 1 | 0 | 43% | 57% |
| older_flooring_style | 6 | 3 | 3 | 0 | 0 | 50% | 50% |
| hard_flooring_scratched_or_worn | 6 | 3 | 2 | 1 | 0 | 33% | 50% |
| worn_or_stained_flooring | 4 | 2 | 0 | 2 | 0 | 0% | 50% |
| worn_or_stained_carpet | 5 | 3 | 2 | 0 | 0 | 40% | 40% |
| cabinets_dated_style | 6 | 4 | 2 | 0 | 0 | 33% | 33% |
| baseboard_wear_scuffs | 3 | 2 | 1 | 0 | 0 | 33% | 33% |
| appliances_dated_or_basic | 3 | 2 | 1 | 0 | 0 | 33% | 33% |
| floor_dirty_or_heavily_soiled | 4 | 3 | 1 | 0 | 0 | 25% | 25% |
| bath_fixtures_stained_or_worn | 4 | 3 | 0 | 1 | 0 | 0% | 25% |
| patio_or_porch_surface_wear | 5 | 4 | 1 | 0 | 0 | 20% | 20% |
| vanity_dated_style | 5 | 4 | 1 | 0 | 0 | 20% | 20% |
| wall_scuffs_marks_or_dents | 6 | 6 | 0 | 0 | 0 | 0% | 0% |
| paint_refresh_recommended | 5 | 5 | 0 | 0 | 0 | 0% | 0% |
| outdated_bathroom_finishes | 4 | 4 | 0 | 0 | 0 | 0% | 0% |
| outdated_kitchen_finishes | 4 | 4 | 0 | 0 | 0 | 0% | 0% |

### by source

| group | n | supported | unsupported | overstated | inconclusive | unsupported % | unsup+over % |
|---|---|---|---|---|---|---|---|
| canary | 82 | 62 | 14 | 6 | 0 | 17% | 24% |
| production | 33 | 24 | 9 | 0 | 0 | 27% | 27% |

## 6. Error types (tags)

| tag | n |
|---|---|
| other | 4 |
| real_but_overstated | 2 |

Revealed before verdict (peeked): 0 of 212.

## 7. Terra self-flips — which replica matched the photo

| property | item | run_1 | run_2 | human | matched |
|---|---|---|---|---|---|
| redfin_80877597 | paint_refresh_recommended | supported | unsupported | terra_claim_supported | run_1 |
| redfin_80990371 | peeling_or_discolored_paint | unsupported | supported | terra_claim_unsupported | run_1 |
| redfin_25809814 | dated_wallpaper_present | supported | unsupported | terra_claim_unsupported | run_2 |
| redfin_11000447 | dated_or_older_windows | unsupported | supported | terra_claim_unsupported | run_1 |
| redfin_125970550 | hard_flooring_broken_or_warped | supported | unsupported | terra_claim_supported | run_1 |
| redfin_125970550 | damaged_or_rotted_siding_or_trim | unsupported | supported | terra_claim_supported | run_2 |
| redfin_10803207 | dated_window_treatment_valance | unsupported | supported | terra_claim_supported | run_2 |
| redfin_10952874 | unfinished_interior_wall_osb_exposed | unsupported | cannot_assess | terra_claim_supported | neither |
| redfin_11185681 | dated_window_treatment_valance | supported | unsupported | terra_claim_unsupported | run_2 |
| redfin_25809814 | dated_window_treatment_valance | unsupported | supported | terra_claim_unsupported | run_1 |
| redfin_80990371 | appliances_damaged_or_missing | supported | unsupported | terra_claim_supported | run_1 |
| redfin_125779232 | bath_fixtures_stained_or_worn | supported | unsupported | terra_claim_overstated | n/a |
| redfin_11079485 | vintage_tile_pattern_style | supported | unsupported | terra_claim_supported | run_1 |
| redfin_25809814 | older_flooring_style | supported | unsupported | terra_claim_unsupported | run_2 |
| redfin_80925528 | dated_window_treatment_valance | unsupported | supported | terra_claim_unsupported | run_1 |
| redfin_126418713 | hard_flooring_scratched_or_worn | supported | unsupported | terra_claim_unsupported | run_2 |

Counter({'run_1': 8, 'run_2': 6, 'neither': 1, 'n/a': 1})

## 8. P1 packages

| property | package | Sol | 2f | human | tag/notes | legacy |
|---|---|---|---|---|---|---|
| redfin_11185681 | package kitchen_modernization @ kitchen_primary | approve | rejected | package_warranted |  | P06 |
| redfin_80917686 | package bathroom_repair @ bathroom_primary | approve | rejected | not_warranted |  |  |
| redfin_80917686 | package kitchen_modernization @ kitchen_primary | approve | rejected | not_warranted |  |  |
| redfin_80917686 | package exterior_repair @ exterior_primary | approve | rejected | package_warranted |  |  |
| redfin_25809814 | package kitchen_modernization @ kitchen_primary | approve | rejected | package_warranted |  | P12 |
| redfin_10806500 | package exterior_repair @ exterior_primary | approve | rejected | package_warranted |  | P02 |
| redfin_11077450 | package kitchen_modernization @ kitchen_primary | approve | rejected | package_warranted |  | P04 |
| redfin_126418713 | package exterior_repair @ exterior_primary | reject | rejected | package_warranted |  | P08 |
| redfin_10803207 | package bathroom_modernization @ bathroom_primary | approve | rejected | package_warranted |  | P01 |
| redfin_11216660 | package exterior_repair @ exterior_primary | approve | rejected | not_warranted |  |  |
| redfin_127468088 | package exterior_repair @ exterior_primary | approve | rejected | package_warranted |  | P09 |
| redfin_125970550 | package exterior_repair @ exterior_primary | reject | rejected | package_warranted |  | P07 |
| redfin_80990371 | package exterior_repair @ exterior_primary | approve | rejected | package_warranted |  | P13 |
| redfin_11185681 | package exterior_repair @ exterior_primary | approve | rejected | not_warranted | One of the photos is indoor not exterior | P05 |
| redfin_80917686 | package bedroom_modernization @ bedroom_5 | approve | rejected | not_warranted |  |  |
| redfin_25809814 | package bedroom_modernization @ bedroom_3 | approve | rejected | not_warranted |  | P11 |
| redfin_25809814 | package bedroom_modernization @ bedroom_2 | approve | rejected | not_warranted |  | P10 |
| redfin_10806500 | package kitchen_modernization @ kitchen_primary | approve | rejected | package_warranted |  | P03 |
| redfin_10735912 | package exterior_repair @ exterior_primary | approve | rejected | not_warranted |  |  |
| redfin_10949071 | package exterior_repair @ exterior_primary | approve | rejected | not_warranted |  |  |
| redfin_10866780 | package living_modernization @ living_room_primary | reject | rejected | not_warranted |  |  |
| redfin_10949071 | package living_modernization @ living_room_primary | approve | rejected | not_warranted |  |  |

Counter({'package_warranted': 11, 'not_warranted': 11})

## 9. P3 bathrooms

| property | listing baths | v4 surrogates | v5 units | human distinct | billing | notes | legacy |
|---|---|---|---|---|---|---|---|
| redfin_11185681 | 3 | 4 | bathroom_primary | 2 | per_bathroom | there are 3 bathrooms | B03 |
| redfin_11216660 | 3 | 3 | bathroom_primary | 1 | per_bathroom |  |  |
| redfin_166147710 | 2 | 3 | bathroom_primary | 3 | per_bathroom |  | B08 |
| redfin_11217387 | 2.5 | 3 | bathroom_primary | 2 | per_bathroom |  |  |
| redfin_25809814 | 2 | 3 | bathroom_primary | 2 | per_bathroom |  | B09 |
| redfin_80917686 | 4 | 6 | bathroom_primary | 4 | per_bathroom |  |  |
| redfin_10952874 | 1.5 | 2 | bathroom_primary | 2 | per_bathroom |  | B02 |
| redfin_125970550 | 3.0 | 4 | bathroom_primary | 4 | per_bathroom |  | B05 |
| redfin_80877597 | 2 | 3 | bathroom_primary | 2 | per_bathroom |  | B10 |
| redfin_125779232 | 2.0 | 2 | bathroom_primary | 2 | per_bathroom |  | B04 |
| redfin_126224899 | 2.5 | 4 | bathroom_primary | 3 | per_bathroom |  | B06 |
| redfin_126418713 | 2.0 | 3 | bathroom_primary | 2 | per_bathroom | The last image is not a bathroom. Its like a utility room | B07 |
| redfin_10740044 | 1 | 2 | bathroom_primary | 1 | once |  |  |
| redfin_10803207 | 2.5 | 4 | bathroom_primary | 3 | per_bathroom |  | B01 |
| redfin_10949071 | None | 2 | bathroom_primary | 1 | once |  |  |
| redfin_10866780 | 2.5 | 4 | bathroom_primary | 2 | per_bathroom |  |  |
| redfin_80916010 | 1 | 2 | bathroom_primary | 1 | once |  |  |
