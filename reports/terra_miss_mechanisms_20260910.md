# Terra miss mechanism census

Offline, zero model calls. Counts are cohort outcomes, not population recall or causal effects.

Run_1: 1047 reviews. Binary dirB: 47 cards; four inconclusive cards retained separately.

## dirb_tables

| Feature | Group | Cards | Properties | Units | Human miss | Correct rejection |
| --- | --- | --- | --- | --- | --- | --- |
| adjacent_any | false | 20 | 11 | 16 | 6 | 14 |
| adjacent_any | true | 27 | 17 | 24 | 14 | 13 |
| adjacent_only | false | 30 | 17 | 24 | 11 | 19 |
| adjacent_only | true | 17 | 11 | 16 | 9 | 8 |
| claim_observation_term_disjoint | false | 39 | 19 | 29 | 16 | 23 |
| claim_observation_term_disjoint | true | 8 | 7 | 8 | 4 | 4 |
| batch_conditions | 2-4 | 7 | 6 | 7 | 4 | 3 |
| batch_conditions | 5-8 | 11 | 8 | 9 | 4 | 7 |
| batch_conditions | 9+ | 29 | 14 | 18 | 12 | 17 |
| batch_photos | 1 | 11 | 9 | 10 | 7 | 4 |
| batch_photos | 2-3 | 13 | 7 | 8 | 4 | 9 |
| batch_photos | 4+ | 23 | 13 | 16 | 9 | 14 |

## dirb_without_six_tables

| Feature | Group | Cards | Properties | Units | Human miss | Correct rejection |
| --- | --- | --- | --- | --- | --- | --- |
| adjacent_any | false | 20 | 11 | 16 | 6 | 14 |
| adjacent_any | true | 21 | 12 | 18 | 8 | 13 |
| adjacent_only | false | 27 | 16 | 22 | 8 | 19 |
| adjacent_only | true | 14 | 9 | 13 | 6 | 8 |
| claim_observation_term_disjoint | false | 35 | 18 | 26 | 12 | 23 |
| claim_observation_term_disjoint | true | 6 | 6 | 6 | 2 | 4 |
| batch_conditions | 2-4 | 5 | 4 | 5 | 2 | 3 |
| batch_conditions | 5-8 | 11 | 8 | 9 | 4 | 7 |
| batch_conditions | 9+ | 25 | 12 | 15 | 8 | 17 |
| batch_photos | 1 | 9 | 7 | 8 | 5 | 4 |
| batch_photos | 2-3 | 12 | 6 | 7 | 3 | 9 |
| batch_photos | 4+ | 20 | 12 | 14 | 6 | 14 |

## Six cases

| Card | Item | Group | Conditions/photos in call | Terra | Factorized |
| --- | --- | --- | --- | --- | --- |
| rc_790b583fcbf0 | peeling_or_discolored_paint | neutral_paint | 4/1 | unsupported | absent |
| rc_905575ed8a3d | soffit_or_porch_ceiling_failed | porch_connection_vs_soffit_panels | 4/1 | cannot_assess | inconclusive |
| rc_a35f1b3a231e | exterior_siding_discoloration_fading | siding_and_trim | 12/13 | unsupported | absent |
| rc_ace7463d6837 | tub_surround_or_shower_pan_damage | shower_hardware_vs_enclosure | 19/4 | unsupported | absent |
| rc_d0197df43080 | hard_flooring_scratched_or_worn | floor_wear_dispute | 9/7 | unsupported | absent |
| rc_eace19540d2d | cabinets_damaged_or_water_stained | cabinet_components | 9/2 | unsupported | absent |

The JSON includes every joined row, matched negation clause, raw human answer, source hash and six-case image fingerprint.
