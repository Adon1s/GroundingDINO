# Pass 2a vs. downstream error attribution

Generated 2026-09-01T01:14:09.394745+00:00 from `reports\error_attribution_queue.json` and `reports\error_attribution_verdicts.jsonl`.

Descriptive only: this is the reviewed discrepancy cohort, not a system-wide
miss or hallucination rate. Canary and production are not pooled into a rate.

## Misses

| attribution | n | share of judged |
| --- | ---: | ---: |
| pass_2a | 10 | 17.2% |
| downstream | 43 | 74.1% |
| unclear | 5 | 8.6% |
| excluded | 5 | - |
| **judged** | **58** | |

## Hallucinations

| attribution | n | share of judged |
| --- | ---: | ---: |
| pass_2a | 10 | 83.3% |
| downstream | 2 | 16.7% |
| unclear | 0 | 0.0% |
| **judged** | **12** | |

## Appendix

| attribution | n | share of judged |
| --- | ---: | ---: |
| pass_2a | 2 | 10.0% |
| downstream | 8 | 40.0% |
| unclear | 10 | 50.0% |
| **judged** | **20** | |

## Downstream stages

| first responsible stage | n |
| --- | ---: |
| 2b | 4 |
| 2c | 5 |
| 2d | 20 |
| condition_projection | 3 |
| terra | 21 |

## Cases by lane

| lane | pass_2a | downstream | unclear | untraceable | excluded | counted_only | pending |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| appendix_inconclusive | 0 | 2 | 4 | 0 | 0 | 0 | 0 |
| appendix_misnamed | 2 | 6 | 0 | 0 | 0 | 0 | 0 |
| appendix_trivial | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| counted_agreement | 0 | 0 | 0 | 0 | 0 | 7 | 0 |
| counted_correct_rejection | 0 | 0 | 0 | 0 | 0 | 31 | 0 |
| counted_orphan | 0 | 0 | 0 | 0 | 0 | 11 | 0 |
| halluc_label | 8 | 1 | 0 | 0 | 0 | 0 | 0 |
| halluc_v1only | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| miss_gold | 10 | 20 | 5 | 0 | 5 | 0 | 0 |
| miss_label | 0 | 20 | 0 | 0 | 0 | 0 | 0 |
| miss_v1only | 0 | 3 | 0 | 0 | 0 | 0 | 0 |

## Gold lane

15 photos, 112 frozen findings, 40 case(s) materialised from the review.

Gold is photo-observation truth, not billable-condition truth: an unmatched
finding is only a miss when the v5 catalog could have carried it.

| decision | n |
| --- | ---: |
| already_cased | 1 |
| gold_incomplete | 15 |
| matched | 44 |
| miss_candidate | 40 |
| out_of_catalog | 27 |

## Truth basis

`v1_1` carries both adjudicated axes; `v1_only` is one collapsed v1 label and
is capped at medium confidence.

- **gold**: downstream 20, excluded 5, pass_2a 10, unclear 5
- **v1_1**: downstream 27, pass_2a 10, unclear 8
- **v1_only**: downstream 6, pass_2a 2, unclear 2

## Confidence

high 39, low 3, medium 53

## Evidence table

| case | lane | basis | property | catalog item | human | terra | attribution | stage | conf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ga_redfin_10806500_photo_001_g1 | miss_gold | gold | redfin_10806500 | None | None | None | pass_2a |  | medium |
| ga_redfin_10806500_photo_001_g3 | miss_gold | gold | redfin_10806500 | None | None | None | downstream | 2c | medium |
| ga_redfin_10806500_photo_001_g4 | miss_gold | gold | redfin_10806500 | None | None | None | downstream | 2c | high |
| ga_redfin_10806500_photo_001_g6 | miss_gold | gold | redfin_10806500 | None | None | None | downstream | 2b | medium |
| ga_redfin_10806500_photo_001_g7 | miss_gold | gold | redfin_10806500 | None | None | None | downstream | 2b | medium |
| ga_redfin_10806500_photo_002_g1 | miss_gold | gold | redfin_10806500 | None | None | None | pass_2a |  | medium |
| ga_redfin_10806500_photo_002_g3 | miss_gold | gold | redfin_10806500 | None | None | None | downstream | 2c | medium |
| ga_redfin_10806500_photo_002_g6 | miss_gold | gold | redfin_10806500 | None | None | None | unclear |  | medium |
| ga_redfin_10806500_photo_002_g8 | miss_gold | gold | redfin_10806500 | None | None | None | pass_2a |  | medium |
| ga_redfin_10806500_photo_003_g4 | miss_gold | gold | redfin_10806500 | None | None | None | pass_2a |  | medium |
| ga_redfin_10806500_photo_004_g4 | miss_gold | gold | redfin_10806500 | None | None | None | downstream | 2d | medium |
| ga_redfin_10806500_photo_005_g3 | miss_gold | gold | redfin_10806500 | None | None | None | downstream | 2c | medium |
| ga_redfin_10806500_photo_005_g4 | miss_gold | gold | redfin_10806500 | None | None | None | downstream | terra | medium |
| ga_redfin_10806500_photo_006_g6 | miss_gold | gold | redfin_10806500 | None | None | None | pass_2a |  | medium |
| ga_redfin_10806500_photo_006_g7 | miss_gold | gold | redfin_10806500 | None | None | None | pass_2a |  | high |
| ga_redfin_10806500_photo_007_g3 | miss_gold | gold | redfin_10806500 | None | None | None | excluded |  | medium |
| ga_redfin_10806500_photo_007_g6 | miss_gold | gold | redfin_10806500 | None | None | None | pass_2a |  | medium |
| ga_redfin_11000447_photo_001_g6 | miss_gold | gold | redfin_11000447 | None | None | None | pass_2a |  | medium |
| ga_redfin_11000447_photo_001_g7 | miss_gold | gold | redfin_11000447 | None | None | None | pass_2a |  | medium |
| ga_redfin_11000447_photo_002_g4 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | 2d | medium |
| ga_redfin_11000447_photo_002_g5 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | 2d | medium |
| ga_redfin_11000447_photo_002_g8 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | 2d | low |
| ga_redfin_11000447_photo_003_g2 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | 2d | medium |
| ga_redfin_11000447_photo_003_g5 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | condition_projection | high |
| ga_redfin_11000447_photo_003_g8 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | condition_projection | medium |
| ga_redfin_11000447_photo_004_g3 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | 2b | medium |
| ga_redfin_11000447_photo_004_g7 | miss_gold | gold | redfin_11000447 | None | None | None | unclear |  | high |
| ga_redfin_11000447_photo_004_g8 | miss_gold | gold | redfin_11000447 | None | None | None | unclear |  | medium |
| ga_redfin_11000447_photo_005_g1 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | 2c | high |
| ga_redfin_11000447_photo_005_g4 | miss_gold | gold | redfin_11000447 | None | None | None | unclear |  | high |
| ga_redfin_11000447_photo_005_g7 | miss_gold | gold | redfin_11000447 | None | None | None | excluded |  | medium |
| ga_redfin_11000447_photo_005_g8 | miss_gold | gold | redfin_11000447 | None | None | None | excluded |  | medium |
| ga_redfin_11000447_photo_006_g4 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | 2b | low |
| ga_redfin_11000447_photo_006_g5 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | 2d | medium |
| ga_redfin_11000447_photo_006_g7 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | 2d | medium |
| ga_redfin_11000447_photo_007_g5 | miss_gold | gold | redfin_11000447 | None | None | None | excluded |  | medium |
| ga_redfin_11000447_photo_007_g7 | miss_gold | gold | redfin_11000447 | None | None | None | unclear |  | medium |
| ga_redfin_11000447_photo_007_g8 | miss_gold | gold | redfin_11000447 | None | None | None | downstream | 2d | medium |
| ga_redfin_11000447_photo_008_g10 | miss_gold | gold | redfin_11000447 | None | None | None | excluded |  | low |
| ga_redfin_11000447_photo_008_g2 | miss_gold | gold | redfin_11000447 | None | None | None | pass_2a |  | medium |
| rc_022497229bd3 | halluc_label | v1_1 | redfin_25809814 | dated_interior_trim | hard_false_billed | supported | pass_2a |  | high |
| rc_05e73bb8810c | halluc_label | v1_1 | redfin_80917686 | vanity_dated_style | hard_false_billed | supported | pass_2a |  | high |
| rc_0eb9a6d23b21 | appendix_trivial | v1_1 | redfin_10949071 | older_flooring_style | trivial_billed | supported | unclear |  | medium |
| rc_128caa6212b8 | miss_label | v1_1 | redfin_11000447 | vinyl_linoleum_torn_or_lifted | dirB_recovery | unsupported | downstream | terra | high |
| rc_165093fb25c2 | appendix_trivial | v1_1 | redfin_25809814 | cabinets_dated_style | trivial_billed | supported | unclear |  | medium |
| rc_3a19f759f2d9 | appendix_misnamed | v1_1 | redfin_10735912 | patio_or_porch_surface_wear | misnamed_billed | supported | downstream | 2d | medium |
| rc_3deb5aaef0f0 | appendix_misnamed | v1_1 | redfin_125779232 | patio_or_porch_surface_wear | misnamed_billed | supported | pass_2a |  | high |
| rc_429ba6851d09 | halluc_v1only | v1_only | redfin_11185681 | dated_window_treatment_valance | terra_claim_unsupported | supported | pass_2a |  | medium |
| rc_45b9d2f6923c | halluc_v1only | v1_only | redfin_25809814 | dated_wallpaper_present | terra_claim_unsupported | supported | downstream | 2d | medium |
| rc_470fb7e0624f | appendix_inconclusive | v1_1 | redfin_10949071 | dated_interior_trim | excluded | supported | unclear |  | medium |
| rc_48a619175251 | halluc_label | v1_1 | redfin_10735912 | peeling_or_discolored_paint | hard_false_billed | supported | downstream | 2d | medium |
| rc_48ea825e33f8 | miss_label | v1_1 | redfin_10952874 | baseboard_wear_scuffs | dirB_recovery | unsupported | downstream | terra | high |
| rc_4c62bf04b442 | appendix_inconclusive | v1_only | redfin_80990371 | brick_weathered_or_discolored | terra_evidence_inconclusive | unsupported | unclear |  | medium |
| rc_4cab6e7be02b | appendix_misnamed | v1_1 | redfin_10806500 | peeling_or_discolored_paint | misnamed_billed | supported | downstream | 2d | high |
| rc_5063066c1a57 | miss_label | v1_1 | redfin_80925528 | brick_weathered_or_discolored | dirB_recovery | unsupported | downstream | terra | high |
| rc_5b3b3cb9e726 | miss_label | v1_1 | redfin_11077450 | brick_weathered_or_discolored | dirB_recovery | unsupported | downstream | terra | high |
| rc_6285e9a79e45 | appendix_misnamed | v1_1 | redfin_125779232 | bath_fixtures_stained_or_worn | misnamed_billed | supported | downstream | 2d | high |
| rc_62b74e10ba7a | appendix_trivial | v1_1 | redfin_25809814 | older_flooring_style | trivial_billed | supported | unclear |  | medium |
| rc_6b3c68f12764 | appendix_trivial | v1_1 | redfin_25809814 | dated_interior_doors | trivial_billed | supported | unclear |  | medium |
| rc_6f4590af65a3 | miss_label | v1_1 | redfin_80990371 | baseboard_wear_scuffs | dirB_recovery | unsupported | downstream | terra | high |
| rc_6f8278034b05 | appendix_misnamed | v1_1 | redfin_80990371 | boarded_up_entry_or_window | misnamed_billed | supported | downstream | 2d | high |
| rc_790b583fcbf0 | miss_label | v1_1 | redfin_80990371 | peeling_or_discolored_paint | dirB_recovery | unsupported | downstream | terra | high |
| rc_7b23faa0bd8f | miss_label | v1_1 | redfin_10740044 | exterior_siding_discoloration_fading | dirB_recovery | unsupported | downstream | terra | medium |
| rc_7cceb5c878de | halluc_label | v1_1 | redfin_11216660 | older_flooring_style | hard_false_billed | supported | pass_2a |  | high |
| rc_8907b19a4a2f | miss_label | v1_1 | redfin_80917686 | peeling_or_discolored_paint | dirB_recovery | unsupported | downstream | terra | high |
| rc_8df38cb68fc0 | miss_label | v1_1 | redfin_80877597 | paint_refresh_recommended | dirB_recovery | unsupported | downstream | terra | high |
| rc_8eb12e7236b6 | appendix_trivial | v1_1 | redfin_25809814 | older_flooring_style | trivial_billed | supported | unclear |  | medium |
| rc_905575ed8a3d | miss_label | v1_1 | redfin_81000709 | soffit_or_porch_ceiling_failed | dirB_recovery | cannot_assess | downstream | terra | high |
| rc_9c702893a6ce | miss_label | v1_1 | redfin_80916010 | vinyl_linoleum_worn_or_stained | dirB_wording_recovery | unsupported | downstream | 2d | high |
| rc_a0e75a18620e | halluc_label | v1_1 | redfin_25809814 | dated_interior_trim | hard_false_billed | supported | pass_2a |  | high |
| rc_a22a241b3bf0 | miss_v1only | v1_only | redfin_10803207 | dated_window_treatment_valance | terra_claim_supported | unsupported | downstream | 2d | medium |
| rc_a35f1b3a231e | miss_label | v1_1 | redfin_166147710 | exterior_siding_discoloration_fading | dirB_recovery | unsupported | downstream | terra | high |
| rc_ab90ab9da2ab | halluc_label | v1_1 | redfin_10949071 | peeling_or_discolored_paint | hard_false_billed | supported | pass_2a |  | high |
| rc_ace7463d6837 | miss_label | v1_1 | redfin_125970550 | tub_surround_or_shower_pan_damage | dirB_recovery | unsupported | downstream | terra | high |
| rc_b2b0e6b89bc5 | appendix_inconclusive | v1_1 | redfin_10866780 | hard_flooring_scratched_or_worn | excluded | supported | unclear |  | high |
| rc_b2e9e5a031b4 | halluc_label | v1_1 | redfin_80917686 | worn_or_stained_carpet | hard_false_billed | supported | pass_2a |  | high |
| rc_b642fe69b86a | miss_label | v1_1 | redfin_80990371 | dated_interior_trim | dirB_recovery | unsupported | downstream | terra | medium |
| rc_bcb50df30b68 | miss_label | v1_1 | redfin_10952874 | paint_refresh_recommended | dirB_recovery | unsupported | downstream | terra | high |
| rc_c135e76317be | halluc_v1only | v1_only | redfin_126418713 | hard_flooring_scratched_or_worn | terra_claim_unsupported | supported | pass_2a |  | medium |
| rc_c19a185f9547 | miss_label | v1_1 | redfin_80990371 | paint_refresh_recommended | dirB_recovery | unsupported | downstream | terra | high |
| rc_c5afc5b26357 | appendix_inconclusive | v1_only | redfin_11217387 | exterior_siding_discoloration_fading | terra_evidence_inconclusive | unsupported | unclear |  | medium |
| rc_c71bf32491e7 | halluc_label | v1_1 | redfin_10866780 | cabinets_dated_style | hard_false_billed | supported | pass_2a |  | high |
| rc_cf0c3ee00fcb | appendix_misnamed | v1_1 | redfin_80925528 | worn_or_stained_flooring | misnamed_billed | supported | pass_2a |  | high |
| rc_d0197df43080 | miss_label | v1_1 | redfin_126418713 | hard_flooring_scratched_or_worn | dirB_recovery | unsupported | downstream | terra | high |
| rc_d60c090abd2c | miss_label | v1_1 | redfin_11000447 | vinyl_linoleum_torn_or_lifted | dirB_recovery | unsupported | downstream | terra | high |
| rc_d857c0db9013 | miss_label | v1_1 | redfin_10735912 | paint_refresh_recommended | dirB_recovery | unsupported | downstream | terra | high |
| rc_dcc3dc3c88f9 | appendix_trivial | v1_1 | redfin_11077450 | cabinets_dated_style | trivial_billed | supported | unclear |  | medium |
| rc_e2ad788f0845 | miss_v1only | v1_only | redfin_125970550 | damaged_or_rotted_siding_or_trim | terra_claim_supported | unsupported | downstream | terra | medium |
| rc_e9d27cf6e9aa | appendix_misnamed | v1_1 | redfin_11000447 | vanity_worn_finish | misnamed_billed | supported | downstream | 2d | medium |
| rc_eace19540d2d | miss_label | v1_1 | redfin_125779232 | cabinets_damaged_or_water_stained | dirB_recovery | unsupported | downstream | terra | high |
| rc_eccd12ababc5 | appendix_inconclusive | v1_only | redfin_11079485 | dated_interior_trim | terra_evidence_inconclusive | unsupported | downstream | 2d | medium |
| rc_eef1daf73e2a | appendix_misnamed | v1_1 | redfin_10952874 | hard_flooring_scratched_or_worn | misnamed_billed | supported | downstream | 2d | high |
| rc_f227f639eb82 | miss_v1only | v1_only | redfin_10952874 | unfinished_interior_wall_osb_exposed | terra_claim_supported | unsupported | downstream | 2d | medium |
| rc_f48a8d9f18d3 | appendix_inconclusive | v1_only | redfin_11185681 | older_flooring_style | terra_evidence_inconclusive | unsupported | downstream | condition_projection | medium |
| rc_ffa9829fd6dd | halluc_label | v1_1 | redfin_80877597 | worn_or_stained_carpet | hard_false_billed | supported | pass_2a |  | high |

## Limitations

- Attribution granularity stops at photo + 2b bullet: neither 2a nor 2b carries observation ids, so a sentence-level 2a excerpt is the reviewer's inference.
- 2c records no rationale for what it dropped; p2c_dropped is a set difference.
- Gold is photo-observation truth, not billable-condition truth; an unmatched gold finding is a miss only once review judges the catalog could have carried it.
- 2b join health across the queue: {'exact': 133}.
