# Review analysis — v5 manual review (frozen cohort)

Generated 2026-08-26T15:46:38 · queue sha `8512b87c2af1` · verdicts sha `0c7ca6a8b55d` · integrity **OK**

## 0. Scope and framing

- Condition, package and bathroom cards are separate analysis units; nothing about Pass 2f correctness is derived from a condition verdict (direction A/B are sampling strata).
- Canary and production are analyzed separately; **no pooled headline rate is published**.
- Every percentage carries its numerator, denominator and unique-listing count; slices with fewer than 10 judgments or fewer than 3 listings are flagged exploratory.
- Interval method: 95% Wilson interval on the uniform-arm sample proportion, propagated linearly through the weighting identity with the dirA arm held fixed (complete census); not a full-design interval.
- Semantic authority: docs/DESIGN_review_method.md. The 11 orphaned verdicts (§1.4) are excluded from every metric.

## 1. Freeze and integrity audit

### 1.1 Inputs

| input | path | sha256 | bytes |
|---|---|---|---|
| queue | C:\Users\Steven\PycharmProjects\realtorvision-backend\reports\review_queue.json | 8512b87c2af18d1104069c15a5eda977d5fe79eb736f9aedd976d174f89d318a | 643760 |
| verdicts | C:\Users\Steven\PycharmProjects\realtorvision-backend\reports\review_verdicts.jsonl | 0c7ca6a8b55d6dac69021a00e81315be1b15cef68ffd59fbe2973cb895a8d70f | 40131 |
| catalog | C:\Users\Steven\PycharmProjects\realtorvision-backend\tools\issue_catalog_kind_v2.json | d1cef743f379918fa17f0684c245779180d7d692ebc8fd3e66b7c69bebf68347 | 227967 |
| canary_input_freeze | C:\Users\Steven\PycharmProjects\realtorvision-backend\artifacts_canary\renovation_session9_20260818\input_freeze.json | f32f68125bd4cbb4ef4ccd71d9931382f95b4963ca5584631303ee55d6b4a973 | 170735 |


uniform_rate 0.07 · MIN_EVIDENCE_PX 500 · generated_from {'canary_root': 'C:\\Users\\Steven\\PycharmProjects\\realtorvision-backend\\artifacts_canary\\renovation_session9_20260818', 'prod_root': 'C:\\Users\\Steven\\IntelliJProjects\\renointel-prod\\artifacts', 'since': '20260821_230000'}

### 1.2 Listing manifest

| source | property | run | artifact sha256 | replica sha256 | carded |
|---|---|---|---|---|---|
| canary | redfin_10803207 | 20260817_222851_c0551630 | 4ba9003576c4 | d7552baad738 | yes |
| canary | redfin_10806500 | 20260817_224354_3633e400 | 71bde20de6a0 | 9bb6561c2a33 | yes |
| canary | redfin_10952874 | 20260818_231243_cb71fce9 | 576b9d7e14e4 | 2d6edf63e34f | yes |
| canary | redfin_11000447 | 20260817_224602_d422d7ae | 06f53abd94a7 | b119e8baa17a | yes |
| canary | redfin_11077450 | 20260818_223200_a4168368 | 4177d9f01af8 | 1236e84646e0 | yes |
| canary | redfin_11079485 | 20260817_224118_67317092 | 34b257ee3318 | e22655b92691 | yes |
| canary | redfin_11185681 | 20260818_224036_370aadff | 9a8bebb5de90 | 883c13dc12c5 | yes |
| canary | redfin_125779232 | 20260817_222351_c069aab3 | f1ac9914959a | 816b83c2a93d | yes |
| canary | redfin_125970550 | 20260817_235322_005806b2 | 919e2dd85b03 | 4b7e9077c5aa | yes |
| canary | redfin_126224899 | 20260817_234148_d8fbd963 | c9dd1fed1381 | 2fbbf945290d | yes |
| canary | redfin_126418713 | 20260818_221718_ebc48de0 | 3b0b0d069712 | de676246861e | yes |
| canary | redfin_127468088 | 20260818_215515_38d48db3 | 4f39d951d664 | 9924a67bf7e7 | yes |
| canary | redfin_166147710 | 20260817_224909_2f979a60 | ceb51bca99c4 | 1850f95f0973 | yes |
| canary | redfin_25809814 | 20260817_223307_7b336e61 | 746a97336ed4 | 15e6511fa84c | yes |
| canary | redfin_80877597 | 20260817_223826_3cefe883 | 86eca2c9c785 | f7d0d27cccdc | yes |
| canary | redfin_80925528 | 20260818_220543_1af5ac93 | 76581672fc21 | 2ff1876b7578 | yes |
| canary | redfin_80990371 | 20260817_221632_3d4a8269 | d1d1c645b8f3 | 70ca4f6fdcb3 | yes |
| canary | redfin_81000709 | 20260817_225858_d13bd83c | 05738c0ec71d | 8d805b2d4122 | yes |
| production | redfin_10735912 | 20260825_045924_209d10ee | fb5619490310 | — | yes |
| production | redfin_10740044 | 20260821_233648_4cd603bb | 065b937ba2af | — | yes |
| production | redfin_10866780 | 20260825_050705_9e0d869e | 35bd17c0069b | — | yes |
| production | redfin_10922002 | 20260825_045313_aabadf96 | cce30388f59f | — | NO — no eligible evidence |
| production | redfin_10949071 | 20260825_052355_3b380cca | 7b53cb95d53b | — | yes |
| production | redfin_10965375 | 20260821_233007_c5b989ee | 0eb352adf2c4 | — | NO — no eligible evidence |
| production | redfin_11216660 | 20260821_233308_46f9d96f | 311519f845a2 | — | yes |
| production | redfin_11217387 | 20260821_232530_13f5d5bf | 8bc10840e351 | — | yes |
| production | redfin_80916010 | 20260825_045644_dda08b6b | 9d82ee759937 | — | yes |
| production | redfin_80917686 | 20260821_234153_6c37dfb0 | 2b89c03cc119 | — | yes |

### 1.3 Verdict stream audit

| metric | value |
|---|---|
| raw lines | 238 |
| invalid lines | 0 |
| undo records | 0 |
| peeked records | 0 |
| distinct card_ids | 223 |
| net verdicts (latest wins) | 223 |
| overwrites — identical re-saves | 14 |
| overwrites — changed verdict | 1 |
| timestamp range | 2026-08-23T20:46:13 → 2026-08-26T12:31:07 |
| latest-wins cross-check vs rc.latest_verdicts | ok |


Changed-verdict overwrite on `rc_23ffa00f3c28`: terra_claim_unsupported (2026-08-26T12:22:38) → terra_claim_supported (2026-08-26T12:23:54). The latest record is used everywhere.

### 1.4 Orphaned verdicts (excluded from every metric)

| card_id | source | property | run | kind | key | classification | verdict |
|---|---|---|---|---|---|---|---|
| rc_29055fb2788a | production | redfin_10965375 | 20260821_233007_c5b989ee | condition | oc1_3bc6d24735a3da0a | excluded_low_res | terra_claim_unsupported |
| rc_46cb65e70972 | production | redfin_10965375 | 20260821_233007_c5b989ee | condition | oc1_a11d842f1f7073cf | excluded_low_res | terra_claim_supported |
| rc_492e1a848515 | production | redfin_10965375 | 20260821_233007_c5b989ee | condition | oc1_4edf8a9890cde82f | excluded_low_res | terra_claim_supported |
| rc_4d4fc50326d1 | production | redfin_10965375 | 20260821_233007_c5b989ee | package | bedroom_modernization|bedroom_4 | excluded_low_res | not_warranted |
| rc_790f2b4ac9d4 | production | redfin_10965375 | 20260821_233007_c5b989ee | condition | oc1_c5ba011a97068e24 | excluded_low_res | terra_claim_unsupported |
| rc_8e2a7d85d9e1 | production | redfin_10965375 | 20260821_233007_c5b989ee | condition | oc1_38b0a263fd54d0f4 | excluded_low_res | terra_claim_unsupported |
| rc_b21b7a1f56e2 | production | redfin_10965375 | 20260821_233007_c5b989ee | condition | oc1_cb3584fd9b71601e | excluded_low_res | terra_claim_unsupported |
| rc_b613cc1e2555 | production | redfin_10965375 | 20260821_233007_c5b989ee | bathroom | bathrooms | excluded_low_res | per_bathroom |
| rc_b8009f046889 | production | redfin_10965375 | 20260821_233007_c5b989ee | condition | oc1_c055c0447de3224c | excluded_low_res | terra_claim_supported |
| rc_c842443ef5ed | production | redfin_10965375 | 20260821_233007_c5b989ee | condition | oc1_46c2faf19a299f4b | excluded_low_res | terra_claim_supported |
| rc_e390852f1544 | production | redfin_10965375 | 20260821_233007_c5b989ee | package | exterior_repair|exterior_primary | excluded_low_res | unsure |

### 1.5 Population reconciliation (artifacts vs queue.meta)

| source | field | queue.meta | enumerated | ok |
|---|---|---|---|---|
| canary | accepted | 743 | 743 | ok |
| canary | accepted_dirA | 32 | 32 | ok |
| canary | accepted_non_dirA | 711 | 711 | ok |
| canary | listings | 18 | 18 | ok |
| canary | low_res_excluded | {} | {} | ok |
| canary | uniform | 40 | 40 | ok |
| production | accepted | 203 | 203 | ok |
| production | accepted_dirA | 19 | 19 | ok |
| production | accepted_non_dirA | 184 | 184 | ok |
| production | listings | 10 | 10 | ok |
| production | low_res_excluded | {"conditions": 64, "accepted": 36, "packages": 2, "bathrooms": 1} | {"conditions": 64, "accepted": 36, "packages": 2, "bathrooms": 1} | ok |
| production | uniform | 14 | 14 | ok |

### 1.6 Census and spot checks

| source | dirA judged | dirA population | census ok |
|---|---|---|---|
| canary | 32 | 32 | ok |
| production | 19 | 19 | ok |


| stratum | card_id | kind | rebuild vs queue |
|---|---|---|---|
| dirA | rc_022497229bd3 | condition | ok |
| terra_flip | rc_32ff789c40d0 | condition | ok |
| uniform | rc_027e592f508b | condition | ok |
| dirB | rc_128caa6212b8 | condition | ok |
| p1_package | rc_0b7facddd869 | package | ok |
| p3_bathroom | rc_1ba169da23ec | bathroom | ok |
| p6_forced_single | rc_2aa55cbedcaf | condition | ok |


Subgroup partition checks: all ok · completion 212-complete · flip cross-check vs rc.terra_flips: 16 == 16

## 2. Population and completion

| source | listings | accepted | dirA | non-dirA | uniform picks | low-res excluded |
|---|---|---|---|---|---|---|
| canary | 18 | 743 | 32 | 711 | 40 | {} |
| production | 10 | 203 | 19 | 184 | 14 | {"conditions": 64, "accepted": 36, "packages": 2, "bathrooms": 1} |


| stratum | reviewed | total |
|---|---|---|
| dirA | 51 | 51 |
| terra_flip | 16 | 16 |
| uniform | 54 | 54 |
| dirB | 51 | 51 |
| p1_package | 22 | 22 |
| p3_bathroom | 17 | 17 |
| p6_forced_single | 5 | 5 |

## 3. Accepted-condition truth (per source)

### 3.1 Label mix per estimator arm

| source | arm | n | supported | unsupported | overstated | inconclusive |
|---|---|---|---|---|---|---|
| canary | dirA_accepted | 32 | 20 | 7 | 5 | 0 |
| canary | uniform | 40 | 35 | 4 | 1 | 0 |
| production | dirA_accepted | 19 | 11 | 8 | 0 | 0 |
| production | uniform | 14 | 13 | 1 | 0 | 0 |

### 3.2 Weighted rates on accepted (billed) conditions

| source | measure | N | N_A | x_A | r_A | N_nonA | n_U | x_U | r_U | weighted | 95% band | listings | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| canary | hard_false | 743 | 32 | 7 | 21.9% | 711 | 40 | 4 | 10.0% | 10.5% | [4.7%, 23.0%] | 17 | — |
| canary | broad | 743 | 32 | 12 | 37.5% | 711 | 40 | 5 | 12.5% | 13.6% | [6.8%, 26.6%] | 17 | — |
| canary | inconclusive | 743 | 32 | 0 | 0.0% | 711 | 40 | 0 | 0.0% | 0.0% | [0.0%, 8.4%] | 17 | — |
| production | hard_false | 203 | 19 | 8 | 42.1% | 184 | 14 | 1 | 7.1% | 10.4% | [5.1%, 32.5%] | 8 | — |
| production | broad | 203 | 19 | 8 | 42.1% | 184 | 14 | 1 | 7.1% | 10.4% | [5.1%, 32.5%] | 8 | — |
| production | inconclusive | 203 | 19 | 0 | 0.0% | 184 | 14 | 0 | 0.0% | 0.0% | [0.0%, 19.5%] | 8 | — |


*Evidence:* weighted broad-problem rate 13.6% (95% band [6.8%, 26.6%]), hard-false 10.5% ([4.7%, 23.0%]), inconclusive 0.0% (N_accepted=743 (32 dirA census + 711 non-dirA via 40 uniform judgments), 17 listings). Concentration: largest catalog-kind contribution: degradation at 66% of weighted error mass. Basis: population-estimating (stratified census + uniform sample). Decision domain: condition prompting and evidence handling. No action is prescribed here.


*Evidence:* weighted broad-problem rate 10.4% (95% band [5.1%, 32.5%]), hard-false 10.4% ([5.1%, 32.5%]), inconclusive 0.0% (N_accepted=203 (19 dirA census + 184 non-dirA via 14 uniform judgments), 8 listings). Concentration: largest catalog-kind contribution: degradation at 83% of weighted error mass. Basis: population-estimating (stratified census + uniform sample). Decision domain: condition prompting and evidence handling. No action is prescribed here.

## 4. Error concentration (weighted subgroups)

Weighted subgroup rates use the same census + uniform identity within the subgroup; `contribution` is the subgroup's share of the source's total weighted broad-error mass.


### canary


#### by catalog_kind
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| degradation | 358 | 15 | 343 | 39 | 15 | 17.4% | [7.8%, 35.7%] | 66% | — |
| modernization | 299 | 16 | 283 | 27 | 13 | 10.9% | [3.9%, 38.1%] | 34% | — |
| defect | 86 | 1 | 85 | 6 | 4 | 0.0% | [0.0%, 42.9%] | 0% | exploratory_small_n |


#### by catalog_item (judged ≥ 3; full set in the JSON)
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| worn_or_stained_flooring | 49 | 2 | 47 | 3 | 2 | 98.0% | [21.9%, 98.0%] | 58% | exploratory_small_n,few_listings |
| baseboard_wear_scuffs | 31 | 0 | 31 | 3 | 3 | 33.3% | [6.1%, 79.2%] | 12% | exploratory_small_n |
| cabinets_dated_style | 15 | 2 | 13 | 4 | 4 | 43.3% | [8.2%, 78.5%] | 8% | exploratory_small_n |
| worn_or_stained_carpet | 24 | 0 | 24 | 4 | 4 | 25.0% | [4.6%, 69.9%] | 7% | exploratory_small_n |
| floor_dirty_or_heavily_soiled | 13 | 0 | 13 | 4 | 4 | 25.0% | [4.6%, 69.9%] | 4% | exploratory_small_n |
| dated_interior_trim | 48 | 4 | 44 | 5 | 3 | 6.2% | [6.2%, 79.0%] | 4% | exploratory_small_n |
| older_flooring_style | 31 | 2 | 29 | 3 | 2 | 6.5% | [6.5%, 80.7%] | 2% | exploratory_small_n,few_listings |
| bath_fixtures_stained_or_worn | 9 | 2 | 7 | 3 | 3 | 11.1% | [11.1%, 72.8%] | 1% | exploratory_small_n |
| hard_flooring_scratched_or_worn | 27 | 1 | 26 | 3 | 3 | 3.7% | [3.7%, 67.0%] | 1% | exploratory_small_n |
| outdated_bathroom_finishes | 15 | 2 | 13 | 3 | 3 | 0.0% | [0.0%, 68.8%] | 0% | exploratory_small_n |
| patio_or_porch_surface_wear | 12 | 2 | 10 | 4 | 4 | 0.0% | [0.0%, 54.8%] | 0% | exploratory_small_n |
| peeling_or_discolored_paint | 32 | 4 | 28 | 4 | 4 | — | — | — | exploratory_small_n,no_uniform_coverage |
| wall_scuffs_marks_or_dents | 65 | 0 | 65 | 4 | 3 | 0.0% | [0.0%, 49.0%] | 0% | exploratory_small_n |


#### by photo_bucket
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| 2+ photos | 335 | 19 | 316 | 39 | 13 | 21.3% | [10.0%, 41.6%] | 75% | — |
| 1 photo | 408 | 13 | 395 | 33 | 11 | 5.8% | [1.8%, 23.8%] | 25% | — |


#### by second_opinion
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| 2f_agreed | 392 | 0 | 392 | 20 | 11 | 15.0% | [5.2%, 36.0%] | 57% | — |
| none | 319 | 0 | 319 | 20 | 9 | 10.0% | [2.8%, 30.1%] | 31% | — |
| 2f_objected | 22 | 22 | 0 | 22 | 10 | 40.9% | [40.9%, 40.9%] | 9% | — |
| 2f_mixed | 10 | 10 | 0 | 10 | 8 | 30.0% | [30.0%, 30.0%] | 3% | — |


#### by batch_conditions
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| 5-8 | 208 | 8 | 200 | 20 | 10 | 26.0% | [10.5%, 53.1%] | 57% | — |
| 9+ | 428 | 20 | 408 | 45 | 14 | 8.8% | [3.3%, 25.0%] | 40% | — |
| 2-4 | 99 | 4 | 95 | 7 | 4 | 3.0% | [3.0%, 56.9%] | 3% | exploratory_small_n |
| 1 | 8 | 0 | 8 | 0 | 0 | — | — | — | exploratory_small_n,few_listings,no_uniform_coverage |


#### by batch_images
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| 4+ | 373 | 19 | 354 | 44 | 13 | 17.1% | [8.0%, 34.8%] | 69% | — |
| 1 | 148 | 3 | 145 | 9 | 7 | 16.3% | [2.9%, 55.2%] | 26% | exploratory_small_n |
| 2-3 | 222 | 10 | 212 | 19 | 8 | 2.3% | [2.3%, 30.8%] | 5% | — |


### production


#### by catalog_kind
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| degradation | 98 | 5 | 93 | 11 | 6 | 19.9% | [6.9%, 57.6%] | 83% | — |
| modernization | 95 | 12 | 83 | 19 | 8 | 4.2% | [4.2%, 35.2%] | 17% | — |
| defect | 10 | 2 | 8 | 3 | 2 | 0.0% | [0.0%, 63.5%] | 0% | exploratory_small_n,few_listings |


#### by catalog_item (judged ≥ 3; full set in the JSON)
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| older_flooring_style | 12 | 2 | 10 | 3 | 2 | 8.3% | [8.3%, 74.5%] | 7% | exploratory_small_n,few_listings |


#### by photo_bucket
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| 1 photo | 132 | 11 | 121 | 20 | 8 | 14.7% | [6.4%, 44.4%] | 91% | — |
| 2+ photos | 71 | 8 | 63 | 13 | 5 | 2.8% | [2.8%, 41.4%] | 9% | — |


#### by second_opinion
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| none | 87 | 0 | 87 | 5 | 3 | 20.0% | [3.6%, 62.4%] | 69% | exploratory_small_n |
| 2f_objected | 12 | 12 | 0 | 12 | 5 | 58.3% | [58.3%, 58.3%] | 28% | — |
| 2f_mixed | 7 | 7 | 0 | 7 | 6 | 14.3% | [14.3%, 14.3%] | 4% | exploratory_small_n |
| 2f_agreed | 97 | 0 | 97 | 9 | 5 | 0.0% | [0.0%, 29.9%] | 0% | exploratory_small_n |


#### by batch_conditions
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| 2-4 | 49 | 9 | 40 | 14 | 8 | 28.6% | [15.2%, 63.2%] | 88% | — |
| 5-8 | 70 | 7 | 63 | 11 | 7 | 1.4% | [1.4%, 45.5%] | 6% | — |
| 9+ | 76 | 3 | 73 | 8 | 5 | 1.3% | [1.3%, 43.0%] | 6% | exploratory_small_n |
| 1 | 8 | 0 | 8 | 0 | 0 | — | — | — | exploratory_small_n,few_listings,no_uniform_coverage |


#### by batch_images
| group | N | N_A | N_nonA | judged | listings | weighted broad | 95% band | contribution | flags |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 4 | 28 | 6 | 6 | 53.1% | [17.6%, 88.6%] | 77% | exploratory_small_n |
| 2-3 | 76 | 9 | 67 | 15 | 7 | 3.9% | [3.9%, 38.4%] | 14% | — |
| 4+ | 95 | 6 | 89 | 12 | 6 | 2.1% | [2.1%, 38.7%] | 9% | — |


### p6_forced_single (targeted raw counts — the accepted-provisionally single-photo items)

| property | item | photos | human verdict | legacy |
|---|---|---|---|---|
| redfin_126418713 | vanity_dated_style | 1 | terra_claim_supported | rar1_7a69e9ae5fbabd7f |
| redfin_25809814 | appliances_dated_or_basic | 1 | terra_claim_unsupported | C062 |
| redfin_127468088 | vanity_dated_style | 1 | terra_claim_supported | rar1_afda9927a9027fb6 |
| redfin_10803207 | vanity_dated_style | 1 | terra_claim_supported | rar1_de0247255b9e220e |
| redfin_10803207 | dated_bathroom_flooring_style | 1 | terra_claim_supported | rar1_32834e11fec24cbe |


Verdict counts: {"terra_claim_supported": 4, "terra_claim_unsupported": 1} over n=5 cards, 4 listings — targeted, not a population estimate.


*Evidence:* subgroup rates are population-estimating only where the uniform arm covers the subgroup; rows flagged `no_uniform_coverage`, `exploratory_small_n` or `few_listings` are targeted or exploratory. Decision domain: condition prompting, evidence handling and pass boundaries. No action is prescribed here.

## 5. Terra-rejected claims (dirB — targeted recovery yield)

| source | terra_verdict | n | listings | human supported | human unsupported | human overstated | human inconclusive |
|---|---|---|---|---|---|---|---|
| all | cannot_assess | 4 | 3 | 1 | 3 | 0 | 0 |
| all | unsupported | 47 | 21 | 19 | 24 | 0 | 4 |
| canary | cannot_assess | 4 | 3 | 1 | 3 | 0 | 0 |
| canary | unsupported | 32 | 15 | 15 | 14 | 0 | 3 |
| production | cannot_assess | 0 | 0 | 0 | 0 | 0 | 0 |
| production | unsupported | 15 | 6 | 4 | 10 | 0 | 1 |


*Evidence:* human labels on Terra-rejected, 2f-confirmed claims as tallied in §5 (n=51 targeted dirB cards, 21 listings). Concentration: split by Terra unsupported vs cannot_assess. Basis: targeted (a recovery yield, not Terra recall). Decision domain: pass boundaries and condition prompting. No action is prescribed here.

## 6. Terra stability (canary run_1 × run_2)

Exact-comparable pairs: **250** · flips: **16** · flip rate 6.4% (95% Wilson [4.0%, 10.1%])


| flip direction | n |
|---|---|
| supported->unsupported | 9 |
| unsupported->cannot_assess | 1 |
| unsupported->supported | 6 |


| human matched | n |
|---|---|
| n/a | 1 |
| neither | 1 |
| run_1 | 8 |
| run_2 | 6 |


| evidence load | pairs | flips | flip rate | 95% Wilson |
|---|---|---|---|---|
| 1 photo | 195 | 13 | 6.7% | [3.9%, 11.1%] |
| 2+ photos | 55 | 3 | 5.5% | [1.9%, 14.9%] |


| property | item | run_1 | run_2 | human | matched | photos |
|---|---|---|---|---|---|---|
| redfin_11079485 | vintage_tile_pattern_style | supported | unsupported | terra_claim_supported | run_1 | 1 |
| redfin_11185681 | dated_window_treatment_valance | supported | unsupported | terra_claim_unsupported | run_2 | 1 |
| redfin_80990371 | peeling_or_discolored_paint | unsupported | supported | terra_claim_unsupported | run_1 | 1 |
| redfin_25809814 | dated_wallpaper_present | supported | unsupported | terra_claim_unsupported | run_2 | 1 |
| redfin_25809814 | dated_window_treatment_valance | unsupported | supported | terra_claim_unsupported | run_1 | 1 |
| redfin_125779232 | bath_fixtures_stained_or_worn | supported | unsupported | terra_claim_overstated | n/a | 2 |
| redfin_11000447 | dated_or_older_windows | unsupported | supported | terra_claim_unsupported | run_1 | 1 |
| redfin_125970550 | hard_flooring_broken_or_warped | supported | unsupported | terra_claim_supported | run_1 | 1 |
| redfin_25809814 | older_flooring_style | supported | unsupported | terra_claim_unsupported | run_2 | 2 |
| redfin_10803207 | dated_window_treatment_valance | unsupported | supported | terra_claim_supported | run_2 | 1 |
| redfin_126418713 | hard_flooring_scratched_or_worn | supported | unsupported | terra_claim_unsupported | run_2 | 3 |
| redfin_80877597 | paint_refresh_recommended | supported | unsupported | terra_claim_supported | run_1 | 1 |
| redfin_125970550 | damaged_or_rotted_siding_or_trim | unsupported | supported | terra_claim_supported | run_2 | 1 |
| redfin_80990371 | appliances_damaged_or_missing | supported | unsupported | terra_claim_supported | run_1 | 1 |
| redfin_10952874 | unfinished_interior_wall_osb_exposed | unsupported | cannot_assess | terra_claim_supported | neither | 1 |
| redfin_80925528 | dated_window_treatment_valance | unsupported | supported | terra_claim_unsupported | run_1 | 1 |


*Evidence:* Terra flip rate 6.4% on exact-comparable replica pairs (95% Wilson [4.0%, 10.1%]) (16 flips / 250 pairs, canary only, 12 flip-card listings). Concentration: human matched run_1 8, run_2 6, neither 1, n/a 1. Basis: population-estimating for the canary replicas; does not attribute flips to batching. Decision domain: model stability. No action is prescribed here.

## 7. Package warrant (P1 — census of the 2f-rejected, Sol-decided slice)

| human verdict | n |
|---|---|
| not_warranted | 11 |
| package_warranted | 11 |


### Sol decision × human verdict

| Sol decision | package_warranted | not_warranted | unsure |
|---|---|---|---|
| approve | 9 | 10 | 0 |
| reject | 2 | 1 | 0 |


### by package type
| package type | package_warranted | not_warranted | unsure |
|---|---|---|---|
| bathroom_modernization | 1 | 0 | 0 |
| bathroom_repair | 0 | 1 | 0 |
| bedroom_modernization | 0 | 3 | 0 |
| exterior_repair | 6 | 4 | 0 |
| kitchen_modernization | 4 | 1 | 0 |
| living_modernization | 0 | 2 | 0 |


### by child count
| child count | package_warranted | not_warranted | unsure |
|---|---|---|---|
| 1 | 2 | 2 | 0 |
| 2 | 2 | 4 | 0 |
| 3 | 3 | 4 | 0 |
| 4 | 1 | 1 | 0 |
| 5 | 2 | 0 | 0 |
| 6 | 1 | 0 | 0 |


### by driver class
| driver class | package_warranted | not_warranted | unsure |
|---|---|---|---|
| includes_damage_or_degradation | 8 | 6 | 0 |
| style_only | 2 | 3 | 0 |
| unknown | 1 | 2 | 0 |


### Per-card detail

| source | property | package | children | Sol | human | driver class | legacy | note |
|---|---|---|---|---|---|---|---|---|
| production | redfin_80917686 | exterior_repair @ exterior_primary | 3 | approve | package_warranted | includes_damage_or_degradation | — |  |
| canary | redfin_25809814 | bedroom_modernization @ bedroom_3 | 3 | approve | not_warranted | style_only | P11 |  |
| production | redfin_80917686 | bedroom_modernization @ bedroom_5 | 1 | approve | not_warranted | includes_damage_or_degradation | — |  |
| canary | redfin_11185681 | kitchen_modernization @ kitchen_primary | 5 | approve | package_warranted | includes_damage_or_degradation | P06 |  |
| canary | redfin_25809814 | bedroom_modernization @ bedroom_2 | 3 | approve | not_warranted | style_only | P10 |  |
| production | redfin_11216660 | exterior_repair @ exterior_primary | 3 | approve | not_warranted | includes_damage_or_degradation | — |  |
| canary | redfin_127468088 | exterior_repair @ exterior_primary | 3 | approve | package_warranted | includes_damage_or_degradation | P09 |  |
| canary | redfin_10806500 | kitchen_modernization @ kitchen_primary | 5 | approve | package_warranted | includes_damage_or_degradation | P03 |  |
| canary | redfin_11185681 | exterior_repair @ exterior_primary | 4 | approve | not_warranted | includes_damage_or_degradation | P05 | One of the photos is indoor not exterior |
| canary | redfin_80990371 | exterior_repair @ exterior_primary | 1 | approve | package_warranted | includes_damage_or_degradation | P13 |  |
| production | redfin_80917686 | bathroom_repair @ bathroom_primary | 1 | approve | not_warranted | style_only | — |  |
| production | redfin_10735912 | exterior_repair @ exterior_primary | 3 | approve | not_warranted | includes_damage_or_degradation | — |  |
| production | redfin_10949071 | living_modernization @ living_room_primary | 2 | approve | not_warranted | unknown | — |  |
| canary | redfin_11077450 | kitchen_modernization @ kitchen_primary | 6 | approve | package_warranted | includes_damage_or_degradation | P04 |  |
| canary | redfin_10806500 | exterior_repair @ exterior_primary | 1 | approve | package_warranted | includes_damage_or_degradation | P02 |  |
| canary | redfin_25809814 | kitchen_modernization @ kitchen_primary | 2 | approve | package_warranted | style_only | P12 |  |
| canary | redfin_126418713 | exterior_repair @ exterior_primary | 2 | reject | package_warranted | includes_damage_or_degradation | P08 |  |
| canary | redfin_125970550 | exterior_repair @ exterior_primary | 4 | reject | package_warranted | unknown | P07 |  |
| canary | redfin_10803207 | bathroom_modernization @ bathroom_primary | 3 | approve | package_warranted | style_only | P01 |  |
| production | redfin_10949071 | exterior_repair @ exterior_primary | 2 | approve | not_warranted | includes_damage_or_degradation | — |  |
| production | redfin_10866780 | living_modernization @ living_room_primary | 2 | reject | not_warranted | unknown | — |  |
| production | redfin_80917686 | kitchen_modernization @ kitchen_primary | 2 | approve | not_warranted | includes_damage_or_degradation | — |  |


*Evidence:* human package-warrant labels and Sol-vs-human confusion as tallied in §7 (n=22 P1 cards (census of the 2f-rejected, Sol-decided slice), 14 listings). Concentration: split by package type, child count and driver class. Basis: targeted census of one contested slice — not package-pipeline accuracy. Decision domain: package formation. No action is prescribed here.

## 8. Packaging flow of reviewed accepted conditions


### canary (n=82, 18 listings)

| human label | standalone | package_driver | package_support | no_active_work_item |
|---|---|---|---|---|
| terra_claim_supported | 24 | 18 | 20 | 0 |
| terra_claim_unsupported | 4 | 5 | 5 | 0 |
| terra_claim_overstated | 0 | 1 | 5 | 0 |
| terra_evidence_inconclusive | 0 | 0 | 0 | 0 |


### production (n=33, 8 listings)

| human label | standalone | package_driver | package_support | no_active_work_item |
|---|---|---|---|---|
| terra_claim_supported | 5 | 9 | 10 | 0 |
| terra_claim_unsupported | 3 | 3 | 3 | 0 |
| terra_claim_overstated | 0 | 0 | 0 | 0 |
| terra_evidence_inconclusive | 0 | 0 | 0 | 0 |


*Evidence:* billing-path composition of reviewed accepted conditions as tallied in §8 (n=115 reviewed accepted conditions across 26 listing-source rows). Concentration: shown per human label and billing path. Basis: raw composition (mixes strata; a packaging-utilization proxy, not weighted and not a usefulness judgment). Decision domain: package formation. No action is prescribed here.

## 9. Package-gate exposure (weighted broad-error mass)

| source | standalone | package driver | package support | no active work item | share packaged | error records | uniform weight | flags |
|---|---|---|---|---|---|---|---|---|
| canary | 35.5 | 38.5 | 26.8 | 0.0 | 64.8% | 17 | 17.77 | — |
| production | 15.1 | 3.0 | 3.0 | 0.0 | 28.4% | 9 | 13.14 | exploratory_small_n |


*Evidence:* weighted broad-error mass split standalone vs packaged as tallied in §9 (error records: canary n=17, production n=9). Concentration: share billed inside an applied package: canary 64.8%, production 28.4%. Basis: population-estimating weights on a small error count — read with the flags. Decision domain: package formation and pass boundaries (where defects survive downstream). No action is prescribed here.

## 10. Multi-bathroom billing (P3 — the selected multi-surrogate cohort)

| source | property | listing baths | surrogates | human distinct | billing | target | v5 applied | v5 candidates | v4 packages | v4 expanded | vs v5 | vs v4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| canary | redfin_25809814 | 2 | 3 | 2 | per_bathroom | 2 | 1 | 1 | 1 | False | under | under |
| production | redfin_11217387 | 2.5 | 3 | 2 | per_bathroom | 2 | 1 | 1 | 1 | False | under | under |
| production | redfin_80916010 | 1 | 2 | 1 | once | 1 | 0 | 0 | 0 | False | under | under |
| canary | redfin_125970550 | 3.0 | 4 | 4 | per_bathroom | 4 | 1 | 1 | 3 | True | under | under |
| canary | redfin_10803207 | 2.5 | 4 | 3 | per_bathroom | 3 | 1 | 1 | 0 | False | under | under |
| production | redfin_11216660 | 3 | 3 | 1 | per_bathroom | 1 | 0 | 0 | 0 | False | under | under |
| production | redfin_10866780 | 2.5 | 4 | 2 | per_bathroom | 2 | 0 | 0 | 0 | False | under | under |
| production | redfin_10740044 | 1 | 2 | 1 | once | 1 | 0 | 1 | 1 | False | under | exact |
| canary | redfin_10952874 | 1.5 | 2 | 2 | per_bathroom | 2 | 1 | 1 | 1 | False | under | under |
| canary | redfin_125779232 | 2.0 | 2 | 2 | per_bathroom | 2 | 1 | 1 | 2 | True | under | exact |
| production | redfin_10949071 | None | 2 | 1 | once | 1 | 1 | 1 | 1 | False | exact | exact |
| canary | redfin_126224899 | 2.5 | 4 | 3 | per_bathroom | 3 | 1 | 1 | 3 | True | under | exact |
| canary | redfin_126418713 | 2.0 | 3 | 2 | per_bathroom | 2 | 1 | 1 | 2 | True | under | exact |
| canary | redfin_80877597 | 2 | 3 | 2 | per_bathroom | 2 | 1 | 1 | 2 | True | under | exact |
| production | redfin_80917686 | 4 | 6 | 4 | per_bathroom | 4 | 0 | 0 | 0 | False | under | under |
| canary | redfin_166147710 | 2 | 3 | 3 | per_bathroom | 3 | 1 | 1 | 1 | False | under | under |
| canary | redfin_11185681 | 3 | 4 | 2 | per_bathroom | 2 | 1 | 1 | 3 | True | under | over |


Aggregate (decisive cards): {"v4_exact": 6, "v4_over": 1, "v4_under": 10, "v5_exact": 1, "v5_under": 16} · unit gap (human target − billed packages): v5 +25, v4 +16. Counts cover bathroom_modernization packages only; bathroom work billed through other package types or standalone lines is outside this comparison.


*Evidence:* exact/under/over vs the human bathroom-count target as tallied in §10; aggregate unit gap v5 +25, v4 +16 (n=17 multi-surrogate bathroom cards, 17 listings). Concentration: per-card table in §10. Basis: targeted (the selected multi-surrogate cohort). Decision domain: room multiplicity. No action is prescribed here.

## 11. Qualitative themes (tags and coded notes)

| error tag | n |
|---|---|
| other | 4 |
| real_but_overstated | 2 |


| theme | n |
|---|---|
| claim_wording | 15 |
| object_room_mismatch | 6 |
| other | 2 |
| room_identity | 2 |
| evidence_ingest | 1 |
| package_coherence | 1 |


### Coded notes (full text, auditable)

| card_id | kind | verdict | theme | note |
|---|---|---|---|---|
| rc_025ca614449b | condition | terra_claim_overstated | claim_wording | Painted doorway casing is chipped and dirty is true. The ceiling appears unevenly discolored in places is false. The ceiling claim is a hallucination based on the reflected colors on the ceiling from the window |
| rc_128caa6212b8 | condition | terra_claim_supported | object_room_mismatch | Looks like hardwood not vinyl or linoleum but yes it is very worn as described |
| rc_42f54b0e00fd | condition | terra_claim_unsupported | other | So the actual claim is false but the hallucination is understandable. I dont think any correction is necessary here |
| rc_45b9d2f6923c | condition | terra_claim_unsupported | object_room_mismatch | I believe it is a backsplash not wallpaper. But it does look dated |
| rc_48a619175251 | condition | terra_claim_unsupported | claim_wording | The paint is not "yellowed" it is the color yellow intentionally. Although it is dated and needs repainting its not peeling, bubbling, or aged |
| rc_61b0f2981ca9 | condition | terra_claim_unsupported | claim_wording | The vanity is just opened. Its not unfinished |
| rc_6285e9a79e45 | condition | terra_claim_overstated | claim_wording | The wear is overstated. It is more just extremely outdated and needing of refresh |
| rc_62d0dc3b9df8 | condition | terra_claim_unsupported | object_room_mismatch | It is a drop ceiling not popcorn. But it does need replacing |
| rc_63fac87a14e4 | condition | terra_claim_unsupported | evidence_ingest | Photo is very dark and its hard to see anything |
| rc_65f169bbe3b4 | condition | terra_claim_unsupported | object_room_mismatch | The main issue is that I dont see wood paneling. I see wood cabinets. The other probelm is shouldnt this be broken into 2 separate issues? I only see wall paneling dated wood paneling |
| rc_6c4e3194c4c9 | condition | terra_claim_unsupported | claim_wording | Its just paint and discolor. Nothing is chipped |
| rc_6f8278034b05 | condition | terra_claim_supported | claim_wording | True but this should be generic presence |
| rc_7057c3c171e5 | condition | terra_claim_unsupported | claim_wording | Trim is missing in some photos, its not visibly scuffed |
| rc_7b7343ba4892 | package | not_warranted | package_coherence | One of the photos is indoor not exterior |
| rc_aa7dae68eb01 | condition | terra_claim_unsupported | object_room_mismatch | It is not carpet it is some other type of flooring. But it does need replacing |
| rc_ceeb17f8e242 | condition | terra_claim_overstated | claim_wording | Cabinets are a bit dated but not worth replacing. The countertop is the main priority. But I wouldn't fault the LLM for this. |
| rc_cf0c3ee00fcb | condition | terra_claim_overstated | claim_wording | The green patterned tile flooring appears worn and uneven in places is false. The green tile or vinyl-style floor appears worn and dirty could be true. The main issue is the color of the flooring is just dated and very ugly |
| rc_d36f87e74356 | bathroom | per_bathroom | room_identity | The last image is not a bathroom. Its like a utility room |
| rc_d58b1f530fd9 | condition | terra_claim_supported | claim_wording | Paint isnt peeling. There are dents in the wall. But the damage is substantiated. So I am approving for that reason |
| rc_df5b348c6ea6 | condition | terra_claim_supported | claim_wording | The claim here about the door is understated. The condition is much worse than is conveyed |
| rc_e2ad788f0845 | condition | terra_claim_supported | object_room_mismatch | One note, this is not an image of the house itself. It is a shed |
| rc_e9d27cf6e9aa | condition | terra_claim_supported | claim_wording | A little worn but the best description would be outdated |
| rc_eef1daf73e2a | condition | terra_claim_overstated | claim_wording | I see stains not scuffs |
| rc_f468f4066e3f | condition | terra_claim_supported | claim_wording | True but minor as described |
| rc_f48a8d9f18d3 | condition | terra_evidence_inconclusive | room_identity | The problem is its talking about the bathroom which is barely visible here. This is a photo of a the living room that just happens to show a small part of the bathroom. |
| rc_f8baa9e1dd45 | bathroom | per_bathroom | other | there are 3 bathrooms |
| rc_fb876c9b1b2d | condition | terra_claim_supported | claim_wording | It is overgrown which is fair. But in this case I believe it is intentional. Just to note. The model should not be penalized though. |


*Evidence:* theme counts over the coded free-text notes as tallied in §11 (27 noted cards, 2 distinct tags). Concentration: themes coded by the analysis session; the full note text is shown so the coding is auditable. Basis: exploratory (absence of a note is not evidence a theme is absent). Decision domain: condition prompting, evidence handling, package formation and room identity. No action is prescribed here.

## 12. What this analysis does not show

- Overall Terra recall (miss rate on rejected claims): rejected claims were sampled only through the targeted dirB stratum (2f-confirmed cases); no uniform sample of the rejected population exists.

- Overall Sol or package-pipeline accuracy: the P1 cards are a census of one disagreement slice (2f-rejected, v5 Sol-decided), not a population sample of packages.

- Pass 2f correctness: condition verdicts judge Terra's condition-level claim only; direction A/B are sampling strata, and no code path here joins a condition verdict to a 2f correctness label.

- Causal effects of Terra batch size: batch buckets are observational and confounded with listing photo volume; the breakdowns are associations only.

- Pricing accuracy or renovation cost: prices are provisional and were hidden during review.

- Standalone-observation usefulness: it was not directly labeled; the packaging-flow table is a utilization proxy, not a human judgment of usefulness.


## Appendix A. Per-card reconciliation

212 in-queue cards (all verdicted) + 11 orphaned verdicts (§1.4). Full per-card rows are in reports/review_analysis.json → `per_card`.

| kind | source | cards | verdicted |
|---|---|---|---|
| bathroom | canary | 10 | 10 |
| bathroom | production | 7 | 7 |
| condition | canary | 125 | 125 |
| condition | production | 48 | 48 |
| package | canary | 13 | 13 |
| package | production | 9 | 9 |
