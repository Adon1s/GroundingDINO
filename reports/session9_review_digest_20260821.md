# Session 9 canary review digest

**967 review items** sorted into three tiers:

- **Tier 1 - 736 auto-accepted**, machine-verified no material change. Skim the criteria; no per-item reading needed.
- **Tier 2 - 77 items in 4 POLICY GROUPS**, prefilled as DRAFTS. Read these: each is one real decision, and the dollar effect is stated.
- **Tier 3 - 154 individual reviews**, left blank for you.

Tier 2 drafts are already written into the prefilled review file. Re-run with `--blank-drafts` to leave them empty instead.

---

## TIER 1 - auto-accepted (machine-verified)

### scope_identical_dollars - 116 items
Criterion/explanation: v4 and v5 bill identical dollars for this item (every low/high equal, machine-verified); only the representation label differs (v4 'absorbed_by:<type>' vs the v5 coverage-ledger form). No scope or dollar change.

- e.g. `rar1_e3944bd94ae03bad` redfin_126224899 `appliances_dated_or_basic|kitchen_primary` 
- e.g. `rar1_223620486dd47c4f` redfin_126224899 `bare_or_missing_finish_flooring|bedroom_4` 
- ... and 114 more

### scope_v5_class - 503 items
Criterion/explanation: Catalog item billed by v5 only: it appears in zero v4 scope rows anywhere in the 36-artifact canary corpus. Member of the v5 billing class accepted as intended scope in Session 8 (48-item class decision, 2026-08-16).

- e.g. `rar1_aaf9f08cfdaa849d` redfin_126224899 `baseboard_wear_scuffs|bedroom_4` +$34/170
- e.g. `rar1_bf1bc982abe0b150` redfin_126224899 `baseboard_wear_scuffs|living_room_primary` +$34/170
- ... and 501 more

### scope_rounding_only - 26 items
Criterion/explanation: Same item, same scope; v5 consolidates v4's per-occurrence rows into one work item and the integer allocation rounds differently - low and high each differ by at most $2 (machine-verified). No material change.

- e.g. `rar1_947a0a203786e0bf` redfin_126224899 `older_flooring_style|kitchen_primary` $557/2,889 -> $557/2,888
- e.g. `rar1_1f12ba0c69ae03e8` redfin_126224899 `visible_mold_or_mildew|bedroom_4` $288/2,383 -> $287/2,383
- ... and 24 more

### scope_merge_neutral - 1 items
Criterion/explanation: v5 merges these catalog items into one work item; the merge group's total is within $2 of v4's sum for the same items at the same unit (machine-verified). The apparent per-item change is only the merged work item's full price being listed under each constituent id.

- e.g. `rar1_f50b336034ae2408` redfin_11000447 `vanity_worn_finish|bathroom_primary` merge group ['vanity_damaged_or_water_stained', 'vanity_worn_finish'] @ bathroom_primary: $380/4,563 -> $380/4,563 (+0/+0)

### package_key_migration - 12 items
Criterion/explanation: Same package on both sides: v4 keys packages without an estimate unit ('type|') while v5 keys per unit ('type|unit'), so one package surfaces as a paired removal+addition. Package type and pricing tier match (machine-verified); no package change.

- e.g. `rar1_d70d9dc5f7e9a557` redfin_126224899 `bathroom_modernization|bathroom_primary` pairs with v4 'bathroom_modernization|' (same tier)
- e.g. `rar1_c608a3d3e1dd66d0` redfin_126224899 `bathroom_modernization|` pairs with v5 'bathroom_modernization|bathroom_primary' (same tier)
- ... and 10 more

### package_metadata_only - 78 items
Criterion/explanation: Identical package on both sides: package_type, estimate_unit_id, pricing_tier, and status all equal (machine-verified). The diff is v5-only metadata fields (Sol decision, reason_code) that v4 rows never carried.

- e.g. `rar1_2d6ccc1e8b8643e9` redfin_126224899 `bathroom_repair|bathroom_primary` bathroom_repair|bathroom_primary tier=repair_heavy
- e.g. `rar1_c6788ae3fa47cd70` redfin_126224899 `bedroom_modernization|bedroom_2` bedroom_modernization|bedroom_2 tier=refresh
- ... and 76 more

---

## TIER 2 - policy decisions (DRAFTED - read and approve)

### v5 merge dedup prices shared scope LOWER
**56 review items | corpus effect -13,616 low / -231,270 high**

**The decision:** v5 replaces several stacked v4 rows for the same unit with ONE merged work item, and charges less than v4's sum. This is the negative residual term measured in Session 8 (docs/analysis/session8_floor_vs_sum_trace.md: 'v5 prices shared scope lower'). Accepting means agreeing that one action covering several co-located observations should not stack prices.

Drafted explanation: v5 merges the co-located observations at this unit into one work item rather than stacking a price per observation, so the group totals less than v4's sum. Accepted as the intended dedup behavior (Session 8 floor-vs-sum analysis, 'v5 prices shared scope lower'); the apparent per-item change is the merged work item's full price listed under each constituent catalog id.

- `rar1_abd44d47bfcf24a4` redfin_10803207 merge group ['cabinets_dated_style', 'outdated_kitchen_finishes'] @ kitchen_primary: $1,810/21,722 -> $1,448/14,481 (-362/-7,241)
- `rar1_0ecfee845ae52571` redfin_10952874 merge group ['bath_fixtures_stained_or_worn', 'outdated_bathroom_finishes'] @ bathroom_primary: $1,148/12,825 -> $1,013/10,125 (-135/-2,700)
- `rar1_dce5272371eee987` redfin_10952874 merge group ['cabinets_dated_style', 'outdated_kitchen_finishes'] @ kitchen_primary: $1,688/20,250 -> $1,350/13,500 (-338/-6,750)
- `rar1_430ab8a7d895762d` redfin_11000447 merge group ['cabinets_dated_style', 'cabinets_worn_finish', 'outdated_kitchen_finishes'] @ kitchen_primary: $2,281/30,423 -> $1,521/15,211 (-760/-15,212)
- `rar1_4adc3c076c9e2d4e` redfin_11000447 merge group ['hard_flooring_broken_or_warped', 'hard_flooring_scratched_or_worn', 'older_flooring_style'] @ living_room_primary: $1,034/9,717 -> $555/3,803 (-479/-5,914)
- `rar1_c6c058e72e51e1bd` redfin_11077450 merge group ['hard_flooring_broken_or_warped', 'hard_flooring_scratched_or_worn'] @ basement_primary: $546/5,963 -> $280/4,236 (-266/-1,727)
- `rar1_22672319ec239437` redfin_11185681 merge group ['bath_fixtures_stained_or_worn', 'outdated_bathroom_finishes'] @ bathroom_primary: $1,486/16,614 -> $1,312/13,117 (-174/-3,497)
- `rar1_1a9b8483d372947a` redfin_11185681 merge group ['cabinets_dated_style', 'cabinets_worn_finish', 'outdated_kitchen_finishes'] @ kitchen_primary: $2,186/26,234 -> $1,749/17,489 (-437/-8,745)
- `rar1_d83195ea6fec14b0` redfin_11185681 merge group ['vanity_dated_style', 'vanity_worn_finish'] @ bathroom_primary: $875/10,493 -> $437/5,247 (-438/-5,246)
- `rar1_71c5e31c2e646529` redfin_125779232 merge group ['bath_fixtures_stained_or_worn', 'outdated_bathroom_finishes'] @ bathroom_primary: $1,238/13,838 -> $1,092/10,924 (-146/-2,914)
- `rar1_fc23c8b93fcbd4bc` redfin_125779232 merge group ['cabinets_dated_style', 'cabinets_worn_finish', 'outdated_kitchen_finishes'] @ kitchen_primary: $2,185/29,132 -> $1,457/14,566 (-728/-14,566)
- `rar1_47dfc769fb0f0fde` redfin_125779232 merge group ['hard_flooring_scratched_or_worn', 'older_flooring_style'] @ kitchen_primary: $755/4,874 -> $515/3,641 (-240/-1,233)
- `rar1_2f7c3c8bcbafd39c` redfin_125970550 merge group ['bath_fixtures_stained_or_worn', 'outdated_bathroom_finishes'] @ bathroom_primary: $1,506/16,823 -> $1,328/13,281 (-178/-3,542)
- `rar1_80706d877ec21024` redfin_125970550 merge group ['cabinets_damaged_or_water_stained', 'cabinets_worn_finish', 'outdated_kitchen_finishes'] @ kitchen_primary: $2,657/35,416 -> $1,771/17,708 (-886/-17,708)
- `rar1_d7df8e6561f10a84` redfin_125970550 merge group ['hard_flooring_broken_or_warped', 'hard_flooring_scratched_or_worn'] @ living_room_primary: $571/6,231 -> $301/4,427 (-270/-1,804)
- `rar1_53d21b94e09c85e5` redfin_125970550 merge group ['missing_vanity_exposed_plumbing', 'vanity_dated_style', 'vanity_worn_finish'] @ bathroom_primary: $1,593/17,707 -> $708/7,083 (-885/-10,624)
- `rar1_c31abb895d0a7da9` redfin_126224899 merge group ['bath_fixtures_stained_or_worn', 'outdated_bathroom_finishes'] @ bathroom_primary: $1,340/14,967 -> $1,182/11,816 (-158/-3,151)
- `rar1_99820dba42667dd3` redfin_126224899 merge group ['cabinets_dated_style', 'cabinets_worn_finish', 'outdated_kitchen_finishes'] @ kitchen_primary: $2,363/31,509 -> $1,575/15,755 (-788/-15,754)
- `rar1_08e3dece9e367b7f` redfin_126224899 merge group ['dated_bathroom_flooring_style', 'older_flooring_style'] @ bathroom_primary: $793/5,646 -> $557/2,889 (-236/-2,757)
- `rar1_4ed4cec9d4e9d0fe` redfin_126224899 merge group ['hard_flooring_scratched_or_worn', 'older_flooring_style'] @ living_room_primary: $792/6,827 -> $556/3,939 (-236/-2,888)
- `rar1_82b78c325781d19a` redfin_126224899 merge group ['vanity_countertop_dated', 'vanity_countertop_worn'] @ bathroom_primary: $392/3,939 -> $197/1,969 (-195/-1,970)
- `rar1_bad81a78f3f76622` redfin_126224899 merge group ['vanity_dated_style', 'vanity_worn_finish'] @ bathroom_primary: $788/9,452 -> $394/4,726 (-394/-4,726)
- `rar1_a251db302068845b` redfin_126418713 merge group ['cabinets_dated_style', 'outdated_kitchen_finishes'] @ kitchen_primary: $1,832/21,987 -> $1,466/14,658 (-366/-7,329)
- `rar1_aaa0c28edd577bd7` redfin_127468088 merge group ['cabinets_dated_style', 'cabinets_worn_finish', 'outdated_kitchen_finishes'] @ kitchen_primary: $2,245/29,945 -> $1,497/14,973 (-748/-14,972)
- `rar1_9a171c080fac5702` redfin_166147710 merge group ['bath_fixtures_stained_or_worn', 'outdated_bathroom_finishes'] @ bathroom_primary: $1,300/14,529 -> $1,147/11,470 (-153/-3,059)
- `rar1_3fe4dfa1d6ac93be` redfin_166147710 merge group ['cabinets_dated_style', 'outdated_kitchen_finishes'] @ kitchen_primary: $1,911/22,940 -> $1,529/15,293 (-382/-7,647)
- `rar1_eb935f3d3eb9304a` redfin_25809814 merge group ['cabinets_dated_style', 'outdated_kitchen_finishes'] @ kitchen_primary: $2,158/25,896 -> $1,726/17,264 (-432/-8,632)
- `rar1_640fc928e0314468` redfin_80877597 merge group ['cabinets_dated_style', 'outdated_kitchen_finishes'] @ kitchen_primary: $1,871/22,451 -> $1,497/14,967 (-374/-7,484)
- `rar1_dbb7366ee2d8c999` redfin_80925528 merge group ['cabinets_dated_style', 'outdated_kitchen_finishes'] @ kitchen_primary: $2,500/30,000 -> $2,000/20,000 (-500/-10,000)
- `rar1_3a5ffcbe9e425fcd` redfin_80925528 merge group ['hard_flooring_scratched_or_worn', 'older_flooring_style'] @ bedroom_1: $1,037/6,692 -> $707/3,667 (-330/-3,025)
- `rar1_75626a931e11bc7e` redfin_80925528 merge group ['hard_flooring_scratched_or_worn', 'older_flooring_style'] @ living_room_primary: $1,036/6,691 -> $706/3,666 (-330/-3,025)
- `rar1_6305385bb400dea9` redfin_80990371 merge group ['bath_fixtures_stained_or_worn', 'outdated_bathroom_finishes'] @ bathroom_primary: $1,272/14,209 -> $1,122/11,218 (-150/-2,991)
- `rar1_7a22cf612ef9fcdc` redfin_80990371 merge group ['cabinets_damaged_or_water_stained', 'cabinets_dated_style', 'cabinets_worn_finish', 'outdated_kitchen_finishes'] @ kitchen_primary: $2,618/37,394 -> $1,496/14,957 (-1,122/-22,437)

### single-item repricing (v5 range wider at the high end)
**16 review items | corpus effect -7 low / +18,712 high**

**The decision:** Same item, same unit, same scope, not merged - v5 simply prices it differently. The observed signature is a near-unchanged low and a materially higher high, concentrated in area-priced surface items (flooring, carpet, ceiling). Accepting means endorsing v5's price range for these items.

Drafted explanation: Same item, unit, and scope on both sides; v5's price range is wider at the high end while the low is essentially unchanged. Accepted as v5 pricing-model output for area-priced surface work; no scope change.

- `rar1_503be2484023380b` redfin_10803207 $239/2,190 -> $217/3,620 (-22/+1,430)
- `rar1_054e549b9e750097` redfin_10952874 $223/2,042 -> $202/3,375 (-21/+1,333)
- `rar1_f5d00044a00943b8` redfin_11077450 $480/3,191 -> $424/6,777 (-56/+3,586)
- `rar1_6e0df1527941e4f1` redfin_11077450 $292/1,726 -> $279/2,562 (-13/+836)
- `rar1_4dee5c78c100108a` redfin_11185681 $618/3,207 -> $639/4,154 (+21/+947)
- `rar1_c8d75c4c7c23cf0d` redfin_11185681 $618/3,207 -> $638/4,153 (+20/+946)
- `rar1_c7fca07c65a0dbd3` redfin_125970550 $305/1,804 -> $301/2,096 (-4/+292)
- `rar1_8334921efbae9d0a` redfin_125970550 $305/1,804 -> $301/2,095 (-4/+291)
- `rar1_1017ffde3ab71358` redfin_125970550 $323/2,678 -> $354/4,427 (+31/+1,749)
- `rar1_e417ab886e45f1a9` redfin_126418713 $249/1,735 -> $242/2,217 (-7/+482)
- `rar1_b0c03215a9c81743` redfin_126418713 $249/1,734 -> $242/2,217 (-7/+483)
- `rar1_d7566d1e047a8446` redfin_127468088 $210/1,218 -> $214/1,569 (+4/+351)
- `rar1_6dfc91307e307d3a` redfin_127468088 $210/1,217 -> $213/1,568 (+3/+351)
- `rar1_3cffec5af5dcce7b` redfin_81000709 $280/2,570 -> $255/4,249 (-25/+1,679)
- `rar1_eaa2ed29d6c02301` redfin_81000709 $620/4,036 -> $680/6,798 (+60/+2,762)
- `rar1_952c2bd7c7d5a696` redfin_81000709 $242/1,780 -> $255/2,974 (+13/+1,194)

### v4 carried the item at $0, v5 prices it
**5 review items | corpus effect +1,751 low / +20,946 high**

**The decision:** v4 has a scope row for this item but at $0/$0 (withheld or evidence-gated); v5 assigns a real price. Accepting means v5 is right to price scope v4 acknowledged but never costed.

Drafted explanation: v4 carried this item at $0/$0 (withheld / evidence-gated) while recognizing the scope; v5 prices it. Accepted: the item is real scope and pricing it is the intended v5 behavior.

- `rar1_32834e11fec24cbe` redfin_10803207 v4 $0/$0 -> $217/2,534
- `rar1_de0247255b9e220e` redfin_10803207 v4 $0/$0 -> $362/4,344
- `rar1_7a69e9ae5fbabd7f` redfin_126418713 v4 $0/$0 -> $366/4,397
- `rar1_afda9927a9027fb6` redfin_127468088 v4 $0/$0 -> $374/4,492
- `rar1_69d1a08831644ff0` redfin_25809814 v4 $0/$0 -> $432/5,179

---

## TIER 3 - individual review (blank)

### scope_merge_increased - 8 items | corpus effect +2,863 low / +16,090 high

- `rar1_abe5851ee2e65160` redfin_10806500 `bath_fixtures_stained_or_worn|bathroom_primary` merge group ['bath_fixtures_stained_or_worn', 'outdated_bathroom_finishes'] @ bathroom_primary: $139/2,771 -> $1,039/10,390 (+900/+7,619)
- `rar1_122f95291f5a6bef` redfin_10806500 `cabinets_dated_style|kitchen_primary` merge group ['cabinets_dated_style', 'cabinets_worn_finish', 'outdated_kitchen_finishes'] @ kitchen_primary: $692/13,854 -> $1,385/13,853 (+693/-1)
- `rar1_1a48b1b6c3c454c3` redfin_11077450 `cabinets_dated_style|kitchen_primary` merge group ['cabinets_dated_style', 'cabinets_worn_finish', 'outdated_kitchen_finishes'] @ kitchen_primary: $424/8,472 -> $1,694/16,944 (+1,270/+8,472)

### scope_added_reattributed - 12 items | corpus effect +7,993 low / +98,672 high

- `rar1_b5db2add1697d6eb` redfin_11185681 `roof_shingles_aged_or_worn|property` +$437/8,745; v4 bills this id at another unit
- `rar1_446fae5742303bd0` redfin_125779232 `damaged_or_rotted_siding_or_trim|area` +$364/5,826; v4 bills this id at another unit
- `rar1_2e925d2f2dd8d49d` redfin_125779232 `roof_shingles_aged_or_worn|property` +$364/7,283; v4 bills this id at another unit
- `rar1_4919a48d9ef7f59c` redfin_126224899 `roof_shingles_aged_or_worn|property` +$394/7,877; v4 bills this id at another unit
- `rar1_8c5d931b1edcf1d9` redfin_126418713 `roof_shingles_aged_or_worn|property` +$366/7,329; v4 bills this id at another unit
- `rar1_4d52e0d3325ce8f4` redfin_127468088 `roof_shingles_aged_or_worn|property` +$374/7,486; v4 bills this id at another unit
- `rar1_69583566af830b76` redfin_166147710 `roof_shingles_aged_or_worn|property` +$382/7,647; v4 bills this id at another unit
- `rar1_2cc214f7431fe354` redfin_80925528 `damaged_or_rotted_siding_or_trim|area` +$500/8,000; v4 bills this id at another unit
- `rar1_22a6d68d245dbf84` redfin_80925528 `roof_shingles_aged_or_worn|property` +$500/10,000; v4 bills this id at another unit
- `rar1_5e5962f641231530` redfin_80990371 `boarded_up_entry_or_window|exterior_primary` +$1,969/10,500; v4 bills this id at another unit
- `rar1_d376939457725f11` redfin_80990371 `boarded_up_entry_or_window|living_room_primary` +$1,969/10,500; v4 bills this id at another unit
- `rar1_ec9fb2ff6ee774fb` redfin_80990371 `roof_shingles_aged_or_worn|property` +$374/7,479; v4 bills this id at another unit

### scope_dropped_disposition - 27 items | corpus effect -6,243 low / -79,371 high

- `rar1_70cdcd080113e4f5` redfin_10952874 `cabinets_worn_finish|kitchen_primary` -$338/6,750; v5 disposition(s): ['excluded']
- `rar1_b003fe48ae738284` redfin_10952874 `damaged_or_rotted_siding_or_trim|area` -$0/0; v5 disposition(s): ['inspection']
- `rar1_047565c0d289820b` redfin_10952874 `hard_flooring_broken_or_warped|bedroom_3` -$202/3,375; v5 disposition(s): ['excluded']
- `rar1_be9d68c88ac9e29b` redfin_10952874 `vanity_countertop_worn|bathroom_primary` -$169/1,688; v5 disposition(s): ['excluded']
- `rar1_98ef738afedd02f5` redfin_10952874 `vanity_damaged_or_water_stained|bathroom_primary` -$0/0; v5 disposition(s): ['excluded']
- `rar1_a2bbb1c2e3ca9264` redfin_10952874 `vanity_worn_finish|bathroom_primary` -$338/4,050; v5 disposition(s): ['excluded']
- `rar1_baae71b693acf271` redfin_11077450 `damaged_or_rotted_siding_or_trim|exterior_primary` -$424/6,777; v5 disposition(s): ['excluded']
- `rar1_1f3549fc4caaee3b` redfin_11079485 `vanity_damaged_or_water_stained|bathroom_primary` -$0/0; v5 disposition(s): ['excluded']
- `rar1_ee1caf0fe0b3e080` redfin_11079485 `vanity_worn_finish|bathroom_primary` -$0/0; v5 disposition(s): ['excluded']
- `rar1_ac84e4ed26bf5743` redfin_11185681 `bare_or_missing_finish_flooring|attic_primary` -$700/5,247; v5 disposition(s): ['excluded']
- `rar1_4b920ac7993723c6` redfin_11185681 `cabinets_damaged_or_water_stained|kitchen_primary` -$0/0; v5 disposition(s): ['excluded']
- `rar1_ad70c25996d35a97` redfin_125779232 `cabinets_damaged_or_water_stained|kitchen_primary` -$364/7,283; v5 disposition(s): ['excluded']
- `rar1_0c406f8397b5502d` redfin_125779232 `exposed_sheathing_or_missing_siding|exterior_primary` -$0/0; v5 disposition(s): ['excluded']
- `rar1_0713595148c131ab` redfin_125779232 `retaining_wall_failure_or_missing_section|exterior_primary` -$146/583; v5 disposition(s): ['inspection']
- `rar1_222252dfbbba2624` redfin_125970550 `roof_shingles_aged_or_worn|exterior_primary` -$443/8,854; v5 disposition(s): ['inspection']
- `rar1_4f08d530d58553ec` redfin_125970550 `tub_surround_or_shower_pan_damage|bathroom_primary` -$443/5,312; v5 disposition(s): ['excluded']
- `rar1_ebb4d494d6531bcb` redfin_126224899 `vanity_damaged_or_water_stained|bathroom_primary` -$0/0; v5 disposition(s): ['excluded']
- `rar1_eb56595b1eb8ed21` redfin_127468088 `major_foundation_or_settlement_signs|exterior_primary` -$150/599; v5 disposition(s): ['excluded']
- `rar1_1cbe21ebb8e15ed3` redfin_166147710 `missing_base_cabinets_exposed_subfloor|kitchen_primary` -$0/0; v5 disposition(s): ['excluded']
- `rar1_ca73bd7ad1f82d26` redfin_25809814 `cabinets_worn_finish|kitchen_primary` -$432/8,632; v5 disposition(s): ['excluded']
- `rar1_2010f2fe6cb278a6` redfin_25809814 `hard_flooring_scratched_or_worn|dining_room_primary` -$0/0; v5 disposition(s): ['excluded']
- `rar1_500d570063938501` redfin_80877597 `bath_fixtures_stained_or_worn|bathroom_primary` -$0/0; v5 disposition(s): ['excluded']
- `rar1_b37d9ba3b4306aa5` redfin_80925528 `ceiling_cracks_or_sagging|living_room_primary` -$500/8,000; v5 disposition(s): ['excluded']
- `rar1_4fb4f333e29f4129` redfin_80925528 `missing_base_cabinets_exposed_subfloor|kitchen_primary` -$0/0; v5 disposition(s): ['excluded']
- `rar1_c8226e2ba3afc4d0` redfin_80925528 `retaining_wall_failure_or_missing_section|exterior_primary` -$200/800; v5 disposition(s): ['inspection']
- `rar1_81626cbda14b6745` redfin_80990371 `ceiling_cracks_or_sagging|living_room_primary` -$374/5,983; v5 disposition(s): ['excluded']
- `rar1_ccc2e90c269c85f0` redfin_81000709 `interior_wall_stripped_to_studs|living_room_primary` -$1,020/5,438; v5 disposition(s): ['excluded']

### scope_dropped_other - 28 items | corpus effect -11,702 low / -129,331 high

- `rar1_c7414f7c3877b7e2` redfin_10803207 `hard_flooring_scratched_or_worn|living_room_primary` -$239/2,190
- `rar1_a820d8f743c2b58f` redfin_10952874 `hard_flooring_scratched_or_worn|bathroom_primary` -$223/2,042
- `rar1_763291344d2bc15b` redfin_11077450 `ceiling_cracks_or_sagging|basement_primary` -$480/3,191
- `rar1_405cca52c0523d34` redfin_11077450 `ceiling_cracks_or_sagging|unspecified` -$480/3,190
- `rar1_1356d2bc7e60bb5f` redfin_11077450 `hard_flooring_scratched_or_worn|bedroom_1` -$292/1,727
- `rar1_2c594eada85a65d2` redfin_11077450 `hard_flooring_scratched_or_worn|kitchen_primary` -$292/1,726
- `rar1_0850bc92b9411452` redfin_11185681 `older_flooring_style|bedroom_5` -$617/3,206
- `rar1_5c1e95245745291f` redfin_11185681 `roof_shingles_aged_or_worn|exterior_primary` -$437/8,745
- `rar1_f193ff3a42b1a84a` redfin_125779232 `damaged_or_rotted_siding_or_trim|exterior_primary` -$364/5,826
- `rar1_634c9527e14116cb` redfin_125779232 `hard_flooring_scratched_or_worn|basement_primary` -$240/2,203
- `rar1_d40b985bb0775fe5` redfin_125779232 `roof_shingles_aged_or_worn|exterior_primary` -$364/7,283
- `rar1_2286bddf1dcab274` redfin_125970550 `damaged_or_rotted_siding_or_trim|exterior_primary` -$0/0
- `rar1_b5235251f317f487` redfin_125970550 `hard_flooring_broken_or_warped|bathroom_primary` -$305/1,804
- `rar1_66774fdf42344bad` redfin_125970550 `visible_mold_or_mildew|bedroom_1` -$323/2,678
- `rar1_60576701b2eb9d73` redfin_126224899 `roof_shingles_aged_or_worn|exterior_primary` -$394/7,877
- `rar1_a442740b558ef398` redfin_126418713 `hard_flooring_scratched_or_worn|bedroom_4` -$249/1,735
- `rar1_45856a33d65fc4be` redfin_126418713 `older_flooring_style|kitchen_primary` -$586/5,863
- `rar1_033c0c78cce9756f` redfin_126418713 `roof_shingles_aged_or_worn|exterior_primary` -$366/7,329
- `rar1_81860b137a1233f9` redfin_127468088 `roof_shingles_aged_or_worn|exterior_primary` -$374/7,486
- `rar1_86a2fafb07900582` redfin_127468088 `worn_or_stained_carpet|bedroom_2` -$210/1,218
- `rar1_ae63c19c8802e97d` redfin_166147710 `roof_shingles_aged_or_worn|exterior_primary` -$382/7,647
- `rar1_396b1799d5a99260` redfin_80925528 `damaged_or_rotted_siding_or_trim|exterior_primary` -$500/8,000
- `rar1_a99cc4b70ff28454` redfin_80925528 `roof_shingles_aged_or_worn|exterior_primary` -$500/10,000
- `rar1_5bd80350e422e567` redfin_80990371 `boarded_up_entry_or_window|unspecified` -$1,969/10,500
- `rar1_ce4fe3709899e20f` redfin_80990371 `roof_shingles_aged_or_worn|exterior_primary` -$374/7,479
- `rar1_c958e2a84ddb4596` redfin_81000709 `hard_flooring_scratched_or_worn|basement_primary` -$280/2,570
- `rar1_8c25503fde4abc19` redfin_81000709 `older_flooring_style|bedroom_3` -$620/4,036
- `rar1_7a2b13b549e81d0c` redfin_81000709 `worn_or_stained_carpet|bedroom_3` -$242/1,780

### package_needs_eyes - 47 items

- `rar1_114e6c1d7930bf30` redfin_10803207 `bathroom_modernization|bathroom_primary` tier=partial_rehab reason=approved_absorbs_children (no v4 package of this type)
- `rar1_f290971b6d281e87` redfin_10806500 `exterior_repair|exterior_primary` tier=repair_light reason=approved_absorbs_children (no v4 package of this type)
- `rar1_287b03ff2a9d42a8` redfin_10806500 `kitchen_modernization|kitchen_primary` tier=full_rehab reason=approved_absorbs_children (no v4 package of this type)
- `rar1_0dfc605101438fd8` redfin_10952874 `bathroom_repair|bathroom_primary` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_9f95f0ec81033f19` redfin_10952874 `bedroom_repair|bedroom_3` changed in place (tier/decision/status)
- `rar1_9d44d8abb7f96cea` redfin_10952874 `kitchen_modernization|kitchen_primary` changed in place (tier/decision/status)
- `rar1_5951f434730ed4f9` redfin_11000447 `bathroom_repair|bathroom_primary` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_2c1686418e1eda35` redfin_11000447 `bedroom_modernization|bedroom_1` changed in place (tier/decision/status)
- `rar1_9e5c8459767428c2` redfin_11077450 `kitchen_modernization|kitchen_primary` tier=full_rehab reason=approved_absorbs_children (no v4 package of this type)
- `rar1_ea78cb35f175ffdd` redfin_11077450 `living_repair|living_room_primary` changed in place (tier/decision/status)
- `rar1_ba8e3be7e76a920c` redfin_11079485 `bathroom_repair|bathroom_primary` changed in place (tier/decision/status)
- `rar1_46149821d3cc1190` redfin_11079485 `kitchen_modernization|kitchen_primary` changed in place (tier/decision/status)
- `rar1_c66a850c5fd2ffb1` redfin_11185681 `bedroom_modernization|bedroom_5` changed in place (tier/decision/status)
- `rar1_b0bf92d062e512ef` redfin_11185681 `bedroom_repair|bedroom_5` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_b8e83bcbb3c56d79` redfin_11185681 `exterior_repair|exterior_primary` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_f0e9d2764645de98` redfin_11185681 `kitchen_modernization|kitchen_primary` tier=full_rehab reason=approved_absorbs_children (no v4 package of this type)
- `rar1_449b6653ce62cf62` redfin_125779232 `bathroom_repair|bathroom_primary` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_8dc6bb567adeaace` redfin_125970550 `bedroom_repair|bedroom_3` changed in place (tier/decision/status)
- `rar1_8f3728ce35d69941` redfin_125970550 `exterior_repair|exterior_primary` tier=repair_heavy reason=decision_rejected (no v4 package of this type)
- `rar1_f19c3c0d70de8c9c` redfin_126224899 `bedroom_modernization|closet` tier=refresh reason=approved_absorbs_children (no v4 package of this type)
- `rar1_7e137d0c54325859` redfin_126224899 `bedroom_modernization|worn_or_stained_carpet:catalogworn_or_stained_carpetscene_groupbedroomroomcloset` changed in place (tier/decision/status)
- `rar1_dbe3e1b786a7ee77` redfin_126224899 `exterior_repair|exterior_primary` changed in place (tier/decision/status)
- `rar1_dd10d212df74f674` redfin_126224899 `living_modernization|living_room_primary` changed in place (tier/decision/status)
- `rar1_6f817ea3717eed8a` redfin_126418713 `bedroom_modernization|bedroom_2` changed in place (tier/decision/status)
- `rar1_ab72cfb517ffa88a` redfin_126418713 `bedroom_modernization|bedroom_4` changed in place (tier/decision/status)
- `rar1_7890067afe85d5fa` redfin_126418713 `bedroom_repair|bedroom_4` changed in place (tier/decision/status)
- `rar1_5f8a8c2d3da296a6` redfin_126418713 `exterior_repair|exterior_primary` tier=repair_light reason=decision_rejected (no v4 package of this type)
- `rar1_ce8784145a70b51c` redfin_127468088 `bedroom_repair|bedroom_3` tier=repair_light reason=approved_absorbs_children (no v4 package of this type)
- `rar1_558a3d4be8d89519` redfin_127468088 `exterior_repair|exterior_primary` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_ffd6869d572ed9d8` redfin_166147710 `bedroom_modernization|bedroom_3` changed in place (tier/decision/status)
- `rar1_eff9f292ae4e5fa1` redfin_166147710 `exterior_repair|exterior_primary` changed in place (tier/decision/status)
- `rar1_798478f4fcdc2641` redfin_166147710 `living_repair|living_room_primary` changed in place (tier/decision/status)
- `rar1_076165dc73cc2ae3` redfin_25809814 `bathroom_repair|bathroom_primary` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_1fe6936fea21c633` redfin_25809814 `bedroom_modernization|bedroom_2` tier=full_rehab reason=approved_absorbs_children (no v4 package of this type)
- `rar1_f33e82acfba2f4b3` redfin_25809814 `bedroom_modernization|bedroom_3` tier=full_rehab reason=approved_absorbs_children (no v4 package of this type)
- `rar1_6c81f2e277107b8f` redfin_25809814 `kitchen_modernization|kitchen_primary` tier=partial_rehab reason=approved_absorbs_children (no v4 package of this type)
- `rar1_b1812216f97da653` redfin_80925528 `bedroom_repair|bedroom_1` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_77587d3ab5268719` redfin_80990371 `bathroom_repair|bathroom_primary` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_3e7e349f18ffdd66` redfin_80990371 `bedroom_repair|bedroom_1` tier=repair_light reason=approved_absorbs_children (no v4 package of this type)
- `rar1_3a5a9660786e095e` redfin_80990371 `bedroom_repair|closet` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_f68f026722242257` redfin_80990371 `bedroom_repair|damaged_drywall_or_cracks:catalogdamaged_drywall_or_cracksscene_groupbedroomroomcloset` changed in place (tier/decision/status)
- `rar1_1c5f0b74145cd943` redfin_80990371 `bedroom_repair|hard_flooring_scratched_or_worn:cataloghard_flooring_scratched_or_wornscene_groupbedroomroomcloset` changed in place (tier/decision/status)
- `rar1_1c2f9ffc27016e94` redfin_80990371 `exterior_repair|exterior_primary` tier=repair_light reason=approved_absorbs_children (no v4 package of this type)
- `rar1_5256bb8362e75619` redfin_81000709 `bedroom_modernization|bedroom_3` changed in place (tier/decision/status)
- `rar1_1c80aecf03a001e4` redfin_81000709 `bedroom_repair|bedroom_1` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_ffd40cdb6bef96be` redfin_81000709 `bedroom_repair|bedroom_3` tier=repair_heavy reason=approved_absorbs_children (no v4 package of this type)
- `rar1_613c84918789ea62` redfin_81000709 `exterior_repair|exterior_primary` changed in place (tier/decision/status)

### headline - 14 items

- `rar1_eed2e48618a86ea7` redfin_10803207 `headline` $14,438/47,625 -> $20,075/57,189 (+39%/+20%)
- `rar1_9f1bed67e6bdb231` redfin_10806500 `headline` $3,775/36,849 -> $23,903/74,107 (+533%/+101%)
- `rar1_8f621b8cf1999e7e` redfin_10952874 `headline` $37,487/121,451 -> $29,653/112,144 (-21%/-8%)
- `rar1_224e2c089781bbba` redfin_11077450 `headline` $15,207/74,554 -> $38,895/132,175 (+156%/+77%)
- `rar1_d571cb97fb80ef64` redfin_11079485 `headline` $16,404/50,261 -> $7,896/60,141 (-52%/+20%)
- `rar1_0d4ef0697bf5bf2d` redfin_11185681 `headline` $42,271/170,376 -> $59,305/201,834 (+40%/+18%)
- `rar1_56aa0653723ad643` redfin_125779232 `headline` $42,296/118,084 -> $40,089/138,281 (-5%/+17%)
- `rar1_8b8f6af8e1f8d4fe` redfin_126418713 `headline` $31,148/96,743 -> $24,374/89,619 (-22%/-7%)
- `rar1_4a19709a2f04d31c` redfin_127468088 `headline` $35,890/98,542 -> $42,936/143,957 (+20%/+46%)
- `rar1_130ac72c3127ce20` redfin_166147710 `headline` $30,158/102,095 -> $30,485/136,689 (+1%/+34%)
- `rar1_66d0ff48643916f2` redfin_25809814 `headline` $11,295/59,130 -> $33,317/94,326 (+195%/+60%)
- `rar1_907cf475274a84b9` redfin_80877597 `headline` $29,716/91,034 -> $25,166/82,388 (-15%/-9%)
- `rar1_afe2778dc5b28366` redfin_80925528 `headline` $30,340/108,667 -> $34,806/133,180 (+15%/+23%)
- `rar1_571226b85e6fa906` redfin_80990371 `headline` $43,109/144,381 -> $50,899/186,831 (+18%/+29%)

### stability - 18 items

- `rar1_1b0255fb8a82cef7` redfin_10803207 `v5` 
- `rar1_17bb113619980309` redfin_10806500 `v5` 
- `rar1_9040ae52dae8e786` redfin_10952874 `v5` 
- `rar1_0c24d071b75723e8` redfin_11000447 `v5` 
- `rar1_8fe3314dda5678aa` redfin_11077450 `v5` 
- `rar1_eb726d82a716906a` redfin_11079485 `v5` 
- `rar1_265e8339912dca93` redfin_11185681 `v5` 
- `rar1_e672b39990b8c5d1` redfin_125779232 `v5` 
- `rar1_f7df347fe03283b1` redfin_125970550 `v5` 
- `rar1_4e3aee24ff4a6b6f` redfin_126224899 `v5` 
- `rar1_af23f8d488094ada` redfin_126418713 `v5` 
- `rar1_101a01fc0bf5047f` redfin_127468088 `v5` 
- `rar1_5e0b1b54b05d8fc8` redfin_166147710 `v5` 
- `rar1_5c3b9bc516f436da` redfin_25809814 `v5` 
- `rar1_c5bb260c7593140d` redfin_80877597 `v5` 
- `rar1_9bf5b757fe7c37e7` redfin_80925528 `v5` 
- `rar1_46d433f275afcad7` redfin_80990371 `v5` 
- `rar1_2d39b5f76551cd5e` redfin_81000709 `v5` 
