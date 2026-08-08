# Kind cutover — per-property headline-delta review sheet

2026-08-08. Evidence sheet for the 4A sign-off checklist item 2 (`approved_headline_deltas` in configs/kind_ontology_cutover.json). Drivers are raw line-item sums from renovation_estimate_v4 in both canary artifacts — they explain direction and magnitude but are pre-package-adjustment, so they do not sum exactly to final_rehab. Kind column = v1 -> v2 catalog kind from the migration map (upgrade->degradation items reprice x0.6 -> x1.0).

**Revised 2026-08-08 (catalog 3.1 re-measure).** `bathroom_layout_modernization_opportunity` and `layout_modernization_opportunity` were retired from the v2 catalog (3.0 → 3.1; layout claims must not price). The 5 candidate properties that had resolved the bathroom item (10806500, 11000447, 11077450, 126224899, 25809814) were re-run end-to-end under 3.1 and their sections below are refreshed from `reports/kind_cutover_canary_catalog31_20260808.json`; pre-3.1 candidate artifacts are preserved in `artifacts_canary/candidate_archive_pre_v31/`. The other 13 properties' deltas stand as measured. A rerun is a fresh VLM measurement, so the 5 refreshed sections move on other drivers too, not only the retired item. Note the baseline (legacy v1) still contains the layout items until Task 4C, so 126224899 shows the retired item as a negative driver.

## Systemic patterns (read before the per-property tables)

1. **RESOLVED — `bathroom_layout_modernization_opportunity` retired (catalog 3.1).**
   Previously the largest single upward driver: newly added on 4 properties at
   $5,000–$25,000 each ($100,000 of combined high-end). The item (and its
   cross-room sibling `layout_modernization_opportunity`) was retired from the
   catalog on 2026-08-08 — layout claims must not price, independent of the
   Pass 2c v3 prompt. The 5 affected candidates were re-measured under 3.1:
   layout observations now go unresolved instead of pricing, 10806500's breach
   fully resolved (-2%/+9%), and the 11077450 / 25809814 / 126224899 high
   bounds fell to -5% / -4% / +3%. Residual breaches on those properties are
   low-bound and driven by other movement (see sections). 11000447 re-measured
   higher on non-layout drivers (fresh-run variance) and remains a breach.
   Substitution risk was checked: layout observations that fall into
   `outdated_bathroom_finishes` / `outdated_kitchen_finishes` add $0 (flat
   room allowances, and every canary instance also has non-spatial evidence).
2. **Split-successor stacking**: v1 single items (e.g. `outdated_or_damaged_cabinets`,
   $500–$10,000) split into worn-finish (degradation) + dated-style
   (modernization) successors that *both* price on the same room — cabinets on
   10 properties, vanity on 6, appliances on 3, roughly doubling those lines
   ($1,000–$20,000 combined). This is the intended ontology (wear and datedness
   are separate claims) but it is a deliberate pricing-posture question: does one
   renovation fix both? If both-stack is not intended, that is a costing/package
   rule to add — not a reason to reject the classification.
3. **Like-for-like repricing** (wear items upgrade→degradation, x0.6 → x1.0)
   and **repair↔modernization package lane swaps** account for most of the
   remaining movement and match the approved reclassification policy.

## redfin_10806500

Headline: $20,000–$63,455 → $19,650–$69,200 (low -2%, high +9%)
Packages: unchanged

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |
| outdated_or_damaged_cabinets | defect→defect | removed | $500–$10,000 | $0–$0 | $-10,000 |
| older_flooring_style | upgrade→modernization | added | $0–$0 | $800–$8,000 | $8,000 |
| appliances_dated_or_basic | defect→modernization | added | $0–$0 | $500–$6,000 | $6,000 |
| outdated_or_damaged_vanity | defect→defect | removed | $500–$6,000 | $0–$0 | $-6,000 |
| vanity_damaged_or_water_stained | defect→defect | added | $0–$0 | $500–$6,000 | $6,000 |
| visible_mold_or_mildew | defect→defect | removed | $400–$5,000 | $0–$0 | $-5,000 |
| bath_fixtures_stained_or_worn | defect→degradation | added | $0–$0 | $200–$4,000 | $4,000 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_11000447  — BREACH (high_delta_pct, low_delta_pct)

Headline: $25,590–$78,570 → $49,500–$140,500 (low +93%, high +79%)
Packages: -['bathroom_repair__bathroom_primary', 'bedroom_repair__bedroom_2'] +['bathroom_modernization__bathroom_primary', 'bedroom_repair__bedroom_3']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| outdated_bathroom_finishes | upgrade→modernization | added | $0–$0 | $1,500–$15,000 | $15,000 |
| cabinets_damaged_or_water_stained | defect→defect | added | $0–$0 | $500–$10,000 | $10,000 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |
| cabinets_worn_finish | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| older_flooring_style | upgrade→modernization | added | $0–$0 | $800–$8,000 | $8,000 |
| scratched_or_damaged_flooring | defect→defect | removed | $660–$6,050 | $0–$0 | $-6,050 |
| outdated_or_damaged_vanity | defect→defect | removed | $500–$6,000 | $0–$0 | $-6,000 |
| vanity_damaged_or_water_stained | defect→defect | added | $0–$0 | $500–$6,000 | $6,000 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_25809814  — BREACH (low_delta_pct)

Headline: $23,800–$70,000 → $13,252–$67,500 (low -44%, high -4%)
Packages: -['kitchen_modernization__kitchen_primary'] +[]

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |
| older_flooring_style | upgrade→modernization | repriced | $800–$8,000 | $3,440–$14,000 | $6,000 |
| outdated_or_damaged_vanity | defect→defect | removed | $500–$6,000 | $0–$0 | $-6,000 |
| vanity_worn_finish | defect→degradation | added | $0–$0 | $500–$6,000 | $6,000 |
| dated_bathroom_flooring_style | upgrade→modernization | removed | $300–$3,500 | $0–$0 | $-3,500 |
| dated_or_worn_vanity_countertop | upgrade→degradation | removed | $250–$2,500 | $0–$0 | $-2,500 |
| vanity_countertop_dated | upgrade→modernization | added | $0–$0 | $250–$2,500 | $2,500 |
| major_foundation_or_settlement_signs | defect→defect | added | $0–$0 | $200–$800 | $800 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_80925528  — BREACH (high_delta_pct, low_delta_pct)

Headline: $28,350–$94,000 → $44,840–$131,000 (low +58%, high +39%)
Packages: -['bedroom_repair__bedroom_1'] +['bedroom_modernization__bedroom_1']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| roof_shingles_aged_or_worn | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| damaged_or_aged_roof_shingles | defect→defect | removed | $500–$10,000 | $0–$0 | $-10,000 |
| cabinets_worn_finish | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |
| scratched_or_damaged_flooring | defect→defect | removed | $660–$6,050 | $0–$0 | $-6,050 |
| hard_flooring_scratched_or_worn | defect→degradation | added | $0–$0 | $660–$6,050 | $6,050 |
| appliances_worn_or_neglected | defect→degradation | added | $0–$0 | $500–$6,000 | $6,000 |
| appliances_dated_or_basic | defect→modernization | added | $0–$0 | $500–$6,000 | $6,000 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_80990371  — BREACH (high_delta_pct, low_delta_pct)

Headline: $42,843–$148,995 → $60,488–$202,178 (low +41%, high +36%)
Packages: -['bedroom_modernization__bedroom_2'] +['bedroom_repair__damaged_drywall_or_cracks:catalogdamaged_drywall_or_cracksscene_groupbedroomroomcloset', 'bedroom_repair__hard_flooring_scratched_or_worn:cataloghard_flooring_scratched_or_wornscene_groupbedroomroomcloset', 'living_repair__dining_room_primary']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| roof_shingles_missing_or_curling | defect→defect | added | $0–$0 | $500–$10,000 | $10,000 |
| roof_shingles_aged_or_worn | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| outdated_or_damaged_cabinets | defect→defect | removed | $500–$10,000 | $0–$0 | $-10,000 |
| damaged_or_aged_roof_shingles | defect→defect | removed | $500–$10,000 | $0–$0 | $-10,000 |
| cabinets_worn_finish | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |
| cabinets_damaged_or_water_stained | defect→defect | added | $0–$0 | $500–$10,000 | $10,000 |
| hard_flooring_scratched_or_worn | defect→degradation | added | $0–$0 | $1,380–$8,150 | $8,150 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_127468088  — BREACH (high_delta_pct, low_delta_pct)

Headline: $25,984–$82,790 → $39,507–$109,368 (low +52%, high +32%)
Packages: -['bedroom_modernization__bedroom_1', 'bedroom_repair__bedroom_3', 'exterior_repair__exterior_primary'] +['bedroom_modernization__bedroom_primary']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| older_flooring_style | upgrade→modernization | repriced | $641–$6,410 | $3,285–$12,419 | $6,009 |
| roof_shingles_aged_or_worn | defect→degradation | added | $0–$0 | $401–$8,012 | $8,012 |
| outdated_or_damaged_cabinets | defect→defect | removed | $401–$8,012 | $0–$0 | $-8,012 |
| damaged_or_aged_roof_shingles | defect→defect | removed | $401–$8,012 | $0–$0 | $-8,012 |
| cabinets_worn_finish | defect→degradation | added | $0–$0 | $401–$8,012 | $8,012 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $401–$8,012 | $8,012 |
| damaged_or_rotted_siding_or_trim | defect→defect | repriced | $401–$6,410 | $0–$0 | $-6,410 |
| appliances_dated_or_basic | defect→modernization | added | $0–$0 | $401–$4,807 | $4,807 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_10952874  — BREACH (high_delta_pct, low_delta_pct)

Headline: $45,293–$149,305 → $58,378–$195,883 (low +29%, high +31%)
Packages: -['bedroom_repair__bedroom_6', 'bedroom_repair__scratched_or_damaged_flooring:catalogscratched_or_damaged_flooringscene_groupbedroomroomcloset'] +['bedroom_repair__bedroom_7', 'bedroom_repair__hard_flooring_scratched_or_worn:cataloghard_flooring_scratched_or_wornscene_groupbedroomroomcloset']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| scratched_or_damaged_flooring | defect→defect | removed | $1,740–$9,200 | $0–$0 | $-9,200 |
| outdated_or_damaged_cabinets | defect→defect | removed | $500–$10,000 | $0–$0 | $-10,000 |
| cabinets_worn_finish | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |
| ceiling_cracks_or_sagging | defect→defect | added | $0–$0 | $500–$8,000 | $8,000 |
| hard_flooring_scratched_or_worn | defect→degradation | added | $0–$0 | $1,020–$7,100 | $7,100 |
| bare_or_missing_finish_flooring | defect→defect | added | $0–$0 | $800–$6,000 | $6,000 |
| vanity_worn_finish | defect→degradation | added | $0–$0 | $500–$6,000 | $6,000 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_80877597  — BREACH (high_delta_pct, low_delta_pct)

Headline: $30,440–$84,794 → $36,591–$108,502 (low +20%, high +28%)
Packages: -[] +['living_modernization__living_room_primary']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |
| appliances_dated_or_basic | defect→modernization | added | $0–$0 | $500–$6,000 | $6,000 |
| vanity_countertop_dated | upgrade→modernization | added | $0–$0 | $250–$2,500 | $2,500 |
| dated_or_worn_vanity_countertop | upgrade→degradation | removed | $250–$2,500 | $0–$0 | $-2,500 |
| older_flooring_style | upgrade→modernization | repriced | $2,120–$11,000 | $2,780–$12,500 | $1,500 |
| worn_or_stained_carpet | defect→degradation | repriced | $840–$4,880 | $1,380–$6,260 | $1,380 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_166147710  — BREACH (high_delta_pct, low_delta_pct)

Headline: $53,212–$170,576 → $35,803–$128,994 (low -33%, high -24%)
Packages: -['bathroom_repair__bathroom_primary', 'bedroom_modernization__bedroom_2', 'bedroom_repair__bedroom_1', 'bedroom_repair__bedroom_2', 'exterior_repair__exterior_primary'] +[]

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| rotted_subfloor_or_structural_framing | defect→defect | removed | $7,020–$24,960 | $0–$0 | $-24,960 |
| missing_base_cabinets_exposed_subfloor | defect→defect | removed | $3,218–$17,160 | $0–$0 | $-17,160 |
| boarded_up_entry_or_window | defect→defect | added | $0–$0 | $2,633–$14,040 | $14,040 |
| older_flooring_style | upgrade→modernization | added | $0–$0 | $1,460–$9,500 | $9,500 |
| scratched_or_damaged_flooring | defect→defect | removed | $1,740–$9,200 | $0–$0 | $-9,200 |
| roof_shingles_aged_or_worn | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| damaged_or_aged_roof_shingles | defect→defect | removed | $500–$10,000 | $0–$0 | $-10,000 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_10803207  — BREACH (high_delta_pct, low_delta_pct)

Headline: $25,650–$66,450 → $21,285–$80,000 (low -17%, high +20%)
Packages: -['bathroom_modernization__bathroom_primary'] +['bathroom_repair__bathroom_primary']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |
| vanity_dated_style | defect→modernization | added | $0–$0 | $500–$6,000 | $6,000 |
| appliances_dated_or_basic | defect→modernization | added | $0–$0 | $500–$6,000 | $6,000 |
| scratched_or_damaged_flooring | defect→defect | removed | $300–$5,000 | $0–$0 | $-5,000 |
| hard_flooring_scratched_or_worn | defect→degradation | added | $0–$0 | $300–$5,000 | $5,000 |
| older_flooring_style | upgrade→modernization | repriced | $1,460–$9,500 | $2,780–$12,500 | $3,000 |
| dated_bathroom_flooring_style | upgrade→modernization | added | $0–$0 | $300–$3,500 | $3,500 |
| dated_or_worn_vanity_countertop | upgrade→degradation | removed | $250–$2,500 | $0–$0 | $-2,500 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_125779232  — BREACH (high_delta_pct, low_delta_pct)

Headline: $32,780–$102,960 → $43,119–$121,816 (low +32%, high +18%)
Packages: unchanged

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| roof_shingles_aged_or_worn | defect→degradation | added | $0–$0 | $364–$7,283 | $7,283 |
| outdated_or_damaged_cabinets | defect→defect | removed | $364–$7,283 | $0–$0 | $-7,283 |
| damaged_or_aged_roof_shingles | defect→defect | removed | $364–$7,283 | $0–$0 | $-7,283 |
| cabinets_worn_finish | defect→degradation | added | $0–$0 | $364–$7,283 | $7,283 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $364–$7,283 | $7,283 |
| cabinets_damaged_or_water_stained | defect→defect | added | $0–$0 | $364–$7,283 | $7,283 |
| scratched_or_damaged_flooring | defect→defect | removed | $481–$4,406 | $0–$0 | $-4,406 |
| vanity_dated_style | defect→modernization | added | $0–$0 | $364–$4,370 | $4,370 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_81000709  — BREACH (high_delta_pct)

Headline: $22,456–$110,648 → $19,710–$92,382 (low -12%, high -17%)
Packages: -['bedroom_modernization__bedroom_3', 'bedroom_modernization__bedroom_4', 'bedroom_repair__bedroom_1', 'bedroom_repair__bedroom_4'] +['bedroom_repair__bedroom_3', 'living_modernization__living_room_primary']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| roof_shingles_aged_or_worn | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| damaged_or_aged_roof_shingles | defect→defect | removed | $500–$10,000 | $0–$0 | $-10,000 |
| damaged_or_rotted_siding_or_trim | defect→defect | removed | $500–$8,000 | $0–$0 | $-8,000 |
| interior_wall_stripped_to_studs | defect→defect | added | $0–$0 | $1,200–$6,400 | $6,400 |
| bare_or_missing_finish_flooring | defect→defect | removed | $800–$6,000 | $0–$0 | $-6,000 |
| scratched_or_damaged_flooring | defect→defect | removed | $660–$6,050 | $0–$0 | $-6,050 |
| hard_flooring_scratched_or_worn | defect→degradation | added | $0–$0 | $660–$6,050 | $6,050 |
| visible_mold_or_mildew | defect→defect | removed | $400–$5,000 | $0–$0 | $-5,000 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_11077450  — BREACH (low_delta_pct)

Headline: $31,915–$104,840 → $18,760–$99,922 (low -41%, high -5%)
Packages: -['kitchen_modernization__kitchen_primary'] +[]

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| cabinets_worn_finish | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| outdated_or_damaged_cabinets | defect→defect | removed | $500–$10,000 | $0–$0 | $-10,000 |
| hard_flooring_scratched_or_worn | defect→degradation | added | $0–$0 | $660–$6,050 | $6,050 |
| appliances_dated_or_basic | defect→modernization | added | $0–$0 | $500–$6,000 | $6,000 |
| hard_flooring_broken_or_warped | defect→defect | added | $0–$0 | $300–$5,000 | $5,000 |
| scratched_or_damaged_flooring | defect→defect | removed | $300–$5,000 | $0–$0 | $-5,000 |
| dated_or_worn_vanity_countertop | upgrade→degradation | removed | $250–$2,500 | $0–$0 | $-2,500 |
| ceiling_cracks_or_sagging | defect→defect | repriced | $1,100–$9,650 | $500–$8,000 | $-1,650 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_11185681  — BREACH (low_delta_pct)

Headline: $49,408–$163,100 → $58,900–$186,250 (low +19%, high +14%)
Packages: -['bedroom_modernization__bedroom_5', 'bedroom_repair__bedroom_6'] +[]

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| scratched_or_damaged_flooring | defect→defect | removed | $3,540–$12,000 | $0–$0 | $-12,000 |
| hard_flooring_scratched_or_worn | defect→degradation | added | $0–$0 | $2,100–$10,250 | $10,250 |
| roof_shingles_aged_or_worn | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| outdated_or_damaged_cabinets | defect→defect | removed | $500–$10,000 | $0–$0 | $-10,000 |
| damaged_or_aged_roof_shingles | defect→defect | removed | $500–$10,000 | $0–$0 | $-10,000 |
| cabinets_worn_finish | defect→degradation | added | $0–$0 | $500–$10,000 | $10,000 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |
| older_flooring_style | upgrade→modernization | added | $0–$0 | $800–$8,000 | $8,000 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_11079485

Headline: $35,673–$115,520 → $37,633–$125,040 (low +5%, high +8%)
Packages: -['bedroom_modernization__bedroom_2'] +['bedroom_modernization__bedroom_1', 'bedroom_repair__bedroom_2']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| outdated_kitchen_finishes | upgrade→modernization | removed | $2,000–$20,000 | $0–$0 | $-20,000 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $500–$10,000 | $10,000 |
| scratched_or_damaged_flooring | defect→defect | removed | $660–$6,050 | $0–$0 | $-6,050 |
| hard_flooring_scratched_or_worn | defect→degradation | added | $0–$0 | $660–$6,050 | $6,050 |
| vanity_worn_finish | defect→degradation | added | $0–$0 | $500–$6,000 | $6,000 |
| vanity_damaged_or_water_stained | defect→defect | added | $0–$0 | $500–$6,000 | $6,000 |
| outdated_or_damaged_vanity | defect→defect | removed | $500–$6,000 | $0–$0 | $-6,000 |
| appliances_worn_or_neglected | defect→degradation | added | $0–$0 | $500–$6,000 | $6,000 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_125970550

Headline: $61,623–$219,554 → $63,829–$201,895 (low +4%, high -8%)
Packages: -['bedroom_modernization__bedroom_1', 'bedroom_modernization__bedroom_2', 'bedroom_repair__bedroom_2', 'bedroom_repair__bedroom_4', 'exterior_repair__exterior_primary'] +['bedroom_modernization__bedroom_4']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| boarded_up_entry_or_window | defect→defect | removed | $2,331–$12,431 | $0–$0 | $-12,431 |
| roof_shingles_aged_or_worn | defect→degradation | added | $0–$0 | $443–$8,854 | $8,854 |
| outdated_or_damaged_cabinets | defect→defect | removed | $443–$8,854 | $0–$0 | $-8,854 |
| cabinets_worn_finish | defect→degradation | added | $0–$0 | $443–$8,854 | $8,854 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $443–$8,854 | $8,854 |
| cabinets_damaged_or_water_stained | defect→defect | added | $0–$0 | $443–$8,854 | $8,854 |
| hard_flooring_broken_or_warped | defect→defect | added | $0–$0 | $1,222–$7,216 | $7,216 |
| older_flooring_style | upgrade→modernization | added | $0–$0 | $708–$7,083 | $7,083 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_126224899  — BREACH (low_delta_pct)

Headline: $51,796–$188,757 → $63,023–$194,011 (low +22%, high +3%)
Packages: -['bathroom_modernization__bathroom_primary__bathroom_3', 'bedroom_repair__bedroom_3'] +['bathroom_modernization__bathroom_primary__bathroom_1']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| bathroom_layout_modernization_opportunity | upgrade→retired | removed | $3,939–$19,693 | $0–$0 | $-19,693 |
| boarded_up_entry_or_window | defect→defect | removed | $2,074–$11,060 | $0–$0 | $-11,060 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $394–$7,877 | $7,877 |
| cabinets_worn_finish | defect→degradation | added | $0–$0 | $394–$7,877 | $7,877 |
| damaged_or_aged_roof_shingles | defect→defect | removed | $394–$7,877 | $0–$0 | $-7,877 |
| outdated_or_damaged_cabinets | defect→defect | removed | $394–$7,877 | $0–$0 | $-7,877 |
| roof_shingles_aged_or_worn | defect→degradation | added | $0–$0 | $394–$7,877 | $7,877 |
| interior_wall_stripped_to_studs | defect→defect | added | $0–$0 | $945–$5,042 | $5,042 |

Decision: [ ] approve   [ ] reject   — reason: ____________

## redfin_126418713

Headline: $30,903–$91,640 → $32,761–$97,036 (low +6%, high +6%)
Packages: -['bedroom_modernization__bedroom_2'] +['bedroom_repair__bedroom_4']

| Driver (catalog item) | v1→v2 kind | Status | Baseline | Candidate | Δ high |
|---|---|---|---|---|---|
| roof_shingles_aged_or_worn | defect→degradation | added | $0–$0 | $366–$7,329 | $7,329 |
| damaged_or_aged_roof_shingles | defect→defect | removed | $366–$7,329 | $0–$0 | $-7,329 |
| cabinets_dated_style | defect→modernization | added | $0–$0 | $366–$7,329 | $7,329 |
| scratched_or_damaged_flooring | defect→defect | removed | $1,011–$5,973 | $0–$0 | $-5,973 |
| hard_flooring_scratched_or_worn | defect→degradation | added | $0–$0 | $748–$5,204 | $5,204 |
| interior_wall_stripped_to_studs | defect→defect | added | $0–$0 | $879–$4,691 | $4,691 |
| vanity_dated_style | defect→modernization | added | $0–$0 | $366–$4,397 | $4,397 |
| appliances_dated_or_basic | defect→modernization | added | $0–$0 | $366–$4,397 | $4,397 |

Decision: [ ] approve   [ ] reject   — reason: ____________
