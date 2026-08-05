# Catalog Estimate-Scope Audit

Data-hygiene pass over `tools/issue_catalog.json` confirming that each item's
`estimate_scope` classification routes it into the correct renewal-toggle headline
tier (`final_rehab_required` / `final_rehab_resale_ready` / `final_rehab_full_renewal`),
which are built from `totals_by_scope_capped` by `build_scope_headline_tiers`
([tools/estimate_scope.py](../tools/estimate_scope.py)). Follow-up to
[HANDOFF_catalog_scope_audit.md](HANDOFF_catalog_scope_audit.md). See also the
embed-text/structure audit in [catalog_audit_report.md](catalog_audit_report.md).

The classifier is mostly a substring heuristic, so a mis-bucket silently moves
real money between tiers (a repair demoted to renewal vanishes from
`final_rehab_required`; a cosmetic promoted to required inflates the must-fix number).

## Method (reproducible)

Enumeration is **observe-only** — `scripts/audit_estimate_scope.py` calls the real
`classify_estimate_scope_with_reason({}, item, None)` and `classify_package_scope`;
it never re-implements scope logic. The empty `{}` candidate makes the catalog
values the sole source of truth (pure baseline, no inspect posture).

```
# before snapshot
.venv\Scripts\python.exe scripts/audit_estimate_scope.py --json artifacts/estimate_scope_before.json
# ...apply overrides, then diff:
.venv\Scripts\python.exe scripts/audit_estimate_scope.py \
    --baseline artifacts/estimate_scope_before.json \
    --json artifacts/estimate_scope_after.json \
    --report docs/catalog_estimate_scope_audit_raw.md
```

Adjudication followed the rubric in the audit plan: defects/safety/function → `required_rehab`;
dated-but-functional cosmetics → `marketability_rehab`; discretionary layout/expansion → `optional_value_add`;
inherently latent/unconfirmable risk → `inspection_risk`. Overrides were applied
**judiciously** (only clear mis-buckets, ambiguous high-impact rows, and substring
traps), not mass-added.

> **Pass 2 (matcher conversion) is recorded in its own section at the end of this
> document.** Everything between here and there describes pass 1. Current totals are
> **23 overrides** and a **37 / 57 / 11 / 1** distribution, not the 14 / 45-48-12-1
> that pass 1 ended at.

## Summary counts (before → after)

| metric | before | after |
| --- | --- | --- |
| Total catalog items audited | 106 | 106 |
| Items with `estimate_scope` override | 2 | 14 |
| Items still classified by `fallback_required` | 0 | 0 |
| Items still heuristic (no override) | 104 | 92 |

### Scope distribution

| scope | before | after | delta |
| --- | --- | --- | --- |
| `required_rehab` | 55 | 45 | **−10** |
| `marketability_rehab` | 41 | 48 | +7 |
| `optional_value_add` | 10 | 12 | +2 |
| `inspection_risk` | 0 | 1 | +1 |

The net effect is a **10-item reduction in the `required` (must-fix) tier**: nine
cosmetics that the heuristic over-required were demoted to `marketability`, and one
inherently-latent item was routed to `inspection_risk` (out of every headline tier).
No item was promoted *into* `required` — the heuristic's bias (severity-threshold +
substring required-terms) errs toward over-requiring, so every correction is a
demotion.

### Reason histogram movement (heuristic reasons that decreased)

| reason | before | after |
| --- | --- | --- |
| `required_condition_signal` | 15 | 8 |
| `defect_severity_threshold` | 32 | 30 |
| `defect_default` | 4 | 3 |
| `upgrade_marketability` | 28 | 26 |

`fallback_required` was **0 both before and after** — every catalog item is a
`defect` or `upgrade`, so the non-defect/non-upgrade ultimate fallback never fires.
There were no `fallback_required` rows to adjudicate (addition #3).

## Overrides applied (12)

Each sets `estimate_scope` + a descriptive `estimate_scope_reason` (prefix
`catalog_override_`). Grouped by why the heuristic was wrong.

### Substring/severity over-fire → demoted to `marketability_rehab` (9)

| id | was | reason it mis-fired |
| --- | --- | --- |
| `exterior_door_paint_failure` | required | `"failure"` in "paint **failure**" matched a required-term; sev-1 `scope: cosmetic` repaint |
| `shed_exterior_paint_failure` | required | same `"failure"` trap; sev-1 cosmetic shed repaint |
| `deck_surface_weathering` | required | matched `"structural"`/`"failure"` **inside the negation** "not obvious structural failure"; sev-1 refinish |
| `ceiling_fan_mismatched_blades` | required | `"missing"` came from `support_any` ("missing blade") retrieval text; sev-1 cosmetic fan |
| `scratched_or_damaged_flooring` | required | `"broken"` (broken floor pieces) over-fired; sev-1 worn/scratched flooring |
| `baseboard_wear_scuffs` | required | `"missing"` from embed retrieval text; sev-1 cosmetic trim (modernization support) |
| `countertop_damage` | required | `"missing"` (missing section); sev-2 functional-but-ugly surface — matches its bathroom twin `vanity_countertop_damage` (already marketability) |
| `outdated_or_damaged_vanity` | required | sev-3 hit `defect_severity_threshold` **before** the marketability check; it is the modernization twin of `outdated_or_damaged_cabinets` (marketability) |
| `tree_stump_present_in_yard` | required | `defect_default` defaulted a sev-1 curb-appeal yard item to required; not a must-fix |

### Inherently latent / unconfirmable → `inspection_risk` (1)

| id | was | rationale |
| --- | --- | --- |
| `roofline_water_damage_suspected` | required | "**Possible** Water Damage at Roofline… suggests **possible** moisture intrusion **requiring evaluation**… **Inspect** and repair." Inherently unconfirmable from a photo → must be surfaced as risk, not auto-priced into the headline. (Visually-confirmed water items — `water_stain_ceiling`, `active_water_damage_bathroom` — correctly stay `required`.) |

### Discretionary full-renewal → `optional_value_add` (2)

| id | was | rationale |
| --- | --- | --- |
| `unfinished_basement_present` | marketability | finishing a basement is a discretionary value-add, not a resale-ready essential (the marketability route came from `"finish"` matching inside "un**finish**ed") |
| `bathroom_layout_modernization_opportunity` | marketability | layout reconfiguration / full redesign is discretionary; consistent with `layout_modernization_opportunity` (already optional) |

## Reviewed and accepted unchanged (representative)

Per the no-mass-override discipline, many flagged rows were reviewed and **left as
the heuristic classified them** — recorded here so the review is auditable:

- **`cosmetic_defect_marketability` (11 rows)** — all genuinely cosmetic
  (worn/stained carpet & vinyl, scuffs, dirty floors, faded siding, discolored
  paint, vanity-top damage). Correctly `marketability`; not promoted.
- **Bathroom moisture defects kept `required`** despite cosmetic substrings:
  `worn_or_damaged_bathroom_flooring` ("soft subfloor", moisture-rated),
  `peeling_or_damaged_bathroom_paint` (efflorescence/bubbling near plumbing),
  `tub_surround_or_shower_pan_damage`, `missing_bathroom_tile_exposed_substrate`,
  `missing_vanity_exposed_plumbing`, `missing_or_damaged_caulk_at_tub_or_shower`.
- **`fence_damaged_or_weathered`** kept `required` (matched `"rotted"` — real repair).
- **`driveway_or_walkway_cracking`, `trees_or_vegetation_too_close`** kept `required`
  (`defect_default`; defensible as trip-hazard / structural-proximity maintenance).
- **`optional_modernization` rows** (dated trim/doors/paneling/decor/fireplace/exterior
  finishes, stained-glass fixture) kept `optional` — deliberately `tier: optional`
  nice-to-haves. (`marketability` vs `optional` only shifts resale-ready vs
  full-renewal — both renewal, neither in `required` — so low cost-consequence.)

## Package surface (reviewed clean)

`package_strength` is **runtime-computed** (`compute_package_strength`,
[rehab_packages.py](../tools/rehab_packages.py)) and is *not* a catalog field, so
nothing was added for it. The audit checked the catalog `package_category` fields
and the static `_PACKAGE_TYPE_TO_CATEGORY` + `PACKAGE_AFFINITY` maps.

Four package types are represented (covering 13 drivers + 17 supports): all carry a
catalog `package_category` that **agrees with the static-map category**, all
`PACKAGE_AFFINITY` entries agree with the map, and the deterministic
strong→`marketability` / moderate→`optional` (modernization) and `repair`→`required`
placements read correctly. **No package metadata changes were needed** (0 review
flags, 0 affinity conflicts).

| package_type | category | strong | moderate |
| --- | --- | --- | --- |
| `bathroom_modernization` | modernization | marketability_rehab | optional_value_add |
| `bathroom_repair` | repair | required_rehab | required_rehab |
| `kitchen_modernization` | modernization | marketability_rehab | optional_value_add |
| `living_modernization` | modernization | marketability_rehab | optional_value_add |

## Real-property spot-check (`redfin_125727311`, 27 detected issues)

The available `photo_intel.json` artifacts predate issue extraction (no
`estimate_issues_flat`), so `issues_flat` was synthesized from the property's own
`renovation_needs.issues` (real detected catalog `issue_id`s) and run through
`compute_renovation_estimate_v4`.

- **Scope split:** 15 `required` (roof shingles, siding, deck, mold, electrical,
  drywall, foundation/garage, gutters, grading, trip hazard…), 11 `marketability`
  (paint, cabinets, bath fixtures, curb appeal, landscaping, dated finishes), 1
  `optional` (`dated_lighting_fixtures`), 0 `inspection_risk`. No obvious mis-bucket.
- **Override impact on real data:** `scratched_or_damaged_flooring`, a *real detected
  issue*, now lands in `marketability` instead of inflating `required`.
- **Dollar tiers nest correctly:** required $2,500–$38,200 ≤ resale-ready
  $3,800–$55,200 ≤ full-renewal $3,800–$55,200 (low and high).

## Verification / definition of done

- ✅ Enumeration re-run: all 12 corrected items resolve to the intended scope via
  `catalog_override`; all `fallback_required` rows reviewed (none exist); substring
  traps checked with source-field attribution.
- ✅ `.venv\Scripts\python.exe -m pytest tests/test_renovation_estimate_v4.py tests/test_package_taxonomy.py -q` → **83 passed**, including the new
  `TestRepresentativeCatalogScopes` regression set.
- ✅ Real-property spot-check shows sane `required ≤ resale_ready ≤ full_renewal`.
- ✅ No obvious repair dropped from `final_rehab_required`; no obvious cosmetic-only
  item inflates it.
- ✅ Package metadata reviewed; no changes required.

### Follow-ups (out of scope here, documented per the audit guardrails)

- The `outdated_or_damaged_*` / `*_paint_failure` items conflate cosmetic and repair
  intents in one catalog entry; a future split (separate latent/required vs cosmetic
  IDs) would remove the reliance on overrides.
- `compute_package_strength` thresholds were not touched; if a property ever shows a
  modernization package landing in the wrong tier, investigate strength there.

---

# Pass 2 — `term_matches` conversion and the curb-appeal sweep

Pass 1 audited the catalog against the classifier as it stood. Pass 2 fixes the
classifier itself: `estimate_scope._contains_any` was the **last raw-substring matcher
in the codebase** (every other keyword surface moved to word-anchored `term_matches`
during the term-hygiene pass — see [HANDOFF_catalog_term_hygiene.md](HANDOFF_catalog_term_hygiene.md) §4).
It mattered because `_catalog_text` concatenates the candidate's `supporting_observations`,
so free-form VLM text flows through these terms and decides which tier the money lands in.

## Two substring bugs, both measured

Corpus: 4,572 unique observations from 21 `catalog_audit_*.json` runs (harvested with
`scripts/audit_catalog_terms.harvest_corpus`). Grid = 106 items × 4,572 observations
= 484,632 (item, observation) pairs.

**1. `finish` matched "unfinished".** Four items were riding on it, all landing in
`marketability_rehab` only because the substring fired.

**2. `mold` prefix-matched "crown molding".** This one was *not* in the handoff and is
the larger of the two. Word-start anchoring fixes *interior* collisions
(`ding`⊂`siding`) but not *prefix* ones, so anchoring alone would not have caught it.
On the 19 corpus observations mentioning crown molding, **14 purely cosmetic items** —
worn carpet, wall scuffs, dirty floors, peeling paint, dated cabinets, siding fade,
patio wear — were promoted into `required_rehab`. Any property whose photos mention
trim was inflating its must-fix headline.

I checked every prefix extension of every scope term against the corpus. All are wanted
inflections (`finish`→finishes/finished, `fixture`→fixtures, `paint`→painted,
`scuff`→scuffs/scuffed, `style`→styles, `structural`→structurally, `repaint`→repainting,
`refresh`→refreshing, `cosmetic`→cosmetically) **except** `mold`→molding/molded. So
`mold$` is the only marker warranted, and the leading-only default stays untouched.

| change | pair flips | items |
| --- | --- | --- |
| `_contains_any` → `term_matches` | 307 (0.063%) | 4 |
| ...same conversion on the inspect-posture path | 1 | 1 (`pest_or_rodent_evidence`, required → inspection_risk — correct direction) |
| `mold` → `mold$` | 266 | 14 |
| `strip_term_marker` in `_catalog_text` | 0 | 0 |
| **all code changes together** | 546 (0.113%) | 15 |
| **plus the 9 overrides below** | 32,549 (6.7%) | 20 |
| catalog text alone, no observations | **0** | 0 |

## Code changes ([tools/estimate_scope.py](../tools/estimate_scope.py))

1. `_contains_any` calls `term_matches` (the single matcher in
   [pipeline_common.py](../tools/pipeline_common.py)) instead of `term in text`.
2. `_catalog_text` applies `strip_term_marker` to `support_any` values. Five catalog
   items carry `$`-marked retrieval terms (`mold$`, `tub$`, `wall$`) and those markers
   were landing in the scope blob as literal text. Measured no-op today — it is there
   so a data marker never has to line up with a code term.
3. `mold` → `mold$` and `visible mold` → `visible mold$` in `_REQUIRED_TERMS` and
   `_VISIBLE_REQUIRED_CONDITION_TERMS`. `mildew` left alone. Only meaningful after (1),
   so all three ship together.

## Overrides applied (9) — the curb-appeal sweep

The conversion exposed that `defect_default → required_rehab` is the wrong fallback for
cosmetic exterior maintenance; the substring bug had been masking it. Rather than ship
that flip bare, the four affected items were adjudicated explicitly — and the sweep was
then widened to **every** item landing in `required_rehab` that reads as curb appeal.
All nine → `marketability_rehab`.

| id | was | why |
| --- | --- | --- |
| `driveway_or_walkway_cracking` | required (`defect_default`) | sev-2 concrete weathering; its own description says "safety **or curb appeal**" |
| `trees_or_vegetation_too_close` | required (`defect_default`) | sev-2, `scope: service`, LANDSCAPE_TRIM — cheap trimming |
| `brick_weathering_or_mortar_deterioration` | required (`defect_default`) | sev-2; "weathering, discoloration" dominates the detection over repointing |
| `gutter_maintenance_needed` | optional (`optional_upgrade`) | turnover service done before listing; also hallucination-prone from photos |
| `clogged_or_damaged_gutters` | required (`defect_severity_threshold`) | keeps the gutter pair in one tier |
| `metal_carport_rust_corrosion` | required (`required_condition_signal`) | surface rust is appearance; structural failure would be a different find |
| `fence_damaged_or_weathered` | required (`required_condition_signal`) | fired on `"rotted"`, but "weathered boards" is the common detection |
| `empty_or_deteriorated_inground_pool` | required (`defect_severity_threshold`) | restore-or-fill is discretionary and carries large dollars into must-fix |
| `appliance_damage_or_missing` | required (`required_category`) | "significantly aged" is renewal; appliances are often excluded from sale |

### Reviewed and deliberately left `required`

`damaged_drywall_or_cracks` (settlement cracking is structural even though nail pops
are not), `broken_or_fogged_windows` (broken glass dominates), and
`damaged_soffit_or_porch_ceiling` (soffit damage signals water intrusion) — all three
are conflated entries, but the required reading is the safer default. The
bathroom-moisture set stays as pass 1 adjudicated it.

## Summary counts (pass 1 → pass 2)

| metric | pass 1 | pass 2 |
| --- | --- | --- |
| Items with `estimate_scope` override | 14 | **23** |
| Items still heuristic | 92 | 83 |
| `fallback_required` | 0 | 0 |

| scope | pass 1 | pass 2 | delta |
| --- | --- | --- | --- |
| `required_rehab` | 45 | **37** | **−8** |
| `marketability_rehab` | 48 | **57** | +9 |
| `optional_value_add` | 12 | **11** | −1 |
| `inspection_risk` | 1 | 1 | — |

At pair level (real observation text), `required_rehab` falls 202,076 → 171,837 (−15%).

## Real-property spot-check (`redfin_125727311`, 27 real detected issues)

Run against the committed `HEAD` tree in a scratch worktree, then against this change:

| | before | after |
| --- | --- | --- |
| `totals_by_scope_capped.required_rehab` | $2,500 – $38,200 | $2,000 – $32,200 |
| `totals_by_scope_capped.marketability_rehab` | $1,800 – $27,000 | $2,300 – $33,000 |
| `final_rehab_required` | $8,427 – $32,273 | **$6,570 – $27,630** |
| `final_rehab_resale_ready` | $16,331 – $53,169 | $16,331 – $53,169 |
| `final_rehab_full_renewal` | $16,331 – $53,169 | $16,331 – $53,169 |

This is the shape the change should have: the must-fix tier drops, the two cumulative
renewal tiers are **unchanged to the dollar**. Money was re-bucketed, not re-priced.

## Verification / definition of done

- ✅ Full suite: `.venv\Scripts\python.exe -m pytest -q` → **1,194 passed, 1 skipped**.
- ✅ `scripts/audit_estimate_scope.py --baseline artifacts/estimate_scope_after.json`
  reports exactly the 9 items above as scope changes, every one via `catalog_override`;
  `items=106 overrides=23 fallback_required=0`, `packages=7 flagged=0`.
- ✅ `scripts/audit_catalog_terms.py --baseline artifacts/catalog_terms_before.json`
  unchanged: `lost-to-anchoring=0`, `unretrievable require_any items=1`
  (`pest_or_rodent_evidence`, knowingly dead). This change touches no catalog terms;
  the audit is the guard against having accidentally done so.
- ✅ Every one of the 13 catalog ids asserted by the v4 scope tests classifies
  identically (scope *and* reason) — including the four rows that negatively assert
  "must stay heuristic". No existing assertion was modified.
- ✅ New regression coverage: four new `TestRepresentativeCatalogScopes` cases, plus
  `test_crown_molding_does_not_promote_a_cosmetic_item_to_required` and
  `test_real_mold_observation_still_classifies_required` — the second exists so the
  mold fix can't be traded away for the crown-molding fix.

### Follow-ups

- `_contains_any` was the last raw-substring matcher; there is now no un-anchored
  keyword surface left in the pipeline.
- The three conflated entries left at `required` (drywall, windows, soffit) are the
  same class of problem as pass 1's follow-up: one catalog entry carrying two intents.
  Splitting them into separate IDs would retire the overrides rather than accumulate them.
