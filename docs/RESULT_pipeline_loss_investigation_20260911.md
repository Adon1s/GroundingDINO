# Where useful observations are being lost

This investigation is ongoing. The findings below are supported by stored
lineage and all 18 properties in the first completed fresh replica; the two-replica acceptance and
production smoke are not complete. No new tuning change is being adopted here.

The evidence supports investigating **meaning lost during 2b/2c, catalog
eligibility and claim selection, and Terra-to-package amplification separately**.
A broader Pass 2d prompt adjustment cannot recover an observation that 2c never
forwarded, or resolve a kitchen-degradation condition absent from that catalog
slice. A final Terra rejection can also follow an earlier change in meaning.

## Concrete tuning leads

| Priority | Case and first supported boundary | Evidence | Recommendation |
| --- | --- | --- | --- |
| 1 | `ga_redfin_11000447_photo_004_g3`: 2b loses the condition modifier | 2a groups the range under “Very dated kitchen”; 2b says only “The kitchen has a freestanding electric range.” The neutral bullet is then dropped by 2c. `appliances_dated_or_basic` exists in both catalog versions. The photo was inspected in this task. | Test whether structuring preserves condition modifiers on each subject of a compound sentence. Measure against neutral-inventory controls so this does not manufacture datedness. |
| 1 | `ga_redfin_10806500_photo_001_g4`: 2c drops a spatially meaningful observation | 2b says “A downspout discharge is located near the foundation.” It never reaches 2d. The photo shows the extension lying at grade beside the basement window/foundation. The original review assigns 2c with high confidence. | Test whether 2c recognizes actionable component/location relationships without requiring a damage adjective. Preserve the reason for exclusion. Re-evaluate the complete downstream claim before counting recovery. |
| 2 | `1081afb497a60a1b`, redfin_11000447/photo_002: scene restriction plus a compound observation | “Kitchen cabinets, appliances, and lighting are older.” survives 2c as modernization. Its stored pool contains lighting/decor/windows, but no kitchen cabinet or appliance item. The photo's living-area classification excludes kitchen-scoped candidates. | Separate candidate eligibility from model choice. A scene/subject eligibility probe and compound-bullet probe would answer different questions; do not count this as an LLM declining an available cabinet item. |
| 2 | `70f737d3dba9a8ff`, redfin_10806500/photo_004: catalog/kind coverage | “Laminate countertops show visible wear.” survives as degradation. None of its eight candidates is a countertop item. Both catalogs have kitchen `countertop_damage` as **defect**, requiring burns/chips/cracks/swollen seams/missing sections; the worn-countertop item is bathroom vanity scope. | Correct the older attribution's unresolved catalog assumption: this is a kitchen wear coverage gap under exact-kind retrieval, not proof of bad selection. Decide scope before attempting resolver tuning; do not force damage from wear. |
| 3 | `rc_ace7463d6837`: the selected claim changes the subject | The observation concerns damaged shower valve/trim; the selected item asserts tub surround/shower pan damage. The factorized response even acknowledges damaged hardware while rejecting enclosure damage. | Audit component identity at the observation-to-claim boundary. Preserve the original 20/3 historical attribution separately; this evidence supports a mismatch lead, not a blanket conclusion that Terra lost the hardware observation. |
| 3 | `c222ee01cb90772f`, redfin_11000447/photo_006: a real available-candidate decline to inspect | The ceiling-edge/header discoloration observation reaches 2d; `peeling_or_discolored_paint` is its top candidate (0.712), but selection is null. This differs from the countertop coverage gap. | Keep in a focused “plausible item was offered but declined” slice, with image-based confirmation of surface/condition. The current evidence does not justify lowering global thresholds. |
| 4 | redfin_10806500 fresh replay: Terra verdict changes package tier | `cabinets_worn_finish` changes from supported to unsupported. Cabinet replacement remains represented through dated-style/kitchen-finish conditions, but the kitchen package shifts from full to partial rehab. | Measure package-price sensitivity to one supported-condition flip separately from actual loss of work. A price reduction is not automatically observation deletion. |

The downspout case also exposes a downstream scope mismatch worth recording:
the catalog description of `clogged_or_damaged_gutters` mentions dumping water
near the foundation, while its atomic claim lists clogged, sagging, or
disconnected drainage. Even recovering that bullet at 2c would not establish
that Terra can validate the intended drainage claim. This is a hypothesis for
a bounded follow-up, not an authorized catalog expansion.

## What the counts do and do not show

Across the 18 canary properties, 1,059 forwarded observations have null 2d
selections. They mix intentional declines, generic/compound wording, catalog
scope gaps, and potential resolver errors. They are **not 1,059 useful losses**.

All 350 observations removed by the product filter are electrical items under
the existing quarantine: 151 dated lighting, 80 outlets/switches, 65 visible
electrical risks, 37 older fans, 12 vanity lights, 4 vintage fixtures, and 1
exhaust fan. Keep these as policy exclusions. Likewise, 2e display suppression
is not a billing loss if the canonical estimate lane retains the observation;
package absorption still represents the work.

The prior condition-based human miss review assigned 20 cases to Terra and 3
to 2d. That cohort starts with an existing condition, so it cannot measure most
pre-condition losses. The separate two-property gold cohort assigned 4 to 2b,
5 to 2c, 8 to 2d, 2 to projection, and 1 to Terra, with 10 at 2a, 5 unclear,
and 5 excluded. Several gold attributions explicitly lacked a catalog check;
the countertop example above resolves one of those uncertainties. These
figures are separate cohort evidence, not population-wide error rates.

## First fresh property's dollar effect

For redfin_10806500, the first fresh artifact passes the production verifier.
Its headline is **$13,513–$51,036**, compared with stored run 1's
$23,903–$74,107 and stored run 2's $19,889–$57,538. This exceeds the 15% review
threshold and remains pending Steven's disposition and the fresh second replica.

The entire difference from stored run 1 is the kitchen package allowance:
$20,780–$48,486 becomes $10,390–$25,415. Standalone, bathroom, and exterior
amounts are unchanged. Cabinet replacement remains in the package; the lost
`cabinets_worn_finish` support changes the package's pricing tier. Both Sol
decisions approve their respective candidate, so the reduction is already in
the deterministic candidate presented to Sol. Terra also changes a vinyl-floor
rejection to `cannot_assess`/inspection. These are recorded individually rather
than explained away using an aggregate noise rate.

## Completed first replica: useful losses versus policy and pricing

All 18 artifacts pass the production verifier and catalog invariants. The
3,668 observation selections contain 3,501 unchanged selections, 70 item-to-item
changes, 55 item-to-null changes, and 42 null-to-item changes. Catalog, temperature,
and model changes are confounded in the stored-to-fresh comparison; the second
fresh replica is needed to measure repeatability under the same configuration.

Of the 125 human-labeled cases, 37 belong to production properties outside this
replay. The 88 in-cohort cases currently contain 60 warranted work items present,
8 warranted work items missing, 7 accepted-policy dispositions, 7 needing semantic
review, and 6 false/unwarranted work items still present. These are reviewed-case
counts, not precision or recall estimates for the entire pipeline.

Two of the eight missing warranted cases were previously supported:

- `rc_32ff789c40d0`, redfin_11079485/photo_005: “The shower tile is dated.”
  resolves to `vintage_tile_pattern_style`; fresh Terra rejects a clearly vintage
  pattern. The task's image inspection confirms large neutral square tiles in a
  worn bathroom, but does not settle the narrower vintage-pattern claim. Preserve
  the human label and examine broad datedness versus the selected claim's scope.
  The removed work allowance is $256–$1,278.
- `rc_f468f4066e3f`, redfin_11185681/photos_017,020,044: the wall wear/scuff
  observations remain available to Terra, which now rejects clear marks. All three
  photos were inspected: photo_020 has small scattered wall marks; photo_017 has
  pronounced sheen/unevenness, and the floor wear is much more conspicuous across
  the set. Keep this as a surface-specific evidence/sensitivity review, without
  substituting floor wear for wall damage. The removed allowance is $33–$168.

The other six labeled misses remain unbilled. Examples show why their final
Terra endpoint is not enough to assign causality: the lifting-seam observation
does not identify vinyl, yet the claim requires vinyl/linoleum; the weathered
masonry group mixes brick and stone observations under a brick claim; “tired”
paint reaches a peeling/bubbling claim. These belong in a subject/material/condition
preservation audit before any general relaxation of Terra acceptance.

Five of the six closeout cases now receive supported verdicts: porch soffit,
siding/trim, shower enclosure, hard-floor wear, and lower-cabinet damage. Tired
paint remains unsupported. These are apparent recoveries, not repeatable tuning
wins; the shower hardware/enclosure mismatch still needs semantic review.

Five properties exceed the 15% headline-change threshold. The complete price
packet is `reports/backend_acceptance_replica1_packet_20260911.md`, with applied
package amounts, standalone amounts, and exact low/high reconciliation. Besides
the kitchen-tier change above, a major cause is the existing QP3 policy:
opportunity-only bedroom/living modernization packages are not applied even
when Sol approves them. This affects redfin_25809814, redfin_80925528, and
redfin_81000709. Their individual work remains priced standalone. The reason
`opportunity_only_interior_modernization` comes from deterministic reconciliation,
not a Sol rejection and not an observation disappearing. Other changes include
the absent bathroom modernization package in redfin_11079485 and a newly applied
exterior repair package in redfin_81000709. Each is recorded separately.

Replica 1 consumed exactly 912,131 Terra and 136,606 Sol tokens, reconciled to
148 and 18 settled ledger calls respectively; no unknown reservations remain.
The original shared Terra batch has 1,087,869 tokens remaining. The paid second
replica is constrained to September 12 UTC (September 11 at 7 p.m. Central or
later); its local 2d preparation can finish beforehand.

## Named acceptance subsets and remaining limits

- LQ-1: 29 of the 33 historical bathroom candidate-change rows are in this frozen
  replay; four are outside its exact source-run cohort and remain explicitly
  unresolved. Three of the 29 selections change: vintage tile to dated bathroom
  flooring, generic wallpaper to null, and generic paint refresh to bathroom
  paint refresh. Twenty-six remain the same, including four nulls. These counts
  record choices, without declaring all 29 semantically correct.
- LQ-2: all 55 fresh trim conditions route to `no_action`, with no work item.
- Work deduplication: all 72 collision records keep one active work item and
  suppress the others; the production verifier checks the resulting coverage.
- CCF-13/worklist row 23: redfin_25809814 no longer bills both generic and bathroom
  wallpaper. Generic wallpaper is excluded by policy; bathroom wallpaper still
  reaches Terra, which calls the surface behind the vanity textured tile and
  rejects it. There are **zero** active wallpaper work items, so the duplicate
  is removed but the plan's expected retention of one charge is not demonstrated.
  Preserve this distinction for replica 2 and final semantic disposition.

The scorer now records all of these subsets by original row and replica. It
also records fresh Sol agreement across replicas when both paid results exist.
The independent `backend_acceptance_local_stability.py` report can compare
verified 2d preparations before those paid results exist. Its partial results
must not be presented as all-property repeatability.

## Evidence files

- `reports/pipeline_loss_ledger_20260910.json`: every original reviewed
  population row, prior attribution, exact run reference, and joined issue ids.
- `artifacts_canary/backend_acceptance_20260910/investigation/observation_traces.json`:
  full observations, candidate lists, 2e/product presence, Terra rationale,
  routing, work, package coverage, and unmatched 2b bullets.
- `reports/backend_acceptance_score_20260910.json`: fresh replica dispositions,
  per-observation selection changes, package changes, and dollar-review flags.

The immediate recommendation is to finish the bounded replay, then select a
small stage-specific experiment from these named losses. The additional
experimental budget remains reserved until Steven approves a concrete design.

## Attribution correction: the two projection cases

The gold cases `ga_redfin_11000447_photo_003_g5` (loose cable, issue `7d2f54e49febb555`) and `ga_redfin_11000447_photo_003_g8` (dated fan, issue `76ecc527150e80c1`) were previously attributed to condition projection. The raw estimate lane retains both, and the product-filtered lane removes both under electrical quarantine, before condition projection runs. Record these as intentional product-policy exclusions; they are not evidence of a condition-projection implementation defect. The original attributions remain in the ledger for audit. A comparator window-treatment issue mentioned in the fan rationale is not the fan observation and is excluded from its primary lineage.
