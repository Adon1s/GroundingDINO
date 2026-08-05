# Handoff: catalog `kind` drift — the wear band is classified both ways

## Status

Branch `benchmark_system`, 2026-08-04. Reproduced and measured, not started.

Surfaced while shipping the Pass 2c exterior work
([HANDOFF_pass2c_exterior_recall.md](HANDOFF_pass2c_exterior_recall.md)). That change made Pass 2c
emit `upgrade_candidate` for worn/weathered finishes — a deliberate product call: wear is not an
immediate repair. The catalog was never migrated to the same principle, so the two now disagree
on the same observations.

This is a **kind/ontology** problem, not a data-entry problem. Do not start by editing 107 items.

## The two principles now in conflict

| | rule | example |
|---|---|---|
| Pass 2c (as of 2026-08-04) | wear/weathering/dated → `upgrade`; broken/missing/leaking → `defect` | "wood siding appears weathered" → `upgrade_candidate` |
| `issue_catalog.json` | any *condition* problem → `defect`; only style/opportunity → `upgrade` | `exterior_siding_discoloration_fading` → `kind: defect` |

They agree on the ends and disagree on the middle. The middle is large.

## The clearest proof of drift

Two items, same concept, same severity, same scope, **opposite kind**:

```
kind=upgrade  sev1  scope=cosmetic  concrete_driveway_surface_wear
kind=defect   sev1  scope=cosmetic  patio_or_porch_surface_wear
kind=defect   sev1  scope=cosmetic  deck_surface_weathering
kind=defect   sev1  scope=cosmetic  exterior_siding_discoloration_fading
```

No stated principle separates the first from the rest. Twelve items sit in this "wear band";
eleven are `defect`, one is `upgrade`.

## Why it bites now: routing is asymmetric and exterior-only

`evaluate_kind_routing` ([scene_classifier_passes.py](../tools/scene_classifier_passes.py)) widens
an `upgrade` observation to `("upgrade", "defect")` only when the text hits **both** a component
term and a condition term. A `defect` observation never widens (`non_upgrade_kind`).

`_ROUTING_COMPONENT_PATTERNS` is **exterior-only** — 23 terms: siding, deck, porch, roof, shingle,
gutter, downspout, fascia, soffit, brick, mortar, stucco, driveway, walkway, concrete, asphalt,
fence, chimney, foundation, slab, trim, baseboard, brickwork. There is no carpet, flooring,
cabinet, countertop, vanity, tile, wall, ceiling or paint.

Measured consequence — the same wear claim routes differently by room:

```
WIDENS  ['upgrade','defect']  Wood lap siding appears weathered, with staining and aged paint.
WIDENS  ['upgrade','defect']  Deck surface shows fading and graying from weather exposure.
LOCKED  ['upgrade']           Hardwood floors appear to be real wood with a worn, dull finish.
LOCKED  ['upgrade']           Carpet is visibly worn and stained in the traffic areas.
LOCKED  ['upgrade']           The vanity cabinetry appears worn.
LOCKED  ['upgrade']           Baseboards are minimal and show uneven paint or scuffs.
```

So **exterior wear self-heals** (it can still reach the `defect` items) and **interior wear does
not**. The interior targets are unreachable:

| observation | natural target | its kind | reachable? |
|---|---|---|---|
| worn carpet | `worn_or_stained_carpet` | defect | **no** |
| worn vinyl | `worn_or_stained_vinyl_linoleum` | defect | **no** |
| scuffed baseboards | `baseboard_wear_scuffs` | defect | **no** |
| worn cabinets | `outdated_or_damaged_cabinets` | defect | **no** |
| worn vanity | `outdated_or_damaged_vanity` | defect | **no** |

This is a live recall hole introduced on 2026-08-04, not a hypothetical.

## `kind` is doing three jobs at once

This is why the drift is hard to fix casually. `kind` currently determines:

1. **claim semantics** — is the thing broken or merely tired;
2. **retrieval routing** — `allowed_kinds` at Pass 2d, per the asymmetry above;
3. **de-facto pricing eligibility** — measured:

```
defect   n=67  has estimate block=45 (67%)  cost modes: allowance 45, heuristic 21
upgrade  n=40  has estimate block= 8 (20%)  cost modes: allowance 15, heuristic 22
```

Flipping an item `defect → upgrade` therefore silently moves it toward the unpriced lane unless an
`estimate` block is authored at the same time. **Any kind migration is a pricing change.** Size it
with `scripts/audit_exterior_estimate_coverage.py` before and after.

## Items that conflate two claims in one concept

Eight items assert two different claim types in a single id — the same disease as the
absent/damaged/maintenance gutter collapse that
[HANDOFF_pass2c_exterior_recall.md](HANDOFF_pass2c_exterior_recall.md) refused to ship. These
cannot be assigned a correct `kind` because they *are* two items:

```
[damage, wear ] defect   damaged_or_aged_roof_shingles
[damage, wear ] defect   stained_or_damaged_bath_fixtures
[damage, wear ] defect   worn_or_damaged_bathroom_flooring
[damage, wear ] defect   brick_weathering_or_mortar_deterioration
[damage, style] defect   outdated_or_damaged_cabinets
[damage, style] defect   outdated_or_damaged_vanity
[style,  wear ] upgrade  dated_or_worn_vanity_countertop
[style,  wear ] upgrade  dated_door_hardware
```

`style`+`wear` pairs are benign (both are upgrades). The six `damage`+X pairs are not: a photo
showing *aged* shingles and a photo showing *missing* shingles resolve to the same id and the same
price.

## Binding constraints

1. **Decide the ontology before touching items.** Either the catalog adopts the 2c principle
   (wear → upgrade) or 2c is reverted to the catalog's. Do not migrate items one at a time under an
   unstated rule — that is how the current drift happened.
2. **A kind flip without an `estimate` block is a silent price deletion.** See the 67% vs 20% split.
3. **Do not split a conflated item without a pricing decision for each half.** `aged shingles` and
   `missing shingles` need different cost models, which is the whole reason to split them.
4. **`severity` is not a proxy for `kind`.** defects run 1–5, upgrades 1–3, and they overlap
   heavily at 1–2 (defect 35 items, upgrade 35 items). Severity cannot be used to infer the migration.
5. Existing `scripts/audit_issue_catalog.py` covers room-term/concept naming only and
   `tools/catalog_validation.py` validates enums, not semantics. Neither will catch kind drift;
   a new check is needed if this is to stay fixed.

## Work items

| # | Change | Files | Risk | Budget |
|---|---|---|---|---|
| 1 | Record the ontology decision: does the catalog adopt "wear → upgrade"? | this doc | — | ±20 |
| 2 | Add interior component terms to `_ROUTING_COMPONENT_PATTERNS` (carpet, flooring, cabinet, countertop, vanity, tile, wall, ceiling, paint) so interior upgrades widen like exterior ones | `scene_classifier_passes.py` | low | ±12 |
| 3 | Audit script: flag kind vs claim-language mismatches, conflated names, and kind-without-estimate | `scripts/audit_catalog_kind.py` | none | ±200 |
| 4 | Migrate the 12 wear-band items to the decided kind, authoring `estimate` blocks where the flip demands one | `issue_catalog.json` | **high** | ±150 |
| 5 | Split the six `damage`+X conflated items, with per-half costing | `issue_catalog.json` | **high** | ±250 |
| 6 | Validator rule so new items cannot reintroduce the drift | `catalog_validation.py`, `tests/test_catalog_validation.py` | low | ±80 |

**Item 2 is the urgent one** and is independently landable — it closes the live interior recall
hole regardless of how the ontology question lands, because widening only ever adds candidates.
Items 4 and 5 are gated on 1. Item 3 should precede 4 so the migration has a before/after.

## Verification

```
.venv\Scripts\python.exe -m pytest tests/test_catalog_validation.py tests/test_scene_classifier_passes.py tests/test_candidate_provider.py -q
```

For item 2, assert the routing table directly — `evaluate_kind_routing("Carpet is visibly worn and
stained.", "upgrade")` must return `expanded_kinds == ("upgrade", "defect")`.

For items 4–5, snapshot estimate coverage before and after:
```
.venv\Scripts\python.exe scripts/audit_exterior_estimate_coverage.py --artifacts-root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts --json artifacts/kind_migration_before.json
```

## Gotchas

- The corpus in `backend/artifacts/` is not the real one; it lives at
  `C:\Users\Steven\IntelliJProjects\renointel-prod\artifacts` (~1,266 runs). `.env ARTIFACTS_ROOT`
  is stale.
- Re-analysis is required before any of this shows up in stored artifacts — a catalog edit does not
  retroactively change a written `photo_intel.json`. Replay the current catalog over stored
  descriptions the way `scripts/audit_pass2c_funnel.py` does if you need a before/after without
  re-running the pipeline.
- `category` is not `kind`. `category=cosmetic` currently holds 20 defects and 4 upgrades, and
  `category=opportunity` is 28/28 upgrades — the latter is the only category that is internally
  consistent today.
