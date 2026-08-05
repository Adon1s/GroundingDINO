# Handoff: renewal-vs-repair costing toggle

> **Status (update):** Change 1 has **shipped** — `compute_renovation_estimate_v4` now emits the
> third named tier `final_rehab_full_renewal` (= required + marketability + optional_value_add),
> alongside the existing `final_rehab_required` / `final_rehab_resale_ready`, via
> `build_scope_headline_tiers` in `tools/estimate_scope.py`. Change 2 was implemented as
> **emit-only** (no `renewal_costing_level` param; `final_rehab` unchanged) — a UI selects the tier.
> Remaining follow-ups: the catalog scope audit ([HANDOFF_catalog_scope_audit.md](HANDOFF_catalog_scope_audit.md))
> and, if renewal-ON is a common view, Option 3 ([HANDOFF_whole_home_cosmetic_rollup.md](HANDOFF_whole_home_cosmetic_rollup.md)).

## Goal

Let a consumer choose whether **renewal** work (cosmetic/modernization upgrades — popcorn, dated trim, dated finishes, lighting, flooring style, etc.) is included in the headline renovation cost, separately from **repair** work (must-fix defects). The selector must be a **category**, not a hand-maintained item list, and packages are the primary cost surface.

## Key finding: this is ~80% already built

The repair-vs-renewal axis already exists end-to-end as the **estimate scope** classification ([tools/estimate_scope.py](../tools/estimate_scope.py)). Every line item *and every package* is sorted into one of four scopes, which map onto the toggle directly:

| Scope | Toggle bucket |
|---|---|
| `REQUIRED_REHAB` | **repair** (must-fix) |
| `MARKETABILITY_REHAB` | **renewal** — strong cosmetic/modernization + turnover |
| `OPTIONAL_VALUE_ADD` | **renewal** — weaker/“nice to have” modernization, layout, value-add |
| `INSPECTION_RISK` | latent risk (kept separate) |

Packages route by category via `classify_package_scope` ([estimate_scope.py:274](../tools/estimate_scope.py#L274)) — verified:
`repair → REQUIRED_REHAB`, `modernization(strong) → MARKETABILITY_REHAB`, `modernization(moderate) → OPTIONAL_VALUE_ADD`, `turnover → MARKETABILITY_REHAB`, `inspection_risk → INSPECTION_RISK`.

And `compute_renovation_estimate_v4` already emits, every run ([renovation_estimate_v4.py:257-268](../tools/renovation_estimate_v4.py#L257)): `final_rehab_required`, `final_rehab_resale_ready`, and `totals_by_scope_capped`/`_raw` (all four scopes broken out). `build_required_and_resale_ready` ([estimate_scope.py:411](../tools/estimate_scope.py#L411)) defines `resale_ready = required + marketability`.

So the toggle is fundamentally a **view selection over already-computed per-scope totals — not a new cost engine.**

## Real-property evidence (`redfin_126231380`, run `20260602_161312_0dd2661a`)

```
final_rehab            (current headline) : $28,000 – 71,000
final_rehab_required   (repair only)      : $0 – 0
final_rehab_resale_ready (req + market.)  : $21,000 – 43,000
totals_by_scope_capped:
  required_rehab     : $0
  marketability_rehab: $21,000 – 43,000
  optional_value_add : $7,000  – 28,000
  inspection_risk    : $0
```

Three observations that shape the design:
1. **This property is 100% renewal, 0% repair** (`required = $0`). A naive binary “renewal OFF” would make the headline `$0` — which reads wrong for a house that visibly needs cosmetic work. So **present layered tiers, not a single toggle that can zero the number** (see UX note).
2. **There are effectively three tiers already**, but only two are named:
   - repair only = `$0` (= `final_rehab_required`)
   - + strong renewal = `$21–43k` (= `final_rehab_resale_ready` = required + marketability)
   - + all renewal = `$28–71k` (= marketability + optional_value_add = the *current* `final_rehab`)
   The current headline already includes **everything** (renewal effectively always-on). `optional_value_add` is in `final_rehab` but in **neither named scope-headline** — that's the gap to close.
3. **The fan-out we discussed is visible here**: `popcorn_or_acoustic_ceiling_texture` is a support in **5** of the package candidates (2 bedroom + 3 living), and there are **3 over-split `living_modernization`** packages (`living_room_1/2/3`). Mechanism #1 (ambient demotion) + Mechanism #2 (single-instance collapse) — already implemented in `tools/` — would cut this 6→2 before the toggle even applies. The toggle and the dedup are orthogonal and compose.

## Design

Principle: **always compute all tiers (already true); the toggle picks which tier the headline points at. No recompute.**

### Change 1 — name all three tiers (small, in `tools/estimate_scope.py` + the v4 emit)
Add a third headline bucket alongside the existing two:
- `final_rehab_required` = `required_rehab` (exists)
- `final_rehab_resale_ready` = `required + marketability_rehab` (exists)
- **NEW** `final_rehab_full_renewal` = `required + marketability_rehab + optional_value_add`
  (this is what `final_rehab` already equals today; making it a named scope bucket closes the gap and removes the “`final_rehab` mysteriously exceeds `resale_ready`” confusion).
`inspection_risk` stays a separate risk line, never folded into the headline.

Extend `build_required_and_resale_ready` (or add a sibling `build_scope_headline_tiers`) to return all three; wire the new key into the v4 bucket list ([renovation_estimate_v4.py:257](../tools/renovation_estimate_v4.py#L257)).

### Change 2 — expose the selector (lowest-risk: view-only)
**Recommended v1:** do **not** mutate `final_rehab` server-side. Guarantee all three named tiers are always emitted and let the UI/API pick which to display. The “toggle” is purely a default-display preference — zero recompute, zero risk to existing consumers.

**If a server-side default is wanted:** add a `renewal_costing_level: "repair_only" | "strong_renewal" | "full_renewal"` parameter to `compute_renovation_estimate_v4` that sets which tier `final_rehab` aliases. Default = `full_renewal` for back-compat (current behavior), or `strong_renewal` if product prefers a more conservative default headline — a product call.

### Change 3 — packages need no change
Packages already carry `estimate_scope` and already flow into the per-scope totals (verified: the moderate `bedroom/living_modernization` packages above sit in `optional_value_add`, the strong ones in `marketability_rehab`). The toggle inherits package routing for free.

### Granularity
- **v1:** the 3 stacked tiers above (repair / +strong renewal / +all renewal).
- **v1.5 (cheap):** per-scope inclusion (marketability separate from optional_value_add) — `totals_by_scope_capped` already separates them, so it’s a UI affordance, no backend work.
- **Avoid per-item toggles** — UX + maintenance burden, and it contradicts the “it’s a category” framing.

## The long pole: catalog scope-classification hygiene (data, not code)

`classify_estimate_scope` ([estimate_scope.py:94](../tools/estimate_scope.py#L94)) classifies via term-matching + `kind`/`tier`/posture (`_MARKETABILITY_TERMS`, `_VALUE_ADD_TERMS`, `_VISIBLE_REQUIRED_CONDITION_TERMS`). Term-matching mis-buckets sometimes. The toggle is only as trustworthy as this classification, so the real effort is an **audit pass**: confirm defects land in `required`, cosmetics in `marketability`, layout/modernization in `optional_value_add`. Spot-check on a handful of real properties (the `redfin_126231380` split above looks correct: bathroom finishes + popcorn + lighting → marketability/optional; gfci-repair correctly its own repair candidate).

## UX note (important)
For all-cosmetic properties, `repair_only` = `$0` (real example above). Present the tiers as a **stacked breakdown** (“Repair $0 · +Strong renewal $21–43k · +All renewal $28–71k”), not a binary that nukes the headline to zero. Surface renewal as a **range with a low-confidence / area-dependent caveat** — these costs are allowance-based and depend on total sqft, which is unknown from images (the original motivation for making them optional).

## Relationship to the dedup work / Option 3
- **Independent of Mechanism #1 & #2** (already shipped): those cut package fan-out (popcorn no longer mints 5 packages; living collapses 3→1). The toggle decides *scope inclusion*; dedup decides *how many packages exist*. They compose.
- **Depends on Option 3 for honesty when renewal is ON.** Package-only recurring items (popcorn has no `estimate` block → `package_evidence_only`, no line item) contribute `$0` unless they mint a package. After Mechanism #1 demotes them, a recurring cosmetic costs `$0` even in the full-renewal tier. [HANDOFF_whole_home_cosmetic_rollup.md](HANDOFF_whole_home_cosmetic_rollup.md) (Option 3) is what assigns that work a sane capped whole-home cost. **If the product defaults renewal OFF, Option 3 is low-urgency; if renewal ON is a common view, build Option 3 so the number isn’t under-counted.**

## Verification
1. **Unit test** (`tests/test_renovation_estimate_v4.py` or `tests/test_estimate_sanity.py`): a property with mixed scopes emits all three named tiers with `required ≤ resale_ready ≤ full_renewal`, and `full_renewal == required + marketability + optional_value_add` from `totals_by_scope_capped`.
2. **Real-property check:** `redfin_126231380` → required `$0`, resale_ready `$21–43k`, full_renewal `$28–71k` (matches the artifact above).
3. **Regression:** existing `final_rehab` / `final_rehab_resale_ready` values unchanged; new key additive only.
