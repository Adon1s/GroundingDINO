# Handoff: whole-home cosmetic roll-up (Option 3 — give demoted ambient work one property-level home)

## Status / framing

This is **Option 3** of the three-part fix for "a recurring cross-room finding mints multiple rehab packages." **Option 1 already shipped** (ambient-support demotion in `tools/rehab_packages.py`). Option 2 (catalog `support_scope` tagging) is a separate handoff. This one is independent of Option 2.

## The problem it solves

After Option 1, a recurring cosmetic support (e.g. popcorn ceiling across many rooms) no longer mints a per-room package in each driverless room. **What happens to its cost depends on whether the item produces a line item:**

- **Items that affect the estimate** (have an `estimate` block / `affects_estimate=True`) survive as normal costed line items (verified by `test_demoted_ambient_support_stays_visible_and_costed`, [tests/test_renovation_estimate_v4.py](../tests/test_renovation_estimate_v4.py)) — just **scattered** across per-room lines with no bundled representation.
- **Package-only items** — and popcorn is one: it has no `estimate` block, so it's `package_evidence_only` (no line item, routed via `_extract_package_only_candidates`, [renovation_estimate_v4.py:299](../tools/renovation_estimate_v4.py#L299)). Its **only** path to cost was minting a package. Once demoted, **it contributes $0**. Confirmed on `redfin_126231380`: popcorn was a support in 5 packages; after demotion it has no standalone cost.

So Option 3 isn't just "prettier bundling" — for package-only recurring items it's the **only** thing that assigns any cost at all. A whole-home trait ("smooth all popcorn ceilings") should read as ONE property-level package, costed across rooms (capped/sublinear) but justified once.

## Relationship to the renewal costing toggle (primary consumer)

Option 3's output lands in a **renewal** scope (`marketability_rehab` / `optional_value_add`), so its urgency is set by [HANDOFF_renewal_costing_toggle.md](HANDOFF_renewal_costing_toggle.md):

- If the product defaults **renewal OFF**, package-only popcorn correctly contributes `$0` to the headline and Option 3 is **low priority**.
- If **renewal ON** is a common view, Option 3 becomes **needed** — otherwise `final_rehab_full_renewal` (the full-renewal tier, now emitted on every run — see [HANDOFF_renewal_costing_toggle.md](HANDOFF_renewal_costing_toggle.md)) under-counts whole-home cosmetic work, showing `$0` for package-only recurring cosmetics like popcorn even with renewal on. The backend tier naming has shipped; build Option 3 when/if renewal-on accuracy matters.

## Current gap

`aggregate_whole_home_turnover` ([tools/rehab_packages.py:2475](../tools/rehab_packages.py#L2475)) already rolls per-room **turnover** packages into a single `interior_paint_flooring_refresh__whole_home` property-level package (≥2 distinct rooms; `package_level: property`; `verification_status: confirmed_by_rule`; `estimate_eligible: True`). But it only covers `package_category == turnover` ([tools/rehab_packages.py:2485](../tools/rehab_packages.py#L2485)). Recurring cosmetics like popcorn route to **modernization** support (via affinity, `_register_generic_support`, [:827](../tools/rehab_packages.py#L827)), so they are never aggregated today.

## The change

Add a modernization-side analogue (or generalize `aggregate_whole_home_turnover`) that surfaces demoted ambient cosmetic work as ONE property-level package:

- **Trigger:** the `ambient_support_ids` computed in `infer_package_candidates` (Option 1) — recompute the same support-role recurrence tally, or thread the set out of inference. When a support is ambient (recurs across ≥`_AMBIENT_SUPPORT_MIN_UNITS` units), it is a candidate for the roll-up.
- **Shape:** mirror `aggregate_whole_home_turnover` — `package_level: PACKAGE_LEVEL_PROPERTY`, `room: ROOM_WHOLE_HOME`, `verification_status: PACKAGE_VERIFICATION_CONFIRMED_BY_RULE`, `estimate_eligible: True`, `ui_eligible: True`. Likely reuse `PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH` and `MARKETABILITY_REHAB` scope, or introduce a sibling `..._cosmetic_refresh` type if the UI needs to distinguish "modernization refresh" from "turnover refresh."
- **Cost:** sum the per-room costs of the ambient findings (do **not** double-count — the per-room line items those findings produce must be absorbed/superseded by the aggregate, the same way `reconcile_packages_and_estimate_units` absorbs children, [:2883](../tools/rehab_packages.py#L2883); the `absorbed_by_package_id` marking is the mechanism). Decide explicitly whether the aggregate absorbs the line items or sits alongside them; the turnover aggregate's `contributing_package_ids` pattern is the reference. **Caveat for package-only items (popcorn):** they have *no* per-room line item or per-room package cost to sum (that's why they're `$0` today), so "sum the per-room costs" yields nothing for them — the aggregate must instead be **priced by a room-count-aware capped profile** (cf. the turnover aggregate's `pricing_tier`/`pricing_profile`). Sum-and-absorb only covers ambient items that *do* produce line items; the package-only case needs an explicit allowance. Resolve this in design.
- **Call site:** add next to the existing `aggregate_whole_home_turnover` call in `compute_renovation_estimate_v4` ([tools/renovation_estimate_v4.py:220](../tools/renovation_estimate_v4.py#L220)), appending to `packages` and `package_candidates_audit`.

## Open decisions for that session

1. Extend `aggregate_whole_home_turnover` to also accept modernization-category ambient cosmetics, vs. a new parallel aggregator. (Parallel is cleaner if UI labels differ.)
2. Absorption vs. coexistence of the per-room line items (avoid double-counting cost).
3. Minimum room count for the cosmetic roll-up (the turnover one uses ≥2 distinct rooms; ambient demotion uses ≥3 — pick one and document why).
4. Whether the roll-up should run Pass 2f or stay `confirmed_by_rule` like the turnover aggregate.

## Tests

- A roll-up emitted once across ≥N rooms; total cost equals the sum of the contributing per-room ambient findings (no double count).
- The contributing per-room cosmetic line items are absorbed/superseded with no double-count: the aggregate's `absorbed_total_*` reflects the superseded line items, and their per-room costs do not also appear in `totals_by_scope_*`. (Note: the legacy `package_total_below_absorbed_total_*` warning was **replaced** by the `cost_floor_applied` floor in Phase C of `reconcile_packages_and_estimate_units` ([rehab_packages.py:2999-3009](../tools/rehab_packages.py#L2999)) — assert on absorbed-total accounting / the floor, not the obsolete warning.)
- Regression: turnover aggregate behavior unchanged.

## Gotchas

- Living-area catalog items must use `scene_groups: ["living_areas"]` (not `["living"]`) or retrieval drops them.
- `_AMBIENT_SUPPORT_MIN_UNITS` is the shared knob with Option 1 — read it, don't fork a second threshold.
