# Catalog 3.2 — findings deliberately deferred

Written 2026-08-30 alongside the contextual-repair-support implementation.

Everything here was discovered while building Catalog 3.2 and **deliberately not
fixed**. None of it blocks the change. Each entry states what is true, the
evidence, and what a decision would cost — so a later session can act without
re-deriving it.

Related: `docs/HANDOFF_package_affinity_repair_family_gap.md` (the precursor that
raised the wear-vs-repair routing question), `reports_scratch/cat32_census/`
(the provider-free canary census produced by this work).

---

## 1. `kitchen_repair` has no catalog driver, so 8 of the 24 routes are inert

**Status: expected, verified, unresolved.**

`kitchen_repair` is a fully wired package type — it is in `VALID_PACKAGE_TYPES`
(`tools/rehab_packages.py:168`), has light and heavy tier specs
(`tools/rehab_packages.py:60-61`, `:359-365`), and has a dedicated profile
resolver `_resolve_kitchen_repair_profile` (`tools/rehab_packages.py:1473`).
**No catalog item routes to it.** Every kitchen defect — including
`cabinets_damaged_or_water_stained` and `missing_base_cabinets_exposed_subfloor`
— routes to `kitchen_modernization` as `package_driver`.

Contextual repair support only fires when the room already has a *primary*
defect/degradation repair driver. The kitchen has none, so its 8 marked routes
(paint, carpet, vinyl, worn cabinets, worn appliances, baseboards, wall scuffs,
generic worn flooring) can never fire.

The 24-route matrix was authored in full anyway, as specified. The canary census
confirms the prediction exactly: `kitchen_modernization` is **17 → 17** and no
kitchen candidate changed on any of the 18 properties.

**To resolve** someone must decide whether kitchen defects belong in
`kitchen_repair` rather than `kitchen_modernization`. That is a base-route
change — it moves real dollars and re-tiers existing kitchen packages — so it
was out of scope here. Pinned by
`tests/test_rehab_packages.py::test_real_catalog_marked_routes_have_a_wired_repair_family`,
which asserts `kitchen` has no repair driver: if that test starts failing, this
finding has been resolved and the assertion should be updated.

---

## 2. v4 keeps its pre-3.2 routing

**Status: deliberate. Steven's call to overturn.**

`infer_package_candidates` is shared by v4 (`tools/renovation_estimate_v4.py:170-177`)
and v5 (`tools/renovation_architecture/package_candidates.py:279`). The re-route
sits behind `contextual_repair_support`, which defaults to `False`; only v5
passes `True`.

Reasons for the default:

- The approved verification for this change is the provider-free replay, which
  is **v5-only**. A v4 change would ship unmeasured.
- v4 is on the retirement path, and re-routing it would require Pass 2f
  re-verification of the affected packages.

**Consequence to record:** v4 and v5 now disagree about where wear belongs in a
driven room, on top of their existing differences. The comparator gates on a 15%
v5-vs-v4 headline delta (`configs/renovation_architecture_cutover.json:5`,
`tools/compare_renovation_architecture_cutover.py:516`), and this widens it. The
frontend renders v4, so **the change is not user-visible until either the FE
adopts v5 or the flag is turned on for v4**.

Pinned by `TestContextualRepairSupportFlag::test_default_is_off_so_v4_routing_is_unchanged`.

---

## 3. `room_surrogate_id` on a condition is a first-wins collapse

**Status: known imprecision, inherited.**

`tools/renovation_architecture/conditions.py:187` sets
`room_surrogate_id = surrogate_ids[0] if surrogate_ids else ""` — the first
sorted surrogate wins. Where one condition draws evidence from several physical
rooms inside one billable unit, the scalar field names only one of them.

Contextual repair support matches driver and wear on that scalar field, so
surrogate isolation is exactly as precise as the condition layer is — no more.
The full tuple is available as `source_room_surrogate_ids`, and per-photo
surrogates survive on the evidence refs (`conditions.py:156-163`), if a later
session wants stricter matching.

The estimate-unit fallback (used only when either side has no surrogate) is
deliberately permissive: within a bucket the unit is already equal, so the
fallback means "same billable room, identity unknown".

---

## 4. `vanity_dated_style` is a `bathroom_repair` driver with `kind: modernization`

**Status: pre-existing, out of scope.**

The catalog routes `vanity_dated_style` (kind `modernization`) into
`bathroom_repair` as `package_driver`. A dated vanity can therefore open a
bathroom *repair* package today, with no defect present.

Catalog 3.2's driver gate excludes it — only `defect`/`degradation` drivers pull
wear in, so a dated-only bathroom cannot bootstrap contextual repair support.
But the change does **not** remove the underlying route. If "repair packages
should be opened only by concrete conditions" is a policy, this is where it is
violated, and it is untouched.

---

## 5. Ambient-support demotion is measured before the move

**Status: deliberate.**

`support_unit_tally` and `ambient_support_ids` (`tools/rehab_packages.py:2026-2035`)
are computed in the routing phase, off pre-move routes. The re-route runs after.

This is intentional: ambient demotion measures how many *distinct estimate units*
an item supports in, and moving an occurrence between families inside one unit
does not change that count. Worth re-checking if QP6 or a future rule calibrates
against ambient counts.

---

## 6. `damaged_drywall_or_cracks` has no bathroom repair route

**Status: out of scope by instruction.**

It drives `bedroom_repair` and `living_repair`, but routes to
`bathroom_modernization` as support and has no `bathroom_repair` entry
(`tools/issue_catalog_kind_v2.json`). Cracked drywall in a bathroom therefore
cannot open a bathroom repair package or pull wear into one.

---

## 7. `hard_flooring_scratched_or_worn` is not in the 10

It is already a `bedroom_repair` / `living_repair` **driver** (kind
`degradation`), and modernization-only in the kitchen. It is a driver of
contextual repair support, not a beneficiary. Noted because it looks like an
omission from the wear list and is not.

---

## 8. One work item can appear under two families in the same unit

**Status: pre-existing v5 collapse artifact; slightly reshaped by 3.2.**

v5 collapses conditions into work items by action code, so a single work item
can parent several occurrences. When one of those occurrences is marked and
moves while another is not, the *work item* becomes a child of both the
modernization and the repair candidate. Two real examples from the census:

- `redfin_80925528` / `PAINT_INTERIOR`: `peeling_or_discolored_paint` (marked)
  moved to `living_repair`; `paint_refresh_recommended` (unmarked) stayed in
  `living_modernization`.
- `redfin_126224899` / `VANITY_TOP_UPDATE`: `vanity_countertop_worn` @ bathroom_3
  moved; `vanity_countertop_dated` @ bathroom_1 stayed, because only bathroom_3
  had a repair driver.

Both are correct routing. **No occurrence is in two families** — that invariant
holds by construction (the occurrence is removed from one bucket before being
appended to the other) and is pinned by
`TestContextualRepairSupport::test_an_occurrence_is_never_in_both_families`.

The pattern is not new and got *less* common: 20 such work-item pairs in the
baseline, 14 after 3.2. It matters only for reporting that sums dollars per
family, where one work item's range can be counted under two families.

---

## 9. QP3's gate predicate is deliberately duplicated

`_qp3_application_gated` exists independently in
`tools/renovation_architecture/reconciliation.py:63-83` and
`tools/renovation_architecture/validators.py:183-279` — the validator copy is a
deliberate independent check, not dead code. Catalog 3.2 adds no gating
predicate, but any future one must land in **both**.

---

## 10. Pass 2a reachability update is blocked outside this repo

**Status: sequenced, not deferred indefinitely.**

The reachability audit that Catalog 3.2 changes the verdicts of
(`benchmark_pass2a_reachability.py`, `package_reference_v2`, `GOLD_POLICIES`)
exists **only** in the dirty sibling worktree
`C:\Users\Steven\PycharmProjects\rv-pass2a-bench`. This repo has no
`diagnostic_only` or reachability code at all. The transplant is specified by
`docs/HANDOFF_pass2a_bench_v3_port.md` (a 15-file port, not a branch merge).

Once that lands: flip the two bedroom-paint and bathroom-vanity targets from
`diagnostic_only` to `strict`, regenerate `gold_reachability.json`, and rescore
offline. Old 3.1 artifacts stay labelled historical — they do not demonstrate
3.2 package behaviour.

Note the audit's `_subsumption_bridge` is directional by design (repair never
subsumes modernization), which is precisely the wall Catalog 3.2 routes around.

**Port landed 2026-08-30; one trap for the flip.** The benchmark computes
packages through v4, which deliberately does not pass
`contextual_repair_support` (§2), and the ported reachability audit predates
the `repair_support_when_driven` marker — so flipping these targets to
`strict` before either the audit learns the marker or the §2 v4-flag question
is resolved produces strict-but-unreachable, which fails gold validation by
design.

---

## 11. Replay totals are an undercount while candidates are unreviewed

**Status: inherent to provider-free replay; do not misread.**

`scripts/replay_renovation_architecture.py` drops any candidate whose stored Sol
decision no longer matches — now including payload mismatches. Its
`replayed_totals` are computed over the **kept** set, so on the 10 changed
properties they exclude the 33 changed candidates entirely.

Those headline numbers are therefore an undercount, not a comparable headline.
The census reports counters and per-family counts and deliberately omits dollar
deltas. The real dollar effect is only measurable after the fresh Sol reviews,
which is also when the 15% headline-delta gate becomes meaningful.
