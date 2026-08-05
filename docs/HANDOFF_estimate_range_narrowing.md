# HANDOFF: Narrow the headline rehab range to decision-grade for investor upside

## Why

The renovation estimate exists to feed **upside = ARV − purchase − rehab − transaction/holding**
for fixer-upper investors. The ARV side is already anchored (square-mile historical sold data,
frontend). Rehab is now the noisiest term: current headline ranges run **1.8x–3x wide**
(e.g. redfin_80860084 resale-ready $79,420–$142,167; typical "average fixer" output $30k–$80k).

The deal-math problem: on a typical target deal (~$120k purchase, ~$200k ARV), selling/holding
costs plus target profit mean the entire go/no-go decision lives inside a ~$30k window. A $50k-wide
rehab range spans the whole window — the tool cannot separate a great deal from a dead one.
Photo-only estimating has an honest floor of ±20–30%; investors can work with ±20–25%. Target:
**headline band ≈ midpoint ±20–25%**, without lying about per-item uncertainty and without
touching worst-case semantics (being pessimistic is `worst_case_exposure`'s job).

Four workstreams, ordered. B and C are straight bug fixes (do first — they cut the high end for
free). A is the main design change. D/E are calibration and pinning. Findings below are verified
against code and the 2026-07-01 production artifacts
(`C:\Users\Steven\IntelliJProjects\realtorvision\artifacts\redfin_*\202607*\photo_intel_debug.json`);
implementation specifics are the new session's to decide unless marked as a constraint.

---

## B. Bug: stale `pricing_tier` after tier escalation (dollar impact via subsumption)

**Finding.** `_build_package_candidate` ([tools/rehab_packages.py](../tools/rehab_packages.py)
~1703–1719) gets `spec, pricing_tier, decision_notes = _resolve_pricing_profile(...)`, then
`_escalate_pricing_tier_if_undercut` may replace `spec` (refresh → partial → full when absorbed
driver base costs exceed the tier ceiling) — but `pricing_tier` is never reassigned. The package
ships with full-rehab pricing and a "refresh" label.

**Consumers of the stale label:**
- `apply_same_unit_package_subsumption` (~2331): `_MODERNIZATION_TIER_RANK` (2192) and
  `_REPAIR_SUBSUMING_MODERNIZATION_TIERS = {"partial_rehab", "full_rehab"}` (2196) key off
  `pricing_tier`. A full-rehab-priced package labeled "refresh" **fails to subsume its same-unit
  repair package → double count.**
- The turnover-downgrade check (~2379, `pricing_tier == "turnover_std"`).
- UI display (a "$24k–$56k refresh" reads as absurd and destroys trust).

**Live evidence (redfin_80860084 artifact):**
- `bathroom_modernization__bathroom_primary`: tier "refresh", cost $24,000–$56,000
  (= `BATHROOM_FULL_REHAB` 15k–35k × 1.6 unit factor) coexisting with
  `bathroom_repair__bathroom_primary` $4,000–$12,800 → ~$4k–12.8k counted twice.
- `kitchen_modernization__kitchen_primary`: tier "partial_rehab", cost $48,000–$112,000
  (= `KITCHEN_FULL_REHAB` 30k–70k × 1.6).
- `bedroom_repair__bedroom_2/3`: tier "repair_light", cost $2,500–$8,000 (= heavy tier).

**Constraint:** after the fix, `pricing_tier` must always reflect the spec actually priced. How is
open — derive the generic tier from the escalated profile key (profile keys end with the tier
name, e.g. `bathroom_full_rehab` → `full_rehab`), or carry labels through
`_PRICING_TIER_ESCALATION` (1618). Note `absorption_scope` already correctly uses the escalated
`pricing_profile` — don't break that. Add regression tests in `tests/test_rehab_packages.py`
(escalated package subsumes same-unit repair; label == priced tier).

## C. Bug: ×1.6 multi-room factor applied to same-room merges

**Finding.** `_build_package_candidate` (~1722):
`unit_count = min(max(len(source_room_surrogate_ids), 1), MAX_UNITS_PER_PACKAGE)` then
`factor = 1 + PER_EXTRA_UNIT_FACTOR × (unit_count − 1)` (constants in
[tools/cost_factors.py](../tools/cost_factors.py): 0.6, max 4). But estimate units merge repeated
surrogates precisely because they're **probably the same physical room** — then the package prices
the merged unit as N rooms. This re-introduces the photo-order multi-counting that estimate-unit
merging was built to kill (a kitchen photographed twice = 1.6 kitchens).

**The signal needed already exists.** Estimate unit records
([tools/estimate_units.py](../tools/estimate_units.py), `add_unit`) carry `merge_reason` and
`confidence`:
- Same-room semantics (physical count 1): `single_family_default_one_kitchen` /
  `repeated_kitchen_surrogates_merged_by_default` (confidence `default_assumption`),
  `repeated_bathroom_surrogates_merged_conservatively`, `metadata_single_bath_cap`,
  `metadata_single_bedroom_cap`.
- Genuinely-spans-rooms semantics (factor justified, count = capped room count):
  `bathroom_metadata_cap_applied`, `bedroom_metadata_cap_applied`,
  `multiple_kitchens_allowed_by_metadata_or_evidence` (distinct units, not merged).

The `estimate_units` list is already passed into `infer_package_candidates`
([tools/renovation_estimate_v4.py:164](../tools/renovation_estimate_v4.py)); threading the unit
record (or a derived physical-room count) down to `_build_package_candidate` is the gap to fill.
Prefer data-driven (read the unit record) over string-matching reason names if a cleaner field can
be added at unit-build time (e.g. `physical_room_count` on the unit record — one authoritative
place). **Watch the interplay:** `_build_expanded_bathroom_package` / bathroom expansion rebases
per-surrogate copies using `pre_unit_factor_cost_low/high` (~1816–1821, ~2462) — keep that path
consistent. Note metadata caps only exist once property metadata flows
(see `HANDOFF_property_metadata_wiring.md`); with empty metadata, default merges are the norm, so
this fix matters most right now.

**Expected effect on the live artifact:** kitchen package $48k–112k → $30k–70k; bathroom
$24k–56k → $15k–35k, before B even applies.

## A. Design change: stop summing extremes at the headline — correlated rollup

**Finding.** Every aggregate is `Σ(item lows) → Σ(item highs)`: `compute_group_estimate` (sum
stack), reconciliation Phase D/E, `_build_scope_rollups`
([tools/rehab_packages.py:3490](../tools/rehab_packages.py)), then
`build_scope_headline_tiers` ([tools/estimate_scope.py:411](../tools/estimate_scope.py)). The
aggregate low assumes every line item and package lands best-case simultaneously; the high assumes
all worst-case. With 10–20 items + packages, honest aggregate width grows ~√N (partially
correlated), not N. No variance treatment exists anywhere. This is why per-item ranges that are
individually defensible ($2.5k–8k bathroom repair) compound into a 2–3x headline.

**Design intent (constraints, not implementation):**
- Headline tiers (`final_rehab_required` / `final_rehab_resale_ready` / `final_rehab_full_renewal`)
  become **midpoint ± blended width**: treat each contributor's half-width
  `w_i = (high_i − low_i)/2`; blended `W = √(Σw²) + ρ·(Σw − √(Σw²))` with `ρ ≈ 0.4` as the
  starting default. ρ (and any interval-interpretation constant) live in `cost_factors.py` next to
  the other tunable judgment calls — one place, documented as judgment.
- Accumulate `Σw` and `Σw²` per estimate scope where contributors are already iterated —
  `_build_scope_rollups` is the natural spot (children + packages per scope, capped and raw).
  New session decides whether widths come from capped or raw child allocations; state the choice in
  the audit block.
- **Invariants (enforce + test):**
  1. Shrunk band ⊆ raw band, low ≥ 0.
  2. Tier nesting must survive: required ⊆ resale ⊆ renewal (independent shrinking can break
     monotonicity — clamp lows non-increasing/highs non-decreasing across tiers after shrink).
  3. Homogeneity: `scale_estimate_dollars` runs after reconciliation and uniformly scales all
     `*_low/*_high` — the blend is positively homogeneous (√(Σ(cw)²) = c√(Σw²)), so order is safe;
     just make sure new dollar fields follow the `_low/_high` naming so they get scaled, and
     midpoints stay consistent with `_MIDPOINT_TRIPLES` recompute.
  4. Keep raw sums in the payload for audit (e.g. `raw_low/raw_high` or a band-audit block with
     method, ρ, Σw, √Σw²).
- **Do NOT shrink:** `visible_rehab`, `package_adjusted_rehab`, `latent_risk_exposure`,
  `worst_case_exposure`, per-group/per-item ranges, and the sanity-flag inputs (estimate_sanity
  reads package_adjusted/worst_case highs — unaffected by design).
- Frontend contract: `RehabRange` normalization at
  `realtorvision/lib/types/renovationEstimate.ts:749–752` tolerates the current shape; check
  `docs/frontend_contract_packages.md` before adding/renaming fields. Note
  `final_rehab_full_renewal` is currently **missing from the TS type** (216–219) — add it while
  in there if D lands.

**Effect sizing:** a $30k–80k headline (m=55k, Σw=25k) over ~12 contributors lands around
$45k–67k at ρ=0.4 — decision-grade, and still honest.

## D. Pin upside to `resale_ready`

**Finding.** The number investors' upside math consumes is `capexLow/MostLikely/High`, synced in
`realtorvision/lib/reno/updateRunWithEstimate.ts` (~84–131) from
`getPrimaryRehabRange` (`realtorvision/lib/types/renovationEstimate.ts:793`), which currently
prefers `final_rehab` — and backend `final_rehab` = `package_adjusted_rehab` (all scopes,
including optional value-add packages). For upside, the correct budget is "renovate to the quality
of the comps that set the ARV" = **`final_rehab_resale_ready`** (required + marketability, capped;
excludes optional value-add). Since the square-mile sold comps are mostly renovated houses, that
tier matches the ARV basis.

**Decision for the new session — single change point, two options:**
1. Frontend: make `getPrimaryRehabRange` prefer `final_rehab_resale_ready` (fallback chain to
   `final_rehab` → `package_adjusted_rehab` for old artifacts). Smallest blast radius.
2. Backend: redefine `final_rehab`'s basis to resale-ready. Moves every consumer at once
   (capex sync, notifications via `economics.capexLow/High`, UI) — but audit all readers of
   `final_rehab` and the frontend contract doc first.

Either way, keep the tier trio visible in the UI (investors think in scopes: "cosmetic flip vs
gut", not ± bands).

## E. Calibration: tier tables are homeowner-retail, audience is investors

**Finding.** Package tier constants ([tools/rehab_packages.py:53–91](../tools/rehab_packages.py)):
`KITCHEN_FULL_REHAB` 30k–70k, `BATHROOM_FULL_REHAB` 15k–35k are retail-remodel prices. Flipper
economics (investor-grade finishes, own crews) typically run **50–70% of retail**: kitchen
~$15–30k, bath ~$8–15k mid-market. Nothing in the system has ever been checked against a real
quote — all audit tooling validates detection, not dollars.

**Guidance:**
- Recalibrate the full/partial tiers toward investor-grade. **Interplay warning:** shrinking tier
  ceilings makes `_escalate_pricing_tier_if_undercut` (driven by `_absorbed_cost_estimate` = sum of
  driver **catalog base costs**) escalate more often, and the Phase C absorbed-cost floor (~3122)
  re-inflates whatever escalation misses. The big catalog allowances
  (`issue_catalog.json`: e.g. `outdated_kitchen_finishes` base_high $20k / cap $60k,
  `outdated_bathroom_finishes` base_high $15k / cap $35k) must be recalibrated **jointly** with
  the tiers or the change will be fought by the guardrails.
- Do it as a measured pass: recompute the 17 July artifacts before/after
  (`tools/backfill_reno_v4.py` recomputes v4 in place; `tools/compare_reno_estimates.py` diffs) and
  eyeball per-property movements. Do this AFTER B/C/A so effects aren't confounded.
- Cheap guard to add while there ($/sqft cross-check, `tools/estimate_sanity.py`): flipper
  rules of thumb — light cosmetic ~$15–25/sqft, medium ~$25–45, gut ~$50–100. Flag when the
  resale-ready midpoint / sqft falls outside plausible bands (both directions). Requires sqft →
  depends on `HANDOFF_property_metadata_wiring.md` landing first.
- Ground truth beats all of this: even 10–15 real data points (contractor walkthrough bids on
  analyzed listings, or completed flips in target zips: purchase → permit scope → resale) to check
  rank-order and midpoint bias. Rank-order correctness matters more than absolute accuracy for
  ranking deals.

## Ordering & dependencies

1. **B, C** — pure bug fixes, immediately cut inflated highs. No dependencies.
2. **A** — correlated rollup (backend), then **D** (one consumer change).
3. **E** — after A–D and ideally after metadata wiring (so the market factor participates and
   sqft flags work), with the before/after artifact comparison.

## Verification

- Tests: `.venv\Scripts\python.exe -m pytest tests/test_rehab_packages.py tests/test_estimate_scope.py tests/test_renovation_estimate_v4.py -q`
  (bare `python`/`py` won't resolve the venv; 7 pre-existing failures exist elsewhere on
  `reno_estimate_refactor` — scope to touched files; `tests/` is untracked — never `git stash`
  test paths).
- Recompute a July artifact with `tools/backfill_reno_v4.py` and diff with
  `tools/compare_reno_estimates.py`. Concrete expectations for redfin_80860084 after B+C alone:
  kitchen package 48–112k → 30–70k; bathroom modernization 24–56k → 15–35k AND it now subsumes the
  4–12.8k repair package; resale-ready high drops from $142k to roughly $95–105k before A even
  applies. After A, the resale-ready band should tighten to roughly ±20–25% around its midpoint.
- Invariant tests for A: band containment, tier nesting, homogeneity under
  `scale_estimate_dollars`, midpoint consistency.

## Non-goals (explicitly out of scope)

- Don't shrink or cap `worst_case_exposure` / `latent_risk_exposure` — pessimism is their job.
- Don't cap estimates by list price (distressed properties can legitimately need near-price rehab).
- No per-metro cost database, no per-item variance modeling, no new config files — constants in
  `cost_factors.py`, judgment documented in comments, per the slim-code philosophy (line budgets;
  data-driven over special cases).
