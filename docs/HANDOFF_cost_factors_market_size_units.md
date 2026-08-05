# Handoff: market/size cost factors + room-count scaling (slim design)

## Status / framing

Package tier ranges ([tools/rehab_packages.py](../tools/rehab_packages.py) tier constants), catalog cost blocks, heuristic bases ([tools/costing.py](../tools/costing.py)), and `GROUP_BUDGET_CAPS` ([tools/renovation_estimate.py:588](../tools/renovation_estimate.py#L588)) are all flat national numbers. A kitchen refresh in rural Ohio and in San Jose price identically; a 900 sqft condo and a 4,000 sqft house price identically; four worn-carpet bedrooms merged into `bedroom_primary` price as ONE `bedroom_refresh`. This handoff fixes all three.

**Design constraint from the user (binding):** the codebase is being actively de-bloated — no hardcoded special cases, no per-room or per-trade factor tables, no new data files for a handful of constants. Target ≤ ~150 lines of non-test code total. The design below was chosen for exactly that; do not "improve" it with geographic lookup tables or per-trade multipliers.

## The design — three numbers, two integration points

### Why no regional/zip table

`_PROPERTY_METADATA_KEYS` ([tools/artifact_writers.py:250](../tools/artifact_writers.py#L250)) extracts **no location fields** (raw scrape.json does carry address/lat/long, but they aren't extracted), so a geo table would require extending scrape extraction AND maintaining a zip→index dataset. Rejected. Instead: **list price per sqft is already scraped** (`price_per_sqft`, or derivable from `list_price / sqft`) — one continuous signal pricing the market's labor rates and finish expectations, with zero maintained data.

### Known bias: ppsf is the SUBJECT's ratio, not an area average (accepted, mitigated)

Verified against a real scrape.json: `price_per_sqft` is the listing's own `price / sqft` (e.g. 450000 / 3567 → 126), straight off the Redfin listing page. It is NOT a zip-level market average. That makes it **condition-endogenous**: a distressed fixer lists below its area's ppsf *because* it needs renovation, so the market factor systematically underestimates local labor costs for exactly the distressed properties this product targets (and overestimates for renovated flips).

Why the design accepts this: cross-market ppsf variation is 3–5× (Huntsville ~$126 vs San Jose $800+) while within-market condition discounts are typically 10–30% — the signal is mostly market. Two parts of the design deliberately blunt the residual bias, and must not be "simplified" away:

- the `PPSF_EXPONENT = 0.4` dampening compresses a 30% condition discount to ~13% factor error;
- the lower clamp (`0.75`) is a structural floor on how far a discounted fixer can drag its own factor down.

Forward-compatibility (cheap, do it now): make the ppsf lookup chain `area_price_per_sqft` → `price_per_sqft` → `list_price / sqft`, and **add `"area_price_per_sqft"` to `_PROPERTY_METADATA_KEYS`** ([tools/artifact_writers.py:250](../tools/artifact_writers.py#L250)) — metadata extraction filters by that tuple, so without the key the field would be silently dropped even when provided. Record which key was used in the audit dict.

The `area_price_per_sqft` key doesn't exist yet; two future sources are planned (neither is this handoff's job to build):
1. **Near term:** the frontend repo (`C:\Users\Steven\IntelliJProjects\realtorvision`) already computes zip-median ppsf baselines (`lib/topPicks/zipMedians.ts`) from its listing DB — it can pass the subject's zip baseline into `property_metadata` when invoking backend analysis.
2. **Later:** sold-price scraping (zip-level sold ppsf over trailing years), which is higher quality than active-ask medians.

This handoff only guarantees the slot exists and is preferred when populated.

### The factors (new module `tools/cost_factors.py`)

```python
# All constants live at the top of the module with these comments — they are
# the entire "data" of this feature. No JSON file for six numbers.
PPSF_BASELINE = 230.0      # ~national median list $/sqft (2025); factor 1.0 here
PPSF_EXPONENT = 0.4        # materials are national, labor is local — sublinear
PPSF_CLAMP = (0.75, 1.5)
SQFT_BASELINE = 1800.0     # typical single-family listing
SQFT_EXPONENT = 0.3        # more surface area, but fixed costs don't scale
SQFT_CLAMP = (0.9, 1.25)
PER_EXTRA_UNIT_FACTOR = 0.6  # 2nd+ room in a merged unit: shared mobilization
MAX_UNITS_PER_PACKAGE = 4

def resolve_property_cost_factor(property_metadata) -> Tuple[float, Dict[str, Any]]:
    """One multiplicative factor + audit dict. Neutral (1.0) on missing data."""
```

`factor = market_factor * size_factor` where `market_factor = clamp((ppsf / 230) ** 0.4, 0.75, 1.5)` and `size_factor = clamp((sqft / 1800) ** 0.3, 0.9, 1.25)`. Read `price_per_sqft` first, else `list_price / sqft`, via the same alias-tolerant number parsing as [tools/estimate_sanity.py:28](../tools/estimate_sanity.py#L28) (`_metadata_number` — reuse or mirror it). Missing input ⇒ that factor is 1.0 and the audit dict says why. Sanity: ppsf $100 → 0.75 (clamped); $500 → ≈1.36; $800 → 1.5 (clamped).

### Integration point 1 — one post-hoc uniform scaler (the property factor)

**Key insight that keeps this slim:** every dollar operation in the pipeline — sums, group-cap mins, floor maxes, integer splits — is positively homogeneous, so `scale(output, f)` ≡ scaling every input *including the caps*. Therefore the property factor needs exactly ONE integration site: scale the assembled v4 estimate once, instead of threading a multiplier through costing.py, the tier tables, and four `GROUP_BUDGET_CAPS` read sites.

`scale_estimate_dollars(estimate: dict, factor: float) -> None` (also in cost_factors.py, ~40 lines):

- Recursive walk of the dict/list tree. Scale any int/float value whose key **ends with `_low` or `_high`, or equals `"low"`/`"high"`** (this suffix vocabulary covers every dollar field in the v4 output: `cost_low`, `allocated_high`, `absorbed_total_low`, `risk_exposure_high`, nested `{"low": .., "high": ..}` buckets, …). Round to int where the original was int.
- After scaling a dict, **recompute** `"midpoint"` / `"cost_midpoint"` as `(low + high) // 2` where those keys exist — never scale midpoints independently (rounding drift breaks the `midpoint == (low+high)//2` invariant tests may assert).
- **Track visited object ids.** Package dicts are aliased between `packages`, `package_candidates`, and reconciliation audit refs (this aliasing is deliberate, established by the subsumption work). Without an `id()`-visited set you will double-scale shared dicts. This is the one genuinely dangerous bug in this design — test for it explicitly.
- `factor == 1.0` short-circuits to a no-op.

**Call site (exactly one):** in `compute_renovation_estimate_v4` ([tools/renovation_estimate_v4.py](../tools/renovation_estimate_v4.py)), after `project_scope_breakdown` is attached and **before** `build_estimate_sanity_flags` — sanity flags compare totals against list price and must see scaled dollars. Attach the audit: `v4_estimate["cost_adjustment"] = {factor, market_factor, size_factor, ppsf, sqft, baselines, reasons}`. Add `"cost_adjustment"` to `provenance.v4_phases_applied`.

The standalone v3 estimate stays untouched (v4 is the product surface; v3 is kept for comparison — the docstring already promises "v3 output is left untouched").

### Integration point 2 — unit-count factor (per merged room package)

The bedroom-underpricing gap: `build_estimate_units` merges multiple bedroom surrogates into `bedroom_primary`, and the package prices one room. The fix is generic, not per-room: in `_build_package_candidate` ([tools/rehab_packages.py:1689](../tools/rehab_packages.py#L1689)), after tier resolution + escalation:

```python
n = min(max(len(source_room_surrogate_ids), 1), MAX_UNITS_PER_PACKAGE)
unit_factor = 1 + PER_EXTRA_UNIT_FACTOR * (n - 1)   # 1.0, 1.6, 2.2, 2.8
```

Multiply `cost_low/cost_high` (and `candidate_cost_*`) by `unit_factor`, append `f"unit_count_factor={unit_factor}_n={n}"` to `level_decision_notes` when n > 1. That's the whole change — it uses `source_room_surrogate_ids`, which `_build_package_candidate` already receives from `_source_room_ids_for_unit`.

Why this composes correctly with everything downstream, for free:

- **Bathroom expansion**: expanded per-surrogate copies get `source_room_surrogate_ids = [surrogate_id]` ([`_build_expanded_bathroom_package`](../tools/rehab_packages.py#L2396)), and expansion deep-copies the original package's cost. If the original merged bathroom package was scaled (n=2), the expansion produces two full-cost copies from an already-scaled original — **double counting**. Guard: expansion must rebase each copy's cost to the un-scaled tier range (divide by the recorded unit_factor, or store `pre_unit_factor_cost_low/high` on the package and use those). This is the one interaction requiring care; test it.
- **Whole-home turnover aggregate** sums per-room packages — picks up scaled values automatically.
- **Phase C floor / escalation**: escalation runs before the factor (tier-table space); the floor compares package cost vs absorbed children, and a multi-surrogate unit's absorbed children span those same rooms — consistent.
- **Subsumption** compares categories/tiers, not costs — unaffected.

Mild known overlap: `size_factor` (total sqft) and unit-count both grow with house size. Both are deliberately dampened (0.3 exponent; 0.6 per extra room) — accept it, don't add a correction term.

## Where to change

| File | Change |
|---|---|
| `tools/cost_factors.py` (new) | constants + `resolve_property_cost_factor` + `scale_estimate_dollars` (~100 lines) |
| [tools/renovation_estimate_v4.py](../tools/renovation_estimate_v4.py) | one call site before sanity flags; `cost_adjustment` audit key; provenance entry |
| [tools/rehab_packages.py](../tools/rehab_packages.py) | ~6 lines in `_build_package_candidate` (unit factor); expansion rebase guard in `_build_expanded_bathroom_package` |
| [tools/artifact_writers.py](../tools/artifact_writers.py) | one entry: `"area_price_per_sqft"` added to `_PROPERTY_METADATA_KEYS` |

Nothing else. If the implementation wants to touch costing.py, the tier tables, or GROUP_BUDGET_CAPS, the design has been misunderstood — stop and re-read "Integration point 1".

## Tests

Run with `.venv\Scripts\python.exe -m pytest`. **7 pre-existing unrelated failures** (test_model_comparison ×2, test_rerun_pass_2f_artifact ×5) — baseline first. `tests/` is untracked — never `git stash` test paths.

New `tests/test_cost_factors.py` + additions to existing suites:

1. **Identity:** no metadata (or factor forced 1.0) ⇒ v4 output deep-equal to pre-change output. This is the parity keystone.
2. Factor math: known ppsf/sqft ⇒ expected factor; clamps at both ends; missing ppsf ⇒ market 1.0 with audit reason; lookup chain order: `area_price_per_sqft` (if present) beats `price_per_sqft` beats derived `list_price / sqft`; audit records which key was used.
3. Scaler: nested `{"low","high"}` buckets scaled; `*_low`/`*_high` keys scaled; midpoints recomputed (invariant `midpoint == (low+high)//2` holds post-scale); counts/severities/`confidence_score` untouched.
4. **Aliasing:** a package dict referenced from both `packages` and `package_candidates` is scaled exactly once.
5. Unit factor: package with 3 source surrogates gets 2.2×; single-surrogate gets 1.0×; n clamps at `MAX_UNITS_PER_PACKAGE`; decision note appended.
6. Expansion interplay: merged 2-surrogate bathroom package (unit_factor 1.6) expands into two copies each at the **un-scaled** tier cost — combined ≠ 2 × 1.6 × tier.
7. End-to-end: v4 with `price_per_sqft=460, sqft=2400` ⇒ headline tiers scale by the computed factor; `cost_adjustment` audit present; sanity flags computed on scaled totals.

## Gotchas

- The scaler's suffix rule (`_low`/`_high`/`"low"`/`"high"`) was checked against the current v4 output vocabulary and has no false positives today. If a future field like `confidence_low` ever appears, add an explicit exclusion set to the scaler — start the set empty, don't pre-engineer it.
- Scale **only the v4 estimate dict**. photo_intel costing, rehab scores (severity-point-based, not dollars), and the v3 estimate are out of bounds.
- `build_estimate_sanity_flags` ordering (must run after scaling) is the only orchestration-order constraint.
- Round-trip integer discipline: original ints stay ints after scaling (`round()`, not `int()` truncation) so `_split_integer`-style sums elsewhere don't drift in tests.
- Baseline constants (`PPSF_BASELINE=230`, `SQFT_BASELINE=1800`) are judgment calls, stated in code comments as such — flag them in the PR description for the user to tune, don't research-rabbit-hole them.
- `compare_reno_estimates.py` and stored golden artifacts will show across-the-board deltas on metadata-bearing properties — expected and intended; say so in the PR description.

## Rejected alternatives (don't relitigate)

- **Zip/metro regional cost index table** — location exists in raw scrape.json (address/lat/long) but isn't extracted into metadata, and the approach requires a maintained zip→index dataset regardless; ppsf is already scraped and captures most of the same variation. If the subject-ppsf condition bias proves material in practice, the right escalation path is an *area-level* ppsf captured at scrape time (see lookup chain above), not a geo table in this repo.
- **Threading a multiplier through costing.py / tier tables / caps** — 4+ integration sites doing what one homogeneity-based post-hoc scaler does; more code, more drift surface.
- **Per-trade or per-room factor tables** (labor-heavy vs material-heavy scaling) — real effect, but exactly the hardcoded-special-case bloat being removed from this codebase. Revisit only with calibration data (#4).
- **Generalizing bathroom-expansion to all room types** for the unit-count gap — heavyweight; the one-line multiplicative factor captures the economics (shared mobilization, per-room materials) without new pipeline machinery.

## Out of scope

- Cost-knowledge consolidation + tier-vs-catalog calibration audit (#4 — natural successor once this lands; the factor constants would move there if a costs data file ever materializes).
- Finish-level inference from photos; year_built-based system-age adjustments (interesting, separate).
- Any change to retrieval, routing, subsumption, or validation.
