# Handoff: rehab_packages.py architectural review + calibration fixes

> **Superseded note (June 2026):** package routing described in this document
> via flat catalog fields (`package_type`/`package_role`/`package_category`/`room`)
> and the in-code `PACKAGE_AFFINITY` dict has been replaced by per-item
> `package_affinity` blocks in `tools/issue_catalog.json` (room-keyed;
> category/room derived from `package_type` at load via
> `build_package_affinity`). References below to flat fields or
> `PACKAGE_AFFINITY` are historical.

You're picking this up after a bathroom-catalog overhaul shipped to `tools/issue_catalog.json` (15 new bathroom items + guardrails + severity-inversion fixes) and a small extension to `tools/rehab_packages.py` (`_STRONG_SIGNAL_CATALOG_IDS` grew from 1 entry to 4). All 204 tests in `tests/test_rehab_packages.py`, `tests/test_renovation_estimate_v4.py`, and `tests/test_renovation_estimate.py` are green as of this writing.

**Your job in this session is to confirm a diagnosis and produce a plan. Do not implement yet.** Enter plan mode and exit with a written plan when you have answers.

---

## The single critical question

**Hypothesis**: `tools/rehab_packages.py` contains two parallel package-inference paths. One is the production path, one is dead code kept alive only by tests. Specifically:

- **`infer_package_candidates`** (line 1172) → `_resolve_*_profile` dispatch (lines 874-1018) — catalog-driven, strength/confidence-scored, integrates with Pass 2f verification. **Production.**
- **`infer_packages`** (line 1698) → `_infer_kitchen_package` (1853) / `_infer_bathroom_package` (1925) / `_infer_room_refresh` (2027) — hand-coded component thresholds, no strength/verification, simpler schema. **Suspected dead code.**

### Evidence supporting the hypothesis

Run these greps to confirm:

```bash
# Production callers of each path (exclude tests)
rg -n "infer_package_candidates\(" --glob "!tests/**"
rg -n "infer_packages\(" --glob "!tests/**"

# All imports of rehab_packages
rg -n "from tools.rehab_packages import|import.*rehab_packages"
```

Expected findings (verify these match):

- `infer_package_candidates` called from:
  - `tools/renovation_estimate_v4.py:151` (the v4 pipeline, current production entrypoint)
  - `tools/rerun_pass_2f_artifact.py:262` (artifact reprocessing tool)
- `infer_packages` called from:
  - **Only `tests/test_rehab_packages.py`** (~20 call sites: lines 1044, 1079, 1097, 1115, 1123, 1145, 1178, 1206, 1232, 1262, 1285, 1302, 1318, 1357, 1378, 1433, 1447, 1472, 1482, 1494)

Cross-check production flow:
- `tools/artifact_writers.py:818-819` calls `compute_renovation_estimate_v4` — that's the production entry
- `tools/renovation_estimate_v4.py:79` defines `compute_renovation_estimate_v4`
- At line 151 it calls `infer_package_candidates` (the production path)
- At line 205 it calls the v3 `compute_renovation_estimate` for the quick estimate, which does NOT touch package inference at all (verify: `rg "infer_packages|infer_package_candidates" tools/renovation_estimate.py` should return nothing)

### Stale relics that point to the same conclusion

- **`tools/rehab_packages.py:6-13`** docstring still advertises `infer_packages` as "Public API". Stale.
- **`docs/HANDOFF_bathroom_catalog_overhaul.md:165`** says "`_infer_bathroom_package` rule changes — already done" — was treating dead code as live.
- **`.claude/settings.local.json:93`** whitelists a python -c that imports `infer_packages` for command-line sanity checks.

### What to verify in this session

1. Re-run the greps above to confirm the call-site counts.
2. **Open 3-5 of the `infer_packages`-using tests** (e.g., `tests/test_rehab_packages.py:1044, 1145, 1262`) and skim what scenarios they exercise. The risk: a test might be touching production-relevant behavior via a side door (e.g., calling `_build_package` directly or testing `reconcile_packages_and_estimate_units` with hand-built packages that match the legacy schema). If you find one, flag it — but otherwise this is a clean delete.
3. Check whether `infer_packages` appears in any `__all__`, public exports, or API documentation outside `tools/`.

### Decision point

If the hypothesis holds:
- **Option A (recommended): delete the legacy path entirely.** Remove `infer_packages`, `_infer_kitchen_package`, `_infer_bathroom_package`, `_infer_room_refresh`, and any helpers used only by those four functions. Update the module docstring. Migrate the legacy tests to exercise `infer_package_candidates` instead.
- **Option B: keep `infer_packages` as a thin backward-compat shim** that delegates to `infer_package_candidates`. Reduces breakage risk but preserves the schema-divergence issue (see Issue 5 / "smaller observations" below).
- **Option C: keep both** and rationalize the schema divergence. Significantly more work; only justified if there's a use case I'm missing.

Migrating the legacy tests is the biggest unknown. Cost depends on how 1:1 the scenarios map between paths — the legacy path lacks strength/confidence scoring, so equivalent assertions may need adjustment.

---

## Five real bugs in the production path (the "rest of the list")

All five are independently valid regardless of the architectural decision above. They live in the production path (`infer_package_candidates` and friends), so they must be fixed even if you keep the legacy path. Numbering matches the original review.

### Issue 1 (architectural) — see above

### Issue 2 — Package pricing tiers don't reflect what they absorb

**Symptom**: a single sev-3 `outdated_or_damaged_vanity` driver:
- Routes to `_resolve_bathroom_repair_profile` ([tools/rehab_packages.py:985](tools/rehab_packages.py#L985)) via `package_type: bathroom_repair`.
- No "heavy" components in scope → returns `BATHROOM_REPAIR_LIGHT` at $400–$2,500.
- Catalog cost on that single item: $500–$6,000.
- Package absorbs the line item; `pkg.cost - absorbed_total` goes negative.
- `_build_scope_rollups` clips to 0 — the difference is silently lost.

You should see `package_total_below_absorbed_total_low/high` warnings in Phase C of `reconcile_packages_and_estimate_units` when this fires.

**Fix (recommended)**: floor approach. At the end of each `_resolve_*_profile`, take `max(tier_cost, sum_of_absorbed_low/high)` so the package never undercuts what it ate. Simpler than reshuffling tier dispatch.

**Alternative**: smarter dispatch — if any single driver's catalog cost range exceeds the candidate tier, escalate (light→heavy, refresh→partial, partial→full). More invasive.

**Where to fix**: `_resolve_bathroom_repair_profile`, `_resolve_bathroom_modernization_profile`, `_resolve_bathroom_turnover_profile`, and the corresponding kitchen profiles (`_resolve_kitchen_*` at lines 874-944). Probably also worth a helper that does the `max()` in one place rather than duplicating it.

### Issue 3 — `classify_component` conflates trade-bucket with severity weight

**Symptom**: in [tools/rehab_packages.py:479-523](tools/rehab_packages.py#L479), `classify_component` returns the bare string `"electrical"` for any electrical trade bucket. Then in `_resolve_bathroom_repair_profile`:

```python
heavy_components = components.intersection({"plumbing", "electrical", "moisture", "flooring"})
if drivers and heavy_components:
    return BATHROOM_REPAIR_HEAVY, "repair_heavy", notes  # $2,500-$8,000
```

So a single missing GFCI ($150–$1,500 catalog) or a missing exhaust fan ($120–$1,200) triggers `BATHROOM_REPAIR_HEAVY` at $2,500-$8,000. This is the inverse of Issue 2 — package cost wildly *exceeds* absorbed cost for trivially light electrical items.

**Fix options**:
- **A**: Split `electrical` into `electrical_heavy` (panel work, large rewiring, exposed-wire `visible_electrical_risks` at sev 4) vs `electrical_light` (GFCI, exhaust fan, dated outlets). Only `electrical_heavy` triggers `repair_heavy`. Requires updating `classify_component` and every consumer in `_resolve_*_profile`.
- **B**: Weight the "heavy" check by severity — `severity <= 2` electrical doesn't trigger heavy. Simpler change but coupling severity to bucket logic.
- **C**: The same floor fix from Issue 2 in reverse — cap downward when the only "heavy" signal is a low-cost item.

A and B are mutually exclusive; C can stack on either. Recommend evaluating A first since it's the cleanest semantic separation.

### Issue 4 — Strong-signal sentinel maintenance overhead

**Current state** ([tools/rehab_packages.py:539-545](tools/rehab_packages.py#L539)):

```python
_STRONG_SIGNAL_CATALOG_IDS = frozenset({
    "missing_base_cabinets_exposed_subfloor",
    "missing_bathroom_tile_exposed_substrate",
    "active_water_damage_bathroom",
    "missing_vanity_exposed_plumbing",
})
```

And in `_has_strong_signal` (lines 544-553):

```python
if (candidate.severity or 0) >= 3 and cat_id == "outdated_kitchen_finishes":
    return True
if (candidate.severity or 0) >= 3 and cat_id == "outdated_bathroom_finishes":
    return True
```

Now that the catalog overhaul bumped `outdated_or_damaged_vanity` to sev 3 with `package_role: package_driver`, it logically deserves strong-signal treatment but is missing from both lists. Same risk with future sev-3 drivers — every one needs manual sentinel-list maintenance.

**Recommended fix**: replace the per-catalog-id hardcoded checks with a generic rule:

```python
def _has_strong_signal(candidates):
    for candidate in candidates or []:
        cat_id = candidate.catalog_item_id or ""
        if cat_id in _STRONG_SIGNAL_CATALOG_IDS:
            return True
        cat = catalog_lookup.get(cat_id, {})
        if (cat.get("kind") == "defect"
            and (candidate.severity or 0) >= 3
            and cat.get("package_role") == "package_driver"):
            return True
    return False
```

This removes the per-catalog-id maintenance and naturally includes new sev-3 drivers. **Note**: `_has_strong_signal` currently doesn't take a `catalog_lookup` argument — you'll need to thread one in from `compute_package_strength` (line 556) and update all callers. Check whether `EstimateCandidate` already carries `kind` / `package_role` denormalized; if so, just read from there without a lookup.

`_STRONG_SIGNAL_CATALOG_IDS` would still be useful as an explicit override for *exceptional* sentinels that don't meet the generic rule (e.g., sev-2 items that should still short-circuit because of visible-substrate severity). The 4 currently-listed items meet the generic rule already — they could move to the rule-based path entirely, but keeping the frozenset for documentation/override purposes is fine.

### Issue 5 — Latent namespace bug in `absorption_scope` fallback

**Setup** (lines 117-121): `package_type` strings ("bathroom_repair", "bathroom_modernization", "bathroom_turnover") and pricing-tier strings ("bathroom_repair_light", "bathroom_repair_heavy", "bathroom_refresh", etc.) are intentionally separate namespaces. `_PACKAGE_ABSORPTION_SCOPES` is keyed by **pricing-tier strings**. Both production builders set `absorption_scope` at construction using the pricing-tier key, so it works in the happy path.

**The bug**: in `reconcile_packages_and_estimate_units` phase B (search for `setdefault.*absorption_scope`):

```python
pkg.setdefault(
    "absorption_scope",
    _package_absorption_scope(str(pkg.get("package_type") or "")),  # ← wrong namespace
)
```

The `setdefault` only fires if `absorption_scope` was missing. **If it ever does fire**, it looks up using the package_type ("bathroom_repair") against a dict keyed by pricing tiers ("bathroom_repair_light") — silent miss, returns `{"family": "", "groups": [], "trade_buckets": [], "components": []}`, **breaks all absorption for that package**.

**Fix options**:
- **A (recommended)**: Remove the fallback. If a package reaches phase B without `absorption_scope`, let it `KeyError` loudly so we catch the upstream omission immediately rather than silently no-op'ing absorption.
- **B**: Derive the pricing tier from the package before lookup. The package object already carries `pricing_tier` (or equivalent) — use that key instead of `package_type`. Verify by inspecting a built package.
- **C**: Wrap `_package_absorption_scope` to accept either namespace and return the right value. Most robust but most code.

If you go with A, add a unit test that constructs a package without explicit `absorption_scope` to confirm the loud-failure behavior.

---

## Smaller observations (3, with varying relevance)

### Smaller obs 1 — `aggregate_whole_home_turnover` default fallback

Line 1413 requires ≥2 distinct rooms (reasonable). Strongest-package selection at lines 1418-1422 uses `max(_STRENGTH_RANK.get(...))` with `default=-1`. If all turnover packages somehow have no strength set, you'll silently default to `PACKAGE_STRENGTH_MODERATE`. Fine fallback but worth a debug log when it happens so we notice the upstream omission.

### Smaller obs 2 — `_infer_room_refresh` OR-logic

Line 2057: `(c.severity or 0) <= 2 or cat.get("category") == "cosmetic"`. The `or` is doing a lot of work — a sev-4 cosmetic-tagged item would qualify for room-refresh evidence. Probably intentional (cosmetic kind is the gate, severity is a fallback for non-cosmetic items) but reads like a bug. **Moot if `_infer_room_refresh` gets deleted in Issue 1's Option A.** Otherwise worth a comment explaining the intent.

### Smaller obs 3 — Schema divergence

Newer builder (the one used by `infer_package_candidates`) sets `audit_only: True` and `estimate_eligible: False` until Pass 2f verification runs (lines 1141-1142). The older `_build_package` (used by `infer_packages`) doesn't set these at all. **Direct consequence of Issue 1**: if anything downstream checks `audit_only` / `estimate_eligible`, legacy-built packages will look "real" without ever going through verification — bypass risk. **Moot if Option A.** Otherwise needs reconciliation.

---

## Recent work to be aware of (current repo state)

Catalog overhaul that just landed in this branch (verify with `git log --oneline -20`):

**Added 15 new bathroom items** to `tools/issue_catalog.json`. Bathroom count went from 6 to 21. Full list and their fields are documented in `docs/HANDOFF_bathroom_catalog_overhaul.md`. Highlights:
- 8 new `package_driver` defects under `package_type: bathroom_repair`
- 5 new `package_support` defects
- 2 new `package_support` upgrades under `package_type: bathroom_modernization`
- Severity inversions vs cross-room generics were corrected (bath-specific items now meet or exceed the corresponding generic's severity)
- `require_any` / `deny_any` guardrails added to 7 items to prevent cross-room misfires

**Extended `_STRONG_SIGNAL_CATALOG_IDS`** ([tools/rehab_packages.py:539-545](tools/rehab_packages.py#L539)) with three bathroom sentinel IDs (`missing_bathroom_tile_exposed_substrate`, `active_water_damage_bathroom`, `missing_vanity_exposed_plumbing`).

**Tests**: all 204 pass. Component coverage verified — every bathroom resolver component class (`vanity`, `tile`, `tub_shower`, `fixture`, `flooring`, `plumbing`, `electrical`, `moisture`, `paint`, `bath_finish`) has ≥1 bathroom-tagged catalog item.

**Decision recorded**: bathroom-specific items co-fire with cross-room generics (no `drop_if_generic: true` applied to existing generics), because `drop_if_generic` is a global kill switch ([scene_classifier_passes.py:1497-1513](tools/scene_classifier_passes.py#L1497)) and would regress non-bathroom flows.

---

## Proposed sequence (subject to your planning)

| Step | Work | Risk | Test cost |
|---|---|---|---|
| 1 | **Verify the diagnosis** (Issue 1) by spot-checking 3-5 `infer_packages` tests for any production-relevant coverage. | None | Read-only |
| 2 | **If diagnosis holds: migrate ~20 legacy tests** to `infer_package_candidates`. Some scenarios may not map 1:1 — the legacy path lacks strength/confidence, so assertions need adjustment. | High in test churn, low in product risk | ~20 tests to port |
| 3 | **Delete `infer_packages`, `_infer_kitchen_package`, `_infer_bathroom_package`, `_infer_room_refresh`** + update module docstring + clean `.claude/settings.local.json` allowlist. | Low — purely code removal, validated by step 2 | None — tests migrated |
| 4 | **Fix Issue 4 (sentinel rule)** — replace hardcoded `outdated_kitchen_finishes` / `outdated_bathroom_finishes` checks with the generic `kind==defect AND severity>=3 AND package_role==package_driver` rule. | Low | Targeted test updates |
| 5 | **Fix Issue 5 (absorption_scope namespace)** — remove silent-no-op fallback. | Low — fixes latent trap | One new unit test |
| 6 | **Fix Issue 2 (tier-undercut floor)** — apply `max(tier_cost, absorbed_total)` floor in each `_resolve_*_profile`. | Medium — recomputes values; tests likely need recalibration | Test fixture recalibration |
| 7 | **Fix Issue 3 (electrical heavy/light split)** — split `classify_component`'s `electrical` return into heavy/light; only heavy triggers `repair_heavy`. | Medium — same recalibration | Same test fixtures as 6 |

Step 1 is a research gate. Steps 4-7 are independent and can be reordered; I'd hold them until 2+3 are done so we're not updating tests we're about to delete.

---

## Questions to resolve before you write the plan

1. **Confirm or refute the diagnosis** by completing the verification in Issue 1 above. If you find anything in production that depends on `infer_packages`, escalate immediately — that changes the whole plan.
2. **Delete-vs-shim decision** — Option A (delete) or Option B (backward-compat shim)? My recommendation is A, but the user should sign off given the test-migration cost.
3. **Issue 2 fix preference** — floor approach (`max(tier, absorbed)`) or smarter dispatch (escalate tier)?
4. **Issue 3 fix preference** — split `electrical` bucket, weight by severity, or downward cap?
5. **Issue 4 rule placement** — keep the explicit `_STRONG_SIGNAL_CATALOG_IDS` frozenset as override for exceptional sentinels, or fold everything into the generic rule and drop the frozenset?

---

## Critical files and their roles

- `tools/rehab_packages.py` — the module under review. ~2200 lines. Both paths live here.
- `tools/renovation_estimate_v4.py` — the production caller of `infer_package_candidates`.
- `tools/renovation_estimate.py` — the v3 estimate. Does **not** call package inference. Used by v4 for the quick estimate.
- `tools/rerun_pass_2f_artifact.py` — secondary caller of `infer_package_candidates` for artifact reprocessing.
- `tools/artifact_writers.py:809-819` — the entry point that invokes both v3 and v4 estimates.
- `tools/issue_catalog.json` — the catalog. Recently overhauled for bathroom parity (see `HANDOFF_bathroom_catalog_overhaul.md`).
- `tests/test_rehab_packages.py` — the test file with the ~20 legacy `infer_packages` calls and 7 `infer_package_candidates` calls.
- `.claude/settings.local.json:93` — has a python -c import sanity check that references `infer_packages`. Will need updating.

---

## Verification approach (after any fix)

1. **JSON lint** — `py -c "import json; json.load(open('tools/issue_catalog.json'))"`.
2. **Tests** — `py -m pytest tests/test_rehab_packages.py tests/test_renovation_estimate_v4.py tests/test_renovation_estimate.py tests/test_package_taxonomy.py --tb=short`. All 204 (and growing) must remain green.
3. **Import smoke** — `py -c "from tools.rehab_packages import infer_package_candidates, reconcile_packages_and_estimate_units, classify_component, _has_strong_signal; print('imports OK')"`. If you delete `infer_packages`, update the `.claude/settings.local.json:93` smoke command too.
4. **Live audit run** — pick a bathroom-heavy property in `C:\Users\Steven\IntelliJProjects\realtorvision\artifacts` and run `py -m tools.audit_runner --artifacts-root <path> --property <id>`. Inspect `photo_intel.json` for:
   - `renovation_estimate_v4.package_candidates` entries (sanity)
   - `pass_2f_trace.attempted_count > 0`
   - No `package_total_below_absorbed_total_low/high` warnings after Issue 2 fix
5. **No-regression check** — run the same audit on a kitchen-heavy property to confirm kitchen packages still emit at the expected tiers.

---

## Out of scope (explicitly)

- Frontend changes — separate handoff.
- Pass 2f prompt edits — already shipped.
- Adding new bathroom or kitchen catalog items — the catalog overhaul is complete for this round.
- Changing `_PACKAGE_ABSORPTION_SCOPES` definitions — those drive the absorption mechanics; Issue 5's fix is on the consumer side (the `setdefault` fallback), not the definitions.
- New pricing tier constants — the `BATHROOM_*` and `KITCHEN_*` tier constants are the canonical set.
- Pass 2c / 2d / 2e routing prompts — `drop_if_generic` semantics live there but aren't being changed in this work.

---

## How to start

1. **Enter plan mode** when you begin.
2. Run the verification greps for Issue 1 in your first message.
3. Spot-check the legacy tests for hidden production coverage.
4. Write the plan file with your recommendations on all 5 issues + the 3 smaller observations.
5. Ask the user the open questions (delete-vs-shim, fix preferences) via `AskUserQuestion` before finalizing.
6. Exit plan mode with the approved plan; implement step-by-step from there.

Good luck.
