# Handoff: issue_catalog validator + small data fixes ("low" tier, "paint" group)

## Status / framing

Two recent changes made [tools/issue_catalog.json](../tools/issue_catalog.json) fully load-bearing: package routing now lives in per-item `package_affinity` blocks (commit d887622, "Catalog driven package routing implemented"), and same-unit package subsumption leans on catalog-driven categories and line-item children. The only validation that exists today is `build_package_affinity` ([tools/rehab_packages.py:615](../tools/rehab_packages.py#L615)), which validates affinity blocks and nothing else. Every other field — tiers, groups, costs, scene_groups, enums — is consumed permissively with silent fallbacks.

This handoff has two parts:

- **Part A:** three small, deliberate data fixes for known catalog bugs (two of them change estimate output — intended).
- **Part B:** a hand-rolled validator module + test gate that locks all catalog invariants so this class of drift can't recur.

**Do not add a `jsonschema` dependency** — it is not installed in the venv, and cross-field rules (cost required iff tier high/medium, type↔room consistency) are more natural in plain Python anyway.

## Part A — data fixes

### A1. `estimate_tier: "low"` is an invalid magic value (5 items)

`resolve_estimate_meta` ([tools/renovation_estimate.py:84](../tools/renovation_estimate.py#L84)) accepts only `high | medium | minor`; anything else silently coerces to `minor`, which makes `affects_estimate=False` — the item never becomes a line item. Five bathroom items use `"low"`:

| item | affinity role | fix |
|---|---|---|
| `bathroom_gfci_missing_or_damaged` | bathroom_repair **driver** | → `"medium"` |
| `missing_or_damaged_caulk_at_tub_or_shower` | bathroom_repair support | → `"minor"` |
| `exhaust_fan_missing_or_damaged` | bathroom_repair support | → `"minor"` |
| `dated_bathroom_wallpaper` | bathroom_modernization support | → `"minor"` |
| `bathroom_paint_refresh_recommended` | bathroom_modernization support | → `"minor"` |

Rationale: a missing/damaged GFCI is a safety finding; today it is costed **$0 unless a bathroom package happens to emit**, and after the subsumption pass its cost only ever rides implicitly inside a package range. It already has a manual allowance cost block (150–1500, cap 2500), so promoting to `"medium"` makes it a real priced line item — **an intended estimate-output change**. The other four are genuinely minor cosmetic/maintenance signals whose evidence-only behavior is correct; writing `"minor"` explicitly formalizes it with **zero behavior change** (minor ⇒ `affects_estimate=False`, same as the accidental coercion). Keep their `estimate` blocks (group/strategy/unit_policy) intact.

Also: update the `severity` comment on `EstimateCandidate` ([tools/renovation_estimate.py:137](../tools/renovation_estimate.py#L137)) from `# 1-4` to `# 1-5` — `rotted_subfloor_or_structural_framing` legitimately carries severity 5.

### A2. `estimate.group: "paint"` is not a real group (3 items)

`GROUP_BUDGET_CAPS` ([tools/renovation_estimate.py:588](../tools/renovation_estimate.py#L588)) has a `paint_drywall` entry (800–10,000) that **no catalog item uses**, while 3 items use `estimate.group: "paint"`, which isn't in the caps table and falls to the default cap (200–5,000) in `_recompute_retained_group` ([tools/rehab_packages.py](../tools/rehab_packages.py)) and group-cap stacking.

Fix: rename `"paint"` → `"paint_drywall"` on `peeling_or_damaged_bathroom_paint`, `dated_bathroom_wallpaper`, `bathroom_paint_refresh_recommended`.

Expected behavior change (intended): the paint group's cap rises from the (200, 5,000) default to (800, 10,000), and reconciliation children from these items carry group token `paint_drywall` (which also group-matches the `room_refresh`/`room_repair_heavy` absorption scopes — harmless; their `trade_bucket` is already `paint_drywall`). After A1, two of the three are evidence-only anyway; `peeling_or_damaged_bathroom_paint` (medium) is the one with live line items.

### A3. Cost blocks for affinity drivers that have none (1 required, 1 optional)

`worn_or_stained_vinyl_linoleum` is a bedroom/living **modernization driver** with no `cost` block, so it contributes $0 to tier escalation (`_absorbed_cost_estimate`, [tools/rehab_packages.py:1638](../tools/rehab_packages.py#L1638)). Copy the `cost` block from `worn_or_stained_carpet` as the baseline (equivalent flooring-replacement work) and **flag the numbers for user review in the PR description** — don't silently invent pricing. Optionally do the same for `worn_or_stained_flooring` (support role, lower stakes). The validator (Part B) warns on any remaining driver-without-cost.

## Part B — the validator

New module `tools/catalog_validation.py`:

```python
@dataclass
class CatalogValidationResult:
    errors: List[str]      # each "item_id: message" — must be empty for a valid catalog
    warnings: List[str]    # advisory, not gated

def validate_issue_catalog(issue_catalog: Dict[str, Any]) -> CatalogValidationResult
```

Import vocabularies from their owning modules rather than re-declaring (no import cycle: this module may import from `renovation_estimate`, `rehab_packages`, `estimate_scope`, `catalog_cost_model`; none import back).

### ERROR rules

1. `id`: present, unique across items, `^[a-z0-9_]+$`.
2. `kind` ∈ `{defect, upgrade}`; `scope` ∈ `{repair, replace, cosmetic, service}`; `tier` ∈ `{work, optional}`; `severity` int in 1–5.
3. `trade_bucket` ∈ the catalog's own declared `trade_buckets[].id` list (top of the JSON).
4. `scene_groups`: non-empty list, every token ∈ `set(SCENE_GROUPS_UI) | {"pool"}` (`SCENE_GROUPS_UI` from [tools/pipeline_common.py:13](../tools/pipeline_common.py#L13)). This permanently catches the classic `"living"` vs `"living_areas"` mistake that silently drops items from retrieval. (`"pool"` is not a UI group — it's a scene that maps to `exterior` — but 14 items carry it redundantly alongside `exterior`; accept it as vocabulary, do NOT clean it up in this PR.)
5. `estimate` block, when present: `estimate_tier` ∈ `{high, medium, minor}` (this is the rule that would have caught `"low"`); `strategy` ∈ `{repair_only, replace_only, repair_or_replace, inspect_only, service_only}`; `stack_behavior` ∈ `{sum, group_cap, max_only}`; `unit_policy` ∈ `_VALID_UNIT_POLICIES` ([tools/renovation_estimate.py:60](../tools/renovation_estimate.py#L60)); `group` ∈ `GROUP_BUDGET_CAPS` keys (catches `"paint"`).
6. `cost` block, when present: `mode` ∈ `{allowance, heuristic}`; all amounts numeric and ≥ 0; `base_low ≤ base_high`; `cap_low ≤ cap_high`; `base_high ≤ cap_high` when both present; same for per_occurrence pair.
7. `cost` block REQUIRED when `estimate.estimate_tier` ∈ `{high, medium}` (a priced line item with no cost falls into opaque heuristics).
8. Static `estimate_scope`, when present, ∈ `VALID_ESTIMATE_SCOPES` ([tools/estimate_scope.py:18](../tools/estimate_scope.py#L18)); static `display_class`, when present, ∈ `VALID_DISPLAY_CLASSES` ([tools/rehab_packages.py](../tools/rehab_packages.py)).
9. Authored `cost_model`, when present, ∈ `{line_item, room_allowance}` — `package_allowance`/`inspection_allowance` are derived at runtime ([tools/catalog_cost_model.py:47](../tools/catalog_cost_model.py#L47)) and must not be authored.
10. `package_affinity`: call `build_package_affinity({"items": [item]})` per item (or once for the whole catalog) and convert any raised `ValueError` into an error entry — reuse, don't duplicate, that validation.
11. Flat routing fields are FORBIDDEN on items: `package_type`, `package_category`, `room` must not exist (they were stripped in d887622; this stops re-introduction). `package_role`, when present, ∈ `{standalone, ignore}` only — driver/support live exclusively inside `package_affinity` blocks now.
12. `defaultHidden` / `drop_if_generic` / `deny_any` / `support_any` / `require_any`: type checks only (bool / bool / list / list / list). Do NOT rename `defaultHidden` (its camelCase is ugly but consumers exist; out of scope).

### WARNING rules (advisory)

- An affinity **driver** without a `cost` block (tier escalation sees $0 — A3's class of bug).
- Item missing `display_class` (falls back to fragile keyword sniffing in `catalog_display_class`, [tools/rehab_packages.py:511](../tools/rehab_packages.py#L511)).
- Declared-but-unused trade bucket (currently: `hvac`).

### Wiring

- **The gate:** `tests/test_catalog_validation.py` —
  - `test_shipped_catalog_has_no_errors`: load `tools/issue_catalog.json`, assert `errors == []`. This is the whole point of the handoff; it runs with every pytest invocation forever.
  - `test_shipped_catalog_warnings_snapshot`: assert the warning list equals a recorded snapshot (so new warnings surface in review instead of accumulating silently). Update the snapshot deliberately when warnings change.
  - Per-rule unit tests with minimal synthetic items (one bad item per rule, assert the error fires and names the item id).
- Optional nicety: a `--validate` flag or unconditional validation section at the top of [scripts/audit_issue_catalog.py](../scripts/audit_issue_catalog.py) output.
- Do **NOT** hook validation into runtime catalog loading (analyzer_server) — a hard raise on a bad edit would take down the server; the test gate is the enforcement point.

## Order of work

Write the validator FIRST and run it against the shipped catalog — it should reproduce exactly the Part A findings (5× `estimate_tier "low"`, 3× `group "paint"`, plus the A3 warnings) and nothing else. If it finds anything additional, list it in the PR description rather than fixing en passant. Then apply Part A and assert zero errors.

## Tests

Run with `.venv\Scripts\python.exe -m pytest` (bare `python`/`py` don't resolve the venv). **7 tests fail on this branch pre-existing and unrelated** (test_model_comparison ×2, test_rerun_pass_2f_artifact ×5) — baseline before you start, diff after. `tests/` is untracked on this branch — never `git stash` test paths.

Beyond `tests/test_catalog_validation.py` (above), expect A1/A2 fallout in existing suites and handle it deliberately:

- Any test asserting GFCI (or the bathroom group) totals may shift — GFCI becomes a medium-tier line item with a (150–1500) allowance and `requires_2f_for_estimate=True` (the default for medium; same path every other medium item takes).
- Any test asserting the `paint` group name or its (200, 5,000) default cap.
- Run the full suite and confirm only the 7 baseline failures remain; update assertions that encoded the old buggy behavior, and say so in the PR description.

## Gotchas

- `estimate_tier: "minor"` vs deleting the tier field: keep the explicit `"minor"` so intent is visible; absent tier also coerces to minor but reads as an accident.
- The four `"minor"` items keep routing into bathroom packages via `package_affinity` — Part A must not touch their affinity blocks.
- `severity 5` is legal (one item); the validator range is 1–5, not 1–4.
- `"pool"` in `scene_groups` is redundant vocabulary, not a bug — every pool-tagged item also carries `exterior`. Accept, don't clean.
- Vocabulary imports: `SCENE_GROUPS_UI` (pipeline_common), `GROUP_BUDGET_CAPS` + `_VALID_UNIT_POLICIES` (renovation_estimate), `VALID_ESTIMATE_SCOPES` (estimate_scope), `VALID_DISPLAY_CLASSES` + `build_package_affinity` (rehab_packages), `VALID_COST_MODELS` constants (catalog_cost_model). If `_VALID_UNIT_POLICIES` being private bothers you, re-export it publicly from renovation_estimate rather than copying the set.
- The subsumption pass (`apply_same_unit_package_subsumption`) and affinity routing are FROZEN — this handoff touches catalog data and adds a validator; it does not change routing or reconciliation code.

## Out of scope (separate handoffs planned)

- Region/size cost factors and room-count scaling (next major fix, #3).
- Cost-knowledge consolidation / tier-vs-catalog calibration audit (#4).
- `defaultHidden` rename, retrieval-side scene-group handling, runtime validation hook.
