# Handoff: scene_group normalization — `exterior` is silently rewritten to `other`

## Status

Branch `benchmark_system`, 2026-07-31. Reproduced, unfixed, low urgency. Isolated during the
exterior-package work; deliberately left alone there because the exterior fix routed around it.

## The bug

Every exterior issue's estimate scope key reads `scene_group:other`, e.g. from the real artifact
`redfin_126975740/20260731_101711_1dc6f7ee`:

```
catalog:damaged_or_rotted_siding_or_trim|scene_group:other|room:exterior_back
```

`_estimate_scope_key_for_issue` ([renovation_estimate.py:333](../tools/renovation_estimate.py#L333))
does `scene_group = _meaningful_scope_hint(issue.get("scene_group")) or "other"`, and
`_meaningful_scope_hint` ([:321-325](../tools/renovation_estimate.py#L321)) returns `""` for any
member of `_GENERIC_SCOPE_HINTS` ([:306-310](../tools/renovation_estimate.py#L306)) — which
contains `"exterior"`.

Of the seven canonical scene groups (`kitchen, bathroom, bedroom, living_areas, utility,
exterior, other` — derived from `SCENE_SPECS`,
[pipeline_common.py:44-79](../tools/pipeline_common.py#L44)), **only `exterior` is collateral
damage**; `other` maps to `other` anyway and the remaining five survive the filter. That
asymmetry is the evidence this is an unintended side-effect rather than intent: the denylist was
written for *room/location* hints, where "exterior" is too coarse to identify a room, and then
reused verbatim on `scene_group`, a closed 7-token vocabulary where `exterior` is a legitimate
value.

Corroborating inconsistency: the same candidate simultaneously reports
`scene_groups_seen: ["exterior"]` (stamped from the raw issue values,
[:424-426](../tools/renovation_estimate.py#L424)) and `scene_group:other` in its scope key.

## Binding constraints

1. **Do NOT simply remove `"exterior"` from `_GENERIC_SCOPE_HINTS`.** That helper is shared: it
   also filters `room_surrogate_id`/`room_id`/`location_hint` hints
   ([:336-342](../tools/renovation_estimate.py#L336)) and — critically — the stamped
   `estimate_unit_id` in **both** extraction lanes
   ([renovation_estimate.py:435-439](../tools/renovation_estimate.py#L435),
   [renovation_estimate_v4.py:428-432](../tools/renovation_estimate_v4.py#L428)).
   The exterior package family **depends** on `"exterior"` being filtered there: that is exactly
   why the property-level surrogate is named `exterior_primary` and not `exterior`. Removing the
   token from the shared set would let a bare `"exterior"` unit id through and change bucketing
   behavior in a way no current test covers.
2. **Fix shape:** add a scene-group-specific normalizer over the closed vocabulary
   (`SCENE_GROUPS_UI` / `SCENE_TO_GROUP_UI` keys), used only for the `scene_group` half of the
   scope key. Leave the shared hint helper untouched.
3. **Table-test all seven groups** — the whole point is that the current behavior is
   non-uniform, so a test that only checks `exterior` would not prevent the next instance.
4. **Fixing re-keys estimate units.** The scope key feeds grouping
   ([:413-414](../tools/renovation_estimate.py#L413)), the candidate `unit_id`
   ([:450](../tools/renovation_estimate.py#L450)), the v4 package-only dedupe key
   ([renovation_estimate_v4.py:482](../tools/renovation_estimate_v4.py#L482)), and estimate-unit
   merging ([estimate_units.py:551, 845, 958, 1028](../tools/estimate_units.py#L551)). A
   before/after artifact comparison is mandatory, not optional.
5. Tests: `.venv\Scripts\python.exe -m pytest`. Baseline on this branch: **1235 passed,
   1 skipped**.

## Blast radius, measured

Low but non-zero. The exterior package work already changed some exterior line items' billable
unit ids (with **no** cost change) — see the 131 Alex Ln before/after. Expect this fix to change
scope-key *strings* broadly for exterior issues and possibly merge units that currently sit
apart. Dollars should not move; prove it rather than assume it.

Reference property: `redfin_126975740/20260731_101711_1dc6f7ee`, which has 27 exterior issues
across 8 exterior photos and now carries a confirmed `exterior_repair__exterior_primary` package.

## Work items

| # | Change | Files | Risk | Budget |
|---|---|---|---|---|
| 1 | Scene-group normalizer over the closed vocabulary + use it in `_estimate_scope_key_for_issue` | `renovation_estimate.py` | medium | ±35 |
| 2 | Table test across all seven scene groups | `tests/test_renovation_estimate.py` | none | ±40 |
| 3 | Before/after artifact comparison (scope keys, estimate units, group totals, headline) | `tools/compare_reno_estimates.py` or ad-hoc | none | ±0 |
| 4 | Regression assertion that `_meaningful_scope_hint("exterior") == ""` still holds | `tests/test_rehab_packages.py` (already asserted in `TestExteriorRepairV4Integration`) | none | ±0 |

Item 4 already exists — `test_exterior_estimate_unit_id_survives_scope_hint_filter` pins both
halves of constraint 1. Keep it green; if the fix breaks it, the fix is wrong.

## Verification

```
.venv\Scripts\python.exe -m pytest tests/test_renovation_estimate.py tests/test_renovation_estimate_v4.py tests/test_estimate_units.py tests/test_rehab_packages.py -q
```
Then replay the reference property and diff `groups[].line_items[]` (unit ids, cost_low/high),
`estimate_units`, group totals, and `rehab_evidence_projection_v1.headline`. Dollar movement
means the fix changed more than string identity — stop and investigate.
