# Handoff: exterior estimate coverage (unverified line-item dollars)

## Status

Branch `benchmark_system`. **Done, 2026-08-03.** The estimate guard shipped opt-in, ten exterior
items were populated behind it, and an observe-only corpus audit script landed with four
snapshots. Exterior coverage went from 5/25 to 15/25 with **zero movement in any headline total**.

The original hazard — "adding an `estimate` block prices immediately and unverified" — is resolved
for the new items: they price into an additive `withheld_estimate` lane and stay out of every
headline until something confirms them.

## What shipped

### 1. The estimate verification guard (`tools/renovation_estimate.py`)

`requires_2f_for_estimate` finally has a consumer. `classify_estimate_verification` sorts every
candidate into `not_required` / `confirmed` / `confirmed_by_rule` / `unconfirmed` / `invalidated`;
`unconfirmed` candidates are removed from the working set inside `compute_renovation_estimate`,
right after `resolve_estimate_units` and before the empty-candidate short-circuit.

Because they leave the candidate list entirely, they are absent from `groups`, the tier summary,
reconciliation, package adjustment, `project_scope_breakdown`, the evidence projection and
`ui_priorities_v1` **by construction** — not by filtering at each site.

New v4 payload:

- `withheld_estimate.policy_version` = `"requires_2f_guard_v1"`
- `withheld_estimate.line_items[]` — full cost/audit fields plus `estimate_verification_status`,
  `estimate_eligible: false`, `withheld_reason: "requires_2f_confirmation"`
- `withheld_estimate.total` / `.risk_exposure_total`
- `meta.priced_candidate_count` / `meta.withheld_candidate_count`
- `provenance.estimate_verification_policy_version`, plus `estimate_verification_guard` in
  `v4_phases_applied`
- `estimate_verification_status` on **every** line item, priced or withheld

`totals.unreviewed_risk_total` is now the withheld total — it flipped from a double-counted subset
of `probable_total` to an additive, disjoint lane. Same `{low, high}` shape, so naive readers are
unaffected. It is aliased to the same dict as `withheld_estimate.total`; `scale_estimate_dollars`
dedupes by identity, so it scales exactly once.

`withheld_estimate.total` is a **plain uncapped sum**, deliberately. Group caps and `max_only` are
a blending artifact of a *visible* group; applying them to a partial set would silently drop
items. The plain sum is the conservative upper bound on what confirmation could add.

Implementation notes worth keeping:

- The partition **must** stay after `resolve_estimate_units`. `per_property` / `per_area` clusters
  collapse N candidates of one catalog item into one priced unit, so partitioning earlier could
  split a cluster whose members disagree on status and change the survivor's unit count.
- `_cluster_key` is keyed on `catalog_item_id`
  ([estimate_units.py:549](../tools/estimate_units.py#L549)), so unit-resolution clusters never mix
  catalog items. That is *why* removing withheld candidates cannot perturb priced ones, and why
  adding new catalog items cannot perturb existing ones.
- `renovation_estimate` mirrors `"confirmed_by_rule"` as a module-local literal because
  `rehab_packages` imports `EstimateCandidate` from it — importing the constant back is circular.
  A test asserts the two agree.
- `_build_line_item` is the single pricing path shared by priced groups and the withheld lane, so
  a withheld range is exactly the range it would have carried had it been confirmed.

### 2. The guard is **opt-in**, and that was a measured decision

`resolve_estimate_meta` defaults `requires_2f_for_estimate` to `False`
([renovation_estimate.py:95](../tools/renovation_estimate.py#L95)). The tier-derived default
(`tier in ("high","medium")`) was implemented, measured against all 1193 auditable artifacts, and
**rejected**:

| Metric | Default False | Default True | Change |
|---|---|---|---|
| `final_rehab` high | $35,609,864 | $22,491,122 | **−37%** |
| `final_rehab` low | $9,614,509 | $8,154,261 | −15% |
| `evidence_headline` high | — | — | −$11.4M |
| `latent_risk_exposure` high | — | — | −$6.75M |
| Artifacts affected | 0 | 919 / 1193 | **77%** |

The reason it is not a correctness win: **Pass 2f reviews packages, not items.** A standalone item
with no `package_affinity` can never be confirmed, so "withhold until confirmed" resolves to
"withhold forever". Top withheld drivers were ordinary, well-evidenced items:

| Catalog item | Occurrences | Withheld high |
|---|---|---|
| `damaged_or_aged_roof_shingles` | 485 | $4,221,595 |
| `older_flooring_style` | 299 | $2,769,537 |
| `boarded_up_entry_or_window` | 106 | $1,264,129 |
| `scratched_or_damaged_flooring` | 197 | $1,176,614 |
| `bare_or_missing_finish_flooring` | 164 | $1,005,500 |

**Guard items one at a time, as a verifier that can satisfy them lands.** Never flip the default
back without re-running the audit.

### 3. Catalog population

Ten items, all with an **explicit** `requires_2f_for_estimate` (never rely on the default):

| Catalog item | Tier | Strategy | Group | Stacking | Unit | Guard |
|---|---|---|---|---|---|---|
| `standing_water_or_poor_grading` | high | repair_or_replace | landscaping | group_cap | per_area | ✅ |
| `driveway_or_walkway_cracking` | medium | repair_or_replace | landscaping | group_cap | per_area | ✅ |
| `clogged_or_damaged_gutters` | medium | service_only | **exterior** | group_cap | per_property | ✅ |
| `trees_or_vegetation_too_close` | medium | service_only | landscaping | group_cap | per_property | ✅ |
| `shed_exterior_paint_failure` | medium | repair_only | exterior | group_cap | per_property | ✅ |
| `metal_carport_rust_corrosion` | medium | repair_only | exterior | group_cap | per_property | ✅ |
| `uneven_gravel_drive_or_parking` | medium | repair_only | landscaping | group_cap | per_area | ✅ |
| `tree_stump_present_in_yard` | medium | service_only | landscaping | group_cap | per_property | ✅ |
| `exterior_door_paint_failure` | medium | repair_only | windows_doors | group_cap | per_opening | ✅ |
| `roofline_water_damage_suspected` | high | inspect_only | roof | max_only | per_property | ❌ |

Two routing decisions that are not obvious:

- **Gutters live in `exterior`, not `roof`.** `_resolve_dominant_stack`
  ([renovation_estimate.py:752](../tools/renovation_estimate.py#L752)) promotes an entire group to
  `max_only` if any member has it, and `damaged_or_aged_roof_shingles` is `max_only`. Gutters in
  `roof` would price **$0** on any property with shingle damage. `project_scope_breakdown` keys off
  `trade_bucket` (`roof_gutters`), not the estimate group, so the trade lane stays correct.
- **`roofline_water_damage_suspected` is deliberately unguarded.** Inspect-only items price at the
  flat `INSPECT_ALLOWANCE` ($200–800), not a repair claim. Guarding it would suppress a latent-risk
  signal precisely *because* it is unverified — inverting its purpose. Corpus-wide, `inspect_only`
  items carry only $166k of cost but **$6.75M of risk exposure**.

`fence_damaged_or_weathered` got `{"mode": "heuristic"}` and nothing else — inert today, but it can
now take a tier without tripping [catalog_validation.py:261](../tools/catalog_validation.py#L261).

**Deliberate omissions (10).** Five `tier: "optional"` items (`landscape_improvement_needed`,
`curb_appeal_upgrade`, `yard_debris_overgrown_leaves`, `gutter_maintenance_needed`,
`concrete_driveway_surface_wear`) are suppressed upstream by Pass 2e's `tier_optional_suppressed`,
so an estimate block would never fire. Five package-affinity items
(`damaged_or_unsafe_deck_or_porch`, `exterior_siding_discoloration_fading`,
`deck_surface_weathering`, `brick_weathering_or_mortar_deterioration`,
`patio_or_porch_surface_wear`) already price through `exterior_repair`; a line item would
double-count. The two sets are disjoint, and `tests/test_exterior_estimate_coverage.py` locks all
of this in.

### 4. Audit script

`scripts/audit_exterior_estimate_coverage.py` — observe-only, never writes an artifact, never
calls the VLM. It reuses the real resolver (`resolve_catalog_estimate_meta`, promoted from private)
and the real engine, replaying each artifact's stored verdicts via `collect_stored_verifications`.

```bash
.venv/Scripts/python.exe scripts/audit_exterior_estimate_coverage.py --artifacts-root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts --baseline artifacts/exterior_estimate_coverage_before.json --json artifacts/exterior_estimate_coverage_after.json --report docs/exterior_estimate_coverage_audit.md --fail-on-headline-delta
```

`--artifacts-root` is required on purpose: `ARTIFACTS_ROOT` in `.env` still points at
`C:/Users/Steven/IntelliJProjects/realtorvision/artifacts`, **a path that no longer exists** (the
repo was renamed to `renointel-prod`). Defaulting to it would silently audit nothing. That stale
`.env` value is a separate pre-existing bug worth fixing on its own.

`--fail-on-headline-delta` gates on `probable_total`, `visible_rehab`, `package_adjusted_rehab`,
`final_rehab`, the three scope tiers, and `evidence_headline`. It deliberately **excludes**
`unreviewed_risk_total`, `withheld_estimate.*`, `latent_risk_exposure` and
`inspection_allowance_total` — those are expected to move.

An artifact is skipped only when the stored copy has **active** packages whose verdicts cannot be
replayed. Candidates that are all `not_run` with no active package lose nothing on recompute, so
they stay in the corpus (that distinction recovered 418 artifacts).

## Results

1264 artifacts scanned, 1193 audited, 71 skipped (43 no usable issue lane, 28 no v4).

| Snapshot | After | Exterior missing | `final_rehab` | Withheld |
|---|---|---|---|---|
| A `..._pre_guard.json` | audit script only | 8,288 | $9,614,509–$35,609,864 | — |
| B `..._before.json` | guard (opt-in) | 8,288 | $9,614,509–$35,609,864 | $0 |
| C `..._after.json` | +9 guarded items | 4,919 | $9,614,509–$35,609,864 | $551,182–$9,408,007 |
| D `..._after_roofline.json` | +roofline | 4,740 | $9,614,509–$35,609,864 | $551,182–$9,408,007 |

- **A→B and B→C: `headline_changes=0`, exit 0.** The guard is provably inert, and the nine guarded
  items are dollar-neutral on every headline.
- **C→D:** 137 artifacts moved, `final_rehab` still unchanged. Roofline is inspect-only, so it
  lands in `inspection_allowance_total` (+$92,865 high) and `latent_risk_exposure` (+$452,732 high)
  rather than visible rehab. This is the one intentional dollar change, isolated on purpose.
- Exterior missing-estimate occurrences fell **8,288 → 4,740 (−43%)**; 906 artifacts now carry a
  withheld lane.

Reference property `redfin_126975740` (131 Alex Ln): exterior missing **18 → 14** exactly as
predicted, `final_rehab` $43,060–$158,017 unchanged across all four snapshots, driveway range
($400–$8,000) present in the withheld lane only, and all nine packages — including
`exterior_repair__exterior_primary` — byte-identical on replay.

## Verification

```bash
.venv/Scripts/python.exe -m tools.catalog_validation
```

```bash
.venv/Scripts/python.exe -m pytest --basetemp=C:/Users/Steven/AppData/Local/Temp/claude/pytest -q
```

**1394 passed, 1 skipped** (5 consecutive clean runs). Catalog validation: 0 errors, 34 warnings —
unchanged from baseline, because no `display_class` was added. Do **not** add `display_class` to
any of these ids: all ten plus `fence_damaged_or_weathered` are on the hardcoded warnings snapshot
at [test_catalog_validation.py:68](../tests/test_catalog_validation.py#L68).

New test modules: `tests/test_estimate_verification_guard.py` (28),
`tests/test_exterior_estimate_coverage.py` (44), `tests/test_audit_exterior_estimate_coverage.py`
(22). The shared fixtures needed **no** opt-out churn — an opt-in guard leaves them alone by
construction.

Known unrelated issues seen while verifying: a stale duplicate `tools/test_rehab_evidence_projection.py`
collides with the `tests/` copy under a bare root collection in a clean checkout, and 24
benchmark/model-comparison tests fail at `HEAD` (they pass in the working tree, which has newer
config). Neither is touched by this work.

## NEXT: the evidence-sufficiency gate (`min_photo_evidence`)

This is the second half of the same problem, not a separate project. It is the highest-value
remaining work in this area and should be done before anything else here.

### Why it exists

`requires_2f_for_estimate` conflates three independent gates and only answers the third:

1. **Evidence sufficiency** — how many photos support this? Computable *today*, deterministic,
   zero model cost. `distinct_photo_count` / `distinct_scene_group_count` already exist on the
   candidate and are already stamped as `supporting_photo_count` /
   `supporting_scene_group_count` on every line item.
2. **Visual confirmation** — did a VLM look at *this item* and agree? **Does not exist.** Pass 2f
   reviews packages. This is the per-item 2f revival.
3. **Package absorption** — is a confirmed bundle covering it? Exists, works.

Gate 2 is why the 2f guard had to ship opt-in. Gate 1 is satisfiable right now and targets the
same hallucination-shaped exposure, so it is the better next move.

**Corpus-wide, 46.7% of all estimate `cost_high` ($17,412,699 of $37,282,934) rests on a single
supporting photo.** That is the size of the prize.

### Candidate set — this is NOT a shingles fix

Two distinct populations show up in the data, and conflating them would be a mistake:

**(a) Items where more evidence is available but absent.** Exterior/site findings on properties
that carry many exterior photos. A second photo is a real signal here, and its absence is
meaningful. These are the gate's targets:

| Catalog item | n | $ at ≤1 photo | $ total | Share | 1-scene |
|---|---|---|---|---|---|
| `damaged_or_aged_roof_shingles` | 485 | $2,261,490 | $4,221,595 | 53.6% | 95.1% |
| `driveway_or_walkway_cracking` | 733 | $2,183,358 | $5,178,285 | 42.2% | 97.0% |
| `standing_water_or_poor_grading` | 287 | $1,615,915 | $2,038,489 | 79.3% | 99.7% |
| `trees_or_vegetation_too_close` | 379 | $1,033,538 | $1,629,442 | 63.4% | 99.2% |
| `clogged_or_damaged_gutters` | 176 | $318,646 | $377,504 | 84.4% | 100% |
| `major_foundation_or_settlement_signs` | 136 | $81,064 | $96,384 | 84.1% | 91.9% |
| `roofline_water_damage_suspected` | 137 | $78,127 | $92,863 | 84.1% | 99.3% |
| `uneven_gravel_drive_or_parking` | 56 | $52,886 | $61,463 | 86.0% | 100% |
| `metal_carport_rust_corrosion` | 40 | $49,445 | $51,983 | 95.1% | 100% |
| `retaining_wall_failure_or_missing_section` | 69 | $44,326 | $49,033 | 90.4% | 100% |
| `shed_exterior_paint_failure` | 55 | $14,980 | $18,763 | 79.8% | 100% |

Roof shingles and driveway cracking alone account for **$4.44M** of single-photo exposure. Both
are known hallucination surfaces: ground-level roof assessment and "is that a crack or a shadow"
on paving.

**(b) Items that are structurally single-view.** A bathroom defect lives in one bathroom and gets
one photo — `bathroom_plumbing_visible_issue` is 100% single-photo, `tub_surround_or_shower_pan_damage`
95.4%, `mold_or_mildew_visible_bathroom` 93.9%. A photo-count gate here would withhold correct
findings for a reason that has nothing to do with confidence. **Do not gate these.**

That split is exactly why this must be a per-item catalog opt-in and never a global threshold.

**Also note: `min_distinct_scene_groups` is not worth building.** The 1-scene column is ~100% for
almost every item, so it would fire on nearly everything and discriminate nothing. Photo count is
the lever.

### Work

1. Add `min_photo_evidence` to the catalog `estimate` block — `CatalogEstimateMeta` and
   `resolve_estimate_meta` in [tools/renovation_estimate.py](../tools/renovation_estimate.py).
   Default `0`/absent so it is opt-in per item, exactly like the 2f guard.
2. Extend `classify_estimate_verification` with an `insufficient_evidence` state, routing those
   candidates into the **same existing `withheld_estimate` lane** with
   `withheld_reason: "insufficient_photo_evidence"`. The lane deliberately does not care why
   something was withheld — that is the whole point of its shape. Decide precedence against
   `invalidated` and `not_required` explicitly and test it.
3. Validate the new field in [tools/catalog_validation.py](../tools/catalog_validation.py).
4. Set `min_photo_evidence: 2` on the population-(a) items above. Start with the four largest
   (`damaged_or_aged_roof_shingles`, `driveway_or_walkway_cracking`,
   `standing_water_or_poor_grading`, `trees_or_vegetation_too_close`) so the delta is legible,
   then extend.
5. Measure with the existing audit script. **This will move headline totals** — unlike the
   exterior population, which was dollar-neutral by design — so do **not** pass
   `--fail-on-headline-delta`; report the delta for review instead:

```bash
.venv/Scripts/python.exe scripts/audit_exterior_estimate_coverage.py --artifacts-root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts --baseline artifacts/exterior_estimate_coverage_after_roofline.json --json artifacts/evidence_gate_after.json --report docs/exterior_estimate_coverage_audit.md
```

6. Tests alongside [tests/test_estimate_verification_guard.py](../tests/test_estimate_verification_guard.py).

Budget ±120. Risk: medium — it moves headline dollars on purpose, but every item is opt-in and the
audit quantifies it before anyone sees it.

### After that

Only once a per-item verifier exists (gate 2) should individual items get
`requires_2f_for_estimate: true`, one at a time, each measured with the audit script. Shingles is
the obvious first candidate: Steven's stated bar is ≥2 photos **and** 2f approval before shingle
damage is costed at all.

## What already landed earlier (do not redo)

`exterior_repair` package family: constants, pricing tiers (`exterior_repair_light` 1k–5k,
`exterior_repair_heavy` 5k–18k), absorption scopes, an exterior Pass 2f prompt, a property-level
`exterior_primary` room surrogate, coverage-aware review-photo selection, and 8 `package_affinity`
blocks (4 drivers, 4 supports). Roof, gutters, driveway and landscaping were deliberately excluded
from the package — different trades, and ground-level roof assessment is the pipeline's worst
hallucination surface. They are now covered as **guarded** standalone line items instead.

Two errors in the previous version of this doc, now corrected: its 20-row table listed
`damaged_soffit_or_porch_ceiling` (which has **no `category` field**, so it was never one of the
25) and omitted `landscape_improvement_needed` (which is). It also said 6 optional-tier items among
the 25; the real count is 5.
