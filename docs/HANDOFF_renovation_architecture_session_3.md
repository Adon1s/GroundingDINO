# Handoff — Renovation Architecture Session 3

Date: 2026-08-14. Branch: `renovation_architecture_rework` (Sessions 1–2 are
committed as `f808e48` + `e00698d`; all Session 3 work is uncommitted by
design — nothing staged or pushed). Scope executed: Session 3 of
`docs/IMPLEMENTATION_PLAN_renovation_scope_estimate_architecture.md` —
deterministic work-item derivation from the validated Session 2 condition
review, max-envelope deduplication with suppressed audit records, and a
package-independent standalone estimate with exact totals by estimate scope.
Output remains private shadow data under
`analysis_debug.renovation_architecture_shadow_v1`; v4 output, frontend
contracts, packages, Sol, and reconciliation are unchanged; `current` mode
stays a no-op. The catalog file is byte-identical: SHA-256
`5BE11ABB6DAD9921C6F1A6851C972DA2DCE57F95CCB1FEB8577742A59349E1EC` verified
before and after.

## Files added

- `tools/renovation_architecture/work_items.py` —
  `derive_standalone_estimate(review_result, projection, property_metadata,
  estimate_id)`: validates the frozen Session 2 result first
  (`ReviewResultInvalid` dependency failure otherwise), consumes ONLY
  `accepted_for_work` dispositions, prices through the legacy costing core
  (`tools/costing.compute_item_cost_range` verbatim — heuristic fallback,
  `base + (N-1)*per_occurrence`, caps, kind x scope x trade multipliers once,
  manual allowances exempt), applies
  `tools/cost_factors.resolve_property_cost_factor` exactly once per source
  aggregate BEFORE any split, splits room-like aggregates with a local
  6-line `_split_integer` copy (divmod, remainder $1 at a time to the first
  k — byte-compatible with `rehab_packages._split_integer`, copied so the
  new package never imports the legacy monolith), dedups on
  `(action_code, trade_bucket, unit_policy, billable_unit_id)` with
  max-envelope merging, and self-validates the superset result
  (`StandaloneResultInvalid` parse failure otherwise). The input is never
  mutated.
- `tests/test_renovation_architecture_work_items.py` (39 tests) — unit-policy
  semantics, pricing pins, factor/split interaction, dedup, lanes/totals,
  failures, and one mutation test per new validator invariant.
- `docs/HANDOFF_renovation_architecture_session_3.md` (this file).

## Files changed

- `tools/renovation_architecture/contracts.py` — schema v3 (contracts AND
  envelope), `PROJECTION_VERSION = "renovation_catalog_projection_v2"`, new
  state `standalone_estimate_complete`, three new policy versions
  (`work_derivation_v1`, `work_dedup_max_envelope_v1`,
  `standalone_pricing_v1`) added to `POLICY_VERSIONS` (7 keys total),
  `DEDUP_SUPPRESSION_REASON = "dedup_collision"`, `ESTIMATE_SCOPES` +
  `ESTIMATE_SCOPE_MERGE_PRIORITY`, `ObservedCondition.opening_instance_hints`
  (tuple of "field:value"), **WorkItem v3** (merged lineage: `condition_ids`,
  `catalog_item_ids`, `source_estimate_unit_ids`, `billable_unit_id`,
  `action_sources`, `pricing_modes`, `identity_ambiguous`, `estimate_scope`,
  `estimate_scope_reason` replace the v2 scalars), new `WorkDedupCollision`
  and `StandaloneEstimate` records.
- `tools/renovation_architecture/ids.py` — `make_work_item_id` kwarg
  `estimate_unit_id` → `billable_unit_id` (recipe unchanged);
  `make_merged_work_item_id` (`wk1`, distinct `"work_merged"` namespace so
  source/merged collisions are structurally impossible);
  `make_work_dedup_collision_id` (`wdc1`).
- `tools/renovation_architecture/catalog_projection.py` — every
  `work_policy` entry gains `estimate_scope` + `estimate_scope_reason`:
  inspection routes are `("inspection_risk", "terminal_route_inspection")`
  (a routing fact); work routes classify through the existing
  `tools/estimate_scope.classify_estimate_scope_with_reason({}, raw_item)`
  (catalog-override-first; the empty candidate reduces it to its
  catalog-only baseline). Classifier `ValueError`s are collected and raised
  as `RenovationCatalogError` (a second raise point after the projection
  loop). Shipped distribution: 35 required_rehab / 61 marketability_rehab /
  7 optional_value_add / 5 inspection_risk over the 108 work+inspection
  items (pinned in tests).
- `tools/renovation_architecture/conditions.py` — captures tier-1 opening
  instance hints per issue in the grouping loop (first matching field of
  `_OPENING_INSTANCE_FIELDS` passing `_meaningful_unit_hint`, as
  `"field:value"`, imported from `tools.estimate_units` — byte-compatible
  with legacy `_explicit_opening_hints`), union-sorted per condition.
- `tools/renovation_architecture/validators.py` — `wdc1` id pattern; hints
  on `_CONDITION_FIELDS` (sorted-unique, may be empty); v3
  `_validate_work_item` (sorted-unique lineage tuples, vocabulary subsets,
  suppressed ⇒ reason exactly `dedup_collision`, active ⇒ null); new
  `_validate_work_dedup_collision` and `_validate_standalone_estimate`;
  new **`validate_standalone_estimate_result`** (see gates below);
  `validate_envelope` gains the `standalone_estimate_complete` branch;
  `validate_complete_result` touched ONLY for the v3 WorkItem shape (catalog
  membership + new source-unit membership in the lineage loop).
- `tools/renovation_architecture/runtime.py` — `build_shadow_envelope` calls
  `derive_standalone_estimate` after `run_condition_review` inside the same
  try/except; success state is now `standalone_estimate_complete`; failures
  ride the existing `classify_failure` path unchanged.
- `tools/renovation_architecture/__init__.py` — exports
  `validate_standalone_estimate_result`.
- `tools/artifact_writers.py` — last-resort envelope literal bumped to
  schema 3 (one line).
- Tests updated: `test_renovation_architecture_contracts.py` (builders → v3
  + hints, `_standalone_result`/`_standalone_envelope` helpers, new
  `TestStandaloneEnvelope` incl. the 7-key `POLICY_VERSIONS` pin and the
  `ESTIMATE_SCOPES` == `estimate_scope.VALID_ESTIMATE_SCOPES` pin, new id
  tests), `test_renovation_architecture_catalog.py` (projection v2 literal,
  scope-distribution pins 35/61/7/5, synthetic override/classified/
  inspection scope tests), `test_renovation_architecture_conditions.py`
  (hint capture / generic filtering / duplicate collapse),
  `test_renovation_architecture_runtime.py` (envelope pins →
  `standalone_estimate_complete` + empty work lanes + zero buckets + neutral
  factor; two tests renamed accordingly), `test_renovation_architecture_terra.py`
  (end-to-end envelope → new state + the 270/1350 work-item pin).
  `test_renovation_architecture_disposition.py` /
  `test_renovation_architecture_usage_guard.py` consume shared builders and
  needed no edits; `tests/test_artifact_writers.py` untouched (it has no
  shadow references — the seam tests live in the runtime test file).

## The derivation algorithm (deterministic, exact)

Per catalog item over its accepted conditions:

| unit_policy | source items | billable_unit_id | priced N | unit_count |
|---|---|---|---|---|
| per_property / per_system / per_area | 1 (all conditions) | `"property"` / `"system"` / `"area"` | 1 | 1 |
| per_room / per_kitchen / per_bathroom | 1 per distinct accepted unit | the unit | distinct units | 1 each |
| per_opening | 1 per condition | the condition's unit | max(1, hints in unit) | = N |
| per_scope (default) | 1 per condition | the condition's unit | 1 | 1 |

Order per group: costing scope = strategy-mapped (`repair_only→repair`,
`replace_only→replace`, `service_only→service`; `repair_or_replace`/absent →
catalog scope — the deterministic projection of legacy `POSTURE_TO_SCOPE`;
`inspect_only` is unreachable) → `compute_item_cost_range` → property factor
once (`int(round(x * factor))`, monotone so low ≤ high survives) → room-like
only: sum-preserving integer split paired against sorted unit ids (per-index
low ≤ high holds because the allocation is monotone in the total). Dedup:
singleton groups pass through; collisions suppress every source
(`dedup_collision`, dollars retained as audit) and emit one merged active
with `low = max(source lows)`, `high = max(source highs)` — never summed —
lineage tuples as sorted unions, `unit_count = max`, `identity_ambiguous =
any`, `estimate_scope` = most-required among sources
(required > marketability > optional > inspection_risk), reason from the
lexicographically-first (by work_item_id) winning-scope source, plus one
`WorkDedupCollision`. Totals: per-scope sums over ACTIVE items only;
headline = componentwise bucket sum. Empty accepted set → empty lanes, zero
buckets, factor still recorded.

## Decisions and deviations (owner-approved this session)

- **Room-like N = distinct accepted physical units.** The legacy room-name
  billable matchers (`_is_kitchen_like` etc.) were deliberately NOT ported:
  conditions already arrive one-per-physical-unit and Terra accepted the
  condition in that unit. Deviation vs v4 on fallback-heavy listings
  (scope_room fallback units each count); flagged for the Session 6 delta
  review.
- **per_opening is per-unit.** One work item per accepted condition;
  occurrences penalized inside the unit (`base + (hints-1)*per_occurrence`);
  the legacy listing-wide aggregation and the tier-2 weak-language heuristic
  are dropped (tier-1 explicit hints + conservative fallback only). Three
  single-opening rooms price 3 bases where v4 priced base + 2 increments.
- **Collision lane = most-required.** Money moves between visible buckets
  only, never the headline.
- **The draft's "split the aggregate" claim was corrected**: legacy pricing
  has no split (that lives in reconciliation). Session 3 composes aggregate
  pricing + factor + the reconciliation-style split, in that order, so
  legacy economics are preserved per catalog item while records stay
  per-unit.
- **`PROJECTION_VERSION` bumped to v2** because the projection now embeds a
  classification policy; provenance's `projection_version` +
  `projection_fingerprint` pin it, so no separate scope-policy key was added
  to `POLICY_VERSIONS`.
- **Suppression vocabulary is closed**: `dedup_collision` is the only legal
  suppressed reason this session (validator-pinned). Session 5 extends it if
  reconciliation needs more.
- **`validate_complete_result` (Session 5 gate)** was touched only where the
  v3 WorkItem shape forced it; its invariant structure is otherwise frozen.
  Note it still has no Terra usage keys — reconciling the Session 2 usage
  block into the complete result is Session 5's call.
- **Terra checkpoints invalidate automatically**: the request fingerprint's
  first input is `CONTRACTS_SCHEMA_VERSION` (now 3) and its second is the
  projection fingerprint (changed by the scope metadata), so every pre-v3
  checkpoint misses cleanly and is reviewed fresh. No checkpoint code
  changed; historical artifacts are not rewritten.

## Commands run and results

All on 2026-08-14, all green:

```
.venv\Scripts\python.exe -m pytest tests\test_renovation_architecture_work_items.py tests\test_renovation_architecture_contracts.py tests\test_renovation_architecture_catalog.py tests\test_renovation_architecture_runtime.py tests\test_renovation_architecture_conditions.py tests\test_renovation_architecture_terra.py tests\test_renovation_architecture_disposition.py tests\test_renovation_architecture_usage_guard.py -q --basetemp=C:\tmp\rv_pytest_session3_focused_20260814
  -> 289 passed, 3.81s

.venv\Scripts\python.exe -m pytest tests\test_cost_factors.py tests\test_estimate_units.py tests\test_renovation_estimate.py tests\test_renovation_estimate_v4.py tests\test_rehab_packages.py tests\test_artifact_writers.py tests\test_catalog_validation.py -q --basetemp=C:\tmp\rv_pytest_session3_regression_20260814
  -> 537 passed, 1 skipped, 8.36s

.venv\Scripts\python.exe -m pytest tests -q --basetemp=C:\tmp\rv_pytest_session3_full_20260814
  -> 2128 passed, 6 skipped, 23.63s   (Session 2 baseline: 2067 passed, 6 skipped; +61 new tests)
```

`git diff --check` clean; `git status` shows no catalog modification;
`Get-FileHash tools\issue_catalog_kind_v2.json` reproduces
`5BE11ABB6DAD9921C6F1A6851C972DA2DCE57F95CCB1FEB8577742A59349E1EC`. All
warnings in the runs are pre-existing (utcnow deprecation, google-genai).
Fake providers only; no live Terra call was made at any point.

## Session 3 acceptance gates

- Every accepted condition maps to exactly one ACTIVE work item —
  validator-enforced (`accepted scope cannot vanish` / `multiple active`).
- Non-accepted conditions (excluded, inspection, withheld, no_action,
  unsupported, cannot_assess) create no work and no dollars; their Session 2
  lane records are carried into the Session 3 result verbatim (the review
  subset is re-validated by the frozen Session 2 gate inside the new
  validator).
- No physical work is billed twice: the dedup key is the physical job, the
  merged active is the exact max envelope (never a sum), and active-group
  uniqueness is validator-enforced.
- Every suppressed item belongs to exactly one collision audit; collision
  lineage (condition/catalog/unit/action-source/pricing-mode unions,
  unit_count max, ambiguity any-of, scope priority, reason provenance) is
  recomputed and enforced.
- Standalone low/high totals reconcile exactly without packages: per-scope
  buckets are exact sums over active work, the headline is the exact bucket
  sum, and the split is sum-preserving — no epsilon anywhere.
- The pricing corpus is untouched (catalog byte-identical); pricing reuses
  `compute_item_cost_range` verbatim (pinned against it in tests, plus the
  270/1350 hand pin).
- Deterministic failures produce a private `failed` envelope through the
  existing taxonomy path; completed Terra checkpoints and v4 output survive.
- Current mode adds zero keys and v4 output is structurally unaffected
  (existing writer-seam and estimator tests re-run green).

## Risks / notes for the reviewer

- **Dollar deltas vs v4 are expected and intentional** on: fallback-heavy
  listings (room-like unit counting), multi-room opening items (per-unit
  pricing), and colliding catalog items (max envelope vs whatever v4's
  subsumption did). Every such case is discoverable via
  `work_dedup_collisions` and the lineage tuples; Session 6's canary review
  is the checkpoint.
- **Estimate IDs changed** (projection fingerprint moved with the v2
  metadata) — by design; nothing pinned the old ids.
- **The scope classifier runs at projection build (startup)** — a future
  catalog regeneration that makes an item unclassifiable fails worker
  startup with a collected `RenovationCatalogError`, same posture as every
  other projection precondition.
- **`per_system` remains catalog-unused** (0 items) but is implemented and
  tested — it prices like per_property with billable token `"system"`.
- The derivation's `WorkPolicyMissing` guard is defensive: reachable only if
  a review result citing a non-work-route item is fed in externally (the
  disposition table cannot accept a non-work route as `accepted_for_work`).

## Session 4 prerequisites

- Package candidates are built from ACTIVE work-item IDs only
  (`result["work_items"]` with `status == "active"`); suppressed items and
  `work_dedup_collisions` are audit-only and must never enter candidates.
- The input contract is `validate_standalone_estimate_result` — treat it as
  frozen; Session 4 adds its own state/keys additively (with a schema bump
  if fields change). `validate_complete_result` stays the Session 5 gate.
- `PackageCandidate.child_work_item_ids` already validates against active
  work in `validate_complete_result`; the max-envelope merged actives are
  ordinary work items from the package layer's perspective (their
  `catalog_item_ids` tuple is the only visible difference).
- New Sol policies extend `POLICY_VERSIONS` (validators pin exact key
  equality — the intended friction).
- Deterministic candidate construction needs no checkpoints; Sol calls
  should follow the Session 2 Terra pattern (usage guard, per-unit
  fingerprints, settle-before-parse) if budgeted.
