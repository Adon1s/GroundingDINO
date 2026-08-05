# Handoff: same-unit package subsumption (stop modernization / repair / turnover stacking)

## Status / framing

`infer_package_candidates` ([tools/rehab_packages.py:1836](../tools/rehab_packages.py#L1836)) groups by `(estimate_unit_id, package_type)`, so one room can legitimately emit a modernization AND a repair AND a turnover package simultaneously (e.g. `bathroom_modernization__bathroom_primary` + `bathroom_repair__bathroom_primary` + `bathroom_turnover__bathroom_primary`). Child absorption prevents *line items* from being double-counted (each child is absorbed at most once, [tools/rehab_packages.py:2760](../tools/rehab_packages.py#L2760) Phase B), but the package **tier ranges themselves are independent and all sum** into the rollups:

- repair → `required_rehab`, modernization → `marketability_rehab` (`classify_package_scope`, [tools/estimate_scope.py:274](../tools/estimate_scope.py#L274)), and the headline tiers are cumulative (`build_scope_headline_tiers`, [tools/estimate_scope.py:411](../tools/estimate_scope.py#L411)): `resale_ready = required + marketability`. A bathroom with `bathroom_full_rehab` ($15–35k) and `bathroom_repair_heavy` ($2.5–8k) prices both — but a gut rehab subsumes the repair work; nobody repairs the vanity and then replaces the whole bathroom.
- turnover (paint/flooring/cleaning) → `marketability_rehab`, same bucket as modernization, whose refresh tier already includes paint + flooring. Same room, same scope, both ranges count.

Group caps blunt the extremes but the structural double-count exists below the caps. **This handoff adds a deterministic post-finalization subsumption pass.** It is the highest-impact accuracy fix on heavy-rehab properties.

## The rules (decided — do not redesign)

Operate per unit key: `estimate_unit_id` when non-empty (always true at the point this pass runs — see "Where it slots in"), else `room_surrogate_id`. Only consider ACTIVE packages (`verification_status in ACTIVE_PACKAGE_STATUSES`) with `package_level == "room"`.

Within one unit:

1. **Modernization at `pricing_tier in {"partial_rehab", "full_rehab"}` subsumes the repair package.** The repair package is dropped from the active list (see "Mechanics"). Modernization at `refresh` tier does NOT subsume repair — a cosmetic refresh doesn't fix plumbing; both stay.
2. **Modernization at ANY tier suppresses the turnover package.** Even `*_refresh` covers turnover's whole scope (paint + flooring + finish), and both land in `marketability_rehab`.
3. **Repair alone does NOT suppress turnover.** Their scopes barely overlap (repair tiers carry no `paint_drywall`; only `*_repair_heavy` shares flooring). The residual flooring overlap is an accepted imprecision. *Optional refinement, only if cheap:* when `*_repair_heavy` and `*_turnover_std` coexist in a unit, downgrade the turnover to its `light` tier (flooring already covered). If you implement this, re-derive `absorption_scope` from the new pricing profile via `_package_absorption_scope` ([tools/rehab_packages.py:872](../tools/rehab_packages.py#L872)) and note it in `level_decision_notes`.

### Why "drop", not "merge evidence" (important — read before implementing)

The obvious alternative — transferring the repair package's `supporting_issue_ids` into the modernization package so its children get absorbed there — was considered and **rejected**: supporting-issue absorption ([`_package_child_absorption_reason`, tools/rehab_packages.py:1003](../tools/rehab_packages.py#L1003)) bypasses scope checks, so required-scope children (active water damage, plumbing leaks) would be absorbed into a marketability-scoped package and vanish from the `required_rehab` headline tier. If you skip the modernization, you still must fix the leak — required work must stay required.

With the drop design, the subsumed repair package's children simply lose their `supporting_issue` absorber. In Phase B they are then either:
- **broad-absorbed by the modernization package** when its `absorption_scope` covers them (tile, flooring, paint, bathroom-group items) — pre-existing behavior, reason `same_unit_line_item_scope`; the Phase C floor ([tools/rehab_packages.py:2760](../tools/rehab_packages.py#L2760), Phase C) raises the modernization cost if absorbed totals exceed its tier ceiling; or
- **retained as ordinary line items in their own scope** (plumbing / electrical / moisture children — the modernization scopes exclude those trades), keeping required-scope work in `required_rehab` as raw line items.

Net effect: no work is lost, the repair *bundle premium* (the tier range) stops double-counting, and required vs marketability classification of the underlying work is preserved. The accepted tradeoff: the `required_rehab` tier reverts from "repair package range" to "sum of line items" for that room — that is v3 semantics and is the lesser error. Document this in the audit (`subsumption_audit.tradeoff_note`).

## Mechanics

New public function in [tools/rehab_packages.py](../tools/rehab_packages.py):

```python
def apply_same_unit_package_subsumption(
    packages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (active_packages, subsumption_audit)."""
```

- Build `by_unit: {unit_key: {package_category: pkg}}` from active room-level packages. `package_category` is already on every package dict (`modernization` / `repair` / `turnover`).
- Apply rules 1–2. For each dropped package, **mutate the shared dict in place**:
  - `estimate_eligible = False`, `ui_eligible = False`, `audit_only = True`
  - `subsumed_by_package_id` (rule 1) or `suppressed_by_package_id` (rule 2) = the winning package's `package_id`
  - append `"subsumed_by_same_unit_modernization"` / `"suppressed_by_same_unit_modernization"` to `level_decision_notes`
  - **NEVER touch `verification_status`** — `"confirmed"` must keep meaning "the VLM looked and said yes". Eligibility flags are the gate, same pattern as `_apply_verification_to_package` ([tools/rehab_packages.py:2070](../tools/rehab_packages.py#L2070)).
- On the winning modernization package, append `subsumed_package_ids` / `suppressed_package_ids` lists for the audit trail.
- Return the filtered active list (drops removed, order otherwise preserved — do NOT re-sort; Phase B absorption order is list order) and an audit dict shaped like `bathroom_expansion_audit`: `{applied: bool, subsumptions: [{unit, winner_package_id, loser_package_id, rule}], tradeoff_note: "..."}`.

**Why in-place mutation matters:** `finalize_package_candidates` ([tools/rehab_packages.py:2133](../tools/rehab_packages.py#L2133)) appends the *same dict objects* to both `estimate_packages` and `audit_packages`. Mutating eligibility in place keeps `package_candidates_audit` (the `package_candidates` key on the v4 output) automatically consistent — the audit shows a confirmed-but-subsumed package, which is exactly the truth.

## Where it slots in

In `compute_renovation_estimate_v4` ([tools/renovation_estimate_v4.py](../tools/renovation_estimate_v4.py)), between finalize and bathroom expansion:

```
packages, package_candidates_audit = finalize_package_candidates(...)   # line ~206
packages, subsumption_audit = apply_same_unit_package_subsumption(packages)   # NEW
packages, bathroom_expansion_audit = expand_bathroom_modernization_packages(...)  # line ~214
whole_home_turnover = aggregate_whole_home_turnover(packages)            # line ~220
```

Ordering is load-bearing, both directions:
- **Before bathroom expansion** — expanded bathroom packages get `estimate_unit_id = ""` ([`_build_expanded_bathroom_package`, tools/rehab_packages.py:2183](../tools/rehab_packages.py#L2183)), which would break unit matching. Pre-expansion, every room package still carries its unit id.
- **Before whole-home turnover aggregation** — `aggregate_whole_home_turnover` ([tools/rehab_packages.py:2352](../tools/rehab_packages.py#L2352)) must only see surviving turnover packages, so a suppressed room no longer contributes to (or single-handedly creates) the property-wide bundle. It requires ≥2 distinct rooms; after suppression that check does the right thing automatically.

Attach the audit: `v4_estimate["package_subsumption_audit"] = subsumption_audit`, and add `"package_subsumption"` to `provenance.v4_phases_applied`.

Also verify (should already hold, assert in tests rather than changing code): `apply_package_verifications_to_candidates` runs BEFORE this pass, so line-item candidates gated by a confirmed-then-subsumed repair package stay `is_valid_detection=True` — their children are exactly what the drop design relies on being retained/re-absorbed.

## Tests

Run with `.venv\Scripts\python.exe -m pytest` (bare `python`/`py` don't resolve the venv), scoped to `tests/test_rehab_packages.py tests/test_package_taxonomy.py tests/test_renovation_estimate_v4.py`. **7 tests fail on this branch pre-existing and unrelated** — take a baseline before changing anything and diff against it. `tests/` is untracked — never `git stash` test paths.

New unit tests (suggest a `TestSameUnitSubsumption` class in test_rehab_packages.py):

1. partial/full modernization + repair, same unit → repair dropped from active list; `subsumed_by_package_id` set; `verification_status` unchanged; winner carries `subsumed_package_ids`.
2. refresh modernization + repair, same unit → both survive.
3. modernization (each tier incl. refresh) + turnover, same unit → turnover suppressed.
4. repair + turnover, no modernization → both survive (regression of current behavior).
5. same package types in different units → untouched.
6. whole-home aggregation interplay: turnover in 2 rooms, one suppressed by rule 2 → `aggregate_whole_home_turnover` returns None (only 1 distinct room remains).
7. end-to-end through `reconcile_packages_and_estimate_units`: with a subsumed repair package, (a) its plumbing-trade child is retained (not absorbed by modernization — scope excludes plumbing) and still lands in `required_rehab` rollups; (b) its tile/flooring child IS broad-absorbed by the modernization package with reason `same_unit_line_item_scope`; (c) `package_total_low/high` exclude the dropped repair range.
8. ordering: subsumption before bathroom expansion still expands a surviving modernization package normally.

## Gotchas

- `pricing_tier` on the package dict is the short token (`"partial_rehab"`, `"refresh"`, `"repair_heavy"`); `pricing_profile` is the long key (`"bathroom_partial_rehab"`). Gate rule 1 on `pricing_tier`.
- Phase B absorption order is currently "modernization < repair < turnover" purely because `infer_package_candidates` sorts by `(unit_id, package_type)` and the alphabet cooperates. After this pass the within-unit competition mostly disappears, but **do not reorder the list** — absorption is first-come-first-absorbed and order changes are silent behavior changes.
- Do not deepcopy the loser packages when mutating — the audit-list aliasing (above) is the mechanism, not a bug.
- `_build_project_scope_breakdown` ([tools/renovation_estimate_v4.py:440](../tools/renovation_estimate_v4.py#L440)) and `_build_scope_rollups` ([tools/rehab_packages.py:3244](../tools/rehab_packages.py#L3244)) both iterate the active `packages` list — they pick up the change automatically; don't special-case them.
- Turnover packages are `confirmed_by_rule` (no VLM), modernization is VLM-`confirmed` — both are ACTIVE; the rules above don't distinguish, and shouldn't.
- [docs/frontend_contract_packages.md](frontend_contract_packages.md): the active `packages` list shrinks and dropped packages remain in `package_candidates` with `audit_only=true` plus the new `subsumed_by_package_id`/`suppressed_by_package_id` fields — add a short note there if the frontend renders package candidates.
- Pass 2f scaffolding (per-item premium verification revival is planned) — this pass runs entirely after verification; don't remove or refactor anything in `run_pass_2f_batch` / verification plumbing.
- Fix #1 (catalog-driven `package_affinity`) just landed in the working tree — line numbers in this doc reflect that state; re-grep anchors if more changes land before you start.

## Out of scope

- Region/size cost factors, tier-table calibration (separate handoff).
- `estimate_tier: "low"` coercion, `estimate.group: "paint"` cap vocabulary, catalog JSON-Schema validator (separate handoff).
- Cross-room dedupe (ambient supports) — already handled by the ambient-support demotion; unrelated mechanism.
- Making absorption precedence an explicit constant — nice-to-have hardening; only do it if trivially provable as a no-op, otherwise leave.
