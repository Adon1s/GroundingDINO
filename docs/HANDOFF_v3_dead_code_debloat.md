# Handoff: v3 dead-code removal + `project_scope_breakdown` $0 fix

## Status / framing

A frontend session wiring the Project Scope Breakdown card reported that the backend emits **$0 costs and 0 item counts for every `project_scope_breakdown` entry** while still listing contributing packages. This session root-caused that bug (Finding A, reproduced, unfixed) and, while tracing it, mapped exactly which v3-era outputs are dead (Finding B, fully verified against backend, tests, and the frontend repo at `C:\Users\Steven\IntelliJProjects\realtorvision`).

**User directive (binding):** perform a thorough but careful debloat / legacy / dead-code removal. This handoff is the findings; the next session plans and implements. Nothing below has been changed yet — all line refs are against branch `reno_estimate_refactor` as of 2026-07-05.

## Binding constraints — read before deleting anything

1. **Do NOT delete the dormant per-item Pass 2f scaffolding.** The user plans a per-photo, issue-level premium verification pass; this machinery is its landing pad. Keep: `v3_reviewed_candidates` param + `_reuse_pass_2f_fields` ([tools/renovation_estimate_v4.py:553](../tools/renovation_estimate_v4.py#L553)), `resolve_pass_2f_model_config`, `pass_2f_review_audit` output, per-candidate `pass_2f_*` fields, and the `reviewed_candidates` plumbing in [tools/artifact_writers.py:774](../tools/artifact_writers.py#L774) (hardcoded `None` today — dormant, not dead).
2. **Slim philosophy:** deletions only where readers are proven absent; no compatibility shims, no "just in case" flags. When a semantics decision is needed (see WI-1 cap question), pick one and document it in [docs/frontend_contract_packages.md](frontend_contract_packages.md), don't build both.
3. **Test environment facts:** run tests with `.venv\Scripts\python.exe -m pytest` (bare `python` doesn't resolve the venv). 7 tests fail on this branch pre-existing — scope pytest to touched files. `tests/` is untracked on this branch: never `git stash` test paths.
4. `backfill_reno_v4.py` is **unsafe on VLM-confirmed artifacts** (recompute without replaying stored `package_verifications` zeroes all packages via the `not_run` gate). Don't "fix" it as part of this work; only adjust its gate key if WI-4 requires.

## Finding A — bug: `project_scope_breakdown` emits $0 (fix first; it's the product surface)

`_build_project_scope_breakdown` ([tools/renovation_estimate_v4.py:454](../tools/renovation_estimate_v4.py#L454)) was written against the **line-item** shape but iterates **groups**:

- Line 468 reads `group.get("trade_bucket")` — group dicts (built by `compute_group_estimate`, [tools/renovation_estimate.py:851](../tools/renovation_estimate.py#L851)) have no such key, so `trade` is always `""` and line 470's `continue` skips **every group**.
- Even past the guard, lines 484-485 read `group.get("cost_low"/"cost_high")` — group costs are keyed `low`/`high`. The keys it reads (`trade_bucket`, `cost_low`, `cost_high`) exist only on `group["line_items"][*]`.
- The packages loop (lines 489-511) then creates all entries via `setdefault` and by design adds only `trade_buckets` + `contributing_package_ids`, never dollars.

Net effect (reproduced with a correctly-shaped kitchen group $8k–16k + one package): every entry $0 / 0 items with packages listed; **scopes with line items but no package are missing from the array entirely**; `trade_buckets` reflects only package absorption scopes.

**Fix (~10 lines):** iterate line items —

```python
for group in groups or []:
    for li in group.get("line_items") or []:
        trade = str(li.get("trade_bucket") or "")
        if not trade:
            continue
        scope_id = get_project_scope(trade, strict=False)
        entry = by_scope.setdefault(scope_id, {...})
        entry["cost_low"] += int(li.get("cost_low") or 0)
        entry["cost_high"] += int(li.get("cost_high") or 0)
        entry["item_count"] += 1
        entry["trade_buckets"].add(trade)
```

Notes for the implementer:

- The `try/except KeyError` around `get_project_scope(trade, strict=False)` is dead — `strict=False` never raises ([tools/project_scopes.py:90](../tools/project_scopes.py#L90) returns `"unknown"`). Drop it in the same edit (both loops).
- **Semantics decision:** line-item sums are pre-cap; for `max_only`/`group_cap` groups the breakdown will total above the capped headline. Accept that (it's a presentation rollup; v3's equivalent had the same semantics) and document it, OR allocate capped group totals proportionally (pattern: `allocate_capped_scope_totals` in [tools/rehab_packages.py](../tools/rehab_packages.py)). Recommend: accept raw sums for now.
- **Do not add package dollars to the breakdown.** Absorbed line items already carry base cost; adding package cost double-counts. Package premium above absorbed items being absent from this card is a known, accepted gap.
- `scale_estimate_dollars` ([tools/cost_factors.py:116](../tools/cost_factors.py#L116)) walks `*_low`/`*_high` keys generically — the fixed numbers pick up the market/size factor with no further change.
- **Zero tests exist for this function.** Add one feeding a group shaped exactly like `compute_group_estimate` output (keys: `group`, `low`, `high`, `line_items[*].{trade_bucket,cost_low,cost_high}`) plus a package with `absorption_scope.trade_buckets`; assert nonzero costs, correct item_count, and that a package-less scope still appears.
- Frontend interim guidance (already relayed): hide the card while every entry is `cost_low == 0 && cost_high == 0`.

## Finding B — the v3 dead-code map (three layers, different verdicts)

Context: v4 is a decorator around the v3 engine, not a separate pipeline. "v3 fed v4" is historical; both feed channels are now severed (see Layer 2).

### Layer 1 — v3 engine: ALIVE, do not touch

`compute_renovation_estimate` ([tools/renovation_estimate.py:1007](../tools/renovation_estimate.py#L1007)) is v4's core aggregation engine — v4 calls it at [tools/renovation_estimate_v4.py:232](../tools/renovation_estimate_v4.py#L232) with `prebuilt_candidates=v4_candidates` and decorates the result. All candidate/group/tier machinery stays.

### Layer 2 — standalone v3 estimate (`quick_est` / `photo_intel["renovation_estimate"]`): ZOMBIE

Every pipeline run computes a **second, standalone** v3 estimate ([tools/artifact_writers.py:814](../tools/artifact_writers.py#L814)) and stores it at [line 821](../tools/artifact_writers.py#L821). Its content is never read by v4:

- `v3_estimate=quick_est` → v4 uses it **only** as a None-gate ([tools/renovation_estimate_v4.py:115](../tools/renovation_estimate_v4.py#L115)); `compute_renovation_estimate` never returns None, so the gate never fires in the writer path.
- `v3_reviewed_candidates` → always `None` in prod ([tools/artifact_writers.py:774](../tools/artifact_writers.py#L774)). (Dormant landing pad — keep, per constraint 1.)
- Writer failure path nulls **both** keys together ([tools/artifact_writers.py:873-874](../tools/artifact_writers.py#L873)) → new artifacts can never have v3 without v4.

Complete reader map for the stored `renovation_estimate` key (re-verified this session):

| Reader | Where | Nature |
|---|---|---|
| `compare_reno_estimates.py` | [tools/compare_reno_estimates.py:46](../tools/compare_reno_estimates.py#L46) | manual diagnostic CLI, v3-vs-v4 comparison |
| `rerun_pass_2f_artifact.py` | [:281](../tools/rerun_pass_2f_artifact.py#L281) reads `old_total`; [:298,318](../tools/rerun_pass_2f_artifact.py#L298) recomputes & rewrites both keys | tooling |
| `backfill_reno_v4.py` | [:119](../tools/backfill_reno_v4.py#L119) existence gate; [:138](../tools/backfill_reno_v4.py#L138) passes to the None-gate | tooling (unsafe anyway, constraint 4) |
| writer logging | [tools/artifact_writers.py:862-870](../tools/artifact_writers.py#L862) logs `quick_est["raw_totals"]/["meta"]` | log only |
| artifact slimming | [tools/artifact_writers.py:923](../tools/artifact_writers.py#L923) strips its 2f rationale | maintenance of the zombie |
| Frontend | [app/api/analysis/route.ts:684](../../..//Users/Steven/IntelliJProjects/realtorvision/app/api/analysis/route.ts) and `normalizePhotoIntelRenovationEstimate` ([lib/types/renovationEstimate.ts:781](../../../Users/Steven/IntelliJProjects/realtorvision/lib/types/renovationEstimate.ts)): `renovation_estimate_v4 ?? renovation_estimate` | **legacy-artifact fallback only** (fires only on pre-v4 artifacts; audit feed routes already "drop pre-v4 deprecated analyses") |
| Tests | `test_renovation_estimate.py:2012` (slimming), `test_rerun_pass_2f_artifact.py:178`, `test_compare_reno_estimates.py:233` | fixtures/assertions to update with WI-4 |

Cost of the zombie: duplicate candidate-extraction/grouping CPU (trivial, no VLM) and a **full duplicate estimate blob (groups + line_items + tier_summary + project_scope_summary) in every `photo_intel.json`** — the real debloat payoff is artifact size.

Prior-intent note: [HANDOFF_cost_factors_market_size_units.md](HANDOFF_cost_factors_market_size_units.md) line 65 said v3 is "kept for comparison." The user's debloat directive supersedes that; the compare tool keeps working on legacy artifacts, which already have the key baked in.

### Layer 3 — fully dead output fields

- **`project_scope_summary`** ([tools/renovation_estimate.py:1216-1241](../tools/renovation_estimate.py#L1216), returned at [:1253](../tools/renovation_estimate.py#L1253)): two references in the universe — its own build and its own return. Zero backend readers, zero frontend source hits, **zero test references**. Unconditionally deletable.
- **`tier_summary` output key** ([:1252](../tools/renovation_estimate.py#L1252)): zero production readers (backend tools and frontend both confirmed). BUT (a) the `compute_tier_summary` *computation* is alive — `totals.validated_total`/`unreviewed_risk_total` are derived from it at [:1183-1195](../tools/renovation_estimate.py#L1183) and the frontend normalizer reads `totals.*`; (b) **4 test sites** read the key directly: `test_renovation_estimate.py:615, 855, 859, 937`. Drop the key from the return, keep the computation, update the tests. Do not delete `pass_2f_review_audit` (constraint 1; frontend normalizer also reads it).

## Work items (suggested order)

| # | Change | Files | Risk | Budget |
|---|---|---|---|---|
| WI-1 | Fix `_build_project_scope_breakdown` (line-items iteration, drop dead try/except) + new test | renovation_estimate_v4.py, tests | low — no readers of the current (broken) values | ~15 lines + test |
| WI-2 | Delete `project_scope_summary` build + return key (and its now-unused imports if any) | renovation_estimate.py | none — zero readers incl. tests | −28 lines |
| WI-3 | Drop `tier_summary` from return (keep computation feeding `totals`); update 4 test sites | renovation_estimate.py, test_renovation_estimate.py | low | −1 line + test edits |
| WI-4 | Retire standalone `quick_est`: stop computing/storing `photo_intel["renovation_estimate"]`; remove the `v3_estimate` None-gate param from `compute_renovation_estimate_v4`; update writer log to v4 totals; simplify failure handler; update `rerun_pass_2f_artifact` (v4-only recompute + `old_total` from v4), `backfill_reno_v4` gate (`renovation_estimate_v4`), slimming line 923; verify `compare_reno_estimates` degrades gracefully when the key is absent (check its None handling at [:46](../tools/compare_reno_estimates.py#L46)); update the 3 test fixtures | artifact_writers.py, renovation_estimate_v4.py, rerun_pass_2f_artifact.py, backfill_reno_v4.py, tests | medium — several coordinated call sites; **keep `v3_reviewed_candidates` param** | net −60..−80 lines |
| WI-5 (optional, frontend repo, separate decision) | Drop v3 acceptance in `normalizeRenovationEstimate` (:709) + `?? renovation_estimate` fallbacks — only if pre-v4 legacy artifacts no longer need to render | realtorvision frontend | product decision, not this repo's PR | — |

WI-1..3 can ship as one PR. WI-4 should be its own PR. Do not fold WI-5 into backend work.

## Verification

- Scoped tests: `.venv\Scripts\python.exe -m pytest tests/test_renovation_estimate.py tests/test_renovation_estimate_v4.py tests/test_rerun_pass_2f_artifact.py tests/test_compare_reno_estimates.py` (remember the 7 pre-existing failures — compare against a pre-change baseline run of the same files).
- After WI-4: run one artifact write end-to-end (or the writer tests) and assert `renovation_estimate_v4` present, `renovation_estimate` absent, `ui_priorities_v1` unchanged (it reads v4 only, [tools/artifact_writers.py:836](../tools/artifact_writers.py#L836)); then `grep -rn '\["renovation_estimate"\]'` for stragglers.
- Frontend contract: after WI-1, refresh the example in [docs/frontend_contract_packages.md §2](frontend_contract_packages.md) if semantics were decided differently than documented; the frontend has `scripts/verify-renovation-v4-frontend.ts` for shape checks.
