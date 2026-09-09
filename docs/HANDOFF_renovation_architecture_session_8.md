# Handoff — Session 8: class triage, offline analyses, Sol guard, rerun readiness

Date: 2026-08-17. Branch: `renovation_architecture_rework` (HEAD at session
start: `b77b388`). Everything in this session was offline — zero Terra/Sol
provider tokens spent. Companions:
`docs/HANDOFF_renovation_architecture_67_item_class.md` (now carries a
Session 8 correction block), `docs/RUNBOOK_renovation_architecture_session_6.md`
(rerun procedure, updated), the three `docs/analysis/session8_*.md` memos.

## Decisions made (Steven, 2026-08-16, via Q&A)

1. **The 48-item class is accepted as intended v5 scope** (the class is 48
   billable items, not 67 — see the correction block in the 67-item handoff).
   No wholesale suppression, no package-feed-only contract extension, and the
   headline keeps summing all estimate scopes.
2. **Five opportunity/presence items are triaged out of billing**:
   `unfinished_basement_present`, `staging_or_decluttering_opportunity`,
   `mismatched_or_inconsistent_furniture_staging`, `curb_appeal_upgrade`,
   `landscaping_enhancement_opportunity`. Explicitly kept billing:
   `paint_refresh_recommended`, `landscaping_overgrown_or_neglected`.
3. Mechanism: a new **non-economic `route_override` catalog field** (v4
   ignores it; estimate blocks were rejected because five v4 readers would
   activate the items in production).

## What landed

### route_override (the triage mechanism)

- `tools/catalog_validation.py`: `VALID_ROUTE_OVERRIDES = {"no_action"}` +
  `_validate_route_override` (rejects unknown values, dead overrides under
  quarantine/`drop_if_generic`/`inspect_only`, and redundant overrides on
  items with no economics — reason codes must state intent).
- `tools/renovation_architecture/catalog_projection.py`:
  `resolve_terminal_route` gains a 4th-precedence branch →
  `("no_action", "route_override_no_action")`; unknown values fail the build.
- `tools/renovation_architecture/contracts.py`:
  `TERMINAL_ROUTE_POLICY_VERSION = "terminal_route_v2"`. All Terra/Sol
  checkpoint fingerprints from prior runs are dead (they already were, since
  `b77b388` killed the freeze).
- Authoring: 4 carryover items in v1 `tools/issue_catalog.json`;
  `landscaping_enhancement_opportunity` via its split-successor `overrides`
  block in `tools/catalog_migrations/kind_v2_decisions.json` (NEVER on the v1
  parent — the degradation sibling would inherit it).
- `scripts/migrate_catalog_kind_v2.py`: `route_override` added to
  `INHERITED_FIELDS`, plus a hardening fix — unknown override keys on split
  successors used to be silently ignored; they now hard-fail.
- Regenerated `tools/issue_catalog_kind_v2.json` (exactly +5 lines; manifest
  and audit byte-identical). **New pins: routes 12/4/5/9/98, scope
  distribution 35/58/5/5.**
- Tests updated/added across `tests/test_renovation_architecture_catalog.py`
  (precedence, shipped pins, the `landscaping_overgrown_or_neglected == work`
  sibling tripwire), `tests/test_renovation_architecture_runtime.py`,
  `tests/test_catalog_kind_v2.py` (generator strictness + shipped override
  pins), `tests/test_catalog_validation.py`.

### Offline replay tool (new)

`scripts/replay_renovation_architecture.py` — replays stored canary
artifacts through the current working-tree code/catalog with zero provider
calls: stored verdicts and Sol decisions reused verbatim (held-fixed
`estimate_id` keeps every derived ID joinable), dispositions recomputed from
verdicts, work/candidates/reconciliation rebuilt, full Session 4+5 gates and
audits enforced, sanitization counted. Outputs per-property JSON + a corpus
report under `<canary root>/analysis_session8/<label>/`.

- **Baseline run** (pre-triage code): reproduced the Session 7 residual to
  the dollar — corpus +$52,789 low / +$469,719 high vs v4 (+8.9%/+24.3%),
  zero sanitization events, zero disposition diffs.
- **Triage run** (post-route_override): the A/B delta is exactly the 5-item
  effect — 13 conditions flipped to `no_action`/`route_no_action`, 13 work
  items removed ($7,548/$141,582 standalone), package formation structurally
  stable (no candidates appeared/vanished; the ambient-support non-monotone
  risk did not materialize), audits empty. **Headline effect −$6,810 low /
  −$129,767 high; corpus residual vs v4 now +7.7% low / +17.6% high.**
  The two staging items were dormant in this corpus (no active work).

### Analyses (`scripts/analyze_session8_canary.py`)

- `docs/analysis/session8_group_caps_measurement.md`: v4-style group caps
  over v5's honestly-cappable items remove $16,845/$47,813 at the work
  layer, of which package absorption neutralizes $8,926/$9,148 — a
  **realized headline reduction of $7,919 low / $38,665 high (1.22%/1.61%)**.
  v5 already caps each catalog item across occurrences
  (`tools/costing.py:286`); the group cap only adds a cross-item ceiling.
  Recommendation: document intentional divergence; revisit inside price
  calibration. **Decision open (Steven).**
- `docs/analysis/session8_floor_vs_sum_trace.md`: exact per-property
  decomposition of v5−v4. Corpus: class scope +$67k/+$660k, package floor
  lift +$463k/+$640k, residual −$478k/−$830k (v5 prices shared scope lower).
  The class was the headline story; floor lift and negative residual largely
  cancel. Feeds the rerun's >15% delta reviews.
- `docs/analysis/session8_terra_budget_denominator.md`: **verified from a
  production artifact** — the Terra service already carries upstream passes
  2a/2b/2c/**2d** in production, and production 2f rides the **Sol** service
  (canary routing differed: 2d local, 2f Terra). Readings: ~48 listings/day
  (architecture-only) vs ~10–12/day (all-service). Recommendation:
  all-service. **Decision open (Steven)** — including whether the Sol 250k
  budget covers production 2f.

### Sol daily budget guard

- `tools/renovation_architecture/usage_guard.py`: refactored into a
  parameterized per-model ledger (Terra public API byte-compatible);
  `SolUsageLedger`, `SolDailyBudgetExceeded`,
  `estimate_sol_reservation_tokens` (10k floor — Sol calls average ~8.6k;
  Terra's 25k floor would spuriously deny), `sol_usage.sqlite3` beside the
  Terra ledger under the same `RENOVATION_TERRA_USAGE_ROOT`.
- Hook in `tools/renovation_architecture/sol_review.py` fresh-call branch
  only: reserve → call → settle-to-provider-truth-before-parse; denial is
  `PassExecutionError` code `SolDailyBudgetExceeded` → category `quota`
  (`tools/failure_taxonomy.py`). Checkpoint replay and zero-candidate paths
  never touch the ledger. `SolCall` contract unchanged (no schema-v6 bump);
  the stale "Sol has no budget" docstrings were corrected.
- Config: `RENOVATION_SOL_DAILY_TOKEN_CEILING` (default 250,000) in
  `tools/pipeline_config.py`.
- New `tests/test_renovation_architecture_sol_guard.py` (14 tests: formula,
  denial-before-call with zero provider requests, settle ordering,
  Terra/Sol ledger isolation, checkpoint/zero-candidate no-touch, config
  honoring, resolver).

## Tests

Convention `--basetemp=C:\tmp\rv_pytest_<session>_<scope>_<date>`.

- Session 8 baseline (pre-change, at `b77b388`): **2251 passed / 6 skipped**
  (`C:\tmp\rv_pytest_session8_baseline_20260817`).
- Final full suite: **2287 passed / 6 skipped, 24.05s** (+36 new tests)
  (`.venv\Scripts\python.exe -m pytest tests -q
  --basetemp=C:\tmp\rv_pytest_session8_full_20260817`).

## Decisions closed at session end (Steven, 2026-08-17)

1. **Group caps: intentional divergence — do not build.** Recorded in
   `docs/analysis/session8_group_caps_measurement.md` (realized effect
   1.22%/1.61% of headline; v5's per-item caps + dedup already cover the
   risk; revisit inside price calibration if ever).
2. **Terra denominator: all-service — settled by fact.** The 2.5M/day is
   the OpenAI free daily token allowance on the Terra model's tier
   (input+output combined); Sol's tier has ~250k/day. One canary replica
   (~3.98M on the Terra model) exceeds a full free day by itself. Recorded
   in `docs/analysis/session8_terra_budget_denominator.md`.
3. Remaining before the rerun report: sign-off on the triage deltas
   (`analysis_session8/triage/report.md`).

## Next session: token budget / affordable rerun (decision handoff written)

`docs/HANDOFF_renovation_architecture_token_budget.md` — observations and
constraints only; the next session owns the design decisions. It records
the budget facts (one replica ≈ 3.98M Terra tokens > one free-quota day),
the verified seam (`tools/vlm_client.py` is the single OpenAI choke point
with per-pass attribution already in memory, never persisted), the
double-debit and reservation-sizing hazards, the resume machinery, the
option space, and the exact context/reading list that session needs
(including: the source architecture docx is NOT needed for it).

## Handed off to future tasks (explicitly NOT Session 8)

- Terra batch-quality review (20-condition/12-image batches).
- Upstream API usage reduction (~170k/listing upstream; cached-input reuse).
- Acceptance-rate audit (71.4% Terra / 93.7% Sol).
- Widening the Terra guard's meter to upstream passes (if all-service wins).
- Artifact-visible Sol debit field (deliberate schema-v6 change, if wanted).

## The paid rerun (next session)

Follow `docs/RUNBOOK_renovation_architecture_session_6.md` with:
- a **new output root and fresh `input_freeze.json`** (Session 8 changed the
  catalog sha, projection fingerprint, and `terminal_route_v2`; every old
  checkpoint is dead);
- the Sol guard active; replicas on **separate days** via `--replicate 1|2`
  (two same-day replicas ≈ 309k Sol tokens > 250k ceiling);
- expected comparison anchors: corpus residual vs v4 ≈ +7.7% low / +17.6%
  high (replica-1 replay); per-property delta explanations from
  `docs/analysis/session8_floor_vs_sum_trace.md`;
- per-pass Terra token recording so the denominator memo's all-service
  number stops being a proxy.

## Risks / gotchas carried forward

- The `landscaping_overgrown_or_neglected == work` pin is a permanent
  tripwire — an override on the v1 parent `landscape_improvement_needed`
  would silence both landscaping successors.
- `route_override` is validation-gated to billable items only; authoring it
  next to `drop_if_generic`/quarantine/`inspect_only` or on no-economics
  items is a hard catalog error by design.
- Replay tool: dispositions must always be recomputed from verdicts, and
  `estimate_id` held fixed — reusing stored dispositions replays stale
  routes.
- Nobody "quickly re-runs the canary to check" — every seam-mode run against
  any root now re-buys all provider calls.
