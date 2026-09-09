# Handoff — Renovation Architecture Session 5

Date: 2026-08-16. Branch: `renovation_architecture_rework` (Sessions 1–4 are
committed as `f808e48` + `e00698d` + `893f09c` + `f33c35e`; the four Session 4
files remain untracked-by-design and are part of the working baseline; all
Session 5 work is uncommitted — nothing staged or pushed). Scope executed:
Session 5 of `docs/IMPLEMENTATION_PLAN_renovation_scope_estimate_architecture.md`
— deterministic coverage reconciliation after the frozen Session 4
`validate_package_review_result` boundary: package applications, one
reason-coded ledger entry per ACTIVE work item, exact ledger-derived totals,
recomputed audits, observability, the complete v5 envelope, publication-gate
enforcement, and the `audit_runner` runtime wiring. v4 output, frontend
contracts, and `current` mode are unchanged; the catalog file is
byte-identical: SHA-256
`5BE11ABB6DAD9921C6F1A6851C972DA2DCE57F95CCB1FEB8577742A59349E1EC` verified
before and after.

## What was implemented

The private shadow path now runs the full chain to `state="complete"`:
condition review → standalone estimate → deterministic candidates → bounded
Sol review → **deterministic reconciliation**. Reconciliation applies Sol's
decisions under the conservative combine/split policy, produces one
`PackageApplication` per candidate and one `CoverageLedgerEntry` per active
work item, derives totals exclusively from ledger ownership plus applied
effective ranges, recomputes five audit lists (all must be empty), assembles
observability (phase timings, token rollups, funnel counts), and
self-validates before the envelope is returned. The finished envelope is
written privately as `analysis_debug.renovation_estimate_v5` (shadow mode
only, stripped from the slim artifact) and the publication gate now rejects
any malformed, partial, or mixed-version envelope before writing.

### Conservative combine/split policy (approved plan decision)

- **Combine** recommendations become non-economic grouping metadata: every
  member of a combine closure shares a deterministic `combine_group_id`
  (`cg1`, hashed over the sorted member candidate IDs) while retaining its
  own deterministic pricing. Totals are unaffected by grouping.
- **Split** recommendations make the candidate non-billable
  (`not_applied`/`split_recommended`); its children fall back to their exact
  standalone allowances unless independently absorbed by another approved
  candidate. No derived-package pricing is invented anywhere.
- Priced transformed packages require a separately approved follow-up only if
  the Session 6 canary shows this conservative behavior is inadequate;
  cutover must stop rather than invent allocation policy during Session 6.

### Deterministic application semantics

- Absorption-eligible ⇔ decision `approve` AND not `display_only` AND no
  `split_groups`.
- Shared children resolve input-order-independently by the legacy absorption
  ordering (`rehab_packages._absorption_priority_key` adapted to v5 dicts):
  room level before property, repair before modernization before turnover,
  higher `_MODERNIZATION_TIER_RANK` first (imported — single definition, no
  drift), stable candidate ID last. Each child is owned at most once.
- An applied package's effective range is
  `max(unfloored tier spec, Σ actually-owned child allowances)` at both
  endpoints — never the stored candidate floor, which would double-count a
  shared child lost to a higher-priority package.
- An eligible candidate that owns zero children is `not_applied` /
  `no_owned_children` (no tier-floor billing without owned work).
- The display-only whole-home aggregate is `display_only` / 0-0 and can never
  absorb or enter totals (it has no children, structurally).
- Inspection/no-action conditions remain reason-coded dispositions; no work
  items are synthesized for them, the ledger holds active billable work only,
  and the `totals.inspection` lane is 0/0 this schema (the `inspection` /
  `no_action` ledger representations stay in the vocabulary but are
  validator-rejected as "reserved" until a future session gives them reason
  codes).

## Files added

- `tools/renovation_architecture/reconciliation.py` — the focused Session 5
  module: `absorption_priority_key`, `compute_reconciliation` (the single
  deterministic computation, shared verbatim with the artifact gate),
  `recompute_reconciliation_audit` (independent defect scan — never a re-run
  of compute, so a bug there cannot vouch for itself),
  `build_observability`, and `build_complete_result` (frozen S4 input gate →
  compute → audits (raise on any defect) → timings/observability →
  `validate_complete_result` self-check). Typed failures ride
  `PassExecutionError("reconciliation", ...)` with codes
  `PackageReviewResultInvalid` (stage `dependency`), `PhaseTimingsInvalid`,
  `ReconciliationAuditFailure`, `CompleteResultInvalid` (stage `parse`, the
  S4 self-validation precedent).
- `tests/test_renovation_architecture_reconciliation.py` (20 tests) — see
  Tests below.
- `docs/HANDOFF_renovation_architecture_session_5.md` (this file).

## Files changed

- `tools/renovation_architecture/contracts.py` — schema v5 (contracts AND
  envelope); `POLICY_VERSIONS` + `package_application_policy`
  (`package_application_v1`) + `coverage_reconciliation_policy`
  (`coverage_reconciliation_v1`) → 11 keys; `SHADOW_DEBUG_KEY` renamed to
  **`renovation_estimate_v5`** (private under `analysis_debug` during
  migration; the same name at the photo_intel root is reserved for the
  Session 6 cutover); closed vocabularies `APPLICATION_STATUSES`,
  `APPLICATION_REASON_CODES` (6), `LEDGER_REASON_CODES` (5),
  `OBSERVABILITY_PHASES` (6, `total` = exact sum of the other five),
  `FUNNEL_KEYS` (18); new contracts **`PackageApplication`** (`pa1`; 11
  fields incl. absorbed/unabsorbed partitions, optional `combine_group_id`,
  effective range), **`ReconciliationAudit`** (5 lists, all must be empty),
  **`EstimateObservability`** (timings/tokens/funnel);
  **`CoverageLedgerEntry` + `reason_code`**.
- `tools/renovation_architecture/ids.py` — `make_package_application_id`
  (`pa1`), `make_combine_group_id` (`cg1`, order-independent over sorted
  members).
- `tools/renovation_architecture/validators.py` — `pa1`/`cg1` patterns; new
  field sets; `_validate_package_application` (status/reason pairing,
  absorbed/unabsorbed disjointness, non-applied ⇒ 0/0 and no absorption),
  `_validate_reconciliation_audit` (emptiness), `_validate_observability`
  (closed keys, `total == Σ phases`, `combined == terra + sol`);
  `_validate_ledger_entry` gains `reason_code` (+ representation pairing;
  `inspection`/`no_action` reserved); `_RESULT_KEYS` redefined as
  `_PACKAGE_REVIEW_RESULT_KEYS ∪ {package_applications, coverage_ledger,
  reconciliation_audit, observability, totals}`; **`validate_complete_result`
  rewritten** on the S4 pattern: the package-review subset re-validated
  verbatim by the frozen S4 gate (which re-runs the S2/S3 gates and snapshot
  fingerprints), then candidate↔application bijection with decision
  provenance, applied eligibility/partition/effective-range recompute,
  at-most-once absorption, exactly-one-ledger-entry coverage with
  application-consistent ownership, exact standalone allowances, standalone
  fallback for non-billable packages, **exact totals from ledger + applied
  effective ranges**, and finally full recompute equality against
  `compute_reconciliation`/`build_observability` (function-level import —
  reconciliation imports validators at load, so the shared definition stays
  acyclic): validity IS the deterministic reconciliation.
- `tools/renovation_architecture/runtime.py` — `build_shadow_envelope` times
  each stage (`time.monotonic`, ms), chains `build_complete_result`, succeeds
  with `state="complete"`, and self-validates the finished envelope via
  `validate_envelope` before returning (`CompleteEnvelopeInvalid` → failed
  envelope through the existing `classify_failure` path). Docstring updated.
- `tools/artifact_writers.py` — seam writes under the imported
  `SHADOW_DEBUG_KEY`; the **last-resort literal envelope is deleted**: a
  seam-level explosion now logs and omits the key entirely (the old literal
  had `estimate_id: None` / `provenance: None` and could never validate — a
  malformed envelope must never reach the artifact).
- `tools/publication_gate.py` — `validate_publication_payload` now ends with
  `_validate_renovation_architecture_keys`: the private
  `analysis_debug.renovation_estimate_v5` must be a valid v5 envelope in
  state `complete` or `failed` (intermediate states are valid envelopes but
  not publishable); a root `renovation_estimate_v5` must already be a valid
  `complete` envelope (reserved for Session 6). Violations `_reject` before
  any write. Imports are local, so publications without the keys pay nothing.
- `tools/audit_runner.py` — `_run_all` wires
  `initialize_renovation_architecture` immediately after catalog load (the
  analyzer entrypoints' startup pattern), closing the Session 1 deferred
  writer entrypoint. No-op in `current` mode; fail-closed at startup in
  `shadow`; `new` mode remains impossible before Session 6.
- Tests updated: `test_renovation_architecture_contracts.py`
  (`_complete_result` fixture now built by the real
  `build_complete_result` over the S4 fixture — hand-built approximations
  cannot pass the recompute-equality gate; `_ledger_entry` gains
  `reason_code`; 11-key POLICY_VERSIONS pin; `pa1`/`cg1` ID shapes; S4
  later-session-key rejection extended to the five v5 keys; TestCompleteResult
  reworked: subset-gate delegation tests now keep earlier layers internally
  consistent, plus new tests for application bijection/pairing, effective
  range recompute, audit emptiness, observability reconciliation
  (timings/tokens/funnel), reason-code drift caught by recompute equality,
  and exact totals), `test_renovation_architecture_runtime.py` (end state
  `complete` with empty reconciliation + zero totals + timings; renamed seam
  test asserts key omission on seam failure),
  `test_renovation_architecture_terra.py` /
  `test_renovation_architecture_sol.py` (end-to-end envelopes now assert the
  reconciliation tail: exact standalone ledger entry and totals; applied
  absorption with 0/0 absorbed entry and effective packaged totals; retry
  still replays Terra checkpoints), `test_kind_ontology_guards.py` (8 new
  publication-gate tests: private complete/failed accepted, malformed /
  stale-schema / partial-state private rejected, root complete accepted,
  root failed / malformed rejected).
- NOT changed: `tools/renovation_architecture/checkpoints.py` (fingerprints
  carry the schema version, so v5 invalidation is automatic),
  `package_candidates.py`, `sol_review.py`, `analyzer_server.py`,
  `analyzer_cli.py`, `pipeline_config.py`, `renovation_estimate_v4.py`,
  `rehab_packages.py` (read-only import of `_MODERNIZATION_TIER_RANK`).

## Decisions and deviations

- **Correction to the approved plan text #1 — packaged totals arithmetic**:
  the pre-S5 `validate_complete_result` summed *stored* candidate `low/high`
  over absorbed packages, which double-counts when a shared collapse-policy
  child is owned by a higher-priority package (the loser's floor still
  contains the child's dollars). The v5 gate computes `packaged` from applied
  applications' effective ranges. This is the sanctioned Session 5 extension
  of the gate (S4 handoff: reconciling result keys "is Session 5's call"),
  bumped with the schema.
- **Correction #2 — last-resort writer envelope**: replaced with key
  omission (see Files changed); the plan's own rule ("omit the key instead
  of writing malformed data") decided it.
- **Validity IS the recompute**: beyond the independent invariant checks,
  the gate compares `package_applications`/`coverage_ledger`/`totals` (and
  observability tokens/funnel) for exact equality with the single shared
  deterministic computation. Reason-code or group-ID drift that no local
  invariant could catch fails the gate.
- **Ledger standalone reason codes** come from the highest-priority covering
  candidate (`package_rejected`/`package_uncertain`/`package_split`), or
  `no_covering_package`; an unowned child covered by any *eligible* candidate
  is impossible (it would have been absorbed), so no further lanes exist.
  `package_not_applied` was dropped from the plan's sketch for that reason.
- **`total` phase timing is defined as the exact sum** of the five phases, so
  the validator enforces equality instead of an inequality; upstream stage
  timings are measured in `build_shadow_envelope`, the reconciliation phase
  inside `build_complete_result`.
- **Deviation — `__init__.py` unchanged**: the plan said to export
  `build_complete_result`, but `reconciliation.py` imports `rehab_packages`
  (the heavy chain the package docstring explicitly keeps out of
  current-mode startup — same rule as `catalog_projection`). Import it as a
  submodule; `validate_complete_result` was already exported.
- **Deviation — audit_runner init error handling**: `RenovationArchitectureInitError`
  propagates out of `_run_all` (aborting the CLI run) rather than being
  formatted like `analyzer_cli`'s JSON summary; fail-closed is the
  requirement and the audit CLI's style is raise-y. No audit_runner test
  suite exists; the wiring is import-covered and the seam behavior is pinned
  by the writer tests.
- **`test_work_item_absorbed_by_two_approved_packages` replaced**: under v5
  two approved candidates sharing a child is *legal input* resolved by
  priority (not a validation error), so the old contracts test became the
  reconciliation suite's priority/order-independence tests plus a
  hand-tamper double-absorption gate test.
- Subset-gate delegation means earlier-layer tampers in complete-result tests
  must keep the tampered layer internally consistent (e.g. a full
  inspection disposition triple; a Terra call for an added condition) so the
  targeted invariant — not a shallower shape error — is what fires.

## Commands run and results

All on 2026-08-16, all green:

```
.venv\Scripts\python.exe -m pytest tests\test_renovation_architecture_reconciliation.py tests\test_renovation_architecture_contracts.py tests\test_renovation_architecture_runtime.py tests\test_renovation_architecture_package_candidates.py tests\test_renovation_architecture_sol.py tests\test_renovation_architecture_work_items.py tests\test_renovation_architecture_terra.py tests\test_renovation_architecture_conditions.py tests\test_renovation_architecture_disposition.py tests\test_renovation_architecture_usage_guard.py tests\test_renovation_architecture_catalog.py tests\test_kind_ontology_guards.py tests\test_artifact_writers.py -q --basetemp=C:\tmp\rv_pytest_session5_focused_20260816
  -> 417 passed, 4.90s

.venv\Scripts\python.exe -m pytest tests\test_cost_factors.py tests\test_estimate_units.py tests\test_renovation_estimate.py tests\test_renovation_estimate_v4.py tests\test_rehab_packages.py tests\test_artifact_writers.py tests\test_catalog_validation.py tests\test_kind_ontology_guards.py -q --basetemp=C:\tmp\rv_pytest_session5_regression_20260816
  -> 566 passed, 1 skipped, 10.77s
     (= the Session 4 baseline 537 passed + 1 skipped for the pure-v4 files,
      plus the 21 pre-existing + 8 new kind-ontology gate tests; the v4
      suites themselves are unchanged from baseline)

.venv\Scripts\python.exe -m pytest tests\test_embeddings_failclosed.py -q --basetemp=C:\tmp\rv_pytest_s5_scratch7
  -> 9 passed   (the audit_runner-adjacent suite; init wiring is inert in current mode)

.venv\Scripts\python.exe -m pytest tests -q --basetemp=C:\tmp\rv_pytest_session5_full_20260816
  -> 2228 passed, 6 skipped, 24.29s   (Session 4 baseline: 2192 passed, 6 skipped; +36 new tests)
```

`git diff --check` clean (pre-existing CRLF warnings only); `git status`
shows exactly the 11 intended tracked modifications plus the 2 new files
(`tools/renovation_architecture/reconciliation.py`,
`tests/test_renovation_architecture_reconciliation.py`) — no unrelated or
staged user changes touched, the DOCX lock file untouched; `Get-FileHash
tools\issue_catalog_kind_v2.json` reproduces
`5BE11ABB6DAD9921C6F1A6851C972DA2DCE57F95CCB1FEB8577742A59349E1EC`. All
warnings in the runs are pre-existing (utcnow deprecation, google-genai).
Fake providers only; no live Terra or Sol call was made at any point.

## Session 5 acceptance gates — all PASS

- **Exact low/high arithmetic reconciliation** — totals are computed only
  from ledger ownership plus applied effective ranges; the gate recomputes
  both endpoints exactly and the recompute-equality net catches everything
  else. Pinned by the partial-ownership, tier-floor, and totals-tamper tests.
- **Rejecting or marking every package uncertain reproduces the exact
  standalone total** — `TestStandaloneParity` (reject, uncertain, split, and
  zero-candidate variants) asserts `totals == standalone_estimate.headline`
  to the dollar with `packaged == 0/0`.
- **No accepted work disappears from the ledger** — exactly one entry per
  ACTIVE work item (gate + independent audit recompute + missing-entry
  test); accepted-condition→work lineage is re-enforced by the delegated S3
  gate inside every v5 validation.
- **No work item is absorbed by more than one package** — at-most-once
  ownership by construction (priority `setdefault`), the gate's independent
  `child_owner` check, the audit's duplicate-absorption scan, and the
  hand-tamper test ("absorbed by both").
- **Unsupported and `cannot_assess` conditions cannot enter visible rehab
  totals** — structural: they never become work items (S2/S3 gates,
  re-validated verbatim in every v5 result), and the ledger covers work
  items only.
- **Current mode continues to emit an unaffected v4 result** — writer-seam
  current-mode test re-run green; the v4 regression suites byte-match the
  Session 4 baseline; catalog byte-identical.
- **Mixed, partial, or invalid new artifacts fail before canonical
  publication** — the publication gate validates both key placements
  (schema-version mismatch, malformed content, and intermediate states all
  `_reject` before the JSON write); the runtime self-validates before
  returning; the writer omits the key rather than writing malformed data.
  Eight gate tests pin both directions.
- **Every suppressed/audit-only record has a reason code** — ledger entries
  and applications carry closed-vocabulary reason codes (validator-enforced);
  suppressed work items keep the S3 `dedup_collision` reason; the
  display-only aggregate is reason-coded `display_only_aggregate`.

## Risks / notes for the reviewer

- **Schema v5 invalidates every pre-v5 Terra and Sol request fingerprint**
  (the fingerprint's first input is the contracts schema version), so the
  next shadow run re-reviews conditions and re-buys the Sol call; the daily
  Terra usage guard controls the spend. Pre-v5 shadow payloads in old
  artifacts are stale debug data, nothing more.
- **The private shadow key changed names**
  (`renovation_architecture_shadow_v1` → `renovation_estimate_v5`): any old
  debug artifact still carries the old key, which the publication gate
  ignores entirely. Session 6 tooling should read only the new key.
- **`validate_complete_result` now imports `reconciliation` (hence
  `rehab_packages`) at call time** — fine in shadow/tests; do not call the
  complete gate on a current-mode hot path that must stay import-light.
- **The publication gate makes a non-finished shadow envelope job-fatal** at
  write time. The runtime can only produce `complete`/`failed` (self-checked),
  and a seam explosion omits the key, so this can fire only on a future
  producer bug — which is exactly when the run *should* fail loudly.
- **`total` timing = sum of phases** by definition; it deliberately excludes
  inter-stage envelope overhead (negligible, and it buys an exact validator).

## Session 6 prerequisites

- Input contract: a `complete` v5 envelope validated by `validate_envelope`;
  the private key is `analysis_debug.renovation_estimate_v5`. Promotion to
  the root `renovation_estimate_v5` key is Session 6's cutover step — the
  publication gate already accepts only a valid `complete` envelope there.
- The comparison must use fresh shadow runs: v5 invalidated all pre-v5
  checkpoints, and no historical shadow payload carries the new
  reconciliation sections.
- Funnel counts, phase timings, and Terra/Sol token rollups for the token
  report are in `result.observability`; per-call detail stays in
  `terra_calls` / `sol_calls`.
- The `new` selector remains impossible before Session 6
  (`initialize_renovation_architecture` raises); enabling it, worker
  restart/smoke, canary comparison, and delta review are Session 6 scope.
- Known S6 review items carried forward from S4: the missing
  `_extract_package_only_candidates` evidence lane, stricter multi-photo
  corroboration, S3 pricing deltas — now joined by the conservative
  combine/split policy (watch whether Sol's combine/split recommendations
  materially diverge from what a priced transformed package would produce).
- If a future session wants inspection/no-action ledger lanes, give those
  representations reason codes and a policy bump; today they are
  validator-reserved.

## Summary

Session 5 is complete and every acceptance gate passes: reconciliation is
deterministic, order-independent, and exact; nothing accepted is lost or
double-counted; non-billable packages fall back to exact standalone
allowances; v4 and current mode are untouched; malformed artifacts cannot be
published; and the full suite is green (2228 passed, 6 skipped).
