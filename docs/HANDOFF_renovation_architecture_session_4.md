# Handoff — Renovation Architecture Session 4

Date: 2026-08-15. Branch: `renovation_architecture_rework` (Sessions 1–3 are
committed as `f808e48` + `e00698d` + `893f09c`; all Session 4 work is
uncommitted by design — nothing staged or pushed). Scope executed: Session 4
of `docs/IMPLEMENTATION_PLAN_renovation_scope_estimate_architecture.md` —
deterministic package candidates from ACTIVE work items via the legacy
inference primitives, and one bounded text-only Sol listing review with a
closed decision contract. The private shadow path now ends at
`package_review_complete`; v4 output, frontend contracts, decision
application, the coverage ledger, and totals are unchanged (Sessions 5–6);
`current` mode stays a no-op. The catalog file is byte-identical: SHA-256
`5BE11ABB6DAD9921C6F1A6851C972DA2DCE57F95CCB1FEB8577742A59349E1EC` verified
before and after.

## Files added

- `tools/renovation_architecture/package_candidates.py` —
  `build_package_candidates(standalone_result, projection, catalog,
  estimate_id)`: validates the frozen Session 3 result first
  (`StandaloneResultInvalid` dependency failure otherwise), builds one
  throwaway mutable adapter record per (ACTIVE work item × contributing
  condition), synthesizes room surrogates / estimate units from the
  condition identity audit trail, and calls
  `rehab_packages.infer_package_candidates` WHOLESALE — so catalog
  affinities, the scene-mismatch guard, ambient-support demotion, the four
  emit lanes, strength thresholds, tier resolvers, and tier escalation run
  with byte-identical legacy semantics and zero v4 changes. Emitted legacy
  dicts collapse back to stable work-item IDs (issue → condition → work;
  driver precedence when one merged work item supplied both roles), are
  vocabulary-checked against the closed v4 sets
  (`PackageVocabularyDrift` otherwise), and get the deterministic cost
  floor: `low/high = max(escalated tier spec, Σ child standalone range)`
  with the unfloored tier spec retained for audit. The whole-home turnover
  aggregate is reimplemented natively (~40 lines; the legacy function takes
  legacy dict shapes): ≥ 2 distinct turnover rooms → one display-only
  candidate with NO children, `contributing_candidate_ids` carrying the
  lineage, range = Σ contributors' floored ranges.
- `tools/renovation_architecture/sol_review.py` — the bounded Sol layer:
  `SOL_SYSTEM_PROMPT` (judge grouping coherence only; never reassess
  condition truth; never emit work/prices/quantities/families),
  `build_listing_request` (ONE text-only listing request: candidates,
  immutable child-work snapshots, and separately labeled
  `objective_evidence` / `terra_verdict` / `deterministic_disposition`
  summaries per condition; no images, no catalog; strict closed response
  schema with a per-call candidate-id enum; request fingerprint =
  sha256_canonical over schema 4 + projection fingerprint + prompt/model/
  effort/token config + candidate payload + snapshot hashes),
  `call_sol_review` (`vlm_client.analyze_text_sync`, provider `openai`),
  `parse_listing_decisions` (post-parse authority: exactly one bounded
  decision per supplied candidate; closed fields; combine = undirected
  edges among approved candidates only; split = ≥2 nonempty groups exactly
  partitioning the candidate's children, ordering normalized without
  membership changes; no candidate in both treatments; display-only in
  neither), and `run_package_review` (zero candidates → no provider call;
  else checkpoint replay or fresh call with the usage delta settled BEFORE
  parsing; assembles and self-validates the `package_review_complete`
  result with `SolResultInvalid` on failure).
- `tests/test_renovation_architecture_package_candidates.py` (13 tests) and
  `tests/test_renovation_architecture_sol.py` (31 tests) — see Tests below.
- `docs/HANDOFF_renovation_architecture_session_4.md` (this file).

## Files changed

- `tools/renovation_architecture/contracts.py` — schema v4 (contracts AND
  envelope), new state `package_review_complete`, two new policy versions
  (`package_candidates_v1`, `sol_package_review_v1`) → 9-key
  `POLICY_VERSIONS`, `SOL_REVIEW_REASONING_EFFORT = "medium"`, closed
  package vocabularies pinned by test against `tools/rehab_packages.py`
  (`PACKAGE_TYPES`/`_CATEGORIES`/`_LEVELS`/`_ROOMS`/`_STRENGTHS` (emitted
  set only)/`_TREATMENTS`) plus the fixed whole-home identity tokens.
  **PackageCandidate v4** (22 fields): type/category/level, room +
  estimate_unit_id identity, child/driver/support work-item ID tuples,
  strength, pricing profile/tier, absorption_scope mapping, treatment,
  unfloored + floored ranges with `cost_floor_applied`, `display_only`,
  `contributing_candidate_ids`. **PackageDecision v4**: + `sol_call_id`,
  `request_fingerprint`, `provider` (closed output otherwise unchanged).
  New `SolCall` (token telemetry, `usage_source`, NO budget field — no
  approved Sol budget), `SolListingUsage`, and `PackageReviewSnapshots`
  (three sha256 section fingerprints).
- `tools/renovation_architecture/ids.py` — `make_package_candidate_id`
  recipe: `room_key` → `estimate_unit_id` (the legacy bucket key is
  (unit, type); a room label would collide two bathrooms; nothing pinned
  the old recipe); new `make_sol_call_id` (`sc1`).
- `tools/renovation_architecture/checkpoints.py` — listing-level Sol
  checkpoint (`sol_package_review_v1`, single `sol_listing_review.json`
  beside the Terra unit files; fingerprint + exact candidate coverage;
  corrupt/stale ⇒ silent fresh call).
- `tools/renovation_architecture/validators.py` — `sc1` id pattern; v4
  candidate/decision field sets; `_validate_package_candidate` v4
  (vocabularies, driver ∪ support == children & disjoint, floored ≥
  unfloored with flag consistency, display-only ⇔ empty children +
  nonempty contributors + whole-home identity pins; non-display ⇒ room
  level, nonempty children, empty contributors); `_validate_package_decision`
  v4 (Sol provenance, per-record combine/split shape + normalization +
  both-treatments ban); new `_validate_sol_call` / `_validate_sol_listing_usage`
  / `_validate_package_review_snapshots`; public
  **`package_review_snapshot_hashes`** (single definition shared with the
  producer); new **`validate_package_review_result`** (the Session 4 gate:
  frozen S3 gate on the standalone subset, ACTIVE-only children,
  recomputed floor arithmetic, whole-home contributor/sum rules,
  candidate↔decision bijection, combine/split semantics, decision→call
  provenance, exactly-one-listing-call coverage, call→listing usage
  reconciliation, and recomputed snapshot fingerprints — the executable
  "Sol changed nothing" check); `validate_envelope` gains the
  `package_review_complete` branch; `validate_complete_result` untouched
  except that the shared per-record validators now enforce the v4 shapes.
- `tools/renovation_architecture/runtime.py` — runtime carries `sol_model`,
  `sol_max_output_tokens`, and the raw `catalog` (the candidate builder
  feeds it to the legacy primitives; pinned by the same catalog_sha256);
  shadow init requires valid Sol routing (same fail-closed posture and
  messages as Terra); `build_shadow_envelope` chains review → standalone →
  candidates → Sol review; success state `package_review_complete`;
  failures ride the existing `classify_failure` path (PassExecutionError's
  own pass/stage/provider/model win inside it, so Sol failures classify as
  Sol without code changes).
- `tools/pipeline_config.py` — `resolve_renovation_sol_model` (falls back
  to OPENAI_MODEL) + `resolve_renovation_sol_max_output_tokens` (default
  8192, invalid ⇒ raise at import) → `RENOVATION_SOL_MODEL`,
  `RENOVATION_SOL_MAX_OUTPUT_TOKENS` (exact Terra pattern).
- `tools/analyzer_server.py` / `tools/analyzer_cli.py` — pass the two Sol
  settings into init; the CLI re-runs the pure resolvers in its env bridge.
- `tools/artifact_writers.py` — last-resort envelope literal bumped to
  schema 4 (one line).
- `tools/renovation_architecture/__init__.py` — exports
  `validate_package_review_result`.
- Tests updated: `test_renovation_architecture_contracts.py` (shared
  `_candidate`/`_decision`/`_sol_call` v4 builders, `_complete_result` →
  v4 candidate/decision + retotaled fixture, new
  `_package_review_result`/`_package_review_envelope` helpers + 14-test
  `TestPackageReviewEnvelope` incl. the 9-key POLICY_VERSIONS pin and the
  rehab_packages vocabulary pins, id-recipe/sol-call id tests),
  `test_renovation_architecture_runtime.py` (Sol resolver tests, Sol init
  gates, raw-catalog pin, envelope/writer-seam pins →
  `package_review_complete` + empty package layer),
  `test_renovation_architecture_terra.py` (harness init gains Sol routing;
  end-to-end envelope → `package_review_complete` with an empty package
  layer for the affinity-less synthetic item).
  `test_renovation_architecture_conditions.py` / `_catalog.py` /
  `_disposition.py` / `_usage_guard.py` / `_work_items.py` and
  `tests/test_artifact_writers.py` needed no edits.

## Decisions and deviations

- **Wholesale reuse of `infer_package_candidates`** (plan decision 1):
  throwaway duck-typed adapters rather than re-implemented emit gates. The
  legacy entrypoint's only side effect (`_suppress_blocked_opportunity_drivers`
  mutating weak opportunity candidates) lands on the adapters; WorkItems
  are never touched (pinned by test: input result deep-equal after a
  suppressed bucket).
- **Adapter photo_keys are the deduped `representative_photo_keys`**, so
  the legacy multi-photo opportunity corroboration counts distinct views,
  never duplicate frames — executable form of the duplicate-evidence
  invariant, and stricter than v4's raw photo lists (Session 6 delta).
- **Deviation from the approved plan text**: cross-candidate child
  disjointness is NOT enforced by the Session 4 gate. A collapse-policy
  work item (per_property/per_area) can legitimately contribute to two
  room candidates; "absorbed at most once" is an APPROVED-absorption
  invariant and stays exactly where it already lives — the frozen
  Session 5 `validate_complete_result` gate. Documented in the S4 gate
  docstring.
- **Whole-home = contributor refs only** (Steven's explicit decision via
  plan review): empty `child_work_item_ids`, lineage through
  `contributing_candidate_ids`, combine/split participation banned. The
  frozen Session 5 disjoint-absorption invariant is structurally untouched
  (no children ⇒ can never absorb). Display-only ⇔ whole-home is
  validator-pinned for this schema; Session 5+ may loosen with a bump.
- **Cost-floor composition**: tier escalation runs first, inside the legacy
  entrypoint, against catalog base costs over distinct billable drivers
  (unchanged legacy semantics); the Session 4 floor then applies against
  the children's actual standalone dollars at candidate build. Both the
  unfloored (escalated tier) and floored ranges are contract fields, and
  the gate recomputes the floor exactly.
- **Sol telemetry has no daily budget guard** (no approved Sol budget):
  `SolCall` carries the 4 token fields and `usage_source` but no
  `budget_debited_tokens`, and there is no ledger/reservation. The usage
  delta is settled into the record before parsing (Terra's ordering).
  `_validate_sol_call` deliberately does not require
  `total == input + output` (provider totals may include reasoning tokens
  — same posture as `_check_terra_tokens`).
- **One listing-level Sol call** is a validator-enforced pin of this
  session's shape (`exactly one ... when candidates exist`); zero
  candidates short-circuit to an empty, still-valid review with no
  provider call and no checkpoint.
- **`classify_failure` kwargs unchanged** in `build_shadow_envelope`: a
  `PassExecutionError`'s own pass/stage/provider/model already win inside
  the classifier, so Sol failures surface as Sol without new plumbing.
- **Runtime stores the raw catalog** (shadow only): `infer_package_candidates`
  needs the catalog dict itself (`package_affinity` blocks, `cost.base_*`).
  The projection remains the versioned contract; both come from the one
  file pinned by `catalog_sha256`, and the projection fingerprint is
  UNCHANGED this session (estimate IDs are stable).

## Commands run and results

All on 2026-08-15, all green:

```
.venv\Scripts\python.exe -m pytest tests\test_renovation_architecture_package_candidates.py tests\test_renovation_architecture_sol.py tests\test_renovation_architecture_contracts.py tests\test_renovation_architecture_catalog.py tests\test_renovation_architecture_runtime.py tests\test_renovation_architecture_conditions.py tests\test_renovation_architecture_terra.py tests\test_renovation_architecture_disposition.py tests\test_renovation_architecture_usage_guard.py tests\test_renovation_architecture_work_items.py -q --basetemp=C:\tmp\rv_pytest_session4_focused_20260815
  -> 353 passed, 4.08s

.venv\Scripts\python.exe -m pytest tests\test_cost_factors.py tests\test_estimate_units.py tests\test_renovation_estimate.py tests\test_renovation_estimate_v4.py tests\test_rehab_packages.py tests\test_artifact_writers.py tests\test_catalog_validation.py -q --basetemp=C:\tmp\rv_pytest_session4_regression_20260815
  -> 537 passed, 1 skipped, 7.78s   (identical to the Session 3 baseline)

.venv\Scripts\python.exe -m pytest tests -q --basetemp=C:\tmp\rv_pytest_session4_full_20260815
  -> 2192 passed, 6 skipped, 22.42s   (Session 3 baseline: 2128 passed, 6 skipped; +64 new tests)
```

`git diff --check` clean (pre-existing CRLF warnings only); `git status`
shows exactly the 13 intended tracked modifications plus the 4 new files —
no unrelated or staged user changes touched; `Get-FileHash
tools\issue_catalog_kind_v2.json` reproduces
`5BE11ABB6DAD9921C6F1A6851C972DA2DCE57F95CCB1FEB8577742A59349E1EC`. All
warnings in the runs are pre-existing (utcnow deprecation, google-genai).
Fake providers only; no live Terra or Sol call was made at any point.

## Session 4 acceptance gates

- **Every package child is an existing ACTIVE accepted work-item ID** —
  builder constructs children only from active-work lineage; the gate
  re-checks membership (`not an ACTIVE work item`); suppressed dedup
  sources are pinned absent by test.
- **Condition and work truth unchanged by Sol** — executable snapshot
  fingerprints (condition sections / work sections / candidates) recorded
  before the Sol call and recomputed by the gate; a one-string tamper
  fails validation. The frozen S3 gate additionally re-validates the
  standalone subset verbatim inside every S4 result.
- **Rejected or uncertain packages preserve all child work as standalone**
  — structural this session: standalone work items are never modified by
  the package layer (deep-equality pinned), decisions live in a separate
  record, and application is deferred to the Session 5 ledger whose frozen
  gate already enforces the standalone fallback.
- **Combine/split cannot alter actions, quantities, prices, or total child
  coverage** — the closed decision contract has no such fields (boundary
  validation), split must exactly partition the existing children with
  ordering-only normalization, and combine is reference-only among
  approved candidates.
- **Sol cannot create unsupplied work or package families** — per-call
  candidate-id enum in the strict schema, post-parse rejection of unknown
  ids/fields, vocabulary-pinned candidate families produced only by the
  deterministic builder.
- **Provider failure is distinguishable from `uncertain`** — provider/
  timeout/JSON/schema/contract failures raise typed `PassExecutionError`s
  that become a private `failed` envelope (category `provider`/`parse`/...),
  while `uncertain` is a bounded decision in a valid
  `package_review_complete` result; both directions pinned by test,
  including the end-to-end Sol-failure envelope that leaves the completed
  Terra checkpoint on disk and replays it at zero cost on retry.
- **Current mode and v4 untouched** — writer-seam tests re-run green;
  regression suites byte-match the Session 3 baseline; catalog
  byte-identical; full suite green.

## Risks / notes for the reviewer

- **Schema v4 invalidates every pre-v4 Terra request fingerprint** (the
  fingerprint's first input is the contracts schema version), so the next
  shadow run re-reviews conditions fresh; the existing daily usage guard
  controls the spend. Estimate IDs are stable — the projection (and its
  fingerprint) did not change this session.
- **v4's `_extract_package_only_candidates` evidence lane has no
  equivalent**: candidates build from accepted work only, so
  no_action-routed catalog concepts no longer corroborate packages
  (intentional per the plan; Session 6 canary reviews the delta).
- **Multi-photo opportunity corroboration is stricter than v4** (deduped
  representatives vs raw photo lists) — flagged for the Session 6 delta
  review alongside the Session 3 pricing deltas.
- **Candidates may share a child across rooms** (collapse-policy work).
  Session 5 must resolve at-most-once absorption deterministically — the
  legacy `_absorption_priority_key` ordering is the natural reuse.
- **Sol has checkpoints and telemetry but no daily budget guard**; if a
  Sol budget is approved later, the Terra `usage_guard` pattern drops in
  at the same seam (reserve → call → settle).

## Session 5 prerequisites

- The input contract is `validate_package_review_result` — treat it as
  frozen; Session 5 adds its own state/keys additively (with a schema bump
  if fields change). `validate_complete_result` stays the Session 5 gate;
  note it still carries neither Terra nor Sol usage keys — reconciling the
  usage blocks into the complete result is Session 5's call.
- Apply decisions from `result["package_decisions"]` only: approve ⇒ the
  candidate may absorb its children (each child once, ledger-enforced);
  reject/uncertain ⇒ children keep their exact standalone allowances.
  Combine/split application semantics (which candidate identity survives a
  combine; how split groups become candidates) are Session 5 design work —
  Session 4 only validates their shape.
- The display-only whole-home aggregate has no children and must never
  absorb or count in totals; it is presentation/audit data for the ledger.
- Shared children across candidates require a deterministic absorption
  priority before ledger application (see risks).
- New reconciliation policies extend `POLICY_VERSIONS` (validators pin
  exact key equality — the intended friction), and the suppression-reason
  vocabulary (`dedup_collision` only, today) extends if the ledger needs
  more lanes.
