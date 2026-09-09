# Session 7 brief — bounded catalog policy checkpoint

Date: 2026-09-08. Status: charter, approved as a plan by Steven on 2026-09-08 (plan review in chat, three
corrections incorporated below). It is not an approved semantic diff and not an execution authorization for
any catalog change or provider call. It replaces the "Session 7" prose in `HANDOFF_SESSION_6.md`
("Required next task"), which described a narrower task (land CAP-022 and re-validate).

## Mission

Produce one accepted, stable catalog checkpoint and a reconciled Pass 2d worklist. Sequence: catalog decisions
→ catalog implementation and targeted validation → separate Pass 2d improvement → combined whole-property
backend acceptance. RenoIntel stays offline. Out of scope: frontend adoption, pricing, package-architecture
redesign, Stage B, and unrelated historical backlog (see `docs/REPORT_deferred_work_inventory_20260907.md`).

The outcome is reached when every in-scope finding has an approved validated change, an explicitly accepted
limitation, or an evidence-dependent deferral with a concrete reopening trigger.

## Hard prerequisites (verified 2026-09-08)

| Input | Identity | State |
|---|---|---|
| Baseline arm | main checkout, `9afe0fa`, branch `terra_factorized_verifier`, tracked-clean | catalog `51bf7e26…aa54`, decisions `47614d82…03e8`, projection fingerprint `6baacaf6…9308` (recomputed) |
| Candidate arm | worktree `C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4`, `6e67eaa`, branch `catalog_audit_session4`, tracked-clean, no `.venv` | catalog `43d8147e…7acf`, decisions `2e49a82f…9c76`, fingerprint `d09c1226…db79` (recomputed) |
| Ruled live bundle | `reports/catalog_audit_live_validation.json` `ee1bcf62…4f5c` | `status: ruled`, `final_recommendation: candidate_rejected`, S6-1 fix first, S5-1 closed |
| Approvals manifest (CAP-007) | `reports/catalog_audit_approvals.json` `a725c89a…c871` | the only authorization to date; pinned by the register, the Tier 1/2 harness and `tests/test_catalog_audit_validation.py:40`; **never edited** |
| CAP-022 proposal | `PROPOSAL_CAP-022_trim_subject_gate.md` `7e1a625d…1d55`; lever-eval files under `artifacts_canary/catalog_audit_session6_20260907/provenance/cap022_lever_eval/` | hashes match the bundle's `rulings.s6_1.offline_lever_evaluation.provenance` |
| Source manifest | `reports/catalog_checkpoint_source_manifest.json` | Phase 0 output; every document, report, evaluation artifact and code file this checkpoint reads, with sha256 and git blobs for both arms |

Both arms stamp version `3.2`; identify them by catalog sha and projection fingerprint only.

## Standing rules (supersede any conflicting historical instruction)

1. **CAP-022 alone is not the trim policy.** A `require_any` subject gate fixes which observations can reach
   `dated_interior_trim`; it leaves the claim "plain, thin, or builder-grade trim package" billable for every
   trim-naming bullet, which ruling 3 says must not bill by itself. "Preservation" of the item means subject
   correctness **and** warranted-work eligibility together; it is compared against no-action and retirement
   with disclosed losses. Terra visibility support cannot supply the eligibility judgment.
2. **Resolution and billability are separate axes.** Benchmark gold records that an observation resolves to
   an item; it says nothing about whether the item should bill. Preserve historical gold as it is. Reachability
   changes that make gold unreachable are a disclosed cost (versioned re-freeze with lineage), never a reason
   to pick or avoid a mechanism, and never a reason to suppress an item's legitimate dated work.
3. **No blanket score tolerance.** Any retrieval change that crosses a shortcut boundary (score floor 0.72,
   margin 0.03) or changes a consequential outcome (first candidate, owner, billable-item membership of the
   top-8, route, disposition, dollars) is assessed on its merits regardless of magnitude; the kitchen row
   changed shortcut behaviour on about 0.00006. Score-only differences that leave every list, boundary and
   outcome unchanged are recorded, not accepted by threshold.
4. Superseded: "only choose CAP-022 terms"; trim-naming plain-grade billing as accepted migration; CAP-022
   deferral restoring publication permission; automatic reuse of the Session 6 baseline arm; requiring exactly
   699 filtered rows; using only the 66 window-treatment cases; handing successor benchmark tests to another
   task; treating Session 6 as Catalog 3.2 whole-property acceptance. A generated `publishable` field is
   structural metadata, not publication approval.
5. Unchanged from the program: no hand-edit of generated artifacts; no v1 (`tools/issue_catalog.json`)
   authoring to dodge v2 support; no runtime alias for a removed id; no edit to a frozen artifact; budgets
   null until Steven fills them; no provider call without fresh explicit authorization; Stage B unauthorized;
   same-prompt Terra controls for any verdict-scored check; matched gold = evidence of the observation it
   states, billability separate.

## Facts that shape the work (verified 2026-09-08)

- **Both validation harnesses are pinned to CAP-007.** `scripts/catalog_audit_validation.py:41-48` hardcodes
  `CANDIDATE_COMMIT`, `APPROVALS_SHA256`, `APPROVED_PROPOSAL_ID` and the three window-treatment ids;
  `scripts/catalog_audit_live_validation.py:49-55` likewise. Neither runs unchanged on a new candidate or a
  new approvals file. Parameterizing them by manifest, with the Session 5/6 pins kept as the historical
  default, is tooling work for Phase 2.
- **Generator gap is broader than CAP-022.** Five of the nine ruling-3 families are carryovers (trim, doors,
  vanity light, ceiling fan, paint); only cabinets, appliances and landscaping are split successors where
  `require_any`/`route_override` are native. `WORDING_OVERRIDE_FIELDS` (`scripts/migrate_catalog_kind_v2.py:83`)
  hard-fails on any other override key; a top-level successor key is silently dropped. The renderer's
  `classify_op` reads the **frozen** evidence bundle's `authoring_surface`, so a generator-only patch leaves
  the proposal contract classifying the op as `gap`; the closure needs a versioned current-surface record.
- **Retirement is native** (`retired`) but removes the id: same stale-id hard-fail cutover as CAP-007.
  `dated_interior_trim` owns 108 of the 1,969 corpus rows (fan 54, cabinets 52, appliances 41, paint 40,
  doors 36, vanity 15, landscaping 4).
- **Claim-text edits change the projection fingerprint** (CCF-8), invalidating every stored Terra checkpoint
  carrying that item; the affected-artifact inventory (97 across six roots is the floor) is regenerated from
  the final diff per family.
- **Benchmark gold on the families:** `cases_holdout.json` `res-mod-006` → `dated_interior_doors`
  ("Interior doors are hollow-core six-panel units."), `res-cab-005` → `cabinets_dated_style` ("… builder-grade
  flat-panel units that read dated but are intact."); `cases_dev.json` `res-app-003`, `res-cab-003`,
  `res-pnt-004`, `res-lnd-002`. No gold on trim. Rule 2 applies.
- **Successor benchmark cases already exist as drafts** in the live manifest (`benchmark_gold_drafts`:
  `res-mod-cap007-blinds`, `res-mod-cap007-fabric`, target `cases_holdout.json`); they unblock
  `test_every_split_successor_is_gold_somewhere`.
- **"Six undetected blinds conditions" is a condition-level rollup**
  (`recommendation.approval_answers.A-4_conditions_undetected`); the ten unresolved rows are itemized in
  `per_case` by `case_id`; derive the six through the crosswalk. Five of the ten were already `excluded` at
  baseline (Phase 1 count; the packet lists all ten).
- Housekeeping done in Phase 0: register §1 row re-pointed (S4-3), `00_OVERALL_CONTEXT.md` §3 count note
  (S4-4), README row for this brief. `LM_STUDIO_URL` is re-pinned in the next manifest (S6-6), not in the old.

## Phase 1 — decision packet (investigation only; no semantic edit, no provider call)

Outputs: `docs/DECISION_PACKET_catalog_policy_checkpoint.md` and
`reports/catalog_policy_checkpoint_packet.json` (hash-pinned, citing `file:line` or hashed artifacts only).

1. **Nine-family matrix**, one row each for `dated_interior_trim`, `cabinets_dated_style`,
   `appliances_dated_or_basic`, `dated_bathroom_vanity_light`, `landscaping_enhancement_opportunity`,
   `older_ceiling_fan_style`, `dated_interior_doors`, `dated_window_valance_or_curtains`,
   `paint_refresh_recommended`: subject; visible state; warranted work under ruling 3; current claim state,
   route and authoring kind (carryover/split); candidate levers with authorability today; **"correct
   resolution target?"** and **"bills only warranted work?"** as separate columns; historical-gold
   reachability cost; corpus rows owned; preserved successes; losses; substitution risk onto neighbours;
   uncertainty. Sources: `kind_v2_decisions.json`, the generated catalog, `catalog_audit_replay_corpus.json`,
   `catalog_audit_redraft.json` `cap008`, `catalog_audit_live_check_results.json`,
   `catalog_audit_renewed_review.json`, the benchmark slices, the pinned photo review.
2. **Trim, three options with loss ledgers.** (a) Preservation = CAP-022 subject gate **plus** an eligibility
   mechanism under which plain/builder-grade presence stops billing while evidenced dated or missing trim keeps
   billing, naming the instrument that would measure it and the contrary evidence to overcome (both
   live-checked wordings failed; Terra supports "plain trim" whenever asked; the single-condition harness
   mismatched production on two of three control cards). (b) `no_action` on the item. (c) Retirement.
   For each: genuine dated/missing-trim losses, substitution onto windows/doors/decor/paneling, cutover size.
   Re-read the four lever-eval losses and the S6-1 rows against photos. CAP-008 is resolved in this row;
   CAP-022 alone is recorded as the subject half only.
3. **Six deferred CAPs** (008, 010, 013, 014, 016, 018): disposition and reopening trigger per the plan
   table; CAP-013 re-reads exactly `rc_ceeb17f8e242`, `rc_7057c3c171e5`, `rc_201ccb0c2313`,
   `rc_57c6faf79c7e` (photo/label conflicts, same-property dependence; prior no-schema-work ruling preserved);
   CAP-014 waits on the CCF-14 precedence ruling (recommendation: an explicit human annotation is evidence of
   the observation it states; automated matching stays fallible; billability separate). Trim/paint/
   baseboard-wear overlap and curtain hardware (A-5 preserved pending an explicit replacement ruling) are rows
   in these same items, not separate projects.
4. **Blinds visibility.** The ten unresolved rows with baseline disposition; the six conditions via the
   crosswalk; accept-the-omission versus preserve-visibility as an explicit product choice with a
   recommendation grounded in the rows. The only preservation lever is the successor's `support_any`, which
   the redraft rejected deliberately. Until Steven accepts omissions, the visibility requirement stands.
5. **Two one-pass corpus checks.** Railing finish wear: at most two strong examples on different properties,
   answered from `frozen.resolved_item_id is None` in the corpus (no model call); the 42 bullets are not
   assumed misses. Wallpaper (CCF-13): trace the recorded example through v5 conditions, work-item collapse,
   package membership and money with the provider-free `replay_renovation_architecture`. Defer with a precise
   trigger if evidence is insufficient.
6. **Tranches and cohort.** Tranche 1 = trim plus any family with decisive evidence; tranche 2 = families whose
   default is "no change, recorded with trigger". Only tranche 1 enters the validation cohort. Show the
   affected-artifact inventory delta per family and a cohort/cost sketch with budgets null.
7. **Support request.** One generator change (rename `WORDING_OVERRIDE_FIELDS` → `CARRYOVER_OVERRIDE_FIELDS`,
   admitting `require_any` and, if any tranche-1 option needs it, `route_override`; economic and unknown keys
   still fail), the renderer's current-surface classification with a versioned surface record, and the
   harness parameterization. Presented for separate authorization per `04_HUMAN_REVIEW_GATE.md`.
8. **Approval manifest v2 template**: `reports/catalog_audit_approvals_v2.json`, schema
   `catalog-audit-approvals-v2`, `approved_against` pinning `6e67eaa`, the post-support generator blob, the
   CAP-022 proposal hash, the packet hash and the source manifest hash; one disposition per in-scope id
   (CAP-008/010/013/014/016/018/022 and any new id), `approved_diff` as `apply_ops` ops.
9. Only product questions with concrete evidence and tradeoffs go to Steven; everything else carries a
   recommendation. Planning approval is not op approval.

## Phase 2 — implementation (only after exact-op approval)

New commits on `6e67eaa` in the candidate worktree, in this order: (1) generator change with tests in
`tests/test_catalog_kind_v2.py` (verbatim landing on a carryover; economic and unknown keys still fail);
(2) renderer current-surface path and versioned surface record with tests; (3) manifest-parameterized
harnesses with historical pins as defaults, tests updated; (4) approved ops applied with `apply_ops` to
`tools/catalog_migrations/kind_v2_decisions.json`, regenerated twice byte-identical, generated diff proven
equal to the ops, validator zero errors, one focused test file per approved op (pattern
`tests/test_catalog_cap007_window_treatment_split.py`); (5) the two drafted successor cases, plus any gold
the approved ops make unreachable, in a new frozen slice with lineage; old slices and results untouched.
Preserve CAP-007's approved successor semantics except for separately approved changes.

## Phase 3 — validation and cutover plan

Freeze the cohort from the final diff before any model run (affected rows, successful uses, negative
controls, neighbouring billable items, the four unpinned parent-owned rows, the CAP-013 cards, accepted
limitations, shortcut-boundary rows). Tier 1/2 through the parameterized harness with a real-POST sidecar
probe and one fresh embedding cache shared by both arms; rule 3 governs every difference. New live manifest
(`catalog_audit_live_experiment_manifest_v2.json`) with budgets null and `LM_STUDIO_URL` re-pinned; Stage A
only when authorized; same-prompt Terra replica as the control; reuse Session 6 units only where the request
fingerprint matches. Affected-artifact inventory from the final diff; cutover plan with new run namespaces
and explicit checkpoint invalidation; no production mutation. Handoff `HANDOFF_SESSION_7.md`; register §9
block.

## Acceptance

- Treatment-to-trim leakage and plain-grade trim billing both satisfy the approved policy; no blanket billing
  exemption for trim-naming bullets.
- All nine families dispositioned; all six deferred CAPs dispositioned with triggers.
- Legitimate-work losses and the six blinds omissions resolved or explicitly accepted; S5-1, no-action
  blinds and the accepted improvised-covering outcome honoured.
- Generated outputs, authoring contracts, successor coverage and regression checks pass; no new
  wrong-neighbour billing hidden in aggregates.
- The candidate has a catalog hash, projection fingerprint, evidence/validation manifest,
  accepted-limitations record and affected-artifact cutover plan.
- The Pass 2d worklist references current successor ids and separates upstream kind exclusion, candidate
  absence, incorrect selection, shortcut behaviour and verification/billability issues; the 26 cases are
  reclassified as needed, not promised fixes.

New findings block only if they contradict this candidate's approved requirements; otherwise park them with
a trigger. No Pass 2d prompt, model or threshold tuning here. Combined whole-property acceptance and fresh
Sol package decisions (the Catalog 3.2 gap) wait until catalog and Pass 2d stabilize.

## Known by-design failures (classify, do not fix)

S4-5 audit-renderer tests and both renderers' `--check` on the candidate arm; `test_full_run_against_frozen_inputs`
in any worktree; S4-2 stale-id offline recompute until the replay; `build_catalog_audit_evidence.py`
`FROZEN_SHA256` failing closed on any candidate arm.

## Phase 0 and Phase 1 outputs (2026-09-08)

| Artifact | sha256 | Contract |
|---|---|---|
| `docs/DECISION_PACKET_catalog_policy_checkpoint.md` | `492ef7de44033e8a…` | the readable packet; section 9 lists the six decisions requested from Steven |
| `reports/catalog_policy_checkpoint_packet.json` | `a7416ac6f4b769a73eabf7bb5d42148420953470d754ad0db4a7186524ae238c` | machine record: family matrix, trim options and lever evaluation, blinds rows, railing and wallpaper checks, deferred CAPs, affected artifacts, Pass 2d worklist axes, dispositions, approval-manifest v2 template |
| `reports/catalog_checkpoint_source_manifest.json` | regenerated last; self-describing (`generated_at_utc`) | every input this checkpoint read, with sha256 and git blobs for both arms, recomputed projection fingerprints, and the latest rulings |
| `artifacts_canary/catalog_checkpoint_20260908/provenance/` | per-file hashes in the packet's `provenance_sha256` | extraction scripts and the trim eligibility lever evaluation (`trim_eligibility_lever_eval/`: script, report, self-check) |

Phase 1 findings that change the plan: the presence-stem `deny_any` eligibility lever migrates plain-trim
bullets onto billable wood-paneling and interior-door items (20 of 54 removed rows) and is rejected; the
recommended trim mechanism is `route_override no_action` (needs carryover `route_override` generator support),
with the CAP-007-style split as the preservation alternative (native for the item, 97-artifact cutover, a
disclosed residual of plain trim billed as dated). Two families are already non-billing through the electrical
product quarantine. The wallpaper double charge is deterministic and provable from one artifact. Nothing is
approved; Phase 2 starts only after Steven's decisions and the support authorization.

## Review pass (2026-09-08, same day)

Steven reviewed the two CAP-013 trim cards against their photos (`REVIEW_trim_photos_steven_20260908.md`):
kitchen rc_ceeb17f8e242 is missing trim and real work; bathroom rc_57c6faf79c7e is not confirmed and the
bathroom must be costed without it. His standing intent: trim is primarily context for broader room work.
The packet's section 2a adds the deterministic with/without replays (two units and the 17 replayable pinned
artifacts) and the three-way classification (direct repair work / supporting room context / unsupported
observation), and two decisions (7: supporting-context mechanism for `no_action` conditions, which would be
new code; 8: representation of the kitchen's missing trim). Packet JSON is now `99a40d74…1637`. Nothing
approved; no semantic change; no provider call.
