# Pause note — catalog audit session 6, Part A (Stage A executed, evidence frozen)

Date: 2026-09-07

Session/task title: Cost-gated live validation (`07_SESSION_LIVE_VALIDATION.md`) — Stage A, Part A of two

**Status: Stage A COMPLETE on all 66 pinned cases in both arms. No stop condition fired. 575,091 of
1,200,000 authorized tokens spent. The evidence bundle is frozen at
`reports/catalog_audit_live_validation.json`, `status: awaiting_evaluation`.**

Part A ran the experiment and recorded what happened. **It applies no rubric and states no
recommendation.** That is Part B's work, deliberately given to a fresh session: the session that
spends the tokens should not also be the session that decides whether the spend justified the
candidate.

## What Part B must do

1. Read `00_OVERALL_CONTEXT.md`, `07_SESSION_LIVE_VALIDATION.md`, `HANDOFF_SESSION_5.md`, this note,
   the manifest, and `reports/catalog_audit_live_validation.json`.
2. Re-verify before trusting anything (list below).
3. Inspect the per-case evidence in both arms — verdicts *and rationales*, not just the counts.
4. Apply the Session 6 brief's rubric against the 6.4% Terra replica noise floor, and judge whether
   any of the brief's behavioural stop conditions would have halted a live run.
5. Write `docs/RESULT_catalog_audit_live_validation_20260907.md`,
   `docs/catalog_audit_program/HANDOFF_SESSION_6.md`, and a FOLLOW_UP_REGISTER §9 Session 6 block;
   then set the bundle's `status` to `evaluated` and append a `recommendation` block. Append only —
   do not alter a Part A row.
6. Make no provider call. Do not run Stage B; it is unauthorized and its go/no-go is Steven's.

## Re-verify these before trusting the bundle

| Check | Expected |
|---|---|
| bundle `reports/catalog_audit_live_validation.json` sha256 | `24bda3c3836d1ba331e45b16151c564b45bc992fc37d5fb71a1a6f9b0fcb6f53` |
| every `raw_artifacts.unit_records[].sha256` (82) and `property_records[].sha256` (44) | as listed in the bundle |
| baseline HEAD / candidate HEAD, both tracked-clean | `9afe0fa` / `6e67eaa`, clean (verified at end of Part A) |
| the 22 pinned artifacts and 63 pinned photos | unchanged (re-verified at end of Part A: 22/22, 63/63) |
| manifest content proof | restoring the null budget block reproduces `8530cca8…` (bundle `manifest.content_proof.ok`) |
| ledger spend ≤ authorization | 575,091 ≤ 1,200,000 |
| every executed case reconciles to the manifest | `aggregates.cases_pinned` 66 = `cases_executed` 66, `cases_not_executed` empty |

## Repository state

- Baseline arm: main checkout, `terra_factorized_verifier`, `9afe0fa5a0490a856abed507dccc4a02ed1de24c`.
  Tracked-clean at start and end.
- Candidate arm: worktree `C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4`,
  `catalog_audit_session4`, `6e67eaa83887cf4d218100dcd060c07d3bd30522`. Unchanged by this session;
  tracked-clean at start and end.
- Nothing committed. All outputs untracked, matching the program's posture.
- No `.checkpoints` directory was read or written; the run root contains none.

## Files created or changed

| Path | Status | Note |
|---|---|---|
| `reports/catalog_audit_live_experiment_manifest.json` | **budget block only** | new file sha `8721ea63…`; see the authorization below |
| `scripts/catalog_audit_live_validation.py` | new | the Stage A harness |
| `tests/test_catalog_audit_live_validation.py` | new, 31 tests, all pass | provider-free |
| `reports/catalog_audit_live_validation.json` | new | the frozen bundle, sha above |
| `artifacts_canary/catalog_audit_session6_20260907/` | new | raw run root: arm records, 82 unit records, 44 property records, 44 Pass 2d records, the shared Terra ledger, `verify*.json`, `compare.json`, `provenance/` |
| `docs/catalog_audit_program/PAUSE_SESSION_6_PART_A.md` | new | this file |

## The authorization

Steven authorized **1,200,000 Stage A tokens in chat on 2026-09-07**; Part A transcribed it into the
manifest's `budget` block (`authorized_by: "Steven (authorized in chat, 2026-09-07)"`). Stage B was
left `null` at his instruction.

Only the budget block was edited. Proof, re-runnable and recorded in
`artifacts_canary/catalog_audit_session6_20260907/provenance/authorization_proof.json`: restoring the
Session 5 null budget block and the Session 5 `manifest_sha256` reproduces the Session 5 file **byte
for byte** (`e1f3b7f1…a42f`), and the canonical hash of everything except the budget is unchanged
(`b21e5013ea009646f0aa46d4188d79139ff1f27ef41947cc9d6c4f8f1d0b9d7c`). An untouched copy of the
Session 5 manifest is at `provenance/manifest_session5_original.json`.

The manifest's own `status` still reads `awaiting_human_cost_authorization`. That was left alone
deliberately: only the budget block was edited, and a stale-but-cautious status fails safe for any
other reader.

## What ran

| | Baseline | Candidate |
|---|---:|---:|
| Properties | 22 | 22 |
| Pinned cases executed | 66 | 66 |
| Terra unit calls | 44 (43 fresh + 1 resumed) | 38 |
| Terra tokens | 306,650 | 268,441 |
| Stop condition | none | none |

Ledger total 575,091 of 1,200,000 authorized (48%). The candidate arm needed fewer unit calls because
ten of its cases resolved to no catalog item at all, so they formed no condition and joined no unit.

Pass 2d ran on the local model at 2.6–3.4 s/call, matching the 3.35 s/call recorded in the stored
production runs. All 132 Pass 2d calls completed; none was recorded as an error in either arm.

## Headline measurements (facts; Part B judges them)

**Baseline fidelity.** The baseline arm reproduced the frozen Pass 2d resolution on **65 of 66**
cases. The single divergence is `canary:redfin_80990371:…:photo_025.jpg:322eb2b7f52e83c3`, which the
frozen run resolved to the parent by the LLM path and the live run resolved to `dated_interior_trim`,
also by the LLM path. Pass 2d runs at temperature 0.2 with no seed, so the LLM path is not
deterministic; this is the measured drift of the instrument, not a catalog effect.

**Where the affected rows land in the candidate arm.** 36 reach the blinds successor, 11 reach the
fabric successor, 19 reach neither (10 of those resolve to no item at all; the rest land on
`dated_or_older_windows` ×3, `dated_interior_trim` ×3, `dated_overall_decor_style` ×1, plus the two
`exterior_door_paint_failure` hallucination controls, which are unchanged).

**Resolution path.** Baseline 29 shortcut / 37 LLM; candidate 32 shortcut / 34 LLM. 32 of the
candidate rows match Session 5's offline shortcut prediction.

**Terra.** 17 of 66 verdicts differ between the arms, against a 4.2 expected-flip count at the 6.4%
replica floor. Baseline 55 supported / 11 unsupported; candidate 49 supported / 6 unsupported /
1 cannot_assess / 10 with no condition to review.

**Disposition and route.** Baseline 54 accepted_for_work, 11 excluded, 1 no_action. Candidate 12
accepted_for_work, 36 no_action, 7 excluded, 1 inspection, 10 none. Route changed on 49 of 66.

**Billing** (standalone over the reviewed units, never a property headline). Baseline $4,521–22,613;
candidate $979–4,895. 39 candidate rows sit at zero dollars on a `no_action` route, which is the A-4
count the approval asked for.

**The two material findings both moved against their offline prediction.**

- `F-SHORTCUT-fee5271eaff08de5` — "The mini blinds feel dated compared with the updated kitchen."
  Session 5 predicted a deterministic shortcut onto the **billable** `outdated_kitchen_finishes` at
  margin 0.03004, four hundred-thousandths above the gate. **Live, the margin came in at 0.029982,
  just below the gate, so no shortcut fired**; the row reached the LLM and resolved to
  `window_blinds_basic_or_plain`, verdict supported, route `no_action`, zero dollars. The feared
  outcome did not occur in this run, and the row is confirmed to sit on the knife edge of the
  threshold.
- `F-REACH-oc1_93a2e3f3d299a7a1` — both observations again failed to reach the fabric successor, as
  Session 5 predicted. Live they landed on `dated_overall_decor_style` (supported, `excluded_generic`,
  no dollars) and `dated_or_older_windows` (cannot_assess, inspection). The approval's count of
  surviving billings is therefore still optimistic by one condition, and the live outcome for this
  condition is not billing.

**Controls, against their predeclared expectations** — the largest open question for Part B:

| Card | Expected | Baseline | Candidate |
|---|---|---|---|
| rc_7c5b9f9bb3aa | still_supported | supported / accepted_for_work | supported / **no_action** on the blinds successor |
| rc_429ba6851d09 | still_supported | supported / accepted_for_work | **no condition** (resolved to no item) |
| rc_64f996e3a886 | still_rejected | **supported** / no_action | unsupported / excluded |
| rc_a22a241b3bf0 | flips_to_supported | unsupported / excluded | supported / accepted_for_work on `dated_interior_trim` |
| rc_4f4b93c40ef3 | flips_to_supported | unsupported / excluded | **no condition** |
| rc_ffad088fa5fe | flips_to_supported | supported / accepted_for_work | supported / no_action on the blinds successor |
| rc_8afb476fa69f (×2) | unchanged | supported / accepted_for_work | supported / accepted_for_work |

Note rc_64f996e3a886, the reviewed **correct rejection**: the *baseline* arm returned `supported`
where the frozen run recorded a rejection. Its route is `no_action` either way, so nothing bills, but
Part B should read this as instrument noise before reading it as a candidate effect.

**Four rows landed on a third item** (neither the frozen owner, a successor, nor the offline
prediction): three on `dated_interior_trim` (supported, route `work`, ~$76–79 each) and one on
`dated_overall_decor_style` (supported, `excluded_generic`, no dollars).

## Deviations, each recorded and validated

Five are in the bundle's `deviations` block (purpose-built harness; Pass 2d re-run for pinned rows
only; four unpinned parent rows excluded from both arms; whole target units reviewed rather than
condition subsets; billing derived as a standalone estimate over reviewed units). Three more arose
during execution and are recorded here:

1. **Pass 2d transport address substituted.** The manifest pins `LM_STUDIO_URL`
   `http://169.254.83.107:1234`, a stale link-local address; the model server is now local to this
   machine. Part A used `http://127.0.0.1:1234` via an explicit `--lm-studio-url` flag, applied
   identically to both arms, and verified separately that the **model identity**
   (`unsloth/qwen3.6-27b@q6_k`) matches the manifest — that is the routing invariant, checked by
   `I26b`. The address is transport, not an experimental variable. This is the "minimal run-wrapper
   fix for an operational defect" the Session 6 brief permits.
2. **Embeddings sidecar restarted.** Mid-session the sidecar began returning
   `decode() failed: vk::Queue::submit: ErrorDeviceLost` on a real POST while `/health` still
   reported ok — the known false-healthy failure. The `verify` gate caught it and refused to make any
   provider call. The sidecar was restarted with `scripts/start-embeddings-server.ps1` and re-probed
   before the run resumed. No experiment input changed.
3. **Harness corrected before any measurement was kept.** The first smoke attempt ran while the GPU
   was saturated by an unrelated application; two Pass 2d calls failed on transport and the harness
   recorded them as cases that resolved to nothing, then exited zero. That would have manufactured a
   "reached no catalog item" finding out of a network failure. The harness now retries transport
   failures, refuses to cache them, and stops the run loudly instead. Those records were deleted and
   every case in the frozen bundle comes from a clean call.

## Unresolved questions and risks for Part B

1. **Ten candidate cases resolved to no catalog item.** These are legitimate Pass 2d outcomes (the
   model saw candidates and chose none), not errors, and each carries `pass_2d_status: unresolved`.
   Two of them are reviewed controls (rc_429ba6851d09, rc_4f4b93c40ef3), which is why those rows have
   no verdict. Whether "no item" is an acceptable landing place for a reviewed positive use is a
   judgement, not a measurement.
2. **17 verdict changes against a 4.2 noise expectation.** Above the floor, but the arms also differ
   in which *item* Terra was asked about, so this is not a clean replica comparison. Part B should
   separate "same claim, different verdict" from "different claim entirely".
3. **The baseline arm itself drifted on two rows** (one Pass 2d resolution, one control verdict).
   That is the instrument's own noise and bounds how tightly any candidate effect can be read.
4. **No repeats were run.** The predeclared repetition rule allowed up to ~12 replicas for
   non-decisive calls, contradicted controls and the two material findings. None were triggered by
   the mechanical stop conditions, and Part A deliberately did not exercise judgement about which
   borderline results deserved one. If Part B wants verdict-stability replicas, that is a new,
   separately budgeted request to Steven — 624,909 tokens remain unspent within the existing
   authorization, but Part A makes no claim on them.
5. **Stage A produced no Sol decisions, package candidates or v4/v5 totals**, so the Catalog 3.2
   acceptance thread still has no fresh data. Stage A reviews conditions only. The vehicle for that
   thread remains a Stage B baseline arm.

## Do not redo

- Do not re-run either arm to "confirm" it. Both are complete, hash-recorded and resumable; a re-run
  would spend tokens and, because Pass 2d and Terra are both non-deterministic, would not reproduce.
- Do not edit a Part A row in the bundle. Part B appends `recommendation` and flips `status`.
- Do not treat the ten unresolved candidate cases as harness failures; the harness stops loudly on a
  real failure and did not stop.
- Do not add a runtime alias for the deprecated parent, and do not edit any stored artifact — the
  four excluded rows are excluded by design and are listed in the bundle.
- Do not set `ISSUE_CATALOG_PATH`; under `observation_kind_v2` it is a hard configuration error.

## Suggested opening prompt for Part B

> Continue catalog-audit Session 6 from `docs/catalog_audit_program/PAUSE_SESSION_6_PART_A.md`. Part A
> ran Stage A on all 66 pinned cases in both arms and froze
> `reports/catalog_audit_live_validation.json` with `status: awaiting_evaluation`. Read the overall
> context, the Session 6 brief, `HANDOFF_SESSION_5.md`, the pause note and the bundle. Re-verify every
> hash and invariant the pause note lists before trusting the bundle. Make no provider call. Inspect
> the per-case evidence in both arms, including Terra's rationales, apply the rubric in
> `07_SESSION_LIVE_VALIDATION.md` against the 6.4% replica noise floor, and decide the recommendation
> with confidence. Then write `docs/RESULT_catalog_audit_live_validation_20260907.md`,
> `HANDOFF_SESSION_6.md` and the FOLLOW_UP_REGISTER §9 block. Do not run Stage B; leave its go/no-go
> to Steven.
