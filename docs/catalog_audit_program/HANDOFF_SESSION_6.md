# Handoff — catalog audit session 6

Date: 2026-09-07

Session/task title: Cost-gated live validation (`07_SESSION_LIVE_VALIDATION.md`) — Stage A on the
CAP-007 candidate, run in two parts: Part A (Opus 5) built and executed the harness and froze the
evidence; Part B (Fable 5.1) re-verified it, read it, and decided.

**Status: COMPLETE and RULED. Stage A plus a same-day repeat stage ran; Part B recommended
`publish_candidate_supported` on the standing priority and stated the strict reading. Steven ruled
(2026-09-07, in chat) fix first, so the program's final recommendation on candidate `6e67eaa` as it
stands is `candidate_rejected` pending the trim lever, drafted the same day as CAP-022
(`PROPOSAL_CAP-022_trim_subject_gate.md`) for his approval at the gate. S5-1 was ruled the same day:
the missed fabric condition is accepted, no wording change. Stage B not run; still unauthorized.
1,149,750 of 2,000,000 authorized tokens spent across both figures. Nothing committed, nothing
published, no catalog, prompt or route changed.**

## Outcome

| Exit criterion (brief) | Result |
|---|---|
| No run exceeded its authorization | Stage A 575,091 ≤ 1,200,000; repeat 572,308 ≤ 800,000; one shared ledger, per-call pre-check; 850,250 unspent of the combined 2,000,000 |
| Every executed case reconciles to the pinned manifest | 66 of 66 in both arms; join by `case_id`, crosswalk to the manifest's 3.1-era condition ids in the bundle |
| Baseline/candidate isolation demonstrated | separate checkouts at `9afe0fa` / `6e67eaa`; catalog sha and projection fingerprint verified at run time; code blobs identical; same photos, same prompts, same models; the only variable is the catalog |
| Recommendation follows the predeclared rubric | clause by clause in the RESULT doc; the bundle carries `recommendation` (Stage A only) and `recommendation_after_repeat` (final) |
| No implementation or publication change | both arms tracked-clean at start and end; no `tools/` edit; no `.checkpoints` touched |
| Handoff states whether an optional publication task is justified | **not on `6e67eaa`**: Steven ruled fix first, so publication follows CAP-022 through Session 7 (see *Required next task*) |

The approved change does what Ruling 5 intended: 26 of the 28 conditions the approval expected to
stop billing did stop, the fabric successor's tighter claim correctly rejects three overstated
billings, no reviewed real condition was lost, and the effect is far outside instrument variance.
Stage A alone could not tell whether the two conditions that migrated onto the billable
`dated_interior_trim` item were a candidate effect or instrument drift, so Part B returned
`inconclusive`. Steven then authorized a repeat the same day: Pass 2d five more times on every row
in both arms and one same-prompt Terra replica of every Stage A unit. The repeat settled it: the
migration is structural (18 of 396 candidate resolutions, the two rows 11 of 12, against 0 of 12 for
those rows under the baseline; no other billable item ever reached; the trim claim Terra-supported 4
of 4). It is real, small, confined to one item, and has a one-line lever outside this diff. Part B
recommends publication on the program's standing priority and states the strict reading
(`candidate_rejected` pending the trim lever) so Steven can choose it.

## Repository state

- Baseline arm: main checkout `C:/Users/Steven/PycharmProjects/realtorvision-backend`, branch
  `terra_factorized_verifier`, commit `9afe0fa5a0490a856abed507dccc4a02ed1de24c`. Tracked-clean at
  start and end.
- Candidate arm: worktree `C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4`, branch
  `catalog_audit_session4`, commit `6e67eaa83887cf4d218100dcd060c07d3bd30522`. Unchanged; tracked-clean
  at start and end.
- Nothing committed. All Session 6 outputs are untracked in the main checkout, matching the
  program's posture; whether to commit the program artifacts remains delegated to
  `docs/HANDOFF_git_repository_audit.md`.
- Pre-existing changes preserved: the 22 pinned artifacts and 63 pinned photos re-hashed unchanged at
  the end of Part A and again at the start of Part B; no production checkpoint store was read or
  written.

## Inputs verified

Part A's `verify` ran 37 checks before any call and again before each arm (`verify_baseline.json`,
`verify_candidate.json` in the run root); Part B re-verified the frozen bundle before reading it.

| Input | Expected | Result |
|---|---|---|
| Baseline / candidate HEAD | `9afe0fa` / `6e67eaa` | match, both clean |
| Catalog sha256 baseline / candidate | `51bf7e26…aa54` / `43d8147e…7acf` | match |
| Projection fingerprint baseline / candidate | `6baacaf6…9308` / `d09c1226…db79` | match (computed via `build_renovation_catalog_projection`) |
| Prompt versions, Terra and Sol module blobs, nine code blobs, both arms | manifest `prompt_identity`, `code_hashes` | match |
| Model map `89c04984…`; model names in `.env` | manifest `model_map` | match |
| `LM_STUDIO_URL` | `http://169.254.83.107:1234` | **stale link-local address, unreachable**; `http://127.0.0.1:1234` substituted for both arms, model identity `unsloth/qwen3.6-27b@q6_k` verified (see *Deviations*) |
| Embeddings sidecar, real POST | dim 1024, jina v5 | match (after one restart, see *Deviations*) |
| Manifest identity | Session 5 file `e1f3b7f1…`, canonical `8530cca8…` | match by restoration proof; authorized file sha `8721ea63…` |
| 22 artifacts, 63 photos | pinned sha256 | 22/22, 63/63 |
| Product-lane presence of all 66 rows with frozen owners; scene groups | — | 66/66 |
| The four unpinned parent-owned rows | exactly `98f32292…`, `cc67016e…`, `62681b05…`, `3e92a074…` | match |
| Budget | Stage A filled by a human; Stage B null | match |
| Part A bundle (Part B check) | `24bda3c3…6f53`, 172 raw-artifact hashes | match |

## Files created or changed

All untracked, all in the main checkout.

| Path | Status | sha256 |
|---|---|---|
| `reports/catalog_audit_live_experiment_manifest.json` | **budget block only** (Stage A 1,200,000 and repeat 800,000, both authorized by Steven in chat; restoring the null block still reproduces Session 5's `8530cca8…`) | `b34f75b308f86882a4f587db98d061143b6725e80ae6f1dcdbf781af024d04d5` |
| `scripts/catalog_audit_live_validation.py` | new — the Stage A harness, plus `replicate` and `compare-replica` for the repeat stage | `439235baee7f9731fbb6d85e0b523abc3bfe28a903c3181d5b0897f243f8abcb` |
| `tests/test_catalog_audit_live_validation.py` | new, 34 tests | `20a3a7d3b61b63734798a913a3df071c76af7625820e0c626d9d7ae58369e064` |
| `reports/catalog_audit_live_validation.json` | new — the bundle; Part A froze it (`24bda3c3…`), Part B appended `recommendation` (`2709d427…`), then `repeat` and `recommendation_after_repeat` (`5573ea34…`), then Steven's rulings (`rulings`, `final_recommendation: candidate_rejected`, status `ruled`); each step append-only and proven | `ee1bcf62ea6e37bd76e8f4b7f32612324736d34373de811cb340171249be4f5c` |
| `docs/RESULT_catalog_audit_live_validation_20260907.md` | new — the readable result, with the repeat stage and the "Rulings" section (ruling note at the top of "Recommendation") | `55f6405d8dab36f370433c4ac8d43a5e5177f1b9c0c3292143c519c261ab89e4` |
| `docs/catalog_audit_program/PAUSE_SESSION_6_PART_A.md` | new — Part A's pause note | `388989705d960aa66856a8d402dc9ea1852b96c247773f31e467d664b8ceb9a4` |
| `docs/catalog_audit_program/HANDOFF_SESSION_6.md` | new — this file | — |
| `docs/catalog_audit_program/FOLLOW_UP_REGISTER.md` | §9 Session 6 block (S6-1…S6-8), the repeat addendum (S6-1/S6-4 updated, S6-9, S6-10) and the rulings addendum (S5-1 closed, S6-1 ruled, S6-11, S6-12) | `17a4ea9ff988c87b1b49affcaf88910da283eab3075e04d7e092f8166ce55d1c` (was `b53c217b…`) |
| `docs/catalog_audit_program/PROPOSAL_CAP-022_trim_subject_gate.md` | new — the trim lever proposal (`require_any` subject gate on `dated_interior_trim`), drafted after Steven's fix-first ruling; awaiting his approval of the exact op | `7e1a625d7c6096cd03700d34176950f14078f9cc99eb64d91cf49fd88b7c1d55` |
| `artifacts_canary/catalog_audit_session6_20260907/provenance/cap022_lever_eval/` | new — the offline lever evaluation (sidecar only, no provider call): `lever_eval.py` `cf4c3e81…`, `lever_eval_report.json` `68d0291b…`, `lever_selfcheck.json` `8aa923ac…`, `lever_selfcheck.py` `3061c206…` | see bundle `rulings.s6_1.offline_lever_evaluation.provenance` |
| `artifacts_canary/catalog_audit_session6_20260907/` | new run root: two arm records, 82 unit records, 44 property records, 44 Pass 2d records, the shared Terra ledger, verify and compare outputs, `provenance/`; **plus the repeat**: 110 Pass 2d replica records per arm, 44 + 38 replica unit records under `units_replica/`, two replica records, `compare_replica.json` | arm records `d2291cc7…` / `2a389ede…`; replica records `95e799d6…` / `d68bf6ea…`; `compare_replica.json` `89288f79…` |

Every Part B edit to the bundle is append-only and was proven at write time (the rulings append likewise: stripping `rulings` and `final_recommendation` and restoring status `evaluated` reproduces `5573ea34…`): stripping `repeat` and
`recommendation_after_repeat` reproduces `2709d427…`; further stripping `recommendation` and restoring
`status: awaiting_evaluation` reproduces Part A's `24bda3c3…` byte for byte.

## Work completed

- Transcribed Steven's chat authorization into the manifest with a byte-level proof that nothing
  else moved.
- Built a harness that composes production functions only (Pass 2d resolver, condition projection,
  evidence, Terra unit request and fresh-call path, disposition policy, standalone pricing) with an
  explicit budget gate, a shared ledger, per-unit resume keyed on request fingerprint, and a
  fail-loud rule for transport failures. 31 provider-free tests, including a golden that rebuilds one
  frozen property from its own provenance and reproduces its 22 condition ids exactly.
- Verified 37 invariants; dry-ran the plan (44 target units per arm, 332 conditions, projected
  ~593k tokens both arms); smoke-tested one property.
- Ran Stage A: baseline then candidate, 22 properties each, 66 cases each, whole target units
  reviewed. Compared mechanically and froze the bundle.
- Part B re-verified every hash, read every rationale on every control, finding and changed row,
  reconciled at the condition grain against the approval's census, measured the instrument's own
  noise three ways, applied the rubric, and wrote the result, this handoff and the register block.
- Steven authorized a repeat the same day. Part B added `replicate` and `compare-replica` to the
  harness (same case set, prompts, models, thresholds; Stage A records read-only; the rebuilt Terra
  request's fingerprint asserted equal to Stage A's so every replica is a same-prompt control), ran
  Pass 2d five more times on every row and Terra once more on every Stage A unit in both arms,
  appended the results to the bundle, and revised the recommendation.

## Decisions and invariants

- **The rubric is conjunctive and was applied that way, twice.** At Stage A one clause was
  undetermined, so the answer was `inconclusive`, not `supported` with a footnote. After the repeat
  the clause is determined: a real, bounded, reproducible migration onto one item. Part B reads it on
  the program's standing priority and recommends publication; it states the strict reading
  (`candidate_rejected` pending the trim lever) rather than absorbing it.
- **Same-prompt replication is the control that this family needs.** Terra flips about 8% of
  verdicts on identical prompts in both arms (11.4% on the parent claim, 7.3% on the candidate's);
  a human-adjudicated correct rejection came back supported in 3 of 4 identical calls. Pass 2d is
  stable (97.9–99.4% replica agreement); its instability is confined to subject-generic bullets.
- **Instrument noise on this family is much larger than the floor of record.** Stored 3.1 run
  → live baseline, identical claim and photos: 11 of 66 Terra verdicts flipped (16.7%). Neighbours
  with identical claims reproduced the 6.4% floor (20/251). Any future verdict-scored gate on the
  window-treatment family needs its own same-prompt control arm.
- **Production feeds the product-filtered lane** (`product_estimate_issues_flat`), not
  `estimate_issues_flat`; the latter builds ~17% more conditions than the run ever had.
- **Terra reviews whole estimate units.** A subset payload changes the photo set, the prompt's
  numbering and the fingerprint; Stage A reviewed whole target units so neighbours are measured too.
- **The manifest's condition ids are 3.1-era** (they hash the catalog sha and fingerprint) and
  reproduce in neither 3.2 arm; join by `case_id`/`issue_id` and use the bundle's crosswalk.
- **Pass 2d's LLM path is not deterministic** (temperature 0.2, no seed): live baseline drifted from
  the frozen resolution on 1 of 66 rows, onto `dated_interior_trim`.
- **A transport failure is never an outcome.** "The model could not be reached" and "the observation
  resolved to nothing" are different facts; the harness retries the first and stops loudly, and never
  caches it.
- **Part A wrote facts; Part B judged.** The split is deliberate and should be kept for any later
  cost-gated session.
- **S6-1 ruled fix first (Steven, 2026-09-07, in chat).** The strict reading governs:
  `candidate_rejected` on `6e67eaa` pending the trim lever. The lever was evaluated offline the same
  day on all 1,969 replay-corpus rows (sidecar only, no provider call): the authorable
  `deny_any "window treatment"` fixes the S6-1 row but opens a new billable shortcut onto
  `dated_or_older_windows` for "Window trim and window treatment are dated." (S6-11); a `require_any`
  subject gate on the trim item changes no shortcut and loses only four subject-mismatched trim rows,
  and is the recommended CAP-022, but it is a system gap (carryover overrides are wording-only, S6-12)
  that Session 7 closes with a one-line generator change.
- **S5-1 ruled: accept the missed fabric condition (Steven, 2026-09-07).** The fabric successor's
  wording stays as approved; `oc1_93a2e3f3d299a7a1` is an accepted miss at $0, and the approval's
  surviving-billings count is accepted as approximate (optimistic by four in this set). Closed.

## Deviations from the session brief

| Deviation | Reason | Impact |
|---|---|---|
| A purpose-built harness rather than an existing driver | no entry point replays a stored Pass 2c into a live Pass 2d: the canary drivers re-run 2a/2b/2c, `replay_renovation_architecture` is provider-free by construction, `redecide_renovation_architecture` refuses on catalog drift (all stored artifacts are 3.1 / projection v2) | one untracked script and its tests; no production module changed |
| Pass 2d re-run for the 66 pinned rows only | the manifest freezes 2a/2b/2c and pins the case set; every other row keeps its stored resolution, all of which exist in both 3.2 catalogs (checked) | the replay differs from the stored run only where the experiment intends |
| Four parent-owned rows outside the pinned set excluded from both arms | stale ids under the candidate catalog; the projection fails closed on them | none on any target unit; each is its own condition; listed in the bundle |
| Whole target units reviewed, not condition subsets | Terra judges a unit | neighbours reviewed and reported as migration context |
| Billing derived as a standalone estimate over reviewed units | nothing emits a per-condition `billed_amount`; whole-property figures need whole-property runs | comparative between arms, never a headline |
| `LM_STUDIO_URL` substituted (`127.0.0.1:1234` for the pinned `169.254.83.107:1234`) | the pinned address is a stale link-local route; the server is now local; the model identity is the routing invariant and was verified | the "minimal run-wrapper fix for an operational defect" the brief permits; identical in both arms |
| Embeddings sidecar restarted mid-session | `ErrorDeviceLost` on a real POST while `/health` reported ok; the verify gate blocked spend until it was restarted with `scripts/start-embeddings-server.ps1` | no experiment input changed |
| Harness corrected after a failed smoke attempt | with the GPU saturated by an unrelated application, two Pass 2d calls failed on transport and were recorded as resolved-to-nothing outcomes, exit 0 | fixed (retry, no cache, fail loud); records deleted; every kept measurement is from a clean call |
| The predeclared repetition rule was replaced by a same-day repeat stage | Part A ran the mechanical stop conditions only; Part B judged the Stage A result inconclusive on one clause and Steven authorized 800,000 further tokens in chat; the repeat replicated every row and every unit rather than a hand-picked subset, so no case was selected after seeing outcomes | 572,308 tokens spent; 850,250 remain unspent of the combined 2,000,000 |

## Commands and verification

Run from the main checkout. The Pass 2d address flag precedes the subcommand.

| Command | Result |
|---|---|
| `KIND_ONTOLOGY_VERSION=observation_kind_v2 python scripts/catalog_audit_live_validation.py --lm-studio-url http://127.0.0.1:1234 verify` | **pass, 37/37** |
| `… plan` | 22 properties, 44 target units/arm, 332 conditions/arm, reservations 1.59M/arm (settle to ~5–8k/call) |
| `… run --arm baseline` | 22 properties, 44 units, 306,650 tokens, no stop, 7 min 03 s |
| `… --arm-root C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4 … run --arm candidate` | 22 properties, 38 units, 268,441 tokens, no stop, 6 min 09 s |
| `… compare` / `… freeze` | bundle `24bda3c3…` (Part A) |
| `… --lm-studio-url http://127.0.0.1:1234 replicate --arm baseline --pass-2d-replicas 5 --terra-replica` | 22 properties, 66 rows × 5, 44 same-prompt Terra units, 304,324 tokens, no stop, ~10 min |
| `… --arm-root C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4 … replicate --arm candidate --pass-2d-replicas 5 --terra-replica` | 22 properties, 66 rows × 5, 38 same-prompt Terra units, 267,984 tokens, no stop, ~8 min |
| `… compare-replica` | `compare_replica.json`; appended to the bundle as `repeat` |
| `pytest tests/test_catalog_audit_live_validation.py tests/test_catalog_audit_validation.py -q` | **46 passed** (34 new + 12 Session 5), before and after every run |

Run with `-p no:cacheprovider`, `--basetemp` under the session scratchpad, `HF_HUB_OFFLINE=1`.
`KIND_ONTOLOGY_VERSION=observation_kind_v2` must be set **before** the interpreter starts.

### The measurements the brief and the approval asked for

- **Ruling 5**: 26 of 28 moving conditions stop billing (22 at no_action on the blinds successor, 4
  undetected); 2 bill on `dated_interior_trim`.
- **A-4**: 23 no_action blinds conditions at zero dollars in this set; 6 conditions undetected.
- **A-5**: "The curtain rods are dated." → fabric successor, Terra **unsupported**, $0. Hardware does
  not bill under the fabric claim.
- **A-6**: makeshift covering undetected, as accepted.
- **Surviving on fabric**: 6 bill, 3 Terra-unsupported, 1 unreachable, of 10.
- **F-SHORTCUT-fee5271eaff08de5**: did not fire (live margin 0.029982 vs offline 0.03004); blinds,
  no_action, $0.
- **F-REACH-oc1_93a2e3f3d299a7a1**: confirmed unreached; $0.
- **Controls**: rc_7c5b9f9bb3aa withdrawn as approved; rc_64f996e3a886 preserved (baseline flip is
  noise); rc_ffad088fa5fe flipped to supported, unbillable, as predicted; rc_429ba6851d09 and
  rc_4f4b93c40ef3 formed no condition; rc_a22a241b3bf0 billed $76 on trim; rc_8afb476fa69f unchanged.
- **Dollars (context)**: baseline $4,521–22,613 → candidate $979–4,895 over the reviewed units.
- **Repeat stage**: Pass 2d agreement with Stage A 328/330 (baseline) and 323/330 (candidate); trim
  landings 5/396 vs 18/396; the two candidate-attributable rows on trim 6/6 and 5/6, and 0/12 under
  the baseline; no other billable item reached; Terra same-prompt flips 7.8% / 8.6% overall, 11.4% /
  7.3% on targets; the trim claim supported 4/4.

## Outputs for the next session

| Artifact | Identity | Contract |
|---|---|---|
| `reports/catalog_audit_live_validation.json` | `ee1bcf62…4f5c` | the ruled bundle (status `ruled`, `final_recommendation: candidate_rejected`); Part A rows untouched; `recommendation`, `repeat`, `recommendation_after_repeat`, `rulings` appended |
| `docs/RESULT_catalog_audit_live_validation_20260907.md` | `55f6405d…89e4` | the readable result, with the repeat stage and the rulings |
| `docs/catalog_audit_program/PROPOSAL_CAP-022_trim_subject_gate.md` | `7e1a625d…1d55` | the trim lever proposal for Steven's gate decision; Session 7's input |
| `artifacts_canary/catalog_audit_session6_20260907/` | per-file hashes in the bundle | raw run evidence, resumable; the baseline arm's per-condition outputs are the Catalog 3.2 acceptance thread's partial data |
| `docs/catalog_audit_program/FOLLOW_UP_REGISTER.md` | `17a4ea9f…5d1c` | §9 Session 6 block S6-1…S6-12, S5-1 closed |
| `scripts/catalog_audit_live_validation.py` + tests | `439235ba…` / `20a3a7d3…` | reusable for a frozen-2c Stage B and for any further same-prompt replica |
| Candidate commit | `6e67eaa`, unchanged | still the candidate arm |

## Unresolved questions and risks

1. **S6-1, ruled fix first.** Two conditions migrate onto the billable `dated_interior_trim`
   reproducibly (11 of 12; 0 of 12 under the baseline). Steven chose the strict reading; the lever is
   CAP-022 (`require_any` subject gate on the trim item), which needs his approval of the exact op at
   the gate and a Session 7 implementation with Tier 1/2 and a Stage A re-run on the combined change.
2. **S6-2**: six moving conditions are undetected rather than displayed at no_action — same dollars,
   different frontend envelope, against A-4's "stays visible" intent.
3. **S6-3**: the approval's surviving-billings count was optimistic by four in this set (S5-1's one
   plus three Terra rejections under the tighter fabric claim). A-5's hardware question is answered.
4. **S6-4**: Terra's same-prompt instability, now measured directly: about 8% in both arms on
   reviewed units, 11.4% on the parent-claim targets, 7.3% on the candidate's; a human-adjudicated
   correct rejection supported in 3 of 4 identical calls.
5. **S5-1 closed** (Steven, 2026-09-07: the missed fabric condition is accepted; no wording change);
   **S5-2** confirmed threshold-fragile, benign this run; **S5-4** gold drafts still deferred.
6. **Catalog 3.2 acceptance** still has no fresh Sol data; Stage B's baseline arm remains the
   vehicle. Stage B is no longer blocked by the mapping-hijack stop condition on Part B's reading,
   but it is only worth its ~2.5M tokens if Steven wants that acceptance data.
9. **S6-9**: the redraft's leak-test prediction for rc_429ba6851d09 was wrong (never reaches the
   blinds successor; ends undetected, $0). **S6-10**: the candidate's claims are easier for Terra.
7. **Operational**: the manifest's `LM_STUDIO_URL` is stale; the sidecar's device-loss recurs under
   GPU contention; Pass 2d runs on the local Qwen model in production (Steven's observation, recorded
   for the Pass 2d thread).
8. **Nothing is pushed and nothing is tracked.**

## Required next task

Session 7: land the trim lever (CAP-022) and re-validate the combined change.

Objective: approve, implement and validate `PROPOSAL_CAP-022_trim_subject_gate.md` so that CAP-007
can publish under the strict reading Steven chose.

Required inputs: this handoff, the CAP-022 proposal, `RESULT_catalog_audit_live_validation_20260907.md`
(section "Rulings"), the ruled bundle (`rulings`, `final_recommendation`), `HANDOFF_SESSION_4.md` (how
the candidate branch was built and proven), `06_SESSION_DETERMINISTIC_VALIDATION.md` and
`07_SESSION_LIVE_VALIDATION.md`.

Steps, in order:

1. Gate: Steven approves the exact CAP-022 op. The `require_any` term list is the only free choice
   (set B recommended; set C recorded). If he prefers the authorable `deny_any` form, the proposal
   records its side effect (S6-11) so that choice is made knowingly.
2. Close the system gap (S6-12): admit `require_any` to the generator's carryover override set in
   `scripts/migrate_catalog_kind_v2.py` (one line, renamed since it is no longer wording-only) with a
   test, on the candidate branch `catalog_audit_session4` in the worktree, as a new commit on `6e67eaa`.
3. Apply the op to `tools/catalog_migrations/kind_v2_decisions.json` with the renderer's `apply_ops`,
   regenerate twice (byte-identical), prove the generated diff equals the op (only `dated_interior_trim`
   gains `require_any`; no other item changes in any field), validator 0 errors; commit.
4. Tier 1/2 with Session 5's harness against the new candidate commit. The Tier 2 snapshot compare on
   the replay corpus must show trim leaving 699 top-8 lists, 0 shortcut changes, and exactly the four
   trim-owned losses the proposal lists.
5. A new live manifest pinning the new candidate commit, catalog sha and projection fingerprint; the 66
   cases, prompts, routes and thresholds unchanged; budget fields null until Steven authorizes.
6. Stage A re-run. The baseline arm is reusable from
   `artifacts_canary/catalog_audit_session6_20260907/baseline/` (same commit, catalog, fingerprint);
   only the candidate arm runs, resuming from disk by request fingerprint. Add a local Pass 2d x5 pass
   on the 22 corpus rows where trim was the first candidate, Terra only if a new billable landing
   appears. Expected: "Window treatments are dated and basic." no longer bills; "Window blind and trim
   appear dated." stays on trim as a declared expected landing; nothing else moves; A-4 and the kitchen
   shortcut unchanged.
7. Apply the rubric; if it passes, the publication task is justified on the combined change.

Authorized changes for that task: the generator gap closure with its test, the decisions-file op and
the regenerated catalog on the candidate branch, evaluation artifacts, reports, a new manifest.
Prohibited: any other decisions, catalog, prompt, model-route or threshold change; any edit to a
stored artifact or to Session 6's frozen records; publication without the standing gates; Stage B.

Exit criteria: CAP-022 approved and landed on the candidate branch with the generated-diff proof;
Tier 1/2 green; Stage A re-run reconciled to the manifest and the rubric applied; the trim rows'
outcomes recorded; publication go/no-go put to Steven.

## Do not redo

- The offline trim-lever evaluation: four levers on all 1,969 replay-corpus rows, frozen under `provenance/cap022_lever_eval/` and summarized in the proposal, the RESULT "Rulings" section and bundle `rulings`. Session 7 re-measures with its own Tier 2 snapshot; it does not need to repeat this.
- Do not re-run either arm to "confirm" it; both are complete, hash-recorded and resumable, and
  neither Pass 2d nor Terra would reproduce.
- Do not edit a Part A row in the bundle; Part B appended only, and any later session should do the
  same.
- Do not treat the ten unresolved candidate rows as harness failures; the harness stops loudly on a
  real failure and did not stop.
- Do not treat the baseline arm's flips (rc_64f996e3a886 to supported, one Pass 2d drift onto trim)
  as candidate effects; they are the instrument.
- Do not add a runtime alias for the deprecated parent, edit a stored artifact, or set
  `ISSUE_CATALOG_PATH`.
- Do not re-run the repeat stage; both arms' replicas are complete, hash-recorded and appended.
- Do not run Stage B without its own authorization and a reason to want its Sol data.

## Suggested opening prompt for the next task

> Continue the catalog-audit program from `docs/catalog_audit_program/HANDOFF_SESSION_6.md`. Session
> 6 is complete and ruled: Steven chose fix first on S6-1, so CAP-007's candidate `6e67eaa` is
> `candidate_rejected` pending the trim lever, drafted as `PROPOSAL_CAP-022_trim_subject_gate.md`.
> Read the handoff, the proposal, the RESULT doc and the ruled bundle, re-verify the hashes, and run
> Session 7: obtain Steven's approval of the exact CAP-022 op, close the generator's carryover
> override gap with a test, apply the op and regenerate on the candidate branch with the
> generated-diff proof, run Tier 1/2, write a new live manifest, and re-run Stage A's candidate arm
> (baseline arm reused) once Steven fills in the budget. Make no provider call before that. Do not run
> Stage B.
