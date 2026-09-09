# Handoff — catalog audit session 5

Date: 2026-09-07

Session/task title: Deterministic and retrieval validation (`06_SESSION_DETERMINISTIC_VALIDATION.md`) — CAP-007 candidate

**Status: COMPLETE. Tier 1 pass (36/36), Tier 2 pass (14/14), no provider call, live manifest
written with budget fields left null.** Two material findings are carried to Session 6; neither is
an implementation defect, and one of them means the approval's accounting of surviving billings is
optimistic by one condition. Session 4's two known failures were decided, not papered over.

## Outcome

| Exit criterion | Result |
|---|---|
| Explicit Tier 1 and Tier 2 pass/fail | Tier 1 **pass** 36/36; Tier 2 **pass** 14/14 |
| Actual diff reconciled against the approved diff | reproduced independently from the baseline: ops → generate → byte compare |
| Affected, support and control cases reconciled | 48/48 parent conditions, 46/46 named cases, 7/7 review cards |
| No live provider call | none; the only endpoint used was the embeddings sidecar |
| Live manifest only if passed | written, `status: awaiting_human_cost_authorization` |
| Decision recorded on S4-1 and S4-2 | both decided; see *The two handed-over failures* |

## Repository state

- Baseline arm: main checkout `C:/Users/Steven/PycharmProjects/realtorvision-backend`, branch
  `terra_factorized_verifier`, commit `9afe0fa5a0490a856abed507dccc4a02ed1de24c`. Tracked-clean at
  start and at end.
- Candidate arm: worktree `C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4`, branch
  `catalog_audit_session4`, commit `6e67eaa83887cf4d218100dcd060c07d3bd30522`. **Unchanged by this
  session.** The generator was run there twice to prove byte parity; `git status --porcelain -uno`
  was empty before and after.
- Nothing was committed. All Session 5 outputs are untracked in the main checkout, matching the
  program's posture; whether the program artifacts should be committed remains delegated to
  `docs/HANDOFF_git_repository_audit.md`.
- Pre-existing changes preserved: yes. Every protected artifact re-hashed at the end and unchanged.

## Inputs verified

Every hash in `HANDOFF_SESSION_4.md` and every `approved_against` pin in the approvals manifest was
re-verified mechanically (Tier 1 checks 1.1c and 1.1g), not read across.

| Input | Expected | Result |
|---|---|---|
| Baseline HEAD | `9afe0fa` | match |
| Candidate HEAD | `6e67eaa` | match |
| `reports/catalog_audit_approvals.json` | `a725c89a…c871` | match |
| `reports/catalog_audit_evidence.json` | `43354104…29b7` | match |
| `reports/catalog_audit_proposals.json` | `1411352c…3ef99` | match |
| `reports/catalog_audit_adversarial_review.json` | `d1149875…b5c4` | match |
| redraft / renewed review / gate decisions / live check | as pinned | match |
| baseline decisions / catalog | `47614d82…03e8` / `51bf7e26…aa54` | match |
| candidate decisions / catalog / manifest / audit md | `2e49a82f…9c76` / `43d8147e…7acf` / `6b944b83…` / `cac2750a…` | match |
| generator / validator | `b732022f…a053` / `041637a3…bee2` | match, unchanged by this session |
| `FOLLOW_UP_REGISTER.md` | `b0ecd66f…9b2d` (Session 4 value) | match at start; §9 appended here, new value below |
| 25 pinned run artifacts | `current_sha256` per bundle row | all 25 match |

## Files created or changed

All in the main checkout, all untracked.

| Path | Status | sha256 |
|---|---|---|
| `scripts/catalog_audit_validation.py` | new | `dbacf929b5ddba177e555456ee51d1685ee9b7b79261889c973fc1a1d8127ce6` |
| `tests/test_catalog_audit_validation.py` | new, 12 tests | `ba3bf5a4a059f31f04f56ca82b64cfbef0727789fe00bbeef023e2da1d708684` |
| `reports/catalog_audit_tier1.json` | new | `cbe2bc59b35775c5bf52145436a6d2a4b6fe19720a0057bbd8a45c56100a2b07` |
| `reports/catalog_audit_replay_corpus.json` | new | `159053217cf434793642667646cf79d92198d9900e061fbe360cd25149d0c965` |
| `reports/catalog_audit_replay_snapshot_baseline.json` | new | `07482b02c4d3cd389d281f5dc670ad0639ce12aed6e1bcb427f0ceec3d950181` |
| `reports/catalog_audit_replay_snapshot_candidate.json` | new | `040cc41b79d569a0b39a5df3dde0837bf8d03aa4517120245c9439596c1bdbfb` |
| `reports/catalog_audit_validation_tier12.json` | new | `674a9fccea8a4bb1ddcfcab10b198f3621b47a7553614ce48f817233994aee4a` |
| `reports/catalog_audit_live_experiment_manifest.json` | new | `e1f3b7f1f967266e64c71dedb6f93cef2e329cc02dca78799cfec4a678a4a42f` |
| `docs/RESULT_catalog_audit_validation_tier12_20260907.md` | new | the readable result |
| `docs/catalog_audit_program/HANDOFF_SESSION_5.md` | new | this file |
| `docs/catalog_audit_program/FOLLOW_UP_REGISTER.md` | §9 appended | see *Outputs* |

No production file was touched. The candidate commit is unchanged.

## Work completed

- Verified both arms and every pinned hash before running anything.
- Tier 1: arm identity, double regeneration in the worktree, in-memory baseline reproduction,
  explicit v2 validation on both arms, approved-versus-actual reconciliation, economic/route/package
  diff, item identity and order. 36 checks, all pass.
- Ran the full suite in both arms in a clean environment with a scratchpad base temp, and matched
  every candidate failure to its expected classification by message.
- Built a replay corpus from the 25 pinned artifacts: 4107 stored Pass 2d rows, 1969 selected.
- Proved the replay is comparable at all: today's shortcut function reproduces all 4107 stored
  rows with zero mismatches, and the approved 48/28/10/1 census reproduces from the artifacts under
  the production gate functions.
- Probed the sidecar with a real POST, then replayed both arms over a shared vector cache so the
  only variable is the catalog. The candidate run needed exactly two new embeddings.
- Evaluated 14 Tier 2 rules and produced the four measurements the approval requires.
- Decided S4-1 and S4-2, produced the stale-id inventory and two grounded benchmark gold drafts.
- Built the pinned live manifest with both budget fields null.

## Decisions and invariants

- **Code identity across arms is compared by git blob, not by on-disk hash.** 92 tracked files are
  checked out LF in the main checkout and CRLF in the worktree, with identical blobs. None is in the
  retrieval path. The four catalog artifacts are CRLF in both arms and reproduce their pinned
  hashes, so the program's CRLF pins are unaffected.
- **Both arms stamp catalog version `3.2`.** The version string cannot distinguish them. Live runs
  must be identified by `catalog_sha256` and the projection fingerprint, both pinned in the manifest.
- **Top-K is 8, not 5.** The retriever defaults to 5; production passes `top_k_candidates=8` in the
  Pass 2d context. Replaying at 5 would understate reachability on every row. Pinned in the manifest.
- **The shared vector cache is what makes the arms comparable.** Identical text embeds to identical
  vectors, so residual score drift is float32 accumulation noise from a 129-row matrix instead of a
  128-row one. Measured maximum 3.6e-07, four orders of magnitude below the 0.03 margin gate.
- **Guardrails run after top-K**, so a denied successor still spends a candidate slot. This explains
  most apparent displacement and is why eviction is accounted by score, not by rank.
- The census is a statement about the term gates only. Retrieval adds ranking, which is why the
  reachability audit exists and why it does not match the census exactly.

## Deviations from the session brief

| Deviation | Reason | Impact |
|---|---|---|
| A new script rather than extending an existing comparator | No two-catalog retrieval comparator exists; the evidence builder and both renderers read frozen artifacts and never run retrieval | One focused script plus 12 tests, both untracked, no existing harness modified |
| The catalog-resolution benchmark was not run as the gross regression check | It requires an LLM for Pass 2d selection, which this session may not use | Its two provider-free probes and the shipped-case check were run instead; the affected-observation replay is the authoritative evidence either way |
| The Tier 2 rules distinguish hard rules from a reachability audit | A bare pass/fail would have hidden that 45 of 48 conditions behave as approved and 3 do not | Both are reported; the exceptions are named findings rather than an averaged number |

## Commands and verification

Run from the main checkout unless stated. `--arm-root` precedes the subcommand.

| Command | Result |
|---|---|
| `python scripts/catalog_audit_validation.py tier1 --candidate-root C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4` | **pass, 36/36** |
| `python scripts/catalog_audit_validation.py corpus` | 1969 rows of 4107, 25 artifacts, self-test **pass** (0 mismatches), census 48 = 48 |
| `python scripts/catalog_audit_validation.py probe` | dim **1024**, model and build recorded |
| `KIND_ONTOLOGY_VERSION=observation_kind_v2 python scripts/catalog_audit_validation.py snapshot --arm baseline --cache <scratch>/vector_cache.json --out reports/catalog_audit_replay_snapshot_baseline.json` | 1969 rows, 354 shortcut fires |
| (from the worktree) `KIND_ONTOLOGY_VERSION=observation_kind_v2 python <main>/scripts/catalog_audit_validation.py --arm-root C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4 snapshot --arm candidate …` | 1969 rows, 360 fires, cache 4065 hits / **2 misses** |
| `python scripts/catalog_audit_validation.py compare` | **tier2 pass 14/14**, overall pass; deterministic (run twice, identical) |
| `python scripts/catalog_audit_validation.py manifest --candidate-root …` | 66 cases, 66/66 photos pinned, budgets null |
| `pytest tests/test_catalog_audit_validation.py -q` | **12 passed** |
| Full suite, baseline arm | **2756 passed, 6 skipped, 0 failed** |
| Full suite, candidate arm | **2759 passed, 6 failed, 21 skipped** — the six known, each matched by message |

Suites were run with `-p no:cacheprovider`, `--basetemp` under the session scratchpad,
`HF_HUB_OFFLINE=1`, and `KIND_ONTOLOGY_VERSION` / `ISSUE_CATALOG_PATH` unset (the suite asserts the
v1 default). The snapshot command requires `KIND_ONTOLOGY_VERSION=observation_kind_v2` set **before**
the interpreter starts, because the selector resolves at import.

### The four required Tier 2 measurements

1. **Blinds shortcut fire rate**: 26/46 onto the parent at baseline (56.5%) → 23/46 onto the blinds
   successor (50.0%). Nine rows fall below the 0.03 margin gate and now reach the LLM instead of
   resolving deterministically, because the presence-only successor carries no datedness language
   and sits closer to `dated_or_older_windows` than the parent did.
2. **Post-split ranks against `dated_or_older_windows`**: 913 rows; no ranked position lost where it
   owned the row or ranked top-three; its reviewed correct rejection is bit-identical; blinds
   outranks it on 62 blinds-subject rows; no row moved from it to the blinds successor by shortcut.
3. **Fabric bare stems**: 24 rows; 13 gate-eligible, 7 denied (the A-3 stems doing real work), 4
   kind-excluded, 6 shortcut onto the fabric successor. **No neighbour hijacking.** The approved
   27-row leak test reproduces field for field.
4. **Embedding neighbourhoods**: reported per item, same kind and shared scenes. Fabric-to-windows
   0.738 is marginally closer than fabric-to-blinds 0.737; blinds-to-windows 0.691; the parent was
   0.786 from windows.

## Outputs for the next session

| Artifact | Identity | Contract |
|---|---|---|
| `reports/catalog_audit_live_experiment_manifest.json` | `e1f3b7f1…a42f` | the only authorization surface for Session 6; budgets null |
| `reports/catalog_audit_validation_tier12.json` | `674a9fcc…e4f4` | all rules, metrics and findings |
| `docs/RESULT_catalog_audit_validation_tier12_20260907.md` | — | the readable result |
| `reports/catalog_audit_replay_corpus.json` | `15905321…0965` | the frozen case population |
| Both snapshots | `07482b02…` / `040cc41b…` | per-arm retrieval evidence |
| `docs/catalog_audit_program/FOLLOW_UP_REGISTER.md` | now **`b53c217ba02791f751f744ee1d28c4934b76bdc3d441dfa1be952f895c67061d`** | §9 gained a Session 5 block (S5-1…S5-5) |
| Candidate commit | `6e67eaa`, unchanged | still the candidate arm |

## Unresolved questions and risks

1. **The approval's surviving-billings count is optimistic by one condition.**
   `oc1_93a2e3f3d299a7a1` is recorded among the ten surviving on the fabric successor, but that item
   is not in the candidate top-8 for either of its observations — on the valance-naming bullet,
   generic decor language dominates the embedding. **This is a decision for Steven, not for a later
   session**: accept the accounting as approximate, or reopen the wording. It is not an
   implementation defect; the diff matches the approval exactly.
2. **One blinds observation now resolves deterministically to a billable neighbour.**
   "The mini blinds feel dated compared with the updated kitchen." shortcuts onto
   `outdated_kitchen_finishes` at a margin of 0.03004, four hundred-thousandths above the gate. In
   the baseline the parent held rank 1 below the gate, so the row reached the LLM. Ruling 5 intends
   blinds conditions to become no_action; this one bills under a kitchen item instead. Most
   threshold-fragile decision in the corpus.
3. **91 rows decide the shortcut within 0.002 of the margin threshold.** Any future embedding model
   change, catalog wording change or sidecar build change flips these first. Worth a regression pin
   if the shortcut behaviour is ever treated as stable.
4. **Three subject-generic bullets reach neither successor**, as the approval intended, but two of
   them now shortcut onto `dated_or_older_windows`. That item's route is also `no_action`, so the
   economic outcome is unchanged; the frontend envelope shows a different item id.
5. **97 stored canary artifacts still name the deprecated parent** across six roots. Each fails
   offline recompute against the candidate catalog. Inventory is in the manifest.
6. **`00_OVERALL_CONTEXT.md` §3 decision counts remain stale** (50/19 against the candidate's 49/20)
   — Session 4's S4-4, still open, still a one-line correction.
7. **Nothing is pushed and nothing is tracked.** The branch is local-only, as is the whole program.

## Required next task

Objective: Session 6 per `docs/catalog_audit_program/07_SESSION_LIVE_VALIDATION.md` — cost-gated
live validation of candidate `6e67eaa` against baseline `9afe0fa`, driven **only** by
`reports/catalog_audit_live_experiment_manifest.json`.

Required inputs: `00_OVERALL_CONTEXT.md`, the Session 6 brief, this handoff, the live manifest
(`e1f3b7f1…a42f`), and `reports/catalog_audit_validation_tier12.json`.

Authorized changes: `reports/catalog_audit_live_validation.json`,
`docs/RESULT_catalog_audit_live_validation_<date>.md`, raw run artifacts, `HANDOFF_SESSION_6.md`.

Prohibited: any provider call before a human fills in the manifest's budget fields; changing Pass
2a/2b/2c, prompts, model routing, or the case set; broadening the case set to improve a result;
publication or cutover.

**The live cost gate.** `budget.stage_a_authorized_tokens` and `budget.stage_b_authorized_tokens`
are both `null`. Session 6 must stop and ask before spending anything. Reference costs: about 3.98M
tokens for a full 18-property replica; about 3.2k tokens per single-condition Terra call, so the 66
targeted Stage A cases are on the order of a few hundred thousand tokens if run as single-condition
calls. Stage B requires its own separate authorization even if Stage A passes.

Exit criteria: Stage A run exactly as pinned; the two material findings above measured live; a
decision from the rubric in the Session 6 brief; the Catalog 3.2 acceptance data from the baseline
arm preserved and handed off; no run exceeding its authorization.

## Do not redo

- Do not re-verify the Tier 1 diff by hand; it was reproduced independently from the baseline
  decisions via the approved ops and matched byte for byte, and the report records every check.
- Do not re-run the Tier 2 replay to "confirm" it. It is deterministic and the vector cache is
  keyed on the sidecar's model and build; rerunning proves nothing new unless the catalog changes.
- Do not treat the six candidate-arm test failures as regressions. Three are correct by design,
  one is a worktree environment gap, and two are the handed-over items decided here.
- Do not add a runtime alias for `dated_window_treatment_valance`, edit any stored artifact, or
  edit a frozen benchmark slice to turn a red test green.
- Do not tidy the claim texts (A-7), add `defaultHidden` to the blinds successor (A-4), or remove
  the inherited economics that keep the `route_override` valid.
- Do not touch CAP-008 or any other manifest record.
- Do not set `ISSUE_CATALOG_PATH`; under `observation_kind_v2` it is a hard configuration error.

## Suggested opening prompt for the next task

> Execute Session 6 of the catalog-audit program per
> `docs/catalog_audit_program/07_SESSION_LIVE_VALIDATION.md`. Read `00_OVERALL_CONTEXT.md`, that
> brief, `HANDOFF_SESSION_5.md`, and `reports/catalog_audit_live_experiment_manifest.json`
> (`e1f3b7f1…a42f`). Verify every invariant in the manifest's
> `invariants_to_verify_before_any_call` list first, and stop if any fails. **Make no provider call
> until Steven has filled in the manifest's budget fields.** Then run Stage A on the 66 pinned
> targeted cases only, replaying from the frozen Pass 2c observations named in the manifest and
> keeping Pass 2a/2b/2c, prompts and model routing frozen. Measure the two material findings the
> manifest carries. Do not broaden the case set. A full canary needs its own separate
> authorization. Preserve the baseline arm's package outputs for the Catalog 3.2 acceptance thread,
> and finish with `HANDOFF_SESSION_6.md`.
