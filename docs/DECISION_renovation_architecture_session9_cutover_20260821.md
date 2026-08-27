# Decision record — Session 9 renovation architecture cutover (independent review)

Date: 2026-08-21 (evening) · Branch `renovation_architecture_rework` · HEAD `6640ebb` (tracked tree clean)
Reviewer: independent Claude session, working from the decision packet and
`docs/DESIGN_renovation_architecture_decision_packets.md`. Nothing in
`reports/`, `artifacts_canary/`, `.env`, or application code was modified.
Zero provider calls were made. Every finding in §3 was handed to a separate
skeptic agent instructed to refute it; verdicts and corrections are recorded.

## 1. Decision

**CONDITIONAL GO** for selecting the new renovation architecture
(`RENOVATION_ARCHITECTURE_MODE=new`) — conditional on the five items in §5:
two are Steven's signatures (minutes), two are mechanical (a complete worker
env + launch command, and the two live smokes the runbook already requires),
one adopts an observation rule.

The architecture's own evidence is complete and was reproduced first-hand
(§2). What the packet does **not** say (§3): the runbook's two-variable flip
is *also* the kind-ontology v2 cutover, which production has never had and
whose sign-off checklist is open — and this is enforced in code, not just the
runbook (`new` refuses to start without `observation_kind_v2`); the frontend
reads only `renovation_estimate_v4`, so after cutover the user-visible
estimate is v4-on-the-v2-catalog and v5 runs dark at the artifact root; and the
production worker's environment is the frontend's `.env` under `npm run
worker`, where the runbook's env list is incomplete. None of those rejects the
architecture; all of them change *how* the flip is done and *what it means*.

The design-doc packets P1–P4 (package-level hallucination, condition truth,
multi-bathroom, Sol variance) are **not** conditions. v5 runs dark after this
cutover, v4 stays the displayed estimator, and a dark `new` production turns
every real listing into P1/P2 evidence under *production* routing (§6). They
gate frontend adoption of v5, not this flip.

## 2. Verified first-hand this session

| Check | Result |
|---|---|
| HEAD, branch | `6640ebb8856f6552f065e3ed51f0ca00eb6d9ad8`, `renovation_architecture_rework`; `git status` shows **no tracked modifications** (Session 9 code was committed as `6640ebb` at 17:52:41, two minutes before the report's last write) |
| Packet's five file hashes | all five match byte-for-byte (prefilled reviews `1381fba2…`, pristine template `75c64611…`, report `37694749…`, `input_freeze.json` `f32f6812…`, digest `8e1f46d6…`) |
| Freeze ↔ code provenance | `input_freeze.json` hashes **24 code files + canary manifest + model map + catalog 3.1**; **27/27 equal the working tree at HEAD** (recomputed twice). Its `git_commit 07ee112` is only "HEAD at freeze time with Session 9 edits uncommitted": 7 of the 24 frozen files equal the `6640ebb` blobs, not `07ee112`'s — the frozen snapshot *is* `6640ebb` content. Unfrozen files changed `07ee112→6640ebb`: `tools/vlm_client.py`, `tools/analyzer_cli.py`, `tools/failure_taxonomy.py` — established as HEAD by telemetry (slugged pass keys and per-model buckets in all 36 debug artifacts; 4,066 choke-point ledger rows), and the diff touches no prompt, image, model, effort, cap, or parser (F6) |
| Test suite | `.venv\Scripts\python.exe -m pytest -q` on the working tree → **2337 passed, 6 skipped, exit 0** (34 s). Caveat (critic): the working tree includes **untracked** tests (`tests/test_renovation_architecture_*.py`, `test_vlm_budget_guard.py`) and untracked operational files (`scripts/run_worker_smoke.py`, `scripts/run_renovation_architecture_canary.py`, `scripts/show_daily_token_spend.py`, the Session 6 runbook) — a clean clone of `6640ebb` has neither |
| Comparator exit code (packet open item 1) | re-run offline into the scratchpad with the packet's exact arguments: **exit 0, `release_ready: true`, report byte-identical** to the official one (SHA-256 `37694749…161b`) |
| Report content | `release_ready true`, `gates.passed true`, 0 failures, 18 × 2 replicas, 967 review items (scope 798 / package 137 / stability 18 / headline 14), 0 unreviewed; packet tallies (736/77/154; 56/16/5/0; 8 merge-increase = 3 groups; Terra 1746/282/65; Sol 226/23; +11.9%/+20.1%; +0.6%/+0.9%) all reproduce (F8) |
| Working review file | 967 entries, all `accepted`, no blank decision or explanation (min explanation 173 chars) |
| Item-level audit | **not repeated** — done 2026-08-21 by the audit session (967 verified against envelopes; headline identity on all 14; stability invariant on all 36; design-doc numbers adversarially verified). Relied upon |
| Decision worksheet / design-doc log | `reports/session9_decision_worksheet.json`: **0 verdicts filled**; design doc §9 log: **empty**; Tier-2 explicit sign-off: **not recorded anywhere** |

## 3. What the packet does not say — findings with refute verdicts

| # | Finding | Verdict · corrections |
|---|---|---|
| F1 | **The flip is two cutovers.** All 1,264 production artifacts (`renointel-prod/artifacts`; newest 2026-08-03 local / 2026-08-04T03:44Z, nothing newer) are catalog **2.1 / legacy_v1**, no v5 anywhere. The Session 9 canary (36/36) is catalog **3.1 / observation_kind_v2 / publish**; the comparator pairs v4 and v5 from the *same* v2 artifact, so it measured v4-on-v2 → v5-on-v2, never production's v4-on-v1 → v5. `new` **cannot start** without `observation_kind_v2` (`pipeline_config.py:146-156`, `runtime.py:82-85`), so the flip is the Task 4A kind-ontology cutover by construction. 4A sign-off is open: `configs/kind_ontology_cutover.json` `approved_headline_deltas: {}`; checklist unticked; STATE doc "sign-off not clean"; no approval commit anywhere | **confirmed**. Correction: under catalog 3.1 (what would deploy) the open approval list is **23 headline rows on 14 properties** (`reports/kind_cutover_canary_catalog31_20260808.json`), not 26/15 from the 3.0 report; unresolved-rate gate still FAIL 18/18 |
| F2 | **Rollback to today's production is not an env flip.** `legacy_v1` → `classification_only` → `write_photo_intel` raises "refusing to publish a classification_only result" on every production entry point (server, CLI, audit runner); no env override exists. True v1 publication needs the pinned build — worktree `../rv-legacy-a1972cf` is present, clean, at `a1972cf` (catalog 2.1). Runbook §4 rollback (`mode=current`, ontology stays v2) yields v4-on-v2. The FE `.env` already launches *this* tree, so a default-env launch today would hit the refusal | **confirmed** |
| F3 | **The FE does not consume v5.** `renovationEstimate.ts` / `completionContract.ts` / `/api/analysis` / Prisma cache / top-picks ranking all read `renovation_estimate_v4`; zero references to `renovation_estimate_v5` in FE source, tests, or docs. Displayed estimate after cutover = v4-on-v2 (the 4A deltas); v5 dark | **confirmed**. Corrections: the slim builder does not strip the root v5 key, so under `new` the envelope is in the FE-read `photo_intel.json` and downloadable by **authenticated** users via `app/api/artifacts/[...file]` (guests blocked) — decide if acceptable; `kindOntology.ts` detects a 3.1 artifact only via the `ontology_version` stamp (catalog fallback pinned to `'3.0'`, no 3.1 test) |
| F4 | **The worker env is the FE's `.env`; the runbook's list is short for it.** The persistent Python worker is spawned by the standalone **`npm run worker`** process (`scripts/analysis-worker.ts` → `load-env.ts` `dotenv.config({override:true})` → `PersistentPythonWorker` with `{...process.env}`), never by the Next server; the backend loads no `.env` on that path. FE `.env` has `OPENAI_MODEL=gpt-5.6-terra` and none of `RENOVATION_TERRA_MODEL` / `RENOVATION_SOL_MODEL` / selectors / guard keys; Windows user/machine env has none either. With only the runbook's two variables, Sol resolves to **gpt-5.6-terra** and startup does not refuse (only empty models are rejected); with the guard on, a Terra-named Sol would debit the **Terra** ledger. `run_worker_smoke.py` injects the *backend* `.env` (which sets Sol correctly), so the smoke would not catch it | **partially** — restart target corrected: the **worker process**, not the FE server. Everything else confirmed |
| F5 | **`new` is job-fatal and the ceiling is unmetered by default.** Non-complete v5 → `AuthoritativeEstimateIncomplete` before any write → no artifact, v4 not published (the FE sees `success:false`). Architecture Terra/Sol hook ledgers are always on (2.5M / 250k, at `<artifacts_root>/.renovation_architecture/` = `renointel-prod/artifacts/.renovation_architecture/` in production unless `RENOVATION_TERRA_USAGE_ROOT`) but meter only the architecture's own calls; upstream 2a/2b/2c/2d on gpt-5.6-terra are unmetered unless `RENOVATION_VLM_BUDGET_GUARD=1` + usage root. Production routing: 2a/2b/2c/2d → gpt-5.6-terra (effort none), 2f → gpt-5.6-sol | **confirmed**. Additions: `AuthoritativeEstimateIncomplete` is unmapped in `failure_taxonomy`, so with the guard **off** a quota/ledger root cause reaches the FE as category **`provider`** (retry once after ~10 min, then `review_required`) instead of `quota` (queue pause); with the guard **on**, quota failures re-raise the original exception and arrive as `quota` |
| F6 | **Provenance residual** (three unfrozen files) | **partially** — see §2: content-equivalent to HEAD by telemetry; the only behavioural addition is that with the guard **on** the choke point can *deny* an upstream call before dispatch (loud: `quota`/`dependency`, never a payload change); 36/36 canary artifacts have `failed_calls = 0` |
| F7 | **The report's capacity number is architecture-only.** `terra_usage` sums only condition-review calls (295 rows, all `terra_condition_review_v1`); `canary_budget_debited_tokens 1,836,273` is both replicas of that one call. From the debug telemetry, all-pass gpt-5.6-terra per listing: **median 220,768, worst 359,877** (⇒ 11 / 6 listings per 2.5M day); Terra+Sol OpenAI median 231,290; both replicas together ≈ 8.12M OpenAI tokens | **confirmed** |
| F8 | **Packet internal numbers** | **partially** — all counts reproduce, but the "run 2 +12.6% / +21.1%" is replica-2 **v5 over replica-1 v4**; within replica 2, v5 vs its own v4 is **−3.3% low / +9.5% high** (v4 itself moved +16.4%/+10.6% between replicas on churned upstream inputs). Per-property v5 replica swing reaches **+100% low** (`redfin_11079485`, $7,896 → $15,806) and −42.8% high (`redfin_11000447`), so "±50%" understates the low side |

Also: no production analysis artifact has been written since 2026-08-03, and
on this checkout the `legacy_v1` default cannot publish — there is no "working
current production" the flip would disturb; the realistic comparison is
"flip" vs "keep not publishing / switch `RV_ROOT` to the pinned legacy build".

## 4. Why conditional GO — not NO-GO, not unconditional

**Not NO-GO.** Every gate the architecture set for itself passes and was
independently verified and reproduced; provenance closes on the files that
determine estimate content and, by telemetry, on the rest; the suite is green
on the exact tree; every rejection in v5 fails safe (children stay
standalone); v4 is still emitted and displayed. The within-replica residual
(F8) says v5 is not systematically inflating — the corpus sign flips with
upstream churn. P1–P4 are judgments about model behaviour, not gate failures,
and their fixes (package-warrant veto, bathroom-expansion rule, Sol rules)
need a fresh canary or an offline replay *whether or not* production is on
`new` — holding the flip for them buys no user-facing safety (the user sees
v4) and forfeits production evidence.

**Not unconditional.** The flip carries an unsigned second cutover (F1) that
*is* user-visible; the env list as written would put Sol on the wrong model
(F4); the live `new`/rollback smokes the runbook requires have never been run
(Session 6: "NOT DONE"; nothing since — and the smoke tool is untracked,
non-verifying, and replicates canary routing); the only mechanical defence
against silent Terra overrun is off unless switched on (F5); and the canary
characterised v5 under **canary routing** (2d local, 2f on Terra, upstream
effort low), not production routing (2d on Terra, 2f on Sol, effort none) —
a known caveat (token-budget handoff), not a gate, but a reason the dark
observation period matters before FE adoption.

## 5. Conditions (close all five, then run §7)

| # | Condition | Owner · cost | Closes when |
|---|---|---|---|
| C1 | **Task 4A sign-off or explicit written waiver.** The flip publishes v4-on-v2 to users. Either tick the four lines of the kind-canary sign-off checklist (unresolved-rate option a/b; the **23 headline deltas on 14 properties under catalog 3.1** — at minimum the six "review-first" rows; package lane-swap pattern; kind-mix shift) and fill `approved_headline_deltas` in `configs/kind_ontology_cutover.json`, **or** write one line in the design-doc §9 log that the architecture cutover knowingly carries the 4A deltas as the transitional display. Not a renovation-architecture gate — but the flip trips it by construction | Steven · 15–45 min | log line, or checklist + config |
| C2 | **Tier-2 sign-off** (design doc P6): three lines — (1) merge-dedup lower (56 items, −$13.6k/−$231k), (2) single-item wider-high (16, −$7/+$18.7k), (3) v4-$0→priced (5, +$1.8k/+$20.9k). Recommendation (mine and the design doc's): accept 1 and 2 outright; 3 provisionally — five single-photo items, revisit with P2. The file already records `accepted`; this only closes the governance gap | Steven · 5 min | three lines in §9 log |
| C3 | **Worker env + launch command.** In `renointel-prod/.env`, one edit: `KIND_ONTOLOGY_VERSION=observation_kind_v2`, `RENOVATION_ARCHITECTURE_MODE=new`, `RENOVATION_TERRA_MODEL=gpt-5.6-terra`, `RENOVATION_SOL_MODEL=gpt-5.6-sol` (caps default 8192; `ISSUE_CATALOG_PATH` stays unset — it is). Then restart **`npm run worker -- --premium`** — `--premium` is what routes 1a/2a/2b/2c/2d to Terra (`lib/analysis/workerMode.ts`; `PREMIUM_AI_MODE` in `.env` is unread); without it upstream passes fall to local Qwen and the inputs are nothing the canary measured. **Decide the guard:** `RENOVATION_VLM_BUDGET_GUARD=1` + `RENOVATION_TERRA_USAGE_ROOT=<one stable dir>` makes every OpenAI call debit the 2.5M / 250k ledgers and fail a listing *closed* at the ceiling (arrives at the FE as `quota`, which pauses the queue); off = today's silent behaviour, and a v5 quota failure reaches the FE as `provider` (one retry, then parked). Recommendation: **on** — it is the only mechanical form of the runbook's "silent overrun" stop condition; the FE already rations 2f on Sol (`OPENAI_PASS_2F_PRIORITY_LIMIT=8`/day) so the Sol ledger should rarely bind | Steven (guard) + mechanical | env set; first real artifact shows v5 `sol_call.model == gpt-5.6-sol`, `model_routing` 2c `gpt-5.6-terra` `explicit_override` |
| C4 | **Run the two live smokes the runbook requires** (never run): `scripts/run_worker_smoke.py --mode new --out <scratch>` then `--mode current`, with LM Studio reachable at the **FE's** `LM_STUDIO_URL` (the smoke injects the backend `.env`, whose `LM_STUDIO_URL` is the canary host `169.254.83.107`, not production's `127.0.0.1`) and the embeddings sidecar up (worker startup real-POSTs it and refuses `ready` otherwise). The script returns 0 on `job_done` only — verify the artifacts by hand: `new` → root `renovation_estimate_v5` `state complete`, `provenance.architecture_mode new`, `catalog_version 3.1`, `renovation_estimate_v4` present, no `analysis_debug.renovation_estimate_v5`; `current` → v4 present, no v5 at either location. The smoke forces canary routing (2a/2b/2c/2f → Terra), so production routing is only proven by the first real listing (C3) | mechanical · ~1 listing of tokens each (~220k Terra + ~10k Sol) | both smokes pass + artifacts inspected |
| C5 | **Adopt an observation rule and restate the envelope.** Use the kind runbook's rule for the combined flip: **7 calendar days**, operational invariants only, **rollback after three consecutive** ontology/resolver/writer failures — counting v5 job-fatal failures by code/category (a quota day under `new` would trip this quickly; that is the intended stop, not noise). Restate: ~10–12 listings/day all-service (F7: median 11, worst 6 under the 2.5M free quota); v5 failure = listing failure (no artifact, FE retries per `QUEUE_RETRY_ATTEMPTS`); v5 adds ~51k Terra + ~8.5k Sol per listing of dark output | Steven (already decided 2026-08-17 for capacity) | one paragraph in §9 log |

**Housekeeping, not gates:** commit the untracked Session 9 operational files and tests (runbook, `run_worker_smoke.py`, `run_renovation_architecture_canary.py`, `show_daily_token_spend.py`, `tests/test_renovation_architecture_*.py`, `test_vlm_budget_guard.py`) so a clone of the cutover commit carries them and the 2337 suite is reproducible from git; decide whether the slim builder should strip the root v5 key from the FE-read artifact (F3); optionally map `AuthoritativeEstimateIncomplete` in `failure_taxonomy` to carry the envelope's reason category (F5) — three lines, guard-off hardening only.

**Rollback ladder** (record it in the runbook when executing): L1 `RENOVATION_ARCHITECTURE_MODE=current` + worker restart → v4-on-v2 publishes, no v5 (the C4 smoke proves it); L2 `KIND_ONTOLOGY_VERSION=legacy_v1` on this build → **nothing publishes** (kill switch); L3 point `RV_ROOT`/`ANALYZER_CLI` at `../rv-legacy-a1972cf` → true v1 publication.

## 6. Not conditions — and how the flip works toward P1–P6

- **P1 / P2 / P3 photo passes** (worksheet C###/P##/B##): gate *FE adoption* of v5, not this flip. In `new` mode v4 still runs Pass 2f (on Sol), so every production listing yields 2f package-level verdicts next to Terra/Sol — the P1/P2 disagreement measurement continues on real listings at no extra cost and, unlike the canary, under **production routing**. If Steven would rather ship v5 to production only once, with its final policy, do P1/P3 first (~1.5 h of photos, then possibly a 4-day re-canary if P1(b)/(c) is chosen — and if so, run that canary with the production model map so the next comparator measures what production produces). Recommendation: the dark cutover now.
- **P4** Sol rules / probe: post-cutover, replayable offline; 2/31 flips, all single-child, fail-safe.
- **P5** comparator / cluster tooling: before the *next* canary (changes nothing in estimates). Add to the list: hash `vlm_client.py` / `analyzer_cli.py` / `failure_taxonomy.py` and record `RENOVATION_VLM_BUDGET_GUARD` in the freeze; state the baseline of any cross-replica percentage (F8); report within-replica residuals.
- **P6 item 3** (single-photo opportunity drivers): with P2.
- **Price calibration** (bath high floors, provisional bands): separate thread, needs a renovator.
- **Other open threads** (not gates, unowned): Terra batch-quality review (12-image / 20-condition batches), acceptance-rate audit (are the 282 rejections real hallucinations?).
- **Packet open items now closed:** comparator exit code (rc=0, byte-identical); suite (2337 on the working tree, caveat in §2); provenance (freeze == HEAD on all 27 hashed files; unfrozen files by telemetry).

## 7. Order of operations once C1–C5 are closed

1. LM Studio up at the FE's `LM_STUDIO_URL`; embeddings sidecar up (real-POST; worker startup refuses `ready` otherwise).
2. Edit `renointel-prod/.env` per C3 (one edit); restart the worker: `npm run worker -- --premium`.
3. `new` smoke (C4) → inspect artifact by hand; `current` smoke → inspect; restore `new`.
4. First real listing: v5 `sol_call.model`, `provenance.architecture_mode`, `catalog_version`, `model_routing` (2c `gpt-5.6-terra` `explicit_override`); publication-gate log clean; FE renders the property (kind runbook step 5).
5. Observation (C5): publication-gate rejections, stale-id failures, writer failures, FE rendering, **plus** v5 job failures by category/code and daily ledger spend (`scripts/show_daily_token_spend.py --root <RENOVATION_TERRA_USAGE_ROOT>`).
6. Then the design-doc order of work (§9): photo passes A / A′ / B → P1/P3 decisions → P4 → P5 → FE-adoption planning.

## 8. Evidence appendix

- Hash verification: `sha256sum` over the five packet files and over the 24 `code_files` + manifest + model map + catalog named in `input_freeze.json` (27/27 equal, twice). `git diff --name-only 07ee112 HEAD` lists 12 files; 9 are in the freeze, 3 are not (F6).
- Suite: `pytest_full.log` in the session scratchpad — `2337 passed, 6 skipped, 30 warnings in 34.05s`, `EXIT=0`.
- Comparator: `comparator_rerun.log` / `comparator_rerun_report.json` in the scratchpad — `EXIT=0`, `release_ready: true`, SHA-256 `37694749810b9215417710937871f75f4e8538f35c5a911d31903b3713fc161b`.
- Production state: all 1,264 `renointel-prod/artifacts/*/*/photo_intel.json` stamp `catalog_version "2.1"`, no `ontology_version`, no v5; newest `redfin_11214304/20260803_224131_4dd6dbc1` (`created_at 2026-08-04T03:44:59Z`); `model_routing` 2a/2b/2c/2d `gpt-5.6-terra` (effort none, `explicit_override`), 2f `gpt-5.6-sol` (`run_override`).
- Canary state: 36/36 `catalog_version 3.1`, `ontology_version observation-kind-v2`, `run.pipeline_mode publish`; v4 at root, v5 under `analysis_debug` (shadow); per-listing telemetry in `photo_intel_debug.json` `analysis_debug.token_usage.per_pass`.
- Worker: `renointel-prod/package.json` `"worker": "node --import tsx ... scripts/analysis-worker.ts"`; `scripts/load-env.ts` `dotenv.config({override:true})`; `lib/services/persistentPythonWorker.ts` `{...process.env}`; `lib/analysis/workerMode.ts` `--premium` → `buildWorkerModelOverrides`; `logs/scheduled-worker.log` shows `npm run worker` + dotenv injecting 41 keys. FE `.env` keys (no values printed): `ANALYZER_PERSISTENT`, `RV_ROOT`, `RV_PY`, `OPENAI_MODEL`, `OPENAI_PASS_2F_PRIORITY_*`, `LM_STUDIO_URL`; none of `RENOVATION_*` / `KIND_ONTOLOGY_VERSION`. Backend `.env`: `RENOVATION_TERRA_MODEL=gpt-5.6-terra`, `RENOVATION_SOL_MODEL=gpt-5.6-sol`, `RENOVATION_TERRA_DAILY_TOKEN_CEILING=` (blank → default 2.5M).
- Code paths: `tools/pipeline_config.py` (`resolve_kind_ontology`, `resolve_renovation_architecture`, `resolve_renovation_sol_model`, guard resolvers); `tools/artifact_writers.py` ~432–510 (`new` job-fatal; classification-only refusal ~574–583; seam placement ~1222–1234); `tools/renovation_architecture/runtime.py` (init refusals; failed-envelope path ~351–381); `usage_guard.py::maybe_reserve_vlm_call` (guard off → `None`); `review_pipeline.py` ~239–255 (Terra hook ledger); `tools/failure_taxonomy.py` (`AuthoritativeEstimateIncomplete` unmapped → `provider`); `app/api/artifacts/[...file]/route.ts` (raw artifact route).
- Adversarial pass: workflow `session9-cutover-refute`, 8 refuters + completeness critic, ~806k tokens; full text in the session scratchpad `refute_results.txt`.

## 9. Execution log — closing C1–C5 (2026-08-21/22)

Steven's decisions (2026-08-21 evening): C1 **written waiver**; C2 **accept 1 & 2, 3 provisional**; C3 **guard ON**; also commit the untracked Session 9 operational files and harden the smoke tool. Out of scope by his choice: the production flip / first real listing (his `npm run worker -- --premium`), the FE branch merge, committing `reports/` or the canary root. Plan: `~/.claude/plans/i-need-you-to-idempotent-aho.md`.

| step | what | result |
|---|---|---|
| Preflight | `git rev-parse HEAD` / tracked mods; stray shell env; UTC day; ledger dir; LM Studio `GET 127.0.0.1:1234/v1/models`; embeddings real-POST `127.0.0.1:8081` | HEAD `6640ebb`, no tracked mods; no stray `OPENAI_*`/`LM_STUDIO_*`/`RENOVATION_*`/`KIND_ONTOLOGY_VERSION` exports; UTC 2026-08-22T03:37Z (fresh quota day); `renointel-prod/artifacts/.renovation_architecture/` absent; LM Studio lists `unsloth/qwen3.6-27b@q6_k`; embeddings **connection refused** → started the documented CPU-only sidecar `scripts/start-embeddings-server.ps1` in the background (llama-server PID 34564, 22:38 CT); pre-existing `node` processes (PIDs 9968/24600/30984/31208/32632) left untouched |
| C1 | design doc §9 log rows `Cutover — Task 4A (C1)`; sign-off block appended to `reports/kind_cutover_canary_final_20260807.md` | done (waiver; `approved_headline_deltas` left `{}`) |
| C2 | design doc §9 log rows `P6 Tier-2 sign-off (1)/(2)/(3)` | done |
| C5 | design doc §9 log row `Cutover — observation (C5)`; runbook §4 observation rule + rollback ladder | done |
| Runbook | §3 rewritten: worker = `npm run worker -- --premium` on `renointel-prod/.env`; full env block; verifier as the §3 check; Sol-ledger caveat; §4 ladder | done |
| Smoke tool | `scripts/run_worker_smoke.py`: `--dotenv PATH` (default backend `.env`), `--routing {canary,production}` (production = FE `--premium`: 1a/2a/2b/2c/2d → `OPENAI_MODEL` effort none, 2f → `OPENAI_PASS_2F_PRIORITY_MODEL` medium, `modelRoutingProfile standard`, `detectionBackend rv`, no concurrency override), effective-env print, full `photo_intel_path` print, non-zero exit when `job_done` arrives without a published artifact; new `scripts/verify_renovation_artifact.py` (payload-only: publication gate on full + slim, envelope validator, placement per mode, Terra/Sol call models, `model_routing`) | done (`py_compile` clean) |
| Verifier self-test | 36/36 canary pairs `--expect-mode shadow --expect-terra-model gpt-5.6-terra --expect-sol-model gpt-5.6-sol`; one wrong-mode run | **36/36 PASS (13/13 checks each); wrong-mode (`new` on a shadow artifact) FAIL 6/10, exit 1** |
| Suite | `pytest -q -p no:cacheprovider` after script changes | **2337 passed, 6 skipped, exit 0** (31.9 s) |
| C3 | `renointel-prod/.env` backed up to `.env.bak.20260821` (4,463 B); block appended (6 keys, LF, no key collisions, `ISSUE_CATALOG_PATH` absent confirmed); both files gitignored by `.env*` | done — worker NOT restarted (Steven's flip) |
| C4 new | `run_worker_smoke.py --mode new --out artifacts_canary/worker_smoke_20260821 --dotenv C:/Users/Steven/IntelliJProjects/renointel-prod/.env --routing production --property redfin_10806500` (detached, 22:45 CT) → verifier `--expect-mode new --expect-terra-model gpt-5.6-terra --expect-sol-model gpt-5.6-sol` | **smoke exit 0** (worker ready in mode=new; 7 images; `success: true`); artifact `artifacts_canary/worker_smoke_20260821/redfin_10806500/20260821_224550_2a49de45/photo_intel.json`; **verifier PASS 16/16**: root v5 complete in slim + full, no private copy, provenance `new` / 3.1 / `observation_kind_v2`, publication gate clean on full + slim, v4 present; v5 `terra_calls` 5 × `gpt-5.6-terra`, `sol_calls` 1 × `gpt-5.6-sol`; `model_routing` 1a/2a/2b/2c/2d `gpt-5.6-terra` effort none `explicit_override`, 2f `gpt-5.6-sol` medium `run_override`; effective env printed = FE `.env` values (guard 1, usage root = FE artifacts). Guard path exercised end-to-end: no `VlmBudgetGuardConfigError` |
| C4 current | same command with `--mode current` (detached, 22:48 CT) → verifier `--expect-mode current` | **smoke exit 0** (worker ready in mode=current; `success: true`); artifact `…/redfin_10806500/20260821_224822_ba312d86/photo_intel.json`; **verifier PASS 8/8**: v4-on-v2 only, no v5 at root or under `analysis_debug` in either file, publication gate clean, stamps 3.1 / observation-kind-v2; routing identical to the `new` run. L1 rollback proven |
| Ledger | `show_daily_token_spend.py --root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts` (UTC day 2026-08-22) | after `new`: terra 76,291 (66 reservations) / sol 18,763 (5) — **equal to the artifact's `analysis_debug.token_usage` (total 95,054: 1a 10,894 · 2a 12,865 · 2b 6,264 · 2c 9,318 · 2d 21,411 · 2f 14,714 on Sol · condition review 15,539 · Sol review 4,049; 71 calls, 0 failed)**; after both: terra 133,240 / 2,500,000 (122 reservations), sol 31,192 / 250,000 (8). Ledger files live only at `<FE artifacts>/.renovation_architecture/{terra,sol}_usage.sqlite3`; none under the smoke out dir (usage-root override honoured). Note: 7-image listing, so ~95k/listing here vs the canary's ~220k median for 30-image listings |
| Smoke caveats | what the smokes do not prove | the FE's per-day 2f allocation (`OPENAI_PASS_2F_PRIORITY_LIMIT`, DB-backed) is not reproduced — the smoke always uses the priority (Sol) model for 2f; FE rendering of a `new`-mode artifact not exercised (kind runbook step 5 — do on the first real listing) |
| Commit | operational set (3 docs, 4 scripts, 6 tests, the edited kind-canary report), no push | **`df35f23`** on `renovation_architecture_rework` (14 files, +5,073) — this SHA line is the one edit left uncommitted on purpose |

**State at hand-off (2026-08-21 ~22:55 CT):** production is **staged, not flipped** — `renointel-prod/.env` carries the `new` block; no worker was restarted by this session (pre-existing `node` processes untouched; the last logged worker launch was plain `npm run worker`). The embeddings sidecar started for the smokes (llama-server PID 34564, CPU-only, `scripts/start-embeddings-server.ps1`) is still running. Smoke artifacts: `artifacts_canary/worker_smoke_20260821/` (gitignored). The flip is Steven's: `npm run worker -- --premium` in `renointel-prod`, then `scripts/verify_renovation_artifact.py --artifact <first artifact> --expect-mode new --expect-terra-model gpt-5.6-terra --expect-sol-model gpt-5.6-sol`, confirm the FE renders it, and start the 7-day observation (C5). Watch the Sol ledger on day one (`show_daily_token_spend.py --root <FE artifacts>`).
