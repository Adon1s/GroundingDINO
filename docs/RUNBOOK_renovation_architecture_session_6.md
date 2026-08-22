# Renovation architecture Session 6 canary and cutover runbook

This runbook executes the approved Session 6 sequence without changing the
frozen 18-property corpus or replacing the existing benchmark platform. The
default backend remains `current` until the generated report says
`release_ready: true` and every manual review item has a written explanation.

## Safety and prerequisites

- Obtain explicit authorization to send the frozen property images and stored
  property metadata to the configured embeddings, Terra, and Sol services.
- Confirm `configs/kind_ontology_canary_manifest.json` remains `status: frozen`
  with exactly 18 properties.
- Confirm the embeddings real-POST preflight succeeds.
- Confirm `.env` or the worker environment supplies the intended
  `RENOVATION_TERRA_MODEL`, `RENOVATION_SOL_MODEL`, output caps, and API key.
- Use a new output root. Never mix a rerun with a different input/code freeze.
- The existing Terra daily budget guard remains mandatory. A ceiling failure is
  a release stop, not permission to skip reviews or publish a partial estimate.
- Session 8 added a Sol daily budget guard (250,000 tokens/day, override via
  `RENOVATION_SOL_DAILY_TOKEN_CEILING`; ledger shares
  `RENOVATION_TERRA_USAGE_ROOT`). One replica uses ~155k Sol tokens, so two
  same-day replicas breach the default ceiling — run the replicas on separate
  days via `--replicate 1` / `--replicate 2`, exactly as the flag was designed
  for. A Sol ceiling denial fails the property closed (category `quota`), same
  posture as Terra.
- Session 9 added the choke-point budget guard: the coordinator sets
  `RENOVATION_VLM_BUDGET_GUARD=1`, and EVERY OpenAI call (upstream scene
  passes 2a/2b/2c/2f included, not just the architecture's review calls)
  debits the shared per-model daily ledgers. Terra's ceiling is 2,500,000
  tokens/day (input+output combined — the OpenAI free daily allowance),
  override via `RENOVATION_TERRA_DAILY_TOKEN_CEILING`; setting that override
  is an explicit paid-day decision and is recorded in the coordinator's
  summary JSON. Per-pass token telemetry is persisted in every debug
  artifact under `analysis_debug.token_usage`, so the rerun self-documents
  its spend by pass and model.

## 1. Run two frozen shadow replicas

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_renovation_architecture_canary.py `
  --out artifacts_canary\renovation_session6_<date>
```

The coordinator writes one deterministic `input_freeze.json`, then invokes the
existing canary driver twice. Each debug artifact contains both independently
computed `renovation_estimate_v4` and private
`analysis_debug.renovation_estimate_v5`, produced from the same upstream issue
lane. The two replica directories use the same frozen image paths, image
SHA-256 values, stored property metadata, code hashes, catalog hash, policy and
prompt versions, and model routing.

For a retry of one failed property, keep the same output root and freeze:

```powershell
.\.venv\Scripts\python.exe scripts\run_renovation_architecture_canary.py `
  --out artifacts_canary\renovation_session6_<date> `
  --only <property_key>
```

Completed real artifacts are skipped; an empty failed run directory is not
treated as completion.

### 1a. Session 9 free-paced schedule (strictly inside the free quota)

One replica costs ~3.98M Terra tokens ≈ 1.6 free-quota days, so the paid
rerun spreads across ~4 calendar days with zero paid overage. Decisions
(Steven, 2026-08-17): strictly free pacing; the guard is canary-only opt-in.

- **New output root and fresh `input_freeze.json` are mandatory** (Session 9
  changed hashed files; every pre-Session-9 checkpoint is dead), and **no
  code edits between day 1 and the final day** — the freeze verifier refuses
  to resume under a drifted code hash.
- Daily loop, same `--out` every day: days 1–2 `--replicate 1`, days 3–4
  `--replicate 2` (the Sol guard forces separate days per replica anyway).
  Completed properties are skipped on resume; Terra/Sol checkpoints make a
  re-attempted property's completed review units free (upstream passes are
  re-bought — that is what the stop margin exists to avoid).
- Each day the run continues until the pre-property gate reports the daily
  Terra budget near its ceiling and exits **rc=3** (a clean stop: nothing
  failed, nothing to review). Resume after the UTC midnight rollover
  (19:00–20:00 ET) with the same command. The margin is tunable via the
  inner driver's `--terra-stop-margin` (default 500,000 ≈ 2.3× the measured
  221k/listing mean).
- A property that instead fails closed with category `quota` (the guard
  denied mid-listing) is also fine: it is NOT skipped on resume and re-runs
  from scratch tomorrow.
- **Day 1 runs at a reduced 2,000,000 ceiling** as a tracking-validation day:
  set `RENOVATION_TERRA_DAILY_TOKEN_CEILING=2000000` so that even if our
  metering under-counts by up to ~25%, the real 2.5M free allowance is not
  breached. At the end of day 1 compare `show_daily_token_spend.py` against
  the OpenAI dashboard; once they agree, drop the override for the remaining
  days to keep the schedule near 4 days (at a 2M ceiling it is ~5–6).
  Expect our ledger to read **slightly high**, never low: successful calls
  settle to the provider's exact `total_tokens`, but a failed call or a
  response without usage retains its conservative reservation. Our ledger
  also counts only this run — any other OpenAI work the same UTC day shows
  on the dashboard but not here.
- Set `--terra-stop-margin` at the coordinator if the default 500,000 needs
  changing. Do it before day 1: the freeze hashes the driver scripts, so a
  later edit blocks resume.
- Check spend at any time:

```powershell
.\.venv\Scripts\python.exe scripts\show_daily_token_spend.py --root artifacts_canary\renovation_session6_<date>
```

- Exit codes: 0 = replica complete, 1 = property failures (inspect before
  resuming), 2 = preflight/manifest refusal, 3 = clean daily budget stop.
- If a run ever completes a property whose artifact carries a quota-failed
  v5 envelope, the driver stops loudly: it means the guard env did not reach
  the analyzer. Delete that property's artifact directory and re-run after
  fixing the environment — never leave it, or resume will treat it as done.

## 2. Generate the comparison and review template

```powershell
.\.venv\Scripts\python.exe tools\compare_renovation_architecture_cutover.py `
  --run artifacts_canary\renovation_session6_<date>\run_1\candidate `
  --run artifacts_canary\renovation_session6_<date>\run_2\candidate `
  --freeze artifacts_canary\renovation_session6_<date>\input_freeze.json `
  --report reports\renovation_architecture_session6_canary_<date>.json `
  --review-template reports\renovation_architecture_session6_reviews_<date>.json
```

The comparison fails closed until every scope change, package change, headline
delta above 15%, and semantic run-to-run difference has a review entry with an
`approved` or `accepted` decision and a non-empty explanation. Review IDs are
bound to the input freeze; a review file from another run cannot authorize
cutover.

The report includes:

- schema/publication/reconciliation/accepted-work gates for all 18 properties;
- v4 versus v5 condition, scope, package, ledger, and headline summaries;
- semantic run-to-run stability with run-specific IDs normalized away;
- Terra call, estimate-unit, condition-attribution, and listing token rows;
- median, p90, p95, p99, and worst-case token and phase-latency distributions;
- projected listings/day at median, upper-percentile, and worst observed Terra
  usage under the 2.5-million-token daily ceiling.

After completing the written review file, rerun the comparator with:

```powershell
.\.venv\Scripts\python.exe tools\compare_renovation_architecture_cutover.py `
  --run artifacts_canary\renovation_session6_<date>\run_1\candidate `
  --run artifacts_canary\renovation_session6_<date>\run_2\candidate `
  --freeze artifacts_canary\renovation_session6_<date>\input_freeze.json `
  --reviews reports\renovation_architecture_session6_reviews_<date>.json `
  --report reports\renovation_architecture_session6_canary_<date>.json
```

Do not select `new` unless the command exits zero and the JSON report contains
`release_ready: true`.

## 3. Select the new backend and restart

The persistent worker is the `npm run worker` process in `renointel-prod`
(`scripts/analysis-worker.ts` → `lib/services/persistentPythonWorker.ts`, which
spawns `tools.analyzer_server` with `{...process.env}`). Its environment is
`renointel-prod/.env` (dotenv, override) — the backend `.env` is **not** read on
that path. Note that this flip is also the kind-ontology v2 cutover (production
artifacts are catalog 2.1 / `legacy_v1`; `new` refuses to start without
`observation_kind_v2`) — see the decision record
`docs/DECISION_renovation_architecture_session9_cutover_20260821.md` §3 F1 / §5 C1.

Set all of the following atomically in `renointel-prod/.env` (decision record
§5 C3; budget guard ON per Steven, 2026-08-21):

```text
KIND_ONTOLOGY_VERSION=observation_kind_v2
RENOVATION_ARCHITECTURE_MODE=new
RENOVATION_TERRA_MODEL=gpt-5.6-terra
RENOVATION_SOL_MODEL=gpt-5.6-sol
RENOVATION_VLM_BUDGET_GUARD=1
RENOVATION_TERRA_USAGE_ROOT=C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts
```

`ISSUE_CATALOG_PATH` must stay unset. Without `RENOVATION_SOL_MODEL`, Sol falls
back to `OPENAI_MODEL` (the Terra model) and startup does **not** refuse — only
empty model names are rejected — so the model lines are mandatory.

Restart the worker with `npm run worker -- --premium` (in `renointel-prod`).
`--premium` is what routes 1a/2a/2b/2c/2d to `OPENAI_MODEL`
(`lib/analysis/workerMode.ts`); without it the upstream passes run on local
Qwen — inputs the canary never measured. Startup must refuse readiness if the
catalog, Terra model, Sol model, or token caps are invalid. Run one end-to-end
listing smoke:

```powershell
.\.venv\Scripts\python.exe scripts\run_worker_smoke.py --mode new `
  --out artifacts_canary\worker_smoke_<date> `
  --dotenv C:\Users\Steven\IntelliJProjects\renointel-prod\.env --routing production
.\.venv\Scripts\python.exe scripts\verify_renovation_artifact.py `
  --artifact <printed photo_intel_path> --expect-mode new
```

The verifier checks mechanically what this section used to list by hand:

- root `renovation_estimate_v5` is a schema-valid `complete` envelope;
- its provenance `architecture_mode` is `new` and catalog version is `3.1`;
- `renovation_estimate_v4` is still present and valid for comparison;
- no private v5 envelope is present under `analysis_debug`;
- publication, reconciliation, and arithmetic audits are clean
  (`validate_publication_payload` on the full debug payload and the slim one);
- and it prints the Terra/Sol call models and `model_routing` — expect Sol
  `gpt-5.6-sol`, 1a/2a/2b/2c/2d `gpt-5.6-terra` (effort none), 2f `gpt-5.6-sol`.

Ledgers (architecture hooks and the choke point share them) live under
`<RENOVATION_TERRA_USAGE_ROOT>/.renovation_architecture/`; read them with
`scripts/show_daily_token_spend.py --root <RENOVATION_TERRA_USAGE_ROOT>`. With the
guard ON, production Pass 2f on `gpt-5.6-sol` also debits the 250k Sol ledger
(2f ≈ 24–58k tokens/listing; the FE sends the first
`OPENAI_PASS_2F_PRIORITY_LIMIT` runs per day to Sol), so ~5–6 premium listings
per day can reach the Sol ceiling before Terra's ~11; a denial fails the listing
closed (`quota`, queue pause — visible, not silent). Knobs:
`OPENAI_PASS_2F_PRIORITY_LIMIT` (FE) or an explicit
`RENOVATION_SOL_DAILY_TOKEN_CEILING` (a paid decision).

## 4. Rollback smoke and the rollback ladder

Change only `RENOVATION_ARCHITECTURE_MODE=current`, restart the worker, run the
smoke with `--mode current`, and verify with
`scripts/verify_renovation_artifact.py --expect-mode current`: unchanged v4
result and no v5 key at either the root or the private debug location. Do not
delete or rewrite already-valid v5 artifacts.

Rollback ladder:

- **L1** `RENOVATION_ARCHITECTURE_MODE=current` → v4 on the v2 catalog publishes,
  no v5 (this smoke proves it).
- **L2** `KIND_ONTOLOGY_VERSION=legacy_v1` on this build → **nothing publishes**
  (classification-only is a publication kill switch).
- **L3** point `RV_ROOT` / `ANALYZER_CLI` in `renointel-prod/.env` at the pinned
  worktree `../rv-legacy-a1972cf` → true v1 (catalog 2.1) publication.

Observation (decision record §5 C5): 7 calendar days from enablement,
operational invariants only — publication-gate rejections, stale-id failures,
writer failures, FE rendering, v5 job failures by category/code, daily ledger
spend; roll back after three consecutive ontology/resolver/writer failures (v5
job-fatal failures count). Immediate rollback or release stop is required for
any mixed/invalid artifact, accepted-work loss, double counting, model mutation
of condition/work truth, semantic publication of an operational failure, silent
Terra ceiling overrun, or repeated writer failure.
