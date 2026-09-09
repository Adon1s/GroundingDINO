# Handoff — token budget, telemetry, and the affordable rerun (decision handoff)

Date: 2026-08-17. Branch: `renovation_architecture_rework` (Session 8
complete, uncommitted). Companions:
`docs/HANDOFF_renovation_architecture_session_8.md` (what landed),
`docs/analysis/session8_terra_budget_denominator.md` (the denominator
decision and its evidence), `docs/RUNBOOK_renovation_architecture_session_6.md`
(the rerun procedure this all feeds).

**This document records understanding and observations only. It makes no
implementation decision; the follow-up session owns the design.** The goal
state, in Steven's words: get back to a state where the canary can be tested
again — without scope creep.

## The problem

The migration's remaining gate is the paid two-replica canary rerun, and it
is no longer affordable to run naively. The 2.5M/day Terra ceiling turned
out to be Steven's **OpenAI free daily token allowance** (input + output
combined) on the Terra model's tier; the Sol model's tier has ~250k/day
free. Canary replica 1 consumed **~3.98M billed tokens on the Terra model**
(3,059,408 upstream 2a/2b/2c/2f + 919,478 architecture condition review) —
more than a full free day for one replica. Nothing outside the
architecture's own review calls enforces or even records this spend, so a
rerun today would silently blow through the free quota mid-run.

Two distinct gaps, one situation:

1. **Enforcement**: the ceiling is real (a provider quota) but unenforced
   for ~77% of the spend. The implementation plan itself lists "silent
   continuation after the daily Terra ceiling is reached" as a
   rollback/stop condition — the current state violates the plan's own
   mandate the moment a run crosses the quota.
2. **Observability**: per-pass token usage is measured in memory and thrown
   away, so nobody can say where the 3.98M goes by pass, and the
   all-service capacity number (~10–12 listings/day) remains a proxy.

## Measured facts (trust these; sources inline)

- Replica-1 Terra-model spend: 3,978,886 total = 3,059,408 upstream +
  919,478 condition review (Session 6 handoff telemetry + v5 envelopes).
  Per-pass split of the upstream 3.06M: **unrecoverable** — it was never
  persisted (verified: no token fields anywhere in canary or production
  artifacts outside the v5 envelope).
- Sol spend: 154,361/replica (architecture package review, one text-only
  call per listing, mean ~8,575). Two same-day replicas breach 250k — the
  Session 8 guard now enforces exactly this slice.
- Steven's dashboard showed ~2.5M Terra for replica 1. The free-usage
  counter cannot display more than the allowance, so "~2.5M used" reads as
  *exhausted*, not as the true total; the overflow was either paid tokens
  or a UTC-day straddle. Telemetry would reconcile this.
- Production routing (verified from production artifact
  `20260803_224131_4dd6dbc1`): 2a/2b/2c/**2d** on `gpt-5.6-terra`, **2f on
  `gpt-5.6-sol`** (~6.8 image-bearing calls/listing across the last 40
  runs). The canary's frozen map differs: 2d local, 2f on Terra. So the Sol
  free quota is shared with production 2f, and the Session 8 Sol guard
  meters only the architecture's (smaller) share.

## Verified seam facts (Session 8 did this reconnaissance; don't redo it)

1. **`tools/vlm_client.py` is the single choke point for every OpenAI
   call**, and it already does thread-safe, task-local per-pass token
   attribution: `_run_with_telemetry` (~line 199) sets the pass key via
   ContextVar; `_record_usage` (~line 225) accumulates
   input/cached/output/total/calls/duration into
   `usage_stats["per_pass"][key]`. The measurement exists — in memory only.
2. **Nothing persists it.** `tools/analyzer_cli.py` (~line 500) renders a
   console pass table from `usage_stats["per_pass"]`; that is where the
   data dies. Artifacts carry `model_routing` (which model served each
   pass) and `max_output_tokens` config, but zero token counts outside the
   v5 envelope.
3. **The existing guards are layer-local.** Terra:
   `review_pipeline.py:86-125` reserves/settles per condition-review call
   against `TerraUsageLedger`, and the debit is contract-visible
   (`TerraCall.budget_debited_tokens`). Sol: the Session 8 hook in
   `sol_review.py` does the same against `SolUsageLedger`. Any
   wider metering must not double-debit these calls, and must not break
   the contract field.
4. **Reservation sizing is not one-size.** Terra's 25k reservation floor is
   tuned for large batched review calls. Upstream calls are small and
   numerous (a 19-photo listing makes ~4 calls per photo, concurrently);
   inflated per-call floors under photo-level concurrency would spuriously
   exhaust a ledger long before real spend does. The Session 8 Sol guard
   already set the precedent of a per-model floor (10k vs 25k).
5. **Denial semantics have an established pattern**: reserve → call →
   settle-to-provider-truth-before-parse; a denial raises through
   `PassExecutionError` → failure category `quota`; the property fails
   closed, never a partial artifact.
6. **Resume machinery already exists.** The canary driver skips completed
   artifacts, retries per property (`--only`), stages replicas
   (`--replicate 1|2`), and Terra/Sol checkpoints make a re-attempted
   property's completed units free. A run stopped at a ceiling is
   resumable at property granularity with no wasted spend.
7. **Unverified**: the `reset_usage_stats` cadence in the long-lived
   analyzer worker (`analyzer_server.py`) — whether `usage_stats` is
   per-property or accumulates across a session. Any per-property
   persistence design must check this first.
8. `usage_stats["per_pass"]` does not record the **model** per pass; the
   `model_routing` list does. Joining them is trivial at write time but is
   a join, not a field read.

## Constraints the design must respect

- **Steven's scope guard**: no prompt, batching, model-map, or pass-routing
  changes. Every known token-*reduction* lever (prompt-cache restructuring
  — `cached_input_tokens` was 0 on all observed calls; batch composition;
  moving passes to local models) changes model inputs and therefore
  demands its own quality re-comparison. Those levers belong to the
  post-cutover API-reduction thread; this session's job is to make the
  rerun *possible and observed*, not cheaper.
- **Comparability**: the rerun's model map stays frozen as-is (2d local,
  2f on Terra). Realigning to production routing would change upstream
  detections and sever comparability with both replica-1's replay anchors
  and v4.
- The two replicas already must run on separate days (Sol guard).

## The rerun arithmetic (for whatever schedule the session designs)

- Per replica ≈ 3.98M Terra tokens ≈ 1.6 free-quota days.
- Two replicas ≈ 8M ≈ 3.5 free-quota days → ~4 calendar days if strictly
  free, fewer if overage is paid (mini-tier overage is dollars-cheap; the
  free-vs-paid tradeoff is Steven's, per run).
- Sol: ~155k/replica against 250k/day free — fine at one replica/day.

## The option space as observed (mechanical consequences only, no preference)

- **Choke-point guard in `vlm_client`** (meter every OpenAI call by model
  against the matching ledger): makes the ceiling real for all spend;
  must solve the double-debit hazard (skip-flag set by the architecture
  hooks, vs pass-key filter, vs relocating the architecture hooks — the
  last breaks `budget_debited_tokens` unless re-plumbed) and the
  reservation-sizing problem (observation 4). Denial mid-listing fails the
  property closed; the resume machinery absorbs it.
- **Persist per-pass usage into artifacts** (a `token_usage` block in
  `photo_intel_debug.json`, joined with `model_routing`): pure
  observability, no behavior change; needs the reset-cadence answer
  (observation 7). Makes the rerun self-documenting and retires the proxy
  numbers. Orthogonal to, and independently valuable without, the guard.
- **Scheduling only, no code**: split each replica into property batches
  by hand (`--properties`/`--only`) sized to ~2.5M/day using the known
  per-listing average (~221k Terra). Zero code, available today; but
  nothing stops an overrun mid-batch (per-listing variance was 2×), the
  plan's explicit-stop mandate stays unmet, and the per-pass split stays
  unknown.
- **Pay the overage**: run each replica straight through and accept paid
  tokens past 2.5M. Compresses the rerun to the 2-day Sol-imposed minimum;
  abandons the free-quota framing; still leaves enforcement/observability
  gaps for production later.
- These compose: e.g. telemetry + guard + free-quota pacing, or telemetry
  + paid compression. Only the scheduling-only option leaves the plan's
  stop-condition mandate unsatisfied.

## Context the new session needs (read in this order)

1. This handoff.
2. `docs/HANDOFF_renovation_architecture_session_8.md` — current state of
   the branch, what is uncommitted, the closed decisions.
3. `docs/analysis/session8_terra_budget_denominator.md` — the budget
   evidence and the all-service decision.
4. `docs/RUNBOOK_renovation_architecture_session_6.md` — the rerun this
   enables.
5. Code: `tools/vlm_client.py` (choke point),
   `tools/renovation_architecture/usage_guard.py` (ledger pattern),
   `review_pipeline.py:86-125` + `sol_review.py` fresh-call branch (the
   hooks that must not double-debit).

**Not needed**: the source architecture docx
(`Revised_Renovation_Scope_and_Estimate_Architecture.docx`). It is the
design-philosophy document for the estimate architecture (layers, decision
ownership, invariants) and contains no operational/token/budget content —
verified 2026-08-17. It matters again only for sessions that change what
Terra/Sol are asked or the architecture's decision boundaries, which this
session's scope guard forbids anyway.

## Related but explicitly not this session's scope

Post-cutover API-reduction thread (prompt caching, batch composition, pass
rerouting, the 25k→51k capacity gap); widening production's Sol accounting
to cover 2f; Terra batch-quality review; acceptance-rate audit. See the
Session 8 handoff's hand-off list.
