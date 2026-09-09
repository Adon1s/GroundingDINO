# Session 8 — Terra ceiling denominator (decision memo)

Date: 2026-08-17. Branch: `renovation_architecture_rework`.

## The question

The implementation plan says Terra has "a hard service budget of 2.5 million
tokens per day" with a ~25,000 tokens/listing reference (≈100 listings/day).
The canary measured architecture condition review alone at **51,082
tokens/listing mean** (919,478 for 18 properties, range to 82,754). What the
2.5M/day ceiling denominates decides the real listings/day capacity and what
the usage guard must meter.

## Verified production routing (the fact that was missing)

From the newest production artifact's `model_routing` block
(`renointel-prod` corpus, run `20260803_224131_4dd6dbc1`, 2026-08-03):

| pass | production model | source | canary model (frozen replica 1) |
|---|---|---|---|
| 1a | local qwen | standard_default | local qwen |
| 2a | **gpt-5.6-terra** | explicit_override | gpt-5.6-terra |
| 2b | **gpt-5.6-terra** | explicit_override | gpt-5.6-terra |
| 2c | **gpt-5.6-terra** | explicit_override | gpt-5.6-terra |
| 2d | **gpt-5.6-terra** | explicit_override | local qwen ← differs |
| 2f | **gpt-5.6-sol** | run_override | gpt-5.6-terra (medium) ← differs |

So in production the Terra *service* already serves four upstream passes
(2a/2b/2c/2d), and 2f rides the **Sol** service. Two consequences:

1. The canary's "upstream 2a/2b/2c/2f = 3,059,408 tokens (~170k/listing)"
   is a proxy, not an exact production measurement: production adds 2d load
   to Terra (local in the canary, so unmeasured there) and moves 2f's load
   to Sol.
2. **The new 250k/day Sol guard meters only architecture package review,
   but production 2f consumes the same `gpt-5.6-sol` service.** The Sol
   budget has the same denominator question as Terra's.

## The measurement gap (found while verifying routing)

**Production records no token telemetry.** The newest production artifacts
carry `model_routing` (which model served each pass) and `max_output_tokens`
config, but no input/output/total token counts anywhere — token accounting
was added by the architecture sessions and exists only inside the v5
envelope. Consequences:

- The all-service number can only be estimated from the canary proxy, and
  the canary's routing differs from production in two passes (2d, 2f).
- Nothing today can answer "how much Terra/Sol did production actually use
  yesterday" from artifacts. Whatever denominator is chosen, the ceiling is
  currently unenforced and unobserved for every pass outside the
  architecture.

Sizing production 2f (the Sol-service consumer): across the last 40
production runs it ran on 39, averaging **6.8 package-verification attempts
per listing**, and 2f is image-bearing. The architecture's own Sol usage is
**one text-only call per listing at ~8,575 tokens**. So production 2f is
very plausibly the dominant consumer of the `gpt-5.6-sol` service, and the
new 250k/day Sol guard meters the smaller half. This is unmeasured, not
speculative-by-choice — no telemetry exists to settle it.

## The two readings

| reading | tokens/listing | listings/day under 2.5M |
|---|---|---|
| Architecture-only (condition review) | 51,082 | **~48** |
| All-Terra-service (2a/2b/2c/2d + condition review) | ~200–230k (canary proxy; 2d unmeasured) | **~10–12** |

The plan's own words support the service reading ("hard **service** budget"),
but its 25,000/listing arithmetic only ever matched the architecture-only
reading. The two candidate positions:

- **Architecture-only**: the guard keeps metering exactly what it meters
  today; upstream passes are governed by the separate API-usage thread. The
  2.5M ceiling then does NOT protect the Terra service from upstream load,
  and actual service usage runs ~4-5× what the guard sees.
- **All-service**: the ceiling means what it says; either the upstream Terra
  passes start debiting the same ledger (a change at the analyzer/pass
  boundary, not the architecture layer — follow-up work), or listings/day is
  planned at ~10–12 until the upstream API-reduction work (handed off from
  Session 8) cuts the 170k/listing upstream share.

## Recommendation

Treat 2.5M/day as the **service** ceiling (that is what a provider budget
is), plan capacity at ~10–12 listings/day for now, and keep the
architecture guard's meter unchanged this session — widening the meter to
upstream passes is analyzer-boundary work that belongs with the
API-reduction thread, where per-pass measurement will exist anyway. Revise
the plan's 25k/listing reference to the measured 51,082 (architecture
condition review) either way.

**The prerequisite either reading needs: per-pass token telemetry in the
production pipeline.** Without it the ceiling cannot be observed, let alone
enforced, and the choice below is being made on a proxy. Adding it is small,
mechanical work at the analyzer boundary (the VLM client already exposes
`usage_stats`; the architecture's own settle-before-parse pattern is the
model) and should be the first item of the API-reduction thread — arguably
before the paid rerun, so the rerun measures production-shaped routing
rather than canary-shaped routing.

## Decision (Steven, 2026-08-17): all-service — settled by fact, not preference

The 2.5M/day is the **OpenAI free daily token allowance** (input + output
combined) on the account for the Terra model's tier; the ~250k/day Sol
figure is the same program's allowance on the Sol model's tier. An
account-level provider quota counts everything routed to the model, so the
denominator is all-service by construction. Steven observed roughly the full
2.5M consumed by Terra during canary replica 1 alone (our measured billed
total on the Terra model for that replica was 3,978,886 = 3,059,408 upstream
2a/2b/2c/2f + 919,478 condition review — a run that exceeds the free
allowance either spills into paid tokens or spanned a UTC day boundary; the
dashboard's free-usage counter also cannot display more than the allowance,
so "~2.5M used" reads as "exhausted", not as the true total. Per-pass
telemetry, Session 9, reconciles this).

Consequences:
- Planning capacity is ~10–12 listings/day at current per-listing usage;
  raising it is the API-reduction thread's job (post-cutover).
- One canary replica costs more than one day of free Terra quota. The rerun
  must either spread across days (guard + resume, Session 9) or knowingly
  pay overage.
- The Sol 250k/day allowance likewise covers production 2f plus architecture
  package review; the Session 8 Sol guard meters only the latter today.
- The follow-up that makes the ceiling real is owned by the next session;
  its decision handoff is `docs/HANDOFF_renovation_architecture_token_budget.md`
  (verified seam facts, hazards, option space — no design decided).
