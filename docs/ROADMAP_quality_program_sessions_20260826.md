# Roadmap — quality-program implementation sessions (B–H)

Written 2026-08-26 · backend `renovation_architecture_rework` · commissioned by
Steven from `docs/PROPOSALS_output_quality_improvements_20260826.md` (QP ids
and locked decisions S1–S9 live there; §9 option calls recorded in
`docs/DESIGN_renovation_architecture_decision_packets.md`). **Full roadmap
approved 2026-08-26 with go/no-go gates between waves — any gate can stop the
program.**

Each session is one bounded change with its own validation gate. Sessions are
sequential only where they share files. A session implements its charter and
nothing more; each session ends by writing the next session's handoff (repo
convention). Session B's handoff exists now
(`docs/HANDOFF_quality_session_B_harness_tooling.md`); later handoffs are
written by the preceding session so they carry real results, not guesses.

## Simplicity guardrails (every session inherits these)

- Smallest change that closes the QP. No new abstractions or pipeline stages
  beyond what the QP names.
- Light variant first: QP2's rules-only prompt is the default; structured
  fields is a harness A/B arm, not a commitment.
- One-off measurements stay one-off (the scene-identity check is a read-only
  count, not a new audit stage, unless its numbers demand more).
- Any blanket-vs-scoped rule choice is flagged to Steven, never encoded
  silently.

## Hygiene rules (every handoff inherits these)

- `RV_ROOT` is the live production working tree: no `git checkout` of other
  branches; use a worktree for anything that needs one.
- Frozen review inputs are untouchable: `reports/review_queue.json`,
  `reports/review_verdicts.jsonl`, all `reports/session9_*` outputs, the
  canary root `artifacts_canary/renovation_session9_20260818`. Never rebuild
  the frozen queue; new review work uses a separate queue.
- Tests: `.venv\Scripts\python.exe -m pytest` (bare `python` does not resolve
  the venv). Suite green before commit; commit per session.
- Budget guard stays ON; all live provider calls debit the shared ledgers.
- Observation window: open until end of **2026-08-28**. Building code is fine;
  no estimate-behaviour/prompt/catalog change ships in-window and no live
  provider calls run in-window (C5 monitors daily ledger spend).
- `scripts/replay_renovation_architecture.py` stays structurally provider-free
  — never add live calls to it.
- Do not delete the dormant per-item Pass-2f scaffolding; never bump
  `PASS_2F_PROMPT_VERSION` without extending
  `PASS_2F_ISSUE_INDEPENDENT_PROMPT_VERSIONS`.

## Sessions

| session | scope | when | depends on |
|---|---|---|---|
| **B — Harness + tooling** | QP1 re-decide harness build (no live runs) + scoring vs the frozen labels; P5 tooling fixes 1–5 (+track 6); commit the uncommitted review tooling; QP8 inspection-lane audit + scene-identity measurement (both read-only) | starts now (window-safe) | — |
| **C — Package policy I** | QP3 interior-modernization application gate + QP6 single-child rule (present blanket vs interior-repair scoping; Steven picks); v5-path gating; per-change replays + comparator | post-window | — |
| **D — Package policy II** | QP5 split materialization (sub-candidate ids, decision provenance, reason codes, the four eligibility recompute sites, validators); replay | after C | C |
| **E — Multi-bath expansion** | QP4 full build: per-surrogate re-pricing + child partitioning, strict per-surrogate qualifying rule, id scheme, cap persistence, replay Sol-inheritance extension; ship-gate scoring vs the §10 human targets (ship iff exactness ≥ 6/17 and over ≤ 1/17); **combined C+D+E replay = the wave-1 baseline** | after D | C, D |
| **F — Terra v2 tuning** | Variant design (light rules / structured fields / remove-observations), harness A/Bs with a same-prompt control arm (~1.8M Terra tokens per comparison, one variant-day at a time against the shared ledger), catalog wording edits prepared via the decisions file; deliverable = variant scorecard + canary plan | post-window; parallel with C–E (harness-side overrides only — production `terra_review.py` untouched) | B |
| **G — Wave-2 canary** | Fresh freeze, two replicas carrying the winning Terra v2 + catalog wording; comparator vs the **stored canary replayed through the wave-1 policies** (not the raw session9 baseline); prompt-version cutover | after E + F | E, F |
| **H — Round-2 review + measurement** | Extend review tooling per QP-M's two uniform arms (rejected conditions for recall; all package candidates for warrant precision) on a **separate queue**; Steven reviews (~2.5–3h); analysis vs the acceptance gates with sample-size math | after G | G |

Sequencing notes:

- C→D→E are sequential because all three touch
  `tools/renovation_architecture/package_candidates.py` /
  `reconciliation.py` / `validators.py` in the one live tree.
- F runs in parallel with C–E because its variants are harness-side overrides;
  nothing it does touches the modules C–E edit.
- The wave-2 comparator **must** diff against the wave-1-replayed baseline
  produced by Session E, or QP2's effect is conflated with the wave-1
  policies.

## Go/no-go gates

1. **After B:** harness demonstrates the 6.4% noise floor on a control-arm
   run design; P5 landed; audits reported. If the harness can't beat the noise
   floor, wave 2 is rethought before any tokens are spent on variants.
2. **After E:** wave-1 replay results (package/dollar deltas per change +
   combined; QP4 gate table vs §10 targets). QP4 ships only on its gate.
3. **After F:** variant scorecard — hard-false flips vs supported-card losses
   vs dirB recovery, each vs control. No winner → no canary.
4. **After G:** canary green + comparator reviewed against the wave-1
   baseline.
5. **After H:** measured rates vs the (sample-size-checked) acceptance gates →
   the FE-adoption discussion (and the S8 per-item-2F decision) reopens with
   data.

## Standing items outside the sessions

- FE stays on v4 until gate 5 (locked with S1–S9); the hold burns the binding
  Sol quota via production 2f — its expiry is the FE-adoption call.
- Thumbnail-ingest fix thread runs independently (starts 2026-08-29+, own
  handoff); nothing here depends on it or duplicates it.
- Price calibration remains a separate future thread (needs a renovator).
