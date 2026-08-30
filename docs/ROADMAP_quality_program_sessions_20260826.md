# Roadmap — quality-program implementation sessions (B–H)

Written 2026-08-26 · backend `renovation_architecture_rework` · commissioned by
Steven from `docs/PROPOSALS_output_quality_improvements_20260826.md` (QP ids
and locked decisions S1–S9 live there; §9 option calls recorded in
`docs/DESIGN_renovation_architecture_decision_packets.md`). **Full roadmap
approved 2026-08-26 with go/no-go gates between waves — any gate can stop the
program.**

**Amended 2026-08-27:** Steven narrowed wave 1 to QP3 only after reviewing the
combined C+D+E handoff. QP3 is implemented and replay-validated; wave 1 is
complete. QP6, QP5, and QP4 are deferred backlog items, not prerequisites for
Session F or G. See `docs/RESULT_quality_wave1_qp3_20260827.md`.

**Amended 2026-08-28:** the 46-card mechanism-vs-perception re-tag is complete.
Its formal result is inconclusive, so Session F must run rubric-only,
coarsened-only, and combined wording arms. See
`docs/RESULT_retag_mechanism_vs_perception_20260828.md`.

**Amended 2026-08-28 (later):** a **label repair (v1.1)** is inserted before
Session F. The re-tag's free-text notes contain outright reversals of the
reviewer's own v1 calls ("Crap this one I messed up", "This is my miss, terra
is correct"), and the v1 vocabulary collapses three judgments — is the
condition there, is the exact claim accurate, is the work worth doing — into
one label. Running three arms at ~1.8M Terra tokens each against labels known
to contain reversals would buy an expensive answer to the wrong question.
Design, pre-committed mappings and tooling:
`docs/DESIGN_label_v1_1_adjudication.md`. v1 stays frozen and published; v1.1
is an overlay used for development only.

**Amended 2026-08-29:** the label repair is **complete** (135/135) and its
result is `docs/RESULT_label_v1_1_20260829.md`. **Session F does not run as
chartered.** The canary scoreable population no longer supports the wording
arms: `hard_false_billed` fell 11 → 3 (a paired test tops out at p = 0.125,
unreachable at any outcome) and `dirB_wording_recovery` is 0. Corrected rates
are not publishable either — every v1.1 band overlaps its v1 band, and each
hard-false rate rests on one uniform card. The bottleneck is **review coverage,
not tokens**; the next four actions cost ~3.5 h of review and zero tokens. The
severity/threshold follow-up is independently actionable now.

**Direction call commissioned 2026-08-29.** Because F is blocked and the FE
hold's stated expiry can no longer be reached, this roadmap is **provisional
past L2**. A session is commissioned to decide the path forward — including the
deferred architectural items (work-item identity / S9, split authority, package
warrant), whether tuning belongs upstream of Terra at all given the 35%/51%
observation churn, and what the benchmark architecture should be. Context and
open questions: `docs/HANDOFF_quality_program_direction_20260829.md`. Sessions
F, G and H should be re-planned against its outcome, not executed as written.

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
- Observation window: **CLOSED early by Steven 2026-08-27** (outcome recorded
  in `docs/DECISION_renovation_architecture_session9_cutover_20260821.md` §10;
  no rollback trigger fired). The freeze on estimate behaviour, prompts, and
  catalog is lifted — every "post-window" gate below is now open, including
  wave-1 shipping and live harness runs. Live runs still debit the shared
  ledgers; check daily headroom before a 0.9M-token pass.
- `scripts/replay_renovation_architecture.py` stays structurally provider-free
  — never add live calls to it.
- Do not delete the dormant per-item Pass-2f scaffolding; never bump
  `PASS_2F_PROMPT_VERSION` without extending
  `PASS_2F_ISSUE_INDEPENDENT_PROMPT_VERSIONS`.

## Sessions

| session | scope | when | depends on |
|---|---|---|---|
| **B — Harness + tooling** | QP1 re-decide harness build (no live runs) + scoring vs the frozen labels; P5 tooling fixes 1–5 (+track 6); commit the uncommitted review tooling; QP8 inspection-lane audit + scene-identity measurement (both read-only) | starts now (window-safe) | — |
| **C — QP3 / wave 1** | **Complete:** opportunity-only bedroom/living modernization candidates remain reviewable but do not apply; children retain ledger coverage | closed 2026-08-27 | — |
| **D — QP5** | **Deferred outside wave 1:** split authority and derived-package contracts require a separate design | backlog | — |
| **E — QP4** | **Deferred outside wave 1:** multi-bath work-item identity must be designed before package expansion | backlog | — |
| **L — v1.1 label repair** | **Complete 2026-08-29** (135/135, integrity clean, 0 leaks). Result: `docs/RESULT_label_v1_1_20260829.md`. Canary hard-false 11 → 3; the failure mass is now misnamed 49% / trivial 27% / absent 24%. Every corrected band overlaps v1's; do not publish 2.66% | closed 2026-08-29 | B |
| **L2 — coverage repair** | The four zero-token review actions that unblock everything else: (1) ~15 min adjudicate the 20 unasked canary dirB cards — the only place coarsening can be tested, and the missing dirB *loss* population; (2) ~25 min a blind repeat set covering the `absent` boundary (current repeats have zero coverage of it); (3) ~2.5 h expand the canary uniform arm (200-card enriched draw) to fix the 2.392 pp/card leverage; (4) ~45 min a second reviewer on a 40–60 card overlap — the only thing that converts "re-adjudicated" into "verified". Then re-derive B's thresholds against the directional floors (4.05% billed / 23.1% dirB) | next | L |
| **F — Terra v2 tuning** | **BLOCKED on L2 (2026-08-29): does not run as chartered** — see `docs/RESULT_label_v1_1_20260829.md`. Re-scope against the corrected classes before any live pass. **Re-tag input complete:** the formal result is inconclusive (`docs/RESULT_retag_mechanism_vs_perception_20260828.md`). Required wording arms are **rubric-only, coarsened-only, and combined**, each against a same-prompt control; fine mechanism wording is not retired before the scorecard. Structured-fields and remove-observations remain separate harness hypotheses. Score against `--labels reports/labels_v1_1.json`, declaring coarsening arms with `--coarsened-variant-root`. Run one variant-day at a time against the shared ledger (~1.8M Terra tokens per comparison), prepare any catalog wording through the decisions file, and deliver the variant scorecard + canary plan | blocked | B, L, **L2** |
| **G — Wave-2 canary** | Fresh freeze, two replicas carrying the winning Terra v2 + catalog wording; comparator vs the **stored canary replayed through the completed QP3 wave-1 policy** (not the raw session9 baseline); prompt-version cutover | after C + F | C, F |
| **H — Round-2 review + measurement** | Extend review tooling per QP-M's two uniform arms (rejected conditions for recall; all package candidates for warrant precision) on a **separate queue**; Steven reviews (~2.5–3h); analysis vs the acceptance gates with sample-size math | after G | G |

Sequencing notes:

- The original C→D→E combined charter is retired. Only QP3 landed in wave 1;
  QP6/QP5/QP4 require new authorization and their own evidence/design gates.
- F is independent of the deferred package-policy backlog.
- The wave-2 comparator **must** diff against the wave-1-replayed baseline
  produced by Session C's QP3 replay, or QP2's effect is conflated with wave-1
  policies.

## Go/no-go gates

1. **After B:** harness demonstrates the 6.4% noise floor on a control-arm
   run design; P5 landed; audits reported. If the harness can't beat the noise
   floor, wave 2 is rethought before any tokens are spent on variants.
2. **After C — closed 2026-08-27:** QP3 replay gated exactly 8 candidates,
   preserved all 119 candidates/decisions, and passed every reconciliation
   audit. This is the final wave-1 baseline; no QP4 ship gate remains in wave 1.
2b. **After L — closed 2026-08-29, gate NOT met.** The label set is complete
   (125/125), but the scorecard can no longer separate an arm from the noise
   floor on its primary leg: `hard_false_billed` = 3, paired best-attainable
   p = 0.125. Per the pre-committed rule the fix is a larger scoreable
   population, **not** a looser threshold, so **no Terra tokens are spent until
   L2 lands**. Two scorer defects must also be fixed first: there is no dirB
   *loss* population, and `load_population_v1_1` silently excludes production.
3. **After F:** variant scorecard — hard-false flips vs supported-card losses
   vs dirB recovery vs wording recovery, each vs control. `trivial_billed` is
   reported but never decides an arm (severity is a separate lever). No winner
   → no canary.
4. **After G:** canary green + comparator reviewed against the wave-1
   baseline.
5. **After H:** measured rates vs the (sample-size-checked) acceptance gates →
   the FE-adoption discussion (and the S8 per-item-2F decision) reopens with
   data.

## Standing items outside the sessions

- **The mechanism-vs-perception re-tag is complete** (46/46, 2026-08-28):
  `docs/RESULT_retag_mechanism_vs_perception_20260828.md`. The inconclusive
  result sends rubric-only, coarsened-only, and combined wording arms to
  Session F; it does not retire fine mechanism wording. The separately planned
  structured-fields and remove-observations hypotheses remain in scope. A
  severity/threshold follow-up is open because `too_trivial` crossed its gate.

- FE stays on v4 until gate 5 (locked with S1–S9); the hold burns the binding
  Sol quota via production 2f — its expiry is the FE-adoption call.
- Thumbnail-ingest fix thread runs independently (starts 2026-08-29+, own
  handoff); nothing here depends on it or duplicates it.
- Price calibration remains a separate future thread (needs a renovator).
