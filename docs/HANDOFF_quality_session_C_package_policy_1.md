# Handoff — Session C: package policy I (QP3 + QP6)

Written 2026-08-27 by Session B · backend `renovation_architecture_rework` ·
commissioned by `docs/ROADMAP_quality_program_sessions_20260826.md`
(Session C) · proposals = `docs/PROPOSALS_output_quality_improvements_20260826.md`
(QP3, QP6; option calls recorded in
`docs/DESIGN_renovation_architecture_decision_packets.md` §9). Inherit the
roadmap's hygiene + simplicity guardrail blocks in full. **Post-window
session: work may start any time, but no estimate-behaviour change ships
before the observation window closes (end of 2026-08-28).** Both changes are
offline-replay-validated (wave 1); no live provider calls are needed.

Sequencing: C→D→E are sequential because all three touch
`tools/renovation_architecture/package_candidates.py` / `reconciliation.py` /
`validators.py` in the one live tree. Session F (Terra variants) runs in
parallel on the Session B harness and touches none of these files.

## 1. Deliverable 1 — QP3: interior-modernization application gate

A deterministic rule: a `bedroom_modernization` / `living_modernization`
candidate whose drivers are ALL opportunity-kind (exactly:
`proposed_treatment` ∈ {`opportunity_driver_with_corroboration`,
`opportunity_driver_with_multiphoto_corroboration`} — any defect/degradation
driver forces `package_driver`, `rehab_packages.py:2077-2088`) is **not
applied**; its children bill standalone at their exact allowances. Kitchens
and bathrooms are untouched. Policy phrasing, not a special case: wholesale
modernization of a bedroom/living room is not a warranted package on
style-only evidence (P1 census: 0/5 interior-modernization warranted vs
kitchen 4/1, exterior 6/4 — fragile N, but demotion-to-standalone is not
deletion; the harm cap is the tier uplift).

Implementation constraints (verified in the proposals doc — re-verify at
implementation):

- **v5-path only.** `_build_package_candidate` is shared legacy code; an
  ungated change alters live v4/FE output outside any replay, and v4's
  post-2f re-tier (`rehab_packages.py:2401-2412`) would bypass a
  candidate-build-time cap anyway. Gate in
  `package_candidates.build_package_candidates` or at application
  eligibility in `reconciliation.py`.
- If done at reconciliation: new application/ledger reason codes, and **all
  four eligibility recompute sites must agree** (application builder,
  validators' recompute, replay, comparator expectations).
- In the reviewed slice the rule is a no-op on every human-warranted
  style-only card (P01/P12 are already partial tiers); it binds exactly on
  P10/P11-class packages.

**Alternative to present to Steven before landing:** the generic per-family
"demote one step down the escalation chain" cap (P1 option d as drafted).
Rejected in the proposals because bedroom/living have no partial tier
(refresh→full_rehab, so the demotion is full→refresh ≈ −60%/package), a cap
would deepen bathroom D4 under-billing, and kitchens are a near-no-op — but
the choice between one systematic rule and a scoped one is Steven's, never
encoded silently.

## 2. Deliverable 2 — QP6: single-child rule (Steven picks the scope)

Do not emit single-child **interior repair** candidates (bedroom/living
repair — the class where both measured Sol flips occurred; Session B's
comparator now detects these directly and found exactly those 2). Their work
items bill standalone via the existing `no_covering_package` ledger path (no
contract change — verified). Keep exterior singles: both human-warranted
single-child P1 cards were exterior_repair.

**The blanket-vs-scoped choice is Steven's (roadmap + S4 record):**

- (a) **Type-scoped (recommended in the proposals):** suppress only
  bedroom/living repair singles. Kills 100% of the measured Sol-variance
  class; zero goal-1 contribution; preserves exterior packaging the evidence
  supports.
- (b) **Blanket:** suppress every single-child candidate. Simpler rule;
  destroys the two human-warranted exterior singles; bigger Sol-quota
  relief.

Side effect either way: the whole-home display-only rollup consumes the same
candidate list and will shrink — decide (and record) whether it should
aggregate suppressed singles anyway. Coordination note for Session E: QP4's
bathroom clones are often single-child; whatever scope is picked must not
suppress them (type-scoped (a) does not; blanket (b) needs a carve-out).

## 3. Validation (per-change replay + comparator)

Replay is free and offline; run **one replay per change plus the combined
run** (a single combined run cannot attribute a regression):

1. Baseline: `scripts/replay_renovation_architecture.py --root
   artifacts_canary/renovation_session9_20260818/run_1 --label c_baseline`
   on the pre-change working tree (or reuse a stored baseline replay).
2. QP3 alone → `--label c_qp3`; QP6 alone → `--label c_qp6`; both →
   `--label c_combined`. Report package count/dollar deltas per change; the
   replay holds stored Sol decisions fixed (gated/suppressed candidates
   drop their stored decisions — expect `stored_decisions_unused` /
   `candidates_dropped_no_stored_decision` sanitization counters to rise by
   exactly the gated set; anything else is a bug).
3. Comparator on the combined result (scratch path, never the audited
   `reports/session9_*`): the Session B comparator now keys v4 packages by
   `package_id`, carries dollars + 2f verdicts, and has a Sol-flip section —
   use it to eyeball that only the intended candidates disappeared and the
   children re-appear as standalone scope rows with identical dollars.
4. Suite green (`.venv\Scripts\python.exe -m pytest`); tests for the gate
   (opportunity-only drivers → not applied; any package_driver → untouched;
   kitchens/bathrooms untouched) and for the single-child scope choice.

## 4. Constraints

- No estimate-behaviour change ships in-window (to end 2026-08-28); code
  may land behind the replay validation but the ship commit waits.
- Frozen inputs untouchable (roadmap hygiene block): canary root, frozen
  review reports, audited `reports/session9_*`.
- Never `git add .` — the repo root still carries untracked scratch; stage
  named files.
- Wave-1 baseline note for Session E: the **combined C+D+E replay is the
  wave-1 baseline** the wave-2 canary diffs against; keep each session's
  replay outputs under a labelled `analysis_*` dir so E can reproduce the
  chain.
- QP4 gate-evidence note for Session E (decision-packets §7 item 6 is now
  an erratum): redfin_125779232 and redfin_80877597's "duplicated v4 rows"
  were per-surrogate bathroom expansion clones — distinct bathrooms. Both
  properties count as QP4 under-billing gate evidence in the §10 scoring,
  not as benign-cleanup examples.

## 5. Exit criteria

1. QP3 gate + QP6 rule landed v5-path-gated with tests; Steven's scope
   choices (QP3 scoped-vs-generic, QP6 type-scoped-vs-blanket, whole-home
   rollup treatment) recorded in the §9 decision log.
2. Per-change + combined replays run with deltas reported; comparator
   scratch run reviewed.
3. Suite green; session commit(s) made.
4. Session D's handoff written (`docs/HANDOFF_quality_session_D_package_policy_2.md`):
   QP5 split materialization — sub-candidate ids, decision provenance,
   reason codes, the four eligibility recompute sites, validators, replay
   plan — carrying real results from this session, not guesses.
