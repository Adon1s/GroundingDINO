# PROGRAM STATE — kind ontology rework (read this first)

2026-08-08 · branch `pass_2c_redesign` · suite 1709 passed / 6 skipped.

Orientation only. Every detail lives in the linked handoff — if this file and a
handoff disagree, **the handoff wins**.

## Where we are

The `defect|upgrade` → `defect|degradation|modernization` rework is **built
end-to-end and sits behind one env var**, `KIND_ONTOLOGY_VERSION` (default
`legacy_v1`). Production still runs the v1 catalog and stops after Pass 2c.

**Cutover is no longer blocked by the ontology.** It is blocked by **run-to-run
estimate variance under gpt-5.6-terra**: identical inputs produce estimates
differing by tens of thousands, because observation generation (2a/2b/2c) is
unstable and flat room allowances turn one flipped observation into a whole
$10–20k line. gpt-5.6 exposes no temperature parameter. This surfaced from the
catalog 3.1 canary re-measure, whose deltas didn't match the known retirement.

## Status

| | |
|---|---|
| Tasks 1–3 (contract, catalog+2d, downstream consumers) | done |
| Task 4A (selector, publication gate, canary, comparator) | implemented, **sign-off not clean** |
| Catalog 3.1 (layout items retired) | done; `11000447` still breaches, `25809814` gained a variance breach |
| Estimate-variance investigation | scoped only — `HANDOFF_estimate_variance_gpt56.md` |
| **Pass 2a prompt ablation** | **next task** — `HANDOFF_pass2a_variance_ablation.md` |
| 4A sign-off decision | open: accept a noise band, or stabilize with repeat runs |
| 4B historical corpus migration | not started — `HANDOFF_kind_ontology_4A_to_4B_active_reanalysis.md` |
| 4C legacy deletion | blocked on 4B acceptance — `..._4A_to_4C_history_retirement.md` |

Per-task detail: `HANDOFF_kind_ontology_task1.md` / `task2.md` / `task3.md`.
Task 3's handoff has the authoritative per-kind behavior table.

## Facts that cost money if forgotten

- **The 42 split successors do not have authored prices.** They inherit their v1
  parent's economics verbatim (`inherited_from_split_parent`), and the
  `KIND_MULT` degradation/modernization values are likewise temporary. Real
  pricing is a separate project. Do not read current estimates as intentional.
- **The 2a ablation is measurement only, and has a stop-gate.** Run the
  attribution leg first (replay downstream from one frozen 2a capture); if
  issues still flip, the variance is in 2c and tuning 2a won't fix it — stop and
  report. Exclude `pass_2c_kind_v3` and origin's `2e1b9ed` from any measurement;
  both alter 2c/photo handling and confound it.
- **A shipped 2a change invalidates the current 18-property canary deltas.** The
  4A sign-off record must state which 2a prompt its numbers were measured under.
- **Never hand-edit** `issue_catalog_kind_v2.json`, the migration manifest, or
  the audit report — edit `tools/catalog_migrations/kind_v2_decisions.json` and
  re-run `scripts/migrate_catalog_kind_v2.py`. A parity test pins them.
- **Embeddings sidecar `:8081` `/health` lies** — 200 while every call fails.
  Probe with a real POST (`scripts/run_kind_canary.py::_preflight_embeddings`).
- **Canary reruns skip properties that already have an artifact** — archive,
  don't delete.
- **Stored artifacts are `legacy_v1` and immutable.** The migration manifest is
  an audit record, not a runtime alias table; the FE reads v1 through a
  version-aware adapter.
- **Task 1's ontology is fixed.** A decision that seems to need a kind
  reinterpreted is a blocking contradiction to raise, not to patch.
- If Pass 2f enters scope, it runs on **Sol**.

## Known-open, not lost

Two v1-inherited catalog overlaps recorded as `task3_flags` in the decisions
file (staging/clutter, general/bathroom mold); `trees_or_vegetation_too_close`
flagged as hallucination-prone and probably not worth costing. The
`catalog-resolution-v2` benchmark scores 1.000 on holdout and is saturated — it
only catches gross regressions now, and it never measured composed 2c×2d error
because kinds were gold-supplied. Replaying real stored observations is the
informative next step, but it inherits the variance problem, so it follows 2a.
