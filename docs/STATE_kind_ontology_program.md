# PROGRAM STATE — kind ontology rework (read this first)

2026-08-08. Single entry point for the `defect | upgrade` →
`defect | degradation | modernization` program. This is an **index and status
board**, not a spec: each section points at the authoritative handoff. If this
file and a handoff disagree, the handoff wins and this file is stale.

Branch: `pass_2c_redesign`. Suite: **1709 passed / 6 skipped**.

---

## TL;DR — where we actually are

The ontology rework is **built end-to-end and sitting behind one env var**
(`KIND_ONTOLOGY_VERSION`, default `legacy_v1`). Production still runs the v1
catalog and stops after Pass 2c.

**What is blocking cutover is no longer the ontology.** It is
**run-to-run estimate variance under gpt-5.6-terra**: identical inputs and
identical code produce estimates differing by tens of thousands of dollars, so
the 18-property canary's per-property BREACH labels partly measure noise rather
than v1-vs-v2. Steven's call: **freeze 2c work, attack variance at Pass 2a.**

Immediate next task: **Pass 2a prompt ablation**
(`docs/HANDOFF_pass2a_variance_ablation.md`).

---

## Task ladder — status

| Task | What it did | Status |
|---|---|---|
| **1** | Pass 2b/2c three-kind contract, atomic observations, fail-closed validation | ✅ done |
| **2** | v2 catalog + Pass 2d strict exact-kind retrieval + resolution benchmark | ✅ done |
| **3** | Every downstream consumer migrated to three kinds; Pass 2e revived | ✅ done |
| **4A** | Cutover machinery: selector, publication gate, canary, comparator | ✅ implemented, **sign-off not clean** |
| **Catalog 3.1** | Retired 2 layout-priced items (pre-cutover blocker) | ✅ done, 5 canaries re-measured |
| **Variance investigation** | Discovered during 3.1 re-measure; scoped only | 🔶 **scoped, not started** |
| **2a ablation** | Prompt ablation to reduce variance | 🔶 **next task** |
| **4B** | Historical corpus migration (~1,264 artifacts) | ⬜ not started |
| **4C** | Legacy two-kind surface deletion | ⬜ blocked on 4B acceptance |

---

## What each task actually implemented

### Task 1 — classification contract
`docs/HANDOFF_kind_ontology_task1.md`

Hard replace of the Pass 2b/2c contract. 2b emits atomic single-claim
observations; 2c returns indexed kind/exclude decisions over the three kinds
plus a six-reason excluded lane, validated fail-closed (no silent coercion).
Prompt provenance added (`pass_2b_atomic_v2`, `pass_2c_kind_v2`). 8/8 holdout
gates on Terra.

Three binding rules from Steven's review, still in force:
flatwork cracking is degradation unless heaved/trip-edged; water stains are
defects (moisture evidence) while cosmetic staining is degradation; 2c
classifies what the text **asserts** — groundedness belongs to 2a/2b.

### Task 2 — catalog + resolution
`docs/HANDOFF_kind_ontology_task2.md` · commits `0098d7f`..`27dd240`

- `tools/issue_catalog_kind_v2.json` created as a **separate, non-publishable**
  file. 107 legacy items → 130: 32 unchanged, 52 reclassified, 4 narrowed,
  19 split into 42 successors.
- Generated deterministically from `tools/catalog_migrations/kind_v2_decisions.json`
  by `scripts/migrate_catalog_kind_v2.py`, which also emits the audit-only
  107-entry manifest `2.1_to_3.0.json` and an audit report. **Never hand-edit
  the catalog/manifest/audit — a parity test pins them.**
- `evaluate_kind_routing` became a singleton exact-kind route (widening
  retired). Closed the bug where an empty/unknown `allowed_kinds` silently
  searched the whole catalog.
- `pipeline_mode` added; the dormant v1 block (shadow lane, `_label_to_kind`,
  kind coercion) deleted.
- Second `write_photo_intel` guard on catalog `publication_status`.
- `benchmarks/catalog-resolution-v2` built; **10/10 holdout gates** —
  recall@5 and final accuracy 1.000 vs legacy 0.870.

Task 2 made **no pricing or product change** by construction.

### Task 3 — downstream consumers
`docs/HANDOFF_kind_ontology_task3.md` · commits `0de9332`..`342dd95`

Every consumer is three-kind-native. The per-kind behavior table in that
handoff is the authoritative reference for how each kind routes. Highlights:
`costing.KIND_MULT` now **raises** on unpriced kinds instead of silently
defaulting to 1.0; `estimate_scope` no longer matches the kind label as text
(the `"modernization"` token collision); `property_summary_pass` went
three-kind (`summary_v1` → 2.0); **Pass 2e revived** and runs 2c→2d→2e in
benchmark mode.

### Task 4A — cutover machinery
`docs/HANDOFF_kind_ontology_4A_to_4B_active_reanalysis.md` · commits
`352160c`..`a8558ef`

- **v2 catalog became `publishable`.** The 42 split successors no longer defer
  pricing — they **inherit their v1 parent's economics verbatim**
  (`pricing_status: inherited_from_split_parent`). This is an explicit
  **temporary bridge**, not authored prices; a dedicated pricing project
  replaces them.
- `KIND_MULT`: degradation 1.0 / modernization 0.6 (temporary bridge);
  `upgrade` 0.6 retained so legacy artifacts still read.
- **`KIND_ONTOLOGY_VERSION` selector** (`tools/pipeline_config.py`) derives
  catalog *and* pipeline depth atomically so they cannot disagree.
  `legacy_v1` → v1 catalog + `classification_only`; `observation_kind_v2` →
  v2 catalog + `publish`. Under v2 an `ISSUE_CATALOG_PATH` override is a hard
  error. Invalid values raise at import.
- `tools/publication_gate.py` — permanent write-boundary validation.
- 18-property canary + comparator; stale-id fail-loud.
- FE version-aware adapter landed in renointel-prod (`kind_ontology_v2_compat`).

**4A migrated no historical artifacts.** Stored v1 artifacts are immutable and
the FE reads them through the adapter.

### Catalog 3.1 — layout retirement
`docs/HANDOFF_layout_item_audit.md` · commits `58c073e`, `a8fcac7`

Retired `bathroom_layout_modernization_opportunity` and
`layout_modernization_opportunity`: **layout/spatial claims must not price.**
The bathroom item drove unapproved headline-delta breaches on 5 of 18 canaries.
Introduced a `retired` change type (deprecated, zero successors). Catalog is now
**3.1, 128 items (52 defect / 36 degradation / 40 modernization)**.

The 5 affected canaries were re-run under 3.1. Outcome: `10806500` resolved
both breaches; `11077450` and `25809814` resolved high bounds; **`25809814`
picked up a new low-bound breach (fresh-run variance)**; **`11000447` remains a
breach on non-layout drivers.** That unresolved pair is what exposed the
variance problem.

---

## The blocker: run-to-run estimate variance

`docs/HANDOFF_estimate_variance_gpt56.md` — **investigation-scoping only, no fix
designed.**

Identical pipeline, identical inputs, materially different estimates. Never seen
with local Qwen (temperature turned well down); the canary routes 2a/2b/2c and
2f to `gpt-5.6-terra`, and the gpt-5.6 API **exposes no temperature parameter** —
there is no dial to turn down.

Measured (artifacts preserved in `artifacts_canary/candidate_archive_pre_v31/`
vs `artifacts_canary/candidate/`), after backing out the known −$25k retirement:
`11000447` ≈ **+$45k** churn, `10806500` ≈ **−$40k**, `25809814` low bound swung
+8% → −44%.

**Mechanism:** variance is in *observation generation* (2a/2b/2c), not
resolution or costing. 2d is fixed-vector embedding retrieval, 2e is rule-based,
costing v4 is pure arithmetic, 1a is local Qwen. Costing then **amplifies**:
most items are flat room allowances, occurrence counts don't scale cost, so a
single observation appearing or vanishing flips an entire $10–20k allowance on
or off.

**Consequences:** canary deltas in the ±20–40% band carry substantial noise, and
post-cutover the same listing re-analyzed twice could hand a user a range
differing by tens of thousands.

---

## NEXT TASK — Pass 2a prompt ablation

`docs/HANDOFF_pass2a_variance_ablation.md` — read the variance handoff first.

**Hypothesis:** `PASS_2A_USER_PROMPT = "What stands out here to a renovator"` is
salience-framed — it asks the model to *sample* what's notable, which is
inherently unstable. Inventory framing should force enumeration toward a
repeatable set.

Staged variants: (1) baseline, (2) whole-image inventory wording — one wording
change only, (3) variant 2 + visible-evidence boundary, **only if** variant 2
produces unsupported claims.

**Do the attribution leg first:** replay 2b→2c→2d→costing k=3 from ONE frozen 2a
capture. **Decision gate — if frozen-2a replays still flip cost-bearing issues
heavily, the variance lives in 2c extraction and tuning 2a won't fix it: stop
and report rather than running variants 2/3.**

Fixed inputs: `redfin_11000447` and `redfin_10806500` (7 photos each, both have
archived run pairs for free historical comparison). k=3 per variant. Metrics:
supported recall vs hand-gold, unsupported observations, run-to-run Jaccard of
resolved catalog-id sets, cost-bearing issue flips, **and bias** (inventory
wording will likely emit more observations, and flat allowances mean each
marginal one can flip a $10–20k line ON — report the headline shift, don't judge
it).

**Scope guards:** measurement only, no shipped prompt change comes out of it.
Branch `pass_2c_redesign` @ `cdefc2b` — **not** `pass_2c_kind_v3`, **not**
origin's `2e1b9ed` (both alter 2c/photo handling and confound the measurement).
2c prompts stay frozen. A later shipped 2a change invalidates the current
18-property canary deltas as-measured — the 4A sign-off record must state which
2a prompt its numbers were measured under.

---

## Then: 4A sign-off decision

Per the variance handoff, step 5: either **explicitly accept a noise band** in
`approved_headline_deltas`, or **stabilize the re-measured properties with
repeat runs** before approval. This decision is currently open and is what gates
production enablement.

Cutover config (`configs/kind_ontology_cutover.json`): default `legacy_v1`,
production switch `KIND_ONTOLOGY_VERSION=observation_kind_v2`, rollback is the
same switch back, **7-day observation window** starting at production
enablement, 18 canary properties minimum / 3 per stratum, thresholds —
unresolved-rate increase ≤0.02, estimate-coverage drop ≤0.05, unapproved
headline delta ≤0.15.

---

## Then: 4B — historical corpus migration

`docs/HANDOFF_kind_ontology_4A_to_4B_active_reanalysis.md`

~1,264 stored `photo_intel.json` in `renointel-prod/artifacts/`, all
`catalog_version: "2.1"` / `legacy_v1`. Three migration paths per issue:

1. **Deterministic same-id reprojection** — 88 of 107 legacy ids are
   unchanged/reclassified/narrowed. Metadata restamp + derived-view recompute,
   **no model calls**.
2. **Stored-description re-resolution for splits** — issues on the 19 deprecated
   split parents need re-resolution against successors.
3. (third path per the handoff)

Untracked prototype exists: `tools/backfill_kind_v2.py`.

**Gotcha:** stored `photos[*].photo.image_path` values point at the dead
`IntelliJProjects\realtorvision\...` path — resolve images via the current repo
name, not the stored absolute path.

Corpus survey 2026-08-06: 849 property dirs, 840 with runs, 821 with both a
completed run and images on disk. Projection mix: 888 reprojected / 349 native /
27 needs_reanalysis.

---

## Then: 4C — legacy retirement

`docs/HANDOFF_kind_ontology_4A_to_4C_history_retirement.md`

**Cannot start until 4B is accepted** and every ledger item is proven
reader-free. Both status inputs in that doc are currently **pending** (7-day
observation outcome; 4B acceptance) and must be updated before 4C begins.
Deletes the two-kind compatibility surface, e.g. `KIND_MULT["upgrade"]`, kept
solely for rescoring/reprojecting legacy_v1 artifacts.

---

## Unrelated open threads (don't lose these)

- **Pricing project** — the 42 split successors currently inherit v1 parent
  economics as a bridge. Authoring real prices is a distinct project, and
  `KIND_MULT` degradation/modernization values are temporary. Task 3's handoff
  documents the measured tension: no single per-kind multiplier preserves both
  the ex-defect and ex-upgrade lineages.
- **Two catalog overlaps flagged, not fixed** (both inherited from v1, both
  recorded as `task3_flags` in the decisions file):
  `staging_or_decluttering_opportunity` vs `indoor_storage_clutter_heavy`, and
  `visible_mold_or_mildew` vs `mold_or_mildew_visible_bathroom`.
- **`trees_or_vegetation_too_close`** — flagged as a common hallucination and
  probably not worth costing; classification left correct, economics untouched.
- **`origin/pass_2c_redesign` commit `2e1b9ed`** (original photo quality setting
  + 2c prompt changes) is **not in the cutover build** and must stay out of
  variance measurements.
- **`pass_2c_kind_v3` branch** exists with 2c boundary tightening — also
  excluded from measurement.
- **Benchmark saturation** — `catalog-resolution-v2` scores 1.000 on holdout, so
  it has no headroom left and can only catch gross regressions. Cases were
  authored by the same session that authored the catalog, and kinds were
  gold-supplied rather than 2c-produced, so composed 2c×2d error is unmeasured.
  The informative next step is replaying **real stored 2b observations** from
  the corpus — but that inherits the variance problem, so it should follow 2a
  stabilization.

---

## Standing rules

- If Pass 2f enters scope, it runs on **Sol**.
- Embeddings sidecar `:8081` `/health` **lies** — it returns 200 while every
  call fails. Probe with a real POST (`scripts/run_kind_canary.py`
  `_preflight_embeddings`).
- Canary rerun mechanics: the script **skips** properties whose candidate dir
  already has an artifact. **Archive, don't delete.**
- Never hand-edit the v2 catalog, manifest, or audit report — edit
  `kind_v2_decisions.json` and regenerate.
- Task 1's ontology is fixed. A decision that seems to require reinterpreting a
  kind is a blocking contradiction to raise, not to patch.
