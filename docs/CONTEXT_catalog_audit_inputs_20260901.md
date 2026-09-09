# Context pack: catalog audit — evidence, architecture, and traps

Written 2026-09-01 for the session that will **plan** (and later the session
that will **execute**) the catalog-audit project. Self-contained: read this
before touching anything. It exists because the evidence base for the audit was
produced by an error-attribution review whose session context does not
transfer, and because the catalog's architecture makes several "obvious"
approaches wrong.

Scope reminder (from the project brief): the audit *proposes* surgical catalog
changes in a review document; a human approves; only then does anything get
implemented. Nothing in this doc is a pre-approved fix.

---

## 1. The finding this project descends from (TL;DR)

Full result: `docs/RESULT_error_attribution_20260831.md` (commits `bd62b0c`,
`e9a7dc3`). 95 attributable error cases (55 human-adjudicated + 40 gold-lane)
were each attributed to the first pipeline stage that went wrong, with
photos in hand and adversarial verification on every headline claim.

- **Misses are downstream**: 74.1% downstream vs 17.2% pass_2a (judged 58).
  On the human-adjudicated subset it is absolute: 23/23 downstream — 20 at
  Terra, 3 at 2d. The VLM described these conditions; the pipeline lost them.
- **Hallucinations are Pass 2a**: 10 of 12 (83.3%). 2a editorializes neutral
  finishes into billable-sounding judgements; downstream transmits faithfully;
  Terra "verifies" by restating 2a's descriptor.
- **Trivial billing is the billability chain**: 6/6 `appendix_trivial` cases
  re-ruled downstream (5× 2e tier call, 1× projection overriding 2e's own
  optional flag), all riding `route_work` into `accepted_for_work`.
- Stage table over all 59 downstream attributions:
  terra 21, 2d 20, 2e 5, 2c 5, condition_projection 4, 2b 4.

Why this project: **2d owns the naming problem** (6 of 8 `appendix_misnamed`
cases) and **the catalog's vocabulary drives both 2d misnames and a large
share of the Terra rejections** (see §4). The catalog is the highest-yield
downstream lever identified by the review.

## 2. The catalog is a GENERATED file — this changes everything

**Do not edit `tools/issue_catalog_kind_v2.json` by hand.** It is 7,075 lines,
catalog_version 3.2, 128 items — and it is the deterministic, byte-stable
output of:

```
tools/issue_catalog.json                             (v1, version 2.1, 107 items — untouched input)
+ tools/catalog_migrations/kind_v2_decisions.json    ("the reviewable source of truth")
--[ scripts/migrate_catalog_kind_v2.py ]-->
tools/issue_catalog_kind_v2.json                     (the catalog the pipeline loads)
+ tools/catalog_migrations/2.1_to_3.0.json           (audit manifest)
+ tools/catalog_migrations/2.1_to_3.0_audit.md       (generated audit report)
```

Running the generator twice is byte-identical (parity test:
`tests/test_catalog_kind_v2.py`). Consequences for the audit:

- **Every proposed change must be expressed against
  `kind_v2_decisions.json`**, then materialised by regeneration. The
  migration's own vocabulary (unchanged / reclassified / narrowed / split,
  wording overrides, fresh-authored successors) maps nearly one-to-one onto
  the brief's change types (reword / broaden / narrow / split / merge / add /
  deprecate). The review document should propose decision-file entries — the
  human then reviews the exact artifact the machine consumes.
- **Economic fields are inheritance-ruled in code**: split successors inherit
  the parent's pricing verbatim (`pricing_status=inherited_from_split_parent`);
  successor overrides can never rewrite an economic field (hard error); and
  costing **raises** on unpriced kinds — so an `add` proposal must carry a
  pricing answer, not just wording.
- **`ISSUE_CATALOG_PATH` overrides are a hard startup error** under
  `KIND_ONTOLOGY_VERSION=observation_kind_v2` (`tools/pipeline_config.py:95-101`).
  Phase-3 A/B cannot point the pipeline at a modified catalog copy via env;
  it must edit decisions + regenerate, ideally in a worktree.

## 3. The tracing already exists — consume it, do not rebuild it

The brief asks how to "connect Pass 2a wording to surviving observations, 2d
mappings, catalog entries, Terra decisions, and final outcomes." **That join is
built, frozen, and committed.** Do not re-derive it.

| artifact | what it holds |
|---|---|
| `reports/error_attribution_queue.json` | frozen, sha `b58dcca7…40df` — 104 cases + 15 gold photos, each with full lineage: 2a prose (`per_photo[*].p2a_prose`) → 2b bullets (+exact joins) → 2c surviving/dropped → `p2d.resolved_item_id` → `projected_condition_id` → Terra verdict + rationale, plus `claim_text` **as Terra saw it at run time**. **Never regenerate it** (generated_at alone breaks provenance). |
| `reports/error_attribution_verdicts.jsonl` | 107 lines, append-only latest-wins by file order. Per-case attribution, rationale, `p2a_excerpt` (verbatim substring of prose), skeptic outcomes, orchestrator revisions with `convention_note`. |
| `reports/error_attribution_gold_cases.json` | 40 materialised gold cases + the full matching table: matched / miss_candidate / out_of_catalog / already_cased per gold finding, with notes. |
| `reports/error_attribution.json` / `.md` | reconciled tallies. Regenerate with `.venv\Scripts\python.exe scripts\error_attribution_report.py --gold-cases reports/error_attribution_gold_cases.json` (must exit 0; strict). |
| `docs/HANDOFF_error_attribution_review.md` §7 | how to read a case record. |
| `docs/PLAN_error_attribution_review_20260831.md` | method + verified repo facts (ledger footguns, image locations). |

Reading rules that already burned people once: latest-wins is **file line
order** (ts is ignored); a ledger line with a *missing* `attribution` key acts
as an undo; `p2c_surviving[].catalogItemId` is `None` for every issue and
carries **no information about 2d's choices** (a skeptic was overturned for
reading it as one); `condition_id` is run-scoped — never join across runs.
Evidence photos live under
`C:\Users\Steven\IntelliJProjects\renointel-prod\public\images\properties\`,
not the artifact roots.

## 4. Case selection: attribution stage ≠ fix location

Filtering to "cases attributed to 2d" (20) misses the largest catalog signal.
The catalog-shaped evidence set is:

1. **2d-attributed cases** (20; 6 of 8 `appendix_misnamed` are here). Exemplars:
   bare weathered wood decking resolved to `patio_or_porch_surface_wear`
   ("coating breakdown" framing, `rc_3a19f759f2d9`); material-neutral
   "wallcovering is dated" resolved to `dated_wallpaper_present`
   (`rc_45b9d2f6923c`); "worn/dirty" green tile billed as wear when the true
   mechanism is style obsolescence (`rc_cf0c3ee00fcb`).
2. **Terra-attributed miss cases whose rejection ground was a
   catalog-introduced detail** (~19–20 `miss_label` cases). Prototype:
   `rc_128caa6212b8` — worn plank flooring, 2a material-neutral, item
   `vinyl_linoleum_torn_or_lifted`, Terra rejected the *material* ("hard plank
   material"), condition lost. Steven's own reviewer note proposed the fix
   shape: a "hard flooring" item that doesn't force a material commitment.
   Also `rc_a22a241b3bf0`: item named `dated_window_treatment_valance`, claim
   text spans "valance **or treatments**", Terra adjudicated only the absent
   valance and never ruled on the observed dated blinds. These are attributed
   `terra`, but the *lever* is catalog naming/wording.
3. **The 50 misnamed candidates kept from the factorized-verifier run**
   explicitly for a future remap lane: `docs/RESULT_factorized_replay_20260831.md`
   + `reports/factorized_review_scorecard.json`.
4. **Gold open coverage questions** — recorded verbatim in the gold matching
   table notes, conditioned on catalog coverage that was never checked:
   weathered/stained concrete entry steps (10806500/photo_001 g1), downspout
   discharging at the foundation (g4), missing base trim (11000447/photo_008
   g10, excluded at low confidence — "a reviewer who finds a base-trim item in
   the catalog would reasonably restore this as a miss").
5. **Already-parked catalog issues**: `docs/FINDINGS_catalog_3_2_deferred_issues.md`.
   Absorb these; do not rediscover them.

**Counterexamples are mandatory.** The attribution cohort is a discrepancy
cohort — it shows every item failing and no item succeeding. Example:
`vanity_dated_style` has 1 hallucination but 3 counted agreements. Success
evidence per implicated item lives in the queue's `counted_agreement` /
`counted_correct_rejection` lanes, `reports/labels_v1_1.json` (the v1.1 human
truth spine; `misnamed_billed` is the human-labeled catalog-pain class), and
`reports/review_analysis.json` (reviewer notes contain catalog suggestions
verbatim).

**Evidence bar** (noise floor: 80% human self-consistency on the v1.1 repeat
arm; 6.4% Terra replica floor): a proposed change needs **≥2 independent
cases, or 1 case + gold/factorized corroboration, or an explicit argument for
why one case suffices.** A split inside the noise is not a finding.

Repeat-offender families already quantified (RESULT §9.4) — where corroboration
requirements may beat rewording: style/modernization
(`older_flooring_style` ×6 in error lanes, `dated_interior_trim` ×6,
`cabinets_dated_style` ×3, `dated_window_treatment_valance` ×4) and
wear-on-intact-surfaces (`peeling_or_discolored_paint` ×7 — errs in **both**
directions, `hard_flooring_scratched_or_worn` ×6, `worn_or_stained_carpet` ×5).
The existing per-item evidence-gate mechanism is `min_photo_evidence`
(`estimate_guard_v2`) — 4 items already gated at 2 photos. "No catalog change /
gate the item instead" is a first-class proposal type.

## 5. Retrieval traps: how a "safe" wording change breaks silently

- **2d is strict exact-kind.** An item's `kind` (defect / degradation /
  modernization) gates which observations can retrieve it. Reclassifying an
  item changes what 2d can see at all.
- **`scene_groups` gates retrieval.** Known gotcha: living-area items need
  `["living_areas"]`, not `["living"]`, or retrieval silently drops them. Any
  moved/reworded/split item needs a retrieval check, not just a wording review.
- **`embed_text` drives candidate retrieval** (embeddings sidecar). Rewording
  an item shifts its retrieval neighborhood; neighbors must be checked for
  newly-created overlap. (Sidecar health note: `:8081/health` returns 200 even
  when calls fail — probe with a real POST.)
- **The electrical trade is product-quarantined by design** (2026-07-12,
  fail-closed lanes; wiring observations parked as true-but-unactionable
  2026-08-11). Do not propose electrical coverage; gold electrical findings
  were excluded on exactly this ground.
- `package_strength` is computed at runtime and `package_category` has
  static-map fallbacks — they are not catalog fields to author.

## 6. Baseline: state it explicitly

The frozen evidence (18-listing canary `run_1` of 2026-08-17/18 + 10
production runs) was produced under the **pre-3.2 catalog**. The tree today
holds generated catalog 3.2 (contextual-repair support, commit `a9ed7ad`),
whose publish is pending Sol reviews, with the v5-only flag defaulted off.
Therefore:

- The audit's "current wording" inspection must diff against the
  **run-stamped `claim_text`** in the queue (what Terra actually saw), not
  assume the current file matches the evidence.
- The plan must pick and state its baseline (3.2 as-generated is the sensible
  one) and reconcile with the pending 3.2 publish rather than racing it.

## 7. Phase-3 validation: tier it, because there is a cost cliff

1. **Tier 1 — free, always:** regenerate the catalog; byte-parity + full suite
   (`.venv\Scripts\python.exe -m pytest` — bare `python` doesn't resolve the
   venv); catalog validator; diff affected items' claim_text/embed_text
   against the frozen cases they're meant to fix.
2. **Tier 2 — cheap:** retrieval-level replay through the embeddings sidecar
   for every affected observation (proves reworded/split items still retrieve,
   and surfaces new neighbor overlap). No LLM calls.
3. **Tier 3 — live and expensive, the real gate:** canary replay through
   2d/Terra. One full replica ≈ **3.98M tokens** (more than a free OpenAI day
   at the 2.5M/day quota). The plan must say which approved changes justify
   this, and batch them into one replay. Note `artifacts_canary/renovation_session9_20260818/`
   holds **two** independent replicas (`run_1`, `run_2`, 18 debug artifacts
   each) — useful for variance context and offline comparisons.

Post-change re-scoring against the frozen attribution cases is *offline* for
wording diffs but **live** for behavior (2d resolution and Terra verdicts are
LLM calls). Keep Pass 2a and other prompts untouched during the experiment so
the catalog effect is isolated — the brief already requires this; it is also
what makes the run_1/run_2 replicas usable as a baseline.

## 8. Standing priorities (session-external, so stated here)

- **Hallucinations over pricing.** Steven is not a renovator; prices are
  provisional. Judge changes on fewer hallucinations / fewer lost real
  conditions, not dollar effects. Price calibration is a separate future
  thread.
- **Slim, data-driven, surgical.** No hardcoded special cases; prefer the
  smallest justified ontology correction; expansion is not inherently good.
  A proposal that says "no catalog change — fix 2e tiering / Terra prompt /
  add an evidence gate instead" is a success, not a failure of the audit.
- Attribution-convention precedent (ruled by Steven 2026-08-31, recorded in
  the ledger): false content originates at 2a; true-but-shouldn't-bill belongs
  to the billability chain. The audit inherits this vocabulary.

## 9. Suggested reading order for the audit session

1. This document.
2. `docs/RESULT_error_attribution_20260831.md` (findings + §9 levers).
3. `scripts/migrate_catalog_kind_v2.py` docstring + `tools/catalog_migrations/kind_v2_decisions.json` (the edit surface).
4. `docs/HANDOFF_error_attribution_review.md` §7, then skim 3–4 case records
   in the queue (start with `rc_128caa6212b8`, `rc_3a19f759f2d9`,
   `rc_45b9d2f6923c`) with their ledger lines.
5. `docs/FINDINGS_catalog_3_2_deferred_issues.md` + `docs/STATE_kind_ontology_program.md`.
6. `reports/labels_v1_1.json` / `reports/review_analysis.json` for the
   counterexample side.
