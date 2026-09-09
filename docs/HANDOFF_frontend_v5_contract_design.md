# Handoff — design the frontend contract for the v5 renovation estimate

Written 2026-08-27 · backend `renovation_architecture_rework` · FE repo
`C:\Users\Steven\IntelliJProjects\renointel-prod`

**Purpose.** A fresh session is to (1) understand what the v5 renovation
estimate actually contains and means, (2) investigate what the frontend needs
in order to render it, and (3) design the contract between them — including
whether the backend should emit new FE-shaped keys or the FE should derive
everything from the raw v5 result, what happens to the surfaces v4 fed that
v5 has no equivalent for, and how the implementation should be sequenced.

**This document is context only.** It contains file locations, data shapes,
semantics, measured facts, constraints, and the open questions on record. It
deliberately contains **no design, no recommended contract, no field mapping,
and no implementation plan** — those are the new session's to produce. Where a
prior session's investigation is cited, it is cited as something to verify,
not as an answer to adopt.

---

## 1. Steven's decisions already on record (constraints, not open questions)

- **No compatibility adapter.** The FE is to read v5, not to abstract over
  v4-and-v5. A permanent dual-shape layer is explicitly unwanted; it is the
  bloat the migration is trying to remove.
- **renointel is OFFLINE.** Steven is the only user right now and has taken
  the product down while making these changes. There is no live-user
  constraint on breaking or temporarily removing FE surfaces. This materially
  widens the option space compared with every earlier document, several of
  which assume a live product.
- **v4 is going away.** Retirement is a settled direction. Its *timing
  relative to this work* is open — see §7.
- **Prices are provisional.** Steven is not a renovator; dollar figures are
  uncalibrated and price calibration is a separate future thread. Do not
  design around the numbers being trustworthy in absolute terms; do not
  propose price changes.
- **Slim-code philosophy** (standing): no hardcoded special cases,
  data-driven designs, line budgets in handoffs, MVP-stage pragmatism.
- Evaluation frame for the estimate itself: fewer hallucinations, more useful
  observations that build packages; dollar deflation from dedup is accepted.

## 2. Where this sits

The backend migration is complete: production runs `RENOVATION_ARCHITECTURE_MODE=new`
(flipped 2026-08-21 ~23:20 CT), and every published artifact carries a complete
`renovation_estimate_v5` envelope at the **root** of `photo_intel.json`. The
7-day observation window closed 2026-08-27 with no rollback trigger fired
(`docs/DECISION_renovation_architecture_session9_cutover_20260821.md` §10).

The FE still renders `renovation_estimate_v4`, so v5 has never been displayed
to anyone. FE adoption is the last structural piece of the v4→v5 migration and
gates: v4/2f retirement, the throughput recovery below, and the historical
backfill's open "how do estimates come back" question (kind-ontology Task 4B).

**Measured cost of the current state** (production ledgers + `AnalysisRun`,
2026-08-27): both active post-cutover days completed exactly **5 listings**
and then hit the Sol ceiling (91.1% / 87.8% of 250k/day) while Terra idled at
73% / 47% of 2.5M. The majority of Sol spend is **v4's Pass 2f** (~24–58k per
listing vs v5's Sol review ~8.5k), running solely to feed the FE. Retiring
v4+2f moves the bottleneck to Terra, roughly **doubling** capacity to ~10–11
listings/day.

Load-bearing detail: **Pass 2f cannot be disabled while v4 is displayed.**
Verified in code — with no 2f client, packages get `not_run` status, which is
not in `ACTIVE_PACKAGE_STATUSES`, so `finalize_package_candidates`
(`tools/rehab_packages.py:2528-2548`) promotes none of them and v4's estimate
collapses to standalone lines only. The capacity win is therefore unlocked by
this FE work, not available before it.

## 3. What the v5 artifact actually contains

Contract source of truth: `tools/renovation_architecture/contracts.py`.
Placement rules enforced by `tools/publication_gate.py`; artifact-level gate:
`scripts/verify_renovation_artifact.py`.

**Envelope** (root key `renovation_estimate_v5`, written in `new` mode):

```
{ schema_version, estimate_id, state, reason, error_detail, provenance, result }
```

`state` must be `complete` for a published artifact in `new` mode; a
non-complete envelope fails the job rather than publishing. Any reader must
handle `state` / `reason` and unwrap `.result` — the payload lists are *not*
at the top level of the key.

**`result` sections** (all condition-id / work-item-id joinable):

| section | what it holds |
|---|---|
| `observed_conditions` | one per (catalog_item, estimate_unit); ids, `catalog_kind`, `scene_group`, `estimate_unit_id`, `room_surrogate_id`, `source_room_surrogate_ids`, `identity_ambiguous`, `unit_resolution_source/reason`, `issue_ids`. **Carries no free text.** |
| `evidence_facts` | per condition: `photo_keys`, `representative_photo_keys`, `distinct_photo_count`, `distinct_view_count`, `duplicate_groups`, `min_photo_evidence_required`, and `evidence_refs` (per issue+photo, incl. the upstream `observation` sentence) |
| `condition_reviews` | Terra's verdict per condition: `supported` / `unsupported` / `cannot_assess`, plus a one-sentence `rationale` and `terra_call_id` |
| `condition_dispositions` | deterministic outcome per condition: `accepted_for_work` / `excluded` / `inspection` / `no_action` / `withheld`, each with a `reason_code` |
| `work_items` | the billing lines; `action_code`, `trade_bucket`, `unit_policy`, `billable_unit_id`, `unit_count`, `estimate_scope`, `low`/`high`, `condition_ids`, `catalog_item_ids`, status (`active` / `suppressed`) |
| `package_candidates` | one per (package_type, estimate_unit); `pricing_tier`, `proposed_treatment`, `strength`, `driver_work_item_ids`, `support_work_item_ids`, `child_work_item_ids`, `low`/`high`, `unfloored_low`/`high`, `display_only` |
| `package_decisions` | Sol's per-candidate call: `approve` / `reject` / `uncertain`, `rationale`, `combine_with`, `split_groups` |
| `package_applications` | what actually billed: `status` (`applied` / `not_applied`), `reason_code`, `effective_low`/`high`, `absorbed_work_item_ids` |
| `coverage_ledger` | exactly one entry per active work item, reason-coded (`absorbed_by_package` 0/0, or standalone at the work item's range with `package_rejected` / `package_uncertain` / `package_split` / `no_covering_package`) |
| `totals` | `schema_version`, `currency`, then `standalone`, `packaged`, `inspection`, `headline` — each `{low, high}` |
| also | `standalone_estimate`, `package_review_snapshots`, `reconciliation_audit`, `observability`, `work_dedup_collisions`, `terra_calls`, `terra_unit_usage`, `terra_listing_usage`, `sol_calls`, `sol_listing_usage` |

Verified against the live artifact 2026-08-27 — the section names, types and
counts above are exact, not approximate.

**Sample scale** (production `redfin_10949071/20260825_052355_3b380cca`):
65 observed_conditions / 65 reviews / 65 dispositions, 44 work_items,
7 package_candidates / decisions / applications, 36 coverage_ledger entries;
`totals` = standalone 2,523–21,102 · packaged 36,900–95,888 · inspection 0–0 ·
headline 39,423–116,990. The same artifact's v4 projection headline is
39,046–69,626 — i.e. **the high end moves ~+68%**, which will cascade into
whatever the FE derives from it.

`tools/review_cards.py` (`v5_result`, `index_result`, `join_2f`) is a working
reference implementation of reading these sections and joining them.

## 4. The semantic model behind those objects

Design intent: `docs/Revised_Renovation_Scope_and_Estimate_Architecture.docx`
(read it — it defines the decision boundaries the contract must respect).

- **Terra** judges only whether a claimed condition is visible. It does not
  choose scope, packages, or dollars, and the design explicitly refuses to
  produce model confidence percentages.
- **Deterministic policy** turns verdict × catalog terminal route × objective
  evidence (distinct *view* count, not photo count) into a disposition.
- **Work items** derive only from `accepted_for_work` conditions.
- **Sol** judges package *coherence only* and is instructed never to reassess
  whether a condition is real. Its approval is not a statement that the photos
  warrant the package.
- **Reconciliation** is deterministic: at-most-once child ownership, applied
  package bills `max(unfloored tier, sum of owned children)`, absorbed children
  ledger at 0/0, everything else bills standalone at its exact allowance.
- **Seven invariants** the design treats as load-bearing, including: every
  condition reaches an explicit disposition; evidence strength comes from
  objective corroboration not model confidence; package decisions affect
  bundling and allowances, not condition truth; rejected or uncertain packages
  fall back to accepted child work rather than deleting it; totals reconcile
  without double counting.

Two consequences worth understanding before designing anything: the
**inspection lane is contractually 0/0** in this schema (inspection-routed
conditions produce no work item and no dollars), and there is **no confidence
score anywhere** by design.

## 5. What v4 feeds the FE today that v5 has no direct equivalent for

This is the substance of the contract problem. A prior read-only FE sweep
inventoried the surfaces currently fed by v4 (verify independently — these are
pointers, not gospel):

- **Pass 2f verification fields** — `verification_status`,
  `confirmed_issue_ids` / `rejected_issue_ids`, `pass_2f_applied/attempted/fallback_reason`,
  `visual_verification_status`. Feed verdict badges, approval filters, and the
  audit pages. v5's nearest objects are `condition_reviews`,
  `condition_dispositions`, and `package_decisions` — a different vocabulary,
  not a rename.
- **`confidence_score`** — drives a "% confidence" chip. No v5 equivalent, by
  design (see §4).
- **`package_strength`** — v4 computes it at runtime; v5 candidates carry a
  `strength` field. Confirm whether the semantics match.
- **`groups[].line_items[]`** — feeds cost-by-trade, the itemized table, the
  results row, the mobile investment card, and the per-issue cost badge. v5
  has `work_items[]` with `trade_bucket`, but no `groups` structure.
- **`project_scope_breakdown`**, **`suppressed_package_candidates`**,
  **`bathroom_room_count_signal`**, **`bathroom_expansion_audit`** — no v5
  equivalents.
- **`rehab_evidence_projection_v1`** — the split-dollar contract
  (`photo_supported` + `needs_inspection` == headline, risk excluded), with a
  `projection_id` / provenance / reconciliation envelope. v5 has bare `totals`
  and an always-zero inspection lane.
- **`ui_priorities_v1`** — a *published root key* derived 100% from v4 inside
  `tools/artifact_writers.py:75-184` (called at `:1113-1117`). Reported to be
  the *preferred* package source for the desktop card and Top Picks, with raw
  v4 packages as fallback. Nothing re-derives it from v5 today.
- **`evidence_projection_policy_version` / `evidence_projection_status`** —
  root keys set only inside the v4 block (`artifact_writers.py:1097-1112`).
- The FE's **run-completion contract** reportedly fails a run when
  `renovation_estimate_v4` is null or missing `pass_2f_trace`.

Also on the FE list already, unrelated to v5 but in the same contract surface:
the **severity display defect** — the card badge shows the bucket max of
*boosted* `display_severity` while the photo row shows *raw* catalog severity,
same 1–5 word scale, different quantities
(`memory: ui-severity-display-defect`; `tools/property_summary_pass.py:254-271`).

## 6. Prior investigation to verify, not to trust

Two read-only sweeps were run before this handoff. Treat their outputs as
leads:

- A backend sweep of what still computes/reads v4, what the mode selector
  switches, and exactly which artifact keys carry which engine. Its findings
  are summarized in §2/§5 above and in
  `memory: v5-migration-remaining-and-window-closed`.
- An FE sweep reporting **zero references to `renovation_estimate_v5`** or its
  field names anywhere in the FE repo, and a single normalizer chokepoint that
  rejects v5 by construction (a `version` allowlist of v3/v4 plus a required
  `groups[]` array) — reported at `lib/types/renovationEstimate.ts:788-796`,
  with roughly a dozen inline `version === 'renovation_estimate_v4'`
  narrowings downstream. It also reported an existing *kind-ontology*
  version-aware adapter (`lib/analysis/kindOntology.ts`) that governs the
  issue-kind axis only and explicitly warns it is a different axis from the
  estimate.

Also relevant and already written: `docs/analysis/session_b_qp8_inspection_lane.md`
— an audit of whether v5's inspection lane surfaces anywhere the user sees,
which reached the FE-contract list.

## 7. Open questions this session should surface and resolve (they are not decided)

These are recorded as open, with the facts that bear on them. Several are
Steven's calls; this session's job is to make them decidable.

1. **Ordering: delete v4 before or after the FE work?** Earlier documents
   assume "after," because deleting v4 breaks a live product. That objection
   is now void — the app is offline with no users. Facts on both sides:
   deleting v4 first removes `ui_priorities_v1`, the evidence-projection keys,
   the L1 rollback rung (`mode=current` publishes v4 only), and the 2f
   disagreement telemetry; keeping it costs ~half the daily listing capacity
   and keeps two engines alive. Note also that v4-as-rollback only has value
   while something can render v4.
2. **`ui_priorities_v1`: re-derive from v5, retire, or replace?** It is a
   published backend-derived key, so this is a backend *and* FE decision.
3. **Where does the split-dollar / inspection lane go?** v5's inspection lane
   is contractually 0/0, so a `needs_inspection` display fed from v5 would
   always be zero. Options range from retiring the lane to changing what v5
   routes into inspection — the latter is an estimate-behaviour change with
   its own cost class (§8).
4. **Does the backend emit FE-shaped keys, or does the FE derive from raw
   v5?** The precedent (`ui_priorities_v1`, `rehab_evidence_projection_v1`)
   is backend-derived; the v5 design's instinct is auditable raw structure.
5. **What replaces the verification UI** (badges, approval filters, audit
   pages) given `condition_dispositions` / `package_decisions` carry different
   semantics than 2f verdicts, and no confidence score exists.
6. **Cached-estimate resync.** The FE caches `capexLow/MostLikely/High` and a
   serialized estimate JSON; the v5 headline differs materially (§3), so
   cutting over implies a resync strategy for stored values.
7. **Does the FE need a `work_items → line_items`-shaped projection at all,**
   or do the surfaces it feeds get redesigned around v5's structure?

## 8. Change mechanics — what each kind of change costs

- **FE-only changes**: free of backend cost class entirely. No canary, no
  quota, no freeze.
- **Backend changes that only add or reshape *derived* artifact keys**
  (e.g. a v5-sourced `ui_priorities`): deterministic, offline-replayable
  against stored artifacts via `scripts/replay_renovation_architecture.py`
  (zero provider calls). No fresh canary.
- **Anything that changes what a model sees** (Terra/Sol prompts, evidence
  selection, batching) or that shifts dispositions: requires a fresh input
  freeze and a new two-replica canary (~8M Terra tokens ≈ 4 free-quota days;
  no code edits mid-canary). This is the expensive class — a proposal to
  reroute conditions into the inspection lane, for example, lands here.
- **Quotas**: Terra 2.5M/day and Sol 250k/day are shared free allowances; the
  budget guard is ON and fails a listing closed on denial. Current production
  capacity is ~5 listings/day while 2f runs.
- The observation-window freeze is **lifted** as of 2026-08-27; nothing is
  time-gated now.

## 9. Concurrent work this session must not collide with or duplicate

- **The quality program (Sessions C–H)**, `docs/ROADMAP_quality_program_sessions_20260826.md`.
  Wave-1 changes will alter v5's package shape: an interior-modernization
  application gate, deterministic multi-bathroom expansion (per-surrogate
  candidates), Sol splits materialized as sub-packages, and a single-child
  candidate rule. Decisions S1–S9 are recorded in
  `docs/DESIGN_renovation_architecture_decision_packets.md` §9. **Read this
  before fixing a contract** — new application/ledger reason codes and
  per-surrogate bathroom candidates are expected.
- **The thumbnail-ingest fix thread** (`docs/HANDOFF_thumbnail_photo_ingest_fix.md`),
  which will add photo dimensions to artifacts — relevant if the FE wants to
  display or filter on evidence quality.
- **A running background task** fixing a lone-surrogate `UnicodeEncodeError`
  in artifact postprocessing.
- Do not delete the dormant per-item Pass 2f scaffolding
  (`tools/renovation_estimate.py:691-728`, `:309-333`, `:176-206`).
- **`tools/rehab_packages.py` is shared v5 infrastructure** — v5 imports
  `infer_package_candidates`, `build_package_affinity`, and
  `_MODERNIZATION_TIER_RANK` from it. Retiring v4 means removing
  `tools/renovation_estimate_v4.py` and the Pass-2f surface in
  `tools/scene_classifier_passes.py`, **not** `rehab_packages.py`.

## 10. Working rules

- `RV_ROOT` (`C:\Users\Steven\PycharmProjects\realtorvision-backend`) is the
  live production working tree — no `git checkout` of other branches; use a
  worktree if one is needed.
- Artifacts are read-only. Frozen review evidence
  (`reports/review_queue.json`, `reports/review_verdicts.jsonl`,
  `reports/review_analysis.*`, all `reports/session9_*`) must not be modified
  or rebuilt.
- Tests: `.venv\Scripts\python.exe -m pytest` (bare `python` does not resolve
  the venv).
- Artifact gate: `scripts/verify_renovation_artifact.py --artifact <run dir>
  --expect-mode new --expect-terra-model gpt-5.6-terra --expect-sol-model gpt-5.6-sol`.
- Production artifacts: `C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts/<key>/<run>/`;
  post-cutover run dirs are ≥ `20260821_230000` (10 of them). Canary corpus:
  `artifacts_canary/renovation_session9_20260818` (18 listings × 2 replicas).
  Addresses are in the FE's `prisma/dev.db` (open read-only, `mode=ro`).
- Note two production listings (`redfin_10965375`, `redfin_10922002`) were
  analysed entirely on CDN thumbnails — do not use them as representative
  samples.

## 11. Suggested reading order (procedural, not a steer)

1. `tools/renovation_architecture/contracts.py` — the schema and vocabularies.
2. A real artifact end to end: `redfin_10949071/20260825_052355_3b380cca`
   (`photo_intel.json` root key `renovation_estimate_v5`), alongside the same
   file's `renovation_estimate_v4` for comparison.
3. `docs/Revised_Renovation_Scope_and_Estimate_Architecture.docx` — the
   decision boundaries and the seven invariants.
4. `tools/renovation_architecture/reconciliation.py` and `disposition.py` —
   how dollars and dispositions are actually decided.
5. `tools/artifact_writers.py` — what is written where, slim vs debug, and the
   `ui_priorities_v1` / evidence-projection derivations.
6. `tools/review_cards.py` — a working v5 reader to crib joins from.
7. The FE repo, from its normalizer chokepoint outward.
8. `docs/ROADMAP_quality_program_sessions_20260826.md` — what is about to
   change in the v5 shape.

The deliverable format is this session's choice. Steven asked that it come to
its own conclusions and make its own recommendations rather than inherit any.
