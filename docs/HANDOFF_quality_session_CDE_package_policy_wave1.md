# Handoff — Session C+D+E combined: wave-1 package policy (QP3, QP6, QP5, QP4)

> **Superseded 2026-08-27.** After source and artifact review, Steven narrowed
> wave 1 to QP3 only and declared it complete. QP6, QP5, and QP4 were not
> implemented and are deferred outside wave 1. Do not execute this combined
> charter. The implementation and replay record is
> `docs/RESULT_quality_wave1_qp3_20260827.md`.

Written 2026-08-27 by Session B · backend `renovation_architecture_rework`
(clean at `bdc27ff`) · commissioned by
`docs/ROADMAP_quality_program_sessions_20260826.md` (sessions C, D, E, run
together at Steven's direction 2026-08-27) · proposals =
`docs/PROPOSALS_output_quality_improvements_20260826.md` (QP3, QP6, QP5,
QP4) · decisions S2/S3/S4/S5/S9 in
`docs/DESIGN_renovation_architecture_decision_packets.md` §9. **Supersedes
`docs/HANDOFF_quality_session_C_package_policy_1.md`** (C-only), which had
two errors this document corrects (see §7).

Inherit the roadmap's hygiene + simplicity guardrail blocks in full.

**Why combined.** The roadmap has no go/no-go gate between C, D and E (gate
1 is after B, gate 2 after E), it already defines the wave-1 baseline as the
*combined* C+D+E replay, and D and E share two mechanisms that are only built
well once: the derived-candidate id scheme
(`make_package_candidate_id` keys on `(estimate_id, package_type,
estimate_unit_id)`, `ids.py:85-91` — both QP5 sub-packages and QP4
per-surrogate clones need a fourth token) and the replay Sol-inheritance
rule (`_map_stored_decisions` drops candidates the stored run never judged,
`scripts/replay_renovation_architecture.py:156-167` — without inheritance
both QP5 sub-packages and QP4 clones replay as zero).

**Window.** Building is unrestricted. No estimate-behaviour change ships
before the observation window closes (end of 2026-08-28). No live provider
calls are needed anywhere in this session — replay is free and offline.

---

## 1. Decisions

**Only one decision is open.** Both others named in the superseded C handoff
are already settled or moot:

- **OPEN — S5, QP6 scoping (Steven).** Type-scoped (bedroom/living repair
  singles only) vs blanket (all single-child candidates). Measured on canary
  run_1: single-child candidates are `bedroom_repair` 15, `living_repair` 5,
  `bathroom_repair` 3, `exterior_repair` 2. **Type-scoped suppresses 20;
  blanket suppresses 25**, and the extra 5 are exactly the ones the evidence
  argues against touching — both human-warranted single-child P1 cards were
  `exterior_repair`, and the 3 `bathroom_repair` singles sit in the room
  family QP4 is about to expand. Recommendation stands: **type-scoped**.
- **CLOSED — QP3 scope.** S2 already reads "Adopted — interior-modernization
  application gate (not the tier demotion)". The generic per-family tier cap
  is not an open alternative; do not re-present it.
- **MOOT — the whole-home rollup side effect.** The proposals warned QP6
  would shrink the display-only whole-home aggregate. It cannot:
  `_aggregate_whole_home` filters contributors to
  `package_category == "turnover"` (`package_candidates.py:369-373`), while
  QP3 touches `modernization` and QP6 touches `repair`. Structural, not
  data-dependent. (Canary run_1 additionally has zero turnover candidates.)

A second decision arrives at the end of the session — see §6, the QP4 ship
gate arithmetic, which as written cannot be evaluated by replay.

---

## 2. Order of work

Strict order, each change its own commit(s) and its own replay:

1. **QP3** — interior-modernization application gate
2. **QP6** — single-child rule (needs the S5 answer)
3. **QP5** — Sol splits as sub-packages
4. **QP4** — multi-bath expansion, **last and isolated**

QP4 goes last because it is the only change with a ship gate; if it fails,
it must be holdable without unpicking QP3/QP6/QP5. Keep its commits free of
incidental refactors to the earlier three.

---

## 3. QP3 — interior-modernization application gate

**Rule.** A `bedroom_modernization` / `living_modernization` candidate whose
drivers are all opportunity-kind is not applied; children bill standalone at
their exact allowances.

**The predicate is a pure field read — no re-derivation.** Verified at
`rehab_packages.py:2077-2088`: the `trigger_reason` if-chain assigns
`package_driver` whenever `defect_drivers` is non-empty (`:2077-2080`), so
`proposed_treatment ∈ {opportunity_driver_with_corroboration,
opportunity_driver_with_multiphoto_corroboration}` already *means* "all
drivers are opportunity-kind". Read `PackageCandidate.proposed_treatment`
(`contracts.py:453`); do not recompute driver kinds.

**Boundary that is easy to get wrong:** `multiple_package_support_same_estimate_unit`
(`:2089-2092`, driverless) is **not** in the gated set. It has no drivers at
all rather than opportunity-only drivers, and the proposals' definition is
exact. Canary run_1 has 5 of these among bedroom/living modernization; they
must survive.

**Where.** v5 path only — `_build_package_candidate` is shared legacy code
and an ungated change alters live v4/FE output outside any replay, and v4's
post-2f re-tier (`rehab_packages.py:2401-2412`) would bypass a
candidate-build-time gate. Two viable sites:
`package_candidates.build_package_candidates` (suppress emission) or
reconciliation application eligibility. **Prefer the application side** —
gating at emission also removes the candidate from Sol's review set, which
changes the Sol request and makes the replay comparison noisier than it
needs to be. If done at reconciliation, it is a new non-eligible branch and
needs a new reason code (see §5's vocabulary checklist — the same machinery
QP5 touches).

**Predicted effect (canary run_1, measure against this).** 33
bedroom/living modernization candidates exist; treatment mix is
`package_driver` 20, `opportunity_driver_with_corroboration` 8,
`multiple_package_support_same_estimate_unit` 5
(`opportunity_driver_with_multiphoto_corroboration` does not occur).
**8 currently-applied packages would gate, carrying effective
$26,638 / $90,822.** Those dollars do not vanish — children fall standalone
at their own allowances, so the headline delta is the tier uplift only
(package effective range minus the sum of its owned children's standalone
ranges). If the replay shows the full $26,638/$90,822 leaving the headline,
something is wrong: the children are being lost, not demoted.

---

## 4. QP6 — single-child rule

**Rule (type-scoped form).** Do not emit a `bedroom_repair` /
`living_repair` candidate with exactly one child work item. Its work item
bills standalone through the existing ledger path — no contract change:
a work item with no covering candidate already gets
`representation="standalone"` / `reason_code="no_covering_package"`
(`reconciliation.py:209-231`).

**Where.** Emission-side in `build_package_candidates` (the main
construction loop is `package_candidates.py:293-349`). Suppressing at
emission is correct here — the point is partly to shrink the Sol listing
call and ease the binding Sol quota, which only happens if the candidate
never reaches Sol.

**Predicted effect (canary run_1).** Type-scoped suppresses 20 candidates
(bedroom_repair 15 + living_repair 5) out of 129 non-display candidates.
Blanket would suppress 25.

**Interaction with QP5 — decide this explicitly.** On
`redfin_166147710`, Sol rejected `living_repair@living_room_primary` with
`split_groups` of shape `[1, 1]`. Under QP5 that materializes **two
single-child `living_repair` sub-packages** — exactly what type-scoped QP6
suppresses. Pick one and encode it once: either QP6's predicate applies only
to originally-built candidates (sub-packages are Sol-endorsed groupings and
survive), or it applies to derived sub-packages too. **Recommendation:**
apply QP6 at emission only, so QP5 sub-packages are unaffected — Sol
explicitly said those items group that way, which is stronger evidence than
the single-child heuristic that QP6 encodes.

**Interaction with QP4 under the blanket option.** QP4's per-surrogate
bathroom clones are frequently single-child, and canary run_1 already has 3
single-child `bathroom_repair` candidates. Blanket QP6 needs an explicit
carve-out for bathroom clones or it will silently delete what QP4 builds.
Type-scoped needs none.

---

## 5. QP5 — Sol splits as deterministic sub-packages

**The canary cohort is 5 cases, and they are not the shape the proposals
describe.** All five stored `split_groups` decisions in canary run_1 are
**reject**-with-split, not approve-with-split:

| property | candidate | Sol | candidate $ | split group sizes |
|---|---|---|---|---|
| redfin_11079485 | `kitchen_modernization@kitchen_primary` | reject | 10,920 / 25,480 | [4, 3] |
| redfin_125970550 | `exterior_repair@exterior_primary` | reject | 4,427 / 15,937 | [1, 1, 1, 1] |
| redfin_126224899 | `exterior_repair@exterior_primary` | reject | 3,939 / 14,179 | [2, 1, 1, 1] |
| redfin_126418713 | `exterior_repair@exterior_primary` | reject | 733 / 3,664 | [1, 1] |
| redfin_166147710 | `living_repair@living_room_primary` | reject | 2,294 / 7,647 | [1, 1] |

Consequence: the `split_recommended` application reason code is **never
exercised** on this data (canary run_1 application reason codes are
`approved_absorbs_children` 112 and `decision_rejected` 7, of which 5 carry
splits). The ladder in `reconciliation.py:157-170` tests `reject` (`:159`)
*before* `split_groups` (`:163`), so a rejected split is coded
`decision_rejected` and its groups are discarded silently. **QP5 must handle
reject-with-split as the primary case** — Sol saying "this is not one
package, and here is how it actually groups" — with approve-with-split as
the untested secondary path. The decision-packets §6 narrative already
states this correctly; the QP5 proposal text does not.

**The structural constraint the proposals missed — read this before
designing.** Sub-candidates **cannot** be appended to
`result["package_candidates"]` after Sol review. Four independent checks
reject it:

- Snapshot immutability: `candidate_snapshot_sha256` is recorded pre-review
  and recomputed over the final list at `validators.py:2706-2716`
  (`package_review_snapshot_hashes`, `:2371-2386`). An enlarged list fails.
- Candidate↔decision bijection: `validators.py:2553-2576`.
- Candidate↔application bijection: `validators.py:1499-1529`.
- `sol_calls[].package_candidate_ids` must equal `sorted(candidates)`
  exactly: `validators.py:2684-2690`.

**Therefore: emit sub-packages into a new result section**, leaving
`package_candidates`, `package_decisions` and the snapshots untouched.
Register the new top-level key or `_check_unknown` rejects it
(`_RESULT_KEYS` / `_PACKAGE_REVIEW_RESULT_KEYS`, enforced at
`validators.py:1441` and `:2409`). Do **not** take the shortcut of
recomputing the candidate snapshot over an enlarged list — it silences the
check but defeats the immutability guarantee `contracts.py:516-526` claims,
and the bijections still fail.

**Id scheme is feasible.** `validators.py` never imports `ids.py`; every
candidate-id check is a shape regex `^pk1_[0-9a-f]{16}$`
(`validators.py:71-77`, `_require_id` `:366-373`). A sub-package id may be
derived any way you like provided it renders in that shape. The same holds
for QP4's clones.

**The five eligibility sites (the proposals say four — there are five).**
The predicate `approve AND NOT display_only AND NOT split_groups` is written
out longhand at:

1. `reconciliation.py:131-137` — `_eligible()`, gates ownership
2. `reconciliation.py:157-170` — the status/reason ladder (the only place
   application reason codes are minted)
3. `reconciliation.py:351-357` — `recompute_reconciliation_audit`, feeds
   `unsupported_billing`
4. `validators.py:1540-1544` — per-application approval-basis check
5. `validators.py:1674-1678` — forces children of non-eligible candidates to
   be `standalone` in the ledger

All five must agree. Additionally `validators.py:1732-1744` re-runs
`compute_reconciliation` and requires byte-equality of
`package_applications` / `coverage_ledger` / `totals`, so **all new logic
must live inside `compute_reconciliation`** — nothing bolted on afterwards
will survive.

**Vocabulary checklist for any new reason code.** Add to the frozenset
(`contracts.py:163-170`) *and* to `_STATUS_REASONS`
(`validators.py:1066-1076`) or validation fails with "reason_code X is not
valid for status Y". Note also `validators.py:1077-1087` (`applied` must
absorb ≥1 child; every non-`applied` status must absorb nothing and be 0/0)
and the representation↔reason coupling at `validators.py:1003-1023`. Funnel
counters count status strings (`reconciliation.py:480-484`, re-verified at
`validators.py:1745-1754`) — a new status or row changes them.

**Replay hazard, currently latent.** `replay_renovation_architecture.py:174-185`
filters split members to surviving children and degrades the split to `[]`
when fewer than 2 groups survive or coverage is incomplete
(`split_degraded_to_no_split`). A degraded split on an *approve* becomes an
ordinary applied package. Inert today because all five canary splits are
rejects, but QP5 changes what "surviving children" means — watch that
counter.

**Driver check.** Per the proposals: apply each split group that carries a
driver; groups without a driver stay standalone. Driver membership comes
from `PackageCandidate.driver_work_item_ids` intersected with the group.

---

## 6. QP4 — multi-bath expansion

**v4's expansion is a good template, and it already partitions.** The
helper is `_build_expanded_bathroom_package`
(`rehab_packages.py:3177-3297`) — there is no `_build_surrogate_package`
(the proposals name it wrongly). It rebuilds every evidence-derived field
from that surrogate's refs: evidence items filtered by
`ref["room_surrogate_id"] == surrogate_id` and confirmed issue (`:3209-3231`),
`estimate_unit_id` cleared and `source_room_surrogate_ids` set to the single
surrogate (`:3233-3236`), all id lists re-projected (`:3237-3271`), and
`retier_audit` popped so the re-price runs fresh (`:3272-3281`). The
anti-naive-clone guard is the hard skip when `builder_inputs` has no entry
for the source package (`:3402-3412`, `fallback_reason="missing_builder_inputs"`);
the docstring at `:3343-3348` names inherited pricing as "the defect this
rework removes". Per-surrogate re-tier: `_derive_surrogate_builder_inputs`
(`:3300-3324`) then `_retier_package_from_confirmed_evidence` (`:2332`),
wired at `:3439-3458`.

Two v4 behaviours to reconsider rather than copy: it expands **at most one
package per house** (`:3398`, only the first candidate — justified in the
comment at `:3394-3397`), and the metadata cap truncates by **lexicographic
surrogate id** (`sorted_surrogates[:cap]`, `:3428-3429`), not by evidence
strength. The lexicographic truncation is arbitrary; prefer strongest
evidence.

**Why partitioning is mandatory in v5 — the collapse happens twice.**
Conditions group by `(catalog_item_id, unit_id)` (`conditions.py:166`), so
all refs from every bathroom surrogate in one unit land in one condition
with a single `room_surrogate_id` (`:187`) and a *union*
`source_room_surrogate_ids` (`:191`). Work items then union again across
conditions (`work_items.py:127-131`, `_dedup_work_items` `:277-285`;
`WorkItem` docstring `contracts.py:383-390` says a merged active "spans
every colliding source's conditions, catalog items, and estimate units").
Combined with the cost floor at `package_candidates.py:318-321`
(`low = max(unfloored_low, child_low)` where `child_low` sums the *full*
child set), an unpartitioned clone floors at the merged price regardless of
tier — k× billing, exactly v4's named defect.

**The partition data exists.** `EvidenceFacts.evidence_refs` carries
per-photo `room_surrogate_id`, built at `conditions.py:156-163`, persisted
through `evidence.py:187`, and whitelisted by the closed-field gate
(`_EVIDENCE_REF_FIELDS`, `validators.py:112-114`). Already reachable in the
candidate layer at `package_candidates.py:100`. **Caveat:** only `issue_id`
and `photo_key` are required non-empty (`validators.py:489-491`);
`room_surrogate_id` may legitimately be `""` when the photo is not in the
surrogate map. Treat empty-surrogate refs as non-qualifying, as v4 does
(`rehab_packages.py:3161-3163`). This is also the answer to the strict
qualifying rule: require each surrogate to carry its **own** refs from
supported modernization conditions, never the union field.

**Where to hook.** `build_package_candidates` signature is
`package_candidates.py:223-232`; the construction loop is `:293-349` and the
whole-home rollup is appended at `:351-353`. Expansion belongs after `:349`
and before `:351`.

**`bathroom_metadata_cap` needs a contract change, not just a stamp.**
Source in v4: `tools/estimate_units.py:463-476` (full+half baths, ceil;
fallbacks; `None` if absent). It is currently persisted nowhere in v5 (zero
hits across `tools/renovation_architecture/`). `build_package_candidates`
does **not** receive `property_metadata` (confirmed at the call site,
`runtime.py:308-313`, while the two upstream steps do get it at `:292` and
`:303`), so the cap must ride the result. `derive_standalone_estimate`
(`work_items.py:331-337`) does receive `property_metadata`, and the result
is assembled at `:400-413` — but both key sets are closed and enforced:
`_STANDALONE_RESULT_KEYS` (`validators.py:248-250`, enforced `:2047`) and
`_STANDALONE_ESTIMATE_FIELDS` (`validators.py:156-160`), with
`StandaloneEstimate` a frozen dataclass (`contracts.py:607-621`) and
`derive_standalone_estimate` self-validating at `work_items.py:414-421`.
Budget for a contracts + validators change, and note downstream key sets
inherit (`_PACKAGE_REVIEW_RESULT_KEYS`, `validators.py:254-257`).

**Replay Sol-inheritance.** Clones are candidates the stored run never
judged, so `_map_stored_decisions` drops them
(`replay_renovation_architecture.py:156-167`,
`candidates_dropped_no_stored_decision`) and the replay scores zero
bathroom packages. Add an inheritance rule — a derived candidate inherits
its merged parent's stored decision — under a **new sanitization counter**
so the report shows how many decisions were inherited rather than measured.
The same mechanism serves QP5's sub-packages; build it once.

### The ship gate cannot be evaluated as written — Steven's call

Gate S3 / roadmap gate 2 reads "exactness ≥ v4's 6/17, over ≤ 1/17". Those
counts are over all 17 §10 multi-surrogate cards, but **7 of the 17 are
production listings that are not in the canary and cannot be replayed.**
The replayable subset is the 10 canary rows, where the same §10 table gives:

| arm | exact | over | under |
|---|---|---|---|
| v4 (canary subset) | 4 | 1 | 5 |
| v5 today (canary subset) | 0 | 0 | 10 |

**Proposed restatement, preserving the gate's intent ("beat v4's exactness
without beating its over-billing"): ship QP4 iff exactness ≥ 4/10 and
over-billing ≤ 1/10 on the canary rows.** This needs Steven's explicit
confirmation before scoring, because it changes the stated denominator.
Do not silently score 10 rows against a /17 threshold. The canary rows and
their human targets are: redfin_25809814 → 2, redfin_125970550 → 4,
redfin_10803207 → 3, redfin_10952874 → 2, redfin_125779232 → 2,
redfin_126224899 → 3, redfin_126418713 → 2, redfin_80877597 → 2,
redfin_166147710 → 3, redfin_11185681 → 2.

Two of those (redfin_125779232, redfin_80877597) are the properties whose
audit-record wording was corrected by the P5 item 6 erratum — they are QP4
gate evidence, not benign-cleanup examples
(`docs/DESIGN_renovation_architecture_decision_packets.md` §7 item 6).

---

## 7. Corrections this document makes to earlier records

Carry these forward; do not re-derive from the superseded sources.

- The C-only handoff presented **QP3 scoped-vs-generic as an open choice**.
  It is closed by S2. Only QP6's scoping (S5) is open.
- The C-only handoff carried the proposals' warning that **QP6 shrinks the
  whole-home rollup**. It cannot — the rollup takes `turnover`-category
  contributors only (`package_candidates.py:369-373`).
- The QP5 proposal text describes redfin_11079485 as **approve-with-split**;
  the stored decision is `reject` with a split attached. All five canary
  split cases are rejects.
- The QP4 proposal names the v4 helper **`_build_surrogate_package`**; it is
  `_build_expanded_bathroom_package` (`rehab_packages.py:3177`).
- The QP5 proposal says **four** eligibility recompute sites; there are five
  (§5).

---

## 8. Validation

Replay is free and offline. Run one replay per change, in order, each
immediately after its change lands, so the delta against the previous replay
attributes to that change alone. The final cumulative run **is** the
combined wave-1 baseline.

```bash
.venv/Scripts/python.exe scripts/replay_renovation_architecture.py --root artifacts_canary/renovation_session9_20260818/run_1 --label w1_baseline
```

Then `w1_qp3`, `w1_qp6`, `w1_qp5`, `w1_qp4` after each change, same form.
Keep all outputs under one labelled `analysis_*` directory so Session G can
reproduce the chain — the wave-2 canary **must** diff against this baseline,
not the raw session9 numbers, or QP2's effect is conflated with wave 1.

Watch the sanitization counters, not just the dollars. QP3/QP6 should raise
`candidates_dropped_no_stored_decision` and `stored_decisions_unused` by
exactly the gated/suppressed set — anything else is a bug. QP4/QP5 should
show the new inheritance counter, and `split_degraded_to_no_split` should
stay at its baseline value.

Comparator on the combined result, to a scratch path — never the audited
`reports/session9_*`:

```bash
.venv/Scripts/python.exe tools/compare_renovation_architecture_cutover.py --run artifacts_canary/renovation_session9_20260818/run_1/candidate --run artifacts_canary/renovation_session9_20260818/run_2/candidate --freeze artifacts_canary/renovation_session9_20260818/input_freeze.json --report reports_scratch/wave1/comparator.json
```

The Session B comparator keys v4 packages by `package_id`, carries dollars
and 2f `verification_status`, and has a Sol-flip section — use it to confirm
that only the intended candidates disappeared and that their children
reappear as standalone scope rows at the same dollars. Note package review
items are one-sided by construction post-P5 (v4 and v5 ids never match);
pairing lives in `scripts/cluster_session9_reviews.py`.

Suite green (`.venv\Scripts\python.exe -m pytest`, baseline 2387 passed /
6 skipped) before each commit. Tests to add: QP3 gate (opportunity-only
gated; any `package_driver` untouched; `multiple_package_support_same_estimate_unit`
untouched; kitchens/bathrooms untouched), QP6 predicate at the chosen scope,
QP5 sub-package materialization on a reject-with-split fixture plus the
snapshot/bijection invariants holding, QP4 per-surrogate partition and
re-price (a clone must **not** carry the merged child sum) plus the cap and
the strict qualifying rule rejecting empty-surrogate refs.

---

## 9. Constraints

- Nothing ships before end of 2026-08-28. Code may land; the ship commit
  waits.
- Frozen inputs untouchable: canary root, `reports/review_*`, audited
  `reports/session9_*`.
- Never `git add .` / `-A` — the repo root carries ~110 untracked scratch
  files. Stage named paths. (Known untracked code that is *not* yours to
  commit: `scripts/analyze_session8_canary.py`,
  `scripts/build_retag_queue.py`, `scripts/build_session9_decision_packets.py`,
  `scripts/retag_tally.py`, `configs/renovation_architecture_cutover.json`.)
- `scripts/replay_renovation_architecture.py` stays provider-free.
- Do not bump `PASS_2F_PROMPT_VERSION`; do not delete the dormant per-item
  Pass-2f scaffolding.

## 10. Exit criteria

1. QP3, QP6, QP5, QP4 landed v5-path-gated, each in its own commit(s) with
   tests; suite green.
2. Five replays run (`w1_baseline` → `w1_qp4`) with per-change deltas
   reported; comparator scratch run reviewed.
3. S5 recorded in the §9 decision log; the QP4 gate denominator confirmed
   with Steven and the gate table scored against it.
4. QP4 ships only on its gate; if held, say so explicitly and record that
   the wave-1 baseline is C+D only, because Session G's comparator needs to
   know which baseline it received.
5. Session F is unaffected and may run in parallel. Write the next handoff:
   Session G (wave-2 canary) needs both this session's baseline and Session
   F's variant scorecard, so it can only be written once both exist — if F
   is still open, write the wave-1 half and say so.
