# Proposals — output-quality improvements from the completed manual review

Written 2026-08-26 · backend `renovation_architecture_rework` · commissioned by
`docs/HANDOFF_quality_improvement_proposals.md` · evidence =
`reports/review_analysis.md`/`.json` (frozen, queue sha `8512b87c2af1`, verdicts
sha `0c7ca6a8b55d`) read against the v5/v4 source and the Session 9 decision
packets. Every proposal here was drafted by this session and then
**adversarially verified** (one skeptic per proposal + a completeness critic,
each reading the repo independently); the corrections they surfaced are folded
in and the surviving caveats are stated inline. **Nothing here is shipped or
decided — the observation window is open until end of 2026-08-28, and every
option call below is Steven's.**

---

## 0. What "quality" is taken to mean here

Steven's recorded frame: **(1) fewer hallucinations, (2) more useful
observations that build packages**; prices provisional; dedup deflation
accepted; price calibration is a separate thread. Against that frame, the
review evidence decomposes "quality" into four measured defect classes, which
is what these proposals target:

| # | defect class | measured size | where it lives |
|---|---|---|---|
| D1 | **False billed conditions** | weighted broad-error on billed conditions: canary 13.6% [6.8–26.6], production 10.4% [5.1–32.5]; hard-false 10.5%/10.4% | degradation-kind wear/stain claims carry 66%/83% of error mass; dominant qualitative mechanism = **mis-description of a real scene** (stains→"scuffs", intentional yellow→"yellowed", dated→"worn"): claim_wording 15/27 coded notes |
| D2 | **Lost real conditions** | dirB (targeted): 20/51 Terra rejections were human-supported; +6 overstated-but-real billed claims | same mechanism, opposite direction — several dirB notes say "it does need replacing" while the claim named the wrong object/material |
| D3 | **Package-level failures** | P1 census 11 warranted / 11 not; Sol approve = 9/10 (no warrant signal); 21/26 billed error conditions rode inside packages (64.8% of canary error mass billed through packages); Sol reject-with-split zeroed a human-warranted package (P07) and caused a −52% headline | v5 has no package-level photo gate by design; Sol judges grouping coherence only |
| D4 | **Multi-bath under-billing** | v5 under-bills 16/17 multi-surrogate cards, unit gap +25 (v4: 6 exact / 10 under / 1 over, +16); humans said `per_bathroom` 14/17 | deliberate conservative bathroom-surrogate merge (`estimate_units.py:192-205`) + no v5 expansion stage |

Two facts shape everything below:

- **The 223 human verdicts are a tuning asset, not just a report.** They join
  to the *stored canary run_1* conditions. Upstream replica churn (35%/51%)
  means they will **not** transfer to any fresh canary — so anything that wants
  to be scored against ground truth must be scored **before** the next canary,
  via re-decide on the stored inputs (QP1).
- **The FE still renders v4.** No v5 quality change is user-visible until the
  FE-adoption call (gated on the P1/P3 option calls). That makes the
  post-window weeks a low-risk period to land v5-side work.
- **Recommendation, now explicit (amended 2026-08-26): hold the FE on v4**
  until wave 1 + wave 2 land and the fresh canary passes. The hold has a real
  cost — v4's Pass 2f keeps burning the *binding* Sol quota (~5–6 premium
  listings/day) — and a natural expiry: FE adoption → 2f sunset (≈3× Sol
  headroom) → the S8 per-item-2F decision.

---

## 1. The proposal set at a glance

| id | proposal | defect | cost class | when |
|---|---|---|---|---|
| QP1 | Re-decide harness scored on the frozen human labels | enabler | build = free; run ≈ 0.9M Terra tokens/pass (+0.9M control arm) | build now, run post-window |
| QP2 | Terra rubric v2 (mechanism-match) + catalog claim-wording fixes | D1, D2 | live Terra change → **one** fresh freeze + two-replica canary (~8M tokens, ~4 free days) | wave 2 |
| QP3 | Interior-modernization application gate (opportunity-only drivers → standalone) | D3 | offline replay, code only | wave 1 |
| QP4 | Deterministic multi-bath package expansion (per-surrogate re-priced) | D4 | offline replay, but a real build (re-pricing + id scheme + replay extension) | wave 1 build; ship on replay gate |
| QP5 | Sol split → deterministically applied sub-packages | D3 (goal-2) | offline replay, contract-touching | wave 1 |
| QP6 | Single-child candidates, type-scoped: interior repair → never a package | Sol variance | offline replay, code only | wave 1, rides along |
| QP7 | Defer the package-warrant photo veto; productionize disagreement telemetry; per-item-2F decision point | D3 | no build / review-side | standing |
| QP8 | Inspection-lane visibility audit (`cannot_assess` → where does it surface?) | D2 | read-only artifact audit | now (window-safe) |
| QP9 | P5 review-tooling fixes + the sequencing/attribution rules in §3 | meta | tooling only | now (window-safe) |
| QP-M | Round-2 measurement: uniform rejected-condition + package-candidate arms | measurement | review-side | with S7 round 2 |
| QP10 | Do-nots (negative space) | — | — | standing |

Recommended priority: **QP1 and QP2 are the program** (they attack D1/D2 where
66–83% of the error mass sits); QP4 is the largest single billing-accuracy gap;
QP3/QP5/QP6 are cheap bounded wins; the rest is hygiene. QP2 without QP1 is
ship-and-pray; QP1 without QP2 is a harness with nothing to measure.

---

## 2. The proposals

### QP1 — Build the re-decide harness; score prompt variants against the frozen human labels

**What.** A sibling script (e.g. `scripts/redecide_renovation_architecture.py`)
that rebuilds each stored canary run_1 Terra unit request from
`observed_conditions` + `evidence_facts` (reusing `build_unit_request` /
`call_terra_review` / `parse_unit_reviews`), re-submits it live — with the
current or a candidate v2 prompt — and scores the verdicts against
`reports/review_verdicts.jsonl`. **Not** a mode of
`scripts/replay_renovation_architecture.py`: that module's documented invariant
is that it is structurally incapable of spending provider tokens; keep it that
way.

**Scoreable population (corrected).** Canary run_1 condition cards only — 125
cards: 32 dirA, 36 dirB, 40 uniform, 13 flip-only, 4 p6. Within them: 17
billed-error cards (**11 hard-false + 6 overstated**), 55 human-supported
billed cards, 16 dirB human-supported (recovery targets). The 48 production
condition cards and all package/bathroom cards are out of scope (production
runs aren't in the canary; packages/bathrooms aren't Terra verdicts).

**Scoring rule (corrected — this is the part that protects goal 2).** Split
hard-false from overstated: a variant scores a *win* when a hard-false card
flips to unsupported, a *loss* when a human-supported billed card flips away
from supported, and the 6 overstated-but-real cards are a **cost** if deleted
(they are real conditions; the right outcome for them is re-worded claims —
QP2's catalog half — not exclusion). dirB recovery (16 cards) scores goal-2
gains. Because the Terra replica flip noise floor is 6.4% (16/250; humans
matched run_1 vs run_2 8:6 — a coin toss), every comparison needs a
**same-prompt control arm** and thresholds that beat ~1–2 chance flips on the
decisive ~27-card subset.

**Mechanics that must be honored.** (a) Every variant bumps
`TERRA_REVIEW_PROMPT_VERSION` or carries its own fingerprint namespace — the
system-prompt *text* is not fingerprinted; (b) never read/write the canary
`.checkpoints` dirs (same-fingerprint reuse would silently return stored
verdicts and measure nothing); (c) debit the shared Terra ledger
(`usage_guard.py`) — one 0.9M pass/day coexists with production (~1.1–1.3M/day
upstream); two don't; (d) assert photo byte-identity via the stored
`request_fingerprint` before trusting a re-decide (the images live in the
mutable renointel-prod tree); (e) **run live passes after 2026-08-28** — C5
monitors daily ledger spend, so an in-window 0.9M debit perturbs a monitored
invariant. Building in-window is safe (the script is not freeze-hashed).

**Variant scope (amended 2026-08-26).** The harness covers prompt/*payload*
variants — including a **remove-upstream-observations arm** (the cross-session
anchoring-risk hypothesis: Terra may anchor on the upstream sentence it is
shown; testable here because payloads are rebuilt from stored inputs, and
risky to assert untested because observations also *localize* generic claims —
removal could raise false positives or cannot_assess). It does **not** cover
unit-structure variants (pre-Terra bathroom units change batching and evidence
grouping). All variants are harness-side overrides; production
`terra_review.py` is untouched until wave 2.

**Cost.** ~0.9M Terra tokens per full 18-listing pass; ~1.8M per
variant-vs-control comparison ≈ one free Terra day each. Share the
variant-label/ledger/no-checkpoint scaffolding so the P4 **Sol** consistency
probe can plug in later — QP1 delivers the Terra half of the packet's
"re-decide mode", not all of P4.

**What closes it.** Harness + scoring script committed; one control-arm run
demonstrating the noise floor; a scorecard format Steven can read per variant.

---

### QP2 — Terra rubric v2: mechanism-match + updated-room contradiction; catalog claim-wording fixes ride the same canary

**The flagship goal-1 change.** Two rubric rules, one catalog pass, one canary.

**(a) Mechanism-match rule (degradation claims).** The verdict must match the
*claimed mechanism*, not just the general area: staining, soiling,
discoloration, or intentional color is not wear/damage **unless the claim
itself is about staining/soiling** — `worn_or_stained_flooring` legitimately
bills on stains, so the rule is keyed to the claim text, *not* blanket-applied
to the degradation kind (verifier correction). In-repo template: the 2f
exterior rule "Distinguish damage from soiling… confirm only when accompanied
by visible material failure" (`scene_classifier_passes.py:1815-1817`) — the
only text in either architecture that names this exact failure mode, which is
also Steven's own diagnosis ("often confuses staining with worn").

**(b) Updated-room contradiction rule (style claims).** A `*_dated_style` /
`outdated_*` / `dated_*` claim is unsupported when the room presents
predominantly current finishes; a single isolated dated element does not carry
a dated/style claim against an otherwise updated room. Template: the 2f
bathroom contradiction rule (`scene_classifier_passes.py:1674-1679`). Target:
the measured style cascade — 7 of 17 canary billed-error records are one
listing's (redfin_25809814) dated/style claims across bedrooms and kitchen,
all 2f-objected, all absorbed into modernization packages. (Honest caveat:
that is ONE listing; the rule is fitted to few exemplars and QP1 is what keeps
it honest.)

**Prompt shape — two variants, the harness decides (amended 2026-08-26).**
Default = the light shape: the two rules above plus "name what you actually
see" in the rationale. The heavy variant (from the cross-session review) asks
per-condition sub-answers — subject match / claimed-state match / location
match / evidence sufficiency — and derives the final verdict deterministically
from them. Better-structured mis-description signal, at the cost of larger
responses per condition, stricter parsing, and a derivation table for
disagreeing sub-answers. Run both as QP1 arms; ship the winner.

**Delivery.** Rules as per-condition `verification_guidance` keys in the
payload dict (`terra_review.py:143-150` — free-form JSON, automatically
fingerprinted) plus system-prompt text; either way **bump
`TERRA_REVIEW_PROMPT_VERSION`**. Also instruct Terra to name in the rationale
*what it actually sees* (material/object) — the rationale is a free-text,
zero-schema-change channel; mining it supplies the mis-description analytics.

**The "misdescribed" verdict lane — recommend NOT now.** A fourth verdict
(`present_but_misdescribed` → inspection) has **billing outcomes identical to
prompt-only hardening** (both route the condition out of billed work; work
items only build from `accepted_for_work`, the inspection lane is contractually
0/0). What it buys is signal separation — at the cost of breaking the closed
`REVIEW_VERDICTS` frozenset, the `decide_disposition` if-chain,
`DISPOSITION_REASON_CODES`, a `CONDITION_DISPOSITION_POLICY_VERSION` bump that
makes validators reject every stored v1 artifact (re-validation tooling must
learn versioned acceptance), and a lane whose precision the harness cannot
measure (ground truth = ~15 coded notes from a 3-verdict review). Rationale
mining gets most of the analytics for free. Revisit the lane only if mining
shows material volume.

**(c) Catalog claim-wording fixes — same canary, zero marginal quota.** The
catalog `atomic_claim` text is a Terra model input (`_claim_text`,
`terra_review.py:76-90`), so wording fixes are canary-class — which means they
are **free riders on QP2's canary** and pointless to ship separately. From the
coded notes: mechanism-honest wording for the stain/wear family; split the
combined `dated_wood_paneling` item (wall paneling vs cabinets,
rc_65f169bbe3b4); the "generic presence" note (rc_6f8278034b05). Catalog edits
go through `tools/catalog_migrations/kind_v2_decisions.json` + regeneration,
never hand-edits.

**The goal-2 trade, stated plainly.** Hardening converts today's
billed-but-misworded real problems into unbilled rows (13 of 26 error cards
rode as package_support, 8 as driver). Neither variant recovers them for
billing — recovery would need an item-remap lane (right problem → right catalog
item), which is a bigger architectural change deliberately **not** proposed
here; the catalog wording fixes shrink the class at the source instead. This is
a priority-1-over-priority-2 trade and it is Steven's to accept.

**Validation & proof.** Tune on QP1 (headroom ≈ 1–2 full passes/day alongside
production). Ship via **one** fresh freeze + two-replica canary (~8M Terra
tokens ≈ 4 free-paced days; 2-day paid floor; no code edits mid-canary).
**Budget a second manual review** (~200 cards, ~2.5h of Steven's time) after
the canary: the labels don't transfer to the new canary's conditions, the CIs
are wide, and without a second review the improvement is unverifiable —
harness deltas alone are the fallback, stated as such.

---

### QP3 — Interior-modernization application gate: opportunity-only drivers don't apply as packages

**What.** A deterministic rule: a `bedroom_modernization` /
`living_modernization` candidate whose drivers are all opportunity-kind
(exactly: `proposed_treatment` ∈ {`opportunity_driver_with_corroboration`,
`opportunity_driver_with_multiphoto_corroboration`} — by construction any
defect/degradation driver forces `package_driver`,
`rehab_packages.py:2077-2088`) is **not applied**; children bill standalone at
their exact allowances. Phrased as policy, not a special case: *wholesale
modernization of a bedroom or living room is not a warranted package on
style-only evidence*; kitchens and bathrooms — where modernization is a
standard package — are untouched.

**Evidence.** P1 census: 2f-rejected bedroom_modernization 0/3 and
living_modernization 0/2 warranted (0/5 interior-modernization) vs kitchen 4/1
and exterior 6/4. Fragile N (0/5 has p≈3% under a true 50% rate) and the
census is the contested slice, not the pipeline — but demotion-to-standalone
is not deletion: every observation stays billed, so the harm cap is the tier
uplift.

**Why not the generic tier cap (P1 option d as originally drafted).** Verified:
bedroom/living have **no partial tier** — the escalation chain is
refresh→full_rehab, so "cap at partial" is undefined there and the real
demotion is full→refresh (≈ −60%/package). The application gate avoids that
undefined target, avoids touching bathrooms (where a cap would deepen the D4
under-billing — `outdated_bathroom_finishes` carries default severity 3 and
trips the severe-finish full path while opportunity-only), and kitchens'
severe-finish path already carries a defect-kind item
(`outdated_or_damaged_cabinets`), so a kind-keyed cap is a near-no-op there.
Present the generic per-family "demote one step down the escalation chain" cap
as the alternative if Steven prefers one systematic rule over a scoped one.

**Implementation notes (verified).** v5-side in
`package_candidates.build_package_candidates` (or at application eligibility) —
**gate it to the v5 path**: `_build_package_candidate` is shared legacy code
and an ungated change alters live v4/FE output outside any replay, and v4's
post-2f re-tier (`rehab_packages.py:2401-2412`) would bypass a cap placed only
at candidate build. New application/ledger reason codes if done at
reconciliation; all four eligibility recompute sites must agree. In the
reviewed slice the rule is a no-op on every human-warranted style-only card
(P01 and P12 are already partial tiers) — it binds exactly on P10/P11-class
packages.

**Validation.** Offline replay on the stored canary, own run + the combined
wave-1 run; report package count/dollar deltas. Replay holds stored Sol
decisions fixed (Sol will see gated candidates live — small unmeasured
difference, Sol flip evidence 2/31 suggests minor).

---

### QP4 — Deterministic multi-bath expansion, done properly (per-surrogate re-priced)

**What.** Port the *shape* of v4's `expand_bathroom_modernization_packages`
into v5 at candidate-construction time: when the merged bathroom unit's
evidence shows k ≥ 2 surrogates each independently qualifying, mint one
per-surrogate `bathroom_modernization` candidate (capped at
`bathroom_metadata_cap`), so Sol reviews per-bathroom groupings live and
reconciliation is untouched.

**The three corrections that make this a real build, not a port.**

1. **Per-surrogate re-pricing and child partitioning are mandatory.** v5
   collapses conditions one-per-unit, so work items span surrogates; naive
   cloning gives every clone the merged tier price and full child set, and the
   cost floor + application pricing then bill ≈ k× the merged price — v4's own
   docstring calls exactly this "the defect this rework removes"
   (`rehab_packages.py:3343-3348`). The port must partition evidence refs by
   per-photo `room_surrogate_id`, re-run tier resolution per surrogate, and
   assign each work item to exactly one clone (v4's
   `PackageBuilderInputs`-based re-tier is the template).
2. **Strict qualifying rule.** `source_room_surrogate_ids` is a union over
   refs, so one supported condition with one mis-attributed photo would mint a
   second full package. Require each qualifying surrogate to carry its **own**
   refs from supported modernization conditions (minimum ref/condition count —
   tune on the stored data). The review documented real over-splits (a utility
   room counted as a bathroom surrogate; humans counted fewer baths than
   surrogates on 11216660 3→1, 10866780 4→2, 80917686 6→4), and the metadata
   cap only helps when metadata ≤ human count.
3. **Plumbing.** `make_package_candidate_id` keys on (estimate, type, unit) —
   clones need a per-surrogate token (validators only pin the prefix, so
   feasible). `bathroom_metadata_cap` must be persisted into the result
   (stamp at `derive_standalone_estimate`, where `property_metadata` is in
   scope) because `build_package_candidates`' input contract is the result
   alone. The replay harness needs a **Sol-decision inheritance rule** (clone
   the merged parent's stored decision, tracked under a new sanitization key)
   — without it replay drops every clone and scores zero bathroom packages.

**Framing correction.** This is *not* a design-invariant bug-fix (the merged
unit satisfies "one condition instance per estimate unit"); it reverses a
deliberate conservative merge policy because the humans said so: billing
target `per_bathroom` 14/17, v5 under 16/17, gap +25. That evidence stands on
its own.

**Target architecture (amended 2026-08-26, locked as S9).** P3 option (c) —
resolving distinct bathrooms *before* Terra, one estimate unit and one
independently reviewed condition set per bathroom — is the stated target; QP4
is the constrained interim. QP4's replay result against the §10 human targets
is the cheap experiment that says whether (c) is worth building. Risk (c)
imports, stated now: it makes surrogate clustering load-bearing earlier, and
the review showed surrogates *over-count* real bathrooms (the utility-room
case) — so (c) waits for that evidence.

**Ship gate (my proposed rule).** Combined wave-1 replay scored against the
§10 human distinct-bathroom targets, labelled "under inherited Sol approval":
ship if exactness ≥ v4's (6/17) with over-billing ≤ v4's (1/17). Bounded
downside post-ship: Sol has never seen the clone shape (often single-child —
coordinate with QP6's scoping so bathroom clones are not suppressed); if Sol
rejects clones, children fall standalone ≈ today's under-billing, i.e. the
failure mode is the status quo. Interaction stated honestly: expansion
multiplies whatever false-bathroom-condition rate Terra has (2 of 26 error
cards were bathroom items); Steven can choose to hold QP4's *flip* until QP2's
canary lands — the build is wave-1 either way.

---

### QP5 — Apply Sol splits as sub-packages instead of zeroing the package

**What.** Today an approve-with-split or reject-with-split is non-economic:
the candidate goes `not_applied` and children fall standalone — on
redfin_11079485 that turned a $10,920/25,480 kitchen package into $1,514/21,045
(the −52% headline), and reject-with-splits zeroed three exterior packages
including P07, which the human judged **warranted**. Proposal (from the P4
menu): when Sol returns `split_groups`, deterministically build sub-candidates
from the exact groups, re-run tier/pricing per group, and apply each group
that carries a driver; groups without a driver stay standalone. Sol's split
*is* the recommendation — applying it respects the judgment instead of
discarding it, and it is the one clearly goal-2-positive package change on
the table (more applied packages, from evidence Sol already endorsed).

**Cost.** Offline-replayable but contract-touching: sub-candidate ids, decision
provenance (the sub-package's authority = the parent's Sol split), application
reason codes, the four eligibility recompute sites, validators. Validate by
replay on the stored canary (the split cases are all in it).

---

### QP6 — Single-child candidates, type-scoped: interior repair singles stay standalone

**What.** Do not emit single-child **interior repair** candidates
(bedroom/living repair — the class where both measured Sol flips occurred);
their work items bill standalone via the existing `no_covering_package` ledger
path (no contract change — verified). Keep exterior singles: both
human-warranted single-child P1 cards were exterior_repair, so a blanket rule
would destroy packaging the evidence supports (verifier correction; the
original blanket draft conceded too much goal-2 for a 2-flip problem).

**Effects.** Kills 100% of the measured Sol variance class by construction;
shrinks the Sol listing call (eases the binding Sol quota); side effect: the
whole-home display-only rollup consumes the same candidate list and will
shrink — decide whether it should aggregate suppressed singles anyway. Honest
weight: zero goal-1 contribution; smallest item in the set; rides the wave-1
replay.

---

### QP7 — Package-warrant veto: defer, but keep the question instrumented

**Recommendation.** Do **not** build the P1(b) package-level photo veto this
cycle. The census says no measured judge is reliable on the contested slice
(2f's package rejections were themselves wrong ~half the time; Sol approve is
uninformative), so a veto deletes ~half true packages to remove ~half false
ones — and QP3 already takes the one type-scoped slice with measured 0/5
warrant. Whether QP2 shrinks the not-warranted class (by removing the false
*children* that form false packages) is a **hypothesis the post-QP2 review is
designed to test**, not a fact — stated per the bounded-inference rules.

**Instrumentation (corrected by verification).** The 2f-vs-Terra disagreement
telemetry is *available*, not free or automatic: v4+Pass-2f still computes
alongside v5 (verified: unconditional in `artifact_writers.py` under `new`
mode) and rides the **binding** Sol quota (24–58k/listing of 250k/day — it is
why production caps at ~5–6 listings/day); the review tooling
(`tools/review_cards.py`, `scripts/review_analysis.py`, queue/server/tally) is
**uncommitted** and must be landed; the recurring cost is Steven's review
time; the join needs `photo_intel_debug.json` present and dies under L1
rollback. And the signal's expiry is real: v4/2f retirement is the
un-numbered follow-on gated on FE v5 adoption + observation (not kind-ontology
4B/4C), and when it happens **both** dirA and dirB die. **Decision point for
Steven:** before approving v4 retirement, either accept losing the
second-opinion telemetry or ship the per-item Pass-2F revival (the dormant
scaffolding kept for exactly this) as the replacement — which is also the
natural landing pad if a targeted second-look pass on high-risk conditions is
ever wanted. Sunsetting 2f after FE adoption roughly triples Sol-side
throughput headroom; that is the forcing function.

**Fallback (amended 2026-08-26, part of the S6 record).** If package-level
control is wanted before the post-QP2 evidence, the sanctioned middle option
is a **high-consequence pilot**: a bounded warrant check only on full
modernizations, high-uplift packages, or packages driven by a single weak
condition — veto-to-standalone only, never deletion. Reserved, not default.

---

### QP8 — Inspection-lane visibility audit (window-safe, read-only)

The canary carries 69 `inspection` dispositions; v5's inspection totals lane is
contractually 0/0 and inspection conditions never become work. The v4
split-dollar contract has a `needs_inspection` lane the FE renders. **Audit
whether v5's inspection route feeds anything the FE/user ever sees** — if not,
every `cannot_assess` and future escalated condition is currently invisible,
which quietly wastes the design's own "uncertain conditions remain auditable"
lane and any QP2 routing into it. Also record the (thin) dirB `cannot_assess`
evidence: 1/4 human-supported. Read-only artifact + FE-contract check; do it
now; output = a short note routed to the FE-contract task list.

---

### QP9 — Tooling and sequencing rules

**P5 fixes (before the next canary, unchanged from the packet):** comparator
keys v4 packages by `package_id` (not `type|unit` — expanded packages
currently collapse; fix v5-side keying at `:211` symmetrically); dollars on
package review rows; 2f `verification_status` on package items; Sol flip
detector; disagreement tier in the cluster script. Track packet item 6 (the
two mis-worded headline explanations) somewhere even though it is outside the
close condition.

**Sequencing (corrected).**

- **Now (window, to 2026-08-28):** this document + Steven's option calls; P5;
  QP1 build (no live runs); QP8 audit; the **scene-identity mismatch
  measurement** (amended 2026-08-26: a one-off, read-only count of how often a
  candidate's evidence photos' scene assignments conflict with the package
  room — the P05 "indoor photo in an exterior package" failure mode; NOT a new
  pipeline stage unless the numbers demand one); commit the review tooling
  (QP7); ongoing production review, if wanted, uses a **separate queue** — the
  frozen `reports/review_queue.json` is never rebuilt.
- **Wave 1 (offline, post-window):** QP3, QP4-build, QP5, QP6. Replay is free:
  run **one replay per change plus the combined run** (a single combined run
  cannot attribute a regression); comparator on the combined result; QP4 ships
  on its gate.
- **Wave 2 (live):** QP2 tuned on QP1 (live passes post-window; 1–2 full
  passes/day of Terra headroom), then **one** fresh freeze + two-replica
  canary. **Attribution baseline:** diff the fresh canary against the *stored
  canary replayed through the wave-1 policies* — not the raw session9 baseline
  — or QP2's effect is conflated with wave 1. Then the second ~200-card manual
  review.
- **The thumbnail backend backstop does NOT ride this canary** (correction):
  its own handoff classes it offline-replay-validated, 0/18 canary listings
  are thumbnail-affected, and its checklist expects zero disposition changes
  on canary replay — it is wave-1-class work owned by its own thread,
  strictly non-blocking here.

---

### QP-M — Measurement additions for the round-2 review (amended in 2026-08-26)

Two commitments, folded into the S7 round-2 review (from the cross-session
comparison — the genuine gap in the original set):

- **A uniform sample of rejected conditions.** Terra recall is currently
  unmeasurable (§12; dirB is targeted, 2f-confirmed-only). A uniform
  rejected-arm gives the first recall estimate.
- **A uniform sample of all package candidates.** P1 is a census of one
  contested slice; a uniform candidate arm gives package-warrant precision a
  denominator for the first time.

Recorded as optional notes, **not** commitments: a work-action label (correct
action / wrong trade / excessive scope / no action — adds review time per
card) and listing-clustered intervals (a stats nicety at round-2 n).

**Acceptance-gates caveat.** Aspirational gates from the cross-session review
(hard-false ≤ 5%; package-warrant precision ≥ 85% on the uniform arm; bathroom
exactness-or-explicit-uncertainty > 90%) are recorded as targets — but at
round-2 sample sizes a true 5% vs 10% is barely resolvable, so the round-2
design must do the sample-size math or the gates are unfalsifiable.

---

### QP10 — Do-nots (negative space, corrected)

- **Don't shrink Terra batches for quality.** The batch-size association is
  confounded and *contradicts between the small/mid buckets across sources*
  (canary worst 5–8, production worst 2–4) — though both sources agree 9+ is
  low-error; smaller batches re-send images and raise cost. (Batch
  *enlargement* is a token question for the API-reduction thread, not banned
  here.) Precision on quotas: **Sol** binds production throughput (~5–6
  listings/day); **Terra** is the cost of canary validation and shared
  upstream headroom.
- **Don't add a photo-count gate.** The photo-bucket evidence contradicts
  across sources (canary error mass 75% in 2+ photos; production 91% in
  1-photo — the latter partially entangled with the thumbnail defect), and the
  p6 single-photo items held 4/5 (targeted n=5). Carve-out: the design-doc's
  *escalation lane* for high-consequence single-view conditions stays
  available — this do-not forbids gates, not escalation.
- **Don't target individual catalog items off the concentration tables.** The
  top rows are each one uniform card wide (canary `worn_or_stained_flooring`
  58% contribution = 2 physical cards; production `peeling_or_discolored_paint`
  80% = 1 card, hidden by the md's judged≥3 filter — cite the JSON, not the
  md). The *kind family* signal is robust; item-level targeting isn't. (Item
  claim-**wording** fixes are different — they ride QP2's canary and are
  motivated by the qualitative notes, not these rows.)
- **Don't duplicate or depend on the thumbnail thread** (including any
  evidence-selection change — that class is fresh-canary anyway). Don't treat
  redfin_10965375 / redfin_10922002 artifacts as valid evidence.
- **Don't restructure Sol into a warrant judge this cycle.** Not doctrine —
  evidence and cost: Sol is text-only (it would judge warrant without images);
  no measurement says a warrant-judging Sol would beat its measured
  uninformative approvals; a Sol role change is non-replayable and attacks the
  binding quota. The P1 option call stays Steven's; this is cycle-scoped, not
  permanent.

---

## 3. Decisions — status as of 2026-08-26

Steven adopted the recommendations below on 2026-08-26 (recorded in the
decision-packets §9 log, `docs/DESIGN_renovation_architecture_decision_packets.md`);
S8 stays open by design. Implementation is commissioned across sessions B–H:
`docs/ROADMAP_quality_program_sessions_20260826.md`.

| # | decision | call (2026-08-26) |
|---|---|---|
| S1 | QP2 rubric v2 + catalog wording, goal-2 trade | **Adopted** — ship rules + wording; prompt-only lane, no 4th verdict; shape (light vs structured-fields) decided by the QP1 harness |
| S2 | QP3 scope | **Adopted** — interior-modernization application gate (not the tier demotion) |
| S3 | QP4 | **Adopted** — build; ships only on the replay gate (exactness ≥ v4's 6/17, over ≤ 1/17) |
| S4 | QP5 split→sub-package | **Adopted** — build |
| S5 | QP6 single-child rule | **Adopted** — rule in; blanket vs interior-repair scoping picked at Session C (one predicate either way) |
| S6 | Package-warrant veto (P1(b)) | **Adopted** — deferred; high-consequence pilot reserved as the fallback |
| S7 | Round-2 manual review after wave 2 | **Committed** — rescoped per QP-M (~2.5–3h of review time) |
| S8 | v4/2f sunset linkage | **Open** — decide at FE-adoption time (per-item-2F replacement vs accept telemetry loss) |
| S9 | P3 target architecture | **Adopted** — pre-Terra per-surrogate bathroom resolution is the target; QP4 is the gated interim |

The P6(3) provisional acceptance is **confirmed** in the §9 log on the p6
evidence (4/5 held, targeted).

---

## 4. Provenance

- Evidence: `reports/review_analysis.md` / `.json` (frozen);
  `reports/review_tally.md`; `docs/DESIGN_review_method.md` (semantic
  authority); `docs/DESIGN_renovation_architecture_decision_packets.md` §§1–9.
- Source read: `tools/renovation_architecture/*` (terra_review, evidence,
  conditions, disposition, work_items, package_candidates, sol_review,
  reconciliation, contracts, validators, runtime), `tools/rehab_packages.py`,
  `tools/renovation_estimate_v4.py`, `tools/estimate_units.py`,
  `tools/scene_classifier_passes.py` (2f prompts),
  `scripts/replay_renovation_architecture.py`.
- Constraints: `docs/RUNBOOK_renovation_architecture_session_6.md` §1a,
  `docs/HANDOFF_renovation_architecture_token_budget.md`,
  `docs/analysis/session8_terra_budget_denominator.md`,
  `docs/HANDOFF_thumbnail_photo_ingest_fix.md`,
  `docs/Revised_Renovation_Scope_and_Estimate_Architecture.docx` (design
  intent + invariants).
- Method: 7 parallel source/evidence readers, then 8 per-proposal adversarial
  skeptics + 1 completeness critic (all repo-grounded); every verdict was
  "sound with corrections" and the corrections are incorporated above. This
  session preloaded no observations and inherited no conclusions.
- Cross-session comparison (2026-08-26 amendment): an independent session's
  conclusions were assessed against this evidence. Six amendments adopted
  (explicit FE-hold; remove-observations harness arm; structured-fields QP2
  variant; P3 target architecture / S9; QP-M measurement arms; high-consequence
  pilot fallback + scene-identity check) and one recommendation **rejected**:
  comparing Terra prompt variants only through fresh canaries. Rejection
  rationale, recorded so it is not relitigated: the 223 human labels join only
  to the *stored* canary conditions (35%/51% upstream churn breaks the join on
  any fresh canary), so canary-only comparison measures variants blind at ~8M
  tokens each; the live re-decide harness (QP1) is **not** the deterministic
  replay tool and does measure against ground truth.
