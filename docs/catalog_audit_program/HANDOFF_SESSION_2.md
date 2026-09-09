# Handoff — catalog audit session 2

Date: 2026-09-04

Session/task title: Semantic audit and proposal drafting (`02_SESSION_SEMANTIC_AUDIT.md`), Phases A–C

**Status: COMPLETE.** Every exit criterion in the brief is met. 20 clusters each carry exactly one outcome, all 63
coverage rows are triaged, both native proposals dry-run clean in memory, and the decisions file and both catalogs are
byte-identical to the session start.

## Outcome

Phase A built the tooling and 17 provisional clusters. Phase B was the pinned photo/lineage review. Phase C, recorded
here, reshaped the clusters, assigned outcomes, triaged coverage, produced two native proposals with verified
in-memory dry runs, and separated the Pass 2d and Pass 2a work the 2026-09-03 direction puts out of scope.

Final validation: `validation final ok=True errors=0 pending=0`, and `--check` reports the JSON and Markdown both
match what is on disk.

| Outcome | Count | Clusters |
|---|---|---|
| `native_decisions_proposal` | 2 | CAP-007, CAP-008 |
| `migration_system_gap` | 1 | CAP-013 |
| `non_catalog_action` | 6 | CAP-001, CAP-003, CAP-011, CAP-012, CAP-019, CAP-021 |
| `no_change` | 7 | CAP-002, CAP-004, CAP-005, CAP-006, CAP-009, CAP-015, CAP-017 |
| `deferred_insufficient_evidence` | 4 | CAP-010, CAP-014, CAP-016, CAP-018 |

Evidence bars: 6 `two_independent_human`, 1 `structural_exception`, 13 `none` (each carrying a non-native outcome
that is refutation-backed or a documented deferral). Mismatch classes: 15 `operationally_consequential`,
3 `operationally_equivalent`, 2 `exact_match`. Coverage triage: 19 `not_billable_observation`,
14 `covered_by_existing_item`, 12 `coverage_gap_candidate`, 11 `product_quarantined`, 6 `unclear`,
1 `attaches_to_cluster`. 26 Pass 2d/retrieval follow-ups in 16 bounded confusion families; 28 rows in the Pass 2a
regression register.

**The catalog-first result is that the catalog is mostly not what failed.** Only two clusters justify a catalog
change, and both carry `confidence: low` with a blocking question for the human gate. Thirteen clusters resolve to
`no_change` or `non_catalog_action` with the real defect routed to the Pass 2d worklist or recorded as a verification
error, which is what the 2026-09-03 direction asked for.

## Repository state

- Starting commit: `9afe0fa5a0490a856abed507dccc4a02ed1de24c`, branch `terra_factorized_verifier`
- Ending commit: unchanged, nothing committed (Steven decides)
- Dirty state: no modified tracked files at start or end (`git status --porcelain -uall` on tracked paths is empty);
  all session files are untracked working-tree files pinned by hash below
- Pre-existing changes preserved: yes

## Inputs verified

| Input | Expected | Actual | Result |
|---|---|---|---|
| `reports/catalog_audit_evidence.json` | `43354104…29b7`, fingerprint `e4c0a2b8…0c08` | same; `--check` exit 0 | match |
| `reports/catalog_audit_photo_review_packet.json` | `7b4621df…bc80` | same | match |
| `reports/catalog_audit_photo_review.json` | `2225db9c…9150` | same | match, byte-preserved |
| `docs/catalog_audit_program/RECONCILIATION_SESSION_2_PHASE_B.md` | `35534d8f…4b67` | same | match |
| `tools/catalog_migrations/kind_v2_decisions.json` | `47614d82…03e8` | same before and after every run | match |
| `tools/issue_catalog_kind_v2.json` | `51bf7e26…aa54` | same before and after every run | match |
| `tools/issue_catalog.json`, generator, validator | bundle `sources` hashes | same | match; in-memory `generate()` reproduces the on-disk 3.2 catalog |
| Steven's decision docs (`HANDOFF` §rulings, `00`, `02`) | on-disk state at 2026-09-03 12:55 | `f1578718…`, `5fde64ba…`, `661202aa…` | unchanged during the session |

## Files created or changed

| Path | Status | sha256 | Purpose |
|---|---|---|---|
| `reports/catalog_audit_proposals.json` | rewritten (final) | `1411352c42e9d3a555c3a1561470323da2234447badc2869c9874c3e8fa3ef99` | 20 clusters + 63 coverage rows, judgment + derived |
| `docs/PROPOSAL_catalog_audit_20260901.md` | rewritten (final) | `771572e7c1d095281756b6baf9f28e65cffedc7f9b04cabfd559117768b17591` | deterministic rendering, appendices A–H |
| `scripts/render_catalog_audit_proposals.py` | extended | `d0305682810c8a9b5d79d9d9e47c8e6ecf277fa117fe723aa9de65debc2c6cb9` | pins, bindings, refutation bar, mismatch/deferral fields, registers |
| `tests/test_catalog_audit_proposals.py` | extended | `2ede383facd502fa2b03e36faba73b8396fb415438fc9ba1c08f7c9265b95cea` | 34 tests |
| `docs/catalog_audit_program/HANDOFF_SESSION_2.md` | rewritten (final) | — | this file |

The filename of the proposal document still derives from `proposal_date` (`2026-09-01`), deliberately left unchanged
so the document is not orphaned and `--check` keeps comparing the same path.

## Steven's Phase C adjudication and convention decisions (2026-09-02)

These are authoritative human inputs, carried forward verbatim in substance. They do not modify the pinned Phase B
review, which Phase C preserved byte-for-byte.

- When catalog fields disagree, judge semantic fit against the **strictest claim-bearing wording**. Terra receives the
  `atomic_claim`; name/description/embedding/retrieval text can separately create a catalog-coherence or Pass 2d
  problem.
- Judge disjunctive claims and multi-photo conditions **at the characteristic issue/photo level**. A true branch or one
  supporting photo does not rescue a differently named mechanism on another constituent issue; a broad-but-true
  mapping can remain a Pass 2d specificity problem.
- **Plain/basic/builder-grade presence is not billable by itself.** A condition needs visible datedness,
  deterioration, incompleteness, or another concrete modernization need.
- **Dated style bills on concrete tells only** (2026-09-02): an identifiable era-specific finish or visible
  deterioration qualifies; a bare "neutral a buyer would repaint" does not.
- **Blinds** stay detectable but must be a `no_action` catalog condition; the ruling is about blinds only, and
  valances and curtains remain open.
- The five reconciled cases (entry steps not billable; ordinary mini-blinds not billed; deck weathering real but a 2d
  specificity question; the door surround is casing/jamb trim and never siding; heavy dated trim is catalog-owned
  wording) were each used and are recorded as `human_inputs` provenance on the clusters that relied on them.

## Steven's Phase C scope direction (2026-09-03)

Catalog-first intervention; Pass 2a frozen and its cases retained only as regression controls; Pass 2d deferred with
reachable-right-item cases recorded for a later bounded review; operational equivalence judged on subject, repair,
scope, billability, trade/route and safety; internal catalog precision separated from UI granularity; work performed
cluster by cluster in bounded families. All of this is encoded in the artifact's `policy` block
(`scope_direction`, `operational_equivalence`) and enforced by the validator.

## Work completed

- **Tooling.** The renderer gained: sha256 pins on the review and packet with fail-closed `verify_pins`; a `--packet`
  refusal while the pinned review exists; `--review` restricted to the pinned path; a guard snapshot extended to the
  evidence bundle, packet and review; review-row binding (`review_row_id`, `transfer_note`, `cluster.reshape`,
  `controls.review_source_cluster`) so a unit keeps the row it was actually reviewed under when a cluster is
  reshaped; refutation-backed non-native outcomes with `reviewed_refutations` / `lead_refutations` / `mixed_signal`;
  a structural-exception guard; `mismatch_class` + `equivalence_dimensions`; `deferral_reason`; `pass_2d_follow_ups`
  and `regression_controls` with aggregated registers; an id ledger; and rendering for all of it
  (Appendix G = Pass 2d/retrieval worklist, Appendix H = Pass 2a regression register).
- **Analysis.** Work was fanned out over five semantic families and six coverage concept groups, each family
  producing a structure plan and then one authoring agent per cluster, followed by adversarial verification on up to
  three lenses (evidence and binding, scope and ownership, expressibility and regression), tiered by stakes.
  Findings and verdicts are preserved outside the repo in the session scratchpad; the lead's corrections are recorded
  separately so provenance stays separable from adjudication.
- **Reshaping.** CAP-002 split three ways (retained: the shed rot unit; **CAP-018** stone billed as brickwork;
  **CAP-019** door casing billed as siding fading). CAP-006 split two ways (retained: the bathroom component cluster;
  **CAP-021** plumbing valve trim resolved to interior millwork). Every moved unit is bound to its original CAP-002 or
  CAP-006 review row with a transfer note, and the new clusters read controls from their source cluster.
  **CAP-020 was reserved and deliberately not allocated**: the windows family found the evidence did not support a
  second cluster there. Ids are never reused, so CAP-020 remains permanently unallocated.
- **Coverage.** All 63 rows triaged, then independently re-checked; the checkers returned "revise" on all six groups
  and their material objections were adjudicated by the lead (below).

## Decisions and invariants

- **Evidence classes and the bar.** `human_review` and `gold_reference` are human evidence; `model_judge` leads never
  satisfy the bar and never ground a non-deferred outcome. Only units with role `support` adjudicated exactly
  `supports_claim` count.
- **Refutation-backed outcomes (Steven, 2026-09-02).** `no_change`, `non_catalog_action` and `migration_system_gap`
  may rest on a reviewed human/gold refutation with photos available. `native_decisions_proposal` still requires a met
  bar. `unclear`, `unavailable`, missing and model-only rows never qualify.
- **Review binding.** A unit's adjudication is read from the row named by `review_row_id`; cross-cluster binding
  requires `cluster.reshape` plus a per-unit `transfer_note`; one reviewed row grounds exactly one cluster;
  adjudications are never rewritten.
- **Packet drift is expected and is now `true`.** The pinned packet remains the reviewed one; reshaping changes what a
  freshly generated packet would contain. `load_review` still pins the results to the packet file's bytes, so the
  review binding is intact. Do not regenerate the packet.
- **Operational equivalence and mismatch class** are recorded on every cluster; `equivalence_dimensions` is non-empty
  exactly when the class is `operationally_consequential`.
- **Vocabulary additions** (deviations from the original brief, recorded below): `CAUSES` gained
  `upstream_pass_2a_observation`; new judgment fields `mismatch_class`, `equivalence_dimensions`, `deferral_reason`,
  `human_inputs`, `pass_2d_follow_ups`, `proposal.regression_controls`, `proposal.target_items`.
- **Id ledger.** `KNOWN_IDS` = CAP-001…CAP-017 plus CAP-018, CAP-019, CAP-021. A ledger id may never disappear from
  the artifact and a new id must exceed the ledger maximum.

## Facts the lead verified mechanically (not argued)

1. **Terminal route precedence** (`catalog_projection.resolve_terminal_route`): quarantine → `drop_if_generic` →
   `inspect_only` → `route_override` → no-economics → work. Of 128 items, 98 bill; 30 do not, across five reasons.
   The full map is reproducible from the catalog and was used to settle every billability dispute.
2. **Two checker corrections were rejected as factually wrong.** `dated_exterior_finishes` is not a "billing twin" of
   `curb_appeal_upgrade`: it carries `drop_if_generic: true`, which outranks `route_override`, so it resolves to
   `excluded_generic` and never bills. And "a `defaultHidden` cover does not bill" is false: `defaultHidden` is a
   display flag with no role in routing, so those items bill but are not shown.
3. **CAP-020/CAP-007 routing.** `dated_or_older_windows` has neither a `cost` block nor a `work_item_code`, so it
   resolves to `no_action` / `no_economics_approved_gap` — one of four deliberate approved economics gaps. The unbilled
   window condition could never have billed whatever Terra said. This corrected an earlier lead hypothesis that
   blamed tier-optional routing.
4. **All four candidate native operations dry-run native**, pass both validators, preserve item order and leave the
   guarded files byte-identical. Expressibility only; not evidence that any change is warranted.
5. **CAP-007's blocking unknown is resolved.** The author deferred on whether the item's only human-approved billing
   (`rc_7c5b9f9bb3aa`) would survive the split, because its 2b bullet is absent from the evidence bundle. Read from the
   pinned, hash-verified run artifact (`…/redfin_81000709/20260817_225858_d13bd83c/photo_intel_debug.json`, sha256
   `05738c0e…`, `verified: true`): the bullet is **"The blinds are old."** — blinds only, no fabric term — and Terra's
   rationale is "Yellowed horizontal blinds and an improvised window covering are visible." So the blinds successor's
   `deny_any` never fires and the split **does** withdraw that billing. See the unresolved questions.
6. **Coverage-triage ordering.** Billability is a precondition for quarantine: labelling non-billable observations as
   `product_quarantined` overstates what the electrical quarantine costs. Five rows were reclassified accordingly.

## Lead corrections applied on top of analyst findings

Analyst findings are immutable provenance; every lead change is separately recorded.

| Target | Change | Why |
|---|---|---|
| CAP-008 | regression controls restated (7 entries), regression risk rewritten as a full ledger, confidence → low, 3 blocking questions added | The proposal withdraws three human-approved billings and risks flipping a correct rejection; it is not the clean win the draft implied |
| CAP-007 | regression risk rewritten with the resolved artifact fact, confidence → low, blocking question restated for Steven | The author's top risk is now answered, and answered against the proposal as drafted |
| CAP-013 | the missing-baseboard gold unit bound in from the coverage lane with a transfer note | It is literally the cluster's subject and was triaged to it by the coverage group |
| coverage ×5 | 3 electrical rows reclassified off `product_quarantined`; the cornice row re-filed from `attaches_to_cluster` to `coverage_gap_candidate` | The checkers' material objections were correct; CAP-002 was narrowed and no longer hosts the cornice |
| 9 clusters | long prose confusion-family names normalised to slugs, prose preserved in the note | Steven asked for bounded families; 16 usable families now group 26 follow-ups |

Verifier `required_changes` that are prose refinements rather than validity or outcome problems were deliberately not
applied; they remain in the findings files and are handed to Session 3, which is the adversarial-review session by
design.

## Deviations from the session brief

| Deviation | Reason | Impact |
|---|---|---|
| New judgment fields and one new `CAUSES` value | The 2026-09-03 direction requires recording equivalence, deferral reasons, 2d follow-ups and 2a regression expectations | Additive; all existing rules still enforced |
| Refutation-backed non-native outcomes | Steven's 2026-09-02 answer | Recorded in `policy.evidence_bar` and `policy.non_deferred_outcomes` |
| Three validator rules relaxed during authoring | A cluster may name itself as a reshape source (documents "same id, narrowed claim"); `better_rank` may be null when the better item was absent from the frozen pool (a reachability finding, not a selection failure); `regression_controls` may name any bundle-known card, not only hallucination and correct-rejection cards | Native proposals still must cover every hallucination and correct-rejection control |
| Analysis fanned out over subagents | Steven's ultracode instruction | Findings and verdicts preserved outside the repo; provenance separable |
| CAP-020 reserved but never allocated | The windows family found no second cluster was warranted | Permanent gap in the id sequence; ids are never reused |

## Commands and verification

| Command | Result |
|---|---|
| `scripts\build_catalog_audit_evidence.py --check` | exit 0, `fingerprint matches: e4c0a2b8…0c08` |
| `scripts\render_catalog_audit_proposals.py` | `validation final ok=True errors=0 pending=0`, exit 0 |
| `scripts\render_catalog_audit_proposals.py --check` | `proposals json matches; markdown matches; validation final ok=True errors=0 pending=0`, exit 0 |
| `scripts\render_catalog_audit_proposals.py --packet` | refused, exit 1: "pinned review … exists on disk; regenerating the packet would orphan it" |
| `pytest tests/test_catalog_audit_proposals.py` | 34 passed |
| `pytest` proposals + evidence + kind_v2 + validation + renovation_architecture_catalog + product_quarantine | **295 passed, 1 skipped** |
| Protected-file rehash, start vs end | all 16 unchanged (evidence, packet, review, reconciliation, decisions, both catalogs, generator, validator, Steven's docs) |
| In-memory dry runs, CAP-007 and CAP-008 | both `ok=True`, all ops `native`, `order_ok=True`, zero metadata problems, zero catalog and manifest validator errors |

## Outputs for the next session

| Artifact | Hash | Contract |
|---|---|---|
| `reports/catalog_audit_proposals.json` | `1411352c…3ef99` | 20 clusters, one outcome each; 63 triaged coverage rows; `human_disposition` blank throughout |
| `docs/PROPOSAL_catalog_audit_20260901.md` | `771572e7…b17591` | reviewable document; Appendix G is the Pass 2d/retrieval worklist, Appendix H the Pass 2a regression register |
| `reports/catalog_audit_evidence.json` | `43354104…29b7` | unchanged Session 1 bundle |
| `reports/catalog_audit_photo_review.json` | `2225db9c…9150` | unchanged pinned Phase B review |

## Unresolved questions and risks

1. **CAP-008, blocking before implementation.** Does the rewritten claim newly *accept* `rc_73d15ca14405` (heavy
   period trim in clean condition, correctly rejected today)? Period millwork is era-specific, so the claim's own tell
   may collide with a correct rejection. Only a live Terra call on those two photos settles it. If it fires, the
   proposal manufactures a hallucination and must not be implemented as drafted.
2. **CAP-008, for the human gate.** The proposal withdraws three human-approved billings
   (`rc_9e889631af67`, `rc_ceeb17f8e242`, `rc_d0ffde500a92`). That is convention 3 applied to labels that predate it,
   not a defect repaired. If those billings should stand, the proposal falls and the divergence becomes documentation.
3. **CAP-007, for Steven alone.** Convention 5 was given against ordinary, clean, intact mini-blinds. The item's only
   human-approved billing is also a blinds billing, but Terra's rationale there cites *yellowed* blinds — visible
   deterioration, not plain presence. Does the `no_action` ruling extend to visibly deteriorated blinds? If yes, the
   proposal is right and the lost billing is the convention being applied. If no, the blinds successor needs a
   deterioration tell in its atomic claim and the proposal must be redrafted.
4. **Retrieval effects are reasoned, never measured.** `embed_text` is the entire embedding input for an item and no
   offline embedding store exists, so every post-change ranking claim in both native proposals is an expectation.
   Session 5 Tier 2 is where they become measurable.
5. **CAP-014** holds 12 follow-ups on ten model leads with the single human/gold unit refuted; it is deferred
   `lead_only` and would move with two human reviews on different properties.
6. **A frozen-attribution discrepancy** worth recording in the error-attribution record: `rc_e9d27cf6e9aa` is
   attributed to stage 2d, but the reachability scan shows the better item was kind-gated out of that bullet's
   candidate set entirely.
7. **`tub_surround_or_shower_pan_damage` has zero reviewed controls of any kind**, so any future proposal touching it
   starts from an empty control family.

## Required next task

Objective: Session 3, adversarial review, per `03_SESSION_ADVERSARIAL_REVIEW.md`, against the artifacts above.

Required inputs: `00_OVERALL_CONTEXT.md`, `03_SESSION_ADVERSARIAL_REVIEW.md`, this handoff, the pinned evidence
bundle, the pinned review, and the proposal JSON + document. Session 3 must not need this session's conversation.

Prohibited: editing the review, evidence, migration decisions, either catalog, prompts or runtime; provider, pipeline
or live-retrieval calls.

Exit criteria: every proposal id receives a disposition; the two native proposals are attacked on the blocking
questions above; no-change and deferred clusters are reviewed for false negatives on the same evidence standard.

## Do not redo

- Do not regenerate the evidence bundle or the packet. The packet is pinned to the review by hash; `--packet` is
  refused by the tool while the review exists.
- Do not edit `reports/catalog_audit_photo_review.json`. Adjudications are facts.
- Do not re-review all 191 photos. Two photos and one pinned run artifact were reopened in Phase C, each recorded with
  the question it answered.
- Do not count model leads as human evidence, and do not treat a lead-only cluster as eligible for a non-deferred
  outcome.
- Do not turn Pass 2a hallucinations into a remediation workstream, and do not modify Pass 2a.
- Do not modify Pass 2d in this program; the bounded worklist is Appendix G.
- Do not quote replayed dollar totals as a headline while catalog 3.2 acceptance is outstanding.

## Suggested opening prompt for the next task

> Execute Session 3 of the catalog-audit program, the adversarial review. Read `00_OVERALL_CONTEXT.md`,
> `03_SESSION_ADVERSARIAL_REVIEW.md`, `HANDOFF_SESSION_2.md`, and the hash-verified
> `reports/catalog_audit_proposals.json` plus `docs/PROPOSAL_catalog_audit_20260901.md`. Verify every pinned hash
> first. Give each of the 20 proposal ids a disposition. Attack the two native proposals hardest: CAP-008 on whether
> its rewritten claim newly accepts the heavy-period-trim correct rejection and on whether withdrawing three
> human-approved billings is acceptable, and CAP-007 on whether convention 5 extends to visibly deteriorated blinds
> given that the split withdraws the item's only demonstrated correct billing. Also review the no-change and deferred
> clusters for false negatives on the same evidence standard. Do not edit the review, evidence, decisions, either
> catalog, prompts or runtime, and make no provider or retrieval calls.
