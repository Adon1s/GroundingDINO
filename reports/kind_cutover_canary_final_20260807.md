# Kind Ontology v2 Canary — FINAL (18 of 18 properties)

2026-08-07. Both sides complete: 18 baseline (pinned a1972cf) + 18 candidate
(observation_kind_v2), same 572 stored photos, identical model map
(Terra 2a/2b/2c/2f, local Qwen 1a/2d). Data:
`kind_cutover_canary_final_20260807.json`. Supersedes the 16/18 preliminary.

## Gate results

| Gate | Result |
|---|---|
| Representative count (≥18) | **PASS** — 18/18 compared |
| Stale / deprecated ids | **PASS** — zero across all properties |
| Unresolved successor ids | **PASS** — zero |
| Mixed-ontology payloads | **PASS** — every candidate artifact stamped observation-kind-v2 / 3.0 |
| Estimate coverage (≤5 pp drop) | **PASS** — no property failed |
| Unresolved rate (≤2 pp rise) | **FAIL on all 18** — denominator effect; see analysis |
| Headline delta (≤15 % unapproved) | **26 bound-breaches on 15 properties** — awaiting written approval |

## Unresolved rate — rate up, coverage up (decision required)

Aggregate: issues 2,061 → 3,329 (+62%, atomic Pass 2b), resolved 1,369 →
1,676 (**+22% absolute**). Unresolved defects fell sharply (the defect lane is
cleaner); the unresolved surplus is fine-grained modernization/degradation
style-and-layout claims with no catalog concept, which never reach estimates
(estimate-coverage gate passed everywhere).

Decision options:
- (a) approve the deviation for this cutover and re-scope the gate to absolute
  resolved coverage for future comparisons;
- (b) treat as blocking and tune 2c exclusions / 2e speculation gating first.
Recommendation: (a), with the 2c/2e display-noise tuning tracked as a
post-cutover item (it does not affect money surfaces).

## Classification changes — 15 matched claims, all ontology-correct

upgrade→modernization 9 (popcorn/textured ceilings, dated railing),
defect→degradation 5 ("roof appears weathered", "hardwood floors heavily
worn/scratched/scuffed", "faint ceiling discoloration near fan"),
upgrade→degradation 1 ("wall paint appears uneven"). Every change matches the
approved reclassification policy; no misfiled failures found. Kind mix shifted
defect 73%→12% share — v2 defect means failed function/safety only.

## Headline deltas — 26 bounds on 15 properties (written approval list)

Mechanically explained: ex-upgrade wear items reprice ×0.6→×1.0 (degradation)
and +22% more resolved issues feed estimates. Direction mostly up. Review-first
cases (largest / structural):

| Property | Low Δ | High Δ | Note |
|---|---|---|---|
| redfin_166147710 | −33% | −24% | lost 5 packages incl. exterior_repair + bathroom_repair |
| redfin_11077450 | −40% | +16% | range widened; kitchen_modernization → living_modernization swap |
| redfin_81000709 | −12% | −17% | major-system stratum, down |
| redfin_10806500 | +92% | +111% | modernization-heavy; degradation reprice |
| redfin_80925528 | +58% | +39% | exterior weathering reprice |
| redfin_127468088 | +52% | +32% | lost exterior_repair; up anyway |

Full per-property table in the JSON (`final_rehab` per row).

## Package selection — 16 of 18 properties changed

Patterns: kind-driven repair↔modernization lane swaps (wear → repair lane,
dated → modernization lane); split-successor ids correctly replacing parents
in package keys; two properties lost exterior_repair (166147710, 127468088);
11077450 swapped kitchen_modernization for living_modernization. All changes
require written approval per the 4A plan.

## Sign-off checklist (Steven)

- [ ] Unresolved-rate gate: option (a) or (b) above.
- [ ] Headline deltas: per-property approval (record in
      `configs/kind_ontology_cutover.json` `approved_headline_deltas`) or
      rejection with reason.
- [ ] Package changes: approve the kind-driven lane-swap pattern + the four
      named structural cases.
- [ ] Kind-mix shift (defect 73%→12% share) acknowledged as intended.

### Sign-off record (2026-08-21, Steven — written waiver)

Recorded through the independent renovation-architecture cutover review
(`docs/DECISION_renovation_architecture_session9_cutover_20260821.md` §5 C1):
the architecture cutover (`KIND_ONTOLOGY_VERSION=observation_kind_v2` +
`RENOVATION_ARCHITECTURE_MODE=new`) knowingly carries the Task 4A v4-on-v2
deltas as the transitional display until the frontend adopts v5. The boxes
above are left as originally written; this block is the decision.

- Unresolved-rate gate: **option (a)** — approve the deviation; re-scope the
  gate to absolute resolved coverage for future comparisons.
- Headline deltas: **waived**, not per-property approved. Under catalog 3.1
  (what deploys) the open list is 23 rows on 14 properties
  (`reports/kind_cutover_canary_catalog31_20260808.json`), not the 26/15 of the
  3.0 table above; `approved_headline_deltas` in
  `configs/kind_ontology_cutover.json` is deliberately left `{}`.
- Package changes (lane-swap pattern + four structural cases): **waived** —
  v4-on-v2 packages are transitional display; v5 package formation replaces
  them once the FE adopts v5.
- Kind-mix shift (defect 73%→12%): **acknowledged as intended**.

On approval: merge/deploy FE branch `kind_ontology_v2_compat`, then enable
`KIND_ONTOLOGY_VERSION=observation_kind_v2` in production per
docs/RUNBOOK_kind_ontology_release_rollback.md, and start the 7-day
observation window.
