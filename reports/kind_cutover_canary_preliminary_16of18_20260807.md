# Kind Ontology v2 Canary — PRELIMINARY (16 of 18 properties)

2026-08-07. Candidate side paused at 16/18 for API credits; missing:
`redfin_11077450`, `redfin_11185681` (both interior_cosmetic). All conclusions
provisional until they complete. Companion data:
`kind_cutover_canary_preliminary_16of18_20260807.json`.

## Gate results

| Gate | Result |
|---|---|
| Stale / deprecated ids | **PASS** — zero across 16 properties |
| Unresolved successor ids | **PASS** — zero |
| Mixed-ontology payloads | **PASS** — all candidate artifacts stamped observation-kind-v2 / 3.0 |
| Estimate coverage (≤5 pp drop) | **PASS** — no property failed |
| Representative count (≥18) | FAIL — 16/18, by design of the pause |
| Unresolved rate (≤2 pp rise) | **FAIL on all 16** — see analysis: denominator effect |
| Headline delta (≤15 % unapproved) | **23 bound-breaches on 12 properties** — awaiting review |

## Unresolved rate: rate up, coverage up

Baseline resolved 1,240 of 1,816 display-lane issues (68%); candidate resolved
1,518 of 2,931 (52%). **Absolute resolution rose 22%** — candidate resolves
more real work than baseline on every stratum — but v2's atomic Pass 2b emits
61% more claims, and the extra claims are dominated by fine-grained
modernization/degradation observations with no catalog concept
(unresolved-by-kind: 766 modernization, 501 degradation, 146 defect — note
unresolved *defects* fell 354 → 146, a much cleaner defect lane).

Sampled unresolved candidate claims are style/layout notes ("hall has limited
natural light", "sink and toilet appear closely arranged", "cabinet hardware
consists of small dark knobs") — display-lane noise, not lost work: estimates
consume only resolved issues, and the estimate-coverage gate passed.

Two follow-ups this raises:
1. The gate as written measures a *rate*, which structurally penalizes
   atomization. Options: re-scope to absolute resolved coverage, or accept
   with written approval and re-tune post-cutover.
2. Some sampled claims arguably belong in 2c's excluded lane
   (neutral_presence / not_renovation_related) or under 2e's speculation gate
   rather than the display lane — a prompt/2e tuning question, not a resolver
   defect. Recorded here; not a 4A change.

## Matched-claim classification changes: 12, all ontology-correct

Transitions: upgrade→modernization 9, defect→degradation 2,
upgrade→degradation 1. Every sampled change matches the approved policy:
popcorn/textured ceilings → modernization (dated finish); "roof appears
weathered" and "flooring heavily worn and stained" → degradation (wear band);
"metal railing is dated" → modernization. No misfiled failures observed.

Aggregate kind mix shifted defect 73%→12% share (degradation 44%,
modernization 42%) — the redefinition working as intended: v2 defect means
*failed* function/safety only. Diagnostic, explained by the ontology.

## Headline deltas (awaiting per-property written approval)

12 of 16 properties breach ±15% on low and/or high. Direction is mostly UP,
explained by two compounding mechanisms: ex-upgrade items reclassified to
degradation now price at ×1.0 (was ×0.6), and +22% more resolved issues feed
the estimate. Largest movers:

- `redfin_10806500` +92% low / +111% high (modernization-heavy; degradation reprice)
- `redfin_80925528` +58% / +39%
- `redfin_127468088` +52% / +32%
- **DOWN**: `redfin_166147710` −33% / −24% (lost 5 packages incl. exterior_repair
  and bathroom_repair) and `redfin_81000709` −12% / −17% — the two cases most
  needing individual review, both major_system_defect stratum.

## Package selection: 14 of 16 properties changed

Recurring patterns: bathroom/bedroom repair↔modernization lane swaps
(kind-driven; a wear item now lands the repair lane where an "upgrade" landed
modernization, and vice versa for dated items), split-successor ids replacing
parents in package keys (e.g. `hard_flooring_scratched_or_worn` replacing
`scratched_or_damaged_flooring` — the migration working), and two properties
losing `exterior_repair` outright (166147710, 127468088 — review with the
headline drops above).

## Status

Preliminary. Next: finish `redfin_11077450` + `redfin_11185681` (~250 Terra
calls), regenerate the full 18/18 report, then Steven's written review of
headline deltas + package changes and a decision on the unresolved-rate gate.
