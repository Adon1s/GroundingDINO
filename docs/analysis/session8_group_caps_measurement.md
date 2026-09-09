# Session 8 — group-caps parity measurement (decision memo)

Date: 2026-08-17. Branch: `renovation_architecture_rework`.
Method: `scripts/analyze_session8_canary.py` over the Session 8 baseline
replay (`scripts/replay_renovation_architecture.py --label baseline`,
replica-1 artifacts, current code, all 18 properties, reconciliation audits
empty). Sidecar: `artifacts_canary/renovation_session6_20260816_02/analysis_session8/group_caps.json`.

## What was measured

v4 caps every estimate group with `GROUP_BUDGET_CAPS`
(`tools/renovation_estimate.py:737`): per group, dominant stack behavior
(`max_only > group_cap > sum` over the members), `group_cap` resolves to
`max(min(sum, cap), best_item)` per endpoint, and the property cost factor
scales the whole estimate afterward (`tools/renovation_estimate_v4.py:337-338`).
v5 has no cap mechanism — every group sums uncapped.

This simulation applies the full v4 stack rule to v5's ACTIVE standalone work
items, with the caps scaled by each property's cost factor (equivalent to
v4's cap-then-scale ordering).

**Honesty buckets.** Only bucket (a) is capped: work items whose catalog
items v4 actually priced (estimate block with tier high/medium — every such
item carries an explicit `group`). Bucket (b) — the v4-invisible class and
ungrouped items — is reported but never capped: v4 never priced these, so
blanket-applying the `"other"` (200, 5,000) fallback would fabricate a parity
target v4 never enforced. Attribution was exact in this corpus: zero merged
work items mix estimable and non-estimable catalog items, zero members span
two groups.

## Result

Two measurements: what caps remove at the **work layer**, and what actually
reaches the **published headline** after package absorption (packages bill
`max(tier spec, owned-child sum)`, so capping an absorbed child changes
nothing unless the child sum was the binding term).

| | low | high |
|---|---|---|
| Removed at the work layer (bucket a, corpus) | 16,845 | 47,813 |
| …neutralized by package absorption | −8,926 | −9,148 |
| **Realized reduction in the published headline** | **7,919** | **38,665** |
| Corpus headline (baseline replay) | 648,180 | 2,406,165 |
| Bucket (b) dollars, never capped by v4 | 67,105 | 659,622 |

Realized removal is **1.22% low / 1.61% high of the pre-triage v5 headline**;
adopting caps would move the corpus residual vs v4 from +8.9%/+24.3% to
+7.5%/+22.3%. For scale, the Session 8 five-item triage removed 129,767 high
from bucket (b) — about 3.4× the entire realized cap effect.

> **Correction (2026-08-17):** an earlier revision of this memo stated
> "~0.7% low / ~1.7% high". Those figures divided the low removal by the
> *high* headline and the high removal by the *stored* (pre-factor-fix)
> headline. The work-layer removal is 2.6% low / 2.0% high of headline; the
> realized post-absorption removal is the 1.22%/1.61% above. The
> recommendation is unchanged.

By group (corpus removed, low/high, member work items):

| group | low | high | items |
|---|---|---|---|
| structure | 4,102 | 30,693 | 17 |
| flooring | 9,309 | 14,491 | 85 |
| bathroom | 1,887 | 2,629 | 40 |
| kitchen | 380 | 0 | 32 |
| landscaping | 986 | 0 | 15 |
| remediation | 181 | 0 | 4 |

12 of 18 properties see a high-side removal of $0; the worst single property
is redfin_125970550 at 2,797/14,942.

Note on the low side: most low-side "removal" comes from v4's
`min(sum_low, cap_low)` rule — the cap ceilings the LOW endpoint too (e.g.
structure cap_low 1,000). That is v4's actual behavior, reproduced here, but
it reads as an artifact of the rule rather than an intended budget floor.

## What v5 already does (why the effect is small)

v5 is not uncapped in general — `tools/costing.py:286` applies each catalog
item's own `cap_low`/`cap_high` across occurrences, and max-envelope dedup
prevents same-unit double billing. So "one condition firing in twelve rooms"
is already bounded. What v4's group cap adds on top is a **cross-item**
ceiling: many *different* catalog items in one trade category each firing at
their own cap (85 flooring work items in this corpus). That residual
overcounting risk is what the 1.22%/1.61% measures.

## Caveats

- v4's caps operate over its own candidate pool; v5's work pool differs
  (dedup, per-unit splits). The simulation answers "what would v4's rule do
  to v5's items", which is the relevant question for a v5 cap step, not a
  byte-parity reconstruction of v4 groups.
- Package application status is dollar-independent (it follows decisions and
  ownership priority), so capping cannot change which packages apply — the
  realized simulation above is exact given that, not an approximation.
- The low side is proportionally the harsher endpoint (2.6% vs 2.0% at the
  work layer) because v4's rule ceilings the LOW endpoint too
  (`min(sum_low, cap_low)`, e.g. flooring cap_low 2,000). Adopting parity
  imports that behavior, which reads more like a rule artifact than an
  intended budget floor.

## Implementation sketch (if parity is adopted)

Priced so the decision includes its cost:
- Extend `work_policy` in `catalog_projection.py` with `group` +
  `stack_behavior` (new keys ⇒ bump `PROJECTION_VERSION` to
  `renovation_catalog_projection_v3`; all checkpoints invalidate — free while
  the freeze is already dead).
- Deterministic capping step inside `derive_standalone_estimate` between
  dedup and scope totals, allocating each group's removal pro-rata across
  member work items so ledger/reconciliation arithmetic and the validators'
  exact-sum checks stay consistent; bump
  `STANDALONE_PRICING_POLICY_VERSION` to `standalone_pricing_v2`.
- Validator + pin updates (scope totals, headline identity, new audit field
  for capped dollars with reason codes per the suppressed-work precedent).

Estimated scope: a focused session (projection, work_items, validators,
tests), no contract schema bump.

## Decision (Steven, 2026-08-17): intentional divergence — DO NOT build caps

v5 ships without group caps, deliberately. Rationale: the realized effect is
1.22%/1.61% of headline; v5's per-item occurrence caps and max-envelope
dedup already handle the overcounting risk caps existed for; adopting parity
would import uncalibrated inherited numbers and v4's min-on-low artifact.
Group budgets may be revisited inside the deferred price-calibration
project, designed rather than inherited. The measurement harness
(`scripts/analyze_session8_canary.py`) stays for that day.

Rejected alternatives, for the record:
- Adopt v4-parity caps (recovers 7.9k/38.7k realized; imports the artifacts).
- Modified caps now (price calibration is out of migration scope).
