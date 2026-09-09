# Handoff — the 67-item non-estimable class (decision handoff)

> **Session 8 corrections (2026-08-17) — read before trusting this doc.**
> The decision this handoff requested was made and implemented in Session 8
> (`docs/HANDOFF_renovation_architecture_session_8.md`). Four premises below
> did not survive verification:
>
> 1. **The billable class is 48 items, not 67.** 67 is the count of
>    v4-non-estimable items, but 19 of those are ALSO excluded by v5
>    (11 quarantine, 4 generic, 4 no-action); only 48 route to `work`.
>    (Unrelatedly, exactly 67 *other* items carry estimate blocks — a numeric
>    coincidence that likely produced the wrong count.) The corpus dollar
>    figures below are correct; they were measured, not derived from the 67.
> 2. **The Session 7 factor fix was already committed** as `b77b388`, not
>    uncommitted; the canary freeze was therefore already invalid when this
>    doc was written.
> 3. **The "Session 7 plan's deferred list" (Related open decisions) does not
>    exist**, and no session-6 handoff contains a written group-caps
>    decision. The group-caps analysis now exists:
>    `docs/analysis/session8_group_caps_measurement.md` (measured realized
>    effect: 1.22% low / 1.61% high of headline — not a material parity
>    lever).
> 4. **Observation 4 understates the cost of route changes**: any route edit
>    changes the projection fingerprint, which invalidates every Terra
>    checkpoint — route changes are token-negative, not token-neutral (moot
>    once the freeze is dead, as it already was).
>
> Decision outcome (Steven, 2026-08-16): accept the class as intended v5
> scope; triage exactly 5 opportunity/presence items to `no_action` via the
> new non-economic `route_override` catalog field
> (`unfinished_basement_present`, `staging_or_decluttering_opportunity`,
> `mismatched_or_inconsistent_furniture_staging`, `curb_appeal_upgrade`,
> `landscaping_enhancement_opportunity`); keep the full headline. Measured
> effect: −$6,810 low / −$129,767 high; corpus residual vs v4 now
> +7.7% low / +17.6% high.

Date: 2026-08-16. Branch: `renovation_architecture_rework`.
Companions: `docs/HANDOFF_renovation_architecture_session_6_canary_cutover.md`
(canary findings), the Session 7 factor-fix work (committed as `b77b388`; see
`PACKAGE_CANDIDATE_POLICY_VERSION = "package_candidates_v2"`).

**This document records understanding and observations only. It makes no
implementation decision; the follow-up session owns that.** All dollar figures
are from canary replica 1 (`artifacts_canary/renovation_session6_20260816_02`,
18 properties) and are property-cost-factor-scaled work-item dollars unless
noted.

## What the class is

67 of the 128 items in `tools/issue_catalog_kind_v2.json` are **non-estimable
under v4 but produce billable work under v5**. Membership predicate (v4's own
resolution semantics, `resolve_estimate_meta` in `tools/renovation_estimate.py:87-128`):

```python
def v4_estimable(item):
    est = item.get("estimate")
    if not isinstance(est, dict):
        return False                     # no estimate block -> defaults
    tier = est.get("estimate_tier", "minor")
    if tier not in ("high", "medium", "minor"):
        tier = "minor"
    a = est.get("affects_estimate")
    if a is None:
        a = tier in ("high", "medium")
    return bool(a)
```

Class composition by catalog kind: 31 modernization, 20 degradation,
16 defect.

## Why each engine does what it does (both are "by design")

**v4**: the `estimate` block is opt-in per item. A missing block silently
resolves to `estimate_tier="minor"`, `affects_estimate=False`, and the v4
candidate builder only admits high/medium (`tools/renovation_estimate.py:480-481`).
The item then produces **no line item, no dollars, no artifact record at
all** (verified: `unfinished_basement_present` has zero mentions anywhere in
the v4 JSON of the three canary properties where v5 billed it).

**v5**: `resolve_terminal_route` (`tools/renovation_architecture/catalog_projection.py:46-67`)
routes any item carrying a `cost` or `work_item_code` to `work`. Route
distribution over 128 items: 12 quarantine / 4 generic / 5 inspection /
4 no-action / 103 work. This was a deliberate Session 1 decision — the plan
mandates that "missing work/package data must fail validation or map
intentionally to inspection, no_action, or excluded; it must not disappear
through a default", which is precisely what v4's missing-block default does.
The v5 projection ignores `estimate_tier`/`affects_estimate` entirely; the
only estimate-block field it reads is `strategy == "inspect_only"`.

So the class is the collision of two internally consistent conventions:
v4's "no estimate block means not estimable" versus v5's "cost data means a
work route". The dollar consequence only became visible in the canary.

## Measured canary impact (replica 1, 18 properties)

- Class conditions reviewed by Terra: **745 of 1,080 total (69%)**.
  Dispositions: 504 accepted_for_work / 151 excluded / 61 no_action /
  29 inspection. (All 61 corpus no_action dispositions and 151 of the 201
  corpus exclusions come from this class.)
- Active work from the class: **465 work items, $67,105 low / $659,622 high**
  — against $117,882 / $1,106,499 from v4-estimable items.
- Ledger split of the class dollars: **standalone $38,028 / $419,870**;
  absorbed by packages $29,077 / $239,752.
- For scale: after the factor fix, the whole-corpus v5-vs-v4 residual is
  ≈ +$53k low / +$470k high (+8.9% / +24.3%). The class's standalone lane is
  therefore essentially the entire high-side standalone residual.
- 40 of the 67 items produced active work in this corpus; 27 were dormant.

By resolved estimate scope (v5 `estimate_scope` on the class's active work):

| scope | items | low | high |
|---|---|---|---|
| marketability_rehab | 284 | 34,137 | 282,425 |
| required_rehab | 100 | 17,969 | 180,784 |
| optional_value_add | 81 | 14,999 | 196,413 |

Top class aggregates by high dollars (all active work, catalog id | action):

| aggregate | n | low | high |
|---|---|---|---|
| unfinished_basement_present \| DRYWALL_PATCH | 3 | 4,876 | 97,536 |
| landscaping_overgrown_or_neglected \| LANDSCAPE_CLEANUP | 9 | 3,647 | 58,344 |
| peeling_or_discolored_paint | 28 | 4,495 | 56,170 |
| paint_refresh_recommended | 10 | 3,823 | 53,547 |
| damaged_drywall_or_cracks, wall_scuffs_marks_or_dents | 25 | 4,022 | 40,195 |
| curb_appeal_upgrade | 8 | 1,934 | 32,231 |
| worn_or_stained_flooring | 46 | 6,406 | 31,913 |
| garage_or_basement_damage \| FOUNDATION_REPAIR | 2 | 1,273 | 23,868 |
| soffit_or_porch_ceiling_weathered | 8 | 2,604 | 22,792 |
| dated_interior_trim | 54 | 4,492 | 22,477 |

## Observations that constrain or inform the decision

1. **The class is heterogeneous.** It spans presence/opportunity observations
   (`unfinished_basement_present` — an "opportunity for future finishing"
   priced $2k–40k base allowance), routine cosmetic wear (scuffs, dated trim),
   and items that resolve to **required_rehab scope with $181k high** —
   meaning v4's convention also silently hides what v5's scope resolver
   considers required work (e.g. severity-driven defects whose estimate block
   was simply never authored). The exclusion and the inclusion are each
   defensible for different subsets; neither engine's behavior is uniformly
   "the correct one" across all 67.

2. **Package coupling is asymmetric between engines, and it is the
   structural crux.** In v4, package inference consumes issues with no tier
   gate (`_OPTIONAL_TIER` in `tools/rehab_packages.py:108` is a dead
   constant): 24 of the 67 items appear in v4 package support lists in this
   corpus — `wall_scuffs_marks_or_dents` in 38 of 123 packages,
   `dated_interior_trim` in 28. v4's economic semantics for the class are
   therefore **"may feed packages, can never bill standalone."** In v5,
   packages absorb only work-item IDs, and only ACTIVE work items become
   children (`tools/renovation_architecture/package_candidates.py` docstring).
   So the class currently (a) feeds v5 candidate formation as children,
   (b) enters the child-sum cost floor (which can raise a package above its
   tier spec — after the factor fix the child sum wins on the high end in
   15 of 118 applied packages), and (c) bills standalone when uncovered.
   Any change that removes the class's work items also removes their package
   children — package formation, strength, tiers, and floors would shift, not
   just standalone dollars. Route-to-`no_action` is NOT dollar-equivalent to
   v4 parity.

3. **v4's exact economic semantics are not expressible in the current v5
   contracts.** "Work that exists for package formation but never bills
   standalone" has no representation: the ledger's representations are
   standalone / absorbed_by(package) / inspection / no_action, every ACTIVE
   work item gets exactly one entry, and suppressed work items cannot be
   package children. Reproducing v4's behavior would require a contract/ledger
   extension (a new representation or work status with its own reason codes,
   validator support, and reconciliation rules), which is why it is a
   decision and not a patch.

4. **Route changes alone save no Terra tokens.** Disposition policy runs
   after the model verdict (`tools/renovation_architecture/disposition.py`
   maps verdict × route; conditions are built and reviewed for every route).
   The 69% share of reviewed conditions coming from this class is a large
   token observation for the API-cost thread, but harvesting it would mean
   filtering before condition review — a separate contract decision that
   touches Session 2's "every input condition ends in one explicit terminal
   disposition" gate.

5. **The catalog change path is fixed.** `tools/issue_catalog_kind_v2.json`
   is generated from v1 (`tools/issue_catalog.json`) + `tools/catalog_migrations/kind_v2_decisions.json`
   by `scripts/migrate_catalog_kind_v2.py`; migrated items keep v1 economics
   byte-identical and `tests/test_catalog_kind_v2.py` pins regeneration
   parity. Economic/estimate-block edits therefore go into v1 and regenerate.
   Direct v2 edits fail parity. v1 is also what production (legacy_v1) loads,
   so v1 edits need a v1-consumer audit — for reference, v4 reads the
   estimate block only through `resolve_estimate_meta`, and adding estimate
   blocks to v1 items would *activate* them in v4, which is itself a behavior
   change.

6. **Scope-lane mechanics already exist if presentation is part of the
   answer.** The v5 headline sums every estimate scope
   (`tools/renovation_architecture/work_items.py:394-397`), but each work item
   carries `estimate_scope`, and `standalone_estimate.totals_by_estimate_scope`
   plus the ledger preserve the split — a decision that moves
   `optional_value_add` (or any scope) out of the headline needs no new
   derivation, only a totals/presentation rule. Note the class is NOT
   coextensive with `optional_value_add`: only 30% of its high dollars are
   optional scope; 43% resolve to marketability and 27% to required.

7. **Terra review does not gate this class.** Presence/wear claims are easy
   to support photographically; the class's acceptance rate (504/745 ≈ 68%)
   is essentially the corpus rate (71.4%). Whatever the decision is, it must
   be deterministic policy — the review layer will not filter these out.

8. **Batching/freeze.** The canary freeze is already invalidated by the
   Session 7 factor fix. Whatever this decision produces should batch with it
   (and with the deferred group-caps decision, see the Session 6 handoff)
   before paying for the two-replica rerun.

## The option space as observed (mechanical consequences only, no preference)

- **v4-parity routing** (route class → no_action): removes standalone AND
  package-child participation; v5 packages would form from a smaller work
  pool than v4's issue pool — likely fewer/weaker packages than v4, not
  parity (observation 2).
- **Teach `resolve_terminal_route` to read the estimate block**: same
  consequence as above; also imports v4's silent-default convention into the
  engine whose Session 1 mandate rejected silent defaults.
- **Author estimate blocks / explicit routes item-by-item in v1**: 67
  individual product decisions; activates any newly-blocked items in v4 too
  (observation 5); the only option that resolves the class's heterogeneity
  (observation 1) at the item level.
- **Contract extension for package-feed-only work**: reproduces v4's
  economic semantics exactly; new work status or ledger representation,
  validators, reconciliation rules, and reason codes (observation 3).
- **Accept as intended surfaced scope**: no code change; the +24.3% high
  residual becomes largely a deliberate product position and moves to the
  per-property delta review for judgment.
- **Scope-lane presentation change**: orthogonal partial lever; only touches
  27% of the class's high dollars unless combined with something else
  (observation 6).
- **Per-item triage of worst offenders only** (e.g. the basement item via
  the v1 no-economics route): bounded, but leaves the class question open
  and repeats per item.

These are not mutually exclusive; several combine.

## How to recompute

All numbers derive from the replica-1 artifacts: per property, the newest
run dir under `artifacts_canary/renovation_session6_20260816_02/run_1/candidate/<key>/`,
v5 result at `photo_intel_debug.json → analysis_debug.renovation_estimate_v5.result`
(work_items / coverage_ledger / condition_dispositions / package_applications),
v4 beside it in `photo_intel.json → renovation_estimate_v4` (packages,
`supporting_catalog_item_ids`, `final_rehab`). Class membership via the
predicate above over `tools/issue_catalog_kind_v2.json`. Candidate rebuilds
are deterministic and free (no provider calls), so counterfactuals can be
replayed offline against the stored standalone results.

## Related open decisions (not this document's scope)

Group-caps parity, the Terra 2.5M/day ceiling denominator, and the
listings/day target — see `docs/HANDOFF_renovation_architecture_session_6_canary_cutover.md`
and the Session 7 plan's deferred list.
