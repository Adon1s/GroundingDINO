# Session B — QP8 inspection-lane visibility audit

Written 2026-08-27 · read-only audit (backend artifacts + FE repo
`C:\Users\Steven\IntelliJProjects\renointel-prod`) · **route to the
FE-contract task list** (the v5-adoption discussion at roadmap gate 5).

## Question

v5 routes `cannot_assess` → disposition `inspection`
(`disposition.py:34-35`), the inspection totals lane is contractually 0/0
(`contracts.py:154-161`; `reconciliation.py` never emits
`representation="inspection"`, so the lane sums an always-empty list), and
inspection conditions never become work. Does anything the FE/user sees
consume v5's inspection conditions?

## Volume (reproduced from stored artifacts)

- Canary: 31 (run_1) + 38 (run_2) = **69** inspection dispositions — the
  proposals doc's 69 is both replicas combined.
- Post-cutover production (runs ≥ 20260821_230000): **18** of 409
  dispositions.
- dirB `cannot_assess` human evidence remains thin: 1/4 human-supported
  (`reports/review_analysis.json`, dirb_recovery).

## Answer: nothing user-visible consumes v5's inspection lane — and the FE has no v5 reader at all

- The FE has **zero** references to `renovation_estimate_v5`,
  `condition_dispositions`, `totals.inspection`, `cannot_assess`, or
  `analysis_debug`. The version wall is fail-closed:
  `lib/types/renovationEstimate.ts:792` rejects any version but v3/v4;
  `normalizePhotoIntelRenovationEstimate` (`:864-866`) tries
  `renovation_estimate_v4` → `renovation_estimate` → raw, with no v5 or
  root-placement branch; `lib/analysis/rehabProjection.ts:50` looks for
  `rehab_evidence_projection_v1` only inside v4 containers.
- What the user DOES see is entirely v4-sourced: the desktop split-dollar
  rows "Photo-supported work / + Needs inspection"
  (`components/property/desktop/DesktopInvestmentCard.tsx:199-205`, fed by
  `rehabProjection.ts:156` from v4's `needs_inspection` lane; desktop only —
  the mobile card renders no inspection split), the "Needs inspection
  allowance" line from v4 `totals.inspection_allowance_total`
  (`app/property/[id]/page.tsx:1583-1584`), and the "Potential inspection
  exposure" panel from v4 `totals.risk_exposure_total`
  (`DesktopInvestmentCard.tsx:241-259`). The `inspection_risk` package
  category is deliberately suppressed on every user-facing path (admin
  audit pages only).

So today, every v5 `cannot_assess` and any future QP2-escalated condition is
invisible end to end: reason-coded in the artifact, never work, never
dollars, never rendered. The design's "uncertain conditions remain
auditable" lane exists only inside `photo_intel_debug.json`.

## If the FE adopted v5 with the adapter as it exists

Two independent silent failures stack: no `renovation_estimate_v4` key →
the normalizer returns null and every inspection-allowance line vanishes;
no `rehab_evidence_projection_v1` → the projection is null and
`DesktopInvestmentCard` degrades to the generic "Evidence behind this
estimate" chip list (`:213-225`) — no split, no inspection row, no error.

## Contract hazard (the actionable item for the FE-contract list)

A half-wired v5 adapter that maps the contractual 0/0 `totals.inspection`
into `allocation.raw.needsInspection` would trip `fullyPhotoSupported`
(`lib/property/desktopInvestment.ts:432`, `needsInspection.high === 0`) and
render the affirmative claim *"Estimate includes photo-supported work only;
hidden conditions and inspection findings are not included"*
(`DesktopInvestmentCard.tsx:188-192`) — factually false for any property
carrying inspection dispositions. **The 0/0 lane and that check-marked
assurance string must not ship together.** The v5 FE contract needs either
a real needs_inspection equivalent (derived from inspection dispositions,
not the empty ledger lane) or the assurance line keyed to something other
than a zero that is zero by construction.
