# Result — wave 1 complete (QP3 only)

Date: 2026-08-27
Branch: `renovation_architecture_rework`

## Final scope

Steven narrowed wave 1 to QP3 after review of the combined C+D+E handoff.
QP6, QP5, and QP4 were not implemented and are deferred outside wave 1. This
record supersedes the combined handoff as an execution charter.

## Implemented policy

An approved `bedroom_modernization` or `living_modernization` candidate does
not apply when `proposed_treatment` is
`opportunity_driver_with_corroboration` or
`opportunity_driver_with_multiphoto_corroboration`.

The gate is application-side and v5-only. Package candidates, Sol decisions,
request membership, and snapshot hashes remain unchanged. Gated packages
absorb no work and carry the explicit reason
`opportunity_only_interior_modernization`; each child remains standalone at
its exact allowance unless another eligible approved package legitimately
owns it. Defect-driven packages, driverless support packages, kitchens, and
bathrooms are unchanged.

## Verification

- Focused reconciliation + contract suites: 146 passed.
- Broader v5 package/runtime/canary suites: 218 passed.
- Full suite: 2,395 passed, 6 skipped.
- Offline replay: all 18 stored canary properties completed with no provider
  calls and no disposition changes.
- Exactly 8 candidates changed from `applied` to `not_applied`; application
  counts moved from 112 applied / 7 rejected to 104 applied / 7 rejected /
  8 QP3-gated.
- Candidate and decision counts remained 119/119. Every replay sanitization
  counter remained zero and every reconciliation audit list remained empty.
- Corpus headline delta versus stored v5: −$20,574 low / −$46,869 high. The
  change is removed package uplift; child work remained covered.

Replay artifacts and the per-property report are under
`reports_scratch/wave1_qp3_final/`. That replay is the completed wave-1
baseline for subsequent comparisons.
