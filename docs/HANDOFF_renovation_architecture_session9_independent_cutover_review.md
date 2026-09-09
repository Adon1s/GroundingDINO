# Session 9 renovation architecture cutover — independent decision packet

Date prepared: 2026-08-21  
Repository: `C:\Users\Steven\PycharmProjects\realtorvision-backend`  
Branch observed: `renovation_architecture_rework`  
HEAD observed: `6640ebb8856f6552f065e3ed51f0ca00eb6d9ad8`

## Purpose

This document supplies the recorded evidence and unresolved verification points for another session to make an independent GO/NO-GO cutover decision. It does not make or imply that decision.

The decision under review is whether to select the new renovation architecture in the worker by setting:

```text
KIND_ONTOLOGY_VERSION=observation_kind_v2
RENOVATION_ARCHITECTURE_MODE=new
```

## Current recorded state

- The current comparator report contains `release_ready: true`.
- Its aggregate gate object contains `passed: true` and zero failures.
- The report contains 967 manual-review items and zero unreviewed items.
- The working review file contains 967 entries. Every entry has decision `accepted`; none has a blank decision or explanation.
- The report covers 18 properties and 2 replicas, or 36 candidate artifacts.
- The report and reviews are bound to logical freeze SHA-256 `b2e69d274580d4bf40323431bb728dce94ca05026f01245ee4aedd66a95c5a1f`.
- The report was last written at `2026-08-21T17:54:38.4173048-05:00`.
- The review file was last written at `2026-08-21T15:58:06.1583940-05:00`.

The report content was inspected after Steven stated that he believed the comparator had finished. The terminal output and process exit code were not available to this session, so an observed `exit 0` is not part of this packet's evidence. The current report content itself is green.

## Scope and provenance

The original Session 9 handoff recorded the following:

- Session 9 code was uncommitted.
- The suite had 2,337 passing tests.
- Both canary replicas were complete and verified.
- All 36 candidate artifacts had complete v5 envelopes.
- Inputs were frozen under `artifacts_canary/renovation_session9_20260818/`.
- Mechanical gates for schema, publication, reconciliation, and accepted-work passed before manual review was completed.
- Manual review was the only failing gate at that time.

The 2,337-test result is inherited from the handoff; this review-assistance session did not rerun or independently verify it. This review-assistance work did not regenerate artifacts, call providers, or change application code. It did edit the designated working review JSON. The pristine blank review template was left unchanged.

The relevant report, review, digest, runbook, and supporting handoff files are currently untracked in the observed Git worktree. The branch also contains other pre-existing uncommitted/untracked work. This packet does not establish which uncommitted code state produced the frozen artifacts.

## Canary construction and aggregate results

- Properties: 18
- Replicas per property: 2
- Candidate artifacts: 36
- Run 1, v5 relative to v4 corpus total: `+11.9%` low and `+20.1%` high
- Run 2, v5 relative to v4 corpus total: `+12.6%` low and `+21.1%` high
- Session 8 replay anchor: `+7.7%` low and `+17.6%` high
- Replica 2 relative to replica 1 corpus total: `+0.6%` low and `+0.9%` high
- Individual-property replica swings: up to approximately `±50%`
- Recorded upstream condition churn: approximately 25% of conditions per property

The scope comparison is v4 versus v5 within run 1. The stability comparison is v5 run 1 versus v5 run 2.

## Comparator gates

The current report records all comparator gates as passing. The gate families described in the original handoff are:

- schema validity and complete envelopes;
- publication shape;
- reconciliation and arithmetic;
- preservation of accepted work;
- completion of manual review.

The final manual-review state is 967 reviewed and 0 unreviewed. A current read-only check of the JSON confirmed 967 non-empty decisions and explanations.

## Manual-review organization and recorded decisions

The 967 items were clustered into three tiers by `scripts/cluster_session9_reviews.py`.

### Tier 1 — 736 items

These were prefilled as machine-verified no-material-change cases. The stated classes were:

- identical dollars;
- integer rounding differences no greater than $2;
- catalog IDs absent from every v4 scope row in the 36-artifact corpus, corresponding to the previously accepted 48-item v5 billing class;
- package-key migration;
- package metadata-only differences.

### Tier 2 — 77 items

These were prefilled as policy-group drafts. The recorded groups were:

- 56 merge-dedup items where shared scope prices lower: corpus effect `-$13,616` low and `-$231,270` high;
- 16 single-item wider-high repricing items: corpus effect `-$7` low and `+$18,712` high;
- 5 v4-zero-now-priced items: corpus effect `+$1,751` low and `+$20,946` high;
- a fourth stricter-scope policy existed in the clustering script but matched zero items.

All 77 currently have `accepted` decisions and explanations in the working file. The conversation captured explicit decisions for the Tier 3 categories listed below, but it did not capture a separate explicit statement from Steven approving each Tier 2 policy group. The file nevertheless records them as accepted. An independent reviewer can treat that distinction as a governance/provenance question.

### Tier 3 — 154 items

The original category counts were:

- 8 merge-increase review items representing 3 merge groups;
- 47 package reviews;
- 27 dropped-by-Terra-disposition scope items;
- 28 other dropped scope items;
- 12 unit reattributions;
- 14 headline deltas greater than 15%;
- 18 run-to-run stability items, one per property.

Recorded review sequence:

- Steven explicitly accepted the 3 merge groups.
- Steven explicitly accepted the unit reattributions and their paired removals.
- Steven explicitly accepted all 27 disposition drops.
- Steven explicitly accepted the remaining stated clear exclusions.
- Steven explicitly accepted the 5 inspection drops.
- Those decisions covered 75 Tier 3 scope items.
- A subsequent session populated the remaining 79 Tier 3 items: 47 package reviews, 14 headline-delta reviews, and 18 stability reviews.

The current working file contains accepted decisions and non-empty explanations for all 154 Tier 3 items.

## Merge-review interpretation

The comparator scope view lists a merged v5 work item's full price under each constituent catalog ID. A three-item merge can therefore appear as three increases even when the merge group's combined v5 total is lower than the v4 sum. The digest evaluates merges at group level. The 8 Tier 3 `merge_increased` items represent 3 groups, not 8 independent group-level increases.

## Reattributions and paired removals

The recorded review treated a reattribution and the disappearance of the old unit assignment as one change. The roof cases were classified as property-level rather than `exterior_primary` because the roof is a whole-building system and is not scoped to a single exterior observation unit. The paired removal prevents the same work from remaining billed at the prior unit.

This paragraph records the rationale used in the accepted explanations; it is not an independent conclusion by this packet.

## Dropped scope

The 27 disposition drops include the v5 Terra disposition inline in the digest and review evidence. The other dropped-scope review covered exclusions, inspection-only cases, reattributions, and paired removals. The working file contains the item-specific rationale for every drop.

The relevant independent checks are whether each dropped item is intentionally excluded, inspection-only, relocated without loss, or rejected by recorded disposition, and whether accepted work is neither lost nor double counted.

## Package-review evidence

The package-review handoff records the following results for the 47 package items:

- 24 newly approved packages;
- 2 newly rejected packages that were not applied;
- 21 packages changed in place.

The recorded audit states:

- each of the 24 newly approved packages had supported drivers, coherent child work, and each child was covered once;
- the 2 rejected packages had effective `$0/$0`, absorbed no work, and left supported work standalone;
- the 21 changed packages were attributed to canonical migration, consolidation, Sol gating, Terra exclusion or inspection, retiering, or package-to-standalone movement, without recorded lost work;
- overlap ownership was checked so shared child work had one owning package;
- bathroom high floors were treated as visible system behavior; general price calibration was outside the migration review scope.

Recorded package-review hotspots:

- high floors: `redfin_80990371`, `redfin_125779232`, `redfin_10952874`, `redfin_11185681`;
- new full-kitchen packages: `redfin_10806500`, `redfin_11077450`, `redfin_11185681`;
- package-to-standalone movement: `redfin_11079485`, `redfin_166147710`, `redfin_126224899`, `redfin_125970550`, `redfin_11077450`;
- closet migrations: `redfin_126224899`, `redfin_80990371`.

The item-level evidence and explanations remain in the working review file and the detailed manual-review handoff.

## Headline-delta decomposition

For each of the 14 headline deltas, the recorded analysis decomposed total movement into package and standalone/non-package movement:

| Property | Total low/high | Package low/high | Standalone low/high |
|---|---:|---:|---:|
| `redfin_10803207` | `+5,637 / +9,564` | `+5,792 / +14,666` | `-155 / -5,102` |
| `redfin_10806500` | `+20,128 / +37,258` | `+21,473 / +59,568` | `-1,345 / -22,310` |
| `redfin_10952874` | `-7,834 / -9,307` | `-9,787 / -26,004` | `+1,953 / +16,697` |
| `redfin_11077450` | `+23,688 / +57,621` | `+22,873 / +50,831` | `+815 / +6,790` |
| `redfin_11079485` | `-8,508 / +9,880` | `-11,211 / -28,392` | `+2,703 / +38,272` |
| `redfin_11185681` | `+17,034 / +31,458` | `+14,400 / +31,554` | `+2,634 / -96` |
| `redfin_125779232` | `-2,207 / +20,197` | `-4,005 / +728` | `+1,798 / +19,469` |
| `redfin_126418713` | `-6,774 / -7,124` | `-9,893 / -35,912` | `+3,119 / +28,788` |
| `redfin_127468088` | `+7,046 / +45,415` | `+4,117 / +15,348` | `+2,929 / +30,067` |
| `redfin_166147710` | `+327 / +34,594` | `-4,206 / -8,412` | `+4,533 / +43,006` |
| `redfin_25809814` | `+22,022 / +35,196` | `+23,738 / +59,453` | `-1,716 / -24,257` |
| `redfin_80877597` | `-4,550 / -8,646` | `-5,987 / -17,049` | `+1,437 / +8,403` |
| `redfin_80925528` | `+4,466 / +24,513` | `+2,500 / +5,483` | `+1,966 / +19,030` |
| `redfin_80990371` | `+7,790 / +42,450` | `+2,618 / +9,347` | `+5,172 / +33,103` |

The recorded sums close exactly in each row. The item-level explanations attribute the component changes to reviewed v5-class additions, package changes or floor lifts, merge deduplication, and standalone movements.

## Replica stability evidence

For every property in both runs, the recorded invariant is:

```text
headline low/high = standalone low/high + packaged low/high
```

The detailed audit records zero failures of that invariant.

The largest recorded replica swing is `redfin_11000447`:

- run 1: `$39,121 / $119,247`;
- run 2: `$19,370 / $68,185`;
- delta: `-$19,751 / -$51,062`;
- standalone component: `+$24 / -$1,245`;
- package component: `-$19,775 / -$49,817`;
- detected conditions: 39 in run 1 and 30 in run 2;
- unique conditions: 17 run-1-only and 8 run-2-only.

The recorded evidence loss includes dated cabinets and kitchen finishes, bathroom finishes, damaged kitchen drywall, and flooring detections. In the package state, bathroom modernization disappears, bathroom and bedroom repairs are rejected, and kitchen modernization changes from full to partial. The detailed handoff contains all 18 stability rows.

The comparator's corpus-wide replica delta is small, while property-level swings are materially larger. Both facts are part of the decision record.

## Terra usage recorded in the report

- Daily ceiling: 2,500,000 tokens
- Canary budget-debited tokens: 1,836,273
- Listing total tokens: median 54,232.5; p90 80,612; p95 83,767.25; p99 85,865.95; worst 86,532
- Projected listings per day: median 46; p90 31; p95 29; p99 29; worst 28

These are reported operating-capacity measurements, not a cutover conclusion.

## Evidence limitations and questions for the independent reviewer

The following items are not resolved by the green `release_ready` field alone:

1. The comparator process exit code was not captured in this session, although the resulting report is green and has a later modification time.
2. The Tier 2 policy decisions are accepted in the JSON, but a separate explicit Steven signoff on the four-group walkthrough is not present in the conversation record supplied to this packet.
3. The 2,337-test result is reported by the original handoff and was not rerun here.
4. The canary/report/review materials and Session 9 code are uncommitted or untracked in the observed worktree. The provenance relationship between the current code state and the frozen artifacts should be established if required by release policy.
5. General price calibration, including the accepted visible bathroom high-floor behavior, was outside the migration review scope.
6. Corpus-wide replica movement is small, but individual properties show swings up to approximately ±50% associated with upstream detection churn.
7. The operational mode switch, worker restart, end-to-end production smoke, and rollback smoke have not been performed or verified in this review-assistance conversation.
8. The current hashes below identify the files inspected for this packet. Any subsequent edit changes the evidence set.

## File identities

| File | Bytes | SHA-256 |
|---|---:|---|
| `reports/renovation_architecture_session9_reviews_20260821.prefilled.json` | 320,741 | `1381fba28d368ff0a29e913f0f5336de7b4b710d7bcd2599b2f1598d847245f8` |
| `reports/renovation_architecture_session9_reviews_20260821.json` | 85,234 | `75c646113cb4c2b4babeeca6fe9111948bf09cb1cbef74cb85e6dba750c8e5d0` |
| `reports/renovation_architecture_session9_canary_20260821.json` | 3,285,732 | `37694749810b9215417710937871f75f4e8538f35c5a911d31903b3713fc161b` |
| `artifacts_canary/renovation_session9_20260818/input_freeze.json` | 170,735 | `f32f68125bd4cbb4ef4ccd71d9931382f95b4963ca5584631303ee55d6b4a973` |
| `reports/session9_review_digest_20260821.md` | 31,870 | `8e1f46d60d5dbdb9acc641041b291d7230df4538e198e0b4dbe0c1a4b85df4bb` |

The logical freeze SHA in the report and review file is not the same object as the raw file hash of `input_freeze.json`; both values are recorded above.

## Read-only verification commands

PowerShell summary of report and review completeness:

```powershell
$report = Get-Content -Raw -LiteralPath 'reports\renovation_architecture_session9_canary_20260821.json' | ConvertFrom-Json
$reviewFile = Get-Content -Raw -LiteralPath 'reports\renovation_architecture_session9_reviews_20260821.prefilled.json' | ConvertFrom-Json
$rows = @($reviewFile.reviews.PSObject.Properties | ForEach-Object { $_.Value })

[pscustomobject]@{
  ReleaseReady     = $report.release_ready
  GatesPassed      = $report.gates.passed
  GateFailures     = @($report.gates.failures).Count
  Properties       = $report.property_count
  Replicas         = $report.replicate_count
  ReviewItems      = @($report.review_items).Count
  Unreviewed       = @($report.unreviewed_items).Count
  ReviewRows       = $rows.Count
  BlankOrIncomplete = @($rows | Where-Object {
    [string]::IsNullOrWhiteSpace($_.decision) -or
    [string]::IsNullOrWhiteSpace($_.explanation)
  }).Count
  ReportFreezeSha  = $report.freeze_sha256
  ReviewFreezeSha  = $reviewFile.freeze_sha256
}
```

PowerShell file hashes:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath `
  'reports\renovation_architecture_session9_reviews_20260821.prefilled.json', `
  'reports\renovation_architecture_session9_reviews_20260821.json', `
  'reports\renovation_architecture_session9_canary_20260821.json', `
  'artifacts_canary\renovation_session9_20260818\input_freeze.json', `
  'reports\session9_review_digest_20260821.md'
```

The comparator command Steven reported running, expressed for Git Bash, is below. The original review-assistance constraint said not to rerun; an independent session should run it only if Steven authorizes a further rerun. Do not regenerate the review template because review IDs are bound to the freeze SHA.

```bash
./.venv/Scripts/python.exe tools/compare_renovation_architecture_cutover.py \
  --run artifacts_canary/renovation_session9_20260818/run_1/candidate \
  --run artifacts_canary/renovation_session9_20260818/run_2/candidate \
  --freeze artifacts_canary/renovation_session9_20260818/input_freeze.json \
  --reviews reports/renovation_architecture_session9_reviews_20260821.prefilled.json \
  --report reports/renovation_architecture_session9_canary_20260821.json
echo $?
```

## Independent decision checklist

The next session can independently record evidence for each item before making its decision:

- [ ] Confirm the branch, HEAD, and exact uncommitted code state associated with the frozen canary.
- [ ] Confirm the five file hashes against this packet or document any expected changes.
- [ ] Confirm report `release_ready`, aggregate gates, 967 review items, and zero unreviewed items.
- [ ] Confirm all 967 working-review entries have an allowed decision and a non-empty explanation.
- [ ] Determine whether Tier 2's file-recorded acceptance satisfies the required human-approval policy.
- [ ] Review the three merge-increase groups at group level, not by duplicated constituent rows.
- [ ] Sample or fully audit disposition drops, exclusions, inspection-only drops, and reattribution pairs.
- [ ] Audit package child ownership, rejected-package zero application, and package-to-standalone preservation.
- [ ] Confirm the 14 headline decompositions and all 18 stability reconciliation invariants.
- [ ] Decide whether the observed property-level replica variance is acceptable for cutover.
- [ ] Decide whether price calibration outside migration scope requires a separate gate.
- [ ] Establish the comparator exit code if release procedure requires direct process evidence.
- [ ] Establish the 2,337-test result against the exact candidate code state if release procedure requires it.
- [ ] Prepare operational smoke and rollback execution before changing the worker selector.

## Cutover and rollback procedure from the runbook

The runbook says not to select the new architecture unless the comparator exits 0 and reports `release_ready: true`.

If the independent decision is GO, the recorded operational sequence is:

1. Set persistent worker environment variables atomically:

   ```text
   KIND_ONTOLOGY_VERSION=observation_kind_v2
   RENOVATION_ARCHITECTURE_MODE=new
   ```

2. Restart the worker.
3. Run one end-to-end smoke and verify:
   - root `renovation_estimate_v5` is schema-valid and complete;
   - provenance reports architecture mode `new` and catalog `3.1`;
   - `renovation_estimate_v4` remains present and valid;
   - no private v5 artifact appears under `analysis_debug`;
   - publication, reconciliation, and arithmetic are clean.

Recorded rollback sequence:

1. Set `RENOVATION_ARCHITECTURE_MODE=current`.
2. Restart the worker.
3. Run an end-to-end smoke.
4. Verify unchanged v4 publication and no v5 root or private publication.

Recorded stop/rollback conditions include mixed or invalid artifacts, accepted-work loss, double counting, model mutation of condition/work truth, semantic publication failure, silent Terra-ceiling overrun, or repeated writer failure.

## Primary evidence files

- `reports/renovation_architecture_session9_canary_20260821.json` — current comparator report
- `reports/renovation_architecture_session9_reviews_20260821.prefilled.json` — completed working reviews
- `reports/renovation_architecture_session9_reviews_20260821.json` — pristine blank template; do not modify or regenerate
- `reports/session9_review_digest_20260821.md` — clustered review evidence
- `docs/HANDOFF_renovation_architecture_session9_manual_review_complete.md` — detailed final-79 review audit
- `docs/analysis/session8_floor_vs_sum_trace.md` — Session 8 merge floor-versus-sum trace
- `docs/RUNBOOK_renovation_architecture_session_6.md` — cutover, smoke, stop, and rollback procedure
- `artifacts_canary/renovation_session9_20260818/run_1/candidate/` — replica 1 candidate artifacts
- `artifacts_canary/renovation_session9_20260818/run_2/candidate/` — replica 2 candidate artifacts
- `artifacts_canary/renovation_session9_20260818/input_freeze.json` — frozen inputs

## Decision

Intentionally omitted. The next session should issue its own GO/NO-GO decision and identify the evidence and release-policy interpretation supporting it.
