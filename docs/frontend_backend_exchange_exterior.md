# Frontend ↔ backend exchange: how packages and issues reach the UI

**Reference document. Facts only — no design proposals, no work items.** Written 2026-07-31
while shipping the `exterior_repair` package family, so a later, separately-scoped task can
design against the real contract instead of rediscovering it.

## Which checkout is authoritative

Both exist locally and the relevant files are **identical**:

| Checkout | Branch | Last commit |
|---|---|---|
| `C:\Users\Steven\IntelliJProjects\renointel-prod` | `master` | `bbcd0b0` 2026-07-30 |
| `C:\Users\Steven\IntelliJProjects\renointel-dev` | `deal_margin_verdict` | `f96c822` 2026-07-30 |

`lib/property/photoInspection.ts` and `lib/topPicks/scopeBand.ts` are byte-identical across the
two. Artifacts are written under **`renointel-prod\artifacts\<property_key>\<run_id>\`**, which
is what the backend replay tooling reads. Line references below were verified against
`renointel-prod`; confirm the checkout before citing paths in new work.

## What the backend writes

`photo_intel.json` (slimmed) and `photo_intel_debug.json` (full) per run —
[artifact_writers.py:1011-1017](../tools/artifact_writers.py#L1011). The destination is
`job.artifacts_dir`, supplied by the caller via `--artifacts-root`; the write path hardcodes
nothing. Several module-level defaults still point at the **old** `realtorvision` path
(`.env` `ARTIFACTS_ROOT`, `model_comparison.py:92`, `catalog_auditor.py:64`, `benchmark.py:37`,
`audit_runner.py:15`) and are stale.

Fields the frontend consumes:

| Field | Consumed by |
|---|---|
| `photos[<key>].issues.final` | per-photo issue list, and the global issue index |
| `photos[<key>].scene.{id,group}` | room label on the inspect view |
| `renovation_estimate_v4.packages` | package cards, costs, scope band, deal margin |
| `renovation_estimate_v4.groups` | cost-by-trade, itemized estimate table |
| `renovation_estimate_v4.project_scope_breakdown` | project-scope card, mobile noticed-items |
| `renovation_estimate_v4.rehab_evidence_projection_v1` | headline range |
| `ui_priorities_v1` | priority lanes, confirmed-package groupings |
| `scoring`, `renovation_estimate` | score surfaces, v3 fallback |

## The API projection

[app/api/analysis/route.ts](../../../Users/Steven/IntelliJProjects/renointel-prod/app/api/analysis/route.ts)

- **No filtering by package room, package_type, or package_category anywhere in the route.**
  `renovation_estimate_v4.packages` passes through verbatim (`:366-370` → `normalizePackage`).
  All package gating is client-side.
- Audience gate at `:679-689`. `?audience=audit` requires an authenticated session; guests always
  get the public projection.
- **Public audience drops:** issues whose catalog `trade_bucket` is quarantined (`:301-306`; the
  audit lane keeps them marked `defaultHidden`, `:241-244`), and sanitized `_compat` pass-2e
  debug lists (`:310-312`).
- **Both audiences drop** when `readProjectionMeta(...).derivedSurfacesServable` is false
  (`:351`, `:382-389`): `scoring`, `renovation_estimate`, `renovation_estimate_v4`,
  `ui_priorities_v1`, `renovationNeeds`, and the DB-cached estimate. Fail-closed.
- `project_scope_breakdown` gets quarantined trade **tags** stripped (`:359-365`) — labels only,
  dollars untouched.

## Client-side gating — this is what decides visibility

[lib/property/photoInspection.ts](../../../Users/Steven/IntelliJProjects/renointel-prod/lib/property/photoInspection.ts)

1. **`isApprovedPackage` (`:136-141`)** — three negative predicates only, no allowlist:
   rejects `verification_status === 'rejected'`, `package_level === 'property'`,
   `room === 'whole_home'`. Any other room or package_type passes automatically.
2. **Photo attribution (`:269-288`)** — a package reaches a photo via
   `evidence_items[].photo_keys`, `evidence_items[].issue_refs[].photo_key`, or issue ids
   (`confirmed_issue_ids` / `supporting_issue_ids` minus `rejected_issue_ids`) resolved through
   the global index built from `photos[].issues.final` (`:241-246`). A package with no
   photo-resolvable evidence renders nowhere while still counting in `totalPackages` (`:348`).
3. **Non-package issues are hidden by default (`:317-321`)** —
   `otherRaw = includeUnverified ? allIssues.filter(...) : []`, and
   `includeUnverified = searchParams.get('unverified') === '1'`
   (`components/property/desktop/inspect/DesktopPhotoInspection.tsx:81`). Even with the flag,
   an issue must have a non-null `catalogName` and `defaultHidden === false`.
4. With no package and no flag, `pkgCount === 0 && otherCount === 0` renders
   **"No issues detected in this photo"**
   (`DesktopInspectionFindings.tsx:243`, `MobilePhotoInspection.tsx:246`).

`catalogName` / `defaultHidden` are **enrichment**, not artifact fields — added by
`lib/analysis/enrichIssue.ts:55,64` and `app/api/analysis/route.ts:232,241` from the catalog.

## Vocabularies the client derives rather than reads

- **`package_label` does not exist** in the artifact or the frontend. Card labels are composed:
  `ROOM_LABEL[room] + ' ' + CATEGORY_LABEL[category]` → e.g. `room: "exterior"` +
  `package_category: "repair"` renders **"Exterior repairs"** with a "Repair" chip
  (`photoInspection.ts:95-106`, attached at `:183-186`). Fallback chain ends at
  `pkg.name || pkg.package_type` truncated to 36 chars.
- `PackageRoom` and `PACKAGE_ROOMS` (`lib/types/renovationEstimate.ts:24,41`) already include
  `'exterior'`. `package_type` is a free-form string (`:184`, normalizer `:622`) — **no zod
  anywhere in the package path**. Unknown `room`/`package_category` values are normalized to
  `null` (`:436-442`), a silent downgrade rather than a rejection.
- `normalizeRenovationEstimate` (`:791-796`) drops the whole estimate unless `version` is v3/v4
  **and** `groups` is an array. `normalizeGroup` (`:562-576`) accepts any `group` string — no
  whitelist, so estimate groups render without frontend changes.
- Room labels/icons are duplicated across four maps (`photoInspection.ts:74-81`,
  `topPicks/packageLabels.ts:11-18`, `property/desktopInvestment.ts:100-107`,
  `saved/SavedPropertyCard.tsx:29-36`) plus a `Trees` icon for exterior
  (`topPicks/EvidenceTopPickCard.tsx:34`). All are `Record<PackageRoom, …>`, so a *new*
  `PackageRoom` value would break the build; existing values do not.

## Behavioral couplings a new package room touches

- **Scope band → deal margin.** `deriveRehabScopeBand`
  ([lib/topPicks/scopeBand.ts:159-173](../../../Users/Steven/IntelliJProjects/renointel-prod/lib/topPicks/scopeBand.ts#L159)):
  `heavy` when `packages.length >= 6 || (hasKitchen && hasBath && hasExterior)`; `moderate` when
  `packages.length >= 3 || (hasKitchen && (hasBath || hasExterior))`; else `light`. A
  line-item-sourced band is capped at moderate (`deriveScopeBandFromExtraction`).
  `computeDealMargin` picks the sold-comp target percentile off the band
  (`lib/property/dealMargin.ts:105`), so the band moves dollars.

  **Measured 2026-07-31** over 1,215 artifacts in `renointel-prod\artifacts`, estimating which
  properties would gain an `exterior_repair` package (has ≥1 of the four exterior drivers in
  `issues_flat`, no exterior package today):

  | | count |
  |---|---|
  | artifacts scanned | 1,215 |
  | would likely gain an exterior package | 599 (49%) |
  | of those, scope band unchanged | 156 |
  | **band flips** | **443 (36% of corpus)** |
  | — `None → light` (no packages today) | 277 |
  | — `light → moderate` | 84 |
  | — `moderate → heavy` | 82 |

  Upper bound: a driver issue does not guarantee a package survives the emit gate or Pass 2f,
  and the band is derived from `extractTopPackages(...).allMerged` rather than raw `packages[]`.
  131 Alex Ln itself did not flip — it was already `heavy` on package count (8 ≥ 6).

- **Opportunity score** (`lib/topPicks/ranking.ts:588-630`) rewards only kitchen and bathroom
  modernization packages. Exterior packages contribute to `allPackages` (count-driven band,
  `deriveSignalConfidence` `:634`) but not to the score.
- **Featured signal summary** (`topPicks/packageLabels.ts:75-119`) fills its two primary slots
  with kitchen then bathroom first; other rooms only land there when those are absent.
- **Audit feed routes** exist per-room only — `app/api/audits/{kitchen,bathroom,bedroom,living}-feed/`.
  There is no exterior feed.
- **Surrogate disambiguation** (`property/desktopInvestment.ts:205-225`) recognizes
  bathroom/bedroom/kitchen/living/garage; the comment at `:208-209` explicitly excludes
  `exterior_front`. Exterior surrogates get no "Exterior 2" style label — not an error.

## Two distinct "exterior" axes

Do not conflate them:

- **Package room** — `pkg.room === 'exterior'`, from `renovation_estimate_v4`. Drives package
  cards, scope band, estimate grouping.
- **Scene group** — `photo.scene.group === 'exterior'`, from the scene classifier. Drives issue
  filtering (`lib/types/issues.ts:14`, `components/issues/FiltersRail.tsx:439-448`) and is
  down-weighted 0.75× in `lib/opportunities/opportunityScoring.ts:217`.

There is also an `exterior_score` subscore (`lib/types/renovationEstimate.ts:99`, surfaced as
"Siding, Landscaping, Windows" at `app/property/[id]/page.tsx:303`) on a third axis again.
