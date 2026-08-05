# Handoff: Build the bathroom audit page (frontend)

This is a self-contained brief for a follow-up session. The backend bathroom-parity work has landed in `realtorvision-backend` on branch `reno_estimate_refactor`. You are picking up the **frontend** half in the separate Next.js app at `C:\Users\Steven\IntelliJProjects\realtorvision`. No backend changes are in scope.

## What the backend now produces

Bathroom packages emit through the same v4 pipeline as kitchen, with these fields populated on each `RenoPackage`:

- `package_type ∈ {"bathroom_modernization", "bathroom_repair", "bathroom_turnover"}` (plus the existing kitchen and `interior_paint_flooring_refresh` whole-home variants)
- `room: "bathroom"`
- `package_category ∈ {"modernization", "repair", "turnover"}`
- `pricing_tier ∈ {"bathroom_full_rehab", "bathroom_partial_rehab", "bathroom_refresh", "bathroom_repair_light", "bathroom_repair_heavy", "bathroom_turnover_light", "bathroom_turnover_std", "bathroom_minor_repair"}`
- `verification_status` set by Pass 2f (which now uses a bathroom-specific prompt — no kitchen vocabulary)
- Bathroom turnover packages get `verification_status: "confirmed_by_rule"` (deterministic; no VLM call), feeding `aggregate_whole_home_turnover` which produces an `interior_paint_flooring_refresh__whole_home` property-level package when both kitchen + bathroom turnover signals exist.

## Generate live data to test against

Pick a bathroom-heavy test property in `C:\Users\Steven\IntelliJProjects\realtorvision\artifacts\<property_key>\` and run the audit runner from the backend repo:

```
python -m tools.audit_runner \
  --artifacts-root C:\Users\Steven\IntelliJProjects\realtorvision\artifacts \
  --scene-group bathroom \
  --property <property_key>
```

This writes a new artifact run dir with bathroom-only photos re-analyzed through the new pipeline. Use it as the source for `/audits/bathroom` to verify your UI renders correctly.

## Required new files (frontend repo)

### 1. `app\api\audits\bathroom-feed\route.ts`

Exact mirror of `app\api\audits\kitchen-feed\route.ts` (256 lines). Copy verbatim, then identifier swap:

- `KitchenPhotoEntry` → `BathroomPhotoEntry`
- `Pass2fTallyKitchen` → `Pass2fTallyBathroom`
- `KitchenFeedListing` → `BathroomFeedListing`
- `kitchenPhotos` / `kitchenPhotoCount` → `bathroomPhotos` / `bathroomPhotoCount`
- `kitchenGroup` / `findKitchenGroup` → `bathroomGroup` / `findBathroomGroup`
- `isKitchenPackage` → `isBathroomPackage`
- `pass2fTallyKitchen` → `pass2fTallyBathroom`
- Header comment (line 1) — swap filename

Specific string swaps:
- Line 73-76 `isKitchenPackage`: `t === 'kitchen' || t.startsWith('kitchen_')` → `t === 'bathroom' || t.startsWith('bathroom_')`
- Line 80 `findKitchenGroup`: `g.group.toLowerCase() === 'kitchen'` → `'bathroom'`
- Line 194 scene filter: `sceneGroup.toLowerCase() !== 'kitchen'` → `'bathroom'`

Everything else (artifact loading, prisma query, JOB_ID_RE, image resolution, audit_meta skipping, v4 normalization) stays identical.

### 2. `app\audits\bathroom\page.tsx`

Exact mirror of `app\audits\kitchen\page.tsx` (~1059 lines). Same identifier swap as the feed route, plus copy/URL swaps:

- Heading (line 907): `"Kitchen review"` → `"Bathroom review"`
- Subheading (line 909-910): `"Kitchen photos, issues, and Pass 2f packages..."` → `"Bathroom photos, issues, and Pass 2f packages..."`
- Empty state (line 744): `"No kitchen packages."` → `"No bathroom packages."`
- Empty listings (line 1012): both `"kitchen"` references → `"bathroom"`
- Fetch URL: `/api/audits/kitchen-feed` → `/api/audits/bathroom-feed`
- Room-aware filter (line 245): `pkg.room !== 'kitchen'` → `pkg.room !== 'bathroom'`

`CATEGORY_BADGE` (line 136) is room-agnostic — keep verbatim; bathroom packages reuse the same modernization/repair/turnover badges. The `confirmedByCategory` partitioning (lines 641-643) is also room-agnostic.

### 3. Add a navigation entry

Grep the frontend for the existing link to `/audits/kitchen` — likely in `app/audits/page.tsx` (if it exists as an audit hub), plus any sidebar/header component (`app/components/*`, `app/layout.tsx`). Add a sibling `/audits/bathroom` link labeled "Bathroom review" using the same component style.

## Out of scope

- **Do not refactor** the kitchen page or bathroom page into a shared `[room]/page.tsx` route. Keep them duplicated for now — they share ~95% of code but evolving them independently lets you iterate room-specific copy and category tweaks without coupling. Re-evaluate factoring after both pages settle in production.
- **Do not change** `IssueReviewCard`, `CATEGORY_BADGE`, `confirmedByCategory`, or any v4 types — they are room-agnostic and bathroom is just another consumer.
- **Do not change** anything in the backend repo.

## QC checklist before declaring done

1. Run the audit-runner command above for a bathroom-heavy test property. Confirm the produced `photo_intel.json` contains:
   - `renovation_estimate_v4.package_candidates` with at least one entry where `package_type` starts with `bathroom_`.
   - For modernization/repair candidates: `verification_status ∈ {confirmed, rejected, uncertain}`.
   - For turnover candidates: `verification_status === "confirmed_by_rule"`.
   - `pass_2f_trace.attempted_count > 0` (Pass 2f actually ran, with the bathroom-specific prompt).

2. Boot the frontend (`npm run dev` in `C:\Users\Steven\IntelliJProjects\realtorvision`).

3. Open `/audits/bathroom`. Confirm:
   - Listings render with the same layout as `/audits/kitchen` — header, tally chips, package subsections by category (Modernization / Repair / Turnover), candidates-for-review subsection, line items table.
   - Bathroom-scene photos render through `IssueReviewCard`.
   - Pass 2f tally chip shows plausible confirmed/rejected/uncertain counts.
   - For a property with both kitchen and bathroom turnover signals, the whole-home turnover (`interior_paint_flooring_refresh__whole_home`) is filtered OUT of the bathroom listing (it's property-level, belongs on the property page, not per-room) — this is handled by the existing `isPropertyWideAggregate` filter on the kitchen page; mirror it.

4. **Hallucination filter sanity** — pick a property where Qwen fabricated a bathroom finding (visible as a rejected package in `pass_2f` trace). On `/audits/bathroom`, that fabrication must NOT appear in `packagesConfirmed`. It MAY appear in `packageCandidates` with `verification_status: "rejected"` — this mirrors kitchen behavior and is the user's whole point ("hallucinations don't make it into packages").

5. The "Catalog only" filter toggle should work for bathroom photos the same way it does for kitchen — only show issues that matched a catalog item.

## Pitfalls to watch for

- The `audit_meta` skip in the feed route (line 173-174 of kitchen-feed) is essential — without it, the page would surface the audit-only run alongside the main run. Preserve it verbatim in `bathroom-feed`.
- `enrichIssue` and `loadEnrichmentMaps` are shared utilities — do not duplicate them.
- The frontend has its own copy of `renovationEstimate` types (`@/lib/types/renovationEstimate`). If you find `bathroom_modernization` / `bathroom_repair` / `bathroom_turnover` strings or `bathroom_*` pricing tiers are not present in the existing type unions, you'll need to widen the union. Check the file before assuming it's already broad. Search for the existing `kitchen_modernization` string to find the right place.
- `getInvalidationLabel` and `isInvalidatedLineItem` are room-agnostic — reuse, do not duplicate.

## Reference reading

- `C:\Users\Steven\IntelliJProjects\realtorvision\app\audits\kitchen\page.tsx` — golden reference; 1059 lines.
- `C:\Users\Steven\IntelliJProjects\realtorvision\app\api\audits\kitchen-feed\route.ts` — golden reference; 256 lines.
- Backend types live in `C:\Users\Steven\PycharmProjects\realtorvision-backend\tools\rehab_packages.py` (PACKAGE_TYPE_* constants, lines 106-128).
- Backend Pass 2f prompts (informational only — not consumed by the frontend): `tools\scene_classifier_passes.py` `PASS_2F_ROOM_PROMPTS`.
