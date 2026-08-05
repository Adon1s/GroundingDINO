# HANDOFF: Wire property metadata (CSV → DB → analyzer) so the estimate's metadata layer comes back to life

## Problem

Every production artifact currently ships `property_metadata: {}` (verified across all 17 runs from
2026-07-01). Because of that, the entire metadata-dependent layer of the v4 renovation estimate is
silently inert:

- `tools/cost_factors.py` — market factor and size factor always 1.0 (a Birmingham fixer and a Bay
  Area house price identically),
- `tools/estimate_units.py` — bedroom/bathroom metadata caps and multi-kitchen evidence never apply,
- `tools/estimate_sanity.py` — every price-relative and sqft-relative sanity flag can never fire.

All that code works and is tested; it just never receives data.

## Root cause (verified, do not re-derive)

The v4 estimator gets metadata from `artifact_writers._resolve_property_metadata(job)`
([tools/artifact_writers.py:360](../tools/artifact_writers.py)), which merges two sources,
job-provided winning over scrape:

1. `getattr(job, "property_metadata", None)` — **no producer exists**. `PropertyAnalysisJob`
   (dataclass at [tools/analyzer_cli.py:66](../tools/analyzer_cli.py), also constructed at
   [tools/analyzer_server.py:520](../tools/analyzer_server.py)) has no such field and neither
   intake path accepts one.
2. Scrape fallback (`_load_scrape_metadata_for_job`) — finds the scrape.json fine, but current
   batch scrape.json files contain only
   `url, scraped_at, source, is_new_construction, photo_count, address, photos, downloaded_images`.
   **The scraper no longer captures listing facts — intentionally.** Steven moved the source of
   truth to the CSV funnel; the scraper is only a photo fetcher now. Do NOT "fix" the scraper.

The intended source of truth is the frontend DB. Confirmed populated: the Prisma `Property` model
(frontend repo `C:\Users\Steven\IntelliJProjects\realtorvision`, `prisma/schema.prisma`, SQLite at
`prisma/dev.db`) has `price, beds, baths, sqft, yearBuilt, lotSize, propertyType, status,
daysOnMarket, hoaFees, description` with `sourceType='CSV_FUNNEL'` — **544/553 rows have
price+beds+sqft**, including all July analysis properties. Example:
`redfin_80860084 → price=119900, beds=3, baths=1.5, sqft=1710, yearBuilt=1970,
propertyType='Single Family Residential'`.

So the CSV → DB half already works. The missing piece is DB → analyzer request → job →
`_resolve_property_metadata`. That last function needs zero changes — it already prefers job
metadata over scrape.

## The fix

Whole thing is ~60–80 lines across both repos. Keep it that slim.

### 1. Frontend: pass the Property row with the analyze request

File: `lib/services/analysisQueueService.ts` (frontend repo).

The full Prisma `Property` row is **already in scope** at the dispatch site — `job.property` is
validated at line ~331 and both analyzer paths are called at lines ~395/402. Add one small mapper
(the backend allowlist is snake_case/short names — see table below) and pass its output through
both paths:

- **Persistent path** (`runPersistentAnalyzer`, line ~695): add `propertyMetadata` to the request
  object built at line ~730, and to the `AnalyzerJobRequest` interface in
  `lib/services/persistentPythonWorker.ts` (line ~9).
- **Spawn path** (`runPythonAnalyzer`, line ~751): append
  `'--property-metadata-json', JSON.stringify(metadata)` to `analyzerArgs` (line ~762). Node
  `spawn` with an args array handles quoting; the payload is tiny.

Field mapping (TS `Property` → payload key). Only send non-null values; the backend allowlist
`_PROPERTY_METADATA_KEYS` ([tools/artifact_writers.py:250](../tools/artifact_writers.py)) drops
unknown keys, and `price → list_price`, `beds → bedrooms`, etc. aliases are added backend-side
(lines 284–293), so send exactly these:

| TS field       | payload key       |
|----------------|-------------------|
| `price`        | `price`           |
| `beds`         | `beds`            |
| `baths`        | `baths`           |
| `sqft`         | `sqft`            |
| `yearBuilt`    | `year_built`      |
| `lotSize`      | `lot_size`        |
| `propertyType` | `property_type`   |
| `status`       | `status`          |
| `daysOnMarket` | `days_on_market`  |
| `hoaFees`      | `hoa`             |
| `description`  | `description`     |

Include `description` — `estimate_units._metadata_text` scans it for multi-kitchen terms
("in-law", "second kitchen", "kitchenette"). Also set `metadata_source: "csv_funnel_db"` so
artifacts record provenance.

Note: `validateListingMetadata` (already called at line ~335) only gates address/city/state/zip —
it does NOT guarantee beds/baths/sqft. That's fine: send what exists, the backend treats every
missing key as neutral. Do not add hard validation for these fields (9/553 rows lack some).

### 2. Backend: accept it on both intake paths

- `tools/analyzer_server.py::_process_job` (line ~307): read
  `request.get("propertyMetadata")` next to the other request reads (~line 327), sanity-check
  `isinstance(dict)`, and pass it into the `PropertyAnalysisJob(...)` construction at line ~520.
  The `modelOverrides` handling at line ~349 is the pattern to copy for defensive parsing.
- `tools/analyzer_cli.py`: add field `property_metadata: Optional[Dict[str, Any]] = None` to the
  `PropertyAnalysisJob` dataclass (line 66) — this is the same class the server imports, so one
  edit covers both. Add argparse flag `--property-metadata-json` (JSON string, default None,
  `json.loads` with a try/except that logs and continues on garbage), and set it on the job the
  CLI builds (grep `PropertyAnalysisJob(` in the file).

Nothing else backend-side: `_resolve_property_metadata` already reads `job.property_metadata`,
merges scrape underneath it, and `write_photo_intel` already persists it to the artifact
(line ~664–696) and threads it into `compute_renovation_estimate_v4`.

### 3. Backend: one-line bug that would keep a flag dead even after wiring

`estimate_sanity._is_single_family` ([tools/estimate_sanity.py:218](../tools/estimate_sanity.py))
matches only `{"single_family", "sfh"}`. The CSV value is `"Single Family Residential"` →
normalizes to `single_family_residential` → no match → the `multiple_billable_kitchens_single_family`
flag stays dead. Change to a prefix match (`normalized.startswith("single_family") or normalized == "sfh"`).

### 4. Optional, phase 2 (skip if time-boxed)

- `area_price_per_sqft`: `cost_factors._resolve_ppsf` prefers it over the subject's own ppsf
  (a distressed fixer lists below its area's ppsf exactly when it needs work — rationale documented
  in [tools/cost_factors.py:32](../tools/cost_factors.py)). The frontend has zip medians in
  `lib/topPicks/zipMedians.ts`; if a median for the property's `zipCode` is available at dispatch
  time, add `area_price_per_sqft` to the payload. Backend already handles it.
- `tools/backfill_reno_v4.py` recovers metadata only from artifact → scrape.json, so existing July
  artifacts can't be healed by backfill alone. Either add an optional `--metadata-json` there, or
  just re-analyze the handful of properties that matter. Re-analysis is probably fine.

## Verification

1. Unit tests (backend): `.venv\Scripts\python.exe -m pytest tests/test_artifact_writers.py
   tests/test_cost_factors.py tests/test_estimate_sanity.py -q` — bare `python`/`py` do not resolve
   to the project venv. Note: 7 pre-existing failures exist on `reno_estimate_refactor` in other
   files; scope pytest to touched files. `tests/` is untracked on this branch — never `git stash`
   test paths.
2. E2E without the frontend: run `tools.analyzer_cli` directly on an already-downloaded property
   (images in frontend repo `public/images/properties/redfin_80860084/`) with
   `--property-metadata-json '{"price":119900,"beds":3,"baths":1.5,"sqft":1710,"year_built":1970,"property_type":"Single Family Residential"}'`,
   or cheaper: unit-test `_resolve_property_metadata` with a job carrying the field.
3. Expected concrete outcomes for redfin_80860084 (price 119,900, sqft 1,710, last run's
   package-adjusted high was 142,167):
   - artifact `property_metadata` non-empty, `metadata_source: "csv_funnel_db"`;
   - `cost_adjustment.factor ≈ 0.74` (market (70.1/230)^0.4 ≈ 0.62 → clamped to 0.75; size
     (1710/1800)^0.3 ≈ 0.98) — estimates drop ~26%;
   - sanity flags finally fire: scaled package-adjusted high ≈ 105k vs 119.9k list →
     `package_adjusted_high_gt_50pct_price` (0.88 > 0.5) and
     `worst_case_high_gt_80pct_price` (0.88 > 0.8);
   - bedroom units ≤ 3, bathroom cap ceil(1.5) = 2.
4. Restart the persistent worker after backend edits (it's a long-running process; the queue
   respawns it when nulled, but a stale worker will run old code).

## Compatibility

New TS + old backend: extra request field ignored (`request.get`). Old TS + new backend: falls back
to scrape discovery → `{}` exactly as today. No migration needed.

## Style constraints (Steven's)

Slim, data-driven, no hardcoded special cases. One mapping function on the TS side, one optional
dataclass field + one argparse flag + one request read on the Python side, one prefix-match fix in
estimate_sanity. Resist the urge to add a metadata validation layer, a config file, or scraper
changes.
