# Renovation Package & Project Scope — Frontend Contract

The backend now emits an explicit package taxonomy and a project-scope breakdown. This doc describes the new JSON fields in the `renovation_estimate_v4` payload and the migration path for frontend consumers in `lib/types/renovationEstimate.ts`, `lib/topPicks/notificationContent.ts`, and `lib/topPicks/ranking.ts`.

Scope of this round: **kitchen-first**. Bathroom / living-room items will follow once the kitchen flow is proven; the schema below already handles them.

---

## 1. Per-package fields

Every entry in `renovation_estimate_v4.packages[]` (and `package_candidates[]`) now carries the following four-axis taxonomy plus a strength signal and a confidence score.

| Field | Type | Values | Notes |
|---|---|---|---|
| `package_category` | string enum | `"modernization" \| "repair" \| "turnover" \| "inspection_risk"` | High-level UI / economic meaning. The single field that should drive Top Picks bucketing on the frontend (replaces the `REPAIR_PREFIXES` regex heuristic). |
| `room` | string enum or null | `"kitchen" \| "bathroom" \| "bedroom" \| "living" \| "exterior" \| "whole_home"` | Where the package applies. `whole_home` is reserved for property-wide aggregates. |
| `package_level` | string enum | `"property" \| "room" \| "component" \| "system"` | **New structural axis.** Most kitchen packages are `"room"`. The whole-home turnover aggregate is `"property"`. Future per-component packages (e.g. `cabinet_refacing`) will be `"component"`. |
| `package_type` | string | controlled-ish family id, e.g. `"kitchen_modernization"`, `"kitchen_repair"`, `"kitchen_turnover"`, `"interior_paint_flooring_refresh"` | Exact package identity. The set of valid values is `VALID_PACKAGE_TYPES` in `tools/rehab_packages.py`. |
| `package_strength` | string enum | `"strong" \| "moderate"` | **Canonical strength signal.** Frontend `deriveConfidencePosture` can collapse to a passthrough on this field. `"weak"` is a real internal value but is never emitted on a package — see §5. |
| `confidence_score` | float | `0.0`–`1.0` | Prior of 0.30 (strong) or 0.15 (moderate); boosted by Pass 2f confirmations, penalised by rejections. Use for finer Top Picks ranking. |

### Renamed field

| Old | New | Notes |
|---|---|---|
| `package_level` (with values `"refresh" \| "partial_rehab" \| "full_rehab"`) | `pricing_tier` (same values, plus `"repair_light"`, `"repair_heavy"`, `"turnover_light"`, `"turnover_std"`, `"property_turnover_aggregate"`) | The original `package_level` was a pricing depth tier, not a structural level — renamed to make room for the new structural `package_level`. Old artifacts on disk still have the old name; consumers can fall back via `pkg.pricing_tier ?? pkg.package_level`. |

### Verification status

`verification_status` gained one new value:

| Value | Meaning |
|---|---|
| `"confirmed"` | A VLM (Pass 2f) reviewed the photos and confirmed the package. |
| `"confirmed_by_rule"` | **New.** Deterministic bundling rule fired and is sufficient on its own (no VLM review). Used for turnover packages and the whole-home turnover aggregate. |
| `"rejected"` | VLM rejected the package. |
| `"uncertain"` | VLM couldn't decide, or no images available. |
| `"not_run"` | Verification hasn't run yet. |

Both `"confirmed"` and `"confirmed_by_rule"` should be treated as **active** for ranking / display purposes. The backend exposes this as `ACTIVE_PACKAGE_STATUSES` in `tools/rehab_packages.py`. Frontend equivalent:

```ts
const ACTIVE_STATUSES = new Set(["confirmed", "confirmed_by_rule"]);
```

UI hint: if you want to distinguish them visually (e.g. a small icon difference), `"confirmed"` = "AI-verified" and `"confirmed_by_rule"` = "Rule-based" both work. They are equally trustworthy for cost rollups.

---

## 2. Top-level `project_scope_breakdown`

New top-level array on `renovation_estimate_v4`. Matches the existing (but unused) `EstimatesProjectScopeBreakdown` type in `lib/types/issues.ts:154`:

```json
"project_scope_breakdown": [
  {
    "scope_id": "interior_generalist",
    "scope_name": "Interior Generalist",
    "cost_low": 12000,
    "cost_high": 24000,
    "item_count": 8,
    "trade_buckets": ["flooring", "paint_drywall", "trim_doors_windows"],
    "contributing_package_ids": ["kitchen_modernization__unit_kitchen_1"]
  }
]
```

Source: aggregates each `renovation_estimate_v4.groups[].line_items[]` by its `trade_bucket` (via the `tools/project_scopes.py` mapping), plus the trade buckets and package ids of confirmed packages. Costs are **pre-cap** line-item sums: for `max_only`/`group_cap` groups the breakdown may total above the capped `final_rehab` headline — it is a composition view of where work falls, not a reconciled total. Package premium above absorbed line items is intentionally not added here (it would double-count the absorbed base cost). The frontend can now finally render the project-scope view.

---

## 3. UI priorities lanes

`ui_priorities_v1` gains one additive field and re-sources its legacy lane so the name stays semantically accurate:

```jsonc
{
  "version": "ui_priorities_v1",
  "verified_estimate_drivers": [...],
  "high_concern_issues": [...],
  // Legacy lane — same shape as before but now strictly modernization-only,
  // even when repair/turnover packages exist alongside.
  "confirmed_modernization_packages": [...],
  "marketability_signals": [...],
  "audit_only_suppressed_or_unverified": [...],
  // NEW: categorized map keyed by package_category.
  "confirmed_packages_by_category": {
    "modernization": [...],
    "repair": [...],
    "turnover": [...]
  }
}
```

Migration recipe for the frontend:
1. Continue reading `confirmed_modernization_packages` until convenient — values are unchanged.
2. New consumers (Top Picks, scope breakdown UI) should read from `confirmed_packages_by_category`.
3. Once all consumers have switched, the legacy lane can be removed from the backend.

---

## 4. Whole-home turnover aggregate

When two or more rooms produce a turnover package, the backend appends a single aggregate package with:

```json
{
  "package_id": "interior_paint_flooring_refresh__whole_home",
  "package_type": "interior_paint_flooring_refresh",
  "package_category": "turnover",
  "room": "whole_home",
  "package_level": "property",
  "package_strength": "<strongest of constituents>",
  "confidence_score": "<max of constituents>",
  "verification_status": "confirmed_by_rule",
  "cost_low": "<sum of per-room costs>",
  "cost_high": "<sum of per-room costs>",
  "contributing_package_ids": ["kitchen_turnover__unit_kitchen_1", "bathroom_turnover__unit_bathroom_1"]
}
```

The per-room turnover packages stay too — the aggregate is purely additive for a property-wide summary. Threshold: aggregate only emits when ≥2 distinct rooms contribute (avoids a noisy "whole-home turnover" from a single kitchen).

---

## 5. Same-unit package subsumption

When one unit produces overlapping package bundles, the backend drops the subsumed ones from `packages[]` after verification:

- A modernization package at `partial_rehab`/`full_rehab` tier subsumes the unit's repair package(s).
- A modernization package at any tier (including `refresh`) suppresses the unit's turnover package(s).
- A `repair_heavy` package alongside a `turnover_std` package downgrades the turnover to its `*_turnover_light` profile (flooring is already covered); the turnover stays active at the lighter tier.

Dropped packages remain in `package_candidates[]` with `audit_only: true`, their original `verification_status`, and a pointer to the winner: `subsumed_by_package_id` (repair) or `suppressed_by_package_id` (turnover). The winning modernization package carries the inverse lists `subsumed_package_ids` / `suppressed_package_ids`. The full decision trail is in the top-level `package_subsumption_audit`. Frontends rendering package candidates should treat these like other `audit_only` entries (not active scope), and may use the pointers for "covered by the full renovation" messaging.

---

## 6. Weak signals & suppressed candidates

Weak-strength packages are **never emitted** to `packages[]` or `package_candidates[]`. The line-item evidence they would have absorbed continues to flow through `groups[].line_items[]` as standalone cost lines, so cost coverage is preserved.

For debug / audit visibility, a new top-level array `suppressed_package_candidates[]` records weak-strength inference attempts:

```json
{
  "estimate_unit_id": "kitchen_primary",
  "package_type": "kitchen_modernization",
  "package_strength": "weak",
  "suppression_reason": "weak_strength_below_emit_threshold",
  "driver_catalog_item_ids": [],
  "support_catalog_item_ids": ["countertop_damage"]
}
```

Frontends should not surface these — they exist for catalog tuning and "why didn't this become a package?" investigations.

---

## 6. Frontend file migration notes

### `lib/types/renovationEstimate.ts`

Update the `RenoPackage` interface:

```ts
interface RenoPackage {
  // existing fields...
  package_id: string;
  package_type: string;
  // RENAMED — was package_level; values unchanged
  pricing_tier: string;
  // NEW — taxonomy
  package_category: "modernization" | "repair" | "turnover" | "inspection_risk";
  room: "kitchen" | "bathroom" | "bedroom" | "living" | "exterior" | "whole_home" | null;
  package_level: "property" | "room" | "component" | "system";
  // NEW — canonical strength + score
  package_strength: "strong" | "moderate";
  confidence_score: number; // 0..1
  // verification_status gains "confirmed_by_rule"
  verification_status: "confirmed" | "confirmed_by_rule" | "rejected" | "uncertain" | "not_run";
  // existing — supporting_issue_ids[], confirmed_issue_ids[], rejected_issue_ids[]...
}
```

Add `EstimatesProjectScopeBreakdown` import / use — the field is now populated.

### `lib/topPicks/notificationContent.ts`

- `deriveConfidencePosture` can collapse to a passthrough that maps `package_strength` to its UI label. The five existing postures (STRUCTURAL / COSMETIC / MULTI_VERIFIED / PRELIMINARY / EARLY) can be re-derived from `(package_category, package_strength, verification_status)` instead of from raw counts.
- Retire `REPAIR_PREFIXES` — read `package_category === "repair"` instead.

### `lib/topPicks/ranking.ts`

- `extractTopIssuesFromDisplayData` (around lines 201-225) currently flattens to `string[]`. Change to carry the full package object (or the four fields `package_category`, `package_strength`, `confidence_score`, `package_id`) through so ranking can sort by `(package_strength, confidence_score)`.

---

## 7. Backwards compatibility

- All new fields are **additive** on package objects.
- The `pricing_tier` rename has a one-shot fallback on the backend (`pkg.get("pricing_tier") or pkg.get("package_level")`); frontends consuming older artifact files on disk should mirror that fallback during the transition.
- `confirmed_modernization_packages` keeps its existing shape — no breaking change for current consumers.
- `verification_status === "confirmed"` checks still work for VLM-confirmed packages. To also include rule-based confirmation, switch to `ACTIVE_STATUSES.has(verification_status)`.
