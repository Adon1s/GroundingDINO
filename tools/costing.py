"""
costing.py
----------
Property-level scoring and low-level cost helpers.

Pure functions — no LLM calls, no I/O.  Called from artifact_writers.write_photo_intel()
after issues_flat is assembled.

Scoring formula (per issue):
    issue_points = severity * kind_mult * scope_mult * trade_mult

Diminishing returns (geometric decay per catalog item group):
    group_points = base_pts * sum(DIMINISH_FACTOR^k for k=0..n-1)

Rehab score (0–100, saturating exponential):
    rehab_score = round(100 * (1 - exp(-raw / K)))

Cost range helper (allowance-based per catalog item):
    low  = min(base_low  + (n-1)*per_occurrence_low,  cap_low)
    high = min(base_high + (n-1)*per_occurrence_high, cap_high)
    manual allowances are final ranges; heuristic/missing-cost items still use
    kind_mult * scope_mult * trade_mult (NOT severity).

Uncertainty (widen high only):
    high_final = high * (1 + widen_pct)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from math import exp
from typing import Any, Dict, List, Optional, Tuple

from tools.project_scopes import get_project_scope, get_project_scope_name

logger = logging.getLogger(__name__)


class CatalogDataError(Exception):
    """Raised when a catalog item has invalid or missing fields."""
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

KIND_MULT: Dict[str, float] = {
    "defect":  1.0,
    "upgrade": 0.6,
}

SCOPE_MULT: Dict[str, float] = {
    "repair":   1.0,
    "replace":  1.3,
    "cosmetic": 0.8,
    "service":  0.7,
}

TRADE_MULT: Dict[str, float] = {
    "roof_gutters":              1.3,
    "foundation_structure":      1.3,
    "hvac":                      1.3,
    "plumbing":                  1.2,
    "electrical":                1.2,
    "moisture_mold":             1.2,
    "kitchen_cabinets_counters": 1.1,
    "bathroom_fixtures_tile":    1.0,
    "exterior_siding_trim":      1.0,
    "safety_general":            1.0,
    "trim_doors_windows":        0.9,
    "flooring":                  0.9,
    "paint_drywall":             0.8,
    "landscaping_drains":        0.8,
    "masonry_exterior_structure": 1.1,
    "exterior_paint_trim":       0.8,
    "cleaning_turnover":         0.7,
    "interior_finishes":         0.9,
}

DIMINISH_FACTOR: float = 0.35
REHAB_K: float = 18.0

# Subscore grouping — keyed by trade_bucket, not category
TRADE_TO_SUBSCORE: Dict[str, str] = {
    "hvac":                      "systems_score",
    "plumbing":                  "systems_score",
    "electrical":                "systems_score",
    "foundation_structure":      "structure_score",
    "roof_gutters":              "structure_score",
    "moisture_mold":             "structure_score",
    "safety_general":            "structure_score",
    "flooring":                  "cosmetic_score",
    "paint_drywall":             "cosmetic_score",
    "trim_doors_windows":        "cosmetic_score",
    "kitchen_cabinets_counters": "cosmetic_score",
    "bathroom_fixtures_tile":    "cosmetic_score",
    "exterior_siding_trim":      "exterior_score",
    "landscaping_drains":        "exterior_score",
    "masonry_exterior_structure": "exterior_score",
    "exterior_paint_trim":       "exterior_score",
    "cleaning_turnover":         "cosmetic_score",
    "interior_finishes":         "cosmetic_score",
}

MAJOR_SYSTEM_BUCKETS = frozenset({
    "foundation_structure", "roof_gutters", "hvac",
    "plumbing", "electrical", "moisture_mold",
})

# Heuristic cost defaults — used for items without explicit cost objects
SEVERITY_BASE_HEURISTIC: Dict[int, Tuple[int, int]] = {
    1: (100,  500),
    2: (300,  1_500),
    3: (500,  3_000),
    4: (1_500, 8_000),
    5: (3_000, 15_000),
}

SCOPE_COST_MULT: Dict[str, float] = {
    "repair":   1.0,
    "replace":  1.5,
    "cosmetic": 0.6,
    "service":  0.5,
}

# Minimum cost floor by trade bucket (prevents unrealistically cheap estimates
# for trades that always have a minimum mobilisation cost)
TRADE_MIN_BASE_LOW: Dict[str, int] = {
    "roof_gutters":         300,
    "foundation_structure": 500,
    "hvac":                 300,
    "plumbing":             150,
    "electrical":           150,
    "moisture_mold":        200,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_issue_points(
    severity: int,
    kind: str,
    scope: str,
    trade_bucket: str,
) -> float:
    """Per-issue impact points: severity * kind_mult * scope_mult * trade_mult."""
    return (
        severity
        * KIND_MULT.get(kind, 1.0)
        * SCOPE_MULT.get(scope, 1.0)
        * TRADE_MULT.get(trade_bucket, 1.0)
    )


def compute_group_points(base_pts: float, n: int) -> float:
    """
    Geometric diminishing returns for n occurrences of the same catalog item.

    occurrence 1: 1.0x
    occurrence 2: 0.35x
    occurrence 3: 0.35^2 x  (0.1225x)
    ...

    Returns total contributed points for the group.
    """
    if n <= 0:
        return 0.0
    total = 0.0
    factor = 1.0
    for _ in range(n):
        total += base_pts * factor
        factor *= DIMINISH_FACTOR
    return total


def compute_rehab_score(raw_points: float, K: float = REHAB_K) -> int:
    """Map raw points to 0–100 via exponential saturation."""
    if raw_points <= 0:
        return 0
    return round(100.0 * (1.0 - exp(-raw_points / K)))


# ═══════════════════════════════════════════════════════════════════════════════
# Costing functions
# ═══════════════════════════════════════════════════════════════════════════════

def heuristic_cost_range(
    severity: int,
    scope: str,
    trade_bucket: str,
) -> Dict[str, Any]:
    """
    Generate a cost object at runtime for items without explicit cost data.

    Uses severity as the primary driver, adjusted by scope.
    Applies guardrails so ranges are never inverted or unrealistically tight.
    """
    base_low, base_high = SEVERITY_BASE_HEURISTIC.get(severity, (200, 800))
    scope_m = SCOPE_COST_MULT.get(scope, 1.0)
    base_low = round(base_low * scope_m)
    base_high = round(base_high * scope_m)

    # Trade-bucket minimum floor
    trade_floor = TRADE_MIN_BASE_LOW.get(trade_bucket, 0)
    if base_low < trade_floor:
        base_low = trade_floor

    # Guardrail: base_high >= base_low * 1.5
    if base_high < base_low * 1.5:
        base_high = round(base_low * 2.0)

    per_occurrence_low = max(25, round((base_high - base_low) * 0.2))
    per_occurrence_high = max(per_occurrence_low, round((base_high - base_low) * 0.35))

    cap_low = round(base_high * 1.5)
    cap_high = round(base_high * 3.0)

    # Guardrail: cap >= base
    cap_low = max(cap_low, base_low)
    cap_high = max(cap_high, base_high)

    return {
        "mode": "heuristic",
        "base_low": base_low,
        "base_high": base_high,
        "per_occurrence_low": per_occurrence_low,
        "per_occurrence_high": per_occurrence_high,
        "cap_low": cap_low,
        "cap_high": cap_high,
    }


def compute_item_cost_range(
    cost_obj: Dict[str, Any],
    n_occurrences: int,
    kind: str,
    scope: str,
    trade_bucket: str,
    severity: int = 2,
) -> Tuple[int, int]:
    """
    Allowance-based cost for a catalog item across all its occurrences.

    If cost_obj has mode=heuristic or is missing keys, falls back to heuristic.
    Applies kind/scope/trade multipliers (NOT severity — already in base costs).
    """
    # Branch to heuristic if needed
    if not cost_obj or cost_obj.get("mode") == "heuristic":
        cost_obj = heuristic_cost_range(severity, scope, trade_bucket)

    base_low = cost_obj.get("base_low", 200)
    base_high = cost_obj.get("base_high", 800)
    per_occ_low = cost_obj.get("per_occurrence_low", 50)
    per_occ_high = cost_obj.get("per_occurrence_high", 150)
    cap_low = cost_obj.get("cap_low", base_high * 2)
    cap_high = cost_obj.get("cap_high", base_high * 4)

    additional = max(0, n_occurrences - 1)
    raw_low = base_low + additional * per_occ_low
    raw_high = base_high + additional * per_occ_high

    # Apply caps
    raw_low = min(raw_low, cap_low)
    raw_high = min(raw_high, cap_high)

    # Apply multipliers (kind, scope, trade — NOT severity)
    manual_allowance = (
        cost_obj.get("mode") == "allowance"
        and cost_obj.get("cost_source") == "manual"
    )
    if manual_allowance:
        low = round(raw_low)
        high = round(raw_high)
    else:
        mult = (
            KIND_MULT.get(kind, 1.0)
            * SCOPE_MULT.get(scope, 1.0)
            * TRADE_MULT.get(trade_bucket, 1.0)
        )
        low = round(raw_low * mult)
        high = round(raw_high * mult)

    # Final guardrail: high >= low
    if high < low:
        high = low

    return low, high


def compute_uncertainty_widen(
    total_high: int,
    n_photos: int,
    has_speculative: bool,
    has_major_systems: bool,
    unresolved_count: int,
) -> Tuple[int, int]:
    """
    Widen high-end of cost range for low-confidence situations.
    Low stays stable (renovator preference).

    Returns (adjusted_high, widen_pct as integer 0-100).
    """
    widen = 0.0
    if n_photos < 15:
        widen += 0.15
    if has_speculative:
        widen += 0.20
    if has_major_systems:
        widen += 0.10
    if unresolved_count > 0:
        widen += 0.10

    widen_pct = round(widen * 100)
    adjusted_high = round(total_high * (1.0 + widen))
    return adjusted_high, widen_pct


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scoring(
    issues_flat: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
    n_photos: int = 0,
) -> Dict[str, Any]:
    """
    Compute rehab scoring only.

    This intentionally emits no dollar figures. Costing and budget rollups are
    owned by renovation_estimate.
    """
    catalog_items = issue_catalog.get("items", []) or []
    catalog_lookup: Dict[str, Dict[str, Any]] = {}
    for item in catalog_items:
        if isinstance(item, dict) and item.get("id"):
            catalog_lookup[item["id"]] = item

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    has_speculative = False
    unresolved_count = 0
    for issue in issues_flat or []:
        cat_id = issue.get("catalog_item_id") or issue.get("issue_id") or issue.get("id")
        if not cat_id or cat_id not in catalog_lookup:
            unresolved_count += 1
            continue
        status = issue.get("status", "confirmed")
        if status in {"unresolved", "unknown"}:
            unresolved_count += 1
            continue
        if status in {"speculative", "suspected"}:
            has_speculative = True
        grouped.setdefault(cat_id, []).append(issue)

    total_raw = 0.0
    subscore_accum: Dict[str, float] = {
        "systems_score": 0.0,
        "structure_score": 0.0,
        "cosmetic_score": 0.0,
        "exterior_score": 0.0,
    }
    per_item_detail: List[Dict[str, Any]] = []
    scope_accum: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"points": 0.0, "item_count": 0, "scope_name": ""}
    )
    has_major_systems = False

    for cat_id, occurrences in grouped.items():
        cat = catalog_lookup[cat_id]
        sev = cat.get("severity", 2)
        kind = cat.get("kind", "defect")
        scope = cat.get("scope", "repair")
        if not isinstance(scope, str):
            raise CatalogDataError(
                f"Faulty catalog item '{cat_id}': 'scope' must be a string "
                f"(repair|replace|cosmetic|service), got {type(scope).__name__}: {scope!r}"
            )
        tb = cat.get("trade_bucket", "safety_general")
        if tb in MAJOR_SYSTEM_BUCKETS:
            has_major_systems = True

        base_pts = compute_issue_points(sev, kind, scope, tb)
        group_pts = round(compute_group_points(base_pts, len(occurrences)), 3)
        total_raw += group_pts

        subscore_key = TRADE_TO_SUBSCORE.get(tb, "cosmetic_score")
        subscore_accum[subscore_key] += group_pts

        scope_id = get_project_scope(tb, strict=False)
        scope_name = get_project_scope_name(scope_id)
        scope_accum[scope_id]["points"] += group_pts
        scope_accum[scope_id]["item_count"] += 1
        scope_accum[scope_id]["scope_name"] = scope_name

        per_item_detail.append({
            "catalog_item_id": cat_id,
            "name": cat.get("name", cat_id),
            "occurrences": len(occurrences),
            "severity": sev,
            "kind": kind,
            "scope": scope,
            "trade_bucket": tb,
            "impact_points": group_pts,
            "project_scope_id": scope_id,
            "project_scope_name": scope_name,
        })

    total_raw = round(total_raw, 3)
    rehab_score = compute_rehab_score(total_raw)
    subscores = {
        key: compute_rehab_score(pts) for key, pts in subscore_accum.items()
    }

    total_items = sum(s["item_count"] for s in scope_accum.values())
    total_scope_points = sum(s["points"] for s in scope_accum.values())
    dominant_by_points = (
        max(scope_accum.items(), key=lambda x: x[1]["points"])[0]
        if scope_accum else "unknown"
    )
    dominant_by_items = (
        max(scope_accum.items(), key=lambda x: x[1]["item_count"])[0]
        if scope_accum else "unknown"
    )
    project_scope_signals = {
        "dominant_project_scope_by_points": dominant_by_points,
        "dominant_project_scope_by_items": dominant_by_items,
        "scope_concentration_by_points": (
            round(scope_accum[dominant_by_points]["points"] / total_scope_points, 3)
            if total_scope_points > 0 and dominant_by_points in scope_accum else 0.0
        ),
        "scope_concentration_by_items": (
            round(scope_accum[dominant_by_items]["item_count"] / total_items, 3)
            if total_items > 0 and dominant_by_items in scope_accum else 0.0
        ),
        "scope_fragmentation_count": len(scope_accum),
    }

    _, widen_pct = compute_uncertainty_widen(
        total_high=0,
        n_photos=n_photos,
        has_speculative=has_speculative,
        has_major_systems=has_major_systems,
        unresolved_count=unresolved_count,
    )

    return {
        "version": "scoring_v1",
        "raw_points": total_raw,
        "rehab_score": rehab_score,
        "K": REHAB_K,
        "systems_score": subscores.get("systems_score", 0),
        "structure_score": subscores.get("structure_score", 0),
        "cosmetic_score": subscores.get("cosmetic_score", 0),
        "exterior_score": subscores.get("exterior_score", 0),
        "uncertainty": {
            "uncertainty_pct": widen_pct,
            "n_photos": n_photos,
            "has_speculative_items": has_speculative,
            "has_major_system_items": has_major_systems,
            "unresolved_item_count": unresolved_count,
        },
        "project_scope_signals": project_scope_signals,
        "per_item": sorted(per_item_detail, key=lambda x: -x["impact_points"]),
        "meta": {
            "issues_scored": sum(len(v) for v in grouped.values()),
            "unique_catalog_items": len(grouped),
            "unresolved_issues": unresolved_count,
            "diminish_factor": DIMINISH_FACTOR,
            "notes": [
                "Scoring is for ranking/filtering and contains no dollar estimates.",
                "Minor issues still contribute to scoring.",
            ],
        },
    }

