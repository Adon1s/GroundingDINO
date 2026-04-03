"""
project_scopes.py
-----------------
Project scope taxonomy — the broad, market-facing grouping layer.

Maps fine-grained trade_buckets to broader operational scopes that reflect
how renovation work is actually bundled and understood by contractors,
investors, and project managers.

This is Layer B (additive).  Layer A (trade_bucket) remains the primary
classification for cost estimation, scoring, and catalog assignment.

Usage:
    from tools.project_scopes import get_project_scope, get_project_scope_name

    scope = get_project_scope("flooring")           # "interior_generalist"
    name  = get_project_scope_name("mechanical")    # "Mechanical Trades"

v1 mapping policy — treat as initial taxonomy, not eternal truth.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Project scope definitions
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_SCOPES: List[Dict[str, str]] = [
    {"id": "interior_generalist", "name": "Interior Generalist"},
    {"id": "kitchen_bath",        "name": "Kitchen & Bath"},
    {"id": "mechanical",          "name": "Mechanical Trades"},
    {"id": "exterior",            "name": "Exterior"},
    {"id": "structure",           "name": "Structure"},
    {"id": "site_drainage",       "name": "Site & Drainage"},
    {"id": "remediation",         "name": "Remediation & Safety"},
]

PROJECT_SCOPE_IDS: frozenset = frozenset(s["id"] for s in PROJECT_SCOPES)

_SCOPE_ID_TO_NAME: Dict[str, str] = {s["id"]: s["name"] for s in PROJECT_SCOPES}

# ═══════════════════════════════════════════════════════════════════════════════
# Trade bucket → project scope mapping  (v1 policy)
# ═══════════════════════════════════════════════════════════════════════════════

TRADE_BUCKET_TO_PROJECT_SCOPE: Dict[str, str] = {
    # interior_generalist
    "flooring":                  "interior_generalist",
    "paint_drywall":             "interior_generalist",
    "trim_doors_windows":        "interior_generalist",   # v1: window replacement may later split to exterior
    "interior_finishes":         "interior_generalist",
    "cleaning_turnover":         "interior_generalist",

    # kitchen_bath
    "kitchen_cabinets_counters": "kitchen_bath",
    "bathroom_fixtures_tile":    "kitchen_bath",

    # mechanical
    "plumbing":                  "mechanical",
    "electrical":                "mechanical",
    "hvac":                      "mechanical",

    # exterior
    "roof_gutters":              "exterior",
    "exterior_siding_trim":      "exterior",
    "exterior_paint_trim":       "exterior",
    "masonry_exterior_structure": "exterior",

    # structure
    "foundation_structure":      "structure",

    # site_drainage
    "landscaping_drains":        "site_drainage",

    # remediation
    "moisture_mold":             "remediation",
    "safety_general":            "remediation",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_project_scope(trade_bucket: str, *, strict: bool = True) -> str:
    """Return the project scope for a given trade bucket.

    Args:
        trade_bucket: A trade_bucket id (e.g. "flooring").
        strict: If True (default), raises KeyError on unmapped buckets so
                schema drift is caught immediately.  Use strict=False only
                in presentation / display contexts where a fallback is safe.

    Returns:
        Project scope id string.

    Raises:
        KeyError: If strict=True and trade_bucket has no mapping.
    """
    scope = TRADE_BUCKET_TO_PROJECT_SCOPE.get(trade_bucket)
    if scope is not None:
        return scope

    if strict:
        raise KeyError(
            f"Trade bucket '{trade_bucket}' has no project_scope mapping. "
            f"Add it to TRADE_BUCKET_TO_PROJECT_SCOPE in project_scopes.py."
        )

    logger.warning(
        "Trade bucket '%s' has no project_scope mapping — returning 'unknown'.",
        trade_bucket,
    )
    return "unknown"


def get_project_scope_name(scope_id: str) -> str:
    """Return the display name for a project scope id."""
    return _SCOPE_ID_TO_NAME.get(scope_id, scope_id.replace("_", " ").title())


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level validation (runs at import time)
# ═══════════════════════════════════════════════════════════════════════════════

# Every mapped value must be a valid project scope id
_invalid_targets = {
    v for v in TRADE_BUCKET_TO_PROJECT_SCOPE.values()
    if v not in PROJECT_SCOPE_IDS
}
assert not _invalid_targets, (
    f"TRADE_BUCKET_TO_PROJECT_SCOPE maps to invalid scope ids: {_invalid_targets}"
)

# Every project scope must have at least one trade bucket mapped to it
_mapped_scopes = set(TRADE_BUCKET_TO_PROJECT_SCOPE.values())
_orphan_scopes = PROJECT_SCOPE_IDS - _mapped_scopes
assert not _orphan_scopes, (
    f"Project scopes with no trade buckets mapped: {_orphan_scopes}"
)
