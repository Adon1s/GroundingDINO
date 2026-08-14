"""Deterministic condition disposition policy (condition_disposition_v1).

Pure: verdict x terminal route x objective evidence -> disposition, applied
after Terra's verdict and never consulting model confidence. The
insufficient-distinct-views gate runs before route resolution on supported
verdicts, so a high-consequence single-view finding is withheld regardless of
its route. validators.py recomputes this function to reject any result whose
recorded dispositions contradict the policy.
"""
from __future__ import annotations

from typing import Optional, Tuple

# route -> (disposition, reason_code) for supported verdicts with adequate
# evidence. The table is the whole policy — no per-item special cases.
_SUPPORTED_ROUTE_TABLE = {
    "work": ("accepted_for_work", "route_work"),
    "inspection": ("inspection", "route_inspection"),
    "no_action": ("no_action", "route_no_action"),
    "excluded_generic": ("excluded", "route_excluded_generic"),
    "excluded_quarantine": ("excluded", "route_excluded_quarantine"),
}


def decide_disposition(
    verdict: str,
    terminal_route: str,
    distinct_view_count: Optional[int],
    min_photo_evidence_required: Optional[int],
) -> Tuple[str, str]:
    """Resolve one condition's terminal disposition as (disposition, reason_code)."""
    if verdict == "unsupported":
        return "excluded", "verdict_unsupported"
    if verdict == "cannot_assess":
        return "inspection", "verdict_cannot_assess"
    if verdict != "supported":
        raise ValueError(f"unknown review verdict {verdict!r}")
    if (
        min_photo_evidence_required is not None
        and (distinct_view_count or 0) < min_photo_evidence_required
    ):
        return "withheld", "insufficient_distinct_views"
    try:
        return _SUPPORTED_ROUTE_TABLE[terminal_route]
    except KeyError:
        raise ValueError(f"unknown terminal route {terminal_route!r}") from None
