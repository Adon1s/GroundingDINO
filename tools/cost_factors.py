"""Property-level cost factors for the v4 renovation estimate.

Two adjustments derived from scraped listing metadata, applied as ONE uniform
post-hoc scale of the assembled v4 estimate (valid because every dollar
operation in the pipeline — sums, caps, floors, splits — is positively
homogeneous):

  - market factor: list $/sqft vs national baseline (labor is local)
  - size factor:   living sqft vs typical listing (more surface area)

The constants below are the entire "data" of this feature. Baselines are
judgment calls, stated as such — tune here, no data file.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tools.estimate_sanity import _metadata_number

PPSF_BASELINE = 230.0      # ~national median list $/sqft (2025); factor 1.0 here
PPSF_EXPONENT = 0.4        # materials are national, labor is local — sublinear
PPSF_CLAMP = (0.75, 1.5)
SQFT_BASELINE = 1800.0     # typical single-family listing
SQFT_EXPONENT = 0.3        # more surface area, but fixed costs don't scale
SQFT_CLAMP = (0.9, 1.25)
# Headline-tier band blend: how correlated per-item cost outcomes are treated.
# 0 = independent (pure √Σw² quadrature), 1 = perfectly correlated (today's
# straight sum of extremes). Judgment call.
ROLLUP_CORRELATION_RHO = 0.4

# area_price_per_sqft is a zip/area-level baseline (frontend zip medians or
# future sold-price scraping); preferred over the subject's own ratio because
# a distressed fixer lists below its area's ppsf precisely when it needs work.
_PPSF_KEYS = ("area_price_per_sqft", "price_per_sqft")
_LIST_PRICE_KEYS = ("list_price", "price", "listing_price")
_SQFT_KEYS = ("sqft", "square_feet", "living_area_sqft")


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def _resolve_ppsf(metadata: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    for key in _PPSF_KEYS:
        value = _metadata_number(metadata, key)
        if value is not None and value > 0:
            return value, key
    list_price = _metadata_number(metadata, *_LIST_PRICE_KEYS)
    sqft = _metadata_number(metadata, *_SQFT_KEYS)
    if list_price and sqft and list_price > 0 and sqft > 0:
        return list_price / sqft, "list_price/sqft"
    return None, None


def resolve_property_cost_factor(
    property_metadata: Optional[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """One multiplicative factor + audit dict. Neutral (1.0) on missing data."""
    metadata = property_metadata or {}
    reasons: List[str] = []

    ppsf, ppsf_source_key = _resolve_ppsf(metadata)
    if ppsf is None:
        market_factor = 1.0
        reasons.append("no_ppsf_signal_market_factor_neutral")
    else:
        market_factor = _clamp((ppsf / PPSF_BASELINE) ** PPSF_EXPONENT, PPSF_CLAMP)

    sqft = _metadata_number(metadata, *_SQFT_KEYS)
    if sqft is None or sqft <= 0:
        sqft = None
        size_factor = 1.0
        reasons.append("no_sqft_size_factor_neutral")
    else:
        size_factor = _clamp((sqft / SQFT_BASELINE) ** SQFT_EXPONENT, SQFT_CLAMP)

    factor = market_factor * size_factor
    audit = {
        "factor": factor,
        "market_factor": market_factor,
        "size_factor": size_factor,
        "ppsf": ppsf,
        "ppsf_source_key": ppsf_source_key,
        "sqft": sqft,
        "baselines": {
            "ppsf_baseline": PPSF_BASELINE,
            "ppsf_exponent": PPSF_EXPONENT,
            "ppsf_clamp": list(PPSF_CLAMP),
            "sqft_baseline": SQFT_BASELINE,
            "sqft_exponent": SQFT_EXPONENT,
            "sqft_clamp": list(SQFT_CLAMP),
        },
        "reasons": reasons,
    }
    return factor, audit


# Midpoints are recomputed from their scaled pair, never scaled independently
# (independent rounding would break the midpoint == (low+high)//2 invariant).
_MIDPOINT_TRIPLES = (
    ("midpoint", "low", "high"),
    ("cost_midpoint", "cost_low", "cost_high"),
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_dollar_key(key: Any) -> bool:
    return isinstance(key, str) and (
        key in ("low", "high") or key.endswith("_low") or key.endswith("_high")
    )


def scale_estimate_dollars(estimate: Dict[str, Any], factor: float) -> None:
    """Uniformly scale every dollar field in an estimate tree, in place.

    Dollar fields are values keyed "low"/"high" or "*_low"/"*_high" — that
    suffix vocabulary covers every dollar in the v4 output and nothing else.
    Package and bucket dicts are aliased across packages/package_candidates/
    reconciliation, so the visited set is what keeps shared dicts from being
    scaled twice.
    """
    if factor == 1.0:
        return
    visited: set = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if id(node) in visited:
                return
            visited.add(id(node))
            for key, value in node.items():
                if _is_dollar_key(key) and _is_number(value):
                    scaled = value * factor
                    node[key] = round(scaled) if isinstance(value, int) else scaled
                else:
                    _walk(value)
            for mid_key, low_key, high_key in _MIDPOINT_TRIPLES:
                if (
                    _is_number(node.get(mid_key))
                    and _is_number(node.get(low_key))
                    and _is_number(node.get(high_key))
                ):
                    node[mid_key] = (node[low_key] + node[high_key]) // 2
        elif isinstance(node, list):
            if id(node) in visited:
                return
            visited.add(id(node))
            for item in node:
                _walk(item)

    _walk(estimate)
