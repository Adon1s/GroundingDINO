"""Consumer-safe rehab evidence allocation projection (v1).

Builds the versioned public allocation whose authoritative headline is
``photo_supported + needs_inspection`` (endpoint-wise):

  * ``photo_supported``  — the resale-ready rehab range (work directly
    supported by listing-photo evidence).
  * ``needs_inspection`` — the inspection allowance historically excluded
    from that range.
  * ``headline``         — endpoint-wise sum of the two, mutually exclusive
    by construction (tier totals, never package/line-item sums).
  * ``risk_exposure``    — inspection-only exposure, carried alongside but
    NEVER added to the headline (``risk_exposure_included: false``).

The projection must be built AFTER ``scale_estimate_dollars`` so every input
range is already market/size scaled; summing the scaled endpoints is what
makes the raw reconciliation exact. Display rounding is applied once at the
frontend API boundary under the ``compact_currency_v1`` policy declared here.

Fail-closed: any malformed input range yields ``None`` (no projection) rather
than a fabricated allocation. Excluded risk may be served without a headline,
but never with invented allocation dollars or a reconciliation assertion.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

EVIDENCE_PROJECTION_VERSION = "rehab_evidence_projection_v1"
EVIDENCE_PROJECTION_POLICY_VERSION = "rehab_evidence_projection_v1"
EVIDENCE_ROUNDING_POLICY = "compact_currency_v1"
SERVABLE_EVIDENCE_PROJECTION_STATUSES = ("native", "reprojected")

_ZERO_RANGE = {"low": 0, "high": 0}


def _coerce_range(value: Any) -> Optional[Dict[str, int]]:
    """Whitelist a {low, high} dollar range: finite, non-negative, ordered."""
    if not isinstance(value, dict):
        return None
    low = value.get("low")
    high = value.get("high")
    for endpoint in (low, high):
        if isinstance(endpoint, bool) or not isinstance(endpoint, (int, float)):
            return None
        if endpoint != endpoint or endpoint in (float("inf"), float("-inf")):
            return None
    low_i = int(round(low))
    high_i = int(round(high))
    if low_i < 0 or high_i < 0 or low_i > high_i:
        return None
    return {"low": low_i, "high": high_i}


def _optional_range(container: Dict[str, Any], key: str) -> tuple:
    """Return (range, ok). Absent/null -> zero range; present-but-malformed
    -> (None, False) so callers fail closed instead of fabricating dollars."""
    raw = container.get(key)
    if raw is None:
        return dict(_ZERO_RANGE), True
    coerced = _coerce_range(raw)
    if coerced is None:
        return None, False
    return coerced, True


def build_rehab_evidence_projection(
    v4_estimate: Any,
) -> Optional[Dict[str, Any]]:
    """Build the raw (unrounded) evidence allocation from a scaled v4 estimate.

    Returns None when the estimate cannot support a truthful projection.
    Provenance (run id, completion time, projection id) is stamped separately
    by ``stamp_evidence_projection_provenance`` at artifact-write time.
    """
    if not isinstance(v4_estimate, dict):
        return None
    totals = v4_estimate.get("totals")
    totals = totals if isinstance(totals, dict) else {}

    risk, risk_ok = _optional_range(totals, "risk_exposure_total")
    if not risk_ok:
        return None

    raw_photo = v4_estimate.get("final_rehab_resale_ready")
    if raw_photo is None:
        # Risk-only case: excluded exposure without a headline is permitted,
        # but allocation dollars and a reconciliation assertion are not.
        if risk["high"] <= 0:
            return None
        return {
            "version": EVIDENCE_PROJECTION_VERSION,
            "policy_version": EVIDENCE_PROJECTION_POLICY_VERSION,
            "currency": "USD",
            "rounding_policy": EVIDENCE_ROUNDING_POLICY,
            "risk_exposure": risk,
            "risk_exposure_included": False,
        }

    photo = _coerce_range(raw_photo)
    if photo is None:
        return None
    inspection, inspection_ok = _optional_range(totals, "inspection_allowance_total")
    if not inspection_ok:
        return None

    headline = {
        "low": photo["low"] + inspection["low"],
        "high": photo["high"] + inspection["high"],
    }
    return {
        "version": EVIDENCE_PROJECTION_VERSION,
        "policy_version": EVIDENCE_PROJECTION_POLICY_VERSION,
        "currency": "USD",
        "rounding_policy": EVIDENCE_ROUNDING_POLICY,
        "photo_supported": photo,
        "needs_inspection": inspection,
        "headline": headline,
        "risk_exposure": risk,
        "risk_exposure_included": False,
        "reconciliation": {
            "identity": "photo_supported + needs_inspection == headline",
            "raw_exact": True,
        },
    }


def _projection_id(projection: Dict[str, Any]) -> str:
    provenance = projection.get("provenance") or {}
    content = {
        "version": projection.get("version"),
        "policy_version": projection.get("policy_version"),
        "photo_supported": projection.get("photo_supported"),
        "needs_inspection": projection.get("needs_inspection"),
        "headline": projection.get("headline"),
        "risk_exposure": projection.get("risk_exposure"),
        "source_run_id": provenance.get("source_run_id"),
    }
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"rep1_{digest[:16]}"


def stamp_evidence_projection_provenance(
    projection: Dict[str, Any],
    *,
    run_id: str,
    completed_at: Any,
    source_artifact: str,
    projection_status: str,
    product_policy_version: str,
) -> None:
    """Stamp run/artifact provenance and the deterministic projection id.

    ``completed_at`` is the single run-completion timestamp shared by the
    finished run, the cached estimate, and this provenance block.
    ``source_artifact`` must be a relative identity, never an absolute path.
    """
    projection["provenance"] = {
        "source_run_id": str(run_id or ""),
        "artifact_job_id": str(run_id or ""),
        "completed_at": completed_at,
        "source_artifact": source_artifact,
        "product_policy_version": product_policy_version,
        "projection_status": projection_status,
    }
    projection["projection_id"] = _projection_id(projection)
