"""
tools/renovation_estimate_v4.py

Room-aware renovation estimate orchestrator.

Pipeline:
  1. Cluster photos into room surrogates (kitchen_1, kitchen_2, ...).
  2. Resolve billable estimate units (kitchen_primary, bathroom_primary, ...)
     using property_metadata caps for kitchen, bathroom, and bedroom counts.
  3. Deep-copy issues_flat and stamp both `room_surrogate_id` and
     `estimate_unit_id` so the same catalog item can split or merge per
     billable unit without mutating v3 inputs.
  4. Re-extract candidates, reuse Pass 2f review fields from v3 reviewed
     candidates (no VLM re-run), apply scope and cost-model classification.
  5. Run the v3 group-estimate machinery on the room-aware candidates.
  6. Infer kitchen / bathroom / room packages keyed by estimate_unit_id and
     reconcile them against group line items (absorption, net delta,
     group-cap-aware totals).
  7. Assemble explicit estimate buckets (visible_rehab,
     package_adjusted_rehab, latent_risk_exposure, worst_case_exposure,
     final_rehab) and append non-mutating sanity flags.

v3 output is left untouched throughout.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

from tools.catalog_cost_model import derive_cost_model
from tools.estimate_scope import apply_estimate_scope
from tools.estimate_sanity import build_estimate_sanity_flags
from tools.rehab_packages import (
    infer_packages,
    reconcile_packages_and_estimate_units,
)
from tools.renovation_estimate import (
    EstimateCandidate,
    compute_renovation_estimate,
    extract_estimate_candidates,
)
from tools.estimate_units import build_estimate_units
from tools.room_surrogates import build_room_surrogates

logger = logging.getLogger(__name__)

_PASS_2F_REUSE_FIELDS = (
    "is_valid_detection",
    "review_posture",
    "review_visible_scope",
    "review_rationale",
    "review_consistency_flags",
    "review_source",
    "review_image_path",
    "effective_posture",
    "pass_2f_attempted",
    "pass_2f_applied",
    "pass_2f_fallback_reason",
)

_REUSE_METHOD_EXACT = "exact"
_REUSE_METHOD_SUBSET = "subset"
_REUSE_METHOD_COLLAPSE_AGREEING = "collapse_agreeing_postures"

_REUSE_CONFIDENCE_HIGH = "high"
_REUSE_CONFIDENCE_MEDIUM = "medium"


def compute_renovation_estimate_v4(
    issues_flat: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
    photos: Dict[str, Any],
    *,
    v3_reviewed_candidates: Optional[List[EstimateCandidate]] = None,
    v3_estimate: Optional[Dict[str, Any]] = None,
    property_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute the v4 (room-aware) renovation estimate.

    Runs the full pipeline: surrogate clustering, estimate-unit resolution,
    Pass 2f reuse, candidate scope and cost-model classification, group
    aggregation, package inference, reconciliation, and sanity flags.

    `property_metadata` carries scraped listing facts (beds, baths, price,
    sqft, property_type, multi-kitchen evidence) used by estimate-unit caps
    and sanity flags. None disables those features but the rest of the
    pipeline still runs.
    """
    if v3_estimate is None:
        return None

    surrogates = build_room_surrogates(photos or {})
    surrogate_records = surrogates.get("room_surrogates", [])
    estimate_unit_resolution = build_estimate_units(
        photos or {},
        surrogate_records,
        property_metadata=property_metadata,
    )
    photo_to_surrogate = surrogates.get("photo_key_to_room_surrogate_id", {}) or {}
    photo_to_estimate_unit = (
        estimate_unit_resolution.get("photo_to_estimate_unit_id", {}) or {}
    )

    v4_issues: List[Dict[str, Any]] = []
    for issue in issues_flat or []:
        v4_issue = copy.deepcopy(issue)
        photo_key = v4_issue.get("photo_key")
        if photo_key and photo_key in photo_to_surrogate:
            v4_issue["room_surrogate_id"] = photo_to_surrogate[photo_key]
        if photo_key and photo_key in photo_to_estimate_unit:
            v4_issue["estimate_unit_id"] = photo_to_estimate_unit[photo_key]
        v4_issues.append(v4_issue)

    v4_candidates = extract_estimate_candidates(v4_issues, issue_catalog)

    pass_2f_reuse_audit = _reuse_pass_2f_fields(
        v4_candidates,
        v3_reviewed_candidates or [],
    )
    catalog_lookup = {
        item["id"]: item
        for item in (issue_catalog.get("items") or [])
        if isinstance(item, dict) and item.get("id")
    }
    for candidate in v4_candidates:
        apply_estimate_scope(
            candidate,
            catalog_lookup.get(candidate.catalog_item_id, {}),
        )
        candidate.cost_model, candidate.cost_model_source = derive_cost_model(
            catalog_lookup.get(candidate.catalog_item_id, {}),
            candidate,
        )

    v4_estimate = compute_renovation_estimate(
        issues_flat=v4_issues,
        issue_catalog=issue_catalog,
        prebuilt_candidates=v4_candidates,
    )

    packages = infer_packages(
        v4_candidates,
        surrogate_records,
        issue_catalog,
        estimate_units=estimate_unit_resolution.get("estimate_units", []),
    )
    reconciliation = reconcile_packages_and_estimate_units(
        v4_estimate.get("groups", []),
        packages,
    )

    v4_estimate["version"] = "renovation_estimate_v4"
    v4_estimate["room_surrogates"] = surrogate_records
    v4_estimate["estimate_units"] = estimate_unit_resolution.get("estimate_units", [])
    v4_estimate["photo_to_estimate_unit_id"] = photo_to_estimate_unit
    v4_estimate["room_surrogate_to_estimate_unit_id"] = (
        estimate_unit_resolution.get("room_surrogate_to_estimate_unit_id", {}) or {}
    )
    v4_estimate["estimate_unit_merge_decisions"] = (
        estimate_unit_resolution.get("merge_decisions", []) or []
    )
    v4_estimate["estimate_unit_warnings"] = (
        estimate_unit_resolution.get("warnings", []) or []
    )
    v4_estimate["packages"] = packages
    v4_estimate["reconciliation"] = reconciliation
    v4_estimate["warnings"] = list(reconciliation.get("warnings") or [])
    for bucket_name in (
        "visible_rehab",
        "package_adjusted_rehab",
        "latent_risk_exposure",
        "worst_case_exposure",
        "final_rehab",
        "totals_by_scope_raw",
        "totals_by_scope_capped",
        "final_rehab_required",
        "final_rehab_resale_ready",
    ):
        v4_estimate[bucket_name] = reconciliation[bucket_name]
    v4_estimate["sanity_flags"] = build_estimate_sanity_flags(
        v4_estimate,
        property_metadata,
        v4_estimate.get("estimate_units", []),
    )
    v4_estimate["pass_2f_reuse_audit"] = pass_2f_reuse_audit
    v4_estimate["provenance"] = {
        "mode": "room_aware_line_item_estimate",
        "derived_from": "renovation_estimate_v3",
        "v3_pass_2f_reused": bool(v3_reviewed_candidates),
        "v4_phases_applied": ["surrogates", "estimate_units", "packages", "reconciliation"],
        "packages_enabled": True,
        "reconciliation_enabled": True,
    }

    return v4_estimate


def _reuse_pass_2f_fields(
    v4_candidates: List[EstimateCandidate],
    v3_reviewed_candidates: List[EstimateCandidate],
) -> Dict[str, int]:
    """Copy Pass 2f review fields from v3 candidates onto matching v4 candidates.

    Casework on the v4 candidate's issue_ids vs v3 candidates with the same
    catalog_item_id:
      - Exact match (V4_IDS == V3_IDS for one v3): copy.
      - Subset (V4_IDS subset of exactly one v3 superset): copy. The typical
        "v3 lumped, v4 split by room" case.
      - Subset (V4_IDS subset of multiple v3 supersets): ambiguous, skip.
      - Collapse (union of disjoint v3 subsets equals V4_IDS): copy ONLY
        when all source v3 candidates share the same review_posture.
        Disagreement falls through to ambiguous. Dominance ordering is
        deferred to a later PR.
      - Otherwise: unmatched.
    """
    audit = {
        "matched_count": 0,
        "unmatched_v4_count": 0,
        "ambiguous_count": 0,
        # Reserved for dominance-ordering collapse; currently always 0.
        "dominant_posture_collapses": 0,
    }

    by_catalog_id: Dict[str, List[EstimateCandidate]] = {}
    for v3_c in v3_reviewed_candidates:
        by_catalog_id.setdefault(v3_c.catalog_item_id, []).append(v3_c)

    for v4_c in v4_candidates:
        v3_pool = by_catalog_id.get(v4_c.catalog_item_id, [])
        if not v3_pool:
            audit["unmatched_v4_count"] += 1
            continue

        v4_ids = frozenset(v4_c.issue_ids)

        exact = [v3 for v3 in v3_pool if frozenset(v3.issue_ids) == v4_ids]
        if len(exact) == 1:
            _copy_pass_2f_fields(
                v4_c, exact[0],
                method=_REUSE_METHOD_EXACT,
                confidence=_REUSE_CONFIDENCE_HIGH,
            )
            audit["matched_count"] += 1
            continue
        if len(exact) > 1:
            audit["ambiguous_count"] += 1
            continue

        supersets = [v3 for v3 in v3_pool if frozenset(v3.issue_ids) > v4_ids]
        if len(supersets) == 1:
            _copy_pass_2f_fields(
                v4_c, supersets[0],
                method=_REUSE_METHOD_SUBSET,
                confidence=_REUSE_CONFIDENCE_MEDIUM,
            )
            audit["matched_count"] += 1
            continue
        if len(supersets) > 1:
            audit["ambiguous_count"] += 1
            continue

        subsets = [v3 for v3 in v3_pool if frozenset(v3.issue_ids) < v4_ids]
        union_ids = frozenset().union(
            *(frozenset(v3.issue_ids) for v3 in subsets)
        ) if subsets else frozenset()
        if subsets and union_ids == v4_ids:
            postures = {v3.review_posture for v3 in subsets}
            if len(postures) == 1:
                _copy_pass_2f_fields(
                    v4_c, subsets[0],
                    method=_REUSE_METHOD_COLLAPSE_AGREEING,
                    confidence=_REUSE_CONFIDENCE_MEDIUM,
                )
                audit["matched_count"] += 1
            else:
                audit["ambiguous_count"] += 1
            continue

        audit["unmatched_v4_count"] += 1

    return audit


def _copy_pass_2f_fields(
    target: EstimateCandidate,
    source: EstimateCandidate,
    *,
    method: str,
    confidence: str,
) -> None:
    for field in _PASS_2F_REUSE_FIELDS:
        value = getattr(source, field, None)
        if isinstance(value, list):
            value = list(value)
        setattr(target, field, value)
    setattr(target, "pass_2f_reuse_method", method)
    setattr(target, "pass_2f_reuse_confidence", confidence)
