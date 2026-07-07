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
import asyncio
import concurrent.futures
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.catalog_cost_model import derive_cost_model
from tools.cost_factors import resolve_property_cost_factor, scale_estimate_dollars
from tools.estimate_scope import apply_estimate_scope
from tools.estimate_sanity import build_estimate_sanity_flags
from tools.rehab_packages import (
    ACTIVE_PACKAGE_STATUSES,
    aggregate_whole_home_turnover,
    apply_package_verifications_to_candidates,
    apply_same_unit_package_subsumption,
    build_package_affinity,
    expand_bathroom_modernization_packages,
    finalize_package_candidates,
    infer_package_candidates,
    package_affinity_for,
    reconcile_packages_and_estimate_units,
    run_pass_2f_batch,
)
from tools.renovation_estimate import (
    EstimateCandidate,
    _clean_scope_component,
    _estimate_scope_key_for_issue,
    _meaningful_scope_hint,
    _resolve_catalog_estimate_meta,
    compute_renovation_estimate,
    extract_estimate_candidates,
    issue_evidence_refs,
)
from tools.estimate_units import bathroom_metadata_cap, build_estimate_units
from tools.project_scopes import get_project_scope, get_project_scope_name
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
    "package_id",
    "package_type",
    "package_role",
    "visual_verification_status",
    "package_verification_source",
)

_REUSE_METHOD_EXACT = "exact"
_REUSE_METHOD_SUBSET = "subset"
_REUSE_METHOD_COLLAPSE_AGREEING = "collapse_agreeing_postures"

def compute_renovation_estimate_v4(
    issues_flat: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
    photos: Dict[str, Any],
    *,
    v3_reviewed_candidates: Optional[List[EstimateCandidate]] = None,
    property_metadata: Optional[Dict[str, Any]] = None,
    package_verifications: Optional[Dict[str, Dict[str, Any]]] = None,
    pass_2f_vlm_client: Any = None,
    pass_2f_model_config: Optional[Dict[str, Any]] = None,
    photo_key_to_path: Optional[Dict[str, Path]] = None,
    pass_2f_provider: str = "premium",
) -> Dict[str, Any]:
    """Compute the v4 (room-aware) renovation estimate.

    Runs the full pipeline: surrogate clustering, estimate-unit resolution,
    Pass 2f reuse, candidate scope and cost-model classification, group
    aggregation, package inference, reconciliation, and sanity flags.

    `property_metadata` carries scraped listing facts (beds, baths, price,
    sqft, property_type, multi-kitchen evidence) used by estimate-unit caps
    and sanity flags. None disables those features but the rest of the
    pipeline still runs.
    """

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
    package_only_candidates = _extract_package_only_candidates(v4_issues, issue_catalog)

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

    suppressed_package_candidates: List[Dict[str, Any]] = []
    package_inference_candidates = v4_candidates + package_only_candidates
    package_candidates = infer_package_candidates(
        package_inference_candidates,
        surrogate_records,
        issue_catalog,
        estimate_units=estimate_unit_resolution.get("estimate_units", []),
        suppressed_out=suppressed_package_candidates,
    )
    pass_2f_trace = {
        "ran": False,
        "reason": "no_package_candidates" if not package_candidates else "not_requested",
        "candidate_count": len(package_candidates),
        "attempted_count": 0,
        "confirmed_count": 0,
        "rejected_count": 0,
        "uncertain_count": 0,
        "no_image_count": 0,
    }
    if (
        package_candidates
        and package_verifications is None
        and pass_2f_vlm_client is not None
        and pass_2f_model_config
    ):
        package_verifications, pass_2f_trace = _run_pass_2f_sync(
            package_candidates,
            vlm_client=pass_2f_vlm_client,
            model_config=pass_2f_model_config,
            photo_key_to_path=photo_key_to_path or {},
            provider=pass_2f_provider,
        )
    elif package_verifications is not None:
        pass_2f_trace = {
            **pass_2f_trace,
            "ran": False,
            "reason": "provided_verifications",
            "verification_count": len(package_verifications or {}),
        }

    apply_package_verifications_to_candidates(
        v4_candidates,
        package_candidates,
        package_verifications,
        provider=pass_2f_provider,
    )
    packages, package_candidates_audit = finalize_package_candidates(
        package_candidates,
        package_verifications,
        require_confirmation=True,
    )

    # Must run before bathroom expansion (expanded packages lose their
    # estimate_unit_id) and before whole-home turnover aggregation (which must
    # only see surviving turnover packages).
    packages, subsumption_audit = apply_same_unit_package_subsumption(packages)

    bathroom_signal = _build_bathroom_room_count_signal(package_candidates_audit)
    bathroom_cap = bathroom_metadata_cap(property_metadata or {})
    packages, bathroom_expansion_audit = expand_bathroom_modernization_packages(
        packages,
        bathroom_room_count_signal=bathroom_signal,
        bathroom_metadata_cap=bathroom_cap,
    )

    whole_home_turnover = aggregate_whole_home_turnover(packages)
    if whole_home_turnover is not None:
        packages.append(whole_home_turnover)
        package_candidates_audit.append(whole_home_turnover)

    v4_estimate = compute_renovation_estimate(
        issues_flat=v4_issues,
        issue_catalog=issue_catalog,
        prebuilt_candidates=v4_candidates,
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
    v4_estimate["package_candidates"] = package_candidates_audit
    v4_estimate["bathroom_room_count_signal"] = bathroom_signal
    v4_estimate["bathroom_expansion_audit"] = bathroom_expansion_audit
    v4_estimate["package_subsumption_audit"] = subsumption_audit
    v4_estimate["suppressed_package_candidates"] = suppressed_package_candidates
    v4_estimate["pass_2f_trace"] = pass_2f_trace
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
        "final_rehab_full_renewal",
    ):
        v4_estimate[bucket_name] = reconciliation[bucket_name]
    v4_estimate["project_scope_breakdown"] = _build_project_scope_breakdown(
        groups=v4_estimate.get("groups") or [],
        packages=packages,
    )
    # Market/size scaling must precede sanity flags, which compare scaled
    # totals against list price.
    cost_factor, cost_factor_audit = resolve_property_cost_factor(property_metadata)
    scale_estimate_dollars(v4_estimate, cost_factor)
    v4_estimate["cost_adjustment"] = cost_factor_audit
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
        "v4_phases_applied": [
            "surrogates",
            "estimate_units",
            "package_candidates",
            "pass_2f",
            "package_finalization",
            "package_subsumption",
            "reconciliation",
            "cost_adjustment",
        ],
        "packages_enabled": True,
        "reconciliation_enabled": True,
        "package_confirmation_required": True,
    }

    return v4_estimate


def _extract_package_only_candidates(
    issues_flat: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
) -> List[EstimateCandidate]:
    """Extract package-role evidence that should not become line items."""
    catalog_lookup = {
        item["id"]: item
        for item in (issue_catalog.get("items") or [])
        if isinstance(item, dict) and item.get("id")
    }
    affinity_table = build_package_affinity(issue_catalog)
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
    for issue in issues_flat or []:
        cat_id = str(issue.get("catalog_item_id") or "")
        cat = catalog_lookup.get(cat_id)
        if not cat:
            continue
        # The catalog's package_affinity blocks are the router: admit an issue
        # when its scene group resolves to an affinity entry (single-room
        # blocks fall back to their only room when the scene doesn't resolve).
        # The overlay in infer_package_candidates re-applies package_type/role
        # downstream.
        if package_affinity_for(issue.get("scene_group", ""), cat_id, affinity_table) is None:
            continue
        est_meta = _resolve_catalog_estimate_meta(cat)
        if est_meta.affects_estimate:
            continue
        scope_key, room_surrogate = _estimate_scope_key_for_issue(cat_id, issue)
        grouped.setdefault((cat_id, scope_key, room_surrogate), []).append(issue)

    candidates: List[EstimateCandidate] = []
    for (cat_id, scope_key, room_surrogate), occurrences in grouped.items():
        cat = catalog_lookup[cat_id]
        est_meta = _resolve_catalog_estimate_meta(cat)
        scene_groups = sorted(set(
            issue.get("scene_group", "other") or "other" for issue in occurrences
        ))
        photo_keys = sorted(set(
            issue.get("photo_key", "") for issue in occurrences
            if issue.get("photo_key")
        ))
        issue_ids = [
            issue.get("issue_id", "") for issue in occurrences
            if issue.get("issue_id")
        ]
        billable_estimate_unit_ids = sorted(set(
            _meaningful_scope_hint(issue.get("estimate_unit_id"))
            for issue in occurrences
            if _meaningful_scope_hint(issue.get("estimate_unit_id"))
        ))
        source_room_surrogate_ids = sorted(set(
            _meaningful_scope_hint(issue.get("room_surrogate_id"))
            for issue in occurrences
            if _meaningful_scope_hint(issue.get("room_surrogate_id"))
        ))
        observations = [
            issue.get("description", "") for issue in occurrences
            if issue.get("description")
        ]
        unit_id = f"{cat_id}:{_clean_scope_component(scope_key)}"
        candidate = EstimateCandidate(
            catalog_item_id=cat_id,
            catalog_item_name=cat.get("name", cat_id),
            estimate_meta=est_meta,
            kind=cat.get("kind", "defect"),
            severity=cat.get("severity", 2),
            scope=cat.get("scope", "repair"),
            trade_bucket=cat.get("trade_bucket", "safety_general"),
            cost_obj=cat.get("cost") or {},
            occurrences=len(occurrences),
            estimate_unit_count=1,
            scene_groups_seen=scene_groups,
            photo_keys=photo_keys,
            issue_ids=issue_ids,
            estimate_unit_id=unit_id,
            billable_estimate_unit_id=(
                billable_estimate_unit_ids[0]
                if len(billable_estimate_unit_ids) == 1
                else ("multiple_estimate_units" if billable_estimate_unit_ids else "")
            ),
            estimate_scope_key=scope_key,
            room_surrogate_id=room_surrogate,
            source_room_surrogate_ids=source_room_surrogate_ids,
            supporting_observations=observations,
            evidence_refs=issue_evidence_refs(occurrences),
            distinct_photo_count=len(photo_keys),
            distinct_scene_group_count=len(scene_groups),
            representative_issue_id=issue_ids[0] if issue_ids else None,
            package_evidence_only=True,
        )
        apply_estimate_scope(candidate, cat)
        candidate.cost_model, candidate.cost_model_source = derive_cost_model(
            cat,
            candidate,
        )
        candidates.append(candidate)

    candidates.sort(key=lambda c: (
        c.catalog_item_id,
        c.estimate_scope_key,
        c.estimate_unit_id,
    ))
    return candidates


def _build_bathroom_room_count_signal(
    package_candidates_audit: List[Dict[str, Any]],
) -> Dict[str, Any]:
    multiple_package_ids: List[str] = []
    one_count = 0
    unclear_count = 0
    for package in package_candidates_audit or []:
        if str(package.get("room") or "").lower() != "bathroom":
            continue
        if "visible_room_count" not in package:
            continue
        value = str(package.get("visible_room_count") or "unclear").strip().lower()
        if value == "multiple_rooms":
            status = str(package.get("verification_status") or "").strip().lower()
            if status and status not in ACTIVE_PACKAGE_STATUSES:
                continue
            package_id = str(package.get("package_id") or "")
            if package_id:
                multiple_package_ids.append(package_id)
        elif value == "one_room":
            one_count += 1
        else:
            unclear_count += 1
    multiple_count = len(multiple_package_ids)
    return {
        "likely_multiple_visible_bathrooms": multiple_count >= 1 and one_count == 0,
        "multiple_rooms_vote_count": multiple_count,
        "one_room_vote_count": one_count,
        "unclear_vote_count": unclear_count,
        "supporting_package_ids": multiple_package_ids,
        "basis": "requires_active_multiple_room_vote_no_one_room_votes",
    }


def _build_project_scope_breakdown(
    *,
    groups: List[Dict[str, Any]],
    packages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate group + package costs by project scope for the frontend.

    Mirrors the frontend's ``EstimatesProjectScopeBreakdown`` type:
    each entry contains scope_id, scope_name, summed costs, item count,
    contributing trade_buckets, and contributing package_ids.
    """
    by_scope: Dict[str, Dict[str, Any]] = {}

    for group in groups or []:
        for li in group.get("line_items") or []:
            trade = str(li.get("trade_bucket") or "")
            if not trade:
                continue
            scope_id = get_project_scope(trade, strict=False)
            entry = by_scope.setdefault(scope_id, {
                "scope_id": scope_id,
                "scope_name": get_project_scope_name(scope_id),
                "cost_low": 0,
                "cost_high": 0,
                "item_count": 0,
                "trade_buckets": set(),
                "contributing_package_ids": set(),
            })
            entry["cost_low"] += int(li.get("cost_low") or 0)
            entry["cost_high"] += int(li.get("cost_high") or 0)
            entry["item_count"] += 1
            entry["trade_buckets"].add(trade)

    for pkg in packages or []:
        absorbed = pkg.get("absorption_scope") or {}
        trade_buckets = absorbed.get("trade_buckets") or []
        if not trade_buckets:
            continue
        package_id = str(pkg.get("package_id") or "")
        for trade in trade_buckets:
            scope_id = get_project_scope(str(trade), strict=False)
            entry = by_scope.setdefault(scope_id, {
                "scope_id": scope_id,
                "scope_name": get_project_scope_name(scope_id),
                "cost_low": 0,
                "cost_high": 0,
                "item_count": 0,
                "trade_buckets": set(),
                "contributing_package_ids": set(),
            })
            entry["trade_buckets"].add(str(trade))
            if package_id:
                entry["contributing_package_ids"].add(package_id)

    breakdown: List[Dict[str, Any]] = []
    for scope_id in sorted(by_scope.keys()):
        entry = by_scope[scope_id]
        breakdown.append({
            "scope_id": entry["scope_id"],
            "scope_name": entry["scope_name"],
            "cost_low": entry["cost_low"],
            "cost_high": entry["cost_high"],
            "item_count": entry["item_count"],
            "trade_buckets": sorted(entry["trade_buckets"]),
            "contributing_package_ids": sorted(entry["contributing_package_ids"]),
        })
    return breakdown


def _run_pass_2f_sync(
    package_candidates: List[Dict[str, Any]],
    *,
    vlm_client: Any,
    model_config: Dict[str, Any],
    photo_key_to_path: Dict[str, Path],
    provider: str,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    coro = run_pass_2f_batch(
        package_candidates,
        vlm_client=vlm_client,
        model_config=model_config,
        photo_key_to_path=photo_key_to_path,
        provider=provider,
    )
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


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
) -> None:
    for field in _PASS_2F_REUSE_FIELDS:
        value = getattr(source, field, None)
        if isinstance(value, list):
            value = list(value)
        setattr(target, field, value)
    setattr(target, "pass_2f_reuse_method", method)
