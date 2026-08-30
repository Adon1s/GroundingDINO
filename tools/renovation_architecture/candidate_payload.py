"""The per-candidate payload Sol judges — shaped here, provider-free.

Split out of sol_review so the offline replay harness can hash exactly what was
reviewed without importing the Sol client module (scripts/replay_renovation_
architecture.py is required to stay structurally incapable of spending provider
tokens). This module holds pure data shaping: no client, no network, no budget.

Why a payload hash exists at all: make_package_candidate_id covers only
(estimate_id, package_type, estimate_unit_id), so a candidate that gains or
loses members — or changes a child's role, its pricing tier or its range —
keeps its ID while becoming a materially different package. Matching a stored
Sol decision on ID alone would silently attribute a judgement of one package to
another. The hash is the difference between "same ID" and "same package".
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from tools.comparison_common import sha256_canonical


def condition_summary(
    condition_id: str,
    *,
    conditions: Mapping[str, Mapping[str, Any]],
    evidence_by_condition: Mapping[str, Mapping[str, Any]],
    reviews_by_condition: Mapping[str, Mapping[str, Any]],
    dispositions_by_condition: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """The three upstream layers, separately labeled, never blended."""
    condition = conditions[condition_id]
    evidence = evidence_by_condition[condition_id]
    review = reviews_by_condition[condition_id]
    disposition = dispositions_by_condition[condition_id]
    return {
        "condition_id": condition_id,
        "catalog_item_id": condition["catalog_item_id"],
        "objective_evidence": {
            "distinct_photo_count": evidence["distinct_photo_count"],
            "distinct_view_count": evidence["distinct_view_count"],
        },
        "terra_verdict": {
            "verdict": review["verdict"],
            "rationale": review["rationale"],
        },
        "deterministic_disposition": {
            "disposition": disposition["disposition"],
            "reason_code": disposition["reason_code"],
        },
    }


def build_candidate_payload(
    candidate: Mapping[str, Any],
    *,
    work_items: Mapping[str, Any],
    conditions: Mapping[str, Any],
    evidence_by_condition: Mapping[str, Any],
    reviews_by_condition: Mapping[str, Any],
    dispositions_by_condition: Mapping[str, Any],
) -> Dict[str, Any]:
    """One candidate as Sol sees it: identity, economics, and child roles."""
    drivers = set(candidate["driver_work_item_ids"])
    children: List[Dict[str, Any]] = []
    for work_id in candidate["child_work_item_ids"]:
        work = work_items[work_id]
        children.append({
            "work_item_id": work_id,
            "role": "driver" if work_id in drivers else "support",
            "action_code": work["action_code"],
            "trade_bucket": work["trade_bucket"],
            "billable_unit_id": work["billable_unit_id"],
            "unit_count": work["unit_count"],
            "estimate_scope": work["estimate_scope"],
            "low": work["low"],
            "high": work["high"],
            "conditions": [
                condition_summary(
                    condition_id,
                    conditions=conditions,
                    evidence_by_condition=evidence_by_condition,
                    reviews_by_condition=reviews_by_condition,
                    dispositions_by_condition=dispositions_by_condition,
                )
                for condition_id in work["condition_ids"]
            ],
        })
    return {
        "package_candidate_id": candidate["package_candidate_id"],
        "package_type": candidate["package_type"],
        "package_category": candidate["package_category"],
        "package_level": candidate["package_level"],
        "room": candidate["room"],
        "estimate_unit_id": candidate["estimate_unit_id"],
        "strength": candidate["strength"],
        "pricing_tier": candidate["pricing_tier"],
        "proposed_treatment": candidate["proposed_treatment"],
        "allowance_low": candidate["low"],
        "allowance_high": candidate["high"],
        "display_only": candidate["display_only"],
        "contributing_candidate_ids": list(candidate["contributing_candidate_ids"]),
        "child_work_snapshots": children,
    }


def candidate_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Stable hash of one candidate payload, minus its own id.

    The id is excluded so the hash answers "is this the same package?" rather
    than "is this the same id?" — the two questions diverge exactly when
    membership, roles, tier or cost move under a stable id.
    """
    return sha256_canonical(
        {k: v for k, v in payload.items() if k != "package_candidate_id"}
    )


def candidate_payload_index(result: Mapping[str, Any]) -> Dict[str, Any]:
    """The by-id lookups build_candidate_payload needs, as kwargs.

    Works on any result carrying the standalone keys — the live standalone
    result, or a stored artifact read back off disk.
    """
    return {
        "conditions": {
            condition["condition_id"]: condition
            for condition in result["observed_conditions"]
        },
        "evidence_by_condition": {
            evidence["condition_id"]: evidence
            for evidence in result["evidence_facts"]
        },
        "reviews_by_condition": {
            review["condition_id"]: review
            for review in result["condition_reviews"]
        },
        "dispositions_by_condition": {
            disposition["condition_id"]: disposition
            for disposition in result["condition_dispositions"]
        },
        "work_items": {
            item["work_item_id"]: item for item in result["work_items"]
        },
    }


def candidate_payload_hashes(
    result: Mapping[str, Any],
    package_candidates: List[Dict[str, Any]],
) -> Dict[str, str]:
    """{package_candidate_id: payload hash} for one result's candidates."""
    index = candidate_payload_index(result)
    return {
        candidate["package_candidate_id"]: candidate_payload_sha256(
            build_candidate_payload(candidate, **index)
        )
        for candidate in package_candidates
    }
