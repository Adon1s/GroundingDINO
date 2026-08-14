"""Deterministic prefixed IDs for the renovation architecture contracts.

Every ID is ``f"{prefix}_{stable_hash_id(namespace, *parts, length=16)}"``
(sha256 over "|"-joined parts, tools/pipeline_common.py). Each hash input
starts with a namespace literal so two record types can never collide even
with identical remaining parts. Inputs are immutable IDs only — never names,
descriptions, or display text.

The estimate ID includes both the catalog file sha and the projection
fingerprint: same photos + same catalog + changed projection policy yields a
different estimate identity, which the Session 6 shadow comparison needs.
"""
from __future__ import annotations

from tools.pipeline_common import stable_hash_id


def _make(prefix: str, namespace: str, *parts: str) -> str:
    return f"{prefix}_{stable_hash_id(namespace, *parts, length=16)}"


def make_estimate_id(
    *, property_key: str, source_run_id: str,
    catalog_sha256: str, projection_fingerprint: str,
) -> str:
    return _make(
        "rea1", "renovation_estimate",
        property_key, source_run_id, catalog_sha256, projection_fingerprint,
    )


def make_condition_id(
    *, estimate_id: str, catalog_item_id: str, estimate_unit_id: str
) -> str:
    return _make("oc1", "condition", estimate_id, catalog_item_id, estimate_unit_id)


def make_evidence_id(*, estimate_id: str, condition_id: str) -> str:
    return _make("ev1", "evidence", estimate_id, condition_id)


def make_review_id(*, estimate_id: str, condition_id: str) -> str:
    return _make("cr1", "review", estimate_id, condition_id)


def make_disposition_id(*, estimate_id: str, condition_id: str) -> str:
    return _make("cd1", "disposition", estimate_id, condition_id)


def make_terra_call_id(
    *, estimate_id: str, estimate_unit_id: str, request_fingerprint: str
) -> str:
    return _make(
        "tc1", "terra_call", estimate_id, estimate_unit_id, request_fingerprint
    )


def make_work_item_id(
    *, estimate_id: str, catalog_item_id: str, estimate_unit_id: str, action_code: str
) -> str:
    return _make(
        "wk1", "work", estimate_id, catalog_item_id, estimate_unit_id, action_code
    )


def make_package_candidate_id(
    *, estimate_id: str, package_type: str, room_key: str
) -> str:
    return _make("pk1", "package", estimate_id, package_type, room_key)


def make_package_decision_id(*, estimate_id: str, package_candidate_id: str) -> str:
    return _make("pd1", "decision", estimate_id, package_candidate_id)


def make_ledger_entry_id(*, estimate_id: str, work_item_id: str) -> str:
    return _make("cl1", "ledger", estimate_id, work_item_id)
