"""Contract, ID, and validator tests for tools/renovation_architecture.

Covers: deterministic prefixed IDs, frozen JSON-safe dataclasses, strict
envelope validation per state (including the Session 2
condition_review_complete state), and the complete-result invariants (frozen
so Sessions 3-5 build against them; production use begins Session 5).

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_contracts.py -q
"""
import copy
import dataclasses
import json
import re

import pytest

from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    DEDUP_SUPPRESSION_REASON,
    ENVELOPE_SCHEMA_VERSION,
    ESTIMATE_SCOPES,
    EVIDENCE_DEDUP_POLICY_VERSION,
    CONDITION_DISPOSITION_POLICY_VERSION,
    EstimateProvenance,
    ObservedCondition,
    PACKAGE_CANDIDATE_POLICY_VERSION,
    PACKAGE_CATEGORIES,
    PACKAGE_LEVELS,
    PACKAGE_ROOMS,
    PACKAGE_STRENGTHS,
    PACKAGE_TYPES,
    POLICY_VERSIONS,
    PROJECTION_VERSION,
    RenovationEstimateEnvelope,
    SCAFFOLD_REASON,
    SOL_REVIEW_PROMPT_VERSION,
    STANDALONE_PRICING_POLICY_VERSION,
    WHOLE_HOME_PACKAGE_TYPE,
    WHOLE_HOME_PRICING_PROFILE,
    WHOLE_HOME_PRICING_TIER,
    WHOLE_HOME_UNIT_ID,
    WORK_DEDUP_POLICY_VERSION,
)
from tools.renovation_architecture.ids import (
    make_combine_group_id,
    make_condition_id,
    make_disposition_id,
    make_estimate_id,
    make_evidence_id,
    make_ledger_entry_id,
    make_merged_work_item_id,
    make_package_application_id,
    make_package_candidate_id,
    make_package_decision_id,
    make_review_id,
    make_sol_call_id,
    make_terra_call_id,
    make_work_dedup_collision_id,
    make_work_item_id,
)
from tools.renovation_architecture.validators import (
    package_review_snapshot_hashes,
    validate_complete_result,
    validate_condition_review_result,
    validate_envelope,
    validate_package_review_result,
    validate_standalone_estimate_result,
)

EST_ID = make_estimate_id(
    property_key="prop_1",
    source_run_id="run_1",
    catalog_sha256="a" * 64,
    projection_fingerprint="b" * 64,
)
FP = "c" * 64  # a fixed Terra request fingerprint shared by the builders


def _assert_error(res, fragment):
    assert not res.ok, f"expected a validation error containing {fragment!r}"
    assert any(fragment in error for error in res.errors), (
        f"no error contains {fragment!r}:\n" + "\n".join(res.errors)
    )


# ── builders ─────────────────────────────────────────────────────────────────

def _provenance(**over):
    base = EstimateProvenance(
        schema_version=CONTRACTS_SCHEMA_VERSION,
        architecture_mode="shadow",
        contracts_schema_version=CONTRACTS_SCHEMA_VERSION,
        projection_version=PROJECTION_VERSION,
        catalog_version="3.1",
        catalog_ontology_version="observation-kind-v2",
        catalog_sha256="a" * 64,
        projection_fingerprint="b" * 64,
        kind_ontology_selector="observation_kind_v2",
        policy_versions=dict(POLICY_VERSIONS),
        property_key="prop_1",
        source_run_id="run_1",
        source_artifact="prop_1/run_1/photo_intel.json",
        created_at="2026-08-14T00:00:00Z",
    ).to_dict()
    base.update(over)
    return base


def _scaffold_envelope(**over):
    base = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "estimate_id": EST_ID,
        "state": "scaffold",
        "reason": SCAFFOLD_REASON,
        "error_detail": None,
        "provenance": _provenance(),
        "result": None,
    }
    base.update(over)
    return base


def _condition(catalog_item_id, unit_id):
    scope_key = f"catalog:{catalog_item_id}|scene_group:kitchen|room:{unit_id}"
    return {
        "condition_id": make_condition_id(
            estimate_id=EST_ID, catalog_item_id=catalog_item_id, estimate_unit_id=unit_id
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "catalog_item_id": catalog_item_id,
        "catalog_kind": "defect",
        "scope_key": scope_key,
        "estimate_unit_id": unit_id,
        "room_surrogate_id": unit_id,
        "scene_group": "kitchen",
        "issue_ids": [f"issue_{catalog_item_id}"],
        "identity_ambiguous": False,
        "source_room_surrogate_ids": [unit_id],
        "source_scope_keys": [scope_key],
        "unit_resolution_source": "photo_estimate_unit",
        "unit_resolution_reason": "single_surrogate",
        "opening_instance_hints": [],
    }


# disposition -> (reason_code, terminal_route) defaults consistent with the
# condition_disposition_v1 table for a supported verdict.
_DISPOSITION_DEFAULTS = {
    "accepted_for_work": ("route_work", "work"),
    "inspection": ("route_inspection", "inspection"),
    "no_action": ("route_no_action", "no_action"),
    "excluded": ("route_excluded_generic", "excluded_generic"),
    "withheld": ("insufficient_distinct_views", "work"),
}


def _lattice(condition, verdict="supported", disposition="accepted_for_work"):
    """The per-condition evidence/review/disposition triple."""
    cid = condition["condition_id"]
    evidence_id = make_evidence_id(estimate_id=EST_ID, condition_id=cid)
    review_id = make_review_id(estimate_id=EST_ID, condition_id=cid)
    reason_code, terminal_route = _DISPOSITION_DEFAULTS[disposition]
    return (
        {
            "evidence_id": evidence_id,
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "condition_id": cid,
            "photo_keys": ["img_001.jpg", "img_002.jpg"],
            "distinct_photo_count": 2,
            "distinct_view_count": 2,
            "duplicate_groups": [],
            "evidence_refs": [
                {
                    "issue_id": condition["issue_ids"][0],
                    "photo_key": "img_001.jpg",
                    "observation": "observed in photo",
                    "room_surrogate_id": condition["room_surrogate_id"],
                }
            ],
            "min_photo_evidence_required": None,
            "representative_photo_keys": ["img_001.jpg", "img_002.jpg"],
            "exact_duplicate_groups": [],
            "near_duplicate_groups": [],
            "dedup_policy_version": EVIDENCE_DEDUP_POLICY_VERSION,
        },
        {
            "review_id": review_id,
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "condition_id": cid,
            "verdict": verdict,
            "rationale": "clearly visible",
            "model": "terra-test",
            "prompt_version": "v1",
            "terra_call_id": make_terra_call_id(
                estimate_id=EST_ID,
                estimate_unit_id=condition["estimate_unit_id"],
                request_fingerprint=FP,
            ),
            "request_fingerprint": FP,
            "provider": "openai",
        },
        {
            "disposition_id": make_disposition_id(estimate_id=EST_ID, condition_id=cid),
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "condition_id": cid,
            "review_id": review_id,
            "disposition": disposition,
            "reason_code": reason_code,
            "policy_version": CONDITION_DISPOSITION_POLICY_VERSION,
            "evidence_id": evidence_id,
            "terminal_route": terminal_route,
        },
    )


def _work_item(condition, action_code, low, high, **over):
    """A v3 source work item: singleton lineage, one billable physical unit."""
    base = {
        "work_item_id": make_work_item_id(
            estimate_id=EST_ID,
            catalog_item_id=condition["catalog_item_id"],
            billable_unit_id=condition["estimate_unit_id"],
            action_code=action_code,
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "condition_ids": [condition["condition_id"]],
        "catalog_item_ids": [condition["catalog_item_id"]],
        "source_estimate_unit_ids": [condition["estimate_unit_id"]],
        "billable_unit_id": condition["estimate_unit_id"],
        "action_code": action_code,
        "action_sources": ["work_item_code"],
        "trade_bucket": "kitchen_cabinets_counters",
        "unit_policy": "per_kitchen",
        "unit_count": 1,
        "pricing_modes": ["catalog_allowance"],
        "identity_ambiguous": False,
        "estimate_scope": "marketability_rehab",
        "estimate_scope_reason": "catalog_estimate_scope",
        "low": low,
        "high": high,
        "status": "active",
        "reason_code": None,
    }
    base.update(over)
    return base


def _ledger_entry(work_item, representation, package_id=None, low=0, high=0,
                  reason_code=None):
    if reason_code is None:
        reason_code = (
            "absorbed_by_approved_package"
            if representation == "absorbed_by_package" else "no_covering_package"
        )
    return {
        "entry_id": make_ledger_entry_id(
            estimate_id=EST_ID, work_item_id=work_item["work_item_id"]
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "work_item_id": work_item["work_item_id"],
        "representation": representation,
        "package_id": package_id,
        "reason_code": reason_code,
        "low": low,
        "high": high,
    }


def _find_entry(result, representation):
    """The first coverage-ledger entry with the given representation."""
    return next(
        entry for entry in result["coverage_ledger"]
        if entry["representation"] == representation
    )


SOL_FP = "d" * 64  # a fixed Sol request fingerprint shared by the builders
SOL_CALL_ID = make_sol_call_id(estimate_id=EST_ID, request_fingerprint=SOL_FP)


def _candidate(package_type, estimate_unit_id, work_items, **over):
    """A v4 room candidate: every child a driver, floored against the child
    standalone sum."""
    child_ids = sorted(item["work_item_id"] for item in work_items)
    child_low = sum(item["low"] for item in work_items)
    child_high = sum(item["high"] for item in work_items)
    unfloored_low = over.pop("unfloored_low", child_low)
    unfloored_high = over.pop("unfloored_high", child_high)
    low = max(unfloored_low, child_low)
    high = max(unfloored_high, child_high)
    base = {
        "package_candidate_id": make_package_candidate_id(
            estimate_id=EST_ID, package_type=package_type,
            estimate_unit_id=estimate_unit_id,
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "package_type": package_type,
        "package_category": package_type.rsplit("_", 1)[1],
        "package_level": "room",
        "room": package_type.split("_", 1)[0],
        "estimate_unit_id": estimate_unit_id,
        "child_work_item_ids": child_ids,
        "driver_work_item_ids": child_ids,
        "support_work_item_ids": [],
        "strength": "moderate",
        "pricing_profile": f"{package_type.split('_', 1)[0]}_refresh",
        "pricing_tier": "refresh",
        "absorption_scope": {
            "family": package_type.split("_", 1)[0],
            "groups": [package_type.split("_", 1)[0]],
            "trade_buckets": [],
            "components": [],
        },
        "proposed_treatment": "package_driver",
        "unfloored_low": unfloored_low,
        "unfloored_high": unfloored_high,
        "cost_floor_applied": (low, high) != (unfloored_low, unfloored_high),
        "low": low,
        "high": high,
        "display_only": False,
        "contributing_candidate_ids": [],
    }
    base.update(over)
    return base


def _decision(candidate, decision="approve", **over):
    base = {
        "decision_id": make_package_decision_id(
            estimate_id=EST_ID,
            package_candidate_id=candidate["package_candidate_id"],
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "package_candidate_id": candidate["package_candidate_id"],
        "decision": decision,
        "combine_with": [],
        "split_groups": [],
        "rationale": "coherent scope",
        "model": "sol-test",
        "prompt_version": SOL_REVIEW_PROMPT_VERSION,
        "sol_call_id": SOL_CALL_ID,
        "request_fingerprint": SOL_FP,
        "provider": "openai",
    }
    base.update(over)
    return base


def _sol_call(candidates, *, input_tokens=900, output_tokens=120, **over):
    base = {
        "call_id": SOL_CALL_ID,
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "package_candidate_ids": sorted(
            candidate["package_candidate_id"] for candidate in candidates
        ),
        "request_fingerprint": SOL_FP,
        "provider": "openai",
        "model": "sol-test",
        "prompt_version": SOL_REVIEW_PROMPT_VERSION,
        "usage_source": "provider",
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    base.update(over)
    return base


def _complete_result():
    """The Session 4 package-review fixture run through the real Session 5
    reconciliation, so IDs, reason codes, ledger, totals, audit, and
    observability are exactly the deterministic policy's output (the v5 gate
    enforces recompute equality — hand-built approximations cannot pass)."""
    from tools.renovation_architecture.reconciliation import build_complete_result

    return build_complete_result(_package_review_result(), estimate_id=EST_ID)


_TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
    "budget_debited_tokens",
)


def _terra_call(condition, *, input_tokens, output_tokens):
    total = input_tokens + output_tokens
    return {
        "call_id": make_terra_call_id(
            estimate_id=EST_ID,
            estimate_unit_id=condition["estimate_unit_id"],
            request_fingerprint=FP,
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "estimate_unit_id": condition["estimate_unit_id"],
        "condition_ids": [condition["condition_id"]],
        "request_fingerprint": FP,
        "provider": "openai",
        "model": "terra-test",
        "prompt_version": "v1",
        "usage_source": "provider",
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "budget_debited_tokens": total,
    }


def _unit_usage(call):
    usage = {
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "estimate_unit_id": call["estimate_unit_id"],
        "call_ids": [call["call_id"]],
    }
    for name in _TOKEN_FIELDS:
        usage[name] = call[name]
    return usage


def _review_result():
    """Two conditions in two estimate units, one Terra call per unit, exact
    call -> unit -> listing usage reconciliation."""
    c1 = _condition("outdated_or_damaged_cabinets", "kitchen_primary")
    c2 = _condition("bathroom_vanity_worn", "bathroom_1")
    ev1, rv1, dp1 = _lattice(c1)
    ev2, rv2, dp2 = _lattice(c2)
    t1 = _terra_call(c1, input_tokens=1000, output_tokens=200)
    t2 = _terra_call(c2, input_tokens=800, output_tokens=150)
    u1, u2 = _unit_usage(t1), _unit_usage(t2)
    listing = {"schema_version": CONTRACTS_SCHEMA_VERSION, "call_count": 2}
    for name in _TOKEN_FIELDS:
        listing[name] = u1[name] + u2[name]
    return {
        "observed_conditions": [c1, c2],
        "evidence_facts": [ev1, ev2],
        "condition_reviews": [rv1, rv2],
        "condition_dispositions": [dp1, dp2],
        "terra_calls": [t1, t2],
        "terra_unit_usage": [u1, u2],
        "terra_listing_usage": listing,
    }


def _review_envelope(**over):
    base = _scaffold_envelope(
        state="condition_review_complete", reason=None, result=_review_result()
    )
    base.update(over)
    return base


def _standalone_estimate(work_items, factor=1.0, **over):
    """Exact totals recomputed from the active work items."""
    totals = {scope: [0, 0] for scope in sorted(ESTIMATE_SCOPES)}
    for item in work_items:
        if item["status"] != "active":
            continue
        totals[item["estimate_scope"]][0] += item["low"]
        totals[item["estimate_scope"]][1] += item["high"]
    base = {
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "currency": "USD",
        "pricing_policy_version": STANDALONE_PRICING_POLICY_VERSION,
        "property_cost_factor": factor,
        "property_cost_factor_audit": {"reasons": []},
        "totals_by_estimate_scope": {
            scope: {"low": low, "high": high}
            for scope, (low, high) in totals.items()
        },
        "headline": {
            "low": sum(low for low, _ in totals.values()),
            "high": sum(high for _, high in totals.values()),
        },
    }
    base.update(over)
    return base


def _standalone_result(**over):
    """The Session 2 review result plus one active work item per accepted
    condition, no collisions, exact totals."""
    base = _review_result()
    c1, c2 = base["observed_conditions"]
    items = sorted(
        [
            _work_item(c1, "CABINETS_REPLACE", 1000, 3000),
            _work_item(c2, "VANITY_REPLACE", 500, 1500),
        ],
        key=lambda item: item["work_item_id"],
    )
    base["work_items"] = items
    base["work_dedup_collisions"] = []
    base["standalone_estimate"] = _standalone_estimate(items)
    base.update(over)
    return base


def _standalone_envelope(**over):
    base = _scaffold_envelope(
        state="standalone_estimate_complete", reason=None,
        result=_standalone_result(),
    )
    base.update(over)
    return base


# ── IDs ──────────────────────────────────────────────────────────────────────

class TestIds:
    def test_ids_are_deterministic(self):
        again = make_estimate_id(
            property_key="prop_1",
            source_run_id="run_1",
            catalog_sha256="a" * 64,
            projection_fingerprint="b" * 64,
        )
        assert again == EST_ID

    def test_id_shapes(self):
        pairs = [
            (EST_ID, "rea1"),
            (make_condition_id(estimate_id=EST_ID, catalog_item_id="x", estimate_unit_id="u"), "oc1"),
            (make_evidence_id(estimate_id=EST_ID, condition_id="c"), "ev1"),
            (make_review_id(estimate_id=EST_ID, condition_id="c"), "cr1"),
            (make_disposition_id(estimate_id=EST_ID, condition_id="c"), "cd1"),
            (make_work_item_id(estimate_id=EST_ID, catalog_item_id="x", billable_unit_id="u", action_code="a"), "wk1"),
            (make_merged_work_item_id(estimate_id=EST_ID, action_code="a", trade_bucket="t", unit_policy="per_scope", billable_unit_id="u"), "wk1"),
            (make_work_dedup_collision_id(estimate_id=EST_ID, active_work_item_id="w"), "wdc1"),
            (make_package_candidate_id(estimate_id=EST_ID, package_type="t", estimate_unit_id="u"), "pk1"),
            (make_package_decision_id(estimate_id=EST_ID, package_candidate_id="p"), "pd1"),
            (make_sol_call_id(estimate_id=EST_ID, request_fingerprint="f" * 64), "sc1"),
            (make_ledger_entry_id(estimate_id=EST_ID, work_item_id="w"), "cl1"),
            (make_package_application_id(estimate_id=EST_ID, package_candidate_id="p"), "pa1"),
            (make_combine_group_id(estimate_id=EST_ID, member_candidate_ids=["b", "a"]), "cg1"),
        ]
        for value, prefix in pairs:
            assert re.fullmatch(rf"{prefix}_[0-9a-f]{{16}}", value), value

    def test_namespace_separates_record_types(self):
        """Identical remaining parts must never collide across record types."""
        evidence = make_evidence_id(estimate_id=EST_ID, condition_id="c")
        review = make_review_id(estimate_id=EST_ID, condition_id="c")
        disposition = make_disposition_id(estimate_id=EST_ID, condition_id="c")
        assert len({evidence[4:], review[4:], disposition[4:]}) == 3

    def test_merged_and_source_work_ids_cannot_collide(self):
        """The merged recipe hashes a distinct namespace, so even identical
        remaining parts produce a different id than any source recipe."""
        source = make_work_item_id(
            estimate_id=EST_ID, catalog_item_id="a",
            billable_unit_id="u", action_code="code",
        )
        merged = make_merged_work_item_id(
            estimate_id=EST_ID, action_code="a",
            trade_bucket="u", unit_policy="code", billable_unit_id="",
        )
        assert source != merged

    def test_estimate_id_varies_with_each_part(self):
        base = dict(
            property_key="prop_1", source_run_id="run_1",
            catalog_sha256="a" * 64, projection_fingerprint="b" * 64,
        )
        for key, changed in (
            ("property_key", "prop_2"),
            ("source_run_id", "run_2"),
            ("catalog_sha256", "c" * 64),
            ("projection_fingerprint", "d" * 64),
        ):
            assert make_estimate_id(**{**base, key: changed}) != EST_ID


# ── dataclasses ──────────────────────────────────────────────────────────────

class TestContracts:
    def test_contracts_are_frozen(self):
        condition = ObservedCondition(
            condition_id="oc1_" + "0" * 16,
            schema_version=CONTRACTS_SCHEMA_VERSION,
            catalog_item_id="x",
            catalog_kind="defect",
            scope_key="k",
            estimate_unit_id="u",
            room_surrogate_id="r",
            scene_group="kitchen",
            issue_ids=("i",),
            identity_ambiguous=False,
            source_room_surrogate_ids=("r",),
            source_scope_keys=("k",),
            unit_resolution_source="photo_estimate_unit",
            unit_resolution_reason="single_surrogate",
            opening_instance_hints=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            condition.catalog_item_id = "y"

    def test_to_dict_is_json_safe(self):
        envelope = _scaffold_envelope()
        text = json.dumps(envelope)
        assert isinstance(json.loads(text), dict)
        # tuples became lists all the way down
        assert isinstance(envelope["provenance"]["policy_versions"], dict)

    def test_scaffold_envelope_via_dataclass_round_trips(self):
        envelope = RenovationEstimateEnvelope(
            schema_version=ENVELOPE_SCHEMA_VERSION,
            estimate_id=EST_ID,
            state="scaffold",
            reason=SCAFFOLD_REASON,
            error_detail=None,
            provenance=EstimateProvenance(**_provenance()),
            result=None,
        ).to_dict()
        assert validate_envelope(envelope).ok


# ── scaffold envelope validation ─────────────────────────────────────────────

class TestScaffoldEnvelope:
    def test_scaffold_envelope_validates(self):
        assert validate_envelope(_scaffold_envelope()).ok

    def test_scaffold_result_must_be_null(self):
        res = validate_envelope(_scaffold_envelope(result={"observed_conditions": []}))
        _assert_error(res, "scaffold result must be null")

    def test_scaffold_reason_is_pinned(self):
        _assert_error(
            validate_envelope(_scaffold_envelope(reason="not_done_yet")),
            SCAFFOLD_REASON,
        )
        _assert_error(validate_envelope(_scaffold_envelope(reason=None)), SCAFFOLD_REASON)

    def test_error_detail_null_outside_failed(self):
        res = validate_envelope(_scaffold_envelope(error_detail="boom"))
        _assert_error(res, "error_detail must be null")

    def test_unknown_envelope_field_rejected(self):
        _assert_error(
            validate_envelope(_scaffold_envelope(extra_field=1)), "unknown field"
        )

    def test_estimate_id_shape_enforced(self):
        _assert_error(
            validate_envelope(_scaffold_envelope(estimate_id="rea1_nothex")),
            "estimate_id",
        )

    def test_invalid_state_rejected(self):
        _assert_error(validate_envelope(_scaffold_envelope(state="draft")), "state")

    @pytest.mark.parametrize(
        "field",
        ["catalog_version", "catalog_ontology_version", "catalog_sha256",
         "projection_fingerprint", "kind_ontology_selector", "property_key",
         "source_run_id", "source_artifact", "created_at"],
    )
    def test_scaffold_provenance_fields_required(self, field):
        envelope = _scaffold_envelope(provenance=_provenance(**{field: None}))
        _assert_error(validate_envelope(envelope), field)

    def test_provenance_pins_both_ontology_spellings(self):
        """Catalog stamp is hyphenated; the env selector is underscored."""
        wrong_catalog = _provenance(catalog_ontology_version="observation_kind_v2")
        _assert_error(
            validate_envelope(_scaffold_envelope(provenance=wrong_catalog)),
            "observation-kind-v2",
        )
        wrong_selector = _provenance(kind_ontology_selector="observation-kind-v2")
        _assert_error(
            validate_envelope(_scaffold_envelope(provenance=wrong_selector)),
            "observation_kind_v2",
        )

    @pytest.mark.parametrize(
        "path",
        ["C:/abs/photo_intel.json", "/abs/photo_intel.json",
         "prop/../other/photo_intel.json", "prop\\run\\photo_intel.json"],
    )
    def test_source_artifact_must_be_relative(self, path):
        envelope = _scaffold_envelope(provenance=_provenance(source_artifact=path))
        _assert_error(validate_envelope(envelope), "source_artifact")

    def test_bad_architecture_mode_rejected(self):
        envelope = _scaffold_envelope(provenance=_provenance(architecture_mode="hybrid"))
        _assert_error(validate_envelope(envelope), "architecture_mode")

    def test_mixed_schema_version_rejected(self):
        envelope = _scaffold_envelope(provenance=_provenance(schema_version=1))
        _assert_error(validate_envelope(envelope), "schema_version")

    def test_policy_versions_keys_are_exact(self):
        provenance = _provenance(
            policy_versions={"terminal_route_policy": "v1", "surprise": "v9"}
        )
        _assert_error(
            validate_envelope(_scaffold_envelope(provenance=provenance)),
            "policy_versions",
        )

    def test_bad_created_at_rejected(self):
        envelope = _scaffold_envelope(provenance=_provenance(created_at="yesterday"))
        _assert_error(validate_envelope(envelope), "created_at")


# ── failed envelope validation ───────────────────────────────────────────────

class TestFailedEnvelope:
    def _failed(self, **over):
        provenance = _provenance(
            catalog_version=None,
            catalog_ontology_version=None,
            catalog_sha256=None,
            projection_fingerprint=None,
            kind_ontology_selector=None,
        )
        base = _scaffold_envelope(
            state="failed",
            reason="runtime_not_initialized",
            error_detail=None,
            provenance=provenance,
        )
        base.update(over)
        return base

    def test_failed_with_null_catalog_provenance_validates(self):
        assert validate_envelope(self._failed()).ok

    def test_failed_allows_error_detail(self):
        assert validate_envelope(self._failed(error_detail="stack trace")).ok

    def test_failed_requires_reason(self):
        _assert_error(validate_envelope(self._failed(reason=None)), "failed reason")
        _assert_error(validate_envelope(self._failed(reason="")), "failed reason")

    def test_failed_result_must_be_null(self):
        _assert_error(
            validate_envelope(self._failed(result={})), "failed result must be null"
        )

    def test_null_catalog_provenance_fails_on_scaffold(self):
        """The nullable group is a failed-state concession only."""
        envelope = _scaffold_envelope(provenance=_provenance(catalog_sha256=None))
        _assert_error(validate_envelope(envelope), "catalog_sha256")


# ── condition_review_complete envelope (Session 2) ───────────────────────────

class TestConditionReviewEnvelope:
    def test_happy_path_validates(self):
        assert validate_envelope(_review_envelope()).ok

    def test_empty_review_result_validates(self):
        """A listing with no product-lane issues still completes review."""
        empty = {
            "observed_conditions": [], "evidence_facts": [],
            "condition_reviews": [], "condition_dispositions": [],
            "terra_calls": [], "terra_unit_usage": [],
            "terra_listing_usage": {
                "schema_version": CONTRACTS_SCHEMA_VERSION, "call_count": 0,
                "input_tokens": 0, "cached_input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0,
                "budget_debited_tokens": 0,
            },
        }
        assert validate_envelope(_review_envelope(result=empty)).ok

    def test_reason_must_be_null(self):
        _assert_error(
            validate_envelope(_review_envelope(reason="done")),
            "reason must be null",
        )

    def test_result_must_be_an_object(self):
        _assert_error(
            validate_envelope(_review_envelope(result=None)),
            "result must be an object",
        )

    def test_later_session_layers_are_rejected(self):
        """The Session 2 result may not smuggle work/package/totals keys."""
        result = _review_result()
        result["work_items"] = []
        _assert_error(
            validate_condition_review_result(result, estimate_id=EST_ID),
            "unknown field",
        )

    def test_unit_without_terra_call_rejected(self):
        result = _review_result()
        result["terra_calls"].pop()
        _assert_error(
            validate_condition_review_result(result, estimate_id=EST_ID),
            "has no Terra call",
        )

    def test_listing_rollup_must_reconcile(self):
        result = _review_result()
        result["terra_listing_usage"]["total_tokens"] += 1
        _assert_error(
            validate_condition_review_result(result, estimate_id=EST_ID),
            "from the unit rollups",
        )

    def test_cached_tokens_cannot_exceed_input(self):
        result = _review_result()
        result["terra_calls"][0]["cached_input_tokens"] = (
            result["terra_calls"][0]["input_tokens"] + 1
        )
        _assert_error(
            validate_condition_review_result(result, estimate_id=EST_ID),
            "cached_input_tokens",
        )

    def test_review_must_match_its_call_fingerprint(self):
        result = _review_result()
        result["condition_reviews"][0]["request_fingerprint"] = "d" * 64
        _assert_error(
            validate_condition_review_result(result, estimate_id=EST_ID),
            "does not match",
        )


# ── standalone_estimate_complete envelope (Session 3) ────────────────────────

class TestStandaloneEnvelope:
    def test_happy_path_validates(self):
        assert validate_envelope(_standalone_envelope()).ok

    def test_empty_standalone_result_validates(self):
        """No accepted work still completes the standalone estimate: empty
        work lanes and all-zero scope buckets."""
        empty_review = {
            "observed_conditions": [], "evidence_facts": [],
            "condition_reviews": [], "condition_dispositions": [],
            "terra_calls": [], "terra_unit_usage": [],
            "terra_listing_usage": {
                "schema_version": CONTRACTS_SCHEMA_VERSION, "call_count": 0,
                "input_tokens": 0, "cached_input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0,
                "budget_debited_tokens": 0,
            },
        }
        result = {
            **empty_review,
            "work_items": [],
            "work_dedup_collisions": [],
            "standalone_estimate": _standalone_estimate([]),
        }
        assert validate_envelope(_standalone_envelope(result=result)).ok

    def test_reason_must_be_null(self):
        _assert_error(
            validate_envelope(_standalone_envelope(reason="done")),
            "reason must be null",
        )

    def test_later_session_layers_are_rejected(self):
        """The Session 3 result may not smuggle package/ledger/totals keys."""
        for key in ("package_candidates", "package_decisions",
                    "coverage_ledger", "totals"):
            result = _standalone_result()
            result[key] = []
            _assert_error(
                validate_standalone_estimate_result(result, estimate_id=EST_ID),
                "unknown field",
            )

    def test_review_subset_still_runs_the_frozen_gate(self):
        """Breaking a Session 2 invariant inside a Session 3 result is caught
        by the delegated review validator, verbatim."""
        result = _standalone_result()
        result["terra_listing_usage"]["total_tokens"] += 1
        _assert_error(
            validate_standalone_estimate_result(result, estimate_id=EST_ID),
            "from the unit rollups",
        )

    def test_policy_versions_carry_the_session_5_policies(self):
        assert set(POLICY_VERSIONS) == {
            "terminal_route_policy", "condition_disposition_policy",
            "evidence_dedup_policy", "terra_review_prompt",
            "work_derivation_policy", "work_dedup_policy",
            "standalone_pricing_policy", "package_candidate_policy",
            "sol_review_prompt", "package_application_policy",
            "coverage_reconciliation_policy",
        }
        assert POLICY_VERSIONS["work_derivation_policy"] == "work_derivation_v1"
        assert POLICY_VERSIONS["work_dedup_policy"] == "work_dedup_max_envelope_v1"
        assert POLICY_VERSIONS["standalone_pricing_policy"] == "standalone_pricing_v1"
        assert POLICY_VERSIONS["package_candidate_policy"] == PACKAGE_CANDIDATE_POLICY_VERSION
        assert POLICY_VERSIONS["sol_review_prompt"] == SOL_REVIEW_PROMPT_VERSION
        assert POLICY_VERSIONS["package_application_policy"] == "package_application_v1"
        assert POLICY_VERSIONS["coverage_reconciliation_policy"] == "coverage_reconciliation_v1"

    def test_estimate_scopes_pin_the_estimator_vocabulary(self):
        """ESTIMATE_SCOPES must never drift from tools/estimate_scope.py."""
        from tools.estimate_scope import VALID_ESTIMATE_SCOPES

        assert ESTIMATE_SCOPES == frozenset(VALID_ESTIMATE_SCOPES)

    @pytest.mark.parametrize(
        "collection", ["work_items", "work_dedup_collisions"]
    )
    def test_unknown_field_rejected_per_new_record(self, collection):
        result = _standalone_result()
        collision = {
            "collision_id": make_work_dedup_collision_id(
                estimate_id=EST_ID, active_work_item_id="wk1_" + "0" * 16
            ),
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "action_code": "CABINETS_REPLACE",
            "trade_bucket": "kitchen_cabinets_counters",
            "unit_policy": "per_kitchen",
            "billable_unit_id": "kitchen_primary",
            "active_work_item_id": "wk1_" + "0" * 16,
            "suppressed_work_item_ids": ["wk1_" + "1" * 16, "wk1_" + "2" * 16],
            "policy_version": WORK_DEDUP_POLICY_VERSION,
        }
        result["work_dedup_collisions"] = [collision]
        result[collection][0]["surprise"] = True
        _assert_error(
            validate_standalone_estimate_result(result, estimate_id=EST_ID),
            "unknown field",
        )

    def test_standalone_estimate_unknown_field_rejected(self):
        result = _standalone_result()
        result["standalone_estimate"]["surprise"] = True
        _assert_error(
            validate_standalone_estimate_result(result, estimate_id=EST_ID),
            "unknown field",
        )

    def test_suppressed_reason_code_is_pinned(self):
        result = _standalone_result()
        result["work_items"][0]["status"] = "suppressed"
        result["work_items"][0]["reason_code"] = "some_other_reason"
        _assert_error(
            validate_standalone_estimate_result(result, estimate_id=EST_ID),
            DEDUP_SUPPRESSION_REASON,
        )

    def test_active_reason_code_must_be_null(self):
        result = _standalone_result()
        result["work_items"][0]["reason_code"] = DEDUP_SUPPRESSION_REASON
        _assert_error(
            validate_standalone_estimate_result(result, estimate_id=EST_ID),
            "must be null",
        )


# ── package_review_complete envelope (Session 4) ─────────────────────────────

def _snapshots(base, candidates):
    return {
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        **package_review_snapshot_hashes(
            {**base, "package_candidates": candidates}
        ),
    }


def _package_review_result(**over):
    """The Session 3 standalone result plus one approved single-child
    candidate, its decision, one Sol call, reconciled usage, and snapshots."""
    base = _standalone_result()
    p1 = _candidate(
        "kitchen_modernization", "kitchen_primary", [base["work_items"][0]],
        package_category="modernization",
    )
    d1 = _decision(p1)
    call = _sol_call([p1])
    result = {
        **base,
        "package_candidates": [p1],
        "package_decisions": [d1],
        "sol_calls": [call],
        "sol_listing_usage": {
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "call_count": 1,
            "input_tokens": call["input_tokens"],
            "cached_input_tokens": call["cached_input_tokens"],
            "output_tokens": call["output_tokens"],
            "total_tokens": call["total_tokens"],
        },
        "package_review_snapshots": _snapshots(base, [p1]),
    }
    result.update(over)
    return result


def _package_review_envelope(**over):
    base = _scaffold_envelope(
        state="package_review_complete", reason=None,
        result=_package_review_result(),
    )
    base.update(over)
    return base


class TestPackageReviewEnvelope:
    def test_happy_path_validates(self):
        res = validate_envelope(_package_review_envelope())
        assert res.ok, res.errors

    def test_empty_package_review_validates(self):
        """Zero candidates -> zero decisions, zero calls, zero usage; the
        snapshots still fingerprint the (empty) sections."""
        base = _standalone_result()
        result = {
            **base,
            "package_candidates": [],
            "package_decisions": [],
            "sol_calls": [],
            "sol_listing_usage": {
                "schema_version": CONTRACTS_SCHEMA_VERSION, "call_count": 0,
                "input_tokens": 0, "cached_input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0,
            },
            "package_review_snapshots": _snapshots(base, []),
        }
        res = validate_envelope(_package_review_envelope(result=result))
        assert res.ok, res.errors

    def test_reason_must_be_null(self):
        _assert_error(
            validate_envelope(_package_review_envelope(reason="done")),
            "reason must be null",
        )

    def test_later_session_layers_are_rejected(self):
        """The Session 4 result may not smuggle Session 5 keys."""
        for key in ("package_applications", "coverage_ledger",
                    "reconciliation_audit", "observability", "totals"):
            result = _package_review_result(**{key: []})
            _assert_error(
                validate_package_review_result(result, estimate_id=EST_ID),
                "unknown field",
            )

    def test_standalone_subset_still_runs_the_frozen_gate(self):
        result = _package_review_result()
        result["terra_listing_usage"]["total_tokens"] += 1
        _assert_error(
            validate_package_review_result(result, estimate_id=EST_ID),
            "from the unit rollups",
        )

    def test_vocabularies_pin_the_legacy_package_constants(self):
        """The import-light contract vocabularies must never drift from
        tools/rehab_packages.py (the UNIT_POLICIES pattern)."""
        from tools import rehab_packages as rp

        assert PACKAGE_TYPES == rp.VALID_PACKAGE_TYPES
        assert PACKAGE_CATEGORIES == rp.VALID_PACKAGE_CATEGORIES
        assert PACKAGE_ROOMS == rp.VALID_ROOMS
        assert PACKAGE_STRENGTHS == rp.VALID_EMITTED_PACKAGE_STRENGTHS
        assert PACKAGE_LEVELS < rp.VALID_PACKAGE_LEVELS
        assert WHOLE_HOME_PACKAGE_TYPE == rp.PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH
        assert WHOLE_HOME_UNIT_ID == "whole_home"

    def test_candidate_child_must_be_active_work(self):
        result = _package_review_result()
        candidate = result["package_candidates"][0]
        ghost = "wk1_" + "9" * 16
        candidate["child_work_item_ids"] = [ghost]
        candidate["driver_work_item_ids"] = [ghost]
        result["package_review_snapshots"] = _snapshots(
            result, result["package_candidates"]
        )
        _assert_error(
            validate_package_review_result(result, estimate_id=EST_ID),
            "not an ACTIVE work item",
        )

    def test_floored_range_is_recomputed(self):
        result = _package_review_result()
        result["package_candidates"][0]["low"] -= 1
        result["package_candidates"][0]["unfloored_low"] -= 1
        result["package_review_snapshots"] = _snapshots(
            result, result["package_candidates"]
        )
        _assert_error(
            validate_package_review_result(result, estimate_id=EST_ID),
            "max(tier spec, children standalone sum)",
        )

    def test_every_candidate_needs_exactly_one_decision(self):
        result = _package_review_result()
        result["package_decisions"] = []
        _assert_error(
            validate_package_review_result(result, estimate_id=EST_ID),
            "has no decision",
        )
        result = _package_review_result()
        extra = copy.deepcopy(result["package_decisions"][0])
        extra["decision_id"] = "pd1_" + "e" * 16
        result["package_decisions"].append(extra)
        _assert_error(
            validate_package_review_result(result, estimate_id=EST_ID),
            "more than one decision",
        )

    def test_decision_must_match_its_sol_call(self):
        result = _package_review_result()
        result["package_decisions"][0]["request_fingerprint"] = "e" * 64
        _assert_error(
            validate_package_review_result(result, estimate_id=EST_ID),
            "does not match its Sol call",
        )

    def test_sol_call_must_cover_exactly_the_candidates(self):
        result = _package_review_result()
        result["sol_calls"][0]["package_candidate_ids"] = ["pk1_" + "9" * 16]
        _assert_error(
            validate_package_review_result(result, estimate_id=EST_ID),
            "exactly the supplied candidates",
        )

    def test_listing_usage_must_reconcile(self):
        result = _package_review_result()
        result["sol_listing_usage"]["total_tokens"] += 1
        _assert_error(
            validate_package_review_result(result, estimate_id=EST_ID),
            "from the Sol calls",
        )

    def test_snapshot_tamper_is_detected(self):
        """Mutating work truth after the snapshots were taken must fail the
        immutability gate even when every other invariant still holds."""
        result = _package_review_result()
        result["work_items"][1]["estimate_scope_reason"] = "sol_touched_this"
        _assert_error(
            validate_package_review_result(result, estimate_id=EST_ID),
            "recomputed section hash",
        )

    def test_decision_layer_boundary(self):
        result = _package_review_result()
        result["package_decisions"][0]["price"] = 500
        _assert_error(
            validate_package_review_result(result, estimate_id=EST_ID),
            "layer boundary",
        )


# ── complete result invariants ───────────────────────────────────────────────

class TestCompleteResult:
    def test_happy_path_validates(self):
        result = _complete_result()
        assert validate_complete_result(result, estimate_id=EST_ID).ok
        envelope = _scaffold_envelope(state="complete", reason=None, result=result)
        assert validate_envelope(envelope).ok

    def test_unknown_result_key_rejected(self):
        result = _complete_result()
        result["bonus_lane"] = []
        _assert_error(validate_complete_result(result, estimate_id=EST_ID), "unknown field")

    def test_duplicate_condition_per_unit(self):
        result = _complete_result()
        clone = copy.deepcopy(result["observed_conditions"][0])
        clone["condition_id"] = "oc1_" + "f" * 16
        result["observed_conditions"].append(clone)
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "duplicates catalog condition",
        )

    def test_condition_missing_disposition(self):
        result = _complete_result()
        result["condition_dispositions"].pop()
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "has no terminal disposition",
        )

    @pytest.mark.parametrize(
        "collection", ["evidence_facts", "condition_reviews", "condition_dispositions"]
    )
    def test_orphan_lattice_record(self, collection):
        result = _complete_result()
        result[collection][0]["condition_id"] = "oc1_" + "9" * 16
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID), "unknown condition"
        )

    def test_disposition_must_cite_the_conditions_review(self):
        result = _complete_result()
        result["condition_dispositions"][0]["review_id"] = "cr1_" + "9" * 16
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "the condition's review",
        )

    @pytest.mark.parametrize("field", ["low", "confidence", "work"])
    def test_review_layer_boundary(self, field):
        result = _complete_result()
        result["condition_reviews"][0][field] = 100
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID), "layer boundary"
        )

    def test_decision_layer_boundary(self):
        result = _complete_result()
        result["package_decisions"][0]["price"] = 500
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID), "layer boundary"
        )

    @pytest.mark.parametrize(
        "collection",
        ["observed_conditions", "evidence_facts", "condition_reviews",
         "condition_dispositions", "work_items", "package_candidates",
         "package_decisions", "package_applications", "coverage_ledger"],
    )
    def test_unknown_field_rejected_per_record(self, collection):
        result = _complete_result()
        result[collection][0]["surprise"] = True
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID), "unknown field"
        )

    def test_evidence_photo_count_must_match(self):
        result = _complete_result()
        result["evidence_facts"][0]["distinct_photo_count"] = 5
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "distinct_photo_count",
        )

    def test_duplicate_group_members_must_be_known_photos(self):
        result = _complete_result()
        result["evidence_facts"][0]["duplicate_groups"] = [["img_999.jpg"]]
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "outside photo_keys",
        )

    def test_work_item_citing_non_accepted_condition(self):
        # The full consistent inspection triple, so the Session 2 subset gate
        # passes and the work-lineage check is what fires.
        result = _complete_result()
        result["condition_dispositions"][1].update({
            "disposition": "inspection",
            "reason_code": "route_inspection",
            "terminal_route": "inspection",
        })
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "not accepted_for_work",
        )

    def test_accepted_condition_with_no_work_item(self):
        # A fully-reviewed extra condition (its own Terra call and reconciled
        # usage) that no work item references.
        result = _complete_result()
        c3 = _condition("cracked_tile", "bathroom_2")
        ev3, rv3, dp3 = _lattice(c3)
        t3 = _terra_call(c3, input_tokens=700, output_tokens=100)
        result["observed_conditions"].append(c3)
        result["evidence_facts"].append(ev3)
        result["condition_reviews"].append(rv3)
        result["condition_dispositions"].append(dp3)
        result["terra_calls"].append(t3)
        result["terra_unit_usage"].append(_unit_usage(t3))
        result["terra_listing_usage"]["call_count"] += 1
        for name in _TOKEN_FIELDS:
            result["terra_listing_usage"][name] += t3[name]
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "accepted scope cannot vanish",
        )

    def test_package_child_must_be_active(self):
        result = _complete_result()
        ghost = "wk1_" + "9" * 16
        result["package_candidates"][0]["child_work_item_ids"] = [ghost]
        result["package_candidates"][0]["driver_work_item_ids"] = [ghost]
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "not an ACTIVE work item",
        )

    def test_suppressed_work_is_caught_through_the_subset_gate(self):
        result = _complete_result()
        result["work_items"][0]["status"] = "suppressed"
        result["work_items"][0]["reason_code"] = DEDUP_SUPPRESSION_REASON
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID), "suppressed"
        )

    def test_candidate_allows_at_most_one_decision(self):
        result = _complete_result()
        extra = copy.deepcopy(result["package_decisions"][0])
        extra["decision_id"] = "pd1_" + "e" * 16
        result["package_decisions"].append(extra)
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "more than one decision",
        )

    def test_combine_with_cannot_cite_self(self):
        result = _complete_result()
        decision = result["package_decisions"][0]
        decision["combine_with"] = [decision["package_candidate_id"]]
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID), "cites itself"
        )

    def test_split_groups_must_partition_children(self):
        result = _complete_result()
        # Two well-formed groups whose union is NOT the candidate's children
        # (w2 is not a child): the per-record shape passes, the cross-record
        # partition check must fire.
        result["package_decisions"][0]["split_groups"] = sorted(
            (
                [result["work_items"][0]["work_item_id"]],
                [result["work_items"][1]["work_item_id"]],
            ),
            key=tuple,
        )
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "exactly partition",
        )

    def test_missing_ledger_entry(self):
        result = _complete_result()
        result["coverage_ledger"].pop()
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "has no ledger entry",
        )

    def test_duplicate_ledger_entry(self):
        result = _complete_result()
        extra = copy.deepcopy(result["coverage_ledger"][1])
        extra["entry_id"] = "cl1_" + "e" * 16
        result["coverage_ledger"].append(extra)
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "more than one ledger entry",
        )

    def test_billing_requires_an_approval_basis(self):
        """Flipping the decision to reject after application leaves an
        applied package without its approval basis."""
        result = _complete_result()
        result["package_decisions"][0]["decision"] = "reject"
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "without an approval basis",
        )

    def test_absorbed_entry_must_carry_zero_dollars(self):
        result = _complete_result()
        entry = _find_entry(result, "absorbed_by_package")
        entry["low"] = 1
        entry["high"] = 1
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID), "0/0"
        )

    def test_uncertain_package_children_stay_standalone(self):
        """Downgrading the decision and application while leaving the child's
        ledger entry absorbed must trip the standalone-fallback invariant."""
        result = _complete_result()
        result["package_decisions"][0]["decision"] = "uncertain"
        app = result["package_applications"][0]
        absorbed = app["absorbed_work_item_ids"]
        app.update({
            "status": "not_applied",
            "reason_code": "decision_uncertain",
            "absorbed_work_item_ids": [],
            "unabsorbed_child_work_item_ids": sorted(
                absorbed + app["unabsorbed_child_work_item_ids"]
            ),
            "effective_low": 0,
            "effective_high": 0,
        })
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "must remain standalone",
        )

    def test_effective_range_is_recomputed(self):
        result = _complete_result()
        result["package_applications"][0]["effective_high"] += 1
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "max(unfloored tier spec, owned child sum)",
        )

    def test_every_candidate_needs_exactly_one_application(self):
        result = _complete_result()
        missing = _complete_result()
        missing["package_applications"] = []
        _assert_error(
            validate_complete_result(missing, estimate_id=EST_ID),
            "has no application",
        )
        extra = copy.deepcopy(result["package_applications"][0])
        extra["application_id"] = "pa1_" + "e" * 16
        result["package_applications"].append(extra)
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "more than one application",
        )

    def test_application_status_reason_pairing(self):
        result = _complete_result()
        result["package_applications"][0]["reason_code"] = "decision_rejected"
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "not valid for status",
        )

    def test_audit_lists_must_be_empty(self):
        result = _complete_result()
        result["reconciliation_audit"]["lost_work_item_ids"] = ["wk1_" + "9" * 16]
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "must be empty",
        )

    def test_observability_total_must_be_the_phase_sum(self):
        result = _complete_result()
        result["observability"]["phase_timings_ms"]["total"] += 1
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "sum of the other phases",
        )

    def test_observability_tokens_must_reconcile(self):
        result = _complete_result()
        result["observability"]["terra_total_tokens"] += 1
        result["observability"]["combined_total_tokens"] += 1
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "do not reconcile",
        )

    def test_observability_funnel_must_reconcile(self):
        result = _complete_result()
        result["observability"]["funnel"]["ledger_entries"] += 1
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "do not reconcile",
        )

    def test_recompute_equality_catches_reason_code_drift(self):
        """A per-record-valid reason code that differs from the deterministic
        policy's output is caught only by the recompute-equality net."""
        result = _complete_result()
        entry = _find_entry(result, "standalone")
        assert entry["reason_code"] == "no_covering_package"
        entry["reason_code"] = "package_rejected"
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "deterministic reconciliation recompute",
        )

    def test_totals_must_reconcile_exactly(self):
        result = _complete_result()
        result["totals"]["headline"]["high"] += 1
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "headline must be exactly",
        )

    def test_all_errors_are_collected(self):
        result = _complete_result()
        result["condition_reviews"][0]["confidence"] = 0.9
        result["work_items"][0]["unit_policy"] = "per_galaxy"
        result["coverage_ledger"][1]["low"] = 999999
        res = validate_complete_result(result, estimate_id=EST_ID)
        assert len(res.errors) >= 2
