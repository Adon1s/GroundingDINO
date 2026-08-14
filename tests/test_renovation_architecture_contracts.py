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
    ENVELOPE_SCHEMA_VERSION,
    EVIDENCE_DEDUP_POLICY_VERSION,
    CONDITION_DISPOSITION_POLICY_VERSION,
    EstimateProvenance,
    ObservedCondition,
    POLICY_VERSIONS,
    PROJECTION_VERSION,
    RenovationEstimateEnvelope,
    SCAFFOLD_REASON,
)
from tools.renovation_architecture.ids import (
    make_condition_id,
    make_disposition_id,
    make_estimate_id,
    make_evidence_id,
    make_ledger_entry_id,
    make_package_candidate_id,
    make_package_decision_id,
    make_review_id,
    make_terra_call_id,
    make_work_item_id,
)
from tools.renovation_architecture.validators import (
    validate_complete_result,
    validate_condition_review_result,
    validate_envelope,
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


def _work_item(condition, action_code, low, high):
    return {
        "work_item_id": make_work_item_id(
            estimate_id=EST_ID,
            catalog_item_id=condition["catalog_item_id"],
            estimate_unit_id=condition["estimate_unit_id"],
            action_code=action_code,
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "condition_ids": [condition["condition_id"]],
        "catalog_item_id": condition["catalog_item_id"],
        "estimate_unit_id": condition["estimate_unit_id"],
        "action_code": action_code,
        "action_source": "work_item_code",
        "trade_bucket": "kitchen_cabinets_counters",
        "unit_policy": "per_kitchen",
        "unit_count": 1,
        "pricing_mode": "catalog_allowance",
        "low": low,
        "high": high,
        "status": "active",
        "reason_code": None,
    }


def _ledger_entry(work_item, representation, package_id=None, low=0, high=0):
    return {
        "entry_id": make_ledger_entry_id(
            estimate_id=EST_ID, work_item_id=work_item["work_item_id"]
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "work_item_id": work_item["work_item_id"],
        "representation": representation,
        "package_id": package_id,
        "low": low,
        "high": high,
    }


def _complete_result():
    """Two conditions -> two work items -> one approved single-child package
    plus one standalone work item, with exact totals."""
    c1 = _condition("outdated_or_damaged_cabinets", "kitchen_primary")
    c2 = _condition("bathroom_vanity_worn", "bathroom_1")
    ev1, rv1, dp1 = _lattice(c1)
    ev2, rv2, dp2 = _lattice(c2)
    w1 = _work_item(c1, "CABINETS_REPLACE", 1000, 3000)
    w2 = _work_item(c2, "VANITY_REPLACE", 500, 1500)
    p1 = {
        "package_candidate_id": make_package_candidate_id(
            estimate_id=EST_ID, package_type="kitchen_modernization", room_key="kitchen"
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "package_type": "kitchen_modernization",
        "room_key": "kitchen",
        "child_work_item_ids": [w1["work_item_id"]],
        "proposed_treatment": "full kitchen refresh",
        "low": 900,
        "high": 2800,
    }
    d1 = {
        "decision_id": make_package_decision_id(
            estimate_id=EST_ID, package_candidate_id=p1["package_candidate_id"]
        ),
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "package_candidate_id": p1["package_candidate_id"],
        "decision": "approve",
        "combine_with": [],
        "split_groups": [],
        "rationale": "coherent scope",
        "model": "sol-test",
        "prompt_version": "v1",
    }
    e1 = _ledger_entry(w1, "absorbed_by_package", package_id=p1["package_candidate_id"])
    e2 = _ledger_entry(w2, "standalone", low=500, high=1500)
    return {
        "observed_conditions": [c1, c2],
        "evidence_facts": [ev1, ev2],
        "condition_reviews": [rv1, rv2],
        "condition_dispositions": [dp1, dp2],
        "work_items": [w1, w2],
        "package_candidates": [p1],
        "package_decisions": [d1],
        "coverage_ledger": [e1, e2],
        "totals": {
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "currency": "USD",
            "standalone": {"low": 500, "high": 1500},
            "packaged": {"low": 900, "high": 2800},
            "inspection": {"low": 0, "high": 0},
            "headline": {"low": 1400, "high": 4300},
        },
    }


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
            (make_work_item_id(estimate_id=EST_ID, catalog_item_id="x", estimate_unit_id="u", action_code="a"), "wk1"),
            (make_package_candidate_id(estimate_id=EST_ID, package_type="t", room_key="r"), "pk1"),
            (make_package_decision_id(estimate_id=EST_ID, package_candidate_id="p"), "pd1"),
            (make_ledger_entry_id(estimate_id=EST_ID, work_item_id="w"), "cl1"),
        ]
        for value, prefix in pairs:
            assert re.fullmatch(rf"{prefix}_[0-9a-f]{{16}}", value), value

    def test_namespace_separates_record_types(self):
        """Identical remaining parts must never collide across record types."""
        evidence = make_evidence_id(estimate_id=EST_ID, condition_id="c")
        review = make_review_id(estimate_id=EST_ID, condition_id="c")
        disposition = make_disposition_id(estimate_id=EST_ID, condition_id="c")
        assert len({evidence[4:], review[4:], disposition[4:]}) == 3

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
         "package_decisions", "coverage_ledger"],
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
        result = _complete_result()
        result["condition_dispositions"][1]["disposition"] = "inspection"
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "not accepted_for_work",
        )

    def test_accepted_condition_with_no_work_item(self):
        result = _complete_result()
        c3 = _condition("cracked_tile", "bathroom_2")
        ev3, rv3, dp3 = _lattice(c3)
        result["observed_conditions"].append(c3)
        result["evidence_facts"].append(ev3)
        result["condition_reviews"].append(rv3)
        result["condition_dispositions"].append(dp3)
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "accepted scope cannot vanish",
        )

    def test_package_child_must_be_known(self):
        result = _complete_result()
        result["package_candidates"][0]["child_work_item_ids"] = ["wk1_" + "9" * 16]
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "not a known work item",
        )

    def test_package_child_must_be_active(self):
        result = _complete_result()
        result["work_items"][0]["status"] = "suppressed"
        result["work_items"][0]["reason_code"] = "below_floor"
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID), "suppressed"
        )

    def test_work_item_absorbed_by_two_approved_packages(self):
        result = _complete_result()
        p1 = result["package_candidates"][0]
        p2 = copy.deepcopy(p1)
        p2["package_candidate_id"] = make_package_candidate_id(
            estimate_id=EST_ID, package_type="kitchen_repair", room_key="kitchen"
        )
        p2["package_type"] = "kitchen_repair"
        d2 = copy.deepcopy(result["package_decisions"][0])
        d2["decision_id"] = make_package_decision_id(
            estimate_id=EST_ID, package_candidate_id=p2["package_candidate_id"]
        )
        d2["package_candidate_id"] = p2["package_candidate_id"]
        result["package_candidates"].append(p2)
        result["package_decisions"].append(d2)
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "absorbed by both",
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
        result["package_decisions"][0]["split_groups"] = [
            [result["work_items"][0]["work_item_id"], result["work_items"][1]["work_item_id"]]
        ]
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

    def test_absorbed_requires_approved_package(self):
        result = _complete_result()
        result["package_decisions"][0]["decision"] = "reject"
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "not an approved package",
        )

    def test_absorbed_entry_must_carry_zero_dollars(self):
        result = _complete_result()
        result["coverage_ledger"][0]["low"] = 1
        result["coverage_ledger"][0]["high"] = 1
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID), "0/0"
        )

    def test_rejected_package_children_stay_standalone(self):
        result = _complete_result()
        result["package_decisions"][0]["decision"] = "uncertain"
        result["coverage_ledger"][0] = _ledger_entry(
            result["work_items"][0], "inspection", low=1000, high=3000
        )
        _assert_error(
            validate_complete_result(result, estimate_id=EST_ID),
            "must remain standalone",
        )

    def test_totals_must_reconcile_exactly(self):
        result = _complete_result()
        result["totals"]["headline"]["high"] = 4301
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
