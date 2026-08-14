"""Deterministic disposition policy tests (condition_disposition_v1).

The full verdict x route matrix, the insufficient-distinct-views precedence,
and the validator's recompute-and-compare consistency gate.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_disposition.py -q
"""
import pytest

from tests.test_renovation_architecture_contracts import (
    EST_ID,
    _assert_error,
    _review_result,
)
from tools.renovation_architecture.contracts import (
    CONDITION_DISPOSITION_POLICY_VERSION,
    DISPOSITION_REASON_CODES,
    DISPOSITIONS,
    TERMINAL_ROUTES,
)
from tools.renovation_architecture.disposition import decide_disposition
from tools.renovation_architecture.validators import (
    validate_condition_review_result,
)

# The whole policy, spelled out: (verdict, route, views, threshold) -> outcome.
FULL_MATRIX = [
    # unsupported -> excluded regardless of route or evidence
    ("unsupported", "work", 5, None, ("excluded", "verdict_unsupported")),
    ("unsupported", "inspection", 1, 2, ("excluded", "verdict_unsupported")),
    ("unsupported", "no_action", 0, None, ("excluded", "verdict_unsupported")),
    ("unsupported", "excluded_generic", 3, None, ("excluded", "verdict_unsupported")),
    ("unsupported", "excluded_quarantine", 3, None, ("excluded", "verdict_unsupported")),
    # cannot_assess -> inspection regardless of route or evidence
    ("cannot_assess", "work", 5, None, ("inspection", "verdict_cannot_assess")),
    ("cannot_assess", "inspection", 1, 2, ("inspection", "verdict_cannot_assess")),
    ("cannot_assess", "no_action", 0, None, ("inspection", "verdict_cannot_assess")),
    ("cannot_assess", "excluded_generic", 3, None, ("inspection", "verdict_cannot_assess")),
    ("cannot_assess", "excluded_quarantine", 3, None, ("inspection", "verdict_cannot_assess")),
    # supported with adequate evidence -> the terminal route decides
    ("supported", "work", 2, 2, ("accepted_for_work", "route_work")),
    ("supported", "work", 1, None, ("accepted_for_work", "route_work")),
    ("supported", "inspection", 3, None, ("inspection", "route_inspection")),
    ("supported", "no_action", 3, None, ("no_action", "route_no_action")),
    ("supported", "excluded_generic", 3, None, ("excluded", "route_excluded_generic")),
    ("supported", "excluded_quarantine", 3, None, ("excluded", "route_excluded_quarantine")),
    # supported but under the explicit evidence threshold -> withheld, on
    # every route (the gate outranks even exclusion routes)
    ("supported", "work", 1, 2, ("withheld", "insufficient_distinct_views")),
    ("supported", "inspection", 1, 2, ("withheld", "insufficient_distinct_views")),
    ("supported", "no_action", 1, 2, ("withheld", "insufficient_distinct_views")),
    ("supported", "excluded_generic", 1, 2, ("withheld", "insufficient_distinct_views")),
    ("supported", "excluded_quarantine", 1, 2, ("withheld", "insufficient_distinct_views")),
    ("supported", "work", 0, 1, ("withheld", "insufficient_distinct_views")),
    ("supported", "work", None, 2, ("withheld", "insufficient_distinct_views")),
]


class TestDecideDisposition:
    @pytest.mark.parametrize(
        "verdict,route,views,threshold,expected", FULL_MATRIX
    )
    def test_full_matrix(self, verdict, route, views, threshold, expected):
        assert decide_disposition(verdict, route, views, threshold) == expected

    def test_every_outcome_uses_the_closed_vocabularies(self):
        for _, _, _, _, (disposition, reason_code) in FULL_MATRIX:
            assert disposition in DISPOSITIONS
            assert reason_code in DISPOSITION_REASON_CODES

    def test_matrix_covers_every_verdict_and_route(self):
        verdicts = {row[0] for row in FULL_MATRIX}
        routes = {row[1] for row in FULL_MATRIX}
        assert verdicts == {"supported", "unsupported", "cannot_assess"}
        assert routes == TERMINAL_ROUTES

    def test_no_threshold_never_withholds(self):
        for views in (None, 0, 1, 50):
            disposition, _ = decide_disposition("supported", "work", views, None)
            assert disposition == "accepted_for_work"

    def test_unknown_verdict_raises(self):
        with pytest.raises(ValueError, match="verdict"):
            decide_disposition("maybe", "work", 2, None)

    def test_unknown_route_raises(self):
        with pytest.raises(ValueError, match="terminal route"):
            decide_disposition("supported", "detour", 2, None)

    def test_policy_version_constant(self):
        assert CONDITION_DISPOSITION_POLICY_VERSION == "condition_disposition_v1"

    def test_determinism(self):
        first = decide_disposition("supported", "work", 1, 2)
        assert all(
            decide_disposition("supported", "work", 1, 2) == first
            for _ in range(3)
        )


class TestValidatorConsistencyGate:
    def test_contradicting_disposition_is_rejected(self):
        """A result whose recorded disposition disagrees with the recomputed
        policy cannot validate — the table is executable, not advisory."""
        result = _review_result()
        record = result["condition_dispositions"][0]
        assert record["disposition"] == "accepted_for_work"
        record["disposition"] = "no_action"
        record["reason_code"] = "route_no_action"
        _assert_error(
            validate_condition_review_result(result, estimate_id=EST_ID),
            "computes",
        )

    def test_wrong_reason_code_alone_is_rejected(self):
        result = _review_result()
        result["condition_dispositions"][0]["reason_code"] = "route_inspection"
        _assert_error(
            validate_condition_review_result(result, estimate_id=EST_ID),
            "computes",
        )

    def test_withheld_requires_the_evidence_to_say_so(self):
        """Marking a condition withheld while its evidence clears the
        threshold is a contradiction, not a policy choice."""
        result = _review_result()
        record = result["condition_dispositions"][0]
        record["disposition"] = "withheld"
        record["reason_code"] = "insufficient_distinct_views"
        _assert_error(
            validate_condition_review_result(result, estimate_id=EST_ID),
            "computes",
        )
