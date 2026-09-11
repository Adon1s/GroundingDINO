"""Guard the offline analysis's consequential joins and counterfactual semantics."""
import pytest

from scripts.analysis.corroboration_gate_tradeoff import DEFAULT_FAMILIES, gate_outcome, label_class
from scripts.analysis.terra_miss_mechanisms import signals


def test_adjacent_screen_distinguishes_observed_terms_from_alternative_terms():
    row = {"claim": "interior paint peeling, bubbling, or visibly aged finish",
           "observations": ["The paint appears tired."],
           "rationale": "The paint does not show peeling or bubbling."}
    result = signals(row)
    assert result["adjacent_only"]
    assert result["adjacent_terms"] == ["bubble", "peel"]
    row["rationale"] = "The paint does not show peeling, bubbling, or aged finish."
    assert signals(row)["adjacent_any"]
    assert not signals(row)["adjacent_only"]


def test_negation_does_not_inherit_the_positive_clause_after_but():
    row = {"claim": "hard flooring scratches or worn finish",
           "observations": ["The flooring is older."],
           "rationale": "No broken components, but the flooring has scratches and wear."}
    assert not signals(row)["adjacent_any"]


@pytest.mark.parametrize("verdict,views,selected,accepted,expected", [
    ("supported", 1, True, True, True),
    ("supported", 2, True, True, False),
    ("unsupported", 1, True, False, False),
    ("cannot_assess", 1, True, False, False),
    ("supported", 1, False, True, False),
    ("supported", 1, True, False, False),
])
def test_gate_counts_only_newly_withheld_work(verdict, views, selected, accepted, expected):
    assert gate_outcome(verdict, views, selected, accepted)[0] == expected


def test_repaired_labels_keep_misnamed_trivial_and_absent_separate():
    assert label_class({"claim": "absent", "work": "none"}) == "absent"
    assert label_class({"claim": "misnamed", "work": "warranted"}) == "misnamed"
    assert label_class({"claim": "exact", "work": "trivial"}) == "trivial"
    assert label_class({"claim": "exact", "work": "warranted"}) == "supported_and_warranted"
    assert label_class({"claim": None, "work": None}) == "inconclusive"


def test_gate_default_respects_closed_catalog_decisions():
    assert "dated_window_treatment_valance" not in DEFAULT_FAMILIES
    assert "dated_interior_trim" not in DEFAULT_FAMILIES
    assert "worn_or_stained_flooring" in DEFAULT_FAMILIES
