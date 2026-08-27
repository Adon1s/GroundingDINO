"""Session B: scoring re-decided verdicts against the frozen human labels."""
import json
from pathlib import Path

import pytest

from scripts.score_redecide_variants import (
    classify,
    load_arm,
    load_population,
    score_arm,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE = REPO_ROOT / "reports" / "review_queue.json"
VERDICTS = REPO_ROOT / "reports" / "review_verdicts.jsonl"


def _card(strata, accepted, condition_id="c1"):
    return {"strata": strata,
            "meta": {"accepted": accepted, "condition_id": condition_id}}


def test_classify_matches_the_frozen_class_definitions():
    assert classify(_card(["dirA"], True), "terra_claim_unsupported") == (
        "hard_false_billed"
    )
    assert classify(_card(["uniform"], True), "terra_claim_supported") == (
        "supported_billed"
    )
    assert classify(_card(["dirA"], True), "terra_claim_overstated") == (
        "overstated_billed"
    )
    assert classify(_card(["dirB"], False), "terra_claim_supported") == (
        "dirB_recovery"
    )
    assert classify(_card(["dirB"], False), "terra_claim_unsupported") == (
        "dirB_other"
    )
    # Flip-only and p6-only cards never enter the billed classes.
    assert classify(_card(["terra_flip"], True), "terra_claim_supported") == (
        "out_of_scope"
    )
    assert classify(
        _card(["p6_forced_single"], True), "terra_claim_supported"
    ) == "out_of_scope"


def _row(klass, condition_id, stored="supported", human="terra_claim_supported"):
    return {"card_id": f"card_{condition_id}", "property_key": "prop",
            "condition_id": condition_id, "stored_verdict": stored,
            "human_verdict": human, "klass": klass}


def test_score_arm_counts_wins_losses_costs_and_coverage():
    population = [
        _row("hard_false_billed", "c1", human="terra_claim_unsupported"),
        _row("supported_billed", "c2"),
        _row("overstated_billed", "c3", human="terra_claim_overstated"),
        _row("dirB_recovery", "c4", stored="unsupported"),
        _row("supported_billed", "c5"),  # not re-decided
    ]
    arm = {"label": "test", "dry_run": False, "statuses": {},
           "verdicts": {("prop", "c1"): "unsupported",
                        ("prop", "c2"): "cannot_assess",
                        ("prop", "c3"): "unsupported",
                        ("prop", "c4"): "supported"}}
    classes = score_arm(population, arm)["classes"]
    assert classes["hard_false_billed"]["wins"] == 1
    assert classes["supported_billed"]["losses"] == 1
    assert classes["supported_billed"]["not_redecided"] == 1
    assert classes["overstated_billed"]["costs"] == 1
    assert classes["dirB_recovery"]["wins"] == 1
    assert classes["hard_false_billed"]["transitions"] == {
        "supported->unsupported": 1
    }


def test_load_arm_reads_unit_records(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"variant": {"label": "arm_x"}, "dry_run": False}),
        encoding="utf-8",
    )
    units = tmp_path / "prop" / "units"
    units.mkdir(parents=True)
    (units / "terra_unit_aa.json").write_text(
        json.dumps({"property_key": "prop", "status": "redecided",
                    "condition_ids": ["c1"],
                    "reviews": {"c1": {"verdict": "unsupported",
                                       "rationale": "r"}}}),
        encoding="utf-8",
    )
    arm = load_arm(tmp_path)
    assert arm["label"] == "arm_x"
    assert arm["verdicts"] == {("prop", "c1"): "unsupported"}


@pytest.mark.skipif(
    not (QUEUE.is_file() and VERDICTS.is_file()),
    reason="frozen review inputs not present",
)
def test_frozen_population_matches_the_audited_class_counts():
    """125 canary condition cards, 11/55/6/16 — load_population hard-asserts
    these, so this test failing means the frozen inputs drifted."""
    rows = load_population(QUEUE, VERDICTS)
    assert len(rows) == 125
    assert all(row["human_verdict"] for row in rows)
