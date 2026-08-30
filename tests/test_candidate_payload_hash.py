"""Per-candidate payload hashing (Catalog 3.2).

The offline replay harness joins stored Sol decisions onto rebuilt candidates.
make_package_candidate_id hashes only (estimate_id, package_type,
estimate_unit_id), so a candidate can gain or lose members — or change a
child's role, its pricing tier or its range — while keeping its ID. Matching on
ID alone would reuse Sol's judgement of a package that no longer exists in that
shape. These tests pin what the payload hash is sensitive to.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_candidate_payload_hash.py -q
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from tools.renovation_architecture.candidate_payload import (
    build_candidate_payload,
    candidate_payload_hashes,
    candidate_payload_index,
    candidate_payload_sha256,
)


def _work(work_item_id, low=100, high=200, **over):
    item = {
        "work_item_id": work_item_id,
        "action_code": "REPLACE",
        "trade_bucket": "flooring",
        "billable_unit_id": "bedroom_primary",
        "unit_count": 1,
        "estimate_scope": "photo_supported",
        "low": low,
        "high": high,
        "condition_ids": [f"cond_{work_item_id}"],
    }
    item.update(over)
    return item


def _result(work_items):
    condition_ids = [cid for item in work_items for cid in item["condition_ids"]]
    return {
        "work_items": work_items,
        "observed_conditions": [
            {"condition_id": cid, "catalog_item_id": cid.replace("cond_", "cat_")}
            for cid in condition_ids
        ],
        "evidence_facts": [
            {"condition_id": cid, "distinct_photo_count": 2, "distinct_view_count": 2}
            for cid in condition_ids
        ],
        "condition_reviews": [
            {"condition_id": cid, "verdict": "supported", "rationale": "visible"}
            for cid in condition_ids
        ],
        "condition_dispositions": [
            {"condition_id": cid, "disposition": "accepted_for_work",
             "reason_code": "route_work"}
            for cid in condition_ids
        ],
    }


def _candidate(children=("wk_a", "wk_b"), drivers=("wk_a",), **over):
    candidate = {
        "package_candidate_id": "pk1_deadbeefdeadbeef",
        "package_type": "bedroom_repair",
        "package_category": "repair",
        "package_level": "room",
        "room": "bedroom",
        "estimate_unit_id": "bedroom_primary",
        "strength": "moderate",
        "pricing_tier": "bedroom_repair_light",
        "proposed_treatment": "repair",
        "low": 500,
        "high": 2500,
        "display_only": False,
        "contributing_candidate_ids": ["cd1_aaaa"],
        "child_work_item_ids": list(children),
        "driver_work_item_ids": list(drivers),
    }
    candidate.update(over)
    return candidate


def _hash(candidate, work_items):
    result = _result(work_items)
    return candidate_payload_sha256(
        build_candidate_payload(candidate, **candidate_payload_index(result))
    )


BASE_WORK = [_work("wk_a"), _work("wk_b")]


def test_identical_candidates_hash_identically():
    assert _hash(_candidate(), BASE_WORK) == _hash(_candidate(), BASE_WORK)


def test_hash_ignores_the_candidate_id():
    """The hash answers 'same package?', not 'same id?'."""
    other = _candidate(package_candidate_id="pk1_0000000000000000")
    assert _hash(other, BASE_WORK) == _hash(_candidate(), BASE_WORK)


def test_membership_change_moves_the_hash():
    """The Catalog 3.2 case: a modernization candidate that loses a member to
    the repair family keeps its id."""
    shrunk = _candidate(children=("wk_a",), drivers=("wk_a",))
    assert _hash(shrunk, BASE_WORK) != _hash(_candidate(), BASE_WORK)


def test_child_role_flip_moves_the_hash():
    """Same members, different roles — exactly what forcing a moved occurrence
    to support does."""
    flipped = _candidate(drivers=("wk_a", "wk_b"))
    assert _hash(flipped, BASE_WORK) != _hash(_candidate(), BASE_WORK)


@pytest.mark.parametrize("field,value", [
    ("pricing_tier", "bedroom_repair_heavy"),
    ("low", 900),
    ("high", 9000),
    ("package_type", "bedroom_modernization"),
    ("strength", "strong"),
    ("display_only", True),
])
def test_scalar_changes_move_the_hash(field, value):
    assert _hash(_candidate(**{field: value}), BASE_WORK) != _hash(
        _candidate(), BASE_WORK
    )


def test_child_range_change_moves_the_hash():
    """A child re-priced upstream changes the package Sol judged."""
    repriced = [_work("wk_a", low=999), _work("wk_b")]
    assert _hash(_candidate(), repriced) != _hash(_candidate(), BASE_WORK)


def test_candidate_payload_hashes_indexes_by_candidate_id():
    result = _result(BASE_WORK)
    hashes = candidate_payload_hashes(result, [_candidate()])
    assert set(hashes) == {"pk1_deadbeefdeadbeef"}
    assert hashes["pk1_deadbeefdeadbeef"] == _hash(_candidate(), BASE_WORK)


# ── the replay join ──────────────────────────────────────────────────────────

def _replay():
    spec = importlib.util.spec_from_file_location(
        "replay_renovation_architecture",
        "scripts/replay_renovation_architecture.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stored(candidate, work_items):
    return {
        **_result(work_items),
        "package_candidates": [candidate],
        "package_decisions": [{
            "package_candidate_id": candidate["package_candidate_id"],
            "decision_id": "dc1_1111",
            "split_groups": [],
            "combine_with": [],
        }],
        "sol_calls": [{
            "package_candidate_ids": [candidate["package_candidate_id"]],
        }],
    }


def test_replay_reuses_a_decision_when_the_payload_is_unchanged():
    replay = _replay()
    counters = {key: 0 for key in replay._SANITIZATION_KEYS}
    stored = _stored(_candidate(), BASE_WORK)
    kept, decisions, _ = replay._map_stored_decisions(
        [_candidate()], stored, counters, rebuilt_result=_result(BASE_WORK)
    )
    assert len(kept) == 1 and len(decisions) == 1
    assert counters["stored_decisions_payload_mismatch"] == 0
    assert counters["candidates_dropped_no_stored_decision"] == 0


def test_replay_drops_a_decision_when_the_payload_changed():
    """Same candidate id, different package: the decision must NOT be reused."""
    replay = _replay()
    counters = {key: 0 for key in replay._SANITIZATION_KEYS}
    stored = _stored(_candidate(), BASE_WORK)
    rebuilt = _candidate(children=("wk_a",), drivers=("wk_a",))
    kept, decisions, _ = replay._map_stored_decisions(
        [rebuilt], stored, counters, rebuilt_result=_result(BASE_WORK)
    )
    assert kept == [] and decisions == []
    assert counters["stored_decisions_payload_mismatch"] == 1
    assert counters["stored_decisions_unused"] == 1


def test_payload_mismatch_is_a_reported_counter():
    replay = _replay()
    assert "stored_decisions_payload_mismatch" in replay._SANITIZATION_KEYS


def test_replay_never_imports_a_provider_module():
    """Three docs pin this invariant; the payload helpers were split out of
    sol_review specifically to keep it true.

    Runs in a subprocess: importing the module in-process would see whatever
    the rest of the suite already loaded, so an in-process sys.modules check
    passes or fails by test ordering rather than by the import graph.
    """
    probe = (
        "import importlib.util, sys; "
        "spec = importlib.util.spec_from_file_location("
        "'replay_probe', 'scripts/replay_renovation_architecture.py'); "
        "m = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(m); "
        "print(','.join(sorted(n for n in sys.modules if 'sol_review' in n "
        "or 'terra_review' in n or 'vlm_client' in n)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        "replay pulled in a provider module: " + completed.stdout.strip()
    )
