"""
Unit tests for scripts/benchmark_kind_ontology.py (kind-ontology-v2 runner).

The runner is a script, not a package module — loaded via importlib like
tests/test_audit_pass2c_funnel.py. No LLM calls: repeats are fed as literal
outcome dicts and the fakes below drive the execution paths.
"""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "benchmark_kind_ontology", ROOT / "scripts" / "benchmark_kind_ontology.py"
)
bko = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bko)


# ─── fixtures ────────────────────────────────────────────────────────────────

ATOMIC_DEFECT = {
    "case_id": "a-defect-1", "case_type": "atomic", "slice": "development",
    "scene": "exterior_front", "batch_id": "b1",
    "input": "Deck boards are rotted through near the stairs.",
    "gold": {"kind": "defect"}, "source": "synthetic",
}
ATOMIC_DEGRADATION = {
    "case_id": "a-deg-1", "case_type": "atomic", "slice": "development",
    "scene": "exterior_front", "batch_id": "b1",
    "input": "Wood siding is faded and weathered but remains intact.",
    "gold": {"kind": "degradation"}, "source": "synthetic",
}
ATOMIC_MODERNIZATION = {
    "case_id": "a-mod-1", "case_type": "atomic", "slice": "development",
    "scene": "kitchen", "batch_id": "b2",
    "input": "Kitchen has dated oak cabinets in good repair.",
    "gold": {"kind": "modernization"}, "source": "synthetic",
}
EXCLUDED_NEUTRAL = {
    "case_id": "e-neutral-1", "case_type": "excluded", "slice": "development",
    "scene": "bedroom", "batch_id": "b2",
    "input": "The room contains a ceiling fan.",
    "gold": {"reason": "neutral_presence"}, "source": "synthetic",
}
MIXED_CASE = {
    "case_id": "m-1", "case_type": "mixed", "slice": "development",
    "scene": "exterior_front",
    "input": "The deck boards are weathered and rotted near the stairs.",
    "gold": {"claims": [
        {"component": "deck", "anchors_any": ["weathered"], "kind": "degradation"},
        {"component": "deck", "anchors_any": ["rot", "rotted"], "kind": "defect"},
    ]},
    "source": "synthetic",
}
CASES = [ATOMIC_DEFECT, ATOMIC_DEGRADATION, ATOMIC_MODERNIZATION, EXCLUDED_NEUTRAL, MIXED_CASE]


def _perfect_repeat():
    return {
        "a-defect-1": {"kind": "defect"},
        "a-deg-1": {"kind": "degradation"},
        "a-mod-1": {"kind": "modernization"},
        "e-neutral-1": {"exclude": "neutral_presence"},
        "m-1": {"observations": [
            {"description": "Deck boards are weathered.", "kind": "degradation"},
            {"description": "Deck boards are rotted near the stairs.", "kind": "defect"},
        ], "excluded": []},
    }


# ─── case validation and fingerprint ─────────────────────────────────────────

def test_validate_cases_accepts_the_fixture_set():
    bko.validate_cases(CASES)


def test_validate_cases_rejects_bad_kind():
    bad = dict(ATOMIC_DEFECT, case_id="x", gold={"kind": "upgrade"})
    with pytest.raises(ValueError, match="invalid"):
        bko.validate_cases([bad])


def test_validate_cases_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        bko.validate_cases([ATOMIC_DEFECT, dict(ATOMIC_DEGRADATION, case_id="a-defect-1")])


def test_fingerprint_changes_when_a_gold_label_changes():
    fp1 = bko.case_fingerprint(CASES)
    flipped = [dict(c) for c in CASES]
    flipped[0] = dict(flipped[0], gold={"kind": "degradation"})
    assert bko.case_fingerprint(flipped) != fp1


def test_batches_are_grouped_and_ordered_deterministically():
    batches = bko.group_observation_batches(CASES)
    assert [[c["case_id"] for c in b] for b in batches] == [
        ["a-defect-1", "a-deg-1"], ["a-mod-1", "e-neutral-1"],
    ]


# ─── mixed-case matcher ──────────────────────────────────────────────────────

def test_mixed_full_success():
    score = bko.score_mixed_outcome(MIXED_CASE, _perfect_repeat()["m-1"])
    assert score["full_success"] is True
    assert score["cross_kind_bundled"] is False


def test_mixed_bundled_observation_fails_the_case():
    outcome = {"observations": [
        {"description": "Deck boards are weathered and rotted.", "kind": "defect"},
    ], "excluded": []}
    score = bko.score_mixed_outcome(MIXED_CASE, outcome)
    assert score["full_success"] is False
    assert score["bundled"] is True
    assert score["cross_kind_bundled"] is True


def test_mixed_wrong_kind_fails_without_bundling():
    outcome = {"observations": [
        {"description": "Deck boards are weathered.", "kind": "modernization"},
        {"description": "Deck boards are rotted near the stairs.", "kind": "defect"},
    ], "excluded": []}
    score = bko.score_mixed_outcome(MIXED_CASE, outcome)
    assert score["full_success"] is False
    assert score["cross_kind_bundled"] is False


def test_matcher_is_word_start_anchored():
    claim = {"component": "mold", "anchors_any": ["visible"], "kind": "defect"}
    assert bko.observation_matches_claim("Visible mold on the wall.", claim) is True
    assert bko.observation_matches_claim("Visible crown molding detail.", claim) is True  # word-start: mold(ing)
    assert bko.observation_matches_claim("Visible smoldering damage.", claim) is False


# ─── v2 scoring ──────────────────────────────────────────────────────────────

def test_score_v2_perfect_run_passes_all_gates():
    repeats = [_perfect_repeat() for _ in range(5)]
    metrics = bko.score_v2(CASES, repeats)

    assert metrics["schema_failure_calls"] == 0
    assert metrics["atomic"]["kind_accuracy_mean"] == 1.0
    assert metrics["repeatability"]["unanimous_fraction"] == 1.0
    assert metrics["repeatability"]["pairwise_agreement"] == 1.0
    assert metrics["excluded"]["false_classification_rate"] == 0.0
    assert metrics["mixed"]["full_case_success_rate"] == 1.0

    gates = bko.evaluate_gates(metrics)
    assert all(g["passed"] for g in gates)


def test_score_v2_confusion_and_recall_track_a_miss():
    bad = _perfect_repeat()
    bad["a-deg-1"] = {"kind": "defect"}          # degradation misread as defect
    bad["e-neutral-1"] = {"kind": "modernization"}  # excluded text classified
    repeats = [_perfect_repeat(), bad]

    metrics = bko.score_v2(CASES, repeats)
    assert metrics["confusion_matrix"]["degradation"] == {"degradation": 1, "defect": 1}
    assert metrics["atomic"]["per_kind"]["degradation"]["recall"] == 0.5
    assert metrics["excluded"]["false_classification_rate"] == 0.5
    assert metrics["repeatability"]["unanimous_fraction"] == pytest.approx(0.5)

    gates = {g["gate"]: g["passed"] for g in bko.evaluate_gates(metrics)}
    assert gates["per_kind_recall"] is False
    assert gates["excluded_false_classification"] is False


def test_score_v2_counts_schema_failures_and_fails_the_partition_gate():
    bad = _perfect_repeat()
    bad["a-defect-1"] = {"error": "parse: missing decisions"}
    metrics = bko.score_v2(CASES, [bad])

    assert metrics["schema_failure_calls"] == 1
    assert metrics["confusion_matrix"]["defect"]["schema_failure"] == 1
    gates = {g["gate"]: g["passed"] for g in bko.evaluate_gates(metrics)}
    assert gates["valid_partitions"] is False


# ─── v1 baseline scoring ─────────────────────────────────────────────────────

def test_score_v1_reports_the_degradation_split():
    rep1 = {
        "a-defect-1": {"label": "defect_or_damage"},
        "a-deg-1": {"label": "defect_or_damage"},
        "a-mod-1": {"label": "upgrade_candidate"},
        "e-neutral-1": {"label": "generic_presence"},
        "m-1": {"labeled": []},
    }
    rep2 = dict(rep1, **{"a-deg-1": {"label": "upgrade_candidate"}})
    metrics = bko.score_v1(CASES, [rep1, rep2])

    split = metrics["degradation_split"]
    assert split["decisions"] == 2
    assert split["defect_or_damage"] == 1
    assert split["upgrade_candidate"] == 1
    assert split["unanimous_cases"] == 0
    assert metrics["mapped_accuracy"]["defect_as_defect_or_damage"] == 1.0
    assert metrics["excluded_forwarded_rate"] == 0.0


# ─── execution paths with a fake client ──────────────────────────────────────

class FakeClient:
    """Speaks both v2 (decisions) and v1 (labeled) shapes, keyed off prompts."""

    def __init__(self):
        self.calls = []

    async def analyze_text(self, system_prompt, user_prompt, **model_config):
        self.calls.append(system_prompt[:40])
        sys_lower = system_prompt.lower()
        if "classify each numbered observation" in sys_lower:
            n = len([l for l in user_prompt.splitlines() if l[:1].isdigit()])
            decisions = []
            for i in range(1, n + 1):
                decisions.append({"index": i, "kind": "degradation"})
            return json.dumps({"decisions": decisions})
        if "split freeform photo notes" in sys_lower:
            return '{"observations":[{"description":"Deck boards are weathered."}]}'
        if "label each observation" in sys_lower:
            rows = json.loads(user_prompt.split("OBSERVATIONS_JSON:", 1)[1])
            return json.dumps({"labeled": [
                {"description": r["description"], "label": "defect_or_damage"} for r in rows
            ]})
        return "{}"


def test_run_v2_repeat_maps_outcomes_by_description():
    client = FakeClient()
    cases = [ATOMIC_DEFECT, ATOMIC_DEGRADATION]
    outcomes = asyncio.run(bko.run_v2_repeat(client, {}, cases))
    assert outcomes == {
        "a-defect-1": {"kind": "degradation"},
        "a-deg-1": {"kind": "degradation"},
    }


def test_run_v1_repeat_uses_frozen_prompts():
    client = FakeClient()
    prompts = bko.load_v1_prompts()
    cases = [ATOMIC_DEFECT, MIXED_CASE]
    outcomes = asyncio.run(bko.run_v1_repeat(client, {}, prompts, cases))
    assert outcomes["a-defect-1"] == {"label": "defect_or_damage"}
    assert outcomes["m-1"]["labeled"][0]["label"] == "defect_or_damage"


# ─── report writing ──────────────────────────────────────────────────────────

def test_write_report_emits_json_and_markdown(tmp_path):
    repeats = [_perfect_repeat()]
    metrics = bko.score_v2(CASES, repeats)
    report = {
        "benchmark": bko.BENCHMARK_ID,
        "schema_version": bko.BENCHMARK_SCHEMA_VERSION,
        "meta": {
            "timestamp": "t", "contract": "v2", "cases_slice": "dev",
            "case_fingerprint": bko.case_fingerprint(CASES),
            "model_label": "fake", "model_config": {"model": "fake", "api_key": "secret"},
            "repeats": 1, "ontology_version": bko.ONTOLOGY_VERSION,
            "pass_2b_prompt_version": bko.PASS_2B_PROMPT_VERSION,
            "pass_2b_prompt_sha256": bko.PASS_2B_PROMPT_SHA256,
            "pass_2c_prompt_version": bko.PASS_2C_PROMPT_VERSION,
            "pass_2c_prompt_sha256": bko.PASS_2C_PROMPT_SHA256,
        },
        "metrics": metrics,
        "raw_repeats": repeats,
        "gates": bko.evaluate_gates(metrics),
    }
    path = bko.write_report(report, tmp_path / "out")
    assert path.exists()
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Confusion matrix" in md
    assert "secret" not in md, "api_key leaked into the markdown report"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metrics"]["atomic"]["kind_accuracy_mean"] == 1.0
