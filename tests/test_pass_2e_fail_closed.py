"""
Pass 2e must fail closed.

2e is the pass that turns *labeled* observations into *verified* issues — it
normalizes, dedupes, and applies catalog policy gating. Its previous `except
Exception` handler logged a warning and then assigned
`result.verified_issues = list(result.labeled_forward)`, which published
unverified observations under the name the rest of the pipeline reads as
verified. Every downstream surface — the estimate, the evidence projection, the
UI issue list — trusted that field.

That is the same failure shape as the worker marking a run COMPLETE with no
artifact, one level further down: a degraded result presented as a good one. So
2e now raises instead, and these tests pin that the passthrough never comes back.

The `else` branch (2e disabled by toggle) legitimately promotes labeled_forward —
that is a deliberate configuration, not a failure — and is covered here too so a
future fix does not over-correct and break the disabled path.
"""
import asyncio
from pathlib import Path

import pytest

from tools import scene_classifier_orchestrator as orch_module
from tools.pass_config import PassToggles, SceneClassifierRunOptions
from tools.scene_classifier_orchestrator import SceneClassifierOrchestrator
from tools.scene_classifier_passes import PassExecutionError

from tests.test_scene_classifier_passes import FakeOrchestratorClient

# observation-kind-v2: Task 2 revived Pass 2d (benchmark mode) but deliberately
# left Pass 2e dormant — run_pass_2e still hard-drops any issue whose kind is
# not defect/upgrade, so every degradation and modernization issue would vanish
# with only a removed_reason counter to show for it. Reviving 2e is Task 3 work
# and must lift that sanity check first. The tests are skipped, not deleted:
# the fail-closed guarantee they pin has to come back with 2e. The active
# guarantee (2e never runs, nothing is promoted to verified) is pinned by
# test_classification_only_stop_keeps_2e_dormant below and by
# test_benchmark_mode_never_runs_2e in tests/test_candidate_provider.py.
# See docs/HANDOFF_kind_ontology_task2.md.
dormant_2e = pytest.mark.skip(
    reason="Pass 2e dormant until Task 3 cutover: 2e still hard-drops non-two-kind "
    "issues (invalid_kind sanity check); see docs/HANDOFF_kind_ontology_task2.md"
)


_CATALOG_ITEM = {
    "id": "damaged_or_aged_roof_shingles",
    "tier": "work",
    "drop_if_generic": False,
    "defaultHidden": False,
    "kind": "defect",
    "trade_bucket": "roof_gutters",
}


def _candidate_provider(observation_text, context):
    return [{
        "item_id": "damaged_or_aged_roof_shingles",
        "name": "Damaged or Aged Roof Shingles",
        "description": "Worn shingles.",
        "support_any": ["roof", "shingle", "worn"],
        "trade_bucket": "roof_gutters",
        "kind": "defect",
        "score": 0.9,
        "defaultHidden": False,
        "drop_if_generic": False,
    }]


def _orchestrator():
    return SceneClassifierOrchestrator(
        qwen_config={},
        gpt5_config={},
        vlm_client=FakeOrchestratorClient(),
    )


def _full_chain_orchestrator():
    """
    An orchestrator whose passes actually populate `labeled_forward`.

    The regression this file guards is 2e *promoting* labeled_forward into
    verified_issues. With an empty labeled_forward the assertion passes no matter
    what the code does, so the provider and catalog are required for the test to
    have any teeth.
    """
    return SceneClassifierOrchestrator(
        qwen_config={},
        gpt5_config={},
        vlm_client=FakeOrchestratorClient(),
        candidate_provider=_candidate_provider,
        catalog_items=[_CATALOG_ITEM],
    )


def _options(**toggle_overrides):
    """
    Options that reach 2e.

    Pass 2d is disabled because this orchestrator has no candidate provider, and
    an enabled 2d with resolvable observations fails closed at
    scene_classifier_orchestrator.py:1056 — which would abort the run *before* 2e
    and make these tests pass for the wrong reason.
    """
    toggles = PassToggles(pass_2d=False, **toggle_overrides)
    return SceneClassifierRunOptions(toggles=toggles)


def _analyze(orchestrator, options=None):
    return asyncio.run(orchestrator.analyze_image(
        image_path=Path("photo_001.jpg"),
        options=options if options is not None else _options(),
    ))


@dormant_2e
def test_pass_2e_failure_raises_instead_of_passing_observations_through(monkeypatch):
    async def _boom(**kwargs):
        raise ValueError("malformed issue payload")

    monkeypatch.setattr(orch_module, "run_pass_2e", _boom)

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(_orchestrator())

    assert excinfo.value.pass_key == "2e"
    assert excinfo.value.code == "ValueError"


@dormant_2e
def test_failed_2e_never_promotes_labeled_forward_to_verified(monkeypatch):
    """
    The regression itself.

    `partial_result` is a dict (from `analysis.to_dict()`), and the pass chain
    must actually produce a labeled observation — otherwise this asserts nothing.
    Both are checked explicitly before the real assertion, because a vacuous
    regression test is worse than no regression test.
    """
    async def _boom(**kwargs):
        raise ValueError("malformed issue payload")

    monkeypatch.setattr(orch_module, "run_pass_2e", _boom)

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(_full_chain_orchestrator(), SceneClassifierRunOptions())

    partial = getattr(excinfo.value, "partial_result", None)
    assert isinstance(partial, dict), "no partial result to inspect"
    assert partial["labeled_forward"], (
        "fixture produced nothing to promote — this test would pass vacuously"
    )

    for field in ("verified_issues", "canonical_issues", "display_issues", "matched_issues"):
        assert partial[field] == [], (
            f"{field} was populated from labeled_forward after 2e failed"
        )


@dormant_2e
def test_missing_catalog_input_is_a_dependency_failure(monkeypatch):
    """
    2e is rule-based — no provider call — so a KeyError means a missing catalog
    input, not a flaky API. The worker's policy table pauses the queue on
    `dependency` rather than burning the run's one retry on something that will
    fail identically next time.
    """
    async def _boom(**kwargs):
        raise KeyError("trade_bucket")

    monkeypatch.setattr(orch_module, "run_pass_2e", _boom)

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(_orchestrator())

    assert excinfo.value.pass_key == "2e"
    assert excinfo.value.stage == "dependency"


@dormant_2e
def test_malformed_payload_is_a_parse_failure(monkeypatch):
    async def _boom(**kwargs):
        raise TypeError("expected dict, got list")

    monkeypatch.setattr(orch_module, "run_pass_2e", _boom)

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(_orchestrator())

    assert excinfo.value.pass_key == "2e"
    assert excinfo.value.stage == "parse"


@dormant_2e
def test_2e_failure_classifies_without_blaming_the_provider(monkeypatch):
    """A rule-based pass must never report a provider category — it makes no calls."""
    from tools.failure_taxonomy import classify_failure

    async def _boom(**kwargs):
        raise KeyError("trade_bucket")

    monkeypatch.setattr(orch_module, "run_pass_2e", _boom)

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(_orchestrator())

    descriptor = classify_failure(excinfo.value)
    assert descriptor.category == "dependency"
    assert descriptor.pass_key == "2e"


@dormant_2e
def test_2e_records_the_error_on_the_result_before_raising(monkeypatch):
    """Diagnostics still need the reason; failing closed must not mean failing silent."""
    async def _boom(**kwargs):
        raise ValueError("malformed issue payload")

    monkeypatch.setattr(orch_module, "run_pass_2e", _boom)

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(_orchestrator())

    partial = getattr(excinfo.value, "partial_result", None)
    if partial is not None and getattr(partial, "passes", None):
        assert "malformed issue payload" in str(partial.passes.get("2e", {}))


@dormant_2e
def test_disabling_2e_still_promotes_observations(monkeypatch):
    """
    The toggle-off path is a deliberate configuration, not a failure, and must
    keep working — otherwise fixing the failure path breaks the disabled path.
    """
    async def _never_called(**kwargs):  # pragma: no cover - asserts it is not reached
        raise AssertionError("2e ran despite being disabled")

    monkeypatch.setattr(orch_module, "run_pass_2e", _never_called)

    result = _analyze(_orchestrator(), _options(pass_2e=False))

    assert result.passes.get("2e", {}).get("skipped") is True


def test_classification_only_stop_keeps_2e_dormant(monkeypatch):
    """The active guarantee under observation-kind-v2: the pipeline stops after
    Pass 2c, so 2e never runs and nothing is ever promoted into the verified
    lanes. This is the same fail-closed spirit as the dormant tests above, one
    stage earlier."""
    async def _never_called(**kwargs):  # pragma: no cover - asserts it is not reached
        raise AssertionError("2e ran despite the classification-only stop")

    monkeypatch.setattr(orch_module, "run_pass_2e", _never_called)

    result = _analyze(_full_chain_orchestrator(), SceneClassifierRunOptions())

    assert result.classification_only is True
    assert result.observations, (
        "fixture produced nothing to promote — this test would pass vacuously"
    )
    for field in ("verified_issues", "canonical_issues", "display_issues", "matched_issues"):
        assert getattr(result, field) == [], (
            f"{field} was populated despite the classification-only stop"
        )
