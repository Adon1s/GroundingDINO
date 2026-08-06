"""
Pass 2e must fail closed.

2e is the pass that turns observations into *verified* issues — it normalizes,
dedupes, and applies catalog policy gating. Its previous `except Exception`
handler logged a warning and then promoted the unverified observations under
the name the rest of the pipeline reads as verified. Every downstream surface —
the estimate, the evidence projection, the UI issue list — trusted that field.

That is the same failure shape as the worker marking a run COMPLETE with no
artifact, one level further down: a degraded result presented as a good one. So
2e raises instead, and these tests pin that the passthrough never comes back.

2e runs only in catalog_resolution_benchmark mode (production stops after 2c,
pinned by test_classification_only_stop_keeps_2e_dormant below); these tests
therefore run in benchmark mode. The `else` branch (2e disabled by toggle)
legitimately promotes observations — that is a deliberate configuration, not a
failure — and is covered here too so a future fix does not over-correct and
break the disabled path.
"""
import asyncio
from pathlib import Path

import pytest

from tools import scene_classifier_orchestrator as orch_module
from tools.pass_config import (
    PIPELINE_MODE_CATALOG_RESOLUTION_BENCHMARK,
    PassToggles,
    SceneClassifierRunOptions,
)
from tools.scene_classifier_orchestrator import SceneClassifierOrchestrator
from tools.scene_classifier_passes import PassExecutionError

from tests.test_scene_classifier_passes import FakeOrchestratorClient


# FakeOrchestratorClient's 2c contract classifies its observation as
# "degradation"; the candidate and catalog rows must agree or strict 2d raises
# kind_purity_violation before 2e is ever reached.
_CATALOG_ITEM = {
    "id": "damaged_or_aged_roof_shingles",
    "tier": "work",
    "drop_if_generic": False,
    "defaultHidden": False,
    "kind": "degradation",
    "trade_bucket": "roof_gutters",
}


def _candidate_provider(observation_text, context):
    return [{
        "item_id": "damaged_or_aged_roof_shingles",
        "name": "Damaged or Aged Roof Shingles",
        "description": "Worn shingles.",
        "support_any": ["roof", "shingle", "worn"],
        "trade_bucket": "roof_gutters",
        "kind": "degradation",
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
    An orchestrator whose passes actually populate `observations` and resolve
    them through 2d.

    The regression this file guards is 2e *promoting* observations into
    verified_issues. With empty observations the assertion passes no matter
    what the code does, so the provider and catalog are required for the test
    to have any teeth.
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
    Options that reach 2e: benchmark mode (production stops after 2c).

    Pass 2d is disabled because this orchestrator has no candidate provider, and
    an enabled 2d with resolvable observations fails closed with
    MissingCandidateProvider — which would abort the run *before* 2e and make
    these tests pass for the wrong reason. 2e runs regardless of 2d.
    """
    toggles = PassToggles(pass_2d=False, **toggle_overrides)
    return SceneClassifierRunOptions(
        toggles=toggles,
        pipeline_mode=PIPELINE_MODE_CATALOG_RESOLUTION_BENCHMARK,
    )


def _benchmark_options(**toggle_overrides):
    """Benchmark-mode options with the full 2b→2e chain enabled."""
    return SceneClassifierRunOptions(
        toggles=PassToggles(**toggle_overrides),
        pipeline_mode=PIPELINE_MODE_CATALOG_RESOLUTION_BENCHMARK,
    )


def _analyze(orchestrator, options=None):
    return asyncio.run(orchestrator.analyze_image(
        image_path=Path("photo_001.jpg"),
        options=options if options is not None else _options(),
    ))


def test_pass_2e_failure_raises_instead_of_passing_observations_through(monkeypatch):
    async def _boom(**kwargs):
        raise ValueError("malformed issue payload")

    monkeypatch.setattr(orch_module, "run_pass_2e", _boom)

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(_orchestrator())

    assert excinfo.value.pass_key == "2e"
    assert excinfo.value.code == "ValueError"


def test_failed_2e_never_promotes_observations_to_verified(monkeypatch):
    """
    The regression itself.

    `partial_result` is a dict (from `analysis.to_dict()`), and the pass chain
    must actually produce an observation — otherwise this asserts nothing.
    Both are checked explicitly before the real assertion, because a vacuous
    regression test is worse than no regression test.
    """
    async def _boom(**kwargs):
        raise ValueError("malformed issue payload")

    monkeypatch.setattr(orch_module, "run_pass_2e", _boom)

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(_full_chain_orchestrator(), _benchmark_options())

    partial = getattr(excinfo.value, "partial_result", None)
    assert isinstance(partial, dict), "no partial result to inspect"
    assert partial["observations"], (
        "fixture produced nothing to promote — this test would pass vacuously"
    )

    for field in ("verified_issues", "canonical_issues", "display_issues", "matched_issues"):
        assert partial[field] == [], (
            f"{field} was populated from observations after 2e failed"
        )


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


def test_malformed_payload_is_a_parse_failure(monkeypatch):
    async def _boom(**kwargs):
        raise TypeError("expected dict, got list")

    monkeypatch.setattr(orch_module, "run_pass_2e", _boom)

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(_orchestrator())

    assert excinfo.value.pass_key == "2e"
    assert excinfo.value.stage == "parse"


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


def test_disabling_2e_still_promotes_observations(monkeypatch):
    """
    The toggle-off path is a deliberate configuration, not a failure, and must
    keep working — otherwise fixing the failure path breaks the disabled path.
    Observations are promoted verbatim: the 2c fail-closed contract already
    guarantees a valid kind, so there is no re-derivation.
    """
    async def _never_called(**kwargs):  # pragma: no cover - asserts it is not reached
        raise AssertionError("2e ran despite being disabled")

    monkeypatch.setattr(orch_module, "run_pass_2e", _never_called)

    result = _analyze(_orchestrator(), _options(pass_2e=False))

    assert result.passes.get("2e", {}).get("skipped") is True
    assert result.observations, (
        "fixture produced nothing to promote — this test would pass vacuously"
    )
    assert result.verified_issues == result.observations
    assert result.canonical_issues == result.observations


def test_classification_only_stop_keeps_2e_dormant(monkeypatch):
    """The production guarantee under observation-kind-v2: in the default
    classification_only mode the pipeline stops after Pass 2c, so 2e never runs
    and nothing is ever promoted into the verified lanes. Only the benchmark
    mode (not reachable from from_analysis_profile) continues into 2d/2e."""
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
