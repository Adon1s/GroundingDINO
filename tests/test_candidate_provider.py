"""
Tests for Pass 2d candidate retrieval and per-observation debug rows.

Two defects motivated this:

1. The shadow lane and Pass 2d each hand-rolled "call the provider, catch
   TypeError, retry with one argument". Probing by TypeError cannot tell a
   legacy one-argument provider from a two-argument provider that raised
   TypeError *internally*, so a genuine provider bug was silently retried and
   could resolve against the wrong signature. Arity is now decided by binding
   the signature before the call.

   Retrieval failures stay fail-closed in Pass 2d (a dependency failure, not
   "no candidates") and stay skippable in the log-only shadow lane.

2. The missing-issue_id branch appended the *same* debug_row object a second
   time, so that observation serialized twice in
   debug["pass_2d_per_observation"].
"""
import asyncio
import functools
from pathlib import Path

import pytest

import tools.scene_classifier_orchestrator as orchestrator_module
from tools.pass_config import PassModelOverrides, SceneClassifierRunOptions
from tools.scene_classifier_orchestrator import (
    PROVIDER_IGNORED_CONTEXT,
    SceneClassifierOrchestrator,
    _provider_accepts_context,
    _retrieve_candidates,
)
from tools.scene_classifier_passes import PassExecutionError

OBSERVATION = "Shingles appear aged and weathered from an aerial angle."
# Kind matches the 2c decision below: exact-kind retrieval means the candidate
# pool is pure, and the orchestrator fails closed on an off-kind candidate.
CANDIDATE = {
    "item_id": "roof_shingles_aged_or_worn",
    "name": "Aged or Worn Roof Shingles",
    "description": "Worn shingle granules, roof surface discoloration, moss growth.",
    "support_any": ["roof", "shingle", "worn"],
    "trade_bucket": "roof_gutters",
    "kind": "degradation",
    "score": 0.70,
    "defaultHidden": False,
    "drop_if_generic": False,
}


class FakeOrchestratorClient:
    """Drives 1a -> 2c so Pass 2d has one forward observation to resolve."""

    async def analyze_image(self, image_path, system_prompt, user_prompt, **model_config):
        if "scene type" in (system_prompt or "").lower():
            return '{"scene":"exterior_front","reasoning":"front exterior"}'
        return OBSERVATION

    async def analyze_text(self, system_prompt, user_prompt, **model_config):
        system_lower = (system_prompt or "").lower()
        user_lower = (user_prompt or "").lower()
        if "split freeform photo notes" in system_lower:
            return '{"observations":[{"description":"%s"}]}' % OBSERVATION
        if "classify each numbered observation" in system_lower:
            return '{"decisions":[{"index":1,"kind":"degradation"}]}'
        if "map this observation to a catalog item id" in user_lower:
            return '{"resolved_item_id":"roof_shingles_aged_or_worn"}'
        return "{}"


# observation-kind-v2 (Task 2): Pass 2d runs only under the catalog-resolution
# benchmark mode. These tests drive that mode so the fail-closed retrieval
# policy and one-row-per-observation debug contract stay pinned; production
# runs still stop after Pass 2c.
BENCHMARK_MODE = "catalog_resolution_benchmark"


def _benchmark_options(**kwargs):
    return SceneClassifierRunOptions(pipeline_mode=BENCHMARK_MODE, **kwargs)


def _analyze(candidate_provider, options=None):
    orchestrator = SceneClassifierOrchestrator(
        qwen_config={},
        gpt5_config={},
        vlm_client=FakeOrchestratorClient(),
        candidate_provider=candidate_provider,
        top_k_candidates=5,
    )
    return asyncio.run(orchestrator.analyze_image(
        image_path=Path("photo_002.jpg"),
        options=options or _benchmark_options(),
    ))


# ── signature detection ─────────────────────────────────────────────────────

def _two_arg(description, context):
    return [dict(CANDIDATE)]


def _one_arg(description):
    return [dict(CANDIDATE)]


def test_two_argument_provider_is_detected():
    assert _provider_accepts_context(_two_arg) is True


def test_legacy_one_argument_provider_is_detected():
    assert _provider_accepts_context(_one_arg) is False


def test_provider_with_default_context_is_detected():
    assert _provider_accepts_context(lambda d, c=None: []) is True


def test_partially_applied_provider_is_detected():
    assert _provider_accepts_context(functools.partial(_two_arg, context={})) is False


def test_non_introspectable_callable_assumes_current_contract():
    """object() has no signature; assume the two-argument contract."""
    class Opaque:
        def __call__(self, *args, **kwargs):
            return []

    assert _provider_accepts_context(Opaque()) is True


# ── retrieval adapter ───────────────────────────────────────────────────────

def test_retrieve_passes_context_to_a_two_argument_provider():
    seen = {}

    def provider(description, context):
        seen["description"] = description
        seen["context"] = context
        return [dict(CANDIDATE)]

    result, note = _retrieve_candidates(provider, "desc", {"kind": "defect"})
    assert result == [dict(CANDIDATE)]
    assert note is None
    assert seen == {"description": "desc", "context": {"kind": "defect"}}


def test_retrieve_falls_back_to_the_legacy_signature_with_a_note():
    result, note = _retrieve_candidates(_one_arg, "desc", {"kind": "defect"})
    assert result == [dict(CANDIDATE)]
    assert note == PROVIDER_IGNORED_CONTEXT


def test_internal_type_error_is_not_retried_as_a_legacy_provider():
    """The old probe re-called a broken provider with one argument."""
    calls = []

    def provider(description, context):
        calls.append((description, context))
        raise TypeError("provider guts exploded")

    with pytest.raises(TypeError, match="provider guts exploded"):
        _retrieve_candidates(provider, "desc", {})
    assert len(calls) == 1


def test_other_exceptions_propagate_to_the_caller():
    def provider(description, context):
        raise ValueError("index unavailable")

    with pytest.raises(ValueError, match="index unavailable"):
        _retrieve_candidates(provider, "desc", {})


# ── Pass 2d failure policy (fail-closed) ────────────────────────────────────

@pytest.mark.parametrize("exc", [
    ValueError("index unavailable"),
    RuntimeError("socket closed"),
    TypeError("provider guts exploded"),
])
def test_provider_failure_becomes_a_typed_dependency_failure(exc):
    """
    A raw exception escaping _run_passes bypasses the PassExecutionError
    boundary entirely, skipping _finalize and collect-mode handling.
    """
    def provider(description, context):
        raise exc

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(provider)
    assert excinfo.value.pass_key == "2d"
    assert excinfo.value.stage == "dependency"


def test_broken_provider_is_called_once_per_observation():
    calls = []

    def provider(description, context):
        calls.append(description)
        raise TypeError("provider guts exploded")

    with pytest.raises(PassExecutionError):
        _analyze(provider)
    assert len(calls) == 1


# ── Pass 2d debug rows: exactly one per observation ─────────────────────────

def _rows(result):
    return result.debug["pass_2d_per_observation"]


def test_legacy_provider_records_one_row_with_the_context_note():
    result = _analyze(_one_arg)
    rows = _rows(result)
    assert len(rows) == 1
    assert rows[0]["skipped_reason"] == PROVIDER_IGNORED_CONTEXT


def test_async_provider_records_one_row():
    async def provider(description, context):
        return [dict(CANDIDATE)]

    rows = _rows(_analyze(provider))
    assert len(rows) == 1
    assert "returned_coroutine" in rows[0]["skipped_reason"]


def test_non_list_provider_records_one_row():
    def provider(description, context):
        return {"item_id": "not-a-list"}

    rows = _rows(_analyze(provider))
    assert len(rows) == 1
    assert "returned_nonlist (dict)" in rows[0]["skipped_reason"]


def test_empty_candidates_records_one_row():
    def provider(description, context):
        return []

    rows = _rows(_analyze(provider))
    assert len(rows) == 1
    assert rows[0]["skipped_reason"] == "no_candidates"
    assert rows[0]["candidate_count"] == 0


def test_successful_resolution_records_one_row():
    rows = _rows(_analyze(_two_arg))
    assert len(rows) == 1
    assert rows[0]["skipped_reason"] is None
    assert rows[0]["top_candidate_id"] == "roof_shingles_aged_or_worn"


def test_missing_issue_id_records_exactly_one_row(monkeypatch):
    """The regression: this branch used to append the same row a second time."""
    monkeypatch.setattr(
        orchestrator_module, "_make_issue_id", lambda *args, **kwargs: "",
    )
    result = _analyze(_two_arg)
    rows = _rows(result)

    assert len(rows) == 1
    assert rows[0]["skipped_reason"] == "missing_issue_id (expected stamped in 2c)"
    # the observation is skipped, so nothing is resolved
    assert result.resolved_items == []


# ── model routing sources ───────────────────────────────────────────────────

def _sources(result):
    return {entry["pass"]: entry["source"] for entry in result.model_routing}


def test_standard_profile_reports_standard_default():
    result = _analyze(_two_arg, _benchmark_options(premium=False))
    assert _sources(result)["1a"] == "standard_default"


def test_premium_profile_reports_premium_default():
    result = _analyze(_two_arg, _benchmark_options(premium=True))
    assert _sources(result)["1a"] == "premium_default"


def test_model_override_reports_explicit_override():
    options = _benchmark_options(
        model_overrides=PassModelOverrides(model_1a="gpt-5.6-sol"),
    )
    result = _analyze(_two_arg, options)
    assert _sources(result)["1a"] == "explicit_override"


@pytest.mark.parametrize("premium", [False, True])
def test_orchestrator_never_reports_env_override(premium):
    """
    The docstring advertised an "env_override" source the orchestrator cannot
    produce. (It is still a real source in rerun_pass_2f_artifact.py.)
    """
    result = _analyze(_two_arg, SceneClassifierRunOptions(premium=premium))
    sources = set(_sources(result).values())
    assert "env_override" not in sources
    assert sources <= {"explicit_override", "premium_default", "standard_default"}


# ── pipeline_mode contract (observation-kind-v2, Task 2) ────────────────────

def test_default_mode_stops_after_2c_and_never_retrieves():
    """Production options default to classification_only: Pass 2d must not run
    and the candidate provider must never be called."""
    calls = []

    def provider(description, context):
        calls.append(description)
        return [dict(CANDIDATE)]

    result = _analyze(provider, SceneClassifierRunOptions())

    assert calls == []
    assert result.debug["pipeline_mode"] == "classification_only"
    assert result.classification_only is True
    assert result.resolved_items == []
    assert "pass_2d_per_observation" not in result.debug


def test_benchmark_mode_resolves_but_stays_non_publishable():
    result = _analyze(_two_arg)

    assert result.debug["pipeline_mode"] == BENCHMARK_MODE
    assert [row["resolved_item_id"] for row in result.resolved_items] == [
        "roof_shingles_aged_or_worn"
    ]
    assert result.resolved_items[0]["resolved_kind"] == "degradation"
    # The freeze holds in benchmark mode too — write_photo_intel still refuses.
    assert result.classification_only is True


def test_benchmark_mode_runs_deterministic_2e():
    """Task 3 revived 2e: benchmark mode continues past 2d into the rule-based
    2e, which promotes resolved observations into the verified lanes. The
    result stays classification_only, so write_photo_intel still refuses it."""
    result = _analyze(_two_arg)

    assert '2e' in result.passes_run
    assert len(result.verified_issues) == 1
    issue = result.verified_issues[0]
    assert issue["kind"] == "degradation"
    assert issue["catalogItemId"] == "roof_shingles_aged_or_worn"
    assert issue["catalogItemKind"] == "degradation"
    assert result.canonical_issues == result.matched_issues
    assert result.classification_only is True


def test_benchmark_mode_gate_counts_are_three_kind():
    gate = _analyze(_two_arg).debug["pass_2d_gate"]
    assert gate["by_kind"] == {"defect": 0, "degradation": 1, "modernization": 0}
    assert gate["observation_count"] == 1

    summary = _analyze(_two_arg).debug["pass_2d_summary"]
    assert summary["resolved_by_kind"] == {"defect": 0, "degradation": 1, "modernization": 0}
    assert summary["resolved_total"] == 1


def test_unknown_pipeline_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported pipeline_mode"):
        _analyze(_two_arg, SceneClassifierRunOptions(pipeline_mode="publish_everything"))


def test_invalid_observation_kind_fails_before_retrieval():
    """A kind outside the ontology must fail closed with zero provider calls —
    the empty allowed_kinds set is exactly the input that used to be read as
    'no filter' and search the whole catalog."""
    calls = []

    class RetiredKindClient(FakeOrchestratorClient):
        async def analyze_text(self, system_prompt, user_prompt, **model_config):
            if "classify each numbered observation" in (system_prompt or "").lower():
                return '{"decisions":[{"index":1,"kind":"degradation"}]}'
            return await super().analyze_text(system_prompt, user_prompt, **model_config)

    def provider(description, context):
        calls.append(description)
        return [dict(CANDIDATE)]

    orchestrator = SceneClassifierOrchestrator(
        qwen_config={}, gpt5_config={},
        vlm_client=RetiredKindClient(),
        candidate_provider=provider,
        top_k_candidates=5,
    )
    # Force the retired v1 kind onto the observation after 2c classification.
    original = orchestrator_module.evaluate_kind_routing
    try:
        orchestrator_module.evaluate_kind_routing = lambda desc, kind: original(desc, "upgrade")
        with pytest.raises(PassExecutionError) as excinfo:
            asyncio.run(orchestrator.analyze_image(
                image_path=Path("photo_002.jpg"), options=_benchmark_options(),
            ))
    finally:
        orchestrator_module.evaluate_kind_routing = original

    assert excinfo.value.pass_key == "2d"
    assert excinfo.value.code == "invalid_kind"
    assert calls == []


def test_off_kind_candidate_is_a_purity_violation():
    """A provider leaking a candidate of another kind means the filter is
    broken; that must be loud, not silently reranked."""
    def provider(description, context):
        return [dict(CANDIDATE, kind="defect")]

    with pytest.raises(PassExecutionError) as excinfo:
        _analyze(provider)
    assert excinfo.value.pass_key == "2d"
    assert excinfo.value.code == "kind_purity_violation"
