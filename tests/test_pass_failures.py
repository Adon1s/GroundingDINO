"""
Tests for the pass-failure contract.

The invariant: an enabled pass may return a valid result or raise, but must
never return an empty/fabricated result that reads downstream as "no findings".

Historically every pass swallowed exceptions -- 1a returned scene="other",
2a returned observations_freeform="", 2b returned observations=[] -- so a
provider outage was indistinguishable from a clean photo.
"""
import asyncio
import json
from pathlib import Path

import pytest

from tools.pass_config import SceneClassifierRunOptions
from tools.scene_classifier_orchestrator import SceneClassifierOrchestrator
from tools.scene_classifier_passes import (
    PassExecutionError,
    run_pass_1a_scene_type,
    run_pass_2a,
    run_pass_2b,
    run_pass_2c,
    run_pass_2d,
)
from tools.vlm_client import (
    OpenAIEmptyResponse,
    OpenAIIncompleteResponse,
    OpenAIRefusal,
    VLMClient,
)

from tests.conftest import openai_response


OPENAI_CFG = {"provider": "openai", "model": "gpt-5.4-mini", "api_key": "k"}


class RaisingClient:
    """VLM client whose every call fails the same way."""

    def __init__(self, exc):
        self.exc = exc

    async def analyze_image(self, **kwargs):
        raise self.exc

    async def analyze_text(self, **kwargs):
        raise self.exc

    async def analyze_images(self, **kwargs):
        raise self.exc


class TextClient:
    """VLM client returning a canned string."""

    def __init__(self, text):
        self.text = text

    async def analyze_image(self, **kwargs):
        return self.text

    async def analyze_text(self, **kwargs):
        return self.text


def _run_pass(pass_key, client):
    """Invoke a pass with the minimal arguments it requires."""
    if pass_key == "1a":
        return asyncio.run(run_pass_1a_scene_type(Path("p.jpg"), client, dict(OPENAI_CFG)))
    if pass_key == "2a":
        return asyncio.run(run_pass_2a(Path("p.jpg"), client, dict(OPENAI_CFG)))
    if pass_key == "2b":
        return asyncio.run(run_pass_2b(client, dict(OPENAI_CFG), "cracked tile in bathroom"))
    if pass_key == "2c":
        return asyncio.run(run_pass_2c(
            client, dict(OPENAI_CFG),
            [{"description": "cracked tile"}], "bathroom",
        ))
    if pass_key == "2d":
        return asyncio.run(run_pass_2d(
            observation="cracked tile",
            candidates=[{"item_id": "tile_cracked", "kind": "defect"}],
            vlm_client=client,
            model_config=dict(OPENAI_CFG),
            kind="defect",
        ))
    raise AssertionError(f"unhandled pass {pass_key}")


ALL_LLM_PASSES = ["1a", "2a", "2b", "2c", "2d"]


@pytest.mark.parametrize("pass_key", ALL_LLM_PASSES)
def test_provider_exception_raises(pass_key):
    with pytest.raises(PassExecutionError) as excinfo:
        _run_pass(pass_key, RaisingClient(RuntimeError("api down")))
    err = excinfo.value
    assert err.pass_key == pass_key
    assert err.stage == "request"
    assert err.code == "RuntimeError"
    assert err.provider == "openai"
    assert err.model == "gpt-5.4-mini"


@pytest.mark.parametrize("pass_key", ALL_LLM_PASSES)
@pytest.mark.parametrize("exc_type", [
    OpenAIIncompleteResponse, OpenAIRefusal, OpenAIEmptyResponse,
])
def test_incomplete_refusal_and_empty_all_raise(pass_key, exc_type):
    """
    Truncation, refusal, and empty output must stay distinguishable from each
    other and from a clean result. A truncated 2f response used to be reported
    as verification_status="uncertain".
    """
    with pytest.raises(PassExecutionError) as excinfo:
        _run_pass(pass_key, RaisingClient(exc_type("boom")))
    assert excinfo.value.code == exc_type.__name__


def test_pass_1a_never_fabricates_a_scene():
    """The worst of the old fallbacks: a failed 1a returned a real-looking label."""
    with pytest.raises(PassExecutionError):
        _run_pass("1a", RaisingClient(RuntimeError("api down")))


def test_valid_zero_finding_response_still_succeeds():
    """An explicit, structurally valid 'nothing here' is NOT an error."""
    result = _run_pass("2b", TextClient(json.dumps({"observations": []})))
    assert result.observations == []
    assert result.raw_response is not None


def test_pass_1a_valid_response_succeeds():
    # `confidence` is not requested by the prompt and no longer parsed; an
    # extra key in the response must simply be ignored.
    result = _run_pass("1a", TextClient(json.dumps({"scene": "kitchen", "confidence": 0.9})))
    assert result.scene == "kitchen"
    assert not hasattr(result, "confidence")


# ── provider error typing in vlm_client ─────────────────────────────────────

@pytest.mark.parametrize("response,expected", [
    (openai_response(status="incomplete", incomplete_reason="max_output_tokens"),
     OpenAIIncompleteResponse),
    (openai_response(refusal="I can't help with that"), OpenAIRefusal),
    (openai_response(text=None), OpenAIEmptyResponse),
])
def test_extract_output_text_raises_typed_errors(response, expected):
    with pytest.raises(expected):
        VLMClient()._extract_openai_output_text(response)


def test_incomplete_error_names_the_reason():
    """max_output_tokens truncation must be attributable to the token cap."""
    with pytest.raises(OpenAIIncompleteResponse, match="max_output_tokens"):
        VLMClient()._extract_openai_output_text(
            openai_response(status="incomplete", incomplete_reason="max_output_tokens")
        )


# ── orchestrator failure boundary ───────────────────────────────────────────

def _orchestrator(client):
    local = {"provider": "lmstudio", "model": "qwen", "url": "http://localhost:1234"}
    return SceneClassifierOrchestrator(
        qwen_config=local, gpt5_config=local, vlm_client=client,
    )


def test_collect_mode_records_error_and_marks_pass_failed():
    orch = _orchestrator(RaisingClient(RuntimeError("provider exploded")))
    options = SceneClassifierRunOptions.from_analysis_profile(
        "standard", failure_mode="collect",
    )
    result = asyncio.run(orch.analyze_image(Path("p.jpg"), options))

    assert result.pass_states["1a"] == "failed"
    errors = result.debug["pass_errors"]
    assert len(errors) == 1
    assert errors[0]["pass"] == "1a"
    assert errors[0]["stage"] == "request"


def test_strict_is_the_default_and_propagates():
    options = SceneClassifierRunOptions.from_analysis_profile("standard")
    assert options.failure_mode == "strict"

    orch = _orchestrator(RaisingClient(RuntimeError("provider exploded")))
    with pytest.raises(PassExecutionError) as excinfo:
        asyncio.run(orch.analyze_image(Path("p.jpg"), options))
    # the partial result rides along so the failed image stays diagnosable
    assert excinfo.value.partial_result["pass_states"]["1a"] == "failed"


def test_dependent_passes_do_not_run_after_a_failure():
    orch = _orchestrator(RaisingClient(RuntimeError("provider exploded")))
    options = SceneClassifierRunOptions.from_analysis_profile(
        "standard", failure_mode="collect",
    )
    result = asyncio.run(orch.analyze_image(Path("p.jpg"), options))
    # 1a failed, so nothing downstream should claim to have executed
    assert "2a" not in result.passes_run
    assert result.pass_states["2a"] == "skipped"


def test_invalid_failure_mode_rejected():
    with pytest.raises(ValueError, match="unsupported failure_mode"):
        SceneClassifierRunOptions.from_analysis_profile("standard", failure_mode="lenient")


# ── options plumbing ────────────────────────────────────────────────────────

def test_with_meta_preserves_failure_mode():
    """
    with_meta used to rebuild the dataclass field-by-field, so any newly added
    field was silently dropped.
    """
    options = SceneClassifierRunOptions.from_analysis_profile(
        "standard", failure_mode="collect",
    )
    copied = options.with_meta(run_id="r1", photo_key="p.jpg")
    assert copied.failure_mode == "collect"
    assert copied.meta == {"run_id": "r1", "photo_key": "p.jpg"}


def test_pass_toggles_reach_the_orchestrator():
    """
    _t guessed attribute names ("2d"/"p2d"/"_2d") but the field is pass_2d, so
    every CLI/server toggle was silently ignored.
    """
    options = SceneClassifierRunOptions.from_analysis_profile(
        "standard", toggles={"2d": False, "2a": False},
    )
    assert SceneClassifierOrchestrator._t(options.toggles, "2d") is False
    assert SceneClassifierOrchestrator._t(options.toggles, "2a") is False
    assert SceneClassifierOrchestrator._t(options.toggles, "1a") is True
