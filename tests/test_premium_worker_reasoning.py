import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from tools.analyzer_cli import _build_reasoning_efforts
from tools.analyzer_server import (
    _checkpoint_policy_fingerprint,
    _prepare_checkpoint_dir,
)
from tools.pass_config import (
    SceneClassifierRunOptions,
    get_model_config_for_pass,
    normalize_reasoning_efforts,
)
from tools.vlm_client import VLMClient


def test_reasoning_efforts_accept_none_and_reject_invalid_values():
    assert normalize_reasoning_efforts({"1a": "NONE", "2f": "medium"}) == {
        "1a": "none",
        "2f": "medium",
    }
    with pytest.raises(ValueError, match="unsupported reasoning effort"):
        normalize_reasoning_efforts({"2f": "ultra"})
    with pytest.raises(ValueError, match="unsupported reasoning-effort pass key"):
        normalize_reasoning_efforts({"2e": "none"})


def test_cli_reasoning_map_is_strict_and_preserves_none():
    args = Namespace(reasoning_map=json.dumps({"1a": "none", "2f": "medium"}))
    assert _build_reasoning_efforts(args) == {"1a": "none", "2f": "medium"}

    with pytest.raises(SystemExit, match="unsupported effort"):
        _build_reasoning_efforts(Namespace(reasoning_map='{"2f":"ultra"}'))


def test_model_config_attaches_explicit_reasoning_effort():
    options = SceneClassifierRunOptions.from_analysis_profile(
        "standard",
        model_overrides={"1a": "gpt-5.6-terra", "2f": "gpt-5.6-sol"},
        reasoning_efforts={"1a": "none", "2f": "medium"},
    )
    qwen = {"provider": "lmstudio", "model": "local"}
    openai = {"provider": "openai", "model": "base", "api_key": "test"}

    assert get_model_config_for_pass("1a", options, qwen, openai) == {
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "api_key": "test",
        "reasoning_effort": "none",
        # resolve_openai_invocation now also applies the per-pass token budget
        "max_tokens": 2000,
        "max_output_tokens": 2000,
    }
    assert get_model_config_for_pass("2f", options, qwen, openai)["reasoning_effort"] == "medium"
    # 2f gets the larger multi-image budget, not the 1a-2d default
    assert get_model_config_for_pass("2f", options, qwen, openai)["max_output_tokens"] == 4096


def test_vlm_client_sends_none_to_responses_api(monkeypatch):
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="ok", usage=None)

    client = VLMClient()
    monkeypatch.setattr(
        client,
        "_get_openai_client",
        lambda _api_key: SimpleNamespace(responses=Responses()),
    )

    result = asyncio.run(client._analyze_text_openai(
        system_prompt="system",
        user_prompt="user",
        model="gpt-5.6-terra",
        api_key="test",
        max_tokens=100,
        reasoning_effort="none",
    ))
    assert result == "ok"
    assert captured["reasoning"] == {"effort": "none"}


def test_checkpoint_policy_change_discards_partial_results(tmp_path):
    checkpoint_dir = tmp_path / "run"
    first = _checkpoint_policy_fingerprint(
        {"1a": "gpt-5.6-terra", "2f": "gpt-5.6-sol"},
        {"1a": "none", "2f": "medium"},
    )
    same_different_order = _checkpoint_policy_fingerprint(
        {"2f": "gpt-5.6-sol", "1a": "gpt-5.6-terra"},
        {"2f": "medium", "1a": "none"},
    )
    assert same_different_order == first

    _prepare_checkpoint_dir(checkpoint_dir, first)
    image_checkpoint = checkpoint_dir / "image_0000.json"
    image_checkpoint.write_text("{}", encoding="utf-8")
    _prepare_checkpoint_dir(checkpoint_dir, first)
    assert image_checkpoint.exists()

    normal_policy = _checkpoint_policy_fingerprint(
        {"2f": "gpt-5.6-terra"},
        {"2f": "medium"},
    )
    _prepare_checkpoint_dir(checkpoint_dir, normal_policy)
    assert not image_checkpoint.exists()
    assert json.loads((checkpoint_dir / "policy.json").read_text(encoding="utf-8")) == {
        "policy_fingerprint": normal_policy,
    }
def test_orchestrator_factory_reuses_supplied_job_client(monkeypatch):
    from tools import vlm_client as vlm_module
    from tools.scene_classifier_orchestrator import create_orchestrator_from_config

    supplied = object()
    monkeypatch.setattr(
        vlm_module,
        "get_model_configs_from_pipeline_config",
        lambda _config: (
            {"provider": "lmstudio", "model": "local"},
            {"provider": "openai", "model": "cloud"},
        ),
    )

    orchestrator = create_orchestrator_from_config(
        SimpleNamespace(TOP_K_CANDIDATES=8, MAX_RESOLVE_PER_IMAGE=25),
        vlm_client=supplied,
    )

    assert orchestrator.vlm_client is supplied
