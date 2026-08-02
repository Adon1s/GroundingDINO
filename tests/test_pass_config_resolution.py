"""
Tests for the unified OpenAI invocation resolver (pass_config.resolve_openai_invocation).

The regressions guarded here:
  - 1a and 2f were absent from the old hand-maintained key_map, so 1a fell back
    to the global default and 2f never reached the cap helper at all.
  - 2f's larger multi-image budget must outrank the catch-all default.
  - An unparseable cap used to log a warning and run uncapped.
"""
import pytest

from tools.pass_config import (
    OPENAI_MAX_TOKENS_DEFAULTS,
    OPENAI_REASONING_DEFAULTS,
    REASONING_PASS_KEYS,
    resolve_openai_invocation,
)


# ── token budget ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pass_key,expected", [
    ("1a", 2000), ("2a", 2000), ("2b", 2000), ("2c", 2000), ("2d", 2000),
    ("2f", 4096),
])
def test_per_pass_default_cap(pass_key, expected, openai_config, clean_token_env):
    resolved = resolve_openai_invocation(pass_key, dict(openai_config))
    assert resolved["max_output_tokens"] == expected
    # VLMClient binds max_tokens as a named param; max_output_tokens is the
    # Responses payload key. Both must be set or the cap silently vanishes.
    assert resolved["max_tokens"] == expected


def test_pass_1a_and_2f_both_receive_a_cap(openai_config, clean_token_env):
    """The exact gap in the old key_map: neither pass got one."""
    for pass_key in ("1a", "2f"):
        assert "max_output_tokens" in resolve_openai_invocation(pass_key, dict(openai_config))


def test_2f_budget_beats_the_catchall_default(openai_config, clean_token_env):
    """
    pipeline_config bakes in OPENAI_DEFAULT_MAX_TOKENS=2000. If the resolver
    honoured that attribute, 2f would silently lose its 4096 multi-image budget.
    """
    assert resolve_openai_invocation("2f", dict(openai_config))["max_output_tokens"] == 4096


def test_env_precedence(openai_config, clean_token_env, monkeypatch):
    monkeypatch.setenv("OPENAI_DEFAULT_MAX_TOKENS", "1234")
    assert resolve_openai_invocation("2f", dict(openai_config))["max_output_tokens"] == 1234

    monkeypatch.setenv("OPENAI_PASS_2F_MAX_TOKENS", "777")
    assert resolve_openai_invocation("2f", dict(openai_config))["max_output_tokens"] == 777
    # per-pass override must not leak to other passes
    assert resolve_openai_invocation("2a", dict(openai_config))["max_output_tokens"] == 1234


def test_explicit_config_value_wins(openai_config, clean_token_env):
    cfg = {**openai_config, "max_output_tokens": 50}
    assert resolve_openai_invocation("2f", cfg)["max_output_tokens"] == 50


@pytest.mark.parametrize("bad", ["oops", 0, -5])
def test_invalid_cap_raises_instead_of_running_uncapped(bad, openai_config, clean_token_env):
    with pytest.raises(ValueError):
        resolve_openai_invocation("2f", {**openai_config, "max_output_tokens": bad})


def test_invalid_env_cap_raises(openai_config, clean_token_env, monkeypatch):
    monkeypatch.setenv("OPENAI_PASS_2A_MAX_TOKENS", "not-a-number")
    with pytest.raises(ValueError, match="OPENAI_PASS_2A_MAX_TOKENS"):
        resolve_openai_invocation("2a", dict(openai_config))


# ── provider gating ─────────────────────────────────────────────────────────

def test_local_provider_is_untouched(qwen_config, clean_token_env):
    """LM Studio calls must not acquire OpenAI-only knobs."""
    assert resolve_openai_invocation("2a", dict(qwen_config)) == qwen_config


def test_api_key_without_provider_is_treated_as_openai(clean_token_env):
    resolved = resolve_openai_invocation("1a", {"model": "gpt-5.4", "api_key": "k"})
    assert resolved["max_output_tokens"] == 2000


# ── reasoning effort ────────────────────────────────────────────────────────

def test_default_reasoning_efforts(openai_config, clean_token_env):
    for pass_key in ("1a", "2a", "2b", "2c", "2d"):
        assert resolve_openai_invocation(pass_key, dict(openai_config))["reasoning_effort"] == "low"
    assert resolve_openai_invocation("2f", dict(openai_config))["reasoning_effort"] == "medium"


def test_explicit_effort_overrides_default(openai_config, clean_token_env):
    resolved = resolve_openai_invocation("2f", dict(openai_config), "high")
    assert resolved["reasoning_effort"] == "high"


def test_effort_omitted_for_non_reasoning_model(openai_config, clean_token_env):
    """A non-GPT-5 model rejects the reasoning param, so it must not be sent."""
    resolved = resolve_openai_invocation("2f", {**openai_config, "model": "o3-mini"}, "high")
    assert "reasoning_effort" not in resolved


def test_invalid_effort_rejected(openai_config, clean_token_env):
    with pytest.raises(ValueError, match="unsupported reasoning effort"):
        resolve_openai_invocation("2f", dict(openai_config), "ultra")


def test_2e_has_no_reasoning_default():
    """2e is rule-based with no LLM call."""
    assert "2e" not in OPENAI_REASONING_DEFAULTS
    assert "2e" not in REASONING_PASS_KEYS
    assert "2e" not in OPENAI_MAX_TOKENS_DEFAULTS


def test_reasoning_defaults_are_valid_pass_keys():
    assert set(OPENAI_REASONING_DEFAULTS) <= REASONING_PASS_KEYS


# ── the cap must reach the wire, not just the config dict ───────────────────

def test_2f_cap_reaches_the_multi_image_request(tmp_path, clean_token_env, monkeypatch, fake_openai):
    """
    The headline regression: Pass 2f bypassed the orchestrator's cap helper
    entirely and silently inherited VLMClient.default_max_tokens (4096), so
    OPENAI_PASS_2F_MAX_TOKENS had no effect on the real request.
    """
    import asyncio

    from tools.pass_config import resolve_openai_invocation
    from tools.scene_classifier_passes import run_pass_2f
    from tools.vlm_client import VLMClient

    monkeypatch.setenv("OPENAI_PASS_2F_MAX_TOKENS", "321")

    image = tmp_path / "kitchen.jpg"
    image.write_bytes(b"image-bytes")

    fake = fake_openai(responses=[openai_2f_response()])
    client = VLMClient()
    fake.attach(client)

    model_config = resolve_openai_invocation(
        "2f", {"provider": "openai", "model": "gpt-5.4-mini", "api_key": "k"}
    )
    asyncio.run(run_pass_2f(
        image_paths=[image],
        vlm_client=client,
        model_config=model_config,
        room="kitchen",
        package_id="pkg_1",
        package_type="kitchen_modernization",
        evidence_items=[{"catalog_item_id": "c1", "issue_ids": ["issue_1"]}],
    ))

    assert fake.last_request["max_output_tokens"] == 321
    assert fake.last_request["reasoning"] == {"effort": "medium"}


def openai_2f_response():
    """A schema-valid Pass 2f verdict."""
    import json

    from tests.conftest import openai_response
    return openai_response(text=json.dumps({
        "verification_status": "confirmed",
        "confirmed_issue_ids": ["issue_1"],
        "rejected_issue_ids": [],
        "evidence_summary": "Dated cabinets visible across both photos.",
    }))
