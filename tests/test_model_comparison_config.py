import argparse
import json
from pathlib import Path

import pytest

from tools.model_comparison import parse_args

from tools.model_comparison_config import (
    COMPARISON_SCHEMA_VERSION,
    DEFAULT_LOCAL_MODEL,
    ComparisonConfig,
    ModelSpec,
    RunSettings,
    load_comparison_profile,
    load_dotenv_without_override,
    with_cli_overrides,
)


def _write_profile(root: Path, model_a: dict, model_b: dict) -> Path:
    directory = root / "configs" / "model_comparison"
    directory.mkdir(parents=True)
    path = directory / "test.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "name": "test_pair",
        "models": {"model_a": model_a, "model_b": model_b},
        "fixture": {"provider": "openai", "model_env": "MODEL_COMPARISON_FIXTURE_MODEL"},
        "judge": {"provider": "openai", "model_env": "MODEL_COMPARISON_JUDGE_MODEL"},
        "run": {"concurrency": 2, "skills": ["2a", "2b"]},
    }), encoding="utf-8")
    return path


@pytest.fixture
def provider_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "top-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:1234")
    monkeypatch.setenv("MODEL_COMPARISON_JUDGE_MODEL", "neutral-judge")
    monkeypatch.setenv("MODEL_COMPARISON_FIXTURE_MODEL", "neutral-fixture")


def test_dotenv_does_not_override_exported_environment(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("KEEP=from-file\nNEW_VALUE='loaded'\n", encoding="utf-8")
    monkeypatch.setenv("KEEP", "from-shell")
    monkeypatch.delenv("NEW_VALUE", raising=False)

    load_dotenv_without_override(path)

    assert __import__("os").environ["KEEP"] == "from-shell"
    assert __import__("os").environ["NEW_VALUE"] == "loaded"


@pytest.mark.parametrize(
    ("provider_a", "provider_b"),
    [("openai", "openai"), ("lmstudio", "openai"), ("lmstudio", "lmstudio")],
)
def test_supported_provider_pairings(tmp_path, provider_env, provider_a, provider_b):
    path = _write_profile(
        tmp_path,
        {"label": "A", "provider": provider_a, "model": "candidate-a"},
        {"label": "B", "provider": provider_b, "model": "candidate-b"},
    )

    config = load_comparison_profile(str(path), tmp_path)

    assert config.validate() == []
    assert config.model_a.to_vlm_config()["provider"] == provider_a
    assert config.model_b.to_vlm_config()["provider"] == provider_b


def test_cli_overrides_profile_and_redacts_secrets(tmp_path, provider_env):
    path = _write_profile(
        tmp_path,
        {"label": "A", "provider": "lmstudio", "model": "local-a"},
        {"label": "B", "provider": "openai", "model": "cloud-b"},
    )
    config = load_comparison_profile(str(path), tmp_path)
    args = argparse.Namespace(
        model_a="cloud-a",
        provider_a="openai",
        model_b=None,
        provider_b=None,
        fixture_model=None,
        judge_model="judge-override",
        judge_reasoning_effort="high",
        judge_verbosity="low",
        concurrency=5,
        judge_delay=0.5,
        skills=["2a"],
        confirm_every=4,
    )

    resolved = with_cli_overrides(config, args)
    rendered = json.dumps(resolved.redacted_dict())

    assert resolved.model_a.provider == "openai"
    assert resolved.model_a.model == "cloud-a"
    assert resolved.judge.model == "judge-override"
    assert resolved.run == RunSettings(5, 0.5, ("2a",), 4)
    assert "top-secret" not in rendered
    assert "https://example.invalid" not in rendered
    assert resolved.fingerprint() == resolved.fingerprint()
    assert COMPARISON_SCHEMA_VERSION == 2

@pytest.mark.parametrize("candidate", ("a", "b"))
def test_switching_either_candidate_to_lmstudio_uses_local_default(
    tmp_path, provider_env, monkeypatch, candidate
):
    monkeypatch.delenv("MODEL_COMPARISON_LOCAL_MODEL", raising=False)
    path = _write_profile(
        tmp_path,
        {"label": "A", "provider": "openai", "model": "cloud-a"},
        {"label": "B", "provider": "openai", "model": "cloud-b"},
    )
    config = load_comparison_profile(str(path), tmp_path)
    args = argparse.Namespace(
        model_a=None,
        provider_a="lmstudio" if candidate == "a" else None,
        model_b=None,
        provider_b="lmstudio" if candidate == "b" else None,
        fixture_model=None,
        judge_model=None,
        judge_reasoning_effort=None,
        judge_verbosity=None,
        concurrency=None,
        judge_delay=None,
        skills=None,
        confirm_every=None,
    )

    resolved = with_cli_overrides(config, args)

    assert getattr(resolved, f"model_{candidate}").model == DEFAULT_LOCAL_MODEL
    other = "b" if candidate == "a" else "a"
    assert getattr(resolved, f"model_{other}").model == f"cloud-{other}"


def test_local_default_environment_override_is_lmstudio_only(tmp_path, provider_env, monkeypatch):
    monkeypatch.setenv("MODEL_COMPARISON_LOCAL_MODEL", "local-from-env")
    path = _write_profile(
        tmp_path,
        {"label": "A", "provider": "openai", "model": "cloud-a"},
        {"label": "B", "provider": "openai", "model": "cloud-b"},
    )
    config = load_comparison_profile(str(path), tmp_path)
    args = argparse.Namespace(
        model_a=None,
        provider_a="lmstudio",
        model_b=None,
        provider_b=None,
        fixture_model=None,
        judge_model=None,
        judge_reasoning_effort=None,
        judge_verbosity=None,
        concurrency=None,
        judge_delay=None,
        skills=None,
        confirm_every=None,
    )

    resolved = with_cli_overrides(config, args)

    assert resolved.model_a.model == "local-from-env"
    assert resolved.model_b.model == "cloud-b"

def test_same_contestant_is_rejected(provider_env):
    spec = ModelSpec("same", "openai", "same-model")
    config = ComparisonConfig(
        name="bad",
        model_a=spec,
        model_b=spec,
        fixture=ModelSpec("fixture", "openai", "fixture"),
        judge=ModelSpec("judge", "openai", "judge"),
    )
    with pytest.raises(ValueError, match="same provider/model"):
        config.validate()


def test_contestant_judge_emits_fairness_warning(provider_env):
    config = ComparisonConfig(
        name="warning",
        model_a=ModelSpec("A", "openai", "candidate-a"),
        model_b=ModelSpec("B", "openai", "candidate-b"),
        fixture=ModelSpec("fixture", "openai", "neutral"),
        judge=ModelSpec("judge", "openai", "candidate-a"),
    )
    warnings = config.validate()
    assert len(warnings) == 1
    assert "same-model preference" in warnings[0]


def test_cli_profile_and_one_off_overrides(provider_env):
    args = parse_args([
        "--profile", "sol_vs_terra",
        "--property", "redfin_123",
        "--model-b", "future-terra",
        "--concurrency", "7",
        "--skills", "2a", "2c",
        "--validate-config",
    ])
    assert args.comparison_config.model_a.model == "gpt-5.6-sol"
    assert args.comparison_config.model_b.model == "future-terra"
    assert args.concurrency == 7
    assert args.skip_skills == ["2b", "2c+2d_coupled", "2d_isolated"]
    assert "redfin_123_sol_vs_terra" in args.output

def test_cli_role_override_can_replace_missing_profile_environment(provider_env, monkeypatch):
    monkeypatch.delenv("MODEL_COMPARISON_JUDGE_MODEL")
    monkeypatch.delenv("MODEL_COMPARISON_FIXTURE_MODEL")
    args = parse_args([
        "--profile", "sol_vs_terra",
        "--judge-model", "cli-neutral",
        "--validate-config",
    ])
    assert args.comparison_config.judge.model == "cli-neutral"
    assert args.comparison_config.fixture.model == "cli-neutral"