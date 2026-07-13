"""Profile and environment configuration for the model-comparison tools."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROFILE_SCHEMA_VERSION = 1
COMPARISON_SCHEMA_VERSION = 2
SUPPORTED_PROVIDERS = {"openai", "lmstudio"}
DEFAULT_SKILLS = ("2a", "2b", "2c", "2d_isolated", "2c+2d_coupled")
DEFAULT_LOCAL_MODEL = "unsloth/qwen3.6-27b@q6_k"


def local_model_default() -> str:
    """Return the contestant default used only for an explicit LM Studio role."""
    return os.environ.get("MODEL_COMPARISON_LOCAL_MODEL", "").strip() or DEFAULT_LOCAL_MODEL


def load_dotenv_without_override(path: Path) -> None:
    """Load the small KEY=VALUE subset used by this repository's .env file.

    Exported process variables always win. Values may be quoted; blank lines and
    comments are ignored. Keeping this loader local avoids making comparison
    scripts depend on IDE-specific dotenv support.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    provider: str
    model: str
    url_env: Optional[str] = None
    timeout_seconds: int = 360
    manual_load: bool = False
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None

    def validate(self, role: str) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"{role}.provider must be one of {sorted(SUPPORTED_PROVIDERS)}, "
                f"got {self.provider!r}"
            )
        if not self.model.strip():
            raise ValueError(f"{role}.model resolved to an empty value")
        if self.timeout_seconds <= 0:
            raise ValueError(f"{role}.timeout_seconds must be positive")
        if self.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(f"{role} uses openai but OPENAI_API_KEY is not set")
        if self.provider == "lmstudio":
            env_name = self.url_env or "LM_STUDIO_URL"
            if not os.environ.get(env_name):
                raise ValueError(f"{role} uses lmstudio but {env_name} is not set")

    def to_vlm_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "model": self.model,
            "provider": self.provider,
            "timeout": self.timeout_seconds,
        }
        if self.provider == "openai":
            config["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        else:
            config["url"] = os.environ.get(self.url_env or "LM_STUDIO_URL", "")
        if self.reasoning_effort:
            config["reasoning_effort"] = self.reasoning_effort
        if self.verbosity:
            config["verbosity"] = self.verbosity
        return config

    def redacted_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.provider == "openai":
            result["credential_env"] = "OPENAI_API_KEY"
            result["base_url_env"] = "OPENAI_BASE_URL"
        else:
            result["base_url_env"] = self.url_env or "LM_STUDIO_URL"
        return {k: v for k, v in result.items() if v is not None}


@dataclass(frozen=True)
class RunSettings:
    concurrency: int = 3
    judge_delay: float = 2.0
    skills: tuple[str, ...] = DEFAULT_SKILLS
    confirm_every: int = 0


@dataclass(frozen=True)
class ComparisonConfig:
    name: str
    model_a: ModelSpec
    model_b: ModelSpec
    fixture: ModelSpec
    judge: ModelSpec
    run: RunSettings = field(default_factory=RunSettings)
    source_path: Optional[str] = None

    def validate(self, *, require_credentials: bool = True) -> list[str]:
        warnings: list[str] = []
        for role, spec in self.roles().items():
            if require_credentials:
                spec.validate(role)
            else:
                if spec.provider not in SUPPORTED_PROVIDERS or not spec.model.strip():
                    spec.validate(role)
        if self.model_a.provider == self.model_b.provider and self.model_a.model == self.model_b.model:
            raise ValueError("model_a and model_b resolve to the same provider/model")
        if self.run.concurrency <= 0:
            raise ValueError("run.concurrency must be positive")
        unknown = sorted(set(self.run.skills) - set(DEFAULT_SKILLS))
        if unknown:
            raise ValueError(f"run.skills contains unknown skills: {unknown}")
        contestants = {
            (self.model_a.provider, self.model_a.model),
            (self.model_b.provider, self.model_b.model),
        }
        for role in ("fixture", "judge"):
            spec = getattr(self, role)
            if (spec.provider, spec.model) in contestants:
                warnings.append(
                    f"{role} uses contestant model {spec.model!r}; judging remains blinded "
                    "but same-model preference is possible"
                )
        return warnings

    def roles(self) -> Dict[str, ModelSpec]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "fixture": self.fixture,
            "judge": self.judge,
        }

    def redacted_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "name": self.name,
            "source_path": self.source_path,
            "models": {
                "model_a": self.model_a.redacted_dict(),
                "model_b": self.model_b.redacted_dict(),
            },
            "fixture": self.fixture.redacted_dict(),
            "judge": self.judge.redacted_dict(),
            "run": {
                "concurrency": self.run.concurrency,
                "judge_delay": self.run.judge_delay,
                "skills": list(self.run.skills),
                "confirm_every": self.run.confirm_every,
            },
        }

    def fingerprint(self) -> str:
        semantic = self.redacted_dict()
        semantic.pop("source_path", None)
        payload = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def profile_directory(project_root: Path) -> Path:
    return project_root / "configs" / "model_comparison"


def list_profiles(project_root: Path) -> list[Path]:
    directory = profile_directory(project_root)
    return sorted(directory.glob("*.json")) if directory.exists() else []


def resolve_profile_path(reference: str, project_root: Path) -> Path:
    candidate = Path(reference)
    if candidate.exists():
        return candidate.resolve()
    name = reference if reference.lower().endswith(".json") else f"{reference}.json"
    candidate = profile_directory(project_root) / name
    if candidate.exists():
        return candidate.resolve()
    available = ", ".join(p.stem for p in list_profiles(project_root)) or "none"
    raise ValueError(f"Unknown comparison profile {reference!r}; available profiles: {available}")


def _resolve_model_value(raw: Dict[str, Any], role: str, *, fallback_env: Optional[str] = None) -> str:
    direct = str(raw.get("model") or "").strip()
    if direct:
        return direct
    env_name = str(raw.get("model_env") or fallback_env or "").strip()
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    raise ValueError(f"{role} requires either model or a populated model_env")


def _model_spec(
    raw: Dict[str, Any],
    role: str,
    *,
    fallback_model_env: Optional[str] = None,
    fallback_model: Optional[str] = None,
) -> ModelSpec:
    provider = str(raw.get("provider") or "openai").strip().lower()
    try:
        model = _resolve_model_value(raw, role, fallback_env=fallback_model_env)
    except ValueError:
        if not fallback_model:
            raise
        model = fallback_model
    default_label = role.replace("_", " ").title()
    return ModelSpec(
        label=str(raw.get("label") or default_label),
        provider=provider,
        model=model,
        url_env=str(raw.get("url_env") or "").strip() or None,
        timeout_seconds=int(raw.get("timeout_seconds") or 360),
        manual_load=bool(raw.get("manual_load", provider == "lmstudio")),
        reasoning_effort=str(raw.get("reasoning_effort") or "").strip() or None,
        verbosity=str(raw.get("verbosity") or "").strip() or None,
    )


def load_comparison_profile(
    reference: str,
    project_root: Path,
    model_overrides: Optional[Dict[str, str]] = None,
) -> ComparisonConfig:
    path = resolve_profile_path(reference, project_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read comparison profile {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Comparison profile {path} must contain a JSON object")
    if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Comparison profile schema must be {PROFILE_SCHEMA_VERSION}, "
            f"got {raw.get('schema_version')!r}"
        )
    models = raw.get("models") or {}
    model_overrides = {key: value for key, value in (model_overrides or {}).items() if value}
    if not isinstance(models, dict):
        raise ValueError("models must be an object with model_a and model_b")
    model_a_raw = dict(models.get("model_a") or {})
    model_b_raw = dict(models.get("model_b") or {})
    judge_raw = dict(raw.get("judge") or {})
    fixture_raw = dict(raw.get("fixture") or {})
    if model_overrides.get("model_a"):
        model_a_raw["model"] = model_overrides["model_a"]
    if model_overrides.get("model_b"):
        model_b_raw["model"] = model_overrides["model_b"]
    if model_overrides.get("judge"):
        judge_raw["model"] = model_overrides["judge"]
    if model_overrides.get("fixture"):
        fixture_raw["model"] = model_overrides["fixture"]
    model_a = _model_spec(model_a_raw, "model_a")
    model_b = _model_spec(model_b_raw, "model_b")
    judge = _model_spec(
        judge_raw,
        "judge",
        fallback_model_env="MODEL_COMPARISON_JUDGE_MODEL",
    )
    fixture = _model_spec(
        fixture_raw,
        "fixture",
        fallback_model_env="MODEL_COMPARISON_FIXTURE_MODEL",
        fallback_model=judge.model,
    )
    run_raw = raw.get("run") or {}
    skills_raw: Iterable[str] = run_raw.get("skills") or DEFAULT_SKILLS
    run = RunSettings(
        concurrency=int(run_raw.get("concurrency") or 3),
        judge_delay=float(run_raw.get("judge_delay") if run_raw.get("judge_delay") is not None else 2.0),
        skills=tuple(str(x) for x in skills_raw),
        confirm_every=int(run_raw.get("confirm_every") or 0),
    )
    return ComparisonConfig(
        name=str(raw.get("name") or path.stem),
        model_a=model_a,
        model_b=model_b,
        fixture=fixture,
        judge=judge,
        run=run,
        source_path=str(path),
    )


def with_cli_overrides(config: ComparisonConfig, args: Any) -> ComparisonConfig:
    def override(spec: ModelSpec, prefix: str) -> ModelSpec:
        values = asdict(spec)
        model = getattr(args, prefix, None)
        provider = getattr(args, f"provider_{prefix[-1]}", None) if prefix.startswith("model_") else None
        if model:
            values["model"] = model
        if provider:
            if provider == "lmstudio" and spec.provider != "lmstudio" and not model:
                values["model"] = local_model_default()
            values["provider"] = provider
            values["manual_load"] = provider == "lmstudio"
        return ModelSpec(**values)

    model_a = override(config.model_a, "model_a")
    model_b = override(config.model_b, "model_b")
    fixture_values = asdict(config.fixture)
    judge_values = asdict(config.judge)
    if getattr(args, "fixture_model", None):
        fixture_values["model"] = args.fixture_model
    if getattr(args, "judge_model", None):
        judge_values["model"] = args.judge_model
    if getattr(args, "judge_reasoning_effort", None):
        judge_values["reasoning_effort"] = args.judge_reasoning_effort
    if getattr(args, "judge_verbosity", None):
        judge_values["verbosity"] = args.judge_verbosity
    run = RunSettings(
        concurrency=getattr(args, "concurrency", None) or config.run.concurrency,
        judge_delay=(
            args.judge_delay if getattr(args, "judge_delay", None) is not None else config.run.judge_delay
        ),
        skills=tuple(getattr(args, "skills", None) or config.run.skills),
        confirm_every=(
            args.confirm_every if getattr(args, "confirm_every", None) is not None else config.run.confirm_every
        ),
    )
    return ComparisonConfig(
        name=config.name,
        model_a=model_a,
        model_b=model_b,
        fixture=ModelSpec(**fixture_values),
        judge=ModelSpec(**judge_values),
        run=run,
        source_path=config.source_path,
    )
