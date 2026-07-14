"""Configuration for artifact-first Terra-vs-Sol Pass 2f comparisons."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from tools import pipeline_config
from tools.model_comparison_config import ModelSpec


PROFILE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 2
DEFAULT_REQUIRED_LOCAL_MODEL = "unsloth/qwen3.6-27b@q6_k"
DEFAULT_EXECUTIVE_MODEL = "gpt-5.6-sol"
_ALLOWED_ENV_KEYS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
}
_ALLOWED_REASONING = {None, "none", "low", "medium", "high", "xhigh"}
_ALLOWED_VERBOSITY = {None, "low", "medium", "high"}


def load_allowed_dotenv(path: Path) -> None:
    """Load only OpenAI credential/endpoint keys without overriding exports."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in _ALLOWED_ENV_KEYS:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class SourceRequirements:
    required_local_model: str = DEFAULT_REQUIRED_LOCAL_MODEL
    require_local_upstream_routing: bool = True


@dataclass(frozen=True)
class ExecutiveReviewSettings:
    enabled: bool
    model: ModelSpec
    only: str = "asymmetric_approvals"


@dataclass(frozen=True)
class Pass2fRunSettings:
    max_images_per_package: int = 3
    contestant_order: tuple[str, ...] = ("terra", "sol")
    include_qwen: bool = False
    fail_fast: bool = True
    strict: bool = True


@dataclass(frozen=True)
class Pass2fComparisonConfig:
    name: str
    source: SourceRequirements
    terra: ModelSpec
    sol: ModelSpec
    qwen: ModelSpec
    executive_review: ExecutiveReviewSettings
    run: Pass2fRunSettings
    source_path: Optional[str] = None

    def validate(self, *, require_credentials: bool = False) -> None:
        openai_specs = {
            "terra": self.terra,
            "sol": self.sol,
            "executive_review": self.executive_review.model,
        }
        for role, spec in openai_specs.items():
            if spec.provider != "openai":
                raise ValueError(f"{role}.provider must be 'openai'")
            if not spec.model.strip():
                raise ValueError(f"{role}.model must not be empty")
            if re.search(r"gpt[-_]?5\.5", spec.model, flags=re.IGNORECASE):
                raise ValueError("GPT-5.5 is forbidden in the Pass 2f comparison")
            if spec.reasoning_effort not in _ALLOWED_REASONING:
                raise ValueError(f"{role}.reasoning_effort is invalid")
            if spec.verbosity not in _ALLOWED_VERBOSITY:
                raise ValueError(f"{role}.verbosity is invalid")
            if spec.timeout_seconds <= 0:
                raise ValueError(f"{role}.timeout_seconds must be positive")
            if require_credentials:
                spec.validate(role)
        if self.qwen.provider != "lmstudio":
            raise ValueError("qwen.provider must be 'lmstudio'")
        if not self.qwen.model.strip():
            raise ValueError("qwen.model must not be empty")
        if self.qwen.timeout_seconds <= 0:
            raise ValueError("qwen.timeout_seconds must be positive")
        if self.terra.model == self.sol.model:
            raise ValueError("Terra and Sol must resolve to different model IDs")
        if self.source.required_local_model != DEFAULT_REQUIRED_LOCAL_MODEL:
            raise ValueError(
                "source.required_local_model must be exactly "
                f"{DEFAULT_REQUIRED_LOCAL_MODEL}"
            )
        if self.qwen.model != self.source.required_local_model:
            raise ValueError(
                "qwen.model must match source.required_local_model exactly"
            )
        if self.source.require_local_upstream_routing is not True:
            raise ValueError(
                "source.require_local_upstream_routing must be true"
            )
        if self.executive_review.enabled:
            raise ValueError(
                "executive review must be requested explicitly with the CLI"
            )
        if self.executive_review.model.model != DEFAULT_EXECUTIVE_MODEL:
            raise ValueError(
                f"executive_review.model must be exactly {DEFAULT_EXECUTIVE_MODEL}"
            )
        if self.executive_review.only != "asymmetric_approvals":
            raise ValueError(
                "executive_review.only must be 'asymmetric_approvals'"
            )
        if not self.source.required_local_model.strip():
            raise ValueError("source.required_local_model must not be empty")
        if self.run.max_images_per_package <= 0:
            raise ValueError("run.max_images_per_package must be positive")
        if (
            len(self.run.contestant_order) != 2
            or set(self.run.contestant_order) != {"terra", "sol"}
        ):
            raise ValueError(
                "run.contestant_order must contain terra and sol exactly once"
            )
        if self.run.strict is not True:
            raise ValueError("comparison Pass 2f strict mode cannot be disabled")
        if require_credentials and self.run.include_qwen and not qwen_endpoint():
            raise ValueError(
                "Qwen is enabled but no LM Studio endpoint is available"
            )

    def redacted_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "name": self.name,
            "source_path": self.source_path,
            "source": asdict(self.source),
            "models": {
                "terra": self.terra.redacted_dict(),
                "sol": self.sol.redacted_dict(),
                "qwen": self.qwen.redacted_dict(),
            },
            "executive_review": {
                "enabled": self.executive_review.enabled,
                "only": self.executive_review.only,
                **self.executive_review.model.redacted_dict(),
            },
            "run": {
                "max_images_per_package": self.run.max_images_per_package,
                "contestant_order": list(self.run.contestant_order),
                "include_qwen": self.run.include_qwen,
                "fail_fast": self.run.fail_fast,
                "strict": self.run.strict,
            },
        }


def profile_directory(project_root: Path) -> Path:
    return project_root / "configs" / "pass_2f_comparison"


def qwen_endpoint() -> str:
    """Resolve the local endpoint without placing it in profiles or reports."""
    return (
        os.environ.get("LM_STUDIO_URL", "").strip()
        or str(getattr(pipeline_config, "LM_STUDIO_URL", "")).strip()
    )


def qwen_vlm_config(config: Pass2fComparisonConfig) -> Dict[str, Any]:
    """Build the runtime-only local config, including the resolved endpoint."""
    result = config.qwen.to_vlm_config()
    result["url"] = qwen_endpoint()
    return result


def list_profiles(project_root: Path) -> list[Path]:
    directory = profile_directory(project_root)
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def resolve_profile_path(reference: str, project_root: Path) -> Path:
    candidate = Path(reference)
    if candidate.is_file():
        return candidate.resolve()
    name = reference if reference.lower().endswith(".json") else f"{reference}.json"
    candidate = profile_directory(project_root) / name
    if candidate.is_file():
        return candidate.resolve()
    for profile in list_profiles(project_root):
        try:
            declared_name = json.loads(
                profile.read_text(encoding="utf-8")
            ).get("name")
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if declared_name == reference:
            return profile.resolve()
    available = ", ".join(path.stem for path in list_profiles(project_root))
    raise ValueError(
        f"Unknown Pass 2f comparison profile {reference!r}; "
        f"available profiles: {available or 'none'}"
    )


def _reject_secret_or_indirect_fields(raw: Dict[str, Any], role: str) -> None:
    forbidden = {
        "api_key",
        "api_key_env",
        "base_url",
        "url",
        "url_env",
        "model_env",
        "fallback_model",
    }
    found = sorted(forbidden.intersection(raw))
    if found:
        raise ValueError(
            f"{role} contains forbidden secret, endpoint, indirect, or fallback "
            f"fields: {found}"
        )


def _model_spec(raw: Dict[str, Any], role: str) -> ModelSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"{role} must be an object")
    _reject_secret_or_indirect_fields(raw, role)
    if "model" not in raw:
        raise ValueError(f"{role}.model must be specified directly")
    provider = str(raw.get("provider") or "openai").strip().lower()
    return ModelSpec(
        label=str(raw.get("label") or role.replace("_", " ").title()),
        provider=provider,
        model=str(raw.get("model") or "").strip(),
        timeout_seconds=int(raw.get("timeout_seconds") or 360),
        manual_load=False,
        reasoning_effort=(
            str(raw.get("reasoning_effort") or "").strip() or None
        ),
        verbosity=str(raw.get("verbosity") or "").strip() or None,
    )


def load_pass_2f_comparison_profile(
    reference: str,
    project_root: Path,
) -> Pass2fComparisonConfig:
    path = resolve_profile_path(reference, project_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read profile {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Pass 2f comparison profile must be a JSON object")
    if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Pass 2f profile schema must be {PROFILE_SCHEMA_VERSION}"
        )

    models = raw.get("models")
    if not isinstance(models, dict):
        raise ValueError("models must contain terra, sol, and qwen")
    terra = _model_spec(models.get("terra") or {}, "terra")
    sol = _model_spec(models.get("sol") or {}, "sol")
    qwen = _model_spec(models.get("qwen") or {}, "qwen")

    source_raw = raw.get("source") or {}
    if not isinstance(source_raw, dict):
        raise ValueError("source must be an object")
    source = SourceRequirements(
        required_local_model=str(
            source_raw.get("required_local_model")
            or DEFAULT_REQUIRED_LOCAL_MODEL
        ).strip(),
        require_local_upstream_routing=bool(
            source_raw.get("require_local_upstream_routing", True)
        ),
    )

    executive_raw = raw.get("executive_review") or {}
    if not isinstance(executive_raw, dict):
        raise ValueError("executive_review must be an object")
    executive_spec_raw = {
        key: value
        for key, value in executive_raw.items()
        if key not in {"enabled", "only"}
    }
    executive = ExecutiveReviewSettings(
        enabled=bool(executive_raw.get("enabled", False)),
        only=str(
            executive_raw.get("only") or "asymmetric_approvals"
        ).strip(),
        model=_model_spec(executive_spec_raw, "executive_review"),
    )

    run_raw = raw.get("run") or {}
    if not isinstance(run_raw, dict):
        raise ValueError("run must be an object")
    order = run_raw.get("contestant_order") or ["terra", "sol"]
    run = Pass2fRunSettings(
        max_images_per_package=int(
            run_raw.get("max_images_per_package") or 3
        ),
        contestant_order=tuple(str(role) for role in order),
        include_qwen=bool(run_raw.get("include_qwen", False)),
        fail_fast=bool(run_raw.get("fail_fast", True)),
        strict=True,
    )
    config = Pass2fComparisonConfig(
        name=str(raw.get("name") or path.stem),
        source=source,
        terra=terra,
        sol=sol,
        qwen=qwen,
        executive_review=executive,
        run=run,
        source_path=str(path),
    )
    config.validate(require_credentials=False)
    return config


def with_cli_overrides(
    config: Pass2fComparisonConfig,
    *,
    terra_model: Optional[str] = None,
    sol_model: Optional[str] = None,
    max_images_per_package: Optional[int] = None,
    include_qwen: bool = False,
) -> Pass2fComparisonConfig:
    terra = (
        replace(config.terra, model=terra_model.strip())
        if terra_model
        else config.terra
    )
    sol = (
        replace(config.sol, model=sol_model.strip())
        if sol_model
        else config.sol
    )
    run = replace(
        config.run,
        max_images_per_package=(
            max_images_per_package
            if max_images_per_package is not None
            else config.run.max_images_per_package
        ),
        include_qwen=(include_qwen or config.run.include_qwen),
    )
    result = replace(config, terra=terra, sol=sol, run=run)
    result.validate(require_credentials=False)
    return result


def credentials_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

