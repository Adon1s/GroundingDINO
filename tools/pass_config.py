"""
Pass Configuration for Scene Classifier Pipeline
-------------------------------------------------
Defines pass toggles, model selection, and premium routing logic.

Pass Overview:
- 1a: Scene type classification (always Qwen - fast/cheap)
- 1b: Feature notes - stubbed (outputs unused downstream; no LLM call)
- 1c: Feature notes -> JSON structuring - stubbed (no LLM call)
- 2a: Observations - freeform (GPT-5 when premium)
- 2b: Observations -> JSON (always Qwen - text-only)
- 2c: Label observations + debug/forward split (always Qwen - text-only)
- 2d: Resolve defect_id from candidates (GPT-5 when premium, optional)
- 2e: Normalize / filter / deduplicate issues (rule-based, no LLM)
- 2f: Package visual verification (GPT-5 when premium, post-processing)
Legacy passes (not currently executed by orchestrator but still supported):
- 4:  Property summary (legacy)
- 4a: Room summaries
- 4b: Renovation intel - scopes + work items
- 4c: Final narrative / verdict / priorities
"""

import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Literal, Optional, TypeAlias

# Type definitions
PassKey: TypeAlias = Literal['1a', '1b', '1c', '2a', '2b', '2c', '2d', '2e', '2f', '4', '4a', '4b', '4c']
ModelName = Literal['qwen', 'gpt5']
ReasoningEffort: TypeAlias = Literal['none', 'low', 'medium', 'high', 'xhigh', 'max']
FailureMode: TypeAlias = Literal['strict', 'collect']
ALLOWED_FAILURE_MODES: frozenset[str] = frozenset({'strict', 'collect'})

# All valid pass keys (in execution order)
ALL_PASSES: tuple[PassKey, ...] = ('1a', '1b', '1c', '2a', '2b', '2c', '2d', '2e', '2f', '4', '4a', '4b', '4c')
ALLOWED_REASONING_EFFORTS: frozenset[str] = frozenset(
    {'none', 'low', 'medium', 'high', 'xhigh', 'max'}
)
REASONING_PASS_KEYS: frozenset[str] = frozenset(
    {'1a', '1b', '1c', '2a', '2b', '2c', '2d', '2f'}
)

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI invocation policy
#
# `max_output_tokens` covers reasoning tokens *plus* visible output, so a cap
# that is too low surfaces as status="incomplete" rather than short prose. 2f
# reviews several images against a package and needs the larger budget.
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_MAX_TOKENS_DEFAULTS: Dict[PassKey, int] = {
    '1a': 2000, '1b': 2000, '1c': 2000,
    '2a': 2000, '2b': 2000, '2c': 2000, '2d': 2000,
    '2f': 4096,
}

# Explicit effort per pass. GPT-5.6 defaults to medium and GPT-5.4 to none, so
# leaving this implicit makes the token budget mean different things per model.
# 2e is absent by design (rule-based, no LLM call).
OPENAI_REASONING_DEFAULTS: Dict[PassKey, ReasoningEffort] = {
    '1a': 'low', '1b': 'low', '1c': 'low',
    '2a': 'low', '2b': 'low', '2c': 'low', '2d': 'low',
    '2f': 'medium',
}

assert set(OPENAI_REASONING_DEFAULTS) <= REASONING_PASS_KEYS


def _int_setting(name: str, raw: Any) -> Optional[int]:
    """
    Coerce a configured cap, raising on a present-but-unusable value.

    Falling through on a bad value would silently run uncapped, which is the
    same class of failure this module exists to prevent.
    """
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {raw!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def resolve_openai_invocation(
        pass_key: PassKey,
        model_config: dict,
        reasoning_effort: Optional[str] = None,
) -> dict:
    """
    Apply token-budget and reasoning policy to an OpenAI-bound pass config.

    Non-OpenAI configs (LM Studio/Qwen) are returned untouched. Resolution order
    for the cap is: value already on the config -> OPENAI_PASS_<KEY>_MAX_TOKENS
    -> an explicitly exported OPENAI_DEFAULT_MAX_TOKENS -> the per-pass default
    above. The per-pass env var name is derived from the pass key, so every pass
    has one without a lookup table.

    OPENAI_DEFAULT_MAX_TOKENS is read from the environment only. pipeline_config
    bakes in a fallback value for it, and honouring that attribute would let a
    catch-all default outrank the pass-specific numbers above -- which is how 2f
    would silently lose its larger multi-image budget.
    """
    provider = model_config.get("provider")
    if provider is None and model_config.get("api_key"):
        provider = "openai"
    if provider != "openai":
        return model_config

    try:
        from tools import pipeline_config as cfg
    except ImportError:
        cfg = None

    per_pass_name = f"OPENAI_PASS_{str(pass_key).upper()}_MAX_TOKENS"
    # Explicit key check, not `a or b`: a configured 0 must be rejected rather
    # than falling through to a default.
    cap = None
    for field_name in ("max_output_tokens", "max_tokens"):
        if model_config.get(field_name) is not None:
            cap = _int_setting(field_name, model_config[field_name])
            break
    if cap is None:
        raw = getattr(cfg, per_pass_name, None) if cfg is not None else None
        if raw is None:
            raw = os.environ.get(per_pass_name)
        cap = _int_setting(per_pass_name, raw)
    if cap is None:
        cap = _int_setting(
            "OPENAI_DEFAULT_MAX_TOKENS", os.environ.get("OPENAI_DEFAULT_MAX_TOKENS")
        )
    if cap is None:
        cap = OPENAI_MAX_TOKENS_DEFAULTS.get(pass_key)

    resolved = {**model_config}
    if cap is not None:
        # max_tokens is the named parameter on VLMClient.analyze_*; the
        # max_output_tokens alias is what the Responses payload actually sends.
        resolved["max_tokens"] = int(cap)
        resolved["max_output_tokens"] = int(cap)

    effort = reasoning_effort or OPENAI_REASONING_DEFAULTS.get(pass_key)
    if effort and effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            f"unsupported reasoning effort for pass {pass_key}: {effort!r}; "
            f"expected one of {sorted(ALLOWED_REASONING_EFFORTS)}"
        )

    from tools.vlm_client import model_supports_reasoning
    if effort and model_supports_reasoning(resolved.get("model")):
        resolved["reasoning_effort"] = effort
    else:
        resolved.pop("reasoning_effort", None)

    return resolved


def normalize_reasoning_efforts(
        raw: Optional[Dict[str, str]],
) -> Dict[PassKey, ReasoningEffort]:
    """Validate and normalize a per-pass Responses API reasoning-effort map."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("reasoning efforts must be an object of {pass: effort}")

    normalized: Dict[PassKey, ReasoningEffort] = {}
    for pass_key, effort in raw.items():
        if pass_key not in REASONING_PASS_KEYS:
            raise ValueError(f"unsupported reasoning-effort pass key: {pass_key!r}")
        if not isinstance(effort, str) or effort.strip().lower() not in ALLOWED_REASONING_EFFORTS:
            raise ValueError(
                f"unsupported reasoning effort for pass {pass_key}: {effort!r}; "
                f"expected one of {sorted(ALLOWED_REASONING_EFFORTS)}"
            )
        normalized[pass_key] = effort.strip().lower()  # type: ignore[assignment]
    return normalized


@dataclass
class PassToggles:
    """Enable/disable individual passes."""
    pass_1a: bool = True   # Scene type classification
    pass_1b: bool = True   # Feature notes (freeform)
    pass_1c: bool = True   # Feature notes -> JSON structuring
    pass_2a: bool = True   # Observations (freeform)
    pass_2b: bool = True   # Observations -> JSON
    pass_2c: bool = True   # Label observations + debug/forward split
    pass_2d: bool = True   # Resolve defect_id from candidates (requires candidate_provider)
    pass_2e: bool = True   # Normalize / filter / dedupe verified issues (rule-based, no LLM)
    pass_2f: bool = True   # Package visual verification (post-processing, requires package candidates + VLM)
    pass_4: bool = False   # Property summary (legacy, not executed by current orchestrator)
    pass_4a: bool = False  # Room summaries (legacy, not executed by current orchestrator)
    pass_4b: bool = False  # Renovation intel (legacy, not executed by current orchestrator)
    pass_4c: bool = False  # Final narrative (legacy, not executed by current orchestrator)

    def __getitem__(self, key: PassKey) -> bool:
        return getattr(self, f'pass_{key}', False)  # default False for safety

    def __setitem__(self, key: PassKey, value: bool):
        setattr(self, f'pass_{key}', value)

    def to_dict(self) -> Dict[PassKey, bool]:
        return {
            '1a': self.pass_1a,
            '1b': self.pass_1b,
            '1c': self.pass_1c,
            '2a': self.pass_2a,
            '2b': self.pass_2b,
            '2c': self.pass_2c,
            '2d': self.pass_2d,
            '2e': self.pass_2e,
            '2f': self.pass_2f,
            '4': self.pass_4,
            '4a': self.pass_4a,
            '4b': self.pass_4b,
            '4c': self.pass_4c,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, bool]]) -> 'PassToggles':
        if not d:
            return cls()
        return cls(
            pass_1a=d.get('1a', True),
            pass_1b=d.get('1b', True),
            pass_1c=d.get('1c', True),
            pass_2a=d.get('2a', True),
            pass_2b=d.get('2b', True),
            pass_2c=d.get('2c', True),
            pass_2d=d.get('2d', True),
            pass_2e=d.get('2e', True),
            pass_2f=d.get('2f', True),
            pass_4=d.get('4', False),
            pass_4a=d.get('4a', False),
            pass_4b=d.get('4b', False),
            pass_4c=d.get('4c', False),
        )


@dataclass
class PassModelOverrides:
    """
    Per-pass model overrides. Each value is a *concrete* OpenAI model name
    (e.g. "gpt-5.6-sol") supplied per-run via the analyzer's model-map argument;
    a supplied name routes that pass to OpenAI with that model. (The legacy
    'qwen'/'gpt5' family labels are no longer used on the wire.)
    """
    model_1a: Optional[str] = None
    model_1b: Optional[str] = None
    model_1c: Optional[str] = None
    model_2a: Optional[str] = None
    model_2b: Optional[str] = None
    model_2c: Optional[str] = None
    model_2d: Optional[str] = None
    model_2e: Optional[str] = None
    model_2f: Optional[str] = None
    model_4: Optional[str] = None
    model_4a: Optional[str] = None
    model_4b: Optional[str] = None
    model_4c: Optional[str] = None

    def __getitem__(self, key: PassKey) -> Optional[str]:
        return getattr(self, f'model_{key}', None)  # default None for safety

    def __setitem__(self, key: PassKey, value: Optional[str]):
        setattr(self, f'model_{key}', value)

    def to_dict(self) -> Dict[PassKey, Optional[str]]:
        return {
            '1a': self.model_1a,
            '1b': self.model_1b,
            '1c': self.model_1c,
            '2a': self.model_2a,
            '2b': self.model_2b,
            '2c': self.model_2c,
            '2d': self.model_2d,
            '2e': self.model_2e,
            '2f': self.model_2f,
            '4': self.model_4,
            '4a': self.model_4a,
            '4b': self.model_4b,
            '4c': self.model_4c,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, str]]) -> 'PassModelOverrides':
        if not d:
            return cls()

        def to_model(v: Optional[str]) -> Optional[str]:
            # A concrete model name (e.g. "gpt-5.6-sol") routes that pass to
            # OpenAI. Non-strings / blanks are ignored.
            if isinstance(v, str) and v.strip():
                return v.strip()
            return None

        return cls(
            model_1a=to_model(d.get('1a')),
            model_1b=to_model(d.get('1b')),
            model_1c=to_model(d.get('1c')),
            model_2a=to_model(d.get('2a')),
            model_2b=to_model(d.get('2b')),
            model_2c=to_model(d.get('2c')),
            model_2d=to_model(d.get('2d')),
            model_2e=to_model(d.get('2e')),
            model_2f=to_model(d.get('2f')),
            model_4=to_model(d.get('4')),
            model_4a=to_model(d.get('4a')),
            model_4b=to_model(d.get('4b')),
            model_4c=to_model(d.get('4c')),
        )


@dataclass
class SceneClassifierRunOptions:
    """Complete run options for scene classifier pipeline."""
    premium: bool = False
    toggles: PassToggles = field(default_factory=PassToggles)
    model_overrides: PassModelOverrides = field(default_factory=PassModelOverrides)
    reasoning_efforts: Dict[PassKey, ReasoningEffort] = field(default_factory=dict)
    # How to handle a pass that fails.
    #   strict  - propagate, failing the image and ultimately the property run
    #   collect - record the error on the result and stop that image's passes
    # 'collect' is for tests and diagnostics; paid runs are always strict.
    failure_mode: FailureMode = 'strict'
    # Runtime metadata (run_id, property_key, photo_key, etc.)
    # Used by the orchestrator to build deterministic issue_ids per image.
    meta: Dict[str, Any] = field(default_factory=dict)

    def with_meta(self, **kwargs) -> "SceneClassifierRunOptions":
        """Return a copy of this options object with extra meta fields merged in."""
        m = dict(self.meta or {})
        m.update(kwargs)
        # dataclasses.replace, not a hand-listed rebuild: the previous version
        # enumerated every field, so any field added later was silently dropped.
        return replace(self, meta=m)

    @classmethod
    def from_analysis_profile(
            cls,
            analysis_profile: str,
            toggles: Optional[Dict[str, bool]] = None,
            model_overrides: Optional[Dict[str, str]] = None,
            reasoning_efforts: Optional[Dict[str, str]] = None,
            failure_mode: str = 'strict',
    ) -> 'SceneClassifierRunOptions':
        """Create options from analysis profile string."""
        if failure_mode not in ALLOWED_FAILURE_MODES:
            raise ValueError(
                f"unsupported failure_mode: {failure_mode!r}; "
                f"expected one of {sorted(ALLOWED_FAILURE_MODES)}"
            )
        return cls(
            premium=(analysis_profile == 'premium'),
            toggles=PassToggles.from_dict(toggles),
            model_overrides=PassModelOverrides.from_dict(model_overrides),
            reasoning_efforts=normalize_reasoning_efforts(reasoning_efforts),
            failure_mode=failure_mode,  # type: ignore[arg-type]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Model Selection Logic
# ═══════════════════════════════════════════════════════════════════════════════

# Premium model mapping (when premium=True)
# - 1a stays Qwen (scene type is simple/fast)
# - 1b stays Qwen (feature notes are text-only structuring)
# - 1c stays Qwen (purely structuring, text-only)
# - 2a uses GPT-5 (observations detection needs strong vision)
# - 2b stays Qwen (structuring is text-only)
# - 2c stays Qwen (labeling is text-only)
# - 2d stays Qwen (resolver uses local model)
# - 2e stays Qwen (rule-based normalizer, no LLM call)
# - 2f uses GPT-5 (big-ticket review needs strong vision for posture decisions)
# - 4/4a/4b/4c stay Qwen (legacy, may be deprecated)

PREMIUM_MODEL_MAP: Dict[PassKey, ModelName] = {
    '1a': 'qwen',
    '1b': 'qwen',
    '1c': 'qwen',
    '2a': 'gpt5',   # observations detection needs strong vision
    '2b': 'qwen',
    '2c': 'qwen',
    '2d': 'qwen',   # resolver uses local model
    '2e': 'qwen',   # rule-based normalizer, no LLM call
    '2f': 'gpt5',   # big-ticket review benefits from strong vision
    '4': 'qwen',
    '4a': 'qwen',
    '4b': 'qwen',
    '4c': 'qwen',
}

STANDARD_MODEL_MAP: Dict[PassKey, ModelName] = {
    '1a': 'qwen',
    '1b': 'qwen',
    '1c': 'qwen',
    '2a': 'qwen',
    '2b': 'qwen',
    '2c': 'qwen',
    '2d': 'qwen',
    '2e': 'qwen',
    '2f': 'qwen',
    '4': 'qwen',
    '4a': 'qwen',
    '4b': 'qwen',
    '4c': 'qwen',
}


def pick_model_for_pass(
        pass_key: PassKey,
        premium: bool,
        overrides: Optional[PassModelOverrides] = None,
) -> ModelName:
    """
    Determine which model to use for a given pass.

    Priority:
    1. Explicit override (for dev/testing)
    2. Premium mapping (if premium=True)
    3. Standard mapping (always Qwen)

    Args:
        pass_key: Which pass ('1a', '1b', '1c', '2a', '2b', '2c', '2d', '3', etc.)
        premium: Whether premium analysis is enabled
        overrides: Optional per-pass model overrides

    Returns:
        'qwen' or 'gpt5' (the model *family*). A concrete per-run override name
        always maps to 'gpt5' (OpenAI).
    """
    # A concrete per-run model name means OpenAI (gpt5 family).
    if overrides and overrides[pass_key]:
        return 'gpt5'

    # Use premium or standard mapping
    if premium:
        return PREMIUM_MODEL_MAP[pass_key]
    else:
        return STANDARD_MODEL_MAP[pass_key]


def get_model_config_for_pass(
        pass_key: PassKey,
        options: SceneClassifierRunOptions,
        qwen_config: dict,
        gpt5_config: dict,
) -> dict:
    """
    Get the actual model configuration (URL, model name, etc.) for a pass.

    Args:
        pass_key: Which pass
        options: Run options with premium flag and overrides
        qwen_config: Config dict for Qwen (e.g., {'url': ..., 'model': ...})
        gpt5_config: Config dict for GPT-5

    Returns:
        The appropriate config dict
    """
    # A concrete per-run override routes this pass to OpenAI with that exact
    # model name (no dependence on cfg.GPT_PASS_* — those are gone).
    override = options.model_overrides[pass_key] if options.model_overrides else None
    if override:
        model_config = {**gpt5_config, 'model': override, 'provider': 'openai'}
    else:
        model = pick_model_for_pass(pass_key, options.premium, options.model_overrides)
        model_config = gpt5_config if model == 'gpt5' else qwen_config

    return resolve_openai_invocation(
        pass_key,
        model_config,
        options.reasoning_efforts.get(pass_key),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass Descriptions (for logging/debugging)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_DESCRIPTIONS: Dict[PassKey, str] = {
    '1a': 'Scene Type Classification',
    '1b': 'Feature Notes (freeform)',
    '1c': 'Feature Notes → JSON Structuring',
    '2a': 'Observations (freeform)',
    '2b': 'Observations → JSON',
    '2c': 'Label Observations (debug/forward)',
    '2d': 'Resolve defect_id from candidates',
    '2e': 'Normalize / Filter / Deduplicate Issues',
    '2f': 'Package Visual Verification (Pass 2f)',
    '4': 'Property Summary (legacy)',
    '4a': 'Room Summaries (legacy)',
    '4b': 'Renovation Intel (legacy)',
    '4c': 'Final Narrative (legacy)',
}


def describe_run_plan(options: SceneClassifierRunOptions) -> str:
    """Generate a human-readable description of the planned run."""
    lines = [
        f"Analysis Profile: {'PREMIUM' if options.premium else 'STANDARD'}",
        "Pass Configuration:",
    ]

    for pass_key in ALL_PASSES:
        enabled = options.toggles[pass_key]
        if enabled:
            model = pick_model_for_pass(pass_key, options.premium, options.model_overrides)
            override = options.model_overrides[pass_key]
            override_note = " (override)" if override else ""
            lines.append(f"  {pass_key}: {PASS_DESCRIPTIONS[pass_key]} → {model.upper()}{override_note}")
        else:
            lines.append(f"  {pass_key}: {PASS_DESCRIPTIONS[pass_key]} → DISABLED")

    return "\n".join(lines)
