"""
Pass Configuration for Scene Classifier Pipeline
-------------------------------------------------
Defines pass toggles, model selection, and premium routing logic.

Pass Overview:
- 1a: Scene type classification (always Qwen - fast/cheap)
- 1b: Feature notes - freeform (always Qwen)
- 1c: Feature notes -> JSON structuring (always Qwen - text-only)
- 2a: Observations - freeform (GPT-5 when premium)
- 2b: Observations -> JSON (always Qwen - text-only)
- 2c: Label observations + debug/forward split (always Qwen - text-only)
- 2d: Resolve defect_id from candidates (GPT-5 when premium, optional)
- 2e: Normalize / filter / deduplicate issues (rule-based, no LLM)
- 2f: Big-ticket estimate review (GPT-5 when premium, post-processing)
- 3:  Keyword extraction (always Qwen - text-only)

Legacy passes (not currently executed by orchestrator but still supported):
- 4:  Property summary (legacy)
- 4a: Room summaries
- 4b: Renovation intel - scopes + work items
- 4c: Final narrative / verdict / priorities
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypeAlias

# Type definitions
PassKey: TypeAlias = Literal['1a', '1b', '1c', '2a', '2b', '2c', '2d', '2e', '2f', '3', '4', '4a', '4b', '4c']
ModelName = Literal['qwen', 'gpt5']

# All valid pass keys (in execution order)
ALL_PASSES: tuple[PassKey, ...] = ('1a', '1b', '1c', '2a', '2b', '2c', '2d', '2e', '2f', '3', '4', '4a', '4b', '4c')


@dataclass
class PassToggles:
    """Enable/disable individual passes."""
    pass_1a: bool = True   # Scene type classification
    pass_1b: bool = True   # Feature notes (freeform)
    pass_1c: bool = True   # Feature notes -> JSON structuring
    pass_2a: bool = True   # Observations (freeform)
    pass_2b: bool = True   # Observations -> JSON
    pass_2c: bool = True   # Label observations + debug/forward split
    pass_2d: bool = False  # Resolve defect_id from candidates (optional, requires candidate_provider)
    pass_2e: bool = True   # Normalize / filter / dedupe verified issues (rule-based, no LLM)
    pass_2f: bool = True   # Big-ticket estimate review (post-processing, requires eligible candidates + VLM)
    pass_3: bool = True    # Keyword extraction
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
            '3': self.pass_3,
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
            pass_2d=d.get('2d', False),  # default False - requires candidate_provider
            pass_2e=d.get('2e', True),
            pass_2f=d.get('2f', True),
            pass_3=d.get('3', True),
            pass_4=d.get('4', False),
            pass_4a=d.get('4a', False),
            pass_4b=d.get('4b', False),
            pass_4c=d.get('4c', False),
        )

    @classmethod
    def from_skip_list(cls, skip: Optional[List[str]] = None) -> 'PassToggles':
        """
        Create toggles by disabling passes listed in *skip*.

        Intended for use with pipeline_config.SKIP_PASSES or the
        SKIP_PASSES env var (comma-separated pass keys).

        Examples:
            PassToggles.from_skip_list(['1a', '1b', '1c'])
            PassToggles.from_skip_list(cfg.SKIP_PASSES)
        """
        toggles = cls()  # all defaults (1a-3 True, 2d/4x False)
        if not skip:
            return toggles
        for key in skip:
            attr = f'pass_{key.strip().lower()}'
            if hasattr(toggles, attr):
                setattr(toggles, attr, False)
        return toggles

    @classmethod
    def from_env(cls) -> 'PassToggles':
        """
        Build toggles from the SKIP_PASSES environment variable.

        Set SKIP_PASSES=1a,1b,1c to disable those passes.
        Unset or empty means all passes use their normal defaults.
        """
        raw = os.environ.get("SKIP_PASSES", "")
        skip = [s.strip().lower() for s in raw.split(",") if s.strip()]
        return cls.from_skip_list(skip)


@dataclass
class PassModelOverrides:
    """Override model selection for specific passes (dev/testing)."""
    model_1a: Optional[ModelName] = None
    model_1b: Optional[ModelName] = None
    model_1c: Optional[ModelName] = None
    model_2a: Optional[ModelName] = None
    model_2b: Optional[ModelName] = None
    model_2c: Optional[ModelName] = None
    model_2d: Optional[ModelName] = None
    model_2e: Optional[ModelName] = None
    model_2f: Optional[ModelName] = None
    model_3: Optional[ModelName] = None
    model_4: Optional[ModelName] = None
    model_4a: Optional[ModelName] = None
    model_4b: Optional[ModelName] = None
    model_4c: Optional[ModelName] = None

    def __getitem__(self, key: PassKey) -> Optional[ModelName]:
        return getattr(self, f'model_{key}', None)  # default None for safety

    def __setitem__(self, key: PassKey, value: Optional[ModelName]):
        setattr(self, f'model_{key}', value)

    def to_dict(self) -> Dict[PassKey, Optional[ModelName]]:
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
            '3': self.model_3,
            '4': self.model_4,
            '4a': self.model_4a,
            '4b': self.model_4b,
            '4c': self.model_4c,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, str]]) -> 'PassModelOverrides':
        if not d:
            return cls()

        def to_model(v: Optional[str]) -> Optional[ModelName]:
            if v in ('qwen', 'gpt5'):
                return v  # type: ignore
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
            model_3=to_model(d.get('3')),
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
    # Runtime metadata passed from AutoAnalyzer (run_id, property_key, photo_key, etc.)
    # Used by the orchestrator to build deterministic issue_ids per image.
    meta: Dict[str, Any] = field(default_factory=dict)

    def with_meta(self, **kwargs) -> "SceneClassifierRunOptions":
        """Return a copy of this options object with extra meta fields merged in."""
        m = dict(self.meta or {})
        m.update(kwargs)
        return SceneClassifierRunOptions(
            premium=self.premium,
            toggles=self.toggles,
            model_overrides=self.model_overrides,
            meta=m,
        )

    @classmethod
    def from_analysis_profile(
            cls,
            analysis_profile: str,
            toggles: Optional[Dict[str, bool]] = None,
            model_overrides: Optional[Dict[str, str]] = None,
    ) -> 'SceneClassifierRunOptions':
        """Create options from analysis profile string."""
        return cls(
            premium=(analysis_profile == 'premium'),
            toggles=PassToggles.from_dict(toggles),
            model_overrides=PassModelOverrides.from_dict(model_overrides),
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
# - 2d uses GPT-5 (resolver benefits from reasoning)
# - 2e stays Qwen (rule-based normalizer, no LLM call)
# - 2f uses GPT-5 (big-ticket review needs strong vision for posture decisions)
# - 3 stays Qwen (keyword extraction is straightforward, text-only)
# - 4/4a/4b/4c stay Qwen (legacy, may be deprecated)

PREMIUM_MODEL_MAP: Dict[PassKey, ModelName] = {
    '1a': 'qwen',
    '1b': 'qwen',
    '1c': 'qwen',
    '2a': 'gpt5',   # observations detection needs strong vision
    '2b': 'qwen',
    '2c': 'qwen',
    '2d': 'gpt5',   # resolver benefits from GPT reasoning
    '2e': 'qwen',   # rule-based normalizer, no LLM call
    '2f': 'gpt5',   # big-ticket review benefits from strong vision
    '3': 'qwen',
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
    '3': 'qwen',
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
        'qwen' or 'gpt5'
    """
    # Check for explicit override first
    if overrides:
        override = overrides[pass_key]
        if override:
            return override

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
    model = pick_model_for_pass(pass_key, options.premium, options.model_overrides)
    return gpt5_config if model == 'gpt5' else qwen_config


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
    '2f': 'Big-ticket Estimate Review (Pass 2f)',
    '3': 'Keyword Extraction',
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