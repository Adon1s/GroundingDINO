"""
Pass Configuration for Scene Classifier Pipeline
-------------------------------------------------
Defines pass toggles, model selection, and premium routing logic.

Pass Overview:
- 1a: Scene type classification (always Qwen - fast/cheap)
- 1b: Overall impression (GPT-5 when premium)
- 2a: Issue detection (GPT-5 when premium)
- 2b: Issue verification (always Qwen - high volume)
- 3:  Keyword extraction (always Qwen)
- 4:  Property summary (GPT-5 when premium)
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

# Type definitions
PassKey = Literal['1a', '1b', '2a', '2b', '3', '4']
ModelName = Literal['qwen', 'gpt5']

# All valid pass keys
ALL_PASSES: tuple[PassKey, ...] = ('1a', '1b', '2a', '2b', '3', '4')


@dataclass
class PassToggles:
    """Enable/disable individual passes."""
    pass_1a: bool = True  # Scene type classification
    pass_1b: bool = True  # Overall impression
    pass_2a: bool = True  # Issue detection
    pass_2b: bool = True  # Issue verification
    pass_3: bool = True  # Keyword extraction
    pass_4: bool = True  # Property summary (often skipped in standard)

    def __getitem__(self, key: PassKey) -> bool:
        return getattr(self, f'pass_{key}')

    def __setitem__(self, key: PassKey, value: bool):
        setattr(self, f'pass_{key}', value)

    def to_dict(self) -> Dict[PassKey, bool]:
        return {
            '1a': self.pass_1a,
            '1b': self.pass_1b,
            '2a': self.pass_2a,
            '2b': self.pass_2b,
            '3': self.pass_3,
            '4': self.pass_4,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, bool]]) -> 'PassToggles':
        if not d:
            return cls()
        return cls(
            pass_1a=d.get('1a', True),
            pass_1b=d.get('1b', True),
            pass_2a=d.get('2a', True),
            pass_2b=d.get('2b', True),
            pass_3=d.get('3', True),
            pass_4=d.get('4', True),
        )


@dataclass
class PassModelOverrides:
    """Override model selection for specific passes (dev/testing)."""
    model_1a: Optional[ModelName] = None
    model_1b: Optional[ModelName] = None
    model_2a: Optional[ModelName] = None
    model_2b: Optional[ModelName] = None
    model_3: Optional[ModelName] = None
    model_4: Optional[ModelName] = None

    def __getitem__(self, key: PassKey) -> Optional[ModelName]:
        return getattr(self, f'model_{key}')

    def __setitem__(self, key: PassKey, value: Optional[ModelName]):
        setattr(self, f'model_{key}', value)

    def to_dict(self) -> Dict[PassKey, Optional[ModelName]]:
        return {
            '1a': self.model_1a,
            '1b': self.model_1b,
            '2a': self.model_2a,
            '2b': self.model_2b,
            '3': self.model_3,
            '4': self.model_4,
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
            model_2a=to_model(d.get('2a')),
            model_2b=to_model(d.get('2b')),
            model_3=to_model(d.get('3')),
            model_4=to_model(d.get('4')),
        )


@dataclass
class SceneClassifierRunOptions:
    """Complete run options for scene classifier pipeline."""
    premium: bool = False
    toggles: PassToggles = field(default_factory=PassToggles)
    model_overrides: PassModelOverrides = field(default_factory=PassModelOverrides)

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
# - 1b uses GPT-5 (overall impression benefits from reasoning)
# - 2a uses GPT-5 (issue detection needs strong vision)
# - 2b stays Qwen (verification is high-volume)
# - 3 stays Qwen (keyword extraction is straightforward)
# - 4 uses GPT-5 (property summary is user-facing)

PREMIUM_MODEL_MAP: Dict[PassKey, ModelName] = {
    '1a': 'qwen',
    '1b': 'gpt5',
    '2a': 'gpt5',
    '2b': 'qwen',
    '3': 'qwen',
    '4': 'gpt5',
}

STANDARD_MODEL_MAP: Dict[PassKey, ModelName] = {
    '1a': 'qwen',
    '1b': 'qwen',
    '2a': 'qwen',
    '2b': 'qwen',
    '3': 'qwen',
    '4': 'qwen',
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
        pass_key: Which pass ('1a', '1b', '2a', '2b', '3', '4')
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
    '1b': 'Overall Impression',
    '2a': 'Issue Detection',
    '2b': 'Issue Verification',
    '3': 'Keyword Extraction',
    '4': 'Property Summary',
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