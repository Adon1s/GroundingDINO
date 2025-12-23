"""
Pass Configuration for Scene Classifier Pipeline
-------------------------------------------------
Defines pass toggles, model selection, and premium routing logic.

Pass Overview:
- 1a: Scene type classification (always Qwen - fast/cheap)
- 1b: Positives/inventory notes - freeform (GPT-5 when premium)
- 1c: Positives notes -> JSON structuring (always Qwen - text-only)
- 2a: Issue detection (GPT-5 when premium)
- 2b: Issue verification (always Qwen - high volume)
- 3:  Keyword extraction (always Qwen - text-only)
- 4:  Property summary (GPT-5 when premium)
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, TypeAlias

# Type definitions
PassKey: TypeAlias = Literal['1a', '1b', '1c', '2a', '2b', '3', '4', '4a', '4b']
ModelName = Literal['qwen', 'gpt5']

# All valid pass keys (in execution order)
ALL_PASSES: tuple[PassKey, ...] = ('1a', '1b', '1c', '2a', '2b', '3', '4', '4a', '4b')


@dataclass
class PassToggles:
    """Enable/disable individual passes."""
    pass_1a: bool = True  # Scene type classification
    pass_1b: bool = True  # Positives/inventory notes (freeform)
    pass_1c: bool = True  # Positives notes -> JSON structuring
    pass_2a: bool = True  # Issue detection
    pass_2b: bool = True  # Issue verification
    pass_3: bool = True  # Keyword extraction
    pass_4: bool = True  # Property summary (often skipped in standard)
    pass_4a: bool = True  # Room summaries
    pass_4b: bool = True  # Property card fields

    def __getitem__(self, key: PassKey) -> bool:
        return getattr(self, f'pass_{key}')

    def __setitem__(self, key: PassKey, value: bool):
        setattr(self, f'pass_{key}', value)

    def to_dict(self) -> Dict[PassKey, bool]:
        return {
            '1a': self.pass_1a,
            '1b': self.pass_1b,
            '1c': self.pass_1c,
            '2a': self.pass_2a,
            '2b': self.pass_2b,
            '3': self.pass_3,
            '4': self.pass_4,
            '4a': self.pass_4a,
            '4b': self.pass_4b,
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
            pass_3=d.get('3', True),
            pass_4=d.get('4', True),
            pass_4a=d.get('4a', True),
            pass_4b=d.get('4b', True),
        )


@dataclass
class PassModelOverrides:
    """Override model selection for specific passes (dev/testing)."""
    model_1a: Optional[ModelName] = None
    model_1b: Optional[ModelName] = None
    model_1c: Optional[ModelName] = None
    model_2a: Optional[ModelName] = None
    model_2b: Optional[ModelName] = None
    model_3: Optional[ModelName] = None
    model_4: Optional[ModelName] = None
    model_4a: Optional[ModelName] = None
    model_4b: Optional[ModelName] = None

    def __getitem__(self, key: PassKey) -> Optional[ModelName]:
        return getattr(self, f'model_{key}')

    def __setitem__(self, key: PassKey, value: Optional[ModelName]):
        setattr(self, f'model_{key}', value)

    def to_dict(self) -> Dict[PassKey, Optional[ModelName]]:
        return {
            '1a': self.model_1a,
            '1b': self.model_1b,
            '1c': self.model_1c,
            '2a': self.model_2a,
            '2b': self.model_2b,
            '3': self.model_3,
            '4': self.model_4,
            '4a': self.model_4a,
            '4b': self.model_4b,
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
            model_3=to_model(d.get('3')),
            model_4=to_model(d.get('4')),
            model_4a=to_model(d.get('4a')),
            model_4b=to_model(d.get('4b')),
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
# - 1b uses GPT-5 (positives notes benefit from reasoning)
# - 1c stays Qwen (purely structuring, high-volume, text-only)
# - 2a uses GPT-5 (issue detection needs strong vision)
# - 2b stays Qwen (verification is high-volume)
# - 3 stays Qwen (keyword extraction is straightforward, text-only)
# - 4 uses GPT-5 (property summary is user-facing)

PREMIUM_MODEL_MAP: Dict[PassKey, ModelName] = {
    '1a': 'qwen',
    '1b': 'gpt5',
    '1c': 'qwen',
    '2a': 'gpt5',
    '2b': 'gpt5',
    '3': 'gpt5',
    '4': 'gpt5',
    '4a': 'gpt5',
    '4b': 'gpt5',
}

STANDARD_MODEL_MAP: Dict[PassKey, ModelName] = {
    '1a': 'qwen',
    '1b': 'qwen',
    '1c': 'qwen',
    '2a': 'qwen',
    '2b': 'qwen',
    '3': 'qwen',
    '4': 'qwen',
    '4a': 'qwen',
    '4b': 'qwen',
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
        pass_key: Which pass ('1a', '1b', '1c', '2a', '2b', '3', '4')
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
    '1b': 'Positives/Inventory Notes (freeform)',
    '1c': 'Positives Notes → JSON Structuring',
    '2a': 'Issue Detection',
    '2b': 'Issue Verification',
    '3': 'Keyword Extraction',
    '4': 'Property Summary',
    '4a': 'Room Summaries',
    '4b': 'Property Card Fields',
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
