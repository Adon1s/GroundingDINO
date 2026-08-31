"""Pinned runtime for the Pass 2a comparator.

These constants are owned here rather than read from
`benchmarks/pass2a-prompt/config.json`, so an edit to the benchmark's own
configuration cannot silently change what the comparator ran. They are recorded
verbatim in every baseline fingerprint, so a change here forces a new baseline
instead of contaminating an existing one.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    """Populate os.environ from .env before anything imports pipeline_config.

    `tools.pipeline_config` snapshots the environment at import time, so the
    model name and API key have to be in place first. `setdefault` keeps a real
    ambient value winning over the file, matching `benchmark_pass2a._ensure_env`.
    """
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env()

BENCH_DIR = ROOT / "benchmarks" / "pass2a-prompt"
MANIFEST_PATH = BENCH_DIR / "manifest.json"
GOLD_PATH = BENCH_DIR / "gold" / "reference.json"
COMPARATOR_DIR = BENCH_DIR / "runs" / "comparator"
BASELINES_DIR = COMPARATOR_DIR / "baselines"
EXPERIMENTS_DIR = COMPARATOR_DIR / "experiments"

#: Source files whose content defines the Pass 2a call path. Hashed whole into
#: the fingerprint (Steven, 2026-08-30): maximally conservative, so any edit to
#: either file requires a fresh baseline rather than risking a silent behaviour
#: change under a reused one.
PASS_PATH_SOURCES = (
    ROOT / "tools" / "scene_classifier_passes.py",
    ROOT / "tools" / "vlm_client.py",
)

COMPARATOR_SCHEMA_VERSION = "pass2a_comparator_v1"

#: Offline seam for the zero-token walkthrough. It is part of the fingerprint,
#: so stub runs address their own baseline directory and a later paid run can
#: never reuse - or be misled by - a stub output.
STUB_ENV = "PASS2A_COMPARATOR_STUB"


def stub_mode() -> bool:
    return bool(os.environ.get(STUB_ENV))

REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 8000
REPEATS = 3
CONCURRENCY = 3
#: Not a parameter: `vlm_client._analyze_image_openai` hardcodes
#: `"detail": "original"`. Recorded so the fingerprint would notice if it moved.
IMAGE_DETAIL = "original"

#: The 8000-token cap exceeds Pass 2a's 2000-token production cap, so nothing
#: measured here is production-equivalent on its own.
PRODUCTION_MAX_OUTPUT_TOKENS = 2000
PRODUCTION_EQUIVALENT_REASON = (
    "The 8000-token output cap exceeds Pass 2a's 2000-token production cap. "
    "A promising prompt still requires separate production-cap and downstream "
    "validation before shipping."
)

#: The guard keeps its daily ledgers under this root. There is no single
#: shared production ledger to point at - the guard is off in production and
#: the only ledgers on disk are per-canary-root - so the comparator meters
#: itself here. Consequence worth knowing: the 2.5M/day Terra ceiling is
#: enforced ACROSS COMPARATOR RUNS, not jointly with production. An operator
#: who sets RENOVATION_TERRA_USAGE_ROOT explicitly gets joint accounting.
USAGE_ROOT = COMPARATOR_DIR


def budget_guard_env() -> Dict[str, str]:
    """Environment the app adds when launching a run.

    Forcing the guard on without a usage root raises before the first call, so
    the two settings travel together (Steven, 2026-08-30).
    """
    env = {"RENOVATION_VLM_BUDGET_GUARD": "1"}
    if not (os.environ.get("RENOVATION_TERRA_USAGE_ROOT") or "").strip():
        env["RENOVATION_TERRA_USAGE_ROOT"] = str(USAGE_ROOT)
    return env


def assert_budget_guard_live() -> None:
    """Refuse to spend unmetered. Called only when a real client is built."""
    from tools import pipeline_config as cfg

    if not getattr(cfg, "RENOVATION_VLM_BUDGET_GUARD", False):
        raise ComparatorConfigError(
            "RENOVATION_VLM_BUDGET_GUARD is off, so these calls would spend "
            "Terra quota without a ledger row or a ceiling. Launch from the "
            "Streamlit app, or set RENOVATION_VLM_BUDGET_GUARD=1 and "
            "RENOVATION_TERRA_USAGE_ROOT before running the module directly."
        )
    if not (getattr(cfg, "RENOVATION_TERRA_USAGE_ROOT", "") or "").strip():
        raise ComparatorConfigError(
            "RENOVATION_VLM_BUDGET_GUARD is set but RENOVATION_TERRA_USAGE_ROOT "
            "is empty - the guard has nowhere to keep its daily ledgers."
        )

REASON_TAGS = (
    "better coverage",
    "better grounding",
    "better materiality",
    "better consistency",
    "missed material condition",
    "unsupported/speculative claim",
    "micro-observation noise",
    "lower consistency",
    "excessive verbosity",
    "other",
)

BLIND_VERDICTS = ("A better", "B better", "same", "unclear")
FINAL_VERDICTS = (
    "positive",
    "negative",
    "mixed",
    "no material change",
    "inconclusive",
)


class ComparatorConfigError(RuntimeError):
    """The pinned runtime could not be resolved. Never falls back silently."""


# ---------------------------------------------------------------------------
# Scene placeholder
# ---------------------------------------------------------------------------
# Production already computes the scene and hands it to Pass 2a - the
# orchestrator sets context['scene'] (scene_classifier_orchestrator.py:702,720)
# and passes it in - but run_pass_2a never reads it. A candidate prompt can
# opt into it here with {scene}, rendered from the benchmark's already-frozen
# Pass 1a capture, so no Pass 1a call is ever made.
#
# A prompt that uses it is NOT shippable as-is: production run_pass_2a would
# have to read context['scene'] and format it too. Reports say so.

SCENE_PLACEHOLDER = "scene"
_PLACEHOLDER_RE = re.compile(r"\{([^{}]*)\}")


def placeholders(text: str) -> Set[str]:
    """Placeholder names in a prompt, ignoring escaped `{{` / `}}`."""
    return set(_PLACEHOLDER_RE.findall(text.replace("{{", "").replace("}}", "")))


def validate_prompt_placeholders(text: str, label: str) -> None:
    """Reject anything but {scene} at authoring time, not mid-run.

    An unknown placeholder would otherwise surface as a KeyError on call 1 of
    45, after the baseline has already been paid for.
    """
    unknown = sorted(placeholders(text) - {SCENE_PLACEHOLDER})
    if unknown:
        raise ComparatorConfigError(
            f"{label} uses unsupported placeholder(s) {unknown}; only "
            f"{{{SCENE_PLACEHOLDER}}} is available. Use {{{{ and }}}} for a "
            f"literal brace."
        )


def uses_scene(*texts: str) -> bool:
    return any(SCENE_PLACEHOLDER in placeholders(t) for t in texts)


def render_prompt(text: str, scene: str) -> str:
    """Substitute {scene}; leave a prompt without the placeholder untouched.

    str.format handles {{ and }} escaping; the early return keeps a prompt that
    never opted in byte-identical, braces and all.
    """
    if SCENE_PLACEHOLDER not in placeholders(text):
        return text
    if not scene:
        raise ComparatorConfigError(
            "prompt uses {scene} but this photo has no frozen scene"
        )
    return text.format(scene=scene)


# ---------------------------------------------------------------------------
# Production prompts + pinned invocation
# ---------------------------------------------------------------------------

def production_prompts() -> Tuple[str, str]:
    """(system, user) imported from the real pass module. Never copied.

    A copy here drifts silently and makes the comparator show a prompt the run
    never used - the defect `tools/artifact_viewer/app.py` demonstrates.
    """
    from tools.scene_classifier_passes import (
        PASS_2A_SYSTEM_PROMPT,
        PASS_2A_USER_PROMPT,
    )

    return PASS_2A_SYSTEM_PROMPT, PASS_2A_USER_PROMPT


def build_model_config() -> Dict[str, Any]:
    """Resolve the pinned Pass 2a invocation through the backend config path.

    Resolving the name through `cfg.RENOVATION_TERRA_MODEL` is load-bearing, not
    cosmetic: OPENAI_MODEL is unset in this .env, and `maybe_reserve_vlm_call`
    fails closed unless the model matches the Terra or Sol model name.
    """
    from tools import pipeline_config as cfg
    from tools.pass_config import resolve_openai_invocation
    from tools.vlm_client import get_model_configs_from_pipeline_config

    model = getattr(cfg, "RENOVATION_TERRA_MODEL", "") or ""
    if not model:
        raise ComparatorConfigError(
            "RENOVATION_TERRA_MODEL is empty - set it in .env. There is "
            "intentionally no fallback model name."
        )

    _, gpt5_config = get_model_configs_from_pipeline_config(cfg)
    model_config = resolve_openai_invocation(
        "2a",
        {
            **gpt5_config,
            "model": model,
            "provider": "openai",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        REASONING_EFFORT,
    )

    # Fail loudly rather than measure a prompt under an unintended invocation.
    expected = {
        "model": model,
        "provider": "openai",
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    actual = {key: model_config.get(key) for key in expected}
    if actual != expected:
        raise ComparatorConfigError(
            f"resolved Pass 2a invocation does not match the pin: "
            f"expected {expected}, got {actual}"
        )
    return model_config
