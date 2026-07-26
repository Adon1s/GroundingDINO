#!/usr/bin/env python3
"""
CLI entrypoint for running the analysis pipeline as an external tool.

Designed to be called from RealtorVision via child_process.spawn.

Supports:
- Premium vs Standard analysis profiles
- Per-pass enable/disable toggles (for development)
- Per-pass model overrides (for testing)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import config first
try:
    from tools import pipeline_config as cfg  # type: ignore
except Exception as exc:  # pragma: no cover - external dependency
    print(f"Failed to import pipeline_config: {exc}", file=sys.stderr)
    sys.exit(1)


from tools.pass_config import ALL_PASSES, SceneClassifierRunOptions

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight job container for write_photo_intel()
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ImageResult:
    """Per-image result container matching write_photo_intel()'s res contract."""
    image_path: str
    scene_data: Optional[Dict[str, Any]] = None
    scene_classifier: Optional[Dict[str, Any]] = None  # alias; writer checks both
    scene: str = "unknown"
    processing_time: float = 0.0
    error: Optional[str] = None
    detection_count: int = 0
    verified_count: int = 0


@dataclass
class PropertyAnalysisJob:
    """Job container matching write_photo_intel()'s job contract."""
    property_key: str
    job_id: str
    artifacts_dir: str
    timestamp: str
    results: List[ImageResult] = field(default_factory=list)
    total_processing_time: float = 0.0
    property_metadata: Optional[Dict[str, Any]] = None

# Environment variable keys that should be resolved as filesystem paths
PATH_OVERRIDE_KEYS = {
    "ANALYZER_CLI",
    "ISSUE_CATALOG_PATH",
}

# Environment variable keys that should be passed as-is (strings)
# ✅ FIXED: Added all GPT/OpenAI config key variations
STRING_OVERRIDE_KEYS = {
    # LM Studio / Qwen
    "LM_STUDIO_URL",
    "LM_STUDIO_MODEL",

    # OpenAI (single base-model source; per-pass names come via --model-map)
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",

    # Detection backend
    "DETECTION_BACKEND",

    # Analysis profile
    "ANALYSIS_PROFILE",

    # DINO-X / DDS
    "DINOX_API_TOKEN",
    "DINOX_API_KEY",
    "DDS_API_TOKEN",
    "DDS_REGION",
    "DDS_DETECTOR_MODEL",

    # Premium-specific (string only - typed handled separately)
    "PREMIUM_SUMMARY_MODEL",

    # Summary model
    "SUMMARY_MODEL",
}


def _apply_env_overrides() -> None:
    """
    Allow env vars to override key config values when called from Node.

    This is critical for bridging RealtorVision's .env settings into the
    Python pipeline_config module.
    """
    # Filesystem path overrides – keep them as Path objects, not strings.
    for key in PATH_OVERRIDE_KEYS:
        val = os.environ.get(key)
        if val:
            resolved = Path(val).resolve()
            setattr(cfg, key, resolved)
            logger.debug("Override %s = %s", key, resolved)

    # Plain string overrides (no Path.resolve)
    for key in STRING_OVERRIDE_KEYS:
        val = os.environ.get(key)
        if val:
            setattr(cfg, key, val)
            logger.debug("Override %s = %s", key, val)

    # OPENAI_MODEL is handled by the generic string-override loop above (it is in
    # STRING_OVERRIDE_KEYS). Per-pass model names arrive via --model-map, not env,
    # so there is no GPT_MODEL / GPT_PASS_* materialization here anymore.

    # If premium summary model is set explicitly, honor it too
    prem_sum = os.environ.get("PREMIUM_SUMMARY_MODEL")
    if prem_sum:
        setattr(cfg, "PREMIUM_SUMMARY_MODEL", prem_sum)

    # Typed overrides (avoid turning ints/bools into strings on cfg)
    def _to_int_or_none(v: Optional[str]) -> Optional[int]:
        try:
            return int(v) if v is not None and v.strip() != "" else None
        except Exception:
            return None

    def _to_bool(v: Optional[str]) -> bool:
        return (v or "").strip().lower() in ("1", "true", "yes", "y", "on")

    if os.environ.get("PREMIUM_MAX_KEYWORDS"):
        setattr(cfg, "PREMIUM_MAX_KEYWORDS", int(os.environ["PREMIUM_MAX_KEYWORDS"]))

    # Token caps are resolved by pass_config.resolve_openai_invocation, which reads
    # OPENAI_PASS_<KEY>_MAX_TOKENS / OPENAI_DEFAULT_MAX_TOKENS straight from the
    # environment. Mirroring them onto cfg here is redundant.


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RealtorVision auto analyzer with premium analysis support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pass Control Examples:
  # Run with premium profile (uses GPT-5 for 2a, 2d)
  --analysis-profile premium

  # Disable specific passes for faster testing
  --disable-2d --disable-4

  # Override per-pass models with concrete OpenAI names (dev/testing)
  --model-map '{"2f":"gpt-5.6-sol","2a":"gpt-5.4-mini"}'

  # Enable only scene classification (fast mode)
  --disable-1b --disable-2a --disable-2b --disable-2c --disable-2d --disable-2e --disable-4
        """
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Required arguments
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--property-key", required=True, dest="property_key",
                        help="Property identifier")
    parser.add_argument("--images", required=True, nargs="+",
                        help="Absolute image paths")
    parser.add_argument("--artifacts-root", required=True, dest="artifacts_root",
                        help="Root directory for output artifacts")

    # ─────────────────────────────────────────────────────────────────────────
    # Output options
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--output-json", dest="output_json",
                        help="Path to write JSON summary")
    parser.add_argument("--property-metadata-json", dest="property_metadata_json",
                        default=None,
                        help="JSON string of listing facts (price/beds/baths/sqft/...) "
                             "from the CSV funnel DB; feeds cost_factors/estimate_units/sanity")

    # ─────────────────────────────────────────────────────────────────────────
    # Runtime options
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--python-exe", dest="python_exe", default=sys.executable,
                        help="Python executable for subprocesses")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent image analysis tasks (default: %(default)s)")

    # ─────────────────────────────────────────────────────────────────────────
    # Detection/threshold options
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--box-threshold", type=float,
                        help="Detection box confidence threshold")
    parser.add_argument("--text-threshold", type=float,
                        help="Text grounding threshold")
    # ─────────────────────────────────────────────────────────────────────────
    # Backend/Profile selection
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--detection-backend",
        dest="detection_backend",
        choices=["dinox"],
        help="Detection backend: dinox",
    )

    parser.add_argument(
        "--analysis-profile",
        dest="analysis_profile",
        choices=["standard", "premium"],
        help="Analysis profile: standard (all Qwen) or premium (GPT-5 for 2a, 2d)",
    )
    parser.add_argument(
        "--model-routing-profile",
        dest="model_routing_profile",
        choices=["standard", "premium"],
        default=None,
        help="Optional model-family routing profile independent of analysis metadata",
    )

    # Force legacy scene classifier (skip orchestrator)
    parser.add_argument(
        "--disable-pass-architecture",
        dest="use_pass_architecture",
        action="store_false",
        default=None,
        help="Force use of legacy scene classifier instead of pass architecture",
    )

    parser.add_argument(
        "--enable-pass-architecture",
        dest="use_pass_architecture",
        action="store_true",
        default=None,
        help="Force use of new pass architecture even in standard mode",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Per-pass enable/disable toggles (for development/testing)
    # ─────────────────────────────────────────────────────────────────────────
    pass_group = parser.add_argument_group(
        'Pass Toggles',
        'Enable or disable individual analysis passes (dev/testing)'
    )

    for pass_key in ALL_PASSES:
        # --enable-1a / --disable-1a style arguments
        pass_group.add_argument(
            f"--enable-{pass_key}",
            dest=f"enable_{pass_key.replace('-', '_')}",
            action="store_true",
            default=None,
            help=f"Force enable pass {pass_key}",
        )
        pass_group.add_argument(
            f"--disable-{pass_key}",
            dest=f"disable_{pass_key.replace('-', '_')}",
            action="store_true",
            default=None,
            help=f"Force disable pass {pass_key}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Per-pass model overrides — a single JSON map of concrete OpenAI model names.
    # This is the sole model-override transport (replaces the old --model-{pass}
    # gpt5|qwen flags + OPENAI_PASS_*_MODEL env vars).
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model-map",
        dest="model_map",
        default=None,
        help='JSON object of concrete per-pass OpenAI model names, e.g. '
             '\'{"2f":"gpt-5.6-sol","2a":"gpt-5.4-mini"}\'. A supplied name routes '
             'that pass to OpenAI.',
    )
    parser.add_argument(
        "--reasoning-map",
        dest="reasoning_map",
        default=None,
        help='JSON object of explicit GPT-5.6 reasoning efforts by pass, e.g. '
             '\'{"1a":"none","2f":"medium"}\'.',
    )
    parser.add_argument(
        "--failure-mode",
        dest="failure_mode",
        choices=("strict", "collect"),
        default="strict",
        help="strict (default): a failed pass fails the run and no artifact is "
             "written. collect: record pass errors and keep going — for "
             "diagnostics only, never for a run whose output will be used.",
    )

    return parser.parse_args()


def _build_pass_toggles(args: argparse.Namespace) -> Dict[str, bool]:
    """Build pass toggles dict from CLI arguments."""
    toggles = {}

    for pass_key in ALL_PASSES:
        key = pass_key.replace('-', '_')
        enable = getattr(args, f"enable_{key}", None)
        disable = getattr(args, f"disable_{key}", None)

        # Disable takes precedence over enable
        if disable:
            toggles[pass_key] = False
        elif enable:
            toggles[pass_key] = True
        # Else: not specified, use default (True)

    return toggles


# Passes whose model may be overridden via --model-map (mirrors the TS allowlist
# in lib/analysis/modelOverrides.ts).
_ALLOWED_MODEL_MAP_KEYS = {"1a", "1b", "1c", "2a", "2b", "2c", "2d", "2f"}


def _build_model_overrides(args: argparse.Namespace) -> Dict[str, str]:
    """Parse and validate the --model-map JSON into {pass_key: model_name}."""
    raw = getattr(args, "model_map", None)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"--model-map is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise SystemExit("--model-map must be a JSON object of {pass: model_name}")
    overrides: Dict[str, str] = {}
    for k, v in parsed.items():
        if not isinstance(k, str) or not isinstance(v, str) or not v.strip():
            continue
        if k not in _ALLOWED_MODEL_MAP_KEYS:
            logger.warning("Ignoring unsupported --model-map pass key: %s", k)
            continue
        overrides[k] = v.strip()
    return overrides


_ALLOWED_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


def _build_reasoning_efforts(args: argparse.Namespace) -> Dict[str, str]:
    """Parse and strictly validate the --reasoning-map JSON."""
    raw = getattr(args, "reasoning_map", None)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"--reasoning-map is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise SystemExit("--reasoning-map must be a JSON object of {pass: effort}")

    efforts: Dict[str, str] = {}
    for key, value in parsed.items():
        if key not in _ALLOWED_MODEL_MAP_KEYS:
            raise SystemExit(f"--reasoning-map contains unsupported pass key: {key!r}")
        if not isinstance(value, str) or value.strip().lower() not in _ALLOWED_REASONING_EFFORTS:
            raise SystemExit(
                f"--reasoning-map contains unsupported effort for pass {key}: {value!r}; "
                f"expected one of {sorted(_ALLOWED_REASONING_EFFORTS)}"
            )
        efforts[key] = value.strip().lower()
    return efforts


_TIMING_PASS_ORDER = ['1a', '1b', '1c', '2a', '2b', '2c', '2d', '2e', '2f']
_STATE_PRIORITY = {'failed': 5, 'executed': 4, 'rule_based': 3, 'stubbed': 2, 'skipped': 1}


def _compute_timing_stats(
    results: List["ImageResult"],
    total_wall_clock: Optional[float] = None,
    usage_stats: Optional[Dict[str, Any]] = None,
    *,
    phase_timings: Optional[Dict[str, Any]] = None,
    attempt_results: Optional[List["ImageResult"]] = None,
    requested_photo_count: Optional[int] = None,
    reused_photo_count: int = 0,
    configured_concurrency: Optional[int] = None,
    status: str = "complete",
    failed_phase: Optional[str] = None,
) -> Dict[str, Any]:
    """Build scope-aligned timing schema v2 plus compatibility aliases."""
    phases_in = dict(phase_timings or {})
    legacy_wall = float(total_wall_clock or 0.0)
    photo_wall = float(phases_in.get("photo_analysis_sec", legacy_wall) or 0.0)
    post_wall = float(phases_in.get("postprocessing_sec", 0.0) or 0.0)
    pass_2f_wall = float(phases_in.get("pass_2f_sec", 0.0) or 0.0)
    end_to_end = float(
        phases_in.get("end_to_end_sec", legacy_wall or (photo_wall + post_wall)) or 0.0
    )

    current_results = list(results if attempt_results is None else attempt_results)
    successful_all = [res for res in results if not res.error]
    failed_all = [res for res in results if res.error]
    successful_current = [res for res in current_results if not res.error]
    failed_current = [res for res in current_results if res.error]
    requested = int(requested_photo_count if requested_photo_count is not None else len(results))

    pass_totals: Dict[str, float] = {}
    pass_units: Dict[str, int] = {}
    observed_states: Dict[str, set] = {}
    photo_latencies: List[float] = []
    for res in successful_current:
        photo_latencies.append(float(res.processing_time or 0.0))
        data = res.scene_data or res.scene_classifier or {}
        timings = data.get("pass_timings", {}) or {}
        for pass_key, seconds in timings.items():
            key = str(pass_key)
            pass_totals[key] = pass_totals.get(key, 0.0) + float(seconds or 0.0)
            pass_units[key] = pass_units.get(key, 0) + 1
        states = data.get("pass_states", {}) or {}
        if states:
            for pass_key, pass_state in states.items():
                observed_states.setdefault(str(pass_key), set()).add(str(pass_state))
        else:
            ran = {str(value) for value in (data.get("passes_run", []) or [])}
            for pass_key in timings:
                observed_states.setdefault(str(pass_key), set()).add(
                    "executed" if str(pass_key) in ran else "skipped"
                )

    usage = dict(usage_stats or {})
    usage_by_pass = usage.get("per_pass", {}) or {}
    pass_keys = set(_TIMING_PASS_ORDER) | set(pass_totals) | set(observed_states) | set(usage_by_pass)
    if pass_2f_wall > 0:
        pass_keys.add("2f")
    ordered_passes = [key for key in _TIMING_PASS_ORDER if key in pass_keys]
    ordered_passes += sorted(key for key in pass_keys if key not in ordered_passes)

    passes: Dict[str, Dict[str, Any]] = {}
    for pass_key in ordered_passes:
        pass_usage = dict(usage_by_pass.get(pass_key, {}) or {})
        states = set(observed_states.get(pass_key, set()))
        if pass_key == "2f":
            work_sec = pass_2f_wall
            units = int(pass_usage.get("attempted_calls", 0) or 0)
            if units or work_sec > 0:
                states.add("executed")
        else:
            work_sec = float(pass_totals.get(pass_key, 0.0) or 0.0)
            units = int(pass_units.get(pass_key, 0) or 0)
        failed_calls = int(pass_usage.get("failed_calls", 0) or 0)
        successful_calls = int(pass_usage.get("calls", 0) or 0)
        if failed_calls:
            states.add("failed")
        state = max(states or {"skipped"}, key=lambda value: _STATE_PRIORITY.get(value, 0))
        passes[pass_key] = {
            "state": state,
            "work_sec": round(work_sec, 3),
            "api_work_sec": round(float(pass_usage.get("api_duration_sec", 0.0) or 0.0), 3),
            "units": units,
            "avg_unit_sec": round(work_sec / units, 3) if units else 0.0,
            "attempted_calls": int(pass_usage.get("attempted_calls", 0) or 0),
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "metered_calls": int(pass_usage.get("metered_calls", 0) or 0),
            "input_tokens": int(pass_usage.get("input_tokens", 0) or 0),
            "output_tokens": int(pass_usage.get("output_tokens", 0) or 0),
            "total_tokens": int(pass_usage.get("total_tokens", 0) or 0),
        }

    photo_pass_work = sum(pass_totals.values())
    llm_api_work = float(usage.get("api_duration_sec", 0.0) or 0.0)
    avg_latency = sum(photo_latencies) / len(photo_latencies) if photo_latencies else 0.0
    throughput_sec = photo_wall / len(successful_current) if successful_current else 0.0
    photo_throughput = len(successful_current) / photo_wall if photo_wall > 0 else 0.0
    effective_parallelism = photo_pass_work / photo_wall if photo_wall > 0 else 0.0
    total_tokens = int(usage.get("total_tokens", 0) or 0)

    stats: Dict[str, Any] = {
        "schema_version": 2,
        "status": {"state": status, "failed_phase": failed_phase},
        "phases": {
            "end_to_end_sec": round(end_to_end, 3),
            "photo_analysis_sec": round(photo_wall, 3),
            "postprocessing_sec": round(post_wall, 3),
            "pass_2f_sec": round(pass_2f_wall, 3),
            "other_postprocessing_sec": round(max(0.0, post_wall - pass_2f_wall), 3),
        },
        "photos": {
            "requested": requested,
            "processed_this_attempt": len(current_results),
            "reused_from_checkpoint": int(reused_photo_count or 0),
            "successful_total": len(successful_all),
            "failed_total": len(failed_all),
            "successful_this_attempt": len(successful_current),
            "failed_this_attempt": len(failed_current),
            "avg_latency_sec": round(avg_latency, 3),
            "min_latency_sec": round(min(photo_latencies), 3) if photo_latencies else 0.0,
            "max_latency_sec": round(max(photo_latencies), 3) if photo_latencies else 0.0,
            "throughput_sec_per_photo": round(throughput_sec, 3),
            "throughput_photos_per_sec": round(photo_throughput, 3),
            "configured_concurrency": configured_concurrency,
            "effective_parallelism": round(effective_parallelism, 3),
            "cumulative_pass_work_sec": round(photo_pass_work, 3),
        },
        "passes": passes,
        "usage": {
            "attempted_calls": int(usage.get("attempted_calls", usage.get("calls", 0)) or 0),
            "successful_calls": int(usage.get("calls", 0) or 0),
            "failed_calls": int(usage.get("failed_calls", 0) or 0),
            "metered_calls": int(usage.get("metered_calls", 0) or 0),
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": total_tokens,
            "api_work_sec": round(llm_api_work, 3),
            "job_tokens_per_sec": round(total_tokens / end_to_end, 3) if end_to_end > 0 else 0.0,
        },
    }

    # Compatibility aliases. Their v2 meanings are intentionally explicit:
    # total wall is end-to-end; LLM work is provider-call duration.
    stats.update({
        "photo_count": len(successful_all),
        "total_wall_clock_sec": round(end_to_end, 2),
        "total_llm_work_sec": round(llm_api_work, 2),
        "parallelism_ratio": round(effective_parallelism, 2),
        "avg_photo_sec": round(avg_latency, 2),
        "min_photo_sec": round(min(photo_latencies), 2) if photo_latencies else 0.0,
        "max_photo_sec": round(max(photo_latencies), 2) if photo_latencies else 0.0,
        "per_pass_total_sec": {key: value["work_sec"] for key, value in passes.items()},
        "per_pass_avg_sec": {key: value["avg_unit_sec"] for key, value in passes.items()},
        "passes_run": [key for key, value in passes.items() if value["state"] in {"executed", "rule_based"}],
        "passes_skipped": [key for key, value in passes.items() if value["state"] in {"skipped", "stubbed"}],
        "input_tokens": stats["usage"]["input_tokens"],
        "output_tokens": stats["usage"]["output_tokens"],
        "total_tokens": total_tokens,
        "llm_calls": stats["usage"]["successful_calls"],
        "metered_llm_calls": stats["usage"]["metered_calls"],
    })
    return stats


def _log_timing_stats(stats: Dict[str, Any], property_key: str) -> None:
    """Log schema-v2 phase wall time and non-additive pass work separately."""
    phases = stats.get("phases", {}) or {}
    photos = stats.get("photos", {}) or {}
    usage = stats.get("usage", {}) or {}
    passes = stats.get("passes", {}) or {}
    state = (stats.get("status", {}) or {}).get("state", "complete")
    total = float(phases.get("end_to_end_sec", 0.0) or 0.0)

    logger.info("=" * 76)
    logger.info(
        "TIMING STATS v2: %s (%s, %d photos, %.1fs end-to-end)",
        property_key,
        state,
        int(photos.get("requested", 0) or 0),
        total,
    )
    logger.info("=" * 76)
    logger.info("  PHASE WALL TIME (additive; Pass 2f is nested in postprocessing)")
    logger.info("  %-24s %10.2f", "Photo analysis", float(phases.get("photo_analysis_sec", 0) or 0))
    logger.info("  %-24s %10.2f", "Postprocessing/artifacts", float(phases.get("postprocessing_sec", 0) or 0))
    logger.info("  %-24s %10.2f", "  of which Pass 2f", float(phases.get("pass_2f_sec", 0) or 0))
    logger.info("  %-24s %10.2f", "End-to-end", total)
    logger.info("")
    logger.info("  PASS WORK (cumulative; concurrent work can exceed wall time)")
    logger.info("  %-6s %-11s %9s %9s %7s %10s", "Pass", "State", "Work(s)", "API(s)", "Calls", "Tokens")
    logger.info("  %-6s %-11s %9s %9s %7s %10s", "------", "-----------", "---------", "---------", "-------", "----------")
    for pass_key in _TIMING_PASS_ORDER + sorted(key for key in passes if key not in _TIMING_PASS_ORDER):
        if pass_key not in passes:
            continue
        row = passes[pass_key]
        logger.info(
            "  %-6s %-11s %9.2f %9.2f %7d %10s",
            pass_key,
            row.get("state", "skipped"),
            float(row.get("work_sec", 0) or 0),
            float(row.get("api_work_sec", 0) or 0),
            int(row.get("attempted_calls", 0) or 0),
            f'{int(row.get("total_tokens", 0) or 0):,}',
        )
    logger.info("")
    logger.info(
        "  Photos: requested=%d processed=%d reused=%d successful=%d failed=%d",
        int(photos.get("requested", 0) or 0),
        int(photos.get("processed_this_attempt", 0) or 0),
        int(photos.get("reused_from_checkpoint", 0) or 0),
        int(photos.get("successful_total", 0) or 0),
        int(photos.get("failed_total", 0) or 0),
    )
    logger.info(
        "  Photo latency avg: %.2fs | throughput: %.3f photos/sec (%.2fs/photo) | effective parallelism: %.2fx (configured=%s)",
        float(photos.get("avg_latency_sec", 0) or 0),
        float(photos.get("throughput_photos_per_sec", 0) or 0),
        float(photos.get("throughput_sec_per_photo", 0) or 0),
        float(photos.get("effective_parallelism", 0) or 0),
        photos.get("configured_concurrency") if photos.get("configured_concurrency") is not None else "unknown",
    )
    logger.info(
        "  API calls: %d attempted / %d successful / %d failed / %d metered",
        int(usage.get("attempted_calls", 0) or 0),
        int(usage.get("successful_calls", 0) or 0),
        int(usage.get("failed_calls", 0) or 0),
        int(usage.get("metered_calls", 0) or 0),
    )
    logger.info(
        "  Tokens: %s (%s in / %s out) | Job token throughput: %.1f tokens/sec",
        f'{int(usage.get("total_tokens", 0) or 0):,}',
        f'{int(usage.get("input_tokens", 0) or 0):,}',
        f'{int(usage.get("output_tokens", 0) or 0):,}',
        float(usage.get("job_tokens_per_sec", 0) or 0),
    )
    failed_phase = (stats.get("status", {}) or {}).get("failed_phase")
    if failed_phase:
        logger.info("  Incomplete phase: %s", failed_phase)
    logger.info("=" * 76)


def _build_summary(
        job: Any,
        photo_intel_path: Optional[Path] = None,
        detection_backend: Optional[str] = None,
        analysis_profile: Optional[str] = None,
        pass_toggles: Optional[Dict[str, bool]] = None,
        model_overrides: Optional[Dict[str, str]] = None,
        used_pass_architecture: Optional[bool] = None,
        timing_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build JSON summary for Node.js caller."""
    total_detections = sum((res.detection_count or 0) for res in job.results)
    verified = [res for res in job.results if res.verified_count is not None]
    total_verified = sum(res.verified_count or 0 for res in verified)

    summary = {
        "success": True,
        "jobId": job.job_id,
        "property_key": job.property_key,
        "artifacts_dir": job.artifacts_dir,
        "total_detections": total_detections,
        "verified_detections": total_verified,
        "total_processing_time": job.total_processing_time,
        "photo_intel_path": str(photo_intel_path) if photo_intel_path else None,
        "detection_backend": detection_backend or "dinox",
        "analysis_profile": analysis_profile or "standard",
    }

    # Include pass config if any overrides were used
    if pass_toggles:
        summary["pass_toggles"] = pass_toggles
    if model_overrides:
        summary["model_overrides"] = model_overrides
    if used_pass_architecture is not None:
        summary["used_pass_architecture"] = used_pass_architecture
    if timing_stats:
        summary["timing_stats"] = timing_stats

    return summary


import re  # noqa: E402 – used by PayloadRedactFilter below


class PayloadRedactFilter(logging.Filter):
    # Redact classic data URL base64s
    DATA_URL_RE = re.compile(
        r"(data:image\/[a-zA-Z0-9.+-]+;base64,)[A-Za-z0-9+/=]+"
    )

    # Redact long image_url values in dict-like logs (single quotes)
    IMAGE_URL_FIELD_RE = re.compile(
        r"('image_url'\s*:\s*')([^']{200,})(')"
    )

    # Redact long image_url values in JSON-like logs (double quotes)
    IMAGE_URL_FIELD_RE_JSON = re.compile(
        r'("image_url"\s*:\s*")([^"]{200,})(")'
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()

        if "image_url" in msg or "base64" in msg:
            cleaned = msg
            cleaned = self.DATA_URL_RE.sub(r"\1<base64_redacted>", cleaned)
            cleaned = self.IMAGE_URL_FIELD_RE.sub(r"\1<image_url_redacted>\3", cleaned)
            cleaned = self.IMAGE_URL_FIELD_RE_JSON.sub(r'\1<image_url_redacted>\3', cleaned)

            if cleaned != msg:
                record.msg = cleaned
                record.args = ()

        return True


def install_payload_redactor():
    filt = PayloadRedactFilter()

    # Attach to ROOT handlers (important!)
    root = logging.getLogger()
    for h in root.handlers:
        h.addFilter(filt)

    # Also attach to likely named emitters
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).addFilter(filt)


def _log_config_summary() -> None:
    """Log important configuration for debugging."""
    logger.info("=" * 60)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 60)

    # LM Studio / Qwen
    logger.info(f"LM_STUDIO_URL: {getattr(cfg, 'LM_STUDIO_URL', 'NOT SET')}")
    logger.info(f"LM_STUDIO_MODEL: {getattr(cfg, 'LM_STUDIO_MODEL', 'NOT SET')}")

    # OpenAI / GPT
    api_key = getattr(cfg, 'OPENAI_API_KEY', None) or os.environ.get('OPENAI_API_KEY', '')
    api_key_status = "SET" if api_key else "NOT SET"
    logger.info(f"OPENAI_API_KEY: {api_key_status}")
    logger.info(f"OPENAI_MODEL: {getattr(cfg, 'OPENAI_MODEL', '') or 'NOT SET'}")
    logger.info(f"OPENAI_BASE_URL:   {getattr(cfg, 'OPENAI_BASE_URL', os.environ.get('OPENAI_BASE_URL', '')) or 'DEFAULT'}")

    # Token caps
    logger.info(f"OPENAI_DEFAULT_MAX_TOKENS: {getattr(cfg, 'OPENAI_DEFAULT_MAX_TOKENS', os.environ.get('OPENAI_DEFAULT_MAX_TOKENS', '')) or 'NOT SET'}")
    logger.info(f"OPENAI_PASS_1B_MAX_TOKENS: {getattr(cfg, 'OPENAI_PASS_1B_MAX_TOKENS', os.environ.get('OPENAI_PASS_1B_MAX_TOKENS', '')) or 'NOT SET'}")
    logger.info(f"OPENAI_PASS_2A_MAX_TOKENS: {getattr(cfg, 'OPENAI_PASS_2A_MAX_TOKENS', os.environ.get('OPENAI_PASS_2A_MAX_TOKENS', '')) or 'NOT SET'}")
    logger.info(f"OPENAI_PASS_2C_MAX_TOKENS: {getattr(cfg, 'OPENAI_PASS_2C_MAX_TOKENS', os.environ.get('OPENAI_PASS_2C_MAX_TOKENS', '')) or 'NOT SET'}")
    logger.info(f"OPENAI_PASS_2D_MAX_TOKENS: {getattr(cfg, 'OPENAI_PASS_2D_MAX_TOKENS', os.environ.get('OPENAI_PASS_2D_MAX_TOKENS', '')) or 'NOT SET'}")

    # Detection backend
    logger.info(f"DETECTION_BACKEND: {getattr(cfg, 'DETECTION_BACKEND', 'dinox')}")

    # Analysis profile
    logger.info(f"ANALYSIS_PROFILE: {getattr(cfg, 'ANALYSIS_PROFILE', 'standard')}")

    logger.info("=" * 60)


def main() -> int:
    args = _parse_args()

    # Explicitly log to stderr so progress parsing works in Node.js
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    install_payload_redactor()

    # Apply environment overrides before pipeline runs
    _apply_env_overrides()

    # Log configuration for debugging
    if args.debug:
        _log_config_summary()

    images: List[Path] = [Path(img).resolve() for img in args.images]
    artifacts_root = Path(args.artifacts_root).resolve()

    # Build pass configuration from CLI args
    pass_toggles = _build_pass_toggles(args)
    model_overrides = _build_model_overrides(args)
    reasoning_efforts = _build_reasoning_efforts(args)

    # Log configuration
    logger.info(f"Property: {args.property_key}")
    logger.info(f"Images: {len(images)}")
    logger.info(f"Detection Backend: {args.detection_backend or 'default'}")
    logger.info(f"Analysis Profile: {args.analysis_profile or 'default'}")

    if pass_toggles:
        logger.info(f"Pass Toggles: {pass_toggles}")
    if model_overrides:
        logger.info(f"Model Overrides: {model_overrides}")
    if reasoning_efforts:
        logger.info(f"Reasoning Efforts: {reasoning_efforts}")

    # ─────────────────────────────────────────────────────────────────────────
    # Set up orchestrator pipeline
    # ─────────────────────────────────────────────────────────────────────────
    analysis_profile = args.analysis_profile or getattr(cfg, "ANALYSIS_PROFILE", "standard")
    model_routing_profile = args.model_routing_profile or analysis_profile
    detection_backend = args.detection_backend or getattr(cfg, "DETECTION_BACKEND", "dinox")

    try:
        from tools.artifact_writers import write_photo_intel, load_issue_catalog, Pass2fModelUnavailable
        from tools.scene_classifier_orchestrator import create_orchestrator_from_config
        from tools.vlm_client import get_model_configs_from_pipeline_config, create_vlm_client
    except Exception as exc:
        logger.error(f"Failed to import pipeline components: {exc}", exc_info=True)
        summary = {"success": False, "error": str(exc), "property_key": args.property_key}
        print(json.dumps(summary, ensure_ascii=False))
        return 1

    # Load issue catalog
    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)

    # Pass 2d preflight. The sidecar is a hard dependency when 2d is enabled:
    # without it every observation resolves to nothing, which downstream reads as
    # a property with no findings rather than as a broken run. Disabling 2d is the
    # explicit way to run without embeddings.
    candidate_provider = None
    embeddings_status = "disabled_by_pass_toggle"
    if pass_toggles.get("2d", True):
        from tools.catalog_embeddings import EmbeddingsRuntimeError, build_candidate_provider
        try:
            candidate_provider = build_candidate_provider(catalog)
            embeddings_status = "ready"
        except EmbeddingsRuntimeError as exc:
            logger.error(
                f"Pass 2d embeddings retriever unavailable: {exc}. "
                "Start the embeddings sidecar, or pass --disable-2d to run without it."
            )
            summary = {
                "success": False,
                "error": f"embeddings_init: {exc}",
                "property_key": args.property_key,
            }
            print(json.dumps(summary, ensure_ascii=False))
            return 1
    else:
        logger.info("Pass 2d disabled by toggle; skipping embeddings retriever init")

    # One shared client supplies both photo-pass and Pass 2f telemetry.
    _, gpt5_config = get_model_configs_from_pipeline_config(cfg)
    vlm_client = create_vlm_client()
    vlm_client.reset_usage_stats()

    # Create orchestrator
    orchestrator = create_orchestrator_from_config(
        cfg,
        candidate_provider=candidate_provider,
        catalog_items=catalog.get("items"),
        vlm_client=vlm_client,
    )

    # Build run options from CLI args + profile
    failure_mode = getattr(args, "failure_mode", "strict")
    options = SceneClassifierRunOptions.from_analysis_profile(
        analysis_profile=model_routing_profile,
        toggles=pass_toggles if pass_toggles else None,
        model_overrides=model_overrides if model_overrides else None,
        reasoning_efforts=reasoning_efforts if reasoning_efforts else None,
        failure_mode=failure_mode,
    )

    # Generate job ID and create artifacts directory
    job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = artifacts_root / args.property_key / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-image analysis loop
    # ─────────────────────────────────────────────────────────────────────────
    results: List[ImageResult] = []
    total_start = time.perf_counter()
    job_started = total_start

    async def _analyze_all():
        sem = asyncio.Semaphore(args.concurrency)
        completed = 0

        async def _analyze_one(idx, image_path):
            nonlocal completed
            async with sem:
                img_start = time.perf_counter()
                logger.info(f"[start] Analyzing: {image_path.name} ({idx + 1}/{len(images)})")

                try:
                    img_options = options.with_meta(
                        run_id=job_id,
                        photo_key=image_path.name,
                        property_key=args.property_key,
                    )
                    analysis = await orchestrator.analyze_image(
                        image_path=image_path,
                        options=img_options,
                    )
                    elapsed = time.perf_counter() - img_start

                    img_result = ImageResult(
                        image_path=str(image_path),
                        scene_data=analysis.to_dict(),
                        scene=analysis.scene or "unknown",
                        processing_time=elapsed,
                    )
                    logger.info(f"  {image_path.name} -> {analysis.scene} ({elapsed:.1f}s)")

                except Exception as exc:
                    elapsed = time.perf_counter() - img_start
                    logger.error(f"  ❌ {image_path.name} failed: {exc}", exc_info=args.debug)
                    img_result = ImageResult(
                        image_path=str(image_path),
                        # PassExecutionError carries the partial result so the
                        # failed image's pass_states/pass_errors survive for
                        # diagnostics in photo_intel_debug.json.
                        scene_data=getattr(exc, "partial_result", None),
                        scene="unknown",
                        processing_time=elapsed,
                        error=str(exc),
                    )

                completed += 1
                logger.info(f"  [{completed}/{len(images)}] images complete")
                return img_result

        tasks = [_analyze_one(i, img) for i, img in enumerate(images)]
        results.extend(await asyncio.gather(*tasks))

    asyncio.run(_analyze_all())

    total_time = time.perf_counter() - total_start
    phase_timings = {"photo_analysis_sec": total_time, "pass_2f_sec": 0.0}
    postprocessing_started = job_started + total_time

    # ─────────────────────────────────────────────────────────────────────────
    # Build job and write artifacts
    # ─────────────────────────────────────────────────────────────────────────
    property_metadata: Optional[Dict[str, Any]] = None
    if args.property_metadata_json:
        try:
            parsed = json.loads(args.property_metadata_json)
            if isinstance(parsed, dict):
                property_metadata = parsed
            else:
                logger.warning("--property-metadata-json is not a JSON object; ignoring")
        except (ValueError, TypeError) as exc:
            logger.warning(f"Failed to parse --property-metadata-json: {exc}; ignoring")

    job = PropertyAnalysisJob(
        property_key=args.property_key,
        job_id=job_id,
        artifacts_dir=str(job_dir),
        timestamp=datetime.utcnow().isoformat() + "Z",
        results=results,
        total_processing_time=total_time,
        property_metadata=property_metadata,
    )

    # model_overrides already carries concrete model names (from --model-map), so
    # it IS the resolved view. The model_routing array remains the canonical
    # per-pass record.
    resolved_model_overrides: Dict[str, str] = dict(model_overrides or {})

    fatal_error = None
    failed_phase = None

    # An image that failed a pass produced no valid analysis. Publishing the
    # remaining photos would present a partial property as a complete one, so
    # under strict mode the run fails and no artifact is written. Other images
    # were still processed so the failure summary covers the whole property.
    failed_images = [r for r in results if getattr(r, "error", None)]
    if failed_images and failure_mode == "strict":
        fatal_error = RuntimeError(
            f"{len(failed_images)} of {len(results)} images failed analysis: "
            + "; ".join(f"{Path(r.image_path).name}: {r.error}" for r in failed_images[:5])
        )
        failed_phase = "photo_analysis"
        logger.error(str(fatal_error))

    photo_intel_path = None
    try:
        if fatal_error is None:
            photo_intel_path = write_photo_intel(
                cfg=cfg,
                job=job,
                detection_backend=detection_backend,
                analysis_profile=analysis_profile,
                use_pass_architecture=True,
                pass_toggles=pass_toggles if pass_toggles else {},
                model_overrides=resolved_model_overrides,
                gpt_config=gpt5_config,
                issue_catalog=catalog,
                vlm_client=vlm_client,
                reasoning_efforts=reasoning_efforts,
                timing_recorder=phase_timings,
                dependency_status={"embeddings": embeddings_status},
            )
    except Pass2fModelUnavailable as exc:
        # Pass 2f is OpenAI-only. Rather than silently fall back to Qwen, fail
        # the whole run so the misconfiguration is visible.
        logger.error(f"Pass 2f model unavailable — failing run (no Qwen fallback): {exc}")
        photo_intel_path = None
        fatal_error = exc
        failed_phase = "pass_2f"
    except Exception as exc:
        logger.error(f"Failed to write photo_intel: {exc}", exc_info=True)
        photo_intel_path = None
        fatal_error = exc
        failed_phase = "postprocessing"

    if not failed_phase:
        failed_phase = phase_timings.get("failed_phase")

    phase_timings["postprocessing_sec"] = time.perf_counter() - postprocessing_started
    phase_timings["end_to_end_sec"] = time.perf_counter() - job_started

    # ─────────────────────────────────────────────────────────────────────────
    # Compute and log timing statistics
    # ─────────────────────────────────────────────────────────────────────────
    timing_stats = _compute_timing_stats(
        results,
        phase_timings["end_to_end_sec"],
        usage_stats=vlm_client.usage_stats,
        phase_timings=phase_timings,
        attempt_results=results,
        requested_photo_count=len(images),
        reused_photo_count=0,
        configured_concurrency=args.concurrency,
        status="failed" if fatal_error else ("partial" if failed_phase else "complete"),
        failed_phase=failed_phase,
    )
    _log_timing_stats(timing_stats, args.property_key)

    if fatal_error is not None:
        summary = {
            "success": False,
            "error": str(fatal_error),
            "property_key": args.property_key,
            "timing_stats": timing_stats,
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 1

    # ─────────────────────────────────────────────────────────────────────────
    # Build summary for Node.js caller
    # ─────────────────────────────────────────────────────────────────────────
    summary = _build_summary(
        job,
        photo_intel_path=photo_intel_path,
        detection_backend=detection_backend,
        analysis_profile=analysis_profile,
        pass_toggles=pass_toggles if pass_toggles else None,
        model_overrides=model_overrides if model_overrides else None,
        used_pass_architecture=True,
        timing_stats=timing_stats,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Last line on stdout must be JSON for the Node caller
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
