#!/usr/bin/env python3
"""
CLI entrypoint for running the GroundingDINO pipeline as an external tool.

Designed to be called from RealtorVision via child_process.spawn using the
GDINO_PY virtualenv interpreter.

Supports:
- Premium vs Standard analysis profiles
- Per-pass enable/disable toggles (for development)
- Per-pass model overrides (for testing)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import config first. IMPORTANT: delay importing auto_analyzer until AFTER env overrides
try:
    from tools import pipeline_config as cfg  # type: ignore
except Exception as exc:  # pragma: no cover - external dependency
    print(f"Failed to import pipeline_config: {exc}", file=sys.stderr)
    sys.exit(1)


def _import_auto_analyzer():
    try:
        from tools.auto_analyzer import AutoAnalyzer
        return AutoAnalyzer
    except Exception as exc:
        print(f"Failed to import auto_analyzer: {exc}", file=sys.stderr)
        raise


# Optional: import pass config if available (for new architecture)
try:
    from tools.pass_config import PassToggles, PassModelOverrides, SceneClassifierRunOptions

    PASS_CONFIG_AVAILABLE = True
except ImportError:
    PASS_CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Environment variable keys that should be resolved as filesystem paths
PATH_OVERRIDE_KEYS = {
    "GDINO_CONFIG",
    "GDINO_CHECKPOINT",
    "GDINO_INFER_SCRIPT",
    "SCENE_CLASSIFIER_PY",
    "CHIP_VERIFIER_PY",
    "ANALYZER_CLI",
    "ISSUE_CATALOG_PATH",
}

# Environment variable keys that should be passed as-is (strings)
# ✅ FIXED: Added all GPT/OpenAI config key variations
STRING_OVERRIDE_KEYS = {
    # LM Studio / Qwen
    "LM_STUDIO_URL",
    "LM_STUDIO_MODEL",

    # OpenAI / GPT - support multiple naming conventions
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_PASS1B_MODEL",
    "OPENAI_PASS2A_MODEL",
    "OPENAI_PASS2C_MODEL",
    "OPENAI_PASS2D_MODEL",
    "OPENAI_PASS4_MODEL",
    "OPENAI_PASS_1B_MODEL",
    "OPENAI_PASS_2A_MODEL",
    "OPENAI_PASS_2C_MODEL",
    "OPENAI_PASS_2D_MODEL",
    "OPENAI_PASS_4_MODEL",
    "OPENAI_CHIP_MODEL",

    # GPT naming alternatives (some apps use GPT5_ prefix)
    "GPT_MODEL",
    "GPT5_URL",
    "GPT5_MODEL",

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

# All passes for CLI argument generation
ALL_PASSES = ['1a', '1b', '1c', '2a', '2b', '2c', '2d', '2e', '3', '4', '4a', '4b', '4c']


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

    # ✅ FIXED: Special handling for GPT model to handle multiple naming conventions
    # Priority: GPT5_MODEL > GPT_MODEL > OPENAI_MODEL
    gpt_model = (
            os.environ.get("GPT5_MODEL") or
            os.environ.get("GPT_MODEL") or
            os.environ.get("OPENAI_MODEL")
    )
    if gpt_model:
        setattr(cfg, "GPT_MODEL", gpt_model)
        setattr(cfg, "GPT5_MODEL", gpt_model)
        setattr(cfg, "OPENAI_MODEL", gpt_model)
        logger.debug("GPT model unified to: %s", gpt_model)

    # Ensure pass-specific GPT model vars are materialized on cfg
    def _first(*keys: str) -> Optional[str]:
        for k in keys:
            v = os.environ.get(k)
            if v:
                return v
        return None

    p1b = _first("OPENAI_PASS_1B_MODEL", "OPENAI_PASS1B_MODEL")
    p2a = _first("OPENAI_PASS_2A_MODEL", "OPENAI_PASS2A_MODEL")
    p2c = _first("OPENAI_PASS_2C_MODEL", "OPENAI_PASS2C_MODEL")
    p2d = _first("OPENAI_PASS_2D_MODEL", "OPENAI_PASS2D_MODEL")
    p4  = _first("OPENAI_PASS_4_MODEL",  "OPENAI_PASS4_MODEL")

    if p1b:
        setattr(cfg, "GPT_PASS_1B_MODEL", p1b)
    if p2a:
        setattr(cfg, "GPT_PASS_2A_MODEL", p2a)
    if p2c:
        setattr(cfg, "GPT_PASS_2C_MODEL", p2c)
    if p2d:
        setattr(cfg, "GPT_PASS_2D_MODEL", p2d)
    if p4:
        setattr(cfg, "GPT_PASS_4_MODEL", p4)

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

    if os.environ.get("PREMIUM_SKIP_VERIFICATION") is not None:
        setattr(cfg, "PREMIUM_SKIP_VERIFICATION", _to_bool(os.environ.get("PREMIUM_SKIP_VERIFICATION")))

    if os.environ.get("OPENAI_DEFAULT_MAX_TOKENS"):
        setattr(cfg, "OPENAI_DEFAULT_MAX_TOKENS", int(os.environ["OPENAI_DEFAULT_MAX_TOKENS"]))

    for k in ("OPENAI_PASS_1B_MAX_TOKENS", "OPENAI_PASS_1C_MAX_TOKENS", "OPENAI_PASS_2A_MAX_TOKENS", "OPENAI_PASS_2C_MAX_TOKENS", "OPENAI_PASS_2D_MAX_TOKENS", "OPENAI_PASS_4_MAX_TOKENS"):
        if os.environ.get(k) is not None:
            setattr(cfg, k, _to_int_or_none(os.environ.get(k)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GroundingDINO auto analyzer with premium analysis support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pass Control Examples:
  # Run with premium profile (uses GPT-5 for 2a, 2d)
  --analysis-profile premium

  # Disable specific passes for faster testing
  --disable-2d --disable-4

  # Override model for a specific pass (dev/testing)
  --model-2a gpt5 --model-2d qwen

  # Enable only scene classification (fast mode)
  --disable-1b --disable-2a --disable-2b --disable-2c --disable-2d --disable-2e --disable-3 --disable-4
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
    parser.add_argument("--html-report", action="store_true",
                        help="Write HTML report")
    parser.add_argument("--output-json", dest="output_json",
                        help="Path to write JSON summary")

    # ─────────────────────────────────────────────────────────────────────────
    # Runtime options
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--python-exe", dest="python_exe", default=sys.executable,
                        help="Python executable for subprocesses")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    # ─────────────────────────────────────────────────────────────────────────
    # Detection/threshold options
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--box-threshold", type=float,
                        help="Detection box confidence threshold")
    parser.add_argument("--text-threshold", type=float,
                        help="Text grounding threshold")
    parser.add_argument("--chip-margin", type=float,
                        help="Margin for detection chip extraction")
    parser.add_argument("--max-keywords", type=int,
                        help="Maximum keywords for detection")
    parser.add_argument("--include-conditions", action="store_const", const=True, default=None,
                        help="Include defect/condition keywords")
    parser.add_argument("--no-verify", dest="skip_verification", action="store_true",
                        help="Skip chip verification pass")

    # ─────────────────────────────────────────────────────────────────────────
    # Backend/Profile selection
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--detection-backend",
        dest="detection_backend",
        choices=["groundingdino", "dinox"],
        help="Detection backend: groundingdino (default) or dinox (premium)",
    )

    parser.add_argument(
        "--analysis-profile",
        dest="analysis_profile",
        choices=["standard", "premium"],
        help="Analysis profile: standard (all Qwen) or premium (GPT-5 for 2a, 2d)",
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
    # Per-pass model overrides (for development/testing)
    # ─────────────────────────────────────────────────────────────────────────
    model_group = parser.add_argument_group(
        'Model Overrides',
        'Override model selection for specific passes (dev/testing)'
    )

    for pass_key in ALL_PASSES:
        model_group.add_argument(
            f"--model-{pass_key}",
            dest=f"model_{pass_key.replace('-', '_')}",
            choices=["qwen", "gpt5"],
            help=f"Override model for pass {pass_key}",
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


def _build_model_overrides(args: argparse.Namespace) -> Dict[str, str]:
    """Build model overrides dict from CLI arguments."""
    overrides = {}

    for pass_key in ALL_PASSES:
        key = pass_key.replace('-', '_')
        model = getattr(args, f"model_{key}", None)
        if model:
            overrides[pass_key] = model

    return overrides


def _build_summary(
        job: Any,
        photo_intel_path: Optional[Path] = None,
        detection_backend: Optional[str] = None,
        analysis_profile: Optional[str] = None,
        pass_toggles: Optional[Dict[str, bool]] = None,
        model_overrides: Optional[Dict[str, str]] = None,
        used_pass_architecture: Optional[bool] = None,
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
        "detection_backend": detection_backend or "groundingdino",
        "analysis_profile": analysis_profile or "standard",
    }

    # Include pass config if any overrides were used
    if pass_toggles:
        summary["pass_toggles"] = pass_toggles
    if model_overrides:
        summary["model_overrides"] = model_overrides
    if used_pass_architecture is not None:
        summary["used_pass_architecture"] = used_pass_architecture

    return summary


import logging
import re


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
    logger.info(f"GPT_MODEL: {getattr(cfg, 'GPT_MODEL', 'NOT SET')}")

    # Pass-specific models
    logger.info(f"GPT_PASS_1B_MODEL: {getattr(cfg, 'GPT_PASS_1B_MODEL', 'NOT SET')}")
    logger.info(f"GPT_PASS_2A_MODEL: {getattr(cfg, 'GPT_PASS_2A_MODEL', 'NOT SET')}")
    logger.info(f"GPT_PASS_2C_MODEL: {getattr(cfg, 'GPT_PASS_2C_MODEL', 'NOT SET')}")
    logger.info(f"GPT_PASS_2D_MODEL: {getattr(cfg, 'GPT_PASS_2D_MODEL', 'NOT SET')}")
    logger.info(f"GPT_PASS_4_MODEL:  {getattr(cfg, 'GPT_PASS_4_MODEL',  'NOT SET')}")
    logger.info(f"OPENAI_BASE_URL:   {getattr(cfg, 'OPENAI_BASE_URL', os.environ.get('OPENAI_BASE_URL', '')) or 'DEFAULT'}")

    # Token caps
    logger.info(f"OPENAI_DEFAULT_MAX_TOKENS: {getattr(cfg, 'OPENAI_DEFAULT_MAX_TOKENS', os.environ.get('OPENAI_DEFAULT_MAX_TOKENS', '')) or 'NOT SET'}")
    logger.info(f"OPENAI_PASS_1B_MAX_TOKENS: {getattr(cfg, 'OPENAI_PASS_1B_MAX_TOKENS', os.environ.get('OPENAI_PASS_1B_MAX_TOKENS', '')) or 'NOT SET'}")
    logger.info(f"OPENAI_PASS_2A_MAX_TOKENS: {getattr(cfg, 'OPENAI_PASS_2A_MAX_TOKENS', os.environ.get('OPENAI_PASS_2A_MAX_TOKENS', '')) or 'NOT SET'}")
    logger.info(f"OPENAI_PASS_2C_MAX_TOKENS: {getattr(cfg, 'OPENAI_PASS_2C_MAX_TOKENS', os.environ.get('OPENAI_PASS_2C_MAX_TOKENS', '')) or 'NOT SET'}")
    logger.info(f"OPENAI_PASS_2D_MAX_TOKENS: {getattr(cfg, 'OPENAI_PASS_2D_MAX_TOKENS', os.environ.get('OPENAI_PASS_2D_MAX_TOKENS', '')) or 'NOT SET'}")
    logger.info(f"OPENAI_PASS_4_MAX_TOKENS:  {getattr(cfg, 'OPENAI_PASS_4_MAX_TOKENS',  os.environ.get('OPENAI_PASS_4_MAX_TOKENS',  '')) or 'NOT SET'}")

    # Detection backend
    logger.info(f"DETECTION_BACKEND: {getattr(cfg, 'DETECTION_BACKEND', 'groundingdino')}")

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

    # ✅ CRITICAL: Apply environment overrides BEFORE importing auto_analyzer
    _apply_env_overrides()

    # Log configuration for debugging
    if args.debug:
        _log_config_summary()

    images: List[Path] = [Path(img).resolve() for img in args.images]
    artifacts_root = Path(args.artifacts_root).resolve()

    # Build pass configuration from CLI args
    pass_toggles = _build_pass_toggles(args)
    model_overrides = _build_model_overrides(args)

    # Log configuration
    logger.info(f"Property: {args.property_key}")
    logger.info(f"Images: {len(images)}")
    logger.info(f"Detection Backend: {args.detection_backend or 'default'}")
    logger.info(f"Analysis Profile: {args.analysis_profile or 'default'}")

    if pass_toggles:
        logger.info(f"Pass Toggles: {pass_toggles}")
    if model_overrides:
        logger.info(f"Model Overrides: {model_overrides}")

    # Build analyzer kwargs
    analyzer_kwargs = {
        "python_exe": args.python_exe,
        "artifacts_root": artifacts_root,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "chip_margin": args.chip_margin,
        "max_keywords": args.max_keywords,
        "include_conditions": args.include_conditions,
        "skip_verification": args.skip_verification,
        "debug": args.debug,
        "detection_backend": args.detection_backend,
        "analysis_profile": args.analysis_profile,
        "use_pass_architecture": args.use_pass_architecture,
    }

    # Pass per-pass options if AutoAnalyzer supports them
    # (graceful degradation if not yet implemented)
    if pass_toggles:
        analyzer_kwargs["pass_toggles"] = pass_toggles
    if model_overrides:
        analyzer_kwargs["model_overrides"] = model_overrides

    # Filter out None values to let AutoAnalyzer use its defaults
    analyzer_kwargs = {k: v for k, v in analyzer_kwargs.items() if v is not None}

    # Import AutoAnalyzer AFTER env overrides have been applied
    AutoAnalyzer = _import_auto_analyzer()

    try:
        analyzer = AutoAnalyzer(**analyzer_kwargs)
    except TypeError as e:
        # If AutoAnalyzer doesn't support new args yet, fall back
        logger.warning(f"AutoAnalyzer doesn't support some args: {e}")
        basic_kwargs = {
            k: v for k, v in analyzer_kwargs.items()
            if k not in ('pass_toggles', 'model_overrides', 'use_pass_architecture')
        }
        analyzer = AutoAnalyzer(**basic_kwargs)

    job = analyzer.analyze_property(args.property_key, images)

    photo_intel_path = analyzer.save_photo_intel(job)

    if args.html_report:
        report_path = artifacts_root / args.property_key / job.job_id / "report.html"
        analyzer.create_html_report(job, report_path)

    # Use the analyzer's resolved values for accurate reporting
    summary = _build_summary(
        job,
        photo_intel_path,
        detection_backend=getattr(analyzer, "detection_backend", args.detection_backend),
        analysis_profile=getattr(analyzer, "analysis_profile", args.analysis_profile),
        pass_toggles=pass_toggles if pass_toggles else None,
        model_overrides=model_overrides if model_overrides else None,
        used_pass_architecture=getattr(analyzer, "use_pass_architecture", None),
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