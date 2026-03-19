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


# Optional: import pass config if available
try:
    from tools.pass_config import PassToggles, PassModelOverrides, SceneClassifierRunOptions

    PASS_CONFIG_AVAILABLE = True
except ImportError:
    PASS_CONFIG_AVAILABLE = False

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

# Environment variable keys that should be resolved as filesystem paths
PATH_OVERRIDE_KEYS = {
    "SCENE_CLASSIFIER_PY",
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
ALL_PASSES = ['1a', '1b', '1c', '2a', '2b', '2c', '2d', '2e', '2f', '4', '4a', '4b', '4c']


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

    if os.environ.get("OPENAI_DEFAULT_MAX_TOKENS"):
        setattr(cfg, "OPENAI_DEFAULT_MAX_TOKENS", int(os.environ["OPENAI_DEFAULT_MAX_TOKENS"]))

    for k in ("OPENAI_PASS_1B_MAX_TOKENS", "OPENAI_PASS_1C_MAX_TOKENS", "OPENAI_PASS_2A_MAX_TOKENS", "OPENAI_PASS_2C_MAX_TOKENS", "OPENAI_PASS_2D_MAX_TOKENS", "OPENAI_PASS_4_MAX_TOKENS"):
        if os.environ.get(k) is not None:
            setattr(cfg, k, _to_int_or_none(os.environ.get(k)))


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

  # Override model for a specific pass (dev/testing)
  --model-2a gpt5 --model-2d qwen

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

    # Log configuration
    logger.info(f"Property: {args.property_key}")
    logger.info(f"Images: {len(images)}")
    logger.info(f"Detection Backend: {args.detection_backend or 'default'}")
    logger.info(f"Analysis Profile: {args.analysis_profile or 'default'}")

    if pass_toggles:
        logger.info(f"Pass Toggles: {pass_toggles}")
    if model_overrides:
        logger.info(f"Model Overrides: {model_overrides}")

    # ─────────────────────────────────────────────────────────────────────────
    # Set up orchestrator pipeline
    # ─────────────────────────────────────────────────────────────────────────
    analysis_profile = args.analysis_profile or getattr(cfg, "ANALYSIS_PROFILE", "standard")
    detection_backend = args.detection_backend or getattr(cfg, "DETECTION_BACKEND", "dinox")

    try:
        from tools.artifact_writers import write_photo_intel, load_issue_catalog
        from tools.scene_classifier_orchestrator import create_orchestrator_from_config
        from tools.vlm_client import get_model_configs_from_pipeline_config, create_vlm_client
    except Exception as exc:
        logger.error(f"Failed to import pipeline components: {exc}", exc_info=True)
        summary = {"success": False, "error": str(exc), "property_key": args.property_key}
        print(json.dumps(summary, ensure_ascii=False))
        return 1

    # Load issue catalog
    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)

    # Build candidate_provider for Pass 2d (embeddings-based catalog matching)
    candidate_provider = None
    if getattr(cfg, "USE_EMBEDDINGS_CATALOG", False):
        try:
            from tools.catalog_embeddings import CatalogEmbeddingsRetriever
            from dataclasses import asdict

            retriever = CatalogEmbeddingsRetriever(
                catalog_v2=catalog,
                model_name=getattr(cfg, "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                device=getattr(cfg, "EMBEDDINGS_DEVICE", "cpu"),
                trust_remote_code=getattr(cfg, "EMBEDDINGS_TRUST_REMOTE_CODE", False),
                default_topk=getattr(cfg, "EMBEDDINGS_TOPK", 10),
            )

            def candidate_provider(observation_text: str, context: dict) -> list:
                kind = (context.get("kind") or "").strip().lower()
                topk = context.get("top_k_candidates")
                scene_group = context.get("scene_group")
                allowed_groups = {scene_group} if scene_group else None
                allowed_kinds = {kind} if kind in ("defect", "upgrade") else None
                matches = retriever.retrieve_candidates(
                    observation_text,
                    topk=topk,
                    allowed_kinds=allowed_kinds,
                    allowed_groups=allowed_groups,
                )
                return [asdict(m) for m in matches]

            logger.info(f"Pass 2d candidate_provider ready (model={retriever.model_name}, items={len(retriever._items)})")
        except Exception as exc:
            logger.warning(f"Could not initialize embeddings retriever for Pass 2d: {exc}. Pass 2d will be skipped.")
            candidate_provider = None

    # Create orchestrator
    orchestrator = create_orchestrator_from_config(
        cfg,
        candidate_provider=candidate_provider,
        catalog_items=catalog.get("items"),
    )

    # Build run options from CLI args + profile
    options = SceneClassifierRunOptions.from_analysis_profile(
        analysis_profile=analysis_profile,
        toggles=pass_toggles if pass_toggles else None,
        model_overrides=model_overrides if model_overrides else None,
    )

    # Get GPT config for artifact writing (Pass 2f, etc.)
    _, gpt5_config = get_model_configs_from_pipeline_config(cfg)
    vlm_client = create_vlm_client()

    # Generate job ID and create artifacts directory
    job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = artifacts_root / args.property_key / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-image analysis loop
    # ─────────────────────────────────────────────────────────────────────────
    results: List[ImageResult] = []
    total_start = time.time()

    async def _analyze_all():
        for idx, image_path in enumerate(images):
            img_start = time.time()
            logger.info(f"[{idx + 1}/{len(images)}] Analyzing: {image_path.name}")

            try:
                img_options = options.with_meta(
                    run_id=job_id,
                    photo_key=image_path.name,
                    property_key=args.property_key,
                )
                result = await orchestrator.analyze_image(
                    image_path=image_path,
                    options=img_options,
                )
                elapsed = time.time() - img_start
                result_dict = result.to_dict()

                results.append(ImageResult(
                    image_path=str(image_path),
                    scene_data=result_dict,
                    scene=result.scene or "unknown",
                    processing_time=elapsed,
                ))
                logger.info(f"  ✅ {image_path.name} → {result.scene} ({elapsed:.1f}s)")

            except Exception as exc:
                elapsed = time.time() - img_start
                logger.error(f"  ❌ {image_path.name} failed: {exc}", exc_info=args.debug)
                results.append(ImageResult(
                    image_path=str(image_path),
                    scene="unknown",
                    processing_time=elapsed,
                    error=str(exc),
                ))

    asyncio.run(_analyze_all())

    total_time = time.time() - total_start

    # ─────────────────────────────────────────────────────────────────────────
    # Build job and write artifacts
    # ─────────────────────────────────────────────────────────────────────────
    job = PropertyAnalysisJob(
        property_key=args.property_key,
        job_id=job_id,
        artifacts_dir=str(job_dir),
        timestamp=datetime.utcnow().isoformat() + "Z",
        results=results,
        total_processing_time=total_time,
    )

    try:
        photo_intel_path = write_photo_intel(
            cfg=cfg,
            job=job,
            detection_backend=detection_backend,
            analysis_profile=analysis_profile,
            use_pass_architecture=True,
            pass_toggles=pass_toggles if pass_toggles else {},
            model_overrides=model_overrides if model_overrides else {},
            gpt_config=gpt5_config,
            issue_catalog=catalog,
            vlm_client=vlm_client,
        )
    except Exception as exc:
        logger.error(f"Failed to write photo_intel: {exc}", exc_info=True)
        photo_intel_path = None

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