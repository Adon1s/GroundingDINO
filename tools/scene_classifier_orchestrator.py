"""
Scene Classifier Pipeline Orchestrator
---------------------------------------
Coordinates the execution of all passes with proper model selection
and toggle handling.

Usage:
    from scene_classifier_orchestrator import SceneClassifierOrchestrator
    from vlm_client import create_vlm_client

    orchestrator = SceneClassifierOrchestrator(
        qwen_config={'url': '...', 'model': '...'},
        gpt5_config={'url': '...', 'model': '...', 'api_key': '...'},
        vlm_client=create_vlm_client(),
    )

    result = await orchestrator.analyze_image(f
        image_path=Path('/path/to/image.jpg'),
        options=SceneClassifierRunOptions(premium=True),
    )
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Dict, List, Optional

try:
    from tools import pipeline_config as cfg
except ImportError:
    cfg = None

from tools.pass_config import (
    PassKey,
    PassToggles,
    PassModelOverrides,
    SceneClassifierRunOptions,
    pick_model_for_pass,
    get_model_config_for_pass,
    describe_run_plan,
    ALL_PASSES,
)

from scene_classifier_passes import (
    Pass1aResult,
    Pass1bResult,
    Pass1cResult,
    Pass2aResult,
    Pass2bResult,
    Pass3Result,
    Pass4Result,
    run_pass_1a_scene_type,
    run_pass_1b_feature_notes,
    run_pass_1c_feature_structuring,
    run_pass_2a_issue_detection,
    run_pass_2b_issue_verification,
    run_pass_3_keyword_extraction,
    run_pass_4_property_summary,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysisResult:
    """Complete analysis result for a single image."""
    image_path: str

    # Pass results (None if pass was disabled)
    pass_1a: Optional[Pass1aResult] = None
    pass_1b: Optional[Pass1bResult] = None
    pass_1c: Optional[Pass1cResult] = None
    pass_2a: Optional[Pass2aResult] = None
    pass_2b: Optional[Pass2bResult] = None
    pass_3: Optional[Pass3Result] = None

    # Computed/merged fields for backwards compatibility
    scene: str = "other"

    # Structured positives (from 1c)
    overall_impression: str = ""
    image_summary: str = ""
    notable_features: List[str] = field(default_factory=list)

    # Raw notes (for Pass 4 notes-only summary)
    feature_notes: str = ""
    positives_notes: str = ""  # legacy alias, keep temporarily
    issues_notes: str = ""

    keywords: List[str] = field(default_factory=list)
    catalog_flags: Dict[str, Any] = field(default_factory=dict)
    verified_issues: List[Dict[str, Any]] = field(default_factory=list)
    issues_natural_language: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    passes_run: List[str] = field(default_factory=list)
    models_used: Dict[str, str] = field(default_factory=dict)
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_path": self.image_path,
            "scene": self.scene,

            # structured positives (UI + pass3)
            "overall_impression": self.overall_impression,
            "image_summary": self.image_summary,
            "notable_features": self.notable_features,

            # raw notes (pass4)
            "feature_notes": self.feature_notes,
            "positives_notes": self.positives_notes,  # legacy
            "issues_notes": self.issues_notes,

            "keywords": self.keywords,
            "catalog_flags": self.catalog_flags,
            "verified_issues": self.verified_issues,
            "issues_natural_language": self.issues_natural_language,
            "passes_run": self.passes_run,
            "models_used": self.models_used,
            "processing_time": self.processing_time,
        }


@dataclass
class PropertyAnalysisResult:
    """Complete analysis result for a property (all images + summary)."""
    property_key: str
    image_results: List[ImageAnalysisResult] = field(default_factory=list)
    pass_4: Optional[Pass4Result] = None

    # Aggregated fields
    property_summary: str = ""
    investment_considerations: List[str] = field(default_factory=list)
    estimated_condition: str = ""
    confidence: Optional[float] = None
    total_issues: int = 0
    total_processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'property_key': self.property_key,
            'property_summary': self.property_summary,
            'investment_considerations': self.investment_considerations,
            'estimated_condition': self.estimated_condition,
            'confidence': self.confidence,
            'total_issues': self.total_issues,
            'total_processing_time': self.total_processing_time,
            'images': {r.image_path: r.to_dict() for r in self.image_results},
        }


class SceneClassifierOrchestrator:
    """
    Orchestrates scene classification passes with configurable model selection.

    Handles:
    - Per-pass enable/disable via toggles
    - Per-pass model selection (Qwen vs GPT-5)
    - Premium vs standard profile routing
    - Development overrides for testing
    """

    def __init__(
        self,
        qwen_config: Dict[str, Any],
        gpt5_config: Dict[str, Any],
        vlm_client: Any,
        issue_catalog: Optional[Dict[str, Any]] = None,
        max_keywords: int = 20,
    ):
        """
        Initialize the orchestrator.

        Args:
            qwen_config: Configuration for Qwen model calls
                         {'url': '...', 'model': '...'}
            gpt5_config: Configuration for GPT-5 model calls
                         {'url': '...', 'model': '...', 'api_key': '...'}
            vlm_client: VLM client instance for making API calls
            issue_catalog: Issue catalog for Pass 2a
            max_keywords: Maximum keywords for Pass 3
        """
        self.qwen_config = qwen_config
        self.gpt5_config = gpt5_config
        self.vlm_client = vlm_client
        self.issue_catalog = issue_catalog
        self.max_keywords = max_keywords

    def _attach_openai_token_cap(self, pass_key: PassKey, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Only apply max_tokens for OpenAI calls.
        Leave LM Studio/Qwen untouched.
        """
        # Harden provider detection: infer OpenAI if api_key present but provider missing
        provider = model_config.get("provider")
        if provider is None and model_config.get("api_key"):
            provider = "openai"

        if provider != "openai":
            return model_config

        key_map = {
            "1b": "OPENAI_PASS_1B_MAX_TOKENS",
            "1c": "OPENAI_PASS_1C_MAX_TOKENS",
            "2a": "OPENAI_PASS_2A_MAX_TOKENS",
            "4": "OPENAI_PASS_4_MAX_TOKENS",
        }

        attr = key_map.get(str(pass_key))
        cap = None

        if attr:
            if cfg:
                cap = getattr(cfg, attr, None)
            if cap is None:
                cap = os.environ.get(attr)

        if cap is None:
            if cfg:
                cap = getattr(cfg, "OPENAI_DEFAULT_MAX_TOKENS", None)
            if cap is None:
                cap = os.environ.get("OPENAI_DEFAULT_MAX_TOKENS")

        if cap:
            try:
                cap_int = int(cap)
                return {
                    **model_config,
                    # Keep this for any internal code that expects it
                    "max_tokens": cap_int,
                    # This matches the actual OpenAI Responses payload
                    "max_output_tokens": cap_int,
                }
            except Exception:
                logger.warning(f"Invalid OpenAI cap for pass {pass_key}: {cap}")

        return model_config

    def _get_model_config(
        self,
        pass_key: PassKey,
        options: SceneClassifierRunOptions,
    ) -> Dict[str, Any]:
        """Get the model config for a specific pass."""
        base = get_model_config_for_pass(
            pass_key=pass_key,
            options=options,
            qwen_config=self.qwen_config,
            gpt5_config=self.gpt5_config,
        )

        # If this pass is routed to OpenAI, honor pass-specific model strings from cfg
        provider = base.get("provider")
        if provider is None and base.get("api_key"):
            provider = "openai"

        if provider == "openai" and cfg:
            attr = f"GPT_PASS_{str(pass_key).upper()}_MODEL"
            pass_model = getattr(cfg, attr, None)
            if pass_model:
                base = {**base, "model": pass_model}

        return self._attach_openai_token_cap(pass_key, base)

    def _get_model_name(
        self,
        pass_key: PassKey,
        options: SceneClassifierRunOptions,
    ) -> str:
        """Get the model name for logging."""
        return pick_model_for_pass(pass_key, options.premium, options.model_overrides)

    async def analyze_image(
        self,
        image_path: Path,
        options: Optional[SceneClassifierRunOptions] = None,
        issue_catalog: Optional[Dict[str, Any]] = None,
    ) -> ImageAnalysisResult:
        """
        Run all enabled passes on a single image.

        Args:
            image_path: Path to the image file
            options: Run options (premium, toggles, overrides)
            issue_catalog: Optional override for issue catalog (caller override > orchestrator default > empty)

        Returns:
            ImageAnalysisResult with all pass results
        """
        import time
        start_time = time.time()

        options = options or SceneClassifierRunOptions()
        toggles = options.toggles

        # ✅ Resolve catalog: caller override > orchestrator default > empty
        resolved_catalog = issue_catalog or self.issue_catalog or {}

        result = ImageAnalysisResult(image_path=str(image_path))
        context: Dict[str, Any] = {}

        logger.info(f"Analyzing image: {image_path.name}")
        logger.debug(describe_run_plan(options))

        # ─────────────────────────────────────────────────────────────────────
        # Pass 1a: Scene Type Classification
        # ─────────────────────────────────────────────────────────────────────
        if toggles['1a']:
            model_config = self._get_model_config('1a', options)
            model_name = self._get_model_name('1a', options)

            logger.debug(f"Running Pass 1a with {model_name}")
            result.pass_1a = await run_pass_1a_scene_type(
                image_path=image_path,
                vlm_client=self.vlm_client,
                model_config=model_config,
            )

            result.scene = result.pass_1a.scene
            context['scene'] = result.scene
            result.passes_run.append('1a')
            result.models_used['1a'] = model_name

        # ─────────────────────────────────────────────────────────────────────
        # Pass 1b: Feature/Market Appeal Notes (FREEFORM)
        # ─────────────────────────────────────────────────────────────────────
        feature_notes = ""

        if toggles['1b']:
            model_config = self._get_model_config('1b', options)
            model_name = self._get_model_name('1b', options)

            logger.debug(f"Running Pass 1b with {model_name}")
            result.pass_1b = await run_pass_1b_feature_notes(
                image_path=image_path,
                vlm_client=self.vlm_client,
                model_config=model_config,
                context=context,
            )

            feature_notes = result.pass_1b.feature_notes
            result.feature_notes = feature_notes
            result.positives_notes = feature_notes  # legacy alias
            result.passes_run.append('1b')
            result.models_used['1b'] = model_name

        # ─────────────────────────────────────────────────────────────────────
        # Pass 1c: Feature Notes -> JSON Structuring (text-only)
        # ─────────────────────────────────────────────────────────────────────
        if toggles['1c']:
            model_config = self._get_model_config('1c', options)
            model_name = self._get_model_name('1c', options)

            logger.debug(f"Running Pass 1c with {model_name}")
            result.pass_1c = await run_pass_1c_feature_structuring(
                vlm_client=self.vlm_client,
                model_config=model_config,
                feature_notes=feature_notes,
            )

            # Back-compat fields for UI
            result.overall_impression = result.pass_1c.overall_impression or ""
            result.image_summary = result.pass_1c.image_summary or ""
            result.notable_features = result.pass_1c.notable_features or []

            # Context for Pass 3
            context["notable_features"] = result.notable_features

            result.passes_run.append('1c')
            result.models_used['1c'] = model_name

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2a: Issue Detection (freeform notes)
        # ─────────────────────────────────────────────────────────────────────
        issues_notes = ""

        if toggles['2a']:
            model_config = self._get_model_config('2a', options)
            model_name = self._get_model_name('2a', options)

            logger.debug(f"Running Pass 2a with {model_name}")
            result.pass_2a = await run_pass_2a_issue_detection(
                image_path=image_path,
                vlm_client=self.vlm_client,
                model_config=model_config,
                context=context,
                issue_catalog=resolved_catalog,
            )

            issues_notes = result.pass_2a.issues_notes
            result.issues_notes = issues_notes
            result.passes_run.append('2a')
            result.models_used['2a'] = model_name

            logger.debug(f"Pass 2a freeform length (orchestrator): {len(issues_notes)}")

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2b: Freeform to JSON Conversion
        # ─────────────────────────────────────────────────────────────────────
        if toggles['2b']:
            model_config = self._get_model_config('2b', options)
            model_name = self._get_model_name('2b', options)

            logger.debug(f"Running Pass 2b with {model_name}")
            result.pass_2b = await run_pass_2b_issue_verification(
                image_path=image_path,
                vlm_client=self.vlm_client,
                model_config=model_config,
                freeform_notes=issues_notes,
                context=context,
                issue_catalog=resolved_catalog,
            )

            # ✅ Store structured output for downstream use
            result.catalog_flags = result.pass_2b.catalog_flags
            # Store in both fields for compatibility
            result.issues_natural_language = result.pass_2b.issues_natural_language
            result.verified_issues = result.pass_2b.issues_natural_language

            # Context for Pass 3
            context["issues_natural_language"] = result.issues_natural_language

            result.passes_run.append('2b')
            result.models_used['2b'] = model_name

            logger.debug(
                f"Pass 2b issues={len(result.pass_2b.issues_natural_language)} "
                f"flags={len(result.pass_2b.catalog_flags)}"
            )

        # ─────────────────────────────────────────────────────────────────────
        # Pass 3: Keyword Extraction (text-only, from structured facts)
        # ─────────────────────────────────────────────────────────────────────
        if toggles['3']:
            model_config = self._get_model_config('3', options)
            model_name = self._get_model_name('3', options)

            # ensure context contains what Pass 3 expects
            context.setdefault("scene", result.scene)
            context.setdefault("notable_features", result.notable_features)
            context.setdefault("issues_natural_language", result.issues_natural_language)

            logger.debug(f"Running Pass 3 with {model_name}")
            result.pass_3 = await run_pass_3_keyword_extraction(
                vlm_client=self.vlm_client,
                model_config=model_config,
                context=context,
                max_keywords=self.max_keywords,
            )

            result.keywords = result.pass_3.keywords
            result.passes_run.append('3')
            result.models_used['3'] = model_name

        result.processing_time = time.time() - start_time
        logger.info(
            f"Completed {image_path.name}: scene={result.scene}, "
            f"issues={len(result.verified_issues)}, "
            f"time={result.processing_time:.1f}s"
        )

        return result

    async def analyze_property(
        self,
        property_key: str,
        image_paths: List[Path],
        options: Optional[SceneClassifierRunOptions] = None,
        on_progress: Optional[callable] = None,
    ) -> PropertyAnalysisResult:
        """
        Run analysis on all images for a property, then run Pass 4 summary.

        Args:
            property_key: Property identifier
            image_paths: List of image paths to analyze
            options: Run options
            on_progress: Optional callback for progress updates

        Returns:
            PropertyAnalysisResult with all image results and property summary
        """
        import time
        start_time = time.time()

        options = options or SceneClassifierRunOptions()

        logger.info(f"Analyzing property {property_key}: {len(image_paths)} images")
        logger.info(describe_run_plan(options))

        result = PropertyAnalysisResult(property_key=property_key)

        # Analyze each image
        for i, image_path in enumerate(image_paths):
            if on_progress:
                on_progress({
                    'itemsDone': i,
                    'itemsTotal': len(image_paths),
                    'currentImage': image_path.name,
                })

            image_result = await self.analyze_image(image_path, options)
            result.image_results.append(image_result)
            result.total_issues += len(image_result.verified_issues)

        # ─────────────────────────────────────────────────────────────────────
        # Pass 4: Property Summary
        # ─────────────────────────────────────────────────────────────────────
        if options.toggles['4'] and result.image_results:
            model_config = self._get_model_config('4', options)
            model_name = self._get_model_name('4', options)

            logger.debug(f"Running Pass 4 with {model_name}")

            # Aggregate image results for Pass 4
            all_results = {
                r.image_path: r.to_dict()
                for r in result.image_results
            }

            result.pass_4 = await run_pass_4_property_summary(
                vlm_client=self.vlm_client,
                model_config=model_config,
                all_results=all_results,
            )

            result.property_summary = result.pass_4.property_summary
            result.investment_considerations = result.pass_4.investment_considerations or []
            result.estimated_condition = result.pass_4.estimated_condition or ""
            result.confidence = getattr(result.pass_4, "confidence", None)

        result.total_processing_time = time.time() - start_time

        if on_progress:
            on_progress({
                'itemsDone': len(image_paths),
                'itemsTotal': len(image_paths),
                'status': 'complete',
            })

        logger.info(
            f"Completed property {property_key}: "
            f"{len(result.image_results)} images, "
            f"{result.total_issues} total issues, "
            f"time={result.total_processing_time:.1f}s"
        )

        return result

# ═══════════════════════════════════════════════════════════════════════════════
# Factory function for easy instantiation
# ═══════════════════════════════════════════════════════════════════════════════

def create_orchestrator_from_config(
    config: Any,
    issue_catalog: Optional[Dict[str, Any]] = None,
) -> SceneClassifierOrchestrator:
    """
    Create an orchestrator from a pipeline_config module.

    Args:
        config: pipeline_config module with LM_STUDIO_URL, etc.
        issue_catalog: Optional issue catalog to use (if None, loads from config path)

    Returns:
        Configured SceneClassifierOrchestrator
    """
    from tools.vlm_client import (
        create_vlm_client,
        get_model_configs_from_pipeline_config,
    )

    # Get model configs
    qwen_config, gpt5_config = get_model_configs_from_pipeline_config(config)

    # Create VLM client
    vlm_client = create_vlm_client()

    # ✅ Use provided catalog, or load from config path if not provided
    resolved_catalog = issue_catalog
    if resolved_catalog is None:
        catalog_path = getattr(config, 'ISSUE_CATALOG_PATH', None)
        if catalog_path:
            import json
            try:
                with open(catalog_path, 'r') as f:
                    resolved_catalog = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load issue catalog: {e}")

    return SceneClassifierOrchestrator(
        qwen_config=qwen_config,
        gpt5_config=gpt5_config,
        vlm_client=vlm_client,
        issue_catalog=resolved_catalog,
        max_keywords=getattr(config, 'MAX_KEYWORDS', 20),
    )