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

    result = await orchestrator.analyze_image(
        image_path=Path('/path/to/image.jpg'),
        options=SceneClassifierRunOptions(premium=True),
    )
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Callable, Dict, List, Optional

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
    Pass2cResult,
    Pass2dResult,
    Pass3Result,
    run_pass_1a_scene_type,
    run_pass_1b_feature_notes,
    run_pass_1c_feature_structuring,
    run_pass_2a,
    run_pass_2b,
    run_pass_2c,
    run_pass_2d,
    run_pass_3_keyword_extraction,
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
    pass_2c: Optional[Pass2cResult] = None
    pass_2d: Optional[List[Pass2dResult]] = None  # List because one per defect observation
    pass_3: Optional[Pass3Result] = None

    # Computed/merged fields for backwards compatibility
    scene: str = "other"

    # Structured positives (from 1c)
    overall_impression: str = ""
    image_summary: str = ""
    notable_features: List[str] = field(default_factory=list)

    # Raw notes
    feature_notes: str = ""
    positives_notes: str = ""  # legacy alias, keep temporarily
    observations_freeform: str = ""

    # Structured outputs (v2)
    features_struct: Dict[str, Any] = field(default_factory=dict)
    observations_struct: Dict[str, Any] = field(default_factory=dict)

    labeled_debug: List[Dict[str, Any]] = field(default_factory=list)
    labeled_forward: List[Dict[str, Any]] = field(default_factory=list)

    # Optional resolver output (2d). Orchestrator stores results if run elsewhere.
    resolved_defects: List[Dict[str, Any]] = field(default_factory=list)

    keywords: List[str] = field(default_factory=list)

    # Metadata
    passes_run: List[str] = field(default_factory=list)
    models_used: Dict[str, str] = field(default_factory=dict)
    pass_timings: Dict[str, float] = field(default_factory=dict)
    total_pass_time: float = 0.0
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

            # raw notes
            "feature_notes": self.feature_notes,
            "positives_notes": self.positives_notes,  # legacy
            "observations_freeform": self.observations_freeform,

            # structured outputs (v2)
            "features_struct": self.features_struct,
            "observations_struct": self.observations_struct,
            "labeled_debug": self.labeled_debug,
            "labeled_forward": self.labeled_forward,
            "resolved_defects": self.resolved_defects,

            "keywords": self.keywords,
            "passes_run": self.passes_run,
            "models_used": self.models_used,
            "pass_timings": self.pass_timings,
            "total_pass_time": self.total_pass_time,
            "processing_time": self.processing_time,
        }


@dataclass
class PropertyAnalysisResult:
    """Complete analysis result for a property (all images)."""
    property_key: str
    image_results: List[ImageAnalysisResult] = field(default_factory=list)
    total_processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_key": self.property_key,
            "total_processing_time": self.total_processing_time,
            "images": {r.image_path: r.to_dict() for r in self.image_results},
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
        max_keywords: int = 20,
        candidate_provider: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
        top_k_candidates: int = 8,
        max_resolve_per_image: int = 25,
    ):
        """
        Initialize the orchestrator.

        Args:
            qwen_config: Configuration for Qwen model calls
                         {'url': '...', 'model': '...'}
            gpt5_config: Configuration for GPT-5 model calls
                         {'url': '...', 'model': '...', 'api_key': '...'}
            vlm_client: VLM client instance for making API calls
            max_keywords: Maximum keywords for Pass 3
            candidate_provider: Optional callback to retrieve defect candidates for Pass 2d
                               Signature: (observation_text, context) -> List[Dict]
            top_k_candidates: Number of candidates to retrieve per observation
            max_resolve_per_image: Maximum observations to resolve per image in Pass 2d
        """
        self.qwen_config = qwen_config
        self.gpt5_config = gpt5_config
        self.vlm_client = vlm_client
        self.max_keywords = max_keywords
        self.candidate_provider = candidate_provider
        self.top_k_candidates = top_k_candidates
        self.max_resolve_per_image = max_resolve_per_image

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
            "2b": "OPENAI_PASS_2B_MAX_TOKENS",
            "2c": "OPENAI_PASS_2C_MAX_TOKENS",
            "2d": "OPENAI_PASS_2D_MAX_TOKENS",
            "3":  "OPENAI_PASS_3_MAX_TOKENS",
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
    ) -> ImageAnalysisResult:
        """
        Run all enabled passes on a single image.

        Args:
            image_path: Path to the image file
            options: Run options (premium, toggles, overrides)

        Returns:
            ImageAnalysisResult with all pass results
        """
        import time
        start_time = time.time()

        options = options or SceneClassifierRunOptions()
        toggles = options.toggles

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
            t0 = time.time()
            result.pass_1a = await run_pass_1a_scene_type(
                image_path=image_path,
                vlm_client=self.vlm_client,
                model_config=model_config,
            )
            result.pass_timings['1a'] = round(time.time() - t0, 3)

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
            t0 = time.time()
            result.pass_1b = await run_pass_1b_feature_notes(
                image_path=image_path,
                vlm_client=self.vlm_client,
                model_config=model_config,
                context=context,
            )
            result.pass_timings['1b'] = round(time.time() - t0, 3)

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
            t0 = time.time()
            result.pass_1c = await run_pass_1c_feature_structuring(
                vlm_client=self.vlm_client,
                model_config=model_config,
                feature_notes=feature_notes,
            )
            result.pass_timings['1c'] = round(time.time() - t0, 3)

            # Pass 1c now produces overall_impression, image_summary, notable_features
            result.overall_impression = result.pass_1c.overall_impression
            result.image_summary = result.pass_1c.image_summary
            result.notable_features = result.pass_1c.notable_features or []

            # Store features_struct for Streamlit parity
            result.features_struct = {
                "overall_impression": result.overall_impression,
                "image_summary": result.image_summary,
                "notable_features": result.notable_features,
            }
            context["features_struct"] = result.features_struct

            result.passes_run.append('1c')
            result.models_used['1c'] = model_name

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2a: Observations Freeform (vision)
        # ─────────────────────────────────────────────────────────────────────
        observations_freeform = ""

        if toggles['2a']:
            model_config = self._get_model_config('2a', options)
            model_name = self._get_model_name('2a', options)

            logger.debug(f"Running Pass 2a with {model_name}")
            t0 = time.time()
            result.pass_2a = await run_pass_2a(
                image_path=image_path,
                vlm_client=self.vlm_client,
                model_config=model_config,
                context=context,
            )
            result.pass_timings['2a'] = round(time.time() - t0, 3)

            observations_freeform = result.pass_2a.observations_freeform
            result.observations_freeform = observations_freeform
            result.passes_run.append('2a')
            result.models_used['2a'] = model_name

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2b: Observations -> JSON (text-only)
        # ─────────────────────────────────────────────────────────────────────
        if toggles['2b']:
            model_config = self._get_model_config('2b', options)
            model_name = self._get_model_name('2b', options)

            logger.debug(f"Running Pass 2b with {model_name}")
            t0 = time.time()
            result.pass_2b = await run_pass_2b(
                vlm_client=self.vlm_client,
                model_config=model_config,
                observations_freeform=observations_freeform,
            )
            result.pass_timings['2b'] = round(time.time() - t0, 3)

            observations_list = result.pass_2b.observations or []
            result.observations_struct = {"observations": observations_list}
            context["observations_struct"] = result.observations_struct

            result.passes_run.append('2b')
            result.models_used['2b'] = model_name

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2c: Label Observations + Debug/Forward Split (text-only)
        # ─────────────────────────────────────────────────────────────────────
        if toggles['2c']:
            model_config = self._get_model_config('2c', options)
            model_name = self._get_model_name('2c', options)

            observations_in = []
            if isinstance(result.observations_struct, dict):
                observations_in = result.observations_struct.get("observations") or []

            logger.debug(f"Running Pass 2c with {model_name}")
            t0 = time.time()
            result.pass_2c = await run_pass_2c(
                vlm_client=self.vlm_client,
                model_config=model_config,
                observations=observations_in,
            )
            result.pass_timings['2c'] = round(time.time() - t0, 3)

            result.labeled_debug = result.pass_2c.labeled_debug or []
            result.labeled_forward = result.pass_2c.labeled_forward or []

            context["labeled_debug"] = result.labeled_debug
            context["labeled_forward"] = result.labeled_forward

            result.passes_run.append('2c')
            result.models_used['2c'] = model_name

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2d: Resolve defect_id from candidates (text-only, optional)
        # ─────────────────────────────────────────────────────────────────────
        if toggles['2d'] and self.candidate_provider:
            model_config = self._get_model_config('2d', options)
            model_name = self._get_model_name('2d', options)

            # Only resolve defect_or_damage observations
            defects_to_resolve = [
                obs for obs in result.labeled_forward
                if obs.get("label") == "defect_or_damage"
            ][:self.max_resolve_per_image]

            if defects_to_resolve:
                logger.debug(f"Running Pass 2d with {model_name} for {len(defects_to_resolve)} defects")
                t0 = time.time()

                pass_2d_results: List[Pass2dResult] = []
                resolved_defects: List[Dict[str, Any]] = []

                # Context for candidate provider
                ctx_for_provider = {**context, "top_k_candidates": self.top_k_candidates}

                for obs in defects_to_resolve:
                    description = obs.get("description", "")
                    if not description:
                        continue

                    # Retrieve candidates via provider
                    candidates = self.candidate_provider(description, ctx_for_provider)

                    if not candidates:
                        continue

                    # Call Pass 2d
                    pass_2d_result = await run_pass_2d(
                        vlm_client=self.vlm_client,
                        model_config=model_config,
                        observation=description,
                        candidates=candidates,
                    )
                    pass_2d_results.append(pass_2d_result)

                    # Get top candidate for debugging
                    top_candidate = candidates[0] if candidates else None

                    resolved_defects.append({
                        "description": description,
                        "label": "defect_or_damage",
                        "resolved_defect_id": pass_2d_result.resolved_defect_id,
                        "top_candidate_id": top_candidate.get("defect_id") if top_candidate else None,
                        "top_candidate_score": top_candidate.get("score") if top_candidate else None,
                        "candidates": candidates,
                        "raw_response": pass_2d_result.raw_response,
                    })

                result.pass_timings['2d'] = round(time.time() - t0, 3)
                result.pass_2d = pass_2d_results
                result.resolved_defects = resolved_defects
                result.passes_run.append('2d')
                result.models_used['2d'] = model_name

                logger.debug(f"Pass 2d resolved {len(resolved_defects)} defects")
            else:
                logger.debug("Pass 2d: no defect_or_damage observations to resolve.")

        # ─────────────────────────────────────────────────────────────────────
        # Pass 3: Keyword Extraction (text-only, from structured facts)
        # ─────────────────────────────────────────────────────────────────────
        if toggles['3']:
            model_config = self._get_model_config('3', options)
            model_name = self._get_model_name('3', options)

            # ensure context contains what Pass 3 expects
            context.setdefault("scene", result.scene)
            context.setdefault("features_struct", result.features_struct)

            # For pass3: prefer labeled_forward if present, else observations_struct
            context.setdefault("labeled_forward", result.labeled_forward)
            if isinstance(result.observations_struct, dict):
                context.setdefault("observations", result.observations_struct.get("observations") or [])
            else:
                context.setdefault("observations", [])

            logger.debug(f"Running Pass 3 with {model_name}")
            t0 = time.time()
            result.pass_3 = await run_pass_3_keyword_extraction(
                vlm_client=self.vlm_client,
                model_config=model_config,
                context=context,
                max_keywords=self.max_keywords,
            )
            result.pass_timings['3'] = round(time.time() - t0, 3)

            result.keywords = result.pass_3.keywords
            result.passes_run.append('3')
            result.models_used['3'] = model_name

        result.total_pass_time = round(sum(result.pass_timings.values()), 3)
        result.processing_time = time.time() - start_time
        logger.info(
            f"Completed {image_path.name}: scene={result.scene}, "
            f"forward_obs={len(result.labeled_forward)}, "
            f"time={result.processing_time:.1f}s (LLM={result.total_pass_time:.1f}s)"
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
        Run analysis on all images for a property.

        Args:
            property_key: Property identifier
            image_paths: List of image paths to analyze
            options: Run options
            on_progress: Optional callback for progress updates

        Returns:
            PropertyAnalysisResult with all image results
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
            f"time={result.total_processing_time:.1f}s"
        )

        return result

# ═══════════════════════════════════════════════════════════════════════════════
# Factory function for easy instantiation
# ═══════════════════════════════════════════════════════════════════════════════

def create_orchestrator_from_config(
    config: Any,
    candidate_provider: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
) -> SceneClassifierOrchestrator:
    """
    Create an orchestrator from a pipeline_config module.

    Args:
        config: pipeline_config module with LM_STUDIO_URL, etc.
        candidate_provider: Optional callback to retrieve defect candidates for Pass 2d

    Returns:
        Configured SceneClassifierOrchestrator
    """
    from tools.vlm_client import (
        create_vlm_client,
        get_model_configs_from_pipeline_config,
    )

    qwen_config, gpt5_config = get_model_configs_from_pipeline_config(config)
    vlm_client = create_vlm_client()

    return SceneClassifierOrchestrator(
        qwen_config=qwen_config,
        gpt5_config=gpt5_config,
        vlm_client=vlm_client,
        max_keywords=getattr(config, "MAX_KEYWORDS", 20),
        candidate_provider=candidate_provider,
        top_k_candidates=getattr(config, "TOP_K_CANDIDATES", 8),
        max_resolve_per_image=getattr(config, "MAX_RESOLVE_PER_IMAGE", 25),
    )