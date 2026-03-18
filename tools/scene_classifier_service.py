# tools/scene_classifier_service.py
"""
Service wrapper for scene classification via the pass-architecture orchestrator.

Provides:
  - SceneClassification dataclass (result container)
  - scene_classifier_payload() normalizer
  - parse_orchestrator_result() converter
  - SceneClassifierService class (sync wrapper around async orchestrator)
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.pipeline_common import maybe_backfill_planner_hints

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneClassification:
    scene: str
    reasoning: str
    keywords: List[str]
    prompt: str
    planner_targets: List[Dict[str, Any]]
    planner_hints: Dict[str, str]
    payload: Dict[str, Any]


def _run_coro_sync(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def scene_classifier_payload(
    scene_data: Optional[Dict[str, Any]],
    scene_override: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize scene classifier output so all fields are present."""
    payload: Dict[str, Any] = dict(scene_data or {})

    if scene_override:
        payload.setdefault("scene", scene_override)

    payload.setdefault("scene", "unknown")
    payload.setdefault("is_staged", None)
    payload.setdefault("overall_impression", "")
    payload.setdefault("image_summary", "")
    payload.setdefault("notable_features", [])
    payload.setdefault("feature_notes", "")
    payload.setdefault("positives_notes", "")
    payload.setdefault("observations_freeform", "")
    payload.setdefault("reasoning", "" if error is None else error)
    payload.setdefault("keywords", [])
    payload.setdefault("keyword_categories", None)
    payload.setdefault("issues_natural_language", [])
    payload.setdefault("verified_issues", [])
    payload.setdefault("catalog_flags", {})
    payload.setdefault("processing_time", payload.get("processing_time"))
    payload.setdefault("pass_timings", payload.get("pass_timings", {}))
    payload.setdefault("total_pass_time", payload.get("total_pass_time", 0.0))

    if error and not payload.get("error"):
        payload["error"] = error
    payload.setdefault("error", payload.get("error"))

    return payload


def parse_orchestrator_result(result: Any) -> SceneClassification:
    """Parse orchestrator ImageAnalysisResult into a SceneClassification."""
    # Handle both dataclass and dict results
    if hasattr(result, 'to_dict'):
        data = result.to_dict()
    elif hasattr(result, '__dict__'):
        data = result.__dict__
    else:
        data = dict(result) if isinstance(result, dict) else {}

    scene = data.get("scene", "unknown")
    reasoning = ""
    if hasattr(result, 'pass_1a') and result.pass_1a:
        reasoning = result.pass_1a.reasoning or ""

    keywords = data.get("keywords", []) or []

    # Robust category extraction
    kw_cats = data.get("keyword_categories")

    # Build grounding prompt from keywords
    prompt = ". ".join(keywords) + "." if keywords else ""

    # Extract targets if available
    planner_targets: List[Dict[str, Any]] = []
    planner_hints: Dict[str, str] = {}

    # Build scene payload
    payload = scene_classifier_payload(data, scene_override=scene)
    payload['keywords'] = keywords
    payload['keyword_categories'] = kw_cats
    payload['overall_impression'] = data.get('overall_impression', '')
    payload['image_summary'] = data.get('image_summary', '')
    payload['notable_features'] = data.get('notable_features', []) or []
    payload['feature_notes'] = data.get('feature_notes', '') or data.get('positives_notes', '') or ""
    payload['positives_notes'] = payload['feature_notes']
    payload['observations_freeform'] = data.get('observations_freeform', '') or ""
    payload['catalog_flags'] = data.get('catalog_flags', {})
    payload['issues_natural_language'] = data.get('issues_natural_language', [])
    payload['verified_issues'] = data.get('verified_issues', [])
    payload['models_used'] = data.get('models_used', {})

    # V2 fields from orchestrator
    payload["features_struct"] = data.get("features_struct", {}) or {}
    payload["observations_struct"] = data.get("observations_struct", {}) or {}
    payload["labeled_debug"] = data.get("labeled_debug", []) or []
    payload["labeled_forward"] = data.get("labeled_forward", []) or []
    payload["resolved_items"] = data.get("resolved_items", []) or []

    # Meta
    payload["passes_run"] = data.get("passes_run", []) or []
    payload["pass_timings"] = data.get("pass_timings", {}) or {}
    payload["total_pass_time"] = data.get("total_pass_time", 0.0) or 0.0

    # Passes dict from orchestrator (always provided)
    payload["passes"] = data.get("passes", {})

    return SceneClassification(
        scene=scene,
        reasoning=reasoning,
        keywords=keywords,
        prompt=prompt,
        planner_targets=planner_targets,
        planner_hints=planner_hints,
        payload=payload,
    )


class SceneClassifierService:
    """
    Thin sync wrapper around the async orchestrator.
    AutoAnalyzer gives it:
      - orchestrator
      - run_options
      - cfg
    """
    def __init__(self, cfg: Any, orchestrator: Any, run_options: Any):
        self.cfg = cfg
        self.orchestrator = orchestrator
        self.run_options = run_options

    def classify(self, image_path: Path, *, meta: Dict[str, str]) -> SceneClassification:
        logger.info(f"Classifying scene: {image_path.name}")

        if not self.orchestrator:
            err = "Orchestrator not available"
            logger.error(err)
            payload = scene_classifier_payload(None, scene_override="unknown", error=err)
            return SceneClassification("unknown", err, [], "", [], {}, payload)

        options = self.run_options
        if options is not None and hasattr(options, "with_meta"):
            options = options.with_meta(**meta)

        logger.info(f"  Using orchestrator (premium={getattr(self.run_options, 'premium', 'N/A')})")

        async def _run():
            return await self.orchestrator.analyze_image(image_path=image_path, options=options)

        try:
            result = _run_coro_sync(_run())
            out = parse_orchestrator_result(result)

            hints = maybe_backfill_planner_hints(self.cfg, out.scene, out.planner_hints)
            payload = dict(out.payload or {})
            payload["planner_hints"] = hints

            return SceneClassification(
                scene=out.scene,
                reasoning=out.reasoning,
                keywords=out.keywords,
                prompt=out.prompt,
                planner_targets=out.planner_targets,
                planner_hints=hints,
                payload=payload,
            )
        except Exception as e:
            logger.error(f"  Orchestrator failed: {e}", exc_info=True)
            payload = scene_classifier_payload(None, scene_override="unknown", error=str(e))
            return SceneClassification("unknown", str(e), [], "", [], {}, payload)
