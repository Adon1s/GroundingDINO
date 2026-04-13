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

import hashlib
import logging
import os
from collections import defaultdict
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

from tools.scene_classifier_passes import (
    Pass1aResult,
    Pass1bResult,
    Pass1cResult,
    Pass2aResult,
    Pass2bResult,
    Pass2cResult,
    Pass2dResult,
    Pass2eResult,
    run_pass_1a_scene_type,
    run_pass_1b_feature_notes,
    run_pass_1c_feature_structuring,
    run_pass_2a,
    run_pass_2b,
    run_pass_2c,
    run_pass_2d,
    run_pass_2e,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic ID helpers
# ─────────────────────────────────────────────────────────────────────────────


def _photo_key_from_path(image_path: str) -> str:
    """Return just the filename portion of a path (e.g. 'photo_034.jpg')."""
    try:
        return Path(image_path).name
    except Exception:
        return str(image_path)


def _stable_hash_id(*parts: str, length: int = 16) -> str:
    """SHA-256 based deterministic short ID."""
    combined = "|".join(str(p) if p is not None else "" for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:length]


def _make_issue_id(run_id: str, photo_key: str, description: str,
                   location_hint: str, label: str, ordinal: int) -> str:
    """Deterministic issue ID for stable, joinable references across the pipeline."""
    return _stable_hash_id(run_id, photo_key, description, location_hint, label, str(ordinal), length=16)


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
    pass_2d: Optional[List[Pass2dResult]] = None  # List because one per resolvable observation

    # Computed/merged fields for backwards compatibility
    scene: str = "other"
    photo_key: str = ""  # filename only, e.g. "photo_034.jpg"

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
    resolved_items: List[Dict[str, Any]] = field(default_factory=list)    # unified: defects + upgrades

    # Display-filtered issues after Pass 2e (populated if 2e ran)
    verified_issues: List[Dict[str, Any]] = field(default_factory=list)
    # Canonical issues that passed sanity + exact dedupe (superset of verified_issues)
    matched_issues: List[Dict[str, Any]] = field(default_factory=list)
    canonical_issues: List[Dict[str, Any]] = field(default_factory=list)
    display_issues: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    passes_run: List[str] = field(default_factory=list)
    passes: Dict[str, Any] = field(default_factory=dict)   # per-pass structured output (mirrors direct-path schema)
    models_used: Dict[str, str] = field(default_factory=dict)
    pass_timings: Dict[str, float] = field(default_factory=dict)
    total_pass_time: float = 0.0
    processing_time: float = 0.0

    # Debug info for troubleshooting (serialized to JSON artifact)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_path": self.image_path,
            "photo_key": self.photo_key,
            "scene": self.scene,

            # structured positives
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
            "resolved_items": self.resolved_items,
            "verified_issues": self.verified_issues,
            "matched_issues": self.matched_issues,
            "canonical_issues": self.canonical_issues,
            "display_issues": self.display_issues,

            "passes_run": self.passes_run,
            "passes": self.passes,
            "models_used": self.models_used,
            "pass_timings": self.pass_timings,
            "total_pass_time": self.total_pass_time,
            "processing_time": self.processing_time,
            "debug": self.debug,
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
        candidate_provider: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
        top_k_candidates: int = 8,
        max_resolve_per_image: int = 25,
        catalog_items: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            qwen_config: Configuration for Qwen model calls
                         {'url': '...', 'model': '...'}
            gpt5_config: Configuration for GPT-5 model calls
                         {'url': '...', 'model': '...', 'api_key': '...'}
            vlm_client: VLM client instance for making API calls
            candidate_provider: Optional callback to retrieve catalog candidates for Pass 2d
                               Signature: (observation_text, context) -> List[Dict]
                               context may include 'kind' ("defect" or "upgrade") and 'top_k_candidates'
            top_k_candidates: Number of candidates to retrieve per observation
            max_resolve_per_image: Maximum observations to resolve per image in Pass 2d
            catalog_items: Optional list of catalog item dicts (from issue_catalog["items"]).
                           Used to build catalog_meta_by_id for Pass 2e policy gating.
        """
        self.qwen_config = qwen_config
        self.gpt5_config = gpt5_config
        self.vlm_client = vlm_client
        self.candidate_provider = candidate_provider
        self.top_k_candidates = top_k_candidates
        self.max_resolve_per_image = max_resolve_per_image

        # Build catalog metadata lookup for Pass 2e policy gating
        self.catalog_meta_by_id: Dict[str, Dict[str, Any]] = {}
        for item in (catalog_items or []):
            if not isinstance(item, dict):
                continue
            item_id = (item.get("id") or "").strip()
            if item_id:
                self.catalog_meta_by_id[item_id] = {
                    "tier": item.get("tier", "work"),
                    "drop_if_generic": bool(item.get("drop_if_generic", False)),
                    "defaultHidden": bool(item.get("defaultHidden", False)),
                    "kind": item.get("kind", "defect"),
                    "trade_bucket": item.get("trade_bucket", ""),
                }
        if self.catalog_meta_by_id:
            logger.info("Orchestrator: catalog_meta_by_id built with %d items", len(self.catalog_meta_by_id))

    @staticmethod
    def _t(toggles, key: str, default: bool = True) -> bool:
        """Robustly read a pass toggle, handling dict/dataclass/object."""
        if toggles is None:
            return default
        if isinstance(toggles, dict):
            return bool(toggles.get(key, default))
        # dataclass / object: try exact, then common naming patterns
        k = key.replace("-", "_")
        for name in (k, f"p{k}", f"_{k}"):
            if hasattr(toggles, name):
                return bool(getattr(toggles, name))
        return default

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
            "2e": "OPENAI_PASS_2E_MAX_TOKENS",
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
        result.photo_key = image_path.name
        context: Dict[str, Any] = {}

        logger.info(f"Analyzing image: {image_path.name}")
        logger.debug(describe_run_plan(options))

        # ─────────────────────────────────────────────────────────────────────
        # Pass 1a: Scene Type Classification
        # ─────────────────────────────────────────────────────────────────────
        if self._t(toggles, '1a'):
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
        # DISABLED — outputs not consumed downstream. Stubbed for compat.
        # ─────────────────────────────────────────────────────────────────────
        feature_notes = ""

        if self._t(toggles, '1b'):
            logger.debug("Pass 1b: skipped (disabled) — outputting blank stub")
            result.pass_1b = Pass1bResult(feature_notes="")
            result.pass_timings['1b'] = 0.0
            result.feature_notes = ""
            result.positives_notes = ""
            result.passes_run.append('1b')
            result.models_used['1b'] = "none"

        # ─────────────────────────────────────────────────────────────────────
        # Pass 1c: Feature Notes -> JSON Structuring (text-only)
        # DISABLED — outputs not consumed downstream. Stubbed for compat.
        # ─────────────────────────────────────────────────────────────────────
        if self._t(toggles, '1c'):
            logger.debug("Pass 1c: skipped (disabled) — outputting blank stub")
            result.pass_1c = Pass1cResult()
            result.pass_timings['1c'] = 0.0
            result.overall_impression = ""
            result.image_summary = ""
            result.notable_features = []
            result.features_struct = {
                "overall_impression": "",
                "image_summary": "",
                "notable_features": [],
            }
            context["features_struct"] = result.features_struct
            result.passes_run.append('1c')
            result.models_used['1c'] = "none"

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2a: Observations Freeform (vision)
        # ─────────────────────────────────────────────────────────────────────
        observations_freeform = ""

        if self._t(toggles, '2a'):
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
        if self._t(toggles, '2b'):
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
        if self._t(toggles, '2c'):
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
                scene=result.scene or "other",
            )
            result.pass_timings['2c'] = round(time.time() - t0, 3)

            result.labeled_debug = result.pass_2c.labeled_debug or []
            result.labeled_forward = result.pass_2c.labeled_forward or []

            context["labeled_debug"] = result.labeled_debug
            context["labeled_forward"] = result.labeled_forward

            # ── Stamp deterministic issue_id on every forward observation ──
            # Uses _make_issue_id() for stable, deterministic IDs.
            _run_id = (getattr(options, "meta", None) or {}).get("run_id", "")
            _photo_key = (getattr(options, "meta", None) or {}).get("photo_key") or image_path.name
            _sig_counts: Dict[tuple, int] = {}
            for _obs in (result.labeled_forward or []):
                if not isinstance(_obs, dict):
                    continue
                _desc = (_obs.get("description") or "").strip()
                if not _desc:
                    continue
                _label = (_obs.get("label") or "").strip()
                _loc = (_obs.get("location_hint") or "").strip()
                _sig = (_desc, _loc, _label)
                _ordinal = _sig_counts.get(_sig, 0)
                _sig_counts[_sig] = _ordinal + 1
                # Forward-only: only assign if missing
                if not _obs.get("issue_id"):
                    _obs["issue_id"] = _make_issue_id(_run_id, _photo_key, _desc, _loc, _label, _ordinal)
                _obs.setdefault("source_photo_key", _photo_key)

            result.passes_run.append('2c')
            result.models_used['2c'] = model_name

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2d: Resolve catalog item ID from candidates (text-only, optional)
        # ─────────────────────────────────────────────────────────────────────

        # ── Normalize kind + scene_group on every labeled_forward item ────────
        # Must happen before 2d gate so gating and retrieval use consistent values.
        # Ensures consistent values for gating and retrieval.
        _UPGRADE_LABELS = {
            "opportunity", "upgrade", "improvement", "cosmetic_upgrade",
            "feature", "upgrade_candidate",
        }

        def _label_to_kind(lbl: str) -> str:
            return "upgrade" if (lbl or "").strip().lower() in _UPGRADE_LABELS else "defect"

        # Minimal scene→group map
        _SCENE_TO_GROUP: Dict[str, str] = {
            "kitchen": "kitchen", "pantry": "kitchen",
            "bathroom": "bathroom",
            "bedroom": "bedroom", "closet": "bedroom",
            "living_room": "living_areas", "dining_room": "living_areas",
            "home_office": "living_areas", "hallway": "living_areas", "stairway": "living_areas",
            "laundry_room": "utility", "basement": "utility", "attic": "utility",
            "garage": "utility", "hvac": "utility",
            "exterior_front": "exterior", "exterior_back": "exterior", "exterior_side": "exterior",
            "yard": "exterior", "patio": "exterior", "deck": "exterior", "balcony": "exterior",
            "driveway": "exterior", "pool": "exterior", "garden": "exterior",
            "roof": "other", "other": "other", "unknown": "other",
            "floor_plan": "other", "aerial_view": "other", "street_view": "other",
        }
        _scene_for_2d = result.scene or "unknown"
        _scene_group_for_2d = _SCENE_TO_GROUP.get(_scene_for_2d, "other")

        for _obs in (result.labeled_forward or []):
            if not isinstance(_obs, dict):
                continue
            # Trust existing kind if already valid; derive from label otherwise
            _existing_kind = (_obs.get("kind") or "").strip().lower()
            _obs["kind"] = _existing_kind if _existing_kind in {"defect", "upgrade"} else _label_to_kind(_obs.get("label", ""))
            # Stamp scene_group so candidate_provider can gate retrieval
            _obs.setdefault("scene_group", _scene_group_for_2d)
            _obs.setdefault("scene", _scene_for_2d)
        pass_2d_toggle = self._t(toggles, "2d", default=True)
        pass_2d_provider_present = self.candidate_provider is not None

        labeled_forward = result.labeled_forward or []
        to_resolve_all = [
            obs for obs in labeled_forward
            if obs.get("kind") in {"defect", "upgrade"} and (obs.get("description") or "").strip()
        ][:self.max_resolve_per_image]

        # Persist into JSON so you can see it via website artifact
        result.debug["pass_2d_gate"] = {
            "toggle": pass_2d_toggle,
            "candidate_provider_present": pass_2d_provider_present,
            "labeled_forward_count": len(labeled_forward),
            "defect_forward_count": len([x for x in labeled_forward if x.get("kind") == "defect"]),
            "upgrade_forward_count": len([x for x in labeled_forward if x.get("kind") == "upgrade"]),
            "total_resolve_count": len(to_resolve_all),
        }

        # Use INFO so it shows up in typical web logs
        logger.info(
            "Pass 2d gate: toggle=%s provider=%s labeled_forward=%d defects=%d upgrades=%d total_resolve=%d",
            pass_2d_toggle,
            pass_2d_provider_present,
            result.debug["pass_2d_gate"]["labeled_forward_count"],
            result.debug["pass_2d_gate"]["defect_forward_count"],
            result.debug["pass_2d_gate"]["upgrade_forward_count"],
            result.debug["pass_2d_gate"]["total_resolve_count"],
        )

        if pass_2d_toggle and pass_2d_provider_present and to_resolve_all:
            model_config = self._get_model_config('2d', options)
            model_name = self._get_model_name('2d', options)

            logger.debug(f"Running Pass 2d with {model_name} for {len(to_resolve_all)} observations")
            t0 = time.time()

            pass_2d_results: List[Pass2dResult] = []
            resolved_items: List[Dict[str, Any]] = []    # unified list (defects + upgrades)

            # Base context for candidate provider — include scene so provider can gate retrieval
            base_ctx_for_provider = {
                **context,
                "top_k_candidates": self.top_k_candidates,
                "scene": result.scene,
                "scene_group": _scene_group_for_2d,
            }

            # Initialize per-observation debug list
            result.debug["pass_2d_per_observation"] = []

            for obs in to_resolve_all:
                description = (obs.get("description") or "").strip()
                if not description:
                    continue

                kind = obs.get("kind")   # already "defect" or "upgrade" from normalization above
                ctx_for_provider = {**base_ctx_for_provider, "kind": kind}

                debug_row = {
                    "observation": description,
                    "label": obs.get("label", ""),
                    "kind": kind,
                    "scene_group": obs.get("scene_group"),
                    "candidate_count": 0,
                    "skipped_reason": None,
                    "top_candidate_id": None,
                    "top_candidate_score": None,
                }

                # Retrieve candidates via provider (tolerant of signature variants)
                try:
                    candidates = self.candidate_provider(description, ctx_for_provider)
                except TypeError:
                    try:
                        candidates = self.candidate_provider(description)
                        debug_row["skipped_reason"] = "provider_ignored_context (signature lacks ctx; cannot control topk/kind)"
                    except Exception as e2:
                        debug_row["skipped_reason"] = f"candidate_provider_call_failed ({e2})"
                        result.debug["pass_2d_per_observation"].append(debug_row)
                        continue

                # If provider is async by accident, this will reveal it cleanly in JSON
                if hasattr(candidates, "__await__"):
                    debug_row["skipped_reason"] = "candidate_provider_returned_coroutine (provider must be sync or await it here)"
                    result.debug["pass_2d_per_observation"].append(debug_row)
                    continue

                if not isinstance(candidates, list):
                    debug_row["skipped_reason"] = f"candidate_provider_returned_nonlist ({type(candidates).__name__})"
                    result.debug["pass_2d_per_observation"].append(debug_row)
                    continue

                debug_row["candidate_count"] = len(candidates)

                if not candidates:
                    debug_row["skipped_reason"] = "no_candidates"
                    result.debug["pass_2d_per_observation"].append(debug_row)
                    continue

                # Keep only top_k if provider returns more
                candidates = candidates[: self.top_k_candidates]

                # ID key differs for upgrades; be tolerant
                top_candidate = candidates[0]
                top_id = (
                    top_candidate.get("item_id")
                    or top_candidate.get("defect_id")
                    or top_candidate.get("upgrade_id")
                    or top_candidate.get("id")
                )
                debug_row["top_candidate_id"] = top_id
                debug_row["top_candidate_score"] = top_candidate.get("score")
                result.debug["pass_2d_per_observation"].append(debug_row)

                # Call Pass 2d (pass kind so prompt and result are kind-aware)
                pass_2d_result = await run_pass_2d(
                    vlm_client=self.vlm_client,
                    model_config=model_config,
                    observation=description,
                    candidates=candidates,
                    kind=kind,
                )
                pass_2d_results.append(pass_2d_result)

                resolved_item_id = pass_2d_result.resolved_item_id  # canonical
                # issue_id must be stamped during Pass 2c — if it's missing something went wrong upstream.
                issue_id = (obs.get("issue_id") or "").strip()
                if not issue_id:
                    debug_row["skipped_reason"] = "missing_issue_id (expected stamped in 2c)"
                    result.debug["pass_2d_per_observation"].append(debug_row)
                    continue
                photo_key = _photo_key_from_path(str(image_path))
                row = {
                    "issue_id": issue_id,
                    "source_image_path": str(image_path),
                    "source_photo_key": photo_key,
                    "description": description,
                    "label": obs.get("label", ""),
                    "resolved_item_id": resolved_item_id,
                    "resolved_kind": kind,
                    # Candidates: keep for auditability (score retained for unmapped-issue debugging)
                    "candidates": [
                        {"item_id": c.get("item_id"), "name": c.get("name"), "trade_bucket": c.get("trade_bucket"), "score": c.get("score")}
                        for c in candidates
                    ],
                    "raw_response": pass_2d_result.raw_response,
                }
                resolved_items.append(row)

            result.pass_timings['2d'] = round(time.time() - t0, 3)
            result.pass_2d = pass_2d_results
            result.resolved_items = resolved_items
            result.passes_run.append('2d')
            result.models_used['2d'] = model_name

            n_defects = sum(1 for x in resolved_items if x.get("resolved_kind") == "defect")
            n_upgrades = sum(1 for x in resolved_items if x.get("resolved_kind") == "upgrade")
            logger.debug(f"Pass 2d resolved {len(resolved_items)} items ({n_defects} defects, {n_upgrades} upgrades)")

            # Add summary debug info
            result.debug["pass_2d_summary"] = {
                "attempted_total": len(to_resolve_all),
                "resolved_total": len(resolved_items),
                "resolved_defects": n_defects,
                "resolved_upgrades": n_upgrades,
            }
            logger.info(
                "Pass 2d summary: attempted=%d resolved=%d (defects=%d upgrades=%d)",
                result.debug["pass_2d_summary"]["attempted_total"],
                result.debug["pass_2d_summary"]["resolved_total"],
                n_defects,
                n_upgrades,
            )
        else:
            if to_resolve_all and not pass_2d_provider_present:
                logger.debug("Pass 2d: %d resolvable items but no candidate provider.", len(to_resolve_all))
            elif not to_resolve_all:
                logger.debug("Pass 2d: no resolvable observations (kind=defect/upgrade) in labeled_forward.")
            result.debug["pass_2d_summary"] = {
                "attempted_total": 0,
                "resolved_total": 0,
                "resolved_defects": 0,
                "resolved_upgrades": 0,
            }

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2e: Normalize / Filter / Deduplicate Verified Issues (rule-based)
        # Input:  labeled_forward (already has issue_id, kind, description stamped)
        # Output: result.verified_issues — clean, deduplicated, scoring-free
        # ─────────────────────────────────────────────────────────────────────
        if self._t(toggles, '2e'):
            model_config = self._get_model_config('2e', options)
            model_name = self._get_model_name('2e', options)

            # Build input: start from labeled_forward; enrich with resolution data where available
            resolution_index: Dict[str, Dict[str, Any]] = {
                row["issue_id"]: row
                for row in (result.resolved_items or [])
                if row.get("issue_id")
            }

            issues_for_2e: List[Dict[str, Any]] = []
            for obs in (result.labeled_forward or []):
                if not isinstance(obs, dict):
                    continue
                issue = dict(obs)  # shallow copy — don't mutate labeled_forward
                # Merge resolution data if available (catalogItemId etc.)
                res = resolution_index.get(str(issue.get("issue_id") or ""))
                if res and res.get("resolved_item_id"):
                    issue.setdefault("catalogItemId", res["resolved_item_id"])
                    issue.setdefault("catalogItemKind", res.get("resolved_kind"))
                # kind is already stamped by the normalization block above;
                # this setdefault is a safety net for any item that somehow slipped through.
                if not issue.get("kind"):
                    issue["kind"] = _label_to_kind(issue.get("label", ""))
                issues_for_2e.append(issue)

            # Inject catalog metadata and policy into context for Pass 2e
            if self.catalog_meta_by_id:
                context["catalog_meta_by_id"] = self.catalog_meta_by_id
            context["policy"] = {"include_optional": False, "mode": "renovator_strict"}

            t0 = time.time()
            try:
                pass_2e_result = await run_pass_2e(
                    vlm_client=self.vlm_client,
                    model_config=model_config,
                    verified_issues=issues_for_2e,
                    context=context,
                )
                result.verified_issues = pass_2e_result.display_issues or pass_2e_result.verified_issues or []
                result.matched_issues = pass_2e_result.canonical_issues or pass_2e_result.matched_issues or []
                result.canonical_issues = result.matched_issues
                result.display_issues = result.verified_issues
                result.passes_run.append('2e')
                result.models_used['2e'] = model_name

                removed = pass_2e_result.removed_invalid or pass_2e_result.removed or []
                suppressed = pass_2e_result.display_suppressed_issues or pass_2e_result.suppressed_issues or []
                result.passes["2e"] = {
                    "notes": pass_2e_result.notes,
                    "input_count": pass_2e_result.input_count,
                    "deduped_count": pass_2e_result.deduped_count,
                    "final_count": pass_2e_result.final_count,
                    "canonical_count": len(result.canonical_issues),
                    "display_count": len(result.display_issues),
                    "removed_count": pass_2e_result.removed_count,
                    "removed_reason_counts": pass_2e_result.removed_reason_counts,
                    "suppressed_reason_counts": pass_2e_result.suppressed_reason_counts,
                    "suppressed_samples": pass_2e_result.suppressed_samples,
                    "kept_issue_ids": [
                        x["issue_id"] for x in result.verified_issues if x.get("issue_id")
                    ],
                    "canonical_issue_ids": [
                        x["issue_id"] for x in result.canonical_issues if x.get("issue_id")
                    ],
                    "removed": [
                        {
                            "issue_id": x.get("issue_id"),
                            "description": x.get("description", ""),
                            "reason": x.get("removed_reason", ""),
                        }
                        for x in removed
                    ],
                    "suppressed": [
                        {
                            "issue_id": x.get("issue_id"),
                            "description": x.get("description", ""),
                            "reason": x.get("suppressed_reason", ""),
                        }
                        for x in suppressed
                    ],
                }
                result.debug["pass_2e_summary"] = {
                    "input_count": pass_2e_result.input_count,
                    "deduped_count": pass_2e_result.deduped_count,
                    "final_count": pass_2e_result.final_count,
                    "canonical_count": len(result.canonical_issues),
                    "display_count": len(result.display_issues),
                    "removed_count": pass_2e_result.removed_count,
                    "removed_reason_counts": pass_2e_result.removed_reason_counts,
                    "suppressed_reason_counts": pass_2e_result.suppressed_reason_counts,
                    "notes": pass_2e_result.notes,
                }
                logger.info(
                    "Pass 2e: input=%d matched=%d final=%d removed=%d suppressed=%s",
                    pass_2e_result.input_count,
                    pass_2e_result.deduped_count,
                    pass_2e_result.final_count,
                    pass_2e_result.removed_count,
                    pass_2e_result.suppressed_reason_counts,
                )
            except Exception as exc:
                logger.warning(f"Pass 2e failed: {exc}")
                # Fallback: pass labeled_forward through unchanged as verified_issues
                result.verified_issues = list(result.labeled_forward or [])
                result.matched_issues = list(result.labeled_forward or [])
                result.canonical_issues = result.matched_issues
                result.display_issues = result.verified_issues
                result.passes["2e"] = {"error": str(exc)}
                result.debug["pass_2e_summary"] = {"error": str(exc)}
            result.pass_timings['2e'] = round(time.time() - t0, 3)
        else:
            # 2e skipped — promote labeled_forward to verified_issues with minimal normalization
            # so downstream always gets a valid kind regardless of which path ran.
            _out: List[Dict[str, Any]] = []
            for _obs in (result.labeled_forward or []):
                if not isinstance(_obs, dict):
                    continue
                _x = dict(_obs)
                # Trust existing kind if valid; derive from label via _label_to_kind otherwise.
                # _label_to_kind is defined in the 2d normalization block above.
                _existing = (_x.get("kind") or "").strip().lower()
                _x["kind"] = _existing if _existing in {"defect", "upgrade"} else _label_to_kind(_x.get("label", ""))
                _out.append(_x)
            result.verified_issues = _out
            result.display_issues = _out
            result.matched_issues = list(_out)
            result.canonical_issues = result.matched_issues
            result.passes["2e"] = {"skipped": True}
            result.debug["pass_2e_summary"] = {"skipped": True}

        result.total_pass_time = round(sum(result.pass_timings.values()), 3)
        result.processing_time = time.time() - start_time
        logger.info(
            f"Completed {image_path.name}: scene={result.scene}, "
            f"forward_obs={len(result.labeled_forward)}, "
            f"time={result.processing_time:.1f}s (LLM={result.total_pass_time:.1f}s)"
        )

        return result

# ═══════════════════════════════════════════════════════════════════════════════
# Factory function for easy instantiation
# ═══════════════════════════════════════════════════════════════════════════════

def create_orchestrator_from_config(
    config: Any,
    candidate_provider: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
    catalog_items: Optional[List[Dict[str, Any]]] = None,
) -> SceneClassifierOrchestrator:
    """
    Create an orchestrator from a pipeline_config module.

    Args:
        config: pipeline_config module with LM_STUDIO_URL, etc.
        candidate_provider: Optional callback to retrieve catalog candidates for Pass 2d
        catalog_items: Optional list of catalog item dicts for Pass 2e policy gating

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
        candidate_provider=candidate_provider,
        top_k_candidates=getattr(config, "TOP_K_CANDIDATES", 8),
        max_resolve_per_image=getattr(config, "MAX_RESOLVE_PER_IMAGE", 25),
        catalog_items=catalog_items,
    )
