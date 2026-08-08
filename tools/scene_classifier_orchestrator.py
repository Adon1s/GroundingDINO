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
import inspect
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Callable, Dict, List, Optional

try:
    from tools import pipeline_config as cfg
except ImportError:
    cfg = None

from tools.pipeline_common import SCENE_TO_GROUP_UI

from tools.pass_config import (
    ALLOWED_PIPELINE_MODES,
    PIPELINE_MODE_CLASSIFICATION_ONLY,
    PIPELINE_MODE_PUBLISH,
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
    EXCLUSION_REASONS,
    OBSERVATION_KINDS,
    ONTOLOGY_VERSION,
    PASS_2B_PROMPT_SHA256,
    PASS_2B_PROMPT_VERSION,
    PASS_2C_PROMPT_SHA256,
    PASS_2C_PROMPT_VERSION,
    PASS_2D_PROMPT_SHA256,
    PASS_2D_PROMPT_VERSION,
    Pass1aResult,
    Pass1bResult,
    Pass1cResult,
    Pass2aResult,
    Pass2bResult,
    Pass2cResult,
    Pass2dResult,
    Pass2eResult,
    evaluate_kind_routing,
    PassExecutionError,
    _pass_failure,
    run_pass_1a_scene_type,
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


PROVIDER_IGNORED_CONTEXT = (
    "provider_ignored_context (signature lacks ctx; cannot control topk/kind)"
)


def _provider_accepts_context(provider: Any) -> bool:
    """True when *provider* takes the ``(description, context)`` contract.

    Decided by binding the signature rather than by calling and catching
    TypeError: a provider that raises TypeError *internally* must surface as a
    real failure, not be silently retried as a legacy one-argument provider.
    Non-introspectable callables (builtins, C extensions) are assumed to take
    the current two-argument contract.
    """
    try:
        signature = inspect.signature(provider)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind("", {})
    except TypeError:
        return False
    return True


def _retrieve_candidates(
    provider: Callable[..., Any],
    description: str,
    context: Dict[str, Any],
) -> tuple[Any, Optional[str]]:
    """Call the candidate provider once, tolerating the legacy one-arg signature.

    Returns ``(raw_result, legacy_note)``. Exceptions are deliberately *not*
    swallowed — the caller applies its own policy (Pass 2d treats a retrieval
    failure as a dependency failure).
    """
    if _provider_accepts_context(provider):
        return provider(description, context), None
    return provider(description), PROVIDER_IGNORED_CONTEXT


async def resolve_observation_against_catalog(
    *,
    vlm_client: Any,
    model_config: Dict[str, Any],
    candidate_provider: Callable[..., Any],
    observation: Dict[str, Any],
    base_context: Optional[Dict[str, Any]] = None,
    top_k: int = 8,
    source_image_path: str = "",
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[Pass2dResult]]:
    """Resolve one Pass 2c observation to a catalog item under strict exact-kind retrieval.

    Returns ``(resolved_row | None, debug_row, pass_2d_result | None)``. Exactly
    one debug row is produced per call, whatever the outcome.

    Fail-closed points, both raising ``PassExecutionError`` rather than
    degrading quietly:

    - an observation kind outside the ontology fails **before** any retrieval;
    - a candidate whose kind differs from the observation's is a broken filter,
      not a ranking nuisance — the whole point of exact-kind routing is that the
      pool is pure, so a leak must be loud.

    Shared by the orchestrator's benchmark-mode loop and the catalog-resolution
    benchmark, so the benchmark measures the production resolution path.
    """
    description = (observation.get("description") or "").strip()
    kind = (observation.get("kind") or "").strip().lower()
    scene_group = observation.get("scene_group") or (base_context or {}).get("scene_group")

    debug_row: Dict[str, Any] = {
        "observation": description,
        "kind": kind,
        "scene_group": scene_group,
        "candidate_count": 0,
        "skipped_reason": None,
        "top_candidate_id": None,
        "top_candidate_score": None,
        "routing_reason": None,
        "resolution_path": None,
        "shortcut_reason": None,
    }

    routing = evaluate_kind_routing(description, kind)
    debug_row["routing_reason"] = routing.reason
    if not routing.expanded_kinds:
        # Never retrieve on an invalid kind: an empty allowed-kinds set is the
        # one input that used to fall through as "search everything".
        raise PassExecutionError(
            '2d', 'parse',
            f"observation kind {kind!r} is not in the observation-kind-v2 ontology "
            f"{sorted(OBSERVATION_KINDS)}",
            code="invalid_kind",
        )

    ctx_for_provider = {
        **(base_context or {}),
        "kind": kind,
        "allowed_kinds": list(routing.expanded_kinds),
        "top_k_candidates": top_k,
        "scene_group": scene_group,
    }

    # A retrieval failure is a dependency failure, not "no candidates": skipping
    # it here would zero out this observation and read downstream as a photo
    # with nothing to resolve.
    try:
        candidates, legacy_note = _retrieve_candidates(
            candidate_provider, description, ctx_for_provider,
        )
    except Exception as exc:
        raise _pass_failure('2d', 'dependency', exc, model_config) from exc
    if legacy_note:
        debug_row["skipped_reason"] = legacy_note

    if hasattr(candidates, "__await__"):
        debug_row["skipped_reason"] = (
            "candidate_provider_returned_coroutine (provider must be sync or await it here)"
        )
        return None, debug_row, None

    if not isinstance(candidates, list):
        debug_row["skipped_reason"] = (
            f"candidate_provider_returned_nonlist ({type(candidates).__name__})"
        )
        return None, debug_row, None

    debug_row["candidate_count"] = len(candidates)
    if not candidates:
        debug_row["skipped_reason"] = "no_candidates"
        return None, debug_row, None

    candidates = candidates[:top_k]

    off_kind = sorted({
        str(c.get("kind") or "") for c in candidates
        if (c.get("kind") or "").strip().lower() != kind
    })
    if off_kind:
        raise PassExecutionError(
            '2d', 'dependency',
            f"candidate provider returned kinds {off_kind} for a {kind!r} observation; "
            "exact-kind retrieval requires a pure candidate pool",
            code="kind_purity_violation",
        )

    top_candidate = candidates[0]
    debug_row["top_candidate_id"] = (
        top_candidate.get("item_id")
        or top_candidate.get("defect_id")
        or top_candidate.get("upgrade_id")
        or top_candidate.get("id")
    )
    debug_row["top_candidate_score"] = top_candidate.get("score")

    pass_2d_result = await run_pass_2d(
        vlm_client=vlm_client,
        model_config=model_config,
        observation=description,
        candidates=candidates,
        kind=kind,
    )
    debug_row["resolution_path"] = pass_2d_result.resolution_path
    debug_row["shortcut_reason"] = pass_2d_result.shortcut_reason

    issue_id = (observation.get("issue_id") or "").strip()
    if not issue_id:
        # issue_id is stamped during Pass 2c; missing means something went wrong
        # upstream. The row is returned once — the caller appends it.
        debug_row["skipped_reason"] = "missing_issue_id (expected stamped in 2c)"
        return None, debug_row, pass_2d_result

    row = {
        "issue_id": issue_id,
        "source_image_path": source_image_path,
        "source_photo_key": (
            observation.get("source_photo_key") or _photo_key_from_path(source_image_path)
        ),
        "description": description,
        "resolved_item_id": pass_2d_result.resolved_item_id,
        "resolved_kind": pass_2d_result.resolved_kind or kind,
        "original_kind": kind,
        "routing_reason": routing.reason,
        "resolution_path": pass_2d_result.resolution_path,
        "shortcut_reason": pass_2d_result.shortcut_reason,
        # Candidates kept for auditability (score retained for unmapped-issue debugging)
        "candidates": [
            {
                "item_id": c.get("item_id"),
                "name": c.get("name"),
                "trade_bucket": c.get("trade_bucket"),
                "kind": c.get("kind"),
                "score": c.get("score"),
                "description": c.get("description"),
                "support_any": c.get("support_any"),
                "defaultHidden": c.get("defaultHidden"),
                "drop_if_generic": c.get("drop_if_generic"),
            }
            for c in candidates
        ],
        "raw_response": pass_2d_result.raw_response,
    }
    return row, debug_row, pass_2d_result


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

    # observation-kind-v2 lanes (Pass 2c output)
    observations: List[Dict[str, Any]] = field(default_factory=list)          # [{"description","kind",...}]
    excluded_observations: List[Dict[str, Any]] = field(default_factory=list)  # [{"description","reason"}]

    # The v2 pipeline ends after classification until the catalog migration
    # (Task 2) and downstream cutover (Task 3) land. Non-publishable:
    # artifact_writers.write_photo_intel rejects classification_only payloads.
    classification_only: bool = True
    ontology_version: str = ONTOLOGY_VERSION

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
    pass_states: Dict[str, str] = field(default_factory=dict)
    passes: Dict[str, Any] = field(default_factory=dict)   # per-pass structured output (mirrors direct-path schema)
    models_used: Dict[str, str] = field(default_factory=dict)
    pass_timings: Dict[str, float] = field(default_factory=dict)
    total_pass_time: float = 0.0
    processing_time: float = 0.0

    # Per-pass routing record. One entry per LLM pass that ran, in execution order.
    # Each entry has shape:
    #   {"pass": "2a", "model_family": "gpt5", "model": "gpt-5.4-mini", "source": "explicit_override"}
    # Used to verify per-pass model selection without trusting the global meta.models block.
    model_routing: List[Dict[str, Any]] = field(default_factory=list)

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
            "observations": self.observations,
            "excluded_observations": self.excluded_observations,
            "classification_only": self.classification_only,
            "ontology_version": self.ontology_version,
            "resolved_items": self.resolved_items,
            "verified_issues": self.verified_issues,
            "matched_issues": self.matched_issues,
            "canonical_issues": self.canonical_issues,
            "display_issues": self.display_issues,

            "passes_run": self.passes_run,
            "pass_states": self.pass_states,
            "passes": self.passes,
            "models_used": self.models_used,
            "pass_timings": self.pass_timings,
            "total_pass_time": self.total_pass_time,
            "processing_time": self.processing_time,
            "model_routing": self.model_routing,
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
                               context may include 'kind' (an OBSERVATION_KINDS value) and 'top_k_candidates'
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
                    "trade_bucket": item.get("trade_bucket", ""),
                }
        if self.catalog_meta_by_id:
            logger.info("Orchestrator: catalog_meta_by_id built with %d items", len(self.catalog_meta_by_id))

    @staticmethod
    def _t(toggles, key: str, default: bool = True) -> bool:
        """
        Read a pass toggle, handling dict / PassToggles / bare object.

        PassToggles.to_dict() is the canonical pass-key view. Guessing attribute
        names instead ("2d", "p2d", "_2d") silently missed the real field name
        (pass_2d), which made every CLI/server toggle a no-op here.
        """
        if toggles is None:
            return default
        if isinstance(toggles, dict):
            return bool(toggles.get(key, default))
        to_dict = getattr(toggles, "to_dict", None)
        if callable(to_dict):
            mapping = to_dict()
            if key in mapping:
                return bool(mapping[key])
            return default
        k = key.replace("-", "_")
        for name in (f"pass_{k}", k, f"p{k}", f"_{k}"):
            if hasattr(toggles, name):
                return bool(getattr(toggles, name))
        return default

    def _get_model_config(
        self,
        pass_key: PassKey,
        options: SceneClassifierRunOptions,
    ) -> Dict[str, Any]:
        """Get the model config for a specific pass."""
        # get_model_config_for_pass applies the OpenAI token/reasoning policy via
        # resolve_openai_invocation; per-run overrides already carry a concrete
        # OpenAI model name, and there is no cfg.GPT_PASS_* layer anymore.
        return get_model_config_for_pass(
            pass_key=pass_key,
            options=options,
            qwen_config=self.qwen_config,
            gpt5_config=self.gpt5_config,
        )

    def _get_model_name(
        self,
        pass_key: PassKey,
        options: SceneClassifierRunOptions,
    ) -> str:
        """Get the model name for logging."""
        return pick_model_for_pass(pass_key, options.premium, options.model_overrides)

    def _record_model_routing(
        self,
        pass_key: PassKey,
        options: SceneClassifierRunOptions,
        model_config: Dict[str, Any],
        result: "ImageAnalysisResult",
    ) -> None:
        """
        Record a per-pass model routing entry on the result.

        Each entry captures the *resolved* model family + name + source label so
        downstream consumers can verify routing without trusting the global
        `meta.models` block (which only carries one GPT model name).

        ``source`` is the dominant reason this pass ended up with this exact
        model, in priority order. These three are the only values this method
        can emit:
          - "explicit_override": ``options.model_overrides`` named a concrete
            model for this pass (per-run, via the analyzer's ``--model-map``).
          - "premium_default": premium profile mapping put this pass on its
            family.
          - "standard_default": standard profile mapping (the default path).

        Idempotent per pass key: if the record already exists for this pass, we
        leave the original in place. (Routing is constant for a given run, so
        repeated calls per image would otherwise duplicate entries.)
        """
        # Skip if already recorded for this pass.
        for existing in result.model_routing:
            if existing.get("pass") == str(pass_key):
                return

        model_family = pick_model_for_pass(
            pass_key, options.premium, options.model_overrides
        )

        # Source resolution.
        source: str
        override_set = bool(options.model_overrides and options.model_overrides[pass_key])
        if override_set:
            source = "explicit_override"
        elif options.premium:
            source = "premium_default"
        else:
            source = "standard_default"

        routing_entry = {
            "pass": str(pass_key),
            "model_family": str(model_family),
            "model": str(model_config.get("model") or ""),
            "source": source,
        }
        reasoning_effort = model_config.get("reasoning_effort")
        if reasoning_effort is not None:
            routing_entry["reasoning_effort"] = str(reasoning_effort)
        max_output_tokens = model_config.get("max_output_tokens")
        if max_output_tokens is not None:
            routing_entry["max_output_tokens"] = int(max_output_tokens)
        result.model_routing.append(routing_entry)

    async def analyze_image(
        self,
        image_path: Path,
        options: Optional[SceneClassifierRunOptions] = None,
    ) -> ImageAnalysisResult:
        """
        Run all enabled passes on a single image.

        Failure boundary for PassExecutionError. A failed pass leaves its
        dependents without valid input, so the remaining passes are abandoned
        rather than run on empty data -- continuing would just push the same
        "looks like no findings" problem one level down.

        In 'strict' mode (the default, and always for paid runs) the error is
        re-raised with the partial result attached for diagnostics. In 'collect'
        mode the partial result is returned with the error recorded in
        debug["pass_errors"] and pass_states[key] == "failed".
        """
        import time
        start_time = time.perf_counter()

        options = options or SceneClassifierRunOptions()

        result = ImageAnalysisResult(image_path=str(image_path))
        result.photo_key = image_path.name

        try:
            await self._run_passes(image_path, options, result)
        except PassExecutionError as err:
            logger.error("Image %s failed at pass %s: %s", image_path.name, err.pass_key, err.message)
            result.passes[err.pass_key] = {"error": err.message}
            result.debug.setdefault("pass_errors", []).append(err.to_dict())
            self._finalize(result, image_path, start_time)
            if options.failure_mode == "collect":
                return result
            err.partial_result = result.to_dict()
            raise

        self._finalize(result, image_path, start_time)
        return result

    def _finalize(
        self,
        result: ImageAnalysisResult,
        image_path: Path,
        start_time: float,
    ) -> None:
        """Derive pass_states and timings. Runs on both the success and failure paths."""
        import time
        for pass_key in ('1a', '1b', '1c', '2a', '2b', '2c', '2d', '2e'):
            pass_payload = result.passes.get(pass_key) or {}
            if isinstance(pass_payload, dict) and pass_payload.get("error"):
                result.pass_states[pass_key] = "failed"
            elif pass_key not in result.passes_run:
                result.pass_states[pass_key] = "skipped"
            elif result.models_used.get(pass_key) == "none":
                result.pass_states[pass_key] = "stubbed"
            elif pass_key == "2e":
                result.pass_states[pass_key] = "rule_based"
            else:
                result.pass_states[pass_key] = "executed"

        result.total_pass_time = sum(result.pass_timings.values())
        result.processing_time = time.perf_counter() - start_time
        logger.info(
            f"Completed {image_path.name}: scene={result.scene}, "
            f"classified_obs={len(result.observations)}, "
            f"excluded_obs={len(result.excluded_observations)}, "
            f"time={result.processing_time:.1f}s (LLM={result.total_pass_time:.1f}s)"
        )

    async def _run_passes(
        self,
        image_path: Path,
        options: SceneClassifierRunOptions,
        result: ImageAnalysisResult,
    ) -> None:
        """Execute the enabled passes in order, mutating *result*."""
        import time
        toggles = options.toggles
        context: Dict[str, Any] = {}

        logger.info(f"Analyzing image: {image_path.name}")
        logger.debug(describe_run_plan(options))

        # ─────────────────────────────────────────────────────────────────────
        # Pass 1a: Scene Type Classification
        # ─────────────────────────────────────────────────────────────────────
        if self._t(toggles, '1a'):
            model_config = self._get_model_config('1a', options)
            model_name = self._get_model_name('1a', options)
            self._record_model_routing('1a', options, model_config, result)

            logger.debug(f"Running Pass 1a with {model_name}")
            t0 = time.perf_counter()
            result.pass_1a = await run_pass_1a_scene_type(
                image_path=image_path,
                vlm_client=self.vlm_client,
                model_config=model_config,
            )
            result.pass_timings['1a'] = time.perf_counter() - t0

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
            # Benchmark hooks (options.meta, absent in production requests):
            #   pass_2a_frozen_freeform — replay downstream passes from a stored
            #     2a capture without any vision call (attribution measurement).
            #   pass_2a_user_prompt — ablation wording override for the 2a call.
            _meta = getattr(options, "meta", None) or {}
            _frozen_2a = _meta.get("pass_2a_frozen_freeform")
            if _frozen_2a is not None:
                result.pass_2a = Pass2aResult(
                    observations_freeform=str(_frozen_2a).strip(),
                    raw_response=str(_frozen_2a),
                )
                result.pass_timings['2a'] = 0.0
                observations_freeform = result.pass_2a.observations_freeform
                result.observations_freeform = observations_freeform
                result.passes_run.append('2a')
                result.models_used['2a'] = "frozen_replay"
            else:
                model_config = self._get_model_config('2a', options)
                model_name = self._get_model_name('2a', options)
                self._record_model_routing('2a', options, model_config, result)

                logger.debug(f"Running Pass 2a with {model_name}")
                t0 = time.perf_counter()
                result.pass_2a = await run_pass_2a(
                    image_path=image_path,
                    vlm_client=self.vlm_client,
                    model_config=model_config,
                    context=context,
                    user_prompt=_meta.get("pass_2a_user_prompt"),
                )
                result.pass_timings['2a'] = time.perf_counter() - t0

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
            self._record_model_routing('2b', options, model_config, result)

            logger.debug(f"Running Pass 2b with {model_name}")
            t0 = time.perf_counter()
            result.pass_2b = await run_pass_2b(
                vlm_client=self.vlm_client,
                model_config=model_config,
                observations_freeform=observations_freeform,
            )
            result.pass_timings['2b'] = time.perf_counter() - t0

            observations_list = result.pass_2b.observations or []
            result.observations_struct = {"observations": observations_list}
            context["observations_struct"] = result.observations_struct

            result.passes_run.append('2b')
            result.models_used['2b'] = model_name

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2c: Classify observations — observation-kind-v2 (text-only)
        # ─────────────────────────────────────────────────────────────────────
        if self._t(toggles, '2c'):
            model_config = self._get_model_config('2c', options)
            model_name = self._get_model_name('2c', options)
            self._record_model_routing('2c', options, model_config, result)

            observations_in = []
            if isinstance(result.observations_struct, dict):
                observations_in = result.observations_struct.get("observations") or []

            logger.debug(f"Running Pass 2c with {model_name}")
            t0 = time.perf_counter()
            result.pass_2c = await run_pass_2c(
                vlm_client=self.vlm_client,
                model_config=model_config,
                observations=observations_in,
                scene=result.scene or "other",
            )
            result.pass_timings['2c'] = time.perf_counter() - t0

            result.observations = list(result.pass_2c.observations or [])
            result.excluded_observations = list(result.pass_2c.excluded or [])

            context["observations"] = result.observations
            context["excluded_observations"] = result.excluded_observations

            # ── Stamp deterministic issue_id on every classified observation ──
            # Uses _make_issue_id() for stable, deterministic IDs (kind takes
            # the label slot in the signature under observation-kind-v2).
            _run_id = (getattr(options, "meta", None) or {}).get("run_id", "")
            _photo_key = (getattr(options, "meta", None) or {}).get("photo_key") or image_path.name
            _sig_counts: Dict[tuple, int] = {}
            for _obs in (result.observations or []):
                if not isinstance(_obs, dict):
                    continue
                _desc = (_obs.get("description") or "").strip()
                if not _desc:
                    continue
                _kind = (_obs.get("kind") or "").strip()
                _loc = (_obs.get("location_hint") or "").strip()
                _sig = (_desc, _loc, _kind)
                _ordinal = _sig_counts.get(_sig, 0)
                _sig_counts[_sig] = _ordinal + 1
                # Forward-only: only assign if missing
                if not _obs.get("issue_id"):
                    _obs["issue_id"] = _make_issue_id(_run_id, _photo_key, _desc, _loc, _kind, _ordinal)
                _obs.setdefault("source_photo_key", _photo_key)

            result.passes_run.append('2c')
            result.models_used['2c'] = model_name

        # ─────────────────────────────────────────────────────────────────────
        # observation-kind-v2: pipeline mode dispatch
        # ─────────────────────────────────────────────────────────────────────
        # classification_only ends after Pass 2c and stays non-publishable.
        # catalog_resolution_benchmark additionally runs Pass 2d + 2e, still
        # non-publishable. publish runs the full 2c -> 2d -> 2e chain and is
        # the only mode whose results write_photo_intel accepts.
        mode = getattr(options, "pipeline_mode", PIPELINE_MODE_CLASSIFICATION_ONLY)
        if mode not in ALLOWED_PIPELINE_MODES:
            raise ValueError(
                f"unsupported pipeline_mode: {mode!r}; "
                f"expected one of {sorted(ALLOWED_PIPELINE_MODES)}"
            )

        result.classification_only = (mode != PIPELINE_MODE_PUBLISH)
        result.debug["pipeline_mode"] = mode
        if result.classification_only:
            result.debug["classification_only"] = {
                "reason": "classification_only_v2",
                "detail": (
                    f"non-publishable pipeline_mode ({mode}); "
                    "write_photo_intel rejects this payload"
                ),
            }
        result.debug["ontology"] = {
            "ontology_version": ONTOLOGY_VERSION,
            "pass_2b_prompt_version": PASS_2B_PROMPT_VERSION,
            "pass_2b_prompt_sha256": PASS_2B_PROMPT_SHA256,
            "pass_2c_prompt_version": PASS_2C_PROMPT_VERSION,
            "pass_2c_prompt_sha256": PASS_2C_PROMPT_SHA256,
            "pass_2d_prompt_version": PASS_2D_PROMPT_VERSION,
            "pass_2d_prompt_sha256": PASS_2D_PROMPT_SHA256,
        }
        if mode == PIPELINE_MODE_CLASSIFICATION_ONLY:
            return

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2d: strict exact-kind catalog resolution (benchmark + publish)
        # ─────────────────────────────────────────────────────────────────────
        # Consumes Pass 2c observations directly: kind is assigned by the v2
        # contract, so there is no label mapping, no kind coercion, and no
        # shadow lane. Retrieval searches exactly the observation's kind.
        _scene_for_2d = result.scene or "unknown"
        _scene_group_for_2d = SCENE_TO_GROUP_UI.get(_scene_for_2d, "other")

        observations = [
            obs for obs in (result.observations or [])
            if isinstance(obs, dict) and (obs.get("description") or "").strip()
        ]
        to_resolve_all = observations[:self.max_resolve_per_image]

        pass_2d_toggle = self._t(toggles, "2d", default=True)
        pass_2d_provider_present = self.candidate_provider is not None
        by_kind = {
            kind: sum(1 for obs in observations if obs.get("kind") == kind)
            for kind in sorted(OBSERVATION_KINDS)
        }
        result.debug["pass_2d_gate"] = {
            "toggle": pass_2d_toggle,
            "candidate_provider_present": pass_2d_provider_present,
            "observation_count": len(observations),
            "by_kind": by_kind,
            "total_resolve_count": len(to_resolve_all),
        }
        logger.info(
            "Pass 2d gate: toggle=%s provider=%s observations=%d by_kind=%s total_resolve=%d",
            pass_2d_toggle, pass_2d_provider_present, len(observations),
            by_kind, len(to_resolve_all),
        )

        run_2d = bool(pass_2d_toggle and pass_2d_provider_present and to_resolve_all)
        if not run_2d:
            if to_resolve_all and pass_2d_toggle and not pass_2d_provider_present:
                # Entry points fail closed at preflight when 2d is enabled, so this
                # should be unreachable; it is the assertion that keeps it that way.
                raise PassExecutionError(
                    '2d', 'dependency',
                    f"Pass 2d is enabled with {len(to_resolve_all)} resolvable observations "
                    "but no candidate provider was supplied",
                    code="MissingCandidateProvider",
                )
            if not to_resolve_all:
                logger.debug("Pass 2d: no resolvable observations from Pass 2c.")
            result.debug["pass_2d_summary"] = {
                "attempted_total": 0,
                "resolved_total": 0,
                "resolved_by_kind": {kind: 0 for kind in sorted(OBSERVATION_KINDS)},
            }
        else:
            model_config = self._get_model_config('2d', options)
            model_name = self._get_model_name('2d', options)
            self._record_model_routing('2d', options, model_config, result)

            logger.debug("Running Pass 2d with %s for %d observations", model_name, len(to_resolve_all))
            t0 = time.perf_counter()

            base_ctx_for_provider = {
                **context,
                "top_k_candidates": self.top_k_candidates,
                "scene": result.scene,
                "scene_group": _scene_group_for_2d,
            }

            pass_2d_results: List[Pass2dResult] = []
            resolved_items: List[Dict[str, Any]] = []
            result.debug["pass_2d_per_observation"] = []

            for obs in to_resolve_all:
                obs.setdefault("scene", _scene_for_2d)
                obs.setdefault("scene_group", _scene_group_for_2d)
                resolved_row, debug_row, pass_2d_result = await resolve_observation_against_catalog(
                    vlm_client=self.vlm_client,
                    model_config=model_config,
                    candidate_provider=self.candidate_provider,
                    observation=obs,
                    base_context=base_ctx_for_provider,
                    top_k=self.top_k_candidates,
                    source_image_path=str(image_path),
                )
                result.debug["pass_2d_per_observation"].append(debug_row)
                if pass_2d_result is not None:
                    pass_2d_results.append(pass_2d_result)
                if resolved_row is not None:
                    resolved_items.append(resolved_row)

            result.pass_timings['2d'] = time.perf_counter() - t0
            result.pass_2d = pass_2d_results
            result.resolved_items = resolved_items
            result.passes_run.append('2d')
            result.models_used['2d'] = model_name

            resolved_by_kind = {
                kind: sum(1 for row in resolved_items if row.get("resolved_kind") == kind)
                for kind in sorted(OBSERVATION_KINDS)
            }
            result.debug["pass_2d_summary"] = {
                "attempted_total": len(to_resolve_all),
                "resolved_total": len(resolved_items),
                "resolved_by_kind": resolved_by_kind,
            }
            logger.info(
                "Pass 2d summary: attempted=%d resolved=%d by_kind=%s",
                len(to_resolve_all), len(resolved_items), resolved_by_kind,
            )

        # ─────────────────────────────────────────────────────────────────────
        # Pass 2e: Normalize / Filter / Deduplicate Verified Issues (rule-based)
        # Output: result.verified_issues — clean, deduplicated, scoring-free
        #
        # Runs whether or not 2d resolved anything (v1 semantics): unresolved
        # observations carry no catalogItemId and pass through the catalog-keyed
        # policy gates on defaults.
        # ─────────────────────────────────────────────────────────────────────
        if self._t(toggles, '2e'):
            model_config = self._get_model_config('2e', options)
            model_name = self._get_model_name('2e', options)

            # Build input: observations enriched with 2d resolution data where
            # available, joined by issue_id.
            resolution_index: Dict[str, Dict[str, Any]] = {
                row["issue_id"]: row
                for row in (result.resolved_items or [])
                if row.get("issue_id")
            }

            issues_for_2e: List[Dict[str, Any]] = []
            for obs in (result.observations or []):
                if not isinstance(obs, dict):
                    continue
                issue = dict(obs)  # shallow copy — don't mutate observations
                # Merge resolution data if available (catalogItemId etc.).
                # Kind purity: an off-kind resolution already raised in 2d, so
                # the kind stamp is a no-op in practice but is the artifact
                # contract for resolved issues.
                res = resolution_index.get(str(issue.get("issue_id") or ""))
                if res and res.get("resolved_item_id"):
                    issue.setdefault("catalogItemId", res["resolved_item_id"])
                    if res.get("resolved_kind") in OBSERVATION_KINDS:
                        issue["kind"] = res["resolved_kind"]
                        issue["catalogItemKind"] = res["resolved_kind"]
                issues_for_2e.append(issue)

            # Inject catalog metadata and policy into context for Pass 2e
            if self.catalog_meta_by_id:
                context["catalog_meta_by_id"] = self.catalog_meta_by_id
            context["policy"] = {"include_optional": False, "mode": "renovator_strict"}

            t0 = time.perf_counter()
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
            except PassExecutionError:
                result.pass_timings['2e'] = time.perf_counter() - t0
                raise
            except Exception as exc:
                # No passthrough fallback. 2e is what turns observations into
                # *verified* issues; promoting observations on failure published
                # unverified findings as verified — precisely the class of error
                # the fail-closed contract exists to prevent. Record the failure
                # and fail the run instead.
                logger.error(f"Pass 2e failed: {exc}", exc_info=True)
                result.passes["2e"] = {"error": str(exc)}
                result.debug["pass_2e_summary"] = {"error": str(exc)}
                result.pass_timings['2e'] = time.perf_counter() - t0
                # 2e is rule-based — no provider call — so a failure is a missing
                # catalog input or a malformed issue payload, never a provider fault.
                stage = "dependency" if isinstance(exc, (KeyError, LookupError)) else "parse"
                raise PassExecutionError(
                    "2e", stage, str(exc)[:300], code=type(exc).__name__
                ) from exc
            result.pass_timings['2e'] = time.perf_counter() - t0
        else:
            # 2e disabled by toggle (deliberate diagnostics config): promote
            # observations verbatim. No kind derivation — the 2c fail-closed
            # contract already guarantees a valid kind on every observation.
            _out: List[Dict[str, Any]] = [
                dict(_obs) for _obs in (result.observations or [])
                if isinstance(_obs, dict)
            ]
            result.verified_issues = _out
            result.display_issues = _out
            result.matched_issues = list(_out)
            result.canonical_issues = result.matched_issues
            result.passes["2e"] = {"skipped": True}
            result.debug["pass_2e_summary"] = {"skipped": True}


# ═══════════════════════════════════════════════════════════════════════════════
# Factory function for easy instantiation
# ═══════════════════════════════════════════════════════════════════════════════

def create_orchestrator_from_config(
    config: Any,
    candidate_provider: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
    catalog_items: Optional[List[Dict[str, Any]]] = None,
    vlm_client: Any = None,
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
    vlm_client = vlm_client or create_vlm_client()

    return SceneClassifierOrchestrator(
        qwen_config=qwen_config,
        gpt5_config=gpt5_config,
        vlm_client=vlm_client,
        candidate_provider=candidate_provider,
        top_k_candidates=getattr(config, "TOP_K_CANDIDATES", 8),
        max_resolve_per_image=getattr(config, "MAX_RESOLVE_PER_IMAGE", 25),
        catalog_items=catalog_items,
    )
