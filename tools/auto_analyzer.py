#!/usr/bin/env python3
"""
Auto-Analyzer for GroundingDINO Pipeline
-----------------------------------------
Orchestrates: Scene Classification → Detection → Verification → Property Summary
Uses settings from pipeline_config.py

Supports multiple detection backends:
- groundingdino (default): Local GroundingDINO inference
- dinox: DINO-X API or local script

Supports analysis profiles:
- standard: All passes use local Qwen via LM Studio
- premium: Key passes (1b, 2a, 4) use OpenAI GPT-5/GPT-4o
"""

import os
import sys
import json
import time
import uuid
import shutil
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from PIL import Image

from run_pipeline import redraw_overlay
from scene_classifier import load_issue_catalog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import configuration
try:
    import pipeline_config as cfg
except ImportError:
    print("ERROR: pipeline_config.py not found!")
    sys.exit(1)

# Import renovation cost table from dedicated module
try:
    from renovation_costs import RENOVATION_COST_TABLE
except ImportError:
    print("WARNING: renovation_costs.py not found, using empty cost table")
    RENOVATION_COST_TABLE = {}

# Import NMS postprocessing
try:
    from postprocess import (
        class_aware_nms,
        enforce_scene_caps,
        apply_roi_hint_bonus_overlap,
        apply_special_case_filters,
    )
except ImportError:
    print("ERROR: postprocess.py not found!")
    sys.exit(1)

# Import property summarizer (optional - graceful fallback if not available)
try:
    from property_summarizer import PropertySummarizer
except ImportError:
    PropertySummarizer = None
    print("INFO: property_summarizer.py not found, property summaries will be skipped")

# Import DINO-X client (optional - graceful fallback to legacy mode)
try:
    from dinox_client import DINOXClient, create_dinox_client_from_config

    DINOX_CLIENT_AVAILABLE = True
except ImportError:
    DINOXClient = None
    create_dinox_client_from_config = None
    DINOX_CLIENT_AVAILABLE = False
    print("INFO: dinox_client.py not found, DINO-X will use legacy script mode only")

# Embeddings catalog matcher (optional)
try:
    from catalog_embeddings import CatalogEmbedMatcher
    EMBEDDINGS_MATCHER_AVAILABLE = True
except Exception:
    CatalogEmbedMatcher = None
    EMBEDDINGS_MATCHER_AVAILABLE = False

# Import VLM client (for direct calls when orchestrator unavailable)
try:
    from vlm_client import VLMClient, create_vlm_client, get_model_configs_from_pipeline_config

    VLM_CLIENT_AVAILABLE = True
except ImportError:
    VLMClient = None
    create_vlm_client = None
    get_model_configs_from_pipeline_config = None
    VLM_CLIENT_AVAILABLE = False

# Import pass architecture components (optional - graceful fallback)
try:
    from pass_config import (
        SceneClassifierRunOptions,
        PassToggles,
        PassModelOverrides,
        pick_model_for_pass,
        get_model_config_for_pass, PassKey,
    )
    from scene_classifier_orchestrator import (
        SceneClassifierOrchestrator,
        create_orchestrator_from_config,
    )
    from scene_classifier_passes import (
        run_pass_1a_scene_type,
        run_pass_1b_positive_notes,
        run_pass_1c_positive_structuring,
        run_pass_2a_issue_detection,
        run_pass_2b_issue_verification,
        run_pass_3_keyword_extraction,
        run_pass_4_property_summary,
        run_pass_4a_room_summaries,
        run_pass_4b_property_card_fields,
        Pass4aRoomSummariesResult,
        Pass4bLegacyCardResult,
        SCENE_TO_GROUP,
    )

    PASS_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    SceneClassifierRunOptions = None
    PASS_ARCHITECTURE_AVAILABLE = False
    print(f"INFO: Pass architecture modules not found ({e}), using legacy scene classifier")

# Console encoding safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Logging
logging.basicConfig(
    level=logging.DEBUG if cfg.DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

default_catalog_path = getattr(
    cfg, "PROJECT_ROOT", Path(__file__).resolve().parent.parent
) / "issue_catalog.json"
ISSUE_CATALOG_PATH = Path(getattr(cfg, "ISSUE_CATALOG_PATH", default_catalog_path))
ISSUE_CATALOG = load_issue_catalog(ISSUE_CATALOG_PATH)

SEVERITY_RANK = {
    "none": 0,
    "minor_repair": 1,
    "moderate_repair": 2,
    "full_replacement": 3,
}

# Valid detection backends (canonical names)
VALID_BACKENDS = {"groundingdino", "dinox"}

# Aliases for user convenience
BACKEND_ALIASES = {
    "grounding-dino": "groundingdino",
    "dino-x": "dinox",
}

# Room grouping map for UI aggregation
SCENE_GROUPS_UI = {
    "kitchen": ["kitchen", "pantry"],
    "bathroom": ["bathroom"],
    "bedroom": ["bedroom", "closet"],
    "living_areas": ["living_room", "dining_room", "home_office", "hallway", "stairway"],
    "utility": ["laundry_room", "basement", "attic", "garage", "hvac"],
    "exterior": ["exterior_front", "exterior_back", "exterior_side", "yard", "patio", "deck", "balcony", "driveway",
                 "pool", "garden"],
    "other": ["roof", "other", "unknown", "floor_plan", "aerial_view", "street_view"],
}

# Reverse lookup: scene -> group
SCENE_TO_GROUP_UI = {}
for _group, _scenes in SCENE_GROUPS_UI.items():
    for _scene in _scenes:
        SCENE_TO_GROUP_UI[_scene] = _group


# ── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class ImageAnalysisResult:
    """Result from analyzing a single image."""
    image_path: str
    scene: str
    scene_data: Optional[Dict[str, Any]]
    keywords_used: List[str]
    detection_count: int
    verified_count: Optional[int] = None
    output_dir: str = ""
    processing_time: float = 0.0
    error: Optional[str] = None
    scene_classifier: Optional[Dict[str, Any]] = None


@dataclass
class PropertyAnalysisJob:
    """Complete property analysis job."""
    job_id: str
    property_key: str
    timestamp: str
    results: List[ImageAnalysisResult]
    artifacts_dir: str
    total_processing_time: float
    parameters: Dict[str, Any]


def _worst_severity(current: str, new: str) -> str:
    if SEVERITY_RANK.get(new, 0) > SEVERITY_RANK.get(current, 0):
        return new
    return current


def _parse_catalog_flag(flag: Any) -> Tuple[str, str, str]:
    """Normalize catalog flag inputs from dicts or dataclass-like objects."""

    if isinstance(flag, dict):
        present = str(flag.get("present", "")).lower() or "uncertain"
        evidence = str(flag.get("evidence", ""))
        severity = str(flag.get("severity", "none")).lower() or "none"
    else:
        present = str(getattr(flag, "present", "uncertain") or "uncertain").lower()
        evidence = str(getattr(flag, "evidence", ""))
        severity = str(getattr(flag, "severity", "none") or "none").lower()

    if present != "yes":
        severity = "none"

    return present, severity, evidence


def build_renovation_needs(job: "PropertyAnalysisJob", issue_catalog: Optional[Dict[str, Any]] = None) -> Dict[
    str, Any]:
    """
    Aggregate per-photo catalog_flags + severity into a property-level
    renovation_needs structure.

    Args:
        job: PropertyAnalysisJob with results
        issue_catalog: Issue catalog to use (defaults to module global ISSUE_CATALOG)
    """
    catalog = issue_catalog or ISSUE_CATALOG or {}

    issue_meta: Dict[str, Dict[str, str]] = {}
    for item in catalog.get("defect_issues", []) or []:
        if isinstance(item, dict) and item.get("id"):
            issue_meta[item["id"]] = {
                "name": item.get("name", item["id"]),
                "category": item.get("category", "unknown"),
            }
    for item in catalog.get("opportunity_flags", []) or []:
        if isinstance(item, dict) and item.get("id"):
            issue_meta[item["id"]] = {
                "name": item.get("name", item["id"]),
                "category": item.get("category", "unknown"),
            }

    aggregates: Dict[str, Dict[str, Any]] = {}
    totals_by_category: Dict[str, Dict[str, float]] = {}

    for result in job.results:
        scene_payload = (result.scene_classifier or result.scene_data) or {}
        if not isinstance(scene_payload, dict):
            continue

        flags = scene_payload.get("catalog_flags", {}) or {}
        photo_name = Path(result.image_path).name

        for issue_id, flag in flags.items():
            present, severity, evidence = _parse_catalog_flag(flag)
            if present != "yes":
                continue

            meta = issue_meta.get(issue_id, {"name": issue_id, "category": "unknown"})
            agg = aggregates.setdefault(issue_id, {
                "issue_id": issue_id,
                "name": meta["name"],
                "category": meta["category"],
                "worst_severity": "none",
                "occurrences": 0,
                "present_in_photos": [],
                "sample_evidence": "",
                "est_cost_low": 0.0,
                "est_cost_high": 0.0,
            })

            agg["occurrences"] += 1
            agg["present_in_photos"].append(photo_name)
            agg["worst_severity"] = _worst_severity(agg["worst_severity"], severity)

            if not agg["sample_evidence"] and evidence:
                agg["sample_evidence"] = evidence

            cost_cfg = RENOVATION_COST_TABLE.get(issue_id, {})
            sev_cost = cost_cfg.get(severity)
            if sev_cost:
                low, high = sev_cost
                agg["est_cost_low"] += low
                agg["est_cost_high"] += high

    for issue in aggregates.values():
        cat = issue.get("category", "unknown")
        cat_totals = totals_by_category.setdefault(cat, {
            "est_cost_low": 0.0,
            "est_cost_high": 0.0,
        })
        cat_totals["est_cost_low"] += issue["est_cost_low"]
        cat_totals["est_cost_high"] += issue["est_cost_high"]

    grand_low = sum(cat.get("est_cost_low", 0.0) for cat in totals_by_category.values())
    grand_high = sum(cat.get("est_cost_high", 0.0) for cat in totals_by_category.values())

    issues_sorted = sorted(
        aggregates.values(),
        key=lambda x: x.get("est_cost_high", 0.0),
        reverse=True,
    )

    return {
        "issues": issues_sorted,
        "totals_by_category": totals_by_category,
        "grand_total": {
            "est_cost_low": grand_low,
            "est_cost_high": grand_high,
        },
    }


# ── Auto-Analyzer Class ──────────────────────────────────────────────────────
class AutoAnalyzer:
    def __init__(self,
                 python_exe: str = sys.executable,
                 artifacts_root: str = None,
                 box_threshold: float = None,
                 text_threshold: float = None,
                 chip_margin: float = None,
                 max_keywords: int = None,
                 include_conditions: bool = None,
                 skip_verification: bool = None,
                 debug: bool = None,
                 detection_backend: Optional[str] = None,
                 analysis_profile: Optional[str] = None,
                 # NEW: Pass architecture parameters
                 pass_toggles: Optional[Dict[str, bool]] = None,
                 model_overrides: Optional[Dict[str, str]] = None,
                 use_pass_architecture: Optional[bool] = None):

        # Use config defaults if not specified
        self.python_exe = Path(python_exe)
        self.artifacts_root = Path(artifacts_root or cfg.ARTIFACTS_ROOT)
        self.box_threshold = box_threshold if box_threshold is not None else cfg.BOX_THRESHOLD
        self.text_threshold = text_threshold if text_threshold is not None else cfg.TEXT_THRESHOLD
        self.chip_margin = chip_margin if chip_margin is not None else cfg.CHIP_MARGIN
        self.max_keywords = max_keywords if max_keywords is not None else cfg.MAX_KEYWORDS
        self.include_conditions = include_conditions if include_conditions is not None else cfg.INCLUDE_CONDITIONS
        self.skip_verification = skip_verification if skip_verification is not None else cfg.SKIP_VERIFICATION
        self.debug = debug if debug is not None else cfg.DEBUG_MODE

        # ✅ Instance-owned issue catalog (avoid relying on module globals)
        self.issue_catalog_path = ISSUE_CATALOG_PATH
        self.issue_catalog = ISSUE_CATALOG or {}

        # Helpful sanity logging
        try:
            defect_count = len((self.issue_catalog.get("defect_issues", []) or []))
            opp_count = len((self.issue_catalog.get("opportunity_flags", []) or []))
            logger.info(f"ISSUE_CATALOG_PATH = {self.issue_catalog_path}")
            logger.info(f"Issue catalog counts: defect={defect_count} opportunity={opp_count}")
        except Exception as e:
            logger.warning(f"Failed to log issue catalog counts: {e}")

        # ── Embeddings-based catalog matcher (optional) ─────────────────────
        self.catalog_matcher = None
        if getattr(cfg, "USE_EMBEDDINGS_CATALOG", False) and EMBEDDINGS_MATCHER_AVAILABLE:
            try:
                self.catalog_matcher = CatalogEmbedMatcher(
                    issue_catalog=self.issue_catalog,
                    model_name=getattr(cfg, "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                    topk=getattr(cfg, "EMBEDDINGS_TOPK", 5),
                    threshold_defect=getattr(cfg, "EMBEDDINGS_THRESHOLD_DEFECT", 0.58),
                    threshold_opportunity=getattr(cfg, "EMBEDDINGS_THRESHOLD_OPPORTUNITY", 0.56),
                    route_by_rough_category=getattr(cfg, "EMBEDDINGS_ROUTE_BY_ROUGH_CATEGORY", True),
                    trust_remote_code=bool(getattr(cfg, "EMBEDDINGS_TRUST_REMOTE_CODE", False)),
                    device=str(getattr(cfg, "EMBEDDINGS_DEVICE", "cpu")),
                )
                logger.info("Embeddings catalog matcher initialized")
                logger.info(f"Embeddings model device: {getattr(self.catalog_matcher, 'device', 'unknown')}")
            except Exception as e:
                logger.warning(f"Failed to init embeddings catalog matcher: {e}")
                self.catalog_matcher = None

        # ✅ Backend selection (CLI arg > cfg/env > default)
        raw_backend = (
                detection_backend
                or getattr(cfg, "DETECTION_BACKEND", None)
                or "groundingdino"
        ).strip().lower()

        # Normalize aliases to canonical names
        raw_backend = BACKEND_ALIASES.get(raw_backend, raw_backend)

        if raw_backend not in VALID_BACKENDS:
            raise ValueError(f"Unknown detection backend: {raw_backend}")

        self.detection_backend = raw_backend

        # ✅ Analysis profile selection (CLI arg > cfg/env > default)
        self.analysis_profile = (
                analysis_profile
                or getattr(cfg, "ANALYSIS_PROFILE", None)
                or "standard"
        ).strip().lower()

        # ✅ Pass architecture configuration
        self.pass_toggles = pass_toggles or {}
        self.model_overrides = model_overrides or {}

        # ✅ Initialize VLM client and model configs FIRST (needed for all paths)
        self.vlm_client = None
        self.qwen_config = None
        self.gpt_config = None
        if VLM_CLIENT_AVAILABLE:
            try:
                self.vlm_client = create_vlm_client()
                self.qwen_config, self.gpt_config = get_model_configs_from_pipeline_config(cfg)
                gpt_model = self.gpt_config.get('model', 'N/A') if self.gpt_config else 'N/A'
                logger.info(f"VLM client initialized (GPT model: {gpt_model})")
            except Exception as e:
                logger.warning(f"Failed to initialize VLM client: {e}")

        # Determine whether to use pass architecture:
        # - If explicitly set, use that value
        # - Otherwise, auto-enable if modules available (both standard and premium)
        if use_pass_architecture is not None:
            self.use_pass_architecture = bool(use_pass_architecture) and PASS_ARCHITECTURE_AVAILABLE
        else:
            self.use_pass_architecture = PASS_ARCHITECTURE_AVAILABLE
            if self.use_pass_architecture:
                logger.info("Auto-enabling pass architecture (modules available)")

        # Initialize pass architecture components if enabled
        self.orchestrator = None
        self.run_options = None
        if self.use_pass_architecture:
            try:
                self.run_options = SceneClassifierRunOptions.from_analysis_profile(
                    self.analysis_profile,
                    toggles=self.pass_toggles,
                    model_overrides=self.model_overrides,
                )
                # ✅ Give orchestrator a default catalog if factory supports it
                try:
                    self.orchestrator = create_orchestrator_from_config(cfg, issue_catalog=self.issue_catalog)
                except TypeError:
                    self.orchestrator = create_orchestrator_from_config(cfg)
                logger.info(f"Pass architecture initialized (premium={self.run_options.premium})")
            except Exception as e:
                logger.warning(f"Failed to initialize pass architecture: {e}, falling back to direct pass calls")
                self.use_pass_architecture = False

        # ✅ Initialize DINO-X client if using dinox backend
        self.dinox_client = None
        if self.detection_backend == "dinox" and DINOX_CLIENT_AVAILABLE:
            try:
                self.dinox_client = create_dinox_client_from_config(cfg)
                logger.info("DINO-X client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize DINO-X client: {e}")

        # Apply profile-specific overrides
        self._apply_profile_settings()

        # Validate environment
        self._validate_environment()

    def _apply_profile_settings(self):
        """Apply settings based on analysis profile."""
        if self.analysis_profile == "premium":
            # Premium profile: use enhanced settings
            premium_keywords = getattr(cfg, "PREMIUM_MAX_KEYWORDS", None)
            if premium_keywords is not None:
                self.max_keywords = premium_keywords

            # Use premium summary model if configured
            self.summary_model = getattr(cfg, "PREMIUM_SUMMARY_MODEL", None) or getattr(cfg, "GPT_MODEL", None)

            # Enable stricter verification if configured
            premium_skip_verify = getattr(cfg, "PREMIUM_SKIP_VERIFICATION", None)
            if premium_skip_verify is not None:
                self.skip_verification = premium_skip_verify

            logger.info(f"Using PREMIUM analysis profile (max_keywords={self.max_keywords})")
        else:
            # Standard profile: use defaults
            self.summary_model = getattr(cfg, "SUMMARY_MODEL", None)
            logger.info(f"Using STANDARD analysis profile (max_keywords={self.max_keywords})")

    def _get_model_config_for_pass(self, pass_key: PassKey) -> Dict[str, Any]:
        """
        Get the appropriate model config for a specific pass.

        Args:
            pass_key: Pass identifier ('1a', '1b', '1c', '2a', '2b', '3', '4', '4a', '4b')

        Returns:
            Model config dict with 'url', 'model', 'api_key' etc.
        """
        # If pass_config module available, use its routing logic
        if PASS_ARCHITECTURE_AVAILABLE and self.run_options:
            cfg_for_pass = get_model_config_for_pass(
                pass_key=pass_key,
                options=self.run_options,
                qwen_config=self.qwen_config or {},
                gpt5_config=self.gpt_config or {},
            )
            return self._attach_openai_token_cap(pass_key, cfg_for_pass)

        # Fallback: manual routing based on profile
        if not self.qwen_config:
            # No VLM client available, return LM Studio defaults
            base = {
                'url': getattr(cfg, 'LM_STUDIO_URL', 'http://localhost:1234'),
                'model': getattr(cfg, 'LM_STUDIO_MODEL', 'qwen-vl'),
                'provider': 'lmstudio',
            }
            return base  # no OpenAI cap possible here

        # Premium uses GPT for passes 1b, 2a, 4
        if self.analysis_profile == 'premium' and pass_key in ('1b', '2a', '4'):
            if self.gpt_config and self.gpt_config.get('api_key'):
                # Check for pass-specific model override
                base = self._get_pass_specific_gpt_config(pass_key)
                return self._attach_openai_token_cap(pass_key, base)
            else:
                logger.warning(f"Premium pass {pass_key} requested but no GPT API key, using Qwen")

        return self.qwen_config

    def _get_pass_specific_gpt_config(self, pass_key: str) -> Dict[str, Any]:
        """
        Get GPT config for a specific pass, checking for per-pass model overrides.
        """
        base_config = dict(self.gpt_config) if self.gpt_config else {
            'model': 'gpt-4o',
            'api_key': os.environ.get('OPENAI_API_KEY', ''),
            'provider': 'openai',
        }

        # Check for pass-specific model override in config
        pass_model_attrs = {
            '1b': ['GPT_PASS_1B_MODEL', 'OPENAI_PASS1B_MODEL'],
            '2a': ['GPT_PASS_2A_MODEL', 'OPENAI_PASS2A_MODEL'],
            '2b': ['GPT_PASS_2B_MODEL', 'OPENAI_PASS2B_MODEL', 'OPENAI_CHIP_MODEL'],
            '4': ['GPT_PASS_4_MODEL', 'OPENAI_PASS4_MODEL'],
        }

        attr_names = pass_model_attrs.get(pass_key, [])
        for attr_name in attr_names:
            pass_model = getattr(cfg, attr_name, None) or os.environ.get(attr_name)
            if pass_model:
                base_config['model'] = pass_model
                logger.debug(f"Using pass-specific model for {pass_key}: {pass_model}")
                break

        return base_config

    def _attach_openai_token_cap(self, pass_key: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Only apply max_tokens for OpenAI calls.
        Leave LM Studio/Qwen untouched.
        """
        if not isinstance(model_config, dict):
            return model_config

        # Harden provider detection: infer OpenAI if api_key present but provider missing
        provider = model_config.get("provider")
        if provider is None and model_config.get("api_key"):
            provider = "openai"

        if provider != "openai":
            return model_config

        # Per-pass override keys
        key_map = {
            "1b": "OPENAI_PASS_1B_MAX_TOKENS",
            "2a": "OPENAI_PASS_2A_MAX_TOKENS",
            "4": "OPENAI_PASS_4_MAX_TOKENS",
        }

        attr = key_map.get(pass_key)
        cap = None

        if attr:
            cap = getattr(cfg, attr, None) or os.environ.get(attr)

        # Fallback if some other pass is routed to OpenAI
        if cap is None:
            cap = getattr(cfg, "OPENAI_DEFAULT_MAX_TOKENS", None) or os.environ.get("OPENAI_DEFAULT_MAX_TOKENS")

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
                logger.warning(f"Invalid OpenAI max tokens value for pass {pass_key}: {cap}")

        return model_config

    @staticmethod
    def _normalize_label_for_hint(label: Optional[str]) -> str:
        if not label:
            return ""
        s = str(label).strip().lower()
        s = s.replace("-", " ").replace("_", " ")
        return " ".join(s.split())

    def _get_roi_hint_map_for_scene(self, scene: str) -> Dict[str, str]:
        """Get ROI hint mapping for a given scene."""
        m = getattr(cfg, "ROI_HINTS_BY_SCENE", None)
        if not isinstance(m, dict):
            return {}

        # Prefer exact scene key, then group key, then default
        keys = [scene]
        group = SCENE_TO_GROUP_UI.get(scene)
        if group:
            keys.append(group)
        keys.append("default")

        scene_map = {}
        for k in keys:
            v = m.get(k)
            if isinstance(v, dict) and v:
                scene_map = v
                break

        # Normalize mapping keys to match _normalize_label_for_hint usage
        out = {}
        for lbl, zone in (scene_map or {}).items():
            nl = self._normalize_label_for_hint(lbl)
            if nl and zone:
                out[nl] = str(zone).strip().lower()
        return out

    def _maybe_backfill_planner_hints(
            self,
            scene: str,
            planner_hints: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        """Backfill planner hints from config if not already present."""
        if not getattr(cfg, "ROI_HINTS_ENABLED", False):
            return planner_hints or {}
        existing = dict(planner_hints or {})
        if existing:
            return existing
        return self._get_roi_hint_map_for_scene(scene)

    def _validate_environment(self):
        """Validate that required components are available."""
        required_paths = {
            "Python executable": self.python_exe,
        }

        # Only require GDINO assets if we're using GDINO
        if self.detection_backend == "groundingdino":
            required_paths.update({
                "GDINO config": cfg.GDINO_CONFIG,
                "GDINO checkpoint": cfg.GDINO_CHECKPOINT,
                "GDINO infer script": cfg.GDINO_INFER_SCRIPT,
            })

        # If using DINO-X, validate its requirements
        elif self.detection_backend == "dinox":
            dinox_script = getattr(cfg, "DINOX_INFER_SCRIPT", None)
            api_token = (
                    getattr(cfg, "DINOX_API_TOKEN", None) or
                    getattr(cfg, "DINOX_API_KEY", None) or
                    os.environ.get("DINOX_API_TOKEN") or
                    os.environ.get("DINOX_API_KEY")
            )

            if dinox_script:
                required_paths["DINO-X infer script"] = Path(dinox_script)
            elif not api_token:
                raise RuntimeError(
                    "DINO-X backend selected but neither DINOX_INFER_SCRIPT nor "
                    "DINOX_API_TOKEN/DINOX_API_KEY is configured."
                )

        # ✅ Validate premium profile requirements
        if self.analysis_profile == "premium":
            api_key = (
                    getattr(cfg, "OPENAI_API_KEY", None) or
                    os.environ.get("OPENAI_API_KEY")
            )
            if not api_key:
                logger.warning(
                    "Premium profile selected but OPENAI_API_KEY is not set. "
                    "Premium passes will fall back to Qwen."
                )
            if not VLM_CLIENT_AVAILABLE:
                logger.warning(
                    "Premium profile selected but vlm_client.py is not available. "
                    "Cannot route to GPT models."
                )
            if not PASS_ARCHITECTURE_AVAILABLE:
                logger.warning(
                    "Premium profile selected but pass architecture modules not available. "
                    "Using direct pass calls as fallback."
                )

        if not self.skip_verification:
            chip_verifier = cfg.TOOLS_DIR / "chip_verifier.py"
            if chip_verifier.exists():
                required_paths["Chip verifier"] = chip_verifier

        missing = []
        for name, path in required_paths.items():
            if isinstance(path, Path) and not path.exists():
                missing.append(f"{name}: {path}")

        if missing:
            error_msg = "Missing required components:\n" + "\n".join(missing)
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Create artifacts root
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a command and capture output."""
        import subprocess

        if self.debug:
            logger.debug(f"Running: {' '.join(str(c) for c in cmd)}")

        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = proc.communicate()
        return proc.returncode or 0, stdout, stderr

    def classify_scene(
            self, image_path: Path
    ) -> Tuple[str, str, List[str], str, List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
        """
        Classify the scene type and get keywords for detection.

        ✅ FIXED: Now properly routes through orchestrator or direct pass calls.
        No more subprocess to non-existent scene_classifier.py.

        Returns:
            Tuple of (scene, reasoning, keywords, grounding_prompt, planner_targets,
            planner_hints, scene_payload)
        """
        logger.info(f"Classifying scene: {image_path.name}")

        # Priority 1: Use orchestrator if pass architecture is enabled and working
        if self.use_pass_architecture and self.orchestrator:
            try:
                logger.info(f"  Using orchestrator (premium={self.run_options.premium})")
                out = self._classify_scene_with_orchestrator(image_path)
                scene, reasoning, keywords, prompt, planner_targets, planner_hints, payload = out
                planner_hints = self._maybe_backfill_planner_hints(scene, planner_hints)
                if isinstance(payload, dict):
                    payload["planner_hints"] = planner_hints
                    payload = self._maybe_apply_embeddings_catalog(payload)
                return (scene, reasoning, keywords, prompt, planner_targets, planner_hints, payload)
            except Exception as e:
                logger.warning(f"  Orchestrator failed: {e}, trying direct pass calls")

        # Priority 2: Direct pass function calls (works with or without premium)
        if VLM_CLIENT_AVAILABLE and self.vlm_client:
            try:
                logger.info(f"  Using direct pass calls (profile={self.analysis_profile})")
                out = self._classify_scene_direct(image_path)
                scene, reasoning, keywords, prompt, planner_targets, planner_hints, payload = out
                planner_hints = self._maybe_backfill_planner_hints(scene, planner_hints)
                if isinstance(payload, dict):
                    payload["planner_hints"] = planner_hints
                    payload = self._maybe_apply_embeddings_catalog(payload)
                return (scene, reasoning, keywords, prompt, planner_targets, planner_hints, payload)
            except Exception as e:
                logger.error(f"  Direct pass calls failed: {e}")
                # Return error result
                empty_payload = self._scene_classifier_payload(
                    None, scene_override="unknown", error=str(e)
                )
                return "unknown", str(e), [], "", [], {}, empty_payload

        # Priority 3: No VLM available - cannot classify
        logger.error("No scene classification method available (VLM client not initialized)")
        empty_payload = self._scene_classifier_payload(
            None, scene_override="unknown", error="No VLM client available"
        )
        return "unknown", "No VLM client available", [], "", [], {}, empty_payload

    def _classify_scene_with_orchestrator(
            self, image_path: Path
    ) -> Tuple[str, str, List[str], str, List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
        """
        Classify scene using the pass architecture orchestrator.
        This runs the full multi-pass pipeline with premium model routing.
        """

        async def run_orchestrator():
            # ✅ Prefer new signature that accepts issue_catalog
            try:
                return await self.orchestrator.analyze_image(
                    image_path=image_path,
                    options=self.run_options,
                    issue_catalog=self.issue_catalog,
                )
            except TypeError:
                # Backward-compatible fallback
                return await self.orchestrator.analyze_image(
                    image_path=image_path,
                    options=self.run_options,
                )

        # Run async orchestrator
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(run_orchestrator())
        finally:
            loop.close()

        # Convert orchestrator result to expected tuple format
        return self._parse_orchestrator_result(result)

    def _classify_scene_direct(
            self, image_path: Path
    ) -> Tuple[str, str, List[str], str, List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
        """
        Run scene classification using direct pass function calls.

        ✅ This is the fallback when orchestrator isn't available but VLM client is.
        Still supports premium routing through _get_model_config_for_pass().
        """

        async def run_passes():
            results = {
                'scene': 'unknown',
                'reasoning': '',
                'overall_impression': '',
                'image_summary': '',
                'notable_features': [],
                'positives_notes': '',
                'issues_notes': '',
                'keywords': [],
                'catalog_flags': {},
                'issues_natural_language': [],
                'verified_issues': [],
                'passes': {},  # Store per-pass outputs
                'models_used': {},  # Track which model was used for each pass
            }
            context = {}

            # Pass 1a: Scene Type (always Qwen)
            model_config_1a = self._get_model_config_for_pass('1a')
            logger.debug(f"  Pass 1a using model: {model_config_1a.get('model')}")
            results['models_used']['1a'] = model_config_1a.get('model')

            try:
                pass_1a = await run_pass_1a_scene_type(
                    image_path=image_path,
                    vlm_client=self.vlm_client,
                    model_config=model_config_1a,
                )
                results['scene'] = pass_1a.scene
                results['reasoning'] = pass_1a.reasoning or ''
                context['scene'] = pass_1a.scene

                # Store Pass 1a output
                results['passes']['1a'] = {
                    'scene': pass_1a.scene,
                    'confidence': pass_1a.scene_confidence,
                    'reasoning': pass_1a.reasoning,
                }
            except Exception as e:
                logger.warning(f"  Pass 1a failed: {e}")

            # Pass 1b: Positives/Inventory notes (FREEFORM; GPT in premium)
            model_config_1b = self._get_model_config_for_pass('1b')
            logger.debug(f"  Pass 1b using model: {model_config_1b.get('model')}")
            results['models_used']['1b'] = model_config_1b.get('model')

            positives_notes = ""
            try:
                pass_1b = await run_pass_1b_positive_notes(
                    image_path=image_path,
                    vlm_client=self.vlm_client,
                    model_config=model_config_1b,
                    context=context,
                )
                positives_notes = pass_1b.positives_notes
                results['positives_notes'] = positives_notes

                # Store Pass 1b output
                results['passes']['1b'] = {
                    'positives_notes': positives_notes,
                }
            except Exception as e:
                logger.warning(f"  Pass 1b failed: {e}")

            # Pass 1c: Positives notes -> JSON structuring (text-only)
            model_config_1c = self._get_model_config_for_pass('1c')
            logger.debug(f"  Pass 1c using model: {model_config_1c.get('model')}")
            results['models_used']['1c'] = model_config_1c.get('model')

            try:
                pass_1c = await run_pass_1c_positive_structuring(
                    vlm_client=self.vlm_client,
                    model_config=model_config_1c,
                    positives_notes=positives_notes,
                )
                results['overall_impression'] = pass_1c.overall_impression or ""
                results['image_summary'] = pass_1c.image_summary or ""
                results['notable_features'] = pass_1c.notable_features or []
                context['notable_features'] = results['notable_features']

                # Store Pass 1c output
                results['passes']['1c'] = {
                    'overall_impression': results['overall_impression'],
                    'image_summary': results['image_summary'],
                    'notable_features': results['notable_features'],
                }
            except Exception as e:
                logger.warning(f"  Pass 1c failed: {e}")

            # Pass 2a: Issue Detection - freeform notes (GPT in premium)
            model_config_2a = self._get_model_config_for_pass('2a')
            logger.debug(f"  Pass 2a using model: {model_config_2a.get('model')}")
            results['models_used']['2a'] = model_config_2a.get('model')

            freeform_notes = ""
            try:
                pass_2a = await run_pass_2a_issue_detection(
                    image_path=image_path,
                    vlm_client=self.vlm_client,
                    model_config=model_config_2a,
                    context=context,
                    issue_catalog=self.issue_catalog,
                )
                freeform_notes = pass_2a.issues_notes
                results['issues_notes'] = freeform_notes

                # Store Pass 2a output
                results['passes']['2a'] = {
                    'issues_notes': freeform_notes,
                }
            except Exception as e:
                logger.warning(f"  Pass 2a failed: {e}")

            # Pass 2b: Freeform to JSON conversion (always Qwen)
            model_config_2b = self._get_model_config_for_pass('2b')
            logger.debug(f"  Pass 2b using model: {model_config_2b.get('model')}")
            results['models_used']['2b'] = model_config_2b.get('model')

            try:
                pass_2b = await run_pass_2b_issue_verification(
                    image_path=image_path,
                    vlm_client=self.vlm_client,
                    model_config=model_config_2b,
                    freeform_notes=freeform_notes,
                    context=context,
                    issue_catalog=self.issue_catalog,
                )
                results['issues_natural_language'] = pass_2b.issues_natural_language
                results['catalog_flags'] = pass_2b.catalog_flags
                results['verified_issues'] = pass_2b.issues_natural_language
                context['issues_natural_language'] = results['issues_natural_language']

                # Store Pass 2b output
                results['passes']['2b'] = {
                    'issues_natural_language': results['issues_natural_language'],
                    'catalog_flags': results['catalog_flags'],
                }
            except Exception as e:
                logger.warning(f"  Pass 2b failed: {e}")

            # Pass 3: Keyword Extraction (text-only, always Qwen)
            model_config_3 = self._get_model_config_for_pass('3')
            logger.debug(f"  Pass 3 using model: {model_config_3.get('model')}")
            results['models_used']['3'] = model_config_3.get('model')

            try:
                # Ensure Pass 3 context matches new Pass 3 prompt (structured facts only)
                context.setdefault("scene", results.get("scene", "property"))
                context.setdefault("notable_features", results.get("notable_features", []))
                context.setdefault("issues_natural_language", results.get("issues_natural_language", []))

                pass_3 = await run_pass_3_keyword_extraction(
                    vlm_client=self.vlm_client,
                    model_config=model_config_3,
                    context=context,
                    max_keywords=self.max_keywords,
                )
                results['keywords'] = pass_3.keywords

                # Store Pass 3 output
                results['passes']['3'] = {
                    'keywords': pass_3.keywords,
                    'categories': pass_3.keyword_categories,
                }
            except Exception as e:
                logger.warning(f"  Pass 3 failed: {e}")

            return results

        # Run async passes
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(run_passes())
        finally:
            loop.close()

        return self._parse_direct_results(results)

    def _parse_orchestrator_result(
            self, result: Any
    ) -> Tuple[str, str, List[str], str, List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
        """Parse orchestrator ImageAnalysisResult into the expected tuple format."""
        # Handle both dataclass and dict results
        if hasattr(result, 'to_dict'):
            data = result.to_dict()
        elif hasattr(result, '__dict__'):
            data = result.__dict__
        else:
            data = dict(result) if isinstance(result, dict) else {}

        scene = data.get("scene", "unknown")
        reasoning = ""
        confidence = None
        if hasattr(result, 'pass_1a') and result.pass_1a:
            reasoning = result.pass_1a.reasoning or ""
            confidence = getattr(result.pass_1a, 'scene_confidence', None)

        keywords = data.get("keywords", []) or []

        # Build grounding prompt from keywords
        prompt = ". ".join(keywords) + "." if keywords else ""

        # Extract targets if available
        targets = []
        planner_hints: Dict[str, str] = {}

        # Build scene payload for backwards compatibility
        scene_payload = self._scene_classifier_payload(data, scene_override=scene)
        scene_payload['keywords'] = keywords
        scene_payload['overall_impression'] = data.get('overall_impression', '')
        scene_payload['image_summary'] = data.get('image_summary', '')
        scene_payload['notable_features'] = data.get('notable_features', []) or []
        scene_payload['positives_notes'] = data.get('positives_notes', '') or ""
        scene_payload['issues_notes'] = data.get('issues_notes', '') or ""
        scene_payload['groundingdino_prompt'] = prompt
        scene_payload['catalog_flags'] = data.get('catalog_flags', {})
        scene_payload['issues_natural_language'] = data.get('issues_natural_language', [])
        scene_payload['verified_issues'] = data.get('verified_issues', [])
        scene_payload['models_used'] = data.get('models_used', {})

        # Build passes dict for consistent schema
        scene_payload['passes'] = {
            "1a": {
                "scene": scene,
                "confidence": confidence,
                "reasoning": reasoning,
            },
            "1b": {"positives_notes": scene_payload.get("positives_notes", "")},
            "1c": {
                "overall_impression": scene_payload.get("overall_impression", ""),
                "image_summary": scene_payload.get("image_summary", ""),
                "notable_features": scene_payload.get("notable_features", []),
            },
            "2a": {"issues_notes": scene_payload.get("issues_notes", "")},
            "2b": {
                "issues_natural_language": scene_payload.get("issues_natural_language", []),
                "catalog_flags": scene_payload.get("catalog_flags", {}),
            },
            "3": {
                "keywords": keywords,
                "categories": data.get("keyword_categories"),
            },
        }

        return (
            scene,
            reasoning,
            keywords,
            prompt,
            targets,
            planner_hints,
            scene_payload,
        )

    def _parse_direct_results(
            self, results: Dict[str, Any]
    ) -> Tuple[str, str, List[str], str, List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
        """Parse direct pass call results into the expected tuple format."""
        scene = results.get('scene', 'unknown')
        reasoning = results.get('reasoning', '')
        keywords = results.get('keywords', []) or []

        # Build grounding prompt from keywords
        prompt = ". ".join(keywords) + "." if keywords else ""

        targets = []
        planner_hints: Dict[str, str] = {}

        # Build scene payload
        scene_payload = self._scene_classifier_payload(results, scene_override=scene)
        scene_payload['notable_features'] = results.get('notable_features', []) or []
        scene_payload['positives_notes'] = results.get('positives_notes', '') or ""
        scene_payload['issues_notes'] = results.get('issues_notes', '') or ""
        scene_payload['groundingdino_prompt'] = prompt

        # Include passes dict if present
        if results.get('passes'):
            scene_payload['passes'] = results['passes']

        # Include models_used if present
        if results.get('models_used'):
            scene_payload['models_used'] = results['models_used']

        return (
            scene,
            reasoning,
            keywords,
            prompt,
            targets,
            planner_hints,
            scene_payload,
        )

    def run_detection(self,
                      image_path: Path,
                      output_dir: Path,
                      text_prompt: str) -> Dict[str, Any]:
        """Run detection using the selected backend."""
        backend = self.detection_backend

        if backend == "groundingdino":
            return self._run_detection_groundingdino(image_path, output_dir, text_prompt)

        if backend == "dinox":
            return self._run_detection_dinox(image_path, output_dir, text_prompt)

        return {"error": f"Unknown detection backend: {backend}"}

    def _run_detection_groundingdino(self,
                                     image_path: Path,
                                     output_dir: Path,
                                     text_prompt: str) -> Dict[str, Any]:
        """Run GroundingDINO detection on a single image."""
        logger.info(f"Running GroundingDINO detection: {image_path.name}")

        cmd = [
            self.python_exe,
            cfg.GDINO_INFER_SCRIPT,
            "--config_file", cfg.GDINO_CONFIG,
            "--checkpoint_path", cfg.GDINO_CHECKPOINT,
            "--image_path", str(image_path),
            "--text_prompt", text_prompt,
            "--output_dir", str(output_dir),
            "--box_threshold", str(self.box_threshold),
            "--text_threshold", str(self.text_threshold),
            "--extract-chips",
            "--chip-margin", str(self.chip_margin),
        ]

        if cfg.CPU_ONLY:
            cmd.append("--cpu-only")

        if cfg.COMPUTE_CHIP_QUALITY:
            cmd.append("--chip-quality")

        if cfg.CREATE_THUMBNAILS:
            cmd.extend(["--create-thumbnail", "--thumbnail-size", str(cfg.THUMBNAIL_SIZE)])

        code, stdout, stderr = self._run_command(cmd)

        if code != 0:
            logger.error(f"GroundingDINO detection failed: {stderr}")
            return {"error": stderr, "code": code}

        pred_json = output_dir / "pred.json"
        if pred_json.exists():
            with open(pred_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"error": "pred.json not produced"}

    def _run_detection_dinox(self,
                             image_path: Path,
                             output_dir: Path,
                             text_prompt: str) -> Dict[str, Any]:
        """Run DINO-X detection on a single image."""
        logger.info(f"Running DINO-X detection: {image_path.name}")

        dinox_script = getattr(cfg, "DINOX_INFER_SCRIPT", None)

        # Mode 1: Local script
        if dinox_script and Path(dinox_script).exists():
            cmd = [
                self.python_exe,
                dinox_script,
                "--image_path", str(image_path),
                "--text_prompt", text_prompt,
                "--output_dir", str(output_dir),
                "--box_threshold", str(self.box_threshold),
                "--text_threshold", str(self.text_threshold),
            ]

            if getattr(cfg, "DINOX_EXTRACT_CHIPS", False):
                cmd.append("--extract-chips")
                cmd.extend(["--chip-margin", str(self.chip_margin)])

            if getattr(cfg, "DINOX_CPU_ONLY", cfg.CPU_ONLY):
                cmd.append("--cpu-only")

            code, stdout, stderr = self._run_command(cmd)

            if code != 0:
                logger.error(f"DINO-X detection failed: {stderr}")
                return {"error": stderr, "code": code}

            pred_json = output_dir / "pred.json"
            if pred_json.exists():
                with open(pred_json, 'r', encoding='utf-8') as f:
                    return json.load(f)

            return {"error": "pred.json not produced by DINO-X"}

        # Mode 2: API client
        elif self.dinox_client:
            try:
                dinox_prompt = text_prompt.replace(". ", ".").replace(".", ".").strip(".")
                logger.info(f"  Using DINO-X API with prompt: {dinox_prompt}")

                result = self.dinox_client.detect(
                    image_path=image_path,
                    prompt=dinox_prompt,
                    bbox_threshold=self.box_threshold,
                    targets=["bbox"],
                )

                detections = []
                for obj in result.objects:
                    detections.append({
                        "label": obj.category,
                        "score": obj.score,
                        "box": obj.bbox.to_list(),
                    })

                pred_result = {
                    "image": str(image_path),
                    "detections": detections,
                    "detection_count": len(detections),
                    "dinox_task_uuid": result.task_uuid,
                    "processing_time": result.processing_time,
                }

                try:
                    with Image.open(image_path) as pil_im:
                        w, h = pil_im.size
                        pred_result["size"] = {"width": w, "height": h}
                except Exception:
                    pass

                output_dir.mkdir(parents=True, exist_ok=True)
                pred_json = output_dir / "pred.json"
                with open(pred_json, 'w', encoding='utf-8') as f:
                    json.dump(pred_result, f, indent=2)

                logger.info(f"  DINO-X detected {len(detections)} objects in {result.processing_time:.1f}s")
                return pred_result

            except Exception as e:
                logger.error(f"DINO-X API detection failed: {e}")
                return {"error": str(e)}

        # Mode 3: No option available
        else:
            api_token = (
                    getattr(cfg, "DINOX_API_TOKEN", None) or
                    getattr(cfg, "DINOX_API_KEY", None) or
                    os.environ.get("DINOX_API_TOKEN") or
                    os.environ.get("DINOX_API_KEY")
            )

            if api_token and not DINOX_CLIENT_AVAILABLE:
                return {
                    "error": "DINOX_API_TOKEN is set but dinox_client.py is not available."
                }

            return {
                "error": "DINO-X backend selected but no detection method available."
            }

    def run_verification(self, output_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Run chip verification on detection results.

        Note: Pass 2b (verification) always uses Qwen even in premium mode
        because it's high-volume and doesn't benefit much from GPT.
        """
        import subprocess

        if self.skip_verification:
            return {"skipped": True}

        chip_verifier = cfg.TOOLS_DIR / "chip_verifier.py"
        if not chip_verifier.exists():
            logger.warning("chip_verifier.py not found, skipping verification")
            return {"skipped": True, "reason": "chip_verifier.py not found"}

        logger.info(f"Verifying detections in: {output_dir.name}")

        # Get model config for pass 2b - always Qwen even in premium
        model_config = self._get_model_config_for_pass('2b')

        # Verification always uses LM Studio/Qwen (high volume pass)
        cmd = [
            self.python_exe,
            chip_verifier,
            str(output_dir),
            "--model", model_config.get('model', cfg.LM_STUDIO_MODEL),
            "--lm-studio-url", model_config.get('url', cfg.LM_STUDIO_URL),
            "--max-chips", str(cfg.MAX_CHIPS_PER_DETECTION),
        ]

        if self.debug:
            cmd.append("--debug")

        code, stdout, stderr = self._run_command(cmd)

        if code != 0:
            logger.error(f"Verification failed: {stderr}")
            return {"error": stderr, "code": code}

        ver_json = output_dir / "verification_results.json"
        if ver_json.exists():
            with open(ver_json, 'r', encoding='utf-8') as f:
                return json.load(f)

        return None

    def analyze_image(self,
                      image_path: Path,
                      output_dir: Path) -> ImageAnalysisResult:
        """Analyze a single image through the complete pipeline."""
        t0 = time.time()

        try:
            # 1. Classify scene and get keywords
            (
                scene,
                reasoning,
                keywords,
                prompt,
                planner_targets,
                planner_hints,
                scene_details,
            ) = self.classify_scene(image_path)

            logger.info(f"  Scene: {scene}")
            logger.info(f"  Keywords: {len(keywords)} objects")

            if not prompt:
                logger.warning(f"  No detection prompt for {scene}, skipping detection")
                return ImageAnalysisResult(
                    image_path=str(image_path),
                    scene=scene,
                    scene_data=scene_details,
                    scene_classifier=scene_details,
                    keywords_used=keywords,
                    detection_count=0,
                    output_dir=str(output_dir),
                    processing_time=time.time() - t0,
                    error="No detection prompt generated"
                )

            # 2. Run detection (routed based on backend)
            detection_result = self.run_detection(image_path, output_dir, prompt)

            if "error" in detection_result:
                return ImageAnalysisResult(
                    image_path=str(image_path),
                    scene=scene,
                    scene_data=scene_details,
                    scene_classifier=scene_details,
                    keywords_used=keywords,
                    detection_count=0,
                    output_dir=str(output_dir),
                    processing_time=time.time() - t0,
                    error=detection_result["error"]
                )

            detections = detection_result.get("detections", []) or []
            detection_result["planner_targets"] = planner_targets
            detection_result["planner_hints"] = planner_hints

            size_info = detection_result.get("size") or {}
            img_w = int(size_info.get("width") or 0)
            img_h = int(size_info.get("height") or 0)
            if (img_w <= 0 or img_h <= 0) and image_path.exists():
                try:
                    with Image.open(image_path) as pil_im:
                        img_w, img_h = pil_im.size
                except Exception:
                    pass

            if (
                    getattr(cfg, "ROI_HINTS_ENABLED", False)
                    and planner_hints  # Skip if planner_hints is empty
                    and detections
                    and img_w > 0
                    and img_h > 0
            ):
                detections_by_class: Dict[str, List[Dict[str, Any]]] = {}
                unlabeled: List[Dict[str, Any]] = []
                for det in detections:
                    if det.get("score") is None:
                        det["score"] = 0.0
                    label_name = str(det.get("label") or "").strip()
                    if not label_name:
                        unlabeled.append(det)
                        continue
                    detections_by_class.setdefault(label_name, []).append(det)

                for label_name, dets in detections_by_class.items():
                    norm_label = self._normalize_label_for_hint(label_name)
                    roi_hint = planner_hints.get(norm_label, "unknown")
                    apply_roi_hint_bonus_overlap(
                        dets=dets,
                        roi_hint=roi_hint,
                        W=img_w,
                        H=img_h,
                        full_bonus=getattr(cfg, "ROI_FULL_BONUS", 0.06),
                        half_bonus=getattr(cfg, "ROI_HALF_BONUS", 0.03),
                        penalty=getattr(cfg, "ROI_PENALTY", 0.03),
                        hi=getattr(cfg, "ROI_OVERLAP_HI", 0.40),
                        lo=getattr(cfg, "ROI_OVERLAP_LO", 0.10),
                        attach_debug=True,
                    )

                if unlabeled:
                    apply_roi_hint_bonus_overlap(
                        dets=unlabeled,
                        roi_hint="unknown",
                        W=img_w,
                        H=img_h,
                        full_bonus=getattr(cfg, "ROI_FULL_BONUS", 0.06),
                        half_bonus=getattr(cfg, "ROI_HALF_BONUS", 0.03),
                        penalty=getattr(cfg, "ROI_PENALTY", 0.03),
                        hi=getattr(cfg, "ROI_OVERLAP_HI", 0.40),
                        lo=getattr(cfg, "ROI_OVERLAP_LO", 0.10),
                        attach_debug=True,
                    )

            # Sort by score for stable behavior
            detections.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)

            # Special-case filters (mirror containment, etc.)
            special_case_cfg = getattr(cfg, "SPECIAL_CASE_FILTERS", {})
            if detections and special_case_cfg:
                pre_case = len(detections)
                detections = apply_special_case_filters(
                    detections,
                    image_size=(img_w, img_h),
                    config=special_case_cfg,
                )
                if pre_case != len(detections):
                    logger.info(
                        f"  Special cases: {pre_case} → {len(detections)} detections"
                    )

            # NMS (AFTER detection, BEFORE verification)
            if getattr(cfg, "USE_NMS", True) and detections:
                pre_nms_count = len(detections)
                detections = class_aware_nms(
                    detections,
                    per_class_iou=getattr(cfg, "NMS_PER_CLASS", None),
                    default_iou=getattr(cfg, "NMS_DEFAULT_IOU", 0.3),
                )
                logger.info(f"  NMS: {pre_nms_count} → {len(detections)} detections")

            # Optional: enforce per-scene caps
            if getattr(cfg, "USE_SCENE_CAPS", False) and detections:
                pre_cap_count = len(detections)
                detections = enforce_scene_caps(
                    detections,
                    scene=scene,
                    caps_map=getattr(cfg, "SCENE_CAPS", None),
                )
                if pre_cap_count != len(detections):
                    logger.info(f"  Scene caps: {pre_cap_count} → {len(detections)} detections")

            if (
                    getattr(cfg, "ROI_HINTS_ENABLED", False)
                    and detections
                    and logger.isEnabledFor(logging.DEBUG)
            ):
                for det in detections:
                    dbg = det.get("roi_debug", {})
                    logger.debug(
                        f"ROI [{det.get('label')}]: score={float(det.get('score', 0.0)):.3f} "
                        f"hint={dbg.get('hint')} adj={dbg.get('adj')} "
                        f"overlap={dbg.get('overlap_ratio', 0):.2f} "
                        f"maj={dbg.get('majority_zone')}({dbg.get('majority_r', 0):.2f}) "
                        f"img%={dbg.get('img_frac', 0):.3f}"
                    )

            # Write survivors back
            detection_result["detections"] = detections

            for k in ("count", "num_detections", "detections_count"):
                if k in detection_result:
                    detection_result[k] = len(detections)

            pred_json_path = output_dir / "pred.json"
            with open(pred_json_path, "w", encoding="utf-8") as f:
                json.dump(detection_result, f, indent=2, ensure_ascii=False)

            # (Optional) prune chips for detections that were dropped
            if getattr(cfg, "PRUNE_DROPPED_CHIPS", False):
                try:
                    chip_dir = Path(
                        (detection_result.get("chip_extraction", {}) or {}).get("chips_directory",
                                                                                str(output_dir / "chips"))
                    )
                    keep = set()
                    for d in detections:
                        fn = ((d.get("chip_info") or {}).get("filename"))
                        if fn:
                            keep.add(str((chip_dir / fn).resolve()))

                    if chip_dir.exists():
                        pruned = 0
                        for fp in chip_dir.glob("*"):
                            if keep and str(fp.resolve()) not in keep:
                                try:
                                    fp.unlink()
                                    pruned += 1
                                except Exception:
                                    pass
                        if pruned > 0:
                            logger.info(f"  Pruned {pruned} dropped chip file(s)")
                except Exception as e:
                    logger.warning(f"  Chip pruning failed: {e}")

            # Redraw overlay
            try:
                nms_overlay = output_dir / "pred_nms.jpg"
                redraw_overlay(image_path, detections, nms_overlay)

                raw_overlay = output_dir / "pred.jpg"
                before_overlay = output_dir / "pred_before.jpg"
                if raw_overlay.exists():
                    try:
                        shutil.copy2(raw_overlay, before_overlay)
                    except Exception:
                        before_overlay.write_bytes(raw_overlay.read_bytes())

                nms_overlay.replace(raw_overlay)
                logger.info("  Overlay updated with NMS/ROI filtered detections")
            except Exception as e:
                logger.warning(f"  Failed to redraw overlay: {e}")

            detection_count = len(detections)

            # 3. Verify detections (optional)
            verified_count = None
            if not self.skip_verification and detection_count > 0:
                logger.info(f"  Verifying detections for {Path(image_path).name} in: {Path(output_dir).name}")
                verification_result = self.run_verification(output_dir)
                if verification_result and "summary" in verification_result:
                    verified_count = verification_result["summary"].get("valid", 0)
                    logger.info(f"  Verified: {verified_count}/{detection_count}")

            return ImageAnalysisResult(
                image_path=str(image_path),
                scene=scene,
                scene_data=scene_details,
                scene_classifier=scene_details,
                keywords_used=keywords,
                detection_count=detection_count,
                verified_count=verified_count,
                output_dir=str(output_dir),
                processing_time=time.time() - t0
            )

        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            fallback_scene = self._scene_classifier_payload(
                None, scene_override="unknown", error=str(e)
            )
            return ImageAnalysisResult(
                image_path=str(image_path),
                scene="unknown",
                scene_data=fallback_scene,
                scene_classifier=fallback_scene,
                keywords_used=[],
                detection_count=0,
                output_dir=str(output_dir),
                processing_time=time.time() - t0,
                error=str(e)
            )

    def analyze_property(self,
                         property_key: str,
                         images: List[Path]) -> PropertyAnalysisJob:
        """Analyze all images for a property."""
        t0 = time.time()

        job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        job_dir = self.artifacts_root / property_key / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Property: {property_key}")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Images: {len(images)}")
        logger.info(f"Detection Backend: {self.detection_backend}")
        logger.info(f"Analysis Profile: {self.analysis_profile}")
        logger.info(f"Pass Architecture: {self.use_pass_architecture}")
        if self.analysis_profile == "premium" and self.gpt_config:
            logger.info(f"GPT Model: {self.gpt_config.get('model', 'N/A')}")
        logger.info(f"{'=' * 60}\n")

        results = []

        for idx, image_path in enumerate(images):
            logger.info(f"\n[{idx + 1}/{len(images)}] Processing: {image_path.name}")
            logger.info(f"{'-' * 60}")

            output_dir = job_dir / f"img_{idx:03d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"\n[{idx + 1}/{len(images)}] Processing: {image_path.name} → {output_dir.name}")

            result = self.analyze_image(image_path, output_dir)
            results.append(result)

        job = PropertyAnalysisJob(
            job_id=job_id,
            property_key=property_key,
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            results=results,
            artifacts_dir=str(job_dir),
            total_processing_time=time.time() - t0,
            parameters={
                "detection_backend": self.detection_backend,
                "analysis_profile": self.analysis_profile,
                "use_pass_architecture": self.use_pass_architecture,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "chip_margin": self.chip_margin,
                "max_keywords": self.max_keywords,
                "include_conditions": self.include_conditions,
                "skip_verification": self.skip_verification,
                "use_nms": getattr(cfg, "USE_NMS", True),
                "use_scene_caps": getattr(cfg, "USE_SCENE_CAPS", False),
            }
        )

        logger.info("\n" + "=" * 60)
        logger.info("✅ ANALYSIS COMPLETE")
        logger.info("=" * 60)

        return job

    def _maybe_apply_embeddings_catalog(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backfill/override catalog_flags using embeddings based on issues_natural_language.
        Keeps your downstream renovation_needs + UI schema intact.
        """
        if not isinstance(payload, dict):
            return payload
        if not self.catalog_matcher:
            return payload
        if not getattr(cfg, "USE_EMBEDDINGS_CATALOG", False):
            return payload

        override = bool(getattr(cfg, "EMBEDDINGS_OVERRIDE_EXISTING_FLAGS", True))
        attach = bool(getattr(cfg, "EMBEDDINGS_ATTACH_CANDIDATES", True))

        # Prefer issues from passes["2b"], fallback to flat field
        passes = payload.get("passes") if isinstance(payload.get("passes"), dict) else {}
        issues = None
        if isinstance(passes, dict):
            p2b = passes.get("2b") if isinstance(passes.get("2b"), dict) else {}
            issues = p2b.get("issues_natural_language")

        if issues is None:
            issues = payload.get("issues_natural_language")

        if not isinstance(issues, list) or not issues:
            return payload

        existing = payload.get("catalog_flags") or {}
        if existing and not override:
            return payload

        flags = self.catalog_matcher.build_catalog_flags_and_annotate(
            issues_natural_language=issues,
            attach_candidates=attach,
        )

        payload["catalog_flags"] = flags

        # Keep passes schema consistent too
        if isinstance(passes, dict):
            p2b = passes.setdefault("2b", {})
            if isinstance(p2b, dict):
                p2b["issues_natural_language"] = issues  # (may now have annotations)
                p2b["catalog_flags"] = flags

        # Also keep flat field in sync (your save_photo_intel reads both)
        payload["issues_natural_language"] = issues

        return payload

    @staticmethod
    def _scene_classifier_payload(
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
        payload.setdefault("positives_notes", "")
        payload.setdefault("issues_notes", "")
        payload.setdefault("reasoning", "" if error is None else error)
        payload.setdefault("targets", [])
        payload.setdefault("gdino_terms", [])
        payload.setdefault("keywords", [])
        payload.setdefault("groundingdino_prompt", "")
        payload.setdefault("issues_natural_language", [])
        payload.setdefault("verified_issues", [])
        payload.setdefault("catalog_flags", {})
        payload.setdefault("issue_visual_anchors", [])
        payload.setdefault("processing_time", payload.get("processing_time"))
        payload.setdefault("prompt_version", payload.get("prompt_version", ""))
        payload.setdefault("scene_policy_version", payload.get("scene_policy_version", ""))
        payload.setdefault("image", payload.get("image"))

        if error and not payload.get("error"):
            payload["error"] = error
        payload.setdefault("error", payload.get("error"))

        return payload

    @staticmethod
    def _build_photo_entry(result: ImageAnalysisResult) -> Dict[str, Any]:
        """Create a serializable photo entry including scene classifier details."""
        scene_payload = AutoAnalyzer._scene_classifier_payload(result.scene_classifier or result.scene_data)

        return {
            "image_path": result.image_path,
            "scene": result.scene,
            "is_staged": scene_payload.get("is_staged"),
            "overall_impression": scene_payload.get("overall_impression", ""),
            "reasoning": scene_payload.get("reasoning", ""),
            "targets": scene_payload.get("targets", []),
            "gdino_terms": scene_payload.get("gdino_terms", []),
            "keywords": scene_payload.get("keywords", result.keywords_used),
            "groundingdino_prompt": scene_payload.get("groundingdino_prompt", ""),
            "issues_natural_language": scene_payload.get("issues_natural_language", []),
            "verified_issues": scene_payload.get("verified_issues", []),
            "catalog_flags": scene_payload.get("catalog_flags", {}),
            "issue_visual_anchors": scene_payload.get("issue_visual_anchors", []),
            "scene_classifier": scene_payload,
            "keywords_used": result.keywords_used,
            "detection_count": result.detection_count,
            "verified_count": result.verified_count,
            "output_dir": result.output_dir,
            "processing_time": result.processing_time,
            "error": result.error,
        }

    @staticmethod
    def _pick_first_metadata(results: List[ImageAnalysisResult], key: str, default: str = "") -> str:
        for res in results:
            payload = AutoAnalyzer._scene_classifier_payload(res.scene_classifier or res.scene_data)
            val = payload.get(key)
            if isinstance(val, str) and val:
                return val
        return default

    def save_photo_intel(
            self,
            job: PropertyAnalysisJob,
            output_path: Optional[Path] = None,
            generate_summary: bool = True
    ) -> Path:
        """Persist per-photo intelligence (including scene classifier fields)."""
        created_at = datetime.utcnow().isoformat() + "Z"

        def _safe_list(x):
            """Ensure x is a list."""
            return x if isinstance(x, list) else []

        photos: Dict[str, Any] = {}
        for res in job.results:
            image_key = Path(res.image_path).name
            payload = self._scene_classifier_payload(res.scene_classifier or res.scene_data)

            # Use passes from payload if present (from _classify_scene_direct),
            # otherwise reconstruct for backwards compatibility (legacy runs)
            passes = payload.get("passes", None)
            if passes is None:
                # Fallback: reconstruct passes from flat fields (legacy runs only)
                passes = {
                    "1a": {
                        "scene": payload.get("scene", "unknown"),
                        "confidence": payload.get("scene_confidence"),
                        "reasoning": payload.get("reasoning", ""),
                    },
                    "1b": {"positives_notes": payload.get("positives_notes", "")},
                    "1c": {
                        "overall_impression": payload.get("overall_impression", ""),
                        "image_summary": payload.get("image_summary", ""),
                        "notable_features": payload.get("notable_features", []) or [],
                    },
                    "2a": {"issues_notes": payload.get("issues_notes", "")},
                    "2b": {
                        "issues_natural_language": payload.get("issues_natural_language", []) or [],
                        "catalog_flags": payload.get("catalog_flags", {}) or {},
                    },
                    "3": {
                        "keywords": payload.get("keywords", []) or [],
                        "categories": payload.get("keyword_categories"),
                    },
                }

            # Keep flat fields for backwards compatibility, but UI should read passes
            photos[image_key] = {
                "image_path": res.image_path,
                "scene": payload.get("scene", res.scene),
                **payload,
                "passes": passes,
            }

        # Build room_groups aggregation
        room_groups: Dict[str, Any] = {}

        for img_key, p in photos.items():
            scene = (p.get("scene") or "unknown").strip()
            group = SCENE_TO_GROUP_UI.get(scene, "other")

            g = room_groups.setdefault(group, {
                "scenes_included": SCENE_GROUPS_UI.get(group, []),
                "image_keys": [],
                "image_count": 0,
                "positives": {"notes": [], "notable_features": []},
                "issues": {"notes": [], "issues_natural_language": []},
            })

            g["image_keys"].append(img_key)
            g["image_count"] += 1

            passes = p.get("passes", {}) or {}

            # Positives
            pos_notes = ((passes.get("1b", {}) or {}).get("positives_notes") or "").strip()
            if pos_notes:
                g["positives"]["notes"].append(pos_notes)

            nf = _safe_list((passes.get("1c", {}) or {}).get("notable_features"))
            for feat in nf:
                s = str(feat).strip()
                if s and s not in g["positives"]["notable_features"]:
                    g["positives"]["notable_features"].append(s)

            # Issues
            issue_notes = ((passes.get("2a", {}) or {}).get("issues_notes") or "").strip()
            if issue_notes:
                g["issues"]["notes"].append(issue_notes)

            issues = _safe_list((passes.get("2b", {}) or {}).get("issues_natural_language"))
            for it in issues:
                if isinstance(it, dict) and it.get("description"):
                    g["issues"]["issues_natural_language"].append({
                        "source_image": img_key,
                        "description": it.get("description", ""),
                        "rough_category": it.get("rough_category", ""),
                        "location_hint": it.get("location_hint", ""),
                    })

        photo_intel = {
            "run_id": job.job_id,
            "job_id": job.job_id,
            "property_key": job.property_key,
            "timestamp": job.timestamp,
            "created_at": created_at,
            "artifacts_dir": job.artifacts_dir,
            "detection_backend": self.detection_backend,
            "analysis_profile": self.analysis_profile,
            "used_pass_architecture": self.use_pass_architecture,
            "pass_toggles": self.pass_toggles if self.pass_toggles else None,
            "model_overrides": self.model_overrides if self.model_overrides else None,
            "model": getattr(cfg, "LM_STUDIO_MODEL", ""),
            "gpt_model": self.gpt_config.get('model') if self.gpt_config else None,
            "prompt_version": self._pick_first_metadata(job.results, "prompt_version", ""),
            "scene_policy_version": self._pick_first_metadata(job.results, "scene_policy_version", ""),
            "photos": photos,
            "room_groups": room_groups,
        }

        try:
            photo_intel["renovation_needs"] = build_renovation_needs(job, issue_catalog=self.issue_catalog)
        except Exception as exc:
            logger.error(f"Failed to build renovation_needs: {exc}", exc_info=True)

        output_path = output_path or Path(job.artifacts_dir) / "photo_intel.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(photo_intel, f, indent=2, ensure_ascii=False)

        logger.info(f"Photo intel saved to: {output_path}")

        if generate_summary:
            summary_path = output_path.parent / "property_summary.json"
            self.generate_property_summary(job, photo_intel_path=output_path, output_path=summary_path)

        return output_path

    def generate_property_summary(
            self,
            job: PropertyAnalysisJob,
            photo_intel_path: Optional[Path] = None,
            output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Generate a property-level summary from the analysis results using Pass 4, 4a, and 4b.

        Always writes property_summary.json and embeds property_pass4 into photo_intel.json,
        even on failure (with error field and empty values) so the UI has consistent structure.
        """
        if not getattr(cfg, "GENERATE_PROPERTY_SUMMARY", True):
            logger.info("Property summary generation disabled in config")
            return None

        logger.info(f"\n{'=' * 60}")
        logger.info("Generating Property Summary...")
        logger.info(f"{'=' * 60}")

        if output_path is None:
            output_path = Path(job.artifacts_dir) / "property_summary.json"

        # Build all_results from job for Pass 4
        all_results = {}
        scene_counts: Dict[str, int] = {}
        for res in job.results:
            image_key = Path(res.image_path).name
            payload = self._scene_classifier_payload(res.scene_classifier or res.scene_data)
            all_results[image_key] = payload
            # Count scenes
            scene = payload.get("scene", "unknown")
            scene_counts[scene] = scene_counts.get(scene, 0) + 1

        # Initialize with empty/error values for Pass 4
        property_summary = ""
        investment_considerations = []
        estimated_condition = ""
        confidence = None
        error_msg = None
        model_used = ""

        # Initialize Pass 4a/4b outputs with defaults
        room_summaries: Dict[str, Any] = {}
        issues_by_category: Dict[str, int] = {}
        total_issues_found = 0

        # Legacy UI card fields (from Pass 4b)
        overall_condition = ""
        overall_summary = ""
        investment_verdict = ""
        investment_rationale = ""
        renovation_scope = ""
        renovation_priorities: List[str] = []
        risk_flags: List[str] = []
        deferred_maintenance: List[str] = []

        errors: Dict[str, str] = {}

        logger.warning(
            f"Pass4 gate: VLM_CLIENT_AVAILABLE={VLM_CLIENT_AVAILABLE} "
            f"vlm_client={bool(self.vlm_client)} "
            f"PASS_ARCHITECTURE_AVAILABLE={PASS_ARCHITECTURE_AVAILABLE} "
            f"auto_analyzer_file={__file__}"
        )

        # Try Pass 4 if VLM client is available
        if VLM_CLIENT_AVAILABLE and self.vlm_client and PASS_ARCHITECTURE_AVAILABLE:
            # --- Pass 4 (original property summary) ---
            try:
                logger.info("  Using Pass 4 (run_pass_4_property_summary)")
                model_config_4 = self._get_model_config_for_pass('4')
                model_used = model_config_4.get('model', '')
                logger.info(f"  Model: {model_used}")

                async def run_pass4():
                    return await run_pass_4_property_summary(
                        vlm_client=self.vlm_client,
                        model_config=model_config_4,
                        all_results=all_results,
                    )

                loop = asyncio.new_event_loop()
                try:
                    pass_4_result = loop.run_until_complete(run_pass4())
                finally:
                    loop.close()

                # Extract successful results
                property_summary = pass_4_result.property_summary or ""
                investment_considerations = pass_4_result.investment_considerations or []
                estimated_condition = pass_4_result.estimated_condition or ""
                confidence = getattr(pass_4_result, "confidence", None)

                logger.info("  Pass 4 completed successfully")

            except Exception as e:
                errors["pass4"] = f"Pass 4 failed: {e}"
                logger.error(errors["pass4"], exc_info=True)

            # --- Pass 4a (room summaries aggregation) ---
            try:
                logger.info("  Using Pass 4a (run_pass_4a_room_summaries)")
                model_config_4a = self._get_model_config_for_pass('4a')
                logger.info(f"  Model: {model_config_4a.get('model', '')}")

                async def run_pass4a():
                    return await run_pass_4a_room_summaries(
                        vlm_client=self.vlm_client,
                        model_config=model_config_4a,
                        all_results=all_results,
                        scene_counts=scene_counts,
                    )

                loop = asyncio.new_event_loop()
                try:
                    pass_4a_result = loop.run_until_complete(run_pass4a())
                finally:
                    loop.close()

                room_summaries = pass_4a_result.room_summaries or {}
                issues_by_category = pass_4a_result.issues_by_category or {}
                total_issues_found = pass_4a_result.total_issues_found

                logger.info(f"  Pass 4a completed: {len(room_summaries)} room groups, {total_issues_found} issues")

            except Exception as e:
                errors["pass4a"] = f"Pass 4a failed: {e}"
                logger.error(errors["pass4a"], exc_info=True)

            # --- Pass 4b (legacy UI card fields) ---
            try:
                logger.info("  Using Pass 4b (run_pass_4b_property_card_fields)")
                model_config_4b = self._get_model_config_for_pass('4b')
                logger.info(f"  Model: {model_config_4b.get('model', '')}")

                async def run_pass4b():
                    return await run_pass_4b_property_card_fields(
                        vlm_client=self.vlm_client,
                        model_config=model_config_4b,
                        room_summaries=room_summaries,
                        total_issues_found=total_issues_found,
                        total_images_analyzed=len(all_results),
                        issues_by_category=issues_by_category,
                    )

                loop = asyncio.new_event_loop()
                try:
                    pass_4b_result = loop.run_until_complete(run_pass4b())
                finally:
                    loop.close()

                overall_condition = pass_4b_result.overall_condition or ""
                overall_summary = pass_4b_result.overall_summary or ""
                investment_verdict = pass_4b_result.investment_verdict or ""
                investment_rationale = pass_4b_result.investment_rationale or ""
                renovation_scope = pass_4b_result.renovation_scope or ""
                renovation_priorities = pass_4b_result.renovation_priorities or []
                risk_flags = pass_4b_result.risk_flags or []
                deferred_maintenance = pass_4b_result.deferred_maintenance or []

                logger.info("  Pass 4b completed successfully")

            except Exception as e:
                errors["pass4b"] = f"Pass 4b failed: {e}"
                logger.error(errors["pass4b"], exc_info=True)

        else:
            # VLM client or pass architecture not available
            missing = []
            if not VLM_CLIENT_AVAILABLE:
                missing.append("VLM client")
            if not self.vlm_client:
                missing.append("vlm_client instance")
            if not PASS_ARCHITECTURE_AVAILABLE:
                missing.append("pass architecture")
            error_msg = f"Passes unavailable: missing {', '.join(missing)}"
            logger.warning(error_msg)

        # Combine errors
        if errors:
            error_msg = "; ".join(errors.values())

        # Build summary dict - always write even on failure
        # Includes both Pass 4 fields AND legacy UI fields from 4a/4b
        summary_data = {
            "property_key": job.property_key,
            "job_id": job.job_id,
            "timestamp": job.timestamp,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "summary_version": "pass4_v2",
            "analysis_profile": self.analysis_profile,

            # Pass 4 fields (original)
            "property_summary": property_summary,
            "investment_considerations": investment_considerations,
            "estimated_condition": estimated_condition,
            "confidence": confidence,

            # Pass 4a fields (room summaries)
            "room_summaries": room_summaries,
            "issues_by_category": issues_by_category,
            "total_issues_found": total_issues_found,

            # Pass 4b fields (legacy UI card)
            "overall_condition": overall_condition,
            "overall_summary": overall_summary,
            "investment_verdict": investment_verdict,
            "investment_rationale": investment_rationale,
            "renovation_scope": renovation_scope,
            "renovation_priorities": renovation_priorities,
            "risk_flags": risk_flags,
            "deferred_maintenance": deferred_maintenance,

            # Metadata
            "total_images_analyzed": len(job.results),
            "model_used": model_used,
            "error": error_msg,
            "errors": errors if errors else None,
        }

        # Add renovation_needs if available
        try:
            summary_data["renovation_needs"] = build_renovation_needs(job, issue_catalog=self.issue_catalog)
        except Exception as exc:
            logger.warning(f"Could not build renovation_needs: {exc}")

        # Write property_summary.json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Property summary saved to: {output_path}")

        # Embed Pass 4 output into photo_intel.json so UI has single payload
        if photo_intel_path and photo_intel_path.exists():
            try:
                with open(photo_intel_path, 'r', encoding='utf-8') as f:
                    photo_intel = json.load(f)

                photo_intel["property_pass4"] = {
                    "property_summary": property_summary,
                    "investment_considerations": investment_considerations,
                    "estimated_condition": estimated_condition,
                    "confidence": confidence,
                    "error": errors.get("pass4"),
                }

                photo_intel["property_pass4a"] = {
                    "room_summaries": room_summaries,
                    "issues_by_category": issues_by_category,
                    "total_issues_found": total_issues_found,
                    "error": errors.get("pass4a"),
                }

                photo_intel["property_pass4b"] = {
                    "overall_condition": overall_condition,
                    "overall_summary": overall_summary,
                    "investment_verdict": investment_verdict,
                    "investment_rationale": investment_rationale,
                    "renovation_scope": renovation_scope,
                    "renovation_priorities": renovation_priorities,
                    "risk_flags": risk_flags,
                    "deferred_maintenance": deferred_maintenance,
                    "error": errors.get("pass4b"),
                }

                # Also embed full summary for convenience
                photo_intel["property_summary"] = summary_data

                with open(photo_intel_path, "w", encoding="utf-8") as f:
                    json.dump(photo_intel, f, indent=2, ensure_ascii=False)

                logger.info(f"Pass 4/4a/4b outputs embedded into: {photo_intel_path}")
            except Exception as exc:
                logger.warning(f"Could not embed property_pass4 into photo_intel.json: {exc}")

        return output_path

    def create_html_report(self, job: PropertyAnalysisJob, output_path: Path):
        """Generate a simple HTML report."""
        profile_color = "#9C27B0" if self.analysis_profile == "premium" else "#607D8B"
        gpt_model_info = ""
        if self.analysis_profile == "premium" and self.gpt_config:
            gpt_model_info = f"<p><strong>GPT Model:</strong> {self.gpt_config.get('model', 'N/A')}</p>"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Analysis Report - {job.property_key}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .error {{ color: red; }}
        .backend {{ font-weight: bold; color: #2196F3; }}
        .profile {{ font-weight: bold; color: {profile_color}; }}
    </style>
</head>
<body>
    <h1>Property Analysis Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Property:</strong> {job.property_key}</p>
        <p><strong>Job ID:</strong> {job.job_id}</p>
        <p><strong>Timestamp:</strong> {job.timestamp}</p>
        <p><strong>Detection Backend:</strong> <span class="backend">{self.detection_backend}</span></p>
        <p><strong>Analysis Profile:</strong> <span class="profile">{self.analysis_profile.upper()}</span></p>
        <p><strong>Pass Architecture:</strong> {self.use_pass_architecture}</p>
        {gpt_model_info}
        <p><strong>Images Processed:</strong> {len(job.results)}</p>
        <p><strong>Total Time:</strong> {job.total_processing_time:.1f} seconds</p>
        <p><strong>Artifacts:</strong> {job.artifacts_dir}</p>
    </div>

    <h2>Parameters</h2>
    <ul>
"""

        for key, value in job.parameters.items():
            html += f"        <li><strong>{key}:</strong> {value}</li>\n"

        html += """    </ul>

    <h2>Results</h2>
    <table>
        <tr>
            <th>Image</th>
            <th>Scene</th>
            <th>Keywords</th>
            <th>Detections</th>
            <th>Verified</th>
            <th>Time (s)</th>
        </tr>
"""

        for result in job.results:
            img_name = Path(result.image_path).name
            verified_text = str(result.verified_count) if result.verified_count is not None else "N/A"
            error_class = ' class="error"' if result.error else ''

            html += f"""        <tr{error_class}>
            <td>{img_name}</td>
            <td>{result.scene}</td>
            <td>{len(result.keywords_used)}</td>
            <td>{result.detection_count}</td>
            <td>{verified_text}</td>
            <td>{result.processing_time:.1f}</td>
        </tr>
"""

        html += """    </table>
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML report saved to: {output_path}")