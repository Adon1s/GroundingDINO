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

# Ensure repo root is in sys.path BEFORE any tools.* imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from tools.run_pipeline import redraw_overlay

# Import configuration
try:
    from tools import pipeline_config as cfg
except ImportError:
    print("ERROR: pipeline_config.py not found!")
    sys.exit(1)

# Import renovation cost table from dedicated module
try:
    from tools.renovation_costs import RENOVATION_COST_TABLE
except ImportError:
    print("WARNING: renovation_costs.py not found, using empty cost table")
    RENOVATION_COST_TABLE = {}

# Import defect events layer
try:
    from tools.defect_events import build_defect_events, generate_work_items, build_search_index
    DEFECT_EVENTS_AVAILABLE = True
except ImportError:
    DEFECT_EVENTS_AVAILABLE = False

# Import NMS postprocessing
try:
    from tools.postprocess import (
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
    from tools.property_summarizer import PropertySummarizer
except ImportError:
    PropertySummarizer = None
    print("INFO: property_summarizer.py not found, property summaries will be skipped")

# Import DINO-X client (optional - graceful fallback to legacy mode)
try:
    from tools.dinox_client import DINOXClient, create_dinox_client_from_config

    DINOX_CLIENT_AVAILABLE = True
except ImportError:
    DINOXClient = None
    create_dinox_client_from_config = None
    DINOX_CLIENT_AVAILABLE = False
    print("INFO: dinox_client.py not found, DINO-X will use legacy script mode only")

# Embeddings catalog matcher (optional)
try:
    from tools.catalog_embeddings import CatalogEmbeddingsRetriever

    EMBEDDINGS_MATCHER_AVAILABLE = True
except Exception:
    CatalogEmbeddingsRetriever = None
    EMBEDDINGS_MATCHER_AVAILABLE = False

# Import VLM client (for direct calls when orchestrator unavailable)
try:
    from tools.vlm_client import VLMClient, create_vlm_client, get_model_configs_from_pipeline_config

    VLM_CLIENT_AVAILABLE = True
except ImportError:
    VLMClient = None
    create_vlm_client = None
    get_model_configs_from_pipeline_config = None
    VLM_CLIENT_AVAILABLE = False

try:
    from tools.pass_config import (
        SceneClassifierRunOptions,
        PassToggles,
        PassModelOverrides,
        pick_model_for_pass,
        get_model_config_for_pass, PassKey,
    )
    from tools.scene_classifier_orchestrator import (
        SceneClassifierOrchestrator,
        create_orchestrator_from_config,
    )
    from tools.scene_classifier_passes import (
        run_pass_1a_scene_type,
        run_pass_1b_feature_notes,
        run_pass_1c_feature_structuring,
        run_pass_2a,
        run_pass_2b,
        run_pass_2c,
        run_pass_2d,
        run_pass_3_keyword_extraction,
    )

    PASS_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        f"Pass architecture import failed: {e}. "
        "Fix scene_classifier_passes exports / names. "
        "Legacy fallback is disabled."
    ) from e

# Console encoding safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Logging
logging.basicConfig(
    level=logging.DEBUG if cfg.DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_issue_catalog(path: Path) -> dict:
    """
    Load the issue catalog from JSON.

    Returns a dict with canonical keys:
      - 'items': unified list of catalog entries (each has 'id' and 'kind' field)
      - 'trade_buckets': list of trade bucket definitions
      - 'version': catalog version string
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Catalog file loaded: {path.resolve()} raw_keys={list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
    except FileNotFoundError:
        logger.warning(f"Issue catalog not found at {path}; using empty catalog.")
        data = {}
    except Exception as exc:
        logger.error(f"Failed to load issue catalog from {path}: {exc}")
        data = {}

    # Normalize trade_buckets: v3 expects a list, tolerate older dict shapes
    tb = data.get("trade_buckets", [])
    if isinstance(tb, dict):
        tb = list(tb.values())

    return {
        # v3 canonical: single "items" array (each has 'id' and 'kind' field)
        "items": data.get("items", []) or data.get("defects", []) or data.get("defect_issues", []),
        "trade_buckets": tb,
        "version": data.get("version"),
    }


def _log_catalog_load(path: Path, cat: dict):
    """Log catalog load details at INFO so they're always visible."""
    items = cat.get("items", []) or []
    version = cat.get("version")
    kind_counts = {}
    for d in items:
        k = d.get("kind", "defect") if isinstance(d, dict) else "?"
        kind_counts[k] = kind_counts.get(k, 0) + 1
    logger.info(
        f"Catalog load: path={path.resolve()} exists={path.exists()} "
        f"version={version} items={len(items)} "
        f"kinds={kind_counts} "
        f"keys={list(cat.keys())}"
    )
    sample = items[:2]
    if sample:
        logger.info(f"Catalog sample items[0:2]={[{k: v for k, v in s.items() if k in ('id','name','kind','trade_bucket')} for s in sample if isinstance(s, dict)]}")
    else:
        logger.info("Catalog sample: items is EMPTY")


# Compute default catalog path - prefer tools/issue_catalog.json if it exists
_repo_root = Path(__file__).resolve().parent.parent
_tools_dir = getattr(cfg, "TOOLS_DIR", Path(__file__).resolve().parent)
_tools_catalog = Path(_tools_dir) / "issue_catalog.json"
_root_catalog = (getattr(cfg, "PROJECT_ROOT", None) or _repo_root) / "issue_catalog.json"

# Prefer tools/ location, fallback to repo root
if _tools_catalog.exists():
    default_catalog_path = _tools_catalog
else:
    default_catalog_path = _root_catalog

ISSUE_CATALOG_PATH = Path(getattr(cfg, "ISSUE_CATALOG_PATH", default_catalog_path))

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

# ── Schema Version Constants ────────────────────────────────────────────────────
PHOTO_INTEL_SCHEMA_VERSION = "photo_intel_v2"
PROPERTY_SUMMARY_SCHEMA_VERSION = "property_summary_v3"
NORMALIZATION_POLICY_VERSION = "workitem_v1"


def _stable_hash_id(*parts: str, length: int = 12) -> str:
    """Generate a stable, deterministic short hash ID from input parts."""
    import hashlib
    # Include all parts, normalize None to empty string (don't drop falsy values)
    combined = "|".join(str(p) if p is not None else "" for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:length]


def _make_photo_id(property_key: str, run_id: str, photo_key: str) -> str:
    """Generate deterministic photo ID."""
    return _stable_hash_id(property_key, run_id, photo_key, length=16)


def _make_issue_id(run_id: str, photo_key: str, description: str, location_hint: str, label: str,
                   ordinal: int = 0) -> str:
    """Generate deterministic issue ID. Ordinal handles duplicate issues in same photo."""
    return _stable_hash_id(run_id, photo_key, description, location_hint, label, str(ordinal), length=16)


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
        issue_catalog: Issue catalog to use
    """
    catalog = issue_catalog or {}

    issue_meta: Dict[str, Dict[str, str]] = {}
    for item in catalog.get("items", []) or []:
        if isinstance(item, dict):
            iid = item.get("id") or item.get("defect_id")
            if iid:
                issue_meta[iid] = {
                    "name": item.get("name", iid),
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

        # ✅ Instance-owned issue catalog (loaded at init, not module level)
        self.issue_catalog_path = ISSUE_CATALOG_PATH
        try:
            self.issue_catalog = load_issue_catalog(self.issue_catalog_path) or {}
        except Exception as e:
            logger.error(f"Failed to load issue catalog at {self.issue_catalog_path}: {e}")
            self.issue_catalog = {}

        # Helpful sanity logging
        _log_catalog_load(self.issue_catalog_path, self.issue_catalog)

        # ── Embeddings-based catalog matcher (optional) ─────────────────────
        self.catalog_matcher = None
        if getattr(cfg, "USE_EMBEDDINGS_CATALOG", False) and EMBEDDINGS_MATCHER_AVAILABLE:
            try:
                self.catalog_matcher = CatalogEmbeddingsRetriever(
                    catalog_v2=self.issue_catalog,  # or just self.issue_catalog as first positional arg
                    model_name=getattr(cfg, "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                    default_topk=getattr(cfg, "EMBEDDINGS_TOPK", 5),
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
                # ✅ Build candidate_provider from embeddings matcher or catalog
                candidate_provider = None
                if self.catalog_matcher:
                    candidate_provider = self.catalog_matcher.get_candidates if hasattr(self.catalog_matcher, 'get_candidates') else None
                    if candidate_provider:
                        logger.info("Candidate provider: embeddings-based (catalog_matcher.get_candidates)")

                if candidate_provider is None and self.issue_catalog:
                    # Simple catalog-based provider fallback
                    def _catalog_candidate_provider(query, rough_category=None, topk=8):
                        """Simple candidate provider from catalog items (kind=='defect' only)."""
                        candidates = [
                            item for item in (self.issue_catalog.get("items", []) or [])
                            if isinstance(item, dict) and item.get("kind", "defect") == "defect"
                        ]
                        logger.info(f"2d diag: catalog_candidate_pool={len(candidates)} (topk={topk}, query={query[:60]})")
                        return candidates[:topk]

                    candidate_provider = _catalog_candidate_provider
                    logger.info("Candidate provider: simple catalog-based fallback")

                # Wrap to normalize signature + output shape
                if candidate_provider is not None:
                    candidate_provider = self._wrap_candidate_provider(candidate_provider)

                # ✅ Give orchestrator candidate_provider (catalog flows through provider, not directly)
                self.orchestrator = create_orchestrator_from_config(
                    cfg,
                    candidate_provider=candidate_provider,
                )
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

    @staticmethod
    def _toggle(options, key: str, default: bool = True) -> bool:
        """Robustly read a pass toggle from run_options, handling dict/dataclass/object."""
        t = getattr(options, "toggles", None) if options else None
        if t is None:
            return default
        if isinstance(t, dict):
            return bool(t.get(key, default))
        # dataclass / object: try exact, then common naming patterns
        for name in (key, f"p{key}", f"_{key}", key.replace("-", "_")):
            if hasattr(t, name):
                return bool(getattr(t, name))
        return default

    @staticmethod
    def _normalize_candidate(item: dict) -> dict:
        """Normalize a raw catalog item into the shape run_pass_2d and orchestrator expect.

        Accepts both embeddings retriever output (has defect_id, score, trade_bucket)
        and raw catalog items (has id, aliases, category).
        """
        item_id = item.get("defect_id") or item.get("id") or item.get("key", "")
        return {
            "defect_id": item_id,  # run_pass_2d expects this field name
            "score": item.get("score", 1.0),
            "name": item.get("name") or item.get("label") or item_id,
            "category": item.get("category", ""),
            "aliases": item.get("aliases", []),
            "trade_bucket": item.get("trade_bucket", ""),
            "kind": item.get("kind", ""),
            "severity": item.get("severity", 0),
        }

    def _wrap_candidate_provider(self, raw_provider):
        """
        Wrap any candidate provider so it supports both call conventions
        and returns normalized candidate dicts.

        Supported underlying signatures:
          (query)                              - simple
          (query, rough_category=None, topk=8) - catalog fallback
          (query, context_dict)                - orchestrator convention

        Output: always returns list of {defect_id, score, name, ...}
        """
        def wrapped(query, context_or_category=None, topk=8):
            # Determine topk from context dict if present
            if isinstance(context_or_category, dict):
                topk = context_or_category.get("top_k_candidates", topk)

            # Try calling with just query first (safest)
            try:
                result = raw_provider(query)
            except TypeError:
                try:
                    result = raw_provider(query, None, topk)
                except TypeError:
                    result = raw_provider(query, context_or_category)

            if not isinstance(result, list):
                logger.info(f"2d diag: raw_provider_type={type(result).__name__} (not list)")
                return []

            logger.info(f"2d diag: raw_provider_len={len(result)} sample_keys={list(result[0].keys()) if result else []}")

            normalized = [self._normalize_candidate(item) for item in result[:topk]]
            logger.info(f"2d diag: normalized_len={len(normalized)} sample={normalized[0] if normalized else '{}'}")
            return normalized

        return wrapped

    async def _call_pass_2d(self, *, model_config, items, catalog_matcher=None, candidate_provider=None):
        """
        Run Pass 2d for each observation (defect OR upgrade), resolving to a canonical catalog item.

        For each item:
          - Read `kind` field (already set by 2c enrichment: "defect" or "opportunity")
          - Retrieve kind-filtered candidates via embeddings (preferred) or fallback provider
          - Call run_pass_2d to pick the best candidate

        Returns:
            namespace with .resolved_items list and .pass_2d_results list
        """
        resolved_items = []
        pass_2d_results = []

        for obs in items:
            description = obs.get("description", "") if isinstance(obs, dict) else str(obs)
            if not description:
                continue

            # Use `kind` set during 2c enrichment — this is the authoritative field
            kind_field = (obs.get("kind") or "").lower() if isinstance(obs, dict) else ""
            kind = "upgrade" if kind_field == "opportunity" else "defect"

            # Get kind-filtered candidates — prefer embeddings retriever
            candidates = []
            if catalog_matcher:
                try:
                    if kind == "upgrade":
                        raw_cands = catalog_matcher.embeddings_retrieve_upgrade_candidates(description, topk=8)
                    else:
                        raw_cands = catalog_matcher.embeddings_retrieve_defect_candidates(description, topk=8)
                    candidates = [
                        {
                            "defect_id": c.defect_id,
                            "name": c.name,
                            "kind": c.kind,
                            "trade_bucket": c.trade_bucket,
                            "severity": c.severity,
                            "score": float(c.score),
                        }
                        for c in raw_cands
                    ]
                except Exception as e:
                    logger.warning(f"2d diag: embeddings retrieval failed for {kind}: {e}")

            # Fallback to generic candidate provider (already normalized)
            if not candidates and candidate_provider:
                try:
                    candidates = candidate_provider(description)
                except TypeError:
                    try:
                        candidates = candidate_provider(description, None, 8)
                    except Exception:
                        candidates = []

            if not isinstance(candidates, list) or not candidates:
                logger.info(f"2d diag: no {kind} candidates for observation: {description[:80]}")
                continue

            # Call run_pass_2d with kind parameter
            try:
                pass_2d_result = await run_pass_2d(
                    vlm_client=self.vlm_client,
                    model_config=model_config,
                    observation=description,
                    candidates=candidates,
                    kind=kind,
                )
                pass_2d_results.append(pass_2d_result)

                # Extract resolved ID — tolerate both dict and dataclass
                if isinstance(pass_2d_result, dict):
                    rid = (pass_2d_result.get("resolved_item_id")
                           or pass_2d_result.get("resolved_defect_id")
                           or pass_2d_result.get("defect_id"))
                    raw = pass_2d_result.get("raw_response", "")
                else:
                    rid = (getattr(pass_2d_result, "resolved_item_id", None)
                           or getattr(pass_2d_result, "resolved_defect_id", None))
                    raw = getattr(pass_2d_result, "raw_response", "")

                top_candidate = candidates[0] if candidates else None
                resolved_items.append({
                    "description": description,
                    "label": obs.get("label", "") if isinstance(obs, dict) else "",
                    "kind": kind,
                    "resolved_item_id": rid,
                    "resolved_kind": kind,
                    "top_candidate_id": top_candidate.get("defect_id") if top_candidate else None,
                    "top_candidate_score": top_candidate.get("score") if top_candidate else None,
                    "top_candidate_trade_bucket": top_candidate.get("trade_bucket") if top_candidate else None,
                    "raw_response": raw,
                })
            except Exception as e:
                logger.warning(f"  Pass 2d failed for {kind} observation: {e}")

        # Return namespace-like object for backward compat
        class _Result:
            pass
        r = _Result()
        r.resolved_items = resolved_items
        r.resolved_defects = resolved_items  # legacy alias
        r.pass_2d_results = pass_2d_results
        return r

    def _get_model_config_for_pass(self, pass_key: str) -> Dict[str, Any]:
        """
        Get the appropriate model config for a specific pass.

        Args:
            pass_key: Pass identifier ('1a', '1b', '1c', '2a', '2b', '2c', '2d', '3', '4a', '4b', '4c')

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

        # Premium uses GPT for passes 1b, 2a, 2d, 4, 4a, 4b, 4c
        if self.analysis_profile == 'premium' and (
                pass_key in ('1b', '2a', '2d', '4', '4a', '4b', '4c') or pass_key.startswith('4')):
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
            '2c': ['GPT_PASS_2C_MODEL', 'OPENAI_PASS2C_MODEL'],
            '2d': ['GPT_PASS_2D_MODEL', 'OPENAI_PASS2D_MODEL'],
            '4': ['GPT_PASS_4_MODEL', 'OPENAI_PASS4_MODEL'],
            '4a': ['GPT_PASS_4A_MODEL', 'OPENAI_PASS4A_MODEL'],
            '4b': ['GPT_PASS_4B_MODEL', 'OPENAI_PASS4B_MODEL'],
            '4c': ['GPT_PASS_4C_MODEL', 'OPENAI_PASS4C_MODEL'],
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
            "2c": "OPENAI_PASS_2C_MAX_TOKENS",
            "2d": "OPENAI_PASS_2D_MAX_TOKENS",
            "4": "OPENAI_PASS_4_MAX_TOKENS",
            "4a": "OPENAI_PASS_4A_MAX_TOKENS",
            "4b": "OPENAI_PASS_4B_MAX_TOKENS",
            "4c": "OPENAI_PASS_4C_MAX_TOKENS",
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
            p = Path(path) if not isinstance(path, Path) else path
            if not p.exists():
                missing.append(f"{name}: {p}")

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
                'feature_notes': '',
                'positives_notes': '',  # legacy alias
                'observations_freeform': '',
                'keywords': [],
                'catalog_flags': {},
                'issues_natural_language': [],
                'verified_issues': [],
                'passes': {},  # Store per-pass outputs
                'models_used': {},  # Track which model was used for each pass
                'pass_timings': {},  # Track per-pass execution time in seconds
            }
            context = {}

            # Pass 1a: Scene Type (always Qwen)
            model_config_1a = self._get_model_config_for_pass('1a')
            logger.debug(f"  Pass 1a using model: {model_config_1a.get('model')}")
            results['models_used']['1a'] = model_config_1a.get('model')

            t0 = time.time()
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
                    'confidence': None,  # scene_confidence retired
                    'reasoning': pass_1a.reasoning,
                }
            except Exception as e:
                logger.warning(f"  Pass 1a failed: {e}")
            results['pass_timings']['1a'] = round(time.time() - t0, 3)

            # Pass 1b: Feature/Market Appeal notes (FREEFORM; GPT in premium)
            model_config_1b = self._get_model_config_for_pass('1b')
            logger.debug(f"  Pass 1b using model: {model_config_1b.get('model')}")
            results['models_used']['1b'] = model_config_1b.get('model')

            feature_notes = ""
            t0 = time.time()
            try:
                pass_1b = await run_pass_1b_feature_notes(
                    image_path=image_path,
                    vlm_client=self.vlm_client,
                    model_config=model_config_1b,
                    context=context,
                )
                feature_notes = pass_1b.feature_notes
                results['feature_notes'] = feature_notes
                results['positives_notes'] = feature_notes  # legacy alias

                # Store Pass 1b output
                results['passes']['1b'] = {
                    'feature_notes': feature_notes,
                    'positives_notes': feature_notes,  # legacy alias
                }
            except Exception as e:
                logger.warning(f"  Pass 1b failed: {e}")
            results['pass_timings']['1b'] = round(time.time() - t0, 3)

            # Pass 1c: Feature notes -> JSON structuring (text-only)
            model_config_1c = self._get_model_config_for_pass('1c')
            logger.debug(f"  Pass 1c using model: {model_config_1c.get('model')}")
            results['models_used']['1c'] = model_config_1c.get('model')

            t0 = time.time()
            try:
                pass_1c = await run_pass_1c_feature_structuring(
                    vlm_client=self.vlm_client,
                    model_config=model_config_1c,
                    feature_notes=feature_notes,
                )
                results['notable_features'] = pass_1c.notable_features or []
                results['overall_impression'] = getattr(pass_1c, 'overall_impression', '') or ''
                results['image_summary'] = getattr(pass_1c, 'image_summary', '') or ''
                context['notable_features'] = results['notable_features']

                # Store Pass 1c output
                results['passes']['1c'] = {
                    'overall_impression': results['overall_impression'],
                    'image_summary': results['image_summary'],
                    'notable_features': results['notable_features'],
                }
            except Exception as e:
                logger.warning(f"  Pass 1c failed: {e}")
            results['pass_timings']['1c'] = round(time.time() - t0, 3)

            # Pass 2a: Issue Detection - freeform notes (GPT in premium)
            model_config_2a = self._get_model_config_for_pass('2a')
            logger.debug(f"  Pass 2a using model: {model_config_2a.get('model')}")
            results['models_used']['2a'] = model_config_2a.get('model')

            obs_freeform = ""
            t0 = time.time()
            try:
                pass_2a = await run_pass_2a(
                    image_path=image_path,
                    vlm_client=self.vlm_client,
                    model_config=model_config_2a,
                    context=context,
                )
                obs_freeform = pass_2a.observations_freeform
                results['observations_freeform'] = obs_freeform

                # Store Pass 2a output
                results['passes']['2a'] = {
                    'observations_freeform': obs_freeform,
                }
            except Exception as e:
                logger.warning(f"  Pass 2a failed: {e}")
            results['pass_timings']['2a'] = round(time.time() - t0, 3)

            # Pass 2b: Freeform to JSON conversion (always Qwen)
            model_config_2b = self._get_model_config_for_pass('2b')
            logger.debug(f"  Pass 2b using model: {model_config_2b.get('model')}")
            results['models_used']['2b'] = model_config_2b.get('model')

            observations: List[Dict[str, Any]] = []
            t0 = time.time()
            try:
                pass_2b = await run_pass_2b(
                    vlm_client=self.vlm_client,
                    model_config=model_config_2b,
                    observations_freeform=obs_freeform,
                )
                observations = pass_2b.observations or []
                results['observations'] = observations

                # Store Pass 2b output
                results['passes']['2b'] = {
                    'observations': observations,
                }
            except Exception as e:
                logger.warning(f"  Pass 2b failed: {e}")
            results['pass_timings']['2b'] = round(time.time() - t0, 3)

            # Pass 2c: Issue classification + labeling (always Qwen, text-only)
            model_config_2c = self._get_model_config_for_pass('2c')
            logger.debug(f"  Pass 2c using model: {model_config_2c.get('model')}")
            results['models_used']['2c'] = model_config_2c.get('model')

            labeled_debug: List[Dict[str, Any]] = []
            labeled_forward: List[Dict[str, Any]] = []
            t0 = time.time()
            try:
                pass_2c = await run_pass_2c(
                    vlm_client=self.vlm_client,
                    model_config=model_config_2c,
                    observations=observations,
                )
                labeled_debug = pass_2c.labeled_debug or []
                labeled_forward = pass_2c.labeled_forward or []

                # Enrich forward items to keep downstream schema happy
                # Classify kind based on label: default to "defect" since 2a/2b/2c
                # is the issue detection chain. Known opportunity labels get "opportunity".
                _OPPORTUNITY_LABELS = {"opportunity", "upgrade", "improvement", "cosmetic_upgrade", "feature", "upgrade_candidate"}

                forward_enriched = [
                    {
                        "description": x.get("description", ""),
                        "label": x.get("label", ""),
                        "rough_category": "",  # deprecated — keep empty for schema compatibility
                        "location_hint": "",
                        "kind": "opportunity" if x.get("label", "").lower() in _OPPORTUNITY_LABELS else "defect",
                        "searchable": "yes",  # ✅ required for build_grouped_issues gate
                    }
                    for x in labeled_forward
                    if isinstance(x, dict) and x.get("description")
                ]

                results['labeled_debug'] = labeled_debug
                results['labeled_forward'] = forward_enriched
                results['catalog_flags'] = {}  # embeddings owns this, not passes

                # Bridge to legacy fields for downstream compatibility
                results['issues_natural_language'] = forward_enriched
                results['verified_issues'] = forward_enriched

                # Store Pass 2c output
                results['passes']['2c'] = {
                    'labeled_debug': labeled_debug,
                    'labeled_forward': forward_enriched,
                    'verified_issues': forward_enriched,           # ✅ downstream reads this
                    'issues_natural_language': forward_enriched,   # ✅ fallback key for compatibility
                }
            except Exception as e:
                logger.warning(f"  Pass 2c failed: {e}")
                # Fallback: bridge observations directly
                fallback_issues = [
                    {"description": x.get("description", ""), "label": "", "rough_category": "", "location_hint": "", "kind": "defect", "searchable": "yes"}
                    for x in observations if isinstance(x, dict) and x.get("description")
                ]
                results['labeled_debug'] = []
                results['labeled_forward'] = fallback_issues
                results['issues_natural_language'] = fallback_issues
                results['verified_issues'] = fallback_issues
            results['pass_timings']['2c'] = round(time.time() - t0, 3)

            # Pass 2d: Catalog Resolution (optional, defects + upgrades)
            # Gating: run if toggles allow and there are resolvable items
            toggle_2d = self._toggle(self.run_options, "2d", default=True)

            all_forward = results.get('labeled_forward', []) or []
            # Gate on `kind` (the semantic field we control) rather than `label` (LLM output)
            # Exclude safety/good_condition/generic_presence — only defects and upgrades are resolvable
            resolvable_items = [
                x for x in all_forward
                if isinstance(x, dict)
                and x.get("kind") in {"defect", "opportunity"}
                and x.get("description")
            ]
            has_items = len(resolvable_items) > 0

            # 2d needs either embeddings matcher or a fallback candidate provider
            has_resolver = self.catalog_matcher is not None or bool(self.issue_catalog)

            # Build fallback candidate provider (only used when embeddings unavailable)
            fallback_candidate_provider = None
            if not self.catalog_matcher and self.issue_catalog:
                def _simple_candidates(query, rough_category=None, topk=8):
                    items_pool = [
                        item for item in (self.issue_catalog.get("items", []) or [])
                        if isinstance(item, dict)
                    ]
                    logger.info(f"2d diag: simple_candidate_pool={len(items_pool)} (topk={topk}, query={query[:60]})")
                    return items_pool[:topk]
                fallback_candidate_provider = self._wrap_candidate_provider(_simple_candidates)

            # Debug gate status
            catalog_item_count = len(self.issue_catalog.get("items", []) or []) if self.issue_catalog else 0
            defect_count = len([x for x in resolvable_items if x.get("kind") == "defect"])
            upgrade_count = len([x for x in resolvable_items if x.get("kind") == "opportunity"])
            logger.info(
                f"  Pass 2d gate: toggle_2d={toggle_2d}, has_items={has_items} "
                f"(defects={defect_count}, upgrades={upgrade_count}, total={len(all_forward)}), "
                f"has_resolver={has_resolver}, catalog_items={catalog_item_count}"
            )

            if toggle_2d and has_items and has_resolver:
                model_config_2d = self._get_model_config_for_pass('2d')
                logger.info(f"  Pass 2d using model: {model_config_2d.get('model')}")
                results['models_used']['2d'] = model_config_2d.get('model')

                t0 = time.time()
                try:
                    pass_2d = await self._call_pass_2d(
                        model_config=model_config_2d,
                        items=resolvable_items,
                        catalog_matcher=self.catalog_matcher,
                        candidate_provider=fallback_candidate_provider,
                    )
                    resolved = pass_2d.resolved_items if hasattr(pass_2d, 'resolved_items') else (pass_2d.resolved_defects if hasattr(pass_2d, 'resolved_defects') else [])
                    resolved = resolved or []
                    results['resolved_defects'] = resolved  # legacy key name kept

                    results['passes']['2d'] = {
                        'resolutions': resolved,
                        'resolved_defects': resolved,  # legacy alias
                        'resolved_items': resolved,
                    }
                    logger.info(f"  Pass 2d completed: {len(resolved)} resolutions")

                    # ── Join 2d results back into verified_issues ──
                    # Build catalog item lookup for deterministic trade bucket resolution
                    catalog_items_by_id: Dict[str, Dict[str, Any]] = {}
                    for cat_item in (self.issue_catalog.get("items", []) or []):
                        if isinstance(cat_item, dict):
                            cid = cat_item.get("defect_id") or cat_item.get("upgrade_id") or cat_item.get("id") or ""
                            if cid:
                                catalog_items_by_id[cid] = cat_item

                    bucket_name_map = self._build_trade_bucket_name_map()

                    # Index resolutions by description for joining
                    resolution_by_desc: Dict[str, Dict[str, Any]] = {}
                    for res in resolved:
                        if isinstance(res, dict) and res.get("description"):
                            resolution_by_desc[res["description"]] = res

                    verified = results.get('verified_issues', []) or []
                    for issue in verified:
                        if not isinstance(issue, dict):
                            continue
                        desc = issue.get("description", "")
                        res = resolution_by_desc.get(desc)
                        if res and res.get("resolved_item_id"):
                            rid = res["resolved_item_id"]
                            cat_entry = catalog_items_by_id.get(rid, {})
                            tb_id = cat_entry.get("trade_bucket", "")

                            issue["catalogItemId"] = rid
                            issue["catalogItemName"] = cat_entry.get("name", rid)
                            issue["catalogItemKind"] = res.get("resolved_kind", res.get("kind", ""))
                            issue["tradeBucketId"] = tb_id
                            issue["tradeBucketName"] = bucket_name_map.get(tb_id, tb_id)
                            issue["topCandidateScore"] = res.get("top_candidate_score")
                        elif not issue.get("catalogItemId"):
                            # No resolution — null out fields for consistent schema
                            issue.setdefault("catalogItemId", None)
                            issue.setdefault("catalogItemName", None)
                            issue.setdefault("catalogItemKind", None)
                            issue.setdefault("tradeBucketId", None)
                            issue.setdefault("tradeBucketName", None)
                            issue.setdefault("topCandidateScore", None)

                    # Sync all downstream fields
                    results['verified_issues'] = verified
                    results['issues_natural_language'] = verified
                    if isinstance(results.get('passes', {}).get('2c'), dict):
                        results['passes']['2c']['verified_issues'] = verified
                        results['passes']['2c']['issues_natural_language'] = verified

                except Exception as e:
                    logger.warning(f"  Pass 2d failed: {e}")
                    results['passes']['2d'] = {'resolutions': [], 'resolved_defects': [], 'resolved_items': [], 'error': str(e)}
                    results['resolved_defects'] = []
                    # Fallback: use embeddings-only annotation when 2d fails
                    verified = results.get('verified_issues', []) or []
                    verified = self._annotate_verified_issues_with_embeddings(verified)
                    results['verified_issues'] = verified
                    results['issues_natural_language'] = verified
                results['pass_timings']['2d'] = round(time.time() - t0, 3)
            else:
                skip_reasons = []
                if not toggle_2d:
                    skip_reasons.append("toggle off")
                if not has_items:
                    skip_reasons.append("no resolvable items in labeled_forward")
                if not has_resolver:
                    skip_reasons.append("no catalog matcher or catalog")
                logger.info(f"  Pass 2d skipped: {', '.join(skip_reasons)}")
                results['passes']['2d'] = {'resolutions': [], 'skipped': True, 'skip_reasons': skip_reasons}
                results['resolved_defects'] = []

                # Fallback: use embeddings-only annotation when 2d is skipped
                verified = results.get('verified_issues', []) or []
                verified = self._annotate_verified_issues_with_embeddings(verified)
                results['verified_issues'] = verified
                results['issues_natural_language'] = verified

            # Pass 3: Keyword Extraction (text-only, always Qwen)
            model_config_3 = self._get_model_config_for_pass('3')
            logger.debug(f"  Pass 3 using model: {model_config_3.get('model')}")
            results['models_used']['3'] = model_config_3.get('model')

            t0 = time.time()
            try:
                # Pass 3 context must match new Pass 3 prompt expectations
                context["scene"] = results.get("scene", "property")
                context["features_struct"] = {
                    "overall_impression": results.get("overall_impression", ""),
                    "image_summary": results.get("image_summary", ""),
                    "notable_features": results.get("notable_features", []),
                }
                context["observations"] = observations
                context["labeled_forward"] = results.get("labeled_forward", [])

                pass_3 = await run_pass_3_keyword_extraction(
                    vlm_client=self.vlm_client,
                    model_config=model_config_3,
                    context=context,
                    max_keywords=self.max_keywords,
                )
                results['keywords'] = pass_3.keywords
                results['keyword_categories'] = pass_3.keyword_categories

                # Store Pass 3 output
                results['passes']['3'] = {
                    'keywords': pass_3.keywords,
                    'categories': pass_3.keyword_categories,
                }
            except Exception as e:
                logger.warning(f"  Pass 3 failed: {e}")
            results['pass_timings']['3'] = round(time.time() - t0, 3)

            # Compute total LLM time
            results['total_pass_time'] = round(sum(results['pass_timings'].values()), 3)

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
        confidence = None  # scene_confidence retired
        if hasattr(result, 'pass_1a') and result.pass_1a:
            reasoning = result.pass_1a.reasoning or ""

        keywords = data.get("keywords", []) or []

        # Robust category extraction - try data first, fallback to pass_3 attribute
        kw_cats = data.get("keyword_categories")
        if kw_cats is None and getattr(result, "pass_3", None):
            kw_cats = getattr(result.pass_3, "keyword_categories", None)

        # Build grounding prompt from keywords
        prompt = ". ".join(keywords) + "." if keywords else ""

        # Extract targets if available
        targets = []
        planner_hints: Dict[str, str] = {}

        # Build scene payload for backwards compatibility
        scene_payload = self._scene_classifier_payload(data, scene_override=scene)
        scene_payload['keywords'] = keywords
        scene_payload['keyword_categories'] = kw_cats  # flat copy for convenience
        scene_payload['overall_impression'] = data.get('overall_impression', '')
        scene_payload['image_summary'] = data.get('image_summary', '')
        scene_payload['notable_features'] = data.get('notable_features', []) or []
        scene_payload['feature_notes'] = data.get('feature_notes', '') or data.get('positives_notes', '') or ""
        scene_payload['positives_notes'] = scene_payload['feature_notes']  # legacy alias
        scene_payload['observations_freeform'] = data.get('observations_freeform', '') or ""
        scene_payload['groundingdino_prompt'] = prompt
        scene_payload['catalog_flags'] = data.get('catalog_flags', {})
        scene_payload['issues_natural_language'] = data.get('issues_natural_language', [])
        scene_payload['verified_issues'] = data.get('verified_issues', [])
        scene_payload['models_used'] = data.get('models_used', {})

        # Copy v2 fields through from orchestrator
        scene_payload["observations_freeform"] = data.get("observations_freeform", "")
        scene_payload["features_struct"] = data.get("features_struct", {}) or {}
        scene_payload["observations_struct"] = data.get("observations_struct", {}) or {}
        scene_payload["labeled_debug"] = data.get("labeled_debug", []) or []
        scene_payload["labeled_forward"] = data.get("labeled_forward", []) or []
        scene_payload["resolved_defects"] = data.get("resolved_defects", []) or []

        # Keep meta
        scene_payload["passes_run"] = data.get("passes_run", []) or []
        scene_payload["pass_timings"] = data.get("pass_timings", {}) or {}
        scene_payload["total_pass_time"] = data.get("total_pass_time", 0.0) or 0.0

        # TEMP bridge: treat labeled_forward as "issues" so legacy paths still function
        if scene_payload["labeled_forward"] and not scene_payload.get("issues_natural_language"):
            scene_payload["issues_natural_language"] = [
                {
                    "description": x.get("description", ""),
                    "rough_category": "",  # deprecated
                    "location_hint": "",
                    "label": x.get("label", ""),
                }
                for x in (scene_payload["labeled_forward"] or [])
                if isinstance(x, dict) and x.get("description")
            ]
            scene_payload["verified_issues"] = scene_payload["issues_natural_language"]

        # Build passes dict for consistent schema - preserve orchestrator passes if present
        existing_passes = data.get("passes")
        if isinstance(existing_passes, dict) and existing_passes:
            scene_payload["passes"] = existing_passes
        else:
            # Fallback only if orchestrator didn't provide passes
            scene_payload['passes'] = {
                "1a": {
                    "scene": scene,
                    "confidence": confidence,
                    "reasoning": reasoning,
                },
                "1b": {
                    "feature_notes": scene_payload.get("feature_notes", ""),
                    "positives_notes": scene_payload.get("positives_notes", ""),  # legacy alias
                },
                "1c": {
                    "overall_impression": scene_payload.get("overall_impression", ""),
                    "image_summary": scene_payload.get("image_summary", ""),
                    "notable_features": scene_payload.get("notable_features", []),
                },
                "2a": {"observations_freeform": scene_payload.get("observations_freeform", "")},
                "2b": {"observations": scene_payload.get("observations_struct", {}).get("observations", [])},
                "2c": {
                    "labeled_debug": scene_payload.get("labeled_debug", []),
                    "labeled_forward": scene_payload.get("labeled_forward", []),
                },
                "2d": {"resolutions": scene_payload.get("resolved_defects", [])},
                "3": {
                    "keywords": keywords,
                    "categories": kw_cats,
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
        scene_payload['feature_notes'] = results.get('feature_notes', '') or results.get('positives_notes', '') or ""
        scene_payload['positives_notes'] = scene_payload['feature_notes']  # legacy alias
        scene_payload['observations_freeform'] = results.get('observations_freeform', '') or ""
        scene_payload['keyword_categories'] = results.get('keyword_categories')
        scene_payload['groundingdino_prompt'] = prompt

        # Include passes dict if present
        if results.get('passes'):
            scene_payload['passes'] = results['passes']

        # Include models_used if present
        if results.get('models_used'):
            scene_payload['models_used'] = results['models_used']

        # ✅ Forward 2d resolved_defects and pass timings
        scene_payload['resolved_defects'] = results.get('resolved_defects', []) or []
        scene_payload['pass_timings'] = results.get('pass_timings', {}) or {}
        scene_payload['total_pass_time'] = results.get('total_pass_time', 0.0) or 0.0

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

            # Update all count fields unconditionally to match filtered detections
            final_count = len(detections)
            for k in ("count", "num_detections", "detections_count", "detection_count"):
                detection_result[k] = final_count

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

        # ✅ Per-job file handler so UI can always read run.log from artifacts
        job_log_handler = None
        try:
            log_path = job_dir / "run.log"
            job_log_handler = logging.FileHandler(log_path, encoding="utf-8")
            job_log_handler.setLevel(logging.DEBUG if self.debug else logging.INFO)
            job_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logging.getLogger().addHandler(job_log_handler)
            logger.info(f"Logging to: {log_path}")
        except Exception as e:
            logger.warning(f"Could not set up per-job log file: {e}")

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

        # Log live catalog state at run-start (catches mutation between init and run)
        live_items = len((self.issue_catalog.get("items", []) or [])) if self.issue_catalog else 0
        logger.info(f"Catalog live at run-start: items={live_items}, catalog_id={id(self.issue_catalog)}")

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

        # ✅ Remove per-job file handler to avoid leaking across jobs
        if job_log_handler:
            try:
                logging.getLogger().removeHandler(job_log_handler)
                job_log_handler.close()
            except Exception:
                pass

        return job

    def _build_trade_bucket_name_map(self) -> Dict[str, str]:
        """Build a {bucket_id -> bucket_name} lookup from the issue catalog."""
        bucket_map: Dict[str, str] = {}
        if not self.issue_catalog:
            return bucket_map
        for tb in self.issue_catalog.get("trade_buckets", []):
            if isinstance(tb, dict):
                bid = tb.get("id") or tb.get("bucket_id") or ""
                bname = tb.get("name") or tb.get("label") or bid
                if bid:
                    bucket_map[bid] = bname
        return bucket_map

    def _annotate_verified_issues_with_embeddings(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Annotate each verified issue with trade-bucket and catalog-match info
        using the embeddings retriever.

        For defect-labeled issues  → retrieve defect candidates
        For upgrade-labeled issues → retrieve upgrade candidates
        For unknown labels         → search both, take top match

        Adds to each issue dict:
          tradeBucketId, tradeBucketName, catalogItemId, catalogItemName,
          catalogItemKind, topCandidateScore
        """
        if not self.catalog_matcher or not issues:
            return issues

        bucket_map = self._build_trade_bucket_name_map()

        for issue in issues:
            if not isinstance(issue, dict):
                continue

            desc = issue.get("description", "")
            if not desc:
                continue

            # Use `kind` field (set during 2c enrichment) as the authoritative routing
            kind_field = (issue.get("kind") or "").lower()

            # Decide which candidate pool to search
            candidates = []
            if kind_field == "opportunity":
                candidates = self.catalog_matcher.embeddings_retrieve_upgrade_candidates(desc, topk=1)
                matched_kind = "upgrade"
            elif kind_field == "defect":
                candidates = self.catalog_matcher.embeddings_retrieve_defect_candidates(desc, topk=1)
                matched_kind = "defect"
            else:
                # No kind set (e.g. fallback items) — search both, take best
                defect_cands = self.catalog_matcher.embeddings_retrieve_defect_candidates(desc, topk=1)
                upgrade_cands = self.catalog_matcher.embeddings_retrieve_upgrade_candidates(desc, topk=1)
                d_score = defect_cands[0].score if defect_cands else -1
                u_score = upgrade_cands[0].score if upgrade_cands else -1
                if u_score > d_score:
                    candidates = upgrade_cands
                    matched_kind = "upgrade"
                else:
                    candidates = defect_cands
                    matched_kind = "defect"

            if candidates:
                top = candidates[0]
                tb_id = top.trade_bucket or ""
                issue["tradeBucketId"] = tb_id
                issue["tradeBucketName"] = bucket_map.get(tb_id, tb_id)
                issue["catalogItemId"] = top.defect_id  # upgrade_id in practice for upgrades
                issue["catalogItemName"] = top.name
                issue["catalogItemKind"] = matched_kind
                issue["topCandidateScore"] = round(top.score, 4)
            else:
                issue["tradeBucketId"] = None
                issue["tradeBucketName"] = None
                issue["catalogItemId"] = None
                issue["catalogItemName"] = None
                issue["catalogItemKind"] = None
                issue["topCandidateScore"] = None

        return issues

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
        payload.setdefault("scene_confidence", None)  # retired but keep for schema stability
        payload.setdefault("is_staged", None)
        payload.setdefault("overall_impression", "")
        payload.setdefault("image_summary", "")
        payload.setdefault("notable_features", [])
        payload.setdefault("feature_notes", "")
        payload.setdefault("positives_notes", "")  # legacy alias
        payload.setdefault("observations_freeform", "")
        payload.setdefault("reasoning", "" if error is None else error)
        payload.setdefault("targets", [])
        payload.setdefault("gdino_terms", [])
        payload.setdefault("keywords", [])
        payload.setdefault("keyword_categories", None)
        payload.setdefault("groundingdino_prompt", "")
        payload.setdefault("issues_natural_language", [])
        payload.setdefault("verified_issues", [])
        payload.setdefault("catalog_flags", {})
        payload.setdefault("issue_visual_anchors", [])
        payload.setdefault("processing_time", payload.get("processing_time"))
        payload.setdefault("pass_timings", payload.get("pass_timings", {}))
        payload.setdefault("total_pass_time", payload.get("total_pass_time", 0.0))
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
        issues_flat: List[Dict[str, Any]] = []  # Flat index of all issues

        for photo_index, res in enumerate(job.results, start=1):
            image_key = Path(res.image_path).name
            payload = self._scene_classifier_payload(res.scene_classifier or res.scene_data)

            # Use passes from payload if present (from _classify_scene_direct),
            # otherwise reconstruct for backwards compatibility (legacy runs)
            passes = payload.get("passes", None)
            if passes is None:
                # Fallback: reconstruct passes from flat fields (legacy runs only)
                feat_notes = payload.get("feature_notes", "") or payload.get("positives_notes", "")
                passes = {
                    "1a": {
                        "scene": payload.get("scene", "unknown"),
                        "confidence": None,  # scene_confidence retired
                        "reasoning": payload.get("reasoning", ""),
                    },
                    "1b": {
                        "feature_notes": feat_notes,
                        "positives_notes": feat_notes,  # legacy alias
                    },
                    "1c": {
                        "overall_impression": payload.get("overall_impression", ""),
                        "image_summary": payload.get("image_summary", ""),
                        "notable_features": payload.get("notable_features", []) or [],
                    },
                    "2a": {"observations_freeform": payload.get("observations_freeform", "")},
                    "2b": {
                        "issues_natural_language": payload.get("issues_natural_language", []) or [],
                        "catalog_flags": {},  # canonical: embeddings owns catalog_flags, not 2b
                    },
                    "3": {
                        "keywords": payload.get("keywords", []) or [],
                        "categories": payload.get("keyword_categories"),
                    },
                }

            # Generate stable photo ID
            photo_id = _make_photo_id(job.property_key, job.job_id, image_key)
            scene = payload.get("scene", res.scene) or "unknown"
            scene_group = SCENE_TO_GROUP_UI.get(scene, "other")
            scene_confidence = (passes.get("1a", {}) or {}).get("confidence")

            # Backfill issues with stable IDs and source linkage
            # Track signature counts to handle duplicate issues in same photo
            issue_sig_counts: Dict[Tuple[str, str, str], int] = {}

            # Prefer verified_issues (post-2c searchable=yes gate)
            issues_nl = _safe_list((passes.get("2c", {}) or {}).get("verified_issues"))
            if not issues_nl:
                issues_nl = _safe_list(payload.get("verified_issues"))
            if not issues_nl:
                # Fallback to 2b if 2c not available
                issues_nl = _safe_list((passes.get("2b", {}) or {}).get("issues_natural_language"))

            for issue in issues_nl:
                if isinstance(issue, dict) and issue.get("description"):
                    # Compute signature for ordinal tracking
                    sig = (
                        issue.get("description", ""),
                        issue.get("location_hint", ""),
                        issue.get("label", ""),
                    )
                    ordinal = issue_sig_counts.get(sig, 0)
                    issue_sig_counts[sig] = ordinal + 1

                    # Generate stable issue ID with ordinal to handle duplicates
                    issue_id = _make_issue_id(
                        job.job_id,
                        image_key,
                        sig[0],  # description
                        sig[1],  # location_hint
                        sig[2],  # label
                        ordinal,
                    )
                    # Add identity and context fields
                    issue["issue_id"] = issue_id
                    issue["source_photo_key"] = image_key
                    issue["source_photo_id"] = photo_id
                    issue["scene"] = scene
                    issue["scene_group"] = scene_group
                    # Placeholders for future normalization (Phase B)
                    issue.setdefault("trade", None)
                    issue.setdefault("severity", None)
                    issue.setdefault("confidence", None)
                    issue.setdefault("fix_code", None)

                    # Add to flat index (include placeholders for unified shape)
                    issues_flat.append({
                        "issue_id": issue_id,
                        "photo_id": photo_id,
                        "photo_key": image_key,
                        "scene": scene,
                        "scene_group": scene_group,
                        "description": issue.get("description", ""),
                        "label": issue.get("label", ""),
                        "location_hint": issue.get("location_hint", ""),
                        # Phase B placeholders
                        "trade": None,
                        "severity": None,
                        "confidence": None,
                        "fix_code": None,
                    })

            # Keep flat fields for backwards compatibility, but UI should read passes
            # Put **payload first so our explicit v2 fields override any collisions
            photos[image_key] = {
                **payload,
                "image_path": res.image_path,
                "scene": scene,
                # v2 lifted fields (override payload if present)
                "photo_id": photo_id,
                "photo_key": image_key,
                "photo_index": photo_index,
                "scene_confidence": scene_confidence,
                "scene_group": scene_group,
                "passes": passes,
                "pass_timings": payload.get("pass_timings", {}),
                "total_pass_time": payload.get("total_pass_time", 0.0),
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

            # Positives (prefer feature_notes, fallback to positives_notes for legacy)
            pass_1b = (passes.get("1b", {}) or {})
            feat_notes = (pass_1b.get("feature_notes") or pass_1b.get("positives_notes") or "").strip()
            if feat_notes:
                g["positives"]["notes"].append(feat_notes)

            nf = _safe_list((passes.get("1c", {}) or {}).get("notable_features"))
            for feat in nf:
                s = str(feat).strip()
                if s and s not in g["positives"]["notable_features"]:
                    g["positives"]["notable_features"].append(s)

            # Issues (prefer verified_issues from 2c, fallback to 2b)
            obs_freeform = ((passes.get("2a", {}) or {}).get("observations_freeform") or "").strip()
            if obs_freeform:
                g["issues"]["notes"].append(obs_freeform)

            # Prefer verified_issues (post-2c searchable=yes gate)
            issues = _safe_list((passes.get("2c", {}) or {}).get("verified_issues"))
            if not issues:
                issues = _safe_list(p.get("verified_issues"))
            if not issues:
                # Fallback to 2b if 2c not available
                issues = _safe_list((passes.get("2b", {}) or {}).get("issues_natural_language"))

            for it in issues:
                if isinstance(it, dict) and it.get("description"):
                    g["issues"]["issues_natural_language"].append({
                        "source_image": img_key,
                        "issue_id": it.get("issue_id"),
                        "photo_id": p.get("photo_id"),
                        "photo_key": img_key,
                        "description": it.get("description", ""),
                        "label": it.get("label", ""),
                        "location_hint": it.get("location_hint", ""),
                    })

        photo_intel = {
            # v2 schema versioning
            "artifact_schema_version": PHOTO_INTEL_SCHEMA_VERSION,
            "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
            # Existing fields
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
            # v2 flat issue index
            "issues_flat": issues_flat,
            "issues_flat_count": len(issues_flat),
        }

        try:
            photo_intel["renovation_needs"] = build_renovation_needs(job, issue_catalog=self.issue_catalog)
        except Exception as exc:
            logger.error(f"Failed to build renovation_needs: {exc}", exc_info=True)

        # ── Build defect events, work items, and search index ─────────────
        if DEFECT_EVENTS_AVAILABLE:
            try:
                defect_events = build_defect_events(
                    photos=photos,
                    catalog=self.issue_catalog,
                    run_id=job.job_id,
                )
                work_items = generate_work_items(defect_events, self.issue_catalog)
                search_index = build_search_index(defect_events, self.issue_catalog)

                photo_intel["defect_events"] = defect_events
                photo_intel["work_items"] = work_items
                photo_intel["search_index"] = search_index
            except Exception as exc:
                logger.error(f"Failed to build defect events layer: {exc}", exc_info=True)
                photo_intel["defect_events"] = []
                photo_intel["work_items"] = []
                photo_intel["search_index"] = {}
        else:
            photo_intel["defect_events"] = []
            photo_intel["work_items"] = []
            photo_intel["search_index"] = {}

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

        # Initialize with defaults
        error_msg = None
        model_used = ""
        scene_counts: Dict[str, int] = {}

        # Initialize Pass 4a outputs with defaults
        room_summaries: Dict[str, Any] = {}
        issues_by_category: Dict[str, int] = {}
        total_issues_found = 0

        # Pass 4b fields (renovation intel)
        room_scopes: Dict[str, str] = {}
        room_work_items: Dict[str, List[str]] = {}
        top_work_items: List[str] = []

        # Pass 4c UI card fields
        overall_condition = ""
        overall_summary = ""
        investment_verdict = ""
        investment_rationale = ""
        renovation_scope = ""
        renovation_priorities: List[str] = []
        risk_flags: List[str] = []
        deferred_maintenance: List[str] = []

        errors: Dict[str, str] = {}

        logger.info(
            f"Pass4 gate: VLM_CLIENT_AVAILABLE={VLM_CLIENT_AVAILABLE} "
            f"vlm_client={bool(self.vlm_client)} "
            f"PASS_ARCHITECTURE_AVAILABLE={PASS_ARCHITECTURE_AVAILABLE} "
            f"auto_analyzer_file={__file__}"
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # Aggregate from job.results and run Pass 4a/4b/4c
        # (Uses already-computed per-image results, does NOT re-run passes 1-3)
        # ═══════════════════════════════════════════════════════════════════════════
        if VLM_CLIENT_AVAILABLE and self.vlm_client and PASS_ARCHITECTURE_AVAILABLE:
            logger.info("  Aggregating from job.results for Pass 4a/4b/4c")

            # Build all_results from job (uses already-computed passes, no re-analysis)
            all_results = {}
            for res in job.results:
                image_key = Path(res.image_path).name
                payload = self._scene_classifier_payload(res.scene_classifier or res.scene_data)
                all_results[image_key] = payload
                # Count scenes
                scene = payload.get("scene", "unknown")
                scene_counts[scene] = scene_counts.get(scene, 0) + 1

            # Lazy import Pass 4 functions to prevent import-time failures
            try:
                from tools.scene_classifier_passes import (
                    run_pass_4a_room_summaries,
                    run_pass_4b_renovation_intel,
                    run_pass_4c_property_card_fields,
                    build_grouped_issues,
                    derive_property_scope,
                )
            except ImportError as e:
                logger.warning(f"Pass 4 imports missing, skipping property summary: {e}")
                error_msg = f"Pass 4 imports unavailable: {e}"
                # Fall through to write empty summary
            else:
                # Compute grouped_issues ONCE from all_results (single source of truth)
                # build_grouped_issues filters to searchable=yes issues only
                grouped_issues, _fallback_count = build_grouped_issues(all_results)

                # Compute deterministic totals from verified issues
                total_images_analyzed = len(all_results)
                total_issues_found = sum(len(issues) for issues in grouped_issues.values())
                issues_by_category = {}
                for group_issues in grouped_issues.values():
                    for issue in group_issues:
                        cat = issue.get("label", "general") or "general"
                        issues_by_category[cat] = issues_by_category.get(cat, 0) + 1

                # --- Pass 4a (room summaries aggregation) ---
                try:
                    logger.info("  Using Pass 4a (run_pass_4a_room_summaries)")
                    model_config_4a = self._get_model_config_for_pass('4a')
                    model_used = model_config_4a.get('model', '')
                    logger.info(f"  Model: {model_used}")

                    async def run_pass4a():
                        return await run_pass_4a_room_summaries(
                            vlm_client=self.vlm_client,
                            model_config=model_config_4a,
                            grouped_issues=grouped_issues,
                            scene_counts=scene_counts,
                            total_images_analyzed=total_images_analyzed,
                            total_issues_found=total_issues_found,
                            issues_by_category=issues_by_category,
                        )

                    loop = asyncio.new_event_loop()
                    try:
                        pass_4a_result = loop.run_until_complete(run_pass4a())
                    finally:
                        loop.close()

                    room_summaries = pass_4a_result.room_summaries or {}
                    logger.info(f"  Pass 4a completed: {len(room_summaries)} room groups")

                except Exception as e:
                    errors["pass4a"] = f"Pass 4a failed: {e}"
                    logger.error(errors["pass4a"], exc_info=True)

                # --- Pass 4b (renovation intel: scopes + work items) ---
                try:
                    logger.info("  Using Pass 4b (run_pass_4b_renovation_intel)")
                    model_config_4b = self._get_model_config_for_pass('4b')
                    logger.info(f"  Model: {model_config_4b.get('model', '')}")

                    async def run_pass4b():
                        return await run_pass_4b_renovation_intel(
                            vlm_client=self.vlm_client,
                            model_config=model_config_4b,
                            grouped_issues=grouped_issues,
                        )

                    loop = asyncio.new_event_loop()
                    try:
                        pass_4b_result = loop.run_until_complete(run_pass4b())
                    finally:
                        loop.close()

                    room_scopes = pass_4b_result.room_scopes or {}
                    room_work_items = pass_4b_result.room_work_items or {}
                    top_work_items = pass_4b_result.top_work_items or []

                    # Derive property scope deterministically from room scopes
                    renovation_scope = derive_property_scope(room_scopes)

                    logger.info(f"  Pass 4b completed: scope={renovation_scope}, top_items={len(top_work_items)}")

                except Exception as e:
                    errors["pass4b"] = f"Pass 4b failed: {e}"
                    logger.error(errors["pass4b"], exc_info=True)

                # --- Pass 4c (property card fields for UI) ---
                try:
                    logger.info("  Using Pass 4c (run_pass_4c_property_card_fields)")
                    model_config_4c = self._get_model_config_for_pass('4c')
                    logger.info(f"  Model: {model_config_4c.get('model', '')}")

                    async def run_pass4c():
                        return await run_pass_4c_property_card_fields(
                            vlm_client=self.vlm_client,
                            model_config=model_config_4c,
                            room_summaries=room_summaries,
                            room_scopes=room_scopes,
                            room_work_items=room_work_items,
                            top_work_items=top_work_items,
                            total_issues_found=total_issues_found,
                            total_images_analyzed=total_images_analyzed,
                            issues_by_category=issues_by_category,
                            property_scope=renovation_scope,
                        )

                    loop = asyncio.new_event_loop()
                    try:
                        pass_4c_result = loop.run_until_complete(run_pass4c())
                    finally:
                        loop.close()

                    overall_condition = pass_4c_result.overall_condition or ""
                    overall_summary = pass_4c_result.overall_summary or ""
                    investment_verdict = pass_4c_result.investment_verdict or ""
                    investment_rationale = pass_4c_result.investment_rationale or ""
                    renovation_priorities = pass_4c_result.renovation_priorities or []
                    risk_flags = pass_4c_result.risk_flags or []
                    deferred_maintenance = pass_4c_result.deferred_maintenance or []

                    logger.info("  Pass 4c completed successfully")

                except Exception as e:
                    errors["pass4c"] = f"Pass 4c failed: {e}"
                    logger.error(errors["pass4c"], exc_info=True)

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
        # Pass 4a: room summaries, Pass 4b: renovation intel, Pass 4c: UI card fields
        summary_data = {
            # v3 schema versioning
            "artifact_schema_version": PROPERTY_SUMMARY_SCHEMA_VERSION,
            "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
            # Existing fields
            "property_key": job.property_key,
            "run_id": job.job_id,  # explicit run_id for consistency
            "job_id": job.job_id,
            "timestamp": job.timestamp,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "summary_version": "pass4_v3",
            "analysis_profile": self.analysis_profile,
            "scene_counts": scene_counts,  # for UI filters

            # Pass 4a fields (room summaries)
            "room_summaries": room_summaries,
            "issues_by_category": issues_by_category,
            "total_issues_found": total_issues_found,

            # Pass 4b fields (renovation intel)
            "room_scopes": room_scopes,
            "room_work_items": room_work_items,
            "top_work_items": top_work_items,
            "renovation_scope": renovation_scope,  # derived from room_scopes

            # Pass 4c fields (UI card)
            "overall_condition": overall_condition,
            "overall_summary": overall_summary,
            "investment_verdict": investment_verdict,
            "investment_rationale": investment_rationale,
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

        # Add defect events layer from photo_intel (already computed in save_photo_intel)
        if DEFECT_EVENTS_AVAILABLE and photo_intel_path and photo_intel_path.exists():
            try:
                with open(photo_intel_path, 'r', encoding='utf-8') as f:
                    pi = json.load(f)
                summary_data["defect_events"] = pi.get("defect_events", [])
                summary_data["work_items"] = pi.get("work_items", [])
                summary_data["search_index"] = pi.get("search_index", {})
            except Exception as exc:
                logger.warning(f"Could not load defect events from photo_intel: {exc}")
                summary_data["defect_events"] = []
                summary_data["work_items"] = []
                summary_data["search_index"] = {}
        else:
            summary_data["defect_events"] = []
            summary_data["work_items"] = []
            summary_data["search_index"] = {}

        # Write property_summary.json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Property summary saved to: {output_path}")

        # Embed Pass 4a/4b/4c output into photo_intel.json so UI has single payload
        if photo_intel_path and photo_intel_path.exists():
            try:
                with open(photo_intel_path, 'r', encoding='utf-8') as f:
                    photo_intel = json.load(f)

                photo_intel["property_pass4a"] = {
                    "room_summaries": room_summaries,
                    "issues_by_category": issues_by_category,
                    "total_issues_found": total_issues_found,
                    "scene_counts": scene_counts,
                    "total_images_analyzed": len(job.results),
                    "error": errors.get("pass4a"),
                }

                photo_intel["property_pass4b"] = {
                    "room_scopes": room_scopes,
                    "room_work_items": room_work_items,
                    "top_work_items": top_work_items,
                    "renovation_scope": renovation_scope,
                    "error": errors.get("pass4b"),
                }

                photo_intel["property_pass4c"] = {
                    "overall_condition": overall_condition,
                    "overall_summary": overall_summary,
                    "investment_verdict": investment_verdict,
                    "investment_rationale": investment_rationale,
                    "renovation_priorities": renovation_priorities,
                    "risk_flags": risk_flags,
                    "deferred_maintenance": deferred_maintenance,
                    "error": errors.get("pass4c"),
                }

                # Also embed full summary for convenience
                photo_intel["property_summary"] = summary_data

                with open(photo_intel_path, "w", encoding="utf-8") as f:
                    json.dump(photo_intel, f, indent=2, ensure_ascii=False)

                logger.info(f"Pass 4a/4b/4c outputs embedded into: {photo_intel_path}")
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