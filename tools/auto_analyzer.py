#!/usr/bin/env python3
"""
Auto-Analyzer for GroundingDINO Pipeline
-----------------------------------------
Orchestrates: Scene Classification -> Detection -> Verification -> Property Summary
Uses settings from pipeline_config.py

Supports multiple detection backends:
- groundingdino (default): Local GroundingDINO inference
- dinox: DINO-X API or local script

Supports analysis profiles:
- standard: All passes use local Qwen via LM Studio
- premium: Key passes (1b, 2a, 2c, 4) use GPT-5
"""

import os
import sys
import json
import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Ensure repo root is in sys.path BEFORE any tools.* imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import configuration
try:
    from tools import pipeline_config as cfg
except ImportError:
    print("ERROR: pipeline_config.py not found!")
    sys.exit(1)

# --- Extracted modules ---
from tools.pipeline_common import (
    SCENE_GROUPS_UI,
    SCENE_TO_GROUP_UI,
)
from tools.scene_classifier_service import (
    SceneClassifierService,
    scene_classifier_payload,
)
from tools.detection_pipeline import run_detection_stage
from tools.artifact_writers import (
    load_issue_catalog,
    log_catalog_load,
    write_photo_intel,
)

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
        run_pass_2e,
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

# Valid detection backends (canonical names)
VALID_BACKENDS = {"groundingdino", "dinox"}

# Aliases for user convenience
BACKEND_ALIASES = {
    "grounding-dino": "groundingdino",
    "dino-x": "dinox",
}


# -- Data Classes --------------------------------------------------------------
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


# -- Auto-Analyzer Class -------------------------------------------------------
class AutoAnalyzer:
    def __init__(self,
                 python_exe: str = sys.executable,
                 artifacts_root: str = None,
                 box_threshold: float = None,
                 text_threshold: float = None,
                 chip_margin: float = None,
                 max_keywords: int = None,
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
        self.skip_verification = skip_verification if skip_verification is not None else cfg.SKIP_VERIFICATION
        self.debug = debug if debug is not None else cfg.DEBUG_MODE

        # Instance-owned issue catalog (loaded at init, not module level)
        self.issue_catalog_path = ISSUE_CATALOG_PATH
        try:
            self.issue_catalog = load_issue_catalog(self.issue_catalog_path) or {}
        except Exception as e:
            logger.error(f"Failed to load issue catalog at {self.issue_catalog_path}: {e}")
            self.issue_catalog = {}

        # Helpful sanity logging
        log_catalog_load(self.issue_catalog_path, self.issue_catalog)

        # -- Embeddings-based catalog matcher (optional) -----------------------
        self.catalog_matcher = None
        if getattr(cfg, "USE_EMBEDDINGS_CATALOG", False) and EMBEDDINGS_MATCHER_AVAILABLE:
            try:
                self.catalog_matcher = CatalogEmbeddingsRetriever(
                    catalog_v2=self.issue_catalog,
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

        # Backend selection (CLI arg > cfg/env > default)
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

        # Analysis profile selection (CLI arg > cfg/env > default)
        self.analysis_profile = (
                analysis_profile
                or getattr(cfg, "ANALYSIS_PROFILE", None)
                or "standard"
        ).strip().lower()

        # Pass architecture configuration
        # If no explicit pass_toggles provided, check SKIP_PASSES from config/env.
        if pass_toggles:
            self.pass_toggles = pass_toggles
        else:
            skip_list = getattr(cfg, "SKIP_PASSES", [])
            if skip_list:
                # Convert SKIP_PASSES list → dict of {pass_key: False} for disabled,
                # merged with defaults so from_dict sees explicit False values.
                self.pass_toggles = PassToggles.from_skip_list(skip_list).to_dict()
                logger.info(f"SKIP_PASSES active — disabled passes: {skip_list}")
            else:
                self.pass_toggles = {}
        self.model_overrides = model_overrides or {}

        # Initialize VLM client and model configs FIRST (needed for all paths)
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
                # Build candidate_provider from embeddings matcher or catalog
                candidate_provider = None
                if self.catalog_matcher:
                    # Kind-aware, scene-group-gated provider: dispatches to defect or upgrade
                    # retrieval based on ctx. ctx["scene_group"] gates to scene-relevant items only;
                    # if ctx gives a raw scene id (e.g. "living_room") it's mapped to a group first.
                    def _embeddings_candidate_provider(query, ctx=None):
                        _ctx = ctx if isinstance(ctx, dict) else {}
                        kind = _ctx.get("kind", "defect")
                        topk = _ctx.get("top_k_candidates", 8)
                        # Prefer pre-resolved scene_group; fall back to mapping raw scene id
                        scene_group = _ctx.get("scene_group")
                        if not scene_group:
                            scene_id = _ctx.get("scene")
                            scene_group = SCENE_TO_GROUP_UI.get(scene_id) if scene_id else None
                        allowed_groups = {scene_group} if scene_group else None
                        if kind == "upgrade":
                            cands = self.catalog_matcher.embeddings_retrieve_upgrade_candidates(
                                query, topk=topk, allowed_groups=allowed_groups
                            )
                        else:
                            cands = self.catalog_matcher.embeddings_retrieve_defect_candidates(
                                query, topk=topk, allowed_groups=allowed_groups
                            )
                        return [
                            {
                                "item_id": c.item_id,
                                "name": c.name,
                                "kind": c.kind,
                                "trade_bucket": c.trade_bucket,
                            }
                            for c in cands
                        ]
                    candidate_provider = _embeddings_candidate_provider
                    logger.info("Candidate provider: kind-aware, scene-group-gated embeddings")

                if candidate_provider is None:
                    logger.info("No candidate provider available -- 2d will skip (no embeddings matcher)")

                # Give orchestrator candidate_provider + catalog items for 2e policy gating
                self.orchestrator = create_orchestrator_from_config(
                    cfg,
                    candidate_provider=candidate_provider,
                    catalog_items=self.issue_catalog.get("items") if self.issue_catalog else None,
                )
                logger.info(f"Pass architecture initialized (premium={self.run_options.premium})")
            except Exception as e:
                logger.warning(f"Failed to initialize pass architecture: {e}, falling back to direct pass calls")
                self.use_pass_architecture = False

        # Initialize scene classifier service
        self.scene_service = SceneClassifierService(cfg, self.orchestrator, self.run_options)

        # Initialize DINO-X client if using dinox backend
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
            '2e': ['GPT_PASS_2E_MODEL', 'OPENAI_PASS2E_MODEL'],
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
            "2e": "OPENAI_PASS_2E_MAX_TOKENS",
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

        # Validate premium profile requirements
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
            # 1. Classify scene and get keywords via SceneClassifierService
            meta = {
                "run_id": getattr(self, "_active_run_id", "") or "",
                "photo_key": image_path.name,
                "property_key": getattr(self, "_active_property_key", "") or "",
            }
            sc = self.scene_service.classify(image_path, meta=meta)

            scene = sc.scene
            reasoning = sc.reasoning
            keywords = sc.keywords
            prompt = sc.prompt
            planner_targets = sc.planner_targets
            planner_hints = sc.planner_hints
            scene_details = sc.payload

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

            # 2. Run detection + postprocessing via detection_pipeline
            detection_result, detections = run_detection_stage(
                cfg=cfg,
                python_exe=self.python_exe,
                backend=self.detection_backend,
                image_path=image_path,
                output_dir=output_dir,
                prompt=prompt,
                scene=scene,
                planner_targets=planner_targets,
                planner_hints=planner_hints,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                chip_margin=self.chip_margin,
                debug=self.debug,
                dinox_client=self.dinox_client,
            )

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
            fallback_scene = scene_classifier_payload(
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

        # Make run_id available to scene classifier service for deterministic issue IDs
        self._active_run_id = job_id
        self._active_property_key = property_key

        # Per-job file handler so UI can always read run.log from artifacts
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
            logger.info(f"\n[{idx + 1}/{len(images)}] Processing: {image_path.name} -> {output_dir.name}")

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
                "skip_verification": self.skip_verification,
                "use_nms": getattr(cfg, "USE_NMS", True),
                "use_scene_caps": getattr(cfg, "USE_SCENE_CAPS", False),
            }
        )

        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)

        # Remove per-job file handler to avoid leaking across jobs
        if job_log_handler:
            try:
                logging.getLogger().removeHandler(job_log_handler)
                job_log_handler.close()
            except Exception:
                pass

        # Clear run-scoped identifiers
        self._active_run_id = None
        self._active_property_key = None

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

    def save_photo_intel(self, job, output_path=None):
        """Thin wrapper delegating to artifact_writers.write_photo_intel."""
        return write_photo_intel(
            cfg=cfg,
            job=job,
            detection_backend=self.detection_backend,
            analysis_profile=self.analysis_profile,
            use_pass_architecture=self.use_pass_architecture,
            pass_toggles=self.pass_toggles,
            model_overrides=self.model_overrides,
            gpt_config=self.gpt_config,
            issue_catalog=self.issue_catalog,
            output_path=output_path,
        )
