"""
RealtorVision Pipeline Configuration
-------------------------------------
Edit this file to change pipeline parameters without touching the code.

This version is configured for tools/ directory placement.
All scripts are in the tools/ directory.
"""

import os
from pathlib import Path
from typing import Optional


# =============================================================================
# Helpers
# =============================================================================

def _env_any(*keys: str) -> str:
    """Return first non-empty env var among keys, else empty string."""
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return ""


def _to_int_or_none(v) -> Optional[int]:
    try:
        return int(v) if v is not None and str(v).strip() != "" else None
    except Exception:
        return None


def _to_bool_or_none(v: Optional[str]) -> Optional[bool]:
    """
    Parse env var into bool, but keep None when unset/empty.
    Accepts: 1/true/yes/on and 0/false/no/off (case-insensitive).
    """
    if v is None:
        return None
    s = str(v).strip().lower()
    if s == "":
        return None
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return None


# =============================================================================
# PROJECT PATHS
# =============================================================================
# Config file is in tools/, so parent is project root
TOOLS_DIR = Path(__file__).parent  # tools/
PROJECT_ROOT = TOOLS_DIR.parent
DEMO_DIR = PROJECT_ROOT / "demo"

# Output directory
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

# Issue catalog path
ISSUE_CATALOG_PATH = TOOLS_DIR / "issue_catalog.json"

# =============================================================================
# LM STUDIO / VLM SETTINGS (Qwen - local)
# =============================================================================
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://169.254.83.107:1234")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "unsloth/gemma-4-26b-a4b-it")

# =============================================================================
# OPENAI / GPT SETTINGS (Premium - cloud)
# =============================================================================
# API key (required for premium profile)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# GPT model configuration
# Priority: GPT5_MODEL > GPT_MODEL > OPENAI_MODEL > default
GPT_MODEL = (
        os.environ.get("GPT5_MODEL")
        or os.environ.get("GPT_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-5.6-sol"  #Need to stop relying on this and use env variables
)

# Alias for backward compatibility with vlm_client.py
OPENAI_MODEL = GPT_MODEL

# Optional: separate model for specific passes
GPT_PASS_1B_MODEL = _env_any("OPENAI_PASS_1B_MODEL", "OPENAI_PASS1B_MODEL") or GPT_MODEL  # Pass 1b
GPT_PASS_1C_MODEL = _env_any("OPENAI_PASS_1C_MODEL", "OPENAI_PASS1C_MODEL") or GPT_MODEL  # Pass 1c
GPT_PASS_2A_MODEL = _env_any("OPENAI_PASS_2A_MODEL", "OPENAI_PASS2A_MODEL") or GPT_MODEL  # Pass 2a
GPT_PASS_2B_MODEL = _env_any("OPENAI_PASS_2B_MODEL", "OPENAI_PASS2B_MODEL") or GPT_MODEL  # Pass 2b (if ever routed)
GPT_PASS_2C_MODEL = _env_any("OPENAI_PASS_2C_MODEL", "OPENAI_PASS2C_MODEL") or GPT_MODEL  # Pass 2c
GPT_PASS_2D_MODEL = _env_any("OPENAI_PASS_2D_MODEL", "OPENAI_PASS2D_MODEL") or GPT_MODEL  # Pass 2d

# Optional: explicit Pass 4a/4b/4c models
GPT_PASS_4A_MODEL = _env_any("OPENAI_PASS_4A_MODEL", "OPENAI_PASS4A_MODEL") or GPT_MODEL
GPT_PASS_4B_MODEL = _env_any("OPENAI_PASS_4B_MODEL", "OPENAI_PASS4B_MODEL") or GPT_MODEL
GPT_PASS_4C_MODEL = _env_any("OPENAI_PASS_4C_MODEL", "OPENAI_PASS4C_MODEL") or GPT_MODEL

# =============================================================================
# GOOGLE GEMINI SETTINGS (Cloud - for catalog_auditor)
# =============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBVoSb4gygaBh2ScxfceIIAJ7-1bjnQJLc")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")

# =============================================================================
# Embeddings-based catalog matching
# =============================================================================
USE_EMBEDDINGS_CATALOG = True
EMBEDDINGS_MODEL_NAME = "jinaai/jina-embeddings-v3"
EMBEDDINGS_TRUST_REMOTE_CODE = True
EMBEDDINGS_TOPK = 5
EMBEDDINGS_THRESHOLD_DEFECT = 0.58
EMBEDDINGS_THRESHOLD_OPPORTUNITY = 0.56
EMBEDDINGS_ROUTE_BY_ROUGH_CATEGORY = True
EMBEDDINGS_OVERRIDE_EXISTING_FLAGS = True
EMBEDDINGS_ATTACH_CANDIDATES = True
EMBEDDINGS_DEVICE = "cpu"
PASS_2D_SHORTCUT_MIN_SCORE = float(os.environ.get("PASS_2D_SHORTCUT_MIN_SCORE", "0.72"))
PASS_2D_SHORTCUT_MIN_MARGIN = float(os.environ.get("PASS_2D_SHORTCUT_MIN_MARGIN", "0.03"))
PASS_2D_ROUTING_NEGATION_PATTERNS = [
    r"\bno\s+visible\s+damage\b",
    r"\bno\s+damage\s+is\s+visible\b",
    r"\bwithout\s+visible\s+damage\b",
    r"\bintact\b",
    r"\bconsistent\s+with\s+(?:the\s+)?age\b",
]

# =============================================================================
# Pass 2c shadow lane
# -----------------------------------------------------------------------------
# Pass 2c is a single-label filter: only `defect_or_damage` and
# `upgrade_candidate` reach the matcher. The failure mode we're guarding against
# is `generic_presence` (and optionally `other`) swallowing a real issue. The
# shadow lane re-checks those observations when they carry physical-condition
# language, retrieves catalog candidates with widened kinds, and (optionally)
# promotes them when a specific non-generic catalog item clearly beats the
# broad style/dated alternatives.
#
# Shipping posture: ENABLED on, PROMOTE off. The lane runs and writes audit
# rows to `result.debug["shadow_lane"]` so we can measure how often it would
# have rescued a real issue. Flip SHADOW_LANE_PROMOTE=1 once the per-observation
# rows confirm the lane is catching genuine misses without noise.
# =============================================================================
SHADOW_LANE_ENABLED = os.environ.get("SHADOW_LANE_ENABLED", "1") not in {"0", "false", "False"}
SHADOW_LANE_PROMOTE = os.environ.get("SHADOW_LANE_PROMOTE", "0") not in {"0", "false", "False"}
SHADOW_LANE_LABELS = [
    s.strip().lower()
    for s in os.environ.get("SHADOW_LANE_LABELS", "generic_presence").split(",")
    if s.strip()
]
SHADOW_LANE_MIN_SCORE = float(os.environ.get("SHADOW_LANE_MIN_SCORE", "0.72"))
SHADOW_LANE_MIN_MARGIN = float(os.environ.get("SHADOW_LANE_MIN_MARGIN", "0.03"))
SHADOW_LANE_MIN_SPECIFIC_OVER_GENERIC = float(
    os.environ.get("SHADOW_LANE_MIN_SPECIFIC_OVER_GENERIC", "0.02")
)

# =============================================================================
# PASS TOGGLE SETTINGS
# =============================================================================
# Comma-separated list of passes to SKIP (disable).
# Valid keys: 1a, 1b, 1c, 2a, 2b, 2c, 2d, 2e, 2f, 4, 4a, 4b, 4c
#
# Examples:
#   SKIP_PASSES=1a,1b,1c          # Skip all feature-extraction passes
#   SKIP_PASSES=2d                # Skip resolver only
#   SKIP_PASSES=                  # Skip nothing (all passes enabled)
#
# This is the easiest way to temporarily disable passes without touching code.
# Downstream passes degrade gracefully when upstream passes are skipped.
SKIP_PASSES = [
    s.strip().lower()
    for s in os.environ.get("SKIP_PASSES", "").split(",")
    if s.strip()
]

# =============================================================================
# ANALYSIS PROFILE SETTINGS
# =============================================================================
# Default profile: "standard" (all Qwen) or "premium" (GPT for key passes)
ANALYSIS_PROFILE = os.environ.get("ANALYSIS_PROFILE", "standard")

# Premium-specific overrides
PREMIUM_MAX_KEYWORDS = int(os.environ.get("PREMIUM_MAX_KEYWORDS", "30"))

# =============================================================================
# OPENAI TOKEN CAPS
# =============================================================================
OPENAI_DEFAULT_MAX_TOKENS = int(os.environ.get("OPENAI_DEFAULT_MAX_TOKENS", "600"))

# Optional per-pass caps (leave None if unset)
OPENAI_PASS_1B_MAX_TOKENS = _to_int_or_none(os.environ.get("OPENAI_PASS_1B_MAX_TOKENS"))
OPENAI_PASS_1C_MAX_TOKENS = _to_int_or_none(os.environ.get("OPENAI_PASS_1C_MAX_TOKENS"))
OPENAI_PASS_2A_MAX_TOKENS = _to_int_or_none(os.environ.get("OPENAI_PASS_2A_MAX_TOKENS"))

# Legacy Pass 4 cap (kept for compatibility)
OPENAI_PASS_4_MAX_TOKENS = _to_int_or_none(os.environ.get("OPENAI_PASS_4_MAX_TOKENS"))

# Optional: Pass 4a/4b/4c caps
OPENAI_PASS_4A_MAX_TOKENS = _to_int_or_none(os.environ.get("OPENAI_PASS_4A_MAX_TOKENS"))
OPENAI_PASS_4B_MAX_TOKENS = _to_int_or_none(os.environ.get("OPENAI_PASS_4B_MAX_TOKENS"))
OPENAI_PASS_4C_MAX_TOKENS = _to_int_or_none(os.environ.get("OPENAI_PASS_4C_MAX_TOKENS"))

# =============================================================================
# DINO-X Configuration Variables
# =============================================================================

# ─── Backend Selection ───────────────────────────────────────────────────────
DETECTION_BACKEND = os.environ.get("DETECTION_BACKEND", "dinox")

# ─── DINO-X API Settings ─────────────────────────────────────────────────────
# Get your API token from https://cloud.deepdataspace.com/
# Supports both DINOX_API_TOKEN and DDS_API_TOKEN for flexibility
DINOX_API_TOKEN = (
        os.environ.get("DINOX_API_TOKEN")
        or os.environ.get("DDS_API_TOKEN")
        or ""
)

# API Endpoints (v2 API)
DINOX_DETECTION_ENDPOINT = "https://api.deepdataspace.com/v2/task/dinox/detection"
DINOX_STATUS_ENDPOINT = "https://api.deepdataspace.com/v2/task_status"

# Model selection
DINOX_MODEL = os.environ.get("DDS_DETECTOR_MODEL", "DINO-X-1.0")

# Detection targets - what outputs to request from DINO-X
# Options typically include: "bbox", "mask", "keypoint"
DINOX_TARGETS = ["bbox"]

# Thresholds
DINOX_BBOX_THRESHOLD = 0.25  # Minimum confidence for bounding boxes
DINOX_IOU_THRESHOLD = 0.8  # IoU threshold for NMS on DINO-X side

# Timeouts (in seconds)
DINOX_REQUEST_TIMEOUT = 60  # Timeout for individual HTTP requests
DINOX_POLL_TIMEOUT = 120  # Max time to wait for task completion
DINOX_POLL_INTERVAL = 1.0  # Seconds between status polls

# =============================================================================
# DETECTION PARAMETERS
# =============================================================================
BOX_THRESHOLD = 0.30  # Confidence threshold for detections (0-1)
TEXT_THRESHOLD = 0.25  # Text-image matching threshold (0-1)

# =============================================================================
# SCENE CLASSIFICATION PARAMETERS
# =============================================================================
INCLUDE_CONDITIONS = False  # Include defect keywords (crack, stain, damage, etc.)
INCLUDE_COMMON = True  # Include common object keywords in prompts

# =============================================================================
# PROCESSING OPTIONS
# =============================================================================
CREATE_THUMBNAILS = True  # Create thumbnail with detection overlays
THUMBNAIL_SIZE = 384  # Thumbnail dimension in pixels
CPU_ONLY = False  # Run detection on CPU only (slower)

# =============================================================================
# OUTPUT OPTIONS
# =============================================================================
GENERATE_HTML_REPORT = True  # Create HTML summary report
GENERATE_PROPERTY_SUMMARY = True  # Generate property-level summary
SAVE_JSON_SUMMARY = True  # Save JSON summary of results
DEBUG_MODE = os.environ.get("DEBUG_MODE", "true").lower() == "true"

# =============================================================================
# ROI HINTS
# =============================================================================
ROI_HINTS_ENABLED = True

# 3×3 grid thresholds and scoring
ROI_FULL_BONUS = 0.06  # add to score when overlap >= ROI_OVERLAP_HI
ROI_HALF_BONUS = 0.03  # add when ROI_OVERLAP_LO <= overlap < ROI_OVERLAP_HI
ROI_PENALTY = 0.03  # subtract if clearly opposite zone and overlap < ROI_OVERLAP_LO

ROI_OVERLAP_HI = 0.40  # fraction of detection area inside hinted zone for full bonus
ROI_OVERLAP_LO = 0.10  # fraction for half bonus lower bound

# Map of {scene OR scene-group: {normalized_label: zone}}
# Labels should be written naturally ("light fixture", not "light_fixture")
ROI_HINTS_BY_SCENE = {
    # Scene-group keys
    "kitchen": {
        "sink": "bottom_center",
        "faucet": "bottom_center",
        "range": "center",
        "stove": "center",
        "oven": "center",
        "dishwasher": "bottom_right",
        "refrigerator": "mid_left",
        "cabinet": "top_center",
        "countertop": "center",
        "microwave": "top_right",
    },
    "bathroom": {
        "toilet": "bottom_center",
        "sink": "bottom_center",
        "vanity": "bottom_center",
        "mirror": "top_center",
        "bathtub": "mid_right",
        "shower": "mid_right",
        "faucet": "bottom_center",
        "light fixture": "top_center",
    },
    "living_areas": {
        "sofa": "bottom_center",
        "couch": "bottom_center",
        "tv": "center",
        "fireplace": "center",
        "ceiling fan": "top_center",
        "light fixture": "top_center",
        "window": "top_center",
        "door": "mid_left",
    },
    "bedroom": {
        "bed": "bottom_center",
        "window": "top_center",
        "closet": "mid_right",
        "dresser": "bottom_right",
        "door": "mid_left",
    },
    "exterior": {
        "roof": "top_center",
        "front door": "center",
        "garage door": "mid_left",
        "driveway": "bottom_center",
        "yard": "bottom_center",
        "lawn": "bottom_center",
        "deck": "bottom_center",
        "patio": "bottom_center",
    },
    "default": {
        # safe fallbacks
        "window": "top_center",
        "door": "mid_left",
        "light fixture": "top_center",
    },
}

# =============================================================================
# SPECIAL CASE FILTERS
# =============================================================================
# Configure bespoke post-processing passes.
# NOTE: Detection labels in your pipeline are often natural-language ("light fixture"),
# so include both natural and underscore variants if you have any legacy label sources.
SPECIAL_CASE_FILTERS = {
    "mirror_containment": {
        "enabled": True,
        "mirror_labels": ["mirror"],
        # Optional slack in pixels when deciding containment (helps w/ rounding)
        "containment_eps": 0.0,
    },
    "fixture_collapse": {
        "enabled": True,
        "fixture_labels": [
            "light fixture", "vanity light", "ceiling light",
            "light_fixture", "vanity_light", "ceiling_light",
        ],
        # Optional slack in pixels when deciding containment (helps w/ rounding)
        "containment_eps": 0.0,
    },
}

# =============================================================================
# SCENE KEYWORDS QUICK EDIT
# =============================================================================
# You can add custom keywords here that will be merged with defaults
# Format: {"scene_name": ["keyword1", "keyword2", ...]}
CUSTOM_SCENE_KEYWORDS = {
    # Example: add more kitchen items
    # "kitchen": ["blender", "mixer", "food processor"],
    # Example: add outdoor features
    # "yard": ["pergola", "gazebo", "fountain"],
}

# =============================================================================
# QUICK PRESETS
# =============================================================================
# Uncomment a preset to use it (will override settings above)

# Fast testing preset - lower thresholds
# BOX_THRESHOLD = 0.35
# CREATE_THUMBNAILS = False

# High quality preset - stricter thresholds
# BOX_THRESHOLD = 0.35
# TEXT_THRESHOLD = 0.30

# Condition detection preset - look for damage/defects
# INCLUDE_CONDITIONS = True
# BOX_THRESHOLD = 0.25
MAX_KEYWORDS = int(os.environ.get("MAX_KEYWORDS", "30"))
