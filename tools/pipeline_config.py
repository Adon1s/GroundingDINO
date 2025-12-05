"""
GroundingDINO Pipeline Configuration
-------------------------------------
Edit this file to change pipeline parameters without touching the code.

This version is configured for tools/ directory placement.
All scripts are in the tools/ directory.
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
# Config file is in tools/, so parent is GroundingDINO root
TOOLS_DIR = Path(__file__).parent  # tools/
PROJECT_ROOT = TOOLS_DIR.parent     # GroundingDINO/
DEMO_DIR = PROJECT_ROOT / "demo"

# GroundingDINO setup
GDINO_CONFIG = PROJECT_ROOT / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GDINO_CHECKPOINT = PROJECT_ROOT / "weights" / "groundingdino_swint_ogc.pth"
GDINO_INFER_SCRIPT = DEMO_DIR / "inference_on_a_image.py"

# Output directory
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

# ============================================================================
# LM STUDIO / VLM SETTINGS
# ============================================================================
LM_STUDIO_URL = "http://169.254.83.107:1234"
LM_STUDIO_MODEL = "qwen/qwen3-vl-30b"

# =============================================================================
# DINO-X Configuration Variables
# Add these to your pipeline_config.py to enable DINO-X backend support
# =============================================================================

# ─── Backend Selection ───────────────────────────────────────────────────────
# Options: "groundingdino" (local) or "dinox" (remote API)
DETECTION_BACKEND = "groundingdino"  # Default to local GroundingDINO

# ─── DINO-X API Settings ─────────────────────────────────────────────────────
# Get your API token from https://cloud.deepdataspace.com/
DINOX_API_TOKEN = os.environ.get("DINOX_API_TOKEN", "")

# API Endpoints (v2 API)
DINOX_DETECTION_ENDPOINT = "https://api.deepdataspace.com/v2/task/dinox/detection"
DINOX_STATUS_ENDPOINT = "https://api.deepdataspace.com/v2/task_status"

# Model selection
DINOX_MODEL = "DINO-X-1.0"

# Detection targets - what outputs to request from DINO-X
# Options typically include: "bbox", "mask", "keypoint"
DINOX_TARGETS = ["bbox"]

# Thresholds
DINOX_BBOX_THRESHOLD = 0.25  # Minimum confidence for bounding boxes
DINOX_IOU_THRESHOLD = 0.8    # IoU threshold for NMS on DINO-X side

# Timeouts (in seconds)
DINOX_REQUEST_TIMEOUT = 60   # Timeout for individual HTTP requests
DINOX_POLL_TIMEOUT = 120     # Max time to wait for task completion
DINOX_POLL_INTERVAL = 1.0    # Seconds between status polls


# =============================================================================
# Usage Examples
# =============================================================================
#
# 1. Using local GroundingDINO (default):
#    DETECTION_BACKEND = "groundingdino"
#
# 2. Using DINO-X API:
#    DETECTION_BACKEND = "dinox"
#    DINOX_API_TOKEN = "your-api-token-here"
#
# 3. Setting token via environment variable (recommended for security):
#    export DINOX_API_TOKEN="your-api-token-here"
#
# 4. Programmatic override when creating analyzer:
#    analyzer = AutoAnalyzer(detection_backend="dinox")
#
# =============================================================================
# Notes
# =============================================================================
#
# - When using DINO-X, chip extraction is not available, so skip_verification
#   is automatically forced to True
#
# - Prompts are automatically normalized for DINO-X format:
#   "water stain. cracked tile." -> "waterstain.crackedtile"
#
# - DINO-X detection results include a "source": "dinox" field for tracking
#
# - Raw DINO-X API responses are saved to dinox_raw.json for debugging
#
# - The pred.json schema remains compatible with the rest of your pipeline
#   (NMS, ROI hints, overlays, photo_intel, etc. all work unchanged)
#

# ============================================================================
# DETECTION PARAMETERS (GroundingDINO)
# ============================================================================
BOX_THRESHOLD = 0.30        # Confidence threshold for detections (0-1)
TEXT_THRESHOLD = 0.25       # Text-image matching threshold (0-1)
CHIP_MARGIN = 0.15          # Extra margin around crops (0.15 = 15%)

# ============================================================================
# SCENE CLASSIFICATION PARAMETERS
# ============================================================================
MAX_KEYWORDS = 25           # Maximum keywords per scene for detection
INCLUDE_CONDITIONS = False  # Include defect keywords (crack, stain, damage, etc.)
INCLUDE_COMMON = True       # Include common object keywords in prompts

# ============================================================================
# VERIFICATION PARAMETERS
# ============================================================================
SKIP_VERIFICATION = False   # Set to True to skip chip verification (faster)
MAX_CHIPS_PER_DETECTION = 3 # Number of chips to verify per detection

# Verification thresholds
VERIFY_CONSENSUS_RATIO = 0.60   # Fraction of chips that must be valid
VERIFY_AVG_CONFIDENCE = 0.60    # Average model confidence across chips

# ============================================================================
# PROCESSING OPTIONS
# ============================================================================
CREATE_THUMBNAILS = True    # Create thumbnail with detection overlays
THUMBNAIL_SIZE = 384        # Thumbnail dimension in pixels
COMPUTE_CHIP_QUALITY = True # Calculate quality metrics for chips
CPU_ONLY = False            # Run detection on CPU only (slower)

# ============================================================================
# OUTPUT OPTIONS
# ============================================================================
GENERATE_HTML_REPORT = True # Create HTML summary report
SAVE_JSON_SUMMARY = True    # Save JSON summary of results
DEBUG_MODE = True          # Enable detailed logging

# ============================================================================
# ROI HINTS
# ============================================================================
ROI_HINTS_ENABLED = True

# 3×3 grid thresholds and scoring
ROI_FULL_BONUS = 0.06      # add to score when overlap >= ROI_OVERLAP_HI
ROI_HALF_BONUS = 0.03      # add when ROI_OVERLAP_LO <= overlap < ROI_OVERLAP_HI
ROI_PENALTY    = 0.03      # subtract if clearly opposite zone and overlap < ROI_OVERLAP_LO

ROI_OVERLAP_HI = 0.40      # fraction of detection area inside hinted zone for full bonus
ROI_OVERLAP_LO = 0.10      # fraction for half bonus lower bound

# ============================================================================
# SPECIAL CASE FILTERS
# ============================================================================
# Configure bespoke post-processing passes (easy to extend later).
SPECIAL_CASE_FILTERS = {
    "mirror_containment": {
        "enabled": True,
        "mirror_labels": ["mirror"],
        # Optional slack in pixels when deciding containment (helps w/ rounding)
        "containment_eps": 0.0,
    },
    "fixture_collapse": {
        "enabled": True,
        "fixture_labels": ["light_fixture", "vanity_light", "ceiling_light"],
        # Optional slack in pixels when deciding containment (helps w/ rounding)
        "containment_eps": 0.0,
    },
}

# ============================================================================
# SCENE KEYWORDS QUICK EDIT
# ============================================================================
# You can add custom keywords here that will be merged with defaults
# Format: {"scene_name": ["keyword1", "keyword2", ...]}
CUSTOM_SCENE_KEYWORDS = {
    # Example: add more kitchen items
    # "kitchen": ["blender", "mixer", "food processor"],

    # Example: add outdoor features
    # "yard": ["pergola", "gazebo", "fountain"],
}

# ============================================================================
# QUICK PRESETS
# ============================================================================
# Uncomment a preset to use it (will override settings above)

# Fast testing preset - skip verification, lower thresholds
# SKIP_VERIFICATION = True
# BOX_THRESHOLD = 0.35
# CREATE_THUMBNAILS = False
# COMPUTE_CHIP_QUALITY = False

# High quality preset - stricter thresholds, full verification
# BOX_THRESHOLD = 0.35
# TEXT_THRESHOLD = 0.30
# SKIP_VERIFICATION = False
# VERIFY_CONSENSUS_RATIO = 0.70
# VERIFY_AVG_CONFIDENCE = 0.65

# Condition detection preset - look for damage/defects
# INCLUDE_CONDITIONS = True
# BOX_THRESHOLD = 0.25
# MAX_KEYWORDS = 30