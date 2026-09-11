"""
RealtorVision Pipeline Configuration
-------------------------------------
Edit this file to change pipeline parameters without touching the code.

This version is configured for tools/ directory placement.
All scripts are in the tools/ directory.
"""

import os
from collections import namedtuple
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

# =============================================================================
# KIND ONTOLOGY CUTOVER SELECTOR
# =============================================================================
# One atomic switch derives the catalog and the pipeline depth so incompatible
# combinations cannot be configured:
#   legacy_v1           -> v1 catalog + classification_only (three-kind output
#                          stays non-publishable; the real rollback path is the
#                          pinned pre-v2 build, not this selector)
#   observation_kind_v2 -> v2 catalog + publish (full 2c -> 2d -> 2e)
# Invalid values raise here at import — i.e. process startup — in every entry
# point. Under observation_kind_v2 an ISSUE_CATALOG_PATH override is a hard
# error: the catalog must come from the selector.

KIND_ONTOLOGY_LEGACY_V1 = "legacy_v1"
KIND_ONTOLOGY_V2 = "observation_kind_v2"

KindOntologySelection = namedtuple(
    "KindOntologySelection", ("version", "catalog_path", "pipeline_mode")
)


def resolve_kind_ontology(
    raw: str, *, catalog_path_override: str = ""
) -> KindOntologySelection:
    """Map a KIND_ONTOLOGY_VERSION value to (catalog, pipeline depth)."""
    if raw == KIND_ONTOLOGY_LEGACY_V1:
        # The legacy catalog path stays overridable through the analyzer CLI's
        # env-override layer (applied after import), exactly as before.
        return KindOntologySelection(
            raw, TOOLS_DIR / "issue_catalog.json", "classification_only"
        )
    if raw == KIND_ONTOLOGY_V2:
        if catalog_path_override:
            raise ValueError(
                "ISSUE_CATALOG_PATH must not be set under "
                f"KIND_ONTOLOGY_VERSION={KIND_ONTOLOGY_V2}: catalog and "
                "pipeline depth both derive from the selector so they can "
                "never disagree"
            )
        return KindOntologySelection(
            raw, TOOLS_DIR / "issue_catalog_kind_v2.json", "publish"
        )
    raise ValueError(
        f"invalid KIND_ONTOLOGY_VERSION {raw!r}; expected "
        f"{KIND_ONTOLOGY_LEGACY_V1!r} or {KIND_ONTOLOGY_V2!r}"
    )


_KIND_ONTOLOGY = resolve_kind_ontology(
    os.environ.get("KIND_ONTOLOGY_VERSION", KIND_ONTOLOGY_LEGACY_V1),
    catalog_path_override=os.environ.get("ISSUE_CATALOG_PATH", ""),
)
KIND_ONTOLOGY_VERSION = _KIND_ONTOLOGY.version
ISSUE_CATALOG_PATH = _KIND_ONTOLOGY.catalog_path
# Plain string (not the pass_config Literal) to keep this module import-free;
# membership in pass_config.ALLOWED_PIPELINE_MODES is pinned by test.
PIPELINE_MODE = _KIND_ONTOLOGY.pipeline_mode

# =============================================================================
# RENOVATION ARCHITECTURE SELECTOR
# =============================================================================
# One atomic switch for the new estimate architecture (see
# docs/IMPLEMENTATION_PLAN_renovation_scope_estimate_architecture.md):
#   current -> the existing v4 estimator only; no new output anywhere
#   shadow  -> v4 unchanged, plus a private scaffold envelope in analysis_debug
#              (requires the v2 ontology selector: the new engine accepts only
#              the v3.2 catalog)
#   new     -> the new engine is authoritative: a complete v5 envelope at the
#              artifact root, v4 still emitted independently, no private copy
#              (same v2-ontology requirement as shadow)
# Invalid values and incompatible combinations raise here at import — i.e.
# process startup — in every entry point. This selector derives nothing else:
# ISSUE_CATALOG_PATH and PIPELINE_MODE stay owned by the kind-ontology
# selector above, and there is deliberately no env-override path for it.

RENOVATION_ARCH_CURRENT = "current"
RENOVATION_ARCH_SHADOW = "shadow"
RENOVATION_ARCH_NEW = "new"


def resolve_renovation_architecture(raw: str, *, kind_ontology_version: str) -> str:
    """Validate a RENOVATION_ARCHITECTURE_MODE value against the ontology."""
    if raw == RENOVATION_ARCH_CURRENT:
        return raw
    if raw in (RENOVATION_ARCH_SHADOW, RENOVATION_ARCH_NEW):
        if kind_ontology_version != KIND_ONTOLOGY_V2:
            raise ValueError(
                f"RENOVATION_ARCHITECTURE_MODE={raw} requires "
                f"KIND_ONTOLOGY_VERSION={KIND_ONTOLOGY_V2}, got "
                f"{kind_ontology_version!r}: the new engine accepts only the "
                "v3.2 catalog"
            )
        return raw
    raise ValueError(
        f"invalid RENOVATION_ARCHITECTURE_MODE {raw!r}; expected "
        f"{RENOVATION_ARCH_CURRENT!r}, {RENOVATION_ARCH_SHADOW!r}, or "
        f"{RENOVATION_ARCH_NEW!r}"
    )


RENOVATION_ARCHITECTURE_MODE = resolve_renovation_architecture(
    os.environ.get("RENOVATION_ARCHITECTURE_MODE", RENOVATION_ARCH_CURRENT),
    kind_ontology_version=KIND_ONTOLOGY_VERSION,
)

# =============================================================================
# LM STUDIO / VLM SETTINGS (Qwen - local)
# =============================================================================
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://100.102.92.1:1234")
LM_STUDIO_MODEL = os.environ.get(
    "LM_STUDIO_MODEL",
    "unsloth/qwen3.6-27b@q6_k",
)

# =============================================================================
# OPENAI / GPT SETTINGS (Premium - cloud)
# =============================================================================
# API key (required for premium profile)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# OPENAI_MODEL is the single base-model source (Terra tier in production). Per-run
# per-pass model names arrive via the analyzer's --model-map argument and override
# this at runtime. There is intentionally NO hardcoded model fallback: a missing
# OPENAI_MODEL fails loudly rather than silently using a stale default.
# (Legacy GPT_MODEL / GPT5_MODEL aliases and GPT_PASS_*_MODEL were removed.)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "")

# =============================================================================
# TERRA CONDITION REVIEW (renovation architecture Session 2)
# =============================================================================
# Model + output cap for the shadow-mode Terra condition-review stage.
# RENOVATION_TERRA_MODEL falls back to OPENAI_MODEL; both empty resolves to ""
# here, and shadow-mode runtime init rejects the empty model at startup
# (current mode never reads it). The token cap must be a positive integer;
# invalid values raise at import, i.e. worker startup, in every entry point.


def resolve_renovation_terra_model(raw: str, *, openai_model: str) -> str:
    return (raw or "").strip() or (openai_model or "").strip()


def resolve_renovation_terra_max_output_tokens(raw: Optional[str]) -> int:
    if raw is None or str(raw).strip() == "":
        return 8192
    try:
        value = int(str(raw).strip())
    except ValueError:
        value = None
    if value is None or value <= 0:
        raise ValueError(
            f"RENOVATION_TERRA_MAX_OUTPUT_TOKENS must be a positive integer, "
            f"got {raw!r}"
        )
    return value


def resolve_renovation_terra_usage_root(raw: Optional[str]) -> Optional[str]:
    """Optional shared root for the Terra daily-budget ledger.

    Empty (the default) keeps the ledger under each run's own artifacts root.
    Setting it points several artifacts roots at ONE daily ledger, so the
    2.5M/day ceiling is enforced across them — the Session 6 canary runs two
    isolated replica roots that must share one budget.
    """
    value = (raw or "").strip()
    return value or None


RENOVATION_TERRA_MODEL = resolve_renovation_terra_model(
    os.environ.get("RENOVATION_TERRA_MODEL", ""), openai_model=OPENAI_MODEL
)
RENOVATION_TERRA_MAX_OUTPUT_TOKENS = resolve_renovation_terra_max_output_tokens(
    os.environ.get("RENOVATION_TERRA_MAX_OUTPUT_TOKENS")
)
RENOVATION_TERRA_USAGE_ROOT = resolve_renovation_terra_usage_root(
    os.environ.get("RENOVATION_TERRA_USAGE_ROOT")
)

# =============================================================================
# SOL PACKAGE REVIEW (renovation architecture Session 4)
# =============================================================================
# Model + output cap for the Sol package-review stage.
# RENOVATION_SOL_MODEL is mandatory in new mode. Current/shadow retain the
# historical OPENAI_MODEL fallback; both empty resolves to "" and shadow
# runtime init rejects it (current mode never reads it). An invalid token cap
# raises at import in every entry point.


def resolve_renovation_sol_model(raw: str, *, openai_model: str, mode: str = "current") -> str:
    explicit = (raw or "").strip()
    if mode == "new" and not explicit:
        raise ValueError("new mode requires an explicit RENOVATION_SOL_MODEL; OPENAI_MODEL fallback is disabled")
    return explicit or (openai_model or "").strip()


def resolve_renovation_sol_max_output_tokens(raw: Optional[str]) -> int:
    if raw is None or str(raw).strip() == "":
        return 8192
    try:
        value = int(str(raw).strip())
    except ValueError:
        value = None
    if value is None or value <= 0:
        raise ValueError(
            f"RENOVATION_SOL_MAX_OUTPUT_TOKENS must be a positive integer, "
            f"got {raw!r}"
        )
    return value


def resolve_renovation_sol_daily_ceiling(raw: Optional[str]) -> int:
    """Daily Sol token budget (Session 8 guard). The ledger shares the
    RENOVATION_TERRA_USAGE_ROOT location but debits its own file, so the two
    budgets stay separate. Default matches usage_guard.SOL_DAILY_TOKEN_CEILING
    (250k/day — the operating budget the Session 6 handoff recorded)."""
    if raw is None or str(raw).strip() == "":
        return 250_000
    try:
        value = int(str(raw).strip())
    except ValueError:
        value = None
    if value is None or value <= 0:
        raise ValueError(
            f"RENOVATION_SOL_DAILY_TOKEN_CEILING must be a positive integer, "
            f"got {raw!r}"
        )
    return value


RENOVATION_SOL_MODEL = resolve_renovation_sol_model(
    os.environ.get("RENOVATION_SOL_MODEL", ""), openai_model=OPENAI_MODEL,
    mode=RENOVATION_ARCHITECTURE_MODE,
)
RENOVATION_SOL_MAX_OUTPUT_TOKENS = resolve_renovation_sol_max_output_tokens(
    os.environ.get("RENOVATION_SOL_MAX_OUTPUT_TOKENS")
)
RENOVATION_SOL_DAILY_TOKEN_CEILING = resolve_renovation_sol_daily_ceiling(
    os.environ.get("RENOVATION_SOL_DAILY_TOKEN_CEILING")
)

# =============================================================================
# VLM CHOKE-POINT BUDGET GUARD (renovation architecture Session 9)
# =============================================================================
# Opt-in metering of EVERY OpenAI call (upstream scene passes included)
# against the per-model daily ledgers in usage_guard. Off by default:
# production behavior is unchanged unless RENOVATION_VLM_BUDGET_GUARD is set
# (the canary coordinator sets it). The Terra ceiling env exists so a paid
# overage day is an explicit, recorded decision rather than a silent overrun.


def resolve_renovation_vlm_budget_guard(raw: Optional[str]) -> bool:
    value = (str(raw).strip().lower() if raw is not None else "")
    if value in ("", "0", "false"):
        return False
    if value in ("1", "true"):
        return True
    raise ValueError(
        f"RENOVATION_VLM_BUDGET_GUARD must be one of ''/0/false/1/true, "
        f"got {raw!r}"
    )


def resolve_renovation_terra_daily_ceiling(raw: Optional[str]) -> int:
    """Daily Terra token budget (input+output combined — the OpenAI free
    allowance on the Terra model's tier). Default matches
    usage_guard.TERRA_DAILY_TOKEN_CEILING (2.5M/day)."""
    if raw is None or str(raw).strip() == "":
        return 2_500_000
    try:
        value = int(str(raw).strip())
    except ValueError:
        value = None
    if value is None or value <= 0:
        raise ValueError(
            f"RENOVATION_TERRA_DAILY_TOKEN_CEILING must be a positive integer, "
            f"got {raw!r}"
        )
    return value


RENOVATION_VLM_BUDGET_GUARD = resolve_renovation_vlm_budget_guard(
    os.environ.get("RENOVATION_VLM_BUDGET_GUARD")
)
RENOVATION_TERRA_DAILY_TOKEN_CEILING = resolve_renovation_terra_daily_ceiling(
    os.environ.get("RENOVATION_TERRA_DAILY_TOKEN_CEILING")
)

# =============================================================================
# GOOGLE GEMINI SETTINGS (Cloud - for catalog_auditor)
# =============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBVoSb4gygaBh2ScxfceIIAJ7-1bjnQJLc")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")

# =============================================================================
# Embeddings-based catalog matching
# =============================================================================
# (A USE_EMBEDDINGS_CATALOG flag was removed: it was hardcoded True with no env
# override, and it gated only the *attempt* to build the retriever, never its
# success. The Pass 2d toggle is now the single switch -- disable 2d to run
# without embeddings.)

# Backend transport for catalog embeddings.
#   "openai_compatible" (default) → local llama-server sidecar serving the Q8_0
#       GGUF text-matching model over /v1/embeddings. Start it first with
#       scripts/start-embeddings-server.ps1 (no v3 fallback if unavailable).
#   "sentence_transformer" → in-process SentenceTransformer using
#       EMBEDDINGS_ST_MODEL_NAME (rollback / tests). Flip EMBEDDINGS_BACKEND only.
EMBEDDINGS_BACKEND = os.environ.get("EMBEDDINGS_BACKEND", "openai_compatible")
EMBEDDINGS_BASE_URL = os.environ.get("EMBEDDINGS_BASE_URL", "http://127.0.0.1:8081/v1")

# HTTP backend: the served model id (also the llama-server alias). Recorded in artifacts.
EMBEDDINGS_MODEL_NAME = os.environ.get(
    "EMBEDDINGS_MODEL_NAME",
    "jinaai/jina-embeddings-v5-text-small-text-matching-GGUF:Q8_0",
)
# SentenceTransformer backend model (a GGUF id cannot be loaded by SentenceTransformer()).
EMBEDDINGS_ST_MODEL_NAME = os.environ.get("EMBEDDINGS_ST_MODEL_NAME", "jinaai/jina-embeddings-v3")
EMBEDDINGS_DIMENSION = int(os.environ.get("EMBEDDINGS_DIMENSION", "1024"))

EMBEDDINGS_TRUST_REMOTE_CODE = True
EMBEDDINGS_TOPK = 5
EMBEDDINGS_DEVICE = "cpu"
# Retrieval match thresholds live with their only consumer, catalog_auditor.py.
# Applied only by run_pass_2d, never to the shared Qwen model configuration.
PASS_2D_TEMPERATURE = float(os.environ.get("PASS_2D_TEMPERATURE", "0.1"))
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
# PASS TOGGLE SETTINGS
# =============================================================================
# Pass toggles are set per-run via the analyzer CLI's --enable-<key>/--disable-<key>
# flags (or the server's request payload), which is the only transport that reaches
# photo_intel["run"]["pass_toggles"]. A previous SKIP_PASSES env var was removed:
# it was parsed here but read by nothing, and it left no trace in run metadata.

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
OPENAI_DEFAULT_MAX_TOKENS = int(os.environ.get("OPENAI_DEFAULT_MAX_TOKENS", "2000"))

# Optional per-pass caps (leave None if unset)
OPENAI_PASS_1B_MAX_TOKENS = _to_int_or_none(os.environ.get("OPENAI_PASS_1B_MAX_TOKENS"))
OPENAI_PASS_1C_MAX_TOKENS = _to_int_or_none(os.environ.get("OPENAI_PASS_1C_MAX_TOKENS"))
OPENAI_PASS_2A_MAX_TOKENS = _to_int_or_none(os.environ.get("OPENAI_PASS_2A_MAX_TOKENS"))

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
