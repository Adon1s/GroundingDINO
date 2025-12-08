#!/usr/bin/env python3
from __future__ import annotations

"""
Scene Classifier with Issue-Driven Target Planning for Property Photos
-----------------------------------------------------------------------
Purpose: Classify property photos into scene types and generate appropriate
object detection targets for GroundingDINO based on detected issues.

Pipeline (5 VLM passes per image):
1. Pass 1 - Scene Summary: scene type + overall impression
2. Pass 2a - Freeform Defect Notes: image + simple question → plain text notes
3. Pass 2b - Defect Structuring: text-only → issues_natural_language + catalog_flags
4. Pass 3 - Staging Detection: is_staged
5. Pass 4 - Issue-Driven Target Planning: image + issues → targets for GroundingDINO

Note: Target planning happens ONLY in Pass 4 (with image context).
      Pass 2b is purely text-based structuring of the freeform notes.

Output Fields:
- keywords: Scene-based keywords (core only vs core + staging), controlled by
            --max-keywords and --include-conditions CLI flags
- gdino_terms: Issues-driven detection terms from the planner (Pass 4 only)
- groundingdino_prompt: Formatted gdino_terms for GroundingDINO input
- targets: Structured target objects with labels, synonyms, roi_hints, priorities (Pass 4 only)
- vlm_outputs: Debug capture of ALL raw VLM responses for every pass

Features:
- Detects scene type (room/area)
- Determines if photo is staged or vacant
- Returns scene-based keywords (core only vs core + staging)
- Plans issues-driven targets for detection with priorities (Pass 4 only)
- Filters out generic surfaces (wall, floor, ceiling, carpet) from targets
- Captures all VLM outputs for debugging/analysis

Input: Path to a property photo
Output: JSON with scene classification, staging status, targets, and GroundingDINO prompt

Usage:
  python scene_classifier.py path/to/image.jpg
  python scene_classifier.py path/to/image.jpg --model qwen3-vl-30b --debug
  python scene_classifier.py path/to/image.jpg --max-keywords 15 --include-conditions
  python scene_classifier.py path/to/image.jpg --max-targets 8 --allow-free-discoveries 1
"""

import os
import sys
import json
import base64
import time
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, Optional, List, Literal
from dataclasses import dataclass, asdict, field
import requests

# ── Console encoding (Windows safety) ─────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration ────────────────────────────────────────────────────────────
# Try to import from pipeline_config first, then fall back to env vars
try:
    import pipeline_config as cfg

    LM_STUDIO_URL = cfg.LM_STUDIO_URL
    DEFAULT_MODEL = cfg.LM_STUDIO_MODEL
except ImportError:
    LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://169.254.83.107:1234")
    DEFAULT_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-vl-30b")

# ── Constants ────────────────────────────────────────────────────────────────
PROMPT_VERSION = "planner_v2_split"
DEFAULT_ISSUE_CATALOG = "issue_catalog.json"

# Static mapping from issue_id -> support-object labels for GroundingDINO
# NOTE: We intentionally avoid generic wall/ceiling/floor/carpet surfaces here
# to prevent huge, low-value chips. Those issues are handled in text only.
ANCHOR_MAP = {
    "outdated_kitchen_finishes": ["kitchen cabinets", "countertop"],
    "outdated_bathroom_finishes": ["bathroom vanity", "bathroom tile"],
    "damaged_or_aged_roof_shingles": ["roof"],
    "damaged_or_rotted_siding_or_trim": ["siding", "exterior trim"],
}
DEFAULT_DINO_CONFIG = Path("groundingdino/config/GroundingDINO_SwinT_OGC.py")
DEFAULT_DINO_WEIGHTS = Path("weights/groundingdino_swint_ogc.pth")

# ── Scene Types ──────────────────────────────────────────────────────────────
VALID_SCENES = [
    # Interior rooms
    "living_room",
    "kitchen",
    "bedroom",
    "bathroom",
    "dining_room",
    "home_office",
    "laundry_room",
    "hallway",
    "stairway",
    "basement",
    "attic",
    "garage",
    "closet",
    "pantry",

    # Exterior views
    "exterior_front",
    "exterior_back",
    "exterior_side",
    "yard",
    "patio",
    "deck",
    "balcony",
    "driveway",
    "pool_area",
    "garden",

    # Other/special
    "floor_plan",
    "aerial_view",
    "street_view",
    "unknown"
]

# ── NEW KEYWORD SCHEMA (Core vs Staging) ─────────────────────────────────────
# Core: Permanent fixtures, appliances, built-ins
# Staging: Furniture, decor, movable items
SCENE_KEYWORDS = {
    "living_room": {
        "core": ["fireplace", "mantel", "hearth", "built-in shelf", "built-in cabinet",
                 "ceiling fan", "recessed light", "bay window", "floor vent", "baseboard",
                 "media niche", "door", "radiator", "light fixture"],
        "staging": ["sofa", "sectional", "armchair", "coffee table", "end table",
                    "television", "tv stand", "media console", "floor lamp", "table lamp",
                    "rug", "bookshelf", "ottoman"]
    },

    "kitchen": {
        "core": ["range", "cooktop", "oven", "refrigerator", "microwave", "dishwasher",
                 "sink", "faucet", "countertop", "backsplash", "cabinet", "island",
                 "vent hood", "pantry", "pendant light"],
        "staging": ["bar stool", "counter stool", "chair", "table", "dish rack",
                    "fruit bowl", "trash can"]
    },

    "bedroom": {
        "core": ["door", "ceiling fan", "ceiling light", "baseboard heater",
                 "radiator", "smoke detector", "bay window",
                 "floor register"],
        "staging": ["bed", "headboard", "nightstand", "dresser", "wardrobe", "desk",
                    "chair", "table lamp", "floor lamp", "mirror", "television", "bench"]
    },

    "bathroom": {
        "core": ["toilet", "vanity", "sink", "faucet", "mirror", "shower", "bathtub", "towel bar",
                 "toilet paper holder", "grab bar", "floor drain"],
        "staging": ["bath mat", "shower curtain", "towel", "hamper", "storage cart"]
    },

    "dining_room": {
        "core": ["chandelier", "ceiling fan", "cabinet", "shelf",
                 "bay window", "door",
                 "floor vent"],
        "staging": ["dining table", "dining chair", "sideboard", "buffet", "hutch",
                    "bench", "rug", "bar cart"]
    },

    "home_office": {
        "core": ["shelf", "cabinet", "closet door", "ceiling fan",
                 "ceiling light",
                 "bay window"],
        "staging": ["desk", "office chair", "monitor", "computer", "printer",
                    "filing cabinet", "bookshelf", "task lamp"]
    },

    "laundry_room": {
        "core": ["washer", "dryer", "sink", "faucet", "countertop", "cabinet",
                 "shelf", "hanging rod", "vent duct", "floor drain",
                 "laundry hookups", "water heater", "electrical panel"],
        "staging": ["laundry basket", "hamper", "detergent bottle", "drying rack",
                    "ironing board"]
    },

    "hallway": {
        "core": ["ceiling light", "sconce", "smoke detector", "return vent",
                 "thermostat", "linen closet", "attic hatch", "baseboard heater",
                 "floor register", "handrail"],
        "staging": ["console table", "bench", "coat rack", "mirror", "wall art"]
    },

    "stairway": {
        "core": ["landing", "handrail", "railing", "baluster", "newel post",
                 "wall sconce", "light fixture", "stair gate"],
        "staging": ["runner", "wall art", "mirror", "bench"]
    },

    "basement": {
        "core": ["water heater", "furnace", "boiler", "electrical panel",
                 "support column", "support beam", "duct", "pipe",
                 "laundry hookups", "egress window", "floor drain", "fan"],
        "staging": ["shelving unit", "workbench", "storage rack", "tool chest",
                    "storage bin", "folding table", "dehumidifier"]
    },

    "attic": {
        "core": ["rafter", "beam", "truss", "vent", "attic fan", "duct", "chimney",
                 "hatch", "pull-down ladder", "skylight"],
        "staging": ["storage bin", "box", "shelving unit"]
    },

    "garage": {
        "core": ["garage door", "door opener", "track", "workbench", "cabinet", "shelf",
                 "electrical panel", "water heater", "furnace", "laundry hookups",
                 "attic ladder", "hose bib", "ev charger", "floor drain"],
        "staging": ["tool chest", "storage bin", "ladder", "bicycle", "lawn mower",
                    "storage rack", "freezer", "refrigerator", "trash can"]
    },

    "closet": {
        "core": ["shelf", "hanger rod", "drawer", "shoe rack", "organizer", "mirror",
                 "light fixture", "bi-fold door", "sliding door", "safe"],
        "staging": ["clothes", "hanger", "storage bin", "laundry basket", "luggage"]
    },

    "pantry": {
        "core": ["shelf", "cabinet", "drawer", "pull-out basket", "spice rack",
                 "wine rack", "pantry door", "light fixture", "freezer", "appliance shelf"],
        "staging": ["can", "jar", "bottle", "container", "bin", "crate"]
    },

    # Exterior scenes
    "exterior_front": {
        "core": ["front door", "garage door", "porch", "porch light", "column",
                 "railing", "gutter", "downspout", "chimney", "mailbox", "walkway",
                 "stair", "bay window", "security camera", "house number"],
        "staging": ["bench", "chair", "planter", "porch swing", "doormat", "wreath"]
    },

    "exterior_back": {
        "core": ["patio", "deck", "back door", "sliding door", "porch", "porch light",
                 "fence", "gate", "gutter", "downspout", "shed", "stair", "hose bib"],
        "staging": ["lounge chair", "dining chair", "dining table", "umbrella",
                    "grill cart", "storage box", "hammock"]
    },

    "exterior_side": {
        "core": ["side gate", "fence", "gutter", "downspout", "hose bib",
                 "electric meter", "gas meter", "hvac condenser", "utility box",
                 "satellite dish", "vent", "crawlspace door"],
        "staging": ["trash bin", "storage tote"]
    },

    "yard": {
        "core": ["fence", "gate", "shed", "sprinkler head", "irrigation control",
                 "playset", "swing set", "raised bed", "fire pit", "retaining wall",
                 "pathway", "composter"],
        "staging": ["patio chair", "table", "umbrella", "trampoline", "garden bench",
                    "hammock"]
    },

    "patio": {
        "core": ["pergola", "gazebo", "awning", "outdoor kitchen", "grill island",
                 "fire pit", "ceiling fan", "railing", "stair", "privacy screen", "heater"],
        "staging": ["chair", "table", "sofa", "coffee table", "umbrella", "storage box"]
    },

    "deck": {
        "core": ["railing", "stair", "post", "baluster", "gate", "pergola", "awning",
                 "lattice", "bench", "skirting"],
        "staging": ["chair", "table", "umbrella", "storage box", "planter"]
    },

    "balcony": {
        "core": ["railing", "privacy screen", "awning", "sliding door", "drain",
                 "ceiling fan", "guard rail", "support bracket", "sunshade"],
        "staging": ["chair", "table", "planter", "side table"]
    },

    "driveway": {
        "core": ["garage door", "carport", "gate", "lamp post", "mailbox", "drain",
                 "retaining wall", "walkway", "curb cut"],
        "staging": ["trash bin", "basketball hoop", "portable gate"]
    },

    "pool_area": {
        "core": ["pool", "spa", "hot tub", "ladder", "handrail", "diving board",
                 "slide", "pool light", "filter", "pump", "heater", "skimmer",
                 "pool cover", "fence", "gate"],
        "staging": ["lounge chair", "chair", "table", "umbrella", "storage box", "cabana"]
    },

    "garden": {
        "core": ["raised bed", "trellis", "greenhouse", "irrigation line",
                 "drip emitter", "hose reel", "fence", "gate", "planter box",
                 "rain barrel", "garden shed"],
        "staging": ["bench", "table", "chair", "wheelbarrow", "planter pot", "storage bin"]
    },

    # Special/other (no staging concept for these)
    "floor_plan": {
        "core": ["wall", "door", "window", "stair", "dimension", "label", "arrow",
                 "north arrow", "scale bar", "room label", "closet", "appliance symbol"],
        "staging": []
    },

    "aerial_view": {
        "core": ["roof", "chimney", "solar panel", "driveway", "street", "sidewalk",
                 "fence", "pool", "outbuilding", "carport", "hvac condenser",
                 "septic lid", "well head", "property gate"],
        "staging": []
    },

    "street_view": {
        "core": ["house", "sidewalk", "driveway", "street", "curb", "crosswalk",
                 "stop sign", "streetlight", "fire hydrant", "mailbox", "utility pole",
                 "traffic signal"],
        "staging": []
    },

    "unknown": {
        "core": ["cabinet", "countertop", "sink", "faucet", "appliance", "light fixture",
                 "ceiling fan", "railing", "fence", "garage door", "water heater",
                 "electrical panel", "vent", "smoke detector"],
        "staging": ["sofa", "chair", "table", "bed", "dresser", "television",
                    "lamp", "rug", "bar stool"]
    }
}

# Objects related to condition/defects (optional)
CONDITION_KEYWORDS = [
    "crack", "stain", "damage", "hole", "scratch", "dent",
    "rust", "mold", "water damage", "peeling paint", "broken"
]

# Surfaces we NEVER want to draw bounding boxes on
# These are generic surface labels that produce huge, low-value chips
BANNED_SURFACE_LABELS = {
    # Exact matches only - single words
    "ceiling",
    "ceilings",
    "wall",
    "walls",
    "floor",
    "floors",
    "flooring",
    "carpet",
    # Compound surface terms we explicitly ban
    "hardwood_floor",
    "wood_floor",
    "laminate_floor",
    "basement_floor",
    "basement_walls",
    "tile_floor",
    "vinyl_floor",
    "floor_tiles",
    "carpet_floor",
    "concrete_floor",
    "wood_flooring",
    "laminate_flooring",
}

# Fixtures that contain surface words but ARE valid targets
# These override the token-based check
ALLOWED_FIXTURE_PATTERNS = {
    "floor_lamp",
    "floor_vent",
    "floor_drain",
    "floor_register",
    "ceiling_fan",
    "ceiling_light",
    "ceiling_fixture",
    "wall_sconce",
    "wall_mount",
    "wall_outlet",
    "wall_switch",
    "wall_vent",
}


def _is_banned_surface_label(label: str) -> bool:
    """
    Return True if label is a generic wall/ceiling/floor surface.

    We want to ban:
    - Single-word surfaces: "wall", "floor", "ceiling", "carpet"
    - Surface compound terms: "tile_floor", "hardwood_floor", "basement_walls"

    We do NOT want to ban fixtures that happen to contain surface words:
    - "floor_lamp", "ceiling_fan", "wall_sconce" are valid fixtures
    """
    if not label:
        return False
    norm = _sanitize_term(label).replace(" ", "_")

    # Check explicit allowlist first - these are valid fixtures
    if norm in ALLOWED_FIXTURE_PATTERNS:
        return False

    # Check if the full normalized label is in the banned set
    if norm in BANNED_SURFACE_LABELS:
        return True

    # For single-token labels, check if it's a banned surface word
    tokens = norm.split("_")
    if len(tokens) == 1:
        return norm in {"ceiling", "ceilings", "wall", "walls", "floor", "floors", "flooring", "carpet"}

    # For multi-token labels, only ban if the LAST token indicates it's a surface
    # e.g., "tile_floor" (surface) vs "floor_lamp" (fixture)
    # The pattern: <material>_<surface> should be banned
    #              <surface>_<fixture> should be allowed
    surface_words = {"floor", "floors", "flooring", "wall", "walls", "ceiling", "ceilings", "carpet"}
    last_token = tokens[-1]

    # If it ends with a surface word, it's probably a surface type
    if last_token in surface_words:
        return True

    return False


def _filter_banned_surface_targets(targets: List[Dict]) -> List[Dict]:
    """
    Filter targets to remove generic surfaces from becoming GroundingDINO terms.

    - Drop targets whose LABEL is a banned surface (wall, floor, ceiling, etc.)
    - Strip banned surface words from synonyms (but keep the target if label is good)

    This is less aggressive than before: a good label like "cabinet_door" won't be
    dropped just because the VLM included "kitchen wall" in the synonyms.
    """
    filtered: List[Dict] = []
    for t in targets or []:
        if not isinstance(t, dict):
            continue

        label = t.get("label", "")

        # Only check the label for ban decision
        if _is_banned_surface_label(label):
            continue

        # Clean synonyms: strip any banned surface words, but don't drop the target
        synonyms = t.get("synonyms", [])
        if isinstance(synonyms, list):
            clean_synonyms = [s for s in synonyms if not _is_banned_surface_label(str(s))]
            t["synonyms"] = clean_synonyms

        filtered.append(t)

    return filtered


# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class VLMDebugOutput:
    """Container for a single VLM call's debug info."""
    pass_name: str
    prompt_summary: str  # Brief description of what was asked
    raw_response: str
    parsed_result: Optional[Dict] = None  # Parsed JSON if applicable
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class VLMDebugOutputs:
    """Container for ALL VLM outputs from a single image classification."""
    pass1_scene_summary: Optional[VLMDebugOutput] = None
    pass2a_freeform_notes: Optional[VLMDebugOutput] = None
    pass2b_defect_structuring: Optional[VLMDebugOutput] = None
    pass3_staging_detection: Optional[VLMDebugOutput] = None
    pass4_target_planning: Optional[VLMDebugOutput] = None


Presence = Literal["yes", "no", "uncertain"]
SeverityLabel = Literal["none", "minor_repair", "moderate_repair", "full_replacement"]


@dataclass
class NLIssue:
    description: str
    rough_category: str
    location_hint: str = ""


@dataclass
class CatalogFlag:
    present: Presence
    evidence: str = ""
    # severity of work implied if present == "yes"
    severity: SeverityLabel = "none"


@dataclass
class DefectAnalysis:
    """
    Output of Pass 2b (text-only defect structuring).

    This pass is intentionally limited to:
    - issues_natural_language: 0–5 calm, factual issues extracted from the freeform notes
    - catalog_flags: presence / severity / evidence for every catalog issue_id
    - freeform_notes: the raw text from Pass 2a (for debugging / UI)
    """
    issues_natural_language: List[NLIssue]
    catalog_flags: Dict[str, CatalogFlag]
    freeform_notes: str = ""  # Raw text from Pass 2a


@dataclass
class VisualAnchor:
    issue_id: str
    support_object_label: str
    bbox: List[float]
    confidence: float


@dataclass
class SceneSummary:
    scene: str
    image_summary: str
    overall_impression: str


@dataclass
class SceneClassification:
    scene: str
    is_staged: bool
    reasoning: str
    targets: List[Dict]
    gdino_terms: List[str]
    keywords: List[str]
    groundingdino_prompt: str
    processing_time: float = 0.0
    prompt_version: str = PROMPT_VERSION
    scene_policy_version: str = "v1"
    raw_response: Optional[str] = None  # classification raw (legacy, kept for compat)
    planner_raw_response: Optional[str] = None  # planner raw (legacy)
    error: Optional[str] = None
    image_summary: str = ""
    overall_impression: str = ""
    issues_natural_language: List[NLIssue] = field(default_factory=list)
    catalog_flags: Dict[str, CatalogFlag] = field(default_factory=dict)
    issue_visual_anchors: List[VisualAnchor] = field(default_factory=list)
    freeform_defect_notes: str = ""  # Raw text from Pass 2a
    vlm_outputs: Optional[VLMDebugOutputs] = None  # All VLM outputs for debugging


# ── Helper Functions ─────────────────────────────────────────────────────────
def _to_bool(v) -> bool:
    """Convert various representations to boolean."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        return s in {"true", "yes", "1", "staged", "furnished"}
    return bool(v)  # or return False to be conservative


def _sanitize_term(s: str) -> str:
    """Sanitize a term for detection: lowercase, strip, collapse whitespace."""
    return " ".join(str(s).lower().strip().split())


def dedup_preserving_order(seq: List[str]) -> List[str]:
    """Remove duplicates from list while preserving order."""
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_issue_catalog(path: Path) -> dict:
    """
    Load the issue catalog from JSON.

    Returns a dict with at least keys:
      - 'defect_issues': list of issue dicts
      - 'opportunity_flags': list of issue dicts
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Loaded issue catalog from {path}")
    except FileNotFoundError:
        logger.warning(f"Issue catalog not found at {path}; using empty catalog.")
        data = {}
    except Exception as exc:
        logger.error(f"Failed to load issue catalog from {path}: {exc}")
        data = {}

    return {
        "defect_issues": data.get("defect_issues", []),
        "opportunity_flags": data.get("opportunity_flags", []),
    }


def get_catalog_ids(issue_catalog: dict) -> List[str]:
    """Flatten all issue ids from defect_issues and opportunity_flags."""
    ids = []
    for key in ("defect_issues", "opportunity_flags"):
        for item in issue_catalog.get(key, []) or []:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
    return ids


def _serialize_catalog_flags(flags: Dict[str, CatalogFlag]) -> Dict[str, Dict[str, str]]:
    return {iid: asdict(flag) for iid, flag in (flags or {}).items()}


def _clean_keywords(keywords: List[str]) -> List[str]:
    """Clean and deduplicate keywords."""
    return dedup_preserving_order([_sanitize_term(kw) for kw in keywords if kw])


def _extract_json(text: str) -> Optional[dict]:
    """
    Extract a single JSON object from an LLM response.

    Assumptions:
    - Response is either pure JSON, or
      wrapped in ```json ... ``` fences, possibly with minor extra text.
    - We do NOT try to "repair" truly broken / truncated JSON.
    """
    if not text:
        return None

    s = text.strip()

    # Strip markdown code fences: ```json\n...\n``` or ```\n...\n```
    if s.startswith("```"):
        # Drop the first line (``` or ```json)
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        # Strip trailing ```
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    # 1) Try to parse the whole string as JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) Fallback: grab from first '{' to last '}' and try that
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start: end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None


def encode_image_to_b64(image_path: Path) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def build_scene_summary_prompt() -> str:
    """
    Build the prompt for VLM Pass 1: scene classification + overall impression.
    This prompt should NOT mention defects, targets, chips, or the issue catalog.
    """
    return (
        "You are analyzing a single real-estate listing photo.\n\n"
        "Use ONLY what is clearly visible in the image. Do NOT guess about hidden issues, exact ages, renovation dates, or anything that is not visually obvious.\n\n"
        "Your job for this image is to:\n\n"
        "Set \"scene\" to a simple room/area type such as:\n"
        "\"kitchen\", \"bathroom\", \"living_room\", \"bedroom\", \"dining_room\",\n"
        "\"hallway\", \"laundry_room\", \"basement\", \"garage\", \"exterior_front\",\n"
        "\"exterior_back\", \"yard\", etc.\n\n"
        "If the space is ambiguous, say unknown.\n"
        "Set \"overall_impression\" to 2–3 concise sentences written for a buyer or investor that:\n\n"
        "Briefly describe what the photo shows.\n"
        "Highlight the main positives (e.g., natural light, modern/updated finishes, clean presentation, good layout, spaciousness).\n"
        "Highlight any clear negatives (e.g., visible wear or damage, clutter, dark/cramped feel, obviously dated finishes, mismatched updates).\n"
        "You may use high-level judgments like:\n\n"
        "\"dated\", \"basic\", \"builder-grade\", \"somewhat modern\", \"recently updated\",\n"
        "\"move-in ready\", \"shows wear\", \"needs cosmetic refresh\" BUT only when these are clearly supported by what you can see (finishes, fixtures, appliances, flooring, overall condition). Describe the visual impression, not the renovation history.\n\n"
        "Important:\n\n"
        "Focus on features that affect aesthetics or salability.\n"
        "Do NOT comment on tiny personal items like fridge magnets, papers, or small toys unless they create noticeable clutter that affects how the space presents.\n"
        "Do NOT mention brand names or speculate about price or quality of materials.\n"
        "Return a SINGLE JSON object with this exact structure and valid JSON syntax (double quotes around keys and strings):\n\n"
        "{\n"
        "\"scene\": \"<string room/area label>\",\n"
        "\"overall_impression\": \"<2-3 sentence description of the main positives and negatives, based only on what is visible>\"\n"
        "}\n\n"
        "Do NOT wrap the JSON in backticks or markdown; output raw JSON only."
    )


def _format_catalog_for_prompt(issue_catalog: dict) -> str:
    """Summarize catalog ids, names, and descriptions for prompt context."""
    parts = []
    for section in ("defect_issues", "opportunity_flags"):
        section_items = issue_catalog.get(section, []) or []
        if not section_items:
            continue
        parts.append(f"{section} ({len(section_items)} items):")
        for item in section_items:
            if not isinstance(item, dict):
                continue
            iid = item.get("id", "")
            name = item.get("name", "")
            desc = item.get("description", "")
            category = item.get("category", "")
            severity = item.get("severity", "")
            parts.append(f"- {iid}: {name} [{category}/{severity}] - {desc}")
    return "\n".join(parts)


def _format_issues_for_planner(issues: List[NLIssue]) -> str:
    """
    Convert issues_natural_language into a numbered list string for the planner prompt.

    Each line should include:
      - description
      - rough_category
      - location_hint

    If there are no issues, return a short line indicating that.
    """
    if not issues:
        return "None explicitly detected; rely primarily on scene policies and the image."

    lines = []
    for i, iss in enumerate(issues, 1):
        desc = iss.description or ""
        category = iss.rough_category or "unspecified"
        loc = iss.location_hint or "unspecified area"
        lines.append(
            f"{i}. {desc} (category={category}, location={loc})"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1: Scene Summary
# ═══════════════════════════════════════════════════════════════════════════════

def run_scene_summary(image_bytes: bytes, model_name: str, lm_studio_url: str = LM_STUDIO_URL,
                      http_client=requests) -> tuple[SceneSummary, VLMDebugOutput]:
    """
    Run VLM Pass 1 for a single image:
    - Calls the VLM with the Pass 1 prompt.
    - Parses the JSON response.
    - Returns a SceneSummary object AND the debug output.
    """
    t0 = time.time()
    logger.debug("  Building scene summary prompt...")
    prompt = build_scene_summary_prompt()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a property photo analyzer. Respond with valid JSON only."
        },
        {"role": "user", "content": content}
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False
    }

    debug_output = VLMDebugOutput(
        pass_name="pass1_scene_summary",
        prompt_summary="Scene classification + overall impression (JSON)",
        raw_response="",
    )

    try:
        logger.debug(f"  Sending scene summary request to {lm_studio_url}...")
        resp = http_client.post(
            f"{lm_studio_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        if resp.status_code != 200:
            logger.error(f"  Scene summary API error: HTTP {resp.status_code}")
            debug_output.error = f"API error: HTTP {resp.status_code}"
            debug_output.duration_seconds = time.time() - t0
            raise RuntimeError(f"API error: HTTP {resp.status_code} - {resp.text[:200]}")

        data = resp.json()
        raw_response = data["choices"][0]["message"]["content"]
        logger.debug(f"  Scene summary raw response length: {len(raw_response)} chars")

        debug_output.raw_response = raw_response
        debug_output.duration_seconds = time.time() - t0

        parsed = _extract_json(raw_response) or {}
        debug_output.parsed_result = parsed

        scene = str(parsed.get("scene", "unknown"))
        overall_impression = str(parsed.get("overall_impression", "")).strip()

        logger.debug(f"  Scene summary parsed: scene='{scene}'")

        return SceneSummary(
            scene=scene,
            image_summary="",  # no longer used; kept for backward compatibility
            overall_impression=overall_impression,
        ), debug_output

    except Exception as e:
        logger.error(f"  VLM Pass 1 (scene summary) failed: {e}")
        debug_output.error = str(e)
        debug_output.duration_seconds = time.time() - t0
        return SceneSummary(scene="unknown", image_summary="", overall_impression=""), debug_output


def _normalize_presence(value: str) -> Presence:
    v = str(value).strip().lower()
    if v in {"yes", "no", "uncertain"}:
        return v  # type: ignore[return-value]
    return "uncertain"


def _normalize_severity(value: str) -> SeverityLabel:
    """
    Map noisy model severity strings into a small fixed set:
    'none', 'minor_repair', 'moderate_repair', 'full_replacement'.
    """
    v = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    mapping = {
        # no work / no meaningful issue
        "none": "none",
        "na": "none",
        "n_a": "none",
        "not_applicable": "none",
        "no_issue": "none",
        "no_repair": "none",

        # minor repair
        "minor": "minor_repair",
        "small": "minor_repair",
        "spot_repair": "minor_repair",
        "minor_repair": "minor_repair",

        # moderate repair
        "moderate": "moderate_repair",
        "medium": "moderate_repair",
        "partial_repair": "moderate_repair",
        "moderate_repair": "moderate_repair",

        # full replacement
        "full": "full_replacement",
        "replacement": "full_replacement",
        "full_replacement": "full_replacement",
        "replace": "full_replacement",
    }

    return mapping.get(v, "none")


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2a: Freeform Defect Notes (Vision + Simple Question → Plain Text)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2A_USER_PROMPT = "What issues do you see that a realtor might want to know about? Dont be overdramatic. There might not be any issues at all"


def run_freeform_defect_notes(
        image_bytes: bytes,
        scene: str,
        model_name: str,
        lm_studio_url: str = LM_STUDIO_URL,
        http_client=requests,
) -> tuple[str, VLMDebugOutput]:
    """
    Run VLM Pass 2a: Freeform defect notes.

    This is a simple vision call with the image and a plain-text question.
    Returns plain text (not JSON) - a narrative of what the VLM thinks are issues.

    Args:
        image_bytes: Raw bytes of the image
        scene: Scene label from Pass 1 (for context, not included in prompt per spec)
        model_name: LM Studio model name
        lm_studio_url: LM Studio endpoint
        http_client: HTTP client (for testing)

    Returns:
        Tuple of (freeform_notes: str, debug_output: VLMDebugOutput)
    """
    t0 = time.time()
    logger.debug(f"  Pass 2a: Sending freeform defect notes request...")

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    # Per the spec: exact user prompt, no extra instructions
    content = [
        {"type": "text", "text": PASS_2A_USER_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,  # Keep low for consistency
        "max_tokens": 1024,
        "stream": False,
    }

    debug_output = VLMDebugOutput(
        pass_name="pass2a_freeform_notes",
        prompt_summary=f"Freeform defect notes (plain text) for scene={scene}",
        raw_response="",
    )

    try:
        resp = http_client.post(
            f"{lm_studio_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=40,
        )
        if resp.status_code != 200:
            logger.error(f"  Pass 2a API error: HTTP {resp.status_code}")
            debug_output.error = f"API error: HTTP {resp.status_code}"
            debug_output.duration_seconds = time.time() - t0
            return "", debug_output

        data = resp.json()
        raw_response = data["choices"][0]["message"]["content"]
        logger.debug(f"  Pass 2a raw response length: {len(raw_response)} chars")

        debug_output.raw_response = raw_response
        debug_output.duration_seconds = time.time() - t0
        # No parsed_result for plain text

        return raw_response.strip(), debug_output

    except Exception as e:
        logger.error(f"  Pass 2a (freeform notes) failed: {e}")
        debug_output.error = str(e)
        debug_output.duration_seconds = time.time() - t0
        return "", debug_output


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2b: Defect Structuring (Text-only → Structured JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def build_defect_structuring_prompt(scene: str, freeform_notes: str, issue_catalog: dict) -> str:
    catalog_text = _format_catalog_for_prompt(issue_catalog)
    catalog_ids = get_catalog_ids(issue_catalog)
    catalog_list_str = ", ".join(catalog_ids)

    return f"""You are a text analysis assistant that structures freeform property inspection notes into a conservative, factual JSON format.

CONTEXT:
- Scene type: {scene}
- Issue catalog IDs: {catalog_list_str}

FREEFORM NOTES (from a vision model looking at a property photo):
---
{freeform_notes}
---

ISSUE CATALOG REFERENCE:
{catalog_text}

YOUR TASK:
Convert the freeform notes above into structured JSON. Be VERY CONSERVATIVE:
- Treat the notes as potentially noisy or dramatic.
- Only create issues for things that would typically require money, time, or a contractor to address (repair, replacement, or significant maintenance).
- Layout, size, and preference items (e.g. small bedroom, single-car garage, mailbox at curb, houses close together) are NOT issues unless the notes clearly describe a safety concern or defect.
- Avoid over-interpreting speculative language ("might be water damage", "could be unsafe") unless there is a clearly described visible condition.
- If the notes say "no issues" or similar, output empty lists and mark all catalog_flags as "no".

OUTPUT FORMAT (strict JSON only, no markdown):
{{
  "issues_natural_language": [
    {{
      "description": "<string: calm factual description of a visible, concrete issue>",
      "rough_category": "<cosmetic|moisture|structure|systems|exterior|opportunity>",
      "location_hint": "<string: where in the photo>"
    }}
  ],
  "catalog_flags": {{
    "<issue_id>": {{
      "present": "yes|no|uncertain",
      "severity": "none|minor_repair|moderate_repair|full_replacement",
      "evidence": "<string or empty>"
    }}
  }}
}}

RULES:
1. issues_natural_language:
   - Only include items where there is a clearly visible condition or defect (e.g., staining, damage, heavy wear, missing component, obvious neglect).
   - Do NOT create issues for normal, expected conditions such as:
     * a standard-size garage door,
     * a typical sloped driveway with no visible damage,
     * a mailbox at the curb or property line,
     * houses being close together in a subdivision.
   - Light cosmetic yard care (slightly patchy grass, a few dry spots) should appear at most as a single low-severity cosmetic issue.

2. catalog_flags:
   - Include EVERY issue_id from the catalog ({catalog_list_str}).
   - present:
     * "yes" only if the notes clearly describe that specific issue as visible in the image.
     * If the notes use words like "may", "might", "could", "potentially", or "possible" without describing an actual visible defect, set present="uncertain" and severity="none".
     * "no" if the notes explicitly say the issue is not present.
   - If present is "no" or "uncertain", force severity="none".
   - Only use "moderate_repair" or "full_replacement" when the notes clearly imply substantial work, not just age or preference.
   - evidence: short phrase from notes, or empty.

Respond with raw JSON only. No markdown, no backticks, no explanations."""


def run_defect_structuring(
        scene: str,
        freeform_notes: str,
        issue_catalog: dict,
        model_name: str,
        lm_studio_url: str = LM_STUDIO_URL,
        http_client=requests,
) -> tuple[DefectAnalysis, VLMDebugOutput]:
    """
    Run VLM Pass 2b: Text-only structuring of freeform notes into DefectAnalysis.

    This call does NOT include the image - it's purely text-based.

    Args:
        scene: Scene label from Pass 1
        freeform_notes: Plain text notes from Pass 2a
        issue_catalog: The issue catalog dict
        model_name: LM Studio model name
        lm_studio_url: LM Studio endpoint
        http_client: HTTP client (for testing)

    Returns:
        Tuple of (DefectAnalysis, VLMDebugOutput)
    """
    t0 = time.time()
    logger.debug(f"  Pass 2b: Structuring freeform notes for scene='{scene}'...")

    catalog_ids = get_catalog_ids(issue_catalog)
    prompt = build_defect_structuring_prompt(scene, freeform_notes, issue_catalog)

    # Text-only: no image in this call
    messages = [
        {"role": "system", "content": "You are a structured data extraction assistant. Respond with valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False,
    }

    debug_output = VLMDebugOutput(
        pass_name="pass2b_defect_structuring",
        prompt_summary=f"Text-only structuring of freeform notes → JSON for scene={scene}",
        raw_response="",
    )

    default_flags = {
        iid: CatalogFlag(present="uncertain", evidence="", severity="none")
        for iid in catalog_ids
    }

    try:
        logger.debug(f"  Sending defect structuring request to {lm_studio_url}...")
        resp = http_client.post(
            f"{lm_studio_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=40,
        )
        if resp.status_code != 200:
            logger.error(f"  Pass 2b API error: HTTP {resp.status_code}")
            debug_output.error = f"API error: HTTP {resp.status_code}"
            debug_output.duration_seconds = time.time() - t0
            return DefectAnalysis(
                issues_natural_language=[],
                catalog_flags=default_flags,
                freeform_notes=freeform_notes,
            ), debug_output

        data = resp.json()
        raw_response = data["choices"][0]["message"]["content"]
        logger.debug(f"  Pass 2b raw response length: {len(raw_response)} chars")

        debug_output.raw_response = raw_response
        debug_output.duration_seconds = time.time() - t0

        parsed = _extract_json(raw_response) or {}
        debug_output.parsed_result = parsed

        # Parse issues_natural_language
        nl_issues: List[NLIssue] = []
        for item in parsed.get("issues_natural_language", []) or []:
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description", "")).strip()
            category = str(item.get("rough_category", "")).strip()
            location_hint = str(item.get("location_hint", "")).strip()
            if desc:
                nl_issues.append(NLIssue(description=desc, rough_category=category, location_hint=location_hint))

        logger.debug(f"  Found {len(nl_issues)} natural language issues")

        # Parse catalog_flags ensuring all ids exist
        parsed_flags = parsed.get("catalog_flags", {}) or {}
        catalog_flags: Dict[str, CatalogFlag] = {}

        issues_present = 0
        for iid in catalog_ids:
            if isinstance(parsed_flags, dict) and iid in parsed_flags:
                entry = parsed_flags.get(iid, {}) or {}

                if isinstance(entry, dict):
                    present_raw = entry.get("present", "uncertain")
                    evidence = str(entry.get("evidence", "")).strip()
                    severity_raw = entry.get("severity", "none")
                else:
                    present_raw = "uncertain"
                    evidence = ""
                    severity_raw = "none"

                present = _normalize_presence(present_raw)
                severity = _normalize_severity(severity_raw)

                # Enforce: if not clearly present, there is no work to estimate
                if present != "yes":
                    severity = "none"
                else:
                    issues_present += 1

                catalog_flags[iid] = CatalogFlag(
                    present=present,
                    evidence=evidence,
                    severity=severity,
                )
            else:
                catalog_flags[iid] = CatalogFlag(
                    present="uncertain",
                    evidence="",
                    severity="none",
                )

        logger.debug(f"  Catalog flags: {issues_present}/{len(catalog_ids)} issues marked present")

        return DefectAnalysis(
            issues_natural_language=nl_issues,
            catalog_flags=catalog_flags,
            freeform_notes=freeform_notes,
        ), debug_output

    except Exception as e:
        logger.error(f"  Pass 2b (defect structuring) failed: {e}")
        debug_output.error = str(e)
        debug_output.duration_seconds = time.time() - t0
        return DefectAnalysis(
            issues_natural_language=[],
            catalog_flags=default_flags,
            freeform_notes=freeform_notes,
        ), debug_output


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 3: Staging Detection
# ═══════════════════════════════════════════════════════════════════════════════

def build_staging_prompt(scene: str) -> str:
    """Build prompt for staging detection only (scene already determined)."""

    # Get scene-specific staging keywords to help the VLM
    scene_data = SCENE_KEYWORDS.get(scene, SCENE_KEYWORDS.get("unknown", {}))
    staging_items = scene_data.get("staging", [])
    staging_examples = ", ".join(staging_items[:8]) if staging_items else "furniture, decor, movable items"

    return (
        "You are an expert at analyzing real estate property photos.\n\n"
        f"CONTEXT: This photo has been identified as a '{scene}'.\n\n"
        "TASK: Determine if the photo is STAGED (furnished) or VACANT (empty).\n\n"
        "STAGING DETECTION:\n"
        "- STAGED: Photo shows furniture, decor, or movable items\n"
        f"  Examples for {scene}: {staging_examples}\n"
        "- VACANT: Photo shows only permanent fixtures (cabinets, appliances, built-ins)\n"
        "- Consider: Is there furniture? Are there decorative items? Is it furnished?\n\n"
        "RULES:\n"
        "- If uncertain, pick the most likely option\n"
        "- Keep 'reasoning' VERY short (one brief phrase, max ~15 words)\n"
        "- Focus ONLY on staging status, not scene identification\n\n"
        "RESPOND ONLY WITH JSON (no extra text):\n"
        "{\n"
        '  "is_staged": true/false,\n'
        '  "reasoning": "very short phrase explaining why you chose staged or vacant"\n'
        "}"
    )


def run_staging_detection(
        image_bytes: bytes,
        scene: str,
        model_name: str,
        lm_studio_url: str = LM_STUDIO_URL,
        http_client=requests,
) -> tuple[bool, str, VLMDebugOutput]:
    """
    Run VLM Pass 3: Staging detection.

    Returns:
        Tuple of (is_staged: bool, reasoning: str, debug_output: VLMDebugOutput)
    """
    t0 = time.time()
    logger.debug(f"  Pass 3: Staging detection for scene='{scene}'...")

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    prompt = build_staging_prompt(scene)

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]

    messages = [
        {"role": "system", "content": "You are a property photo analyzer. Respond with valid JSON only."},
        {"role": "user", "content": content},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False,
    }

    debug_output = VLMDebugOutput(
        pass_name="pass3_staging_detection",
        prompt_summary=f"Staging detection (is_staged + reasoning) for scene={scene}",
        raw_response="",
    )

    try:
        resp = http_client.post(
            f"{lm_studio_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            debug_output.error = f"API error: HTTP {resp.status_code}"
            debug_output.duration_seconds = time.time() - t0
            return True, "API error, defaulting to staged", debug_output

        data = resp.json()
        raw_response = data["choices"][0]["message"]["content"]

        debug_output.raw_response = raw_response
        debug_output.duration_seconds = time.time() - t0

        parsed = _extract_json(raw_response) or {}
        debug_output.parsed_result = parsed

        is_staged = _to_bool(parsed.get("is_staged", True))
        reasoning = str(parsed.get("reasoning", ""))[:200]

        return is_staged, reasoning, debug_output

    except Exception as e:
        logger.error(f"  Pass 3 (staging detection) failed: {e}")
        debug_output.error = str(e)
        debug_output.duration_seconds = time.time() - t0
        return True, "Error, defaulting to staged", debug_output


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 4: Issue-Driven Target Planning
# ═══════════════════════════════════════════════════════════════════════════════

def get_scene_policy(scene: str) -> str:
    """
    Return a per-scene policy block for the planner prompt.
    Provides concrete policies for kitchen, bathroom, and exterior_front.
    Returns minimal general policy for all other scenes.
    """
    policies = {
        "kitchen": """- Focus: sink/faucet, countertop material, range/hood, refrigerator, cabinet age/condition (peeling/dated), backsplash grout/caulk.
- Defects: water stains (ceiling/wall), mold/mildew, damaged grout/caulk, burn marks, delaminated veneer.
- Positives: quartz/granite counters, updated appliances, tile backsplash.
- Exclude: decor (fruit bowls, staging) unless it indicates condition.""",

        "bathroom": """- Focus: toilet, vanity/sink/faucet, shower/tub, tile/grout/caulk, ventilation fan, GFCI outlet (proxy).
- Defects: mildew/mold, water damage, cracked tile, missing caulk, rust.
- Positives: updated vanity, modern tile, frameless shower.
- Exclude: towels/props unless relevant to condition.""",

        "exterior_front": """- Focus: roof shingle condition, chimney cracks, gutters/downspouts, fascia/soffit, siding damage, windows, front door, house number, walkway/driveway cracks.
- Positives: new roof, fiber-cement siding, updated windows.
- Heuristics: house number small near entry; walkway touches bottom edge; chimney is vertical brick column."""
    }

    # Return specific policy or general fallback
    if scene in policies:
        return policies[scene]
    else:
        return """- Focus: permanent fixtures, structural elements, and condition indicators.
- Defects: damage, wear, aging, water stains, cracks.
- Positives: updates, quality materials, good condition.
- Exclude: movable items unless they indicate property condition."""


def build_planner_prompt(scene: str, is_staged: bool, policy_block: str,
                         max_targets: int, allow_free: int,
                         issues: List[NLIssue]) -> str:
    """
    Build the planner prompt requesting structured targets.

    The planner is now ISSUES-DRIVEN: it uses the defect/issue analysis from Pass 2
    as its primary source for selecting targets, with scene policy as helper context.
    """
    issues_text = _format_issues_for_planner(issues)

    tmpl = (
        "You are an expert at analyzing real-estate listing photos.\n\n"

        "GOAL\n"
        f"- Propose up to {{max_targets}} high-value TARGETS to draw bounding boxes around.\n"
        "- A TARGET is a physical surface or fixture that affects price, marketability, or inspection risk.\n"
        f"- Use the issue list below as your PRIMARY guide. You may add up to {{allow_free}} extra targets only for clearly visible, high-severity problems.\n\n"

        "SCENE\n"
        f"- scene: {{scene}}\n"
        f"- is_staged: {{is_staged}}\n\n"

        "ISSUES ALREADY DETECTED (from a separate analysis pass):\n"
        "{issues_text}\n\n"
        "Interpretation:\n"
        "- Do NOT invent new problems that are not clearly implied by the issue text.\n"
        "- For each issue, pick ONE or a FEW specific surfaces/fixtures that best represent it.\n"
        "- Prefer concrete objects like 'cabinet_door', 'countertop', 'faucet' over vague areas.\n\n"

        "SCENE POLICY HINTS\n"
        "{policy_block}\n\n"

        "LABEL RULES\n"
        "- `label` is the physical object/surface only, not the problem description.\n"
        "- snake_case, lowercase, 1–2 tokens, no repeated tokens.\n"
        "- Do NOT include scene words ('kitchen', 'bathroom', etc.).\n"
        "- Do NOT include adjectives or condition words (cracked, stained, dirty, dated, mold, etc.).\n"
        "- NEVER use generic surfaces: ceiling, ceilings, wall, walls, floor, floors, flooring, carpet, "
        "hardwood_floor, laminate_floor, tile_floor, floor_tiles, carpet_floor, basement_floor, basement_walls.\n\n"

        "SYNONYMS & REASON\n"
        "- `synonyms`: 2–5 short nouns/noun-phrases someone might use for the same object.\n"
        "- `reason` MUST be consistent with the issues above.\n"
        "- If the issues are vague (e.g. 'outdated', 'older style'), keep `reason` vague.\n"
        "- Do NOT invent specific defects like stains, leaks, cracks, mold, discoloration, grime, or warping "
        "if they are not explicitly described in the issues.\n\n"

        "TARGET SELECTION\n"
        "- Focus on a small set of high-impact targets tied to the issues.\n"
        "- If several issues affect the same object, use ONE target and summarize the main concern in `reason`.\n"
        "- Ignore decor, personal items, and clutter.\n\n"

        "SELF-CHECK BEFORE ANSWERING\n"
        "- For every target, verify:\n"
        "  * `label` follows the rules and is not a generic surface.\n"
        "  * `reason` does NOT mention a specific defect type that is missing from the issue list.\n\n"

        "OUTPUT FORMAT (RAW JSON ONLY)\n"
        "{{\n"
        '  "scene": "<one from known scene list>",\n'
        '  "is_staged": true/false,\n'
        '  "reasoning": "≤200 chars on why these targets matter overall",\n'
        '  "targets": [\n'
        "    {{\n"
        '      "label": "<snake_case physical object/surface>",\n'
        '      "synonyms": ["term1","term2","term3"],\n'
        '      "reason": "≤120 chars, consistent with the issue list only",\n'
        '      "roi_hint": "<top_left|top_center|top_right|mid_left|center|mid_right|bottom_left|bottom_center|bottom_right|unknown>",\n'
        '      "priority": "<high|medium|low>"\n'
        "    }}\n"
        "  ]\n"
        "}}\n\n"
        "RULES\n"
        "- No duplicate targets; merge similar ones.\n"
        "- If uncertain, omit the target instead of guessing.\n"
        "- Respond with JSON ONLY (no markdown or commentary)."
    )
    return tmpl.format(
        scene=scene,
        is_staged=str(is_staged).lower(),
        policy_block=policy_block,
        max_targets=max_targets,
        allow_free=allow_free,
        issues_text=issues_text,
    )


def run_target_planning(
        image_bytes: bytes,
        scene: str,
        is_staged: bool,
        issues: List[NLIssue],
        max_targets: int,
        allow_free: int,
        model_name: str,
        lm_studio_url: str = LM_STUDIO_URL,
        http_client=requests,
) -> tuple[List[Dict], str, VLMDebugOutput]:
    """
    Run VLM Pass 4: Issue-driven target planning.

    Returns:
        Tuple of (targets: List[Dict], reasoning: str, debug_output: VLMDebugOutput)
    """
    t0 = time.time()
    logger.debug(f"  Pass 4: Target planning for scene='{scene}'...")

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    policy_block = get_scene_policy(scene)
    prompt = build_planner_prompt(scene, is_staged, policy_block, max_targets, allow_free, issues)

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]

    messages = [
        {"role": "system",
         "content": "You are an expert at analyzing real-estate listing photos. Respond with valid JSON only."},
        {"role": "user", "content": content},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False,
    }

    debug_output = VLMDebugOutput(
        pass_name="pass4_target_planning",
        prompt_summary=f"Issue-driven target planning for scene={scene}, {len(issues)} issues",
        raw_response="",
    )

    try:
        resp = http_client.post(
            f"{lm_studio_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            debug_output.error = f"API error: HTTP {resp.status_code}"
            debug_output.duration_seconds = time.time() - t0
            return [], "API error", debug_output

        data = resp.json()
        raw_response = data["choices"][0]["message"]["content"]

        debug_output.raw_response = raw_response
        debug_output.duration_seconds = time.time() - t0

        parsed = _extract_json(raw_response) or {}
        debug_output.parsed_result = parsed

        reasoning = str(parsed.get("reasoning", ""))[:200]
        raw_targets = parsed.get("targets", []) or []
        targets = _filter_banned_surface_targets(raw_targets)[:max_targets]

        return targets, reasoning, debug_output

    except Exception as e:
        logger.error(f"  Pass 4 (target planning) failed: {e}")
        debug_output.error = str(e)
        debug_output.duration_seconds = time.time() - t0
        return [], "Error", debug_output


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def keywords_for_scene(scene: str,
                       is_staged: bool = True,
                       include_conditions: bool = False,
                       max_keywords: int = 12) -> List[str]:
    """
    Get detection keywords for a scene type based on staging status.

    Args:
        scene: Scene type (e.g., 'living_room', 'kitchen')
        is_staged: Whether the photo shows staged/furnished space
        include_conditions: Add defect/condition keywords
        max_keywords: Maximum number of keywords to return

    Returns:
        List of cleaned, deduplicated keywords
    """
    # Get scene keywords
    scene_data = SCENE_KEYWORDS.get(scene, SCENE_KEYWORDS["unknown"])

    # Start with core keywords (always included)
    base_keywords = list(scene_data.get("core", []))

    # Add staging keywords if photo is staged
    if is_staged and "staging" in scene_data:
        base_keywords.extend(scene_data["staging"])

    # Add condition keywords if requested
    if include_conditions:
        base_keywords.extend(CONDITION_KEYWORDS)

    # Clean and limit
    cleaned = _clean_keywords(base_keywords)
    return cleaned[:max_keywords]


def format_for_grounding_dino(keywords: List[str]) -> str:
    """
    Format keywords as GroundingDINO text prompt.

    GroundingDINO expects lowercase, period-separated terms.
    Example: "sofa . coffee table . fireplace"
    """
    if not keywords:
        return ""

    # Clean and deduplicate
    cleaned = _clean_keywords(keywords)

    # Join with " . " separator
    return " . ".join(cleaned)


def run_groundingdino_for_issues(
        image_path: Path,
        defect_analysis: DefectAnalysis,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
) -> List[VisualAnchor]:
    """
    Run GroundingDINO on catalog issues marked 'present' to produce visual anchors.
    """
    try:
        import torch
        from groundingdino.util.inference import load_model, load_image, predict
    except ImportError as ie:
        logger.warning(f"GroundingDINO not available: {ie}")
        return []

    # Find issues marked present
    present_ids = [
        iid for iid, flag in defect_analysis.catalog_flags.items()
        if flag.present == "yes"
    ]
    if not present_ids:
        logger.debug("No issues marked present; skipping GroundingDINO anchors.")
        return []

    # Load model once
    if not DEFAULT_DINO_CONFIG.exists() or not DEFAULT_DINO_WEIGHTS.exists():
        logger.warning("GroundingDINO config or weights not found; skipping anchors.")
        return []

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(DEFAULT_DINO_CONFIG), str(DEFAULT_DINO_WEIGHTS), device=dev)
    image_tensor, _ = load_image(str(image_path))

    from PIL import Image
    with Image.open(image_path) as pil_img:
        width, height = pil_img.size

    anchors: List[VisualAnchor] = []

    for issue_id in present_ids:
        labels = ANCHOR_MAP.get(issue_id, [])
        if not labels:
            continue
        prompt = " . ".join(labels)

        try:
            boxes, logits, phrases = predict(
                model,
                image_tensor,
                prompt,
                box_threshold,
                text_threshold,
                device=dev,
            )
        except Exception as exc:
            logger.error(f"GroundingDINO prediction failed for {issue_id}: {exc}")
            continue

        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = [float(v) for v in box.tolist()]
            x0 = (cx - bw / 2.0) * width
            y0 = (cy - bh / 2.0) * height
            x1 = (cx + bw / 2.0) * width
            y1 = (cy + bh / 2.0) * height
            anchors.append(
                VisualAnchor(
                    issue_id=issue_id,
                    support_object_label=phrase.strip() or labels[0],
                    bbox=[x0, y0, x1, y1],
                    confidence=float(logit),
                )
            )

    return anchors


# ═══════════════════════════════════════════════════════════════════════════════
# Scene Classifier Class
# ═══════════════════════════════════════════════════════════════════════════════

class SceneClassifier:
    def __init__(self, lm_studio_url: str = LM_STUDIO_URL,
                 model_name: str = DEFAULT_MODEL,
                 include_conditions: bool = False,
                 max_keywords: int = 12,
                 max_targets: int = 8,
                 allow_free_discoveries: int = 1,
                 scene_policy_version: str = "v1",
                 debug: bool = False,
                 issue_catalog_path: str | Path = DEFAULT_ISSUE_CATALOG,
                 with_dino_anchors: bool = False):
        self.lm_studio_url = lm_studio_url.rstrip('/')
        self.model_name = model_name
        self.include_conditions = include_conditions
        self.max_keywords = max_keywords
        self.max_targets = max_targets
        self.allow_free_discoveries = allow_free_discoveries
        self.scene_policy_version = scene_policy_version
        self.debug = debug
        self.with_dino_anchors = with_dino_anchors
        self.issue_catalog = load_issue_catalog(Path(issue_catalog_path))

        logger.info(f"SceneClassifier initialized:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  LM Studio URL: {self.lm_studio_url}")
        logger.info(f"  Max targets: {self.max_targets}")
        logger.info(f"  Issue catalog: {len(get_catalog_ids(self.issue_catalog))} issues loaded")

    def _normalize_scene(self, scene: str) -> str:
        """Normalize scene name to canonical form."""
        scene = scene.lower().strip()
        scene = scene.replace('-', '_').replace(' ', '_')

        # Check if it's already valid
        if scene in VALID_SCENES:
            return scene

        # Try to find the closest match
        for valid in VALID_SCENES:
            if valid in scene or scene in valid:
                return valid

        # Default to unknown if no match
        return "unknown"

    def _create_fallback_targets(self, scene: str) -> List[Dict]:
        """Create fallback targets from scene core keywords."""
        scene_data = SCENE_KEYWORDS.get(scene, SCENE_KEYWORDS["unknown"])
        core_keywords = scene_data.get("core", [])[:4]  # Take first 4 core items

        targets = []
        for kw in core_keywords:
            targets.append({
                "label": kw.replace(" ", "_").lower(),
                "synonyms": [kw],
                "reason": "Scene core item",
                "roi_hint": "unknown",
                "priority": "medium"
            })
        return targets

    def classify_image(self, image_path: Path) -> SceneClassification:
        """Classify a single image and plan detection targets."""
        t0 = time.time()
        img_name = image_path.name

        logger.info(f"")
        logger.info(f"[{img_name}] {'═' * 60}")
        logger.info(f"[{img_name}] Starting classification pipeline (5 VLM passes)")

        # Initialize debug outputs container
        vlm_outputs = VLMDebugOutputs()

        scene_summary = SceneSummary(scene="unknown", image_summary="", overall_impression="")
        catalog_default_flags = {
            iid: CatalogFlag(present="uncertain", evidence="")
            for iid in get_catalog_ids(self.issue_catalog)
        }
        defect_analysis = DefectAnalysis(
            issues_natural_language=[],
            catalog_flags=catalog_default_flags,
            freeform_notes="",
        )

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            file_size_kb = image_path.stat().st_size / 1024
            logger.info(f"[{img_name}] Loading image ({file_size_kb:.1f} KB)")
            image_bytes = image_path.read_bytes()

            # ═══════════════════════════════════════════════════════════════════
            # PASS 1: Scene Summary (VLM call 1/5)
            # ═══════════════════════════════════════════════════════════════════
            t_pass1 = time.time()
            logger.info(f"[{img_name}] PASS 1: Scene Summary (VLM call 1/5)")
            scene_summary, pass1_debug = run_scene_summary(image_bytes, self.model_name, self.lm_studio_url)
            vlm_outputs.pass1_scene_summary = pass1_debug
            pass1_duration = time.time() - t_pass1
            logger.info(f"[{img_name}] PASS 1 complete: scene='{scene_summary.scene}' ({pass1_duration:.2f}s)")
            if scene_summary.overall_impression:
                logger.debug(f"[{img_name}]   Impression: {scene_summary.overall_impression[:100]}...")

            # ═══════════════════════════════════════════════════════════════════
            # PASS 2a: Freeform Defect Notes (VLM call 2/5)
            # ═══════════════════════════════════════════════════════════════════
            t_pass2a = time.time()
            analysis_scene = self._normalize_scene(scene_summary.scene) if scene_summary.scene else "unknown"
            logger.info(f"[{img_name}] PASS 2a: Freeform Defect Notes (VLM call 2/5) [scene={analysis_scene}]")
            freeform_notes, pass2a_debug = run_freeform_defect_notes(
                image_bytes,
                analysis_scene,
                self.model_name,
                self.lm_studio_url,
            )
            vlm_outputs.pass2a_freeform_notes = pass2a_debug
            pass2a_duration = time.time() - t_pass2a
            notes_preview = freeform_notes[:100].replace('\n', ' ') if freeform_notes else "(empty)"
            logger.info(f"[{img_name}] PASS 2a complete: {len(freeform_notes)} chars ({pass2a_duration:.2f}s)")
            logger.debug(f"[{img_name}]   Notes preview: {notes_preview}...")

            # ═══════════════════════════════════════════════════════════════════
            # PASS 2b: Defect Structuring (VLM call 3/5) - TEXT ONLY
            # ═══════════════════════════════════════════════════════════════════
            t_pass2b = time.time()
            logger.info(f"[{img_name}] PASS 2b: Defect Structuring (VLM call 3/5) [text-only]")
            defect_analysis, pass2b_debug = run_defect_structuring(
                analysis_scene,
                freeform_notes,
                self.issue_catalog,
                self.model_name,
                self.lm_studio_url,
            )
            vlm_outputs.pass2b_defect_structuring = pass2b_debug
            pass2b_duration = time.time() - t_pass2b
            issues_present = sum(1 for f in defect_analysis.catalog_flags.values() if f.present == "yes")
            logger.info(
                f"[{img_name}] PASS 2b complete: {len(defect_analysis.issues_natural_language)} NL issues, {issues_present} catalog flags present ({pass2b_duration:.2f}s)")

            # ═══════════════════════════════════════════════════════════════════
            # PASS 3: Staging Detection (VLM call 4/5)
            # ═══════════════════════════════════════════════════════════════════
            t_pass3 = time.time()
            scene = analysis_scene
            logger.info(f"[{img_name}] PASS 3: Staging Detection (VLM call 4/5) [scene={scene}]")
            is_staged, reasoning, pass3_debug = run_staging_detection(
                image_bytes,
                scene,
                self.model_name,
                self.lm_studio_url,
            )
            vlm_outputs.pass3_staging_detection = pass3_debug
            pass3_duration = time.time() - t_pass3
            staging_str = "STAGED" if is_staged else "VACANT"
            logger.info(
                f"[{img_name}] PASS 3 complete: {staging_str}, reason='{reasoning[:50]}...' ({pass3_duration:.2f}s)")

            # ═══════════════════════════════════════════════════════════════════
            # PASS 4: Issue-driven Target Planning (VLM call 5/5)
            # ═══════════════════════════════════════════════════════════════════
            t_pass4 = time.time()
            logger.info(f"[{img_name}] PASS 4: Issue-driven Target Planning (VLM call 5/5)")
            logger.debug(
                f"[{img_name}]   Using {len(defect_analysis.issues_natural_language)} issues from Pass 2b to guide target selection")

            targets, planner_reasoning, pass4_debug = run_target_planning(
                image_bytes,
                scene,
                is_staged,
                defect_analysis.issues_natural_language,
                self.max_targets,
                self.allow_free_discoveries,
                self.model_name,
                self.lm_studio_url,
            )
            vlm_outputs.pass4_target_planning = pass4_debug

            # Use planner reasoning if available
            if planner_reasoning:
                reasoning = planner_reasoning

            pass4_duration = time.time() - t_pass4
            logger.info(f"[{img_name}] PASS 4 complete: {len(targets)} targets from planner ({pass4_duration:.2f}s)")

            # ═══════════════════════════════════════════════════════════════════
            # POST-PROCESSING: Build GDINO terms from targets
            # ═══════════════════════════════════════════════════════════════════
            logger.info(f"[{img_name}] Post-processing: building GDINO terms from planner targets")

            gdino_terms: List[str] = []
            if targets:
                all_terms = []
                for target in targets:
                    if not isinstance(target, dict):
                        continue
                    if "label" in target:
                        label_term = _sanitize_term(target["label"]).replace("_", " ")
                        if label_term:
                            all_terms.append(label_term)
                gdino_terms = dedup_preserving_order(all_terms)[:15]

            # Fallback if no targets
            if not gdino_terms:
                logger.warning(f"[{img_name}]   No GDINO terms from targets, creating fallback from scene keywords")
                targets = self._create_fallback_targets(scene)
                targets = _filter_banned_surface_targets(targets)
                all_terms = []
                for target in targets:
                    label_term = _sanitize_term(target["label"]).replace("_", " ")
                    if label_term:
                        all_terms.append(label_term)
                gdino_terms = dedup_preserving_order(all_terms)[:15]

            # Compute scene-based keywords
            keywords = keywords_for_scene(
                scene,
                is_staged=is_staged,
                include_conditions=self.include_conditions,
                max_keywords=self.max_keywords,
            )
            groundingdino_prompt = format_for_grounding_dino(gdino_terms)

            logger.info(f"[{img_name}] GDINO prompt: '{groundingdino_prompt[:80]}...' ({len(gdino_terms)} terms)")
            logger.debug(f"[{img_name}]   Scene keywords: {len(keywords)} terms")

            # ═══════════════════════════════════════════════════════════════════
            # OPTIONAL: Visual Anchors via GroundingDINO
            # ═══════════════════════════════════════════════════════════════════
            issue_visual_anchors: List[VisualAnchor] = []
            if self.with_dino_anchors:
                logger.info(f"[{img_name}] Running GroundingDINO for issue visual anchors...")
                try:
                    t_anchors = time.time()
                    issue_visual_anchors = run_groundingdino_for_issues(image_path, defect_analysis)
                    logger.info(
                        f"[{img_name}]   Found {len(issue_visual_anchors)} visual anchors ({time.time() - t_anchors:.2f}s)")
                except Exception as exc:
                    logger.error(f"[{img_name}]   Failed to generate GroundingDINO anchors: {exc}")

            # ═══════════════════════════════════════════════════════════════════
            # FINALIZE: Build result
            # ═══════════════════════════════════════════════════════════════════
            final_scene = scene
            total_time = time.time() - t0

            result = SceneClassification(
                scene=final_scene,
                image_summary=scene_summary.image_summary,
                overall_impression=scene_summary.overall_impression,
                is_staged=is_staged,
                reasoning=reasoning,
                targets=targets,
                gdino_terms=gdino_terms,
                keywords=keywords,
                groundingdino_prompt=groundingdino_prompt,
                issues_natural_language=defect_analysis.issues_natural_language,
                catalog_flags=defect_analysis.catalog_flags,
                issue_visual_anchors=issue_visual_anchors,
                freeform_defect_notes=freeform_notes,
                processing_time=total_time,
                prompt_version=PROMPT_VERSION,
                scene_policy_version=self.scene_policy_version,
                # Legacy fields for backward compatibility
                raw_response=pass3_debug.raw_response if self.debug else None,
                planner_raw_response=pass4_debug.raw_response if self.debug else None,
                # New: all VLM outputs
                vlm_outputs=vlm_outputs,
            )

            # Final summary log
            logger.info(f"[{img_name}] {'─' * 60}")
            logger.info(f"[{img_name}] CLASSIFICATION COMPLETE")
            logger.info(f"[{img_name}]   Scene: {final_scene} ({staging_str})")
            logger.info(f"[{img_name}]   Targets: {len(targets)}")
            logger.info(f"[{img_name}]   NL Issues: {len(defect_analysis.issues_natural_language)}")
            logger.info(f"[{img_name}]   Catalog flags present: {issues_present}")
            logger.info(f"[{img_name}]   Total time: {total_time:.2f}s")
            logger.info(f"[{img_name}]     Pass 1 (scene summary):      {pass1_duration:.2f}s")
            logger.info(f"[{img_name}]     Pass 2a (freeform notes):    {pass2a_duration:.2f}s")
            logger.info(f"[{img_name}]     Pass 2b (defect structuring): {pass2b_duration:.2f}s")
            logger.info(f"[{img_name}]     Pass 3 (staging detection):  {pass3_duration:.2f}s")
            logger.info(f"[{img_name}]     Pass 4 (target planning):    {pass4_duration:.2f}s")
            logger.info(f"[{img_name}] {'═' * 60}")

            return result

        except Exception as e:
            logger.error(f"[{img_name}] ERROR: Classification failed: {e}")
            import traceback
            logger.debug(f"[{img_name}] Traceback: {traceback.format_exc()}")

            # Create fallback response
            fallback_scene = self._normalize_scene(scene_summary.scene) if scene_summary.scene else "unknown"
            fallback_targets = self._create_fallback_targets(fallback_scene)
            fallback_targets = _filter_banned_surface_targets(fallback_targets)

            all_terms = []
            for target in fallback_targets:
                label_term = _sanitize_term(target["label"]).replace("_", " ")
                if label_term:
                    all_terms.append(label_term)
            gdino_terms = dedup_preserving_order(all_terms)[:15]

            fallback_keywords = keywords_for_scene(
                fallback_scene,
                is_staged=False,
                include_conditions=self.include_conditions,
                max_keywords=self.max_keywords,
            )

            total_time = time.time() - t0
            logger.warning(
                f"[{img_name}] Returning fallback result (scene='{fallback_scene}', {len(fallback_targets)} targets)")

            return SceneClassification(
                scene=fallback_scene,
                is_staged=False,
                reasoning="Classification failed",
                targets=fallback_targets,
                gdino_terms=gdino_terms,
                keywords=fallback_keywords,
                groundingdino_prompt=format_for_grounding_dino(gdino_terms),
                image_summary=scene_summary.image_summary,
                overall_impression=scene_summary.overall_impression,
                issues_natural_language=defect_analysis.issues_natural_language,
                catalog_flags=defect_analysis.catalog_flags,
                issue_visual_anchors=[],
                freeform_defect_notes=defect_analysis.freeform_notes,
                error=str(e),
                processing_time=total_time,
                prompt_version=PROMPT_VERSION,
                scene_policy_version=self.scene_policy_version,
                raw_response=None,
                planner_raw_response=None,
                vlm_outputs=vlm_outputs,
            )

    def classify_batch(self, image_paths: List[Path],
                       save_results: bool = True) -> Dict[str, SceneClassification]:
        """Classify multiple images."""
        results = {}
        total_images = len(image_paths)

        logger.info(f"")
        logger.info(f"{'█' * 70}")
        logger.info(f"BATCH CLASSIFICATION: {total_images} images")
        logger.info(f"{'█' * 70}")

        for idx, img_path in enumerate(image_paths, 1):
            logger.info(f"")
            logger.info(f"[BATCH {idx}/{total_images}] Processing: {img_path.name}")
            result = self.classify_image(img_path)
            results[str(img_path)] = result

            # Rate limiting
            time.sleep(0.5)

        if save_results:
            # Save aggregated results
            output = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "model": self.model_name,
                "prompt_version": PROMPT_VERSION,
                "scene_policy_version": self.scene_policy_version,
                "classifications": {}
            }

            for path, classification in results.items():
                result_data = _serialize_classification(classification)
                output["classifications"][path] = result_data

            results_file = Path("scene_classifications.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to: {results_file}")

        # Batch summary
        logger.info(f"")
        logger.info(f"{'█' * 70}")
        logger.info(f"BATCH COMPLETE: {total_images} images processed")

        scene_counts = {}
        total_time = 0
        for result in results.values():
            scene_counts[result.scene] = scene_counts.get(result.scene, 0) + 1
            total_time += result.processing_time

        logger.info(f"Scene distribution:")
        for scene, count in sorted(scene_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {scene}: {count}")
        logger.info(f"Total processing time: {total_time:.1f}s (avg: {total_time / total_images:.1f}s/image)")
        logger.info(f"{'█' * 70}")

        return results


def _serialize_vlm_debug_output(output: Optional[VLMDebugOutput]) -> Optional[Dict]:
    """Serialize a VLMDebugOutput to a dict."""
    if output is None:
        return None
    return {
        "pass_name": output.pass_name,
        "prompt_summary": output.prompt_summary,
        "raw_response": output.raw_response,
        "parsed_result": output.parsed_result,
        "duration_seconds": output.duration_seconds,
        "error": output.error,
    }


def _serialize_vlm_debug_outputs(outputs: Optional[VLMDebugOutputs]) -> Optional[Dict]:
    """Serialize all VLM debug outputs to a dict."""
    if outputs is None:
        return None
    return {
        "pass1_scene_summary": _serialize_vlm_debug_output(outputs.pass1_scene_summary),
        "pass2a_freeform_notes": _serialize_vlm_debug_output(outputs.pass2a_freeform_notes),
        "pass2b_defect_structuring": _serialize_vlm_debug_output(outputs.pass2b_defect_structuring),
        "pass3_staging_detection": _serialize_vlm_debug_output(outputs.pass3_staging_detection),
        "pass4_target_planning": _serialize_vlm_debug_output(outputs.pass4_target_planning),
    }


def _serialize_classification(classification: SceneClassification) -> Dict:
    """Serialize a SceneClassification to a dict for JSON output."""
    result = {
        "scene": classification.scene,
        "image_summary": classification.image_summary,
        "overall_impression": classification.overall_impression,
        "is_staged": classification.is_staged,
        "reasoning": classification.reasoning,
        "targets": classification.targets,
        "gdino_terms": classification.gdino_terms,
        "keywords": classification.keywords,
        "groundingdino_prompt": classification.groundingdino_prompt,
        "issues_natural_language": [asdict(issue) for issue in classification.issues_natural_language],
        "catalog_flags": _serialize_catalog_flags(classification.catalog_flags),
        "issue_visual_anchors": [asdict(anchor) for anchor in classification.issue_visual_anchors],
        "freeform_defect_notes": classification.freeform_defect_notes,
        "processing_time": classification.processing_time,
        "prompt_version": classification.prompt_version,
        "scene_policy_version": classification.scene_policy_version,
        "error": classification.error,
    }

    # Always include VLM outputs for debugging
    result["vlm_outputs"] = _serialize_vlm_debug_outputs(classification.vlm_outputs)

    # Legacy fields (for backward compatibility)
    if classification.raw_response:
        result["raw_response"] = classification.raw_response
    if classification.planner_raw_response:
        result["planner_raw_response"] = classification.planner_raw_response

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Classify property photos and generate detection keywords"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to image file(s)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LM Studio model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--lm-studio-url",
        default=LM_STUDIO_URL,
        help=f"LM Studio API URL (default: {LM_STUDIO_URL})"
    )
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=12,
        help="Maximum keywords per scene (default: 12)"
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=8,
        help="Maximum targets to plan per image (default: 8)"
    )
    parser.add_argument(
        "--allow-free-discoveries",
        type=int,
        default=1,
        help="Allow VLM to add off-policy high-priority targets (default: 1)"
    )
    parser.add_argument(
        "--scene-policy-version",
        default="v1",
        help="Scene policy version to use (default: v1)"
    )
    parser.add_argument(
        "--include-conditions",
        action="store_true",
        help="Include defect/condition keywords (crack, stain, damage, etc.)"
    )
    parser.add_argument(
        "--issue-catalog", "--catalog",
        default=DEFAULT_ISSUE_CATALOG,
        help=f"Path to issue catalog JSON (default: {DEFAULT_ISSUE_CATALOG})"
    )
    parser.add_argument(
        "--with-dino-anchors",
        action="store_true",
        help="Run GroundingDINO on catalog issues marked present to produce visual anchors"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file (default: stdout for single image, file for batch)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List all valid scene types and exit"
    )

    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Handle --list-scenes
    if args.list_scenes:
        print("Valid scene types:")
        for scene in VALID_SCENES:
            scene_data = SCENE_KEYWORDS.get(scene, {})
            core_count = len(scene_data.get("core", []))
            staging_count = len(scene_data.get("staging", []))
            print(f"  {scene:20} - {core_count:2} core, {staging_count:2} staging keywords")
        sys.exit(0)

    # Setup classifier
    classifier = SceneClassifier(
        lm_studio_url=args.lm_studio_url,
        model_name=args.model,
        include_conditions=args.include_conditions,
        max_keywords=args.max_keywords,
        max_targets=args.max_targets,
        allow_free_discoveries=args.allow_free_discoveries,
        scene_policy_version=args.scene_policy_version,
        debug=args.debug,
        issue_catalog_path=args.issue_catalog,
        with_dino_anchors=args.with_dino_anchors,
    )

    # Convert paths
    image_paths = [Path(p) for p in args.images]

    # Validate paths
    for path in image_paths:
        if not path.exists():
            logger.error(f"Image not found: {path}")
            sys.exit(1)

    # Single image vs batch
    if len(image_paths) == 1:
        # Single image - output to stdout by default
        result = classifier.classify_image(image_paths[0])

        output = {
            "image": str(image_paths[0]),
            **_serialize_classification(result)
        }

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Result saved to: {args.output}")
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))

    else:
        # Batch processing
        save_to_file = args.output is not None
        results = classifier.classify_batch(image_paths, save_results=not save_to_file)

        if args.output:
            # Custom output file
            output = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "model": classifier.model_name,
                "prompt_version": PROMPT_VERSION,
                "scene_policy_version": classifier.scene_policy_version,
                "classifications": {}
            }

            for path, classification in results.items():
                output["classifications"][path] = _serialize_classification(classification)

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {args.output}")

        # Print summary
        print("\nClassification Summary:")
        print("-" * 70)
        for path, result in results.items():
            targets_count = len(result.targets)
            staging = "STAGED" if result.is_staged else "VACANT"
            print(f"{Path(path).name:25} → {result.scene:15} {staging:7} [{targets_count} targets]")


if __name__ == "__main__":
    main()