#!/usr/bin/env python3
from __future__ import annotations
"""
Scene Classifier with Keyword Selection for Property Photos
-----------------------------------------------------------
Purpose: Classify property photos into scene types and generate appropriate
object detection keywords for GroundingDINO.

Features:
- Detects scene type (room/area)
- Determines if photo is staged or vacant
- Returns appropriate keywords (core only vs core + staging)
- Plans structured targets for detection with priorities

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
    LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://100.102.92.1:1234")
    DEFAULT_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-vl-30b")

# ── Constants ────────────────────────────────────────────────────────────────
PROMPT_VERSION = "planner_v1"
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
BANNED_SURFACE_LABELS = {
    "ceiling",
    "ceilings",
    "wall",
    "walls",
    "floor",
    "floors",
    "flooring",
    "carpet",
    "hardwood_floor",
    "wood_floor",
    "laminate_floor",
    "basement_floor",
    "basement_walls",
    "tile_floor",
    "vinyl_floor",
}


def _is_banned_surface_label(label: str) -> bool:
    """Return True if label is a generic wall/ceiling/floor surface."""
    if not label:
        return False
    norm = _sanitize_term(label).replace(" ", "_")
    return norm in BANNED_SURFACE_LABELS


def _filter_banned_surface_targets(targets: List[Dict]) -> List[Dict]:
    """
    Drop any target whose label or synonyms are generic wall/ceiling/floor
    so they never become GroundingDINO terms.
    """
    filtered: List[Dict] = []
    for t in targets or []:
        if not isinstance(t, dict):
            continue

        label = t.get("label", "")
        synonyms = t.get("synonyms", [])
        if not isinstance(synonyms, list):
            synonyms = []

        # Check label
        if _is_banned_surface_label(label):
            continue

        # Check synonyms
        banned = False
        for s in synonyms:
            if _is_banned_surface_label(str(s)):
                banned = True
                break

        if banned:
            continue

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
    raw_response: Optional[str] = None  # classification raw
    planner_raw_response: Optional[str] = None  # planner raw
    error: Optional[str] = None
    image_summary: str = ""
    overall_impression: str = ""
    issues_natural_language: List[NLIssue] = field(default_factory=list)
    catalog_flags: Dict[str, CatalogFlag] = field(default_factory=dict)
    issue_visual_anchors: List[VisualAnchor] = field(default_factory=list)


@dataclass
class SceneSummary:
    scene: str
    image_summary: str
    overall_impression: str


Presence = Literal["yes", "no", "uncertain"]


@dataclass
class NLIssue:
    description: str
    rough_category: str
    location_hint: str = ""


@dataclass
class CatalogFlag:
    present: Presence
    evidence: str = ""


@dataclass
class DefectAnalysis:
    issues_natural_language: List[NLIssue]
    catalog_flags: Dict[str, CatalogFlag]
    targets: List[Dict]


@dataclass
class VisualAnchor:
    issue_id: str
    support_object_label: str
    bbox: List[float]
    confidence: float


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
        "You are an assistant that analyzes real-estate listing photos.\n"
        "You ONLY use what is clearly visible in the image. Do not guess about things you cannot see.\n\n"
        "Your job for this image is ONLY to:\n"
        "1) Identify the high-level scene type (e.g. 'kitchen', 'bathroom', 'living_room', 'bedroom', 'exterior_front', 'exterior_back', 'basement', 'hallway', 'dining_room', 'laundry_room', etc.).\n"
        "2) Provide a 2–3 sentence 'overall_impression' that briefly describes what the photo shows and gives a buyer or investor perspective on appeal and condition.\n\n"
        "Important:\n"
        "- Do NOT talk about pricing or dollar values.\n"
        "- Do NOT invent defects or problems; only comment on what is clearly visible.\n"
        "- Do NOT mention any issue catalog, targets, chips, or bounding boxes.\n\n"
        "Return a SINGLE JSON object with this exact structure:\n"
        "{\n"
        '  "scene": "<string scene label>",\n'
        '  "overall_impression": "<2-3 sentence description + high-level impression>"\n'
        "}\n"
        "Do NOT wrap the JSON in backticks or markdown; output raw JSON only.\n"
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


def build_defect_analysis_prompt(scene: str, issue_catalog: dict) -> str:
    """
    Build the prompt for the Defect Analysis pass.
    This prompt:
      - RECEIVES the scene label from run_scene_summary as context.
      - RECEIVES the issue catalog (ids, names, descriptions).
      - MUST NOT ask for scene or general summary again.
      - ASKS ONLY for:
          * issues_natural_language[]
          * catalog_flags{issue_id -> present/evidence}
          * targets[] with ROI hints (similar to existing planner behavior).
    """
    catalog_text = _format_catalog_for_prompt(issue_catalog)
    catalog_ids = get_catalog_ids(issue_catalog)
    catalog_list_str = ", ".join(catalog_ids)

    return (
        "You are an assistant that ONLY analyzes defects and renovation opportunities visible in a real-estate photo.\n"
        "Do NOT restate the scene or provide a general description.\n"
        f"scene: {scene}\n\n"
        "ISSUE CATALOG (fixed ids, include ALL of them in catalog_flags):\n"
        f"{catalog_text}\n\n"
        "What to return (strict JSON only, no markdown):\n"
        "{\n"
        '  "issues_natural_language": [\n'
        "    {\n"
        '      "description": "<string>",\n'
        '      "rough_category": "<cosmetic|moisture|structure|systems|exterior|opportunity>",\n'
        '      "location_hint": "<string>"\n'
        "    }\n"
        "  ],\n"
        '  "catalog_flags": {\n'
        '    "<issue_id>": {\n'
        '      "present": "yes|no|uncertain",\n'
        '      "evidence": "<string or empty>"\n'
        "    }\n"
        "  },\n"
        '  "targets": [\n'
        "    {\n"
        '      "label": "<string or issue_id>",\n'
        '      "description": "<string>",\n'
        '      "roi_hint": "<string>",\n'
        '      "priority": "<low|medium|high or numeric>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Focus ONLY on defects/condition issues/renovation opportunities visible in the photo.\n"
        "- Use rough_category from: cosmetic, moisture, structure, systems, exterior, opportunity.\n"
        "- catalog_flags MUST include every issue_id from the catalog:"
        f" {catalog_list_str}. If unsure, use 'uncertain'.\n"
        "- When present == 'yes', add brief evidence text.\n"
        "- targets should be actionable ROI hints (like planner targets) using concise labels.\n"
        "- Respond with raw JSON only. Do NOT wrap in backticks or add explanations."
    )


def run_scene_summary(image_bytes: bytes, model_name: str, lm_studio_url: str = LM_STUDIO_URL,
                      http_client=requests) -> SceneSummary:
    """
    Run VLM Pass 1 for a single image:
    - Calls the VLM with the Pass 1 prompt.
    - Parses the JSON response.
    - Returns a SceneSummary object.
    """
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

    try:
        resp = http_client.post(
            f"{lm_studio_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        if resp.status_code != 200:
            raise RuntimeError(f"API error: HTTP {resp.status_code} - {resp.text[:200]}")

        data = resp.json()
        raw_response = data["choices"][0]["message"]["content"]
        parsed = _extract_json(raw_response) or {}

        scene = str(parsed.get("scene", "unknown"))
        overall_impression = str(parsed.get("overall_impression", "")).strip()

        return SceneSummary(
            scene=scene,
            image_summary="",  # no longer used; kept for backward compatibility
            overall_impression=overall_impression,
        )

    except Exception as e:
        logger.error(f"VLM Pass 1 failed: {e}")
        return SceneSummary(scene="unknown", image_summary="", overall_impression="")


def _normalize_presence(value: str) -> Presence:
    v = str(value).strip().lower()
    if v in {"yes", "no", "uncertain"}:
        return v  # type: ignore[return-value]
    return "uncertain"


def run_defect_analysis(
    image_bytes: bytes,
    scene: str,
    issue_catalog: dict,
    model_name: str,
    lm_studio_url: str = LM_STUDIO_URL,
    http_client=requests,
) -> DefectAnalysis:
    """
    Run the Defect Analysis pass for a single image:
      - Use build_defect_analysis_prompt(scene, issue_catalog) as the user prompt.
      - Call the same VLM endpoint as used elsewhere.
      - Parse raw JSON into a DefectAnalysis object.
    """
    prompt = build_defect_analysis_prompt(scene, issue_catalog)
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    catalog_ids = get_catalog_ids(issue_catalog)

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
        "max_tokens": 2048,
        "stream": False,
    }

    default_flags = {iid: CatalogFlag(present="uncertain", evidence="") for iid in catalog_ids}

    try:
        resp = http_client.post(
            f"{lm_studio_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=40,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"API error: HTTP {resp.status_code} - {resp.text[:200]}")

        data = resp.json()
        raw_response = data["choices"][0]["message"]["content"]
        parsed = _extract_json(raw_response) or {}

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

        # Parse catalog_flags ensuring all ids exist
        parsed_flags = parsed.get("catalog_flags", {}) or {}
        catalog_flags: Dict[str, CatalogFlag] = {}
        for iid in catalog_ids:
            if isinstance(parsed_flags, dict) and iid in parsed_flags:
                entry = parsed_flags.get(iid, {}) or {}
                present_raw = entry.get("present", "uncertain") if isinstance(entry, dict) else "uncertain"
                evidence = ""
                if isinstance(entry, dict):
                    evidence = str(entry.get("evidence", "")).strip()
                catalog_flags[iid] = CatalogFlag(present=_normalize_presence(present_raw), evidence=evidence)
            else:
                catalog_flags[iid] = CatalogFlag(present="uncertain", evidence="")

        # Parse targets
        parsed_targets = parsed.get("targets", []) or []
        targets = parsed_targets if isinstance(parsed_targets, list) else []

        return DefectAnalysis(
            issues_natural_language=nl_issues,
            catalog_flags=catalog_flags,
            targets=targets,
        )

    except Exception as e:
        logger.error(f"Defect analysis failed: {e}")
        return DefectAnalysis(issues_natural_language=[], catalog_flags=default_flags, targets=[])


def build_classification_prompt() -> str:
    """Build the scene classification prompt with staging detection."""
    scenes_json = json.dumps(VALID_SCENES, indent=2)

    return (
        "You are an expert at analyzing real estate property photos.\n\n"
        "TASK: Identify the scene type AND determine if the photo is staged/furnished.\n\n"
        f"VALID_SCENES = {scenes_json}\n\n"
        "STAGING DETECTION:\n"
        "- STAGED: Photo shows furniture, decor, or movable items (sofa, bed, table, etc.)\n"
        "- VACANT: Photo shows only permanent fixtures (cabinets, appliances, built-ins)\n"
        "- Consider: Is there furniture? Are there decorative items? Is it furnished?\n\n"
        "RULES:\n"
        "- Choose exactly ONE scene from VALID_SCENES above\n"
        "- Copy the scene name verbatim from the list\n"
        "- Determine if photo is 'staged' (has furniture/decor) or 'vacant' (empty/unfurnished)\n"
        "- If uncertain, pick the most likely option\n"
        "- Use 'unknown' only if you cannot determine the scene or you believe that the scene depicted is not one of the listed scenes\n"
        "- Keep 'reasoning' VERY short (one brief phrase, max ~15 words).\n\n"
        "RESPOND ONLY WITH JSON (no extra text):\n"
        "{\n"
        '  "scene": "<one item from VALID_SCENES>",\n'
        '  "is_staged": true/false,\n'
        '  "reasoning": "very short phrase explaining why you chose this scene and staging"\n'
        "}"
    )


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
                         max_targets: int, allow_free: int) -> str:
    """Build the planner prompt requesting structured targets."""
    tmpl = (
        "You are an expert at analyzing real-estate listing photos.\n\n"
        "GOAL\n"
        "- Propose up to {max_targets} high-value TARGETS to draw bounding boxes around.\n"
        "- A TARGET is a physical surface or fixture that matters to buyers/agents "
        "for price, marketability, or inspection risk.\n"
        "- Prefer structural/condition-relevant items over decor and clutter.\n"
        "- You may add up to {allow_free} extra high-severity \"free discoveries\" beyond the scene policy if important.\n\n"

        "SCENE CONTEXT\n"
        "- scene: {scene}\n"
        "- is_staged: {is_staged}\n\n"

        "SCENE POLICY (preferred surfaces/fixtures for this scene):\n"
        "{policy_block}\n\n"

        "REALTOR PRIORITIES (examples)\n"
        "- kitchens: kitchen_cabinets, countertops, backsplash, appliances, sink, faucet, walls_paint, light_fixtures, island\n"
        "- bathrooms: bathroom_vanity, shower_tub, shower_tile, toilet, mirror, walls_paint, light_fixtures\n"
        "- interior: windows, interior_doors, stair_rail, fireplace, built_ins, closet_system\n"
        "- exterior: roof_surface, siding, exterior_paint, driveway, walkway, front_door, garage_door, deck, patio, fence, landscaping\n\n"

        "LABEL RULES\n"
        "- `label` MUST be the underlying physical object/surface, NOT the problem description.\n"
        "- Format: snake_case, lowercase.\n"
        "- Length: 1–2 tokens when split on underscores.\n"
        "- No repeated tokens (never 'cabinet_cabinet').\n"
        "- Do NOT include scene words like 'kitchen', 'bathroom', 'living_room'.\n"
        "- Do NOT include adjectives or condition words in `label` (old, new, cracked, stained, dirty, dated, mold, etc.).\n\n"
        "- IMPORTANT: Do NOT use generic surfaces like ceiling, ceilings, wall, walls, floor, flooring, carpet, hardwood floor, laminate floor, tile floor, basement floor/walls as TARGET labels. Those issues should be described in text only, not boxed.\n"


        "SYNONYMS & CONDITION\n"
        "- `synonyms`: 2–5 short nouns/noun-phrases (1–3 words) that a human might say for the same surface.\n"
        "- Condition words (crack, stain, damaged, dirty, outdated, mold, etc.) are allowed ONLY in `reason`, "
        "never in `label` or `synonyms`.\n\n"

        "EXAMPLES\n"
        "- Crack in driveway:\n"
        '  label: "driveway"\n'
        '  synonyms: ["driveway","concrete_drive","front_drive"]\n'
        '  reason: "Visible cracking in concrete; driveway condition affects curb appeal and inspection risk."\n'
        "- Water stains on ceiling:\n"
        '  label: "ceiling_area"\n'
        '  synonyms: ["ceiling_area","upper_drywall"]\n'
        '  reason: "Water staining suggests possible roof or plumbing leak, which worries buyers."\n'
        "- Dated kitchen cabinets:\n"
        '  label: "kitchen_cabinets"\n'
        '  synonyms: ["kitchen_cabinets","upper_cabinets","lower_cabinets"]\n'
        '  reason: "Cabinets look worn and dated; kitchen cabinetry is a major cosmetic cost."\n\n'

        "TARGET SELECTION RULES\n"
        "- Prefer a small set of high-impact targets over many minor ones.\n"
        "- If multiple issues exist on the same surface, use ONE target and describe the most important issue in `reason`.\n"
        "- Skip small decor, personal items, and generic clutter.\n\n"

        "SELF-CHECK BEFORE ANSWERING\n"
        "- For every target, verify:\n"
        "  * `label` is snake_case, 1–2 tokens, no duplicates.\n"
        "  * `label` does not contain scene or condition words.\n"
        "  * Condition words appear only in `reason`.\n\n"

        "OUTPUT FORMAT (STRICT JSON ONLY)\n"
        "{{\n"
        '  "scene": "<one from known scene list>",\n'
        '  "is_staged": true/false,\n'
        '  "reasoning": "≤200 chars why these targets matter overall",\n'
        '  "targets": [\n'
        "    {{\n"
        '      "label": "<snake_case physical object/surface>",\n'
        '      "synonyms": ["term1","term2","term3"],\n'
        '      "reason": "≤120 chars, including any condition or upgrade description",\n'
        '      "roi_hint": "<top_left|top_center|top_right|mid_left|center|mid_right|bottom_left|bottom_center|bottom_right|unknown>",\n'
        '      "priority": "<high|medium|low>"\n'
        "    }}\n"
        "  ]\n"
        "}}\n\n"
        "RULES\n"
        "- No duplicate targets; merge similar ones.\n"
        "- Use short, detector-friendly nouns/noun-phrases in `label` and `synonyms`.\n"
        "- If uncertain, omit the target rather than guessing.\n"
        "- Respond with JSON ONLY (no markdown or commentary)."
    )
    return tmpl.format(
        scene=scene,
        is_staged=str(is_staged).lower(),
        policy_block=policy_block,
        max_targets=max_targets,
        allow_free=allow_free,
    )




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

    GroundingDINO expects: "object1. object2. object3."
    """
    if not keywords:
        return ""
    return ". ".join(keywords) + "."


def run_groundingdino_for_issues(
    image_path: Path,
    defect_analysis: DefectAnalysis,
    config_path: Path = DEFAULT_DINO_CONFIG,
    weights_path: Path = DEFAULT_DINO_WEIGHTS,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: Optional[str] = None,
) -> List[VisualAnchor]:
    """
    For each issue_id with present == 'yes' in defect_analysis.catalog_flags:
      - Look up support-object labels from ANCHOR_MAP.
      - For each support-object label, call GroundingDINO to detect that object.
      - For each detection, create a VisualAnchor(issue_id, support_object_label, bbox, confidence).
    """
    anchors: List[VisualAnchor] = []
    try:
        from groundingdino.util.inference import load_model, load_image, predict
        import torch
    except Exception as exc:
        logger.error(f"GroundingDINO dependencies unavailable: {exc}")
        return anchors

    if not config_path.exists() or not weights_path.exists():
        logger.warning(
            f"GroundingDINO config/weights missing (config={config_path}, weights={weights_path}); skipping anchors."
        )
        return anchors

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = load_model(str(config_path), str(weights_path), device=dev)
        image_source, image_tensor = load_image(str(image_path))
        height, width = image_source.shape[:2]
    except Exception as exc:
        logger.error(f"Failed to initialize GroundingDINO: {exc}")
        return anchors

    for issue_id, flag in defect_analysis.catalog_flags.items():
        if flag.present != "yes":
            continue
        labels = ANCHOR_MAP.get(issue_id)
        if not labels:
            continue
        prompt = ". ".join(labels) + "."
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


# ── Scene Classifier Class ───────────────────────────────────────────────────
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
        scene_summary = SceneSummary(scene="unknown", image_summary="", overall_impression="")
        catalog_default_flags = {iid: CatalogFlag(present="uncertain", evidence="") for iid in get_catalog_ids(self.issue_catalog)}
        defect_analysis = DefectAnalysis(issues_natural_language=[], catalog_flags=catalog_default_flags, targets=[])

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image_bytes = image_path.read_bytes()
            scene_summary = run_scene_summary(image_bytes, self.model_name, self.lm_studio_url)
            analysis_scene = self._normalize_scene(scene_summary.scene) if scene_summary.scene else "unknown"
            defect_analysis = run_defect_analysis(
                image_bytes,
                analysis_scene,
                self.issue_catalog,
                self.model_name,
                self.lm_studio_url,
            )
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # First, get scene classification
            prompt = build_classification_prompt()

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
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2048,
                "stream": False
            }

            if self.debug:
                logger.info(f"Sending classification request for: {image_path.name}")

            # Make API request for scene classification
            resp = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )

            if resp.status_code != 200:
                raise RuntimeError(f"API error: HTTP {resp.status_code} - {resp.text[:200]}")

            # Parse response
            data = resp.json()
            raw_response = data["choices"][0]["message"]["content"]

            if self.debug:
                logger.info(f"Raw classification response: {raw_response[:300]}")

            parsed = _extract_json(raw_response) or {}

            # Extract and normalize scene
            scene_raw = str(parsed.get("scene", "unknown"))
            scene = self._normalize_scene(scene_raw)
            is_staged = _to_bool(parsed.get("is_staged", True))
            reasoning = str(parsed.get("reasoning", ""))[:200]

            # Now get the planner targets
            policy_block = get_scene_policy(scene)
            planner_prompt = build_planner_prompt(
                scene, is_staged, policy_block,
                self.max_targets, self.allow_free_discoveries
            )

            # Make second API call for planner
            planner_content = [
                {"type": "text", "text": planner_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]

            planner_messages = [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing real-estate listing photos. Respond with valid JSON only."
                },
                {"role": "user", "content": planner_content}
            ]

            planner_payload = {
                "model": self.model_name,
                "messages": planner_messages,
                "temperature": 0.1,
                "max_tokens": 2048,
                "stream": False
            }

            if self.debug:
                logger.info(f"Sending planner request for: {image_path.name}")

            # Make planner API request
            planner_resp = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json=planner_payload,
                timeout=30
            )

            targets = []
            planner_raw = None  # Initialize
            gdino_terms: List[str] = []

            if planner_resp.status_code == 200:
                planner_data = planner_resp.json()
                planner_raw = planner_data["choices"][0]["message"]["content"]

                if self.debug:
                    logger.info(f"Raw planner response: {planner_raw[:300]}")

                planner_parsed = _extract_json(planner_raw) or {}

                if self.debug and not planner_parsed:
                    logger.warning("Planner JSON parse failed; falling back to scene core targets.")

                # Update reasoning if planner provided one
                if "reasoning" in planner_parsed:
                    reasoning = str(planner_parsed["reasoning"])[:200]

                # Extract targets
                raw_targets = planner_parsed.get("targets", []) or []
                targets = _filter_banned_surface_targets(raw_targets)[: self.max_targets]

            defect_targets = defect_analysis.targets or []
            if defect_targets:
                merged_targets = list(targets) if targets else []
                existing_labels = {
                    t.get("label")
                    for t in merged_targets
                    if isinstance(t, dict) and t.get("label")
                }
                for tgt in defect_targets:
                    if not isinstance(tgt, dict):
                        continue
                    label = tgt.get("label")
                    if label and label in existing_labels:
                        continue
                    merged_targets.append(tgt)
                    if label:
                        existing_labels.add(label)
                targets = _filter_banned_surface_targets(merged_targets)[: self.max_targets] if merged_targets else targets

            # Build gdino_terms from merged targets
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
                targets = self._create_fallback_targets(scene)
                # Build gdino_terms from fallback
                all_terms = []
                for target in targets:
                    # Convert snake_case label to spaced phrase for detector
                    label_term = _sanitize_term(target["label"]).replace("_", " ")
                    if label_term:
                        all_terms.append(label_term)
                gdino_terms = dedup_preserving_order(all_terms)[:15]

            # For backward compatibility
            keywords = gdino_terms
            groundingdino_prompt = format_for_grounding_dino(gdino_terms)

            issue_visual_anchors: List[VisualAnchor] = []
            if self.with_dino_anchors:
                try:
                    issue_visual_anchors = run_groundingdino_for_issues(image_path, defect_analysis)
                except Exception as exc:
                    logger.error(f"Failed to generate GroundingDINO anchors: {exc}")

            pass1_scene = self._normalize_scene(scene_summary.scene)
            final_scene = pass1_scene or scene
            if final_scene == "unknown" and scene != "unknown":
                final_scene = scene

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
                processing_time=time.time() - t0,
                prompt_version=PROMPT_VERSION,
                scene_policy_version=self.scene_policy_version,
                raw_response=raw_response if self.debug else None,
                planner_raw_response=planner_raw if self.debug else None
            )

            if self.debug:
                staging_str = "STAGED" if is_staged else "VACANT"
                logger.info(f"Pass 1 scene: {final_scene}")
                logger.info(f"Classification: {scene} ({staging_str})")
                if scene_summary.image_summary:
                    logger.info(f"Image summary: {scene_summary.image_summary[:200]}")
                if scene_summary.overall_impression:
                    logger.info(f"Overall impression: {scene_summary.overall_impression[:200]}")
                logger.info(f"Targets: {len(targets)} targets")
                logger.info(f"GDINO terms: {len(gdino_terms)} terms")

            return result

        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {e}")

            # Create fallback response
            fallback_scene = self._normalize_scene(scene_summary.scene) if scene_summary.scene else "unknown"
            fallback_targets = self._create_fallback_targets(fallback_scene)

            # Build gdino_terms from fallback
            all_terms = []
            for target in fallback_targets:
                # Convert snake_case label to spaced phrase for detector
                label_term = _sanitize_term(target["label"]).replace("_", " ")
                if label_term:
                    all_terms.append(label_term)
            gdino_terms = dedup_preserving_order(all_terms)[:15]

            return SceneClassification(
                scene=fallback_scene,
                is_staged=False,
                reasoning="Classification failed",
                targets=fallback_targets,
                gdino_terms=gdino_terms,
                keywords=gdino_terms,
                groundingdino_prompt=format_for_grounding_dino(gdino_terms),
                image_summary=scene_summary.image_summary,
                overall_impression=scene_summary.overall_impression,
                issues_natural_language=defect_analysis.issues_natural_language,
                catalog_flags=defect_analysis.catalog_flags,
                issue_visual_anchors=[],
                error=str(e),
                processing_time=time.time() - t0,
                prompt_version=PROMPT_VERSION,
                scene_policy_version=self.scene_policy_version,
                raw_response=None,
                planner_raw_response=None
            )

    def classify_batch(self, image_paths: List[Path],
                       save_results: bool = True) -> Dict[str, SceneClassification]:
        """Classify multiple images."""
        results = {}

        for img_path in image_paths:
            logger.info(f"Classifying: {img_path.name}")
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
                result_data = {
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
                    "processing_time": classification.processing_time,
                    "prompt_version": classification.prompt_version,
                    "scene_policy_version": classification.scene_policy_version,
                    "error": classification.error
                }
                # Add debug fields if available
                if classification.raw_response:
                    result_data["raw_response"] = classification.raw_response
                if classification.planner_raw_response:
                    result_data["planner_raw_response"] = classification.planner_raw_response

                output["classifications"][path] = result_data

            results_file = Path("scene_classifications.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to: {results_file}")

        return results


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
            "scene": result.scene,
            "image_summary": result.image_summary,
            "overall_impression": result.overall_impression,
            "is_staged": result.is_staged,
            "reasoning": result.reasoning,
            "targets": result.targets,
            "gdino_terms": result.gdino_terms,
            "keywords": result.keywords,
            "groundingdino_prompt": result.groundingdino_prompt,
            "issues_natural_language": [asdict(issue) for issue in result.issues_natural_language],
            "catalog_flags": _serialize_catalog_flags(result.catalog_flags),
            "issue_visual_anchors": [asdict(anchor) for anchor in result.issue_visual_anchors],
            "processing_time": result.processing_time,
            "prompt_version": result.prompt_version,
            "scene_policy_version": result.scene_policy_version,
            "error": result.error
        }

        # Add debug fields if available
        if result.raw_response:
            output["raw_response"] = result.raw_response
        if result.planner_raw_response:
            output["planner_raw_response"] = result.planner_raw_response

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
                result_data = {
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
                    "processing_time": classification.processing_time,
                    "prompt_version": classification.prompt_version,
                    "scene_policy_version": classification.scene_policy_version,
                    "error": classification.error
                }
                # Add debug fields if available
                if classification.raw_response:
                    result_data["raw_response"] = classification.raw_response
                if classification.planner_raw_response:
                    result_data["planner_raw_response"] = classification.planner_raw_response

                output["classifications"][path] = result_data

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