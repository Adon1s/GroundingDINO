#!/usr/bin/env python3
"""
Scene Classifier with Keyword Selection for Property Photos
-----------------------------------------------------------
Purpose: Classify property photos into scene types and generate appropriate
object detection keywords for GroundingDINO.

Features:
- Detects scene type (room/area)
- Determines if photo is staged or vacant
- Returns appropriate keywords (core only vs core + staging)

Input: Path to a property photo
Output: JSON with scene classification, staging status, and GroundingDINO prompt

Usage:
  python scene_classifier.py path/to/image.jpg
  python scene_classifier.py path/to/image.jpg --model qwen3-vl-30b --debug
  python scene_classifier.py path/to/image.jpg --max-keywords 15 --include-conditions
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
from typing import Dict, Optional, List
from dataclasses import dataclass
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
                 "outlet", "media niche", "sliding door", "radiator"],
        "staging": ["sofa", "sectional", "armchair", "coffee table", "end table",
                    "television", "tv stand", "media console", "floor lamp", "table lamp",
                    "rug", "bookshelf", "ottoman"]
    },

    "kitchen": {
        "core": ["range", "cooktop", "oven", "refrigerator", "microwave", "dishwasher",
                 "sink", "faucet", "countertop", "backsplash", "cabinet", "island",
                 "vent hood", "pot filler", "pantry door", "pendant light"],
        "staging": ["bar stool", "counter stool", "chair", "table", "dish rack",
                    "fruit bowl", "trash can"]
    },

    "bedroom": {
        "core": ["closet door", "ceiling fan", "ceiling light", "baseboard heater",
                 "radiator", "smoke detector", "bay window", "sliding door",
                 "floor register", "outlet"],
        "staging": ["bed", "headboard", "nightstand", "dresser", "wardrobe", "desk",
                    "chair", "table lamp", "floor lamp", "mirror", "television", "bench"]
    },

    "bathroom": {
        "core": ["toilet", "vanity", "sink", "faucet", "mirror", "shower", "bathtub",
                 "shower head", "glass door", "shower niche", "towel bar",
                 "toilet paper holder", "exhaust fan", "grab bar", "floor drain"],
        "staging": ["bath mat", "shower curtain", "towel", "hamper", "storage cart"]
    },

    "dining_room": {
        "core": ["chandelier", "ceiling fan", "built-in cabinet", "built-in shelf",
                 "bay window", "pocket door", "wall sconce", "sliding door",
                 "floor vent", "outlet"],
        "staging": ["dining table", "dining chair", "sideboard", "buffet", "hutch",
                    "bench", "rug", "bar cart"]
    },

    "home_office": {
        "core": ["built-in shelf", "built-in cabinet", "closet door", "ceiling fan",
                 "ceiling light", "outlet", "ethernet jack", "return vent",
                 "bay window", "smoke detector"],
        "staging": ["desk", "office chair", "monitor", "computer", "printer",
                    "filing cabinet", "bookshelf", "task lamp"]
    },

    "laundry_room": {
        "core": ["washer", "dryer", "laundry sink", "faucet", "countertop", "cabinet",
                 "shelf", "hanging rod", "vent duct", "drain pan", "floor drain",
                 "laundry hookups", "water heater", "electrical panel"],
        "staging": ["laundry basket", "hamper", "detergent bottle", "drying rack",
                    "ironing board"]
    },

    "hallway": {
        "core": ["ceiling light", "sconce", "smoke detector", "return vent",
                 "thermostat", "linen closet", "attic hatch", "baseboard heater",
                 "floor register", "handrail"],
        "staging": ["console table", "bench", "coat rack", "mirror", "runner", "wall art"]
    },

    "stairway": {
        "core": ["stairs", "landing", "handrail", "railing", "baluster", "newel post",
                 "wall sconce", "skylight", "stair gate"],
        "staging": ["runner", "wall art", "mirror", "bench"]
    },

    "basement": {
        "core": ["water heater", "furnace", "boiler", "electrical panel", "sump pump",
                 "support column", "support beam", "duct", "pipe", "dehumidifier",
                 "laundry hookups", "egress window", "floor drain", "radon fan"],
        "staging": ["shelving unit", "workbench", "storage rack", "tool chest",
                    "storage bin", "folding table"]
    },

    "attic": {
        "core": ["rafter", "beam", "truss", "vent", "attic fan", "duct", "chimney",
                 "hatch", "pull-down ladder", "skylight", "air handler"],
        "staging": ["storage bin", "box", "shelving unit"]
    },

    "garage": {
        "core": ["garage door", "door opener", "track", "workbench", "cabinet", "shelf",
                 "electrical panel", "water heater", "furnace", "laundry hookups",
                 "attic ladder", "hose bib", "ev charger", "floor drain"],
        "staging": ["tool chest", "storage bin", "ladder", "bicycle", "lawn mower",
                    "storage rack", "freezer", "trash can"]
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
                 "fence", "gate", "gutter", "downspout", "shed", "stair",
                 "exterior outlet", "hose bib"],
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
    keywords: List[str]
    groundingdino_prompt: str
    processing_time: float = 0.0
    raw_response: Optional[str] = None
    error: Optional[str] = None


# ── Keyword Helper Functions ─────────────────────────────────────────────────
def _clean_keywords(keywords: List[str]) -> List[str]:
    """Clean and deduplicate keywords."""
    seen = set()
    cleaned = []
    for kw in keywords:
        # Normalize: lowercase, strip, collapse whitespace
        kw_clean = " ".join(str(kw).lower().strip().split())
        if kw_clean and kw_clean not in seen:
            seen.add(kw_clean)
            cleaned.append(kw_clean)
    return cleaned


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


# ── Scene Classifier Class ───────────────────────────────────────────────────
class SceneClassifier:
    def __init__(self, lm_studio_url: str = LM_STUDIO_URL,
                 model_name: str = DEFAULT_MODEL,
                 include_conditions: bool = False,
                 max_keywords: int = 12,
                 debug: bool = False):
        self.lm_studio_url = lm_studio_url.rstrip('/')
        self.model_name = model_name
        self.include_conditions = include_conditions
        self.max_keywords = max_keywords
        self.debug = debug

    @staticmethod
    def encode_image_to_b64(image_path: Path) -> str:
        """Encode image file to base64 string."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    @staticmethod
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
            "- Use 'unknown' only if you cannot determine the scene or you believe that the scene depicted is not one of the listed scenes\n\n"
            "RESPOND ONLY WITH JSON (no extra text):\n"
            "{\n"
            '  "scene": "<one item from VALID_SCENES>",\n'
            '  "is_staged": true/false,\n'
            '  "reasoning": "brief explanation including staging status"\n'
            "}"
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract JSON object from text response."""
        # Strip code fences if present
        if "```" in text:
            m = re.search(r"```json\s*(\{[\s\S]*?})\s*```", text, re.IGNORECASE)
            if m:
                text = m.group(1)
            else:
                text = re.sub(r"^```[\w-]*|```$", "", text.strip())

        # Find first balanced { ... }
        brace_stack = []
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if not brace_stack:
                    start = i
                brace_stack.append('{')
            elif ch == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start is not None:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            pass
        return None

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

    def classify_image(self, image_path: Path) -> SceneClassification:
        """Classify a single image and generate detection keywords."""
        t0 = time.time()

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Build request
            prompt = self.build_classification_prompt()
            image_b64 = self.encode_image_to_b64(image_path)

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
                "max_tokens": 400,
                "stream": False
            }

            if self.debug:
                logger.info(f"Sending classification request for: {image_path.name}")

            # Make API request
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
                logger.info(f"Raw response: {raw_response[:300]}")

            parsed = self._extract_json(raw_response) or {}

            # Extract and normalize scene
            scene_raw = str(parsed.get("scene", "unknown"))
            scene = self._normalize_scene(scene_raw)

            is_staged = bool(parsed.get("is_staged", True))  # Default to staged if not specified
            reasoning = str(parsed.get("reasoning", ""))[:500]

            # Generate keywords based on scene AND staging status
            keywords = keywords_for_scene(
                scene,
                is_staged=is_staged,
                include_conditions=self.include_conditions,
                max_keywords=self.max_keywords
            )

            # Format as GroundingDINO prompt
            gdino_prompt = format_for_grounding_dino(keywords)

            result = SceneClassification(
                scene=scene,
                is_staged=is_staged,
                reasoning=reasoning,
                keywords=keywords,
                groundingdino_prompt=gdino_prompt,
                processing_time=time.time() - t0,
                raw_response=raw_response if self.debug else None
            )

            if self.debug:
                staging_str = "STAGED" if is_staged else "VACANT"
                logger.info(f"Classification: {scene} ({staging_str})")
                logger.info(f"Keywords: {len(keywords)} objects")

            return result

        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {e}")
            return SceneClassification(
                scene="unknown",
                is_staged=False,
                reasoning="Classification failed",
                keywords=[],
                groundingdino_prompt="",
                error=str(e),
                processing_time=time.time() - t0
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
                "classifications": {}
            }

            for path, classification in results.items():
                output["classifications"][path] = {
                    "scene": classification.scene,
                    "is_staged": classification.is_staged,
                    "reasoning": classification.reasoning,
                    "keywords": classification.keywords,
                    "groundingdino_prompt": classification.groundingdino_prompt,
                    "processing_time": classification.processing_time,
                    "error": classification.error
                }

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
        "--include-conditions",
        action="store_true",
        help="Include defect/condition keywords (crack, stain, damage, etc.)"
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
        debug=args.debug
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
            "is_staged": result.is_staged,
            "reasoning": result.reasoning,
            "keywords": result.keywords,
            "groundingdino_prompt": result.groundingdino_prompt,
            "processing_time": result.processing_time
        }

        if result.error:
            output["error"] = result.error

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
                "classifications": {}
            }

            for path, classification in results.items():
                output["classifications"][path] = {
                    "scene": classification.scene,
                    "is_staged": classification.is_staged,
                    "reasoning": classification.reasoning,
                    "keywords": classification.keywords,
                    "groundingdino_prompt": classification.groundingdino_prompt,
                    "processing_time": classification.processing_time
                }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {args.output}")

        # Print summary
        print("\nClassification Summary:")
        print("-" * 70)
        for path, result in results.items():
            kw_count = len(result.keywords)
            staging = "STAGED" if result.is_staged else "VACANT"
            print(f"{Path(path).name:25} → {result.scene:15} {staging:7} [{kw_count} kw]")


if __name__ == "__main__":
    main()