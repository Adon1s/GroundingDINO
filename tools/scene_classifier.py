#!/usr/bin/env python3
"""
Scene Classifier with Keyword Selection for Property Photos
-----------------------------------------------------------
Purpose: Classify property photos into scene types and generate appropriate
object detection keywords for GroundingDINO.

Input: Path to a property photo
Output: JSON with scene classification, confidence, and GroundingDINO prompt

Usage:
  python scene_classifier.py path/to/image.jpg
  python scene_classifier.py path/to/image.jpg --model gemma-3-27b-it --debug
  python scene_classifier.py path/to/image.jpg --max-keywords 15 --include-conditions

Environment variables:
  LM_STUDIO_URL     - LM Studio API endpoint (default: http://localhost:1234)
  LM_STUDIO_MODEL   - Model to use (default: gemma-3-27b-it)
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
from typing import Dict, Optional, List, Tuple
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
    LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://192.168.86.143:1234")
    DEFAULT_MODEL = os.getenv("LM_STUDIO_MODEL", "gemma-3-27b-it")

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

# Scene aliases for better matching
SCENE_ALIASES = {
    "living": "living_room",
    "family_room": "living_room",
    "great_room": "living_room",
    "master_bedroom": "bedroom",
    "guest_bedroom": "bedroom",
    "kids_bedroom": "bedroom",
    "master_bathroom": "bathroom",
    "powder_room": "bathroom",
    "half_bath": "bathroom",
    "full_bath": "bathroom",
    "study": "home_office",
    "den": "home_office",
    "utility_room": "laundry_room",
    "mudroom": "laundry_room",
    "backyard": "yard",
    "front_yard": "yard",
    "side_yard": "yard",
    "exterior": "exterior_front",
    "outside": "exterior_front"
}

# ── Keyword Mappings ─────────────────────────────────────────────────────────
# Object keywords for each scene type (for GroundingDINO detection)
SCENE_KEYWORDS = {
    "living_room": [
        "sofa", "chair", "television", "coffee table", "lamp", "window",
        "fireplace", "rug", "painting", "plant", "bookshelf", "curtain",
        "ottoman", "side table", "vase", "cushion", "remote", "clock"
    ],

    "kitchen": [
        "refrigerator", "stove", "microwave", "sink", "cabinet", "counter",
        "dishwasher", "table", "chair", "window", "toaster", "coffee maker",
        "backsplash", "island", "bar stool", "cutting board", "pot", "pan"
    ],

    "bedroom": [
        "bed", "nightstand", "dresser", "lamp", "window", "closet",
        "mirror", "chair", "desk", "television", "rug", "painting",
        "curtain", "blanket", "pillow", "wardrobe", "clock", "plant"
    ],

    "bathroom": [
        "toilet", "sink", "bathtub", "shower", "mirror", "towel",
        "cabinet", "window", "tile", "faucet", "counter", "light fixture",
        "towel rack", "mat", "shower curtain", "soap dispenser", "toilet paper"
    ],

    "dining_room": [
        "table", "chair", "chandelier", "window", "cabinet", "painting",
        "rug", "mirror", "sideboard", "vase", "plant", "curtain",
        "centerpiece", "place setting", "candlestick", "buffet", "wine rack"
    ],

    "home_office": [
        "desk", "chair", "computer", "monitor", "keyboard", "bookshelf",
        "lamp", "window", "printer", "filing cabinet", "plant", "clock",
        "whiteboard", "phone", "pen holder", "notebook", "mouse"
    ],

    "garage": [
        "car", "door", "shelf", "tool", "bicycle", "workbench",
        "storage box", "ladder", "trash can", "light", "cabinet", "hose",
        "lawn mower", "rake", "shovel", "paint can", "cooler"
    ],

    "laundry_room": [
        "washer", "dryer", "basket", "shelf", "cabinet", "sink",
        "iron", "ironing board", "detergent", "hanger", "rack", "counter",
        "window", "hamper", "folding table", "storage bin"
    ],

    "hallway": [
        "door", "light", "painting", "mirror", "table", "rug",
        "coat rack", "shoe rack", "bench", "stairs", "railing", "plant",
        "photograph", "sconce", "console table"
    ],

    "stairway": [
        "stairs", "railing", "landing", "window", "light", "carpet",
        "painting", "photograph", "bannister", "spindle", "newel post"
    ],

    "basement": [
        "stairs", "window", "shelf", "box", "furniture", "light",
        "pipe", "water heater", "furnace", "support beam", "storage",
        "workbench", "tool", "dehumidifier"
    ],

    "attic": [
        "box", "insulation", "beam", "window", "light", "trunk",
        "storage", "fan", "vent", "ladder", "christmas decorations"
    ],

    "closet": [
        "clothes", "hanger", "shelf", "shoe", "box", "mirror",
        "drawer", "rod", "basket", "light", "tie rack", "belt",
        "organizer", "hamper"
    ],

    "pantry": [
        "shelf", "food", "jar", "can", "box", "basket",
        "container", "spice rack", "bottle", "bag", "light"
    ],

    # Exterior scenes
    "exterior_front": [
        "house", "door", "window", "roof", "garage", "driveway",
        "lawn", "tree", "bush", "mailbox", "pathway", "porch",
        "light", "gutter", "siding", "brick", "fence", "flower"
    ],

    "exterior_back": [
        "house", "door", "window", "patio", "deck", "lawn",
        "tree", "fence", "garden", "grill", "furniture", "umbrella",
        "shed", "pathway", "light", "planter", "swing"
    ],

    "exterior_side": [
        "house", "window", "fence", "gate", "pathway", "lawn",
        "tree", "bush", "siding", "brick", "gutter", "light",
        "air conditioner", "meter"
    ],

    "yard": [
        "lawn", "tree", "bush", "flower", "fence", "pathway",
        "garden", "sprinkler", "shed", "bench", "bird bath",
        "planter", "rock", "mulch", "grass"
    ],

    "patio": [
        "furniture", "chair", "table", "umbrella", "grill", "plant",
        "light", "door", "railing", "cushion", "fire pit", "heater",
        "fan", "rug", "planter"
    ],

    "deck": [
        "railing", "chair", "table", "grill", "stairs", "light",
        "planter", "bench", "umbrella", "door", "wood", "post"
    ],

    "balcony": [
        "railing", "chair", "table", "plant", "door", "light",
        "planter", "view", "floor", "ceiling", "privacy screen"
    ],

    "driveway": [
        "car", "garage door", "concrete", "asphalt", "mailbox",
        "lamp post", "basketball hoop", "trash can", "gate"
    ],

    "pool_area": [
        "pool", "water", "chair", "umbrella", "table", "fence",
        "diving board", "ladder", "slide", "hot tub", "deck",
        "towel", "float", "light", "filter", "skimmer"
    ],

    "garden": [
        "plant", "flower", "tree", "bush", "pathway", "bench",
        "fountain", "statue", "planter", "trellis", "mulch",
        "rock", "grass", "vegetable", "herb", "fence"
    ],

    # Special/other
    "floor_plan": [
        "room", "wall", "door", "window", "dimension", "label",
        "square footage", "layout", "arrow", "measurement"
    ],

    "aerial_view": [
        "roof", "house", "property", "tree", "driveway", "lawn",
        "street", "neighbor", "pool", "fence", "boundary"
    ],

    "street_view": [
        "house", "street", "sidewalk", "tree", "car", "mailbox",
        "streetlight", "neighbor", "curb", "lawn", "driveway"
    ],

    "unknown": [
        "object", "item", "thing", "structure", "furniture", "fixture",
        "appliance", "decoration", "feature", "element"
    ]
}

# Common objects that appear in many scenes
COMMON_OBJECTS = [
    "window", "door", "light", "wall", "floor", "ceiling",
    "outlet", "switch", "vent", "smoke detector"
]

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
    confidence: float
    reasoning: str
    keywords: List[str]
    groundingdino_prompt: str
    alternatives: Optional[List[Tuple[str, float]]] = None
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
                       include_common: bool = True,
                       include_conditions: bool = False,
                       max_keywords: int = 25) -> List[str]:
    """
    Get detection keywords for a scene type.

    Args:
        scene: Scene type (e.g., 'living_room', 'kitchen')
        include_common: Add common objects like window, door, light
        include_conditions: Add defect/condition keywords
        max_keywords: Maximum number of keywords to return

    Returns:
        List of cleaned, deduplicated keywords
    """
    # Get scene-specific keywords
    base_keywords = list(SCENE_KEYWORDS.get(scene, SCENE_KEYWORDS["unknown"]))

    # Add common objects if requested
    if include_common:
        for obj in COMMON_OBJECTS:
            if obj not in base_keywords:
                base_keywords.append(obj)

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
                 include_common: bool = True,
                 include_conditions: bool = False,
                 max_keywords: int = 25,
                 debug: bool = False):
        self.lm_studio_url = lm_studio_url.rstrip('/')
        self.model_name = model_name
        self.include_common = include_common
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
        """Build the scene classification prompt with full scene list."""
        # Use the full list as JSON array for exact token matching
        scenes_json = json.dumps(VALID_SCENES, indent=2)

        return (
            "You are an expert at analyzing real estate property photos.\n\n"
            "TASK: Identify the single primary scene type (room/area) shown in this property photo.\n\n"
            f"VALID_SCENES = {scenes_json}\n\n"
            "RULES:\n"
            "- Choose exactly ONE value from VALID_SCENES above\n"
            "- Copy the scene name verbatim from the list (e.g., 'living_room', 'kitchen', 'exterior_front')\n"
            "- Consider the main subject of the photo, not background elements\n"
            "- If uncertain between options, pick the most likely one\n"
            "- Use 'unknown' only if truly unidentifiable\n"
            "- Do NOT invent new scene labels outside this list\n\n"
            "RESPOND ONLY WITH JSON (no extra text):\n"
            "{\n"
            '  "scene": "<one item from VALID_SCENES>",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "reasoning": "brief explanation",\n'
            '  "alternatives": [["<also from VALID_SCENES>", 0.0-1.0], ["<another from VALID_SCENES>", 0.0-1.0]]\n'
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

        # Check aliases
        if scene in SCENE_ALIASES:
            return SCENE_ALIASES[scene]

        # Check if it's already valid
        if scene in VALID_SCENES:
            return scene

        # Try to find closest match
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
                "max_tokens": 300,
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
                logger.info(f"Raw response: {raw_response[:200]}")

            parsed = self._extract_json(raw_response) or {}

            # Extract and normalize scene
            scene_raw = str(parsed.get("scene", "unknown"))
            scene = self._normalize_scene(scene_raw)

            # Extract other fields
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", ""))[:500]

            # Process alternatives if provided
            alternatives = []
            if "alternatives" in parsed and isinstance(parsed["alternatives"], list):
                for alt in parsed["alternatives"][:3]:  # Max 3 alternatives
                    if isinstance(alt, list) and len(alt) >= 2:
                        alt_scene = self._normalize_scene(str(alt[0]))
                        alt_conf = float(alt[1]) if len(alt) > 1 else 0.0
                        alternatives.append((alt_scene, alt_conf))

            # Generate keywords for the detected scene
            keywords = keywords_for_scene(
                scene,
                include_common=self.include_common,
                include_conditions=self.include_conditions,
                max_keywords=self.max_keywords
            )

            # Format as GroundingDINO prompt
            gdino_prompt = format_for_grounding_dino(keywords)

            result = SceneClassification(
                scene=scene,
                confidence=confidence,
                reasoning=reasoning,
                keywords=keywords,
                groundingdino_prompt=gdino_prompt,
                alternatives=alternatives if alternatives else None,
                processing_time=time.time() - t0,
                raw_response=raw_response if self.debug else None
            )

            if self.debug:
                logger.info(f"Classification: {scene} (confidence: {confidence:.2f})")
                logger.info(f"Keywords: {len(keywords)} objects")

            return result

        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {e}")
            return SceneClassification(
                scene="unknown",
                confidence=0.0,
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
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                    "keywords": classification.keywords,
                    "groundingdino_prompt": classification.groundingdino_prompt,
                    "alternatives": classification.alternatives,
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
        default=25,
        help="Maximum keywords per scene (default: 25)"
    )
    parser.add_argument(
        "--include-conditions",
        action="store_true",
        help="Include defect/condition keywords (crack, stain, damage, etc.)"
    )
    parser.add_argument(
        "--no-common",
        action="store_true",
        help="Exclude common objects (window, door, light, etc.)"
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
            kw_count = len(SCENE_KEYWORDS.get(scene, []))
            print(f"  {scene:20} - {kw_count} base keywords")
        sys.exit(0)

    # Setup classifier
    classifier = SceneClassifier(
        lm_studio_url=args.lm_studio_url,
        model_name=args.model,
        include_common=not args.no_common,
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
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "keywords": result.keywords,
            "groundingdino_prompt": result.groundingdino_prompt,
            "alternatives": result.alternatives,
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
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                    "keywords": classification.keywords,
                    "groundingdino_prompt": classification.groundingdino_prompt,
                    "alternatives": classification.alternatives,
                    "processing_time": classification.processing_time
                }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {args.output}")

        # Print summary
        print("\nClassification Summary:")
        print("-" * 50)
        for path, result in results.items():
            kw_count = len(result.keywords)
            print(f"{Path(path).name:30} → {result.scene:15} ({result.confidence:.1%}) [{kw_count} keywords]")


if __name__ == "__main__":
    main()