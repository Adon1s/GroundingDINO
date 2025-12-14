"""
Scene Classifier Pass Implementations
--------------------------------------
Individual pass functions for the scene classification pipeline.

Pass 1a: Scene Type Classification (fast, always Qwen)
Pass 1b: Overall Impression (premium uses GPT-5)
Pass 2a: Issue Detection (premium uses GPT-5)
Pass 2b: Issue Verification (high volume, always Qwen)
Pass 3:  Keyword Extraction (always Qwen)
Pass 4:  Property Summary (premium uses GPT-5)
"""

from llm_json import extract_json_object
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes for Pass Results
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Pass1aResult:
    """Result from Pass 1a: Scene Type Classification."""
    scene: str
    scene_confidence: Optional[float] = None
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class Pass1bResult:
    """Result from Pass 1b: Overall Impression."""
    overall_impression: str
    image_summary: Optional[str] = None
    notable_features: Optional[List[str]] = None
    raw_response: Optional[str] = None


@dataclass
class Pass2aResult:
    """Result from Pass 2a: Issue Detection (freeform notes)."""
    freeform_notes: str
    raw_response: Optional[str] = None


@dataclass
class Pass2bResult:
    """Result from Pass 2b: Freeform to JSON Conversion."""
    issues_natural_language: List[Dict[str, Any]]
    catalog_flags: Dict[str, Any]
    raw_response: Optional[str] = None


@dataclass
class Pass3Result:
    """Result from Pass 3: Keyword Extraction."""
    keywords: List[str]
    keyword_categories: Optional[Dict[str, List[str]]] = None
    raw_response: Optional[str] = None


@dataclass
class Pass4Result:
    """Result from Pass 4: Property Summary."""
    property_summary: str
    investment_considerations: Optional[List[str]] = None
    estimated_condition: Optional[str] = None
    raw_response: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1a: Scene Type Classification
# ═══════════════════════════════════════════════════════════════════════════════

PASS_1A_SYSTEM_PROMPT = """You are a real estate image classifier. Your task is to identify the scene type shown in a property photo.

Classify the image into exactly ONE of these categories:
- exterior_front: Front view of the property
- exterior_back: Back/rear view of the property
- exterior_side: Side view of the property
- living_room: Living room or family room
- kitchen: Kitchen area
- bedroom: Bedroom
- bathroom: Bathroom (full or half)
- dining_room: Dining room or eating area
- basement: Basement or cellar
- attic: Attic space
- garage: Garage (interior or exterior)
- yard: Yard, garden, or outdoor space
- pool: Pool or spa area
- roof: Roof view
- hvac: HVAC equipment, water heater, electrical panel
- other: Any other space not listed

Respond with ONLY a JSON object:
{
  "scene": "<category>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}"""

PASS_1A_USER_PROMPT = "Classify the scene type in this real estate photo."


async def run_pass_1a_scene_type(
    image_path: Path,
    vlm_client: Any,
    model_config: dict,
) -> Pass1aResult:
    """
    Pass 1a: Classify the scene type of an image.

    This is a fast, focused pass that only determines what type of space
    is shown in the image. Always uses Qwen for speed.

    Args:
        image_path: Path to the image file
        vlm_client: VLM client instance
        model_config: Model configuration (url, model name, etc.)

    Returns:
        Pass1aResult with scene classification
    """
    logger.debug(f"Pass 1a: Classifying scene type for {image_path.name}")

    try:
        response = await vlm_client.analyze_image(
            image_path=image_path,
            system_prompt=PASS_1A_SYSTEM_PROMPT,
            user_prompt=PASS_1A_USER_PROMPT,
            **model_config,
        )

        # Parse JSON response
        result = extract_json_object(response) or {}

        return Pass1aResult(
            scene=result.get('scene', 'other'),
            scene_confidence=result.get('confidence'),
            reasoning=result.get('reasoning'),
            raw_response=response,
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Pass 1a: Failed to parse JSON response: {e}")
        return Pass1aResult(
            scene='other',
            reasoning=f"Parse error: {e}",
            raw_response=str(response) if 'response' in dir() else None,
        )
    except Exception as e:
        logger.error(f"Pass 1a: Error classifying scene: {e}")
        return Pass1aResult(
            scene='other',
            reasoning=f"Error: {e}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1b: Overall Impression
# ═══════════════════════════════════════════════════════════════════════════════

PASS_1B_SYSTEM_PROMPT = """You are a real estate analyst providing first impressions of property photos.

Given a property photo, provide:
1. A concise overall impression (1-3 sentences) suitable for a potential buyer/investor
2. A brief image summary describing what's visible
3. Notable features worth highlighting

Consider: condition, style, quality, notable features, potential concerns.

Respond with ONLY a JSON object:
{
  "overall_impression": "<buyer-focused impression>",
  "image_summary": "<factual description of what's visible>",
  "notable_features": ["<feature1>", "<feature2>", ...]
}"""

PASS_1B_USER_PROMPT_TEMPLATE = """Provide your overall impression of this {scene} photo."""


async def run_pass_1b_overall_impression(
    image_path: Path,
    vlm_client: Any,
    model_config: dict,
    context: Optional[Dict[str, Any]] = None,
) -> Pass1bResult:
    """
    Pass 1b: Generate overall impression of the image.

    This pass focuses on providing a buyer/investor-friendly impression.
    Uses GPT-5 in premium mode for better reasoning.

    Args:
        image_path: Path to the image file
        vlm_client: VLM client instance
        model_config: Model configuration
        context: Optional context from Pass 1a (e.g., {'scene': 'kitchen'})

    Returns:
        Pass1bResult with overall impression
    """
    scene = context.get('scene', 'property') if context else 'property'
    user_prompt = PASS_1B_USER_PROMPT_TEMPLATE.format(scene=scene)

    logger.debug(f"Pass 1b: Generating impression for {image_path.name} (scene: {scene})")

    try:
        response = await vlm_client.analyze_image(
            image_path=image_path,
            system_prompt=PASS_1B_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )

        result = extract_json_object(response) or {}

        return Pass1bResult(
            overall_impression=result.get('overall_impression', ''),
            image_summary=result.get('image_summary'),
            notable_features=result.get('notable_features', []),
            raw_response=response,
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Pass 1b: Failed to parse JSON response: {e}")
        # Try to extract impression from raw text
        raw = str(response) if 'response' in dir() else ''
        return Pass1bResult(
            overall_impression=raw[:200] if raw else "Unable to generate impression",
            raw_response=raw,
        )
    except Exception as e:
        logger.error(f"Pass 1b: Error generating impression: {e}")
        return Pass1bResult(
            overall_impression=f"Error generating impression: {e}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2a: Issue Detection
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2A_SYSTEM_PROMPT_TEMPLATE = "What issues do you see that a realtor might want to know about? Dont be overdramatic. There might not be any issues at all"


async def run_pass_2a_issue_detection(
    image_path: Path,
    vlm_client: Any,
    model_config: dict,
    context: Optional[Dict[str, Any]] = None,
    issue_catalog: Optional[Dict[str, Any]] = None,
) -> Pass2aResult:
    """
    Pass 2a: Detect issues and defects in the image (freeform notes).

    Uses GPT-5 in premium mode for better issue detection.
    Returns freeform text notes - JSON conversion happens in Pass 2b.

    Args:
        image_path: Path to the image file
        vlm_client: VLM client instance
        model_config: Model configuration
        context: Optional context from previous passes
        issue_catalog: Issue catalog (not used in 2a, but kept for signature consistency)

    Returns:
        Pass2aResult with freeform notes
    """
    logger.debug(f"Pass 2a: Detecting issues in {image_path.name}")

    try:
        response = await vlm_client.analyze_image(
            image_path=image_path,
            system_prompt=PASS_2A_SYSTEM_PROMPT_TEMPLATE,
            user_prompt="Analyze this image for any issues, defects, or concerns.",
            **model_config,
        )

        # ✅ Do NOT parse JSON. Treat as freeform notes.
        freeform = (response or "").strip()

        logger.debug(f"Pass 2a freeform length: {len(freeform)} chars")

        return Pass2aResult(
            freeform_notes=freeform,
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 2a: Error detecting issues: {e}")
        return Pass2aResult(
            freeform_notes="",
            raw_response=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2b: Freeform Issue json Conversion
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2B_SYSTEM_PROMPT = """You convert noisy property-photo notes into STRICT JSON. Be conservative and high-signal.

Scene: {scene}
Catalog IDs: {catalog_list_str}

Notes:
{freeform_notes}

Catalog reference:
{catalog_text}

Rules:
- Output ONLY the most important visible defects (0–3 items). Skip tiny cosmetic wear, staging/clutter, preference/layout comments.
- Do NOT turn “not visible” into “missing” (e.g., smoke detector not seen ≠ missing).
- Do NOT speculate. If wording is “may/might/could/possible” without a clearly described visible defect → present="uncertain", severity="none".
- If notes say no issues → issues_natural_language = [] and all catalog_flags present="no", severity="none".

Output JSON only (no markdown):
{{
  "issues_natural_language": [
    {{
      "description": "calm, factual, only what is visibly described in the notes",
      "rough_category": "cosmetic|moisture|structure|systems|exterior|opportunity",
      "location_hint": "where"
    }}
  ],
  "catalog_flags": {{
    "<issue_id>": {{
      "present": "yes|no|uncertain",
      "severity": "none|minor_repair|moderate_repair|full_replacement",
      "evidence": "short quote or empty"
    }}
  }}
}}

Catalog_flags requirements:
- Include EVERY issue_id from {catalog_list_str}.
- If present != "yes" → severity MUST be "none".
- Only use moderate_repair/full_replacement when notes clearly imply substantial work.
"""



async def run_pass_2b_issue_verification(
    image_path: Path,
    vlm_client: Any,
    model_config: dict,
    freeform_notes: str,
    context: Optional[Dict[str, Any]] = None,
    issue_catalog: Optional[Dict[str, Any]] = None,
) -> Pass2bResult:
    """
    Pass 2b: Convert freeform notes from Pass 2a into structured JSON.

    Always uses Qwen for speed (high volume conversion).
    Note: This is a text-only pass - does not need the image.

    Args:
        image_path: Path to the image file (kept for signature consistency, not used)
        vlm_client: VLM client instance
        model_config: Model configuration
        freeform_notes: Freeform notes from Pass 2a
        context: Optional context from previous passes
        issue_catalog: Issue catalog for structured flagging

    Returns:
        Pass2bResult with issues_natural_language and catalog_flags
    """
    # Build catalog data for prompt (needed for both early return and full run)
    defect_items = (issue_catalog or {}).get("defect_issues", []) or []
    opp_items = (issue_catalog or {}).get("opportunity_flags", []) or []

    all_items = []
    for x in defect_items + opp_items:
        if isinstance(x, dict) and x.get("id"):
            all_items.append(x)

    # If no freeform notes, return empty issues but populate catalog_flags with "no"
    if not freeform_notes or not freeform_notes.strip():
        empty_flags = {}
        for item in all_items:
            issue_id = item["id"]
            empty_flags[issue_id] = {
                "present": "no",
                "severity": "none",
                "evidence": "",
            }
        return Pass2bResult(
            issues_natural_language=[],
            catalog_flags=empty_flags,
            raw_response=None,
        )

    # Build context for prompt
    scene = context.get("scene", "property") if context else "property"

    catalog_ids = [i["id"] for i in all_items]
    catalog_list_str = ", ".join(catalog_ids)

    catalog_text = "\n".join(
        f"- {i.get('id')}: {i.get('name', '')}"
        for i in all_items
    )

    # Format the system prompt with all placeholders
    system_prompt = PASS_2B_SYSTEM_PROMPT.format(
        scene=scene,
        catalog_list_str=catalog_list_str or "(none)",
        freeform_notes=freeform_notes or "(no notes provided)",
        catalog_text=catalog_text or "(no catalog provided)",
    )

    logger.debug(f"Pass 2b: Converting freeform notes to JSON for {image_path.name}")

    try:
        # ✅ Text-only call - does not need the image
        response = await vlm_client.analyze_text(
            system_prompt=system_prompt,
            user_prompt="Convert the notes into the JSON format.",
            **model_config,
        )

        result = extract_json_object(response) or {}

        return Pass2bResult(
            issues_natural_language=result.get("issues_natural_language", []),
            catalog_flags=result.get("catalog_flags", {}),
            raw_response=response,
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Pass 2b: Failed to parse JSON response: {e}")
        return Pass2bResult(
            issues_natural_language=[],
            catalog_flags={},
            raw_response=str(response) if 'response' in dir() else None,
        )
    except Exception as e:
        logger.error(f"Pass 2b: Error converting notes to JSON: {e}")
        return Pass2bResult(
            issues_natural_language=[],
            catalog_flags={},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 3: Keyword Extraction
# ═══════════════════════════════════════════════════════════════════════════════

PASS_3_SYSTEM_PROMPT = """You are extracting search keywords from a real estate property photo.

Generate keywords that would help detect objects and features in this image.
Focus on:
- Structural elements (walls, floors, ceilings, windows, doors)
- Fixtures (cabinets, counters, appliances, lighting)
- Condition indicators (stains, cracks, damage, wear)
- Style/quality markers (modern, dated, luxury, basic)

Respond with ONLY a JSON object:
{
  "keywords": ["<keyword1>", "<keyword2>", ...],
  "categories": {
    "structural": ["<kw>", ...],
    "fixtures": ["<kw>", ...],
    "condition": ["<kw>", ...],
    "style": ["<kw>", ...]
  }
}"""


async def run_pass_3_keyword_extraction(
    image_path: Path,
    vlm_client: Any,
    model_config: dict,
    context: Optional[Dict[str, Any]] = None,
    max_keywords: int = 20,
) -> Pass3Result:
    """
    Pass 3: Extract detection keywords from image.

    Always uses Qwen for speed.

    Args:
        image_path: Path to the image file
        vlm_client: VLM client instance
        model_config: Model configuration
        context: Optional context from previous passes
        max_keywords: Maximum keywords to return

    Returns:
        Pass3Result with extracted keywords
    """
    scene = context.get('scene', 'property') if context else 'property'

    user_prompt = f"Extract detection keywords for this {scene} photo. Maximum {max_keywords} keywords."

    logger.debug(f"Pass 3: Extracting keywords from {image_path.name}")

    try:
        response = await vlm_client.analyze_image(
            image_path=image_path,
            system_prompt=PASS_3_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )

        result = extract_json_object(response) or {}

        keywords = result.get('keywords', [])[:max_keywords]

        return Pass3Result(
            keywords=keywords,
            keyword_categories=result.get('categories'),
            raw_response=response,
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Pass 3: Failed to parse JSON response: {e}")
        return Pass3Result(
            keywords=[],
            raw_response=str(response) if 'response' in dir() else None,
        )
    except Exception as e:
        logger.error(f"Pass 3: Error extracting keywords: {e}")
        return Pass3Result(keywords=[])


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 4: Property Summary (aggregation pass)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_4_SYSTEM_PROMPT = """You are a real estate investment analyst synthesizing property photo analysis.

Given the analysis results from multiple property photos, create:
1. A cohesive property summary suitable for investors
2. Key investment considerations
3. Overall condition assessment

Respond with ONLY a JSON object:
{
  "property_summary": "<2-3 sentence investment-focused summary>",
  "investment_considerations": ["<point1>", "<point2>", ...],
  "estimated_condition": "excellent|good|fair|poor",
  "confidence": <0.0-1.0>
}"""


async def run_pass_4_property_summary(
    vlm_client: Any,
    model_config: dict,
    all_results: Dict[str, Any],
) -> Pass4Result:
    """
    Pass 4: Generate property-level summary from all image analyses.

    Uses GPT-5 in premium mode for better synthesis.
    Note: This pass may not receive images, just aggregated text.

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        all_results: Aggregated results from all images

    Returns:
        Pass4Result with property summary
    """
    # Format aggregated data for prompt
    images_summary = []
    for img_key, data in all_results.items():
        scene = data.get('scene', 'unknown')
        impression = data.get('overall_impression', '')
        issues = data.get('verified_issues', [])
        issue_count = len(issues) if isinstance(issues, list) else 0

        images_summary.append(
            f"- {img_key} ({scene}): {impression[:100]}... [{issue_count} issues]"
        )

    user_prompt = f"""Synthesize these property photo analyses into an investment summary:

{chr(10).join(images_summary[:20])}  # Limit to 20 images in prompt

Total images analyzed: {len(all_results)}"""

    logger.debug(f"Pass 4: Generating property summary from {len(all_results)} images")

    try:
        # Note: This may be a text-only call if VLM client supports it
        response = await vlm_client.analyze_text(
            system_prompt=PASS_4_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )

        result = extract_json_object(response) or {}

        return Pass4Result(
            property_summary=result.get('property_summary', ''),
            investment_considerations=result.get('investment_considerations', []),
            estimated_condition=result.get('estimated_condition'),
            raw_response=response,
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Pass 4: Failed to parse JSON response: {e}")
        return Pass4Result(
            property_summary="Unable to generate summary",
            raw_response=str(response) if 'response' in dir() else None,
        )
    except Exception as e:
        logger.error(f"Pass 4: Error generating summary: {e}")
        return Pass4Result(
            property_summary=f"Error generating summary: {e}",
        )