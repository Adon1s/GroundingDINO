"""
Scene Classifier Pass Implementations
--------------------------------------
Individual pass functions for the scene classification pipeline.

Pass 1a: Scene Type Classification (fast, always Qwen)
Pass 1b: Positives/Inventory Notes - FREEFORM (premium uses GPT-5)
Pass 1c: Positives Notes → JSON Structuring (text-only)
Pass 2a: Issue Detection (premium uses GPT-5)
Pass 2b: Issue Verification (high volume, always Qwen)
Pass 3:  Keyword Extraction (text-only, from structured facts)
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
    """Result from Pass 1b: Positives/Inventory FREEFORM notes."""
    positives_notes: str
    raw_response: Optional[str] = None


@dataclass
class Pass1cResult:
    """Result from Pass 1c: Positives notes structured into JSON."""
    overall_impression: str
    image_summary: Optional[str] = None
    notable_features: Optional[List[str]] = None
    raw_response: Optional[str] = None


@dataclass
class Pass2aResult:
    """Result from Pass 2a: Issue Detection (freeform notes)."""
    issues_notes: str
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
    confidence: Optional[float] = None
    raw_response: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1a: Scene Type Classification (UNCHANGED)
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

    except Exception as e:
        logger.error(f"Pass 1a: Error classifying scene: {e}")
        return Pass1aResult(
            scene='other',
            reasoning=f"Error: {e}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1b: Positives/Inventory Notes (FREEFORM - NEW)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_1B_SYSTEM_PROMPT = "What positive features or upgrades do you see in this photo that a realtor might want to highlight? Only mention things that are clearly visible. There might not be any positives at all"

PASS_1B_USER_PROMPT_TEMPLATE = """Write positives/inventory notes for this {scene} photo in this exact format:

IMPRESSION: <1-2 conservative sentences>
SUMMARY: <1 factual sentence describing what's visible>
FEATURES:
- <feature>
- <feature>
(If none, write FEATURES: - none)
"""


async def run_pass_1b_positive_notes(
    image_path: Path,
    vlm_client: Any,
    model_config: dict,
    context: Optional[Dict[str, Any]] = None,
) -> Pass1bResult:
    """
    Pass 1b: Generate freeform positives/inventory notes for the image.

    This pass focuses on identifying positive features visible in the photo.
    Uses GPT-5 in premium mode for better reasoning.
    Output is plain text (no JSON parsing).

    Args:
        image_path: Path to the image file
        vlm_client: VLM client instance
        model_config: Model configuration
        context: Optional context from Pass 1a (e.g., {'scene': 'kitchen'})

    Returns:
        Pass1bResult with freeform positives notes
    """
    scene = context.get('scene', 'property') if context else 'property'
    user_prompt = PASS_1B_USER_PROMPT_TEMPLATE.format(scene=scene)

    logger.debug(f"Pass 1b: Generating positives notes for {image_path.name} (scene: {scene})")

    try:
        response = await vlm_client.analyze_image(
            image_path=image_path,
            system_prompt=PASS_1B_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )

        notes = (response or "").strip()

        return Pass1bResult(
            positives_notes=notes,
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 1b: Error generating positives notes: {e}")
        return Pass1bResult(
            positives_notes="",
            raw_response=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1c: Positives Notes → JSON Structuring (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_1C_SYSTEM_PROMPT = """You convert FREEFORM positives/inventory notes about a real estate photo into STRICT JSON.

INPUT NOTES:
---
{positives_notes}
---

Rules:
- Use ONLY what is stated in the notes. Do not add new features or claims.
- Keep language conservative and factual.
- If the notes do not contain enough information for overall_impression or image_summary, output "" for that field (do not infer).
- If the notes indicate there are no positives, output an empty notable_features list.
- notable_features must be a list of short strings (2–10 words each), deduplicated.

Output JSON only (no markdown):
{{
  "overall_impression": "<1-2 conservative sentences>",
  "image_summary": "<1-2 factual sentences>",
  "notable_features": ["<feature1>", "<feature2>", ...]
}}
"""

PASS_1C_USER_PROMPT = "Convert the notes into the JSON format."


async def run_pass_1c_positive_structuring(
    vlm_client: Any,
    model_config: dict,
    positives_notes: str,
) -> Pass1cResult:
    """
    Pass 1c: Convert freeform positives notes to structured JSON.

    This is a text-only pass that structures the freeform notes from Pass 1b.

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        positives_notes: Freeform notes from Pass 1b

    Returns:
        Pass1cResult with structured positives data
    """
    logger.debug("Pass 1c: Converting positives notes to JSON")

    if not positives_notes or not positives_notes.strip():
        return Pass1cResult(
            overall_impression="",
            image_summary="",
            notable_features=[],
            raw_response=None,
        )

    system_prompt = PASS_1C_SYSTEM_PROMPT.format(positives_notes=positives_notes)

    try:
        response = await vlm_client.analyze_text(
            system_prompt=system_prompt,
            user_prompt=PASS_1C_USER_PROMPT,
            **model_config,
        )

        result = extract_json_object(response) or {}

        nf = result.get("notable_features") or []
        if isinstance(nf, str):
            nf = [nf]
        elif not isinstance(nf, list):
            nf = []
        nf = [str(x).strip() for x in nf if str(x).strip()]
        nf = list(dict.fromkeys(nf))  # preserve order, dedupe

        return Pass1cResult(
            overall_impression=result.get("overall_impression", ""),
            image_summary=result.get("image_summary", ""),
            notable_features=nf,
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 1c: Error converting positives notes to JSON: {e}")
        return Pass1cResult(
            overall_impression="",
            image_summary="",
            notable_features=[],
            raw_response=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2a: Issue Detection (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2A_SYSTEM_PROMPT = "What issues do you see that a realtor might want to know about? Dont be overdramatic. There might not be any issues at all"


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
            system_prompt=PASS_2A_SYSTEM_PROMPT,
            user_prompt="Analyze this image for any issues, defects, or concerns.",
            **model_config,
        )

        # ✅ Do NOT parse JSON. Treat as freeform notes.
        freeform = (response or "").strip()

        logger.debug(f"Pass 2a freeform length: {len(freeform)} chars")

        return Pass2aResult(
            issues_notes=freeform,
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 2a: Error detecting issues: {e}")
        return Pass2aResult(
            issues_notes="",
            raw_response=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2b: Freeform Issue JSON Conversion (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2B_SYSTEM_PROMPT = """You convert noisy property-photo notes into STRICT JSON. Be conservative and high-signal.

Scene: {scene}
Catalog IDs: {catalog_list_str}

Notes:
{freeform_notes}

Catalog reference:
{catalog_text}

Rules:
- Output ONLY the most important visible defects (0–3 items). Skip tiny cosmetic wear, staging/clutter, preference/layout, dont worry about GFCI outlets.
- Do NOT turn "not visible" into "missing" (e.g., smoke detector not seen ≠ missing).
- Do NOT speculate. If wording is "may/might/could/possible" without a clearly described visible defect → present="uncertain", severity="none".
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

    except Exception as e:
        logger.error(f"Pass 2b: Error converting notes to JSON: {e}")
        return Pass2bResult(
            issues_natural_language=[],
            catalog_flags={},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 3: Keyword Extraction (UPDATED - text-only from structured facts)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_3_SYSTEM_PROMPT = """You generate object-detection keywords for a real estate photo using ONLY provided extracted facts.

Scene: {scene}

Positives (visible features):
{notable_features_json}

Issues (visible concerns):
{issues_json}

Rules:
- Keywords must be visually detectable objects/materials (1–4 words).
- No speculation and no code/safety claims.
- Deduplicate and keep high-signal. Max 20 keywords.

Output JSON only:
{{
  "keywords": ["<keyword1>", "<keyword2>", ...],
  "categories": {{
    "structural": ["<kw>", ...],
    "fixtures": ["<kw>", ...],
    "condition": ["<kw>", ...],
    "style": ["<kw>", ...]
  }}
}}
"""

PASS_3_USER_PROMPT = "Generate detection keywords."


async def run_pass_3_keyword_extraction(
    vlm_client: Any,
    model_config: dict,
    context: Optional[Dict[str, Any]] = None,
    max_keywords: int = 20,
) -> Pass3Result:
    """
    Pass 3: Generate detection keywords from structured facts (text-only).

    Always uses Qwen for speed.
    Note: This is now a text-only pass that uses structured outputs from
    previous passes instead of re-analyzing the image.

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        context: Context containing 'scene', 'notable_features', 'issues_natural_language'
        max_keywords: Maximum keywords to return

    Returns:
        Pass3Result with extracted keywords
    """
    scene = context.get("scene", "property") if context else "property"
    notable_features = context.get("notable_features", []) if context else []
    issues = context.get("issues_natural_language", []) if context else []

    system_prompt = PASS_3_SYSTEM_PROMPT.format(
        scene=scene,
        notable_features_json=json.dumps(notable_features, ensure_ascii=False),
        issues_json=json.dumps(issues, ensure_ascii=False),
    )

    logger.debug("Pass 3: Generating keywords from structured facts (text-only)")

    try:
        response = await vlm_client.analyze_text(
            system_prompt=system_prompt,
            user_prompt=PASS_3_USER_PROMPT,
            **model_config,
        )

        result = extract_json_object(response) or {}
        keywords = (result.get("keywords") or [])[:max_keywords]

        return Pass3Result(
            keywords=keywords,
            keyword_categories=result.get("categories"),
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 3: Error extracting keywords: {e}")
        return Pass3Result(keywords=[], raw_response=None)


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 4: Property Summary (UPDATED - uses freeform notes only)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_4_SYSTEM_PROMPT = """You are a real estate investment analyst synthesizing property photo notes.

You will be given:
- POSITIVES NOTES: freeform positives/inventory notes from multiple photos
- ISSUES NOTES: freeform issues/concerns notes from multiple photos

Rules:
- Use ONLY what is explicitly stated in the notes. Do not add new features, issues, or assumptions.
- Keep it balanced: strengths + risks.
- Be conservative; avoid strong claims unless clearly supported by the notes.

Respond with ONLY a JSON object:
{
  "property_summary": "<2-3 sentence investment-focused summary grounded in the notes>",
  "investment_considerations": ["<fact-based point1>", "<fact-based point2>", ...],
  "estimated_condition": "excellent|good|fair|poor",
  "confidence": <0.0-1.0>
}
"""


async def run_pass_4_property_summary(
    vlm_client: Any,
    model_config: dict,
    all_results: Dict[str, Any],
) -> Pass4Result:
    """
    Pass 4: Generate property-level summary from all image analyses.

    Uses GPT-5 in premium mode for better synthesis.
    This pass uses freeform notes from Pass 1b (positives) and Pass 2a (issues).

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        all_results: Aggregated results from all images, containing:
            - 'scene': scene type
            - 'positives_notes': freeform notes from Pass 1b
            - 'issues_notes': freeform notes from Pass 2a

    Returns:
        Pass4Result with property summary
    """
    positives_blocks = []
    issues_blocks = []

    for img_key, data in list(all_results.items())[:20]:
        scene = data.get("scene", "unknown")
        pos = (data.get("positives_notes") or "").strip()
        neg = (data.get("issues_notes") or "").strip()

        if pos:
            positives_blocks.append(f"- {img_key} ({scene}): {pos}")
        if neg:
            issues_blocks.append(f"- {img_key} ({scene}): {neg}")

    user_prompt = (
        "POSITIVES NOTES:\n---\n"
        + "\n".join(positives_blocks) + "\n---\n\n"
        + "ISSUES NOTES:\n---\n"
        + "\n".join(issues_blocks) + "\n---\n"
        + f"\nTotal images analyzed: {len(all_results)}"
    )

    logger.debug(f"Pass 4: Generating property summary from freeform notes ({len(all_results)} images)")

    try:
        response = await vlm_client.analyze_text(
            system_prompt=PASS_4_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )

        result = extract_json_object(response) or {}

        return Pass4Result(
            property_summary=result.get("property_summary", ""),
            investment_considerations=result.get("investment_considerations", []),
            estimated_condition=result.get("estimated_condition"),
            confidence=result.get("confidence"),
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 4: Error generating summary: {e}")
        return Pass4Result(property_summary="", raw_response=None)