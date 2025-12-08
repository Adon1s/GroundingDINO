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
    """Result from Pass 2a: Issue Detection."""
    detected_issues: List[Dict[str, Any]]
    catalog_flags: Dict[str, Any]
    raw_response: Optional[str] = None


@dataclass
class Pass2bResult:
    """Result from Pass 2b: Issue Verification."""
    verified_issues: List[Dict[str, Any]]
    rejected_issues: List[Dict[str, Any]]
    verification_notes: Optional[str] = None
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
1. A concise overall impression (1-2 sentences) suitable for a potential buyer/investor
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

PASS_2A_SYSTEM_PROMPT_TEMPLATE = """You are a property inspector analyzing a {scene} photo for potential issues.

Identify any visible issues, defects, or concerns. For each issue found:
- Describe what you see
- Assess severity (minor_repair, moderate_repair, full_replacement)
- Note the location in the image
- Flag relevant catalog IDs from the issue catalog

Issue Catalog (use these IDs when applicable):
{catalog_excerpt}

Respond with ONLY a JSON object:
{{
  "detected_issues": [
    {{
      "description": "<what you see>",
      "severity": "<minor_repair|moderate_repair|full_replacement>",
      "location": "<where in image>",
      "catalog_ids": ["<id1>", "<id2>"],
      "confidence": <0.0-1.0>
    }}
  ],
  "catalog_flags": {{
    "<catalog_id>": {{
      "present": "yes|no|uncertain",
      "severity": "<severity>",
      "evidence": "<brief evidence>"
    }}
  }}
}}"""


async def run_pass_2a_issue_detection(
    image_path: Path,
    vlm_client: Any,
    model_config: dict,
    context: Optional[Dict[str, Any]] = None,
    issue_catalog: Optional[Dict[str, Any]] = None,
) -> Pass2aResult:
    """
    Pass 2a: Detect issues and defects in the image.
    
    Uses GPT-5 in premium mode for better issue detection.
    
    Args:
        image_path: Path to the image file
        vlm_client: VLM client instance
        model_config: Model configuration
        context: Optional context from previous passes
        issue_catalog: Issue catalog for flagging
    
    Returns:
        Pass2aResult with detected issues
    """
    scene = context.get('scene', 'property') if context else 'property'
    
    # Build catalog excerpt for prompt
    catalog_excerpt = ""
    if issue_catalog:
        defects = issue_catalog.get('defect_issues', [])[:10]  # Top 10 for prompt size
        catalog_excerpt = "\n".join(
            f"- {d.get('id')}: {d.get('name')}" for d in defects if isinstance(d, dict)
        )
    
    system_prompt = PASS_2A_SYSTEM_PROMPT_TEMPLATE.format(
        scene=scene,
        catalog_excerpt=catalog_excerpt or "(no catalog provided)",
    )
    
    logger.debug(f"Pass 2a: Detecting issues in {image_path.name}")
    
    try:
        response = await vlm_client.analyze_image(
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt="Analyze this image for any issues, defects, or concerns.",
            **model_config,
        )
        
        result = extract_json_object(response) or {}
        
        return Pass2aResult(
            detected_issues=result.get('detected_issues', []),
            catalog_flags=result.get('catalog_flags', {}),
            raw_response=response,
        )
        
    except json.JSONDecodeError as e:
        logger.warning(f"Pass 2a: Failed to parse JSON response: {e}")
        return Pass2aResult(
            detected_issues=[],
            catalog_flags={},
            raw_response=str(response) if 'response' in dir() else None,
        )
    except Exception as e:
        logger.error(f"Pass 2a: Error detecting issues: {e}")
        return Pass2aResult(
            detected_issues=[],
            catalog_flags={},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2b: Issue Verification
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2B_SYSTEM_PROMPT = """You are a quality control reviewer verifying detected property issues.

Review each proposed issue and determine if it's valid based on the image.
Be conservative - only verify issues you can clearly see.

For each issue:
- VERIFY if clearly visible and correctly identified
- REJECT if not visible, misidentified, or a false positive
- Provide brief justification

Respond with ONLY a JSON object:
{
  "verified": [
    {"issue_index": 0, "reason": "<why verified>"},
    ...
  ],
  "rejected": [
    {"issue_index": 1, "reason": "<why rejected>"},
    ...
  ],
  "notes": "<any overall observations>"
}"""


async def run_pass_2b_issue_verification(
    image_path: Path,
    vlm_client: Any,
    model_config: dict,
    detected_issues: List[Dict[str, Any]],
) -> Pass2bResult:
    """
    Pass 2b: Verify detected issues from Pass 2a.
    
    Always uses Qwen for speed (high volume verification).
    
    Args:
        image_path: Path to the image file
        vlm_client: VLM client instance
        model_config: Model configuration
        detected_issues: Issues from Pass 2a to verify
    
    Returns:
        Pass2bResult with verified and rejected issues
    """
    if not detected_issues:
        return Pass2bResult(
            verified_issues=[],
            rejected_issues=[],
            verification_notes="No issues to verify",
        )
    
    # Format issues for prompt
    issues_text = "\n".join(
        f"{i}. {issue.get('description', 'Unknown')} (severity: {issue.get('severity', 'unknown')})"
        for i, issue in enumerate(detected_issues)
    )
    
    user_prompt = f"Verify these detected issues:\n{issues_text}"
    
    logger.debug(f"Pass 2b: Verifying {len(detected_issues)} issues in {image_path.name}")
    
    try:
        response = await vlm_client.analyze_image(
            image_path=image_path,
            system_prompt=PASS_2B_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )
        
        result = extract_json_object(response) or {}
        
        # Map verified/rejected indices back to full issues
        verified = []
        rejected = []
        
        for v in result.get('verified', []):
            idx = v.get('issue_index', -1)
            if 0 <= idx < len(detected_issues):
                issue = detected_issues[idx].copy()
                issue['verification_reason'] = v.get('reason', '')
                verified.append(issue)
        
        for r in result.get('rejected', []):
            idx = r.get('issue_index', -1)
            if 0 <= idx < len(detected_issues):
                issue = detected_issues[idx].copy()
                issue['rejection_reason'] = r.get('reason', '')
                rejected.append(issue)
        
        return Pass2bResult(
            verified_issues=verified,
            rejected_issues=rejected,
            verification_notes=result.get('notes'),
            raw_response=response,
        )
        
    except json.JSONDecodeError as e:
        logger.warning(f"Pass 2b: Failed to parse JSON response: {e}")
        # Conservative fallback: verify all
        return Pass2bResult(
            verified_issues=detected_issues,
            rejected_issues=[],
            verification_notes=f"Parse error, defaulting to verify all: {e}",
        )
    except Exception as e:
        logger.error(f"Pass 2b: Error verifying issues: {e}")
        return Pass2bResult(
            verified_issues=detected_issues,
            rejected_issues=[],
            verification_notes=f"Error: {e}",
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
