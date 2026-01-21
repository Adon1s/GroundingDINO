"""
Scene Classifier Pass Implementations
--------------------------------------
Individual pass functions for the scene classification pipeline.

Pass 1a: Scene Type Classification (fast, always Qwen)
Pass 1b: Feature/Market Appeal Notes - FREEFORM (premium uses GPT-5.2)
Pass 1c: Feature Notes → JSON Structuring (text-only)
Pass 2a: Issue Detection (premium uses GPT-5.2)
Pass 2b: Issue Verification (high volume, always Qwen)
Pass 3:  Keyword Extraction (text-only, from structured facts)
Pass 4:  Property Summary (premium uses GPT-5.2)
"""

from tools.llm_json import extract_json_object
import json
import logging
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper for detecting empty/no-issue notes
# ═══════════════════════════════════════════════════════════════════════════════

def _is_effectively_empty_notes(s: str) -> bool:
    """
    Check if notes string is effectively empty or indicates no findings.

    This prevents calling structuring models on "none" responses,
    which is where accidental hallucinations creep in.
    """
    if not s:
        return True
    t = s.strip().lower()
    return t in {"none", "no", "no issues", "no issue", "n/a", "na", "nothing", "nothing notable"}


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
    """Result from Pass 1b: Feature/market appeal FREEFORM notes."""
    feature_notes: str
    raw_response: Optional[str] = None


@dataclass
class Pass1cResult:
    """Result from Pass 1c: Feature notes structured into JSON."""
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
# Pass 1b: Feature/Market Appeal Notes (FREEFORM)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_1B_SYSTEM_PROMPT = "You are a real estate photo analyst"

PASS_1B_USER_PROMPT_TEMPLATE = ("This is a {scene} photo. Explain any visible features or finishes worth noting "
                                "(materials, fixtures, appliances, amenities). Do not mention issues, damage, "
                                "or drawbacks. If none, reply: none")


async def run_pass_1b_feature_notes(
        image_path: Path,
        vlm_client: Any,
        model_config: dict,
        context: Optional[Dict[str, Any]] = None,
) -> Pass1bResult:
    """
    Pass 1b: Extract freeform positive features from a property photo.

    This pass asks a vision-language model to identify any clearly visible
    positive features or upgrades that a realtor might want to highlight.
    The output is intentionally unstructured, plain text and may be:
      - a short list of features,
      - a brief sentence or two, or
      - the literal string "none" if no positives are visible.

    No formatting, categorization, or inference is required or expected here.
    All structuring and normalization is handled downstream in Pass 1c.

    The scene type (from Pass 1a) may be provided for context, but this pass
    does not enforce scene-specific formatting or content.

    Args:
        image_path: Path to the image file being analyzed.
        vlm_client: Vision-language model client used to analyze the image.
        model_config: Model configuration (provider, model name, token limits, etc.).
        context: Optional context from earlier passes (e.g., {'scene': 'kitchen'}).

    Returns:
        Pass1bResult containing:
            - feature_notes: Freeform text describing visible positive features,
              or "none" if no positives are present.
            - raw_response: The raw model response text.
    """
    scene = context.get('scene', 'property') if context else 'property'
    user_prompt = PASS_1B_USER_PROMPT_TEMPLATE

    logger.debug(f"Pass 1b: Generating feature notes for {image_path.name} (scene: {scene})")

    try:
        response = await vlm_client.analyze_image(
            image_path=image_path,
            system_prompt=PASS_1B_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )

        notes = (response or "").strip()

        return Pass1bResult(
            feature_notes=notes,
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 1b: Error generating feature notes: {e}")
        return Pass1bResult(
            feature_notes="",
            raw_response=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1c: Feature Notes → JSON Structuring
# ═══════════════════════════════════════════════════════════════════════════════

PASS_1C_SYSTEM_PROMPT = """You convert FREEFORM notes about visible features into STRICT JSON.

INPUT NOTES:
---
{notes}
---

Rules:
- Use ONLY what is stated in the notes. Do not add new features or claims.
- Keep language conservative and factual.
- Do NOT restate defects, problems, or renovation conclusions.
- If notes indicate none (e.g., "none"), output empty fields and [].
- notable_features must be a list of short strings (2–10 words), deduplicated.

Respond with ONLY a JSON object:
{{
  "notable_features": ["..."]
}}
"""

PASS_1C_USER_PROMPT = "Convert the notes into the JSON format."


async def run_pass_1c_feature_structuring(
        vlm_client: Any,
        model_config: dict,
        feature_notes: str,
) -> Pass1cResult:
    """
    Pass 1c: Convert freeform feature notes to structured JSON.

    This is a text-only pass that structures the freeform notes from Pass 1b.

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        feature_notes: Freeform notes from Pass 1b

    Returns:
        Pass1cResult with structured feature data
    """
    logger.debug("Pass 1c: Converting feature notes to JSON")

    if _is_effectively_empty_notes(feature_notes):
        return Pass1cResult(
            overall_impression="",
            image_summary="",
            notable_features=[],
            raw_response=None,
        )

    system_prompt = PASS_1C_SYSTEM_PROMPT.format(feature_notes=feature_notes)

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
        logger.error(f"Pass 1c: Error converting feature notes to JSON: {e}")
        return Pass1cResult(
            overall_impression="",
            image_summary="",
            notable_features=[],
            raw_response=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2a: Issue Detection (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2A_SYSTEM_PROMPT = "What do you see in this image that is negative that a real estate investor would like to know about. If nothing looks wrong or notable, reply with only the word 'none'"


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

PASS_2B_SYSTEM_PROMPT = """You convert raw photo notes into structured JSON issues.

FREEFORM NOTES (from a vision model analyzing a property photo):
---
{freeform_notes}
---

GOAL:
- Convert each distinct issue into one JSON item.
- Keep wording factual and conservative.

KEEP:
- Wear, damage, stains, discoloration, missing parts.
- Mentions of "older", "dated", or "outdated" as VALID signals.
  - Use rough_category = "opportunity" for these.

DROP:
- Subjective opinions with no visible condition.
- Anything not explicitly stated in the notes.

RULES:
- Do NOT invent or infer problems.
- Do NOT merge unrelated items.
- If location is unclear, use an empty string.

OUTPUT:
Return JSON only.

{{
  "issues_natural_language": [
    {{
      "description": "Calm factual restatement",
      "rough_category": "cosmetic | moisture | structure | systems | exterior | opportunity",
      "location_hint": ""
    }}
  ]
}}
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
    Returns:
        Pass2bResult with issues_natural_language
    """
    # If no freeform notes or effectively empty, return empty result (embeddings will fill catalog_flags when needed)
    if _is_effectively_empty_notes(freeform_notes):
        return Pass2bResult(
            issues_natural_language=[],
            catalog_flags={},
            raw_response=None,
        )

    # Format the system prompt with all placeholders
    system_prompt = PASS_2B_SYSTEM_PROMPT.format(
        freeform_notes=freeform_notes or "(no notes provided)",
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
            catalog_flags={},
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
- FEATURE NOTES: freeform positives/inventory notes from multiple photos
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
    This pass uses freeform notes from Pass 1b (features) and Pass 2a (issues).

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        all_results: Aggregated results from all images, containing:
            - 'scene': scene type
            - 'feature_notes': freeform notes from Pass 1b
            - 'issues_notes': freeform notes from Pass 2a

    Returns:
        Pass4Result with property summary
    """
    feature_blocks = []
    issues_blocks = []

    for img_key, data in list(all_results.items())[:20]:
        scene = data.get("scene", "unknown")
        feat = (data.get("feature_notes") or "").strip()
        neg = (data.get("issues_notes") or "").strip()

        if feat:
            feature_blocks.append(f"- {img_key} ({scene}): {feat}")
        if neg:
            issues_blocks.append(f"- {img_key} ({scene}): {neg}")

    user_prompt = (
            "FEATURE NOTES:\n---\n"
            + "\n".join(feature_blocks) + "\n---\n\n"
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


# ═══════════════════════════════════════════════════════════════════════════════
# Scene grouping helpers (needed by orchestrator + pass 4a/4b)
# ═══════════════════════════════════════════════════════════════════════════════

SCENE_GROUPS_UI: Dict[str, List[str]] = {
    "kitchen": ["kitchen", "pantry"],
    "bathroom": ["bathroom"],
    "bedroom": ["bedroom", "closet"],
    "living_areas": ["living_room", "dining_room", "home_office", "hallway", "stairway"],
    "utility": ["laundry_room", "basement", "attic", "garage", "hvac"],
    "exterior": ["exterior_front", "exterior_back", "exterior_side", "yard", "patio", "deck", "balcony", "driveway",
                 "pool", "garden"],
    "other": ["roof", "other", "unknown", "floor_plan", "aerial_view", "street_view"],
}

# Reverse lookup: scene -> group
SCENE_TO_GROUP: Dict[str, str] = {}
for _group, _scenes in SCENE_GROUPS_UI.items():
    for _scene in _scenes:
        SCENE_TO_GROUP[_scene] = _group


@dataclass
class Pass4aRoomSummariesResult:
    """Result from Pass 4a: Room-group summaries + issue counts."""
    room_summaries: Dict[str, str]
    issues_by_category: Dict[str, int]
    total_issues_found: int
    raw_response: Optional[str] = None


@dataclass
class Pass4bLegacyCardResult:
    """Result from Pass 4b: UI card fields (legacy)."""
    overall_condition: str
    overall_summary: str
    investment_verdict: str
    investment_rationale: str
    renovation_scope: str
    renovation_priorities: List[str]
    risk_flags: List[str]
    deferred_maintenance: List[str]
    raw_response: Optional[str] = None


PASS_4A_SYSTEM_PROMPT = """You generate conservative room-group summaries for a property photo analysis.

You will receive per-photo extracted facts (scene, positives, issues).
Rules:
- Be factual, conservative, and brief.
- Do NOT add new issues or features.
- If there is no evidence for a room group, output an empty string for that group.

Return ONLY JSON:
{
  "room_summaries": {
    "kitchen": "<1-3 sentences or ''>",
    "bathroom": "<1-3 sentences or ''>",
    "bedroom": "<1-3 sentences or ''>",
    "living_areas": "<1-3 sentences or ''>",
    "utility": "<1-3 sentences or ''>",
    "exterior": "<1-3 sentences or ''>",
    "other": "<1-3 sentences or ''>"
  }
}
"""

PASS_4B_SYSTEM_PROMPT = """You generate concise UI card fields for a property analysis.

Rules:
- Conservative, buyer/investor-friendly.
- Use ONLY what is provided. Do not invent issues/features.
- Keep it a few sentences to a paragraph.

Return ONLY JSON:
{
  "overall_condition": "excellent|good|fair|poor",
  "overall_summary": "<1-3 sentences>",
  "investment_verdict": "buy|maybe|pass",
  "investment_rationale": "<1-3 sentences>",
  "renovation_scope": "light|moderate|heavy",
  "renovation_priorities": ["<short>", "..."],
  "risk_flags": ["<short>", "..."],
  "deferred_maintenance": ["<short>", "..."]
}
"""


def _coerce_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return []


def _collect_issues(all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _, payload in (all_results or {}).items():
        issues = (payload or {}).get("issues_natural_language") or []
        if isinstance(issues, list):
            for it in issues:
                if isinstance(it, dict):
                    out.append(it)
    return out


async def run_pass_4a_room_summaries(
        vlm_client: Any,
        model_config: dict,
        all_results: Dict[str, Any],
        scene_counts: Optional[Dict[str, int]] = None,
) -> Pass4aRoomSummariesResult:
    """
    Pass 4a: Produce room-group summaries and deterministic issue counts.
    """
    # Deterministic counts (don't make the model do bookkeeping)
    issues = _collect_issues(all_results)
    issues_by_category: Dict[str, int] = {}
    for it in issues:
        cat = str(it.get("rough_category") or "other").strip() or "other"
        issues_by_category[cat] = issues_by_category.get(cat, 0) + 1
    total_issues_found = len(issues)

    # Build compact, grouped input
    groups: Dict[str, List[str]] = {k: [] for k in SCENE_GROUPS_UI.keys()}
    for img_key, data in (all_results or {}).items():
        scene = str((data or {}).get("scene") or "unknown").strip()
        group = SCENE_TO_GROUP.get(scene, "other")

        feat = str((data or {}).get("feature_notes") or "").strip()
        neg = str((data or {}).get("issues_notes") or "").strip()

        if feat:
            groups[group].append(f"- {img_key} ({scene}) FEAT: {feat}")
        if neg:
            groups[group].append(f"- {img_key} ({scene}) ISSUES: {neg}")

    # Keep prompts bounded
    for g in groups:
        groups[g] = groups[g][:30]

    user_payload = {
        "scene_counts": scene_counts or {},
        "grouped_notes": groups,
        "total_images_analyzed": len(all_results or {}),
        "total_issues_found": total_issues_found,
        "issues_by_category": issues_by_category,
    }

    try:
        response = await vlm_client.analyze_text(
            system_prompt=PASS_4A_SYSTEM_PROMPT,
            user_prompt=json.dumps(user_payload, ensure_ascii=False),
            **model_config,
        )
        result = extract_json_object(response) or {}
        rs = result.get("room_summaries") if isinstance(result.get("room_summaries"), dict) else {}

        # Ensure all keys exist
        room_summaries: Dict[str, str] = {}
        for k in SCENE_GROUPS_UI.keys():
            v = rs.get(k, "")
            room_summaries[k] = str(v).strip() if isinstance(v, str) else ""

        return Pass4aRoomSummariesResult(
            room_summaries=room_summaries,
            issues_by_category=issues_by_category,
            total_issues_found=total_issues_found,
            raw_response=response,
        )
    except Exception as e:
        logger.error(f"Pass 4a: Error generating room summaries: {e}")
        return Pass4aRoomSummariesResult(
            room_summaries={k: "" for k in SCENE_GROUPS_UI.keys()},
            issues_by_category=issues_by_category,
            total_issues_found=total_issues_found,
            raw_response=None,
        )


async def run_pass_4b_property_card_fields(
        vlm_client: Any,
        model_config: dict,
        room_summaries: Dict[str, Any],
        total_issues_found: int,
        total_images_analyzed: int,
        issues_by_category: Dict[str, int],
) -> Pass4bLegacyCardResult:
    """
    Pass 4b: Generate legacy UI card fields from room summaries + totals.
    """
    user_payload = {
        "room_summaries": room_summaries or {},
        "total_issues_found": int(total_issues_found or 0),
        "total_images_analyzed": int(total_images_analyzed or 0),
        "issues_by_category": issues_by_category or {},
    }

    try:
        response = await vlm_client.analyze_text(
            system_prompt=PASS_4B_SYSTEM_PROMPT,
            user_prompt=json.dumps(user_payload, ensure_ascii=False),
            **model_config,
        )
        result = extract_json_object(response) or {}

        return Pass4bLegacyCardResult(
            overall_condition=str(result.get("overall_condition") or "").strip(),
            overall_summary=str(result.get("overall_summary") or "").strip(),
            investment_verdict=str(result.get("investment_verdict") or "").strip(),
            investment_rationale=str(result.get("investment_rationale") or "").strip(),
            renovation_scope=str(result.get("renovation_scope") or "").strip(),
            renovation_priorities=_coerce_list(result.get("renovation_priorities")),
            risk_flags=_coerce_list(result.get("risk_flags")),
            deferred_maintenance=_coerce_list(result.get("deferred_maintenance")),
            raw_response=response,
        )
    except Exception as e:
        logger.error(f"Pass 4b: Error generating UI card fields: {e}")
        return Pass4bLegacyCardResult(
            overall_condition="",
            overall_summary="",
            investment_verdict="",
            investment_rationale="",
            renovation_scope="",
            renovation_priorities=[],
            risk_flags=[],
            deferred_maintenance=[],
            raw_response=None,
        )