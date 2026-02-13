"""
Scene Classifier Pass Implementations
--------------------------------------
Individual pass functions for the scene classification pipeline.

Pass 1a: Scene Type Classification (fast, always Qwen)
Pass 1b: Feature/Market Appeal Notes - FREEFORM (premium uses GPT-5.2)
Pass 1c: Feature Notes → JSON Structuring (text-only)
Pass 2a: Observations freeform (premium uses GPT-5.2)
Pass 2b: Observations → JSON (text-only)
Pass 2c: Label observations + debug/forward split (text-only)
Pass 2d: Resolve catalog item ID from candidates (text-only, optional)
Pass 3:  Keyword Extraction (text-only, from structured facts)
"""

from tools.llm_json import extract_json_object
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def safe_format_prompt(template: str, **kwargs) -> str:
    """
    Makes str.format() safe even if the template contains JSON { } examples.

    It:
    1) temporarily protects intended placeholders like {notes}
    2) escapes all remaining { } into {{ }}
    3) restores placeholders and formats
    """
    for k in kwargs.keys():
        template = template.replace("{" + k + "}", f"@@@__{k}__@@@")

    template = template.replace("{", "{{").replace("}", "}}")

    for k in kwargs.keys():
        template = template.replace(f"@@@__{k}__@@@", "{" + k + "}")

    return template.format(**kwargs)


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
    confidence: Optional[float] = None
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
    overall_impression: str = ""
    image_summary: str = ""
    notable_features: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None


@dataclass
class Pass2aResult:
    """Result from Pass 2a: Observations freeform."""
    observations_freeform: str
    raw_response: Optional[str] = None


@dataclass
class Pass2bResult:
    """Result from Pass 2b: Observations → JSON."""
    observations: List[Dict[str, str]] = field(default_factory=list)  # [{"description": "..."}]
    raw_response: Optional[str] = None


@dataclass
class Pass2cResult:
    """Result from Pass 2c: Labeled observations with debug/forward split."""
    labeled_debug: List[Dict[str, str]] = field(default_factory=list)
    labeled_forward: List[Dict[str, str]] = field(default_factory=list)
    raw_response: Optional[str] = None


@dataclass
class Pass2dResult:
    """Result from Pass 2d: Resolved catalog item ID from candidates."""
    observation: str
    resolved_item_id: Optional[str] = None
    resolved_kind: Optional[str] = None  # "defect" or "upgrade"
    raw_response: Optional[str] = None


@dataclass
class Pass3Result:
    """Result from Pass 3: Keyword Extraction."""
    keywords: List[str]
    keyword_categories: Optional[Dict[str, List[str]]] = None
    raw_response: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1a: Scene Type Classification
# ═══════════════════════════════════════════════════════════════════════════════

PASS_1A_SYSTEM_PROMPT = """You are a real estate image classifier. Your task is to identify the scene type shown in a property photo.

Classify the image into exactly ONE of these categories:
- exterior_front
- exterior_back
- exterior_side
- living_room
- kitchen
- bedroom
- bathroom
- dining_room
- basement
- attic
- garage
- yard
- pool
- roof
- hvac
- other

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

        conf = None
        try:
            if result.get("confidence") is not None:
                conf = float(result.get("confidence"))
        except Exception:
            conf = None

        return Pass1aResult(
            scene=str(result.get("scene", "other")).strip() or "other",
            confidence=conf,
            reasoning=result.get("reasoning"),
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 1a: Error classifying scene: {e}")
        return Pass1aResult(
            scene="other",
            reasoning=f"Error: {e}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1b: Feature/Market Appeal Notes (FREEFORM)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_1B_SYSTEM_PROMPT = "You are a real estate photo analyst."
PASS_1B_USER_PROMPT_TEMPLATE = (
    "This is a {scene} photo. List any clearly visible features or finishes worth noting "
    "(materials, fixtures, appliances, amenities). Be factual and concise. "
    "Do not mention issues, damage, or drawbacks. If none, reply: none"
)


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
    scene = context.get("scene", "property") if context else "property"
    user_prompt = PASS_1B_USER_PROMPT_TEMPLATE.format(scene=scene)

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

PASS_1C_SYSTEM_PROMPT_TEMPLATE = """You convert FREEFORM notes about visible features into STRICT JSON.

INPUT NOTES:
---
{notes}
---

Rules:
- Use ONLY what is stated in the notes. Do not add new features or claims.
- Keep language conservative and factual.
- If notes indicate none (e.g., "none"), output empty fields and [].
- notable_features must be a list of short strings (2–10 words), deduplicated.

Respond with ONLY a JSON object:
{
  "overall_impression": "...",
  "image_summary": "...",
  "notable_features": ["..."]
}"""

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

    system_prompt = safe_format_prompt(PASS_1C_SYSTEM_PROMPT_TEMPLATE, notes=feature_notes)

    try:
        response = await vlm_client.analyze_text(
            system_prompt=system_prompt,
            user_prompt=PASS_1C_USER_PROMPT,
            **model_config,
        )

        result = extract_json_object(response) or {}

        # Parse overall_impression
        oi = result.get("overall_impression") or ""
        if not isinstance(oi, str):
            oi = ""
        oi = oi.strip()

        # Parse image_summary
        ims = result.get("image_summary") or ""
        if not isinstance(ims, str):
            ims = ""
        ims = ims.strip()

        # Parse notable_features
        nf = result.get("notable_features") or []
        if isinstance(nf, str):
            nf = [nf]
        elif not isinstance(nf, list):
            nf = []
        nf = [str(x).strip() for x in nf if str(x).strip()]
        nf = list(dict.fromkeys(nf))  # preserve order, dedupe

        return Pass1cResult(
            overall_impression=oi,
            image_summary=ims,
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
# Pass 2a: Observations Freeform (Vision)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2A_SYSTEM_PROMPT = ""  # blank on purpose
PASS_2A_USER_PROMPT = (
    "What stands out here to a renovator, based on what's visible? "
    "If nothing stands out, reply with only the word 'none'."
)


async def run_pass_2a(
        image_path: Path,
        vlm_client: Any,
        model_config: dict,
        context: Optional[Dict[str, Any]] = None,
) -> Pass2aResult:
    """
    Pass 2a: Detect observations in the image (freeform notes).

    Uses GPT-5 in premium mode for better observation detection.
    Returns freeform text notes - JSON conversion happens in Pass 2b.

    Args:
        image_path: Path to the image file
        vlm_client: VLM client instance
        model_config: Model configuration
        context: Optional context from previous passes

    Returns:
        Pass2aResult with freeform observations
    """
    logger.debug(f"Pass 2a: Detecting observations in {image_path.name}")

    try:
        response = await vlm_client.analyze_image(
            image_path=image_path,
            system_prompt=PASS_2A_SYSTEM_PROMPT,
            user_prompt=PASS_2A_USER_PROMPT,
            **model_config,
        )

        # Do NOT parse JSON. Treat as freeform notes.
        freeform = (response or "").strip()

        logger.debug(f"Pass 2a freeform length: {len(freeform)} chars")

        return Pass2aResult(
            observations_freeform=freeform,
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 2a: Error detecting observations: {e}")
        return Pass2aResult(
            observations_freeform="",
            raw_response=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2b: Observations → JSON (Text-only)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2B_SYSTEM_PROMPT_TEMPLATE = """You split FREEFORM photo notes into distinct, concrete observations.

INPUT NOTES:
---
{notes}
---

Rules:
- Only output observations explicitly stated or directly described in the notes.
- One observation per item.
- Description must be 5–25 words.
- Be factual and non-speculative.
- Do NOT infer causes, consequences, or hidden problems.
- If the notes are empty or the word "none", return an empty list.

Return JSON only:
{
  "observations": [
    { "description": "..." }
  ]
}"""

PASS_2B_USER_PROMPT = "Convert the notes into the JSON format."


def _coerce_observations_2b(x: Any) -> List[Dict[str, str]]:
    """Normalize Pass 2b observations to list of dicts with description."""
    if not isinstance(x, list):
        return []
    out = []
    for it in x:
        if isinstance(it, dict):
            desc = str(it.get("description") or "").strip()
        else:
            desc = str(it or "").strip()
        if desc:
            out.append({"description": desc})
    return out


async def run_pass_2b(
        vlm_client: Any,
        model_config: dict,
        observations_freeform: str,
) -> Pass2bResult:
    """
    Pass 2b: Convert freeform notes from Pass 2a into structured JSON.

    This is a text-only pass - does not need the image.

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        observations_freeform: Freeform notes from Pass 2a

    Returns:
        Pass2bResult with observations list
    """
    # If no freeform notes or effectively empty, return empty result
    if _is_effectively_empty_notes(observations_freeform):
        return Pass2bResult(
            observations=[],
            raw_response=None,
        )

    system_prompt = safe_format_prompt(PASS_2B_SYSTEM_PROMPT_TEMPLATE, notes=observations_freeform)

    logger.debug("Pass 2b: Converting observations freeform to JSON")

    try:
        response = await vlm_client.analyze_text(
            system_prompt=system_prompt,
            user_prompt=PASS_2B_USER_PROMPT,
            **model_config,
        )

        result = extract_json_object(response) or {}
        observations = _coerce_observations_2b(result.get("observations"))

        return Pass2bResult(
            observations=observations,
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 2b: Error converting observations to JSON: {e}")
        return Pass2bResult(
            observations=[],
            raw_response=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2c: Label Observations + Debug/Forward Split (Text-only)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2C_SYSTEM_PROMPT = """Label each observation with a simple type.

Allowed labels:
- defect_or_damage: visible wear, damage, missing, broken, poor condition
- safety: tripping hazards, exposed wiring, unsafe conditions
- upgrade_candidate: dated/cheap fixture/finish that a renovator would likely replace
- good_condition: explicitly says looks good / intact / clean
- generic_presence: neutral existence of an item (e.g., "there is a door")
- other: anything else

Rules:
- Do NOT add new observations.
- Use ONLY the provided descriptions.
- One label per item.

Return JSON only:
{
  "labeled": [
    { "description": "...", "label": "defect_or_damage|safety|upgrade_candidate|good_condition|generic_presence|other" }
  ]
}
"""

PASS_2C_USER_PROMPT_TEMPLATE = """OBSERVATIONS_JSON:
{observations_json}
"""

VALID_LABELS = {"defect_or_damage", "safety", "upgrade_candidate", "good_condition", "generic_presence", "other"}


def _coerce_labeled_2c(x: Any) -> List[Dict[str, str]]:
    """Normalize Pass 2c labeled observations."""
    if not isinstance(x, list):
        return []
    out: List[Dict[str, str]] = []
    for it in x:
        if not isinstance(it, dict):
            continue
        desc = str(it.get("description") or "").strip()
        if not desc:
            continue
        label = str(it.get("label") or "").strip().lower()
        if label not in VALID_LABELS:
            label = "other"
        out.append({
            "description": desc,
            "label": label,
        })
    return out


async def run_pass_2c(
        vlm_client: Any,
        model_config: dict,
        observations: List[Dict[str, str]],
) -> Pass2cResult:
    """
    Pass 2c: Label observations and split into debug/forward lists.

    labeled_debug: all labeled observations (for debugging)
    labeled_forward: only defect_or_damage and upgrade_candidate (for downstream)

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        observations: Observations from Pass 2b

    Returns:
        Pass2cResult with labeled_debug and labeled_forward
    """
    if not observations:
        return Pass2cResult(labeled_debug=[], labeled_forward=[], raw_response=None)

    observations_json = json.dumps(observations, ensure_ascii=False)
    user_prompt = safe_format_prompt(PASS_2C_USER_PROMPT_TEMPLATE, observations_json=observations_json)

    logger.debug("Pass 2c: Labeling observations (text-only)")

    try:
        response = await vlm_client.analyze_text(
            system_prompt=PASS_2C_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )
        result = extract_json_object(response) or {}
        labeled_debug = _coerce_labeled_2c(result.get("labeled"))

        # Split: labeled_forward = defect_or_damage + upgrade_candidate only
        labeled_forward = [
            x for x in labeled_debug
            if x.get("label") in {"defect_or_damage", "upgrade_candidate"}
        ]

        return Pass2cResult(
            labeled_debug=labeled_debug,
            labeled_forward=labeled_forward,
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 2c: Error labeling observations: {e}")
        return Pass2cResult(
            labeled_debug=[],
            labeled_forward=[],
            raw_response=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2d: Resolve Catalog Item ID from Candidates (Text-only, Optional)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2D_SYSTEM_PROMPT = ""
PASS_2D_USER_PROMPT_TEMPLATE = """Map this observation to a catalog item ID using ONLY the candidates.

OBSERVATION:
{observation}

ITEM KIND: {kind}

CANDIDATES:
{candidates_text}

Rules:
- Choose 0 or 1 item_id that best matches the observation.
- If none fit, return null.
- Use ONLY item_id values from the candidate list.

Return JSON only:
{{
  "resolved_item_id": "..." or null
}}
"""


def _candidate_item_id(c: Dict[str, Any]) -> str:
    """Extract the canonical item ID from a candidate dict, regardless of key convention."""
    return str(
        c.get("item_id")
        or c.get("defect_id")
        or c.get("upgrade_id")
        or c.get("id")
        or ""
    ).strip()


def format_candidates_text(candidates: List[Dict[str, Any]]) -> str:
    """Format candidate list to text for the prompt."""
    if not candidates:
        return "(none)"
    lines = []
    for c in candidates:
        item_id = _candidate_item_id(c)
        name = c.get("name", "")
        trade = c.get("trade_bucket", "")
        kind = c.get("kind", "")
        score = float(c.get("score", 0.0) or 0.0)
        lines.append(f"- {item_id} | {name} | trade={trade} | kind={kind} | score={score:.3f}")
    return "\n".join(lines) or "(none)"


async def run_pass_2d(
        vlm_client: Any,
        model_config: dict,
        observation: str,
        candidates: List[Dict[str, Any]],
        kind: str = "defect",
) -> Pass2dResult:
    """
    Pass 2d: Resolve a canonical catalog item ID from embedding candidates.

    Handles both defect and upgrade observations. The `kind` parameter
    controls prompt framing and is passed through to the result.

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        observation: The observation description string
        candidates: List of candidate dicts from embeddings retrieval
        kind: "defect" or "upgrade" — determines which pool was searched

    Returns:
        Pass2dResult with resolved_item_id and resolved_kind
    """
    if not observation or not candidates:
        return Pass2dResult(
            observation=observation,
            resolved_item_id=None,
            resolved_kind=kind,
            raw_response=None,
        )

    candidates_text = format_candidates_text(candidates)
    user_prompt = safe_format_prompt(
        PASS_2D_USER_PROMPT_TEMPLATE,
        observation=observation,
        candidates_text=candidates_text,
        kind=kind,
    )

    logger.debug(f"Pass 2d: Resolving catalog item for {kind} observation: {observation[:50]}...")

    try:
        response = await vlm_client.analyze_text(
            system_prompt=PASS_2D_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )
        result = extract_json_object(response) or {}

        resolved_id = None
        if isinstance(result, dict):
            # Accept both the new key and legacy keys
            resolved_id = result.get("resolved_item_id") or result.get("resolved_defect_id") or result.get("resolved_upgrade_id")
            if resolved_id is not None:
                resolved_id = str(resolved_id).strip() if resolved_id else None

        return Pass2dResult(
            observation=observation,
            resolved_item_id=resolved_id,
            resolved_kind=kind,
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"Pass 2d: Error resolving catalog item: {e}")
        return Pass2dResult(
            observation=observation,
            resolved_item_id=None,
            resolved_kind=kind,
            raw_response=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 3: Keyword Extraction (Text-only, from structured facts)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_3_SYSTEM_PROMPT_TEMPLATE = """You generate object-detection keywords using ONLY provided extracted facts.

Scene: {scene}

Features:
{features_json}

Observations:
{observations_json}

Rules:
- Keywords must be visually detectable objects/materials (1–4 words).
- Deduplicate and keep high-signal.

Output JSON only:
{
  "keywords": ["<kw>", "..."],
  "categories": {
    "structural": ["<kw>", "..."],
    "fixtures": ["<kw>", "..."],
    "condition": ["<kw>", "..."],
    "style": ["<kw>", "..."]
  }
}"""

PASS_3_USER_PROMPT = "Generate detection keywords."


async def run_pass_3_keyword_extraction(
        vlm_client: Any,
        model_config: dict,
        context: Optional[Dict[str, Any]] = None,
        max_keywords: int = 20,
) -> Pass3Result:
    """
    Pass 3: Generate detection keywords from structured facts (text-only).

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        context: Context containing:
            - 'scene': scene type from Pass 1a
            - 'features_struct': dict with 'notable_features' from Pass 1c
            - 'observations': list from Pass 2b (observations_struct.get("observations"))
            - 'labeled_forward': list from Pass 2c (preferred if non-empty)
        max_keywords: Maximum keywords to return

    Returns:
        Pass3Result with extracted keywords
    """
    scene = context.get("scene", "property") if context else "property"

    # Get notable_features from features_struct
    features_struct = context.get("features_struct", {}) if context else {}
    notable_features = features_struct.get("notable_features", [])

    # Use labeled_forward if non-empty, else fall back to observations
    labeled_forward = context.get("labeled_forward", []) if context else []
    observations = context.get("observations", []) if context else []

    observations_for_keywords = labeled_forward if labeled_forward else observations

    system_prompt = safe_format_prompt(
        PASS_3_SYSTEM_PROMPT_TEMPLATE,
        scene=scene,
        features_json=json.dumps(notable_features, ensure_ascii=False),
        observations_json=json.dumps(observations_for_keywords, ensure_ascii=False),
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