"""
Scene Classifier Pass Implementations
--------------------------------------
Individual pass functions for the scene classification pipeline.

Pass 1a: Scene Type Classification (fast, always Qwen)
Pass 1b: Feature/Market Appeal Notes - FREEFORM (premium uses GPT-5.2) --DEPRECATED
Pass 1c: Feature Notes → JSON Structuring (text-only) --DEPRECATED
Pass 2a: Observations freeform (premium uses GPT-5.2)
Pass 2b: Observations → JSON (text-only)
Pass 2c: Label observations + debug/forward split (text-only)
Pass 2d: Resolve catalog item ID from candidates (text-only, optional)
Pass 2e: Normalize canonical issues and build display-filtered issues (Issue cleaning for UI. Rule-based, no LLM)
Pass 2f: Visual package verification (multi-image; per-room prompts for
         kitchen / bathroom via PASS_2F_ROOM_PROMPTS). Confirms / rejects /
         marks-uncertain a proposed renovation package against the photos.
         Visual-truth only; no pricing posture or cost estimation. Selector
         is room-keyed so future rooms (exterior, etc.) register here without
         touching run_pass_2f internals.
"""

from tools.llm_json import extract_json_object
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tools import pipeline_config as _pipeline_cfg
except Exception:
    _pipeline_cfg = None

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


# ─────────────────────────────────────────────────────────────────────────────
# Dimension-string detection (MLS floorplan labels like "12'6 x 10'")
# ─────────────────────────────────────────────────────────────────────────────

_DIM_RE = re.compile(
    r"""
    (?<!\w)                              # not preceded by a word char (no \b — avoids quote edge cases)
    \d{1,2}                              # first number (feet or plain)
    (?:\s*'\s*\d{1,2}\s*"?              # ...feet-inches: 14'6  or 14'6"
     |\s*'                              # ...feet only:   14'
     |\s*\d{1,2}\s*"                    # ...bare inches: 14"
    )?                                   # whole foot/inch group is optional -> matches plain "14"
    \s*[x\xd7]\s*                        # separator: x or x, with optional surrounding spaces
    \d{1,2}                              # second number
    (?:\s*'\s*\d{1,2}\s*"?
     |\s*'
     |\s*\d{1,2}\s*"
    )?
    (?!\w)                               # not followed by a word char
    """,
    re.IGNORECASE | re.VERBOSE,
)


def force_other_if_dimensions(labeled: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Force label to 'other' for any observation whose description contains
    a room-dimension string (e.g. "12'6 x 10'", "14 x 12", "10'x8'6\"").

    MLS floorplan overlay text frequently gets OCR'd into photo descriptions.
    These are measurement artefacts, not real observations, and should never
    reach labeled_forward (or the UI as defects/upgrades).
    """
    out = []
    for x in labeled or []:
        desc = str(x.get("description") or "").strip()
        if not desc:
            continue
        if _DIM_RE.search(desc) and x.get("label") in {"defect_or_damage", "upgrade_candidate"}:
            logger.debug(f"Pass 2c: Forcing label=other (dimension string) → {desc!r}")
            x2 = dict(x)
            x2["label"] = "other"
            out.append(x2)
        else:
            out.append(x)
    return out


def _cfg_value(name: str, default: Any) -> Any:
    if _pipeline_cfg is None:
        return default
    return getattr(_pipeline_cfg, name, default)


_ROUTING_COMPONENT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("baseboards", "baseboard"),
    ("baseboard", "baseboard"),
    ("shingles", "shingle"),
    ("shingle", "shingle"),
    ("gutters", "gutter"),
    ("gutter", "gutter"),
    ("downspout", "downspout"),
    ("fascia", "fascia"),
    ("soffit", "soffit"),
    ("siding", "siding"),
    ("brickwork", "brickwork"),
    ("brick", "brick"),
    ("mortar", "mortar"),
    ("stucco", "stucco"),
    ("driveway", "driveway"),
    ("walkway", "walkway"),
    ("concrete", "concrete"),
    ("asphalt", "asphalt"),
    ("deck", "deck"),
    ("porch", "porch"),
    ("fence", "fence"),
    ("chimney", "chimney"),
    ("foundation", "foundation"),
    ("slab", "slab"),
    ("trim", "trim"),
    ("roof", "roof"),
)

_ROUTING_VAGUE_CONDITION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("aging", "aged"),
    ("aged", "aged"),
    ("weathering", "weathered"),
    ("weathered", "weathered"),
)

_ROUTING_CONCRETE_CONDITION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("cracks", "crack"),
    ("crack", "crack"),
    ("staining", "stain"),
    ("stained", "stain"),
    ("stain", "stain"),
    ("discoloration", "discolor"),
    ("discolored", "discolor"),
    ("discolor", "discolor"),
    ("fading", "fade"),
    ("faded", "fade"),
    ("fade", "fade"),
    ("worn", "worn"),
    ("wear", "worn"),
    ("deteriorat", "deteriorated"),
    ("peeling", "peeling"),
    ("missing", "missing"),
    ("broken", "broken"),
    ("rusting", "rust"),
    ("rusted", "rust"),
    ("rust", "rust"),
    ("rotting", "rot"),
    ("rotted", "rot"),
    ("rot", "rot"),
    ("sagging", "sag"),
    ("sag", "sag"),
    ("leaking", "leak"),
    ("leak", "leak"),
    ("curling", "curl"),
    ("curled", "curl"),
    ("curl", "curl"),
    ("patchy", "patchy"),
    ("mold", "mold"),
    ("mildew", "mildew"),
)

_VAGUE_ROUTING_CONDITIONS = {canonical for _, canonical in _ROUTING_VAGUE_CONDITION_PATTERNS}
_SOFTENING_NEGATION_PATTERNS: Tuple[str, ...] = tuple(_cfg_value(
    "PASS_2D_ROUTING_NEGATION_PATTERNS",
    (
        r"\bno\s+visible\s+damage\b",
        r"\bno\s+damage\s+is\s+visible\b",
        r"\bwithout\s+visible\s+damage\b",
        r"\bintact\b",
        r"\bconsistent\s+with\s+(?:the\s+)?age\b",
    ),
))
_SPECIFIC_NEGATION_RULES: Tuple[Tuple[str, str], ...] = (
    (r"\bno\s+cracks?\s+(?:are\s+)?visible\b", "crack"),
    (r"\bwithout\s+cracks?\b", "crack"),
    (r"\bno\s+staining\s+visible\b", "stain"),
    (r"\bno\s+leaks?\s+(?:are\s+)?visible\b", "leak"),
)


def _ordered_unique(values: List[str]) -> Tuple[str, ...]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _normalize_signal_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _collect_signal_hits(text_lower: str, patterns: Tuple[Tuple[str, str], ...]) -> Tuple[str, ...]:
    hits: List[str] = []
    for needle, canonical in patterns:
        if re.search(r"\b" + re.escape(needle), text_lower):
            hits.append(canonical)
    return _ordered_unique(hits)


def _extract_component_terms(text: str) -> Tuple[str, ...]:
    return _collect_signal_hits(_normalize_signal_text(text), _ROUTING_COMPONENT_PATTERNS)


def _extract_condition_terms(text: str) -> Tuple[str, ...]:
    text_lower = _normalize_signal_text(text)
    vague = _collect_signal_hits(text_lower, _ROUTING_VAGUE_CONDITION_PATTERNS)
    concrete = _collect_signal_hits(text_lower, _ROUTING_CONCRETE_CONDITION_PATTERNS)
    return _ordered_unique(list(vague) + list(concrete))


def _analyze_visible_condition_signal(text: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], bool]:
    text_lower = _normalize_signal_text(text)
    component_hits = _collect_signal_hits(text_lower, _ROUTING_COMPONENT_PATTERNS)
    condition_hits = list(_extract_condition_terms(text_lower))

    negated_terms = {
        term
        for pattern, term in _SPECIFIC_NEGATION_RULES
        if re.search(pattern, text_lower)
    }
    effective_conditions = tuple(term for term in condition_hits if term not in negated_terms)
    blocked_by_negation = bool(negated_terms) or any(
        re.search(pattern, text_lower)
        for pattern in _SOFTENING_NEGATION_PATTERNS
    )
    return component_hits, effective_conditions, blocked_by_negation


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
class Pass2cShadowDecision:
    """Per-observation audit row for the Pass 2c shadow lane.

    The shadow lane re-checks observations Pass 2c labeled as `generic_presence`
    (or any label in `SHADOW_LANE_LABELS`) that nonetheless carry physical-
    condition language. It runs them through the matcher with widened kinds
    and decides whether a specific non-generic catalog item clearly beats the
    broad style/dated alternatives. Every shadow input gets one of these rows
    so we can measure recall regardless of whether promotion fired.
    """
    description: str
    label: str
    matched_components: Tuple[str, ...] = ()
    matched_conditions: Tuple[str, ...] = ()
    blocked_by_negation: bool = False
    evaluated: bool = False                       # True iff candidates were retrieved
    candidate_count: int = 0
    top_specific_id: Optional[str] = None
    top_specific_score: Optional[float] = None
    top_specific_kind: Optional[str] = None       # "defect" | "upgrade" | None
    top_generic_id: Optional[str] = None
    top_generic_score: Optional[float] = None
    second_specific_score: Optional[float] = None
    promoted: bool = False
    decision_reason: str = ""                     # see EVALUATOR REASONS below


# Decision-reason vocabulary for Pass2cShadowDecision.decision_reason.
# Stable string set — downstream tooling and dashboards depend on these labels.
SHADOW_DECISION_REASONS: Tuple[str, ...] = (
    "promoted",
    "no_signal",                                  # filtered upstream; shouldn't appear in evaluator output
    "no_candidates",
    "top_is_generic",
    "below_min_score",
    "below_min_margin",
    "specific_not_strong_enough_vs_generic",
)


@dataclass(frozen=True)
class KindRoutingDecision:
    """Routing decision for catalog retrieval after Pass 2c labeling."""
    original_kind: str
    expanded_kinds: Tuple[str, ...]
    reason: str
    matched_component_terms: Tuple[str, ...] = ()
    matched_condition_terms: Tuple[str, ...] = ()
    blocked_by_negation: bool = False


@dataclass
class Pass2dResult:
    """Result from Pass 2d: Resolved catalog item ID from candidates."""
    observation: str
    resolved_item_id: Optional[str] = None
    resolved_kind: Optional[str] = None  # "defect" or "upgrade"
    raw_response: Optional[str] = None
    resolution_path: str = "llm"
    shortcut_reason: Optional[str] = None


class Pass2fInvalidResponseError(ValueError):
    """Raised when Pass 2f returns no parseable or actionable JSON decision."""


@dataclass
class Pass2fResult:
    """Result from Pass 2f visual package verification."""
    package_id: str
    package_type: str
    verification_status: str = "uncertain"  # confirmed | rejected | uncertain
    confirmed_issue_ids: List[str] = field(default_factory=list)
    rejected_issue_ids: List[str] = field(default_factory=list)
    evidence_summary: str = ""
    visible_room_count: str = "unclear"
    visible_room_count_evidence: str = ""
    raw_response: Optional[str] = None


def evaluate_kind_routing(description: str, kind: str) -> KindRoutingDecision:
    """Decide whether Pass 2d retrieval should stay single-kind or widen."""
    normalized_kind = (kind or "").strip().lower()
    if normalized_kind != "upgrade":
        expanded = (normalized_kind,) if normalized_kind in {"defect", "upgrade"} else ()
        return KindRoutingDecision(
            original_kind=normalized_kind,
            expanded_kinds=expanded,
            reason="non_upgrade_kind",
        )

    component_hits, condition_hits, blocked_by_negation = _analyze_visible_condition_signal(description)
    if not component_hits or not condition_hits:
        return KindRoutingDecision(
            original_kind=normalized_kind,
            expanded_kinds=("upgrade",),
            reason="no_visible_condition_signal",
            matched_component_terms=component_hits,
            matched_condition_terms=condition_hits,
            blocked_by_negation=blocked_by_negation,
        )

    only_vague_conditions = all(term in _VAGUE_ROUTING_CONDITIONS for term in condition_hits)
    if blocked_by_negation and only_vague_conditions:
        return KindRoutingDecision(
            original_kind=normalized_kind,
            expanded_kinds=("upgrade",),
            reason="blocked_by_negation",
            matched_component_terms=component_hits,
            matched_condition_terms=condition_hits,
            blocked_by_negation=True,
        )

    expanded_kinds = ("upgrade", "defect")
    return KindRoutingDecision(
        original_kind=normalized_kind,
        expanded_kinds=expanded_kinds,
        reason="visible_condition_signal",
        matched_component_terms=component_hits,
        matched_condition_terms=condition_hits,
        blocked_by_negation=blocked_by_negation,
    )


def is_generic_resolution_candidate(candidate: Dict[str, Any]) -> bool:
    return bool(candidate.get("drop_if_generic") or candidate.get("defaultHidden"))


def prioritize_resolution_candidates(
    candidates: List[Dict[str, Any]],
    *,
    widened_routing: bool = False,
) -> List[Dict[str, Any]]:
    ordered = list(candidates or [])
    if not widened_routing:
        return ordered
    return sorted(ordered, key=is_generic_resolution_candidate)


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2c shadow lane helpers
# ═══════════════════════════════════════════════════════════════════════════════


def has_physical_condition_signal(description: str) -> bool:
    """True if `description` carries an unambiguous component+condition pair.

    Used to decide whether a Pass 2c `generic_presence` (or `other`) observation
    deserves a second look from the matcher. We require both a component term
    (e.g. "roof", "tile") AND a condition term (e.g. "cracked", "stained"), and
    we reject negated/softened phrases like "no visible damage" or "intact".
    """
    components, conditions, blocked = _analyze_visible_condition_signal(description)
    return bool(components and conditions and not blocked)


def evaluate_pass_2c_shadow_candidate(
    description: str,
    candidates: List[Dict[str, Any]],
    *,
    min_score: float,
    min_margin: float,
    min_specific_over_generic: float,
    label: str = "",
) -> Pass2cShadowDecision:
    """Decide whether a shadow-lane observation qualifies for promotion.

    Promotion requires ALL of:
      - top non-generic candidate score >= `min_score`
      - top non-generic >= second non-generic by `min_margin`
      - top non-generic >= top generic candidate by `min_specific_over_generic`
        (so a specific defect must clearly beat any broad style/dated upgrade)

    `candidates` is the list returned by the same `candidate_provider` that
    Pass 2d uses, optionally already passed through `prioritize_resolution_candidates`.
    A non-generic candidate is one for which `is_generic_resolution_candidate`
    returns False (i.e. neither `drop_if_generic` nor `defaultHidden`).

    The returned `Pass2cShadowDecision` always populates `description` and
    `label`; remaining fields are populated based on what the candidate list
    contained. `decision_reason` is one of `SHADOW_DECISION_REASONS`.
    """
    components, conditions, blocked = _analyze_visible_condition_signal(description)
    decision = Pass2cShadowDecision(
        description=description,
        label=label,
        matched_components=tuple(components),
        matched_conditions=tuple(conditions),
        blocked_by_negation=blocked,
    )

    if not candidates:
        decision.decision_reason = "no_candidates"
        return decision

    decision.evaluated = True
    decision.candidate_count = len(candidates)

    # Partition by generic flag, preserving original (score-sorted) order.
    specific_candidates = [c for c in candidates if not is_generic_resolution_candidate(c)]
    generic_candidates = [c for c in candidates if is_generic_resolution_candidate(c)]

    if generic_candidates:
        top_generic = generic_candidates[0]
        decision.top_generic_id = _candidate_item_id(top_generic) or None
        decision.top_generic_score = float(top_generic.get("score") or 0.0)

    if not specific_candidates:
        decision.decision_reason = "top_is_generic"
        return decision

    top_specific = specific_candidates[0]
    top_specific_score = float(top_specific.get("score") or 0.0)
    decision.top_specific_id = _candidate_item_id(top_specific) or None
    decision.top_specific_score = top_specific_score
    decision.top_specific_kind = (
        (top_specific.get("kind") or "").strip().lower() or None
    )
    if len(specific_candidates) > 1:
        decision.second_specific_score = float(specific_candidates[1].get("score") or 0.0)

    # The top OVERALL candidate is generic and the specific runner-up trails it:
    # this is the broad-style-beats-specific case the user explicitly called out.
    overall_top = candidates[0]
    if is_generic_resolution_candidate(overall_top):
        overall_top_score = float(overall_top.get("score") or 0.0)
        if (top_specific_score - overall_top_score) < min_specific_over_generic:
            decision.decision_reason = "specific_not_strong_enough_vs_generic"
            return decision

    if top_specific_score < min_score:
        decision.decision_reason = "below_min_score"
        return decision

    second_specific_score = decision.second_specific_score or 0.0
    if (top_specific_score - second_specific_score) < min_margin:
        decision.decision_reason = "below_min_margin"
        return decision

    # Final guard: specific must beat the best generic by min_specific_over_generic
    # even when the overall top wasn't generic (covers the case where prioritization
    # already pushed generics down).
    top_generic_score = decision.top_generic_score or 0.0
    if (top_specific_score - top_generic_score) < min_specific_over_generic:
        decision.decision_reason = "specific_not_strong_enough_vs_generic"
        return decision

    if decision.top_specific_kind not in {"defect", "upgrade"}:
        # Defensive: the matcher should always tag candidates with a kind. If it
        # doesn't, we can't safely route the promoted observation through Pass 2d.
        decision.decision_reason = "specific_not_strong_enough_vs_generic"
        return decision

    decision.promoted = True
    decision.decision_reason = "promoted"
    return decision


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
- closet
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

PASS_2A_SYSTEM_PROMPT = "You are a real estate photo analyst"
PASS_2A_USER_PROMPT = "What stands out here to a renovator"


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
- Be factual and non-speculative
- Do NOT infer causes, consequences, or hidden problems.
- If the notes are empty or contain just the word none or none as the last word, return an empty list.

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
- defect_or_damage: visible wear, damage, missing, broken, poor condition, or safety hazards (exposed wiring, tripping hazards, unsafe conditions)
- upgrade_candidate: dated/cheap fixture/finish that a renovator would likely replace or update
- good_condition: explicitly says looks good / intact / clean
- generic_presence: neutral existence of an item (e.g., “there is a door”)
- other: anything else

Rules:
- Do NOT add new observations.
- Use ONLY the provided descriptions.
- One label per item.
- If the description is advice/process language (e.g., “needs inspection”, “recommend evaluation”, “cannot determine from photo”), label “other”.
- If the description mentions hidden systems (structural/foundation, electrical, plumbing, HVAC) but does NOT mention a specific visible sign (e.g., stain, crack, leak, rust, exposed wire, damage), label “other”.
- If the description suggests a renovation action for a visible finish/surface (refinish/replace/update/paint) such as floors, cabinets, counters, fixtures, tile, paint, label “upgrade_candidate”.

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

VALID_LABELS = {"defect_or_damage", "upgrade_candidate", "good_condition", "generic_presence", "other"}


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
        scene: str = "other",
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
    user_prompt = f"Scene: {scene}\n\n" + safe_format_prompt(PASS_2C_USER_PROMPT_TEMPLATE, observations_json=observations_json)

    logger.debug("Pass 2c: Labeling observations (text-only)")

    try:
        response = await vlm_client.analyze_text(
            system_prompt=PASS_2C_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )
        result = extract_json_object(response) or {}
        labeled_debug = _coerce_labeled_2c(result.get("labeled"))

        # Override label → "other" for any observation that contains a room
        # dimension string (e.g. MLS floorplan overlays like "12'6 x 10'").
        labeled_debug = force_other_if_dimensions(labeled_debug)

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
- Choose 0 or 1 item_id whose name and trade best match the observation semantically.
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
        desc = str(c.get("description") or "").strip()
        support_any = c.get("support_any") or []
        if isinstance(support_any, str):
            support_any = [support_any]
        support_text = ", ".join(str(x).strip() for x in support_any[:6] if str(x).strip())

        parts = [f"- {item_id}", f"name={name}", f"trade={trade}", f"kind={kind}"]
        if desc:
            parts.append(f"description={desc}")
        if support_text:
            parts.append(f"support_terms={support_text}")
        if c.get("drop_if_generic"):
            parts.append("drop_if_generic=true")
        if c.get("defaultHidden"):
            parts.append("default_hidden=true")
        lines.append(" | ".join(parts))
    return "\n".join(lines) or "(none)"


def _candidate_support_terms(candidate: Dict[str, Any]) -> List[str]:
    raw = candidate.get("support_any") or []
    if isinstance(raw, str):
        raw = [raw]
    return [_normalize_signal_text(str(x)) for x in raw if str(x).strip()]


def _candidate_name_and_support_signal_terms(candidate: Dict[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    raw_support = candidate.get("support_any") or []
    if isinstance(raw_support, str):
        raw_support = [raw_support]
    support_text = " ".join(str(x).strip() for x in raw_support if str(x).strip())
    signal_text = " ".join(filter(None, [str(candidate.get("name") or "").strip(), support_text]))
    return _extract_component_terms(signal_text), _extract_condition_terms(signal_text)


def _resolved_kind_for_candidate(candidate: Optional[Dict[str, Any]], fallback_kind: str) -> str:
    if not isinstance(candidate, dict):
        return fallback_kind
    candidate_kind = (candidate.get("kind") or "").strip().lower()
    if candidate_kind in {"defect", "upgrade"}:
        return candidate_kind
    return fallback_kind


def _resolve_candidate_via_lexical_shortcut(
    observation: str,
    candidates: List[Dict[str, Any]],
    *,
    kind: str,
    kind_routing: Optional[KindRoutingDecision] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not candidates:
        return None, None, None
    if kind_routing and kind_routing.blocked_by_negation:
        return None, None, None

    top_candidate = candidates[0]
    if is_generic_resolution_candidate(top_candidate):
        return None, None, None

    top_score = float(top_candidate.get("score") or 0.0)
    second_score = float(candidates[1].get("score") or 0.0) if len(candidates) > 1 else 0.0
    min_score = float(_cfg_value("PASS_2D_SHORTCUT_MIN_SCORE", 0.72))
    min_margin = float(_cfg_value("PASS_2D_SHORTCUT_MIN_MARGIN", 0.03))
    if top_score < min_score or (top_score - second_score) < min_margin:
        return None, None, None

    observation_lower = _normalize_signal_text(observation)
    support_terms = _candidate_support_terms(top_candidate)
    if any(phrase and phrase in observation_lower for phrase in support_terms):
        resolved_id = _candidate_item_id(top_candidate)
        return resolved_id, _resolved_kind_for_candidate(top_candidate, kind), "support_phrase_hit"

    observation_components = set(_extract_component_terms(observation_lower))
    observation_conditions = set(_extract_condition_terms(observation_lower))
    candidate_components, candidate_conditions = _candidate_name_and_support_signal_terms(top_candidate)
    if observation_components.intersection(candidate_components) and observation_conditions.intersection(candidate_conditions):
        resolved_id = _candidate_item_id(top_candidate)
        return resolved_id, _resolved_kind_for_candidate(top_candidate, kind), "component_condition_overlap"

    return None, None, None


async def run_pass_2d(
        vlm_client: Any,
        model_config: dict,
        observation: str,
        candidates: List[Dict[str, Any]],
        kind: str = "defect",
        kind_routing: Optional[KindRoutingDecision] = None,
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
            resolution_path="llm",
        )

    resolved_id, resolved_kind, shortcut_reason = _resolve_candidate_via_lexical_shortcut(
        observation,
        candidates,
        kind=kind,
        kind_routing=kind_routing,
    )
    if resolved_id:
        logger.debug("Pass 2d: lexical shortcut resolved %r -> %s", observation[:60], resolved_id)
        return Pass2dResult(
            observation=observation,
            resolved_item_id=resolved_id,
            resolved_kind=resolved_kind or kind,
            raw_response=None,
            resolution_path="lexical_shortcut",
            shortcut_reason=shortcut_reason,
        )

    candidates_text = format_candidates_text(candidates)
    user_prompt = safe_format_prompt(
        PASS_2D_USER_PROMPT_TEMPLATE,
        observation=observation,
        candidates_text=candidates_text,
        kind=kind,
    )
    if kind_routing and len(kind_routing.expanded_kinds) > 1:
        user_prompt += (
            "\nAdditional routing guidance:\n"
            "- Both defect and upgrade candidates may be present.\n"
            "- If a specific physical-condition defect item and a broad dated/style upgrade are both plausible, prefer the specific physical-condition item.\n"
            "- Generic candidates marked drop_if_generic/default_hidden are lower priority than specific visible-condition matches.\n"
        )

    logger.debug(f"Pass 2d: Resolving catalog item for {kind} observation: {observation[:50]}...")

    try:
        response = await vlm_client.analyze_text(
            system_prompt=PASS_2D_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **model_config,
        )
        result = extract_json_object(response) or {}
        candidate_by_id = {
            _candidate_item_id(candidate): candidate
            for candidate in candidates
            if _candidate_item_id(candidate)
        }

        resolved_id = None
        resolved_kind = kind
        if isinstance(result, dict):
            # Accept both the new key and legacy keys
            resolved_id = result.get("resolved_item_id") or result.get("resolved_defect_id") or result.get("resolved_upgrade_id")
            if resolved_id is not None:
                resolved_id = str(resolved_id).strip() if resolved_id else None

        # Validate resolved_id exists in the candidate list
        if resolved_id:
            valid_ids = {_candidate_item_id(c) for c in candidates}
            if resolved_id not in valid_ids:
                logger.warning("Pass 2d: hallucinated ID %r, setting to None", resolved_id)
                resolved_id = None
            else:
                resolved_kind = _resolved_kind_for_candidate(candidate_by_id.get(resolved_id), kind)

        return Pass2dResult(
            observation=observation,
            resolved_item_id=resolved_id,
            resolved_kind=resolved_kind,
            raw_response=response,
            resolution_path="llm",
        )

    except Exception as e:
        logger.error(f"Pass 2d: Error resolving catalog item: {e}")
        return Pass2dResult(
            observation=observation,
            resolved_item_id=None,
            resolved_kind=kind,
            raw_response=None,
            resolution_path="llm",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2e: Canonical issue normalization + display filtering (rule-based, no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

# Default deny phrases — loaded from cfg.PASS_2E_DENY_PHRASES at call time when available
_DEFAULT_DENY_PHRASES: List[str] = [
    # Generic upgrade suggestions that aren't real defects
    "updated lighting",
    "update lighting",
    "new lighting",
    "replace lighting",
    "modern lighting",
    "add lighting",
    # Weasel-word improvement suggestions without a concrete finding
    "could be enhanced",
    "could be improved",
    "would benefit from",
    "might be improved",
    "consider updating",
    "consider adding",
]

# Regex patterns for generic advice without a concrete observation
_GENERIC_ADVICE_PATTERNS: List[str] = [
    r"\badd\b.*\blighting\b",
    r"\bupdate\b.*\blighting\b",
    r"\breplace\b.*\blighting\b",
    r"\bupgrade\b.*\blighting\b",
]


@dataclass
class Pass2eResult:
    """Result from Pass 2e: canonical truth lane plus display-filtered lane."""
    # Backward-compatible aliases:
    #   verified_issues == display_issues
    #   matched_issues == canonical_issues
    verified_issues: List[Dict[str, Any]]       # display lane (renovator/UI-facing)
    matched_issues: List[Dict[str, Any]] = field(default_factory=list)  # canonical lane alias
    canonical_issues: List[Dict[str, Any]] = field(default_factory=list)
    display_issues: List[Dict[str, Any]] = field(default_factory=list)
    removed_invalid: List[Dict[str, Any]] = field(default_factory=list)
    display_suppressed_issues: List[Dict[str, Any]] = field(default_factory=list)
    removed: List[Dict[str, Any]] = field(default_factory=list)         # alias for removed_invalid
    suppressed_issues: List[Dict[str, Any]] = field(default_factory=list)  # alias for display_suppressed_issues
    notes: Optional[str] = None
    # Telemetry counters
    input_count: int = 0
    deduped_count: int = 0
    final_count: int = 0
    removed_count: int = 0
    removed_reason_counts: Dict[str, int] = field(default_factory=dict)
    suppressed_reason_counts: Dict[str, int] = field(default_factory=dict)
    suppressed_samples: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep older call sites that only look at verified/matched working.
        if not self.display_issues and self.verified_issues:
            self.display_issues = self.verified_issues
        if not self.verified_issues and self.display_issues:
            self.verified_issues = self.display_issues
        if not self.canonical_issues and self.matched_issues:
            self.canonical_issues = self.matched_issues
        if not self.matched_issues and self.canonical_issues:
            self.matched_issues = self.canonical_issues
        if not self.removed_invalid and self.removed:
            self.removed_invalid = self.removed
        if not self.removed and self.removed_invalid:
            self.removed = self.removed_invalid
        if not self.display_suppressed_issues and self.suppressed_issues:
            self.display_suppressed_issues = self.suppressed_issues
        if not self.suppressed_issues and self.display_suppressed_issues:
            self.suppressed_issues = self.display_suppressed_issues


def _2e_norm_text(s: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", (s or "").strip())



def _2e_is_sanity_junk(description: str) -> Optional[str]:
    """True sanity check — returns removal reason or None if clean.

    Only catches issues that should never appear in *any* output:
    - empty/whitespace-only description
    """
    d = _2e_norm_text(description)
    if not d:
        return "empty_description"
    return None


# Speculation markers — upgrade descriptions containing these are excluded from final.
# Only modal uncertainty words are included here. Visual hedges like "appears" or
# "looks" are legitimate observational language and should NOT suppress upgrades.
# "likely" was also removed — it often accompanies factual assessments ("likely original").
_SPECULATION_MARKERS: List[str] = [
    "potential",    # "potential upgrade" / "potentially"
    "possibly",     # "possibly original"
    "might",        # "might need replacing"
    "could",        # "could benefit from"
    "suggesting",   # "suggesting wear"
]

# High-signal damage/condition tokens. If an issue's description contains any of
# these, it indicates a concrete visible finding — not just a style opinion.
# Used by Gate 2 to override tier_optional_suppressed (keep in final even if
# the catalog tier is "optional").
_HIGH_SIGNAL_DAMAGE_TOKENS: List[str] = [
    "worn", "damaged", "peeling", "cracked", "stained",
    "missing", "broken", "rust", "rusted", "rusting",
    "rot", "rotted", "rotting", "mold", "mildew",
    "warped", "sagging", "leaking", "leak", "discolored",
    "chipped", "scratched", "dented", "corroded", "frayed",
    "torn", "deteriorat",   # prefix-matches deteriorated/deteriorating
]


def _has_high_signal_damage(desc_lower: str) -> bool:
    """Return True if the lowered description contains any high-signal damage token.

    Uses word-boundary prefix matching so "deteriorat" catches
    "deteriorated" and "deteriorating", etc.
    """
    for token in _HIGH_SIGNAL_DAMAGE_TOKENS:
        if re.search(r"\b" + re.escape(token), desc_lower):
            return True
    return False


def _2e_policy_reason(
    issue: Dict[str, Any],
    deny_phrases: List[str],
    catalog_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Determine if an issue should be excluded from *final* (but kept in matched).

    Returns a reason string if the issue should be suppressed, else None.
    Policy gates (in priority order):
      1. drop_if_generic (catalog-driven kill switch)
      2. tier == optional (suppressed unless policy says include)
      3. speculative upgrade (speculation word in upgrade description)
      4. deny_phrase / generic_advice pattern match
    """
    policy = policy or {}
    desc_lower = _2e_norm_text(issue.get("description", "")).lower()
    kind = (issue.get("kind") or "").strip().lower()

    # Gate 1 — Catalog-driven kill switch
    cid = (issue.get("catalogItemId") or "").strip()
    meta = (catalog_meta or {}).get(cid, {}) if cid else {}
    if meta.get("drop_if_generic"):
        return "drop_if_generic"

    # Gate 2 — Tier inclusion (with visibility severity override)
    tier = meta.get("tier", "work")
    if tier == "optional" and not policy.get("include_optional", False):
        # Override: if description contains high-signal damage tokens,
        # keep in final even though catalog tier is "optional".
        # This prevents "final is empty" syndrome for items that are
        # technically optional but describe real visible damage.
        if not _has_high_signal_damage(desc_lower):
            return "tier_optional_suppressed"

    # Gate 3 — Speculation suppression (upgrades only)
    # Use prefix match (\b but no trailing \b) so "potential" catches "potentially" etc.
    if kind == "upgrade":
        for marker in _SPECULATION_MARKERS:
            if re.search(r"\b" + re.escape(marker), desc_lower):
                return "speculative_upgrade"

    # Gate 4 — Deny phrases / generic advice patterns
    for phrase in deny_phrases:
        if phrase.lower() in desc_lower:
            return "generic_advice"
    for pat in _GENERIC_ADVICE_PATTERNS:
        if re.search(pat, desc_lower):
            return "generic_advice"

    return None


def _2e_dedupe_key(issue: Dict[str, Any]) -> str:
    """
    Exact-artifact deduplication key for the canonical truth lane.

    This intentionally does not collapse by catalogItemId alone. Multiple rooms
    can legitimately share a catalog item; broad consolidation belongs in the
    display lane or the estimate-unit builder, not canonical 2e output.
    """
    source = _2e_norm_text(
        issue.get("source_photo_key")
        or issue.get("photo_key")
        or issue.get("source_photo_id")
        or issue.get("photo_id")
        or issue.get("image_key")
        or ""
    ).lower()
    cid = _2e_norm_text(
        issue.get("catalogItemId")
        or issue.get("catalog_item_id")
        or issue.get("resolved_item_id")
        or ""
    ).lower()
    kind = _2e_norm_text(issue.get("kind", "")).lower()
    desc = _2e_norm_text(issue.get("description", "")).lower()
    location = _2e_norm_text(issue.get("location_hint", "")).lower()
    return "|".join([
        f"src:{source}",
        f"kind:{kind}",
        f"cid:{cid}",
        f"desc:{desc}",
        f"loc:{location}",
    ])


async def run_pass_2e(
    *,
    vlm_client: Any,
    model_config: Dict[str, Any],
    verified_issues: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> Pass2eResult:
    """
    Pass 2e: Rule-based normalization, filtering, deduplication, and policy gating.

    No LLM call — deterministic and fast.

    Pipeline (in order):
      1. Sanity validate (empty desc, invalid kind) → removed
      2. Normalize description whitespace
      3. Strip scoring fields
      4. Deduplicate → produces matched_issues
      5. Apply policy gating → produces final_issues (verified_issues)

    Policy gates (applied to matched to decide final):
      - drop_if_generic (catalog-driven kill switch)
      - tier == optional suppression
      - speculation suppression for upgrades
      - deny phrase / generic advice suppression

    Context may include:
      - deny_phrases: List[str]
      - catalog_meta_by_id: Dict[str, Dict] (from orchestrator)
      - policy: Dict (include_optional, mode, etc.)
    """
    context = context or {}

    # Load deny phrases: context override → cfg → built-in defaults
    deny_phrases: List[str] = []
    if isinstance(context.get("deny_phrases"), list):
        deny_phrases = context["deny_phrases"]
    else:
        try:
            from tools import pipeline_config as _cfg  # type: ignore
            deny_phrases = list(getattr(_cfg, "PASS_2E_DENY_PHRASES", None) or [])
        except Exception:
            pass
    if not deny_phrases:
        deny_phrases = list(_DEFAULT_DENY_PHRASES)

    # Catalog meta and policy from context (injected by orchestrator)
    catalog_meta = context.get("catalog_meta_by_id") or {}
    policy = context.get("policy") or {}

    input_count = len(verified_issues or [])
    removed: List[Dict[str, Any]] = []
    removed_reason_counts: Dict[str, int] = {}
    seen: set = set()
    canonical: List[Dict[str, Any]] = []

    # ── Stage 1-4: sanity → normalize → strip → dedupe → matched_issues ──
    for issue in (verified_issues or []):
        if not isinstance(issue, dict):
            continue
        issue = dict(issue)

        desc = issue.get("description", "")

        # Stage 1a: Sanity — empty description
        sanity_reason = _2e_is_sanity_junk(desc)
        if sanity_reason:
            removed.append({**issue, "removed_reason": sanity_reason})
            removed_reason_counts[sanity_reason] = removed_reason_counts.get(sanity_reason, 0) + 1
            continue

        # Stage 1b: Sanity — kind must be defect or upgrade
        kind = (issue.get("kind") or "").strip().lower()
        if kind not in {"defect", "upgrade"}:
            reason = f"invalid_kind:{kind or 'missing'}"
            removed.append({**issue, "removed_reason": reason})
            removed_reason_counts[reason] = removed_reason_counts.get(reason, 0) + 1
            continue

        # Stage 2: Normalize description whitespace
        issue["description"] = _2e_norm_text(desc)

        # Stage 3: Strip ranking/scoring fields — never persisted past 2e
        issue.pop("topCandidateScore", None)
        issue.pop("top_candidate_score", None)
        issue.pop("score", None)
        issue.pop("severity", None)

        # Stage 4: Deduplicate
        key = _2e_dedupe_key(issue)
        if key in seen:
            removed.append({**issue, "removed_reason": "duplicate"})
            removed_reason_counts["duplicate"] = removed_reason_counts.get("duplicate", 0) + 1
            continue
        seen.add(key)

        canonical.append(issue)

    deduped_count = len(canonical)

    # ── Stage 5: Policy gating → final_issues ──
    display: List[Dict[str, Any]] = []
    suppressed_issues: List[Dict[str, Any]] = []
    suppressed_reason_counts: Dict[str, int] = {}
    suppressed_samples: List[Dict[str, Any]] = []
    _MAX_SUPPRESSED_SAMPLES = 10

    for issue in canonical:
        reason = _2e_policy_reason(issue, deny_phrases, catalog_meta, policy)
        if reason:
            suppressed_reason_counts[reason] = suppressed_reason_counts.get(reason, 0) + 1
            suppressed_issues.append({**issue, "suppressed_reason": reason})
            if len(suppressed_samples) < _MAX_SUPPRESSED_SAMPLES:
                suppressed_samples.append({
                    "issue_id": issue.get("issue_id"),
                    "description": (issue.get("description") or "")[:120],
                    "kind": issue.get("kind"),
                    "catalogItemId": issue.get("catalogItemId"),
                    "suppressed_reason": reason,
                })
            continue
        display.append(issue)

    final_count = len(display)
    removed_count = len(removed)

    logger.debug(
        "Pass 2e: input=%d canonical=%d display=%d removed=%d display_suppressed=%d",
        input_count, deduped_count, final_count, removed_count,
        sum(suppressed_reason_counts.values()),
    )
    if suppressed_reason_counts:
        logger.debug("Pass 2e suppression reasons: %s", suppressed_reason_counts)

    return Pass2eResult(
        verified_issues=display,
        matched_issues=canonical,
        canonical_issues=canonical,
        display_issues=display,
        removed_invalid=removed,
        display_suppressed_issues=suppressed_issues,
        removed=removed,
        suppressed_issues=suppressed_issues,
        notes="canonical_display_split_v1",
        input_count=input_count,
        deduped_count=deduped_count,
        final_count=final_count,
        removed_count=removed_count,
        removed_reason_counts=removed_reason_counts,
        suppressed_reason_counts=suppressed_reason_counts,
        suppressed_samples=suppressed_samples,
    )
# -----------------------------------------------------------------------------
# Pass 2f: Visual package verification (multi-image, visual truth only)
# -----------------------------------------------------------------------------

# ── Shared Pass 2f framework ─────────────────────────────────────────────────
# These two blocks compose into every per-room template; the room body slots in
# between them. The shared rules and the JSON output schema are defined once so
# kitchen / bathroom / future rooms cannot drift on visual-truth semantics or
# downstream parsing.

PASS_2F_SHARED_RULES = (
    "Rules:\n"
    "- Do not estimate prices, costs, repair scope, replacement scope, or rehab budgets.\n"
    "- Do not infer hidden damage or unseen rooms.\n"
    "- Confirm only when the package-level pattern is visibly supported.\n"
    "- Reject when the proposed evidence is not visible or clearly contradicted.\n"
    "- Use uncertain when the images are insufficient, ambiguous, cropped, too distant, or mixed.\n"
    "- Return only the requested JSON object.\n"
)

PASS_2F_OUTPUT_SCHEMA = (
    "Return exactly this JSON shape:\n"
    "{{\n"
    '  "verification_status": "confirmed" or "rejected" or "uncertain",\n'
    '  "confirmed_issue_ids": [],\n'
    '  "rejected_issue_ids": [],\n'
    '  "evidence_summary": "Brief visible-only explanation"\n'
    "}}\n"
)

PASS_2F_BATHROOM_OUTPUT_SCHEMA = (
    "Return exactly this JSON shape:\n"
    "{{\n"
    '  "verification_status": "confirmed" or "rejected" or "uncertain",\n'
    '  "confirmed_issue_ids": [],\n'
    '  "rejected_issue_ids": [],\n'
    '  "evidence_summary": "Brief visible-only explanation",\n'
    '  "visible_room_count": "one_room" or "multiple_rooms" or "unclear",\n'
    '  "visible_room_count_evidence": "Brief fixed-feature basis"\n'
    "}}\n"
)

# ── Kitchen Pass 2f prompt ───────────────────────────────────────────────────
# Self-contained: no bathroom / exterior vocabulary anywhere in the body.

PASS_2F_KITCHEN_SYSTEM_PROMPT = (
    "You are verifying a proposed real-estate renovation package from kitchen photos. "
    "Use only visible evidence in the supplied images. Your job is visual truth only: "
    "confirm, reject, or mark uncertain whether the proposed package is supported by the photos.\n\n"
    + PASS_2F_SHARED_RULES +
    "\nKitchen-specific guidance:\n"
    "- A modernization or \"outdated finishes\" package is clearly contradicted when "
    "the kitchen shows predominantly updated finishes (shaker or slab cabinetry in "
    "current colors, stainless appliances, modern countertops, recent backsplash and "
    "flooring). A single isolated dated detail is not sufficient to confirm a "
    "modernization package against an otherwise updated kitchen.\n"
    "- Strong kitchen support signals: cabinet face damage at severity >= 2, missing "
    "base cabinets exposing subfloor, dated honey-oak cabinetry, postform laminate "
    "counters, absent or minimal backsplash, dated appliance suites.\n"
)

PASS_2F_KITCHEN_USER_PROMPT = (
    "Analyze these kitchen photos together.\n\n"
    "Proposed package:\n"
    "- package_id: {package_id}\n"
    "- package_type: {package_type}\n"
    "- package_label: {package_label}\n\n"
    "Candidate evidence items:\n"
    "{evidence_json}\n\n"
    + PASS_2F_OUTPUT_SCHEMA
)

# ── Bathroom Pass 2f prompt ──────────────────────────────────────────────────
# Self-contained: no kitchen / exterior vocabulary anywhere in the body.

PASS_2F_BATHROOM_SYSTEM_PROMPT = (
    "You are verifying a proposed real-estate renovation package from bathroom photos. "
    "Use only visible evidence in the supplied images. Your job is visual truth only: "
    "confirm, reject, or mark uncertain whether the proposed package is supported by the photos.\n\n"
    + PASS_2F_SHARED_RULES +
    "\nBathroom-specific guidance:\n"
    "- A modernization or \"outdated finishes\" package is clearly contradicted when "
    "the bathroom shows predominantly updated finishes (recently-tiled shower surround, "
    "modern vanity in current colors, contemporary fixtures and faucet, new flooring, "
    "fresh paint, undermount or vessel sink, subway or large-format tile). A single "
    "isolated dated detail is not sufficient to confirm a modernization package against "
    "an otherwise updated bathroom.\n"
    "- Strong bathroom support signals: visible tile or grout damage at severity >= 2, "
    "exposed substrate where tile has fallen off, active water damage around tub/shower "
    "or vanity, missing or damaged vanity exposing plumbing, fixtures with corroded or "
    "stained finish, dated vanity-bar lighting, vintage tile patterns (pink/yellow/blue "
    "ceramic, hex tile floors), worn laminate vanity tops.\n"
    "- Common bathroom photo limitations: tight framing, mirror reflections, partial "
    "views of vanity or shower. Use uncertain when key surfaces are not visible in any "
    "of the supplied images.\n"
    "- Also report visible_room_count for audit only. Use one_room when the supplied "
    "photos appear visually consistent with one bathroom. Use multiple_rooms only when "
    "fixed bathroom features visibly conflict, such as vanity/cabinet style, tile "
    "color or pattern, tub/shower type, window or mirror placement, layout, or room "
    "geometry. Use unclear when the photos are too cropped or ambiguous.\n"
    "- Mixed-looking bathrooms do not automatically make the package uncertain. If one "
    "coherent visible bathroom supports the package, you may confirm that package and "
    "reject only unrelated issue IDs.\n"
)

PASS_2F_BATHROOM_USER_PROMPT = (
    "Analyze these bathroom photos together.\n\n"
    "Proposed package:\n"
    "- package_id: {package_id}\n"
    "- package_type: {package_type}\n"
    "- package_label: {package_label}\n\n"
    "Candidate evidence items:\n"
    "{evidence_json}\n\n"
    + PASS_2F_BATHROOM_OUTPUT_SCHEMA
)

# Per-room selector. Future rooms (exterior, etc.) register here without
# touching run_pass_2f internals.
PASS_2F_ROOM_PROMPTS = {
    "kitchen":  (PASS_2F_KITCHEN_SYSTEM_PROMPT,  PASS_2F_KITCHEN_USER_PROMPT),
    "bathroom": (PASS_2F_BATHROOM_SYSTEM_PROMPT, PASS_2F_BATHROOM_USER_PROMPT),
}

PASS_2F_VALID_STATUSES = {"confirmed", "rejected", "uncertain"}
PASS_2F_ROOM_COUNT_VALUES = {"one_room", "multiple_rooms", "unclear"}


def _normalize_issue_id_list(value: Any, valid_issue_ids: set) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for raw in value:
        item = str(raw or "").strip()
        if not item or item not in valid_issue_ids or item in out:
            continue
        out.append(item)
    return out


def _coerce_pass_2f(
    raw: Dict[str, Any],
    *,
    package_id: str,
    package_type: str,
    valid_issue_ids: set,
    room: str = "",
) -> Pass2fResult:
    status = str(raw.get("verification_status") or "uncertain").strip().lower()
    if status not in PASS_2F_VALID_STATUSES:
        status = "uncertain"
    confirmed_issue_ids = _normalize_issue_id_list(
        raw.get("confirmed_issue_ids"),
        valid_issue_ids,
    )
    rejected_issue_ids = _normalize_issue_id_list(
        raw.get("rejected_issue_ids"),
        valid_issue_ids,
    )
    if status == "rejected" and not rejected_issue_ids:
        rejected_issue_ids = sorted(valid_issue_ids)
    visible_room_count = "unclear"
    visible_room_count_evidence = ""
    if str(room or "").lower() == "bathroom":
        visible_room_count = str(raw.get("visible_room_count") or "unclear").strip().lower()
        if visible_room_count not in PASS_2F_ROOM_COUNT_VALUES:
            visible_room_count = "unclear"
        visible_room_count_evidence = str(
            raw.get("visible_room_count_evidence") or ""
        ).strip()[:240]
    return Pass2fResult(
        package_id=package_id,
        package_type=package_type,
        verification_status=status,
        confirmed_issue_ids=confirmed_issue_ids,
        rejected_issue_ids=rejected_issue_ids,
        evidence_summary=str(raw.get("evidence_summary") or "").strip()[:400],
        visible_room_count=visible_room_count,
        visible_room_count_evidence=visible_room_count_evidence,
    )


async def run_pass_2f(
    image_paths: List[Path],
    vlm_client: Any,
    model_config: dict,
    *,
    room: str,
    package_id: str,
    package_type: str,
    evidence_items: List[Dict[str, Any]],
    package_label: str = "Renovation package",
) -> Pass2fResult:
    """
    Pass 2f visual package verification.

    This pass confirms/rejects the visual truth of a package candidate across
    multiple representative images. It intentionally does not ask the model for
    pricing posture, repair/replace scope, or dollar estimates.

    The `room` argument selects a room-specific prompt template from
    PASS_2F_ROOM_PROMPTS. Each room template is self-contained (no cross-room
    vocabulary) to avoid attention bleed when verifying e.g. a bathroom package.
    """
    try:
        system_prompt, user_prompt_template = PASS_2F_ROOM_PROMPTS[room]
    except KeyError as exc:
        raise ValueError(
            f"Pass 2f: no prompt template registered for room={room!r}; "
            f"registered rooms: {sorted(PASS_2F_ROOM_PROMPTS)}"
        ) from exc

    valid_issue_ids = {
        str(issue_id)
        for item in (evidence_items or [])
        for issue_id in (item.get("issue_ids") or [])
        if str(issue_id or "").strip()
    }
    evidence_json = json.dumps(evidence_items or [], ensure_ascii=False, indent=2)
    user_prompt = safe_format_prompt(
        user_prompt_template,
        package_id=package_id,
        package_type=package_type,
        package_label=package_label,
        evidence_json=evidence_json,
    )

    logger.debug("Pass 2f: reviewing %s (room=%s) with %d images", package_id, room, len(image_paths or []))

    try:
        if hasattr(vlm_client, "analyze_images"):
            response = await vlm_client.analyze_images(
                image_paths=image_paths,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **model_config,
            )
        else:
            if not image_paths:
                raise ValueError("no review images supplied")
            response = await vlm_client.analyze_image(
                image_path=image_paths[0],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **model_config,
            )
        parsed = extract_json_object(response)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("missing JSON object")
        result = _coerce_pass_2f(
            parsed,
            package_id=package_id,
            package_type=package_type,
            valid_issue_ids=valid_issue_ids,
            room=room,
        )
        result.raw_response = response
        return result
    except Exception as e:
        logger.error("Pass 2f: error reviewing %s: %s", package_id, e)
        return Pass2fResult(
            package_id=package_id,
            package_type=package_type,
            verification_status="uncertain",
            evidence_summary=f"Pass 2f verification failed: {e}",
        )
