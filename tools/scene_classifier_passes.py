"""
Scene Classifier Pass Implementations
--------------------------------------
Individual pass functions for the scene classification pipeline.

Pass 1a: Scene Type Classification (fast, always Qwen)
Pass 1b: Feature/Market Appeal Notes - FREEFORM (premium uses GPT-5.2) --DEPRECATED
Pass 1c: Feature Notes → JSON Structuring (text-only) --DEPRECATED
Pass 2a: Observations freeform (premium uses GPT-5.2)
Pass 2b: Observations → JSON, atomic per condition claim (text-only)
Pass 2c: Classify observations under observation-kind-v2 (text-only).
         Emits kinds (defect | degradation | modernization) or exclusion
         reasons. The pipeline is classification-only until the catalog
         migration (Task 2) and downstream cutover (Task 3) land; results
         are non-publishable.
Pass 2d: Resolve catalog item ID from candidates (text-only, optional)
Pass 2e: Normalize canonical issues and build display-filtered issues (Issue cleaning for UI. Rule-based, no LLM)
Pass 2f: Visual package verification (multi-image; per-room prompts for
         kitchen / bathroom / bedroom / living / exterior via
         PASS_2F_ROOM_PROMPTS). Confirms / rejects /
         marks-uncertain a proposed renovation package against the photos.
         Visual-truth only; no pricing posture or cost estimation. Selector
         is room-keyed so future rooms (exterior, etc.) register here without
         touching run_pass_2f internals.
"""

from tools.llm_json import extract_json_object
from tools.pipeline_common import (
    PASS_1A_SCENE_IDS,
    normalize_scene_id,
    strip_term_marker,
    term_matches,
)
import hashlib
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


def _with_analysis_pass(model_config: dict, pass_label: str) -> dict:
    """Attach a pass label for VLM request logs without mutating caller config."""
    return {**(model_config or {}), "analysis_pass": pass_label}


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


# An overlay is a dimension plus a room name and nothing else. Anything longer
# is a real observation that merely cites a size ("cracked 12 x 12 floor tiles").
_DIM_OVERLAY_MAX_RESIDUE_CHARS = 15
_DIM_OVERLAY_MAX_RESIDUE_WORDS = 2


def _is_dimension_overlay(desc: str) -> bool:
    """True only when the description is essentially *just* a dimension string.

    Presence of a dimension is not enough: tile and framing sizes ("dated 4 x 4
    tile backsplash", "water stain near the 2 x 4 framing") are real findings,
    and dropping them here is silent because it happens before classification.
    Two guards: what remains after removing the dimension must be short, and
    concrete damage language always wins — overlays say "12' x 10'", never
    "12' x 10' with water damage".
    """
    if not _DIM_RE.search(desc):
        return False
    if _has_high_signal_damage(desc.lower()):
        return False
    residue = " ".join(_DIM_RE.sub(" ", desc).split())
    return (len(residue) <= _DIM_OVERLAY_MAX_RESIDUE_CHARS
            or len(residue.split()) <= _DIM_OVERLAY_MAX_RESIDUE_WORDS)


def partition_dimension_overlays(
    observations: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split observations into (to_classify, overlay_excluded) before the Pass 2c
    LLM call. Room-dimension overlays (e.g. "Primary Bedroom 12'6 x 10'",
    "Living Room 14 x 12") are MLS floorplan text OCR'd into photo notes —
    measurement artefacts, not observations. They go straight to the excluded
    lane with reason "measurement_overlay" and are never sent to the model.
    """
    to_classify: List[Dict[str, str]] = []
    overlay_excluded: List[Dict[str, str]] = []
    for x in observations or []:
        desc = str(x.get("description") or "").strip()
        if not desc:
            continue
        if _is_dimension_overlay(desc):
            logger.debug(f"Pass 2c: Excluding dimension overlay pre-LLM → {desc!r}")
            overlay_excluded.append({"description": desc, "reason": "measurement_overlay"})
        else:
            to_classify.append({"description": desc})
    return to_classify, overlay_excluded


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
        if term_matches(needle, text_lower):
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


# Observation-kind ontology v2: the authoritative enum lives in
# tools/observation_kinds.py; re-imported here so existing importers
# (orchestrator, benchmark, tests) keep working unchanged.
from tools.observation_kinds import (
    ONTOLOGY_VERSION,
    OBSERVATION_KINDS,
    EXCLUSION_REASONS,
)


@dataclass
class Pass2cResult:
    """Result from Pass 2c: classified observations under observation-kind-v2."""
    observations: List[Dict[str, str]] = field(default_factory=list)  # [{"description","kind"}]
    excluded: List[Dict[str, str]] = field(default_factory=list)      # [{"description","reason"}]
    raw_response: Optional[str] = None
    ontology_version: str = ONTOLOGY_VERSION


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
    """Routing decision for catalog retrieval after Pass 2c classification."""
    original_kind: str
    expanded_kinds: Tuple[str, ...]
    reason: str


@dataclass
class Pass2dResult:
    """Result from Pass 2d: Resolved catalog item ID from candidates."""
    observation: str
    resolved_item_id: Optional[str] = None
    resolved_kind: Optional[str] = None  # defect | degradation | modernization
    raw_response: Optional[str] = None
    resolution_path: str = "llm"
    shortcut_reason: Optional[str] = None


class PassExecutionError(RuntimeError):
    """
    An enabled pass could not produce a valid result.

    Exists so a pass failure can never be mistaken for a pass that ran and found
    nothing. Passes raise this instead of returning an empty result; the caller
    decides whether to abort the run (strict) or record it and stop (collect).

    stage:
        dependency - a required input/service was unavailable (e.g. embeddings)
        request    - the provider call itself failed, refused, or was truncated
        response   - a response arrived but carried no usable content
        parse      - content arrived but was not valid for this pass
    """

    def __init__(
        self,
        pass_key: str,
        stage: str,
        message: str,
        *,
        code: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(f"Pass {pass_key} {stage} failure: {message}")
        self.pass_key = pass_key
        self.stage = stage
        self.code = code
        self.message = message
        self.provider = provider
        self.model = model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass": self.pass_key,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "provider": self.provider,
            "model": self.model,
        }


def _pass_failure(
    pass_key: str,
    stage: str,
    exc: BaseException,
    model_config: Optional[dict] = None,
) -> PassExecutionError:
    """Wrap a provider/parse exception as a PassExecutionError with routing context."""
    cfg_ = model_config or {}
    return PassExecutionError(
        pass_key,
        stage,
        str(exc)[:300],
        code=type(exc).__name__,
        provider=cfg_.get("provider"),
        model=cfg_.get("model"),
    )


class Pass2fInvalidResponseError(ValueError):
    """Raised when Pass 2f returns no parseable or actionable JSON decision."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: Optional[str] = None,
        parsed_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.parsed_response = parsed_response


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
    parsed_response: Optional[Dict[str, Any]] = None


def evaluate_kind_routing(description: str, kind: str) -> KindRoutingDecision:
    """Singleton exact-kind route under observation-kind-v2.

    Retrieval searches exactly the observation's kind — no widening, no
    fallback. An invalid kind yields an empty route; callers must fail closed
    before retrieval (the orchestrator raises, and the retrieval filter treats
    an empty/unknown kind set as "no candidates", never "no filter").

    `description` no longer influences routing; the parameter is retained for
    call-site compatibility (blocked harnesses migrate in Task 3).
    """
    normalized_kind = (kind or "").strip().lower()
    if normalized_kind in OBSERVATION_KINDS:
        return KindRoutingDecision(
            original_kind=normalized_kind,
            expanded_kinds=(normalized_kind,),
            reason="exact_kind",
        )
    return KindRoutingDecision(
        original_kind=normalized_kind,
        expanded_kinds=(),
        reason="invalid_kind",
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

# Categories come from the canonical scene table so the prompt, the downstream
# scene→group map, and room-surrogate clustering can never drift apart.
PASS_1A_SYSTEM_PROMPT = (
    "You are a real estate image classifier. Your task is to identify the scene type shown in a property photo.\n"
    "\n"
    "Classify the image into exactly ONE of these categories:\n"
    + "".join(f"- {scene}\n" for scene in PASS_1A_SCENE_IDS)
    + "\n"
    "Respond with ONLY a JSON object:\n"
    "{\n"
    '  "scene": "<category>",\n'
    '  "reasoning": "<brief explanation>"\n'
    "}"
)

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
            **_with_analysis_pass(model_config, "Pass 1a (scene type)"),
        )
    except Exception as e:
        logger.error(f"Pass 1a: Error classifying scene: {e}")
        raise _pass_failure('1a', 'request', e, model_config) from e

    # Parse JSON response
    try:
        result = extract_json_object(response) or {}
    except Exception as e:
        logger.error(f"Pass 1a: Unparseable scene response: {e}")
        raise _pass_failure('1a', 'parse', e, model_config) from e

    return Pass1aResult(
        scene=normalize_scene_id(result.get("scene")),
        reasoning=result.get("reasoning"),
        raw_response=response,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Passes 1b/1c: REMOVED
#
# Their outputs were not consumed downstream. The orchestrator stubs both inline
# (blank Pass1bResult/Pass1cResult, models_used="none") for artifact compat, so
# the LLM-calling implementations were dead code. Pass1bResult/Pass1cResult are
# still defined above because the stubs construct them.
# ═══════════════════════════════════════════════════════════════════════════════


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
            **_with_analysis_pass(model_config, "Pass 2a (observations freeform)"),
        )
    except Exception as e:
        logger.error(f"Pass 2a: Error detecting observations: {e}")
        raise _pass_failure('2a', 'request', e, model_config) from e

    # Do NOT parse JSON. Treat as freeform notes.
    freeform = (response or "").strip()

    logger.debug(f"Pass 2a freeform length: {len(freeform)} chars")

    return Pass2aResult(
        observations_freeform=freeform,
        raw_response=response,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2b: Observations → JSON (Text-only)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2B_SYSTEM_PROMPT_TEMPLATE = """You split FREEFORM photo notes into distinct, atomic observations.

INPUT NOTES:
---
{notes}
---

Rules:
- Only output observations explicitly stated or directly described in the notes.
- One condition claim per observation. When one sentence mixes different claim types about an item, split it into separate observations:
  - damage or failure language (broken, rotted, leaking, missing, unsafe, water-stained, water damage) is its own observation;
  - wear or aging language (worn, faded, stained, scuffed, weathered) is its own observation;
  - dated or style language (dated, old-fashioned, basic, builder-grade) is its own observation.
  Example: "weathered and rotted deck boards" becomes "Deck boards are weathered." and "Deck boards are rotted."
- Details of the same claim type about the same item stay together ("siding is faded and stained" stays one observation).
- Description must be 5–25 words.
- Be factual and non-speculative.
- Do NOT infer causes, consequences, hidden problems, or repair advice.
- If the notes are empty or contain just the word none or none as the last word, return an empty list.

Return JSON only:
{
  "observations": [
    { "description": "..." }
  ]
}"""

PASS_2B_USER_PROMPT = "Convert the notes into the JSON format."

PASS_2B_PROMPT_VERSION = "pass_2b_atomic_v2"
PASS_2B_PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "version": PASS_2B_PROMPT_VERSION,
            "system_template": PASS_2B_SYSTEM_PROMPT_TEMPLATE,
            "user": PASS_2B_USER_PROMPT,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()


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
            **_with_analysis_pass(model_config, "Pass 2b (observations JSON)"),
        )
    except Exception as e:
        logger.error(f"Pass 2b: Error converting observations to JSON: {e}")
        raise _pass_failure('2b', 'request', e, model_config) from e

    try:
        result = extract_json_object(response) or {}
        observations = _coerce_observations_2b(result.get("observations"))
    except Exception as e:
        logger.error(f"Pass 2b: Unparseable observations response: {e}")
        raise _pass_failure('2b', 'parse', e, model_config) from e

    return Pass2bResult(
        observations=observations,
        raw_response=response,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2c: Classify Observations — observation-kind-v2 (Text-only)
# ═══════════════════════════════════════════════════════════════════════════════

PASS_2C_SYSTEM_PROMPT = """Classify each numbered observation about a property photo.

Assign exactly one decision per observation: either a "kind" or an "exclude" reason.

Kinds (the condition of the item):
- defect: expected function, safety, integrity, or protection has FAILED. Failure is stated or visible: broken or missing required parts, active leaks, rot, structural damage, unsafe conditions (exposed wiring, tripping hazards), failed weather protection, rusted-through metal.
- degradation: the item still works and nothing has failed, but it has visibly deteriorated: wear, fading, staining, scuffing, surface rust or corrosion, peeling finish, aging, weathering.
- modernization: the item is functional and acceptably maintained, but dated, basic, low-grade, or an improvement opportunity a renovator might take.

Kind rules:
- Choose defect only when failure is stated or visible. Deterioration without failure is degradation.
- Visible deterioration wins over dated/style language. Purely dated or basic appearance is modernization.
- A required or protective component asserted to be missing or broken is a defect. An absent optional feature is modernization.
- Paving only (driveways, walkways, patios): cracking, surface wear, or minor unevenness (including uneven joints) is degradation; heaving, raised trip edges, or crumbling is a defect.
- Everywhere except paving, cracks are defects: cracked walls, ceilings, tiles, panes, basins, or fixtures.
- Mold or mildew growth is a defect.
- Explicit water stains or water damage (on ceilings, walls, cabinetry, or floors) evidence moisture intrusion: defect. Ordinary dirt or cosmetic staining is degradation.
- An item described only by a low-grade or dated material or grade (laminate, hollow-core, builder-grade, basic) is modernization, not neutral_presence.
- If something is merely "not visible" or "cannot be determined", exclude it as unsupported_or_speculative.

Exclude reasons (text that gets no kind):
- good_condition: says the item looks good, intact, well maintained, clean, or new.
- neutral_presence: neutral existence of an item ("there is a door").
- advice_or_process: advice, process, or verification language ("needs inspection", "recommend evaluation", "cannot determine from photo").
- unsupported_or_speculative: possible or hidden problems with no visible sign; absence inferred only because something is not visible in the photo; hidden systems (structural/foundation, electrical, plumbing, HVAC) mentioned without a specific visible sign (stain, crack, leak, rust, exposed wire, damage).
- measurement_overlay: room-dimension text from floorplan overlays ("Primary Bedroom 12'6 x 10'").
- not_renovation_related: anything else that is not about the renovation-relevant condition of the property.

Response rules:
- One decision per input index. Use every index exactly once.
- Do NOT add, merge, drop, or rewrite observations.
- Each decision has "index" and exactly one of "kind" or "exclude".

Return JSON only:
{
  "decisions": [
    { "index": 1, "kind": "defect|degradation|modernization" },
    { "index": 2, "exclude": "good_condition|neutral_presence|advice_or_process|unsupported_or_speculative|measurement_overlay|not_renovation_related" }
  ]
}
"""

PASS_2C_USER_PROMPT_TEMPLATE = """Scene: {scene}

OBSERVATIONS:
{numbered_observations}
"""

PASS_2C_PROMPT_VERSION = "pass_2c_kind_v2"
PASS_2C_PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "version": PASS_2C_PROMPT_VERSION,
            "system": PASS_2C_SYSTEM_PROMPT,
            "user_template": PASS_2C_USER_PROMPT_TEMPLATE,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()


def _validate_pass_2c_decisions(payload: Any, expected_count: int) -> Dict[int, Dict[str, str]]:
    """
    Validate the indexed Pass 2c response against the observation-kind-v2
    contract. Fail closed: any missing/duplicate index, unknown kind or
    exclusion reason, or malformed row raises ValueError (surfaced as a
    PassExecutionError parse failure). There is deliberately no silent
    coercion — the v1 unknown-label→"other" fallback made prompt/contract
    drift invisible.
    """
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("'decisions' must be a list")
    seen: Dict[int, Dict[str, str]] = {}
    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError(f"decision row is not an object: {row!r}")
        raw_idx = row.get("index")
        if isinstance(raw_idx, bool):
            raise ValueError(f"index is not an integer: {raw_idx!r}")
        if isinstance(raw_idx, int):
            idx = raw_idx
        elif isinstance(raw_idx, str) and raw_idx.strip().isdigit():
            idx = int(raw_idx.strip())
        else:
            raise ValueError(f"index is not an integer: {raw_idx!r}")
        if not (1 <= idx <= expected_count):
            raise ValueError(f"index {idx} outside 1..{expected_count}")
        if idx in seen:
            raise ValueError(f"duplicate index {idx}")
        kind = row.get("kind")
        exclude = row.get("exclude")
        if (kind is None) == (exclude is None):
            raise ValueError(f"index {idx}: exactly one of 'kind' or 'exclude' required")
        if kind is not None:
            k = str(kind).strip().lower()
            if k not in OBSERVATION_KINDS:
                raise ValueError(f"index {idx}: unknown kind {kind!r}")
            seen[idx] = {"kind": k}
        else:
            r = str(exclude).strip().lower()
            if r not in EXCLUSION_REASONS:
                raise ValueError(f"index {idx}: unknown exclusion reason {exclude!r}")
            seen[idx] = {"exclude": r}
    missing = [i for i in range(1, expected_count + 1) if i not in seen]
    if missing:
        raise ValueError(f"missing decisions for indexes {missing}")
    return seen


async def run_pass_2c(
        vlm_client: Any,
        model_config: dict,
        observations: List[Dict[str, str]],
        scene: str = "other",
) -> Pass2cResult:
    """
    Pass 2c: Classify observations under observation-kind-v2.

    Dimension overlays are excluded deterministically before the LLM call.
    The model returns indexed decisions only — it cannot rewrite descriptions.
    Validation is fail-closed: an invalid or incomplete partition raises
    PassExecutionError instead of degrading silently.

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        observations: Observations from Pass 2b
        scene: Scene id for context

    Returns:
        Pass2cResult with classified observations and the excluded lane
    """
    to_classify, overlay_excluded = partition_dimension_overlays(observations)
    if not to_classify:
        return Pass2cResult(observations=[], excluded=overlay_excluded, raw_response=None)

    numbered = "\n".join(
        f"{i}. {obs['description']}" for i, obs in enumerate(to_classify, start=1)
    )
    user_prompt = safe_format_prompt(
        PASS_2C_USER_PROMPT_TEMPLATE, scene=scene, numbered_observations=numbered
    )

    logger.debug("Pass 2c: Classifying observations (text-only)")

    try:
        response = await vlm_client.analyze_text(
            system_prompt=PASS_2C_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **_with_analysis_pass(model_config, "Pass 2c (classify observations)"),
        )
    except Exception as e:
        logger.error(f"Pass 2c: Error classifying observations: {e}")
        raise _pass_failure('2c', 'request', e, model_config) from e

    try:
        payload = extract_json_object(response) or {}
        decisions = _validate_pass_2c_decisions(payload, len(to_classify))
    except Exception as e:
        logger.error(f"Pass 2c: Invalid classification response: {e}")
        raise _pass_failure('2c', 'parse', e, model_config) from e

    classified: List[Dict[str, str]] = []
    excluded: List[Dict[str, str]] = list(overlay_excluded)
    for i, obs in enumerate(to_classify, start=1):
        decision = decisions[i]
        if "kind" in decision:
            classified.append({"description": obs["description"], "kind": decision["kind"]})
        else:
            excluded.append({"description": obs["description"], "reason": decision["exclude"]})

    return Pass2cResult(
        observations=classified,
        excluded=excluded,
        raw_response=response,
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
- All candidates share this kind; the kind is decided upstream and is not yours to change.
- If none fit, return null.
- Use ONLY item_id values from the candidate list.

Return JSON only:
{{
  "resolved_item_id": "..." or null
}}
"""

PASS_2D_PROMPT_VERSION = "pass_2d_exact_kind_v2"
PASS_2D_PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "version": PASS_2D_PROMPT_VERSION,
            "system": PASS_2D_SYSTEM_PROMPT,
            "user_template": PASS_2D_USER_PROMPT_TEMPLATE,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()


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
        support_text = ", ".join(strip_term_marker(str(x).strip()) for x in support_any[:6] if str(x).strip())

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
    support_text = " ".join(strip_term_marker(str(x).strip()) for x in raw_support if str(x).strip())
    signal_text = " ".join(filter(None, [str(candidate.get("name") or "").strip(), support_text]))
    return _extract_component_terms(signal_text), _extract_condition_terms(signal_text)


def _resolved_kind_for_candidate(candidate: Optional[Dict[str, Any]], fallback_kind: str) -> str:
    if not isinstance(candidate, dict):
        return fallback_kind
    candidate_kind = (candidate.get("kind") or "").strip().lower()
    if candidate_kind in OBSERVATION_KINDS:
        return candidate_kind
    return fallback_kind


def _resolve_candidate_via_lexical_shortcut(
    observation: str,
    candidates: List[Dict[str, Any]],
    *,
    kind: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not candidates:
        return None, None, None
    # Never shortcut onto a condition the observation explicitly negates
    # ("no cracks are visible", "intact") — the LLM must weigh those itself.
    _, _, blocked_by_negation = _analyze_visible_condition_signal(observation)
    if blocked_by_negation:
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
    if any(term_matches(phrase, observation_lower) for phrase in support_terms):
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
        kind: str,
) -> Pass2dResult:
    """
    Pass 2d: Resolve a canonical catalog item ID from embedding candidates.

    Args:
        vlm_client: VLM client instance
        model_config: Model configuration
        observation: The observation description string
        candidates: List of candidate dicts from embeddings retrieval
        kind: the observation's kind (defect | degradation | modernization) —
              the pool that was searched; under strict exact-kind retrieval
              every candidate shares it

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

    logger.debug(f"Pass 2d: Resolving catalog item for {kind} observation: {observation[:50]}...")

    try:
        response = await vlm_client.analyze_text(
            system_prompt=PASS_2D_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **_with_analysis_pass(model_config, "Pass 2d (catalog resolution)"),
        )
    except Exception as e:
        logger.error(f"Pass 2d: Error resolving catalog item: {e}")
        raise _pass_failure('2d', 'request', e, model_config) from e

    try:
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
    except Exception as e:
        logger.error(f"Pass 2d: Unparseable resolution response: {e}")
        raise _pass_failure('2d', 'parse', e, model_config) from e

    # Validate resolved_id exists in the candidate list. A hallucinated ID is a
    # model mistake with a defined answer ("no match"), not a pass failure.
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
    return any(term_matches(token, desc_lower) for token in _HIGH_SIGNAL_DAMAGE_TOKENS)


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
    "- The package verdict and the per-issue lists are independent decisions. Even when you "
    "reject the package or mark it uncertain, put every issue ID that IS visibly present in "
    "confirmed_issue_ids, and only the ones you cannot see in rejected_issue_ids.\n"
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

# ── Bedroom Pass 2f prompt ───────────────────────────────────────────────────
# Self-contained: no kitchen / bathroom / exterior vocabulary anywhere in the
# body. Uses the standard output schema (no visible_room_count telemetry).

PASS_2F_BEDROOM_SYSTEM_PROMPT = (
    "You are verifying a proposed real-estate renovation package from bedroom photos. "
    "Use only visible evidence in the supplied images. Your job is visual truth only: "
    "confirm, reject, or mark uncertain whether the proposed package is supported by the photos.\n\n"
    + PASS_2F_SHARED_RULES +
    "\nBedroom-specific guidance:\n"
    "- A modernization or \"outdated finishes\" package is clearly contradicted when "
    "the bedroom shows predominantly updated finishes (newer wood, laminate, or vinyl-plank "
    "flooring or clean modern carpet, fresh neutral paint, a flat or smooth ceiling, modern "
    "light fixture or ceiling fan, intact contemporary trim, doors, and closet). A single "
    "isolated dated detail is not sufficient to confirm a modernization package against an "
    "otherwise updated bedroom.\n"
    "- Strong bedroom support signals: worn, stained, or dated carpet; damaged or heavily "
    "worn hard flooring; popcorn or acoustic ceiling texture; wood paneling or dated "
    "wallpaper; holes, cracks, or damaged drywall; water stains on ceiling or walls; a "
    "dated builder-grade ceiling fan or light fixture; damaged, missing, or dated closet "
    "doors; dated trim or scuffed baseboards.\n"
    "- Common bedroom photo limitations: tight framing and staging furniture or rugs that "
    "obscure floors and walls. Use uncertain when key surfaces are not visible in any of "
    "the supplied images.\n"
)

PASS_2F_BEDROOM_USER_PROMPT = (
    "Analyze these bedroom photos together.\n\n"
    "Proposed package:\n"
    "- package_id: {package_id}\n"
    "- package_type: {package_type}\n"
    "- package_label: {package_label}\n\n"
    "Candidate evidence items:\n"
    "{evidence_json}\n\n"
    + PASS_2F_OUTPUT_SCHEMA
)

# ── Living room Pass 2f prompt ───────────────────────────────────────────────
# Self-contained: no kitchen / bathroom / exterior vocabulary anywhere in the
# body. Uses the standard output schema (no visible_room_count telemetry).

PASS_2F_LIVING_SYSTEM_PROMPT = (
    "You are verifying a proposed real-estate renovation package from living-room photos. "
    "Use only visible evidence in the supplied images. Your job is visual truth only: "
    "confirm, reject, or mark uncertain whether the proposed package is supported by the photos.\n\n"
    + PASS_2F_SHARED_RULES +
    "\nLiving-room-specific guidance:\n"
    "- A modernization or \"outdated finishes\" package is clearly contradicted when "
    "the living area shows predominantly updated finishes (newer hardwood, laminate, or "
    "vinyl-plank flooring or clean modern carpet, fresh neutral paint, a flat or smooth "
    "ceiling, modern light fixtures, intact contemporary trim and built-ins, and an updated "
    "fireplace surround). A single isolated dated detail is not sufficient to confirm a "
    "modernization package against an otherwise updated living area.\n"
    "- Strong living-room support signals: worn, stained, or dated carpet or large expanses "
    "of dated hard flooring; popcorn or acoustic ceiling texture; wood paneling or dated "
    "wallpaper; holes, cracks, or damaged drywall; water stains on ceiling or walls; a "
    "dated builder-grade light fixture or ceiling fan; a dated brick, stone, or tile "
    "fireplace surround; dated built-in shelving or cabinetry; dated trim or scuffed "
    "baseboards.\n"
    "- Common living-area photo limitations: wide open-plan framing and staging furniture "
    "that obscures floors and walls. Use uncertain when key surfaces are not visible in any "
    "of the supplied images.\n"
)

PASS_2F_LIVING_USER_PROMPT = (
    "Analyze these living-room photos together.\n\n"
    "Proposed package:\n"
    "- package_id: {package_id}\n"
    "- package_type: {package_type}\n"
    "- package_label: {package_label}\n\n"
    "Candidate evidence items:\n"
    "{evidence_json}\n\n"
    + PASS_2F_OUTPUT_SCHEMA
)

# ── Exterior Pass 2f prompt ──────────────────────────────────────────────────
# Self-contained: no kitchen / bathroom / bedroom / living vocabulary anywhere
# in the body. Uses the standard output schema (no visible_room_count).
#
# Exterior packages are repair-only, so the body describes envelope damage, not
# curb-appeal modernization. Landscaping, yard, driveway and roof-covering
# findings are deliberately absent: they are not part of this package and
# naming them here would invite exactly the confabulation this pass exists to
# catch.

PASS_2F_EXTERIOR_SYSTEM_PROMPT = (
    "You are verifying a proposed real-estate renovation package from exterior photos. "
    "Use only visible evidence in the supplied images. Your job is visual truth only: "
    "confirm, reject, or mark uncertain whether the proposed package is supported by the photos.\n\n"
    + PASS_2F_SHARED_RULES +
    "\nExterior-specific guidance:\n"
    "- This package covers repair of the building envelope and attached structures: "
    "siding, exterior trim, soffit and fascia, porches, decks, stairs and railings, and "
    "masonry. Judge only those surfaces.\n"
    "- A repair package is clearly contradicted when the visible envelope is sound: "
    "intact siding with no rot, splitting, or missing sections, continuous undamaged trim "
    "soffit and fascia, solid deck and porch boards with secure railings, and masonry "
    "without displaced or crumbling mortar. Recent paint or new siding on an otherwise "
    "sound envelope is a contradiction, not support. A single weathered or discolored "
    "detail is not sufficient to confirm a repair package against an otherwise sound "
    "exterior.\n"
    "- Strong exterior support signals: rotted, split, buckled, or missing siding boards; "
    "exposed sheathing or building wrap; separated or rotted trim; sagging, holed, or "
    "detached soffit or fascia panels; deck or porch boards that are rotted, broken, or "
    "visibly deflecting; loose, missing, or detached railings and stair treads; posts with "
    "rot at the base; cracked, spalling, or displaced masonry with failed mortar joints.\n"
    "- Distinguish damage from soiling. Dirt, algae, chalking, fading, and water staining "
    "are surface conditions; confirm the package on them only when accompanied by visible "
    "material failure.\n"
    "- Common exterior photo limitations: the elevation is shot from a distance so surface "
    "condition cannot be resolved; vegetation, vehicles, or fencing occlude the lower wall "
    "and foundation; strong shadow, glare, or overcast flattening hides texture; wet "
    "surfaces read as staining. Use uncertain when the proposed surfaces are not legible in "
    "any of the supplied images.\n"
)

PASS_2F_EXTERIOR_USER_PROMPT = (
    "Analyze these exterior photos together.\n\n"
    "Proposed package:\n"
    "- package_id: {package_id}\n"
    "- package_type: {package_type}\n"
    "- package_label: {package_label}\n\n"
    "Candidate evidence items:\n"
    "{evidence_json}\n\n"
    + PASS_2F_OUTPUT_SCHEMA
)

# Per-room selector. Future rooms register here without touching run_pass_2f
# internals. NOTE: the living key is "living" (the room constant /
# package["room"]), not the "living_room" scene id; likewise "exterior" is the
# room constant, not the exterior_front/back/side scene ids.
PASS_2F_ROOM_PROMPTS = {
    "kitchen":  (PASS_2F_KITCHEN_SYSTEM_PROMPT,  PASS_2F_KITCHEN_USER_PROMPT),
    "bathroom": (PASS_2F_BATHROOM_SYSTEM_PROMPT, PASS_2F_BATHROOM_USER_PROMPT),
    "bedroom":  (PASS_2F_BEDROOM_SYSTEM_PROMPT,  PASS_2F_BEDROOM_USER_PROMPT),
    "living":   (PASS_2F_LIVING_SYSTEM_PROMPT,   PASS_2F_LIVING_USER_PROMPT),
    "exterior": (PASS_2F_EXTERIOR_SYSTEM_PROMPT, PASS_2F_EXTERIOR_USER_PROMPT),
}

PASS_2F_PROMPT_VERSION = "pass_2f_package_v2"
PASS_2F_PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "version": PASS_2F_PROMPT_VERSION,
            "rooms": {
                room: {"system": prompts[0], "user": prompts[1]}
                for room, prompts in PASS_2F_ROOM_PROMPTS.items()
            },
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()

PASS_2F_VALID_STATUSES = {"confirmed", "rejected", "uncertain"}
PASS_2F_ROOM_COUNT_VALUES = {"one_room", "multiple_rooms", "unclear"}


def _validate_pass_2f_schema(raw: Dict[str, Any], *, room: str) -> None:
    status = raw.get("verification_status")
    if not isinstance(status, str) or status not in PASS_2F_VALID_STATUSES:
        raise ValueError(
            "verification_status must be confirmed, rejected, or uncertain"
        )
    for key in ("confirmed_issue_ids", "rejected_issue_ids"):
        value = raw.get(key)
        if not isinstance(value, list) or any(
            not isinstance(issue_id, str) for issue_id in value
        ):
            raise ValueError(f"{key} must be an array of strings")
    if not isinstance(raw.get("evidence_summary"), str):
        raise ValueError("evidence_summary must be a string")
    if str(room or "").lower() == "bathroom":
        count = raw.get("visible_room_count")
        if (
            not isinstance(count, str)
            or count not in PASS_2F_ROOM_COUNT_VALUES
        ):
            raise ValueError(
                "visible_room_count must be one_room, multiple_rooms, or unclear"
            )
        if not isinstance(raw.get("visible_room_count_evidence"), str):
            raise ValueError("visible_room_count_evidence must be a string")


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
    # An id in both lists is model confusion — explicit rejection wins. But a
    # SYNTHESIZED blanket rejection must not override an explicit confirmation,
    # so the fallback below subtracts what the model actually confirmed.
    explicit_rejected = set(rejected_issue_ids)
    confirmed_issue_ids = [
        issue_id for issue_id in confirmed_issue_ids
        if issue_id not in explicit_rejected
    ]
    if status == "rejected" and not rejected_issue_ids:
        rejected_issue_ids = sorted(valid_issue_ids - set(confirmed_issue_ids))
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

    Always raises on provider or response failure. The former `strict` parameter
    defaulted to False and production never passed it, so a truncated or refused
    response silently became verification_status="uncertain" -- indistinguishable
    from the model genuinely being unsure. Batch-level policy lives in
    rehab_packages.run_pass_2f_batch.
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
                **_with_analysis_pass(model_config, "Pass 2f (package verification)"),
            )
        else:
            if not image_paths:
                raise ValueError("no review images supplied")
            response = await vlm_client.analyze_image(
                image_path=image_paths[0],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **_with_analysis_pass(model_config, "Pass 2f (package verification)"),
            )
    except Exception as exc:
        logger.error("Pass 2f: provider error reviewing %s: %s", package_id, exc)
        raise _pass_failure('2f', 'request', exc, model_config) from exc

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed_value = extract_json_object(response)
        if not isinstance(parsed_value, dict) or not parsed_value:
            raise ValueError("missing JSON object")
        parsed = parsed_value
        # Validation is unconditional. An unvalidated response can silently
        # mis-assign confirmed_issue_ids / rejected_issue_ids.
        _validate_pass_2f_schema(parsed, room=room)
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, Pass2fInvalidResponseError)
            else Pass2fInvalidResponseError(
                f"invalid Pass 2f response: {exc}",
                raw_response=response,
                parsed_response=parsed,
            )
        )
        logger.error("Pass 2f: invalid response for %s: %s", package_id, error)
        if error is exc:
            raise
        raise error from exc

    result = _coerce_pass_2f(
        parsed,
        package_id=package_id,
        package_type=package_type,
        valid_issue_ids=valid_issue_ids,
        room=room,
    )
    result.raw_response = response
    result.parsed_response = dict(parsed)
    return result
