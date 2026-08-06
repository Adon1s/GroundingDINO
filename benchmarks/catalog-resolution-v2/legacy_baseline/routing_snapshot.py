"""FROZEN v1 retrieval semantics — the catalog-resolution legacy baseline.

Verbatim copy of the two-kind routing and candidate-filter behavior as it stood
at commit 029a5f9 (tools/scene_classifier_passes.py, tools/catalog_embeddings.py),
immediately before Task 2 replaced it with strict exact-kind retrieval.

Why a code copy rather than an import: the live constants are gone. This mirrors
benchmarks/kind-ontology-v2/v1_baseline_prompts.json — a baseline you can still
run after the thing it measures has been deleted. DO NOT "fix" anything in this
file; its bugs are the measurement, including:

  * upgrade observations widen to ("upgrade", "defect") on an exterior-only
    component-term list, while defect observations never widen;
  * an unknown kind produces an EMPTY allowed_kinds, which legacy
    retrieve_candidates treats as "no filter" and silently searches the whole
    catalog.

Only retrieval semantics are replayed. Resolution in both lanes uses the current
Pass 2d prompt, so the comparison isolates catalog structure + retrieval rather
than confounding it with a prompt change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tools.pipeline_common import term_matches

LEGACY_CATALOG_KINDS = frozenset({"defect", "upgrade"})


def _cfg_value(name: str, default: Any) -> Any:
    from tools import pipeline_config
    return getattr(pipeline_config, name, default)


@dataclass(frozen=True)
class LegacyKindRoutingDecision:
    original_kind: str
    expanded_kinds: Tuple[str, ...]
    reason: str
    matched_component_terms: Tuple[str, ...] = ()
    matched_condition_terms: Tuple[str, ...] = ()
    blocked_by_negation: bool = False


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
        term for pattern, term in _SPECIFIC_NEGATION_RULES if re.search(pattern, text_lower)
    }
    effective_conditions = tuple(term for term in condition_hits if term not in negated_terms)
    blocked_by_negation = bool(negated_terms) or any(
        re.search(pattern, text_lower) for pattern in _SOFTENING_NEGATION_PATTERNS
    )
    return component_hits, effective_conditions, blocked_by_negation


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


def legacy_evaluate_kind_routing(description: str, kind: str) -> LegacyKindRoutingDecision:
    """Decide whether Pass 2d retrieval should stay single-kind or widen."""
    normalized_kind = (kind or "").strip().lower()
    if normalized_kind != "upgrade":
        expanded = (normalized_kind,) if normalized_kind in {"defect", "upgrade"} else ()
        return LegacyKindRoutingDecision(
            original_kind=normalized_kind,
            expanded_kinds=expanded,
            reason="non_upgrade_kind",
        )

    component_hits, condition_hits, blocked_by_negation = _analyze_visible_condition_signal(description)
    if not component_hits or not condition_hits:
        return LegacyKindRoutingDecision(
            original_kind=normalized_kind,
            expanded_kinds=("upgrade",),
            reason="no_visible_condition_signal",
            matched_component_terms=component_hits,
            matched_condition_terms=condition_hits,
            blocked_by_negation=blocked_by_negation,
        )

    only_vague_conditions = all(term in _VAGUE_ROUTING_CONDITIONS for term in condition_hits)
    if blocked_by_negation and only_vague_conditions:
        return LegacyKindRoutingDecision(
            original_kind=normalized_kind,
            expanded_kinds=("upgrade",),
            reason="blocked_by_negation",
            matched_component_terms=component_hits,
            matched_condition_terms=condition_hits,
            blocked_by_negation=True,
        )

    expanded_kinds = ("upgrade", "defect")
    return LegacyKindRoutingDecision(
        original_kind=normalized_kind,
        expanded_kinds=expanded_kinds,
        reason="visible_condition_signal",
        matched_component_terms=component_hits,
        matched_condition_terms=condition_hits,
        blocked_by_negation=blocked_by_negation,
    )




def legacy_is_generic_resolution_candidate(candidate: Dict[str, Any]) -> bool:
    return bool(candidate.get("drop_if_generic") or candidate.get("defaultHidden"))


def legacy_prioritize_resolution_candidates(
    candidates: List[Dict[str, Any]],
    *,
    widened_routing: bool = False,
) -> List[Dict[str, Any]]:
    ordered = list(candidates or [])
    if not widened_routing:
        return ordered
    return sorted(ordered, key=legacy_is_generic_resolution_candidate)


def legacy_candidate_provider_factory(retriever: Any):
    """The v1 candidate_provider closure (catalog_embeddings.py @ 029a5f9).

    Reproduces the two bugs verbatim: kinds outside {defect, upgrade} are
    filtered out of the requested set, and a resulting EMPTY/None set reaches a
    retriever whose kind filter is `if allowed_kinds:` — i.e. no filter at all.
    """
    from dataclasses import asdict

    def candidate_provider(observation_text: str, context: dict) -> list:
        kind = (context.get("kind") or "").strip().lower()
        topk = context.get("top_k_candidates")
        scene_group = context.get("scene_group")
        allowed_groups = {scene_group} if scene_group else None
        allowed_kinds_ctx = context.get("allowed_kinds")
        if allowed_kinds_ctx:
            allowed_kinds = {
                str(k).strip().lower()
                for k in allowed_kinds_ctx
                if str(k).strip().lower() in LEGACY_CATALOG_KINDS
            }
        else:
            allowed_kinds = {kind} if kind in LEGACY_CATALOG_KINDS else None
        widened_routing = bool(allowed_kinds and len(allowed_kinds) > 1)
        requested_topk = topk
        if widened_routing and topk:
            requested_topk = max(int(topk), int(topk) * 2)
        # v1 semantics: an empty set means "no kind filter" (whole catalog).
        matches = retriever.retrieve_candidates(
            observation_text,
            topk=requested_topk,
            allowed_kinds=allowed_kinds or None,
            allowed_groups=allowed_groups,
        )
        cands = [asdict(m) for m in matches]
        return legacy_prioritize_resolution_candidates(cands, widened_routing=widened_routing)

    return candidate_provider
