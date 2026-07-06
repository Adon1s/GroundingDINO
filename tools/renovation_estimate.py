"""
renovation_estimate.py
----------------------
Primary renovation cost estimation engine.

Sits beside the existing detection, severity, and costing systems.
Powers the renovation estimate pipeline that answers:
  - What are the major cost drivers visible from photos?
  - Which are likely repair vs replace?
  - What are the likely budget buckets?
  - Which items are too uncertain and should be inspection flags?
  - Which package-eligible items must wait for v4 Pass 2f confirmation?

Items are classified into three tiers:
  - high:   major cost drivers (roof, cabinets, foundation, etc.)
  - medium: commonly renovated items (scratched flooring, countertop damage, etc.)
  - minor:  cosmetic/noise items — excluded from quick estimate entirely

Core functions are pure (no LLM, no I/O). The old per-item Pass 2f runner is
retired; v4 owns the package-level Pass 2f visual verification flow.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from tools.catalog_cost_model import (
    COST_MODEL_SOURCE_LEGACY_DEFAULT,
    LINE_ITEM,
    derive_cost_model,
)
from tools.estimate_scope import REQUIRED_REHAB, apply_estimate_scope

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Type aliases
# ═══════════════════════════════════════════════════════════════════════════════

EstimateTier = Literal["high", "medium", "minor"]
EstimateStrategy = Literal[
    "repair_only", "replace_only", "repair_or_replace",
    "service_only", "inspect_only",
]
EstimateGroup = Literal[
    "kitchen", "bathroom", "flooring", "paint_drywall",
    "windows_doors", "roof", "exterior", "structure",
    "remediation", "plumbing", "electrical", "landscaping", "pool", "other",
]
EstimateStackBehavior = Literal["sum", "group_cap", "max_only"]
EstimateUnitPolicy = Literal[
    "per_scope", "per_property", "per_room", "per_opening",
    "per_kitchen", "per_bathroom", "per_system", "per_area",
]
_VALID_UNIT_POLICIES = {
    "per_scope", "per_property", "per_room", "per_opening",
    "per_kitchen", "per_bathroom", "per_system", "per_area",
}
# Public alias for catalog validation (tools/catalog_validation.py).
VALID_UNIT_POLICIES = _VALID_UNIT_POLICIES

# ═══════════════════════════════════════════════════════════════════════════════
# Catalog estimate metadata
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CatalogEstimateMeta:
    """Estimate behaviour attached to a catalog item."""
    estimate_tier: EstimateTier = "minor"
    strategy: EstimateStrategy = "repair_only"
    group: EstimateGroup = "other"
    stack_behavior: EstimateStackBehavior = "sum"
    unit_policy: EstimateUnitPolicy = "per_scope"
    affects_estimate: bool = False
    requires_2f_for_estimate: bool = False


ESTIMATE_DEFAULTS = CatalogEstimateMeta()


def resolve_estimate_meta(raw: Optional[Dict[str, Any]]) -> CatalogEstimateMeta:
    """Parse an estimate block from catalog JSON, falling back to defaults."""
    if not raw or not isinstance(raw, dict):
        return ESTIMATE_DEFAULTS
    tier = raw.get("estimate_tier", "minor")
    if tier not in ("high", "medium", "minor"):
        tier = "minor"
    affects_estimate = raw.get("affects_estimate")
    if affects_estimate is None:
        affects_estimate = tier in ("high", "medium")
    requires_2f = raw.get("requires_2f_for_estimate")
    if requires_2f is None:
        # Backward-compatible default: existing high/medium estimate blocks keep
        # the old behavior unless the catalog opts out per item.
        requires_2f = tier in ("high", "medium")
    unit_policy = raw.get("unit_policy", "per_scope")
    if unit_policy not in _VALID_UNIT_POLICIES:
        unit_policy = "per_scope"
    return CatalogEstimateMeta(
        estimate_tier=tier,
        strategy=raw.get("strategy", "repair_only"),
        group=raw.get("group", "other"),
        stack_behavior=raw.get("stack_behavior", "sum"),
        unit_policy=unit_policy,
        affects_estimate=bool(affects_estimate),
        requires_2f_for_estimate=bool(requires_2f),
    )


def _resolve_catalog_estimate_meta(item: Dict[str, Any]) -> CatalogEstimateMeta:
    """Resolve estimate meta, allowing top-level flags to override the block."""
    raw_estimate = item.get("estimate")
    if isinstance(raw_estimate, dict):
        raw = dict(raw_estimate)
    else:
        raw = {}
    for field_name in ("affects_estimate", "requires_2f_for_estimate"):
        if field_name in item and field_name not in raw:
            raw[field_name] = item[field_name]
    return resolve_estimate_meta(raw or None)


# ═══════════════════════════════════════════════════════════════════════════════
# EstimateCandidate — clean handoff from detection to estimate engine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EstimateCandidate:
    """A catalog item that qualifies for the quick estimate pipeline."""
    catalog_item_id: str
    catalog_item_name: str
    estimate_meta: CatalogEstimateMeta
    kind: str                               # defect | upgrade
    severity: int                           # 1-5
    scope: str                              # repair | replace | cosmetic | service
    trade_bucket: str                       # original trade bucket (for cost lookup)
    cost_obj: Dict[str, Any] = field(default_factory=dict)
    occurrences: int = 1                 # supporting detections, not a pricing multiplier
    estimate_unit_count: int = 1          # scope count used for pricing
    scene_groups_seen: List[str] = field(default_factory=list)
    photo_keys: List[str] = field(default_factory=list)
    issue_ids: List[str] = field(default_factory=list)
    estimate_unit_id: str = ""
    billable_estimate_unit_id: str = ""
    estimate_scope_key: str = ""
    resolved_cluster_key: str = ""
    room_surrogate_id: str = ""
    source_room_surrogate_ids: List[str] = field(default_factory=list)
    estimate_scope: str = REQUIRED_REHAB
    estimate_scope_reason: str = "scope_default"
    baseline_scope_before_posture: str = REQUIRED_REHAB
    visible_required_with_inspect_posture: bool = False
    required_baseline_included: bool = True
    inspection_risk_added: bool = False
    cost_model: str = LINE_ITEM
    cost_model_source: str = COST_MODEL_SOURCE_LEGACY_DEFAULT
    supporting_observations: List[str] = field(default_factory=list)
    distinct_photo_count: int = 0
    distinct_scene_group_count: int = 0
    representative_issue_id: Optional[str] = None
    estimate_unit_label: str = ""
    unit_resolution_method: str = ""
    unit_resolution_confidence: str = ""
    unit_resolution_notes: List[str] = field(default_factory=list)
    unit_members: List[Dict[str, Any]] = field(default_factory=list)
    source_estimate_unit_ids: List[str] = field(default_factory=list)
    source_issue_ids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    package_evidence_only: bool = False
    # ── Pass 2f review override fields (populated by revisit pass) ────────
    is_valid_detection: Optional[bool] = None  # None = 2f didn't run; False = model deemed invalid
    review_posture: Optional[str] = None    # e.g. "repair", "replace", "keep_default"
    review_visible_scope: Optional[str] = None  # localized | partial | room_wide | unknown
    review_rationale: Optional[str] = None  # debug-only LLM explanation
    review_consistency_flags: List[str] = field(default_factory=list)  # debug-only Pass 2f checks
    review_source: Optional[str] = None     # "pass_2f" | None (indicates origin)
    # ── Resolved posture (set by resolve_effective_posture) ────────────
    effective_posture: Optional[str] = None # authoritative posture for pricing
    review_image_path: Optional[str] = None # path of image used for 2f review
    # ── Pass 2f fallback tracking ────────────────────────────────────────
    pass_2f_attempted: bool = False          # True if VLM call was made
    pass_2f_applied: bool = False            # True if authoritative posture was set
    pass_2f_fallback_reason: Optional[str] = None  # None when applied, else reason
    # Package-level visual verification fields (v4 package-first path)
    package_id: Optional[str] = None
    package_type: Optional[str] = None
    package_role: Optional[str] = None
    visual_verification_status: Optional[str] = None
    package_verification_source: Optional[str] = None


# Valid postures that 2f can authoritatively set
_AUTHORITATIVE_POSTURES = {"repair", "replace", "inspect"}

# ── Pass 2f fallback reason constants ──
PASS_2F_FALLBACK_NOT_ELIGIBLE = "not_eligible"
PASS_2F_FALLBACK_NO_IMAGE     = "no_valid_image"
PASS_2F_FALLBACK_VLM_ERROR    = "vlm_error"
PASS_2F_FALLBACK_INVALID_RESPONSE = "invalid_response"
PASS_2F_FALLBACK_KEEP_DEFAULT = "keep_default"
PASS_2F_FALLBACK_PACKAGE_VERIFICATION = "package_verification_required"
PASS_2F_FALLBACK_RETIRED = "legacy_pass_2f_retired"


def resolve_effective_posture(candidate: EstimateCandidate) -> str:
    """
    Determine the effective pricing posture for a candidate.

    If Pass 2f returned a valid authoritative posture (repair/replace),
    use it directly.  Otherwise fall back to the catalog's strategy field.
    "keep_default" intentionally falls through to catalog default.
    """
    if (candidate.review_posture is not None
            and candidate.review_posture in _AUTHORITATIVE_POSTURES):
        return candidate.review_posture
    return candidate.estimate_meta.strategy


def _dedupe_consistency_flags(flags: List[Any]) -> List[str]:
    return sorted({
        str(flag).strip()
        for flag in (flags or [])
        if str(flag or "").strip()
    })


def _resolve_pass_2f_consistency_flags(
    candidate: EstimateCandidate,
    result: Any,
) -> List[str]:
    flags: List[str] = list(getattr(result, "consistency_flags", []) or [])
    visible_scope = getattr(result, "visible_scope", None)
    pricing_posture = getattr(result, "pricing_posture", None)

    if visible_scope == "room_wide" and pricing_posture == "repair":
        flags.append("room_wide_repair")
    if (
        visible_scope == "localized"
        and pricing_posture == "replace"
        and candidate.estimate_meta.strategy != "replace_only"
    ):
        flags.append("localized_replace_non_replacement_only")
    if visible_scope == "unknown" and pricing_posture == "replace":
        flags.append("unknown_replace")

    return _dedupe_consistency_flags(flags)


def _consistency_flag_counts(candidates: List[EstimateCandidate]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for candidate in candidates:
        for flag in candidate.review_consistency_flags or []:
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


_TIER_RANK = {"high": 0, "medium": 1, "minor": 2}


_GENERIC_SCOPE_HINTS = {
    "", "unknown", "other", "n/a", "na", "none", "room", "area", "space",
    "interior", "exterior", "floor", "flooring", "wall", "walls", "ceiling",
    "visible", "general",
}


def _clean_scope_component(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_/-]+", "", text)
    text = text.strip("_-/")
    return text


def _meaningful_scope_hint(value: Any) -> str:
    hint = _clean_scope_component(value)
    if hint in _GENERIC_SCOPE_HINTS:
        return ""
    return hint


def _estimate_scope_key_for_issue(
    catalog_item_id: str,
    issue: Dict[str, Any],
) -> Tuple[str, str]:
    """Build a scope key where photos are evidence, not multipliers."""
    scene_group = _meaningful_scope_hint(issue.get("scene_group")) or "other"

    room_surrogate = ""
    for field_name in (
        "room_surrogate_id", "room_id", "room_key", "area_id",
        "location_hint", "scope_hint",
    ):
        room_surrogate = _meaningful_scope_hint(issue.get(field_name))
        if room_surrogate:
            break

    if not room_surrogate:
        room_surrogate = (
            _meaningful_scope_hint(issue.get("scene"))
            or scene_group
            or "property"
        )

    scope_key = (
        f"catalog:{catalog_item_id}|"
        f"scene_group:{scene_group}|"
        f"room:{room_surrogate}"
    )
    return scope_key, room_surrogate


def issue_evidence_ref(issue: Dict[str, Any]) -> Dict[str, Any]:
    """Return the per-issue reference used by package-level verification."""
    return {
        "issue_id": str(issue.get("issue_id") or ""),
        "photo_key": str(issue.get("photo_key") or ""),
        "observation": str(issue.get("description") or ""),
        "room_surrogate_id": str(issue.get("room_surrogate_id") or ""),
    }


def issue_evidence_refs(occurrences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    seen = set()
    for issue in occurrences or []:
        ref = issue_evidence_ref(issue)
        key = (ref["issue_id"], ref["photo_key"])
        if not ref["issue_id"] or key in seen:
            continue
        seen.add(key)
        refs.append(ref)
    return refs


def extract_estimate_candidates(
    issues_flat: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
) -> List[EstimateCandidate]:
    """
    Filter issues_flat into estimate-ready candidates.

    Only returns items where estimate_tier is 'high' or 'medium'.
    Minor-tier items are excluded from the quick estimate entirely.
    """
    catalog_lookup: Dict[str, Dict[str, Any]] = {}
    for item in issue_catalog.get("items", []):
        if isinstance(item, dict) and item.get("id"):
            catalog_lookup[item["id"]] = item

    # Group issues into estimate units. Photos are evidence inside a unit, not
    # separate cost multipliers.
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for issue in (issues_flat or []):
        cat_id = issue.get("catalog_item_id")
        if not cat_id or cat_id not in catalog_lookup:
            continue
        est_meta = _resolve_catalog_estimate_meta(catalog_lookup[cat_id])
        if not est_meta.affects_estimate:
            continue
        scope_key, room_surrogate = _estimate_scope_key_for_issue(cat_id, issue)
        grouped.setdefault((cat_id, scope_key, room_surrogate), []).append(issue)

    candidates: List[EstimateCandidate] = []
    for (cat_id, scope_key, room_surrogate), occurrences in grouped.items():
        cat = catalog_lookup[cat_id]
        est_meta = _resolve_catalog_estimate_meta(cat)

        if not est_meta.affects_estimate:
            continue

        scene_groups = sorted(set(
            issue.get("scene_group", "other") or "other" for issue in occurrences
        ))
        photo_keys = sorted(set(
            issue.get("photo_key", "") for issue in occurrences
            if issue.get("photo_key")
        ))
        issue_ids = [
            issue.get("issue_id", "") for issue in occurrences
            if issue.get("issue_id")
        ]
        billable_estimate_unit_ids = sorted(set(
            _meaningful_scope_hint(issue.get("estimate_unit_id"))
            for issue in occurrences
            if _meaningful_scope_hint(issue.get("estimate_unit_id"))
        ))
        source_room_surrogate_ids = sorted(set(
            _meaningful_scope_hint(issue.get("room_surrogate_id"))
            for issue in occurrences
            if _meaningful_scope_hint(issue.get("room_surrogate_id"))
        ))
        observations = [
            issue.get("description", "") for issue in occurrences
            if issue.get("description")
        ]
        evidence_refs = issue_evidence_refs(occurrences)
        unit_id = f"{cat_id}:{_clean_scope_component(scope_key)}"

        candidate = EstimateCandidate(
            catalog_item_id=cat_id,
            catalog_item_name=cat.get("name", cat_id),
            estimate_meta=est_meta,
            kind=cat.get("kind", "defect"),
            severity=cat.get("severity", 2),
            scope=cat.get("scope", "repair"),
            trade_bucket=cat.get("trade_bucket", "safety_general"),
            cost_obj=cat.get("cost") or {},
            occurrences=len(occurrences),
            estimate_unit_count=1,
            scene_groups_seen=scene_groups,
            photo_keys=photo_keys,
            issue_ids=issue_ids,
            estimate_unit_id=unit_id,
            billable_estimate_unit_id=(
                billable_estimate_unit_ids[0]
                if len(billable_estimate_unit_ids) == 1
                else ("multiple_estimate_units" if billable_estimate_unit_ids else "")
            ),
            estimate_scope_key=scope_key,
            room_surrogate_id=room_surrogate,
            source_room_surrogate_ids=source_room_surrogate_ids,
            supporting_observations=observations,
            evidence_refs=evidence_refs,
            distinct_photo_count=len(photo_keys),
            distinct_scene_group_count=len(scene_groups),
            representative_issue_id=issue_ids[0] if issue_ids else None,
        )
        apply_estimate_scope(candidate, cat)
        candidate.cost_model, candidate.cost_model_source = derive_cost_model(
            cat,
            candidate,
        )
        candidates.append(candidate)

    # Sort: high tier first, then medium, then severity desc, evidence desc.
    candidates.sort(key=lambda c: (
        _TIER_RANK.get(c.estimate_meta.estimate_tier, 2),
        -c.severity,
        -c.occurrences,
        c.estimate_scope_key,
    ))

    return candidates


def resolve_estimate_units(
    candidates: List[EstimateCandidate],
    issues_flat: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
) -> List[EstimateCandidate]:
    """Resolve extracted evidence candidates into conservative pricing units."""
    from tools.estimate_units import resolve_estimate_units as _resolve_estimate_units

    return _resolve_estimate_units(candidates, issues_flat, issue_catalog)


# ═══════════════════════════════════════════════════════════════════════════════
# Representative image selection for Pass 2f
# ═══════════════════════════════════════════════════════════════════════════════

def _select_representative_image(
    candidate: EstimateCandidate,
    issues_flat: List[Dict[str, Any]],
    photo_key_to_path: Dict[str, Path],
) -> Optional[Path]:
    """
    Select the best representative image for Pass 2f review.

    Deterministic priority:
      1. photo_key with a direct issue match for this catalog_item_id
         AND matching scene_group
      2. photo_key with a direct issue match (any scene)
      3. photo_key whose scene matches the candidate's primary scene_group
      4. First valid photo_key (stable sort on key name)

    Returns Path or None if no valid image found.
    """
    # Build set of photo_keys that have a direct issue for this candidate
    direct_issue_keys: set = set()
    key_scenes: Dict[str, set] = {}   # photo_key → set of scene_groups
    for issue in (issues_flat or []):
        pk = issue.get("photo_key")
        sg = issue.get("scene_group")
        if pk and sg:
            key_scenes.setdefault(pk, set()).add(sg)
        if issue.get("catalog_item_id") == candidate.catalog_item_id and pk:
            direct_issue_keys.add(pk)

    primary_scene = candidate.scene_groups_seen[0] if candidate.scene_groups_seen else None

    def _rank(pk: str) -> Tuple[int, str]:
        has_issue = pk in direct_issue_keys
        scene_match = primary_scene and primary_scene in key_scenes.get(pk, set())
        if has_issue and scene_match:
            return (0, pk)
        if has_issue:
            return (1, pk)
        if scene_match:
            return (2, pk)
        return (3, pk)

    valid_keys = sorted(
        (pk for pk in candidate.photo_keys if pk in photo_key_to_path),
        key=_rank,
    )
    for pk in valid_keys:
        path = photo_key_to_path[pk]
        if path.exists():
            return path
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2f provider resolution
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_pass_2f_model_config(
    *,
    provider: str,
    local_config: Optional[dict],
    premium_config: Optional[dict],
) -> Optional[dict]:
    """Select model config for Pass 2f based on provider setting."""
    if provider == "premium":
        return premium_config
    if provider == "local":
        return local_config
    return premium_config  # default to premium


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2f batch runner — optional LLM revisit for eligible candidates
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_PASS_2F_TIERS = frozenset({"high", "medium"})


async def run_pass_2f_batch(
    candidates: List[EstimateCandidate],
    issues_flat: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
    vlm_client: Any,
    model_config: dict,
    photo_key_to_path: Dict[str, Path],
    provider: str = "premium",
    pass_2f_tiers: Optional[frozenset] = None,
) -> List[EstimateCandidate]:
    """
    Deprecated per-item Pass 2f runner.

    Pass 2f is now package-level visual verification in v4. This function is
    retained as a no-op compatibility shim so older call sites cannot trigger
    the retired per-item pricing/posture API calls.
    """
    del issues_flat, vlm_client, model_config, photo_key_to_path, provider, pass_2f_tiers

    cat_lookup: Dict[str, Dict[str, Any]] = {}
    for item in issue_catalog.get("items", []):
        if isinstance(item, dict) and item.get("id"):
            cat_lookup[item["id"]] = item

    for c in candidates or []:
        c.pass_2f_attempted = False
        c.pass_2f_applied = False
        c.pass_2f_fallback_reason = PASS_2F_FALLBACK_RETIRED
        if c.effective_posture is None:
            c.effective_posture = resolve_effective_posture(c)
        apply_estimate_scope(c, cat_lookup.get(c.catalog_item_id, {}))
        c.cost_model, c.cost_model_source = derive_cost_model(
            cat_lookup.get(c.catalog_item_id, {}),
            c,
        )

    logger.info("Legacy per-item Pass 2f is retired; skipped %d candidates", len(candidates or []))
    return candidates


# =============================================================================
# Group-based estimate engine
# =============================================================================

# Group-level budget caps — plausible total for the ENTIRE group regardless of
# how many individual items fire.  Prevents naive overcounting.
GROUP_BUDGET_CAPS: Dict[str, Tuple[int, int]] = {
    "kitchen":       (3_000,  35_000),
    "bathroom":      (2_000,  25_000),
    "flooring":      (2_000,  20_000),
    "roof":          (1_000,  25_000),
    "structure":     (1_000,  50_000),
    "remediation":   (500,    15_000),
    "windows_doors": (500,    15_000),
    "exterior":      (1_000,  25_000),
    "plumbing":      (500,    12_000),
    "electrical":    (500,    15_000),
    "paint_drywall": (800,    10_000),
    "landscaping":   (500,    15_000),
    "pool":          (2_000,  40_000),
    "other":         (200,    5_000),
}

# Inspection allowance for inspect_only items — avoids fake precision
INSPECT_ALLOWANCE: Tuple[int, int] = (200, 800)

# Posture → scope override mapping for pricing.
# Maps effective_posture values to the scope parameter for compute_item_cost_range.
# None means ambiguous/inspect — handled specially.
POSTURE_TO_SCOPE: Dict[str, Optional[str]] = {
    "repair":             "repair",       # SCOPE_MULT 1.0
    "replace":            "replace",      # SCOPE_MULT 1.3
    "inspect":            None,           # → INSPECT_ALLOWANCE
    # Strategy values that appear as effective_posture when 2f didn't run
    "repair_only":        "repair",
    "replace_only":       "replace",
    "repair_or_replace":  None,           # ambiguous → use catalog scope
    "service_only":       "service",
    "inspect_only":       None,           # → INSPECT_ALLOWANCE
}

# Postures that resolve to INSPECT_ALLOWANCE rather than costed pricing
_INSPECT_POSTURES = {"inspect", "inspect_only"}


def _is_manual_allowance_cost(cost_obj: Dict[str, Any]) -> bool:
    return (
        isinstance(cost_obj, dict)
        and cost_obj.get("mode") == "allowance"
        and cost_obj.get("cost_source") == "manual"
    )


def _pricing_cost_basis(candidate: EstimateCandidate, effective_posture: str) -> str:
    if candidate.is_valid_detection is False:
        return "invalidated"
    if effective_posture in _INSPECT_POSTURES:
        return "inspection_allowance"
    if _is_manual_allowance_cost(candidate.cost_obj):
        return "manual_allowance"
    if not candidate.cost_obj or candidate.cost_obj.get("mode") == "heuristic":
        return "heuristic"
    return str(candidate.cost_obj.get("mode") or "allowance")


def resolve_pricing_band(
    candidate: EstimateCandidate,
    effective_posture: str,
) -> Tuple[int, int]:
    """
    Compute cost range based on the resolved effective posture.

    This is the single point where posture decisions become prices:
      - "repair"      → compute_item_cost_range with scope="repair"  (SCOPE_MULT 1.0)
      - "replace"     → compute_item_cost_range with scope="replace" (SCOPE_MULT 1.3)
      - "inspect"     → INSPECT_ALLOWANCE
      - strategy vals → mapped via POSTURE_TO_SCOPE
      - unknown       → fall back to candidate.scope (catalog default)
    """
    from tools.costing import compute_item_cost_range

    # Inspect-type postures → flat allowance
    if effective_posture in _INSPECT_POSTURES:
        return INSPECT_ALLOWANCE

    # Map posture to costing scope
    scope_override = POSTURE_TO_SCOPE.get(effective_posture)
    costing_scope = scope_override if scope_override is not None else candidate.scope

    return compute_item_cost_range(
        cost_obj=candidate.cost_obj,
        n_occurrences=max(1, candidate.estimate_unit_count),
        kind=candidate.kind,
        scope=costing_scope,
        trade_bucket=candidate.trade_bucket,
        severity=candidate.severity,
    )


def resolve_risk_exposure_band(
    candidate: EstimateCandidate,
    effective_posture: str,
) -> Tuple[int, int]:
    """
    Estimate latent exposure for inspect-only items without adding it to probable cost.
    """
    if effective_posture not in _INSPECT_POSTURES or candidate.is_valid_detection is False:
        return 0, 0

    from tools.costing import compute_item_cost_range

    return compute_item_cost_range(
        cost_obj=candidate.cost_obj,
        n_occurrences=max(1, candidate.estimate_unit_count),
        kind=candidate.kind,
        scope=candidate.scope,
        trade_bucket=candidate.trade_bucket,
        severity=candidate.severity,
    )


def _resolve_dominant_stack(candidates: List[EstimateCandidate]) -> str:
    """Determine the dominant stack behavior: max_only > group_cap > sum."""
    behaviors = set(c.estimate_meta.stack_behavior for c in candidates)
    if "max_only" in behaviors:
        return "max_only"
    if "group_cap" in behaviors:
        return "group_cap"
    return "sum"


def compute_group_estimate(
    group: str,
    candidates: List[EstimateCandidate],
) -> Dict[str, Any]:
    """
    Compute the estimate for a single estimate group.

    Stack behaviors:
      - sum:       add all item costs together
      - group_cap: sum items but cap at GROUP_BUDGET_CAPS; floor at highest item
      - max_only:  take only the single highest-cost item
    """
    if not candidates:
        return {
            "group": group,
            "low": 0, "high": 0,
            "raw_sum_low": 0, "raw_sum_high": 0,
            "risk_exposure_low": 0, "risk_exposure_high": 0,
            "inspection_allowance_low": 0, "inspection_allowance_high": 0,
            "has_high_tier": False,
            "stack_behavior": "sum",
            "inspection_only": False,
            "item_count": 0,
            "line_items": [],
        }

    # Compute per-candidate costs
    line_items: List[Dict[str, Any]] = []
    for c in candidates:
        effective = c.effective_posture or c.estimate_meta.strategy
        low, high = resolve_pricing_band(c, effective)
        risk_low, risk_high = resolve_risk_exposure_band(c, effective)
        cost_basis = _pricing_cost_basis(c, effective)
        # Zero out cost for detections the model deemed invalid
        if c.is_valid_detection is False:
            low, high = 0, 0
            risk_low, risk_high = 0, 0
        li: Dict[str, Any] = {
            "estimate_unit_id": c.estimate_unit_id,
            "billable_estimate_unit_id": c.billable_estimate_unit_id,
            "estimate_scope_key": c.estimate_scope_key,
            "resolved_cluster_key": c.resolved_cluster_key,
            "room_surrogate_id": c.room_surrogate_id,
            "source_room_surrogate_ids": c.source_room_surrogate_ids,
            "catalog_item_id": c.catalog_item_id,
            "name": c.catalog_item_name,
            "trade_bucket": c.trade_bucket,
            "estimate_scope": c.estimate_scope,
            "estimate_scope_reason": c.estimate_scope_reason,
            "baseline_scope_before_posture": c.baseline_scope_before_posture,
            "visible_required_with_inspect_posture": c.visible_required_with_inspect_posture,
            "required_baseline_included": c.required_baseline_included,
            "inspection_risk_added": c.inspection_risk_added,
            "cost_model": c.cost_model,
            "cost_model_source": c.cost_model_source,
            "occurrences": c.occurrences,
            "estimate_unit_count": c.estimate_unit_count,
            "unit_policy": c.estimate_meta.unit_policy,
            "estimate_unit_label": c.estimate_unit_label,
            "unit_resolution_method": c.unit_resolution_method,
            "unit_resolution_confidence": c.unit_resolution_confidence,
            "unit_resolution_notes": c.unit_resolution_notes,
            "unit_members": c.unit_members,
            "source_estimate_unit_ids": c.source_estimate_unit_ids,
            "source_issue_ids": c.source_issue_ids,
            "supporting_photo_count": c.distinct_photo_count or len(c.photo_keys),
            "supporting_scene_group_count": c.distinct_scene_group_count or len(c.scene_groups_seen),
            "supporting_observations": c.supporting_observations,
            "strategy": c.estimate_meta.strategy,
            "stack_behavior": c.estimate_meta.stack_behavior,
            "estimate_tier": c.estimate_meta.estimate_tier,
            "requires_2f_for_estimate": c.estimate_meta.requires_2f_for_estimate,
            "cost_basis": cost_basis,
            "cost_low": low,
            "cost_high": high,
            "risk_exposure_low": risk_low,
            "risk_exposure_high": risk_high,
            # ── Validity & posture audit trail (always present) ──
            "is_valid_detection": c.is_valid_detection,
            "default_posture": c.estimate_meta.strategy,
            "review_posture": c.review_posture,
            "pricing_posture": c.review_posture,
            "effective_posture": effective,
            "review_source": c.review_source,
            "review_image_path": c.review_image_path,
            # ── Pass 2f fallback tracking ──
            "pass_2f_attempted": c.pass_2f_attempted,
            "pass_2f_applied": c.pass_2f_applied,
            "pass_2f_fallback_reason": c.pass_2f_fallback_reason,
            "package_id": c.package_id,
            "package_type": c.package_type,
            "package_role": c.package_role,
            "visual_verification_status": c.visual_verification_status,
            "package_verification_source": c.package_verification_source,
        }
        line_items.append(li)

    dominant = _resolve_dominant_stack(candidates)
    raw_sum_low = sum(li["cost_low"] for li in line_items)
    raw_sum_high = sum(li["cost_high"] for li in line_items)
    raw_risk_exposure_low = sum(li["risk_exposure_low"] for li in line_items)
    raw_risk_exposure_high = sum(li["risk_exposure_high"] for li in line_items)
    inspection_allowance_low = sum(
        li["cost_low"] for li in line_items
        if li.get("effective_posture") in _INSPECT_POSTURES
    )
    inspection_allowance_high = sum(
        li["cost_high"] for li in line_items
        if li.get("effective_posture") in _INSPECT_POSTURES
    )
    has_high_tier = any(c.estimate_meta.estimate_tier == "high" for c in candidates)
    all_inspect = all(li.get("effective_posture") in _INSPECT_POSTURES for li in line_items)

    # Apply stack behavior
    if dominant == "max_only":
        best = max(line_items, key=lambda li: li["cost_high"])
        capped_low = best["cost_low"]
        capped_high = best["cost_high"]
    elif dominant == "group_cap":
        cap_low, cap_high = GROUP_BUDGET_CAPS.get(group, (200, 5_000))
        capped_low = min(raw_sum_low, cap_low)
        capped_high = min(raw_sum_high, cap_high)
        # Floor: never go below the single highest item
        best = max(line_items, key=lambda li: li["cost_high"])
        capped_low = max(capped_low, best["cost_low"])
        capped_high = max(capped_high, best["cost_high"])
    else:  # "sum"
        capped_low = raw_sum_low
        capped_high = raw_sum_high

    return {
        "group": group,
        "low": capped_low,
        "high": capped_high,
        "raw_sum_low": raw_sum_low,
        "raw_sum_high": raw_sum_high,
        "risk_exposure_low": raw_risk_exposure_low,
        "risk_exposure_high": raw_risk_exposure_high,
        "inspection_allowance_low": inspection_allowance_low,
        "inspection_allowance_high": inspection_allowance_high,
        "has_high_tier": has_high_tier,
        "stack_behavior": dominant,
        "inspection_only": all_inspect,
        "item_count": len(candidates),
        "line_items": line_items,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tier summary — elevates Pass 2f-validated high/medium items
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tier_summary(
    candidates: List[EstimateCandidate],
    groups_out: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Partition high+medium tier items by Pass 2f validation status.

    Only items that were explicitly validated by Pass 2f (pass_2f_attempted=True
    AND is_valid_detection=True) contribute to validated_total.  Unreviewed and
    invalidated items are listed separately for transparency.

    Costs are pulled from the already-computed group line items so that group
    caps and stack behaviors are respected — no re-computation.
    """
    reviewable = [c for c in candidates if c.estimate_meta.estimate_tier in ("high", "medium")]

    if not reviewable:
        return {
            "version": "tier_summary_v1",
            "validated_total": {"low": 0, "high": 0},
            "validated_items": [],
            "invalidated_items": [],
            "unreviewed_items": [],
            "confidence": "low",
            "meta": {
                "total_reviewable": 0,
                "validated_count": 0,
                "invalidated_count": 0,
                "unreviewed_count": 0,
            },
        }

    # Build a lookup from group line_items for cost retrieval
    # key: catalog_item_id → line_item dict (with cost_low / cost_high)
    line_item_lookup: Dict[str, Dict[str, Any]] = {}
    for g in groups_out:
        for li in g.get("line_items", []):
            line_item_lookup[li.get("estimate_unit_id") or li["catalog_item_id"]] = li

    def _item_record(c: EstimateCandidate) -> Dict[str, Any]:
        li = line_item_lookup.get(c.estimate_unit_id or c.catalog_item_id, {})
        return {
            "estimate_unit_id": c.estimate_unit_id,
            "billable_estimate_unit_id": c.billable_estimate_unit_id,
            "estimate_scope_key": c.estimate_scope_key,
            "resolved_cluster_key": c.resolved_cluster_key,
            "room_surrogate_id": c.room_surrogate_id,
            "source_room_surrogate_ids": c.source_room_surrogate_ids,
            "catalog_item_id": c.catalog_item_id,
            "name": c.catalog_item_name,
            "estimate_scope": c.estimate_scope,
            "estimate_scope_reason": c.estimate_scope_reason,
            "baseline_scope_before_posture": c.baseline_scope_before_posture,
            "visible_required_with_inspect_posture": c.visible_required_with_inspect_posture,
            "required_baseline_included": c.required_baseline_included,
            "inspection_risk_added": c.inspection_risk_added,
            "cost_model": c.cost_model,
            "cost_model_source": c.cost_model_source,
            "estimate_tier": c.estimate_meta.estimate_tier,
            "group": c.estimate_meta.group,
            "occurrences": c.occurrences,
            "estimate_unit_count": c.estimate_unit_count,
            "unit_policy": c.estimate_meta.unit_policy,
            "estimate_unit_label": c.estimate_unit_label,
            "unit_resolution_method": c.unit_resolution_method,
            "unit_resolution_confidence": c.unit_resolution_confidence,
            "unit_resolution_notes": c.unit_resolution_notes,
            "unit_members": c.unit_members,
            "source_estimate_unit_ids": c.source_estimate_unit_ids,
            "source_issue_ids": c.source_issue_ids,
            "cost_basis": li.get("cost_basis"),
            "cost_low": li.get("cost_low", 0),
            "cost_high": li.get("cost_high", 0),
            "risk_exposure_low": li.get("risk_exposure_low", 0),
            "risk_exposure_high": li.get("risk_exposure_high", 0),
            "effective_posture": c.effective_posture or c.estimate_meta.strategy,
            "review_posture": c.review_posture,
            "pricing_posture": c.review_posture,
            "review_source": c.review_source,
            "pass_2f_attempted": c.pass_2f_attempted,
            "pass_2f_applied": c.pass_2f_applied,
            "pass_2f_fallback_reason": c.pass_2f_fallback_reason,
            "is_valid_detection": c.is_valid_detection,
        }

    validated: List[Dict[str, Any]] = []
    invalidated: List[Dict[str, Any]] = []
    unreviewed: List[Dict[str, Any]] = []

    for c in reviewable:
        rec = _item_record(c)
        if c.is_valid_detection is False:
            invalidated.append(rec)
        elif c.pass_2f_applied and c.is_valid_detection is True:
            validated.append(rec)
        else:
            unreviewed.append(rec)

    validated_low = sum(r["cost_low"] for r in validated)
    validated_high = sum(r["cost_high"] for r in validated)

    # Confidence based on review coverage
    total = len(reviewable)
    n_validated = len(validated)
    n_invalidated = len(invalidated)
    n_reviewed = n_validated + n_invalidated  # 2f ran on these

    if n_reviewed == total and n_validated > 0:
        confidence = "high"
    elif n_reviewed > 0:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "version": "tier_summary_v1",
        "validated_total": {"low": validated_low, "high": validated_high},
        "validated_items": validated,
        "invalidated_items": invalidated,
        "unreviewed_items": unreviewed,
        "confidence": confidence,
        "meta": {
            "total_reviewable": total,
            "validated_count": n_validated,
            "invalidated_count": n_invalidated,
            "unreviewed_count": len(unreviewed),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def compute_renovation_estimate(
    issues_flat: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
    *,
    prebuilt_candidates: Optional[List[EstimateCandidate]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for the renovation estimate pipeline.

    Returns a dict ready for JSON serialisation into photo_intel["renovation_estimate"].

    If prebuilt_candidates is provided (e.g. already processed by Pass 2f),
    those are used directly instead of re-extracting from issues_flat.
    """
    candidates = prebuilt_candidates or extract_estimate_candidates(
        issues_flat, issue_catalog,
    )
    candidates = resolve_estimate_units(candidates, issues_flat, issue_catalog)
    catalog_lookup = {
        item["id"]: item
        for item in (issue_catalog.get("items") or [])
        if isinstance(item, dict) and item.get("id")
    }
    for candidate in candidates:
        apply_estimate_scope(
            candidate,
            catalog_lookup.get(candidate.catalog_item_id, {}),
        )
        candidate.cost_model, candidate.cost_model_source = derive_cost_model(
            catalog_lookup.get(candidate.catalog_item_id, {}),
            candidate,
        )

    if not candidates:
        return {
            "version": "renovation_estimate_v3",
            "groups": [],
            "totals": {
                "validated_total": {"low": 0, "high": 0},
                "probable_total": {"low": 0, "high": 0},
                "unreviewed_risk_total": {"low": 0, "high": 0},
                "inspection_allowance_total": {"low": 0, "high": 0},
                "risk_exposure_total": {"low": 0, "high": 0},
            },
            "raw_totals": {"low": 0, "high": 0},
            "primary_estimate": {
                "low": 0, "high": 0,
                "source": "probable_total",
                "validated_portion": {"low": 0, "high": 0},
            },
            "meta": {
                "candidate_count": 0,
                "estimate_unit_count": 0,
                "high_tier_count": 0,
                "medium_tier_count": 0,
                "groups_active": 0,
                "pass_2f_reviewed_count": 0,
                "pass_2f_applied_count": 0,
                "pass_2f_invalidated_count": 0,
            },
            "pass_2f_review_audit": {
                "ran": False,
                "reviewed_count": 0,
                "applied_count": 0,
                "invalidated_count": 0,
                "fallback_count": 0,
                "consistency_flag_counts": {},
                "items": [],
            },
            "disclaimer": "No estimate-eligible items detected.",
        }

    # Group candidates by estimate group
    by_group: Dict[str, List[EstimateCandidate]] = {}
    for c in candidates:
        by_group.setdefault(c.estimate_meta.group, []).append(c)

    groups_out: List[Dict[str, Any]] = []
    for group, group_candidates in by_group.items():
        result = compute_group_estimate(group, group_candidates)
        groups_out.append(result)

    # Sort: high-tier groups first, then by high desc
    groups_out.sort(key=lambda g: (
        -int(g["has_high_tier"]),
        -g["high"],
    ))

    total_low = sum(g["low"] for g in groups_out)
    total_high = sum(g["high"] for g in groups_out)
    inspection_allowance_low = sum(g.get("inspection_allowance_low", 0) for g in groups_out)
    inspection_allowance_high = sum(g.get("inspection_allowance_high", 0) for g in groups_out)
    risk_exposure_low = sum(g.get("risk_exposure_low", 0) for g in groups_out)
    risk_exposure_high = sum(g.get("risk_exposure_high", 0) for g in groups_out)
    high_tier_count = sum(1 for c in candidates if c.estimate_meta.estimate_tier == "high")
    medium_tier_count = sum(1 for c in candidates if c.estimate_meta.estimate_tier == "medium")

    # ── Build 2f review audit from eligible candidates ──
    eligible_candidates = [c for c in candidates if c.estimate_meta.estimate_tier in ("high", "medium")]
    pass_2f_reviewed_count = sum(1 for c in candidates if c.pass_2f_attempted)
    pass_2f_applied_count = sum(1 for c in candidates if c.pass_2f_applied)
    pass_2f_invalidated_count = sum(1 for c in candidates if c.is_valid_detection is False)
    consistency_flag_counts = _consistency_flag_counts(eligible_candidates)
    pass_2f_review_audit = {
        "ran": any(c.pass_2f_attempted for c in candidates),
        "reviewed_count": pass_2f_reviewed_count,
        "applied_count": pass_2f_applied_count,
        "invalidated_count": pass_2f_invalidated_count,
        "fallback_count": sum(
            1 for c in candidates
            if c.pass_2f_attempted and not c.pass_2f_applied
        ),
        "consistency_flag_counts": consistency_flag_counts,
        "items": [
            {
                "estimate_unit_id": c.estimate_unit_id,
                "billable_estimate_unit_id": c.billable_estimate_unit_id,
                "estimate_scope_key": c.estimate_scope_key,
                "resolved_cluster_key": c.resolved_cluster_key,
                "room_surrogate_id": c.room_surrogate_id,
                "source_room_surrogate_ids": c.source_room_surrogate_ids,
                "catalog_item_id": c.catalog_item_id,
                "name": c.catalog_item_name,
                "estimate_scope": c.estimate_scope,
                "estimate_scope_reason": c.estimate_scope_reason,
                "baseline_scope_before_posture": c.baseline_scope_before_posture,
                "visible_required_with_inspect_posture": c.visible_required_with_inspect_posture,
                "required_baseline_included": c.required_baseline_included,
                "inspection_risk_added": c.inspection_risk_added,
                "cost_model": c.cost_model,
                "cost_model_source": c.cost_model_source,
                "estimate_tier": c.estimate_meta.estimate_tier,
                "occurrences": c.occurrences,
                "estimate_unit_count": c.estimate_unit_count,
                "unit_policy": c.estimate_meta.unit_policy,
                "estimate_unit_label": c.estimate_unit_label,
                "unit_resolution_method": c.unit_resolution_method,
                "unit_resolution_confidence": c.unit_resolution_confidence,
                "unit_resolution_notes": c.unit_resolution_notes,
                "unit_members": c.unit_members,
                "source_estimate_unit_ids": c.source_estimate_unit_ids,
                "source_issue_ids": c.source_issue_ids,
                "is_valid_detection": c.is_valid_detection,
                "visible_scope": c.review_visible_scope,
                "default_posture": c.estimate_meta.strategy,
                "review_posture": c.review_posture,
                "pricing_posture": c.review_posture,
                "effective_posture": c.effective_posture,
                "consistency_flags": c.review_consistency_flags,
                "review_source": c.review_source,
                "review_image_path": c.review_image_path,
                "rationale": c.review_rationale,
                "pass_2f_attempted": c.pass_2f_attempted,
                "pass_2f_applied": c.pass_2f_applied,
                "pass_2f_fallback_reason": c.pass_2f_fallback_reason,
            }
            for c in eligible_candidates
        ],
    }

    # ── Tier summary & primary estimate ──
    tier_summary = compute_tier_summary(candidates, groups_out)
    validated_low = tier_summary["validated_total"]["low"]
    validated_high = tier_summary["validated_total"]["high"]

    unreviewed_low = sum(
        item.get("cost_low", 0)
        for item in tier_summary.get("unreviewed_items", [])
        if item.get("is_valid_detection") is not False
    )
    unreviewed_high = sum(
        item.get("cost_high", 0)
        for item in tier_summary.get("unreviewed_items", [])
        if item.get("is_valid_detection") is not False
    )
    totals = {
        "validated_total": {"low": validated_low, "high": validated_high},
        "probable_total": {"low": total_low, "high": total_high},
        "unreviewed_risk_total": {"low": unreviewed_low, "high": unreviewed_high},
        "inspection_allowance_total": {
            "low": inspection_allowance_low,
            "high": inspection_allowance_high,
        },
        "risk_exposure_total": {
            "low": risk_exposure_low,
            "high": risk_exposure_high,
        },
    }
    primary_estimate = {
        "low": total_low,
        "high": total_high,
        "source": "probable_total",
        "validated_portion": {"low": validated_low, "high": validated_high},
    }

    return {
        "version": "renovation_estimate_v3",
        "groups": groups_out,
        "totals": totals,
        "raw_totals": {
            "low": total_low,
            "high": total_high,
        },
        "primary_estimate": primary_estimate,
        "meta": {
            "candidate_count": len(candidates),
            "estimate_unit_count": sum(max(1, c.estimate_unit_count) for c in candidates),
            "high_tier_count": high_tier_count,
            "medium_tier_count": medium_tier_count,
            "groups_active": len(groups_out),
            "pass_2f_reviewed_count": pass_2f_reviewed_count,
            "pass_2f_applied_count": pass_2f_applied_count,
            "pass_2f_invalidated_count": pass_2f_invalidated_count,
        },
        "pass_2f_review_audit": pass_2f_review_audit,
        "disclaimer": (
            "Quick photo-based estimate. Ranges reflect allowance-level budgets, "
            "not contractor bids. Inspect-only items carry a nominal inspection allowance. "
            "Primary estimate mirrors probable_total; validated and risk exposure totals "
            "are shown separately."
        ),
    }
