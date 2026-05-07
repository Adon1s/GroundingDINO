"""
Estimate scope classification and rollups.

This module is deliberately pure: it accepts candidate/catalog-like dicts or
objects and returns deterministic scope labels for audit and UI selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


REQUIRED_REHAB = "required_rehab"
MARKETABILITY_REHAB = "marketability_rehab"
OPTIONAL_VALUE_ADD = "optional_value_add"
INSPECTION_RISK = "inspection_risk"

VALID_ESTIMATE_SCOPES = (
    REQUIRED_REHAB,
    MARKETABILITY_REHAB,
    OPTIONAL_VALUE_ADD,
    INSPECTION_RISK,
)

VISIBLE_SCOPE_PRIORITY = (
    REQUIRED_REHAB,
    MARKETABILITY_REHAB,
    OPTIONAL_VALUE_ADD,
)

_INSPECT_POSTURES = {"inspect", "inspect_only"}
_STRUCTURAL_CATEGORIES = {
    "safety", "structure", "structural", "systems", "system",
    "moisture", "remediation", "foundation", "electrical", "plumbing",
}
_REQUIRED_TERMS = (
    "missing", "bare", "stripped", "exposed subfloor", "exposed framing",
    "exposed studs", "open framing", "stripped to studs", "boarded",
    "boarded-up", "boarded up", "nonfunctional", "not functional",
    "not working", "inoperable", "broken", "active leak", "water damage",
    "water intrusion", "rotted", "rot ", "mold", "structural",
    "sagging", "collapsed", "failure",
)
_VISIBLE_REQUIRED_CONDITION_TERMS = (
    # Missing, stripped, bare, or removed finish/major components.
    "missing", "bare", "stripped", "removed", "no drywall", "no cabinets",
    "no cabinet", "missing finish", "unfinished floor", "exposed subfloor",
    "exposed framing", "exposed studs", "open framing", "stripped to studs",
    "exposed insulation",
    # Boarded, broken, or damaged openings.
    "boarded", "boarded-up", "boarded up", "secured opening",
    "broken window", "broken door", "damaged opening", "damaged entry",
    "missing window", "missing door",
    # Visible water/moisture damage.
    "active leak", "water damage", "water intrusion", "moisture intrusion",
    "moisture damage", "moisture-damaged", "visible mold", "mold",
    "mildew", "rotted", "rot ", "wood rot", "saturated",
    # Visible structural damage.
    "structural damage", "structurally compromised", "compromised framing",
    "load-bearing", "collapsed", "sagging", "soft floor",
    "damaged framing", "damaged structural", "severe moisture damage",
    # Unsafe visible systems.
    "unsafe", "electrical risk", "electrical hazard", "exposed wiring",
    "missing outlet cover", "scorch marks", "unfinished light connection",
    "fixture wiring exposed", "plumbing leak", "gas leak",
    # Removed/nonfunctional major components.
    "nonfunctional", "not functional", "not working", "inoperable",
    "major component", "cabinet removed",
)
_MARKETABILITY_TERMS = (
    "dated", "outdated", "older", "old ", "worn", "stained", "scuff",
    "cosmetic", "refresh", "paint", "repaint", "style", "builder grade",
    "builder-grade", "vanity", "fixture", "lighting", "hardware",
    "finish", "finishes", "wallpaper",
)
_VALUE_ADD_TERMS = (
    "layout", "modernization", "modernize", "premium", "value add",
    "value-add", "addition", "finish basement", "basement finishing",
    "reconfigure", "open walls", "open wall", "conversion",
)


@dataclass(frozen=True)
class EstimateScopeClassification:
    estimate_scope: str
    estimate_scope_reason: str
    baseline_scope_before_posture: str
    baseline_scope_reason: str
    visible_required_with_inspect_posture: bool
    required_baseline_included: bool
    inspection_risk_added: bool


def classify_estimate_scope(
    candidate: Any,
    catalog_item: Dict[str, Any],
    pass_2f: Optional[Any] = None,
) -> str:
    """Return one of the four valid estimate scope labels."""
    scope, _reason = classify_estimate_scope_with_reason(
        candidate,
        catalog_item,
        pass_2f,
    )
    return scope


def classify_estimate_scope_with_reason(
    candidate: Any,
    catalog_item: Dict[str, Any],
    pass_2f: Optional[Any] = None,
) -> Tuple[str, str]:
    """Classify a candidate and include a compact audit reason."""
    result = classify_estimate_scope_details(candidate, catalog_item, pass_2f)
    return result.estimate_scope, result.estimate_scope_reason


def classify_estimate_scope_details(
    candidate: Any,
    catalog_item: Dict[str, Any],
    pass_2f: Optional[Any] = None,
) -> EstimateScopeClassification:
    """Classify scope and expose the inspect-posture audit overlay."""
    catalog_item = catalog_item or {}
    baseline_scope, baseline_reason = _classify_baseline_scope_with_reason(
        candidate,
        catalog_item,
    )
    has_inspect_posture = _has_inspect_posture(candidate, pass_2f)
    invalid_detection = _is_invalid_detection(candidate, pass_2f)
    visible_required_with_inspect = (
        has_inspect_posture
        and not invalid_detection
        and baseline_scope == REQUIRED_REHAB
        and _is_visible_required_condition(candidate, catalog_item)
    )

    estimate_scope = baseline_scope
    estimate_scope_reason = baseline_reason
    inspection_risk_added = has_inspect_posture and not invalid_detection
    if inspection_risk_added and not visible_required_with_inspect:
        estimate_scope = INSPECTION_RISK
        estimate_scope_reason = "inspect_posture"

    required_baseline_included = (
        baseline_scope == REQUIRED_REHAB
        and estimate_scope == REQUIRED_REHAB
    )
    return EstimateScopeClassification(
        estimate_scope=estimate_scope,
        estimate_scope_reason=estimate_scope_reason,
        baseline_scope_before_posture=baseline_scope,
        baseline_scope_reason=baseline_reason,
        visible_required_with_inspect_posture=visible_required_with_inspect,
        required_baseline_included=required_baseline_included,
        inspection_risk_added=inspection_risk_added,
    )


def _classify_baseline_scope_with_reason(
    candidate: Any,
    catalog_item: Dict[str, Any],
) -> Tuple[str, str]:
    """Classify visible scope before Pass 2f inspect posture is applied."""
    override = _catalog_scope_override(catalog_item)
    if override:
        reason = (
            catalog_item.get("estimate_scope_reason")
            or (catalog_item.get("estimate") or {}).get("estimate_scope_reason")
            or "catalog_override"
        )
        return override, str(reason)

    kind = str(_get(candidate, "kind", catalog_item.get("kind", "defect")) or "").lower()
    tier = str(catalog_item.get("tier") or "").lower()
    category = str(catalog_item.get("category") or "").lower()
    severity = _int(_get(candidate, "severity", catalog_item.get("severity", 0)))
    scope = str(_get(candidate, "scope", catalog_item.get("scope", "")) or "").lower()
    text = _catalog_text(candidate, catalog_item)

    has_value_add = _contains_any(text, _VALUE_ADD_TERMS)
    has_marketability = (
        _contains_any(text, _MARKETABILITY_TERMS)
        or category in {"cosmetic", "opportunity"}
        or scope == "cosmetic"
    )

    if tier == "optional" and has_value_add and not has_marketability:
        return OPTIONAL_VALUE_ADD, "optional_value_add_signal"

    if kind == "upgrade":
        if tier == "optional" and has_value_add:
            return OPTIONAL_VALUE_ADD, "optional_modernization"
        if has_marketability:
            return MARKETABILITY_REHAB, "upgrade_marketability"
        if tier == "optional":
            return OPTIONAL_VALUE_ADD, "optional_upgrade"
        return MARKETABILITY_REHAB, "upgrade_default"

    if kind == "defect":
        if tier == "optional":
            return OPTIONAL_VALUE_ADD, "optional_defect"
        if severity >= 3:
            return REQUIRED_REHAB, "defect_severity_threshold"
        if category in _STRUCTURAL_CATEGORIES:
            return REQUIRED_REHAB, "required_category"
        if _contains_any(text, _REQUIRED_TERMS):
            return REQUIRED_REHAB, "required_condition_signal"
        if has_marketability:
            return MARKETABILITY_REHAB, "cosmetic_defect_marketability"
        return REQUIRED_REHAB, "defect_default"

    if has_value_add:
        return OPTIONAL_VALUE_ADD, "value_add_signal"
    if has_marketability:
        return MARKETABILITY_REHAB, "marketability_signal"
    return REQUIRED_REHAB, "fallback_required"


def apply_estimate_scope(
    candidate: Any,
    catalog_item: Dict[str, Any],
    pass_2f: Optional[Any] = None,
) -> Any:
    """Stamp estimate_scope and estimate_scope_reason onto a candidate."""
    result = classify_estimate_scope_details(
        candidate,
        catalog_item,
        pass_2f,
    )
    setattr(candidate, "estimate_scope", result.estimate_scope)
    setattr(candidate, "estimate_scope_reason", result.estimate_scope_reason)
    setattr(
        candidate,
        "baseline_scope_before_posture",
        result.baseline_scope_before_posture,
    )
    setattr(
        candidate,
        "visible_required_with_inspect_posture",
        result.visible_required_with_inspect_posture,
    )
    setattr(
        candidate,
        "required_baseline_included",
        result.required_baseline_included,
    )
    setattr(candidate, "inspection_risk_added", result.inspection_risk_added)
    return candidate


def choose_scope(scope_reason_pairs: Iterable[Tuple[Any, Any]]) -> Tuple[str, str]:
    """Merge several scope records into one conservative representative scope."""
    cleaned: List[Tuple[str, str]] = []
    for raw_scope, raw_reason in scope_reason_pairs:
        scope = str(raw_scope or "").strip()
        if scope not in VALID_ESTIMATE_SCOPES:
            continue
        cleaned.append((scope, str(raw_reason or "")))

    if not cleaned:
        return REQUIRED_REHAB, "scope_default"

    priority = (INSPECTION_RISK, REQUIRED_REHAB, MARKETABILITY_REHAB, OPTIONAL_VALUE_ADD)
    for scope in priority:
        reasons = [reason for s, reason in cleaned if s == scope and reason]
        if reasons:
            return scope, reasons[0]
        if any(s == scope for s, _reason in cleaned):
            return scope, "merged_scope"
    return REQUIRED_REHAB, "scope_default"


def classify_package_scope(
    package_type: str,
    supporting_candidates: Iterable[Any],
    trigger_reason: str = "",
) -> Tuple[str, str]:
    """Classify a package from its package type and supporting drivers."""
    package_type_l = str(package_type or "").lower()
    trigger_l = str(trigger_reason or "").lower()
    candidates = list(supporting_candidates or [])
    pairs = [
        (
            _get(candidate, "estimate_scope", ""),
            _get(candidate, "estimate_scope_reason", ""),
        )
        for candidate in candidates
    ]
    scopes = [scope for scope, _reason in pairs if scope in VALID_ESTIMATE_SCOPES]

    if "inspect" in package_type_l or "hidden" in package_type_l or INSPECTION_RISK in scopes:
        return INSPECTION_RISK, "package_inspection_risk"
    if "refresh" in package_type_l or "refresh" in trigger_l:
        return MARKETABILITY_REHAB, "package_refresh_marketability"
    if REQUIRED_REHAB in scopes:
        return REQUIRED_REHAB, "package_required_defect_driver"
    if MARKETABILITY_REHAB in scopes:
        return MARKETABILITY_REHAB, "package_marketability_drivers"
    if OPTIONAL_VALUE_ADD in scopes:
        return OPTIONAL_VALUE_ADD, "package_optional_value_add"
    return choose_scope(pairs)


def empty_scope_totals() -> Dict[str, Dict[str, int]]:
    return {scope: {"low": 0, "high": 0} for scope in VALID_ESTIMATE_SCOPES}


def add_scope_amount(
    totals: Dict[str, Dict[str, int]],
    scope: Any,
    low: Any,
    high: Any,
) -> None:
    scope_key = str(scope or "")
    if scope_key not in VALID_ESTIMATE_SCOPES:
        scope_key = REQUIRED_REHAB
    bucket = totals.setdefault(scope_key, {"low": 0, "high": 0})
    bucket["low"] = int(bucket.get("low") or 0) + _int(low)
    bucket["high"] = int(bucket.get("high") or 0) + _int(high)


def sum_scope_totals(
    totals_by_group: Dict[str, Dict[str, Dict[str, int]]],
) -> Dict[str, Dict[str, int]]:
    out = empty_scope_totals()
    for group_totals in totals_by_group.values():
        for scope, bucket in group_totals.items():
            add_scope_amount(out, scope, bucket.get("low", 0), bucket.get("high", 0))
    return out


def allocate_capped_scope_totals(
    raw_by_group: Dict[str, Dict[str, Dict[str, int]]],
    caps_by_group: Dict[str, Dict[str, int]],
    *,
    inspection_risk: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, int]]:
    """Allocate each group cap to visible scopes in priority order."""
    out = empty_scope_totals()
    for group_name, group_totals in raw_by_group.items():
        cap = caps_by_group.get(group_name, {})
        low_remaining = _int(cap.get("low", 0))
        high_remaining = _int(cap.get("high", 0))
        for scope in VISIBLE_SCOPE_PRIORITY:
            raw = group_totals.get(scope, {})
            low = min(_int(raw.get("low", 0)), low_remaining)
            high = min(_int(raw.get("high", 0)), high_remaining)
            add_scope_amount(out, scope, low, high)
            low_remaining = max(0, low_remaining - low)
            high_remaining = max(0, high_remaining - high)

    if inspection_risk:
        out[INSPECTION_RISK] = {
            "low": _int(inspection_risk.get("low", 0)),
            "high": _int(inspection_risk.get("high", 0)),
        }
    return out


def build_final_bucket(
    low: Any,
    high: Any,
    *,
    basis: str,
    source: str = "renovation_estimate_v4",
) -> Dict[str, Any]:
    low_i = _int(low)
    high_i = _int(high)
    return {
        "low": low_i,
        "high": high_i,
        "midpoint": (low_i + high_i) // 2,
        "basis": basis,
        "source": source,
    }


def build_required_and_resale_ready(
    capped_totals: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    required = capped_totals.get(REQUIRED_REHAB, {})
    marketability = capped_totals.get(MARKETABILITY_REHAB, {})
    required_bucket = build_final_bucket(
        required.get("low", 0),
        required.get("high", 0),
        basis="totals_by_scope_capped.required_rehab",
    )
    resale_low = _int(required.get("low", 0)) + _int(marketability.get("low", 0))
    resale_high = _int(required.get("high", 0)) + _int(marketability.get("high", 0))
    resale_bucket = build_final_bucket(
        resale_low,
        resale_high,
        basis="totals_by_scope_capped.required_rehab_plus_marketability_rehab",
    )
    return required_bucket, resale_bucket


def _catalog_scope_override(catalog_item: Dict[str, Any]) -> str:
    raw = (
        catalog_item.get("estimate_scope")
        or (catalog_item.get("estimate") or {}).get("estimate_scope")
    )
    scope = str(raw or "").strip()
    return scope if scope in VALID_ESTIMATE_SCOPES else ""


def _has_inspect_posture(candidate: Any, pass_2f: Optional[Any]) -> bool:
    postures = {
        str(_get(candidate, "review_posture", "") or "").lower(),
        str(_get(candidate, "effective_posture", "") or "").lower(),
    }
    estimate_meta = _get(candidate, "estimate_meta", None)
    if estimate_meta is not None:
        postures.add(str(getattr(estimate_meta, "strategy", "") or "").lower())
    if pass_2f is not None:
        for field_name in ("pricing_posture", "review_posture", "effective_posture"):
            value = _get(pass_2f, field_name, None)
            if value is not None:
                postures.add(str(value or "").lower())
    return bool(postures & _INSPECT_POSTURES)


def _is_invalid_detection(candidate: Any, pass_2f: Optional[Any]) -> bool:
    if _get(candidate, "is_valid_detection", None) is False:
        return True
    return pass_2f is not None and _get(pass_2f, "is_valid_detection", None) is False


def _is_visible_required_condition(candidate: Any, catalog_item: Dict[str, Any]) -> bool:
    """Detect definite visible required-condition defects without ID allowlists."""
    kind = str(_get(candidate, "kind", catalog_item.get("kind", "defect")) or "").lower()
    tier = str(catalog_item.get("tier") or "").lower()
    if kind != "defect" or tier == "optional":
        return False

    text = _catalog_text(candidate, catalog_item)
    if _contains_any(text, _VISIBLE_REQUIRED_CONDITION_TERMS):
        return True

    category = str(catalog_item.get("category") or "").lower()
    trade_bucket = str(catalog_item.get("trade_bucket") or "").lower()
    severity = _int(_get(candidate, "severity", catalog_item.get("severity", 0)))
    is_system = category in {"safety", "systems", "system", "electrical", "plumbing"}
    is_system = is_system or trade_bucket in {"electrical", "plumbing", "safety_general"}
    return severity >= 3 and is_system and _contains_any(text, (
        "hazard", "unsafe", "exposed", "missing", "broken", "nonfunctional",
        "not working", "inoperable", "leak",
    ))


def _catalog_text(candidate: Any, catalog_item: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for source in (catalog_item,):
        for field_name in (
            "id", "name", "category", "description", "scope", "trade_bucket",
            "work_item_code", "embed_text", "tier",
        ):
            value = source.get(field_name)
            if value is not None:
                chunks.append(str(value))
        support_any = source.get("support_any")
        if isinstance(support_any, list):
            chunks.extend(str(v) for v in support_any if v is not None)
    for field_name in (
        "catalog_item_id", "catalog_item_name", "kind", "scope", "trade_bucket",
    ):
        value = _get(candidate, field_name, None)
        if value is not None:
            chunks.append(str(value))
    for value in (_get(candidate, "supporting_observations", []) or []):
        if value is not None:
            chunks.append(str(value))
    return " ".join(chunks).lower()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _get(obj: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
