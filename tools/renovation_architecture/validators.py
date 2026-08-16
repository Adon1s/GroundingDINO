"""Strict validators for the renovation architecture contracts.

Hand-rolled, stdlib only — matching tools/catalog_validation.py and
tools/benchmarking/schemas.py. (The one repo import beyond the contracts is
sha256_canonical, so the snapshot fingerprints have a single definition.)
Validators operate on the plain-dict form of the contracts (the JSON
boundary), collect ALL errors rather than stopping at the first, and format
every message as "where: message".

Unknown-field rejection is the enforcement mechanism for the layer
boundaries: a ConditionReview or PackageDecision carrying work, package,
price, quantity, or confidence fields cannot validate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from tools.comparison_common import sha256_canonical
from tools.renovation_architecture.contracts import (
    ACTION_SOURCES,
    APPLICATION_REASON_CODES,
    APPLICATION_STATUSES,
    ARCHITECTURE_MODES,
    CONDITION_DISPOSITION_POLICY_VERSION,
    CONTRACTS_SCHEMA_VERSION,
    DEDUP_SUPPRESSION_REASON,
    DISPOSITION_REASON_CODES,
    DISPOSITIONS,
    ENVELOPE_SCHEMA_VERSION,
    ENVELOPE_STATES,
    ESTIMATE_SCOPE_MERGE_PRIORITY,
    ESTIMATE_SCOPES,
    EVIDENCE_DEDUP_POLICY_VERSION,
    FUNNEL_KEYS,
    LEDGER_REASON_CODES,
    LEDGER_REPRESENTATIONS,
    OBSERVABILITY_PHASES,
    OBSERVATION_KINDS_V2,
    PACKAGE_CATEGORIES,
    PACKAGE_DECISIONS,
    PACKAGE_LEVELS,
    PACKAGE_ROOMS,
    PACKAGE_STRENGTHS,
    PACKAGE_TREATMENTS,
    PACKAGE_TYPES,
    POLICY_VERSION_KEYS,
    PRICING_MODES,
    PROJECTION_VERSION,
    REQUIRED_CATALOG_ONTOLOGY,
    REQUIRED_CATALOG_VERSION,
    REQUIRED_KIND_ONTOLOGY_SELECTOR,
    REVIEW_RATIONALE_MAX_CHARS,
    REVIEW_VERDICTS,
    SCAFFOLD_REASON,
    STANDALONE_PRICING_POLICY_VERSION,
    TERMINAL_ROUTES,
    TERRA_USAGE_SOURCES,
    UNIT_POLICIES,
    UNIT_RESOLUTION_SOURCES,
    WHOLE_HOME_PACKAGE_TYPE,
    WHOLE_HOME_PRICING_PROFILE,
    WHOLE_HOME_PRICING_TIER,
    WHOLE_HOME_UNIT_ID,
    WORK_DEDUP_POLICY_VERSION,
    WORK_ITEM_STATUSES,
)
from tools.renovation_architecture.disposition import decide_disposition

_ID_PATTERNS = {
    prefix: re.compile(rf"^{prefix}_[0-9a-f]{{16}}$")
    for prefix in (
        "rea1", "oc1", "ev1", "cr1", "cd1", "tc1", "wk1", "wdc1", "pk1",
        "pd1", "sc1", "cl1", "pa1", "cg1"
    )
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Concepts that belong to other layers. Their appearance on a model-verdict
# record is reported as a boundary violation, not a generic unknown field.
_LAYER_BOUNDARY_FIELDS = frozenset(
    {"work", "work_items", "package", "packages", "price", "prices",
     "low", "high", "cost", "confidence", "quantity", "quantities"}
)

_ENVELOPE_FIELDS = frozenset(
    {"schema_version", "estimate_id", "state", "reason", "error_detail",
     "provenance", "result"}
)
_PROVENANCE_FIELDS = frozenset(
    {"schema_version", "architecture_mode", "contracts_schema_version",
     "projection_version", "catalog_version", "catalog_ontology_version",
     "catalog_sha256", "projection_fingerprint", "kind_ontology_selector",
     "policy_versions", "property_key", "source_run_id", "source_artifact",
     "created_at"}
)
_CONDITION_FIELDS = frozenset(
    {"condition_id", "schema_version", "catalog_item_id", "catalog_kind",
     "scope_key", "estimate_unit_id", "room_surrogate_id", "scene_group",
     "issue_ids", "identity_ambiguous", "source_room_surrogate_ids",
     "source_scope_keys", "unit_resolution_source", "unit_resolution_reason",
     "opening_instance_hints"}
)
_EVIDENCE_FIELDS = frozenset(
    {"evidence_id", "schema_version", "condition_id", "photo_keys",
     "distinct_photo_count", "distinct_view_count", "duplicate_groups",
     "evidence_refs", "min_photo_evidence_required",
     "representative_photo_keys", "exact_duplicate_groups",
     "near_duplicate_groups", "dedup_policy_version"}
)
_EVIDENCE_REF_FIELDS = frozenset(
    {"issue_id", "photo_key", "observation", "room_surrogate_id"}
)
_REVIEW_FIELDS = frozenset(
    {"review_id", "schema_version", "condition_id", "verdict", "rationale",
     "model", "prompt_version", "terra_call_id", "request_fingerprint",
     "provider"}
)
_DISPOSITION_FIELDS = frozenset(
    {"disposition_id", "schema_version", "condition_id", "review_id",
     "disposition", "reason_code", "policy_version", "evidence_id",
     "terminal_route"}
)
_TERRA_CALL_FIELDS = frozenset(
    {"call_id", "schema_version", "estimate_unit_id", "condition_ids",
     "request_fingerprint", "provider", "model", "prompt_version",
     "usage_source", "input_tokens", "cached_input_tokens", "output_tokens",
     "total_tokens", "budget_debited_tokens"}
)
_TERRA_UNIT_USAGE_FIELDS = frozenset(
    {"schema_version", "estimate_unit_id", "call_ids", "input_tokens",
     "cached_input_tokens", "output_tokens", "total_tokens",
     "budget_debited_tokens"}
)
_TERRA_LISTING_USAGE_FIELDS = frozenset(
    {"schema_version", "call_count", "input_tokens", "cached_input_tokens",
     "output_tokens", "total_tokens", "budget_debited_tokens"}
)
_TERRA_TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
    "budget_debited_tokens",
)
_WORK_ITEM_FIELDS = frozenset(
    {"work_item_id", "schema_version", "condition_ids", "catalog_item_ids",
     "source_estimate_unit_ids", "billable_unit_id", "action_code",
     "action_sources", "trade_bucket", "unit_policy", "unit_count",
     "pricing_modes", "identity_ambiguous", "estimate_scope",
     "estimate_scope_reason", "low", "high", "status", "reason_code"}
)
_WORK_DEDUP_COLLISION_FIELDS = frozenset(
    {"collision_id", "schema_version", "action_code", "trade_bucket",
     "unit_policy", "billable_unit_id", "active_work_item_id",
     "suppressed_work_item_ids", "policy_version"}
)
_STANDALONE_ESTIMATE_FIELDS = frozenset(
    {"schema_version", "currency", "pricing_policy_version",
     "property_cost_factor", "property_cost_factor_audit",
     "totals_by_estimate_scope", "headline"}
)
# The dedup-key fields shared verbatim by a collision record, its merged
# active, and every suppressed source.
_WORK_DEDUP_KEY_FIELDS = (
    "action_code", "trade_bucket", "unit_policy", "billable_unit_id"
)
# Collapse unit policies bill one fixed synthetic unit; every other policy
# bills the physical estimate unit the conditions resolved to.
_COLLAPSE_BILLABLE_UNITS = {
    "per_property": "property", "per_system": "system", "per_area": "area"
}
_PACKAGE_CANDIDATE_FIELDS = frozenset(
    {"package_candidate_id", "schema_version", "package_type",
     "package_category", "package_level", "room", "estimate_unit_id",
     "child_work_item_ids", "driver_work_item_ids", "support_work_item_ids",
     "strength", "pricing_profile", "pricing_tier", "absorption_scope",
     "proposed_treatment", "unfloored_low", "unfloored_high",
     "cost_floor_applied", "low", "high", "display_only",
     "contributing_candidate_ids"}
)
_ABSORPTION_SCOPE_FIELDS = frozenset(
    {"family", "groups", "trade_buckets", "components"}
)
_PACKAGE_DECISION_FIELDS = frozenset(
    {"decision_id", "schema_version", "package_candidate_id", "decision",
     "combine_with", "split_groups", "rationale", "model", "prompt_version",
     "sol_call_id", "request_fingerprint", "provider"}
)
_SOL_CALL_FIELDS = frozenset(
    {"call_id", "schema_version", "package_candidate_ids",
     "request_fingerprint", "provider", "model", "prompt_version",
     "usage_source", "input_tokens", "cached_input_tokens", "output_tokens",
     "total_tokens"}
)
_SOL_LISTING_USAGE_FIELDS = frozenset(
    {"schema_version", "call_count", "input_tokens", "cached_input_tokens",
     "output_tokens", "total_tokens"}
)
_SOL_TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
)
_SNAPSHOT_FIELDS = frozenset(
    {"schema_version", "condition_snapshot_sha256", "work_snapshot_sha256",
     "candidate_snapshot_sha256"}
)
# The layers Sol must never change, exactly as grouped into the snapshot
# fingerprints (package_review_snapshot_hashes below).
_CONDITION_SNAPSHOT_KEYS = (
    "observed_conditions", "evidence_facts", "condition_reviews",
    "condition_dispositions",
)
_WORK_SNAPSHOT_KEYS = (
    "work_items", "work_dedup_collisions", "standalone_estimate",
)
_LEDGER_FIELDS = frozenset(
    {"entry_id", "schema_version", "work_item_id", "representation",
     "package_id", "reason_code", "low", "high"}
)
_APPLICATION_FIELDS = frozenset(
    {"application_id", "schema_version", "package_candidate_id",
     "decision_id", "status", "reason_code", "absorbed_work_item_ids",
     "unabsorbed_child_work_item_ids", "combine_group_id", "effective_low",
     "effective_high"}
)
_AUDIT_LIST_FIELDS = (
    "lost_work_item_ids", "duplicate_absorption", "unsupported_billing",
    "orphan_children", "arithmetic_mismatches",
)
_AUDIT_FIELDS = frozenset(_AUDIT_LIST_FIELDS) | {"schema_version"}
_OBSERVABILITY_FIELDS = frozenset(
    {"schema_version", "phase_timings_ms", "terra_total_tokens",
     "sol_total_tokens", "combined_total_tokens", "funnel"}
)
_TOTALS_FIELDS = frozenset(
    {"schema_version", "currency", "standalone", "packaged", "inspection",
     "headline"}
)
# The Session 2 intermediate result: the review layer is finished, the work
# and package layers do not exist yet. terra_listing_usage is one object,
# every other key is a list.
_REVIEW_RESULT_KEYS = frozenset(
    {"observed_conditions", "evidence_facts", "condition_reviews",
     "condition_dispositions", "terra_calls", "terra_unit_usage",
     "terra_listing_usage"}
)
# The Session 3 intermediate result: the frozen Session 2 result plus the
# deterministic work layer. standalone_estimate is one object; work_items and
# work_dedup_collisions are lists.
_STANDALONE_RESULT_KEYS = _REVIEW_RESULT_KEYS | frozenset(
    {"work_items", "work_dedup_collisions", "standalone_estimate"}
)
# The Session 4 intermediate result: the frozen Session 3 result plus the
# package layer. sol_listing_usage and package_review_snapshots are objects;
# the other new keys are lists.
_PACKAGE_REVIEW_RESULT_KEYS = _STANDALONE_RESULT_KEYS | frozenset(
    {"package_candidates", "package_decisions", "sol_calls",
     "sol_listing_usage", "package_review_snapshots"}
)
# The Session 5 complete result: the frozen Session 4 result plus the
# reconciliation layer. reconciliation_audit, observability, and totals are
# objects; package_applications and coverage_ledger are lists.
_RESULT_KEYS = _PACKAGE_REVIEW_RESULT_KEYS | frozenset(
    {"package_applications", "coverage_ledger", "reconciliation_audit",
     "observability", "totals"}
)


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")

    def extend(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


# ── field helpers ────────────────────────────────────────────────────────────

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_unknown(
    res: ValidationResult, where: str, obj: Dict[str, Any],
    allowed: FrozenSet[str], *, boundary: bool = False,
) -> None:
    for key in sorted(set(obj) - allowed):
        if boundary and key in _LAYER_BOUNDARY_FIELDS:
            res.error(
                where,
                f"field {key!r} violates the layer boundary — this record may "
                "not carry work, package, price, quantity, or confidence fields",
            )
        else:
            res.error(where, f"unknown field {key!r}")


def _check_schema_version(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    if obj.get("schema_version") != CONTRACTS_SCHEMA_VERSION:
        res.error(
            where,
            f"schema_version must be {CONTRACTS_SCHEMA_VERSION}, "
            f"got {obj.get('schema_version')!r}",
        )


def _require_str(
    res: ValidationResult, where: str, obj: Dict[str, Any], name: str,
    *, allow_empty: bool = False, nullable: bool = False,
) -> Optional[str]:
    value = obj.get(name)
    if value is None:
        if nullable and name in obj:
            return None
        res.error(where, f"{name} is required")
        return None
    if not isinstance(value, str):
        res.error(where, f"{name} must be a string, got {type(value).__name__}")
        return None
    if not value and not allow_empty:
        res.error(where, f"{name} must be non-empty")
        return None
    return value


def _require_int(
    res: ValidationResult, where: str, obj: Dict[str, Any], name: str,
    *, minimum: Optional[int] = None, nullable: bool = False,
) -> Optional[int]:
    value = obj.get(name)
    if value is None:
        if nullable and name in obj:
            return None
        res.error(where, f"{name} is required")
        return None
    if not _is_int(value):
        res.error(where, f"{name} must be an integer, got {type(value).__name__}")
        return None
    if minimum is not None and value < minimum:
        res.error(where, f"{name} must be >= {minimum}, got {value}")
        return None
    return value


def _require_choice(
    res: ValidationResult, where: str, obj: Dict[str, Any], name: str,
    choices: FrozenSet[str],
) -> Optional[str]:
    value = _require_str(res, where, obj, name)
    if value is not None and value not in choices:
        res.error(where, f"{name} must be one of {sorted(choices)}, got {value!r}")
        return None
    return value


def _require_id(
    res: ValidationResult, where: str, obj: Dict[str, Any], name: str, prefix: str,
) -> Optional[str]:
    value = _require_str(res, where, obj, name)
    if value is not None and not _ID_PATTERNS[prefix].match(value):
        res.error(where, f"{name} must match {prefix}_<16 hex>, got {value!r}")
        return None
    return value


def _require_str_list(
    res: ValidationResult, where: str, obj: Dict[str, Any], name: str,
    *, allow_empty: bool,
) -> Optional[List[str]]:
    value = obj.get(name)
    if not isinstance(value, list):
        res.error(where, f"{name} must be a list, got {type(value).__name__}")
        return None
    if not value and not allow_empty:
        res.error(where, f"{name} must be non-empty")
        return None
    for index, entry in enumerate(value):
        if not isinstance(entry, str) or not entry:
            res.error(where, f"{name}[{index}] must be a non-empty string")
            return None
    return value


def _require_money_pair(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    low = _require_int(res, where, obj, "low", minimum=0)
    high = _require_int(res, where, obj, "high", minimum=0)
    if low is not None and high is not None and low > high:
        res.error(where, f"low ({low}) must not exceed high ({high})")


# ── record validators ────────────────────────────────────────────────────────

def _validate_condition(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    _check_unknown(res, where, obj, _CONDITION_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "condition_id", "oc1")
    _require_str(res, where, obj, "catalog_item_id")
    _require_choice(res, where, obj, "catalog_kind", OBSERVATION_KINDS_V2)
    _require_str(res, where, obj, "scope_key")
    _require_str(res, where, obj, "estimate_unit_id")
    _require_str(res, where, obj, "room_surrogate_id", allow_empty=True)
    _require_str(res, where, obj, "scene_group")
    _require_str_list(res, where, obj, "issue_ids", allow_empty=False)
    if not isinstance(obj.get("identity_ambiguous"), bool):
        res.error(where, "identity_ambiguous must be a boolean")
    # Empty is legitimate only on the scope-room fallback path, where no
    # surrogate ever existed.
    _require_str_list(res, where, obj, "source_room_surrogate_ids", allow_empty=True)
    _require_str_list(res, where, obj, "source_scope_keys", allow_empty=False)
    _require_choice(res, where, obj, "unit_resolution_source", UNIT_RESOLUTION_SOURCES)
    _require_str(res, where, obj, "unit_resolution_reason")
    # Empty is the norm: only issues carrying explicit opening-instance
    # identifiers contribute hints.
    hints = _require_str_list(res, where, obj, "opening_instance_hints", allow_empty=True)
    if hints is not None and (len(hints) != len(set(hints)) or hints != sorted(hints)):
        res.error(where, "opening_instance_hints must be sorted and unique")


def _check_photo_groups(
    res: ValidationResult, where: str, obj: Dict[str, Any], name: str,
    known: Optional[set],
) -> None:
    groups = obj.get(name)
    if not isinstance(groups, list):
        res.error(where, f"{name} must be a list")
        return
    if known is None:
        return
    for index, group in enumerate(groups):
        if not isinstance(group, list) or not all(
            isinstance(key, str) for key in group
        ):
            res.error(where, f"{name}[{index}] must be a list of strings")
        elif not set(group) <= known:
            res.error(
                where,
                f"{name}[{index}] references photos outside photo_keys",
            )


def _validate_evidence(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    _check_unknown(res, where, obj, _EVIDENCE_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "evidence_id", "ev1")
    _require_id(res, where, obj, "condition_id", "oc1")
    photo_keys = _require_str_list(res, where, obj, "photo_keys", allow_empty=False)
    count = _require_int(res, where, obj, "distinct_photo_count", minimum=1)
    if photo_keys is not None and count is not None and count != len(set(photo_keys)):
        res.error(
            where,
            f"distinct_photo_count ({count}) must equal the number of distinct "
            f"photo_keys ({len(set(photo_keys))})",
        )
    _require_int(res, where, obj, "distinct_view_count", minimum=0, nullable=True)
    known = set(photo_keys) if photo_keys is not None else None
    for name in ("duplicate_groups", "exact_duplicate_groups", "near_duplicate_groups"):
        _check_photo_groups(res, where, obj, name, known)
    representatives = _require_str_list(
        res, where, obj, "representative_photo_keys", allow_empty=False
    )
    if representatives is not None and known is not None and not set(representatives) <= known:
        res.error(where, "representative_photo_keys must be a subset of photo_keys")
    dedup_version = _require_str(res, where, obj, "dedup_policy_version")
    if dedup_version is not None and dedup_version != EVIDENCE_DEDUP_POLICY_VERSION:
        res.error(
            where,
            f"dedup_policy_version must be {EVIDENCE_DEDUP_POLICY_VERSION!r}, "
            f"got {dedup_version!r}",
        )
    refs = obj.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        res.error(where, "evidence_refs must be a non-empty list")
    else:
        for index, ref in enumerate(refs):
            if not isinstance(ref, dict):
                res.error(where, f"evidence_refs[{index}] must be an object")
                continue
            _check_unknown(res, f"{where}.evidence_refs[{index}]", ref, _EVIDENCE_REF_FIELDS)
            for key in ("issue_id", "photo_key"):
                if not isinstance(ref.get(key), str) or not ref.get(key):
                    res.error(where, f"evidence_refs[{index}].{key} must be a non-empty string")
    _require_int(res, where, obj, "min_photo_evidence_required", minimum=0, nullable=True)


def _validate_review(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    _check_unknown(res, where, obj, _REVIEW_FIELDS, boundary=True)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "review_id", "cr1")
    _require_id(res, where, obj, "condition_id", "oc1")
    _require_choice(res, where, obj, "verdict", REVIEW_VERDICTS)
    rationale = _require_str(res, where, obj, "rationale", allow_empty=True)
    if rationale is not None and len(rationale) > REVIEW_RATIONALE_MAX_CHARS:
        res.error(
            where,
            f"rationale must be at most {REVIEW_RATIONALE_MAX_CHARS} characters, "
            f"got {len(rationale)}",
        )
    _require_str(res, where, obj, "model")
    _require_str(res, where, obj, "prompt_version")
    _require_id(res, where, obj, "terra_call_id", "tc1")
    fingerprint = _require_str(res, where, obj, "request_fingerprint")
    if fingerprint is not None and not _SHA256_RE.match(fingerprint):
        res.error(where, "request_fingerprint must be 64 lowercase hex characters")
    _require_str(res, where, obj, "provider")


def _validate_disposition(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    _check_unknown(res, where, obj, _DISPOSITION_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "disposition_id", "cd1")
    _require_id(res, where, obj, "condition_id", "oc1")
    _require_id(res, where, obj, "review_id", "cr1")
    _require_id(res, where, obj, "evidence_id", "ev1")
    _require_choice(res, where, obj, "disposition", DISPOSITIONS)
    _require_choice(res, where, obj, "reason_code", DISPOSITION_REASON_CODES)
    _require_choice(res, where, obj, "terminal_route", TERMINAL_ROUTES)
    policy_version = _require_str(res, where, obj, "policy_version")
    if policy_version is not None and policy_version != CONDITION_DISPOSITION_POLICY_VERSION:
        res.error(
            where,
            f"policy_version must be {CONDITION_DISPOSITION_POLICY_VERSION!r}, "
            f"got {policy_version!r}",
        )


def _validate_terra_call(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    _check_unknown(res, where, obj, _TERRA_CALL_FIELDS, boundary=True)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "call_id", "tc1")
    _require_str(res, where, obj, "estimate_unit_id")
    condition_ids = _require_str_list(res, where, obj, "condition_ids", allow_empty=False)
    if condition_ids is not None and len(condition_ids) != len(set(condition_ids)):
        res.error(where, "condition_ids must be unique")
    fingerprint = _require_str(res, where, obj, "request_fingerprint")
    if fingerprint is not None and not _SHA256_RE.match(fingerprint):
        res.error(where, "request_fingerprint must be 64 lowercase hex characters")
    _require_str(res, where, obj, "provider")
    _require_str(res, where, obj, "model")
    _require_str(res, where, obj, "prompt_version")
    _require_choice(res, where, obj, "usage_source", TERRA_USAGE_SOURCES)
    _check_terra_tokens(res, where, obj)


def _check_terra_tokens(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    for name in _TERRA_TOKEN_FIELDS:
        _require_int(res, where, obj, name, minimum=0)
    cached = obj.get("cached_input_tokens")
    inputs = obj.get("input_tokens")
    if _is_int(cached) and _is_int(inputs) and cached > inputs:
        res.error(
            where,
            f"cached_input_tokens ({cached}) must not exceed input_tokens ({inputs})",
        )


def _validate_terra_unit_usage(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _TERRA_UNIT_USAGE_FIELDS)
    _check_schema_version(res, where, obj)
    _require_str(res, where, obj, "estimate_unit_id")
    call_ids = _require_str_list(res, where, obj, "call_ids", allow_empty=False)
    if call_ids is not None and len(call_ids) != len(set(call_ids)):
        res.error(where, "call_ids must be unique")
    _check_terra_tokens(res, where, obj)


def _validate_terra_listing_usage(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _TERRA_LISTING_USAGE_FIELDS)
    _check_schema_version(res, where, obj)
    _require_int(res, where, obj, "call_count", minimum=0)
    _check_terra_tokens(res, where, obj)


def _require_sorted_unique(
    res: ValidationResult, where: str, obj: Dict[str, Any], name: str,
) -> Optional[List[str]]:
    """Non-empty list of unique, lexicographically sorted strings — the
    byte-deterministic lineage-tuple shape."""
    value = _require_str_list(res, where, obj, name, allow_empty=False)
    if value is None:
        return None
    if len(value) != len(set(value)):
        res.error(where, f"{name} must be unique")
        return None
    if value != sorted(value):
        res.error(where, f"{name} must be sorted")
        return None
    return value


def _validate_work_item(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    _check_unknown(res, where, obj, _WORK_ITEM_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "work_item_id", "wk1")
    _require_sorted_unique(res, where, obj, "condition_ids")
    _require_sorted_unique(res, where, obj, "catalog_item_ids")
    _require_sorted_unique(res, where, obj, "source_estimate_unit_ids")
    _require_str(res, where, obj, "billable_unit_id")
    _require_str(res, where, obj, "action_code")
    sources = _require_sorted_unique(res, where, obj, "action_sources")
    if sources is not None and not set(sources) <= ACTION_SOURCES:
        res.error(
            where,
            f"action_sources must be a subset of {sorted(ACTION_SOURCES)}, "
            f"got {sources}",
        )
    _require_str(res, where, obj, "trade_bucket")
    _require_choice(res, where, obj, "unit_policy", UNIT_POLICIES)
    _require_int(res, where, obj, "unit_count", minimum=1)
    modes = _require_sorted_unique(res, where, obj, "pricing_modes")
    if modes is not None and not set(modes) <= PRICING_MODES:
        res.error(
            where,
            f"pricing_modes must be a subset of {sorted(PRICING_MODES)}, "
            f"got {modes}",
        )
    if not isinstance(obj.get("identity_ambiguous"), bool):
        res.error(where, "identity_ambiguous must be a boolean")
    _require_choice(res, where, obj, "estimate_scope", ESTIMATE_SCOPES)
    _require_str(res, where, obj, "estimate_scope_reason")
    _require_money_pair(res, where, obj)
    status = _require_choice(res, where, obj, "status", WORK_ITEM_STATUSES)
    reason_code = obj.get("reason_code")
    if status == "suppressed":
        if reason_code != DEDUP_SUPPRESSION_REASON:
            res.error(
                where,
                f"reason_code must be {DEDUP_SUPPRESSION_REASON!r} when status "
                f"is 'suppressed', got {reason_code!r}",
            )
    elif status == "active" and reason_code is not None:
        res.error(where, "reason_code must be null unless status is 'suppressed'")


def _validate_work_dedup_collision(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _WORK_DEDUP_COLLISION_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "collision_id", "wdc1")
    _require_str(res, where, obj, "action_code")
    _require_str(res, where, obj, "trade_bucket")
    _require_choice(res, where, obj, "unit_policy", UNIT_POLICIES)
    _require_str(res, where, obj, "billable_unit_id")
    _require_id(res, where, obj, "active_work_item_id", "wk1")
    suppressed = _require_sorted_unique(res, where, obj, "suppressed_work_item_ids")
    if suppressed is not None:
        if len(suppressed) < 2:
            res.error(
                where,
                "suppressed_work_item_ids needs at least two members — one "
                "source is not a collision",
            )
        for index, work_id in enumerate(suppressed):
            if not _ID_PATTERNS["wk1"].match(work_id):
                res.error(
                    where,
                    f"suppressed_work_item_ids[{index}] must match wk1_<16 hex>, "
                    f"got {work_id!r}",
                )
    policy_version = _require_str(res, where, obj, "policy_version")
    if policy_version is not None and policy_version != WORK_DEDUP_POLICY_VERSION:
        res.error(
            where,
            f"policy_version must be {WORK_DEDUP_POLICY_VERSION!r}, "
            f"got {policy_version!r}",
        )


def _validate_standalone_estimate(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _STANDALONE_ESTIMATE_FIELDS)
    _check_schema_version(res, where, obj)
    if obj.get("currency") != "USD":
        res.error(where, f"currency must be 'USD', got {obj.get('currency')!r}")
    policy_version = _require_str(res, where, obj, "pricing_policy_version")
    if policy_version is not None and policy_version != STANDALONE_PRICING_POLICY_VERSION:
        res.error(
            where,
            f"pricing_policy_version must be {STANDALONE_PRICING_POLICY_VERSION!r}, "
            f"got {policy_version!r}",
        )
    factor = obj.get("property_cost_factor")
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or factor <= 0:
        res.error(where, f"property_cost_factor must be a positive number, got {factor!r}")
    if not isinstance(obj.get("property_cost_factor_audit"), dict):
        res.error(where, "property_cost_factor_audit must be an object")
    totals = obj.get("totals_by_estimate_scope")
    if not isinstance(totals, dict) or set(totals) != ESTIMATE_SCOPES:
        res.error(
            where,
            "totals_by_estimate_scope must carry exactly the keys "
            f"{sorted(ESTIMATE_SCOPES)}",
        )
    else:
        for scope in sorted(totals):
            entry = totals[scope]
            if not isinstance(entry, dict) or set(entry) != {"low", "high"}:
                res.error(where, f"totals_by_estimate_scope[{scope!r}] must be an object with exactly low/high")
                continue
            _require_money_pair(res, f"{where}.totals_by_estimate_scope[{scope!r}]", entry)
    headline = obj.get("headline")
    if not isinstance(headline, dict) or set(headline) != {"low", "high"}:
        res.error(where, "headline must be an object with exactly low/high")
    else:
        _require_money_pair(res, f"{where}.headline", headline)


def _require_sorted_unique_allow_empty(
    res: ValidationResult, where: str, obj: Dict[str, Any], name: str,
) -> Optional[List[str]]:
    value = _require_str_list(res, where, obj, name, allow_empty=True)
    if value is None:
        return None
    if len(value) != len(set(value)):
        res.error(where, f"{name} must be unique")
        return None
    if value != sorted(value):
        res.error(where, f"{name} must be sorted")
        return None
    return value


def _validate_package_candidate(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _PACKAGE_CANDIDATE_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "package_candidate_id", "pk1")
    _require_choice(res, where, obj, "package_type", PACKAGE_TYPES)
    _require_choice(res, where, obj, "package_category", PACKAGE_CATEGORIES)
    _require_choice(res, where, obj, "package_level", PACKAGE_LEVELS)
    _require_choice(res, where, obj, "room", PACKAGE_ROOMS)
    _require_str(res, where, obj, "estimate_unit_id")
    children = _require_sorted_unique_allow_empty(res, where, obj, "child_work_item_ids")
    drivers = _require_sorted_unique_allow_empty(res, where, obj, "driver_work_item_ids")
    supports = _require_sorted_unique_allow_empty(res, where, obj, "support_work_item_ids")
    if children is not None and drivers is not None and supports is not None:
        if set(drivers) & set(supports):
            res.error(
                where,
                "driver_work_item_ids and support_work_item_ids must be "
                "disjoint — driver precedence resolves dual roles",
            )
        if set(drivers) | set(supports) != set(children):
            res.error(
                where,
                "child_work_item_ids must be exactly the union of driver and "
                "support work item ids",
            )
    _require_choice(res, where, obj, "strength", PACKAGE_STRENGTHS)
    _require_str(res, where, obj, "pricing_profile")
    _require_str(res, where, obj, "pricing_tier")
    scope = obj.get("absorption_scope")
    if not isinstance(scope, dict) or set(scope) != _ABSORPTION_SCOPE_FIELDS:
        res.error(
            where,
            "absorption_scope must be an object with exactly "
            f"{sorted(_ABSORPTION_SCOPE_FIELDS)}",
        )
    else:
        if not isinstance(scope.get("family"), str):
            res.error(where, "absorption_scope.family must be a string")
        for name in ("groups", "trade_buckets", "components"):
            _require_sorted_unique_allow_empty(
                res, f"{where}.absorption_scope", scope, name
            )
    _require_choice(res, where, obj, "proposed_treatment", PACKAGE_TREATMENTS)
    unfloored_low = _require_int(res, where, obj, "unfloored_low", minimum=0)
    unfloored_high = _require_int(res, where, obj, "unfloored_high", minimum=0)
    if (
        unfloored_low is not None and unfloored_high is not None
        and unfloored_low > unfloored_high
    ):
        res.error(
            where,
            f"unfloored_low ({unfloored_low}) must not exceed "
            f"unfloored_high ({unfloored_high})",
        )
    _require_money_pair(res, where, obj)
    low, high = obj.get("low"), obj.get("high")
    if _is_int(low) and _is_int(high) and unfloored_low is not None and unfloored_high is not None:
        if low < unfloored_low or high < unfloored_high:
            res.error(
                where,
                "the floored range may only raise the unfloored tier range, "
                "never lower it",
            )
        expected_floor = (low, high) != (unfloored_low, unfloored_high)
        if obj.get("cost_floor_applied") != expected_floor:
            res.error(
                where,
                f"cost_floor_applied must be {expected_floor} for "
                f"{unfloored_low}/{unfloored_high} -> {low}/{high}",
            )
    if not isinstance(obj.get("cost_floor_applied"), bool):
        res.error(where, "cost_floor_applied must be a boolean")
    display_only = obj.get("display_only")
    if not isinstance(display_only, bool):
        res.error(where, "display_only must be a boolean")
        return
    contributing = _require_sorted_unique_allow_empty(
        res, where, obj, "contributing_candidate_ids"
    )
    if display_only:
        # The only display-only candidate this schema emits is the
        # whole-home turnover aggregate: no children (contributor refs
        # carry the lineage, so it can never absorb), fixed identity.
        if children:
            res.error(
                where,
                "display-only candidates carry no children — lineage flows "
                "through contributing_candidate_ids",
            )
        if contributing is not None:
            if not contributing:
                res.error(
                    where,
                    "display-only candidates must cite their contributing "
                    "candidates",
                )
            for candidate_id in contributing:
                if not _ID_PATTERNS["pk1"].match(candidate_id):
                    res.error(
                        where,
                        f"contributing candidate id {candidate_id!r} must "
                        "match pk1_<16 hex>",
                    )
        for name, expected in (
            ("package_type", WHOLE_HOME_PACKAGE_TYPE),
            ("package_level", "property"),
            ("room", "whole_home"),
            ("estimate_unit_id", WHOLE_HOME_UNIT_ID),
            ("package_category", "turnover"),
            ("proposed_treatment", "whole_home_turnover_aggregate"),
            ("pricing_profile", WHOLE_HOME_PRICING_PROFILE),
            ("pricing_tier", WHOLE_HOME_PRICING_TIER),
            ("cost_floor_applied", False),
        ):
            if obj.get(name) != expected:
                res.error(
                    where,
                    f"display-only candidates must carry {name}={expected!r}, "
                    f"got {obj.get(name)!r}",
                )
    else:
        if children is not None and not children:
            res.error(
                where,
                "child_work_item_ids must be non-empty — packages exist only "
                "over accepted work",
            )
        if contributing:
            res.error(
                where,
                "contributing_candidate_ids must be empty unless the "
                "candidate is display-only",
            )
        if obj.get("package_level") == "property":
            res.error(
                where,
                "property-level candidates must be display-only aggregates",
            )
        if obj.get("package_type") == WHOLE_HOME_PACKAGE_TYPE:
            res.error(
                where,
                f"{WHOLE_HOME_PACKAGE_TYPE} is the display-only aggregate "
                "and cannot be a room candidate",
            )


def _validate_package_decision(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _PACKAGE_DECISION_FIELDS, boundary=True)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "decision_id", "pd1")
    _require_id(res, where, obj, "package_candidate_id", "pk1")
    _require_choice(res, where, obj, "decision", PACKAGE_DECISIONS)
    combine = _require_sorted_unique_allow_empty(res, where, obj, "combine_with")
    candidate_id = obj.get("package_candidate_id")
    if combine and isinstance(candidate_id, str) and candidate_id in combine:
        res.error(where, "combine_with cites itself")
    groups = obj.get("split_groups")
    if not isinstance(groups, list):
        res.error(where, "split_groups must be a list")
    else:
        members: List[str] = []
        normalized: List[List[str]] = []
        groups_ok = True
        for index, group in enumerate(groups):
            if not isinstance(group, list) or not group or not all(
                isinstance(entry, str) and entry for entry in group
            ):
                res.error(
                    where,
                    f"split_groups[{index}] must be a non-empty list of work item ids",
                )
                groups_ok = False
                continue
            if group != sorted(group):
                res.error(where, f"split_groups[{index}] must be sorted")
                groups_ok = False
            members.extend(group)
            normalized.append(list(group))
        if groups_ok and groups:
            if len(groups) < 2:
                res.error(where, "split_groups must contain at least two groups")
            if len(members) != len(set(members)):
                res.error(where, "split_groups must not repeat a work item")
            if normalized != sorted(normalized, key=tuple):
                res.error(where, "split_groups must be sorted by group")
    if combine and groups:
        res.error(
            where,
            "a candidate cannot participate in both combine and split treatment",
        )
    _require_str(res, where, obj, "rationale", allow_empty=True)
    _require_str(res, where, obj, "model")
    _require_str(res, where, obj, "prompt_version")
    _require_id(res, where, obj, "sol_call_id", "sc1")
    fingerprint = _require_str(res, where, obj, "request_fingerprint")
    if fingerprint is not None and not _SHA256_RE.match(fingerprint):
        res.error(where, "request_fingerprint must be a sha256 hex digest")
    _require_str(res, where, obj, "provider")


def _validate_sol_call(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    _check_unknown(res, where, obj, _SOL_CALL_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "call_id", "sc1")
    candidate_ids = _require_sorted_unique_allow_empty(
        res, where, obj, "package_candidate_ids"
    )
    if candidate_ids is not None and not candidate_ids:
        res.error(where, "package_candidate_ids must be non-empty")
    fingerprint = _require_str(res, where, obj, "request_fingerprint")
    if fingerprint is not None and not _SHA256_RE.match(fingerprint):
        res.error(where, "request_fingerprint must be a sha256 hex digest")
    _require_str(res, where, obj, "provider")
    _require_str(res, where, obj, "model")
    _require_str(res, where, obj, "prompt_version")
    _require_choice(res, where, obj, "usage_source", TERRA_USAGE_SOURCES)
    for name in _SOL_TOKEN_FIELDS:
        _require_int(res, where, obj, name, minimum=0)
    cached = obj.get("cached_input_tokens")
    inputs = obj.get("input_tokens")
    if _is_int(cached) and _is_int(inputs) and cached > inputs:
        res.error(
            where,
            f"cached_input_tokens ({cached}) must not exceed input_tokens ({inputs})",
        )


def _validate_sol_listing_usage(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _SOL_LISTING_USAGE_FIELDS)
    _check_schema_version(res, where, obj)
    _require_int(res, where, obj, "call_count", minimum=0)
    for name in _SOL_TOKEN_FIELDS:
        _require_int(res, where, obj, name, minimum=0)


def _validate_package_review_snapshots(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _SNAPSHOT_FIELDS)
    _check_schema_version(res, where, obj)
    for name in sorted(_SNAPSHOT_FIELDS - {"schema_version"}):
        value = _require_str(res, where, obj, name)
        if value is not None and not _SHA256_RE.match(value):
            res.error(where, f"{name} must be a sha256 hex digest")


def _validate_ledger_entry(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    _check_unknown(res, where, obj, _LEDGER_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "entry_id", "cl1")
    _require_id(res, where, obj, "work_item_id", "wk1")
    representation = _require_choice(res, where, obj, "representation", LEDGER_REPRESENTATIONS)
    package_id = obj.get("package_id")
    if representation == "absorbed_by_package":
        if not isinstance(package_id, str) or not _ID_PATTERNS["pk1"].match(package_id):
            res.error(where, "package_id (pk1_<16 hex>) is required when absorbed_by_package")
    elif representation is not None and package_id is not None:
        res.error(where, "package_id must be null unless representation is absorbed_by_package")
    reason = _require_choice(res, where, obj, "reason_code", LEDGER_REASON_CODES)
    if representation == "absorbed_by_package":
        if reason is not None and reason != "absorbed_by_approved_package":
            res.error(
                where,
                "absorbed_by_package entries must carry reason_code "
                f"'absorbed_by_approved_package', got {reason!r}",
            )
    elif representation == "standalone":
        if reason == "absorbed_by_approved_package":
            res.error(
                where,
                "standalone entries cannot carry the absorption reason code",
            )
    elif representation is not None:
        # inspection and no_action conditions never become work items this
        # schema, so no reason codes exist for their representations yet.
        res.error(
            where,
            f"representation {representation!r} is reserved — the ledger "
            "holds active billable work only in this schema",
        )
    _require_money_pair(res, where, obj)
    if representation in ("absorbed_by_package", "no_action"):
        if obj.get("low") != 0 or obj.get("high") != 0:
            res.error(
                where,
                f"{representation} entries must carry 0/0 dollars — package "
                "dollars travel with the package exactly once",
            )


def _validate_package_application(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _APPLICATION_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "application_id", "pa1")
    _require_id(res, where, obj, "package_candidate_id", "pk1")
    _require_id(res, where, obj, "decision_id", "pd1")
    status = _require_choice(res, where, obj, "status", APPLICATION_STATUSES)
    reason = _require_choice(res, where, obj, "reason_code", APPLICATION_REASON_CODES)
    absorbed = _require_sorted_unique_allow_empty(
        res, where, obj, "absorbed_work_item_ids"
    )
    unabsorbed = _require_sorted_unique_allow_empty(
        res, where, obj, "unabsorbed_child_work_item_ids"
    )
    if absorbed and unabsorbed and set(absorbed) & set(unabsorbed):
        res.error(
            where,
            "absorbed_work_item_ids and unabsorbed_child_work_item_ids must "
            "be disjoint",
        )
    group_id = obj.get("combine_group_id")
    if group_id is not None and (
        not isinstance(group_id, str) or not _ID_PATTERNS["cg1"].match(group_id)
    ):
        res.error(where, "combine_group_id must be null or cg1_<16 hex>")
    for name in ("effective_low", "effective_high"):
        _require_int(res, where, obj, name, minimum=0)
    low, high = obj.get("effective_low"), obj.get("effective_high")
    if _is_int(low) and _is_int(high) and low > high:
        res.error(where, f"effective_low ({low}) must not exceed effective_high ({high})")
    _STATUS_REASONS = {
        "applied": {"approved_absorbs_children"},
        "display_only": {"display_only_aggregate"},
        "not_applied": {"decision_rejected", "decision_uncertain",
                        "split_recommended", "no_owned_children"},
    }
    if status is not None and reason is not None and reason not in _STATUS_REASONS[status]:
        res.error(
            where,
            f"reason_code {reason!r} is not valid for status {status!r}",
        )
    if status == "applied":
        if absorbed is not None and not absorbed:
            res.error(where, "applied packages must absorb at least one child")
    elif status is not None:
        if absorbed:
            res.error(where, f"{status} applications must absorb nothing")
        if (low, high) != (0, 0):
            res.error(
                where,
                f"{status} applications must carry 0/0 effective dollars",
            )


def _validate_reconciliation_audit(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _AUDIT_FIELDS)
    _check_schema_version(res, where, obj)
    for name in _AUDIT_LIST_FIELDS:
        value = obj.get(name)
        if not isinstance(value, list):
            res.error(where, f"{name} must be a list")
        elif value:
            res.error(
                where,
                f"{name} must be empty — a complete result cannot carry "
                f"reconciliation defects, got {len(value)}",
            )


def _validate_observability(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _OBSERVABILITY_FIELDS)
    _check_schema_version(res, where, obj)
    timings = obj.get("phase_timings_ms")
    if not isinstance(timings, dict):
        res.error(where, "phase_timings_ms must be an object")
    else:
        if set(timings) != set(OBSERVABILITY_PHASES):
            res.error(
                where,
                f"phase_timings_ms keys must be exactly "
                f"{sorted(OBSERVABILITY_PHASES)}, got {sorted(timings)}",
            )
        else:
            for phase in OBSERVABILITY_PHASES:
                value = timings[phase]
                if not _is_int(value) or value < 0:
                    res.error(
                        where,
                        f"phase_timings_ms[{phase!r}] must be an int >= 0",
                    )
            if all(_is_int(timings[p]) for p in OBSERVABILITY_PHASES):
                expected = sum(
                    timings[p] for p in OBSERVABILITY_PHASES if p != "total"
                )
                if timings["total"] != expected:
                    res.error(
                        where,
                        f"phase_timings_ms['total'] must be exactly the sum "
                        f"of the other phases ({expected}), got "
                        f"{timings['total']}",
                    )
    for name in ("terra_total_tokens", "sol_total_tokens", "combined_total_tokens"):
        _require_int(res, where, obj, name, minimum=0)
    terra = obj.get("terra_total_tokens")
    sol = obj.get("sol_total_tokens")
    combined = obj.get("combined_total_tokens")
    if _is_int(terra) and _is_int(sol) and _is_int(combined) and combined != terra + sol:
        res.error(
            where,
            f"combined_total_tokens must be exactly {terra + sol}, got {combined}",
        )
    funnel = obj.get("funnel")
    if not isinstance(funnel, dict):
        res.error(where, "funnel must be an object")
    else:
        if set(funnel) != set(FUNNEL_KEYS):
            res.error(
                where,
                f"funnel keys must be exactly {sorted(FUNNEL_KEYS)}, "
                f"got {sorted(funnel)}",
            )
        for key, value in funnel.items():
            if not _is_int(value) or value < 0:
                res.error(where, f"funnel[{key!r}] must be an int >= 0")


def _validate_totals(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        res.error(where, "totals must be an object")
        return
    _check_unknown(res, where, obj, _TOTALS_FIELDS)
    _check_schema_version(res, where, obj)
    if obj.get("currency") != "USD":
        res.error(where, f"currency must be 'USD', got {obj.get('currency')!r}")
    for lane in ("standalone", "packaged", "inspection", "headline"):
        entry = obj.get(lane)
        if not isinstance(entry, dict) or set(entry) != {"low", "high"}:
            res.error(where, f"{lane} must be an object with exactly low/high")
            continue
        _require_money_pair(res, f"{where}.{lane}", entry)


# ── provenance ───────────────────────────────────────────────────────────────

def _looks_relative(path: str) -> bool:
    if path.startswith(("/", "\\")) or "\\" in path:
        return False
    if re.match(r"^[A-Za-z]:", path):
        return False
    return ".." not in path.split("/")


def _validate_provenance(
    res: ValidationResult, where: str, obj: Any, *, state: Optional[str]
) -> None:
    if not isinstance(obj, dict):
        res.error(where, "provenance must be an object")
        return
    _check_unknown(res, where, obj, _PROVENANCE_FIELDS)
    _check_schema_version(res, where, obj)
    _require_choice(res, where, obj, "architecture_mode", ARCHITECTURE_MODES)
    if obj.get("contracts_schema_version") != CONTRACTS_SCHEMA_VERSION:
        res.error(
            where,
            f"contracts_schema_version must be {CONTRACTS_SCHEMA_VERSION}, "
            f"got {obj.get('contracts_schema_version')!r}",
        )
    if obj.get("projection_version") != PROJECTION_VERSION:
        res.error(
            where,
            f"projection_version must be {PROJECTION_VERSION!r}, "
            f"got {obj.get('projection_version')!r}",
        )

    # The catalog identity group is nullable only in the failed state — a
    # failure before projection build cannot know the catalog it never loaded.
    nullable = state == "failed"
    pinned = (
        ("catalog_version", REQUIRED_CATALOG_VERSION),
        ("catalog_ontology_version", REQUIRED_CATALOG_ONTOLOGY),
        ("kind_ontology_selector", REQUIRED_KIND_ONTOLOGY_SELECTOR),
    )
    for name, expected in pinned:
        value = _require_str(res, where, obj, name, nullable=nullable)
        if value is not None and value != expected:
            res.error(where, f"{name} must be {expected!r}, got {value!r}")
    for name in ("catalog_sha256", "projection_fingerprint"):
        value = _require_str(res, where, obj, name, nullable=nullable)
        if value is not None and not _SHA256_RE.match(value):
            res.error(where, f"{name} must be 64 lowercase hex characters")

    policies = obj.get("policy_versions")
    if not isinstance(policies, dict):
        res.error(where, "policy_versions must be an object")
    else:
        if set(policies) != POLICY_VERSION_KEYS:
            res.error(
                where,
                f"policy_versions keys must be exactly {sorted(POLICY_VERSION_KEYS)}, "
                f"got {sorted(policies)}",
            )
        for key, value in policies.items():
            if not isinstance(value, str) or not value:
                res.error(where, f"policy_versions[{key!r}] must be a non-empty string")

    _require_str(res, where, obj, "property_key")
    _require_str(res, where, obj, "source_run_id")
    source_artifact = _require_str(res, where, obj, "source_artifact")
    if source_artifact is not None and not _looks_relative(source_artifact):
        res.error(
            where,
            "source_artifact must be a relative identity (forward slashes, no "
            f"drive letters, no traversal), got {source_artifact!r}",
        )
    created_at = _require_str(res, where, obj, "created_at")
    if created_at is not None and not ("T" in created_at and created_at.endswith("Z")):
        res.error(where, f"created_at must be an ISO-8601 UTC timestamp, got {created_at!r}")


# ── envelope and complete result ─────────────────────────────────────────────

def validate_envelope(payload: Any) -> ValidationResult:
    """Validate a RenovationEstimateEnvelope dict in any state."""
    res = ValidationResult()
    where = "envelope"
    if not isinstance(payload, dict):
        res.error(where, f"must be an object, got {type(payload).__name__}")
        return res
    _check_unknown(res, where, payload, _ENVELOPE_FIELDS)
    if payload.get("schema_version") != ENVELOPE_SCHEMA_VERSION:
        res.error(
            where,
            f"schema_version must be {ENVELOPE_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}",
        )
    estimate_id = _require_id(res, where, payload, "estimate_id", "rea1")
    state = _require_choice(res, where, payload, "state", ENVELOPE_STATES)
    _validate_provenance(res, "envelope.provenance", payload.get("provenance"), state=state)

    reason = payload.get("reason")
    error_detail = payload.get("error_detail")
    result = payload.get("result")
    if state == "scaffold":
        if reason != SCAFFOLD_REASON:
            res.error(where, f"scaffold reason must be {SCAFFOLD_REASON!r}, got {reason!r}")
        if result is not None:
            res.error(where, "scaffold result must be null")
    elif state == "failed":
        if not isinstance(reason, str) or not reason:
            res.error(where, "failed reason must be a non-empty failure category")
        if result is not None:
            res.error(where, "failed result must be null")
    elif state == "condition_review_complete":
        if reason is not None:
            res.error(where, "condition_review_complete reason must be null")
        if not isinstance(result, dict):
            res.error(where, "condition_review_complete result must be an object")
        else:
            res.extend(
                validate_condition_review_result(result, estimate_id=estimate_id or "")
            )
    elif state == "standalone_estimate_complete":
        if reason is not None:
            res.error(where, "standalone_estimate_complete reason must be null")
        if not isinstance(result, dict):
            res.error(where, "standalone_estimate_complete result must be an object")
        else:
            res.extend(
                validate_standalone_estimate_result(result, estimate_id=estimate_id or "")
            )
    elif state == "package_review_complete":
        if reason is not None:
            res.error(where, "package_review_complete reason must be null")
        if not isinstance(result, dict):
            res.error(where, "package_review_complete result must be an object")
        else:
            res.extend(
                validate_package_review_result(result, estimate_id=estimate_id or "")
            )
    elif state == "complete":
        if reason is not None:
            res.error(where, "complete reason must be null")
        if not isinstance(result, dict):
            res.error(where, "complete result must be an object")
        else:
            res.extend(
                validate_complete_result(result, estimate_id=estimate_id or "")
            )
    if state != "failed" and error_detail is not None:
        res.error(where, "error_detail must be null unless state is failed")
    elif state == "failed" and error_detail is not None and not isinstance(error_detail, str):
        res.error(where, "error_detail must be a string or null")
    return res


def _index_by(
    res: ValidationResult, where: str, records: List[Dict[str, Any]], key: str
) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        record_id = record.get(key)
        if not isinstance(record_id, str) or not record_id:
            continue  # the record validator already reported it
        if record_id in indexed:
            res.error(where, f"duplicate id {record_id!r}")
        else:
            indexed[record_id] = record
    return indexed


def _validate_condition_lattice(
    res: ValidationResult, result: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Shared condition-layer invariants for both result validators: at most
    one condition per (catalog condition, physical estimate unit), exactly one
    evidence/review/disposition per condition with no orphans, and each
    disposition citing its condition's actual review and evidence records.
    Returns (conditions by id, {lane: {condition_id: record}})."""
    conditions = _index_by(res, "result.observed_conditions", result["observed_conditions"], "condition_id")
    evidence = _index_by(res, "result.evidence_facts", result["evidence_facts"], "evidence_id")
    reviews = _index_by(res, "result.condition_reviews", result["condition_reviews"], "review_id")
    dispositions = _index_by(res, "result.condition_dispositions", result["condition_dispositions"], "disposition_id")

    seen_units: Dict[Tuple[str, str], str] = {}
    for condition_id, condition in conditions.items():
        unit_key = (condition["catalog_item_id"], condition["estimate_unit_id"])
        if unit_key in seen_units:
            res.error(
                "result.observed_conditions",
                f"{condition_id} duplicates catalog condition "
                f"{unit_key[0]!r} in estimate unit {unit_key[1]!r} "
                f"(already held by {seen_units[unit_key]})",
            )
        else:
            seen_units[unit_key] = condition_id

    lattices = (
        ("evidence_facts", evidence),
        ("condition_reviews", reviews),
        ("condition_dispositions", dispositions),
    )
    by_condition: Dict[str, Dict[str, Dict[str, Any]]] = {name: {} for name, _ in lattices}
    for name, records in lattices:
        for record in records.values():
            condition_id = record["condition_id"]
            if condition_id not in conditions:
                res.error(f"result.{name}", f"references unknown condition {condition_id!r}")
                continue
            if condition_id in by_condition[name]:
                res.error(f"result.{name}", f"condition {condition_id!r} has more than one record")
            else:
                by_condition[name][condition_id] = record
    for condition_id in conditions:
        for name, _ in lattices:
            if condition_id not in by_condition[name]:
                message = (
                    "has no terminal disposition"
                    if name == "condition_dispositions"
                    else f"has no {name} record"
                )
                res.error("result", f"condition {condition_id!r} {message}")
    for condition_id, disposition in by_condition["condition_dispositions"].items():
        review = by_condition["condition_reviews"].get(condition_id)
        if review is not None and disposition["review_id"] != review["review_id"]:
            res.error(
                "result.condition_dispositions",
                f"condition {condition_id!r} disposition cites review "
                f"{disposition['review_id']!r} but the condition's review is "
                f"{review['review_id']!r}",
            )
        evidence_record = by_condition["evidence_facts"].get(condition_id)
        if evidence_record is not None and disposition["evidence_id"] != evidence_record["evidence_id"]:
            res.error(
                "result.condition_dispositions",
                f"condition {condition_id!r} disposition cites evidence "
                f"{disposition['evidence_id']!r} but the condition's evidence is "
                f"{evidence_record['evidence_id']!r}",
            )
    return conditions, by_condition


def validate_complete_result(result: Any, *, estimate_id: str) -> ValidationResult:
    """Enforce the Session 5 complete-result invariants over a result dict.

    The result is the frozen Session 4 package-review result plus the
    reconciliation layer. The package-review subset is re-validated verbatim
    by the frozen Session 4 gate (which itself re-runs the Session 2/3 gates
    and the snapshot fingerprints), the application/ledger/totals invariants
    are enforced independently on the recorded artifacts, and finally the
    whole reconciliation is recomputed through the one shared deterministic
    computation and compared for exact equality — validity IS the
    deterministic reconciliation. The packaged totals lane sums applied
    applications' effective ranges, never stored candidate floors: a
    candidate that lost a shared child to a higher-priority package must not
    bill that child's dollars through its floor.
    """
    res = ValidationResult()
    where = "result"
    if not isinstance(result, dict):
        res.error(where, "must be an object")
        return res
    _check_unknown(res, where, result, _RESULT_KEYS)
    package_review_subset = {
        key: result[key] for key in _PACKAGE_REVIEW_RESULT_KEYS if key in result
    }
    res.extend(
        validate_package_review_result(package_review_subset, estimate_id=estimate_id)
    )
    for key in ("package_applications", "coverage_ledger"):
        if not isinstance(result.get(key), list):
            res.error(where, f"{key} must be a list")
            return res
    for key, validator in (
        ("package_applications", _validate_package_application),
        ("coverage_ledger", _validate_ledger_entry),
    ):
        for index, record in enumerate(result[key]):
            record_where = f"result.{key}[{index}]"
            if not isinstance(record, dict):
                res.error(record_where, "must be an object")
                continue
            validator(res, record_where, record)
    audit = result.get("reconciliation_audit")
    if not isinstance(audit, dict):
        res.error(where, "reconciliation_audit must be an object")
    else:
        _validate_reconciliation_audit(res, "result.reconciliation_audit", audit)
    observability = result.get("observability")
    if not isinstance(observability, dict):
        res.error(where, "observability must be an object")
    else:
        _validate_observability(res, "result.observability", observability)
    _validate_totals(res, "result.totals", result.get("totals"))
    if not res.ok:
        # Cross-record invariants assume individually valid records; reporting
        # them over broken records would bury the root cause in noise.
        return res

    candidates = _index_by(
        res, "result.package_candidates", result["package_candidates"],
        "package_candidate_id",
    )
    decisions_by_candidate = {
        decision["package_candidate_id"]: decision
        for decision in result["package_decisions"]
    }
    applications = _index_by(
        res, "result.package_applications", result["package_applications"],
        "application_id",
    )
    ledger = _index_by(
        res, "result.coverage_ledger", result["coverage_ledger"], "entry_id"
    )
    active_work = {
        item["work_item_id"]: item
        for item in result["work_items"]
        if item["status"] == "active"
    }

    # Candidate <-> application bijection with decision provenance.
    app_by_candidate: Dict[str, Dict[str, Any]] = {}
    for app_id, app in sorted(applications.items()):
        app_where = "result.package_applications"
        candidate_id = app["package_candidate_id"]
        if candidate_id not in candidates:
            res.error(
                app_where,
                f"{app_id} references unknown candidate {candidate_id!r}",
            )
            continue
        if candidate_id in app_by_candidate:
            res.error(
                app_where,
                f"candidate {candidate_id!r} has more than one application",
            )
            continue
        app_by_candidate[candidate_id] = app
        decision = decisions_by_candidate.get(candidate_id)
        if decision is None or decision["decision_id"] != app["decision_id"]:
            res.error(
                app_where,
                f"{app_id} decision_id does not reference candidate "
                f"{candidate_id!r}'s decision",
            )
    for candidate_id in sorted(set(candidates) - set(app_by_candidate)):
        res.error(
            "result.package_applications",
            f"candidate {candidate_id!r} has no application — every decision "
            "must be deterministically applied",
        )
    if not res.ok:
        return res

    # Application status, eligibility, child partition, and effective ranges,
    # independent of the recompute below.
    app_where = "result.package_applications"
    child_owner: Dict[str, str] = {}
    for candidate_id, app in sorted(app_by_candidate.items()):
        candidate = candidates[candidate_id]
        decision = decisions_by_candidate[candidate_id]
        eligible = (
            decision["decision"] == "approve"
            and not candidate["display_only"]
            and not decision["split_groups"]
        )
        if candidate["display_only"] != (app["status"] == "display_only"):
            res.error(
                app_where,
                f"candidate {candidate_id!r} status must be display_only iff "
                "the candidate is the display-only aggregate",
            )
        if app["status"] == "applied":
            if not eligible:
                res.error(
                    app_where,
                    f"candidate {candidate_id!r} is applied without an "
                    "approval basis (approve, non-display, no split)",
                )
            union = sorted(
                list(app["absorbed_work_item_ids"])
                + list(app["unabsorbed_child_work_item_ids"])
            )
            if union != sorted(candidate["child_work_item_ids"]):
                res.error(
                    app_where,
                    f"candidate {candidate_id!r} absorbed + unabsorbed must "
                    "exactly partition its children",
                )
            owned = [
                child_id for child_id in app["absorbed_work_item_ids"]
                if child_id in active_work
            ]
            for child_id in app["absorbed_work_item_ids"]:
                if child_id not in active_work:
                    res.error(
                        app_where,
                        f"candidate {candidate_id!r} absorbs {child_id!r}, "
                        "which is not an ACTIVE work item",
                    )
                elif child_id in child_owner:
                    res.error(
                        app_where,
                        f"work item {child_id!r} is absorbed by both "
                        f"{child_owner[child_id]!r} and {candidate_id!r}",
                    )
                else:
                    child_owner[child_id] = candidate_id
            expected_range = (
                max(
                    candidate["unfloored_low"],
                    sum(active_work[c]["low"] for c in owned),
                ),
                max(
                    candidate["unfloored_high"],
                    sum(active_work[c]["high"] for c in owned),
                ),
            )
            if (app["effective_low"], app["effective_high"]) != expected_range:
                res.error(
                    app_where,
                    f"candidate {candidate_id!r} effective range must be "
                    "exactly max(unfloored tier spec, owned child sum) "
                    f"{expected_range[0]}/{expected_range[1]}, got "
                    f"{app['effective_low']}/{app['effective_high']}",
                )
        else:
            if sorted(app["unabsorbed_child_work_item_ids"]) != sorted(
                candidate["child_work_item_ids"]
            ):
                res.error(
                    app_where,
                    f"candidate {candidate_id!r} is not applied, so every "
                    "child must be listed unabsorbed",
                )

    # Coverage: exactly one ledger entry per active work item, consistent
    # with the application ownership above.
    entries_by_work: Dict[str, Dict[str, Any]] = {}
    for entry_id, entry in sorted(ledger.items()):
        entry_where = "result.coverage_ledger"
        work_id = entry["work_item_id"]
        if work_id not in active_work:
            res.error(
                entry_where,
                f"{entry_id} covers unknown or suppressed work item {work_id!r}",
            )
            continue
        if work_id in entries_by_work:
            res.error(
                entry_where,
                f"work item {work_id!r} has more than one ledger entry",
            )
            continue
        entries_by_work[work_id] = entry
        if entry["representation"] == "absorbed_by_package":
            owner_id = entry["package_id"]
            owner_app = app_by_candidate.get(owner_id)
            if (
                owner_app is None
                or owner_app["status"] != "applied"
                or work_id not in owner_app["absorbed_work_item_ids"]
            ):
                res.error(
                    entry_where,
                    f"{entry_id} is absorbed by {owner_id!r}, whose "
                    "application does not bill it",
                )
        else:  # standalone (other representations rejected per record)
            work = active_work[work_id]
            if (entry["low"], entry["high"]) != (work["low"], work["high"]):
                res.error(
                    entry_where,
                    f"{entry_id} standalone entry must equal work item "
                    f"{work_id!r}'s exact allowance "
                    f"{work['low']}/{work['high']}, got "
                    f"{entry['low']}/{entry['high']}",
                )
            if work_id in child_owner:
                res.error(
                    entry_where,
                    f"work item {work_id!r} is standalone in the ledger but "
                    f"absorbed by {child_owner[work_id]!r}",
                )
    for work_id in sorted(set(active_work) - set(entries_by_work)):
        res.error(
            "result.coverage_ledger",
            f"active work item {work_id!r} has no ledger entry — accepted work "
            "cannot disappear",
        )

    # Rejected/uncertain/split packages preserve standalone children.
    for candidate_id in sorted(candidates):
        decision = decisions_by_candidate[candidate_id]
        candidate = candidates[candidate_id]
        eligible = (
            decision["decision"] == "approve"
            and not candidate["display_only"]
            and not decision["split_groups"]
        )
        if eligible:
            continue
        for child_id in candidate["child_work_item_ids"]:
            if child_id in child_owner:
                continue  # legitimately absorbed by a different applied package
            entry = entries_by_work.get(child_id)
            if entry is not None and entry["representation"] != "standalone":
                res.error(
                    "result.coverage_ledger",
                    f"work item {child_id!r} belongs to non-billable package "
                    f"{candidate_id!r} and must remain standalone, got "
                    f"{entry['representation']!r}",
                )

    if not res.ok:
        return res

    # Exact totals arithmetic, both endpoints: standalone/inspection from
    # ledger entries, packaged from applied effective ranges.
    def _lane_sum(representation: str) -> Tuple[int, int]:
        low = sum(e["low"] for e in entries_by_work.values() if e["representation"] == representation)
        high = sum(e["high"] for e in entries_by_work.values() if e["representation"] == representation)
        return low, high

    applied_apps = [
        app for app in app_by_candidate.values() if app["status"] == "applied"
    ]
    totals = result["totals"]
    expected = {
        "standalone": _lane_sum("standalone"),
        "packaged": (
            sum(app["effective_low"] for app in applied_apps),
            sum(app["effective_high"] for app in applied_apps),
        ),
        "inspection": _lane_sum("inspection"),
    }
    expected["headline"] = (
        sum(low for low, _ in expected.values()),
        sum(high for _, high in expected.values()),
    )
    for lane, (low, high) in expected.items():
        actual = totals[lane]
        if (actual["low"], actual["high"]) != (low, high):
            res.error(
                "result.totals",
                f"{lane} must be exactly {low}/{high} from the ledger, got "
                f"{actual['low']}/{actual['high']}",
            )

    # Full deterministic recompute: ownership priority, reason codes, combine
    # groups, and IDs must all match the single shared computation exactly.
    # Function-level import — reconciliation imports this module at load, so
    # the shared definition stays acyclic.
    from tools.renovation_architecture.reconciliation import (
        build_observability,
        compute_reconciliation,
    )

    recomputed = compute_reconciliation(result, estimate_id=estimate_id)
    for section in ("package_applications", "coverage_ledger", "totals"):
        if result[section] != recomputed[section]:
            res.error(
                f"result.{section}",
                "does not match the deterministic reconciliation recompute — "
                "the complete result must be exactly the policy's output",
            )
    recomputed_observability = build_observability(
        result,
        phase_timings_ms=result["observability"]["phase_timings_ms"],
    )
    if result["observability"] != recomputed_observability:
        res.error(
            "result.observability",
            "token totals or funnel counts do not reconcile with the result "
            "sections",
        )
    return res


# ── Session 2 condition-review result ────────────────────────────────────────

def _check_evidence_views(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    """duplicate_groups must be the sorted disjoint closure of the exact and
    near families; representatives and distinct_view_count must follow from
    it. Runs only after the record validators passed, so groups are known to
    be lists of photo_keys members."""
    photo_keys = sorted(set(obj["photo_keys"]))
    exact = obj["exact_duplicate_groups"]
    near = obj["near_duplicate_groups"]
    for name, groups in (
        ("exact_duplicate_groups", exact), ("near_duplicate_groups", near)
    ):
        seen: set = set()
        for group in groups:
            if len(group) < 2:
                res.error(where, f"{name} entries must have at least two members")
            for key in group:
                if key in seen:
                    res.error(
                        where, f"{name} must be disjoint — {key!r} appears in two groups"
                    )
                seen.add(key)

    parent = {key: key for key in photo_keys}

    def _find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for group in list(exact) + list(near):
        for other in group[1:]:
            parent[_find(other)] = _find(group[0])
    classes: Dict[str, List[str]] = {}
    for key in photo_keys:
        classes.setdefault(_find(key), []).append(key)
    view_classes = sorted(sorted(members) for members in classes.values())
    expected_merged = [members for members in view_classes if len(members) > 1]
    if [list(group) for group in obj["duplicate_groups"]] != expected_merged:
        res.error(
            where,
            "duplicate_groups must be the sorted disjoint closure of "
            "exact_duplicate_groups and near_duplicate_groups",
        )
    view_count = obj["distinct_view_count"]
    if view_count is None:
        res.error(where, "distinct_view_count is required in a condition-review result")
    elif view_count != len(view_classes):
        res.error(
            where,
            f"distinct_view_count ({view_count}) must equal the number of "
            f"distinct view classes ({len(view_classes)})",
        )
    expected_representatives = sorted(members[0] for members in view_classes)
    if list(obj["representative_photo_keys"]) != expected_representatives:
        res.error(
            where,
            "representative_photo_keys must be the sorted lexicographically-"
            "first member of each view class",
        )


def _check_terra_usage(
    res: ValidationResult,
    result: Dict[str, Any],
    conditions: Dict[str, Dict[str, Any]],
    by_condition: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Exact call -> estimate unit -> listing reconciliation."""
    calls = _index_by(res, "result.terra_calls", result["terra_calls"], "call_id")

    units: Dict[str, List[str]] = {}
    for condition_id, condition in conditions.items():
        units.setdefault(condition["estimate_unit_id"], []).append(condition_id)

    calls_by_unit: Dict[str, Dict[str, Any]] = {}
    for call_id, call in calls.items():
        unit_id = call["estimate_unit_id"]
        if unit_id not in units:
            res.error(
                "result.terra_calls",
                f"{call_id} references estimate unit {unit_id!r} with no conditions",
            )
            continue
        if unit_id in calls_by_unit:
            res.error(
                "result.terra_calls",
                f"estimate unit {unit_id!r} has more than one Terra call",
            )
            continue
        calls_by_unit[unit_id] = call
        if sorted(call["condition_ids"]) != sorted(units[unit_id]):
            res.error(
                "result.terra_calls",
                f"{call_id} condition_ids must be exactly the conditions of "
                f"estimate unit {unit_id!r}",
            )
    for unit_id in sorted(set(units) - set(calls_by_unit)):
        res.error("result.terra_calls", f"estimate unit {unit_id!r} has no Terra call")

    for condition_id, review in sorted(by_condition["condition_reviews"].items()):
        condition = conditions[condition_id]
        call = calls.get(review["terra_call_id"])
        if call is None:
            res.error(
                "result.condition_reviews",
                f"condition {condition_id!r} review cites unknown terra call "
                f"{review['terra_call_id']!r}",
            )
            continue
        if call["estimate_unit_id"] != condition["estimate_unit_id"]:
            res.error(
                "result.condition_reviews",
                f"condition {condition_id!r} review cites a call for estimate "
                f"unit {call['estimate_unit_id']!r}, not the condition's "
                f"{condition['estimate_unit_id']!r}",
            )
        elif condition_id not in call["condition_ids"]:
            res.error(
                "result.condition_reviews",
                f"condition {condition_id!r} is not among its cited call's condition_ids",
            )
        for name in ("request_fingerprint", "provider", "model", "prompt_version"):
            if review[name] != call[name]:
                res.error(
                    "result.condition_reviews",
                    f"condition {condition_id!r} review {name} does not match "
                    "its terra call",
                )

    usage_by_unit: Dict[str, Dict[str, Any]] = {}
    for index, usage in enumerate(result["terra_unit_usage"]):
        usage_where = f"result.terra_unit_usage[{index}]"
        unit_id = usage["estimate_unit_id"]
        if unit_id not in units:
            res.error(usage_where, f"references estimate unit {unit_id!r} with no conditions")
            continue
        if unit_id in usage_by_unit:
            res.error(usage_where, f"estimate unit {unit_id!r} has more than one usage record")
            continue
        usage_by_unit[unit_id] = usage
        call = calls_by_unit.get(unit_id)
        if call is None:
            continue  # the missing-call error already covers this unit
        if list(usage["call_ids"]) != [call["call_id"]]:
            res.error(usage_where, "call_ids must list exactly the unit's Terra calls")
        for name in _TERRA_TOKEN_FIELDS:
            if usage[name] != call[name]:
                res.error(
                    usage_where,
                    f"{name} must equal the sum over the unit's calls "
                    f"({call[name]}), got {usage[name]}",
                )
    for unit_id in sorted(set(calls_by_unit) - set(usage_by_unit)):
        res.error(
            "result.terra_unit_usage", f"estimate unit {unit_id!r} has no usage record"
        )

    listing = result["terra_listing_usage"]
    listing_where = "result.terra_listing_usage"
    if listing["call_count"] != len(calls):
        res.error(
            listing_where,
            f"call_count must be {len(calls)}, got {listing['call_count']}",
        )
    for name in _TERRA_TOKEN_FIELDS:
        expected = sum(usage[name] for usage in usage_by_unit.values())
        if listing[name] != expected:
            res.error(
                listing_where,
                f"{name} must be exactly {expected} from the unit rollups, "
                f"got {listing[name]}",
            )


def validate_condition_review_result(result: Any, *, estimate_id: str) -> ValidationResult:
    """Enforce the Session 2 condition-review invariants over a result dict.

    The review layer is finished; work items, packages, and totals do not
    exist yet, so their keys are rejected. Empty condition lists are valid —
    a listing with no product-lane issues still completes review."""
    res = ValidationResult()
    where = "result"
    if not isinstance(result, dict):
        res.error(where, "must be an object")
        return res
    _check_unknown(res, where, result, _REVIEW_RESULT_KEYS)
    for key in sorted(_REVIEW_RESULT_KEYS - {"terra_listing_usage"}):
        if not isinstance(result.get(key), list):
            res.error(where, f"{key} must be a list")
            return res
    validators = (
        ("observed_conditions", _validate_condition),
        ("evidence_facts", _validate_evidence),
        ("condition_reviews", _validate_review),
        ("condition_dispositions", _validate_disposition),
        ("terra_calls", _validate_terra_call),
        ("terra_unit_usage", _validate_terra_unit_usage),
    )
    for key, validator in validators:
        for index, record in enumerate(result[key]):
            record_where = f"result.{key}[{index}]"
            if not isinstance(record, dict):
                res.error(record_where, "must be an object")
                continue
            validator(res, record_where, record)
    listing = result.get("terra_listing_usage")
    if not isinstance(listing, dict):
        res.error(where, "terra_listing_usage must be an object")
    else:
        _validate_terra_listing_usage(res, "result.terra_listing_usage", listing)
    if not res.ok:
        # Cross-record invariants assume individually valid records; reporting
        # them over broken records would bury the root cause in noise.
        return res

    conditions, by_condition = _validate_condition_lattice(res, result)

    for condition_id, evidence_record in sorted(by_condition["evidence_facts"].items()):
        _check_evidence_views(
            res, f"result.evidence_facts ({evidence_record['evidence_id']})",
            evidence_record,
        )

    # The recorded disposition must be exactly what the deterministic policy
    # computes from the verdict and the objective evidence.
    for condition_id, disposition in sorted(by_condition["condition_dispositions"].items()):
        review = by_condition["condition_reviews"].get(condition_id)
        evidence_record = by_condition["evidence_facts"].get(condition_id)
        if review is None or evidence_record is None:
            continue  # the lattice already reported the gap
        try:
            expected = decide_disposition(
                review["verdict"],
                disposition["terminal_route"],
                evidence_record["distinct_view_count"],
                evidence_record["min_photo_evidence_required"],
            )
        except ValueError:
            continue  # the record validators already rejected the vocabulary
        actual = (disposition["disposition"], disposition["reason_code"])
        if actual != expected:
            res.error(
                "result.condition_dispositions",
                f"condition {condition_id!r} records {actual!r} but "
                f"{CONDITION_DISPOSITION_POLICY_VERSION} computes {expected!r}",
            )

    _check_terra_usage(res, result, conditions, by_condition)
    return res


# ── Session 3 standalone-estimate result ─────────────────────────────────────

def _expected_unit_count(
    work: Dict[str, Any], conditions: Dict[str, Dict[str, Any]]
) -> int:
    """Deterministic unit_count for a non-merged (source) work item: the
    explicit opening-instance count for per_opening, one everywhere else
    (collapse policies bill one synthetic unit; room-like and per_scope
    sources bill exactly the one physical unit their conditions resolved
    to)."""
    if work["unit_policy"] != "per_opening":
        return 1
    hints: set = set()
    for condition_id in work["condition_ids"]:
        condition = conditions.get(condition_id)
        if condition is not None:
            hints.update(condition.get("opening_instance_hints") or [])
    return max(1, len(hints))


def validate_standalone_estimate_result(
    result: Any, *, estimate_id: str
) -> ValidationResult:
    """Enforce the Session 3 standalone-estimate invariants over a result dict.

    The result is the frozen Session 2 condition-review result plus the
    deterministic work layer: the review subset is re-validated verbatim by
    the frozen Session 2 gate, then the work layer's lineage, dedup, and
    exact-total invariants are enforced on top. Packages, Sol, and the
    coverage ledger do not exist yet, so their keys are rejected."""
    res = ValidationResult()
    where = "result"
    if not isinstance(result, dict):
        res.error(where, "must be an object")
        return res
    _check_unknown(res, where, result, _STANDALONE_RESULT_KEYS)
    for key in ("work_items", "work_dedup_collisions"):
        if not isinstance(result.get(key), list):
            res.error(where, f"{key} must be a list")
            return res

    review_subset = {
        key: result[key] for key in _REVIEW_RESULT_KEYS if key in result
    }
    res.extend(validate_condition_review_result(review_subset, estimate_id=estimate_id))

    for key, validator in (
        ("work_items", _validate_work_item),
        ("work_dedup_collisions", _validate_work_dedup_collision),
    ):
        for index, record in enumerate(result[key]):
            record_where = f"result.{key}[{index}]"
            if not isinstance(record, dict):
                res.error(record_where, "must be an object")
                continue
            validator(res, record_where, record)
    standalone = result.get("standalone_estimate")
    if not isinstance(standalone, dict):
        res.error(where, "standalone_estimate must be an object")
    else:
        _validate_standalone_estimate(res, "result.standalone_estimate", standalone)
    if not res.ok:
        # Cross-record invariants assume individually valid records; reporting
        # them over broken records would bury the root cause in noise.
        return res

    # The review gate passed, so rebuilding its indexes cannot produce errors.
    conditions, by_condition = _validate_condition_lattice(ValidationResult(), result)
    work_items = _index_by(res, "result.work_items", result["work_items"], "work_item_id")
    collisions = _index_by(
        res, "result.work_dedup_collisions", result["work_dedup_collisions"],
        "collision_id",
    )

    accepted = {
        condition_id
        for condition_id, disposition in by_condition["condition_dispositions"].items()
        if disposition["disposition"] == "accepted_for_work"
    }
    merged_actives = {
        collision["active_work_item_id"] for collision in collisions.values()
    }

    # Per-item lineage, billable-unit consistency, and unit counts.
    active_by_condition: Dict[str, List[str]] = {}
    group_owner: Dict[Tuple[str, str, str, str], str] = {}
    for work_id, work in sorted(work_items.items()):
        item_where = "result.work_items"
        lineage_ok = True
        cited: List[Dict[str, Any]] = []
        for condition_id in work["condition_ids"]:
            condition = conditions.get(condition_id)
            if condition is None:
                res.error(item_where, f"{work_id} references unknown condition {condition_id!r}")
                lineage_ok = False
            elif condition_id not in accepted:
                res.error(
                    item_where,
                    f"{work_id} references condition {condition_id!r} whose "
                    "disposition is not accepted_for_work — non-accepted "
                    "conditions create no work",
                )
                lineage_ok = False
            else:
                cited.append(condition)
        if lineage_ok:
            expected_items = sorted({c["catalog_item_id"] for c in cited})
            if list(work["catalog_item_ids"]) != expected_items:
                res.error(
                    item_where,
                    f"{work_id} catalog_item_ids must be exactly {expected_items}, "
                    f"got {list(work['catalog_item_ids'])}",
                )
            expected_units = sorted({c["estimate_unit_id"] for c in cited})
            if list(work["source_estimate_unit_ids"]) != expected_units:
                res.error(
                    item_where,
                    f"{work_id} source_estimate_unit_ids must be exactly "
                    f"{expected_units}, got {list(work['source_estimate_unit_ids'])}",
                )
            expected_ambiguous = any(c["identity_ambiguous"] for c in cited)
            if work["identity_ambiguous"] != expected_ambiguous:
                res.error(
                    item_where,
                    f"{work_id} identity_ambiguous must be {expected_ambiguous} "
                    "(any source condition ambiguous)",
                )
        unit_policy = work["unit_policy"]
        collapse_unit = _COLLAPSE_BILLABLE_UNITS.get(unit_policy)
        if collapse_unit is not None:
            if work["billable_unit_id"] != collapse_unit:
                res.error(
                    item_where,
                    f"{work_id} billable_unit_id must be {collapse_unit!r} for "
                    f"{unit_policy}, got {work['billable_unit_id']!r}",
                )
        elif list(work["source_estimate_unit_ids"]) != [work["billable_unit_id"]]:
            res.error(
                item_where,
                f"{work_id} must bill exactly its one source estimate unit "
                f"({work['billable_unit_id']!r}), got "
                f"{list(work['source_estimate_unit_ids'])}",
            )
        if work_id not in merged_actives and lineage_ok:
            expected_count = _expected_unit_count(work, conditions)
            if work["unit_count"] != expected_count:
                res.error(
                    item_where,
                    f"{work_id} unit_count must be {expected_count}, "
                    f"got {work['unit_count']}",
                )
        if work["status"] == "active":
            for condition_id in work["condition_ids"]:
                active_by_condition.setdefault(condition_id, []).append(work_id)
            group_key = tuple(work[name] for name in _WORK_DEDUP_KEY_FIELDS)
            if group_key in group_owner:
                res.error(
                    item_where,
                    f"{work_id} and {group_owner[group_key]} share the active "
                    f"work group {group_key!r} — active groups must be unique",
                )
            else:
                group_owner[group_key] = work_id

    # Every accepted condition belongs to exactly one active work item.
    for condition_id in sorted(accepted):
        owners = active_by_condition.get(condition_id, [])
        if not owners:
            res.error(
                "result",
                f"accepted condition {condition_id!r} is referenced by no "
                "active work item — accepted scope cannot vanish",
            )
        elif len(owners) > 1:
            res.error(
                "result",
                f"accepted condition {condition_id!r} is referenced by "
                f"multiple active work items {sorted(owners)}",
            )

    # Collision audits: complete lineage, exact max envelope, one collision
    # per suppressed item.
    suppressed_owner: Dict[str, str] = {}
    audited_actives: Dict[str, str] = {}
    for collision_id, collision in sorted(collisions.items()):
        coll_where = "result.work_dedup_collisions"
        active = work_items.get(collision["active_work_item_id"])
        if active is None:
            res.error(
                coll_where,
                f"{collision_id} cites unknown active work item "
                f"{collision['active_work_item_id']!r}",
            )
            continue
        active_id = collision["active_work_item_id"]
        if active_id in audited_actives:
            res.error(
                coll_where,
                f"{collision_id} and {audited_actives[active_id]} both audit "
                f"active work item {active_id!r}",
            )
            continue
        audited_actives[active_id] = collision_id
        if active["status"] != "active":
            res.error(coll_where, f"{collision_id} active work item {active_id!r} is not active")
        for name in _WORK_DEDUP_KEY_FIELDS:
            if collision[name] != active[name]:
                res.error(
                    coll_where,
                    f"{collision_id} {name} {collision[name]!r} does not match "
                    f"its active work item's {active[name]!r}",
                )
        sources: List[Dict[str, Any]] = []
        sources_ok = True
        for suppressed_id in collision["suppressed_work_item_ids"]:
            source = work_items.get(suppressed_id)
            if source is None:
                res.error(coll_where, f"{collision_id} cites unknown suppressed item {suppressed_id!r}")
                sources_ok = False
                continue
            if source["status"] != "suppressed":
                res.error(
                    coll_where,
                    f"{collision_id} source {suppressed_id!r} is not suppressed",
                )
                sources_ok = False
            if suppressed_id in suppressed_owner:
                res.error(
                    coll_where,
                    f"suppressed item {suppressed_id!r} belongs to both "
                    f"{suppressed_owner[suppressed_id]!r} and {collision_id!r}",
                )
                sources_ok = False
            else:
                suppressed_owner[suppressed_id] = collision_id
            for name in _WORK_DEDUP_KEY_FIELDS:
                if source.get(name) != collision[name]:
                    res.error(
                        coll_where,
                        f"{collision_id} source {suppressed_id!r} {name} does "
                        "not match the collision key",
                    )
                    sources_ok = False
            sources.append(source)
        if not sources_ok or not sources:
            continue
        expected_low = max(source["low"] for source in sources)
        expected_high = max(source["high"] for source in sources)
        if (active["low"], active["high"]) != (expected_low, expected_high):
            res.error(
                coll_where,
                f"{collision_id} active range must be the exact max envelope "
                f"{expected_low}/{expected_high} over its sources, got "
                f"{active['low']}/{active['high']} — colliding work is never summed",
            )
        for name, label in (
            ("condition_ids", "condition"),
            ("catalog_item_ids", "catalog item"),
            ("source_estimate_unit_ids", "estimate unit"),
        ):
            expected_union = sorted({
                value for source in sources for value in source[name]
            })
            if list(active[name]) != expected_union:
                res.error(
                    coll_where,
                    f"{collision_id} active {name} must be the exact {label} "
                    f"union of its sources",
                )
        expected_sources_union = sorted({
            value for source in sources for value in source["action_sources"]
        })
        if list(active["action_sources"]) != expected_sources_union:
            res.error(coll_where, f"{collision_id} active action_sources must be the union of its sources'")
        expected_modes_union = sorted({
            value for source in sources for value in source["pricing_modes"]
        })
        if list(active["pricing_modes"]) != expected_modes_union:
            res.error(coll_where, f"{collision_id} active pricing_modes must be the union of its sources'")
        if active["identity_ambiguous"] != any(s["identity_ambiguous"] for s in sources):
            res.error(coll_where, f"{collision_id} active identity_ambiguous must be any-of its sources'")
        expected_count = max(source["unit_count"] for source in sources)
        if active["unit_count"] != expected_count:
            res.error(
                coll_where,
                f"{collision_id} active unit_count must be the max over its "
                f"sources ({expected_count}), got {active['unit_count']}",
            )
        source_scopes = {source["estimate_scope"] for source in sources}
        expected_scope = next(
            scope for scope in ESTIMATE_SCOPE_MERGE_PRIORITY if scope in source_scopes
        )
        if active["estimate_scope"] != expected_scope:
            res.error(
                coll_where,
                f"{collision_id} active estimate_scope must be the most-required "
                f"source scope {expected_scope!r}, got {active['estimate_scope']!r}",
            )
        else:
            winning = min(
                (source for source in sources if source["estimate_scope"] == expected_scope),
                key=lambda source: source["work_item_id"],
            )
            if active["estimate_scope_reason"] != winning["estimate_scope_reason"]:
                res.error(
                    coll_where,
                    f"{collision_id} active estimate_scope_reason must come from "
                    "the lexicographically-first winning-scope source",
                )

    # Every suppressed item belongs to exactly one collision audit.
    for work_id, work in sorted(work_items.items()):
        if work["status"] == "suppressed" and work_id not in suppressed_owner:
            res.error(
                "result.work_items",
                f"suppressed item {work_id!r} belongs to no collision audit — "
                "suppression without an audit trail is a lost record",
            )

    if not res.ok:
        return res

    # Exact totals: per-scope sums over active items, both endpoints, and the
    # headline as their componentwise sum.
    expected_totals = {scope: [0, 0] for scope in ESTIMATE_SCOPES}
    for work in work_items.values():
        if work["status"] != "active":
            continue
        expected_totals[work["estimate_scope"]][0] += work["low"]
        expected_totals[work["estimate_scope"]][1] += work["high"]
    standalone = result["standalone_estimate"]
    totals = standalone["totals_by_estimate_scope"]
    for scope in sorted(ESTIMATE_SCOPES):
        expected_low, expected_high = expected_totals[scope]
        actual = totals[scope]
        if (actual["low"], actual["high"]) != (expected_low, expected_high):
            res.error(
                "result.standalone_estimate",
                f"totals_by_estimate_scope[{scope!r}] must be exactly "
                f"{expected_low}/{expected_high} from active work, got "
                f"{actual['low']}/{actual['high']}",
            )
    headline = standalone["headline"]
    expected_headline = (
        sum(low for low, _ in expected_totals.values()),
        sum(high for _, high in expected_totals.values()),
    )
    if (headline["low"], headline["high"]) != expected_headline:
        res.error(
            "result.standalone_estimate",
            f"headline must be exactly {expected_headline[0]}/"
            f"{expected_headline[1]} from the scope buckets, got "
            f"{headline['low']}/{headline['high']}",
        )
    return res


# ── Session 4 package-review result ──────────────────────────────────────────

def package_review_snapshot_hashes(result: Mapping[str, Any]) -> Dict[str, str]:
    """The immutability fingerprints over the layers Sol must never change.

    One definition serves both the producer (tools/renovation_architecture/
    sol_review.py) and the recompute check in the Session 4 gate below."""
    return {
        "condition_snapshot_sha256": sha256_canonical(
            {key: result[key] for key in _CONDITION_SNAPSHOT_KEYS}
        ),
        "work_snapshot_sha256": sha256_canonical(
            {key: result[key] for key in _WORK_SNAPSHOT_KEYS}
        ),
        "candidate_snapshot_sha256": sha256_canonical(
            result["package_candidates"]
        ),
    }


def validate_package_review_result(
    result: Any, *, estimate_id: str
) -> ValidationResult:
    """Enforce the Session 4 package-review invariants over a result dict.

    The result is the frozen Session 3 standalone result plus the package
    layer: the standalone subset is re-validated verbatim by the frozen
    Session 3 gate, then candidate lineage/economics, the closed Sol decision
    semantics, call/usage telemetry reconciliation, and the snapshot
    immutability fingerprints are enforced on top. Decision application, the
    coverage ledger, and totals do not exist yet, so their keys are rejected.

    Candidates may share a child work item (a collapse-policy work item can
    contribute to two room candidates); absorbing it at most once is the
    Session 5 ledger gate's invariant, not a candidate-shape constraint."""
    res = ValidationResult()
    where = "result"
    if not isinstance(result, dict):
        res.error(where, "must be an object")
        return res
    _check_unknown(res, where, result, _PACKAGE_REVIEW_RESULT_KEYS)
    for key in ("package_candidates", "package_decisions", "sol_calls"):
        if not isinstance(result.get(key), list):
            res.error(where, f"{key} must be a list")
            return res

    standalone_subset = {
        key: result[key] for key in _STANDALONE_RESULT_KEYS if key in result
    }
    res.extend(
        validate_standalone_estimate_result(standalone_subset, estimate_id=estimate_id)
    )

    for key, validator in (
        ("package_candidates", _validate_package_candidate),
        ("package_decisions", _validate_package_decision),
        ("sol_calls", _validate_sol_call),
    ):
        for index, record in enumerate(result[key]):
            record_where = f"result.{key}[{index}]"
            if not isinstance(record, dict):
                res.error(record_where, "must be an object")
                continue
            validator(res, record_where, record)
    listing_usage = result.get("sol_listing_usage")
    if not isinstance(listing_usage, dict):
        res.error(where, "sol_listing_usage must be an object")
    else:
        _validate_sol_listing_usage(res, "result.sol_listing_usage", listing_usage)
    snapshots = result.get("package_review_snapshots")
    if not isinstance(snapshots, dict):
        res.error(where, "package_review_snapshots must be an object")
    else:
        _validate_package_review_snapshots(
            res, "result.package_review_snapshots", snapshots
        )
    if not res.ok:
        # Cross-record invariants assume individually valid records; reporting
        # them over broken records would bury the root cause in noise.
        return res

    candidates = _index_by(
        res, "result.package_candidates", result["package_candidates"],
        "package_candidate_id",
    )
    decisions = _index_by(
        res, "result.package_decisions", result["package_decisions"], "decision_id"
    )
    calls = _index_by(res, "result.sol_calls", result["sol_calls"], "call_id")
    active_work = {
        item["work_item_id"]: item
        for item in result["work_items"]
        if item["status"] == "active"
    }
    display_only_ids = {
        candidate_id for candidate_id, candidate in candidates.items()
        if candidate["display_only"]
    }
    if len(display_only_ids) > 1:
        res.error(
            "result.package_candidates",
            "at most one display-only aggregate may exist, got "
            f"{sorted(display_only_ids)}",
        )

    # Candidate lineage and floored economics.
    for candidate_id, candidate in sorted(candidates.items()):
        cand_where = "result.package_candidates"
        if candidate["display_only"]:
            contributors: List[Dict[str, Any]] = []
            contributors_ok = True
            for other_id in candidate["contributing_candidate_ids"]:
                other = candidates.get(other_id)
                if other is None:
                    res.error(
                        cand_where,
                        f"{candidate_id} cites unknown contributing candidate "
                        f"{other_id!r}",
                    )
                    contributors_ok = False
                elif other["display_only"] or other["package_category"] != "turnover":
                    res.error(
                        cand_where,
                        f"{candidate_id} contributor {other_id!r} must be a "
                        "room turnover candidate",
                    )
                    contributors_ok = False
                else:
                    contributors.append(other)
            if not contributors_ok:
                continue
            if len({other["room"] for other in contributors}) < 2:
                res.error(
                    cand_where,
                    f"{candidate_id} needs turnover contributors from at "
                    "least two distinct rooms",
                )
            expected = (
                sum(other["low"] for other in contributors),
                sum(other["high"] for other in contributors),
            )
            if (candidate["low"], candidate["high"]) != expected:
                res.error(
                    cand_where,
                    f"{candidate_id} range must be exactly the sum of its "
                    f"contributors' floored ranges {expected[0]}/{expected[1]}, "
                    f"got {candidate['low']}/{candidate['high']}",
                )
            if (candidate["unfloored_low"], candidate["unfloored_high"]) != expected:
                res.error(
                    cand_where,
                    f"{candidate_id} unfloored range must equal its floored "
                    "range — the aggregate has no tier spec",
                )
            continue
        child_low = 0
        child_high = 0
        lineage_ok = True
        for child_id in candidate["child_work_item_ids"]:
            child = active_work.get(child_id)
            if child is None:
                res.error(
                    cand_where,
                    f"{candidate_id} child {child_id!r} is not an ACTIVE work "
                    "item — packages may only group active accepted work",
                )
                lineage_ok = False
            else:
                child_low += child["low"]
                child_high += child["high"]
        if lineage_ok:
            expected_range = (
                max(candidate["unfloored_low"], child_low),
                max(candidate["unfloored_high"], child_high),
            )
            if (candidate["low"], candidate["high"]) != expected_range:
                res.error(
                    cand_where,
                    f"{candidate_id} floored range must be exactly "
                    "max(tier spec, children standalone sum) "
                    f"{expected_range[0]}/{expected_range[1]}, got "
                    f"{candidate['low']}/{candidate['high']}",
                )

    # Candidate <-> decision bijection: exactly one decision per candidate.
    decided: Dict[str, Dict[str, Any]] = {}
    for decision_id, decision in sorted(decisions.items()):
        dec_where = "result.package_decisions"
        candidate_id = decision["package_candidate_id"]
        if candidate_id not in candidates:
            res.error(
                dec_where,
                f"{decision_id} references unknown candidate {candidate_id!r}",
            )
            continue
        if candidate_id in decided:
            res.error(
                dec_where,
                f"candidate {candidate_id!r} has more than one decision",
            )
            continue
        decided[candidate_id] = decision
    for candidate_id in sorted(set(candidates) - set(decided)):
        res.error(
            "result.package_decisions",
            f"candidate {candidate_id!r} has no decision — Sol must return "
            "exactly one decision per supplied candidate",
        )
    if not res.ok:
        return res

    # Combine/split semantics and decision provenance. Combine references are
    # undirected edges among approved candidates; a candidate participating
    # in any combine edge cannot also split; display-only aggregates
    # participate in neither treatment.
    combine_participants: set = set()
    for candidate_id, decision in decided.items():
        if decision["combine_with"]:
            combine_participants.add(candidate_id)
            combine_participants.update(decision["combine_with"])
    for candidate_id, decision in sorted(decided.items()):
        dec_where = "result.package_decisions"
        edges = decision["combine_with"]
        groups = decision["split_groups"]
        if edges:
            if candidate_id in display_only_ids:
                res.error(
                    dec_where,
                    f"display-only candidate {candidate_id!r} may not combine",
                )
            elif decision["decision"] != "approve":
                res.error(
                    dec_where,
                    f"candidate {candidate_id!r} proposes combine_with but is "
                    f"{decision['decision']!r} — combine edges may only join "
                    "approved candidates",
                )
            for other_id in edges:
                if other_id not in candidates:
                    res.error(
                        dec_where,
                        f"candidate {candidate_id!r} combine_with cites "
                        f"unknown candidate {other_id!r}",
                    )
                elif other_id in display_only_ids:
                    res.error(
                        dec_where,
                        f"candidate {candidate_id!r} combine_with cites the "
                        f"display-only candidate {other_id!r}",
                    )
                elif decided[other_id]["decision"] != "approve":
                    res.error(
                        dec_where,
                        f"candidate {candidate_id!r} combine_with cites "
                        f"{other_id!r}, whose decision is "
                        f"{decided[other_id]['decision']!r} — combine edges "
                        "may only join approved candidates",
                    )
        if groups:
            if candidate_id in display_only_ids:
                res.error(
                    dec_where,
                    f"display-only candidate {candidate_id!r} may not split",
                )
            elif candidate_id in combine_participants:
                res.error(
                    dec_where,
                    f"candidate {candidate_id!r} participates in both combine "
                    "and split treatment",
                )
            else:
                proposed = sorted(
                    child for group in groups for child in group
                )
                if proposed != list(candidates[candidate_id]["child_work_item_ids"]):
                    res.error(
                        dec_where,
                        f"candidate {candidate_id!r} split_groups must exactly "
                        "partition its children — no additions, omissions, or "
                        "overlap",
                    )
        call = calls.get(decision["sol_call_id"])
        if call is None:
            res.error(
                dec_where,
                f"decision for {candidate_id!r} cites unknown Sol call "
                f"{decision['sol_call_id']!r}",
            )
        else:
            if decision["request_fingerprint"] != call["request_fingerprint"]:
                res.error(
                    dec_where,
                    f"decision for {candidate_id!r} fingerprint does not match "
                    "its Sol call",
                )
            if (
                decision["model"] != call["model"]
                or decision["prompt_version"] != call["prompt_version"]
            ):
                res.error(
                    dec_where,
                    f"decision for {candidate_id!r} model/prompt provenance "
                    "does not match its Sol call",
                )

    # Sol call coverage and telemetry reconciliation. This session makes at
    # most one listing-level call; zero candidates make zero calls.
    if candidates and len(calls) != 1:
        res.error(
            "result.sol_calls",
            "exactly one listing-level Sol call is required when candidates "
            f"exist, got {len(calls)}",
        )
    if not candidates and calls:
        res.error("result.sol_calls", "no Sol call may exist without candidates")
    for call_id, call in sorted(calls.items()):
        if list(call["package_candidate_ids"]) != sorted(candidates):
            res.error(
                "result.sol_calls",
                f"{call_id} package_candidate_ids must be exactly the "
                "supplied candidates",
            )
    listing = result["sol_listing_usage"]
    if listing["call_count"] != len(calls):
        res.error(
            "result.sol_listing_usage",
            f"call_count must be {len(calls)}, got {listing['call_count']}",
        )
    for name in _SOL_TOKEN_FIELDS:
        expected_tokens = sum(call[name] for call in calls.values())
        if listing[name] != expected_tokens:
            res.error(
                "result.sol_listing_usage",
                f"{name} must be exactly {expected_tokens} from the Sol "
                f"calls, got {listing[name]}",
            )

    # Snapshot immutability: condition, work, and candidate truth must be
    # byte-identical through Sol review.
    expected_hashes = package_review_snapshot_hashes(result)
    recorded = result["package_review_snapshots"]
    for name in sorted(expected_hashes):
        if recorded[name] != expected_hashes[name]:
            res.error(
                "result.package_review_snapshots",
                f"{name} does not match the recomputed section hash — Sol "
                "review may not change condition, work, or candidate truth",
            )
    return res
