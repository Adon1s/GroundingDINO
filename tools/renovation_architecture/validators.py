"""Strict validators for the renovation architecture contracts.

Hand-rolled, stdlib only — matching tools/catalog_validation.py and
tools/benchmarking/schemas.py. Validators operate on the plain-dict form of
the contracts (the JSON boundary), collect ALL errors rather than stopping at
the first, and format every message as "where: message".

Unknown-field rejection is the enforcement mechanism for the layer
boundaries: a ConditionReview or PackageDecision carrying work, package,
price, quantity, or confidence fields cannot validate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from tools.renovation_architecture.contracts import (
    ACTION_SOURCES,
    ARCHITECTURE_MODES,
    CONDITION_DISPOSITION_POLICY_VERSION,
    CONTRACTS_SCHEMA_VERSION,
    DISPOSITION_REASON_CODES,
    DISPOSITIONS,
    ENVELOPE_SCHEMA_VERSION,
    ENVELOPE_STATES,
    EVIDENCE_DEDUP_POLICY_VERSION,
    LEDGER_REPRESENTATIONS,
    OBSERVATION_KINDS_V2,
    PACKAGE_DECISIONS,
    POLICY_VERSION_KEYS,
    PRICING_MODES,
    PROJECTION_VERSION,
    REQUIRED_CATALOG_ONTOLOGY,
    REQUIRED_CATALOG_VERSION,
    REQUIRED_KIND_ONTOLOGY_SELECTOR,
    REVIEW_RATIONALE_MAX_CHARS,
    REVIEW_VERDICTS,
    SCAFFOLD_REASON,
    TERMINAL_ROUTES,
    TERRA_USAGE_SOURCES,
    UNIT_POLICIES,
    UNIT_RESOLUTION_SOURCES,
    WORK_ITEM_STATUSES,
)
from tools.renovation_architecture.disposition import decide_disposition

_ID_PATTERNS = {
    prefix: re.compile(rf"^{prefix}_[0-9a-f]{{16}}$")
    for prefix in (
        "rea1", "oc1", "ev1", "cr1", "cd1", "tc1", "wk1", "pk1", "pd1", "cl1"
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
     "source_scope_keys", "unit_resolution_source", "unit_resolution_reason"}
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
    {"work_item_id", "schema_version", "condition_ids", "catalog_item_id",
     "estimate_unit_id", "action_code", "action_source", "trade_bucket",
     "unit_policy", "unit_count", "pricing_mode", "low", "high", "status",
     "reason_code"}
)
_PACKAGE_CANDIDATE_FIELDS = frozenset(
    {"package_candidate_id", "schema_version", "package_type", "room_key",
     "child_work_item_ids", "proposed_treatment", "low", "high"}
)
_PACKAGE_DECISION_FIELDS = frozenset(
    {"decision_id", "schema_version", "package_candidate_id", "decision",
     "combine_with", "split_groups", "rationale", "model", "prompt_version"}
)
_LEDGER_FIELDS = frozenset(
    {"entry_id", "schema_version", "work_item_id", "representation",
     "package_id", "low", "high"}
)
_TOTALS_FIELDS = frozenset(
    {"schema_version", "currency", "standalone", "packaged", "inspection",
     "headline"}
)
_RESULT_KEYS = frozenset(
    {"observed_conditions", "evidence_facts", "condition_reviews",
     "condition_dispositions", "work_items", "package_candidates",
     "package_decisions", "coverage_ledger", "totals"}
)
# The Session 2 intermediate result: the review layer is finished, the work
# and package layers do not exist yet. terra_listing_usage is one object,
# every other key is a list.
_REVIEW_RESULT_KEYS = frozenset(
    {"observed_conditions", "evidence_facts", "condition_reviews",
     "condition_dispositions", "terra_calls", "terra_unit_usage",
     "terra_listing_usage"}
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


def _validate_work_item(res: ValidationResult, where: str, obj: Dict[str, Any]) -> None:
    _check_unknown(res, where, obj, _WORK_ITEM_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "work_item_id", "wk1")
    condition_ids = _require_str_list(res, where, obj, "condition_ids", allow_empty=False)
    if condition_ids is not None and len(condition_ids) != len(set(condition_ids)):
        res.error(where, "condition_ids must be unique")
    _require_str(res, where, obj, "catalog_item_id")
    _require_str(res, where, obj, "estimate_unit_id")
    _require_str(res, where, obj, "action_code")
    _require_choice(res, where, obj, "action_source", ACTION_SOURCES)
    _require_str(res, where, obj, "trade_bucket")
    _require_choice(res, where, obj, "unit_policy", UNIT_POLICIES)
    _require_int(res, where, obj, "unit_count", minimum=1)
    _require_choice(res, where, obj, "pricing_mode", PRICING_MODES)
    _require_money_pair(res, where, obj)
    status = _require_choice(res, where, obj, "status", WORK_ITEM_STATUSES)
    reason_code = obj.get("reason_code")
    if status == "suppressed":
        if not isinstance(reason_code, str) or not reason_code:
            res.error(where, "reason_code is required when status is 'suppressed'")
    elif status == "active" and reason_code is not None:
        res.error(where, "reason_code must be null unless status is 'suppressed'")


def _validate_package_candidate(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _PACKAGE_CANDIDATE_FIELDS)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "package_candidate_id", "pk1")
    _require_str(res, where, obj, "package_type")
    _require_str(res, where, obj, "room_key")
    children = _require_str_list(res, where, obj, "child_work_item_ids", allow_empty=False)
    if children is not None and len(children) != len(set(children)):
        res.error(where, "child_work_item_ids must be unique")
    _require_str(res, where, obj, "proposed_treatment")
    _require_money_pair(res, where, obj)


def _validate_package_decision(
    res: ValidationResult, where: str, obj: Dict[str, Any]
) -> None:
    _check_unknown(res, where, obj, _PACKAGE_DECISION_FIELDS, boundary=True)
    _check_schema_version(res, where, obj)
    _require_id(res, where, obj, "decision_id", "pd1")
    _require_id(res, where, obj, "package_candidate_id", "pk1")
    _require_choice(res, where, obj, "decision", PACKAGE_DECISIONS)
    _require_str_list(res, where, obj, "combine_with", allow_empty=True)
    groups = obj.get("split_groups")
    if not isinstance(groups, list):
        res.error(where, "split_groups must be a list")
    else:
        for index, group in enumerate(groups):
            if not isinstance(group, list) or not group or not all(
                isinstance(entry, str) and entry for entry in group
            ):
                res.error(
                    where,
                    f"split_groups[{index}] must be a non-empty list of work item ids",
                )
    _require_str(res, where, obj, "rationale", allow_empty=True)
    _require_str(res, where, obj, "model")
    _require_str(res, where, obj, "prompt_version")


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
    _require_money_pair(res, where, obj)
    if representation in ("absorbed_by_package", "no_action"):
        if obj.get("low") != 0 or obj.get("high") != 0:
            res.error(
                where,
                f"{representation} entries must carry 0/0 dollars — package "
                "dollars travel with the package exactly once",
            )


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
    """Enforce the complete-result invariants over a result dict.

    Production use begins in Session 5; implemented and test-covered now so
    the contracts and their invariants are frozen together.
    """
    res = ValidationResult()
    where = "result"
    if not isinstance(result, dict):
        res.error(where, "must be an object")
        return res
    _check_unknown(res, where, result, _RESULT_KEYS)
    for key in sorted(_RESULT_KEYS - {"totals"}):
        if not isinstance(result.get(key), list):
            res.error(where, f"{key} must be a list")
            return res
    validators = (
        ("observed_conditions", _validate_condition),
        ("evidence_facts", _validate_evidence),
        ("condition_reviews", _validate_review),
        ("condition_dispositions", _validate_disposition),
        ("work_items", _validate_work_item),
        ("package_candidates", _validate_package_candidate),
        ("package_decisions", _validate_package_decision),
        ("coverage_ledger", _validate_ledger_entry),
    )
    for key, validator in validators:
        for index, record in enumerate(result[key]):
            record_where = f"result.{key}[{index}]"
            if not isinstance(record, dict):
                res.error(record_where, "must be an object")
                continue
            validator(res, record_where, record)
    _validate_totals(res, "result.totals", result.get("totals"))
    if not res.ok:
        # Cross-record invariants assume individually valid records; reporting
        # them over broken records would bury the root cause in noise.
        return res

    conditions, by_condition = _validate_condition_lattice(res, result)
    work_items = _index_by(res, "result.work_items", result["work_items"], "work_item_id")
    candidates = _index_by(res, "result.package_candidates", result["package_candidates"], "package_candidate_id")
    decisions = _index_by(res, "result.package_decisions", result["package_decisions"], "decision_id")
    ledger = _index_by(res, "result.coverage_ledger", result["coverage_ledger"], "entry_id")

    # Accepted-work lineage.
    accepted = {
        condition_id
        for condition_id, disposition in by_condition["condition_dispositions"].items()
        if disposition["disposition"] == "accepted_for_work"
    }
    referenced_conditions: set = set()
    for work_id, work in work_items.items():
        for condition_id in work["condition_ids"]:
            referenced_conditions.add(condition_id)
            condition = conditions.get(condition_id)
            if condition is None:
                res.error("result.work_items", f"{work_id} references unknown condition {condition_id!r}")
            elif condition_id not in accepted:
                res.error(
                    "result.work_items",
                    f"{work_id} references condition {condition_id!r} whose "
                    "disposition is not accepted_for_work",
                )
            elif condition["catalog_item_id"] != work["catalog_item_id"]:
                res.error(
                    "result.work_items",
                    f"{work_id} catalog_item_id {work['catalog_item_id']!r} does "
                    f"not match condition {condition_id!r} "
                    f"({condition['catalog_item_id']!r})",
                )
    for condition_id in sorted(accepted - referenced_conditions):
        res.error(
            "result",
            f"accepted condition {condition_id!r} is referenced by no work item "
            "— accepted scope cannot vanish",
        )

    # Package integrity.
    active_work = {
        work_id for work_id, work in work_items.items() if work["status"] == "active"
    }
    for candidate_id, candidate in candidates.items():
        for child_id in candidate["child_work_item_ids"]:
            if child_id not in work_items:
                res.error(
                    "result.package_candidates",
                    f"{candidate_id} child {child_id!r} is not a known work item",
                )
            elif child_id not in active_work:
                res.error(
                    "result.package_candidates",
                    f"{candidate_id} child {child_id!r} is suppressed — packages "
                    "may only group active accepted work",
                )
    decided: Dict[str, Dict[str, Any]] = {}
    for decision_id, decision in decisions.items():
        candidate_id = decision["package_candidate_id"]
        candidate = candidates.get(candidate_id)
        if candidate is None:
            res.error(
                "result.package_decisions",
                f"{decision_id} references unknown candidate {candidate_id!r}",
            )
            continue
        if candidate_id in decided:
            res.error(
                "result.package_decisions",
                f"candidate {candidate_id!r} has more than one decision",
            )
        else:
            decided[candidate_id] = decision
        for other_id in decision["combine_with"]:
            if other_id == candidate_id:
                res.error("result.package_decisions", f"{decision_id} combine_with cites itself")
            elif other_id not in candidates:
                res.error(
                    "result.package_decisions",
                    f"{decision_id} combine_with cites unknown candidate {other_id!r}",
                )
        if decision["split_groups"]:
            proposed = [child for group in decision["split_groups"] for child in group]
            if sorted(proposed) != sorted(candidate["child_work_item_ids"]):
                res.error(
                    "result.package_decisions",
                    f"{decision_id} split_groups must exactly partition the "
                    "candidate's children — no additions, omissions, or overlap",
                )

    # Duplicate absorption: approved packages own disjoint child sets.
    approved = {
        candidate_id
        for candidate_id, decision in decided.items()
        if decision["decision"] == "approve"
    }
    child_owner: Dict[str, str] = {}
    for candidate_id in sorted(approved):
        for child_id in candidates[candidate_id]["child_work_item_ids"]:
            if child_id in child_owner:
                res.error(
                    "result.package_candidates",
                    f"work item {child_id!r} is absorbed by both "
                    f"{child_owner[child_id]!r} and {candidate_id!r}",
                )
            else:
                child_owner[child_id] = candidate_id

    # Coverage: exactly one ledger entry per active work item.
    entries_by_work: Dict[str, Dict[str, Any]] = {}
    for entry_id, entry in ledger.items():
        work_id = entry["work_item_id"]
        if work_id not in work_items:
            res.error("result.coverage_ledger", f"{entry_id} references unknown work item {work_id!r}")
            continue
        if work_id not in active_work:
            res.error(
                "result.coverage_ledger",
                f"{entry_id} covers suppressed work item {work_id!r}",
            )
            continue
        if work_id in entries_by_work:
            res.error(
                "result.coverage_ledger",
                f"work item {work_id!r} has more than one ledger entry",
            )
            continue
        entries_by_work[work_id] = entry
        if entry["representation"] == "absorbed_by_package":
            package_id = entry["package_id"]
            if package_id not in approved:
                res.error(
                    "result.coverage_ledger",
                    f"{entry_id} is absorbed by {package_id!r}, which is not an "
                    "approved package",
                )
            elif work_id not in candidates[package_id]["child_work_item_ids"]:
                res.error(
                    "result.coverage_ledger",
                    f"{entry_id} is absorbed by {package_id!r}, which does not "
                    f"list {work_id!r} as a child",
                )
    for work_id in sorted(active_work - set(entries_by_work)):
        res.error(
            "result.coverage_ledger",
            f"active work item {work_id!r} has no ledger entry — accepted work "
            "cannot disappear",
        )

    # Rejected/uncertain packages preserve standalone children.
    for candidate_id, decision in decided.items():
        if decision["decision"] == "approve":
            continue
        for child_id in candidates[candidate_id]["child_work_item_ids"]:
            if child_id in child_owner:
                continue  # legitimately absorbed by a different approved package
            entry = entries_by_work.get(child_id)
            if entry is not None and entry["representation"] != "standalone":
                res.error(
                    "result.coverage_ledger",
                    f"work item {child_id!r} belongs to non-approved package "
                    f"{candidate_id!r} and must remain standalone, got "
                    f"{entry['representation']!r}",
                )

    if not res.ok:
        return res

    # Exact totals arithmetic, both endpoints.
    def _lane_sum(representation: str) -> Tuple[int, int]:
        low = sum(e["low"] for e in entries_by_work.values() if e["representation"] == representation)
        high = sum(e["high"] for e in entries_by_work.values() if e["representation"] == representation)
        return low, high

    absorbed_packages = {
        entry["package_id"]
        for entry in entries_by_work.values()
        if entry["representation"] == "absorbed_by_package"
    }
    packaged_low = sum(candidates[pid]["low"] for pid in absorbed_packages)
    packaged_high = sum(candidates[pid]["high"] for pid in absorbed_packages)
    totals = result["totals"]
    expected = {
        "standalone": _lane_sum("standalone"),
        "packaged": (packaged_low, packaged_high),
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
