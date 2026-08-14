"""Versioned v1 contracts for the new renovation architecture.

These are the durable data boundaries of the three-layer ownership model
(observed conditions -> work items -> packages/estimates) described in
docs/IMPLEMENTATION_PLAN_renovation_scope_estimate_architecture.md. The frozen
dataclasses are the construction-side type safety; the artifact-side gate is
tools/renovation_architecture/validators.py, which operates on the plain-dict
form produced by ``to_dict()``.

Schema v2 (Session 2) delivered the Terra side of the v1 deferred set: token
telemetry, request fingerprints, prompt provenance, and the unit-resolution
audit trail on ObservedCondition. Still deferred to the session that produces
it (additive with a CONTRACTS_SCHEMA_VERSION bump): Sol token telemetry and
package-review provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

CONTRACTS_SCHEMA_VERSION = 2
ENVELOPE_SCHEMA_VERSION = 2
PROJECTION_VERSION = "renovation_catalog_projection_v1"
TERMINAL_ROUTE_POLICY_VERSION = "terminal_route_v1"
CONDITION_DISPOSITION_POLICY_VERSION = "condition_disposition_v1"
# Names the exact/near-duplicate identity rules (normalized-pixel sha256;
# 64-bit 8x8 average-hash, Hamming <= 6 AND max-channel mean-RGB delta <= 16).
# Any change to the hash or thresholds must bump this so request fingerprints
# and Terra checkpoints invalidate cleanly.
EVIDENCE_DEDUP_POLICY_VERSION = "evidence_dedup_v1"
TERRA_REVIEW_PROMPT_VERSION = "terra_condition_review_v1"
TERRA_REVIEW_REASONING_EFFORT = "medium"
REVIEW_RATIONALE_MAX_CHARS = 400

# The private debug key the shadow seam writes under photo_intel["analysis_debug"].
SHADOW_DEBUG_KEY = "renovation_architecture_shadow_v1"
SCAFFOLD_REASON = "session_1_not_implemented"

# The catalog's own ontology stamp uses hyphens; the KIND_ONTOLOGY_VERSION env
# selector value uses underscores. Provenance records both, in separate fields,
# so the two spellings can never be conflated.
REQUIRED_CATALOG_VERSION = "3.1"
REQUIRED_CATALOG_ONTOLOGY = "observation-kind-v2"
REQUIRED_KIND_ONTOLOGY_SELECTOR = "observation_kind_v2"

# scaffold stays in the vocabulary for the minimal-valid-envelope tests even
# though the runtime never emits it after Session 2; removal is Session 6
# cleanup. condition_review_complete is the Session 2 terminal state: the
# review layer is done, work/packages/totals do not exist yet.
ENVELOPE_STATES = frozenset(
    {"scaffold", "condition_review_complete", "complete", "failed"}
)
ARCHITECTURE_MODES = frozenset({"current", "shadow", "new"})
OBSERVATION_KINDS_V2 = frozenset({"defect", "degradation", "modernization"})
REVIEW_VERDICTS = frozenset({"supported", "unsupported", "cannot_assess"})
DISPOSITIONS = frozenset(
    {"accepted_for_work", "excluded", "inspection", "withheld", "no_action"}
)
DISPOSITION_REASON_CODES = frozenset(
    {"verdict_unsupported", "verdict_cannot_assess", "insufficient_distinct_views",
     "route_work", "route_inspection", "route_no_action",
     "route_excluded_generic", "route_excluded_quarantine"}
)
TERRA_USAGE_SOURCES = frozenset({"provider", "checkpoint"})
UNIT_RESOLUTION_SOURCES = frozenset({"photo_estimate_unit", "scope_room_fallback"})
PACKAGE_DECISIONS = frozenset({"approve", "reject", "uncertain"})
LEDGER_REPRESENTATIONS = frozenset(
    {"standalone", "absorbed_by_package", "inspection", "no_action"}
)
TERMINAL_ROUTES = frozenset(
    {"excluded_quarantine", "excluded_generic", "inspection", "no_action", "work"}
)
ACTION_SOURCES = frozenset({"work_item_code", "catalog_scope"})
PRICING_MODES = frozenset({"catalog_allowance", "heuristic"})
WORK_ITEM_STATUSES = frozenset({"active", "suppressed"})
# Pinned by test against renovation_estimate's Literal aliases so the two
# vocabularies can never drift.
UNIT_POLICIES = frozenset(
    {"per_scope", "per_property", "per_room", "per_opening",
     "per_kitchen", "per_bathroom", "per_system", "per_area"}
)
STRATEGIES = frozenset(
    {"repair_only", "replace_only", "repair_or_replace",
     "service_only", "inspect_only"}
)

# The complete provenance policy set. Every envelope (including failed ones)
# carries exactly these keys; sessions extend the map together with the schema
# version. POLICY_VERSION_KEYS is derived so the two can never drift.
POLICY_VERSIONS = {
    "terminal_route_policy": TERMINAL_ROUTE_POLICY_VERSION,
    "condition_disposition_policy": CONDITION_DISPOSITION_POLICY_VERSION,
    "evidence_dedup_policy": EVIDENCE_DEDUP_POLICY_VERSION,
    "terra_review_prompt": TERRA_REVIEW_PROMPT_VERSION,
}
POLICY_VERSION_KEYS = frozenset(POLICY_VERSIONS)


def _plain(value: Any) -> Any:
    """Convert a contract tree into plain JSON-safe dicts/lists/scalars."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class _Contract:
    def to_dict(self) -> Dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class MoneyRange(_Contract):
    low: int
    high: int


@dataclass(frozen=True)
class ObservedCondition(_Contract):
    """One catalog condition instance per physical estimate unit."""
    condition_id: str
    schema_version: int
    catalog_item_id: str
    catalog_kind: str
    scope_key: str
    estimate_unit_id: str
    room_surrogate_id: str
    scene_group: str
    issue_ids: Tuple[str, ...]
    identity_ambiguous: bool
    # Unit-resolution audit trail: which surrogates/scope keys contributed and
    # how the estimate unit was resolved (photo map vs scope-room fallback).
    source_room_surrogate_ids: Tuple[str, ...]
    source_scope_keys: Tuple[str, ...]
    unit_resolution_source: str
    unit_resolution_reason: str


@dataclass(frozen=True)
class EvidenceFacts(_Contract):
    """Objective photo/view/duplicate facts. No judgment, no confidence.

    duplicate_groups is the disjoint transitive closure of the exact and near
    groups; distinct_view_count (never a filename count) is what catalog
    min_photo_evidence gates."""
    evidence_id: str
    schema_version: int
    condition_id: str
    photo_keys: Tuple[str, ...]
    distinct_photo_count: int
    distinct_view_count: Optional[int]
    duplicate_groups: Tuple[Tuple[str, ...], ...]
    evidence_refs: Tuple[Mapping[str, Any], ...]
    min_photo_evidence_required: Optional[int]
    representative_photo_keys: Tuple[str, ...]
    exact_duplicate_groups: Tuple[Tuple[str, ...], ...]
    near_duplicate_groups: Tuple[Tuple[str, ...], ...]
    dedup_policy_version: str


@dataclass(frozen=True)
class ConditionReview(_Contract):
    """Terra's verdict. The closed field set IS the layer boundary: a review
    carrying work, package, price, quantity, or confidence fields cannot
    validate."""
    review_id: str
    schema_version: int
    condition_id: str
    verdict: str
    rationale: str
    model: str
    prompt_version: str
    terra_call_id: str
    request_fingerprint: str
    provider: str


@dataclass(frozen=True)
class ConditionDisposition(_Contract):
    """Deterministic policy outcome applied after the model verdict."""
    disposition_id: str
    schema_version: int
    condition_id: str
    review_id: str
    disposition: str
    reason_code: str
    policy_version: str
    evidence_id: str
    terminal_route: str


@dataclass(frozen=True)
class TerraCall(_Contract):
    """One Terra provider call (or its checkpoint republication) for one
    estimate unit. usage_source records provenance; checkpoint reuse keeps the
    original token numbers but debits zero, so the budget ledger never counts
    provider tokens twice."""
    call_id: str
    schema_version: int
    estimate_unit_id: str
    condition_ids: Tuple[str, ...]
    request_fingerprint: str
    provider: str
    model: str
    prompt_version: str
    usage_source: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    budget_debited_tokens: int


@dataclass(frozen=True)
class TerraUnitUsage(_Contract):
    """Per-estimate-unit rollup of its Terra calls."""
    schema_version: int
    estimate_unit_id: str
    call_ids: Tuple[str, ...]
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    budget_debited_tokens: int


@dataclass(frozen=True)
class TerraListingUsage(_Contract):
    """Listing-level rollup of every Terra call in the run."""
    schema_version: int
    call_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    budget_debited_tokens: int


@dataclass(frozen=True)
class WorkItem(_Contract):
    work_item_id: str
    schema_version: int
    condition_ids: Tuple[str, ...]
    catalog_item_id: str
    estimate_unit_id: str
    action_code: str
    action_source: str
    trade_bucket: str
    unit_policy: str
    unit_count: int
    pricing_mode: str
    low: int
    high: int
    status: str
    reason_code: Optional[str]


@dataclass(frozen=True)
class PackageCandidate(_Contract):
    package_candidate_id: str
    schema_version: int
    package_type: str
    room_key: str
    child_work_item_ids: Tuple[str, ...]
    proposed_treatment: str
    low: int
    high: int


@dataclass(frozen=True)
class PackageDecision(_Contract):
    """Sol's verdict on a supplied candidate. Same closed-field-set boundary
    as ConditionReview: no work truth, no prices."""
    decision_id: str
    schema_version: int
    package_candidate_id: str
    decision: str
    combine_with: Tuple[str, ...]
    split_groups: Tuple[Tuple[str, ...], ...]
    rationale: str
    model: str
    prompt_version: str


@dataclass(frozen=True)
class CoverageLedgerEntry(_Contract):
    """Exactly one final representation per active work item. Absorbed and
    no-action entries carry 0/0 dollars — package dollars travel with the
    package exactly once, which makes double counting structurally
    impossible."""
    entry_id: str
    schema_version: int
    work_item_id: str
    representation: str
    package_id: Optional[str]
    low: int
    high: int


@dataclass(frozen=True)
class EstimateTotals(_Contract):
    schema_version: int
    currency: str
    standalone: MoneyRange
    packaged: MoneyRange
    inspection: MoneyRange
    headline: MoneyRange


@dataclass(frozen=True)
class EstimateProvenance(_Contract):
    schema_version: int
    architecture_mode: str
    contracts_schema_version: int
    projection_version: str
    catalog_version: Optional[str]
    catalog_ontology_version: Optional[str]
    catalog_sha256: Optional[str]
    projection_fingerprint: Optional[str]
    kind_ontology_selector: Optional[str]
    policy_versions: Mapping[str, str]
    property_key: str
    source_run_id: str
    source_artifact: str
    created_at: str


@dataclass(frozen=True)
class RenovationEstimateEnvelope(_Contract):
    schema_version: int
    estimate_id: str
    state: str
    reason: Optional[str]
    error_detail: Optional[str]
    provenance: EstimateProvenance
    result: Optional[Mapping[str, Any]]
