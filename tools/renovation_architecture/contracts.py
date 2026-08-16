"""Versioned v1 contracts for the new renovation architecture.

These are the durable data boundaries of the three-layer ownership model
(observed conditions -> work items -> packages/estimates) described in
docs/IMPLEMENTATION_PLAN_renovation_scope_estimate_architecture.md. The frozen
dataclasses are the construction-side type safety; the artifact-side gate is
tools/renovation_architecture/validators.py, which operates on the plain-dict
form produced by ``to_dict()``.

Schema v2 (Session 2) delivered the Terra side of the v1 deferred set: token
telemetry, request fingerprints, prompt provenance, and the unit-resolution
audit trail on ObservedCondition. Schema v3 (Session 3) delivered the work
layer: merged-lineage WorkItem, WorkDedupCollision, StandaloneEstimate, and
opening instance hints on ObservedCondition. Schema v4 (Session 4) delivered
the package layer: the deterministic PackageCandidate (family/roles/strength/
tier/floors), Sol call telemetry and decision provenance, and the immutable
snapshot fingerprints that make "Sol changed nothing upstream" an executable
check. Schema v5 (Session 5) delivered reconciliation: PackageApplication
(deterministic decision application with at-most-once child ownership),
the reason-coded CoverageLedgerEntry, ReconciliationAudit (must be empty in
a complete result), and EstimateObservability (phase timings, token rollups,
funnel counts).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

CONTRACTS_SCHEMA_VERSION = 5
ENVELOPE_SCHEMA_VERSION = 5
PROJECTION_VERSION = "renovation_catalog_projection_v2"
TERMINAL_ROUTE_POLICY_VERSION = "terminal_route_v1"
CONDITION_DISPOSITION_POLICY_VERSION = "condition_disposition_v1"
# Session 3 deterministic policies. Derivation maps accepted conditions to
# priced work through the projection's work_policy; dedup is the max-envelope
# collision rule (never sum colliding work); standalone pricing composes the
# legacy costing core, the property factor (once, pre-split), and the
# sum-preserving integer split.
WORK_DERIVATION_POLICY_VERSION = "work_derivation_v1"
WORK_DEDUP_POLICY_VERSION = "work_dedup_max_envelope_v1"
STANDALONE_PRICING_POLICY_VERSION = "standalone_pricing_v1"
# Session 4 deterministic package policy: candidates come from the legacy
# inference primitives (affinities, roles, strength, tiers, escalation) fed
# with ACTIVE work lineage, then the cost floor is applied against the child
# standalone range. Sol reviews coherence only, through the closed
# PackageDecision contract.
PACKAGE_CANDIDATE_POLICY_VERSION = "package_candidates_v1"
SOL_REVIEW_PROMPT_VERSION = "sol_package_review_v1"
SOL_REVIEW_REASONING_EFFORT = "medium"
# Session 5 deterministic reconciliation policies. Application: absorption
# eligibility (approved, non-display, no split recommendation), the legacy
# absorption priority ordering, at-most-once child ownership, and effective
# ranges recomputed from actually-owned children only; combine groups are
# non-economic metadata. Reconciliation: exactly one reason-coded ledger entry
# per ACTIVE work item and totals derived from ledger ownership plus applied
# effective ranges — never from stored candidate floors.
PACKAGE_APPLICATION_POLICY_VERSION = "package_application_v1"
COVERAGE_RECONCILIATION_POLICY_VERSION = "coverage_reconciliation_v1"
# The only work-item suppression reason this session: the item lost its dedup
# group to a merged max-envelope active and is retained as an audit record.
DEDUP_SUPPRESSION_REASON = "dedup_collision"
# Names the exact/near-duplicate identity rules (normalized-pixel sha256;
# 64-bit 8x8 average-hash, Hamming <= 6 AND max-channel mean-RGB delta <= 16).
# Any change to the hash or thresholds must bump this so request fingerprints
# and Terra checkpoints invalidate cleanly.
EVIDENCE_DEDUP_POLICY_VERSION = "evidence_dedup_v1"
TERRA_REVIEW_PROMPT_VERSION = "terra_condition_review_v1"
TERRA_REVIEW_REASONING_EFFORT = "medium"
REVIEW_RATIONALE_MAX_CHARS = 400

# The versioned artifact key of the new engine. During migration the shadow
# seam writes it privately under photo_intel["analysis_debug"] (stripped from
# the slim artifact); the same name at the photo_intel root is reserved for
# the Session 6 cutover, and the publication gate enforces both placements.
SHADOW_DEBUG_KEY = "renovation_estimate_v5"
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
# standalone_estimate_complete is the Session 3 terminal state: deterministic
# work items and the package-independent standalone estimate exist; packages,
# Sol, and the coverage ledger do not. package_review_complete is the
# Session 4 terminal state: deterministic candidates and Sol decisions exist;
# decision application, the coverage ledger, and totals do not. complete is
# the Session 5+ success state: decisions applied, ledger and totals
# reconciled, audits empty, observability recorded.
ENVELOPE_STATES = frozenset(
    {"scaffold", "condition_review_complete", "standalone_estimate_complete",
     "package_review_complete", "complete", "failed"}
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
# Package vocabularies. Pinned by test against tools/rehab_packages.py's
# constants (the UNIT_POLICIES pattern; contracts stays import-light) so the
# two vocabularies can never drift. STRENGTHS holds only the emittable set:
# weak buckets are suppressed inside the legacy inference and never become
# candidates. LEVELS holds only what the new layer can emit (room candidates
# plus the property-level whole-home aggregate).
PACKAGE_TYPES = frozenset(
    {"kitchen_modernization", "kitchen_repair", "kitchen_turnover",
     "bathroom_modernization", "bathroom_repair", "bathroom_turnover",
     "bedroom_modernization", "bedroom_repair", "bedroom_turnover",
     "living_modernization", "living_repair", "living_turnover",
     "exterior_repair", "interior_paint_flooring_refresh"}
)
PACKAGE_CATEGORIES = frozenset(
    {"modernization", "repair", "turnover", "inspection_risk"}
)
PACKAGE_LEVELS = frozenset({"room", "property"})
PACKAGE_ROOMS = frozenset(
    {"kitchen", "bathroom", "bedroom", "living", "exterior", "whole_home"}
)
PACKAGE_STRENGTHS = frozenset({"strong", "moderate"})
# proposed_treatment is the deterministic emit lane: the four legacy trigger
# reasons plus the whole-home aggregate token.
PACKAGE_TREATMENTS = frozenset(
    {"package_driver", "opportunity_driver_with_corroboration",
     "opportunity_driver_with_multiphoto_corroboration",
     "multiple_package_support_same_estimate_unit",
     "whole_home_turnover_aggregate"}
)
# The whole-home display-only aggregate's fixed identity (legacy tokens).
WHOLE_HOME_PACKAGE_TYPE = "interior_paint_flooring_refresh"
WHOLE_HOME_UNIT_ID = "whole_home"
WHOLE_HOME_PRICING_PROFILE = "interior_paint_flooring_refresh"
WHOLE_HOME_PRICING_TIER = "property_turnover_aggregate"
LEDGER_REPRESENTATIONS = frozenset(
    {"standalone", "absorbed_by_package", "inspection", "no_action"}
)
# Session 5 application/ledger vocabularies. inspection/no_action ledger
# representations stay in LEDGER_REPRESENTATIONS but have no reason codes in
# this schema: inspection and no-action conditions are reason-coded
# dispositions, never synthesized work items, so the ledger holds active
# billable work only and the inspection totals lane is 0/0.
APPLICATION_STATUSES = frozenset({"applied", "not_applied", "display_only"})
APPLICATION_REASON_CODES = frozenset(
    {"approved_absorbs_children", "decision_rejected", "decision_uncertain",
     "split_recommended", "no_owned_children", "display_only_aggregate"}
)
LEDGER_REASON_CODES = frozenset(
    {"absorbed_by_approved_package", "no_covering_package",
     "package_rejected", "package_uncertain", "package_split"}
)
# Closed observability key sets. total is defined as the exact sum of the
# other five phases, so the validator can enforce it arithmetically.
OBSERVABILITY_PHASES = (
    "condition_review", "standalone_estimate", "package_candidates",
    "sol_review", "reconciliation", "total",
)
FUNNEL_KEYS = (
    "observations", "conditions", "condition_reviews",
    "dispositions_accepted_for_work", "dispositions_excluded",
    "dispositions_inspection", "dispositions_withheld",
    "dispositions_no_action", "work_items_active", "work_items_suppressed",
    "package_candidates", "package_decisions", "applications_applied",
    "applications_not_applied", "applications_display_only",
    "ledger_standalone", "ledger_absorbed", "ledger_entries",
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
# Pinned by test against tools/estimate_scope.py's label constants so the two
# vocabularies can never drift (the UNIT_POLICIES pattern; contracts stays
# import-light). MERGE_PRIORITY resolves a dedup collision whose sources carry
# different scopes: the most-required lane wins.
ESTIMATE_SCOPES = frozenset(
    {"required_rehab", "marketability_rehab", "optional_value_add",
     "inspection_risk"}
)
ESTIMATE_SCOPE_MERGE_PRIORITY = (
    "required_rehab", "marketability_rehab", "optional_value_add",
    "inspection_risk",
)

# The complete provenance policy set. Every envelope (including failed ones)
# carries exactly these keys; sessions extend the map together with the schema
# version. POLICY_VERSION_KEYS is derived so the two can never drift.
POLICY_VERSIONS = {
    "terminal_route_policy": TERMINAL_ROUTE_POLICY_VERSION,
    "condition_disposition_policy": CONDITION_DISPOSITION_POLICY_VERSION,
    "evidence_dedup_policy": EVIDENCE_DEDUP_POLICY_VERSION,
    "terra_review_prompt": TERRA_REVIEW_PROMPT_VERSION,
    "work_derivation_policy": WORK_DERIVATION_POLICY_VERSION,
    "work_dedup_policy": WORK_DEDUP_POLICY_VERSION,
    "standalone_pricing_policy": STANDALONE_PRICING_POLICY_VERSION,
    "package_candidate_policy": PACKAGE_CANDIDATE_POLICY_VERSION,
    "sol_review_prompt": SOL_REVIEW_PROMPT_VERSION,
    "package_application_policy": PACKAGE_APPLICATION_POLICY_VERSION,
    "coverage_reconciliation_policy": COVERAGE_RECONCILIATION_POLICY_VERSION,
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
    # Objective "field:value" opening-instance identifiers from the source
    # issues (opening_id, window_key, ...). Usually empty; per_opening work
    # derivation counts them, everything else ignores them.
    opening_instance_hints: Tuple[str, ...]


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
    """One billable renovation job, with merged lineage.

    A source item carries singleton lineage tuples; a merged dedup active
    spans every colliding source's conditions, catalog items, and estimate
    units. billable_unit_id is the physical unit being billed, or the fixed
    collapse token ("property"/"system"/"area") for collapse unit policies.
    Suppressed items keep their dollars as audit data; only active items
    enter totals."""
    work_item_id: str
    schema_version: int
    condition_ids: Tuple[str, ...]
    catalog_item_ids: Tuple[str, ...]
    source_estimate_unit_ids: Tuple[str, ...]
    billable_unit_id: str
    action_code: str
    action_sources: Tuple[str, ...]
    trade_bucket: str
    unit_policy: str
    unit_count: int
    pricing_modes: Tuple[str, ...]
    identity_ambiguous: bool
    estimate_scope: str
    estimate_scope_reason: str
    low: int
    high: int
    status: str
    reason_code: Optional[str]


@dataclass(frozen=True)
class WorkDedupCollision(_Contract):
    """Audit record linking one merged active work item to the suppressed
    sources that shared its dedup key. The active range is the exact max
    envelope over the sources — colliding work is never summed."""
    collision_id: str
    schema_version: int
    action_code: str
    trade_bucket: str
    unit_policy: str
    billable_unit_id: str
    active_work_item_id: str
    suppressed_work_item_ids: Tuple[str, ...]
    policy_version: str


@dataclass(frozen=True)
class PackageCandidate(_Contract):
    """One deterministic package opportunity over ACTIVE work items.

    Children/drivers/supports are stable work-item IDs (driver precedence
    when one merged work item supplied both roles). low/high is the floored
    allowance — max of the escalated tier spec and the children's standalone
    range — with the unfloored tier spec retained for audit. The display-only
    whole-home aggregate carries NO children: its lineage flows through
    contributing_candidate_ids, so it can never absorb work."""
    package_candidate_id: str
    schema_version: int
    package_type: str
    package_category: str
    package_level: str
    room: str
    estimate_unit_id: str
    child_work_item_ids: Tuple[str, ...]
    driver_work_item_ids: Tuple[str, ...]
    support_work_item_ids: Tuple[str, ...]
    strength: str
    pricing_profile: str
    pricing_tier: str
    absorption_scope: Mapping[str, Any]
    proposed_treatment: str
    unfloored_low: int
    unfloored_high: int
    cost_floor_applied: bool
    low: int
    high: int
    display_only: bool
    contributing_candidate_ids: Tuple[str, ...]


@dataclass(frozen=True)
class PackageDecision(_Contract):
    """Sol's verdict on a supplied candidate. Same closed-field-set boundary
    as ConditionReview: no work truth, no prices. Provenance ties the
    decision to the one listing-level Sol call that produced it."""
    decision_id: str
    schema_version: int
    package_candidate_id: str
    decision: str
    combine_with: Tuple[str, ...]
    split_groups: Tuple[Tuple[str, ...], ...]
    rationale: str
    model: str
    prompt_version: str
    sol_call_id: str
    request_fingerprint: str
    provider: str


@dataclass(frozen=True)
class SolCall(_Contract):
    """One Sol provider call (or its checkpoint republication) for the whole
    listing. Telemetry only — Sol has no approved daily budget, so there is
    no ledger and no debit field; checkpoint reuse keeps the original token
    numbers with usage_source recording the provenance."""
    call_id: str
    schema_version: int
    package_candidate_ids: Tuple[str, ...]
    request_fingerprint: str
    provider: str
    model: str
    prompt_version: str
    usage_source: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class SolListingUsage(_Contract):
    """Listing-level rollup of every Sol call in the run."""
    schema_version: int
    call_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class PackageReviewSnapshots(_Contract):
    """Immutability fingerprints over the layers Sol must never change.

    Each hash is sha256_canonical over the corresponding result sections
    exactly as they entered the Sol request; the validator recomputes them,
    so any post-review mutation of condition, work, or candidate truth is a
    validation failure, not a diff review."""
    schema_version: int
    condition_snapshot_sha256: str
    work_snapshot_sha256: str
    candidate_snapshot_sha256: str


@dataclass(frozen=True)
class PackageApplication(_Contract):
    """Deterministic application of one Sol decision to one candidate.

    Applied packages own their absorbed children at most once (the legacy
    absorption priority resolves shared children) and bill the effective
    range max(unfloored tier spec, sum of actually-owned child allowances) —
    never the stored candidate floor, which would double-count children lost
    to a higher-priority package. Rejected, uncertain, split-recommended,
    display-only, and zero-owned candidates are non-billable with a reason
    code. combine_group_id is non-economic grouping metadata shared by every
    member of one combine closure."""
    application_id: str
    schema_version: int
    package_candidate_id: str
    decision_id: str
    status: str
    reason_code: str
    absorbed_work_item_ids: Tuple[str, ...]
    unabsorbed_child_work_item_ids: Tuple[str, ...]
    combine_group_id: Optional[str]
    effective_low: int
    effective_high: int


@dataclass(frozen=True)
class ReconciliationAudit(_Contract):
    """Recomputed defect lists over the finished reconciliation. A complete
    result requires every list to be empty — a dirty audit fails closed to a
    failed envelope instead of publishing."""
    schema_version: int
    lost_work_item_ids: Tuple[str, ...]
    duplicate_absorption: Tuple[str, ...]
    unsupported_billing: Tuple[str, ...]
    orphan_children: Tuple[str, ...]
    arithmetic_mismatches: Tuple[str, ...]


@dataclass(frozen=True)
class EstimateObservability(_Contract):
    """Run telemetry: monotonic phase timings (total == exact sum of the
    other phases), token rollups recomputable from the Terra/Sol usage
    blocks, and funnel counts recomputable from the result sections."""
    schema_version: int
    phase_timings_ms: Mapping[str, int]
    terra_total_tokens: int
    sol_total_tokens: int
    combined_total_tokens: int
    funnel: Mapping[str, int]


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
    reason_code: str
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
class StandaloneEstimate(_Contract):
    """Package-independent totals from ACTIVE work items only.

    property_cost_factor(_audit) records the one pricing input that is not
    derivable from the catalog plus the result, keeping the estimate
    reproducible. totals_by_estimate_scope carries exactly the four
    ESTIMATE_SCOPES keys; headline is their componentwise sum, which equals
    the sum over all active work items."""
    schema_version: int
    currency: str
    pricing_policy_version: str
    property_cost_factor: float
    property_cost_factor_audit: Mapping[str, Any]
    totals_by_estimate_scope: Mapping[str, MoneyRange]
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
