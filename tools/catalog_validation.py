"""Validation for tools/issue_catalog.json.

The issue catalog is fully load-bearing: package routing lives in per-item
``package_affinity`` blocks and the estimate engine consumes tiers, groups,
costs and scene_groups permissively, with silent fallbacks (an unknown
``estimate_tier`` coerces to "minor", an unknown ``group`` falls to the
default budget cap, a wrong scene_group token drops the item from retrieval).
This module makes that drift loud.

Hand-rolled on purpose — no jsonschema dependency. Cross-field rules
(cost required iff tier is high/medium, package_type must match its room)
are more natural in plain Python, and the vocabularies are imported from
their owning modules so they cannot drift from the consumers.

Enforcement point is ``tests/test_catalog_validation.py``; validation is
deliberately NOT hooked into runtime catalog loading so a bad edit cannot
take down the analyzer server.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, get_args

from tools.catalog_cost_model import LINE_ITEM, ROOM_ALLOWANCE
from tools.estimate_scope import VALID_ESTIMATE_SCOPES
from tools.observation_kinds import (
    LEGACY_CATALOG_KINDS,
    OBSERVATION_KINDS,
    ONTOLOGY_VERSION,
)
from tools.pipeline_common import SCENE_GROUPS_UI, TERM_WHOLE_WORD_MARKER
from tools.rehab_packages import (
    PACKAGE_ROLE_DRIVER,
    PACKAGE_ROLE_IGNORE,
    PACKAGE_ROLE_STANDALONE,
    VALID_DISPLAY_CLASSES,
    build_package_affinity,
)
from tools.renovation_estimate import (
    EstimateStackBehavior,
    EstimateStrategy,
    EstimateTier,
    GROUP_BUDGET_CAPS,
    VALID_UNIT_POLICIES,
)


_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")

# Kind vocabulary is versioned off the catalog's own root metadata: a catalog
# stamped ontology_version == observation-kind-v2 uses the three-kind enum,
# anything else (the shipped v1 file has no stamp) uses the legacy two kinds.
VALID_KINDS = LEGACY_CATALOG_KINDS
VALID_KINDS_V2 = OBSERVATION_KINDS

# v2 root metadata vocabulary. The writer guard (artifact_writers) treats any
# non-"publishable" status as blocked; the validator pins the enum.
VALID_PUBLICATION_STATUSES = frozenset({"blocked_pending_pricing", "publishable"})
V2_CATALOG_VERSION = "3.0"

# Fields that carry economic behavior. Split successors inherit them verbatim
# from their v1 parent (pricing_status == "inherited_from_split_parent" — the
# Task 4A bridge until the dedicated pricing project authors real successor
# prices). The migration generator (scripts/migrate_catalog_kind_v2.py)
# imports this list so generator and validator can never disagree.
ECONOMIC_FIELDS = (
    "cost",
    "estimate",
    "work_item_code",
    "cost_model",
    "package_affinity",
    "package_role",
    "estimate_scope",
    "estimate_scope_reason",
)
PRICING_STATUS_INHERITED = "inherited_from_split_parent"
VALID_PRICING_STATUSES = frozenset({PRICING_STATUS_INHERITED})

VALID_CHANGE_TYPES = frozenset({"unchanged", "reclassified", "narrowed", "split"})

VALID_SCOPES = frozenset({"repair", "replace", "cosmetic", "service"})
VALID_ITEM_TIERS = frozenset({"work", "optional"})
VALID_COST_MODES = frozenset({"allowance", "heuristic"})

# Estimate-block vocabularies owned by renovation_estimate's Literal aliases.
VALID_ESTIMATE_TIERS = frozenset(get_args(EstimateTier))
VALID_ESTIMATE_STRATEGIES = frozenset(get_args(EstimateStrategy))
VALID_STACK_BEHAVIORS = frozenset(get_args(EstimateStackBehavior))

# package_allowance / inspection_allowance are derived at runtime
# (catalog_cost_model.derive_cost_model) and must never be authored.
AUTHORABLE_COST_MODELS = frozenset({LINE_ITEM, ROOM_ALLOWANCE})

# "pool" is not a UI scene group (it is a scene that maps to "exterior") but
# 14 items carry it redundantly alongside "exterior"; accepted vocabulary.
VALID_SCENE_GROUP_TOKENS = frozenset(SCENE_GROUPS_UI) | {"pool"}

# Routing was moved into per-item package_affinity blocks (d887622); flat
# routing fields must not be re-introduced.
FORBIDDEN_FLAT_ROUTING_FIELDS = ("package_type", "package_category", "room")
VALID_FLAT_PACKAGE_ROLES = frozenset({PACKAGE_ROLE_STANDALONE, PACKAGE_ROLE_IGNORE})

_COST_AMOUNT_KEYS = (
    "base_low", "base_high",
    "per_occurrence_low", "per_occurrence_high",
    "cap_low", "cap_high",
)
_COST_ORDERED_PAIRS = (
    ("base_low", "base_high"),
    ("per_occurrence_low", "per_occurrence_high"),
    ("cap_low", "cap_high"),
    ("base_high", "cap_high"),
)

_BOOL_FIELDS = ("defaultHidden", "drop_if_generic")
_LIST_FIELDS = ("deny_any", "support_any", "require_any")


@dataclass
class CatalogValidationResult:
    errors: List[str] = field(default_factory=list)    # "item_id: message"
    warnings: List[str] = field(default_factory=list)  # advisory, not gated

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_issue_catalog(issue_catalog: Dict[str, Any]) -> CatalogValidationResult:
    result = CatalogValidationResult()
    catalog = issue_catalog if isinstance(issue_catalog, dict) else {}

    is_v2 = catalog.get("ontology_version") == ONTOLOGY_VERSION
    kind_vocabulary = VALID_KINDS_V2 if is_v2 else VALID_KINDS
    if is_v2:
        _validate_v2_root(catalog, result)

    declared_buckets = [
        str(bucket.get("id") or "")
        for bucket in catalog.get("trade_buckets") or []
        if isinstance(bucket, dict)
    ]
    declared_bucket_set = {b for b in declared_buckets if b}
    if not declared_bucket_set:
        result.errors.append("<catalog>: trade_buckets section missing or empty")

    used_buckets = set()
    seen_ids = set()

    for index, item in enumerate(catalog.get("items") or []):
        if not isinstance(item, dict):
            result.errors.append(f"<catalog>: items[{index}] is not an object")
            continue

        raw_id = item.get("id")
        label = str(raw_id) if raw_id else f"items[{index}]"

        # 1. id
        if not raw_id or not isinstance(raw_id, str):
            result.errors.append(f"{label}: id missing or not a string")
        elif not _ID_PATTERN.fullmatch(raw_id):
            result.errors.append(f"{label}: id must match ^[a-z0-9_]+$")
        elif raw_id in seen_ids:
            result.errors.append(f"{label}: duplicate id")
        else:
            seen_ids.add(raw_id)

        _validate_core_enums(item, label, result, kind_vocabulary)
        if is_v2:
            _validate_v2_item(item, label, result)
        _validate_trade_bucket(item, label, declared_bucket_set, used_buckets, result)
        _validate_scene_groups(item, label, result)
        _validate_estimate_block(item, label, result)
        _validate_cost_block(item, label, result)
        _validate_static_classifications(item, label, result)
        _validate_cost_model(item, label, result)
        _validate_package_affinity(item, label, result)
        _validate_flat_routing_fields(item, label, result)
        _validate_field_types(item, label, result)

        # Warnings
        if _is_affinity_driver(item) and not item.get("cost"):
            result.warnings.append(
                f"{label}: package_affinity driver without a cost block "
                "(tier escalation sees $0 for this item)"
            )
        if not item.get("display_class"):
            result.warnings.append(
                f"{label}: missing display_class "
                "(falls back to keyword sniffing in catalog_display_class)"
            )

    for bucket in declared_buckets:
        if bucket and bucket not in used_buckets:
            result.warnings.append(
                f"<catalog>: trade bucket {bucket!r} declared but used by no item"
            )

    return result


def _validate_core_enums(item: Dict[str, Any], label: str,
                         result: CatalogValidationResult,
                         kind_vocabulary: frozenset = VALID_KINDS) -> None:
    kind = item.get("kind")
    if kind not in kind_vocabulary:
        result.errors.append(f"{label}: kind {kind!r} not in {sorted(kind_vocabulary)}")
    scope = item.get("scope")
    if scope not in VALID_SCOPES:
        result.errors.append(f"{label}: scope {scope!r} not in {sorted(VALID_SCOPES)}")
    tier = item.get("tier")
    if tier not in VALID_ITEM_TIERS:
        result.errors.append(f"{label}: tier {tier!r} not in {sorted(VALID_ITEM_TIERS)}")
    severity = item.get("severity")
    if not isinstance(severity, int) or isinstance(severity, bool) \
            or not 1 <= severity <= 5:
        result.errors.append(f"{label}: severity {severity!r} must be an int in 1-5")


def _validate_trade_bucket(item: Dict[str, Any], label: str,
                           declared: set, used: set,
                           result: CatalogValidationResult) -> None:
    bucket = item.get("trade_bucket")
    if isinstance(bucket, str) and bucket:
        used.add(bucket)
    # Skip membership checks when the declaration section itself is bad —
    # that already produced a catalog-level error; avoid one error per item.
    if not declared:
        return
    if bucket not in declared:
        result.errors.append(
            f"{label}: trade_bucket {bucket!r} is not declared in trade_buckets"
        )


def _validate_scene_groups(item: Dict[str, Any], label: str,
                           result: CatalogValidationResult) -> None:
    scene_groups = item.get("scene_groups")
    if not isinstance(scene_groups, list) or not scene_groups:
        result.errors.append(f"{label}: scene_groups must be a non-empty list")
        return
    for token in scene_groups:
        if token not in VALID_SCENE_GROUP_TOKENS:
            result.errors.append(
                f"{label}: scene_groups token {token!r} not in "
                f"{sorted(VALID_SCENE_GROUP_TOKENS)}"
            )


def _validate_estimate_block(item: Dict[str, Any], label: str,
                             result: CatalogValidationResult) -> None:
    estimate = item.get("estimate")
    if estimate is None:
        return
    if not isinstance(estimate, dict):
        result.errors.append(f"{label}: estimate must be an object")
        return
    # estimate_tier is required in an authored block: an absent tier coerces
    # to "minor" at runtime, which reads as an accident rather than intent.
    tier = estimate.get("estimate_tier")
    if tier not in VALID_ESTIMATE_TIERS:
        result.errors.append(
            f"{label}: estimate_tier {tier!r} not in {sorted(VALID_ESTIMATE_TIERS)}"
        )
    if "strategy" in estimate and estimate["strategy"] not in VALID_ESTIMATE_STRATEGIES:
        result.errors.append(
            f"{label}: estimate strategy {estimate['strategy']!r} not in "
            f"{sorted(VALID_ESTIMATE_STRATEGIES)}"
        )
    if "stack_behavior" in estimate \
            and estimate["stack_behavior"] not in VALID_STACK_BEHAVIORS:
        result.errors.append(
            f"{label}: stack_behavior {estimate['stack_behavior']!r} not in "
            f"{sorted(VALID_STACK_BEHAVIORS)}"
        )
    if "unit_policy" in estimate \
            and estimate["unit_policy"] not in VALID_UNIT_POLICIES:
        result.errors.append(
            f"{label}: unit_policy {estimate['unit_policy']!r} not in "
            f"{sorted(VALID_UNIT_POLICIES)}"
        )
    if "group" in estimate and estimate["group"] not in GROUP_BUDGET_CAPS:
        result.errors.append(
            f"{label}: estimate group {estimate['group']!r} not in "
            f"GROUP_BUDGET_CAPS {sorted(GROUP_BUDGET_CAPS)}"
        )


def _validate_cost_block(item: Dict[str, Any], label: str,
                         result: CatalogValidationResult) -> None:
    cost = item.get("cost")
    estimate = item.get("estimate") if isinstance(item.get("estimate"), dict) else {}
    tier = (estimate or {}).get("estimate_tier")
    if cost is None:
        # A priced line item with no cost falls into opaque heuristics.
        if tier in ("high", "medium"):
            result.errors.append(
                f"{label}: estimate_tier {tier!r} requires a cost block"
            )
        return
    if not isinstance(cost, dict):
        result.errors.append(f"{label}: cost must be an object")
        return
    mode = cost.get("mode")
    if mode not in VALID_COST_MODES:
        result.errors.append(
            f"{label}: cost mode {mode!r} not in {sorted(VALID_COST_MODES)}"
        )
    # Heuristic blocks legitimately carry only "mode"; amounts are checked
    # only when present.
    for key in _COST_AMOUNT_KEYS:
        if key in cost and (not _is_number(cost[key]) or cost[key] < 0):
            result.errors.append(
                f"{label}: cost {key} {cost[key]!r} must be a number >= 0"
            )
    for low_key, high_key in _COST_ORDERED_PAIRS:
        low, high = cost.get(low_key), cost.get(high_key)
        if _is_number(low) and _is_number(high) and low > high:
            result.errors.append(
                f"{label}: cost {low_key} ({low}) > {high_key} ({high})"
            )


def _validate_static_classifications(item: Dict[str, Any], label: str,
                                     result: CatalogValidationResult) -> None:
    estimate_scope = item.get("estimate_scope")
    if estimate_scope is not None and estimate_scope not in VALID_ESTIMATE_SCOPES:
        result.errors.append(
            f"{label}: estimate_scope {estimate_scope!r} not in "
            f"{sorted(VALID_ESTIMATE_SCOPES)}"
        )
    display_class = item.get("display_class")
    if display_class is not None and display_class not in VALID_DISPLAY_CLASSES:
        result.errors.append(
            f"{label}: display_class {display_class!r} not in "
            f"{sorted(VALID_DISPLAY_CLASSES)}"
        )


def _validate_cost_model(item: Dict[str, Any], label: str,
                         result: CatalogValidationResult) -> None:
    estimate = item.get("estimate") if isinstance(item.get("estimate"), dict) else {}
    for source, value in (("cost_model", item.get("cost_model")),
                          ("estimate.cost_model", (estimate or {}).get("cost_model"))):
        if value is not None and value not in AUTHORABLE_COST_MODELS:
            result.errors.append(
                f"{label}: {source} {value!r} not in {sorted(AUTHORABLE_COST_MODELS)} "
                "(package/inspection allowances are derived at runtime, never authored)"
            )


def _validate_package_affinity(item: Dict[str, Any], label: str,
                               result: CatalogValidationResult) -> None:
    if item.get("package_affinity") is None:
        return
    # Reuse the runtime validation rather than duplicating it; called per item
    # so one bad block does not mask errors on other items.
    try:
        build_package_affinity({"items": [item]})
    except ValueError as exc:
        message = str(exc)
        if not message.startswith(label):
            message = f"{label}: {message}"
        result.errors.append(message)


def _validate_flat_routing_fields(item: Dict[str, Any], label: str,
                                  result: CatalogValidationResult) -> None:
    for field_name in FORBIDDEN_FLAT_ROUTING_FIELDS:
        if field_name in item:
            result.errors.append(
                f"{label}: flat routing field {field_name!r} is forbidden — "
                "routing lives in package_affinity blocks"
            )
    package_role = item.get("package_role")
    if package_role is not None and package_role not in VALID_FLAT_PACKAGE_ROLES:
        result.errors.append(
            f"{label}: flat package_role {package_role!r} not in "
            f"{sorted(VALID_FLAT_PACKAGE_ROLES)} — driver/support live in "
            "package_affinity blocks"
        )


def _validate_field_types(item: Dict[str, Any], label: str,
                          result: CatalogValidationResult) -> None:
    for field_name in _BOOL_FIELDS:
        if field_name in item and not isinstance(item[field_name], bool):
            result.errors.append(f"{label}: {field_name} must be a bool")
    for field_name in _LIST_FIELDS:
        if field_name in item and not isinstance(item[field_name], list):
            result.errors.append(f"{label}: {field_name} must be a list")
            continue
        for term in item.get(field_name) or []:
            # "$" opts a term in to a trailing word boundary (term_matches).
            # It is only meaningful as the final character, and a lone "$"
            # never matches — both are authoring typos, not intent.
            text = str(term)
            if TERM_WHOLE_WORD_MARKER in text[:-1] or text == TERM_WHOLE_WORD_MARKER:
                result.errors.append(
                    f"{label}: {field_name} term {text!r} may only carry "
                    f"'{TERM_WHOLE_WORD_MARKER}' as a trailing whole-word marker"
                )


def _is_affinity_driver(item: Dict[str, Any]) -> bool:
    affinity = item.get("package_affinity")
    if not isinstance(affinity, dict):
        return False
    return any(
        isinstance(entry, dict) and entry.get("package_role") == PACKAGE_ROLE_DRIVER
        for entry in affinity.values()
    )


def _validate_v2_root(catalog: Dict[str, Any], result: CatalogValidationResult) -> None:
    version = catalog.get("version")
    if version != V2_CATALOG_VERSION:
        result.errors.append(
            f"<catalog>: v2 catalog version {version!r} must be {V2_CATALOG_VERSION!r}"
        )
    status = catalog.get("publication_status")
    if status not in VALID_PUBLICATION_STATUSES:
        result.errors.append(
            f"<catalog>: publication_status {status!r} not in "
            f"{sorted(VALID_PUBLICATION_STATUSES)}"
        )


def _validate_v2_item(item: Dict[str, Any], label: str,
                      result: CatalogValidationResult) -> None:
    claim = item.get("atomic_claim")
    if not isinstance(claim, dict):
        result.errors.append(f"{label}: atomic_claim missing or not an object")
    else:
        for key in ("subject", "state", "ontology_basis"):
            value = claim.get(key)
            if not isinstance(value, str) or not value.strip():
                result.errors.append(
                    f"{label}: atomic_claim.{key} must be a non-empty string"
                )
    pricing_status = item.get("pricing_status")
    if pricing_status is not None:
        if pricing_status not in VALID_PRICING_STATUSES:
            result.errors.append(
                f"{label}: pricing_status {pricing_status!r} not in "
                f"{sorted(VALID_PRICING_STATUSES)}"
            )
        elif not any(f in item for f in ECONOMIC_FIELDS):
            # Allowed (the parent may have carried nothing), but worth surfacing.
            result.warnings.append(
                f"{label}: pricing_status {pricing_status!r} with no inherited "
                "economic fields (parent carried none)"
            )


def validate_migration_manifest(
    manifest: Dict[str, Any],
    v1_catalog: Dict[str, Any],
    v2_catalog: Dict[str, Any],
) -> CatalogValidationResult:
    """Validate the audit-only migration manifest against both catalogs.

    Checks: exactly one entry per legacy id, successor integrity (every
    successor exists in v2 with the kind the manifest claims), no orphan v2
    items (each is the successor of exactly one entry), well-formed flags, and
    re-resolution consistency (any kind change or split requires it).
    """
    result = CatalogValidationResult()
    v1_by_id = {it["id"]: it for it in (v1_catalog.get("items") or [])}
    v2_by_id = {it["id"]: it for it in (v2_catalog.get("items") or [])}

    entries = manifest.get("entries") or []
    if manifest.get("audit_only") is not True:
        result.errors.append("<manifest>: audit_only must be true — this is not a runtime alias table")

    seen_legacy: set = set()
    successor_parents: Dict[str, str] = {}
    for entry in entries:
        legacy_id = entry.get("legacy_id")
        label = str(legacy_id)
        if legacy_id in seen_legacy:
            result.errors.append(f"{label}: duplicate manifest entry")
            continue
        seen_legacy.add(legacy_id)

        v1_item = v1_by_id.get(legacy_id)
        if v1_item is None:
            result.errors.append(f"{label}: not a legacy catalog id")
            continue
        if entry.get("legacy_kind") != v1_item.get("kind"):
            result.errors.append(
                f"{label}: legacy_kind {entry.get('legacy_kind')!r} disagrees with "
                f"the v1 catalog ({v1_item.get('kind')!r})"
            )

        change_type = entry.get("change_type")
        if change_type not in VALID_CHANGE_TYPES:
            result.errors.append(
                f"{label}: change_type {change_type!r} not in {sorted(VALID_CHANGE_TYPES)}"
            )
            continue
        successors = entry.get("successors") or []
        if change_type == "split":
            if len(successors) < 2:
                result.errors.append(f"{label}: split entry needs >= 2 successors")
            if entry.get("deprecated") is not True:
                result.errors.append(f"{label}: split parent must be deprecated")
        else:
            if len(successors) != 1 or successors[0].get("id") != legacy_id:
                result.errors.append(
                    f"{label}: non-split entry must have exactly one successor keeping the legacy id"
                )
            if entry.get("deprecated"):
                result.errors.append(f"{label}: non-split entry must not be deprecated")

        kind_changed = False
        for succ in successors:
            succ_id = succ.get("id")
            v2_item = v2_by_id.get(succ_id)
            if v2_item is None:
                result.errors.append(f"{label}: successor {succ_id!r} not in the v2 catalog")
                continue
            if v2_item.get("kind") != succ.get("kind"):
                result.errors.append(
                    f"{label}: successor {succ_id!r} kind {succ.get('kind')!r} disagrees "
                    f"with the v2 catalog ({v2_item.get('kind')!r})"
                )
            if succ_id in successor_parents:
                result.errors.append(
                    f"{label}: successor {succ_id!r} already claimed by "
                    f"{successor_parents[succ_id]!r}"
                )
            else:
                successor_parents[succ_id] = str(legacy_id)
            if succ.get("kind") != v1_item.get("kind"):
                kind_changed = True

        must_re_resolve = change_type == "split" or kind_changed
        if bool(entry.get("requires_re_resolution")) != must_re_resolve:
            result.errors.append(
                f"{label}: requires_re_resolution must be {must_re_resolve} "
                f"(change_type={change_type}, kind_changed={kind_changed})"
            )
        if not str(entry.get("atomicity_rationale") or "").strip():
            result.errors.append(f"{label}: atomicity_rationale missing")

    missing = set(v1_by_id) - seen_legacy
    if missing:
        result.errors.append(f"<manifest>: legacy ids without an entry: {sorted(missing)}")
    orphans = set(v2_by_id) - set(successor_parents)
    if orphans:
        result.errors.append(f"<manifest>: v2 items with no manifest parent: {sorted(orphans)}")

    return result


def load_shipped_catalog() -> Dict[str, Any]:
    path = Path(__file__).resolve().parent / "issue_catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_shipped_catalog_v2() -> Dict[str, Any]:
    path = Path(__file__).resolve().parent / "issue_catalog_kind_v2.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_shipped_manifest() -> Dict[str, Any]:
    path = Path(__file__).resolve().parent / "catalog_migrations" / "2.1_to_3.0.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    result = validate_issue_catalog(load_shipped_catalog())
    print(f"errors: {len(result.errors)}")
    for entry in result.errors:
        print(f"  ERROR {entry}")
    print(f"warnings: {len(result.warnings)}")
    for entry in result.warnings:
        print(f"  WARN  {entry}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
