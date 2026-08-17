"""Strict, versioned projection of the newest issue catalog (v3.1).

The new engine accepts only the generated v3.1 / observation-kind-v2 catalog
(tools/issue_catalog_kind_v2.json — never hand-edit it; edit
tools/catalog_migrations/kind_v2_decisions.json and re-run the generator).
The build collects every problem and raises once, at startup, so an invalid
catalog can never fail midway through a listing.

The builder enforces STRUCTURAL completeness: every catalog condition resolves
to exactly one terminal route, and unknown routing data fails the build rather
than disappearing through a default. The exact route distribution
(12 quarantine / 4 generic / 5 inspection / 9 no-action / 98 work over 128
items) is pinned in tests, not here, so a future catalog regeneration updates
test expectations instead of breaking shadow-mode worker startup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Tuple

from tools.catalog_validation import VALID_ROUTE_OVERRIDES, validate_issue_catalog
from tools.comparison_common import sha256_canonical, sha256_file
from tools.estimate_scope import classify_estimate_scope_with_reason
from tools.rehab_packages import build_package_affinity
from tools.renovation_architecture.contracts import (
    PROJECTION_VERSION,
    REQUIRED_CATALOG_ONTOLOGY,
    REQUIRED_CATALOG_VERSION,
    STRATEGIES,
    TERMINAL_ROUTE_POLICY_VERSION,
    TERMINAL_ROUTES,
    UNIT_POLICIES,
)


class RenovationCatalogError(ValueError):
    """The catalog cannot back the new architecture. Carries every problem."""

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        super().__init__(
            "renovation catalog projection rejected:\n" + "\n".join(self.errors)
        )


def resolve_terminal_route(
    item: Mapping[str, Any], *, quarantined_buckets: FrozenSet[str]
) -> Tuple[str, str]:
    """Resolve one item's terminal route as (route, reason_code).

    The precedence is load-bearing: dated_electrical_outlets_switches carries
    drop_if_generic AND sits in the quarantined electrical bucket, and must
    land in quarantine. drop_if_generic is absent on some items — absent means
    False. inspect_only items keep work codes and cost; inspection is a
    routing decision, not a pricing gap. An explicit route_override forces an
    otherwise-billable item out of billing (the Session 8 opportunity/presence
    triage) and must outrank the no-economics check so its reason code states
    intent rather than an inferred gap. The four items lacking BOTH cost and
    work_item_code are the user-approved optional gaps.
    """
    if item.get("trade_bucket") in quarantined_buckets:
        return "excluded_quarantine", "product_quarantined_trade"
    if item.get("drop_if_generic") is True:
        return "excluded_generic", "drop_if_generic"
    estimate = item.get("estimate") or {}
    if estimate.get("strategy") == "inspect_only":
        return "inspection", "strategy_inspect_only"
    if item.get("route_override") == "no_action":
        return "no_action", "route_override_no_action"
    if not item.get("cost") and not item.get("work_item_code"):
        return "no_action", "no_economics_approved_gap"
    return "work", "work_default"


def build_renovation_catalog_projection(
    catalog: Mapping[str, Any], *, catalog_path: Path
) -> Dict[str, Any]:
    errors: List[str] = []

    version = catalog.get("version")
    if version != REQUIRED_CATALOG_VERSION:
        errors.append(
            f"<catalog>: version must be {REQUIRED_CATALOG_VERSION!r}, got {version!r}"
        )
    ontology = catalog.get("ontology_version")
    if ontology != REQUIRED_CATALOG_ONTOLOGY:
        errors.append(
            f"<catalog>: ontology_version must be {REQUIRED_CATALOG_ONTOLOGY!r} "
            f"(the hyphenated catalog stamp), got {ontology!r}"
        )
    publication_status = catalog.get("publication_status")
    if publication_status != "publishable":
        errors.append(
            f"<catalog>: publication_status must be 'publishable', got "
            f"{publication_status!r}"
        )

    validation = validate_issue_catalog(dict(catalog))
    errors.extend(validation.errors)

    try:
        build_package_affinity(dict(catalog))
    except ValueError as exc:
        errors.append(f"<catalog>: package_affinity invalid: {exc}")

    quarantined_buckets = frozenset(
        str(bucket.get("id"))
        for bucket in catalog.get("trade_buckets") or []
        if isinstance(bucket, dict) and bucket.get("product_quarantined") is True
    )

    items = [item for item in catalog.get("items") or [] if isinstance(item, dict)]
    routes: Dict[str, Dict[str, str]] = {}
    for item in items:
        item_id = str(item.get("id") or "")
        if not item_id:
            continue  # validate_issue_catalog already reported it
        estimate = item.get("estimate") or {}
        strategy = estimate.get("strategy")
        if strategy is not None and strategy not in STRATEGIES:
            errors.append(
                f"{item_id}: unknown estimate.strategy {strategy!r} — unknown "
                "routing data must not disappear through a default"
            )
        unit_policy = estimate.get("unit_policy")
        if unit_policy is not None and unit_policy not in UNIT_POLICIES:
            errors.append(f"{item_id}: unknown estimate.unit_policy {unit_policy!r}")
        route_override = item.get("route_override")
        if route_override is not None and route_override not in VALID_ROUTE_OVERRIDES:
            errors.append(
                f"{item_id}: unknown route_override {route_override!r} — "
                "unknown routing data must not disappear through a default"
            )
        route, reason_code = resolve_terminal_route(
            item, quarantined_buckets=quarantined_buckets
        )
        if route not in TERMINAL_ROUTES or item_id in routes:
            errors.append(f"{item_id}: did not resolve to exactly one terminal route")
        else:
            routes[item_id] = {"route": route, "reason_code": reason_code}

    if len(routes) != len(items):
        errors.append(
            f"<catalog>: {len(items)} items but {len(routes)} terminal routes — "
            "every condition needs an explicit terminal route"
        )
    if errors:
        raise RenovationCatalogError(errors)

    observables: Dict[str, Any] = {}
    work_policy: Dict[str, Any] = {}
    affinities: Dict[str, Any] = {}
    flat_roles: Dict[str, str] = {}
    for item in items:
        item_id = item["id"]
        estimate = item.get("estimate") or {}
        min_photo = estimate.get("min_photo_evidence")
        observables[item_id] = {
            "kind": item.get("kind"),
            "scope": item.get("scope"),
            "tier": item.get("tier"),
            "severity": item.get("severity"),
            "scene_groups": sorted(item.get("scene_groups") or []),
            "atomic_claim": item.get("atomic_claim"),
            "min_photo_evidence": (
                min_photo
                if isinstance(min_photo, int)
                and not isinstance(min_photo, bool)
                and min_photo > 0
                else None
            ),
        }
        if routes[item_id]["route"] in ("work", "inspection"):
            cost = item.get("cost")
            work_code = item.get("work_item_code")
            # Deterministic risk-lane metadata (projection v2). Inspection is
            # a routing fact, not a text classification; work items classify
            # through the existing estimate-scope policy with an empty
            # candidate, which reduces it to its catalog-only baseline.
            if routes[item_id]["route"] == "inspection":
                estimate_scope = "inspection_risk"
                estimate_scope_reason = "terminal_route_inspection"
            else:
                try:
                    estimate_scope, estimate_scope_reason = (
                        classify_estimate_scope_with_reason({}, item)
                    )
                except ValueError as exc:
                    errors.append(f"{item_id}: estimate scope unclassifiable: {exc}")
                    continue
            work_policy[item_id] = {
                # The action is the catalog work code, or the catalog scope
                # when no code exists — codes and prices are never invented.
                "action_code": work_code or item.get("scope"),
                "action_source": "work_item_code" if work_code else "catalog_scope",
                "trade_bucket": item.get("trade_bucket"),
                "strategy": estimate.get("strategy"),
                "unit_policy": estimate.get("unit_policy") or "per_scope",
                # Missing cost and mode=="heuristic" are identical downstream
                # (tools/costing.py heuristic fallback); the projection records
                # the merged fact rather than pretending to distinguish them.
                "pricing_mode": (
                    "heuristic"
                    if not cost or cost.get("mode") == "heuristic"
                    else "catalog_allowance"
                ),
                "cost": cost,
                "estimate_scope": estimate_scope,
                "estimate_scope_reason": estimate_scope_reason,
            }
        affinity = item.get("package_affinity")
        if isinstance(affinity, dict) and affinity:
            affinities[item_id] = {
                room: {
                    "package_type": entry.get("package_type"),
                    "package_role": entry.get("package_role"),
                }
                for room, entry in sorted(affinity.items())
            }
        flat_role = item.get("package_role")
        if flat_role is not None:
            flat_roles[item_id] = flat_role

    if errors:
        # Estimate-scope classification failures from the projection loop.
        raise RenovationCatalogError(errors)

    route_counts: Dict[str, int] = {route: 0 for route in sorted(TERMINAL_ROUTES)}
    for entry in routes.values():
        route_counts[entry["route"]] += 1

    projection: Dict[str, Any] = {
        "version": PROJECTION_VERSION,
        "route_policy_version": TERMINAL_ROUTE_POLICY_VERSION,
        "catalog_version": version,
        "catalog_ontology_version": ontology,
        "catalog_sha256": sha256_file(Path(catalog_path)),
        "catalog_validation_warning_count": len(validation.warnings),
        "observables": observables,
        "work_policy": work_policy,
        "package_policy": {"affinities": affinities, "flat_roles": flat_roles},
        "terminal_routes": routes,
        "route_counts": route_counts,
    }
    projection["fingerprint"] = sha256_canonical(projection)
    return projection
