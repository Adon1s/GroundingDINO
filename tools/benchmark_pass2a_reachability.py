"""Gold reachability audit for the package-outcome benchmark.

Answers, per gold target, whether the production catalog + assembly policy
can produce a scoreable output for it at all: catalog existence, trade
quarantine, scene eligibility, a priced standalone path, a package-affinity
path into an expected package (including the family-subsumption bridge), and
tier suppression. Strict targets that are unreachable fail gold validation;
known-incompatible targets stay visible as explicit diagnostic_only entries.

Read-only over the catalog — this module never modifies it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

# Scene-group token(s) a canonical room family exposes to catalog
# scene_groups matching. living -> living_areas is the known naming split.
ROOM_FAMILY_SCENE_GROUPS: Dict[str, Set[str]] = {
    "kitchen": {"kitchen"},
    "bathroom": {"bathroom"},
    "bedroom": {"bedroom"},
    "living": {"living_areas"},
    "exterior": {"exterior"},
    "utility": {"utility"},
    "other": {"other"},
}


def _catalog_by_id(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item.get("id"): item for item in catalog.get("items") or []}


def _expected_types_by_room(entry: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """{room_id: {package_type: target_pricing_profile}} over expected packages."""
    out: Dict[str, Dict[str, str]] = {}
    for pkg in entry.get("expected_packages") or []:
        out.setdefault(str(pkg.get("room_id")), {})[
            str(pkg.get("package_type"))] = str(pkg.get("target_pricing_profile"))
    return out


def _subsumption_bridge(route_type: str, expected: Dict[str, str],
                        family: str) -> Optional[str]:
    """An affinity route into {family}_repair can still land in gold's
    modernization package when that package's target tier subsumes repair
    (rehab_packages._REPAIR_SUBSUMING_MODERNIZATION_TIERS). The reverse never
    bridges: refresh does not subsume repair, repair does not subsume
    modernization."""
    from tools.rehab_packages import _REPAIR_SUBSUMING_MODERNIZATION_TIERS
    if route_type != f"{family}_repair":
        return None
    winner = f"{family}_modernization"
    profile = expected.get(winner)
    if not profile:
        return None
    tier = profile.partition("_")[2]
    if tier in _REPAIR_SUBSUMING_MODERNIZATION_TIERS:
        return winner
    return None


def _audit_work_item(work: Dict[str, Any], room: Dict[str, Any],
                     expected: Dict[str, str],
                     catalog_by_id: Dict[str, Dict[str, Any]],
                     quarantined: Set[str],
                     affinity_table: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    from tools.rehab_packages import catalog_package_role

    cid = str(work.get("catalog_item_id"))
    family = str(room.get("room") or "")
    routes: List[str] = []
    blockers: List[str] = []
    notes: List[str] = []
    affinity_agreement = "none"

    item = catalog_by_id.get(cid)
    if item is None:
        blockers.append("catalog_missing")
    else:
        if item.get("trade_bucket") in quarantined:
            blockers.append(f"quarantined_trade:{item.get('trade_bucket')}")
        scene_groups = set(item.get("scene_groups") or [])
        room_groups = ROOM_FAMILY_SCENE_GROUPS.get(family, {family})
        if scene_groups and not (scene_groups & room_groups):
            blockers.append(
                f"scene_ineligible:{sorted(scene_groups)} vs room {family}")

        flat_role = catalog_package_role(item)
        if flat_role == "standalone" and item.get("estimate"):
            routes.append("standalone_priced")
        elif flat_role == "standalone":
            blockers.append("standalone_without_estimate_block")

        affinity = affinity_table.get((family, cid))
        if affinity:
            route_type = str(affinity.get("package_type"))
            if route_type in expected:
                routes.append(f"{affinity.get('package_role')}:{route_type}")
                affinity_agreement = "match"
            else:
                bridge = _subsumption_bridge(route_type, expected, family)
                if bridge:
                    routes.append(f"subsumed_into:{bridge}")
                    affinity_agreement = "subsumed"
                else:
                    affinity_agreement = "mismatch"
                    blockers.append(
                        f"affinity_package_mismatch:{route_type} vs expected "
                        f"{sorted(expected) or ['(no expected package)']}")
        elif flat_role != "standalone":
            blockers.append("no_package_affinity_for_room")
        if (item.get("tier") == "optional"
                and item.get("kind") in ("upgrade", "modernization")):
            notes.append("tier=optional opportunity item "
                         "(driver suppression may gate it)")

    if not routes and not blockers:
        blockers.append("no_scoreable_route")
    return {
        "policy": work.get("policy") or "strict",
        "reachable": bool(routes),
        "routes": routes,
        "blockers": blockers,
        "affinity_agreement": affinity_agreement,
        "notes": "; ".join(notes),
    }


def _audit_expected_package(pkg: Dict[str, Any], room: Dict[str, Any],
                            affinity_table: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    ptype = str(pkg.get("package_type"))
    family = str(room.get("room") or "")
    blockers: List[str] = []
    drivers = sorted(
        cid for (fam, cid), aff in affinity_table.items()
        if fam == family and aff.get("package_type") == ptype
        and aff.get("package_role") == "package_driver")
    if not drivers:
        blockers.append(f"no_catalog_driver_for:{ptype}")
    return {
        "policy": pkg.get("policy") or "strict",
        "reachable": not blockers,
        "routes": [f"drivers:{len(drivers)}"] if drivers else [],
        "blockers": blockers,
        "affinity_agreement": "match" if drivers else "none",
        "notes": "",
    }


def audit_gold_reachability(gold: Dict[str, Any],
                            catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{prop: {target_key: verdict}} where target_key is
    'pkg:{package_type}__{room_id}' or 'item:{catalog_item_id}@{room_id}'."""
    from tools.rehab_packages import build_package_affinity
    from tools.renovation_estimate import product_quarantined_trade_buckets

    catalog_by_id = _catalog_by_id(catalog)
    quarantined = set(product_quarantined_trade_buckets(catalog))
    affinity_table = build_package_affinity(catalog)

    out: Dict[str, Dict[str, Any]] = {}
    for prop, entry in sorted((gold.get("properties") or {}).items()):
        rooms = {r.get("room_id"): r for r in entry.get("canonical_rooms") or []}
        expected_by_room = _expected_types_by_room(entry)
        verdicts: Dict[str, Any] = {}
        for pkg in entry.get("expected_packages") or []:
            room = rooms.get(pkg.get("room_id")) or {}
            key = f"pkg:{pkg.get('package_type')}__{pkg.get('room_id')}"
            verdicts[key] = _audit_expected_package(pkg, room, affinity_table)
        for work in entry.get("required_work_items") or []:
            room = rooms.get(work.get("room_id")) or {}
            expected = expected_by_room.get(str(work.get("room_id")), {})
            key = f"item:{work.get('catalog_item_id')}@{work.get('room_id')}"
            verdicts[key] = _audit_work_item(
                work, room, expected, catalog_by_id, quarantined, affinity_table)
        out[prop] = verdicts
    return out


def strict_unreachable_errors(audit: Dict[str, Dict[str, Any]]) -> List[str]:
    """Gold-validation failures: strict targets with no scoreable route."""
    errors: List[str] = []
    for prop, verdicts in sorted(audit.items()):
        for key, verdict in sorted(verdicts.items()):
            if verdict["policy"] == "strict" and not verdict["reachable"]:
                errors.append(
                    f"{prop} {key}: strict target is unreachable "
                    f"({'; '.join(verdict['blockers']) or 'no route'}) — fix the "
                    "gold or mark it diagnostic_only with a reason")
    return errors
