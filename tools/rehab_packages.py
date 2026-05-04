"""
tools/rehab_packages.py

Deterministic package inference and reconciliation for renovation_estimate_v4.

Public API:
  - infer_packages(candidates, room_surrogates, issue_catalog) -> list[dict]
        Walk per-room candidates and emit kitchen / bathroom / room packages.
  - reconcile_packages_and_estimate_units(groups_out, packages) -> dict
        Normalize line items into per-billable-member child unit records,
        absorb children into matching packages all-or-nothing, recompute
        retained group totals using v3's stack-behavior rule, emit audit
        + final_rehab. Mutates groups_out and packages in place.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from tools.renovation_estimate import GROUP_BUDGET_CAPS, EstimateCandidate

logger = logging.getLogger(__name__)


# ── Package level cost ranges (name, low, high) ─────────────────────────────
KITCHEN_MINOR_REPAIR    = ("kitchen_minor_repair",    1_000,  5_000)
KITCHEN_REFRESH         = ("kitchen_refresh",         6_000, 18_000)
KITCHEN_PARTIAL_REHAB   = ("kitchen_partial_rehab",  15_000, 35_000)
KITCHEN_FULL_REHAB      = ("kitchen_full_rehab",     30_000, 70_000)

BATHROOM_MINOR_REPAIR   = ("bathroom_minor_repair",     500,  3_000)
BATHROOM_REFRESH        = ("bathroom_refresh",        3_000, 10_000)
BATHROOM_PARTIAL_REHAB  = ("bathroom_partial_rehab",  8_000, 20_000)
BATHROOM_FULL_REHAB     = ("bathroom_full_rehab",    15_000, 35_000)

ROOM_REFRESH            = ("room_refresh",              750,  4_000)
ROOM_REPAIR_HEAVY       = ("room_repair_heavy",       3_000, 10_000)


_QUALIFIED_POSTURES = frozenset({
    "repair", "replace", "keep_default",                    # Pass 2f outputs
    "repair_only", "replace_only", "repair_or_replace",     # catalog strategy fallbacks
})
# Implicitly excludes: "inspect", "inspect_only", "service_only", None.

_OPTIONAL_TIER = "optional"


def _effective_posture(candidate: EstimateCandidate) -> Optional[str]:
    """Return effective_posture if set, else fall back to catalog strategy.

    `compute_renovation_estimate` only populates `effective_posture` on its own
    resolved-candidate list, so candidates passed straight into `infer_packages`
    may still have `effective_posture is None`. Falling back to the catalog
    strategy mirrors v3's `resolve_effective_posture` keep-default semantics.
    """
    if candidate.effective_posture:
        return candidate.effective_posture
    return candidate.estimate_meta.strategy


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _split_integer(amount: int, n: int) -> List[int]:
    """Split amount into n integers summing exactly to amount.

    Distributes any remainder one unit at a time to the first k entries.
    Example: _split_integer(100, 3) == [34, 33, 33]; sum == 100.
    """
    if n <= 0:
        return []
    base, remainder = divmod(amount, n)
    return [base + 1 if i < remainder else base for i in range(n)]


def classify_component(candidate: EstimateCandidate) -> Optional[str]:
    """Map a candidate to a component class for package inference.

    Returns one of: cabinets, counter, kitchen_finish, appliance, vanity, tile,
    tub_shower, fixture, bath_finish, flooring, plumbing, electrical, moisture,
    paint — or None when not relevant for any package rule.
    """
    cat_id = (candidate.catalog_item_id or "").lower()
    trade = candidate.trade_bucket or ""

    if trade == "kitchen_cabinets_counters":
        if "appliance" in cat_id:
            return "appliance"
        if "cabinet" in cat_id:
            return "cabinets"
        if "counter" in cat_id:
            return "counter"
        return "kitchen_finish"

    if trade == "bathroom_fixtures_tile":
        if "vanity" in cat_id:
            return "vanity"
        if "tile" in cat_id or "grout" in cat_id:
            return "tile"
        if "tub" in cat_id or "shower" in cat_id:
            return "tub_shower"
        if "fixture" in cat_id or "faucet" in cat_id:
            return "fixture"
        return "bath_finish"

    if trade == "flooring":
        return "flooring"
    if trade == "plumbing":
        return "plumbing"
    if trade == "electrical":
        return "electrical"
    if trade == "moisture_mold":
        return "moisture"
    if trade == "paint_drywall":
        return "paint"
    return None


def is_defect_driver(
    candidate: EstimateCandidate,
    catalog_item: Dict[str, Any],
) -> bool:
    """Real defect that can drive a partial/full rehab package."""
    if candidate.is_valid_detection is False:
        return False
    if catalog_item.get("kind") != "defect":
        return False
    if catalog_item.get("tier") == _OPTIONAL_TIER:
        return False
    if _effective_posture(candidate) not in _QUALIFIED_POSTURES:
        return False
    return True


def is_dated_cosmetic_evidence(
    candidate: EstimateCandidate,
    catalog_item: Dict[str, Any],
) -> bool:
    """Upgrade-kind cosmetic/opportunity item, used ONLY as refresh evidence."""
    if candidate.is_valid_detection is False:
        return False
    if catalog_item.get("tier") == _OPTIONAL_TIER:
        return False
    if _effective_posture(candidate) not in _QUALIFIED_POSTURES:
        return False
    if catalog_item.get("kind") != "upgrade":
        return False
    return catalog_item.get("category") in ("cosmetic", "opportunity")


# ═══════════════════════════════════════════════════════════════════════════
# Package inference
# ═══════════════════════════════════════════════════════════════════════════

def infer_packages(
    candidates: List[EstimateCandidate],
    room_surrogates: List[Dict[str, Any]],
    issue_catalog: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Walk per-room candidates, emit kitchen/bathroom/room packages."""
    catalog_lookup: Dict[str, Dict[str, Any]] = {}
    for item in (issue_catalog.get("items") or []):
        if isinstance(item, dict) and item.get("id"):
            catalog_lookup[item["id"]] = item

    by_surrogate: Dict[str, List[EstimateCandidate]] = {}
    for c in candidates:
        if c.room_surrogate_id:
            by_surrogate.setdefault(c.room_surrogate_id, []).append(c)

    packages: List[Dict[str, Any]] = []
    for surrogate in room_surrogates or []:
        sid = surrogate.get("room_surrogate_id")
        if not sid:
            continue
        scene = surrogate.get("scene") or ""
        room_candidates = by_surrogate.get(sid, [])
        if not room_candidates:
            continue

        if scene == "kitchen":
            pkg = _infer_kitchen_package(sid, room_candidates, catalog_lookup)
        elif scene == "bathroom":
            pkg = _infer_bathroom_package(sid, room_candidates, catalog_lookup)
        else:
            pkg = _infer_room_refresh(sid, scene, room_candidates, catalog_lookup)

        if pkg is not None:
            packages.append(pkg)

    return packages


def _build_package(
    spec: Tuple[str, int, int],
    *,
    room_surrogate_id: str,
    estimate_group: str,
    supporting_candidates: List[EstimateCandidate],
    trigger_reason: str,
    decision_notes: List[str],
) -> Dict[str, Any]:
    pkg_type, cost_low, cost_high = spec
    issue_ids: List[str] = []
    cat_ids: List[str] = []
    for c in supporting_candidates:
        for iid in c.issue_ids:
            if iid not in issue_ids:
                issue_ids.append(iid)
        if c.catalog_item_id and c.catalog_item_id not in cat_ids:
            cat_ids.append(c.catalog_item_id)
    return {
        "package_id": f"{pkg_type}__{room_surrogate_id}",
        "package_type": pkg_type,
        "room_surrogate_id": room_surrogate_id,
        "estimate_group": estimate_group,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "cost_midpoint": (cost_low + cost_high) // 2,
        "absorbed_unit_member_refs": [],
        "absorbed_total_low": 0,
        "absorbed_total_high": 0,
        "replacement_delta_low": 0,
        "replacement_delta_high": 0,
        "supporting_issue_ids": issue_ids,
        "supporting_catalog_item_ids": cat_ids,
        "trigger_reason": trigger_reason,
        "level_decision_notes": list(decision_notes),
    }


def _bucket_by_component(
    candidates: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, List[EstimateCandidate]], List[EstimateCandidate], List[EstimateCandidate]]:
    """Return (defect_drivers_by_component, dated_evidence, all_defect_drivers)."""
    drivers: Dict[str, List[EstimateCandidate]] = {}
    dated: List[EstimateCandidate] = []
    all_drivers: List[EstimateCandidate] = []
    for c in candidates:
        cat = catalog_lookup.get(c.catalog_item_id or "", {})
        if is_defect_driver(c, cat):
            all_drivers.append(c)
            comp = classify_component(c)
            if comp:
                drivers.setdefault(comp, []).append(c)
        elif is_dated_cosmetic_evidence(c, cat):
            dated.append(c)
    return drivers, dated, all_drivers


def _infer_kitchen_package(
    room_surrogate_id: str,
    candidates: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    drivers, dated, all_drivers = _bucket_by_component(candidates, catalog_lookup)
    notes: List[str] = []

    cabinets   = drivers.get("cabinets",   [])
    counter    = drivers.get("counter",    [])
    flooring   = drivers.get("flooring",   [])
    appliance  = drivers.get("appliance",  [])
    plumbing   = drivers.get("plumbing",   [])
    electrical = drivers.get("electrical", [])

    cabinets_sev3 = [c for c in cabinets if (c.severity or 0) >= 3]
    if cabinets_sev3 and counter and flooring and (appliance or plumbing or electrical):
        return _build_package(
            KITCHEN_FULL_REHAB,
            room_surrogate_id=room_surrogate_id,
            estimate_group="kitchen",
            supporting_candidates=cabinets_sev3 + counter + flooring + appliance + plumbing + electrical,
            trigger_reason="cabinets_sev3 + counter + flooring + (appliance|plumbing|electrical)",
            decision_notes=["full_rehab matched"],
        )
    notes.append("full_rehab not met")

    if cabinets and (counter or flooring):
        return _build_package(
            KITCHEN_PARTIAL_REHAB,
            room_surrogate_id=room_surrogate_id,
            estimate_group="kitchen",
            supporting_candidates=cabinets + counter + flooring,
            trigger_reason="cabinets + (counter|flooring)",
            decision_notes=notes + ["partial_rehab matched"],
        )
    notes.append("partial_rehab not met")

    if len(dated) >= 2 and len(all_drivers) >= 1:
        return _build_package(
            KITCHEN_REFRESH,
            room_surrogate_id=room_surrogate_id,
            estimate_group="kitchen",
            supporting_candidates=dated + all_drivers,
            trigger_reason="2+ dated_evidence + 1+ defect_driver",
            decision_notes=notes + ["refresh matched"],
        )
    notes.append("refresh not met")

    if 1 <= len(all_drivers) <= 2:
        return _build_package(
            KITCHEN_MINOR_REPAIR,
            room_surrogate_id=room_surrogate_id,
            estimate_group="kitchen",
            supporting_candidates=all_drivers,
            trigger_reason="1-2 isolated defect drivers",
            decision_notes=notes + ["minor_repair matched"],
        )
    return None


def _infer_bathroom_package(
    room_surrogate_id: str,
    candidates: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    drivers, dated, all_drivers = _bucket_by_component(candidates, catalog_lookup)
    notes: List[str] = []

    vanity     = drivers.get("vanity",     [])
    tile       = drivers.get("tile",       [])
    flooring   = drivers.get("flooring",   [])
    tub_shower = drivers.get("tub_shower", [])
    fixture    = drivers.get("fixture",    [])
    moisture   = drivers.get("moisture",   [])
    plumbing   = drivers.get("plumbing",   [])

    if vanity and tile and flooring and (tub_shower or fixture or moisture or plumbing):
        return _build_package(
            BATHROOM_FULL_REHAB,
            room_surrogate_id=room_surrogate_id,
            estimate_group="bathroom",
            supporting_candidates=vanity + tile + flooring + tub_shower + fixture + moisture + plumbing,
            trigger_reason="vanity + tile + flooring + (tub_shower|fixture|moisture|plumbing)",
            decision_notes=["full_rehab matched"],
        )
    notes.append("full_rehab not met")

    if (vanity or tile) and (flooring or tub_shower):
        return _build_package(
            BATHROOM_PARTIAL_REHAB,
            room_surrogate_id=room_surrogate_id,
            estimate_group="bathroom",
            supporting_candidates=vanity + tile + flooring + tub_shower,
            trigger_reason="(vanity|tile) + (flooring|tub_shower)",
            decision_notes=notes + ["partial_rehab matched"],
        )
    notes.append("partial_rehab not met")

    if len(dated) >= 2 and len(all_drivers) >= 1:
        return _build_package(
            BATHROOM_REFRESH,
            room_surrogate_id=room_surrogate_id,
            estimate_group="bathroom",
            supporting_candidates=dated + all_drivers,
            trigger_reason="2+ dated_evidence + 1+ defect_driver",
            decision_notes=notes + ["refresh matched"],
        )
    notes.append("refresh not met")

    if 1 <= len(all_drivers) <= 2:
        return _build_package(
            BATHROOM_MINOR_REPAIR,
            room_surrogate_id=room_surrogate_id,
            estimate_group="bathroom",
            supporting_candidates=all_drivers,
            trigger_reason="1-2 isolated defect drivers",
            decision_notes=notes + ["minor_repair matched"],
        )
    return None


def _infer_room_refresh(
    room_surrogate_id: str,
    scene: str,
    candidates: List[EstimateCandidate],
    catalog_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Room refresh / repair-heavy for non-kitchen, non-bathroom surrogates."""
    _drivers, _dated, all_drivers = _bucket_by_component(candidates, catalog_lookup)

    minor_sev = [c for c in all_drivers if (c.severity or 0) <= 2]
    medium_sev = [c for c in all_drivers if (c.severity or 0) == 3]

    if len(minor_sev) >= 2 and len(medium_sev) >= 1:
        return _build_package(
            ROOM_REPAIR_HEAVY,
            room_surrogate_id=room_surrogate_id,
            estimate_group="other",
            supporting_candidates=minor_sev + medium_sev,
            trigger_reason="2+ minor_severity + 1+ medium_severity defect drivers",
            decision_notes=["room_repair_heavy matched"],
        )

    qualified_for_refresh: List[EstimateCandidate] = []
    for c in all_drivers:
        cat = catalog_lookup.get(c.catalog_item_id or "", {})
        if (c.severity or 0) <= 2 or cat.get("category") == "cosmetic":
            qualified_for_refresh.append(c)
    if len(qualified_for_refresh) >= 3:
        return _build_package(
            ROOM_REFRESH,
            room_surrogate_id=room_surrogate_id,
            estimate_group="other",
            supporting_candidates=qualified_for_refresh,
            trigger_reason="3+ qualified non-optional cosmetic-or-minor-severity defect drivers",
            decision_notes=["room_refresh matched"],
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Reconciliation
# ═══════════════════════════════════════════════════════════════════════════

def reconcile_packages_and_estimate_units(
    groups_out: List[Dict[str, Any]],
    packages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reconcile packages against group line items.

    Mutations (in place):
      - Each line_item gets `unit_member_allocations` (list of child records).
      - Absorbed children get `absorbed_by_package_id` set.
      - Parent unit_members of absorbed children get `absorbed_by_package_id` mirrored.
      - Each package gets `absorbed_unit_member_refs`, `absorbed_total_low/high`,
        `replacement_delta_low/high` populated.

    Returns audit dict with totals, warnings, and `final_rehab`.
    """
    # Phase A: normalize line items into child records, indexed by group
    children_by_group: Dict[str, List[Dict[str, Any]]] = {}
    for group in groups_out or []:
        group_name = group.get("group", "other")
        group_children: List[Dict[str, Any]] = []
        for line_item in group.get("line_items", []) or []:
            children = _normalize_line_item(line_item, group_name)
            line_item["unit_member_allocations"] = children
            group_children.extend(children)
        children_by_group.setdefault(group_name, []).extend(group_children)

    all_children = [c for lst in children_by_group.values() for c in lst]

    # Phase B: absorb children into packages (first-come-first-absorbed)
    for pkg in packages or []:
        pkg.setdefault("absorbed_unit_member_refs", [])
        pkg["absorbed_total_low"] = 0
        pkg["absorbed_total_high"] = 0
        pkg_supporting_issue_ids = set(pkg.get("supporting_issue_ids") or [])
        pkg_room_id = pkg.get("room_surrogate_id")
        pkg_id = pkg.get("package_id")

        for child in all_children:
            if child.get("absorbed_by_package_id") is not None:
                continue
            if child.get("room_surrogate_id") != pkg_room_id:
                continue
            child_issue_ids = set(child.get("issue_ids") or [])
            if not (child_issue_ids & pkg_supporting_issue_ids):
                continue
            child["absorbed_by_package_id"] = pkg_id
            pkg["absorbed_unit_member_refs"].append({
                "child_id": child["child_id"],
                "parent_line_item_id": child["parent_line_item_id"],
                "allocated_low": child["allocated_low"],
                "allocated_high": child["allocated_high"],
            })
            pkg["absorbed_total_low"] += child["allocated_low"]
            pkg["absorbed_total_high"] += child["allocated_high"]
            _mark_parent_member_absorbed(groups_out, child, pkg_id)

    # Phase C: per-package replacement_delta + warnings
    warnings: List[Dict[str, Any]] = []
    for pkg in packages or []:
        delta_low = pkg["cost_low"] - pkg["absorbed_total_low"]
        delta_high = pkg["cost_high"] - pkg["absorbed_total_high"]
        pkg["replacement_delta_low"] = delta_low
        pkg["replacement_delta_high"] = delta_high
        if delta_low < 0:
            warnings.append({
                "code": "package_total_below_absorbed_total_low",
                "package_id": pkg["package_id"],
                "absorbed_total_low": pkg["absorbed_total_low"],
                "package_cost_low": pkg["cost_low"],
            })
        if delta_high < 0:
            warnings.append({
                "code": "package_total_below_absorbed_total_high",
                "package_id": pkg["package_id"],
                "absorbed_total_high": pkg["absorbed_total_high"],
                "package_cost_high": pkg["cost_high"],
            })

    # Phase D: recompute retained group totals using v3's stack rule
    retained_group_totals: List[Dict[str, Any]] = []
    retained_total_low = 0
    retained_total_high = 0
    for group in groups_out or []:
        group_name = group.get("group", "other")
        group_children = children_by_group.get(group_name, [])
        retained_children = [c for c in group_children if c.get("absorbed_by_package_id") is None]
        rt_low, rt_high = _recompute_retained_group(group_name, retained_children)
        retained_group_totals.append({"group": group_name, "low": rt_low, "high": rt_high})
        retained_total_low += rt_low
        retained_total_high += rt_high

    # Phase E: assemble totals + final_rehab
    package_total_low = sum(int(pkg.get("cost_low") or 0) for pkg in packages or [])
    package_total_high = sum(int(pkg.get("cost_high") or 0) for pkg in packages or [])
    absorbed_total_low = sum(int(pkg.get("absorbed_total_low") or 0) for pkg in packages or [])
    absorbed_total_high = sum(int(pkg.get("absorbed_total_high") or 0) for pkg in packages or [])
    absorbed_member_count = sum(len(pkg.get("absorbed_unit_member_refs") or []) for pkg in packages or [])

    existing_risk_exposure_high = sum(
        int(group.get("risk_exposure_high") or 0) for group in (groups_out or [])
    )

    final_low = retained_total_low + package_total_low
    final_high = retained_total_high + package_total_high + existing_risk_exposure_high
    final_mid = (final_low + final_high) // 2

    return {
        "absorbed_total_low": absorbed_total_low,
        "absorbed_total_high": absorbed_total_high,
        "package_total_low": package_total_low,
        "package_total_high": package_total_high,
        "net_delta_low": package_total_low - absorbed_total_low,
        "net_delta_high": package_total_high - absorbed_total_high,
        "absorbed_member_count": absorbed_member_count,
        "package_count": len(packages or []),
        "retained_group_totals": retained_group_totals,
        "reconciliation_warnings": warnings,
        "final_rehab": {
            "low": final_low,
            "high": final_high,
            "midpoint": final_mid,
            "source": "renovation_estimate_v4",
        },
    }


def _normalize_line_item(line_item: Dict[str, Any], group_name: str) -> List[Dict[str, Any]]:
    """Build child unit records for a single line item.

    One child per billable unit_member; cost is split exactly via _split_integer.
    Line items with no billable members get one synthesized child carrying the
    full parent cost so reconciliation never loses a line item.
    """
    parent_id = line_item.get("estimate_unit_id") or ""
    cost_low = int(line_item.get("cost_low") or 0)
    cost_high = int(line_item.get("cost_high") or 0)
    stack_behavior = line_item.get("stack_behavior") or "sum"
    unit_members = line_item.get("unit_members") or []
    billable = [m for m in unit_members if m.get("counts_toward_estimate")]

    if not billable:
        return [{
            "child_id": f"{parent_id}::__synthetic__",
            "parent_line_item_id": parent_id,
            "group": group_name,
            "room_surrogate_id": line_item.get("room_surrogate_id") or "",
            "issue_ids": list(line_item.get("source_issue_ids") or []),
            "scope_keys": [],
            "stack_behavior": stack_behavior,
            "allocated_low": cost_low,
            "allocated_high": cost_high,
            "absorbed_by_package_id": None,
        }]

    n = len(billable)
    alloc_low = _split_integer(cost_low, n)
    alloc_high = _split_integer(cost_high, n)
    children: List[Dict[str, Any]] = []
    for i, member in enumerate(billable):
        unit_key = member.get("unit_key") or f"member_{i}"
        children.append({
            "child_id": f"{parent_id}::{unit_key}",
            "parent_line_item_id": parent_id,
            "group": group_name,
            "room_surrogate_id": member.get("room_surrogate_id") or "",
            "issue_ids": list(member.get("issue_ids") or []),
            "scope_keys": list(member.get("estimate_scope_keys") or []),
            "stack_behavior": stack_behavior,
            "allocated_low": alloc_low[i],
            "allocated_high": alloc_high[i],
            "absorbed_by_package_id": None,
        })
    return children


def _mark_parent_member_absorbed(
    groups_out: List[Dict[str, Any]],
    child: Dict[str, Any],
    package_id: str,
) -> None:
    """Mirror absorbed_by_package_id onto the matching parent unit_member."""
    parent_id = child["parent_line_item_id"]
    child_id = child["child_id"]
    member_unit_key = child_id.split("::", 1)[-1] if "::" in child_id else None
    if not member_unit_key or member_unit_key == "__synthetic__":
        return
    for group in groups_out or []:
        for li in group.get("line_items", []) or []:
            if li.get("estimate_unit_id") != parent_id:
                continue
            for m in li.get("unit_members", []) or []:
                if m.get("unit_key") == member_unit_key:
                    m["absorbed_by_package_id"] = package_id
            return


def _recompute_retained_group(
    group_name: str,
    retained_children: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """Apply v3's stack-behavior rule to retained children only."""
    if not retained_children:
        return (0, 0)
    raw_low = sum(c["allocated_low"] for c in retained_children)
    raw_high = sum(c["allocated_high"] for c in retained_children)
    behaviors = {c.get("stack_behavior") or "sum" for c in retained_children}
    if "max_only" in behaviors:
        best = max(retained_children, key=lambda c: c["allocated_high"])
        return (best["allocated_low"], best["allocated_high"])
    if "group_cap" in behaviors:
        cap_low, cap_high = GROUP_BUDGET_CAPS.get(group_name, (200, 5_000))
        capped_low = min(raw_low, cap_low)
        capped_high = min(raw_high, cap_high)
        best = max(retained_children, key=lambda c: c["allocated_high"])
        capped_low = max(capped_low, best["allocated_low"])
        capped_high = max(capped_high, best["allocated_high"])
        return (capped_low, capped_high)
    return (raw_low, raw_high)
