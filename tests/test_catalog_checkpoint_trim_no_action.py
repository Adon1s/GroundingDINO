"""Focused regression pins for the 2026-09-08 catalog policy checkpoint: the
`route_override: no_action` on `dated_interior_trim` (decision D1).

Authorization: `reports/catalog_audit_approvals_v2.json`, disposition CAP-023,
approved by Steven on 2026-09-08. Human reasoning and the accepted losses are in
`docs/DECISION_RECORD_catalog_checkpoint_20260908.md`. Both records are untracked
program artifacts, so the invariants they protect are restated here rather than
merely cited.

What this change is, in one line: plain, basic or builder-grade trim presence
does not bill by itself (ruling 3), so the item stops billing while staying in
the retrieval pool and displaying at zero dollars.

What it deliberately is NOT:
  * not a retirement and not a split - the id survives, so no stale-id cutover
    and no new benchmark gold is owed;
  * not a claim edit - `atomic_claim` is untouched, so stored Terra verdicts stay
    claim-compatible;
  * not CAP-022 - the `require_any` subject gate was declined as unnecessary,
    because it would move a subject-mismatched row onto billable wood paneling.

Run:
    python -m pytest tests/test_catalog_checkpoint_trim_no_action.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.renovation_architecture.catalog_projection import (
    build_renovation_catalog_projection,
)

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_V2_PATH = ROOT / "tools" / "issue_catalog_kind_v2.json"
DECISIONS_PATH = ROOT / "tools" / "catalog_migrations" / "kind_v2_decisions.json"

TRIM_ID = "dated_interior_trim"

# The approved surface of the item after the op. Every other field is inherited
# from v1 and must not have moved.
APPROVED_TRIM_ROUTING = {
    "route_override": "no_action",
    "work_item_code": "TRIM_REPLACE",
    "cost": {"mode": "heuristic"},
}

# The claim as it stood before the checkpoint and must still stand after it.
APPROVED_TRIM_CLAIM = {
    "subject": "interior trim",
    "state": "plain, thin, or builder-grade trim package",
    "ontology_basis": "low_grade_material",
}


@pytest.fixture(scope="module")
def catalog():
    return json.loads(SHIPPED_V2_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def items(catalog):
    return {it["id"]: it for it in catalog["items"]}


@pytest.fixture(scope="module")
def projection(catalog):
    return build_renovation_catalog_projection(catalog, catalog_path=SHIPPED_V2_PATH)


@pytest.fixture(scope="module")
def decisions():
    return json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))


# ── the op landed, and landed as a carryover override ────────────────────────

def test_the_op_is_authored_as_a_carryover_override(decisions):
    """It must be authored on the decisions entry, not hand-edited into the
    generated catalog and not smuggled into the v1 catalog to dodge the v2
    authoring rules."""
    entry = next(e for e in decisions["entries"] if e["legacy_id"] == TRIM_ID)
    assert entry["change_type"] != "split", "trim is a carryover, not a split"
    successor = entry["successors"][0]
    assert successor["id"] == TRIM_ID, "a carryover keeps its id"
    assert successor["overrides"]["route_override"] == "no_action"


def test_the_v1_parent_carries_no_route_override():
    """The override belongs to the v2 decision. Authoring it in v1 would change
    legacy_v1 behaviour too, which the program forbids."""
    v1 = json.loads((ROOT / "tools" / "issue_catalog.json").read_text(encoding="utf-8"))
    parent = next(it for it in v1["items"] if it["id"] == TRIM_ID)
    assert "route_override" not in parent


def test_trim_routing_surface_matches_the_approval(items):
    item = items[TRIM_ID]
    for field, expected in APPROVED_TRIM_ROUTING.items():
        assert item.get(field) == expected, field


# ── the route is what stops the billing ──────────────────────────────────────

def test_trim_routes_to_no_action_by_explicit_override(projection):
    entry = projection["terminal_routes"][TRIM_ID]
    assert entry["route"] == "no_action"
    # Explicit product decision, NOT the inferred no-economics gap: the item
    # still has cost and a work item code, which is what the reason code says.
    assert entry["reason_code"] == "route_override_no_action"


def test_trim_has_no_work_policy(projection):
    """A no_action item is not projected into the work lane, which is the
    mechanism by which it stops billing. Without this, the route would be
    cosmetic."""
    assert TRIM_ID not in projection["work_policy"]


def test_the_override_is_neither_dead_nor_redundant(items):
    """`tools/catalog_validation.py` rejects a route_override that could not
    change anything. Trim keeps its economics precisely so the override remains
    meaningful; do not 'tidy them up'."""
    item = items[TRIM_ID]
    assert item.get("cost") or item.get("work_item_code"), "override would be redundant"
    assert item.get("drop_if_generic") is not True, "override would be dead"
    assert (item.get("estimate") or {}).get("strategy") != "inspect_only", "override would be dead"
    assert item.get("trade_bucket") == "trim_doors_windows"


def test_absolute_route_counts(projection):
    """The absolute pin, which moved here from the CAP-007 module: that module
    now states its own delta, since later approved changes legitimately move
    these totals."""
    assert projection["route_counts"]["no_action"] == 11
    assert projection["route_counts"]["work"] == 97
    assert sum(projection["route_counts"].values()) == 129


# ── what the change must NOT have done ───────────────────────────────────────

def test_the_claim_is_untouched(items):
    """A claim edit would move the projection fingerprint's claim surface and
    invalidate stored Terra verdicts for a different reason than the route does.
    The checkpoint deliberately changed routing only."""
    assert items[TRIM_ID]["atomic_claim"] == APPROVED_TRIM_CLAIM


def test_the_item_still_exists_and_is_still_retrievable(items):
    """The whole point of choosing a route over a retirement or a deny lever:
    nothing leaves the retrieval pool, so no observation migrates onto a
    billable neighbour. Retirement would have moved 55 of 108 corpus rows."""
    item = items[TRIM_ID]
    assert item["kind"] == "modernization"
    assert item["scene_groups"] == ["kitchen", "bathroom", "bedroom", "living_areas", "utility"]
    assert item["support_any"], "retrieval text must survive"
    assert item["embed_text"], "retrieval text must survive"


def test_no_require_any_gate_was_added(items):
    """CAP-022 was declined, not deferred-and-quietly-applied. Its subject gate
    would have moved 'The built-ins are dated.' onto billable dated_wood_paneling."""
    assert "require_any" not in items[TRIM_ID]


def test_package_affinity_is_retained_as_the_d7_landing_pad(items):
    """Dead configuration by design: a no_action condition forms no work item, so
    these routes are unreachable today. Decision D7 keeps the block as the
    landing pad for a possible zero-dollar-support mechanism, and no validator
    forbids the combination. Do not remove it as 'unused'."""
    affinity = items[TRIM_ID].get("package_affinity")
    assert isinstance(affinity, dict)
    assert sorted(affinity) == ["bathroom", "bedroom", "kitchen", "living"]
    assert all(e["package_role"] == "package_support" for e in affinity.values())


def test_trim_is_not_a_repair_support_marker_item(items):
    """Catalog 3.2's contextual-repair re-route must not pick this item up."""
    affinity = items[TRIM_ID].get("package_affinity") or {}
    assert not any(e.get("repair_support_when_driven") for e in affinity.values())


def test_the_repair_sibling_still_bills(items, projection):
    """Trim WEAR is a separate, still-billable concept. The checkpoint removed
    style/presence billing, not repair coverage."""
    assert projection["terminal_routes"]["baseboard_wear_scuffs"]["route"] == "work"
    assert items["baseboard_wear_scuffs"]["work_item_code"] == "TRIM_REPAIR"


def test_no_other_item_uses_the_trim_work_code(items):
    """TRIM_REPLACE has exactly one owner, so removing trim from the work lane
    cannot un-merge another item's deduplicated work."""
    owners = sorted(i for i, it in items.items() if it.get("work_item_code") == "TRIM_REPLACE")
    assert owners == [TRIM_ID]
