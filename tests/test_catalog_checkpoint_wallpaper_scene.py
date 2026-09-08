"""Focused regression pins for the 2026-09-08 catalog policy checkpoint: the
bathroom scene exclusion on `dated_wallpaper_present` (decision D4, CCF-13).

Authorization: `reports/catalog_audit_approvals_v2.json`, disposition CCF-13,
approved by Steven on 2026-09-08. Reasoning and the disclosed residual are in
`docs/DECISION_RECORD_catalog_checkpoint_20260908.md`.

The defect: one bathroom wall was charged twice. `dated_wallpaper_present`
(generic, `interior_finishes`, `per_scope`) and `dated_bathroom_wallpaper`
(bathroom-specific, `paint_drywall`, `per_bathroom`) both bill
`WALLPAPER_REMOVE_PAINT`. The work-item dedup key is
(action_code, trade_bucket, unit_policy, billable_unit), and those differ, so
nothing collapses them; and the generic item has no bathroom `package_affinity`,
so nothing absorbs it either. Reproduced byte-identically under 3.2 on
redfin_25809814 photo_019.

The fix: `scene_groups` is a retrieval PRE-filter, so removing `bathroom` makes
the generic item unreachable in bathrooms, leaving the bathroom-specific item as
the single owner. Nothing is stranded, because the generic item has no bathroom
affinity route to lose.

Not fixed through the dedup key (code, outside this program) and not through
`package_affinity` (economic, never authorable).

Run:
    python -m pytest tests/test_catalog_checkpoint_wallpaper_scene.py -q
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

GENERIC_ID = "dated_wallpaper_present"
BATHROOM_ID = "dated_bathroom_wallpaper"
SHARED_ACTION_CODE = "WALLPAPER_REMOVE_PAINT"

APPROVED_GENERIC_SCENES = ["kitchen", "bedroom", "living_areas", "utility"]


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


def test_the_op_is_authored_as_a_carryover_override(decisions):
    entry = next(e for e in decisions["entries"] if e["legacy_id"] == GENERIC_ID)
    assert entry["change_type"] != "split"
    successor = entry["successors"][0]
    assert successor["id"] == GENERIC_ID
    assert successor["overrides"]["scene_groups"] == APPROVED_GENERIC_SCENES


def test_generic_item_no_longer_reaches_bathrooms(items):
    item = items[GENERIC_ID]
    assert item["scene_groups"] == APPROVED_GENERIC_SCENES
    assert "bathroom" not in item["scene_groups"]


def test_the_bathroom_specific_item_is_untouched(items, projection):
    """The remaining owner must still be able to bill the work; the fix removes a
    duplicate charge, not the coverage."""
    item = items[BATHROOM_ID]
    assert item["scene_groups"] == ["bathroom"]
    assert item["work_item_code"] == SHARED_ACTION_CODE
    assert projection["terminal_routes"][BATHROOM_ID]["route"] == "work"
    assert item["package_affinity"]["bathroom"]["package_type"] == "bathroom_modernization"


def test_exactly_one_item_can_bill_wallpaper_removal_in_a_bathroom(items):
    """The invariant the whole change exists to establish."""
    owners = sorted(
        item_id for item_id, it in items.items()
        if it.get("work_item_code") == SHARED_ACTION_CODE
        and "bathroom" in (it.get("scene_groups") or [])
    )
    assert owners == [BATHROOM_ID]


def test_nothing_was_stranded_by_the_exclusion(items):
    """Removing a scene from an item that HAD a package_affinity for that room
    would strand the route. The generic item never had a bathroom affinity, which
    is exactly why its bathroom billing was never absorbed."""
    affinity = items[GENERIC_ID].get("package_affinity") or {}
    assert "bathroom" not in affinity
    assert sorted(affinity) == ["bedroom", "kitchen", "living"]


def test_the_generic_item_still_covers_its_other_scenes(items, projection):
    """The exclusion is bathroom-only; the item must keep working elsewhere."""
    item = items[GENERIC_ID]
    for scene in ("kitchen", "bedroom", "living_areas", "utility"):
        assert scene in item["scene_groups"]
    assert projection["terminal_routes"][GENERIC_ID]["route"] == "work"
    assert item["work_item_code"] == SHARED_ACTION_CODE


def test_the_two_items_still_differ_in_the_dedup_dimensions(items):
    """Documents WHY a scene exclusion was needed at all: the dedup key cannot
    collapse these two, so they had to be separated at retrieval instead. If a
    later change aligns these fields, the scene exclusion becomes redundant and
    this test should be revisited rather than deleted."""
    generic, bathroom = items[GENERIC_ID], items[BATHROOM_ID]
    assert generic["work_item_code"] == bathroom["work_item_code"] == SHARED_ACTION_CODE
    assert generic["trade_bucket"] != bathroom["trade_bucket"]
    assert (generic.get("estimate") or {}).get("unit_policy") != \
        (bathroom.get("estimate") or {}).get("unit_policy")


def test_the_bathroom_items_require_any_is_the_disclosed_residual(items):
    """The accepted residual: a bathroom bullet saying 'wallcovering' but not
    'wallpaper' fails this gate and now has no owner, because the generic item no
    longer reaches bathrooms. Recorded so the gap is found deliberately rather
    than rediscovered as a bug."""
    require_any = items[BATHROOM_ID].get("require_any") or []
    assert require_any, "the residual depends on this gate existing"
    assert not any("wallcovering" in term for term in require_any)
