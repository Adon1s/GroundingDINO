"""CAP-007: the approved window-treatment split, pinned.

The catalog-audit human gate (2026-09-06) approved exactly one native catalog
change: `dated_window_treatment_valance` splits by subject into

  * `window_blinds_basic_or_plain`  — presence-only, `route_override: no_action`
  * `dated_window_valance_or_curtains` — the billable fabric treatment

The authorization is `reports/catalog_audit_approvals.json`
(`dispositions[CAP-007].approved_diff`); the successor detail is in
`reports/catalog_audit_redraft.json` `cap007`. Neither file is tracked in git,
so the invariants they authorize are pinned HERE as literals — this module is
the in-repo record of what was approved, and any drift is a deliberate edit
with a visible diff rather than a silent regeneration effect.

What each pin protects:

- Q-1 (all blinds in any condition are `no_action`): the blinds successor is a
  presence claim. No wear or damage word may appear on any of its authored
  surfaces, or the claim stops being presence-only and Terra is asked a
  condition question again.
- A-3: the fabric successor's `deny_any` ends with the lead-added stems
  `shower` and `tub`, which are what keep shower/tub curtain bullets off a
  billable item.
- A-4: the blinds successor stays visible (`defaultHidden` false,
  `display_class` marketability). The no_action route is the billing
  mechanism, not hiding.
- Task 4A bridge: both successors inherit the v1 parent's economics verbatim.
  On the blinds successor those fields are inert (projection applies
  `route_override` first) but must stay present — `catalog_validation` rejects
  an override on an item with neither cost nor work_item_code as redundant.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_catalog_cap007_window_treatment_split.py -q
"""
import json
from pathlib import Path

import pytest

from tools.catalog_validation import (
    ECONOMIC_FIELDS,
    load_shipped_catalog,
    load_shipped_catalog_v2,
    load_shipped_manifest,
)
from tools.renovation_architecture.catalog_projection import (
    build_renovation_catalog_projection,
)

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_V2_PATH = ROOT / "tools" / "issue_catalog_kind_v2.json"

LEGACY_ID = "dated_window_treatment_valance"
BLINDS_ID = "window_blinds_basic_or_plain"
FABRIC_ID = "dated_window_valance_or_curtains"

# The approved successor surfaces, verbatim from the approvals manifest's ops.
APPROVED_BLINDS = {
    "kind": "modernization",
    "severity": 1,
    "name": "Basic or Plain Window Blinds",
    "description": (
        "Window blinds or shades are a basic or plain style. Recorded as a "
        "visible presentation detail; blinds are not billed as renovation work."
    ),
    "embed_text": (
        "Basic or plain window blinds. Mini blinds, horizontal blinds, venetian "
        "blinds, vertical blinds, roller shades, and window shades in a plain "
        "builder-grade style. Blind and shade presence is recorded for "
        "presentation context only and is not renovation work."
    ),
    "support_any": [
        "blind", "mini blind", "horizontal blind", "venetian blind",
        "vertical blind", "roller shade", "window shade",
    ],
    "atomic_claim": {
        "subject": "window blinds or shades",
        "state": "basic or plain window blinds",
        "ontology_basis": "presentation_only",
    },
    "route_override": "no_action",
    "deny_any": [
        "valance", "cornice", "swag", "curtain", "drape", "drapery", "sheer",
        "fabric", "broken window", "fogged window", "failed seal",
    ],
}

APPROVED_FABRIC = {
    "kind": "modernization",
    "severity": 1,
    "name": "Dated Valance or Curtains",
    "description": (
        "A dated valance, cornice, swag, or curtain and drapery set reduces "
        "perceived condition; updating the fabric treatment improves appeal."
    ),
    "embed_text": (
        "Dated window valance, cornice, or swag. Dated curtains, drapes, "
        "draperies, and sheers hung at a window. Update dated fabric window "
        "treatments for a modern look."
    ),
    "support_any": [
        "valance", "cornice", "swag", "curtain", "drape", "drapery", "sheer",
        "fabric",
    ],
    "atomic_claim": {
        "subject": "window valance or curtains",
        "state": "dated valance or curtains",
        "ontology_basis": "dated_style",
    },
    "require_any": [
        "valance", "cornice", "swag", "curtain", "drape", "drapery", "sheer",
        "fabric",
    ],
    "deny_any": [
        "shower curtain", "improvised", "makeshift", "broken window",
        "fogged window", "failed seal", "shower", "tub",
    ],
}

# Structural fields both successors inherit from the v1 parent (they author
# none of these), pinned so an inheritance change is visible.
INHERITED_FROM_PARENT = ("trade_bucket", "scope", "tier", "defaultHidden",
                         "drop_if_generic", "category", "display_class",
                         "scene_groups")

# A presence claim may not describe a condition. Substrings, so inflections
# ("worn", "damaged", "deteriorating") are covered by their stem.
CONDITION_WORDS = (
    "wear", "worn", "damage", "broken", "bent", "torn", "crack", "deteriorat",
    "aged", "faded", "fading", "stain", "dirty", "soiled", "fail", "missing",
    "sagging", "warped", "old", "outdated",
)


@pytest.fixture(scope="module")
def v1_catalog():
    return load_shipped_catalog()


@pytest.fixture(scope="module")
def v2_catalog():
    return load_shipped_catalog_v2()


@pytest.fixture(scope="module")
def manifest():
    return load_shipped_manifest()


@pytest.fixture(scope="module")
def items(v2_catalog):
    return {it["id"]: it for it in v2_catalog["items"]}


@pytest.fixture(scope="module")
def parent(v1_catalog):
    return next(it for it in v1_catalog["items"] if it["id"] == LEGACY_ID)


@pytest.fixture(scope="module")
def projection(v2_catalog):
    return build_renovation_catalog_projection(v2_catalog, catalog_path=SHIPPED_V2_PATH)


# ── shape of the split ───────────────────────────────────────────────────────

def test_parent_id_is_gone_and_both_successors_exist(items):
    """The split parent leaves the catalog. There is no runtime alias, so a
    stored artifact carrying the old id must fail hard rather than resolve."""
    assert LEGACY_ID not in items
    assert BLINDS_ID in items
    assert FABRIC_ID in items


def test_catalog_has_129_items(v2_catalog):
    """128 before the split, 129 after: two successors replace one parent."""
    assert len(v2_catalog["items"]) == 129


def test_successors_sit_adjacently_in_the_parents_position(v2_catalog):
    """Generation walks the v1 catalog in order and emits an entry's successors
    in authored order, so the pair lands exactly where the parent was and no
    other item moves."""
    ids = [it["id"] for it in v2_catalog["items"]]
    assert ids[86:88] == [BLINDS_ID, FABRIC_ID]
    assert ids[85] == "stained_glass_or_vintage_light_fixture"
    assert ids[88] == "dated_wallpaper_present"


def test_manifest_records_the_split(manifest):
    entry = next(e for e in manifest["entries"] if e["legacy_id"] == LEGACY_ID)
    assert entry["change_type"] == "split"
    assert entry["deprecated"] is True
    assert entry["requires_re_resolution"] is True
    assert entry["successors"] == [
        {"id": BLINDS_ID, "kind": "modernization"},
        {"id": FABRIC_ID, "kind": "modernization"},
    ]


def test_the_split_is_same_kind_on_both_sides(items):
    """The first same-kind split in the decisions file: both successors stay
    modernization, so `estimate_scope` text matching and the package-category
    vocabulary behave exactly as they did for the parent."""
    assert items[BLINDS_ID]["kind"] == items[FABRIC_ID]["kind"] == "modernization"


# ── authored surfaces, verbatim ──────────────────────────────────────────────

@pytest.mark.parametrize("item_id,approved", [(BLINDS_ID, APPROVED_BLINDS),
                                              (FABRIC_ID, APPROVED_FABRIC)])
def test_authored_surfaces_match_the_approved_diff(items, item_id, approved):
    item = items[item_id]
    for field, expected in approved.items():
        assert item.get(field) == expected, field


def test_blinds_successor_has_no_require_any_gate(items):
    """Deliberate: a subject-generic bullet is not denied at the blinds
    successor either, so it stays detectable at no_action rather than becoming
    invisible. The parent had no require_any and none is authored."""
    assert "require_any" not in items[BLINDS_ID]


def test_fabric_deny_any_ends_with_the_lead_added_stems(items):
    """A-3. The drafted `shower curtain` guard matched only that literal
    phrase; the bare stems are what actually keep a shower or tub curtain
    bullet off the billable successor."""
    assert items[FABRIC_ID]["deny_any"][-2:] == ["shower", "tub"]


def test_fabric_successor_carries_no_route_override(items):
    """Only the blinds side is routed out of billing. The fabric successor
    keeps the parent's work route."""
    assert "route_override" not in items[FABRIC_ID]


# ── Q-1: the blinds claim is presence-only ───────────────────────────────────

@pytest.mark.parametrize("item_id", [BLINDS_ID, FABRIC_ID])
def test_no_condition_word_on_any_authored_surface(items, item_id):
    """Q-1 for the blinds successor, and the same discipline on the fabric one:
    neither claim may assert wear or damage. `dated` is a style word and is
    allowed; `broken window` and `failed seal` appear only in `deny_any`, which
    is a retrieval guardrail, not a claim surface."""
    item = items[item_id]
    surfaces = [item["name"], item["description"], item["embed_text"],
                item["atomic_claim"]["subject"], item["atomic_claim"]["state"]]
    haystack = " ".join(surfaces).lower()
    found = sorted({w for w in CONDITION_WORDS if w in haystack})
    assert found == [], (item_id, found)


def test_blinds_claim_is_presentation_only(items):
    assert items[BLINDS_ID]["atomic_claim"]["ontology_basis"] == "presentation_only"


def test_fabric_claim_is_dated_style(items):
    assert items[FABRIC_ID]["atomic_claim"]["ontology_basis"] == "dated_style"


# ── inheritance: structure and economics ─────────────────────────────────────

@pytest.mark.parametrize("item_id", [BLINDS_ID, FABRIC_ID])
def test_structural_fields_are_inherited_from_the_v1_parent(items, parent, item_id):
    item = items[item_id]
    for field in INHERITED_FROM_PARENT:
        assert item.get(field) == parent.get(field), (item_id, field)


@pytest.mark.parametrize("item_id", [BLINDS_ID, FABRIC_ID])
def test_economics_are_inherited_verbatim(items, parent, item_id):
    """Task 4A bridge. Present fields carry over byte-identically and absent
    fields stay absent; a successor override can never rewrite one."""
    item = items[item_id]
    assert item["pricing_status"] == "inherited_from_split_parent"
    for field in ECONOMIC_FIELDS:
        assert (field in parent) == (field in item), (item_id, field)
        if field in parent:
            assert item[field] == parent[field], (item_id, field)


def test_the_inherited_economics_are_the_expected_ones(items):
    """Named explicitly so the next reader does not have to open the v1
    catalog: a heuristic cost and a work item code, no estimate block and no
    package affinity."""
    for item_id in (BLINDS_ID, FABRIC_ID):
        item = items[item_id]
        assert item["cost"] == {"mode": "heuristic"}
        assert item["work_item_code"] == "WINDOW_TREATMENT_UPDATE"
        assert item["package_role"] == "ignore"
        assert "estimate" not in item
        assert "package_affinity" not in item


def test_neither_successor_carries_a_repair_support_marker(items):
    """CCF-7: the parent is not among the ten `repair_support_when_driven`
    items, so the split is not blocked by the marker section and neither
    successor gains package metadata."""
    for item_id in (BLINDS_ID, FABRIC_ID):
        assert not items[item_id].get("package_affinity")


# ── projection: the billing consequence ──────────────────────────────────────

def test_blinds_successor_routes_to_no_action_by_override(projection):
    """The whole point of the split. The inherited cost and work item code are
    inert because projection applies `route_override` before the costing route,
    and they are also what keeps the override from being flagged redundant."""
    entry = projection["terminal_routes"][BLINDS_ID]
    assert entry["route"] == "no_action"
    assert entry["reason_code"] == "route_override_no_action"
    assert BLINDS_ID not in projection["work_policy"]


def test_fabric_successor_keeps_the_work_route(projection):
    entry = projection["terminal_routes"][FABRIC_ID]
    assert entry["route"] == "work"
    assert FABRIC_ID in projection["work_policy"]
    assert projection["work_policy"][FABRIC_ID]["action_code"] == "WINDOW_TREATMENT_UPDATE"


def test_the_split_moved_exactly_one_item_into_no_action(projection):
    """Baseline had 9 no_action and 98 work over 128 items. The split adds one
    no_action item; the parent's work route survives on the fabric successor,
    so `work` is unchanged."""
    assert projection["route_counts"]["no_action"] == 10
    assert projection["route_counts"]["work"] == 98
    assert sum(projection["route_counts"].values()) == 129


# ── the deprecated parent is not reachable anywhere ──────────────────────────

def test_no_generated_artifact_still_offers_the_parent_as_an_item():
    """The manifest is an audit record, not a runtime alias table: the parent
    may appear there as a `legacy_id`, but nowhere as a live catalog item."""
    catalog = json.loads(SHIPPED_V2_PATH.read_text(encoding="utf-8"))
    assert all(it["id"] != LEGACY_ID for it in catalog["items"])
    manifest_path = ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    successor_ids = {s["id"] for e in manifest["entries"] for s in e["successors"]}
    assert LEGACY_ID not in successor_ids
