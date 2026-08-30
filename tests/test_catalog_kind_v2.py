"""observation-kind-v2 catalog, migration manifest, and generator parity.

The v2 catalog (tools/issue_catalog_kind_v2.json) is generated from
tools/catalog_migrations/kind_v2_decisions.json by
scripts/migrate_catalog_kind_v2.py. These tests pin:

1. the shipped v2 catalog and manifest validate clean;
2. root metadata (version 3.0, ontology stamp, publishable status);
3. the versioned kind vocabulary (three kinds only under the v2 stamp);
4. atomic-claim rules and Task 4A inherited-economics rules (every split
   successor carries its parent's economic fields byte-identically);
5. manifest integrity over all 107 legacy ids;
6. byte-parity: regenerating from the decisions file reproduces the committed
   catalog + manifest exactly.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tools.catalog_validation import (
    ECONOMIC_FIELDS,
    load_shipped_catalog,
    load_shipped_catalog_v2,
    load_shipped_manifest,
    validate_issue_catalog,
    validate_migration_manifest,
)
from tools.observation_kinds import OBSERVATION_KINDS, ONTOLOGY_VERSION
from tools.rehab_packages import REPAIR_SUPPORT_MARKER, paired_repair_package_type

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v1_catalog():
    return load_shipped_catalog()


@pytest.fixture(scope="module")
def v2_catalog():
    return load_shipped_catalog_v2()


@pytest.fixture(scope="module")
def manifest():
    return load_shipped_manifest()


def _minimal_v2_catalog(item_overrides=None, root_overrides=None):
    item = {
        "id": "carpet_worn",
        "name": "Worn Carpet",
        "kind": "degradation",
        "severity": 1,
        "trade_bucket": "flooring",
        "scope": "replace",
        "tier": "work",
        "defaultHidden": False,
        "description": "Carpet shows visible wear.",
        "embed_text": "Worn carpet.",
        "scene_groups": ["living_areas"],
        "atomic_claim": {
            "subject": "carpet",
            "state": "visible wear",
            "ontology_basis": "visible_deterioration",
        },
    }
    item.update(item_overrides or {})
    catalog = {
        "version": "3.2",
        "ontology_version": ONTOLOGY_VERSION,
        "publication_status": "blocked_pending_pricing",
        "trade_buckets": [{"id": "flooring", "name": "Flooring"}],
        "items": [item],
    }
    catalog.update(root_overrides or {})
    return catalog


# ── shipped artifacts ────────────────────────────────────────────────────────

def test_shipped_v2_catalog_has_no_errors(v2_catalog):
    result = validate_issue_catalog(v2_catalog)
    assert result.errors == []


def test_shipped_v2_root_metadata(v2_catalog):
    assert v2_catalog["version"] == "3.2"
    assert v2_catalog["ontology_version"] == ONTOLOGY_VERSION
    assert v2_catalog["publication_status"] == "publishable"


def test_shipped_v2_kinds_are_exactly_the_ontology(v2_catalog):
    kinds = {it["kind"] for it in v2_catalog["items"]}
    assert kinds == OBSERVATION_KINDS


def test_shipped_v2_every_item_has_an_atomic_claim(v2_catalog):
    for item in v2_catalog["items"]:
        claim = item["atomic_claim"]
        assert claim["subject"].strip()
        assert claim["state"].strip()
        assert claim["ontology_basis"].strip()


def _without_repair_support_marker(value):
    """Strip the Catalog 3.2 affinity marker so parity compares the inherited
    economics only. The marker is added by a post-inheritance generator stage
    and is additive metadata — it never rewrites package_type, package_role or
    any cost field."""
    if not isinstance(value, dict):
        return value
    return {
        room: {k: v for k, v in entry.items() if k != REPAIR_SUPPORT_MARKER}
        if isinstance(entry, dict) else entry
        for room, entry in value.items()
    }


def test_shipped_v2_split_successors_inherit_parent_economics(
    v1_catalog, v2_catalog, manifest
):
    """Task 4A bridge: every split successor carries exactly the economic
    fields its v1 parent carried, byte-identically; absent stays absent.

    The one recorded exception is the Catalog 3.2 `repair_support_when_driven`
    marker inside package_affinity — see
    test_repair_support_marker_parity_exceptions_are_recorded for the pin.
    """
    inherited = [it for it in v2_catalog["items"] if it.get("pricing_status")]
    assert len(inherited) == 42
    v1_by_id = {it["id"]: it for it in v1_catalog["items"]}
    parent_of = {
        s["id"]: e["legacy_id"]
        for e in manifest["entries"] if e["change_type"] == "split"
        for s in e["successors"]
    }
    for item in inherited:
        assert item["pricing_status"] == "inherited_from_split_parent"
        parent = v1_by_id[parent_of[item["id"]]]
        for field in ECONOMIC_FIELDS:
            assert (field in parent) == (field in item), (item["id"], field)
            if field not in parent:
                continue
            expected, actual = parent[field], item[field]
            if field == "package_affinity":
                actual = _without_repair_support_marker(actual)
            assert expected == actual, (item["id"], field)


def test_shipped_v2_has_no_deferred_pricing_status(v2_catalog):
    assert not any(
        it.get("pricing_status") == "deferred_post_task3"
        for it in v2_catalog["items"]
    )


def test_shipped_v2_trade_buckets_preserved_verbatim(v1_catalog, v2_catalog):
    assert v2_catalog["trade_buckets"] == v1_catalog["trade_buckets"]


def test_shipped_manifest_validates_clean(manifest, v1_catalog, v2_catalog):
    result = validate_migration_manifest(manifest, v1_catalog, v2_catalog)
    assert result.errors == []


def test_shipped_manifest_covers_all_107_legacy_ids(manifest, v1_catalog):
    legacy_ids = {it["id"] for it in v1_catalog["items"]}
    assert len(legacy_ids) == 107
    assert {e["legacy_id"] for e in manifest["entries"]} == legacy_ids
    assert len(manifest["entries"]) == 107


def test_shipped_manifest_is_audit_only(manifest):
    assert manifest["audit_only"] is True


def test_shipped_manifest_split_parents_are_deprecated_and_absent_from_v2(manifest, v2_catalog):
    v2_ids = {it["id"] for it in v2_catalog["items"]}
    splits = [e for e in manifest["entries"] if e["change_type"] == "split"]
    assert splits, "expected split entries"
    for entry in splits:
        assert entry["deprecated"] is True
        assert entry["requires_re_resolution"] is True
        assert entry["legacy_id"] not in v2_ids
        assert len(entry["successors"]) >= 2


# ── versioned kind vocabulary ────────────────────────────────────────────────

def test_v2_marked_catalog_rejects_legacy_upgrade_kind():
    catalog = _minimal_v2_catalog(item_overrides={"kind": "upgrade"})
    result = validate_issue_catalog(catalog)
    assert any("kind 'upgrade'" in e for e in result.errors)


def test_legacy_catalog_still_validates_two_kinds(v1_catalog):
    result = validate_issue_catalog(v1_catalog)
    assert result.errors == []


def test_legacy_catalog_rejects_v2_kinds():
    """Without the v2 stamp the legacy vocabulary applies — degradation is not
    silently accepted into a v1 catalog."""
    catalog = {
        "trade_buckets": [{"id": "flooring", "name": "Flooring"}],
        "items": [_minimal_v2_catalog()["items"][0]],
    }
    result = validate_issue_catalog(catalog)
    assert any("kind 'degradation'" in e for e in result.errors)


# ── v2 item rules ────────────────────────────────────────────────────────────

def test_v2_item_missing_atomic_claim_errors():
    catalog = _minimal_v2_catalog()
    del catalog["items"][0]["atomic_claim"]
    result = validate_issue_catalog(catalog)
    assert any("atomic_claim missing" in e for e in result.errors)


def test_v2_item_blank_atomic_claim_field_errors():
    catalog = _minimal_v2_catalog(
        item_overrides={"atomic_claim": {"subject": "carpet", "state": " ", "ontology_basis": "x"}}
    )
    result = validate_issue_catalog(catalog)
    assert any("atomic_claim.state" in e for e in result.errors)


def test_v2_retired_deferred_pricing_status_errors():
    """deferred_post_task3 was retired in Task 4A; it must not resurface."""
    catalog = _minimal_v2_catalog(item_overrides={"pricing_status": "deferred_post_task3"})
    result = validate_issue_catalog(catalog)
    assert any("pricing_status 'deferred_post_task3'" in e for e in result.errors)


def test_v2_inherited_item_with_economics_is_clean():
    catalog = _minimal_v2_catalog(item_overrides={
        "pricing_status": "inherited_from_split_parent",
        "cost": {"mode": "allowance", "cost_source": "catalog", "base_low": 100, "base_high": 400},
    })
    result = validate_issue_catalog(catalog)
    assert result.errors == []


def test_v2_inherited_item_without_economics_warns_but_validates():
    catalog = _minimal_v2_catalog(item_overrides={
        "pricing_status": "inherited_from_split_parent",
    })
    result = validate_issue_catalog(catalog)
    assert result.errors == []
    assert any("no inherited economic fields" in w for w in result.warnings)


def test_v2_unknown_pricing_status_errors():
    catalog = _minimal_v2_catalog(item_overrides={"pricing_status": "priced_later"})
    result = validate_issue_catalog(catalog)
    assert any("pricing_status 'priced_later'" in e for e in result.errors)


def test_v2_bad_root_metadata_errors():
    result = validate_issue_catalog(_minimal_v2_catalog(root_overrides={"version": "2.9"}))
    assert any("must be '3.2'" in e for e in result.errors)
    result = validate_issue_catalog(
        _minimal_v2_catalog(root_overrides={"publication_status": "shippable"})
    )
    assert any("publication_status 'shippable'" in e for e in result.errors)


# ── manifest rules on synthetic mutations ────────────────────────────────────

def test_manifest_detects_missing_entry(manifest, v1_catalog, v2_catalog):
    broken = copy.deepcopy(manifest)
    broken["entries"] = broken["entries"][1:]
    result = validate_migration_manifest(broken, v1_catalog, v2_catalog)
    assert any("legacy ids without an entry" in e for e in result.errors)


def test_manifest_detects_kind_disagreement(manifest, v1_catalog, v2_catalog):
    broken = copy.deepcopy(manifest)
    broken["entries"][0]["successors"][0]["kind"] = "modernization"
    result = validate_migration_manifest(broken, v1_catalog, v2_catalog)
    assert any("disagrees" in e for e in result.errors)


def test_manifest_detects_unknown_successor(manifest, v1_catalog, v2_catalog):
    broken = copy.deepcopy(manifest)
    broken["entries"][0]["successors"][0]["id"] = "no_such_item"
    result = validate_migration_manifest(broken, v1_catalog, v2_catalog)
    assert any("not in the v2 catalog" in e for e in result.errors)


def test_manifest_detects_orphan_v2_item(manifest, v1_catalog, v2_catalog):
    grown = copy.deepcopy(v2_catalog)
    grown["items"].append(dict(grown["items"][0], id="orphan_item"))
    result = validate_migration_manifest(manifest, v1_catalog, grown)
    assert any("no manifest parent" in e for e in result.errors)


def test_manifest_detects_re_resolution_inconsistency(manifest, v1_catalog, v2_catalog):
    broken = copy.deepcopy(manifest)
    split = next(e for e in broken["entries"] if e["change_type"] == "split")
    split["requires_re_resolution"] = False
    result = validate_migration_manifest(broken, v1_catalog, v2_catalog)
    assert any("requires_re_resolution must be True" in e for e in result.errors)


def test_manifest_rejects_runtime_alias_framing(manifest, v1_catalog, v2_catalog):
    broken = copy.deepcopy(manifest)
    broken["audit_only"] = False
    result = validate_migration_manifest(broken, v1_catalog, v2_catalog)
    assert any("not a runtime alias table" in e for e in result.errors)


# ── generator parity ─────────────────────────────────────────────────────────

def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "migrate_catalog_kind_v2", ROOT / "scripts" / "migrate_catalog_kind_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_reproduces_committed_artifacts_byte_for_byte(v1_catalog):
    gen = _load_generator()
    decisions = json.loads(
        (ROOT / "tools" / "catalog_migrations" / "kind_v2_decisions.json").read_text(encoding="utf-8")
    )
    catalog, manifest_out = gen.generate(v1_catalog, decisions)
    audit = gen.render_audit(catalog, manifest_out, decisions)

    expected_catalog = (ROOT / "tools" / "issue_catalog_kind_v2.json").read_text(encoding="utf-8")
    expected_manifest = (ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0.json").read_text(encoding="utf-8")
    expected_audit = (ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0_audit.md").read_text(encoding="utf-8")

    assert json.dumps(catalog, indent=2, ensure_ascii=False) + "\n" == expected_catalog
    assert json.dumps(manifest_out, indent=2, ensure_ascii=False) + "\n" == expected_manifest
    assert audit + "\n" == expected_audit


def test_audit_report_kind_counts_match_the_catalog(v2_catalog):
    audit = (ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0_audit.md").read_text(encoding="utf-8")
    from collections import Counter

    kinds = Counter(it["kind"] for it in v2_catalog["items"])
    expected = (
        f"- v2 items: {len(v2_catalog['items'])} "
        f"({kinds['defect']} defect / {kinds['degradation']} degradation / "
        f"{kinds['modernization']} modernization)"
    )
    assert expected in audit


# ── Catalog 3.2: contextual repair support ───────────────────────────────────

# The approved matrix: 10 items, 24 {room}_modernization routes. Pinned here so
# widening or narrowing it is a deliberate edit with a visible diff, not a
# side effect of regenerating the catalog.
EXPECTED_REPAIR_SUPPORT_MATRIX = {
    "peeling_or_discolored_paint": {"bedroom", "kitchen", "living"},
    "worn_or_stained_carpet": {"bedroom", "kitchen", "living"},
    "vinyl_linoleum_worn_or_stained": {"bedroom", "kitchen", "living"},
    "worn_or_stained_flooring": {"bedroom", "kitchen", "living"},
    "baseboard_wear_scuffs": {"bathroom", "bedroom", "kitchen", "living"},
    "wall_scuffs_marks_or_dents": {"bathroom", "bedroom", "kitchen", "living"},
    "cabinets_worn_finish": {"kitchen"},
    "appliances_worn_or_neglected": {"kitchen"},
    "vanity_countertop_worn": {"bathroom"},
    "vanity_countertop_dated": {"bathroom"},
}

# Split successors inherit package_affinity verbatim; these routes carry the
# marker anyway, which is the recorded departure from that parity.
EXPECTED_PARITY_EXCEPTION_IDS = frozenset({
    "vinyl_linoleum_worn_or_stained",
    "cabinets_worn_finish",
    "appliances_worn_or_neglected",
    "vanity_countertop_worn",
    "vanity_countertop_dated",
})


def _marked_routes(catalog):
    return {
        (item["id"], room)
        for item in catalog["items"]
        for room, entry in (item.get("package_affinity") or {}).items()
        if entry.get(REPAIR_SUPPORT_MARKER)
    }


def test_shipped_v2_repair_support_matrix(v2_catalog):
    expected = {
        (item_id, room)
        for item_id, rooms in EXPECTED_REPAIR_SUPPORT_MATRIX.items()
        for room in rooms
    }
    assert _marked_routes(v2_catalog) == expected
    assert len(expected) == 24
    assert len(EXPECTED_REPAIR_SUPPORT_MATRIX) == 10


def test_marked_routes_are_modernization_with_a_paired_repair_family(v2_catalog):
    """The marker's whole contract: a primary modernization route that has
    somewhere to move to."""
    for item in v2_catalog["items"]:
        for room, entry in (item.get("package_affinity") or {}).items():
            if not entry.get(REPAIR_SUPPORT_MARKER):
                continue
            assert entry["package_type"] == f"{room}_modernization", (item["id"], room)
            assert paired_repair_package_type(entry["package_type"]) == f"{room}_repair"


def test_marker_does_not_disturb_base_routing(v1_catalog, v2_catalog):
    """Additive metadata only: every marked route keeps the package_type and
    package_role it had before Catalog 3.2, and gains no other key."""
    v1_by_id = {it["id"]: it for it in v1_catalog["items"]}
    for item_id, rooms in EXPECTED_REPAIR_SUPPORT_MATRIX.items():
        item = next(it for it in v2_catalog["items"] if it["id"] == item_id)
        for room in rooms:
            entry = item["package_affinity"][room]
            assert set(entry) == {"package_type", "package_role", REPAIR_SUPPORT_MARKER}
            parent = v1_by_id.get(item_id)
            if parent is None:  # split successor — parity is covered elsewhere
                continue
            inherited = parent["package_affinity"][room]
            assert entry["package_type"] == inherited["package_type"]
            assert entry["package_role"] == inherited["package_role"]


def test_repair_support_marker_parity_exceptions_are_recorded(manifest):
    """Every split-successor marker is listed in the generated manifest, so the
    departure from inherited-affinity parity is reviewable, not implicit."""
    routing = manifest["package_routing"]
    assert routing["policy"] == "contextual_repair_support_v1"
    assert routing["marker"] == REPAIR_SUPPORT_MARKER
    assert len(routing["marked_routes"]) == 24
    exceptions = routing["inherited_affinity_parity_exceptions"]
    assert {e["id"] for e in exceptions} == EXPECTED_PARITY_EXCEPTION_IDS
    for entry in exceptions:
        assert entry["inherited_from"]
        assert entry["package_type"].endswith("_modernization")


def test_audit_reports_every_marked_route():
    audit = (ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0_audit.md").read_text(encoding="utf-8")
    assert "## Contextual repair support (Catalog 3.2)" in audit
    assert "### Inherited-affinity parity exceptions" in audit
    for item_id, rooms in EXPECTED_REPAIR_SUPPORT_MATRIX.items():
        for room in rooms:
            assert f"| `{item_id}` | {room} |" in audit


def _decisions():
    return json.loads(
        (ROOT / "tools" / "catalog_migrations" / "kind_v2_decisions.json").read_text(encoding="utf-8")
    )


def test_generator_refuses_marker_on_unknown_item(v1_catalog):
    gen = _load_generator()
    broken = copy.deepcopy(_decisions())
    broken["package_routing_decisions"][REPAIR_SUPPORT_MARKER].append(
        {"id": "no_such_item", "rooms": ["bedroom"]}
    )
    with pytest.raises(SystemExit, match="unknown item"):
        gen.generate(v1_catalog, broken)


def test_generator_refuses_marker_on_a_room_without_a_route(v1_catalog):
    gen = _load_generator()
    broken = copy.deepcopy(_decisions())
    rule = next(r for r in broken["package_routing_decisions"][REPAIR_SUPPORT_MARKER]
                if r["id"] == "cabinets_worn_finish")
    rule["rooms"] = ["exterior"]
    with pytest.raises(SystemExit, match="no package_affinity route"):
        gen.generate(v1_catalog, broken)


def test_generator_refuses_marker_on_a_repair_route(v1_catalog):
    """A repair route has no paired repair family — marking one is an authoring
    error, not a silent no-op."""
    gen = _load_generator()
    broken = copy.deepcopy(_decisions())
    broken["package_routing_decisions"][REPAIR_SUPPORT_MARKER].append(
        {"id": "damaged_drywall_or_cracks", "rooms": ["bedroom"]}
    )
    with pytest.raises(SystemExit, match="no paired"):
        gen.generate(v1_catalog, broken)


def test_generator_refuses_duplicate_marked_route(v1_catalog):
    gen = _load_generator()
    broken = copy.deepcopy(_decisions())
    broken["package_routing_decisions"][REPAIR_SUPPORT_MARKER].append(
        {"id": "cabinets_worn_finish", "rooms": ["kitchen"]}
    )
    with pytest.raises(SystemExit, match="twice"):
        gen.generate(v1_catalog, broken)


def test_generator_refuses_an_unknown_routing_policy(v1_catalog):
    gen = _load_generator()
    broken = copy.deepcopy(_decisions())
    broken["package_routing_decisions"]["policy"] = "something_else_v9"
    with pytest.raises(SystemExit, match="package_routing_decisions.policy"):
        gen.generate(v1_catalog, broken)


def test_generator_refuses_an_empty_room_list(v1_catalog):
    gen = _load_generator()
    broken = copy.deepcopy(_decisions())
    broken["package_routing_decisions"][REPAIR_SUPPORT_MARKER].append(
        {"id": "cabinets_worn_finish", "rooms": []}
    )
    with pytest.raises(SystemExit, match="lists no rooms"):
        gen.generate(v1_catalog, broken)


def test_generator_refuses_economic_override_on_split_successor(v1_catalog):
    gen = _load_generator()
    decisions = json.loads(
        (ROOT / "tools" / "catalog_migrations" / "kind_v2_decisions.json").read_text(encoding="utf-8")
    )
    broken = copy.deepcopy(decisions)
    split = next(e for e in broken["entries"] if e["change_type"] == "split")
    split["successors"][0].setdefault("overrides", {})["cost"] = {"mode": "heuristic"}
    with pytest.raises(SystemExit, match="economic fields"):
        gen.generate(v1_catalog, broken)


def test_generator_refuses_unknown_override_key_on_split_successor(v1_catalog):
    """An override outside INHERITED_FIELDS used to be silently ignored by the
    inheritance loop; a typo'd key must fail, not vanish (Session 8)."""
    gen = _load_generator()
    decisions = json.loads(
        (ROOT / "tools" / "catalog_migrations" / "kind_v2_decisions.json").read_text(encoding="utf-8")
    )
    broken = copy.deepcopy(decisions)
    split = next(e for e in broken["entries"] if e["change_type"] == "split")
    split["successors"][0].setdefault("overrides", {})["route_overide"] = "no_action"
    with pytest.raises(SystemExit, match="non-inherited fields"):
        gen.generate(v1_catalog, broken)


def test_shipped_v2_route_override_pins(v1_catalog, v2_catalog):
    """The Session 8 triage: exactly these five opportunity/presence items
    carry route_override — four carried over from v1 and one authored on the
    landscaping split successor. The degradation sibling must never inherit
    it, which requires the v1 split parent to stay override-free."""
    carrying = {
        it["id"]: it["route_override"]
        for it in v2_catalog["items"] if "route_override" in it
    }
    assert carrying == {
        "unfinished_basement_present": "no_action",
        "staging_or_decluttering_opportunity": "no_action",
        "mismatched_or_inconsistent_furniture_staging": "no_action",
        "curb_appeal_upgrade": "no_action",
        "landscaping_enhancement_opportunity": "no_action",
    }
    parent = next(
        it for it in v1_catalog["items"]
        if it["id"] == "landscape_improvement_needed"
    )
    assert "route_override" not in parent


def test_generator_refuses_missing_pricing_policy(v1_catalog):
    gen = _load_generator()
    decisions = json.loads(
        (ROOT / "tools" / "catalog_migrations" / "kind_v2_decisions.json").read_text(encoding="utf-8")
    )
    broken = copy.deepcopy(decisions)
    del broken["split_successor_pricing"]
    with pytest.raises(SystemExit, match="split_successor_pricing"):
        gen.generate(v1_catalog, broken)


def test_generator_refuses_incomplete_coverage(v1_catalog):
    gen = _load_generator()
    decisions = json.loads(
        (ROOT / "tools" / "catalog_migrations" / "kind_v2_decisions.json").read_text(encoding="utf-8")
    )
    broken = copy.deepcopy(decisions)
    broken["entries"] = broken["entries"][:-1]
    with pytest.raises(SystemExit, match="cover exactly"):
        gen.generate(v1_catalog, broken)
