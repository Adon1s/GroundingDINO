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
        "version": "3.0",
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
    assert v2_catalog["version"] == "3.0"
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


def test_shipped_v2_split_successors_inherit_parent_economics(
    v1_catalog, v2_catalog, manifest
):
    """Task 4A bridge: every split successor carries exactly the economic
    fields its v1 parent carried, byte-identically; absent stays absent."""
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
            if field in parent:
                assert parent[field] == item[field], (item["id"], field)


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
    assert any("must be '3.0'" in e for e in result.errors)
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
