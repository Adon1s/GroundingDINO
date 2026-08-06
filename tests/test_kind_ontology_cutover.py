"""Task 4A cutover contract: the KIND_ONTOLOGY_VERSION selector, the canary
comparator gates, and the fail-loud behavior of current engines on stale ids.

Historical-artifact backfill is Task 4B; its prototype (tools/backfill_kind_v2.py)
is deliberately untracked and untested here — see the 4B handoff.
"""
import json
from pathlib import Path

import pytest

from tools import pipeline_config as cfg
from tools.artifact_writers import load_issue_catalog
from tools.compare_kind_cutover import compare_property, evaluate_gates
from tools.costing import CatalogDataError, compute_scoring
from tools.pass_config import (
    ALLOWED_PIPELINE_MODES,
    PIPELINE_MODE_CLASSIFICATION_ONLY,
    PIPELINE_MODE_PUBLISH,
    SceneClassifierRunOptions,
)
from tools.renovation_estimate import extract_estimate_candidates

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v2_catalog():
    return load_issue_catalog(ROOT / "tools" / "issue_catalog_kind_v2.json")


@pytest.fixture(scope="module")
def cutover_config():
    return json.loads(
        (ROOT / "configs" / "kind_ontology_cutover.json").read_text(encoding="utf-8")
    )


# ── selector ─────────────────────────────────────────────────────────────────

def test_legacy_v1_is_the_runtime_default():
    """Repo/local default stays legacy_v1; production opts into v2 explicitly."""
    assert cfg.KIND_ONTOLOGY_VERSION == cfg.KIND_ONTOLOGY_LEGACY_V1
    assert Path(cfg.ISSUE_CATALOG_PATH).name == "issue_catalog.json"
    assert cfg.PIPELINE_MODE == PIPELINE_MODE_CLASSIFICATION_ONLY


def test_selector_modes_are_valid_pipeline_modes():
    for raw in (cfg.KIND_ONTOLOGY_LEGACY_V1, cfg.KIND_ONTOLOGY_V2):
        assert cfg.resolve_kind_ontology(raw).pipeline_mode in ALLOWED_PIPELINE_MODES


def test_v2_selector_derives_catalog_and_publish_mode_atomically():
    sel = cfg.resolve_kind_ontology(cfg.KIND_ONTOLOGY_V2)
    assert sel.version == "observation_kind_v2"
    assert sel.catalog_path.name == "issue_catalog_kind_v2.json"
    assert sel.pipeline_mode == PIPELINE_MODE_PUBLISH


def test_invalid_selector_value_fails_startup():
    for bad in ("v2", "legacy", "observation-kind-v2", ""):
        with pytest.raises(ValueError, match="KIND_ONTOLOGY_VERSION"):
            cfg.resolve_kind_ontology(bad)


def test_v2_selector_rejects_catalog_path_override():
    with pytest.raises(ValueError, match="ISSUE_CATALOG_PATH"):
        cfg.resolve_kind_ontology(
            cfg.KIND_ONTOLOGY_V2, catalog_path_override="C:/somewhere/else.json"
        )


def test_publish_mode_flows_through_profile_factory():
    options = SceneClassifierRunOptions.from_analysis_profile(
        "standard", pipeline_mode=PIPELINE_MODE_PUBLISH
    )
    assert options.pipeline_mode == PIPELINE_MODE_PUBLISH
    with pytest.raises(ValueError, match="pipeline_mode"):
        SceneClassifierRunOptions.from_analysis_profile(
            "standard", pipeline_mode="publish_without_resolution"
        )


# ── cutover configuration ────────────────────────────────────────────────────

def test_cutover_config_matches_the_4a_rollout_contract(cutover_config):
    rollout = cutover_config["rollout"]
    assert rollout["default"] == "legacy_v1"
    assert rollout["rollback_switch"] == "KIND_ONTOLOGY_VERSION=legacy_v1"
    assert rollout["observation_window_days"] == 7
    assert cutover_config["minimum_canary_properties"] == 18
    assert cutover_config["minimum_per_stratum"] == 3
    assert len(cutover_config["representative_strata"]) == 6


# ── current engines fail loud on stale catalog ids ───────────────────────────

def test_current_engines_fail_loud_on_stale_catalog_ids(v2_catalog):
    issue = {"catalog_item_id": "damaged_soffit_or_porch_ceiling", "status": "confirmed"}
    with pytest.raises(CatalogDataError, match="unknown/stale"):
        compute_scoring([issue], v2_catalog)
    with pytest.raises(ValueError, match="unknown/stale"):
        extract_estimate_candidates([issue], v2_catalog)


# ── canary comparator gates ──────────────────────────────────────────────────

def test_canary_gate_reports_stale_kind_and_headline_delta(v2_catalog):
    baseline = {
        "issues_flat": [{"description": "Worn roof", "catalog_item_id": "old", "kind": "defect"}],
        "renovation_estimate_v4": {"final_rehab": {"low": 100, "high": 200}},
    }
    valid_id = v2_catalog["items"][0]["id"]
    candidate = {
        "issues_flat": [{"description": "Worn roof", "catalog_item_id": valid_id, "kind": "upgrade"}],
        "renovation_estimate_v4": {"final_rehab": {"low": 200, "high": 400}},
    }
    row = compare_property(
        "p1",
        baseline,
        candidate,
        valid_catalog_ids=frozenset(item["id"] for item in v2_catalog["items"]),
    )
    failures = evaluate_gates(
        [row],
        {
            "minimum_canary_properties": 1,
            "thresholds": {"max_unapproved_headline_delta": 0.15},
            "approved_headline_deltas": {},
        },
    )
    assert {failure["gate"] for failure in failures} >= {
        "no_stale_kinds",
        "unapproved_headline_delta",
    }


def test_canary_gate_enforces_minimum_property_count():
    failures = evaluate_gates([], {"minimum_canary_properties": 18})
    assert failures == [
        {"gate": "representative_property_count", "actual": 0, "minimum": 18}
    ]
