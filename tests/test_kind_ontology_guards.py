"""
observation-kind-v2 publication guards.

These tests pin the permanent invariants that keep incomplete or mismatched
output from leaking into canonical artifacts:

1. write_photo_intel refuses to publish a classification_only payload —
   classification-only output is never publishable.
2. write_photo_intel refuses to publish against a catalog whose
   publication_status is anything but publishable (or absent, for v1).
   As of Task 4A the shipped v2 catalog IS publishable; the gate is pinned
   with synthetic blocked catalogs.
3. Stored artifacts without an ontology_version read as legacy_v1 and are
   never reinterpreted against the three-kind ontology.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.artifact_writers import load_issue_catalog, write_photo_intel
from tools.pipeline_common import (
    LEGACY_ONTOLOGY_VERSION,
    artifact_ontology_version,
)
from tools.scene_classifier_passes import ONTOLOGY_VERSION

ROOT = Path(__file__).resolve().parents[1]


def _job_with_payload(payload):
    return SimpleNamespace(
        property_key="prop_1",
        job_id="job_1",
        results=[SimpleNamespace(
            image_path="photo_001.jpg",
            scene_classifier=payload,
            scene_data=None,
            scene="exterior_front",
        )],
    )


def test_write_photo_intel_rejects_classification_only_results():
    job = _job_with_payload({
        "scene": "exterior_front",
        "classification_only": True,
        "ontology_version": ONTOLOGY_VERSION,
        "observations": [{"description": "Siding is faded.", "kind": "degradation"}],
    })

    with pytest.raises(RuntimeError, match="classification_only"):
        write_photo_intel(
            cfg=None,
            job=job,
            detection_backend="none",
            analysis_profile="standard",
            use_pass_architecture=True,
            pass_toggles={},
            model_overrides={},
            gpt_config=None,
            issue_catalog={"items": []},
        )


def test_artifact_ontology_version_defaults_to_legacy():
    assert artifact_ontology_version({}) == LEGACY_ONTOLOGY_VERSION
    assert artifact_ontology_version(None) == LEGACY_ONTOLOGY_VERSION
    assert artifact_ontology_version({"ontology_version": ""}) == LEGACY_ONTOLOGY_VERSION


def test_artifact_ontology_version_reads_stamped_value():
    assert artifact_ontology_version({"ontology_version": ONTOLOGY_VERSION}) == ONTOLOGY_VERSION


def test_ontology_constants_are_the_shared_module_objects():
    """scene_classifier_passes re-exports tools.observation_kinds — identity,
    not equality, so the two names can never drift apart."""
    from tools import observation_kinds
    from tools import scene_classifier_passes as passes

    assert passes.ONTOLOGY_VERSION is observation_kinds.ONTOLOGY_VERSION
    assert passes.OBSERVATION_KINDS is observation_kinds.OBSERVATION_KINDS
    assert passes.EXCLUSION_REASONS is observation_kinds.EXCLUSION_REASONS


# ── catalog publication_status guard (Task 2) ───────────────────────────────

def _publishable_job():
    """A job whose payload is NOT classification_only, so only the catalog
    guard can reject it."""
    return _job_with_payload({"scene": "exterior_front", "verified_issues": []})


def _write(job, issue_catalog):
    return write_photo_intel(
        cfg=None,
        job=job,
        detection_backend="none",
        analysis_profile="standard",
        use_pass_architecture=True,
        pass_toggles={},
        model_overrides={},
        gpt_config=None,
        issue_catalog=issue_catalog,
    )


def test_write_photo_intel_rejects_a_non_publishable_catalog():
    with pytest.raises(RuntimeError, match="publication_status"):
        _write(_publishable_job(), {"items": [], "publication_status": "blocked_pending_pricing"})


def test_write_photo_intel_rejects_any_unrecognized_publication_status():
    """Fail closed: only the exact "publishable" value clears the gate."""
    with pytest.raises(RuntimeError, match="publication_status"):
        _write(_publishable_job(), {"items": [], "publication_status": "probably_fine"})


def test_absent_publication_status_is_treated_as_publishable():
    """The shipped v1 catalog carries no status; runtime behavior is unchanged."""
    catalog = load_issue_catalog(ROOT / "tools" / "issue_catalog.json")
    assert catalog["publication_status"] is None
    # Reaching the classification_only guard proves the catalog gate passed.
    with pytest.raises(RuntimeError, match="classification_only"):
        _write(
            _job_with_payload({"scene": "exterior_front", "classification_only": True}),
            catalog,
        )


def test_shipped_v2_catalog_clears_the_publication_status_gate():
    """End-to-end pin: the real v2 catalog, loaded the way production loads a
    catalog, is publishable as of Task 4A. Reaching the classification_only
    guard proves the catalog gate passed."""
    catalog = load_issue_catalog(ROOT / "tools" / "issue_catalog_kind_v2.json")
    assert catalog["publication_status"] == "publishable"
    assert catalog["ontology_version"] == ONTOLOGY_VERSION
    with pytest.raises(RuntimeError, match="classification_only"):
        _write(
            _job_with_payload({"scene": "exterior_front", "classification_only": True}),
            catalog,
        )


def test_load_issue_catalog_passes_root_metadata_through(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({
        "version": "3.0",
        "ontology_version": ONTOLOGY_VERSION,
        "publication_status": "blocked_pending_pricing",
        "trade_buckets": [],
        "items": [{"id": "x", "kind": "degradation"}],
    }), encoding="utf-8")

    loaded = load_issue_catalog(path)
    assert loaded["version"] == "3.0"
    assert loaded["ontology_version"] == ONTOLOGY_VERSION
    assert loaded["publication_status"] == "blocked_pending_pricing"
