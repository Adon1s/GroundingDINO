"""
observation-kind-v2 freeze guards.

Task 1 of the kind-ontology rework ends the pipeline after Pass 2c
(classification only). These tests pin the two invariants that keep a v2
result from leaking into canonical outputs or being misread later:

1. write_photo_intel refuses to publish a classification_only payload —
   the enforcement point of the analysis freeze until Task 2/3 land.
2. Stored artifacts without an ontology_version read as legacy_v1 and are
   never reinterpreted against the three-kind ontology.

See docs/HANDOFF_kind_ontology_task1.md.
"""
from types import SimpleNamespace

import pytest

from tools.artifact_writers import write_photo_intel
from tools.pipeline_common import (
    LEGACY_ONTOLOGY_VERSION,
    artifact_ontology_version,
)
from tools.scene_classifier_passes import ONTOLOGY_VERSION


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
