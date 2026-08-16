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
from tools.publication_gate import (
    deprecated_legacy_ids,
    validate_publication_payload,
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


# ── publication payload gate (Task 4A) ───────────────────────────────────────

@pytest.fixture(scope="module")
def v2_catalog():
    return load_issue_catalog(ROOT / "tools" / "issue_catalog_kind_v2.json")


@pytest.fixture(scope="module")
def v1_catalog():
    return load_issue_catalog(ROOT / "tools" / "issue_catalog.json")


def _v2_payload(v2_catalog, issue_overrides=None, **root_overrides):
    item = next(it for it in v2_catalog["items"] if it["kind"] == "degradation")
    issue = {
        "issue_id": "i-1",
        "description": "Visible wear.",
        "catalog_item_id": item["id"],
        "catalog_item_kind": item["kind"],
    }
    issue.update(issue_overrides or {})
    payload = {
        "ontology_version": ONTOLOGY_VERSION,
        "catalog_version": str(v2_catalog["version"]),
        "issues_flat": [issue],
    }
    payload.update(root_overrides)
    return payload


def test_gate_accepts_consistent_v2_payload(v2_catalog):
    validate_publication_payload(_v2_payload(v2_catalog), v2_catalog)


def test_gate_rejects_stale_upgrade_kind(v2_catalog):
    payload = _v2_payload(v2_catalog, issue_overrides={"catalog_item_kind": "upgrade"})
    with pytest.raises(RuntimeError, match="stale kind or a mixed"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_deprecated_split_parent(v2_catalog):
    payload = _v2_payload(v2_catalog, issue_overrides={
        "catalog_item_id": "damaged_soffit_or_porch_ceiling",
        "catalog_item_kind": "defect",
    })
    with pytest.raises(RuntimeError, match="deprecated split parent") as exc:
        validate_publication_payload(payload, v2_catalog)
    assert "soffit_or_porch_ceiling_failed" in str(exc.value)


def test_gate_rejects_retired_item(v2_catalog):
    payload = _v2_payload(v2_catalog, issue_overrides={
        "catalog_item_id": "bathroom_layout_modernization_opportunity",
        "catalog_item_kind": "modernization",
    })
    with pytest.raises(RuntimeError, match="retired catalog id"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_unknown_catalog_id(v2_catalog):
    payload = _v2_payload(v2_catalog, issue_overrides={"catalog_item_id": "no_such_item"})
    with pytest.raises(RuntimeError, match="does not exist in"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_kind_mismatched_with_catalog_entry(v2_catalog):
    payload = _v2_payload(v2_catalog, issue_overrides={"catalog_item_kind": "defect"})
    with pytest.raises(RuntimeError, match="canonical kind"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_missing_ontology_stamp(v2_catalog):
    payload = _v2_payload(v2_catalog)
    del payload["ontology_version"]
    with pytest.raises(RuntimeError, match="no root ontology_version"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_catalog_version_mismatch(v2_catalog):
    payload = _v2_payload(v2_catalog, catalog_version="2.1")
    with pytest.raises(RuntimeError, match="catalog_version"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_mixed_ontology_payload(v2_catalog):
    """A payload stamped legacy_v1 can never publish against the v2 catalog."""
    payload = _v2_payload(v2_catalog, ontology_version=LEGACY_ONTOLOGY_VERSION)
    with pytest.raises(RuntimeError, match="ontology_version"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_accepts_v1_payload_against_v1_catalog(v1_catalog):
    item = next(it for it in v1_catalog["items"] if it.get("kind") == "upgrade")
    payload = {
        "ontology_version": LEGACY_ONTOLOGY_VERSION,
        "catalog_version": str(v1_catalog["version"]),
        "issues_flat": [{
            "issue_id": "i-1",
            "description": "Dated finishes.",
            "catalog_item_id": item["id"],
            "catalog_item_kind": item["kind"],
        }],
    }
    validate_publication_payload(payload, v1_catalog)


def test_gate_rejects_v2_kind_against_v1_catalog(v1_catalog):
    """Mixed payload in the other direction: three-kind rows under legacy_v1."""
    payload = {
        "ontology_version": LEGACY_ONTOLOGY_VERSION,
        "catalog_version": str(v1_catalog["version"]),
        "issues_flat": [{
            "issue_id": "i-1",
            "description": "Visible wear.",
            "catalog_item_kind": "degradation",
        }],
    }
    with pytest.raises(RuntimeError, match="stale kind or a mixed"):
        validate_publication_payload(payload, v1_catalog)


def test_deprecated_legacy_ids_are_the_split_parents_and_retired_items():
    ids = deprecated_legacy_ids()
    assert len(ids) == 21  # 19 split parents + 2 retired layout items (catalog 3.1)
    assert "damaged_soffit_or_porch_ceiling" in ids
    assert "bathroom_layout_modernization_opportunity" in ids
    assert "layout_modernization_opportunity" in ids


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


# ── renovation-architecture envelope gate (Session 5) ────────────────────────
#
# The private shadow key (analysis_debug.<SHADOW_DEBUG_KEY>) may hold only a
# valid finished envelope; the same key at the photo_intel root is reserved
# for the Session 6 cutover and must already be a valid complete envelope.

def _failed_envelope():
    from tests.test_renovation_architecture_contracts import _scaffold_envelope

    return _scaffold_envelope(
        state="failed", reason="provider", error_detail="synthetic",
        result=None,
    )


def _complete_envelope():
    from tests.test_renovation_architecture_contracts import (
        _complete_result,
        _scaffold_envelope,
    )

    return _scaffold_envelope(
        state="complete", reason=None, result=_complete_result()
    )


def _shadow_key():
    from tools.renovation_architecture.contracts import SHADOW_DEBUG_KEY

    return SHADOW_DEBUG_KEY


def test_gate_accepts_private_failed_envelope(v2_catalog):
    payload = _v2_payload(
        v2_catalog, analysis_debug={_shadow_key(): _failed_envelope()}
    )
    validate_publication_payload(payload, v2_catalog)


def test_gate_accepts_private_complete_envelope(v2_catalog):
    payload = _v2_payload(
        v2_catalog, analysis_debug={_shadow_key(): _complete_envelope()}
    )
    validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_malformed_private_envelope(v2_catalog):
    broken = _complete_envelope()
    broken["result"]["totals"]["headline"]["high"] += 1
    payload = _v2_payload(
        v2_catalog, analysis_debug={_shadow_key(): broken}
    )
    with pytest.raises(RuntimeError, match="is not a valid envelope"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_stale_schema_private_envelope(v2_catalog):
    stale = _failed_envelope()
    stale["schema_version"] -= 1
    payload = _v2_payload(
        v2_catalog, analysis_debug={_shadow_key(): stale}
    )
    with pytest.raises(RuntimeError, match="is not a valid envelope"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_partial_private_envelope(v2_catalog):
    """Intermediate session states are valid envelopes but not publishable —
    only finished complete/failed shadow output may be written."""
    from tests.test_renovation_architecture_contracts import (
        _package_review_envelope,
    )

    payload = _v2_payload(
        v2_catalog,
        analysis_debug={_shadow_key(): _package_review_envelope()},
    )
    with pytest.raises(RuntimeError, match="not publishable"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_accepts_valid_complete_root_envelope(v2_catalog):
    payload = _v2_payload(v2_catalog, **{_shadow_key(): _complete_envelope()})
    validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_non_complete_root_envelope(v2_catalog):
    """The root key is reserved for the cutover: even a valid failed envelope
    may not occupy it."""
    payload = _v2_payload(v2_catalog, **{_shadow_key(): _failed_envelope()})
    with pytest.raises(RuntimeError, match="reserved for the cutover"):
        validate_publication_payload(payload, v2_catalog)


def test_gate_rejects_malformed_root_envelope(v2_catalog):
    payload = _v2_payload(v2_catalog, **{_shadow_key(): {"state": "complete"}})
    with pytest.raises(RuntimeError, match="is not a valid envelope"):
        validate_publication_payload(payload, v2_catalog)
