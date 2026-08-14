"""Condition-aggregation and evidence-identity tests (pure stages: no
provider, no runtime, no envelope).

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_conditions.py -q
"""
import pytest

from tests.conftest import tiny_png
from tools.failure_taxonomy import classify_failure
from tools.renovation_architecture.conditions import (
    FALLBACK_RESOLUTION_REASON,
    build_observed_conditions,
)
from tools.renovation_architecture.contracts import (
    EVIDENCE_DEDUP_POLICY_VERSION,
)
from tools.renovation_architecture.disposition import decide_disposition
from tools.renovation_architecture.evidence import (
    build_evidence_facts,
    build_photo_identity_index,
    dedup_photos,
)
from tools.renovation_architecture.ids import make_estimate_id
from tools.scene_classifier_passes import PassExecutionError

EST_ID = make_estimate_id(
    property_key="prop",
    source_run_id="run_1",
    catalog_sha256="a" * 64,
    projection_fingerprint="b" * 64,
)


# ── builders ─────────────────────────────────────────────────────────────────

def _photos(*specs):
    """specs: (photo_key, scene) pairs, indexed in listing order."""
    return {
        key: {"photo": {"index": index}, "scene": {"id": scene}}
        for index, (key, scene) in enumerate(specs, start=1)
    }


def _issue(catalog_item_id, photo_key, **over):
    base = {
        "issue_id": f"iss_{catalog_item_id}_{photo_key}",
        "photo_key": photo_key,
        "scene_group": "kitchen",
        "catalog_item_id": catalog_item_id,
        "description": f"{catalog_item_id} observed",
    }
    base.update(over)
    return base


def _projection(**items):
    """items: item_id -> min_photo_evidence (None for no threshold)."""
    return {
        "observables": {
            item_id: {
                "kind": "degradation",
                "scope": "repair",
                "atomic_claim": {"subject": "surface", "state": "worn"},
                "min_photo_evidence": threshold,
            }
            for item_id, threshold in items.items()
        }
    }


def _build(issues, photos, projection=None):
    return build_observed_conditions(
        issues_flat=issues,
        photos=photos,
        property_metadata=None,
        projection=projection or _projection(worn_counter=None),
        estimate_id=EST_ID,
    )


def _files(tmp_path, colors):
    """colors: photo_key -> (r, g, b); writes real 1x1 PNGs."""
    paths = {}
    for key, color in colors.items():
        path = tmp_path / key
        path.write_bytes(tiny_png(*color))
        paths[key] = path
    return paths


# ── condition aggregation ────────────────────────────────────────────────────

class TestBuildObservedConditions:
    def test_same_unit_repeats_collapse_into_one_condition(self):
        photos = _photos(("k1.png", "kitchen"), ("k2.png", "kitchen"))
        issues = [
            _issue("worn_counter", "k1.png"),
            _issue("worn_counter", "k2.png"),
        ]
        (draft,) = _build(issues, photos)
        condition = draft.condition
        assert condition.estimate_unit_id == "kitchen_primary"
        assert condition.issue_ids == (
            "iss_worn_counter_k1.png", "iss_worn_counter_k2.png"
        )
        assert condition.unit_resolution_source == "photo_estimate_unit"
        assert len(draft.evidence_refs) == 2

    def test_distinct_units_stay_distinct(self):
        photos = _photos(("k1.png", "kitchen"), ("e1.png", "exterior"))
        issues = [
            _issue("worn_counter", "k1.png"),
            _issue("worn_counter", "e1.png", scene_group="exterior"),
        ]
        drafts = _build(issues, photos)
        assert len(drafts) == 2
        units = {draft.condition.estimate_unit_id for draft in drafts}
        assert len(units) == 2

    def test_unmapped_photo_falls_back_to_the_scope_room(self):
        photos = _photos(("k1.png", "kitchen"))
        issues = [
            _issue("worn_counter", "elsewhere.png", location_hint="pantry"),
        ]
        (draft,) = _build(issues, photos)
        condition = draft.condition
        assert condition.estimate_unit_id == "pantry"
        assert condition.unit_resolution_source == "scope_room_fallback"
        assert condition.unit_resolution_reason == FALLBACK_RESOLUTION_REASON
        assert condition.identity_ambiguous is True
        assert condition.source_room_surrogate_ids == ()

    def test_mixed_contributors_keep_the_weaker_source(self):
        """A photo-mapped issue and a fallback issue landing on the same unit
        leave the honest weaker lineage on the merged condition."""
        photos = _photos(("k1.png", "kitchen"))
        issues = [
            _issue("worn_counter", "k1.png"),
            _issue("worn_counter", "lost.png", location_hint="kitchen_primary"),
        ]
        (draft,) = _build(issues, photos)
        condition = draft.condition
        assert condition.estimate_unit_id == "kitchen_primary"
        assert condition.unit_resolution_source == "scope_room_fallback"
        assert condition.unit_resolution_reason == FALLBACK_RESOLUTION_REASON
        assert condition.identity_ambiguous is True

    def test_output_order_is_stable_under_input_shuffle(self):
        photos = _photos(("k1.png", "kitchen"), ("e1.png", "exterior"))
        issues = [
            _issue("worn_counter", "k1.png"),
            _issue("worn_siding", "e1.png", scene_group="exterior"),
            _issue("worn_counter", "e1.png", scene_group="exterior"),
        ]
        projection = _projection(worn_counter=None, worn_siding=None)
        forward = _build(issues, photos, projection)
        backward = _build(list(reversed(issues)), photos, projection)
        assert [d.condition.condition_id for d in forward] == [
            d.condition.condition_id for d in backward
        ]
        assert [d.condition.to_dict() for d in forward] == [
            d.condition.to_dict() for d in backward
        ]

    def test_stale_catalog_id_fails_closed(self):
        photos = _photos(("k1.png", "kitchen"))
        issues = [_issue("retired_item", "k1.png")]
        with pytest.raises(PassExecutionError) as excinfo:
            _build(issues, photos)
        assert excinfo.value.code == "StaleCatalogItemId"
        assert "retired_item" in excinfo.value.message
        # An operational dependency failure — never a cannot_assess verdict.
        assert classify_failure(excinfo.value).category == "dependency"

    def test_uncatalogued_issue_is_skipped(self):
        photos = _photos(("k1.png", "kitchen"))
        unmatched = _issue("worn_counter", "k1.png", issue_id="iss_unmatched")
        unmatched["catalog_item_id"] = None
        issues = [_issue("worn_counter", "k1.png"), unmatched]
        (draft,) = _build(issues, photos)
        assert "iss_unmatched" not in draft.condition.issue_ids

    def test_issue_without_a_photo_fails_closed(self):
        photos = _photos(("k1.png", "kitchen"))
        issues = [_issue("worn_counter", "")]
        with pytest.raises(PassExecutionError) as excinfo:
            _build(issues, photos)
        assert excinfo.value.code == "MissingPhotoReference"


# ── evidence identity ────────────────────────────────────────────────────────

class TestEvidenceIdentity:
    def test_exact_duplicates_group(self, tmp_path):
        paths = _files(tmp_path, {
            "a.png": (200, 30, 30),
            "b.png": (200, 30, 30),
            "c.png": (10, 200, 240),
        })
        index = build_photo_identity_index(list(paths), paths)
        dedup = dedup_photos(("a.png", "b.png", "c.png"), index)
        assert dedup.exact_groups == (("a.png", "b.png"),)
        assert dedup.representatives == ("a.png", "c.png")
        assert dedup.distinct_view_count == 2

    def test_near_duplicates_group_within_the_rgb_gate(self, tmp_path):
        paths = _files(tmp_path, {
            "a.png": (100, 100, 100),
            "b.png": (100, 100, 116),  # max-channel delta 16: near
        })
        index = build_photo_identity_index(list(paths), paths)
        dedup = dedup_photos(("a.png", "b.png"), index)
        assert dedup.near_groups == (("a.png", "b.png"),)
        assert dedup.distinct_view_count == 1

    def test_the_rgb_gate_rescues_degenerate_hashes(self, tmp_path):
        """Uniform images share the same trivial average-hash; only the
        mean-RGB delta separates them. Delta 17 must stay distinct."""
        paths = _files(tmp_path, {
            "a.png": (100, 100, 100),
            "b.png": (100, 100, 117),  # max-channel delta 17: distinct
        })
        index = build_photo_identity_index(list(paths), paths)
        assert (
            index["a.png"].average_hash == index["b.png"].average_hash
        )  # the degenerate case is real
        dedup = dedup_photos(("a.png", "b.png"), index)
        assert dedup.near_groups == ()
        assert dedup.distinct_view_count == 2

    def test_exact_and_near_chains_merge_disjointly(self, tmp_path):
        paths = _files(tmp_path, {
            "a.png": (100, 100, 100),
            "b.png": (100, 100, 100),  # exact duplicate of a
            "c.png": (100, 100, 112),  # near b (and a)
        })
        index = build_photo_identity_index(list(paths), paths)
        dedup = dedup_photos(("a.png", "b.png", "c.png"), index)
        assert dedup.exact_groups == (("a.png", "b.png"),)
        assert dedup.merged_groups == (("a.png", "b.png", "c.png"),)
        assert dedup.representatives == ("a.png",)
        assert dedup.distinct_view_count == 1

    def test_duplicate_photos_do_not_satisfy_multi_view_thresholds(self, tmp_path):
        """Three filenames, one view: distinct_view_count — not the filename
        count — is what min_photo_evidence gates, so the supported verdict
        is withheld."""
        photos = _photos(
            ("k1.png", "kitchen"), ("k2.png", "kitchen"), ("k3.png", "kitchen")
        )
        issues = [
            _issue("roof_worn", key) for key in ("k1.png", "k2.png", "k3.png")
        ]
        projection = _projection(roof_worn=2)
        (draft,) = _build(issues, photos, projection)
        paths = _files(tmp_path, {
            key: (200, 30, 30) for key in ("k1.png", "k2.png", "k3.png")
        })
        index = build_photo_identity_index(list(paths), paths)
        evidence = build_evidence_facts(
            draft,
            identity_index=index,
            observables=projection["observables"],
            estimate_id=EST_ID,
        )
        assert evidence.distinct_photo_count == 3
        assert evidence.distinct_view_count == 1
        assert evidence.min_photo_evidence_required == 2
        assert evidence.dedup_policy_version == EVIDENCE_DEDUP_POLICY_VERSION
        assert decide_disposition(
            "supported", "work",
            evidence.distinct_view_count, evidence.min_photo_evidence_required,
        ) == ("withheld", "insufficient_distinct_views")

    def test_missing_photo_file_fails_closed(self, tmp_path):
        with pytest.raises(PassExecutionError) as excinfo:
            build_photo_identity_index(["ghost.png"], {})
        assert excinfo.value.code == "MissingEvidencePhoto"
        assert excinfo.value.stage == "dependency"

    def test_undecodable_photo_fails_closed(self, tmp_path):
        junk = tmp_path / "junk.png"
        junk.write_bytes(b"not an image at all")
        with pytest.raises(PassExecutionError) as excinfo:
            build_photo_identity_index(["junk.png"], {"junk.png": junk})
        assert excinfo.value.stage == "dependency"
        assert classify_failure(excinfo.value).category == "input"

    def test_evidence_refs_and_threshold_flow_through(self, tmp_path):
        photos = _photos(("k1.png", "kitchen"))
        issues = [_issue("worn_counter", "k1.png")]
        projection = _projection(worn_counter=None)
        (draft,) = _build(issues, photos, projection)
        paths = _files(tmp_path, {"k1.png": (200, 30, 30)})
        index = build_photo_identity_index(list(paths), paths)
        evidence = build_evidence_facts(
            draft,
            identity_index=index,
            observables=projection["observables"],
            estimate_id=EST_ID,
        )
        assert evidence.min_photo_evidence_required is None
        assert evidence.photo_keys == ("k1.png",)
        assert evidence.representative_photo_keys == ("k1.png",)
        (ref,) = evidence.evidence_refs
        assert ref["issue_id"] == "iss_worn_counter_k1.png"
        assert ref["photo_key"] == "k1.png"
        assert ref["room_surrogate_id"]  # the kitchen surrogate contributed
