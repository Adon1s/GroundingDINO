"""Dataset integrity, containment, fingerprints, and sealing.

A benchmark is only meaningful if its inputs cannot move. These tests are the
enforcement point for that claim.
"""
import json

import pytest

from tools.benchmarking import dataset as ds
from tools.comparison_common import ComparisonError, atomic_json

LFS_POINTER = (b"version https://git-lfs.github.com/spec/v1\n"
               b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
               b"size 69\n")


def reviewed_reference(vocabulary, listing_id="hsv-001", photo_keys=("photo_001.png",)):
    """A complete phase-4 reference so `seal` has something legal to accept."""
    return {
        "schema_version": 1,
        "listing_id": listing_id,
        "dataset_version": "renovation-v1",
        "tier": "gold",
        "review_status": "reviewed",
        "annotation_phase": 4,
        "reference_coverage": "targeted",
        "vocabulary_fingerprint": vocabulary["fingerprint"],
        "rooms": [{"room_id": "kitchen-1", "room_type": "kitchen"}],
        "photo_expectations": [
            {"photo_key": key, "primary_room_id": "kitchen-1",
             "visible_room_ids": ["kitchen-1"], "expected_group": "kitchen",
             "accepted_scene_ids": ["kitchen"], "indeterminate": False}
            for key in photo_keys
        ],
        "findings": [{
            "id": "finding-001",
            "description": "Ceiling drywall crack above the sink run",
            "room_id": "kitchen-1", "presence": "present",
            "reporting_expectation": "required", "visual_sufficiency": "sufficient",
            "catalog_status": "matched",
            "catalog_item_ids": ["damaged_drywall_or_cracks"],
            "critical": False, "aliases": [], "actionability": "repair",
            "billable_component_key": "kitchen-1-drywall",
            "evidence": [{"photo_key": photo_keys[0]}],
        }],
        "billable_components": [{
            "key": "kitchen-1-drywall", "room_id": "kitchen-1",
            "label": "Kitchen ceiling drywall repair",
            "finding_ids": ["finding-001"], "expected_units": 1,
        }],
        "packages": [{
            "package_id": "pkg-001", "package_type": "kitchen_repair",
            "room_id": "kitchen-1", "decision": "expected",
            "confirmed_finding_ids": ["finding-001"],
            "component_keys": ["kitchen-1-drywall"], "reason": "Localized repair.",
        }],
        "scope": {"scope_band": "light", "band_confidence": "medium"},
        "cost": {"expected_band": {"low": 400, "high": 900}, "confidence": "medium",
                 "basis": "Given the human scope above."},
    }


@pytest.fixture
def populated(dataset_builder, frozen_vocabulary):
    """A one-listing draft dataset with a reviewed reference."""
    listing = dataset_builder.add_listing(photo_count=1)
    dataset_builder.write_reference(reviewed_reference(frozen_vocabulary))
    return dataset_builder, listing


class TestValidate:
    def test_clean_dataset_passes(self, populated):
        builder, _ = populated
        assert ds.validate_dataset(builder.path).errors == []

    def test_tampered_photo_byte_is_caught(self, populated):
        builder, _ = populated
        photo = builder.photo()
        photo.write_bytes(photo.read_bytes() + b"\x00")
        errors = ds.validate_dataset(builder.path).errors
        assert any("sha256 is" in e for e in errors)
        assert any("byte_size is" in e for e in errors)

    def test_lfs_pointer_is_caught(self, populated):
        """A clone without `git lfs pull` would otherwise "analyze" a text stub."""
        builder, _ = populated
        builder.photo().write_bytes(LFS_POINTER)
        assert any("Git LFS pointer" in e and "git lfs pull" in e
                   for e in ds.validate_dataset(builder.path).errors)

    def test_missing_photo_is_caught(self, populated):
        builder, _ = populated
        builder.photo().unlink()
        assert any("photo file is missing" in e for e in ds.validate_dataset(builder.path).errors)

    def test_skip_photo_bytes_skips_hashing(self, populated):
        builder, _ = populated
        builder.photo().write_bytes(b"not even a png")
        assert ds.validate_dataset(builder.path, check_photo_bytes=False).errors == []

    def test_draft_without_a_reference_is_valid(self, dataset_builder):
        """Phase-1 annotation has not necessarily started yet."""
        dataset_builder.add_listing(photo_count=1)
        assert ds.validate_dataset(dataset_builder.path).errors == []

    def test_single_listing_filter(self, dataset_builder, frozen_vocabulary):
        dataset_builder.add_listing("hsv-001", photo_count=1)
        dataset_builder.add_listing("hsv-002", photo_count=1)
        dataset_builder.photo("hsv-002").write_bytes(b"broken")
        assert ds.validate_dataset(dataset_builder.path, listing_ids=["hsv-001"]).errors == []
        assert ds.validate_dataset(dataset_builder.path, listing_ids=["hsv-002"]).errors

    def test_traversing_metadata_path_is_refused(self, dataset_builder, tmp_path):
        """Caught at the cheapest layer: the manifest shape check, before
        anything touches the filesystem."""
        atomic_json(tmp_path / "outside.json", {"schema_version": 1})
        dataset_builder.add_listing(photo_count=1)
        entries = dataset_builder.read_manifest()["listings"]
        entries[0]["metadata_path"] = "listings/hsv-001/../../../outside.json"
        dataset_builder.write_manifest(entries)
        with pytest.raises(ComparisonError, match="must not traverse upward"):
            ds.validate_dataset(dataset_builder.path)

    @pytest.mark.parametrize("relative", [
        "listings/../../outside.json",
        "../sibling/metadata.json",
    ])
    def test_resolve_within_refuses_an_escape(self, dataset_builder, relative):
        """The second layer: containment on resolved paths, so a `..` that got
        past the shape check still cannot read outside the dataset."""
        with pytest.raises(ComparisonError, match="outside the dataset directory"):
            ds._resolve_within(dataset_builder.path, relative, label="metadata_path")

    def test_resolve_within_allows_a_contained_path(self, dataset_builder):
        resolved = ds._resolve_within(dataset_builder.path, "listings/hsv-001/metadata.json",
                                      label="metadata_path")
        assert resolved.is_relative_to(dataset_builder.path.resolve())

    def test_undeclared_duplicate_photos_are_caught(self, dataset_builder):
        dataset_builder.add_listing(photo_count=3, duplicate_last=True)
        assert any("share sha256" in e for e in ds.validate_dataset(dataset_builder.path).errors)


class TestPhotoOrder:
    def test_photo_paths_follow_frozen_order_not_filesystem_order(self, dataset_builder):
        """Production sorts filenames at serve time with two different
        comparators. Order is resolved once, at import, and read back verbatim."""
        listing = dataset_builder.add_listing(photo_count=3)
        reversed_listing = dict(listing)
        reversed_listing["photos"] = [
            dict(p, order=3 - index) for index, p in enumerate(listing["photos"])
        ]
        paths = ds.photo_paths(dataset_builder.path, "hsv-001", reversed_listing)
        assert [p.name for p in paths] == ["photo_003.png", "photo_002.png", "photo_001.png"]

    def test_photo_paths_reject_an_escaping_filename(self, dataset_builder):
        listing = dataset_builder.add_listing(photo_count=1)
        listing["photos"][0]["filename"] = "../../../etc/passwd"
        with pytest.raises(ComparisonError, match="outside the dataset directory"):
            ds.photo_paths(dataset_builder.path, "hsv-001", listing)


class TestFingerprint:
    def test_stable_across_key_reordering(self, populated):
        builder, _ = populated
        manifest = builder.read_manifest()
        before = ds.dataset_fingerprint(builder.path, manifest)
        # Rewrite the metadata with keys in a different order; content is equal.
        path = builder.listing_path() / "metadata.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(dict(reversed(list(data.items()))), indent=2),
                        encoding="utf-8")
        assert ds.dataset_fingerprint(builder.path, manifest) == before

    def test_changes_when_a_photo_hash_changes(self, populated):
        builder, _ = populated
        manifest = builder.read_manifest()
        before = ds.dataset_fingerprint(builder.path, manifest)
        path = builder.listing_path() / "metadata.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["photos"][0]["sha256"] = "f" * 64
        atomic_json(path, data)
        assert ds.dataset_fingerprint(builder.path, manifest) != before

    def test_changes_when_the_reference_changes(self, populated, frozen_vocabulary):
        builder, _ = populated
        manifest = builder.read_manifest()
        before = ds.dataset_fingerprint(builder.path, manifest)
        reference = reviewed_reference(frozen_vocabulary)
        reference["findings"][0]["description"] = "Different description"
        builder.write_reference(reference)
        assert ds.dataset_fingerprint(builder.path, manifest) != before

    def test_excludes_state_so_sealing_does_not_change_it(self, populated):
        """Otherwise the fingerprint recorded at seal time could never match."""
        builder, _ = populated
        manifest = builder.read_manifest()
        draft = ds.dataset_fingerprint(builder.path, manifest)
        assert ds.dataset_fingerprint(builder.path, dict(manifest, state="sealed")) == draft

    def test_changes_when_a_slice_changes(self, populated):
        builder, _ = populated
        manifest = builder.read_manifest()
        before = ds.dataset_fingerprint(builder.path, manifest)
        manifest["listings"][0]["slices"] = ["gold-holdout"]
        assert ds.dataset_fingerprint(builder.path, manifest) != before


class TestSeal:
    def test_writes_vocabulary_and_fingerprints(self, populated, issue_catalog):
        builder, _ = populated
        manifest = ds.seal(builder.path, issue_catalog)
        assert manifest["state"] == "sealed"
        assert manifest["vocabulary_fingerprint"]
        assert manifest["dataset_fingerprint"]
        assert (builder.path / ds.VOCABULARY_FILENAME).is_file()
        assert ds.validate_dataset(builder.path).errors == []

    def test_refuses_a_draft_reference(self, dataset_builder, issue_catalog, frozen_vocabulary):
        """Sealing a draft reference would gate releases on unreviewed truth."""
        dataset_builder.add_listing(photo_count=1)
        reference = reviewed_reference(frozen_vocabulary)
        reference["review_status"] = "draft"
        dataset_builder.write_reference(reference)
        with pytest.raises(ComparisonError, match="review_status is 'draft'"):
            ds.seal(dataset_builder.path, issue_catalog)

    def test_refuses_a_missing_reference(self, dataset_builder, issue_catalog):
        dataset_builder.add_listing(photo_count=1)
        with pytest.raises(ComparisonError, match="no compiled reference"):
            ds.seal(dataset_builder.path, issue_catalog)

    def test_sealed_fingerprint_detects_later_tampering(self, populated, issue_catalog):
        builder, _ = populated
        ds.seal(builder.path, issue_catalog)
        path = builder.listing_path() / "metadata.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["asking_price"] = 999999
        atomic_json(path, data)
        assert any("contents hash to" in e for e in ds.validate_dataset(builder.path).errors)

    def test_sealed_dataset_missing_its_vocabulary_is_caught(self, populated, issue_catalog):
        builder, _ = populated
        ds.seal(builder.path, issue_catalog)
        (builder.path / ds.VOCABULARY_FILENAME).unlink()
        assert any("missing reference_vocabulary.json" in e
                   for e in ds.validate_dataset(builder.path).errors)


class TestSealedImmutability:
    @pytest.mark.parametrize("operation", ["importing", "compiling", "sealing"])
    def test_require_unsealed_blocks_every_mutation(self, operation):
        with pytest.raises(ComparisonError, match="is sealed"):
            ds.require_unsealed({"state": "sealed", "dataset_version": "renovation-v1"}, operation)

    def test_require_unsealed_allows_a_draft(self):
        ds.require_unsealed({"state": "draft"}, "importing")

    def test_import_refuses_a_sealed_dataset(self, populated, issue_catalog, tmp_path):
        builder, _ = populated
        ds.seal(builder.path, issue_catalog)
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(ComparisonError, match="is sealed"):
            ds.import_listing(builder.path, listing_id="hsv-002", source_dir=source)


class TestImport:
    def test_assigns_contiguous_order_and_real_hashes(self, dataset_builder, tmp_path):
        from tests.conftest import tiny_png

        source = tmp_path / "src"
        source.mkdir()
        for index, color in enumerate([(1, 2, 3), (4, 5, 6), (7, 8, 9)]):
            (source / f"photo_{index + 1:03d}.png").write_bytes(tiny_png(*color))

        listing = ds.import_listing(dataset_builder.path, listing_id="hsv-001",
                                    source_dir=source, tier="gold",
                                    slices=["gold-development"],
                                    market_inputs={"area_ppsf": 118.0, "source": "manual",
                                                   "sample_size": 5})
        assert [p["order"] for p in listing["photos"]] == [1, 2, 3]
        assert len({p["sha256"] for p in listing["photos"]}) == 3
        assert all(len(p["sha256"]) == 64 for p in listing["photos"])

        entry = ds.manifest_entry(dataset_builder.read_manifest(), "hsv-001")
        assert entry["tier"] == "gold"
        assert entry["slices"] == ["gold-development"]

    def test_declares_byte_identical_copies(self, dataset_builder, tmp_path):
        """Duplicates get declared at import so validation passes, while the
        duplicate stays named explicitly in the metadata."""
        from tests.conftest import tiny_png

        source = tmp_path / "src"
        source.mkdir()
        (source / "a.png").write_bytes(tiny_png(1, 2, 3))
        (source / "b.png").write_bytes(tiny_png(1, 2, 3))
        listing = ds.import_listing(dataset_builder.path, listing_id="hsv-001",
                                    source_dir=source,
                                    market_inputs={"area_ppsf": 1.0, "source": "manual"})
        assert listing["photos"][1]["intentional_duplicate_of"] == "a.png"

    def test_rejects_a_source_with_no_images(self, dataset_builder, tmp_path):
        source = tmp_path / "empty"
        source.mkdir()
        with pytest.raises(ComparisonError, match="no images found"):
            ds.import_listing(dataset_builder.path, listing_id="hsv-001", source_dir=source)

    def test_rejects_a_missing_source(self, dataset_builder, tmp_path):
        with pytest.raises(ComparisonError, match="source image directory not found"):
            ds.import_listing(dataset_builder.path, listing_id="hsv-001",
                              source_dir=tmp_path / "ghost")

    def test_reimport_replaces_rather_than_duplicates_the_entry(self, dataset_builder, tmp_path):
        from tests.conftest import tiny_png

        source = tmp_path / "src"
        source.mkdir()
        (source / "a.png").write_bytes(tiny_png(1, 2, 3))
        for _ in range(2):
            ds.import_listing(dataset_builder.path, listing_id="hsv-001", source_dir=source,
                              market_inputs={"area_ppsf": 1.0, "source": "manual"})
        assert len(dataset_builder.read_manifest()["listings"]) == 1


class TestManifestHelpers:
    def test_unknown_listing_lists_what_is_known(self, populated):
        builder, _ = populated
        with pytest.raises(ComparisonError, match="hsv-001"):
            ds.manifest_entry(builder.read_manifest(), "ghost")

    def test_load_listing_validates(self, populated):
        builder, _ = populated
        manifest = builder.read_manifest()
        path = builder.listing_path() / "metadata.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["sqft"] = -5
        atomic_json(path, data)
        with pytest.raises(ComparisonError, match="sqft"):
            ds.load_listing(builder.path, manifest, "hsv-001")
