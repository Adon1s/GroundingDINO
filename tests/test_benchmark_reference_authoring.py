"""Reference authoring: template generation, YAML compilation, and defaulting.

The load-bearing property here is that compilation **materializes every default
explicitly**. A sealed reference must not depend on an implicit default, because
changing that default later would retroactively alter truth a human already
reviewed and a baseline was already accepted against.
"""
import json

import pytest
import yaml

from tools.benchmarking import reference_authoring as authoring
from tools.comparison_common import ComparisonError

MINIMAL_DRAFT = """
tier: gold
review_status: draft
annotation_phase: 1
reference_coverage: targeted

rooms:
  - room_id: kitchen-1
    room_type: kitchen

photo_expectations:
  - photo_key: photo_001.png
    primary_room_id: kitchen-1
    visible_room_ids: [kitchen-1]
    expected_group: kitchen
    accepted_scene_ids: [kitchen]

findings:
  - id: finding-001
    description: Ceiling drywall crack above the sink run
    room_id: kitchen-1
    catalog_item_ids: [damaged_drywall_or_cracks]
    evidence:
      - {photo_key: photo_001.png, note: primary}
"""


@pytest.fixture
def listing(dataset_builder):
    return dataset_builder.add_listing(photo_count=1)


class TestTemplate:
    def test_pre_lists_every_photo(self, dataset_builder, frozen_vocabulary):
        """Pre-listing is what stops a photo being silently skipped."""
        subject = dataset_builder.add_listing(photo_count=4)
        text = authoring.template(subject, frozen_vocabulary)
        parsed = yaml.safe_load(text)
        keys = [e["photo_key"] for e in parsed["photo_expectations"]]
        assert keys == ["photo_001.png", "photo_002.png", "photo_003.png", "photo_004.png"]

    def test_is_valid_yaml_and_starts_at_phase_one(self, listing, frozen_vocabulary):
        parsed = yaml.safe_load(authoring.template(listing, frozen_vocabulary))
        assert parsed["annotation_phase"] == 1
        assert parsed["review_status"] == "draft"
        assert parsed["rooms"] == []
        assert parsed["findings"] == []

    def test_contains_no_model_output(self, listing, frozen_vocabulary):
        """Blank-first: prefilling from a model would let that model define the
        universe of findings the gold set can contain."""
        parsed = yaml.safe_load(authoring.template(listing, frozen_vocabulary))
        assert parsed["findings"] == []
        assert all(e["accepted_scene_ids"] == [] for e in parsed["photo_expectations"])
        assert all(e["primary_room_id"] is None for e in parsed["photo_expectations"])

    def test_documents_the_controlled_vocabularies(self, listing, frozen_vocabulary):
        text = authoring.template(listing, frozen_vocabulary)
        for token in ("must_not_report", "missing_catalog_item", "inspection_risk",
                      "indeterminate", "expected_units", "critical"):
            assert token in text, f"template does not mention {token}"

    def test_hidden_system_conditions_example_is_false(self, listing, frozen_vocabulary):
        """The commented example must not contradict the advice above it."""
        text = authoring.template(listing, frozen_vocabulary)
        assert "#   hidden_system_conditions: false" in text

    def test_refuses_to_clobber_annotation_work(self, dataset_builder, listing,
                                               frozen_vocabulary):
        destination = dataset_builder.listing_path() / authoring.DRAFT_FILENAME
        authoring.write_template(destination, listing, frozen_vocabulary)
        with pytest.raises(ComparisonError, match="refusing to overwrite"):
            authoring.write_template(destination, listing, frozen_vocabulary)

    def test_force_overwrites(self, dataset_builder, listing, frozen_vocabulary):
        destination = dataset_builder.listing_path() / authoring.DRAFT_FILENAME
        authoring.write_template(destination, listing, frozen_vocabulary)
        destination.write_text("clobber me", encoding="utf-8")
        authoring.write_template(destination, listing, frozen_vocabulary, overwrite=True)
        assert "clobber me" not in destination.read_text(encoding="utf-8")


class TestNormalize:
    def _compile(self, dataset_builder, listing, vocabulary, draft=MINIMAL_DRAFT):
        path = dataset_builder.write_draft(draft)
        reference, result = authoring.check(path, listing=listing, vocabulary=vocabulary)
        return reference, result

    def test_materializes_every_default_explicitly(self, dataset_builder, listing,
                                                   frozen_vocabulary):
        reference, result = self._compile(dataset_builder, listing, frozen_vocabulary)
        assert result.ok, result.errors
        finding = reference["findings"][0]
        assert finding["presence"] == "present"
        assert finding["reporting_expectation"] == "required"
        assert finding["catalog_status"] == "matched"
        assert finding["visual_sufficiency"] == "sufficient"
        assert finding["critical"] is False

    def test_derives_actionability_from_the_catalog(self, dataset_builder, listing,
                                                   frozen_vocabulary):
        reference, _ = self._compile(dataset_builder, listing, frozen_vocabulary)
        assert reference["findings"][0]["actionability"] == "repair"

    def test_stamps_the_vocabulary_fingerprint(self, dataset_builder, listing,
                                              frozen_vocabulary):
        reference, _ = self._compile(dataset_builder, listing, frozen_vocabulary)
        assert reference["vocabulary_fingerprint"] == frozen_vocabulary["fingerprint"]

    def test_takes_listing_identity_from_the_listing_not_the_draft(self, dataset_builder,
                                                                  listing, frozen_vocabulary):
        """A draft cannot lie about which listing it describes."""
        draft = MINIMAL_DRAFT + "\nlisting_id: some-other-listing\n"
        reference, _ = self._compile(dataset_builder, listing, frozen_vocabulary, draft)
        assert reference["listing_id"] == "hsv-001"

    def test_fills_primary_room_into_visible_rooms(self, dataset_builder, listing,
                                                   frozen_vocabulary):
        """Omitting visible_room_ids is an authoring slip, not a claim."""
        draft = MINIMAL_DRAFT.replace("    visible_room_ids: [kitchen-1]\n", "")
        reference, result = self._compile(dataset_builder, listing, frozen_vocabulary, draft)
        assert result.ok, result.errors
        assert reference["photo_expectations"][0]["visible_room_ids"] == ["kitchen-1"]

    def test_omits_later_phase_blocks_when_absent(self, dataset_builder, listing,
                                                  frozen_vocabulary):
        reference, _ = self._compile(dataset_builder, listing, frozen_vocabulary)
        assert "packages" not in reference
        assert "billable_components" not in reference
        assert "cost" not in reference

    def test_explicit_values_win_over_defaults(self, dataset_builder, listing,
                                               frozen_vocabulary):
        draft = MINIMAL_DRAFT.replace(
            "    catalog_item_ids: [damaged_drywall_or_cracks]",
            "    catalog_item_ids: [damaged_drywall_or_cracks]\n"
            "    reporting_expectation: acceptable\n"
            "    visual_sufficiency: limited",
        )
        reference, result = self._compile(dataset_builder, listing, frozen_vocabulary, draft)
        assert result.ok, result.errors
        assert reference["findings"][0]["reporting_expectation"] == "acceptable"
        assert reference["findings"][0]["visual_sufficiency"] == "limited"

    def test_rejects_a_non_mapping_draft(self):
        with pytest.raises(ComparisonError, match="must be a YAML mapping"):
            authoring.normalize(["not", "a", "mapping"], listing={}, vocabulary={})


class TestDeriveActionability:
    def test_single_matched_item(self, frozen_vocabulary):
        assert authoring.derive_actionability(
            ["damaged_drywall_or_cracks"], frozen_vocabulary) == "repair"

    def test_agreeing_items_collapse(self, frozen_vocabulary):
        table = frozen_vocabulary["actionability_by_catalog_item"]
        repairs = [k for k, v in sorted(table.items()) if v == "repair"][:2]
        assert authoring.derive_actionability(repairs, frozen_vocabulary) == "repair"

    def test_disagreeing_items_return_none(self, frozen_vocabulary):
        """A finding mapping to both a repair and a modernization item is a
        decision a human should make, not one to guess from ordering."""
        table = frozen_vocabulary["actionability_by_catalog_item"]
        repair = next(k for k, v in sorted(table.items()) if v == "repair")
        modernization = next(k for k, v in sorted(table.items()) if v == "modernization")
        assert authoring.derive_actionability([repair, modernization], frozen_vocabulary) is None

    def test_unknown_item_returns_none(self, frozen_vocabulary):
        assert authoring.derive_actionability(["not_a_real_item"], frozen_vocabulary) is None


class TestCompile:
    def test_writes_canonical_json(self, dataset_builder, listing, frozen_vocabulary):
        path = dataset_builder.write_draft(MINIMAL_DRAFT)
        written = authoring.compile_draft(path, listing=listing, vocabulary=frozen_vocabulary)
        assert written.name == authoring.COMPILED_FILENAME
        text = written.read_text(encoding="utf-8")
        data = json.loads(text)
        assert list(data.keys()) == sorted(data.keys()), "compiled JSON must have sorted keys"

    def test_is_deterministic(self, dataset_builder, listing, frozen_vocabulary):
        path = dataset_builder.write_draft(MINIMAL_DRAFT)
        first = authoring.compile_draft(path, listing=listing,
                                       vocabulary=frozen_vocabulary).read_bytes()
        second = authoring.compile_draft(path, listing=listing,
                                        vocabulary=frozen_vocabulary).read_bytes()
        assert first == second

    def test_refuses_to_emit_on_a_validation_error(self, dataset_builder, listing,
                                                   frozen_vocabulary):
        draft = MINIMAL_DRAFT.replace("damaged_drywall_or_cracks", "totally_invented_item")
        path = dataset_builder.write_draft(draft)
        with pytest.raises(ComparisonError, match="not in the frozen vocabulary"):
            authoring.compile_draft(path, listing=listing, vocabulary=frozen_vocabulary)
        assert not (dataset_builder.listing_path() / authoring.COMPILED_FILENAME).exists()

    def test_reports_every_error_at_once(self, dataset_builder, listing, frozen_vocabulary):
        draft = MINIMAL_DRAFT.replace(
            "    room_id: kitchen-1\n    catalog_item_ids: [damaged_drywall_or_cracks]",
            "    room_id: ghost-room\n    catalog_item_ids: [invented_one, invented_two]",
        )
        path = dataset_builder.write_draft(draft)
        with pytest.raises(ComparisonError) as excinfo:
            authoring.compile_draft(path, listing=listing, vocabulary=frozen_vocabulary)
        assert str(excinfo.value).count("  - ") >= 3

    def test_round_trip_survives_recompilation(self, dataset_builder, listing,
                                               frozen_vocabulary):
        """Compiling, then re-validating the compiled output, must agree."""
        from tools.benchmarking.schemas import validate_reference

        path = dataset_builder.write_draft(MINIMAL_DRAFT)
        written = authoring.compile_draft(path, listing=listing, vocabulary=frozen_vocabulary)
        compiled = json.loads(written.read_text(encoding="utf-8"))
        result = validate_reference(compiled, listing=listing, vocabulary=frozen_vocabulary)
        assert result.errors == []


class TestLoadDraft:
    def test_rejects_missing(self, tmp_path):
        with pytest.raises(ComparisonError, match="not found"):
            authoring.load_draft(tmp_path / "ghost.yaml")

    def test_rejects_empty(self, dataset_builder):
        path = dataset_builder.write_draft("# only a comment\n")
        with pytest.raises(ComparisonError, match="is empty"):
            authoring.load_draft(path)

    def test_rejects_invalid_yaml(self, dataset_builder):
        path = dataset_builder.write_draft("findings: [unclosed\n")
        with pytest.raises(ComparisonError, match="not valid YAML"):
            authoring.load_draft(path)

    def test_rejects_a_yaml_list(self, dataset_builder):
        path = dataset_builder.write_draft("- one\n- two\n")
        with pytest.raises(ComparisonError, match="must be a mapping"):
            authoring.load_draft(path)
