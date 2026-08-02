"""Reference/listing/manifest/config validation and the reporting-category derivation.

The combination-rule tests are the load-bearing ones: they are what keeps
``reporting_category`` disjoint, so it needs no precedence rules.
"""
import copy

import pytest

from tools.benchmarking.schemas import (
    reporting_category,
    validate_config,
    validate_listing,
    validate_manifest,
    validate_reference,
)


def finding(**overrides):
    base = {
        "id": "finding-001",
        "description": "Ceiling drywall crack above the sink run",
        "room_id": "kitchen-1",
        "presence": "present",
        "reporting_expectation": "required",
        "visual_sufficiency": "sufficient",
        "catalog_status": "matched",
        "catalog_item_ids": ["damaged_drywall_or_cracks"],
        "critical": False,
        "aliases": [],
        "evidence": [{"photo_key": "photo_001.png"}],
    }
    base.update(overrides)
    return base


def reference(**overrides):
    base = {
        "schema_version": 1,
        "listing_id": "hsv-001",
        "dataset_version": "renovation-v1",
        "tier": "gold",
        "review_status": "draft",
        "annotation_phase": 1,
        "reference_coverage": "targeted",
        "rooms": [{"room_id": "kitchen-1", "room_type": "kitchen"}],
        "photo_expectations": [{
            "photo_key": "photo_001.png",
            "primary_room_id": "kitchen-1",
            "visible_room_ids": ["kitchen-1"],
            "expected_group": "kitchen",
            "accepted_scene_ids": ["kitchen"],
            "indeterminate": False,
        }],
        "findings": [finding()],
    }
    base.update(overrides)
    return base


def listing(**overrides):
    base = {
        "schema_version": 1,
        "listing_id": "hsv-001",
        "dataset_version": "renovation-v1",
        "asking_price": 185000,
        "sqft": 1450,
        "beds": 3,
        "baths": 2,
        "market_inputs": {"area_ppsf": 118.0, "source": "manual", "sample_size": 12},
        "photos": [{
            "order": 1, "filename": "photo_001.png",
            "sha256": "a" * 64, "byte_size": 69,
        }],
    }
    base.update(overrides)
    return base


def errors_for(ref, **kwargs):
    return validate_reference(ref, **kwargs).errors


# ---------------------------------------------------------------------------
# reporting_category
# ---------------------------------------------------------------------------

class TestReportingCategory:
    @pytest.mark.parametrize("presence,expectation,expected", [
        ("present", "required", "required"),
        ("present", "acceptable", "acceptable"),
        ("present", "must_not_report", "unsupported"),
        ("absent", "must_not_report", "unsupported"),
        ("indeterminate", "required", "indeterminate"),
        ("indeterminate", "acceptable", "indeterminate"),
    ])
    def test_derivation(self, presence, expectation, expected):
        assert reporting_category(
            {"presence": presence, "reporting_expectation": expectation}
        ) == expected

    def test_the_sol_case(self, frozen_vocabulary):
        """Real drywall damage is `required` even while the kitchen-modernization
        package it might imply is `unsupported`. One enum could not say both."""
        ref = reference(findings=[finding(id="finding-001")])
        assert errors_for(ref, vocabulary=frozen_vocabulary) == []
        assert reporting_category(ref["findings"][0]) == "required"

    def test_insufficient_photos_land_on_indeterminate(self):
        f = finding(presence="indeterminate", visual_sufficiency="insufficient",
                    reporting_expectation="required")
        assert reporting_category(f) == "indeterminate"


# ---------------------------------------------------------------------------
# Combination rules
# ---------------------------------------------------------------------------

class TestCombinationRules:
    def test_absent_requires_must_not_report(self, frozen_vocabulary):
        ref = reference(findings=[finding(presence="absent", reporting_expectation="required",
                                          evidence=[])])
        assert any("requires reporting_expectation 'must_not_report'" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_indeterminate_cannot_be_must_not_report(self, frozen_vocabulary):
        ref = reference(findings=[finding(presence="indeterminate",
                                          reporting_expectation="must_not_report", evidence=[])])
        assert any("cannot be 'must_not_report'" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_insufficient_requires_indeterminate_presence(self, frozen_vocabulary):
        ref = reference(findings=[finding(visual_sufficiency="insufficient")])
        matching = [e for e in errors_for(ref, vocabulary=frozen_vocabulary)
                    if "visual_sufficiency 'insufficient'" in e]
        assert len(matching) == 1, f"expected exactly one error, got {matching}"

    def test_indeterminate_cannot_be_critical(self, frozen_vocabulary):
        ref = reference(findings=[finding(presence="indeterminate", critical=True, evidence=[])])
        assert any("critical requires presence 'present'" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    @pytest.mark.parametrize("overrides", [
        {"reporting_expectation": "acceptable"},
        {"visual_sufficiency": "limited"},
        {"presence": "indeterminate", "evidence": []},
    ])
    def test_critical_requires_present_required_sufficient(self, overrides, frozen_vocabulary):
        ref = reference(findings=[finding(critical=True, **overrides)])
        assert any("critical requires" in e for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_present_plus_must_not_report_is_allowed(self, frozen_vocabulary):
        """A real but trivial condition a good analyzer should stay quiet about.
        This is the mechanism for scoring noise, so it must not be rejected."""
        ref = reference(findings=[finding(reporting_expectation="must_not_report")])
        assert errors_for(ref, vocabulary=frozen_vocabulary) == []

    def test_indeterminate_plus_required_is_allowed(self, frozen_vocabulary):
        """Photos cannot settle a possible foundation issue, but a good analyzer
        should still flag it for inspection."""
        ref = reference(findings=[finding(
            presence="indeterminate", visual_sufficiency="insufficient",
            reporting_expectation="required", catalog_status="missing_catalog_item",
            catalog_item_ids=[], actionability="inspection_risk", evidence=[],
        )])
        assert errors_for(ref, vocabulary=frozen_vocabulary) == []

    def test_present_plus_limited_is_allowed(self, frozen_vocabulary):
        """Visible but not sizable is `present` + `limited`, not indeterminate."""
        ref = reference(findings=[finding(visual_sufficiency="limited")])
        assert errors_for(ref, vocabulary=frozen_vocabulary) == []


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

class TestFindings:
    def test_present_requires_evidence(self, frozen_vocabulary):
        ref = reference(findings=[finding(evidence=[])])
        assert any("requires at least one evidence occurrence" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_duplicate_evidence_photo_is_rejected(self, frozen_vocabulary):
        ref = reference(findings=[finding(evidence=[{"photo_key": "photo_001.png"},
                                                    {"photo_key": "photo_001.png"}])])
        assert any("duplicate evidence photo_key" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_missing_catalog_item_must_have_empty_ids(self, frozen_vocabulary):
        ref = reference(findings=[finding(catalog_status="missing_catalog_item",
                                          actionability="repair")])
        assert any("must be empty" in e for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_matched_requires_catalog_ids(self, frozen_vocabulary):
        ref = reference(findings=[finding(catalog_item_ids=[])])
        assert any("catalog_item_ids is required" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_unknown_catalog_id_is_rejected(self, frozen_vocabulary):
        ref = reference(findings=[finding(catalog_item_ids=["totally_invented"])])
        assert any("not in the frozen vocabulary" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_missing_catalog_item_requires_hand_authored_actionability(self, frozen_vocabulary):
        """It cannot be derived without a catalog item to derive it from."""
        ref = reference(findings=[finding(catalog_status="missing_catalog_item",
                                          catalog_item_ids=[])])
        assert any("actionability is required" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_duplicate_finding_id_is_rejected(self, frozen_vocabulary):
        ref = reference(findings=[finding(), finding()])
        assert any("duplicate finding id" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_unknown_room_is_rejected(self, frozen_vocabulary):
        ref = reference(findings=[finding(room_id="bathroom-9")])
        assert any("is not a declared room" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_reports_every_error_not_just_the_first(self, frozen_vocabulary):
        """A hand-authored reference has many findings; one error per run would
        make fixing a draft an afternoon of round-trips."""
        ref = reference(findings=[
            finding(id="finding-001", catalog_item_ids=["nope_one"]),
            finding(id="finding-002", catalog_item_ids=["nope_two"]),
            finding(id="finding-003", room_id="ghost-room"),
        ])
        assert len(errors_for(ref, vocabulary=frozen_vocabulary)) >= 3


# ---------------------------------------------------------------------------
# Rooms and photo expectations
# ---------------------------------------------------------------------------

class TestRoomsAndPhotoExpectations:
    def test_room_type_must_be_a_scene_id(self, frozen_vocabulary):
        ref = reference(rooms=[{"room_id": "kitchen-1", "room_type": "scullery"}])
        assert any("is not a scene id" in e for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_non_room_scene_is_rejected_as_a_room_type(self, frozen_vocabulary):
        ref = reference(rooms=[{"room_id": "kitchen-1", "room_type": "floor_plan"}])
        assert any("is not a physical room" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_primary_room_must_be_visible(self, frozen_vocabulary):
        ref = reference()
        ref["photo_expectations"][0]["visible_room_ids"] = []
        assert any("must also appear in visible_room_ids" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_expected_group_must_match_an_accepted_scene(self, frozen_vocabulary):
        ref = reference()
        ref["photo_expectations"][0]["expected_group"] = "bathroom"
        assert any("is not the group of any accepted scene id" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_open_plan_photo_spanning_groups_warns_but_passes(self, frozen_vocabulary):
        """One photo showing kitchen and living is legitimate; scoring credits
        either group."""
        ref = reference(rooms=[
            {"room_id": "kitchen-1", "room_type": "kitchen"},
            {"room_id": "living-1", "room_type": "living_room"},
        ])
        ref["photo_expectations"][0].update({
            "visible_room_ids": ["kitchen-1", "living-1"],
            "accepted_scene_ids": ["kitchen", "living_room"],
        })
        result = validate_reference(ref, vocabulary=frozen_vocabulary)
        assert result.ok
        assert any("span multiple groups" in w for w in result.warnings)

    def test_indeterminate_photo_needs_no_scene(self, frozen_vocabulary):
        ref = reference()
        ref["photo_expectations"][0] = {
            "photo_key": "photo_001.png", "primary_room_id": None,
            "visible_room_ids": [], "expected_group": None,
            "accepted_scene_ids": [], "indeterminate": True,
        }
        assert errors_for(ref, vocabulary=frozen_vocabulary) == []

    def test_every_photo_needs_an_entry(self, frozen_vocabulary):
        subject = listing(photos=[
            {"order": 1, "filename": "photo_001.png", "sha256": "a" * 64, "byte_size": 69},
            {"order": 2, "filename": "photo_002.png", "sha256": "b" * 64, "byte_size": 69},
        ])
        errors = errors_for(reference(), listing=subject, vocabulary=frozen_vocabulary)
        assert any("every photo needs exactly one entry" in e for e in errors)

    def test_duplicate_photo_entry_is_rejected(self, frozen_vocabulary):
        ref = reference()
        ref["photo_expectations"].append(copy.deepcopy(ref["photo_expectations"][0]))
        assert any("duplicate entry for photo" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_unknown_photo_key_is_rejected(self, frozen_vocabulary):
        errors = errors_for(reference(), listing=listing(), vocabulary=frozen_vocabulary)
        ref = reference()
        ref["photo_expectations"][0]["photo_key"] = "ghost.png"
        assert any("is not a photo in this listing" in e
                   for e in errors_for(ref, listing=listing(), vocabulary=frozen_vocabulary))
        assert errors == []


# ---------------------------------------------------------------------------
# Components, packages, cost
# ---------------------------------------------------------------------------

class TestComponentsAndPackages:
    def _phase2(self, **overrides):
        base = reference(
            annotation_phase=2,
            findings=[finding(billable_component_key="kitchen-1-drywall")],
            billable_components=[{
                "key": "kitchen-1-drywall", "room_id": "kitchen-1",
                "label": "Kitchen ceiling drywall repair",
                "finding_ids": ["finding-001"], "expected_units": 1,
            }],
            packages=[{
                "package_id": "pkg-001", "package_type": "kitchen_repair",
                "room_id": "kitchen-1", "decision": "expected",
                "confirmed_finding_ids": ["finding-001"],
                "component_keys": ["kitchen-1-drywall"], "reason": "Localized repair.",
            }],
        )
        base.update(overrides)
        return base

    def test_valid_phase_two_reference(self, frozen_vocabulary):
        assert errors_for(self._phase2(), vocabulary=frozen_vocabulary) == []

    def test_expected_units_must_be_at_least_one(self, frozen_vocabulary):
        ref = self._phase2()
        ref["billable_components"][0]["expected_units"] = 0
        assert any("expected_units must be an integer >= 1" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_component_backref_must_be_symmetric(self, frozen_vocabulary):
        """An asymmetric link would compare a component's expected_units against
        the wrong evidence."""
        ref = self._phase2()
        ref["findings"][0].pop("billable_component_key")
        assert any("does not point back at this component" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_component_must_list_the_finding_that_points_at_it(self, frozen_vocabulary):
        ref = self._phase2()
        ref["billable_components"][0]["finding_ids"] = []
        errors = errors_for(ref, vocabulary=frozen_vocabulary)
        assert any("finding_ids must list at least one finding" in e for e in errors)
        assert any("does not list it in finding_ids" in e for e in errors)

    def test_unsupported_package_requires_a_reason(self, frozen_vocabulary):
        """That reason is exactly the judgment the 2f judge is scored against."""
        ref = self._phase2()
        ref["packages"][0].update({"decision": "unsupported", "reason": ""})
        assert any("requires a reason" in e for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_expected_package_requires_confirmed_findings(self, frozen_vocabulary):
        ref = self._phase2()
        ref["packages"][0]["confirmed_finding_ids"] = []
        assert any("requires at least one confirmed_finding_ids" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_unknown_package_type_is_rejected(self, frozen_vocabulary):
        ref = self._phase2()
        ref["packages"][0]["package_type"] = "solarium_modernization"
        assert any("not in the frozen vocabulary" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_cost_must_not_restate_market_inputs(self, frozen_vocabulary):
        """Two copies would eventually disagree and silently change cost scoring."""
        ref = self._phase2(cost={
            "expected_band": {"low": 1000, "high": 2000}, "confidence": "medium",
            "basis": "Given the scope.", "market_assumptions": {"area_ppsf": 99},
        })
        assert any("must not appear in the reference" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_inverted_cost_band_is_rejected(self, frozen_vocabulary):
        ref = self._phase2(cost={
            "expected_band": {"low": 9000, "high": 1000}, "confidence": "medium",
            "basis": "Given the scope.",
        })
        assert any("exceeds high" in e for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_unknown_scope_band_is_rejected(self, frozen_vocabulary):
        ref = self._phase2(scope={"scope_band": "catastrophic", "band_confidence": "low"})
        assert any("scope_band 'catastrophic'" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))


# ---------------------------------------------------------------------------
# Phase awareness and coverage
# ---------------------------------------------------------------------------

class TestPhaseAwareness:
    def test_phase_one_draft_without_packages_is_valid(self, frozen_vocabulary):
        assert errors_for(reference(), vocabulary=frozen_vocabulary) == []

    def test_phase_two_requires_packages(self, frozen_vocabulary):
        ref = reference(annotation_phase=2)
        errors = errors_for(ref, vocabulary=frozen_vocabulary)
        assert any("requires packages" in e for e in errors)
        assert any("requires billable_components" in e for e in errors)

    def test_phase_four_requires_cost_and_scope(self, frozen_vocabulary):
        ref = reference(annotation_phase=4, packages=[], billable_components=[])
        errors = errors_for(ref, vocabulary=frozen_vocabulary)
        assert any("requires cost" in e for e in errors)
        assert any("requires scope" in e for e in errors)

    def test_reviewed_requires_every_phase_complete(self, frozen_vocabulary):
        ref = reference(review_status="reviewed")
        assert any("requires all annotation phases complete" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_exhaustive_requires_coverage_domain(self, frozen_vocabulary):
        ref = reference(reference_coverage="exhaustive")
        assert any("requires a coverage_domain" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))

    def test_coverage_domain_needs_every_key(self, frozen_vocabulary):
        ref = reference(reference_coverage="exhaustive",
                        coverage_domain={"visible_conditions": True})
        assert len(errors_for(ref, vocabulary=frozen_vocabulary)) >= 4

    def test_hidden_system_conditions_warns(self, frozen_vocabulary):
        """Photographs cannot establish hidden electrical or foundation state."""
        ref = reference(reference_coverage="exhaustive", coverage_domain={
            "visible_conditions": True, "hidden_system_conditions": True,
            "exterior": True, "interior": True, "marketability_opportunities": True,
        })
        result = validate_reference(ref, vocabulary=frozen_vocabulary)
        assert any("photographs cannot establish" in w for w in result.warnings)

    def test_vocabulary_fingerprint_mismatch_is_rejected(self, frozen_vocabulary):
        ref = reference(vocabulary_fingerprint="deadbeef" * 8)
        assert any("does not match the dataset's" in e
                   for e in errors_for(ref, vocabulary=frozen_vocabulary))


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

class TestListing:
    def test_valid(self):
        assert validate_listing(listing(), listing_id="hsv-001").errors == []

    def test_id_must_match_the_manifest(self):
        assert any("does not match manifest id" in e
                   for e in validate_listing(listing(), listing_id="other").errors)

    @pytest.mark.parametrize("field", ["asking_price", "sqft"])
    def test_positive_numbers_required(self, field):
        assert any(field in e for e in validate_listing(listing(**{field: None})).errors)

    def test_market_inputs_required(self):
        assert any("frozen market inputs are required" in e
                   for e in validate_listing(listing(market_inputs=None)).errors)

    def test_order_must_be_contiguous(self):
        subject = listing(photos=[
            {"order": 1, "filename": "a.png", "sha256": "a" * 64, "byte_size": 1},
            {"order": 3, "filename": "b.png", "sha256": "b" * 64, "byte_size": 1},
        ])
        assert any("contiguous 1..2" in e for e in validate_listing(subject).errors)

    def test_duplicate_order_is_rejected(self):
        subject = listing(photos=[
            {"order": 1, "filename": "a.png", "sha256": "a" * 64, "byte_size": 1},
            {"order": 1, "filename": "b.png", "sha256": "b" * 64, "byte_size": 1},
        ])
        assert any("contiguous" in e for e in validate_listing(subject).errors)

    def test_duplicate_filename_is_rejected(self):
        subject = listing(photos=[
            {"order": 1, "filename": "a.png", "sha256": "a" * 64, "byte_size": 1},
            {"order": 2, "filename": "a.png", "sha256": "b" * 64, "byte_size": 1},
        ])
        assert any("duplicate filename" in e for e in validate_listing(subject).errors)

    def test_filename_must_be_a_basename(self):
        subject = listing(photos=[
            {"order": 1, "filename": "../escape.png", "sha256": "a" * 64, "byte_size": 1},
        ])
        assert any("bare basename" in e for e in validate_listing(subject).errors)

    def test_bad_sha256_is_rejected(self):
        subject = listing(photos=[
            {"order": 1, "filename": "a.png", "sha256": "NOTAHASH", "byte_size": 1},
        ])
        assert any("64 lowercase hex" in e for e in validate_listing(subject).errors)

    def test_undeclared_duplicate_hash_is_rejected(self):
        """Usually an import bug: the same file copied twice."""
        subject = listing(photos=[
            {"order": 1, "filename": "a.png", "sha256": "c" * 64, "byte_size": 1},
            {"order": 2, "filename": "b.png", "sha256": "c" * 64, "byte_size": 1},
        ])
        assert any("share sha256" in e for e in validate_listing(subject).errors)

    def test_declared_duplicate_hash_is_allowed(self):
        """Duplicate-stress slices need this, but the intent must be explicit."""
        subject = listing(photos=[
            {"order": 1, "filename": "a.png", "sha256": "c" * 64, "byte_size": 1},
            {"order": 2, "filename": "b.png", "sha256": "c" * 64, "byte_size": 1,
             "intentional_duplicate_of": "a.png"},
        ])
        assert validate_listing(subject).errors == []

    def test_duplicate_declaration_must_point_at_a_real_photo(self):
        subject = listing(photos=[
            {"order": 1, "filename": "a.png", "sha256": "c" * 64, "byte_size": 1},
            {"order": 2, "filename": "b.png", "sha256": "c" * 64, "byte_size": 1,
             "intentional_duplicate_of": "ghost.png"},
        ])
        assert any("not a photo in this listing" in e for e in validate_listing(subject).errors)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def _manifest(self, **overrides):
        base = {
            "schema_version": 1, "dataset_id": "renovation",
            "dataset_version": "renovation-v1", "state": "draft",
            "listings": [{
                "id": "hsv-001", "tier": "gold", "slices": ["gold-development"],
                "metadata_path": "listings/hsv-001/metadata.json",
                "reference_path": "listings/hsv-001/reference.json",
            }],
        }
        base.update(overrides)
        return base

    def test_valid(self):
        assert validate_manifest(self._manifest()).errors == []

    def test_empty_draft_is_valid(self):
        """`import` is what adds the first listing, so requiring one here would
        make a fresh dataset unloadable."""
        assert validate_manifest(self._manifest(listings=[])).errors == []

    def test_sealed_requires_listings_and_fingerprints(self):
        errors = validate_manifest(self._manifest(state="sealed", listings=[])).errors
        assert any("at least one listing" in e for e in errors)
        assert any("dataset_fingerprint" in e for e in errors)
        assert any("vocabulary_fingerprint" in e for e in errors)

    def test_duplicate_listing_id_is_rejected(self):
        entry = self._manifest()["listings"][0]
        assert any("duplicate listing id" in e
                   for e in validate_manifest(self._manifest(listings=[entry, dict(entry)])).errors)

    @pytest.mark.parametrize("bad", ["/abs/metadata.json", "C:/abs/metadata.json",
                                     "../../escape.json"])
    def test_paths_must_be_relative_and_contained(self, bad):
        entry = dict(self._manifest()["listings"][0], metadata_path=bad)
        assert validate_manifest(self._manifest(listings=[entry])).errors

    def test_unknown_tier_is_rejected(self):
        entry = dict(self._manifest()["listings"][0], tier="bronze")
        assert any("tier" in e for e in validate_manifest(self._manifest(listings=[entry])).errors)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def _config(self, **overrides):
        base = {
            "schema_version": 1,
            "name": "terra-upstream-terra-2f",
            "axes": {"upstream": "terra", "pass_2f": "terra"},
            "pipeline": {"pass_toggles": {k: True for k in
                                          ("1a", "1b", "1c", "2a", "2b", "2c", "2d", "2e", "2f")},
                         "concurrency": 4},
            "model_map": {k: {"provider": "openai", "model": "gpt-5.6-terra"}
                          for k in ("1a", "2a", "2b", "2c", "2d", "2f")},
            "repetitions": 1,
        }
        base.update(overrides)
        return base

    def test_shipped_config_is_valid(self):
        import json
        from pathlib import Path

        path = (Path(__file__).resolve().parents[1] / "benchmarks" / "configs"
                / "terra-upstream-terra-2f.json")
        result = validate_config(json.loads(path.read_text(encoding="utf-8")))
        assert result.errors == [], result.errors

    def test_valid(self):
        assert validate_config(self._config()).errors == []

    @pytest.mark.parametrize("stub,reason", [
        ("1b", "compatibility stub"), ("1c", "compatibility stub"), ("2e", "rule-based"),
    ])
    def test_stub_passes_are_rejected_from_model_map(self, stub, reason):
        """A model entry for a pass that never calls an LLM would be silently
        ignored at runtime — exactly the config error this benchmark exists to catch."""
        config = self._config()
        config["model_map"][stub] = {"provider": "openai", "model": "gpt-5.6-terra"}
        errors = validate_config(config).errors
        assert any(reason in e for e in errors), errors

    def test_enabled_pass_without_a_model_entry_is_rejected(self):
        config = self._config()
        config["model_map"].pop("2d")
        assert any("no model entry" in e for e in validate_config(config).errors)

    def test_disabled_pass_needs_no_model_entry(self):
        config = self._config()
        config["model_map"].pop("2d")
        config["pipeline"]["pass_toggles"]["2d"] = False
        assert validate_config(config).errors == []

    def test_axes_are_required(self):
        assert any("axes" in e for e in validate_config(self._config(axes={})).errors)

    def test_unknown_pass_toggle_is_rejected(self):
        config = self._config()
        config["pipeline"]["pass_toggles"]["9z"] = True
        assert any("unknown pass" in e for e in validate_config(config).errors)

    def test_repetitions_must_be_positive(self):
        assert any("repetitions" in e for e in validate_config(self._config(repetitions=0)).errors)

    def test_semantic_threshold_range(self):
        config = self._config(evaluation={"semantic_threshold": 1.5})
        assert any("semantic_threshold" in e for e in validate_config(config).errors)

    def test_negative_threshold_is_rejected(self):
        config = self._config(regression_thresholds={"required_recall_drop_pp": -1})
        assert any("non-negative" in e for e in validate_config(config).errors)
