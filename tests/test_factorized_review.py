"""Factorized condition review (terra_factorized_review_v1): the closed parse
contract, the full derivation matrix, the ledger ordering, and the boundary
invariant that no production module imports it. No live provider is ever
called.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_factorized_review.py -q
"""
import json
from itertools import product
from pathlib import Path

import pytest

from tools.renovation_architecture.contracts import REVIEW_RATIONALE_MAX_CHARS
from tools.renovation_architecture.factorized_review import (
    DERIVED_CLASSES,
    FACTOR_KEYS,
    FACTOR_VALUES,
    FACTORIZED_PROMPT_VERSION,
    OBSERVED_DESCRIPTION_MAX_CHARS,
    REVIEW_KEYS,
    build_factorized_response_schema,
    derive_class,
    factorized_unit_fresh,
    parse_factorized_reviews,
)
from tools.scene_classifier_passes import PassExecutionError

CID = "oc1_0123456789abcdef"
OTHER = "oc1_fedcba9876543210"

# (visible, accurate, material) -> derived class. Every row is the design
# doc's §1.2 table; the parametrized test below proves it covers all 27
# combinations and emits nothing outside DERIVED_CLASSES.
FULL_MATRIX = [
    # visible=no is decisive: nothing absent can be misnamed or trivial.
    *[(("no", a, m), "absent")
      for a, m in product(sorted(FACTOR_VALUES), repeat=2)],
    # visible=unclear is decisive the other way.
    *[(("unclear", a, m), "inconclusive")
      for a, m in product(sorted(FACTOR_VALUES), repeat=2)],
    # visible=yes: any remaining unclear refuses to guess.
    (("yes", "unclear", "yes"), "inconclusive"),
    (("yes", "unclear", "no"), "inconclusive"),
    (("yes", "unclear", "unclear"), "inconclusive"),
    (("yes", "yes", "unclear"), "inconclusive"),
    (("yes", "no", "unclear"), "inconclusive"),
    # the four named classes
    (("yes", "yes", "yes"), "exact_and_warranted"),
    (("yes", "yes", "no"), "exact_but_trivial"),
    (("yes", "no", "yes"), "misnamed_but_warranted"),
    (("yes", "no", "no"), "misnamed_and_trivial"),
]


def _factors(visible="yes", accurate="yes", material="yes"):
    return {
        "visible": visible,
        "claim_accurate_as_written": accurate,
        "material_enough_for_work": material,
    }


def _review(condition_id=CID, *, visible="yes", accurate="yes", material="yes",
            description="", rationale="Cabinets are worn."):
    return {
        "condition_id": condition_id,
        **_factors(visible, accurate, material),
        "observed_description": description,
        "rationale": rationale,
    }


def _payload(*reviews):
    return json.dumps({"reviews": list(reviews)})


class TestDeriveClass:
    @pytest.mark.parametrize("factors,expected", FULL_MATRIX)
    def test_full_matrix(self, factors, expected):
        visible, accurate, material = factors
        assert derive_class(_factors(visible, accurate, material)) == expected

    def test_matrix_covers_every_combination_exactly_once(self):
        assert len(FULL_MATRIX) == 27
        assert len({factors for factors, _ in FULL_MATRIX}) == 27

    def test_every_outcome_is_in_the_closed_vocabulary(self):
        assert {expected for _, expected in FULL_MATRIX} <= set(DERIVED_CLASSES)

    def test_non_bounded_factor_raises(self):
        with pytest.raises(ValueError, match="non-bounded"):
            derive_class(_factors(visible="probably"))
        with pytest.raises(ValueError, match="non-bounded"):
            derive_class({"visible": "yes"})  # missing the other two


class TestLabelSchemaPin:
    """The derived class strings ARE the v1.1 label slugs, so the scorer joins
    without a translation table. label_schema is the source of truth."""

    def test_classes_are_label_slugs(self):
        from tools.label_schema import ADJUDICATION_KEYS, CLAIM_AXIS

        slugs = set(ADJUDICATION_KEYS.values())
        derived = set(DERIVED_CLASSES)
        assert derived - {"inconclusive"} <= slugs
        # The one deliberate fold: the label vocabulary separates
        # wrong_object_or_place from absent and its own claim axis maps both
        # to `absent`, which is the class the derivation emits.
        unmatched = slugs - derived
        assert unmatched == {"wrong_object_or_place"}
        assert CLAIM_AXIS["wrong_object_or_place"] == "absent"
        assert CLAIM_AXIS["absent"] == "absent"


class TestResponseSchema:
    def test_closed_at_both_levels(self):
        schema = build_factorized_response_schema([CID, OTHER])
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["reviews"]
        item = schema["properties"]["reviews"]["items"]
        assert item["additionalProperties"] is False
        assert item["required"] == list(REVIEW_KEYS)

    def test_enums_are_per_call_and_bounded(self):
        item = build_factorized_response_schema(
            [OTHER, CID]
        )["properties"]["reviews"]["items"]
        assert item["properties"]["condition_id"]["enum"] == sorted([CID, OTHER])
        for key in FACTOR_KEYS:
            assert item["properties"][key]["enum"] == sorted(FACTOR_VALUES)


class TestParseContract:
    def test_happy_path_derives_the_class(self):
        parsed = parse_factorized_reviews(
            _payload(_review(visible="yes", accurate="no", material="yes",
                             description="water staining, not scuffing")),
            condition_ids=(CID,),
        )
        assert parsed[CID]["derived_class"] == "misnamed_but_warranted"
        assert parsed[CID]["observed_description"] == "water staining, not scuffing"
        assert parsed[CID]["visible"] == "yes"

    def test_not_json(self):
        with pytest.raises(PassExecutionError) as exc:
            parse_factorized_reviews("{nope", condition_ids=(CID,))
        assert exc.value.code == "JSONDecodeError"

    @pytest.mark.parametrize("raw,match", [
        (json.dumps({"reviews": [], "extra": 1}), "exactly one 'reviews' key"),
        (json.dumps({"reviews": {}}), "must be an array"),
        (json.dumps({"reviews": ["nope"]}), "must be an object"),
    ])
    def test_envelope_violations(self, raw, match):
        with pytest.raises(PassExecutionError, match=match):
            parse_factorized_reviews(raw, condition_ids=(CID,))

    def test_extra_field_is_refused(self):
        entry = _review()
        entry["price"] = 1500
        with pytest.raises(PassExecutionError, match="outside the closed"):
            parse_factorized_reviews(_payload(entry), condition_ids=(CID,))

    def test_unknown_condition(self):
        with pytest.raises(PassExecutionError, match="unknown condition"):
            parse_factorized_reviews(
                _payload(_review(OTHER)), condition_ids=(CID,)
            )

    def test_duplicate_condition(self):
        with pytest.raises(PassExecutionError, match="duplicates"):
            parse_factorized_reviews(
                _payload(_review(), _review()), condition_ids=(CID,)
            )

    def test_missing_condition(self):
        with pytest.raises(PassExecutionError, match="no answer for 1"):
            parse_factorized_reviews(
                _payload(_review()), condition_ids=(CID, OTHER)
            )

    @pytest.mark.parametrize("key", FACTOR_KEYS)
    def test_out_of_vocabulary_factor(self, key):
        entry = _review()
        entry[key] = "probably"
        with pytest.raises(PassExecutionError, match="not a bounded factor"):
            parse_factorized_reviews(_payload(entry), condition_ids=(CID,))

    @pytest.mark.parametrize("key", ["rationale", "observed_description"])
    def test_non_string_free_text(self, key):
        entry = _review()
        entry[key] = 12
        with pytest.raises(PassExecutionError, match="must be a string"):
            parse_factorized_reviews(_payload(entry), condition_ids=(CID,))

    def test_free_text_is_truncated_not_refused(self):
        parsed = parse_factorized_reviews(
            _payload(_review(accurate="no", description="d" * 5000,
                             rationale="r" * 5000)),
            condition_ids=(CID,),
        )
        assert len(parsed[CID]["rationale"]) == REVIEW_RATIONALE_MAX_CHARS
        assert len(parsed[CID]["observed_description"]) == (
            OBSERVED_DESCRIPTION_MAX_CHARS
        )


class TestParseLeniency:
    """Strict on the bounded vocabulary, lenient on free text and on prompt
    adherence: parsing runs AFTER settle, so failing a unit over a missing
    description would burn the tokens without measuring anything."""

    def test_empty_description_on_a_misnamed_claim_is_not_a_failure(self):
        parsed = parse_factorized_reviews(
            _payload(_review(accurate="no", description="")),
            condition_ids=(CID,),
        )
        assert parsed[CID]["observed_description"] == ""
        assert parsed[CID]["derived_class"] == "misnamed_but_warranted"

    def test_non_unclear_factors_under_invisible_are_not_a_failure(self):
        # The prompt asks for `unclear` here; the derivation ignores them.
        parsed = parse_factorized_reviews(
            _payload(_review(visible="no", accurate="yes", material="yes")),
            condition_ids=(CID,),
        )
        assert parsed[CID]["derived_class"] == "absent"


class _Ledger:
    """Records ordering so the settle-before-parse invariant is observable."""

    def __init__(self):
        self.events = []
        self._next = 0

    def reserve(self, **kwargs):
        self._next += 1
        self.events.append(("reserve", kwargs["tokens"]))
        return self._next

    def settle(self, reservation_id, *, provider_total_tokens):
        self.events.append(("settle", provider_total_tokens))
        return provider_total_tokens or 25_000


class _Client:
    def __init__(self, raw, *, raises=None):
        self._raw = raw
        self._raises = raises
        self.usage_stats = {"input_tokens": 0, "cached_input_tokens": 0,
                            "output_tokens": 0, "total_tokens": 0,
                            "metered_calls": 0}

    def analyze_images_sync(self, **kwargs):
        self.calls = kwargs
        if self._raises is not None:
            raise self._raises
        self.usage_stats = {"input_tokens": 900, "cached_input_tokens": 0,
                            "output_tokens": 120, "total_tokens": 1020,
                            "metered_calls": 1}
        return self._raw


def _request(tmp_path: Path):
    from tools.renovation_architecture.terra_review import TerraUnitRequest

    image = tmp_path / "photo_001.jpg"
    image.write_bytes(b"not-read-by-this-path")
    return TerraUnitRequest(
        estimate_unit_id="kitchen_primary",
        condition_ids=(CID,),
        system_prompt="sys",
        user_prompt="user",
        photo_keys=("photo_001.jpg",),
        image_paths=(image,),
        response_schema=build_factorized_response_schema([CID]),
        request_fingerprint="f" * 64,
        request_bytes=2048,
    )


class TestFactorizedUnitFresh:
    def _run(self, tmp_path, client, ledger):
        return factorized_unit_fresh(
            request=_request(tmp_path),
            vlm_client=client,
            api_key="k",
            terra_model="gpt-5.6-terra",
            terra_max_output_tokens=8192,
            ledger=ledger,
            property_key="prop",
            source_run_id="redecide_factorized_v1",
            estimate_id="rea1_" + "0" * 16,
        )

    def test_reserve_call_settle_parse(self, tmp_path):
        ledger = _Ledger()
        client = _Client(_payload(_review(accurate="no", material="no",
                                          description="light scuffing")))
        call, reviews = self._run(tmp_path, client, ledger)
        assert [name for name, _ in ledger.events] == ["reserve", "settle"]
        assert ledger.events[1][1] == 1020  # settled down to the actual usage
        assert call["prompt_version"] == FACTORIZED_PROMPT_VERSION
        assert call["total_tokens"] == 1020
        (review,) = reviews
        assert review["derived_class"] == "misnamed_and_trivial"
        assert review["prompt_version"] == FACTORIZED_PROMPT_VERSION
        assert review["observed_description"] == "light scuffing"
        # The schema that actually went on the wire is the factorized one.
        assert "visible" in json.dumps(client.calls["response_json_schema"])

    def test_contract_violation_still_settles(self, tmp_path):
        ledger = _Ledger()
        entry = _review()
        entry["confidence"] = 0.9
        client = _Client(_payload(entry))
        with pytest.raises(PassExecutionError, match="outside the closed"):
            self._run(tmp_path, client, ledger)
        # Settled BEFORE the parse failure: the tokens were spent regardless.
        assert [name for name, _ in ledger.events] == ["reserve", "settle"]
        assert ledger.events[1][1] == 1020

    def test_provider_failure_keeps_the_conservative_reservation(self, tmp_path):
        ledger = _Ledger()
        client = _Client("", raises=RuntimeError("boom"))
        with pytest.raises(PassExecutionError):
            self._run(tmp_path, client, ledger)
        assert [name for name, _ in ledger.events] == ["reserve", "settle"]
        assert ledger.events[1][1] is None  # no settle-down on failure


class TestBoundary:
    """The module is shadow-only: nothing in the production pipeline may
    import it, and it is not re-exported from the package."""

    def test_no_production_module_imports_it(self):
        package = Path("tools/renovation_architecture")
        offenders = [
            path.name
            for path in package.glob("*.py")
            if path.name != "factorized_review.py"
            and "factorized_review" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_not_re_exported(self):
        init = (Path("tools/renovation_architecture") / "__init__.py").read_text(
            encoding="utf-8"
        )
        assert "factorized_review" not in init

    def test_production_verdict_vocabulary_is_untouched(self):
        from tools.renovation_architecture.contracts import (
            REVIEW_VERDICTS,
            TERRA_REVIEW_PROMPT_VERSION,
        )

        assert REVIEW_VERDICTS == {"supported", "unsupported", "cannot_assess"}
        assert TERRA_REVIEW_PROMPT_VERSION == "terra_condition_review_v1"
        assert FACTORIZED_PROMPT_VERSION != TERRA_REVIEW_PROMPT_VERSION
