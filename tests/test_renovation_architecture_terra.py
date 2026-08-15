"""Terra condition-review tests: request shape, bounded verdicts, typed
failures, and per-unit checkpoints — over synthetic v3.1 catalogs, tiny real
PNGs, and the conftest FakeOpenAI provider. No live provider is ever called.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_terra.py -q
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import FakeOpenAI, openai_response, tiny_png
from tests.test_renovation_architecture_catalog import _v31_catalog, _v31_item
from tools.failure_taxonomy import classify_failure
from tools.renovation_architecture.checkpoints import terra_checkpoint_dir
from tools.renovation_architecture.conditions import build_observed_conditions
from tools.renovation_architecture.contracts import (
    REVIEW_RATIONALE_MAX_CHARS,
    SHADOW_DEBUG_KEY,
)
from tools.renovation_architecture.ids import make_estimate_id
from tools.renovation_architecture.review_pipeline import run_condition_review
from tools.renovation_architecture.runtime import (
    build_shadow_envelope,
    get_runtime,
    initialize_renovation_architecture,
    reset_runtime_for_tests,
)
from tools.renovation_architecture.validators import validate_envelope
from tools.scene_classifier_passes import PassExecutionError
from tools.vlm_client import VLMClient

# gpt-5* so reasoning effort reaches the wire; the fake never checks the name.
TERRA_MODEL = "gpt-5.4-terra-test"
MAX_OUTPUT_TOKENS = 512
PROPERTY_KEY = "prop"
RUN_ID = "run_1"
CREATED_AT = "2026-08-14T00:00:00Z"


@pytest.fixture(autouse=True)
def _clean_runtime():
    reset_runtime_for_tests()
    yield
    reset_runtime_for_tests()


# ── harness (shared with the disposition/usage-guard test files) ─────────────

def _init_runtime(tmp_path, *items, terra_model=TERRA_MODEL):
    catalog = _v31_catalog(*items)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    initialize_renovation_architecture(
        mode="shadow",
        catalog=catalog,
        catalog_path=path,
        kind_ontology_version="observation_kind_v2",
        terra_model=terra_model,
        terra_max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return get_runtime()


def _photo_files(tmp_path, specs):
    """specs: {photo_key: (scene, (r, g, b))} -> (photos, photo_key_to_path)."""
    photos, paths = {}, {}
    for index, (key, (scene, color)) in enumerate(sorted(specs.items()), start=1):
        file_path = tmp_path / key
        file_path.write_bytes(tiny_png(*color))
        photos[key] = {"photo": {"index": index}, "scene": {"id": scene}}
        paths[key] = file_path
    return photos, paths


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


def _estimate_id(runtime, *, source_run_id=RUN_ID):
    return make_estimate_id(
        property_key=PROPERTY_KEY,
        source_run_id=source_run_id,
        catalog_sha256=runtime.catalog_sha256,
        projection_fingerprint=runtime.projection_fingerprint,
    )


def _conditions(runtime, issues, photos, *, source_run_id=RUN_ID):
    """Discover the ObservedConditions (and their ids) before crafting the
    fake provider responses."""
    return [
        draft.condition
        for draft in build_observed_conditions(
            issues_flat=issues,
            photos=photos,
            property_metadata=None,
            projection=runtime.projection,
            estimate_id=_estimate_id(runtime, source_run_id=source_run_id),
        )
    ]


def _review_response(verdicts, *, usage=(1000, 0, 200), mutate=None):
    """A fake Responses payload reviewing {condition_id: verdict}. usage is
    (input, cached_input, output) or None for a provider that reports none."""
    reviews = [
        {"condition_id": cid, "verdict": verdict, "rationale": "as seen"}
        for cid, verdict in sorted(verdicts.items())
    ]
    if mutate is not None:
        mutate(reviews)
    response = openai_response(text=json.dumps({"reviews": reviews}))
    if usage is not None:
        input_tokens, cached, output_tokens = usage
        response.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
    return response


def _run_review(tmp_path, runtime, issues, photos, paths, fake, *,
                source_run_id=RUN_ID, property_metadata=None):
    client = VLMClient()
    fake.attach(client)
    result = run_condition_review(
        runtime=runtime,
        estimate_id=_estimate_id(runtime, source_run_id=source_run_id),
        property_key=PROPERTY_KEY,
        source_run_id=source_run_id,
        created_at=CREATED_AT,
        issues_flat=issues,
        photos=photos,
        property_metadata=property_metadata,
        photo_key_to_path=paths,
        vlm_client=client,
        api_key="test-key",
        artifacts_root=tmp_path / "artifacts",
    )
    return result, client


def _kitchen_setup(tmp_path, *items, photo_colors=((200, 30, 30),)):
    """One kitchen with N distinct photos and one issue per item per photo."""
    runtime = _init_runtime(tmp_path, *items)
    specs = {
        f"kitchen_{index}.png": ("kitchen", color)
        for index, color in enumerate(photo_colors, start=1)
    }
    photos, paths = _photo_files(tmp_path, specs)
    issues = [
        _issue(item["id"], photo_key)
        for item in items
        for photo_key in sorted(specs)
    ]
    return runtime, issues, photos, paths


# ── request shape ────────────────────────────────────────────────────────────

class TestTerraRequestShape:
    def test_wire_shape_is_strict_and_bounded(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter"),
            photo_colors=((200, 30, 30), (10, 200, 240)),
        )
        (condition,) = _conditions(runtime, issues, photos)
        fake = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"})
        ])
        _run_review(tmp_path, runtime, issues, photos, paths, fake)

        request = fake.last_request
        assert request["model"] == TERRA_MODEL
        assert request["max_output_tokens"] == MAX_OUTPUT_TOKENS
        assert request["reasoning"] == {"effort": "medium"}
        fmt = request["text"]["format"]
        assert fmt["strict"] is True
        assert fmt["name"] == "terra_condition_review"
        item_schema = fmt["schema"]["properties"]["reviews"]["items"]
        assert item_schema["additionalProperties"] is False
        assert item_schema["properties"]["condition_id"]["enum"] == [
            condition.condition_id
        ]
        assert sorted(item_schema["properties"]["verdict"]["enum"]) == [
            "cannot_assess", "supported", "unsupported"
        ]
        content = request["input"][1]["content"]
        images = [entry for entry in content if entry["type"] == "input_image"]
        assert len(images) == 2  # both distinct views, each exactly once
        text = content[0]["text"]
        assert condition.condition_id in text
        assert "worn_counter" in text

    def test_duplicate_photos_are_sent_once(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter"),
            # two byte-identical photos + one distinct view
            photo_colors=((200, 30, 30), (200, 30, 30), (10, 200, 240)),
        )
        (condition,) = _conditions(runtime, issues, photos)
        fake = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"})
        ])
        result, _ = _run_review(tmp_path, runtime, issues, photos, paths, fake)
        content = fake.last_request["input"][1]["content"]
        images = [entry for entry in content if entry["type"] == "input_image"]
        assert len(images) == 2  # one representative per view class
        (evidence,) = result["evidence_facts"]
        assert evidence["distinct_photo_count"] == 3
        assert evidence["distinct_view_count"] == 2

    def test_units_are_batched_into_separate_calls(self, tmp_path):
        runtime = _init_runtime(
            tmp_path,
            _v31_item("worn_counter"),
            _v31_item("worn_siding", scene_groups=["exterior"]),
        )
        photos, paths = _photo_files(tmp_path, {
            "kitchen_1.png": ("kitchen", (200, 30, 30)),
            "front_1.png": ("exterior", (10, 200, 240)),
        })
        issues = [
            _issue("worn_counter", "kitchen_1.png"),
            _issue("worn_siding", "front_1.png", scene_group="exterior"),
        ]
        conditions = _conditions(runtime, issues, photos)
        assert len(conditions) == 2
        assert len({c.estimate_unit_id for c in conditions}) == 2
        # Units run sorted by estimate_unit_id; queue responses accordingly.
        ordered = sorted(conditions, key=lambda c: c.estimate_unit_id)
        fake = FakeOpenAI(responses=[
            _review_response({ordered[0].condition_id: "supported"}),
            _review_response({ordered[1].condition_id: "unsupported"}),
        ])
        result, _ = _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert len(fake.requests) == 2
        verdicts = {
            review["condition_id"]: review["verdict"]
            for review in result["condition_reviews"]
        }
        assert verdicts == {
            ordered[0].condition_id: "supported",
            ordered[1].condition_id: "unsupported",
        }


# ── verdicts and the layer boundary ──────────────────────────────────────────

class TestVerdictsAndBoundary:
    def test_all_three_verdicts_flow_to_dispositions(self, tmp_path):
        items = [
            _v31_item("worn_counter"),
            _v31_item("worn_cabinets"),
            _v31_item("worn_backsplash"),
        ]
        runtime, issues, photos, paths = _kitchen_setup(tmp_path, *items)
        conditions = _conditions(runtime, issues, photos)
        verdict_by_cid = dict(zip(
            [c.condition_id for c in conditions],
            ["supported", "unsupported", "cannot_assess"],
        ))
        fake = FakeOpenAI(responses=[_review_response(verdict_by_cid)])
        result, _ = _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert len(fake.requests) == 1  # one unit -> one call for 3 conditions
        dispositions = {
            record["condition_id"]: record["disposition"]
            for record in result["condition_dispositions"]
        }
        expected = {
            "supported": "accepted_for_work",  # work route
            "unsupported": "excluded",
            "cannot_assess": "inspection",
        }
        for cid, verdict in verdict_by_cid.items():
            assert dispositions[cid] == expected[verdict]

    def test_review_carrying_price_fields_is_rejected(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)

        def add_price(reviews):
            reviews[0]["price"] = 1500

        fake = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"},
                             mutate=add_price)
        ])
        with pytest.raises(PassExecutionError) as excinfo:
            _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert excinfo.value.code == "TerraReviewContract"
        assert "closed review contract" in excinfo.value.message
        assert classify_failure(excinfo.value).category == "parse"

    def test_rationale_is_bounded(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)

        def long_rationale(reviews):
            reviews[0]["rationale"] = "x" * 5000

        fake = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"},
                             mutate=long_rationale)
        ])
        result, _ = _run_review(tmp_path, runtime, issues, photos, paths, fake)
        (review,) = result["condition_reviews"]
        assert len(review["rationale"]) == REVIEW_RATIONALE_MAX_CHARS


# ── operational failures stay operational ────────────────────────────────────

def _failing_run(tmp_path, fake):
    runtime, issues, photos, paths = _kitchen_setup(
        tmp_path, _v31_item("worn_counter")
    )
    with pytest.raises(PassExecutionError) as excinfo:
        _run_review(tmp_path, runtime, issues, photos, paths, fake)
    return excinfo.value


class TestOperationalFailures:
    def test_non_json_response_is_a_parse_failure(self, tmp_path):
        failure = _failing_run(
            tmp_path, FakeOpenAI(responses=[openai_response(text="not json")])
        )
        assert failure.stage == "parse"
        assert failure.code == "JSONDecodeError"
        assert classify_failure(failure).category == "parse"

    @pytest.mark.parametrize("shape", ["missing", "duplicate", "unknown", "verdict"])
    def test_contract_violations_are_parse_failures(self, tmp_path, shape):
        def mutate(reviews):
            if shape == "missing":
                reviews.clear()
            elif shape == "duplicate":
                reviews.append(dict(reviews[0]))
            elif shape == "unknown":
                reviews[0]["condition_id"] = "oc1_" + "f" * 16
            elif shape == "verdict":
                reviews[0]["verdict"] = "probably"

        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        fake = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"}, mutate=mutate)
        ])
        with pytest.raises(PassExecutionError) as excinfo:
            _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert excinfo.value.code == "TerraReviewContract"
        assert classify_failure(excinfo.value).category == "parse"

    def test_provider_exception_is_a_request_failure(self, tmp_path):
        failure = _failing_run(
            tmp_path, FakeOpenAI(raises=RuntimeError("provider melted"))
        )
        assert failure.stage == "request"
        assert classify_failure(failure).category == "provider"

    def test_incomplete_response_is_operational(self, tmp_path):
        failure = _failing_run(
            tmp_path,
            FakeOpenAI(responses=[openai_response(
                status="incomplete", incomplete_reason="max_output_tokens"
            )]),
        )
        assert classify_failure(failure).category == "malformed_response"

    def test_failure_becomes_a_failed_envelope_never_cannot_assess(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        client = VLMClient()
        FakeOpenAI(raises=RuntimeError("provider melted")).attach(client)
        envelope = build_shadow_envelope(
            property_key=PROPERTY_KEY,
            run_id=RUN_ID,
            created_at=CREATED_AT,
            source_artifact="prop/run_1/photo_intel.json",
            issues_flat=issues,
            photos=photos,
            photo_key_to_path=paths,
            vlm_client=client,
            api_key="test-key",
            artifacts_root=tmp_path / "artifacts",
        )
        res = validate_envelope(envelope)
        assert res.ok, res.errors
        assert envelope["state"] == "failed"
        assert envelope["reason"] == "provider"
        assert "provider melted" in envelope["error_detail"]
        assert envelope["result"] is None
        assert "cannot_assess" not in json.dumps(envelope)


# ── checkpoints ──────────────────────────────────────────────────────────────

def _checkpoint_files(tmp_path, *, run_id=RUN_ID):
    directory = terra_checkpoint_dir(
        tmp_path / "artifacts", PROPERTY_KEY, run_id
    )
    return sorted(directory.glob("terra_unit_*.json"))


class TestCheckpoints:
    def test_same_fingerprint_reuses_without_a_provider_call(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        fake = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"},
                             usage=(1000, 250, 200))
        ])
        first, _ = _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert len(_checkpoint_files(tmp_path)) == 1
        assert first["terra_calls"][0]["usage_source"] == "provider"
        assert first["terra_calls"][0]["budget_debited_tokens"] == 1200

        strict = FakeOpenAI(raises=AssertionError("provider must not be called"))
        second, _ = _run_review(tmp_path, runtime, issues, photos, paths, strict)
        assert strict.requests == []
        (call,) = second["terra_calls"]
        assert call["usage_source"] == "checkpoint"
        assert call["budget_debited_tokens"] == 0
        # Original provider token numbers are preserved as provenance.
        assert call["input_tokens"] == 1000
        assert call["cached_input_tokens"] == 250
        assert call["total_tokens"] == 1200
        assert second["condition_reviews"] == first["condition_reviews"]

    def test_model_change_invalidates_the_fingerprint(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        response = _review_response({condition.condition_id: "supported"})
        _run_review(tmp_path, runtime, issues, photos, paths,
                    FakeOpenAI(responses=[response]))

        reset_runtime_for_tests()
        runtime = _init_runtime(
            tmp_path, _v31_item("worn_counter"), terra_model="gpt-5.4-other"
        )
        fresh = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"})
        ])
        _run_review(tmp_path, runtime, issues, photos, paths, fresh)
        assert len(fresh.requests) == 1  # checkpoint ignored, unit re-bought

    def test_image_change_invalidates_the_fingerprint(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        _run_review(tmp_path, runtime, issues, photos, paths, FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"})
        ]))
        # Same key, different pixels -> different evidence identity.
        paths["kitchen_1.png"].write_bytes(tiny_png(90, 90, 90))
        fresh = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"})
        ])
        _run_review(tmp_path, runtime, issues, photos, paths, fresh)
        assert len(fresh.requests) == 1

    def test_corrupt_checkpoint_is_ignored(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        response = _review_response({condition.condition_id: "supported"})
        _run_review(tmp_path, runtime, issues, photos, paths,
                    FakeOpenAI(responses=[response]))
        (checkpoint,) = _checkpoint_files(tmp_path)
        checkpoint.write_text("not json", encoding="utf-8")
        fresh = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"})
        ])
        result, _ = _run_review(tmp_path, runtime, issues, photos, paths, fresh)
        assert len(fresh.requests) == 1
        assert result["terra_calls"][0]["usage_source"] == "provider"

    def test_mid_run_failure_keeps_finished_units_and_publishes_nothing(self, tmp_path):
        runtime = _init_runtime(
            tmp_path,
            _v31_item("worn_counter"),
            _v31_item("worn_siding", scene_groups=["exterior"]),
        )
        photos, paths = _photo_files(tmp_path, {
            "kitchen_1.png": ("kitchen", (200, 30, 30)),
            "front_1.png": ("exterior", (10, 200, 240)),
        })
        issues = [
            _issue("worn_counter", "kitchen_1.png"),
            _issue("worn_siding", "front_1.png", scene_group="exterior"),
        ]
        ordered = sorted(
            _conditions(runtime, issues, photos),
            key=lambda c: c.estimate_unit_id,
        )
        # Unit 1 succeeds; unit 2 drains the queue and gets the default
        # {"ok": true} payload, which violates the review contract.
        fake = FakeOpenAI(responses=[
            _review_response({ordered[0].condition_id: "supported"})
        ])
        with pytest.raises(PassExecutionError):
            _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert len(_checkpoint_files(tmp_path)) == 1  # unit 1 survived

        retry = FakeOpenAI(responses=[
            _review_response({ordered[1].condition_id: "supported"})
        ])
        result, _ = _run_review(tmp_path, runtime, issues, photos, paths, retry)
        assert len(retry.requests) == 1  # only the unfinished unit was bought
        sources = {
            call["estimate_unit_id"]: call["usage_source"]
            for call in result["terra_calls"]
        }
        assert sources[ordered[0].estimate_unit_id] == "checkpoint"
        assert sources[ordered[1].estimate_unit_id] == "provider"

    def test_checkpoints_are_scoped_to_the_source_run_id(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        _run_review(tmp_path, runtime, issues, photos, paths, FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"})
        ]))
        # A different run id resolves different condition ids (estimate
        # identity includes the run) and its own checkpoint directory.
        (other_condition,) = _conditions(
            runtime, issues, photos, source_run_id="run_2"
        )
        fresh = FakeOpenAI(responses=[
            _review_response({other_condition.condition_id: "supported"})
        ])
        _run_review(tmp_path, runtime, issues, photos, paths, fresh,
                    source_run_id="run_2")
        assert len(fresh.requests) == 1
        assert len(_checkpoint_files(tmp_path, run_id="run_2")) == 1

    def test_zero_conditions_completes_without_any_call(self, tmp_path):
        runtime = _init_runtime(tmp_path, _v31_item("worn_counter"))
        strict = FakeOpenAI(raises=AssertionError("provider must not be called"))
        result, _ = _run_review(tmp_path, runtime, [], {}, {}, strict)
        assert result["observed_conditions"] == []
        assert result["terra_calls"] == []
        assert result["terra_listing_usage"]["call_count"] == 0
        assert strict.requests == []
        assert not (tmp_path / "artifacts").exists()  # no ledger, no checkpoints


# ── end-to-end envelope ──────────────────────────────────────────────────────

class TestShadowEnvelopeEndToEnd:
    def test_complete_envelope_validates(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        client = VLMClient()
        FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"})
        ]).attach(client)
        envelope = build_shadow_envelope(
            property_key=PROPERTY_KEY,
            run_id=RUN_ID,
            created_at=CREATED_AT,
            source_artifact="prop/run_1/photo_intel.json",
            issues_flat=issues,
            photos=photos,
            photo_key_to_path=paths,
            vlm_client=client,
            api_key="test-key",
            artifacts_root=tmp_path / "artifacts",
        )
        res = validate_envelope(envelope)
        assert res.ok, res.errors
        assert envelope["state"] == "standalone_estimate_complete"
        assert envelope["reason"] is None
        assert SHADOW_DEBUG_KEY  # the key constant is what the seam writes
        (review,) = envelope["result"]["condition_reviews"]
        assert review["verdict"] == "supported"
        assert review["model"] == TERRA_MODEL
        # The accepted condition became exactly one active work item, priced
        # by the legacy heuristic core: severity-2 repair base (300, 1500)
        # x flooring trade multiplier 0.9 -> 270/1350 (neutral property
        # factor: the kitchen fixture has no ppsf/sqft metadata).
        (work_item,) = envelope["result"]["work_items"]
        assert work_item["status"] == "active"
        assert work_item["condition_ids"] == [condition.condition_id]
        assert work_item["action_code"] == "FLOORING_REPAIR"
        assert (work_item["low"], work_item["high"]) == (270, 1350)
        assert envelope["result"]["work_dedup_collisions"] == []
        standalone = envelope["result"]["standalone_estimate"]
        assert standalone["headline"] == {"low": 270, "high": 1350}
        assert standalone["totals_by_estimate_scope"]["marketability_rehab"] == {
            "low": 270, "high": 1350,
        }
