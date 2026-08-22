"""Bounded Sol package-review tests (Session 4): text-only minimized
requests, exact decision coverage, closed fields, combine/split constraints,
checkpoint reuse, telemetry reconciliation, and provider-failure semantics —
over the conftest FakeOpenAI provider. No live provider is ever called.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_sol.py -q
"""
import json
from types import SimpleNamespace

import pytest

from tests.conftest import FakeOpenAI, openai_response
from tests.test_renovation_architecture_contracts import (
    EST_ID,
    _candidate,
    _standalone_result,
)
from tools.failure_taxonomy import classify_failure
from tools.renovation_architecture.checkpoints import (
    sol_checkpoint_path,
    terra_checkpoint_dir,
)
from tools.renovation_architecture.contracts import (
    SOL_REVIEW_PROMPT_VERSION,
)
from tools.renovation_architecture.sol_review import run_package_review
from tools.renovation_architecture.validators import (
    validate_package_review_result,
)
from tools.scene_classifier_passes import PassExecutionError
from tools.vlm_client import VLMClient

# gpt-5* so reasoning effort reaches the wire; the fake never checks the name.
SOL_MODEL = "gpt-5.4-sol-test"
MAX_OUTPUT_TOKENS = 512
PROPERTY_KEY = "prop"
RUN_ID = "run_1"
CREATED_AT = "2026-08-15T00:00:00Z"


def _runtime(sol_model=SOL_MODEL):
    """run_package_review touches only these three runtime fields."""
    return SimpleNamespace(
        projection_fingerprint="b" * 64,
        sol_model=sol_model,
        sol_max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def _fixture():
    """The Session 3 result plus two single-child room candidates."""
    base = _standalone_result()
    by_catalog = {
        item["catalog_item_ids"][0]: item for item in base["work_items"]
    }
    kitchen = _candidate(
        "kitchen_modernization", "kitchen_primary",
        [by_catalog["outdated_or_damaged_cabinets"]],
        package_category="modernization",
    )
    bathroom = _candidate(
        "bathroom_modernization", "bathroom_1",
        [by_catalog["bathroom_vanity_worn"]],
        package_category="modernization",
    )
    candidates = sorted(
        [kitchen, bathroom], key=lambda c: c["package_candidate_id"]
    )
    return base, candidates


def _sol_response(decision_map, *, usage=(900, 0, 120), mutate=None):
    """A fake Responses payload deciding {candidate_id: decision or full
    entry}. usage is (input, cached_input, output) or None."""
    decisions = []
    for candidate_id, decision in sorted(decision_map.items()):
        entry = {
            "package_candidate_id": candidate_id,
            "decision": decision if isinstance(decision, str) else "approve",
            "combine_with": [],
            "split_groups": [],
            "rationale": "coherent",
        }
        if isinstance(decision, dict):
            entry.update(decision)
        decisions.append(entry)
    if mutate is not None:
        mutate(decisions)
    response = openai_response(text=json.dumps({"decisions": decisions}))
    if usage is not None:
        input_tokens, cached, output_tokens = usage
        response.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
    return response


def _run(tmp_path, base, candidates, fake, *, runtime=None):
    client = VLMClient()
    fake.attach(client)
    result = run_package_review(
        runtime=runtime or _runtime(),
        estimate_id=EST_ID,
        property_key=PROPERTY_KEY,
        source_run_id=RUN_ID,
        created_at=CREATED_AT,
        standalone_result=base,
        package_candidates=candidates,
        vlm_client=client,
        api_key="test-key",
        artifacts_root=tmp_path / "artifacts",
    )
    return result, client


def _approve_all(candidates):
    return {c["package_candidate_id"]: "approve" for c in candidates}


# ── happy path and request shape ─────────────────────────────────────────────

class TestSolRequestAndResult:
    def test_happy_path_validates_and_reconciles(self, tmp_path):
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[_sol_response(_approve_all(candidates))])
        result, _ = _run(tmp_path, base, candidates, fake)
        res = validate_package_review_result(result, estimate_id=EST_ID)
        assert res.ok, res.errors
        assert [d["decision"] for d in result["package_decisions"]] == [
            "approve", "approve",
        ]
        (call,) = result["sol_calls"]
        assert call["usage_source"] == "provider"
        assert call["model"] == SOL_MODEL
        assert call["prompt_version"] == SOL_REVIEW_PROMPT_VERSION
        assert (call["input_tokens"], call["output_tokens"]) == (900, 120)
        assert call["total_tokens"] == 1020
        assert result["sol_listing_usage"]["call_count"] == 1
        assert result["sol_listing_usage"]["total_tokens"] == 1020
        for decision in result["package_decisions"]:
            assert decision["sol_call_id"] == call["call_id"]
            assert decision["request_fingerprint"] == call["request_fingerprint"]
        # The result is a superset of the untouched Session 3 result.
        for key in ("work_items", "standalone_estimate", "condition_reviews"):
            assert result[key] == base[key]

    def test_request_is_text_only_minimized_and_layer_separated(self, tmp_path):
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[_sol_response(_approve_all(candidates))])
        _run(tmp_path, base, candidates, fake)
        request = fake.last_request
        assert request["model"] == SOL_MODEL
        assert request["max_output_tokens"] == MAX_OUTPUT_TOKENS
        assert request["reasoning"] == {"effort": "medium"}
        fmt = request["text"]["format"]
        assert fmt["strict"] is True
        assert fmt["name"] == "sol_package_review"
        item_schema = fmt["schema"]["properties"]["decisions"]["items"]
        assert item_schema["additionalProperties"] is False
        assert item_schema["properties"]["package_candidate_id"]["enum"] == [
            c["package_candidate_id"] for c in candidates
        ]
        assert sorted(item_schema["properties"]["decision"]["enum"]) == [
            "approve", "reject", "uncertain",
        ]
        # Text-only: no image parts anywhere in the request.
        assert "input_image" not in str(request)
        # Separately labeled layers and immutable child snapshots; no photos,
        # no catalog dump.
        text = str(request["input"])
        assert "objective_evidence" in text
        assert "terra_verdict" in text
        assert "deterministic_disposition" in text
        assert "child_work_snapshots" in text
        assert "CABINETS_REPLACE" in text
        for candidate in candidates:
            assert candidate["package_candidate_id"] in text

    def test_zero_candidates_completes_without_any_call(self, tmp_path):
        base, _ = _fixture()
        result = run_package_review(
            runtime=_runtime(),
            estimate_id=EST_ID,
            property_key=PROPERTY_KEY,
            source_run_id=RUN_ID,
            created_at=CREATED_AT,
            standalone_result=base,
            package_candidates=[],
            vlm_client=None,
            api_key="",
            artifacts_root=None,
        )
        res = validate_package_review_result(result, estimate_id=EST_ID)
        assert res.ok, res.errors
        assert result["package_candidates"] == []
        assert result["package_decisions"] == []
        assert result["sol_calls"] == []
        assert result["sol_listing_usage"]["call_count"] == 0

    def test_missing_seam_inputs_fail_closed_when_candidates_exist(self, tmp_path):
        base, candidates = _fixture()
        with pytest.raises(PassExecutionError) as excinfo:
            run_package_review(
                runtime=_runtime(),
                estimate_id=EST_ID,
                property_key=PROPERTY_KEY,
                source_run_id=RUN_ID,
                created_at=CREATED_AT,
                standalone_result=base,
                package_candidates=candidates,
                vlm_client=None,
                api_key="",
                artifacts_root=None,
            )
        assert excinfo.value.code == "MissingSeamInput"

    def test_uncertain_is_a_bounded_decision_not_a_failure(self, tmp_path):
        base, candidates = _fixture()
        decision_map = _approve_all(candidates)
        decision_map[candidates[0]["package_candidate_id"]] = "uncertain"
        fake = FakeOpenAI(responses=[_sol_response(decision_map)])
        result, _ = _run(tmp_path, base, candidates, fake)
        decisions = {
            d["package_candidate_id"]: d["decision"]
            for d in result["package_decisions"]
        }
        assert decisions[candidates[0]["package_candidate_id"]] == "uncertain"

    def test_provider_without_usage_records_zero_tokens(self, tmp_path):
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[
            _sol_response(_approve_all(candidates), usage=None)
        ])
        result, _ = _run(tmp_path, base, candidates, fake)
        (call,) = result["sol_calls"]
        assert call["usage_source"] == "provider"
        assert call["total_tokens"] == 0
        assert result["sol_listing_usage"]["total_tokens"] == 0


# ── operational failures stay operational ────────────────────────────────────

class TestOperationalFailures:
    def test_provider_exception_is_a_request_failure(self, tmp_path):
        base, candidates = _fixture()
        fake = FakeOpenAI(raises=RuntimeError("provider melted"))
        with pytest.raises(PassExecutionError) as excinfo:
            _run(tmp_path, base, candidates, fake)
        assert excinfo.value.stage == "request"
        assert classify_failure(excinfo.value).category == "provider"

    def test_non_json_response_is_a_parse_failure(self, tmp_path):
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[openai_response(text="not json")])
        with pytest.raises(PassExecutionError) as excinfo:
            _run(tmp_path, base, candidates, fake)
        assert excinfo.value.stage == "parse"
        assert excinfo.value.code == "JSONDecodeError"

    @pytest.mark.parametrize("shape", ["missing", "duplicate", "unknown", "verdict"])
    def test_coverage_violations_are_contract_failures(self, tmp_path, shape):
        def mutate(decisions):
            if shape == "missing":
                decisions.pop()
            elif shape == "duplicate":
                decisions.append(dict(decisions[0]))
            elif shape == "unknown":
                decisions[0]["package_candidate_id"] = "pk1_" + "f" * 16
            elif shape == "verdict":
                decisions[0]["decision"] = "maybe"

        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[
            _sol_response(_approve_all(candidates), mutate=mutate)
        ])
        with pytest.raises(PassExecutionError) as excinfo:
            _run(tmp_path, base, candidates, fake)
        assert excinfo.value.code == "SolReviewContract"
        assert classify_failure(excinfo.value).category == "parse"

    def test_closed_fields_reject_extra_content(self, tmp_path):
        def mutate(decisions):
            decisions[0]["price"] = 12000

        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[
            _sol_response(_approve_all(candidates), mutate=mutate)
        ])
        with pytest.raises(PassExecutionError) as excinfo:
            _run(tmp_path, base, candidates, fake)
        assert excinfo.value.code == "SolReviewContract"
        assert "closed decision contract" in excinfo.value.message


# ── combine/split constraints ────────────────────────────────────────────────

class TestCombineSplit:
    def _run_with(self, tmp_path, decision_map, *, fixture=None):
        base, candidates = fixture or _fixture()
        fake = FakeOpenAI(responses=[_sol_response(decision_map)])
        return _run(tmp_path, base, candidates, fake)

    def test_combine_edges_join_approved_candidates(self, tmp_path):
        base, candidates = _fixture()
        first, second = candidates
        decision_map = {
            first["package_candidate_id"]: {
                "decision": "approve",
                "combine_with": [second["package_candidate_id"]],
            },
            second["package_candidate_id"]: "approve",
        }
        result, _ = self._run_with(
            tmp_path, decision_map, fixture=(base, candidates)
        )
        decisions = {
            d["package_candidate_id"]: d for d in result["package_decisions"]
        }
        assert decisions[first["package_candidate_id"]]["combine_with"] == [
            second["package_candidate_id"]
        ]

    @pytest.mark.parametrize("bad", ["self", "unknown", "rejected_target", "uncertain_source"])
    def test_invalid_combine_edges_are_contract_failures(self, tmp_path, bad):
        base, candidates = _fixture()
        first, second = candidates
        first_id = first["package_candidate_id"]
        second_id = second["package_candidate_id"]
        decision_map = {first_id: "approve", second_id: "approve"}
        if bad == "self":
            decision_map[first_id] = {
                "decision": "approve", "combine_with": [first_id],
            }
        elif bad == "unknown":
            decision_map[first_id] = {
                "decision": "approve", "combine_with": ["pk1_" + "f" * 16],
            }
        elif bad == "rejected_target":
            decision_map[first_id] = {
                "decision": "approve", "combine_with": [second_id],
            }
            decision_map[second_id] = "reject"
        elif bad == "uncertain_source":
            decision_map[first_id] = {
                "decision": "uncertain", "combine_with": [second_id],
            }
        with pytest.raises(PassExecutionError) as excinfo:
            self._run_with(tmp_path, decision_map, fixture=(base, candidates))
        assert excinfo.value.code == "SolReviewContract"

    def _split_fixture(self):
        """One candidate holding BOTH work items, so a legal two-group split
        exists, plus the bathroom candidate sharing a child."""
        base = _standalone_result()
        both = _candidate(
            "kitchen_modernization", "kitchen_primary", base["work_items"],
            package_category="modernization",
        )
        other = _candidate(
            "bathroom_modernization", "bathroom_1", [base["work_items"][0]],
            package_category="modernization",
        )
        candidates = sorted(
            [both, other], key=lambda c: c["package_candidate_id"]
        )
        return base, candidates, both, other

    def test_split_partitions_and_is_normalized(self, tmp_path):
        base, candidates, both, other = self._split_fixture()
        w1, w2 = both["child_work_item_ids"]
        decision_map = {
            both["package_candidate_id"]: {
                "decision": "approve",
                # Deliberately unnormalized group order.
                "split_groups": [[w2], [w1]],
            },
            other["package_candidate_id"]: "approve",
        }
        fake = FakeOpenAI(responses=[_sol_response(decision_map)])
        result, _ = _run(tmp_path, base, candidates, fake)
        decision = next(
            d for d in result["package_decisions"]
            if d["package_candidate_id"] == both["package_candidate_id"]
        )
        # Ordering normalized, membership untouched.
        assert decision["split_groups"] == sorted([[w1], [w2]], key=tuple)

    @pytest.mark.parametrize("bad", ["one_group", "wrong_partition", "overlap"])
    def test_invalid_splits_are_contract_failures(self, tmp_path, bad):
        base, candidates, both, other = self._split_fixture()
        w1, w2 = both["child_work_item_ids"]
        groups = {
            "one_group": [[w1, w2]],
            "wrong_partition": [[w1], ["wk1_" + "f" * 16]],
            "overlap": [[w1], [w1, w2]],
        }[bad]
        decision_map = {
            both["package_candidate_id"]: {
                "decision": "approve", "split_groups": groups,
            },
            other["package_candidate_id"]: "approve",
        }
        fake = FakeOpenAI(responses=[_sol_response(decision_map)])
        with pytest.raises(PassExecutionError) as excinfo:
            _run(tmp_path, base, candidates, fake)
        assert excinfo.value.code == "SolReviewContract"

    def test_a_candidate_cannot_combine_and_split(self, tmp_path):
        base, candidates, both, other = self._split_fixture()
        w1, w2 = both["child_work_item_ids"]
        decision_map = {
            # `both` splits AND is cited as a combine target by `other`.
            both["package_candidate_id"]: {
                "decision": "approve", "split_groups": [[w1], [w2]],
            },
            other["package_candidate_id"]: {
                "decision": "approve",
                "combine_with": [both["package_candidate_id"]],
            },
        }
        fake = FakeOpenAI(responses=[_sol_response(decision_map)])
        with pytest.raises(PassExecutionError) as excinfo:
            _run(tmp_path, base, candidates, fake)
        assert excinfo.value.code == "SolReviewContract"
        assert "both combine and split" in excinfo.value.message


# ── display-only aggregate at the Sol boundary ───────────────────────────────

def _whole_home_fixture():
    """Two room turnover candidates plus the display-only aggregate."""
    base = _standalone_result()
    by_catalog = {
        item["catalog_item_ids"][0]: item for item in base["work_items"]
    }
    bedroom = _candidate(
        "bedroom_turnover", "bedroom_1",
        [by_catalog["outdated_or_damaged_cabinets"]],
        package_category="turnover",
    )
    living = _candidate(
        "living_turnover", "living_1",
        [by_catalog["bathroom_vanity_worn"]],
        package_category="turnover",
    )
    rooms = sorted([bedroom, living], key=lambda c: c["package_candidate_id"])
    aggregate = _candidate(
        "interior_paint_flooring_refresh", "whole_home", [],
        package_category="turnover",
        package_level="property",
        room="whole_home",
        child_work_item_ids=[],
        driver_work_item_ids=[],
        support_work_item_ids=[],
        pricing_profile="interior_paint_flooring_refresh",
        pricing_tier="property_turnover_aggregate",
        proposed_treatment="whole_home_turnover_aggregate",
        display_only=True,
        contributing_candidate_ids=sorted(
            room["package_candidate_id"] for room in rooms
        ),
        unfloored_low=sum(room["low"] for room in rooms),
        unfloored_high=sum(room["high"] for room in rooms),
        low=sum(room["low"] for room in rooms),
        high=sum(room["high"] for room in rooms),
        cost_floor_applied=False,
    )
    candidates = sorted(
        rooms + [aggregate], key=lambda c: c["package_candidate_id"]
    )
    return base, candidates, aggregate


class TestDisplayOnlyBoundary:
    def test_aggregate_is_reviewable_but_never_combines_or_splits(self, tmp_path):
        base, candidates, aggregate = _whole_home_fixture()
        decision_map = _approve_all(candidates)
        decision_map[aggregate["package_candidate_id"]] = "reject"
        fake = FakeOpenAI(responses=[_sol_response(decision_map)])
        result, _ = _run(tmp_path, base, candidates, fake)
        res = validate_package_review_result(result, estimate_id=EST_ID)
        assert res.ok, res.errors
        decisions = {
            d["package_candidate_id"]: d["decision"]
            for d in result["package_decisions"]
        }
        assert decisions[aggregate["package_candidate_id"]] == "reject"

    @pytest.mark.parametrize("treatment", ["combine", "split", "edge_target"])
    def test_display_only_participation_is_rejected(self, tmp_path, treatment):
        base, candidates, aggregate = _whole_home_fixture()
        rooms = [c for c in candidates if not c["display_only"]]
        decision_map = _approve_all(candidates)
        if treatment == "combine":
            decision_map[aggregate["package_candidate_id"]] = {
                "decision": "approve",
                "combine_with": [rooms[0]["package_candidate_id"]],
            }
        elif treatment == "split":
            decision_map[aggregate["package_candidate_id"]] = {
                "decision": "approve",
                "split_groups": [
                    [rooms[0]["child_work_item_ids"][0]],
                    [rooms[1]["child_work_item_ids"][0]],
                ],
            }
        else:
            decision_map[rooms[0]["package_candidate_id"]] = {
                "decision": "approve",
                "combine_with": [aggregate["package_candidate_id"]],
            }
        fake = FakeOpenAI(responses=[_sol_response(decision_map)])
        with pytest.raises(PassExecutionError) as excinfo:
            _run(tmp_path, base, candidates, fake)
        assert excinfo.value.code == "SolReviewContract"
        assert "display-only" in excinfo.value.message


# ── checkpoints ──────────────────────────────────────────────────────────────

class TestSolCheckpoints:
    def _checkpoint_file(self, tmp_path):
        return sol_checkpoint_path(terra_checkpoint_dir(
            tmp_path / "artifacts", PROPERTY_KEY, RUN_ID
        ))

    def test_same_fingerprint_reuses_without_a_provider_call(self, tmp_path):
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[_sol_response(_approve_all(candidates))])
        first, _ = _run(tmp_path, base, candidates, fake)
        assert self._checkpoint_file(tmp_path).exists()

        strict = FakeOpenAI(raises=AssertionError("provider must not be called"))
        second, _ = _run(tmp_path, base, candidates, strict)
        assert strict.requests == []
        (call,) = second["sol_calls"]
        assert call["usage_source"] == "checkpoint"
        # Original provider token numbers are preserved as provenance.
        assert call["total_tokens"] == 1020
        assert second["package_decisions"] == first["package_decisions"]
        res = validate_package_review_result(second, estimate_id=EST_ID)
        assert res.ok, res.errors

    def test_model_change_invalidates_the_fingerprint(self, tmp_path):
        base, candidates = _fixture()
        _run(tmp_path, base, candidates,
             FakeOpenAI(responses=[_sol_response(_approve_all(candidates))]))
        fresh = FakeOpenAI(responses=[_sol_response(_approve_all(candidates))])
        _run(tmp_path, base, candidates, fresh,
             runtime=_runtime(sol_model="gpt-5.4-other"))
        assert len(fresh.requests) == 1  # checkpoint ignored, listing re-bought

    def test_corrupt_checkpoint_is_ignored(self, tmp_path):
        base, candidates = _fixture()
        _run(tmp_path, base, candidates,
             FakeOpenAI(responses=[_sol_response(_approve_all(candidates))]))
        self._checkpoint_file(tmp_path).write_text("not json", encoding="utf-8")
        fresh = FakeOpenAI(responses=[_sol_response(_approve_all(candidates))])
        result, _ = _run(tmp_path, base, candidates, fresh)
        assert len(fresh.requests) == 1
        assert result["sol_calls"][0]["usage_source"] == "provider"


# ── end-to-end: Sol failure preserves Terra checkpoints ──────────────────────

class TestShadowEnvelopeEndToEnd:
    def test_sol_failure_fails_closed_and_the_retry_replays_terra(self, tmp_path):
        """A Sol contract failure after a successful Terra review produces a
        private failed envelope; the completed Terra checkpoint survives, so
        the retry replays it at zero cost and only re-buys the Sol call."""
        from tests.test_renovation_architecture_package_candidates import _pkg_item
        from tests.test_renovation_architecture_terra import (
            _checkpoint_files,
            _conditions,
            _estimate_id,
            _init_runtime,
            _issue,
            _photo_files,
            _review_response,
        )
        from tools.renovation_architecture.ids import make_package_candidate_id
        from tools.renovation_architecture.runtime import (
            build_shadow_envelope,
            reset_runtime_for_tests,
        )
        from tools.renovation_architecture.validators import validate_envelope

        reset_runtime_for_tests()
        try:
            # A defect driver with two distinct views: the review accepts it
            # and the candidate builder emits one kitchen candidate.
            runtime = _init_runtime(tmp_path, _pkg_item(
                "worn_kitchen_floor", "kitchen", "kitchen_modernization",
                "package_driver",
            ))
            photos, paths = _photo_files(tmp_path, {
                "kitchen_1.png": ("kitchen", (200, 30, 30)),
                "kitchen_2.png": ("kitchen", (10, 200, 240)),
            })
            issues = [
                _issue("worn_kitchen_floor", "kitchen_1.png"),
                _issue("worn_kitchen_floor", "kitchen_2.png"),
            ]
            (condition,) = _conditions(runtime, issues, photos)

            def _envelope(fake):
                client = VLMClient()
                fake.attach(client)
                return build_shadow_envelope(
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

            # First run: Terra succeeds; the drained queue hands Sol the
            # default {"ok": true} payload, which violates the decision
            # contract -> a private failed envelope, never `uncertain`.
            failed = _envelope(FakeOpenAI(responses=[
                _review_response({condition.condition_id: "supported"})
            ]))
            res = validate_envelope(failed)
            assert res.ok, res.errors
            assert failed["state"] == "failed"
            assert failed["reason"] == "parse"
            assert "uncertain" not in json.dumps(failed["result"])
            assert len(_checkpoint_files(tmp_path)) == 1  # Terra survived

            # Retry: Terra replays from its checkpoint (zero new calls); only
            # the Sol listing call is bought.
            candidate_id = make_package_candidate_id(
                estimate_id=_estimate_id(runtime),
                package_type="kitchen_modernization",
                estimate_unit_id=condition.estimate_unit_id,
            )
            retry_fake = FakeOpenAI(responses=[
                _sol_response({candidate_id: "approve"})
            ])
            envelope = _envelope(retry_fake)
            res = validate_envelope(envelope)
            assert res.ok, res.errors
            assert envelope["state"] == "complete"
            assert len(retry_fake.requests) == 1  # the Sol call only
            (terra_call,) = envelope["result"]["terra_calls"]
            assert terra_call["usage_source"] == "checkpoint"
            (sol_call,) = envelope["result"]["sol_calls"]
            assert sol_call["usage_source"] == "provider"
            (decision,) = envelope["result"]["package_decisions"]
            assert decision["decision"] == "approve"
            assert decision["package_candidate_id"] == candidate_id
            # Reconciliation applied the approval: the child is absorbed
            # exactly once and packaged totals carry the effective range.
            (application,) = envelope["result"]["package_applications"]
            assert application["status"] == "applied"
            assert application["package_candidate_id"] == candidate_id
            (entry,) = envelope["result"]["coverage_ledger"]
            assert entry["representation"] == "absorbed_by_package"
            assert entry["package_id"] == candidate_id
            assert (entry["low"], entry["high"]) == (0, 0)
            totals = envelope["result"]["totals"]
            assert totals["standalone"] == {"low": 0, "high": 0}
            assert totals["packaged"] == {
                "low": application["effective_low"],
                "high": application["effective_high"],
            }
        finally:
            reset_runtime_for_tests()
