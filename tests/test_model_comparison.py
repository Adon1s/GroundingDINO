"""
Tests for tools.model_comparison — head-to-head Model A vs Model B across pipeline passes.

Covers the 8 cases from the plan:
- test_build_cells_structure
- test_parse_judge_verdict
- test_aggregate_leaderboard
- test_2c_2d_coupling_in_coupled_mode
- test_suggested_config_never_splits_2c_2d
- test_2d_cell_enrichment
- test_coupled_2d_preserves_2c_context
- test_checkpoint_resume
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


from tools.model_comparison import (
    FIXTURE_CACHE_SCHEMA_VERSION,
    FIXTURE_PIPELINE_VERSION,
    SKILLS,
    FixtureCells,
    ImageRecord,
    ModelCells,
    TwoDRow,
    _ab_mapping_for,
    _build_user_prompt_2a,
    _canonicalize_verdict,
    _judge_json_schema_for,
    _record_from_dict,
    _record_to_dict,
    _run_2d_batch,
    _run_phase_0,
    _twod_row_to_prompt_dict,
    aggregate_per_skill,
    backfill_2d_judge_into_rows,
    build_recommendation,
    hydrate_records_from_fixture_caches,
    load_checkpoint,
    load_fixture_cache,
    run_judge_for_skill,
    run_analysis_model_cells,
    save_checkpoint,
    save_fixture_cache,
    upsert_fixture_cache_entry,
)
from tools.scene_classifier_passes import (
    Pass2aResult,
    Pass2bResult,
    Pass2cResult,
    Pass2dResult,
)


# ─── Shared fixture builders ─────────────────────────────────────────────────


def _make_pass_2a_result(text="Ceiling stains visible above kitchen."):
    return Pass2aResult(observations_freeform=text, raw_response=text)


def _make_pass_2b_result(descriptions=None):
    descriptions = descriptions or [
        "Water stain on ceiling above sink",
        "Dated cabinet finish",
    ]
    return Pass2bResult(
        observations=[{"description": d} for d in descriptions],
        raw_response="[]",
    )


# observation-kind-v2: model_comparison's live-2c cells still consume the
# retired v1 label vocabulary (labeled_debug/labeled_forward), and the harness
# entry point is blocked with a RuntimeError until it is migrated (Task 3
# measurement work). Fixture-replay and judge/aggregation tests keep running.
# See docs/HANDOFF_kind_ontology_task1.md.
blocked_live_2c = pytest.mark.skip(
    reason="model_comparison live-2c cells blocked: Pass 2c contract migrated to "
    "observation-kind-v2; see docs/HANDOFF_kind_ontology_task1.md"
)


def _make_pass_2c_result(rows=None):
    rows = rows or [
        {"description": "Water stain on ceiling", "label": "defect_or_damage"},
        {"description": "Dated cabinet finish", "label": "upgrade_candidate"},
    ]
    forward = [
        r for r in rows if r["label"] in {"defect_or_damage", "upgrade_candidate"}
    ]
    return Pass2cResult(
        labeled_debug=rows,
        labeled_forward=forward,
        raw_response="[]",
    )


def _make_pass_2d_result(obs, item_id, kind="defect"):
    return Pass2dResult(
        observation=obs,
        resolved_item_id=item_id,
        resolved_kind=kind,
        raw_response="{}",
    )


def _stub_vlm_client():
    """VLMClient with AsyncMock analyze_image / analyze_text. Customize per test."""
    client = MagicMock()
    client.analyze_image = AsyncMock(return_value="{}")
    client.analyze_text = AsyncMock(return_value="{}")
    return client


def _stub_retriever(candidates=None):
    """CatalogEmbeddingsRetriever stub. retrieve_candidates returns MatchCandidate-like objs."""
    from tools.catalog_embeddings import MatchCandidate

    default = candidates if candidates is not None else [
        MatchCandidate(
            item_id="ceiling_water_stain",
            name="Ceiling Water Stain",
            kind="defect",
            trade_bucket="drywall",
            severity=3,
            description="Water staining or discoloration on a ceiling surface.",
            support_any=("stain", "ceiling"),
            defaultHidden=False,
            drop_if_generic=False,
            score=0.82,
            scene_groups=("kitchen",),
        ),
        MatchCandidate(
            item_id="generic_repair",
            name="Generic Repair",
            kind="defect",
            trade_bucket="drywall",
            severity=2,
            description="Generic drywall repair candidate.",
            support_any=("repair",),
            defaultHidden=False,
            drop_if_generic=False,
            score=0.71,
            scene_groups=("kitchen",),
        ),
    ]
    retriever = MagicMock()
    retriever.retrieve_candidates = MagicMock(return_value=default)
    return retriever


def _make_image_info(photo_key="img_001.jpg", scene="kitchen"):
    return {
        "property_id": "test_prop",
        "photo_key": photo_key,
        "image_path": f"/fake/{photo_key}",
        "scene": scene,
        "scene_group": "kitchen",
    }


def _make_cache_image_info(tmp_path, photo_key="img_001.jpg", scene="kitchen", data=b"image-bytes"):
    image_path = tmp_path / photo_key
    image_path.write_bytes(data)
    image = _make_image_info(photo_key=photo_key, scene=scene)
    image["image_path"] = str(image_path)
    return image


def _make_cache_payload(property_id="test_prop"):
    return {
        "schema_version": FIXTURE_CACHE_SCHEMA_VERSION,
        "fixture_pipeline_version": FIXTURE_PIPELINE_VERSION,
        "property_id": property_id,
        "entries": {},
    }


def _make_record_for_image(image):
    return ImageRecord(
        property_id=image["property_id"],
        photo_key=image["photo_key"],
        image_path=image["image_path"],
        scene=image["scene"],
        scene_group=image["scene_group"],
    )


def _make_fixture():
    return FixtureCells(
        pass_2a="GPT fixture freeform",
        pass_2b=[
            {"description": "Water stain on ceiling"},
            {"description": "Dated cabinet finish"},
        ],
        pass_2c=[
            {"description": "Water stain on ceiling", "label": "defect_or_damage"},
            {"description": "Dated cabinet finish", "label": "upgrade_candidate"},
        ],
    )


# ─── test_build_cells_structure ──────────────────────────────────────────────


class TestBuildCellsStructure:
    """Stub VLM → every image gets 5 cells for each analysis model."""

    @blocked_live_2c
    def test_run_analysis_model_cells_produces_all_five_cells(self):
        image = _make_image_info()
        fixture = _make_fixture()
        retriever = _stub_retriever()
        vlm = _stub_vlm_client()

        with patch("tools.model_comparison.run_pass_2a", new=AsyncMock(
                return_value=_make_pass_2a_result("Local model 2a freeform"))), \
             patch("tools.model_comparison.run_pass_2b", new=AsyncMock(
                return_value=_make_pass_2b_result())), \
             patch("tools.model_comparison.run_pass_2c", new=AsyncMock(
                return_value=_make_pass_2c_result())), \
             patch("tools.model_comparison.run_pass_2d", new=AsyncMock(
                side_effect=lambda observation, **kw: _make_pass_2d_result(
                    observation, "ceiling_water_stain"))):

            cells = _run_async(run_analysis_model_cells(
                image_info=image,
                fixture=fixture,
                vlm_client=vlm,
                model_config={"model": "test_model", "url": "http://x", "provider": "lmstudio"},
                retriever=retriever,
                skip_skills=set(),
            ))

        assert cells.pass_2a == "Local model 2a freeform"
        assert len(cells.pass_2b) == 2
        assert len(cells.pass_2c) == 2
        assert len(cells.pass_2d_isolated) > 0
        assert len(cells.pass_2c_2d_coupled) > 0

    @blocked_live_2c
    def test_skip_skills_are_respected(self):
        image = _make_image_info()
        fixture = _make_fixture()
        retriever = _stub_retriever()
        vlm = _stub_vlm_client()

        with patch("tools.model_comparison.run_pass_2a", new=AsyncMock(
                return_value=_make_pass_2a_result())), \
             patch("tools.model_comparison.run_pass_2b", new=AsyncMock(
                return_value=_make_pass_2b_result())), \
             patch("tools.model_comparison.run_pass_2c", new=AsyncMock(
                return_value=_make_pass_2c_result())), \
             patch("tools.model_comparison.run_pass_2d", new=AsyncMock(
                side_effect=lambda observation, **kw: _make_pass_2d_result(
                    observation, "ceiling_water_stain"))):

            cells = _run_async(run_analysis_model_cells(
                image_info=image,
                fixture=fixture,
                vlm_client=vlm,
                model_config={"model": "test_model", "url": "http://x", "provider": "lmstudio"},
                retriever=retriever,
                skip_skills={"2d_isolated", "2c+2d_coupled"},
            ))

        assert cells.pass_2a is not None
        assert cells.pass_2b
        assert cells.pass_2c
        assert cells.pass_2d_isolated == []
        assert cells.pass_2c_2d_coupled == []


# ─── test_parse_judge_verdict ────────────────────────────────────────────────


class TestParseJudgeVerdict:
    """Each per-skill judge prompt: happy path + structured-output failure handling."""

    def _make_record_with_cells(self):
        rec = ImageRecord(
            property_id="p",
            photo_key="img.jpg",
            image_path="/fake/img.jpg",
            scene="kitchen",
            scene_group="kitchen",
        )
        rec.fixture = _make_fixture()
        rec.model_a = ModelCells(
            pass_2a="Model A sees things",
            pass_2b=[{"description": "Model A observation"}],
            pass_2c=[{"description": "Model A observation", "label": "defect_or_damage"}],
            pass_2d_isolated=[
                TwoDRow(
                    observation="Water stain on ceiling",
                    kind="defect",
                    candidates=[{"item_id": "ceiling_water_stain", "name": "Water Stain",
                                 "trade_bucket": "drywall", "score": 0.82}],
                    chosen_id="ceiling_water_stain",
                    chosen_null=False,
                )
            ],
            pass_2c_2d_coupled=[
                TwoDRow(
                    observation="Water stain on ceiling",
                    kind="defect",
                    candidates=[{"item_id": "ceiling_water_stain", "name": "Water Stain",
                                 "trade_bucket": "drywall", "score": 0.82}],
                    chosen_id="ceiling_water_stain",
                    chosen_null=False,
                    from_2c={"label": "defect_or_damage", "kind": "defect"},
                )
            ],
        )
        rec.model_b = ModelCells(
            pass_2a="Model B sees things",
            pass_2b=[{"description": "Model B observation"}],
            pass_2c=[{"description": "Model B observation", "label": "defect_or_damage"}],
            pass_2d_isolated=[
                TwoDRow(
                    observation="Water stain on ceiling",
                    kind="defect",
                    candidates=[{"item_id": "ceiling_water_stain", "name": "Water Stain",
                                 "trade_bucket": "drywall", "score": 0.82}],
                    chosen_id=None,
                    chosen_null=True,
                )
            ],
            pass_2c_2d_coupled=[
                TwoDRow(
                    observation="Water stain on ceiling",
                    kind="defect",
                    candidates=[{"item_id": "ceiling_water_stain", "name": "Water Stain",
                                 "trade_bucket": "drywall", "score": 0.82}],
                    chosen_id=None,
                    chosen_null=True,
                    from_2c={"label": "defect_or_damage", "kind": "defect"},
                )
            ],
        )
        return rec

    def test_happy_path_each_skill(self):
        rec = self._make_record_with_cells()
        for skill in SKILLS:
            vlm = _stub_vlm_client()
            verdict_payload = {"skill": skill, "winner": "model_a",
                               "scores": {"model_a": {"x": 5}, "model_b": {"x": 2}},
                               "rationale": "because"}
            vlm.analyze_image = AsyncMock(return_value=json.dumps(verdict_payload))
            vlm.analyze_text = AsyncMock(return_value=json.dumps(verdict_payload))

            verdict, parse_failures = _run_async(run_judge_for_skill(
                skill=skill,
                record=rec,
                vlm_client=vlm,
                judge_config={"model": "gpt-5.4", "api_key": "k", "provider": "openai"},
            ))
            assert verdict is not None, f"skill={skill} returned None"
            assert parse_failures == 0
            assert verdict["winner"] == "model_a"

    def test_structured_malformed_json_returns_none_without_retry(self):
        rec = self._make_record_with_cells()
        for skill in SKILLS:
            vlm = _stub_vlm_client()
            vlm.analyze_image = AsyncMock(return_value="not a JSON response")
            vlm.analyze_text = AsyncMock(return_value="not a JSON response")

            verdict, parse_failures = _run_async(run_judge_for_skill(
                skill=skill,
                record=rec,
                vlm_client=vlm,
                judge_config={"model": "gpt-5.4", "api_key": "k", "provider": "openai"},
            ))
            assert verdict is None, f"skill={skill} should have returned None for malformed JSON"
            assert parse_failures == 1
            if skill == "2a":
                assert vlm.analyze_image.call_count == 1
            else:
                assert vlm.analyze_text.call_count == 1

    def test_judge_passes_openai_structured_output_options(self):
        rec = self._make_record_with_cells()
        vlm = _stub_vlm_client()
        vlm.analyze_text = AsyncMock(return_value=json.dumps(
            {"skill": "2b", "winner": "tie", "scores": {"model_a": {}, "model_b": {}}, "rationale": ""}
        ))

        _run_async(run_judge_for_skill(
            skill="2b",
            record=rec,
            vlm_client=vlm,
            judge_config={"model": "gpt-5.5", "api_key": "k", "provider": "openai"},
        ))

        kwargs = vlm.analyze_text.call_args.kwargs
        assert kwargs["response_json_schema"]["type"] == "object"
        assert kwargs["response_schema_name"].startswith("model_comparison_judge_2b")
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["verbosity"] == "low"

    def test_2a_judge_uses_image_endpoint(self):
        """2a needs image; other skills should use text endpoint."""
        rec = self._make_record_with_cells()
        vlm = _stub_vlm_client()
        vlm.analyze_image = AsyncMock(return_value=json.dumps(
            {"skill": "2a", "winner": "tie", "scores": {"model_a": {}, "model_b": {}}, "rationale": ""}
        ))
        vlm.analyze_text = AsyncMock(return_value="should not be used")
        _run_async(run_judge_for_skill("2a", rec, vlm,
                                       {"model": "gpt-5.4", "api_key": "k", "provider": "openai"}))
        vlm.analyze_image.assert_called_once()
        vlm.analyze_text.assert_not_called()

    def test_non_image_skills_use_text_endpoint(self):
        rec = self._make_record_with_cells()
        for skill in ("2b", "2c", "2d_isolated", "2c+2d_coupled"):
            vlm = _stub_vlm_client()
            vlm.analyze_text = AsyncMock(return_value=json.dumps(
                {"skill": skill, "winner": "tie", "scores": {"model_a": {}, "model_b": {}}, "rationale": ""}
            ))
            vlm.analyze_image = AsyncMock(return_value="should not be used")

            _run_async(run_judge_for_skill(skill, rec, vlm,
                                           {"model": "gpt-5.4", "api_key": "k", "provider": "openai"}))
            vlm.analyze_text.assert_called_once()
            vlm.analyze_image.assert_not_called()


# ─── test_aggregate_leaderboard ──────────────────────────────────────────────


class TestAggregateLeaderboard:
    """Canned verdicts → correct per-pass win counts + split detection."""

    def _make_record_with_judge(self, judge_winners):
        """judge_winners: {skill: winner}"""
        rec = ImageRecord(property_id="p", photo_key="img.jpg", image_path="/fake/img.jpg")
        for s, w in judge_winners.items():
            rec.judge[s] = {
                "skill": s,
                "winner": w,
                "scores": {
                    "model_a": {"accuracy": 4},
                    "model_b": {"accuracy": 3},
                },
                "rationale": "",
            }
        return rec

    def test_aggregate_counts_per_skill(self):
        # 3 images, Model A wins 2a twice, Model B wins once
        records = [
            self._make_record_with_judge({"2a": "model_a"}),
            self._make_record_with_judge({"2a": "model_a"}),
            self._make_record_with_judge({"2a": "model_b"}),
        ]
        agg = aggregate_per_skill(records, "2a")
        assert agg["judged_images"] == 3
        assert agg["model_a_wins"] == 2
        assert agg["model_b_wins"] == 1
        assert agg["ties"] == 0

    def test_aggregate_averages_scores(self):
        records = [
            self._make_record_with_judge({"2a": "model_a"}),
            self._make_record_with_judge({"2a": "model_b"}),
        ]
        agg = aggregate_per_skill(records, "2a")
        assert agg["avg_scores"]["model_a"]["accuracy"] == 4.0
        assert agg["avg_scores"]["model_b"]["accuracy"] == 3.0

    def test_recommendation_winners_split_true(self):
        # Model A wins 2a+2b; Model B wins 2c+2d+coupled → split=True
        aggregate = {
            "2a":              {"judged_images": 3, "model_a_wins": 3, "model_b_wins": 0, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2b":              {"judged_images": 3, "model_a_wins": 3, "model_b_wins": 0, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2c":              {"judged_images": 3, "model_a_wins": 0, "model_b_wins": 3, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2d_isolated":     {"judged_images": 3, "model_a_wins": 0, "model_b_wins": 3, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2c+2d_coupled":   {"judged_images": 3, "model_a_wins": 0, "model_b_wins": 3, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
        }
        rec = build_recommendation(aggregate)
        assert rec["winners_split"] is True
        assert rec["best_2a"] == "model_a"
        assert rec["best_2b"] == "model_a"
        assert rec["best_2c"] == "model_b"
        assert rec["best_2d_isolated"] == "model_b"
        assert rec["best_2c+2d_coupled"] == "model_b"

    def test_recommendation_winners_split_false_when_all_same(self):
        # Model A sweeps everything
        aggregate = {
            s: {"judged_images": 3, "model_a_wins": 3, "model_b_wins": 0, "ties": 0,
                "avg_scores": {"model_a": {}, "model_b": {}}}
            for s in SKILLS
        }
        rec = build_recommendation(aggregate)
        assert rec["winners_split"] is False
        assert all(rec[k] == "model_a" for k in
                   ("best_2a", "best_2b", "best_2c", "best_2d_isolated", "best_2c+2d_coupled"))
        assert rec["confidence"] == "high"

    def test_no_judged_images_returns_tie(self):
        agg = aggregate_per_skill([], "2a")
        assert agg["judged_images"] == 0
        assert agg["model_a_wins"] == 0
        assert agg["model_b_wins"] == 0


# ─── test_2c_2d_coupling_in_coupled_mode ─────────────────────────────────────


class TestCouplingConstraint:
    """Coupled 2c+2d must never cross-feed — each model's own 2c goes to its own 2d."""

    @blocked_live_2c
    def test_run_local_cells_feeds_own_2c_into_own_2d_coupled(self):
        """Model A's coupled 2d uses Model A's own 2c rows (by construction)."""
        image = _make_image_info()
        fixture = _make_fixture()
        retriever = _stub_retriever()
        vlm = _stub_vlm_client()

        # Model A's 2c returns a DIFFERENT set of rows than the fixture's 2c.
        model_a_2c_rows = [
            {"description": "Model A-only cabinet issue", "label": "defect_or_damage"},
        ]
        with patch("tools.model_comparison.run_pass_2a", new=AsyncMock(
                return_value=_make_pass_2a_result())), \
             patch("tools.model_comparison.run_pass_2b", new=AsyncMock(
                return_value=_make_pass_2b_result())), \
             patch("tools.model_comparison.run_pass_2c", new=AsyncMock(
                return_value=_make_pass_2c_result(model_a_2c_rows))), \
             patch("tools.model_comparison.run_pass_2d", new=AsyncMock(
                side_effect=lambda observation, **kw: _make_pass_2d_result(
                    observation, "ceiling_water_stain"))):

            cells = _run_async(run_analysis_model_cells(
                image_info=image,
                fixture=fixture,
                vlm_client=vlm,
                model_config={"model": "test_model", "url": "http://x", "provider": "lmstudio"},
                retriever=retriever,
                skip_skills=set(),
            ))

        # The coupled 2d rows were derived from the model's OWN 2c, not the fixture's 2c.
        coupled_obs = {r.observation for r in cells.pass_2c_2d_coupled}
        assert "Model A-only cabinet issue" in coupled_obs
        # And must NOT contain the fixture 2c observations
        fixture_obs_set = {r["description"] for r in fixture.pass_2c}
        assert not coupled_obs.intersection(fixture_obs_set)

    @blocked_live_2c
    def test_isolated_2d_uses_fixture_not_own_2c(self):
        """Isolated 2d must use the fixture's F_2c (identical across models)."""
        image = _make_image_info()
        fixture = _make_fixture()
        retriever = _stub_retriever()
        vlm = _stub_vlm_client()

        # Model's own 2c is arbitrary (won't be used for isolated 2d).
        own_2c = [{"description": "Ignored for isolated", "label": "defect_or_damage"}]

        with patch("tools.model_comparison.run_pass_2a", new=AsyncMock(
                return_value=_make_pass_2a_result())), \
             patch("tools.model_comparison.run_pass_2b", new=AsyncMock(
                return_value=_make_pass_2b_result())), \
             patch("tools.model_comparison.run_pass_2c", new=AsyncMock(
                return_value=_make_pass_2c_result(own_2c))), \
             patch("tools.model_comparison.run_pass_2d", new=AsyncMock(
                side_effect=lambda observation, **kw: _make_pass_2d_result(
                    observation, "ceiling_water_stain"))):

            cells = _run_async(run_analysis_model_cells(
                image_info=image,
                fixture=fixture,
                vlm_client=vlm,
                model_config={"model": "test_model", "url": "http://x", "provider": "lmstudio"},
                retriever=retriever,
                skip_skills=set(),
            ))

        iso_obs = {r.observation for r in cells.pass_2d_isolated}
        fixture_obs_set = {r["description"] for r in fixture.pass_2c
                           if r["label"] in {"defect_or_damage", "upgrade_candidate"}}
        assert iso_obs == fixture_obs_set


# ─── test_suggested_config_never_splits_2c_2d ────────────────────────────────


class TestSuggestedConfigNeverSplits:
    """suggested_config_per_pass must never send 2c and 2d to different models."""

    def test_split_2c_2d_defers_both_to_coupled_winner(self):
        # 2c → model_a, 2d_iso → model_b, coupled → model_b
        aggregate = {
            "2a":              {"judged_images": 2, "model_a_wins": 2, "model_b_wins": 0, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2b":              {"judged_images": 2, "model_a_wins": 2, "model_b_wins": 0, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2c":              {"judged_images": 2, "model_a_wins": 2, "model_b_wins": 0, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2d_isolated":     {"judged_images": 2, "model_a_wins": 0, "model_b_wins": 2, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2c+2d_coupled":   {"judged_images": 2, "model_a_wins": 0, "model_b_wins": 2, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
        }
        rec = build_recommendation(aggregate)

        # best_2c and best_2d_isolated are still shown independently…
        assert rec["best_2c"] == "model_a"
        assert rec["best_2d_isolated"] == "model_b"

        # …but suggested_config_per_pass should have 2c == 2d
        per_pass = rec["suggested_config_per_pass"]
        assert per_pass["2c"] == per_pass["2d"], (
            f"per_pass_config split 2c={per_pass['2c']} vs 2d={per_pass['2d']}"
        )

        # And should defer to the coupled winner
        assert per_pass["2c"] == "model_b"
        assert per_pass["2d"] == "model_b"

        # Rationale mentions the split
        assert "split" in rec["rationale"].lower()

    def test_no_split_keeps_per_pass_winners(self):
        aggregate = {
            "2a":              {"judged_images": 2, "model_a_wins": 1, "model_b_wins": 1, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2b":              {"judged_images": 2, "model_a_wins": 2, "model_b_wins": 0, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2c":              {"judged_images": 2, "model_a_wins": 2, "model_b_wins": 0, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2d_isolated":     {"judged_images": 2, "model_a_wins": 2, "model_b_wins": 0, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
            "2c+2d_coupled":   {"judged_images": 2, "model_a_wins": 2, "model_b_wins": 0, "ties": 0,
                                "avg_scores": {"model_a": {}, "model_b": {}}},
        }
        rec = build_recommendation(aggregate)
        assert rec["suggested_config_per_pass"]["2c"] == "model_a"
        assert rec["suggested_config_per_pass"]["2d"] == "model_a"


# ─── test_2d_cell_enrichment ─────────────────────────────────────────────────


class TestTwoDEnrichment:
    """2d cell rows contain full diagnostic context."""

    def test_twod_prompt_serialization_strips_unused_candidate_fields(self):
        row = TwoDRow(
            observation="Water stain on ceiling",
            kind="defect",
            candidates=[{
                "item_id": "ceiling_water_stain",
                "name": "Ceiling Water Stain",
                "kind": "defect",
                "trade_bucket": "drywall",
                "score": 0.82,
                "description": "Long catalog copy that does not belong in judge prompts.",
                "support_any": ["stain"],
                "defaultHidden": False,
                "drop_if_generic": False,
            }],
            chosen_id="ceiling_water_stain",
            chosen_null=False,
        )

        out = _twod_row_to_prompt_dict(row, "row_001")

        assert out["row_id"] == "row_001"
        assert out["candidates"] == [{
            "item_id": "ceiling_water_stain",
            "name": "Ceiling Water Stain",
            "kind": "defect",
            "trade_bucket": "drywall",
            "score": 0.82,
        }]

    def test_row_has_diagnostic_fields_after_batch(self):
        retriever = _stub_retriever()
        vlm = _stub_vlm_client()

        labeled_rows = [
            {"description": "Water stain on ceiling", "label": "defect_or_damage"},
        ]

        with patch("tools.model_comparison.run_pass_2d", new=AsyncMock(
                side_effect=lambda observation, **kw: _make_pass_2d_result(
                    observation, "ceiling_water_stain"))):
            rows = _run_async(_run_2d_batch(
                labeled_rows=labeled_rows,
                vlm_client=vlm,
                model_config={"model": "m"},
                retriever=retriever,
                scene_group="kitchen",
                include_from_2c=False,
            ))

        assert len(rows) == 1
        r = rows[0]
        # Shape
        assert r.observation == "Water stain on ceiling"
        assert r.kind == "defect"
        assert r.candidates and all("score" in c for c in r.candidates)
        assert r.chosen_id == "ceiling_water_stain"
        assert r.chosen_null is False
        # Pre-judge — these are None until backfill
        assert r.judge_correct_id is None
        assert r.judge_correct_rank is None
        assert r.judge_correct_in_candidates is None
        # Non-coupled → no from_2c
        assert r.from_2c is None

    def test_backfill_writes_judge_correct_id_and_rank(self):
        row = TwoDRow(
            observation="Water stain on ceiling",
            kind="defect",
            candidates=[
                {"item_id": "generic_repair", "name": "Generic", "trade_bucket": "x", "score": 0.7},
                {"item_id": "ceiling_water_stain", "name": "Stain", "trade_bucket": "x", "score": 0.8},
            ],
            chosen_id="generic_repair",
            chosen_null=False,
        )
        verdict = {
            "per_observation": [
                {"observation": "Water stain on ceiling",
                 "judge_correct_id": "ceiling_water_stain"},
            ],
        }
        backfill_2d_judge_into_rows([row], verdict)
        assert row.judge_correct_id == "ceiling_water_stain"
        assert row.judge_correct_rank == 2  # 1-indexed
        assert row.judge_correct_in_candidates is True

    def test_backfill_uses_structured_row_ids_for_coupled_rows(self):
        model_a_row = TwoDRow(
            observation="Model A observation",
            kind="defect",
            candidates=[{"item_id": "model_a_pick", "name": "Model A Pick", "trade_bucket": "x", "score": 0.8}],
            chosen_id=None,
            chosen_null=True,
        )
        model_b_row = TwoDRow(
            observation="Model B observation",
            kind="defect",
            candidates=[{"item_id": "model_b_pick", "name": "Model B Pick", "trade_bucket": "x", "score": 0.8}],
            chosen_id=None,
            chosen_null=True,
        )
        verdict = {
            "per_model_rows": [
                {"model_key": "model_a", "row_id": "row_001", "judge_correct_id": "model_a_pick", "correct": True,
                 "failure_attribution": "none"},
                {"model_key": "model_b", "row_id": "row_001", "judge_correct_id": "model_b_pick", "correct": True,
                 "failure_attribution": "none"},
            ]
        }

        backfill_2d_judge_into_rows([model_a_row], verdict, model_key="model_a")
        backfill_2d_judge_into_rows([model_b_row], verdict, model_key="model_b")

        assert model_a_row.judge_correct_id == "model_a_pick"
        assert model_b_row.judge_correct_id == "model_b_pick"

    def test_backfill_rank_null_when_correct_not_in_candidates(self):
        row = TwoDRow(
            observation="Water stain on ceiling",
            kind="defect",
            candidates=[
                {"item_id": "generic_repair", "name": "Generic", "trade_bucket": "x", "score": 0.7},
            ],
            chosen_id="generic_repair",
            chosen_null=False,
        )
        verdict = {
            "per_observation": [
                {"observation": "Water stain on ceiling",
                 "judge_correct_id": "something_else_not_in_candidates"},
            ],
        }
        backfill_2d_judge_into_rows([row], verdict)
        assert row.judge_correct_id == "something_else_not_in_candidates"
        assert row.judge_correct_rank is None
        assert row.judge_correct_in_candidates is False

    def test_backfill_handles_null_judge_correct_id(self):
        row = TwoDRow(
            observation="Water stain on ceiling",
            kind="defect",
            candidates=[{"item_id": "generic_repair", "name": "x", "trade_bucket": "x",
                         "score": 0.5}],
            chosen_id="generic_repair",
            chosen_null=False,
        )
        verdict = {
            "per_observation": [
                {"observation": "Water stain on ceiling", "judge_correct_id": None},
            ],
        }
        backfill_2d_judge_into_rows([row], verdict)
        assert row.judge_correct_id is None
        assert row.judge_correct_rank is None
        assert row.judge_correct_in_candidates is False

    def test_empty_candidates_yields_null_row(self):
        retriever = MagicMock()
        retriever.retrieve_candidates = MagicMock(return_value=[])
        vlm = _stub_vlm_client()

        labeled_rows = [{"description": "obscure obs", "label": "defect_or_damage"}]

        with patch("tools.model_comparison.run_pass_2d", new=AsyncMock(
                return_value=_make_pass_2d_result("obscure obs", None))):
            rows = _run_async(_run_2d_batch(
                labeled_rows=labeled_rows,
                vlm_client=vlm,
                model_config={"model": "m"},
                retriever=retriever,
                scene_group="kitchen",
                include_from_2c=False,
            ))
        assert len(rows) == 1
        assert rows[0].candidates == []
        assert rows[0].chosen_id is None
        assert rows[0].chosen_null is True


# ─── test_coupled_2d_preserves_2c_context ────────────────────────────────────


class TestCoupledPreserves2cContext:
    """Coupled rows must include from_2c: {label, kind} matching the same model's 2c."""

    def test_coupled_rows_have_from_2c(self):
        retriever = _stub_retriever()
        vlm = _stub_vlm_client()

        labeled_rows = [
            {"description": "Water stain on ceiling", "label": "defect_or_damage"},
            {"description": "Dated cabinet finish", "label": "upgrade_candidate"},
        ]

        with patch("tools.model_comparison.run_pass_2d", new=AsyncMock(
                side_effect=lambda observation, **kw: _make_pass_2d_result(
                    observation, "ceiling_water_stain", kind=kw.get("kind", "defect")))):
            rows = _run_async(_run_2d_batch(
                labeled_rows=labeled_rows,
                vlm_client=vlm,
                model_config={"model": "m"},
                retriever=retriever,
                scene_group="kitchen",
                include_from_2c=True,
            ))

        assert len(rows) == 2
        # Both rows carry from_2c
        assert all(r.from_2c is not None for r in rows)

        # from_2c.label/kind matches what went in
        for row in rows:
            assert row.from_2c["kind"] == row.kind
            if row.from_2c["label"] == "defect_or_damage":
                assert row.from_2c["kind"] == "defect"
            elif row.from_2c["label"] == "upgrade_candidate":
                assert row.from_2c["kind"] == "upgrade"

    def test_isolated_rows_have_no_from_2c(self):
        retriever = _stub_retriever()
        vlm = _stub_vlm_client()

        labeled_rows = [{"description": "Water stain", "label": "defect_or_damage"}]

        with patch("tools.model_comparison.run_pass_2d", new=AsyncMock(
                return_value=_make_pass_2d_result("Water stain", "ceiling_water_stain"))):
            rows = _run_async(_run_2d_batch(
                labeled_rows=labeled_rows,
                vlm_client=vlm,
                model_config={"model": "m"},
                retriever=retriever,
                scene_group="kitchen",
                include_from_2c=False,
            ))
        assert all(r.from_2c is None for r in rows)


# ─── test_checkpoint_resume ──────────────────────────────────────────────────


class TestCheckpointResume:
    """Partial checkpoint → only missing phases re-run."""

    def test_save_and_load_roundtrip(self, tmp_path):
        output_path = tmp_path / "out.json"
        rec = ImageRecord(
            property_id="p",
            photo_key="img.jpg",
            image_path="/fake/img.jpg",
            scene="kitchen",
            scene_group="kitchen",
        )
        rec.fixture = _make_fixture()
        rec.model_a = ModelCells(
            pass_2a="model_a text",
            pass_2b=[{"description": "obs"}],
            pass_2c=[{"description": "obs", "label": "defect_or_damage"}],
            pass_2d_isolated=[TwoDRow(
                observation="obs", kind="defect", candidates=[{"item_id": "x", "score": 0.5}],
                chosen_id="x", chosen_null=False,
                judge_correct_id="x", judge_correct_rank=1, judge_correct_in_candidates=True,
            )],
            pass_2c_2d_coupled=[TwoDRow(
                observation="obs", kind="defect", candidates=[{"item_id": "x", "score": 0.5}],
                chosen_id="x", chosen_null=False,
                from_2c={"label": "defect_or_damage", "kind": "defect"},
            )],
        )
        rec.judge["2a"] = {"skill": "2a", "winner": "model_a"}

        save_checkpoint(output_path, {"p/img.jpg": rec})
        loaded = load_checkpoint(output_path)

        assert "p/img.jpg" in loaded
        r2 = loaded["p/img.jpg"]
        assert r2.property_id == "p"
        assert r2.fixture.pass_2a == "GPT fixture freeform"
        assert r2.model_a.pass_2a == "model_a text"
        assert len(r2.model_a.pass_2d_isolated) == 1
        assert r2.model_a.pass_2d_isolated[0].judge_correct_id == "x"
        assert r2.model_a.pass_2d_isolated[0].judge_correct_rank == 1
        assert r2.model_a.pass_2c_2d_coupled[0].from_2c == {
            "label": "defect_or_damage", "kind": "defect"
        }
        assert r2.judge["2a"] == {"skill": "2a", "winner": "model_a"}

    def test_load_missing_checkpoint_returns_empty(self, tmp_path):
        output_path = tmp_path / "nope.json"
        loaded = load_checkpoint(output_path)
        assert loaded == {}

    def test_phase_complete_detection(self):
        from tools.model_comparison import _phase_complete

        rec = ImageRecord(property_id="p", photo_key="k", image_path="/fake")
        # Fresh record → nothing done
        assert _phase_complete(rec, "fixture") is False
        assert _phase_complete(rec, "model_a") is False
        assert _phase_complete(rec, "model_b") is False
        assert _phase_complete(rec, "judge") is False

        # Fixture only
        rec.fixture = _make_fixture()
        assert _phase_complete(rec, "fixture") is True
        assert _phase_complete(rec, "model_a") is False

        # Add model_a 2a
        rec.model_a.pass_2a = "something"
        assert _phase_complete(rec, "model_a") is True
        assert _phase_complete(rec, "model_b") is False

        # A downstream cell does not make the phase complete while active 2a is missing.
        rec.model_b.pass_2b = [{"description": "x"}]
        assert _phase_complete(rec, "model_b") is False
        assert _phase_complete(rec, "model_b", {"2a"}) is True

        # Judge — all 5 skills needed
        for s in SKILLS:
            rec.judge[s] = {"winner": "model_a"}
        assert _phase_complete(rec, "judge") is True

        # Remove one judge → no longer complete
        rec.judge["2a"] = None
        assert _phase_complete(rec, "judge") is False

    def test_record_roundtrip_preserves_2d_enrichment(self):
        rec = ImageRecord(property_id="p", photo_key="k", image_path="/fake")
        rec.fixture = FixtureCells(pass_2a="a", pass_2b=[], pass_2c=[])
        rec.model_a.pass_2d_isolated = [TwoDRow(
            observation="obs", kind="defect",
            candidates=[{"item_id": "x", "score": 0.5}],
            chosen_id="x", chosen_null=False,
            judge_correct_id="x", judge_correct_rank=1, judge_correct_in_candidates=True,
        )]
        d = _record_to_dict(rec)
        r2 = _record_from_dict(d)
        row = r2.model_a.pass_2d_isolated[0]
        assert row.judge_correct_id == "x"
        assert row.judge_correct_rank == 1
        assert row.judge_correct_in_candidates is True


# ─── Concurrency tests (added for the concurrency addendum) ──────────────────


class TestFixtureCache:
    """Persistent Phase 0 fixture cache."""

    def test_fixture_cache_save_and_load_roundtrip(self, tmp_path):
        image = _make_cache_image_info(tmp_path)
        payload = _make_cache_payload()
        model_config = {"model": "gpt-5.5", "provider": "openai", "api_key": "secret"}

        assert upsert_fixture_cache_entry(payload, image, _make_fixture(), model_config) is True

        cache_path = tmp_path / "fixture_cache.json"
        save_fixture_cache(cache_path, payload)
        loaded = load_fixture_cache(cache_path)

        entry = loaded["entries"][image["photo_key"]]
        assert entry["property_id"] == image["property_id"]
        assert entry["photo_key"] == image["photo_key"]
        assert entry["fixture_model"] == "gpt-5.5"
        assert entry["fixture"]["pass_2a"] == "GPT fixture freeform"
        assert "api_key" not in json.dumps(entry)

    def test_valid_cache_hit_hydrates_fresh_record(self, tmp_path):
        image = _make_cache_image_info(tmp_path)
        payload = _make_cache_payload()
        model_config = {"model": "gpt-5.5", "provider": "openai"}
        upsert_fixture_cache_entry(payload, image, _make_fixture(), model_config)

        rec = _make_record_for_image(image)
        records = {f"{image['property_id']}/{image['photo_key']}": rec}

        hits = hydrate_records_from_fixture_caches(
            [image],
            records,
            {image["property_id"]: payload},
            model_config,
        )

        assert hits == 1
        assert rec.fixture.pass_2a == "GPT fixture freeform"
        assert len(rec.fixture.pass_2b) == 2

    @pytest.mark.parametrize("stale_case", ["image_hash", "scene", "model_config", "schema"])
    def test_stale_cache_misses(self, tmp_path, stale_case):
        image = _make_cache_image_info(tmp_path)
        payload = _make_cache_payload()
        model_config = {"model": "gpt-5.5", "provider": "openai"}
        upsert_fixture_cache_entry(payload, image, _make_fixture(), model_config)

        lookup_image = dict(image)
        lookup_config = dict(model_config)
        if stale_case == "image_hash":
            Path(image["image_path"]).write_bytes(b"changed-image-bytes")
        elif stale_case == "scene":
            lookup_image["scene"] = "bathroom"
        elif stale_case == "model_config":
            lookup_config["model"] = "gpt-5.5-mini"
        elif stale_case == "schema":
            payload["entries"][image["photo_key"]]["schema_version"] = -1

        rec = _make_record_for_image(lookup_image)
        records = {f"{lookup_image['property_id']}/{lookup_image['photo_key']}": rec}

        hits = hydrate_records_from_fixture_caches(
            [lookup_image],
            records,
            {lookup_image["property_id"]: payload},
            lookup_config,
        )

        assert hits == 0
        assert rec.fixture.pass_2a is None

    def test_checkpoint_fixture_takes_precedence_over_persistent_cache(self, tmp_path):
        image = _make_cache_image_info(tmp_path)
        payload = _make_cache_payload()
        model_config = {"model": "gpt-5.5", "provider": "openai"}
        upsert_fixture_cache_entry(payload, image, _make_fixture(), model_config)

        rec = _make_record_for_image(image)
        rec.fixture = FixtureCells(pass_2a="checkpoint fixture", pass_2b=[], pass_2c=[])
        records = {f"{image['property_id']}/{image['photo_key']}": rec}

        hits = hydrate_records_from_fixture_caches(
            [image],
            records,
            {image["property_id"]: payload},
            model_config,
        )

        assert hits == 0
        assert rec.fixture.pass_2a == "checkpoint fixture"

    def test_run_phase_0_writes_fresh_fixture_to_cache(self, tmp_path):
        image = _make_cache_image_info(tmp_path)
        rec = _make_record_for_image(image)
        records = {f"{image['property_id']}/{image['photo_key']}": rec}
        model_config = {"model": "gpt-5.5", "provider": "openai"}
        payload = _make_cache_payload()
        cache_path = tmp_path / "phase0_fixture_cache.json"

        async def _save_fixture(image_info, fixture):
            upsert_fixture_cache_entry(payload, image_info, fixture, model_config)
            save_fixture_cache(cache_path, payload)

        with patch("tools.model_comparison.build_fixture_for_image", new=AsyncMock(
                return_value=_make_fixture())):
            _run_async(_run_phase_0(
                images=[image],
                records=records,
                vlm_client=_stub_vlm_client(),
                judge_config=model_config,
                output_path=tmp_path / "out.json",
                save_ckpt=AsyncMock(),
                concurrency=1,
                save_fixture_cache_entry=_save_fixture,
            ))

        loaded = load_fixture_cache(cache_path)
        assert image["photo_key"] in loaded["entries"]
        assert loaded["entries"][image["photo_key"]]["fixture"]["pass_2a"] == "GPT fixture freeform"

    def test_refresh_fixture_cache_forces_phase_0_rebuild(self, tmp_path):
        image = _make_cache_image_info(tmp_path)
        payload = _make_cache_payload()
        model_config = {"model": "gpt-5.5", "provider": "openai"}
        upsert_fixture_cache_entry(payload, image, _make_fixture(), model_config)

        rec = _make_record_for_image(image)
        records = {f"{image['property_id']}/{image['photo_key']}": rec}
        hits = hydrate_records_from_fixture_caches(
            [image],
            records,
            {image["property_id"]: payload},
            model_config,
            refresh=True,
        )
        assert hits == 0
        assert rec.fixture.pass_2a is None

        build_mock = AsyncMock(return_value=_make_fixture())
        with patch("tools.model_comparison.build_fixture_for_image", new=build_mock):
            _run_async(_run_phase_0(
                images=[image],
                records=records,
                vlm_client=_stub_vlm_client(),
                judge_config=model_config,
                output_path=tmp_path / "out.json",
                save_ckpt=AsyncMock(),
                concurrency=1,
            ))

        assert build_mock.await_count == 1
        assert rec.fixture.pass_2a == "GPT fixture freeform"


class TestConcurrency:
    """Concurrency refactor: _run_phase_0 bounded by semaphore, resume still works."""

    def _make_image_record_pair(self, photo_key: str, with_fixture: bool = False):
        img = {
            "property_id": "prop",
            "photo_key": photo_key,
            "image_path": f"/fake/{photo_key}",
            "scene": "kitchen",
            "scene_group": "kitchen",
        }
        rec = ImageRecord(
            property_id="prop",
            photo_key=photo_key,
            image_path=f"/fake/{photo_key}",
            scene="kitchen",
            scene_group="kitchen",
        )
        if with_fixture:
            rec.fixture = _make_fixture()
        return img, rec

    def test_concurrency_respects_semaphore_limit(self, tmp_path):
        """Stub build_fixture_for_image with a counter — assert concurrent in-flight ≤ N."""
        from tools.model_comparison import _run_phase_0

        N_IMAGES = 10
        CONCURRENCY = 3

        # Build 10 images + fresh records (no fixtures yet).
        images = []
        records = {}
        for i in range(N_IMAGES):
            key = f"img_{i:03d}.jpg"
            img, rec = self._make_image_record_pair(key, with_fixture=False)
            images.append(img)
            records[f"prop/{key}"] = rec

        # Shared state tracking max concurrent in-flight calls.
        state = {"in_flight": 0, "max_in_flight": 0, "call_count": 0}

        async def _stub_build(image_info, vlm_client, judge_config):
            state["in_flight"] += 1
            state["call_count"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            # Sleep briefly so overlap is possible. Without this, each call finishes
            # before the next starts and max_in_flight would always be 1.
            await asyncio.sleep(0.05)
            state["in_flight"] -= 1
            return _make_fixture()

        save_ckpt_mock = AsyncMock()

        with patch("tools.model_comparison.build_fixture_for_image", new=_stub_build):
            _run_async(_run_phase_0(
                images=images,
                records=records,
                vlm_client=_stub_vlm_client(),
                judge_config={"model": "gpt-5.4", "api_key": "k", "provider": "openai"},
                output_path=tmp_path / "ckpt.json",
                save_ckpt=save_ckpt_mock,
                concurrency=CONCURRENCY,
            ))

        # All 10 images were processed
        assert state["call_count"] == N_IMAGES
        # Every record got its fixture set
        assert all(r.fixture is not None for r in records.values())
        # Concurrency bound respected — never more than CONCURRENCY in flight.
        assert state["max_in_flight"] <= CONCURRENCY, (
            f"max_in_flight={state['max_in_flight']} exceeded "
            f"concurrency={CONCURRENCY}"
        )
        # And the semaphore actually DID let us run concurrently (otherwise
        # the bound is trivially respected and the test is meaningless).
        assert state["max_in_flight"] >= 2, (
            f"Only reached max_in_flight={state['max_in_flight']} — "
            f"semaphore may not be enabling concurrency at all"
        )

    def test_resume_with_concurrency(self, tmp_path):
        """Partial checkpoint: 5/10 fixture-done → only 5 build_fixture calls."""
        from tools.model_comparison import _run_phase_0

        N_IMAGES = 10
        N_ALREADY_DONE = 5
        CONCURRENCY = 3

        images = []
        records = {}
        for i in range(N_IMAGES):
            key = f"img_{i:03d}.jpg"
            already_done = i < N_ALREADY_DONE  # First 5 are "resumed" (have fixture).
            img, rec = self._make_image_record_pair(key, with_fixture=already_done)
            images.append(img)
            records[f"prop/{key}"] = rec

        state = {"call_count": 0, "called_keys": []}

        async def _stub_build(image_info, vlm_client, judge_config):
            state["call_count"] += 1
            state["called_keys"].append(image_info["photo_key"])
            return _make_fixture()

        save_ckpt_mock = AsyncMock()

        with patch("tools.model_comparison.build_fixture_for_image", new=_stub_build):
            _run_async(_run_phase_0(
                images=images,
                records=records,
                vlm_client=_stub_vlm_client(),
                judge_config={"model": "gpt-5.4", "api_key": "k", "provider": "openai"},
                output_path=tmp_path / "ckpt.json",
                save_ckpt=save_ckpt_mock,
                concurrency=CONCURRENCY,
            ))

        # Only the 5 that were NOT already done should have been built.
        assert state["call_count"] == N_IMAGES - N_ALREADY_DONE, (
            f"Expected {N_IMAGES - N_ALREADY_DONE} fixture calls for the "
            f"unfinished images, got {state['call_count']}"
        )
        # And the ones called should be exactly the unfinished keys (img_005.. img_009).
        expected_keys = {f"img_{i:03d}.jpg" for i in range(N_ALREADY_DONE, N_IMAGES)}
        assert set(state["called_keys"]) == expected_keys
        # Records that were pre-populated should keep their existing fixture.
        for i in range(N_ALREADY_DONE):
            rec = records[f"prop/img_{i:03d}.jpg"]
            assert rec.fixture is not None
            assert rec.fixture.pass_2a == "GPT fixture freeform"
        # Records that were unfinished should now have fixtures too.
        for i in range(N_ALREADY_DONE, N_IMAGES):
            rec = records[f"prop/img_{i:03d}.jpg"]
            assert rec.fixture is not None

class TestProviderNeutralBlinding:
    def test_prompt_contains_only_blind_display_names(self):
        record = ImageRecord(property_id="p", photo_key="photo.jpg", image_path="/fake")
        record.model_a.pass_2a = "candidate A result"
        record.model_b.pass_2a = "candidate B result"
        prompt = _build_user_prompt_2a(
            record,
            {"A": "model_b", "B": "model_a"},
        )
        assert "Model A" in prompt
        assert "Model B" in prompt
        assert "model_a" not in prompt
        assert "model_b" not in prompt
        assert "Sol" not in prompt
        assert "Terra" not in prompt

    def test_schema_and_canonicalization_avoid_alias_collisions(self):
        schema = _judge_json_schema_for("2a")
        properties = schema["properties"]
        assert "A_unique_good" in properties
        assert "B_unique_good" in properties
        assert "model_a_unique_good" not in properties

        verdict = {
            "winner": "A",
            "scores": {"A": {"accuracy": 5}, "B": {"accuracy": 2}},
            "A_unique_good": ["from presented A"],
            "B_unique_good": ["from presented B"],
            "A_hallucinations": [],
            "B_hallucinations": ["bad"],
        }
        result = _canonicalize_verdict(
            verdict,
            {"A": "model_b", "B": "model_a"},
        )
        assert result["winner"] == "model_b"
        assert result["model_b_unique_good"] == ["from presented A"]
        assert result["model_a_unique_good"] == ["from presented B"]
        assert result["model_a_hallucinations"] == ["bad"]

    def test_forced_and_random_orders_are_neutral_and_deterministic(self):
        assert _ab_mapping_for("model_a_first", "p", "2a") == {
            "A": "model_a", "B": "model_b"
        }
        assert _ab_mapping_for("model_b_first", "p", "2a") == {
            "A": "model_b", "B": "model_a"
        }
        assert _ab_mapping_for("random", "p", "2a") == _ab_mapping_for("random", "p", "2a")
class TestNeutralCheckpointSchema:
    def test_fingerprint_mismatch_is_rejected(self, tmp_path):
        output = tmp_path / "run.json"
        save_checkpoint(output, {}, "fingerprint-a", {"model_a": {"usage": {"calls": 2}}})
        meta = {}
        assert load_checkpoint(output, "fingerprint-a", meta) == {}
        assert meta["phase_stats"]["model_a"]["usage"]["calls"] == 2
        with pytest.raises(ValueError, match="configuration does not match"):
            load_checkpoint(output, "fingerprint-b")

    def test_legacy_checkpoint_is_rejected_cleanly(self, tmp_path):
        output = tmp_path / "run.json"
        checkpoint = output.with_suffix(".checkpoint.json")
        checkpoint.write_text(json.dumps({"p/photo.jpg": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="legacy Gemma/Qwen schema"):
            load_checkpoint(output, "new-fingerprint")