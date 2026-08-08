"""Unit tests for the Pass 2a prompt-ablation benchmark (no API calls).

Covers: the run_pass_2a prompt override, the orchestrator's benchmark meta
hooks (frozen replay + prompt override), fingerprint resume rejection,
photo-checkpoint round trips, the attribution material-flip stop gate,
candidate decision gates, blinding determinism/de-aliasing, and the judge
payload shape (all repeats of exactly one image per Sol call).

The pre-2f totals recompute is covered by an offline integration test that
runs the real costing arithmetic against a stored canary artifact when one is
available (skipped otherwise). End-to-end with live models is the documented
smoke run, not a unit test.
"""
import asyncio
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools import benchmark_pass2a as bench
from tools.scene_classifier_passes import (
    PASS_2A_SYSTEM_PROMPT,
    PASS_2A_USER_PROMPT,
    run_pass_2a,
)


# ---------------------------------------------------------------------------
# run_pass_2a override
# ---------------------------------------------------------------------------

def _client_capturing(calls):
    async def analyze_image(**kwargs):
        calls.append(kwargs)
        return "some freeform notes"
    return SimpleNamespace(analyze_image=analyze_image)


def test_run_pass_2a_default_prompt_unchanged(tmp_path):
    calls = []
    result = asyncio.run(run_pass_2a(
        image_path=tmp_path / "x.jpg",
        vlm_client=_client_capturing(calls),
        model_config={"model": "m"},
    ))
    assert calls[0]["user_prompt"] == PASS_2A_USER_PROMPT
    assert calls[0]["system_prompt"] == PASS_2A_SYSTEM_PROMPT
    assert result.observations_freeform == "some freeform notes"


def test_run_pass_2a_override_passthrough(tmp_path):
    calls = []
    asyncio.run(run_pass_2a(
        image_path=tmp_path / "x.jpg",
        vlm_client=_client_capturing(calls),
        model_config={"model": "m"},
        user_prompt="Enumerate every visible condition.",
    ))
    assert calls[0]["user_prompt"] == "Enumerate every visible condition."


# ---------------------------------------------------------------------------
# Orchestrator meta hooks (frozen replay / prompt override)
# ---------------------------------------------------------------------------

def _bare_orchestrator(vlm_client):
    from tools.scene_classifier_orchestrator import SceneClassifierOrchestrator
    return SceneClassifierOrchestrator(
        qwen_config={"url": "http://localhost:1", "model": "qwen"},
        gpt5_config={"url": "http://localhost:1", "model": "gpt", "api_key": "k"},
        vlm_client=vlm_client,
    )


def _options_2a_only(**meta):
    from tools.pass_config import SceneClassifierRunOptions
    options = SceneClassifierRunOptions.from_analysis_profile(
        analysis_profile="standard",
        toggles={"1a": False, "1b": False, "1c": False,
                 "2b": False, "2c": False, "2d": False, "2e": False, "2f": False},
        pipeline_mode="classification_only",
    )
    return options.with_meta(run_id="t", photo_key="x.jpg", **meta)


def test_orchestrator_frozen_freeform_skips_vision_call(tmp_path):
    client = SimpleNamespace(analyze_image=AsyncMock(side_effect=AssertionError(
        "frozen replay must not call the vision model")))
    orchestrator = _bare_orchestrator(client)
    result = asyncio.run(orchestrator.analyze_image(
        image_path=tmp_path / "x.jpg",
        options=_options_2a_only(pass_2a_frozen_freeform="  frozen capture  "),
    ))
    assert result.observations_freeform == "frozen capture"
    assert result.models_used["2a"] == "frozen_replay"
    client.analyze_image.assert_not_called()


def test_orchestrator_meta_prompt_override_reaches_2a(tmp_path):
    calls = []

    async def analyze_image(**kwargs):
        calls.append(kwargs)
        return "notes"
    orchestrator = _bare_orchestrator(SimpleNamespace(analyze_image=analyze_image))
    asyncio.run(orchestrator.analyze_image(
        image_path=tmp_path / "x.jpg",
        options=_options_2a_only(pass_2a_user_prompt="inventory wording"),
    ))
    assert calls[0]["user_prompt"] == "inventory wording"


def test_orchestrator_without_meta_uses_production_prompt(tmp_path):
    calls = []

    async def analyze_image(**kwargs):
        calls.append(kwargs)
        return "notes"
    orchestrator = _bare_orchestrator(SimpleNamespace(analyze_image=analyze_image))
    asyncio.run(orchestrator.analyze_image(
        image_path=tmp_path / "x.jpg",
        options=_options_2a_only(),
    ))
    assert calls[0]["user_prompt"] == PASS_2A_USER_PROMPT


# ---------------------------------------------------------------------------
# Fingerprint guard / resume rejection
# ---------------------------------------------------------------------------

def test_guard_fingerprint_accepts_identical_and_rejects_changed(tmp_path):
    fp = {"git_head": "abc", "prompt_sha256": "p1", "catalog_sha256": "c1"}
    bench.guard_fingerprint(tmp_path, fp)
    bench.guard_fingerprint(tmp_path, dict(fp))  # identical resume OK

    for key in fp:
        changed = {**fp, key: "DIFFERENT"}
        with pytest.raises(SystemExit) as excinfo:
            bench.guard_fingerprint(tmp_path, changed)
        assert key in str(excinfo.value)


# ---------------------------------------------------------------------------
# Photo checkpoint round trip
# ---------------------------------------------------------------------------

def test_photo_checkpoint_roundtrip(tmp_path):
    from tools.analyzer_cli import ImageResult
    image_paths = [tmp_path / "photo_001.jpg", tmp_path / "photo_002.jpg"]
    ckpt_dir = tmp_path / ".photos"
    ckpt_dir.mkdir()
    saved = ImageResult(
        image_path=str(image_paths[0]),
        scene_data={"observations_freeform": "x", "scene": "kitchen"},
        scene="kitchen",
        processing_time=1.25,
    )
    (ckpt_dir / "photo_001.jpg.json").write_text(
        json.dumps(dataclasses.asdict(saved)), encoding="utf-8")

    cached = bench._load_photo_checkpoints(ckpt_dir, image_paths)
    assert list(cached) == [0]
    assert cached[0].scene == "kitchen"
    assert cached[0].scene_data["observations_freeform"] == "x"

    # A checkpoint whose recorded path differs (moved images) is not reused.
    other = bench._load_photo_checkpoints(
        ckpt_dir, [tmp_path / "elsewhere" / "photo_001.jpg", image_paths[1]])
    assert other == {}


# ---------------------------------------------------------------------------
# Attribution stop gate
# ---------------------------------------------------------------------------

def _totals(midpoint, items):
    return {
        "final_rehab": {"low": midpoint - 10, "high": midpoint + 10,
                        "midpoint": midpoint},
        "line_items": [
            {"catalog_item_id": cid, "cost_low": low, "cost_high": high}
            for cid, low, high in items
        ],
    }


def test_attribution_gate_fires_on_material_flip():
    totals = {"prop": {
        1: _totals(50_000, [("stable_id", 2_000, 4_000),
                            ("kitchen_pkg", 10_000, 20_000)]),
        2: _totals(50_000, [("stable_id", 2_000, 4_000)]),
        3: _totals(50_000, [("stable_id", 2_000, 4_000),
                            ("kitchen_pkg", 10_000, 20_000)]),
    }}
    gate = bench.evaluate_attribution_gate(totals, flip_abs_usd=10_000, flip_pct=15)
    assert not gate["passed"]
    (flip,) = gate["failing_flips"]
    assert flip["catalog_item_id"] == "kitchen_pkg"
    assert flip["present_in_reps"] == [1, 3]
    assert flip["line_midpoint_usd"] == 15_000


def test_attribution_gate_holds_on_stable_or_immaterial_flips():
    totals = {"prop": {
        1: _totals(100_000, [("stable_id", 10_000, 20_000),
                             ("small_flip", 1_000, 2_000)]),
        2: _totals(100_000, [("stable_id", 10_000, 20_000)]),
        3: _totals(100_000, [("stable_id", 10_000, 20_000),
                             ("small_flip", 1_000, 2_000)]),
    }}
    gate = bench.evaluate_attribution_gate(totals, flip_abs_usd=10_000, flip_pct=15)
    assert gate["passed"]
    # The immaterial flip is still recorded for the report.
    assert gate["properties"]["prop"]["flips"][0]["catalog_item_id"] == "small_flip"
    assert gate["properties"]["prop"]["flips"][0]["material"] is False


def test_attribution_gate_pct_threshold():
    # $6k line on a $30k midpoint = 20% -> material even though < $10k.
    totals = {"prop": {
        1: _totals(30_000, [("flip", 4_000, 8_000)]),
        2: _totals(30_000, []),
        3: _totals(30_000, [("flip", 4_000, 8_000)]),
    }}
    gate = bench.evaluate_attribution_gate(totals, flip_abs_usd=10_000, flip_pct=15)
    assert not gate["passed"]


# ---------------------------------------------------------------------------
# Metric math + candidate gates
# ---------------------------------------------------------------------------

def test_jaccard():
    assert bench.jaccard(set(), set()) == 1.0
    assert bench.jaccard({"a"}, set()) == 0.0
    assert bench.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def _stab(jaccard_mean, spread, midpoint):
    return {
        "mean_resolved_id_jaccard": jaccard_mean,
        "median_midpoint_spread_usd": spread,
        "per_property": {"prop": {"median_midpoint_usd": midpoint}},
    }


def _jm(per_variant):
    return {"per_variant": per_variant, "review_queue": []}


GATES = {
    "unsupported_rate_pp": 2, "unsupported_min_claims": 2,
    "recall_drop_pp": 5, "midpoint_shift_pct": 10,
    "jaccard_gain": 0.05, "spread_reduction_pct": 25,
}


def _variant_counts(unsupported=0, total=50, recall=90.0, critical=None):
    return {
        "claims_total": total,
        "supported": total - unsupported,
        "unsupported": unsupported,
        "uncertain": 0,
        "unsupported_rate_pct": round(100 * unsupported / total, 2),
        "uncertain_rate_pct": 0.0,
        "critical_hallucination_photos": critical or {},
        "critical_in_2plus_repeats": {
            k: v for k, v in (critical or {}).items() if len(set(v)) >= 2},
        "supported_recall_vs_gold_pct": recall,
    }


def test_candidate_advances_on_jaccard_gain():
    result = bench.evaluate_candidate_gates(
        _stab(0.60, 40_000, 100_000), _stab(0.70, 35_000, 102_000),
        _jm({"baseline": _variant_counts(), "checklist": _variant_counts()}),
        "baseline", "checklist", GATES,
    )
    assert result["verdict"] == "advance"
    assert result["delta_jaccard"] == pytest.approx(0.10)


def test_candidate_rejected_on_new_repeated_critical_hallucination():
    result = bench.evaluate_candidate_gates(
        _stab(0.60, 40_000, 100_000), _stab(0.80, 10_000, 100_000),
        _jm({
            "baseline": _variant_counts(),
            "checklist": _variant_counts(
                critical={"p/photo_001.jpg": [1, 2]}),
        }),
        "baseline", "checklist", GATES,
    )
    assert result["verdict"] == "reject"
    assert "critical hallucination" in result["reject_reasons"][0]


def test_candidate_rejected_on_unsupported_rate_rise():
    result = bench.evaluate_candidate_gates(
        _stab(0.60, 40_000, 100_000), _stab(0.80, 10_000, 100_000),
        _jm({"baseline": _variant_counts(unsupported=1),
             "checklist": _variant_counts(unsupported=4)}),
        "baseline", "checklist", GATES,
    )
    assert result["verdict"] == "reject"


def test_candidate_rejected_on_recall_drop():
    result = bench.evaluate_candidate_gates(
        _stab(0.60, 40_000, 100_000), _stab(0.80, 10_000, 100_000),
        _jm({"baseline": _variant_counts(recall=92.0),
             "checklist": _variant_counts(recall=80.0)}),
        "baseline", "checklist", GATES,
    )
    assert result["verdict"] == "reject"


def test_candidate_no_improvement_without_gains():
    result = bench.evaluate_candidate_gates(
        _stab(0.60, 40_000, 100_000), _stab(0.62, 38_000, 100_000),
        _jm({"baseline": _variant_counts(), "checklist": _variant_counts()}),
        "baseline", "checklist", GATES,
    )
    assert result["verdict"] == "no_improvement"


def test_midpoint_shift_is_flagged_not_autorejected():
    result = bench.evaluate_candidate_gates(
        _stab(0.60, 40_000, 100_000), _stab(0.70, 20_000, 120_000),
        _jm({"baseline": _variant_counts(), "checklist": _variant_counts()}),
        "baseline", "checklist", GATES,
    )
    assert result["verdict"] == "advance"
    assert any("midpoint shifted" in f for f in result["flags_for_steven"])


# ---------------------------------------------------------------------------
# Blinding + judge de-aliasing
# ---------------------------------------------------------------------------

def test_blinding_is_deterministic_per_round_and_photo():
    import random
    seen = set()
    for photo in ("photo_001.jpg", "photo_002.jpg", "photo_003.jpg"):
        seed = f"round1|prop|{photo}"
        first = random.Random(seed).random() < 0.5
        again = random.Random(seed).random() < 0.5
        assert first == again
        seen.add(first)
    # Sanity: the seed actually varies assignments across many photos.
    outcomes = {random.Random(f"round1|prop|photo_{i}.jpg").random() < 0.5
                for i in range(50)}
    assert outcomes == {True, False}


def _judgment(mapping, claims, gold=None):
    return {
        "blinding": mapping,
        "verdict": {"claims": claims,
                    "per_prompt": {"A": {}, "B": {}}},
        "gold_coverage": gold,
    }


def test_judge_metrics_dealias_and_critical_detection():
    claims = [
        {"prompt": "A", "repeat": 1, "claim_index": 0, "label": "supported",
         "hallucination_category": "none", "cost_bearing": True},
        {"prompt": "B", "repeat": 1, "claim_index": 0, "label": "unsupported",
         "hallucination_category": "structural_failure", "cost_bearing": True},
        {"prompt": "B", "repeat": 2, "claim_index": 1, "label": "unsupported",
         "hallucination_category": "structural_failure", "cost_bearing": False},
        {"prompt": "B", "repeat": 3, "claim_index": 0, "label": "uncertain",
         "hallucination_category": "none", "cost_bearing": False},
    ]
    judgments = {
        "prop/photo_001.jpg": _judgment(
            # A is checklist here: de-aliasing must credit variants, not letters
            {"A": "checklist", "B": "baseline"}, claims,
            gold={"conditions": [
                {"gold_id": "g1", "covered_by_A": True, "covered_by_B": False},
                {"gold_id": "g2", "covered_by_A": True, "covered_by_B": True},
            ]},
        ),
    }
    metrics = bench.judge_metrics(judgments, repeats=3)
    baseline = metrics["per_variant"]["baseline"]
    checklist = metrics["per_variant"]["checklist"]
    assert checklist["supported"] == 1 and checklist["unsupported"] == 0
    assert baseline["unsupported"] == 2 and baseline["uncertain"] == 1
    # structural_failure in repeats 1 and 2 -> critical in 2+ repeats
    assert "prop/photo_001.jpg" in baseline["critical_in_2plus_repeats"]
    assert checklist["critical_in_2plus_repeats"] == {}
    assert checklist["supported_recall_vs_gold_pct"] == 100.0
    assert baseline["supported_recall_vs_gold_pct"] == 50.0
    # Judge-flagged claims land in the review queue with variant names.
    assert {r["variant"] for r in metrics["review_queue"]} == {"baseline"}
    assert len(metrics["review_queue"]) == 3


# ---------------------------------------------------------------------------
# Judge payload: one call, one image, ALL repeats of both variants
# ---------------------------------------------------------------------------

def test_judge_photo_payload_contains_all_repeats(tmp_path, monkeypatch):
    records = {}

    def fake_records(stage_label, rep, property_key, photo_key):
        records.setdefault(stage_label, []).append(rep)
        return {"claims": [f"{stage_label} r{rep} claim"], "kept": [],
                "excluded": [], "resolved_ids": [], "priced": []}

    monkeypatch.setattr(bench, "load_photo_repeat_records", fake_records)

    image_calls = []

    async def analyze_image(**kwargs):
        image_calls.append(kwargs)
        return json.dumps({
            "claims": [], "per_prompt": {
                "A": {"coverage": "", "missing_visible_conditions": [],
                      "verbosity_bias": "", "systematic_patterns": ""},
                "B": {"coverage": "", "missing_visible_conditions": [],
                      "verbosity_bias": "", "systematic_patterns": ""},
            },
        })

    ctx = SimpleNamespace(
        vlm_client=SimpleNamespace(analyze_image=analyze_image,
                                   analyze_text=AsyncMock()),
        gpt5_config={"provider": "openai", "model": "x", "api_key": "k"},
    )
    config = {"repeats": 3,
              "judge": {"model": "gpt-5.6-sol", "reasoning_effort": "medium",
                        "max_tokens": 100}}
    manifest = {"images_root": str(tmp_path)}
    result = asyncio.run(bench.judge_photo(
        ctx, config, manifest, "baseline_vs_checklist",
        "baseline", "checklist", "prop",
        {"photo_key": "photo_001.jpg"}, gold_conditions=[],
    ))

    # Exactly one image call, carrying this photo.
    assert len(image_calls) == 1
    assert image_calls[0]["image_path"].name == "photo_001.jpg"
    assert image_calls[0]["model"] == "gpt-5.6-sol"
    assert image_calls[0]["reasoning_effort"] == "medium"
    assert image_calls[0]["response_json_schema"]["required"] == ["claims", "per_prompt"]
    # All 3 repeats of BOTH variants were loaded and included.
    assert sorted(records["variant_baseline"]) == [1, 2, 3]
    assert sorted(records["variant_checklist"]) == [1, 2, 3]
    payload = result["payload_prompts"]
    assert {len(payload["A"]), len(payload["B"])} == {3}
    # The concealed mapping covers both variants under blinded letters.
    assert sorted(result["blinding"].values()) == ["baseline", "checklist"]
    # No gold -> no text call.
    ctx.vlm_client.analyze_text.assert_not_called()


# ---------------------------------------------------------------------------
# stage_init validation (fixture source tree)
# ---------------------------------------------------------------------------

def _fake_source_tree(tmp_path, freeform="old paint, worn floor"):
    images_root = tmp_path / "images"
    (images_root / "prop_x").mkdir(parents=True)
    (images_root / "prop_x" / "photo_001.jpg").write_bytes(b"jpegdata")
    run_dir = tmp_path / "artifacts" / "prop_x" / "20260808_000000_aaaa"
    run_dir.mkdir(parents=True)
    (run_dir / "photo_intel_debug.json").write_text(json.dumps({
        "catalog_version": "3.1",
        "property_metadata": {"beds": 3},
        "photos": {"photo_001.jpg": {
            "scene": {"id": "kitchen"},
            "features": {"observations_freeform": freeform},
        }},
    }), encoding="utf-8")
    return {
        "properties": ["prop_x"],
        "images_root": str(images_root),
        "source_artifacts_root": str(tmp_path / "artifacts"),
    }


def test_stage_init_builds_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "MANIFEST_PATH", tmp_path / "manifest.json")
    config = _fake_source_tree(tmp_path)
    bench.stage_init(config)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["photo_count"] == 1
    photo = manifest["properties"]["prop_x"]["photos"][0]
    assert photo["scene"] == "kitchen"
    assert photo["frozen_2a"] == "old paint, worn floor"
    assert len(photo["image_sha256"]) == 64
    assert manifest["properties"]["prop_x"]["property_metadata"] == {"beds": 3}


def test_stage_init_rejects_missing_frozen_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "MANIFEST_PATH", tmp_path / "manifest.json")
    config = _fake_source_tree(tmp_path, freeform="   ")
    with pytest.raises(SystemExit, match="frozen 2a capture"):
        bench.stage_init(config)


# ---------------------------------------------------------------------------
# Prompts / config sanity
# ---------------------------------------------------------------------------

def test_prompts_file_baseline_matches_production():
    prompts = bench.load_prompts()
    assert prompts["baseline"]["text"] == PASS_2A_USER_PROMPT
    assert set(prompts) >= {"baseline", "checklist", "evidence"}
    assert prompts["evidence"]["text"].startswith(prompts["checklist"]["text"])
    for entry in prompts.values():
        assert len(entry["sha256"]) == 64


def test_config_routes_terra_low_and_disables_2f():
    config = bench.load_config()
    assert config["model_overrides"] == {k: "gpt-5.6-terra" for k in ("2a", "2b", "2c")}
    assert "2d" not in config["model_overrides"]  # local Qwen route
    assert all(v == "low" for v in config["reasoning_efforts"].values())
    assert config["pass_toggles"] == {"2f": False}
    assert config["judge"]["model"] == "gpt-5.6-sol"
    assert config["repeats"] == 3


# ---------------------------------------------------------------------------
# Pre-2f totals: offline integration against a stored canary artifact
# ---------------------------------------------------------------------------

STORED_ARTIFACT = Path(
    "C:/Users/Steven/PycharmProjects/realtorvision-backend/artifacts_canary/"
    "candidate/redfin_11000447/20260808_085243_19c4a4d0/photo_intel_debug.json"
)


@pytest.mark.skipif(not STORED_ARTIFACT.is_file(),
                    reason="stored canary artifact not available")
def test_compute_pre2f_totals_force_confirms_packages():
    # Explicit v2 catalog path: under pytest, tools.pipeline_config is already
    # imported before the env selector could be set, so cfg.ISSUE_CATALOG_PATH
    # would resolve to v1 here. The CLI sets the env before any tools import.
    from tools.artifact_writers import load_issue_catalog
    catalog = load_issue_catalog(bench.ROOT / "tools" / "issue_catalog_kind_v2.json")
    assert catalog.get("version") == "3.1"
    artifact = json.loads(STORED_ARTIFACT.read_text(encoding="utf-8"))

    totals = bench.compute_pre2f_totals(artifact, catalog)
    assert totals["label"] == "pre_2f_all_packages_assumed_confirmed"
    final = totals["final_rehab"]
    assert final["high"] > final["low"] >= 0
    assert final["midpoint"] == pytest.approx((final["low"] + final["high"]) / 2, abs=1)
    assert totals["line_items"], "expected priced line items"
    # The whole point: with packages inferred, force-confirmation must price
    # package-member line items rather than zeroing them via not_run.
    if totals["confirmed_package_count"]:
        assert any(li["package_id"] for li in totals["line_items"]), (
            "packages were inferred but no line item carries a package_id — "
            "the not_run zeroing gate is still biting"
        )
