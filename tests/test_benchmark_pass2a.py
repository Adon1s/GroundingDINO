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
import csv
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
        "catalog_version": "3.2",
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
    # Raised caps apply to every variant identically (checklist output
    # overflowed the production 2000-token cap).
    assert set(config["openai_max_output_tokens"]) == {"2a", "2b", "2c"}
    assert all(int(v) >= 8000 for v in config["openai_max_output_tokens"].values())


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
    assert catalog.get("version") == "3.2"
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


# ---------------------------------------------------------------------------
# Matcher contract: text-only payload, provider-independent validation
# ---------------------------------------------------------------------------

GOLD_ROWS = [
    {"gold_id": "g1", "condition": "Front steps are cracked and patched"},
    {"gold_id": "g3", "condition": "Gutter is detached at the right corner"},
]

CLAIMS = ["Steps are cracked.", "Gutter is detached and actively leaking."]

MATCHER_REPLY = json.dumps({"rows": [
    {"claim_index": 0, "decision": "match", "gold_ids": ["g1"],
     "explanation": "same cracked steps"},
    {"claim_index": 1, "decision": "ambiguous", "gold_ids": ["g3"],
     "explanation": "adds an unsupported active leak"},
]})


def test_matcher_payload_carries_text_only():
    payload = bench.matcher_payload(
        {"photo_key": "photo_001.jpg", "scene": "exterior_front"},
        GOLD_ROWS, CLAIMS)
    assert payload["claims"] == [
        {"claim_index": 0, "text": CLAIMS[0]},
        {"claim_index": 1, "text": CLAIMS[1]},
    ]
    assert [g["gold_id"] for g in payload["gold_conditions"]] == ["g1", "g3"]
    # Nothing that would let the matcher score on catalog identity or lineage.
    blob = json.dumps(payload)
    for leak in ("resolved_item_id", "catalog_item_id", "kind", "priced",
                 "excluded", "kept", "image", "variant"):
        assert leak not in blob


def test_matcher_prompt_pins_the_match_contract():
    prompt = bench.MATCHER_SYSTEM_PROMPT
    assert "never shown the photo" in prompt
    assert "same component" in prompt and "same condition" in prompt
    # The compound-observation rule is the reason 'ambiguous' exists at all.
    assert "'ambiguous', never 'match'" in prompt
    assert "Paraphrase counts" in prompt


def test_validate_matcher_rows_accepts_a_clean_payload():
    rows = bench.validate_matcher_rows(
        [{"claim_index": 1, "decision": "no_match", "gold_ids": [],
          "explanation": "nothing in gold"},
         {"claim_index": 0, "decision": "match", "gold_ids": ["g3", "g1", "g1"],
          "explanation": "same steps, same cracking"}],
        2, ["g1", "g3"])
    assert [r["claim_index"] for r in rows] == [0, 1]   # restored to claim order
    assert rows[0]["gold_ids"] == ["g1", "g3"]          # deduped and sorted


@pytest.mark.parametrize("rows,problem", [
    ([{"claim_index": 0, "decision": "match", "gold_ids": ["g1"],
       "explanation": ""}], "skipped claim_index"),
    ([{"claim_index": 0, "decision": "match", "gold_ids": ["g1"], "explanation": ""},
      {"claim_index": 0, "decision": "no_match", "gold_ids": [], "explanation": ""}],
     "duplicate claim_index"),
    ([{"claim_index": 0, "decision": "match", "gold_ids": ["g9"], "explanation": ""},
      {"claim_index": 1, "decision": "no_match", "gold_ids": [], "explanation": ""}],
     "not on this photo"),
    ([{"claim_index": 0, "decision": "match", "gold_ids": [], "explanation": ""},
      {"claim_index": 1, "decision": "no_match", "gold_ids": [], "explanation": ""}],
     "carries no gold_ids"),
    ([{"claim_index": 0, "decision": "sort_of", "gold_ids": [], "explanation": ""},
      {"claim_index": 1, "decision": "no_match", "gold_ids": [], "explanation": ""}],
     "bad decision"),
    ("not a list", "no 'rows' list"),
])
def test_validate_matcher_rows_rejects_broken_payloads(rows, problem):
    with pytest.raises(ValueError) as excinfo:
        bench.validate_matcher_rows(rows, 2, ["g1", "g3"])
    assert problem in str(excinfo.value)


class _FakeTextClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def analyze_text(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _run_matcher(config, client):
    return asyncio.run(bench.run_matcher_call(
        client, {"api_key": "k"}, config,
        {"photo_key": "photo_001.jpg", "scene": "exterior_front"},
        GOLD_ROWS, CLAIMS))


def test_matcher_openai_gets_a_server_enforced_schema():
    client = _FakeTextClient(MATCHER_REPLY)
    rows = _run_matcher(
        {"matcher": {"provider": "openai", "model": "gpt-5.6-terra",
                     "reasoning_effort": "low"}}, client)
    call = client.calls[0]
    assert call["response_schema_name"] == "pass2a_gold_match_v1"
    assert call["response_json_schema"] == bench._matcher_schema()
    assert call["reasoning_effort"] == "low"
    # Enforced server-side, so it stays out of the prompt.
    assert "additionalProperties" not in call["user_prompt"]
    assert rows[1]["decision"] == "ambiguous"


def test_matcher_lmstudio_gets_the_schema_in_the_prompt_instead():
    client = _FakeTextClient(MATCHER_REPLY)
    rows = _run_matcher(
        {"matcher": {"provider": "lmstudio", "model": "qwen",
                     "url": "http://localhost:1234"}}, client)
    call = client.calls[0]
    # analyze_text drops response_json_schema for lmstudio, so it goes in-band.
    assert "response_json_schema" not in call
    assert "additionalProperties" in call["user_prompt"]
    assert call["url"] == "http://localhost:1234"
    # Identical rows from either provider: the harness is the gate, not the server.
    assert rows == _run_matcher(
        {"matcher": {"provider": "openai", "model": "gpt-5.6-terra"}},
        _FakeTextClient(MATCHER_REPLY))


def test_matcher_retries_once_then_gives_up():
    client = _FakeTextClient("not json at all", MATCHER_REPLY)
    assert _run_matcher({"matcher": {"provider": "openai", "model": "m"}},
                        client)[0]["decision"] == "match"
    assert len(client.calls) == 2

    with pytest.raises(RuntimeError, match="unusable payload"):
        _run_matcher({"matcher": {"provider": "openai", "model": "m"}},
                     _FakeTextClient("garbage", "still garbage"))


def test_match_blinding_is_deterministic_and_uses_both_orders():
    seen = set()
    for photo in (f"photo_{i:03d}.jpg" for i in range(1, 12)):
        mapping = bench.match_blinding("r", "prop", photo, "baseline", "checklist")
        assert mapping == bench.match_blinding(
            "r", "prop", photo, "baseline", "checklist")
        assert set(mapping.values()) == {"baseline", "checklist"}
        seen.add(mapping["A"])
    assert seen == {"baseline", "checklist"}


# ---------------------------------------------------------------------------
# Claim lineage
# ---------------------------------------------------------------------------

def _lineage_record(claims, kept, excluded=(), resolved=None, cut=None):
    kept, resolved = list(kept), (resolved or {})
    cut = len(kept) if cut is None else cut
    return {
        "claims": list(claims),
        "kept": [{"description": t, "kind": "degradation", "issue_id": f"i{n}"}
                 for n, t in enumerate(kept)],
        "excluded": [{"description": t, "reason": "neutral_presence"}
                     for t in excluded],
        "resolved_by_issue": {f"i{n}": resolved[n] for n in resolved if n < cut},
        "total_resolve_count": cut,
        "skipped_by_text": {},
    }


def test_classify_claim_lineage_covers_every_lane():
    record = _lineage_record(
        claims=["resolved one", "declined one", "no candidates",
                "past the cap", "filtered out", "from nowhere"],
        kept=["resolved one", "declined one", "no candidates", "past the cap"],
        excluded=["filtered out"],
        resolved={0: "item_a", 1: None},
        cut=3,
    )
    record["skipped_by_text"] = {"no candidates": "no_candidates"}
    rows = bench.classify_claim_lineage(record)
    assert [r["status"] for r in rows] == [
        "resolved", "retained_unresolved", "resolution_skipped",
        "resolution_not_attempted", "filtered_2c", "unknown_lane",
    ]
    # The lane set is closed: the report and CSV both key off it.
    assert set(bench.LINEAGE_STATUSES) == {r["status"] for r in rows}
    assert rows[0]["resolved_item_id"] == "item_a"
    assert rows[2]["detail"] == "no_candidates"
    assert rows[4]["detail"] == "neutral_presence"


def test_lineage_truncated_claims_are_not_linkage_misses():
    rows = bench.classify_claim_lineage(_lineage_record(
        claims=["a", "b"], kept=["a", "b"], resolved={0: None}, cut=1))
    assert rows[0]["status"] in bench.LINKAGE_MISS_STATUSES
    assert rows[1]["status"] == "resolution_not_attempted"
    assert rows[1]["status"] not in bench.LINKAGE_MISS_STATUSES


def test_lineage_consumes_duplicate_texts_once_per_lane():
    record = _lineage_record(claims=["same text", "same text"],
                             kept=["same text"], excluded=["same text"],
                             resolved={0: "item_a"})
    assert [r["status"] for r in bench.classify_claim_lineage(record)] == [
        "resolved", "filtered_2c"]


def test_lineage_matches_text_despite_whitespace_and_case():
    record = _lineage_record(claims=["Steps  are   Cracked."],
                             kept=["steps are cracked."], resolved={0: "item_a"})
    assert bench.classify_claim_lineage(record)[0]["status"] == "resolved"


# ---------------------------------------------------------------------------
# Human decisions over matcher proposals
# ---------------------------------------------------------------------------

def _matcher_row(decision="match", gold_ids=("g1",)):
    return {"claim_index": 0, "decision": decision,
            "gold_ids": list(gold_ids), "explanation": ""}


def test_auto_match_is_accepted_and_other_verdicts_stay_pending():
    known, sha = {"g1", "g3"}, "sha1"
    assert bench.effective_decision(_matcher_row(), None, known, sha) == (
        "match", ["g1"])
    for proposal in ("no_match", "ambiguous"):
        verdict, _ = bench.effective_decision(
            _matcher_row(proposal, ()), None, known, sha)
        assert verdict == "pending"


def test_human_decision_overrides_the_matcher_in_both_directions():
    known, sha = {"g1", "g3"}, "sha1"
    assert bench.effective_decision(
        _matcher_row("match", ["g1"]),
        {"decision": "unsupported", "gold_ids": [], "gold_sha256": sha},
        known, sha) == ("unsupported", [])
    assert bench.effective_decision(
        _matcher_row("no_match", []),
        {"decision": "match", "gold_ids": ["g3"], "gold_sha256": sha},
        known, sha) == ("match", ["g3"])


def test_decisions_go_stale_when_the_gold_moves_under_them():
    sha, newer = "sha1", "sha2"
    for decision in ({"decision": "match", "gold_ids": ["g9"], "gold_sha256": sha},
                     {"decision": "gold_gap", "gold_ids": [], "gold_sha256": sha}):
        verdict, _ = bench.effective_decision(
            _matcher_row(), decision, {"g1", "g3"}, newer)
        assert verdict == "needs_rereview"
    # Decisions untouched by the edit survive it.
    verdict, _ = bench.effective_decision(
        _matcher_row(), {"decision": "unsupported", "gold_ids": [],
                        "gold_sha256": sha}, {"g1", "g3"}, newer)
    assert verdict == "unsupported"


# ---------------------------------------------------------------------------
# Net-observation scoring and property verdicts
# ---------------------------------------------------------------------------

def _scored_row(variant, decision, gold_ids=(), *, rep=1, index=0,
                status="resolved", critical=False, prop="prop",
                photo="photo_001.jpg", observation="obs"):
    return {
        "row_id": bench.review_row_id(variant, rep, prop, photo, index),
        "property": prop, "photo": photo, "variant": variant, "repeat": rep,
        "claim_index": index, "observation": observation,
        "pipeline_status": status, "resolved_item_id": "",
        "proposed_match": decision, "proposed_gold_ids": "",
        "matcher_explanation": "", "audit": "", "human_decision": "",
        "corrected_gold_ids": "", "critical": "", "reviewer_note": "note",
        "effective_decision": decision, "effective_gold_ids": list(gold_ids),
        "critical_flag": critical,
    }


def _stabilities(base_spread, cand_spread, prop="prop"):
    return {
        "baseline": {"per_property": {prop: {"midpoint_spread_usd": base_spread}},
                     "median_midpoint_spread_usd": base_spread},
        "checklist": {"per_property": {prop: {"midpoint_spread_usd": cand_spread}},
                      "median_midpoint_spread_usd": cand_spread},
    }


def _score(rows, base_spread=1000, cand_spread=1000):
    return bench.score_match_round(
        rows, _stabilities(base_spread, cand_spread), "baseline", "checklist")


def test_many_generated_observations_cover_one_gold_condition_once():
    result = _score([
        _scored_row("baseline", "match", ["g1"], index=0),
        _scored_row("baseline", "match", ["g1"], index=1),
        _scored_row("checklist", "match", ["g1", "g3"], index=0),
    ])
    base = result["properties"]["prop"]["baseline"]
    cand = result["properties"]["prop"]["candidate"]
    assert base["gold_matches_by_rep"]["1"] == 1     # two paraphrases, one gold
    assert cand["gold_matches_by_rep"]["1"] == 2     # one claim, two golds


def test_net_score_is_coverage_minus_confirmed_unsupported():
    result = _score([
        _scored_row("checklist", "match", ["g1"], index=0),
        _scored_row("checklist", "match", ["g3"], index=1),
        _scored_row("checklist", "unsupported", index=2),
        _scored_row("baseline", "match", ["g1"], index=0),
    ])
    cand = result["properties"]["prop"]["candidate"]
    assert cand["gold_matches_by_rep"]["1"] == 2
    assert cand["unsupported_by_rep"]["1"] == 1
    assert cand["net_by_rep"]["1"] == 1
    assert result["properties"]["prop"]["verdict"] == "tied"   # net 1 vs net 1


def test_unsupported_counts_regardless_of_downstream_filtering():
    # A hallucination 2c filtered out still counts against the prompt.
    result = _score([
        _scored_row("checklist", "unsupported", index=0, status="filtered_2c"),
        _scored_row("baseline", "match", ["g1"], index=0),
    ])
    assert result["properties"]["prop"]["candidate"]["unsupported_by_rep"]["1"] == 1
    assert result["properties"]["prop"]["verdict"] == "hurt"


def test_linkage_misses_exclude_claims_the_resolver_never_saw():
    result = _score([
        _scored_row("checklist", "match", ["g1"], index=0,
                    status="retained_unresolved"),
        _scored_row("checklist", "match", ["g3"], index=1,
                    status="resolution_skipped"),
        _scored_row("checklist", "match", ["g1"], index=2,
                    status="resolution_not_attempted"),
        _scored_row("checklist", "match", ["g3"], index=3, status="resolved"),
    ])
    assert result["properties"]["prop"]["candidate"]["catalog_linkage_misses"] == 2


def test_exclude_rows_drop_out_of_scoring_entirely():
    result = _score([
        _scored_row("checklist", "exclude", index=0),
        _scored_row("baseline", "exclude", index=0),
    ])
    assert result["status"] == "final"
    cand = result["properties"]["prop"]["candidate"]
    assert cand["gold_matches_by_rep"]["1"] == 0
    assert cand["unsupported_by_rep"]["1"] == 0


@pytest.mark.parametrize("decision", ["pending", "gold_gap", "needs_rereview"])
def test_unadjudicated_rows_block_the_report(decision):
    result = _score([
        _scored_row("checklist", decision, index=0),
        _scored_row("baseline", "match", ["g1"], index=0),
    ])
    assert result["status"] == "blocked"
    assert result["pending_review"][decision] == 1
    assert result["overall"] is None
    assert result["properties"]["prop"]["verdict"] is None


def test_clean_round_reports_verdicts():
    result = _score([
        _scored_row("checklist", "match", ["g1", "g3"], index=0),
        _scored_row("baseline", "match", ["g1"], index=0),
    ])
    assert result["status"] == "final"
    assert result["overall"] == {"helped": 1, "tied": 0, "hurt": 0}
    assert result["properties"]["prop"]["verdict"] == "helped"


def test_confirmed_critical_hallucination_loses_the_property():
    result = _score([
        # The candidate covers strictly more gold, but one claim is critical.
        _scored_row("checklist", "match", ["g1", "g3"], index=0),
        _scored_row("checklist", "unsupported", index=1, critical=True),
        _scored_row("baseline", "match", ["g1"], index=0),
    ])
    assert result["properties"]["prop"]["verdict"] == "hurt"
    (finding,) = result["critical_findings"]
    assert finding["variant"] == "checklist"
    assert finding["reviewer_note"] == "note"


@pytest.mark.parametrize("base_nets,cand_nets,expected", [
    ([5, 5, 5], [7, 6, 6], "helped"),
    ([5, 5, 5], [4, 4, 3], "hurt"),
    ([5, 6, 4], [4, 5, 6], "tied"),          # equal medians, equal spread below
])
def test_property_verdict_follows_the_median_net(base_nets, cand_nets, expected):
    assert bench.property_verdict(
        base_nets, cand_nets, False, 1000, 1000) == expected


@pytest.mark.parametrize("base_spread,cand_spread,expected", [
    (1000, 750, "helped"),    # exactly 25% lower — inclusive
    (1000, 751, "tied"),
    (1000, 1250, "hurt"),     # exactly 25% higher — inclusive
    (1000, 1249, "tied"),
    (0, 0, "tied"),
    (0, 500, "hurt"),         # candidate introduced spread
    (500, 0, "helped"),
])
def test_spread_tie_breaker_only_runs_on_an_exact_tie(base_spread, cand_spread,
                                                      expected):
    assert bench.property_verdict(
        [5], [5], False, base_spread, cand_spread) == expected
    # A decided median is never overturned by spread.
    assert bench.property_verdict(
        [5], [9], False, base_spread, cand_spread) == "helped"


def test_critical_loss_outranks_every_other_signal():
    assert bench.property_verdict([1], [99], True, 9999, 0) == "hurt"


# ---------------------------------------------------------------------------
# Match stage identity + the review CSV round trip (on a temp benchmark tree)
# ---------------------------------------------------------------------------

ROUND = "baseline_vs_checklist"
PHOTO = "photo_001.jpg"
GOLD_FILE = {"photos": {f"prop/{PHOTO}": [
    {"gold_id": "g1", "condition": "Front steps are cracked"},
    {"gold_id": "g3", "condition": "Gutter is detached"},
]}}
BENCH_CONFIG = {
    "repeats": 1,
    "matcher": {"provider": "openai", "model": "gpt-5.6-terra",
                "reasoning_effort": "low", "url": None},
    "gates": {"supported_audit_rate": 1.0},
}


@pytest.fixture
def bench_tree(tmp_path, monkeypatch):
    bench_dir = tmp_path / "benchmarks" / "pass2a-prompt"
    (bench_dir / "gold").mkdir(parents=True)
    paths = SimpleNamespace(
        dir=bench_dir, runs=bench_dir / "runs",
        gold=bench_dir / "gold" / "reference.json",
        manifest=bench_dir / "manifest.json",
        prompts=bench_dir / "prompts.json",
    )
    monkeypatch.setattr(bench, "BENCH_DIR", bench_dir)
    monkeypatch.setattr(bench, "RUNS_DIR", paths.runs)
    monkeypatch.setattr(bench, "GOLD_PATH", paths.gold)
    monkeypatch.setattr(bench, "MANIFEST_PATH", paths.manifest)
    monkeypatch.setattr(bench, "PROMPTS_PATH", paths.prompts)
    return paths


def _write_variant_photo(runs, variant, claims, resolved=None, cut=None):
    """The per-photo checkpoint + totals load_photo_repeat_records reads."""
    resolved = resolved or {i: f"item_{i}" for i in range(len(claims))}
    cut = len(claims) if cut is None else cut
    stage = f"variant_{variant}"
    prop_dir = runs / stage / "rep1" / "prop"
    (prop_dir / ".photos").mkdir(parents=True, exist_ok=True)
    observations = [{"description": t, "kind": "degradation",
                     "issue_id": f"{variant}-{i}"}
                    for i, t in enumerate(claims)]
    (prop_dir / ".photos" / f"{PHOTO}.json").write_text(json.dumps({
        "image_path": PHOTO,
        "scene_data": {
            "observations_struct": {"observations": [
                {"description": t} for t in claims]},
            "observations": observations,
            "excluded_observations": [],
            "resolved_items": [
                {"issue_id": observations[i]["issue_id"],
                 "description": claims[i], "resolved_item_id": resolved.get(i)}
                for i in range(min(cut, len(claims)))],
            "debug": {"pass_2d_gate": {"total_resolve_count": cut},
                      "pass_2d_per_observation": []},
        },
    }), encoding="utf-8")
    job_dir = prop_dir / f"{stage}_rep1"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "photo_intel.json").write_text("{}", encoding="utf-8")
    (job_dir / "benchmark_totals.json").write_text(
        json.dumps({"line_items": [], "final_rehab": {"midpoint": 0}}),
        encoding="utf-8")


def _seed_round(tree, claims_by_variant, matcher_rows_by_variant):
    from tools.comparison_common import sha256_file
    tree.gold.write_text(json.dumps(GOLD_FILE), encoding="utf-8")
    tree.prompts.write_text(json.dumps(
        {"baseline": {"text": "salience wording"},
         "checklist": {"text": "inventory wording"}}), encoding="utf-8")
    manifest = {
        "images_root": "/img",
        "gold_sha256": sha256_file(tree.gold),
        "properties": {"prop": {"photos": [
            {"photo_key": PHOTO, "image_sha256": "ih",
             "frozen_2a_sha256": "fh", "scene": "exterior_front"}]}},
    }
    tree.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    for variant, claims in claims_by_variant.items():
        _write_variant_photo(tree.runs, variant, claims)
    mapping = bench.match_blinding(ROUND, "prop", PHOTO, "baseline", "checklist")
    artifact = {
        "property_key": "prop", "photo_key": PHOTO, "blinding": mapping,
        "gold_ids": ["g1", "g3"],
        "calls": {f"{letter}|rep1": {"rows": matcher_rows_by_variant[variant],
                                     "matched_at": "now"}
                  for letter, variant in mapping.items()},
    }
    path = tree.runs / f"match_{ROUND}" / "photos" / f"prop__{PHOTO}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return manifest


def _standard_round(tree):
    return _seed_round(
        tree,
        {"baseline": ["Steps are cracked.", "Roof looks new."],
         "checklist": ["Steps are cracked.", "Gutter is detached and leaking."]},
        {"baseline": [
            {"claim_index": 0, "decision": "match", "gold_ids": ["g1"],
             "explanation": "same cracked steps"},
            {"claim_index": 1, "decision": "no_match", "gold_ids": [],
             "explanation": "no gold condition for the roof"}],
         "checklist": [
            {"claim_index": 0, "decision": "match", "gold_ids": ["g1"],
             "explanation": "same cracked steps"},
            {"claim_index": 1, "decision": "ambiguous", "gold_ids": ["g3"],
             "explanation": "adds an unsupported active leak"}]},
    )


def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _edit_csv(path, edits):
    rows = _read_csv(path)
    for row in rows:
        row.update(edits.get(row["row_id"], {}))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=bench.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_review_export_puts_unadjudicated_rows_first(bench_tree):
    manifest = _standard_round(bench_tree)
    rows = _read_csv(bench.review_export(BENCH_CONFIG, manifest, ROUND))
    assert list(rows[0]) == bench.REVIEW_COLUMNS
    assert len(rows) == 4
    # The two rows needing a human come first, auto-matches after.
    assert [r["proposed_match"] for r in rows[:2]] == ["ambiguous", "no_match"]
    assert all(r["proposed_match"] == "match" for r in rows[2:])
    # Lineage travels with the row so a reviewer can see what the pipeline did.
    assert {r["pipeline_status"] for r in rows} == {"resolved"}
    # Auto-matches carry a seeded spot-check flag (rate 1.0 in the test config).
    assert all(r["audit"] == "audit" for r in rows[2:])
    assert all(r["audit"] == "" for r in rows[:2])


def test_review_roundtrip_records_and_clears_decisions(bench_tree):
    manifest = _standard_round(bench_tree)
    path = bench.review_export(BENCH_CONFIG, manifest, ROUND)
    ambiguous = next(r["row_id"] for r in _read_csv(path)
                     if r["proposed_match"] == "ambiguous")
    no_match = next(r["row_id"] for r in _read_csv(path)
                    if r["proposed_match"] == "no_match")
    _edit_csv(path, {
        ambiguous: {"human_decision": "unsupported", "critical": "x",
                    "reviewer_note": "no leak is visible"},
        no_match: {"human_decision": "exclude"},
    })
    bench.review_import(BENCH_CONFIG, manifest, ROUND, str(path))

    decisions = bench.load_decisions(ROUND)
    assert decisions[ambiguous]["decision"] == "unsupported"
    assert decisions[ambiguous]["critical"] is True
    assert decisions[no_match]["decision"] == "exclude"

    # Re-export shows the decisions and the round is no longer blocked.
    rows = _read_csv(bench.review_export(BENCH_CONFIG, manifest, ROUND))
    by_id = {r["row_id"]: r for r in rows}
    assert by_id[ambiguous]["human_decision"] == "unsupported"
    scored = bench.score_match_round(
        bench.build_review_rows(BENCH_CONFIG, manifest, ROUND,
                                json.loads(bench_tree.gold.read_text()),
                                decisions),
        _stabilities(1000, 1000), "baseline", "checklist")
    assert scored["status"] == "final"
    assert scored["properties"]["prop"]["verdict"] == "hurt"   # critical loss

    # Blanking the cell takes the decision back.
    _edit_csv(path, {ambiguous: {"human_decision": "", "critical": ""}})
    bench.review_import(BENCH_CONFIG, manifest, ROUND, str(path))
    assert ambiguous not in bench.load_decisions(ROUND)
    assert no_match in bench.load_decisions(ROUND)


def test_review_import_carries_a_decision_to_the_same_text_elsewhere(bench_tree):
    manifest = _standard_round(bench_tree)
    path = bench.review_export(BENCH_CONFIG, manifest, ROUND)
    # "Steps are cracked." appears under both variants; decide it once.
    target = bench.review_row_id("baseline", 1, "prop", PHOTO, 0)
    twin = bench.review_row_id("checklist", 1, "prop", PHOTO, 0)
    _edit_csv(path, {target: {"human_decision": "unsupported"}})
    bench.review_import(BENCH_CONFIG, manifest, ROUND, str(path))

    decisions = bench.load_decisions(ROUND)
    assert decisions[target]["decision"] == "unsupported"
    assert decisions[twin]["decision"] == "unsupported"
    assert decisions[twin]["propagated_from"] == target
    assert "propagated_from" not in decisions[target]


@pytest.mark.parametrize("edit,problem", [
    ({"human_decision": "probably"}, "not one of"),
    ({"human_decision": "match", "corrected_gold_ids": "g9"}, "are not on"),
])
def test_review_import_rejects_bad_edits(bench_tree, edit, problem):
    manifest = _standard_round(bench_tree)
    path = bench.review_export(BENCH_CONFIG, manifest, ROUND)
    row_id = _read_csv(path)[0]["row_id"]
    _edit_csv(path, {row_id: edit})
    with pytest.raises(SystemExit, match=problem):
        bench.review_import(BENCH_CONFIG, manifest, ROUND, str(path))


def test_review_import_rejects_unknown_row_ids(bench_tree):
    manifest = _standard_round(bench_tree)
    path = bench.review_export(BENCH_CONFIG, manifest, ROUND)
    rows = _read_csv(path)
    rows[0]["row_id"] = "checklist|rep9|prop/photo_404.jpg|c0"
    rows[0]["human_decision"] = "unsupported"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=bench.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit, match="unknown row_id"):
        bench.review_import(BENCH_CONFIG, manifest, ROUND, str(path))


def test_review_row_id_roundtrips():
    row_id = bench.review_row_id("checklist", 3, "prop", PHOTO, 17)
    assert bench.parse_review_row_id(row_id) == (
        "checklist", 3, f"prop/{PHOTO}", 17)


def test_match_fingerprint_rejects_every_material_change(bench_tree):
    manifest = _standard_round(bench_tree)
    for variant in ("baseline", "checklist"):
        bench.guard_fingerprint(bench.stage_dir_for(f"variant_{variant}"),
                                {"prompt_sha256": variant})
    base = bench.compute_match_fingerprint(
        manifest, BENCH_CONFIG, ROUND, "baseline", "checklist")
    # Harness commits must not orphan paid matcher calls.
    assert "git_head" not in base

    stage_dir = bench.stage_dir_for(f"match_{ROUND}")
    bench.guard_fingerprint(stage_dir, base)
    bench.guard_fingerprint(stage_dir, dict(base))       # identical resume is fine

    def _rejects(fingerprint, key):
        assert fingerprint[key] != base[key], f"{key} did not change"
        with pytest.raises(SystemExit, match="fingerprint mismatch"):
            bench.guard_fingerprint(stage_dir, fingerprint)

    _rejects(bench.compute_match_fingerprint(
        manifest, {**BENCH_CONFIG, "matcher": {**BENCH_CONFIG["matcher"],
                                               "model": "qwen"}},
        ROUND, "baseline", "checklist"), "matcher_model")

    bench_tree.prompts.write_text(json.dumps(
        {"baseline": {"text": "salience wording"},
         "checklist": {"text": "REWORDED"}}), encoding="utf-8")
    _rejects(bench.compute_match_fingerprint(
        manifest, BENCH_CONFIG, ROUND, "baseline", "checklist"),
        "variant_prompt_shas")

    bench_tree.gold.write_text(json.dumps({"photos": {f"prop/{PHOTO}": [
        {"gold_id": "g1", "condition": "Front steps are cracked and patched"}]}}),
        encoding="utf-8")
    _rejects(bench.compute_match_fingerprint(
        manifest, BENCH_CONFIG, ROUND, "baseline", "checklist"), "gold_sha256")

    # Regenerating a variant run (new git head, model or cap) invalidates too.
    (bench.stage_dir_for("variant_checklist") / "fingerprint.json").write_text(
        json.dumps({"prompt_sha256": "regenerated"}), encoding="utf-8")
    _rejects(bench.compute_match_fingerprint(
        manifest, BENCH_CONFIG, ROUND, "baseline", "checklist"),
        "variant_run_fingerprint_shas")


def test_repin_gold_acknowledges_an_edit_and_busts_the_match(bench_tree):
    from tools.comparison_common import sha256_file
    manifest = _standard_round(bench_tree)
    before = bench.compute_match_fingerprint(
        manifest, BENCH_CONFIG, ROUND, "baseline", "checklist")

    bench_tree.gold.write_text(json.dumps({"photos": {f"prop/{PHOTO}": [
        {"gold_id": "g1", "condition": "Front steps are cracked"},
        {"gold_id": "g3", "condition": "Gutter is detached"},
        {"gold_id": "g4", "condition": "Downspout is missing"}]}}),
        encoding="utf-8")
    # Until the pin is refreshed, match refuses to run against a moved gold.
    with pytest.raises(SystemExit, match="repin-gold"):
        bench.stage_match(BENCH_CONFIG, manifest, ROUND)

    assert bench.stage_repin_gold(manifest) == sha256_file(bench_tree.gold)
    assert json.loads(bench_tree.manifest.read_text())["gold_sha256"] == \
        manifest["gold_sha256"]
    after = bench.compute_match_fingerprint(
        manifest, BENCH_CONFIG, ROUND, "baseline", "checklist")
    assert after["gold_sha256"] != before["gold_sha256"]
