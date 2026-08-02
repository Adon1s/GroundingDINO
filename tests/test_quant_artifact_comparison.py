from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from tools.quant_artifact_comparison import (
    ComparisonError,
    build_judge_prompt,
    compare_2d,
    compare_packages,
    execute,
    load_artifact,
    logical_package_matches,
    main,
    match_issues,
    parse_args,
    parse_judgment,
    resolve_artifact,
    rows_2c,
    status_class,
    validate_pair,
    wilson,
)


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            text = text.lower()
            if "cabinet" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "tile" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=float)


def package(package_id, status="confirmed", **overrides):
    value = {
        "package_id": package_id,
        "package_type": package_id.split("__", 1)[0],
        "room": "kitchen",
        "estimate_unit_id": "kitchen_primary",
        "review_photo_keys": ["photo_001.jpg"],
        "supporting_catalog_item_ids": ["dated_cabinets"],
        "verification_status": status,
        "pricing_tier": "refresh",
        "pricing_profile": "kitchen_refresh",
        "cost_low": 100,
        "cost_high": 200,
        "estimate_eligible": status in {"confirmed", "confirmed_by_rule"},
        "ui_eligible": status in {"confirmed", "confirmed_by_rule"},
        "audit_only": status not in {"confirmed", "confirmed_by_rule"},
    }
    value.update(overrides)
    return value


def artifact_payload(model, image, *, final=None, candidate_rows=None, property_key="redfin_1", pass2f_model="gpt-test"):
    final = final if final is not None else [package("kitchen_modernization__kitchen_primary")]
    candidate_rows = candidate_rows if candidate_rows is not None else list(final)
    issue = {
        "issue_id": f"{model}-issue",
        "description": "The kitchen cabinets are visibly dated.",
        "label": "upgrade_candidate",
        "kind": "upgrade",
    }
    resolution = {
        **issue,
        "resolved_item_id": "dated_kitchen_cabinets",
        "resolved_kind": "upgrade",
        "resolution_path": "llm",
        "candidates": [{"item_id": "dated_kitchen_cabinets"}],
    }
    routes = [
        {"pass": name, "model_family": "qwen", "model": model, "source": "standard_default"}
        for name in ("1a", "2a", "2b", "2c", "2d")
    ] + [{"pass": "2f", "model_family": "gpt5", "model": pass2f_model, "source": "premium_default"}]
    return {
        "schema_version": "photo_intel_v3",
        "normalization_policy_version": "v1",
        "run": {
            "default_local_model": model,
            "analysis_profile": "standard",
            "used_pass_architecture": True,
            "pass_toggles": None,
            "detection_backend": "dinox",
        },
        "property": {"property_key": property_key},
        "property_metadata": {"beds": 3, "baths": 2},
        "model_routing": routes,
        "photos": {
            "photo_001.jpg": {
                "photo": {"image_path": str(image)},
                "processing_time": 10,
                "trace": {"timings_sec": {"2c": 1, "2d": 2}},
                "debug": {
                    "labeled_debug": [issue],
                    "labeled_forward": [issue],
                    "resolved_items": [resolution],
                },
            }
        },
        "renovation_estimate_v4": {
            "packages": final,
            "package_candidates": candidate_rows,
            "pass_2f_trace": {"ran": True, "provider": "premium", "model": pass2f_model},
            "final_rehab": {"low": 100, "high": 200},
        },
    }


def write_run(root: Path, name: str, payload: dict) -> Path:
    run = root / name
    run.mkdir()
    (run / "photo_intel_debug.json").write_text(json.dumps(payload), encoding="utf-8")
    (run / "photo_intel.json").write_text("{}", encoding="utf-8")
    return run


def make_pair(tmp_path, *, base_final=None, cand_final=None, base_candidates=None, cand_candidates=None):
    image = tmp_path / "photo_001.jpg"
    image.write_bytes(b"same-image")
    base = write_run(tmp_path, "q6", artifact_payload("qwen@q6", image, final=base_final, candidate_rows=base_candidates))
    cand = write_run(tmp_path, "q5", artifact_payload("qwen@q5", image, final=cand_final, candidate_rows=cand_candidates))
    return base, cand


def test_resolve_run_and_slim_artifact(tmp_path):
    base, _ = make_pair(tmp_path)
    expected = base / "photo_intel_debug.json"
    assert resolve_artifact(base) == expected
    assert resolve_artifact(base / "photo_intel.json") == expected
    assert resolve_artifact(expected) == expected


def test_status_classes_and_wilson():
    assert status_class("confirmed") == "approved"
    assert status_class("confirmed_by_rule") == "approved"
    assert status_class("rejected") == "denied"
    assert status_class("uncertain") == "indeterminate"
    assert wilson(10, 10)["high"] == 1.0
    assert wilson(0, 0) is None


def test_approved_package_missing_is_hard_difference():
    base = {"renovation_estimate_v4": {"packages": [package("kitchen_modernization__kitchen")], "package_candidates": [package("kitchen_modernization__kitchen")]}}
    cand = {"renovation_estimate_v4": {"packages": [], "package_candidates": [package("kitchen_modernization__kitchen", "rejected")]}}
    result = compare_packages(base, cand)
    assert result["final"]["exact_package_ids_equal"] is False
    assert result["final"]["missing_approved_package_ids"] == ["kitchen_modernization__kitchen"]
    assert result["candidates"]["status_confusion"] == {"confirmed->rejected": 1}


def test_denied_only_status_difference_keeps_final_sets_equal():
    approved = package("kitchen_modernization__kitchen")
    denied = package("kitchen_repair__kitchen", "rejected")
    uncertain = package("kitchen_repair__kitchen", "uncertain")
    base = {"renovation_estimate_v4": {"packages": [approved], "package_candidates": [approved, denied]}}
    cand = {"renovation_estimate_v4": {"packages": [approved], "package_candidates": [approved, uncertain]}}
    result = compare_packages(base, cand)
    assert result["final"]["exact_package_ids_equal"] is True
    assert result["candidates"]["exact_statuses_equal"] is False


def test_logical_match_does_not_change_strict_identity():
    base = package("kitchen_modernization__kitchen_primary")
    cand = package("kitchen_modernization__kitchen_1")
    result = compare_packages(
        {"renovation_estimate_v4": {"packages": [base], "package_candidates": [base]}},
        {"renovation_estimate_v4": {"packages": [cand], "package_candidates": [cand]}},
    )
    assert result["final"]["exact_package_ids_equal"] is False
    assert result["candidates"]["logical_matches"][0]["reason"] == "unit"


def test_semantic_matching_is_one_to_one():
    base = [
        {"description": "Dated kitchen cabinets", "normalized": "dated kitchen cabinets"},
        {"description": "Cracked bathroom tile", "normalized": "cracked bathroom tile"},
    ]
    cand = [
        {"description": "Kitchen cabinetry looks old", "normalized": "kitchen cabinetry looks old"},
        {"description": "Bathroom tile is cracked", "normalized": "bathroom tile is cracked"},
    ]
    matches, un_b, un_c = match_issues(base, cand, FakeEncoder())
    assert {(row["baseline_index"], row["candidate_index"]) for row in matches} == {(0, 0), (1, 1)}
    assert un_b == [] and un_c == []


def test_2d_compares_aligned_issue_rows():
    base_issue = {"issue_id": "b", "description": "Dated cabinets", "normalized": "dated cabinets"}
    cand_issue = {"issue_id": "c", "description": "Old cabinets", "normalized": "old cabinets"}
    base_photo = {"debug": {"resolved_items": [{"issue_id": "b", "resolved_item_id": "dated_cabinets", "resolved_kind": "upgrade", "resolution_path": "llm", "candidates": [{"item_id": "dated_cabinets"}]}]}}
    cand_photo = {"debug": {"resolved_items": [{"issue_id": "c", "resolved_item_id": None, "resolved_kind": "upgrade", "resolution_path": "llm", "candidates": [{"item_id": "dated_cabinets"}]}]}}
    result = compare_2d([base_issue], [cand_issue], [{"baseline_index": 0, "candidate_index": 0}], base_photo, cand_photo)
    assert result["baseline_nonnull_retention"] == 0.0
    assert result["null_transition_matrix"] == {"nonnull->null": 1}


def test_parse_judgment_canonicalizes_blinded_rows():
    display = {"A_001": ("candidate", 3), "B_001": ("baseline", 2), "A_002": ("candidate", 4)}
    parsed = {
        "equivalent_pairs": [{"A_row_id": "A_001", "B_row_id": "B_001"}],
        "A_supported_ids": ["A_002"],
        "B_supported_ids": [],
        "A_hallucination_ids": [],
        "B_hallucination_ids": [],
        "missed_by_both": [],
    }
    result = parse_judgment(parsed, display)
    assert result["matches"][0]["baseline_index"] == 2
    assert result["matches"][0]["candidate_index"] == 3
    assert result["candidate_supported_indices"] == [4]


def test_validation_rejects_same_model_and_different_2f(tmp_path):
    base, cand = make_pair(tmp_path)
    base_loaded, cand_loaded = load_artifact(base), load_artifact(cand)
    validate_pair(base_loaded, cand_loaded)
    cand_loaded["local_model"] = base_loaded["local_model"]
    with pytest.raises(ComparisonError, match="identical"):
        validate_pair(base_loaded, cand_loaded)
    cand_loaded["local_model"] = "qwen@q5"
    cand_loaded["pass2f"] = ("premium", "different")
    with pytest.raises(ComparisonError, match="Pass 2f"):
        validate_pair(base_loaded, cand_loaded)


def test_validation_rejects_nonterminal_status(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    payload = artifact_payload("qwen@q6", image, candidate_rows=[package("x__kitchen", "not_run")])
    run = write_run(tmp_path, "bad", payload)
    loaded = load_artifact(run)
    with pytest.raises(ComparisonError, match="non-terminal"):
        validate_pair(loaded, {**loaded, "local_model": "qwen@q5"})


def test_mocked_end_to_end_writes_json_markdown_and_checkpoint(tmp_path, monkeypatch):
    base, cand = make_pair(tmp_path)
    output = tmp_path / "report.json"
    class ExactOnlyEncoder:
        def encode(self, texts):
            raise AssertionError("identical text should not need embeddings")
    monkeypatch.setattr("tools.quant_artifact_comparison.SentenceEncoder", ExactOnlyEncoder)
    args = parse_args(["--baseline-run", str(base), "--candidate-run", str(cand), "--output", str(output)])
    report, code = asyncio.run(execute(args))
    assert code == 0
    assert report["verdicts"]["overall"] == "PASS"
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    assert output.with_suffix(".checkpoint.json").is_file()


def test_cli_returns_two_for_missing_approved_package(tmp_path, monkeypatch):
    approved = package("kitchen_modernization__kitchen_primary")
    rejected = package("kitchen_modernization__kitchen_primary", "rejected")
    base, cand = make_pair(tmp_path, base_final=[approved], cand_final=[], base_candidates=[approved], cand_candidates=[rejected])
    output = tmp_path / "failed.json"
    class ExactOnlyEncoder:
        def encode(self, texts):
            raise AssertionError
    monkeypatch.setattr("tools.quant_artifact_comparison.SentenceEncoder", ExactOnlyEncoder)
    assert main(["--baseline-run", str(base), "--candidate-run", str(cand), "--output", str(output)]) == 2



def test_forwarded_status_comes_only_from_labeled_forward():
    issue = {"issue_id": "x", "description": "Dated cabinets", "label": "upgrade_candidate"}
    rows = rows_2c({"debug": {"labeled_debug": [issue], "labeled_forward": []}})
    assert rows[0]["forwarded"] is False


def test_validation_rejects_image_hash_and_configuration_changes(tmp_path):
    base, cand = make_pair(tmp_path)
    base_loaded, cand_loaded = load_artifact(base), load_artifact(cand)
    cand_loaded["image_hashes"]["photo_001.jpg"] = "different"
    with pytest.raises(ComparisonError, match="photo bytes differ"):
        validate_pair(base_loaded, cand_loaded)
    cand_loaded = load_artifact(cand)
    cand_loaded["payload"]["run"]["analysis_profile"] = "different"
    with pytest.raises(ComparisonError, match="analysis_profile"):
        validate_pair(base_loaded, cand_loaded)


def test_manifest_validate_only_is_deterministic(tmp_path):
    base, cand = make_pair(tmp_path)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"schema_version": 1, "pairs": [{"property_id": "redfin_1", "baseline_run": "q6", "candidate_run": "q5"}]}), encoding="utf-8")
    output = tmp_path / "validation.json"
    args = parse_args(["--manifest", str(manifest), "--output", str(output), "--validate-only"])
    first, first_code = asyncio.run(execute(args))
    first_bytes = output.read_bytes()
    second, second_code = asyncio.run(execute(args))
    assert first_code == second_code == 0
    assert first == second
    assert output.read_bytes() == first_bytes
    assert not output.with_suffix(".checkpoint.json").exists()


def test_blinded_judge_prompt_is_stable_and_hides_model_identity():
    base = [{"description": "Dated cabinets", "label": "upgrade_candidate", "forwarded": True}]
    cand = [{"description": "Old cabinetry", "label": "upgrade_candidate", "forwarded": True}]
    prompt_1, display_1 = build_judge_prompt("fingerprint", "property", "photo", base, cand, [0], [0])
    prompt_2, display_2 = build_judge_prompt("fingerprint", "property", "photo", base, cand, [0], [0])
    assert prompt_1 == prompt_2
    assert display_1 == display_2
    assert "Q5" not in prompt_1 and "Q6" not in prompt_1 and "quant" not in prompt_1.lower()