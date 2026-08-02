from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools import rerun_pass_2f_artifact


def _catalog(*, requires_2f: bool = True) -> dict:
    item = {
        "id": "outdated_kitchen_finishes",
        "name": "Outdated kitchen finishes",
        "description": "Dated kitchen finishes.",
        "kind": "upgrade",
        "severity": 3,
        "scope": "replace",
        "trade_bucket": "kitchen_cabinets_counters",
        "scene_groups": ["kitchen"],
        "display_class": "marketability",
        "cost": {
            "mode": "allowance",
            "cost_source": "manual",
            "base_low": 1000,
            "base_high": 2000,
            "per_occurrence_low": 0,
            "per_occurrence_high": 0,
            "cap_low": 1000,
            "cap_high": 2000,
        },
        "estimate": {
            "estimate_tier": "high",
            "strategy": "replace_only",
            "group": "kitchen",
            "stack_behavior": "sum",
            "unit_policy": "per_kitchen",
            "affects_estimate": True,
            "requires_2f_for_estimate": True,
        },
    }
    # A lone opportunity driver on a single photo is suppressed by the emit
    # gate, so pair it with a kitchen package_support to corroborate.
    support = {
        **item,
        "id": "older_flooring_style",
        "name": "Older flooring style",
        "description": "Dated kitchen flooring.",
        "trade_bucket": "flooring",
        "estimate": {**item["estimate"], "group": "flooring"},
    }
    items = [item, support]
    if requires_2f:
        item["package_affinity"] = {
            "kitchen": {
                "package_type": "kitchen_modernization",
                "package_role": "package_driver",
            }
        }
        support["package_affinity"] = {
            "kitchen": {
                "package_type": "kitchen_modernization",
                "package_role": "package_support",
            }
        }
    return {"items": items}


def _artifact(image_path: Path | str, *, include_estimate_issues: bool = True) -> dict:
    issue = {
        "issue_id": "issue_1",
        "photo_id": "photo_1",
        "photo_key": "img.jpg",
        "scene": "kitchen",
        "scene_group": "kitchen",
        "description": "Visible dated kitchen finishes.",
        "label": "upgrade_candidate",
        "location_hint": "",
        "source_lane": "canonical",
        "catalog_item_id": "outdated_kitchen_finishes",
        "catalog_item_kind": "upgrade",
    }
    support_issue = {
        **issue,
        "issue_id": "issue_2",
        "description": "Dated kitchen flooring.",
        "catalog_item_id": "older_flooring_style",
    }
    artifact = {
        "schema_version": "x",
        "property": {"property_key": "redfin_test_001"},
        "property_metadata": {"list_price": 250000},
        "photos": {
            "img.jpg": {
                "photo": {
                    "photo_key": "img.jpg",
                    "image_path": str(image_path),
                },
                "scene": {"id": "kitchen", "group": "kitchen"},
            }
        },
        "issues_flat": [issue, support_issue],
        "renovation_estimate": {
            "version": "renovation_estimate_v3",
            "raw_totals": {"low": 10, "high": 20},
        },
        "renovation_estimate_v4": {"old": True},
        "model_routing": [{"pass": "2a", "model": "old"}, {"pass": "2f", "model": "old2f"}],
        "unrelated": {"keep": "me"},
    }
    if include_estimate_issues:
        artifact["estimate_issues_flat"] = [issue, support_issue]
    return artifact


def _write_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cfg(*, api_key: str = "test-key", gpt_model: str = "gpt-base") -> SimpleNamespace:
    return SimpleNamespace(
        LM_STUDIO_URL="http://localhost:1234",
        LM_STUDIO_MODEL="local-base",
        GPT_MODEL=gpt_model,
        OPENAI_MODEL=gpt_model,
        OPENAI_API_KEY=api_key,
    )


def test_resolves_run_from_file_or_directory(tmp_path: Path):
    artifact_path = tmp_path / "run" / "photo_intel.json"
    _write_artifact(artifact_path, _artifact(tmp_path / "img.jpg"))

    assert rerun_pass_2f_artifact._resolve_artifact_path(artifact_path) == artifact_path
    assert rerun_pass_2f_artifact._resolve_artifact_path(artifact_path.parent) == artifact_path


def test_dry_run_does_not_write_backup_or_mutate_artifact(tmp_path: Path):
    artifact_path = tmp_path / "photo_intel.json"
    original = _artifact(tmp_path / "img.jpg")
    _write_artifact(artifact_path, original)

    result = rerun_pass_2f_artifact.replay_pass_2f_artifact(
        artifact_path,
        _catalog(),
        dry_run=True,
        cfg_module=_cfg(api_key=""),
    )

    assert result["status"] == "dry_run"
    assert result["package_candidate_count"] == 1
    assert not list(tmp_path.glob("*.pre_2f_replay_*.json"))
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == original


def test_successful_replay_creates_backup_and_updates_estimates(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake")
    artifact_path = tmp_path / "photo_intel.json"
    original = _artifact(image_path)
    _write_artifact(artifact_path, original)

    monkeypatch.setattr(
        rerun_pass_2f_artifact,
        "compute_renovation_estimate_v4",
        lambda **kwargs: {
            "final_rehab": {"low": 111, "high": 222},
            "packages": [],
            "pass_2f_trace": {
                "candidate_count": 1,
                "attempted_count": 1,
                "confirmed_count": 1,
                "rejected_count": 0,
                "uncertain_count": 0,
                "no_image_count": 0,
            },
            "pass_2f_review_audit": {
                "consistency_flag_counts": {"debug_flag": 1},
                "items": [{"rationale": "remove me", "consistency_flags": ["remove me"]}],
            },
        },
    )

    result = rerun_pass_2f_artifact.replay_pass_2f_artifact(
        artifact_path,
        _catalog(),
        model_override="gpt-special",
        cfg_module=_cfg(),
        vlm_client=object(),
    )

    assert result["status"] == "updated"
    assert result["attempted_count"] == 1
    backups = list(tmp_path.glob("photo_intel.pre_2f_replay_*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == original

    patched = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert patched["unrelated"] == {"keep": "me"}
    assert patched["renovation_estimate_v4"]["final_rehab"] == {"low": 111, "high": 222}
    assert [r["pass"] for r in patched["model_routing"]] == ["2a", "2f"]
    assert patched["model_routing"][-1]["model"] == "gpt-special"
    assert patched["model_routing"][-1]["source"] == "cli_override"

    assert patched["renovation_estimate"] == original["renovation_estimate"]  # legacy key untouched
    v4_audit = patched["renovation_estimate_v4"]["pass_2f_review_audit"]
    assert "consistency_flag_counts" not in v4_audit
    assert "rationale" not in v4_audit["items"][0]


def test_missing_image_paths_fallback_without_crashing(tmp_path: Path):
    artifact_path = tmp_path / "photo_intel.json"
    _write_artifact(artifact_path, _artifact(tmp_path / "missing.jpg"))

    result = rerun_pass_2f_artifact.replay_pass_2f_artifact(
        artifact_path,
        _catalog(),
        provider="local",
        cfg_module=_cfg(),
        vlm_client=object(),
    )

    assert result["status"] == "updated"
    assert result["package_candidate_count"] == 1
    assert result["attempted_count"] == 0
    assert result["no_image_count"] == 1


def test_no_eligible_candidates_skips_without_writing(tmp_path: Path):
    artifact_path = tmp_path / "photo_intel.json"
    original = _artifact(tmp_path / "img.jpg")
    _write_artifact(artifact_path, original)

    result = rerun_pass_2f_artifact.replay_pass_2f_artifact(
        artifact_path,
        _catalog(requires_2f=False),
        cfg_module=_cfg(),
    )

    assert result["status"] == "skip_no_eligible"
    assert not list(tmp_path.glob("*.pre_2f_replay_*.json"))
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == original


def test_missing_issue_lists_returns_error(tmp_path: Path):
    artifact_path = tmp_path / "photo_intel.json"
    artifact = _artifact(tmp_path / "img.jpg", include_estimate_issues=False)
    artifact["issues_flat"] = []
    _write_artifact(artifact_path, artifact)

    result = rerun_pass_2f_artifact.replay_pass_2f_artifact(
        artifact_path,
        _catalog(),
        cfg_module=_cfg(),
    )

    assert result["status"] == "error"
    assert "no usable estimate_issues_flat or issues_flat" in result["reason"]


def test_env_pass_2f_model_is_honored_when_model_omitted(tmp_path: Path, monkeypatch):
    artifact_path = tmp_path / "photo_intel.json"
    _write_artifact(artifact_path, _artifact(tmp_path / "img.jpg"))
    monkeypatch.setenv("OPENAI_PASS_2F_MODEL", "gpt-env-2f")

    result = rerun_pass_2f_artifact.replay_pass_2f_artifact(
        artifact_path,
        _catalog(),
        dry_run=True,
        cfg_module=_cfg(),
    )

    assert result["status"] == "dry_run"
    assert result["model"] == "gpt-env-2f"
    assert result["model_source"] == "env_override"


def test_cli_model_override_wins_over_env(tmp_path: Path, monkeypatch):
    artifact_path = tmp_path / "photo_intel.json"
    _write_artifact(artifact_path, _artifact(tmp_path / "img.jpg"))
    monkeypatch.setenv("OPENAI_PASS_2F_MODEL", "gpt-env-2f")

    result = rerun_pass_2f_artifact.replay_pass_2f_artifact(
        artifact_path,
        _catalog(),
        model_override="gpt-cli-2f",
        dry_run=True,
        cfg_module=_cfg(),
    )

    assert result["status"] == "dry_run"
    assert result["model"] == "gpt-cli-2f"
    assert result["model_source"] == "cli_override"
