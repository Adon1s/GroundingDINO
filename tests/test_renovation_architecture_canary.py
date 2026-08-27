import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import run_kind_canary, run_renovation_architecture_canary
from tools.compare_renovation_architecture_cutover import (
    _sol_flips,
    _v4_packages,
    _v5_digest,
    _v5_packages,
    compare_canary,
)
from tools.renovation_architecture.contracts import SHADOW_DEBUG_KEY
from tests.test_renovation_architecture_contracts import (
    _complete_result,
    _scaffold_envelope,
)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _freeze():
    payload = {
        "schema_version": 1,
        "manifest_sha256": "a" * 64,
        "catalog_sha256": "b" * 64,
        "properties": {"prop": {}},
    }
    payload["freeze_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _artifact():
    envelope = _scaffold_envelope(
        state="complete", reason=None, result=_complete_result()
    )
    envelope["provenance"]["architecture_mode"] = "shadow"
    return {
        "property": {"property_key": "prop"},
        "run": {"created_at": "2026-08-16T00:00:00Z"},
        "renovation_estimate_v4": {
            "final_rehab": {"low": 1500, "high": 4500},
            "groups": [],
            "packages": [],
        },
        "analysis_debug": {SHADOW_DEBUG_KEY: envelope},
    }


def _write_run(root, artifact=None):
    path = root / "prop" / "run_1" / "photo_intel_debug.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact or _artifact()), encoding="utf-8")


def _compare(tmp_path, *, reviews=None, freeze=None):
    run_1, run_2 = tmp_path / "run_1", tmp_path / "run_2"
    _write_run(run_1)
    _write_run(run_2)
    return compare_canary(
        run_roots=[run_1, run_2],
        manifest={
            "status": "frozen",
            "properties": [{"property_key": "prop", "stratum": "test"}],
        },
        config={
            "minimum_canary_properties": 1,
            "required_replicates": 2,
            "headline_delta_review_threshold": 0.15,
            "terra_daily_token_ceiling": 2_500_000,
            "terra_condition_token_allocation_policy": (
                "batch_total_evenly_across_reviewed_conditions_v1"
            ),
        },
        freeze=freeze or _freeze(),
        reviews_payload=reviews,
    )


def test_canary_requires_written_review_for_every_scope_and_package_change(tmp_path):
    first = _compare(tmp_path)
    assert not first["release_ready"]
    assert first["review_items"]
    assert {row["category"] for row in first["review_items"]} == {
        "scope", "package"
    }
    assert any(
        row["gate"] == "manual_review" for row in first["gates"]["failures"]
    )

    reviews = {
        "freeze_sha256": first["freeze_sha256"],
        "reviews": {
            row["review_id"]: {
                "decision": "approved",
                "explanation": "Synthetic fixture intentionally differs from v4.",
            }
            for row in first["review_items"]
        },
    }
    approved = _compare(tmp_path, reviews=reviews)
    assert approved["release_ready"]
    assert approved["gates"]["failures"] == []


def test_canary_reports_call_unit_condition_listing_tokens_and_capacity(tmp_path):
    first = _compare(tmp_path)
    reviews = {
        "freeze_sha256": first["freeze_sha256"],
        "reviews": {
            row["review_id"]: {
                "decision": "accepted",
                "explanation": "Reviewed test delta.",
            }
            for row in first["review_items"]
        },
    }
    report = _compare(tmp_path, reviews=reviews)
    usage = report["terra_usage"]
    assert usage["listing_total_tokens"]["median"] == 2150
    assert usage["projected_listings_per_day"]["median"] == 1162
    assert len(usage["call_rows"]) == 4
    assert len(usage["estimate_unit_rows"]) == 4
    assert len(usage["condition_rows"]) == 4
    assert usage["condition_allocated_tokens"]["worst"] == 1200


def test_canary_rejects_invalid_or_failed_envelopes(tmp_path):
    run_1, run_2 = tmp_path / "run_1", tmp_path / "run_2"
    failed = _artifact()
    failed_envelope = failed["analysis_debug"][SHADOW_DEBUG_KEY]
    failed_envelope.update(
        state="failed", reason="provider", error_detail="synthetic", result=None
    )
    _write_run(run_1, failed)
    _write_run(run_2)
    report = compare_canary(
        run_roots=[run_1, run_2],
        manifest={"status": "frozen", "properties": [{"property_key": "prop"}]},
        config={"minimum_canary_properties": 1, "required_replicates": 2},
        freeze=_freeze(),
    )
    assert not report["release_ready"]
    assert any(
        row["gate"] == "complete_artifact" for row in report["gates"]["failures"]
    )


def test_canary_rejects_tampered_freeze(tmp_path):
    freeze = _freeze()
    freeze["catalog_sha256"] = "c" * 64
    report = _compare(tmp_path, freeze=freeze)
    assert any(
        row["gate"] == "freeze_integrity" for row in report["gates"]["failures"]
    )


def test_canary_rejects_review_file_from_another_freeze(tmp_path):
    first = _compare(tmp_path)
    reviews = {
        "freeze_sha256": "wrong",
        "reviews": {
            row["review_id"]: {
                "decision": "approved",
                "explanation": "Wrong freeze must not authorize cutover.",
            }
            for row in first["review_items"]
        },
    }
    report = _compare(tmp_path, reviews=reviews)
    assert any(
        row["gate"] == "review_freeze_mismatch"
        for row in report["gates"]["failures"]
    )


def test_canary_rejects_review_file_with_no_freeze_binding(tmp_path):
    """An unbound review file could be carried over from an earlier canary and
    approve deltas nobody looked at."""
    first = _compare(tmp_path)
    reviews = {
        "reviews": {
            row["review_id"]: {
                "decision": "approved",
                "explanation": "Unbound review must not authorize cutover.",
            }
            for row in first["review_items"]
        },
    }
    report = _compare(tmp_path, reviews=reviews)
    assert not report["release_ready"]
    assert any(
        row["gate"] == "review_freeze_mismatch"
        for row in report["gates"]["failures"]
    )


def test_coordinator_freeze_is_write_once_per_output_root(tmp_path):
    """A rerun under a changed freeze must demand a new output root rather
    than silently mixing inputs across replicas."""
    path = tmp_path / "input_freeze.json"
    freeze = {"schema_version": 1, "freeze_sha256": "a" * 64, "properties": {}}
    run_renovation_architecture_canary._write_or_verify_freeze(path, freeze)
    # Same freeze: idempotent.
    run_renovation_architecture_canary._write_or_verify_freeze(path, freeze)
    drifted = dict(freeze, freeze_sha256="b" * 64)
    with pytest.raises(RuntimeError, match="new output root"):
        run_renovation_architecture_canary._write_or_verify_freeze(path, drifted)


def test_existing_canary_driver_uses_frozen_images_and_metadata(tmp_path, monkeypatch):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"frozen-image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "frozen",
                "properties": [{"property_key": "prop", "stratum": "test"}],
            }
        ),
        encoding="utf-8",
    )
    model_map = tmp_path / "models.json"
    model_map.write_text('{"model_map": {}}', encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "properties": {
                    "prop": {
                        "images": [
                            {
                                "path": str(image),
                                "sha256": hashlib.sha256(b"frozen-image").hexdigest(),
                            }
                        ],
                        "property_metadata": {"sqft": 1234},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def _fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_kind_canary.subprocess, "run", _fake_run)
    result = run_kind_canary.main(
        [
            "--side", "candidate",
            "--build-root", str(tmp_path),
            "--out", str(tmp_path / "out"),
            "--manifest", str(manifest),
            "--model-map", str(model_map),
            "--input-freeze", str(freeze),
            "--skip-preflight",
        ]
    )
    assert result == 0
    assert len(calls) == 1
    command = calls[0][0]
    assert str(image) in command
    metadata_arg = command[command.index("--property-metadata-json") + 1]
    assert json.loads(metadata_arg) == {"sqft": 1234}


def test_existing_canary_driver_rejects_changed_frozen_image(tmp_path, monkeypatch):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"changed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "frozen",
                "properties": [{"property_key": "prop", "stratum": "test"}],
            }
        ),
        encoding="utf-8",
    )
    model_map = tmp_path / "models.json"
    model_map.write_text('{"model_map": {}}', encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "properties": {
                    "prop": {
                        "images": [{"path": str(image), "sha256": "0" * 64}],
                        "property_metadata": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    called = False

    def _unexpected(*args, **kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_kind_canary.subprocess, "run", _unexpected)
    result = run_kind_canary.main(
        [
            "--side", "candidate",
            "--build-root", str(tmp_path),
            "--out", str(tmp_path / "out"),
            "--manifest", str(manifest),
            "--model-map", str(model_map),
            "--input-freeze", str(freeze),
            "--skip-preflight",
        ]
    )
    assert result == 1
    assert not called


# ── Session B (P5): package keying, dollars, 2f verdicts, Sol flips ──────────

def _v4_expansion_artifact():
    """Two expanded bathroom clones: same type, empty estimate_unit_id,
    distinct package_id — the shape a type|unit key silently collapses."""
    return {
        "renovation_estimate_v4": {
            "packages": [
                {
                    "package_id": "bathroom_modernization__bathroom__rs_a",
                    "package_type": "bathroom_modernization",
                    "estimate_unit_id": "",
                    "pricing_tier": "refresh",
                    "cost_low": 4000,
                    "cost_high": 9000,
                    "verification_status": "confirmed",
                    "raw_pass_2f_response": json.dumps(
                        {"evidence_summary": "tile wear visible"}
                    ),
                },
                {
                    "package_id": "bathroom_modernization__bathroom__rs_b",
                    "package_type": "bathroom_modernization",
                    "estimate_unit_id": "",
                    "pricing_tier": "refresh",
                    "cost_low": 5000,
                    "cost_high": 11000,
                    "verification_status": "rejected",
                    "evidence_summary": "fallback summary",
                },
            ],
        }
    }


def test_expanded_v4_packages_survive_package_id_keying():
    rows = _v4_packages(_v4_expansion_artifact())
    assert set(rows) == {
        "bathroom_modernization__bathroom__rs_a",
        "bathroom_modernization__bathroom__rs_b",
    }
    first = rows["bathroom_modernization__bathroom__rs_a"]
    assert (first["low"], first["high"]) == (4000, 9000)
    assert first["verification_status"] == "confirmed"
    assert first["evidence_summary"] == "tile wear visible"
    second = rows["bathroom_modernization__bathroom__rs_b"]
    assert (second["low"], second["high"]) == (5000, 11000)
    assert second["verification_status"] == "rejected"
    assert second["evidence_summary"] == "fallback summary"
    assert "status" not in first  # the hardcoded "applied" label is gone


def test_v5_packages_key_by_candidate_id_with_effective_dollars():
    result = _complete_result()
    application = result["package_applications"][0]
    candidate_id = result["package_candidates"][0]["package_candidate_id"]
    rows = _v5_packages(result)
    assert set(rows) == {candidate_id}
    row = rows[candidate_id]
    assert row["package_candidate_id"] == candidate_id
    assert row["status"] == application["status"]
    assert row["effective_low"] == application["effective_low"]
    assert row["effective_high"] == application["effective_high"]


def test_sol_flip_detector_fires_only_on_identical_children():
    first = _complete_result()
    flipped = copy.deepcopy(first)
    flipped["package_decisions"][0]["decision"] = "reject"
    flips = _sol_flips(first, flipped)
    assert len(flips) == 1
    assert flips[0]["decisions"] == ["approve", "reject"]

    # Same identity but different child work: not a flip.
    changed = copy.deepcopy(flipped)
    children = set(changed["package_candidates"][0]["child_work_item_ids"])
    other = [
        item["work_item_id"] for item in changed["work_items"]
        if item["work_item_id"] not in children
    ]
    changed["package_candidates"][0]["child_work_item_ids"] = [other[0]]
    assert _sol_flips(first, changed) == []


def test_stability_digest_ignores_candidate_id_churn():
    """Candidate ids embed the run-specific estimate id; renaming every
    reference to one candidate must not change the run-to-run digest."""
    first = _complete_result()
    second = copy.deepcopy(first)
    old_id = second["package_candidates"][0]["package_candidate_id"]
    new_id = "pk1_" + "0" * 16
    second["package_candidates"][0]["package_candidate_id"] = new_id
    for row in second["package_decisions"] + second["package_applications"]:
        if row.get("package_candidate_id") == old_id:
            row["package_candidate_id"] = new_id
    for row in second["coverage_ledger"]:
        if row.get("package_id") == old_id:
            row["package_id"] = new_id
    assert _v5_digest(first) == _v5_digest(second)


# ── Session 9: pre-property budget gate and quota belt-and-braces ────────────

def _driver_argv(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"frozen-image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "frozen",
                "properties": [{"property_key": "prop", "stratum": "test"}],
            }
        ),
        encoding="utf-8",
    )
    model_map = tmp_path / "models.json"
    model_map.write_text('{"model_map": {}}', encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "properties": {
                    "prop": {
                        "images": [
                            {
                                "path": str(image),
                                "sha256": hashlib.sha256(
                                    b"frozen-image"
                                ).hexdigest(),
                            }
                        ],
                        "property_metadata": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return [
        "--side", "candidate",
        "--build-root", str(tmp_path),
        "--out", str(tmp_path / "out"),
        "--manifest", str(manifest),
        "--model-map", str(model_map),
        "--input-freeze", str(freeze),
        "--skip-preflight",
    ]


def _seed_budget_env(tmp_path, monkeypatch, *, spent):
    from tools.renovation_architecture.usage_guard import TerraUsageLedger

    budget_root = tmp_path / "budget"
    monkeypatch.setenv("RENOVATION_VLM_BUDGET_GUARD", "1")
    monkeypatch.setenv("RENOVATION_TERRA_USAGE_ROOT", str(budget_root))
    monkeypatch.delenv("RENOVATION_TERRA_DAILY_TOKEN_CEILING", raising=False)
    if spent:
        TerraUsageLedger(budget_root).reserve(
            property_key="seed", source_run_id="seed",
            estimate_unit_id="seed", request_fingerprint="f" * 64,
            tokens=spent,
        )
    return budget_root


def _fake_run_factory(calls):
    def _fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    return _fake_run


def test_driver_gate_stops_cleanly_when_budget_is_low(tmp_path, monkeypatch):
    """Remaining 100k < the default 500k margin: rc=3, nothing spawned,
    nothing recorded as a failure."""
    _seed_budget_env(tmp_path, monkeypatch, spent=2_400_000)
    calls = []
    monkeypatch.setattr(
        run_kind_canary.subprocess, "run", _fake_run_factory(calls)
    )
    assert run_kind_canary.main(_driver_argv(tmp_path)) == 3
    assert calls == []


def test_driver_gate_respects_custom_stop_margin(tmp_path, monkeypatch):
    _seed_budget_env(tmp_path, monkeypatch, spent=2_400_000)
    calls = []
    monkeypatch.setattr(
        run_kind_canary.subprocess, "run", _fake_run_factory(calls)
    )
    assert run_kind_canary.main(
        _driver_argv(tmp_path) + ["--terra-stop-margin", "50000"]
    ) == 0
    assert len(calls) == 1


def test_driver_runs_normally_with_guard_off(tmp_path, monkeypatch):
    """A seeded ledger without the guard env must not gate anything."""
    _seed_budget_env(tmp_path, monkeypatch, spent=2_400_000)
    monkeypatch.delenv("RENOVATION_VLM_BUDGET_GUARD")
    calls = []
    monkeypatch.setattr(
        run_kind_canary.subprocess, "run", _fake_run_factory(calls)
    )
    assert run_kind_canary.main(_driver_argv(tmp_path)) == 0
    assert len(calls) == 1


def test_coordinator_passes_stop_margin_and_guard_env_to_the_driver(
    tmp_path, monkeypatch
):
    """The margin must be settable at the coordinator: once the first day
    writes input_freeze.json, these scripts are hash-locked for the rerun."""
    captured = {}

    def _fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env") or {}
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        run_renovation_architecture_canary.subprocess, "run", _fake_run
    )
    monkeypatch.setattr(
        run_renovation_architecture_canary, "build_freeze",
        lambda **kwargs: {"schema_version": 1, "freeze_sha256": "a" * 64,
                          "properties": {}},
    )
    assert run_renovation_architecture_canary.main([
        "--out", str(tmp_path / "out"),
        "--replicate", "1",
        "--terra-stop-margin", "250000",
        "--skip-preflight",
    ]) == 0
    command = captured["command"]
    assert command[command.index("--terra-stop-margin") + 1] == "250000"
    assert captured["env"]["RENOVATION_VLM_BUDGET_GUARD"] == "1"
    assert captured["env"]["RENOVATION_TERRA_USAGE_ROOT"] == str(
        (tmp_path / "out").resolve()
    )


def test_driver_stops_on_quota_failed_artifact_despite_guard(
    tmp_path, monkeypatch
):
    """A quota-failed v5 envelope written under an active guard means the
    guard env never reached the analyzer subprocess: loud stop, not a
    'completed' property the resume machinery would skip forever."""
    _seed_budget_env(tmp_path, monkeypatch, spent=0)
    argv = _driver_argv(tmp_path)

    def _fake_run(command, **kwargs):
        run_dir = tmp_path / "out" / "candidate" / "prop" / "run_1"
        run_dir.mkdir(parents=True)
        (run_dir / "photo_intel.json").write_text("{}", encoding="utf-8")
        (run_dir / "photo_intel_debug.json").write_text(
            json.dumps(
                {
                    "analysis_debug": {
                        "renovation_estimate_v5": {
                            "state": "failed", "reason": "quota",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_kind_canary.subprocess, "run", _fake_run)
    assert run_kind_canary.main(argv) == 1
