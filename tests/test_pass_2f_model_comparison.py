from __future__ import annotations

import asyncio
from dataclasses import replace
import importlib
import json
import re
from pathlib import Path

import pytest

from tools import pass_2f_model_comparison as comparison
from tools import pipeline_config
from tools.pass_2f_comparison_config import (
    DEFAULT_REQUIRED_LOCAL_MODEL,
    load_allowed_dotenv,
    load_pass_2f_comparison_profile,
    with_cli_overrides,
)
from tools.rehab_packages import prepare_pass_2f_cases
from tools.scene_classifier_passes import (
    PASS_2F_PROMPT_SHA256,
    PASS_2F_PROMPT_VERSION,
    Pass2fInvalidResponseError,
    PassExecutionError,
    run_pass_2f,
)


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_pass_2f_comparison_profile("terra_vs_sol_2f", ROOT)


def _source_artifact() -> dict:
    return {
        "run": {"default_local_model": DEFAULT_REQUIRED_LOCAL_MODEL},
        "model_routing": [
            {
                "pass": "2a",
                "model_family": "qwen",
                "model": DEFAULT_REQUIRED_LOCAL_MODEL,
                "source": "standard_default",
            },
            {
                "pass": "2f",
                "model_family": "gpt5",
                "model": "ignored-source-result",
            },
        ],
    }


def _evidence(photo_key: str = "img.jpg") -> list[dict]:
    return [{
        "catalog_item_id": "outdated_kitchen_finishes",
        "issue_ids": ["issue_1"],
        "photo_keys": [photo_key],
        "observations": ["dated cabinets"],
        "issue_refs": [{
            "issue_id": "issue_1",
            "photo_key": photo_key,
            "observation": "dated cabinets",
        }],
    }]


def _package(
    package_id: str,
    *,
    room: str = "kitchen",
    category: str = "modernization",
    photo_key: str = "img.jpg",
    evidence: list[dict] | None = None,
) -> dict:
    return {
        "package_id": package_id,
        "package_type": "kitchen_modernization",
        "package_category": category,
        "room": room,
        "review_photo_keys": [photo_key],
        "supporting_issue_ids": ["issue_1"],
        "evidence_items": _evidence(photo_key) if evidence is None else evidence,
    }


class _ResponseVLM:
    def __init__(self, response: str | Exception):
        self.response = response
        self.calls = []

    async def analyze_images(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _valid_response(status: str = "confirmed") -> str:
    return json.dumps({
        "verification_status": status,
        "confirmed_issue_ids": ["issue_1"] if status == "confirmed" else [],
        "rejected_issue_ids": ["issue_1"] if status == "rejected" else [],
        "evidence_summary": "Visible package evidence.",
    })


def test_profile_loading_overrides_and_redaction(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")
    config = with_cli_overrides(
        _config(),
        terra_model="gpt-terra-one-off",
        sol_model="gpt-sol-one-off",
        max_images_per_package=2,
    )
    redacted = json.dumps(config.redacted_dict())
    assert config.terra.model == "gpt-terra-one-off"
    assert config.sol.model == "gpt-sol-one-off"
    assert config.qwen.provider == "lmstudio"
    assert config.qwen.model == DEFAULT_REQUIRED_LOCAL_MODEL
    assert config.run.max_images_per_package == 2
    assert config.run.include_qwen is False
    assert "super-secret" not in redacted
    assert "OPENAI_API_KEY" in redacted


@pytest.mark.parametrize("model", ["gpt-5.5", "gpt5.5", "gpt_5.5"])
def test_profile_rejects_gpt_5_5_override(model):
    with pytest.raises(ValueError, match="GPT-5.5"):
        with_cli_overrides(_config(), terra_model=model)



def test_profile_requires_fixed_source_and_cli_only_executive():
    config = _config()
    with pytest.raises(ValueError, match="required_local_model"):
        replace(
            config,
            source=replace(config.source, required_local_model="other"),
        ).validate()
    with pytest.raises(ValueError, match="upstream_routing"):
        replace(
            config,
            source=replace(
                config.source,
                require_local_upstream_routing=False,
            ),
        ).validate()
    with pytest.raises(ValueError, match="requested explicitly"):
        replace(
            config,
            executive_review=replace(config.executive_review, enabled=True),
        ).validate()
    with pytest.raises(ValueError, match="qwen.provider"):
        replace(
            config,
            qwen=replace(config.qwen, provider="openai"),
        ).validate()
    with pytest.raises(ValueError, match="source.required_local_model"):
        replace(
            config,
            qwen=replace(config.qwen, model="another-local-model"),
        ).validate()


def test_dotenv_loader_ignores_model_selection_variables(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "OPENAI_API_KEY=allowed\nGPT_MODEL=gpt-5.5\n"
        "OPENAI_PASS_2F_MODEL=gpt-5.5\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GPT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_PASS_2F_MODEL", raising=False)
    load_allowed_dotenv(dotenv)
    assert comparison.credentials_available()
    assert "GPT_MODEL" not in comparison.os.environ
    assert "OPENAI_PASS_2F_MODEL" not in comparison.os.environ



def test_local_endpoint_is_required_only_when_qwen_is_enabled(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LM_STUDIO_URL", raising=False)
    monkeypatch.setattr(pipeline_config, "LM_STUDIO_URL", "")
    _config().validate(require_credentials=True)
    with pytest.raises(ValueError, match="LM Studio endpoint"):
        _qwen_config().validate(require_credentials=True)

def test_pipeline_local_model_fallback_is_qwen(monkeypatch):
    monkeypatch.delenv("LM_STUDIO_MODEL", raising=False)
    reloaded = importlib.reload(pipeline_config)
    assert reloaded.LM_STUDIO_MODEL == DEFAULT_REQUIRED_LOCAL_MODEL


def test_source_validation_requires_local_qwen_and_ignores_source_2f():
    comparison.validate_source_artifact(_source_artifact(), _config())
    bad = _source_artifact()
    bad["model_routing"][0]["model_family"] = "gpt5"
    bad["model_routing"][0]["model"] = "gpt-5.6-sol"
    with pytest.raises(ValueError, match="upstream pass 2a"):
        comparison.validate_source_artifact(bad, _config())


def test_legacy_source_uses_verified_routing_when_run_model_is_missing():
    legacy = _source_artifact()
    legacy["run"].pop("default_local_model")
    comparison.validate_source_artifact(legacy, _config())

    no_routes = _source_artifact()
    no_routes["run"].pop("default_local_model")
    no_routes["model_routing"] = [no_routes["model_routing"][-1]]
    with pytest.raises(ValueError, match="no verifiable upstream"):
        comparison.validate_source_artifact(no_routes, _config())

    mismatch = _source_artifact()
    mismatch["run"]["default_local_model"] = "wrong-local-model"
    with pytest.raises(ValueError, match="default_local_model"):
        comparison.validate_source_artifact(mismatch, _config())


@pytest.mark.parametrize(
    ("terra", "sol", "expected"),
    [
        ("confirmed", "confirmed", "both_approved"),
        ("rejected", "uncertain", "neither_approved"),
        ("confirmed", "rejected", "terra_only_approved"),
        ("uncertain", "confirmed", "sol_only_approved"),
    ],
)
def test_approval_buckets(terra, sol, expected):
    assert comparison.classify_approval_bucket(terra, sol) == expected


def test_prepare_cases_freezes_vlm_rule_and_not_evaluated_inputs(tmp_path):
    image = tmp_path / "img.jpg"
    image.write_bytes(b"image")
    packages = [
        _package("eligible"),
        _package("turnover", category="turnover"),
        _package("unsupported", room="garage"),
        _package("no_image", photo_key="missing.jpg"),
        _package("no_ids", evidence=[{
            "issue_ids": [],
            "photo_keys": ["img.jpg"],
            "observations": ["unknown"],
        }]),
    ]
    cases, trace = prepare_pass_2f_cases(
        packages,
        photo_key_to_path={"img.jpg": image},
        max_images=3,
    )
    by_id = {case["package_id"]: case for case in cases}
    assert by_id["eligible"]["evaluation_kind"] == "vlm_eligible"
    assert by_id["eligible"]["prepared_input"]["review_image_paths"] == [
        str(image)
    ]
    assert by_id["turnover"]["evaluation_kind"] == "rule_confirmed"
    assert by_id["unsupported"]["not_evaluated_reason"] == "unsupported_room"
    assert by_id["no_image"]["not_evaluated_reason"] == "no_review_images"
    assert by_id["no_ids"]["not_evaluated_reason"] == "no_reviewed_issue_ids"
    assert trace["vlm_eligible_count"] == 1
    assert trace["rule_confirmed_count"] == 1
    assert trace["not_evaluated_count"] == 3
    assert all(
        case["prepared_input"]["prompt_template_version"]
        == PASS_2F_PROMPT_VERSION
        for case in cases
    )
    assert all(
        case["prepared_input"]["prompt_template_sha256"]
        == PASS_2F_PROMPT_SHA256
        for case in cases
    )


def test_run_pass_2f_strict_preserves_valid_uncertain(tmp_path):
    image = tmp_path / "img.jpg"
    image.write_bytes(b"image")
    client = _ResponseVLM(_valid_response("uncertain"))
    result = asyncio.run(run_pass_2f(
        [image],
        client,
        {"model": "test", "provider": "openai"},
        room="kitchen",
        package_id="p1",
        package_type="kitchen_modernization",
        evidence_items=_evidence(),
    ))
    assert result.verification_status == "uncertain"
    assert result.parsed_response["verification_status"] == "uncertain"
    assert result.raw_response == _valid_response("uncertain")


def test_run_pass_2f_strict_raises_parse_and_provider_errors(tmp_path):
    image = tmp_path / "img.jpg"
    image.write_bytes(b"image")
    invalid = _ResponseVLM('{"verification_status":"confirmed"}')
    with pytest.raises(Pass2fInvalidResponseError):
        asyncio.run(run_pass_2f(
            [image],
            invalid,
            {"model": "test", "provider": "openai"},
            room="kitchen",
            package_id="p1",
            package_type="kitchen_modernization",
            evidence_items=_evidence(),
        ))
    # Provider errors surface as PassExecutionError (was: a bare TimeoutError
    # only when strict, and an "uncertain" verdict otherwise).
    provider = _ResponseVLM(TimeoutError("timed out"))
    with pytest.raises(PassExecutionError) as excinfo:
        asyncio.run(run_pass_2f(
            [image],
            provider,
            {"model": "test", "provider": "openai"},
            room="kitchen",
            package_id="p1",
            package_type="kitchen_modernization",
            evidence_items=_evidence(),
        ))
    assert excinfo.value.pass_key == "2f"
    assert excinfo.value.code == "TimeoutError"



def _record(
    package_id: str,
    image: Path,
    *,
    kind: str = "vlm_eligible",
    reason: str | None = None,
) -> dict:
    bucket = (
        "rule_confirmed"
        if kind == "rule_confirmed"
        else "not_evaluated"
        if kind == "not_evaluated"
        else None
    )
    return {
        "package_id": package_id,
        "package_type": "kitchen_modernization",
        "package_label": "Kitchen modernization",
        "room": "kitchen",
        "evaluation_kind": kind,
        "source_package": {"package_id": package_id},
        "prepared_input": {
            "evidence_items": _evidence(),
            "review_photo_keys": ["img.jpg"],
            "review_image_paths": [str(image)] if kind == "vlm_eligible" else [],
            "reviewed_issue_ids": ["issue_1"],
            "review_image_sha256": (
                [comparison._sha256_file(image)]
                if kind == "vlm_eligible"
                else []
            ),
            "prompt_template_version": PASS_2F_PROMPT_VERSION,
            "prompt_template_sha256": PASS_2F_PROMPT_SHA256,
            "max_images_per_package": 3,
        },
        "prepared_input_sha256": "case-" + package_id,
        "rule_confirmation": (
            {
                "verification_status": "confirmed_by_rule",
                "confirmed_issue_ids": ["issue_1"],
            }
            if kind == "rule_confirmed"
            else None
        ),
        "terra": None,
        "sol": None,
        "qwen": None,
        "comparison": {"bucket": bucket, "is_focus": False},
        "executive_review": None,
        "not_evaluated_reason": reason,
    }


def _context(config, records):
    return {
        "config": config,
        "config_fingerprint": "fingerprint-test",
        "source": {
            "artifact_path": "source.json",
            "artifact_sha256": "artifact-hash",
            "catalog_path": "catalog.json",
            "catalog_sha256": "catalog-hash",
            "property_id": "property-test",
            "local_model": DEFAULT_REQUIRED_LOCAL_MODEL,
            "model_routing": [],
        },
        "package_records": records,
        "preparation_trace": {
            "candidate_count": len(records),
            "vlm_eligible_count": sum(
                record["evaluation_kind"] == "vlm_eligible"
                for record in records
            ),
            "rule_confirmed_count": sum(
                record["evaluation_kind"] == "rule_confirmed"
                for record in records
            ),
            "not_evaluated_count": sum(
                record["evaluation_kind"] == "not_evaluated"
                for record in records
            ),
            "not_evaluated_reasons": {},
        },
    }


class _ComparisonVLM:
    def __init__(self, statuses, *, fail=None, executive_response=None):
        self.statuses = statuses
        self.fail = fail
        self.executive_response = executive_response
        self.calls = []
        self.usage_stats = {
            "calls": 0,
            "metered_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def analyze_images(self, **kwargs):
        self.calls.append(kwargs)
        model = kwargs["model"]
        if kwargs.get("analysis_pass") == "Pass 2f executive review":
            self._meter()
            if isinstance(self.executive_response, Exception):
                raise self.executive_response
            if self.executive_response is not None:
                return self.executive_response

            return json.dumps({
                "operational_decision": "needs_manual_review",
                "reasoning": "The blinded decisions differ.",
                "supported_issue_ids": ["issue_1"],
                "rejected_issue_ids": [],
            })
        match = re.search(
            r"- package_id: ([^\n]+)",
            kwargs.get("user_prompt") or "",
        )
        package_id = match.group(1).strip() if match else ""
        if self.fail == (package_id, model):
            raise TimeoutError("provider timeout")
        self._meter()
        status = self.statuses[(package_id, model)]
        return _valid_response(status)

    def _meter(self):
        self.usage_stats["calls"] += 1
        self.usage_stats["metered_calls"] += 1
        self.usage_stats["input_tokens"] += 10
        self.usage_stats["output_tokens"] += 4
        self.usage_stats["total_tokens"] += 14


def test_mocked_end_to_end_retains_all_packages_and_focuses_asymmetric(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    image = tmp_path / "img.jpg"
    image.write_bytes(b"image")
    config = _config()
    records = [
        _record("both", image),
        _record("neither", image),
        _record("terra_only", image),
        _record("sol_only", image),
        _record("rule", image, kind="rule_confirmed"),
        _record(
            "skipped",
            image,
            kind="not_evaluated",
            reason="unsupported_room",
        ),
    ]
    statuses = {
        ("both", config.terra.model): "confirmed",
        ("both", config.sol.model): "confirmed",
        ("neither", config.terra.model): "rejected",
        ("neither", config.sol.model): "uncertain",
        ("terra_only", config.terra.model): "confirmed",
        ("terra_only", config.sol.model): "rejected",
        ("sol_only", config.terra.model): "uncertain",
        ("sol_only", config.sol.model): "confirmed",
    }
    client = _ComparisonVLM(statuses)
    output = tmp_path / "comparison.json"
    report, success = asyncio.run(comparison.execute_comparison(
        _context(config, records),
        output,
        vlm_client=client,
    ))
    assert success
    assert len(report["packages"]) == 6
    assert report["aggregate"] == {
        "package_candidate_count": 6,
        "vlm_eligible_count": 4,
        "vlm_evaluated_count": 4,
        "qwen_evaluated_count": 0,
        "qwen_approved_count": 0,
        "both_approved": 1,
        "neither_approved": 1,
        "terra_only_approved": 1,
        "sol_only_approved": 1,
        "rule_confirmed": 1,
        "not_evaluated": 1,
        "execution_failed": 0,
        "focus_package_count": 2,
    }
    assert len(client.calls) == 8
    terra_calls = client.calls[:4]
    sol_calls = client.calls[4:]
    for terra_call, sol_call in zip(terra_calls, sol_calls):
        assert terra_call["image_paths"] == sol_call["image_paths"]
        assert terra_call["system_prompt"] == sol_call["system_prompt"]
        assert terra_call["user_prompt"] == sol_call["user_prompt"]
        assert terra_call["analysis_pass"] == "Pass 2f (package verification)"
        assert sol_call["analysis_pass"] == "Pass 2f (package verification)"
    comparison._print_summary(report, output)
    terminal = capsys.readouterr().out
    assert "terra_only" in terminal
    assert "sol_only" in terminal
    focus_section = terminal.split("focus_packages:", 1)[1]
    assert "both (" not in focus_section
    assert "neither (" not in focus_section
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert len(saved["packages"]) == 6
    assert saved["packages"][0]["terra"]["raw_response"]
    assert saved["packages"][0]["terra"]["parsed_response"]
    assert saved["phase_stats"]["terra"]["usage"]["total_tokens"] == 56
    assert "test-key" not in output.read_text(encoding="utf-8")



class _ConcurrentGate:
    def __init__(self, expected_models):
        self.expected_models = set(expected_models)
        self.started = set()
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()


class _GatedComparisonVLM(_ComparisonVLM):
    def __init__(self, statuses, gate):
        super().__init__(statuses)
        self.gate = gate

    async def analyze_images(self, **kwargs):
        self.calls.append(kwargs)
        model = kwargs["model"]
        self.gate.started.add(model)
        if self.gate.started == self.gate.expected_models:
            self.gate.all_started.set()
        await self.gate.release.wait()
        match = re.search(
            r"- package_id: ([^\n]+)",
            kwargs.get("user_prompt") or "",
        )
        package_id = match.group(1).strip() if match else ""
        self._meter()
        return _valid_response(self.statuses[(package_id, model)])


def _qwen_config():
    config = _config()
    return replace(
        config,
        run=replace(config.run, include_qwen=True),
    )


def test_default_execution_creates_only_two_cloud_clients(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    created = []

    def factory(*, timeout):
        client = _ComparisonVLM({})
        created.append((timeout, client))
        return client

    monkeypatch.setattr(comparison, "create_vlm_client", factory)
    monkeypatch.setattr(
        comparison,
        "qwen_vlm_config",
        lambda config: pytest.fail("local Qwen config resolved while disabled"),
    )
    report, success = asyncio.run(comparison.execute_comparison(
        _context(_config(), []),
        tmp_path / "comparison.json",
    ))
    assert success
    assert len(created) == 2
    assert report["execution"]["qwen_requested"] is False
    assert report["phase_stats"]["qwen"]["attempted_count"] == 0


def test_include_qwen_runs_three_independent_streams_and_keeps_pairwise_bucket(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = _qwen_config()
    image = tmp_path / "img.jpg"
    image.write_bytes(b"image")
    statuses = {
        ("p1", config.terra.model): "confirmed",
        ("p1", config.sol.model): "rejected",
        ("p1", config.qwen.model): "confirmed",
    }
    gate = _ConcurrentGate({
        config.terra.model,
        config.sol.model,
        config.qwen.model,
    })
    clients = {
        stage: _GatedComparisonVLM(statuses, gate)
        for stage in ("terra", "sol", "qwen")
    }
    output = tmp_path / "comparison.json"

    async def scenario():
        task = asyncio.create_task(comparison.execute_comparison(
            _context(config, [_record("p1", image)]),
            output,
            vlm_clients=clients,
        ))
        await asyncio.wait_for(gate.all_started.wait(), timeout=1)
        assert gate.started == gate.expected_models
        gate.release.set()
        return await task

    report, success = asyncio.run(scenario())
    assert success
    package = report["packages"][0]
    assert package["comparison"] == {
        "bucket": "terra_only_approved",
        "is_focus": True,
    }
    assert package["qwen"]["verification_status"] == "confirmed"
    assert report["aggregate"]["qwen_evaluated_count"] == 1
    assert report["aggregate"]["qwen_approved_count"] == 1
    calls = [clients[stage].calls[0] for stage in ("terra", "sol", "qwen")]
    assert all(call["image_paths"] == calls[0]["image_paths"] for call in calls)
    assert all(call["system_prompt"] == calls[0]["system_prompt"] for call in calls)
    assert all(call["user_prompt"] == calls[0]["user_prompt"] for call in calls)
    assert all(
        call["analysis_pass"] == "Pass 2f (package verification)"
        for call in calls
    )
    comparison._print_summary(report, output)
    assert "Qwen=confirmed" in capsys.readouterr().out


def test_qwen_failure_is_incomplete_and_resumes_only_missing_result(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = _qwen_config()
    image = tmp_path / "img.jpg"
    image.write_bytes(b"image")
    statuses = {
        ("p1", config.terra.model): "confirmed",
        ("p1", config.sol.model): "rejected",
        ("p1", config.qwen.model): "uncertain",
    }
    output = tmp_path / "comparison.json"
    first_clients = {
        "terra": _ComparisonVLM(statuses),
        "sol": _ComparisonVLM(statuses),
        "qwen": _ComparisonVLM(
            statuses,
            fail=("p1", config.qwen.model),
        ),
    }
    first, success = asyncio.run(comparison.execute_comparison(
        _context(config, [_record("p1", image)]),
        output,
        vlm_clients=first_clients,
    ))
    assert not success
    assert first["packages"][0]["terra"] is not None
    assert first["packages"][0]["sol"] is not None
    assert first["packages"][0]["qwen"] is None
    assert first["packages"][0]["comparison"]["bucket"] == (
        "terra_only_approved"
    )

    resumed_clients = {
        stage: _ComparisonVLM(statuses)
        for stage in ("terra", "sol", "qwen")
    }
    resumed, success = asyncio.run(comparison.execute_comparison(
        _context(config, [_record("p1", image)]),
        output,
        resume=True,
        vlm_clients=resumed_clients,
    ))
    assert success
    assert resumed_clients["terra"].calls == []
    assert resumed_clients["sol"].calls == []
    assert len(resumed_clients["qwen"].calls) == 1
    assert resumed["packages"][0]["qwen"]["verification_status"] == "uncertain"
    assert resumed["errors"][0]["resolved"] is True


def test_qwen_runtime_endpoint_is_redacted_and_output_prefix_is_separate(
    monkeypatch,
):
    endpoint = "http://local-qwen.invalid:1234"
    monkeypatch.setenv("LM_STUDIO_URL", endpoint)
    default = _config()
    with_qwen = _qwen_config()
    runtime = comparison.qwen_vlm_config(with_qwen)
    assert runtime["url"] == endpoint
    assert endpoint not in json.dumps(with_qwen.redacted_dict())
    assert endpoint not in comparison._sanitize_error(
        f"connection failed for {endpoint}/v1/chat/completions"
    )
    default_prefix = comparison._output_prefix(_context(default, []))
    qwen_prefix = comparison._output_prefix(_context(with_qwen, []))
    assert default_prefix != qwen_prefix
    assert "with_qwen" not in default_prefix
    assert "with_qwen" in qwen_prefix

def test_failure_is_resumable_and_never_becomes_nonapproval(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    image = tmp_path / "img.jpg"
    image.write_bytes(b"image")
    config = _config()
    context = _context(config, [_record("p1", image)])
    statuses = {
        ("p1", config.terra.model): "confirmed",
        ("p1", config.sol.model): "rejected",
    }
    output = tmp_path / "comparison.json"
    failing = _ComparisonVLM(
        statuses,
        fail=("p1", config.sol.model),
    )
    first, success = asyncio.run(comparison.execute_comparison(
        context,
        output,
        vlm_client=failing,
    ))
    assert not success
    package = first["packages"][0]
    assert package["terra"]["verification_status"] == "confirmed"
    assert package["sol"] is None
    assert package["comparison"]["bucket"] is None
    assert first["aggregate"]["execution_failed"] == 1
    resumed_client = _ComparisonVLM(statuses)
    resumed, success = asyncio.run(comparison.execute_comparison(
        context,
        output,
        resume=True,
        vlm_client=resumed_client,
    ))
    assert success
    assert [call["model"] for call in resumed_client.calls] == [config.sol.model]
    assert resumed["packages"][0]["comparison"]["bucket"] == (
        "terra_only_approved"
    )
    assert resumed["errors"][0]["resolved"]
    assert not list(tmp_path.glob("*.tmp"))


def test_optional_executive_review_is_sol_only_blinded_and_non_independent(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    image = tmp_path / "img.jpg"
    image.write_bytes(b"image")
    config = _config()
    record = _record("p1", image)
    identity_text = (
        f"Terra Sol {config.terra.model} {config.sol.model} {config.name}"
    )
    record["prepared_input"]["evidence_items"][0]["observations"] = [identity_text]
    statuses = {
        ("p1", config.terra.model): "confirmed",
        ("p1", config.sol.model): "rejected",
    }
    client = _ComparisonVLM(statuses)
    output = tmp_path / "comparison.json"
    report, success = asyncio.run(comparison.execute_comparison(
        _context(config, [record]),
        output,
        executive_review=True,
        vlm_client=client,
    ))
    assert success
    assert len(client.calls) == 3
    executive_call = client.calls[-1]
    assert executive_call["model"] == "gpt-5.6-sol"
    prompt = executive_call["system_prompt"] + executive_call["user_prompt"]
    for forbidden in (
        "Terra",
        "Sol",
        config.terra.model,
        config.sol.model,
        config.name,
    ):
        assert forbidden not in prompt
    executive = report["packages"][0]["executive_review"]
    assert executive["operational_decision"] == "needs_manual_review"
    assert executive["non_independent"] is True
    assert executive["review_model_is_contestant"] is True
    _, prompt_again, mapping_again = comparison.build_executive_prompts(
        report["packages"][0],
        config,
        "fingerprint-test",
    )
    assert prompt_again == executive_call["user_prompt"]
    assert mapping_again == executive["decision_mapping"]


def test_executive_invalid_response_fails_without_fallback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    image = tmp_path / "img.jpg"
    image.write_bytes(b"image")
    config = _config()
    statuses = {
        ("p1", config.terra.model): "confirmed",
        ("p1", config.sol.model): "rejected",
    }
    client = _ComparisonVLM(
        statuses,
        executive_response="not-json",
    )
    output = tmp_path / "comparison.json"
    report, success = asyncio.run(comparison.execute_comparison(
        _context(config, [_record("p1", image)]),
        output,
        executive_review=True,
        vlm_client=client,
    ))
    assert not success
    package = report["packages"][0]
    assert package["executive_review"] is None
    assert package["comparison"]["bucket"] == "terra_only_approved"
    error = report["errors"][-1]
    assert error["stage"] == "executive_review"
    assert error["error_type"] == "Pass2fInvalidResponseError"
    assert error["raw_response"] == "not-json"
    assert len(client.calls) == 3
    assert client.calls[-1]["model"] == "gpt-5.6-sol"


def test_checkpoint_fingerprint_mismatch_is_rejected(tmp_path):
    output = tmp_path / "report.json"
    report = {
        "schema_version": comparison.REPORT_SCHEMA_VERSION,
        "config_fingerprint": "fingerprint-a",
        "source": {},
        "aggregate": {},
        "phase_stats": {},
        "execution": {},
        "packages": [],
        "errors": [],
    }
    comparison.save_state(output, report)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        comparison.load_checkpoint(output, "fingerprint-b")



def test_prepare_comparison_keeps_source_and_catalog_read_only(
    tmp_path,
    monkeypatch,
):
    image = tmp_path / "img.jpg"
    image.write_bytes(b"image")
    artifact_path = tmp_path / "photo_intel.json"
    artifact = {
        **_source_artifact(),
        "property": {"property_key": "property-test"},
        "estimate_issues_flat": [{"issue_id": "issue_1"}],
        "photos": {},
    }
    artifact["model_routing"][1]["api_key"] = "route-secret"
    artifact["model_routing"][1]["base_url"] = "https://secret.invalid"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text('{"items":[{"id":"x"}]}', encoding="utf-8")
    artifact_before = artifact_path.read_bytes()
    catalog_before = catalog_path.read_bytes()

    monkeypatch.setattr(
        comparison,
        "load_issue_catalog",
        lambda path: {"items": [{"id": "x"}]},
    )
    monkeypatch.setattr(
        comparison,
        "prepare_replay_inputs",
        lambda artifact, catalog: ([{"issue_id": "issue_1"}], [], {}),
    )
    monkeypatch.setattr(
        comparison,
        "infer_package_candidates",
        lambda candidates, rooms, catalog: [_package("p1")],
    )
    monkeypatch.setattr(
        comparison,
        "prepare_pass_2f_cases",
        lambda packages, **kwargs: (
            [{
                "package_id": "p1",
                "package_type": "kitchen_modernization",
                "package_label": "Kitchen modernization",
                "room": "kitchen",
                "source_package": packages[0],
                "evaluation_kind": "vlm_eligible",
                "not_evaluated_reason": None,
                "rule_confirmation": None,
                "prepared_input": {
                    "evidence_items": _evidence(),
                    "review_photo_keys": ["img.jpg"],
                    "review_image_paths": [str(image)],
                    "reviewed_issue_ids": ["issue_1"],
                    "prompt_template_version": PASS_2F_PROMPT_VERSION,
                    "prompt_template_sha256": PASS_2F_PROMPT_SHA256,
                    "max_images_per_package": 3,
                },
            }],
            {
                "candidate_count": 1,
                "vlm_eligible_count": 1,
                "rule_confirmed_count": 0,
                "not_evaluated_count": 0,
                "not_evaluated_reasons": {},
            },
        ),
    )
    context = comparison.prepare_comparison(
        _config(),
        artifact_path,
        catalog_path=catalog_path,
    )
    assert context["preparation_trace"]["vlm_eligible_count"] == 1
    source_json = json.dumps(context["source"])
    assert "route-secret" not in source_json
    assert "https://secret.invalid" not in source_json
    assert "[REDACTED]" in source_json
    assert context["package_records"][0]["prepared_input"][
        "review_image_sha256"
    ] == [comparison._sha256_file(image)]
    assert artifact_path.read_bytes() == artifact_before
    assert catalog_path.read_bytes() == catalog_before


def test_validate_config_path_never_creates_provider_client(
    tmp_path,
    monkeypatch,
    capsys,
):
    artifact_path = tmp_path / "photo_intel.json"
    artifact_path.write_text("{}", encoding="utf-8")
    config = _config()
    context = _context(config, [])
    monkeypatch.setattr(
        comparison,
        "prepare_comparison",
        lambda config, artifact_path: context,
    )
    monkeypatch.setattr(
        comparison,
        "create_vlm_client",
        lambda *args, **kwargs: pytest.fail("provider client created"),
    )
    result = comparison.main([
        "--profile",
        "terra_vs_sol_2f",
        "--run",
        str(artifact_path),
        "--validate-config",
    ])
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["preview"]["package_candidate_count"] == 0


def test_parse_args_accepts_multiple_run_paths():
    args = comparison.parse_args([
        "--profile",
        "terra_vs_sol_2f",
        "--run",
        "run-one",
        "run-two",
        "run-three",
    ])
    assert args.runs == ["run-one", "run-two", "run-three"]
    assert args.include_qwen is False


def test_parse_args_accepts_include_qwen():
    args = comparison.parse_args([
        "--profile",
        "terra_vs_sol_2f",
        "--run",
        "run-one",
        "--include-qwen",
    ])
    assert args.include_qwen is True


def test_batch_validate_prepares_all_runs_without_provider(
    tmp_path,
    monkeypatch,
    capsys,
):
    artifacts = []
    for name in ("property-one", "property-two"):
        artifact = tmp_path / name / "photo_intel.json"
        artifact.parent.mkdir()
        artifact.write_text("{}", encoding="utf-8")
        artifacts.append(artifact)

    prepared = []

    def fake_prepare(config, artifact_path):
        prepared.append(artifact_path)
        context = _context(config, [])
        context["config_fingerprint"] = f"fingerprint-{artifact_path.parent.name}"
        context["source"] = {
            **context["source"],
            "artifact_path": str(artifact_path),
            "artifact_sha256": (artifact_path.parent.name + "0" * 64)[:64],
            "property_id": artifact_path.parent.name,
        }
        return context

    monkeypatch.setattr(comparison, "prepare_comparison", fake_prepare)
    monkeypatch.setattr(
        comparison,
        "create_vlm_client",
        lambda *args, **kwargs: pytest.fail("provider client created"),
    )

    result = comparison.main([
        "--profile",
        "terra_vs_sol_2f",
        "--run",
        *(str(path) for path in artifacts),
        "--validate-config",
    ])

    assert result == 0
    assert prepared == artifacts
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["run_count"] == 2
    assert [run["source"]["property_id"] for run in payload["runs"]] == [
        "property-one",
        "property-two",
    ]


def test_batch_execution_prepares_all_then_runs_sequentially(
    tmp_path,
    monkeypatch,
):
    artifacts = []
    for name in ("property-one", "property-two"):
        artifact = tmp_path / name / "photo_intel.json"
        artifact.parent.mkdir()
        artifact.write_text("{}", encoding="utf-8")
        artifacts.append(artifact)

    events = []
    output_paths = []

    def fake_prepare(config, artifact_path):
        property_id = artifact_path.parent.name
        events.append(("prepare", property_id))
        context = _context(config, [])
        context["config_fingerprint"] = f"fingerprint-{property_id}"
        context["source"] = {
            **context["source"],
            "artifact_path": str(artifact_path),
            "artifact_sha256": (property_id + "0" * 64)[:64],
            "property_id": property_id,
        }
        return context

    async def fake_execute(
        context,
        output_path,
        *,
        resume=False,
        executive_review=False,
        vlm_client=None,
    ):
        property_id = context["source"]["property_id"]
        events.append(("execute", property_id))
        output_paths.append(output_path)
        return {"aggregate": {}, "packages": []}, True

    monkeypatch.setattr(comparison, "prepare_comparison", fake_prepare)
    monkeypatch.setattr(comparison, "execute_comparison", fake_execute)
    output_dir = tmp_path / "reports"

    result = comparison.main([
        "--profile",
        "terra_vs_sol_2f",
        "--run",
        *(str(path) for path in artifacts),
        "--output",
        str(output_dir),
        "--resume",
    ])

    assert result == 0
    assert events == [
        ("prepare", "property-one"),
        ("prepare", "property-two"),
        ("execute", "property-one"),
        ("execute", "property-two"),
    ]
    assert output_dir.is_dir()
    assert len(set(output_paths)) == 2
    assert all(path.parent == output_dir.resolve() for path in output_paths)

