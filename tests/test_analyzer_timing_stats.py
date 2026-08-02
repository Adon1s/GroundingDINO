import logging
import json
import sys
from types import SimpleNamespace

from tools.analyzer_cli import ImageResult, _compute_timing_stats, _log_timing_stats


def test_compute_timing_stats_separates_work_from_wall_time():
    results = [
        ImageResult(
            image_path="photo_001.jpg",
            scene_data={
                "pass_timings": {"1a": 2.0, "2a": 10.0},
                "passes_run": ["1a", "2a"],
            },
            processing_time=12.0,
        ),
        ImageResult(
            image_path="photo_002.jpg",
            scene_data={
                "pass_timings": {"1a": 3.0, "2a": 9.0},
                "passes_run": ["1a", "2a"],
            },
            processing_time=12.0,
        ),
    ]

    stats = _compute_timing_stats(
        results,
        total_wall_clock=10.0,
        usage_stats={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
            "calls": 4,
            "attempted_calls": 4,
            "failed_calls": 0,
            "api_duration_sec": 8.0,
            "per_pass": {"1a": {"attempted_calls": 2, "calls": 2}},
            "metered_calls": 1,
        },
        phase_timings={"photo_analysis_sec": 10.0, "postprocessing_sec": 4.0, "pass_2f_sec": 3.0, "end_to_end_sec": 14.0},
        configured_concurrency=4,
        requested_photo_count=2,
    )

    assert stats["total_wall_clock_sec"] == 14.0
    assert stats["total_llm_work_sec"] == 8.0
    assert stats["parallelism_ratio"] == 2.4
    assert stats["llm_calls"] == 4
    assert stats["metered_llm_calls"] == 1

    assert stats["schema_version"] == 2
    assert stats["phases"]["photo_analysis_sec"] + stats["phases"]["postprocessing_sec"] == stats["phases"]["end_to_end_sec"]
    assert stats["photos"]["avg_latency_sec"] == 12.0
    assert stats["photos"]["throughput_sec_per_photo"] == 5.0
    assert stats["passes"]["1a"]["attempted_calls"] == 2

def test_log_timing_stats_labels_cumulative_work_and_metered_calls(caplog):
    stats = {
        "schema_version": 2,
        "status": {"state": "complete", "failed_phase": None},
        "phases": {"end_to_end_sec": 14.0, "photo_analysis_sec": 10.0, "postprocessing_sec": 4.0, "pass_2f_sec": 3.0},
        "photos": {"requested": 2, "processed_this_attempt": 2, "reused_from_checkpoint": 0, "successful_total": 2, "failed_total": 0, "avg_latency_sec": 12.0, "throughput_sec_per_photo": 5.0, "effective_parallelism": 2.4, "configured_concurrency": 4},
        "usage": {"attempted_calls": 4, "successful_calls": 4, "failed_calls": 0, "metered_calls": 1, "input_tokens": 100, "output_tokens": 25, "total_tokens": 125, "job_tokens_per_sec": 8.929},
        "photo_count": 2,
        "total_wall_clock_sec": 10.0,
        "total_llm_work_sec": 24.0,
        "parallelism_ratio": 2.4,
        "per_pass_total_sec": {"1a": 5.0, "2a": 19.0},
        "per_pass_avg_sec": {"1a": 2.5, "2a": 9.5},
        "passes_run": ["1a", "2a"],
        "passes_skipped": [],
        "input_tokens": 100,
        "output_tokens": 25,
        "total_tokens": 125,
        "llm_calls": 4,
        "metered_llm_calls": 1,
        "passes": {
            "1a": {"state": "executed", "work_sec": 5.0, "api_work_sec": 2.0, "attempted_calls": 2, "total_tokens": 50},
            "2f": {"state": "executed", "work_sec": 3.0, "api_work_sec": 3.0, "attempted_calls": 2, "total_tokens": 75},
        },
    }

    with caplog.at_level(logging.INFO, logger="tools.analyzer_cli"):
        _log_timing_stats(stats, "redfin_test")

    assert "PHASE WALL TIME" in caplog.text
    assert "PASS WORK" in caplog.text
    assert "2f" in caplog.text
    assert "Job token throughput" in caplog.text

def test_checkpoint_resume_uses_only_current_attempt_work():
    cached = ImageResult(
        image_path="cached.jpg",
        scene_data={"pass_timings": {"2a": 100.0}, "pass_states": {"2a": "executed"}},
        processing_time=100.0,
    )
    current = ImageResult(
        image_path="current.jpg",
        scene_data={
            "pass_timings": {"1b": 0.0, "2a": 4.0, "2e": 0.1},
            "pass_states": {"1b": "stubbed", "2a": "executed", "2e": "rule_based"},
        },
        processing_time=5.0,
    )

    stats = _compute_timing_stats(
        [cached, current],
        phase_timings={
            "photo_analysis_sec": 2.0,
            "postprocessing_sec": 1.0,
            "end_to_end_sec": 3.0,
        },
        attempt_results=[current],
        requested_photo_count=2,
        reused_photo_count=1,
        configured_concurrency=4,
    )

    assert stats["photos"]["processed_this_attempt"] == 1
    assert stats["photos"]["reused_from_checkpoint"] == 1
    assert stats["photos"]["successful_total"] == 2
    assert stats["passes"]["2a"]["work_sec"] == 4.0
    assert stats["passes"]["1b"]["state"] == "stubbed"
    assert stats["passes"]["2e"]["state"] == "rule_based"
    assert stats["parallelism_ratio"] == 2.05

def test_observed_29_photo_four_way_regression_includes_pass_2f():
    pass_totals = {
        "1a": 46.95,
        "1b": 0.0,
        "1c": 0.0,
        "2a": 167.43,
        "2b": 68.87,
        "2c": 61.04,
        "2d": 71.27,
        "2e": 0.0,
    }
    states = {
        "1a": "executed",
        "1b": "stubbed",
        "1c": "stubbed",
        "2a": "executed",
        "2b": "executed",
        "2c": "executed",
        "2d": "executed",
        "2e": "rule_based",
    }
    results = [
        ImageResult(
            image_path=f"photo_{index:03d}.jpg",
            scene_data={
                "pass_timings": {
                    key: value / 29 for key, value in pass_totals.items()
                },
                "pass_states": states,
            },
            processing_time=14.33,
        )
        for index in range(29)
    ]
    usage = {
        "attempted_calls": 210,
        "calls": 210,
        "failed_calls": 0,
        "metered_calls": 210,
        "input_tokens": 47000,
        "output_tokens": 3000,
        "total_tokens": 50000,
        "api_duration_sec": 445.56,
        "per_pass": {
            "2f": {
                "attempted_calls": 7,
                "calls": 7,
                "failed_calls": 0,
                "metered_calls": 7,
                "input_tokens": 22073,
                "output_tokens": 1853,
                "total_tokens": 23926,
                "api_duration_sec": 30.0,
            },
        },
    }

    stats = _compute_timing_stats(
        results,
        phase_timings={
            "photo_analysis_sec": 113.82,
            "postprocessing_sec": 38.0,
            "pass_2f_sec": 30.0,
            "end_to_end_sec": 151.82,
        },
        attempt_results=results,
        requested_photo_count=29,
        configured_concurrency=4,
        usage_stats=usage,
    )

    assert stats["photos"]["requested"] == 29
    assert stats["photos"]["configured_concurrency"] == 4
    assert stats["photos"]["cumulative_pass_work_sec"] == 415.56
    assert stats["parallelism_ratio"] == 3.65
    assert stats["phases"]["photo_analysis_sec"] + stats["phases"]["postprocessing_sec"] == stats["phases"]["end_to_end_sec"]
    assert stats["passes"]["2f"]["attempted_calls"] == 7
    assert stats["passes"]["2f"]["total_tokens"] == 23926
    assert stats["passes"]["1b"]["state"] == "stubbed"
    assert stats["passes"]["2e"]["state"] == "rule_based"
    assert stats["total_llm_work_sec"] == 445.56

def test_failed_zero_success_attempt_handles_missing_usage_and_partial_phases():
    failed = ImageResult(
        image_path="failed.jpg",
        scene="unknown",
        processing_time=1.5,
        error="provider unavailable",
    )
    stats = _compute_timing_stats(
        [failed],
        phase_timings={
            "photo_analysis_sec": 1.5,
            "postprocessing_sec": 0.5,
            "pass_2f_sec": 0.0,
            "end_to_end_sec": 2.0,
        },
        attempt_results=[failed],
        requested_photo_count=1,
        usage_stats={
            "attempted_calls": 1,
            "calls": 0,
            "failed_calls": 1,
            "per_pass": {
                "2a": {
                    "attempted_calls": 1,
                    "calls": 0,
                    "failed_calls": 1,
                }
            },
        },
        status="failed",
        failed_phase="postprocessing",
    )

    assert stats["status"] == {
        "state": "failed",
        "failed_phase": "postprocessing",
    }
    assert stats["photos"]["successful_this_attempt"] == 0
    assert stats["photos"]["failed_this_attempt"] == 1
    assert stats["photos"]["throughput_photos_per_sec"] == 0.0
    assert stats["usage"]["total_tokens"] == 0
    assert stats["usage"]["job_tokens_per_sec"] == 0.0
    assert stats["passes"]["2a"]["state"] == "failed"
    assert stats["passes"]["2f"]["state"] == "skipped"

def test_persistent_worker_emits_timing_stats_when_artifact_generation_fails(
    tmp_path,
    monkeypatch,
):
    from tools import analyzer_server

    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"image")

    class FakeClient:
        def __init__(self):
            self.usage_stats = {
                "attempted_calls": 1,
                "calls": 1,
                "failed_calls": 0,
                "metered_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "api_duration_sec": 0.01,
                "per_pass": {
                    "1a": {
                        "attempted_calls": 1,
                        "calls": 1,
                        "failed_calls": 0,
                        "api_duration_sec": 0.01,
                    }
                },
            }

        def reset_usage_stats(self):
            return None

    class FakeOrchestrator:
        async def analyze_image(self, **_kwargs):
            return SimpleNamespace(
                scene="kitchen",
                to_dict=lambda: {
                    "pass_timings": {"1a": 0.02},
                    "pass_states": {
                        "1a": "executed",
                        "1b": "stubbed",
                        "1c": "stubbed",
                        "2a": "skipped",
                        "2b": "skipped",
                        "2c": "skipped",
                        "2d": "skipped",
                        "2e": "rule_based",
                    },
                },
            )

    events = []
    monkeypatch.setattr(analyzer_server, "_emit", events.append)

    def fail_artifacts(**_kwargs):
        raise RuntimeError("artifact write failed")

    analyzer_server._process_job(
        request={
            "jobId": "job-1",
            "runId": "run-1",
            "propertyKey": "redfin_test",
            "images": [str(image_path)],
            "artifactsRoot": str(tmp_path / "artifacts"),
            "analysisProfile": "standard",
            "modelRoutingProfile": "standard",
            "concurrency": 1,
        },
        orchestrator=FakeOrchestrator(),
        catalog={},
        gpt5_config={},
        vlm_client=FakeClient(),
        write_photo_intel=fail_artifacts,
    )

    result_event = next(event for event in events if event.get("type") == "result")
    assert result_event["success"] is False
    assert result_event["timing_stats"]["schema_version"] == 2
    assert result_event["timing_stats"]["status"] == {
        "state": "failed",
        "failed_phase": "postprocessing",
    }
    assert result_event["timing_stats"]["photos"]["processed_this_attempt"] == 1
    assert events[-1] == {"type": "job_done", "jobId": "job-1"}

def test_cli_path_uses_one_client_for_photo_passes_and_pass_2f(
    tmp_path,
    monkeypatch,
    capsys,
):
    from tools import analyzer_cli
    from tools import artifact_writers
    from tools import scene_classifier_orchestrator
    from tools import vlm_client as vlm_module

    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"image")
    captured = {}

    class FakeClient:
        def __init__(self):
            self.reset_count = 0
            self.usage_stats = {}

        def reset_usage_stats(self):
            self.reset_count += 1
            self.usage_stats = {
                "attempted_calls": 0,
                "calls": 0,
                "failed_calls": 0,
                "metered_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "api_duration_sec": 0.0,
                "per_pass": {},
            }

    client = FakeClient()

    class FakeOrchestrator:
        async def analyze_image(self, **_kwargs):
            client.usage_stats.update({
                "attempted_calls": 1,
                "calls": 1,
                "metered_calls": 1,
                "input_tokens": 8,
                "output_tokens": 2,
                "total_tokens": 10,
                "api_duration_sec": 0.02,
                "per_pass": {
                    "1a": {
                        "attempted_calls": 1,
                        "calls": 1,
                        "failed_calls": 0,
                        "metered_calls": 1,
                        "input_tokens": 8,
                        "output_tokens": 2,
                        "total_tokens": 10,
                        "api_duration_sec": 0.02,
                    }
                },
            })
            return SimpleNamespace(
                scene="kitchen",
                to_dict=lambda: {
                    "pass_timings": {"1a": 0.02},
                    "pass_states": {"1a": "executed"},
                },
            )

    def fake_factory(_cfg, **kwargs):
        captured["factory_client"] = kwargs["vlm_client"]
        return FakeOrchestrator()

    def fake_write_photo_intel(**kwargs):
        captured["writer_client"] = kwargs["vlm_client"]
        kwargs["timing_recorder"]["pass_2f_sec"] = 0.03
        client.usage_stats.update({
            "attempted_calls": 2,
            "calls": 2,
            "metered_calls": 2,
            "input_tokens": 19,
            "output_tokens": 4,
            "total_tokens": 23,
            "api_duration_sec": 0.05,
        })
        client.usage_stats["per_pass"]["2f"] = {
            "attempted_calls": 1,
            "calls": 1,
            "failed_calls": 0,
            "metered_calls": 1,
            "input_tokens": 11,
            "output_tokens": 2,
            "total_tokens": 13,
            "api_duration_sec": 0.03,
        }
        return tmp_path / "photo_intel.json"

    # --disable-2d below is what keeps this test off the embeddings sidecar.
    monkeypatch.setattr(artifact_writers, "load_issue_catalog", lambda _path: {"items": []})
    monkeypatch.setattr(artifact_writers, "write_photo_intel", fake_write_photo_intel)
    monkeypatch.setattr(
        scene_classifier_orchestrator,
        "create_orchestrator_from_config",
        fake_factory,
    )
    monkeypatch.setattr(
        vlm_module,
        "get_model_configs_from_pipeline_config",
        lambda _cfg: ({}, {"provider": "openai", "model": "test"}),
    )
    monkeypatch.setattr(vlm_module, "create_vlm_client", lambda: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyzer_cli",
            "--property-key",
            "redfin_cli_test",
            "--images",
            str(image_path),
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--concurrency",
            "1",
            "--disable-2d",
        ],
    )

    assert analyzer_cli.main() == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    timing = summary["timing_stats"]

    assert client.reset_count == 1
    assert captured["factory_client"] is client
    assert captured["writer_client"] is client
    assert timing["schema_version"] == 2
    assert timing["passes"]["1a"]["total_tokens"] == 10
    assert timing["passes"]["2f"]["total_tokens"] == 13
    assert timing["passes"]["2f"]["work_sec"] == 0.03
    assert timing["usage"]["total_tokens"] == 23
    assert timing["phases"]["photo_analysis_sec"] + timing["phases"]["postprocessing_sec"] == timing["phases"]["end_to_end_sec"]
