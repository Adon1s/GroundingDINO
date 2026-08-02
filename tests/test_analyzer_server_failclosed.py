"""
Persistent-server behavior when a photo fails.

Covers the two things that make fail-fast safe rather than merely cheap:

**Checkpoints must survive and stay accurate.** Successful photos are persisted
so a retry resumes; failed and *aborted* photos must not be, or the retry would
skip work that never happened. `_clear_checkpoint` runs only on full success, so
a failed run must leave the directory intact.

**The emitted failure must be classifiable.** The TS worker decides retry vs
review vs pause-the-whole-queue from the `failure` object; if it is missing or
reports one fault as forty occurrences, the circuit breaker misfires.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.photo_pass_runner import ABORTED_ERROR_KIND


class _FakeClient:
    def __init__(self):
        self.usage_stats = {
            "attempted_calls": 1, "calls": 1, "failed_calls": 0, "metered_calls": 0,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "api_duration_sec": 0.01, "per_pass": {},
        }

    def reset_usage_stats(self):
        return None


def _scene(name="kitchen"):
    return SimpleNamespace(
        scene=name,
        to_dict=lambda: {
            "pass_timings": {"1a": 0.01},
            "pass_states": {"1a": "executed", "2e": "rule_based"},
        },
    )


class _FailAfter:
    """Succeeds for the first `n` photos, then raises for every photo after."""

    def __init__(self, n, exc):
        self.n = n
        self.exc = exc
        self.calls = 0

    async def analyze_image(self, **kwargs):
        self.calls += 1
        if self.calls > self.n:
            raise self.exc
        return _scene()


def _run_job(tmp_path, monkeypatch, orchestrator, photo_count, concurrency=1):
    from tools import analyzer_server

    images = []
    for i in range(photo_count):
        p = tmp_path / f"photo_{i}.jpg"
        p.write_bytes(b"image")
        images.append(str(p))

    events = []
    monkeypatch.setattr(analyzer_server, "_emit", events.append)

    artifacts_root = tmp_path / "artifacts"
    analyzer_server._process_job(
        request={
            "jobId": "job-1",
            "runId": "run-1",
            "propertyKey": "redfin_test",
            "images": images,
            "artifactsRoot": str(artifacts_root),
            "analysisProfile": "standard",
            "modelRoutingProfile": "standard",
            "concurrency": concurrency,
        },
        orchestrator=orchestrator,
        catalog={},
        gpt5_config={},
        vlm_client=_FakeClient(),
        write_photo_intel=lambda **_kw: str(tmp_path / "photo_intel.json"),
    )

    ckpt_dir = artifacts_root / "redfin_test" / ".checkpoints" / "run-1"
    return events, ckpt_dir


def _result_event(events):
    return next(e for e in events if e.get("type") == "result")


# ── fail-fast at the job level ───────────────────────────────────────────────

def test_photo_failure_stops_the_job_and_reports_success_false(tmp_path, monkeypatch):
    orchestrator = _FailAfter(0, RuntimeError("provider exploded"))
    events, _ = _run_job(tmp_path, monkeypatch, orchestrator, photo_count=8)

    result = _result_event(events)
    assert result["success"] is False
    assert orchestrator.calls == 1, "photos kept being analyzed after the first failure"
    assert events[-1] == {"type": "job_done", "jobId": "job-1"}


def test_skipped_photos_are_reported_separately_from_failures(tmp_path, monkeypatch):
    events, _ = _run_job(
        tmp_path, monkeypatch, _FailAfter(0, RuntimeError("boom")), photo_count=8
    )

    result = _result_event(events)
    assert result["photos_attempted"] == 1
    assert result["photos_skipped"] == 7
    assert len(result["photo_failures"]) == 1


def test_one_provider_fault_reports_one_occurrence(tmp_path, monkeypatch):
    """
    The count the circuit breaker reads. Inflating it to the photo count would
    pause the entire queue on a single fault.
    """
    events, _ = _run_job(
        tmp_path, monkeypatch, _FailAfter(0, RuntimeError("boom")), photo_count=40
    )

    assert _result_event(events)["failure"]["occurrence_count"] == 1


# ── the failure object ───────────────────────────────────────────────────────

def test_failure_object_is_present_and_classified(tmp_path, monkeypatch):
    events, _ = _run_job(
        tmp_path, monkeypatch, _FailAfter(0, FileNotFoundError("photo.jpg")), photo_count=3
    )

    failure = _result_event(events)["failure"]
    assert failure["category"] == "input"
    assert failure["code"] == "FileNotFoundError"
    assert failure["fingerprint"]


def test_failure_payload_is_json_serializable(tmp_path, monkeypatch):
    """It goes out over NDJSON; an unserializable field would kill the worker link."""
    events, _ = _run_job(
        tmp_path, monkeypatch, _FailAfter(0, RuntimeError("boom")), photo_count=2
    )

    round_tripped = json.loads(json.dumps(_result_event(events)))
    assert round_tripped["success"] is False


# ── checkpoints ──────────────────────────────────────────────────────────────

def test_only_successful_photos_are_checkpointed(tmp_path, monkeypatch):
    """
    Photo 0 succeeds, photo 1 fails, photos 2-5 are aborted. Only photo 0 may
    have a checkpoint — checkpointing an aborted photo would make the retry skip
    work that never ran.
    """
    events, ckpt_dir = _run_job(
        tmp_path, monkeypatch, _FailAfter(1, RuntimeError("boom")), photo_count=6
    )

    saved = sorted(p.name for p in ckpt_dir.glob("image_*.json"))
    assert saved == ["image_0000.json"]


def test_checkpoints_survive_a_failed_run(tmp_path, monkeypatch):
    """
    `_clear_checkpoint` runs only after a fully successful job. A failed run must
    leave the directory intact so the retry can resume rather than re-pay.
    """
    _, ckpt_dir = _run_job(
        tmp_path, monkeypatch, _FailAfter(2, RuntimeError("boom")), photo_count=6
    )

    assert ckpt_dir.exists()
    assert len(list(ckpt_dir.glob("image_*.json"))) == 2
    assert (ckpt_dir / "policy.json").exists(), "policy fingerprint lost; retry would rebuild"


def test_successful_run_clears_checkpoints(tmp_path, monkeypatch):
    """The complement — otherwise the previous test would pass on a no-op."""
    class _AlwaysOk:
        async def analyze_image(self, **kwargs):
            return _scene()

    events, ckpt_dir = _run_job(tmp_path, monkeypatch, _AlwaysOk(), photo_count=3)

    assert _result_event(events).get("success") is not False
    assert not list(ckpt_dir.glob("image_*.json"))


def test_retry_resumes_from_surviving_checkpoints(tmp_path, monkeypatch):
    """
    End to end: a failed run leaves checkpoints, and a second run with the same
    runId re-analyzes only what is missing. This is what makes the worker's
    one-retry budget affordable.
    """
    first = _FailAfter(2, RuntimeError("boom"))
    _run_job(tmp_path, monkeypatch, first, photo_count=5)
    assert first.calls == 3  # 2 ok + 1 failed

    second = _FailAfter(99, RuntimeError("unused"))
    events, _ = _run_job(tmp_path, monkeypatch, second, photo_count=5)

    assert second.calls == 3, "resume re-analyzed photos that were already checkpointed"
    assert _result_event(events).get("success") is not False
