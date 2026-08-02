"""
Fail-fast photo scheduling.

The property under test is cost: once one photo has failed, strict mode cannot
publish the property anyway, so every further provider call is money spent on an
artifact that will never be written. These tests pin that no such call happens,
and — just as importantly — that stopping early does not corrupt the two things
the rest of the pipeline depends on: index alignment with `image_paths` (which is
what makes `image_{idx:04d}.json` checkpoints meaningful) and the failure
occurrence count (which feeds the worker's circuit breaker).

Async tests use bare `asyncio.run()`; no pytest.ini exists, so no plugins.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.failure_taxonomy import FailureDescriptor
from tools.photo_pass_runner import (
    ABORTED_ERROR_KIND,
    PhotoFailFast,
    is_aborted,
    run_photo_passes,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _paths(n):
    return [Path(f"/photos/img_{i:04d}.jpg") for i in range(n)]


def _make_failed(idx, path, exc):
    return SimpleNamespace(image_path=str(path), error=str(exc), error_kind=None, scene_data=None)


def _make_aborted(idx, path):
    return SimpleNamespace(
        image_path=str(path), error="aborted", error_kind=ABORTED_ERROR_KIND, scene_data=None
    )


def _ok(path):
    return SimpleNamespace(image_path=str(path), error=None, error_kind=None, scene_data=None)


def _run(**kwargs):
    kwargs.setdefault("make_failed", _make_failed)
    kwargs.setdefault("make_aborted", _make_aborted)
    return asyncio.run(run_photo_passes(**kwargs))


# ── the core guarantee ───────────────────────────────────────────────────────

def test_no_provider_calls_after_the_first_failure():
    """
    Concurrency 1 makes the schedule fully deterministic: photo 0 fails, so
    photos 1-9 must never reach analyze_one.
    """
    attempted = []

    async def analyze_one(idx, path):
        attempted.append(idx)
        if idx == 0:
            raise RuntimeError("provider exploded")
        return _ok(path)

    outcome = _run(image_paths=_paths(10), concurrency=1, analyze_one=analyze_one)

    assert attempted == [0]
    assert outcome.attempted == 1
    assert outcome.skipped == 9


def test_results_stay_index_aligned_with_image_paths():
    """
    Checkpoint filenames are derived from the index. A short or reordered result
    list would silently attach one photo's analysis to another photo's slot.
    """
    async def analyze_one(idx, path):
        if idx == 3:
            raise RuntimeError("boom")
        return _ok(path)

    outcome = _run(image_paths=_paths(10), concurrency=1, analyze_one=analyze_one)

    assert len(outcome.results) == 10
    for idx, result in enumerate(outcome.results):
        assert Path(result.image_path).name == f"img_{idx:04d}.jpg"


def test_already_active_requests_are_allowed_to_finish():
    """
    A photo past the gate holds a live provider request. Cancelling it mid-flight
    risks a leaked connection and ambiguous billing, so it must complete.
    """
    started = asyncio.Event()

    async def analyze_one(idx, path):
        if idx == 0:
            # Let photo 1 get past the gate before failing.
            await started.wait()
            raise RuntimeError("boom")
        started.set()
        await asyncio.sleep(0.02)
        return _ok(path)

    outcome = _run(image_paths=_paths(10), concurrency=2, analyze_one=analyze_one)

    assert outcome.results[0].error == "boom"
    assert outcome.results[1].error is None, "in-flight photo was cancelled instead of finishing"


def test_collect_mode_still_attempts_every_photo():
    """
    failure_mode='collect' exists to see every photo's failure; the benchmark
    tooling and test_pass_failures.py depend on it.
    """
    attempted = []

    async def analyze_one(idx, path):
        attempted.append(idx)
        if idx == 0:
            raise RuntimeError("boom")
        return _ok(path)

    outcome = _run(
        image_paths=_paths(10), concurrency=2, analyze_one=analyze_one, fail_fast=False
    )

    assert sorted(attempted) == list(range(10))
    assert outcome.skipped == 0


# ── aborted is not failed ────────────────────────────────────────────────────

def test_skipped_photos_are_tagged_aborted_not_failed():
    async def analyze_one(idx, path):
        if idx == 0:
            raise RuntimeError("boom")
        return _ok(path)

    outcome = _run(image_paths=_paths(5), concurrency=1, analyze_one=analyze_one)

    assert not is_aborted(outcome.results[0]), "the real failure must not be tagged aborted"
    for result in outcome.results[1:]:
        assert is_aborted(result)


def test_one_fault_reports_one_occurrence_not_one_per_photo():
    """
    The count feeds a circuit breaker that pauses the *entire* queue at two
    matching failures. If aborted photos counted, a single provider fault in one
    property would pause everything on its own.
    """
    async def analyze_one(idx, path):
        if idx == 0:
            raise RuntimeError("boom")
        return _ok(path)

    outcome = _run(image_paths=_paths(40), concurrency=1, analyze_one=analyze_one)

    assert outcome.skipped == 39
    assert outcome.occurrence_count == 1


def test_genuinely_concurrent_faults_are_counted_separately():
    """Two photos that really did fail should count twice — that is a live signal."""
    async def analyze_one(idx, path):
        raise RuntimeError("boom")

    outcome = _run(
        image_paths=_paths(4), concurrency=4, analyze_one=analyze_one, fail_fast=False
    )

    assert outcome.occurrence_count == 4


# ── failure reporting ────────────────────────────────────────────────────────

def test_first_failure_is_the_one_reported():
    async def analyze_one(idx, path):
        if idx == 0:
            raise FileNotFoundError("the original cause")
        raise RuntimeError("a later, less interesting failure")

    outcome = _run(
        image_paths=_paths(3), concurrency=1, analyze_one=analyze_one, fail_fast=False
    )

    assert outcome.first_failure.category == "input"
    assert "original cause" in outcome.first_failure.message


def test_failure_returned_rather_than_raised_still_trips_the_run():
    """
    Some callers catch internally and report by returning a result with `.error`.
    Both routes must stop the run; otherwise the fail-fast is trivially bypassed.
    """
    attempted = []

    async def analyze_one(idx, path):
        attempted.append(idx)
        if idx == 0:
            return SimpleNamespace(
                image_path=str(path), error="failed quietly", error_kind=None, scene_data=None
            )
        return _ok(path)

    outcome = _run(image_paths=_paths(6), concurrency=1, analyze_one=analyze_one)

    assert attempted == [0]
    assert outcome.skipped == 5


def test_pass_context_survives_into_the_descriptor():
    """A returned failure carrying pass_errors should classify by that, not fall back."""
    async def analyze_one(idx, path):
        return SimpleNamespace(
            image_path=str(path),
            error="pass 2d blew up",
            error_kind=None,
            scene_data={"debug": {"pass_errors": {
                "2d": {"stage": "dependency", "code": "EmbeddingsRuntimeError",
                       "message": "no provider", "provider": "local", "model": None},
            }}},
        )

    outcome = _run(image_paths=_paths(1), concurrency=1, analyze_one=analyze_one)

    assert outcome.first_failure.category == "dependency"
    assert outcome.first_failure.pass_key == "2d"
    assert outcome.first_failure.code == "EmbeddingsRuntimeError"


# ── resume / checkpoints ─────────────────────────────────────────────────────

def test_cached_photos_are_replayed_and_never_gated():
    """
    Resume must work even when the previous attempt died: replaying finished work
    costs nothing, so cache hits are exempt from the fail-fast gate.
    """
    cached = {0: _ok(Path("/photos/img_0000.jpg")), 1: _ok(Path("/photos/img_0001.jpg"))}
    seen_cached, attempted = [], []

    async def analyze_one(idx, path):
        attempted.append(idx)
        raise RuntimeError("boom")

    outcome = _run(
        image_paths=_paths(4), concurrency=1, analyze_one=analyze_one,
        cached_results=cached, on_cached=lambda idx, r: seen_cached.append(idx),
    )

    assert seen_cached == [0, 1]
    assert outcome.reused == 2
    assert attempted == [2], "only the first uncached photo should be attempted"
    assert outcome.skipped == 1


def test_on_result_fires_for_fresh_results_only():
    """
    on_result is where callers checkpoint. Firing it for cache hits would rewrite
    checkpoints that already exist; firing it for aborted photos would checkpoint
    work that never happened.
    """
    checkpointed = []

    async def analyze_one(idx, path):
        if idx == 2:
            raise RuntimeError("boom")
        return _ok(path)

    _run(
        image_paths=_paths(5), concurrency=1, analyze_one=analyze_one,
        cached_results={0: _ok(Path("/photos/img_0000.jpg"))},
        on_result=lambda idx, r: checkpointed.append((idx, r.error)),
    )

    # 0 was cached; 3 and 4 were aborted before reaching the semaphore.
    assert [idx for idx, _ in checkpointed] == [1, 2]
    assert checkpointed[0][1] is None
    assert checkpointed[1][1] == "boom"


# ── PhotoFailFast unit ───────────────────────────────────────────────────────

def test_failfast_keeps_the_first_descriptor_but_tallies_the_rest():
    failfast = PhotoFailFast()
    first = FailureDescriptor(category="quota", code="RateLimitError", message="first")
    same = FailureDescriptor(category="quota", code="RateLimitError", message="second")

    assert not failfast.tripped
    failfast.trip(first)
    failfast.trip(same)

    assert failfast.tripped
    assert failfast.first.message == "first"
    assert failfast.occurrence_count() == 2


def test_failfast_does_not_conflate_different_root_causes():
    failfast = PhotoFailFast()
    failfast.trip(FailureDescriptor(category="quota", code="RateLimitError", message="a"))
    failfast.trip(FailureDescriptor(category="timeout", code="APITimeoutError", message="b"))

    assert failfast.occurrence_count() == 1


def test_empty_photo_set_is_not_an_error():
    async def analyze_one(idx, path):  # pragma: no cover - must never run
        raise AssertionError("should not be called")

    outcome = _run(image_paths=[], concurrency=4, analyze_one=analyze_one)

    assert outcome.results == []
    assert outcome.first_failure is None
