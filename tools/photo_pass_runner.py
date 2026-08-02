"""
Per-photo pass execution with fail-fast.

Shared by `analyzer_server` and `analyzer_cli` so the two entrypoints cannot
drift on when a run stops. Before this existed, every coroutine was created up
front and the semaphore only throttled concurrency — so a quota wall on photo 1
still burned provider calls for the remaining 39 photos, all of which were
guaranteed to fail and none of which could be published anyway (strict mode
refuses to write `photo_intel.json` when any photo errored).

Two design choices worth knowing before editing:

**The gate, not `task.cancel()`.** A coroutine still waiting on the semaphore has
not issued a provider request, so letting it acquire and immediately bail is
behaviorally identical to cancelling it — without the hazards of cancelling
mid-`await client.responses.create()` (leaked connections, ambiguous billing, an
httpx pool left in an undefined state). Tasks already past the gate are never
interrupted, which is exactly the "only already-active requests may finish" rule.

**Aborted is not failed.** Photos skipped because an earlier photo failed are
tagged `error_kind="aborted"` and excluded from the occurrence tally. If they
counted as failures, one real quota error in a 40-photo property would report 40
occurrences and trip the worker's circuit breaker — which pauses the entire queue
— on what was a single provider fault. This is the sharpest edge in the module.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

from tools.failure_taxonomy import FailureDescriptor, classify_failure

ABORTED_ERROR_KIND = "aborted"
ABORTED_MESSAGE = "aborted: run stopped after an earlier photo failed"


class PhotoFailFast:
    """
    First hard failure wins.

    Later failures are tallied — the count feeds the worker's circuit breaker —
    but never replace the first one, so the reported cause is the failure that
    actually stopped the run rather than whichever coroutine happened to lose a
    race afterwards.
    """

    def __init__(self) -> None:
        self.first: Optional[FailureDescriptor] = None
        self.occurrences: Counter = Counter()

    @property
    def tripped(self) -> bool:
        return self.first is not None

    def trip(self, descriptor: FailureDescriptor) -> None:
        self.occurrences[descriptor.fingerprint()] += 1
        if self.first is None:
            self.first = descriptor

    def occurrence_count(self) -> int:
        """How many times the *first* failure's root cause was seen."""
        if self.first is None:
            return 0
        return self.occurrences[self.first.fingerprint()]


@dataclass
class PhotoRunOutcome:
    #: Index-aligned with `image_paths`, always full length. Alignment is what
    #: keeps checkpoint filenames (`image_{idx:04d}.json`) meaningful.
    results: List[Any] = field(default_factory=list)
    attempted: int = 0
    skipped: int = 0
    reused: int = 0
    first_failure: Optional[FailureDescriptor] = None
    occurrences: Dict[str, int] = field(default_factory=dict)

    @property
    def occurrence_count(self) -> int:
        if self.first_failure is None:
            return 0
        return self.occurrences.get(self.first_failure.fingerprint(), 1)


def is_aborted(result: Any) -> bool:
    return getattr(result, "error_kind", None) == ABORTED_ERROR_KIND


async def run_photo_passes(
    image_paths: Sequence[Path],
    *,
    concurrency: int,
    analyze_one: Callable[[int, Path], Awaitable[Any]],
    make_failed: Callable[[int, Path, BaseException], Any],
    make_aborted: Callable[[int, Path], Any],
    cached_results: Optional[Mapping[int, Any]] = None,
    on_cached: Optional[Callable[[int, Any], None]] = None,
    on_result: Optional[Callable[[int, Any], None]] = None,
    fail_fast: bool = True,
) -> PhotoRunOutcome:
    """
    Run `analyze_one` across `image_paths`, stopping early on the first failure.

    `analyze_one` may raise; the exception is classified here so every entrypoint
    gets the same taxonomy. `make_failed` and `make_aborted` build the caller's
    result type (the runner stays agnostic of `ImageResult` to avoid importing
    the entrypoints that import it).

    `on_result` fires for every freshly-computed result, successful or not — it is
    where callers checkpoint and emit progress. It does *not* fire for cache hits;
    `on_cached` covers those.

    `fail_fast=False` reproduces the old attempt-everything behavior, which
    `failure_mode="collect"` and the benchmark tooling depend on.
    """
    failfast = PhotoFailFast()
    cached = cached_results or {}
    sem = asyncio.Semaphore(max(1, concurrency))
    total = len(image_paths)
    results: List[Optional[Any]] = [None] * total
    counters = {"attempted": 0, "skipped": 0, "reused": 0}

    async def _guarded(idx: int, image_path: Path) -> None:
        # Resume path: a prior attempt already completed this photo. Cache hits
        # are never gated — replaying finished work costs nothing and keeps the
        # progress percentage moving during a resume.
        cached_result = cached.get(idx)
        if cached_result is not None and not getattr(cached_result, "error", None):
            results[idx] = cached_result
            counters["reused"] += 1
            if on_cached is not None:
                on_cached(idx, cached_result)
            return

        # Gate 1 — never scheduled. The common case once a run has tripped.
        if fail_fast and failfast.tripped:
            results[idx] = make_aborted(idx, image_path)
            counters["skipped"] += 1
            return

        async with sem:
            # Gate 2 — was queued behind the semaphore when the run tripped. This
            # is the coroutine that would otherwise have issued a doomed request.
            if fail_fast and failfast.tripped:
                results[idx] = make_aborted(idx, image_path)
                counters["skipped"] += 1
                return

            counters["attempted"] += 1
            try:
                result = await analyze_one(idx, image_path)
            except Exception as exc:  # noqa: BLE001 — classified, then recorded
                failfast.trip(classify_failure(exc))
                result = make_failed(idx, image_path, exc)
            else:
                # `analyze_one` may report failure by returning rather than
                # raising; treat both identically so neither can slip past.
                error = getattr(result, "error", None)
                if error and not is_aborted(result):
                    failfast.trip(_descriptor_for_result(result, error))

            results[idx] = result

        # Outside the semaphore: checkpointing and progress emission must not
        # occupy a concurrency slot.
        if on_result is not None:
            on_result(idx, result)

    # gather() must still return for every index — a raise here would leave holes
    # in `results` and desynchronize checkpoint filenames from image_paths.
    await asyncio.gather(*(_guarded(i, p) for i, p in enumerate(image_paths)))

    return PhotoRunOutcome(
        results=[r for r in results if r is not None],
        attempted=counters["attempted"],
        skipped=counters["skipped"],
        reused=counters["reused"],
        first_failure=failfast.first,
        occurrences=dict(failfast.occurrences),
    )


def _descriptor_for_result(result: Any, error: str) -> FailureDescriptor:
    """
    Classify a failure that arrived as a returned result rather than an exception.

    `analyze_one` implementations that catch internally lose the exception object,
    so the pass/stage context on the result is the best signal available.
    """
    from tools.failure_taxonomy import descriptor_from_error_text

    scene_data = getattr(result, "scene_data", None) or {}
    pass_errors = {}
    if isinstance(scene_data, dict):
        pass_errors = (scene_data.get("debug") or {}).get("pass_errors") or {}
    if isinstance(pass_errors, dict) and pass_errors:
        pass_key, detail = next(iter(pass_errors.items()))
        if isinstance(detail, dict):
            return FailureDescriptor(
                category=_category_for_stage(detail.get("stage")),
                code=str(detail.get("code") or "UNCLASSIFIED"),
                message=str(detail.get("message") or error)[:300],
                pass_key=str(pass_key),
                stage=detail.get("stage"),
                provider=detail.get("provider"),
                model=detail.get("model"),
            )
    return descriptor_from_error_text(error)


def _category_for_stage(stage: Optional[str]) -> str:
    from tools.failure_taxonomy import DEFAULT_CATEGORY, _STAGE_CATEGORY

    return _STAGE_CATEGORY.get(stage or "", DEFAULT_CATEGORY)


__all__ = [
    "ABORTED_ERROR_KIND",
    "ABORTED_MESSAGE",
    "PhotoFailFast",
    "PhotoRunOutcome",
    "is_aborted",
    "run_photo_passes",
]
