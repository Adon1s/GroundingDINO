"""Generate Pass 2a outputs for one comparator side. Nothing downstream runs.

This module issues the only paid calls in the comparator. It calls
`scene_classifier_passes.run_pass_2a` directly - not the orchestrator - so no
Pass 1a, no Pass 2b-2f, no catalog resolution and no costing can be reached from
here. Pass 2a ignores its `context` argument, so there is genuinely nothing
upstream to supply.

Run (normally launched by the Streamlit app, which adds the budget-guard env):

    .venv\\Scripts\\python.exe -m tools.pass2a_comparator.runner --experiment <id>

Progress is line-delimited JSON on stdout so a parent process can follow it
without parsing logs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.comparison_common import sha256_bytes  # noqa: E402
from tools.pass2a_comparator import config as cc  # noqa: E402
from tools.pass2a_comparator import storage as store  # noqa: E402

#: Defined in config so the fingerprint can see it; re-exported for callers.
STUB_ENV = cc.STUB_ENV

BASELINE_STAGE = "baseline"
CANDIDATE_STAGE = "candidate"


class ComparatorRunError(RuntimeError):
    """Generation stopped. Checkpoints are intact; rerun to resume."""


@dataclass
class CallOutcome:
    """One (photo, repeat) result.

    `error` and `error_kind` are the attribute names `run_photo_passes` reads,
    so this plugs into the shared runner without adapting it.
    """
    photo: store.Photo
    rep: int
    text: str = ""
    error: Optional[str] = None
    error_kind: Optional[str] = None
    duration_s: float = 0.0


def emit(event: Dict[str, Any]) -> None:
    """One JSON object per line on stdout, flushed - this is the wire protocol."""
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Run lock - concurrent-start rejection
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock:
    """Exclusive per-experiment lock, taken over only when its owner is dead.

    A crashed run must not block every later attempt, but a live one must not be
    joined by a second writer - two processes checkpointing the same call would
    double-bill it.
    """

    def __init__(self, path: Path):
        self.path = path
        self._held = False

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = store.load_call(self.path) or {}
            pid = int(owner.get("pid") or 0)
            if _pid_alive(pid):
                raise ComparatorRunError(
                    f"a run is already active for this experiment (pid {pid}). "
                    f"Cancel it before starting another."
                )
            self.path.unlink(missing_ok=True)
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "started_at": store.now_iso()}, fh)
        self._held = True
        return self

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class _StubVLMClient:
    """Deterministic offline client for the mocked walkthrough. Zero tokens."""

    def __init__(self) -> None:
        self.budget_context: Dict[str, Any] = {}
        self.calls: List[Dict[str, Any]] = []

    async def analyze_image(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        digest = sha256_bytes(
            (str(kwargs.get("image_path")) + kwargs["system_prompt"] + kwargs["user_prompt"])
            .encode("utf-8")
        )[:8]
        return (
            f"STUB OUTPUT {digest} - offline stub, no model was called.\n\n"
            f"- system: {kwargs['system_prompt'][:60]}\n"
            f"- user: {kwargs['user_prompt'][:60]}\n"
            f"- image: {Path(str(kwargs.get('image_path'))).name}"
        )


def build_client(source_run_id: str, property_key: str = "") -> Any:
    from tools.vlm_client import create_vlm_client

    client = create_vlm_client()
    client.budget_context = {
        "property_key": property_key or "pass2a_comparator",
        "source_run_id": source_run_id,
    }
    return client


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_side(
    *,
    photos: List[store.Photo],
    images_root: Path,
    out_dir: Path,
    system_prompt: str,
    user_prompt: str,
    model_config: Dict[str, Any],
    stage: str,
    client_factory: Callable[[str], Any],
    repeats: int = cc.REPEATS,
    concurrency: int = cc.CONCURRENCY,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Fill `out_dir` with one output per (photo, repeat). Idempotent.

    Already-complete calls whose prompts and image hash still match are reused,
    so a resume issues exactly the missing and previously failed calls.
    """
    from tools.photo_pass_runner import run_photo_passes
    from tools.scene_classifier_passes import run_pass_2a

    # Hashes are of the TEMPLATE; the rendered text varies by scene, and
    # call_is_reusable compares the scene separately.
    system_sha = sha256_bytes(system_prompt.encode("utf-8"))
    user_sha = sha256_bytes(user_prompt.encode("utf-8"))
    scene_conditional = cc.uses_scene(system_prompt, user_prompt)
    total = store.expected_call_count(photos, repeats)
    stats = {"stage": stage, "total": total, "done": 0, "reused": 0, "called": 0}

    def _progress(**extra: Any) -> None:
        if on_progress is not None:
            on_progress({"type": "progress", **stats, **extra})

    by_property: Dict[str, List[store.Photo]] = {}
    for photo in photos:
        by_property.setdefault(photo.property_key, []).append(photo)

    for rep in range(1, repeats + 1):
        for property_key in sorted(by_property):
            batch = by_property[property_key]
            client = client_factory(property_key)

            cached: Dict[int, CallOutcome] = {}
            for index, photo in enumerate(batch):
                record = store.load_call(store.call_path(out_dir, photo, rep))
                if store.call_is_reusable(
                    record,
                    system_sha=system_sha,
                    user_sha=user_sha,
                    image_sha=photo.image_sha256,
                    scene=photo.scene if scene_conditional else None,
                ):
                    cached[index] = CallOutcome(
                        photo=photo, rep=rep, text=record["text"]
                    )

            async def analyze_one(index: int, _path: Path, _rep: int = rep,
                                  _batch: List[store.Photo] = batch,
                                  _client: Any = client) -> CallOutcome:
                photo = _batch[index]
                started = time.monotonic()
                result = await run_pass_2a(
                    store.resolve_image(photo, images_root),
                    _client,
                    model_config,
                    # Rendered from the benchmark's frozen Pass 1a capture, so
                    # a scene-conditional prompt still costs zero 1a calls.
                    system_prompt=cc.render_prompt(system_prompt, photo.scene),
                    user_prompt=cc.render_prompt(user_prompt, photo.scene),
                )
                return CallOutcome(
                    photo=photo,
                    rep=_rep,
                    text=result.observations_freeform,
                    duration_s=round(time.monotonic() - started, 3),
                )

            def make_failed(index: int, _path: Path, exc: BaseException,
                            _rep: int = rep,
                            _batch: List[store.Photo] = batch) -> CallOutcome:
                return CallOutcome(
                    photo=_batch[index], rep=_rep,
                    error=f"{type(exc).__name__}: {exc}"[:500], error_kind="failed",
                )

            def make_aborted(index: int, _path: Path, _rep: int = rep,
                             _batch: List[store.Photo] = batch) -> CallOutcome:
                return CallOutcome(
                    photo=_batch[index], rep=_rep,
                    error="aborted: run stopped after an earlier call failed",
                    error_kind="aborted",
                )

            def on_result(_index: int, outcome: CallOutcome) -> None:
                if outcome.error_kind == "aborted":
                    return  # never issued a request; leave the slot uncheckpointed
                store.write_call(
                    store.call_path(out_dir, outcome.photo, outcome.rep),
                    {
                        "property_key": outcome.photo.property_key,
                        "photo_key": outcome.photo.photo_key,
                        "rep": outcome.rep,
                        "status": "error" if outcome.error else "ok",
                        "text": outcome.text,
                        "error": outcome.error,
                        "duration_s": outcome.duration_s,
                        "model": model_config.get("model"),
                        "reasoning_effort": model_config.get("reasoning_effort"),
                        "max_output_tokens": model_config.get("max_output_tokens"),
                        "image_detail": cc.IMAGE_DETAIL,
                        "system_prompt_sha256": system_sha,
                        "user_prompt_sha256": user_sha,
                        "image_sha256": outcome.photo.image_sha256,
                        "scene": outcome.photo.scene,
                        "scene_conditional": scene_conditional,
                        "stub": cc.stub_mode(),
                        "completed_at": store.now_iso(),
                    },
                )
                if not outcome.error:
                    stats["done"] += 1
                    stats["called"] += 1
                _progress(photo=outcome.photo.key, rep=outcome.rep)

            def on_cached(_index: int, outcome: CallOutcome) -> None:
                stats["done"] += 1
                stats["reused"] += 1
                _progress(photo=outcome.photo.key, rep=outcome.rep)

            outcome = await run_photo_passes(
                [store.image_path(p, images_root) for p in batch],
                concurrency=concurrency,
                analyze_one=analyze_one,
                make_failed=make_failed,
                make_aborted=make_aborted,
                cached_results=cached,
                on_cached=on_cached,
                on_result=on_result,
                fail_fast=True,
            )
            if outcome.first_failure is not None:
                # Stop the whole side, not just this batch: with the budget
                # guard on, the first failure is usually the daily ceiling, and
                # the remaining calls would be refused anyway.
                raise ComparatorRunError(
                    f"{stage} stopped at {property_key} rep{rep}: "
                    f"{outcome.first_failure.message}. "
                    f"Checkpoints kept - rerun to resume."
                )

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_client_factory(
    label: str, client_factory: Optional[Callable[[str], Any]],
    on_progress: Optional[Callable[[Dict[str, Any]], None]],
) -> Callable[[str], Any]:
    """Real client, or the offline stub when the stub env var is set."""
    if client_factory is not None:
        return client_factory
    if cc.stub_mode():
        stub = _StubVLMClient()
        if on_progress is not None:
            on_progress({"type": "log", "message": "STUB MODE - no model called"})
        return lambda _property_key: stub
    cc.assert_budget_guard_live()
    return lambda property_key: build_client(f"pass2a_comparator/{label}", property_key)


def run_baseline(
    *,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    client_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Bring the baseline for the current fingerprint to completion.

    Standalone so a baseline can be created before any candidate exists. A
    changed fingerprint addresses a new directory, so earlier baselines are
    never overwritten.
    """
    photos, images_root = store.load_dataset()
    model_config = cc.build_model_config()
    system_prompt, user_prompt = cc.production_prompts()
    fingerprint = store.compute_fingerprint(photos, model_config["model"])
    bid, base_dir, _ = store.ensure_baseline(fingerprint)
    factory = resolve_client_factory(bid, client_factory, on_progress)
    if on_progress is not None:
        on_progress({"type": "stage", "stage": BASELINE_STAGE,
                     "total": store.expected_call_count(photos)})
    with RunLock(base_dir / "run.lock"):
        stats = asyncio.run(
            generate_side(
                photos=photos, images_root=images_root, out_dir=base_dir,
                system_prompt=system_prompt, user_prompt=user_prompt,
                model_config=model_config, stage=BASELINE_STAGE,
                client_factory=factory, on_progress=on_progress,
            )
        )
    return {"baseline_id": bid, BASELINE_STAGE: stats}


def run_experiment(
    exp_id: str,
    *,
    baseline_only: bool = False,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    client_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Bring the baseline and then the candidate to completion. Resumable."""
    photos, images_root = store.load_dataset()
    experiment = store.load_experiment(exp_id)
    fingerprint = experiment["fingerprint"]
    current = store.compute_fingerprint(photos, experiment["runtime"]["model"])
    if current != fingerprint:
        raise ComparatorRunError(
            "the Pass 2a call path changed since this experiment was created "
            f"(differs on: {store.fingerprint_diff(fingerprint, current)}). "
            "Create a fresh baseline and a new experiment."
        )

    model_config = cc.build_model_config()
    system_prompt, user_prompt = cc.production_prompts()
    exp_dir = store.experiment_dir(exp_id)
    bid, base_dir, _ = store.ensure_baseline(fingerprint)
    if bid != experiment["baseline_id"]:
        raise ComparatorRunError(
            f"experiment points at baseline {experiment['baseline_id']} but the "
            f"current fingerprint resolves to {bid}"
        )

    factory = resolve_client_factory(exp_id, client_factory, on_progress)

    def _stage(stage: str) -> None:
        if on_progress is not None:
            on_progress({"type": "stage", "stage": stage,
                         "total": store.expected_call_count(photos)})

    results: Dict[str, Any] = {}
    with RunLock(exp_dir / "run.lock"):
        _stage(BASELINE_STAGE)
        results[BASELINE_STAGE] = asyncio.run(
            generate_side(
                photos=photos, images_root=images_root, out_dir=base_dir,
                system_prompt=system_prompt, user_prompt=user_prompt,
                model_config=model_config,
                stage=BASELINE_STAGE, client_factory=factory,
                on_progress=on_progress,
            )
        )
        if not baseline_only:
            _stage(CANDIDATE_STAGE)
            results[CANDIDATE_STAGE] = asyncio.run(
                generate_side(
                    photos=photos, images_root=images_root, out_dir=exp_dir,
                    system_prompt=experiment["candidate_system_prompt"],
                    user_prompt=experiment["candidate_user_prompt"],
                    model_config=model_config,
                    stage=CANDIDATE_STAGE, client_factory=factory,
                    on_progress=on_progress,
                )
            )
    store.write_call(exp_dir / "state.json", {"finished_at": store.now_iso(), **results})
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Pass 2a comparator outputs.")
    parser.add_argument("--experiment")
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Fill the baseline set only; do not run the candidate.",
    )
    args = parser.parse_args(argv)
    if not args.experiment and not args.baseline_only:
        parser.error("--experiment is required unless --baseline-only is given")
    try:
        if args.experiment:
            results = run_experiment(
                args.experiment, baseline_only=args.baseline_only, on_progress=emit
            )
        else:
            results = run_baseline(on_progress=emit)
    except Exception as exc:  # noqa: BLE001 - reported on the wire, not raised
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1
    emit({"type": "done", **{k: v.get("done") for k, v in results.items()
                             if isinstance(v, dict)}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
