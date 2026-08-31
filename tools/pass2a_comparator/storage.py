"""On-disk layout, fingerprints and checkpoints for the Pass 2a comparator.

Everything lives under `benchmarks/pass2a-prompt/runs/comparator/`, which that
benchmark's own .gitignore (`runs/`) already covers. The benchmark's dataset,
gold and existing run artifacts are read-only here and are never written.

Baselines are content-addressed by their fingerprint, so a changed fingerprint
lands in a new directory instead of overwriting paid evidence. Git head and
timestamps go in a sibling info.json, never in the compared dict - harness
commits must not orphan paid artifacts (benchmark_pass2a_packages.py:648).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tools.comparison_common import (
    atomic_json,
    sha256_bytes,
    sha256_canonical,
    sha256_file,
)
from tools.pass2a_comparator import config as cc


class ComparatorDataError(RuntimeError):
    """Dataset, gold or stored-run inputs are missing or inconsistent."""


# ---------------------------------------------------------------------------
# Dataset + gold (read-only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Photo:
    property_key: str
    photo_key: str
    image_sha256: str
    scene: str

    @property
    def key(self) -> str:
        """Gold key and stable photo identity: `<property_key>/<photo_key>`."""
        return f"{self.property_key}/{self.photo_key}"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ComparatorDataError(f"missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(manifest_path: Optional[Path] = None) -> Tuple[List["Photo"], Path]:
    """Return the benchmark photos in a stable order, plus their images root.

    Sorted by (property_key, photo_key) so blinding and checkpoint identity
    survive any reordering of the manifest.
    """
    manifest = _read_json(manifest_path or cc.MANIFEST_PATH)
    images_root = Path(manifest["images_root"])
    photos: List[Photo] = []
    for property_key in sorted(manifest.get("properties") or {}):
        entry = manifest["properties"][property_key]
        for photo in entry.get("photos") or []:
            photos.append(
                Photo(
                    property_key=property_key,
                    photo_key=photo["photo_key"],
                    image_sha256=photo["image_sha256"],
                    scene=photo.get("scene") or "",
                )
            )
    if not photos:
        raise ComparatorDataError(f"manifest declares no photos: {manifest_path}")
    photos.sort(key=lambda p: (p.property_key, p.photo_key))
    return photos, images_root


def image_path(photo: "Photo", images_root: Path) -> Path:
    return images_root / photo.property_key / photo.photo_key


def resolve_image(photo: "Photo", images_root: Path) -> Path:
    """Path to a photo, verified present and byte-identical to the manifest.

    A silently substituted image would invalidate a comparison without changing
    any fingerprint the operator can see, so this is checked at use time.
    """
    path = image_path(photo, images_root)
    if not path.is_file():
        raise ComparatorDataError(f"image missing for {photo.key}: {path}")
    actual = sha256_file(path)
    if actual != photo.image_sha256:
        raise ComparatorDataError(
            f"image hash mismatch for {photo.key}: manifest has "
            f"{photo.image_sha256[:12]}, file is {actual[:12]}"
        )
    return path


def load_gold(gold_path: Optional[Path] = None) -> Dict[str, List[Dict[str, str]]]:
    """Frozen human gold, keyed `<property>/<photo>` -> [{gold_id, condition}]."""
    gold = _read_json(gold_path or cc.GOLD_PATH)
    photos = gold.get("photos")
    if not isinstance(photos, dict) or not photos:
        raise ComparatorDataError(f"gold file has no photos map: {gold_path}")
    return photos


def gold_for(
    gold: Dict[str, List[Dict[str, str]]], photo: "Photo"
) -> List[Dict[str, str]]:
    """Gold conditions for one photo. Absence is an error, not an empty review."""
    if photo.key not in gold:
        raise ComparatorDataError(f"no gold entry for {photo.key}")
    return gold[photo.key]


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def compute_fingerprint(photos: List["Photo"], model: str) -> Dict[str, Any]:
    """Everything that must be identical for a baseline to stay reusable.

    Whole-file source hashes are deliberate (Steven, 2026-08-30): any edit to
    the Pass 2a call path requires a fresh baseline rather than risking a silent
    behaviour change under a reused one. Git head is NOT here - it belongs in
    info.json, so an unrelated commit cannot orphan 45 paid calls.
    """
    system_prompt, user_prompt = cc.production_prompts()
    fingerprint = {
        "schema_version": cc.COMPARATOR_SCHEMA_VERSION,
        "image_hashes_sha": sha256_canonical({p.key: p.image_sha256 for p in photos}),
        "production_system_prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")),
        "production_user_prompt_sha256": sha256_bytes(user_prompt.encode("utf-8")),
        "model": model,
        "reasoning_effort": cc.REASONING_EFFORT,
        "max_output_tokens": cc.MAX_OUTPUT_TOKENS,
        "image_detail": cc.IMAGE_DETAIL,
        "repeats": cc.REPEATS,
        "stub": cc.stub_mode(),
    }
    for source in cc.PASS_PATH_SOURCES:
        fingerprint[f"{source.stem}_sha256"] = sha256_file(source)
    return fingerprint


def baseline_id(fingerprint: Dict[str, Any]) -> str:
    return "b_" + sha256_canonical(fingerprint)[:12]


def fingerprint_diff(stored: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Keys that differ, in the `guard_fingerprint` style, for the operator."""
    return sorted(
        key for key in set(stored) | set(current)
        if stored.get(key) != current.get(key)
    )


# ---------------------------------------------------------------------------
# Directories, call units, checkpoints
# ---------------------------------------------------------------------------

def baseline_dir(bid: str) -> Path:
    return cc.BASELINES_DIR / bid


def experiment_dir(exp_id: str) -> Path:
    return cc.EXPERIMENTS_DIR / exp_id


def call_path(base_dir: Path, photo: "Photo", rep: int) -> Path:
    return base_dir / "outputs" / photo.property_key / f"{photo.photo_key}__rep{rep}.json"


def iter_calls(
    photos: List["Photo"], repeats: int = cc.REPEATS
) -> Iterator[Tuple["Photo", int]]:
    """(photo, rep) in run order: rep-major, matching `run_matrix`."""
    for rep in range(1, repeats + 1):
        for photo in photos:
            yield photo, rep


def expected_call_count(photos: List["Photo"], repeats: int = cc.REPEATS) -> int:
    return len(photos) * repeats


def load_call(path: Path) -> Optional[Dict[str, Any]]:
    """A stored call, or None when absent or unreadable (treated as missing)."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def call_is_reusable(
    record: Optional[Dict[str, Any]],
    *,
    system_sha: str,
    user_sha: str,
    image_sha: str,
    scene: Optional[str] = None,
) -> bool:
    """A completed call is reused only if these exact inputs produced it.

    Errored and partial records are never reused, so a resume re-runs exactly
    the missing and failed calls. The prompt hashes are of the TEMPLATE, so a
    scene-conditional prompt also has to match the scene it was rendered with.
    """
    if not record or record.get("status") != "ok":
        return False
    if not (record.get("text") or "").strip():
        return False
    if scene is not None and record.get("scene") != scene:
        return False
    return (
        record.get("system_prompt_sha256") == system_sha
        and record.get("user_prompt_sha256") == user_sha
        and record.get("image_sha256") == image_sha
    )


def write_call(path: Path, record: Dict[str, Any]) -> None:
    atomic_json(path, record)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str:
    """Provenance only. Recorded in info.json, never in a compared dict."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cc.ROOT), capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_info(base_dir: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    atomic_json(
        base_dir / "info.json",
        {"git_head": git_head(), "written_at": now_iso(), **(extra or {})},
    )


# ---------------------------------------------------------------------------
# Baseline + experiment records
# ---------------------------------------------------------------------------

def ensure_baseline(fingerprint: Dict[str, Any]) -> Tuple[str, Path, bool]:
    """Return (baseline_id, dir, created). Never overwrites an existing set."""
    bid = baseline_id(fingerprint)
    path = baseline_dir(bid)
    fp_path = path / "fingerprint.json"
    if fp_path.is_file():
        stored = _read_json(fp_path)
        if stored != fingerprint:  # content-addressed, so this means a collision
            raise ComparatorDataError(
                f"baseline {bid} fingerprint does not match its own id "
                f"(differs on: {fingerprint_diff(stored, fingerprint)})"
            )
        return bid, path, False
    atomic_json(fp_path, fingerprint)
    write_info(path, {"kind": "baseline", "baseline_id": bid})
    return bid, path, True


def find_baseline(fingerprint: Dict[str, Any]) -> Optional[Path]:
    """The stored baseline for this fingerprint, if one exists. Read-only."""
    path = baseline_dir(baseline_id(fingerprint))
    return path if (path / "fingerprint.json").is_file() else None


def stale_baselines(fingerprint: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """Existing baselines that no longer match, with the keys that differ.

    Surfaced so a source edit reads as `new baseline required` rather than as
    data loss - the whole-file source hashes make that a routine event.
    """
    current_id = baseline_id(fingerprint)
    out: List[Tuple[str, List[str]]] = []
    if not cc.BASELINES_DIR.is_dir():
        return out
    for path in sorted(cc.BASELINES_DIR.iterdir()):
        if path.name == current_id or not (path / "fingerprint.json").is_file():
            continue
        try:
            stored = _read_json(path / "fingerprint.json")
        except json.JSONDecodeError:
            continue
        out.append((path.name, fingerprint_diff(stored, fingerprint)))
    return out


def progress(
    path: Path, photos: List["Photo"], system_sha: str, user_sha: str,
    scene_conditional: bool = False,
) -> int:
    """How many of a side's calls are complete and reusable under these prompts.

    Scene only participates when the prompt actually renders it. A prompt
    without {scene} produces the same text for every scene, so keying reuse on
    it would orphan paid calls for no behavioural reason.
    """
    return sum(
        1 for photo, rep in iter_calls(photos)
        if call_is_reusable(
            load_call(call_path(path, photo, rep)),
            system_sha=system_sha, user_sha=user_sha,
            image_sha=photo.image_sha256,
            scene=photo.scene if scene_conditional else None,
        )
    )


def baseline_progress(
    path: Path, photos: List["Photo"], fingerprint: Dict[str, Any]
) -> int:
    return progress(path, photos,
                    fingerprint["production_system_prompt_sha256"],
                    fingerprint["production_user_prompt_sha256"],
                    scene_conditional=cc.uses_scene(*cc.production_prompts()))


def experiment_progress(
    path: Path, photos: List["Photo"], experiment: Dict[str, Any]
) -> int:
    return progress(path, photos,
                    experiment["candidate_system_prompt_sha256"],
                    experiment["candidate_user_prompt_sha256"],
                    scene_conditional=bool(experiment.get("scene_conditional")))


def new_experiment_id(system_prompt: str, user_prompt: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    digest = sha256_bytes((system_prompt + "\x00" + user_prompt).encode("utf-8"))
    return f"e_{stamp}_{digest[:8]}"


def create_experiment(
    *,
    system_prompt: str,
    user_prompt: str,
    display_name: str,
    baseline_id_value: str,
    fingerprint: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Tuple[str, Path]:
    """Persist an immutable candidate prompt pair and its runtime."""
    if not system_prompt.strip() or not user_prompt.strip():
        raise ComparatorDataError(
            "both candidate prompts must be non-blank before an experiment is created"
        )
    # Fail here, not on call 1 of 45 after the baseline is already paid for.
    cc.validate_prompt_placeholders(system_prompt, "candidate system prompt")
    cc.validate_prompt_placeholders(user_prompt, "candidate user prompt")
    scene_conditional = cc.uses_scene(system_prompt, user_prompt)
    exp_id = new_experiment_id(system_prompt, user_prompt)
    path = experiment_dir(exp_id)
    if (path / "experiment.json").is_file():
        raise ComparatorDataError(f"experiment {exp_id} already exists")
    atomic_json(
        path / "experiment.json",
        {
            "experiment_id": exp_id,
            "display_name": display_name.strip() or exp_id,
            "created_at": now_iso(),
            "baseline_id": baseline_id_value,
            "fingerprint": fingerprint,
            # Stored verbatim: the exact text is the experiment.
            "candidate_system_prompt": system_prompt,
            "candidate_user_prompt": user_prompt,
            "candidate_system_prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")),
            "candidate_user_prompt_sha256": sha256_bytes(user_prompt.encode("utf-8")),
            "runtime": {
                "model": model_config.get("model"),
                "reasoning_effort": model_config.get("reasoning_effort"),
                "max_output_tokens": model_config.get("max_output_tokens"),
                "image_detail": cc.IMAGE_DETAIL,
                "repeats": cc.REPEATS,
                "concurrency": cc.CONCURRENCY,
            },
            "scene_conditional": scene_conditional,
            "production_equivalent": False,
            "production_equivalent_reason": cc.PRODUCTION_EQUIVALENT_REASON,
        },
    )
    write_info(path, {"kind": "experiment", "experiment_id": exp_id})
    return exp_id, path


def load_experiment(exp_id: str) -> Dict[str, Any]:
    return _read_json(experiment_dir(exp_id) / "experiment.json")


def list_experiments() -> List[Dict[str, Any]]:
    """Newest first, by directory name (the id embeds a UTC timestamp)."""
    if not cc.EXPERIMENTS_DIR.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(cc.EXPERIMENTS_DIR.iterdir(), reverse=True):
        try:
            out.append(_read_json(path / "experiment.json"))
        except (ComparatorDataError, json.JSONDecodeError):
            continue
    return out
