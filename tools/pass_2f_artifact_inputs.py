"""Read-only helpers for reconstructing Pass 2f inputs from saved artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.renovation_estimate import (
    EstimateCandidate,
    extract_estimate_candidates,
    resolve_estimate_units,
)


def resolve_artifact_path(run_path: Path) -> Path:
    """Resolve a run directory or a direct ``photo_intel.json`` path."""
    if run_path.is_file():
        return run_path
    if run_path.is_dir():
        candidate = run_path / "photo_intel.json"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"directory does not contain photo_intel.json: {run_path}"
        )
    raise FileNotFoundError(f"path does not exist: {run_path}")


def select_issues_flat(
    artifact: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Return the persisted estimate issue list used to rebuild candidates."""
    for key in ("estimate_issues_flat", "issues_flat"):
        value = artifact.get(key)
        if isinstance(value, list) and value:
            return value
    return None


def photo_key_to_path(artifact: Dict[str, Any]) -> Dict[str, Path]:
    """Build the persisted photo-key to image-path mapping without writing."""
    photos = artifact.get("photos") or {}
    if isinstance(photos, dict):
        records = photos.items()
    elif isinstance(photos, list):
        records = [(None, item) for item in photos]
    else:
        return {}

    out: Dict[str, Path] = {}
    for fallback_key, record in records:
        if not isinstance(record, dict):
            continue
        photo = record.get("photo") or {}
        if not isinstance(photo, dict):
            continue
        image_path = photo.get("image_path")
        photo_key = photo.get("photo_key") or fallback_key
        if image_path and photo_key:
            out[str(photo_key)] = Path(str(image_path))
    return out


def prepare_replay_inputs(
    artifact: Dict[str, Any],
    catalog: Dict[str, Any],
) -> Tuple[
    Optional[List[Dict[str, Any]]],
    List[EstimateCandidate],
    Dict[str, Path],
]:
    """Rebuild estimate candidates and the photo map from persisted inputs."""
    issues_flat = select_issues_flat(artifact)
    if issues_flat is None:
        return None, [], {}
    candidates = extract_estimate_candidates(issues_flat, catalog)
    candidates = resolve_estimate_units(candidates, issues_flat, catalog)
    return issues_flat, candidates, photo_key_to_path(artifact)
