"""Atomic per-estimate-unit Terra review checkpoints.

Stored beneath the existing run checkpoint directory
(<artifacts_root>/<property_key>/.checkpoints/<stable_run_id>/renovation_architecture/)
so a retry after a Terra failure or budget denial reuses completed units
instead of re-buying them. The request fingerprint covers everything that
shapes the call (projection, prompt/model config, condition payload, sent
image hashes); any drift silently invalidates the checkpoint and the unit is
reviewed fresh. Files ride the server's run-checkpoint lifecycle: cleared on
full job success, rmtree'd on an image-policy change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.comparison_common import atomic_json
from tools.pipeline_common import stable_hash_id

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "terra_condition_review_v1"


def terra_checkpoint_dir(
    artifacts_root: Path, property_key: str, run_id: str
) -> Path:
    return (
        Path(artifacts_root) / property_key / ".checkpoints" / run_id
        / "renovation_architecture"
    )


def unit_checkpoint_path(directory: Path, estimate_unit_id: str) -> Path:
    # Hashed filename: unit ids are catalog-shaped strings today but the
    # fallback path can carry arbitrary room hints; hashing keeps every name
    # filesystem-safe and outside the server's image_*.json glob.
    return directory / (
        f"terra_unit_{stable_hash_id('terra_unit', estimate_unit_id, length=16)}.json"
    )


def save_unit_checkpoint(
    path: Path,
    *,
    estimate_unit_id: str,
    request_fingerprint: str,
    terra_call: Dict[str, Any],
    reviews: List[Dict[str, Any]],
    created_at: str,
) -> None:
    atomic_json(path, {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "estimate_unit_id": estimate_unit_id,
        "request_fingerprint": request_fingerprint,
        "created_at": created_at,
        "terra_call": terra_call,
        "reviews": reviews,
    })


def load_unit_checkpoint(
    path: Path, *, request_fingerprint: str, condition_ids: List[str]
) -> Optional[Dict[str, Any]]:
    """Return {terra_call, reviews} when the checkpoint is intact and covers
    exactly the expected conditions under the same fingerprint; None means a
    fresh call (a corrupt or stale checkpoint is not an error)."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return None
    if payload.get("kind") != CHECKPOINT_KIND:
        return None
    if payload.get("request_fingerprint") != request_fingerprint:
        return None
    terra_call = payload.get("terra_call")
    reviews = payload.get("reviews")
    if not isinstance(terra_call, dict) or not isinstance(reviews, list):
        return None
    reviewed = {
        review.get("condition_id")
        for review in reviews
        if isinstance(review, dict)
    }
    if reviewed != set(condition_ids):
        return None
    return {"terra_call": terra_call, "reviews": reviews}
