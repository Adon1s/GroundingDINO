"""
tools/room_surrogates.py

Pure deterministic room-surrogate clustering for renovation_estimate_v4.

PR 2 of a multi-PR refactor that moves the estimator from a flat line-item
view (v3) to a room-aware, package-aware rehab model (v4). This module is
not yet wired into the estimator — later PRs will use these surrogates to
stamp room_surrogate_id onto v4 issue copies and drive package inference.

Algorithm: a single-active-surrogate state machine over photos sorted by
listing order. Breaking scenes (kitchen, bathroom, bedroom, etc.) open or
extend a surrogate; non-breaking scenes (hallway, closet, exterior_*, etc.)
preserve the active surrogate so a `bath -> hallway -> bath` sequence
stays one bathroom_1.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.pipeline_common import SCENE_TO_GROUP_UI


BREAKING_SCENES = frozenset({
    "kitchen",
    "bathroom",
    "bedroom",
    "living_room",
    "dining_room",
    "laundry_room",
    "garage",
    "basement",
    "home_office",
    "attic",
    "pantry",
})

NON_BREAKING_SCENES = frozenset({
    "hallway",
    "stairway",
    "closet",
    "yard",
    "patio",
    "deck",
    "balcony",
    "driveway",
    "pool",
    "garden",
    "roof",
    "floor_plan",
    "aerial_view",
    "street_view",
    "unknown",
    "other",
    # hvac is a canonical Pass 1a label that the PR description omits; treat
    # as non-breaking since it is typically equipment closeup inside another
    # room (basement / garage / closet).
    "hvac",
})

CLUSTERING_METHOD = "single_active_surrogate_v1"


def _is_breaking(scene: str) -> bool:
    return scene in BREAKING_SCENES


def _is_unrecognized(scene: str) -> bool:
    """True when scene is non-falsy but not in any known set."""
    if not scene:
        return False
    if scene in BREAKING_SCENES or scene in NON_BREAKING_SCENES:
        return False
    if scene.startswith("exterior_"):
        return False
    return True


def _sort_key(item):
    photo_key, photo = item
    idx = ((photo or {}).get("photo") or {}).get("index")
    if idx is None:
        return (1, 0, photo_key)
    return (0, idx, photo_key)


def build_room_surrogates(photos: Dict[str, Any]) -> Dict[str, Any]:
    """Cluster photos into single-active-surrogate room records.

    Args:
        photos: photo_intel-style mapping ``{photo_key: photo_record}``.
            Each record may carry ``photo["photo"]["index"]`` (1-based
            listing order) and ``photo["scene"]["id"]`` (canonical scene
            label). Both fields are read defensively.

    Returns:
        ``{
            "photo_key_to_room_surrogate_id": {photo_key: surrogate_id, ...},
            "room_surrogates": [room_surrogate_record, ...],
        }``

        Only photos whose scene is in ``BREAKING_SCENES`` get a surrogate
        ID. Surrogates are listed in the order they were opened.
    """
    counters: Dict[str, int] = {}
    seen_closed: set = set()
    active: Optional[Dict[str, Any]] = None
    photo_key_to_id: Dict[str, str] = {}
    surrogates: List[Dict[str, Any]] = []

    for photo_key, photo in sorted(photos.items(), key=_sort_key):
        scene = (((photo or {}).get("scene") or {}).get("id")) or ""
        idx = ((photo or {}).get("photo") or {}).get("index")

        if not _is_breaking(scene):
            if active is not None and _is_unrecognized(scene):
                note = f"unrecognized_scene:{scene}"
                if note not in active["notes"]:
                    active["notes"].append(note)
            continue

        if active is not None and active["scene"] == scene:
            active["photo_keys"].append(photo_key)
            if idx is not None:
                active["listing_order_end"] = idx
            photo_key_to_id[photo_key] = active["room_surrogate_id"]
            continue

        if active is not None:
            seen_closed.add(active["scene"])
            active = None

        counters[scene] = counters.get(scene, 0) + 1
        sid = f"{scene}_{counters[scene]}"
        new_surrogate: Dict[str, Any] = {
            "room_surrogate_id": sid,
            "scene": scene,
            "scene_group": SCENE_TO_GROUP_UI.get(scene, "other"),
            "photo_keys": [photo_key],
            "listing_order_start": idx,
            "listing_order_end": idx,
            "clustering_method": CLUSTERING_METHOD,
            "notes": [],
        }
        if scene in seen_closed:
            new_surrogate["notes"].append("split_after_intervening_room")
        surrogates.append(new_surrogate)
        active = new_surrogate
        photo_key_to_id[photo_key] = sid

    return {
        "photo_key_to_room_surrogate_id": photo_key_to_id,
        "room_surrogates": surrogates,
    }
