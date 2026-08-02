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

Exception: exterior photos additionally collapse into one property-level
surrogate appended after the state machine (see `_append_exterior_surrogate`).
They stay non-breaking for interior clustering, but the property needs a single
exterior identity for pricing and package inference.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.pipeline_common import (
    ALL_SCENE_IDS,
    BREAKING_SCENES,
    NON_BREAKING_SCENES,
    SCENE_TO_GROUP_UI,
)

CLUSTERING_METHOD = "single_active_surrogate_v1"
EXTERIOR_CLUSTERING_METHOD = "property_level_exterior_v1"

# The whole property gets exactly one exterior identity. Exterior scenes are
# non-breaking so they never open a per-scene surrogate (that is deliberate —
# an exterior shot between two bathroom shots must not split the bathroom), but
# without any surrogate every exterior issue also lands in its own estimate
# unit, so exterior evidence can never corroborate itself into a package.
#
# The id must not be the bare group token "exterior": both extraction lanes run
# stamped ids through renovation_estimate._meaningful_scope_hint, whose generic
# denylist contains "exterior" and would silently strip it back to "".
EXTERIOR_SURROGATE_ID = "exterior_primary"
EXTERIOR_SURROGATE_SCENE = "exterior"
EXTERIOR_SCENE_GROUP = "exterior"


def _is_breaking(scene: str) -> bool:
    return scene in BREAKING_SCENES


def _is_unrecognized(scene: str) -> bool:
    """True when scene is non-falsy but not a canonical scene id."""
    return bool(scene) and scene not in ALL_SCENE_IDS


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

        Photos whose scene is in ``BREAKING_SCENES`` get a per-room surrogate
        ID, listed in the order they were opened. Exterior-group photos get the
        single property-level ``EXTERIOR_SURROGATE_ID``, appended last. All
        other non-breaking photos get no surrogate.
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

    _append_exterior_surrogate(photos, surrogates, photo_key_to_id)

    return {
        "photo_key_to_room_surrogate_id": photo_key_to_id,
        "room_surrogates": surrogates,
    }


def _append_exterior_surrogate(
    photos: Dict[str, Any],
    surrogates: List[Dict[str, Any]],
    photo_key_to_id: Dict[str, str],
) -> None:
    """Add the single property-level exterior surrogate, if any exterior photos.

    Appended after the state machine has run so it can never become the active
    surrogate and interfere with interior clustering. Downstream, estimate_units
    has no special case for it: `_surrogate_unit_type` reads ``scene`` and the
    fallback branch mints an estimate unit named after the surrogate id.
    """
    exterior_photo_keys = [
        photo_key
        for photo_key, photo in sorted(photos.items(), key=_sort_key)
        if SCENE_TO_GROUP_UI.get(
            (((photo or {}).get("scene") or {}).get("id")) or ""
        ) == EXTERIOR_SCENE_GROUP
    ]
    if not exterior_photo_keys:
        return

    indexes = [
        idx
        for idx in (
            ((photos.get(key) or {}).get("photo") or {}).get("index")
            for key in exterior_photo_keys
        )
        if idx is not None
    ]
    surrogates.append({
        "room_surrogate_id": EXTERIOR_SURROGATE_ID,
        "scene": EXTERIOR_SURROGATE_SCENE,
        "scene_group": EXTERIOR_SCENE_GROUP,
        "photo_keys": exterior_photo_keys,
        "listing_order_start": min(indexes) if indexes else None,
        "listing_order_end": max(indexes) if indexes else None,
        "clustering_method": EXTERIOR_CLUSTERING_METHOD,
        "notes": ["property_level_exterior_identity"],
    })
    for photo_key in exterior_photo_keys:
        photo_key_to_id[photo_key] = EXTERIOR_SURROGATE_ID
