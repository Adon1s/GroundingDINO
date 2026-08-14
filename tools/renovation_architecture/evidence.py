"""Objective photo-evidence identity for the renovation architecture.

evidence_dedup_v1: exact duplicate = sha256 over normalized RGB pixel bytes
(so byte-level re-encodes of the same image collapse); near duplicate =
64-bit 8x8 average-hash within Hamming distance NEAR_HAMMING_MAX AND
max-channel mean-RGB delta <= NEAR_MEAN_RGB_DELTA_MAX (the AND gate carries
degenerate uniform-image hashes, which otherwise match everything).
duplicate_groups is the disjoint union-find closure of both families; one
stable representative per view class (the lexicographically-first photo key)
is what Terra sees, and the view-class count — never the filename count — is
what catalog min_photo_evidence gates. Changing any rule here means bumping
EVIDENCE_DEDUP_POLICY_VERSION so fingerprints and checkpoints invalidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    EVIDENCE_DEDUP_POLICY_VERSION,
    EvidenceFacts,
)
from tools.renovation_architecture.conditions import ConditionDraft
from tools.renovation_architecture.ids import make_evidence_id
from tools.scene_classifier_passes import PassExecutionError

NEAR_HAMMING_MAX = 6
NEAR_MEAN_RGB_DELTA_MAX = 16


@dataclass(frozen=True)
class PhotoIdentity:
    exact_sha256: str
    average_hash: int
    mean_rgb: Tuple[float, float, float]


@dataclass(frozen=True)
class PhotoDedup:
    exact_groups: Tuple[Tuple[str, ...], ...]
    near_groups: Tuple[Tuple[str, ...], ...]
    merged_groups: Tuple[Tuple[str, ...], ...]
    representatives: Tuple[str, ...]
    distinct_view_count: int


def _evidence_failure(message: str, *, code: str) -> PassExecutionError:
    return PassExecutionError("terra_evidence", "dependency", message, code=code)


def compute_photo_identity(path: Path) -> PhotoIdentity:
    """Decode one image and derive its evidence_dedup_v1 identity."""
    import hashlib

    from PIL import Image, ImageStat

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
    except Exception as exc:
        raise _evidence_failure(
            f"cannot decode evidence photo {path}: {exc}",
            code=type(exc).__name__,
        ) from exc
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}|".encode("ascii"))
    digest.update(rgb.tobytes())
    gray = rgb.convert("L").resize((8, 8), resample=Image.LANCZOS)
    pixels = gray.tobytes()  # L mode: one byte per pixel
    mean = sum(pixels) / len(pixels)
    average_hash = 0
    for pixel in pixels:
        average_hash = (average_hash << 1) | (1 if pixel > mean else 0)
    mean_rgb = tuple(ImageStat.Stat(rgb).mean)
    return PhotoIdentity(
        exact_sha256=digest.hexdigest(),
        average_hash=average_hash,
        mean_rgb=(mean_rgb[0], mean_rgb[1], mean_rgb[2]),
    )


def build_photo_identity_index(
    photo_keys: List[str], photo_key_to_path: Mapping[str, Path]
) -> Dict[str, PhotoIdentity]:
    """One identity per photo per listing, shared across conditions."""
    index: Dict[str, PhotoIdentity] = {}
    for photo_key in sorted(set(photo_keys)):
        path = photo_key_to_path.get(photo_key)
        if path is None or not Path(path).is_file():
            raise _evidence_failure(
                f"evidence photo {photo_key!r} has no readable file "
                f"({path}) — evidence must exist before review",
                code="MissingEvidencePhoto",
            )
        index[photo_key] = compute_photo_identity(Path(path))
    return index


def _near_pair(a: PhotoIdentity, b: PhotoIdentity) -> bool:
    if a.exact_sha256 == b.exact_sha256:
        return False  # exact duplicates belong to the exact family
    if (a.average_hash ^ b.average_hash).bit_count() > NEAR_HAMMING_MAX:
        return False
    delta = max(abs(x - y) for x, y in zip(a.mean_rgb, b.mean_rgb))
    return delta <= NEAR_MEAN_RGB_DELTA_MAX


def _closure_groups(
    keys: List[str], pairs: List[Tuple[str, str]]
) -> List[List[str]]:
    """Union-find closure; only classes with >= 2 members, sorted."""
    parent = {key: key for key in keys}

    def _find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for left, right in pairs:
        parent[_find(right)] = _find(left)
    classes: Dict[str, List[str]] = {}
    for key in keys:
        classes.setdefault(_find(key), []).append(key)
    return sorted(
        sorted(members) for members in classes.values() if len(members) > 1
    )


def dedup_photos(
    photo_keys: Tuple[str, ...], identity_index: Mapping[str, PhotoIdentity]
) -> PhotoDedup:
    keys = sorted(set(photo_keys))
    by_exact: Dict[str, List[str]] = {}
    for key in keys:
        by_exact.setdefault(identity_index[key].exact_sha256, []).append(key)
    exact_groups = sorted(
        sorted(members) for members in by_exact.values() if len(members) > 1
    )
    near_pairs = [
        (keys[i], keys[j])
        for i in range(len(keys))
        for j in range(i + 1, len(keys))
        if _near_pair(identity_index[keys[i]], identity_index[keys[j]])
    ]
    near_groups = _closure_groups(keys, near_pairs)
    exact_pairs = [
        (group[0], member) for group in exact_groups for member in group[1:]
    ]
    merged_groups = _closure_groups(keys, exact_pairs + near_pairs)
    grouped = {key for group in merged_groups for key in group}
    view_classes = sorted(
        merged_groups + [[key] for key in keys if key not in grouped]
    )
    return PhotoDedup(
        exact_groups=tuple(tuple(group) for group in exact_groups),
        near_groups=tuple(tuple(group) for group in near_groups),
        merged_groups=tuple(tuple(group) for group in merged_groups),
        representatives=tuple(sorted(members[0] for members in view_classes)),
        distinct_view_count=len(view_classes),
    )


def build_evidence_facts(
    draft: ConditionDraft,
    *,
    identity_index: Mapping[str, PhotoIdentity],
    observables: Mapping[str, Any],
    estimate_id: str,
) -> EvidenceFacts:
    condition = draft.condition
    photo_keys = tuple(sorted({ref["photo_key"] for ref in draft.evidence_refs}))
    dedup = dedup_photos(photo_keys, identity_index)
    observable = observables.get(condition.catalog_item_id) or {}
    return EvidenceFacts(
        evidence_id=make_evidence_id(
            estimate_id=estimate_id, condition_id=condition.condition_id
        ),
        schema_version=CONTRACTS_SCHEMA_VERSION,
        condition_id=condition.condition_id,
        photo_keys=photo_keys,
        distinct_photo_count=len(photo_keys),
        distinct_view_count=dedup.distinct_view_count,
        duplicate_groups=dedup.merged_groups,
        evidence_refs=draft.evidence_refs,
        min_photo_evidence_required=observable.get("min_photo_evidence"),
        representative_photo_keys=dedup.representatives,
        exact_duplicate_groups=dedup.exact_groups,
        near_duplicate_groups=dedup.near_groups,
        dedup_policy_version=EVIDENCE_DEDUP_POLICY_VERSION,
    )
