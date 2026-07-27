"""Dataset freezing: photo integrity, ordering, containment, fingerprints, sealing.

A benchmark is only meaningful if the inputs cannot move. Production photo
ordering comes from a bare ``fs.readdirSync(...).sort()`` in the frontend
(lib/property/propertyImages.ts:29), and the manifest hashing path uses
``localeCompare`` instead (lib/property/imageManifest.ts:52) — two comparators
that can disagree. So order is resolved **once**, at import, and written down.
After that a production sort change cannot silently reorder a sealed dataset.

Everything here is read-only with respect to the frontend repository and the
production artifact tree.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.benchmarking import vocabulary as vocab_mod
from tools.benchmarking.schemas import (
    LISTING_SCHEMA_VERSION,
    ValidationResult,
    validate_listing,
    validate_manifest,
    validate_reference,
)
from tools.comparison_common import ComparisonError, atomic_json, sha256_canonical, sha256_file

DATASET_FINGERPRINT_VERSION = 1

MANIFEST_FILENAME = "manifest.json"
VOCABULARY_FILENAME = "reference_vocabulary.json"
PHOTOS_DIRNAME = "photos"

# Git LFS pointer files are small text stubs. Hashing one produces a perfectly
# stable, perfectly useless fingerprint, and the analyzer would then "analyze" a
# 130-byte text file as a photo. Detect them explicitly.
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

# Mirrors the frontend's IMAGE_EXTENSIONS filter (lib/analysis/utils).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")


# ---------------------------------------------------------------------------
# Paths and containment
# ---------------------------------------------------------------------------

def _resolve_within(root: Path, relative: str, *, label: str) -> Path:
    """Resolve ``relative`` under ``root``, refusing anything that escapes.

    Checked with resolved paths rather than by string inspection so a symlink or
    a ``..`` buried mid-path cannot slip through.
    """
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ComparisonError(
            f"{label} {relative!r} resolves outside the dataset directory "
            f"({candidate}); refusing to read it"
        )
    return candidate


def dataset_dir(datasets_root: Path, dataset_version: str) -> Path:
    return _resolve_within(datasets_root, dataset_version, label="dataset version")


def listing_dir(dataset_path: Path, listing_id: str) -> Path:
    return _resolve_within(dataset_path, f"listings/{listing_id}", label="listing id")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise ComparisonError(f"{label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ComparisonError(f"{label} is unreadable: {path}: {exc}") from exc


def load_manifest(dataset_path: Path) -> Dict[str, Any]:
    manifest = _read_json(dataset_path / MANIFEST_FILENAME, label="manifest")
    result = validate_manifest(manifest)
    if not result.ok:
        raise ComparisonError(_format_errors(f"{dataset_path / MANIFEST_FILENAME}", result))
    return manifest


def is_sealed(manifest: Dict[str, Any]) -> bool:
    return manifest.get("state") == "sealed"


def require_unsealed(manifest: Dict[str, Any], operation: str) -> None:
    """Every mutating command goes through here.

    A sealed dataset is the anchor a baseline was accepted against; editing one
    in place would invalidate every comparison that references it while leaving
    the fingerprint that proves it unchanged.
    """
    if is_sealed(manifest):
        raise ComparisonError(
            f"dataset {manifest.get('dataset_version')!r} is sealed; {operation} would change "
            f"frozen inputs. Create a new dataset version instead."
        )


def load_frozen_vocabulary(dataset_path: Path) -> Optional[Dict[str, Any]]:
    """The sealed vocabulary, or None for a draft dataset that has none yet."""
    path = dataset_path / VOCABULARY_FILENAME
    return vocab_mod.load(path) if path.is_file() else None


def load_listing(dataset_path: Path, manifest: Dict[str, Any], listing_id: str) -> Dict[str, Any]:
    entry = manifest_entry(manifest, listing_id)
    path = _resolve_within(dataset_path, entry["metadata_path"], label="metadata_path")
    listing = _read_json(path, label=f"listing {listing_id} metadata")
    result = validate_listing(listing, listing_id=listing_id)
    if not result.ok:
        raise ComparisonError(_format_errors(str(path), result))
    return listing


def manifest_entry(manifest: Dict[str, Any], listing_id: str) -> Dict[str, Any]:
    for entry in manifest.get("listings") or []:
        if isinstance(entry, dict) and entry.get("id") == listing_id:
            return entry
    known = [e.get("id") for e in (manifest.get("listings") or []) if isinstance(e, dict)]
    raise ComparisonError(f"listing {listing_id!r} is not in the manifest; known: {known}")


def _format_errors(where: str, result: ValidationResult) -> str:
    lines = [f"{where}: {len(result.errors)} validation error(s)"]
    lines.extend(f"  - {message}" for message in result.errors)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Photo integrity
# ---------------------------------------------------------------------------

def _check_photo_file(path: Path, expected: Dict[str, Any], where: str,
                      result: ValidationResult) -> None:
    if not path.is_file():
        result.error(where, f"photo file is missing: {path}")
        return

    try:
        head = path.open("rb").read(len(_LFS_POINTER_PREFIX))
    except OSError as exc:
        result.error(where, f"photo file is unreadable: {path}: {exc}")
        return
    if head == _LFS_POINTER_PREFIX:
        result.error(where, f"{path.name} is a Git LFS pointer, not an image; run `git lfs pull`")
        return

    actual_size = path.stat().st_size
    if actual_size != expected.get("byte_size"):
        result.error(where, f"{path.name} byte_size is {actual_size}, manifest says "
                            f"{expected.get('byte_size')}")
    actual_hash = sha256_file(path)
    if actual_hash != str(expected.get("sha256") or "").lower():
        result.error(where, f"{path.name} sha256 is {actual_hash[:12]}..., manifest says "
                            f"{str(expected.get('sha256'))[:12]}...; the photo changed on disk")


def photo_paths(dataset_path: Path, listing_id: str, listing: Dict[str, Any]) -> List[Path]:
    """Absolute photo paths in frozen manifest order.

    This is the only function that decides what order the analyzer sees, and it
    reads ``order`` rather than sorting, so ordering is a dataset property rather
    than a property of whatever filesystem the benchmark runs on.
    """
    base = listing_dir(dataset_path, listing_id) / PHOTOS_DIRNAME
    ordered = sorted(
        (p for p in (listing.get("photos") or []) if isinstance(p, dict)),
        key=lambda p: p.get("order", 0),
    )
    return [_resolve_within(base, str(p.get("filename") or ""), label="photo filename")
            for p in ordered]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset(dataset_path: Path, *, listing_ids: Optional[Sequence[str]] = None,
                     check_photo_bytes: bool = True) -> ValidationResult:
    """Validate the manifest, every listing, every reference, and photo integrity."""
    result = ValidationResult()
    manifest = load_manifest(dataset_path)
    frozen_vocab = load_frozen_vocabulary(dataset_path)

    if is_sealed(manifest):
        if frozen_vocab is None:
            result.error("dataset", f"sealed dataset is missing {VOCABULARY_FILENAME}")
        else:
            recorded = str(manifest.get("vocabulary_fingerprint") or "")
            if recorded != str(frozen_vocab.get("fingerprint") or ""):
                result.error("dataset", "manifest vocabulary_fingerprint does not match "
                                        f"{VOCABULARY_FILENAME}")

    wanted = set(listing_ids) if listing_ids else None
    for entry in manifest.get("listings") or []:
        if not isinstance(entry, dict):
            continue
        listing_id = str(entry.get("id") or "")
        if wanted is not None and listing_id not in wanted:
            continue
        result.extend(_validate_one_listing(dataset_path, manifest, entry, listing_id,
                                            frozen_vocab, check_photo_bytes))

    if is_sealed(manifest):
        recorded = str(manifest.get("dataset_fingerprint") or "")
        if wanted is None and result.ok:
            actual = dataset_fingerprint(dataset_path, manifest)
            if recorded != actual:
                result.error("dataset", f"dataset_fingerprint is {recorded[:12]}... but the "
                                        f"contents hash to {actual[:12]}...")
    return result


def _validate_one_listing(dataset_path: Path, manifest: Dict[str, Any], entry: Dict[str, Any],
                          listing_id: str, frozen_vocab: Optional[Dict[str, Any]],
                          check_photo_bytes: bool) -> ValidationResult:
    result = ValidationResult()
    try:
        metadata_path = _resolve_within(dataset_path, str(entry.get("metadata_path") or ""),
                                        label="metadata_path")
        reference_path = _resolve_within(dataset_path, str(entry.get("reference_path") or ""),
                                         label="reference_path")
    except ComparisonError as exc:
        result.error(f"listing[{listing_id}]", str(exc))
        return result

    try:
        listing = _read_json(metadata_path, label=f"listing {listing_id} metadata")
    except ComparisonError as exc:
        result.error(f"listing[{listing_id}]", str(exc))
        return result

    listing_result = validate_listing(listing, listing_id=listing_id)
    for message in listing_result.errors:
        result.error(f"listing[{listing_id}]", message)
    for message in listing_result.warnings:
        result.warn(f"listing[{listing_id}]", message)

    if check_photo_bytes and isinstance(listing.get("photos"), list):
        base = listing_dir(dataset_path, listing_id) / PHOTOS_DIRNAME
        for photo in listing["photos"]:
            if not isinstance(photo, dict):
                continue
            filename = str(photo.get("filename") or "")
            if not filename:
                continue
            try:
                path = _resolve_within(base, filename, label="photo filename")
            except ComparisonError as exc:
                result.error(f"listing[{listing_id}].photos", str(exc))
                continue
            _check_photo_file(path, photo, f"listing[{listing_id}].photos", result)

    # A reference is optional on a draft dataset: phase-1 annotation has not
    # necessarily started. Sealing is what makes it mandatory.
    if reference_path.is_file():
        try:
            reference = _read_json(reference_path, label=f"reference {listing_id}")
        except ComparisonError as exc:
            result.error(f"reference[{listing_id}]", str(exc))
            return result
        ref_result = validate_reference(reference, listing=listing, vocabulary=frozen_vocab)
        for message in ref_result.errors:
            result.error(f"reference[{listing_id}]", message)
        for message in ref_result.warnings:
            result.warn(f"reference[{listing_id}]", message)
    elif is_sealed(manifest):
        result.error(f"reference[{listing_id}]", f"sealed dataset is missing "
                                                 f"{entry.get('reference_path')}")
    return result


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def dataset_fingerprint(dataset_path: Path, manifest: Dict[str, Any]) -> str:
    """Fingerprint the dataset's semantic content.

    Covers listing identity, tier, slices, the metadata and reference *content*,
    and every photo's order/name/hash/size. Deliberately excludes the manifest's
    own ``state`` and fingerprint fields so sealing a dataset does not change the
    fingerprint it is recording.

    JSON is hashed canonically rather than by file bytes, so reindenting or
    reordering keys in metadata.json does not read as "the dataset changed" —
    logically equal inputs must fingerprint equal. Photos are hashed as bytes,
    which is the one place byte-exactness genuinely matters.
    """
    listings: List[Dict[str, Any]] = []
    for entry in sorted((e for e in manifest.get("listings") or [] if isinstance(e, dict)),
                        key=lambda e: str(e.get("id") or "")):
        listing_id = str(entry.get("id") or "")
        metadata_path = _resolve_within(dataset_path, str(entry.get("metadata_path") or ""),
                                        label="metadata_path")
        reference_path = _resolve_within(dataset_path, str(entry.get("reference_path") or ""),
                                         label="reference_path")
        listing = _read_json(metadata_path, label=f"listing {listing_id} metadata")
        photos = sorted(
            (p for p in (listing.get("photos") or []) if isinstance(p, dict)),
            key=lambda p: p.get("order", 0),
        )
        listings.append({
            "id": listing_id,
            "tier": entry.get("tier"),
            "slices": sorted(str(s) for s in (entry.get("slices") or [])),
            "metadata_sha256": sha256_canonical(listing),
            "reference_sha256": (
                sha256_canonical(_read_json(reference_path, label=f"reference {listing_id}"))
                if reference_path.is_file() else None
            ),
            "photos": [
                {
                    "order": p.get("order"),
                    "filename": p.get("filename"),
                    "sha256": p.get("sha256"),
                    "byte_size": p.get("byte_size"),
                }
                for p in photos
            ],
        })

    return sha256_canonical({
        "fingerprint_version": DATASET_FINGERPRINT_VERSION,
        "dataset_id": manifest.get("dataset_id"),
        "dataset_version": manifest.get("dataset_version"),
        "vocabulary_fingerprint": manifest.get("vocabulary_fingerprint"),
        "listings": listings,
    })


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_listing(
    dataset_path: Path,
    *,
    listing_id: str,
    source_dir: Path,
    tier: str = "gold",
    slices: Optional[Sequence[str]] = None,
    property_metadata: Optional[Dict[str, Any]] = None,
    market_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Copy a listing's photos into the dataset and write its frozen metadata.

    Ordering is assigned here from ``sorted()`` over image basenames — matching
    what production serves today — and then never recomputed.
    """
    manifest = load_manifest(dataset_path)
    require_unsealed(manifest, f"importing listing {listing_id!r}")

    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise ComparisonError(f"source image directory not found: {source_dir}")

    names = sorted(
        p.name for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("._")
    )
    if not names:
        raise ComparisonError(f"no images found in {source_dir}")

    target = listing_dir(dataset_path, listing_id) / PHOTOS_DIRNAME
    target.mkdir(parents=True, exist_ok=True)

    photos: List[Dict[str, Any]] = []
    by_hash: Dict[str, str] = {}
    for order, name in enumerate(names, start=1):
        source = source_dir / name
        destination = target / name
        shutil.copy2(source, destination)
        digest = sha256_file(destination)
        entry: Dict[str, Any] = {
            "order": order,
            "filename": name,
            "sha256": digest,
            "byte_size": destination.stat().st_size,
        }
        # Declared up front so validation passes; a genuine accident is still
        # visible because the duplicate is named explicitly in the metadata.
        if digest in by_hash:
            entry["intentional_duplicate_of"] = by_hash[digest]
        else:
            by_hash[digest] = name
        photos.append(entry)

    listing: Dict[str, Any] = {
        "schema_version": LISTING_SCHEMA_VERSION,
        "listing_id": listing_id,
        "dataset_version": manifest.get("dataset_version"),
        "original_metadata": property_metadata or {},
        "asking_price": (property_metadata or {}).get("price"),
        "sqft": (property_metadata or {}).get("sqft"),
        "beds": (property_metadata or {}).get("beds"),
        "baths": (property_metadata or {}).get("baths"),
        "location": (property_metadata or {}).get("address"),
        "market_inputs": market_inputs or {},
        "photos": photos,
    }

    metadata_path = listing_dir(dataset_path, listing_id) / "metadata.json"
    atomic_json(metadata_path, listing)

    _register_listing(dataset_path, manifest, listing_id, tier, slices)
    return listing


def _register_listing(dataset_path: Path, manifest: Dict[str, Any], listing_id: str,
                      tier: str, slices: Optional[Sequence[str]]) -> None:
    entries = manifest.setdefault("listings", [])
    relative = f"listings/{listing_id}"
    record = {
        "id": listing_id,
        "tier": tier,
        "slices": sorted(set(slices or ())),
        "metadata_path": f"{relative}/metadata.json",
        "reference_path": f"{relative}/reference.json",
    }
    for index, entry in enumerate(entries):
        if isinstance(entry, dict) and entry.get("id") == listing_id:
            entries[index] = record
            break
    else:
        entries.append(record)
    entries.sort(key=lambda e: str(e.get("id") or ""))
    atomic_json(dataset_path / MANIFEST_FILENAME, manifest)


# ---------------------------------------------------------------------------
# Seal
# ---------------------------------------------------------------------------

def seal(dataset_path: Path, issue_catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Freeze the vocabulary and fingerprint, making the dataset immutable.

    Refuses unless every listing has a ``reviewed`` reference: a sealed dataset
    with a draft reference in it would gate releases on unreviewed truth.
    """
    manifest = load_manifest(dataset_path)
    require_unsealed(manifest, "sealing")

    problems: List[str] = []
    for entry in manifest.get("listings") or []:
        if not isinstance(entry, dict):
            continue
        listing_id = str(entry.get("id") or "")
        reference_path = _resolve_within(dataset_path, str(entry.get("reference_path") or ""),
                                         label="reference_path")
        if not reference_path.is_file():
            problems.append(f"{listing_id}: no compiled reference at {entry.get('reference_path')}")
            continue
        reference = _read_json(reference_path, label=f"reference {listing_id}")
        status = reference.get("review_status")
        if status != "reviewed":
            problems.append(f"{listing_id}: review_status is {status!r}, expected 'reviewed'")
    if problems:
        raise ComparisonError(
            "cannot seal; every listing needs a reviewed reference:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    snapshot = vocab_mod.snapshot(issue_catalog)
    atomic_json(dataset_path / VOCABULARY_FILENAME, snapshot)

    manifest["vocabulary_fingerprint"] = snapshot["fingerprint"]
    manifest["dataset_fingerprint"] = dataset_fingerprint(dataset_path, manifest)
    manifest["state"] = "sealed"
    atomic_json(dataset_path / MANIFEST_FILENAME, manifest)

    result = validate_dataset(dataset_path)
    if not result.ok:
        raise ComparisonError(_format_errors("sealed dataset failed validation", result))
    return manifest
