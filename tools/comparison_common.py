"""Hashing, canonical serialization, and atomic-write helpers shared by the
comparison and benchmark harnesses.

Extracted from ``quant_artifact_comparison.py``, which still re-exports every
name so its CLI and tests are behaviorally unchanged. The extraction exists
because ``tools/benchmarking`` needs byte-identical fingerprinting: a dataset
fingerprint computed here and a report fingerprint computed there must agree,
and two copies of ``canonical_json`` would eventually disagree about
separators or key order and make baselines silently incomparable.

Deliberately dependency-free (stdlib only) so the benchmark's dataset
validation can run without importing torch via sentence-transformers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ComparisonError(ValueError):
    """Operator-facing failure: bad inputs, unresolvable references, drift."""


def canonical_json(value: Any) -> str:
    """Serialize deterministically, for hashing rather than for reading.

    Sorted keys and no whitespace, so logically equal payloads produce equal
    digests regardless of construction order. ``default=str`` keeps Path and
    datetime values hashable instead of raising mid-fingerprint.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in 1 MiB chunks; photos are too large to slurp."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical(value: Any) -> str:
    """Fingerprint a structure. The pairing of canonical_json + sha256 is used
    for every fingerprint in the benchmark, so it gets one name."""
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    """Write indented JSON via a temp file + replace, so a crash mid-write
    cannot leave a half-parsed artifact behind. Sorted keys keep diffs small."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, sort_keys=True), encoding="utf-8")
    temp.replace(path)
