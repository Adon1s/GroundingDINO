"""Publication-boundary validation for photo_intel artifacts.

write_photo_intel calls validate_publication_payload() on the fully assembled
payload immediately before the JSON write. The checks are catalog-generic: the
kind vocabulary and version expectations come from the catalog the run
selected, so v1 and v2 publications get the same integrity guarantees and a
mixed v1/v2 payload can never land on disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from tools.observation_kinds import (
    LEGACY_CATALOG_KINDS,
    OBSERVATION_KINDS,
    ONTOLOGY_VERSION,
)
from tools.pipeline_common import LEGACY_ONTOLOGY_VERSION

_MANIFEST_PATH = Path(__file__).resolve().parent / "catalog_migrations" / "2.1_to_3.0.json"

# Flat issue lanes the gate inspects. Per-photo lanes are derived from the same
# resolved issues, so validating the flat lanes covers every published id/kind.
_ISSUE_LANES = (
    "issues_flat",
    "estimate_issues_flat",
    "product_issues_flat",
    "product_estimate_issues_flat",
)

_deprecated_cache: Optional[Tuple[frozenset, Dict[str, Tuple[str, ...]]]] = None


def _deprecated_manifest_index() -> Tuple[frozenset, Dict[str, Tuple[str, ...]]]:
    """(deprecated legacy ids, {legacy id -> successor ids}) from the shipped
    migration manifest, loaded once per process."""
    global _deprecated_cache
    if _deprecated_cache is None:
        manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        entries = [e for e in manifest.get("entries") or [] if isinstance(e, dict)]
        deprecated = frozenset(
            str(e["legacy_id"]) for e in entries if e.get("deprecated")
        )
        successors = {
            str(e["legacy_id"]): tuple(
                str(s["id"]) for s in e.get("successors") or []
            )
            for e in entries if e.get("deprecated")
        }
        _deprecated_cache = (deprecated, successors)
    return _deprecated_cache


def deprecated_legacy_ids() -> frozenset:
    """The legacy catalog ids retired by the 2.1 -> 3.0 migration."""
    return _deprecated_manifest_index()[0]


def _reject(reason: str) -> None:
    raise RuntimeError(f"write_photo_intel: refusing to publish — {reason}")


def _issue_kind(row: Mapping[str, Any]) -> str:
    return str(
        row.get("canonical_kind")
        or row.get("catalog_item_kind")
        or row.get("kind")
        or ""
    ).strip()


def validate_publication_payload(
    photo_intel: Mapping[str, Any],
    issue_catalog: Optional[Mapping[str, Any]],
) -> None:
    """Raise RuntimeError unless the payload is internally consistent with the
    catalog it was produced against.

    Checks: root ontology/catalog version stamps present and matching the
    selected catalog; every issue kind within the catalog's kind vocabulary
    (rejects stale kinds and mixed v1/v2 payloads); every resolved catalog id
    present in the catalog (deprecated split parents get a dedicated error);
    every resolved issue's kind matching its catalog entry; any
    renovation-architecture envelope (private shadow or reserved root key)
    valid for its placement.
    """
    catalog = issue_catalog or {}
    is_v2 = catalog.get("ontology_version") == ONTOLOGY_VERSION
    expected_ontology = ONTOLOGY_VERSION if is_v2 else LEGACY_ONTOLOGY_VERSION
    expected_catalog_version = str(catalog.get("version") or "")
    kind_vocabulary = OBSERVATION_KINDS if is_v2 else LEGACY_CATALOG_KINDS

    artifact_ontology = str(photo_intel.get("ontology_version") or "")
    artifact_catalog_version = str(photo_intel.get("catalog_version") or "")
    if not artifact_ontology:
        _reject(
            "the payload carries no root ontology_version stamp. "
            "write_photo_intel stamps it from the selected catalog; a missing "
            "stamp means the payload was assembled outside the writer."
        )
    if artifact_ontology != expected_ontology:
        _reject(
            f"root ontology_version {artifact_ontology!r} does not match the "
            f"selected catalog's ontology ({expected_ontology!r}). The catalog "
            "and the payload must come from the same KIND_ONTOLOGY_VERSION "
            "selection."
        )
    if expected_catalog_version and artifact_catalog_version != expected_catalog_version:
        # Both shipped catalogs declare a version, so a missing root stamp is a
        # mismatch here; only an unversioned (test) catalog skips this check.
        _reject(
            f"root catalog_version {artifact_catalog_version!r} does not match "
            f"the selected catalog ({expected_catalog_version!r})."
        )

    items_by_id = {
        str(it["id"]): it
        for it in catalog.get("items") or []
        if isinstance(it, dict) and it.get("id")
    }
    deprecated, successors = (
        _deprecated_manifest_index() if is_v2 else (frozenset(), {})
    )

    for lane in _ISSUE_LANES:
        rows = photo_intel.get(lane)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = f"{lane}[{row.get('issue_id') or row.get('description') or '?'}]"
            kind = _issue_kind(row)
            if kind and kind not in kind_vocabulary:
                _reject(
                    f"{label} carries kind {kind!r}, outside the selected "
                    f"catalog's vocabulary {sorted(kind_vocabulary)}. This is "
                    "a stale kind or a mixed v1/v2 payload."
                )
            item_id = str(row.get("catalog_item_id") or "")
            if not item_id:
                continue
            if item_id in deprecated:
                succ_ids = list(successors.get(item_id, ()))
                detail = (
                    f"resolves to deprecated split parent {item_id!r}; "
                    f"its successors are {succ_ids}"
                    if succ_ids
                    else f"resolves to retired catalog id {item_id!r}, which has no successor"
                )
                _reject(
                    f"{label} {detail}. "
                    "A current run must never emit a deprecated legacy id."
                )
            item = items_by_id.get(item_id)
            if item is None:
                _reject(
                    f"{label} resolves to {item_id!r}, which does not exist in "
                    f"the selected catalog (version "
                    f"{expected_catalog_version!r})."
                )
            catalog_kind = str(item.get("kind") or "")
            if kind and catalog_kind and kind != catalog_kind:
                _reject(
                    f"{label} carries kind {kind!r} but catalog item "
                    f"{item_id!r} is kind {catalog_kind!r}. Resolved issues "
                    "must carry their catalog entry's canonical kind."
                )

    _validate_renovation_architecture_keys(photo_intel)


def _validate_renovation_architecture_keys(photo_intel: Mapping[str, Any]) -> None:
    """The new-architecture envelope placements (Session 5+).

    The private shadow key (analysis_debug.<SHADOW_DEBUG_KEY>) may hold only
    a valid finished envelope — 'complete' or 'failed'; partial, mixed-version,
    or malformed payloads fail before writing. The same key at the photo_intel
    root is reserved for the Session 6 cutover and must already be a valid
    'complete' envelope. Imports are local so publications that carry neither
    key never pay for the validator chain.
    """
    debug = photo_intel.get("analysis_debug")
    from tools.renovation_architecture.contracts import SHADOW_DEBUG_KEY

    private = debug.get(SHADOW_DEBUG_KEY) if isinstance(debug, dict) else None
    root = photo_intel.get(SHADOW_DEBUG_KEY)
    if private is None and root is None:
        return
    from tools.renovation_architecture.validators import validate_envelope

    if private is not None:
        validation = validate_envelope(private)
        if not validation.ok:
            _reject(
                f"analysis_debug.{SHADOW_DEBUG_KEY} is not a valid envelope: "
                + "; ".join(validation.errors[:5])
            )
        state = private.get("state")
        if state not in ("complete", "failed"):
            _reject(
                f"analysis_debug.{SHADOW_DEBUG_KEY} state {state!r} is not "
                "publishable — only finished 'complete' or 'failed' shadow "
                "envelopes may be written."
            )
    if root is not None:
        if not isinstance(root, Mapping):
            _reject(
                f"root {SHADOW_DEBUG_KEY} must be a complete envelope object."
            )
        validation = validate_envelope(root)
        if not validation.ok:
            _reject(
                f"root {SHADOW_DEBUG_KEY} is not a valid envelope: "
                + "; ".join(validation.errors[:5])
            )
        if root.get("state") != "complete":
            _reject(
                f"root {SHADOW_DEBUG_KEY} is reserved for the cutover and "
                f"must be a valid 'complete' envelope, got state "
                f"{root.get('state')!r}."
            )
