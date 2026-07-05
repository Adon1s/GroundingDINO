"""Backfill renovation_estimate_v4 into an existing photo_intel artifact.

Loads a saved photo_intel.json, runs compute_renovation_estimate_v4() against
the already-persisted issues/photos/v3-estimate, and writes the result back
into artifact["renovation_estimate_v4"]. After backfill, compare_reno_estimates.py
produces real package/reconciliation output instead of `(missing)` placeholders.

The live pipeline's `v3_reviewed_candidates` (in-memory EstimateCandidate
objects from a Pass 2f run) are not persisted to the artifact, so this
backfill always passes `v3_reviewed_candidates=None`. The v4 function's
existing fallback path produces `provenance.v3_pass_2f_reused == False` and
reconstructs Pass 2f fields from the v4 candidate extraction.

Usage:
    python tools/backfill_reno_v4.py --run <file_or_dir> --catalog <path>
    python tools/backfill_reno_v4.py --run <path> --catalog <path> --force
    python tools/backfill_reno_v4.py --run <path> --catalog <path> --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifact_writers import (
    _extract_property_metadata_from_mapping,
    _load_scrape_metadata_for_job,
    load_issue_catalog,
)
from tools.renovation_estimate_v4 import compute_renovation_estimate_v4


def _resolve_property_metadata_from_artifact(
    artifact: Dict[str, Any],
) -> Dict[str, Any]:
    """Recover property_metadata for a saved artifact.

    Preference order:
      1. ``artifact["property_metadata"]`` (persisted by the live writer).
      2. Re-discover scrape.json by synthesizing a job-shaped object from the
         artifact's photo paths and calling ``_load_scrape_metadata_for_job``.
    Returns an empty dict if neither source yields anything.
    """
    persisted = artifact.get("property_metadata")
    if isinstance(persisted, dict) and persisted:
        result = dict(persisted)
        result.setdefault("metadata_source", "persisted_artifact")
        return result

    property_key = (
        (artifact.get("property") or {}).get("property_key")
        or artifact.get("property_key")
        or ""
    )
    photos = artifact.get("photos") or {}
    photo_records = photos.values() if isinstance(photos, dict) else photos
    fake_results = []
    for record in photo_records or []:
        if not isinstance(record, dict):
            continue
        image_path = (record.get("photo") or {}).get("image_path")
        if image_path:
            fake_results.append(SimpleNamespace(image_path=image_path))
    if not property_key or not fake_results:
        return {}

    synthetic_job = SimpleNamespace(
        property_key=property_key,
        results=fake_results,
    )
    return _load_scrape_metadata_for_job(synthetic_job) or {}


def _resolve_artifact_path(run_path: Path) -> Path:
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


def backfill(
    artifact_path: Path,
    catalog: Dict[str, Any],
    *,
    force: bool,
    dry_run: bool,
    metadata_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Backfill renovation_estimate_v4 into a single artifact.

    Returns a result dict; never raises. Status values:
      - "skip_no_v3"           : artifact has no renovation_estimate
      - "skip_already_present" : artifact has v4 and force=False
      - "updated"              : v4 written (or would be written in dry-run)
      - "error"                : load/compute/write failed; 'reason' is set
    """
    try:
        with artifact_path.open("r", encoding="utf-8") as f:
            artifact = json.load(f)
    except json.JSONDecodeError as e:
        return {"status": "error", "reason": f"invalid JSON: {e}"}
    except OSError as e:
        return {"status": "error", "reason": f"cannot read file: {e}"}

    if artifact.get("renovation_estimate") is None:
        return {"status": "skip_no_v3"}

    was_present = artifact.get("renovation_estimate_v4") is not None
    if was_present and not force:
        return {"status": "skip_already_present"}

    property_metadata = _resolve_property_metadata_from_artifact(artifact)
    if metadata_override:
        # CLI-supplied metadata wins over artifact/scrape recovery — heals July
        # artifacts that shipped property_metadata: {} without a re-analysis.
        property_metadata = {**property_metadata, **metadata_override}

    try:
        v4 = compute_renovation_estimate_v4(
            issues_flat=artifact.get("estimate_issues_flat") or [],
            issue_catalog=catalog,
            photos=artifact.get("photos") or {},
            v3_reviewed_candidates=None,
            v3_estimate=artifact.get("renovation_estimate"),
            property_metadata=property_metadata or None,
        )
    except Exception as e:
        return {"status": "error", "reason": f"v4 computation failed: {e}"}

    if v4 is None:
        return {"status": "error", "reason": "v4 returned None"}

    final_rehab = v4.get("final_rehab") or {}
    summary = {
        "v4_low": int(final_rehab.get("low") or 0),
        "v4_high": int(final_rehab.get("high") or 0),
        "package_count": len(v4.get("packages") or []),
        "was_present": was_present,
        "metadata_source": (property_metadata or {}).get("metadata_source") or "none",
    }

    if dry_run:
        return {"status": "updated", "dry_run": True, **summary}

    artifact["renovation_estimate_v4"] = v4

    tmp_path = artifact_path.parent / (artifact_path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)
        tmp_path.replace(artifact_path)
    except OSError as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return {"status": "error", "reason": f"write failed: {e}"}

    return {"status": "updated", "dry_run": False, **summary}


def _format_result(artifact_path: Path, result: Dict[str, Any]) -> str:
    status = result["status"]
    if status == "skip_no_v3":
        return f"skip: {artifact_path} (no renovation_estimate, nothing to build from)"
    if status == "skip_already_present":
        return (
            f"skip: {artifact_path} "
            f"(renovation_estimate_v4 already present; use --force to overwrite)"
        )
    if status == "updated":
        if result.get("dry_run"):
            verb = "would overwrite" if result.get("was_present") else "would write"
        else:
            verb = "overwrote" if result.get("was_present") else "wrote"
        return (
            f"{verb}: {artifact_path} — "
            f"v4 final_rehab ${result['v4_low']:,}–${result['v4_high']:,}, "
            f"{result['package_count']} packages, "
            f"metadata_source={result.get('metadata_source', 'none')}"
        )
    return f"error: {artifact_path} — {result.get('reason', 'unknown error')}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill renovation_estimate_v4 into a saved photo_intel artifact."
        ),
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to a photo_intel.json file or a directory containing one.",
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Path to the issue catalog JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite renovation_estimate_v4 even if already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute v4 and print the result, but do not write to disk.",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help=(
            "JSON string of listing facts (price/beds/baths/sqft/...) to inject, "
            "winning over artifact/scrape recovery. Heals artifacts that shipped "
            "property_metadata: {} without re-analysis."
        ),
    )
    args = parser.parse_args(argv)

    metadata_override: Optional[Dict[str, Any]] = None
    if args.metadata_json:
        try:
            parsed = json.loads(args.metadata_json)
        except (ValueError, TypeError) as e:
            print(f"error: failed to parse --metadata-json: {e}", file=sys.stderr)
            return 1
        if not isinstance(parsed, dict):
            print("error: --metadata-json is not a JSON object", file=sys.stderr)
            return 1
        # Run through the same allowlist/alias normalization as the live path.
        metadata_override = _extract_property_metadata_from_mapping(parsed)
        metadata_override.setdefault("metadata_source", "cli_metadata_json")

    run_path = Path(args.run)
    try:
        artifact_path = _resolve_artifact_path(run_path)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    catalog_path = Path(args.catalog)
    if not catalog_path.is_file():
        print(f"error: catalog not found: {catalog_path}", file=sys.stderr)
        return 1

    try:
        catalog = load_issue_catalog(catalog_path)
    except Exception as e:
        print(f"error: failed to load catalog {catalog_path}: {e}", file=sys.stderr)
        return 1

    result = backfill(
        artifact_path,
        catalog,
        force=args.force,
        dry_run=args.dry_run,
        metadata_override=metadata_override,
    )

    line = _format_result(artifact_path, result)
    if result["status"] == "error":
        print(line, file=sys.stderr)
        return 1
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
