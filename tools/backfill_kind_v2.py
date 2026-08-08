"""Safely migrate stored photo_intel artifacts to observation-kind-v2.

Same-id changes are deterministic metadata reprojections. Split legacy ids are
re-resolved from their stored descriptions; ambiguous or unsupported text fails
closed and emits the exact photo set that needs image reanalysis. No runtime
aliasing is performed.

Dry-run is the default. Pass --apply to write an atomic backup-backed migration.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifact_writers import load_issue_catalog
from tools.pipeline_common import artifact_ontology_version, term_matches
from tools.reproject_product_views import reproject_artifact

ONTOLOGY_VERSION = "observation-kind-v2"
ISSUE_LANES = (
    "issues_flat",
    "estimate_issues_flat",
    "product_issues_flat",
    "product_estimate_issues_flat",
)
DERIVED_FIELDS = (
    "scoring",
    "summary_v1",
    "renovation_estimate",
    "renovation_estimate_v4",
    "ui_priorities_v1",
    "product_issues_flat",
    "product_estimate_issues_flat",
)
LEGACY_ID_FIELDS = (
    "defect_id",
    "upgrade_id",
    "resolved_defect_id",
    "resolved_upgrade_id",
)
ID_FIELDS = (
    "catalog_item_id",
    "catalogItemId",
    "resolved_item_id",
    *LEGACY_ID_FIELDS,
)


@dataclass
class MigrationResult:
    status: str
    path: str
    same_id_updates: int = 0
    split_updates: int = 0
    unique_issues: int = 0
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    reanalysis_photo_keys: List[str] = field(default_factory=list)
    reanalysis_image_paths: List[str] = field(default_factory=list)
    backup_path: Optional[str] = None
    reprojection: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: value for key, value in self.__dict__.items()
            if value not in (None, [], 0) or key in {"status", "path"}
        }


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _catalog_id(issue: Mapping[str, Any]) -> str:
    for field_name in ID_FIELDS:
        value = issue.get(field_name)
        if value:
            return str(value)
    return ""


def _description(issue: Mapping[str, Any]) -> str:
    return str(
        issue.get("description")
        or issue.get("observation")
        or issue.get("observation_text")
        or ""
    ).strip()


def _issue_key(issue: Mapping[str, Any], legacy_id: str) -> str:
    issue_id = str(issue.get("issue_id") or "").strip()
    if issue_id:
        return f"issue:{issue_id}|legacy:{legacy_id}"
    return f"description:{_norm(_description(issue))}|legacy:{legacy_id}"


def _iter_issue_dicts(artifact: Dict[str, Any]) -> Iterator[Tuple[Dict[str, Any], str]]:
    for lane in ISSUE_LANES:
        value = artifact.get(lane)
        if isinstance(value, list):
            for issue in value:
                if isinstance(issue, dict):
                    yield issue, lane
    photos = artifact.get("photos")
    if isinstance(photos, dict):
        for photo_key, photo in photos.items():
            if not isinstance(photo, dict):
                continue
            issues = photo.get("issues")
            if not isinstance(issues, dict):
                continue
            for lane in ("final", "matched", "canonical", "display", "removed"):
                rows = issues.get(lane)
                if isinstance(rows, list):
                    for issue in rows:
                        if isinstance(issue, dict):
                            issue.setdefault("photo_key", str(photo_key))
                            yield issue, f"photos.{photo_key}.issues.{lane}"


def _load_resolution_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("resolution map must be a JSON object")
    return {str(key): str(value) for key, value in data.items()}


def _manual_resolution(
    issue: Mapping[str, Any], legacy_id: str, resolution_map: Mapping[str, str],
) -> Optional[str]:
    keys = (
        str(issue.get("issue_id") or ""),
        _issue_key(issue, legacy_id),
        f"description:{_norm(_description(issue))}",
    )
    for key in keys:
        if key and key in resolution_map:
            return resolution_map[key]
    return None


def _support_score(description: str, successor: Mapping[str, Any]) -> Tuple[int, int]:
    terms = [str(term) for term in (successor.get("support_any") or []) if str(term).strip()]
    matched = sum(1 for term in terms if term_matches(term, description.lower()))
    # Longer matching phrases are a deterministic tie-breaker, never a source
    # of confidence on their own.
    longest = max(
        (len(_norm(term)) for term in terms if term_matches(term, description.lower())),
        default=0,
    )
    return matched, longest


def resolve_split_description(
    description: str,
    successors: List[Mapping[str, Any]],
    *,
    kind_hint: str = "",
) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
    if not description.strip():
        return None, "missing_description", []
    candidates = list(successors)
    if kind_hint in {"defect", "degradation", "modernization"}:
        candidates = [row for row in candidates if row.get("kind") == kind_hint]
    scored = [
        {
            "id": row["id"],
            "kind": row.get("kind"),
            "score": _support_score(description, row),
        }
        for row in candidates
    ]
    scored.sort(key=lambda row: (row["score"][0], row["score"][1], row["id"]), reverse=True)
    if not scored or scored[0]["score"][0] == 0:
        return None, "no_successor_support", scored
    if len(scored) > 1 and scored[0]["score"] == scored[1]["score"]:
        return None, "ambiguous_successor_support", scored
    return str(scored[0]["id"]), "stored_description_support", scored


def _stamp_issue(
    issue: Dict[str, Any], item: Mapping[str, Any], catalog_version: str,
) -> None:
    item_id = str(item["id"])
    kind = str(item["kind"])
    issue["catalog_item_id"] = item_id
    if "catalogItemId" in issue:
        issue["catalogItemId"] = item_id
    if "resolved_item_id" in issue:
        issue["resolved_item_id"] = item_id
    for field_name in LEGACY_ID_FIELDS:
        issue.pop(field_name, None)
    issue["kind"] = kind
    issue["catalog_item_kind"] = kind
    if "catalogItemKind" in issue:
        issue["catalogItemKind"] = kind
    issue["canonical_kind"] = kind
    issue["catalog_version"] = catalog_version
    issue["ontology_version"] = ONTOLOGY_VERSION


def _photo_path_index(artifact: Mapping[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    photos = artifact.get("photos")
    if not isinstance(photos, dict):
        return result
    for key, value in photos.items():
        if not isinstance(value, dict):
            continue
        photo = value.get("photo") if isinstance(value.get("photo"), dict) else value
        image_path = photo.get("image_path") or photo.get("image")
        if image_path:
            result[str(key)] = str(image_path)
    return result


def _write_atomic_with_backup(path: Path, artifact: Dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.pre_kind_v2_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    temporary = path.with_name(path.name + ".kind_v2.tmp")
    try:
        temporary.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return backup


def migrate_artifact(
    artifact_path: Path,
    catalog: Dict[str, Any],
    manifest: Dict[str, Any],
    *,
    apply: bool = False,
    resolution_map: Optional[Mapping[str, str]] = None,
    reproject: bool = True,
) -> MigrationResult:
    result = MigrationResult(status="error", path=str(artifact_path))
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.reason = f"cannot load artifact: {exc}"
        return result
    if not isinstance(artifact, dict):
        result.reason = "artifact root must be an object"
        return result

    catalog_version = str(catalog.get("version") or "")
    if catalog.get("ontology_version") != ONTOLOGY_VERSION:
        result.reason = "target catalog is not observation-kind-v2"
        return result
    if artifact_ontology_version(artifact) == ONTOLOGY_VERSION and artifact.get("catalog_version") == catalog_version:
        result.status = "skip_current"
        return result

    catalog_by_id = {
        str(item["id"]): item for item in (catalog.get("items") or [])
        if isinstance(item, dict) and item.get("id")
    }
    manifest_by_id = {
        str(entry["legacy_id"]): entry for entry in (manifest.get("entries") or [])
        if isinstance(entry, dict) and entry.get("legacy_id")
    }
    resolution_map = dict(resolution_map or {})
    working = copy.deepcopy(artifact)
    decisions: Dict[str, Tuple[Optional[str], str, List[Dict[str, Any]]]] = {}
    unique_issue_keys: set[str] = set()
    unresolved_by_key: Dict[str, Dict[str, Any]] = {}
    same_keys: set[str] = set()
    split_keys: set[str] = set()

    for issue, lane in _iter_issue_dicts(working):
        legacy_id = _catalog_id(issue)
        if not legacy_id:
            continue
        entry = manifest_by_id.get(legacy_id)
        if entry is None:
            if legacy_id in catalog_by_id:
                _stamp_issue(issue, catalog_by_id[legacy_id], catalog_version)
                continue
            key = _issue_key(issue, legacy_id)
            unresolved_by_key[key] = {
                "issue_key": key,
                "legacy_id": legacy_id,
                "description": _description(issue),
                "lane": lane,
                "reason": "unknown_legacy_id",
                "photo_key": issue.get("photo_key") or issue.get("source_photo_key"),
            }
            continue

        key = _issue_key(issue, legacy_id)
        unique_issue_keys.add(key)
        successor_rows = list(entry.get("successors") or [])
        if len(successor_rows) == 1 and str(successor_rows[0]["id"]) == legacy_id:
            successor_id = str(successor_rows[0]["id"])
            reason = "same_id_metadata"
            scored: List[Dict[str, Any]] = []
        else:
            cached = decisions.get(key)
            if cached is None:
                manual = _manual_resolution(issue, legacy_id, resolution_map)
                if manual:
                    successor_id, reason, scored = manual, "resolution_map", []
                else:
                    successor_items = [catalog_by_id[str(row["id"])] for row in successor_rows]
                    successor_id, reason, scored = resolve_split_description(
                        _description(issue),
                        successor_items,
                        kind_hint=str(issue.get("canonical_kind") or ""),
                    )
                cached = (successor_id, reason, scored)
                decisions[key] = cached
            successor_id, reason, scored = cached

        if not successor_id or successor_id not in catalog_by_id or successor_id not in {
            str(row["id"]) for row in successor_rows
        }:
            unresolved_by_key[key] = {
                "issue_key": key,
                "issue_id": issue.get("issue_id"),
                "legacy_id": legacy_id,
                "description": _description(issue),
                "lane": lane,
                "reason": reason if not successor_id else "invalid_resolution_map_successor",
                "candidate_scores": scored,
                "successor_ids": [str(row["id"]) for row in successor_rows],
                "photo_key": issue.get("photo_key") or issue.get("source_photo_key"),
            }
            continue
        _stamp_issue(issue, catalog_by_id[successor_id], catalog_version)
        if len(successor_rows) == 1 and successor_id == legacy_id:
            same_keys.add(key)
        else:
            split_keys.add(key)

    result.unique_issues = len(unique_issue_keys)
    if result.unique_issues == 0 and artifact_ontology_version(artifact) != ONTOLOGY_VERSION:
        photos = artifact.get("photos") if isinstance(artifact.get("photos"), dict) else {}
        affected_keys = sorted(
            str(key) for key, photo in photos.items()
            if isinstance(photo, dict) and (
                photo.get("issues_natural_language")
                or ((photo.get("issues") or {}).get("final") if isinstance(photo.get("issues"), dict) else None)
            )
        )
        photo_paths = _photo_path_index(artifact)
        result.status = "needs_reanalysis"
        result.reason = "legacy artifact has no resolved issue lane to migrate"
        result.reanalysis_photo_keys = affected_keys
        result.reanalysis_image_paths = [
            photo_paths[key] for key in affected_keys if key in photo_paths
        ]
        return result
    result.same_id_updates = len(same_keys)
    result.split_updates = len(split_keys)
    result.unresolved = sorted(unresolved_by_key.values(), key=lambda row: row["issue_key"])
    photo_keys = sorted({str(row.get("photo_key")) for row in result.unresolved if row.get("photo_key")})
    photo_paths = _photo_path_index(working)
    result.reanalysis_photo_keys = photo_keys
    result.reanalysis_image_paths = [photo_paths[key] for key in photo_keys if key in photo_paths]
    if result.unresolved:
        result.status = "needs_reanalysis"
        result.reason = (
            "stored descriptions were insufficient for deterministic split resolution; "
            "reanalyze only the reported photos or provide --resolution-map"
        )
        return result

    working["catalog_version"] = catalog_version
    working["ontology_version"] = ONTOLOGY_VERSION
    working["product_projection_status"] = "needs_reprojection"
    for field_name in DERIVED_FIELDS:
        if field_name in working:
            working[field_name] = None
    working["kind_migration"] = {
        "migration": str(manifest.get("migration") or "2.1_to_3.0"),
        "source_ontology_version": artifact_ontology_version(artifact),
        "target_ontology_version": ONTOLOGY_VERSION,
        "catalog_version": catalog_version,
        "same_id_updates": result.same_id_updates,
        "split_updates": result.split_updates,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "resolution_policy": "same_id_metadata_or_unique_stored_description_support",
    }
    if not apply:
        result.status = "dry_run_ready"
        return result

    try:
        backup = _write_atomic_with_backup(artifact_path, working)
    except OSError as exc:
        result.reason = f"write failed: {exc}"
        return result
    result.backup_path = str(backup)
    result.status = "migrated"
    if reproject:
        projection = reproject_artifact(artifact_path, catalog, dry_run=False, force=True)
        result.reprojection = projection
        if projection.get("status") == "error":
            result.status = "error"
            result.reason = str(projection.get("reason") or "reprojection failed")
    return result


def _artifact_paths(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    if target.is_dir() and (target / "photo_intel.json").is_file():
        return [target / "photo_intel.json"]
    if target.is_dir():
        return sorted(target.rglob("photo_intel.json"))
    return []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="photo_intel.json, run directory, or artifact root")
    parser.add_argument("--catalog", default=str(ROOT / "tools" / "issue_catalog_kind_v2.json"))
    parser.add_argument("--manifest", default=str(ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0.json"))
    parser.add_argument("--resolution-map", type=Path)
    parser.add_argument("--apply", action="store_true", help="write migrations; default is dry-run")
    parser.add_argument("--no-reproject", action="store_true", help="leave product views nulled for a later reprojection")
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)

    catalog = load_issue_catalog(Path(args.catalog))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    try:
        resolution_map = _load_resolution_map(args.resolution_map)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: invalid resolution map: {exc}", file=sys.stderr)
        return 2
    paths = _artifact_paths(Path(args.target))
    if not paths:
        print(f"error: no photo_intel.json artifacts found under {args.target}", file=sys.stderr)
        return 2

    rows = [
        migrate_artifact(
            path,
            catalog,
            manifest,
            apply=args.apply,
            resolution_map=resolution_map,
            reproject=not args.no_reproject,
        ).to_dict()
        for path in paths
    ]
    report = {
        "mode": "apply" if args.apply else "dry_run",
        "catalog_version": catalog.get("version"),
        "ontology_version": catalog.get("ontology_version"),
        "totals": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({row["status"] for row in rows})
        },
        "artifacts": rows,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 1 if any(row["status"] in {"error", "needs_reanalysis"} for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())