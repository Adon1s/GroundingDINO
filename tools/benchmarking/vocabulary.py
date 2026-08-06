"""Frozen benchmark vocabulary.

Sealed human truth must never be validated against live production code.
If a gold reference validated against the current catalog, then renaming a
catalog id, consolidating a scene, or retiring a package type would invalidate
the gold set *before* you could measure the effect of that very change — the
benchmark would break exactly when you needed it.

So at seal time the live vocabularies are written down into
``reference_vocabulary.json``. References validate against their own frozen
snapshot forever. ``diff()`` reports how live code has drifted from a snapshot,
and the evaluator uses that to map current pipeline output into the frozen
vocabulary rather than rejecting it.

Two classes of vocabulary live here:

* **Mirrored** - owned by a runtime module and imported, never restated, so a
  snapshot can't disagree with the pipeline it describes.
* **Benchmark-owned** - annotation concepts with no runtime owner (presence,
  reporting expectation, review status). Declared here once, following the
  precedent in ``tools/catalog_validation.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.catalog_validation import (
    VALID_SCENE_GROUP_TOKENS,
    VALID_SCOPES as CATALOG_SCOPES,
)
from tools.comparison_common import ComparisonError, sha256_canonical
from tools.estimate_scope import VALID_ESTIMATE_SCOPES
from tools.observation_kinds import (
    LEGACY_CATALOG_KINDS,
    OBSERVATION_KINDS,
    ONTOLOGY_VERSION,
)
from tools.pipeline_common import SCENE_GROUPS_UI, SCENE_TO_GROUP_UI
from tools.rehab_packages import (
    PACKAGE_CATEGORY_INSPECTION_RISK,
    PACKAGE_CATEGORY_MODERNIZATION,
    PACKAGE_CATEGORY_REPAIR,
    PACKAGE_CATEGORY_TURNOVER,
    VALID_PACKAGE_CATEGORIES,
    VALID_PACKAGE_LEVELS,
    VALID_PACKAGE_TYPES,
    VALID_ROOMS,
)

VOCABULARY_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Benchmark-owned vocabularies (no runtime owner)
# ---------------------------------------------------------------------------

VALID_PRESENCE = frozenset({"present", "absent", "indeterminate"})
VALID_REPORTING_EXPECTATIONS = frozenset({"required", "acceptable", "must_not_report"})
VALID_VISUAL_SUFFICIENCY = frozenset({"sufficient", "limited", "insufficient"})
VALID_CATALOG_STATUSES = frozenset({"matched", "missing_catalog_item", "ambiguous_mapping"})

# Human package judgment. Distinct from the pipeline's verification statuses
# (confirmed/rejected/uncertain/not_run) because "expected" is a statement about
# what *should* happen, not a record of what a model decided.
VALID_PACKAGE_DECISIONS = frozenset({"expected", "acceptable", "unsupported", "indeterminate"})

VALID_TIERS = frozenset({"gold", "silver"})
VALID_REVIEW_STATUSES = frozenset({"draft", "reviewed", "sealed"})
VALID_REFERENCE_COVERAGE = frozenset({"targeted", "exhaustive"})
VALID_BAND_CONFIDENCE = frozenset({"low", "medium", "high"})

# Annotation phases, per benchmarks/README.md:
#   1 blind human inventory, 2 blind package judgment,
#   3 model-assisted reconciliation, 4 cost review.
VALID_ANNOTATION_PHASES = frozenset({1, 2, 3, 4})

# "exhaustive" is meaningless without saying exhaustive *over what*. Photographs
# cannot establish hidden electrical, plumbing, foundation, or HVAC condition,
# so hidden_system_conditions is expected to be false for photo-only references.
COVERAGE_DOMAIN_KEYS = (
    "visible_conditions",
    "hidden_system_conditions",
    "exterior",
    "interior",
    "marketability_opportunities",
)

# Rehab scope bands have NO backend owner: the band is computed in the frontend
# (lib/topPicks/scopeBand.ts) from the package set, and typed as RehabScopeBand
# in lib/types/topPicks.ts. Mirrored here by hand because it is cross-repo; if
# the frontend union changes, this must change with it.
REHAB_SCOPE_BANDS = ("light", "moderate", "heavy")

# Human actionability reuses the pipeline's package categories so the benchmark
# does not invent a parallel vocabulary, plus one value the pipeline has no need
# for: a condition that is real and visible but that no one should pay to fix.
NONACTIONABLE = "nonactionable"
VALID_ACTIONABILITY = frozenset(VALID_PACKAGE_CATEGORIES) | {NONACTIONABLE}

# Trade bucket whose presence alone identifies turnover work.
_TURNOVER_TRADE_BUCKET = "cleaning_turnover"


def default_actionability(item: Dict[str, Any]) -> str:
    """Derive a finding's actionability from its catalog item.

    A matched catalog item already encodes this: ``kind`` separates discretionary
    work from repair work and ``scope`` separates service calls from physical
    work. Deriving it saves the annotator a field on every finding, and the
    derivation is frozen into the snapshot so a later rule change cannot
    silently restate old references. Only ``missing_catalog_item`` findings
    need it hand-authored.

    Four ordered rules, no per-item special cases:
      cleaning_turnover trade          -> turnover
      scope "service"                  -> inspection_risk
      kind "upgrade"/"modernization"   -> modernization
      otherwise (defect, degradation)  -> repair

    Note the name collision: "modernization" is both an observation kind
    (observation-kind-v2) and a package category. The mapping is the identity
    for that kind, but the two vocabularies are distinct — degradation is a
    kind with no category namesake, and it derives to repair because
    deterioration is something you pay to fix.
    """
    if str(item.get("trade_bucket") or "") == _TURNOVER_TRADE_BUCKET:
        return PACKAGE_CATEGORY_TURNOVER
    if str(item.get("scope") or "") == "service":
        return PACKAGE_CATEGORY_INSPECTION_RISK
    if str(item.get("kind") or "") in ("upgrade", "modernization"):
        return PACKAGE_CATEGORY_MODERNIZATION
    return PACKAGE_CATEGORY_REPAIR


# Catalog item fields the reference depends on. Everything else (costs,
# embeddings, affinity blocks) is pipeline machinery the benchmark never reads,
# and snapshotting it would make every cost tweak look like vocabulary drift.
_SNAPSHOT_ITEM_FIELDS = ("name", "kind", "scope", "tier", "trade_bucket")


def snapshot(issue_catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Freeze the live vocabularies into a serializable snapshot.

    Fails closed on an empty catalog: ``load_issue_catalog`` deliberately
    swallows a missing file and returns ``{"items": []}`` so a bad
    ISSUE_CATALOG_PATH cannot take down the analyzer server. Sealing a
    zero-item vocabulary would be far worse than a loud error here, because
    every reference validated against it would accept any catalog id at all.
    """
    items = issue_catalog.get("items") or []
    if not items:
        raise ComparisonError(
            "issue catalog has no items; refusing to snapshot an empty vocabulary "
            "(check ISSUE_CATALOG_PATH - load_issue_catalog returns an empty "
            "catalog rather than raising when the file is missing)"
        )

    catalog_items: Dict[str, Dict[str, Any]] = {}
    actionability: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "")
        if not item_id:
            continue
        catalog_items[item_id] = {
            field: item.get(field) for field in _SNAPSHOT_ITEM_FIELDS
        }
        actionability[item_id] = default_actionability(item)

    if not catalog_items:
        raise ComparisonError("issue catalog contains no items with an 'id'")

    snap: Dict[str, Any] = {
        "vocabulary_schema_version": VOCABULARY_SCHEMA_VERSION,
        "catalog_version": str(issue_catalog.get("version") or ""),
        # scene_id -> UI group. Both directions are derivable from this one map.
        "scene_ids": dict(sorted(SCENE_TO_GROUP_UI.items())),
        "ui_scene_groups": sorted(SCENE_GROUPS_UI),
        # Retrieval tokens the catalog is allowed to carry in scene_groups. This
        # is deliberately NOT the same set as ui_scene_groups: "pool" is a scene
        # whose UI group is "exterior", yet items carry it redundantly. See
        # catalog_validation.VALID_SCENE_GROUP_TOKENS. Recording both separately
        # preserves the skew instead of quietly reconciling it.
        "catalog_scene_group_tokens": sorted(VALID_SCENE_GROUP_TOKENS),
        "catalog_items": dict(sorted(catalog_items.items())),
        "actionability_by_catalog_item": dict(sorted(actionability.items())),
        # Versioned off the catalog's own root metadata (the
        # catalog_validation pattern): a v2-stamped catalog seals the
        # three-kind ontology, anything else seals the legacy pair. A snapshot
        # must describe the catalog it froze, not whichever vocabulary is
        # newest.
        "catalog_kinds": sorted(
            OBSERVATION_KINDS
            if issue_catalog.get("ontology_version") == ONTOLOGY_VERSION
            else LEGACY_CATALOG_KINDS
        ),
        "catalog_scopes": sorted(CATALOG_SCOPES),
        "package_types": sorted(VALID_PACKAGE_TYPES),
        "package_categories": sorted(VALID_PACKAGE_CATEGORIES),
        "package_rooms": sorted(VALID_ROOMS),
        "package_levels": sorted(VALID_PACKAGE_LEVELS),
        "estimate_scopes": sorted(VALID_ESTIMATE_SCOPES),
        "rehab_scope_bands": list(REHAB_SCOPE_BANDS),
        "actionability": sorted(VALID_ACTIONABILITY),
    }
    snap["fingerprint"] = fingerprint(snap)
    return snap


def fingerprint(snap: Dict[str, Any]) -> str:
    """Fingerprint a snapshot, excluding any fingerprint already on it."""
    return sha256_canonical({k: v for k, v in snap.items() if k != "fingerprint"})


def load(path: Path) -> Dict[str, Any]:
    """Load a frozen snapshot and verify it has not been edited in place.

    A hand-edited vocabulary is the one way a sealed reference could start
    accepting values it was never reviewed against, so the fingerprint is
    checked on every load rather than only at seal time.
    """
    import json

    if not path.is_file():
        raise ComparisonError(f"frozen vocabulary not found: {path}")
    try:
        snap = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ComparisonError(f"frozen vocabulary is unreadable: {path}: {exc}") from exc
    if not isinstance(snap, dict):
        raise ComparisonError(f"frozen vocabulary must be an object: {path}")

    recorded = str(snap.get("fingerprint") or "")
    actual = fingerprint(snap)
    if recorded != actual:
        raise ComparisonError(
            f"frozen vocabulary fingerprint mismatch in {path}: recorded {recorded or '(none)'}, "
            f"computed {actual}. The file was edited after sealing; restore it or "
            f"seal a new dataset version."
        )
    if snap.get("vocabulary_schema_version") != VOCABULARY_SCHEMA_VERSION:
        raise ComparisonError(
            f"frozen vocabulary schema version {snap.get('vocabulary_schema_version')!r} "
            f"is not supported (expected {VOCABULARY_SCHEMA_VERSION})"
        )
    return snap


# Vocabularies compared as flat sets by diff(). catalog_items is handled
# separately because its values are structured.
_DIFFABLE_SETS = (
    "ui_scene_groups",
    "catalog_scene_group_tokens",
    "package_types",
    "package_categories",
    "package_rooms",
    "package_levels",
    "estimate_scopes",
    "rehab_scope_bands",
    "actionability",
    "catalog_kinds",
    "catalog_scopes",
)


def diff(frozen: Dict[str, Any], live: Optional[Dict[str, Any]] = None,
         *, issue_catalog: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Report how live code has drifted from a frozen snapshot.

    Renames are reported as a removal plus an addition, not guessed at: with
    only ids and labels there is no reliable way to tell a rename from an
    unrelated pair of edits, and a wrong guess would silently remap human truth
    onto the wrong catalog item.
    """
    if live is None:
        if issue_catalog is None:
            raise ComparisonError("diff() needs either a live snapshot or an issue_catalog")
        live = snapshot(issue_catalog)

    changes: Dict[str, Any] = {
        "compatible": True,
        "catalog_version": {
            "frozen": frozen.get("catalog_version"),
            "live": live.get("catalog_version"),
        },
        "sets": {},
        "scene_ids": {"added": [], "removed": [], "regrouped": []},
        "catalog_items": {"added": [], "removed": [], "changed": []},
    }

    for key in _DIFFABLE_SETS:
        frozen_set = set(frozen.get(key) or ())
        live_set = set(live.get(key) or ())
        added = sorted(live_set - frozen_set)
        removed = sorted(frozen_set - live_set)
        if added or removed:
            changes["sets"][key] = {"added": added, "removed": removed}

    frozen_scenes: Dict[str, str] = frozen.get("scene_ids") or {}
    live_scenes: Dict[str, str] = live.get("scene_ids") or {}
    changes["scene_ids"]["added"] = sorted(set(live_scenes) - set(frozen_scenes))
    changes["scene_ids"]["removed"] = sorted(set(frozen_scenes) - set(live_scenes))
    changes["scene_ids"]["regrouped"] = [
        {"scene_id": scene, "frozen_group": frozen_scenes[scene], "live_group": live_scenes[scene]}
        for scene in sorted(set(frozen_scenes) & set(live_scenes))
        if frozen_scenes[scene] != live_scenes[scene]
    ]

    frozen_items: Dict[str, Any] = frozen.get("catalog_items") or {}
    live_items: Dict[str, Any] = live.get("catalog_items") or {}
    changes["catalog_items"]["added"] = sorted(set(live_items) - set(frozen_items))
    changes["catalog_items"]["removed"] = sorted(set(frozen_items) - set(live_items))
    changes["catalog_items"]["changed"] = [
        {"id": item_id, "frozen": frozen_items[item_id], "live": live_items[item_id]}
        for item_id in sorted(set(frozen_items) & set(live_items))
        if frozen_items[item_id] != live_items[item_id]
    ]

    # Only removals can invalidate existing truth. Additions mean the pipeline
    # can now say something the reference never reviewed, which the evaluator
    # reports as an unreviewed prediction rather than a failure.
    changes["compatible"] = not (
        changes["catalog_items"]["removed"]
        or changes["scene_ids"]["removed"]
        or changes["scene_ids"]["regrouped"]
        or any(v["removed"] for v in changes["sets"].values())
    )
    return changes


def summarize_diff(changes: Dict[str, Any]) -> List[str]:
    """One human-readable line per drift, for CLI output."""
    lines: List[str] = []
    frozen_ver = changes["catalog_version"]["frozen"]
    live_ver = changes["catalog_version"]["live"]
    if frozen_ver != live_ver:
        lines.append(f"catalog version: frozen {frozen_ver!r} -> live {live_ver!r}")
    for key, delta in sorted(changes.get("sets", {}).items()):
        if delta["removed"]:
            lines.append(f"{key}: REMOVED {', '.join(delta['removed'])}")
        if delta["added"]:
            lines.append(f"{key}: added {', '.join(delta['added'])}")
    scenes = changes.get("scene_ids", {})
    if scenes.get("removed"):
        lines.append(f"scene_ids: REMOVED {', '.join(scenes['removed'])}")
    if scenes.get("added"):
        lines.append(f"scene_ids: added {', '.join(scenes['added'])}")
    for entry in scenes.get("regrouped", ()):
        lines.append(
            f"scene_ids: REGROUPED {entry['scene_id']}: "
            f"{entry['frozen_group']} -> {entry['live_group']}"
        )
    items = changes.get("catalog_items", {})
    if items.get("removed"):
        lines.append(f"catalog_items: REMOVED {', '.join(items['removed'])}")
    if items.get("added"):
        lines.append(f"catalog_items: added {len(items['added'])} ({', '.join(items['added'][:5])}...)"
                     if len(items["added"]) > 5
                     else f"catalog_items: added {', '.join(items['added'])}")
    for entry in items.get("changed", ()):
        lines.append(f"catalog_items: changed {entry['id']}")
    return lines
