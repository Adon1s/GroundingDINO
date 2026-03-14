"""
Deterministic property-level summary builder (no LLM).

Reads issues_flat + issue_catalog and produces a hierarchical summary
grouped by trade_bucket → scene_group → catalog_item (block).

Each block gets a computed display_severity based on:
  base_severity + evidence_boost + scope_boost + kind_boost + multi_scene_boost

Called from artifact_writers.write_photo_intel() alongside compute_estimates().

NOTE: This module replaces the LLM-based property_summarizer.py.
property_summarizer.py is dead code and can be safely deleted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Severity modifier constants
# ═══════════════════════════════════════════════════════════════════════════════

EVIDENCE_THRESHOLDS: List[Tuple[int, int]] = [(5, 3), (3, 2), (2, 1)]
SCOPE_BOOST: Dict[str, int] = {
    "cosmetic": 0,
    "repair": 1,
    "replace": 2,
    "service": 1,
    "unknown": 0,
}
KIND_BOOST: Dict[str, int] = {"defect": 0, "upgrade": -1}
MULTI_SCENE_BOOST = 2
MAX_UPGRADE_SEVERITY = 4

VALID_SCOPES = frozenset(SCOPE_BOOST.keys())
VALID_KINDS = frozenset({"defect", "upgrade"})

# Scope ordering for "max scope" within a block (higher index wins)
SCOPE_RANK = {"unknown": 0, "cosmetic": 1, "service": 2, "repair": 3, "replace": 4}


# ═══════════════════════════════════════════════════════════════════════════════
# Catalog index
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CatalogItem:
    id: str
    name: str
    severity: int       # 1-4 (clamped)
    trade_bucket: str
    scope: str          # cosmetic|repair|replace|service|unknown
    kind: str           # defect|upgrade


@dataclass
class CatalogIndex:
    items_by_id: Dict[str, CatalogItem]
    trade_bucket_name_by_id: Dict[str, str]


def _norm_scope(raw: Any) -> str:
    s = str(raw or "").lower().strip()
    return s if s in VALID_SCOPES else "unknown"


def _norm_kind(raw: Any) -> str:
    s = str(raw or "").lower().strip()
    return s if s in VALID_KINDS else "defect"


def load_catalog_index(issue_catalog: dict) -> CatalogIndex:
    """Build a CatalogIndex from an already-loaded catalog dict.

    Expects the normalized format returned by artifact_writers.load_issue_catalog()
    with 'items' list and 'trade_buckets' list.
    """
    items_by_id: Dict[str, CatalogItem] = {}
    for item in issue_catalog.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        item_id = (
            item.get("id")
            or item.get("defect_id")
            or item.get("upgrade_id")
            or ""
        )
        if not item_id:
            continue
        raw_sev = item.get("severity")
        severity = max(1, min(4, int(raw_sev))) if raw_sev is not None else 1
        items_by_id[item_id] = CatalogItem(
            id=item_id,
            name=item.get("name", item_id),
            severity=severity,
            trade_bucket=item.get("trade_bucket", ""),
            scope=_norm_scope(item.get("scope")),
            kind=_norm_kind(item.get("kind")),
        )

    tb_name_map: Dict[str, str] = {}
    for tb in issue_catalog.get("trade_buckets", []) or []:
        if isinstance(tb, dict):
            bid = tb.get("id", "")
            if bid:
                tb_name_map[bid] = tb.get("name", bid)

    return CatalogIndex(items_by_id=items_by_id, trade_bucket_name_by_id=tb_name_map)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp(lo: int, hi: int, val: int) -> int:
    return max(lo, min(hi, val))


def _evidence_boost(count: int) -> int:
    for threshold, boost in EVIDENCE_THRESHOLDS:
        if count >= threshold:
            return boost
    return 0


def _max_scope(a: str, b: str) -> str:
    return a if SCOPE_RANK.get(a, 0) >= SCOPE_RANK.get(b, 0) else b


# ═══════════════════════════════════════════════════════════════════════════════
# Block accumulator (internal)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _BlockAcc:
    catalog_item_id: str
    title: str
    trade_bucket: str
    scene_group: str
    base_severity: int
    scope_max: str
    kind: str
    issue_ids: Set[str] = field(default_factory=set)
    photo_keys: Set[str] = field(default_factory=set)


# ═══════════════════════════════════════════════════════════════════════════════
# Main builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_property_summary_v1(
    *,
    property_key: str,
    run_id: str,
    issues_flat: List[Dict[str, Any]],
    catalog_index: CatalogIndex,
) -> dict:
    """Build the SummaryV1 dict from a property-level issues_flat list.

    Args:
        property_key: Property identifier.
        run_id: Analysis run ID.
        issues_flat: Flat list of issue dicts (from write_photo_intel).
                     Each must have catalog_item_id, issue_id, photo_key,
                     scene, scene_group, catalog_item_kind.
        catalog_index: Pre-built CatalogIndex.

    Returns:
        SummaryV1 dict ready for JSON serialisation.
    """

    # ── Step A: Enrich + filter ───────────────────────────────────────────
    seen_issue_ids: Set[str] = set()
    block_map: Dict[str, _BlockAcc] = {}  # key = (tb, sg, item_id)

    for issue in issues_flat:
        cat_id = issue.get("catalog_item_id")
        if not cat_id:
            continue
        cat_item = catalog_index.items_by_id.get(cat_id)
        if cat_item is None:
            continue

        issue_id = issue.get("issue_id", "")
        if not issue_id or issue_id in seen_issue_ids:
            continue
        seen_issue_ids.add(issue_id)

        photo_key = issue.get("photo_key", "")
        scene_group = issue.get("scene_group") or "other"
        kind = _norm_kind(issue.get("catalog_item_kind") or cat_item.kind)

        block_key = f"{cat_item.trade_bucket}|{scene_group}|{cat_id}"
        acc = block_map.get(block_key)
        if acc is None:
            acc = _BlockAcc(
                catalog_item_id=cat_id,
                title=cat_item.name,
                trade_bucket=cat_item.trade_bucket,
                scene_group=scene_group,
                base_severity=cat_item.severity,
                scope_max=cat_item.scope,
                kind=kind,
            )
            block_map[block_key] = acc

        acc.issue_ids.add(issue_id)
        if photo_key:
            acc.photo_keys.add(photo_key)
        acc.scope_max = _max_scope(acc.scope_max, cat_item.scope)
        # defect wins over upgrade if mixed
        if kind == "defect":
            acc.kind = "defect"

    if not block_map:
        return _empty_summary(property_key, run_id)

    # ── Step B: Detect multi-scene catalog items per trade bucket ─────────
    multi_scene_keys: Set[str] = set()  # keys = "tb|item_id"
    tb_item_scenes: Dict[str, Set[str]] = {}
    for acc in block_map.values():
        ms_key = f"{acc.trade_bucket}|{acc.catalog_item_id}"
        scenes = tb_item_scenes.setdefault(ms_key, set())
        scenes.add(acc.scene_group)

    for ms_key, scenes in tb_item_scenes.items():
        if len(scenes) > 1:
            multi_scene_keys.add(ms_key)

    # ── Step C: Compute display_severity per block ────────────────────────
    finalized_blocks: List[dict] = []
    for acc in block_map.values():
        evidence_count = len(acc.photo_keys)
        ms_key = f"{acc.trade_bucket}|{acc.catalog_item_id}"
        is_multi_scene = ms_key in multi_scene_keys

        eb = _evidence_boost(evidence_count)
        sb = SCOPE_BOOST.get(acc.scope_max, 0)
        kb = KIND_BOOST.get(acc.kind, 0)
        msb = MULTI_SCENE_BOOST if is_multi_scene else 0
        raw = acc.base_severity + eb + sb + kb + msb

        display_severity = _clamp(1, 5, raw)

        # Cap upgrades at MAX_UPGRADE_SEVERITY (Fix 2)
        if acc.kind == "upgrade":
            display_severity = min(display_severity, MAX_UPGRADE_SEVERITY)

        finalized_blocks.append({
            "block_id": f"{acc.trade_bucket}:{acc.scene_group}:{acc.catalog_item_id}",
            "title": acc.title,
            "base_severity": acc.base_severity,
            "display_severity": display_severity,
            "kind": acc.kind,
            "scope_max": acc.scope_max,
            "trade_bucket": acc.trade_bucket,
            "item_ids": [acc.catalog_item_id],
            "issue_ids": sorted(acc.issue_ids),
            "photo_keys": sorted(acc.photo_keys),
            "evidence_count": evidence_count,
            "severity_calc": {
                "base": acc.base_severity,
                "evidence_boost": eb,
                "scope_boost": sb,
                "kind_boost": kb,
                "multi_scene_boost": msb,
                "raw_total": raw,
                "capped": display_severity != raw,
            },
            # internal grouping keys (not in final output, stripped below)
            "_scene_group": acc.scene_group,
            "_trade_bucket": acc.trade_bucket,
        })

    # ── Step D: Assemble hierarchy ────────────────────────────────────────
    # Group blocks into buckets → scene_groups
    bucket_map: Dict[str, Dict[str, List[dict]]] = {}  # tb → sg → [blocks]
    for blk in finalized_blocks:
        tb = blk["_trade_bucket"]
        sg = blk["_scene_group"]
        bucket_map.setdefault(tb, {}).setdefault(sg, []).append(blk)

    buckets: List[dict] = []
    for tb_id, sg_map in bucket_map.items():
        tb_name = catalog_index.trade_bucket_name_by_id.get(tb_id, tb_id)

        scenes: List[dict] = []
        all_bucket_blocks: List[dict] = []
        for sg, blocks in sg_map.items():
            # Sort blocks: display_severity desc, evidence_count desc, title asc
            blocks.sort(key=lambda b: (-b["display_severity"], -b["evidence_count"], b["title"]))

            scene_issue_count = sum(len(b["issue_ids"]) for b in blocks)
            scene_top_sev = max(b["display_severity"] for b in blocks)

            # Strip internal keys before output
            clean_blocks = []
            for b in blocks:
                cb = {k: v for k, v in b.items() if not k.startswith("_")}
                clean_blocks.append(cb)
                all_bucket_blocks.append(b)

            scenes.append({
                "scene_group": sg,
                "top_severity": scene_top_sev,
                "issue_count": scene_issue_count,
                "blocks": clean_blocks,
            })

        # Sort scenes: top_severity desc, issue_count desc
        scenes.sort(key=lambda s: (-s["top_severity"], -s["issue_count"], s["scene_group"]))

        bucket_issue_count = sum(len(b["issue_ids"]) for b in all_bucket_blocks)
        bucket_defect_count = sum(
            len(b["issue_ids"]) for b in all_bucket_blocks if b["kind"] == "defect"
        )
        bucket_upgrade_count = bucket_issue_count - bucket_defect_count
        bucket_top_sev = max(b["display_severity"] for b in all_bucket_blocks)

        # Max base severity among defect blocks (for ranking)
        max_base_sev_defect = max(
            (b["base_severity"] for b in all_bucket_blocks if b["kind"] == "defect"),
            default=0,
        )

        # Summary line: scene-aware, deduped titles (Fix 1)
        summary_line = _make_bucket_summary_line(all_bucket_blocks)

        buckets.append({
            "bucket_id": tb_id,
            "bucket_name": tb_name,
            "top_severity": bucket_top_sev,
            "issue_count": bucket_issue_count,
            "defect_count": bucket_defect_count,
            "upgrade_count": bucket_upgrade_count,
            "max_base_severity_defect": max_base_sev_defect,
            "summary_line": summary_line,
            "scenes": scenes,
        })

    # ── Step E: Sort buckets by rank score (Fix 3) ────────────────────────
    # rank_score = top_severity + 1 if has defects → defect buckets outrank
    # pure-upgrade buckets at same severity. Then max_base_severity_defect
    # to prefer high-base defects (Electrical base-4 > Landscaping base-2).
    buckets.sort(key=lambda b: (
        -(b["top_severity"] + (1 if b["defect_count"] > 0 else 0)),
        -b["max_base_severity_defect"],
        -b["defect_count"],
        -b["issue_count"],
        b["bucket_name"],
    ))

    # ── Step F: Listing-level summary ─────────────────────────────────────
    total_defect_count = sum(b["defect_count"] for b in buckets)
    total_upgrade_count = sum(b["upgrade_count"] for b in buckets)
    listing_top_sev = max(b["top_severity"] for b in buckets) if buckets else 0
    buckets_touched = [b["bucket_id"] for b in buckets]
    one_liner = _make_listing_one_liner(buckets)

    return {
        "version": "1.0",
        "property_key": property_key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "listing": {
            "top_severity": listing_top_sev,
            "defect_count": total_defect_count,
            "upgrade_count": total_upgrade_count,
            "buckets_touched": buckets_touched,
            "one_liner": one_liner,
        },
        "buckets": buckets,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Text generators
# ═══════════════════════════════════════════════════════════════════════════════

def _make_bucket_summary_line(all_bucket_blocks: List[dict]) -> str:
    """Build a scene-aware summary line from all blocks in a bucket.

    Deduplicates titles and appends scene_group context when the same
    catalog item appears in multiple scenes.
    E.g. "Top concerns: Outdated Flooring Style (bathroom, bedroom); scratched flooring."
    """
    if not all_bucket_blocks:
        return ""

    # Sort by severity desc, evidence desc, title asc
    sorted_blocks = sorted(
        all_bucket_blocks,
        key=lambda b: (-b["display_severity"], -b["evidence_count"], b["title"]),
    )

    # Collect unique titles preserving sort order, with scene_groups
    seen_titles: Dict[str, List[str]] = {}  # title → [scene_groups]
    title_order: List[str] = []
    for b in sorted_blocks:
        t = b["title"]
        sg = b.get("_scene_group", "")
        if t not in seen_titles:
            seen_titles[t] = []
            title_order.append(t)
        if sg and sg not in seen_titles[t]:
            seen_titles[t].append(sg)

    # Format title with scene context when multi-scene
    def _fmt(title: str) -> str:
        scenes = seen_titles[title]
        if len(scenes) > 1:
            return f"{title} ({', '.join(scenes)})"
        return title

    if len(title_order) == 1:
        return f"Top concern: {_fmt(title_order[0])}."

    t1 = _fmt(title_order[0])
    t2_raw = _fmt(title_order[1])
    # Lowercase second title only if it starts with uppercase
    t2 = t2_raw[0].lower() + t2_raw[1:] if t2_raw and t2_raw[0].isupper() else t2_raw
    return f"Top concerns: {t1}; {t2}."


def _make_listing_one_liner(buckets: List[dict]) -> str:
    if not buckets:
        return ""
    if len(buckets) == 1:
        return f"Most notable: {buckets[0]['bucket_name']}."
    if len(buckets) == 2:
        return f"Most notable: {buckets[0]['bucket_name']} and {buckets[1]['bucket_name']}."
    other_count = len(buckets) - 2
    area_word = "area" if other_count == 1 else "areas"
    return (
        f"Most notable: {buckets[0]['bucket_name']} and {buckets[1]['bucket_name']}; "
        f"additional issues noted in {other_count} other {area_word}."
    )


def _empty_summary(property_key: str, run_id: str) -> dict:
    return {
        "version": "1.0",
        "property_key": property_key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "listing": {
            "top_severity": 0,
            "defect_count": 0,
            "upgrade_count": 0,
            "buckets_touched": [],
            "one_liner": "",
        },
        "buckets": [],
    }
