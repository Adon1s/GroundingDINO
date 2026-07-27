"""Searchable catalog view for blind annotation.

Phase-1 annotation needs to find the right catalog item without reading 106
JSON objects by eye. The catalog is authored by hand and is not model output, so
consulting it during blind annotation does not anchor the reference to any
model's behavior — unlike prefilling from a run, which would.

Search is substring-based over the fields an annotator actually thinks in:
id, name, description, trade bucket, and the support keywords the retrieval
layer already uses for the same purpose.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from tools.benchmarking.vocabulary import default_actionability

# Fields searched, in match-priority order. `support_any` is included because it
# is the catalog's own synonym list -- the annotator's phrasing ("water stain")
# is far likelier to appear there than in a formal name.
_TEXT_FIELDS = ("id", "name", "description", "trade_bucket", "work_item_code")
_LIST_FIELDS = ("support_any", "scene_groups")


def _haystack(item: Dict[str, Any]) -> str:
    parts: List[str] = [str(item.get(field) or "") for field in _TEXT_FIELDS]
    for field in _LIST_FIELDS:
        parts.extend(str(v) for v in (item.get(field) or []))
    return " ␟ ".join(parts).lower()


def summarize(item: Dict[str, Any]) -> Dict[str, Any]:
    """The fields that matter when choosing an item, including the actionability
    the reference will derive so the annotator sees it before committing."""
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "kind": item.get("kind"),
        "scope": item.get("scope"),
        "tier": item.get("tier"),
        "trade_bucket": item.get("trade_bucket"),
        "scene_groups": list(item.get("scene_groups") or []),
        "actionability": default_actionability(item),
    }


def search(
    issue_catalog: Dict[str, Any],
    query: str,
    *,
    scene_group: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Return matching catalog items, best match first.

    An empty query lists everything (optionally filtered), which is how an
    annotator browses a room's plausible items rather than guessing keywords.
    """
    terms = [t for t in str(query or "").lower().split() if t]
    matches: List[tuple] = []

    for item in issue_catalog.get("items") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        if scene_group and scene_group not in (item.get("scene_groups") or []):
            continue
        if kind and str(item.get("kind") or "") != kind:
            continue

        haystack = _haystack(item)
        if terms and not all(term in haystack for term in terms):
            continue

        # Rank: an id or name hit beats a synonym hit, so exact-ish matches
        # surface first without needing a real scoring model.
        identity = f"{item.get('id')} {item.get('name')}".lower()
        rank = 0 if terms and all(term in identity for term in terms) else 1
        matches.append((rank, str(item.get("id")), item))

    matches.sort(key=lambda row: (row[0], row[1]))
    return [summarize(item) for _, _, item in matches[:limit]]


def format_results(results: Iterable[Dict[str, Any]]) -> str:
    """Aligned plain-text output for the CLI."""
    rows = list(results)
    if not rows:
        return "no matching catalog items"
    width = max(len(str(row["id"])) for row in rows)
    lines = []
    for row in rows:
        groups = ",".join(row["scene_groups"]) or "-"
        lines.append(
            f"{str(row['id']):<{width}}  {row['kind']:<8} {row['scope'] or '-':<9} "
            f"{row['actionability']:<15} {row['trade_bucket'] or '-':<28} {groups}"
        )
        lines.append(f"{'':<{width}}  {row['name']}")
    return "\n".join(lines)
