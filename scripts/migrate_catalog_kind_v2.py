"""Generate the observation-kind-v2 catalog, migration manifest, and audit report.

Deterministic, byte-stable transformation:

    tools/issue_catalog.json  (v1, untouched)
  + tools/catalog_migrations/kind_v2_decisions.json  (the reviewable source of truth)
  ->  tools/issue_catalog_kind_v2.json               (catalog version 3.0, publishable)
      tools/catalog_migrations/2.1_to_3.0.json       (audit-only manifest, one entry per legacy id)
      tools/catalog_migrations/2.1_to_3.0_audit.md   (generated audit report)

Inheritance rules are code, not authoring discipline:
- unchanged/reclassified/narrowed items copy the legacy item verbatim, apply the
  v2 kind + atomic_claim + optional wording overrides, and KEEP every economic
  field byte-identical.
- split successors are authored fresh (name/description/embed_text/support_any/
  severity/kind/atomic_claim required), inherit non-economic structural fields
  from the parent, and — as the Task 4A bridge until the dedicated pricing
  project — inherit the parent's economic fields verbatim, stamped
  pricing_status=inherited_from_split_parent. Successor overrides can never
  rewrite an economic field (hard error, not convention).

Run:  .venv/Scripts/python.exe scripts/migrate_catalog_kind_v2.py
Running twice produces byte-identical outputs (pinned by the parity test in
tests/test_catalog_kind_v2.py).
"""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.catalog_validation import ECONOMIC_FIELDS, PRICING_STATUS_INHERITED  # noqa: E402
from tools.observation_kinds import OBSERVATION_KINDS, ONTOLOGY_VERSION  # noqa: E402

V1_PATH = ROOT / "tools" / "issue_catalog.json"
DECISIONS_PATH = ROOT / "tools" / "catalog_migrations" / "kind_v2_decisions.json"
V2_PATH = ROOT / "tools" / "issue_catalog_kind_v2.json"
MANIFEST_PATH = ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0.json"
AUDIT_PATH = ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0_audit.md"

TARGET_VERSION = "3.0"
PUBLICATION_STATUS = "publishable"

# The one pricing policy generate() accepts from the decisions file. Split
# successors carry their parent's economics verbatim until the dedicated
# pricing project authors real successor prices.
SPLIT_SUCCESSOR_PRICING_POLICY = "inherit_parent_economics_v1"

# Structural, non-economic fields a split successor inherits from its parent
# when it does not author its own value.
INHERITED_FIELDS = (
    "trade_bucket",
    "scope",
    "tier",
    "defaultHidden",
    "drop_if_generic",
    "category",
    "display_class",
    "require_any",
    "deny_any",
    "scene_groups",
)

# Fields a non-split entry's `overrides` may rewrite (wording only).
WORDING_OVERRIDE_FIELDS = frozenset({"name", "description", "embed_text", "support_any", "deny_any"})

SUCCESSOR_REQUIRED_FIELDS = ("id", "kind", "severity", "name", "description", "embed_text", "support_any")


class MigrationError(SystemExit):
    pass


def _fail(msg: str) -> None:
    raise MigrationError(f"migrate_catalog_kind_v2: {msg}")


def _build_split_successor(parent: dict, succ: dict) -> dict:
    for field in SUCCESSOR_REQUIRED_FIELDS:
        if not succ.get(field):
            _fail(f"split successor of {parent['id']!r} missing required field {field!r}")
    overrides = succ.get("overrides") or {}
    bad = set(overrides) & set(ECONOMIC_FIELDS)
    if bad:
        _fail(f"successor {succ['id']!r} override touches economic fields {sorted(bad)}")

    item = {
        "id": succ["id"],
        "name": succ["name"],
        "kind": succ["kind"],
        "severity": succ["severity"],
    }
    for field in INHERITED_FIELDS:
        if field in overrides:
            item[field] = overrides[field]
        elif field in parent:
            item[field] = parent[field]
    item["description"] = succ["description"]
    item["embed_text"] = succ["embed_text"]
    item["support_any"] = succ["support_any"]
    item["atomic_claim"] = succ["atomic_claim"]
    for field in ECONOMIC_FIELDS:
        if field in parent:
            item[field] = copy.deepcopy(parent[field])
    item["pricing_status"] = PRICING_STATUS_INHERITED
    return item


def _build_carryover(parent: dict, succ: dict) -> dict:
    overrides = succ.get("overrides") or {}
    bad = set(overrides) - WORDING_OVERRIDE_FIELDS
    if bad:
        _fail(f"non-split entry {parent['id']!r} override touches non-wording fields {sorted(bad)}")
    item = dict(parent)  # verbatim copy, insertion order preserved
    item["kind"] = succ["kind"]
    item.update(overrides)
    item["atomic_claim"] = succ["atomic_claim"]
    return item


def generate(v1: dict, decisions: dict) -> tuple[dict, dict]:
    v1_items = v1.get("items") or []
    v1_by_id = {it["id"]: it for it in v1_items}
    entries = decisions.get("entries") or []

    pricing_policy = (decisions.get("split_successor_pricing") or {}).get("policy")
    if pricing_policy != SPLIT_SUCCESSOR_PRICING_POLICY:
        _fail(
            f"decisions must declare split_successor_pricing.policy == "
            f"{SPLIT_SUCCESSOR_PRICING_POLICY!r} (got {pricing_policy!r})"
        )

    decided = [e["legacy_id"] for e in entries]
    if len(decided) != len(set(decided)):
        dupes = [i for i, n in Counter(decided).items() if n > 1]
        _fail(f"duplicate decisions entries for {dupes}")
    missing = set(v1_by_id) - set(decided)
    extra = set(decided) - set(v1_by_id)
    if missing or extra:
        _fail(f"decisions must cover exactly the legacy ids (missing={sorted(missing)}, extra={sorted(extra)})")

    by_legacy = {e["legacy_id"]: e for e in entries}
    items: list[dict] = []
    seen_ids: set[str] = set()
    for v1_item in v1_items:  # v1 catalog order; successors in decisions order
        entry = by_legacy[v1_item["id"]]
        change_type = entry["change_type"]
        for succ in entry["successors"]:
            kind = succ.get("kind")
            if kind not in OBSERVATION_KINDS:
                _fail(f"successor {succ.get('id')!r} of {v1_item['id']!r} has invalid kind {kind!r}")
            if change_type == "split":
                if succ["id"] in v1_by_id:
                    _fail(f"split successor id {succ['id']!r} collides with a legacy id")
                item = _build_split_successor(v1_item, succ)
            else:
                if succ["id"] != v1_item["id"]:
                    _fail(f"non-split entry {v1_item['id']!r} must keep its id (got {succ['id']!r})")
                item = _build_carryover(v1_item, succ)
            if item["id"] in seen_ids:
                _fail(f"duplicate v2 item id {item['id']!r}")
            seen_ids.add(item["id"])
            items.append(item)

    catalog = {
        "version": TARGET_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "publication_status": PUBLICATION_STATUS,
        "trade_buckets": v1.get("trade_buckets") or [],
        "items": items,
    }

    manifest = {
        "migration": "2.1_to_3.0",
        "source_version": str(v1.get("version") or ""),
        "target_version": TARGET_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "audit_only": True,
        "note": (
            "Audit record of the v1->v2 kind migration. NOT a runtime alias table: "
            "historical artifacts are migrated by re-resolution (Task 4B), never "
            "aliased in place."
        ),
        "entries": [
            {
                "legacy_id": e["legacy_id"],
                "legacy_kind": e["legacy_kind"],
                "change_type": e["change_type"],
                "deprecated": e["deprecated"],
                "requires_re_resolution": e["requires_re_resolution"],
                "successors": [{"id": s["id"], "kind": s["kind"]} for s in e["successors"]],
                "atomicity_rationale": e["atomicity_rationale"],
                "expected_effects": e["expected_effects"],
                **({"task3_flags": e["task3_flags"]} if e.get("task3_flags") else {}),
            }
            for e in entries
        ],
    }
    return catalog, manifest


def render_audit(catalog: dict, manifest: dict, decisions: dict) -> str:
    entries = manifest["entries"]
    ct = Counter(e["change_type"] for e in entries)
    v1_kinds = Counter(e["legacy_kind"] for e in entries)
    v2_kinds = Counter(it["kind"] for it in catalog["items"])
    inherited = [it for it in catalog["items"] if it.get("pricing_status") == PRICING_STATUS_INHERITED]
    successor_parent = {
        s["id"]: e["legacy_id"]
        for e in entries if e["change_type"] == "split"
        for s in e["successors"]
    }

    lines = [
        "# Catalog kind migration audit — 2.1 -> 3.0 (observation-kind-v2)",
        "",
        "Generated by `scripts/migrate_catalog_kind_v2.py` from",
        "`tools/catalog_migrations/kind_v2_decisions.json`. Do not edit by hand.",
        "",
        "## Policy",
        "",
        decisions.get("policy", ""),
        "",
        "## Totals",
        "",
        f"- Legacy items: {len(entries)} ({v1_kinds['defect']} defect / {v1_kinds['upgrade']} upgrade)",
        f"- v2 items: {len(catalog['items'])} "
        f"({v2_kinds['defect']} defect / {v2_kinds['degradation']} degradation / {v2_kinds['modernization']} modernization)",
        f"- Dispositions: {ct['unchanged']} unchanged, {ct['reclassified']} reclassified, "
        f"{ct['narrowed']} narrowed, {ct['split']} split",
        f"- Split successors with inherited v1 parent economics: {len(inherited)}",
        f"- Merges: 0 (no legacy concepts were combined)",
        "",
        "## Dispositions",
        "",
        "| legacy id | v1 kind | change | successors | re-resolve |",
        "|---|---|---|---|---|",
    ]
    for e in entries:
        succ = "; ".join(f"{s['id']} ({s['kind']})" for s in e["successors"])
        lines.append(
            f"| {e['legacy_id']} | {e['legacy_kind']} | {e['change_type']} | {succ} | "
            f"{'yes' if e['requires_re_resolution'] else 'no'} |"
        )

    lines += ["", "## Splits", ""]
    for e in entries:
        if e["change_type"] != "split":
            continue
        lines.append(f"### {e['legacy_id']} ({e['legacy_kind']}, deprecated)")
        lines.append("")
        lines.append(e["atomicity_rationale"])
        lines.append("")
        for s in e["successors"]:
            lines.append(f"- `{s['id']}` — {s['kind']}")
        lines.append("")

    lines += [
        "## Inherited-economics successors (Task 4A bridge)",
        "",
        "Every split successor inherits its parent's economic fields verbatim",
        "(`pricing_status: inherited_from_split_parent`). This is temporary",
        "compatibility so the v2 catalog can publish — not approval of v1 pricing;",
        "the dedicated pricing project authors real successor prices.",
        "",
    ]
    for it in inherited:
        carried = [f for f in ECONOMIC_FIELDS if f in it]
        absent = [f for f in ECONOMIC_FIELDS if f not in it]
        parts = [f"inherits {', '.join(carried) if carried else 'nothing'}"]
        if absent:
            parts.append(f"absent on parent: {', '.join(absent)}")
        lines.append(f"- `{it['id']}` <- `{successor_parent[it['id']]}`: {'; '.join(parts)}")

    flagged = [e for e in entries if e.get("task3_flags")]
    if flagged:
        lines += ["", "## Flagged for Task 3 (recorded, deliberately not acted on here)", ""]
        for e in flagged:
            lines.append(f"### {e['legacy_id']}")
            lines.append("")
            for note in e["task3_flags"]:
                lines.append(f"- {note}")
            lines.append("")

    lines += [
        "",
        "## Downstream effects",
        "",
        "The per-consumer concerns previously tracked here were resolved by Task 3",
        "(three-kind consumers) and Task 4A (multipliers, inherited successor",
        "economics, publication validation). Per-entry `expected_effects` remain in",
        "the manifest as the historical record.",
        "",
        "- Historical artifacts are legacy_v1 and require re-resolution (Task 4B; see "
        "manifest `requires_re_resolution`).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    decisions = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    catalog, manifest = generate(v1, decisions)
    audit = render_audit(catalog, manifest, decisions)

    V2_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    AUDIT_PATH.write_text(audit + "\n", encoding="utf-8")

    kinds = Counter(it["kind"] for it in catalog["items"])
    print(f"wrote {V2_PATH.name}: {len(catalog['items'])} items {dict(kinds)}")
    print(f"wrote {MANIFEST_PATH.name}: {len(manifest['entries'])} entries")
    print(f"wrote {AUDIT_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
