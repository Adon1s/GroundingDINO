"""Generate the observation-kind-v2 catalog, migration manifest, and audit report.

Deterministic, byte-stable transformation:

    tools/issue_catalog.json  (v1, untouched)
  + tools/catalog_migrations/kind_v2_decisions.json  (the reviewable source of truth)
  ->  tools/issue_catalog_kind_v2.json               (catalog version 3.0, non-publishable)
      tools/catalog_migrations/2.1_to_3.0.json       (audit-only manifest, one entry per legacy id)
      tools/catalog_migrations/2.1_to_3.0_audit.md   (generated audit report)

Inheritance rules are code, not authoring discipline:
- unchanged/reclassified/narrowed items copy the legacy item verbatim, apply the
  v2 kind + atomic_claim + optional wording overrides, and KEEP every economic
  field byte-identical.
- split successors are authored fresh (name/description/embed_text/support_any/
  severity/kind/atomic_claim required), inherit only non-economic structural
  fields from the parent, are stamped pricing_status=deferred_post_task3, and
  can never carry an economic field (hard error, not convention).

Run:  .venv/Scripts/python.exe scripts/migrate_catalog_kind_v2.py
Running twice produces byte-identical outputs (pinned by the parity test in
tests/test_catalog_kind_v2.py).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.catalog_validation import ECONOMIC_FIELDS  # noqa: E402
from tools.observation_kinds import OBSERVATION_KINDS, ONTOLOGY_VERSION  # noqa: E402

V1_PATH = ROOT / "tools" / "issue_catalog.json"
DECISIONS_PATH = ROOT / "tools" / "catalog_migrations" / "kind_v2_decisions.json"
V2_PATH = ROOT / "tools" / "issue_catalog_kind_v2.json"
MANIFEST_PATH = ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0.json"
AUDIT_PATH = ROOT / "tools" / "catalog_migrations" / "2.1_to_3.0_audit.md"

TARGET_VERSION = "3.0"
PUBLICATION_STATUS = "blocked_pending_pricing"

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
    item["pricing_status"] = "deferred_post_task3"
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
            "no analysis runs against v2 until the Task 3 cutover, and historical "
            "artifacts are re-resolved then, never aliased."
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
    deferred = [it["id"] for it in catalog["items"] if it.get("pricing_status") == "deferred_post_task3"]

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
        f"- Split successors with deferred pricing: {len(deferred)}",
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
        "## Deferred-pricing successors",
        "",
        "Every split successor omits cost, estimate, work-item, and package metadata",
        "(`pricing_status: deferred_post_task3`); authoring happens after Task 3.",
        "",
    ]
    lines += [f"- `{i}`" for i in deferred]

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
        "## Unresolved downstream effects (Task 3 scope)",
        "",
        "Aggregated from per-entry `expected_effects`; none of these are live while the",
        "v2 catalog is offline (`publication_status: blocked_pending_pricing`).",
        "",
        "- `costing.KIND_MULT` has no degradation/modernization entries; silent `.get(kind, 1.0)` default.",
        "- `estimate_scope`: kind token 'modernization' collides with `_VALUE_ADD_TERMS` text matching.",
        "- `rehab_packages`: driver/support predicates key on defect/upgrade; "
        "`PACKAGE_CATEGORY_MODERNIZATION` shares the 'modernization' token.",
        "- `property_summary_pass`: second two-kind VALID_KINDS with unknown->defect coercion.",
        "- `catalog_cost_model`: upgrade->ROOM_ALLOWANCE routing has no three-kind mapping.",
        "- Pass 2e: `invalid_kind` hard-drop for anything outside {defect, upgrade}.",
        "- Split parents with estimate_scope overrides lose them on successors (re-author post-Task 3).",
        "- Historical artifacts are legacy_v1 and require re-resolution at cutover (see manifest "
        "`requires_re_resolution`).",
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
