"""
One-off migration: move package routing into catalog `package_affinity` blocks.

Folds the two legacy routing sources into per-item `package_affinity` objects
in tools/issue_catalog.json, keyed by package room:
  1. The in-code PACKAGE_AFFINITY dict (post-_register_generic_support).
  2. The flat catalog fields (package_type/package_role/package_category/room)
     on the routed items, which are stripped after conversion.

Also dumps tests/fixtures/package_affinity_snapshot.json — the union of both
legacy sources in the runtime table shape — which the keystone parity test
asserts against build_package_affinity(catalog) forever after.

Run once from the repo root while the legacy constants still exist:
    .venv\\Scripts\\python.exe scripts/migrate_package_affinity.py

After the legacy constants are deleted from tools/rehab_packages.py this
script can no longer run; it is kept for provenance, and the snapshot fixture
is its durable output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.rehab_packages import (  # noqa: E402
    PACKAGE_AFFINITY,
    PACKAGE_ROLE_DRIVER,
    PACKAGE_ROLE_SUPPORT,
    _PACKAGE_TYPE_TO_CATEGORY,
    _PACKAGE_TYPE_TO_ROOM,
    build_package_affinity,
)

CATALOG_PATH = ROOT / "tools" / "issue_catalog.json"
SNAPSHOT_PATH = ROOT / "tests" / "fixtures" / "package_affinity_snapshot.json"

FLAT_FIELDS = ("package_type", "package_role", "package_category", "room")

# Structural/feature generics that intentionally stay unrouted (see the
# catalog-driven affinity design note in tools/rehab_packages.py).
UNROUTED_GENERICS = frozenset({
    "unfinished_interior_wall_osb_exposed",
    "stained_glass_or_vintage_light_fixture",
    "layout_modernization_opportunity",
})


def main() -> None:
    raw = CATALOG_PATH.read_text(encoding="utf-8")
    catalog = json.loads(raw)
    items = catalog["items"]
    items_by_id = {item["id"]: item for item in items}

    # ── Legacy source 1: in-code PACKAGE_AFFINITY ────────────────────────────
    old_table: Dict[Tuple[str, str], Dict[str, str]] = {
        key: dict(value) for key, value in PACKAGE_AFFINITY.items()
    }
    blocks: Dict[str, Dict[str, Dict[str, str]]] = {}
    for (room, issue_id), meta in PACKAGE_AFFINITY.items():
        assert issue_id in items_by_id, f"PACKAGE_AFFINITY references unknown item {issue_id}"
        blocks.setdefault(issue_id, {})[room] = {
            "package_type": meta["package_type"],
            "package_role": meta["package_role"],
        }

    # ── Legacy source 2: flat catalog fields on routed items ─────────────────
    flat_items = [item for item in items if item.get("package_type")]
    for item in flat_items:
        item_id = item["id"]
        package_type = item["package_type"]
        package_role = item["package_role"]
        room = item["room"]
        assert package_role in (PACKAGE_ROLE_DRIVER, PACKAGE_ROLE_SUPPORT), (
            f"{item_id}: routed item has unexpected package_role {package_role!r}"
        )
        assert _PACKAGE_TYPE_TO_ROOM[package_type] == room, (
            f"{item_id}: flat room {room!r} contradicts package_type {package_type!r}"
        )
        assert item.get("package_category") == _PACKAGE_TYPE_TO_CATEGORY[package_type], (
            f"{item_id}: flat package_category {item.get('package_category')!r} is not "
            f"derivable from {package_type!r}"
        )
        assert (room, item_id) not in old_table, (
            f"{item_id}: flat fields collide with a PACKAGE_AFFINITY entry for {room!r}"
        )
        old_table[(room, item_id)] = {
            "package_type": package_type,
            "package_role": package_role,
            "package_category": item["package_category"],
            "room": room,
        }
        blocks.setdefault(item_id, {})[room] = {
            "package_type": package_type,
            "package_role": package_role,
        }

    for item_id in UNROUTED_GENERICS:
        assert item_id in items_by_id, f"unrouted generic {item_id} missing from catalog"
        assert item_id not in blocks, f"{item_id} must stay unrouted but gained a block"

    # ── Rewrite items: insert blocks, strip flat fields from routed items ────
    routed_flat_ids = {item["id"] for item in flat_items}
    new_items = []
    for item in items:
        block = blocks.get(item["id"])
        if not block:
            new_items.append(item)
            continue
        sorted_block = {room: block[room] for room in sorted(block)}
        new_item: Dict[str, Any] = {}
        inserted = False
        for key, value in item.items():
            if item["id"] in routed_flat_ids and key in FLAT_FIELDS:
                # package_affinity takes the position of the first flat field
                # to keep the JSON diff local.
                if not inserted:
                    new_item["package_affinity"] = sorted_block
                    inserted = True
                continue
            new_item[key] = value
        if not inserted:
            new_item["package_affinity"] = sorted_block
        new_items.append(new_item)
    catalog["items"] = new_items

    # ── Round-trip parity before writing anything ────────────────────────────
    new_table = build_package_affinity(catalog)
    assert new_table == old_table, (
        "round-trip mismatch:\n"
        f"  only_old={sorted(set(old_table) - set(new_table))}\n"
        f"  only_new={sorted(set(new_table) - set(old_table))}\n"
        f"  changed={[k for k in old_table if k in new_table and old_table[k] != new_table[k]]}"
    )
    for item in flat_items:
        derived = new_table[(item["room"], item["id"])]
        for field in FLAT_FIELDS:
            assert derived[field] == item[field], (
                f"{item['id']}: derived {field}={derived[field]!r} != flat {item[field]!r}"
            )

    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        f"{room}|{issue_id}": old_table[(room, issue_id)]
        for room, issue_id in sorted(old_table)
    }
    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"items with package_affinity blocks: {len(blocks)}")
    print(f"flat-field items converted+stripped: {len(flat_items)}")
    print(f"affinity entries total: {len(old_table)}")
    print(f"wrote {CATALOG_PATH.relative_to(ROOT)}")
    print(f"wrote {SNAPSHOT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
