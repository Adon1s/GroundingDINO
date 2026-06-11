import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.catalog_validation import validate_issue_catalog  # noqa: E402


ROOM_TERMS = (
    "living room",
    "bedroom",
    "living",
    "bathroom",
    "kitchen",
    "garage",
    "basement",
)

CONCEPT_STRIP_TERMS = (
    "living_room",
    "living_areas",
    "living room",
    "bedroom",
    "living",
    "bathroom",
    "kitchen",
    "garage",
    "basement",
    "interior",
    "room",
    "area",
)

ROOM_LIKE_SCENE_GROUPS = {
    "kitchen",
    "bathroom",
    "bedroom",
    "living",
    "living_areas",
    "utility",
    "other",
}

# Legacy flat routing fields. Routing now lives in per-item `package_affinity`
# blocks (room-keyed); these fields should no longer appear with routing values
# and are kept only so the audit surfaces any regression/re-introduction.
PACKAGE_FIELDS = ("room", "package_type", "package_role", "package_category")

TAIL_LINE_THRESHOLD = 4694

TAIL_RECOMMENDATIONS = {
    "worn_or_dated_bedroom_carpet": "collapse",
    "dated_bedroom_flooring_style": "collapse",
    "dated_bedroom_light_or_fan": "collapse",
    "bedroom_wood_paneling_or_wallpaper": "collapse",
    "bedroom_popcorn_ceiling": "collapse",
    "damaged_bedroom_flooring": "collapse",
    "bedroom_drywall_damage_or_holes": "collapse",
    "bedroom_water_stain": "collapse",
    "bedroom_paint_refresh_recommended": "collapse",
    "damaged_or_missing_closet_doors": "review",
    "worn_or_dated_living_carpet": "collapse",
    "dated_living_flooring_style": "collapse",
    "dated_living_light_fixture": "collapse",
    "living_wood_paneling_or_wallpaper": "collapse",
    "living_popcorn_ceiling": "collapse",
    "damaged_living_flooring": "collapse",
    "living_drywall_damage_or_holes": "collapse",
    "living_water_stain": "collapse",
    "living_paint_refresh_recommended": "collapse",
    "dated_fireplace_surround": "review",
}

GENERIC_REPLACEMENT_IDS = (
    "worn_or_dated_carpet",
    "dated_flooring_style",
    "scratched_or_damaged_flooring",
    "damaged_drywall_or_cracks",
    "water_stain_ceiling",
    "water_stain_wall_or_ceiling",
    "popcorn_or_acoustic_ceiling_texture",
    "dated_wood_paneling_or_wallpaper",
    "dated_lighting_fixtures",
    "interior_paint_refresh_recommended",
    "damaged_or_missing_closet_doors",
    "dated_fireplace_surround",
    "dated_or_scuffed_trim_baseboards",
)


@dataclass(frozen=True)
class LongEmbedRow:
    item_id: str
    name: str
    word_count: int
    severity: str
    preview: str


@dataclass(frozen=True)
class RoomSpecificRow:
    item_id: str
    name: str
    kind: str
    scene_groups: str
    word_count: int
    matched_terms: str


@dataclass(frozen=True)
class MetadataRow:
    item_id: str
    name: str
    scene_groups: str
    package_affinity: str
    classification: str


@dataclass(frozen=True)
class TailRow:
    item_id: str
    present: str
    line: str
    after_threshold: str
    recommendation: str
    name: str


def load_catalog(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def word_count(text):
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text or ""))


def preview(text, limit=160):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def item_id(item):
    return str(item.get("id") or item.get("defect_id") or item.get("upgrade_id") or "").strip()


def scene_groups(item):
    raw = item.get("scene_groups")
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def missing_scene_groups(item):
    return not scene_groups(item)


def summary(items):
    kind_counts = Counter(str(item.get("kind") or "").strip() or "(missing)" for item in items)
    scene_counts = Counter()
    for item in items:
        for group in scene_groups(item):
            scene_counts[group] += 1
    missing = {
        "id": sum(1 for item in items if not item_id(item)),
        "embed_text": sum(1 for item in items if not str(item.get("embed_text") or "").strip()),
        "kind": sum(1 for item in items if not str(item.get("kind") or "").strip()),
        "scene_groups": sum(1 for item in items if missing_scene_groups(item)),
    }
    return {
        "total": len(items),
        "kind_counts": kind_counts,
        "scene_group_counts": scene_counts,
        "missing": missing,
    }


def embed_severity(count):
    if count > 75:
        return "severe"
    if count > 50:
        return "likely too long"
    if count > 35:
        return "review"
    return ""


def long_embed_rows(items):
    rows = []
    for item in items:
        text = str(item.get("embed_text") or "")
        count = word_count(text)
        severity = embed_severity(count)
        if severity:
            rows.append(LongEmbedRow(
                item_id=item_id(item),
                name=str(item.get("name") or ""),
                word_count=count,
                severity=severity,
                preview=preview(text),
            ))
    return sorted(rows, key=lambda row: (-row.word_count, row.item_id))


def room_term_hits(*texts):
    combined = " ".join(str(t or "").lower().replace("_", " ") for t in texts)
    hits = []
    for term in ROOM_TERMS:
        term_pattern = re.escape(term).replace(r"\ ", r"\s+")
        if re.search(r"(^|\b)" + term_pattern + r"(\b|$)", combined):
            hits.append(term)
    return hits


def room_specific_rows(items):
    rows = []
    for item in items:
        text = str(item.get("embed_text") or "")
        hits = room_term_hits(item_id(item), item.get("name"), text)
        if not hits:
            continue
        rows.append(RoomSpecificRow(
            item_id=item_id(item),
            name=str(item.get("name") or ""),
            kind=str(item.get("kind") or ""),
            scene_groups=", ".join(scene_groups(item)),
            word_count=word_count(text),
            matched_terms=", ".join(hits),
        ))
    return sorted(rows, key=lambda row: (row.matched_terms, row.item_id))


def normalize_concept(value):
    text = str(value or "").lower()
    text = text.replace("&", " and ")
    for term in sorted(CONCEPT_STRIP_TERMS, key=len, reverse=True):
        text = re.sub(r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"[\s_]+") + r"(?![a-z0-9])", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    tokens = [token for token in text.strip("_").split("_") if token]
    return "_".join(tokens)


def duplicate_clusters(items):
    clusters = defaultdict(dict)
    for item in items:
        iid = item_id(item)
        if not iid:
            continue
        for source in (iid, item.get("name")):
            key = normalize_concept(source)
            if key and key != normalize_concept(iid):
                clusters[key][iid] = item
            elif key:
                clusters[key][iid] = item
    out = []
    for key, by_id in clusters.items():
        values = list(by_id.values())
        if len(values) >= 2:
            out.append((key, sorted(values, key=item_id)))
    return sorted(out, key=lambda cluster: (-len(cluster[1]), cluster[0]))


def has_any_package_field(item):
    return "package_affinity" in item or any(field in item for field in PACKAGE_FIELDS)


def _affinity_cell(item):
    """Join an item's package_affinity block into one `room:type/role` cell."""
    block = item.get("package_affinity") or {}
    if not isinstance(block, dict):
        return str(block)
    return "; ".join(
        f"{room}:{(entry or {}).get('package_type', '?')}/{(entry or {}).get('package_role', '?')}"
        for room, entry in sorted(block.items())
    )


def metadata_rows(items):
    rows = []
    for item in items:
        groups = scene_groups(item)
        room_like_count = sum(1 for group in groups if group in ROOM_LIKE_SCENE_GROUPS)
        if room_like_count <= 1 or not has_any_package_field(item):
            continue
        block = item.get("package_affinity") or {}
        if any(field in item for field in ("package_type", "package_category", "room")):
            classification = "legacy flat routing fields (regression)"
        elif len(block) == 1:
            classification = "one-room package tie"
        elif block:
            classification = "multi-room affinity"
        else:
            classification = "role-only metadata"
        rows.append(MetadataRow(
            item_id=item_id(item),
            name=str(item.get("name") or ""),
            scene_groups=", ".join(groups),
            package_affinity=_affinity_cell(item),
            classification=classification,
        ))
    return sorted(rows, key=lambda row: (row.classification, row.item_id))


def line_lookup(lines):
    lookup = {}
    for lineno, line in enumerate(lines, start=1):
        match = re.search(r'"id"\s*:\s*"([^"]+)"', line)
        if match:
            lookup[match.group(1)] = lineno
            continue
        for alt in ("defect_id", "upgrade_id"):
            match = re.search(r'"' + alt + r'"\s*:\s*"([^"]+)"', line)
            if match:
                current_id = match.group(1)
                current_line = lineno
                lookup[current_id] = lineno
                break
    return lookup


def tail_rows(items, lines):
    by_id = {item_id(item): item for item in items if item_id(item)}
    lines_by_id = line_lookup(lines)
    rows = []
    for iid, recommendation in TAIL_RECOMMENDATIONS.items():
        item = by_id.get(iid)
        line = lines_by_id.get(iid)
        rows.append(TailRow(
            item_id=iid,
            present="yes" if item else "no",
            line=str(line) if line else "",
            after_threshold="yes" if line and line > TAIL_LINE_THRESHOLD else ("no" if line else ""),
            recommendation=recommendation,
            name=str((item or {}).get("name") or ""),
        ))
    return rows


def resolver_assumptions():
    return [
        "`embed_text` fully replaces the fallback embedding text in `tools/catalog_embeddings.py`; it is not shown in the Pass 2d prompt.",
        "`support_any` is passed into candidates, shown to Pass 2d, and used by lexical shortcuts; it is not a hard retrieval gate.",
        "`require_any` and `deny_any` are hard lexical guardrails during embedding candidate retrieval.",
        "`scene_groups` hard-filters candidate retrieval by observed scene group; missing `scene_groups` defaults to broad reach.",
        "`drop_if_generic` and `defaultHidden` make candidates generic for prioritization and shortcuts; `drop_if_generic` also suppresses final output in Pass 2e.",
        "`package_affinity` (room-keyed routing blocks; superseded the flat `package_type`/`package_role`/`package_category`/`room` fields) is not a Pass 2d matching field; it belongs to later package construction / Pass 2f behavior.",
    ]


def table(headers, rows):
    widths = [len(header) for header in headers]
    string_rows = []
    for row in rows:
        string_row = [str(cell) for cell in row]
        string_rows.append(string_row)
        widths = [max(width, len(cell)) for width, cell in zip(widths, string_row)]
    lines = []
    lines.append("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    lines.append("  ".join("-" * width for width in widths))
    for row in string_rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
    return "\n".join(lines)


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = []
        for cell in row:
            cells.append(str(cell).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_stdout(data):
    lines = []
    validation = data["validation"]
    lines.append("Catalog Validation")
    lines.append("==================")
    lines.append(f"errors: {len(validation.errors)}")
    for entry in validation.errors:
        lines.append(f"  ERROR {entry}")
    lines.append(f"warnings: {len(validation.warnings)}")
    for entry in validation.warnings:
        lines.append(f"  WARN  {entry}")
    lines.append("")

    summary_data = data["summary"]
    lines.append("Catalog Summary")
    lines.append("===============")
    lines.append(f"total item count: {summary_data['total']}")
    lines.append("")
    lines.append("kind counts:")
    lines.append(table(("kind", "count"), sorted(summary_data["kind_counts"].items())))
    lines.append("")
    lines.append("scene_group counts:")
    lines.append(table(("scene_group", "count"), sorted(summary_data["scene_group_counts"].items())))
    lines.append("")
    lines.append("missing fields:")
    lines.append(table(("field", "count"), sorted(summary_data["missing"].items())))
    lines.append("")

    lines.append("Long embed_text Report")
    lines.append("======================")
    lines.append(table(
        ("id", "name", "words", "severity", "embed_text preview"),
        [(r.item_id, r.name, r.word_count, r.severity, r.preview) for r in data["long_embeds"]],
    ) if data["long_embeds"] else "(none)")
    lines.append("")

    lines.append("Room-Specific Item Report")
    lines.append("=========================")
    lines.append(table(
        ("id", "name", "kind", "scene_groups", "words", "matched_terms"),
        [(r.item_id, r.name, r.kind, r.scene_groups, r.word_count, r.matched_terms) for r in data["room_specific"]],
    ) if data["room_specific"] else "(none)")
    lines.append("")

    lines.append("Likely Duplicate Canonical Concepts")
    lines.append("===================================")
    if data["duplicates"]:
        for key, cluster_items in data["duplicates"]:
            lines.append(f"- {key}")
            for item in cluster_items:
                lines.append(f"  - {item_id(item)} | {item.get('name', '')}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("Generic Items With One-Room Package Metadata")
    lines.append("============================================")
    lines.append(table(
        ("id", "name", "scene_groups", "package_affinity", "classification"),
        [(r.item_id, r.name, r.scene_groups, r.package_affinity, r.classification) for r in data["metadata"]],
    ) if data["metadata"] else "(none)")
    lines.append("")

    lines.append("Latest Tail Section Review")
    lines.append("==========================")
    lines.append(table(
        ("id", "present", "line", "after_4694", "recommendation", "name"),
        [(r.item_id, r.present, r.line, r.after_threshold, r.recommendation, r.name) for r in data["tail"]],
    ))
    lines.append("")

    lines.append("Resolver Assumptions")
    lines.append("====================")
    for assumption in resolver_assumptions():
        lines.append(f"- {assumption}")
    return "\n".join(lines)


def render_markdown(data, catalog_path):
    summary_data = data["summary"]
    absent_tail = [row.item_id for row in data["tail"] if row.present == "no"]
    lines = []
    lines.append("# Catalog Audit Report")
    lines.append("")
    lines.append(f"Catalog audited: `{catalog_path.as_posix()}`")
    lines.append("")
    lines.append("## 1. Summary of Current Catalog Issues")
    lines.append("")
    lines.append(
        f"The catalog currently has {summary_data['total']} items: "
        f"{summary_data['kind_counts'].get('defect', 0)} defects and "
        f"{summary_data['kind_counts'].get('upgrade', 0)} upgrades. "
        f"Missing-field counts are: id={summary_data['missing']['id']}, "
        f"embed_text={summary_data['missing']['embed_text']}, "
        f"kind={summary_data['missing']['kind']}, "
        f"scene_groups={summary_data['missing']['scene_groups']}."
    )
    lines.append("")
    lines.append("### Kind Counts")
    lines.append("")
    lines.append(markdown_table(("kind", "count"), sorted(summary_data["kind_counts"].items())))
    lines.append("")
    lines.append("### Scene Group Counts")
    lines.append("")
    lines.append(markdown_table(("scene_group", "count"), sorted(summary_data["scene_group_counts"].items())))
    lines.append("")
    lines.append("### Missing Fields")
    lines.append("")
    lines.append(markdown_table(("field", "count"), sorted(summary_data["missing"].items())))
    lines.append("")
    lines.append("## Resolver Assumptions")
    lines.append("")
    for assumption in resolver_assumptions():
        lines.append(f"- {assumption}")
    lines.append("")

    lines.append("## 2. Top 25 Longest `embed_text` Items")
    lines.append("")
    top_long = data["long_embeds"][:25]
    lines.append(markdown_table(
        ("id", "name", "words", "severity", "embed_text preview"),
        [(r.item_id, r.name, r.word_count, r.severity, r.preview) for r in top_long],
    ) if top_long else "_No `embed_text` values exceeded 35 words._")
    lines.append("")

    lines.append("## 3. Likely Duplicate Room-Specific Clusters")
    lines.append("")
    if data["duplicates"]:
        for key, cluster_items in data["duplicates"]:
            lines.append(f"### `{key}`")
            lines.append("")
            lines.append(markdown_table(
                ("id", "name", "kind", "scene_groups"),
                [(item_id(item), item.get("name", ""), item.get("kind", ""), ", ".join(scene_groups(item))) for item in cluster_items],
            ))
            lines.append("")
    else:
        lines.append("_No duplicate clusters found after stripping common room/context words._")
        lines.append("")

    lines.append("## 4. Generic Items With One-Room Package Metadata")
    lines.append("")
    lines.append(markdown_table(
        ("id", "name", "scene_groups", "package_affinity", "classification"),
        [(r.item_id, r.name, r.scene_groups, r.package_affinity, r.classification) for r in data["metadata"]],
    ) if data["metadata"] else "_No generic multi-room metadata findings._")
    lines.append("")

    lines.append("## 5. Latest Tail Items and Recommended Action")
    lines.append("")
    if absent_tail:
        lines.append(
            "The requested bedroom/living-room tail IDs are not present in this checkout. "
            "They are still listed below as a safety checklist for the next catalog rewrite."
        )
        lines.append("")
    lines.append(markdown_table(
        ("id", "present", "line", "after_4694", "recommended_action", "name"),
        [(r.item_id, r.present, r.line, r.after_threshold, r.recommendation, r.name) for r in data["tail"]],
    ))
    lines.append("")

    lines.append("## 6. Proposed Generic Replacement IDs for Next Session")
    lines.append("")
    for replacement_id in GENERIC_REPLACEMENT_IDS:
        lines.append(f"- `{replacement_id}`")
    lines.append("")

    lines.append("## Room-Specific Manual Review Appendix")
    lines.append("")
    lines.append(markdown_table(
        ("id", "name", "kind", "scene_groups", "embed_words", "matched_terms"),
        [(r.item_id, r.name, r.kind, r.scene_groups, r.word_count, r.matched_terms) for r in data["room_specific"]],
    ) if data["room_specific"] else "_No room-specific terms found._")
    lines.append("")
    return "\n".join(lines)


def build_audit(catalog_path):
    catalog = load_catalog(catalog_path)
    lines = read_lines(catalog_path)
    items = [item for item in catalog.get("items", []) if isinstance(item, dict)]
    return {
        "validation": validate_issue_catalog(catalog),
        "summary": summary(items),
        "long_embeds": long_embed_rows(items),
        "room_specific": room_specific_rows(items),
        "duplicates": duplicate_clusters(items),
        "metadata": metadata_rows(items),
        "tail": tail_rows(items, lines),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Audit issue_catalog.json for catalog refactor planning.")
    parser.add_argument("--catalog", default="tools/issue_catalog.json", type=Path)
    parser.add_argument("--report", default="docs/catalog_audit_report.md", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    data = build_audit(args.catalog)
    print(render_stdout(data))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(data, args.catalog), encoding="utf-8")


if __name__ == "__main__":
    main()
