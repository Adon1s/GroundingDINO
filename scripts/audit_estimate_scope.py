"""
Estimate-scope classification audit for tools/issue_catalog.json.

Observe-only: this script *calls* the real classifier
(`classify_estimate_scope_with_reason`) and the real package mapping
(`classify_package_scope`) — it never re-implements, forks, or extends scope
logic, and it never invents new reasons. It enumerates every catalog item's
baseline scope + reason, flags substring traps (and the field that carried
them), groups package drivers/supports for a package-surface review, and emits
summary counts. With ``--baseline`` it diffs a saved snapshot to produce the
before/after distribution change.

Usage (run from repo root):
    .venv\\Scripts\\python.exe scripts/audit_estimate_scope.py \\
        --json artifacts/estimate_scope_before.json \\
        --report docs/catalog_estimate_scope_audit.md
    # ...apply overrides, then:
    .venv\\Scripts\\python.exe scripts/audit_estimate_scope.py \\
        --baseline artifacts/estimate_scope_before.json \\
        --json artifacts/estimate_scope_after.json \\
        --report docs/catalog_estimate_scope_audit.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `tools` importable regardless of the caller's working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.estimate_scope import (  # noqa: E402
    VALID_ESTIMATE_SCOPES,
    _catalog_text,
    classify_estimate_scope_with_reason,
    classify_package_scope,
)
from tools.rehab_packages import (  # noqa: E402
    PACKAGE_CATEGORY_MODERNIZATION,
    PACKAGE_ROLE_DRIVER,
    PACKAGE_STRENGTH_MODERATE,
    PACKAGE_STRENGTH_STRONG,
    _PACKAGE_TYPE_TO_CATEGORY,
    build_package_affinity,
)

# Reasons we trust enough to *deprioritize* in review. Hard != presumed correct;
# defect_severity_threshold / required_category can over-fire, so these are
# reviewed last, not skipped.
HARD_REASONS = frozenset({
    "catalog_override",
    "required_category",
    "defect_severity_threshold",
    "required_condition_signal",
})

# Substring traps from the audit handoff (addition #6). A benign retrieval word
# can fire a scope term it was never meant to drive. Ordered most-specific first
# so the reported term is the meaningful one.
TRAP_TERMS = (
    "refinish", "unfinished", "finishes", "finish",
    "fixture", "lighting", "paint",
    "older", "old ", "style",
    "additional", "addition", "open wall",
)

# Fields _catalog_text concatenates from the catalog item, in order. Used to
# attribute which field carried a trap term.
CATALOG_TEXT_FIELDS = (
    "id", "name", "category", "description", "scope", "trade_bucket",
    "work_item_code", "embed_text", "tier",
)

# Tier-1 high-cost / high-trust trades (addition #2). Substring-matched against
# trade_bucket and category.
TIER1_TRADE_TOKENS = (
    "roof", "gutter", "foundation", "structure", "structural", "masonry",
    "electrical", "plumbing", "hvac", "system", "moisture", "mold",
    "remediation", "safety",
)

HIGH_COST_CAP_HIGH = 10000


@dataclass(frozen=True)
class LineRow:
    item_id: str
    kind: str
    tier: str
    category: str
    trade_bucket: str
    severity: str
    scope: str
    reason: str
    hardsoft: str
    priority: int
    has_override: bool
    traps: Tuple[Tuple[str, str], ...]

    def traps_str(self) -> str:
        return "; ".join(f"{term}:{src}" for term, src in self.traps)


@dataclass
class PackageGroup:
    package_type: str
    category: str
    category_source: str
    driver_ids: List[str] = field(default_factory=list)
    support_ids: List[str] = field(default_factory=list)
    driver_scopes: List[str] = field(default_factory=list)
    strong_scope: str = ""
    moderate_scope: str = ""
    flags: List[Tuple[str, str]] = field(default_factory=list)  # (class, reason)


def load_catalog(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _item_id(item: Dict[str, Any]) -> str:
    return str(item.get("id") or "").strip()


def _has_override(item: Dict[str, Any]) -> bool:
    raw = item.get("estimate_scope") or (item.get("estimate") or {}).get("estimate_scope")
    return str(raw or "").strip() in VALID_ESTIMATE_SCOPES


def _field_texts(item: Dict[str, Any]) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for name in CATALOG_TEXT_FIELDS:
        value = item.get(name)
        if value is not None:
            texts[name] = str(value).lower()
    support_any = item.get("support_any")
    if isinstance(support_any, list):
        texts["support_any"] = " ".join(str(v).lower() for v in support_any if v is not None)
    return texts


def _trap_hits(item: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Return (term, source_field) for each distinct trap term present."""
    field_texts = _field_texts(item)
    hits: List[Tuple[str, str]] = []
    seen: set = set()
    for term in TRAP_TERMS:
        if term in seen:
            continue
        for fname, ftext in field_texts.items():
            if term in ftext:
                hits.append((term.strip(), fname))
                seen.add(term)
                break
    return tuple(hits)


def _priority(
    item: Dict[str, Any],
    traps: Tuple[Tuple[str, str], ...],
    driver_ids: set,
) -> int:
    trade = str(item.get("trade_bucket") or "").lower()
    category = str(item.get("category") or "").lower()
    if any(tok in trade or tok in category for tok in TIER1_TRADE_TOKENS):
        return 1
    cap_high = ((item.get("cost") or {}).get("cap_high")) or 0
    is_driver = _item_id(item) in driver_ids
    try:
        cap_high = int(cap_high)
    except (TypeError, ValueError):
        cap_high = 0
    if cap_high >= HIGH_COST_CAP_HIGH or is_driver:
        return 2
    if traps:
        return 3
    return 4


def enumerate_line_items(
    items: List[Dict[str, Any]],
    driver_ids: set,
) -> List[LineRow]:
    rows: List[LineRow] = []
    for item in items:
        scope, reason = classify_estimate_scope_with_reason({}, item, None)
        traps = _trap_hits(item)
        rows.append(LineRow(
            item_id=_item_id(item),
            kind=str(item.get("kind") or ""),
            tier=str(item.get("tier") or ""),
            category=str(item.get("category") or ""),
            trade_bucket=str(item.get("trade_bucket") or ""),
            severity=str(item.get("severity") if item.get("severity") is not None else ""),
            scope=scope,
            reason=reason,
            hardsoft="hard" if reason in HARD_REASONS else "soft",
            priority=_priority(item, traps, driver_ids),
            has_override=_has_override(item),
            traps=traps,
        ))
    return rows


def _package_looks_like(package_type: str) -> Optional[str]:
    """Coarse intent implied by the package_type name, for mismatch detection."""
    pt = package_type.lower()
    if "repair" in pt:
        return "repair"
    if "modernization" in pt:
        return "modernization"
    if "turnover" in pt or "refresh" in pt:
        return "turnover"
    return None


def build_package_groups(
    affinity_table: Dict[Tuple[str, str], Dict[str, str]],
    scope_by_id: Dict[str, str],
) -> List[PackageGroup]:
    """Group affinity-table entries by package_type, one membership per
    (item, room) entry. Routing comes from build_package_affinity, which
    derives package_category from package_type — so category disagreement
    between members is impossible by construction and no longer flagged."""
    members: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: {"driver": [], "support": []}
    )
    for (room, issue_id), meta in sorted(affinity_table.items()):
        role = "driver" if meta["package_role"] == PACKAGE_ROLE_DRIVER else "support"
        members[meta["package_type"]][role].append(issue_id)

    groups: List[PackageGroup] = []
    for ptype in sorted(members):
        driver_ids = members[ptype]["driver"]
        support_ids = members[ptype]["support"]

        static_cat = _PACKAGE_TYPE_TO_CATEGORY.get(ptype)
        category = static_cat or PACKAGE_CATEGORY_MODERNIZATION
        category_source = "derived-from-package-type" if static_cat else "default"

        strong_scope, _ = classify_package_scope(
            ptype, [], "", package_category=category, package_strength=PACKAGE_STRENGTH_STRONG,
        )
        moderate_scope, _ = classify_package_scope(
            ptype, [], "", package_category=category, package_strength=PACKAGE_STRENGTH_MODERATE,
        )

        driver_scopes = [scope_by_id.get(d, "?") for d in driver_ids]

        group = PackageGroup(
            package_type=ptype,
            category=category,
            category_source=category_source,
            driver_ids=driver_ids,
            support_ids=support_ids,
            driver_scopes=driver_scopes,
            strong_scope=strong_scope,
            moderate_scope=moderate_scope,
        )

        # --- deterministic mismatches ---
        if static_cat is None:
            group.flags.append((
                "deterministic-mismatch",
                f"package_type '{ptype}' absent from _PACKAGE_TYPE_TO_CATEGORY",
            ))
        looks = _package_looks_like(ptype)
        if looks and looks != category:
            group.flags.append((
                "deterministic-mismatch",
                f"name implies '{looks}' but category resolves to '{category}'",
            ))

        # --- heuristic suspicion: drivers' line-item scopes vs package scope ---
        if driver_scopes:
            disagree = [s for s in driver_scopes if s not in (strong_scope, moderate_scope)]
            if len(disagree) > len(driver_scopes) / 2:
                group.flags.append((
                    "heuristic-suspicion",
                    f"majority driver line-item scope {Counter(driver_scopes).most_common()} "
                    f"disagrees with package {strong_scope}/{moderate_scope}",
                ))

        groups.append(group)
    return groups


def summarize(rows: List[LineRow]) -> Dict[str, Any]:
    return {
        "total": len(rows),
        "overrides": sum(1 for r in rows if r.has_override),
        "scope_counts": Counter(r.scope for r in rows),
        "reason_counts": Counter(r.reason for r in rows),
        "hardsoft_counts": Counter(r.hardsoft for r in rows),
        "fallback_required": sum(1 for r in rows if r.reason == "fallback_required"),
        "with_traps": sum(1 for r in rows if r.traps),
    }


def build_audit(catalog_path: Path) -> Dict[str, Any]:
    catalog = load_catalog(catalog_path)
    items = [it for it in catalog.get("items", []) if isinstance(it, dict)]
    affinity_table = build_package_affinity(catalog)
    driver_ids = {
        issue_id for (_room, issue_id), meta in affinity_table.items()
        if meta["package_role"] == PACKAGE_ROLE_DRIVER
    }
    rows = enumerate_line_items(items, driver_ids)
    scope_by_id = {r.item_id: r.scope for r in rows}
    packages = build_package_groups(affinity_table, scope_by_id)
    return {
        "rows": rows,
        "packages": packages,
        "summary": summarize(rows),
    }


# ─── snapshot (json) ─────────────────────────────────────────────────────────

def snapshot_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "items": [
            {
                "id": r.item_id,
                "scope": r.scope,
                "reason": r.reason,
                "has_override": r.has_override,
                "hardsoft": r.hardsoft,
                "priority": r.priority,
                "traps": [list(t) for t in r.traps],
            }
            for r in data["rows"]
        ],
        "summary": {
            "total": data["summary"]["total"],
            "overrides": data["summary"]["overrides"],
            "scope_counts": dict(data["summary"]["scope_counts"]),
            "reason_counts": dict(data["summary"]["reason_counts"]),
            "fallback_required": data["summary"]["fallback_required"],
        },
    }


def load_baseline(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        snap = json.load(f)
    return {row["id"]: row for row in snap.get("items", [])}


# ─── rendering ───────────────────────────────────────────────────────────────

def markdown_table(headers: Tuple[str, ...], rows: List[Tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _counter_diff(before: Counter, after: Counter) -> List[Tuple[str, str, str, str]]:
    keys = sorted(set(before) | set(after))
    out = []
    for k in keys:
        b, a = before.get(k, 0), after.get(k, 0)
        delta = a - b
        out.append((k, str(b), str(a), f"{delta:+d}" if delta else "0"))
    return out


def render_markdown(
    data: Dict[str, Any],
    catalog_path: Path,
    baseline: Optional[Dict[str, Any]],
) -> str:
    rows: List[LineRow] = data["rows"]
    summary = data["summary"]
    out: List[str] = []
    out.append("# Catalog Estimate-Scope Audit")
    out.append("")
    out.append(f"Catalog audited: `{catalog_path.as_posix()}`")
    out.append("")
    out.append(
        "_Generated by `scripts/audit_estimate_scope.py` (observe-only — calls the real "
        "`classify_estimate_scope_with_reason` / `classify_package_scope`)._"
    )
    out.append("")

    # --- summary counts ---
    out.append("## Summary counts")
    out.append("")
    base_overrides = sum(1 for v in baseline.values() if v.get("has_override")) if baseline else None
    base_fallback = (
        sum(1 for v in baseline.values() if v.get("reason") == "fallback_required")
        if baseline else None
    )
    summary_rows = [
        ("Total catalog items audited", summary["total"]),
        (
            "Items with estimate_scope override",
            f"{base_overrides} -> {summary['overrides']}" if baseline else summary["overrides"],
        ),
        (
            "Items still classified by fallback_required",
            f"{base_fallback} -> {summary['fallback_required']}"
            if baseline else summary["fallback_required"],
        ),
        ("Items still heuristic (no override)", summary["total"] - summary["overrides"]),
        ("Items with a substring-trap hit", summary["with_traps"]),
    ]
    out.append(markdown_table(("metric", "value"), summary_rows))
    out.append("")

    out.append("### Scope distribution")
    out.append("")
    if baseline:
        before_scopes = Counter(v.get("scope") for v in baseline.values())
        out.append(markdown_table(
            ("scope", "before", "after", "delta"),
            _counter_diff(before_scopes, summary["scope_counts"]),
        ))
    else:
        out.append(markdown_table(
            ("scope", "count"),
            sorted(summary["scope_counts"].items(), key=lambda kv: -kv[1]),
        ))
    out.append("")

    out.append("### Reason histogram")
    out.append("")
    if baseline:
        before_reasons = Counter(v.get("reason") for v in baseline.values())
        out.append(markdown_table(
            ("reason", "before", "after", "delta"),
            _counter_diff(before_reasons, summary["reason_counts"]),
        ))
    else:
        out.append(markdown_table(
            ("reason", "count"),
            sorted(summary["reason_counts"].items(), key=lambda kv: -kv[1]),
        ))
    out.append("")

    # --- changed rows (only with baseline) ---
    if baseline:
        changed = []
        for r in rows:
            prev = baseline.get(r.item_id)
            if prev and prev.get("scope") != r.scope:
                changed.append((r.item_id, prev.get("scope"), r.scope, prev.get("reason"), r.reason))
        out.append("## Scope changes since baseline")
        out.append("")
        if changed:
            out.append(markdown_table(
                ("id", "scope_before", "scope_after", "reason_before", "reason_after"),
                sorted(changed),
            ))
        else:
            out.append("_No scope changes._")
        out.append("")

    # --- line-item table (priority then id) ---
    out.append("## Line-item scope table")
    out.append("")
    out.append(
        "Sorted by review priority (1 = high-cost/high-trust trade, 2 = high unit cost or "
        "package driver, 3 = substring trap, 4 = other). `hard` reasons are deprioritized, "
        "**not** presumed correct."
    )
    out.append("")
    line_rows = sorted(rows, key=lambda r: (r.priority, r.item_id))
    out.append(markdown_table(
        ("id", "kind", "tier", "category", "trade_bucket", "sev", "scope", "reason",
         "hard/soft", "prio", "override", "traps"),
        [
            (r.item_id, r.kind, r.tier, r.category, r.trade_bucket, r.severity, r.scope,
             r.reason, r.hardsoft, r.priority, "yes" if r.has_override else "", r.traps_str())
            for r in line_rows
        ],
    ))
    out.append("")

    # --- substring-trap focus ---
    trap_rows = [r for r in rows if r.traps]
    out.append("## Substring-trap rows")
    out.append("")
    out.append(
        "Rows whose retrieval text contains a trap term. The trap only *matters* when it pulled "
        "the scope away from intent — read the row's reason and the source field."
    )
    out.append("")
    if trap_rows:
        out.append(markdown_table(
            ("id", "kind", "scope", "reason", "traps (term:source_field)"),
            [(r.item_id, r.kind, r.scope, r.reason, r.traps_str())
             for r in sorted(trap_rows, key=lambda r: (r.priority, r.item_id))],
        ))
    else:
        out.append("_No trap terms found._")
    out.append("")

    # --- package surface ---
    packages: List[PackageGroup] = data["packages"]
    out.append("## Package surface")
    out.append("")
    out.append(
        "`strong`/`moderate` show the deterministic scope a runtime strong/moderate strength "
        "would produce (they differ only for `modernization`). `package_strength` itself is "
        "runtime-computed and is **not** a catalog field."
    )
    out.append("")
    out.append(markdown_table(
        ("package_type", "category", "cat_source", "strong", "moderate",
         "drivers", "driver_scopes", "supports", "flags"),
        [
            (
                g.package_type, g.category, g.category_source, g.strong_scope, g.moderate_scope,
                len(g.driver_ids), ", ".join(sorted(set(g.driver_scopes))) or "-",
                len(g.support_ids),
                "; ".join(f"[{cls}] {msg}" for cls, msg in g.flags) or "-",
            )
            for g in packages
        ],
    ))
    out.append("")
    out.append(
        "_Package routing is catalog-driven (`package_affinity` blocks); "
        "`package_category` derives from `package_type`, so category "
        "consistency holds by construction._"
    )
    out.append("")

    return "\n".join(out)


def render_stdout(data: Dict[str, Any]) -> str:
    s = data["summary"]
    out = []
    out.append(f"items={s['total']}  overrides={s['overrides']}  "
               f"fallback_required={s['fallback_required']}  with_traps={s['with_traps']}")
    out.append("scopes: " + ", ".join(f"{k}={v}" for k, v in sorted(s["scope_counts"].items())))
    flagged = [g for g in data["packages"] if g.flags]
    out.append(f"packages={len(data['packages'])}  flagged={len(flagged)}")
    for g in flagged:
        out.append(f"  - {g.package_type}: " + "; ".join(f"[{c}] {m}" for c, m in g.flags))
    return "\n".join(out)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit estimate-scope classification of the issue catalog.")
    parser.add_argument("--catalog", default="tools/issue_catalog.json", type=Path)
    parser.add_argument("--report", default=None, type=Path, help="Write the markdown report here.")
    parser.add_argument("--json", dest="json_out", default=None, type=Path,
                        help="Write a machine-readable snapshot here (for --baseline diffs).")
    parser.add_argument("--baseline", default=None, type=Path,
                        help="A prior --json snapshot to diff against (before/after).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    data = build_audit(args.catalog)
    baseline = load_baseline(args.baseline) if args.baseline else None

    print(render_stdout(data))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(snapshot_dict(data), indent=2), encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(data, args.catalog, baseline), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
