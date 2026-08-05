"""
Exterior estimate-coverage and headline-delta audit over an artifact corpus.

Observe-only: this script never writes to an artifact, never calls the VLM, and
never re-implements estimate logic. It *calls* the real effective-estimate
resolver (`resolve_catalog_estimate_meta`) and the real engine
(`compute_renovation_estimate_v4`), replaying each artifact's stored Pass 2f
package verifications via `collect_stored_verifications` so the recompute is
deterministic and offline.

Per artifact it reports:
  * how many issue occurrences resolve to a catalog item with no effective
    estimate block (overall and for `category == "exterior"`), plus the
    catalog-id histogram behind that count;
  * recomputed headline totals (probable / visible / package-adjusted / final
    and the scope tiers) and the evidence-projection headline;
  * the withheld lane's total and risk exposure;
  * before/after deltas against a prior snapshot, and skip reasons.

Only the keys in `_HEADLINE_PATHS` participate in `--fail-on-headline-delta`.
The withheld lane, `unreviewed_risk_total`, `inspection_allowance_total` and
`latent_risk_exposure` are reported but never fail: they are *expected* to move
when the guard or an inspect-only item lands.

Usage (run from repo root):
    .venv\\Scripts\\python.exe scripts/audit_exterior_estimate_coverage.py \\
        --artifacts-root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts \\
        --json artifacts/exterior_estimate_coverage_before.json \\
        --report docs/exterior_estimate_coverage_audit.md
    # ...apply the change, then:
    .venv\\Scripts\\python.exe scripts/audit_exterior_estimate_coverage.py \\
        --artifacts-root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts \\
        --baseline artifacts/exterior_estimate_coverage_before.json \\
        --json artifacts/exterior_estimate_coverage_after.json \\
        --report docs/exterior_estimate_coverage_audit.md \\
        --fail-on-headline-delta

`--artifacts-root` is required on purpose: the ARTIFACTS_ROOT in .env points at
a path that no longer exists, so defaulting to it would silently audit nothing.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `tools` importable regardless of the caller's working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import pipeline_config as cfg  # noqa: E402
from tools.artifact_writers import load_issue_catalog  # noqa: E402
from tools.backfill_reno_v4 import (  # noqa: E402
    _resolve_property_metadata_from_artifact,
)
from tools.renovation_estimate import (  # noqa: E402
    filter_product_issues,
    resolve_catalog_estimate_meta,
)
from tools.renovation_estimate_v4 import compute_renovation_estimate_v4  # noqa: E402
from tools.reproject_product_views import (  # noqa: E402
    _quarantined_catalog_item_ids,
    collect_stored_verifications,
)

# Totals that must not move when a guarded catalog item is added. Ordered
# outermost-first so a report reads top-down.
_HEADLINE_PATHS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("probable_total", ("totals", "probable_total")),
    ("visible_rehab", ("visible_rehab",)),
    ("package_adjusted_rehab", ("package_adjusted_rehab",)),
    ("final_rehab", ("final_rehab",)),
    ("final_rehab_required", ("final_rehab_required",)),
    ("final_rehab_resale_ready", ("final_rehab_resale_ready",)),
    ("final_rehab_full_renewal", ("final_rehab_full_renewal",)),
    ("evidence_headline", ("rehab_evidence_projection_v1", "headline")),
)

# Reported for visibility; deliberately excluded from the fail check.
_OBSERVED_PATHS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("unreviewed_risk_total", ("totals", "unreviewed_risk_total")),
    ("inspection_allowance_total", ("totals", "inspection_allowance_total")),
    ("latent_risk_exposure", ("latent_risk_exposure",)),
    ("withheld_total", ("withheld_estimate", "total")),
    ("withheld_risk_exposure", ("withheld_estimate", "risk_exposure_total")),
)

_ALL_PATHS = _HEADLINE_PATHS + _OBSERVED_PATHS
_HEADLINE_KEYS = tuple(name for name, _ in _HEADLINE_PATHS)

_EXTERIOR_CATEGORY = "exterior"


# ─── extraction helpers ──────────────────────────────────────────────────────

def _range_at(estimate: Any, path: Tuple[str, ...]) -> Dict[str, int]:
    """Read a {low, high} range out of a nested estimate path.

    Missing or malformed nodes read as zero rather than raising: a corpus walk
    must survive artifacts written by older engine versions.
    """
    node: Any = estimate
    for key in path:
        if not isinstance(node, dict):
            return {"low": 0, "high": 0}
        node = node.get(key)
    if not isinstance(node, dict):
        return {"low": 0, "high": 0}
    low, high = node.get("low"), node.get("high")
    return {
        "low": int(low) if isinstance(low, (int, float)) else 0,
        "high": int(high) if isinstance(high, (int, float)) else 0,
    }


def extract_totals(estimate: Any) -> Dict[str, Dict[str, int]]:
    return {name: _range_at(estimate, path) for name, path in _ALL_PATHS}


def _catalog_lookup(issue_catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        item["id"]: item
        for item in (issue_catalog.get("items") or [])
        if isinstance(item, dict) and item.get("id")
    }


def missing_estimate_coverage(
    issues: List[Dict[str, Any]],
    catalog_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Count issue occurrences whose catalog item has no effective estimate.

    Uses the real resolver, so top-level `affects_estimate` overrides and the
    tier-derived default are both honoured exactly as the engine sees them.
    """
    missing: Counter = Counter()
    missing_exterior: Counter = Counter()
    unmatched = 0
    for issue in issues or []:
        item_id = str(issue.get("catalog_item_id") or "")
        item = catalog_lookup.get(item_id)
        if item is None:
            unmatched += 1
            continue
        if resolve_catalog_estimate_meta(item).affects_estimate:
            continue
        missing[item_id] += 1
        if str(item.get("category") or "") == _EXTERIOR_CATEGORY:
            missing_exterior[item_id] += 1
    return {
        "missing_occurrences": sum(missing.values()),
        "missing_occurrences_exterior": sum(missing_exterior.values()),
        "missing_by_catalog_id": _histogram(missing),
        "missing_exterior_by_catalog_id": _histogram(missing_exterior),
        "unmatched_issue_occurrences": unmatched,
    }


def _histogram(counter: Counter) -> List[Dict[str, Any]]:
    """Deterministic histogram: most frequent first, ties broken by id."""
    return [
        {"catalog_item_id": item_id, "occurrences": count}
        for item_id, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


# ─── per-artifact audit ──────────────────────────────────────────────────────

def _select_issue_lane(artifact: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Mirror the live writer: prefer the raw estimate lane when it exists."""
    raw_display = artifact.get("issues_flat")
    if not isinstance(raw_display, list) or not raw_display:
        return None
    raw_estimate = artifact.get("estimate_issues_flat")
    if isinstance(raw_estimate, list) and raw_estimate:
        return raw_estimate
    return raw_display


def _has_unreplayable_packages(
    artifact: Dict[str, Any],
    verifications: Dict[str, Dict[str, Any]],
) -> bool:
    """True when the stored artifact has ACTIVE packages we cannot replay.

    Withholding a verification is a kill switch: finalization requires
    confirmation, so a package re-inferred without its verdict is dropped from
    dollars entirely rather than merely re-reviewed. That only distorts the
    recompute when the stored artifact actually had active packages — an
    artifact whose candidates were all `not_run` had none to begin with, so
    dropping nothing reproduces it faithfully and is worth auditing.
    """
    if verifications:
        return False
    v4 = artifact.get("renovation_estimate_v4")
    if not isinstance(v4, dict):
        return False
    active = v4.get("packages")
    return isinstance(active, list) and bool(active)


def audit_artifact(
    artifact_path: Path,
    issue_catalog: Dict[str, Any],
    catalog_lookup: Dict[str, Dict[str, Any]],
    quarantined_item_ids: frozenset,
) -> Dict[str, Any]:
    """Audit one photo_intel.json. Returns a status dict; never raises."""
    try:
        with artifact_path.open("r", encoding="utf-8") as f:
            artifact = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"status": "skipped", "reason": f"cannot load artifact: {e}"}
    if not isinstance(artifact, dict):
        return {"status": "skipped", "reason": "artifact root must be a JSON object"}

    if not isinstance(artifact.get("renovation_estimate_v4"), dict):
        return {"status": "skipped", "reason": "no renovation_estimate_v4"}

    lane = _select_issue_lane(artifact)
    if lane is None:
        return {"status": "skipped", "reason": "no usable issues_flat lane"}

    issues = filter_product_issues(lane, issue_catalog)
    verifications, tainted = collect_stored_verifications(
        artifact, quarantined_item_ids,
    )
    if _has_unreplayable_packages(artifact, verifications):
        return {
            "status": "skipped",
            "reason": "stored packages carry no replayable verification",
        }

    try:
        recomputed = compute_renovation_estimate_v4(
            issues_flat=issues,
            issue_catalog=issue_catalog,
            photos=artifact.get("photos") or {},
            property_metadata=(
                _resolve_property_metadata_from_artifact(artifact) or None
            ),
            package_verifications=verifications,
            pass_2f_vlm_client=None,
        )
    except Exception as e:  # noqa: BLE001 - a corpus walk must not abort
        return {"status": "skipped", "reason": f"recompute failed: {e}"}

    coverage = missing_estimate_coverage(issues, catalog_lookup)
    withheld = recomputed.get("withheld_estimate")
    meta = recomputed.get("meta") or {}
    return {
        "status": "audited",
        "property_key": str(
            (artifact.get("property") or {}).get("property_key")
            or artifact.get("property_key")
            or ""
        ),
        "run_id": str((artifact.get("run") or {}).get("run_id") or ""),
        "issue_count": len(issues),
        **coverage,
        "totals": extract_totals(recomputed),
        "stored_totals": extract_totals(artifact.get("renovation_estimate_v4")),
        "verifications_reused": len(verifications),
        "tainted_package_count": len(tainted),
        "withheld_line_item_count": len(
            (withheld or {}).get("line_items") or []
        ) if isinstance(withheld, dict) else 0,
        "priced_candidate_count": meta.get("priced_candidate_count"),
        "withheld_candidate_count": meta.get("withheld_candidate_count"),
    }


# ─── corpus walk ─────────────────────────────────────────────────────────────

def iter_artifact_paths(root: Path) -> List[Path]:
    """Exact `photo_intel.json` files only — the `.pre_*` backups and the
    debug artifact are excluded by the filename match itself."""
    return sorted(root.rglob("photo_intel.json"))


def build_audit(artifacts_root: Path, catalog_path: Path) -> Dict[str, Any]:
    issue_catalog = load_issue_catalog(catalog_path)
    catalog_lookup = _catalog_lookup(issue_catalog)
    quarantined_item_ids = _quarantined_catalog_item_ids(issue_catalog)

    rows: List[Dict[str, Any]] = []
    for artifact_path in iter_artifact_paths(artifacts_root):
        result = audit_artifact(
            artifact_path, issue_catalog, catalog_lookup, quarantined_item_ids,
        )
        result["artifact"] = artifact_path.relative_to(artifacts_root).as_posix()
        rows.append(result)

    rows.sort(key=lambda r: r["artifact"])
    return {
        "artifacts_root": str(artifacts_root),
        "catalog": str(catalog_path),
        "catalog_version": str(issue_catalog.get("version") or ""),
        "artifacts": rows,
        "summary": summarize(rows),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    audited = [r for r in rows if r.get("status") == "audited"]
    skipped: Counter = Counter(
        str(r.get("reason") or "unknown") for r in rows if r.get("status") == "skipped"
    )
    missing: Counter = Counter()
    missing_exterior: Counter = Counter()
    for row in audited:
        for entry in row.get("missing_by_catalog_id") or []:
            missing[entry["catalog_item_id"]] += entry["occurrences"]
        for entry in row.get("missing_exterior_by_catalog_id") or []:
            missing_exterior[entry["catalog_item_id"]] += entry["occurrences"]

    corpus_totals: Dict[str, Dict[str, int]] = {}
    for name, _ in _ALL_PATHS:
        corpus_totals[name] = {
            "low": sum(r["totals"][name]["low"] for r in audited),
            "high": sum(r["totals"][name]["high"] for r in audited),
        }
    return {
        "artifacts_scanned": len(rows),
        "artifacts_audited": len(audited),
        "artifacts_skipped": sum(skipped.values()),
        "skip_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(skipped.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "missing_occurrences": sum(missing.values()),
        "missing_occurrences_exterior": sum(missing_exterior.values()),
        "missing_by_catalog_id": _histogram(missing),
        "missing_exterior_by_catalog_id": _histogram(missing_exterior),
        "corpus_totals": corpus_totals,
        "artifacts_with_withheld": sum(
            1 for r in audited if r.get("withheld_line_item_count")
        ),
    }


# ─── snapshot / baseline ─────────────────────────────────────────────────────

def snapshot_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return data


def load_baseline(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _baseline_index(baseline: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not baseline:
        return {}
    return {
        row["artifact"]: row
        for row in baseline.get("artifacts") or []
        if isinstance(row, dict) and row.get("artifact")
    }


def compute_deltas(
    data: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Per-artifact and corpus-level before/after deltas.

    `headline_changes` lists only artifacts where a key in `_HEADLINE_PATHS`
    moved — that is exactly the set `--fail-on-headline-delta` acts on.
    """
    index = _baseline_index(baseline)
    if not index:
        return {"available": False, "headline_changes": [], "corpus": {}}

    headline_changes: List[Dict[str, Any]] = []
    for row in data.get("artifacts") or []:
        if row.get("status") != "audited":
            continue
        before = index.get(row["artifact"])
        if not before or before.get("status") != "audited":
            continue
        moved: Dict[str, Dict[str, int]] = {}
        for name in _HEADLINE_KEYS:
            b = (before.get("totals") or {}).get(name) or {"low": 0, "high": 0}
            a = row["totals"][name]
            delta_low = a["low"] - int(b.get("low") or 0)
            delta_high = a["high"] - int(b.get("high") or 0)
            if delta_low or delta_high:
                moved[name] = {"low": delta_low, "high": delta_high}
        if moved:
            headline_changes.append({
                "artifact": row["artifact"],
                "property_key": row.get("property_key"),
                "moved": moved,
            })

    before_totals = (baseline.get("summary") or {}).get("corpus_totals") or {}
    after_totals = data["summary"]["corpus_totals"]
    corpus = {}
    for name in after_totals:
        b = before_totals.get(name) or {"low": 0, "high": 0}
        corpus[name] = {
            "before": {"low": int(b.get("low") or 0), "high": int(b.get("high") or 0)},
            "after": after_totals[name],
            "delta": {
                "low": after_totals[name]["low"] - int(b.get("low") or 0),
                "high": after_totals[name]["high"] - int(b.get("high") or 0),
            },
        }
    before_summary = baseline.get("summary") or {}
    return {
        "available": True,
        "headline_changes": headline_changes,
        "corpus": corpus,
        "missing_occurrences_before": before_summary.get("missing_occurrences"),
        "missing_occurrences_exterior_before": before_summary.get(
            "missing_occurrences_exterior"
        ),
    }


# ─── rendering ───────────────────────────────────────────────────────────────

def _fmt_range(value: Dict[str, int]) -> str:
    return f"${value.get('low', 0):,}-${value.get('high', 0):,}"


def _fmt_delta(value: Dict[str, int]) -> str:
    low, high = value.get("low", 0), value.get("high", 0)
    if not low and not high:
        return "0"
    return f"{low:+,}/{high:+,}"


def markdown_table(headers: Tuple[str, ...], rows: List[Tuple[Any, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(data: Dict[str, Any], deltas: Dict[str, Any]) -> str:
    summary = data["summary"]
    out: List[str] = []
    out.append("# Exterior estimate coverage audit")
    out.append("")
    out.append(f"- Artifacts root: `{data['artifacts_root']}`")
    out.append(f"- Catalog: `{data['catalog']}` (version {data['catalog_version']})")
    out.append(
        f"- Scanned {summary['artifacts_scanned']}, "
        f"audited {summary['artifacts_audited']}, "
        f"skipped {summary['artifacts_skipped']}"
    )
    out.append("")

    if summary["skip_reasons"]:
        out.append("## Skipped")
        out.append("")
        out.append(markdown_table(
            ("Reason", "Count"),
            [(r["reason"], r["count"]) for r in summary["skip_reasons"]],
        ))
        out.append("")

    out.append("## Coverage")
    out.append("")
    out.append(
        f"Issue occurrences with no effective estimate: "
        f"**{summary['missing_occurrences']}** "
        f"(exterior category: **{summary['missing_occurrences_exterior']}**)"
    )
    before_missing = deltas.get("missing_occurrences_exterior_before")
    if before_missing is not None:
        out.append("")
        out.append(
            f"Exterior before → after: {before_missing} → "
            f"{summary['missing_occurrences_exterior']}"
        )
    out.append("")
    if summary["missing_exterior_by_catalog_id"]:
        out.append(markdown_table(
            ("Exterior catalog item", "Occurrences"),
            [
                (e["catalog_item_id"], e["occurrences"])
                for e in summary["missing_exterior_by_catalog_id"]
            ],
        ))
        out.append("")

    out.append("## Corpus totals")
    out.append("")
    if deltas.get("available"):
        out.append(markdown_table(
            ("Key", "Before", "After", "Delta (low/high)", "Gated"),
            [
                (
                    name,
                    _fmt_range(entry["before"]),
                    _fmt_range(entry["after"]),
                    _fmt_delta(entry["delta"]),
                    "yes" if name in _HEADLINE_KEYS else "no",
                )
                for name, entry in deltas["corpus"].items()
            ],
        ))
    else:
        out.append(markdown_table(
            ("Key", "Total", "Gated"),
            [
                (name, _fmt_range(value), "yes" if name in _HEADLINE_KEYS else "no")
                for name, value in summary["corpus_totals"].items()
            ],
        ))
    out.append("")

    changes = deltas.get("headline_changes") or []
    out.append("## Headline movement")
    out.append("")
    if not deltas.get("available"):
        out.append("_No baseline supplied._")
    elif not changes:
        out.append("None. Every gated headline total is unchanged against the baseline.")
    else:
        out.append(f"{len(changes)} artifact(s) moved a gated headline total.")
        out.append("")
        out.append(markdown_table(
            ("Artifact", "Property", "Key", "Delta (low/high)"),
            [
                (c["artifact"], c.get("property_key") or "", key, _fmt_delta(value))
                for c in changes
                for key, value in sorted(c["moved"].items())
            ],
        ))
    out.append("")

    withheld = summary["corpus_totals"].get("withheld_total") or {}
    out.append("## Withheld lane")
    out.append("")
    out.append(
        f"{summary['artifacts_with_withheld']} artifact(s) carry withheld line items, "
        f"totalling {_fmt_range(withheld)} across the corpus."
    )
    out.append("")
    return "\n".join(out)


def render_stdout(data: Dict[str, Any], deltas: Dict[str, Any]) -> str:
    s = data["summary"]
    out = [
        f"scanned={s['artifacts_scanned']}  audited={s['artifacts_audited']}  "
        f"skipped={s['artifacts_skipped']}",
        f"missing_estimate_occurrences={s['missing_occurrences']}  "
        f"exterior={s['missing_occurrences_exterior']}",
        "final_rehab=" + _fmt_range(s["corpus_totals"]["final_rehab"])
        + "  withheld=" + _fmt_range(s["corpus_totals"]["withheld_total"])
        + f"  artifacts_with_withheld={s['artifacts_with_withheld']}",
    ]
    for reason in s["skip_reasons"]:
        out.append(f"  skip: {reason['reason']} x{reason['count']}")
    if deltas.get("available"):
        changes = deltas.get("headline_changes") or []
        out.append(f"headline_changes={len(changes)}")
        for name, entry in deltas["corpus"].items():
            if entry["delta"]["low"] or entry["delta"]["high"]:
                gate = "GATED" if name in _HEADLINE_KEYS else "observed"
                out.append(f"  {gate} {name}: {_fmt_delta(entry['delta'])}")
    return "\n".join(out)


# ─── entry point ─────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exterior estimate coverage and headline deltas across an "
            "artifact corpus (observe-only)."
        ),
    )
    parser.add_argument(
        "--artifacts-root", required=True, type=Path,
        help="Root directory to walk for photo_intel.json files.",
    )
    parser.add_argument(
        "--catalog", default=Path(cfg.ISSUE_CATALOG_PATH), type=Path,
    )
    parser.add_argument("--report", default=None, type=Path,
                        help="Write the markdown report here.")
    parser.add_argument("--json", dest="json_out", default=None, type=Path,
                        help="Write a machine-readable snapshot here (for --baseline diffs).")
    parser.add_argument("--baseline", default=None, type=Path,
                        help="A prior --json snapshot to diff against (before/after).")
    parser.add_argument(
        "--fail-on-headline-delta", action="store_true",
        help=(
            "Exit non-zero if any gated headline total moved against --baseline. "
            "The withheld lane and unreviewed/inspection/latent-risk totals are "
            "expected to move and never trigger this."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.artifacts_root.is_dir():
        print(f"error: artifacts root not found: {args.artifacts_root}", file=sys.stderr)
        return 2
    if args.fail_on_headline_delta and not args.baseline:
        print("error: --fail-on-headline-delta requires --baseline", file=sys.stderr)
        return 2

    # Observe-only: engine INFO chatter (package cost floors and the like)
    # would bury the audit summary under a thousand-artifact walk.
    logging.getLogger("tools").setLevel(logging.WARNING)

    data = build_audit(args.artifacts_root, args.catalog)
    baseline = load_baseline(args.baseline) if args.baseline else None
    deltas = compute_deltas(data, baseline)

    print(render_stdout(data, deltas))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(snapshot_dict(data), indent=2), encoding="utf-8",
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(data, deltas), encoding="utf-8")

    if args.fail_on_headline_delta and deltas.get("headline_changes"):
        print(
            f"FAIL: {len(deltas['headline_changes'])} artifact(s) moved a gated "
            "headline total.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
