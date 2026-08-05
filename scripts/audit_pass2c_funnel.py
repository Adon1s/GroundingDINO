"""
Pass 2c funnel and absence-safety audit over an artifact corpus.

Observe-only: this script never writes to an artifact, never calls the VLM, and
never re-implements pass logic. It reads what Pass 2c already recorded.

Pass 2c output lives *only* in `photo_intel_debug.json` — the slim
`photo_intel.json` strips the per-photo `debug` block, so a corpus walk over the
slim artifact would find nothing. See `tools/artifact_writers.py:610`.

Per artifact it reports:
  * the 2c funnel — `debug.labeled_debug` in, `debug.labeled_forward` out —
    bucketed by the photo's scene group (`photos[k].scene.group`, because
    `scene_group` is stamped onto forwarded observations only and can therefore
    never serve as a denominator);
  * the label histogram, split forwarded vs dropped, including the deprecated
    `safety` label that `_coerce_labeled_2c` silently folds into `other`;
  * two lexical cohorts — gutter-absence and exterior-finish observations;
  * a property-level contradiction proxy: properties carrying both an
    absence-shaped gutter observation and other gutter/downspout evidence;
  * the absence cohort's downstream resolution, joined to
    `debug.resolved_items[]`, which is the number that says whether an absence
    claim is reaching a catalog item it must not reach.

THE LEXICAL COHORTS ARE REVIEW QUEUES, NOT HUMAN TRUTH. They are regex families
over model-authored prose; cohort size swings materially with pattern wording.
Use them to find things to look at and to gate a precision-only deny list, never
as ground truth about a property. `--json` always embeds the exact pattern set
that produced the numbers so a cohort count is never quotable without it.

Usage (run from repo root):
    .venv\\Scripts\\python.exe scripts/audit_pass2c_funnel.py \\
        --artifacts-root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts \\
        --json artifacts/pass2c_funnel_before.json \\
        --report docs/pass2c_funnel_audit.md
    # ...apply the change, then:
    .venv\\Scripts\\python.exe scripts/audit_pass2c_funnel.py \\
        --artifacts-root C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts \\
        --baseline artifacts/pass2c_funnel_before.json \\
        --json artifacts/pass2c_funnel_after.json \\
        --report docs/pass2c_funnel_audit.md \\
        --fail-on-absence-resolution

`--artifacts-root` is required on purpose: the ARTIFACTS_ROOT in .env points at
a path that no longer exists, so defaulting to it would silently audit nothing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Make `tools` importable regardless of the caller's working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import pipeline_config as cfg  # noqa: E402
from tools.artifact_writers import load_issue_catalog  # noqa: E402
from tools.catalog_embeddings import build_guardrails_from_catalog  # noqa: E402
from tools.pipeline_common import SCENE_GROUPS_UI, term_matches  # noqa: E402

AUDIT_KEY = "pass2c_funnel_audit_v1"

ARTIFACT_NAME = "photo_intel_debug.json"

# The forward split at scene_classifier_passes.py:1031. Mirrored, not imported,
# so the audit still reports honestly if the split changes underneath it — a
# divergence shows up as a forwarded label outside this set.
FORWARDED_LABELS = ("defect_or_damage", "upgrade_candidate")

# Absence claims must never resolve onto either of these: absent / damaged /
# maintenance are three different claims with three different cost implications.
# See docs/HANDOFF_pass2c_exterior_recall.md.
GUTTER_ITEM_IDS = ("clogged_or_damaged_gutters", "gutter_maintenance_needed")


class DenyGate:
    """Replays the *current* catalog deny lists over *stored* descriptions.

    A stored resolution records the catalog as it was when the run was analysed,
    so a historical corpus can never show the effect of a deny-list change —
    every artifact would have to be re-analysed first. Replaying the live deny
    lists over the recorded descriptions answers the question that actually
    matters ("would this stored resolution still be allowed today?") without
    re-running the pipeline, and is what `--fail-on-absence-resolution` gates on.

    **This is a deny-list replay over historical resolutions, not a fresh
    embedding/retrieval simulation.** It mirrors
    `CatalogEmbeddingsRetriever._passes_guardrails` (catalog_embeddings.py:441)
    for the deny branch only, and needs no embeddings server. `require_any` is
    not replayed: it gates retrieval jointly with the embedding score, which is
    not reconstructible offline. So the count answers "would this stored
    resolution survive the current deny lists?" — never "what would retrieval
    produce today?". A catalog change that alters embed_text, support terms, or
    which candidates score highest is invisible here; only re-analysis shows it.
    """

    def __init__(self, catalog_path: Path):
        self.catalog_path = catalog_path
        catalog = load_issue_catalog(catalog_path)
        self.catalog_version = str(catalog.get("version") or "")
        guardrails = build_guardrails_from_catalog(catalog)
        self.deny_by_item = {
            item_id: tuple(g.get("deny_any") or ())
            for item_id, g in guardrails.items()
        }

    def denies(self, item_id: str, text: str) -> bool:
        terms = self.deny_by_item.get(item_id) or ()
        lowered = text.lower()
        return any(term_matches(term, lowered) for term in terms)


# ─── lexical cohorts (review queues — see module docstring) ──────────────────

GUTTER_TERM = r"(?:gutter|downspout|eaves?\s*trough)"

# Each entry is (name, pattern). `{G}` expands to the gutter-term alternation.
# `(?:(?!\.).){0,40}?` keeps a match inside one sentence: a negation in the
# previous clause is not a negation of this component.
ABSENCE_PATTERN_SPECS: Tuple[Tuple[str, str], ...] = (
    ("no_before_component", r"\bno\b(?:(?!\.).){0,40}?{G}"),
    ("component_not_visible", r"{G}(?:(?!\.).){0,40}?\bnot\s+(?:visible|present|observed|apparent)\b"),
    ("lack_or_absence_of", r"\b(?:lack|absence)\s+of\b(?:(?!\.).){0,40}?{G}"),
    ("without_component", r"\bwithout\b(?:(?!\.).){0,40}?{G}"),
    ("appears_limited", r"{G}(?:(?!\.).){0,40}?\bappears?\s+limited\b"),
    ("missing_component", r"\bmissing\b(?:(?!\.).){0,30}?{G}"),
)

# Exterior finishes that the 2c prompt historically had no example anchor for.
EXTERIOR_FINISH_TERMS: Tuple[str, ...] = (
    "siding", "fascia", "soffit", "trim", "deck", "porch", "masonry",
    "brick", "mortar", "stucco", "shingle", "clapboard", "railing",
)


def _compile(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern.replace("{G}", GUTTER_TERM), re.IGNORECASE)


_GUTTER_RE = _compile(r"{G}")
_ABSENCE_RES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (name, _compile(pat)) for name, pat in ABSENCE_PATTERN_SPECS
)
_EXTERIOR_FINISH_RE = re.compile(
    r"\b(?:" + "|".join(EXTERIOR_FINISH_TERMS) + r")", re.IGNORECASE
)


def mentions_gutter(text: str) -> bool:
    return bool(_GUTTER_RE.search(text))


def absence_pattern_hits(text: str) -> Tuple[str, ...]:
    """Which absence patterns fire. Empty unless a gutter term is also present.

    Returning the pattern names rather than a bool keeps the cohort auditable:
    a reviewer can see *why* a string was queued, and a pattern that only ever
    fires on false positives is visible in the histogram.
    """
    if not mentions_gutter(text):
        return ()
    return tuple(name for name, rx in _ABSENCE_RES if rx.search(text))


def is_absence_shaped(text: str) -> bool:
    return bool(absence_pattern_hits(text))


def mentions_exterior_finish(text: str) -> bool:
    return bool(_EXTERIOR_FINISH_RE.search(text))


def pattern_manifest() -> Dict[str, Any]:
    """Embedded in every snapshot so cohort counts are never quotable alone."""
    return {
        "gutter_term": GUTTER_TERM,
        "absence_patterns": [
            {"name": name, "pattern": pat} for name, pat in ABSENCE_PATTERN_SPECS
        ],
        "exterior_finish_terms": list(EXTERIOR_FINISH_TERMS),
        "caveat": (
            "Lexical cohorts are review queues, not human truth. Cohort size "
            "swings materially with pattern wording; do not quote a count "
            "without this manifest."
        ),
    }


# ─── corpus walk ─────────────────────────────────────────────────────────────

def iter_run_dirs(property_dir: Path) -> List[Path]:
    """Run dirs under a property that actually carry a debug artifact.

    Two things this filters that a naive `iterdir()` does not: `.checkpoints`
    is a sibling directory of the run dirs, and empty run dirs exist in the
    corpus (a run that died before its first write leaves the directory behind).
    """
    if not property_dir.is_dir():
        return []
    return sorted(
        d for d in property_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and (d / ARTIFACT_NAME).is_file()
    )


def _run_sort_key(run_dir: Path) -> Tuple[str, str]:
    """Newest-first ordering key: (`run.created_at`, dir name).

    `created_at` is the authoritative timestamp, but it is absent on older
    artifacts and requires opening the file, so the timestamped dir name is the
    fallback and the tie-breaker. Reading only this one field means selection
    does not pay for a full parse of every run.
    """
    created_at = ""
    try:
        with (run_dir / ARTIFACT_NAME).open("r", encoding="utf-8") as f:
            artifact = json.load(f)
        created_at = str(((artifact.get("run") or {}).get("created_at")) or "")
    except Exception:
        # A malformed artifact still gets to compete on dir name; it is reported
        # as a parse error later rather than silently excluded from selection.
        created_at = ""
    return (created_at, run_dir.name)


def select_artifacts(root: Path, all_runs: bool = False) -> List[Path]:
    """Artifact paths to audit — latest run per property unless `all_runs`."""
    out: List[Path] = []
    for property_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if property_dir.name.startswith("."):
            continue
        run_dirs = iter_run_dirs(property_dir)
        if not run_dirs:
            continue
        if all_runs:
            out.extend(d / ARTIFACT_NAME for d in run_dirs)
        else:
            newest = max(run_dirs, key=_run_sort_key)
            out.append(newest / ARTIFACT_NAME)
    return out


# ─── per-artifact audit ──────────────────────────────────────────────────────

def _observations(photo: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    rows = (photo.get("debug") or {}).get(key)
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _scene_group(photo: Dict[str, Any]) -> str:
    """The photo's scene group — the only sound denominator for a forward rate.

    `scene_group` is stamped onto `labeled_forward` rows only (orchestrator:838)
    and never onto `labeled_debug`, so grouping by an observation field would
    divide a full numerator by an empty denominator.
    """
    group = str(((photo.get("scene") or {}).get("group")) or "").strip().lower()
    return group or "unknown"


def audit_artifact(artifact_path: Path, deny_gate: Optional[DenyGate] = None) -> Dict[str, Any]:
    try:
        with artifact_path.open("r", encoding="utf-8") as f:
            artifact = json.load(f)
    except Exception as exc:
        return {
            "status": "parse_error",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    photos = artifact.get("photos")
    if not isinstance(photos, dict):
        return {"status": "skipped", "reason": "no_photos_map"}

    by_group: Dict[str, Dict[str, int]] = {}
    labels_forwarded: Counter = Counter()
    labels_dropped: Counter = Counter()
    absence_patterns: Counter = Counter()
    absence_labels: Counter = Counter()
    absence_resolutions: Counter = Counter()
    exterior_finish_forwarded = 0
    exterior_finish_dropped = 0

    observations = 0
    forwarded = 0
    absence_total = 0
    absence_forwarded = 0
    absence_unresolved = 0
    still_resolving = 0
    has_absence = False
    has_other_gutter_evidence = False
    absence_examples: List[Dict[str, Any]] = []

    for photo_key in sorted(photos):
        photo = photos.get(photo_key)
        if not isinstance(photo, dict):
            continue
        group = _scene_group(photo)
        debug_rows = _observations(photo, "labeled_debug")
        forward_rows = _observations(photo, "labeled_forward")

        bucket = by_group.setdefault(group, {"observations": 0, "forwarded": 0})
        bucket["observations"] += len(debug_rows)
        bucket["forwarded"] += len(forward_rows)
        observations += len(debug_rows)
        forwarded += len(forward_rows)

        # Forwarding is identity-preserving (the split is a comprehension over
        # the same dicts), so description equality is the join key downstream.
        forwarded_desc = {str(r.get("description") or "") for r in forward_rows}
        resolved_by_desc = {
            str(r.get("description") or ""): r.get("resolved_item_id")
            for r in _observations(photo, "resolved_items")
        }

        for row in debug_rows:
            desc = str(row.get("description") or "")
            label = str(row.get("label") or "") or "<missing>"
            is_forwarded = desc in forwarded_desc
            (labels_forwarded if is_forwarded else labels_dropped)[label] += 1

            if mentions_exterior_finish(desc):
                if is_forwarded:
                    exterior_finish_forwarded += 1
                else:
                    exterior_finish_dropped += 1

            if not mentions_gutter(desc):
                continue
            hits = absence_pattern_hits(desc)
            if not hits:
                has_other_gutter_evidence = True
                continue

            has_absence = True
            absence_total += 1
            absence_labels[label] += 1
            for name in hits:
                absence_patterns[name] += 1
            if not is_forwarded:
                continue
            absence_forwarded += 1
            resolved_id = resolved_by_desc.get(desc)
            if resolved_id:
                absence_resolutions[str(resolved_id)] += 1
                if str(resolved_id) in GUTTER_ITEM_IDS:
                    denied = bool(deny_gate and deny_gate.denies(str(resolved_id), desc))
                    if not denied:
                        still_resolving += 1
                    absence_examples.append({
                        "photo_key": photo_key,
                        "scene_group": group,
                        "label": label,
                        "resolved_item_id": str(resolved_id),
                        "description": desc,
                        "absence_patterns": list(hits),
                        "denied_by_current_catalog": denied,
                    })
            else:
                absence_unresolved += 1

    return {
        "status": "audited",
        "property_key": str(((artifact.get("property") or {}).get("property_key")) or ""),
        "run_id": str(((artifact.get("run") or {}).get("run_id")) or ""),
        "created_at": str(((artifact.get("run") or {}).get("created_at")) or ""),
        "photos": len(photos),
        "observations": observations,
        "forwarded": forwarded,
        "by_scene_group": {
            group: dict(counts) for group, counts in sorted(by_group.items())
        },
        "labels_forwarded": dict(labels_forwarded),
        "labels_dropped": dict(labels_dropped),
        "absence_cohort": {
            "observations": absence_total,
            "forwarded": absence_forwarded,
            "unresolved": absence_unresolved,
            "labels": dict(absence_labels),
            "pattern_hits": dict(absence_patterns),
            "resolutions": dict(absence_resolutions),
            "gutter_item_resolutions": sum(
                count for item_id, count in absence_resolutions.items()
                if item_id in GUTTER_ITEM_IDS
            ),
            # Historical resolutions the live deny lists would no longer allow.
            "gutter_item_resolutions_under_current_catalog": still_resolving,
            "examples": absence_examples,
        },
        "exterior_finish_cohort": {
            "forwarded": exterior_finish_forwarded,
            "dropped": exterior_finish_dropped,
        },
        # The proxy the handoff asks for: an absence claim is least defensible
        # on a property whose other photos *do* show a gutter system.
        "contradiction": bool(has_absence and has_other_gutter_evidence),
    }


def build_audit(
    artifacts_root: Path,
    all_runs: bool = False,
    catalog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    deny_gate = DenyGate(catalog_path or Path(cfg.ISSUE_CATALOG_PATH))

    rows: List[Dict[str, Any]] = []
    for artifact_path in select_artifacts(artifacts_root, all_runs=all_runs):
        row = audit_artifact(artifact_path, deny_gate=deny_gate)
        row["artifact"] = artifact_path.relative_to(artifacts_root).as_posix()
        rows.append(row)

    rows.sort(key=lambda r: r["artifact"])
    return {
        "audit": AUDIT_KEY,
        "artifacts_root": str(artifacts_root),
        "selection": "all_runs" if all_runs else "latest_run_per_property",
        "catalog": str(deny_gate.catalog_path),
        "catalog_version": deny_gate.catalog_version,
        "patterns": pattern_manifest(),
        "artifacts": rows,
        "summary": summarize(rows),
    }


def _histogram(counter: Counter, key: str) -> List[Dict[str, Any]]:
    """Deterministic histogram: most frequent first, ties broken by name."""
    return [
        {key: name, "occurrences": count}
        for name, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _merge(rows: Iterable[Dict[str, Any]], *path: str) -> Counter:
    merged: Counter = Counter()
    for row in rows:
        node: Any = row
        for part in path:
            node = (node or {}).get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            merged.update({str(k): int(v) for k, v in node.items()})
    return merged


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    audited = [r for r in rows if r.get("status") == "audited"]
    parse_errors = [r for r in rows if r.get("status") == "parse_error"]
    skipped: Counter = Counter(
        str(r.get("reason") or "unknown") for r in rows if r.get("status") == "skipped"
    )

    by_group: Dict[str, Dict[str, Any]] = {}
    for row in audited:
        for group, counts in (row.get("by_scene_group") or {}).items():
            bucket = by_group.setdefault(group, {"observations": 0, "forwarded": 0})
            bucket["observations"] += int(counts.get("observations") or 0)
            bucket["forwarded"] += int(counts.get("forwarded") or 0)
    for bucket in by_group.values():
        bucket["forward_rate"] = _rate(bucket["forwarded"], bucket["observations"])

    labels_forwarded = _merge(audited, "labels_forwarded")
    labels_dropped = _merge(audited, "labels_dropped")
    absence_resolutions = _merge(audited, "absence_cohort", "resolutions")

    observations = sum(int(r.get("observations") or 0) for r in audited)
    forwarded = sum(int(r.get("forwarded") or 0) for r in audited)
    absence_observations = sum(
        int((r.get("absence_cohort") or {}).get("observations") or 0) for r in audited
    )
    absence_forwarded = sum(
        int((r.get("absence_cohort") or {}).get("forwarded") or 0) for r in audited
    )
    absence_unresolved = sum(
        int((r.get("absence_cohort") or {}).get("unresolved") or 0) for r in audited
    )
    gutter_item_resolutions = sum(
        count for item_id, count in absence_resolutions.items()
        if item_id in GUTTER_ITEM_IDS
    )

    return {
        "artifacts_scanned": len(rows),
        "artifacts_audited": len(audited),
        "artifacts_parse_error": len(parse_errors),
        "artifacts_skipped": sum(skipped.values()),
        "parse_errors": [
            {"artifact": r.get("artifact"), "reason": r.get("reason")}
            for r in sorted(parse_errors, key=lambda r: str(r.get("artifact")))
        ],
        "skip_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(skipped.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "photos": sum(int(r.get("photos") or 0) for r in audited),
        "observations": observations,
        "forwarded": forwarded,
        "forward_rate": _rate(forwarded, observations),
        "by_scene_group": dict(sorted(by_group.items())),
        "labels_forwarded": _histogram(labels_forwarded, "label"),
        "labels_dropped": _histogram(labels_dropped, "label"),
        "absence_cohort": {
            "observations": absence_observations,
            "forwarded": absence_forwarded,
            "unresolved": absence_unresolved,
            "properties": sum(
                1 for r in audited
                if int((r.get("absence_cohort") or {}).get("observations") or 0)
            ),
            "properties_with_contradiction": sum(
                1 for r in audited if r.get("contradiction")
            ),
            "labels": _histogram(_merge(audited, "absence_cohort", "labels"), "label"),
            "pattern_hits": _histogram(
                _merge(audited, "absence_cohort", "pattern_hits"), "pattern"
            ),
            "resolutions": _histogram(absence_resolutions, "catalog_item_id"),
            "gutter_item_resolutions": gutter_item_resolutions,
            "gutter_item_resolutions_under_current_catalog": sum(
                int((r.get("absence_cohort") or {}).get(
                    "gutter_item_resolutions_under_current_catalog") or 0)
                for r in audited
            ),
        },
        "exterior_finish_cohort": {
            "forwarded": sum(
                int((r.get("exterior_finish_cohort") or {}).get("forwarded") or 0)
                for r in audited
            ),
            "dropped": sum(
                int((r.get("exterior_finish_cohort") or {}).get("dropped") or 0)
                for r in audited
            ),
        },
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


# ─── snapshot / baseline ─────────────────────────────────────────────────────

def snapshot_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return data


def load_baseline(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_deltas(
    data: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Corpus-level before/after movement on the numbers this audit exists for."""
    if not baseline:
        return {"available": False, "corpus": {}, "by_scene_group": {}}

    before = baseline.get("summary") or {}
    after = data.get("summary") or {}

    def _scalar(node: Dict[str, Any], *path: str) -> int:
        cur: Any = node
        for part in path:
            cur = (cur or {}).get(part) if isinstance(cur, dict) else None
        return int(cur or 0)

    corpus: Dict[str, Dict[str, int]] = {}
    for name, path in (
        ("observations", ("observations",)),
        ("forwarded", ("forwarded",)),
        ("absence_observations", ("absence_cohort", "observations")),
        ("absence_forwarded", ("absence_cohort", "forwarded")),
        ("absence_unresolved", ("absence_cohort", "unresolved")),
        ("absence_gutter_item_resolutions", ("absence_cohort", "gutter_item_resolutions")),
        ("absence_gutter_item_resolutions_live",
         ("absence_cohort", "gutter_item_resolutions_under_current_catalog")),
        ("exterior_finish_forwarded", ("exterior_finish_cohort", "forwarded")),
        ("exterior_finish_dropped", ("exterior_finish_cohort", "dropped")),
    ):
        b, a = _scalar(before, *path), _scalar(after, *path)
        corpus[name] = {"before": b, "after": a, "delta": a - b}

    groups: Dict[str, Dict[str, float]] = {}
    before_groups = before.get("by_scene_group") or {}
    for group, counts in (after.get("by_scene_group") or {}).items():
        b = (before_groups.get(group) or {}).get("forward_rate")
        a = counts.get("forward_rate")
        groups[group] = {
            "before": float(b or 0.0),
            "after": float(a or 0.0),
            "delta": round(float(a or 0.0) - float(b or 0.0), 2),
        }

    return {"available": True, "corpus": corpus, "by_scene_group": groups}


# ─── rendering ───────────────────────────────────────────────────────────────

def markdown_table(headers: Tuple[str, ...], rows: List[Tuple[Any, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _group_order(groups: Dict[str, Any]) -> List[str]:
    """Canonical scene-group order first, then anything unexpected."""
    known = [g for g in SCENE_GROUPS_UI if g in groups]
    return known + sorted(g for g in groups if g not in set(known))


def render_markdown(data: Dict[str, Any], deltas: Dict[str, Any]) -> str:
    s = data["summary"]
    absence = s["absence_cohort"]
    out: List[str] = []
    out.append("# Pass 2c funnel audit")
    out.append("")
    out.append(f"- Artifacts root: `{data['artifacts_root']}`")
    out.append(f"- Selection: {data['selection']}")
    out.append(
        f"- Scanned {s['artifacts_scanned']}, audited {s['artifacts_audited']}, "
        f"parse errors {s['artifacts_parse_error']}, skipped {s['artifacts_skipped']}"
    )
    out.append(
        f"- {s['observations']:,} observations → {s['forwarded']:,} forwarded "
        f"({s['forward_rate']}%)"
    )
    out.append("")

    out.append("## Forward rate by scene group")
    out.append("")
    groups = s["by_scene_group"]
    delta_groups = deltas.get("by_scene_group") or {}
    if deltas.get("available"):
        out.append(markdown_table(
            ("Scene group", "Observations", "Forwarded", "Rate", "Δ rate"),
            [
                (
                    g, f"{groups[g]['observations']:,}", f"{groups[g]['forwarded']:,}",
                    f"{groups[g]['forward_rate']}%",
                    f"{(delta_groups.get(g) or {}).get('delta', 0.0):+}",
                )
                for g in _group_order(groups)
            ],
        ))
    else:
        out.append(markdown_table(
            ("Scene group", "Observations", "Forwarded", "Rate"),
            [
                (
                    g, f"{groups[g]['observations']:,}", f"{groups[g]['forwarded']:,}",
                    f"{groups[g]['forward_rate']}%",
                )
                for g in _group_order(groups)
            ],
        ))
    out.append("")

    out.append("## Labels")
    out.append("")
    out.append(markdown_table(
        ("Label", "Forwarded", "Dropped"),
        _label_rows(s),
    ))
    out.append("")

    out.append("## Gutter-absence cohort")
    out.append("")
    out.append(
        "_Review queue, not human truth._ Regex families over model-authored "
        "prose; see `patterns` in the JSON snapshot for the exact set."
    )
    out.append("")
    out.append(
        f"{absence['observations']:,} absence-shaped observations across "
        f"{absence['properties']:,} propert{'y' if absence['properties'] == 1 else 'ies'}; "
        f"{absence['forwarded']:,} forwarded, {absence['unresolved']:,} of those resolved "
        f"to nothing. {absence['properties_with_contradiction']:,} propert"
        f"{'y' if absence['properties_with_contradiction'] == 1 else 'ies'} also carry "
        f"non-absence gutter evidence."
    )
    out.append("")
    out.append(
        f"**Stored resolutions onto a gutter item: "
        f"{absence['gutter_item_resolutions']}** "
        f"(`{'`, `'.join(GUTTER_ITEM_IDS)}`) — absence must never map here. "
        f"Of those, **{absence['gutter_item_resolutions_under_current_catalog']}** "
        f"would still resolve under the current catalog "
        f"(`{data.get('catalog_version') or '?'}`); the rest are now blocked by "
        f"`deny_any`. Stored artifacts record the catalog as it was when they "
        f"were analysed, so only the replayed number can move without a "
        f"corpus-wide re-analysis."
    )
    out.append("")
    if absence["resolutions"]:
        out.append(markdown_table(
            ("Resolved catalog item", "Occurrences", "Forbidden"),
            [
                (
                    e["catalog_item_id"], e["occurrences"],
                    "**yes**" if e["catalog_item_id"] in GUTTER_ITEM_IDS else "no",
                )
                for e in absence["resolutions"]
            ],
        ))
        out.append("")
    if absence["pattern_hits"]:
        out.append(markdown_table(
            ("Absence pattern", "Hits"),
            [(e["pattern"], e["occurrences"]) for e in absence["pattern_hits"]],
        ))
        out.append("")

    offenders = [
        (row.get("property_key") or row.get("artifact"), ex)
        for row in data.get("artifacts") or []
        for ex in ((row.get("absence_cohort") or {}).get("examples") or [])
    ]
    if offenders:
        out.append("### Absence claims resolving onto a gutter item")
        out.append("")
        out.append(markdown_table(
            ("Property", "Photo", "Resolved to", "Now blocked", "Description"),
            [
                (
                    prop, ex["photo_key"], ex["resolved_item_id"],
                    "yes" if ex.get("denied_by_current_catalog") else "**no**",
                    ex["description"],
                )
                for prop, ex in offenders
            ],
        ))
        out.append("")

    ext = s["exterior_finish_cohort"]
    total_ext = ext["forwarded"] + ext["dropped"]
    out.append("## Exterior-finish cohort")
    out.append("")
    out.append(
        f"{total_ext:,} observations mention an exterior finish term; "
        f"{ext['forwarded']:,} forwarded ({_rate(ext['forwarded'], total_ext)}%), "
        f"{ext['dropped']:,} dropped."
    )
    out.append("")

    if s["parse_errors"]:
        out.append("## Parse errors")
        out.append("")
        out.append(markdown_table(
            ("Artifact", "Reason"),
            [(e["artifact"], e["reason"]) for e in s["parse_errors"]],
        ))
        out.append("")
    return "\n".join(out)


def _label_rows(summary: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    fwd = {e["label"]: e["occurrences"] for e in summary["labels_forwarded"]}
    drop = {e["label"]: e["occurrences"] for e in summary["labels_dropped"]}
    names = sorted(set(fwd) | set(drop), key=lambda n: (-(fwd.get(n, 0) + drop.get(n, 0)), n))
    return [(n, f"{fwd.get(n, 0):,}", f"{drop.get(n, 0):,}") for n in names]


def render_stdout(data: Dict[str, Any], deltas: Dict[str, Any]) -> str:
    # ASCII only: the Windows console is cp1252 and a "->" that cannot encode
    # takes down the whole run at the print, after the corpus walk has been paid
    # for. The markdown report is written UTF-8 and keeps the real glyphs.
    s = data["summary"]
    absence = s["absence_cohort"]
    ext = s["exterior_finish_cohort"]
    out = [
        f"scanned={s['artifacts_scanned']}  audited={s['artifacts_audited']}  "
        f"parse_errors={s['artifacts_parse_error']}  skipped={s['artifacts_skipped']}",
        f"observations={s['observations']:,}  forwarded={s['forwarded']:,}  "
        f"rate={s['forward_rate']}%",
        f"absence: obs={absence['observations']}  forwarded={absence['forwarded']}  "
        f"unresolved={absence['unresolved']}  properties={absence['properties']}  "
        f"contradictions={absence['properties_with_contradiction']}",
        f"absence->gutter_item: stored={absence['gutter_item_resolutions']}  "
        f"under_current_catalog="
        f"{absence['gutter_item_resolutions_under_current_catalog']}"
        f"  (the latter is what --fail-on-absence-resolution gates on)",
        f"exterior_finish: forwarded={ext['forwarded']}  dropped={ext['dropped']}",
    ]
    for group in _group_order(s["by_scene_group"]):
        counts = s["by_scene_group"][group]
        out.append(
            f"  {group:<14} {counts['observations']:>7,} -> {counts['forwarded']:>7,}"
            f"  {counts['forward_rate']:>6}%"
        )
    for err in s["parse_errors"]:
        out.append(f"  parse_error: {err['artifact']}: {err['reason']}")
    if deltas.get("available"):
        for name, entry in deltas["corpus"].items():
            if entry["delta"]:
                out.append(
                    f"  delta {name}: {entry['delta']:+,} "
                    f"({entry['before']} -> {entry['after']})"
                )
    return "\n".join(out)


# ─── entry point ─────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Pass 2c funnel and gutter-absence safety across an "
            "artifact corpus (observe-only)."
        ),
    )
    parser.add_argument(
        "--artifacts-root", required=True, type=Path,
        help="Root directory of the artifact corpus (<root>/<property>/<run>/).",
    )
    parser.add_argument(
        "--all-runs", action="store_true",
        help="Audit every run; default is the latest run per property.",
    )
    parser.add_argument(
        "--catalog", default=Path(cfg.ISSUE_CATALOG_PATH), type=Path,
        help="Catalog whose deny lists are replayed over stored descriptions.",
    )
    parser.add_argument("--report", default=None, type=Path,
                        help="Write the markdown report here.")
    parser.add_argument("--json", dest="json_out", default=None, type=Path,
                        help="Write a machine-readable snapshot here (for --baseline diffs).")
    parser.add_argument("--baseline", default=None, type=Path,
                        help="A prior --json snapshot to diff against (before/after).")
    parser.add_argument(
        "--fail-on-absence-resolution", action="store_true",
        help=(
            "Exit non-zero if any absence-shaped observation would still resolve "
            "onto clogged_or_damaged_gutters or gutter_maintenance_needed under "
            "the current catalog's deny lists."
        ),
    )
    parser.add_argument(
        "--max-absence-resolutions", type=int, default=0, metavar="N",
        help=(
            "Allowance for --fail-on-absence-resolution. Default 0, which is the "
            "right setting for any newly analysed corpus. The historical corpus "
            "carries 2 accepted residuals that assert absence and damage in one "
            "sentence; see docs/HANDOFF_pass2c_exterior_recall.md."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.artifacts_root.is_dir():
        print(f"error: artifacts root not found: {args.artifacts_root}", file=sys.stderr)
        return 2

    if not args.catalog.is_file():
        print(f"error: catalog not found: {args.catalog}", file=sys.stderr)
        return 2

    data = build_audit(
        args.artifacts_root, all_runs=args.all_runs, catalog_path=args.catalog,
    )
    summary = data["summary"]

    # A corpus that parsed nothing is a broken invocation, not a clean result.
    # Individual malformed artifacts are reported and walked past.
    if not summary["artifacts_audited"]:
        print(
            f"error: no auditable artifacts under {args.artifacts_root} "
            f"(scanned {summary['artifacts_scanned']}, "
            f"parse errors {summary['artifacts_parse_error']})",
            file=sys.stderr,
        )
        return 1

    baseline = load_baseline(args.baseline) if args.baseline else None
    deltas = compute_deltas(data, baseline)

    print(render_stdout(data, deltas))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(snapshot_dict(data), f, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(data, deltas), encoding="utf-8")

    # Gate on the replayed number, not the historical one: stored artifacts
    # record the catalog as it was when they were analysed, so the historical
    # count can only fall once the whole corpus is re-analysed.
    forbidden = summary["absence_cohort"]["gutter_item_resolutions_under_current_catalog"]
    if args.fail_on_absence_resolution and forbidden > args.max_absence_resolutions:
        print(
            f"error: {forbidden} absence-shaped observation(s) would still resolve "
            f"onto a gutter item ({', '.join(GUTTER_ITEM_IDS)}) under {args.catalog}; "
            f"allowance is {args.max_absence_resolutions}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
