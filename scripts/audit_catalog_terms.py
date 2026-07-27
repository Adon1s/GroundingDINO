"""
Catalog keyword-term audit for tools/issue_catalog.json.

Observe-only: this script *calls* the real matcher (`term_matches` from
tools/pipeline_common.py) — it never re-implements the anchoring regex, so the
report can not drift from production behavior. It harvests the observation
corpus from the `catalog_audit_*.json` runs in the repo root and, per
(item, role, term), reports:

  * anchored hits          - what production matches today
  * whole-word hits        - what a trailing \\b would match
  * raw substring hits     - what the pre-anchoring code matched
  * interior collisions    - words that used to fire and no longer do
  * prefix extensions      - longer words the term still fires inside
  * zero-hit terms         - untested stems the corpus never exercised
  * require_any viability  - per item, can the hard gate still be satisfied

`require_any` is a hard retrieval gate, so an item whose whole require list
scores zero is unretrievable. That roll-up is the headline invariant: re-run
this after any matcher or term change and confirm the zero-hit item set has not
grown.

Usage (run from repo root):
    .venv\\Scripts\\python.exe scripts/audit_catalog_terms.py \\
        --json artifacts/catalog_terms_before.json \\
        --report docs/catalog_term_audit.md
    # ...apply term edits, then:
    .venv\\Scripts\\python.exe scripts/audit_catalog_terms.py \\
        --baseline artifacts/catalog_terms_before.json \\
        --json artifacts/catalog_terms_after.json \\
        --report docs/catalog_term_audit.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Make `tools` importable regardless of the caller's working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pipeline_common import strip_term_marker, term_matches  # noqa: E402

ROLES = ("deny_any", "require_any", "support_any")

# Observation keys inside a per_image entry. The three oldest runs
# (catalog_audit_126224899_full/_v2/_v3.json) wrote `gpt_observations` where
# every later run writes `cloud_observations`; including it raises the corpus
# 4,572 -> 4,879. The two-key default keeps the handoff's baseline reproducible.
DEFAULT_OBS_KEYS = ("local_observations", "cloud_observations")
LEGACY_OBS_KEY = "gpt_observations"

# 3- and 5-image smoke runs, not real corpus.
SMOKE_RUNS = frozenset({"catalog_audit_test.json", "catalog_audit_test_126224899.json"})

# A term is "short" — and so worth eyeballing even when the corpus never reaches
# it — below this length. Word-start anchoring fixed interior collisions but not
# prefix ones, and short stems are where prefix collisions live.
SHORT_TERM_LEN = 6

# Two different notions of "inside a word", deliberately kept apart.
#
# _is_boundary_char mirrors exactly what Python's \b treats as a word char, so
# it decides whether a match is interior/prefix. Hyphens are NOT word chars to
# \b, which is why "mold-like" and "wall-to-wall" still match `mold$`/`wall$` —
# counting them as excluded would misreport what a marker actually costs.
#
# _WORD_EXTRA only widens the *displayed* token so the report shows the whole
# readable word rather than a fragment.
_WORD_EXTRA = "-'"


def _is_boundary_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _is_word_char(ch: str) -> bool:
    return _is_boundary_char(ch) or ch in _WORD_EXTRA


def _norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def harvest_corpus(
    audit_dir: Path,
    *,
    include_legacy_key: bool = False,
    exclude_smoke: bool = False,
) -> Tuple[List[str], List[str]]:
    """Return (unique lowercased observations, source filenames used)."""
    keys = list(DEFAULT_OBS_KEYS)
    if include_legacy_key:
        keys.append(LEGACY_OBS_KEY)

    observations: set = set()
    used: List[str] = []
    for path in sorted(audit_dir.glob("catalog_audit_*.json")):
        if "checkpoint" in path.name:
            continue
        if exclude_smoke and path.name in SMOKE_RUNS:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        used.append(path.name)
        for image in data.get("per_image") or []:
            if not isinstance(image, dict):
                continue
            for key in keys:
                for obs in image.get(key) or []:
                    text = obs.get("description") if isinstance(obs, dict) else obs
                    if text:
                        observations.add(_norm(text))
    return sorted(observations), used


# ---------------------------------------------------------------------------
# Per-term measurement
# ---------------------------------------------------------------------------

@dataclass
class TermStats:
    term: str
    anchored: int = 0            # production behavior (term_matches, marker honored)
    stem: int = 0                # what the term would match without its marker
    whole_word: int = 0          # what a trailing \b would match
    substring: int = 0           # pre-anchoring behavior
    interior: Counter = field(default_factory=Counter)   # words anchoring stopped matching
    prefix: Counter = field(default_factory=Counter)     # longer words still matched

    @property
    def marked(self) -> bool:
        return self.term.endswith("$")

    @property
    def bare(self) -> str:
        return strip_term_marker(self.term)


def measure_term(term: str, texts: Sequence[str]) -> TermStats:
    """Count how `term` behaves across the corpus under all three regimes."""
    stats = TermStats(term=term)
    bare = strip_term_marker(term)
    if not bare:
        return stats
    whole_pattern = re.compile(r"\b" + re.escape(bare) + r"\b")

    for text in texts:
        matches = list(re.finditer(re.escape(bare), text))
        if matches:
            stats.substring += 1
        if whole_pattern.search(text):
            stats.whole_word += 1
        if term_matches(term, text):
            stats.anchored += 1
        if term_matches(bare, text):
            stats.stem += 1

        for match in matches:
            start, end = match.start(), match.end()
            left_is_word = start > 0 and _is_boundary_char(text[start - 1])
            right_is_word = end < len(text) and _is_boundary_char(text[end])
            if left_is_word:
                # Anchoring already killed this one; report what it used to hit.
                stats.interior[_enclosing_word(text, start, end)] += 1
            elif right_is_word:
                # Still fires — the trailing-edge class a marker can address.
                stats.prefix[_enclosing_word(text, start, end)] += 1
    return stats


def _enclosing_word(text: str, start: int, end: int) -> str:
    left = start
    while left > 0 and _is_word_char(text[left - 1]):
        left -= 1
    right = end
    while right < len(text) and _is_word_char(text[right]):
        right += 1
    return text[left:right]


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@dataclass
class TermRow:
    item_id: str
    role: str
    term: str
    stats: TermStats


@dataclass
class Audit:
    rows: List[TermRow]
    stats_by_term: Dict[str, TermStats]
    corpus_size: int
    corpus_files: List[str]
    item_count: int
    require_viability: List[Tuple[str, int, Dict[str, int]]]
    marked_terms: Dict[str, List[Tuple[str, str]]]


def build_audit(catalog_path: Path, texts: Sequence[str], corpus_files: List[str]) -> Audit:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    items = catalog.get("items") or []

    stats_by_term: Dict[str, TermStats] = {}

    def stats_for(term: str) -> TermStats:
        if term not in stats_by_term:
            stats_by_term[term] = measure_term(term, texts)
        return stats_by_term[term]

    rows: List[TermRow] = []
    require_viability: List[Tuple[str, int, Dict[str, int]]] = []
    marked_terms: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for item in items:
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        for role in ROLES:
            for raw in item.get(role) or []:
                term = _norm(raw)
                if not term:
                    continue
                stats = stats_for(term)
                rows.append(TermRow(item_id=item_id, role=role, term=term, stats=stats))
                if stats.marked:
                    marked_terms[term].append((item_id, role))

        require = [_norm(t) for t in (item.get("require_any") or []) if _norm(t)]
        if require:
            per_term = {t: stats_for(t).anchored for t in require}
            require_viability.append((item_id, max(per_term.values()), per_term))

    return Audit(
        rows=rows,
        stats_by_term=stats_by_term,
        corpus_size=len(texts),
        corpus_files=corpus_files,
        item_count=len(items),
        require_viability=sorted(require_viability, key=lambda r: r[1]),
        marked_terms=dict(marked_terms),
    )


def dead_require_items(audit: Audit) -> List[Tuple[str, int, Dict[str, int]]]:
    """Items whose entire require_any list scores zero — unretrievable."""
    return [row for row in audit.require_viability if row[1] == 0]


def zero_hit_rows(audit: Audit, *, role: Optional[str] = None, max_len: Optional[int] = None) -> List[TermRow]:
    out = []
    for row in audit.rows:
        if row.stats.anchored:
            continue
        if role and row.role != role:
            continue
        if max_len is not None and len(row.stats.bare) > max_len:
            continue
        out.append(row)
    return sorted(out, key=lambda r: (r.role, r.term, r.item_id))


def lost_to_anchoring(audit: Audit) -> List[TermRow]:
    """Unmarked terms that used to match by substring and now match nothing.

    Marked terms are excluded: their zeroing is the authored intent, reported in
    the whole-word section instead.
    """
    return sorted(
        (r for r in audit.rows
         if not r.stats.marked and r.stats.anchored == 0 and r.stats.substring > 0),
        key=lambda r: -r.stats.substring,
    )


def prefix_collisions(audit: Audit) -> List[TermRow]:
    """Rows where the term still fires inside a longer word."""
    rows = [r for r in audit.rows if any(w != r.stats.bare for w in r.stats.prefix)]
    return sorted(rows, key=lambda r: -sum(c for w, c in r.stats.prefix.items() if w != r.stats.bare))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def snapshot_dict(audit: Audit) -> Dict[str, Any]:
    return {
        "corpus_size": audit.corpus_size,
        "corpus_files": audit.corpus_files,
        "item_count": audit.item_count,
        "terms": {
            term: {
                "anchored": s.anchored,
                "stem": s.stem,
                "whole_word": s.whole_word,
                "substring": s.substring,
                "prefix": dict(s.prefix),
                "interior": dict(s.interior),
            }
            for term, s in sorted(audit.stats_by_term.items())
        },
        "require_viability": {item_id: best for item_id, best, _ in audit.require_viability},
    }


def diff_baseline(audit: Audit, baseline: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    old_terms = baseline.get("terms") or {}
    new_terms = snapshot_dict(audit)["terms"]

    added = sorted(set(new_terms) - set(old_terms))
    removed = sorted(set(old_terms) - set(new_terms))
    changed = [
        (t, old_terms[t]["anchored"], new_terms[t]["anchored"])
        for t in sorted(set(new_terms) & set(old_terms))
        if old_terms[t]["anchored"] != new_terms[t]["anchored"]
    ]
    out.append(f"- terms added: {len(added)}  removed: {len(removed)}  hit-count changed: {len(changed)}")
    for term in added:
        out.append(f"  + `{term}` -> {new_terms[term]['anchored']} hits")
    for term in removed:
        out.append(f"  - `{term}` (was {old_terms[term]['anchored']} hits)")
    for term, before, after in changed:
        out.append(f"  ~ `{term}`: {before} -> {after}")

    old_req = baseline.get("require_viability") or {}
    new_req = {item_id: best for item_id, best, _ in audit.require_viability}
    newly_dead = sorted(i for i, v in new_req.items() if v == 0 and old_req.get(i, 0) > 0)
    revived = sorted(i for i, v in new_req.items() if v > 0 and old_req.get(i, 1) == 0)
    if newly_dead:
        out.append(
            f"- **REVIEW** require_any newly unretrievable: {', '.join(newly_dead)} "
            f"— intended only if the term was matching nothing real"
        )
    if revived:
        out.append(f"- require_any revived: {', '.join(revived)}")
    if not newly_dead:
        out.append("- no require_any item became unretrievable")
    return out


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def _fmt_words(counter: Counter, bare: str, limit: int = 5) -> str:
    items = [(w, c) for w, c in counter.most_common() if w != bare][:limit]
    return ", ".join(f"`{w}`&nbsp;{c}" for w, c in items) or "-"


def render_markdown(audit: Audit, catalog_path: Path, baseline: Optional[Dict[str, Any]]) -> str:
    out: List[str] = []
    out.append("# Catalog keyword-term audit")
    out.append("")
    try:
        catalog_label = catalog_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        catalog_label = catalog_path.as_posix()
    out.append(
        f"Generated by `scripts/audit_catalog_terms.py` against `{catalog_label}` "
        f"and {len(audit.corpus_files)} audit runs "
        f"({audit.corpus_size:,} unique observations, {audit.item_count} catalog items)."
    )
    out.append("")
    out.append(
        "Matching is word-start anchored (`term_matches`, leading `\\b` only), so a term is a "
        "stem: `stain` covers `stained`/`stains`. A term authored with a trailing `$` opts in "
        "to a closing `\\b` as well."
    )
    out.append("")

    role_counts = Counter(r.role for r in audit.rows)
    out.append("## Summary")
    out.append("")
    out.extend(_table(
        ["metric", "value"],
        [
            ("term instances", sum(role_counts.values())),
            ("distinct terms", len(audit.stats_by_term)),
            *[(f"&nbsp;&nbsp;{role}", role_counts.get(role, 0)) for role in ROLES],
            ("whole-word marked terms", len(audit.marked_terms)),
            ("zero-hit term instances", len(zero_hit_rows(audit))),
            ("terms lost to anchoring", len(lost_to_anchoring(audit))),
            ("items with require_any", len(audit.require_viability)),
            ("**unretrievable items**", f"**{len(dead_require_items(audit))}**"),
        ],
    ))
    out.append("")

    if baseline:
        out.append("## Diff vs baseline")
        out.append("")
        out.extend(diff_baseline(audit, baseline))
        out.append("")

    out.append("## require_any viability")
    out.append("")
    out.append(
        "`require_any` is a hard gate: if no term matches, the item is dropped from retrieval "
        "entirely. Items are listed worst-first; anything at 0 is unretrievable against this "
        "corpus."
    )
    out.append("")
    out.extend(_table(
        ["item", "best term hits", "per-term"],
        [
            (item_id, best, ", ".join(f"`{t}`&nbsp;{n}" for t, n in sorted(per.items(), key=lambda kv: -kv[1])))
            for item_id, best, per in audit.require_viability
            if best < 10
        ],
    ))
    out.append("")

    marked = audit.marked_terms
    if marked:
        out.append("## Whole-word marked terms")
        out.append("")
        out.append(
            "Terms opted in to a trailing `\\b`. `as stem` is what the term would match without "
            "its marker — the difference is what the marker excluded. Verify each excluded word "
            "is a genuine collision, not a wanted inflection."
        )
        out.append("")
        out.extend(_table(
            ["term", "hits", "as stem", "obs excluded", "words excluded (occurrences)", "used by"],
            [
                (
                    f"`{term}`",
                    audit.stats_by_term[term].anchored,
                    audit.stats_by_term[term].stem,
                    audit.stats_by_term[term].stem - audit.stats_by_term[term].anchored,
                    _fmt_words(audit.stats_by_term[term].prefix, audit.stats_by_term[term].bare),
                    ", ".join(f"{i} ({r})" for i, r in uses),
                )
                for term, uses in sorted(marked.items())
            ],
        ))
        out.append("")

    out.append("## Prefix collisions (the trailing-edge class)")
    out.append("")
    out.append(
        "Anchoring fixed interior collisions (`ding` inside `siding`) but a term still matches "
        "a longer word it *prefixes*. Most are wanted inflections; review for the ones that are "
        "a different concept (`mold` inside `molding`)."
    )
    out.append("")
    out.extend(_table(
        ["term", "role", "item", "hits", "whole-word", "extends into"],
        [
            (
                f"`{r.term}`", r.role, r.item_id, r.stats.anchored, r.stats.whole_word,
                _fmt_words(r.stats.prefix, r.stats.bare),
            )
            for r in prefix_collisions(audit)[:40]
        ],
    ))
    out.append("")

    lost = lost_to_anchoring(audit)
    out.append("## Terms lost to anchoring")
    out.append("")
    if lost:
        out.append("These matched by raw substring and now match nothing — author the prefixed form.")
        out.append("")
        out.extend(_table(
            ["term", "role", "item", "was", "used to hit"],
            [
                (f"`{r.term}`", r.role, r.item_id, r.stats.substring, _fmt_words(r.stats.interior, r.stats.bare))
                for r in lost
            ],
        ))
    else:
        out.append("None — every term that used to match still matches.")
    out.append("")

    out.append("## Untested terms")
    out.append("")
    out.append(
        f"The corpus never reaches these, so they are unverified in both directions. Short terms "
        f"(<= {SHORT_TERM_LEN} chars) are listed in full because that is where prefix collisions live; "
        f"longer multi-word phrases carry near-zero collision risk under word-start anchoring."
    )
    out.append("")
    for role in ROLES:
        short = zero_hit_rows(audit, role=role, max_len=SHORT_TERM_LEN)
        total = len(zero_hit_rows(audit, role=role))
        out.append(f"### `{role}` — {total} zero-hit, {len(short)} short")
        out.append("")
        if short:
            out.extend(_table(
                ["term", "item"],
                [(f"`{r.term}`", r.item_id) for r in short],
            ))
        else:
            out.append("_No short zero-hit terms._")
        out.append("")

    return "\n".join(out)


def render_stdout(audit: Audit) -> str:
    dead = dead_require_items(audit)
    lost = lost_to_anchoring(audit)
    out = [
        f"corpus={audit.corpus_size:,} obs from {len(audit.corpus_files)} runs  "
        f"items={audit.item_count}  terms={len(audit.stats_by_term)} distinct "
        f"({len(audit.rows)} instances)",
        f"zero-hit instances={len(zero_hit_rows(audit))}  lost-to-anchoring={len(lost)}  "
        f"marked={len(audit.marked_terms)}",
        f"unretrievable require_any items={len(dead)}"
        + (": " + ", ".join(i for i, _, _ in dead) if dead else ""),
    ]
    for row in lost:
        out.append(f"  LOST  {row.role} {row.term!r} on {row.item_id} (was {row.stats.substring} hits)")
    return "\n".join(out)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit catalog keyword terms against the observation corpus.")
    parser.add_argument("--catalog", default=ROOT / "tools/issue_catalog.json", type=Path)
    parser.add_argument("--audit-dir", default=ROOT, type=Path,
                        help="Directory holding the catalog_audit_*.json runs.")
    parser.add_argument("--include-legacy-key", action="store_true",
                        help="Also read `gpt_observations` (three pre-rename runs); raises the corpus 4,572 -> 4,879.")
    parser.add_argument("--exclude-smoke", action="store_true",
                        help="Drop the 3- and 5-image catalog_audit_test*.json smoke runs (26 observations).")
    parser.add_argument("--report", default=None, type=Path, help="Write the markdown report here.")
    parser.add_argument("--json", dest="json_out", default=None, type=Path,
                        help="Write a machine-readable snapshot here (for --baseline diffs).")
    parser.add_argument("--baseline", default=None, type=Path,
                        help="A prior --json snapshot to diff against (before/after).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    texts, files = harvest_corpus(
        args.audit_dir,
        include_legacy_key=args.include_legacy_key,
        exclude_smoke=args.exclude_smoke,
    )
    if not texts:
        print(f"no observations harvested from {args.audit_dir}", file=sys.stderr)
        return 1

    audit = build_audit(args.catalog, texts, files)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None

    print(render_stdout(audit))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(snapshot_dict(audit), indent=2), encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(audit, args.catalog, baseline), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
