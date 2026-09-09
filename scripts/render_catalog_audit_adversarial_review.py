"""Session 3 adversarial-review artifact for the catalog audit: reconcile, validate, render.

Reads the hand-authored review half of reports/catalog_audit_adversarial_review.json
(one disposition per Session 2 proposal id, reopened clusters, cross-cutting findings,
questions for the human gate), verifies every pinned input it judges, resolves every
evidence reference against the Session 1 bundle / Session 2 proposals / catalog /
repository, fills the tool-owned `derived` half (pins, id reconciliation, counts,
reference resolution, validation) and renders the deterministic review document.

  default    reconcile + validate, write the JSON and the Markdown
  --check    same in memory; compare with what is on disk; write nothing

Nothing this tool reads is ever written: the proposals, the proposal document, the
evidence bundle, the photo review and packet, the decisions file and both catalogs are
hashed before and after every run and the run fails if any of them moved.

Run:
  .venv\\Scripts\\python.exe scripts\\render_catalog_audit_adversarial_review.py [--check]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import render_catalog_audit_proposals as s2  # noqa: E402
from tools.comparison_common import atomic_json, canonical_json, sha256_file  # noqa: E402

SCHEMA_VERSION = 1
REVIEW_JSON = ROOT / "reports" / "catalog_audit_adversarial_review.json"
PROPOSAL_MD = ROOT / "docs" / "PROPOSAL_catalog_audit_20260901.md"
PROPOSALS_SHA256 = "1411352c42e9d3a555c3a1561470323da2234447badc2869c9874c3e8fa3ef99"
PROPOSAL_MD_SHA256 = "771572e7c1d095281756b6baf9f28e65cffedc7f9b04cabfd559117768b17591"
PINS = {  # every input this review judges, as the Session 2 handoff pinned it
    s2._rel(s2.PROPOSALS_JSON): PROPOSALS_SHA256, s2._rel(PROPOSAL_MD): PROPOSAL_MD_SHA256,
    s2._rel(s2.EVIDENCE_JSON): s2.EVIDENCE_SHA256, s2._rel(s2.REVIEW_JSON): s2.REVIEW_SHA256,
    s2._rel(s2.PACKET_JSON): s2.PACKET_SHA256,
    s2._rel(s2.DECISIONS): "47614d822bdd6b672d8596050b46839d04a8a6df0a04c18e052f4a63bd1903e8",
    s2._rel(s2.V2_CATALOG): "51bf7e263ff98109d657ef730b0ce1a705598101f09843eb288b1bdc3e0eaa54",
    s2._rel(s2.V1_CATALOG): "4ba046a1a78337f1c8e47701a011ec16e700c296782ddedbe7bf52cf888314f2",
    s2._rel(s2.GENERATOR): "b732022fcea8cd22109c7781eb0f93d0688d557230c759ccb6216d667250a053",
    "tools/catalog_validation.py": "041637a3684bacd3bc06dd92955b845088b4fedda8bb7e2e31222ac9ec75bee2",
}
GUARDED = s2.GUARDED + s2.PINNED_INPUTS + (s2.PROPOSALS_JSON, PROPOSAL_MD)

DISPOSITIONS = ("sustained", "sustained_with_conditions", "reassign_non_catalog", "insufficient_evidence", "reject")
RISK_CATEGORIES = ("semantic", "evidence", "ownership", "expressibility", "economics", "retrieval", "policy", "provenance")
TEXT_FIELDS = ("challenge_summary", "strongest_counterargument", "resolution", "residual_risk")
DISPOSITION_KEYS = ("proposal_id", "session2_outcome", "disposition", "risk_categories", "challenge_summary",
                    "evidence_inspected", "strongest_counterargument", "resolution", "required_modification",
                    "conditions", "residual_risk", "reopen", "reopen_recommendation", "photos_opened")
REOPEN_KEYS = ("proposal_id", "from_outcome", "recommended_outcome", "summary")
FINDING_KEYS = ("id", "title", "detail", "affects", "refs")
QUESTION_KEYS = ("id", "proposal_ids", "question", "answers")
UNIT_PREFIXES = ("runtime:", "gold:", "lead:")
FILE_REF = re.compile(r"^([A-Za-z0-9_./\\-]+\.(?:py|md|json|jsonl|js|txt)):(\d+)(?:-(\d+))?$")
PHOTO_REF = re.compile(r"^photo:([A-Za-z0-9_]+)/(photo_\d{3}\.jpg)$")
CARD_REF = re.compile(r"^rc_[0-9a-f]{12}$")


class ReviewError(SystemExit):
    pass


def fail(msg: str) -> None:
    raise ReviewError(f"render_catalog_audit_adversarial_review: {msg}")


def cap_num(pid: str) -> int:
    return int(pid[len(s2.PROPOSAL_ID):]) if s2.is_cap_id(pid) else 10 ** 6


# --------------------------------------------------------------------------- inputs

def verify_pins() -> Dict[str, str]:
    actual, problems = {}, []
    for rel, expected in PINS.items():
        path = ROOT / rel
        actual[rel] = sha256_file(path) if path.is_file() else None
        if actual[rel] != expected:
            problems.append(f"{rel}: on disk {actual[rel]}, pinned {expected}")
    if problems:
        fail("pinned inputs drifted: " + "; ".join(problems))
    return actual


def git_head() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001 - git absent or not a checkout; the hash pins are the authority
        return None


class References:
    """Every review claim must point at something that exists: a proposal id, a queue case or
    review card, an evidence unit, a catalog item, a photo, or a repository file (with line range)."""

    def __init__(self, bundle: Dict[str, Any], proposals: Dict[str, Any], catalog: Dict[str, Any]):
        ev = s2.Evidence(bundle)
        self.proposal_ids = {c["proposal_id"] for c in proposals["clusters"]}
        self.cards = set(ev.cases) | set(ev.labels) | set(ev.notes)
        self.units = set(ev.units) | set(ev.gold_rows) | set(ev.leads)
        self.units |= {c.get("unit_key") for c in ev.cases.values() if c.get("unit_key")}
        self.units |= {l.get("unit_key") for l in ev.labels.values() if l.get("unit_key")}
        self.items = {it["id"] for it in catalog["items"]}
        for c in proposals["clusters"]:
            self.items |= set((c["judgment"].get("proposal") or {}).get("target_items") or [])
        self.photo_root = Path(proposals["review"]["photo_root"])

    def kind_of(self, ref: str) -> Optional[str]:
        if ref in self.proposal_ids:
            return "proposal"
        if CARD_REF.match(ref) and ref in self.cards:
            return "card"
        if ref.startswith(UNIT_PREFIXES) and ref in self.units:
            return "unit"
        if ref.startswith("item:") and ref[5:] in self.items:
            return "item"
        m = PHOTO_REF.match(ref)
        if m:
            return "photo" if (not self.photo_root.is_dir() or (self.photo_root / m.group(1) / m.group(2)).is_file()) else None
        m = FILE_REF.match(ref)
        if m:
            path = ROOT / m.group(1).replace("\\", "/")
            if not path.is_file():
                return None
            lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
            lo, hi = int(m.group(2)), int(m.group(3) or m.group(2))
            return "file_line" if 1 <= lo <= hi <= lines else None
        if "/" in ref and (ROOT / ref).is_file():
            return "file"
        return None


# --------------------------------------------------------------------------- derive + validate

def _str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def derive(review: Dict[str, Any], proposals: Dict[str, Any], refs: References, pins: Dict[str, str],
           known_ids: Sequence[str] = s2.KNOWN_IDS) -> Dict[str, Any]:
    errors: List[str] = []
    err = errors.append
    disps = review.get("dispositions") or []
    status = {c["proposal_id"]: c["status"] for c in proposals["clusters"]}
    expected = sorted(set(known_ids) | set(status), key=cap_num)
    seen = Counter(d.get("proposal_id") for d in disps)
    ids = {"expected": expected, "reviewed": sorted(seen, key=cap_num),
           "missing": [i for i in expected if i not in seen], "orphans": sorted(i for i in seen if i not in status),
           "duplicates": sorted(i for i, n in seen.items() if n > 1)}
    for key in ("missing", "orphans", "duplicates"):
        if ids[key]:
            err(f"{key} proposal ids: {ids[key]}")
    if [d.get("proposal_id") for d in disps] != sorted((d.get("proposal_id") for d in disps), key=cap_num):
        err("dispositions must be ordered by proposal id")
    resolution: Dict[str, Counter] = {"by_kind": Counter()}
    unresolved: List[str] = []
    reopen_ids = set()

    def check_refs(owner: str, values: Any) -> None:
        if not isinstance(values, list) or not values:
            err(f"{owner}: needs a non-empty list of evidence references")
            return
        for r in values:
            kind = refs.kind_of(r) if isinstance(r, str) else None
            if kind is None:
                unresolved.append(f"{owner}: {r}")
            else:
                resolution["by_kind"][kind] += 1

    for d in disps:
        pid = d.get("proposal_id")
        if set(d) != set(DISPOSITION_KEYS):
            err(f"{pid}: keys must be exactly {sorted(DISPOSITION_KEYS)}; got {sorted(d)}")
            continue
        if pid in status and d["session2_outcome"] != status[pid]:
            err(f"{pid}: session2_outcome {d['session2_outcome']!r} != proposals status {status[pid]!r}")
        if d["disposition"] not in DISPOSITIONS:
            err(f"{pid}: disposition {d['disposition']!r} not in {DISPOSITIONS}")
        if not d["risk_categories"] or any(r not in RISK_CATEGORIES for r in d["risk_categories"]):
            err(f"{pid}: risk_categories must be a non-empty subset of {RISK_CATEGORIES}")
        for f in TEXT_FIELDS:
            if not _str(d[f]):
                err(f"{pid}: {f} must be non-empty text")
        if d["disposition"] == "sustained":
            if d["required_modification"] is not None or d["conditions"]:
                err(f"{pid}: a sustained disposition carries no required_modification and no conditions")
        elif d["disposition"] == "sustained_with_conditions":
            if not d["conditions"] or not all(_str(c) for c in d["conditions"]):
                err(f"{pid}: sustained_with_conditions needs at least one condition")
        elif not _str(d["required_modification"]):
            err(f"{pid}: {d['disposition']} needs a required_modification")
        if not isinstance(d["reopen"], bool):
            err(f"{pid}: reopen must be a boolean")
        rec = d["reopen_recommendation"]
        if d["reopen"]:
            reopen_ids.add(pid)
            if not (isinstance(rec, dict) and rec.get("recommended_outcome") in s2.OUTCOMES and _str(rec.get("why"))
                    and isinstance(rec.get("evidence_needed"), list)):
                err(f"{pid}: reopen needs reopen_recommendation {{recommended_outcome in OUTCOMES, why, evidence_needed[]}}")
        elif rec is not None:
            err(f"{pid}: reopen_recommendation must be null unless reopen is true")
        check_refs(pid, d["evidence_inspected"])
        for ph in d["photos_opened"]:
            if refs.kind_of(ph) != "photo":
                unresolved.append(f"{pid}: photos_opened {ph}")
    reopened = review.get("reopened") or []
    listed = {r.get("proposal_id") for r in reopened if isinstance(r, dict)}
    if listed != reopen_ids:
        err(f"reopened list {sorted(listed, key=cap_num)} must equal the dispositions flagged reopen {sorted(reopen_ids, key=cap_num)}")
    for r in reopened:
        if set(r) != set(REOPEN_KEYS) or r.get("recommended_outcome") not in s2.OUTCOMES or not _str(r.get("summary")):
            err(f"reopened entry {r.get('proposal_id')}: keys {sorted(REOPEN_KEYS)}, recommended_outcome in OUTCOMES, summary text")
        elif r["proposal_id"] in status and r["from_outcome"] != status[r["proposal_id"]]:
            err(f"reopened {r['proposal_id']}: from_outcome must be the Session 2 status {status[r['proposal_id']]!r}")
    for f in review.get("cross_cutting_findings") or []:
        if set(f) != set(FINDING_KEYS) or not (_str(f.get("title")) and _str(f.get("detail"))):
            err(f"cross-cutting finding {f.get('id')}: keys {sorted(FINDING_KEYS)} with title and detail text")
            continue
        if any(a not in status for a in f["affects"]):
            err(f"cross-cutting finding {f['id']}: affects must name proposal ids; got {f['affects']}")
        check_refs(f"finding {f['id']}", f["refs"])
    for q in review.get("questions_for_human_gate") or []:
        ok = (set(q) == set(QUESTION_KEYS) and _str(q.get("question")) and isinstance(q.get("answers"), list) and q["answers"]
              and all(_str(a.get("answer")) and _str(a.get("implication")) for a in q["answers"] if isinstance(a, dict)))
        if not ok or any(p not in status for p in q.get("proposal_ids") or []):
            err(f"gate question {q.get('id')}: keys {sorted(QUESTION_KEYS)}, question text, answers[{{answer, implication}}], known proposal ids")
    for u in unresolved:
        err(f"unresolved reference {u}")
    counts = {"dispositions": len(disps), "by_disposition": dict(sorted(Counter(d.get("disposition") for d in disps).items())),
              "by_risk_category": dict(sorted(Counter(r for d in disps for r in d.get("risk_categories") or []).items())),
              "by_session2_outcome": {o: dict(sorted(Counter(d["disposition"] for d in disps if d.get("session2_outcome") == o).items()))
                                      for o in sorted({d.get("session2_outcome") for d in disps} - {None})},
              "reopened": len(reopen_ids), "cross_cutting_findings": len(review.get("cross_cutting_findings") or []),
              "questions_for_human_gate": len(review.get("questions_for_human_gate") or [])}
    return {"pins": pins, "git_head": git_head(), "ids": ids, "counts": counts,
            "references": {"resolved_by_kind": dict(sorted(resolution["by_kind"].items())), "unresolved": unresolved},
            "validation": {"ok": not errors, "errors": errors}}


def build(review_path: Path = REVIEW_JSON) -> Tuple[Dict[str, Any], str]:
    pins = verify_pins()
    if not review_path.is_file():
        fail(f"missing review file {review_path}")
    review = s2._read_json(review_path)
    if review.get("schema_version") != SCHEMA_VERSION or review.get("session") != 3:
        fail("review file must carry schema_version 1 and session 3")
    bundle = s2.load_bundle()
    proposals = s2._read_json(s2.PROPOSALS_JSON)
    catalog = s2._read_json(s2.V2_CATALOG)
    out = {k: v for k, v in review.items() if k != "derived"}
    out["derived"] = derive(review, proposals, References(bundle, proposals, catalog), pins)
    return out, render(out)


# --------------------------------------------------------------------------- rendering

_md, _table = s2._md, s2._table


def review_md_path(r: Dict[str, Any]) -> Path:
    return ROOT / "docs" / f"REVIEW_catalog_audit_adversarial_{r['review_date'].replace('-', '')}.md"


def render(r: Dict[str, Any]) -> str:
    d, L = r["derived"], []
    L += [f"# Catalog audit — Session 3 adversarial review ({r['review_date']})", "",
          f"Reviewer: `{r['reviewer']}`. Every Session 2 proposal id receives exactly one disposition; corrections are "
          "recommendations tied to proposal ids and nothing in the proposals, evidence, decisions, or catalogs was modified. "
          f"Rendered from `{s2._rel(REVIEW_JSON)}`.", "", "## Summary", ""]
    c = d["counts"]
    L += _table(["Disposition", "Count"], sorted(c["by_disposition"].items())) + [""]
    L += _table(["Session 2 outcome", "Dispositions"], [(o, ", ".join(f"{k} {n}" for k, n in v.items())) for o, v in c["by_session2_outcome"].items()]) + [""]
    L += _table(["Risk category", "Dispositions touching it"], sorted(c["by_risk_category"].items())) + [""]
    L += [f"Reopened clusters: {c['reopened']}. Cross-cutting findings: {c['cross_cutting_findings']}. "
          f"Questions for the human gate: {c['questions_for_human_gate']}. Validation ok: `{d['validation']['ok']}`.", "",
          "## Dispositions at a glance", ""]
    L += _table(["Id", "Session 2 outcome", "Disposition", "Risk", "Reopen", "Challenge"],
                [(x["proposal_id"], x["session2_outcome"], x["disposition"], x["risk_categories"], x["reopen"], x["challenge_summary"])
                 for x in r["dispositions"]])
    L += ["", "## Dispositions", ""]
    for x in r["dispositions"]:
        L += [f"### {x['proposal_id']} — `{x['disposition']}` (Session 2: `{x['session2_outcome']}`)", "",
              f"**Challenge.** {_md(x['challenge_summary'])}", "", f"**Strongest counterargument.** {_md(x['strongest_counterargument'])}", "",
              f"**Resolution.** {_md(x['resolution'])}", ""]
        if x["required_modification"]:
            L += [f"**Required modification.** {_md(x['required_modification'])}", ""]
        if x["conditions"]:
            L += ["**Conditions.**"] + [f"- {_md(cnd)}" for cnd in x["conditions"]] + [""]
        L += [f"**Residual risk.** {_md(x['residual_risk'])}", ""]
        if x["reopen"]:
            rec = x["reopen_recommendation"]
            L += [f"**Reopen recommendation.** `{rec['recommended_outcome']}` — {_md(rec['why'])} Evidence needed: {_md(rec['evidence_needed'])}", ""]
        L += [f"Risk categories: {_md(x['risk_categories'])}. Photos opened: {_md(x['photos_opened'])}.", "",
              "Evidence inspected: " + "; ".join(f"`{e}`" for e in x["evidence_inspected"]), ""]
    L += ["## Reopened clusters", ""]
    L += (_table(["Id", "From", "Recommended", "Summary"], [(y["proposal_id"], y["from_outcome"], y["recommended_outcome"], y["summary"]) for y in r["reopened"]])
          if r["reopened"] else ["None."])
    L += ["", "## Cross-cutting findings", ""]
    for f in r["cross_cutting_findings"]:
        L += [f"### {f['id']} — {_md(f['title'])}", "", _md(f["detail"]), "", f"Affects: {_md(f['affects'])}. Refs: " + "; ".join(f"`{e}`" for e in f["refs"]), ""]
    L += ["## Questions for the human gate", ""]
    for q in r["questions_for_human_gate"]:
        L += [f"### {q['id']} ({_md(q['proposal_ids'])})", "", _md(q["question"]), ""] + [f"- **{_md(a['answer'])}** → {_md(a['implication'])}" for a in q["answers"]] + [""]
    L += ["## Appendix — pins and reconciliation", ""]
    L += _table(["Input", "sha256"], sorted(d["pins"].items())) + ["", f"git HEAD: `{d['git_head']}`", ""]
    i = d["ids"]
    L += [f"Expected ids: {len(i['expected'])}; reviewed: {len(i['reviewed'])}; missing: {_md(i['missing'])}; orphans: {_md(i['orphans'])}; duplicates: {_md(i['duplicates'])}.", "",
          f"References resolved by kind: `{canonical_json(d['references']['resolved_by_kind'])}`; unresolved: {_md(d['references']['unresolved'])}.", "",
          f"Validation: ok=`{d['validation']['ok']}`" + ("" if not d["validation"]["errors"] else "; errors: " + "; ".join(_md(e) for e in d["validation"]["errors"])), ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- cli

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="reconcile in memory and compare with disk; write nothing")
    args = parser.parse_args(argv)
    before = s2.guard_snapshot(GUARDED)
    out, md = build()
    md_path, v = review_md_path(out), out["derived"]["validation"]
    if args.check:
        existing = s2._read_json(REVIEW_JSON)
        same_json = canonical_json(existing) == canonical_json(out)
        same_md = md_path.is_file() and md_path.read_text(encoding="utf-8") == md
        print(f"review json {'matches' if same_json else 'DIFFERS'}; markdown {'matches' if same_md else 'DIFFERS'}; "
              f"validation ok={v['ok']} errors={len(v['errors'])}")
        rc = 0 if (same_json and same_md and v["ok"]) else 1
    else:
        atomic_json(REVIEW_JSON, out)
        md_path.write_text(md, encoding="utf-8")
        print(f"wrote {REVIEW_JSON} and {md_path}; validation ok={v['ok']} errors={len(v['errors'])}")
        for e in v["errors"][:60]:
            print("  error:", e)
        rc = 0 if v["ok"] else 1
    after = s2.guard_snapshot(GUARDED)
    if after != before:
        fail("guarded files changed during the run: " + ", ".join(sorted(k for k in before if before[k] != after[k])))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
