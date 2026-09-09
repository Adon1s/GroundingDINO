"""Invariants of the catalog-audit adversarial-review tooling (Session 3).

Synthetic dispositions and a stub reference resolver for the reconciliation rules
(one disposition per proposal id, vocabulary, reference resolution, reopen
consistency, deterministic rendering); the real pinned artifacts for the
no-write proof and the parity test, both skipped when an input is absent.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_catalog_audit_adversarial_review.py -q
"""
import copy
import json
from pathlib import Path

import pytest

from scripts import render_catalog_audit_adversarial_review as mod
from scripts import render_catalog_audit_proposals as s2
from tools.comparison_common import canonical_json, sha256_file

IDS = ("CAP-001", "CAP-002", "CAP-003")
STATUS = {"CAP-001": "no_change", "CAP-002": "native_decisions_proposal", "CAP-003": "deferred_insufficient_evidence"}
PINS = {k: "0" * 64 for k in mod.PINS}


# --------------------------------------------------------------------------- fixtures

class StubRefs:
    """Resolver over a fixed universe; file/photo references go through the real matcher."""

    def __init__(self, tmp_path):
        self.proposal_ids, self.cards = set(IDS), {"rc_000000000001"}
        self.units, self.items, self.photo_root = {"gold:redfin_1:photo_001.jpg:g1"}, {"item_a"}, tmp_path / "no_photos"
        self.file = tmp_path / "src.py"
        self.file.write_text("a\nb\nc\n", encoding="utf-8")

    def kind_of(self, ref):
        if ref in self.proposal_ids:
            return "proposal"
        if ref in self.cards:
            return "card"
        if ref in self.units:
            return "unit"
        if ref.startswith("item:") and ref[5:] in self.items:
            return "item"
        if mod.PHOTO_REF.match(ref):
            return "photo"
        m = mod.FILE_REF.match(ref)
        if m and Path(m.group(1)).name == "src.py":
            lo, hi = int(m.group(2)), int(m.group(3) or m.group(2))
            return "file_line" if 1 <= lo <= hi <= 3 else None
        return None


def proposals():
    return {"clusters": [{"proposal_id": i, "status": STATUS[i], "judgment": {"proposal": {"target_items": []}}} for i in IDS],
            "review": {"photo_root": "C:/nowhere"}}


def disposition(pid, disp="sustained", **over):
    d = {"proposal_id": pid, "session2_outcome": STATUS[pid], "disposition": disp, "risk_categories": ["evidence"],
         "challenge_summary": "challenged", "evidence_inspected": ["rc_000000000001", "item:item_a", "src.py:2-3"],
         "strongest_counterargument": "counter", "resolution": "resolved", "required_modification": None,
         "conditions": [], "residual_risk": "low", "reopen": False, "reopen_recommendation": None, "photos_opened": []}
    d.update(over)
    return d


def review(*disps):
    return {"schema_version": 1, "program": "catalog_audit", "session": 3, "review_date": "2026-09-04", "reviewer": "t",
            "dispositions": list(disps) or [disposition(i) for i in IDS], "reopened": [], "cross_cutting_findings": [],
            "questions_for_human_gate": []}


@pytest.fixture
def refs(tmp_path):
    return StubRefs(tmp_path)


def derived(r, refs):
    return mod.derive(r, proposals(), refs, PINS, known_ids=IDS)


# --------------------------------------------------------------------------- reconciliation rules

def test_clean_review_validates_and_counts(refs):
    d = derived(review(), refs)
    assert d["validation"] == {"ok": True, "errors": []}
    assert d["ids"]["missing"] == [] and d["ids"]["orphans"] == [] and d["ids"]["duplicates"] == []
    assert d["counts"]["by_disposition"] == {"sustained": 3} and d["counts"]["by_risk_category"] == {"evidence": 3}
    assert d["references"]["resolved_by_kind"] == {"card": 3, "file_line": 3, "item": 3} and d["references"]["unresolved"] == []


def test_exactly_one_disposition_per_id(refs):
    d = derived(review(disposition("CAP-001"), disposition("CAP-001")), refs)
    assert d["ids"]["missing"] == ["CAP-002", "CAP-003"] and d["ids"]["duplicates"] == ["CAP-001"]
    r = review(); r["dispositions"].append(dict(disposition("CAP-001"), proposal_id="CAP-099", session2_outcome="no_change"))
    assert derived(r, refs)["ids"]["orphans"] == ["CAP-099"]
    r = review(disposition("CAP-002"), disposition("CAP-001"), disposition("CAP-003"))
    assert any("ordered" in e for e in derived(r, refs)["validation"]["errors"])


def test_vocabulary_and_disposition_shape(refs):
    bad = [disposition("CAP-001", "sustained", conditions=["x"]),                      # sustained carries no conditions
           disposition("CAP-002", "sustained_with_conditions"),                         # needs a condition
           disposition("CAP-003", "insufficient_evidence")]                             # needs a modification
    errors = derived(review(*bad), refs)["validation"]["errors"]
    assert sum("CAP-001" in e for e in errors) == 1 and any("CAP-002" in e for e in errors) and any("CAP-003" in e for e in errors)
    errors = derived(review(disposition("CAP-001", "maybe"), disposition("CAP-002", risk_categories=["vibes"]),
                            disposition("CAP-003", session2_outcome="no_change")), refs)["validation"]["errors"]
    assert any("not in" in e for e in errors) and any("risk_categories" in e for e in errors) and any("session2_outcome" in e for e in errors)
    extra = disposition("CAP-001"); extra["surprise"] = 1
    assert any("keys must be exactly" in e for e in derived(review(extra, disposition("CAP-002"), disposition("CAP-003")), refs)["validation"]["errors"])


def test_references_must_resolve(refs):
    r = review(disposition("CAP-001", evidence_inspected=["rc_deadbeefcafe", "src.py:9", "item:nope", "photo:redfin_1/photo_001.jpg"]),
               disposition("CAP-002"), disposition("CAP-003"))
    d = derived(r, refs)
    assert len(d["references"]["unresolved"]) == 3 and d["references"]["resolved_by_kind"]["photo"] == 1
    assert sum("unresolved reference" in e for e in d["validation"]["errors"]) == 3
    r = review(disposition("CAP-001", evidence_inspected=[]), disposition("CAP-002"), disposition("CAP-003"))
    assert any("non-empty list" in e for e in derived(r, refs)["validation"]["errors"])


def test_reopen_consistency(refs):
    rec = {"recommended_outcome": "deferred_insufficient_evidence", "why": "one property", "evidence_needed": ["a second property"]}
    r = review(disposition("CAP-001", "insufficient_evidence", required_modification="relabel", reopen=True, reopen_recommendation=rec),
               disposition("CAP-002"), disposition("CAP-003"))
    assert any("reopened list" in e for e in derived(r, refs)["validation"]["errors"])
    r["reopened"] = [{"proposal_id": "CAP-001", "from_outcome": "no_change", "recommended_outcome": "deferred_insufficient_evidence", "summary": "s"}]
    d = derived(r, refs)
    assert d["validation"]["ok"] and d["counts"]["reopened"] == 1
    r["reopened"][0]["from_outcome"] = "native_decisions_proposal"
    assert any("from_outcome" in e for e in derived(r, refs)["validation"]["errors"])
    r["reopened"][0]["from_outcome"] = "no_change"
    r["dispositions"][0]["reopen_recommendation"] = None
    assert any("reopen needs" in e for e in derived(r, refs)["validation"]["errors"])
    r2 = review(); r2["dispositions"][0]["reopen_recommendation"] = rec
    assert any("must be null" in e for e in derived(r2, refs)["validation"]["errors"])


def test_findings_and_questions_shape(refs):
    r = review()
    r["cross_cutting_findings"] = [{"id": "CCF-1", "title": "t", "detail": "d", "affects": ["CAP-001", "CAP-404"], "refs": ["rc_000000000001"]}]
    r["questions_for_human_gate"] = [{"id": "Q-1", "proposal_ids": ["CAP-002"], "question": "q?", "answers": [{"answer": "yes", "implication": "x"}]}]
    errors = derived(r, refs)["validation"]["errors"]
    assert len(errors) == 1 and "affects" in errors[0]
    r["cross_cutting_findings"][0]["affects"] = ["CAP-001"]
    r["questions_for_human_gate"][0]["answers"] = []
    assert any("gate question" in e for e in derived(r, refs)["validation"]["errors"])


def test_render_deterministic_and_surfaces_fields(refs):
    r = review(disposition("CAP-001", "sustained_with_conditions", conditions=["keep the glazing denies"]), disposition("CAP-002"), disposition("CAP-003"))
    r["cross_cutting_findings"] = [{"id": "CCF-1", "title": "missing brief", "detail": "not in repo", "affects": ["CAP-001"], "refs": ["rc_000000000001"]}]
    r["questions_for_human_gate"] = [{"id": "Q-1", "proposal_ids": ["CAP-002"], "question": "blinds?", "answers": [{"answer": "all", "implication": "as drafted"}]}]
    out = dict(r, derived=derived(r, refs))
    md = mod.render(out)
    assert md == mod.render(copy.deepcopy(out))
    for needle in ("keep the glazing denies", "CCF-1", "missing brief", "Q-1", "blinds?", "sustained_with_conditions", "Validation ok: `True`"):
        assert needle in md
    assert mod.review_md_path(out).name == "REVIEW_catalog_audit_adversarial_20260904.md"


# --------------------------------------------------------------------------- real artifacts (guarded)

REAL = all(p.is_file() for p in mod.GUARDED) and mod.REVIEW_JSON.is_file()


@pytest.mark.skipif(not REAL, reason="pinned inputs or the committed review not present")
def test_build_is_read_only_and_pins_verify():
    before = {s2._rel(p): sha256_file(p) for p in mod.GUARDED}
    assert mod.verify_pins() == {k: v for k, v in mod.PINS.items()}
    out, md = mod.build()
    assert out["derived"]["validation"]["ok"], out["derived"]["validation"]["errors"]
    assert {s2._rel(p): sha256_file(p) for p in mod.GUARDED} == before


@pytest.mark.skipif(not REAL, reason="pinned inputs or the committed review not present")
def test_parity_committed_review():
    out, md = mod.build()
    committed = json.loads(mod.REVIEW_JSON.read_text(encoding="utf-8"))
    assert canonical_json(out) == canonical_json(committed)
    md_path = mod.review_md_path(out)
    assert md_path.is_file() and md_path.read_text(encoding="utf-8") == md
    assert out["derived"]["ids"]["missing"] == [] and out["derived"]["ids"]["orphans"] == []
    assert set(out["derived"]["ids"]["reviewed"]) == set(s2.KNOWN_IDS) and len(out["dispositions"]) == len(s2.KNOWN_IDS)
