"""Session 5 validation tooling: the parts whose correctness the reports depend on.

These are unit tests for the harness, not another copy of the validation. The Tier 1/2 results
live in reports/catalog_audit_validation_tier12.json; what is pinned here is the machinery that
produced them, because a comparator that silently mis-scores an arm would report a clean pass.

The module under test resolves an arm root from argv at import time, so importing it from the
repository root binds it to this checkout - which is what the tests want.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "catalog_audit_validation.py"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="Session 5 validation tooling not present")


def _load():
    spec = importlib.util.spec_from_file_location("catalog_audit_validation", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


mod = _load()


# --------------------------------------------------------------------------- pinned identities

def test_pinned_arm_identities_match_the_approved_handoff():
    """The commits and the approvals hash are the whole basis of the comparison."""
    assert mod.BASELINE_COMMIT == "9afe0fa5a0490a856abed507dccc4a02ed1de24c"
    assert mod.CANDIDATE_COMMIT == "6e67eaa83887cf4d218100dcd060c07d3bd30522"
    assert mod.APPROVALS_SHA256 == "a725c89af131e94b04867f4f5a226bda495278a8ba28fdd7162ed6188d76c871"
    assert mod.PARENT_ID == "dated_window_treatment_valance"
    assert {mod.BLINDS_ID, mod.FABRIC_ID} == {"window_blinds_basic_or_plain",
                                              "dated_window_valance_or_curtains"}


def test_top_k_is_the_production_value_not_the_retriever_default():
    """The retriever defaults to 5; production passes top_k_candidates=8 in the 2d context.
    Replaying at 5 would understate reachability on every row."""
    assert mod.TOP_K == 8


def test_provider_modules_are_refused():
    """The session is provider-free by construction, not by intention."""
    import sys
    sys.modules["tools.renovation_architecture.terra_review"] = object()  # type: ignore[assignment]
    try:
        with pytest.raises(mod.ValidationError):
            mod.assert_no_provider_imports()
    finally:
        del sys.modules["tools.renovation_architecture.terra_review"]


# --------------------------------------------------------------------------- gate semantics

def _item(**over):
    base = {"id": "x", "kind": "modernization", "scene_groups": ["bedroom", "living_areas"]}
    base.update(over)
    return base


def _row(text, kind="modernization", scene="bedroom"):
    return {"observation": text, "kind": kind, "scene_group": scene}


def test_gate_order_is_kind_then_scene_then_deny_then_require():
    """Deny runs before require, and both run after the kind and scene filters. The census and
    the leak test were computed in exactly this order, so a different order would silently
    disagree with the approved accounting."""
    item = _item(deny_any=["shower"], require_any=["curtain"])
    assert mod._gate_outcome(item, _row("the curtain is dated"))["eligible"] is True
    denied = mod._gate_outcome(item, _row("the shower curtain is dated"))
    assert denied["eligible"] is False and denied["blocked_by"] == "deny_any"
    missing = mod._gate_outcome(item, _row("the window treatment is dated"))
    assert missing["eligible"] is False and missing["blocked_by"] == "require_any"
    assert mod._gate_outcome(item, _row("the curtain is dated", kind="degradation"))["blocked_by"] == "kind"
    assert mod._gate_outcome(item, _row("the curtain is dated", scene="kitchen"))["blocked_by"] == "scene"


def test_an_item_without_scene_groups_is_eligible_everywhere():
    """tools/catalog_embeddings.py treats a missing scene_groups as all groups; a gate that
    reads it as none would drop candidates the runtime keeps."""
    item = _item(scene_groups=None)
    assert mod._gate_outcome(item, _row("anything", scene="exterior"))["eligible"] is True


def test_term_matching_is_word_start_anchored_not_substring():
    """'blind' must not fire inside 'blinding', and the approved deny stem 'shower' must fire
    on 'shower curtain' where the two-word phrase alone did not."""
    item = _item(deny_any=["shower"])
    assert mod._gate_outcome(item, _row("the shower rod is dated"))["blocked_by"] == "deny_any"
    assert mod._gate_outcome(_item(require_any=["blind"]), _row("blinds are old"))["eligible"] is True


def test_support_hits_report_the_matching_terms():
    item = _item(support_any=["curtain", "valance"])
    assert mod._support_hits(item, "the curtains and trim are dated") == ["curtain"]
    assert mod._support_hits(item, "nothing here") == []


# --------------------------------------------------------------------------- report invariants

REPORT = ROOT / "reports" / "catalog_audit_validation_tier12.json"
CORPUS = ROOT / "reports" / "catalog_audit_replay_corpus.json"


@pytest.mark.skipif(not REPORT.is_file(), reason="tier12 report not built")
def test_report_carries_an_explicit_verdict_for_every_rule():
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    assert data["tier2_rules"], "no rules evaluated"
    for rule in data["tier2_rules"]:
        assert rule["result"] in {"pass", "FAIL"}
    assert data["result"] in {"pass", "FAIL"}
    # a pass must be a pass in both tiers, never one carrying the other
    if data["result"] == "pass":
        assert data["tier1_result"] == "pass" and data["tier2_result"] == "pass"


@pytest.mark.skipif(not REPORT.is_file(), reason="tier12 report not built")
def test_score_drift_between_arms_stays_below_the_shortcut_thresholds():
    """The shared vector cache is what makes the two arms comparable. If drift ever approached
    the 0.03 margin gate, rank differences would stop being attributable to the split."""
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    stability = data["tier2_metrics"]["score_stability"]
    assert stability["max_delta"] < mod.SCORE_EPSILON
    assert stability["max_delta"] < 0.03 / 1000


@pytest.mark.skipif(not CORPUS.is_file(), reason="replay corpus not built")
def test_corpus_reproduces_the_approved_condition_census():
    """48 / 28 / 10 / 1 is the approval's own accounting. The corpus recomputes it from the
    pinned artifacts; a mismatch means the evidence moved, not that the code is wrong."""
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    rec = corpus["census_reconciliation"]
    assert rec["parent_conditions_in_artifacts"] == rec["redraft_conditions_total"] == 48
    found = rec["census_ids_found_in_artifacts"]
    assert found["condition_ids_moving"] == 28
    assert found["condition_ids_surviving"] == 10
    assert found["condition_ids_neither"] == 1


@pytest.mark.skipif(not CORPUS.is_file(), reason="replay corpus not built")
def test_frozen_shortcut_selftest_has_no_mismatches():
    """Today's shortcut function must reproduce what the evidence-era runs recorded. If it does
    not, the thresholds or negation patterns have moved and no replay is comparable."""
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    selftest = corpus["frozen_shortcut_selftest"]
    assert selftest["result"] == "pass"
    assert selftest["mismatches"] == []
    assert selftest["rows_considered"] > 3000


@pytest.mark.skipif(not CORPUS.is_file(), reason="replay corpus not built")
def test_every_named_case_resolved_to_a_replay_row():
    """A control that silently fails to resolve would drop out of the comparison unnoticed."""
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    unresolved = sorted(k for k, v in corpus["named_cases"].items() if v.get("status") != "resolved")
    assert unresolved == []
