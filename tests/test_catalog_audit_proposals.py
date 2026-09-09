"""Invariants of the catalog-audit proposal tooling (Session 2).

Synthetic bundle/proposal/review fixtures for the schema, evidence-bar, packet
and validation rules; the real v1 catalog + decisions file for the in-memory
decisions dry run (read only: the tests hash the guarded files before and
after). One guarded parity test compares a fresh build with the committed
proposal artifact when every input is present.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_catalog_audit_proposals.py -q
"""
import copy
import json
import re
from pathlib import Path

import pytest

from scripts import render_catalog_audit_proposals as mod
from tools.comparison_common import canonical_json, sha256_file

PROP_A, PROP_B, RUN = "redfin_1", "redfin_2", "20260817_000000_aaaaaaaa"
UNIT_A = f"runtime:canary:{PROP_A}:{RUN}:oc1_a"
UNIT_B = f"runtime:canary:{PROP_B}:{RUN}:oc1_b"
UNIT_LEAD = f"runtime:canary:{PROP_A}:{RUN}:oc1_c"
UNIT_GOLD = f"gold:{PROP_B}:photo_009.jpg:g1"
BLANK = {"disposition": None, "approved_diff": None, "conditions": [], "notes": ""}


# --------------------------------------------------------------------------- fixtures

def item(item_id, kind="degradation", trade="paint_drywall", scenes=("bedroom", "living_areas")):
    return {"id": item_id, "name": item_id, "kind": kind, "severity": 2, "trade_bucket": trade, "scope": "cosmetic",
            "tier": "work", "defaultHidden": False, "drop_if_generic": False, "category": "cosmetic",
            "display_class": "marketability", "scene_groups": list(scenes), "description": f"{item_id} desc",
            "embed_text": f"{item_id} embed", "support_any": ["paint"], "deny_any": [], "require_any": None,
            "route_override": None, "pricing_status": None, "cost": {"mode": "heuristic"}, "estimate": None,
            "work_item_code": "PAINT", "cost_model": None, "package_affinity": {}, "package_role": "ignore",
            "estimate_scope": None, "estimate_scope_reason": None,
            "atomic_claim": {"subject": item_id.replace("_", " "), "state": "worn", "ontology_basis": "wear"}}


def unit(key, method, prop, photos, cond, records, items, unit_type="runtime", independent=True):
    return {"unit_key": key, "unit_type": unit_type, "source": "canary", "property_key": prop, "run_id": RUN if unit_type == "runtime" else None,
            "condition_id": cond, "photo_key": photos[0] if unit_type == "gold" else None, "gold_id": "g1" if unit_type == "gold" else None,
            "photo_keys": list(photos), "records": [{"record_id": r, "record_type": t, "method": method, "role": "independent", "rules": ["R1"]}
                                                    for r, t in records],
            "primary_record_id": records[0][0], "primary_method": method, "rules": ["R1"], "implicated_items": list(items),
            "attached": bool(items), "independent": independent, "corroborates": None, "corroborated_by": [],
            "same_property_units": [], "same_photo_units": []}


def case(cid, prop, cond, item_id, photos, lane="miss_label", terra="unsupported", claim="exact", work="warranted"):
    return {"case_id": cid, "card_id": cid, "lane": lane, "property_key": prop, "run_id": RUN, "source": "canary",
            "condition_id": cond, "condition_item": item_id, "catalog_kind": "degradation", "claim_text": f"{item_id} claim",
            "photo_keys": list(photos), "scene_group": "bedroom", "estimate_unit_id": "bedroom_1", "accepted": terra == "supported",
            "disposition": "accepted_for_work", "reason_code": "route_work", "terra_verdict": terra, "terra_rationale": "because",
            "human_truth": {"claim": claim, "work": work, "class_v1_1": "x"}, "unit_key": f"runtime:canary:{prop}:{RUN}:{cond}",
            "attribution": {"attribution": "downstream", "stage": "terra", "confidence": "high"},
            "issues": [{"issue_id": f"iss_{cid}", "observation": "obs", "photo_key": photos[0], "p2b_join": "exact",
                        "p2e_status": "unknown", "resolution_path": "llm", "resolved_item_id": item_id, "shortcut_reason": None}]}


def world(photo_root: Path):
    items = {"paint_worn": item("paint_worn"), "paint_peeling": item("paint_peeling"),
             "wiring_exposed": item("wiring_exposed", kind="defect", trade="electrical")}
    for prop, keys in ((PROP_A, ("photo_001.jpg", "photo_002.jpg")), (PROP_B, ("photo_003.jpg", "photo_009.jpg"))):
        (photo_root / prop).mkdir(parents=True, exist_ok=True)
        for k in keys:
            (photo_root / prop / k).write_bytes(f"{prop}/{k}".encode())
    cases = {"rc_a": case("rc_a", PROP_A, "oc1_a", "paint_worn", ["photo_001.jpg"]),
             "rc_b": case("rc_b", PROP_B, "oc1_b", "paint_worn", ["photo_003.jpg"]),
             "rc_rej": case("rc_rej", PROP_B, "oc1_r", "paint_worn", ["photo_003.jpg"], lane="counted_correct_rejection", claim="absent", work="none"),
             "rc_hal": case("rc_hal", PROP_A, "oc1_h", "paint_worn", ["photo_002.jpg"], lane="halluc_label", terra="supported", claim="absent", work="none")}
    units = [unit(UNIT_A, "human_review", PROP_A, ["photo_001.jpg"], "oc1_a", [("rc_a", "case")], ["paint_worn"]),
             unit(UNIT_B, "human_review", PROP_B, ["photo_003.jpg"], "oc1_b", [("rc_b", "case")], ["paint_worn"]),
             unit(UNIT_LEAD, "model_judge", PROP_A, ["photo_002.jpg"], "oc1_c", [("lead:redfin_1:oc1_c", "factorized_lead")], ["paint_worn"]),
             unit(UNIT_GOLD, "gold_reference", PROP_B, ["photo_009.jpg"], None, [("ga_redfin_2_photo_009_g1", "gold_case")], [], unit_type="gold")]
    labels = {"rc_pos": {"card_id": "rc_pos", "case_id": None, "property_key": PROP_A, "run_id": RUN, "source": "canary",
                         "catalog_item_id": "paint_worn", "class_v1_1": "supported_billed", "claim": "exact", "work": "warranted",
                         "card_claim": "paint_worn claim", "card_photo_keys": ["photo_002.jpg"], "terra_verdict": "supported",
                         "unit_key": f"runtime:canary:{PROP_A}:{RUN}:oc1_p"}}
    return {
        "evidence_units": units,
        "indexes": {"cases": cases, "by_condition": {c["unit_key"]: cid for cid, c in cases.items()}},
        "gold": {"cases": {"ga_redfin_2_photo_009_g1": {"case_id": "ga_redfin_2_photo_009_g1", "finding": "stairs worn",
                                                        "matching_decision": "miss_candidate", "matching_note": "n",
                                                        "covering_item": None, "covering_item_source": None, "covering_case_id": None,
                                                        "covering_rejected_condition_id": None, "run_id": RUN,
                                                        "attribution": {"attribution": "pass_2a", "stage": None}}},
                 "matching_rows": []},
        "factorized_leads": [{"record_id": "lead:redfin_1:oc1_c", "observed_description": "peeling wall paint", "derived_class": "misnamed_but_warranted",
                              "stored_verdict": "supported", "catalog_item_id": "paint_worn", "item_source": "artifact_observed_conditions",
                              "unit_key": UNIT_LEAD, "property_key": PROP_A, "run_id": RUN, "source": "canary", "condition_id": "oc1_c"}],
        "candidate_enrichment": {"cases_and_leads": {"lead:redfin_1:oc1_c": {"iss_c": {"photo_key": "photo_002.jpg", "observation": "peeling",
                                                                                         "resolved_item_id": "paint_worn", "resolution_path": "llm",
                                                                                         "shortcut_reason": None, "routing_reason": "exact_kind",
                                                                                         "agrees_with_queue": None,
                                                                                         "candidates": [{"rank": 1, "item_id": "paint_worn", "score": 0.8}]}}},
                                 "gold_photos": {}},
        "labels": list(labels.values()), "review_notes": [],
        "worklist": [{"item_id": "paint_worn", "unit_keys": [UNIT_A, UNIT_B, UNIT_LEAD], "positive_uses": ["rc_pos"], "agreements": [],
                      "correct_rejections": ["rc_rej"], "notes": [], "hallucination_annotations": ["rc_hal"]}],
        "families": {"paint_worn": {"item_id": "paint_worn", "family_members": ["paint_peeling"], "frozen_candidate_neighbors": [
            {"item_id": "paint_peeling", "co_listed": 2, "best_rank": 2, "score_min": 0.5, "score_max": 0.6}],
            "migration": {"legacy_id": "paint_worn", "change_type": "unchanged"}}},
        "item_semantics": {"items": items},
        "baseline_comparison": {"changed_items": [{"item_id": "paint_worn", "changes": {"package_affinity": {"evidence_era": {}, "proposal_baseline": {}}}}]},
        "authoring_surface": {"economic_fields": ["cost", "estimate", "work_item_code", "cost_model", "package_affinity", "package_role",
                                                  "estimate_scope", "estimate_scope_reason"],
                              "inherited_fields": ["trade_bucket", "scope", "tier", "defaultHidden", "drop_if_generic", "category",
                                                   "display_class", "require_any", "deny_any", "scene_groups", "route_override"],
                              "successor_required_fields": ["id", "kind", "severity", "name", "description", "embed_text", "support_any"],
                              "wording_override_fields": ["deny_any", "description", "embed_text", "name", "support_any"]},
        "migration": {"change_type": {"paint_worn": "unchanged", "gone": "retired"}, "parent_of": {"paint_worn": "paint_worn"}},
        "lanes": {"product_quarantined_trades": ["electrical"]},
        "seeds": {"records": [{"record_id": "ga_redfin_2_photo_009_g1", "lane": "coverage_question", "unit_key": UNIT_GOLD}]},
        "run_artifacts": [],
        "catalog_identity": {"proposal_baseline": {"commit": "a9ed7ad"}, "evidence_era": {"commit": "07ee112", "sha256_crlf": "d1"}},
        "git": {"head": "9afe0fa"}, "sources": [],
    }


def cluster(pid="CAP-001", outcome=None, units=None, items=("paint_worn",), **judgment):
    units = units if units is not None else [{"unit_key": UNIT_A, "role": "support"}, {"unit_key": UNIT_B, "role": "support"},
                                             {"unit_key": UNIT_LEAD, "role": "support"}]
    native = outcome == "native_decisions_proposal"
    j = {"cluster": {"physical_subject": "wall paint", "visible_state_or_mechanism": "tired finish", "upstream_kinds": ["degradation"],
                     "scenes": ["bedroom"], "unsupported_commitment_types": ["mechanism"], "claim_under_test": "paint is worn",
                     "merge_split_rationale": "one mechanism",
                     "mismatch_class": (("operationally_consequential" if native else "operationally_equivalent") if outcome else None),
                     "equivalence_dimensions": ["billability"] if native else []},
         "implicated_items": list(items), "family": {"added": [], "removed": [], "rationale": None}, "units": units,
         "controls": {"excluded": [], "extra_required": []}, "coverage_gap": {"is_missing_coverage": False, "split_parent": None, "rationale": None},
         "outcome_type": outcome,
         "deferral_reason": "insufficient_evidence" if outcome == "deferred_insufficient_evidence" else None,
         "diagnosis": {"primary_cause": "catalog_wording_or_specificity", "contributing_causes": [], "catalog_ownership_argument": "arg",
                       "rejected_alternative_owners": [{"owner": "terra", "why": "no"}]} if outcome else
         {"primary_cause": None, "contributing_causes": [], "catalog_ownership_argument": None, "rejected_alternative_owners": []},
         "alternative_stage_analysis": {q: ("a" if outcome else None) for q in mod.STAGE_QUESTIONS},
         "proposal": {"ops": [], "expected_item_changes": {}, "semantic_before_after": None, "retrieval_effects": None, "economics_fit": None,
                      "package_route_effects": None, "product_policy": None, "expected_benefit": None, "regression_risk": None,
                      "structural_exception_demonstration": None, "gap_description": None},
         "evidence_bar_claim": {"bar_met_by": "two_independent_human" if outcome else None, "justification": "j" if outcome else None},
         "absence_statements": {"success_uses": "listed", "correct_rejections": "listed", "counterexamples": "listed"},
         "confidence": "medium" if outcome else None, "unresolved_questions": []}
    j.update(judgment)
    return {"proposal_id": pid, "title": "t", "status": outcome or "provisional", "judgment": j, "human_disposition": copy.deepcopy(BLANK)}


def proposals(photo_root: Path, clusters=None, triage=None):
    return {"schema_version": 1, "program": "catalog_audit", "session": 2, "proposal_date": "2026-09-01", "policy": {"p": 1},
            "review": {"photo_root": str(photo_root)}, "clusters": clusters if clusters is not None else [cluster()],
            "coverage_triage": triage if triage is not None else [
                {"unit_key": UNIT_GOLD, "judgment": {"triage_class": None, "ref": None, "note": None, "not_promoted_reason": None}}]}


def review_for(packet, adjud):
    rows = [{"row_id": r["row_id"], "unit_key": r["unit_key"], "adjudication": adjud.get(r["row_id"], "supports_claim"),
             "reason": "r", "what_image_shows": "w", "claim_commitments_supported": "s", "claim_commitments_not_supported": "n",
             "lineage_note": "l"} for r in packet["rows"] if r["priority"] == "required"]
    return {"schema_version": 1, "packet_sha256": None, "reviewer": "test", "rows": rows, "photos": {}}


@pytest.fixture
def w(tmp_path):
    root = tmp_path / "photos"
    bundle = world(root)
    return {"root": root, "bundle": bundle, "ev": mod.Evidence(bundle, {"rc_a": {"rationale": "why"}}),
            "baseline": {"decisions": {"entries": []}, "items": {}, "catalog": {"items": []}, "manifest": {"entries": []},
                         "hashes": {}, "v1": {}, "gen": None}}


def derived(w, props, review=None, known_ids=None):
    return mod.derive(props, w["ev"], w["baseline"], review, None, "fp", known_ids=known_ids)


def finalize(w, tmp_path, props, adjud=None, packet_props=None):
    """Write packet + results to tmp, load them, derive in final mode. `packet_props` builds the
    packet from a pre-reshape layout so bindings to another cluster's rows can be exercised."""
    packet = mod.build_packet(packet_props or props, w["ev"], w["root"])
    packet_path, results_path = tmp_path / "packet.json", tmp_path / "review.json"
    packet_path.write_text(json.dumps(packet, sort_keys=True), encoding="utf-8")
    res = review_for(packet, adjud or {})
    res["packet_sha256"] = sha256_file(packet_path)
    results_path.write_text(json.dumps(res), encoding="utf-8")
    review = mod.load_review(results_path, packet_path)
    return derived(w, props, review), packet


# --------------------------------------------------------------------------- schema and ids

def test_schema_required_keys(w):
    props = proposals(w["root"])
    del props["policy"]
    out = derived(w, props)
    assert any("missing top-level key policy" in e for e in out["derived"]["validation"]["errors"])
    for key in ("cluster", "units", "outcome_type", "coverage_gap", "absence_statements"):
        bad = proposals(w["root"])
        del bad["clusters"][0]["judgment"][key]
        assert any(f"judgment missing {key}" in e for e in derived(w, bad)["derived"]["validation"]["errors"])


def test_ids_unique_immutable_gaps_allowed_and_ordered(w):
    ok = derived(w, proposals(w["root"], [cluster("CAP-001"), cluster("CAP-007")]))
    assert not [e for e in ok["derived"]["validation"]["errors"] if "proposal id" in e or "order" in e or "duplicate" in e]
    errs = derived(w, proposals(w["root"], [cluster("CAP-002"), cluster("CAP-001")]))["derived"]["validation"]["errors"]
    assert any("proposal-id order" in e for e in errs)
    errs = derived(w, proposals(w["root"], [cluster("CAP-001"), cluster("CAP-001")]))["derived"]["validation"]["errors"]
    assert any("duplicate proposal id" in e for e in errs)
    errs = derived(w, proposals(w["root"], [cluster("P1")]))["derived"]["validation"]["errors"]
    assert any("must match CAP-NNN" in e for e in errs)


def test_references_resolve(w):
    props = proposals(w["root"], [cluster(items=("paint_worn", "nope"), units=[{"unit_key": "runtime:x", "role": "support"}])])
    with pytest.raises(KeyError):
        derived(w, props)  # derive itself needs real units; validate is reached only with resolvable keys
    props = proposals(w["root"], [cluster(items=("paint_worn", "nope"))])
    errs = derived(w, props)["derived"]["validation"]["errors"]
    assert any("implicated item nope" in e for e in errs)


def test_worklist_and_coverage_complete(w):
    errs = derived(w, proposals(w["root"], [cluster(items=("paint_peeling",))]))["derived"]["validation"]["errors"]
    assert any("worklist item paint_worn is in no cluster" in e for e in errs)
    out = derived(w, proposals(w["root"], triage=[]))
    assert [t["unit_key"] for t in out["coverage_triage"]] == [UNIT_GOLD]  # derive adds the pending row
    assert any("triage pending" in p for p in out["derived"]["validation"]["pending"])
    bad = proposals(w["root"], triage=[{"unit_key": UNIT_GOLD, "judgment": {"triage_class": "x"}},
                                       {"unit_key": UNIT_GOLD, "judgment": {"triage_class": "unclear"}}])
    errs = derived(w, bad)["derived"]["validation"]["errors"]
    assert any("appears 2 times" in e for e in errs) and any("triage class 'x' invalid" in e for e in errs)


def test_human_disposition_blank(w):
    props = proposals(w["root"])
    props["clusters"][0]["human_disposition"]["disposition"] = "approved"
    assert any("human_disposition must be blank" in e for e in derived(w, props)["derived"]["validation"]["errors"])


# --------------------------------------------------------------------------- outcomes and evidence bar

def test_outcome_exclusive_and_quarantine_forces_non_catalog(w, tmp_path):
    errs = derived(w, proposals(w["root"], [cluster(outcome="bogus")]))["derived"]["validation"]["errors"]
    assert any("outcome_type 'bogus' invalid" in e for e in errs)
    c = cluster(outcome="no_change")
    c["status"] = "provisional"
    assert any("status must mirror" in e for e in derived(w, proposals(w["root"], [c]))["derived"]["validation"]["errors"])
    q = cluster(outcome="no_change", items=("wiring_exposed", "paint_worn"))
    out, _ = finalize(w, tmp_path, proposals(w["root"], [q]))
    assert out["clusters"][0]["derived"]["trade_quarantined"] is True
    assert any("product-quarantined trade must be non_catalog_action" in e for e in out["derived"]["validation"]["errors"])


def test_evidence_bar_counts_only_adjudicated_units_and_lead_only_fails(w):
    c = cluster()
    ev = w["ev"]
    both = mod.evidence_bar(c, ev, {UNIT_A: "supports_claim", UNIT_B: "supports_claim", UNIT_LEAD: "supports_claim"})
    assert both["bar_met_by"] == "two_independent_human" and both["distinct_properties"] == 2 and both["lead_units"] == 1
    assert both["corroboration_type"] == "independent"  # lead is on a different photo and condition
    one = mod.evidence_bar(c, ev, {UNIT_A: "supports_claim", UNIT_B: "refuted", UNIT_LEAD: "unclear"})
    assert one["bar_met_by"] == "none" and one["human_units"] == 1 and one["contextual_units"] == {"refuted": 1, "unclear": 1}
    corr = mod.evidence_bar(c, ev, {UNIT_A: "supports_claim", UNIT_B: "refuted", UNIT_LEAD: "supports_claim"})
    assert corr["bar_met_by"] == "one_human_plus_independent_corroboration"
    lead_only = mod.evidence_bar(cluster(units=[{"unit_key": UNIT_LEAD, "role": "support"}]), ev, {UNIT_LEAD: "supports_claim"})
    assert lead_only["bar_met_by"] == "none" and lead_only["human_units"] == 0 and lead_only["corroboration_type"] == "none"
    pending = mod.evidence_bar(c, ev, {})
    assert pending["bar_met_by"] == "none" and pending["contextual_units"] == {"pending": 3}


def test_same_photo_lead_is_method_corroboration(w):
    c = cluster(units=[{"unit_key": UNIT_A, "role": "support"}, {"unit_key": UNIT_LEAD, "role": "support"}])
    w["ev"].units[UNIT_LEAD]["photo_keys"] = ["photo_001.jpg"]
    bar = mod.evidence_bar(c, w["ev"], {UNIT_A: "supports_claim", UNIT_LEAD: "supports_claim"})
    assert bar["corroboration_type"] == "method" and bar["bar_met_by"] == "none"


def test_non_deferred_outcomes_require_reviewed_support(w, tmp_path):
    # a reviewed human refutation grounds no_change; unclear-only and lead-only support still fail
    c = cluster(outcome="no_change", evidence_bar_claim={"bar_met_by": "none", "justification": "j"})
    refuted = {f"CAP-001:{UNIT_A}": "refuted", f"CAP-001:{UNIT_B}": "unclear", f"CAP-001:{UNIT_LEAD}": "refuted"}
    out, _ = finalize(w, tmp_path, proposals(w["root"], [c]), refuted)
    bar = out["clusters"][0]["derived"]["evidence_bar"]
    assert out["derived"]["validation"]["ok"], out["derived"]["validation"]["errors"]
    assert bar["reviewed_refutations"] == 1 and bar["lead_refutations"] == 1 and bar["mixed_signal"] is False
    out, _ = finalize(w, tmp_path, proposals(w["root"], [c]),
                      {f"CAP-001:{UNIT_A}": "unclear", f"CAP-001:{UNIT_B}": "unclear", f"CAP-001:{UNIT_LEAD}": "refuted"})
    errs = out["derived"]["validation"]["errors"]
    assert any("non-deferred outcome without a reviewed supporting unit" in e for e in errs)
    assert out["clusters"][0]["derived"]["evidence_bar"]["reviewed_refutations"] == 0
    lead_only = cluster(outcome="no_change", units=[{"unit_key": UNIT_LEAD, "role": "support"}],
                        evidence_bar_claim={"bar_met_by": "none", "justification": "j"})
    out, _ = finalize(w, tmp_path, proposals(w["root"], [lead_only]))
    assert any("non-deferred outcome without a reviewed supporting unit" in e for e in out["derived"]["validation"]["errors"])
    d = cluster(outcome="deferred_insufficient_evidence", evidence_bar_claim={"bar_met_by": "none", "justification": "j"})
    out, _ = finalize(w, tmp_path, proposals(w["root"], [d]), refuted)
    assert out["derived"]["validation"]["ok"], out["derived"]["validation"]["errors"]


def test_bar_claim_mismatch_and_bar_not_met_are_flagged(w, tmp_path):
    c = cluster(outcome="native_decisions_proposal")  # claims two_independent_human
    out, _ = finalize(w, tmp_path, proposals(w["root"], [c]), {f"CAP-001:{UNIT_B}": "refuted", f"CAP-001:{UNIT_LEAD}": "unclear"})
    errs = out["derived"]["validation"]["errors"]
    assert any("recomputed as none but claimed two_independent_human" in e for e in errs)
    partial, _ = finalize(w, tmp_path, proposals(w["root"], [c]), {f"CAP-001:{UNIT_B}": "refuted"})
    assert partial["clusters"][0]["derived"]["evidence_bar"]["bar_met_by"] == "one_human_plus_independent_corroboration"
    assert any("bar not met; outcome must be deferred or no_change" in e for e in errs)
    assert out["clusters"][0]["derived"]["evidence_bar"]["consistent_with_claim"] is False


def test_final_mode_valid_cluster_passes(w, tmp_path):
    c = cluster(outcome="no_change")
    out, _ = finalize(w, tmp_path, proposals(w["root"], [c]))
    assert out["derived"]["validation"]["mode"] == "final"
    assert out["derived"]["validation"]["ok"], out["derived"]["validation"]["errors"]
    u = {x["unit_key"]: x for x in out["clusters"][0]["derived"]["units"]}
    assert u[UNIT_A]["lineage"]["cases"][0]["attribution_rationale"] == "why"
    assert u[UNIT_A]["photo_review"]["status"] == "reviewed" and u[UNIT_A]["photos"]["status"] == "available"
    assert u[UNIT_LEAD]["lineage"]["leads"][0]["issues"][0]["candidates"][0]["item_id"] == "paint_worn"


# --------------------------------------------------------------------------- packet and review results

def test_packet_deterministic_and_covers_every_unit(w):
    props = proposals(w["root"])
    p1, p2 = mod.build_packet(props, w["ev"], w["root"]), mod.build_packet(props, w["ev"], w["root"])
    assert canonical_json(p1) == canonical_json(p2)
    rows = {r["row_id"]: r for r in p1["rows"]}
    for key in (UNIT_A, UNIT_B, UNIT_LEAD):
        assert rows[f"CAP-001:{key}"]["priority"] == "required"
    assert rows[f"coverage:{UNIT_GOLD}"]["claim_under_test"] == "stairs worn"
    ctl = {r["card_id"]: r for r in p1["rows"] if r["row_type"] == "control"}
    assert ctl["rc_pos"]["priority"] == "required" and ctl["rc_rej"]["priority"] == "required" and ctl["rc_hal"]["priority"] == "required"
    assert ctl["rc_hal"]["control_role"] == "hallucination"
    assert all(p["exists"] and p["sha256"] for p in p1["photos"].values())
    assert p1["counts"] == {"control:required": 3, "coverage:required": 1, "unit:required": 3}


def test_photo_refs_fallback_and_unavailable(w):
    w["ev"].cases["rc_a"]["photo_keys"] = []
    w["ev"].units[UNIT_A]["photo_keys"] = []
    refs = mod.photo_refs(w["ev"], UNIT_A, ["rc_a"], w["root"], {})
    assert refs["status"] == "unavailable" and refs["note"] == "run artifact not pinned by the queue"


def test_load_review_rejects_missing_or_mismatched_answers(w, tmp_path):
    packet = mod.build_packet(proposals(w["root"]), w["ev"], w["root"])
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet, sort_keys=True), encoding="utf-8")
    good = review_for(packet, {})
    good["packet_sha256"] = sha256_file(packet_path)
    res = tmp_path / "r.json"
    res.write_text(json.dumps(good), encoding="utf-8")
    loaded = mod.load_review(res, packet_path)
    assert loaded["packet_sha256"] == sha256_file(packet_path) and len(loaded["rows"]) == 7
    assert set(loaded["packet_rows"]) == {r["row_id"] for r in packet["rows"]}
    assert loaded["packet_rows"][f"CAP-001:{UNIT_A}"]["row_type"] == "unit"
    for mutate, msg in ((lambda r: r["rows"].pop(), "has no answer"),
                        (lambda r: r.update(packet_sha256="0" * 64), "packet on disk"),
                        (lambda r: r["rows"][0].update(adjudication="maybe"), "not in vocabulary"),
                        (lambda r: r["rows"].append(dict(r["rows"][0])), "answered 2 times"),
                        (lambda r: r["rows"][0].update(unit_key="runtime:other"), "unit_key mismatch")):
        bad = copy.deepcopy(good)
        mutate(bad)
        res.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(SystemExit, match=msg):
            mod.load_review(res, packet_path)


# --------------------------------------------------------------------------- decisions ops and dry run (real files)

@pytest.fixture(scope="module")
def real():
    gen = mod.load_generator()
    v1, decisions = json.loads(mod.V1_CATALOG.read_text(encoding="utf-8")), json.loads(mod.DECISIONS.read_text(encoding="utf-8"))
    catalog, manifest = gen.generate(copy.deepcopy(v1), copy.deepcopy(decisions))
    return {"gen": gen, "v1": v1, "decisions": decisions, "catalog": catalog, "manifest": manifest,
            "items": {it["id"]: it for it in catalog["items"]}, "order": [it["id"] for it in catalog["items"]], "hashes": {}}


def surface():
    return {"economic_fields": ["cost", "estimate", "work_item_code", "cost_model", "package_affinity", "package_role",
                                "estimate_scope", "estimate_scope_reason"],
            "inherited_fields": ["trade_bucket", "scope", "tier", "defaultHidden", "drop_if_generic", "category",
                                 "display_class", "require_any", "deny_any", "scene_groups", "route_override"],
            "successor_required_fields": ["id", "kind", "severity", "name", "description", "embed_text", "support_any"],
            "wording_override_fields": ["deny_any", "description", "embed_text", "name", "support_any"]}


def op(legacy_id, path, after, kind="set"):
    return {"legacy_id": legacy_id, "path": path, "op": kind, "after": after}


def test_apply_ops_and_before_values(real):
    ops = mod.fill_before(real["decisions"], [op("baseboard_wear_scuffs", "/successors/0/overrides/description", "new words"),
                                              op("baseboard_wear_scuffs", "/atomicity_rationale", "r")])
    assert ops[0]["before"] is None  # no overrides block on this carryover
    assert ops[1]["before"] == mod._entry(real["decisions"], "baseboard_wear_scuffs")["atomicity_rationale"]
    snapshot = canonical_json(real["decisions"])
    patched = mod.apply_ops(real["decisions"], ops)
    assert canonical_json(real["decisions"]) == snapshot
    assert mod._entry(patched, "baseboard_wear_scuffs")["successors"][0]["overrides"] == {"description": "new words"}
    removed = mod.apply_ops(patched, [op("baseboard_wear_scuffs", "/successors/0/overrides/description", None, "remove")])
    assert mod._entry(removed, "baseboard_wear_scuffs")["successors"][0]["overrides"] == {}
    appended = mod.apply_ops(real["decisions"], [op("baseboard_wear_scuffs", "/successors/1", {"id": "x"})])
    assert mod._entry(appended, "baseboard_wear_scuffs")["successors"][1] == {"id": "x"}
    with pytest.raises(SystemExit, match="unknown legacy id"):
        mod.apply_ops(real["decisions"], [op("nope", "/change_type", "split")])


def test_dry_run_only_declared_fields_differ_and_order(real):
    ops = mod.fill_before(real["decisions"], [op("baseboard_wear_scuffs", "/successors/0/overrides/description", "Baseboards look tired.")])
    run = mod.dry_run(real, ops, surface())
    assert run["ok"], run
    assert run["added"] == [] and run["removed"] == [] and run["order_ok"]
    assert list(run["item_diff"]) == ["baseboard_wear_scuffs"] and list(run["item_diff"]["baseboard_wear_scuffs"]) == ["description"]
    assert run["classification"][0]["class"] == "native" and run["metadata_problems"] == []
    assert run["catalog_validation"]["errors"] == [] and run["manifest_validation"]["errors"] == []


def test_dry_run_gap_captures_generator_error(real):
    # trade_bucket is forbidden on a carryover in BOTH arms: it is outside the
    # wording-only set the frozen surface describes and outside the widened
    # carryover set of the 2026-09-08 checkpoint. scene_groups is deliberately
    # NOT used here any more -- the checkpoint made it authorable, so it would
    # make this fixture arm-dependent.
    ops = mod.fill_before(real["decisions"], [op("baseboard_wear_scuffs", "/successors/0/overrides/trade_bucket", "paint_drywall")])
    run = mod.dry_run(real, ops, surface())
    assert run["ok"] is False
    assert re.search(r"non-(wording|authorable) fields", run["error"]), run["error"]
    assert run["classification"][0]["class"] == "gap"
    econ = mod.dry_run(real, mod.fill_before(real["decisions"], [op("baseboard_wear_scuffs", "/successors/0/overrides/estimate", {})]), surface())
    assert econ["classification"][0]["class"] == "gap" and "economic" in econ["classification"][0]["reason"]
    assert mod.classify_op(op("x", "/successors/0/name", "n"), "unchanged", surface())[0] == "gap"
    assert mod.classify_op(op("x", "/successors/0/kind", "defect"), "reclassified", surface())[0] == "native"
    assert mod.classify_op(op("x", "/successors/1/overrides/scene_groups", []), "split", surface())[0] == "native"
    assert mod.classify_op(op("x", "/deprecated", True), "split", surface())[0] == "native"


def test_dry_run_no_writes_and_v1_unmutated(real):
    before = {p: sha256_file(p) for p in mod.GUARDED}
    pinned_before = mod.guard_snapshot()
    v1_snapshot, dec_snapshot = canonical_json(real["v1"]), canonical_json(real["decisions"])
    mod.dry_run(real, mod.fill_before(real["decisions"], [op("baseboard_wear_scuffs", "/successors/0/overrides/name", "Baseboard wear")]), surface())
    assert {p: sha256_file(p) for p in mod.GUARDED} == before
    assert mod.guard_snapshot() == pinned_before and set(pinned_before) == {mod._rel(p) for p in mod.GUARDED + mod.PINNED_INPUTS}
    assert canonical_json(real["v1"]) == v1_snapshot and canonical_json(real["decisions"]) == dec_snapshot


def test_noop_diff_rejected(real):
    current = real["items"]["baseboard_wear_scuffs"]["description"]
    run = mod.dry_run(real, mod.fill_before(real["decisions"], [op("baseboard_wear_scuffs", "/successors/0/overrides/description", current)]), surface())
    assert run["ok"] is False and "change nothing" in run["error"]
    assert mod.dry_run(real, [], surface())["error"] == "no ops"


def test_entry_metadata_consistency(real):
    entry = mod._entry(real["decisions"], "baseboard_wear_scuffs")
    split = copy.deepcopy(entry)
    split.update(change_type="split", deprecated=False, requires_re_resolution=False)
    probs = mod.metadata_problems(entry, split, [op("baseboard_wear_scuffs", "/change_type", "split")])
    assert any("deprecated=true" in p for p in probs) and any("requires_re_resolution=true" in p for p in probs)
    assert any("refresh atomicity_rationale" in p for p in probs) and any("refresh expected_effects" in p for p in probs)
    rekind = copy.deepcopy(entry)
    rekind["successors"][0]["kind"] = "defect"
    rekind["requires_re_resolution"] = False
    probs = mod.metadata_problems(entry, rekind, [op("baseboard_wear_scuffs", "/successors/0/kind", "defect"),
                                                  op("baseboard_wear_scuffs", "/atomicity_rationale", "r"),
                                                  op("baseboard_wear_scuffs", "/expected_effects/routing", "e")])
    assert probs == ["a kind change requires requires_re_resolution=true"]
    retired = copy.deepcopy(entry)
    retired.update(change_type="retired", deprecated=True, requires_re_resolution=True)
    assert any("no successors" in p for p in mod.metadata_problems(entry, retired, [op("baseboard_wear_scuffs", "/change_type", "retired"),
                                                                                    op("baseboard_wear_scuffs", "/atomicity_rationale", "r"),
                                                                                    op("baseboard_wear_scuffs", "/expected_effects", {})]))
    wording = copy.deepcopy(entry)
    assert mod.metadata_problems(entry, wording, [op("baseboard_wear_scuffs", "/successors/0/overrides/description", "d")]) == []


# --------------------------------------------------------------------------- pins, bindings, refutations, ledger

def reshaped_props(w):
    """CAP-001 keeps A + LEAD; CAP-002 takes B, bound to CAP-001's packet row, with controls read from CAP-001."""
    base = proposals(w["root"])
    c1 = cluster("CAP-001", units=[{"unit_key": UNIT_A, "role": "support"}, {"unit_key": UNIT_LEAD, "role": "support"}])
    c2 = cluster("CAP-002", units=[{"unit_key": UNIT_B, "role": "support", "review_row_id": f"CAP-001:{UNIT_B}",
                                    "transfer_note": "reviewer text supports the narrower claim"}],
                 controls={"excluded": [], "extra_required": [], "review_source_cluster": "CAP-001"})
    c2["judgment"]["cluster"]["reshape"] = {"source_cluster_ids": ["CAP-001"], "claim_relation": "narrower", "rationale": "split by subject"}
    return base, proposals(w["root"], [c1, c2])


def test_pins_and_cli_refusals(w, tmp_path, monkeypatch):
    packet = mod.build_packet(proposals(w["root"]), w["ev"], w["root"])
    packet_path, res = tmp_path / "packet.json", tmp_path / "review.json"
    packet_path.write_text(json.dumps(packet, sort_keys=True), encoding="utf-8")
    good = review_for(packet, {})
    good["packet_sha256"] = sha256_file(packet_path)
    res.write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(SystemExit, match="drifted"):
        mod.verify_pins(res, packet_path)
    monkeypatch.setattr(mod, "REVIEW_SHA256", sha256_file(res))
    monkeypatch.setattr(mod, "PACKET_SHA256", sha256_file(packet_path))
    assert mod.verify_pins(res, packet_path) == {"review": sha256_file(res), "packet": sha256_file(packet_path)}
    assert mod.guard_snapshot([tmp_path / "nope.json"]) == {mod._rel(tmp_path / "nope.json"): None}
    monkeypatch.setattr(mod, "REVIEW_JSON", res)
    with pytest.raises(SystemExit, match="--packet refused"):
        mod.main(["--packet"])
    with pytest.raises(SystemExit, match="--review must point at the pinned review"):
        mod.main(["--review", str(tmp_path / "other.json")])


@pytest.mark.skipif(not mod.REVIEW_JSON.is_file(), reason="pinned review not present")
def test_build_fails_closed_on_pin_drift(monkeypatch):
    monkeypatch.setattr(mod, "REVIEW_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="drifted"):
        mod.build()


def test_reshape_binds_source_rows_and_keeps_adjudications(w, tmp_path):
    base, props = reshaped_props(w)
    out, _ = finalize(w, tmp_path, props, {f"CAP-001:{UNIT_B}": "refuted"}, packet_props=base)
    errs = out["derived"]["validation"]["errors"]
    assert not [e for e in errs if "bound" in e or "review row" in e or "reshape" in e or "transfer" in e or "controls" in e], errs
    c2 = out["clusters"][1]
    u = {x["unit_key"]: x for x in c2["derived"]["units"]}
    assert u[UNIT_B]["review_row_id"] == f"CAP-001:{UNIT_B}" and u[UNIT_B]["photo_review"]["adjudication"] == "refuted"
    ctl = {x["card_id"]: x for x in c2["derived"]["controls"]}
    assert set(ctl) == {"rc_pos", "rc_rej", "rc_hal"} and c2["derived"]["controls_dropped"] == []
    assert ctl["rc_pos"]["review_row_id"] == "CAP-001:control:rc_pos" and ctl["rc_pos"]["adjudication"] == "supports_claim"
    assert c2["derived"]["evidence_bar"]["reviewed_refutations"] == 1


def test_binding_rejections(w, tmp_path):
    base, props = reshaped_props(w)

    def errs_for(mutate):
        p2 = copy.deepcopy(props)
        mutate(p2)
        out, _ = finalize(w, tmp_path, p2, {}, packet_props=base)
        return out["derived"]["validation"]["errors"]

    def c2(p2):
        return p2["clusters"][1]["judgment"]

    assert any("does not list CAP-001 as a source" in e for e in errs_for(lambda p2: c2(p2)["cluster"].pop("reshape")))
    assert any("needs a transfer_note" in e for e in errs_for(lambda p2: c2(p2)["units"][0].pop("transfer_note")))
    assert any("does not carry unit" in e for e in errs_for(lambda p2: c2(p2)["units"][0].update(review_row_id="CAP-001:runtime:other")))
    assert any("is not in the pinned packet" in e for e in errs_for(
        lambda p2: (c2(p2)["units"][0].update(review_row_id=f"CAP-009:{UNIT_B}"),
                    c2(p2)["cluster"]["reshape"].update(source_cluster_ids=["CAP-001", "CAP-009"]))))
    assert any("unsupported keys" in e for e in errs_for(lambda p2: c2(p2)["units"][0].update(adjudication="supports_claim")))
    assert any("is bound by ['CAP-001', 'CAP-002']" in e for e in errs_for(
        lambda p2: p2["clusters"][0]["judgment"]["units"].append({"unit_key": UNIT_B, "role": "support"})))
    assert any("must be equivalent or narrower" in e for e in errs_for(lambda p2: c2(p2)["cluster"]["reshape"].update(claim_relation="wider")))
    assert any("is not a reshape source" in e for e in errs_for(lambda p2: c2(p2)["cluster"]["reshape"].update(source_cluster_ids=["CAP-003"])))
    assert any("no reviewed controls resolved under CAP-002, but the packet holds control rows for ['paint_worn']" in e
               for e in errs_for(lambda p2: c2(p2)["controls"].pop("review_source_cluster")))


def test_refutation_backed_outcomes_and_mixed_signal(w, tmp_path, monkeypatch):
    refuted = {f"CAP-001:{UNIT_A}": "refuted", f"CAP-001:{UNIT_B}": "unclear", f"CAP-001:{UNIT_LEAD}": "refuted"}
    nca = cluster(outcome="non_catalog_action", evidence_bar_claim={"bar_met_by": "none", "justification": "j"})
    out, _ = finalize(w, tmp_path, proposals(w["root"], [nca]), refuted)
    assert out["derived"]["validation"]["ok"], out["derived"]["validation"]["errors"]
    nat = cluster(outcome="native_decisions_proposal", evidence_bar_claim={"bar_met_by": "none", "justification": "j"})
    errs = finalize(w, tmp_path, proposals(w["root"], [nat]), refuted)[0]["derived"]["validation"]["errors"]
    assert any("bar not met" in e for e in errs) and any("without a reviewed supporting unit" in e for e in errs)
    mixed = cluster(outcome="no_change", evidence_bar_claim={"bar_met_by": "none", "justification": "j"})
    out, _ = finalize(w, tmp_path, proposals(w["root"], [mixed]), {f"CAP-001:{UNIT_B}": "refuted", f"CAP-001:{UNIT_LEAD}": "unclear"})
    assert out["clusters"][0]["derived"]["evidence_bar"]["mixed_signal"] is True
    assert any("needs evidence_bar_claim.conflict_note" in e for e in out["derived"]["validation"]["errors"])
    mixed["judgment"]["evidence_bar_claim"]["conflict_note"] = "A supports, B refutes; the claim is narrowed to A's subject"
    out, _ = finalize(w, tmp_path, proposals(w["root"], [mixed]), {f"CAP-001:{UNIT_B}": "refuted", f"CAP-001:{UNIT_LEAD}": "unclear"})
    assert out["derived"]["validation"]["ok"], out["derived"]["validation"]["errors"]
    monkeypatch.setattr(mod, "REFUTATION_BACKED", ())
    errs = finalize(w, tmp_path, proposals(w["root"], [nca]), refuted)[0]["derived"]["validation"]["errors"]
    assert any("bar not met" in e for e in errs)
    monkeypatch.undo()
    w["ev"].cases["rc_a"]["photo_keys"] = []
    w["ev"].units[UNIT_A]["photo_keys"] = []
    out, _ = finalize(w, tmp_path, proposals(w["root"], [nca]), refuted)
    assert out["clusters"][0]["derived"]["evidence_bar"]["reviewed_refutations"] == 0
    assert any("without a reviewed supporting unit" in e for e in out["derived"]["validation"]["errors"])


def test_structural_exception_guard(w, tmp_path):
    def se(demo=None, gap=False):
        c = cluster(outcome="no_change", units=[{"unit_key": UNIT_A, "role": "support"}],
                    evidence_bar_claim={"bar_met_by": "structural_exception", "justification": "j"})
        c["judgment"]["proposal"]["structural_exception_demonstration"] = demo if demo is not None else {
            "fact": "the name says dated while the atomic claim commits to plain trim", "item_ids": ["paint_worn"], "fields": ["atomic_claim", "name"]}
        if gap:
            c["judgment"]["coverage_gap"]["is_missing_coverage"] = True
        return c

    out, _ = finalize(w, tmp_path, proposals(w["root"], [se()]))
    assert out["clusters"][0]["derived"]["evidence_bar"]["bar_met_by"] == "structural_exception"
    assert out["derived"]["validation"]["ok"], out["derived"]["validation"]["errors"]
    out, _ = finalize(w, tmp_path, proposals(w["root"], [se(gap=True)]))
    assert out["clusters"][0]["derived"]["evidence_bar"]["bar_met_by"] == "none"
    assert any("may not be claimed for a missing-coverage cluster" in e for e in out["derived"]["validation"]["errors"])
    errs = finalize(w, tmp_path, proposals(w["root"], [se(demo="a string")]))[0]["derived"]["validation"]["errors"]
    assert any("must be {fact: str, item_ids: [..], fields: [..]}" in e for e in errs)
    errs = finalize(w, tmp_path, proposals(w["root"], [se(demo={"fact": "f", "item_ids": ["nope"], "fields": ["cost"]})]))[0]["derived"]["validation"]["errors"]
    assert any("item nope is not in the catalog" in e for e in errs) and any("field 'cost' is not a catalog semantic field" in e for e in errs)


def test_human_inputs_shape_and_render(w, tmp_path):
    c = cluster(outcome="no_change", human_inputs=[{"ref": "rc_b642fe69b86a", "source": "HANDOFF_SESSION_2.md:61",
                                                    "ruling": "work warranted", "effect": "catalog-owned wording mismatch"}])
    out, _ = finalize(w, tmp_path, proposals(w["root"], [c]))
    assert not [e for e in out["derived"]["validation"]["errors"] if "human_inputs" in e]
    md = "\n".join(mod.render_cluster(out["clusters"][0], True))
    assert "**Human inputs.**" in md and "rc_b642fe69b86a" in md
    for bad, msg in (({"x": 1}, "human_inputs must be a list"),
                     ([{"ref": "r", "source": "s", "ruling": "r"}], "human_inputs[0] must be"),
                     ([{"ref": "r", "source": "", "ruling": "r", "effect": "e"}], "human_inputs[0].source is empty")):
        errs = finalize(w, tmp_path, proposals(w["root"], [cluster(outcome="no_change", human_inputs=bad)]))[0]["derived"]["validation"]["errors"]
        assert any(msg in e for e in errs), (msg, errs)


def test_coverage_promotion_binds_coverage_rows_both_ways(w, tmp_path):
    base = proposals(w["root"])
    promoted = cluster("CAP-002", units=[{"unit_key": UNIT_GOLD, "role": "support", "review_row_id": f"coverage:{UNIT_GOLD}",
                                          "transfer_note": "gold finding text matches the gap claim"}])
    promoted["judgment"]["coverage_gap"] = {"is_missing_coverage": True, "split_parent": None, "rationale": "no item"}

    def props(triage, second=None):
        return proposals(w["root"], [cluster(), second or promoted], triage=[{"unit_key": UNIT_GOLD, "judgment": triage}])

    good = props({"triage_class": "attaches_to_cluster", "ref": "CAP-002", "note": None, "not_promoted_reason": None})
    out, _ = finalize(w, tmp_path, good, packet_props=base)
    errs = out["derived"]["validation"]["errors"]
    assert not [e for e in errs if "coverage" in e and ("bound" in e or "refs" in e or "packet" in e)], errs
    u = out["clusters"][1]["derived"]["units"][0]
    assert u["review_row_id"] == f"coverage:{UNIT_GOLD}" and u["photo_review"]["adjudication"] == "supports_claim"
    errs = finalize(w, tmp_path, props({"triage_class": "attaches_to_cluster", "ref": "CAP-001", "note": None, "not_promoted_reason": None}),
                    packet_props=base)[0]["derived"]["validation"]["errors"]
    assert any("bound by CAP-002 but triage is attaches_to_cluster ref CAP-001" in e for e in errs)
    assert any("triage refs CAP-001 but CAP-001 does not bind" in e for e in errs)
    errs = finalize(w, tmp_path, props({"triage_class": "covered_by_existing_item", "ref": "paint_worn", "note": None, "not_promoted_reason": None}),
                    packet_props=base)[0]["derived"]["validation"]["errors"]
    assert any("bound by CAP-002 but triage is covered_by_existing_item" in e for e in errs)
    plain = copy.deepcopy(good)
    plain["clusters"][1]["judgment"]["units"][0] = {"unit_key": UNIT_GOLD, "role": "support"}
    errs = finalize(w, tmp_path, plain, packet_props=base)[0]["derived"]["validation"]["errors"]
    assert any("is not in the pinned packet" in e for e in errs)
    gap = cluster("CAP-002", outcome="migration_system_gap", units=promoted["judgment"]["units"],
                  evidence_bar_claim={"bar_met_by": "none", "justification": "j"})
    gap["judgment"]["coverage_gap"] = {"is_missing_coverage": True, "split_parent": None, "rationale": "no item"}
    gap["judgment"]["proposal"]["gap_description"] = "no native add"
    errs = finalize(w, tmp_path, props({"triage_class": "coverage_gap_candidate", "ref": "CAP-002", "note": None, "not_promoted_reason": None}, gap),
                    packet_props=base)[0]["derived"]["validation"]["errors"]
    assert any("declared coverage gap needs a met evidence bar" in e for e in errs)


def test_id_ledger_blocks_disappearance_and_reuse(w):
    errs = derived(w, proposals(w["root"], [cluster("CAP-001")]), known_ids=("CAP-001", "CAP-002"))["derived"]["validation"]["errors"]
    assert any("ledger id CAP-002 is missing from clusters" in e for e in errs)
    errs = derived(w, proposals(w["root"], [cluster("CAP-001"), cluster("CAP-002"), cluster("CAP-005"), cluster("CAP-007")]),
                   known_ids=("CAP-001", "CAP-002", "CAP-007"))["derived"]["validation"]["errors"]
    assert any("CAP-005 is not in the id ledger and not above CAP-007" in e for e in errs)
    out = derived(w, proposals(w["root"], [cluster("CAP-001"), cluster("CAP-002"), cluster("CAP-007"), cluster("CAP-008")]),
                  known_ids=("CAP-001", "CAP-002", "CAP-007"))
    assert not [e for e in out["derived"]["validation"]["errors"] if "ledger" in e]
    assert out["derived"]["id_ledger"] == {"known_ids": ["CAP-001", "CAP-002", "CAP-007"], "max_known": "CAP-007", "new_ids": ["CAP-008"]}
    assert list(mod.KNOWN_IDS) == sorted(set(mod.KNOWN_IDS)) and all(mod.is_cap_id(k) for k in mod.KNOWN_IDS)
    assert mod.KNOWN_IDS[:17] == tuple(f"CAP-{n:03d}" for n in range(1, 18))


def test_mismatch_class_deferral_reason_and_native_prerequisites(w, tmp_path):
    def errs(c):
        return finalize(w, tmp_path, proposals(w["root"], [c]))[0]["derived"]["validation"]["errors"]

    c = cluster(outcome="no_change")
    c["judgment"]["cluster"]["mismatch_class"] = "bogus"
    assert any("mismatch_class 'bogus' invalid" in e for e in errs(c))
    c = cluster(outcome="no_change")
    c["judgment"]["cluster"]["equivalence_dimensions"] = ["repair"]
    assert any("non-empty exactly when" in e for e in errs(c))
    c = cluster(outcome="deferred_insufficient_evidence", evidence_bar_claim={"bar_met_by": "none", "justification": "j"})
    c["judgment"]["deferral_reason"] = None
    assert any("deferred outcome needs deferral_reason" in e for e in errs(c))
    c = cluster(outcome="no_change")
    c["judgment"]["deferral_reason"] = "lead_only"
    assert any("deferral_reason is only for deferred outcomes" in e for e in errs(c))
    nat = cluster(outcome="native_decisions_proposal")
    assert any("lacks regression_controls for ['rc_hal', 'rc_rej']" in e for e in errs(nat))
    nat["judgment"]["proposal"]["regression_controls"] = [{"card_id": "rc_hal", "expected_after": "still_rejected", "note": "n"},
                                                          {"card_id": "rc_rej", "expected_after": "still_rejected", "note": "n"},
                                                          {"card_id": "rc_pos", "expected_after": "still_supported", "note": "n"}]
    e = errs(nat)
    # a positive-use card is a legal regression_controls entry: saying what happens to a reviewed success is
    # exactly the regression question. Only cards unknown to the bundle are rejected.
    assert not [x for x in e if "lacks regression_controls" in x or "regression_controls[2]" in x]
    nat["judgment"]["proposal"]["regression_controls"].append(
        {"card_id": "rc_not_a_real_card", "expected_after": "still_rejected", "note": "n"})
    assert any("not a control of this cluster nor any reviewed card" in x for x in errs(nat))
    nat["judgment"]["proposal"]["regression_controls"].pop()
    nat["judgment"]["proposal"]["regression_controls"][0]["expected_after"] = "maybe"
    assert any("expected_after 'maybe' invalid" in x for x in errs(nat))
    nat["judgment"]["cluster"].update(mismatch_class="operationally_equivalent", equivalence_dimensions=[])
    assert any("needs mismatch_class operationally_consequential" in x for x in errs(nat))


def test_pass_2d_follow_ups_shape_and_registers(w, tmp_path):
    fu = [{"case_ref": "rc_a", "selected_item": "paint_worn", "better_item": "paint_peeling", "better_rank": 2,
           "confusion_family": "interior paint vs wall marks", "note": "reachable at rank 2"}]
    out, _ = finalize(w, tmp_path, proposals(w["root"], [cluster(outcome="no_change", pass_2d_follow_ups=fu)]))
    assert out["derived"]["validation"]["ok"], out["derived"]["validation"]["errors"]
    assert out["derived"]["pass_2d_follow_ups"] == {"interior paint vs wall marks": [{"proposal_id": "CAP-001", **fu[0]}]}
    reg = {(r["card_id"], r["role"]) for r in out["derived"]["regression_register"]}
    assert reg == {("rc_hal", "hallucination"), ("rc_rej", "correct_rejection")}
    md = mod.render(out)
    assert "## Appendix G" in md and "interior paint vs wall marks" in md and "## Appendix H" in md and "rc_hal" in md
    for bad, msg in (([{"case_ref": "rc_a"}], "must be {case_ref"),
                     ([{**fu[0], "better_item": "nope"}], "better_item 'nope' is not in the catalog"),
                     ([{**fu[0], "better_rank": 0}], "must be a positive integer"),
                     ([{**fu[0], "confusion_family": ""}], "needs case_ref and confusion_family")):
        errs = finalize(w, tmp_path, proposals(w["root"], [cluster(outcome="no_change", pass_2d_follow_ups=bad)]))[0]["derived"]["validation"]["errors"]
        assert any(msg in e for e in errs), (msg, errs)
    # a null rank means the better item was absent from the frozen candidate pool: legal with a note
    null_rank = [{**fu[0], "better_rank": None}]
    out, _ = finalize(w, tmp_path, proposals(w["root"], [cluster(outcome="no_change", pass_2d_follow_ups=null_rank)]))
    assert out["derived"]["validation"]["ok"], out["derived"]["validation"]["errors"]
    no_note = [{**fu[0], "better_rank": None, "note": ""}]
    errs = finalize(w, tmp_path, proposals(w["root"], [cluster(outcome="no_change", pass_2d_follow_ups=no_note)]))[0]["derived"]["validation"]["errors"]
    assert any("absent from the frozen candidate pool" in e for e in errs)


def test_render_surfaces_new_fields(w, tmp_path):
    base, props = reshaped_props(w)
    props["clusters"][1]["judgment"]["human_inputs"] = [{"ref": "rc_b", "source": "handoff", "ruling": "r", "effect": "e"}]
    out, _ = finalize(w, tmp_path, props, {f"CAP-001:{UNIT_B}": "refuted"}, packet_props=base)
    md = mod.render(out)
    for needle in ("**Reshape.** from CAP-001", f"CAP-001:{UNIT_B}", "reviewed refutations", "**Human inputs.**", "Id ledger",
                   "pins verified", "## Appendix G", "## Appendix H", "| refutations |", "rows from CAP-001"):
        assert needle in md, needle


# --------------------------------------------------------------------------- rendering and parity

def test_render_deterministic(w, tmp_path):
    out, _ = finalize(w, tmp_path, proposals(w["root"], [cluster(outcome="no_change")]))
    again, _ = finalize(w, tmp_path, proposals(w["root"], [cluster(outcome="no_change")]))
    assert canonical_json(out) == canonical_json(again)
    md = mod.render(out)
    assert md == mod.render(out) and "## Appendix F — validation matrix" in md and "CAP-001" in md


@pytest.mark.skipif(not (mod.EVIDENCE_JSON.is_file() and mod.PROPOSALS_JSON.is_file() and mod.LEDGER.is_file()),
                    reason="evidence bundle, committed proposals, or ledger not present")
def test_parity_committed_proposals():
    out, _, md = mod.build()
    committed = json.loads(mod.PROPOSALS_JSON.read_text(encoding="utf-8"))
    assert canonical_json(out) == canonical_json(committed)
    md_path = mod.proposal_md_path(out)
    assert md_path.is_file() and md_path.read_text(encoding="utf-8") == md


# ── current authoring surface (2026-09-08 catalog checkpoint) ────────────────

def test_current_surface_has_the_frozen_shape_and_real_provenance():
    """The current surface must be a drop-in for the frozen one: same keys, so
    classify_op consumes either without branching."""
    cur = mod.load_current_authoring_surface()
    # The helper above is the frozen surface minus target_version, which the real
    # bundle carries; every key classify_op reads must be present in the current one.
    assert set(surface()) <= set(cur)
    assert cur["target_version"] == "3.2"
    for key in ("economic_fields", "inherited_fields", "successor_required_fields"):
        assert cur[key] == surface()[key], key
    record = mod.current_surface_record()
    assert record["schema_version"] == mod.CURRENT_SURFACE_VERSION
    assert record["generator"]["sha256"] and record["generator"]["path"].endswith("migrate_catalog_kind_v2.py")
    # The frozen bundle is referenced, never recomputed or overwritten.
    assert record["frozen_surface_reference"]["sha256"] == mod.EVIDENCE_SHA256
    assert record["frozen_surface_reference"]["fingerprint"] == mod.EVIDENCE_FINGERPRINT
    assert record["surface"] == cur


def test_frozen_surface_is_never_widened_by_the_current_one():
    """Historical ops keep classifying against the frozen surface. Whatever the
    live generator accepts, the bundle's own record stays wording-only."""
    assert surface()["wording_override_fields"] == [
        "deny_any", "description", "embed_text", "name", "support_any"]
    cur = mod.load_current_authoring_surface()
    assert set(surface()["wording_override_fields"]) <= set(cur["wording_override_fields"])


def test_checkpoint_ops_classify_gap_frozen_and_native_when_authorable():
    """The two 2026-09-08 checkpoint ops. Under the frozen surface both are a
    system gap, which is why the generator support had to be authorized; under a
    surface that admits them they are native. Parameterised on the surface rather
    than on the arm, so this holds in either checkout."""
    trim = op("dated_interior_trim", "/successors/0/overrides/route_override", "no_action")
    wallpaper = op("dated_wallpaper_present", "/successors/0/overrides/scene_groups",
                   ["kitchen", "bedroom", "living_areas", "utility"])
    for candidate in (trim, wallpaper):
        klass, reason = mod.classify_op(candidate, "reclassified", surface())
        assert klass == "gap", (candidate["path"], reason)

    widened = surface()
    widened["wording_override_fields"] = sorted(
        set(widened["wording_override_fields"]) | {"route_override", "scene_groups"})
    for candidate in (trim, wallpaper):
        klass, reason = mod.classify_op(candidate, "reclassified", widened)
        assert klass == "native", (candidate["path"], reason)

    # Widening carryovers must not open economics or unknown keys.
    assert mod.classify_op(op("x", "/successors/0/overrides/cost", {}), "reclassified", widened)[0] == "gap"
    assert mod.classify_op(op("x", "/successors/0/overrides/require_any", []), "reclassified", widened)[0] == "gap"
    assert mod.classify_op(op("x", "/successors/0/overrides/trade_bucket", "y"), "reclassified", widened)[0] == "gap"
