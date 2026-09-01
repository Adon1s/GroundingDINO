"""Joins, dedup, and determinism of the catalog-audit evidence builder.

Synthetic inputs throughout, mirroring the frozen files' shapes: the point is
that every join is run-scoped, every seed is reconciled exactly once, and two
builds of the same inputs are the same bytes. One guarded parity test compares
a fresh build against the committed bundle when the frozen inputs are present.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_catalog_audit_evidence.py -q
"""
import copy
import json
from pathlib import Path

import pytest

from scripts import build_catalog_audit_evidence as mod
from scripts import error_attribution_report as ear
from tools.comparison_common import sha256_bytes

PROP, RUN, CID, ISSUE, PHOTO = "redfin_1", "20260817_000000_aaaaaaaa", "oc1_a", "iss_a", "photo_001.jpg"


# --------------------------------------------------------------------------- fixtures

def catalog_item(item_id, *, kind="defect", trade="flooring", scenes=("kitchen", "bedroom"),
                 wic="FLOORING_REPLACE", claim=("vinyl flooring", "torn or lifting"), embed="vinyl torn embed",
                 affinity=None, pricing_status=None):
    item = {"id": item_id, "name": item_id.replace("_", " "), "kind": kind, "severity": 2,
            "trade_bucket": trade, "scope": "repair", "tier": "work", "defaultHidden": False,
            "scene_groups": list(scenes), "description": f"{item_id} description", "embed_text": embed,
            "support_any": ["vinyl"], "deny_any": [],
            "atomic_claim": {"subject": claim[0], "state": claim[1], "ontology_basis": "asserted_failure"},
            "work_item_code": wic,
            "estimate": {"estimate_tier": "medium", "strategy": "repair_only", "group": "flooring",
                         "stack_behavior": "sum", "unit_policy": "per_room"},
            "package_affinity": affinity if affinity is not None else
            {"kitchen": {"package_type": "kitchen_modernization", "package_role": "package_support"}}}
    if pricing_status:
        item["pricing_status"] = pricing_status
    return item


def catalog(items, version="3.2"):
    return {"version": version, "ontology_version": "observation-kind-v2", "publication_status": "publishable",
            "trade_buckets": [{"id": "flooring", "name": "Flooring"},
                              {"id": "electrical", "name": "Electrical", "product_quarantined": True}],
            "items": items}


def manifest(entries):
    return {"entries": [{"legacy_id": legacy, "legacy_kind": "defect", "change_type": ct,
                         "successors": [{"id": s, "kind": "defect"} for s in succ]}
                        for legacy, ct, succ in entries]}


def queue_case(case_id, *, lane, item="vinyl_torn", source="canary", prop=PROP, run=RUN, cid=CID,
               photos=(PHOTO,), issue=ISSUE, p2c_item=None, observation="Flooring is lifting.",
               attribute=True, verdict="unsupported", disposition="excluded", pinned=True, scene="kitchen"):
    return {
        "case_id": case_id, "lane": lane, "attribute": attribute, "status": "pending", "untraceable_reason": None,
        "source_ids": {"card_id": case_id, "adjudication_card_id": None, "gold_id": None, "gold_photo_key": None},
        "run_ref": {"source": source, "property_key": prop, "run_id": run,
                    "artifact_path": f"C:/art/{prop}/{run}/photo_intel_debug.json" if pinned else None,
                    "artifact_sha256": "ab" * 32 if pinned else None},
        "human_truth": {"basis": "v1_1", "class_v1_1": "dirB_recovery", "slug": "exact_and_warranted",
                        "claim": "exact", "work": "warranted", "note": None, "arm": "reask",
                        "v1_verdict": "terra_claim_supported"},
        "v5_claim": {"condition_id": cid, "catalog_item_id": item, "catalog_kind": "defect",
                     "estimate_unit_id": "kitchen", "scene_group": scene, "claim_text": "vinyl flooring torn",
                     "terra_verdict": verdict, "terra_rationale": "because", "disposition": disposition,
                     "reason_code": "rc", "accepted": verdict == "supported" and disposition == "accepted_for_work"},
        "lineage": {"photo_keys": list(photos), "representative_photo_keys": list(photos), "issue_ids": [issue],
                    "per_photo": {pk: {"photo_key": pk, "p2c_surviving": [
                        {"issue_id": issue, "kind": "defect", "catalog_item_id": p2c_item or item,
                         "description": observation}]} for pk in photos},
                    "per_issue": [{"issue_id": issue, "photo_key": photos[0], "observation": observation,
                                   "p2b_join": {"matched": True, "method": "exact", "bullet_index": 0},
                                   "p2c_present": True,
                                   "p2d": {"resolved_item_id": item, "resolution_path": "llm",
                                           "routing_reason": "exact_kind", "shortcut_reason": None,
                                           "candidate_count": 2},
                                   "p2e_status": "kept", "flat_lane": "present", "projected_condition_id": cid}]},
        "context": {"strata": ["dirB"], "direction": "B", "second_opinion": None, "photo_count": len(photos),
                    "in_factorized_disagreements": []},
        "mechanical_hints": {},
    }


def gold_queue_case(case_id, *, prop=PROP, photo=PHOTO, gold_id="g1", run=RUN, covering=None, finding="Fan is dated"):
    return {"case_id": case_id, "lane": "miss_gold", "attribute": True, "status": "pending",
            "untraceable_reason": None,
            "human_truth": {"basis": "gold", "gold_id": gold_id, "finding": finding, "photo": f"{prop}/{photo}"},
            "v5_claim": {"condition_id": "", "catalog_item_id": None, "catalog_kind": None, "claim_text": None,
                         "covering_rejected_condition_id": covering, "disposition": None, "terra_verdict": None,
                         "accepted": None},
            "run_ref": {"source": "canary", "property_key": prop, "run_id": run,
                        "artifact_path": f"C:/art/{prop}/{run}/photo_intel_debug.json", "artifact_sha256": "ab" * 32},
            "mechanical_hints": {}}


def verdict(case_id, attribution, stage=None, **extra):
    rec = {"case_id": case_id, "ts": "2026-09-01T00:00:00+00:00", "reviewer": "t", "attribution": attribution,
           "confidence": "medium", "rationale": "r", **extra}
    if stage:
        rec["first_responsible_stage"] = stage
    return rec


def card(card_id, *, item="vinyl_torn", source="canary", prop=PROP, run=RUN, cid=CID, accepted=False):
    return {"card_id": card_id, "kind": "condition", "source": source, "property_key": prop, "run_id": run,
            "meta": {"catalog_item_id": item, "catalog_kind": "defect", "scene_group": "kitchen",
                     "condition_id": cid, "terra_verdict": "unsupported", "accepted": accepted,
                     "disposition": "excluded"},
            "claim": {"catalog_claim": "vinyl flooring torn", "observations": ["Flooring is lifting."]},
            "strips": [{"label": "evidence", "photos": [{"key": PHOTO, "path": "x", "wh": [1, 1]}]}],
            "strata": ["dirB"]}


def label(card_id, *, klass="supported_billed", item="vinyl_torn", source="canary", prop=PROP, cid=CID):
    return {"adjudication_card_id": None, "arm": "reask", "source": source, "property_key": prop,
            "condition_id": cid, "catalog_item_id": item, "catalog_kind": "defect", "class_v1_1": klass,
            "slug": "exact_and_warranted", "claim": "exact", "work": "warranted", "note": "n"}


def artifact(*, cid=CID, item="vinyl_torn", issue=ISSUE, photo=PHOTO, candidates=None, catalog_sha="cc" * 32):
    cands = candidates if candidates is not None else [
        {"item_id": item, "score": 0.71}, {"item_id": "hard_flooring_broken", "score": 0.65}]
    return {
        "photos": {photo: {"debug": {"resolved_items": [
            {"issue_id": issue, "resolved_item_id": item, "resolution_path": "llm", "routing_reason": "exact_kind",
             "shortcut_reason": None, "candidates": cands}]}}},
        "renovation_estimate_v5": {
            "schema_version": "v5", "state": "complete",
            "provenance": {"architecture_mode": "new", "catalog_sha256": catalog_sha, "catalog_version": "3.1"},
            "result": {"observed_conditions": [{"condition_id": cid, "catalog_item_id": item, "catalog_kind": "defect",
                                                "scene_group": "kitchen", "issue_ids": [issue]}],
                       "condition_reviews": [{"condition_id": cid, "verdict": "unsupported", "rationale": "r"}],
                       "condition_dispositions": [{"condition_id": cid, "disposition": "excluded"}],
                       "evidence_facts": [{"condition_id": cid, "photo_keys": [photo],
                                           "evidence_refs": [{"issue_id": issue, "photo_key": photo,
                                                              "observation": "Flooring is lifting."}]}],
                       "terra_calls": [], "work_items": []}}}


def loaded_artifact(art, verified=True):
    return {"verified": verified, "artifact": art, **mod.artifact_conditions(art)}


def identity(era_blob, head_blob, on_disk=None):
    return mod.resolve_catalog_identity({sha256_bytes(mod.to_crlf(era_blob))},
                                        [("c_head", head_blob), ("c_era", era_blob)],
                                        head_blob=head_blob, on_disk=on_disk or mod.to_crlf(head_blob))


def world():
    """A small but complete input set for build_bundle."""
    items = [catalog_item("vinyl_torn", pricing_status="inherited_from_split_parent"),
             catalog_item("vinyl_worn", kind="degradation", pricing_status="inherited_from_split_parent"),
             catalog_item("hard_flooring_broken", wic="FLOORING_REPLACE"),
             catalog_item("dated_outlets", kind="modernization", trade="electrical", wic="ELEC")]
    era = catalog(copy.deepcopy(items), version="3.1")
    current = catalog(copy.deepcopy(items))
    current["items"][0]["package_affinity"]["kitchen"]["repair_support_when_driven"] = True
    art = artifact()
    cases = [
        queue_case("rc_2d", lane="appendix_misnamed"),
        queue_case("rc_terra", lane="miss_label", cid="oc1_b", issue="iss_b", photos=("photo_002.jpg",)),
        queue_case("rc_reject", lane="counted_correct_rejection", cid="oc1_c", issue="iss_c", attribute=False),
        queue_case("rc_agree", lane="counted_agreement", cid="oc1_d", issue="iss_d", attribute=False,
                   verdict="supported", disposition="accepted_for_work"),
        queue_case("rc_halluc", lane="halluc_label", cid="oc1_e", issue="iss_e", verdict="supported",
                   disposition="accepted_for_work"),
        queue_case("rc_quarantine", lane="miss_label", item="dated_outlets", cid="oc1_f", issue="iss_f"),
    ]
    from collections import Counter
    queue = {"cases": cases, "lane_counts": dict(Counter(c["lane"] for c in cases)),
             "gold_photos": [{"gold_photo_id": f"{PROP}/{PHOTO}", "property_key": PROP, "photo_key": PHOTO,
                              "run_ref": cases[0]["run_ref"],
                              "lineage": {"p2c_surviving": [{"issue_id": ISSUE, "description": "Flooring is lifting."}]}}]}
    gold = {"cases": [gold_queue_case("ga_cov", gold_id="g1", covering="oc1_c"),
                      gold_queue_case("ga_free", gold_id="g2"),
                      gold_queue_case("ga_art", gold_id="g5", covering="oc1_h")],   # covered by a non-card condition
            "matching_table": [
                {"decision": "miss_candidate", "gold_id": "g1", "finding": "f", "photo": f"{PROP}/{PHOTO}",
                 "case_id": "ga_cov", "matched_condition_id": None, "covering_rejected_condition_id": "oc1_c", "note": ""},
                {"decision": "miss_candidate", "gold_id": "g2", "finding": "f", "photo": f"{PROP}/{PHOTO}",
                 "case_id": "ga_free", "matched_condition_id": None, "covering_rejected_condition_id": None, "note": ""},
                {"decision": "out_of_catalog", "gold_id": "g3", "finding": "mosaic", "photo": f"{PROP}/{PHOTO}",
                 "matched_condition_id": None, "covering_rejected_condition_id": None, "note": ""},
                {"decision": "matched", "gold_id": "g4", "finding": "m", "photo": f"{PROP}/{PHOTO}",
                 "matched_condition_id": CID, "covering_rejected_condition_id": None, "note": ""},
                {"decision": "gold_incomplete", "photo": f"{PROP}/{PHOTO}", "condition_id": "oc1_d",
                 "photo_supports_claim": "yes", "note": ""}]}
    latest = {"rc_2d": verdict("rc_2d", "downstream", "2d"), "rc_terra": verdict("rc_terra", "downstream", "terra"),
              "rc_halluc": verdict("rc_halluc", "pass_2a"), "rc_quarantine": verdict("rc_quarantine", "downstream", "terra"),
              "ga_cov": verdict("ga_cov", "downstream", "2d"), "ga_free": verdict("ga_free", "pass_2a"),
              "ga_art": verdict("ga_art", "downstream", "2c")}
    cards = {c["card_id"]: c for c in [card("rc_2d"), card("rc_terra", cid="oc1_b"), card("rc_reject", cid="oc1_c"),
                                       card("rc_agree", cid="oc1_d", accepted=True), card("rc_halluc", cid="oc1_e"),
                                       card("rc_quarantine", cid="oc1_f", item="dated_outlets"),
                                       card("rc_pos", cid="oc1_g", accepted=True)]}
    labels = {"rc_pos": label("rc_pos", cid="oc1_g"), "rc_2d": label("rc_2d", klass="misnamed_billed")}
    leads = [{"property_key": PROP, "condition_id": CID, "derived_class": "misnamed_but_warranted",
              "observed_description": "worn plank", "stored_verdict": "unsupported"},
             {"property_key": PROP, "condition_id": "oc1_c", "derived_class": "misnamed_but_warranted",
              "observed_description": "x", "stored_verdict": "unsupported"},
             {"property_key": "redfin_unpinned", "condition_id": "oc1_z", "derived_class": "inconclusive",
              "observed_description": "y", "stored_verdict": "supported"}]
    merged = copy.deepcopy(art)
    for extra in (artifact(cid="oc1_c", issue="iss_c"), artifact(cid="oc1_h", issue="iss_h")):
        res = extra["renovation_estimate_v5"]["result"]
        merged["renovation_estimate_v5"]["result"]["observed_conditions"] += res["observed_conditions"]
        merged["renovation_estimate_v5"]["result"]["evidence_facts"] += res["evidence_facts"]
        merged["photos"][PHOTO]["debug"]["resolved_items"] += extra["photos"][PHOTO]["debug"]["resolved_items"]
    era_blob = json.dumps(era).encode("utf-8")
    head_blob = json.dumps(current).encode("utf-8")
    return {
        "queue": queue, "gold_cases": gold, "latest": latest, "labels": labels, "cards": cards,
        "notes": [{"card_id": "rc_2d", "theme": "claim_wording", "tag": None, "title": "t", "verdict": "v", "note": "n"},
                  {"card_id": "rc_pos", "theme": "other", "tag": None, "title": "t", "verdict": "v", "note": "n"}],
        "leads": leads,
        "findings_text": "# doc\n\n## 1. `vinyl_torn` has no driver\n\ntext `kitchen_repair` `vinyl_torn`\n\n## 2. other\n\n`nothing_here`\n",
        "factorized_manifest": {"catalog_sha256": "cc" * 32, "root": "run_1"},
        "catalog_current": current, "catalog_era": era,
        "catalog_identity": identity(era_blob, head_blob),
        "manifest": manifest([("worn_or_torn_vinyl", "split", ["vinyl_torn", "vinyl_worn"]),
                              ("hard_flooring_broken", "unchanged", ["hard_flooring_broken"]),
                              ("dated_outlets", "reclassified", ["dated_outlets"])]),
        "migration": None, "decisions": {"entries": [{"change_type": "split"}, {"change_type": "unchanged"},
                                                     {"change_type": "reclassified"}]},
        "authoring_surface": {"wording_override_fields": ["name"]},
        "artifacts": {mod.run_key("canary", PROP, RUN): loaded_artifact(merged)},
        "run_artifacts": [{"source": "canary", "property_key": PROP, "run_id": RUN, "status": "pinned", "verified": True,
                           "catalog_sha256": "cc" * 32, "artifact_path": "C:/art", "queue_sha256": "ab" * 32,
                           "current_sha256": "ab" * 32}],
        "run_count_by_source": {"canary": {"pinned": 1, "unpinned": 0}},
        "sources": [{"name": "queue", "path": "reports/x", "tier": "frozen", "sha256": "1", "expected_sha256": "1", "match": True}],
        "queue_input_pins": {}, "git": {"head": "abc", "branch": "b", "tracked_changes": [], "tracked_inputs": {}},
    }


def build(inp):
    inp = copy.deepcopy(inp)
    inp["migration"] = mod.migration_index(inp["manifest"])
    return mod.build_bundle(inp)


# --------------------------------------------------------------------------- ledger and pins

def test_ledger_latest_wins_and_undo_reuses_report_loader(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        verdict("rc_1", "unclear"), verdict("rc_1", "downstream", "2d", revision_of="claude"),
        verdict("rc_2", "pass_2a"), {"case_id": "rc_2", "attribution": None}]) + "\n", encoding="utf-8")
    latest = mod.latest_verdicts(path)
    assert mod.latest_verdicts is ear.latest_verdicts
    assert set(latest) == {"rc_1"}
    assert mod.attribution_of(latest["rc_1"]) == {"attribution": "downstream", "stage": "2d", "confidence": "medium",
                                                  "reviewer": "t", "revised": True, "gold_match": None}
    assert mod.attribution_of(None) is None


def test_frozen_pin_mismatch_fails_closed_and_recorded_tier_never_does():
    expected = {"queue": "aa", "labels": "bb"}
    assert mod.verify_pins({"queue": "aa", "labels": "bb", "generator": "zz"}, expected) == []
    problems = mod.verify_pins({"queue": "aa", "labels": "XX"}, expected)
    assert problems == ["labels: sha256 XX != pinned bb"]
    assert mod.verify_pins({"queue": "aa"}, expected) == ["labels: missing"]
    with pytest.raises(SystemExit) as exc:
        mod.fail_closed(problems)
    assert "labels" in str(exc.value)
    mod.fail_closed([])  # nothing to report, nothing raised


# --------------------------------------------------------------------------- catalogs

def test_catalog_identity_resolves_the_crlf_checkout_of_one_git_state():
    era = b'{"version": "3.1", "items": []}\n'
    head = b'{"version": "3.2", "items": []}\n'
    ident = identity(era, head)
    assert ident["evidence_era"]["commit"] == "c_era"
    assert ident["evidence_era"]["matched_convention"] == "crlf"
    assert ident["evidence_era"]["sha256_lf"] == sha256_bytes(era)
    assert ident["evidence_era"]["sha256_crlf"] == sha256_bytes(mod.to_crlf(era)) != sha256_bytes(era)
    assert ident["proposal_baseline"]["working_tree_matches_head"] is True
    assert ident["checkout_line_ending_convention"] == "crlf"
    assert [h["commit"] for h in ident["history"]] == ["c_head", "c_era"]
    with pytest.raises(SystemExit):
        mod.resolve_catalog_identity({"00" * 32}, [("c_era", era)], head_blob=head, on_disk=head)
    drifted = identity(era, head, on_disk=head + b"x")
    assert drifted["proposal_baseline"]["working_tree_matches_head"] is False


def test_compare_catalogs_computes_claim_equivalence_and_markers():
    era = catalog([catalog_item("a"), catalog_item("b", claim=("wall", "cracked")), catalog_item("c")], version="3.1")
    same = compare = mod.compare_catalogs(era, copy.deepcopy(era))
    assert same["atomic_claim_unchanged"] is True and same["claim_text_unchanged"] is True
    assert same["changed_items"] == [] and same["per_field_change_counts"] == {}
    current = copy.deepcopy(era)
    current["items"][0]["embed_text"] = "different embed"
    current["items"][1]["atomic_claim"]["state"] = "cracked or bowed"
    current["items"][2]["cost"] = {"mode": "allowance", "base_low": 1}
    current["items"][2]["package_affinity"]["kitchen"]["repair_support_when_driven"] = True
    current["items"].reverse()
    compare = mod.compare_catalogs(era, current)
    assert compare["atomic_claim_unchanged"] is False and compare["claim_text_unchanged"] is False
    assert compare["order_changed"] is True
    assert compare["per_field_change_counts"] == {"atomic_claim": 1, "claim_text": 1, "cost": 1,
                                                  "embed_source_text": 1, "embed_text": 1, "package_affinity": 1}
    assert [c["id"] for c in compare["changed_items"]] == ["c", "b", "a"]
    assert compare["changed_items"][1]["changes"]["claim_text"] == {"evidence_era": "wall cracked",
                                                                    "proposal_baseline": "wall cracked or bowed"}
    assert compare["repair_support_markers_added"] == [{"id": "c", "room": "kitchen", "package_type": "kitchen_modernization"}]
    assert compare["product_quarantined_trades"] == {"evidence_era": ["electrical"], "proposal_baseline": ["electrical"]}
    removed = copy.deepcopy(era)
    removed["items"].pop()
    assert mod.compare_catalogs(era, removed)["removed_ids"] == ["c"]
    assert mod.compare_catalogs(era, removed)["atomic_claim_unchanged"] is False


def test_embed_source_text_pins_the_unbound_retriever_helper():
    assert mod.embed_source_text({"embed_text": "  vinyl   torn  "}) == "vinyl torn"
    composed = mod.embed_source_text({"id": "x", "name": "Torn vinyl", "description": "Lifting sheet",
                                      "trade_bucket": "flooring", "kind": "defect"})
    assert composed == "Torn vinyl. Lifting sheet. Trade: flooring. Kind: defect."
    assert mod.claim_text(catalog_item("a", claim=("wall", "cracked"))) == "wall cracked"


def test_migration_index_parent_successors_siblings():
    idx = mod.migration_index(manifest([("p", "split", ["s1", "s2"]), ("u", "unchanged", ["u"]), ("r", "retired", [])]))
    assert idx["parent_of"] == {"s1": "p", "s2": "p", "u": "u"}
    assert idx["successors"] == {"p": ["s1", "s2"], "r": [], "u": ["u"]}
    assert idx["siblings"] == {"s1": ["s2"], "s2": ["s1"], "u": []}
    assert idx["change_type"]["r"] == "retired"


def test_findings_refs_intersect_catalog_ids():
    text = "# t\n\n## 1. `kitchen_repair` has no driver\n\nSee `vinyl_torn` and `kitchen_repair`.\n\n## 2. plain\n\nnothing\n"
    refs = mod.extract_findings_refs(text, {"vinyl_torn", "other"})
    assert refs == [{"section": 1, "title": "`kitchen_repair` has no driver", "mentions": ["kitchen_repair", "vinyl_torn"],
                     "item_refs": ["vinyl_torn"]},
                    {"section": 2, "title": "plain", "mentions": [], "item_refs": []}]


# --------------------------------------------------------------------------- cases and joins

def test_authoritative_2d_field_is_p2d_not_p2c_surviving():
    case = queue_case("rc_x", lane="miss_label", item="right_item", p2c_item="wrong_item")
    assert mod.case_resolved_items(case) == {"condition_item": "right_item", "issue_items": ["right_item"]}
    rec = mod.case_record(case, None)
    assert rec["issues"][0]["resolved_item_id"] == "right_item"
    assert rec["unit_key"] == mod.runtime_key("canary", PROP, RUN, CID)
    assert rec["attribution"] is None and rec["artifact_pinned"] is True
    orphan = queue_case("rc_o", lane="counted_orphan", pinned=False)
    orphan["v5_claim"], orphan["lineage"] = None, None
    assert mod.case_unit_key(orphan) is None
    assert mod.case_record(orphan, None)["issue_items"] == []


def test_factorized_join_is_run_scoped_and_unpinned_stays_unattached():
    art = loaded_artifact(artifact())
    lead = {"property_key": PROP, "condition_id": CID, "derived_class": "misnamed_but_warranted",
            "observed_description": "worn plank", "stored_verdict": "unsupported"}
    other_run_key = mod.runtime_key("canary", PROP, "20260818_000000_bbbbbbbb", CID)
    same_run_key = mod.runtime_key("canary", PROP, RUN, CID)
    joined = mod.join_factorized_leads([lead], {PROP: RUN}, {mod.run_key("canary", PROP, RUN): art},
                                       {other_run_key: "rc_other"}, {same_run_key: "rc_lab"}, {same_run_key: "rc_card"})
    assert joined[0]["joined_case_id"] is None  # same (property, condition) in another run never joins
    assert joined[0]["joined_label_id"] == "rc_lab" and joined[0]["joined_card_id"] == "rc_card"
    assert joined[0]["catalog_item_id"] == "vinyl_torn" and joined[0]["item_source"] == "artifact_observed_conditions"
    assert joined[0]["unit_key"] == same_run_key and joined[0]["unattached_reason"] is None
    unpinned = mod.join_factorized_leads([dict(lead, property_key="redfin_np")], {PROP: RUN}, {}, {}, {}, {})
    assert unpinned[0]["unattached_reason"] == "run_identity_unpinned" and unpinned[0]["catalog_item_id"] is None
    absent = mod.join_factorized_leads([dict(lead, condition_id="oc1_missing")], {PROP: RUN},
                                       {mod.run_key("canary", PROP, RUN): art}, {}, {}, {})
    assert absent[0]["unattached_reason"] == "condition_absent_from_pinned_run"
    unverified = mod.join_factorized_leads([lead], {PROP: RUN},
                                           {mod.run_key("canary", PROP, RUN): loaded_artifact(artifact(), verified=False)},
                                           {}, {}, {})
    assert unverified[0]["unattached_reason"] == "artifact_unverified"


def test_label_card_and_note_card_joins():
    cards = {"rc_a": card("rc_a"), "rc_b": card("rc_b", item="other_item", cid="oc1_b")}
    labels = {"rc_a": label("rc_a"), "rc_b": label("rc_b", cid="oc1_b"), "rc_missing": label("rc_missing")}
    cases = {"rc_a": mod.case_record(queue_case("rc_a", lane="miss_label"), None)}
    joined = {j["card_id"]: j for j in mod.join_labels(labels, cards, cases)}
    assert joined["rc_a"]["item_id_matches_card"] is True and joined["rc_a"]["case_id"] == "rc_a"
    assert joined["rc_a"]["unit_key"] == mod.runtime_key("canary", PROP, RUN, CID)
    assert joined["rc_a"]["card_photo_keys"] == [PHOTO] and joined["rc_a"]["run_id"] == RUN
    assert joined["rc_b"]["item_id_matches_card"] is False
    assert joined["rc_missing"]["card_present"] is False and joined["rc_missing"]["unit_key"] is None
    notes = mod.join_notes([{"card_id": "rc_b", "theme": "x", "tag": None, "title": "t", "verdict": "v", "note": "n"},
                            {"card_id": "rc_a", "theme": "y", "tag": "d", "title": "t", "verdict": "v", "note": "n"}],
                           cards, cases, labels)
    assert [n["card_id"] for n in notes] == ["rc_a", "rc_b"]
    assert notes[0]["is_case"] and notes[0]["is_label"] and notes[0]["catalog_item_id"] == "vinyl_torn"
    assert notes[1]["is_case"] is False and notes[1]["catalog_item_id"] == "other_item"


def test_gold_rows_take_run_identity_from_the_gold_photo():
    gold_photos = {f"{PROP}/{PHOTO}": {"run_ref": {"source": "canary", "run_id": RUN}}}
    rows = [{"decision": "matched", "gold_id": "g2", "photo": f"{PROP}/{PHOTO}", "matched_condition_id": CID},
            {"decision": "gold_incomplete", "photo": f"{PROP}/{PHOTO}", "condition_id": "oc1_d", "photo_supports_claim": "yes"},
            {"decision": "out_of_catalog", "gold_id": "g1", "photo": f"{PROP}/{PHOTO}"},
            {"decision": "already_cased", "gold_id": "g3", "photo": f"{PROP}/{PHOTO}",
             "covering_rejected_condition_id": "oc1_c", "already_cased": {"case_id": "rc_c"}},
            {"decision": "miss_candidate", "gold_id": "g4", "photo": "redfin_other/photo_009.jpg", "case_id": "ga_x"}]
    unit = mod.runtime_key("canary", PROP, RUN, CID)
    joined = mod.join_gold_rows(rows, gold_photos, {unit: "rc_a"})
    assert [(r["decision"], r["gold_id"]) for r in joined] == [
        ("gold_incomplete", None), ("out_of_catalog", "g1"), ("matched", "g2"), ("already_cased", "g3"), ("miss_candidate", "g4")]
    by = {r["decision"]: r for r in joined}
    assert by["matched"]["runtime_unit_key"] == unit and by["matched"]["linked_case_id"] == "rc_a"
    assert by["matched"]["role"] == "annotation" and by["out_of_catalog"]["role"] == "seed"
    assert by["gold_incomplete"]["runtime_unit_key"] == mod.runtime_key("canary", PROP, RUN, "oc1_d")
    assert by["already_cased"]["runtime_unit_key"] == mod.runtime_key("canary", PROP, RUN, "oc1_c")
    assert by["out_of_catalog"]["gold_unit_key"] == mod.gold_key(PROP, PHOTO, "g1")
    assert by["miss_candidate"]["run_id"] is None and by["miss_candidate"]["runtime_unit_key"] is None


# --------------------------------------------------------------------------- seeds, units, reconciliation

def test_seed_rules_lane_scoping_quarantine_and_findings():
    bundle = build(world())
    records = {r["record_id"]: r for r in bundle["seeds"]["records"]}
    assert records["rc_2d"]["rules"] == ["R1", "R2"]                     # one record, two rules
    assert records["rc_terra"]["rules"] == ["R3"]
    assert "rc_halluc" not in records                                      # pass_2a hallucination never seeds
    assert bundle["lanes"]["hallucination_annotations"] == ["rc_halluc"]
    assert "rc_reject" not in records and "rc_agree" not in records
    assert records["rc_quarantine"]["lane"] == "product_policy"
    assert bundle["lanes"]["product_policy"] == ["rc_quarantine"]
    assert records["ga_cov"]["rules"] == ["R1", "R5"] and records["ga_cov"]["implicated_items"] == ["vinyl_torn"]
    assert records["ga_free"]["rules"] == ["R5"] and records["ga_free"]["lane"] == "coverage_question"
    assert records["ga_art"]["implicated_items"] == ["vinyl_torn"]   # covering condition resolved via the artifact
    gold_cases = bundle["indexes"]["gold_cases"]
    assert gold_cases["ga_art"]["covering_item_source"] == "artifact_observed_conditions"
    assert gold_cases["ga_art"]["covering_case_id"] is None
    assert gold_cases["ga_cov"]["covering_item_source"] == "queue_case"
    assert gold_cases["ga_free"]["covering_item"] is None and gold_cases["ga_free"]["covering_item_source"] is None
    assert records[f"gold_row:{PROP}/{PHOTO}:g3"]["rules"] == ["R6"]
    assert records["finding:1"]["implicated_items"] == ["vinyl_torn"] and records["finding:2"]["implicated_items"] == []
    assert bundle["lanes"]["deferred_findings"][0]["mentions"] == ["kitchen_repair", "vinyl_torn"]


def test_units_roles_corroboration_and_correlation():
    bundle = build(world())
    units = {u["unit_key"]: u for u in bundle["evidence_units"]}
    main_unit = units[mod.runtime_key("canary", PROP, RUN, CID)]
    roles = {r["record_id"]: r["role"] for r in main_unit["records"]}
    assert roles == {"rc_2d": "independent", f"lead:{PROP}:{CID}": "method_corroboration"}
    assert main_unit["rules"] == ["R1", "R2", "R4"] and main_unit["independent"] is True
    # the second lead sits on a correct-rejection condition that is not a seed: its own unit, independent
    lead_unit = units[mod.runtime_key("canary", PROP, RUN, "oc1_c")]
    assert lead_unit["primary_record_id"] == f"lead:{PROP}:oc1_c" and lead_unit["independent"] is True
    # gold case covering that same condition corroborates the lead's unit rather than counting again
    gold_cov = units[mod.gold_key(PROP, PHOTO, "g1")]
    assert gold_cov["corroborates"] == lead_unit["unit_key"] and gold_cov["independent"] is False
    assert lead_unit["corroborated_by"] == [gold_cov["unit_key"]]
    assert gold_cov["covering_case_id"] == "rc_reject"
    # gold case on the same photo without a condition link stays independent, flagged same_photo
    gold_free = units[mod.gold_key(PROP, PHOTO, "g2")]
    assert gold_free["independent"] is True and main_unit["unit_key"] in gold_free["same_photo_units"]
    assert main_unit["unit_key"] in gold_free["same_property_units"]
    assert gold_free["attached"] is False and gold_free["unattached_reason"] == "no_implicated_item"
    assert main_unit["gold_annotations"] == [{"decision": "matched", "gold_id": "g4", "photo": f"{PROP}/{PHOTO}"}]
    assert {x["record_id"]: x["reason"] for x in bundle["unattached"]} == {
        "finding:1": "backlog_reference", "finding:2": "backlog_reference",
        "lead:redfin_unpinned:oc1_z": "run_identity_unpinned"}


def test_reconciliation_counts_and_exactly_once_failure():
    bundle = build(world())
    rec = bundle["reconciliation"]
    assert rec["every_seed_reconciled_once"] is True
    assert rec["before_dedup"]["records_by_rule"] == {"R1": 2, "R2": 1, "R3": 2, "R4": 3, "R5": 3, "R6": 1, "R7": 2}
    assert rec["before_dedup"]["records"] == 12 and rec["before_dedup"]["rule_record_pairs"] == 14
    assert rec["after_dedup"]["units"] == 8 and rec["after_dedup"]["unattached_records"] == 3
    assert rec["after_dedup"]["record_roles"] == {"independent": 8, "method_corroboration": 1}
    assert rec["after_dedup"]["corroborating_units"] == 1
    records, units, unattached = bundle["seeds"]["records"], bundle["evidence_units"], bundle["unattached"]
    with pytest.raises(SystemExit):
        mod.reconcile_seeds(records, units[1:], unattached)          # a dropped unit loses its records
    with pytest.raises(SystemExit):
        mod.reconcile_seeds(records, units, unattached + unattached[:1])  # a record counted twice
    with pytest.raises(SystemExit):
        mod.reconcile_seeds(records[1:], units, unattached)          # a reconciled id nobody seeded


def test_worklist_and_families_are_counts_and_mechanical_neighbors():
    bundle = build(world())
    work = {w["item_id"]: w for w in bundle["worklist"]}
    assert set(work) == {"vinyl_torn", "dated_outlets"}
    vt = work["vinyl_torn"]
    assert vt["independent_units"] == 4 and vt["corroborating_units"] == 1 and vt["distinct_properties"] == 1
    assert vt["positive_uses"] == ["rc_pos"] and vt["agreements"] == ["rc_agree"] and vt["correct_rejections"] == ["rc_reject"]
    assert vt["hallucination_annotations"] == ["rc_halluc"] and vt["notes"] == ["rc_2d", "rc_pos"]
    assert vt["deferred_finding_refs"] == [1] and vt["lane"] == "catalog_candidate"
    assert work["dated_outlets"]["lane"] == "product_policy"
    assert "evidence_bar" not in vt
    fam = bundle["families"]["vinyl_torn"]
    assert fam["migration"] == {"legacy_id": "worn_or_torn_vinyl", "change_type": "split", "legacy_kind": "defect",
                                "successors_of_parent": ["vinyl_torn", "vinyl_worn"], "siblings": ["vinyl_worn"]}
    assert fam["same_kind_trade_scene"] == ["hard_flooring_broken"]
    assert fam["shared_economics"]["work_item_code"] == ["hard_flooring_broken", "vinyl_worn"]
    assert fam["shared_economics"]["estimate_group"] == ["dated_outlets", "hard_flooring_broken", "vinyl_worn"]
    assert fam["shared_economics"]["package_type"] == ["dated_outlets", "hard_flooring_broken", "vinyl_worn"]
    assert fam["family_members"] == ["dated_outlets", "hard_flooring_broken", "vinyl_worn"]
    # rc_2d's own issue plus the lead unit's condition case (rc_reject, not a seed); rc_terra's
    # issue has no resolved_items entry in the artifact and is recorded unavailable instead.
    assert fam["frozen_candidate_neighbors"] == [{"item_id": "hard_flooring_broken", "co_listed": 2, "best_rank": 2,
                                                  "score_min": 0.65, "score_max": 0.65}]
    assert bundle["candidate_enrichment"]["cases_and_leads"]["rc_terra"]["iss_b"]["status"] == "unavailable"
    assert fam["current_retrieval_neighbors"]["status"] == "unavailable"
    assert fam["reviewer_neighbors"] == {"added": [], "removed": [], "rationale": None}
    assert fam["observations"] == ["Flooring is lifting."]
    assert fam["dimensions"]["claim_text"] == {"evidence_era": "vinyl flooring torn or lifting",
                                               "proposal_baseline": "vinyl flooring torn or lifting"}
    assert vt["family_counts"]["positive_uses"] == 1


def test_bundle_is_deterministic_and_has_no_timestamp():
    base = world()
    first, second = build(base), build(base)
    assert first["fingerprint"] == second["fingerprint"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    permuted = copy.deepcopy(base)
    permuted["queue"]["cases"].reverse()
    permuted["labels"] = dict(reversed(list(permuted["labels"].items())))
    permuted["leads"].reverse()
    permuted["gold_cases"]["matching_table"].reverse()
    assert build(permuted)["fingerprint"] == first["fingerprint"]
    assert "generated_at" not in json.dumps(first)
    changed = copy.deepcopy(base)
    changed["git"]["head"] = "different"
    assert build(changed)["fingerprint"] == first["fingerprint"]  # git block is provenance, not content
    assert first["validation"]["ok"] is True
    assert first["baseline_comparison"]["atomic_claim_unchanged"] is True
    assert first["baseline_comparison"]["repair_support_markers_added"] == [
        {"id": "vinyl_torn", "room": "kitchen", "package_type": "kitchen_modernization"}]
    assert first["review_notes"][0]["card_id"] == "rc_2d" and len(first["review_notes"]) == 2
    assert first["candidate_enrichment"]["cases_and_leads"]["rc_2d"][ISSUE]["candidates"][0] == {
        "rank": 1, "item_id": "vinyl_torn", "score": 0.71}
    assert first["candidate_enrichment"]["cases_and_leads"]["rc_2d"][ISSUE]["agrees_with_queue"] is True


def test_render_markdown_is_deterministic_and_reflects_counts():
    bundle = build(world())
    text = mod.render_markdown(bundle)
    assert text == mod.render_markdown(bundle)
    assert bundle["fingerprint"] in text and "| R1 | 2 |" in text and "`vinyl_torn`" in text
    assert "atomic_claim unchanged: **True**" in text


# --------------------------------------------------------------------------- parity with the committed bundle

@pytest.mark.skipif(not all((mod.ROOT / rel).exists() for rel in mod.REL.values())
                    or not Path("C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts").is_dir()
                    or not mod.OUT_JSON.is_file(),
                    reason="frozen inputs, production artifacts, or the committed bundle are not present")
def test_committed_bundle_matches_a_fresh_build():
    fresh = mod.build_bundle(mod.load_all())
    committed = json.loads(mod.OUT_JSON.read_text(encoding="utf-8"))
    assert committed["fingerprint"] == fresh["fingerprint"] == mod.fingerprint(committed)
