"""Lineage joins and ledger reconciliation for the error-attribution audit.

Synthetic artifacts throughout: the point is the joins, and a fixture that
pins them is worth more than one that happens to agree with today's canary.
The dirB case matters most -- projection runs before Terra, so a rejected
condition must walk back exactly like an accepted one. An accepted-only walk
would silently skip every miss case in the audit.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_error_attribution.py -q
"""
import json

import pytest

from scripts.build_error_attribution_queue import (
    condition_case,
    condition_lineage,
    issue_lineage,
    photo_lineage,
)
from scripts.error_attribution_report import (
    ReconciliationError,
    latest_verdicts,
    reconcile,
    tally,
)

CID = "oc1_1111111111111111"
ISSUE = "iss_aaaa"
OTHER_ISSUE = "iss_bbbb"


def photo(*, bullets, matched, resolved=None, kept=None, prose="A dated kitchen."):
    return {
        "scene": "kitchen",
        "pass_states": {"2a": "executed"},
        "features": {"observations_freeform": prose},
        "issues": {"matched": matched, "final": matched, "removed": []},
        "debug": {
            "observations_struct": {"observations": [{"description": b} for b in bullets]},
            "resolved_items": resolved or [],
            "passes": {"2e": {"kept_issue_ids": kept if kept is not None
                              else [m["issue_id"] for m in matched],
                              "removed_count": 0, "suppressed_reason_counts": {}}},
        },
    }


def artifact(photos, *, conditions, reviews, dispositions, evidence, flat=None):
    return {
        "photos": photos,
        "estimate_issues_flat": flat if flat is not None else [],
        "renovation_estimate_v5": {
            "schema_version": "v5", "state": "complete",
            "provenance": {"architecture_mode": "new"},
            "result": {"observed_conditions": conditions, "condition_reviews": reviews,
                       "condition_dispositions": dispositions, "evidence_facts": evidence,
                       "terra_calls": [], "work_items": []},
        },
    }


def one_condition_artifact(*, terra_verdict, disposition, observation="Countertops are older laminate."):
    """One condition on one photo; verdict/disposition are the only knobs."""
    matched = [{"issue_id": ISSUE, "kind": "modernization", "catalogItemId": "outdated_kitchen_finishes",
                "description": observation}]
    photos = {"photo_001.jpg": photo(
        bullets=["Kitchen is narrow.", observation, "Window trim is painted."],
        matched=matched,
        resolved=[{"issue_id": ISSUE, "resolved_item_id": "outdated_kitchen_finishes",
                   "resolution_path": "llm", "routing_reason": "exact_kind",
                   "shortcut_reason": None, "candidates": [{}, {}]}])}
    return artifact(
        photos,
        conditions=[{"condition_id": CID, "catalog_item_id": "outdated_kitchen_finishes",
                     "catalog_kind": "modernization", "estimate_unit_id": "kitchen",
                     "scene_group": "kitchen", "issue_ids": [ISSUE]}],
        reviews=[{"condition_id": CID, "verdict": terra_verdict, "rationale": "because"}],
        dispositions=[{"condition_id": CID, "disposition": disposition, "reason_code": "rc"}],
        evidence=[{"condition_id": CID, "photo_keys": ["photo_001.jpg"],
                   "representative_photo_keys": ["photo_001.jpg"],
                   "evidence_refs": [{"issue_id": ISSUE, "photo_key": "photo_001.jpg",
                                      "observation": observation}]}],
        flat=[{"issue_id": ISSUE, "photo_key": "photo_001.jpg", "description": observation}])


def listing_for(art, *, run_id="20260818_120000_abcdef12", tmp_path=None):
    from tools import review_cards as rc
    path = (tmp_path / run_id / "photo_intel_debug.json") if tmp_path else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(art), encoding="utf-8")
    res = rc.v5_result(art)
    return {"run_id": run_id, "path": path, "artifact": art, "result": res,
            "idx": rc.index_result(res), "paths": {}}


# --------------------------------------------------------------------------- joins

def test_2b_join_exact_normalized_and_none():
    """The join carries the canonical issue description back to its 2b bullet.

    Exact is the real-world case (2,919/2,919 across the audited corpus); the
    normalized fallback and the miss exist so a failure is data, not a crash."""
    art = one_condition_artifact(terra_verdict="supported", disposition="accepted_for_work")
    from tools import review_cards as rc
    idx = rc.index_result(rc.v5_result(art))

    exact = issue_lineage(art, "photo_001.jpg", ISSUE, "Countertops are older laminate.", idx)
    assert exact["p2b_join"] == {"matched": True, "method": "exact", "bullet_index": 1}

    loose = issue_lineage(art, "photo_001.jpg", ISSUE, "  countertops  are OLDER laminate. ", idx)
    assert loose["p2b_join"]["matched"] is True
    assert loose["p2b_join"]["method"] == "normalized"
    assert loose["p2b_join"]["bullet_index"] == 1

    missing = issue_lineage(art, "photo_001.jpg", ISSUE, "Roof is missing shingles.", idx)
    assert missing["p2b_join"] == {"matched": False, "method": "none", "bullet_index": None}
    assert missing["p2c_present"] is True  # the issue still exists; only the join failed


def test_2c_drop_set_is_bullets_minus_survivors():
    """2c records no rationale for a drop, so the audit can only show the difference."""
    art = one_condition_artifact(terra_verdict="supported", disposition="accepted_for_work")
    lineage = photo_lineage(art, "photo_001.jpg")
    assert lineage["p2b_bullets"] == ["Kitchen is narrow.", "Countertops are older laminate.",
                                      "Window trim is painted."]
    assert lineage["p2c_dropped"] == ["Kitchen is narrow.", "Window trim is painted."]
    assert [s["issue_id"] for s in lineage["p2c_surviving"]] == [ISSUE]
    assert lineage["p2a_present"] is True and lineage["p2a_sha256"]


def test_duplicate_descriptions_across_photos_do_not_bleed():
    """Two photos can carry the same sentence; each photo's drop set is its own."""
    shared = "Walls are scuffed."
    photos = {
        "photo_001.jpg": photo(bullets=[shared], matched=[{"issue_id": ISSUE, "kind": "defect",
                                                           "catalogItemId": "wall_scuffs",
                                                           "description": shared}]),
        "photo_002.jpg": photo(bullets=[shared], matched=[]),
    }
    art = artifact(photos, conditions=[], reviews=[], dispositions=[], evidence=[])
    assert photo_lineage(art, "photo_001.jpg")["p2c_dropped"] == []
    assert photo_lineage(art, "photo_002.jpg")["p2c_dropped"] == [shared]


@pytest.mark.parametrize("terra_verdict,disposition", [
    ("supported", "accepted_for_work"),
    ("unsupported", "excluded"),          # the dirB shape: every miss case in the audit
    ("cannot_assess", "inspection"),
])
def test_condition_walk_is_complete_regardless_of_verdict(terra_verdict, disposition):
    """Projection precedes Terra, so lineage cannot depend on the verdict."""
    art = one_condition_artifact(terra_verdict=terra_verdict, disposition=disposition)
    lineage = condition_lineage(listing_for(art), CID)
    assert lineage["photo_keys"] == ["photo_001.jpg"]
    assert lineage["issue_ids"] == [ISSUE]
    assert len(lineage["per_issue"]) == 1
    issue = lineage["per_issue"][0]
    assert issue["p2b_join"]["matched"] is True
    assert issue["p2c_present"] is True
    assert issue["p2d"]["resolved_item_id"] == "outdated_kitchen_finishes"
    assert issue["p2e_status"] == "kept"
    assert issue["projected_condition_id"] == CID
    assert lineage["per_photo"]["photo_001.jpg"]["p2a_prose"]


def test_2e_status_distinguishes_kept_from_dropped():
    art = one_condition_artifact(terra_verdict="supported", disposition="accepted_for_work")
    art["photos"]["photo_001.jpg"]["debug"]["passes"]["2e"]["kept_issue_ids"] = [OTHER_ISSUE]
    from tools import review_cards as rc
    idx = rc.index_result(rc.v5_result(art))
    assert issue_lineage(art, "photo_001.jpg", ISSUE, "x", idx)["p2e_status"] == "absent"
    assert issue_lineage(art, "photo_001.jpg", OTHER_ISSUE, "x", idx)["p2e_status"] == "kept"


def test_multi_photo_condition_carries_every_evidence_photo():
    """Conditions group on (catalog_item_id, estimate_unit_id) and routinely span
    photos; a miss is only Pass 2a's when *no* evidence photo's prose has it."""
    obs_a, obs_b = "Floor is worn.", "Floor has deep scratches."
    photos = {
        "photo_001.jpg": photo(bullets=[obs_a], prose="Prose A.",
                               matched=[{"issue_id": ISSUE, "kind": "defect",
                                         "catalogItemId": "flooring", "description": obs_a}]),
        "photo_002.jpg": photo(bullets=[obs_b], prose="Prose B.",
                               matched=[{"issue_id": OTHER_ISSUE, "kind": "defect",
                                         "catalogItemId": "flooring", "description": obs_b}]),
    }
    art = artifact(
        photos,
        conditions=[{"condition_id": CID, "catalog_item_id": "flooring", "catalog_kind": "defect",
                     "estimate_unit_id": "living", "issue_ids": [ISSUE, OTHER_ISSUE]}],
        reviews=[{"condition_id": CID, "verdict": "unsupported", "rationale": "r"}],
        dispositions=[{"condition_id": CID, "disposition": "excluded"}],
        evidence=[{"condition_id": CID, "photo_keys": ["photo_001.jpg", "photo_002.jpg"],
                   "evidence_refs": [
                       {"issue_id": ISSUE, "photo_key": "photo_001.jpg", "observation": obs_a},
                       {"issue_id": OTHER_ISSUE, "photo_key": "photo_002.jpg", "observation": obs_b}]}])
    lineage = condition_lineage(listing_for(art), CID)
    assert set(lineage["per_photo"]) == {"photo_001.jpg", "photo_002.jpg"}
    assert {p["p2a_prose"] for p in lineage["per_photo"].values()} == {"Prose A.", "Prose B."}
    assert all(i["p2b_join"]["matched"] for i in lineage["per_issue"])


# --------------------------------------------------------------------------- case resolution

def _case(listings, **over):
    kwargs = dict(case_id="rc_test", lane="miss_label", card={"card_id": "rc_test",
                  "run_id": "20260818_120000_abcdef12", "strata": ["dirB"], "meta": {}},
                  human_truth={"basis": "v1_1", "class_v1_1": "dirB_recovery"},
                  listings=listings, source="canary", property_key="redfin_1",
                  condition_id=CID, claims={"outdated_kitchen_finishes": "the claim"},
                  factorized={})
    kwargs.update(over)
    return condition_case(**kwargs)


def test_case_resolves_to_its_run(tmp_path):
    art = one_condition_artifact(terra_verdict="unsupported", disposition="excluded")
    case = _case({("canary", "redfin_1"): listing_for(art, tmp_path=tmp_path)})
    assert case["status"] == "pending"
    assert case["run_ref"]["artifact_sha256"]
    assert case["v5_claim"]["accepted"] is False
    assert case["v5_claim"]["claim_text"] == "the claim"
    assert case["mechanical_hints"]["all_issues_joined_to_2b"] is True


def test_missing_run_is_untraceable_never_substituted(tmp_path):
    """condition_id is run-scoped, so another run is a different condition."""
    art = one_condition_artifact(terra_verdict="unsupported", disposition="excluded")
    other = listing_for(art, run_id="20260819_090000_ffffffff", tmp_path=tmp_path)

    absent = _case({})
    assert absent["status"] == "untraceable" and absent["lineage"] is None

    mismatched = _case({("canary", "redfin_1"): other})
    assert mismatched["status"] == "untraceable"
    assert "not interchangeable" in mismatched["untraceable_reason"]

    unknown = _case({("canary", "redfin_1"): listing_for(art, tmp_path=tmp_path)},
                    condition_id="oc1_nope")
    assert unknown["status"] == "untraceable"
    assert "absent from run" in unknown["untraceable_reason"]


# --------------------------------------------------------------------------- ledger

def queue_fixture(**over):
    case = {"case_id": "rc_a", "lane": "miss_label", "attribute": True,
            "run_ref": {"source": "canary", "property_key": "p", "run_id": "r"},
            "human_truth": {"basis": "v1_1", "class_v1_1": "dirB_recovery"},
            "v5_claim": {"condition_id": CID, "catalog_item_id": "item",
                         "terra_verdict": "unsupported"},
            "mechanical_hints": {"join_methods": ["exact"]}, "status": "pending"}
    case.update(over)
    return {"schema_version": 1, "lane_counts": {"miss_label": 1}, "cases": [case]}


def verdict(**over):
    rec = {"case_id": "rc_a", "attribution": "downstream", "first_responsible_stage": "2c",
           "confidence": "high", "rationale": "the bullet is gone after 2c"}
    rec.update(over)
    return rec


def test_reconcile_accepts_a_complete_ledger():
    assert reconcile(queue_fixture(), {"rc_a": verdict()})["ok"] is True


def test_reconcile_rejects_unreviewed_and_stageless_and_orphan():
    with pytest.raises(ReconciliationError, match="no verdict"):
        reconcile(queue_fixture(), {})
    with pytest.raises(ReconciliationError, match="first_responsible_stage"):
        reconcile(queue_fixture(), {"rc_a": verdict(first_responsible_stage=None)})
    with pytest.raises(ReconciliationError, match="first_responsible_stage set"):
        reconcile(queue_fixture(), {"rc_a": verdict(attribution="pass_2a")})
    with pytest.raises(ReconciliationError, match="no queue case"):
        reconcile(queue_fixture(), {"rc_a": verdict(), "rc_ghost": verdict(case_id="rc_ghost")})
    with pytest.raises(ReconciliationError, match="no rationale"):
        reconcile(queue_fixture(), {"rc_a": verdict(rationale="  ")})


def test_reconcile_caps_v1_only_confidence_and_demands_join_rationale():
    q = queue_fixture(human_truth={"basis": "v1_only", "v1_verdict": "terra_claim_unsupported"})
    with pytest.raises(ReconciliationError, match="high confidence"):
        reconcile(q, {"rc_a": verdict()})
    assert reconcile(q, {"rc_a": verdict(confidence="medium")})["ok"] is True

    broken = queue_fixture(mechanical_hints={"join_methods": ["none"]})
    with pytest.raises(ReconciliationError, match="join failed"):
        reconcile(broken, {"rc_a": verdict()})
    assert reconcile(broken, {"rc_a": verdict(rationale="the 2b join failed; read the prose")})["ok"]


def test_reconcile_catches_duplicates_and_lane_drift():
    dupe = queue_fixture()
    dupe["cases"] = dupe["cases"] * 2
    dupe["lane_counts"] = {"miss_label": 2}
    with pytest.raises(ReconciliationError, match="duplicate case_id"):
        reconcile(dupe, {"rc_a": verdict()})

    drift = queue_fixture()
    drift["lane_counts"] = {"miss_label": 9}
    with pytest.raises(ReconciliationError, match="lane counts drifted"):
        reconcile(drift, {"rc_a": verdict()})


def test_allow_incomplete_permits_a_review_in_progress():
    status = reconcile(queue_fixture(), {}, allow_incomplete=True)
    assert status["pending"] == ["rc_a"]


def test_untraceable_case_needs_no_verdict():
    q = queue_fixture(status="untraceable", attribute=False)
    assert reconcile(q, {})["ok"] is True


def test_latest_verdict_wins_and_null_undoes(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        verdict(attribution="pass_2a", first_responsible_stage=None),
        verdict(),
        {"case_id": "rc_b", "attribution": "unclear"},
        {"case_id": "rc_b", "attribution": None},
    ]), encoding="utf-8")
    got = latest_verdicts(path)
    assert got["rc_a"]["attribution"] == "downstream"
    assert "rc_b" not in got


def gold_fixture(**over):
    """Gold cases are review output, so they arrive beside the frozen queue."""
    case = {"case_id": "ga_p_photo_001_g1", "lane": "miss_gold", "attribute": True,
            "run_ref": {"source": "canary", "property_key": "p", "run_id": "r"},
            "human_truth": {"basis": "gold", "gold_id": "g1", "finding": "Ceiling is stained."},
            "v5_claim": {"condition_id": "", "catalog_item_id": None, "terra_verdict": None},
            "mechanical_hints": {"join_methods": []}, "status": "pending"}
    case.update(over)
    return {"schema_version": 1, "cases": [case],
            "matching_table": [{"gold_id": "g1", "decision": "miss_candidate"},
                               {"gold_id": "g2", "decision": "out_of_catalog"}]}


def test_gold_cases_merge_into_reconciliation_and_the_miss_headline():
    """The gold lane is real misses; it must reach the headline, not a side table."""
    gold_verdict = verdict(case_id="ga_p_photo_001_g1", first_responsible_stage="2c",
                           confidence="medium", rationale="the bullet never survived 2c")
    verdicts = {"rc_a": verdict(), "ga_p_photo_001_g1": gold_verdict}

    with pytest.raises(ReconciliationError, match="no queue case"):
        reconcile(queue_fixture(), verdicts)  # without --gold-cases it is an orphan

    status = reconcile(queue_fixture(), verdicts, gold=gold_fixture())
    assert status["ok"] is True and status["gold_cases"] == 1

    report = tally(queue_fixture(), verdicts, gold=gold_fixture())
    assert report["misses"]["judged"] == 2
    assert report["by_lane"]["miss_gold"] == {"downstream": 1}
    assert report["by_basis"]["gold"] == {"downstream": 1}
    assert report["gold"]["matching"] == {"miss_candidate": 1, "out_of_catalog": 1}
    assert report["gold"]["cases"] == 1


def test_gold_cases_must_obey_their_own_contract():
    verdicts = {"rc_a": verdict()}
    for over, match in (
        ({"case_id": "rc_a"}, "collide with the queue"),
        ({"case_id": "zz_bad"}, "must start with ga_ or gx_"),
        ({"lane": "miss_label"}, "outside"),
        ({"attribute": False}, "must be attributable"),
        ({"human_truth": {"basis": "v1_1"}}, "basis must be 'gold'"),
    ):
        with pytest.raises(ReconciliationError, match=match):
            reconcile(queue_fixture(), verdicts, gold=gold_fixture(**over),
                      allow_incomplete=True)


def test_gold_extra_cannot_duplicate_a_condition_the_queue_already_carries():
    """A gx_ case anchored on an rc_ case's condition would double-count one error."""
    extra = gold_fixture(case_id="gx_p_" + CID, lane="halluc_gold_extra",
                         v5_claim={"condition_id": CID, "catalog_item_id": "item",
                                   "terra_verdict": "supported"})
    with pytest.raises(ReconciliationError, match="appears in"):
        reconcile(queue_fixture(), {"rc_a": verdict()}, gold=extra, allow_incomplete=True)


def test_tally_totals_reconcile_with_the_ledger():
    report = tally(queue_fixture(), {"rc_a": verdict()})
    assert report["misses"] == {"counts": {"downstream": 1}, "judged": 1, "pass_2a": 0,
                                "downstream": 1, "unclear": 0, "pass_2a_share": 0.0,
                                "downstream_share": 1.0}
    assert report["downstream_stages"] == {"2c": 1}
    assert report["by_basis"]["v1_1"] == {"downstream": 1}
    assert [r["case_id"] for r in report["cases"]] == ["rc_a"]
    # counted-only lanes never enter a judged denominator
    counted = tally(queue_fixture(attribute=False, lane="counted_orphan"), {})
    assert counted["misses"]["judged"] == 0
    assert counted["by_lane"]["counted_orphan"] == {"counted_only": 1}
