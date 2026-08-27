"""scripts/review_analysis — descriptive analysis of the completed manual review."""
import json
from pathlib import Path

import pytest

from scripts import review_analysis as ra
from scripts import review_tally
from tools import review_cards as rc
from tools.renovation_architecture.contracts import SHADOW_DEBUG_KEY

CLAIMS = {"floor_worn": "hard flooring scratched or worn", "paint_dated": "wall paint dated colour",
          "trim_dated": "interior trim dated style", "tile_cracked": "tile cracked"}
KINDS = {"floor_worn": "degradation", "paint_dated": "modernization",
         "trim_dated": "modernization", "tile_cracked": "defect"}
IMG = "C:/img/prop"


# fixture builders — copied from tests/test_review_cards.py (repo convention), with an
# application that carries absorbed_work_item_ids so the lineage path is exercised.

def _cond(cid, item, kind, unit, issues, verdict, disposition, photos, rep=None, call="tc1"):
    return (
        {"condition_id": cid, "catalog_item_id": item, "catalog_kind": kind, "scene_group": "living",
         "estimate_unit_id": unit, "issue_ids": issues},
        {"condition_id": cid, "verdict": verdict, "rationale": f"terra says {verdict}", "terra_call_id": call},
        {"condition_id": cid, "disposition": disposition, "reason_code": "route_work"},
        {"condition_id": cid, "photo_keys": photos, "representative_photo_keys": rep or photos,
         "distinct_photo_count": len(photos),
         "evidence_refs": [{"issue_id": i, "photo_key": photos[0], "observation": f"obs {i}"} for i in issues]},
    )


def _result(verdicts=("supported", "unsupported", "supported", "supported", "supported")):
    specs = [
        ("c1", "floor_worn", "degradation", "living_room_primary", ["i1"], verdicts[0], "accepted_for_work", ["photo_001.jpg"]),
        ("c2", "paint_dated", "modernization", "living_room_primary", ["i2"], verdicts[1], "excluded", ["photo_002.jpg"]),
        ("c3", "trim_dated", "modernization", "bedroom_1", ["i3"], verdicts[2], "accepted_for_work",
         ["photo_003.jpg", "photo_004.jpg"], ["photo_003.jpg"]),
        ("c4", "tile_cracked", "defect", "bathroom_primary", ["i4"], verdicts[3], "accepted_for_work", ["photo_005.jpg"]),
        ("c5", "floor_worn", "degradation", "bedroom_1", ["i5", "i6"], verdicts[4], "accepted_for_work", ["photo_006.jpg"]),
    ]
    rows = [_cond(*s) for s in specs]
    return {
        "observed_conditions": [r[0] for r in rows], "condition_reviews": [r[1] for r in rows],
        "condition_dispositions": [r[2] for r in rows], "evidence_facts": [r[3] for r in rows],
        "terra_calls": [{"call_id": "tc1", "condition_ids": ["c1", "c2", "c3", "c4", "c5"]}],
        "work_items": [
            {"work_item_id": "w1", "condition_ids": ["c1"], "catalog_item_ids": ["floor_worn"], "action_code": "FLOOR", "low": 100, "high": 500, "status": "active"},
            {"work_item_id": "w3", "condition_ids": ["c3"], "catalog_item_ids": ["trim_dated"], "action_code": "TRIM", "low": 50, "high": 300, "status": "active"},
            {"work_item_id": "w4", "condition_ids": ["c4"], "catalog_item_ids": ["tile_cracked"], "action_code": "TILE", "low": 80, "high": 400, "status": "active"},
            {"work_item_id": "w5", "condition_ids": ["c5"], "catalog_item_ids": ["floor_worn"], "action_code": "FLOOR", "low": 90, "high": 450, "status": "active"},
        ],
        "package_candidates": [{"package_candidate_id": "pk1", "package_type": "bedroom_refresh", "estimate_unit_id": "bedroom_1",
                                "child_work_item_ids": ["w3", "w5"], "driver_work_item_ids": ["w3"], "pricing_tier": "light", "low": 1000, "high": 3000}],
        "package_decisions": [{"package_candidate_id": "pk1", "decision": "approve", "rationale": "coherent"}],
        "package_applications": [{"package_candidate_id": "pk1", "status": "applied", "reason_code": "approved",
                                  "absorbed_work_item_ids": ["w3", "w5"], "unabsorbed_child_work_item_ids": [],
                                  "effective_low": 1000, "effective_high": 3000}],
    }


def _v4():
    return {
        "package_candidates": [
            {"package_id": "bedroom_refresh__bedroom_1", "package_type": "bedroom_refresh", "estimate_unit_id": "bedroom_1",
             "verification_status": "rejected", "confirmed_issue_ids": ["i3", "i5"], "rejected_issue_ids": ["i1", "i6"],
             "raw_pass_2f_response": json.dumps({"evidence_summary": "room looks fine"}), "review_photo_keys": ["photo_003.jpg"]},
            {"package_id": "living_refresh__living", "package_type": "living_refresh", "estimate_unit_id": "living_room_primary",
             "verification_status": "confirmed", "confirmed_issue_ids": ["i2"], "rejected_issue_ids": [],
             "raw_pass_2f_response": json.dumps({"evidence_summary": "paint is dated"})},
        ],
        "packages": [{"package_id": "bm_x", "package_type": "bathroom_modernization", "cost_low": 10, "cost_high": 20}],
        "room_surrogates": [{"room_surrogate_id": "bathroom_1", "scene_group": "bathroom", "photo_keys": ["photo_005.jpg"]},
                            {"room_surrogate_id": "bathroom_2", "scene_group": "bathroom", "photo_keys": ["photo_007.jpg"]}],
        "bathroom_expansion_audit": {"expanded": True, "qualifying_surrogate_ids": ["bathroom_1", "bathroom_2"],
                                     "produced_package_ids": ["bm_x"]},
    }


def _art(placement="root", result=None, prop="prop_a"):
    env = {"state": "complete", "provenance": {"architecture_mode": "new" if placement == "root" else "shadow"},
           "result": result or _result()}
    art = {"property": {"property_key": prop, "baths": 2}, "run": {"created_at": "2026-08-22T00:00:00Z"},
           "photos": {f"photo_{n:03d}.jpg": {"photo": {"photo_key": f"photo_{n:03d}.jpg", "image_path": f"{IMG}/photo_{n:03d}.jpg", "index": n}} for n in range(1, 8)},
           "renovation_estimate_v4": _v4()}
    if placement == "root":
        art["renovation_estimate_v5"] = env
    else:
        art["analysis_debug"] = {SHADOW_DEBUG_KEY: env}
    return art


def _listing(prop="prop_a", run_id="20260822_010101_deadbeef", replica=None, source="production"):
    return {"source": source, "property_key": prop, "run_id": run_id, "artifact": _art(prop=prop),
            "replica_artifact": replica, "path": None, "replica_path": None}


def _rec(verdict, source="production", strata=("dirA",), accepted=True, prop="p1", cid="cx",
         meta_extra=None, kind="condition", card_id=None, **rest):
    meta = {"condition_id": cid, "accepted": accepted, "catalog_item_id": "floor_worn",
            "catalog_kind": "degradation", "photo_count": 1, "second_opinion": "2f_objected",
            "terra_batch_conditions": 3, "terra_batch_images": 2, "terra_verdict": "supported"}
    meta.update(meta_extra or {})
    return {"card_id": card_id or f"rc_{cid}_{verdict[:6]}", "verdict": verdict, "tag": None, "notes": None,
            "peeked": False, "extra": rest.pop("extra", None),
            "card": {"card_id": card_id or f"rc_{cid}_{verdict[:6]}", "kind": kind, "source": source,
                     "strata": list(strata), "property_key": prop, "meta": meta,
                     "legacy_item_id": None}, **rest}


# ---------------------------------------------------------------------------

def test_verdict_audit_overwrites_orphans_undos(tmp_path):
    p = tmp_path / "v.jsonl"
    recs = [
        {"card_id": "a", "verdict": "terra_claim_supported", "tag": None, "notes": None, "peeked": False, "extra": None, "ts": "t1"},
        {"card_id": "a", "verdict": "terra_claim_supported", "tag": None, "notes": None, "peeked": False, "extra": None, "ts": "t2"},
        {"card_id": "b", "verdict": "terra_claim_supported", "tag": None, "notes": None, "peeked": False, "extra": None, "ts": "t3"},
        {"card_id": "b", "verdict": "terra_claim_overstated", "tag": None, "notes": None, "peeked": False, "extra": None, "ts": "t4"},
        {"card_id": "c", "verdict": "terra_claim_supported", "tag": None, "notes": None, "peeked": False, "extra": None, "ts": "t5"},
        {"card_id": "c", "verdict": None, "tag": None, "notes": None, "peeked": False, "extra": None, "ts": "t6"},
        {"card_id": "d", "verdict": "not_warranted", "tag": None, "notes": None, "peeked": True, "extra": None, "ts": "t7"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n" + json.dumps({"no_card_id": 1}) + "\n",
                 encoding="utf-8")
    au = ra.audit_verdicts(p)
    assert au["lines"] == 9 and au["invalid_lines"] == 2 and au["undos"] == 1 and au["peeked_records"] == 1
    assert au["distinct_ids"] == 4 and au["net_verdicts"] == 3  # c undone
    assert au["overwrites"]["identical"] == 1 and au["overwrites"]["changed"] == 1
    assert au["overwrites"]["changed_cards"][0]["card_id"] == "b"
    assert au["latest_crosscheck_ok"] and au["latest"] == rc.latest_verdicts(p)
    assert au["latest"]["b"]["verdict"] == "terra_claim_overstated"
    assert au["ts_range"] == ["t1", "t7"]


def test_weighted_estimate_matches_tally_oracle():
    dirA = [_rec("terra_claim_unsupported"), _rec("terra_claim_supported", cid="c2")]
    uni = [_rec("terra_claim_unsupported", strata=("uniform",), cid="c3"),
           _rec("terra_claim_supported", strata=("uniform",), cid="c4")]
    meta = {"accepted": 4, "accepted_dirA": 2, "accepted_non_dirA": 2}
    est = ra.weighted_estimate(2, 2, 1, 2, 1, 2)
    row = review_tally.weighted(meta, dirA, uni, ("terra_claim_unsupported",))
    assert f"{100 * est['point']:.1f}%" == row[-1] == "50.0%"
    assert [est["N"], est["N_A"], est["N_nonA"]] == [row[0], row[1], row[4]]
    assert est["flags"] == [] and est["r_A"] == est["r_U"] == 0.5


def test_wilson_propagation_and_degenerate():
    from tools.quant_artifact_comparison import wilson
    est = ra.weighted_estimate(10, 90, 2, 10, 9, 30)
    ci = wilson(9, 30)
    assert est["point"] == pytest.approx((10 * 0.2 + 90 * 0.3) / 100)
    assert est["low"] == pytest.approx((10 * 0.2 + 90 * ci["low"]) / 100)
    assert est["high"] == pytest.approx((10 * 0.2 + 90 * ci["high"]) / 100)
    # no uniform coverage
    est = ra.weighted_estimate(10, 90, 2, 10, 0, 0)
    assert est["point"] is None and "no_uniform_coverage" in est["flags"]
    # census-only subgroup
    est = ra.weighted_estimate(10, 0, 2, 10, 0, 0)
    assert est["point"] == est["low"] == est["high"] == pytest.approx(0.2)
    # census gap flagged
    est = ra.weighted_estimate(10, 0, 2, 8, 0, 0)
    assert "census_gap" in est["flags"]
    # empty
    assert ra.weighted_estimate(0, 0, 0, 0, 0, 0)["point"] is None


def test_subgroup_weighting_partition_and_contribution():
    pop = ([{"source": "production", "property_key": "p1", "catalog_item_id": "floor_worn", "catalog_kind": "X",
             "photo_count": 1, "second_opinion": "2f_objected", "terra_batch_conditions": 1,
             "terra_batch_images": 1, "is_dirA": d, "uniform_pick": False} for d in (True, True, False, False)]
           + [{"source": "production", "property_key": "p2", "catalog_item_id": "tile_cracked", "catalog_kind": "Y",
               "photo_count": 3, "second_opinion": "2f_agreed", "terra_batch_conditions": 9,
               "terra_batch_images": 9, "is_dirA": d, "uniform_pick": False} for d in (True, False)])
    meta = {"production": {"accepted": 6, "accepted_dirA": 3, "accepted_non_dirA": 3}}
    cond = [
        _rec("terra_claim_unsupported", prop="p1", cid="a1", meta_extra={"catalog_kind": "X"}),
        _rec("terra_claim_supported", prop="p1", cid="a2", meta_extra={"catalog_kind": "X"}),
        _rec("terra_claim_supported", prop="p2", cid="a3", meta_extra={"catalog_kind": "Y", "second_opinion": "2f_agreed", "photo_count": 3, "terra_batch_conditions": 9, "terra_batch_images": 9}),
        _rec("terra_claim_unsupported", strata=("uniform",), prop="p1", cid="u1", meta_extra={"catalog_kind": "X"}),
        _rec("terra_claim_supported", strata=("uniform",), prop="p2", cid="u2", meta_extra={"catalog_kind": "Y", "second_opinion": "2f_agreed", "photo_count": 3, "terra_batch_conditions": 9, "terra_batch_images": 9}),
    ]
    sub = ra.subgroups(pop, cond, meta)
    kinds = sub["production"]["catalog_kind"]
    assert kinds["partition"]["ok"]
    x, y = kinds["labels"]["X"], kinds["labels"]["Y"]
    assert (x["N"], x["N_A"], x["N_nonA"], x["judged"]) == (4, 2, 2, 3)
    # X: r_A = 1/2, r_U = 1/1 -> weighted (2*0.5 + 2*1.0)/4 = 0.75; mass 3.0
    assert x["broad"]["point"] == pytest.approx(0.75)
    # Y: r_A = 0/1, r_U = 0/1 -> 0; mass 0
    assert y["broad"]["point"] == 0.0
    assert x["contribution_share"] == pytest.approx(1.0) and y["contribution_share"] == 0.0
    assert kinds["contribution_complete"]
    assert "exploratory_small_n" in x["flags"] and "few_listings" in x["flags"]


def test_exploratory_flag_boundaries():
    n = 10
    pop = [{"source": "production", "property_key": f"p{i % 3}", "catalog_item_id": "it", "catalog_kind": "X",
            "photo_count": 1, "second_opinion": "2f_objected", "terra_batch_conditions": 1,
            "terra_batch_images": 1, "is_dirA": True, "uniform_pick": False} for i in range(n)]
    meta = {"production": {"accepted": n, "accepted_dirA": n, "accepted_non_dirA": 0}}
    cond = [_rec("terra_claim_supported", prop=f"p{i % 3}", cid=f"c{i}", meta_extra={"catalog_kind": "X"})
            for i in range(n)]
    lab = ra.subgroups(pop, cond, meta)["production"]["catalog_kind"]["labels"]["X"]
    assert lab["judged"] == 10 and lab["listings"] == 3 and lab["flags"] == []
    lab9 = ra.subgroups(pop[:9], cond[:9],
                        {"production": {"accepted": 9, "accepted_dirA": 9, "accepted_non_dirA": 0}}
                        )["production"]["catalog_kind"]["labels"]["X"]
    assert "exploratory_small_n" in lab9["flags"]
    cond2 = [_rec("terra_claim_supported", prop=f"p{i % 2}", cid=f"c{i}", meta_extra={"catalog_kind": "X"})
             for i in range(n)]
    lab2 = ra.subgroups(pop, cond2, meta)["production"]["catalog_kind"]["labels"]["X"]
    assert "few_listings" in lab2["flags"]


def test_stratum_overlap_and_arm_membership():
    both = _rec("terra_claim_supported", strata=("dirA", "terra_flip"), cid="f1")
    dirb = _rec("terra_claim_supported", strata=("dirB",), accepted=False, cid="b1",
                meta_extra={"terra_verdict": "unsupported"})
    not_acc = _rec("terra_claim_supported", strata=("dirA",), accepted=False, cid="na")
    dirA, uni = ra.arm_records([both, dirb, not_acc], "production")
    assert dirA == [both] and uni == []  # flip overlap counted once; dirB and non-accepted dirA never enter


def test_comparable_pairs_mirror_rc():
    r1 = _result()
    r2 = _result(verdicts=("unsupported", "unsupported", "supported", "supported", "supported"))
    pairs = ra.pair_stats(r1, r2)
    assert len(pairs) == 5 and sum(p["flipped"] for p in pairs) == 1 == len(rc.terra_flips(r1, r2))
    assert {p["condition_id"] for p in pairs if p["flipped"]} == {"c1"}
    r2b = _result(verdicts=("unsupported", "unsupported", "supported", "supported", "supported"))
    r2b["evidence_facts"][0]["photo_keys"] = ["photo_009.jpg"]
    pairs = ra.pair_stats(r1, r2b)
    assert len(pairs) == 4 and sum(p["flipped"] for p in pairs) == 0 == len(rc.terra_flips(r1, r2b))
    r2c = _result(verdicts=("unsupported", "unsupported", "supported", "supported", "supported"))
    dup = dict(r2c["observed_conditions"][0], condition_id="c1x")
    r2c["observed_conditions"].append(dup)
    r2c["evidence_facts"].append(dict(r2c["evidence_facts"][0], condition_id="c1x"))
    pairs = ra.pair_stats(r1, r2c)
    assert len(pairs) == 4 and sum(p["flipped"] for p in pairs) == 0 == len(rc.terra_flips(r1, r2c))
    assert ra.pair_stats(r1, None) == []


def test_flip_match_truth_table():
    m = {"terra_verdict": "supported", "replica_terra_verdict": "unsupported"}
    assert ra.flip_match("terra_claim_supported", m) == "run_1"
    assert ra.flip_match("terra_claim_unsupported", m) == "run_2"
    assert ra.flip_match("terra_claim_overstated", m) == "n/a"
    assert ra.flip_match("terra_claim_unsupported", {"terra_verdict": "supported", "replica_terra_verdict": "supported"}) == "neither"
    assert ra.flip_match(rc.CONDITION_VERDICTS[3], m) == "n/a"


def test_label_mapping_and_dirb_split():
    assert ra.MEASURES["hard_false"] == ("terra_claim_unsupported",)
    assert set(ra.MEASURES["broad"]) == {"terra_claim_unsupported", "terra_claim_overstated"}
    assert review_tally.terra_vs_photo("terra_claim_supported", "unsupported") == "miss"
    cond = [
        _rec("terra_claim_supported", strata=("dirB",), accepted=False, cid="b1", meta_extra={"terra_verdict": "unsupported"}),
        _rec("terra_claim_overstated", strata=("dirB",), accepted=False, cid="b2", meta_extra={"terra_verdict": "unsupported"}),
        _rec("terra_claim_supported", strata=("dirB",), accepted=False, cid="b3", meta_extra={"terra_verdict": "cannot_assess"}),
        _rec("terra_claim_supported", strata=("dirA",), cid="a1"),  # not dirB -> excluded
    ]
    d = ra.dirb_recovery(cond)
    assert d["production"]["unsupported"]["n"] == 2 and d["production"]["unsupported"]["supported"] == 1
    assert d["production"]["cannot_assess"]["n"] == 1 and d["all"]["unsupported"]["overstated"] == 1


def test_lineage_billing_and_driver_class():
    res = _result()
    idx = rc.index_result(res)
    amap = ra.absorbed_map(res)
    assert amap == {"w3": "package_driver", "w5": "package_support"}
    assert ra.billing_path("c3", idx, amap) == "package_driver"
    assert ra.billing_path("c5", idx, amap) == "package_support"
    assert ra.billing_path("c4", idx, amap) == "standalone"
    assert ra.billing_path("c2", idx, amap) == "no_active_work_item"
    # a non-applied application absorbs nothing
    res2 = _result()
    res2["package_applications"][0]["status"] = "skipped"
    assert ra.absorbed_map(res2) == {}
    ctx = {("production", "prop_a"): {"res": res, "idx": idx}}
    card = {"source": "production", "property_key": "prop_a",
            "meta": {"package_type": "bedroom_refresh", "estimate_unit_id": "bedroom_1"}}
    dc = ra.driver_class(card, ctx, KINDS)
    assert dc["class"] == "style_only" and dc["driver_items"] == ["trim_dated"]
    dc = ra.driver_class(card, ctx, {})
    assert dc["class"] == "unknown" and dc["unknown_items"] == ["trim_dated"]
    res3 = _result()
    res3["package_candidates"][0]["driver_work_item_ids"] = ["w3", "w5"]
    ctx3 = {("production", "prop_a"): {"res": res3, "idx": rc.index_result(res3)}}
    assert ra.driver_class(card, ctx3, KINDS)["class"] == "includes_damage_or_degradation"


def test_gate_exposure_weights():
    res = _result()
    idx = rc.index_result(res)
    ctx = {("production", "p1"): {"idx": idx, "amap": {"w3": "package_driver", "w5": "package_support"}}}
    cond = [
        _rec("terra_claim_unsupported", cid="c1"),                       # dirA error, w1 -> standalone, weight 1
        _rec("terra_claim_overstated", strata=("uniform",), cid="c3"),   # uniform error -> driver, weight 4/2
        _rec("terra_claim_supported", strata=("uniform",), cid="c4"),    # uniform non-error
    ]
    meta = {"production": {"accepted": 5, "accepted_dirA": 1, "accepted_non_dirA": 4}}
    g = ra.gate_exposure(cond, ctx, meta)["production"]
    assert g["uniform_weight"] == pytest.approx(2.0)
    assert g["mass"]["standalone"] == pytest.approx(1.0)
    assert g["mass"]["package_driver"] == pytest.approx(2.0)
    assert g["share_packaged"] == pytest.approx(2.0 / 3.0)
    assert g["n_error_records"] == 2 and "exploratory_small_n" in g["flags"]


def test_packaging_flow_composition():
    res = _result()
    idx = rc.index_result(res)
    ctx = {("production", "p1"): {"idx": idx, "amap": ra.absorbed_map(res)}}
    cond = [
        _rec("terra_claim_supported", cid="c3"),
        _rec("terra_claim_unsupported", strata=("uniform",), cid="c4"),
        _rec("terra_claim_supported", strata=("dirB",), accepted=False, cid="c2"),  # not accepted -> excluded
    ]
    f = ra.packaging_flow(cond, ctx)["production"]
    assert f["n"] == 2
    assert f["grid"]["terra_claim_supported"] == {"package_driver": 1}
    assert f["grid"]["terra_claim_unsupported"] == {"standalone": 1}


def test_bathroom_targets_and_gap():
    def bath_ctx(v5_applied, with_v4=True, v4_count=1):
        res = {"package_candidates": [{"package_candidate_id": f"bm{i}", "package_type": "bathroom_modernization"}
                                      for i in range(max(v5_applied, 1))],
               "package_applications": [{"package_candidate_id": f"bm{i}", "status": "applied"}
                                        for i in range(v5_applied)]}
        v4 = {"packages": [{"package_type": "bathroom_modernization"}] * v4_count} if with_v4 else {}
        return {"res": res, "v4": v4}

    recs = [
        _rec("per_bathroom", kind="bathroom", prop="pA", cid="bA", extra={"distinct_bathrooms": 3},
             meta_extra={"listing_baths": 3, "surrogates": 3, "v5_bath_units": ["bathroom_primary"], "v4_expanded": False}),
        _rec("once", kind="bathroom", prop="pB", cid="bB", extra={"distinct_bathrooms": 1},
             meta_extra={"listing_baths": 1, "surrogates": 2, "v5_bath_units": ["bathroom_primary"], "v4_expanded": False}),
        _rec("unsure", kind="bathroom", prop="pC", cid="bC", extra={"distinct_bathrooms": 2},
             meta_extra={"listing_baths": 2, "surrogates": 2, "v5_bath_units": ["bathroom_primary"], "v4_expanded": False}),
        _rec("per_bathroom", kind="bathroom", prop="pD", cid="bD", extra={"distinct_bathrooms": 2},
             meta_extra={"listing_baths": 2, "surrogates": 2, "v5_bath_units": ["bathroom_primary"], "v4_expanded": False}),
    ]
    ctx = {("production", "pA"): bath_ctx(1, v4_count=0), ("production", "pB"): bath_ctx(1),
           ("production", "pC"): bath_ctx(1), ("production", "pD"): bath_ctx(2, with_v4=False)}
    b = ra.bathrooms_block(recs, ctx)
    by = {c["property_key"]: c for c in b["cards"]}
    assert by["pA"]["relation_v5"] == "under" and by["pA"]["relation_v4"] == "under"
    assert by["pB"]["relation_v5"] == "exact" and by["pB"]["relation_v4"] == "exact"
    assert by["pC"]["relation_v5"] == "excluded_unsure"
    assert by["pD"]["relation_v5"] == "exact" and by["pD"]["relation_v4"] == "v4_unrecoverable" and by["pD"]["v4_count"] is None
    assert b["unit_gap_v5"] == 2  # pA: 3-1; pB 0; pD 0
    assert b["unit_gap_v4"] == 3  # pA: 3-0; pB 0; pD not recoverable
    assert b["aggregate"]["excluded_unsure"] == 1 and b["aggregate"]["v4_unrecoverable"] == 1


def test_notes_coding_raises_on_uncoded():
    cards = {"rc_known": {"card_id": "rc_known", "kind": "condition", "title": "t"}}
    latest = {"rc_known": {"verdict": "terra_claim_supported", "tag": None, "notes": "a note", "extra": None}}
    with pytest.raises(ValueError, match="no NOTE_THEMES coding"):
        ra.qualitative(latest, cards)
    assert set(ra.NOTE_THEMES.values()) <= set(ra.THEMES)


def _fixture_world(tmp_path):
    listings = [
        _listing(prop="prop_a", source="canary",
                 replica=_art(result=_result(verdicts=("unsupported", "unsupported", "supported", "supported", "supported")))),
        _listing(prop="prop_b", source="production", run_id="20260823_010101_cafecafe"),
    ]
    queue = rc.build_queue_from_listings(listings, uniform_rate=1.0, packets={}, claims=CLAIMS)
    verdicts = tmp_path / "v.jsonl"
    lines = []
    for c in queue["cards"]:
        if c["kind"] == "condition":
            v, extra = "terra_claim_unsupported" if c["meta"]["condition_id"] == "c1" else "terra_claim_supported", None
        elif c["kind"] == "package":
            v, extra = "not_warranted", None
        else:
            v, extra = "per_bathroom", {"distinct_bathrooms": 2}
        lines.append(json.dumps({"card_id": c["card_id"], "verdict": v, "tag": None, "notes": None,
                                 "peeked": False, "extra": extra, "ts": "2026-08-26T00:00:00"}))
    verdicts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return listings, queue, verdicts


def test_analyze_end_to_end_on_fixtures(tmp_path):
    listings, queue, verdicts = _fixture_world(tmp_path)
    a = ra.analyze(queue, verdicts, listings, claims=CLAIMS, kind_of=KINDS)
    assert a["integrity"]["ok"], a["integrity"]
    assert a["integrity"]["reconciliation"]["ok"] and a["integrity"]["completion_ok"]
    assert a["integrity"]["orphans"] == [] and all(c["ok"] for c in a["integrity"]["spot_checks"] if c["card_id"])
    # canary: c1 flips between replicas; 4 comparable pairs survive the changed-photo... all 5 comparable here
    assert a["stability"]["comparable_pairs"] == 5 and a["stability"]["flips"] == 1
    assert a["stability"]["flips"] == a["stability"]["flips_rc_crosscheck"]
    # weighted headline: canary dirA = c1 (unsupported), c5 (supported); uniform = c3, c4 supported
    e = a["condition_truth"]["canary"]["weighted"]["hard_false"]
    assert (e["N_A"], e["n_A"], e["x_A"]) == (2, 2, 1) and e["r_U"] == 0.0
    assert e["point"] == pytest.approx((2 * 0.5 + 2 * 0.0) / 4)
    md = ra.render_md(a)
    for header in ("## 0. Scope", "## 1. Freeze", "## 3. Accepted-condition truth", "## 6. Terra stability",
                   "## 7. Package warrant", "## 10. Multi-bathroom", "## 12. What this analysis does not show"):
        assert header in md
    assert md.count("*Evidence:*") >= 8 and "recommend" not in md.lower()
    json.dumps(a)  # payload must be JSON-serializable


def test_population_matches_queue_meta(tmp_path):
    listings, queue, _ = _fixture_world(tmp_path)
    rows, enum_meta, ctx = ra.enumerate_population(listings, queue["uniform_rate"])
    recon = ra.reconcile(enum_meta, queue["meta"])
    assert recon["ok"], recon
    assert len(rows) == sum(m["accepted"] for m in queue["meta"].values())
    assert sum(1 for r in rows if r["uniform_pick"]) == sum(
        (m.get("strata") or {}).get("uniform", 0) for m in queue["meta"].values())


def test_population_low_res_parity(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    small, big = tmp_path / "small.jpg", tmp_path / "big.jpg"
    PIL.new("RGB", (390, 260)).save(small)
    PIL.new("RGB", (1280, 853)).save(big)
    art = _art()
    for key, p in {"photo_001.jpg": small, "photo_005.jpg": small, "photo_007.jpg": small,
                   "photo_002.jpg": big, "photo_003.jpg": big, "photo_004.jpg": big, "photo_006.jpg": big}.items():
        art["photos"][key]["photo"]["image_path"] = str(p)
    listing = {"source": "production", "property_key": "prop_a", "run_id": "20260822_010101_deadbeef",
               "artifact": art, "replica_artifact": None, "path": None, "replica_path": None}
    queue = rc.build_queue_from_listings([listing], uniform_rate=1.0, packets={}, claims=CLAIMS)
    _rows, enum_meta, _ctx = ra.enumerate_population([listing], 1.0)
    assert enum_meta["production"]["low_res_excluded"] == {"conditions": 2, "accepted": 2, "bathrooms": 1}
    assert ra.reconcile(enum_meta, queue["meta"])["ok"]


def test_orphan_resolution_from_listings(tmp_path):
    listings, queue, verdicts = _fixture_world(tmp_path)
    known = queue["cards"][0]
    latest = {known["card_id"]: {"verdict": "terra_claim_supported", "ts": "t"},
              "rc_ghost0000000": {"verdict": "unsure", "ts": "t"}}
    out = ra.resolve_orphans([known["card_id"], "rc_ghost0000000"], listings, latest)
    by = {o["card_id"]: o for o in out}
    assert by[known["card_id"]]["classification"] == "excluded_low_res"  # resolves to a current listing
    assert by[known["card_id"]]["property_key"] == known["property_key"]
    assert by["rc_ghost0000000"]["classification"] == "unresolved"


# ---------------------------------------------------------------------------

REAL_INPUTS = (rc.CANARY_ROOT / "run_1").is_dir() and rc.PROD_ROOT.is_dir() and ra.QUEUE.is_file() and ra.VERDICTS.is_file()


@pytest.mark.skipif(not REAL_INPUTS, reason="frozen review inputs / artifacts not present")
def test_full_run_against_frozen_inputs(tmp_path):
    queue = json.loads(ra.QUEUE.read_text(encoding="utf-8"))
    listings = ra.load_listings()
    a = ra.analyze(queue, ra.VERDICTS, listings)
    assert a["integrity"]["ok"], {k: v for k, v in a["integrity"].items() if k not in ("manifest", "verdict_audit")}
    assert a["integrity"]["verdict_audit"]["peeked_records"] == 0
    orphans = a["integrity"]["orphans"]
    assert len(orphans) == 11 and all(o["classification"] == "excluded_low_res" for o in orphans)
    assert {o["property_key"] for o in orphans} == {"redfin_10965375"}
    assert {o["run_id"] for o in orphans} == {"20260821_233007_c5b989ee"}
    assert a["stability"]["flips"] == 16 == a["stability"]["flips_rc_crosscheck"]
    assert a["stability"]["comparable_pairs"] >= 16
    assert a["integrity"]["no_eligible_evidence_listings"] == ["redfin_10922002", "redfin_10965375"]
    assert len(a["per_card"]) == 212 and all(pc["verdict"] for pc in a["per_card"])
    # outputs go only where told; this test writes nothing to reports/
    (tmp_path / "a.json").write_text(json.dumps(a, sort_keys=True), encoding="utf-8")
