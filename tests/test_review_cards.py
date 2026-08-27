"""tools/review_cards + scripts/review_tally + scripts/review_server (stdlib review tooling)."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import review_server, review_tally
from tools import review_cards as rc
from tools.renovation_architecture.contracts import SHADOW_DEBUG_KEY

CLAIMS = {"floor_worn": "hard flooring scratched or worn", "paint_dated": "wall paint dated colour",
          "trim_dated": "interior trim dated style", "tile_cracked": "tile cracked"}
IMG = "C:/img/prop"


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
        "package_applications": [{"package_candidate_id": "pk1", "status": "applied", "reason_code": "approved", "effective_low": 1000, "effective_high": 3000}],
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
        "packages": [],
        "room_surrogates": [{"room_surrogate_id": "bathroom_1", "scene_group": "bathroom", "photo_keys": ["photo_005.jpg"]},
                            {"room_surrogate_id": "bathroom_2", "scene_group": "bathroom", "photo_keys": ["photo_007.jpg"]}],
        "bathroom_expansion_audit": {"expanded": True, "qualifying_surrogate_ids": ["bathroom_1", "bathroom_2"]},
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
    return {"source": source, "property_key": prop, "run_id": run_id, "artifact": _art(prop=prop), "replica_artifact": replica}


# ---------------------------------------------------------------------------

def test_v5_result_placements():
    assert rc.v5_result(_art("root")) is not None
    assert rc.v5_result(_art("shadow")) is not None
    bad = _art("root")
    bad["renovation_estimate_v5"]["state"] = "failed"
    assert rc.v5_result(bad) is None
    assert rc.v5_result(None) is None


def test_iter_runs_filters_before_loading(tmp_path):
    root = tmp_path
    (root / ".renovation_architecture").mkdir()
    (root / ".renovation_architecture" / "sol_usage.sqlite3").write_bytes(b"x")
    (root / "prop_a" / "20260801_120000_aaaaaaaa").mkdir(parents=True)  # old run
    (root / "prop_a" / "20260801_120000_aaaaaaaa" / "photo_intel_debug.json").write_text(json.dumps(_art()), encoding="utf-8")
    (root / "prop_a" / "20260822_010101_deadbeef").mkdir()
    (root / "prop_a" / "20260822_010101_deadbeef" / "photo_intel_debug.json").write_text(json.dumps(_art()), encoding="utf-8")
    (root / "prop_a" / "20260822_020202_beefbeef").mkdir()  # newer but empty
    (root / "prop_b" / "notarun").mkdir(parents=True)
    (root / "prop_b" / "notarun" / "photo_intel_debug.json").write_text("{}", encoding="utf-8")
    runs = rc.iter_runs(root, "20260821_230000")
    assert [(p, r) for p, r, _, _ in runs] == [("prop_a", "20260822_010101_deadbeef")]
    assert len(rc.iter_runs(root, None)) == 1  # still the newest complete run


def test_join_and_direction():
    res = _result()
    idx = rc.index_result(res)
    f2 = rc.join_2f(_v4(), idx)
    assert sorted(f2) == ["c1", "c2", "c3", "c5"]
    v = {cid: {r[1] for r in f2[cid]["rows"]} for cid in f2}
    assert rc.direction("supported", v["c1"]) == "A"
    assert rc.direction("unsupported", v["c2"]) == "B"
    assert rc.direction("supported", v["c3"]) == "agree"
    assert rc.direction("supported", v["c5"]) == "mixed"
    assert rc.direction("supported", set()) == "none"
    assert rc.direction("unsupported", {"rejected"}) == "agree"
    assert rc.direction(None, {"confirmed"}) == "none"
    assert len(f2["c5"]["rows"]) == 2 and f2["c5"]["evidence"] == ["room looks fine"]


def test_terra_flips_key_rule():
    r1 = _result()
    r2 = _result(verdicts=("unsupported", "unsupported", "supported", "supported", "supported"))
    assert rc.terra_flips(r1, r2) == [("c1", "supported", "unsupported")]
    # different photo set -> not comparable
    r2b = _result(verdicts=("unsupported", "unsupported", "supported", "supported", "supported"))
    r2b["evidence_facts"][0]["photo_keys"] = ["photo_009.jpg"]
    assert rc.terra_flips(r1, r2b) == []
    # duplicate key within a run -> ignored
    r2c = _result(verdicts=("unsupported", "unsupported", "supported", "supported", "supported"))
    dup = dict(r2c["observed_conditions"][0], condition_id="c1x")
    r2c["observed_conditions"].append(dup)
    r2c["evidence_facts"].append(dict(r2c["evidence_facts"][0], condition_id="c1x"))  # same photo set -> same key
    assert rc.terra_flips(r1, r2c) == []
    assert rc.terra_flips(r1, None) == []


def test_terra_batch():
    idx = rc.index_result(_result())
    assert rc.terra_batch(idx, "c1") == (5, 5)  # 5 conditions, 5 distinct representative images (photo_004 is a dedup'd duplicate)
    idx["revs"]["c1"]["terra_call_id"] = "missing"
    assert rc.terra_batch(idx, "c1") == (None, None)


def test_card_id_and_uniform_sampling():
    a = rc.card_id("production", "prop_a", "run1", "condition", "c1")
    assert a == rc.card_id("production", "prop_a", "run1", "condition", "c1")
    assert a != rc.card_id("production", "prop_a", "run2", "condition", "c1")
    ids = [rc.card_id("s", "p", "r", "condition", f"c{i}") for i in range(400)]
    low = {i for i in ids if rc.uniform_included(i, 0.07)}
    high = {i for i in ids if rc.uniform_included(i, 0.10)}
    assert low <= high and 5 <= len(low) <= 60  # monotone; roughly 7 %
    assert not any(rc.uniform_included(i, 0.0) for i in ids)
    assert all(rc.uniform_included(i, 1.0) for i in ids)


def test_build_queue_strata_phase_and_blind():
    q = rc.build_queue_from_listings([_listing(replica=_art(result=_result(verdicts=("unsupported", "unsupported", "supported", "supported", "supported"))))],
                                     uniform_rate=1.0, packets={}, claims=CLAIMS)
    by = {c["meta"].get("condition_id") or c["kind"]: c for c in q["cards"]}
    assert by["c1"]["strata"] == ["dirA", "terra_flip"] and by["c1"]["phase"] == 1
    assert by["c2"]["strata"] == ["dirB"] and by["c2"]["phase"] == 2
    assert by["c3"]["strata"] == ["uniform"] and by["c4"]["strata"] == ["uniform"]
    assert by["c5"]["strata"] == ["dirA"]  # mixed counts as dirA, never uniform
    assert by["package"]["strata"] == ["p1_package"] and by["package"]["phase"] == 3
    assert by["bathroom"]["strata"] == ["p3_bathroom"] and len(by["bathroom"]["strips"]) == 2
    m = q["meta"]["production"]
    assert (m["accepted"], m["accepted_dirA"], m["accepted_non_dirA"]) == (4, 2, 2)
    # claim = catalog text + observations; strips = representative keys, duplicates muted
    c3 = by["c3"]
    assert c3["claim"] == {"catalog_claim": "interior trim dated style", "observations": ["obs i3"]}
    assert [p["key"] for p in c3["strips"][0]["photos"]] == ["photo_003.jpg"]
    assert c3["strips"][1]["muted"] and [p["key"] for p in c3["strips"][1]["photos"]] == ["photo_004.jpg"]
    assert c3["strips"][0]["photos"][0]["path"].endswith("photo_003.jpg")
    assert by["c1"]["meta"]["replica_terra_verdict"] == "unsupported"
    assert any(r["label"].startswith("Terra verdict, replica 2") for r in by["c1"]["reveal"])
    # blind view hides model text, strata, prices, direction
    b = rc.blind_card(by["c1"])
    assert "reveal" not in b and "hidden" not in b and "strata" not in b
    assert not set(rc.BLIND_META) & set(b["meta"]) and b["meta"]["catalog_item_id"] == "floor_worn"
    r = rc.reveal_payload(by["c1"])
    assert r["strata"] == ["dirA", "terra_flip"] and r["hidden"] == {"low": 100, "high": 500}
    # phase 1 hash order interleaves; tier order puts dirA first
    q2 = rc.build_queue_from_listings([_listing()], uniform_rate=1.0, packets={}, claims=CLAIMS, order="tier")
    assert [c["phase"] for c in q2["cards"]] == sorted(c["phase"] for c in q2["cards"])
    assert "dirA" in q2["cards"][0]["strata"]
    assert rc.order_cards(q2["cards"], "hash") == rc.order_cards(rc.order_cards(q2["cards"], "hash")[::-1], "hash")


def test_uniform_picks_are_listing_independent():
    one = rc.build_queue_from_listings([_listing("prop_a")], uniform_rate=0.5, packets={}, claims=CLAIMS)
    two = rc.build_queue_from_listings([_listing("prop_a"), _listing("prop_b")], uniform_rate=0.5, packets={}, claims=CLAIMS)
    pick = lambda q: {c["card_id"] for c in q["cards"] if c["property_key"] == "prop_a" and "uniform" in c["strata"]}
    assert pick(one) == pick(two)


def test_legacy_ids_and_p6():
    packets = {"p2_contradictions": [{"property": "prop_a", "condition_id": "c1", "item_id": "C001"}],
               "p1_packages": [{"property": "prop_a", "package": "bedroom_refresh|bedroom_1", "item_id": "P01"}],
               "p3_multi_bath": [{"property": "prop_a", "item_id": "B01"}],
               "p6_tier2_zero_items": [{"property": "prop_a", "item": "tile_cracked|bathroom_primary", "review_id": "rar1_x"}]}
    q = rc.build_queue_from_listings([_listing(source="canary")], uniform_rate=0.0, packets=packets, claims=CLAIMS)
    by = {c["meta"].get("condition_id") or c["kind"]: c for c in q["cards"]}
    assert by["c1"]["legacy_item_id"] == "C001" and by["package"]["legacy_item_id"] == "P01" and by["bathroom"]["legacy_item_id"] == "B01"
    assert by["c4"]["strata"] == ["p6_forced_single"] and by["c4"]["legacy_item_id"] == "rar1_x" and by["c4"]["phase"] == 3


def test_latest_verdicts_and_undo(tmp_path):
    p = tmp_path / "v.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"card_id": "a", "verdict": "terra_claim_supported"}, {"card_id": "a", "verdict": "terra_claim_unsupported"},
        {"card_id": "b", "verdict": "terra_claim_supported"}, {"card_id": "b", "verdict": None}, "not json"]) + "\n", encoding="utf-8")
    got = rc.latest_verdicts(p)
    assert set(got) == {"a"} and got["a"]["verdict"] == "terra_claim_unsupported"
    assert rc.latest_verdicts(tmp_path / "missing.jsonl") == {}


def test_tally_terra_vs_photo_and_weighted(tmp_path):
    q = rc.build_queue_from_listings([_listing()], uniform_rate=1.0, packets={}, claims=CLAIMS)
    by = {c["meta"].get("condition_id") or c["kind"]: c for c in q["cards"]}
    done = {by["c1"]["card_id"]: {"card_id": by["c1"]["card_id"], "verdict": "terra_claim_unsupported", "tag": "stain_or_colour_read_as_wear"},
            by["c5"]["card_id"]: {"card_id": by["c5"]["card_id"], "verdict": "terra_claim_supported"},
            by["c3"]["card_id"]: {"card_id": by["c3"]["card_id"], "verdict": "terra_claim_supported"},
            by["c4"]["card_id"]: {"card_id": by["c4"]["card_id"], "verdict": "terra_claim_unsupported", "peeked": True},
            by["c2"]["card_id"]: {"card_id": by["c2"]["card_id"], "verdict": "terra_claim_supported"},
            by["package"]["card_id"]: {"card_id": by["package"]["card_id"], "verdict": "not_warranted"},
            by["bathroom"]["card_id"]: {"card_id": by["bathroom"]["card_id"], "verdict": "once", "extra": {"distinct_bathrooms": 1}}}
    assert review_tally.terra_vs_photo("terra_claim_unsupported", "supported") == "false_positive"
    assert review_tally.terra_vs_photo("terra_claim_supported", "unsupported") == "miss"
    assert review_tally.terra_vs_photo("terra_claim_supported", "supported") == "agree"
    assert review_tally.terra_vs_photo("terra_claim_overstated", "supported") == "overstated"
    # weighted: N_A=2 (r_A = 1/2), N_nonA=2 (r = 1/2) -> overall 50.0%
    row = review_tally.weighted(q["meta"]["production"],
                                [done[by["c1"]["card_id"]], done[by["c5"]["card_id"]]],
                                [done[by["c3"]["card_id"]], done[by["c4"]["card_id"]]], ("terra_claim_unsupported",))
    assert row == [4, 2, "2", "50.0%", 2, "2", "50.0%", "50.0%"]
    report = review_tally.build_report(q, done)
    assert "Terra false positive" in report and "stain_or_colour_read_as_wear" in report and "2f_correct" not in report
    assert "peeked): 1 of 7" in report
    # legacy export fills a copy only
    ws = tmp_path / "ws.json"
    ws.write_text(json.dumps({"instructions": {}, "p2_conditions": {"C001": {"ref": "x", "verdict": "", "notes": ""}},
                              "p1_packages": {"P01": {"ref": "y", "verdict": "", "notes": ""}},
                              "p3_bathrooms": {"B01": {"ref": "z", "distinct_bathrooms": "", "bill": "", "notes": ""}}}), encoding="utf-8")
    before = ws.read_text(encoding="utf-8")
    by["c1"]["legacy_item_id"], by["package"]["legacy_item_id"], by["bathroom"]["legacy_item_id"] = "C001", "P01", "B01"
    out = tmp_path / "ws.filled.json"
    assert review_tally.export_legacy(q, done, ws, out) == 3
    filled = json.loads(out.read_text(encoding="utf-8"))
    assert filled["p2_conditions"]["C001"] == {"ref": "x", "verdict": "terra_claim_unsupported", "notes": "[stain_or_colour_read_as_wear]"}
    assert filled["p1_packages"]["P01"]["verdict"] == "not_warranted"
    assert filled["p3_bathrooms"]["B01"]["distinct_bathrooms"] == 1 and filled["p3_bathrooms"]["B01"]["bill"] == "once"
    assert "terra_correct" not in filled["instructions"]["p2_conditions"]
    assert ws.read_text(encoding="utf-8") == before


def test_server_image_tokens_and_public_cards(tmp_path):
    q = rc.build_queue_from_listings([_listing()], uniform_rate=1.0, packets={}, claims=CLAIMS)
    toks = review_server.image_tokens(q["cards"])
    assert all(isinstance(p, Path) for p in toks.values()) and len(toks) == 7
    pub = review_server.public_cards(q["cards"])
    url = pub[0]["strips"][0]["photos"][0]["url"]
    assert url.startswith("/img/") and url.split("/")[-1] in toks
    assert all("path" not in p for c in pub for s in c["strips"] for p in s["photos"])
    assert all("reveal" not in c and "strata" not in c for c in pub)
    # a verdict round-trips through State.record into the JSONL
    assert all(p.get("path") for c in q["cards"] for s in c["strips"] for p in s["photos"])  # source cards not mutated
    qp, vp = tmp_path / "q.json", tmp_path / "v.jsonl"
    qp.write_text(json.dumps(q), encoding="utf-8")
    st = review_server.State(qp, vp)
    assert len(st.images) == 7 and all("path" in p for c in st.cards.values() for s in c["strips"] for p in s["photos"])
    assert url.split("/")[-1] in st.images
    cid = q["cards"][0]["card_id"]
    rec = st.record({"card_id": cid, "verdict": "terra_claim_supported", "notes": " ok ", "peeked": False})
    assert rec["notes"] == "ok" and rc.latest_verdicts(vp)[cid]["verdict"] == "terra_claim_supported"
    with pytest.raises(KeyError):
        st.record({"card_id": "nope", "verdict": "x"})


def test_low_res_evidence_excluded(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    small, big = tmp_path / "small.jpg", tmp_path / "big.jpg"
    PIL.new("RGB", (390, 260)).save(small)
    PIL.new("RGB", (1280, 853)).save(big)
    art = _art()
    for key, p in {"photo_001.jpg": small, "photo_005.jpg": small, "photo_007.jpg": small,
                   "photo_002.jpg": big, "photo_003.jpg": big, "photo_004.jpg": big, "photo_006.jpg": big}.items():
        art["photos"][key]["photo"]["image_path"] = str(p)
    q = rc.build_queue_from_listings(
        [{"source": "production", "property_key": "prop_a", "run_id": "r1", "artifact": art}],
        uniform_rate=1.0, packets={}, claims=CLAIMS)
    by = {c["meta"].get("condition_id") or c["kind"]: c for c in q["cards"]}
    assert "c1" not in by and "c4" not in by      # evidence entirely 390x260 -> no card
    assert "bathroom" not in by                    # both surrogate strips are thumbnails
    assert {"c2", "c3", "c5", "package"} <= set(by)
    assert by["c3"]["strips"][0]["photos"][0]["wh"] == [1280, 853]
    m = q["meta"]["production"]
    assert m["low_res_excluded"] == {"conditions": 2, "accepted": 2, "bathrooms": 1}
    # excluded accepted conditions leave the weighted-rate denominators
    assert (m["accepted"], m["accepted_dirA"], m["accepted_non_dirA"]) == (2, 1, 1)
    # unknown sizes never exclude
    assert not rc.all_low_res({"a": None}, ["a"]) and rc.all_low_res({"a": (390, 260)}, ["a", "missing"])


# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (rc.CANARY_ROOT / "run_1").is_dir() or not rc.PACKETS_PATH.is_file(),
                    reason="Session 9 canary artifacts / packets not present")
def test_canary_join_reproduces_session9_counts():
    run1, run2 = rc.load_canary(rc.CANARY_ROOT)
    rows = a = b = flips = 0
    for prop, (path, art) in run1.items():
        res = rc.v5_result(art)
        idx = rc.index_result(res)
        f2 = rc.join_2f(art["renovation_estimate_v4"], idx)
        for cid, rec in f2.items():
            t = (idx["revs"].get(cid) or {}).get("verdict")
            d = rc.direction(t, {r[1] for r in rec["rows"]})
            a += d in ("A", "mixed")
            b += d == "B"
            rows += sum(1 for r in rec["rows"] if (r[1] == "rejected" and t == "supported") or (r[1] == "confirmed" and t in ("unsupported", "cannot_assess")))
        flips += len(rc.terra_flips(res, rc.v5_result((run2.get(prop) or (None, None))[1])))
    assert (rows, a, b, flips) == (85, 32, 36, 16)
