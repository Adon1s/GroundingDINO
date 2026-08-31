"""The v1.1 label repair: schema tables, queue construction, label emission.

Design of record: docs/DESIGN_label_v1_1_adjudication.md. The tables in
tools/label_schema.py are pre-committed there, so these tests assert the
mapping the document promises — if a table changes, the doc changes with it.
"""
import json
from pathlib import Path

import pytest

from scripts import build_adjudication_queue as baq
from scripts import build_labels_v1_1 as bl
from tools import label_schema as ls

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN = {name: REPO_ROOT / "reports" / name for name in
          ("review_queue.json", "review_verdicts.jsonl",
           "retag_queue.json", "retag_verdicts.jsonl")}
HAVE_FROZEN = all(p.is_file() for p in FROZEN.values())


# --------------------------------------------------------------- schema tables

def test_every_key_decodes_on_both_axes():
    for slug in ls.ADJUDICATION_KEYS.values():
        assert slug in ls.CLAIM_AXIS, slug
        assert slug in ls.WORK_AXIS, slug
        assert slug in ls.ADJUDICATION_HELP, slug


def test_claim_and_work_axes_are_independent_where_the_design_says_so():
    # the same claim verdict appears with different work verdicts, and vice
    # versa — that separation is the whole point of the repair
    assert ls.CLAIM_AXIS["exact_and_warranted"] == ls.CLAIM_AXIS["exact_but_trivial"] == "exact"
    assert ls.WORK_AXIS["exact_and_warranted"] == "warranted"
    assert ls.WORK_AXIS["exact_but_trivial"] == "trivial"
    assert ls.CLAIM_AXIS["misnamed_but_warranted"] == "misnamed"
    assert ls.WORK_AXIS["misnamed_but_warranted"] == "warranted"


def test_slug_of_recovers_the_slug_from_the_stored_button_label():
    assert ls.slug_of("exact_but_trivial: the claim is true, but ...") == "exact_but_trivial"
    assert ls.slug_of("mechanism_only: a real problem is there") == "mechanism_only"
    assert ls.slug_of(None) is None


@pytest.mark.parametrize("slug,strata,accepted,expected", [
    # billed
    ("absent", ["dirA"], True, "hard_false_billed"),
    ("wrong_object_or_place", ["uniform"], True, "hard_false_billed"),
    ("exact_and_warranted", ["uniform"], True, "supported_billed"),
    ("misnamed_but_warranted", ["dirA"], True, "misnamed_billed"),
    ("exact_but_trivial", ["dirA"], True, "trivial_billed"),
    ("misnamed_and_trivial", ["uniform"], True, "trivial_billed"),
    ("inconclusive", ["dirA"], True, "excluded"),
    # dirB
    ("exact_and_warranted", ["dirB"], False, "dirB_recovery"),
    ("misnamed_but_warranted", ["dirB"], False, "dirB_wording_recovery"),
    ("exact_but_trivial", ["dirB"], False, "dirB_trivial"),
    ("absent", ["dirB"], False, "dirB_terra_correct"),
    ("inconclusive", ["dirB"], False, "excluded"),
    # neither
    ("exact_and_warranted", ["terra_flip"], True, "out_of_scope"),
    ("exact_and_warranted", ["dirA"], False, "out_of_scope"),
])
def test_classify_v1_1_matches_the_design_doc_tables(slug, strata, accepted, expected):
    assert ls.classify_v1_1(strata, accepted, slug) == expected


def test_a_dirb_card_judged_absent_inverts_from_recovery_target_to_loss():
    """`This is my miss, terra is correct` must not stay a recovery win."""
    klass = ls.classify_v1_1(["dirB"], False, "absent")
    assert klass == "dirB_terra_correct"
    assert ls.outcome_for(ls.SCORING, klass, "supported",
                          claim_text_changed=False) == "loss"


def test_the_two_arm_dependent_outcomes_swap_with_claim_text():
    # wording arm: dropping a misnamed-but-real claim is a cost, and recovering
    # one under unchanged wording earns nothing
    assert ls.outcome_for(ls.SCORING, "misnamed_billed", "unsupported",
                          claim_text_changed=False) == "cost"
    assert ls.outcome_for(ls.SCORING, "dirB_wording_recovery", "supported",
                          claim_text_changed=False) == "neutral"
    # coarsening arm: Terra is being asked a different, now-true claim
    assert ls.outcome_for(ls.SCORING, "misnamed_billed", "unsupported",
                          claim_text_changed=True) == "neutral"
    assert ls.outcome_for(ls.SCORING, "dirB_wording_recovery", "supported",
                          claim_text_changed=True) == "win"


def test_severity_classes_are_never_scored():
    for klass in ("trivial_billed", "dirB_trivial"):
        for changed in (False, True):
            for verdict in ("supported", "unsupported"):
                assert ls.outcome_for(ls.SCORING, klass, verdict,
                                      claim_text_changed=changed) in (None, "neutral")


def test_outcome_is_none_when_the_event_did_not_fire():
    assert ls.outcome_for(ls.SCORING, "hard_false_billed", "supported",
                          claim_text_changed=False) is None
    assert ls.outcome_for(ls.SCORING, "dirB_recovery", "unsupported",
                          claim_text_changed=False) is None


def test_v1_classes_and_policy_are_untouched():
    """A v1-labelled run must score exactly as Session B scored it."""
    assert ls.V1_CLASSES == ("hard_false_billed", "supported_billed",
                             "overstated_billed", "dirB_recovery")
    assert ls.outcome_for(ls.SCORING_V1, "overstated_billed", "unsupported",
                          claim_text_changed=True) == "cost"


# ------------------------------------------------------------------- the queue

def _origin(card_id, strata, verdict, source="canary", accepted=True):
    return {
        "card_id": card_id, "kind": "condition", "phase": 1, "source": source,
        "property_key": "prop_1", "run_id": "r1", "address": "1 Main St",
        "title": "worn_or_stained_flooring @ bedroom_1",
        "claim": {"catalog_claim": "flooring worn or stained",
                  "observations": ["uniform fading across the boards"]},
        "strips": [{"label": "evidence", "photos": [
            {"key": "photo_001.jpg", "path": "C:/img/photo_001.jpg", "wh": [1280, 853]}]}],
        "meta": {"catalog_item_id": "worn_or_stained_flooring",
                 "catalog_kind": "degradation", "scene_group": "bedroom",
                 "estimate_unit_id": "bedroom_1", "condition_id": f"oc_{card_id}",
                 "photo_count": 1, "terra_batch_conditions": 7,
                 "terra_batch_images": 3, "terra_verdict": "supported",
                 "replica_terra_verdict": None, "direction": "A",
                 "second_opinion": "2f_objected", "accepted": accepted,
                 "disposition": "accepted_for_work"},
        "reveal": [{"label": "Terra verdict", "text": "supported — visible wear"}],
        "hidden": {"low": 100, "high": 900},
        "strata": strata, "legacy_item_id": "C001",
        "verdict_options": [], "verdict_keys": {}, "tags": {},
    }


def _verdict(card_id, verdict, notes=None):
    return {"card_id": card_id, "verdict": verdict, "notes": notes}


def test_blind_cards_leak_nothing_the_reviewer_decided_before():
    origin = _origin("c1", ["dirA"], "terra_claim_supported")
    card = baq.blind_card(origin, "adj_deadbeef1234", 1)
    assert baq.leaks(card) == []
    served = json.dumps({"public": card, "reveal": card["reveal"]})
    assert "terra_claim" not in served
    assert "accepted_for_work" not in served
    assert card["strata"] == []
    assert card["hidden"] is None
    # the claim's own prose survives untouched, including the word "uniform"
    assert "uniform fading across the boards" in served


def test_the_leak_check_actually_catches_a_leak():
    origin = _origin("c1", ["dirA"], "terra_claim_supported")
    card = baq.blind_card(origin, "adj_deadbeef1234", 1)
    card["reveal"] = [{"label": "oops", "text": "you said terra_claim_supported"}]
    assert "terra_claim_" in baq.leaks(card)
    card = baq.blind_card(origin, "adj_deadbeef1234", 1)
    card["strata"] = ["dirA"]
    assert "dirA" in baq.leaks(card)


def test_reask_cards_deliberately_show_both_earlier_answers():
    origin = _origin("c1", ["dirA"], "terra_claim_overstated")
    card = baq.reask_card(
        origin, "adj_1", 2,
        _verdict("c1", "terra_claim_overstated", "I see stains not scuffs"),
        _verdict("c1", "mechanism_only: a real problem is there", None))
    text = json.dumps(card)
    assert "overstated" in text and "stains not scuffs" in text
    assert "mechanism_only" in text
    assert card["kind"] == "adjudication"      # hides the v1 a-e error tags
    assert card["tags"] == {}


def test_repeats_land_a_full_gap_after_their_origin():
    primaries = [baq.blind_card(_origin(f"c{i}", ["uniform"], "x"),
                                f"adj_{i:012d}", 1) for i in range(40)]
    ordered = baq.hash_order(primaries)
    chosen = baq.pick_repeats(ordered)
    assert chosen, "expected at least one repeat"
    repeats = [(c, dict(c, card_id=c["card_id"] + "_r")) for c in chosen]
    out = baq.insert_repeats(ordered, repeats)
    pos = {c["card_id"]: i for i, c in enumerate(out)}
    # A later insertion between a pair only widens it, so the guarantee is a
    # minimum separation, never an exact one.
    for origin, repeat in repeats:
        assert pos[repeat["card_id"]] - pos[origin["card_id"]] >= baq.REPEAT_GAP
    assert len(out) == len(ordered) + len(repeats)


def test_repeat_candidates_exclude_the_unspaceable_tail():
    primaries = [baq.blind_card(_origin(f"c{i}", ["uniform"], "x"),
                                f"adj_{i:012d}", 1) for i in range(20)]
    ordered = baq.hash_order(primaries)
    tail = {c["card_id"] for c in ordered[len(ordered) - baq.REPEAT_GAP:]}
    assert not {c["card_id"] for c in baq.pick_repeats(ordered)} & tail


def test_build_selects_the_arms_by_the_v1_class_rule():
    cards = [
        _origin("c_sup", ["uniform"], "x"),                       # supported_billed
        _origin("c_prod", ["uniform"], "x", source="production"),  # supported_billed
        _origin("c_hard", ["dirA"], "x"),                          # -> re-ask
        _origin("c_flip", ["terra_flip"], "x"),                    # out of scope
    ]
    done = {
        "c_sup": _verdict("c_sup", "terra_claim_supported"),
        "c_prod": _verdict("c_prod", "terra_claim_supported"),
        "c_hard": _verdict("c_hard", "terra_claim_unsupported"),
        "c_flip": _verdict("c_flip", "terra_claim_supported"),
    }
    retag_queue = {"cards": [dict(_origin("c_hard", ["dirA"], "x"))]}
    retag_done = {"c_hard": _verdict("c_hard", "wholly_false: nothing is there")}
    out = baq.build({"cards": cards}, done, retag_queue, retag_done)

    arms = {p["origin_card_id"]: p["arm"] for p in out["provenance"].values()}
    assert arms == {"c_sup": "blind_canary", "c_prod": "blind_production",
                    "c_hard": "reask"}
    phases = {c["card_id"]: c["phase"] for c in out["cards"]}
    by_origin = {p["origin_card_id"]: aid for aid, p in out["provenance"].items()}
    assert phases[by_origin["c_sup"]] == 1
    assert phases[by_origin["c_hard"]] == 2
    assert phases[by_origin["c_prod"]] == 3
    # provenance carries the v1 linkage the served cards must not
    assert out["provenance"][by_origin["c_hard"]]["retag_answer"] == "wholly_false"
    assert out["provenance"][by_origin["c_sup"]]["v1_class"] == "supported_billed"


def test_minted_ids_are_stable_and_distinct_per_role():
    assert baq.mint("c1", "primary") == baq.mint("c1", "primary")
    assert baq.mint("c1", "primary") != baq.mint("c1", "repeat")
    assert baq.mint("c1", "primary") != baq.mint("c2", "primary")
    assert baq.mint("c1", "primary").startswith("adj_")


# ------------------------------------------------------------------ the labels

def _queue_with(answers_by_origin, strata=("dirA",), accepted=True):
    """A minimal adjudication queue + answer log for the emitter."""
    provenance, answers = {}, {}
    for origin, (slug, repeat_slug, retag) in answers_by_origin.items():
        aid = baq.mint(origin, "primary")
        provenance[aid] = {
            "origin_card_id": origin, "role": "primary", "arm": "blind_canary",
            "source": "canary", "property_key": "p", "condition_id": f"oc_{origin}",
            "catalog_item_id": "worn_or_stained_flooring",
            "catalog_kind": "degradation", "strata": list(strata),
            "accepted": accepted, "stored_terra_verdict": "supported",
            "v1_verdict": "terra_claim_supported",
            "v1_class": ls.classify_v1(strata, accepted, "terra_claim_supported"),
            "retag_answer": retag, "retag_note": None,
        }
        if slug:
            answers[aid] = {"card_id": aid, "verdict": f"{slug}: help"}
        if repeat_slug is not None:
            rid = baq.mint(origin, "repeat")
            provenance[rid] = dict(provenance[aid], role="repeat", repeat_of=aid)
            answers[rid] = {"card_id": rid, "verdict": f"{repeat_slug}: help"}
    return {"provenance": provenance, "sources": {}}, answers


def test_emitter_applies_the_axes_and_keys_by_origin_card_id():
    queue, answers = _queue_with({"c1": ("misnamed_but_warranted", None, None)})
    out = bl.build(queue, answers)
    row = out["labels"]["c1"]
    assert row["claim"] == "misnamed" and row["work"] == "warranted"
    assert row["class_v1_1"] == "misnamed_billed"
    assert row["v1_1_projected_verdict"] == "terra_claim_overstated"
    assert out["movement"]["changed"] == 1


def test_emitter_rejects_a_verdict_log_from_the_wrong_exercise():
    queue, _ = _queue_with({"c1": (None, None, None)})
    aid = baq.mint("c1", "primary")
    with pytest.raises(SystemExit):
        bl.build(queue, {aid: {"card_id": aid,
                               "verdict": "terra_claim_supported"}})


def test_the_primary_answer_wins_and_the_repeat_only_measures_consistency():
    queue, answers = _queue_with(
        {"c1": ("exact_and_warranted", "misnamed_but_warranted", None),
         "c2": ("exact_and_warranted", "exact_and_warranted", None)})
    out = bl.build(queue, answers)
    assert out["labels"]["c1"]["class_v1_1"] == "supported_billed"  # primary
    rc = out["repeat_consistency"]
    assert rc["both_answered"] == 2
    assert rc["exact_agreement"] == 1
    assert len(rc["disagreements"]) == 1


def test_pending_cards_are_reported_not_guessed():
    queue, answers = _queue_with({"c1": ("exact_and_warranted", None, None),
                                  "c2": (None, None, None)})
    out = bl.build(queue, answers)
    assert out["coverage"] == {"cards": 2, "answered": 1, "pending": 1,
                               "canary_complete": False,
                               "per_arm": {"blind_canary":
                                           {"total": 2, "answered": 1, "pending": 1}}}
    assert out["labels"]["c2"]["class_v1_1"] == "unlabelled"


def test_retag_agreement_records_divergence_without_overriding_the_answer():
    queue, answers = _queue_with({
        "c1": ("misnamed_but_warranted", None, "mechanism_only"),   # as expected
        "c2": ("exact_and_warranted", None, "mechanism_only"),      # diverges
    })
    out = bl.build(queue, answers)
    assert out["labels"]["c1"]["retag_expectation_met"] is True
    assert out["labels"]["c2"]["retag_expectation_met"] is False
    assert out["labels"]["c2"]["class_v1_1"] == "supported_billed"
    assert out["retag_agreement"]["mechanism_only"]["met"] == 1
    assert out["retag_agreement"]["mechanism_only"]["diverged"] == 1


def test_borderline_carries_no_expectation():
    queue, answers = _queue_with({"c1": ("exact_but_trivial", None, "borderline")})
    out = bl.build(queue, answers)
    assert out["labels"]["c1"]["retag_expectation_met"] is None


def test_render_is_honest_before_any_answer_exists():
    queue, _ = _queue_with({"c1": (None, None, None)})
    text = bl.render(bl.build(queue, {}))
    assert "No adjudication answers recorded yet" in text


# ------------------------------------------- the real queue, when it is present

@pytest.mark.skipif(not HAVE_FROZEN, reason="frozen review inputs not present")
def test_the_real_queue_covers_every_canary_scoreable_card():
    """Phases 1+2 must re-label the whole population Session F scores."""
    review_queue = json.loads(FROZEN["review_queue.json"].read_text(encoding="utf-8"))
    retag_queue = json.loads(FROZEN["retag_queue.json"].read_text(encoding="utf-8"))
    from tools.review_cards import latest_verdicts
    out = baq.build(review_queue, latest_verdicts(FROZEN["review_verdicts.jsonl"]),
                    retag_queue, latest_verdicts(FROZEN["retag_verdicts.jsonl"]))

    scoreable = {"hard_false_billed", "supported_billed", "overstated_billed",
                 "dirB_recovery"}
    covered = {p["origin_card_id"] for p in out["provenance"].values()
               if p["role"] == "primary" and p["source"] == "canary"
               and p["v1_class"] in scoreable}
    expected = set()
    for card in review_queue["cards"]:
        if card.get("kind") != "condition" or card.get("source") != "canary":
            continue
        verdict = (latest_verdicts(FROZEN["review_verdicts.jsonl"])
                   .get(card["card_id"]) or {}).get("verdict")
        if ls.classify_v1(card.get("strata") or [],
                          (card.get("meta") or {}).get("accepted"),
                          verdict) in scoreable:
            expected.add(card["card_id"])
    assert covered == expected
    assert len(expected) == 88          # 11 + 55 + 6 + 16


@pytest.mark.skipif(not HAVE_FROZEN, reason="frozen review inputs not present")
def test_every_blind_card_in_the_real_queue_is_blind():
    path = REPO_ROOT / "reports" / "adjudication_queue.json"
    if not path.is_file():
        pytest.skip("adjudication queue not built")
    queue = json.loads(path.read_text(encoding="utf-8"))
    leaking = {c["card_id"]: baq.leaks(c) for c in queue["cards"]
               if c["phase"] in (1, 3) and baq.leaks(c)}
    assert leaking == {}
