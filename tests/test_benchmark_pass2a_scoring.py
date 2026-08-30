"""Unit tests for canonical-room alignment and tier distance (no IO).

The integration behavior of score_package_round is covered in
tests/test_benchmark_pass2a_packages.py; this file pins the pure functions:
photo-overlap alignment (including the Carroll ordinal-drift repro), signed
tier distance, and candidate identity for 2f-disagreement attribution.
"""
import pytest

from tools import benchmark_pass2a_scoring as sc


CARROLL_BEDROOMS = [
    {"room_id": "bedroom_A", "room": "bedroom", "photo_keys": ["photo_003.jpg"]},
    {"room_id": "bedroom_B", "room": "bedroom", "photo_keys": ["photo_005.jpg"]},
    {"room_id": "bedroom_C", "room": "bedroom", "photo_keys": ["photo_007.jpg"]},
]


def test_carroll_ordinal_drift_realigns_by_photo():
    # Qwen-drift world: photo_005 became living_room, so photo_007's room was
    # numbered bedroom_2. Photo alignment maps it to canonical bedroom 3.
    aligned = sc.align_room_for({"photo_007.jpg"}, "bedroom", CARROLL_BEDROOMS)
    assert aligned == {"status": "aligned", "room_id": "bedroom_C",
                       "overlap": 1, "candidates": ["bedroom_C"]}


def test_alignment_tie_and_zero_overlap_stay_unmatched():
    tie = sc.align_room_for({"photo_003.jpg", "photo_005.jpg"}, "bedroom",
                            CARROLL_BEDROOMS)
    assert tie["status"] == "tie" and tie["room_id"] is None
    assert tie["candidates"] == ["bedroom_A", "bedroom_B"]
    nothing = sc.align_room_for({"photo_099.jpg"}, "bedroom", CARROLL_BEDROOMS)
    assert nothing == {"status": "no_overlap", "room_id": None, "overlap": 0,
                       "candidates": []}


def test_alignment_respects_room_family():
    rooms = CARROLL_BEDROOMS + [
        {"room_id": "living_room", "room": "living",
         "photo_keys": ["photo_007.jpg", "photo_002.jpg"]}]
    # A living-family unit never aligns to a bedroom, even on photo overlap.
    aligned = sc.align_room_for({"photo_007.jpg"}, "living", rooms)
    assert aligned["room_id"] == "living_room"
    aligned = sc.align_room_for({"photo_007.jpg"}, "bedroom", rooms)
    assert aligned["room_id"] == "bedroom_C"
    # Unknown family considers every room; here the overlap is tied 1-1.
    aligned = sc.align_room_for({"photo_007.jpg"}, None, rooms)
    assert aligned["status"] == "tie"


def test_align_units_to_rooms_uses_unit_families():
    units = {"bedroom_2": {"photo_007.jpg"},
             "living_room_primary": {"photo_002.jpg"}}
    rooms = CARROLL_BEDROOMS + [
        {"room_id": "living_room", "room": "living",
         "photo_keys": ["photo_002.jpg", "photo_006.jpg"]}]
    alignment = sc.align_units_to_rooms(units, rooms)
    assert alignment["bedroom_2"]["room_id"] == "bedroom_C"
    assert alignment["living_room_primary"]["room_id"] == "living_room"


@pytest.mark.parametrize("unit,family", [
    ("kitchen_primary", "kitchen"),
    ("bedroom_1", "bedroom"),
    ("bedroom_2", "bedroom"),
    ("living_room_primary", "living"),
    ("exterior_primary", "exterior"),
    ("basement_primary", "utility"),
    ("mystery_unit_x", None),
])
def test_unit_room_family(unit, family):
    assert sc.unit_room_family(unit) == family


def test_tier_distance_signed_and_chain_aware():
    assert sc.tier_distance("kitchen_full_rehab", "kitchen_full_rehab") == 0
    assert sc.tier_distance("kitchen_full_rehab", "kitchen_refresh") == -2
    assert sc.tier_distance("kitchen_refresh", "kitchen_full_rehab") == 2
    # Bedroom's chain skips the partial middle: one step, not two.
    assert sc.tier_distance("bedroom_refresh", "bedroom_full_rehab") == 1
    # Disconnected sub-chains (modernization vs repair) have no distance.
    assert sc.tier_distance("kitchen_full_rehab", "kitchen_repair_heavy") is None
    assert sc.tier_distance("kitchen_full_rehab", None) is None


def test_candidate_identity_aligns_by_review_photos():
    candidate = {"package_type": "bedroom_repair",
                 "estimate_unit_id": "bedroom_2",
                 "review_photo_keys": ["photo_007.jpg"]}
    assert sc._candidate_identity(candidate, CARROLL_BEDROOMS) == (
        "bedroom_repair", "bedroom_C")
    # Without photos the raw unit id is kept (and flagged by the scorer).
    bare = {"package_type": "bedroom_repair", "estimate_unit_id": "bedroom_2",
            "review_photo_keys": []}
    assert sc._candidate_identity(bare, CARROLL_BEDROOMS) == (
        "bedroom_repair", "bedroom_2")
