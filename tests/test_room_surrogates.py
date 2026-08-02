"""
Tests for tools.room_surrogates — single-active-surrogate clustering.

PR 2 of the renovation_estimate_v4 refactor. The clustering primitive is
not yet wired into the estimator; these tests cover only the pure function
in isolation.
"""

from tools.room_surrogates import build_room_surrogates


def _make_photos(*entries):
    """Build a photos dict from a sequence of scenes.

    Each entry may be a bare scene string or a (scene, photo_key) tuple.
    Indices are assigned 1..N in order. The dict returned uses the same
    insertion order.
    """
    photos = {}
    for i, entry in enumerate(entries, start=1):
        if isinstance(entry, tuple):
            scene, photo_key = entry
        else:
            scene = entry
            photo_key = f"p{i:03d}.jpg"
        photos[photo_key] = {
            "photo": {"photo_key": photo_key, "index": i},
            "scene": {"id": scene, "group": "irrelevant"},
        }
    return photos


def _ids(result):
    return [s["room_surrogate_id"] for s in result["room_surrogates"]]


class TestBuildRoomSurrogates:

    def test_empty_photos_returns_empty_shape(self):
        result = build_room_surrogates({})
        assert result == {
            "photo_key_to_room_surrogate_id": {},
            "room_surrogates": [],
        }

    def test_missing_scene_field_treated_as_non_breaking(self):
        photos = {
            "p001.jpg": {"photo": {"photo_key": "p001.jpg", "index": 1}},  # no scene
            "p002.jpg": {
                "photo": {"photo_key": "p002.jpg", "index": 2},
                "scene": {"id": None},
            },
            "p003.jpg": {
                "photo": {"photo_key": "p003.jpg", "index": 3},
                "scene": {"id": ""},
            },
        }
        result = build_room_surrogates(photos)
        assert result["room_surrogates"] == []
        assert result["photo_key_to_room_surrogate_id"] == {}

    def test_all_non_breaking_scenes_yields_no_room_surrogates(self):
        # Exterior is excluded here: it is still non-breaking for interior
        # clustering, but it does earn the one property-level identity (see
        # test_exterior_collapses_to_one_property_level_surrogate).
        photos = _make_photos("hallway", "closet", "roof")
        result = build_room_surrogates(photos)
        assert result["room_surrogates"] == []
        assert result["photo_key_to_room_surrogate_id"] == {}

    def test_hvac_is_non_breaking(self):
        # hvac is a canonical Pass 1a label not mentioned in the PR
        # description. It must be non-breaking: never opens its own
        # surrogate, never splits an active room.
        photos = _make_photos("hvac")
        result = build_room_surrogates(photos)
        assert result["room_surrogates"] == []
        assert result["photo_key_to_room_surrogate_id"] == {}

        photos_around_kitchen = _make_photos("kitchen", "hvac", "kitchen")
        result_2 = build_room_surrogates(photos_around_kitchen)
        assert _ids(result_2) == ["kitchen_1"]
        kitchen_1 = result_2["room_surrogates"][0]
        assert kitchen_1["photo_keys"] == ["p001.jpg", "p003.jpg"]
        assert kitchen_1["notes"] == []
        assert "p002.jpg" not in result_2["photo_key_to_room_surrogate_id"]

    def test_consecutive_same_breaking_scene_extends(self):
        photos = _make_photos("kitchen", "kitchen", "kitchen")
        result = build_room_surrogates(photos)
        assert _ids(result) == ["kitchen_1"]
        s = result["room_surrogates"][0]
        assert s["photo_keys"] == ["p001.jpg", "p002.jpg", "p003.jpg"]
        assert s["listing_order_start"] == 1
        assert s["listing_order_end"] == 3
        assert s["notes"] == []
        assert result["photo_key_to_room_surrogate_id"] == {
            "p001.jpg": "kitchen_1",
            "p002.jpg": "kitchen_1",
            "p003.jpg": "kitchen_1",
        }

    def test_hallway_does_not_split_active_room(self):
        photos = _make_photos("bathroom", "bathroom", "hallway", "bathroom")
        result = build_room_surrogates(photos)
        assert _ids(result) == ["bathroom_1"]
        s = result["room_surrogates"][0]
        assert s["photo_keys"] == ["p001.jpg", "p002.jpg", "p004.jpg"]
        assert s["listing_order_start"] == 1
        assert s["listing_order_end"] == 4
        assert "p003.jpg" not in result["photo_key_to_room_surrogate_id"]

    def test_stairway_does_not_split_active_room(self):
        photos = _make_photos("bathroom", "stairway", "bathroom")
        result = build_room_surrogates(photos)
        assert _ids(result) == ["bathroom_1"]
        s = result["room_surrogates"][0]
        assert s["photo_keys"] == ["p001.jpg", "p003.jpg"]
        assert s["notes"] == []
        assert "p002.jpg" not in result["photo_key_to_room_surrogate_id"]

    def test_bathroom_split_by_bedroom_becomes_bathroom_2(self):
        photos = _make_photos("bathroom", "bedroom", "bathroom")
        result = build_room_surrogates(photos)
        assert _ids(result) == ["bathroom_1", "bedroom_1", "bathroom_2"]
        bathroom_2 = result["room_surrogates"][2]
        assert bathroom_2["notes"] == ["split_after_intervening_room"]
        assert bathroom_2["photo_keys"] == ["p003.jpg"]

    def test_mixed_kitchen_bathroom_alternation(self):
        photos = _make_photos("kitchen", "bathroom", "kitchen", "bathroom")
        result = build_room_surrogates(photos)
        assert _ids(result) == [
            "kitchen_1",
            "bathroom_1",
            "kitchen_2",
            "bathroom_2",
        ]
        assert result["room_surrogates"][0]["notes"] == []
        assert result["room_surrogates"][1]["notes"] == []
        assert result["room_surrogates"][2]["notes"] == [
            "split_after_intervening_room"
        ]
        assert result["room_surrogates"][3]["notes"] == [
            "split_after_intervening_room"
        ]

    def test_pr_example_1_full_sequence(self):
        photos = _make_photos(
            "kitchen",
            "kitchen",
            "dining_room",
            "living_room",
            "bathroom",
            "bathroom",
            "bedroom",
            "bathroom",
        )
        result = build_room_surrogates(photos)
        assert _ids(result) == [
            "kitchen_1",
            "dining_room_1",
            "living_room_1",
            "bathroom_1",
            "bedroom_1",
            "bathroom_2",
        ]
        kitchen_1 = result["room_surrogates"][0]
        assert kitchen_1["photo_keys"] == ["p001.jpg", "p002.jpg"]
        assert kitchen_1["listing_order_start"] == 1
        assert kitchen_1["listing_order_end"] == 2

        bathroom_1 = result["room_surrogates"][3]
        assert bathroom_1["photo_keys"] == ["p005.jpg", "p006.jpg"]
        assert bathroom_1["notes"] == []

        bathroom_2 = result["room_surrogates"][5]
        assert bathroom_2["photo_keys"] == ["p008.jpg"]
        assert bathroom_2["notes"] == ["split_after_intervening_room"]

    def test_pr_example_2_bathroom_hallway_bathroom(self):
        photos = _make_photos("bathroom", "bathroom", "hallway", "bathroom")
        result = build_room_surrogates(photos)
        assert _ids(result) == ["bathroom_1"]
        bathroom_1 = result["room_surrogates"][0]
        assert bathroom_1["photo_keys"] == ["p001.jpg", "p002.jpg", "p004.jpg"]
        assert bathroom_1["listing_order_start"] == 1
        assert bathroom_1["listing_order_end"] == 4
        assert bathroom_1["notes"] == []

    def test_stable_ordering_by_photo_index(self):
        # Pass photos in shuffled insertion order with explicit indices.
        photos = {}
        order_to_insert = [
            ("p005.jpg", 5, "bedroom"),
            ("p001.jpg", 1, "kitchen"),
            ("p003.jpg", 3, "kitchen"),
            ("p002.jpg", 2, "kitchen"),
            ("p004.jpg", 4, "bathroom"),
        ]
        for photo_key, idx, scene in order_to_insert:
            photos[photo_key] = {
                "photo": {"photo_key": photo_key, "index": idx},
                "scene": {"id": scene},
            }
        result = build_room_surrogates(photos)
        assert _ids(result) == ["kitchen_1", "bathroom_1", "bedroom_1"]
        kitchen_1 = result["room_surrogates"][0]
        assert kitchen_1["photo_keys"] == ["p001.jpg", "p002.jpg", "p003.jpg"]

        # Re-shuffle the dict insertion order; output must be identical.
        photos_shuffled = {}
        for photo_key, idx, scene in reversed(order_to_insert):
            photos_shuffled[photo_key] = photos[photo_key]
        result_2 = build_room_surrogates(photos_shuffled)
        assert result_2 == result

    def test_exterior_prefix_does_not_break_interior_clustering(self):
        # The load-bearing invariant: an exterior shot between two kitchen shots
        # must not split the kitchen. Exterior gets its own property-level
        # surrogate, appended after the state machine so it can never become
        # active mid-sequence.
        photos = _make_photos(
            "exterior_front", "kitchen", "exterior_back", "kitchen"
        )
        result = build_room_surrogates(photos)
        assert _ids(result) == ["kitchen_1", "exterior_primary"]
        kitchen_1 = result["room_surrogates"][0]
        assert kitchen_1["photo_keys"] == ["p002.jpg", "p004.jpg"]
        assert kitchen_1["listing_order_start"] == 2
        assert kitchen_1["listing_order_end"] == 4

    def test_exterior_collapses_to_one_property_level_surrogate(self):
        # Every elevation is the same exterior. Without one shared identity each
        # exterior issue lands in its own estimate unit and exterior evidence can
        # never corroborate itself into a package.
        photos = _make_photos("exterior_front", "exterior_back", "exterior_side", "yard")
        result = build_room_surrogates(photos)
        assert _ids(result) == ["exterior_primary"]
        exterior = result["room_surrogates"][0]
        assert exterior["scene_group"] == "exterior"
        assert exterior["photo_keys"] == ["p001.jpg", "p002.jpg", "p003.jpg", "p004.jpg"]
        assert exterior["listing_order_start"] == 1
        assert exterior["listing_order_end"] == 4
        assert set(result["photo_key_to_room_surrogate_id"].values()) == {"exterior_primary"}

    def test_no_exterior_photos_yields_no_exterior_surrogate(self):
        photos = _make_photos("kitchen", "bathroom")
        result = build_room_surrogates(photos)
        assert "exterior_primary" not in _ids(result)

    def test_unrecognized_scene_treated_as_non_breaking_with_note(self):
        photos = _make_photos("kitchen", "new_future_label", "kitchen")
        result = build_room_surrogates(photos)
        assert _ids(result) == ["kitchen_1"]
        kitchen_1 = result["room_surrogates"][0]
        assert kitchen_1["photo_keys"] == ["p001.jpg", "p003.jpg"]
        assert "unrecognized_scene:new_future_label" in kitchen_1["notes"]

        # Unrecognized scene with no active surrogate must not error or
        # produce a surrogate.
        photos_no_active = _make_photos("new_future_label", "hallway")
        result_2 = build_room_surrogates(photos_no_active)
        assert result_2["room_surrogates"] == []
        assert result_2["photo_key_to_room_surrogate_id"] == {}

    def test_scene_group_mapping_uses_pipeline_common(self):
        photos = _make_photos("kitchen", "garage", "living_room")
        result = build_room_surrogates(photos)
        groups_by_id = {
            s["room_surrogate_id"]: s["scene_group"]
            for s in result["room_surrogates"]
        }
        assert groups_by_id["kitchen_1"] == "kitchen"
        assert groups_by_id["garage_1"] == "utility"
        assert groups_by_id["living_room_1"] == "living_areas"

    def test_listing_order_start_and_end_track_extreme_indices(self):
        photos = {
            "p003.jpg": {
                "photo": {"photo_key": "p003.jpg", "index": 3},
                "scene": {"id": "kitchen"},
            },
            "p005.jpg": {
                "photo": {"photo_key": "p005.jpg", "index": 5},
                "scene": {"id": "hallway"},
            },
            "p007.jpg": {
                "photo": {"photo_key": "p007.jpg", "index": 7},
                "scene": {"id": "kitchen"},
            },
        }
        result = build_room_surrogates(photos)
        assert _ids(result) == ["kitchen_1"]
        kitchen_1 = result["room_surrogates"][0]
        assert kitchen_1["listing_order_start"] == 3
        assert kitchen_1["listing_order_end"] == 7
        assert kitchen_1["photo_keys"] == ["p003.jpg", "p007.jpg"]

    def test_missing_index_falls_back_to_photo_key_sort(self):
        photos = {
            "p_b.jpg": {
                "photo": {"photo_key": "p_b.jpg"},
                "scene": {"id": "bathroom"},
            },
            "p_a.jpg": {
                "photo": {"photo_key": "p_a.jpg"},
                "scene": {"id": "kitchen"},
            },
            "p_c.jpg": {
                "photo": {"photo_key": "p_c.jpg"},
                "scene": {"id": "bedroom"},
            },
        }
        result = build_room_surrogates(photos)
        assert _ids(result) == ["kitchen_1", "bathroom_1", "bedroom_1"]
