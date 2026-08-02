"""
Tests for the canonical scene taxonomy.

The invariant: there is exactly ONE scene table. Six modules used to keep their
own hand-maintained copy of the scene->group map and they had already drifted
(property_summarizer filed `hvac` under exterior; the artifact viewer's Pass 1a
prompt was missing `closet` and advertised a `confidence` field the runtime
prompt never asked for). Everything now derives from SCENE_SPECS, and these
tests pin the derived values so a future edit to the table cannot silently
change grouping, prompt vocabulary, or room-surrogate clustering.
"""
import pytest

from tools.pipeline_common import (
    ALL_SCENE_IDS,
    BREAKING_SCENES,
    NON_BREAKING_SCENES,
    PASS_1A_SCENE_IDS,
    SCENE_GROUPS_UI,
    SCENE_SPECS,
    SCENE_TO_GROUP_UI,
    normalize_scene_id,
)

# The pre-consolidation literals, kept verbatim as the regression oracle.
EXPECTED_GROUPS = {
    "kitchen": ["kitchen", "pantry"],
    "bathroom": ["bathroom"],
    "bedroom": ["bedroom", "closet"],
    "living_areas": ["living_room", "dining_room", "home_office", "hallway", "stairway"],
    "utility": ["laundry_room", "basement", "attic", "garage", "hvac"],
    "exterior": ["exterior_front", "exterior_back", "exterior_side", "yard", "patio",
                 "deck", "balcony", "driveway", "pool", "garden"],
    "other": ["roof", "other", "unknown", "floor_plan", "aerial_view", "street_view"],
}

EXPECTED_PASS_1A = (
    "exterior_front", "exterior_back", "exterior_side", "living_room", "kitchen",
    "bedroom", "closet", "bathroom", "dining_room", "basement", "attic", "garage",
    "yard", "pool", "roof", "hvac", "other",
)

EXPECTED_BREAKING = {
    "kitchen", "bathroom", "bedroom", "living_room", "dining_room", "laundry_room",
    "garage", "basement", "home_office", "attic", "pantry",
}


# ── table shape ─────────────────────────────────────────────────────────────

def test_scene_ids_are_unique():
    ids = [spec.scene_id for spec in SCENE_SPECS]
    assert len(ids) == len(set(ids)) == 31


def test_every_scene_belongs_to_exactly_one_group():
    for scene, group in SCENE_TO_GROUP_UI.items():
        owning = [g for g, scenes in SCENE_GROUPS_UI.items() if scene in scenes]
        assert owning == [group], f"{scene} appears in {owning}"


def test_group_membership_and_order_match_the_pre_consolidation_literal():
    """SCENE_GROUPS_UI values are serialized as `scenes_included`; order matters."""
    assert SCENE_GROUPS_UI == EXPECTED_GROUPS
    for group, scenes in EXPECTED_GROUPS.items():
        assert SCENE_GROUPS_UI[group] == scenes, group


def test_group_insertion_order_is_stable():
    assert list(SCENE_GROUPS_UI) == [
        "kitchen", "bathroom", "bedroom", "living_areas", "utility", "exterior", "other",
    ]


def test_all_scene_ids_covers_the_table():
    assert ALL_SCENE_IDS == set(SCENE_TO_GROUP_UI) == {s.scene_id for s in SCENE_SPECS}


# ── Pass 1a vocabulary ──────────────────────────────────────────────────────

def test_pass_1a_scene_ids_exact_and_ordered():
    assert PASS_1A_SCENE_IDS == EXPECTED_PASS_1A


def test_pass_1a_orders_are_contiguous():
    orders = sorted(s.pass_1a_order for s in SCENE_SPECS if s.pass_1a_order is not None)
    assert orders == list(range(len(PASS_1A_SCENE_IDS)))


def test_generated_prompt_lists_exactly_the_pass_1a_scenes():
    from tools.scene_classifier_passes import PASS_1A_SYSTEM_PROMPT

    bulleted = [
        line[2:].strip()
        for line in PASS_1A_SYSTEM_PROMPT.splitlines()
        if line.startswith("- ")
    ]
    assert bulleted == list(EXPECTED_PASS_1A)


def test_prompt_does_not_request_confidence():
    """The field was parsed but never populated or read; the prompt never asked."""
    from tools.scene_classifier_passes import PASS_1A_SYSTEM_PROMPT

    assert "confidence" not in PASS_1A_SYSTEM_PROMPT


def test_the_14_non_prompt_scenes_are_retained_as_aliases():
    """Legacy/nonconforming artifacts still carry these; they must keep grouping."""
    aliases = ALL_SCENE_IDS - set(PASS_1A_SCENE_IDS)
    assert aliases == {
        "pantry", "home_office", "hallway", "stairway", "laundry_room", "patio",
        "deck", "balcony", "driveway", "garden", "unknown", "floor_plan",
        "aerial_view", "street_view",
    }
    for alias in aliases:
        assert SCENE_TO_GROUP_UI[alias] != "" and alias in SCENE_GROUPS_UI[SCENE_TO_GROUP_UI[alias]]


# ── normalization ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("scene", sorted(ALL_SCENE_IDS))
def test_canonical_ids_round_trip(scene):
    assert normalize_scene_id(scene) == scene


@pytest.mark.parametrize("raw,expected", [
    (" Kitchen ", "kitchen"),
    ("LIVING_ROOM", "living_room"),
    ("laundry_room", "laundry_room"),   # not a Pass 1a category, still recognized
    ("sunroom", "other"),
    ("", "other"),
    (None, "other"),
    (123, "other"),
])
def test_normalize_scene_id(raw, expected):
    assert normalize_scene_id(raw) == expected


# ── HVAC drift (the bug this consolidation fixes) ───────────────────────────

def test_hvac_groups_under_utility_everywhere():
    import tools.catalog_auditor as catalog_auditor
    import tools.model_comparison as model_comparison
    import tools.property_summarizer as property_summarizer

    assert SCENE_TO_GROUP_UI["hvac"] == "utility"
    assert catalog_auditor.SCENE_TO_GROUP["hvac"] == "utility"
    assert model_comparison.SCENE_TO_GROUP["hvac"] == "utility"
    # property_summarizer previously filed hvac under "exterior"
    assert property_summarizer.SCENE_TO_GROUP["hvac"] == "utility"


def test_consumers_share_one_map_object():
    import tools.catalog_auditor as catalog_auditor
    import tools.model_comparison as model_comparison
    import tools.property_summarizer as property_summarizer

    assert catalog_auditor.SCENE_TO_GROUP is SCENE_TO_GROUP_UI
    assert model_comparison.SCENE_TO_GROUP is SCENE_TO_GROUP_UI
    assert property_summarizer.SCENE_TO_GROUP is SCENE_TO_GROUP_UI
    assert property_summarizer.SCENE_GROUPS is SCENE_GROUPS_UI


def test_embeddings_group_fallback_tracks_the_table():
    from tools.catalog_embeddings import _ALL_SCENE_GROUPS

    assert _ALL_SCENE_GROUPS == tuple(SCENE_GROUPS_UI)


# ── room surrogates ─────────────────────────────────────────────────────────

def test_breaking_and_non_breaking_partition_the_table():
    assert BREAKING_SCENES | NON_BREAKING_SCENES == ALL_SCENE_IDS
    assert not (BREAKING_SCENES & NON_BREAKING_SCENES)


def test_breaking_scenes_match_the_pre_consolidation_set():
    assert BREAKING_SCENES == EXPECTED_BREAKING


def test_exterior_scenes_are_recognized_without_a_prefix_special_case():
    """They used to be in neither set and were rescued by startswith("exterior_")."""
    from tools.room_surrogates import _is_unrecognized

    for scene in ("exterior_front", "exterior_back", "exterior_side"):
        assert scene in NON_BREAKING_SCENES
        assert _is_unrecognized(scene) is False


@pytest.mark.parametrize("scene,expected", [
    ("kitchen", False),
    ("hvac", False),
    ("laundry_room", False),
    ("sunroom", True),
    ("", False),
])
def test_is_unrecognized(scene, expected):
    from tools.room_surrogates import _is_unrecognized

    assert _is_unrecognized(scene) is expected


# ── package rooms (kept in rehab_packages; pinned here) ─────────────────────

@pytest.mark.parametrize("scene,room", [
    ("living_room", "living"),
    ("dining_room", "living"),
    ("home_office", "living"),
    ("pantry", "kitchen"),
    ("kitchen", "kitchen"),
    ("bathroom", "bathroom"),
    ("bedroom", "bedroom"),
])
def test_scene_normalizes_to_package_room(scene, room):
    from tools.rehab_packages import _normalize_scene_to_room

    assert _normalize_scene_to_room(scene) == room


@pytest.mark.parametrize("scene", ["hallway", "stairway"])
def test_transitional_scenes_do_not_drive_a_living_package(scene):
    from tools.rehab_packages import _normalize_scene_to_room

    assert _normalize_scene_to_room(scene) == scene


def test_group_to_room_keys_are_real_scene_groups():
    from tools.rehab_packages import _GROUP_TO_ROOM

    assert set(_GROUP_TO_ROOM) <= set(SCENE_GROUPS_UI)


def test_pass_2f_prompt_rooms_are_valid_package_rooms():
    from tools.rehab_packages import VALID_ROOMS
    from tools.scene_classifier_passes import PASS_2F_ROOM_PROMPTS

    assert set(PASS_2F_ROOM_PROMPTS) <= VALID_ROOMS
