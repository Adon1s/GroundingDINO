import asyncio
import json
import logging
import re
from pathlib import Path

import pytest

from tools.pass_config import SceneClassifierRunOptions
from tools.scene_classifier_orchestrator import SceneClassifierOrchestrator
from tools.scene_classifier_passes import (
    EXCLUSION_REASONS,
    OBSERVATION_KINDS,
    ONTOLOGY_VERSION,
    PASS_2B_PROMPT_SHA256,
    PASS_2B_PROMPT_VERSION,
    PASS_2B_SYSTEM_PROMPT_TEMPLATE,
    PASS_2C_PROMPT_SHA256,
    PASS_2C_PROMPT_VERSION,
    PASS_2C_SYSTEM_PROMPT,
    PASS_2D_PROMPT_SHA256,
    PASS_2D_PROMPT_VERSION,
    PASS_2D_USER_PROMPT_TEMPLATE,
    PassExecutionError,
    _validate_pass_2c_decisions,
    evaluate_kind_routing,
    format_candidates_text,
    partition_dimension_overlays,
    prioritize_resolution_candidates,
    run_pass_2c,
    run_pass_2d,
)


class FakeTextClient:
    def __init__(self, response: str = '{"resolved_item_id": null}'):
        self.response = response
        self.calls = 0

    async def analyze_text(self, **kwargs):
        self.calls += 1
        return self.response


class FakeOrchestratorClient:
    async def analyze_image(self, image_path, system_prompt, user_prompt, **model_config):
        prompt_lower = (system_prompt or "").lower()
        if "scene type" in prompt_lower:
            return '{"scene":"exterior_front","confidence":0.95,"reasoning":"front exterior"}'
        if "real estate photo analyst" in prompt_lower:
            return "Shingles appear aged and weathered from an aerial angle."
        return ""

    async def analyze_text(self, system_prompt, user_prompt, **model_config):
        system_lower = (system_prompt or "").lower()
        user_lower = (user_prompt or "").lower()
        if "split freeform photo notes" in system_lower:
            return '{"observations":[{"description":"Shingles appear aged and weathered from an aerial angle."}]}'
        if "classify each numbered observation" in system_lower:
            return '{"decisions":[{"index":1,"kind":"degradation"}]}'
        if "map this observation to a catalog item id" in user_lower:
            return '{"resolved_item_id":"damaged_or_aged_roof_shingles"}'
        return "{}"


@pytest.mark.parametrize("kind", sorted(OBSERVATION_KINDS))
def test_kind_routing_is_a_singleton_exact_route(kind):
    routing = evaluate_kind_routing("Shingles appear aged and weathered.", kind)

    assert routing.original_kind == kind
    assert routing.expanded_kinds == (kind,)
    assert routing.reason == "exact_kind"


@pytest.mark.parametrize("raw, normalized", [
    ("  Defect ", "defect"),
    ("DEGRADATION", "degradation"),
    ("Modernization\n", "modernization"),
])
def test_kind_routing_normalizes_case_and_whitespace(raw, normalized):
    routing = evaluate_kind_routing("Cabinet finish is worn.", raw)

    assert routing.original_kind == normalized
    assert routing.expanded_kinds == (normalized,)
    assert routing.reason == "exact_kind"


@pytest.mark.parametrize("kind", ["upgrade", "", None, "safety", "opportunity", "other"])
def test_kind_routing_fails_closed_on_invalid_kinds(kind):
    """Retired and unknown kinds produce an EMPTY route — never a widened or
    unfiltered one. Callers must refuse to retrieve on an empty route."""
    routing = evaluate_kind_routing("Shingles appear aged and weathered.", kind)

    assert routing.expanded_kinds == ()
    assert routing.reason == "invalid_kind"


def test_kind_routing_never_widens_on_condition_language():
    """The v1 upgrade→{upgrade,defect} component-term widening is retired:
    condition language in the text must not change the route."""
    routing = evaluate_kind_routing(
        "Siding is cracked, stained, faded, and rotted.", "degradation"
    )

    assert routing.expanded_kinds == ("degradation",)
    assert routing.reason == "exact_kind"


def test_prioritize_resolution_candidates_pushes_generic_items_last_when_widened():
    candidates = [
        {"item_id": "dated_exterior_finishes", "score": 0.71, "drop_if_generic": True, "defaultHidden": False},
        {"item_id": "damaged_or_aged_roof_shingles", "score": 0.70, "drop_if_generic": False, "defaultHidden": False},
        {"item_id": "curb_appeal_upgrade", "score": 0.69, "drop_if_generic": False, "defaultHidden": True},
    ]

    ordered = prioritize_resolution_candidates(candidates, widened_routing=True)

    assert [candidate["item_id"] for candidate in ordered] == [
        "damaged_or_aged_roof_shingles",
        "dated_exterior_finishes",
        "curb_appeal_upgrade",
    ]


def test_pass_2d_exterior_staining_uses_strict_shortcut():
    client = FakeTextClient()
    candidates = [
        {
            "item_id": "exterior_siding_discoloration_fading",
            "name": "Exterior Siding Discoloration or Fading",
            "description": "Exterior siding or trim shows fading, chalking, staining, or uneven discoloration indicating aging finishes and reduced curb appeal.",
            "support_any": ["discoloration", "faded siding", "chalky siding", "stained siding"],
            "trade_bucket": "exterior_siding_trim",
            "kind": "defect",
            "score": 0.76,
            "defaultHidden": False,
            "drop_if_generic": False,
        },
        {
            "item_id": "brick_weathering_or_mortar_deterioration",
            "name": "Brick Weathering or Mortar Deterioration",
            "description": "Brickwork shows weathering, discoloration, or deteriorating mortar joints.",
            "support_any": ["brick weathering", "discolored brick", "brick staining"],
            "trade_bucket": "masonry_exterior_structure",
            "kind": "defect",
            "score": 0.71,
            "defaultHidden": False,
            "drop_if_generic": False,
        },
    ]

    result = asyncio.run(run_pass_2d(
        vlm_client=client,
        model_config={},
        observation="There is visible staining or aging on the siding or brickwork.",
        candidates=candidates,
        kind="defect",
    ))

    assert result.resolved_item_id == "exterior_siding_discoloration_fading"
    assert result.resolved_kind == "defect"
    assert result.resolution_path == "lexical_shortcut"
    assert result.shortcut_reason == "component_condition_overlap"
    assert client.calls == 0


def test_pass_2d_negation_blocks_shortcut_for_driveway_case():
    """The shortcut must self-block on negated condition language — the
    protection no longer rides in on a routing decision."""
    client = FakeTextClient()
    candidates = [
        {
            "item_id": "driveway_or_walkway_cracking",
            "name": "Driveway or Walkway Cracking/Settlement",
            "description": "Driveway or walkway shows cracks or settlement.",
            "support_any": ["driveway crack", "walkway crack"],
            "trade_bucket": "landscaping_drains",
            "kind": "modernization",
            "score": 0.91,
            "defaultHidden": False,
            "drop_if_generic": False,
        }
    ]

    result = asyncio.run(run_pass_2d(
        vlm_client=client,
        model_config={},
        observation="Driveway appears older but no cracks are visible.",
        candidates=candidates,
        kind="modernization",
    ))

    assert result.resolved_item_id is None
    assert result.resolution_path == "llm"
    assert result.shortcut_reason is None
    assert client.calls == 1


def test_pass_2d_description_only_overlap_does_not_trigger_shortcut():
    client = FakeTextClient()
    candidates = [
        {
            "item_id": "patio_or_porch_surface_wear",
            "name": "Patio or Porch Surface Wear",
            "description": "Covered patio or porch surface shows aging, surface wear, or coating breakdown.",
            "support_any": ["patio surface", "porch surface", "resurface", "reseal", "worn concrete", "aged patio"],
            "trade_bucket": "exterior_siding_trim",
            "kind": "defect",
            "score": 0.82,
            "defaultHidden": False,
            "drop_if_generic": False,
        },
        {
            "item_id": "dated_exterior_finishes",
            "name": "Dated Exterior Finishes",
            "description": "Overall exterior style appears outdated.",
            "support_any": ["dated exterior", "outdated exterior"],
            "trade_bucket": "exterior_siding_trim",
            "kind": "upgrade",
            "score": 0.70,
            "defaultHidden": False,
            "drop_if_generic": True,
        },
    ]

    result = asyncio.run(run_pass_2d(
        vlm_client=client,
        model_config={},
        observation="Weathered wood porch adds character.",
        candidates=candidates,
        kind="upgrade",
    ))

    assert result.resolved_item_id is None
    assert result.resolution_path == "llm"
    assert result.shortcut_reason is None
    assert client.calls == 1


def test_pass_2d_support_term_does_not_fire_inside_a_longer_word():
    """`support_phrase_hit` skips the 2d LLM call entirely, so a support term
    matching inside an unrelated word silently misroutes the observation.
    "ding" on baseboard_wear_scuffs substring-matched "si(ding)" in 221 corpus
    observations; word-start anchoring must send this to the LLM instead.
    """
    client = FakeTextClient()
    candidates = [
        {
            "item_id": "baseboard_wear_scuffs",
            "name": "Baseboard Wear or Scuffs",
            "description": "Baseboards show wear, scuffs, chips, or dings.",
            "support_any": ["baseboard", "scuff", "ding", "dented"],
            "trade_bucket": "interior_trim",
            "kind": "defect",
            "score": 0.88,
            "defaultHidden": False,
            "drop_if_generic": False,
        }
    ]

    result = asyncio.run(run_pass_2d(
        vlm_client=client,
        model_config={},
        observation="Vertical siding is installed across the exterior.",
        candidates=candidates,
        kind="defect",
    ))

    assert result.resolved_item_id is None
    assert result.resolution_path == "llm"
    assert result.shortcut_reason is None
    assert client.calls == 1


def test_pass_2d_support_term_still_fires_on_inflections():
    """The anchoring must not cost the legitimate shortcut: a "scuff" support
    term still resolves "scuffed"/"scuffs"."""
    client = FakeTextClient()
    candidates = [
        {
            "item_id": "baseboard_wear_scuffs",
            "name": "Baseboard Wear or Scuffs",
            "description": "Baseboards show wear, scuffs, chips, or dings.",
            "support_any": ["baseboard", "scuff", "ding", "dented"],
            "trade_bucket": "interior_trim",
            "kind": "defect",
            "score": 0.88,
            "defaultHidden": False,
            "drop_if_generic": False,
        }
    ]

    result = asyncio.run(run_pass_2d(
        vlm_client=client,
        model_config={},
        observation="The baseboards are scuffed along the hallway.",
        candidates=candidates,
        kind="defect",
    ))

    assert result.resolved_item_id == "baseboard_wear_scuffs"
    assert result.shortcut_reason == "support_phrase_hit"
    assert client.calls == 0


def _mold_candidates():
    return [
        {
            "item_id": "visible_mold_or_mildew",
            "name": "Visible Mold or Mildew",
            "description": "Mold or mildew growth is visible on a surface.",
            "support_any": ["mold$", "mildew", "fungus", "black spots", "growth"],
            "trade_bucket": "remediation",
            "kind": "defect",
            "score": 0.88,
            "defaultHidden": False,
            "drop_if_generic": False,
        }
    ]


def test_pass_2d_marked_support_term_does_not_fire_on_crown_molding():
    """Word-start anchoring alone cannot stop a term matching a word it prefixes.

    "mold" sat inside "crown molding" in 18 of the 4,572 corpus observations —
    as many as the 18 genuine mold observations — and `support_phrase_hit`
    skips the 2d LLM call outright, so every one silently resolved a trim
    description as a moisture defect. The trailing-boundary marker must send
    these to the LLM instead.
    """
    for observation in (
        "Crown molding is basic in style.",
        "Dark staining is present above the windows near the crown molding.",
    ):
        client = FakeTextClient()
        result = asyncio.run(run_pass_2d(
            vlm_client=client,
            model_config={},
            observation=observation,
            candidates=_mold_candidates(),
            kind="defect",
        ))

        assert result.resolved_item_id is None, observation
        assert result.resolution_path == "llm", observation
        assert result.shortcut_reason is None, observation
        assert client.calls == 1, observation


def test_pass_2d_marked_support_term_still_fires_on_real_mold():
    """The marker must not cost the legitimate shortcut, including the
    hyphenated form — "-" is not a word char to \\b, so "mold-like" still
    matches."""
    for observation in (
        "Black mold and mildew are visible along the lower left wall.",
        "Heavy staining and mildew or mold-like buildup are visible in the tub.",
    ):
        client = FakeTextClient()
        result = asyncio.run(run_pass_2d(
            vlm_client=client,
            model_config={},
            observation=observation,
            candidates=_mold_candidates(),
            kind="defect",
        ))

        assert result.resolved_item_id == "visible_mold_or_mildew", observation
        assert result.shortcut_reason == "support_phrase_hit", observation
        assert client.calls == 0, observation


def test_pass_2d_prompt_does_not_leak_the_whole_word_marker():
    """The marker is a matching-layer detail; a model must never see it."""
    rendered = format_candidates_text(_mold_candidates())
    assert "support_terms=mold, mildew, fungus, black spots, growth" in rendered
    assert "mold$" not in rendered


def test_partition_dimension_overlays_only_excludes_floorplan_overlays():
    """The pre-filter targets OCR'd MLS floorplan overlays, which are a
    dimension plus a room name and nothing else. An observation that merely
    cites a tile or framing size is a real finding, and dropping it here is
    invisible downstream because it happens before classification.
    """
    overlays = [
        "Primary Bedroom 14' x 12'",
        "Living Room 20 x 15",
        "14'6 x 10'",
        "Kitchen 10'x8'6\"",
    ]
    keepers = [
        "Cracked 12 x 12 ceramic floor tiles near the entry.",
        "Dated 4 x 4 tile backsplash above the range.",
        "Water stain near the 2 x 4 framing in the basement.",
        "3x6 subway tile backsplash is dated.",
        "Cracked 12x12 tile",
    ]

    observations = [{"description": d} for d in overlays + keepers]
    to_classify, excluded = partition_dimension_overlays(observations)

    assert [x["description"] for x in to_classify] == keepers
    assert [x["description"] for x in excluded] == overlays
    assert all(x["reason"] == "measurement_overlay" for x in excluded)


def test_partition_dimension_overlays_leaves_non_dimension_rows_alone():
    observations = [
        {"description": "The carpet is heavily stained."},
        {"description": "Bedroom has a ceiling fan."},
    ]

    to_classify, excluded = partition_dimension_overlays(observations)
    assert to_classify == observations
    assert excluded == []


def test_orchestrator_stops_after_classification():
    """observation-kind-v2: the pipeline ends after Pass 2c. The shadow lane,
    Pass 2d, and Pass 2e are dormant until the catalog migration (Task 2) and
    downstream cutover (Task 3) land — the candidate provider must never be
    called, and the result must be marked classification_only."""
    provider_calls = []

    def candidate_provider(observation_text, context):
        provider_calls.append(observation_text)
        return []

    orchestrator = SceneClassifierOrchestrator(
        qwen_config={},
        gpt5_config={},
        vlm_client=FakeOrchestratorClient(),
        candidate_provider=candidate_provider,
        top_k_candidates=5,
        catalog_items=[],
    )

    result = asyncio.run(orchestrator.analyze_image(
        image_path=Path("photo_002.jpg"),
        options=SceneClassifierRunOptions(),
    ))

    assert result.classification_only is True
    assert result.ontology_version == ONTOLOGY_VERSION
    assert provider_calls == []

    assert len(result.observations) == 1
    obs = result.observations[0]
    assert obs["description"] == "Shingles appear aged and weathered from an aerial angle."
    assert obs["kind"] == "degradation"
    assert obs["issue_id"]
    assert obs["source_photo_key"] == "photo_002.jpg"
    assert result.excluded_observations == []

    assert result.pass_states["2d"] == "skipped"
    assert result.pass_states["2e"] == "skipped"
    assert result.resolved_items == []
    assert result.verified_issues == []

    ontology_debug = result.debug["ontology"]
    assert ontology_debug["ontology_version"] == ONTOLOGY_VERSION
    assert ontology_debug["pass_2c_prompt_version"] == PASS_2C_PROMPT_VERSION
    assert ontology_debug["pass_2c_prompt_sha256"] == PASS_2C_PROMPT_SHA256
    assert result.debug["classification_only"]["reason"] == "classification_only_v2"

    as_dict = result.to_dict()
    assert as_dict["classification_only"] is True
    assert as_dict["observations"] == result.observations
    assert as_dict["excluded_observations"] == []


# ── observation-kind-v2 prompt invariants ───────────────────────────────────
# The v1 prompt tests (label enum, interior finish anchors, safety-label
# coercion) are superseded: the two-kind label vocabulary is retired and the
# v2 contract fails closed on unknown values instead of coercing to "other".
# The exterior-anchor edit reverted in the working tree stays retired — the
# degradation kind now owns exterior wear.


def _prompt_example_values(key: str) -> set:
    """Pull the enum alternatives out of the Pass 2c return-shape example."""
    match = re.search(rf'"{key}":\s*"([^"]+)"', PASS_2C_SYSTEM_PROMPT)
    assert match, f"Pass 2c prompt is missing a return-shape {key} example"
    return set(match.group(1).split("|"))


def test_pass_2c_prompt_example_kinds_match_contract():
    assert _prompt_example_values("kind") == set(OBSERVATION_KINDS)


def test_pass_2c_prompt_example_reasons_match_contract():
    assert _prompt_example_values("exclude") == set(EXCLUSION_REASONS)


def test_pass_2c_prompt_defines_every_kind_and_reason():
    prompt = PASS_2C_SYSTEM_PROMPT
    for value in sorted(OBSERVATION_KINDS) + sorted(EXCLUSION_REASONS):
        assert f"- {value}:" in prompt, f"prompt is missing a definition for {value}"


def test_pass_2c_prompt_has_no_whole_system_absence_rule():
    """Absence safety lives in the catalog deny lists, not in a blanket 2c
    suppression rule. A prompt-level absence rule over-generalised from gutters
    to kitchen fixtures ("no visible modern vent hood"), where absence is a
    legitimate priced upgrade. The v2 prompt distinguishes *asserted* absence
    (a kind) from absence inferred only from non-visibility (excluded), which
    is a different, narrower rule. See docs/HANDOFF_pass2c_exterior_recall.md.
    """
    assert "whole system is absent" not in PASS_2C_SYSTEM_PROMPT.lower()


def test_pass_2b_prompt_requires_atomic_split():
    prompt = PASS_2B_SYSTEM_PROMPT_TEMPLATE.lower()
    assert "one condition claim per observation" in prompt
    assert "weathered and rotted deck boards" in prompt


def test_prompt_version_constants_are_pinned():
    """Prompt provenance: artifacts and benchmark reports record these; a
    prompt edit must change the SHA (it hashes the prompt text) and should
    bump the version string."""
    assert PASS_2B_PROMPT_VERSION == "pass_2b_atomic_v2"
    assert PASS_2C_PROMPT_VERSION == "pass_2c_kind_v3"
    assert PASS_2D_PROMPT_VERSION == "pass_2d_exact_kind_v2"
    assert len(PASS_2B_PROMPT_SHA256) == 64
    assert len(PASS_2C_PROMPT_SHA256) == 64
    assert len(PASS_2D_PROMPT_SHA256) == 64
    assert len({PASS_2B_PROMPT_SHA256, PASS_2C_PROMPT_SHA256, PASS_2D_PROMPT_SHA256}) == 3


def test_pass_2d_prompt_pins_the_kind_upstream():
    assert "kind is decided upstream" in PASS_2D_USER_PROMPT_TEMPLATE
    assert "routing guidance" not in PASS_2D_USER_PROMPT_TEMPLATE


# ── observation-kind-v2 classification behavior ─────────────────────────────


def test_pass_2c_partitions_kinds_and_exclusions():
    observations = [
        {"description": "Wood siding is faded and weathered but remains intact."},
        {"description": "The room contains a ceiling fan."},
        {"description": "A downspout is disconnected and terminates at the foundation."},
    ]
    client = FakeTextClient(json.dumps({
        "decisions": [
            {"index": 1, "kind": "degradation"},
            {"index": 2, "exclude": "neutral_presence"},
            {"index": 3, "kind": "defect"},
        ]
    }))

    result = asyncio.run(run_pass_2c(
        vlm_client=client,
        model_config={},
        observations=observations,
        scene="exterior_front",
    ))

    assert result.ontology_version == ONTOLOGY_VERSION
    assert result.observations == [
        {"description": "Wood siding is faded and weathered but remains intact.", "kind": "degradation"},
        {"description": "A downspout is disconnected and terminates at the foundation.", "kind": "defect"},
    ]
    assert result.excluded == [
        {"description": "The room contains a ceiling fan.", "reason": "neutral_presence"},
    ]


def test_pass_2c_accepts_string_indexes():
    client = FakeTextClient('{"decisions":[{"index":"1","kind":"modernization"}]}')

    result = asyncio.run(run_pass_2c(
        vlm_client=client,
        model_config={},
        observations=[{"description": "Kitchen has dated oak cabinets in good repair."}],
    ))

    assert result.observations[0]["kind"] == "modernization"


def test_pass_2c_excludes_overlays_before_the_llm_call():
    """Dimension overlays are excluded deterministically — when every input is
    an overlay, the model is never called."""
    client = FakeTextClient()

    result = asyncio.run(run_pass_2c(
        vlm_client=client,
        model_config={},
        observations=[
            {"description": "Primary Bedroom 14' x 12'"},
            {"description": "Living Room 20 x 15"},
        ],
    ))

    assert client.calls == 0
    assert result.observations == []
    assert [x["reason"] for x in result.excluded] == ["measurement_overlay", "measurement_overlay"]


def test_pass_2c_empty_observations_short_circuit():
    client = FakeTextClient()

    result = asyncio.run(run_pass_2c(
        vlm_client=client,
        model_config={},
        observations=[],
    ))

    assert client.calls == 0
    assert result.observations == []
    assert result.excluded == []
    assert result.raw_response is None


# ── observation-kind-v2 fail-closed validation ──────────────────────────────
# Deliberate departure from v1: unknown values raised, never coerced. The v1
# unknown-label→"other" fallback made prompt/contract drift invisible.


def _run_2c_expecting_failure(response: str):
    client = FakeTextClient(response)
    with pytest.raises(PassExecutionError) as excinfo:
        asyncio.run(run_pass_2c(
            vlm_client=client,
            model_config={},
            observations=[
                {"description": "The carpet is heavily stained."},
                {"description": "Bedroom has a ceiling fan."},
            ],
        ))
    assert excinfo.value.pass_key == "2c"
    assert excinfo.value.stage == "parse"
    return excinfo.value


def test_pass_2c_fails_closed_on_missing_index():
    _run_2c_expecting_failure('{"decisions":[{"index":1,"kind":"degradation"}]}')


def test_pass_2c_fails_closed_on_duplicate_index():
    _run_2c_expecting_failure(
        '{"decisions":[{"index":1,"kind":"degradation"},{"index":1,"exclude":"neutral_presence"}]}'
    )


def test_pass_2c_fails_closed_on_unknown_kind():
    _run_2c_expecting_failure(
        '{"decisions":[{"index":1,"kind":"upgrade"},{"index":2,"exclude":"neutral_presence"}]}'
    )


def test_pass_2c_fails_closed_on_unknown_exclusion_reason():
    _run_2c_expecting_failure(
        '{"decisions":[{"index":1,"kind":"degradation"},{"index":2,"exclude":"vibes"}]}'
    )


def test_pass_2c_fails_closed_on_kind_and_exclude_together():
    _run_2c_expecting_failure(
        '{"decisions":[{"index":1,"kind":"degradation","exclude":"good_condition"},'
        '{"index":2,"exclude":"neutral_presence"}]}'
    )


def test_pass_2c_fails_closed_on_out_of_range_index():
    _run_2c_expecting_failure(
        '{"decisions":[{"index":1,"kind":"degradation"},{"index":3,"kind":"defect"}]}'
    )


def test_pass_2c_fails_closed_on_malformed_json():
    _run_2c_expecting_failure("not json at all")


def test_pass_2c_fails_closed_on_rewritten_shape():
    """A model echoing the v1 shape (labeled list) must fail loudly, not be
    silently interpreted."""
    _run_2c_expecting_failure(
        '{"labeled":[{"description":"The carpet is heavily stained.","label":"defect_or_damage"}]}'
    )


def test_validate_pass_2c_decisions_returns_normalized_map():
    decisions = _validate_pass_2c_decisions(
        {"decisions": [
            {"index": 2, "exclude": " Good_Condition "},
            {"index": 1, "kind": " DEFECT "},
        ]},
        expected_count=2,
    )

    assert decisions == {1: {"kind": "defect"}, 2: {"exclude": "good_condition"}}


# ── Pass 1a scene normalization ─────────────────────────────────────────────
# The parser used to pass the model's raw string through (so "Kitchen" failed
# every downstream lookup) and hand-rolled a float coercion for a `confidence`
# key the prompt never requests and no consumer reads.

class _SceneClient:
    def __init__(self, response):
        self.response = response

    async def analyze_image(self, **kwargs):
        return self.response


def _run_1a(response):
    from tools.scene_classifier_passes import run_pass_1a_scene_type

    return asyncio.run(run_pass_1a_scene_type(Path("p.jpg"), _SceneClient(response), {}))


def test_pass_1a_accepts_a_canonical_scene():
    assert _run_1a('{"scene":"exterior_front"}').scene == "exterior_front"


def test_pass_1a_normalizes_case_and_whitespace():
    assert _run_1a('{"scene":" Kitchen "}').scene == "kitchen"


def test_pass_1a_keeps_recognized_non_prompt_scenes():
    """Off-contract but downstream-meaningful output must not be demoted."""
    assert _run_1a('{"scene":"laundry_room"}').scene == "laundry_room"


def test_pass_1a_falls_back_to_other_for_unknown_scenes():
    assert _run_1a('{"scene":"sunroom"}').scene == "other"


def test_pass_1a_falls_back_to_other_when_scene_is_absent():
    assert _run_1a('{"reasoning":"blurry"}').scene == "other"


def test_pass_1a_ignores_an_unrequested_confidence_key():
    result = _run_1a('{"scene":"kitchen","confidence":0.95}')
    assert result.scene == "kitchen"
    assert not hasattr(result, "confidence")
