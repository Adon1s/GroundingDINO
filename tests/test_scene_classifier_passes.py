import asyncio
import json
import logging
import re
from pathlib import Path

from tools.pass_config import SceneClassifierRunOptions
from tools.scene_classifier_orchestrator import SceneClassifierOrchestrator
from tools.scene_classifier_passes import (
    PASS_2C_SYSTEM_PROMPT,
    VALID_LABELS,
    _coerce_labeled_2c,
    evaluate_kind_routing,
    force_other_if_dimensions,
    format_candidates_text,
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
        if "label each observation" in system_lower:
            return '{"labeled":[{"description":"Shingles appear aged and weathered from an aerial angle.","label":"upgrade_candidate"}]}'
        if "map this observation to a catalog item id" in user_lower:
            return '{"resolved_item_id":"damaged_or_aged_roof_shingles"}'
        return "{}"


def test_kind_routing_roof_wear_expands_to_both_kinds():
    routing = evaluate_kind_routing(
        "Shingles appear aged and weathered from an aerial angle.",
        "upgrade",
    )

    assert routing.original_kind == "upgrade"
    assert routing.expanded_kinds == ("upgrade", "defect")
    assert routing.reason == "visible_condition_signal"
    assert "shingle" in routing.matched_component_terms
    assert routing.blocked_by_negation is False


def test_kind_routing_roof_color_stays_upgrade_only():
    routing = evaluate_kind_routing("Roof color appears dated.", "upgrade")

    assert routing.expanded_kinds == ("upgrade",)
    assert routing.reason == "no_visible_condition_signal"


def test_kind_routing_negated_age_only_is_blocked():
    routing = evaluate_kind_routing(
        "Brick exterior appears aged but intact with no visible damage.",
        "upgrade",
    )

    assert routing.expanded_kinds == ("upgrade",)
    assert routing.reason == "blocked_by_negation"
    assert routing.blocked_by_negation is True


def test_kind_routing_no_cracks_visible_is_blocked():
    routing = evaluate_kind_routing(
        "Driveway appears older but no cracks are visible.",
        "upgrade",
    )

    assert routing.expanded_kinds == ("upgrade",)
    assert routing.blocked_by_negation is True


def test_kind_routing_faded_siding_can_still_expand_with_softening_present():
    routing = evaluate_kind_routing(
        "Siding color looks faded but no damage is visible.",
        "upgrade",
    )

    assert routing.expanded_kinds == ("upgrade", "defect")
    assert routing.blocked_by_negation is True
    assert "siding" in routing.matched_component_terms
    assert "fade" in routing.matched_condition_terms


def test_kind_routing_exterior_style_dated_is_not_condition_signal():
    routing = evaluate_kind_routing("Exterior style appears dated.", "upgrade")

    assert routing.expanded_kinds == ("upgrade",)
    assert routing.reason == "no_visible_condition_signal"


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
    client = FakeTextClient()
    routing = evaluate_kind_routing(
        "Driveway appears older but no cracks are visible.",
        "upgrade",
    )
    candidates = [
        {
            "item_id": "driveway_or_walkway_cracking",
            "name": "Driveway or Walkway Cracking/Settlement",
            "description": "Driveway or walkway shows cracks or settlement.",
            "support_any": ["driveway crack", "walkway crack"],
            "trade_bucket": "landscaping_drains",
            "kind": "defect",
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
        kind="upgrade",
        kind_routing=routing,
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


def test_force_other_if_dimensions_only_drops_floorplan_overlays():
    """The filter targets OCR'd MLS floorplan overlays, which are a dimension
    plus a room name and nothing else. An observation that merely cites a tile
    or framing size is a real finding, and dropping it here is invisible
    downstream because it happens before the labeled_forward split.
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

    labeled = (
        [{"description": d, "label": "defect_or_damage"} for d in overlays]
        + [{"description": d, "label": "defect_or_damage"} for d in keepers]
    )
    out = {x["description"]: x["label"] for x in force_other_if_dimensions(labeled)}

    for d in overlays:
        assert out[d] == "other", f"overlay should be forced to other: {d!r}"
    for d in keepers:
        assert out[d] == "defect_or_damage", f"real observation was dropped: {d!r}"


def test_force_other_if_dimensions_leaves_non_dimension_rows_alone():
    labeled = [
        {"description": "The carpet is heavily stained.", "label": "defect_or_damage"},
        {"description": "Bedroom has a ceiling fan.", "label": "other"},
    ]

    assert force_other_if_dimensions(labeled) == labeled


def test_orchestrator_widens_roof_upgrade_retrieval_without_relabeling():
    provider_contexts = []

    def candidate_provider(observation_text, context):
        provider_contexts.append(dict(context))
        return [
            {
                "item_id": "dated_exterior_finishes",
                "name": "Dated Exterior Finishes",
                "description": "Overall exterior style appears outdated.",
                "support_any": ["dated exterior", "outdated exterior"],
                "trade_bucket": "exterior_siding_trim",
                "kind": "upgrade",
                "score": 0.71,
                "defaultHidden": False,
                "drop_if_generic": True,
            },
            {
                "item_id": "damaged_or_aged_roof_shingles",
                "name": "Damaged or Aged Roof Shingles",
                "description": "Missing, curling, patchy, worn shingles or debris suggesting reduced remaining life.",
                "support_any": ["roof", "shingle", "worn"],
                "trade_bucket": "roof_gutters",
                "kind": "defect",
                "score": 0.70,
                "defaultHidden": False,
                "drop_if_generic": False,
            },
        ]

    catalog_items = [
        {
            "id": "dated_exterior_finishes",
            "tier": "optional",
            "drop_if_generic": True,
            "defaultHidden": False,
            "kind": "upgrade",
            "trade_bucket": "exterior_siding_trim",
        },
        {
            "id": "damaged_or_aged_roof_shingles",
            "tier": "work",
            "drop_if_generic": False,
            "defaultHidden": False,
            "kind": "defect",
            "trade_bucket": "roof_gutters",
        },
    ]

    orchestrator = SceneClassifierOrchestrator(
        qwen_config={},
        gpt5_config={},
        vlm_client=FakeOrchestratorClient(),
        candidate_provider=candidate_provider,
        top_k_candidates=5,
        catalog_items=catalog_items,
    )

    result = asyncio.run(orchestrator.analyze_image(
        image_path=Path("photo_002.jpg"),
        options=SceneClassifierRunOptions(),
    ))

    assert result.labeled_forward[0]["label"] == "upgrade_candidate"
    assert provider_contexts[0]["allowed_kinds"] == ["upgrade", "defect"]

    debug_row = result.debug["pass_2d_per_observation"][0]
    assert debug_row["kind_routing"]["original_kind"] == "upgrade"
    assert debug_row["kind_routing"]["expanded_kinds"] == ["upgrade", "defect"]
    assert debug_row["kind_routing"]["reason"] == "visible_condition_signal"
    assert debug_row["resolution_path"] == "llm"
    assert debug_row["shortcut_reason"] is None

    assert result.resolved_items[0]["resolved_item_id"] == "damaged_or_aged_roof_shingles"
    assert result.resolved_items[0]["resolved_kind"] == "defect"
    assert result.verified_issues[0]["catalogItemId"] == "damaged_or_aged_roof_shingles"
    assert result.verified_issues[0]["kind"] == "defect"


def _prompt_return_shape_labels():
    """Pull the label alternatives out of the Pass 2c return-shape example."""
    match = re.search(r'"label":\s*"([^"]+)"', PASS_2C_SYSTEM_PROMPT)
    assert match, "Pass 2c prompt is missing a return-shape label example"
    return set(match.group(1).split("|"))


def test_pass_2c_prompt_example_labels_match_valid_labels():
    assert _prompt_return_shape_labels() == VALID_LABELS


# The upgrade_candidate rule used to enumerate interior finishes only, so
# exterior findings had no example anchor and drifted to generic_presence/other.
# See docs/HANDOFF_pass2c_exterior_recall.md.

def test_pass_2c_prompt_anchors_exterior_finishes():
    prompt = PASS_2C_SYSTEM_PROMPT.lower()

    for anchor in ("siding", "fascia", "soffit", "porch", "masonry"):
        assert anchor in prompt, f"Pass 2c prompt lost its {anchor!r} anchor"


def test_pass_2c_prompt_keeps_interior_finish_anchors():
    """Widening the rule must not displace what it already covered."""
    prompt = PASS_2C_SYSTEM_PROMPT.lower()

    for anchor in ("floors", "cabinets", "counters", "tile", "paint"):
        assert anchor in prompt


def test_pass_2c_prompt_routes_weathered_exterior_finishes_to_upgrade():
    """Deliberate product call: a weathered exterior finish is an upgrade, not a
    defect — it is not an immediate repair.

    Measured against 266 replayed observations, this wording moves the lane
    balance from ~64 defect / ~78 upgrade to ~32 defect / ~102 upgrade. That
    reclassification is the point; do not "fix" it by scoping the anchor back to
    "dated", which reverts the behaviour to parity with the old prompt.
    """
    prompt = PASS_2C_SYSTEM_PROMPT.lower()

    assert "sagging gutter" in prompt
    assert "dated, worn, or weathered exterior finish" in prompt
    assert "upgrade_candidate" in prompt


def test_pass_2c_prompt_has_no_whole_system_absence_rule():
    """Absence safety lives in the catalog deny lists, not in the 2c prompt.

    A prompt-level absence rule over-generalised from gutters to kitchen
    fixtures ("no visible modern vent hood", "no apparent task lighting"),
    where absence is a legitimate priced upgrade. See
    docs/HANDOFF_pass2c_exterior_recall.md.
    """
    assert "whole system is absent" not in PASS_2C_SYSTEM_PROMPT.lower()


def test_pass_2c_forwards_dated_exterior_finish():
    description = "Wood lap siding appears weathered, with staining and aged paint."
    client = FakeTextClient(json.dumps({
        "labeled": [{"description": description, "label": "upgrade_candidate"}]
    }))

    result = asyncio.run(run_pass_2c(
        vlm_client=client,
        model_config={},
        observations=[{"description": description}],
        scene="exterior_front",
    ))

    assert result.labeled_forward == [
        {"description": description, "label": "upgrade_candidate"},
    ]


def test_pass_2c_forwards_visible_gutter_damage():
    """Concrete visible conditions stay forwardable — the absence-safety work
    lives in the catalog deny lists, not in a 2c suppression rule."""
    description = "A downspout is disconnected and terminates at the foundation."
    client = FakeTextClient(json.dumps({
        "labeled": [{"description": description, "label": "defect_or_damage"}]
    }))

    result = asyncio.run(run_pass_2c(
        vlm_client=client,
        model_config={},
        observations=[{"description": description}],
        scene="exterior_front",
    ))

    assert result.labeled_forward == [
        {"description": description, "label": "defect_or_damage"},
    ]


def test_pass_2c_absence_claim_labeled_other_is_dropped():
    description = "No clearly functioning gutter system is visible along the porch edge."
    client = FakeTextClient(json.dumps({
        "labeled": [{"description": description, "label": "other"}]
    }))

    result = asyncio.run(run_pass_2c(
        vlm_client=client,
        model_config={},
        observations=[{"description": description}],
        scene="exterior_front",
    ))

    assert result.labeled_debug == [{"description": description, "label": "other"}]
    assert result.labeled_forward == []


def test_pass_2c_coerce_normalizes_deprecated_safety_label():
    labeled = _coerce_labeled_2c([
        {"description": "Exposed wiring hangs from the ceiling box.", "label": "safety"},
    ])

    assert labeled == [
        {"description": "Exposed wiring hangs from the ceiling box.", "label": "other"},
    ]


def test_pass_2c_coerce_normalizes_unknown_label():
    labeled = _coerce_labeled_2c([
        {"description": "Something the model invented a label for.", "label": "vibes"},
    ])

    assert labeled[0]["label"] == "other"


def test_pass_2c_coerce_warns_on_deprecated_safety_label(caplog):
    with caplog.at_level(logging.WARNING, logger="tools.scene_classifier_passes"):
        _coerce_labeled_2c([
            {"description": "Exposed wiring hangs from the ceiling box.", "label": "safety"},
        ])

    assert any(
        "safety" in record.message and record.levelno == logging.WARNING
        for record in caplog.records
    )


def test_pass_2c_deprecated_safety_result_is_not_forwarded():
    client = FakeTextClient(json.dumps({
        "labeled": [
            {"description": "Exposed wiring hangs from the ceiling box.", "label": "safety"},
        ]
    }))

    result = asyncio.run(run_pass_2c(
        vlm_client=client,
        model_config={},
        observations=[{"description": "Exposed wiring hangs from the ceiling box."}],
    ))

    assert result.labeled_debug[0]["label"] == "other"
    assert result.labeled_forward == []


def test_pass_2c_visible_hazard_labeled_defect_is_forwarded():
    client = FakeTextClient(json.dumps({
        "labeled": [
            {"description": "Exposed wiring hangs from the ceiling box.", "label": "defect_or_damage"},
        ]
    }))

    result = asyncio.run(run_pass_2c(
        vlm_client=client,
        model_config={},
        observations=[{"description": "Exposed wiring hangs from the ceiling box."}],
    ))

    assert result.labeled_forward == [
        {"description": "Exposed wiring hangs from the ceiling box.", "label": "defect_or_damage"},
    ]


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
