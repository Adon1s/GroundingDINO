import json
from pathlib import Path

import numpy as np
import pytest

from tools.catalog_embeddings import (
    CatalogEmbeddingsRetriever,
    build_guardrails_from_catalog,
)

_method = CatalogEmbeddingsRetriever._catalog_text


def _retriever_or_skip(catalog):
    # Pin to the in-process SentenceTransformer backend so these smoke tests do not
    # require the llama-server sidecar; they skip cleanly if the model is unavailable.
    try:
        return CatalogEmbeddingsRetriever(catalog, backend="sentence_transformer")
    except Exception as exc:
        pytest.skip(f"sentence-transformers model unavailable: {exc}")


class DeterministicFakeEncoder:
    """Bag-of-words encoder over a fixed vocab â€” deterministic, no model or server.

    Cosine similarity reflects token overlap, so text that shares vocabulary terms
    with a catalog item ranks above text that does not. Used to exercise the
    retriever's ranking + guardrail paths without loading a real embedding model.
    """

    _VOCAB = (
        "faucet", "leak", "water", "tile", "crack",
        "floor", "carpet", "stain", "hardwood", "ceiling",
    )

    def __init__(self):
        self.dimension = len(self._VOCAB)

    def encode(self, texts):
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            tl = (t or "").lower()
            v = np.array([1.0 if tok in tl else 0.0 for tok in self._VOCAB], dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            out[i] = v
        return out


def test_fake_encoder_ranks_related_item_first():
    catalog = {
        "items": [
            {"id": "leaky_faucet", "name": "Leaky faucet", "description": "faucet drips water",
             "kind": "defect", "trade_bucket": "plumbing"},
            {"id": "cracked_tile", "name": "Cracked tile", "description": "tile is cracked floor",
             "kind": "defect", "trade_bucket": "flooring"},
        ]
    }
    retriever = CatalogEmbeddingsRetriever(catalog, encoder=DeterministicFakeEncoder())
    candidates = retriever.retrieve_candidates("the faucet is leaking water", topk=2, allowed_kinds={"defect"})
    assert [c.item_id for c in candidates][0] == "leaky_faucet"


def test_fake_encoder_retrieval_respects_deny_guardrail():
    catalog = {
        "items": [
            {
                "id": "worn_or_stained_carpet",
                "name": "Worn or stained carpet",
                "description": "carpet has stains",
                "kind": "defect",
                "trade_bucket": "flooring",
                "require_any": ["carpet"],
                "deny_any": ["hardwood"],
            }
        ]
    }
    retriever = CatalogEmbeddingsRetriever(
        catalog,
        encoder=DeterministicFakeEncoder(),
        guardrails=build_guardrails_from_catalog(catalog),
    )
    # Observation mentions hardwood â†’ deny_any must filter the carpet item out.
    candidates = retriever.retrieve_candidates("the bedroom has clean hardwood floor", topk=5, allowed_kinds={"defect"})
    assert "worn_or_stained_carpet" not in [c.item_id for c in candidates]


def test_catalog_text_uses_embed_text_when_present():
    item = {
        "id": "x",
        "name": "X Name",
        "description": "X desc",
        "aliases": ["a1"],
        "trade_bucket": "paint",
        "kind": "defect",
        "embed_text": "  Custom tuned embedding string.  ",
    }
    assert _method(None, item) == "Custom tuned embedding string."


def test_catalog_text_falls_back_when_embed_text_missing():
    item = {
        "id": "x",
        "name": "X",
        "description": "d",
        "trade_bucket": "tb",
        "kind": "defect",
    }
    assert _method(None, item) == "X. d. Trade: tb. Kind: defect."


def test_catalog_text_falls_back_when_embed_text_empty_or_whitespace():
    base = {"id": "x", "name": "X", "description": "d", "kind": "defect"}
    for empty in ("", "   ", "\n\t "):
        assert "X. d." in _method(None, {**base, "embed_text": empty})


def test_catalog_text_falls_back_when_embed_text_not_a_string():
    base = {"id": "x", "name": "X", "description": "d", "kind": "defect"}
    for bad in (None, 123, ["a", "b"], {"k": "v"}):
        assert "X. d." in _method(None, {**base, "embed_text": bad})


def test_carpet_guardrails_block_clean_hardwood_negative():
    retriever = object.__new__(CatalogEmbeddingsRetriever)
    retriever.guardrails = build_guardrails_from_catalog({
        "items": [{
            "id": "worn_or_stained_carpet",
            "require_any": ["carpet", "carpeting", "rug"],
            "deny_any": ["hardwood", "tile", "vinyl"],
        }]
    })

    assert retriever._passes_guardrails(
        "The bedroom has clean hardwood floors.",
        "worn_or_stained_carpet",
    ) is False


def test_lighting_negative_is_not_forced_by_support_terms():
    retriever = object.__new__(CatalogEmbeddingsRetriever)
    retriever.guardrails = build_guardrails_from_catalog({
        "items": [{
            "id": "dated_lighting_fixtures",
            "support_any": ["dated light", "old fixture", "ceiling fan"],
            "deny_any": ["modern ceiling fan", "new ceiling fan"],
        }]
    })

    assert retriever._passes_guardrails(
        "The living room has a modern ceiling fan.",
        "dated_lighting_fixtures",
    ) is False


def test_guardrail_terms_do_not_match_inside_longer_words():
    """Guardrail terms are word-start anchored, not raw substrings.

    Both directions were live in the shipped catalog: `pest_or_rodent_evidence`
    requires "rat", which substring-matched "discolo(rat)ion" (324 corpus hits),
    and `worn_or_stained_flooring` denies "trip", which substring-matched
    "s(trip)ped" (38 hits) and blocked a real flooring defect.
    """
    retriever = object.__new__(CatalogEmbeddingsRetriever)
    retriever.guardrails = build_guardrails_from_catalog({
        "items": [
            {"id": "pest_or_rodent_evidence", "require_any": ["pest", "rodent", "rat"]},
            {"id": "worn_or_stained_flooring", "deny_any": ["trip", "unlevel", "lip"]},
        ]
    })

    assert retriever._passes_guardrails(
        "Discoloration is visible on the ceiling.",
        "pest_or_rodent_evidence",
    ) is False

    assert retriever._passes_guardrails(
        "An exposed baseboard strip is located on the left wall.",
        "worn_or_stained_flooring",
    ) is True


def test_real_catalog_marked_terms_do_not_fire_on_prefixed_words():
    """Leading \\b cannot stop a term matching a word it *prefixes*.

    `mold` sat inside "crown molding" in 18 of the 4,572 corpus observations â€”
    as many as the 18 genuine mold observations â€” so four interior items
    deny-blocked every crown-molding description. Those terms now carry the
    trailing-boundary marker; this locks both directions against the real
    catalog.
    """
    catalog_path = Path(__file__).resolve().parent.parent / "tools" / "issue_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    retriever = object.__new__(CatalogEmbeddingsRetriever)
    retriever.guardrails = build_guardrails_from_catalog(catalog)

    molding = "Dated style and color of crown molding and baseboards."
    for item_id in (
        "wall_scuffs_marks_or_dents",
        "dated_wallpaper_present",
        "dated_overall_decor_style",
    ):
        assert retriever._passes_guardrails(molding, item_id) is True, item_id

    # Real mold must still deny-block those same items.
    real_mold = "Visible mold, mildew, and water staining along lower walls."
    for item_id in ("wall_scuffs_marks_or_dents", "dated_wallpaper_present"):
        assert retriever._passes_guardrails(real_mold, item_id) is False, item_id

    # `wall$` must not let "wallpaper" satisfy the paint-refresh require gate,
    # while the separately authored `walls` keeps plural coverage.
    assert retriever._passes_guardrails(
        "A partially visible bathroom has older wallpaper.",
        "bathroom_paint_refresh_recommended",
    ) is False
    assert retriever._passes_guardrails(
        "The walls need a repaint throughout.",
        "bathroom_paint_refresh_recommended",
    ) is True


def test_guardrail_terms_still_match_inflections():
    """Word-start anchoring keeps terms authorable as stems: a `stain` term must
    still fire on "stained"/"stains", which is why leading-only \\b was chosen
    over full \\b...\\b boundaries."""
    retriever = object.__new__(CatalogEmbeddingsRetriever)
    retriever.guardrails = build_guardrails_from_catalog({
        "items": [{"id": "clean_floor_only", "deny_any": ["stain"]}]
    })

    for text in ("The carpet is stained.", "There are stains near the door.", "Heavy staining."):
        assert retriever._passes_guardrails(text, "clean_floor_only") is False, text


def test_real_catalog_require_terms_survive_word_anchoring():
    """Lock the catalog terms added alongside the anchoring change.

    Anchoring drops the substring matches these items used to rely on
    ("wall" inside "drywall", "paint" inside "repainting", "tub" inside
    "bathtub"), so the terms are now authored explicitly.
    """
    catalog_path = Path(__file__).resolve().parent.parent / "tools" / "issue_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    retriever = object.__new__(CatalogEmbeddingsRetriever)
    retriever.guardrails = build_guardrails_from_catalog(catalog)

    cases = [
        ("bathroom_paint_refresh_recommended", "A large section of missing drywall is on the right side."),
        ("bathroom_paint_refresh_recommended", "Ceiling and wall areas need repair and repainting."),
        ("peeling_or_damaged_bathroom_paint", "The bathtub looks older, stained, and worn."),
        # `tub$` still covers the bare noun it was marked for.
        ("peeling_or_damaged_bathroom_paint", "The tub surround is peeling and stained."),
        # Terms added to reopen gates the corpus was hitting but the catalog
        # did not carry: the item's own description is "vegetation contacting
        # or very close to siding/roof/foundation", but it had no shrub/bush.
        ("trees_or_vegetation_too_close", "Overgrown shrubs are touching the brick foundation area."),
        ("trees_or_vegetation_too_close", "Bushes on the left side of the house are overgrown."),
        # Terms were authored adjective-noun ("cramped layout") while the VLM
        # writes predicate order, so the gate never opened.
        ("layout_modernization_opportunity", "The layout is cramped with limited counter space."),
    ]
    for item_id, observation in cases:
        assert item_id in retriever.guardrails, f"{item_id} has no guardrails â€” case passes vacuously"
        assert retriever._passes_guardrails(observation, item_id) is True, (item_id, observation)


def test_catalog_uses_generic_bedroom_living_issue_ids():
    catalog_path = Path(__file__).resolve().parent.parent / "tools" / "issue_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    ids = {item.get("id") for item in catalog.get("items", []) if isinstance(item, dict)}

    assert {
        "worn_or_stained_carpet",
        "older_flooring_style",
        "damaged_drywall_or_cracks",
        "water_stain_ceiling",
        "popcorn_or_acoustic_ceiling_texture",
    } <= ids
    assert not ({
        "worn_or_dated_bedroom_carpet",
        "worn_or_dated_living_carpet",
        "dated_bedroom_flooring_style",
        "dated_living_flooring_style",
        "bedroom_drywall_damage_or_holes",
        "living_drywall_damage_or_holes",
        "bedroom_water_stain",
        "living_water_stain",
        "bedroom_popcorn_ceiling",
        "living_popcorn_ceiling",
    } & ids)


# ---------------------------------------------------------------------------
# Smoke tests: full retriever instantiation. These load sentence-transformers
# and download the embedding model on first run, so they are slower than the
# unit tests above. They skip cleanly if the package is not installed.
# ---------------------------------------------------------------------------


def test_require_any_blocks_unmatched_observation():
    """A catalog item whose `require_any` token is absent from the observation
    must be filtered out of retrieval results â€” exercises the audit path that
    builds guardrails from the catalog and enforces them inside the retriever.
    """
    pytest.importorskip("sentence_transformers")

    catalog = {
        "items": [
            {
                "id": "impossible_item",
                "name": "Impossible Item",
                "description": "An item that requires an impossible token to match.",
                "kind": "defect",
                "trade_bucket": "general",
                "require_any": ["xyzzy_impossible_token"],
            }
        ]
    }
    guardrails = build_guardrails_from_catalog(catalog)
    retriever = _retriever_or_skip(catalog)
    retriever.guardrails = guardrails

    observation = (
        "There is a stained, peeling vinyl floor in the kitchen that needs replacement."
    )
    candidates = retriever.retrieve_candidates(observation, topk=10, allowed_kinds={"defect"})

    returned_ids = [c.item_id for c in candidates]
    assert "impossible_item" not in returned_ids, (
        f"impossible_item should be filtered by require_any guardrail; got {returned_ids}"
    )


def test_embed_text_drives_retrieval():
    """An item with generic name/description but a distinctive phrase only in
    `embed_text` must outrank a decoy whose name/description compete on the
    observation's domain words. This only succeeds if `embed_text` is actually
    being embedded â€” with the pre-fix code, the target embeds as
    'Item A. An item.' and would lose to the decoy.
    """
    pytest.importorskip("sentence_transformers")

    catalog = {
        "items": [
            {
                "id": "embed_text_target",
                "name": "Item A",
                "description": "An item.",
                "kind": "defect",
                "trade_bucket": "general",
                "embed_text": (
                    "Octocat plumbing manifold leaking under the sink. "
                    "Replace deteriorated octocat manifold assembly."
                ),
            },
            {
                "id": "decoy_plumbing",
                "name": "Generic Plumbing Issue",
                "description": "A plumbing issue requiring a plumber.",
                "kind": "defect",
                "trade_bucket": "general",
            },
        ]
    }
    retriever = _retriever_or_skip(catalog)

    observation = "Octocat plumbing manifold under the kitchen sink is leaking water."
    candidates = retriever.retrieve_candidates(observation, topk=2, allowed_kinds={"defect"})

    assert candidates, "expected at least one candidate"
    top_ids = [c.item_id for c in candidates]
    assert top_ids[0] == "embed_text_target", (
        f"expected embed_text_target as top match (only succeeds if embed_text is being "
        f"embedded â€” without the fix, the target embeds as 'Item A. An item.' which would "
        f"not outrank the decoy); got {top_ids}"
    )


# â”€â”€ strict kind-filter semantics (observation-kind-v2, Task 2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Contract: allowed_kinds=None means deliberately unfiltered; an EMPTY set or a
# set of unknown kinds returns NO candidates. The retired bug treated both as
# "no filter" and silently searched the whole catalog.

_THREE_KIND_CATALOG = {
    "items": [
        {"id": "faucet_leaking", "name": "Leaking faucet", "description": "faucet drips water",
         "kind": "defect", "trade_bucket": "plumbing"},
        {"id": "carpet_worn", "name": "Worn carpet", "description": "carpet stain floor",
         "kind": "degradation", "trade_bucket": "flooring"},
        {"id": "tile_dated_style", "name": "Dated tile style", "description": "tile floor dated",
         "kind": "modernization", "trade_bucket": "flooring"},
    ]
}


def _three_kind_retriever():
    return CatalogEmbeddingsRetriever(_THREE_KIND_CATALOG, encoder=DeterministicFakeEncoder())


def test_retrieve_none_kind_filter_is_deliberately_unfiltered():
    retriever = _three_kind_retriever()
    hits = retriever.retrieve_candidates("stain on the tile floor", allowed_kinds=None, topk=10)
    ids = {c.item_id for c in hits}
    # None means no kind filter: matches from more than one kind are reachable.
    assert {"carpet_worn", "tile_dated_style"} <= ids


def test_retrieve_empty_kind_filter_returns_no_candidates():
    retriever = _three_kind_retriever()
    assert retriever.retrieve_candidates("stain on the tile floor", allowed_kinds=set(), topk=10) == []


def test_retrieve_unknown_kind_filter_returns_no_candidates():
    retriever = _three_kind_retriever()
    assert retriever.retrieve_candidates("stain on the tile floor", allowed_kinds={"safety"}, topk=10) == []
    assert retriever.retrieve_candidates("stain on the tile floor", allowed_kinds={"upgrade"}, topk=10) == []


def test_retrieve_mixed_unknown_and_known_kinds_filters_to_known():
    retriever = _three_kind_retriever()
    hits = retriever.retrieve_candidates(
        "stain on the carpet floor", allowed_kinds={"degradation", "nonsense"}, topk=10
    )
    assert [c.item_id for c in hits] == ["carpet_worn"]


def test_provider_explicit_empty_allowed_kinds_returns_no_candidates():
    from tools.catalog_embeddings import make_candidate_provider

    provider = make_candidate_provider(_three_kind_retriever())
    assert provider("stain on the tile floor", {"allowed_kinds": [], "top_k_candidates": 10}) == []


def test_provider_unknown_kind_never_searches_the_whole_catalog():
    """The headline regression: a retired kind ('upgrade') against a three-kind
    index must yield zero candidates, not a silent whole-catalog search."""
    from tools.catalog_embeddings import make_candidate_provider

    provider = make_candidate_provider(_three_kind_retriever())
    assert provider("stain on the tile floor", {"kind": "upgrade", "top_k_candidates": 10}) == []
    assert provider("stain on the tile floor", {"allowed_kinds": ["upgrade"], "top_k_candidates": 10}) == []


def test_provider_single_kind_context_searches_exactly_that_kind():
    from tools.catalog_embeddings import make_candidate_provider

    provider = make_candidate_provider(_three_kind_retriever())
    hits = provider("stain on the carpet floor", {"kind": "degradation", "top_k_candidates": 10})
    assert [c["item_id"] for c in hits] == ["carpet_worn"]
    assert all(c["kind"] == "degradation" for c in hits)


def test_provider_without_kind_or_filter_is_unfiltered():
    from tools.catalog_embeddings import make_candidate_provider

    provider = make_candidate_provider(_three_kind_retriever())
    hits = provider("stain on the tile floor", {"top_k_candidates": 10})
    assert len(hits) >= 2  # crosses kinds: degradation + modernization both match


def test_build_items_skips_items_without_a_kind():
    """No silent kind coercion at index time: a kindless item is excluded from
    the index instead of defaulting to 'defect'."""
    catalog = {
        "items": [
            {"id": "faucet_leaking", "name": "Leaking faucet", "description": "faucet drips water",
             "kind": "defect", "trade_bucket": "plumbing"},
            {"id": "kindless_item", "name": "No kind", "description": "water faucet leak"},
        ]
    }
    retriever = CatalogEmbeddingsRetriever(catalog, encoder=DeterministicFakeEncoder())
    hits = retriever.retrieve_candidates("the faucet is leaking water", allowed_kinds=None, topk=10)
    assert "kindless_item" not in [c.item_id for c in hits]

