"""Lexical term matching — the single matcher behind every catalog keyword list."""
import pytest

from tools.pipeline_common import (
    TERM_WHOLE_WORD_MARKER,
    strip_term_marker,
    term_matches,
)


def test_bare_term_is_word_start_anchored():
    """The default is leading \\b only, so terms stay authorable as stems."""
    assert term_matches("stain", "the carpet is stained") is True
    assert term_matches("stain", "there are stains near the door") is True
    assert term_matches("stain", "heavy staining on the ceiling") is True

    # ...but no longer fires inside a longer word.
    assert term_matches("ding", "vertical siding is installed") is False
    assert term_matches("rat", "discoloration is visible") is False


def test_multi_word_terms_match_across_spaces():
    """re.escape preserves internal spaces, so phrases work unchanged."""
    assert term_matches("water stain", "a water stain is on the ceiling") is True
    assert term_matches("water stain", "water staining above the window") is True
    assert term_matches("water stain", "a stain is on the ceiling") is False


def test_marker_adds_a_trailing_boundary():
    """A trailing "$" opts the term in to a closing \\b.

    `mold` prefix-matched "crown molding" in 18 of the 4,572 corpus
    observations — exactly as many as the 18 real mold observations — which
    both deny-blocked interior items and force-resolved crown molding as a
    moisture defect. The marker is per-term because a global trailing \\b drops
    `require_any` to zero for the items whose terms are singular stems.
    """
    assert term_matches("mold$", "black mold and mildew are visible") is True
    assert term_matches("mold$", "mold growth along the lower wall") is True
    assert term_matches("mold$", "crown molding is basic in style") is False
    assert term_matches("mold$", "the moldings were replaced") is False

    # The unmarked stem is what used to fire on both.
    assert term_matches("mold", "crown molding is basic in style") is True


def test_marker_treats_hyphen_as_a_boundary():
    """\\b does not treat "-" as a word char, so hyphenated forms still match.

    "mildew or mold-like buildup" is a real mold observation and must survive
    the marker.
    """
    assert term_matches("mold$", "mildew or mold-like buildup in the tub") is True
    assert term_matches("wall$", "wall-to-wall carpet is worn") is True


def test_marker_on_the_shipped_collision_terms():
    """The four terms the catalog actually marks, and the words they exclude."""
    cases = [
        ("wall$", "scuffs are visible on the wall", "older wallpaper is present"),
        ("tub$", "the tub surround is cracked", "a drain tube is visible"),
        ("rat$", "a rat was observed", "the slope runs toward the house rather than away"),
    ]
    for term, should_match, should_not in cases:
        assert term_matches(term, should_match) is True, (term, should_match)
        assert term_matches(term, should_not) is False, (term, should_not)


def test_marker_does_not_break_plural_coverage_when_authored():
    """`wall$` excludes "walls", which is why both items carry `walls` too."""
    assert term_matches("wall$", "the walls are scuffed") is False
    assert term_matches("walls", "the walls are scuffed") is True


@pytest.mark.parametrize("term", ["", TERM_WHOLE_WORD_MARKER])
def test_empty_and_bare_marker_never_match(term):
    assert term_matches(term, "any text at all") is False


def test_strip_term_marker_round_trip():
    assert strip_term_marker("mold$") == "mold"
    assert strip_term_marker("mold") == "mold"
    assert strip_term_marker("water stain$") == "water stain"
    assert strip_term_marker("") == ""
    assert strip_term_marker(None) == ""
