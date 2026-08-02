"""
Tests for the pass registry.

PassToggles and PassModelOverrides used to hand-write the same 13 pass keys in
four places (two to_dict, two from_dict) plus a fifth copy in analyzer_cli. The
serializers are now generated from ALL_PASSES; these tests pin the defaults and
round-trip behavior that generation has to preserve.
"""
import pytest

from tools.pass_config import (
    ALL_PASSES,
    PASS_DESCRIPTIONS,
    PREMIUM_MODEL_MAP,
    STANDARD_MODEL_MAP,
    PassModelOverrides,
    PassToggles,
)

DEFAULT_TOGGLES = {key: True for key in ALL_PASSES}


# ── registry shape ──────────────────────────────────────────────────────────

def test_all_passes_is_the_single_key_list():
    assert ALL_PASSES == ('1a', '1b', '1c', '2a', '2b', '2c', '2d', '2e', '2f')


def test_analyzer_cli_uses_the_shared_list():
    """analyzer_cli kept its own literal copy for CLI flag generation."""
    import tools.analyzer_cli as analyzer_cli

    assert analyzer_cli.ALL_PASSES is ALL_PASSES


def test_the_pass_4_family_is_gone():
    """4/4a/4b/4c were registry entries with no implementation anywhere."""
    for dead in ('4', '4a', '4b', '4c'):
        assert dead not in ALL_PASSES
        assert dead not in PASS_DESCRIPTIONS
        assert dead not in PREMIUM_MODEL_MAP
        assert dead not in STANDARD_MODEL_MAP
        assert not hasattr(PassToggles(), f'pass_{dead}')
        assert not hasattr(PassModelOverrides(), f'model_{dead}')


def test_every_pass_is_enabled_by_default():
    """There is no dormant tier left; all remaining passes default on."""
    assert PassToggles().to_dict() == DEFAULT_TOGGLES
    assert all(PassToggles().to_dict().values())


@pytest.mark.parametrize("registry", [PASS_DESCRIPTIONS, PREMIUM_MODEL_MAP, STANDARD_MODEL_MAP])
def test_side_tables_cover_exactly_the_registry(registry):
    assert set(registry) == set(ALL_PASSES)


# ── PassToggles ─────────────────────────────────────────────────────────────

def test_toggles_to_dict_keys_are_all_passes_in_order():
    assert list(PassToggles().to_dict()) == list(ALL_PASSES)


def test_toggles_defaults_match_the_registry():
    assert PassToggles().to_dict() == DEFAULT_TOGGLES


@pytest.mark.parametrize("empty", [None, {}])
def test_toggles_from_empty_is_defaults(empty):
    assert PassToggles.from_dict(empty).to_dict() == DEFAULT_TOGGLES


def test_toggles_round_trip():
    original = PassToggles(pass_2d=False, pass_2f=False)
    assert PassToggles.from_dict(original.to_dict()) == original


def test_toggles_partial_dict_fills_from_defaults():
    result = PassToggles.from_dict({'2d': False}).to_dict()
    assert result['2d'] is False
    # untouched keys keep their registry default
    assert result['1a'] is True
    assert result['2f'] is True


def test_stored_pass_4_toggles_are_ignored_not_fatal():
    """Historical artifacts carry pass_toggles with 4/4a/4b/4c keys."""
    result = PassToggles.from_dict({'2d': False, '4': True, '4a': False}).to_dict()
    assert result == {**DEFAULT_TOGGLES, '2d': False}


def test_toggles_ignores_unknown_keys():
    assert PassToggles.from_dict({'bogus': True, '99z': False}).to_dict() == DEFAULT_TOGGLES


def test_toggles_getitem_defaults_false_for_unknown_pass():
    assert PassToggles()['nope'] is False


# ── PassModelOverrides ──────────────────────────────────────────────────────

def test_overrides_to_dict_keys_are_all_passes_in_order():
    assert list(PassModelOverrides().to_dict()) == list(ALL_PASSES)


def test_overrides_default_to_none():
    assert PassModelOverrides().to_dict() == {key: None for key in ALL_PASSES}


@pytest.mark.parametrize("empty", [None, {}])
def test_overrides_from_empty_is_all_none(empty):
    assert PassModelOverrides.from_dict(empty).to_dict() == {key: None for key in ALL_PASSES}


def test_overrides_round_trip():
    original = PassModelOverrides(model_2a="gpt-5.6-sol", model_2f="gpt-5.4-mini")
    assert PassModelOverrides.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("raw,expected", [
    ("  gpt-5.6-sol  ", "gpt-5.6-sol"),   # trimmed
    ("gpt-5.6-sol", "gpt-5.6-sol"),
    ("", None),                            # blank is not an override
    ("   ", None),
    (None, None),
    (7, None),                             # non-strings ignored
    (True, None),
])
def test_overrides_normalize_model_names(raw, expected):
    assert PassModelOverrides.from_dict({'2a': raw}).to_dict()['2a'] == expected


def test_overrides_ignores_unknown_keys():
    result = PassModelOverrides.from_dict({'bogus': 'gpt-5.6-sol'}).to_dict()
    assert result == {key: None for key in ALL_PASSES}


def test_overrides_getitem_defaults_none_for_unknown_pass():
    assert PassModelOverrides()['nope'] is None
