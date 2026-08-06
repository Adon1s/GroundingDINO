"""Scope-routing unit tests for tools/estimate_scope.py (three-kind ontology)."""

import pytest

from tools.estimate_scope import (
    MARKETABILITY_REHAB,
    OPTIONAL_VALUE_ADD,
    REQUIRED_REHAB,
    _catalog_text,
    _classify_baseline_scope_with_reason,
    _is_visible_required_condition,
)


def _classify(kind, *, tier="", category="general", severity=2, name="plain item"):
    candidate = {"kind": kind, "severity": severity}
    catalog_item = {
        "id": "sample_entry",
        "name": name,
        "tier": tier,
        "category": category,
        "severity": severity,
    }
    return _classify_baseline_scope_with_reason(candidate, catalog_item)


class TestCatalogTextExcludesKind:
    """The kind label must never reach the scope term matchers: under
    observation-kind-v2 the literal kind string "modernization" would
    self-match _VALUE_ADD_TERMS and flip scope routing (Task 3, A3)."""

    def test_candidate_kind_not_in_matched_text(self):
        candidate = {
            "kind": "modernization",
            "catalog_item_id": "outdated_kitchen_finishes",
            "scope": "cosmetic",
        }
        text = _catalog_text(candidate, {})
        assert "modernization" not in text
        assert "outdated_kitchen_finishes" in text

    def test_catalog_item_kind_never_included(self):
        text = _catalog_text({}, {"id": "some_item", "kind": "modernization"})
        assert "modernization" not in text


class TestBaselineScopeThreeKinds:
    """Branch table for _classify_baseline_scope_with_reason. v1-reachable
    reason strings are pinned byte-identical; degradation mirrors the defect
    gates (Task 3 decision 4); modernization plays the old upgrade role."""

    # ── v1 rows: byte-identical ──
    @pytest.mark.parametrize("kind,kwargs,expected", [
        ("defect", dict(severity=3), (REQUIRED_REHAB, "defect_severity_threshold")),
        ("defect", dict(category="moisture"), (REQUIRED_REHAB, "required_category")),
        ("defect", dict(name="rotted fascia board"), (REQUIRED_REHAB, "required_condition_signal")),
        ("defect", dict(category="cosmetic"), (MARKETABILITY_REHAB, "cosmetic_defect_marketability")),
        ("defect", dict(), (REQUIRED_REHAB, "defect_default")),
        ("defect", dict(tier="optional"), (OPTIONAL_VALUE_ADD, "optional_defect")),
        ("upgrade", dict(category="cosmetic"), (MARKETABILITY_REHAB, "upgrade_marketability")),
        ("upgrade", dict(), (MARKETABILITY_REHAB, "upgrade_default")),
        ("upgrade", dict(tier="optional"), (OPTIONAL_VALUE_ADD, "optional_upgrade")),
        ("upgrade", dict(tier="optional", name="layout reconfiguration"),
         (OPTIONAL_VALUE_ADD, "optional_value_add_signal")),
    ])
    def test_v1_rows_unchanged(self, kind, kwargs, expected):
        assert _classify(kind, **kwargs) == expected

    # ── degradation mirrors the defect gates ──
    @pytest.mark.parametrize("kwargs,expected", [
        (dict(severity=3), (REQUIRED_REHAB, "degradation_severity_threshold")),
        (dict(severity=4), (REQUIRED_REHAB, "degradation_severity_threshold")),
        (dict(category="moisture"), (REQUIRED_REHAB, "required_category")),
        (dict(name="rotted fascia board"), (REQUIRED_REHAB, "required_condition_signal")),
        (dict(), (MARKETABILITY_REHAB, "degradation_default")),
        (dict(name="worn stained carpet"), (MARKETABILITY_REHAB, "degradation_default")),
        (dict(tier="optional"), (OPTIONAL_VALUE_ADD, "optional_degradation")),
    ])
    def test_degradation_rows(self, kwargs, expected):
        assert _classify("degradation", **kwargs) == expected

    # ── modernization takes the old upgrade role ──
    @pytest.mark.parametrize("kwargs,expected", [
        (dict(category="cosmetic"), (MARKETABILITY_REHAB, "modernization_marketability")),
        (dict(), (MARKETABILITY_REHAB, "modernization_default")),
        (dict(tier="optional"), (OPTIONAL_VALUE_ADD, "optional_modernization")),
        # severity/structural gates must NOT promote modernization to required
        (dict(severity=5, category="moisture"), (MARKETABILITY_REHAB, "modernization_default")),
    ])
    def test_modernization_rows(self, kwargs, expected):
        assert _classify("modernization", **kwargs) == expected

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown kind"):
            _classify("flooble")

    def test_empty_kind_keeps_v1_fallthrough(self):
        assert _classify("") == (REQUIRED_REHAB, "fallback_required")
        assert _classify("", category="cosmetic") == (
            MARKETABILITY_REHAB, "marketability_signal")


class TestVisibleRequiredCondition:
    """Visible deterioration must not be downgraded to inspection_risk for
    degradation (Task 3, A5); modernization stays ineligible."""

    def _item(self, kind):
        return (
            {"kind": kind, "severity": 3},
            {"id": "x", "name": "active leak below sink", "tier": "work",
             "category": "moisture", "severity": 3},
        )

    def test_degradation_with_visible_damage_is_required(self):
        assert _is_visible_required_condition(*self._item("degradation")) is True

    def test_defect_unchanged(self):
        assert _is_visible_required_condition(*self._item("defect")) is True

    def test_modernization_ineligible(self):
        assert _is_visible_required_condition(*self._item("modernization")) is False
