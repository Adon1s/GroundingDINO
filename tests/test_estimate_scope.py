"""Scope-routing unit tests for tools/estimate_scope.py (three-kind ontology)."""

from tools.estimate_scope import _catalog_text


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
