"""Catalog search used during blind annotation.

Runs against the real catalog: the point is that an annotator's own phrasing
finds the right item, which a synthetic fixture could not demonstrate.
"""
from tools.benchmarking import catalog_index


class TestSearch:
    def test_finds_by_id_fragment(self, issue_catalog):
        ids = [r["id"] for r in catalog_index.search(issue_catalog, "drywall")]
        assert "damaged_drywall_or_cracks" in ids

    def test_finds_by_annotator_phrasing_via_support_keywords(self, issue_catalog):
        """`support_any` is the catalog's own synonym list, so plain language
        like "water stain" resolves even when the formal name differs."""
        ids = [r["id"] for r in catalog_index.search(issue_catalog, "water stain")]
        assert "water_stain_ceiling" in ids

    def test_ranks_identity_matches_first(self, issue_catalog):
        results = catalog_index.search(issue_catalog, "cabinets", limit=10)
        assert "cabinet" in results[0]["id"] or "cabinet" in results[0]["name"].lower()

    def test_all_terms_must_match(self, issue_catalog):
        assert catalog_index.search(issue_catalog, "drywall zzzznomatch") == []

    def test_empty_query_lists_everything(self, issue_catalog):
        """Browsing a room's plausible items beats guessing keywords."""
        assert len(catalog_index.search(issue_catalog, "", limit=500)) == len(
            [i for i in issue_catalog["items"] if i.get("id")]
        )

    def test_scene_group_filter(self, issue_catalog):
        results = catalog_index.search(issue_catalog, "", scene_group="bathroom", limit=500)
        assert results
        assert all("bathroom" in r["scene_groups"] for r in results)

    def test_kind_filter(self, issue_catalog):
        results = catalog_index.search(issue_catalog, "", kind="upgrade", limit=500)
        assert results
        assert all(r["kind"] == "upgrade" for r in results)

    def test_limit_is_honored(self, issue_catalog):
        assert len(catalog_index.search(issue_catalog, "", limit=3)) == 3

    def test_results_are_deterministic(self, issue_catalog):
        first = catalog_index.search(issue_catalog, "damage", limit=10)
        second = catalog_index.search(issue_catalog, "damage", limit=10)
        assert first == second

    def test_surfaces_the_derived_actionability(self, issue_catalog):
        """The annotator should see what the reference will derive before
        committing to an item."""
        result = catalog_index.search(issue_catalog, "outdated_kitchen_finishes")[0]
        assert result["actionability"] == "modernization"


class TestFormatResults:
    def test_handles_no_matches(self):
        assert catalog_index.format_results([]) == "no matching catalog items"

    def test_includes_id_name_and_actionability(self, issue_catalog):
        text = catalog_index.format_results(
            catalog_index.search(issue_catalog, "drywall", limit=1)
        )
        assert "damaged_drywall_or_cracks" in text
        assert "repair" in text
        assert "Drywall Damage or Cracks" in text
