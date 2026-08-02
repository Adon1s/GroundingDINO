"""
Tests for tools.project_scopes — project scope taxonomy and mapping validation.

Validates that:
- Every catalog trade_bucket maps to exactly one project scope
- Every mapping target is a valid project scope id
- TRADE_MULT and TRADE_TO_SUBSCORE are fully covered
- strict vs non-strict mode behaves correctly
- The interior_paint_drywall duplicate has been cleaned up
"""

import json
from pathlib import Path

import pytest

from tools.project_scopes import (
    PROJECT_SCOPE_IDS,
    PROJECT_SCOPES,
    TRADE_BUCKET_TO_PROJECT_SCOPE,
    get_project_scope,
    get_project_scope_name,
)
from tools.costing import TRADE_MULT, TRADE_TO_SUBSCORE


# ─── Helpers ──────────────────────────────────────────────────────────────────

CATALOG_PATH = Path(__file__).resolve().parent.parent / "tools" / "issue_catalog.json"


def _load_catalog_trade_bucket_ids():
    """Load the trade_bucket ids defined in the catalog JSON."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    return {tb["id"] for tb in catalog["trade_buckets"]}


def _load_catalog_item_trade_buckets():
    """Load all trade_bucket values actually used by catalog items."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    return {item["trade_bucket"] for item in catalog["items"] if "trade_bucket" in item}


# ─── Mapping completeness ────────────────────────────────────────────────────

class TestMappingCompleteness:
    """Every trade_bucket must map to a valid project scope, and vice versa."""

    def test_every_catalog_bucket_has_mapping(self):
        """Every trade_bucket defined in issue_catalog.json has a project scope mapping."""
        catalog_ids = _load_catalog_trade_bucket_ids()
        unmapped = catalog_ids - set(TRADE_BUCKET_TO_PROJECT_SCOPE.keys())
        assert not unmapped, f"Catalog trade_buckets without project_scope mapping: {unmapped}"

    def test_every_mapping_key_exists_in_catalog(self):
        """Every key in the mapping exists in the catalog's trade_buckets array."""
        catalog_ids = _load_catalog_trade_bucket_ids()
        extra = set(TRADE_BUCKET_TO_PROJECT_SCOPE.keys()) - catalog_ids
        assert not extra, f"Mapping keys not in catalog trade_buckets: {extra}"

    def test_every_item_trade_bucket_has_mapping(self):
        """Every trade_bucket actually used by a catalog item has a mapping."""
        item_buckets = _load_catalog_item_trade_buckets()
        unmapped = item_buckets - set(TRADE_BUCKET_TO_PROJECT_SCOPE.keys())
        assert not unmapped, f"Catalog items use unmapped trade_buckets: {unmapped}"

    def test_every_trade_mult_key_has_mapping(self):
        """Every key in TRADE_MULT has a project scope mapping."""
        unmapped = set(TRADE_MULT.keys()) - set(TRADE_BUCKET_TO_PROJECT_SCOPE.keys())
        assert not unmapped, f"TRADE_MULT keys without project_scope mapping: {unmapped}"

    def test_every_trade_to_subscore_key_has_mapping(self):
        """Every key in TRADE_TO_SUBSCORE has a project scope mapping."""
        unmapped = set(TRADE_TO_SUBSCORE.keys()) - set(TRADE_BUCKET_TO_PROJECT_SCOPE.keys())
        assert not unmapped, f"TRADE_TO_SUBSCORE keys without project_scope mapping: {unmapped}"


# ─── Mapping validity ────────────────────────────────────────────────────────

class TestMappingValidity:
    """All mapped values are valid project scope ids."""

    def test_all_mapped_values_are_valid_scope_ids(self):
        invalid = {
            v for v in TRADE_BUCKET_TO_PROJECT_SCOPE.values()
            if v not in PROJECT_SCOPE_IDS
        }
        assert not invalid, f"Mapping points to invalid scope ids: {invalid}"

    def test_every_scope_has_at_least_one_bucket(self):
        mapped_scopes = set(TRADE_BUCKET_TO_PROJECT_SCOPE.values())
        orphan = PROJECT_SCOPE_IDS - mapped_scopes
        assert not orphan, f"Project scopes with no trade buckets mapped: {orphan}"

    def test_project_scopes_have_unique_ids(self):
        ids = [s["id"] for s in PROJECT_SCOPES]
        assert len(ids) == len(set(ids)), f"Duplicate project scope ids: {ids}"


# ─── Lookup helpers ───────────────────────────────────────────────────────────

class TestGetProjectScope:
    """get_project_scope strict vs non-strict behavior."""

    def test_known_bucket_returns_correct_scope(self):
        assert get_project_scope("flooring") == "interior_generalist"
        assert get_project_scope("plumbing") == "mechanical"
        assert get_project_scope("roof_gutters") == "exterior"
        assert get_project_scope("foundation_structure") == "structure"
        assert get_project_scope("moisture_mold") == "remediation"
        assert get_project_scope("landscaping_drains") == "site_drainage"
        assert get_project_scope("kitchen_cabinets_counters") == "kitchen_bath"

    def test_strict_raises_on_unmapped_bucket(self):
        with pytest.raises(KeyError, match="no project_scope mapping"):
            get_project_scope("nonexistent_bucket", strict=True)

    def test_nonstrict_returns_unknown_on_unmapped_bucket(self):
        result = get_project_scope("nonexistent_bucket", strict=False)
        assert result == "unknown"


class TestGetProjectScopeName:
    """get_project_scope_name display name lookups."""

    def test_known_scope_returns_display_name(self):
        assert get_project_scope_name("interior_generalist") == "Interior Generalist"
        assert get_project_scope_name("kitchen_bath") == "Kitchen & Bath"
        assert get_project_scope_name("mechanical") == "Mechanical Trades"

    def test_unknown_scope_returns_title_case_fallback(self):
        assert get_project_scope_name("some_new_scope") == "Some New Scope"


# ─── Duplicate cleanup verification ──────────────────────────────────────────

class TestDuplicateCleanup:
    """Verify the interior_paint_drywall duplicate has been merged."""

    def test_interior_paint_drywall_not_in_catalog_buckets(self):
        catalog_ids = _load_catalog_trade_bucket_ids()
        assert "interior_paint_drywall" not in catalog_ids, (
            "interior_paint_drywall should have been merged into paint_drywall"
        )

    def test_no_catalog_items_use_interior_paint_drywall(self):
        item_buckets = _load_catalog_item_trade_buckets()
        assert "interior_paint_drywall" not in item_buckets, (
            "No catalog items should reference interior_paint_drywall"
        )

    def test_interior_paint_drywall_not_in_mapping(self):
        assert "interior_paint_drywall" not in TRADE_BUCKET_TO_PROJECT_SCOPE, (
            "interior_paint_drywall should not be in the project scope mapping"
        )
