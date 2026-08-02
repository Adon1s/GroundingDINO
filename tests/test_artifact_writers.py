"""Tests for property metadata loading helpers in tools.artifact_writers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from tools.artifact_writers import (
    _extract_property_metadata_from_mapping,
    _load_scrape_metadata_for_job,
    _resolve_property_metadata,
)


def _make_property_layout(
    tmp_path: Path,
    property_key: str,
    *,
    batch: str = "batch_a",
) -> tuple[Path, Path]:
    """Create the realtorvision-style on-disk layout and return (root, image_path)."""
    image_path = (
        tmp_path
        / "public"
        / "images"
        / "properties"
        / property_key
        / "img.jpg"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"")

    scrape_path = (
        tmp_path
        / "data"
        / "scraped"
        / "batches"
        / batch
        / "properties"
        / property_key
        / "scrape.json"
    )
    scrape_path.parent.mkdir(parents=True, exist_ok=True)
    return tmp_path, image_path


def _write_scrape(scrape_path: Path, payload: dict) -> None:
    scrape_path.parent.mkdir(parents=True, exist_ok=True)
    with scrape_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _job_for(image_path: Path, property_key: str):
    return SimpleNamespace(
        property_key=property_key,
        results=[SimpleNamespace(image_path=str(image_path))],
        property_metadata=None,
    )


def test_load_scrape_metadata_picks_latest_by_mtime(tmp_path: Path):
    property_key = "redfin_test_001"
    root, image_path = _make_property_layout(tmp_path, property_key, batch="batch_a")

    older = (
        root / "data" / "scraped" / "batches" / "batch_a"
        / "properties" / property_key / "scrape.json"
    )
    newer_dir = (
        root / "data" / "scraped" / "batches" / "batch_b"
        / "properties" / property_key
    )
    newer_dir.mkdir(parents=True, exist_ok=True)
    newer = newer_dir / "scrape.json"

    _write_scrape(older, {"price": 100000, "beds": 3})
    _write_scrape(newer, {"price": 250000, "beds": 4})

    # Force older to be older on disk (Windows/macOS time resolution friendly).
    old_time = newer.stat().st_mtime - 60
    os.utime(older, (old_time, old_time))

    metadata = _load_scrape_metadata_for_job(_job_for(image_path, property_key))

    assert metadata["list_price"] == 250000
    assert metadata["bedrooms"] == 4
    assert metadata["metadata_source"] == "scrape_json"
    assert metadata["metadata_path"] == str(newer)


def test_alias_normalization_price_beds_baths_sqft(tmp_path: Path):
    property_key = "redfin_test_002"
    root, image_path = _make_property_layout(tmp_path, property_key)
    scrape_path = (
        root / "data" / "scraped" / "batches" / "batch_a"
        / "properties" / property_key / "scrape.json"
    )
    _write_scrape(scrape_path, {
        "price": 320000,
        "beds": 4,
        "baths": 2.5,
        "sqft": 1800,
    })

    metadata = _load_scrape_metadata_for_job(_job_for(image_path, property_key))

    assert metadata["list_price"] == 320000
    assert metadata["bedrooms"] == 4
    assert metadata["bath_count"] == 2.5
    assert metadata["square_feet"] == 1800


def test_resolve_property_metadata_job_overrides_scrape(tmp_path: Path):
    property_key = "redfin_test_003"
    root, image_path = _make_property_layout(tmp_path, property_key)
    scrape_path = (
        root / "data" / "scraped" / "batches" / "batch_a"
        / "properties" / property_key / "scrape.json"
    )
    _write_scrape(scrape_path, {
        "price": 100000,
        "beds": 3,
        "sqft": 1500,
    })

    job = SimpleNamespace(
        property_key=property_key,
        results=[SimpleNamespace(image_path=str(image_path))],
        property_metadata={"price": 999999, "property_type": "single_family"},
    )

    metadata = _resolve_property_metadata(job)

    # Job-supplied price wins; scrape fills the missing sqft / beds.
    assert metadata["list_price"] == 999999
    assert metadata["property_type"] == "single_family"
    assert metadata["bedrooms"] == 3
    assert metadata["square_feet"] == 1500


def test_extract_includes_multi_kitchen_evidence_keys():
    metadata = _extract_property_metadata_from_mapping({
        "kitchen_count": 2,
        "has_adu": True,
        "is_multi_unit": False,
        "number_of_units": 2,
        "price": 500000,
    })

    assert metadata["kitchen_count"] == 2
    assert metadata["has_adu"] is True
    assert metadata["is_multi_unit"] is False
    assert metadata["number_of_units"] == 2
    assert metadata["list_price"] == 500000


def test_extract_includes_area_price_per_sqft():
    metadata = _extract_property_metadata_from_mapping({
        "price_per_sqft": 126,
        "area_price_per_sqft": 152,
    })

    assert metadata["price_per_sqft"] == 126
    assert metadata["area_price_per_sqft"] == 152


def test_extract_keeps_nested_metadata_block(tmp_path: Path):
    payload = {
        "metadata": {"price": 425000, "beds": 5},
        "property_type": "single_family",
    }

    metadata = _extract_property_metadata_from_mapping(payload)

    assert metadata["list_price"] == 425000
    assert metadata["bedrooms"] == 5
    assert metadata["property_type"] == "single_family"


def test_load_scrape_metadata_returns_empty_when_no_file(tmp_path: Path):
    property_key = "redfin_test_missing"
    root, image_path = _make_property_layout(tmp_path, property_key)
    # Intentionally do not write scrape.json.

    metadata = _load_scrape_metadata_for_job(_job_for(image_path, property_key))

    assert metadata == {}
