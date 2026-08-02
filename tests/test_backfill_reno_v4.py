"""Tests for property_metadata recovery in tools.backfill_reno_v4."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import backfill_reno_v4
from tools.backfill_reno_v4 import (
    _resolve_property_metadata_from_artifact,
    backfill,
)


def _write_artifact(
    artifact_path: Path,
    *,
    property_key: str,
    image_path: str,
    property_metadata: dict | None = None,
) -> dict:
    artifact = {
        "property": {"property_key": property_key},
        "renovation_estimate": {"version": "renovation_estimate_v3"},
        "estimate_issues_flat": [],
        "photos": {
            "img.jpg": {
                "photo": {"photo_key": "img.jpg", "image_path": image_path},
            },
        },
    }
    if property_metadata is not None:
        artifact["property_metadata"] = property_metadata
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f)
    return artifact


def _make_image(tmp_path: Path, property_key: str) -> Path:
    image_path = (
        tmp_path / "public" / "images" / "properties" / property_key / "img.jpg"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"")
    return image_path


def _write_scrape(tmp_path: Path, property_key: str, payload: dict) -> Path:
    scrape_path = (
        tmp_path
        / "data"
        / "scraped"
        / "batches"
        / "batch_a"
        / "properties"
        / property_key
        / "scrape.json"
    )
    scrape_path.parent.mkdir(parents=True, exist_ok=True)
    with scrape_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    return scrape_path


def test_resolve_uses_persisted_property_metadata_when_present():
    artifact = {
        "property": {"property_key": "k"},
        "property_metadata": {"list_price": 250000, "bedrooms": 3},
        "photos": {},
    }

    result = _resolve_property_metadata_from_artifact(artifact)

    assert result["list_price"] == 250000
    assert result["bedrooms"] == 3
    assert result["metadata_source"] == "persisted_artifact"


def test_resolve_falls_back_to_scrape_for_legacy_artifact(tmp_path: Path):
    property_key = "redfin_legacy_001"
    image_path = _make_image(tmp_path, property_key)
    _write_scrape(tmp_path, property_key, {"price": 425000, "beds": 4})

    artifact = {
        "property": {"property_key": property_key},
        "photos": {
            "img.jpg": {"photo": {"image_path": str(image_path)}},
        },
        # no property_metadata key on purpose
    }

    result = _resolve_property_metadata_from_artifact(artifact)

    assert result["list_price"] == 425000
    assert result["bedrooms"] == 4
    assert result["metadata_source"] == "scrape_json"


def test_resolve_returns_empty_when_no_sources():
    artifact = {
        "property": {"property_key": "k"},
        "photos": {},
    }

    result = _resolve_property_metadata_from_artifact(artifact)

    assert result == {}


def test_backfill_forwards_persisted_metadata_to_v4(tmp_path: Path, monkeypatch):
    property_key = "redfin_persist_001"
    image_path = _make_image(tmp_path, property_key)
    artifact_path = tmp_path / "photo_intel.json"
    _write_artifact(
        artifact_path,
        property_key=property_key,
        image_path=str(image_path),
        property_metadata={"list_price": 500000, "bedrooms": 4, "metadata_source": "scrape_json"},
    )

    captured: dict = {}

    def _fake_compute(**kwargs):
        captured.update(kwargs)
        return {"final_rehab": {"low": 1000, "high": 2000}, "packages": []}

    monkeypatch.setattr(backfill_reno_v4, "compute_renovation_estimate_v4", _fake_compute)

    result = backfill(artifact_path, catalog={"items": []}, force=False, dry_run=True)

    assert result["status"] == "updated"
    assert result["metadata_source"] == "scrape_json"
    forwarded = captured["property_metadata"]
    assert forwarded["list_price"] == 500000
    assert forwarded["bedrooms"] == 4
    assert forwarded["metadata_source"] == "scrape_json"


def test_backfill_fallback_path_recovers_scrape_json(tmp_path: Path, monkeypatch):
    property_key = "redfin_fallback_001"
    image_path = _make_image(tmp_path, property_key)
    _write_scrape(tmp_path, property_key, {"price": 310000, "baths": 1.5})
    artifact_path = tmp_path / "photo_intel.json"
    _write_artifact(
        artifact_path,
        property_key=property_key,
        image_path=str(image_path),
        property_metadata=None,
    )

    captured: dict = {}

    def _fake_compute(**kwargs):
        captured.update(kwargs)
        return {"final_rehab": {"low": 0, "high": 0}, "packages": []}

    monkeypatch.setattr(backfill_reno_v4, "compute_renovation_estimate_v4", _fake_compute)

    result = backfill(artifact_path, catalog={"items": []}, force=False, dry_run=True)

    assert result["status"] == "updated"
    assert result["metadata_source"] == "scrape_json"
    forwarded = captured["property_metadata"]
    assert forwarded["list_price"] == 310000
    assert forwarded["bath_count"] == 1.5


def test_backfill_passes_none_when_neither_source_available(tmp_path: Path, monkeypatch):
    property_key = "redfin_none_001"
    artifact_path = tmp_path / "photo_intel.json"
    # No image, no scrape, no persisted metadata.
    _write_artifact(
        artifact_path,
        property_key=property_key,
        image_path="/nowhere/img.jpg",
        property_metadata=None,
    )

    captured: dict = {}

    def _fake_compute(**kwargs):
        captured.update(kwargs)
        return {"final_rehab": {"low": 0, "high": 0}, "packages": []}

    monkeypatch.setattr(backfill_reno_v4, "compute_renovation_estimate_v4", _fake_compute)

    result = backfill(artifact_path, catalog={"items": []}, force=False, dry_run=True)

    assert result["status"] == "updated"
    assert result["metadata_source"] == "none"
    assert captured["property_metadata"] is None
