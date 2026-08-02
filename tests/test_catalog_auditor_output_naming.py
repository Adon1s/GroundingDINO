import argparse
import os
import re
import shutil
from pathlib import Path

import pytest

from tools.catalog_auditor import (
    _derive_single_property_id,
    _latest_checkpoint_output_path,
    _next_audit_output_path,
    _resolve_output_path,
    _sanitize_filename_token,
)


def _images(*property_ids):
    return [
        {"property_id": property_id, "photo_key": f"img_{idx:03d}.jpg"}
        for idx, property_id in enumerate(property_ids)
    ]


@pytest.fixture
def output_dir(request):
    root = Path("codex_test_output_naming")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
    path = root / safe_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        if root.exists():
            try:
                root.rmdir()
            except OSError:
                pass


def test_single_property_gets_v001_timestamped_report(output_dir):
    args = argparse.Namespace(output=None, output_dir=str(output_dir), resume_latest=False)

    output_path, version = _resolve_output_path(args, _images("redfin_123"))

    assert version == 1
    assert output_path.parent == output_dir
    assert re.fullmatch(
        r"catalog_audit_redfin_123_v001_\d{8}_\d{6}\.json",
        output_path.name,
    )


def test_existing_reports_increment_next_version(output_dir):
    (output_dir / "catalog_audit_redfin_123_v001_20260101_010101.json").touch()
    (output_dir / "catalog_audit_redfin_123_v002_20260101_020202.json").touch()

    output_path, version = _next_audit_output_path(output_dir, "redfin_123")

    assert version == 3
    assert output_path.name.startswith("catalog_audit_redfin_123_v003_")


def test_existing_checkpoints_count_when_incrementing_version(output_dir):
    (output_dir / "catalog_audit_redfin_123_v004_20260101_040404.checkpoint.json").touch()

    output_path, version = _next_audit_output_path(output_dir, "redfin_123")

    assert version == 5
    assert output_path.name.startswith("catalog_audit_redfin_123_v005_")


def test_property_id_is_sanitized_for_filenames():
    assert _sanitize_filename_token("../redfin 123/abc") == "redfin_123_abc"


def test_multi_property_discovery_is_rejected():
    with pytest.raises(ValueError, match="exactly one property"):
        _derive_single_property_id(_images("redfin_123", "redfin_456"))


def test_manual_json_output_path_is_rejected(output_dir):
    args = argparse.Namespace(
        output=str(output_dir / "catalog_audit_wrong_property.json"),
        output_dir=".",
        resume_latest=False,
    )

    with pytest.raises(ValueError, match="not a report filename"):
        _resolve_output_path(args, _images("redfin_123"))


def test_resume_latest_selects_newest_checkpoint(output_dir):
    older = output_dir / "catalog_audit_redfin_123_v001_20260101_010101.checkpoint.json"
    newer = output_dir / "catalog_audit_redfin_123_v002_20260101_020202.checkpoint.json"
    older.touch()
    newer.touch()
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    assert _latest_checkpoint_output_path(output_dir, "redfin_123") == (
        output_dir / "catalog_audit_redfin_123_v002_20260101_020202.json"
    )

    args = argparse.Namespace(output=None, output_dir=str(output_dir), resume_latest=True)
    output_path, version = _resolve_output_path(args, _images("redfin_123"))

    assert version == 2
    assert output_path == output_dir / "catalog_audit_redfin_123_v002_20260101_020202.json"
