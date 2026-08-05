"""Tests for scripts/audit_exterior_estimate_coverage.py.

Synthetic artifacts under tmp_path — the corpus walk is never pointed at real
data here. Modelled on tests/test_backfill_reno_v4.py.

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_audit_exterior_estimate_coverage.py -q`
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "audit_exterior_estimate_coverage",
    ROOT / "scripts" / "audit_exterior_estimate_coverage.py",
)
audit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audit
_SPEC.loader.exec_module(audit)

from tests.test_renovation_estimate import (  # noqa: E402
    GUARDED_ESTIMATE,
    HIGH_ESTIMATE,
    _make_catalog,
    _make_issue,
    _make_item,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

def _write_catalog(tmp_path: Path, *items) -> Path:
    path = tmp_path / "catalog.json"
    catalog = _make_catalog(*items)
    catalog["version"] = "test-1.0"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def _write_artifact(
    root: Path,
    property_key: str,
    *,
    issues,
    packages=None,
    package_candidates=None,
    include_v4: bool = True,
    run_id: str = "20260101_000000_abcdef12",
) -> Path:
    run_dir = root / property_key / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "property": {"property_key": property_key},
        "run": {"run_id": run_id},
        "issues_flat": issues,
        "photos": {
            issue["photo_key"]: {"photo": {"scene_group": issue.get("scene_group", "other")}}
            for issue in issues
        },
    }
    if include_v4:
        artifact["renovation_estimate_v4"] = {
            "final_rehab": {"low": 1, "high": 2},
            "packages": packages if packages is not None else [],
            "package_candidates": (
                package_candidates if package_candidates is not None else []
            ),
        }
    path = run_dir / "photo_intel.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path):
    """Two artifacts: one fully covered, one with an uncovered exterior item."""
    root = tmp_path / "artifacts"
    catalog_path = _write_catalog(
        tmp_path,
        _make_item("priced", estimate=HIGH_ESTIMATE),
        # No estimate block and category exterior → counts as missing coverage.
        {
            "id": "uncovered_ext",
            "name": "Uncovered Exterior",
            "kind": "defect",
            "severity": 2,
            "scope": "repair",
            "trade_bucket": "landscaping_drains",
            "category": "exterior",
            "cost": {"mode": "heuristic"},
        },
    )
    _write_artifact(
        root, "prop_covered",
        issues=[_make_issue("priced", photo_key="a.jpg")],
    )
    _write_artifact(
        root, "prop_gap",
        issues=[
            _make_issue("priced", photo_key="b.jpg"),
            _make_issue("uncovered_ext", scene_group="exterior", photo_key="c.jpg"),
            _make_issue("uncovered_ext", scene_group="exterior", photo_key="d.jpg"),
        ],
    )
    return root, catalog_path


# ─── walking and ordering ────────────────────────────────────────────────────

class TestCorpusWalk:
    def test_finds_only_exact_photo_intel_files(self, tmp_path):
        root = tmp_path / "artifacts"
        _write_artifact(root, "p1", issues=[])
        run_dir = root / "p1" / "20260101_000000_abcdef12"
        (run_dir / "photo_intel_debug.json").write_text("{}", encoding="utf-8")
        (run_dir / "photo_intel.pre_2f_replay_20260101.json").write_text("{}", encoding="utf-8")

        found = audit.iter_artifact_paths(root)

        assert [p.name for p in found] == ["photo_intel.json"]

    def test_results_are_ordered_by_relative_path(self, tmp_path):
        root = tmp_path / "artifacts"
        catalog_path = _write_catalog(tmp_path, _make_item("priced", estimate=HIGH_ESTIMATE))
        for key in ("zeta", "alpha", "mid"):
            _write_artifact(root, key, issues=[_make_issue("priced", photo_key="a.jpg")])

        data = audit.build_audit(root, catalog_path)

        artifacts = [row["artifact"] for row in data["artifacts"]]
        assert artifacts == sorted(artifacts)
        assert artifacts[0].startswith("alpha/")

    def test_histograms_sort_by_count_then_id(self):
        from collections import Counter

        result = audit._histogram(Counter({"b": 2, "a": 2, "c": 5}))

        assert [entry["catalog_item_id"] for entry in result] == ["c", "a", "b"]


# ─── coverage reporting ──────────────────────────────────────────────────────

class TestCoverage:
    def test_counts_missing_effective_estimates(self, corpus):
        root, catalog_path = corpus

        data = audit.build_audit(root, catalog_path)

        by_key = {row["property_key"]: row for row in data["artifacts"]}
        assert by_key["prop_covered"]["missing_occurrences"] == 0
        assert by_key["prop_gap"]["missing_occurrences"] == 2
        assert by_key["prop_gap"]["missing_occurrences_exterior"] == 2
        assert by_key["prop_gap"]["missing_by_catalog_id"] == [
            {"catalog_item_id": "uncovered_ext", "occurrences": 2},
        ]
        assert data["summary"]["missing_occurrences"] == 2

    def test_unmatched_issues_are_tracked_separately(self, tmp_path):
        root = tmp_path / "artifacts"
        catalog_path = _write_catalog(tmp_path, _make_item("priced", estimate=HIGH_ESTIMATE))
        _write_artifact(
            root, "p1",
            issues=[
                _make_issue("priced", photo_key="a.jpg"),
                _make_issue("not_in_catalog", photo_key="b.jpg"),
            ],
        )

        row = audit.build_audit(root, catalog_path)["artifacts"][0]

        assert row["unmatched_issue_occurrences"] == 1
        assert row["missing_occurrences"] == 0

    def test_withheld_lane_is_reported(self, tmp_path):
        root = tmp_path / "artifacts"
        catalog_path = _write_catalog(
            tmp_path, _make_item("guarded", estimate=GUARDED_ESTIMATE),
        )
        _write_artifact(root, "p1", issues=[_make_issue("guarded", photo_key="a.jpg")])

        data = audit.build_audit(root, catalog_path)
        row = data["artifacts"][0]

        assert row["withheld_line_item_count"] == 1
        assert row["withheld_candidate_count"] == 1
        assert row["totals"]["withheld_total"]["high"] > 0
        assert row["totals"]["final_rehab"] == {"low": 0, "high": 0}
        assert data["summary"]["artifacts_with_withheld"] == 1


# ─── skip behaviour ──────────────────────────────────────────────────────────

class TestSkips:
    def test_artifact_without_v4_is_skipped(self, tmp_path):
        root = tmp_path / "artifacts"
        catalog_path = _write_catalog(tmp_path, _make_item("priced", estimate=HIGH_ESTIMATE))
        _write_artifact(
            root, "p1", issues=[_make_issue("priced", photo_key="a.jpg")],
            include_v4=False,
        )

        row = audit.build_audit(root, catalog_path)["artifacts"][0]

        assert row["status"] == "skipped"
        assert row["reason"] == "no renovation_estimate_v4"

    def test_artifact_without_issue_lane_is_skipped(self, tmp_path):
        root = tmp_path / "artifacts"
        catalog_path = _write_catalog(tmp_path, _make_item("priced", estimate=HIGH_ESTIMATE))
        _write_artifact(root, "p1", issues=[])

        row = audit.build_audit(root, catalog_path)["artifacts"][0]

        assert row["status"] == "skipped"
        assert row["reason"] == "no usable issues_flat lane"

    def test_active_packages_without_replayable_verdicts_are_skipped(self, tmp_path):
        """Dropping a package we cannot replay would understate the recompute,
        so the artifact is skipped rather than silently mis-reported."""
        root = tmp_path / "artifacts"
        catalog_path = _write_catalog(tmp_path, _make_item("priced", estimate=HIGH_ESTIMATE))
        _write_artifact(
            root, "p1",
            issues=[_make_issue("priced", photo_key="a.jpg")],
            packages=[{"package_id": "pkg_1", "cost_low": 1, "cost_high": 2}],
            package_candidates=[{"package_id": "pkg_1", "verification_status": "not_run"}],
        )

        row = audit.build_audit(root, catalog_path)["artifacts"][0]

        assert row["status"] == "skipped"
        assert row["reason"] == "stored packages carry no replayable verification"

    def test_no_active_packages_is_audited_not_skipped(self, tmp_path):
        """All-not_run candidates with no active package lose nothing on
        recompute, so they stay in the corpus."""
        root = tmp_path / "artifacts"
        catalog_path = _write_catalog(tmp_path, _make_item("priced", estimate=HIGH_ESTIMATE))
        _write_artifact(
            root, "p1",
            issues=[_make_issue("priced", photo_key="a.jpg")],
            packages=[],
            package_candidates=[{"package_id": "pkg_1", "verification_status": "not_run"}],
        )

        row = audit.build_audit(root, catalog_path)["artifacts"][0]

        assert row["status"] == "audited"

    def test_stored_verifications_are_reused(self, tmp_path):
        root = tmp_path / "artifacts"
        catalog_path = _write_catalog(tmp_path, _make_item("priced", estimate=HIGH_ESTIMATE))
        _write_artifact(
            root, "p1",
            issues=[_make_issue("priced", photo_key="a.jpg")],
            packages=[{"package_id": "pkg_1"}],
            package_candidates=[{
                "package_id": "pkg_1",
                "package_type": "kitchen_modernization",
                "verification_status": "confirmed",
                "confirmed_issue_ids": ["iss_priced_a.jpg"],
                "rejected_issue_ids": [],
            }],
        )

        row = audit.build_audit(root, catalog_path)["artifacts"][0]

        assert row["status"] == "audited"
        assert row["verifications_reused"] == 1

    def test_unreadable_artifact_is_skipped_not_raised(self, tmp_path):
        root = tmp_path / "artifacts"
        catalog_path = _write_catalog(tmp_path, _make_item("priced", estimate=HIGH_ESTIMATE))
        run_dir = root / "p1" / "20260101_000000_abcdef12"
        run_dir.mkdir(parents=True)
        (run_dir / "photo_intel.json").write_text("{not json", encoding="utf-8")

        row = audit.build_audit(root, catalog_path)["artifacts"][0]

        assert row["status"] == "skipped"
        assert "cannot load artifact" in row["reason"]

    def test_skip_reasons_are_summarised(self, tmp_path):
        root = tmp_path / "artifacts"
        catalog_path = _write_catalog(tmp_path, _make_item("priced", estimate=HIGH_ESTIMATE))
        _write_artifact(root, "p1", issues=[], include_v4=True)
        _write_artifact(root, "p2", issues=[], include_v4=True)

        summary = audit.build_audit(root, catalog_path)["summary"]

        assert summary["artifacts_skipped"] == 2
        assert summary["skip_reasons"] == [
            {"reason": "no usable issues_flat lane", "count": 2},
        ]


# ─── baseline deltas and the fail gate ───────────────────────────────────────

class TestDeltas:
    def test_no_baseline_reports_unavailable(self, corpus):
        root, catalog_path = corpus
        data = audit.build_audit(root, catalog_path)

        deltas = audit.compute_deltas(data, None)

        assert deltas["available"] is False
        assert deltas["headline_changes"] == []

    def test_identical_snapshots_have_no_headline_change(self, corpus):
        root, catalog_path = corpus
        data = audit.build_audit(root, catalog_path)

        deltas = audit.compute_deltas(data, json.loads(json.dumps(data)))

        assert deltas["available"] is True
        assert deltas["headline_changes"] == []
        assert all(
            entry["delta"] == {"low": 0, "high": 0}
            for entry in deltas["corpus"].values()
        )

    def test_headline_movement_is_detected(self, corpus):
        root, catalog_path = corpus
        data = audit.build_audit(root, catalog_path)
        baseline = json.loads(json.dumps(data))
        baseline["artifacts"][0]["totals"]["final_rehab"]["high"] += 500

        deltas = audit.compute_deltas(data, baseline)

        assert len(deltas["headline_changes"]) == 1
        assert deltas["headline_changes"][0]["moved"] == {
            "final_rehab": {"low": 0, "high": -500},
        }

    def test_withheld_movement_is_observed_not_gated(self, corpus):
        """The withheld lane and unreviewed risk are expected to move; they
        must never trip --fail-on-headline-delta."""
        root, catalog_path = corpus
        data = audit.build_audit(root, catalog_path)
        baseline = json.loads(json.dumps(data))
        for key in ("withheld_total", "unreviewed_risk_total", "latent_risk_exposure"):
            baseline["artifacts"][0]["totals"][key]["high"] += 999
            baseline["summary"]["corpus_totals"][key]["high"] += 999

        deltas = audit.compute_deltas(data, baseline)

        # Corpus movement is reported...
        assert deltas["corpus"]["withheld_total"]["delta"]["high"] == -999
        assert deltas["corpus"]["latent_risk_exposure"]["delta"]["high"] == -999
        # ...but never counts as a headline change.
        assert deltas["headline_changes"] == []

    def test_gated_key_set_excludes_the_risk_lanes(self):
        assert "withheld_total" not in audit._HEADLINE_KEYS
        assert "unreviewed_risk_total" not in audit._HEADLINE_KEYS
        assert "latent_risk_exposure" not in audit._HEADLINE_KEYS
        assert "final_rehab" in audit._HEADLINE_KEYS
        assert "evidence_headline" in audit._HEADLINE_KEYS


class TestMain:
    def test_missing_root_exits_2(self, tmp_path):
        code = audit.main(["--artifacts-root", str(tmp_path / "nope")])
        assert code == 2

    def test_fail_flag_without_baseline_exits_2(self, corpus):
        root, catalog_path = corpus
        code = audit.main([
            "--artifacts-root", str(root),
            "--catalog", str(catalog_path),
            "--fail-on-headline-delta",
        ])
        assert code == 2

    def test_clean_run_exits_0_and_writes_outputs(self, corpus, tmp_path):
        root, catalog_path = corpus
        json_out = tmp_path / "out" / "snap.json"
        report = tmp_path / "out" / "report.md"

        code = audit.main([
            "--artifacts-root", str(root),
            "--catalog", str(catalog_path),
            "--json", str(json_out),
            "--report", str(report),
        ])

        assert code == 0
        assert json.loads(json_out.read_text(encoding="utf-8"))["artifacts"]
        assert "Exterior estimate coverage audit" in report.read_text(encoding="utf-8")

    def test_fail_on_headline_delta_exits_1(self, corpus, tmp_path):
        root, catalog_path = corpus
        baseline_path = tmp_path / "baseline.json"
        data = audit.build_audit(root, catalog_path)
        data["artifacts"][0]["totals"]["final_rehab"]["high"] += 500
        baseline_path.write_text(json.dumps(data), encoding="utf-8")

        code = audit.main([
            "--artifacts-root", str(root),
            "--catalog", str(catalog_path),
            "--baseline", str(baseline_path),
            "--fail-on-headline-delta",
        ])

        assert code == 1
