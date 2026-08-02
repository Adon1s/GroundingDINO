"""Product-quarantine tests.

Trades flagged `product_quarantined` in the catalog must never influence
product surfaces: estimate candidates, packages, scoring, summaries, or
priorities. The anchor is the counterfactual test: adding any number of
quarantined observations to an otherwise identical input leaves every
product output unchanged.
"""

import copy
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.artifact_writers import _build_ui_priorities_v1, write_photo_intel
from tools.costing import compute_scoring
from tools.pipeline_common import PRODUCT_POLICY_VERSION
from tools.property_summary_pass import build_property_summary_v1, load_catalog_index
from tools.renovation_estimate import (
    extract_estimate_candidates,
    filter_product_issues,
    product_quarantined_trade_buckets,
)
from tools.renovation_estimate_v4 import (
    _extract_package_only_candidates,
    compute_renovation_estimate_v4,
)
from tools.reproject_product_views import (
    collect_stored_verifications,
    reproject_artifact,
)

ROOT = Path(__file__).resolve().parents[1]

REAL_CATALOG = json.loads(
    (ROOT / "tools" / "issue_catalog.json").read_text(encoding="utf-8")
)

REAL_ELECTRICAL_IDS = sorted(
    item["id"] for item in REAL_CATALOG["items"]
    if item.get("trade_bucket") == "electrical"
)


def _make_issue(catalog_item_id, *, scene_group="bathroom", photo_key="img_001.jpg",
                issue_id=None, **extra):
    issue = {
        "issue_id": issue_id or f"iss_{catalog_item_id}_{photo_key}",
        "catalog_item_id": catalog_item_id,
        "catalog_item_kind": "defect",
        "scene_group": scene_group,
        "photo_key": photo_key,
        "description": f"test {catalog_item_id}",
        "label": "defect_or_damage",
    }
    issue.update(extra)
    return issue


def _make_catalog(*items, quarantine_electrical=True):
    electrical_bucket = {"id": "electrical", "name": "Electrical"}
    if quarantine_electrical:
        electrical_bucket["product_quarantined"] = True
    return {
        "version": "test",
        "trade_buckets": [
            {"id": "flooring", "name": "Flooring"},
            electrical_bucket,
        ],
        "items": list(items),
    }


def _electrical_item(item_id="test_gfci", *, tier="medium", package_affinity=None):
    item = {
        "id": item_id,
        "name": item_id.replace("_", " ").title(),
        "kind": "defect",
        "severity": 3,
        "scope": "replace",
        "trade_bucket": "electrical",
        "cost": {"mode": "allowance", "base_low": 100, "base_high": 1000,
                 "per_occurrence_low": 100, "per_occurrence_high": 400,
                 "cap_low": 1000, "cap_high": 2000},
    }
    if tier is not None:
        item["estimate"] = {
            "estimate_tier": tier,
            "strategy": "replace_only",
            "group": "other",
            "stack_behavior": "sum",
        }
    if package_affinity is not None:
        item["package_affinity"] = package_affinity
    return item


class TestQuarantineHelpers:

    def test_flagged_bucket_detected(self):
        catalog = _make_catalog()
        assert product_quarantined_trade_buckets(catalog) == frozenset({"electrical"})

    def test_unflagged_catalog_empty(self):
        catalog = _make_catalog(quarantine_electrical=False)
        assert product_quarantined_trade_buckets(catalog) == frozenset()

    def test_real_catalog_quarantines_electrical(self):
        assert "electrical" in product_quarantined_trade_buckets(REAL_CATALOG)

    def test_filter_drops_quarantined_keeps_clean_and_unmapped(self):
        catalog = _make_catalog(_electrical_item())
        issues = [
            _make_issue("test_gfci"),
            _make_issue("some_clean_item"),        # not in catalog: kept
            {"issue_id": "no_cat", "description": "unmapped"},  # no id: kept
        ]
        product = filter_product_issues(issues, catalog)
        assert [i.get("issue_id") for i in product] == [
            "iss_some_clean_item_img_001.jpg", "no_cat",
        ]

    def test_filter_is_noop_without_flag(self):
        catalog = _make_catalog(_electrical_item(), quarantine_electrical=False)
        issues = [_make_issue("test_gfci")]
        assert filter_product_issues(issues, catalog) == issues


class TestExtractorGates:

    def test_line_item_gate_flag_is_the_switch(self):
        issue = _make_issue("test_gfci")
        flagged = _make_catalog(_electrical_item())
        unflagged = _make_catalog(_electrical_item(), quarantine_electrical=False)

        assert extract_estimate_candidates([issue], flagged) == []
        candidates = extract_estimate_candidates([issue], unflagged)
        assert [c.catalog_item_id for c in candidates] == ["test_gfci"]

    def test_package_only_gate_flag_is_the_switch(self):
        affinity = {"bathroom": {
            "package_type": "bathroom_repair",
            "package_role": "package_driver",
        }}
        # tier=None → affects_estimate False → package-evidence-only item
        item = _electrical_item(tier=None, package_affinity=affinity)
        issue = _make_issue("test_gfci", scene_group="bathroom")

        flagged = _make_catalog(item)
        unflagged = _make_catalog(item, quarantine_electrical=False)

        assert _extract_package_only_candidates([issue], flagged) == []
        candidates = _extract_package_only_candidates([issue], unflagged)
        assert [c.catalog_item_id for c in candidates] == ["test_gfci"]


class TestRealCatalogEndToEnd:

    def test_gfci_mints_nothing(self):
        """A GFCI detection alone: no line item, no package, no candidate."""
        issues = [_make_issue("bathroom_gfci_missing_or_damaged")]
        v4 = compute_renovation_estimate_v4(issues, REAL_CATALOG, photos={})

        line_items = [
            li for g in v4.get("groups") or []
            for li in g.get("line_items") or []
        ]
        assert line_items == []
        assert v4.get("packages") == []
        assert v4.get("package_candidates") == []

    def test_counterfactual_anchor(self):
        """Adding every electrical observation to an input changes nothing."""
        base_issues = [
            _make_issue("water_stain_ceiling", scene_group="bathroom"),
            _make_issue("worn_or_stained_carpet", scene_group="bedroom",
                        photo_key="img_002.jpg"),
        ]
        with_electrical = base_issues + [
            _make_issue(item_id, photo_key=f"img_{9 + i:03d}.jpg")
            for i, item_id in enumerate(REAL_ELECTRICAL_IDS)
        ]

        # Product lanes are identical.
        assert filter_product_issues(with_electrical, REAL_CATALOG) == \
            filter_product_issues(base_issues, REAL_CATALOG)

        # v4 fed the RAW lanes (bypassing the lane filter) is identical too —
        # the extractor gates alone must hold the invariant.
        v4_base = compute_renovation_estimate_v4(
            copy.deepcopy(base_issues), REAL_CATALOG, photos={},
        )
        v4_elec = compute_renovation_estimate_v4(
            copy.deepcopy(with_electrical), REAL_CATALOG, photos={},
        )
        assert v4_base == v4_elec

        # Scoring and summary consume product lanes.
        lane_base = filter_product_issues(base_issues, REAL_CATALOG)
        lane_elec = filter_product_issues(with_electrical, REAL_CATALOG)
        assert compute_scoring(issues_flat=lane_base, issue_catalog=REAL_CATALOG,
                               n_photos=3) == \
            compute_scoring(issues_flat=lane_elec, issue_catalog=REAL_CATALOG,
                            n_photos=3)

        catalog_index = load_catalog_index(REAL_CATALOG)
        s_base = build_property_summary_v1(
            property_key="p", run_id="r", issues_flat=lane_base,
            catalog_index=catalog_index,
        )
        s_elec = build_property_summary_v1(
            property_key="p", run_id="r", issues_flat=lane_elec,
            catalog_index=catalog_index,
        )
        s_base.pop("generated_at", None)
        s_elec.pop("generated_at", None)
        assert s_base == s_elec

        assert _build_ui_priorities_v1(
            issues_flat=lane_base, issue_catalog=REAL_CATALOG,
            renovation_estimate_v4=v4_base,
        ) == _build_ui_priorities_v1(
            issues_flat=lane_elec, issue_catalog=REAL_CATALOG,
            renovation_estimate_v4=v4_elec,
        )


# ─── Reprojection ────────────────────────────────────────────────────────────

def _legacy_artifact():
    """A pre-quarantine artifact: GFCI-only confirmed package + a clean one."""
    gfci_issue = _make_issue("bathroom_gfci_missing_or_damaged")
    clean_issue = _make_issue("water_stain_ceiling", photo_key="img_002.jpg")
    return {
        "schema_version": "photo_intel_v3",
        "run": {"run_id": "run_1", "job_id": "run_1"},
        "property": {"property_key": "prop_1"},
        "photos": {},
        "issues_flat": [gfci_issue, clean_issue],
        "estimate_issues_flat": [gfci_issue, clean_issue],
        "scoring": {"rehab_score": 55, "systems_score": 40},
        "renovation_estimate_v4": {
            "version": "renovation_estimate_v4",
            "final_rehab": {"low": 5000, "high": 20000},
            "packages": [],
            "package_candidates": [
                {
                    "package_id": "bathroom_repair__bathroom_primary",
                    "package_type": "bathroom_repair",
                    "verification_status": "confirmed",
                    "confirmed_issue_ids": [gfci_issue["issue_id"]],
                    "rejected_issue_ids": [],
                    "evidence_summary": "GFCI missing near sink.",
                    "cost_low": 1120,
                    "cost_high": 7000,
                    "evidence_items": [
                        {"catalog_item_id": "bathroom_gfci_missing_or_damaged"},
                    ],
                },
                {
                    "package_id": "bedroom_modernization__bedroom_1",
                    "package_type": "bedroom_modernization",
                    "verification_status": "confirmed",
                    "confirmed_issue_ids": [],
                    "rejected_issue_ids": [],
                    "evidence_summary": "Dated bedroom finishes.",
                    "cost_low": 5000,
                    "cost_high": 15000,
                    "evidence_items": [
                        {"catalog_item_id": "worn_or_stained_carpet"},
                    ],
                },
                {
                    "package_id": "never_ran__unit",
                    "package_type": "bathroom_repair",
                    "verification_status": "not_run",
                    "evidence_items": [
                        {"catalog_item_id": "water_stain_ceiling"},
                    ],
                },
            ],
        },
    }


class TestCollectStoredVerifications:

    def test_taint_and_reuse_split(self):
        quarantined_ids = frozenset({"bathroom_gfci_missing_or_damaged"})
        verifications, tainted = collect_stored_verifications(
            _legacy_artifact(), quarantined_ids,
        )
        assert set(verifications) == {"bedroom_modernization__bedroom_1"}
        assert [t["package_id"] for t in tainted] == [
            "bathroom_repair__bathroom_primary",
        ]
        assert tainted[0]["quarantined_catalog_item_ids"] == [
            "bathroom_gfci_missing_or_damaged",
        ]
        assert tainted[0]["cost_low"] == 1120
        # not_run entries never carry a verification.
        assert "never_ran__unit" not in verifications


class TestReprojectArtifact:

    def _write(self, tmp_path, artifact):
        p = tmp_path / "photo_intel.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        return p

    def test_gfci_only_package_dropped_and_stamped(self, tmp_path):
        p = self._write(tmp_path, _legacy_artifact())
        result = reproject_artifact(p, REAL_CATALOG)

        assert result["status"] == "reprojected"
        assert [t["package_id"] for t in result["tainted_packages"]] == [
            "bathroom_repair__bathroom_primary",
        ]

        reprojected = json.loads(p.read_text(encoding="utf-8"))
        assert reprojected["product_projection_status"] == "reprojected"
        assert reprojected["product_policy_version"] == PRODUCT_POLICY_VERSION

        # Product lanes exclude the GFCI issue; raw lanes keep it.
        assert len(reprojected["issues_flat"]) == 2
        product_ids = {
            i["catalog_item_id"] for i in reprojected["product_issues_flat"]
        }
        assert "bathroom_gfci_missing_or_damaged" not in product_ids

        # No electrical evidence anywhere in the recomputed estimate.
        v4_text = json.dumps(reprojected["renovation_estimate_v4"])
        assert "bathroom_gfci_missing_or_damaged" not in v4_text
        assert reprojected["renovation_estimate_v4"]["packages"] == []

        # Scoring recomputed from the product lane: no systems contribution.
        assert reprojected["scoring"]["systems_score"] == 0

        # Backup of the original exists.
        backups = list(tmp_path.glob("photo_intel.pre_reprojection_*.json"))
        assert len(backups) == 1

    def test_missing_lanes_needs_reanalysis(self, tmp_path):
        artifact = _legacy_artifact()
        del artifact["issues_flat"]
        del artifact["estimate_issues_flat"]
        p = self._write(tmp_path, artifact)

        result = reproject_artifact(p, REAL_CATALOG)
        assert result["status"] == "needs_reanalysis"

        reprojected = json.loads(p.read_text(encoding="utf-8"))
        assert reprojected["product_projection_status"] == "needs_reanalysis"
        # Never stale or zero totals: product fields are nulled.
        assert reprojected["scoring"] is None
        assert reprojected["renovation_estimate_v4"] is None

    def test_idempotent_skip_and_force(self, tmp_path):
        p = self._write(tmp_path, _legacy_artifact())
        assert reproject_artifact(p, REAL_CATALOG)["status"] == "reprojected"
        assert reproject_artifact(p, REAL_CATALOG)["status"] == "skip_current"
        assert reproject_artifact(p, REAL_CATALOG, force=True)["status"] == "reprojected"

    def test_dry_run_writes_nothing(self, tmp_path):
        artifact = _legacy_artifact()
        p = self._write(tmp_path, artifact)
        before = p.read_text(encoding="utf-8")

        result = reproject_artifact(p, REAL_CATALOG, dry_run=True)
        assert result["status"] == "dry_run"
        assert p.read_text(encoding="utf-8") == before
        assert list(tmp_path.glob("*.pre_reprojection_*")) == []


# ─── Live writer: product lanes and stamps ───────────────────────────────────

class TestWritePhotoIntelProductLanes:

    def test_lanes_stamps_and_consumers(self):
        tmp_path = Path.cwd() / ".pytest_cache" / f"quarantine_{uuid.uuid4().hex}"
        tmp_path.mkdir(parents=True)
        image_path = tmp_path / "bathroom_1.jpg"
        try:
            image_path.write_bytes(b"fake image")
            catalog = _make_catalog(
                _electrical_item("test_gfci"),
                {
                    "id": "clean_floor_issue",
                    "name": "Clean Floor Issue",
                    "kind": "defect",
                    "severity": 3,
                    "scope": "repair",
                    "trade_bucket": "flooring",
                    "cost": {"mode": "heuristic"},
                    "estimate": {
                        "estimate_tier": "medium",
                        "strategy": "repair_only",
                        "group": "other",
                        "stack_behavior": "sum",
                    },
                },
            )
            result = SimpleNamespace(
                image_path=str(image_path),
                scene="bathroom",
                scene_classifier={
                    "scene": "bathroom",
                    "canonical_issues": [
                        {"description": "GFCI outlet missing.",
                         "catalogItemId": "test_gfci", "label": "defect"},
                        {"description": "Floor is damaged.",
                         "catalogItemId": "clean_floor_issue", "label": "defect"},
                    ],
                    "verified_issues": [
                        {"description": "GFCI outlet missing.",
                         "catalogItemId": "test_gfci", "label": "defect"},
                        {"description": "Floor is damaged.",
                         "catalogItemId": "clean_floor_issue", "label": "defect"},
                    ],
                    "matched_issues": [],
                    "passes": {
                        "1a": {"scene": "bathroom", "confidence": 0.9, "reasoning": ""},
                        "1c": {"overall_impression": "", "image_summary": "",
                               "notable_features": []},
                        "2e": {},
                    },
                },
                scene_data=None,
                processing_time=0.1,
                error=None,
            )
            job = SimpleNamespace(
                property_key="prop",
                job_id="job",
                timestamp="2026-07-12T00:00:00Z",
                artifacts_dir=str(tmp_path),
                results=[result],
            )
            output_path = write_photo_intel(
                cfg=SimpleNamespace(LM_STUDIO_MODEL="test-model"),
                job=job,
                detection_backend="test",
                analysis_profile="test",
                use_pass_architecture=True,
                pass_toggles={"2f": False},
                model_overrides={},
                gpt_config=None,
                issue_catalog=catalog,
                output_path=tmp_path / "photo_intel.json",
                vlm_client=None,
            )
            artifact = json.loads(output_path.read_text(encoding="utf-8"))

            assert artifact["product_policy_version"] == PRODUCT_POLICY_VERSION
            assert artifact["product_projection_status"] == "native"
            assert artifact["catalog_version"] == "test"

            raw_ids = {i["catalog_item_id"] for i in artifact["issues_flat"]}
            product_ids = {
                i["catalog_item_id"] for i in artifact["product_issues_flat"]
            }
            assert "test_gfci" in raw_ids            # audit record intact
            assert "test_gfci" not in product_ids    # product lane filtered
            assert "clean_floor_issue" in product_ids

            # Consumers are quarantine-free.
            assert artifact["scoring"]["systems_score"] == 0
            summary_buckets = {
                b["bucket_id"] for b in artifact["summary_v1"]["buckets"]
            }
            assert "electrical" not in summary_buckets
            v4_text = json.dumps(artifact["renovation_estimate_v4"])
            assert "test_gfci" not in v4_text
            assert "test_gfci" not in json.dumps(artifact["ui_priorities_v1"])
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)
