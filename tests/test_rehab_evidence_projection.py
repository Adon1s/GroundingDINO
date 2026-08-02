"""Integration regression tests for the rehab evidence allocation projection.

The consumer-safe allocation contract (photo_supported + needs_inspection ==
headline at each endpoint; risk_exposure carried but excluded) is what lets the
desktop overview show split rehab dollars without double-counting. The
unit-level matrix — malformed-range fail-closed, coercion, risk-only shapes,
deterministic projection ids — lives in ``tools/test_rehab_evidence_projection.py``
and is NOT duplicated here.

This module covers the INTEGRATION surface those unit tests cannot:
  * the projection built through ``compute_renovation_estimate_v4`` reconciles
    against the estimate's OWN scaled totals (never recomputed independently);
  * overlapping same-unit packages do not inflate the projection, because it
    reads tier totals, never a sum of package costs;
  * legacy artifacts gain the projection on reprojection with preserved run
    identity, quarantined evidence cannot inflate it, and unservable artifacts
    drop the evidence stamps;
  * the live artifact writer stamps a native projection;
  * the 2100 Aftonbrae Dr (redfin_125800935) golden raw vectors — the cross-repo
    contract shared verbatim with the script test and the frontend tests.

Display rounding (compact_currency_v1) is owned by the frontend; the backend
pins raw vectors only.

Run: ``.venv\\Scripts\\python.exe -m pytest tests/test_rehab_evidence_projection.py -q``
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from tools.pipeline_common import PRODUCT_POLICY_VERSION
from tools.rehab_evidence_projection import (
    EVIDENCE_PROJECTION_POLICY_VERSION,
    EVIDENCE_PROJECTION_VERSION,
    SERVABLE_EVIDENCE_PROJECTION_STATUSES,
    build_rehab_evidence_projection,
    stamp_evidence_projection_provenance,
)
from tools.renovation_estimate_v4 import compute_renovation_estimate_v4
from tools.reproject_product_views import reproject_artifact

from tests.test_renovation_estimate import (
    HIGH_ESTIMATE,
    MEDIUM_ESTIMATE,
    _make_catalog,
    _make_issue,
    _make_item,
)
from tests.test_renovation_estimate_v4 import _build_simple_v3, _make_photos
from tests.test_product_quarantine import REAL_CATALOG


# ─── Shared reconciliation oracle ────────────────────────────────────────────

def _assert_projection_matches_estimate(v4):
    """The single source of truth for the contract: the projection equals the
    estimate's own (scaled) totals — never a value recomputed here."""
    proj = v4["rehab_evidence_projection_v1"]
    assert isinstance(proj, dict)
    assert proj["version"] == EVIDENCE_PROJECTION_VERSION
    assert proj["policy_version"] == EVIDENCE_PROJECTION_POLICY_VERSION

    photo = v4["final_rehab_resale_ready"]
    totals = v4["totals"]
    inspection = totals.get("inspection_allowance_total") or {"low": 0, "high": 0}
    risk = totals.get("risk_exposure_total") or {"low": 0, "high": 0}

    assert proj["photo_supported"] == {"low": photo["low"], "high": photo["high"]}
    assert proj["needs_inspection"] == {
        "low": inspection["low"],
        "high": inspection["high"],
    }
    for key in ("low", "high"):
        assert proj["headline"][key] == photo[key] + inspection[key]
    assert proj["risk_exposure"] == {"low": risk["low"], "high": risk["high"]}
    assert proj["risk_exposure_included"] is False
    assert proj["reconciliation"]["raw_exact"] is True


# ─── Projection through compute_renovation_estimate_v4 ────────────────────────

class TestProjectionThroughComputeV4:

    def _three_lane_inputs(self):
        """simple v3 (resale-ready work) plus an inspect-posture item that
        contributes an inspection allowance AND a manual risk band."""
        catalog, issues, photos = _build_simple_v3()
        manual_cost = {
            "mode": "allowance",
            "base_low": 2000,
            "base_high": 40000,
            "per_occurrence_low": 3900,
            "per_occurrence_high": 7500,
            "cap_low": 40000,
            "cap_high": 100000,
            "cost_source": "manual",
        }
        catalog["items"].append(
            _make_item(
                "foundation",
                estimate={
                    "estimate_tier": "high",
                    "strategy": "inspect_only",
                    "group": "structure",
                    "stack_behavior": "max_only",
                    "unit_policy": "per_property",
                },
                trade_bucket="foundation_structure",
                scope="repair",
                cost=manual_cost,
            )
        )
        issues.append(
            _make_issue(
                "foundation",
                scene_group="exterior",
                photo_key="ext.jpg",
                issue_id="iss_foundation",
            )
        )
        photos.update(_make_photos(("exterior_front", "ext.jpg")))
        return catalog, issues, photos

    def test_projection_attached_and_reconciles_after_scaling(self):
        catalog, issues, photos = self._three_lane_inputs()
        v4 = compute_renovation_estimate_v4(
            issues,
            catalog,
            photos,
            property_metadata={"price_per_sqft": 460, "sqft": 2400},
        )
        _assert_projection_matches_estimate(v4)

        proj = v4["rehab_evidence_projection_v1"]
        # All three lanes exercised: resale-ready work, an inspection
        # allowance, and an excluded risk band.
        assert proj["photo_supported"]["high"] > 0
        assert proj["needs_inspection"]["high"] > 0
        assert proj["risk_exposure"]["high"] > 0
        # Stamping is a write-time concern; the raw builder must not fabricate it.
        assert "provenance" not in proj
        assert "projection_id" not in proj

    def test_overlapping_packages_do_not_double_count_projection(self):
        """A modernization + repair in the same bathroom overlap: the
        modernization subsumes the repair. Naively summing the two confirmed
        package costs overstates the estimate; the projection must reflect the
        tier total instead, proving it never sums packages."""
        driver = _make_item(
            "outdated_bathroom_finishes",
            estimate={**HIGH_ESTIMATE, "group": "bathroom"},
            kind="upgrade",
            severity=3,
            trade_bucket="bathroom_fixtures_tile",
            scope="replace",
        )
        driver.update({
            "display_class": "marketability",
            "package_affinity": {
                "bathroom": {
                    "package_type": "bathroom_modernization",
                    "package_role": "package_driver",
                },
            },
            "scene_groups": ["bathroom"],
        })
        support = _make_item(
            "vintage_tile_pattern_style",
            kind="upgrade",
            severity=2,
            trade_bucket="bathroom_fixtures_tile",
            scope="replace",
        )
        support.update({
            "display_class": "marketability",
            "package_affinity": {
                "bathroom": {
                    "package_type": "bathroom_modernization",
                    "package_role": "package_support",
                },
            },
            "scene_groups": ["bathroom"],
        })
        repair_driver = _make_item(
            "water_damaged_flooring",
            estimate={**MEDIUM_ESTIMATE, "group": "bathroom"},
            kind="defect",
            severity=3,
            trade_bucket="bathroom_fixtures_tile",
            scope="repair",
            cost={"mode": "allowance", "base_low": 200, "base_high": 800,
                  "per_occurrence_low": 200, "per_occurrence_high": 800,
                  "cap_low": 800, "cap_high": 2000},
        )
        repair_driver.update({
            "display_class": "required",
            "package_affinity": {
                "bathroom": {
                    "package_type": "bathroom_repair",
                    "package_role": "package_driver",
                },
            },
            "scene_groups": ["bathroom"],
        })
        catalog = _make_catalog(driver, support, repair_driver)
        issues = [
            _make_issue("outdated_bathroom_finishes", scene_group="bathroom",
                        photo_key="b1.jpg", issue_id="iss_mod",
                        catalog_item_kind="upgrade"),
            _make_issue("vintage_tile_pattern_style", scene_group="bathroom",
                        photo_key="b2.jpg", issue_id="iss_tile",
                        catalog_item_kind="upgrade"),
            _make_issue("water_damaged_flooring", scene_group="bathroom",
                        photo_key="b1.jpg", issue_id="iss_repair"),
        ]
        photos = _make_photos(("bathroom", "b1.jpg"), ("bathroom", "b2.jpg"))
        v4 = compute_renovation_estimate_v4(
            issues,
            catalog,
            photos,
            package_verifications={
                "bathroom_modernization__bathroom_primary": {
                    "verification_status": "confirmed",
                    "confirmed_issue_ids": ["iss_mod", "iss_tile"],
                    "evidence_summary": "Dated bathroom finishes and tile.",
                    "visible_room_count": "one_room",
                },
                "bathroom_repair__bathroom_primary": {
                    "verification_status": "confirmed",
                    "confirmed_issue_ids": ["iss_repair"],
                    "evidence_summary": "Water-damaged bathroom flooring.",
                    "visible_room_count": "one_room",
                },
            },
        )

        # Precondition: both packages were actually confirmed (guards against a
        # vacuous test where a mislabelled verification key drops a package).
        confirmed = [
            c for c in v4["package_candidates"]
            if c["verification_status"] == "confirmed"
        ]
        confirmed_types = {c["package_type"] for c in confirmed}
        assert confirmed_types == {"bathroom_modernization", "bathroom_repair"}

        # Overlap resolved by subsumption: only the modernization survives.
        active_ids = [p["package_id"] for p in v4["packages"]]
        assert active_ids == ["bathroom_modernization__bathroom_primary"]
        rules = {s["rule"] for s in v4["package_subsumption_audit"]["subsumptions"]}
        assert "modernization_subsumes_repair" in rules

        # The core claim: summing the overlapping package costs overstates the
        # estimate, but the projection equals the tier total, not that sum.
        naive_sum_high = sum(int(c["cost_high"] or 0) for c in confirmed)
        resale_high = v4["final_rehab_resale_ready"]["high"]
        assert naive_sum_high > resale_high
        _assert_projection_matches_estimate(v4)
        assert v4["rehab_evidence_projection_v1"]["photo_supported"]["high"] != naive_sum_high


# ─── Reprojection of stored artifacts ────────────────────────────────────────

def _stored_artifact(*, quarantined=True):
    """A pre-projection artifact (quarantine-safe or not) with a real run
    completion timestamp, modelled on tests/test_product_quarantine.py."""
    clean_issue = _make_issue("water_stain_ceiling", photo_key="img_002.jpg")
    issues = [clean_issue]
    candidates = [
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
    ]
    if quarantined:
        gfci_issue = _make_issue("bathroom_gfci_missing_or_damaged")
        issues.insert(0, gfci_issue)
        candidates.insert(0, {
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
        })
    return {
        "schema_version": "photo_intel_v3",
        "run": {"run_id": "run_1", "job_id": "run_1",
                "created_at": "2026-06-01T00:00:00Z"},
        "property": {"property_key": "prop_1"},
        "photos": {},
        "issues_flat": list(issues),
        "estimate_issues_flat": list(issues),
        "scoring": {"rehab_score": 55, "systems_score": 40},
        "renovation_estimate_v4": {
            "version": "renovation_estimate_v4",
            "final_rehab": {"low": 5000, "high": 20000},
            "packages": [],
            "package_candidates": candidates,
        },
    }


def _write_run_artifact(tmp_path, artifact, run_dir="20260601_000000_run1"):
    run_path = tmp_path / run_dir
    run_path.mkdir()
    p = run_path / "photo_intel.json"
    p.write_text(json.dumps(artifact), encoding="utf-8")
    return p


class TestReprojection:

    def test_reprojection_adds_projection_with_preserved_run_identity(self, tmp_path):
        p = _write_run_artifact(tmp_path, _stored_artifact(quarantined=False))
        result = reproject_artifact(p, REAL_CATALOG)
        assert result["status"] == "reprojected"

        artifact = json.loads(p.read_text(encoding="utf-8"))
        v4 = artifact["renovation_estimate_v4"]
        proj = v4["rehab_evidence_projection_v1"]
        prov = proj["provenance"]

        assert prov["projection_status"] == "reprojected"
        assert prov["source_run_id"] == "run_1"
        # Reprojection changes the projection, not when the run finished.
        assert prov["completed_at"] == "2026-06-01T00:00:00Z"
        assert prov["source_artifact"] == "prop_1/20260601_000000_run1/photo_intel.json"
        assert "\\" not in prov["source_artifact"] and ":" not in prov["source_artifact"]
        assert proj["projection_id"].startswith("rep1_")
        _assert_projection_matches_estimate(v4)

        assert artifact["evidence_projection_policy_version"] == EVIDENCE_PROJECTION_POLICY_VERSION
        assert artifact["evidence_projection_status"] == "reprojected"
        assert artifact["evidence_projection_status"] in SERVABLE_EVIDENCE_PROJECTION_STATUSES

        # Now current on both policies: a second pass is a no-op.
        assert reproject_artifact(p, REAL_CATALOG)["status"] == "skip_current"

    def test_quarantined_package_evidence_cannot_inflate_projection(self, tmp_path):
        # Sibling run dirs with identical names so source_artifact/projection_id
        # differ only by evidence content, not path.
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        p_tainted = _write_run_artifact(tmp_path / "a", _stored_artifact(quarantined=True))
        p_clean = _write_run_artifact(tmp_path / "b", _stored_artifact(quarantined=False))

        res_tainted = reproject_artifact(p_tainted, REAL_CATALOG)
        res_clean = reproject_artifact(p_clean, REAL_CATALOG)
        assert res_tainted["status"] == "reprojected"
        # Quarantine engaged: the electrical-evidence package was withheld.
        assert [t["package_id"] for t in res_tainted["tainted_packages"]] == [
            "bathroom_repair__bathroom_primary",
        ]

        proj_tainted = json.loads(p_tainted.read_text(encoding="utf-8"))[
            "renovation_estimate_v4"]["rehab_evidence_projection_v1"]
        proj_clean = json.loads(p_clean.read_text(encoding="utf-8"))[
            "renovation_estimate_v4"]["rehab_evidence_projection_v1"]
        for key in ("photo_supported", "needs_inspection", "headline", "risk_exposure"):
            assert proj_tainted[key] == proj_clean[key]
        assert proj_tainted["projection_id"] == proj_clean["projection_id"]

    def test_needs_reanalysis_pops_evidence_stamps(self, tmp_path):
        artifact = _stored_artifact(quarantined=False)
        artifact["evidence_projection_policy_version"] = EVIDENCE_PROJECTION_POLICY_VERSION
        artifact["evidence_projection_status"] = "native"
        del artifact["issues_flat"]
        p = _write_run_artifact(tmp_path, artifact)

        assert reproject_artifact(p, REAL_CATALOG)["status"] == "needs_reanalysis"

        reprojected = json.loads(p.read_text(encoding="utf-8"))
        assert "evidence_projection_policy_version" not in reprojected
        assert "evidence_projection_status" not in reprojected
        assert reprojected["renovation_estimate_v4"] is None


# ─── Native stamping through the live writer ─────────────────────────────────

class TestNativeStamping:

    def test_write_photo_intel_stamps_native_projection(self, tmp_path):
        from tools.artifact_writers import write_photo_intel

        image_path = tmp_path / "kitchen_1.jpg"
        image_path.write_bytes(b"fake image")
        catalog = {
            "version": "test",
            "trade_buckets": [{"id": "kitchen_cabinets_counters", "name": "Kitchen"}],
            "items": [
                {
                    "id": "outdated_or_damaged_cabinets",
                    "name": "Outdated Cabinets",
                    "kind": "defect",
                    "severity": 3,
                    "scope": "replace",
                    "trade_bucket": "kitchen_cabinets_counters",
                    "cost": {"mode": "heuristic"},
                    "estimate": {
                        "estimate_tier": "high",
                        "strategy": "replace_only",
                        "group": "kitchen",
                        "stack_behavior": "group_cap",
                    },
                },
            ],
        }
        result = SimpleNamespace(
            image_path=str(image_path),
            scene="kitchen",
            scene_classifier={
                "scene": "kitchen",
                "canonical_issues": [
                    {"description": "Cabinets are dated.",
                     "catalogItemId": "outdated_or_damaged_cabinets", "label": "defect"},
                ],
                "verified_issues": [
                    {"description": "Cabinets are dated.",
                     "catalogItemId": "outdated_or_damaged_cabinets", "label": "defect"},
                ],
                "matched_issues": [],
                "passes": {
                    "1a": {"scene": "kitchen", "confidence": 0.9, "reasoning": ""},
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
            job_id="job_1",
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

        assert artifact["evidence_projection_status"] == "native"
        assert artifact["evidence_projection_policy_version"] == EVIDENCE_PROJECTION_POLICY_VERSION

        v4 = artifact["renovation_estimate_v4"]
        prov = v4["rehab_evidence_projection_v1"]["provenance"]
        assert prov["projection_status"] == "native"
        assert prov["source_run_id"] == "job_1" == prov["artifact_job_id"]
        assert prov["product_policy_version"] == PRODUCT_POLICY_VERSION
        # A single run-completion timestamp is shared everywhere.
        assert prov["completed_at"] == artifact["run"]["created_at"]
        assert prov["source_artifact"] == f"prop/{tmp_path.name}/photo_intel.json"
        assert "\\" not in prov["source_artifact"] and ":" not in prov["source_artifact"]
        assert v4["rehab_evidence_projection_v1"]["projection_id"].startswith("rep1_")
        _assert_projection_matches_estimate(v4)


# ─── 2100 Aftonbrae Dr golden vectors (cross-repo contract) ──────────────────

def test_aftonbrae_golden_projection_vectors():
    """redfin_125800935 — the required end-to-end case. These raw vectors are
    pinned verbatim in tools/test_rehab_evidence_projection.py and in the
    frontend rehabProjection tests; changing them here without the others is a
    cross-repo contract break. Rounding (compact_currency_v1) is asserted only
    in the frontend."""
    proj = build_rehab_evidence_projection({
        "version": "renovation_estimate_v4",
        "final_rehab_resale_ready": {"low": 4567, "high": 16125},
        "totals": {
            "inspection_allowance_total": {"low": 158, "high": 633},
            "risk_exposure_total": {"low": 1187, "high": 19788},
        },
    })
    assert proj["photo_supported"] == {"low": 4567, "high": 16125}
    assert proj["needs_inspection"] == {"low": 158, "high": 633}
    assert proj["headline"] == {"low": 4725, "high": 16758}
    assert proj["risk_exposure"] == {"low": 1187, "high": 19788}
    assert proj["risk_exposure_included"] is False
    assert proj["reconciliation"]["raw_exact"] is True
    assert proj["rounding_policy"] == "compact_currency_v1"

    ids = []
    for status in ("native", "reprojected"):
        stamped = build_rehab_evidence_projection({
            "version": "renovation_estimate_v4",
            "final_rehab_resale_ready": {"low": 4567, "high": 16125},
            "totals": {
                "inspection_allowance_total": {"low": 158, "high": 633},
                "risk_exposure_total": {"low": 1187, "high": 19788},
            },
        })
        stamp_evidence_projection_provenance(
            stamped,
            run_id="20260528_031858_6d1c59a0",
            completed_at="2026-05-28T08:27:50.449531Z",
            source_artifact="redfin_125800935/20260528_031858_6d1c59a0/photo_intel.json",
            projection_status=status,
            product_policy_version="quarantine_v1",
        )
        assert status in SERVABLE_EVIDENCE_PROJECTION_STATUSES
        ids.append(stamped["projection_id"])
    # projection_status is excluded from the content hash: native and
    # reprojected of the same content/run share an id.
    assert ids[0] == ids[1]
    assert ids[0].startswith("rep1_")
