"""Tests for the evidence-sufficiency gate (`min_photo_evidence`).

An opted-in catalog item with fewer distinct supporting photos than its
threshold is withheld from every headline total and parked, fully priced, in
the `withheld_estimate` risk lane — independent of Pass 2f in both directions:
2f confirmation cannot rescue a single-photo finding, and photo count cannot
bypass a `requires_2f_for_estimate` confirmation.

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_min_photo_evidence_gate.py -q`
"""
from __future__ import annotations

from tools.renovation_estimate import (
    ESTIMATE_VERIFICATION_INSUFFICIENT_EVIDENCE,
    ESTIMATE_VERIFICATION_INVALIDATED,
    ESTIMATE_VERIFICATION_NOT_REQUIRED,
    ESTIMATE_VERIFICATION_UNCONFIRMED,
    WITHHELD_REASON_INSUFFICIENT_PHOTO_EVIDENCE,
    WITHHELD_REASON_REQUIRES_2F,
    build_withheld_estimate,
    classify_estimate_verification,
    compute_renovation_estimate,
    extract_estimate_candidates,
    partition_withheld_candidates,
    resolve_catalog_estimate_meta,
    resolve_estimate_meta,
)
from tools.renovation_estimate_v4 import compute_renovation_estimate_v4

from tests.test_renovation_estimate import (
    GUARDED_ESTIMATE,
    HIGH_ESTIMATE,
    _make_catalog,
    _make_issue,
    _make_item,
)
from tests.test_renovation_estimate_v4 import _make_photos


# ─── fixtures ────────────────────────────────────────────────────────────────

# HIGH_ESTIMATE shape, opted in to the photo gate but not the 2f guard.
GATED_ESTIMATE = {**HIGH_ESTIMATE, "min_photo_evidence": 2}

# Both gates at once: photos and 2f confirmation are independent requirements.
GATED_2F_ESTIMATE = {**GUARDED_ESTIMATE, "min_photo_evidence": 2}

GATED_PER_PROPERTY_ESTIMATE = {
    **HIGH_ESTIMATE,
    "unit_policy": "per_property",
    "min_photo_evidence": 2,
}


def _gated_candidate(photo_count=1, estimate=GATED_ESTIMATE, **overrides):
    """One extracted candidate for a gated item with N distinct photos."""
    catalog = _make_catalog(_make_item("gated", estimate=estimate))
    issues = [
        _make_issue("gated", photo_key=f"p{i}.jpg") for i in range(photo_count)
    ] or [_make_issue("gated", photo_key="")]
    (candidate,) = extract_estimate_candidates(issues, catalog)
    for key, value in overrides.items():
        setattr(candidate, key, value)
    return candidate


def _withheld_by_id(estimate_result) -> dict:
    return {
        li["catalog_item_id"]: li
        for li in estimate_result["withheld_estimate"]["line_items"]
    }


# ─── meta resolution ─────────────────────────────────────────────────────────

class TestMinPhotoEvidenceResolution:
    def test_absent_defaults_to_zero(self):
        assert resolve_estimate_meta({"estimate_tier": "high"}).min_photo_evidence == 0
        assert resolve_estimate_meta(None).min_photo_evidence == 0

    def test_invalid_values_default_to_zero(self):
        for bad in ("2", -1, True, 2.5, None, [2]):
            meta = resolve_estimate_meta(
                {"estimate_tier": "high", "min_photo_evidence": bad}
            )
            assert meta.min_photo_evidence == 0, repr(bad)

    def test_block_value_parsed(self):
        meta = resolve_estimate_meta(GATED_ESTIMATE)
        assert meta.min_photo_evidence == 2

    def test_top_level_override_honoured(self):
        item = {
            "estimate": {"estimate_tier": "high"},
            "min_photo_evidence": 2,
        }
        assert resolve_catalog_estimate_meta(item).min_photo_evidence == 2

    def test_block_wins_over_top_level(self):
        item = {
            "estimate": {"estimate_tier": "high", "min_photo_evidence": 3},
            "min_photo_evidence": 2,
        }
        assert resolve_catalog_estimate_meta(item).min_photo_evidence == 3


# ─── classifier matrix ───────────────────────────────────────────────────────

class TestClassification:
    def test_photo_count_threshold(self):
        expected = {
            0: ESTIMATE_VERIFICATION_INSUFFICIENT_EVIDENCE,
            1: ESTIMATE_VERIFICATION_INSUFFICIENT_EVIDENCE,
            2: ESTIMATE_VERIFICATION_NOT_REQUIRED,
            3: ESTIMATE_VERIFICATION_NOT_REQUIRED,
        }
        for photos, status in expected.items():
            candidate = _gated_candidate(photo_count=photos)
            assert classify_estimate_verification(candidate) == status, photos

    def test_pass_2f_cannot_rescue_single_photo(self):
        """A 2f-confirmed one-photo finding is still single-view evidence."""
        for estimate in (GATED_ESTIMATE, GATED_2F_ESTIMATE):
            candidate = _gated_candidate(
                photo_count=1,
                estimate=estimate,
                pass_2f_applied=True,
                is_valid_detection=True,
            )
            assert (
                classify_estimate_verification(candidate)
                == ESTIMATE_VERIFICATION_INSUFFICIENT_EVIDENCE
            )

    def test_photos_do_not_bypass_2f(self):
        candidate = _gated_candidate(photo_count=2, estimate=GATED_2F_ESTIMATE)
        assert (
            classify_estimate_verification(candidate)
            == ESTIMATE_VERIFICATION_UNCONFIRMED
        )

    def test_invalidated_outranks_the_photo_gate(self):
        candidate = _gated_candidate(photo_count=1, is_valid_detection=False)
        assert (
            classify_estimate_verification(candidate)
            == ESTIMATE_VERIFICATION_INVALIDATED
        )

    def test_ungated_item_unchanged(self):
        catalog = _make_catalog(_make_item("open", estimate=HIGH_ESTIMATE))
        (candidate,) = extract_estimate_candidates([_make_issue("open")], catalog)
        assert (
            classify_estimate_verification(candidate)
            == ESTIMATE_VERIFICATION_NOT_REQUIRED
        )


# ─── photo counting ──────────────────────────────────────────────────────────

class TestPhotoCounting:
    def test_repeated_observations_on_one_photo_count_once(self):
        catalog = _make_catalog(_make_item("gated", estimate=GATED_ESTIMATE))
        issues = [
            _make_issue("gated", photo_key="p0.jpg", issue_id="iss_a"),
            _make_issue("gated", photo_key="p0.jpg", issue_id="iss_b"),
        ]
        (candidate,) = extract_estimate_candidates(issues, catalog)
        assert candidate.distinct_photo_count == 1
        assert (
            classify_estimate_verification(candidate)
            == ESTIMATE_VERIFICATION_INSUFFICIENT_EVIDENCE
        )

    def test_other_candidates_photos_do_not_satisfy_the_threshold(self):
        catalog = _make_catalog(
            _make_item("gated", estimate=GATED_ESTIMATE),
            _make_item("open", estimate=HIGH_ESTIMATE),
        )
        issues = [
            _make_issue("gated", photo_key="p0.jpg"),
            _make_issue("open", photo_key="p1.jpg"),
        ]
        result = compute_renovation_estimate(issues, catalog)
        assert set(_withheld_by_id(result)) == {"gated"}

    def test_unit_resolution_unions_photo_sets(self):
        """Two one-photo candidates merging per_property price as one
        two-photo unit — the gate sees post-resolution evidence."""
        catalog = _make_catalog(
            _make_item("gated", estimate=GATED_PER_PROPERTY_ESTIMATE)
        )
        issues = [
            _make_issue("gated", scene_group="kitchen", photo_key="p0.jpg"),
            _make_issue("gated", scene_group="bathroom", photo_key="p1.jpg"),
        ]
        assert len(extract_estimate_candidates(issues, catalog)) == 2

        result = compute_renovation_estimate(issues, catalog)
        assert _withheld_by_id(result) == {}
        (line_item,) = [
            li for group in result["groups"] for li in group["line_items"]
        ]
        assert line_item["supporting_photo_count"] == 2
        assert (
            line_item["estimate_verification_status"]
            == ESTIMATE_VERIFICATION_NOT_REQUIRED
        )


# ─── withheld lane ───────────────────────────────────────────────────────────

class TestWithheldLane:
    def test_partition_withholds_insufficient_evidence(self):
        gated = _gated_candidate(photo_count=1)
        open_catalog = _make_catalog(_make_item("open", estimate=HIGH_ESTIMATE))
        (open_candidate,) = extract_estimate_candidates(
            [_make_issue("open")], open_catalog
        )

        priced, withheld = partition_withheld_candidates([gated, open_candidate])

        assert [c.catalog_item_id for c in priced] == ["open"]
        assert [c.catalog_item_id for c in withheld] == ["gated"]
        assert (
            gated.estimate_verification_status
            == ESTIMATE_VERIFICATION_INSUFFICIENT_EVIDENCE
        )

    def test_withheld_reason_follows_verification_status(self):
        photo_gated = _gated_candidate(
            photo_count=1,
            estimate_verification_status=ESTIMATE_VERIFICATION_INSUFFICIENT_EVIDENCE,
        )
        two_f_gated = _gated_candidate(
            photo_count=2,
            estimate=GATED_2F_ESTIMATE,
            estimate_verification_status=ESTIMATE_VERIFICATION_UNCONFIRMED,
        )

        withheld = build_withheld_estimate([photo_gated, two_f_gated])

        reasons = [li["withheld_reason"] for li in withheld["line_items"]]
        assert reasons == [
            WITHHELD_REASON_INSUFFICIENT_PHOTO_EVIDENCE,
            WITHHELD_REASON_REQUIRES_2F,
        ]
        assert withheld["total"]["high"] == sum(
            li["cost_high"] for li in withheld["line_items"]
        )
        assert withheld["total"]["high"] > 0


# ─── end-to-end (v4) ─────────────────────────────────────────────────────────

class TestComputeV4:
    @staticmethod
    def _v4(*, gated_estimate=None, gated_photos=1):
        items = [_make_item("open", estimate=HIGH_ESTIMATE, trade_bucket="flooring")]
        issues = [_make_issue("open", scene_group="kitchen", photo_key="k1.jpg")]
        photos = [("kitchen", "k1.jpg")]
        if gated_estimate is not None:
            items.append(_make_item(
                "gated", estimate=gated_estimate, trade_bucket="roof_gutters",
            ))
            for i in range(gated_photos):
                issues.append(_make_issue(
                    "gated", scene_group="exterior", photo_key=f"e{i}.jpg",
                ))
                photos.append(("exterior", f"e{i}.jpg"))
        return compute_renovation_estimate_v4(
            issues, _make_catalog(*items), _make_photos(*photos),
        )

    def test_one_photo_gated_item_is_withheld_and_dollar_neutral(self):
        baseline = self._v4()
        after = self._v4(gated_estimate=GATED_ESTIMATE, gated_photos=1)

        withheld = _withheld_by_id(after)
        assert set(withheld) == {"gated"}
        assert (
            withheld["gated"]["withheld_reason"]
            == WITHHELD_REASON_INSUFFICIENT_PHOTO_EVIDENCE
        )

        for key in (
            "visible_rehab", "package_adjusted_rehab", "final_rehab",
            "final_rehab_required", "final_rehab_resale_ready",
            "final_rehab_full_renewal",
        ):
            assert after[key] == baseline[key], key
        assert after["totals"]["probable_total"] == baseline["totals"]["probable_total"]
        assert (
            after["rehab_evidence_projection_v1"]["headline"]
            == baseline["rehab_evidence_projection_v1"]["headline"]
        )

    def test_withheld_absent_from_every_priced_surface(self):
        v4 = self._v4(gated_estimate=GATED_ESTIMATE, gated_photos=1)

        priced_ids = {
            li["catalog_item_id"]
            for group in v4["groups"]
            for li in group["line_items"]
        }
        assert "gated" not in priced_ids

        scope_trades = {
            trade
            for entry in v4["project_scope_breakdown"]
            for trade in entry["trade_buckets"]
        }
        assert "roof_gutters" not in scope_trades

        member_ids = {
            member.get("catalog_item_id")
            for member in v4["reconciliation"]["estimate_members"]
        }
        assert "gated" not in member_ids

    def test_withheld_absent_from_ui_priorities(self):
        from tools.artifact_writers import _build_ui_priorities_v1

        catalog = _make_catalog(
            _make_item("open", estimate=HIGH_ESTIMATE, trade_bucket="flooring"),
            _make_item("gated", estimate=GATED_ESTIMATE, trade_bucket="roof_gutters"),
        )
        issues = [
            _make_issue("open", scene_group="kitchen", photo_key="k1.jpg"),
            _make_issue("gated", scene_group="exterior", photo_key="e0.jpg"),
        ]
        photos = _make_photos(("kitchen", "k1.jpg"), ("exterior", "e0.jpg"))
        v4 = compute_renovation_estimate_v4(issues, catalog, photos)

        priorities = _build_ui_priorities_v1(
            issues_flat=issues, issue_catalog=catalog, renovation_estimate_v4=v4,
        )
        surfaced = {
            record["catalog_item_id"]
            for lane in (
                "verified_estimate_drivers",
                "high_concern_issues",
                "marketability_signals",
            )
            for record in priorities[lane]
            if isinstance(record, dict) and record.get("catalog_item_id")
        }
        assert "gated" not in surfaced

    def test_two_photo_gated_item_prices_identically_to_ungated(self):
        gated = self._v4(gated_estimate=GATED_ESTIMATE, gated_photos=2)
        ungated = self._v4(gated_estimate=HIGH_ESTIMATE, gated_photos=2)

        assert _withheld_by_id(gated) == {}
        for key in ("visible_rehab", "final_rehab", "final_rehab_required"):
            assert gated[key] == ungated[key], key
        assert gated["totals"]["probable_total"] == ungated["totals"]["probable_total"]

    def test_mixed_reason_lane_sums_both_reasons(self):
        items = [
            _make_item("photo_gated", estimate=GATED_ESTIMATE,
                       trade_bucket="roof_gutters"),
            _make_item("two_f_gated", estimate=GUARDED_ESTIMATE,
                       trade_bucket="flooring"),
        ]
        issues = [
            _make_issue("photo_gated", scene_group="exterior", photo_key="e0.jpg"),
            _make_issue("two_f_gated", scene_group="kitchen", photo_key="k1.jpg"),
        ]
        photos = _make_photos(("exterior", "e0.jpg"), ("kitchen", "k1.jpg"))
        v4 = compute_renovation_estimate_v4(issues, _make_catalog(*items), photos)

        withheld = _withheld_by_id(v4)
        assert {
            item_id: li["withheld_reason"] for item_id, li in withheld.items()
        } == {
            "photo_gated": WITHHELD_REASON_INSUFFICIENT_PHOTO_EVIDENCE,
            "two_f_gated": WITHHELD_REASON_REQUIRES_2F,
        }
        lane = v4["withheld_estimate"]
        assert lane["total"]["high"] == sum(
            li["cost_high"] for li in lane["line_items"]
        )
        assert lane["total"]["high"] > 0
