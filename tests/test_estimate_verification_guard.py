"""Tests for the Pass 2f estimate guard.

The guard gives `requires_2f_for_estimate` a consumer: a candidate that asks
for confirmation and has not received it is withheld from every headline total
and parked, fully priced, in an additive `withheld_estimate` risk lane.

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_estimate_verification_guard.py -q`
"""
from __future__ import annotations

import copy

from tools.rehab_packages import PACKAGE_VERIFICATION_CONFIRMED_BY_RULE
from tools.renovation_estimate import (
    ESTIMATE_GUARD_POLICY_VERSION,
    ESTIMATE_VERIFICATION_CONFIRMED,
    ESTIMATE_VERIFICATION_CONFIRMED_BY_RULE,
    ESTIMATE_VERIFICATION_INVALIDATED,
    ESTIMATE_VERIFICATION_NOT_REQUIRED,
    ESTIMATE_VERIFICATION_UNCONFIRMED,
    WITHHELD_REASON_REQUIRES_2F,
    EstimateCandidate,
    _PACKAGE_STATUS_CONFIRMED_BY_RULE,
    build_withheld_estimate,
    classify_estimate_verification,
    compute_renovation_estimate,
    extract_estimate_candidates,
    partition_withheld_candidates,
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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _guarded_candidate(**overrides) -> EstimateCandidate:
    """A single candidate whose catalog item asks for 2f confirmation."""
    catalog = _make_catalog(_make_item("guarded", estimate=GUARDED_ESTIMATE))
    candidate = extract_estimate_candidates([_make_issue("guarded")], catalog)[0]
    for key, value in overrides.items():
        setattr(candidate, key, value)
    return candidate


def _statuses(estimate_result) -> dict:
    """catalog_item_id -> estimate_verification_status, over priced groups."""
    return {
        li["catalog_item_id"]: li["estimate_verification_status"]
        for group in estimate_result["groups"]
        for li in group["line_items"]
    }


def _withheld_ids(estimate_result) -> set:
    return {
        li["catalog_item_id"]
        for li in estimate_result["withheld_estimate"]["line_items"]
    }


# ─── the flag itself ─────────────────────────────────────────────────────────

class TestRequires2fResolution:
    def test_guard_is_opt_in_not_tier_derived(self):
        """High/medium tier alone must not opt an item into the guard.

        Defaulting on tier was measured at a 37% cut to corpus final_rehab
        high, because Pass 2f reviews packages and cannot confirm a standalone
        item. Guarding is a per-item catalog decision.
        """
        assert resolve_estimate_meta({"estimate_tier": "high"}).requires_2f_for_estimate is False
        assert resolve_estimate_meta({"estimate_tier": "medium"}).requires_2f_for_estimate is False
        assert resolve_estimate_meta({"estimate_tier": "minor"}).requires_2f_for_estimate is False

    def test_explicit_flag_is_honoured(self):
        meta = resolve_estimate_meta({
            "estimate_tier": "medium", "requires_2f_for_estimate": True,
        })
        assert meta.requires_2f_for_estimate is True

    def test_top_level_flag_overrides_when_block_is_silent(self):
        from tools.renovation_estimate import resolve_catalog_estimate_meta

        item = {
            "estimate": {"estimate_tier": "high"},
            "requires_2f_for_estimate": True,
        }
        assert resolve_catalog_estimate_meta(item).requires_2f_for_estimate is True

    def test_block_wins_over_top_level(self):
        from tools.renovation_estimate import resolve_catalog_estimate_meta

        item = {
            "estimate": {"estimate_tier": "high", "requires_2f_for_estimate": False},
            "requires_2f_for_estimate": True,
        }
        assert resolve_catalog_estimate_meta(item).requires_2f_for_estimate is False


# ─── classifier state matrix ─────────────────────────────────────────────────

class TestClassifyEstimateVerification:
    def test_opt_out_prices_without_confirmation(self):
        catalog = _make_catalog(_make_item("open", estimate=HIGH_ESTIMATE))
        candidate = extract_estimate_candidates([_make_issue("open")], catalog)[0]
        assert classify_estimate_verification(candidate) == ESTIMATE_VERIFICATION_NOT_REQUIRED

    def test_unconfirmed_standalone_is_withheld(self):
        candidate = _guarded_candidate()
        assert classify_estimate_verification(candidate) == ESTIMATE_VERIFICATION_UNCONFIRMED

    def test_confirmed_package_prices(self):
        candidate = _guarded_candidate(pass_2f_applied=True, is_valid_detection=True)
        assert classify_estimate_verification(candidate) == ESTIMATE_VERIFICATION_CONFIRMED

    def test_confirmed_by_rule_prices(self):
        candidate = _guarded_candidate(
            is_valid_detection=True,
            pass_2f_applied=False,
            visual_verification_status=PACKAGE_VERIFICATION_CONFIRMED_BY_RULE,
        )
        assert classify_estimate_verification(candidate) == ESTIMATE_VERIFICATION_CONFIRMED_BY_RULE

    def test_issue_confirmed_from_rejected_package_prices(self):
        """2f rejected the bundle but confirmed this issue is visibly there.

        apply_package_verifications_to_candidates sets pass_2f_applied=True on
        that path, so the guard must let the line item price standalone.
        """
        candidate = _guarded_candidate(
            is_valid_detection=True,
            pass_2f_applied=True,
            package_id=None,
            visual_verification_status="rejected",
        )
        assert classify_estimate_verification(candidate) == ESTIMATE_VERIFICATION_CONFIRMED

    def test_invalidated_stays_invalidated_not_withheld(self):
        """Invalidation outranks the guard and keeps zero-dollar pricing."""
        candidate = _guarded_candidate(is_valid_detection=False, pass_2f_applied=False)
        assert classify_estimate_verification(candidate) == ESTIMATE_VERIFICATION_INVALIDATED

    def test_invalidated_opt_out_item_also_stays_invalidated(self):
        catalog = _make_catalog(_make_item("open", estimate=HIGH_ESTIMATE))
        candidate = extract_estimate_candidates([_make_issue("open")], catalog)[0]
        candidate.is_valid_detection = False
        assert classify_estimate_verification(candidate) == ESTIMATE_VERIFICATION_INVALIDATED

    def test_confirmed_by_rule_literal_matches_rehab_packages(self):
        """renovation_estimate mirrors the constant to dodge a circular import."""
        assert _PACKAGE_STATUS_CONFIRMED_BY_RULE == PACKAGE_VERIFICATION_CONFIRMED_BY_RULE


class TestPartition:
    def test_partition_stamps_status_and_splits(self):
        guarded = _guarded_candidate()
        open_catalog = _make_catalog(_make_item("open", estimate=HIGH_ESTIMATE))
        open_candidate = extract_estimate_candidates([_make_issue("open")], open_catalog)[0]

        priced, withheld = partition_withheld_candidates([guarded, open_candidate])

        assert [c.catalog_item_id for c in priced] == ["open"]
        assert [c.catalog_item_id for c in withheld] == ["guarded"]
        assert open_candidate.estimate_verification_status == ESTIMATE_VERIFICATION_NOT_REQUIRED
        assert guarded.estimate_verification_status == ESTIMATE_VERIFICATION_UNCONFIRMED

    def test_withheld_payload_shape(self):
        withheld = build_withheld_estimate([_guarded_candidate(
            estimate_verification_status=ESTIMATE_VERIFICATION_UNCONFIRMED,
        )])
        assert withheld["policy_version"] == ESTIMATE_GUARD_POLICY_VERSION
        (item,) = withheld["line_items"]
        assert item["estimate_eligible"] is False
        assert item["withheld_reason"] == WITHHELD_REASON_REQUIRES_2F
        assert item["cost_high"] > 0
        assert withheld["total"]["high"] == item["cost_high"]
        assert withheld["risk_exposure_total"]["high"] == item["risk_exposure_high"]

    def test_withheld_total_is_uncapped_sum(self):
        """Group caps are a visible-group blending artifact; the withheld lane
        is a risk register and must not silently drop items to a cap."""
        candidates = [_guarded_candidate() for _ in range(3)]
        for i, c in enumerate(candidates):
            c.estimate_unit_id = f"guarded:{i}"
        withheld = build_withheld_estimate(candidates)
        assert withheld["total"]["high"] == sum(
            li["cost_high"] for li in withheld["line_items"]
        )
        assert len(withheld["line_items"]) == 3


# ─── engine integration ──────────────────────────────────────────────────────

class TestComputeRenovationEstimate:
    def test_withheld_absent_from_groups_and_totals(self):
        catalog = _make_catalog(
            _make_item("open", estimate=HIGH_ESTIMATE),
            _make_item("guarded", estimate=GUARDED_ESTIMATE),
        )
        issues = [_make_issue("open"), _make_issue("guarded")]

        result = compute_renovation_estimate(issues, catalog)

        assert _statuses(result) == {"open": ESTIMATE_VERIFICATION_NOT_REQUIRED}
        assert _withheld_ids(result) == {"guarded"}
        assert result["meta"]["priced_candidate_count"] == 1
        assert result["meta"]["withheld_candidate_count"] == 1

    def test_withheld_does_not_change_priced_totals(self):
        """Adding a guarded item must be dollar-neutral on every headline."""
        open_only = _make_catalog(_make_item("open", estimate=HIGH_ESTIMATE))
        with_guarded = _make_catalog(
            _make_item("open", estimate=HIGH_ESTIMATE),
            _make_item("guarded", estimate=GUARDED_ESTIMATE),
        )
        baseline = compute_renovation_estimate([_make_issue("open")], open_only)
        after = compute_renovation_estimate(
            [_make_issue("open"), _make_issue("guarded")], with_guarded,
        )

        assert after["totals"]["probable_total"] == baseline["totals"]["probable_total"]
        assert after["raw_totals"] == baseline["raw_totals"]
        assert after["withheld_estimate"]["total"]["high"] > 0

    def test_unreviewed_risk_total_is_the_withheld_total(self):
        catalog = _make_catalog(_make_item("guarded", estimate=GUARDED_ESTIMATE))
        result = compute_renovation_estimate([_make_issue("guarded")], catalog)
        assert (
            result["totals"]["unreviewed_risk_total"]
            is result["withheld_estimate"]["total"]
        )

    def test_all_withheld_short_circuit_keeps_real_payload(self):
        """A property whose only findings await confirmation still reports
        them — the empty-candidate path must not zero the withheld lane."""
        catalog = _make_catalog(_make_item("guarded", estimate=GUARDED_ESTIMATE))
        result = compute_renovation_estimate([_make_issue("guarded")], catalog)

        assert result["groups"] == []
        assert result["totals"]["probable_total"] == {"low": 0, "high": 0}
        assert result["withheld_estimate"]["total"]["high"] > 0
        assert result["totals"]["unreviewed_risk_total"]["high"] > 0
        assert result["meta"]["priced_candidate_count"] == 0
        assert result["meta"]["withheld_candidate_count"] == 1

    def test_invalidated_item_stays_in_groups_at_zero(self):
        catalog = _make_catalog(_make_item("guarded", estimate=GUARDED_ESTIMATE))
        candidates = extract_estimate_candidates([_make_issue("guarded")], catalog)
        candidates[0].is_valid_detection = False

        result = compute_renovation_estimate(
            [_make_issue("guarded")], catalog, prebuilt_candidates=candidates,
        )

        assert _withheld_ids(result) == set()
        assert _statuses(result) == {"guarded": ESTIMATE_VERIFICATION_INVALIDATED}
        (line_item,) = result["groups"][0]["line_items"]
        assert (line_item["cost_low"], line_item["cost_high"]) == (0, 0)

    def test_mixed_group_prices_confirmed_and_withholds_the_rest(self):
        """Two items in one estimate group, one confirmed, one not."""
        guarded_other = {**GUARDED_ESTIMATE, "group": "other"}
        catalog = _make_catalog(
            _make_item("confirmed_item", estimate=guarded_other),
            _make_item("unconfirmed_item", estimate=guarded_other),
        )
        issues = [_make_issue("confirmed_item"), _make_issue("unconfirmed_item")]
        candidates = extract_estimate_candidates(issues, catalog)
        for candidate in candidates:
            if candidate.catalog_item_id == "confirmed_item":
                candidate.pass_2f_applied = True
                candidate.is_valid_detection = True

        result = compute_renovation_estimate(
            issues, catalog, prebuilt_candidates=candidates,
        )

        assert _statuses(result) == {"confirmed_item": ESTIMATE_VERIFICATION_CONFIRMED}
        assert _withheld_ids(result) == {"unconfirmed_item"}

    def test_validated_total_excludes_confirmed_by_rule(self):
        """Rule-confirmed items price but are never labelled image-validated."""
        catalog = _make_catalog(_make_item("guarded", estimate=GUARDED_ESTIMATE))
        candidates = extract_estimate_candidates([_make_issue("guarded")], catalog)
        candidates[0].is_valid_detection = True
        candidates[0].visual_verification_status = PACKAGE_VERIFICATION_CONFIRMED_BY_RULE

        result = compute_renovation_estimate(
            [_make_issue("guarded")], catalog, prebuilt_candidates=candidates,
        )

        assert _statuses(result) == {"guarded": ESTIMATE_VERIFICATION_CONFIRMED_BY_RULE}
        assert result["totals"]["probable_total"]["high"] > 0
        assert result["totals"]["validated_total"] == {"low": 0, "high": 0}


class TestComputeV4:
    @staticmethod
    def _v4(*, include_guarded: bool):
        items = [_make_item("open", estimate=HIGH_ESTIMATE, trade_bucket="flooring")]
        issues = [_make_issue("open", scene_group="kitchen", photo_key="k1.jpg")]
        if include_guarded:
            items.append(_make_item(
                "guarded", estimate=GUARDED_ESTIMATE, trade_bucket="roof_gutters",
            ))
            issues.append(
                _make_issue("guarded", scene_group="exterior", photo_key="e1.jpg")
            )
        photos = _make_photos(("kitchen", "k1.jpg"), ("exterior", "e1.jpg"))
        return compute_renovation_estimate_v4(issues, _make_catalog(*items), photos)

    def test_guarded_item_is_dollar_neutral_end_to_end(self):
        baseline = self._v4(include_guarded=False)
        after = self._v4(include_guarded=True)

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

    def test_withheld_absent_from_project_scope_and_reconciliation(self):
        v4 = self._v4(include_guarded=True)

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
        assert "guarded" not in member_ids

    def test_withheld_lane_is_reachable_by_dollar_scaling(self):
        """withheld_estimate rides in on the v3 return, so it is already in the
        tree when compute_v4 calls scale_estimate_dollars. Verify the walker
        reaches it — and that aliasing unreviewed_risk_total to the same dict
        scales it exactly once, not twice.
        """
        from tools.cost_factors import scale_estimate_dollars

        catalog = _make_catalog(_make_item("guarded", estimate=GUARDED_ESTIMATE))
        issues = [_make_issue("guarded", scene_group="kitchen", photo_key="k1.jpg")]
        v4 = compute_renovation_estimate_v4(
            issues, catalog, _make_photos(("kitchen", "k1.jpg")),
        )
        before = v4["withheld_estimate"]["total"]["high"]
        assert before > 0

        scale_estimate_dollars(v4, 2.0)

        assert v4["withheld_estimate"]["total"]["high"] == before * 2
        assert v4["totals"]["unreviewed_risk_total"]["high"] == before * 2

    def test_provenance_records_the_guard(self):
        v4 = self._v4(include_guarded=True)
        assert (
            v4["provenance"]["estimate_verification_policy_version"]
            == ESTIMATE_GUARD_POLICY_VERSION
        )
        assert "estimate_verification_guard" in v4["provenance"]["v4_phases_applied"]


class TestArtifactSurfaces:
    def test_ui_priorities_ignore_withheld_items(self):
        from tools.artifact_writers import _build_ui_priorities_v1

        items = [
            _make_item("open", estimate=HIGH_ESTIMATE),
            _make_item("guarded", estimate=GUARDED_ESTIMATE),
        ]
        catalog = _make_catalog(*items)
        issues = [
            _make_issue("open", scene_group="kitchen", photo_key="k1.jpg"),
            _make_issue("guarded", scene_group="exterior", photo_key="e1.jpg"),
        ]
        photos = _make_photos(("kitchen", "k1.jpg"), ("exterior", "e1.jpg"))
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
        assert "guarded" not in surfaced

    def test_withheld_survives_into_slim_and_debug_artifacts(self, tmp_path):
        """The slim artifact pops only issue_disposition_audit from v4; the
        withheld lane must reach the frontend-facing file."""
        import json

        from tools.artifact_writers import _strip_pass_2f_audit_rationale

        catalog = _make_catalog(_make_item("guarded", estimate=GUARDED_ESTIMATE))
        issues = [_make_issue("guarded", scene_group="kitchen", photo_key="k1.jpg")]
        v4 = compute_renovation_estimate_v4(
            issues, catalog, _make_photos(("kitchen", "k1.jpg")),
        )
        assert v4["withheld_estimate"]["line_items"]

        slim = copy.deepcopy(v4)
        _strip_pass_2f_audit_rationale(slim)
        slim.pop("issue_disposition_audit", None)

        assert slim["withheld_estimate"]["policy_version"] == ESTIMATE_GUARD_POLICY_VERSION
        assert slim["withheld_estimate"]["line_items"]
        # round-trips as JSON (no dataclasses or sets leaked into the payload)
        assert json.loads(json.dumps(slim))["withheld_estimate"]["total"]
