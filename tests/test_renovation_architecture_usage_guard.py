"""Terra usage-guard tests: reservation formula, the cross-process SQLite
UTC-day ledger, pre-call quota denial, settlement, and the call -> unit ->
listing token reconciliation through the review pipeline.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_usage_guard.py -q
"""
import sqlite3
import threading
from pathlib import Path

import pytest

from tests.conftest import FakeOpenAI
from tests.test_renovation_architecture_catalog import _v31_item
from tests.test_renovation_architecture_terra import (
    _conditions,
    _kitchen_setup,
    _review_response,
    _run_review,
)
from tools.failure_taxonomy import classify_failure
from tools.renovation_architecture.usage_guard import (
    LEDGER_RELATIVE_PATH,
    RESERVATION_FLOOR_TOKENS,
    TERRA_DAILY_TOKEN_CEILING,
    TerraDailyBudgetExceeded,
    TerraUsageLedger,
    estimate_reservation_tokens,
    resolve_ledger_root,
)
from tools.scene_classifier_passes import PassExecutionError

FP = "c" * 64


def _reserve(ledger, *, tokens, unit="kitchen_primary"):
    return ledger.reserve(
        property_key="prop", source_run_id="run_1", estimate_unit_id=unit,
        request_fingerprint=FP, tokens=tokens,
    )


def _rows(artifacts_root):
    conn = sqlite3.connect(str(artifacts_root / LEDGER_RELATIVE_PATH))
    try:
        return conn.execute(
            "SELECT estimate_unit_id, state, debited_tokens, "
            "provider_total_tokens, utc_day FROM terra_usage ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


# ── reservation formula ──────────────────────────────────────────────────────

class TestReservationFormula:
    def test_floor_applies_to_small_requests(self):
        assert estimate_reservation_tokens(
            max_output_tokens=512, request_bytes=1_000, image_count=1
        ) == RESERVATION_FLOOR_TOKENS == 25_000

    def test_large_requests_use_the_component_sum(self):
        assert estimate_reservation_tokens(
            max_output_tokens=8_192, request_bytes=40_000, image_count=3
        ) == 8_192 + 40_000 + 5_000 * 3

    def test_daily_ceiling_constant(self):
        assert TERRA_DAILY_TOKEN_CEILING == 2_500_000


# ── ledger primitives ────────────────────────────────────────────────────────

class TestLedger:
    def test_reserve_then_settle_to_provider_truth(self, tmp_path):
        ledger = TerraUsageLedger(tmp_path)
        reservation_id = _reserve(ledger, tokens=25_000)
        assert _rows(tmp_path) == [
            ("kitchen_primary", "reserved", 25_000, None, _rows(tmp_path)[0][4]),
        ]
        debit = ledger.settle(reservation_id, provider_total_tokens=1_234)
        assert debit == 1_234
        ((_, state, debited, provider_total, _),) = _rows(tmp_path)
        assert (state, debited, provider_total) == ("settled", 1_234, 1_234)

    def test_unknown_usage_retains_the_reservation(self, tmp_path):
        ledger = TerraUsageLedger(tmp_path)
        reservation_id = _reserve(ledger, tokens=25_000)
        debit = ledger.settle(reservation_id, provider_total_tokens=None)
        assert debit == 25_000
        ((_, state, debited, provider_total, _),) = _rows(tmp_path)
        assert (state, debited, provider_total) == ("settled", 25_000, None)

    def test_denial_happens_before_any_insert(self, tmp_path):
        ledger = TerraUsageLedger(tmp_path, daily_ceiling=30_000)
        _reserve(ledger, tokens=25_000)
        with pytest.raises(TerraDailyBudgetExceeded, match="would be exceeded"):
            _reserve(ledger, tokens=25_000, unit="bathroom_primary")
        assert len(_rows(tmp_path)) == 1  # the denied attempt left no row

    def test_settled_savings_free_budget_for_later_calls(self, tmp_path):
        ledger = TerraUsageLedger(tmp_path, daily_ceiling=30_000)
        reservation_id = _reserve(ledger, tokens=25_000)
        ledger.settle(reservation_id, provider_total_tokens=1_000)
        _reserve(ledger, tokens=25_000, unit="bathroom_primary")  # now fits

    def test_utc_rollover_resets_the_budget(self, tmp_path, monkeypatch):
        from tools.renovation_architecture import usage_guard

        monkeypatch.setattr(usage_guard, "_utc_today", lambda: "2026-08-14")
        ledger = TerraUsageLedger(tmp_path, daily_ceiling=30_000)
        reservation_id = _reserve(ledger, tokens=28_000)
        with pytest.raises(TerraDailyBudgetExceeded):
            _reserve(ledger, tokens=25_000, unit="bathroom_primary")
        monkeypatch.setattr(usage_guard, "_utc_today", lambda: "2026-08-15")
        _reserve(ledger, tokens=25_000, unit="bathroom_primary")
        # A call straddling midnight settles onto its reservation's day.
        ledger.settle(reservation_id, provider_total_tokens=2_000)
        rows = _rows(tmp_path)
        assert rows[0][4] == "2026-08-14" and rows[0][2] == 2_000
        assert rows[1][4] == "2026-08-15"

    def test_concurrent_reservations_admit_exactly_one(self, tmp_path):
        ledger = TerraUsageLedger(tmp_path, daily_ceiling=30_000)
        barrier = threading.Barrier(2)
        outcomes = []

        def worker(unit):
            barrier.wait()
            try:
                _reserve(ledger, tokens=25_000, unit=unit)
                outcomes.append("ok")
            except TerraDailyBudgetExceeded:
                outcomes.append("denied")

        threads = [
            threading.Thread(target=worker, args=(unit,))
            for unit in ("kitchen_primary", "bathroom_primary")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(outcomes) == ["denied", "ok"]
        assert len(_rows(tmp_path)) == 1


# ── through the review pipeline ──────────────────────────────────────────────

class TestPipelineUsage:
    def test_cached_tokens_reconcile_call_unit_listing(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        fake = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"},
                             usage=(1_000, 250, 200))
        ])
        result, client = _run_review(
            tmp_path, runtime, issues, photos, paths, fake
        )
        (call,) = result["terra_calls"]
        assert call["input_tokens"] == 1_000
        assert call["cached_input_tokens"] == 250
        assert call["output_tokens"] == 200
        assert call["total_tokens"] == 1_200
        assert call["budget_debited_tokens"] == 1_200
        (unit_usage,) = result["terra_unit_usage"]
        listing = result["terra_listing_usage"]
        for name in ("input_tokens", "cached_input_tokens", "output_tokens",
                     "total_tokens", "budget_debited_tokens"):
            assert unit_usage[name] == call[name]
            assert listing[name] == call[name]
        assert listing["call_count"] == 1
        # The additive VLM accounting saw the cached tokens too.
        assert client.usage_stats["cached_input_tokens"] == 250

    def test_provider_without_usage_keeps_the_reservation(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        fake = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"}, usage=None)
        ])
        result, _ = _run_review(tmp_path, runtime, issues, photos, paths, fake)
        (call,) = result["terra_calls"]
        assert call["total_tokens"] == 0  # nothing reported, nothing invented
        # The single-condition prompt sits far under the floor.
        assert call["budget_debited_tokens"] == RESERVATION_FLOOR_TOKENS
        ((_, state, debited, provider_total, _),) = _rows(
            tmp_path / "artifacts"
        )
        assert (state, debited, provider_total) == (
            "settled", RESERVATION_FLOOR_TOKENS, None
        )

    def test_precall_denial_maps_to_quota_and_calls_nothing(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        # Exhaust the day before the run so the pipeline's own reservation
        # cannot fit.
        ledger = TerraUsageLedger(tmp_path / "artifacts")
        _reserve(ledger, tokens=2_490_000, unit="prior_work")
        fake = FakeOpenAI(responses=[_review_response({})])
        with pytest.raises(PassExecutionError) as excinfo:
            _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert excinfo.value.code == "TerraDailyBudgetExceeded"
        assert classify_failure(excinfo.value).category == "quota"
        assert fake.requests == []  # rejected BEFORE the provider call
        rows = _rows(tmp_path / "artifacts")
        assert len(rows) == 1  # only the pre-existing spend, no new debit

    def test_checkpoint_reuse_never_touches_the_ledger(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        fake = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"})
        ])
        _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert len(_rows(tmp_path / "artifacts")) == 1
        strict = FakeOpenAI(raises=AssertionError("provider must not be called"))
        result, _ = _run_review(tmp_path, runtime, issues, photos, paths, strict)
        assert len(_rows(tmp_path / "artifacts")) == 1  # no new reservation
        assert result["terra_calls"][0]["budget_debited_tokens"] == 0

    def test_ledger_lives_under_the_injected_artifacts_root(self, tmp_path):
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        (condition,) = _conditions(runtime, issues, photos)
        fake = FakeOpenAI(responses=[
            _review_response({condition.condition_id: "supported"})
        ])
        _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert (tmp_path / "artifacts" / LEDGER_RELATIVE_PATH).is_file()


class TestSharedUsageRoot:
    """RENOVATION_TERRA_USAGE_ROOT: several artifacts roots, one daily budget.

    The Session 6 canary runs two isolated replica roots that must not each
    get their own 2.5M/day allowance.
    """

    def test_override_redirects_the_ledger_path(self, tmp_path):
        shared = tmp_path / "shared"
        ledger = TerraUsageLedger(
            tmp_path / "replica_1", usage_root_override=str(shared)
        )
        assert ledger.path == shared / LEDGER_RELATIVE_PATH

    def test_default_keeps_the_per_run_artifacts_root(self, tmp_path):
        for override in (None, "", "   "):
            ledger = TerraUsageLedger(tmp_path, usage_root_override=override)
            assert ledger.path == tmp_path / LEDGER_RELATIVE_PATH

    def test_separate_roots_share_one_daily_ceiling(self, tmp_path):
        """Without the override each replica would get a full allowance; with
        it, replica 2's reservation is denied by replica 1's spend."""
        shared = str(tmp_path / "shared")
        replica_1 = TerraUsageLedger(
            tmp_path / "run_1", usage_root_override=shared, daily_ceiling=100_000
        )
        replica_2 = TerraUsageLedger(
            tmp_path / "run_2", usage_root_override=shared, daily_ceiling=100_000
        )
        _reserve(replica_1, tokens=60_000)
        with pytest.raises(TerraDailyBudgetExceeded):
            _reserve(replica_2, tokens=60_000)
        # The same reservation succeeds when the roots are NOT shared.
        isolated = TerraUsageLedger(tmp_path / "run_2", daily_ceiling=100_000)
        assert _reserve(isolated, tokens=60_000)

    def test_resolver_is_pure(self, tmp_path):
        assert resolve_ledger_root(tmp_path) == tmp_path
        assert resolve_ledger_root(tmp_path, override=" C:/shared ") == Path(
            "C:/shared"
        )
