"""Sol usage-guard tests (Session 8): reservation formula, the per-model
SQLite UTC-day ledger, pre-call quota denial, settlement ordering, ledger
isolation from Terra, and the config resolver — over the conftest FakeOpenAI
provider. No live provider is ever called.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_sol_guard.py -q
"""
import sqlite3

import pytest

from tests.conftest import FakeOpenAI
from tests.test_renovation_architecture_sol import (
    _approve_all,
    _fixture,
    _run,
    _sol_response,
)
from tools import pipeline_config
from tools.failure_taxonomy import classify_failure
from tools.renovation_architecture.usage_guard import (
    LEDGER_RELATIVE_PATH,
    SOL_DAILY_TOKEN_CEILING,
    SOL_LEDGER_RELATIVE_PATH,
    SOL_RESERVATION_FLOOR_TOKENS,
    SolDailyBudgetExceeded,
    SolUsageLedger,
    TerraUsageLedger,
    estimate_sol_reservation_tokens,
)
from tools.scene_classifier_passes import PassExecutionError

FP = "d" * 64


def _reserve(ledger, *, tokens, fingerprint=FP):
    return ledger.reserve(
        property_key="prop", source_run_id="run_1", estimate_unit_id="listing",
        request_fingerprint=fingerprint, tokens=tokens,
    )


def _sol_rows(root):
    conn = sqlite3.connect(str(root / SOL_LEDGER_RELATIVE_PATH))
    try:
        return conn.execute(
            "SELECT estimate_unit_id, state, debited_tokens, "
            "provider_total_tokens, utc_day FROM sol_usage ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


# ── reservation formula and constants ────────────────────────────────────────

class TestSolReservationFormula:
    def test_floor_applies_to_small_requests(self):
        assert estimate_sol_reservation_tokens(
            max_output_tokens=512, request_bytes=1_000
        ) == SOL_RESERVATION_FLOOR_TOKENS == 10_000

    def test_large_requests_use_the_component_sum(self):
        assert estimate_sol_reservation_tokens(
            max_output_tokens=8_192, request_bytes=40_000
        ) == 8_192 + 40_000

    def test_daily_ceiling_constant(self):
        assert SOL_DAILY_TOKEN_CEILING == 250_000


# ── ledger primitives (Sol-specific; shared semantics are pinned by the
#    Terra guard suite over the same base class) ─────────────────────────────

class TestSolLedger:
    def test_denial_happens_before_any_insert(self, tmp_path):
        ledger = SolUsageLedger(tmp_path, daily_ceiling=15_000)
        _reserve(ledger, tokens=10_000)
        with pytest.raises(SolDailyBudgetExceeded, match="daily Sol ceiling"):
            _reserve(ledger, tokens=10_000)
        assert len(_sol_rows(tmp_path)) == 1

    def test_terra_and_sol_budgets_are_isolated(self, tmp_path):
        """One shared root, two ledger files: Terra spend must never debit
        the Sol budget and vice versa."""
        terra = TerraUsageLedger(tmp_path, daily_ceiling=100_000)
        sol = SolUsageLedger(tmp_path, daily_ceiling=15_000)
        terra.reserve(
            property_key="prop", source_run_id="run_1",
            estimate_unit_id="kitchen_primary", request_fingerprint=FP,
            tokens=95_000,
        )
        _reserve(sol, tokens=10_000)  # fits: Terra's 95k did not debit Sol
        assert (tmp_path / LEDGER_RELATIVE_PATH).is_file()
        assert (tmp_path / SOL_LEDGER_RELATIVE_PATH).is_file()
        assert len(_sol_rows(tmp_path)) == 1

    def test_shared_usage_root_override(self, tmp_path):
        shared = tmp_path / "shared"
        ledger = SolUsageLedger(
            tmp_path / "run_1", usage_root_override=str(shared)
        )
        assert ledger.path == shared / SOL_LEDGER_RELATIVE_PATH


# ── through run_package_review ───────────────────────────────────────────────

class TestSolPipelineGuard:
    def test_fresh_call_settles_to_provider_truth(self, tmp_path):
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[_sol_response(_approve_all(candidates))])
        _run(tmp_path, base, candidates, fake)
        ((unit, state, debited, provider_total, _),) = _sol_rows(
            tmp_path / "artifacts"
        )
        assert (unit, state, debited, provider_total) == (
            "listing", "settled", 1_020, 1_020
        )

    def test_provider_without_usage_keeps_the_reservation(self, tmp_path):
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[
            _sol_response(_approve_all(candidates), usage=None)
        ])
        _run(tmp_path, base, candidates, fake)
        ((_, state, debited, provider_total, _),) = _sol_rows(
            tmp_path / "artifacts"
        )
        assert (state, debited, provider_total) == (
            "settled", SOL_RESERVATION_FLOOR_TOKENS, None
        )

    def test_provider_failure_retains_the_reservation(self, tmp_path):
        base, candidates = _fixture()
        fake = FakeOpenAI(raises=RuntimeError("provider melted"))
        with pytest.raises(PassExecutionError):
            _run(tmp_path, base, candidates, fake)
        ((_, state, debited, provider_total, _),) = _sol_rows(
            tmp_path / "artifacts"
        )
        assert (state, debited, provider_total) == (
            "settled", SOL_RESERVATION_FLOOR_TOKENS, None
        )

    def test_parse_failure_still_debits(self, tmp_path):
        """Settle-before-parse: a contract-violating response spent tokens."""
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[
            _sol_response({candidates[0]["package_candidate_id"]: "approve"})
        ])  # missing the second decision -> parse failure after the call
        with pytest.raises(PassExecutionError):
            _run(tmp_path, base, candidates, fake)
        ((_, state, debited, _, _),) = _sol_rows(tmp_path / "artifacts")
        assert state == "settled"
        assert debited == 1_020  # provider truth, not the reservation

    def test_precall_denial_maps_to_quota_and_calls_nothing(self, tmp_path):
        base, candidates = _fixture()
        ledger = SolUsageLedger(tmp_path / "artifacts")
        _reserve(ledger, tokens=SOL_DAILY_TOKEN_CEILING - 5_000)
        fake = FakeOpenAI(responses=[_sol_response(_approve_all(candidates))])
        with pytest.raises(PassExecutionError) as excinfo:
            _run(tmp_path, base, candidates, fake)
        assert excinfo.value.code == "SolDailyBudgetExceeded"
        assert classify_failure(excinfo.value).category == "quota"
        assert fake.requests == []  # rejected BEFORE the provider call
        assert len(_sol_rows(tmp_path / "artifacts")) == 1

    def test_checkpoint_reuse_never_touches_the_ledger(self, tmp_path):
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[_sol_response(_approve_all(candidates))])
        _run(tmp_path, base, candidates, fake)
        assert len(_sol_rows(tmp_path / "artifacts")) == 1
        strict = FakeOpenAI(raises=AssertionError("provider must not be called"))
        result, _ = _run(tmp_path, base, candidates, strict)
        assert len(_sol_rows(tmp_path / "artifacts")) == 1
        assert result["sol_calls"][0]["usage_source"] == "checkpoint"

    def test_zero_candidates_touch_no_ledger(self, tmp_path):
        base, _ = _fixture()
        fake = FakeOpenAI(raises=AssertionError("provider must not be called"))
        _run(tmp_path, base, [], fake)
        assert not (tmp_path / "artifacts" / SOL_LEDGER_RELATIVE_PATH).exists()

    def test_config_ceiling_and_usage_root_are_honored(self, tmp_path, monkeypatch):
        """The hook reads RENOVATION_SOL_DAILY_TOKEN_CEILING and shares
        RENOVATION_TERRA_USAGE_ROOT for the ledger location."""
        shared = tmp_path / "shared_budget"
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_TERRA_USAGE_ROOT", str(shared)
        )
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_SOL_DAILY_TOKEN_CEILING", 9_000
        )
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[_sol_response(_approve_all(candidates))])
        with pytest.raises(PassExecutionError) as excinfo:
            _run(tmp_path, base, candidates, fake)  # floor 10k > ceiling 9k
        assert excinfo.value.code == "SolDailyBudgetExceeded"
        assert fake.requests == []
        assert not (shared / SOL_LEDGER_RELATIVE_PATH).exists() or not _sol_rows(shared)


# ── config resolver ──────────────────────────────────────────────────────────

class TestCeilingResolver:
    def test_default(self):
        for raw in (None, "", "  "):
            assert pipeline_config.resolve_renovation_sol_daily_ceiling(raw) == 250_000

    def test_explicit_value(self):
        assert pipeline_config.resolve_renovation_sol_daily_ceiling("100000") == 100_000

    @pytest.mark.parametrize("raw", ["0", "-5", "many"])
    def test_invalid_raises(self, raw):
        with pytest.raises(ValueError, match="RENOVATION_SOL_DAILY_TOKEN_CEILING"):
            pipeline_config.resolve_renovation_sol_daily_ceiling(raw)
