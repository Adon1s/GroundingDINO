"""Session 9 choke-point budget guard tests: config resolvers, the VLM
reservation formula, strict model->ledger mapping, metering through the real
VLM client over the conftest FakeOpenAI, the external-reservation skip flag
(no double debit with the Terra/Sol review hooks), the shadow-seam quota
carve-outs, and the persisted token_usage telemetry block. No live provider
is ever called.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_vlm_budget_guard.py -q
"""
import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import FakeOpenAI, openai_response
from tests.test_renovation_architecture_runtime import _init_shadow, _run_writer
from tests.test_renovation_architecture_sol import (
    SOL_MODEL as SOL_HOOK_MODEL,
    _approve_all,
    _fixture,
    _run,
    _sol_response,
)
from tests.test_renovation_architecture_terra import (
    TERRA_MODEL as TERRA_HOOK_MODEL,
    _conditions,
    _kitchen_setup,
    _review_response,
    _run_review,
)
from tests.test_renovation_architecture_catalog import _v31_item
from tools import pipeline_config
from tools import vlm_client as vlm_module
from tools.artifact_writers import (
    _build_token_usage_block,
    _write_renovation_architecture_estimate,
)
from tools.failure_taxonomy import classify_failure
from tools.renovation_architecture import runtime as runtime_module
from tools.renovation_architecture import review_pipeline
from tools.renovation_architecture.runtime import (
    build_estimate_envelope,
    reset_runtime_for_tests,
)
from tools.renovation_architecture.usage_guard import (
    LEDGER_RELATIVE_PATH,
    SOL_LEDGER_RELATIVE_PATH,
    VLM_RESERVATION_FLOOR_TOKENS,
    SolDailyBudgetExceeded,
    TerraDailyBudgetExceeded,
    TerraUsageLedger,
    VlmBudgetGuardConfigError,
    estimate_vlm_reservation_tokens,
    external_reservation,
    maybe_reserve_vlm_call,
)
from tools.scene_classifier_passes import PassExecutionError
from tools.vlm_client import VLMClient

TERRA_MODEL = "gpt-5.6-terra"
SOL_MODEL = "gpt-5.6-sol"


@pytest.fixture(autouse=True)
def _clean_runtime():
    reset_runtime_for_tests()
    yield
    reset_runtime_for_tests()


@pytest.fixture
def guard_on(tmp_path, monkeypatch):
    """Guard active with tmp_path as the shared usage root."""
    monkeypatch.setattr(pipeline_config, "RENOVATION_VLM_BUDGET_GUARD", True)
    monkeypatch.setattr(
        pipeline_config, "RENOVATION_TERRA_USAGE_ROOT", str(tmp_path)
    )
    monkeypatch.setattr(pipeline_config, "RENOVATION_TERRA_MODEL", TERRA_MODEL)
    monkeypatch.setattr(pipeline_config, "RENOVATION_SOL_MODEL", SOL_MODEL)
    return tmp_path


def _terra_rows(root):
    conn = sqlite3.connect(str(root / LEDGER_RELATIVE_PATH))
    try:
        return conn.execute(
            "SELECT property_key, source_run_id, estimate_unit_id, state, "
            "debited_tokens, provider_total_tokens FROM terra_usage ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def _sol_rows(root):
    conn = sqlite3.connect(str(root / SOL_LEDGER_RELATIVE_PATH))
    try:
        return conn.execute(
            "SELECT state, debited_tokens, provider_total_tokens "
            "FROM sol_usage ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def _metered_response(*, input_tokens=800, output_tokens=200, cached=0):
    response = openai_response(text='{"ok": true}')
    response.usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    return response


def _analyze_text(client, *, model=TERRA_MODEL, max_tokens=500, **kwargs):
    return asyncio.run(client.analyze_text(
        system_prompt="sys", user_prompt="user", model=model,
        provider="openai", api_key="test-key", max_tokens=max_tokens, **kwargs
    ))


def _quota_error():
    return PassExecutionError(
        "terra_review", "request", "daily Terra ceiling would be exceeded",
        code="TerraDailyBudgetExceeded", provider="openai", model="m",
    )


# ── config resolvers ─────────────────────────────────────────────────────────

class TestResolvers:
    @pytest.mark.parametrize("raw", [None, "", "  ", "0", "false"])
    def test_guard_defaults_off(self, raw):
        assert pipeline_config.resolve_renovation_vlm_budget_guard(raw) is False

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE"])
    def test_guard_on(self, raw):
        assert pipeline_config.resolve_renovation_vlm_budget_guard(raw) is True

    def test_guard_invalid_raises(self):
        with pytest.raises(ValueError, match="RENOVATION_VLM_BUDGET_GUARD"):
            pipeline_config.resolve_renovation_vlm_budget_guard("yes please")

    def test_terra_ceiling_default(self):
        for raw in (None, "", "  "):
            assert pipeline_config.resolve_renovation_terra_daily_ceiling(
                raw
            ) == 2_500_000

    def test_terra_ceiling_explicit(self):
        assert pipeline_config.resolve_renovation_terra_daily_ceiling(
            "100000"
        ) == 100_000

    @pytest.mark.parametrize("raw", ["0", "-5", "many"])
    def test_terra_ceiling_invalid_raises(self, raw):
        with pytest.raises(
            ValueError, match="RENOVATION_TERRA_DAILY_TOKEN_CEILING"
        ):
            pipeline_config.resolve_renovation_terra_daily_ceiling(raw)


# ── reservation formula ──────────────────────────────────────────────────────

class TestVlmReservationFormula:
    def test_floor_applies_to_small_requests(self):
        assert estimate_vlm_reservation_tokens(
            max_output_tokens=500, prompt_chars=100, image_count=0
        ) == VLM_RESERVATION_FLOOR_TOKENS == 4_000

    def test_large_requests_use_the_component_sum(self):
        assert estimate_vlm_reservation_tokens(
            max_output_tokens=8_192, prompt_chars=2_000, image_count=2
        ) == 8_192 + 2_000 + 10_000


# ── model -> ledger mapping (direct calls) ───────────────────────────────────

class TestModelMapping:
    def test_terra_model_debits_the_terra_ledger(self, guard_on):
        reservation = maybe_reserve_vlm_call(
            model=TERRA_MODEL, pass_key="2a", max_output_tokens=500,
            prompt_chars=10, image_count=0,
        )
        reservation.settle(1_234)
        ((_, _, unit, state, debited, provider_total),) = _terra_rows(guard_on)
        assert (unit, state, debited, provider_total) == (
            "2a", "settled", 1_234, 1_234
        )
        assert not (guard_on / SOL_LEDGER_RELATIVE_PATH).exists()

    def test_sol_model_debits_the_sol_ledger(self, guard_on):
        reservation = maybe_reserve_vlm_call(
            model=SOL_MODEL, pass_key="2f", max_output_tokens=500,
            prompt_chars=10, image_count=0,
        )
        reservation.settle(None)
        assert len(_sol_rows(guard_on)) == 1
        assert not (guard_on / LEDGER_RELATIVE_PATH).exists()

    def test_reasoning_suffix_normalizes_to_the_base_model(self, guard_on):
        maybe_reserve_vlm_call(
            model=f"{TERRA_MODEL}:high", pass_key="2f", max_output_tokens=500,
            prompt_chars=10, image_count=0,
        )
        assert len(_terra_rows(guard_on)) == 1

    def test_unknown_model_fails_closed_as_dependency(self, guard_on):
        with pytest.raises(VlmBudgetGuardConfigError, match="no daily ledger"):
            maybe_reserve_vlm_call(
                model="gpt-5.4-mini", pass_key="2a", max_output_tokens=500,
                prompt_chars=10, image_count=0,
            )
        assert classify_failure(
            VlmBudgetGuardConfigError("x")
        ).category == "dependency"
        assert not (guard_on / LEDGER_RELATIVE_PATH).exists()

    def test_missing_usage_root_fails_before_any_ledger(
        self, guard_on, monkeypatch
    ):
        monkeypatch.setattr(pipeline_config, "RENOVATION_TERRA_USAGE_ROOT", "")
        with pytest.raises(
            VlmBudgetGuardConfigError, match="RENOVATION_TERRA_USAGE_ROOT"
        ):
            maybe_reserve_vlm_call(
                model=TERRA_MODEL, pass_key="2a", max_output_tokens=500,
                prompt_chars=10, image_count=0,
            )

    def test_guard_off_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_VLM_BUDGET_GUARD", False
        )
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_TERRA_USAGE_ROOT", str(tmp_path)
        )
        assert maybe_reserve_vlm_call(
            model=TERRA_MODEL, pass_key="2a", max_output_tokens=500,
            prompt_chars=10, image_count=0,
        ) is None
        assert not (tmp_path / LEDGER_RELATIVE_PATH).exists()

    def test_external_reservation_skips_the_choke_point(self, guard_on):
        with external_reservation():
            assert maybe_reserve_vlm_call(
                model=TERRA_MODEL, pass_key="2a", max_output_tokens=500,
                prompt_chars=10, image_count=0,
            ) is None
        assert not (guard_on / LEDGER_RELATIVE_PATH).exists()


# ── through the real client over FakeOpenAI ──────────────────────────────────

class TestThroughClient:
    def test_fresh_call_settles_to_provider_truth(self, guard_on):
        client = VLMClient()
        fake = FakeOpenAI(responses=[_metered_response()]).attach(client)
        _analyze_text(client, analysis_pass="Pass 2a batch")
        ((prop, run, unit, state, debited, provider_total),) = _terra_rows(
            guard_on
        )
        assert (prop, run) == ("vlm_choke_point", "vlm")  # no budget_context
        assert (unit, state, debited, provider_total) == (
            "2a", "settled", 1_000, 1_000
        )
        assert len(fake.requests) == 1

    def test_provider_without_usage_keeps_the_reservation(self, guard_on):
        client = VLMClient()
        FakeOpenAI(responses=[openai_response(text="{}")]).attach(client)
        _analyze_text(client)
        ((_, _, _, state, debited, provider_total),) = _terra_rows(guard_on)
        assert (state, debited, provider_total) == (
            "settled", VLM_RESERVATION_FLOOR_TOKENS, None
        )

    def test_provider_failure_retains_the_reservation(self, guard_on):
        client = VLMClient()
        FakeOpenAI(raises=RuntimeError("provider melted")).attach(client)
        with pytest.raises(RuntimeError, match="provider melted"):
            _analyze_text(client)
        ((_, _, _, state, debited, provider_total),) = _terra_rows(guard_on)
        assert (state, debited, provider_total) == (
            "settled", VLM_RESERVATION_FLOOR_TOKENS, None
        )

    def test_denial_happens_before_the_provider_call(
        self, guard_on, monkeypatch
    ):
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_TERRA_DAILY_TOKEN_CEILING", 4_500
        )
        client = VLMClient()
        fake = FakeOpenAI(
            responses=[_metered_response(), _metered_response()]
        ).attach(client)
        _analyze_text(client)  # settles to 1_000 of the 4_500 ceiling
        with pytest.raises(TerraDailyBudgetExceeded):
            _analyze_text(client)  # 1_000 + 4_000 floor > 4_500
        assert len(fake.requests) == 1  # denial reached no provider
        assert len(_terra_rows(guard_on)) == 1

    def test_guard_off_touches_no_ledger(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_VLM_BUDGET_GUARD", False
        )
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_TERRA_USAGE_ROOT", str(tmp_path)
        )
        client = VLMClient()
        FakeOpenAI(responses=[_metered_response()]).attach(client)
        _analyze_text(client)
        assert not (tmp_path / LEDGER_RELATIVE_PATH).exists()

    def test_lmstudio_calls_are_never_guarded(self, guard_on, monkeypatch):
        class _Response:
            status_code = 200
            text = "ok"

            def json(self):
                return {
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3,
                              "total_tokens": 15},
                }

        monkeypatch.setattr(
            vlm_module.requests, "post", lambda *a, **k: _Response()
        )
        client = VLMClient()
        asyncio.run(client.analyze_text(
            system_prompt="sys", user_prompt="user", model="local-qwen",
            provider="lmstudio", url="http://localhost:1234", max_tokens=100,
        ))
        assert not (guard_on / LEDGER_RELATIVE_PATH).exists()
        assert not (guard_on / SOL_LEDGER_RELATIVE_PATH).exists()

    def test_budget_context_lands_in_the_ledger_row(self, guard_on):
        client = VLMClient()
        client.budget_context = {
            "property_key": "prop_x", "source_run_id": "run_y",
        }
        FakeOpenAI(responses=[_metered_response()]).attach(client)
        _analyze_text(client)
        ((prop, run, _, _, _, _),) = _terra_rows(guard_on)
        assert (prop, run) == ("prop_x", "run_y")


# ── no double debit with the review hooks ────────────────────────────────────

class TestHookSkipFlag:
    def test_sol_hook_debits_exactly_once_under_the_guard(
        self, tmp_path, guard_on, monkeypatch
    ):
        """The hook reserves/settles itself and wraps its provider call in
        external_reservation(); without the flag this call would map to the
        Sol ledger at the choke point too and debit twice."""
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_SOL_MODEL", SOL_HOOK_MODEL
        )
        base, candidates = _fixture()
        fake = FakeOpenAI(responses=[_sol_response(_approve_all(candidates))])
        result, _ = _run(tmp_path, base, candidates, fake)
        assert len(_sol_rows(guard_on)) == 1  # the hook's row, nothing else
        assert result["sol_calls"][0]["usage_source"] == "provider"

    def test_terra_hook_debits_exactly_once_under_the_guard(
        self, tmp_path, guard_on, monkeypatch
    ):
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_TERRA_MODEL", TERRA_HOOK_MODEL
        )
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        conditions = _conditions(runtime, issues, photos)
        fake = FakeOpenAI(responses=[_review_response(
            {c.condition_id: "supported" for c in conditions}
        )])
        _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert len(_terra_rows(guard_on)) == 1

    def test_terra_hook_honors_the_ceiling_env(
        self, tmp_path, guard_on, monkeypatch
    ):
        """The hook's own ledger now reads RENOVATION_TERRA_DAILY_TOKEN_CEILING,
        so a paid-day override applies to the review calls too."""
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_TERRA_DAILY_TOKEN_CEILING", 10_000
        )
        runtime, issues, photos, paths = _kitchen_setup(
            tmp_path, _v31_item("worn_counter")
        )
        fake = FakeOpenAI(responses=[_review_response({})])
        with pytest.raises(PassExecutionError) as excinfo:
            # The 25k review floor cannot fit under the 10k ceiling.
            _run_review(tmp_path, runtime, issues, photos, paths, fake)
        assert excinfo.value.code == "TerraDailyBudgetExceeded"
        assert fake.requests == []


# ── shadow-seam quota carve-outs ─────────────────────────────────────────────

class TestQuotaCarveOuts:
    def _envelope(self, monkeypatch, exc, *, guard):
        _init_shadow()
        monkeypatch.setattr(
            pipeline_config, "RENOVATION_VLM_BUDGET_GUARD", guard
        )

        def _raise(**kwargs):
            raise exc

        monkeypatch.setattr(review_pipeline, "run_condition_review", _raise)
        return lambda: build_estimate_envelope(
            property_key="prop", run_id="run_1",
            created_at="2026-08-17T00:00:00Z", source_artifact="a/b/c.json",
            issues_flat=[], photos={},
        )

    def test_runtime_reraises_quota_with_the_guard_on(self, monkeypatch):
        build = self._envelope(monkeypatch, _quota_error(), guard=True)
        with pytest.raises(PassExecutionError) as excinfo:
            build()
        assert excinfo.value.code == "TerraDailyBudgetExceeded"

    def test_runtime_keeps_the_failed_envelope_with_the_guard_off(
        self, monkeypatch
    ):
        envelope = self._envelope(monkeypatch, _quota_error(), guard=False)()
        assert envelope["state"] == "failed"
        assert envelope["reason"] == "quota"

    def test_runtime_keeps_non_quota_failures_enveloped(self, monkeypatch):
        envelope = self._envelope(
            monkeypatch, RuntimeError("boom"), guard=True
        )()
        assert envelope["state"] == "failed"
        assert envelope["reason"] != "quota"

    def _seam(self, monkeypatch, exc, *, guard):
        monkeypatch.setattr(
            runtime_module, "build_estimate_envelope",
            lambda **kwargs: (_ for _ in ()).throw(exc),
        )
        photo_intel = {"analysis_debug": {}}
        cfg = SimpleNamespace(
            RENOVATION_ARCHITECTURE_MODE="shadow",
            RENOVATION_VLM_BUDGET_GUARD=guard,
        )
        return lambda: _write_renovation_architecture_estimate(
            cfg=cfg, photo_intel=photo_intel, property_key="prop",
            run_id="run_1", created_at="t", source_artifact="a/b/c.json",
        )

    def test_seam_reraises_quota_with_the_guard_on(self, monkeypatch):
        with pytest.raises(PassExecutionError):
            self._seam(monkeypatch, _quota_error(), guard=True)()

    def test_seam_swallows_quota_with_the_guard_off(self, monkeypatch):
        self._seam(monkeypatch, _quota_error(), guard=False)()  # no raise

    def test_seam_swallows_non_quota_with_the_guard_on(self, monkeypatch):
        self._seam(monkeypatch, RuntimeError("boom"), guard=True)()  # no raise


# ── per-pass telemetry: slugs, per-model buckets, persisted block ────────────

class TestTelemetry:
    def test_pass_key_slug_fallback(self):
        assert VLMClient._canonical_pass_key("Pass 2F verification") == "2f"
        assert VLMClient._canonical_pass_key(
            "Terra condition review"
        ) == "terra_condition_review"
        assert VLMClient._canonical_pass_key(None) == "unattributed"

    def test_per_pass_models_bucket_accumulates_and_resets(self):
        client = VLMClient()
        FakeOpenAI(responses=[
            _metered_response(cached=50), _metered_response(cached=0),
        ]).attach(client)
        for _ in range(2):
            _analyze_text(client, analysis_pass="Pass 2a batch")
        bucket = client.usage_stats["per_pass"]["2a"]["models"]
        assert bucket == {
            f"openai/{TERRA_MODEL}": {
                "metered_calls": 2, "input_tokens": 1_600,
                "cached_input_tokens": 50, "output_tokens": 400,
                "total_tokens": 2_000,
            }
        }
        client.reset_usage_stats()
        assert client.usage_stats["per_pass"] == {}

    def test_token_usage_block_joins_model_routing(self):
        usage_stats = {
            "input_tokens": 800, "cached_input_tokens": 0,
            "output_tokens": 200, "total_tokens": 1_000,
            "attempted_calls": 2, "calls": 2, "failed_calls": 0,
            "metered_calls": 1, "api_duration_sec": 1.5,
            "per_pass": {
                "2f": {"total_tokens": 1_000,
                       "models": {"openai/gpt-5.6-terra": {}}},
                "terra_condition_review": {"total_tokens": 0},
            },
        }
        routing = [
            {"pass": "2f", "model": "gpt-5.6-terra", "model_family": "gpt5",
             "source": "run_override"},
        ]
        block = _build_token_usage_block(usage_stats, model_routing=routing)
        assert block["schema_version"] == 1
        assert block["totals"]["total_tokens"] == 1_000
        assert block["totals"]["cached_input_tokens"] == 0
        assert block["per_pass"]["2f"]["routing"]["model"] == "gpt-5.6-terra"
        assert block["per_pass"]["terra_condition_review"]["routing"] is None

    def test_write_photo_intel_persists_token_usage_in_debug_only(
        self, tmp_path
    ):
        stub = SimpleNamespace(usage_stats={
            "input_tokens": 12, "cached_input_tokens": 3, "output_tokens": 4,
            "total_tokens": 16, "attempted_calls": 1, "calls": 1,
            "failed_calls": 0, "metered_calls": 1, "api_duration_sec": 0.1,
            "per_pass": {"1a": {"total_tokens": 16}},
        })
        slim, debug = _run_writer(
            tmp_path, SimpleNamespace(LM_STUDIO_MODEL="test-model"),
            vlm_client=stub,
        )
        block = debug["analysis_debug"]["token_usage"]
        assert block["totals"]["total_tokens"] == 16
        assert block["totals"]["cached_input_tokens"] == 3
        assert block["per_pass"]["1a"]["routing"] is None
        assert "analysis_debug" not in slim

    def test_write_photo_intel_without_client_omits_token_usage(
        self, tmp_path
    ):
        _, debug = _run_writer(
            tmp_path, SimpleNamespace(LM_STUDIO_MODEL="test-model")
        )
        assert "token_usage" not in debug["analysis_debug"]
