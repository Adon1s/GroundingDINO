"""Selector, runtime-init, and writer-seam tests for the renovation
architecture.

The selector is tested through the pure resolver (never by mutating
os.environ and reloading — see tests/test_kind_ontology_cutover.py for the
pattern). The writer seam is exercised through a real write_photo_intel run
so current-mode key-neutrality and shadow-mode privacy are asserted against
the artifacts actually written to disk.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_renovation_architecture_runtime.py -q
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.pipeline_config as cfg
from tools.analyzer_cli import PATH_OVERRIDE_KEYS, STRING_OVERRIDE_KEYS
from tools.artifact_writers import load_issue_catalog, write_photo_intel
from tools.comparison_common import sha256_file
from tools.renovation_architecture.contracts import SHADOW_DEBUG_KEY
from tools.renovation_architecture.runtime import (
    RenovationArchitectureInitError,
    build_shadow_envelope,
    get_runtime,
    initialize_renovation_architecture,
    reset_runtime_for_tests,
)
from tools.renovation_architecture.validators import validate_envelope

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_V2_PATH = ROOT / "tools" / "issue_catalog_kind_v2.json"


@pytest.fixture(autouse=True)
def _clean_runtime():
    reset_runtime_for_tests()
    yield
    reset_runtime_for_tests()


def _init_shadow(*, terra_model="terra-test", terra_max_output_tokens=8192):
    initialize_renovation_architecture(
        mode="shadow",
        catalog=load_issue_catalog(SHIPPED_V2_PATH),
        catalog_path=SHIPPED_V2_PATH,
        kind_ontology_version="observation_kind_v2",
        terra_model=terra_model,
        terra_max_output_tokens=terra_max_output_tokens,
    )


# ── selector (pure resolver; no env mutation) ────────────────────────────────

class TestSelector:
    def test_runtime_default_is_current(self):
        """Repo/local default stays current; shadow is an explicit opt-in."""
        assert cfg.RENOVATION_ARCHITECTURE_MODE == cfg.RENOVATION_ARCH_CURRENT

    def test_current_accepts_either_ontology(self):
        for ontology in (cfg.KIND_ONTOLOGY_LEGACY_V1, cfg.KIND_ONTOLOGY_V2):
            assert (
                cfg.resolve_renovation_architecture(
                    cfg.RENOVATION_ARCH_CURRENT, kind_ontology_version=ontology
                )
                == "current"
            )

    def test_shadow_requires_the_v2_selector(self):
        assert (
            cfg.resolve_renovation_architecture(
                cfg.RENOVATION_ARCH_SHADOW,
                kind_ontology_version=cfg.KIND_ONTOLOGY_V2,
            )
            == "shadow"
        )
        with pytest.raises(ValueError, match="requires"):
            cfg.resolve_renovation_architecture(
                cfg.RENOVATION_ARCH_SHADOW,
                kind_ontology_version=cfg.KIND_ONTOLOGY_LEGACY_V1,
            )

    def test_new_is_recognized_but_unavailable(self):
        with pytest.raises(ValueError, match="Session 6"):
            cfg.resolve_renovation_architecture(
                cfg.RENOVATION_ARCH_NEW, kind_ontology_version=cfg.KIND_ONTOLOGY_V2
            )

    def test_invalid_values_fail_startup(self):
        for bad in ("shadow_mode", "CURRENT", "", "on"):
            with pytest.raises(ValueError, match="RENOVATION_ARCHITECTURE_MODE"):
                cfg.resolve_renovation_architecture(
                    bad, kind_ontology_version=cfg.KIND_ONTOLOGY_V2
                )

    def test_selector_does_not_touch_kind_ontology_resolution(self):
        """The architecture selector derives nothing else — catalog path and
        pipeline depth stay owned by the kind-ontology selector."""
        assert Path(cfg.ISSUE_CATALOG_PATH).name == "issue_catalog.json"
        assert cfg.KIND_ONTOLOGY_VERSION == cfg.KIND_ONTOLOGY_LEGACY_V1

    def test_selector_has_no_env_override_path(self):
        assert "RENOVATION_ARCHITECTURE_MODE" not in PATH_OVERRIDE_KEYS
        assert "RENOVATION_ARCHITECTURE_MODE" not in STRING_OVERRIDE_KEYS


# ── Terra config resolvers (pure; no env mutation) ───────────────────────────

class TestTerraConfigResolvers:
    def test_model_falls_back_to_openai_model(self):
        assert cfg.resolve_renovation_terra_model("", openai_model="gpt-5.4") == "gpt-5.4"
        assert cfg.resolve_renovation_terra_model("terra-x", openai_model="gpt-5.4") == "terra-x"
        # Both empty resolves to "" here; shadow-mode init rejects it at startup.
        assert cfg.resolve_renovation_terra_model("  ", openai_model=" ") == ""

    def test_max_output_tokens_default_and_parse(self):
        assert cfg.resolve_renovation_terra_max_output_tokens(None) == 8192
        assert cfg.resolve_renovation_terra_max_output_tokens("") == 8192
        assert cfg.resolve_renovation_terra_max_output_tokens(" 4096 ") == 4096

    def test_max_output_tokens_fails_closed(self):
        for bad in ("0", "-5", "many", "8.5"):
            with pytest.raises(ValueError, match="RENOVATION_TERRA_MAX_OUTPUT_TOKENS"):
                cfg.resolve_renovation_terra_max_output_tokens(bad)

    def test_terra_model_has_typed_override_not_string_copy(self):
        """The env bridge re-runs the pure resolvers (fallback + fail-closed
        parse); a plain string copy would skip both."""
        assert "RENOVATION_TERRA_MODEL" not in STRING_OVERRIDE_KEYS
        assert "RENOVATION_TERRA_MAX_OUTPUT_TOKENS" not in STRING_OVERRIDE_KEYS


# ── runtime initialization ───────────────────────────────────────────────────

class TestInitialize:
    def test_current_mode_is_a_noop(self):
        initialize_renovation_architecture(
            mode="current",
            catalog={},
            catalog_path=SHIPPED_V2_PATH,
            kind_ontology_version=cfg.KIND_ONTOLOGY_LEGACY_V1,
        )
        assert get_runtime() is None

    def test_shadow_initializes_from_the_shipped_catalog(self):
        _init_shadow()
        runtime = get_runtime()
        assert runtime is not None
        assert runtime.mode == "shadow"
        assert runtime.catalog_sha256 == sha256_file(SHIPPED_V2_PATH)
        assert runtime.projection_fingerprint == runtime.projection["fingerprint"]
        assert runtime.projection["route_counts"]["work"] == 103
        assert runtime.terra_model == "terra-test"
        assert runtime.terra_max_output_tokens == 8192

    def test_shadow_requires_v2_selector_value(self):
        with pytest.raises(RenovationArchitectureInitError, match="KIND_ONTOLOGY_VERSION"):
            initialize_renovation_architecture(
                mode="shadow",
                catalog=load_issue_catalog(SHIPPED_V2_PATH),
                catalog_path=SHIPPED_V2_PATH,
                kind_ontology_version="legacy_v1",
                terra_model="terra-test",
            )

    def test_shadow_requires_a_terra_model(self):
        """Fail-closed startup: shadow with no model must refuse to boot."""
        for empty in ("", "   "):
            with pytest.raises(RenovationArchitectureInitError, match="Terra model"):
                _init_shadow(terra_model=empty)
            assert get_runtime() is None

    def test_shadow_requires_a_positive_token_cap(self):
        for bad in (0, -1, "8192", None):
            with pytest.raises(
                RenovationArchitectureInitError, match="MAX_OUTPUT_TOKENS"
            ):
                _init_shadow(terra_max_output_tokens=bad)
            assert get_runtime() is None

    def test_shadow_with_invalid_catalog_fails_before_ready(self):
        with pytest.raises(RenovationArchitectureInitError, match="projection build failed"):
            initialize_renovation_architecture(
                mode="shadow",
                catalog={"items": [], "trade_buckets": []},
                catalog_path=SHIPPED_V2_PATH,
                kind_ontology_version="observation_kind_v2",
                terra_model="terra-test",
            )
        assert get_runtime() is None

    def test_new_mode_cannot_initialize(self):
        with pytest.raises(RenovationArchitectureInitError, match="Session 6"):
            initialize_renovation_architecture(
                mode="new",
                catalog={},
                catalog_path=SHIPPED_V2_PATH,
                kind_ontology_version="observation_kind_v2",
            )

    def test_reinitialization_is_idempotent(self):
        _init_shadow()
        first = get_runtime()
        _init_shadow()
        second = get_runtime()
        assert first is not None and second is not None
        assert first.projection_fingerprint == second.projection_fingerprint


# ── shadow envelope builder ──────────────────────────────────────────────────

class TestShadowEnvelope:
    def test_empty_standalone_estimate_envelope_after_init(self):
        """No lane issues -> a complete review AND standalone estimate with
        empty lists and zero buckets, without needing a VLM client, an
        artifacts root, or property metadata (neutral factor)."""
        _init_shadow()
        envelope = build_shadow_envelope(
            property_key="prop_1",
            run_id="run_1",
            created_at="2026-08-14T00:00:00Z",
            source_artifact="prop_1/run_1/photo_intel.json",
        )
        res = validate_envelope(envelope)
        assert res.ok, res.errors
        assert envelope["state"] == "standalone_estimate_complete"
        assert envelope["reason"] is None
        assert envelope["result"]["observed_conditions"] == []
        assert envelope["result"]["terra_calls"] == []
        assert envelope["result"]["terra_listing_usage"]["call_count"] == 0
        assert envelope["result"]["terra_listing_usage"]["total_tokens"] == 0
        assert envelope["result"]["work_items"] == []
        assert envelope["result"]["work_dedup_collisions"] == []
        standalone = envelope["result"]["standalone_estimate"]
        assert standalone["property_cost_factor"] == 1.0
        assert standalone["headline"] == {"low": 0, "high": 0}
        assert all(
            bucket == {"low": 0, "high": 0}
            for bucket in standalone["totals_by_estimate_scope"].values()
        )
        assert envelope["provenance"]["catalog_version"] == "3.1"
        assert envelope["provenance"]["catalog_ontology_version"] == "observation-kind-v2"
        assert envelope["provenance"]["kind_ontology_selector"] == "observation_kind_v2"

    def test_failed_envelope_when_uninitialized(self):
        envelope = build_shadow_envelope(
            property_key="prop_1",
            run_id="run_1",
            created_at="2026-08-14T00:00:00Z",
            source_artifact="prop_1/run_1/photo_intel.json",
        )
        res = validate_envelope(envelope)
        assert res.ok, res.errors
        assert envelope["state"] == "failed"
        assert envelope["reason"] == "runtime_not_initialized"
        assert envelope["provenance"]["catalog_sha256"] is None


# ── writer seam through a real write_photo_intel run ─────────────────────────

def _writer_catalog():
    """Same shape as the TestNativeStamping template: an unversioned test
    catalog, so the publication guards see a legacy-tolerant payload."""
    return {
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


def _run_writer(tmp_path, writer_cfg, *, source_run_id=None):
    """A real write_photo_intel run with no lane issues: the shadow seam's
    plumbing (privacy, neutrality, run identity) without any Terra call. The
    end-to-end shadow review with real conditions lives in
    tests/test_renovation_architecture_terra.py."""
    image_path = tmp_path / "kitchen_1.jpg"
    image_path.write_bytes(b"fake image")
    result = SimpleNamespace(
        image_path=str(image_path),
        scene="kitchen",
        scene_classifier={
            "scene": "kitchen",
            "canonical_issues": [],
            "verified_issues": [],
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
        timestamp="2026-08-14T00:00:00Z",
        artifacts_dir=str(tmp_path),
        results=[result],
        source_run_id=source_run_id,
    )
    output_path = write_photo_intel(
        cfg=writer_cfg,
        job=job,
        detection_backend="test",
        analysis_profile="test",
        use_pass_architecture=True,
        pass_toggles={"2f": False},
        model_overrides={},
        gpt_config=None,
        issue_catalog=_writer_catalog(),
        output_path=tmp_path / "photo_intel.json",
        vlm_client=None,
    )
    slim = json.loads(output_path.read_text(encoding="utf-8"))
    debug = json.loads(
        (output_path.parent / "photo_intel_debug.json").read_text(encoding="utf-8")
    )
    return slim, debug


class TestWriterSeam:
    def test_current_mode_adds_zero_keys(self, tmp_path):
        slim, debug = _run_writer(
            tmp_path, SimpleNamespace(LM_STUDIO_MODEL="test-model")
        )
        assert SHADOW_DEBUG_KEY not in json.dumps(debug)
        assert SHADOW_DEBUG_KEY not in json.dumps(slim)

    def test_shadow_writes_private_standalone_envelope(self, tmp_path):
        _init_shadow()
        slim, debug = _run_writer(
            tmp_path,
            SimpleNamespace(LM_STUDIO_MODEL="test-model",
                            RENOVATION_ARCHITECTURE_MODE="shadow"),
        )
        envelope = debug["analysis_debug"][SHADOW_DEBUG_KEY]
        res = validate_envelope(envelope)
        assert res.ok, res.errors
        assert envelope["state"] == "standalone_estimate_complete"
        assert envelope["result"]["observed_conditions"] == []
        assert envelope["result"]["work_items"] == []
        assert envelope["estimate_id"].startswith("rea1_")
        # No stable external run id on the job -> job_id fallback.
        assert envelope["provenance"]["source_run_id"] == "job_1"
        # The run's single completion timestamp is shared.
        assert envelope["provenance"]["created_at"] == debug["run"]["created_at"]
        # Relative artifact identity, exactly like the evidence projection.
        assert (
            envelope["provenance"]["source_artifact"]
            == f"prop/{tmp_path.name}/photo_intel.json"
        )
        # Privacy: the slim frontend artifact carries neither the envelope nor
        # analysis_debug at all.
        assert "analysis_debug" not in slim
        assert SHADOW_DEBUG_KEY not in json.dumps(slim)
        # v4 output is untouched.
        assert isinstance(debug["renovation_estimate_v4"], dict)

    def test_shadow_uses_the_stable_source_run_id(self, tmp_path):
        """Server jobs carry the API runId; it — not the per-attempt internal
        job id — is the architecture run identity."""
        _init_shadow()
        _, debug = _run_writer(
            tmp_path,
            SimpleNamespace(LM_STUDIO_MODEL="test-model",
                            RENOVATION_ARCHITECTURE_MODE="shadow"),
            source_run_id="stable_run_7",
        )
        envelope = debug["analysis_debug"][SHADOW_DEBUG_KEY]
        assert envelope["provenance"]["source_run_id"] == "stable_run_7"

    def test_shadow_uninitialized_runtime_degrades_to_failed_envelope(self, tmp_path):
        slim, debug = _run_writer(
            tmp_path,
            SimpleNamespace(LM_STUDIO_MODEL="test-model",
                            RENOVATION_ARCHITECTURE_MODE="shadow"),
        )
        envelope = debug["analysis_debug"][SHADOW_DEBUG_KEY]
        res = validate_envelope(envelope)
        assert res.ok, res.errors
        assert envelope["state"] == "failed"
        assert envelope["reason"] == "runtime_not_initialized"
        assert isinstance(debug["renovation_estimate_v4"], dict)
        assert "analysis_debug" not in slim

    def test_seam_failure_cannot_fail_the_job(self, tmp_path, monkeypatch):
        _init_shadow()

        def _boom(**_kwargs):
            raise RuntimeError("synthetic seam failure")

        monkeypatch.setattr(
            "tools.renovation_architecture.runtime.build_shadow_envelope", _boom
        )
        slim, debug = _run_writer(
            tmp_path,
            SimpleNamespace(LM_STUDIO_MODEL="test-model",
                            RENOVATION_ARCHITECTURE_MODE="shadow"),
        )
        envelope = debug["analysis_debug"][SHADOW_DEBUG_KEY]
        assert envelope["state"] == "failed"
        assert envelope["reason"] == "shadow_seam_error"
        assert "synthetic seam failure" in envelope["error_detail"]
        assert isinstance(debug["renovation_estimate_v4"], dict)

    def test_seam_fallback_failure_stays_silent(self, tmp_path, monkeypatch):
        _init_shadow()

        def _boom(**_kwargs):
            raise RuntimeError("synthetic seam failure")

        def _logger_error(*_args, **_kwargs):
            raise RuntimeError("logging is also broken")

        from tools import artifact_writers

        monkeypatch.setattr(
            "tools.renovation_architecture.runtime.build_shadow_envelope", _boom
        )
        monkeypatch.setattr(
            artifact_writers,
            "logger",
            SimpleNamespace(
                error=_logger_error,
                info=lambda *a, **k: None,
                warning=lambda *a, **k: None,
                debug=lambda *a, **k: None,
            ),
        )
        slim, debug = _run_writer(
            tmp_path,
            SimpleNamespace(LM_STUDIO_MODEL="test-model",
                            RENOVATION_ARCHITECTURE_MODE="shadow"),
        )
        assert SHADOW_DEBUG_KEY not in json.dumps(debug)
        assert isinstance(debug["renovation_estimate_v4"], dict)
