"""
Tests for embeddings fail-closed preflight and the run-metadata contract.

Previously all three analyzer entry points wrapped retriever construction in
`except Exception: candidate_provider = None`, so a down sidecar produced a run
with zero resolved catalog items, zero-dollar packages, success:true and exit 0.
"""
import json
from pathlib import Path

import pytest

from tools import analyzer_cli, catalog_embeddings
from tools.analyzer_server import _checkpoint_policy_fingerprint, _resolved_max_tokens
from tools.catalog_embeddings import EmbeddingsRuntimeError


# ── the shared factory must not swallow ─────────────────────────────────────

def test_factory_propagates_runtime_error(monkeypatch):
    """Fail-closed is the absence of a handler; nothing may catch this."""
    def boom(*args, **kwargs):
        raise EmbeddingsRuntimeError("sidecar unreachable")

    monkeypatch.setattr(catalog_embeddings, "CatalogEmbeddingsRetriever", boom)
    with pytest.raises(EmbeddingsRuntimeError, match="sidecar unreachable"):
        catalog_embeddings.build_candidate_provider({"items": []})


def test_no_duplicate_retriever_blocks_remain():
    """The three copied init blocks were replaced by the shared factory."""
    for module_path in ("tools/analyzer_cli.py", "tools/analyzer_server.py", "tools/audit_runner.py"):
        source = Path(module_path).read_text(encoding="utf-8")
        assert "CatalogEmbeddingsRetriever(" not in source, module_path


def test_use_embeddings_catalog_flag_is_gone():
    """The 2d pass toggle is now the single switch."""
    from tools import pipeline_config as cfg
    assert not hasattr(cfg, "USE_EMBEDDINGS_CATALOG")


# ── CLI preflight ───────────────────────────────────────────────────────────

def _cli_argv(tmp_path, image, *extra):
    return [
        "analyzer_cli",
        "--property-key", "redfin_test",
        "--images", str(image),
        "--artifacts-root", str(tmp_path / "artifacts"),
        "--concurrency", "1",
        *extra,
    ]


@pytest.fixture
def one_image(tmp_path):
    image = tmp_path / "photo1.jpg"
    image.write_bytes(b"not-a-real-jpeg")
    return image


def test_cli_exits_nonzero_and_writes_nothing_when_embeddings_down(
    tmp_path, one_image, monkeypatch, capsys
):
    monkeypatch.setattr(
        analyzer_cli, "load_issue_catalog", lambda _p: {"items": []}, raising=False
    )
    monkeypatch.setattr(
        catalog_embeddings, "build_candidate_provider",
        lambda _catalog: (_ for _ in ()).throw(EmbeddingsRuntimeError("sidecar down")),
    )
    monkeypatch.setattr("sys.argv", _cli_argv(tmp_path, one_image))

    assert analyzer_cli.main() == 1

    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["success"] is False
    assert "embeddings_init" in summary["error"]
    # nothing may be published from a run that never analysed anything
    assert not list((tmp_path / "artifacts").rglob("photo_intel.json"))


def test_cli_skips_embeddings_entirely_when_2d_disabled(
    tmp_path, one_image, monkeypatch
):
    """--disable-2d is the explicit escape hatch; it must not touch the sidecar."""
    from tools import vlm_client as vlm_module

    monkeypatch.setattr(
        analyzer_cli, "load_issue_catalog", lambda _p: {"items": []}, raising=False
    )
    called = []
    monkeypatch.setattr(
        catalog_embeddings, "build_candidate_provider",
        lambda _catalog: called.append(1),
    )

    # Fail the VLM fast — real connection retries make this test take ~30s, and
    # the client is not what's under test here.
    class _DeadClient:
        usage_stats = {"per_pass": {}}

        def reset_usage_stats(self):
            pass

        async def analyze_image(self, **kwargs):
            raise RuntimeError("no vlm in test")

        async def analyze_text(self, **kwargs):
            raise RuntimeError("no vlm in test")

    monkeypatch.setattr(vlm_module, "create_vlm_client", lambda *a, **k: _DeadClient())
    monkeypatch.setattr("sys.argv", _cli_argv(tmp_path, one_image, "--disable-2d"))

    analyzer_cli.main()  # exit code depends on the VLM, which is not the subject
    assert called == []


# ── checkpoint policy fingerprint ───────────────────────────────────────────

def test_fingerprint_includes_pass_toggles():
    """
    A run with 2d disabled writes checkpoints with no resolved catalog items.
    Reusing them in a 2d-enabled run silently produces an empty property.
    """
    enabled = _checkpoint_policy_fingerprint({}, {}, {"2d": True}, {})
    disabled = _checkpoint_policy_fingerprint({}, {}, {"2d": False}, {})
    assert enabled != disabled


def test_fingerprint_includes_token_caps():
    """Caps change truncation, which changes content."""
    a = _checkpoint_policy_fingerprint({}, {}, {}, {"2a": 2000})
    b = _checkpoint_policy_fingerprint({}, {}, {}, {"2a": 600})
    assert a != b


def test_fingerprint_is_stable_for_equal_inputs():
    args = ({"2f": "gpt-5.6-sol"}, {"2f": "medium"}, {"2d": True}, {"2f": 4096})
    assert _checkpoint_policy_fingerprint(*args) == _checkpoint_policy_fingerprint(*args)


def test_resolved_max_tokens_covers_llm_passes():
    caps = _resolved_max_tokens()
    assert caps["1a"] == 2000
    assert caps["2f"] == 4096
    assert "2e" not in caps  # rule-based, no LLM call
