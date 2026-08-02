"""Unit tests for the OpenAI-compatible embeddings encoder (llama-server sidecar).

The HTTP client is faked, so these tests never touch the network. An opt-in
integration smoke test (RUN_EMBEDDINGS_INTEGRATION=1) exercises the real Q8 server.
"""

import math
import os

import numpy as np
import pytest

from tools.catalog_embeddings import (
    EmbeddingsRuntimeError,
    OpenAICompatibleEncoder,
)


# --------------------------------------------------------------------------- #
# Fake OpenAI client plumbing
# --------------------------------------------------------------------------- #
class _FakeItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeEmbeddings:
    def __init__(self, handler):
        self._handler = handler
        self.calls = []  # list of the `input` batch passed to each create() call

    def create(self, model, input):
        self.calls.append(list(input))
        return self._handler(model, list(input))


class _FakeClient:
    def __init__(self, handler):
        self.embeddings = _FakeEmbeddings(handler)


def _encoder_with(handler, dimension=4, batch_size=32):
    """Build a real encoder but swap in a fake client (no network)."""
    enc = OpenAICompatibleEncoder(
        base_url="http://127.0.0.1:8081/v1",
        model_name="test-model",
        dimension=dimension,
        batch_size=batch_size,
    )
    enc._client = _FakeClient(handler)
    return enc


def _onehot(index, dimension):
    v = [0.0] * dimension
    v[index % dimension] = 1.0
    return v


# --------------------------------------------------------------------------- #
# Happy-path behavior
# --------------------------------------------------------------------------- #
def test_batches_in_groups_of_32():
    def handler(model, inputs):
        return _FakeResp([_FakeItem(i, _onehot(i, 4)) for i in range(len(inputs))])

    enc = _encoder_with(handler, dimension=4, batch_size=32)
    out = enc.encode([f"text-{i}" for i in range(70)])

    assert out.shape == (70, 4)
    # 70 inputs → 32 + 32 + 6 = three create() calls, none larger than 32.
    assert [len(c) for c in enc._client.embeddings.calls] == [32, 32, 6]


def test_restores_order_by_index():
    def handler(model, inputs):
        # Return items with correct indices but in REVERSED order.
        data = [_FakeItem(i, _onehot(i, 4)) for i in range(len(inputs))]
        return _FakeResp(list(reversed(data)))

    enc = _encoder_with(handler, dimension=4)
    out = enc.encode(["a", "b", "c", "d"])
    # Row i must be the one-hot for index i despite the reversed response order.
    for i in range(4):
        assert int(np.argmax(out[i])) == i


def test_converts_to_float32_and_l2_normalizes():
    def handler(model, inputs):
        # Unnormalized [3,4,0,0] → norm 5 → [0.6, 0.8, 0, 0].
        return _FakeResp([_FakeItem(0, [3.0, 4.0, 0.0, 0.0])])

    enc = _encoder_with(handler, dimension=4)
    out = enc.encode(["only"])
    assert out.dtype == np.float32
    assert np.allclose(out[0], [0.6, 0.8, 0.0, 0.0], atol=1e-6)
    assert math.isclose(float(np.linalg.norm(out[0])), 1.0, rel_tol=1e-6)


def test_empty_input_returns_zero_rows():
    def handler(model, inputs):  # pragma: no cover - should not be called
        raise AssertionError("create() should not be called for empty input")

    enc = _encoder_with(handler, dimension=4)
    out = enc.encode([])
    assert out.shape == (0, 4)


# --------------------------------------------------------------------------- #
# Failure handling — every fault raises EmbeddingsRuntimeError
# --------------------------------------------------------------------------- #
def test_server_unavailable_raises():
    def handler(model, inputs):
        raise ConnectionError("connection refused")

    enc = _encoder_with(handler, dimension=4)
    with pytest.raises(EmbeddingsRuntimeError):
        enc.encode(["x"])


def test_timeout_raises():
    def handler(model, inputs):
        raise TimeoutError("request timed out")

    enc = _encoder_with(handler, dimension=4)
    with pytest.raises(EmbeddingsRuntimeError):
        enc.encode(["x"])


def test_incomplete_response_raises():
    def handler(model, inputs):
        # Fewer vectors than inputs.
        return _FakeResp([_FakeItem(0, _onehot(0, 4))])

    enc = _encoder_with(handler, dimension=4)
    with pytest.raises(EmbeddingsRuntimeError):
        enc.encode(["a", "b"])


def test_missing_index_raises():
    def handler(model, inputs):
        return _FakeResp([_FakeItem(None, _onehot(0, 4))])

    enc = _encoder_with(handler, dimension=4)
    with pytest.raises(EmbeddingsRuntimeError):
        enc.encode(["a"])


def test_duplicate_indices_raise():
    def handler(model, inputs):
        return _FakeResp([_FakeItem(0, _onehot(0, 4)), _FakeItem(0, _onehot(1, 4))])

    enc = _encoder_with(handler, dimension=4)
    with pytest.raises(EmbeddingsRuntimeError):
        enc.encode(["a", "b"])


def test_wrong_dimension_raises():
    def handler(model, inputs):
        return _FakeResp([_FakeItem(0, [1.0, 0.0, 0.0])])  # 3-dim, expected 4

    enc = _encoder_with(handler, dimension=4)
    with pytest.raises(EmbeddingsRuntimeError):
        enc.encode(["a"])


def test_zero_vector_raises():
    def handler(model, inputs):
        return _FakeResp([_FakeItem(0, [0.0, 0.0, 0.0, 0.0])])

    enc = _encoder_with(handler, dimension=4)
    with pytest.raises(EmbeddingsRuntimeError):
        enc.encode(["a"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_raise(bad):
    def handler(model, inputs):
        return _FakeResp([_FakeItem(0, [bad, 1.0, 0.0, 0.0])])

    enc = _encoder_with(handler, dimension=4)
    with pytest.raises(EmbeddingsRuntimeError):
        enc.encode(["a"])


# --------------------------------------------------------------------------- #
# Opt-in local integration smoke test (real Q8 server on 127.0.0.1:8081)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("RUN_EMBEDDINGS_INTEGRATION") not in ("1", "true", "yes"),
    reason="set RUN_EMBEDDINGS_INTEGRATION=1 with the llama-server sidecar running",
)
def test_integration_smoke_real_server():
    from tools import pipeline_config as cfg

    enc = OpenAICompatibleEncoder(
        base_url=cfg.EMBEDDINGS_BASE_URL,
        model_name=cfg.EMBEDDINGS_MODEL_NAME,
        dimension=cfg.EMBEDDINGS_DIMENSION,
    )
    vecs = enc.encode([
        "the kitchen faucet is leaking water under the sink",
        "the faucet drips and needs a new washer",
        "the roof has a broken shingle on the exterior",
    ])
    assert vecs.shape == (3, cfg.EMBEDDINGS_DIMENSION)
    for row in vecs:
        assert math.isclose(float(np.linalg.norm(row)), 1.0, rel_tol=1e-3)
    related = float(vecs[0] @ vecs[1])
    unrelated = float(vecs[0] @ vecs[2])
    assert related > unrelated
