"""
Failure classification contract.

Two things are under test and they matter for different reasons.

Categorization decides whether the TS worker retries a run, sends it to review,
or pauses the entire queue — so every category needs at least one exception that
provably lands on it, and the quota-vs-ratelimit split needs its own coverage
because both arrive as HTTP 429 and only one is worth retrying.

The fingerprint literals pin the hash formula. `tests/analysis/failureTaxonomy.test.ts`
in renointel-dev asserts the *same strings*, which is what keeps the two
implementations honest without either repo importing the other. If you change the
formula, both files fail together — that is the point. Do not regenerate these
values from the implementation.
"""
import json

import pytest

from tools.failure_taxonomy import (
    ANALYZER_FAILURE_CATEGORIES,
    DEFAULT_CATEGORY,
    FailureDescriptor,
    classify_failure,
    descriptor_from_error_text,
)
from tools.scene_classifier_passes import (
    Pass2fInvalidResponseError,
    PassExecutionError,
)


# ── helpers ──────────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, headers=None):
        self.headers = headers or {}


def _openai_error(name, *, body=None, headers=None, message="boom"):
    """
    Build a stand-in for an openai SDK error.

    Classification walks the MRO by *name*, so a dynamically-created class with
    the right name exercises the same path a real SDK error would — without
    pinning the test to the SDK's constructor signature, which has churned
    across releases.
    """
    klass = type(name, (Exception,), {})
    exc = klass(message)
    if body is not None:
        exc.body = body
    if headers is not None:
        exc.response = _FakeResponse(headers)
    return exc


# ── categorization ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "exc,expected",
    [
        (_openai_error("AuthenticationError"), "auth"),
        (_openai_error("PermissionDeniedError"), "auth"),
        (_openai_error("NotFoundError"), "model_config"),
        (_openai_error("BadRequestError"), "model_config"),
        (_openai_error("APITimeoutError"), "timeout"),
        (_openai_error("APIConnectionError"), "network"),
        (_openai_error("InternalServerError"), "provider"),
        (_openai_error("APIResponseValidationError"), "malformed_response"),
        (Pass2fInvalidResponseError("no decision"), "malformed_response"),
        (json.JSONDecodeError("bad", "{", 0), "parse"),
        (FileNotFoundError("missing.jpg"), "input"),
        (ConnectionResetError("peer reset"), "network"),
    ],
)
def test_each_category_has_a_reachable_exception(exc, expected):
    assert classify_failure(exc).category == expected


def test_every_produced_category_is_declared():
    """Nothing may emit a category the TS side has never heard of."""
    exceptions = [
        _openai_error("AuthenticationError"),
        _openai_error("RateLimitError", body={"code": "insufficient_quota"}),
        _openai_error("RateLimitError"),
        _openai_error("NotFoundError"),
        _openai_error("APITimeoutError"),
        _openai_error("APIConnectionError"),
        _openai_error("InternalServerError"),
        json.JSONDecodeError("bad", "{", 0),
        FileNotFoundError("missing.jpg"),
        Pass2fInvalidResponseError("no decision"),
        RuntimeError("something nobody anticipated"),
    ]
    for exc in exceptions:
        assert classify_failure(exc).category in ANALYZER_FAILURE_CATEGORIES


def test_unrecognized_exception_falls_back_to_retryable_default():
    """
    An unknown failure must not pause the queue.

    Pausing on a string we cannot interpret would let one odd exception stop all
    analysis; the safe direction is spending this run's one retry.
    """
    descriptor = classify_failure(RuntimeError("entirely novel"))
    assert descriptor.category == DEFAULT_CATEGORY == "provider"
    assert descriptor.code == "RuntimeError"


# ── the 429 split ────────────────────────────────────────────────────────────

def test_exhausted_balance_is_quota_not_ratelimit():
    """Both are 429. Only the body separates 'wait a bit' from 'you are out of money'."""
    exc = _openai_error("RateLimitError", body={"code": "insufficient_quota"})
    assert classify_failure(exc).category == "quota"


def test_quota_detected_from_message_when_body_is_absent():
    exc = _openai_error(
        "RateLimitError", message="You exceeded your current quota, please check your plan"
    )
    assert classify_failure(exc).category == "quota"


def test_real_rate_limit_stays_transient():
    exc = _openai_error("RateLimitError", message="Rate limit reached for gpt-5.4-mini")
    assert classify_failure(exc).category == "ratelimit"


def test_retry_after_header_is_captured():
    exc = _openai_error("RateLimitError", headers={"retry-after": "20"})
    assert classify_failure(exc).retry_after_sec == 20.0


# ── PassExecutionError unwrapping ────────────────────────────────────────────

def test_pass_execution_error_is_unwrapped_not_classified_as_itself():
    """
    PassExecutionError is an envelope. Its captured `code` is the only trace of
    the real exception left, so it must win over the envelope's own class name.
    """
    exc = PassExecutionError(
        "2a", "request", "upstream refused",
        code="AuthenticationError", provider="openai", model="gpt-5.4-mini",
    )
    descriptor = classify_failure(exc)
    assert descriptor.category == "auth"
    assert descriptor.pass_key == "2a"
    assert descriptor.stage == "request"
    assert descriptor.provider == "openai"
    assert descriptor.model == "gpt-5.4-mini"


@pytest.mark.parametrize(
    "stage,expected",
    [("dependency", "dependency"), ("parse", "parse"), ("response", "malformed_response")],
)
def test_stage_drives_category_when_code_is_unrecognized(stage, expected):
    exc = PassExecutionError("2d", stage, "boom", code="SomeVendorSpecificError")
    assert classify_failure(exc).category == expected


def test_unrecognized_request_stage_falls_back_to_provider():
    exc = PassExecutionError("1a", "request", "boom", code="SomeVendorSpecificError")
    assert classify_failure(exc).category == "provider"


def test_embeddings_failure_is_a_dependency():
    """
    Pass 2d without embeddings is a config/dependency problem, not a flaky call —
    the TS policy table pauses the queue on it rather than burning a retry.
    """
    exc = PassExecutionError(
        "2d", "dependency", "embeddings unavailable", code="EmbeddingsRuntimeError"
    )
    assert classify_failure(exc).category == "dependency"


def test_pass_2f_model_unavailable_is_a_config_failure():
    """Pass 2f is OpenAI-only; a missing model must never look transient."""
    exc = PassExecutionError(
        "2f", "dependency", "no OpenAI model", code="Pass2fModelUnavailable"
    )
    assert classify_failure(exc).category == "model_config"


# ── fingerprint ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "descriptor,expected",
    [
        (
            FailureDescriptor(
                category="quota", code="RateLimitError", message="x",
                pass_key="1a", provider="openai", model="gpt-5.4-mini",
            ),
            # sha1("quota|openai|gpt-5.4-mini|1a|RateLimitError")[:16]
            "f22a37a80dc6c544",
        ),
        (
            FailureDescriptor(category="auth", code="AuthenticationError", message="x"),
            # sha1("auth|-|-|-|AuthenticationError")[:16] — absent fields become "-"
            "1e3b628daae1d0ea",
        ),
    ],
)
def test_fingerprint_literals(descriptor, expected):
    """
    Mirrored verbatim in renointel-dev tests/analysis/failureTaxonomy.test.ts.
    Regenerating these from the implementation defeats the whole check.
    """
    assert descriptor.fingerprint() == expected


def test_fingerprint_ignores_run_identity():
    """
    Two properties failing the same way must share a fingerprint — that identity
    is exactly what the circuit breaker counts.
    """
    a = classify_failure(
        PassExecutionError("1a", "request", "quota on property A",
                           code="RateLimitError", provider="openai", model="gpt-5.4-mini")
    )
    b = classify_failure(
        PassExecutionError("1a", "request", "quota on property B",
                           code="RateLimitError", provider="openai", model="gpt-5.4-mini")
    )
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_separates_different_models():
    a = FailureDescriptor(category="ratelimit", code="RateLimitError", message="x",
                          provider="openai", model="gpt-5.4-mini")
    b = FailureDescriptor(category="ratelimit", code="RateLimitError", message="x",
                          provider="openai", model="gpt-5.4")
    assert a.fingerprint() != b.fingerprint()


# ── wire shape ───────────────────────────────────────────────────────────────

def test_to_wire_uses_the_pass_key_the_ts_side_reads():
    descriptor = FailureDescriptor(
        category="quota", code="RateLimitError", message="out of credit",
        pass_key="2f", stage="request", provider="openai", model="gpt-5.4",
    )
    wire = descriptor.to_wire()
    assert wire["pass"] == "2f"          # not "pass_key" — `pass` is a TS-side field name
    assert wire["category"] == "quota"
    assert wire["fingerprint"] == descriptor.fingerprint()
    assert set(wire) == {
        "category", "pass", "stage", "code", "provider",
        "model", "message", "retry_after_sec", "fingerprint",
    }


def test_wire_payload_is_json_serializable():
    """It goes out over NDJSON on stdout; a non-serializable field would kill the job."""
    wire = classify_failure(_openai_error("RateLimitError")).to_wire()
    assert json.loads(json.dumps(wire))["category"] == "ratelimit"


def test_string_only_failures_stay_retryable():
    descriptor = descriptor_from_error_text("photo 3 failed somehow", pass_key="2b")
    assert descriptor.category == DEFAULT_CATEGORY
    assert descriptor.code == "UNCLASSIFIED"
    assert descriptor.pass_key == "2b"


def test_long_messages_are_truncated():
    descriptor = classify_failure(RuntimeError("x" * 5000))
    assert len(descriptor.message) == 300
