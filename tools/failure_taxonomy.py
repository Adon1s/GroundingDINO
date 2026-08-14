"""
Structured failure classification for the analyzer.

Exists so a failure can travel to the TypeScript worker as a *decision* rather
than a string. The worker needs three things from a failure that `str(exc)` can
never carry reliably: is it worth retrying, will it hit the next job too, and do
two failures in different properties share a root cause.

The category list here is the analyzer half of a contract shared with
`lib/analysis/failure/taxonomy.ts` in the renointel-dev repo. That repo adds two
worker-local categories (`output_contract`, `persistence`) that Python has no way
to reach. Parity between the two lists is asserted by
`tests/analysis/failureTaxonomyParity.test.ts` on the TS side, and the fingerprint
formula is pinned from both ends by literal-string tests.

Classification is deliberately name-based (walking the exception MRO) rather than
purely `isinstance`. Two reasons: `PassExecutionError` captures only
`code=type(exc).__name__` and discards the original exception, so the only thing
left to classify by at the boundary is a class-name string; and it keeps this
module importable and testable without the openai SDK present.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# The contract
# ─────────────────────────────────────────────────────────────────────────────

ANALYZER_FAILURE_CATEGORIES: Tuple[str, ...] = (
    "quota",
    "auth",
    "model_config",
    "dependency",
    "ratelimit",
    "timeout",
    "network",
    "provider",
    "malformed_response",
    "parse",
    "input",
)

#: Fallback when nothing matches. Retryable and non-systemic — the safe direction
#: is one more attempt on this run, not pausing the global queue.
DEFAULT_CATEGORY = "provider"

# PassExecutionError.stage -> category, used when the stage is all we can trust.
# `request` is deliberately absent: it means "the provider call failed" without
# saying how, so it must be refined by the captured exception class name.
_STAGE_CATEGORY = {
    "dependency": "dependency",
    "parse": "parse",
    "response": "malformed_response",
}

# Exception class name -> category. Checked against every name in the MRO, so a
# subclass of a known error classifies correctly without being listed.
_CODE_CATEGORY = {
    # openai SDK
    "AuthenticationError": "auth",
    "PermissionDeniedError": "auth",
    "NotFoundError": "model_config",
    "BadRequestError": "model_config",
    "UnprocessableEntityError": "model_config",
    "APITimeoutError": "timeout",
    "APIConnectionError": "network",
    "InternalServerError": "provider",
    "ConflictError": "provider",
    "APIResponseValidationError": "malformed_response",
    "LengthFinishReasonError": "malformed_response",
    "ContentFilterFinishReasonError": "malformed_response",
    "APIStatusError": "provider",
    "APIError": "provider",
    "OpenAIError": "provider",
    # tools/vlm_client.py
    "OpenAIIncompleteResponse": "malformed_response",
    "OpenAIRefusal": "malformed_response",
    "OpenAIEmptyResponse": "malformed_response",
    # tools/artifact_writers.py — Pass 2f is OpenAI-only and must never fall back
    "Pass2fModelUnavailable": "model_config",
    # tools/scene_classifier_passes.py
    "Pass2fInvalidResponseError": "malformed_response",
    # tools/catalog_embeddings.py
    "EmbeddingsRuntimeError": "dependency",
    # tools/renovation_architecture/usage_guard.py — pre-call daily-budget
    # denial: systemic like an exhausted balance, retry today never works
    "TerraDailyBudgetExceeded": "quota",
    # stdlib
    "JSONDecodeError": "parse",
    "TimeoutError": "timeout",
    "ConnectionError": "network",
    "ConnectionResetError": "network",
    "FileNotFoundError": "input",
    "IsADirectoryError": "input",
    "UnidentifiedImageError": "input",
    "ValueError": "parse",
    "KeyError": "dependency",
    "LookupError": "dependency",
}

# `RateLimitError` is 429 for two very different reasons: a real rate limit
# (transient, retry works) or an exhausted balance (systemic, retry never works).
# Only the body distinguishes them, so it is handled outside _CODE_CATEGORY.
_QUOTA_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing_hard_limit_reached",
    "account_deactivated",
)


@dataclass(frozen=True)
class FailureDescriptor:
    """One classified failure, ready for the wire."""

    category: str
    code: str
    message: str
    pass_key: Optional[str] = None
    stage: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    retry_after_sec: Optional[float] = None

    def fingerprint(self) -> str:
        """
        Stable identity for "the same root cause".

        Deliberately excludes property key and run id: the whole point is letting
        the worker notice that two *different* properties failed the same way, so
        anything run-specific would defeat it. The formula is mirrored exactly in
        lib/analysis/failure/taxonomy.ts.
        """
        parts = "|".join(
            [
                self.category,
                self.provider or "-",
                self.model or "-",
                self.pass_key or "-",
                self.code or "-",
            ]
        )
        return hashlib.sha1(parts.encode("utf-8")).hexdigest()[:16]

    def to_wire(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "pass": self.pass_key,
            "stage": self.stage,
            "code": self.code,
            "provider": self.provider,
            "model": self.model,
            "message": self.message,
            "retry_after_sec": self.retry_after_sec,
            "fingerprint": self.fingerprint(),
        }


def _mro_names(exc: BaseException) -> Iterable[str]:
    for klass in type(exc).__mro__:
        yield klass.__name__


def _category_from_code(code: str) -> Optional[str]:
    """Category for a bare exception class name, or None if unrecognized."""
    return _CODE_CATEGORY.get(code)


def _looks_like_quota(exc: BaseException) -> bool:
    """Whether a 429 is an exhausted balance rather than a real rate limit."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        code = str(body.get("code") or (body.get("error") or {}).get("code") or "")
        if code in ("insufficient_quota", "billing_hard_limit_reached"):
            return True
    text = str(exc).lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def _retry_after(exc: BaseException) -> Optional[float]:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        try:
            raw = headers.get(key)
        except Exception:
            raw = None
        if raw:
            try:
                return float(str(raw).rstrip("s"))
            except (TypeError, ValueError):
                continue
    return None


def _categorize(exc: BaseException) -> str:
    """Category for a live exception, by MRO name with a 429 refinement."""
    names = list(_mro_names(exc))
    if "RateLimitError" in names:
        return "quota" if _looks_like_quota(exc) else "ratelimit"
    for name in names:
        category = _category_from_code(name)
        if category is not None:
            return category
    return DEFAULT_CATEGORY


def classify_failure(
    exc: BaseException,
    *,
    pass_key: Optional[str] = None,
    stage: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> FailureDescriptor:
    """
    Turn an exception into a `FailureDescriptor`.

    The single place exceptions become categories. `PassExecutionError` is
    unwrapped rather than classified as itself: it is a envelope carrying the
    pass/stage/code/provider/model of the real failure, so those win over the
    keyword arguments here.
    """
    from tools.scene_classifier_passes import PassExecutionError  # local: avoids a cycle

    if isinstance(exc, PassExecutionError):
        code = exc.code or type(exc).__name__
        # The captured class name is more specific than the stage, so try it
        # first; fall back to the stage mapping, then to `request`'s default.
        category = _category_from_code(code)
        if category is None:
            category = _STAGE_CATEGORY.get(exc.stage, DEFAULT_CATEGORY)
        return FailureDescriptor(
            category=category,
            code=code,
            message=exc.message or str(exc),
            pass_key=exc.pass_key or pass_key,
            stage=exc.stage or stage,
            provider=exc.provider or provider,
            model=exc.model or model,
        )

    return FailureDescriptor(
        category=_categorize(exc),
        code=type(exc).__name__,
        message=str(exc)[:300],
        pass_key=pass_key,
        stage=stage,
        provider=provider,
        model=model,
        retry_after_sec=_retry_after(exc),
    )


def descriptor_from_error_text(
    message: str,
    *,
    pass_key: Optional[str] = None,
    stage: Optional[str] = None,
) -> FailureDescriptor:
    """
    Last-resort descriptor for a failure that only survives as a string.

    Used where an error crossed a boundary that discarded the exception (e.g. an
    `ImageResult.error` string). Categorized as the retryable default so a run
    with no usable classification still gets its one retry rather than either
    silently passing or pausing the whole queue.
    """
    return FailureDescriptor(
        category=DEFAULT_CATEGORY,
        code="UNCLASSIFIED",
        message=str(message)[:300],
        pass_key=pass_key,
        stage=stage,
    )


__all__ = [
    "ANALYZER_FAILURE_CATEGORIES",
    "DEFAULT_CATEGORY",
    "FailureDescriptor",
    "classify_failure",
    "descriptor_from_error_text",
]
