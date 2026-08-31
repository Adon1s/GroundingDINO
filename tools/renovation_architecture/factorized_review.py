"""Experimental factorized condition review (terra_factorized_review_v1).

Shadow measurement only. Terra's `terra_review.py` stays authoritative and
untouched; this module asks the SAME model about the SAME unit requests through
a different question structure, so a replay harness can measure whether the
factorization sees errors the single verdict cannot.

Terra reduces every condition to supported / unsupported / cannot_assess. That
vocabulary cannot separate a real-and-correctly-named condition from a real one
the catalog names wrongly, or from one too minor to stand as its own work item —
and the v1.1 review puts ~76% of the billed-error mass in exactly those two
blind spots. Here the model answers three bounded questions instead
(`visible`, `claim_accurate_as_written`, `material_enough_for_work`) and the
class is derived from them in code by `derive_class`, never asked for: the
per-axis vocabulary stays small and closed, agreement can be measured per axis,
and rare combinations fall out of the cross-product.

Scoped deviation from Terra's contract: `parse_unit_reviews` forbids work
content outright, and question 3 IS a work-materiality judgment. That is the
structural reason this cannot be a Terra prompt-version bump and needs its own
prompt version, schema, and parser. Every other prohibition (packages, prices,
quantities, confidence) is kept.

BOUNDARY: nothing in the production pipeline imports this module, and it is not
re-exported from __init__.py. Its only callers are the replay harness under
--factorized and the tests. Design + gates:
docs/HANDOFF_factorized_verifier_replay.md.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Tuple

from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    REVIEW_RATIONALE_MAX_CHARS,
    TERRA_REVIEW_REASONING_EFFORT,
)
from tools.renovation_architecture.ids import make_review_id, make_terra_call_id
from tools.renovation_architecture.terra_review import (
    TerraUnitRequest,
    call_terra_review,
)
from tools.renovation_architecture.usage_guard import (
    TerraDailyBudgetExceeded,
    TerraUsageLedger,
    estimate_reservation_tokens,
    external_reservation,
)
from tools.scene_classifier_passes import PassExecutionError

FACTORIZED_PROMPT_VERSION = "terra_factorized_review_v1"
FACTOR_VALUES = frozenset({"yes", "no", "unclear"})
# Order matters: it is the reporting order and the schema's `required` order.
FACTOR_KEYS = (
    "visible", "claim_accurate_as_written", "material_enough_for_work",
)
REVIEW_KEYS = ("condition_id",) + FACTOR_KEYS + (
    "observed_description", "rationale",
)
OBSERVED_DESCRIPTION_MAX_CHARS = 200

# Data mirror of tools/label_schema.py::ADJUDICATION_KEYS values, so the
# derived class joins the v1.1 human labels with no translation table. The
# label vocabulary also carries `wrong_object_or_place`, which its own
# CLAIM_AXIS folds to `absent`; the derivation emits the folded class directly.
# tests/test_factorized_review.py pins this against label_schema.
DERIVED_CLASSES = (
    "exact_and_warranted",
    "misnamed_but_warranted",
    "exact_but_trivial",
    "misnamed_and_trivial",
    "absent",
    "inconclusive",
)

# (claim_accurate_as_written, material_enough_for_work) -> class, read only
# once `visible` is yes and neither factor is unclear. The table is the whole
# policy — no per-item special cases.
_VISIBLE_CLASS_TABLE = {
    ("yes", "yes"): "exact_and_warranted",
    ("yes", "no"): "exact_but_trivial",
    ("no", "yes"): "misnamed_but_warranted",
    ("no", "no"): "misnamed_and_trivial",
}

FACTORIZED_SYSTEM_PROMPT = """\
You are a strict photographic evidence reviewer for property renovation
conditions. For each supplied condition, answer three separate questions
about the attached photos. Do not collapse them into one judgment.

Rules:
- Judge only the conditions listed in the request. Never add, merge, or
  invent conditions.
- Answer each question with exactly one of: yes, no, unclear.

1. visible - is the specific thing the claim describes present in the
   photos, on the object and in the room the claim names?
   * yes: it is there, on the named object and in the named room.
   * no: the photos show the relevant area and it is not there, or what is
     there is on a different object or in a different room than the claim
     names.
   * unclear: the photos do not show the area well enough to judge.

2. claim_accurate_as_written - read the claim text literally. Does every
   part of it hold for what you can see?
   * yes: every assertion in the claim is true of what is visible.
   * no: something real is there, but the claim names it wrongly - wrong
     mechanism (staining described as scuffing), wrong material, wrong
     object, or a claim joining two assertions where only one holds.
   * unclear: you can see the area but cannot tell whether the wording fits.
   Judge wording only. A claim can be worded wrongly and still sit on a
   real, significant problem.

3. material_enough_for_work - judge what you can actually SEE, not what the
   claim says. Would a contractor write this up as its own line item?
   * yes: it needs its own repair, replacement, or refinishing line.
   * no: it is the minor wear a routine turnover clean and touch-up paint
     absorbs - faint marks, light scuffing, ordinary aging without damage.
   * unclear: it is visible, but these photos cannot show its extent.

- When visible is `no`, answer the other two questions `unclear`.
- observed_description: when claim_accurate_as_written is `no`, one short
  phrase naming what you actually see instead. Otherwise the empty string.
  Describe only - never name a replacement claim, a repair, or a product.
- rationale: one short sentence describing what you can or cannot see.
- Never mention packages, prices, quantities, or confidence percentages.
Return JSON matching the provided schema with exactly one review per
supplied condition_id.
"""


def _factorized_failure(
    stage: str, message: str, *, code: str, model: str = ""
) -> PassExecutionError:
    return PassExecutionError(
        "factorized_review", stage, message, code=code,
        provider="openai", model=model or None,
    )


def build_factorized_response_schema(condition_ids: List[str]) -> Dict[str, Any]:
    """Strict closed schema; the condition-id enum is per-call."""
    factor_property = {"type": "string", "enum": sorted(FACTOR_VALUES)}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reviews"],
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(REVIEW_KEYS),
                    "properties": {
                        "condition_id": {
                            "type": "string", "enum": sorted(condition_ids)
                        },
                        **{key: dict(factor_property) for key in FACTOR_KEYS},
                        "observed_description": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                },
            },
        },
    }


def derive_class(factors: Mapping[str, str]) -> str:
    """Three bounded factors -> one bounded class. Pure, no I/O, no confidence.

    `visible` is decisive on its own: nothing absent can be misnamed, and
    nothing unseen can be judged trivial. Any remaining `unclear` yields
    `inconclusive` rather than guessing, which also makes the inconclusive
    rate a health metric.
    """
    values = {}
    for key in FACTOR_KEYS:
        value = factors.get(key)
        if value not in FACTOR_VALUES:
            raise ValueError(f"factor {key!r} has non-bounded value {value!r}")
        values[key] = value
    if values["visible"] == "no":
        return "absent"
    if values["visible"] == "unclear":
        return "inconclusive"
    key = (values["claim_accurate_as_written"], values["material_enough_for_work"])
    return _VISIBLE_CLASS_TABLE.get(key, "inconclusive")


def parse_factorized_reviews(
    raw_text: str, *, condition_ids: Tuple[str, ...], model: str = ""
) -> Dict[str, Dict[str, str]]:
    """Enforce the closed contract: exactly one bounded answer set per supplied
    condition, nothing else. Mirrors terra_review.parse_unit_reviews.

    Strict on the bounded vocabulary, lenient on free text. An empty
    `observed_description` where accuracy is `no` is NOT a parse failure:
    parsing happens after settle, so failing the unit would burn the tokens
    without measuring anything, and the factors are what the experiment
    measures. The scorer counts those as prompt adherence instead.
    """
    def contract(message: str) -> PassExecutionError:
        return _factorized_failure(
            "parse", message, code="FactorizedReviewContract", model=model
        )

    try:
        payload = json.loads(raw_text)
    except ValueError as exc:
        raise _factorized_failure(
            "parse", f"Factorized response is not valid JSON: {exc}",
            code="JSONDecodeError", model=model,
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"reviews"}:
        raise contract(
            "Factorized response must be an object with exactly one "
            "'reviews' key"
        )
    reviews = payload["reviews"]
    if not isinstance(reviews, list):
        raise contract("'reviews' must be an array")
    expected = set(condition_ids)
    parsed: Dict[str, Dict[str, str]] = {}
    for index, entry in enumerate(reviews):
        where = f"reviews[{index}]"
        if not isinstance(entry, dict):
            raise contract(f"{where} must be an object")
        extra = set(entry) - set(REVIEW_KEYS)
        if extra:
            raise contract(
                f"{where} carries fields outside the closed factorized "
                f"contract: {sorted(extra)} — the reviewer may not emit "
                "packages, prices, quantities, or confidence content"
            )
        condition_id = entry.get("condition_id")
        if condition_id not in expected:
            raise contract(f"{where} names unknown condition {condition_id!r}")
        if condition_id in parsed:
            raise contract(f"{where} duplicates condition {condition_id!r}")
        factors = {}
        for key in FACTOR_KEYS:
            value = entry.get(key)
            if value not in FACTOR_VALUES:
                raise contract(
                    f"{where} {key} {value!r} is not a bounded factor value"
                )
            factors[key] = value
        rationale = entry.get("rationale")
        if not isinstance(rationale, str):
            raise contract(f"{where} rationale must be a string")
        description = entry.get("observed_description")
        if not isinstance(description, str):
            raise contract(f"{where} observed_description must be a string")
        parsed[condition_id] = {
            **factors,
            "derived_class": derive_class(factors),
            "observed_description": description[:OBSERVED_DESCRIPTION_MAX_CHARS],
            "rationale": rationale[:REVIEW_RATIONALE_MAX_CHARS],
        }
    missing = expected - set(parsed)
    if missing:
        raise contract(
            f"Factorized review returned no answer for {len(missing)} supplied "
            f"condition(s): {sorted(missing)}"
        )
    return parsed


_USAGE_KEYS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
    "metered_calls",
)


def _usage_snapshot(vlm_client: Any) -> Dict[str, int]:
    stats = getattr(vlm_client, "usage_stats", None) or {}
    return {key: int(stats.get(key, 0) or 0) for key in _USAGE_KEYS}


def factorized_unit_fresh(
    *,
    request: TerraUnitRequest,
    vlm_client: Any,
    api_key: str,
    terra_model: str,
    terra_max_output_tokens: int,
    ledger: TerraUsageLedger,
    property_key: str,
    source_run_id: str,
    estimate_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One factorized unit call. Mirrors review_pipeline._review_unit_fresh:75
    exactly — same reserve -> call -> settle -> parse ordering, same ledger
    semantics — so the harness cannot double-debit and a failed call keeps its
    conservative reservation.
    """
    reservation_tokens = estimate_reservation_tokens(
        max_output_tokens=terra_max_output_tokens,
        request_bytes=request.request_bytes,
        image_count=len(request.image_paths),
    )
    try:
        reservation_id = ledger.reserve(
            property_key=property_key,
            source_run_id=source_run_id,
            estimate_unit_id=request.estimate_unit_id,
            request_fingerprint=request.request_fingerprint,
            tokens=reservation_tokens,
        )
    except TerraDailyBudgetExceeded as exc:
        raise _factorized_failure(
            "request", str(exc), code="TerraDailyBudgetExceeded",
            model=terra_model,
        ) from exc

    before = _usage_snapshot(vlm_client)
    try:
        # Already reserved/settled against the Terra ledger here; the Session 9
        # choke-point guard must not debit it again.
        with external_reservation():
            raw_text = call_terra_review(
                vlm_client, request,
                model=terra_model, api_key=api_key,
                max_output_tokens=terra_max_output_tokens,
                reasoning_effort=TERRA_REVIEW_REASONING_EFFORT,
            )
    except PassExecutionError:
        # The provider may or may not have consumed tokens; keeping the
        # conservative reservation debited is the honest choice.
        ledger.settle(reservation_id, provider_total_tokens=None)
        raise
    after = _usage_snapshot(vlm_client)
    delta = {key: after[key] - before[key] for key in _USAGE_KEYS}
    metered = delta["metered_calls"] > 0
    debit = ledger.settle(
        reservation_id,
        provider_total_tokens=delta["total_tokens"] if metered else None,
    )
    # Settle before parsing: the tokens are spent whether or not the response
    # honors the contract.
    parsed = parse_factorized_reviews(
        raw_text, condition_ids=request.condition_ids, model=terra_model
    )
    call_id = make_terra_call_id(
        estimate_id=estimate_id,
        estimate_unit_id=request.estimate_unit_id,
        request_fingerprint=request.request_fingerprint,
    )
    call = {
        "call_id": call_id,
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "estimate_unit_id": request.estimate_unit_id,
        "condition_ids": sorted(request.condition_ids),
        "request_fingerprint": request.request_fingerprint,
        "provider": "openai",
        "model": terra_model,
        "prompt_version": FACTORIZED_PROMPT_VERSION,
        "usage_source": "provider",
        "input_tokens": delta["input_tokens"] if metered else 0,
        "cached_input_tokens": delta["cached_input_tokens"] if metered else 0,
        "output_tokens": delta["output_tokens"] if metered else 0,
        "total_tokens": delta["total_tokens"] if metered else 0,
        "budget_debited_tokens": debit,
    }
    reviews = [
        {
            "review_id": make_review_id(
                estimate_id=estimate_id, condition_id=condition_id
            ),
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "condition_id": condition_id,
            **{key: parsed[condition_id][key] for key in FACTOR_KEYS},
            "derived_class": parsed[condition_id]["derived_class"],
            "observed_description": parsed[condition_id]["observed_description"],
            "rationale": parsed[condition_id]["rationale"],
            "model": terra_model,
            "prompt_version": FACTORIZED_PROMPT_VERSION,
            "terra_call_id": call_id,
            "request_fingerprint": request.request_fingerprint,
            "provider": "openai",
        }
        for condition_id in request.condition_ids
    ]
    return call, reviews
