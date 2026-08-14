"""Bounded Terra condition review (terra_condition_review_v1).

One call per resolved estimate unit: each representative image is sent once
and Terra judges only the supplied observable claims against the supplied
photos, answering through a strict closed JSON schema with exactly one
bounded verdict per condition. Post-parse validation is the authority
regardless of the schema. Missing, duplicated, unknown, malformed, timeout,
or provider responses raise typed operational failures — they never become
`cannot_assess`, which is reserved for Terra honestly judging the imagery
insufficient.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from tools.comparison_common import sha256_canonical
from tools.renovation_architecture.conditions import ConditionDraft
from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    EVIDENCE_DEDUP_POLICY_VERSION,
    EvidenceFacts,
    REVIEW_RATIONALE_MAX_CHARS,
    REVIEW_VERDICTS,
    TERRA_REVIEW_PROMPT_VERSION,
)
from tools.scene_classifier_passes import PassExecutionError

OBSERVATION_MAX_CHARS = 300

TERRA_SYSTEM_PROMPT = """\
You are a strict photographic evidence reviewer for property renovation
conditions. For each supplied condition, judge only whether the claimed
condition appears visible in the attached photos.

Rules:
- Judge only the conditions listed in the request. Never add, merge, or
  invent conditions.
- verdict is exactly one of: supported, unsupported, cannot_assess.
  * supported: the claimed condition is clearly visible in at least one photo.
  * unsupported: the photos show the relevant area and the claimed condition
    is not present.
  * cannot_assess: the photos do not show the area well enough to judge.
- rationale: one short sentence describing what you can or cannot see.
- Never mention renovation work, packages, prices, quantities, or confidence
  percentages.
Return JSON matching the provided schema with exactly one review per supplied
condition_id.
"""


@dataclass(frozen=True)
class TerraUnitRequest:
    estimate_unit_id: str
    condition_ids: Tuple[str, ...]
    system_prompt: str
    user_prompt: str
    photo_keys: Tuple[str, ...]
    image_paths: Tuple[Path, ...]
    response_schema: Dict[str, Any]
    request_fingerprint: str
    request_bytes: int


def _review_failure(
    stage: str, message: str, *, code: str, model: str = ""
) -> PassExecutionError:
    return PassExecutionError(
        "terra_review", stage, message, code=code,
        provider="openai", model=model or None,
    )


def _claim_text(observable: Mapping[str, Any], catalog_item_id: str) -> str:
    """The observable claim shown to Terra: the catalog atomic_claim
    (subject + state when structured), else the scope, else the item id."""
    claim = observable.get("atomic_claim")
    if isinstance(claim, Mapping):
        text = " ".join(
            part for part in (
                str(claim.get("subject") or ""), str(claim.get("state") or "")
            ) if part
        )
        if text:
            return text
    elif claim:
        return str(claim)
    return str(observable.get("scope") or catalog_item_id)


def build_response_schema(condition_ids: List[str]) -> Dict[str, Any]:
    """Strict closed schema; the condition-id enum is per-call."""
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
                    "required": ["condition_id", "verdict", "rationale"],
                    "properties": {
                        "condition_id": {
                            "type": "string", "enum": sorted(condition_ids)
                        },
                        "verdict": {
                            "type": "string", "enum": sorted(REVIEW_VERDICTS)
                        },
                        "rationale": {"type": "string"},
                    },
                },
            },
        },
    }


def build_unit_request(
    *,
    estimate_unit_id: str,
    unit_pairs: List[Tuple[ConditionDraft, EvidenceFacts]],
    observables: Mapping[str, Any],
    photo_key_to_path: Mapping[str, Path],
    identity_index: Mapping[str, Any],
    projection_fingerprint: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> TerraUnitRequest:
    conditions_payload: List[Dict[str, Any]] = []
    photo_keys: List[str] = []
    for draft, evidence in unit_pairs:
        condition = draft.condition
        observable = observables.get(condition.catalog_item_id) or {}
        observations = sorted({
            str(ref.get("observation") or "")[:OBSERVATION_MAX_CHARS]
            for ref in draft.evidence_refs
            if ref.get("observation")
        })
        conditions_payload.append({
            "condition_id": condition.condition_id,
            "catalog_item_id": condition.catalog_item_id,
            "kind": condition.catalog_kind,
            "claim": _claim_text(observable, condition.catalog_item_id),
            "observations": observations,
            "photo_keys": list(evidence.representative_photo_keys),
        })
        photo_keys.extend(evidence.representative_photo_keys)
    ordered_photo_keys = sorted(set(photo_keys))
    condition_ids = [entry["condition_id"] for entry in conditions_payload]

    user_prompt = (
        "Attached photos, in order: "
        + ", ".join(
            f"{index + 1}. {key}" for index, key in enumerate(ordered_photo_keys)
        )
        + "\n\nConditions to review:\n"
        + json.dumps(
            {"estimate_unit": estimate_unit_id, "conditions": conditions_payload},
            indent=2, sort_keys=True,
        )
    )
    fingerprint = sha256_canonical({
        "contracts_schema_version": CONTRACTS_SCHEMA_VERSION,
        "projection_fingerprint": projection_fingerprint,
        "prompt_version": TERRA_REVIEW_PROMPT_VERSION,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "dedup_policy_version": EVIDENCE_DEDUP_POLICY_VERSION,
        "estimate_unit_id": estimate_unit_id,
        "conditions": conditions_payload,
        "image_hashes": {
            key: identity_index[key].exact_sha256 for key in ordered_photo_keys
        },
    })
    return TerraUnitRequest(
        estimate_unit_id=estimate_unit_id,
        condition_ids=tuple(condition_ids),
        system_prompt=TERRA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        photo_keys=tuple(ordered_photo_keys),
        image_paths=tuple(
            Path(photo_key_to_path[key]) for key in ordered_photo_keys
        ),
        response_schema=build_response_schema(condition_ids),
        request_fingerprint=fingerprint,
        request_bytes=len(
            (TERRA_SYSTEM_PROMPT + user_prompt).encode("utf-8")
        ),
    )


def call_terra_review(
    vlm_client: Any,
    request: TerraUnitRequest,
    *,
    model: str,
    api_key: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> str:
    """One provider call. Live exceptions become typed request-stage failures."""
    try:
        return vlm_client.analyze_images_sync(
            image_paths=list(request.image_paths),
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            model=model,
            api_key=api_key or None,
            provider="openai",
            max_tokens=int(max_output_tokens),
            response_json_schema=request.response_schema,
            response_schema_name="terra_condition_review",
            reasoning_effort=reasoning_effort,
            analysis_pass="Terra condition review",
        )
    except PassExecutionError:
        raise
    except Exception as exc:
        raise _review_failure(
            "request",
            f"Terra condition review call failed for estimate unit "
            f"{request.estimate_unit_id!r}: {exc}",
            code=type(exc).__name__,
            model=model,
        ) from exc


def parse_unit_reviews(
    raw_text: str, *, condition_ids: Tuple[str, ...], model: str = ""
) -> Dict[str, Dict[str, str]]:
    """Enforce the closed contract: exactly one bounded verdict per supplied
    condition, nothing else. Returns {condition_id: {verdict, rationale}}."""
    def contract(message: str) -> PassExecutionError:
        return _review_failure(
            "parse", message, code="TerraReviewContract", model=model
        )

    try:
        payload = json.loads(raw_text)
    except ValueError as exc:
        raise _review_failure(
            "parse", f"Terra response is not valid JSON: {exc}",
            code="JSONDecodeError", model=model,
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"reviews"}:
        raise contract(
            "Terra response must be an object with exactly one 'reviews' key"
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
        extra = set(entry) - {"condition_id", "verdict", "rationale"}
        if extra:
            raise contract(
                f"{where} carries fields outside the closed review contract: "
                f"{sorted(extra)} — Terra may not emit work, package, price, "
                "quantity, or confidence content"
            )
        condition_id = entry.get("condition_id")
        if condition_id not in expected:
            raise contract(f"{where} names unknown condition {condition_id!r}")
        if condition_id in parsed:
            raise contract(f"{where} duplicates condition {condition_id!r}")
        verdict = entry.get("verdict")
        if verdict not in REVIEW_VERDICTS:
            raise contract(f"{where} verdict {verdict!r} is not a bounded verdict")
        rationale = entry.get("rationale")
        if not isinstance(rationale, str):
            raise contract(f"{where} rationale must be a string")
        parsed[condition_id] = {
            "verdict": verdict,
            "rationale": rationale[:REVIEW_RATIONALE_MAX_CHARS],
        }
    missing = expected - set(parsed)
    if missing:
        raise contract(
            f"Terra returned no review for {len(missing)} supplied "
            f"condition(s): {sorted(missing)}"
        )
    return parsed
