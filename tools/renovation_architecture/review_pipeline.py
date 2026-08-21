"""Session 2 orchestration: conditions -> evidence -> Terra -> dispositions.

Sequential per estimate unit. Each unit is either replayed from a
fingerprint-valid checkpoint (no reservation, no provider call, zero budget
debit) or bought fresh: reserve -> call -> settle -> checkpoint. Any typed
failure — including a pre-call budget denial — aborts the run without
publishing a partial result; checkpoints already written stay on disk so the
retry only re-buys what it never bought. The assembled result is
self-validated before it is returned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tools.renovation_architecture.checkpoints import (
    load_unit_checkpoint,
    save_unit_checkpoint,
    terra_checkpoint_dir,
    unit_checkpoint_path,
)
from tools.renovation_architecture.conditions import (
    ConditionDraft,
    build_observed_conditions,
)
from tools.renovation_architecture.contracts import (
    CONDITION_DISPOSITION_POLICY_VERSION,
    CONTRACTS_SCHEMA_VERSION,
    EvidenceFacts,
    TERRA_REVIEW_PROMPT_VERSION,
    TERRA_REVIEW_REASONING_EFFORT,
)
from tools.renovation_architecture.disposition import decide_disposition
from tools.renovation_architecture.evidence import (
    build_evidence_facts,
    build_photo_identity_index,
)
from tools.renovation_architecture.ids import (
    make_disposition_id,
    make_review_id,
    make_terra_call_id,
)
from tools.renovation_architecture.terra_review import (
    TerraUnitRequest,
    build_unit_request,
    call_terra_review,
    parse_unit_reviews,
)
from tools.renovation_architecture.usage_guard import (
    TerraDailyBudgetExceeded,
    TerraUsageLedger,
    estimate_reservation_tokens,
    external_reservation,
)
from tools.renovation_architecture.validators import (
    validate_condition_review_result,
)
from tools.scene_classifier_passes import PassExecutionError

_USAGE_KEYS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
    "metered_calls",
)
_TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
    "budget_debited_tokens",
)


def _usage_snapshot(vlm_client: Any) -> Dict[str, int]:
    stats = getattr(vlm_client, "usage_stats", None) or {}
    return {key: int(stats.get(key, 0) or 0) for key in _USAGE_KEYS}


def _review_unit_fresh(
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
        raise PassExecutionError(
            "terra_review", "request", str(exc),
            code="TerraDailyBudgetExceeded", provider="openai",
            model=terra_model,
        ) from exc

    before = _usage_snapshot(vlm_client)
    try:
        # This call is already reserved/settled against the Terra ledger
        # above; the Session 9 choke-point guard must not debit it again.
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
    parsed = parse_unit_reviews(
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
        "prompt_version": TERRA_REVIEW_PROMPT_VERSION,
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
            "verdict": parsed[condition_id]["verdict"],
            "rationale": parsed[condition_id]["rationale"],
            "model": terra_model,
            "prompt_version": TERRA_REVIEW_PROMPT_VERSION,
            "terra_call_id": call_id,
            "request_fingerprint": request.request_fingerprint,
            "provider": "openai",
        }
        for condition_id in request.condition_ids
    ]
    return call, reviews


def run_condition_review(
    *,
    runtime: Any,
    estimate_id: str,
    property_key: str,
    source_run_id: str,
    created_at: str,
    issues_flat: List[Dict[str, Any]],
    photos: Mapping[str, Any],
    property_metadata: Optional[Dict[str, Any]],
    photo_key_to_path: Mapping[str, Path],
    vlm_client: Any,
    api_key: str,
    artifacts_root: Optional[Path],
) -> Dict[str, Any]:
    """Produce the validated condition_review_complete result dict."""
    projection = runtime.projection
    observables = projection["observables"]
    terminal_routes = projection["terminal_routes"]

    drafts = build_observed_conditions(
        issues_flat=issues_flat or [],
        photos=photos or {},
        property_metadata=property_metadata,
        projection=projection,
        estimate_id=estimate_id,
    )
    all_photo_keys = sorted({
        ref["photo_key"] for draft in drafts for ref in draft.evidence_refs
    })
    identity_index = build_photo_identity_index(
        all_photo_keys, photo_key_to_path or {}
    )
    pairs: List[Tuple[ConditionDraft, EvidenceFacts]] = [
        (
            draft,
            build_evidence_facts(
                draft,
                identity_index=identity_index,
                observables=observables,
                estimate_id=estimate_id,
            ),
        )
        for draft in drafts
    ]

    units: Dict[str, List[Tuple[ConditionDraft, EvidenceFacts]]] = {}
    for draft, evidence in pairs:
        units.setdefault(draft.condition.estimate_unit_id, []).append(
            (draft, evidence)
        )

    terra_calls: List[Dict[str, Any]] = []
    reviews: List[Dict[str, Any]] = []
    ledger: Optional[TerraUsageLedger] = None
    for unit_id in sorted(units):
        if vlm_client is None or artifacts_root is None:
            raise PassExecutionError(
                "terra_review", "dependency",
                "shadow condition review needs a VLM client and an artifacts "
                "root before any Terra call",
                code="MissingSeamInput",
            )
        if ledger is None:
            from tools import pipeline_config as cfg
            from tools.renovation_architecture.usage_guard import (
                TERRA_DAILY_TOKEN_CEILING,
            )

            ledger = TerraUsageLedger(
                Path(artifacts_root),
                daily_ceiling=getattr(
                    cfg,
                    "RENOVATION_TERRA_DAILY_TOKEN_CEILING",
                    TERRA_DAILY_TOKEN_CEILING,
                ),
                usage_root_override=getattr(
                    cfg, "RENOVATION_TERRA_USAGE_ROOT", None
                ),
            )
        request = build_unit_request(
            estimate_unit_id=unit_id,
            unit_pairs=units[unit_id],
            observables=observables,
            photo_key_to_path=photo_key_to_path or {},
            identity_index=identity_index,
            projection_fingerprint=runtime.projection_fingerprint,
            model=runtime.terra_model,
            reasoning_effort=TERRA_REVIEW_REASONING_EFFORT,
            max_output_tokens=runtime.terra_max_output_tokens,
        )
        path = unit_checkpoint_path(
            terra_checkpoint_dir(
                Path(artifacts_root), property_key, source_run_id
            ),
            unit_id,
        )
        cached = load_unit_checkpoint(
            path,
            request_fingerprint=request.request_fingerprint,
            condition_ids=list(request.condition_ids),
        )
        if cached is not None:
            call = dict(cached["terra_call"])
            call["usage_source"] = "checkpoint"
            call["budget_debited_tokens"] = 0
            unit_reviews = [dict(review) for review in cached["reviews"]]
        else:
            call, unit_reviews = _review_unit_fresh(
                request=request,
                vlm_client=vlm_client,
                api_key=api_key,
                terra_model=runtime.terra_model,
                terra_max_output_tokens=runtime.terra_max_output_tokens,
                ledger=ledger,
                property_key=property_key,
                source_run_id=source_run_id,
                estimate_id=estimate_id,
            )
            save_unit_checkpoint(
                path,
                estimate_unit_id=unit_id,
                request_fingerprint=request.request_fingerprint,
                terra_call=call,
                reviews=unit_reviews,
                created_at=created_at,
            )
        terra_calls.append(call)
        reviews.extend(unit_reviews)

    reviews_by_condition = {review["condition_id"]: review for review in reviews}
    dispositions: List[Dict[str, Any]] = []
    for draft, evidence in pairs:
        condition = draft.condition
        review = reviews_by_condition[condition.condition_id]
        route = terminal_routes[condition.catalog_item_id]["route"]
        disposition, reason_code = decide_disposition(
            review["verdict"],
            route,
            evidence.distinct_view_count,
            evidence.min_photo_evidence_required,
        )
        dispositions.append({
            "disposition_id": make_disposition_id(
                estimate_id=estimate_id, condition_id=condition.condition_id
            ),
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "condition_id": condition.condition_id,
            "review_id": review["review_id"],
            "disposition": disposition,
            "reason_code": reason_code,
            "policy_version": CONDITION_DISPOSITION_POLICY_VERSION,
            "evidence_id": evidence.evidence_id,
            "terminal_route": route,
        })

    unit_usage = [
        {
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "estimate_unit_id": call["estimate_unit_id"],
            "call_ids": [call["call_id"]],
            **{name: call[name] for name in _TOKEN_FIELDS},
        }
        for call in terra_calls
    ]
    listing_usage = {
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "call_count": len(terra_calls),
        **{
            name: sum(usage[name] for usage in unit_usage)
            for name in _TOKEN_FIELDS
        },
    }
    result = {
        "observed_conditions": [draft.condition.to_dict() for draft, _ in pairs],
        "evidence_facts": [evidence.to_dict() for _, evidence in pairs],
        "condition_reviews": reviews,
        "condition_dispositions": dispositions,
        "terra_calls": terra_calls,
        "terra_unit_usage": unit_usage,
        "terra_listing_usage": listing_usage,
    }
    validation = validate_condition_review_result(result, estimate_id=estimate_id)
    if not validation.ok:
        raise PassExecutionError(
            "terra_review", "parse",
            "assembled condition-review result failed self-validation: "
            + "; ".join(validation.errors[:5]),
            code="TerraResultInvalid",
        )
    return result
