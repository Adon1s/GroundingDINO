"""Bounded Sol package review (sol_package_review_v1).

One text-only listing call: Sol judges only whether the supplied deterministic
package candidates are coherent groupings, answering through a strict closed
JSON schema with exactly one bounded decision per candidate. It receives the
candidates, immutable child-work snapshots, and separately labeled objective
evidence / Terra verdict / deterministic disposition summaries — never images,
never the catalog, and never a request to reassess condition truth.

Post-parse validation is the authority regardless of the schema. Missing,
duplicated, unknown, malformed, timeout, or provider responses raise typed
operational failures — they never become an ``uncertain`` decision, which is
reserved for Sol honestly judging a supplied grouping's coherence.

Budgeted (Session 8): every fresh call reserves against the 250k/day Sol
ledger (usage_guard.SolUsageLedger) before the provider call and settles to
provider truth before parsing, mirroring the Terra guard ordering; a denial
raises a typed operational failure (code SolDailyBudgetExceeded, category
quota), never a partial result. The usage delta is settled into the SolCall
record before parsing. A successfully parsed listing result is checkpointed
under its request fingerprint; reuse republishes the original decisions and
token numbers with usage_source="checkpoint", no provider call, and no
ledger debit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tools.comparison_common import sha256_canonical
from tools.renovation_architecture.checkpoints import (
    load_sol_checkpoint,
    save_sol_checkpoint,
    sol_checkpoint_path,
    terra_checkpoint_dir,
)
from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    PACKAGE_DECISIONS,
    REVIEW_RATIONALE_MAX_CHARS,
    SOL_REVIEW_PROMPT_VERSION,
    SOL_REVIEW_REASONING_EFFORT,
)
from tools.renovation_architecture.ids import (
    make_package_decision_id,
    make_sol_call_id,
)
from tools.renovation_architecture.usage_guard import (
    SOL_DAILY_TOKEN_CEILING,
    SolDailyBudgetExceeded,
    SolUsageLedger,
    estimate_sol_reservation_tokens,
    external_reservation,
)
from tools.renovation_architecture.validators import (
    package_review_snapshot_hashes,
    validate_package_review_result,
)
from tools.scene_classifier_passes import PassExecutionError

SOL_SYSTEM_PROMPT = """\
You are a strict renovation package reviewer. Deterministic policy has
already grouped reviewed work items into package candidates. For each
supplied candidate, judge only whether the proposed grouping is a coherent
renovation package for the visible breadth of work.

Rules:
- Judge only the candidates listed in the request. Never add, merge, or
  invent candidates, work items, package families, quantities, or prices.
- Never reassess whether a condition is real — condition truth was reviewed
  upstream and is final. The evidence, verdict, and disposition summaries
  are context for coherence only.
- decision is exactly one of: approve, reject, uncertain.
  * approve: the grouping is a coherent package as proposed.
  * reject: the grouping is not a coherent package; its work items should
    stay standalone.
  * uncertain: coherence cannot be judged from the supplied information.
- combine_with: only when two or more candidates you APPROVE clearly form
  one renovation scope, list the other candidate ids on each. Leave empty
  otherwise. A candidate may not both combine and split.
- split_groups: only when one candidate you reviewed clearly contains two
  or more independent scopes, partition its exact listed child work item
  ids into two or more non-empty groups. Never drop, add, or repeat a
  child. Leave empty otherwise.
- display_only candidates are informational aggregates: decide
  approve/reject/uncertain on their coherence, but never combine or split
  them.
- rationale: one short sentence about grouping coherence.
Return JSON matching the provided schema with exactly one decision per
supplied package_candidate_id.
"""

_DECISION_FIELDS = frozenset(
    {"package_candidate_id", "decision", "combine_with", "split_groups",
     "rationale"}
)


@dataclass(frozen=True)
class SolListingRequest:
    package_candidate_ids: Tuple[str, ...]
    system_prompt: str
    user_prompt: str
    response_schema: Dict[str, Any]
    request_fingerprint: str
    request_bytes: int


def _review_failure(
    stage: str, message: str, *, code: str, model: str = ""
) -> PassExecutionError:
    return PassExecutionError(
        "sol_review", stage, message, code=code,
        provider="openai", model=model or None,
    )


def build_response_schema(candidate_ids: List[str]) -> Dict[str, Any]:
    """Strict closed schema; the candidate-id enum is per-call."""
    id_enum = {"type": "string", "enum": sorted(candidate_ids)}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "package_candidate_id", "decision", "combine_with",
                        "split_groups", "rationale",
                    ],
                    "properties": {
                        "package_candidate_id": dict(id_enum),
                        "decision": {
                            "type": "string", "enum": sorted(PACKAGE_DECISIONS)
                        },
                        "combine_with": {
                            "type": "array", "items": dict(id_enum)
                        },
                        "split_groups": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "rationale": {"type": "string"},
                    },
                },
            },
        },
    }


def _condition_summary(
    condition_id: str,
    *,
    conditions: Mapping[str, Mapping[str, Any]],
    evidence_by_condition: Mapping[str, Mapping[str, Any]],
    reviews_by_condition: Mapping[str, Mapping[str, Any]],
    dispositions_by_condition: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """The three upstream layers, separately labeled, never blended."""
    condition = conditions[condition_id]
    evidence = evidence_by_condition[condition_id]
    review = reviews_by_condition[condition_id]
    disposition = dispositions_by_condition[condition_id]
    return {
        "condition_id": condition_id,
        "catalog_item_id": condition["catalog_item_id"],
        "objective_evidence": {
            "distinct_photo_count": evidence["distinct_photo_count"],
            "distinct_view_count": evidence["distinct_view_count"],
        },
        "terra_verdict": {
            "verdict": review["verdict"],
            "rationale": review["rationale"],
        },
        "deterministic_disposition": {
            "disposition": disposition["disposition"],
            "reason_code": disposition["reason_code"],
        },
    }


def build_listing_request(
    *,
    standalone_result: Mapping[str, Any],
    package_candidates: List[Dict[str, Any]],
    projection_fingerprint: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> SolListingRequest:
    conditions = {
        condition["condition_id"]: condition
        for condition in standalone_result["observed_conditions"]
    }
    evidence_by_condition = {
        evidence["condition_id"]: evidence
        for evidence in standalone_result["evidence_facts"]
    }
    reviews_by_condition = {
        review["condition_id"]: review
        for review in standalone_result["condition_reviews"]
    }
    dispositions_by_condition = {
        disposition["condition_id"]: disposition
        for disposition in standalone_result["condition_dispositions"]
    }
    work_items = {
        item["work_item_id"]: item for item in standalone_result["work_items"]
    }

    candidates_payload: List[Dict[str, Any]] = []
    for candidate in package_candidates:
        drivers = set(candidate["driver_work_item_ids"])
        children: List[Dict[str, Any]] = []
        for work_id in candidate["child_work_item_ids"]:
            work = work_items[work_id]
            children.append({
                "work_item_id": work_id,
                "role": "driver" if work_id in drivers else "support",
                "action_code": work["action_code"],
                "trade_bucket": work["trade_bucket"],
                "billable_unit_id": work["billable_unit_id"],
                "unit_count": work["unit_count"],
                "estimate_scope": work["estimate_scope"],
                "low": work["low"],
                "high": work["high"],
                "conditions": [
                    _condition_summary(
                        condition_id,
                        conditions=conditions,
                        evidence_by_condition=evidence_by_condition,
                        reviews_by_condition=reviews_by_condition,
                        dispositions_by_condition=dispositions_by_condition,
                    )
                    for condition_id in work["condition_ids"]
                ],
            })
        candidates_payload.append({
            "package_candidate_id": candidate["package_candidate_id"],
            "package_type": candidate["package_type"],
            "package_category": candidate["package_category"],
            "package_level": candidate["package_level"],
            "room": candidate["room"],
            "estimate_unit_id": candidate["estimate_unit_id"],
            "strength": candidate["strength"],
            "pricing_tier": candidate["pricing_tier"],
            "proposed_treatment": candidate["proposed_treatment"],
            "allowance_low": candidate["low"],
            "allowance_high": candidate["high"],
            "display_only": candidate["display_only"],
            "contributing_candidate_ids": list(
                candidate["contributing_candidate_ids"]
            ),
            "child_work_snapshots": children,
        })

    candidate_ids = [
        entry["package_candidate_id"] for entry in candidates_payload
    ]
    user_prompt = (
        "Package candidates to review:\n"
        + json.dumps(
            {"package_candidates": candidates_payload},
            indent=2, sort_keys=True,
        )
    )
    snapshots = package_review_snapshot_hashes({
        **standalone_result, "package_candidates": package_candidates,
    })
    fingerprint = sha256_canonical({
        "contracts_schema_version": CONTRACTS_SCHEMA_VERSION,
        "prompt_version": SOL_REVIEW_PROMPT_VERSION,
        "projection_fingerprint": projection_fingerprint,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "candidates": candidates_payload,
        "snapshots": snapshots,
    })
    return SolListingRequest(
        package_candidate_ids=tuple(candidate_ids),
        system_prompt=SOL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=build_response_schema(candidate_ids),
        request_fingerprint=fingerprint,
        request_bytes=len((SOL_SYSTEM_PROMPT + user_prompt).encode("utf-8")),
    )


def call_sol_review(
    vlm_client: Any,
    request: SolListingRequest,
    *,
    model: str,
    api_key: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> str:
    """One text-only provider call. Live exceptions become typed
    request-stage failures."""
    try:
        return vlm_client.analyze_text_sync(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            model=model,
            api_key=api_key or None,
            provider="openai",
            max_tokens=int(max_output_tokens),
            response_json_schema=request.response_schema,
            response_schema_name="sol_package_review",
            reasoning_effort=reasoning_effort,
            analysis_pass="Sol package review",
        )
    except PassExecutionError:
        raise
    except Exception as exc:
        raise _review_failure(
            "request",
            f"Sol package review call failed: {exc}",
            code=type(exc).__name__,
            model=model,
        ) from exc


def parse_listing_decisions(
    raw_text: str,
    *,
    candidate_ids: Tuple[str, ...],
    children_by_candidate: Mapping[str, Tuple[str, ...]],
    display_only_ids: frozenset,
    model: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Enforce the closed contract: exactly one bounded decision per supplied
    candidate, bounded combine/split, nothing else.

    Returns {candidate_id: {decision, combine_with, split_groups, rationale}}
    with combine/split ordering normalized (membership untouched)."""
    def contract(message: str) -> PassExecutionError:
        return _review_failure(
            "parse", message, code="SolReviewContract", model=model
        )

    try:
        payload = json.loads(raw_text)
    except ValueError as exc:
        raise _review_failure(
            "parse", f"Sol response is not valid JSON: {exc}",
            code="JSONDecodeError", model=model,
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"decisions"}:
        raise contract(
            "Sol response must be an object with exactly one 'decisions' key"
        )
    decisions = payload["decisions"]
    if not isinstance(decisions, list):
        raise contract("'decisions' must be an array")

    expected = set(candidate_ids)
    parsed: Dict[str, Dict[str, Any]] = {}
    for index, entry in enumerate(decisions):
        where = f"decisions[{index}]"
        if not isinstance(entry, dict):
            raise contract(f"{where} must be an object")
        extra = set(entry) - _DECISION_FIELDS
        if extra:
            raise contract(
                f"{where} carries fields outside the closed decision "
                f"contract: {sorted(extra)} — Sol may not emit work, price, "
                "quantity, action, or family content"
            )
        candidate_id = entry.get("package_candidate_id")
        if candidate_id not in expected:
            raise contract(f"{where} names unknown candidate {candidate_id!r}")
        if candidate_id in parsed:
            raise contract(f"{where} duplicates candidate {candidate_id!r}")
        decision = entry.get("decision")
        if decision not in PACKAGE_DECISIONS:
            raise contract(f"{where} decision {decision!r} is not a bounded decision")
        combine_with = entry.get("combine_with")
        if not isinstance(combine_with, list) or not all(
            isinstance(other, str) for other in combine_with
        ):
            raise contract(f"{where} combine_with must be a list of candidate ids")
        for other_id in combine_with:
            if other_id == candidate_id:
                raise contract(f"{where} combine_with cites itself")
            if other_id not in expected:
                raise contract(
                    f"{where} combine_with cites unknown candidate {other_id!r}"
                )
        split_groups = entry.get("split_groups")
        if not isinstance(split_groups, list) or not all(
            isinstance(group, list)
            and group
            and all(isinstance(child, str) and child for child in group)
            for group in split_groups
        ):
            raise contract(
                f"{where} split_groups must be a list of non-empty lists of "
                "work item ids"
            )
        rationale = entry.get("rationale")
        if not isinstance(rationale, str):
            raise contract(f"{where} rationale must be a string")
        parsed[candidate_id] = {
            "decision": decision,
            "combine_with": sorted(set(combine_with)),
            "split_groups": split_groups,
            "rationale": rationale[:REVIEW_RATIONALE_MAX_CHARS],
        }

    missing = expected - set(parsed)
    if missing:
        raise contract(
            f"Sol returned no decision for {len(missing)} supplied "
            f"candidate(s): {sorted(missing)}"
        )

    # Cross-decision semantics. Combine references are undirected edges among
    # approved candidates; split partitions a candidate's exact children; no
    # candidate participates in both treatments; display-only aggregates
    # participate in neither.
    combine_participants: Dict[str, str] = {}
    for candidate_id, entry in parsed.items():
        for other_id in entry["combine_with"]:
            for endpoint, via in ((candidate_id, candidate_id), (other_id, candidate_id)):
                combine_participants.setdefault(endpoint, via)
    for candidate_id, entry in parsed.items():
        edges = entry["combine_with"]
        groups = entry["split_groups"]
        if edges:
            if candidate_id in display_only_ids:
                raise contract(
                    f"display-only candidate {candidate_id!r} may not combine"
                )
            if parsed[candidate_id]["decision"] != "approve":
                raise contract(
                    f"candidate {candidate_id!r} proposes combine_with but is "
                    f"{parsed[candidate_id]['decision']!r} — combine edges may "
                    "only join approved candidates"
                )
            for other_id in edges:
                if other_id in display_only_ids:
                    raise contract(
                        f"candidate {candidate_id!r} combine_with cites the "
                        f"display-only candidate {other_id!r}"
                    )
                if parsed[other_id]["decision"] != "approve":
                    raise contract(
                        f"candidate {candidate_id!r} combine_with cites "
                        f"{other_id!r}, whose decision is "
                        f"{parsed[other_id]['decision']!r} — combine edges may "
                        "only join approved candidates"
                    )
        if groups:
            if candidate_id in display_only_ids:
                raise contract(
                    f"display-only candidate {candidate_id!r} may not split"
                )
            if candidate_id in combine_participants:
                raise contract(
                    f"candidate {candidate_id!r} participates in both combine "
                    "and split treatment"
                )
            if len(groups) < 2:
                raise contract(
                    f"candidate {candidate_id!r} split_groups must contain at "
                    "least two groups"
                )
            proposed = [child for group in groups for child in group]
            if sorted(proposed) != sorted(children_by_candidate[candidate_id]):
                raise contract(
                    f"candidate {candidate_id!r} split_groups must exactly "
                    "partition its children — no additions, omissions, or "
                    "overlap"
                )
            entry["split_groups"] = sorted(
                (sorted(group) for group in groups), key=tuple
            )
    return parsed


def _usage_snapshot(vlm_client: Any) -> Dict[str, int]:
    stats = getattr(vlm_client, "usage_stats", None) or {}
    return {
        key: int(stats.get(key, 0) or 0)
        for key in (
            "input_tokens", "cached_input_tokens", "output_tokens",
            "total_tokens", "metered_calls",
        )
    }


_TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
)


def _empty_usage() -> Dict[str, int]:
    return {name: 0 for name in _TOKEN_FIELDS}


def _assemble_result(
    *,
    standalone_result: Mapping[str, Any],
    package_candidates: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    sol_calls: List[Dict[str, Any]],
    estimate_id: str,
) -> Dict[str, Any]:
    listing_usage = {
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "call_count": len(sol_calls),
        **{
            name: sum(call[name] for call in sol_calls)
            for name in _TOKEN_FIELDS
        },
    }
    snapshots = package_review_snapshot_hashes({
        **standalone_result, "package_candidates": package_candidates,
    })
    result: Dict[str, Any] = {
        **standalone_result,
        "package_candidates": package_candidates,
        "package_decisions": sorted(
            decisions, key=lambda decision: decision["decision_id"]
        ),
        "sol_calls": sol_calls,
        "sol_listing_usage": listing_usage,
        "package_review_snapshots": {
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            **snapshots,
        },
    }
    validation = validate_package_review_result(result, estimate_id=estimate_id)
    if not validation.ok:
        raise _review_failure(
            "parse",
            "assembled package-review result failed self-validation: "
            + "; ".join(validation.errors[:5]),
            code="SolResultInvalid",
        )
    return result


def run_package_review(
    *,
    runtime: Any,
    estimate_id: str,
    property_key: str,
    source_run_id: str,
    created_at: str,
    standalone_result: Mapping[str, Any],
    package_candidates: List[Dict[str, Any]],
    vlm_client: Any,
    api_key: str,
    artifacts_root: Optional[Path],
) -> Dict[str, Any]:
    """Produce the validated package_review_complete result dict.

    Zero candidates short-circuit to an empty review with no provider call.
    Otherwise: checkpoint replay (no call, usage_source="checkpoint") or one
    fresh listing call with the usage delta settled before parsing."""
    if not package_candidates:
        return _assemble_result(
            standalone_result=standalone_result,
            package_candidates=[],
            decisions=[],
            sol_calls=[],
            estimate_id=estimate_id,
        )

    if vlm_client is None or artifacts_root is None:
        raise PassExecutionError(
            "sol_review", "dependency",
            "shadow package review needs a VLM client and an artifacts root "
            "before any Sol call",
            code="MissingSeamInput",
        )

    request = build_listing_request(
        standalone_result=standalone_result,
        package_candidates=package_candidates,
        projection_fingerprint=runtime.projection_fingerprint,
        model=runtime.sol_model,
        reasoning_effort=SOL_REVIEW_REASONING_EFFORT,
        max_output_tokens=runtime.sol_max_output_tokens,
    )
    children_by_candidate = {
        candidate["package_candidate_id"]: tuple(candidate["child_work_item_ids"])
        for candidate in package_candidates
    }
    display_only_ids = frozenset(
        candidate["package_candidate_id"]
        for candidate in package_candidates
        if candidate["display_only"]
    )

    path = sol_checkpoint_path(
        terra_checkpoint_dir(Path(artifacts_root), property_key, source_run_id)
    )
    cached = load_sol_checkpoint(
        path,
        request_fingerprint=request.request_fingerprint,
        package_candidate_ids=list(request.package_candidate_ids),
    )
    if cached is not None:
        call = dict(cached["sol_call"])
        call["usage_source"] = "checkpoint"
        decisions = [dict(decision) for decision in cached["decisions"]]
    else:
        from tools import pipeline_config as cfg

        ledger = SolUsageLedger(
            Path(artifacts_root),
            daily_ceiling=getattr(
                cfg, "RENOVATION_SOL_DAILY_TOKEN_CEILING", SOL_DAILY_TOKEN_CEILING
            ),
            usage_root_override=getattr(cfg, "RENOVATION_TERRA_USAGE_ROOT", None),
        )
        try:
            reservation_id = ledger.reserve(
                property_key=property_key,
                source_run_id=source_run_id,
                estimate_unit_id="listing",
                request_fingerprint=request.request_fingerprint,
                tokens=estimate_sol_reservation_tokens(
                    max_output_tokens=runtime.sol_max_output_tokens,
                    request_bytes=request.request_bytes,
                ),
            )
        except SolDailyBudgetExceeded as exc:
            raise PassExecutionError(
                "sol_review", "request", str(exc),
                code="SolDailyBudgetExceeded", provider="openai",
                model=runtime.sol_model,
            ) from exc
        before = _usage_snapshot(vlm_client)
        try:
            # This call is already reserved/settled against the Sol ledger
            # above; the Session 9 choke-point guard must not debit it again.
            with external_reservation():
                raw_text = call_sol_review(
                    vlm_client, request,
                    model=runtime.sol_model, api_key=api_key,
                    max_output_tokens=runtime.sol_max_output_tokens,
                    reasoning_effort=SOL_REVIEW_REASONING_EFFORT,
                )
        except PassExecutionError:
            # The provider may or may not have consumed tokens; keeping the
            # conservative reservation debited is the honest choice.
            ledger.settle(reservation_id, provider_total_tokens=None)
            raise
        # Settle telemetry before parsing: the tokens are spent whether or
        # not the response honors the contract.
        after = _usage_snapshot(vlm_client)
        delta = {key: after[key] - before[key] for key in before}
        metered = delta["metered_calls"] > 0
        ledger.settle(
            reservation_id,
            provider_total_tokens=delta["total_tokens"] if metered else None,
        )
        usage = (
            {name: delta[name] for name in _TOKEN_FIELDS}
            if metered else _empty_usage()
        )
        parsed = parse_listing_decisions(
            raw_text,
            candidate_ids=request.package_candidate_ids,
            children_by_candidate=children_by_candidate,
            display_only_ids=display_only_ids,
            model=runtime.sol_model,
        )
        call_id = make_sol_call_id(
            estimate_id=estimate_id,
            request_fingerprint=request.request_fingerprint,
        )
        call = {
            "call_id": call_id,
            "schema_version": CONTRACTS_SCHEMA_VERSION,
            "package_candidate_ids": sorted(request.package_candidate_ids),
            "request_fingerprint": request.request_fingerprint,
            "provider": "openai",
            "model": runtime.sol_model,
            "prompt_version": SOL_REVIEW_PROMPT_VERSION,
            "usage_source": "provider",
            **usage,
        }
        decisions = [
            {
                "decision_id": make_package_decision_id(
                    estimate_id=estimate_id, package_candidate_id=candidate_id
                ),
                "schema_version": CONTRACTS_SCHEMA_VERSION,
                "package_candidate_id": candidate_id,
                "decision": parsed[candidate_id]["decision"],
                "combine_with": parsed[candidate_id]["combine_with"],
                "split_groups": parsed[candidate_id]["split_groups"],
                "rationale": parsed[candidate_id]["rationale"],
                "model": runtime.sol_model,
                "prompt_version": SOL_REVIEW_PROMPT_VERSION,
                "sol_call_id": call_id,
                "request_fingerprint": request.request_fingerprint,
                "provider": "openai",
            }
            for candidate_id in request.package_candidate_ids
        ]
        save_sol_checkpoint(
            path,
            request_fingerprint=request.request_fingerprint,
            sol_call=call,
            decisions=decisions,
            created_at=created_at,
        )

    return _assemble_result(
        standalone_result=standalone_result,
        package_candidates=package_candidates,
        decisions=decisions,
        sol_calls=[call],
        estimate_id=estimate_id,
    )
