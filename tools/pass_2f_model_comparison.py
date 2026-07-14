"""Artifact-first Terra-vs-Sol comparison for real package-level Pass 2f."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifact_writers import load_issue_catalog
from tools.llm_json import extract_json_object
from tools.pass_2f_artifact_inputs import prepare_replay_inputs, resolve_artifact_path
from tools.pass_2f_comparison_config import (
    CHECKPOINT_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    Pass2fComparisonConfig,
    credentials_available,
    load_allowed_dotenv,
    load_pass_2f_comparison_profile,
    qwen_endpoint,
    qwen_vlm_config,
    with_cli_overrides,
)
from tools.rehab_packages import infer_package_candidates, prepare_pass_2f_cases
from tools.scene_classifier_passes import (
    PASS_2F_PROMPT_SHA256,
    PASS_2F_PROMPT_VERSION,
    Pass2fInvalidResponseError,
    run_pass_2f,
)
from tools.vlm_client import create_vlm_client


CATALOG_PATH = ROOT / "tools" / "issue_catalog.json"
EXECUTIVE_PROMPT_VERSION = "pass_2f_executive_review_v1"
EXECUTIVE_SYSTEM_PROMPT = (
    "You are performing an operational review of two blinded visual package "
    "decisions. Use only the supplied images, package evidence, and Decision A "
    "and Decision B. Return only the requested JSON. Do not infer costs."
)
EXECUTIVE_OUTPUT_SCHEMA = (
    "Return exactly one JSON object with operational_decision set to approve, "
    "reject, or needs_manual_review; reasoning as a concise string; and "
    "supported_issue_ids and rejected_issue_ids as arrays of issue ID strings."
)
EXECUTIVE_PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "version": EXECUTIVE_PROMPT_VERSION,
            "system": EXECUTIVE_SYSTEM_PROMPT,
            "schema": EXECUTIVE_OUTPUT_SCHEMA,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
FOCUS_BUCKETS = {"terra_only_approved", "sol_only_approved"}
USAGE_KEYS = (
    "calls",
    "metered_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in value
    )
    return cleaned.strip("_") or "unknown"


def _property_id(artifact: Dict[str, Any]) -> str:
    return str(
        (artifact.get("property") or {}).get("property_key")
        or artifact.get("property_key")
        or artifact.get("property_id")
        or "unknown_property"
    )


def validate_source_artifact(
    artifact: Dict[str, Any],
    config: Pass2fComparisonConfig,
) -> None:
    run = artifact.get("run")
    if not isinstance(run, dict):
        raise ValueError("source artifact is missing run metadata")
    actual_model = str(run.get("default_local_model") or "")
    required_model = config.source.required_local_model
    if actual_model and actual_model != required_model:
        raise ValueError(
            "source run.default_local_model does not match required local model: "
            f"{actual_model!r} != {required_model!r}"
        )
    routing_required = (
        config.source.require_local_upstream_routing
        or not actual_model
    )
    if not routing_required:
        return
    routing = artifact.get("model_routing")
    if not isinstance(routing, list):
        raise ValueError("source artifact model_routing must be a list")
    upstream = [
        entry for entry in routing
        if isinstance(entry, dict) and str(entry.get("pass") or "") != "2f"
    ]
    if not upstream:
        raise ValueError("source artifact has no verifiable upstream model routing")
    for entry in upstream:
        pass_name = str(entry.get("pass") or "unknown")
        family = str(entry.get("model_family") or "").lower()
        model = str(entry.get("model") or "")
        provider = str(entry.get("provider") or "").lower()
        if family != "qwen" or model != required_model:
            raise ValueError(
                f"upstream pass {pass_name} was not routed to local Qwen "
                f"{required_model!r}"
            )
        if provider and provider not in {"local", "lmstudio", "qwen"}:
            raise ValueError(
                f"upstream pass {pass_name} used non-local provider {provider!r}"
            )


def classify_approval_bucket(terra_status: str, sol_status: str) -> str:
    terra_approved = terra_status == "confirmed"
    sol_approved = sol_status == "confirmed"
    if terra_approved and sol_approved:
        return "both_approved"
    if terra_approved:
        return "terra_only_approved"
    if sol_approved:
        return "sol_only_approved"
    return "neither_approved"


def _semantic_config(config: Pass2fComparisonConfig) -> Dict[str, Any]:
    redacted = copy.deepcopy(config.redacted_dict())
    redacted.pop("source_path", None)
    return redacted


def prepare_comparison(
    config: Pass2fComparisonConfig,
    artifact_path: Path,
    *,
    catalog_path: Path = CATALOG_PATH,
) -> Dict[str, Any]:
    artifact_bytes = artifact_path.read_bytes()
    catalog_bytes = catalog_path.read_bytes()
    try:
        artifact = json.loads(artifact_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source artifact JSON: {exc}") from exc
    if not isinstance(artifact, dict):
        raise ValueError("source artifact root must be a JSON object")
    validate_source_artifact(artifact, config)
    recorded_local_model = str(
        (artifact.get("run") or {}).get("default_local_model") or ""
    )

    catalog = load_issue_catalog(catalog_path)
    if not isinstance(catalog, dict) or not catalog.get("items"):
        raise ValueError(f"issue catalog is empty or invalid: {catalog_path}")
    issues_flat, estimate_candidates, photo_map = prepare_replay_inputs(
        artifact,
        catalog,
    )
    if issues_flat is None:
        raise ValueError(
            "artifact has no usable estimate_issues_flat or issues_flat"
        )
    package_candidates = infer_package_candidates(estimate_candidates, [], catalog)
    prepared_cases, preparation_trace = prepare_pass_2f_cases(
        package_candidates,
        photo_key_to_path=photo_map,
        max_images=config.run.max_images_per_package,
    )

    case_hashes: List[str] = []
    package_records: List[Dict[str, Any]] = []
    for case in prepared_cases:
        prepared_input = copy.deepcopy(case["prepared_input"])
        image_hashes = [
            _sha256_file(Path(path))
            for path in prepared_input.get("review_image_paths") or []
        ]
        prepared_input["review_image_sha256"] = image_hashes
        hash_payload = {
            "package_id": case["package_id"],
            "package_type": case["package_type"],
            "package_label": case["package_label"],
            "room": case["room"],
            "evaluation_kind": case["evaluation_kind"],
            "not_evaluated_reason": case["not_evaluated_reason"],
            "prepared_input": prepared_input,
        }
        case_hash = _sha256_bytes(_canonical_json(hash_payload).encode("utf-8"))
        case_hashes.append(case_hash)
        initial_bucket = None
        if case["evaluation_kind"] == "rule_confirmed":
            initial_bucket = "rule_confirmed"
        elif case["evaluation_kind"] == "not_evaluated":
            initial_bucket = "not_evaluated"
        package_records.append({
            "package_id": case["package_id"],
            "package_type": case["package_type"],
            "package_label": case["package_label"],
            "room": case["room"],
            "evaluation_kind": case["evaluation_kind"],
            "source_package": case["source_package"],
            "prepared_input": prepared_input,
            "prepared_input_sha256": case_hash,
            "rule_confirmation": case.get("rule_confirmation"),
            "terra": None,
            "sol": None,
            "qwen": None,
            "comparison": {"bucket": initial_bucket, "is_focus": False},
            "executive_review": None,
            "not_evaluated_reason": case.get("not_evaluated_reason"),
        })

    source_info = {
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": _sha256_bytes(artifact_bytes),
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": _sha256_bytes(catalog_bytes),
        "property_id": _property_id(artifact),
        "local_model": (
            recorded_local_model or config.source.required_local_model
        ),
        "local_model_verification": (
            "run_metadata"
            if recorded_local_model
            else "upstream_model_routing_legacy"
        ),
        "model_routing": _sanitize_sensitive(artifact.get("model_routing") or []),
    }
    fingerprint_payload = {
        "config": _semantic_config(config),
        "source": {
            "artifact_sha256": source_info["artifact_sha256"],
            "catalog_sha256": source_info["catalog_sha256"],
        },
        "prepared_case_hashes": case_hashes,
        "pass_2f_prompt_version": PASS_2F_PROMPT_VERSION,
        "pass_2f_prompt_sha256": PASS_2F_PROMPT_SHA256,
        "executive_prompt_version": EXECUTIVE_PROMPT_VERSION,
        "executive_prompt_sha256": EXECUTIVE_PROMPT_SHA256,
    }
    config_fingerprint = _sha256_bytes(
        _canonical_json(fingerprint_payload).encode("utf-8")
    )
    return {
        "config": config,
        "config_fingerprint": config_fingerprint,
        "source": source_info,
        "package_records": package_records,
        "preparation_trace": preparation_trace,
    }



def _empty_usage() -> Dict[str, int]:
    return {key: 0 for key in USAGE_KEYS}


def _empty_phase_stats() -> Dict[str, Any]:
    return {
        "attempted_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "wall_time_seconds": 0.0,
        "usage": _empty_usage(),
    }


def _new_report(context: Dict[str, Any]) -> Dict[str, Any]:
    config: Pass2fComparisonConfig = context["config"]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "config": config.redacted_dict(),
        "config_fingerprint": context["config_fingerprint"],
        "source": copy.deepcopy(context["source"]),
        "aggregate": {},
        "phase_stats": {
            "terra": _empty_phase_stats(),
            "sol": _empty_phase_stats(),
            "qwen": _empty_phase_stats(),
            "executive_review": _empty_phase_stats(),
        },
        "execution": {
            "status": "prepared",
            "executive_review_requested": False,
            "qwen_requested": config.run.include_qwen,
        },
        "packages": [],
        "errors": [],
    }


def _refresh_aggregate(report: Dict[str, Any]) -> None:
    counts = {
        "package_candidate_count": len(report.get("packages") or []),
        "vlm_eligible_count": 0,
        "vlm_evaluated_count": 0,
        "qwen_evaluated_count": 0,
        "qwen_approved_count": 0,
        "both_approved": 0,
        "neither_approved": 0,
        "terra_only_approved": 0,
        "sol_only_approved": 0,
        "rule_confirmed": 0,
        "not_evaluated": 0,
        "execution_failed": 0,
        "focus_package_count": 0,
    }
    for package in report.get("packages") or []:
        qwen = package.get("qwen")
        if isinstance(qwen, dict):
            counts["qwen_evaluated_count"] += 1
            counts["qwen_approved_count"] += int(
                qwen.get("verification_status") == "confirmed"
            )
        kind = package.get("evaluation_kind")
        if kind == "rule_confirmed":
            counts["rule_confirmed"] += 1
            package["comparison"] = {
                "bucket": "rule_confirmed",
                "is_focus": False,
            }
            continue
        if kind == "not_evaluated":
            counts["not_evaluated"] += 1
            package["comparison"] = {
                "bucket": "not_evaluated",
                "is_focus": False,
            }
            continue
        counts["vlm_eligible_count"] += 1
        terra = package.get("terra")
        sol = package.get("sol")
        if isinstance(terra, dict) and isinstance(sol, dict):
            bucket = classify_approval_bucket(
                str(terra.get("verification_status") or ""),
                str(sol.get("verification_status") or ""),
            )
            counts["vlm_evaluated_count"] += 1
            counts[bucket] += 1
            focus = bucket in FOCUS_BUCKETS
            counts["focus_package_count"] += int(focus)
            package["comparison"] = {"bucket": bucket, "is_focus": focus}
        else:
            package["comparison"] = {"bucket": None, "is_focus": False}
    unresolved = [
        error
        for error in report.get("errors") or []
        if not error.get("resolved")
    ]
    counts["execution_failed"] = len({
        (error.get("stage"), error.get("package_id"))
        for error in unresolved
    })
    report["aggregate"] = counts


def checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(".checkpoint.json")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def save_state(output_path: Path, report: Dict[str, Any]) -> None:
    _refresh_aggregate(report)
    checkpoint_payload = copy.deepcopy(report)
    checkpoint_payload["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
    _atomic_write_json(checkpoint_path(output_path), checkpoint_payload)
    _atomic_write_json(output_path, report)


def load_checkpoint(
    output_path: Path,
    expected_fingerprint: str,
) -> Dict[str, Any]:
    path = checkpoint_path(output_path)
    if not path.is_file():
        raise ValueError(f"checkpoint not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load checkpoint {path}: {exc}") from exc
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version does not match")
    if payload.get("config_fingerprint") != expected_fingerprint:
        raise ValueError(
            "checkpoint fingerprint mismatch; use the original source/profile/"
            "overrides or start a new output"
        )
    payload.pop("checkpoint_schema_version", None)
    return payload


def _usage_snapshot(vlm_client: Any) -> Dict[str, int]:
    stats = getattr(vlm_client, "usage_stats", {}) or {}
    return {key: int(stats.get(key) or 0) for key in USAGE_KEYS}


def _usage_delta(
    before: Dict[str, int],
    after: Dict[str, int],
) -> Dict[str, int]:
    return {key: after.get(key, 0) - before.get(key, 0) for key in USAGE_KEYS}


def _add_phase_attempt(
    report: Dict[str, Any],
    stage: str,
    *,
    wall_time: float,
    usage: Dict[str, int],
    completed: bool,
) -> None:
    stats = report["phase_stats"][stage]
    stats["attempted_count"] += 1
    stats["completed_count" if completed else "failed_count"] += 1
    stats["wall_time_seconds"] = round(
        float(stats.get("wall_time_seconds") or 0.0) + wall_time,
        3,
    )
    for key in USAGE_KEYS:
        stats["usage"][key] = int(stats["usage"].get(key) or 0) + int(
            usage.get(key) or 0
        )


def _sanitize_error(message: str) -> str:
    sanitized = str(message)
    for env_name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "LM_STUDIO_URL",
    ):
        value = os.environ.get(env_name)
        if value:
            sanitized = sanitized.replace(value, "[REDACTED]")
    local_endpoint = qwen_endpoint()
    if local_endpoint:
        sanitized = sanitized.replace(local_endpoint, "[REDACTED]")
    sanitized = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+",
        r"\1[REDACTED]",
        sanitized,
    )
    return sanitized


_SENSITIVE_REPORT_KEYS = {
    "api_key",
    "authorization",
    "base_url",
    "endpoint",
    "secret",
    "token",
    "url",
    "access_token",
    "refresh_token",
}


def _sanitize_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        output: Dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if (
                normalized in _SENSITIVE_REPORT_KEYS
                or normalized.endswith("_api_key")
                or normalized.endswith("_base_url")
                or normalized.endswith("_endpoint")
                or normalized.endswith("_secret")
            ):
                output[key] = "[REDACTED]"
            else:
                output[key] = _sanitize_sensitive(item)
        return output
    if isinstance(value, list):
        return [_sanitize_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_sensitive(item) for item in value)
    return _sanitize_error(value) if isinstance(value, str) else value


def _record_error(
    report: Dict[str, Any],
    *,
    stage: str,
    package_id: str,
    exc: Exception,
    wall_time: float,
    usage: Dict[str, int],
) -> None:
    report["errors"].append({
        "stage": stage,
        "package_id": package_id,
        "error_type": type(exc).__name__,
        "message": _sanitize_error(str(exc)),
        "raw_response": _sanitize_sensitive(
            getattr(exc, "raw_response", None)
        ),
        "parsed_response": _sanitize_sensitive(
            getattr(exc, "parsed_response", None)
        ),
        "wall_time_seconds": round(wall_time, 3),
        "usage": usage,
        "resolved": False,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    })


def _resolve_prior_errors(
    report: Dict[str, Any],
    *,
    stage: str,
    package_id: str,
) -> None:
    for error in report.get("errors") or []:
        if (
            error.get("stage") == stage
            and error.get("package_id") == package_id
            and not error.get("resolved")
        ):
            error["resolved"] = True
            error["resolved_at"] = datetime.now(timezone.utc).isoformat()


def _verify_frozen_images(package: Dict[str, Any]) -> None:
    prepared = package["prepared_input"]
    paths = prepared.get("review_image_paths") or []
    expected = prepared.get("review_image_sha256") or []
    if len(paths) != len(expected):
        raise ValueError("prepared image path/hash counts differ")
    for raw_path, expected_hash in zip(paths, expected):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"prepared review image is missing: {path}")
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"prepared review image changed: {path}")


def _serialize_contestant_result(
    result: Any,
    *,
    wall_time: float,
    usage: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "verification_status": result.verification_status,
        "approved": result.verification_status == "confirmed",
        "confirmed_issue_ids": list(result.confirmed_issue_ids or []),
        "rejected_issue_ids": list(result.rejected_issue_ids or []),
        "evidence_summary": result.evidence_summary,
        "visible_room_count": result.visible_room_count,
        "visible_room_count_evidence": result.visible_room_count_evidence,
        "parsed_response": _sanitize_sensitive(
            copy.deepcopy(result.parsed_response)
        ),
        "raw_response": _sanitize_sensitive(result.raw_response),
        "wall_time_seconds": round(wall_time, 3),
        "usage": usage,
    }


def _blind_identity_payload(
    value: Any,
    config: Pass2fComparisonConfig,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _blind_identity_payload(item, config)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _blind_identity_payload(item, config)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(_blind_identity_payload(item, config) for item in value)
    if not isinstance(value, str):
        return value
    forbidden = {
        config.name,
        config.terra.label,
        config.terra.model,
        config.sol.label,
        config.sol.model,
        config.executive_review.model.label,
        config.executive_review.model.model,
    }
    output = value
    for identity in sorted(
        (item for item in forbidden if item),
        key=len,
        reverse=True,
    ):
        output = re.sub(
            re.escape(identity),
            "[blinded]",
            output,
            flags=re.IGNORECASE,
        )
    return output


def _identity_redacted_decision(
    result: Dict[str, Any],
    config: Pass2fComparisonConfig,
) -> Dict[str, Any]:
    summary = str(result.get("evidence_summary") or "")
    forbidden = {
        config.name,
        config.terra.label,
        config.terra.model,
        config.sol.label,
        config.sol.model,
        config.executive_review.model.label,
        config.executive_review.model.model,
    }
    for value in sorted(
        (item for item in forbidden if item),
        key=len,
        reverse=True,
    ):
        summary = re.sub(
            re.escape(value),
            "[blinded]",
            summary,
            flags=re.IGNORECASE,
        )
    return _blind_identity_payload({
        "verification_status": result.get("verification_status"),
        "confirmed_issue_ids": list(result.get("confirmed_issue_ids") or []),
        "rejected_issue_ids": list(result.get("rejected_issue_ids") or []),
        "evidence_summary": summary,
        "visible_room_count": result.get("visible_room_count"),
        "visible_room_count_evidence": result.get(
            "visible_room_count_evidence"
        ),
    }, config)


def _executive_order(
    config_fingerprint: str,
    package_id: str,
) -> Tuple[str, str]:
    digest = hashlib.sha256(
        f"{config_fingerprint}|{package_id}|{EXECUTIVE_PROMPT_VERSION}".encode(
            "utf-8"
        )
    ).digest()
    return ("terra", "sol") if digest[0] % 2 == 0 else ("sol", "terra")


def build_executive_prompts(
    package: Dict[str, Any],
    config: Pass2fComparisonConfig,
    config_fingerprint: str,
) -> Tuple[str, str, Dict[str, str]]:
    first, second = _executive_order(
        config_fingerprint,
        str(package.get("package_id") or ""),
    )
    decisions = {
        "Decision A": _identity_redacted_decision(package[first], config),
        "Decision B": _identity_redacted_decision(package[second], config),
    }
    package_payload = {
        "package_id": package.get("package_id"),
        "package_type": package.get("package_type"),
        "package_label": package.get("package_label"),
        "room": package.get("room"),
        "evidence_items": (
            package["prepared_input"].get("evidence_items") or []
        ),
        "reviewed_issue_ids": (
            package["prepared_input"].get("reviewed_issue_ids") or []
        ),
    }
    package_payload = _blind_identity_payload(package_payload, config)
    user_prompt = (
        "Review the supplied package images and evidence.\n\n"
        "Package:\n"
        + _canonical_json(package_payload)
        + "\n\nBlinded decisions:\n"
        + _canonical_json(decisions)
        + "\n\n"
        + EXECUTIVE_OUTPUT_SCHEMA
    )
    return EXECUTIVE_SYSTEM_PROMPT, user_prompt, {
        "Decision A": first,
        "Decision B": second,
    }


def _parse_executive_response(
    response: str,
    valid_issue_ids: List[str],
) -> Dict[str, Any]:
    try:
        parsed = extract_json_object(response)
    except Exception as exc:
        raise Pass2fInvalidResponseError(
            f"executive review returned invalid JSON: {exc}",
            raw_response=response,
        ) from exc
    if not isinstance(parsed, dict) or not parsed:
        raise Pass2fInvalidResponseError(
            "executive review returned no JSON object",
            raw_response=response,
        )
    decision = parsed.get("operational_decision")
    if decision not in {"approve", "reject", "needs_manual_review"}:
        raise Pass2fInvalidResponseError(
            "executive operational_decision is invalid",
            raw_response=response,
            parsed_response=parsed,
        )
    if not isinstance(parsed.get("reasoning"), str):
        raise Pass2fInvalidResponseError(
            "executive reasoning must be a string",
            raw_response=response,
            parsed_response=parsed,
        )
    valid = set(valid_issue_ids)
    output: Dict[str, Any] = {
        "operational_decision": decision,
        "reasoning": parsed["reasoning"].strip(),
    }
    for key in ("supported_issue_ids", "rejected_issue_ids"):
        values = parsed.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise Pass2fInvalidResponseError(
                f"executive {key} must be an array of strings",
                raw_response=response,
                parsed_response=parsed,
            )
        canonical: List[str] = []
        for value in values:
            if value in valid and value not in canonical:
                canonical.append(value)
        output[key] = canonical
    output["parsed_response"] = _sanitize_sensitive(parsed)
    output["raw_response"] = _sanitize_sensitive(response)
    return output



async def execute_comparison(
    context: Dict[str, Any],
    output_path: Path,
    *,
    resume: bool = False,
    executive_review: bool = False,
    vlm_client: Any = None,
    vlm_clients: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool]:
    config: Pass2fComparisonConfig = context["config"]
    if resume:
        report = load_checkpoint(output_path, context["config_fingerprint"])
    else:
        if output_path.exists() or checkpoint_path(output_path).exists():
            raise ValueError(
                f"output already exists; use --resume or another path: {output_path}"
            )
        report = _new_report(context)
        save_state(output_path, report)
        for package in context["package_records"]:
            report["packages"].append(copy.deepcopy(package))
            save_state(output_path, report)

    requested_executive = executive_review
    report["execution"]["executive_review_requested"] = requested_executive
    report["execution"]["qwen_requested"] = config.run.include_qwen
    config.validate(require_credentials=True)

    active_stages = list(config.run.contestant_order)
    if config.run.include_qwen:
        active_stages.append("qwen")
    if vlm_clients is not None:
        missing_clients = [
            stage for stage in active_stages if stage not in vlm_clients
        ]
        if missing_clients:
            raise ValueError(
                f"missing VLM clients for enabled stages: {missing_clients}"
            )
        clients = {stage: vlm_clients[stage] for stage in active_stages}
    elif vlm_client is not None:
        # Backward-compatible test/integration injection. Production always uses
        # one client per stream so usage deltas cannot overlap.
        clients = {stage: vlm_client for stage in active_stages}
    else:
        clients = {
            stage: create_vlm_client(
                timeout=getattr(config, stage).timeout_seconds
            )
            for stage in active_stages
        }

    model_configs = {
        stage: (
            qwen_vlm_config(config)
            if stage == "qwen"
            else getattr(config, stage).to_vlm_config()
        )
        for stage in active_stages
    }
    save_lock = asyncio.Lock()
    stop_requested = asyncio.Event()
    failed = False

    async def persist_state() -> None:
        async with save_lock:
            save_state(output_path, report)

    async def run_contestant_stream(stage: str) -> None:
        nonlocal failed
        client = clients[stage]
        model_config = model_configs[stage]
        for package in report["packages"]:
            if config.run.fail_fast and stop_requested.is_set():
                break
            if package.get("evaluation_kind") != "vlm_eligible":
                continue
            if package.get(stage) is not None:
                continue
            before = _usage_snapshot(client)
            started = time.perf_counter()
            try:
                _verify_frozen_images(package)
                result = await run_pass_2f(
                    image_paths=[
                        Path(path)
                        for path in package["prepared_input"].get(
                            "review_image_paths"
                        ) or []
                    ],
                    vlm_client=client,
                    model_config=model_config,
                    room=str(package.get("room") or ""),
                    package_id=str(package.get("package_id") or ""),
                    package_type=str(package.get("package_type") or ""),
                    package_label=str(package.get("package_label") or ""),
                    evidence_items=copy.deepcopy(
                        package["prepared_input"].get("evidence_items") or []
                    ),
                    strict=True,
                )
                wall_time = time.perf_counter() - started
                usage = _usage_delta(before, _usage_snapshot(client))
                package[stage] = _serialize_contestant_result(
                    result,
                    wall_time=wall_time,
                    usage=usage,
                )
                _resolve_prior_errors(
                    report,
                    stage=stage,
                    package_id=package["package_id"],
                )
                _add_phase_attempt(
                    report,
                    stage,
                    wall_time=wall_time,
                    usage=usage,
                    completed=True,
                )
                await persist_state()
            except Exception as exc:
                wall_time = time.perf_counter() - started
                usage = _usage_delta(before, _usage_snapshot(client))
                _add_phase_attempt(
                    report,
                    stage,
                    wall_time=wall_time,
                    usage=usage,
                    completed=False,
                )
                _record_error(
                    report,
                    stage=stage,
                    package_id=package["package_id"],
                    exc=exc,
                    wall_time=wall_time,
                    usage=usage,
                )
                report["execution"]["status"] = "failed"
                failed = True
                if config.run.fail_fast:
                    stop_requested.set()
                await persist_state()
                if config.run.fail_fast:
                    break

    await asyncio.gather(*(
        run_contestant_stream(stage) for stage in active_stages
    ))
    client = clients["sol"]

    _refresh_aggregate(report)
    if requested_executive and not failed:
        spec = config.executive_review.model
        model_config = spec.to_vlm_config()
        for package in report["packages"]:
            if package["comparison"].get("bucket") not in FOCUS_BUCKETS:
                continue
            if package.get("executive_review") is not None:
                continue
            before = _usage_snapshot(client)
            started = time.perf_counter()
            try:
                _verify_frozen_images(package)
                system_prompt, user_prompt, mapping = build_executive_prompts(
                    package,
                    config,
                    context["config_fingerprint"],
                )
                response = await client.analyze_images(
                    image_paths=[
                        Path(path)
                        for path in package["prepared_input"].get(
                            "review_image_paths"
                        ) or []
                    ],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    analysis_pass="Pass 2f executive review",
                    **model_config,
                )
                parsed = _parse_executive_response(
                    response,
                    list(
                        package["prepared_input"].get("reviewed_issue_ids") or []
                    ),
                )
                wall_time = time.perf_counter() - started
                usage = _usage_delta(before, _usage_snapshot(client))
                package["executive_review"] = {
                    **parsed,
                    "decision_mapping": mapping,
                    "non_independent": True,
                    "review_model_is_contestant": True,
                    "wall_time_seconds": round(wall_time, 3),
                    "usage": usage,
                }
                _resolve_prior_errors(
                    report,
                    stage="executive_review",
                    package_id=package["package_id"],
                )
                _add_phase_attempt(
                    report,
                    "executive_review",
                    wall_time=wall_time,
                    usage=usage,
                    completed=True,
                )
                save_state(output_path, report)
            except Exception as exc:
                wall_time = time.perf_counter() - started
                usage = _usage_delta(before, _usage_snapshot(client))
                _add_phase_attempt(
                    report,
                    "executive_review",
                    wall_time=wall_time,
                    usage=usage,
                    completed=False,
                )
                _record_error(
                    report,
                    stage="executive_review",
                    package_id=package["package_id"],
                    exc=exc,
                    wall_time=wall_time,
                    usage=usage,
                )
                report["execution"]["status"] = "failed"
                save_state(output_path, report)
                return report, False

    unresolved = [
        error
        for error in report.get("errors") or []
        if not error.get("resolved")
    ]
    success = not failed and not unresolved
    report["execution"]["status"] = "complete" if success else "failed"
    save_state(output_path, report)
    return report, success



def _output_prefix(context: Dict[str, Any]) -> str:
    config: Pass2fComparisonConfig = context["config"]
    return (
        "pass_2f_comparison_"
        + _safe_name(context["source"]["property_id"])
        + "_"
        + _safe_name(
            config.name + ("_with_qwen" if config.run.include_qwen else "")
        )
        + "_"
        + context["source"]["artifact_sha256"][:12]
        + "_"
    )


def resolve_output_path(
    context: Dict[str, Any],
    *,
    requested: Optional[str],
    resume: bool,
    output_dir: Optional[Path] = None,
    allow_fresh_resume: bool = False,
) -> Path:
    if requested:
        return Path(requested).resolve()
    directory = (output_dir or Path.cwd()).resolve()
    prefix = _output_prefix(context)
    if resume:
        candidates = sorted(
            directory.glob(prefix + "*.checkpoint.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return Path(
                str(candidates[0]).replace(".checkpoint.json", ".json")
            ).resolve()
        if not allow_fresh_resume:
            raise ValueError(
                f"--resume found no checkpoint matching {prefix}*.checkpoint.json"
            )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (directory / f"{prefix}{stamp}.json").resolve()


def _print_summary(report: Dict[str, Any], output_path: Path) -> None:
    aggregate = report.get("aggregate") or {}
    ordered = (
        "package_candidate_count",
        "vlm_eligible_count",
        "vlm_evaluated_count",
        "qwen_evaluated_count",
        "qwen_approved_count",
        "both_approved",
        "neither_approved",
        "terra_only_approved",
        "sol_only_approved",
        "rule_confirmed",
        "not_evaluated",
        "execution_failed",
        "focus_package_count",
    )
    for key in ordered:
        print(f"{key}: {int(aggregate.get(key) or 0)}")
    print("focus_packages:")
    focus = [
        package
        for package in report.get("packages") or []
        if package.get("comparison", {}).get("is_focus")
    ]
    if not focus:
        print("  none")
    for package in focus:
        print(
            "  "
            + str(package.get("package_id"))
            + " ("
            + str(package.get("package_type"))
            + "): Terra="
            + str((package.get("terra") or {}).get("verification_status"))
            + " Sol="
            + str((package.get("sol") or {}).get("verification_status"))
            + (
                " Qwen="
                + str((package.get("qwen") or {}).get("verification_status"))
                if bool(
                    ((report.get("config") or {}).get("run") or {}).get(
                        "include_qwen"
                    )
                )
                else ""
            )
        )
    print(f"report_path: {output_path.resolve()}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Terra and Sol on frozen package Pass 2f inputs, with "
            "optional concurrent local Qwen verification."
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--run",
        dest="runs",
        required=True,
        nargs="+",
        metavar="RUN",
        help=(
            "One or more run directories or photo_intel.json files. Multiple "
            "runs are validated first and then compared sequentially."
        ),
    )
    parser.add_argument("--output")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--executive-review", action="store_true")
    parser.add_argument("--include-qwen", action="store_true")
    parser.add_argument("--max-images-per-package", type=int)
    parser.add_argument("--terra-model")
    parser.add_argument("--sol-model")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    load_allowed_dotenv(ROOT / ".env")
    output_path: Optional[Path] = None
    try:
        config = load_pass_2f_comparison_profile(args.profile, ROOT)
        config = with_cli_overrides(
            config,
            terra_model=args.terra_model,
            sol_model=args.sol_model,
            max_images_per_package=args.max_images_per_package,
            include_qwen=args.include_qwen,
        )
        batch_mode = len(args.runs) > 1
        batch_output_dir = (
            Path(args.output).resolve()
            if batch_mode and args.output
            else None
        )
        if batch_output_dir is not None and batch_output_dir.exists():
            if not batch_output_dir.is_dir():
                raise ValueError(
                    "with multiple --run paths, --output must be a directory"
                )

        prepared_runs: List[Tuple[Dict[str, Any], Path]] = []
        seen_artifacts = set()
        for run_value in args.runs:
            output_path = None
            artifact_path = resolve_artifact_path(Path(run_value))
            artifact_key = str(artifact_path.resolve()).casefold()
            if artifact_key in seen_artifacts:
                raise ValueError(f"duplicate --run artifact: {artifact_path}")
            seen_artifacts.add(artifact_key)
            context = prepare_comparison(config, artifact_path)
            output_path = resolve_output_path(
                context,
                requested=(args.output if not batch_mode else None),
                resume=args.resume,
                output_dir=batch_output_dir,
                allow_fresh_resume=batch_mode,
            )
            prepared_runs.append((context, output_path))

        if args.validate_config:
            validations = []
            for context, run_output_path in prepared_runs:
                trace = context["preparation_trace"]
                validations.append({
                    "config_fingerprint": context["config_fingerprint"],
                    "source": context["source"],
                    "preview": {
                        "package_candidate_count": trace["candidate_count"],
                        "vlm_eligible_count": trace["vlm_eligible_count"],
                        "rule_confirmed_count": trace["rule_confirmed_count"],
                        "not_evaluated_count": trace["not_evaluated_count"],
                        "not_evaluated_reasons": trace[
                            "not_evaluated_reasons"
                        ],
                    },
                    "output": str(run_output_path),
                })
            if not batch_mode:
                payload = {
                    "valid": True,
                    "config": config.redacted_dict(),
                    "config_fingerprint": validations[0][
                        "config_fingerprint"
                    ],
                    "credentials_available": credentials_available(),
                    "source": validations[0]["source"],
                    "preview": validations[0]["preview"],
                    "output": validations[0]["output"],
                }
            else:
                payload = {
                    "valid": True,
                    "config": config.redacted_dict(),
                    "credentials_available": credentials_available(),
                    "run_count": len(validations),
                    "runs": validations,
                }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if batch_output_dir is not None:
            batch_output_dir.mkdir(parents=True, exist_ok=True)
        total_runs = len(prepared_runs)
        for index, (context, run_output_path) in enumerate(
            prepared_runs,
            start=1,
        ):
            output_path = run_output_path
            if batch_mode:
                if index > 1:
                    print()
                print(
                    f"comparison_run: {index}/{total_runs} "
                    f"{context['source']['property_id']}"
                )
            report, success = asyncio.run(execute_comparison(
                context,
                output_path,
                resume=args.resume,
                executive_review=args.executive_review,
            ))
            _print_summary(report, output_path)
            if not success:
                return 1
        return 0
    except Exception as exc:
        print(f"error: {_sanitize_error(str(exc))}", file=sys.stderr)
        resolved_output = output_path
        if resolved_output is None and args.output and len(args.runs) == 1:
            resolved_output = Path(args.output).resolve()
        if resolved_output is not None:
            print(
                f"report_path: {resolved_output}",
                file=sys.stderr,
            )
        return 1

if __name__ == "__main__":
    sys.exit(main())

