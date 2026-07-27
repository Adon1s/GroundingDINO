"""Read-only Q6-vs-Q5 comparison of completed photo-intel artifacts."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
JUDGE_PROMPT_VERSION = "quant_issue_disagreement_v1"
SEMANTIC_THRESHOLD = 0.88
APPROVED = {"confirmed", "confirmed_by_rule"}
TERMINAL = APPROVED | {"rejected", "uncertain"}
VERSION_KEYS = (
    "schema_version", "normalization_policy_version", "catalog_version",
    "product_policy_version", "evidence_projection_policy_version",
)


# Hashing/serialization/atomic-write helpers moved to comparison_common so the
# benchmark harness fingerprints identically; re-exported here unchanged.
from tools.comparison_common import (  # noqa: E402
    ComparisonError,
    atomic_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


def resolve_artifact(reference: Path | str) -> Path:
    path = Path(reference).expanduser().resolve()
    if path.is_dir():
        path = path / "photo_intel_debug.json"
    elif path.name == "photo_intel.json":
        path = path.with_name("photo_intel_debug.json")
    if not path.is_file() or path.name != "photo_intel_debug.json":
        raise ComparisonError(f"could not resolve photo_intel_debug.json from {reference}")
    return path


def property_id(payload: dict[str, Any]) -> str:
    prop = payload.get("property") if isinstance(payload.get("property"), dict) else {}
    value = prop.get("property_key") or payload.get("property_id")
    if not value:
        raise ComparisonError("artifact has no property identifier")
    return str(value)


def routing(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["pass"]): row for row in payload.get("model_routing") or []
        if isinstance(row, dict) and row.get("pass")
    }


def local_model(payload: dict[str, Any]) -> str:
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    explicit = str(run.get("default_local_model") or run.get("model") or "").strip()
    models = {
        str(row.get("model") or "").strip() for name, row in routing(payload).items()
        if name != "2f" and row.get("model")
    }
    if len(models) > 1:
        raise ComparisonError(f"upstream passes used multiple models: {sorted(models)}")
    routed = next(iter(models), "")
    if explicit and routed and explicit != routed:
        raise ComparisonError("default_local_model disagrees with model_routing")
    model = explicit or routed
    if not model:
        raise ComparisonError("artifact has no local model identity")
    if "qwen" not in model.lower():
        raise ComparisonError(f"upstream model is not Qwen: {model}")
    for name, row in routing(payload).items():
        if name == "2f":
            continue
        family = str(row.get("model_family") or "").lower()
        provider = str(row.get("provider") or "").lower()
        if family and family not in {"qwen", "local", "lmstudio"}:
            raise ComparisonError(f"upstream pass {name} is not local Qwen")
        if provider and provider not in {"qwen", "local", "lmstudio"}:
            raise ComparisonError(f"upstream pass {name} is not local")
        if str(row.get("model") or "") != model:
            raise ComparisonError(f"upstream pass {name} did not use {model}")
    return model


def pass2f_identity(payload: dict[str, Any]) -> tuple[str, str]:
    v4 = payload.get("renovation_estimate_v4")
    if not isinstance(v4, dict):
        raise ComparisonError("artifact has no renovation_estimate_v4")
    trace = v4.get("pass_2f_trace") if isinstance(v4.get("pass_2f_trace"), dict) else {}
    route = routing(payload).get("2f", {})
    provider = str(trace.get("provider") or route.get("provider") or route.get("model_family") or "")
    model = str(trace.get("model") or route.get("model") or "")
    if not model:
        raise ComparisonError("artifact has no Pass 2f model identity")
    return provider, model


def candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    v4 = payload.get("renovation_estimate_v4") or {}
    return [row for row in v4.get("package_candidates") or [] if isinstance(row, dict)]


def packages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    v4 = payload.get("renovation_estimate_v4") or {}
    return [row for row in v4.get("packages") or [] if isinstance(row, dict)]


def image_path(photo_key: str, photo: dict[str, Any], artifact_dir: Optional[Path] = None) -> Path:
    meta = photo.get("photo") if isinstance(photo.get("photo"), dict) else {}
    path = Path(str(meta.get("image_path") or "")).expanduser()
    if not path.is_absolute() and artifact_dir is not None:
        path = artifact_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ComparisonError(f"missing image for {photo_key}: {path}")
    return path


def load_artifact(reference: Path | str) -> dict[str, Any]:
    path = resolve_artifact(reference)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except Exception as exc:
        raise ComparisonError(f"invalid artifact {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("photos"), dict):
        raise ComparisonError(f"artifact has invalid root/photos: {path}")
    photos = {str(key): value for key, value in payload["photos"].items() if isinstance(value, dict)}
    paths = {key: image_path(key, photo, path.parent) for key, photo in photos.items()}
    return {
        "path": path, "payload": payload, "sha256": sha256_bytes(raw),
        "property_id": property_id(payload), "photos": photos, "image_paths": paths,
        "image_hashes": {key: sha256_file(value) for key, value in paths.items()},
        "local_model": local_model(payload), "pass2f": pass2f_identity(payload),
    }


def comparable_metadata(payload: dict[str, Any]) -> Any:
    value = payload.get("property_metadata")
    if not isinstance(value, dict):
        value = dict(payload.get("property") or {})
        value.pop("artifacts_dir", None)
    else:
        value = dict(value)
    value.pop("metadata_path", None)
    return value


def validate_terminal(payload: dict[str, Any], label: str) -> None:
    v4 = payload.get("renovation_estimate_v4")
    if not isinstance(v4, dict) or not isinstance(v4.get("packages"), list) or not isinstance(v4.get("package_candidates"), list):
        raise ComparisonError(f"{label} has incomplete Pass 2f package data")
    invalid = [
        f"{row.get('package_id', '<missing>')}:{row.get('verification_status', 'missing')}"
        for row in candidates(payload) if str(row.get("verification_status") or "missing") not in TERMINAL
    ]
    if invalid:
        raise ComparisonError(f"{label} has non-terminal Pass 2f statuses: {', '.join(invalid)}")


def validate_pair(base: dict[str, Any], cand: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    if base["property_id"] != cand["property_id"]:
        errors.append("property IDs differ")
    if set(base["photos"]) != set(cand["photos"]):
        errors.append("photo-key sets differ")
    for key in sorted(set(base["photos"]) & set(cand["photos"])):
        if base["image_hashes"][key] != cand["image_hashes"][key]:
            errors.append(f"photo bytes differ for {key}")
    if comparable_metadata(base["payload"]) != comparable_metadata(cand["payload"]):
        errors.append("property metadata differs")
    b_run, c_run = base["payload"].get("run") or {}, cand["payload"].get("run") or {}
    for key in ("analysis_profile", "used_pass_architecture", "pass_toggles", "detection_backend"):
        if b_run.get(key) != c_run.get(key):
            errors.append(f"run.{key} differs")
    if base["local_model"] == cand["local_model"]:
        errors.append("baseline and candidate local models are identical")
    if base["pass2f"] != cand["pass2f"]:
        errors.append("Pass 2f provider/model differs")
    for key in VERSION_KEYS:
        left, right = base["payload"].get(key), cand["payload"].get(key)
        if left and right and left != right:
            errors.append(f"{key} differs")
        elif not left or not right:
            warnings.append(f"{key} is absent from one or both artifacts")
    validate_terminal(base["payload"], "baseline")
    validate_terminal(cand["payload"], "candidate")
    if errors:
        raise ComparisonError("; ".join(errors))
    return warnings


def status_class(value: Any) -> str:
    status = str(value or "missing")
    return "approved" if status in APPROVED else "denied" if status == "rejected" else "indeterminate"


def ratio(a: int | float, b: int | float) -> Optional[float]:
    return round(float(a) / float(b), 6) if b else None


def wilson(successes: int, total: int) -> Optional[dict[str, float]]:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"low": round(max(0.0, center - margin), 6), "high": round(min(1.0, center + margin), 6)}


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = {str(x) for x in left if x}, {str(x) for x in right if x}
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a and b else 0.0


def index_packages(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("package_id") or "")
        if not key or key in result:
            raise ComparisonError(f"missing or duplicate package_id: {key!r}")
        result[key] = row
    return result


def logical_package_matches(base_rows: Sequence[dict[str, Any]], cand_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    options: list[tuple[float, str, str, str]] = []
    for base in base_rows:
        for cand in cand_rows:
            if base.get("package_type") != cand.get("package_type"):
                continue
            unit = bool(base.get("estimate_unit_id")) and base.get("estimate_unit_id") == cand.get("estimate_unit_id")
            room = bool(base.get("room")) and base.get("room") == cand.get("room")
            photos = jaccard(base.get("review_photo_keys") or [], cand.get("review_photo_keys") or [])
            catalog = jaccard(base.get("supporting_catalog_item_ids") or [], cand.get("supporting_catalog_item_ids") or [])
            if not (unit or (room and photos > 0) or catalog >= .5):
                continue
            score = .4 * unit + .15 * room + .25 * photos + .2 * catalog
            options.append((score, str(base["package_id"]), str(cand["package_id"]), "unit" if unit else "room_photo" if room and photos else "catalog"))
    used_b: set[str] = set()
    used_c: set[str] = set()
    result = []
    for score, base_id, cand_id, reason in sorted(options, key=lambda x: (-x[0], x[1], x[2])):
        if base_id in used_b or cand_id in used_c:
            continue
        used_b.add(base_id); used_c.add(cand_id)
        result.append({"baseline_package_id": base_id, "candidate_package_id": cand_id, "score": round(score, 4), "reason": reason})
    return result


def core_package_diff(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    fields = ("verification_status", "pricing_tier", "pricing_profile", "estimate_unit_id", "room", "cost_low", "cost_high", "estimate_eligible", "ui_eligible", "audit_only")
    return {
        "changed_fields": {key: {"baseline": base.get(key), "candidate": cand.get(key)} for key in fields if base.get(key) != cand.get(key)},
        "review_photo_jaccard": round(jaccard(base.get("review_photo_keys") or [], cand.get("review_photo_keys") or []), 4),
        "catalog_evidence_jaccard": round(jaccard(base.get("supporting_catalog_item_ids") or [], cand.get("supporting_catalog_item_ids") or []), 4),
    }


def compare_packages(base_payload: dict[str, Any], cand_payload: dict[str, Any]) -> dict[str, Any]:
    base_final, cand_final = index_packages(packages(base_payload)), index_packages(packages(cand_payload))
    missing, extra = sorted(set(base_final) - set(cand_final)), sorted(set(cand_final) - set(base_final))
    final_diffs = {key: core_package_diff(base_final[key], cand_final[key]) for key in sorted(set(base_final) & set(cand_final))}
    final_diffs = {key: value for key, value in final_diffs.items() if value["changed_fields"] or value["review_photo_jaccard"] < 1 or value["catalog_evidence_jaccard"] < 1}

    base_c, cand_c = index_packages(candidates(base_payload)), index_packages(candidates(cand_payload))
    common = sorted(set(base_c) & set(cand_c))
    base_only, cand_only = sorted(set(base_c) - set(cand_c)), sorted(set(cand_c) - set(base_c))
    confusion: Counter[str] = Counter()
    status_diffs = []
    detail_diffs = {}
    for key in base_only:
        confusion[f"{base_c[key].get('verification_status')}->missing"] += 1
    for key in cand_only:
        confusion[f"missing->{cand_c[key].get('verification_status')}"] += 1
    for key in common:
        bs, cs = str(base_c[key].get("verification_status")), str(cand_c[key].get("verification_status"))
        confusion[f"{bs}->{cs}"] += 1
        if bs != cs:
            status_diffs.append({"package_id": key, "baseline_status": bs, "candidate_status": cs, "baseline_class": status_class(bs), "candidate_class": status_class(cs)})
        diff = core_package_diff(base_c[key], cand_c[key])
        if diff["changed_fields"] or diff["review_photo_jaccard"] < 1 or diff["catalog_evidence_jaccard"] < 1:
            detail_diffs[key] = diff
    baseline_approved_count = sum(status_class(row.get("verification_status")) == "approved" for row in base_c.values())
    candidate_approved_count = sum(status_class(row.get("verification_status")) == "approved" for row in cand_c.values())
    approval_retained_count = sum(status_class(base_c[key].get("verification_status")) == "approved" and status_class(cand_c[key].get("verification_status")) == "approved" for key in common)
    baseline_denied_count = sum(status_class(row.get("verification_status")) == "denied" for row in base_c.values())
    denial_agreement_count = sum(status_class(base_c[key].get("verification_status")) == "denied" and status_class(cand_c[key].get("verification_status")) == "denied" for key in common)
    pricing_tier_change_count = sum("pricing_tier" in value["changed_fields"] or "pricing_profile" in value["changed_fields"] for value in detail_diffs.values())
    cost_range_change_count = sum("cost_low" in value["changed_fields"] or "cost_high" in value["changed_fields"] for value in detail_diffs.values())
    evidence_change_count = sum(value["review_photo_jaccard"] < 1 or value["catalog_evidence_jaccard"] < 1 for value in detail_diffs.values())
    return {
        "final": {"baseline_count": len(base_final), "candidate_count": len(cand_final), "common_count": len(set(base_final) & set(cand_final)), "missing_approved_package_ids": missing, "extra_approved_package_ids": extra, "exact_package_ids_equal": not missing and not extra, "detail_differences": final_diffs},
        "candidates": {"baseline_count": len(base_c), "candidate_count": len(cand_c), "exact_match_count": len(common), "baseline_only_package_ids": base_only, "candidate_only_package_ids": cand_only, "logical_matches": logical_package_matches([base_c[x] for x in base_only], [cand_c[x] for x in cand_only]), "status_confusion": dict(sorted(confusion.items())), "status_differences": status_diffs, "exact_statuses_equal": not status_diffs and not base_only and not cand_only, "approval_retained_count": approval_retained_count, "baseline_approved_count": baseline_approved_count, "candidate_approved_count": candidate_approved_count, "approval_retention": ratio(approval_retained_count, baseline_approved_count), "baseline_denied_count": baseline_denied_count, "denial_agreement_count": denial_agreement_count, "denial_agreement": ratio(denial_agreement_count, baseline_denied_count), "baseline_indeterminate_count": sum(status_class(row.get("verification_status")) == "indeterminate" for row in base_c.values()), "candidate_indeterminate_count": sum(status_class(row.get("verification_status")) == "indeterminate" for row in cand_c.values()), "pricing_tier_change_count": pricing_tier_change_count, "cost_range_change_count": cost_range_change_count, "evidence_change_count": evidence_change_count, "detail_differences": detail_diffs},
    }


_PUNCT = re.compile(r"[^a-z0-9\s]")


def normalize_text(value: Any) -> str:
    return " ".join(_PUNCT.sub(" ", str(value or "").lower()).split())


def rows_2c(photo: dict[str, Any]) -> list[dict[str, Any]]:
    debug = photo.get("debug") or {}
    all_rows, forward_rows = debug.get("labeled_debug") or [], debug.get("labeled_forward") or []
    forward_ids = {str(x.get("issue_id")) for x in forward_rows if isinstance(x, dict) and x.get("issue_id")}
    forward_text = {normalize_text(x.get("description")) for x in forward_rows if isinstance(x, dict)}
    result, seen = [], set()
    for raw in list(all_rows) + list(forward_rows):
        if not isinstance(raw, dict) or not normalize_text(raw.get("description")):
            continue
        key = (str(raw.get("issue_id") or ""), normalize_text(raw.get("description")))
        if key in seen:
            continue
        seen.add(key)
        label = str(raw.get("label") or "other")
        result.append({"index": len(result), "issue_id": raw.get("issue_id"), "description": str(raw.get("description")).strip(), "normalized": key[1], "label": label, "kind": raw.get("kind") or ("upgrade" if label == "upgrade_candidate" else "defect" if label == "defect_or_damage" else ""), "forwarded": str(raw.get("issue_id") or "") in forward_ids or key[1] in forward_text})
    return result


class SentenceEncoder:
    def __init__(self) -> None:
        self.model = None

    def encode(self, texts: Sequence[str]) -> Any:
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            try:
                from tools import pipeline_config as cfg
                # EMBEDDINGS_MODEL_NAME is now the GGUF id served over HTTP, which
                # SentenceTransformer cannot load. Prefer the dedicated ST model name.
                name = getattr(cfg, "EMBEDDINGS_ST_MODEL_NAME", None) or cfg.EMBEDDINGS_MODEL_NAME
            except Exception:
                name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(name)
        return self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def match_issues(base: Sequence[dict[str, Any]], cand: Sequence[dict[str, Any]], encoder: Any = None) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    by_text: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(cand):
        by_text[row["normalized"]].append(idx)
    matches, used_c, un_b = [], set(), []
    for b_idx, row in enumerate(base):
        choices = [idx for idx in by_text[row["normalized"]] if idx not in used_c]
        if choices:
            c_idx = choices[0]; used_c.add(c_idx)
            matches.append({"baseline_index": b_idx, "candidate_index": c_idx, "method": "exact", "similarity": 1.0})
        else:
            un_b.append(b_idx)
    un_c = [idx for idx in range(len(cand)) if idx not in used_c]
    if not un_b or not un_c or encoder is None:
        return matches, un_b, un_c
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    similarities = np.asarray(encoder.encode([base[i]["description"] for i in un_b])) @ np.asarray(encoder.encode([cand[i]["description"] for i in un_c])).T
    rows, cols = linear_sum_assignment(-similarities)
    used_b2, used_c2 = set(), set()
    for r, c in zip(rows.tolist(), cols.tolist()):
        score = float(similarities[r, c])
        if score >= SEMANTIC_THRESHOLD:
            bi, ci = un_b[r], un_c[c]; used_b2.add(bi); used_c2.add(ci)
            matches.append({"baseline_index": bi, "candidate_index": ci, "method": "embedding", "similarity": round(score, 4)})
    return matches, [i for i in un_b if i not in used_b2], [i for i in un_c if i not in used_c2]


def rows_2d(photo: dict[str, Any]) -> list[dict[str, Any]]:
    return [x for x in (photo.get("debug") or {}).get("resolved_items") or [] if isinstance(x, dict)]


def find_2d(rows: Sequence[dict[str, Any]], issue: dict[str, Any]) -> Optional[dict[str, Any]]:
    if issue.get("issue_id"):
        found = next((x for x in rows if x.get("issue_id") == issue["issue_id"]), None)
        if found:
            return found
    return next((x for x in rows if normalize_text(x.get("description")) == issue["normalized"]), None)


def compare_2d(base_2c: Sequence[dict[str, Any]], cand_2c: Sequence[dict[str, Any]], matches: Sequence[dict[str, Any]], base_photo: dict[str, Any], cand_photo: dict[str, Any]) -> dict[str, Any]:
    base_rows, cand_rows = rows_2d(base_photo), rows_2d(cand_photo)
    counts: Counter[str] = Counter(); nulls: Counter[str] = Counter(); overlaps = []; diffs = []
    for match in matches:
        bi, ci = base_2c[match["baseline_index"]], cand_2c[match["candidate_index"]]
        base, cand = find_2d(base_rows, bi), find_2d(cand_rows, ci)
        if base is None and cand is None:
            continue
        counts["compared_count"] += 1
        base_id, cand_id = (base or {}).get("resolved_item_id"), (cand or {}).get("resolved_item_id")
        counts["exact_resolved_id_count"] += base_id == cand_id
        if base_id:
            counts["baseline_nonnull_count"] += 1; counts["baseline_nonnull_retained_count"] += cand_id == base_id
        nulls[f"{'nonnull' if base_id else 'null'}->{'nonnull' if cand_id else 'null'}"] += 1
        base_kind, cand_kind = (base or {}).get("resolved_kind"), (cand or {}).get("resolved_kind")
        counts["kind_disagreement_count"] += base_kind != cand_kind
        counts["resolution_path_disagreement_count"] += (base or {}).get("resolution_path") != (cand or {}).get("resolution_path")
        base_candidates = [str(x.get("item_id")) for x in (base or {}).get("candidates") or [] if isinstance(x, dict) and x.get("item_id")]
        cand_candidates = [str(x.get("item_id")) for x in (cand or {}).get("candidates") or [] if isinstance(x, dict) and x.get("item_id")]
        overlap = jaccard(base_candidates, cand_candidates); overlaps.append(overlap)
        counts["top_candidate_agreement_count"] += bool(base_candidates and cand_candidates and base_candidates[0] == cand_candidates[0])
        if base_id != cand_id or base_kind != cand_kind or overlap < 1 or (base or {}).get("resolution_path") != (cand or {}).get("resolution_path"):
            diffs.append({"baseline_description": bi["description"], "candidate_description": ci["description"], "baseline_resolved_item_id": base_id, "candidate_resolved_item_id": cand_id, "baseline_kind": base_kind, "candidate_kind": cand_kind, "baseline_resolution_path": (base or {}).get("resolution_path"), "candidate_resolution_path": (cand or {}).get("resolution_path"), "candidate_set_jaccard": round(overlap, 4)})
    return {**dict(counts), "exact_resolved_id_rate": ratio(counts["exact_resolved_id_count"], counts["compared_count"]), "baseline_nonnull_retention": ratio(counts["baseline_nonnull_retained_count"], counts["baseline_nonnull_count"]), "null_transition_matrix": dict(sorted(nulls.items())), "candidate_set_compared_count": len(overlaps), "candidate_set_jaccard_sum": round(sum(overlaps), 6), "mean_candidate_set_jaccard": round(sum(overlaps) / len(overlaps), 4) if overlaps else None, "differences": diffs}


JUDGE_SYSTEM = """You are a blinded real-estate photo issue adjudicator. Pair A/B rows only when they describe the same visible condition or upgrade. Mark every unpaired row supported or hallucinated using the photo. Paired rows must not be classified again. Use only supplied row IDs. Return JSON."""


def judge_schema() -> dict[str, Any]:
    array = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "additionalProperties": False, "required": ["equivalent_pairs", "A_supported_ids", "B_supported_ids", "A_hallucination_ids", "B_hallucination_ids", "missed_by_both"], "properties": {"equivalent_pairs": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["A_row_id", "B_row_id"], "properties": {"A_row_id": {"type": "string"}, "B_row_id": {"type": "string"}}}}, "A_supported_ids": array, "B_supported_ids": array, "A_hallucination_ids": array, "B_hallucination_ids": array, "missed_by_both": array}}


def build_judge_prompt(fingerprint: str, prop: str, photo: str, base: Sequence[dict[str, Any]], cand: Sequence[dict[str, Any]], un_b: Sequence[int], un_c: Sequence[int]) -> tuple[str, dict[str, tuple[str, int]]]:
    baseline_first = hashlib.sha256(f"{fingerprint}|{prop}|{photo}|{JUDGE_PROMPT_VERSION}".encode()).digest()[0] % 2 == 0
    order = ("baseline", "candidate") if baseline_first else ("candidate", "baseline")
    source = {"baseline": (base, un_b), "candidate": (cand, un_c)}; display = {}; payload = {}
    for letter, side in zip(("A", "B"), order):
        payload[f"Decision {letter}"] = []
        rows, indices = source[side]
        for ordinal, idx in enumerate(indices, 1):
            row_id = f"{letter}_{ordinal:03d}"; display[row_id] = (side, idx); row = rows[idx]
            payload[f"Decision {letter}"].append({"row_id": row_id, "description": row["description"], "label": row["label"], "forwarded": row["forwarded"]})
    return f"Property: {prop}\nPhoto: {photo}\n\n{canonical_json(payload)}", display


def parse_judgment(value: Any, display: dict[str, tuple[str, int]]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ComparisonError("judge response is not an object")
    result = {"matches": [], "baseline_supported_indices": [], "candidate_supported_indices": [], "baseline_hallucination_indices": [], "candidate_hallucination_indices": [], "missed_by_both": [str(x) for x in value.get("missed_by_both") or []]}; used = set()
    for pair in value.get("equivalent_pairs") or []:
        a, b = pair.get("A_row_id"), pair.get("B_row_id")
        if a not in display or b not in display or display[a][0] == display[b][0]:
            raise ComparisonError("judge returned invalid equivalent pair")
        ai, bi = display[a], display[b]
        result["matches"].append({"baseline_index": ai[1] if ai[0] == "baseline" else bi[1], "candidate_index": ai[1] if ai[0] == "candidate" else bi[1], "method": "judge", "similarity": None}); used.update((a, b))
    for letter in ("A", "B"):
        for kind, suffix in (("supported", "supported_indices"), ("hallucination", "hallucination_indices")):
            for row_id in value.get(f"{letter}_{kind}_ids") or []:
                if row_id not in display or row_id in used:
                    raise ComparisonError("judge returned invalid or duplicate row ID")
                side, idx = display[row_id]; result[f"{side}_{suffix}"].append(idx); used.add(row_id)
    if used != set(display):
        raise ComparisonError(f"judge left rows unclassified: {sorted(set(display) - used)}")
    return result


async def call_judge(client: Any, config: dict[str, Any], path: Path, prompt: str, display: dict[str, tuple[str, int]]) -> dict[str, Any]:
    from tools.llm_json import extract_json_object
    last = None
    for attempt in range(2):
        try:
            response = await client.analyze_image(image_path=path, system_prompt=JUDGE_SYSTEM, user_prompt=prompt if not attempt else "Return valid JSON only.\n\n" + prompt, **config, max_tokens=3000, analysis_pass="Quant issue disagreement judge", response_json_schema=judge_schema(), response_schema_name=JUDGE_PROMPT_VERSION)
            return parse_judgment(extract_json_object(response), display)
        except Exception as exc:
            last = exc
    raise ComparisonError(f"issue judge failed: {last}")


async def compare_issues(base_art: dict[str, Any], cand_art: dict[str, Any], encoder: Any, fingerprint: str, judge_enabled: bool, client: Any, judge_config: Any, judgments: dict[str, Any], save_judgment: Any) -> dict[str, Any]:
    totals: Counter[str] = Counter(); labels: Counter[str] = Counter(); forwards: Counter[str] = Counter(); d2: Counter[str] = Counter(); d2_null: Counter[str] = Counter(); d2_diffs = []; photos = []
    for photo_key in sorted(base_art["photos"]):
        base, cand = rows_2c(base_art["photos"][photo_key]), rows_2c(cand_art["photos"][photo_key])
        matches, un_b, un_c = match_issues(base, cand, encoder)
        judgment = None
        if un_b or un_c:
            if judge_enabled:
                key = f"{base_art['property_id']}/{photo_key}"; judgment = judgments.get(key)
                if judgment is None:
                    prompt, display = build_judge_prompt(fingerprint, base_art["property_id"], photo_key, base, cand, un_b, un_c)
                    judgment = await call_judge(client, judge_config, base_art["image_paths"][photo_key], prompt, display); save_judgment(key, judgment)
                matches.extend(judgment["matches"]); misses = judgment["baseline_supported_indices"]; additions = judgment["candidate_supported_indices"]; b_hall = judgment["baseline_hallucination_indices"]; c_hall = judgment["candidate_hallucination_indices"]
            else:
                misses, additions, b_hall, c_hall = un_b, un_c, [], []
        else:
            misses = additions = b_hall = c_hall = []
        matches.sort(key=lambda x: (x["baseline_index"], x["candidate_index"]))
        for match in matches:
            br, cr = base[match["baseline_index"]], cand[match["candidate_index"]]
            labels[f"{br['label']}->{cr['label']}"] += 1; forwards[f"{'forward' if br['forwarded'] else 'filtered'}->{'forward' if cr['forwarded'] else 'filtered'}"] += 1
            totals["label_disagreements"] += br["label"] != cr["label"]; totals["forward_disagreements"] += br["forwarded"] != cr["forwarded"]; totals["defect_upgrade_disagreements"] += br["kind"] != cr["kind"]
        totals.update({"baseline_total": len(base), "candidate_total": len(cand), "baseline_forward": sum(x["forwarded"] for x in base), "candidate_forward": sum(x["forwarded"] for x in cand), "matched": len(matches), "baseline_supported_misses": len(misses), "candidate_supported_additions": len(additions), "baseline_hallucinations": len(b_hall), "candidate_hallucinations": len(c_hall)})
        current_d2 = compare_2d(base, cand, matches, base_art["photos"][photo_key], cand_art["photos"][photo_key])
        for key in ("compared_count", "exact_resolved_id_count", "baseline_nonnull_count", "baseline_nonnull_retained_count", "kind_disagreement_count", "resolution_path_disagreement_count", "top_candidate_agreement_count", "candidate_set_compared_count", "candidate_set_jaccard_sum"):
            d2[key] += current_d2.get(key, 0)
        for key, value in current_d2["null_transition_matrix"].items(): d2_null[key] += value
        d2_diffs.extend({"photo_key": photo_key, **x} for x in current_d2["differences"])
        photos.append({"photo_key": photo_key, "baseline_2c_count": len(base), "candidate_2c_count": len(cand), "matches": [{**m, "baseline_description": base[m["baseline_index"]]["description"], "candidate_description": cand[m["candidate_index"]]["description"], "baseline_label": base[m["baseline_index"]]["label"], "candidate_label": cand[m["candidate_index"]]["label"]} for m in matches], "baseline_supported_misses": [base[i] for i in misses], "candidate_supported_additions": [cand[i] for i in additions], "baseline_hallucinations": [base[i] for i in b_hall], "candidate_hallucinations": [cand[i] for i in c_hall], "missed_by_both": (judgment or {}).get("missed_by_both", []), "pass_2d": current_d2})
    supported = totals["matched"] + totals["baseline_supported_misses"]
    issue_union = totals["matched"] + totals["baseline_supported_misses"] + totals["candidate_supported_additions"]
    return {"pass_2c": {**dict(totals), "baseline_issue_retention": ratio(totals["matched"], supported), "semantic_issue_overlap": ratio(totals["matched"], issue_union), "forward_set_agreement": ratio(totals["matched"] - totals["forward_disagreements"], totals["matched"]), "label_confusion": dict(sorted(labels.items())), "forward_confusion": dict(sorted(forwards.items())), "judge_disagreements_enabled": judge_enabled}, "pass_2d": {**dict(d2), "exact_resolved_id_rate": ratio(d2["exact_resolved_id_count"], d2["compared_count"]), "baseline_nonnull_retention": ratio(d2["baseline_nonnull_retained_count"], d2["baseline_nonnull_count"]), "null_transition_matrix": dict(sorted(d2_null.items())), "mean_candidate_set_jaccard": ratio(d2["candidate_set_jaccard_sum"], d2["candidate_set_compared_count"]), "differences": d2_diffs}, "photos": photos}


def rehab_range(payload: dict[str, Any]) -> dict[str, Optional[float]]:
    v4 = payload.get("renovation_estimate_v4") or {}; source = next((x for x in (v4.get("final_rehab"), v4.get("totals"), v4.get("final_rehab_resale_ready")) if isinstance(x, dict)), {})
    try: low, high = (float(source.get("low", source.get("cost_low"))) if source.get("low", source.get("cost_low")) is not None else None), (float(source.get("high", source.get("cost_high"))) if source.get("high", source.get("cost_high")) is not None else None)
    except (TypeError, ValueError): low = high = None
    return {"low": low, "high": high, "midpoint": (low + high) / 2 if low is not None and high is not None else None}


def range_delta(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    delta, percent = {}, {}
    for key in ("low", "high", "midpoint"):
        b, c = base[key], cand[key]; delta[key] = round(c - b, 2) if b is not None and c is not None else None; percent[key] = round((c - b) / b, 6) if b not in (None, 0) and c is not None else None
    return {"baseline": base, "candidate": cand, "delta": delta, "percent_delta": percent}


def timings(artifact: dict[str, Any]) -> dict[str, float]:
    values: defaultdict[str, float] = defaultdict(float)
    for photo in artifact["photos"].values():
        values["processing_time"] += float(photo.get("processing_time") or 0)
        for key, value in ((photo.get("trace") or {}).get("timings_sec") or {}).items():
            try: values[f"pass_{key}"] += float(value or 0)
            except (TypeError, ValueError): pass
    return {key: round(value, 3) for key, value in sorted(values.items())}


def fingerprint(pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]], judge: bool, judge_model: Optional[str]) -> str:
    value = {"schema": REPORT_SCHEMA_VERSION, "prompt": JUDGE_PROMPT_VERSION, "threshold": SEMANTIC_THRESHOLD, "judge": judge, "judge_model": judge_model, "pairs": [{"property": b["property_id"], "baseline": b["sha256"], "candidate": c["sha256"], "images": b["image_hashes"]} for b, c in pairs]}
    return sha256_bytes(canonical_json(value).encode())


def aggregate(properties: Sequence[dict[str, Any]]) -> dict[str, Any]:
    property_count = len(properties)
    equal = sum(prop["packages"]["final"]["exact_package_ids_equal"] for prop in properties)
    baseline_packages = sum(prop["packages"]["final"]["baseline_count"] for prop in properties)
    candidate_packages = sum(prop["packages"]["final"]["candidate_count"] for prop in properties)
    missing = sum(len(prop["packages"]["final"]["missing_approved_package_ids"]) for prop in properties)
    extra = sum(len(prop["packages"]["final"]["extra_approved_package_ids"]) for prop in properties)
    common = baseline_packages - missing
    confusion: Counter[str] = Counter()
    pass_2c: Counter[str] = Counter()
    pass_2d: Counter[str] = Counter()
    timing_base: Counter[str] = Counter()
    timing_candidate: Counter[str] = Counter()
    macro_retentions: list[float] = []
    rehab: dict[str, dict[str, Optional[float]]] = {}
    for prop in properties:
        confusion.update(prop["packages"]["candidates"]["status_confusion"])
        current_2c = prop["issues"]["pass_2c"]
        for key in ("baseline_total", "candidate_total", "baseline_forward", "candidate_forward", "matched", "baseline_supported_misses", "candidate_supported_additions", "baseline_hallucinations", "candidate_hallucinations", "label_disagreements", "forward_disagreements", "defect_upgrade_disagreements"):
            pass_2c[key] += current_2c.get(key, 0)
        if current_2c.get("baseline_issue_retention") is not None:
            macro_retentions.append(current_2c["baseline_issue_retention"])
        current_2d = prop["issues"]["pass_2d"]
        for key in ("compared_count", "exact_resolved_id_count", "baseline_nonnull_count", "baseline_nonnull_retained_count", "kind_disagreement_count", "resolution_path_disagreement_count", "top_candidate_agreement_count", "candidate_set_compared_count", "candidate_set_jaccard_sum"):
            pass_2d[key] += current_2d.get(key, 0)
        timing_base.update(prop["timings"]["baseline"])
        timing_candidate.update(prop["timings"]["candidate"])
    issue_union = pass_2c["matched"] + pass_2c["baseline_supported_misses"] + pass_2c["candidate_supported_additions"]
    for key in ("low", "high", "midpoint"):
        deltas = [prop["final_rehab"]["delta"][key] for prop in properties if prop["final_rehab"]["delta"][key] is not None]
        percentages = [prop["final_rehab"]["percent_delta"][key] for prop in properties if prop["final_rehab"]["percent_delta"][key] is not None]
        rehab[key] = {"mean_absolute_delta": round(sum(abs(x) for x in deltas) / len(deltas), 2) if deltas else None, "mean_percent_delta": round(sum(percentages) / len(percentages), 6) if percentages else None}
    timing_keys = sorted(set(timing_base) | set(timing_candidate))
    return {
        "property_count": property_count,
        "exact_approved_property_count": equal,
        "exact_approved_property_rate": ratio(equal, property_count),
        "exact_approved_property_rate_ci95": wilson(equal, property_count),
        "baseline_final_package_count": baseline_packages,
        "candidate_final_package_count": candidate_packages,
        "common_final_package_count": common,
        "missing_approved_package_count": missing,
        "extra_approved_package_count": extra,
        "approval_recall": ratio(common, baseline_packages),
        "approval_recall_ci95": wilson(common, baseline_packages),
        "approval_precision": ratio(common, candidate_packages),
        "approval_precision_ci95": wilson(common, candidate_packages),
        "status_confusion": dict(sorted(confusion.items())),
        "pass_2c": {**dict(pass_2c), "micro_issue_retention": ratio(pass_2c["matched"], pass_2c["matched"] + pass_2c["baseline_supported_misses"]), "macro_issue_retention": round(sum(macro_retentions) / len(macro_retentions), 6) if macro_retentions else None, "semantic_issue_overlap": ratio(pass_2c["matched"], issue_union), "forward_set_agreement": ratio(pass_2c["matched"] - pass_2c["forward_disagreements"], pass_2c["matched"])},
        "pass_2d": {**dict(pass_2d), "exact_resolved_id_rate": ratio(pass_2d["exact_resolved_id_count"], pass_2d["compared_count"]), "baseline_nonnull_retention": ratio(pass_2d["baseline_nonnull_retained_count"], pass_2d["baseline_nonnull_count"]), "top_candidate_agreement": ratio(pass_2d["top_candidate_agreement_count"], pass_2d["candidate_set_compared_count"]), "mean_candidate_set_jaccard": ratio(pass_2d["candidate_set_jaccard_sum"], pass_2d["candidate_set_compared_count"])},
        "final_rehab": rehab,
        "timings": {"baseline": {key: round(timing_base[key], 3) for key in timing_keys}, "candidate": {key: round(timing_candidate[key], 3) for key in timing_keys}, "delta": {key: round(timing_candidate[key] - timing_base[key], 3) for key in timing_keys}},
    }

def verdicts(properties: Sequence[dict[str, Any]]) -> dict[str, Any]:
    approved = all(prop["packages"]["final"]["exact_package_ids_equal"] for prop in properties)
    statuses = all(prop["packages"]["candidates"]["exact_statuses_equal"] for prop in properties)
    no_miss = all(prop["issues"]["pass_2c"].get("baseline_supported_misses", 0) == 0 for prop in properties)
    issue_differences = any(
        prop["issues"]["pass_2c"].get(key, 0)
        for prop in properties
        for key in ("baseline_supported_misses", "candidate_supported_additions", "baseline_hallucinations", "candidate_hallucinations", "label_disagreements", "forward_disagreements", "defect_upgrade_disagreements")
    ) or any(prop["issues"]["pass_2d"].get("differences") for prop in properties)
    overall = "FAIL" if not approved else "PASS_WITH_DIFFERENCES" if not statuses or issue_differences else "PASS"
    return {"approved_package_equivalence": approved, "two_f_status_equivalence": statuses, "issue_zero_miss": no_miss, "overall": overall}

def markdown(report: dict[str, Any]) -> str:
    aggregate_stats = report["aggregate"]
    lines = ["# Q6 vs Q5 Artifact Comparison", "", f"**Overall: {report['verdicts']['overall']}**", "", "## Approved package differences", "", f"- Missing from Q5: {aggregate_stats['missing_approved_package_count']}", f"- Extra in Q5: {aggregate_stats['extra_approved_package_count']}"]
    found = False
    for prop in report["properties"]:
        final = prop["packages"]["final"]
        if final["missing_approved_package_ids"] or final["extra_approved_package_ids"]:
            found = True
            lines += ["", f"### {prop['property_id']}", "", f"- Missing IDs: {', '.join(final['missing_approved_package_ids']) or 'none'}", f"- Extra IDs: {', '.join(final['extra_approved_package_ids']) or 'none'}"]
    if not found:
        lines += ["", "No approved-package ID differences."]
    lines += ["", "## Package statistics", "", f"- Exact approved-property rate: {aggregate_stats['exact_approved_property_rate']} (95% CI: {aggregate_stats['exact_approved_property_rate_ci95']})", f"- Approval recall (Q6 baseline): {aggregate_stats['approval_recall']} (95% CI: {aggregate_stats['approval_recall_ci95']})", f"- Approval precision (Q6 baseline): {aggregate_stats['approval_precision']} (95% CI: {aggregate_stats['approval_precision_ci95']})", "", "## Pass 2f status differences", ""]
    status_lines = 0
    for prop in report["properties"]:
        candidate_stats = prop["packages"]["candidates"]
        if not candidate_stats["exact_statuses_equal"]:
            status_lines += 1
            lines.append(f"- **{prop['property_id']}**: status changes={len(candidate_stats['status_differences'])}, Q6-only={len(candidate_stats['baseline_only_package_ids'])}, Q5-only={len(candidate_stats['candidate_only_package_ids'])}")
    if not status_lines:
        lines.append("No Pass 2f candidate-status differences.")
    issue_stats = aggregate_stats["pass_2c"]
    lines += ["", "## Pass 2c issue differences", ""]
    misses = 0
    for prop in report["properties"]:
        for photo in prop["issues"]["photos"]:
            for issue in photo["baseline_supported_misses"]:
                misses += 1
                lines.append(f"- **Q5 miss — {prop['property_id']} / {photo['photo_key']}**: {issue['description']}")
    if not misses:
        lines.append("No supported Q6-only issue misses.")
    lines += ["", f"- Micro Q6 issue retention: {issue_stats.get('micro_issue_retention')}", f"- Macro Q6 issue retention: {issue_stats.get('macro_issue_retention')}", f"- Semantic issue overlap: {issue_stats.get('semantic_issue_overlap')}", f"- Forward-set agreement: {issue_stats.get('forward_set_agreement')}", f"- Supported Q5-only additions: {issue_stats.get('candidate_supported_additions', 0)}"]
    pass_2d = aggregate_stats["pass_2d"]
    lines += ["", "## Pass 2d catalog resolution", "", f"- Exact resolved-ID agreement: {pass_2d.get('exact_resolved_id_rate')}", f"- Q6 non-null resolution retained by Q5: {pass_2d.get('baseline_nonnull_retention')}", f"- Mean candidate-set Jaccard: {pass_2d.get('mean_candidate_set_jaccard')}", f"- Resolution-path changes: {pass_2d.get('resolution_path_disagreement_count', 0)}", "", "## Rehab and timing deltas", "", f"- Rehab aggregate: `{canonical_json(aggregate_stats['final_rehab'])}`", f"- Timing totals/deltas: `{canonical_json(aggregate_stats['timings'])}`"]
    return "\n".join(lines) + "\n"

def load_manifest(path: Path) -> list[tuple[Optional[str], Path, Path]]:
    try: value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc: raise ComparisonError(f"invalid manifest: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != MANIFEST_SCHEMA_VERSION or not isinstance(value.get("pairs"), list) or not value["pairs"]: raise ComparisonError("manifest must contain schema_version 1 and non-empty pairs")
    result = []
    for row in value["pairs"]:
        if not isinstance(row, dict) or not row.get("baseline_run") or not row.get("candidate_run"): raise ComparisonError("manifest pair requires baseline_run and candidate_run")
        b, c = Path(str(row["baseline_run"])), Path(str(row["candidate_run"])); result.append((str(row.get("property_id")) if row.get("property_id") else None, b if b.is_absolute() else path.parent / b, c if c.is_absolute() else path.parent / c))
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed Q6 and Q5 photo-intel artifacts")
    group = parser.add_mutually_exclusive_group(required=True); group.add_argument("--manifest", type=Path); group.add_argument("--baseline-run", type=Path)
    parser.add_argument("--candidate-run", type=Path); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--judge-disagreements", action="store_true"); parser.add_argument("--judge-model"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.baseline_run and not args.candidate_run: parser.error("--candidate-run is required with --baseline-run")
    if args.manifest and args.candidate_run: parser.error("--candidate-run cannot be used with --manifest")
    return args


async def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    specs = load_manifest(args.manifest) if args.manifest else [(None, args.baseline_run, args.candidate_run)]; loaded = []
    for expected, b_path, c_path in specs:
        base, cand = load_artifact(b_path), load_artifact(c_path)
        if expected and expected != base["property_id"]: raise ComparisonError("manifest property_id does not match artifact")
        loaded.append((base, cand, validate_pair(base, cand)))
    if args.judge_disagreements:
        from tools.model_comparison_config import load_dotenv_without_override
        load_dotenv_without_override(PROJECT_ROOT / ".env")
    judge_model = (args.judge_model or os.environ.get("MODEL_COMPARISON_JUDGE_MODEL")) if args.judge_disagreements else None
    if args.judge_disagreements and not judge_model: raise ComparisonError("judge model is required")
    fp = fingerprint([(b, c) for b, c, _ in loaded], args.judge_disagreements, judge_model)
    validation = {"status": "valid", "pair_count": len(loaded), "warnings": [{"property_id": b["property_id"], "messages": w} for b, _, w in loaded if w]}
    output = args.output.resolve()
    if args.validate_only:
        report = {"schema_version": REPORT_SCHEMA_VERSION, "config_fingerprint": fp, "validation": validation, "execution": {"status": "validated_only"}}
        atomic_json(output, report); output.with_suffix(".md").write_text("# Q6 vs Q5 Artifact Comparison\n\nValidation passed.\n", encoding="utf-8"); return report, 0
    checkpoint_path = output.with_suffix(".checkpoint.json"); checkpoint = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "config_fingerprint": fp, "judgments": {}}
    if args.resume:
        if not checkpoint_path.is_file(): raise ComparisonError("resume checkpoint is missing")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        if checkpoint.get("config_fingerprint") != fp: raise ComparisonError("checkpoint fingerprint mismatch")
    encoder, client, judge_config = SentenceEncoder(), None, None
    if args.judge_disagreements:
        if not os.environ.get("OPENAI_API_KEY"): raise ComparisonError("OPENAI_API_KEY is required for judging")
        from tools.vlm_client import create_vlm_client
        client = create_vlm_client(timeout=360); judge_config = {"provider": "openai", "model": judge_model, "api_key": os.environ["OPENAI_API_KEY"], "reasoning_effort": "low", "verbosity": "low", "timeout": 360}
    def save(key: str, value: dict[str, Any]) -> None: checkpoint["judgments"][key] = value; atomic_json(checkpoint_path, checkpoint)
    properties = []
    for base, cand, warnings in loaded:
        props = compare_packages(base["payload"], cand["payload"]); issues = await compare_issues(base, cand, encoder, fp, args.judge_disagreements, client, judge_config, checkpoint["judgments"], save)
        properties.append({"property_id": base["property_id"], "inputs": {"baseline": {"path": str(base["path"]), "sha256": base["sha256"], "model": base["local_model"]}, "candidate": {"path": str(cand["path"]), "sha256": cand["sha256"], "model": cand["local_model"]}, "pass_2f": {"provider": base["pass2f"][0], "model": base["pass2f"][1]}, "image_hashes": base["image_hashes"]}, "warnings": warnings, "packages": props, "issues": issues, "final_rehab": range_delta(rehab_range(base["payload"]), rehab_range(cand["payload"])), "timings": {"baseline": timings(base), "candidate": timings(cand)}})
    report = {"schema_version": REPORT_SCHEMA_VERSION, "config_fingerprint": fp, "comparison": {"baseline_label": "Q6", "candidate_label": "Q5", "semantic_match_threshold": SEMANTIC_THRESHOLD, "issue_judge_enabled": args.judge_disagreements, "issue_judge_model": judge_model if args.judge_disagreements else None, "issue_judge_prompt_version": JUDGE_PROMPT_VERSION}, "validation": validation, "execution": {"status": "complete"}, "verdicts": verdicts(properties), "aggregate": aggregate(properties), "properties": properties}
    atomic_json(output, report); output.with_suffix(".md").write_text(markdown(report), encoding="utf-8"); atomic_json(checkpoint_path, checkpoint)
    return report, 2 if report["verdicts"]["overall"] == "FAIL" else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv); report, code = asyncio.run(execute(args))
        if args.validate_only: print(f"Validation passed for {report['validation']['pair_count']} pair(s).")
        else:
            a = report["aggregate"]; v = report["verdicts"]
            print(f"Q6 VS Q5: {v['overall']}"); print(f"missing approved packages: {a['missing_approved_package_count']}"); print(f"extra approved packages:   {a['extra_approved_package_count']}"); print(f"approval recall:           {a['approval_recall']}"); print(f"approval precision:        {a['approval_precision']}"); print(f"2f status equivalent:      {v['two_f_status_equivalence']}"); print(f"Q5 supported issue misses: {a['pass_2c'].get('baseline_supported_misses', 0)}"); print(f"Q5 supported additions:    {a['pass_2c'].get('candidate_supported_additions', 0)}"); print(f"2c micro issue retention:  {a['pass_2c'].get('micro_issue_retention')}"); print(f"2d exact ID rate:          {a['pass_2d'].get('exact_resolved_id_rate')}"); print(f"report:                    {args.output.resolve()}")
        return code
    except ComparisonError as exc: print(f"error: {exc}", file=sys.stderr); return 1
    except KeyboardInterrupt: return 1


if __name__ == "__main__":
    raise SystemExit(main())
