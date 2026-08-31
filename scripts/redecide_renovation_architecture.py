"""QP1 re-decide harness: re-submit stored canary Terra unit requests under
prompt/payload variants, to be scored against the frozen human labels by
scripts/score_redecide_variants.py (Session B builds; live passes are
Session F's, post observation window).

Sibling of scripts/replay_renovation_architecture.py, which stays
structurally provider-free — this script is the one place a re-decide may
spend Terra tokens, and only under --live. The default is a dry run that
rebuilds every unit request from the stored envelope, verifies its
fingerprint against the stored terra_calls entry, applies the variant, and
writes the would-be request; the VLM client and review pipeline are only
imported under --live.

Safety invariants:
- Never touches any .checkpoints directory: production checkpoint reuse is
  silent on a fingerprint match, so sharing that store would republish
  stored verdicts and measure nothing. Resume is per-unit output files keyed
  by the variant fingerprint instead.
- Every arm (including the same-prompt control) fingerprints the prompt TEXT
  into a harness namespace, so no harness request can collide with a
  production fingerprint (production fingerprints only the prompt version).
- The output root must lie outside the frozen canary tree and the production
  artifacts root.
- A unit whose recomputed fingerprint mismatches the stored one is refused,
  never re-decided: the photo bytes (mutable renointel-prod tree), the
  catalog, or the unstored knobs (RENOVATION_TERRA_MAX_OUTPUT_TOKENS, the
  reasoning-effort constant) drifted, and a re-decision would measure
  garbage.

The --factorized arm is the shadow experiment in
docs/HANDOFF_factorized_verifier_replay.md: the same payload under a different
question structure (three bounded factors, class derived in code), which also
swaps the response schema. It measures only — it never feeds dispositions.

Usage (dry run, the only Session B mode):
  .venv\\Scripts\\python.exe scripts\\redecide_renovation_architecture.py --control
  .venv\\Scripts\\python.exe scripts\\redecide_renovation_architecture.py \
      --variant configs\\redecide_variants\\example_remove_observations.json
  .venv\\Scripts\\python.exe scripts\\redecide_renovation_architecture.py --factorized
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.comparison_common import sha256_canonical  # noqa: E402
from tools.pass_2f_artifact_inputs import photo_key_to_path  # noqa: E402
from tools.renovation_architecture.catalog_projection import (  # noqa: E402
    build_renovation_catalog_projection,
)
from tools.renovation_architecture.conditions import ConditionDraft  # noqa: E402
from tools.renovation_architecture.contracts import (  # noqa: E402
    TERRA_REVIEW_PROMPT_VERSION,
    TERRA_REVIEW_REASONING_EFFORT,
    EvidenceFacts,
    ObservedCondition,
)
from tools.renovation_architecture.evidence import (  # noqa: E402
    build_photo_identity_index,
)
from tools.renovation_architecture.factorized_review import (  # noqa: E402
    FACTORIZED_PROMPT_VERSION,
    FACTORIZED_SYSTEM_PROMPT,
    build_factorized_response_schema,
)
from tools.renovation_architecture.terra_review import (  # noqa: E402
    TerraUnitRequest,
    build_unit_request,
)
from tools.renovation_architecture.usage_guard import (  # noqa: E402
    estimate_reservation_tokens,
)
from tools.review_cards import PROD_ROOT  # noqa: E402
from tools.scene_classifier_passes import PassExecutionError  # noqa: E402

CATALOG_PATH = REPO_ROOT / "tools" / "issue_catalog_kind_v2.json"
ENVELOPE_KEY = "renovation_estimate_v5"
HARNESS_NAMESPACE = "redecide_v1"
# The literal seam build_unit_request puts between the photo listing and the
# fingerprinted JSON payload; variants mutate only the payload side.
PAYLOAD_DELIMITER = "\n\nConditions to review:\n"
VARIANT_KEYS = frozenset(
    {"label", "target", "system_prompt", "extra_condition_keys",
     "drop_condition_keys", "notes", "response_contract"}
)
# condition_id is what parse_unit_reviews joins on; photo_keys is the photo
# contract. Neither may be dropped or overwritten by a variant.
PROTECTED_CONDITION_KEYS = frozenset({"condition_id", "photo_keys"})
CONTROL_VARIANT = {"label": "control", "target": "terra"}
# The factorized arm (docs/HANDOFF_factorized_verifier_replay.md): same model,
# same payload, three bounded factors instead of one verdict. It swaps the
# system prompt AND the response schema, so it is a built-in arm rather than a
# variant spec — a spec file may not invent a response contract.
FACTORIZED_CONTRACT = "factorized_v1"
FACTORIZED_VARIANT = {
    "label": "factorized_v1",
    "target": "terra",
    "system_prompt": FACTORIZED_SYSTEM_PROMPT,
    "response_contract": FACTORIZED_CONTRACT,
}


def load_variant(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return dict(CONTROL_VARIANT)
    variant = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(variant) - VARIANT_KEYS
    if unknown:
        raise SystemExit(f"variant {path} carries unknown keys {sorted(unknown)}")
    label = str(variant.get("label") or "")
    if not label or not all(c.isalnum() or c == "_" for c in label):
        raise SystemExit("variant label must be a non-empty [A-Za-z0-9_]+ slug")
    if label in ("control", FACTORIZED_VARIANT["label"]) and (
        variant.get("system_prompt")
        or variant.get("extra_condition_keys")
        or variant.get("drop_condition_keys")
    ):
        raise SystemExit(f"the label {label!r} is reserved for a built-in arm")
    contract = variant.get("response_contract")
    if contract is not None and contract != FACTORIZED_CONTRACT:
        raise SystemExit(
            f"unknown response_contract {contract!r}; the only non-default "
            f"contract is {FACTORIZED_CONTRACT!r}, reached with --factorized"
        )
    if str(variant.get("target") or "terra") != "terra":
        raise SystemExit("only target='terra' is implemented (Sol is Session F)")
    touched = set(variant.get("drop_condition_keys") or []) | set(
        variant.get("extra_condition_keys") or {}
    )
    protected = touched & PROTECTED_CONDITION_KEYS
    if protected:
        raise SystemExit(
            f"variant may not mutate protected payload keys {sorted(protected)}"
        )
    return variant


def _is_mutating(variant: Mapping[str, Any]) -> bool:
    return bool(
        variant.get("system_prompt")
        or variant.get("extra_condition_keys")
        or variant.get("drop_condition_keys")
    )


def variant_fingerprint(
    label: str, *, base_fingerprint: str, system_prompt: str, user_prompt: str
) -> str:
    """Harness-namespaced and over the prompt TEXT — unlike production, which
    fingerprints only TERRA_REVIEW_PROMPT_VERSION — so no two arms and no
    arm/production pair can ever share a fingerprint."""
    return sha256_canonical({
        "harness": HARNESS_NAMESPACE,
        "variant": label,
        "base_request_fingerprint": base_fingerprint,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    })


def apply_variant(
    request: TerraUnitRequest, variant: Mapping[str, Any]
) -> TerraUnitRequest:
    prefix, seam, payload_text = request.user_prompt.partition(PAYLOAD_DELIMITER)
    if not seam:
        raise ValueError("unit request has no payload seam to mutate")
    payload = json.loads(payload_text)
    for entry in payload["conditions"]:
        for key in variant.get("drop_condition_keys") or []:
            entry.pop(key, None)
        for key, value in (variant.get("extra_condition_keys") or {}).items():
            entry[key] = value
    user_prompt = prefix + seam + json.dumps(payload, indent=2, sort_keys=True)
    system_prompt = str(variant.get("system_prompt") or request.system_prompt)
    if not _is_mutating(variant) and (
        user_prompt != request.user_prompt
        or system_prompt != request.system_prompt
    ):
        raise AssertionError(
            "control arm failed prompt byte-identity with the rebuilt baseline"
        )
    return replace(
        request,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        request_bytes=len((system_prompt + user_prompt).encode("utf-8")),
        request_fingerprint=variant_fingerprint(
            str(variant["label"]),
            base_fingerprint=request.request_fingerprint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ),
    )


def _tupled(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupled(item) for item in value)
    return value


def _from_stored(cls, stored: Mapping[str, Any]):
    """Rebuild a frozen contract dataclass from its stored dict form. A field
    the stored artifact lacks fails loudly — contract drift must never
    mis-fingerprint silently."""
    values = {}
    for field in fields(cls):
        if field.name not in stored:
            raise KeyError(
                f"stored {cls.__name__} lacks field {field.name!r} — the "
                "contract drifted since the canary was written"
            )
        values[field.name] = _tupled(stored[field.name])
    return cls(**values)


def _newest_run_dir(candidate_dir: Path) -> Optional[Path]:
    runs = sorted(
        run for run in candidate_dir.iterdir()
        if run.is_dir() and (run / "photo_intel_debug.json").is_file()
    )
    return runs[-1] if runs else None


def _load_envelope(run_dir: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """The whole v5 envelope, root placement (new mode) or analysis_debug
    (shadow) — the review_cards.v5_result placements, kept envelope-level
    because the harness needs provenance too."""
    art = json.loads((run_dir / "photo_intel_debug.json").read_text(encoding="utf-8"))
    envelope = art.get(ENVELOPE_KEY)
    if not isinstance(envelope, dict):
        envelope = (art.get("analysis_debug") or {}).get(ENVELOPE_KEY)
    return art, envelope if isinstance(envelope, dict) else None


def _unit_path(out_dir: Path, estimate_unit_id: str) -> Path:
    digest = hashlib.sha256(estimate_unit_id.encode("utf-8")).hexdigest()[:16]
    return out_dir / "units" / f"terra_unit_{digest}.json"


def _refusal(base: Dict[str, Any], status: str, detail: str) -> Dict[str, Any]:
    return {**base, "status": status, "detail": detail}


FINGERPRINT_MISMATCH_DETAIL = (
    "rebuilt request fingerprint differs from the stored terra_calls entry; "
    "possible causes: photo bytes drifted (the images live in the mutable "
    "renointel-prod tree), the catalog/projection drifted, or the unstored "
    "knobs differ from canary time (RENOVATION_TERRA_MAX_OUTPUT_TOKENS env, "
    "TERRA_REVIEW_REASONING_EFFORT constant)"
)


def redecide_property(
    run_dir: Path,
    projection: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    max_output_tokens: int,
    out_dir: Path,
    dry_run: bool,
    live_ctx: Optional[Dict[str, Any]] = None,
    limit_units: Optional[int] = None,
) -> Dict[str, Any]:
    art, envelope = _load_envelope(run_dir)
    if envelope is None or envelope.get("state") != "complete":
        state = envelope.get("state") if envelope else None
        return {"state": "skipped", "reason": f"envelope state {state!r}"}
    provenance = envelope.get("provenance") or {}
    if provenance.get("projection_fingerprint") != projection["fingerprint"]:
        return {
            "state": "refused_catalog_drift",
            "reason": (
                "working-tree projection fingerprint "
                f"{projection['fingerprint']!r} != stored "
                f"{provenance.get('projection_fingerprint')!r} — the catalog "
                "drifted since the canary; claim text would not match"
            ),
        }
    stored = envelope["result"]
    property_key = str(provenance.get("property_key") or "")

    by_unit: Dict[str, List[Dict[str, Any]]] = {}
    for condition in stored["observed_conditions"]:
        by_unit.setdefault(condition["estimate_unit_id"], []).append(condition)
    evidence_by_condition = {
        fact["condition_id"]: fact for fact in stored["evidence_facts"]
    }
    paths = photo_key_to_path(art)
    identity_cache: Dict[str, Any] = {}

    counts = {"verified": 0, "refused": 0, "resumed": 0, "redecided": 0}
    estimated_tokens = 0
    records: List[Dict[str, Any]] = []
    calls = sorted(stored["terra_calls"], key=lambda c: c["estimate_unit_id"])
    if limit_units is not None:
        calls = calls[:limit_units]
    for call in calls:
        unit_id = str(call["estimate_unit_id"])
        base = {
            "schema_version": 1,
            "harness": HARNESS_NAMESPACE,
            "variant_label": variant["label"],
            "property_key": property_key,
            "estimate_unit_id": unit_id,
            "condition_ids": sorted(call["condition_ids"]),
            "stored_request_fingerprint": call["request_fingerprint"],
            "model": call["model"],
            "dry_run": dry_run,
        }
        unit_path = _unit_path(out_dir, unit_id)
        record = _redecide_unit(
            base, call, by_unit.get(unit_id) or [], evidence_by_condition,
            paths, identity_cache, projection, variant,
            max_output_tokens=max_output_tokens, dry_run=dry_run,
            live_ctx=live_ctx, unit_path=unit_path,
            property_key=property_key, estimate_id=str(envelope["estimate_id"]),
        )
        records.append(record)
        status = record["status"]
        if status == "resumed":
            counts["resumed"] += 1
        elif status in ("verified", "redecided"):
            counts[status] += 1
            estimated_tokens += int(record.get("estimated_reservation_tokens") or 0)
        else:
            counts["refused"] += 1
        if status != "resumed":
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            unit_path.write_text(
                json.dumps(record, indent=1, sort_keys=True), encoding="utf-8"
            )
    return {
        "state": "complete",
        "property_key": property_key,
        "counts": counts,
        "estimated_reservation_tokens": estimated_tokens,
        "refusals": [
            {"estimate_unit_id": r["estimate_unit_id"], "status": r["status"]}
            for r in records
            if r["status"] not in ("verified", "redecided", "resumed")
        ],
    }


def _redecide_unit(
    base: Dict[str, Any],
    call: Mapping[str, Any],
    unit_conditions: List[Dict[str, Any]],
    evidence_by_condition: Mapping[str, Dict[str, Any]],
    paths: Mapping[str, Path],
    identity_cache: Dict[str, Any],
    projection: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    max_output_tokens: int,
    dry_run: bool,
    live_ctx: Optional[Dict[str, Any]],
    unit_path: Path,
    property_key: str,
    estimate_id: str,
) -> Dict[str, Any]:
    stored_ids = {c["condition_id"] for c in unit_conditions}
    if stored_ids != set(call["condition_ids"]):
        return _refusal(
            base, "refused_condition_set_mismatch",
            f"stored conditions for the unit {sorted(stored_ids)} != the "
            f"call's condition_ids {sorted(call['condition_ids'])}",
        )

    # Production payload order within a unit is by catalog_item_id (the
    # conditions builder iterates sorted grouped items) — fingerprinted, so
    # the rebuild must sort the same way.
    pairs: List[Tuple[ConditionDraft, EvidenceFacts]] = []
    needed_keys: List[str] = []
    try:
        for condition in sorted(unit_conditions, key=lambda c: c["catalog_item_id"]):
            fact_stored = evidence_by_condition[condition["condition_id"]]
            facts = _from_stored(EvidenceFacts, fact_stored)
            draft = ConditionDraft(
                condition=_from_stored(ObservedCondition, condition),
                evidence_refs=facts.evidence_refs,
            )
            pairs.append((draft, facts))
            needed_keys.extend(facts.representative_photo_keys)
    except KeyError as exc:
        return _refusal(base, "refused_contract_drift", str(exc))

    try:
        for key in sorted(set(needed_keys)):
            if key not in identity_cache:
                identity_cache.update(build_photo_identity_index([key], paths))
    except PassExecutionError as exc:
        return _refusal(base, "refused_photo_unreadable", str(exc))
    identity_index = {key: identity_cache[key] for key in set(needed_keys)}

    request = build_unit_request(
        estimate_unit_id=base["estimate_unit_id"],
        unit_pairs=pairs,
        observables=projection["observables"],
        photo_key_to_path=paths,
        identity_index=identity_index,
        projection_fingerprint=projection["fingerprint"],
        model=str(call["model"]),
        reasoning_effort=TERRA_REVIEW_REASONING_EFFORT,
        max_output_tokens=max_output_tokens,
    )
    base["rebuilt_request_fingerprint"] = request.request_fingerprint
    base["fingerprint_match"] = (
        request.request_fingerprint == call["request_fingerprint"]
    )
    if not base["fingerprint_match"]:
        return _refusal(
            base, "refused_fingerprint_mismatch", FINGERPRINT_MISMATCH_DETAIL
        )

    variant_request = apply_variant(request, variant)
    # The factorized arm also swaps the response schema. That schema is NOT a
    # variant_fingerprint input, but `label` is — and `factorized_v1` is unique
    # — so no arm/arm or arm/production collision is possible. A future arm
    # reusing a system prompt under a different schema must extend the
    # fingerprint rather than rely on this.
    if variant.get("response_contract") == FACTORIZED_CONTRACT:
        variant_request = replace(
            variant_request,
            response_schema=build_factorized_response_schema(
                list(request.condition_ids)
            ),
        )
    base["variant_fingerprint"] = variant_request.request_fingerprint
    base["response_contract"] = str(variant.get("response_contract") or "terra_verdict")

    # Resume: an existing output for this unit under the same variant
    # fingerprint is final — never re-bought, never re-dry-run.
    if unit_path.is_file():
        try:
            previous = json.loads(unit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if (
            isinstance(previous, dict)
            and previous.get("variant_fingerprint")
            == variant_request.request_fingerprint
            and previous.get("status") in ("verified", "redecided")
        ):
            return {**base, "status": "resumed"}

    record = {
        **base,
        "system_prompt": variant_request.system_prompt,
        "user_prompt": variant_request.user_prompt,
        "photo_keys": list(variant_request.photo_keys),
        "image_sha256": {
            key: identity_index[key].exact_sha256
            for key in variant_request.photo_keys
        },
        "request_bytes": variant_request.request_bytes,
        "estimated_reservation_tokens": estimate_reservation_tokens(
            max_output_tokens=max_output_tokens,
            request_bytes=variant_request.request_bytes,
            image_count=len(variant_request.image_paths),
        ),
    }
    if dry_run:
        return {**record, "status": "verified"}

    # Live (Session F): reuse the production reserve -> call -> settle ->
    # parse path verbatim so nothing double-debits and failure keeps the
    # conservative reservation. The factorized arm uses its own mirrored
    # fresh-call (same ordering, different contract). Imported here, never at
    # module level.
    factorized = variant.get("response_contract") == FACTORIZED_CONTRACT
    if factorized:
        from tools.renovation_architecture.factorized_review import (
            FACTOR_KEYS,
            factorized_unit_fresh as unit_fresh,
        )
    else:
        from tools.renovation_architecture.review_pipeline import (
            _review_unit_fresh as unit_fresh,
        )

    terra_call, reviews = unit_fresh(
        request=variant_request,
        vlm_client=live_ctx["vlm_client"],
        api_key=live_ctx["api_key"],
        terra_model=live_ctx["terra_model"],
        terra_max_output_tokens=max_output_tokens,
        ledger=live_ctx["ledger"],
        property_key=property_key,
        source_run_id=f"redecide_{variant['label']}",
        estimate_id=estimate_id,
    )
    if factorized:
        answer_keys = FACTOR_KEYS + (
            "derived_class", "observed_description", "rationale",
        )
    else:
        answer_keys = ("verdict", "rationale")
    return {
        **record,
        "status": "redecided",
        "terra_call": terra_call,
        "reviews": {
            review["condition_id"]: {key: review[key] for key in answer_keys}
            for review in reviews
        },
    }


def _guard_out_root(out_root: Path, canary_root: Path) -> None:
    out = out_root.resolve()
    frozen = canary_root.resolve()
    if (frozen.parent / "input_freeze.json").is_file():
        frozen = frozen.parent  # the whole frozen canary tree, not one run
    for name, protected in (("frozen canary", frozen), ("production", PROD_ROOT)):
        protected = Path(protected).resolve()
        if out == protected or protected in out.parents or out in protected.parents:
            raise SystemExit(
                f"--out-root {out} overlaps the {name} root {protected}; "
                "harness output must be a fresh sibling directory"
            )


def _live_context(out_root: Path, stored_models: set) -> Dict[str, Any]:
    from tools import pipeline_config as cfg
    from tools.renovation_architecture.usage_guard import (
        TERRA_DAILY_TOKEN_CEILING,
        TerraUsageLedger,
    )
    from tools.vlm_client import create_vlm_client

    if not cfg.OPENAI_API_KEY:
        raise SystemExit("--live needs OPENAI_API_KEY")
    model = cfg.RENOVATION_TERRA_MODEL
    if not model:
        raise SystemExit("--live needs RENOVATION_TERRA_MODEL (or OPENAI_MODEL)")
    if stored_models and stored_models != {model}:
        raise SystemExit(
            f"resolved Terra model {model!r} != stored canary call model(s) "
            f"{sorted(stored_models)} — a different model would make every "
            "comparison meaningless"
        )
    client = create_vlm_client()
    client.reset_usage_stats()
    return {
        "vlm_client": client,
        "api_key": cfg.OPENAI_API_KEY,
        "terra_model": model,
        "ledger": TerraUsageLedger(
            out_root,
            daily_ceiling=getattr(
                cfg, "RENOVATION_TERRA_DAILY_TOKEN_CEILING",
                TERRA_DAILY_TOKEN_CEILING,
            ),
            usage_root_override=getattr(cfg, "RENOVATION_TERRA_USAGE_ROOT", None),
        ),
    }


def _stored_models(candidate_root: Path, keys: List[str]) -> set:
    models = set()
    for property_key in keys:
        run_dir = _newest_run_dir(candidate_root / property_key)
        if run_dir is None:
            continue
        _, envelope = _load_envelope(run_dir)
        if envelope and envelope.get("state") == "complete":
            models.update(
                str(call.get("model") or "")
                for call in envelope["result"].get("terra_calls") or []
            )
    return models


def write_report(out_dir: Path, variant: Mapping[str, Any], rows) -> Path:
    lines = [
        f"# Re-decide report — variant `{variant['label']}`",
        "",
        "`est. tokens` is the conservative pre-call RESERVATION ceiling "
        "(output cap + request bytes + per-image charge), not projected "
        "actual spend; canary actuals ran ~51k/listing. Resumed units "
        "report 0 here.",
        "",
        "| property | verified | redecided | resumed | refused | est. tokens |",
        "|---|---|---|---|---|---|",
    ]
    totals = {"verified": 0, "redecided": 0, "resumed": 0, "refused": 0}
    tokens = 0
    problems: List[str] = []
    for row in rows:
        if row["state"] != "complete":
            problems.append(f"{row['property_key']}: {row['state']} — {row.get('reason')}")
            continue
        counts = row["counts"]
        for name in totals:
            totals[name] += counts[name]
        tokens += row["estimated_reservation_tokens"]
        lines.append(
            f"| {row['property_key']} | {counts['verified']} "
            f"| {counts['redecided']} | {counts['resumed']} "
            f"| {counts['refused']} | {row['estimated_reservation_tokens']:,} |"
        )
        problems.extend(
            f"{row['property_key']} {r['estimate_unit_id']}: {r['status']}"
            for r in row["refusals"]
        )
    lines.append(
        f"| **total** | {totals['verified']} | {totals['redecided']} "
        f"| {totals['resumed']} | {totals['refused']} | {tokens:,} |"
    )
    if problems:
        lines += ["", "## Skipped / refused", ""]
        lines += [f"- {problem}" for problem in problems]
    lines.append("")
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("artifacts_canary") / "renovation_session9_20260818" / "run_1",
        help="frozen canary run root (contains candidate/)",
    )
    arm = parser.add_mutually_exclusive_group(required=True)
    arm.add_argument("--variant", type=Path, help="variant spec JSON")
    arm.add_argument("--control", action="store_true",
                     help="the built-in same-prompt control arm")
    arm.add_argument("--factorized", action="store_true",
                     help="the built-in factorized arm: same payload, three "
                          "bounded factors instead of one verdict "
                          "(docs/HANDOFF_factorized_verifier_replay.md)")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--properties", nargs="*", default=None)
    parser.add_argument("--limit-units", type=int, default=None,
                        help="per-property unit cap (smoke)")
    parser.add_argument(
        "--live", action="store_true",
        help="actually call Terra (Session F, post-window); default is dry run",
    )
    args = parser.parse_args(argv)

    if args.factorized:
        variant = dict(FACTORIZED_VARIANT)
    else:
        variant = load_variant(None if args.control else args.variant)
    candidate_root = args.root / "candidate"
    if not candidate_root.is_dir():
        print(f"no candidate directory under {args.root}", file=sys.stderr)
        return 2
    out_root = args.out_root or (
        Path("artifacts_canary")
        / f"redecide_{variant['label']}_{datetime.now(timezone.utc):%Y%m%d}"
    )
    _guard_out_root(out_root, args.root)
    out_root.mkdir(parents=True, exist_ok=True)

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    projection = build_renovation_catalog_projection(
        catalog, catalog_path=CATALOG_PATH
    )
    from tools import pipeline_config as cfg

    max_output_tokens = cfg.RENOVATION_TERRA_MAX_OUTPUT_TOKENS
    keys = args.properties or sorted(
        entry.name for entry in candidate_root.iterdir() if entry.is_dir()
    )
    live_ctx = None
    if args.live:
        live_ctx = _live_context(out_root, _stored_models(candidate_root, keys))

    rows: List[Dict[str, Any]] = []
    for property_key in keys:
        run_dir = _newest_run_dir(candidate_root / property_key)
        if run_dir is None:
            rows.append({"property_key": property_key, "state": "skipped",
                         "reason": "no run dir with photo_intel_debug.json"})
            continue
        if live_ctx is not None:
            live_ctx["vlm_client"].budget_context = {
                "property_key": property_key
            }
        try:
            row = redecide_property(
                run_dir, projection, variant,
                max_output_tokens=max_output_tokens,
                out_dir=out_root / property_key,
                dry_run=not args.live,
                live_ctx=live_ctx,
                limit_units=args.limit_units,
            )
        except Exception as exc:  # surface, never silently drop a property
            rows.append({"property_key": property_key, "state": "failed",
                         "reason": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {property_key}: {exc}", file=sys.stderr)
            continue
        row["property_key"] = property_key
        rows.append(row)
        counts = row.get("counts") or {}
        print(f"{property_key}: {row['state']} {counts}")

    manifest = {
        "schema_version": 1,
        "harness": HARNESS_NAMESPACE,
        "variant": dict(variant),
        "response_contract": str(
            variant.get("response_contract") or "terra_verdict"
        ),
        "prompt_version": (
            FACTORIZED_PROMPT_VERSION
            if variant.get("response_contract") == FACTORIZED_CONTRACT
            else TERRA_REVIEW_PROMPT_VERSION
        ),
        "root": str(args.root),
        "dry_run": not args.live,
        "catalog_sha256": projection["catalog_sha256"],
        "projection_fingerprint": projection["fingerprint"],
        "max_output_tokens": max_output_tokens,
        "reasoning_effort": TERRA_REVIEW_REASONING_EFFORT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8"
    )
    report_path = write_report(out_root, variant, rows)
    print(f"report: {report_path}")
    failed = [row for row in rows if row["state"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
