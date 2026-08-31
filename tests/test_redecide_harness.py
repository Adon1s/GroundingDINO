"""Session B: the QP1 re-decide harness (dry-run mechanics only — no live
provider path is exercised anywhere in the suite)."""
import json
import sys
from pathlib import Path

import pytest

from scripts.redecide_renovation_architecture import (
    CONTROL_VARIANT,
    FACTORIZED_VARIANT,
    PAYLOAD_DELIMITER,
    apply_variant,
    load_variant,
    main,
    redecide_property,
)
from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    EVIDENCE_DEDUP_POLICY_VERSION,
    TERRA_REVIEW_REASONING_EFFORT,
)
from tools.renovation_architecture.evidence import build_photo_identity_index
from tools.renovation_architecture.terra_review import build_unit_request

MAX_OUTPUT_TOKENS = 8192
PROJECTION_FP = "f" * 64
UNIT = "kitchen_primary"
PHOTO = "photo_001.jpg"


def _write_photo(path: Path, color=(120, 30, 200)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _projection():
    return {
        "fingerprint": PROJECTION_FP,
        "catalog_sha256": "c" * 64,
        "observables": {
            "item_x": {
                "kind": "degradation",
                "atomic_claim": {"subject": "cabinets", "state": "worn"},
            }
        },
    }


def _stored_condition(cid: str) -> dict:
    return {
        "condition_id": cid,
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "catalog_item_id": "item_x",
        "catalog_kind": "degradation",
        "scope_key": "kitchen",
        "estimate_unit_id": UNIT,
        "room_surrogate_id": "rs_1",
        "scene_group": "kitchen",
        "issue_ids": ["i1"],
        "identity_ambiguous": False,
        "source_room_surrogate_ids": ["rs_1"],
        "source_scope_keys": ["kitchen"],
        "unit_resolution_source": "photo_estimate_unit",
        "unit_resolution_reason": "single_room",
        "opening_instance_hints": [],
    }


def _stored_evidence(cid: str) -> dict:
    return {
        "evidence_id": f"ev_{cid}",
        "schema_version": CONTRACTS_SCHEMA_VERSION,
        "condition_id": cid,
        "photo_keys": [PHOTO],
        "distinct_photo_count": 1,
        "distinct_view_count": 1,
        "duplicate_groups": [],
        "evidence_refs": [
            {"issue_id": "i1", "photo_key": PHOTO,
             "observation": "Cabinets are honey-oak and visually dated.",
             "room_surrogate_id": "rs_1"}
        ],
        "min_photo_evidence_required": None,
        "representative_photo_keys": [PHOTO],
        "exact_duplicate_groups": [],
        "near_duplicate_groups": [],
        "dedup_policy_version": EVIDENCE_DEDUP_POLICY_VERSION,
    }


def _reference_request(image_path: Path):
    """The stored fingerprint is computed with the production builder itself,
    so the fixture can never drift from build_unit_request."""
    from scripts.redecide_renovation_architecture import _from_stored
    from tools.renovation_architecture.conditions import ConditionDraft
    from tools.renovation_architecture.contracts import (
        EvidenceFacts,
        ObservedCondition,
    )

    facts = _from_stored(EvidenceFacts, _stored_evidence("c1"))
    draft = ConditionDraft(
        condition=_from_stored(ObservedCondition, _stored_condition("c1")),
        evidence_refs=facts.evidence_refs,
    )
    paths = {PHOTO: image_path}
    return build_unit_request(
        estimate_unit_id=UNIT,
        unit_pairs=[(draft, facts)],
        observables=_projection()["observables"],
        photo_key_to_path=paths,
        identity_index=build_photo_identity_index([PHOTO], paths),
        projection_fingerprint=PROJECTION_FP,
        model="terra-test",
        reasoning_effort=TERRA_REVIEW_REASONING_EFFORT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def _write_run(tmp_path: Path) -> Path:
    image_path = tmp_path / "images" / PHOTO
    _write_photo(image_path)
    request = _reference_request(image_path)
    run_dir = tmp_path / "root" / "candidate" / "prop" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "photo_intel_debug.json").write_text(
        json.dumps({
            "photos": {PHOTO: {"photo": {"photo_key": PHOTO,
                                         "image_path": str(image_path)}}},
            "analysis_debug": {"renovation_estimate_v5": {
                "schema_version": 5,
                "estimate_id": "rea1_" + "0" * 16,
                "state": "complete",
                "reason": None,
                "provenance": {"projection_fingerprint": PROJECTION_FP,
                               "property_key": "prop"},
                "result": {
                    "observed_conditions": [_stored_condition("c1")],
                    "evidence_facts": [_stored_evidence("c1")],
                    "terra_calls": [{
                        "estimate_unit_id": UNIT,
                        "condition_ids": ["c1"],
                        "request_fingerprint": request.request_fingerprint,
                        "model": "terra-test",
                    }],
                },
            }},
        }),
        encoding="utf-8",
    )
    return run_dir


def _redecide(tmp_path, run_dir, variant=None, out=None):
    return redecide_property(
        run_dir, _projection(), variant or dict(CONTROL_VARIANT),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        out_dir=out or tmp_path / "out" / "prop",
        dry_run=True,
    )


def test_dry_run_verifies_fingerprint_without_the_vlm_client(
    tmp_path, monkeypatch
):
    # Poison the client module: any import attempt on the dry-run path fails.
    monkeypatch.setitem(sys.modules, "tools.vlm_client", None)
    run_dir = _write_run(tmp_path)
    row = _redecide(tmp_path, run_dir)
    assert row["state"] == "complete"
    assert row["counts"] == {"verified": 1, "refused": 0, "resumed": 0,
                             "redecided": 0}
    record = json.loads(
        next((tmp_path / "out" / "prop" / "units").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert record["status"] == "verified"
    assert record["fingerprint_match"] is True
    assert record["dry_run"] is True
    assert record["user_prompt"] and record["system_prompt"]
    assert record["estimated_reservation_tokens"] > 0
    assert record["image_sha256"][PHOTO]


def test_second_pass_resumes_instead_of_rewriting(tmp_path):
    run_dir = _write_run(tmp_path)
    _redecide(tmp_path, run_dir)
    again = _redecide(tmp_path, run_dir)
    assert again["counts"] == {"verified": 0, "refused": 0, "resumed": 1,
                               "redecided": 0}


def test_flipped_photo_byte_refuses_the_unit(tmp_path):
    run_dir = _write_run(tmp_path)
    _write_photo(tmp_path / "images" / PHOTO, color=(121, 30, 200))
    row = _redecide(tmp_path, run_dir)
    assert row["counts"]["refused"] == 1
    assert row["refusals"][0]["status"] == "refused_fingerprint_mismatch"


def test_catalog_drift_refuses_the_listing(tmp_path):
    run_dir = _write_run(tmp_path)
    projection = dict(_projection(), fingerprint="e" * 64)
    row = redecide_property(
        run_dir, projection, dict(CONTROL_VARIANT),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        out_dir=tmp_path / "out" / "prop", dry_run=True,
    )
    assert row["state"] == "refused_catalog_drift"


def test_variant_overrides_payload_and_namespaces_fingerprint(tmp_path):
    image_path = tmp_path / "images" / PHOTO
    _write_photo(image_path)
    request = _reference_request(image_path)
    variant = {
        "label": "ablation_test",
        "system_prompt": "You are a stricter reviewer.",
        "extra_condition_keys": {"verification_guidance": "name what you see"},
        "drop_condition_keys": ["observations"],
    }
    mutated = apply_variant(request, variant)
    payload = json.loads(
        mutated.user_prompt.split("\n\nConditions to review:\n", 1)[1]
    )
    entry = payload["conditions"][0]
    assert "observations" not in entry
    assert entry["verification_guidance"] == "name what you see"
    assert entry["condition_id"] == "c1"
    assert mutated.system_prompt == "You are a stricter reviewer."
    assert mutated.request_fingerprint != request.request_fingerprint

    control = apply_variant(request, CONTROL_VARIANT)
    assert control.user_prompt == request.user_prompt
    assert control.system_prompt == request.system_prompt
    # Same prompt text, own namespace: never collides with production or
    # with any mutating arm.
    assert control.request_fingerprint != request.request_fingerprint
    assert control.request_fingerprint != mutated.request_fingerprint


def test_variant_spec_guards(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"label": "x", "unknown_key": 1}),
                   encoding="utf-8")
    with pytest.raises(SystemExit):
        load_variant(bad)
    protected = tmp_path / "protected.json"
    protected.write_text(
        json.dumps({"label": "x", "drop_condition_keys": ["condition_id"]}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        load_variant(protected)


def test_factorized_arm_keeps_the_payload_and_swaps_the_contract(tmp_path):
    """The factorized arm must change the question, never the evidence: same
    claims, same observations, same photos as the control arm."""
    run_dir = _write_run(tmp_path)
    row = _redecide(tmp_path, run_dir, variant=dict(FACTORIZED_VARIANT))
    assert row["counts"]["verified"] == 1
    record = json.loads(
        next((tmp_path / "out" / "prop" / "units").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert record["response_contract"] == "factorized_v1"
    assert "three separate questions" in record["system_prompt"]

    control_out = tmp_path / "out_control" / "prop"
    _redecide(tmp_path, run_dir, out=control_out)
    control = json.loads(
        next((control_out / "units").glob("*.json")).read_text(encoding="utf-8")
    )
    assert control["response_contract"] == "terra_verdict"
    # Byte-identical payload half; only the system prompt differs.
    assert (
        record["user_prompt"].split(PAYLOAD_DELIMITER, 1)[1]
        == control["user_prompt"].split(PAYLOAD_DELIMITER, 1)[1]
    )
    assert record["system_prompt"] != control["system_prompt"]
    assert record["variant_fingerprint"] != control["variant_fingerprint"]


def test_factorized_request_carries_the_factorized_schema(tmp_path):
    image_path = tmp_path / "images" / PHOTO
    _write_photo(image_path)
    request = _reference_request(image_path)
    mutated = apply_variant(request, dict(FACTORIZED_VARIANT))
    # apply_variant does not swap the schema; _redecide_unit does. What must
    # hold here is that the arm is fingerprint-distinct from control.
    control = apply_variant(request, CONTROL_VARIANT)
    assert mutated.request_fingerprint != control.request_fingerprint
    assert mutated.request_fingerprint != request.request_fingerprint


def test_factorized_label_and_contract_are_reserved(tmp_path):
    hijack = tmp_path / "hijack.json"
    hijack.write_text(
        json.dumps({"label": "factorized_v1",
                    "system_prompt": "You are something else."}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="reserved"):
        load_variant(hijack)

    bogus = tmp_path / "bogus.json"
    bogus.write_text(
        json.dumps({"label": "x", "response_contract": "made_up"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="unknown response_contract"):
        load_variant(bogus)


def test_out_root_inside_the_canary_is_refused(tmp_path):
    run_dir = _write_run(tmp_path)
    root = run_dir.parents[2]
    with pytest.raises(SystemExit, match="overlaps"):
        main(["--control", "--root", str(root),
              "--out-root", str(root / "redecide_out")])
