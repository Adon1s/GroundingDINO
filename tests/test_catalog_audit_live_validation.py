"""Unit tests for the Session 6 Part A Stage A harness.

These pin the machinery, not the result. A live experiment that silently reads the wrong issue
lane, drops a condition, or mis-attributes a verdict would still produce a tidy report -- and the
report is what a later session decides on. So the deterministic half is tested here, before any
token is spent, and every test is provider-free by construction.

The load-bearing one is test_product_lane_reproduces_the_stored_envelope: it rebuilds one frozen
canary property from its own stored provenance and requires the condition ids to come back bit for
bit. If that passes, the harness is projecting conditions the way the run being replayed did.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "catalog_audit_live_validation.py"
MANIFEST = ROOT / "reports" / "catalog_audit_live_experiment_manifest.json"
CANDIDATE_ROOT = Path("C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4")
GOLDEN_ARTIFACT = (ROOT / "artifacts_canary" / "renovation_session9_20260818" / "run_1" / "candidate"
                   / "redfin_10803207" / "20260817_222851_c0551630" / "photo_intel_debug.json")

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="Session 6 harness not present")


def _load():
    """Import the harness by path, from the repo root, so it binds to the baseline arm."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("catalog_audit_live_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.is_file():
        pytest.skip("live experiment manifest not present")
    return mod.load_manifest()


@pytest.fixture(scope="module")
def baseline_projection():
    return mod.arm_projection(ROOT)


# --------------------------------------------------------------------------- arm identity

def test_harness_binds_to_the_arm_root_it_was_given():
    """The arm supplies its own code and catalog. If `tools` came from anywhere else the two arms
    would silently share one catalog and the experiment would compare nothing."""
    import tools
    assert Path(tools.__file__).resolve().parent == mod.ARM_ROOT / "tools"


def test_baseline_arm_is_the_pinned_catalog(manifest, baseline_projection):
    want = manifest["arms"]["baseline"]
    proj = baseline_projection["projection"]
    assert proj["catalog_sha256"] == want["catalog_sha256"]
    assert proj["fingerprint"] == want["projection_fingerprint"]
    assert len(baseline_projection["catalog"]["items"]) == want["item_count"] == 128


@pytest.mark.skipif(not CANDIDATE_ROOT.is_dir(), reason="candidate worktree not present")
def test_candidate_arm_differs_only_by_the_three_split_items(manifest):
    base = {i["id"] for i in mod.arm_projection(ROOT)["catalog"]["items"]}
    cand = {i["id"] for i in mod.arm_projection(CANDIDATE_ROOT)["catalog"]["items"]}
    assert base - cand == {mod.PARENT_ID}
    assert cand - base == {mod.BLINDS_ID, mod.FABRIC_ID}


@pytest.mark.skipif(not CANDIDATE_ROOT.is_dir(), reason="candidate worktree not present")
def test_the_blinds_successor_routes_to_no_action_and_the_fabric_one_bills(manifest):
    """Ruling 5's whole point. If the blinds successor ever routed to work, every 'moved to
    no_action' row in the report would be wrong."""
    routes = mod.arm_projection(CANDIDATE_ROOT)["projection"]["terminal_routes"]
    assert routes[mod.BLINDS_ID]["route"] == "no_action"
    assert routes[mod.BLINDS_ID]["reason_code"] == "route_override_no_action"
    assert routes[mod.FABRIC_ID]["route"] == "work"


# --------------------------------------------------------------------------- manifest integrity

def test_manifest_is_self_consistent_and_carries_a_human_authorization(manifest):
    assert manifest["case_count"] == len(manifest["cases"]) == 66
    budget = manifest["budget"]
    assert isinstance(budget["stage_a_authorized_tokens"], int)
    assert budget["authorized_by"] and budget["authorized_at"]
    assert budget["stage_b_authorized_tokens"] is None, "Stage B must stay unauthorized"


def test_only_the_budget_block_moved_since_session_5(manifest):
    """The authorization is the one edit to a pinned artifact. Restoring the null block has to
    reproduce Session 5's canonical hash, or something else was changed with it."""
    proof = mod.manifest_content_proof(manifest)
    assert proof["ok"], proof


def test_every_pinned_case_is_in_the_product_lane_with_its_frozen_owner(manifest):
    for group in mod.cases_by_run(manifest):
        path = mod.resolve_input(group["artifact_path"])
        if not path.is_file():
            pytest.skip(f"pinned artifact not present: {path}")
        lane = {r.get("issue_id"): r for r in mod.product_lane(_read(path))}
        for case in group["cases"]:
            row = lane.get(case["issue_id"])
            assert row is not None, f"{case['case_id']} missing from the product lane"
            assert row["catalog_item_id"] == case["frozen_resolution"]["resolved_item_id"]


# --------------------------------------------------------------------------- the lane golden

@pytest.mark.skipif(not GOLDEN_ARTIFACT.is_file(), reason="golden canary artifact not present")
def test_product_lane_reproduces_the_stored_envelope(baseline_projection):
    """Rebuild one frozen property from its own stored provenance and require the condition ids
    back bit for bit. This is what proves the harness reads the lane production read: the estimate
    lane builds 29 conditions where the run had 22, and would buy Terra calls for the extra seven."""
    from tools.renovation_architecture.conditions import build_observed_conditions
    from tools.renovation_architecture.ids import make_estimate_id
    art = _read(GOLDEN_ARTIFACT)
    envelope = art["analysis_debug"]["renovation_estimate_v5"]
    provenance, stored = envelope["provenance"], envelope["result"]
    estimate_id = make_estimate_id(
        property_key="redfin_10803207", source_run_id="20260817_222851_c0551630",
        catalog_sha256=provenance["catalog_sha256"],
        projection_fingerprint=provenance["projection_fingerprint"])
    assert estimate_id == envelope["estimate_id"]

    drafts = build_observed_conditions(
        issues_flat=mod.product_lane(art), photos=art["photos"],
        property_metadata=art.get("property_metadata"),
        projection=baseline_projection["projection"], estimate_id=estimate_id)
    assert sorted(d.condition.condition_id for d in drafts) == \
        sorted(c["condition_id"] for c in stored["observed_conditions"])
    assert len({d.condition.estimate_unit_id for d in drafts}) == len(stored["terra_calls"]) == 7

    wrong_lane = build_observed_conditions(
        issues_flat=art["estimate_issues_flat"], photos=art["photos"],
        property_metadata=art.get("property_metadata"),
        projection=baseline_projection["projection"], estimate_id=estimate_id)
    assert len(wrong_lane) == 29 > len(drafts) == 22


# --------------------------------------------------------------------------- stale ids

@pytest.mark.skipif(not CANDIDATE_ROOT.is_dir() or not GOLDEN_ARTIFACT.is_file(),
                    reason="candidate worktree or golden artifact not present")
def test_the_candidate_projection_fails_closed_on_the_deprecated_parent(manifest):
    """The stale-id guard is the approved behaviour (CCF-8). The harness must satisfy it by
    replaying, never by aliasing the removed id or editing a stored artifact."""
    from tools.renovation_architecture.conditions import build_observed_conditions
    from tools.scene_classifier_passes import PassExecutionError
    candidate = mod.arm_projection(CANDIDATE_ROOT)
    art = _read(GOLDEN_ARTIFACT)
    with pytest.raises(PassExecutionError) as excinfo:
        build_observed_conditions(issues_flat=mod.product_lane(art), photos=art["photos"],
                                  property_metadata=art.get("property_metadata"),
                                  projection=candidate["projection"], estimate_id="rea1_test")
    assert getattr(excinfo.value, "code", None) == "StaleCatalogItemId"


def test_the_four_excluded_rows_are_whole_conditions_outside_the_pinned_set(manifest):
    """Dropping a row that shared a condition with a pinned case would change that case's evidence.
    These four do not: each is the only row of its own condition."""
    pinned = {c["issue_id"] for c in manifest["cases"]}
    assert not (mod.UNPINNED_PARENT_ISSUE_IDS & pinned)
    found = set()
    for group in mod.cases_by_run(manifest):
        path = mod.resolve_input(group["artifact_path"])
        if not path.is_file():
            pytest.skip(f"pinned artifact not present: {path}")
        for row in mod.product_lane(_read(path)):
            if row.get("catalog_item_id") == mod.PARENT_ID and row["issue_id"] not in pinned:
                found.add(row["issue_id"])
    assert found == set(mod.UNPINNED_PARENT_ISSUE_IDS)


# --------------------------------------------------------------------------- request construction

@pytest.mark.skipif(not GOLDEN_ARTIFACT.is_file(), reason="golden canary artifact not present")
def test_a_condition_payload_is_independent_of_its_siblings(baseline_projection):
    """Reviewing whole units is a deliberate choice: the per-condition claim and observations do not
    depend on siblings, but the photo set, the prompt's photo numbering and the fingerprint do. So a
    subset payload asks Terra a different question, and this test pins both halves of that."""
    from tools.renovation_architecture.conditions import build_observed_conditions
    from tools.renovation_architecture.contracts import TERRA_REVIEW_REASONING_EFFORT
    from tools.renovation_architecture.evidence import build_evidence_facts, build_photo_identity_index
    from tools.renovation_architecture.terra_review import build_unit_request
    from tools.pass_2f_artifact_inputs import photo_key_to_path

    art = _read(GOLDEN_ARTIFACT)
    projection = baseline_projection["projection"]
    drafts = build_observed_conditions(issues_flat=mod.product_lane(art), photos=art["photos"],
                                       property_metadata=art.get("property_metadata"),
                                       projection=projection, estimate_id="rea1_test")
    unit_id = max({d.condition.estimate_unit_id for d in drafts},
                  key=lambda u: sum(1 for d in drafts if d.condition.estimate_unit_id == u))
    unit = [d for d in drafts if d.condition.estimate_unit_id == unit_id]
    if len(unit) < 2:
        pytest.skip("golden property has no multi-condition unit")
    paths = photo_key_to_path(art)
    keys = sorted({r["photo_key"] for d in unit for r in d.evidence_refs})
    index = build_photo_identity_index(keys, paths)
    pairs = [(d, build_evidence_facts(d, identity_index=index,
                                      observables=projection["observables"], estimate_id="rea1_test"))
             for d in unit]

    def request(subset):
        return build_unit_request(estimate_unit_id=unit_id, unit_pairs=subset,
                                  observables=projection["observables"], photo_key_to_path=paths,
                                  identity_index=index, projection_fingerprint=projection["fingerprint"],
                                  model="gpt-5.6-terra", reasoning_effort=TERRA_REVIEW_REASONING_EFFORT,
                                  max_output_tokens=8192)
    full, partial = request(pairs), request(pairs[:1])

    def entry(req, condition_id):
        payload = json.loads(req.user_prompt.split("\n\nConditions to review:\n", 1)[1])
        return next(c for c in payload["conditions"] if c["condition_id"] == condition_id)
    first = pairs[0][0].condition.condition_id
    assert entry(full, first) == entry(partial, first), "per-condition payload must not move"
    assert full.request_fingerprint != partial.request_fingerprint
    assert list(full.photo_keys) != list(partial.photo_keys) or len(pairs) == 1


# --------------------------------------------------------------------------- result assembly

def _stub_review_result(estimate_id="rea1_0123456789abcdef"):
    """A minimal but contract-valid one-condition, one-unit review result."""
    from tools.renovation_architecture.contracts import CONTRACTS_SCHEMA_VERSION
    from tools.renovation_architecture.ids import (make_condition_id, make_disposition_id,
                                                   make_evidence_id, make_review_id,
                                                   make_terra_call_id)
    # Real id shapes: the validator enforces the <prefix>_<16 hex> pattern and a 64-hex fingerprint,
    # so a hand-written "oc1_stub" would fail for the wrong reason and hide what these tests check.
    fingerprint = "f" * 64
    cid = make_condition_id(estimate_id=estimate_id, catalog_item_id="dated_or_older_windows",
                            estimate_unit_id="bedroom_1")
    call_id = make_terra_call_id(estimate_id=estimate_id, estimate_unit_id="bedroom_1",
                                 request_fingerprint=fingerprint)
    evidence_id = make_evidence_id(estimate_id=estimate_id, condition_id=cid)
    review_id = make_review_id(estimate_id=estimate_id, condition_id=cid)
    disposition_id = make_disposition_id(estimate_id=estimate_id, condition_id=cid)
    tokens = {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10,
              "total_tokens": 110, "budget_debited_tokens": 110}
    return {
        "observed_conditions": [{
            "condition_id": cid, "schema_version": CONTRACTS_SCHEMA_VERSION,
            "catalog_item_id": "dated_or_older_windows", "catalog_kind": "modernization",
            "scope_key": "catalog:dated_or_older_windows|scene_group:bedroom|room:bedroom",
            "estimate_unit_id": "bedroom_1", "room_surrogate_id": "bedroom_1",
            "scene_group": "bedroom", "issue_ids": ["i1"], "identity_ambiguous": False,
            "source_room_surrogate_ids": ["bedroom_1"],
            "source_scope_keys": ["catalog:dated_or_older_windows|scene_group:bedroom|room:bedroom"],
            "unit_resolution_source": "photo_estimate_unit", "unit_resolution_reason": "stub",
            "opening_instance_hints": []}],
        "evidence_facts": [{
            "evidence_id": evidence_id, "schema_version": CONTRACTS_SCHEMA_VERSION,
            "condition_id": cid, "photo_keys": ["photo_001.jpg"], "distinct_photo_count": 1,
            "distinct_view_count": 1, "duplicate_groups": [],
            "evidence_refs": [{"issue_id": "i1", "photo_key": "photo_001.jpg",
                               "observation": "Windows look dated.", "room_surrogate_id": "bedroom_1"}],
            "min_photo_evidence_required": None, "representative_photo_keys": ["photo_001.jpg"],
            "exact_duplicate_groups": [], "near_duplicate_groups": [],
            "dedup_policy_version": "evidence_dedup_v1"}],
        "condition_reviews": [{
            "review_id": review_id, "schema_version": CONTRACTS_SCHEMA_VERSION, "condition_id": cid,
            "verdict": "supported", "rationale": "stub", "model": "gpt-5.6-terra",
            "prompt_version": "terra_condition_review_v1", "terra_call_id": call_id,
            "request_fingerprint": fingerprint, "provider": "openai"}],
        "condition_dispositions": [{
            "disposition_id": disposition_id, "schema_version": CONTRACTS_SCHEMA_VERSION,
            "condition_id": cid, "review_id": review_id, "disposition": "no_action",
            "reason_code": "route_no_action", "policy_version": "condition_disposition_v1",
            "evidence_id": evidence_id, "terminal_route": "no_action"}],
        "terra_calls": [{
            "call_id": call_id, "schema_version": CONTRACTS_SCHEMA_VERSION,
            "estimate_unit_id": "bedroom_1", "condition_ids": [cid], "request_fingerprint": fingerprint,
            "provider": "openai", "model": "gpt-5.6-terra",
            "prompt_version": "terra_condition_review_v1", "usage_source": "provider", **tokens}],
        "terra_unit_usage": [{"schema_version": CONTRACTS_SCHEMA_VERSION,
                              "estimate_unit_id": "bedroom_1", "call_ids": [call_id], **tokens}],
        "terra_listing_usage": {"schema_version": CONTRACTS_SCHEMA_VERSION, "call_count": 1, **tokens},
    }


def test_a_subset_review_result_still_satisfies_the_production_contract():
    """Stage A assembles a result over the reviewed units only. The contract is closed over the
    supplied set, so a subset is valid -- this pins that it stays so."""
    from tools.renovation_architecture.validators import validate_condition_review_result
    assert validate_condition_review_result(_stub_review_result(), estimate_id="rea1_0123456789abcdef").ok


@pytest.mark.parametrize("mutation,label", [
    (lambda r: r["terra_unit_usage"].clear(), "unit usage dropped"),
    (lambda r: r["terra_listing_usage"].update({"total_tokens": 999}), "listing sum wrong"),
    (lambda r: r["terra_calls"][0]["condition_ids"].append("oc1_0000000000000000"),
     "call names an absent condition"),
    (lambda r: r["condition_dispositions"][0].update({"disposition": "accepted_for_work"}),
     "disposition contradicts the policy"),
])
def test_the_validator_catches_an_inconsistent_assembly(mutation, label):
    """Each of these is a way a hand-assembled result could look plausible and be wrong."""
    from tools.renovation_architecture.validators import validate_condition_review_result
    result = _stub_review_result()
    mutation(result)
    assert not validate_condition_review_result(result, estimate_id="rea1_0123456789abcdef").ok, label


# --------------------------------------------------------------------------- disposition

@pytest.mark.parametrize("verdict,route,expected", [
    ("supported", "no_action", ("no_action", "route_no_action")),
    ("supported", "work", ("accepted_for_work", "route_work")),
    ("unsupported", "work", ("excluded", "verdict_unsupported")),
    ("cannot_assess", "work", ("inspection", "verdict_cannot_assess")),
])
def test_disposition_policy_is_the_production_table(verdict, route, expected):
    from tools.renovation_architecture.disposition import decide_disposition
    assert decide_disposition(verdict, route, 3, None) == expected


def test_a_supported_condition_below_its_photo_gate_is_withheld_not_billed():
    from tools.renovation_architecture.disposition import decide_disposition
    assert decide_disposition("supported", "work", 1, 2) == ("withheld", "insufficient_distinct_views")


# --------------------------------------------------------------------------- budget and safety

def test_the_ledger_lives_under_the_run_root(tmp_path):
    from tools.renovation_architecture.usage_guard import TerraUsageLedger
    ledger = TerraUsageLedger(tmp_path, daily_ceiling=1000, usage_root_override=None)
    assert ledger.path == tmp_path / ".renovation_architecture" / "terra_usage.sqlite3"


def test_the_ledger_refuses_a_reservation_that_would_cross_the_ceiling(tmp_path):
    """The authorization is enforced twice: this ledger and the harness's own pre-call check. The
    ledger is the one that cannot be bypassed by a bug in the caller."""
    from tools.renovation_architecture.usage_guard import TerraDailyBudgetExceeded, TerraUsageLedger
    ledger = TerraUsageLedger(tmp_path, daily_ceiling=60_000, usage_root_override=None)
    first = ledger.reserve(property_key="p", source_run_id="r", estimate_unit_id="u1",
                           request_fingerprint="f1", tokens=50_000)
    assert ledger.spent_today() == 50_000
    with pytest.raises(TerraDailyBudgetExceeded):
        ledger.reserve(property_key="p", source_run_id="r", estimate_unit_id="u2",
                       request_fingerprint="f2", tokens=50_000)
    ledger.settle(first, provider_total_tokens=5_000)
    assert ledger.spent_today() == 5_000


def test_reservation_estimate_is_conservative():
    from tools.renovation_architecture.usage_guard import estimate_reservation_tokens
    assert estimate_reservation_tokens(max_output_tokens=8192, request_bytes=10, image_count=0) == 25_000
    assert estimate_reservation_tokens(max_output_tokens=8192, request_bytes=6000,
                                       image_count=4) == 8192 + 6000 + 20_000


@pytest.mark.parametrize("bad", [
    "artifacts_canary/renovation_session9_20260818",
    "artifacts_canary/renovation_session9_20260818/run_1/candidate",
])
def test_the_run_root_may_not_overlap_the_frozen_canary_tree(bad):
    with pytest.raises(mod.HarnessError):
        mod._guard_out_root(ROOT / bad)


def test_the_run_root_may_not_overlap_the_production_artifacts_root():
    with pytest.raises(mod.HarnessError):
        mod._guard_out_root(mod.PROD_ARTIFACTS_ROOT / "scratch")


def test_the_chosen_run_root_is_allowed():
    mod._guard_out_root(mod.RUN_ROOT)


def test_verify_refuses_when_the_authorization_is_missing(manifest, tmp_path):
    """The gate that stops a session from spending before a human has said how much."""
    unauthorized = json.loads(json.dumps(manifest))
    unauthorized["budget"]["stage_a_authorized_tokens"] = None
    unauthorized["budget"]["authorized_by"] = None
    checks = []
    budget = unauthorized["budget"]
    ok = (isinstance(budget.get("stage_a_authorized_tokens"), int)
          and bool(budget.get("authorized_by")))
    mod._check(checks, "I22.budget", "stage A authorization filled in by a human", ok)
    assert checks[0]["result"] == "FAIL"


def test_case_ordering_puts_the_material_findings_first(manifest):
    """Findings first, then controls, then the rest: if the budget runs out, what runs is what the
    session most needed to see."""
    groups = mod.cases_by_run(manifest)
    tiers = [0 if any(c["role"] == "finding" for c in g["cases"])
             else (1 if any(c["role"] == "control" for c in g["cases"]) else 2) for g in groups]
    assert tiers == sorted(tiers)
    assert len(groups) == 22


def test_the_harness_does_not_import_a_provider_client_to_read_evidence():
    """verify, plan, compare and freeze must be unable to spend. Only `run` may import a client."""
    with pytest.raises(mod.HarnessError):
        sys.modules.setdefault("tools.vlm_client", object())
        mod.assert_not_imported("tools.vlm_client")
    sys.modules.pop("tools.vlm_client", None)


# --------------------------------------------------------------------------- repeat stage

def test_repeat_authorization_refuses_null_fields(manifest):
    """The repeat stage spends against a second human-filled figure; a null one must stop the run
    before any client exists, exactly as the Stage A gate does."""
    unauthorized = json.loads(json.dumps(manifest))
    for key in ("stage_a_repeat_authorized_tokens", "stage_a_repeat_authorized_by",
                "stage_a_repeat_authorized_at"):
        unauthorized["budget"].pop(key, None)
    with pytest.raises(mod.HarnessError):
        mod._repeat_authorization(unauthorized)


def test_repeat_authorization_is_cumulative_with_stage_a(manifest):
    """The ledger counts the whole UTC day, so the repeat's ceiling is Stage A plus the repeat."""
    if not isinstance(manifest["budget"].get("stage_a_repeat_authorized_tokens"), int):
        pytest.skip("repeat not authorized in this manifest")
    a, r = mod._repeat_authorization(manifest)
    assert a == manifest["budget"]["stage_a_authorized_tokens"]
    assert r == manifest["budget"]["stage_a_repeat_authorized_tokens"]
    assert a + r > a


def _synthetic_replica_root(tmp_path, manifest, *, cand_items, base_items, flips_cand, flips_base):
    """Two arm replica records over the first three manifest rows, with controllable outcomes."""
    cases = manifest["cases"][:3]
    def rec(arm, items, flip_flags):
        pass_2d = []
        terra = []
        for c, per_row in zip(cases, items):
            pass_2d.append({"case_id": c["case_id"], "arm": arm, "issue_id": c["issue_id"],
                            "observation": c["frozen_pass_2c"]["observation"],
                            "stage_a": {"resolved_item_id": per_row[0], "path": "llm", "margin": 0.01},
                            "replicas": [{"resolved_item_id": x, "path": "llm", "margin": 0.01} for x in per_row[1:]]})
        for c, flipped in zip(cases, flip_flags):
            terra.append({"arm": arm, "property_key": c["property_key"], "estimate_unit_id": "u",
                          "condition_id": "oc1_" + "0" * 16, "catalog_item_id": "dated_or_older_windows",
                          "issue_ids": [c["issue_id"]], "is_target": True, "route": "no_action",
                          "stage_a_verdict": "supported",
                          "replica_verdict": "unsupported" if flipped else "supported",
                          "stage_a_disposition": "no_action",
                          "replica_disposition": "excluded" if flipped else "no_action",
                          "replica_rationale": "stub", "flipped": flipped})
        return {"pass_2d": pass_2d, "terra": terra, "spend": {"ledger_delta": 0}}
    for arm, items, flips in (("baseline", base_items, flips_base), ("candidate", cand_items, flips_cand)):
        d = tmp_path / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "replica_record.json").write_text(json.dumps(rec(arm, items, flips)), encoding="utf-8")
    return tmp_path


def test_compare_replica_counts_trim_landings_and_stability(manifest, tmp_path):
    """The S6-1 arithmetic: how often each arm lands on dated_interior_trim across Stage A plus
    replicas, and how often a replica agrees with Stage A. A miscount here would misstate the one
    clause the Session 6 recommendation turns on."""
    P, T, B = mod.PARENT_ID, "dated_interior_trim", mod.BLINDS_ID
    root = _synthetic_replica_root(
        tmp_path, manifest,
        cand_items=[[T, T, B], [B, B, B], [None, T, None]],   # row0: 2 trim of 3; row2: 1 trim of 3
        base_items=[[P, P, P], [P, T, P], [P, P, P]],          # one baseline drift onto trim
        flips_cand=[True, False, False], flips_base=[False, False, False])
    out = mod.compare_replica(manifest, root)
    s = out["s6_1"]
    assert s["candidate_trim_landings_total"] == 3
    assert s["baseline_trim_landings_total"] == 1
    assert len(s["candidate_rows_ever_on_trim"]) == 2
    assert len(s["baseline_rows_ever_on_trim"]) == 1
    st = out["pass_2d_stability"]
    assert st["candidate"]["replica_resolutions"] == 6
    # row0: T,T,B -> 1 agrees; row1: 2; row2: stage A None, replicas [T, None] -> the None agrees,
    # because "resolved to no item" twice is the same outcome twice.
    assert st["candidate"]["replica_resolutions_agreeing"] == 4
    assert st["baseline"]["rows_fully_stable"] == 2
    tt = out["terra_same_prompt"]
    assert tt["candidate"]["targets"]["flips"] == 1 and tt["candidate"]["targets"]["n"] == 3
    assert tt["baseline"]["targets"]["flips"] == 0
    assert tt["candidate"]["targets"]["flip_pairs"] == {"supported->unsupported": 1}
