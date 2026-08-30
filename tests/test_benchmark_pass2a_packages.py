"""Unit tests for the package-outcome benchmark extension (no live models).

2d and 2f are mocked; Pass 2e runs for real — it is deterministic and part of
what the cell rebuild must prove. v4 costing is monkeypatched everywhere it
would otherwise dominate test time.
"""
import asyncio
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_pass2a as bench
from tools import benchmark_pass2a_packages as pkg


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PROP = "prop"
PHOTO = "photo_001.jpg"

CATALOG = {
    "trade_buckets": [
        {"id": "electrical", "name": "Electrical", "product_quarantined": True},
        {"id": "masonry_exterior_structure", "name": "Masonry"},
    ],
    "items": [
        # Standalone-priced: reachable anywhere its scene_groups allow.
        {"id": "brick_weathered", "name": "Weathered Brick",
         "trade_bucket": "masonry_exterior_structure", "tier": "work",
         "kind": "degradation", "scene_groups": ["bedroom", "exterior"],
         "package_role": "standalone",
         "estimate": {"estimate_tier": "high", "strategy": "repair_or_replace",
                      "group": "structure", "stack_behavior": "max_only",
                      "unit_policy": "per_room"}},
        # Package-only: kitchen_modernization driver, no standalone path.
        {"id": "steps_cracked", "name": "Cracked Steps",
         "trade_bucket": "masonry_exterior_structure", "tier": "work",
         "kind": "defect", "scene_groups": ["kitchen"],
         "package_affinity": {"kitchen": {
             "package_type": "kitchen_modernization",
             "package_role": "package_driver"}}},
        {"id": "wiring_exposed", "name": "Exposed Wiring",
         "trade_bucket": "electrical", "tier": "work",
         "kind": "defect", "scene_groups": ["kitchen"]},
    ],
}

OBSERVED_UNITS = {PROP: ["bedroom_1", "kitchen_primary"]}


def _config(repeats=1):
    return {
        "repeats": repeats,
        "properties": [PROP],
        "model_overrides": {},
        "reasoning_efforts": {},
        "pass_toggles": {"2f": False},
        "package_eval": {
            "pass_2f": {"model": "gpt-5.6-sol", "reasoning_effort": "medium",
                        "max_tokens": 4096, "max_images": 3},
            "tail_resolution": {"emergency_ceiling": 128, "top_k": 8},
            "gates": {"pass_reps_required": 2, "cost_flag_pct": 20,
                      "cost_flag_abs_usd": 10000},
        },
    }


def _manifest(tmp_path, photos=1):
    scenes = ["kitchen", "bedroom", "bathroom", "living_room"]
    return {
        "images_root": str(tmp_path / "img"),
        "properties": {PROP: {
            "photos": [{"photo_key": f"photo_{i:03d}.jpg",
                        "scene": scenes[(i - 1) % len(scenes)],
                        "image_sha256": f"ih{i}", "frozen_2a_sha256": f"fh{i}"}
                       for i in range(1, photos + 1)],
            "property_metadata": {"beds": 3},
        }},
    }


VALID_GOLD = {
    "schema_version": "package_reference_v2",
    "properties": {PROP: {
        "canonical_rooms": [
            {"room_id": "kitchen", "room": "kitchen",
             "photo_keys": ["photo_001.jpg"]},
            {"room_id": "bedroom_A", "room": "bedroom",
             "photo_keys": ["photo_002.jpg"]},
        ],
        "expected_packages": [{
            "package_type": "kitchen_modernization",
            "room_id": "kitchen",
            "target_pricing_profile": "kitchen_full_rehab",
            "adjacent_acceptable": True,
            "policy": "strict",
            "diagnostic_cost_range": {"low": 30000, "high": 70000},
            "rationale": "dated kitchen across photos",
        }],
        "required_work_items": [{
            "catalog_item_id": "brick_weathered",
            "room_id": "bedroom_A",
            "policy": "strict",
            "accepted_sibling_ids": [],
            "rationale": "weathered brick, photo_002",
        }],
        "notes": "",
    }},
}


@pytest.fixture
def pkg_tree(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    v3 = runs / "v3"
    monkeypatch.setattr(bench, "RUNS_DIR", runs)
    monkeypatch.setattr(bench, "V3_DIR", v3)
    monkeypatch.setattr(bench, "PROMPTS_PATH", tmp_path / "prompts.json")
    (tmp_path / "prompts.json").write_text(json.dumps(
        {"baseline": {"text": "salience"}, "checklist": {"text": "inventory"}}),
        encoding="utf-8")
    monkeypatch.setattr(pkg, "TAIL_DIR", v3 / "package_tail_resolution")
    monkeypatch.setattr(pkg, "CELLS_DIR", v3 / "package_cells")
    monkeypatch.setattr(pkg, "PASS2F_DIR", v3 / "package_2f")
    monkeypatch.setattr(pkg, "REVIEW_DIR", v3 / "package_review")
    monkeypatch.setattr(pkg, "EVAL_DIR", v3 / "package_eval")
    monkeypatch.setattr(pkg, "LEGACY_TAIL_DIR", runs / "package_tail_resolution")
    monkeypatch.setattr(pkg, "LEGACY_CELLS_DIR", runs / "package_cells")
    monkeypatch.setattr(pkg, "LEGACY_PASS2F_DIR", runs / "package_2f")
    monkeypatch.setattr(pkg, "LEGACY_REVIEW_DIR", runs / "package_review")
    monkeypatch.setattr(pkg, "LEGACY_EVAL_DIR", runs / "package_eval")
    monkeypatch.setattr(pkg, "PACKAGE_GOLD_PATH",
                        tmp_path / "gold" / "package_reference.json")
    monkeypatch.setattr(pkg, "PACKAGE_GOLD_TEMPLATE_PATH",
                        tmp_path / "gold" / "package_reference.template.json")
    return tmp_path


@pytest.fixture(autouse=True)
def _catalog_stub(monkeypatch):
    """The gold check and the scorer both read the catalog; every test in
    this file works against the small fixture CATALOG."""
    import tools.artifact_writers as aw
    monkeypatch.setattr(aw, "load_issue_catalog", lambda path: CATALOG)


@pytest.fixture
def gold_stubs(monkeypatch):
    monkeypatch.setattr(pkg, "observed_units_by_property",
                        lambda *a, **k: OBSERVED_UNITS)


def _write_gold(tree, gold=VALID_GOLD):
    pkg.PACKAGE_GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    pkg.PACKAGE_GOLD_PATH.write_text(json.dumps(gold), encoding="utf-8")


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def test_adjacency_symmetric_one_step_no_self():
    adjacency = pkg.pricing_profile_adjacency()
    assert "kitchen_partial_rehab" in adjacency["kitchen_refresh"]
    assert "kitchen_refresh" in adjacency["kitchen_partial_rehab"]      # inverse
    assert "kitchen_full_rehab" in adjacency["kitchen_partial_rehab"]
    # One step only, never self.
    assert "kitchen_full_rehab" not in adjacency["kitchen_refresh"]
    for profile, neighbours in adjacency.items():
        assert profile not in neighbours
    # Bedroom skips the partial middle by design.
    assert adjacency["bedroom_refresh"] == {"bedroom_full_rehab"}


# ---------------------------------------------------------------------------
# Line-item projection (tri-state is_valid_detection)
# ---------------------------------------------------------------------------

def _v4_with_line_items(items):
    return {"groups": [{"line_items": items}]}


def test_v4_line_items_retains_unreviewed_standalone_none():
    v4 = _v4_with_line_items([
        {"catalog_item_id": "ceiling_cracks_or_sagging", "name": "Ceiling",
         "billable_estimate_unit_id": "bedroom_1", "cost_low": 500,
         "cost_high": 8000, "package_id": None, "trade_bucket": "structure",
         "is_valid_detection": None, "room_surrogate_id": "bedroom_1",
         "source_room_surrogate_ids": ["bedroom_1"]},
        {"catalog_item_id": "rejected_item", "name": "Rejected",
         "billable_estimate_unit_id": "bedroom_1", "cost_low": 0,
         "cost_high": 0, "package_id": None, "trade_bucket": "structure",
         "is_valid_detection": False},
        {"catalog_item_id": "confirmed_item", "name": "Confirmed",
         "billable_estimate_unit_id": "kitchen_primary", "cost_low": 100,
         "cost_high": 200, "package_id": "pkg1", "trade_bucket": "structure",
         "is_valid_detection": True},
    ])
    kept = pkg.v4_line_items(v4)
    ids = [li["catalog_item_id"] for li in kept]
    # None (2f never ran on the standalone line) survives; explicit False drops.
    assert ids == ["ceiling_cracks_or_sagging", "confirmed_item"]
    # Tri-state preserved, never coerced to bool.
    assert kept[0]["is_valid_detection"] is None
    assert kept[1]["is_valid_detection"] is True
    # Physical-room provenance is exported for canonical-room alignment.
    assert kept[0]["room_surrogate_id"] == "bedroom_1"
    assert kept[0]["source_room_surrogate_ids"] == ["bedroom_1"]
    assert kept[1]["source_room_surrogate_ids"] == []


def test_v4_line_items_valid_only_false_keeps_rejected():
    v4 = _v4_with_line_items([
        {"catalog_item_id": "rejected_item", "is_valid_detection": False},
    ])
    kept = pkg.v4_line_items(v4, valid_only=False)
    assert [li["catalog_item_id"] for li in kept] == ["rejected_item"]
    assert kept[0]["is_valid_detection"] is False


def test_compute_pre2f_totals_retains_standalone_none(monkeypatch):
    import tools.renovation_estimate_v4 as rev4

    def fake_v4(issues_flat, catalog, photos, property_metadata=None,
                package_verifications=None, **kwargs):
        return {
            "package_candidates": [{"package_id": "pkg1"}],
            "groups": [{"line_items": [
                {"catalog_item_id": "standalone_none", "cost_low": 500,
                 "cost_high": 900, "package_id": None,
                 "is_valid_detection": None},
                {"catalog_item_id": "member_true", "cost_low": 100,
                 "cost_high": 200, "package_id": "pkg1",
                 "is_valid_detection": True},
                {"catalog_item_id": "rejected_false", "cost_low": 0,
                 "cost_high": 0, "package_id": None,
                 "is_valid_detection": False},
            ]}],
            "packages": [], "final_rehab": {"low": 600, "high": 1100},
        }

    monkeypatch.setattr(rev4, "compute_renovation_estimate_v4", fake_v4)
    totals = bench.compute_pre2f_totals({"photos": {}}, CATALOG)
    ids = [li["catalog_item_id"] for li in totals["line_items"]]
    assert ids == ["standalone_none", "member_true"]


# ---------------------------------------------------------------------------
# Gold check + template
# ---------------------------------------------------------------------------

def test_gold_check_accepts_valid_reference(pkg_tree, gold_stubs):
    _write_gold(pkg_tree, VALID_GOLD)
    sha = pkg.package_gold_check(_config(), _manifest(pkg_tree, photos=2))
    assert len(sha) == 64
    audit = json.loads((pkg.EVAL_DIR / "gold_reachability.json")
                       .read_text(encoding="utf-8"))
    assert audit[PROP]["item:brick_weathered@bedroom_A"]["reachable"] is True
    assert audit[PROP]["item:brick_weathered@bedroom_A"]["routes"] == [
        "standalone_priced"]
    assert audit[PROP]["pkg:kitchen_modernization__kitchen"]["reachable"] is True


def test_gold_sha_is_line_ending_independent(tmp_path):
    crlf = json.dumps(VALID_GOLD, indent=2).replace("\n", "\r\n")
    lf = json.dumps(VALID_GOLD, indent=1)
    assert crlf.encode() != lf.encode()
    assert (pkg.package_gold_sha(json.loads(crlf))
            == pkg.package_gold_sha(json.loads(lf)))


@pytest.mark.parametrize("mutate,problem", [
    (lambda g: g["properties"][PROP]["required_work_items"][0].update(
        catalog_item_id="wiring_exposed"), "product-quarantined"),
    (lambda g: g["properties"][PROP]["required_work_items"][0].update(
        catalog_item_id="no_such_item"), "unknown catalog_item_id"),
    (lambda g: g["properties"][PROP]["expected_packages"][0].update(
        package_type="interior_paint_flooring_refresh"),
     "non-authorable package_type"),
    (lambda g: g["properties"][PROP]["expected_packages"][0].update(
        target_pricing_profile="bathroom_full_rehab"), "not in the kitchen family"),
    (lambda g: g["properties"][PROP]["expected_packages"][0].update(
        target_pricing_profile="kitchen_mega_rehab"), "unknown target_pricing_profile"),
    (lambda g: g["properties"][PROP]["expected_packages"][0].update(
        room_id="kitchen_42"), "unknown room_id"),
    (lambda g: g["properties"][PROP]["required_work_items"][0].update(
        room_id="nowhere"), "unknown room_id"),
    (lambda g: g["properties"][PROP]["expected_packages"].append(
        dict(g["properties"][PROP]["expected_packages"][0])),
     "duplicate (package_type, room_id)"),
    (lambda g: g["properties"][PROP]["expected_packages"][0].update(
        rationale="  "), "rationale is required"),
    (lambda g: g["properties"].update(ghost={"expected_packages": [],
                                             "required_work_items": []}),
     "properties mismatch"),
    (lambda g: g.update(schema_version="package_reference_v1"),
     "schema_version must be package_reference_v2"),
    # Canonical-room integrity.
    (lambda g: g["properties"][PROP]["canonical_rooms"][1].update(
        photo_keys=["photo_001.jpg"]), "photo groups must be disjoint"),
    (lambda g: g["properties"][PROP]["canonical_rooms"][1].update(
        photo_keys=["photo_099.jpg"]), "not in the manifest"),
    (lambda g: g["properties"][PROP]["canonical_rooms"][1].update(
        photo_keys=[]), "photo_keys must be non-empty"),
    (lambda g: g["properties"][PROP]["canonical_rooms"][0].update(
        room="ballroom"), "unknown room family"),
    (lambda g: g["properties"][PROP]["canonical_rooms"][0].update(
        room="bedroom"), "package family 'kitchen' != room family 'bedroom'"),
    # Policy integrity.
    (lambda g: g["properties"][PROP]["required_work_items"][0].update(
        policy="maybe"), "policy must be one of"),
    (lambda g: g["properties"][PROP]["required_work_items"][0].update(
        policy="diagnostic_only"), "requires a diagnostic_reason"),
    (lambda g: g["properties"][PROP]["required_work_items"][0].update(
        diagnostic_reason="stray"), "only valid on diagnostic_only"),
    (lambda g: g["properties"][PROP]["required_work_items"][0].update(
        accepted_sibling_ids=["no_such_sibling"]), "unknown accepted sibling"),
    # Reachability: a kitchen-only package item required in a bedroom room.
    (lambda g: g["properties"][PROP]["required_work_items"][0].update(
        catalog_item_id="steps_cracked"), "strict target is unreachable"),
])
def test_gold_check_rejects_bad_gold(pkg_tree, gold_stubs, mutate, problem):
    gold = json.loads(json.dumps(VALID_GOLD))
    mutate(gold)
    _write_gold(pkg_tree, gold)
    with pytest.raises(SystemExit) as excinfo:
        pkg.package_gold_check(_config(), _manifest(pkg_tree, photos=2))
    assert problem in str(excinfo.value)


def test_gold_check_diagnostic_only_bypasses_reachability(pkg_tree, gold_stubs):
    gold = json.loads(json.dumps(VALID_GOLD))
    gold["properties"][PROP]["required_work_items"][0].update(
        catalog_item_id="steps_cracked", policy="diagnostic_only",
        diagnostic_reason="kitchen-affinity item, bedroom room — known "
                          "incompatible, kept visible")
    _write_gold(pkg_tree, gold)
    sha = pkg.package_gold_check(_config(), _manifest(pkg_tree, photos=2))
    assert len(sha) == 64
    audit = json.loads((pkg.EVAL_DIR / "gold_reachability.json")
                       .read_text(encoding="utf-8"))
    verdict = audit[PROP]["item:steps_cracked@bedroom_A"]
    assert verdict["reachable"] is False
    assert verdict["policy"] == "diagnostic_only"
    assert any("no_package_affinity_for_room" in b for b in verdict["blockers"])


def test_gold_template_is_blank_first_with_vocabularies(pkg_tree, gold_stubs):
    manifest = _manifest(pkg_tree, photos=2)
    pkg.package_gold_template(_config(), manifest)
    template = json.loads(pkg.PACKAGE_GOLD_TEMPLATE_PATH.read_text(
        encoding="utf-8"))
    for prop_entry in template["properties"].values():
        assert prop_entry["canonical_rooms"] == []         # blank-first
        assert prop_entry["expected_packages"] == []
        assert prop_entry["required_work_items"] == []
    vocab = template["_vocabulary"]
    assert "kitchen_modernization" in vocab["package_types"]
    assert "interior_paint_flooring_refresh" not in vocab["package_types"]
    # Manifest photos + frozen scenes replace the predicted-unit universe:
    # gold no longer depends on pipeline output.
    assert vocab["manifest_photos"][PROP] == [
        {"photo_key": "photo_001.jpg", "scene": "kitchen"},
        {"photo_key": "photo_002.jpg", "scene": "bedroom"},
    ]
    assert "observed_estimate_units" not in vocab
    quarantined_ids = {r["catalog_item_id"]
                       for r in vocab["quarantined_not_authorable"]}
    assert quarantined_ids == {"wiring_exposed"}
    assert all(r["catalog_item_id"] != "wiring_exposed"
               for r in vocab["catalog_items"])
    # Refuses to clobber authoring work.
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pkg.package_gold_template(_config(), manifest)


# ---------------------------------------------------------------------------
# Tail resolution stage
# ---------------------------------------------------------------------------

def _observation(i, kind="degradation"):
    return {"description": f"observation number {i} shows wear",
            "issue_id": f"i{i}", "kind": kind, "scene": "exterior_front",
            "scene_group": "exterior", "source_photo_key": PHOTO}


def _write_checkpoint(variant, rep, kept_count, resolved_count,
                      photo=PHOTO):
    observations = [_observation(i) for i in range(kept_count)]
    resolved = [{"issue_id": f"i{i}", "description": observations[i]["description"],
                 "resolved_item_id": "brick_weathered" if i % 2 == 0 else None,
                 "resolved_kind": "degradation"}
                for i in range(resolved_count)]
    ckpt_dir = pkg.variant_dir(variant) / f"rep{rep}" / PROP / ".photos"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / f"{photo}.json").write_text(json.dumps({
        "image_path": f"C:/img/{PROP}/{photo}",
        "scene": "exterior_front",
        "processing_time": 1.0,
        "scene_data": {
            "observations": observations,
            "resolved_items": resolved,
            "excluded_observations": [],
            "debug": {"pass_2d_gate": {"total_resolve_count": resolved_count},
                      "pass_2d_per_observation": []},
        },
    }), encoding="utf-8")


@pytest.fixture
def tail_stubs(monkeypatch):
    import tools.artifact_writers as aw
    import tools.catalog_embeddings as ce
    import tools.scene_classifier_orchestrator as orch
    import tools.vlm_client as vc
    monkeypatch.setattr(aw, "load_issue_catalog", lambda path: CATALOG)
    monkeypatch.setattr(ce, "build_candidate_provider",
                        lambda catalog: lambda text, ctx: [])
    monkeypatch.setattr(vc, "create_vlm_client", lambda *a, **k: object())
    calls = []

    async def fake_resolve(**kwargs):
        obs = kwargs["observation"]
        calls.append(kwargs)
        return ({"issue_id": obs["issue_id"], "description": obs["description"],
                 "resolved_item_id": "steps_cracked",
                 "resolved_kind": obs["kind"]},
                {"observation": obs["description"], "skipped_reason": None},
                None)
    monkeypatch.setattr(orch, "resolve_observation_against_catalog",
                        fake_resolve)
    monkeypatch.setattr(pkg, "compute_resolution_fingerprint",
                        lambda *a, **k: {"layer": "test"})
    return calls


def test_tail_resolves_only_beyond_cap_in_order(pkg_tree, tail_stubs):
    _write_checkpoint("checklist", 1, kept_count=6,
                      resolved_count=4)
    _write_checkpoint("baseline", 1, kept_count=3,
                      resolved_count=3)
    pkg.stage_package_tail(_config(), _manifest(pkg_tree),
                           "baseline_vs_checklist", skip_preflight=True)
    assert [c["observation"]["issue_id"] for c in tail_stubs] == ["i4", "i5"]
    out = json.loads((pkg.TAIL_DIR / "rep1" / PROP / f"{PHOTO}.json")
                     .read_text(encoding="utf-8"))
    assert out["tail_count"] == 2
    assert [r["issue_id"] for r in out["resolved_rows"]] == ["i4", "i5"]
    # Resume: a second run makes zero resolver calls.
    tail_stubs.clear()
    pkg.stage_package_tail(_config(), _manifest(pkg_tree),
                           "baseline_vs_checklist", skip_preflight=True)
    assert tail_stubs == []


def test_tail_ceiling_aborts_without_truncating(pkg_tree, tail_stubs):
    _write_checkpoint("checklist", 1, kept_count=129,
                      resolved_count=25)
    _write_checkpoint("baseline", 1, kept_count=3,
                      resolved_count=3)
    with pytest.raises(SystemExit, match="emergency ceiling"):
        pkg.stage_package_tail(_config(), _manifest(pkg_tree),
                               "baseline_vs_checklist", skip_preflight=True)
    assert not (pkg.TAIL_DIR / "rep1" / PROP / f"{PHOTO}.json").is_file()


def test_tail_92_kept_passes_under_ceiling(pkg_tree, tail_stubs):
    _write_checkpoint("checklist", 1, kept_count=92,
                      resolved_count=90)
    _write_checkpoint("baseline", 1, kept_count=3,
                      resolved_count=3)
    pkg.stage_package_tail(_config(), _manifest(pkg_tree),
                           "baseline_vs_checklist", skip_preflight=True)
    assert len(tail_stubs) == 2


def test_baseline_capped_breaks_the_reuse_assumption(pkg_tree, tail_stubs):
    _write_checkpoint("checklist", 1, kept_count=6,
                      resolved_count=4)
    _write_checkpoint("baseline", 1, kept_count=30,
                      resolved_count=25)
    with pytest.raises(SystemExit, match="baseline-reuse assumption"):
        pkg.stage_package_tail(_config(), _manifest(pkg_tree),
                               "baseline_vs_checklist", skip_preflight=True)


# ---------------------------------------------------------------------------
# Cell rebuild (real Pass 2e)
# ---------------------------------------------------------------------------

def test_cell_rebuild_merges_tail_and_reruns_2e(pkg_tree, monkeypatch):
    import tools.artifact_writers as aw
    import tools.vlm_client as vc
    _write_checkpoint("checklist", 1, kept_count=4,
                      resolved_count=2)
    tail_dir = pkg.TAIL_DIR / "rep1" / PROP
    tail_dir.mkdir(parents=True, exist_ok=True)
    (tail_dir / f"{PHOTO}.json").write_text(json.dumps({
        "resolved_rows": [{"issue_id": "i2", "description":
                           "observation number 2 shows wear",
                           "resolved_item_id": "steps_cracked",
                           "resolved_kind": "degradation"}],
        "debug_rows": [{"observation": "observation number 2 shows wear",
                        "skipped_reason": None}],
    }), encoding="utf-8")

    captured = {}

    def fake_write(*, job, **kwargs):
        captured["job"] = job
        captured["pass_toggles"] = kwargs["pass_toggles"]
        out = Path(job.artifacts_dir) / "photo_intel.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return out
    monkeypatch.setattr(aw, "load_issue_catalog", lambda path: CATALOG)
    monkeypatch.setattr(aw, "write_photo_intel", fake_write)
    monkeypatch.setattr(vc, "get_model_configs_from_pipeline_config",
                        lambda cfg: ({}, {"api_key": "k"}))
    monkeypatch.setattr(bench, "ensure_totals", lambda path, catalog: {})
    monkeypatch.setattr(pkg, "compute_resolution_fingerprint",
                        lambda *a, **k: {"layer": "test"})

    pkg.stage_package_cells(_config(), _manifest(pkg_tree),
                            "baseline_vs_checklist")

    scene_data = captured["job"].results[0].scene_data
    assert len(scene_data["resolved_items"]) == 3          # 2 frozen + 1 tail
    assert scene_data["debug"]["pass_2d_gate"]["total_resolve_count"] == 3
    assert captured["pass_toggles"]["2f"] is False
    # Real 2e ran: tail-resolved observation carries the stamped catalog id.
    stamped = {i.get("issue_id"): i.get("catalogItemId")
               for i in scene_data["verified_issues"]}
    assert stamped.get("i2") == "steps_cracked"
    assert stamped.get("i0") == "brick_weathered"          # frozen resolution
    assert scene_data["passes"]["2e"]["kept_issue_ids"]
    assert scene_data["passes"]["2e"]["input_count"] == 4
    # Resume: artifact exists, nothing rebuilt.
    captured.clear()
    pkg.stage_package_cells(_config(), _manifest(pkg_tree),
                            "baseline_vs_checklist")
    assert "job" not in captured


# ---------------------------------------------------------------------------
# 2f stage: save/resume/budget/version guard/recompute passthrough
# ---------------------------------------------------------------------------

def _candidate(ptype="kitchen_modernization", unit="kitchen_primary",
               profile="kitchen_full_rehab", **extra):
    return {"package_id": f"{ptype}__{unit}", "package_type": ptype,
            "estimate_unit_id": unit, "pricing_profile": profile,
            "pricing_tier": profile.split("_", 1)[1], "package_strength": "strong",
            "verification_status": "not_run", "estimate_eligible": False,
            "audit_only": False, "estimate_display_only": False,
            "package_level": "room", "cost_low": 30000, "cost_high": 70000,
            "cost_midpoint": 50000, "supporting_catalog_item_ids": ["cab_dated"],
            "review_photo_keys": [PHOTO], "evidence_summary": "dated", **extra}


@pytest.fixture
def twof_stubs(pkg_tree, monkeypatch):
    import tools.pass_2f_artifact_inputs as p2fi
    import tools.pass_config as pc
    import tools.rehab_packages as rp
    import tools.renovation_estimate_v4 as rev4
    import tools.vlm_client as vc

    artifact = pkg_tree / "artifact.json"
    artifact.write_text(json.dumps({"photos": {}, "product_issues_flat": [
        {"issue_id": "x"}], "property_metadata": {}}), encoding="utf-8")
    cells = {}
    for cell in ("baseline_cap25", "checklist_cap25", "checklist_all_retained"):
        cells[(cell, 1, PROP)] = artifact
    monkeypatch.setattr(pkg, "_load_cell_artifacts", lambda *a, **k: cells)
    monkeypatch.setattr(pkg, "compute_pass2f_fingerprint",
                        lambda *a, **k: {"layer": "test-2f"})
    monkeypatch.setattr(pkg, "two_phase_v4", lambda a, c: (
        {"package_candidates": [_candidate()]}, {}))
    import tools.artifact_writers as aw
    monkeypatch.setattr(aw, "load_issue_catalog", lambda path: CATALOG)
    monkeypatch.setattr(vc, "create_vlm_client", lambda *a, **k: object())
    monkeypatch.setattr(vc, "get_model_configs_from_pipeline_config",
                        lambda cfg: ({}, {"api_key": "k"}))
    monkeypatch.setattr(pc, "resolve_openai_invocation",
                        lambda pass_key, config, effort=None: dict(config))
    monkeypatch.setattr(p2fi, "photo_key_to_path", lambda artifact: {})

    state = {"batch_calls": 0, "verification": {
        "package_id": "kitchen_modernization__kitchen_primary",
        "verification_status": "confirmed",
        "confirmed_issue_ids": ["x"], "rejected_issue_ids": [],
        "prompt_template_version": "pass_2f_package_v2",
    }, "v4_kwargs": []}

    async def fake_batch(candidates, **kwargs):
        state["batch_calls"] += 1
        return ({"kitchen_modernization__kitchen_primary":
                 dict(state["verification"])},
                {"ran": True, "attempted_count": 2})
    monkeypatch.setattr(rp, "run_pass_2f_batch", fake_batch)

    def fake_v4(issues_flat, catalog, photos, **kwargs):
        state["v4_kwargs"].append(kwargs)
        return {"packages": [_candidate(estimate_eligible=True,
                                        verification_status="confirmed")],
                "groups": [], "final_rehab": {"midpoint": 50000},
                "pass_2f_trace": {"reason": "provided_verifications"}}
    monkeypatch.setattr(rev4, "compute_renovation_estimate_v4", fake_v4)
    return state


def test_2f_stage_saves_resumes_and_passes_verifications_through(pkg_tree,
                                                                 twof_stubs):
    pkg.stage_package_2f(_config(), _manifest(pkg_tree),
                         "baseline_vs_checklist")
    assert twof_stubs["batch_calls"] == 3
    saved = json.loads((pkg.PASS2F_DIR / "baseline_cap25" / "rep1" / PROP
                        / "verifications.json").read_text(encoding="utf-8"))
    assert saved["verifications"]["kitchen_modernization__kitchen_primary"][
        "verification_status"] == "confirmed"
    # The recompute received the saved verifications verbatim (replay switch).
    assert all(
        k["package_verifications"]["kitchen_modernization__kitchen_primary"][
            "prompt_template_version"] == "pass_2f_package_v2"
        for k in twof_stubs["v4_kwargs"])
    final = json.loads((pkg.PASS2F_DIR / "checklist_cap25" / "rep1" / PROP
                        / "final_estimate.json").read_text(encoding="utf-8"))
    assert final["packages"][0]["estimate_eligible"] is True
    # Resume: nothing re-verified.
    pkg.stage_package_2f(_config(), _manifest(pkg_tree),
                         "baseline_vs_checklist")
    assert twof_stubs["batch_calls"] == 3


def test_2f_budget_stops_cleanly_between_cells(pkg_tree, twof_stubs):
    pkg.stage_package_2f(_config(), _manifest(pkg_tree),
                         "baseline_vs_checklist", budget=2)
    assert twof_stubs["batch_calls"] == 1     # 2 calls made, budget reached
    pkg.stage_package_2f(_config(), _manifest(pkg_tree),
                         "baseline_vs_checklist", budget=200)
    assert twof_stubs["batch_calls"] == 3     # resumed the remaining cells


def test_2f_prompt_version_drift_aborts(pkg_tree, twof_stubs):
    twof_stubs["verification"]["prompt_template_version"] = "pass_2f_package_v3"
    with pytest.raises(SystemExit, match="prompt_template_version drifted"):
        pkg.stage_package_2f(_config(), _manifest(pkg_tree),
                             "baseline_vs_checklist")


# ---------------------------------------------------------------------------
# Matching + scoring
# ---------------------------------------------------------------------------

ADJ = pkg.pricing_profile_adjacency()
EXPECTED = VALID_GOLD["properties"][PROP]["expected_packages"][0]

# Prediction-side units and their photo groups: kitchen_primary aligns to
# canonical room "kitchen", bedroom_1 to "bedroom_A"; bedroom_2's photo is in
# no canonical room (deliberately unalignable).
UNITS = [
    {"estimate_unit_id": "kitchen_primary", "photo_ids": ["photo_001.jpg"]},
    {"estimate_unit_id": "bedroom_1", "photo_ids": ["photo_002.jpg"]},
    {"estimate_unit_id": "bedroom_2", "photo_ids": ["photo_003.jpg"]},
]
ROOM_OF_UNIT = {"kitchen_primary": "kitchen", "bedroom_1": "bedroom_A",
                "bedroom_2": None}


def test_match_expected_exact_adjacent_wrong_missing():
    from tools import benchmark_pass2a_scoring as sc
    exact = _candidate(profile="kitchen_full_rehab")
    adjacent = _candidate(profile="kitchen_partial_rehab")
    far = _candidate(profile="kitchen_refresh")
    assert sc._match_expected(EXPECTED, [exact], ROOM_OF_UNIT, ADJ) == (
        "matched_exact", "kitchen_full_rehab", "kitchen_primary")
    assert sc._match_expected(EXPECTED, [adjacent], ROOM_OF_UNIT, ADJ) == (
        "matched_adjacent", "kitchen_partial_rehab", "kitchen_primary")
    assert sc._match_expected(EXPECTED, [far], ROOM_OF_UNIT, ADJ) == (
        "wrong_tier", "kitchen_refresh", "kitchen_primary")
    assert sc._match_expected(EXPECTED, [], ROOM_OF_UNIT, ADJ) == (
        "missing", None, None)
    # A right-type package on a unit aligned to a DIFFERENT room never matches.
    elsewhere = _candidate(unit="bedroom_1")
    assert sc._match_expected(EXPECTED, [elsewhere], ROOM_OF_UNIT, ADJ) == (
        "missing", None, None)
    strict = {**EXPECTED, "adjacent_acceptable": False}
    assert sc._match_expected(strict, [adjacent], ROOM_OF_UNIT, ADJ) == (
        "wrong_tier", "kitchen_partial_rehab", "kitchen_primary")


def _write_cell(cell, rep, packages, line_items, midpoint=50000,
                verifications=None, candidates=None, units=UNITS):
    cell_dir = pkg.PASS2F_DIR / cell / f"rep{rep}" / PROP
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "final_estimate.json").write_text(json.dumps({
        "cell": cell, "rep": rep, "property_key": PROP,
        "packages": packages, "line_items": line_items,
        "estimate_units": units,
        "final_rehab": {"midpoint": midpoint},
    }), encoding="utf-8")
    (cell_dir / "candidates.json").write_text(json.dumps({
        "cell": cell, "rep": rep, "property_key": PROP,
        "candidates": candidates if candidates is not None else packages,
        "estimate_units": units,
    }), encoding="utf-8")
    (cell_dir / "verifications.json").write_text(json.dumps({
        "verifications": verifications or {}}), encoding="utf-8")


def _approved(**extra):
    return _candidate(estimate_eligible=True,
                      verification_status="confirmed", **extra)


def _work_item(cid="brick_weathered", unit="bedroom_1", package_id=None,
               cost=1000):
    return {"catalog_item_id": cid, "name": cid,
            "billable_estimate_unit_id": unit, "cost_low": cost,
            "cost_high": cost * 2, "package_id": package_id,
            "trade_bucket": "masonry_exterior_structure",
            "is_valid_detection": True}


def _score(pkg_tree, config=None, gold=VALID_GOLD, decisions=None):
    _write_gold(pkg_tree, gold)
    return pkg.score_package_round(config or _config(), _manifest(pkg_tree),
                                   "baseline_vs_checklist", gold,
                                   decisions or {})


def _fill_all_cells(packages, line_items, repeats=1, **kwargs):
    for cell in ("baseline_cap25", "checklist_cap25", "checklist_all_retained"):
        for rep in range(1, repeats + 1):
            _write_cell(cell, rep, packages, line_items, **kwargs)


def test_complete_rep_requires_packages_and_work(pkg_tree):
    _fill_all_cells([_approved()], [_work_item()])
    scored = _score(pkg_tree)
    rep = scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"]
    assert rep["status"] == "complete"
    assert rep["strict_complete"] is True
    matched = rep["expected_packages"]["kitchen_modernization__kitchen"]
    assert matched["status"] == "matched_exact"
    assert matched["tier_distance"] == 0
    assert matched["family_present"] is True
    assert rep["package_family_recall"] == 1.0
    assert rep["alignment"]["kitchen_primary"] == {
        "status": "aligned", "room_id": "kitchen", "overlap": 1,
        "candidates": ["kitchen"]}
    work = rep["required_work"]["brick_weathered@bedroom_A"]
    assert work["status"] == "satisfied" and work["route"] == "line_item"
    assert rep["component_coverage"] == {"covered": 1, "total": 1,
                                         "missing_components": []}
    assert rep["exact_id_coverage"] == {"covered": 1, "total": 1}
    assert scored["status"] == "final"


def _work_of(scored, gid="brick_weathered@bedroom_A"):
    return scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"][
        "required_work"][gid]


def test_required_work_satisfied_standalone_or_absorbed_only_into_approved(
        pkg_tree):
    approved = _approved()
    # Absorbed into the approved package: satisfied.
    _fill_all_cells([approved],
                    [_work_item(package_id=approved["package_id"])])
    assert _work_of(_score(pkg_tree))["status"] == "satisfied"
    # Filed under a package that is NOT approved: missing.
    _fill_all_cells([approved],
                    [_work_item(package_id="bedroom_repair__bedroom_1")])
    assert _work_of(_score(pkg_tree))["status"] == "missing"
    # No surviving line item, but an approved package aligned to the same
    # canonical room carries the id as absorbed evidence: satisfied (2f
    # prunes unconfirmed members).
    absorbing = _approved(package_id="bedroom_repair__bedroom_1",
                          package_type="bedroom_repair",
                          estimate_unit_id="bedroom_1",
                          pricing_profile="bedroom_repair_heavy",
                          supporting_catalog_item_ids=["brick_weathered"])
    _fill_all_cells([_approved(), absorbing], [])
    result = _work_of(_score(pkg_tree))
    assert result["status"] == "satisfied"
    assert result["route"] == "package_support"
    # Same evidence on a package whose unit aligns to a DIFFERENT (here: no)
    # canonical room does not satisfy it.
    wrong_unit = _approved(package_id="bedroom_repair__bedroom_2",
                           package_type="bedroom_repair",
                           estimate_unit_id="bedroom_2",
                           pricing_profile="bedroom_repair_heavy",
                           supporting_catalog_item_ids=["brick_weathered"])
    _fill_all_cells([_approved(), wrong_unit], [])
    assert _work_of(_score(pkg_tree))["status"] == "missing"


def test_ceiling_crack_standalone_none_satisfies_but_rejected_does_not(
        pkg_tree):
    # The Carroll regression: an unreviewed standalone line (2f never ran on
    # it, is_valid_detection=None) must satisfy required work end-to-end.
    unreviewed = _work_item()
    unreviewed["is_valid_detection"] = None
    _fill_all_cells([_approved()], [unreviewed])
    scored = _score(pkg_tree)
    assert _work_of(scored)["status"] == "satisfied"
    assert scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"][
        "strict_complete"] is True
    # An explicitly rejected line does not (v4_line_items drops it upstream;
    # even if projected with valid_only=False it must not credit gold — the
    # scorer sees only the projected artifact, so simulate the projection).
    v4 = {"groups": [{"line_items": [
        {**_work_item(), "is_valid_detection": False}]}]}
    assert pkg.v4_line_items(v4) == []


def test_accepted_sibling_satisfies_concept_not_exact_id(pkg_tree):
    gold = json.loads(json.dumps(VALID_GOLD))
    gold["properties"][PROP]["required_work_items"][0][
        "accepted_sibling_ids"] = ["steps_cracked"]
    _fill_all_cells([_approved()], [_work_item(cid="steps_cracked")])
    scored = _score(pkg_tree, gold=gold)
    rep = scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"]
    work = rep["required_work"]["brick_weathered@bedroom_A"]
    assert work["status"] == "satisfied_sibling"
    # Concept satisfied (same component, same room) but the exact ID is not.
    assert rep["component_coverage"]["covered"] == 1
    assert rep["exact_id_coverage"] == {"covered": 0, "total": 1}
    # Sibling satisfaction still completes the strict gate.
    assert rep["strict_complete"] is True


def test_diagnostic_only_target_never_fails_strict(pkg_tree):
    gold = json.loads(json.dumps(VALID_GOLD))
    gold["properties"][PROP]["required_work_items"][0].update(
        policy="diagnostic_only",
        diagnostic_reason="unreachable per audit")
    # The work item is NOT produced at all — strict completion still holds.
    _fill_all_cells([_approved()], [])
    scored = _score(pkg_tree, gold=gold)
    rep = scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"]
    work = rep["required_work"]["brick_weathered@bedroom_A"]
    assert work["status"] == "missing" and work["policy"] == "diagnostic_only"
    assert rep["strict_complete"] is True
    assert rep["strict_missing"] == []
    # It stays visible in the missing-target aggregation with its policy.
    row = scored["distinct_missing_targets_across_runs"][
        f"{PROP}|item|brick_weathered@bedroom_A"]
    assert row["policy"] == "diagnostic_only"


def test_wrong_tier_and_missing_fail_the_rep(pkg_tree):
    _fill_all_cells([_approved(profile="kitchen_refresh")], [_work_item()])
    scored = _score(pkg_tree)
    rep = scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"]
    assert rep["status"] == "incomplete"
    row = rep["expected_packages"]["kitchen_modernization__kitchen"]
    assert row["status"] == "wrong_tier"
    assert row["tier_distance"] == -2 and row["abs_tier_distance"] == 2
    assert row["family_present"] is True    # right family, wrong tier
    assert "kitchen_modernization__kitchen" in rep["strict_missing"]
    # Union semantics with occurrence locations: the target is missing in
    # every cell's rep 1, once each.
    entry = scored["distinct_missing_targets_across_runs"][
        f"{PROP}|pkg|kitchen_modernization__kitchen"]
    assert entry["policy"] == "strict"
    assert entry["occurrences"] == 3
    assert entry["cells"] == {"baseline_cap25": [1], "checklist_cap25": [1],
                              "checklist_all_retained": [1]}


def test_extras_closed_world_blocks_then_decisions_resolve(pkg_tree):
    extra = _approved(package_id="bedroom_repair__bedroom_1",
                      package_type="bedroom_repair",
                      estimate_unit_id="bedroom_1",
                      pricing_profile="bedroom_repair_heavy")
    _fill_all_cells([_approved(), extra], [_work_item()])
    # Undecided extra pends THIS property's verdict only.
    scored = _score(pkg_tree)
    assert scored["status"] == "blocked"
    assert scored["pending_review"]["extras_pending"] > 0
    rep = scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"]
    assert rep["status"] == "pending_review"
    assert rep["strict_complete"] is None
    # Diagnostics are not suppressed by the pending row.
    assert rep["package_family_recall"] == 1.0
    assert rep["exact_id_coverage"] == {"covered": 1, "total": 1}
    assert scored["outcomes"]["checklist_cap25"][PROP]["passed"] is None

    sha = pkg.package_gold_sha(VALID_GOLD)
    # The extra is keyed by its ALIGNED canonical room, not the ordinal unit.
    row_id = f"{PROP}|pkg|bedroom_repair__bedroom_A"
    # false_positive: confirmed extra scope, rep fails, round final.
    scored = _score(pkg_tree, decisions={row_id: {
        "decision": "false_positive", "equivalent_gold_id": None,
        "gold_sha256": sha}})
    assert scored["status"] == "final"
    rep = scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"]
    assert rep["status"] == "incomplete" and rep["extras_confirmed"]
    # gold_gap blocks until the gold moves; stale sha needs re-review.
    scored = _score(pkg_tree, decisions={row_id: {
        "decision": "gold_gap", "equivalent_gold_id": None,
        "gold_sha256": sha}})
    assert scored["status"] == "blocked"
    assert scored["pending_review"]["gold_gap"] == 1
    scored = _score(pkg_tree, decisions={row_id: {
        "decision": "gold_gap", "equivalent_gold_id": None,
        "gold_sha256": "stale"}})
    assert scored["pending_review"]["needs_rereview"] == 1


def test_pending_review_scopes_to_its_own_cells(pkg_tree):
    # The extra appears ONLY in checklist_cap25 — the other cells' verdicts
    # must remain real booleans (no global nulling).
    extra = _approved(package_id="bedroom_repair__bedroom_1",
                      package_type="bedroom_repair",
                      estimate_unit_id="bedroom_1",
                      pricing_profile="bedroom_repair_heavy")
    good = ([_approved()], [_work_item()])
    _write_cell("baseline_cap25", 1, *good)
    _write_cell("checklist_cap25", 1, [_approved(), extra], [_work_item()])
    _write_cell("checklist_all_retained", 1, *good)
    config = _config()
    config["package_eval"]["gates"]["pass_reps_required"] = 1
    scored = _score(pkg_tree, config=config)
    assert scored["status"] == "blocked"
    assert scored["outcomes"]["baseline_cap25"][PROP]["passed"] is True
    assert scored["outcomes"]["checklist_cap25"][PROP]["passed"] is None
    assert scored["outcomes"]["checklist_all_retained"][PROP]["passed"] is True


def test_renamed_unit_aligns_by_photos_without_review(pkg_tree):
    # The Carroll drift repro in miniature: the pipeline filed the kitchen
    # package under a different ordinal unit id, but its photo group still
    # identifies the physical kitchen — photo alignment scores it with no
    # human adjudication.
    renamed = _approved(package_id="kitchen_modernization__kitchen_1",
                        estimate_unit_id="kitchen_1")
    units = [{"estimate_unit_id": "kitchen_1", "photo_ids": ["photo_001.jpg"]},
             {"estimate_unit_id": "bedroom_1", "photo_ids": ["photo_002.jpg"]}]
    _fill_all_cells([renamed], [_work_item()], units=units)
    scored = _score(pkg_tree)
    rep = scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"]
    assert rep["expected_packages"][
        "kitchen_modernization__kitchen"]["status"] == "matched_exact"
    assert rep["status"] == "complete"
    assert scored["status"] == "final"


def test_equivalent_decision_credits_the_mapped_gold_id(pkg_tree):
    # A package on a unit with NO photo provenance cannot align; the human
    # 'equivalent' decision still both clears the extra and credits gold.
    unmapped = _approved(package_id="kitchen_modernization__kitchen_9",
                         estimate_unit_id="kitchen_9")
    _fill_all_cells([unmapped], [_work_item()])
    sha = pkg.package_gold_sha(VALID_GOLD)
    scored = _score(pkg_tree, decisions={
        f"{PROP}|pkg|kitchen_modernization__kitchen_9": {
            "decision": "equivalent",
            "equivalent_gold_id": "kitchen_modernization__kitchen",
            "gold_sha256": sha}})
    rep = scored["outcomes"]["checklist_cap25"][PROP]["reps"]["1"]
    assert rep["expected_packages"][
        "kitchen_modernization__kitchen"]["status"] == "matched_equivalent"
    assert rep["status"] == "complete"
    assert scored["status"] == "final"


def test_evidence_and_package_member_items_are_never_extras(pkg_tree):
    approved = _approved()
    member = _work_item(cid="counter_dated", unit="kitchen_primary",
                        package_id=approved["package_id"])
    evidence = _work_item(cid="cab_dated", unit="kitchen_primary")  # standalone
    _fill_all_cells([approved], [_work_item(), member, evidence])
    scored = _score(pkg_tree)
    assert scored["status"] == "final"      # nothing pended for review


def test_two_of_three_gate_and_cap_verdicts_with_noise_attribution(pkg_tree):
    config = _config(repeats=3)
    good = ([_approved()], [_work_item()])
    bad = ([], [_work_item()])
    for rep in range(1, 4):
        _write_cell("baseline_cap25", rep, *good)
        # capped checklist: complete only in rep1 -> 1/3 -> fails the 2-of-3
        _write_cell("checklist_cap25", rep, *(good if rep == 1 else bad),
                    candidates=[_candidate()],
                    verifications={"kitchen_modernization__kitchen_primary": {
                        "verification_status": "rejected" if rep > 1
                        else "confirmed"}})
        # all_retained: complete in all reps, equivalent candidate evidence
        _write_cell("checklist_all_retained", rep, *good,
                    candidates=[_candidate()],
                    verifications={"kitchen_modernization__kitchen_primary": {
                        "verification_status": "confirmed"}})
    scored = _score(pkg_tree, config=config)
    checklist_capped = scored["outcomes"]["checklist_cap25"][PROP]
    allret = scored["outcomes"]["checklist_all_retained"][PROP]
    assert checklist_capped["passing_reps"] == 1
    assert checklist_capped["passed"] is False
    assert allret["passed"] is True
    # Secondary aggregate verdict survives, labeled as such.
    cap = scored["cap_experiment"][PROP]
    assert cap["verdict"] == "cap_hurts"
    assert scored["cap_experiment"]["_note"].startswith("secondary")
    # Equivalent identity + evidence but divergent 2f decisions -> noise,
    # attributed separately from cap effects (reps 2 and 3).
    noise = scored["pass_2f_disagreements"][PROP]
    assert noise["count"] == 2
    assert {r["rep"] for r in noise["records"]} == {2, 3}
    assert all(r["identity"] == "kitchen_modernization__kitchen"
               for r in noise["records"])
    assert cap["verdict_confidence"] == "reduced_by_2f_noise"
    # Paired per-repeat deltas are the primary signal.
    pair = scored["cap_pairs"][PROP]["2"]
    assert pair["status"] == "paired"
    assert pair["strict"] == {"capped": False, "all_retained": True}
    assert pair["delta"]["package_family_recall"] == 1.0
    assert pair["cap_bound"] is None      # no photo checkpoints in fixture


def test_probe_candidates_count_despite_not_run_audit_only():
    # A no-verification probe stamps audit_only=True (status not_run) on every
    # candidate; that must NOT hide candidates from the candidate-side views.
    probe_candidate = _candidate(audit_only=True,
                                 verification_status="not_run")
    assert pkg._candidate_in_scope(probe_candidate)
    assert not pkg._scoreable_package(probe_candidate)     # post-2f view differs
    whole_home = _candidate(estimate_display_only=True)
    assert not pkg._candidate_in_scope(whole_home)


def test_no_noise_attribution_when_candidate_evidence_differs(pkg_tree):
    # The cap changed the candidate's EVIDENCE (different supporting issue
    # ids), so a 2f disagreement is a real cap effect, not noise.
    good = ([_approved()], [_work_item()])
    _write_cell("baseline_cap25", 1, *good)
    _write_cell("checklist_cap25", 1, *good,
                candidates=[_candidate()],
                verifications={"kitchen_modernization__kitchen_primary": {
                    "verification_status": "confirmed"}})
    richer = _candidate(
        supporting_catalog_item_ids=["cab_dated", "counter_dated"])
    _write_cell("checklist_all_retained", 1, *good,
                candidates=[richer],
                verifications={"kitchen_modernization__kitchen_primary": {
                    "verification_status": "rejected"}})
    config = _config()
    config["package_eval"]["gates"]["pass_reps_required"] = 1
    scored = _score(pkg_tree, config=config)
    assert scored["pass_2f_disagreements"][PROP]["count"] == 0


def test_cap_benign_with_cost_flag(pkg_tree):
    good = ([_approved()], [_work_item()])
    _write_cell("baseline_cap25", 1, *good)
    _write_cell("checklist_cap25", 1, *good, midpoint=50000)
    _write_cell("checklist_all_retained", 1, *good, midpoint=90000)
    config = _config()
    config["package_eval"]["gates"]["pass_reps_required"] = 1
    scored = _score(pkg_tree, config=config)
    cap = scored["cap_experiment"][PROP]
    assert cap["verdict"] == "cap_benign"
    assert any("midpoints differ" in f for f in cap["flags"])
    assert cap["verdict_confidence"] == "normal"
    # The paired row carries the raw midpoint delta (all_retained - capped).
    pair = scored["cap_pairs"][PROP]["1"]
    assert pair["delta"]["final_midpoint"] == 40000
    assert pair["strict"] == {"capped": True, "all_retained": True}


def test_display_only_and_audit_only_packages_are_out_of_scope(pkg_tree):
    whole_home = _approved(
        package_id="interior_paint_flooring_refresh__whole_home",
        package_type="interior_paint_flooring_refresh",
        estimate_unit_id="whole_home", estimate_display_only=True)
    audit = _approved(package_id="bedroom_repair__bedroom_1",
                      package_type="bedroom_repair",
                      estimate_unit_id="bedroom_1", audit_only=True)
    _fill_all_cells([_approved(), whole_home, audit], [_work_item()])
    scored = _score(pkg_tree)
    assert scored["status"] == "final"      # neither pended as an extra


# ---------------------------------------------------------------------------
# Review CSV roundtrip
# ---------------------------------------------------------------------------

def test_review_roundtrip_dedupes_by_signature(pkg_tree):
    extra = _approved(package_id="bedroom_repair__bedroom_1",
                      package_type="bedroom_repair",
                      estimate_unit_id="bedroom_1",
                      pricing_profile="bedroom_repair_heavy")
    config = _config(repeats=2)
    for cell in ("baseline_cap25", "checklist_cap25", "checklist_all_retained"):
        for rep in (1, 2):
            _write_cell(cell, rep, [_approved(), extra], [_work_item()])
    _write_gold(pkg_tree)
    out = pkg.package_review_export(config, _manifest(pkg_tree),
                                    "baseline_vs_checklist")
    with out.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [list(rows[0])] == [pkg.PACKAGE_REVIEW_COLUMNS]
    assert len(rows) == 1                                   # deduped signature
    # Keyed by the aligned canonical room, with the alignment shown.
    assert rows[0]["row_id"] == f"{PROP}|pkg|bedroom_repair__bedroom_A"
    assert rows[0]["room_id"] == "bedroom_A"
    assert rows[0]["alignment_status"] == "aligned"
    assert rows[0]["estimate_unit_id"] == "bedroom_1"
    assert "baseline_cap25:1;2" in rows[0]["occurs_in"]

    rows[0]["human_decision"] = "false_positive"
    rows[0]["reviewer_note"] = "cosmetic only"
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=pkg.PACKAGE_REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    pkg.package_review_import(config, _manifest(pkg_tree),
                              "baseline_vs_checklist", str(out))
    decisions = pkg.load_package_decisions()
    assert decisions[rows[0]["row_id"]]["decision"] == "false_positive"

    scored = pkg.score_package_round(config, _manifest(pkg_tree),
                                     "baseline_vs_checklist",
                                     VALID_GOLD, decisions)
    assert scored["status"] == "final"


def test_review_import_rejects_bad_rows(pkg_tree):
    _fill_all_cells([_approved(), _approved(
        package_id="bedroom_repair__bedroom_1", package_type="bedroom_repair",
        estimate_unit_id="bedroom_1")], [_work_item()])
    _write_gold(pkg_tree)
    out = pkg.package_review_export(_config(), _manifest(pkg_tree),
                                    "baseline_vs_checklist")
    with out.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["human_decision"] = "equivalent"
    rows[0]["equivalent_gold_id"] = "no_such_gold_id"
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=pkg.PACKAGE_REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit, match="not a pkg gold id"):
        pkg.package_review_import(_config(), _manifest(pkg_tree),
                                  "baseline_vs_checklist", str(out))


# ---------------------------------------------------------------------------
# Fingerprint layers
# ---------------------------------------------------------------------------

def test_fingerprint_layers_gold_and_gates_touch_scoring_only(pkg_tree):
    config = _config()
    manifest = _manifest(pkg_tree)
    base_r = pkg.compute_resolution_fingerprint(manifest, config,
                                                "baseline", "checklist")
    base_f = pkg.compute_pass2f_fingerprint(manifest, config,
                                            "baseline", "checklist")
    assert "git_head" not in base_r and "git_head" not in base_f

    # Gold edits and gate tweaks touch neither paid layer.
    edited = json.loads(json.dumps(config))
    edited["package_eval"]["gates"]["cost_flag_pct"] = 99
    assert pkg.compute_resolution_fingerprint(
        manifest, edited, "baseline", "checklist") == base_r
    assert pkg.compute_pass2f_fingerprint(
        manifest, edited, "baseline", "checklist") == base_f

    # A 2f model change invalidates F but not R.
    edited["package_eval"]["pass_2f"]["model"] = "gpt-5.6-terra"
    assert pkg.compute_resolution_fingerprint(
        manifest, edited, "baseline", "checklist") == base_r
    assert pkg.compute_pass2f_fingerprint(
        manifest, edited, "baseline", "checklist") != base_f

    # A ceiling change invalidates R (and therefore F).
    edited2 = json.loads(json.dumps(config))
    edited2["package_eval"]["tail_resolution"]["emergency_ceiling"] = 256
    assert pkg.compute_resolution_fingerprint(
        manifest, edited2, "baseline", "checklist") != base_r
