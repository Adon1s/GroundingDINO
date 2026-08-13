"""Package-outcome extension of the Pass 2a benchmark.

Primary question: do the photos produce the human-defined renovation packages
and material work — without missing scope or approving extra scope — regardless
of intermediate observation count. Motivated by the micro-observation finding
(docs/HANDOFF_pass2a_micro_observation_findings.md): observation counting can't
say whether the checklist prompt's ~1,500 true-but-immaterial claims matter.

Also runs the cap experiment: production max_resolve_per_image=25 vs resolving
every 2c-kept observation ("all_retained"), to classify the cap as benign,
harmful, or masking surplus harm.

Stages (CLI lives in tools/benchmark_pass2a.py; see its --help):

  package-gold template   blank-first authoring template with vocabularies
  package-gold check      validate gold/package_reference.json
  package-eval --stage tail    resolve beyond-cap observations (local Qwen, $0)
  package-eval --stage cells   rebuild all_retained artifacts (offline, 2e re-run)
  package-eval --dry-run       extras-scale report, no 2f, no gold needed
  package-eval --stage 2f      real Pass 2f, resumable, --budget N call cap
  package-review export|import compact review of unmatched approved outcomes
  report                       gains the package-outcome section

Evaluation cells: {baseline,checklist} x cap25 plus checklist x all_retained,
x 3 repeats x 2 properties = 18 instances. Capped cells read the EXISTING
frozen variant artifacts; only checklist_all_retained is rebuilt. Baseline
never exceeds the cap (asserted), so baseline_cap25 doubles as its own
all_retained control.

Fingerprints are layered so a gold edit or review import can never orphan paid
artifacts: Layer R guards tail resolutions + rebuilt cells, Layer F guards 2f
verifications, scoring recomputes freely and records provenance shas.
"""
from __future__ import annotations

import asyncio
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tools.benchmark_pass2a import (
    BENCH_DIR,
    RUNS_DIR,
    _load_json,
    _photo_ckpt_dir,
    _split_ids,
    _truthy,
    _utcnow,
    _write_json,
    guard_fingerprint,
    jaccard,
    load_prompts,
    manifest_photos,
    preflight_embeddings,
    rep_job_id,
    stage_dir_for,
)

PACKAGE_GOLD_PATH = BENCH_DIR / "gold" / "package_reference.json"
PACKAGE_GOLD_TEMPLATE_PATH = BENCH_DIR / "gold" / "package_reference.template.json"
TAIL_DIR = RUNS_DIR / "package_tail_resolution"
CELLS_DIR = RUNS_DIR / "package_cells"
PASS2F_DIR = RUNS_DIR / "package_2f"
REVIEW_DIR = RUNS_DIR / "package_review"
EVAL_DIR = RUNS_DIR / "package_eval"

PACKAGE_GOLD_SCHEMA_VERSION = "package_reference_v1"

REVIEW_DECISIONS = ("false_positive", "equivalent", "gold_gap")

PACKAGE_REVIEW_COLUMNS = [
    "row_id", "property", "outcome_kind", "package_type", "estimate_unit_id",
    "catalog_item_id", "item_name", "observed_pricing_profiles",
    "cost_low", "cost_high", "occurs_in", "evidence_photos", "evidence_summary",
    "human_decision", "equivalent_gold_id", "reviewer_note",
]


def cells_for_round(variant_a: str, variant_b: str) -> Dict[str, Tuple[str, str]]:
    """{cell_name: (variant, cap_mode)}. Variant A doubles as its own
    all_retained control because it never exceeds the cap (asserted in tail)."""
    return {
        f"{variant_a}_cap25": (variant_a, "cap25"),
        f"{variant_b}_cap25": (variant_b, "cap25"),
        f"{variant_b}_all_retained": (variant_b, "all_retained"),
    }


def package_round_variants(config: Dict[str, Any], round_label: str) -> Tuple[str, str]:
    variant_a, _, variant_b = round_label.partition("_vs_")
    prompts = load_prompts()
    if not variant_a or not variant_b or variant_a not in prompts \
            or variant_b not in prompts:
        raise SystemExit("--round must look like baseline_vs_checklist")
    return variant_a, variant_b


def package_eval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    block = config.get("package_eval")
    if not block:
        raise SystemExit("config.json has no package_eval block")
    return block


# ---------------------------------------------------------------------------
# Pricing-profile adjacency (symmetric one step, derived from the escalation map)
# ---------------------------------------------------------------------------

def pricing_profile_adjacency() -> Dict[str, Set[str]]:
    """{profile: set of directly adjacent profiles}, one step either direction.

    Derived from rehab_packages._PRICING_TIER_ESCALATION (the one-step-up map)
    plus its inverse, so a catalog-side tier change shows up here — and in the
    scoring provenance sha — automatically.
    """
    from tools.rehab_packages import _PRICING_TIER_ESCALATION
    adjacency: Dict[str, Set[str]] = {}
    for lower, spec in _PRICING_TIER_ESCALATION.items():
        upper = spec[0]
        adjacency.setdefault(lower, set()).add(upper)
        adjacency.setdefault(upper, set()).add(lower)
    return adjacency


def pricing_profile_universe() -> Set[str]:
    return set(pricing_profile_adjacency())


def profiles_for_room(room: str) -> List[str]:
    prefix = f"{room}_"
    return sorted(p for p in pricing_profile_universe() if p.startswith(prefix))


# ---------------------------------------------------------------------------
# Two-phase v4 helper (probe -> confirm-all -> recompute), shared by template,
# dry-run and candidate extraction. Mirrors compute_pre2f_totals but returns
# the full v4 payloads instead of slim totals.
# ---------------------------------------------------------------------------

def artifact_issues_flat(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (
        artifact.get("product_estimate_issues_flat")
        if artifact.get("estimate_issues_flat")
        else artifact.get("product_issues_flat")
    ) or []


def two_phase_v4(artifact: Dict[str, Any],
                 catalog: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(probe_v4, force_confirmed_v4) for one property artifact."""
    from tools.renovation_estimate_v4 import compute_renovation_estimate_v4
    issues_flat = artifact_issues_flat(artifact)
    photos = artifact.get("photos") or {}
    metadata = artifact.get("property_metadata") or None
    probe = compute_renovation_estimate_v4(
        issues_flat, catalog, photos, property_metadata=metadata)
    package_ids = sorted({
        str(p.get("package_id")) for p in (probe.get("package_candidates") or [])
        if isinstance(p, dict) and p.get("package_id")
    })
    verifications = {pid: {"package_id": pid, "verification_status": "confirmed"}
                     for pid in package_ids}
    confirmed = compute_renovation_estimate_v4(
        issues_flat, catalog, photos, property_metadata=metadata,
        package_verifications=verifications)
    return probe, confirmed


def v4_line_items(v4: Dict[str, Any], *, valid_only: bool = True) -> List[Dict[str, Any]]:
    """Flatten v4 groups into the line-item projection scoring reads.

    billable_estimate_unit_id is the room unit; the line item's own
    estimate_unit_id is a synthetic cluster key and deliberately not exported.
    """
    out: List[Dict[str, Any]] = []
    for group in v4.get("groups") or []:
        for li in group.get("line_items") or []:
            if valid_only and not li.get("is_valid_detection"):
                continue
            out.append({
                "catalog_item_id": li.get("catalog_item_id"),
                "name": li.get("name"),
                "billable_estimate_unit_id": li.get("billable_estimate_unit_id"),
                "cost_low": int(li.get("cost_low") or 0),
                "cost_high": int(li.get("cost_high") or 0),
                "package_id": li.get("package_id"),
                "trade_bucket": li.get("trade_bucket"),
                "is_valid_detection": bool(li.get("is_valid_detection")),
            })
    return out


def _scoreable_package(pkg: Dict[str, Any]) -> bool:
    """POST-2F approved packages in scope: whole-home display-only and
    audit-only are excluded from truth matching."""
    return not pkg.get("estimate_display_only") and not pkg.get("audit_only")


def _candidate_in_scope(pkg: Dict[str, Any]) -> bool:
    """PRE-2F candidates in scope. Only the whole-home display aggregate is
    excluded — a probe candidate is audit_only merely because its status is
    not_run, which says nothing about the candidate itself."""
    return not pkg.get("estimate_display_only")


def package_projection(pkg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "package_id": pkg.get("package_id"),
        "package_type": pkg.get("package_type"),
        "estimate_unit_id": pkg.get("estimate_unit_id"),
        "pricing_profile": pkg.get("pricing_profile"),
        "pricing_tier": pkg.get("pricing_tier"),
        "package_strength": pkg.get("package_strength"),
        "verification_status": pkg.get("verification_status"),
        "estimate_eligible": bool(pkg.get("estimate_eligible")),
        "audit_only": bool(pkg.get("audit_only")),
        "estimate_display_only": bool(pkg.get("estimate_display_only")),
        "package_level": pkg.get("package_level"),
        "cost_low": int(pkg.get("cost_low") or 0),
        "cost_high": int(pkg.get("cost_high") or 0),
        "cost_midpoint": int(pkg.get("cost_midpoint") or 0),
        "supporting_catalog_item_ids": list(pkg.get("supporting_catalog_item_ids") or []),
        "review_photo_keys": list(pkg.get("review_photo_keys") or []),
        "evidence_summary": pkg.get("evidence_summary") or "",
    }


# ---------------------------------------------------------------------------
# Gold: template + check
# ---------------------------------------------------------------------------

def _authorable_package_types() -> List[str]:
    """interior_paint_flooring_refresh is derived downstream and display-only —
    it can never be an expected package."""
    from tools.rehab_packages import (
        PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH, VALID_PACKAGE_TYPES)
    return sorted(VALID_PACKAGE_TYPES - {PACKAGE_TYPE_INTERIOR_PAINT_FLOORING_REFRESH})


def _package_type_room(package_type: str) -> Optional[str]:
    from tools.rehab_packages import _PACKAGE_TYPE_TO_ROOM
    return _PACKAGE_TYPE_TO_ROOM.get(package_type)


def observed_units_by_property(config: Dict[str, Any], manifest: Dict[str, Any],
                               catalog: Dict[str, Any],
                               variants: Tuple[str, ...] = ("baseline", "checklist"),
                               ) -> Dict[str, List[str]]:
    """Estimate units the pipeline produced for each property, unioned across
    every available cell (both variants' capped artifacts plus any rebuilt
    all_retained cells). Unit *identity* derives from the frozen photos, but
    unit *production* varies by variant — baseline may never form a package or
    price a line in a room the checklist does, so a single-variant universe
    would wrongly reject valid gold units."""
    repeats = int(config["repeats"])
    out: Dict[str, Set[str]] = {}
    for prop_key in sorted(manifest["properties"]):
        paths: List[Path] = []
        for variant in variants:
            stage = f"variant_{variant}"
            for rep in range(1, repeats + 1):
                paths.append(stage_dir_for(stage) / f"rep{rep}" / prop_key
                             / rep_job_id(stage, rep) / "photo_intel.json")
        for cell_dir in sorted(CELLS_DIR.glob("*")) if CELLS_DIR.is_dir() else []:
            for rep in range(1, repeats + 1):
                paths.append(cell_dir / f"rep{rep}" / prop_key
                             / f"pkgcell_{cell_dir.name}_rep{rep}"
                             / "photo_intel.json")
        units: Set[str] = set()
        for path in paths:
            if not path.is_file():
                continue
            probe, confirmed = two_phase_v4(_load_json(path), catalog)
            units.update(
                str(p.get("estimate_unit_id"))
                for p in (probe.get("package_candidates") or [])
                if p.get("estimate_unit_id") and _candidate_in_scope(p))
            units.update(
                str(li["billable_estimate_unit_id"])
                for li in v4_line_items(confirmed)
                if li.get("billable_estimate_unit_id"))
        out[prop_key] = units
    return {prop: sorted(units) for prop, units in out.items()}


def package_gold_template(config: Dict[str, Any], manifest: Dict[str, Any]) -> Path:
    """Blank-first authoring template. Vocabularies only — no model-proposed
    content, so the model never defines the universe of expected outcomes
    (same philosophy as tools/benchmarking/reference_authoring.py)."""
    from tools import pipeline_config as cfg
    from tools.artifact_writers import load_issue_catalog
    from tools.renovation_estimate import product_quarantined_trade_buckets

    if PACKAGE_GOLD_TEMPLATE_PATH.is_file():
        raise SystemExit(f"template already exists, refusing to overwrite: "
                         f"{PACKAGE_GOLD_TEMPLATE_PATH}")
    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)
    quarantined = product_quarantined_trade_buckets(catalog)
    catalog_rows: List[Dict[str, str]] = []
    quarantined_rows: List[Dict[str, str]] = []
    for item in catalog.get("items") or []:
        row = {"catalog_item_id": item.get("id"), "name": item.get("name"),
               "trade_bucket": item.get("trade_bucket")}
        if item.get("trade_bucket") in quarantined:
            quarantined_rows.append(row)
        else:
            catalog_rows.append(row)

    adjacency = pricing_profile_adjacency()
    template = {
        "schema_version": PACKAGE_GOLD_SCHEMA_VERSION,
        "authored_at": "",
        "author": "",
        "closed_world_note": (
            "Unlisted approved packages and unlisted cost-bearing standalone "
            "work are provisional false positives. Evidence and package-member "
            "observations are never extras. Whole-home display-only and "
            "audit-only packages are out of scope."),
        "properties": {
            prop: {"expected_packages": [], "required_work_items": [], "notes": ""}
            for prop in sorted(manifest["properties"])
        },
        "_vocabulary": {
            "note": "Reference only — delete nothing, the check ignores this block.",
            "package_types": _authorable_package_types(),
            "pricing_profiles_by_room": {
                room: profiles_for_room(room)
                for room in sorted({_package_type_room(t)
                                    for t in _authorable_package_types()})
            },
            "pricing_profile_adjacency": {
                k: sorted(v) for k, v in sorted(adjacency.items())},
            "observed_estimate_units": observed_units_by_property(
                config, manifest, catalog),
            "catalog_items": catalog_rows,
            "quarantined_not_authorable": quarantined_rows,
            "expected_package_fields": {
                "package_type": "one of package_types",
                "estimate_unit_id": "one of observed_estimate_units for the property",
                "target_pricing_profile": "family-qualified, e.g. kitchen_full_rehab",
                "adjacent_acceptable": "bool — one adjacent profile also passes",
                "diagnostic_cost_range": "optional {low, high}, report-only",
                "rationale": "short human reason + photo refs",
            },
            "required_work_item_fields": {
                "catalog_item_id": "non-quarantined catalog id",
                "estimate_unit_id": "room unit the work belongs to",
                "rationale": "short human reason + photo refs",
            },
        },
    }
    _write_json(PACKAGE_GOLD_TEMPLATE_PATH, template)
    print(f"template -> {PACKAGE_GOLD_TEMPLATE_PATH}\n"
          "Author expected packages + required work, save as "
          f"{PACKAGE_GOLD_PATH.name}, then run package-gold check.")
    return PACKAGE_GOLD_TEMPLATE_PATH


def load_package_gold() -> Dict[str, Any]:
    if not PACKAGE_GOLD_PATH.is_file():
        raise SystemExit(f"package gold missing: {PACKAGE_GOLD_PATH} — author it "
                         "from the package-gold template")
    return _load_json(PACKAGE_GOLD_PATH)


def package_gold_ids(gold: Dict[str, Any], prop: str) -> Dict[str, Set[str]]:
    entry = (gold.get("properties") or {}).get(prop) or {}
    return {
        "pkg": {f"{p['package_type']}__{p['estimate_unit_id']}"
                for p in entry.get("expected_packages") or []},
        "item": {f"{w['catalog_item_id']}@{w['estimate_unit_id']}"
                 for w in entry.get("required_work_items") or []},
    }


def package_gold_check(config: Dict[str, Any], manifest: Dict[str, Any],
                       gold: Optional[Dict[str, Any]] = None) -> str:
    """Validate the authored gold; returns its sha256 on success."""
    from tools import pipeline_config as cfg
    from tools.artifact_writers import load_issue_catalog
    from tools.comparison_common import sha256_file
    from tools.renovation_estimate import product_quarantined_trade_buckets

    gold = gold if gold is not None else load_package_gold()
    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)
    quarantined = product_quarantined_trade_buckets(catalog)
    catalog_by_id = {item.get("id"): item for item in catalog.get("items") or []}
    valid_types = set(_authorable_package_types())
    profile_universe = pricing_profile_universe()
    observed_units = observed_units_by_property(config, manifest, catalog)
    errors: List[str] = []

    if gold.get("schema_version") != PACKAGE_GOLD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PACKAGE_GOLD_SCHEMA_VERSION}")
    gold_props = set(gold.get("properties") or {})
    config_props = set(config["properties"])
    if gold_props != config_props:
        errors.append(f"properties mismatch: gold {sorted(gold_props)} vs "
                      f"config {sorted(config_props)}")

    for prop, entry in sorted((gold.get("properties") or {}).items()):
        units = set(observed_units.get(prop) or [])
        seen_pkg: Set[Tuple[str, str]] = set()
        for pkg in entry.get("expected_packages") or []:
            ptype = pkg.get("package_type")
            unit = pkg.get("estimate_unit_id")
            profile = pkg.get("target_pricing_profile")
            where = f"{prop} expected package {ptype}__{unit}"
            if ptype not in valid_types:
                errors.append(f"{where}: unknown or non-authorable package_type")
                continue
            room = _package_type_room(ptype)
            if unit not in units:
                errors.append(f"{where}: estimate_unit_id {unit!r} was never "
                              f"produced for this property (known: {sorted(units)})")
            if profile not in profile_universe:
                errors.append(f"{where}: unknown target_pricing_profile {profile!r}")
            elif room and not str(profile).startswith(f"{room}_"):
                errors.append(f"{where}: profile {profile!r} is not in the "
                              f"{room} family")
            if (ptype, unit) in seen_pkg:
                errors.append(f"{where}: duplicate (package_type, unit)")
            seen_pkg.add((ptype, unit))
            if not (pkg.get("rationale") or "").strip():
                errors.append(f"{where}: rationale is required")
        seen_item: Set[Tuple[str, str]] = set()
        for work in entry.get("required_work_items") or []:
            cid = work.get("catalog_item_id")
            unit = work.get("estimate_unit_id")
            where = f"{prop} required work {cid}@{unit}"
            item = catalog_by_id.get(cid)
            if item is None:
                errors.append(f"{where}: unknown catalog_item_id")
            elif item.get("trade_bucket") in quarantined:
                errors.append(
                    f"{where}: trade {item.get('trade_bucket')!r} is "
                    "product-quarantined — quarantined work cannot be required "
                    "(it can never be satisfied by the pipeline)")
            if unit not in units:
                errors.append(f"{where}: estimate_unit_id {unit!r} was never "
                              f"produced for this property")
            if (cid, unit) in seen_item:
                errors.append(f"{where}: duplicate (catalog_item_id, unit)")
            seen_item.add((cid, unit))
            if not (work.get("rationale") or "").strip():
                errors.append(f"{where}: rationale is required")

    if errors:
        raise SystemExit("package gold check FAILED:\n  " + "\n  ".join(errors))
    sha = sha256_file(PACKAGE_GOLD_PATH)
    print(f"package gold OK — sha256 {sha}")
    return sha


# ---------------------------------------------------------------------------
# Fingerprint layers. git head goes in a sibling info.json, never the compared
# dict: harness commits must not orphan paid Qwen/2f artifacts.
# ---------------------------------------------------------------------------

def _variant_fp_shas(variants: Tuple[str, ...]) -> Dict[str, Optional[str]]:
    from tools.comparison_common import sha256_canonical
    out: Dict[str, Optional[str]] = {}
    for variant in variants:
        path = stage_dir_for(f"variant_{variant}") / "fingerprint.json"
        out[variant] = sha256_canonical(_load_json(path)) if path.is_file() else None
    return out


def compute_resolution_fingerprint(manifest: Dict[str, Any],
                                   config: Dict[str, Any],
                                   variant_a: str, variant_b: str) -> Dict[str, Any]:
    from tools import pipeline_config as cfg
    from tools.benchmark_pass2a import _manifest_hashes
    from tools.comparison_common import sha256_canonical, sha256_file
    tail_cfg = package_eval_config(config).get("tail_resolution") or {}
    return {
        "layer": "package_resolution",
        "image_hashes_sha": sha256_canonical(
            _manifest_hashes(manifest, "image_sha256")),
        "frozen_2a_sha": sha256_canonical(
            _manifest_hashes(manifest, "frozen_2a_sha256")),
        "source_variant_fp_shas": _variant_fp_shas((variant_a, variant_b)),
        "catalog_sha256": sha256_file(Path(cfg.ISSUE_CATALOG_PATH)),
        "embeddings_model": cfg.EMBEDDINGS_MODEL_NAME,
        "qwen_2d": {"model": cfg.LM_STUDIO_MODEL, "provider": "lmstudio",
                    "temperature": 0.2},
        "top_k": int(tail_cfg.get("top_k", 8)),
        "emergency_ceiling": int(tail_cfg.get("emergency_ceiling", 128)),
        "repeats": int(config["repeats"]),
        "pipeline_mode": "publish",
    }


def compute_pass2f_fingerprint(manifest: Dict[str, Any], config: Dict[str, Any],
                               variant_a: str, variant_b: str) -> Dict[str, Any]:
    from tools.comparison_common import sha256_canonical, sha256_file
    from tools.scene_classifier_passes import PASS_2F_PROMPT_VERSION
    resolution = compute_resolution_fingerprint(manifest, config,
                                                variant_a, variant_b)
    artifact_shas: Dict[str, str] = {}
    for cell in cells_for_round(variant_a, variant_b):
        for rep in range(1, int(config["repeats"]) + 1):
            for prop in sorted(manifest["properties"]):
                path = cell_artifact_path(cell, rep, prop, variant_a, variant_b)
                artifact_shas[f"{cell}/rep{rep}/{prop}"] = (
                    sha256_file(path) if path.is_file() else "missing")
    p2f = dict(package_eval_config(config).get("pass_2f") or {})
    return {
        "layer": "package_pass_2f",
        "resolution_layer_sha": sha256_canonical(resolution),
        "artifact_shas": artifact_shas,
        "pass_2f": p2f,
        "prompt_template_version": PASS_2F_PROMPT_VERSION,
        "strict": True,
    }


def _write_stage_info(stage_dir: Path) -> None:
    from tools.benchmark_pass2a import _git_head
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stage_dir / "info.json",
                {"git_head": _git_head(), "written_at": _utcnow()})


# ---------------------------------------------------------------------------
# Stage: tail resolution (all_retained, local Qwen)
# ---------------------------------------------------------------------------

def _checkpoint_lanes(ckpt: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    scene_data = ckpt.get("scene_data") or {}
    kept = [o for o in (scene_data.get("observations") or [])
            if isinstance(o, dict) and (o.get("description") or "").strip()]
    resolved = [r for r in (scene_data.get("resolved_items") or [])
                if isinstance(r, dict)]
    return kept, resolved


def assert_variant_never_capped(config: Dict[str, Any], manifest: Dict[str, Any],
                                variant: str) -> None:
    """The reuse of variant A's capped cells as its all_retained control is only
    valid if A never hit the cap. Abort loudly if that assumption breaks."""
    for rep in range(1, int(config["repeats"]) + 1):
        for prop_key, photo in manifest_photos(manifest):
            ckpt_path = (_photo_ckpt_dir(
                stage_dir_for(f"variant_{variant}") / f"rep{rep}" / prop_key)
                / f"{photo['photo_key']}.json")
            kept, resolved = _checkpoint_lanes(_load_json(ckpt_path))
            if len(kept) > len(resolved):
                raise SystemExit(
                    f"variant_{variant} rep{rep} {prop_key}/{photo['photo_key']}"
                    f" was capped ({len(kept)} kept > {len(resolved)} resolved)"
                    " — the baseline-reuse assumption is broken; the "
                    f"{variant}_all_retained cell must be built explicitly.")


def stage_package_tail(config: Dict[str, Any], manifest: Dict[str, Any],
                       round_label: str, *, skip_preflight: bool = False) -> Path:
    from tools import pipeline_config as cfg
    from tools.artifact_writers import load_issue_catalog
    from tools.catalog_embeddings import build_candidate_provider
    from tools.scene_classifier_orchestrator import resolve_observation_against_catalog
    from tools.vlm_client import create_vlm_client

    variant_a, variant_b = package_round_variants(config, round_label)
    tail_cfg = package_eval_config(config).get("tail_resolution") or {}
    ceiling = int(tail_cfg.get("emergency_ceiling", 128))
    top_k = int(tail_cfg.get("top_k", 8))

    if not skip_preflight and not preflight_embeddings():
        raise SystemExit(2)
    assert_variant_never_capped(config, manifest, variant_a)
    guard_fingerprint(TAIL_DIR, compute_resolution_fingerprint(
        manifest, config, variant_a, variant_b))
    _write_stage_info(TAIL_DIR)

    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)
    candidate_provider = build_candidate_provider(catalog)
    vlm_client = create_vlm_client()
    qwen_config = {"url": cfg.LM_STUDIO_URL, "model": cfg.LM_STUDIO_MODEL,
                   "provider": "lmstudio"}
    images_root = Path(manifest["images_root"])

    done = skipped = resolved_count = 0
    for rep in range(1, int(config["repeats"]) + 1):
        for prop_key, photo in manifest_photos(manifest):
            photo_key = photo["photo_key"]
            out_path = TAIL_DIR / f"rep{rep}" / prop_key / f"{photo_key}.json"
            if out_path.is_file():
                skipped += 1
                continue
            ckpt_path = (_photo_ckpt_dir(
                stage_dir_for(f"variant_{variant_b}") / f"rep{rep}" / prop_key)
                / f"{photo_key}.json")
            kept, resolved = _checkpoint_lanes(_load_json(ckpt_path))
            if len(kept) > ceiling:
                raise SystemExit(
                    f"{prop_key}/{photo_key} rep{rep}: {len(kept)} kept "
                    f"observations exceed the emergency ceiling of {ceiling} — "
                    "aborting rather than truncating. Raise the ceiling "
                    "deliberately in config if this is intended.")
            resolved_ids = {str(r.get("issue_id")) for r in resolved
                            if r.get("issue_id")}
            tail = [o for o in kept if str(o.get("issue_id")) not in resolved_ids]
            rows: List[Dict[str, Any]] = []
            debug_rows: List[Dict[str, Any]] = []
            if tail:
                print(f"[package tail] rep{rep} {prop_key}/{photo_key}: "
                      f"{len(tail)} of {len(kept)} kept ...")
            for obs in tail:
                resolved_row, debug_row, _ = asyncio.run(
                    resolve_observation_against_catalog(
                        vlm_client=vlm_client,
                        model_config=dict(qwen_config),
                        candidate_provider=candidate_provider,
                        observation=obs,
                        base_context={
                            "scene": obs.get("scene"),
                            "scene_group": obs.get("scene_group"),
                            "top_k_candidates": top_k,
                        },
                        top_k=top_k,
                        source_image_path=str(images_root / prop_key / photo_key),
                    ))
                debug_rows.append(debug_row)
                if resolved_row is not None:
                    rows.append(resolved_row)
                    if resolved_row.get("resolved_item_id"):
                        resolved_count += 1
            _write_json(out_path, {
                "property_key": prop_key, "photo_key": photo_key, "rep": rep,
                "kept_count": len(kept), "already_resolved": len(resolved),
                "tail_count": len(tail), "resolved_rows": rows,
                "debug_rows": debug_rows, "resolved_at": _utcnow(),
            })
            done += 1
    print(f"package tail: {done} photos resolved ({resolved_count} new catalog "
          f"ids), {skipped} resumed -> {TAIL_DIR}")
    return TAIL_DIR


# ---------------------------------------------------------------------------
# Stage: rebuild all_retained cell artifacts (offline, deterministic 2e)
# ---------------------------------------------------------------------------

def _catalog_meta_by_id(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Same projection the orchestrator builds for 2e policy gating
    (scene_classifier_orchestrator.py:475-486) without constructing the
    orchestrator (which would probe the embeddings sidecar)."""
    out: Dict[str, Dict[str, Any]] = {}
    for item in catalog.get("items") or []:
        item_id = (item.get("id") or "").strip()
        if item_id:
            out[item_id] = {
                "tier": item.get("tier", "work"),
                "drop_if_generic": bool(item.get("drop_if_generic", False)),
                "defaultHidden": bool(item.get("defaultHidden", False)),
                "trade_bucket": item.get("trade_bucket", ""),
            }
    return out


def rerun_pass_2e(scene_data: Dict[str, Any],
                  catalog_meta: Dict[str, Dict[str, Any]]) -> None:
    """Re-run deterministic Pass 2e over scene_data in place, mirroring the
    orchestrator's invocation (scene_classifier_orchestrator.py:1014-1109).
    run_pass_2e makes no model call — vlm_client/model_config are inert."""
    from tools.observation_kinds import OBSERVATION_KINDS
    from tools.scene_classifier_passes import run_pass_2e

    resolution_index = {
        str(row["issue_id"]): row
        for row in (scene_data.get("resolved_items") or [])
        if row.get("issue_id")
    }
    issues_for_2e: List[Dict[str, Any]] = []
    for obs in scene_data.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        issue = dict(obs)
        res = resolution_index.get(str(issue.get("issue_id") or ""))
        if res and res.get("resolved_item_id"):
            issue.setdefault("catalogItemId", res["resolved_item_id"])
            if res.get("resolved_kind") in OBSERVATION_KINDS:
                issue["kind"] = res["resolved_kind"]
                issue["catalogItemKind"] = res["resolved_kind"]
        issues_for_2e.append(issue)

    context = {"catalog_meta_by_id": catalog_meta,
               "policy": {"include_optional": False, "mode": "renovator_strict"}}
    result = asyncio.run(run_pass_2e(
        vlm_client=None, model_config={}, verified_issues=issues_for_2e,
        context=context))

    scene_data["verified_issues"] = result.display_issues or result.verified_issues or []
    scene_data["matched_issues"] = result.canonical_issues or result.matched_issues or []
    scene_data["canonical_issues"] = scene_data["matched_issues"]
    scene_data["display_issues"] = scene_data["verified_issues"]
    removed = result.removed_invalid or result.removed or []
    suppressed = result.display_suppressed_issues or result.suppressed_issues or []
    scene_data.setdefault("passes", {})["2e"] = {
        "notes": result.notes,
        "input_count": result.input_count,
        "deduped_count": result.deduped_count,
        "final_count": result.final_count,
        "canonical_count": len(scene_data["canonical_issues"]),
        "display_count": len(scene_data["display_issues"]),
        "removed_count": result.removed_count,
        "removed_reason_counts": result.removed_reason_counts,
        "suppressed_reason_counts": result.suppressed_reason_counts,
        "suppressed_samples": result.suppressed_samples,
        "kept_issue_ids": [x["issue_id"] for x in scene_data["verified_issues"]
                           if x.get("issue_id")],
        "canonical_issue_ids": [x["issue_id"] for x in scene_data["canonical_issues"]
                                if x.get("issue_id")],
        "removed": [{"issue_id": x.get("issue_id"),
                     "description": x.get("description", ""),
                     "reason": x.get("removed_reason", "")} for x in removed],
        "suppressed": [{"issue_id": x.get("issue_id"),
                        "description": x.get("description", ""),
                        "reason": x.get("suppressed_reason", "")}
                       for x in suppressed],
    }
    scene_data.setdefault("debug", {})["pass_2e_summary"] = {
        "input_count": result.input_count,
        "deduped_count": result.deduped_count,
        "final_count": result.final_count,
        "canonical_count": len(scene_data["canonical_issues"]),
        "display_count": len(scene_data["display_issues"]),
        "removed_count": result.removed_count,
        "removed_reason_counts": result.removed_reason_counts,
        "suppressed_reason_counts": result.suppressed_reason_counts,
        "notes": result.notes,
    }


def cell_artifact_path(cell: str, rep: int, prop: str,
                       variant_a: str, variant_b: str) -> Path:
    cells = cells_for_round(variant_a, variant_b)
    variant, mode = cells[cell]
    if mode == "cap25":
        stage = f"variant_{variant}"
        return (stage_dir_for(stage) / f"rep{rep}" / prop
                / rep_job_id(stage, rep) / "photo_intel.json")
    return (CELLS_DIR / cell / f"rep{rep}" / prop
            / f"pkgcell_{cell}_rep{rep}" / "photo_intel.json")


def stage_package_cells(config: Dict[str, Any], manifest: Dict[str, Any],
                        round_label: str) -> Path:
    from tools import pipeline_config as cfg
    from tools.analyzer_cli import ImageResult, PropertyAnalysisJob
    from tools.artifact_writers import load_issue_catalog, write_photo_intel
    from tools.benchmark_pass2a import ensure_totals
    from tools.vlm_client import get_model_configs_from_pipeline_config

    variant_a, variant_b = package_round_variants(config, round_label)
    cell = f"{variant_b}_all_retained"
    guard_fingerprint(CELLS_DIR, compute_resolution_fingerprint(
        manifest, config, variant_a, variant_b))
    _write_stage_info(CELLS_DIR)

    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)
    catalog_meta = _catalog_meta_by_id(catalog)
    _, gpt5_config = get_model_configs_from_pipeline_config(cfg)

    built = skipped = 0
    for rep in range(1, int(config["repeats"]) + 1):
        for prop_key in sorted(manifest["properties"]):
            artifact_path = cell_artifact_path(cell, rep, prop_key,
                                               variant_a, variant_b)
            if artifact_path.is_file():
                ensure_totals(artifact_path, catalog)
                skipped += 1
                continue
            print(f"[package cells] {cell} rep{rep} {prop_key} ...")
            results: List[Any] = []
            total_time = 0.0
            for photo in manifest["properties"][prop_key]["photos"]:
                photo_key = photo["photo_key"]
                ckpt = _load_json(_photo_ckpt_dir(
                    stage_dir_for(f"variant_{variant_b}") / f"rep{rep}"
                    / prop_key) / f"{photo_key}.json")
                tail_path = TAIL_DIR / f"rep{rep}" / prop_key / f"{photo_key}.json"
                if not tail_path.is_file():
                    raise SystemExit(f"tail resolution missing for rep{rep} "
                                     f"{prop_key}/{photo_key} — run --stage tail")
                tail = _load_json(tail_path)
                scene_data = json.loads(json.dumps(ckpt.get("scene_data") or {}))
                merged = list(scene_data.get("resolved_items") or [])
                merged.extend(tail.get("resolved_rows") or [])
                scene_data["resolved_items"] = merged
                debug = scene_data.setdefault("debug", {})
                gate = debug.setdefault("pass_2d_gate", {})
                gate["total_resolve_count"] = len(merged)
                gate["all_retained_rebuild"] = True
                per_obs = list(debug.get("pass_2d_per_observation") or [])
                per_obs.extend(tail.get("debug_rows") or [])
                debug["pass_2d_per_observation"] = per_obs
                rerun_pass_2e(scene_data, catalog_meta)
                results.append(ImageResult(
                    image_path=ckpt["image_path"],
                    scene_data=scene_data,
                    scene=ckpt.get("scene") or scene_data.get("scene") or "unknown",
                    processing_time=float(ckpt.get("processing_time") or 0.0),
                ))
                total_time += float(ckpt.get("processing_time") or 0.0)
            job = PropertyAnalysisJob(
                property_key=prop_key,
                job_id=artifact_path.parent.name,
                artifacts_dir=str(artifact_path.parent),
                timestamp=_utcnow(),
                results=results,
                total_processing_time=total_time,
                property_metadata=(manifest["properties"][prop_key]
                                   .get("property_metadata") or None),
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            write_photo_intel(
                cfg=cfg,
                job=job,
                detection_backend="dinox",
                analysis_profile="standard",
                use_pass_architecture=True,
                pass_toggles=dict(config["pass_toggles"]),
                model_overrides=dict(config["model_overrides"]),
                gpt_config=gpt5_config,
                issue_catalog=catalog,
                vlm_client=None,
                reasoning_efforts=dict(config["reasoning_efforts"]),
                dependency_status={"embeddings": "ready"},
            )
            ensure_totals(artifact_path, catalog)
            built += 1
    print(f"package cells: {built} rebuilt, {skipped} already present "
          f"-> {CELLS_DIR / cell}")
    return CELLS_DIR / cell


# ---------------------------------------------------------------------------
# Stage: candidates + real Pass 2f (or the no-2f dry run)
# ---------------------------------------------------------------------------

def _cell_dir(cell: str, rep: int, prop: str) -> Path:
    return PASS2F_DIR / cell / f"rep{rep}" / prop


def _load_cell_artifacts(config: Dict[str, Any], manifest: Dict[str, Any],
                         variant_a: str, variant_b: str) -> Dict[Tuple[str, int, str], Path]:
    out: Dict[Tuple[str, int, str], Path] = {}
    missing: List[str] = []
    for cell in cells_for_round(variant_a, variant_b):
        for rep in range(1, int(config["repeats"]) + 1):
            for prop in sorted(manifest["properties"]):
                path = cell_artifact_path(cell, rep, prop, variant_a, variant_b)
                if path.is_file():
                    out[(cell, rep, prop)] = path
                else:
                    missing.append(f"{cell}/rep{rep}/{prop}")
    if missing:
        raise SystemExit("cell artifacts missing (run tail + cells stages "
                         "first): " + ", ".join(missing))
    return out


def stage_package_2f(config: Dict[str, Any], manifest: Dict[str, Any],
                     round_label: str, *, dry_run: bool = False,
                     budget: Optional[int] = None) -> Path:
    from tools import pipeline_config as cfg
    from tools.artifact_writers import load_issue_catalog

    variant_a, variant_b = package_round_variants(config, round_label)
    artifacts = _load_cell_artifacts(config, manifest, variant_a, variant_b)
    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)

    if dry_run:
        return _package_dry_run(config, manifest, round_label, artifacts, catalog,
                                variant_a, variant_b)

    from tools.pass_2f_artifact_inputs import photo_key_to_path as build_photo_map
    from tools.pass_config import resolve_openai_invocation
    from tools.rehab_packages import run_pass_2f_batch
    from tools.renovation_estimate_v4 import compute_renovation_estimate_v4
    from tools.scene_classifier_passes import PASS_2F_PROMPT_VERSION
    from tools.vlm_client import create_vlm_client, get_model_configs_from_pipeline_config

    guard_fingerprint(PASS2F_DIR, compute_pass2f_fingerprint(
        manifest, config, variant_a, variant_b))
    _write_stage_info(PASS2F_DIR)

    p2f_cfg = dict(package_eval_config(config).get("pass_2f") or {})
    if not p2f_cfg.get("model"):
        raise SystemExit("config package_eval.pass_2f.model is required")
    _, gpt5_config = get_model_configs_from_pipeline_config(cfg)
    model_config = resolve_openai_invocation(
        "2f",
        {**gpt5_config, "model": p2f_cfg["model"], "provider": "openai"},
        p2f_cfg.get("reasoning_effort"),
    )
    vlm_client = create_vlm_client()

    calls_made = 0
    done = skipped = 0
    for (cell, rep, prop), artifact_path in sorted(artifacts.items()):
        out_dir = _cell_dir(cell, rep, prop)
        if (out_dir / "verifications.json").is_file():
            skipped += 1
            continue
        if budget is not None and calls_made >= budget:
            print(f"budget of {budget} VLM package calls reached after "
                  f"{done} cells — rerun to resume ({calls_made} calls made).")
            return PASS2F_DIR
        artifact = _load_json(artifact_path)
        probe, _ = two_phase_v4(artifact, catalog)
        candidates = [p for p in (probe.get("package_candidates") or [])
                      if isinstance(p, dict)]
        _write_json(out_dir / "candidates.json", {
            "cell": cell, "rep": rep, "property_key": prop,
            "candidates": [package_projection(p) for p in candidates],
        })
        print(f"[package 2f] {cell} rep{rep} {prop}: "
              f"{len(candidates)} candidates ...")
        verifications, trace = asyncio.run(run_pass_2f_batch(
            candidates,
            vlm_client=vlm_client,
            model_config=model_config,
            photo_key_to_path=build_photo_map(artifact),
            provider="premium",
            max_images=int(p2f_cfg.get("max_images", 3)),
            strict=True,
        ))
        drifted = [pid for pid, rec in verifications.items()
                   if rec.get("verification_status") not in
                   ("confirmed_by_rule",)
                   and rec.get("prompt_template_version")
                   and rec["prompt_template_version"] != PASS_2F_PROMPT_VERSION]
        if drifted:
            raise SystemExit(
                f"2f prompt_template_version drifted from "
                f"{PASS_2F_PROMPT_VERSION} on {drifted} — the issue-confirmed "
                "revival gate would silently disable; investigate before "
                "spending more.")
        final = compute_renovation_estimate_v4(
            artifact_issues_flat(artifact), catalog,
            artifact.get("photos") or {},
            property_metadata=artifact.get("property_metadata") or None,
            package_verifications=verifications,
        )
        _write_json(out_dir / "pass_2f_trace.json", trace)
        _write_json(out_dir / "verifications.json", {
            "cell": cell, "rep": rep, "property_key": prop,
            "model": p2f_cfg["model"], "verifications": verifications,
            "verified_at": _utcnow(),
        })
        _write_json(out_dir / "final_estimate.json", {
            "cell": cell, "rep": rep, "property_key": prop,
            "packages": [package_projection(p)
                         for p in (final.get("packages") or [])],
            "line_items": v4_line_items(final),
            "final_rehab": final.get("final_rehab") or {},
            "pass_2f_trace": final.get("pass_2f_trace") or {},
        })
        calls_made += int(trace.get("attempted_count") or 0)
        done += 1
    print(f"package 2f: {done} cells verified ({calls_made} VLM calls), "
          f"{skipped} resumed -> {PASS2F_DIR}")
    return PASS2F_DIR


def _package_dry_run(config: Dict[str, Any], manifest: Dict[str, Any],
                     round_label: str,
                     artifacts: Dict[Tuple[str, int, str], Path],
                     catalog: Dict[str, Any],
                     variant_a: str, variant_b: str) -> Path:
    """Extras-scale report: what the closed world would look like, before any
    gold exists and before any 2f spend. Gold-authoring input for Steven."""
    cells = cells_for_round(variant_a, variant_b)
    repeats = int(config["repeats"])
    per_cell: Dict[str, Any] = {}
    for cell in cells:
        per_prop: Dict[str, Any] = {}
        for prop in sorted(manifest["properties"]):
            reps_data = []
            candidate_sets: List[Set[str]] = []
            for rep in range(1, repeats + 1):
                artifact = _load_json(artifacts[(cell, rep, prop)])
                probe, confirmed = two_phase_v4(artifact, catalog)
                candidates = [p for p in (probe.get("package_candidates") or [])
                              if isinstance(p, dict) and _candidate_in_scope(p)]
                standalone = [li for li in v4_line_items(confirmed)
                              if not li.get("package_id")]
                reps_data.append({
                    "rep": rep,
                    "candidates": sorted(
                        f"{p.get('package_type')}__{p.get('estimate_unit_id')}"
                        f" [{p.get('pricing_profile')}]" for p in candidates),
                    "candidate_count": len(candidates),
                    "standalone_items": sorted({
                        f"{li['catalog_item_id']}@{li['billable_estimate_unit_id']}"
                        for li in standalone}),
                    "standalone_cost_low": sum(li["cost_low"] for li in standalone),
                    "standalone_cost_high": sum(li["cost_high"] for li in standalone),
                    "final_midpoint": int((confirmed.get("final_rehab") or {})
                                          .get("midpoint") or 0),
                })
                candidate_sets.append({
                    f"{p.get('package_type')}__{p.get('estimate_unit_id')}"
                    for p in candidates})
            pairs = [(a, b) for i, a in enumerate(candidate_sets)
                     for b in candidate_sets[i + 1:]]
            per_prop[prop] = {
                "reps": reps_data,
                "candidate_stability_jaccard": round(statistics.mean(
                    [jaccard(a, b) for a, b in pairs]), 3) if pairs else None,
                "distinct_candidates": sorted(set().union(*candidate_sets))
                if candidate_sets else [],
                "distinct_standalone_items": sorted({
                    item for r in reps_data for item in r["standalone_items"]}),
            }
        per_cell[cell] = per_prop

    capped, allret = f"{variant_b}_cap25", f"{variant_b}_all_retained"
    cap_diff: Dict[str, Any] = {}
    for prop in sorted(manifest["properties"]):
        c = per_cell[capped][prop]
        a = per_cell[allret][prop]
        cap_only = sorted(set(c["distinct_candidates"]) - set(a["distinct_candidates"]))
        allret_only = sorted(set(a["distinct_candidates"]) - set(c["distinct_candidates"]))
        item_delta = sorted(set(a["distinct_standalone_items"])
                            - set(c["distinct_standalone_items"]))
        cap_diff[prop] = {
            "candidates_only_in_cap25": cap_only,
            "candidates_only_in_all_retained": allret_only,
            "standalone_items_only_in_all_retained": item_delta,
            "standalone_items_only_in_all_retained_count": len(item_delta),
        }

    summary = {
        "round": round_label, "generated_at": _utcnow(), "dry_run": True,
        "cells": per_cell, "cap_diff": cap_diff,
        "estimated_2f_vlm_calls": sum(
            r["candidate_count"] for cell in per_cell.values()
            for prop in cell.values() for r in prop["reps"]),
    }
    out = EVAL_DIR / "dry_run_report.json"
    _write_json(out, summary)
    _write_dry_run_md(EVAL_DIR / "dry_run_report.md", summary)
    print(f"dry run -> {out} (no model calls made)")
    return out


def _write_dry_run_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = ["# Package benchmark — no-2f dry run (extras scale)", "",
             f"Generated {summary['generated_at']}. Dollar figures are pre-2f "
             "force-confirmed totals, not production headlines.", ""]
    for cell, per_prop in summary["cells"].items():
        lines.append(f"## Cell {cell}")
        for prop, row in per_prop.items():
            counts = [r["candidate_count"] for r in row["reps"]]
            lines.append(f"- {prop}: candidates/rep {counts}, stability "
                         f"Jaccard {row['candidate_stability_jaccard']}, "
                         f"{len(row['distinct_standalone_items'])} distinct "
                         "standalone priced items")
        lines.append("")
    lines.append("## Cap experiment surface (checklist)")
    for prop, row in summary["cap_diff"].items():
        lines.append(f"- {prop}: +{len(row['candidates_only_in_all_retained'])} "
                     "candidate packages and "
                     f"+{row['standalone_items_only_in_all_retained_count']} "
                     "standalone priced items appear only when the cap lifts")
        for name in row["candidates_only_in_all_retained"]:
            lines.append(f"  - pkg {name}")
    lines.append("")
    lines.append(f"Estimated 2f VLM calls for the full run: "
                 f"~{summary['estimated_2f_vlm_calls']} (rule-confirmed "
                 "turnover packages will subtract from this).")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def load_package_decisions() -> Dict[str, Any]:
    path = REVIEW_DIR / "decisions.json"
    return (_load_json(path).get("decisions") or {}) if path.is_file() else {}


def _match_expected(expected: Dict[str, Any], approved: List[Dict[str, Any]],
                    adjacency: Dict[str, Set[str]]) -> Tuple[str, Optional[str]]:
    """(status, observed_profile): matched_exact | matched_adjacent |
    wrong_tier | missing."""
    target = expected["target_pricing_profile"]
    for pkg in approved:
        if pkg.get("package_type") == expected["package_type"] and \
                pkg.get("estimate_unit_id") == expected["estimate_unit_id"]:
            profile = pkg.get("pricing_profile")
            if profile == target:
                return "matched_exact", profile
            if expected.get("adjacent_acceptable", True) and \
                    profile in adjacency.get(target, set()):
                return "matched_adjacent", profile
            return "wrong_tier", profile
    return "missing", None


def score_package_round(config: Dict[str, Any], manifest: Dict[str, Any],
                        round_label: str, gold: Dict[str, Any],
                        decisions: Dict[str, Any]) -> Dict[str, Any]:
    from tools.comparison_common import sha256_file
    variant_a, variant_b = package_round_variants(config, round_label)
    cells = cells_for_round(variant_a, variant_b)
    repeats = int(config["repeats"])
    gates = package_eval_config(config).get("gates") or {}
    pass_reps = int(gates.get("pass_reps_required", 2))
    adjacency = pricing_profile_adjacency()
    gold_sha = (sha256_file(PACKAGE_GOLD_PATH)
                if PACKAGE_GOLD_PATH.is_file() else None)

    outcomes: Dict[str, Any] = {}
    review_rows: Dict[str, Dict[str, Any]] = {}
    # Distinct row_ids, not occurrences — one signature spanning three cells is
    # still one review decision.
    pending: Dict[str, Set[str]] = {"extras_pending": set(), "gold_gap": set(),
                                    "needs_rereview": set()}
    missing_for_gold: Set[str] = set()

    for cell in cells:
        per_prop: Dict[str, Any] = {}
        for prop in sorted(manifest["properties"]):
            entry = (gold.get("properties") or {}).get(prop) or {}
            expected = entry.get("expected_packages") or []
            required = entry.get("required_work_items") or []
            gold_ids = package_gold_ids(gold, prop)
            reps: Dict[str, Any] = {}
            complete_reps = 0
            for rep in range(1, repeats + 1):
                cell_dir = _cell_dir(cell, rep, prop)
                final_path = cell_dir / "final_estimate.json"
                if not final_path.is_file():
                    reps[str(rep)] = {"status": "not_evaluated"}
                    continue
                final = _load_json(final_path)
                approved = [p for p in final.get("packages") or []
                            if p.get("estimate_eligible") and _scoreable_package(p)]
                items = final.get("line_items") or []
                approved_ids = {p.get("package_id") for p in approved}
                evidence_ids: Set[str] = set()
                for p in approved:
                    evidence_ids.update(p.get("supporting_catalog_item_ids") or [])

                # Extras first: an 'equivalent' decision both clears the extra
                # and credits the mapped gold id during expected matching.
                extras_confirmed: List[str] = []
                extras_open: List[str] = []
                equivalent_credit = {"pkg": set(), "item": set()}

                def _judge_extra(row_id: str, kind: str) -> Optional[str]:
                    verdict = _extra_verdict(
                        row_id, decisions, gold_ids[kind], gold_sha)
                    if verdict == "equivalent":
                        equivalent_credit[kind].add(
                            decisions[row_id]["equivalent_gold_id"])
                        return verdict
                    if verdict == "false_positive":
                        extras_confirmed.append(row_id)
                    elif verdict in ("gold_gap", "needs_rereview"):
                        pending[verdict].add(row_id)
                        extras_open.append(row_id)
                    else:
                        pending["extras_pending"].add(row_id)
                        extras_open.append(row_id)
                    return verdict

                for pkg in approved:
                    sig = f"{pkg.get('package_type')}__{pkg.get('estimate_unit_id')}"
                    if sig in gold_ids["pkg"]:
                        continue
                    row_id = f"{prop}|pkg|{sig}"
                    if _judge_extra(row_id, "pkg") != "equivalent":
                        _collect_review_row(review_rows, row_id, prop,
                                            "package", pkg, None, cell, rep)
                for li in items:
                    if li.get("package_id"):
                        continue
                    if not (li.get("cost_low") or li.get("cost_high")):
                        continue
                    sig = (f"{li.get('catalog_item_id')}"
                           f"@{li.get('billable_estimate_unit_id')}")
                    if sig in gold_ids["item"]:
                        continue
                    if li.get("catalog_item_id") in evidence_ids:
                        continue  # evidence of an approved package, never an extra
                    row_id = f"{prop}|item|{sig}"
                    if _judge_extra(row_id, "item") != "equivalent":
                        _collect_review_row(review_rows, row_id, prop,
                                            "work_item", None, li, cell, rep)

                pkg_status: Dict[str, Any] = {}
                adjacent_used: List[str] = []
                missing: List[str] = []
                for exp in expected:
                    gid = f"{exp['package_type']}__{exp['estimate_unit_id']}"
                    status, observed = _match_expected(exp, approved, adjacency)
                    if status == "missing" and gid in equivalent_credit["pkg"]:
                        status = "matched_equivalent"
                    pkg_status[gid] = {"status": status,
                                       "observed_profile": observed}
                    if status == "matched_adjacent":
                        adjacent_used.append(gid)
                    elif status in ("wrong_tier", "missing"):
                        missing.append(gid)
                work_status: Dict[str, str] = {}
                for work in required:
                    gid = f"{work['catalog_item_id']}@{work['estimate_unit_id']}"
                    # Satisfied as priced standalone work, as a surviving
                    # package-member line item, or as absorbed evidence of an
                    # approved package on the same unit — 2f's photo-capped
                    # review prunes member line items it did not explicitly
                    # confirm, but the approved package still covers that work.
                    satisfied = gid in equivalent_credit["item"] or any(
                        li.get("catalog_item_id") == work["catalog_item_id"]
                        and li.get("billable_estimate_unit_id") == work["estimate_unit_id"]
                        and (not li.get("package_id")
                             or li.get("package_id") in approved_ids)
                        for li in items) or any(
                        pkg.get("estimate_unit_id") == work["estimate_unit_id"]
                        and work["catalog_item_id"]
                        in (pkg.get("supporting_catalog_item_ids") or [])
                        for pkg in approved)
                    work_status[gid] = "satisfied" if satisfied else "missing"
                    if not satisfied:
                        missing.append(gid)
                matched_signatures = {
                    f"{p.get('package_type')}__{p.get('estimate_unit_id')}"
                    for p in approved
                    if f"{p.get('package_type')}__{p.get('estimate_unit_id')}"
                    in gold_ids["pkg"]}

                complete = (not missing and not extras_confirmed
                            and not extras_open)
                if complete:
                    complete_reps += 1
                missing_for_gold.update(f"{prop}: {m}" for m in missing)
                reps[str(rep)] = {
                    "status": "complete" if complete else "incomplete",
                    "expected_packages": pkg_status,
                    "required_work": work_status,
                    "missing": sorted(missing),
                    "adjacent_used": sorted(adjacent_used),
                    "extras_confirmed": sorted(set(extras_confirmed)),
                    "extras_open": sorted(set(extras_open)),
                    "final_midpoint": int((final.get("final_rehab") or {})
                                          .get("midpoint") or 0),
                    "matched_gold_packages": sorted(matched_signatures),
                }
            evaluated = [r for r in reps.values()
                         if r.get("status") != "not_evaluated"]
            per_prop[prop] = {
                "reps": reps,
                "passing_reps": complete_reps,
                "passed": (complete_reps >= pass_reps) if evaluated else None,
            }
        outcomes[cell] = per_prop

    pending_counts = {k: len(v) for k, v in pending.items()}
    blocked = sum(pending_counts.values()) > 0
    if blocked:
        # Verdicts are withheld until every extra is adjudicated — a pending
        # row could flip pass/fail either way.
        for per_prop in outcomes.values():
            for data in per_prop.values():
                data["passed"] = None
    cap_experiment = _cap_experiment(config, manifest, outcomes, variant_b,
                                     gates, blocked)
    status = ("blocked" if blocked else "final") if any(
        r.get("status") != "not_evaluated"
        for cell in outcomes.values() for prop in cell.values()
        for r in prop["reps"].values()) else "no_2f_results"
    return {
        "round": round_label,
        "status": status,
        "pending_review": pending_counts,
        "outcomes": outcomes,
        "cap_experiment": cap_experiment,
        "candidate_diagnostics": _candidate_diagnostics(
            config, manifest, round_label, gold, adjacency,
            variant_a, variant_b),
        "missing_for_gold_reconsideration": sorted(missing_for_gold),
        "cost_deviation": _cost_deviation(config, manifest, gold, outcomes),
        "_review_rows": review_rows,
        "gold_sha256": gold_sha,
        "decisions_count": len(decisions),
    }


def _extra_verdict(row_id: str, decisions: Dict[str, Any],
                   valid_gold_ids: Set[str],
                   gold_sha: Optional[str]) -> Optional[str]:
    decision = decisions.get(row_id)
    if not decision:
        return None
    kind = decision.get("decision")
    if kind == "equivalent":
        target = decision.get("equivalent_gold_id")
        return "equivalent" if target in valid_gold_ids else "needs_rereview"
    if kind == "gold_gap":
        return "gold_gap" if decision.get("gold_sha256") == gold_sha \
            else "needs_rereview"
    return kind  # false_positive survives gold edits — it judges the property


def _collect_review_row(rows: Dict[str, Dict[str, Any]], row_id: str,
                        prop: str, outcome_kind: str,
                        pkg: Optional[Dict[str, Any]],
                        item: Optional[Dict[str, Any]],
                        cell: str, rep: int) -> None:
    row = rows.setdefault(row_id, {
        "row_id": row_id, "property": prop, "outcome_kind": outcome_kind,
        "package_type": (pkg or {}).get("package_type", ""),
        "estimate_unit_id": ((pkg or {}).get("estimate_unit_id")
                             or (item or {}).get("billable_estimate_unit_id", "")),
        "catalog_item_id": (item or {}).get("catalog_item_id", ""),
        "item_name": (item or {}).get("name", ""),
        "profiles": set(), "cost_low": 0, "cost_high": 0,
        "occurs": {}, "evidence_photos": set(),
        "evidence_summary": (pkg or {}).get("evidence_summary", ""),
    })
    if pkg:
        row["profiles"].add(pkg.get("pricing_profile") or "")
        row["cost_low"] = max(row["cost_low"], int(pkg.get("cost_low") or 0))
        row["cost_high"] = max(row["cost_high"], int(pkg.get("cost_high") or 0))
        row["evidence_photos"].update(pkg.get("review_photo_keys") or [])
    if item:
        row["cost_low"] = max(row["cost_low"], int(item.get("cost_low") or 0))
        row["cost_high"] = max(row["cost_high"], int(item.get("cost_high") or 0))
    row["occurs"].setdefault(cell, set()).add(rep)


def _cap_experiment(config: Dict[str, Any], manifest: Dict[str, Any],
                    outcomes: Dict[str, Any], variant_b: str,
                    gates: Dict[str, Any], blocked: bool) -> Dict[str, Any]:
    capped, allret = f"{variant_b}_cap25", f"{variant_b}_all_retained"
    repeats = int(config["repeats"])
    out: Dict[str, Any] = {}
    for prop in sorted(manifest["properties"]):
        c = (outcomes.get(capped) or {}).get(prop) or {}
        a = (outcomes.get(allret) or {}).get(prop) or {}
        verdict = None
        if not blocked and c.get("passed") is not None and a.get("passed") is not None:
            verdict = {
                (True, True): "cap_benign",
                (False, True): "cap_hurts",
                (True, False): "surplus_hurts",
                (False, False): "both_wrong",
            }[(bool(c["passed"]), bool(a["passed"]))]

        flags: List[str] = []
        instability = _pass_2f_instability(prop, capped, allret, repeats)
        c_mids = [r.get("final_midpoint") or 0 for r in (c.get("reps") or {}).values()
                  if r.get("status") != "not_evaluated"]
        a_mids = [r.get("final_midpoint") or 0 for r in (a.get("reps") or {}).values()
                  if r.get("status") != "not_evaluated"]
        if verdict == "cap_benign" and c_mids and a_mids:
            c_med, a_med = statistics.median(c_mids), statistics.median(a_mids)
            delta = abs(a_med - c_med)
            pct = 100.0 * delta / c_med if c_med else 0.0
            if pct > float(gates.get("cost_flag_pct", 20)) or \
                    delta > float(gates.get("cost_flag_abs_usd", 10000)):
                flags.append(f"both modes pass but median midpoints differ "
                             f"${c_med:,.0f} vs ${a_med:,.0f}")
            c_adj = {g for r in (c.get("reps") or {}).values()
                     for g in r.get("adjacent_used") or []}
            a_adj = {g for r in (a.get("reps") or {}).values()
                     for g in r.get("adjacent_used") or []}
            if c_adj != a_adj:
                flags.append(f"both modes pass but via different adjacent "
                             f"tiers: cap25 {sorted(c_adj)} vs all_retained "
                             f"{sorted(a_adj)}")
        confidence = "normal"
        if verdict and verdict != "cap_benign" and instability["count"]:
            confidence = "reduced_by_2f_noise"
        out[prop] = {"verdict": verdict, "flags": flags,
                     "pass_2f_instability": instability,
                     "verdict_confidence": confidence}
    return out


def _pass_2f_instability(prop: str, capped: str, allret: str,
                         repeats: int) -> Dict[str, Any]:
    """When both cap modes produced the SAME candidate multiset for a repeat and
    2f still decided differently, the disagreement is 2f noise, not a cap
    effect. Rule-confirmed turnover packages are deterministic and never flagged."""
    count = 0
    package_ids: Set[str] = set()
    for rep in range(1, repeats + 1):
        c_path = _cell_dir(capped, rep, prop)
        a_path = _cell_dir(allret, rep, prop)
        if not ((c_path / "candidates.json").is_file()
                and (a_path / "candidates.json").is_file()
                and (c_path / "verifications.json").is_file()
                and (a_path / "verifications.json").is_file()):
            continue

        def _sig(path: Path) -> List[Tuple[str, str, str]]:
            return sorted(
                (p.get("package_type") or "", p.get("estimate_unit_id") or "",
                 p.get("pricing_profile") or "")
                for p in _load_json(path / "candidates.json")["candidates"]
                if _candidate_in_scope(p))

        if _sig(c_path) != _sig(a_path):
            continue
        c_ver = _load_json(c_path / "verifications.json")["verifications"]
        a_ver = _load_json(a_path / "verifications.json")["verifications"]
        for pid in set(c_ver) & set(a_ver):
            c_status = c_ver[pid].get("verification_status")
            a_status = a_ver[pid].get("verification_status")
            if "confirmed_by_rule" in (c_status, a_status):
                continue
            if c_status != a_status:
                count += 1
                package_ids.add(pid)
    return {"count": count, "package_ids": sorted(package_ids)}


def _candidate_diagnostics(config: Dict[str, Any], manifest: Dict[str, Any],
                           round_label: str, gold: Dict[str, Any],
                           adjacency: Dict[str, Set[str]],
                           variant_a: str, variant_b: str) -> Dict[str, Any]:
    repeats = int(config["repeats"])
    out: Dict[str, Any] = {}
    for cell in cells_for_round(variant_a, variant_b):
        per_prop: Dict[str, Any] = {}
        for prop in sorted(manifest["properties"]):
            expected = ((gold.get("properties") or {}).get(prop) or {}) \
                .get("expected_packages") or []
            tiers = {"exact": 0, "adjacent": 0, "wrong": 0, "missing": 0}
            recalls: List[float] = []
            extra_counts: List[int] = []
            id_sets: List[Set[str]] = []
            for rep in range(1, repeats + 1):
                path = _cell_dir(cell, rep, prop) / "candidates.json"
                if not path.is_file():
                    continue
                candidates = [p for p in _load_json(path)["candidates"]
                              if _candidate_in_scope(p)]
                sigs = {f"{p.get('package_type')}__{p.get('estimate_unit_id')}"
                        for p in candidates}
                id_sets.append(sigs)
                gold_sigs = {f"{e['package_type']}__{e['estimate_unit_id']}"
                             for e in expected}
                if gold_sigs:
                    recalls.append(len(sigs & gold_sigs) / len(gold_sigs))
                extra_counts.append(len(sigs - gold_sigs))
                for exp in expected:
                    status, _ = _match_expected(exp, candidates, adjacency)
                    key = {"matched_exact": "exact",
                           "matched_adjacent": "adjacent",
                           "wrong_tier": "wrong", "missing": "missing"}[status]
                    tiers[key] += 1
            pairs = [(a, b) for i, a in enumerate(id_sets)
                     for b in id_sets[i + 1:]]
            per_prop[prop] = {
                "expected_recall_mean": round(statistics.mean(recalls), 3)
                if recalls else None,
                "extra_candidates_mean": round(statistics.mean(extra_counts), 1)
                if extra_counts else None,
                "tier": tiers,
                "stability_jaccard": round(statistics.mean(
                    [jaccard(a, b) for a, b in pairs]), 3) if pairs else None,
            }
        out[cell] = per_prop
    return out


def _cost_deviation(config: Dict[str, Any], manifest: Dict[str, Any],
                    gold: Dict[str, Any],
                    outcomes: Dict[str, Any]) -> Dict[str, Any]:
    """Matched-package cost vs the gold's diagnostic range. Report-only."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for cell, per_prop in outcomes.items():
        for prop, data in per_prop.items():
            expected = ((gold.get("properties") or {}).get(prop) or {}) \
                .get("expected_packages") or []
            ranges = {f"{e['package_type']}__{e['estimate_unit_id']}":
                      e.get("diagnostic_cost_range")
                      for e in expected if e.get("diagnostic_cost_range")}
            if not ranges:
                continue
            for rep, row in (data.get("reps") or {}).items():
                for gid, status in (row.get("expected_packages") or {}).items():
                    rng = ranges.get(gid)
                    if not rng or status["status"] in ("missing",):
                        continue
                    out.setdefault(prop, []).append({
                        "cell": cell, "rep": rep, "gold_id": gid,
                        "observed_profile": status.get("observed_profile"),
                        "diagnostic_range": [rng.get("low"), rng.get("high")],
                    })
    return out


# ---------------------------------------------------------------------------
# Review export / import
# ---------------------------------------------------------------------------

def package_review_export(config: Dict[str, Any], manifest: Dict[str, Any],
                          round_label: str,
                          csv_path: Optional[str] = None) -> Path:
    gold = load_package_gold()
    decisions = load_package_decisions()
    scored = score_package_round(config, manifest, round_label, gold, decisions)
    rows = scored["_review_rows"]
    if scored["status"] == "no_2f_results":
        raise SystemExit("no 2f results yet — run package-eval --stage 2f first")
    out = Path(csv_path) if csv_path else REVIEW_DIR / "review.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (
        0 if not decisions.get(r["row_id"]) else 1,
        r["property"], r["outcome_kind"], r["row_id"]))
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=PACKAGE_REVIEW_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            decision = decisions.get(row["row_id"]) or {}
            writer.writerow({
                **row,
                "observed_pricing_profiles": ";".join(sorted(
                    p for p in row["profiles"] if p)),
                "occurs_in": ", ".join(
                    f"{cell}:{';'.join(str(r) for r in sorted(reps))}"
                    for cell, reps in sorted(row["occurs"].items())),
                "evidence_photos": ";".join(sorted(row["evidence_photos"])),
                "human_decision": decision.get("decision", ""),
                "equivalent_gold_id": decision.get("equivalent_gold_id") or "",
                "reviewer_note": decision.get("note", ""),
            })
    undecided = sum(1 for r in ordered if not decisions.get(r["row_id"]))
    print(f"package review export: {len(ordered)} unmatched approved outcomes "
          f"({undecided} undecided) -> {out}")
    return out


def package_review_import(config: Dict[str, Any], manifest: Dict[str, Any],
                          round_label: str, csv_path: str) -> Path:
    from tools.comparison_common import sha256_file
    gold = load_package_gold()
    gold_sha = sha256_file(PACKAGE_GOLD_PATH)
    valid_rows = score_package_round(config, manifest, round_label, gold,
                                     {})["_review_rows"]
    errors: List[str] = []
    merged = load_package_decisions()
    cleared: List[str] = []
    imported = 0
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        for lineno, raw in enumerate(csv.DictReader(handle), start=2):
            row_id = (raw.get("row_id") or "").strip()
            if not row_id:
                continue
            if row_id not in valid_rows:
                errors.append(f"line {lineno}: unknown row_id {row_id!r}")
                continue
            decision = (raw.get("human_decision") or "").strip().lower()
            if not decision:
                cleared.append(row_id)
                continue
            if decision not in REVIEW_DECISIONS:
                errors.append(f"line {lineno}: human_decision {decision!r} "
                              f"not one of {list(REVIEW_DECISIONS)}")
                continue
            equivalent = (raw.get("equivalent_gold_id") or "").strip() or None
            if decision == "equivalent":
                prop = valid_rows[row_id]["property"]
                kind = "pkg" if valid_rows[row_id]["outcome_kind"] == "package" \
                    else "item"
                if equivalent not in package_gold_ids(gold, prop)[kind]:
                    errors.append(
                        f"line {lineno}: equivalent_gold_id {equivalent!r} is "
                        f"not a {kind} gold id on {prop}")
                    continue
            merged[row_id] = {
                "decision": decision,
                "equivalent_gold_id": equivalent,
                "note": (raw.get("reviewer_note") or "").strip(),
                "gold_sha256": gold_sha,
                "imported_at": _utcnow(),
            }
            imported += 1
    if errors:
        raise SystemExit("package review import rejected:\n  "
                         + "\n  ".join(errors))
    for row_id in cleared:
        merged.pop(row_id, None)
    out = REVIEW_DIR / "decisions.json"
    _write_json(out, {"round": round_label, "updated_at": _utcnow(),
                      "decisions": merged})
    print(f"package review import: {imported} decisions, {len(cleared)} "
          f"cleared -> {out}")
    return out


# ---------------------------------------------------------------------------
# Report integration (called from benchmark_pass2a.stage_report / report md)
# ---------------------------------------------------------------------------

def report_contribution(config: Dict[str, Any], manifest: Dict[str, Any],
                        round_label: str) -> Optional[Dict[str, Any]]:
    dry_path = EVAL_DIR / "dry_run_report.json"
    has_2f = PASS2F_DIR.is_dir() and any(PASS2F_DIR.glob("*/rep*/*/verifications.json"))
    if not has_2f and not dry_path.is_file():
        return None
    if not PACKAGE_GOLD_PATH.is_file() or not has_2f:
        summary = _load_json(dry_path) if dry_path.is_file() else {}
        return {
            "status": "dry_run_only" if dry_path.is_file() else "not_started",
            "dry_run": {
                "estimated_2f_vlm_calls": summary.get("estimated_2f_vlm_calls"),
                "cap_diff": summary.get("cap_diff"),
            } if summary else None,
            "gold_present": PACKAGE_GOLD_PATH.is_file(),
        }
    scored = score_package_round(config, manifest, round_label,
                                 load_package_gold(), load_package_decisions())
    scored.pop("_review_rows", None)
    return scored


def package_report_md_lines(pkg: Dict[str, Any]) -> List[str]:
    lines = ["## Package outcome evaluation", ""]
    if pkg.get("status") in ("dry_run_only", "not_started"):
        lines.append("Dry run only — package gold authoring pending. See "
                     "runs/package_eval/dry_run_report.md for the extras-scale "
                     "summary.")
        lines.append("")
        return lines
    if pkg.get("status") == "blocked":
        pend = ", ".join(f"{v} {k}" for k, v in pkg["pending_review"].items() if v)
        lines.append(f"**Not final — {pend}.** Export the package review CSV, "
                     "adjudicate, import, and rerun the report.")
        lines.append("")
    lines.append("| property | cap verdict | confidence | flags |")
    lines.append("| --- | --- | --- | --- |")
    for prop, row in (pkg.get("cap_experiment") or {}).items():
        lines.append(f"| {prop} | {row.get('verdict') or 'pending'} | "
                     f"{row.get('verdict_confidence')} | "
                     f"{'; '.join(row.get('flags') or []) or '—'} |")
    lines.append("")
    lines.append("| cell | property | passing reps | passed |")
    lines.append("| --- | --- | --- | --- |")
    for cell, per_prop in (pkg.get("outcomes") or {}).items():
        for prop, data in per_prop.items():
            lines.append(f"| {cell} | {prop} | {data['passing_reps']}/3 | "
                         f"{data['passed']} |")
    lines.append("")
    noise = {prop: row["pass_2f_instability"]
             for prop, row in (pkg.get("cap_experiment") or {}).items()
             if row["pass_2f_instability"]["count"]}
    if noise:
        lines.append("2f instability on identical candidate sets (noise, not "
                     "cap effects): "
                     + "; ".join(f"{prop}: {n['count']} ({', '.join(n['package_ids'])})"
                                 for prop, n in noise.items()))
        lines.append("")
    missing = pkg.get("missing_for_gold_reconsideration") or []
    if missing:
        lines.append(f"Missing expected outcomes ({len(missing)}, fail "
                     "directly — listed for gold reconsideration):")
        for m in missing:
            lines.append(f"- {m}")
        lines.append("")
    return lines
