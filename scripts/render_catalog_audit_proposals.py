"""Session 2 proposal artifact for the catalog audit: derive, validate, render.

Reads Session 1's hash-verified evidence bundle plus the hand-authored judgment
half of reports/catalog_audit_proposals.json, fills the tool-owned `derived`
half (semantics per era, lineage records, evidence-bar counts from adjudicated
units, in-memory decisions dry runs), validates the artifact against the
program's rules, and renders the deterministic proposal document.

  default    derive + validate, write the JSON and the Markdown
  --check    same in memory; compare with what is on disk; write nothing
  --packet   write the photo-review packet for the separate review session
             (refused once the pinned review results exist on disk)

The real decisions file and generated catalog are read and never written; every
mode re-hashes them, the evidence bundle, the packet, and the review results
afterwards and fails if they moved. A missing or drifted pinned input fails
closed before anything is derived; in final mode the review results and the
packet must be the pinned pair (REVIEW_SHA256 / PACKET_SHA256).

Run:
  .venv\\Scripts\\python.exe scripts\\render_catalog_audit_proposals.py [--check | --packet]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_catalog_audit_evidence import claim_text, embed_source_text  # noqa: E402
from scripts.error_attribution_report import latest_verdicts  # noqa: E402
from tools import review_cards as rc  # noqa: E402
from tools.catalog_validation import (VALID_CHANGE_TYPES, validate_issue_catalog,  # noqa: E402
                                      validate_migration_manifest)
from tools.comparison_common import atomic_json, canonical_json, sha256_canonical, sha256_file  # noqa: E402

SCHEMA_VERSION = 1
EVIDENCE_JSON = ROOT / "reports" / "catalog_audit_evidence.json"
EVIDENCE_SHA256 = "43354104cc109c768efbe2d2f8b9a825f33e0882210b81e9f12b847ca67e29b7"
EVIDENCE_FINGERPRINT = "e4c0a2b816395782cd84bc4abcdf6207d131a34a0e2d63eaebb39bec4c190c08"
PROPOSALS_JSON = ROOT / "reports" / "catalog_audit_proposals.json"
PACKET_JSON = ROOT / "reports" / "catalog_audit_photo_review_packet.json"
REVIEW_JSON = ROOT / "reports" / "catalog_audit_photo_review.json"
LEDGER = ROOT / "reports" / "error_attribution_verdicts.jsonl"
CURRENT_SURFACE_JSON = ROOT / "reports" / "catalog_authoring_surface_v2.json"
CURRENT_SURFACE_VERSION = "catalog-authoring-surface-v2"
GENERATOR = ROOT / "scripts" / "migrate_catalog_kind_v2.py"
V1_CATALOG = ROOT / "tools" / "issue_catalog.json"
DECISIONS = ROOT / "tools" / "catalog_migrations" / "kind_v2_decisions.json"
V2_CATALOG = ROOT / "tools" / "issue_catalog_kind_v2.json"
GUARDED = (DECISIONS, V2_CATALOG)
PINNED_INPUTS = (EVIDENCE_JSON, PACKET_JSON, REVIEW_JSON)
REVIEW_SHA256 = "2225db9caae573b52b4270f846c314749d39aff468fab4e1a8e2012c49919150"
PACKET_SHA256 = "7b4621dfe6c72aa42369c2f5ef7051dfecbec05b6333a80651b1b57a5b37bc80"

OUTCOMES = ("native_decisions_proposal", "non_catalog_action", "deferred_insufficient_evidence",
            "migration_system_gap", "no_change")
DEFERRED = "deferred_insufficient_evidence"
ADJUDICATIONS = ("supports_claim", "refuted", "unclear", "unavailable")
CAUSES = ("catalog_wording_or_specificity", "missing_catalog_coverage", "overlapping_or_ambiguous_ontology",
          "retrieval_metadata_or_reachability", "pass_2d_selection", "terra_verification",
          "upstream_kind_or_pass_2c", "downstream_tier_projection_routing_package_costing",
          "product_policy_or_quarantine", "insufficient_evidence_or_no_change", "upstream_pass_2a_observation")
TRIAGE = ("covered_by_existing_item", "attaches_to_cluster", "not_billable_observation", "product_quarantined",
          "coverage_gap_candidate", "unclear")
BAR = ("two_independent_human", "one_human_plus_independent_corroboration", "structural_exception", "none")
STAGE_QUESTIONS = ("existing_item_exists", "upstream_kind_wrong", "scene_or_retrieval_metadata_cause",
                   "retrieved_but_2d_selected_other", "terra_adjudicated_wrong_detail",
                   "wording_creates_unsupported_commitments", "steals_neighbors_or_damages_successes",
                   "alters_eligibility_package_economics")
UNIT_ROLES = ("support", "counterexample", "context")
HUMAN_METHODS = ("human_review", "gold_reference")
ENTRY_META = ("deprecated", "requires_re_resolution", "atomicity_rationale", "expected_effects", "task3_flags")
IDENTITY_FIELDS = ("id", "kind", "severity", "name", "description", "embed_text", "support_any", "atomic_claim")
SEMANTIC_SNAPSHOT = ("kind", "name", "severity", "atomic_claim", "description", "embed_text", "support_any",
                     "deny_any", "require_any", "scene_groups", "trade_bucket", "scope", "tier", "route_override",
                     "category", "display_class", "defaultHidden", "drop_if_generic", "pricing_status")
CONTROL_ROLES = {"positive_uses": "positive_use", "agreements": "agreement",
                 "correct_rejections": "correct_rejection", "hallucination_annotations": "hallucination"}
PROPOSAL_ID = "CAP-"
KNOWN_IDS = tuple(f"CAP-{n:03d}" for n in range(1, 18)) + ("CAP-018", "CAP-019", "CAP-021")
# CAP-018/019 split out of CAP-002, CAP-021 out of CAP-006 in Phase C. CAP-020 was reserved for a window-
# treatment split that the evidence did not support, so it stays unallocated: ids are never reused.
REFUTATION_BACKED = ("no_change", "non_catalog_action", "migration_system_gap")
UNIT_ENTRY_KEYS = ("unit_key", "role", "review_row_id", "transfer_note")
RESHAPE_RELATIONS = ("equivalent", "narrower")
HUMAN_INPUT_KEYS = ("ref", "source", "ruling", "effect")
MISMATCH_CLASSES = ("exact_match", "operationally_equivalent", "operationally_consequential", "not_applicable")
EQUIVALENCE_DIMENSIONS = ("subject", "mechanism", "severity", "repair", "scope", "billability", "route", "safety")
DEFERRAL_REASONS = ("insufficient_evidence", "lead_only", "out_of_scope_pass_2a", "pending_promotion")
FOLLOW_UP_KEYS = ("case_ref", "selected_item", "better_item", "better_rank", "confusion_family", "note")
CONTROL_EXPECTATIONS = ("still_rejected", "still_supported", "flips_to_rejected", "flips_to_supported")
REGRESSION_ROLES = ("hallucination", "correct_rejection")
STRUCTURAL_FIELDS = SEMANTIC_SNAPSHOT
COVERAGE_PREFIX = "coverage"


class ProposalError(SystemExit):
    pass


def fail(msg: str) -> None:
    raise ProposalError(f"render_catalog_audit_proposals: {msg}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------- inputs

def load_bundle(path: Path = EVIDENCE_JSON) -> Dict[str, Any]:
    if not path.is_file():
        fail(f"missing evidence bundle {path}")
    actual = sha256_file(path)
    if actual != EVIDENCE_SHA256:
        fail(f"evidence bundle sha256 {actual} != pinned {EVIDENCE_SHA256}")
    bundle = _read_json(path)
    if bundle.get("fingerprint") != EVIDENCE_FINGERPRINT:
        fail(f"evidence fingerprint {bundle.get('fingerprint')} != pinned {EVIDENCE_FINGERPRINT}")
    return bundle


def load_generator():
    spec = importlib.util.spec_from_file_location("migrate_catalog_kind_v2", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def source_hashes(bundle: Dict[str, Any]) -> Dict[str, str]:
    return {s["path"]: s["sha256"] for s in bundle["sources"]}


def load_baseline(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """v1 catalog, decisions, and generated catalog exactly as the bundle pinned them,
    plus an in-memory regeneration that must reproduce the on-disk catalog."""
    pinned = source_hashes(bundle)
    problems = []
    for path in (V1_CATALOG, DECISIONS, V2_CATALOG, GENERATOR):
        expected = pinned.get(_rel(path))
        actual = sha256_file(path) if path.is_file() else None
        if expected is None or actual != expected:
            problems.append(f"{_rel(path)}: on disk {actual}, bundle {expected}")
    if problems:
        fail("baseline inputs drifted from the evidence bundle: " + "; ".join(problems))
    gen = load_generator()
    v1, decisions, on_disk = _read_json(V1_CATALOG), _read_json(DECISIONS), _read_json(V2_CATALOG)
    catalog, manifest = gen.generate(copy.deepcopy(v1), copy.deepcopy(decisions))
    if catalog != on_disk:
        fail("in-memory generate() does not reproduce tools/issue_catalog_kind_v2.json")
    return {"gen": gen, "v1": v1, "decisions": decisions, "catalog": catalog, "manifest": manifest,
            "items": {it["id"]: it for it in catalog["items"]}, "order": [it["id"] for it in catalog["items"]],
            "hashes": {_rel(p): pinned[_rel(p)] for p in (V1_CATALOG, DECISIONS, V2_CATALOG, GENERATOR)}}


def guard_snapshot(paths: Sequence[Path] = GUARDED + PINNED_INPUTS) -> Dict[str, Optional[str]]:
    """Hashes of the files no mode may write (None while a pinned input is absent)."""
    return {_rel(p): (sha256_file(p) if Path(p).is_file() else None) for p in paths}


def verify_pins(review_path: Path = REVIEW_JSON, packet_path: Path = PACKET_JSON) -> Dict[str, Optional[str]]:
    """Final mode only: the on-disk review results and packet must be the pinned pair."""
    actual = {"review": sha256_file(review_path) if Path(review_path).is_file() else None,
              "packet": sha256_file(packet_path) if Path(packet_path).is_file() else None}
    problems = []
    if actual["review"] != REVIEW_SHA256:
        problems.append(f"review results sha256 {actual['review']} != pinned {REVIEW_SHA256}")
    if actual["packet"] != PACKET_SHA256:
        problems.append(f"review packet sha256 {actual['packet']} != pinned {PACKET_SHA256}")
    if problems:
        fail("pinned review inputs drifted: " + "; ".join(problems))
    return actual


# --------------------------------------------------------------------------- bundle views

class Evidence:
    """Read-only indexes over the bundle that the derivations share."""

    def __init__(self, bundle: Dict[str, Any], ledger: Optional[Dict[str, Dict[str, Any]]] = None):
        self.bundle = bundle
        self.units = {u["unit_key"]: u for u in bundle["evidence_units"]}
        self.cases = bundle["indexes"]["cases"]
        self.by_condition = bundle["indexes"]["by_condition"]
        self.by_item = bundle["indexes"].get("by_item") or {}
        self.gold_cases = bundle["gold"]["cases"]
        self.gold_rows = {r["gold_unit_key"]: r for r in bundle["gold"]["matching_rows"] if r.get("gold_unit_key")}
        self.leads = {l["record_id"]: l for l in bundle["factorized_leads"]}
        self.enrich = bundle["candidate_enrichment"]["cases_and_leads"]
        self.gold_enrich = bundle["candidate_enrichment"]["gold_photos"]
        self.labels = {l["card_id"]: l for l in bundle["labels"]}
        self.notes = {n["card_id"]: n for n in bundle["review_notes"]}
        self.worklist = {w["item_id"]: w for w in bundle["worklist"]}
        self.families = bundle["families"]
        self.items = bundle["item_semantics"]["items"]
        self.era_changes = {}
        for ch in bundle["baseline_comparison"]["changed_items"]:
            self.era_changes[ch.get("item_id") or ch.get("id")] = ch["changes"]
        self.surface = bundle["authoring_surface"]
        self.migration = bundle["migration"]
        self.quarantined = set(bundle["lanes"]["product_quarantined_trades"])
        self.coverage_units = sorted({r["unit_key"] for r in bundle["seeds"]["records"]
                                      if r["lane"] == "coverage_question" and r.get("unit_key")})
        self.artifacts = {(a["source"], a["property_key"], a["run_id"]): a for a in bundle["run_artifacts"]}
        self.ledger = ledger or {}

    def item_semantics(self, item_id: str) -> Dict[str, Any]:
        """Proposal-baseline fields plus the evidence-era value of every field that differs."""
        cur = self.items.get(item_id)
        if cur is None:
            return {"present": False}
        snap = {f: cur.get(f) for f in SEMANTIC_SNAPSHOT}
        snap["claim_text"] = claim_text(cur)
        snap["embed_source_text"] = embed_source_text(cur)
        economics = {f: cur.get(f) for f in self.surface["economic_fields"]}
        era = {f: ch["evidence_era"] for f, ch in (self.era_changes.get(item_id) or {}).items()}
        return {"present": True, "proposal_baseline": snap, "economics": economics,
                "evidence_era_differences": era, "trade_quarantined": cur.get("trade_bucket") in self.quarantined}

    def case_lineage(self, case_id: str) -> Dict[str, Any]:
        c = self.cases[case_id]
        led = self.ledger.get(case_id) or {}
        issues = []
        for iss in c["issues"]:
            detail = (self.enrich.get(case_id) or {}).get(iss["issue_id"]) or {}
            issues.append({**iss, "candidates": detail.get("candidates"), "routing_reason": detail.get("routing_reason"),
                           "artifact_resolution_agrees": detail.get("agrees_with_queue")})
        return {"case_id": case_id, "lane": c["lane"], "scene_group": c.get("scene_group"),
                "condition_item": c["condition_item"], "catalog_kind": c.get("catalog_kind"),
                "claim_text": c.get("claim_text"), "issues": issues, "terra_verdict": c.get("terra_verdict"),
                "terra_rationale": c.get("terra_rationale"), "accepted": c.get("accepted"),
                "disposition": c.get("disposition"), "reason_code": c.get("reason_code"),
                "human_truth": c.get("human_truth"), "attribution": c.get("attribution"),
                "attribution_rationale": led.get("rationale"), "reviewer_note": (self.notes.get(case_id) or {}).get("note")}

    def lineage(self, unit_key: str, record_ids: Iterable[str]) -> Dict[str, Any]:
        u = self.units[unit_key]
        record_ids = list(record_ids)
        out: Dict[str, Any] = {"unit_key": unit_key, "unit_type": u["unit_type"], "primary_method": u["primary_method"],
                               "rules": u["rules"], "cases": [], "gold": None, "leads": [], "same_property_units": len(u["same_property_units"]),
                               "same_photo_units": len(u["same_photo_units"]), "independent": u["independent"]}
        case_ids = {r for r in record_ids if r in self.cases}
        cond_case = self.by_condition.get(unit_key)
        if cond_case:
            case_ids.add(cond_case)
        out["cases"] = [self.case_lineage(cid) for cid in sorted(case_ids)]
        for rid in sorted(r for r in record_ids if r in self.leads):
            lead = self.leads[rid]
            out["leads"].append({**{k: lead[k] for k in ("record_id", "observed_description", "derived_class",
                                                         "stored_verdict", "catalog_item_id", "item_source")},
                                 "issues": [{"issue_id": k, **v} for k, v in sorted((self.enrich.get(rid) or {}).items())]})
        gold_ids = [r for r in record_ids if r in self.gold_cases]
        if gold_ids:
            g = self.gold_cases[gold_ids[0]]
            led = self.ledger.get(gold_ids[0]) or {}
            out["gold"] = {k: g.get(k) for k in ("case_id", "finding", "matching_decision", "matching_note",
                                                  "covering_item", "covering_item_source", "covering_case_id",
                                                  "covering_rejected_condition_id", "attribution", "run_id")}
            out["gold"]["attribution_rationale"] = led.get("rationale")
            if g.get("covering_case_id") and g["covering_case_id"] in self.cases:
                out["gold"]["covering_case"] = self.case_lineage(g["covering_case_id"])
        elif u["unit_type"] == "gold":
            row = self.gold_rows.get(unit_key) or {}
            out["gold"] = {"finding": row.get("finding"), "matching_decision": row.get("decision"),
                           "matching_note": row.get("note"), "record_type": "gold_row"}
        if u["unit_type"] == "gold":
            key = f"{u['property_key']}/{u['photo_key']}"
            out["gold_photo_resolutions"] = [{"issue_id": k, **v} for k, v in sorted((self.gold_enrich.get(key) or {}).items())]
        return out


# --------------------------------------------------------------------------- photos

def artifact_photo_keys(ev: Evidence, unit: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
    """Fallback for lead-only units: the pinned run artifact's evidence refs."""
    art = ev.artifacts.get((unit["source"], unit["property_key"], unit["run_id"]))
    if not art or not art.get("artifact_path"):
        return [], "run artifact not pinned by the queue"
    path = Path(art["artifact_path"])
    if not path.is_file():
        return [], "run artifact file missing"
    if sha256_file(path) != art.get("current_sha256"):
        return [], "run artifact hash drifted from the bundle"
    res = rc.v5_result(_read_json(path))
    if res is None:
        return [], "artifact has no complete v5 result"
    idx = rc.index_result(res)
    refs = (idx["evs"].get(unit["condition_id"]) or {}).get("evidence_refs") or []
    keys = sorted({r.get("photo_key") for r in refs if isinstance(r, dict) and r.get("photo_key")})
    return keys, (None if keys else "condition has no photo evidence refs in the artifact")


def photo_refs(ev: Evidence, unit_key: str, record_ids: Iterable[str], photo_root: Path,
               cache: Dict[str, Any]) -> Dict[str, Any]:
    u = ev.units[unit_key]
    keys: set = set()
    if u["unit_type"] == "gold":
        keys.add(u["photo_key"])
    else:
        for rid in record_ids:
            if rid in ev.cases:
                keys.update(ev.cases[rid].get("photo_keys") or [])
            elif rid in ev.enrich:
                keys.update(d.get("photo_key") for d in ev.enrich[rid].values() if d.get("photo_key"))
        cond_case = ev.by_condition.get(unit_key)
        if cond_case:
            keys.update(ev.cases[cond_case].get("photo_keys") or [])
    reason = None
    if not keys and u["unit_type"] == "runtime":
        found, reason = artifact_photo_keys(ev, u)
        keys.update(found)
        if found:
            reason = "resolved from the pinned run artifact's evidence refs"
    photos = [photo_entry(u["property_key"], k, photo_root, cache) for k in sorted(keys)]
    status = "available" if photos and all(p["exists"] for p in photos) else ("partial" if photos else "unavailable")
    return {"status": status, "photos": photos, "note": reason}


def photo_entry(property_key: str, photo_key: str, photo_root: Path, cache: Dict[str, Any]) -> Dict[str, Any]:
    key = f"{property_key}/{photo_key}"
    if key not in cache:
        path = photo_root / property_key / photo_key
        cache[key] = {"key": key, "property_key": property_key, "photo_key": photo_key, "path": str(path),
                      "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None}
    return cache[key]


# --------------------------------------------------------------------------- controls and packet

def select_controls(ev: Evidence, cluster: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reviewed successes, agreements, correct rejections, and hallucination counterexamples
    for the implicated items; a bounded deterministic subset is required review."""
    j = cluster["judgment"]
    seen: Dict[str, Dict[str, Any]] = {}
    for item_id in j.get("implicated_items") or []:
        w = ev.worklist.get(item_id)
        if not w:
            continue
        for field, role in CONTROL_ROLES.items():
            for cid in w.get(field) or []:
                if cid in seen:
                    continue
                if cid in ev.labels:
                    lab = ev.labels[cid]
                    prop, photos, unit_key = lab["property_key"], lab.get("card_photo_keys") or [], lab.get("unit_key")
                    claim, truth = lab.get("card_claim"), {"claim": lab.get("claim"), "work": lab.get("work"), "class_v1_1": lab.get("class_v1_1")}
                else:
                    c = ev.cases.get(cid) or {}
                    prop, photos, unit_key = c.get("property_key"), c.get("photo_keys") or [], c.get("unit_key")
                    claim, truth = c.get("claim_text"), c.get("human_truth")
                seen[cid] = {"card_id": cid, "role": role, "item_id": item_id, "property_key": prop, "photo_keys": photos,
                             "unit_key": unit_key, "item_claim_text": claim, "human_truth": truth,
                             "terra_verdict": (ev.cases.get(cid) or ev.labels.get(cid) or {}).get("terra_verdict"),
                             "reviewer_note": (ev.notes.get(cid) or {}).get("note"), "required": False}
    excluded = set((j.get("controls") or {}).get("excluded") or [])
    extra = set((j.get("controls") or {}).get("extra_required") or [])
    for role_pool, cap in (({"positive_use", "agreement"}, 2), ({"correct_rejection"}, 2)):
        props: set = set()
        for ctl in sorted((c for c in seen.values() if c["role"] in role_pool and c["card_id"] not in excluded),
                          key=lambda c: (c["property_key"] or "", c["card_id"])):
            if len(props) >= cap:
                break
            if ctl["property_key"] in props:
                continue
            props.add(ctl["property_key"])
            ctl["required"] = True
    for ctl in seen.values():
        if ctl["role"] == "hallucination" and ctl["card_id"] not in excluded:
            ctl["required"] = True
        if ctl["card_id"] in extra:
            ctl["required"] = True
    return [seen[k] for k in sorted(seen)]


def build_packet(proposals: Dict[str, Any], ev: Evidence, photo_root: Path) -> Dict[str, Any]:
    cache: Dict[str, Any] = {}
    rows = []
    for cluster in sorted(proposals["clusters"], key=lambda c: c["proposal_id"]):
        j = cluster["judgment"]
        for entry in sorted(j.get("units") or [], key=lambda e: e["unit_key"]):
            u = ev.units[entry["unit_key"]]
            rids = [r["record_id"] for r in u["records"]]
            rows.append({"row_id": f"{cluster['proposal_id']}:{entry['unit_key']}", "row_type": "unit",
                         "priority": "required" if entry["role"] in ("support", "counterexample") else "optional",
                         "cluster_id": cluster["proposal_id"], "cluster_title": cluster["title"],
                         "claim_under_test": (j.get("cluster") or {}).get("claim_under_test"), "unit_key": entry["unit_key"],
                         "unit_role": entry["role"], "evidence_class": u["primary_method"],
                         "implicated_items": u["implicated_items"],
                         "item_claims": {i: (ev.items.get(i) or {}).get("description") for i in u["implicated_items"]},
                         "item_claim_text": {i: claim_text(ev.items[i]) for i in u["implicated_items"] if i in ev.items},
                         "photos": photo_refs(ev, entry["unit_key"], rids, photo_root, cache),
                         "lineage": ev.lineage(entry["unit_key"], rids)})
        for ctl in select_controls(ev, cluster):
            rows.append({"row_id": f"{cluster['proposal_id']}:control:{ctl['card_id']}", "row_type": "control",
                         "priority": "required" if ctl["required"] else "optional",
                         "cluster_id": cluster["proposal_id"], "cluster_title": cluster["title"],
                         "claim_under_test": ctl["item_claim_text"], "control_role": ctl["role"], "card_id": ctl["card_id"],
                         "item_id": ctl["item_id"], "human_truth": ctl["human_truth"], "terra_verdict": ctl["terra_verdict"],
                         "reviewer_note": ctl["reviewer_note"], "unit_key": ctl["unit_key"],
                         "photos": {"status": "available" if ctl["photo_keys"] else "unavailable",
                                    "photos": [photo_entry(ctl["property_key"], k, photo_root, cache) for k in sorted(ctl["photo_keys"])],
                                    "note": None},
                         "lineage": ev.case_lineage(ctl["card_id"]) if ctl["card_id"] in ev.cases else None})
    for unit_key in ev.coverage_units:
        u = ev.units[unit_key]
        rids = [r["record_id"] for r in u["records"]]
        lin = ev.lineage(unit_key, rids)
        rows.append({"row_id": f"coverage:{unit_key}", "row_type": "coverage", "priority": "required",
                     "cluster_id": None, "claim_under_test": (lin.get("gold") or {}).get("finding"), "unit_key": unit_key,
                     "evidence_class": u["primary_method"], "photos": photo_refs(ev, unit_key, rids, photo_root, cache),
                     "lineage": lin})
    rows.sort(key=lambda r: r["row_id"])
    counts = Counter(f"{r['row_type']}:{r['priority']}" for r in rows)
    return {"schema_version": SCHEMA_VERSION, "purpose": "photo and lineage review packet for catalog audit Session 2",
            "generated_from": {"evidence_sha256": EVIDENCE_SHA256, "evidence_fingerprint": EVIDENCE_FINGERPRINT},
            "photo_root": str(photo_root), "adjudication_vocabulary": list(ADJUDICATIONS),
            "answer_template": {"row_id": None, "unit_key": None, "adjudication": None, "reason": None,
                                "what_image_shows": None, "claim_commitments_supported": None,
                                "claim_commitments_not_supported": None, "lineage_note": None},
            "counts": dict(sorted(counts.items())),
            "photos": {k: v for k, v in sorted(cache.items())}, "rows": rows}


def load_review(path: Path, packet_path: Path) -> Dict[str, Any]:
    if not path.is_file():
        fail(f"review results missing: {path}")
    if not packet_path.is_file():
        fail(f"review packet missing: {packet_path}")
    review, packet = _read_json(path), _read_json(packet_path)
    packet_sha = sha256_file(packet_path)
    problems = []
    if review.get("packet_sha256") != packet_sha:
        problems.append(f"results packet_sha256 {review.get('packet_sha256')} != packet on disk {packet_sha}")
    rows = review.get("rows") or []
    ids = Counter(r.get("row_id") for r in rows)
    packet_rows = {r["row_id"]: r for r in packet["rows"]}
    for rid, n in sorted(ids.items()):
        if n > 1:
            problems.append(f"row {rid} answered {n} times")
        if rid not in packet_rows:
            problems.append(f"row {rid} is not in the packet")
    for r in rows:
        if r.get("adjudication") not in ADJUDICATIONS:
            problems.append(f"row {r.get('row_id')} adjudication {r.get('adjudication')!r} not in vocabulary")
        if packet_rows.get(r.get("row_id"), {}).get("unit_key") != r.get("unit_key"):
            problems.append(f"row {r.get('row_id')} unit_key mismatch")
    for rid, prow in sorted(packet_rows.items()):
        if prow["priority"] == "required" and rid not in ids:
            problems.append(f"required row {rid} has no answer")
    if problems:
        fail("review results rejected: " + "; ".join(problems[:20]) + (" ..." if len(problems) > 20 else ""))
    return {"path": _rel(path), "sha256": sha256_file(path), "packet_sha256": packet_sha,
            "reviewer": review.get("reviewer"), "rows": {r["row_id"]: r for r in rows},
            "photos": review.get("photos") or {},
            "packet_rows": {rid: {k: pr.get(k) for k in ("row_type", "priority", "unit_key", "card_id", "cluster_id",
                                                          "item_id", "control_role")} for rid, pr in packet_rows.items()}}


# --------------------------------------------------------------------------- decisions ops and dry run

def _parts(path: str) -> List[str]:
    parts = [p for p in path.strip("/").split("/") if p != ""]
    if not parts:
        fail(f"empty op path {path!r}")
    return parts


def _entry(decisions: Dict[str, Any], legacy_id: str) -> Dict[str, Any]:
    for e in decisions.get("entries") or []:
        if e.get("legacy_id") == legacy_id:
            return e
    fail(f"op names unknown legacy id {legacy_id!r}")
    return {}


def _walk(container: Any, parts: Sequence[str], create: bool) -> Tuple[Any, Any]:
    """Parent container and final key for a pointer; only an `overrides` dict is created on demand."""
    node = container
    for i, p in enumerate(parts[:-1]):
        if isinstance(node, list):
            node = node[int(p)]
        elif isinstance(node, dict):
            if p not in node:
                if create and p == "overrides":
                    node[p] = {}
                else:
                    fail(f"op path component {p!r} absent in {'/'.join(parts)}")
            node = node[p]
        else:
            fail(f"op path {'/'.join(parts)} descends into a scalar")
    key: Any = parts[-1]
    if isinstance(node, list):
        key = int(key)
    return node, key


def fill_before(decisions: Dict[str, Any], ops: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for op in ops:
        entry = _entry(decisions, op["legacy_id"])
        parts = _parts(op["path"])
        before: Any = None
        try:
            node, key = _walk(entry, parts, create=False)
            if isinstance(node, list):
                before = node[key] if 0 <= key < len(node) else None
            else:
                before = node.get(key)
        except ProposalError:
            before = None
        out.append({**op, "before": copy.deepcopy(before)})
    return out


def apply_ops(decisions: Dict[str, Any], ops: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Returns a patched deep copy; the input is never touched."""
    patched = copy.deepcopy(decisions)
    for op in ops:
        entry = _entry(patched, op["legacy_id"])
        parts = _parts(op["path"])
        node, key = _walk(entry, parts, create=(op["op"] == "set"))
        if op["op"] == "set":
            value = copy.deepcopy(op.get("after"))
            if isinstance(node, list):
                if key == len(node):
                    node.append(value)
                else:
                    node[key] = value
            else:
                node[key] = value
        elif op["op"] == "remove":
            if isinstance(node, list):
                del node[key]
            else:
                node.pop(key, None)
        else:
            fail(f"unknown op {op['op']!r}")
    return patched


def load_current_authoring_surface() -> Dict[str, Any]:
    """The authoring surface of the generator ON DISK NOW.

    `Evidence.surface` is the surface frozen into the evidence bundle, and the
    bundle is sha- and fingerprint-pinned, so it can never describe a generator
    that has since been widened. Historical proposals must keep classifying
    against the frozen surface; a NEW op has to be classified against the
    surface that will actually generate it. Same dict shape and same key names,
    so classify_op consumes either without a branch.
    """
    import importlib.util
    generator = ROOT / "scripts" / "migrate_catalog_kind_v2.py"
    spec = importlib.util.spec_from_file_location("migrate_catalog_kind_v2_live", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    from tools.catalog_validation import ECONOMIC_FIELDS
    # The constant was renamed when carryover overrides stopped being wording-only.
    # Accept either name so this reads correctly in an arm on either side of that
    # change: an older arm legitimately has the narrower surface.
    carryover = getattr(module, "CARRYOVER_OVERRIDE_FIELDS", None)
    if carryover is None:
        carryover = module.WORDING_OVERRIDE_FIELDS
    return {
        # Key name is the frozen bundle's schema and stays stable even though the
        # constant behind it stopped being wording-only.
        "wording_override_fields": sorted(carryover),
        "inherited_fields": list(module.INHERITED_FIELDS),
        "successor_required_fields": list(module.SUCCESSOR_REQUIRED_FIELDS),
        "economic_fields": list(ECONOMIC_FIELDS),
        "target_version": module.TARGET_VERSION,
    }


def current_surface_record() -> Dict[str, Any]:
    """The current surface plus the provenance that dates it."""
    generator = ROOT / "scripts" / "migrate_catalog_kind_v2.py"
    surface = load_current_authoring_surface()
    return {
        "schema_version": CURRENT_SURFACE_VERSION,
        "program": "catalog_audit",
        "what_this_is": (
            "The authoring surface of the generator at this commit, for classifying NEW ops. "
            "The evidence bundle's authoring_surface is the historical surface and is unchanged; "
            "it stays correct for every op classified before the generator was widened."
        ),
        "generator": {
            "path": "scripts/migrate_catalog_kind_v2.py",
            "sha256": sha256_file(generator),
            "git_blob": _git_blob(generator),
            "carryover_constant": _carryover_constant_name(),
        },
        "frozen_surface_reference": {
            "path": "reports/catalog_audit_evidence.json",
            "sha256": EVIDENCE_SHA256,
            "fingerprint": EVIDENCE_FINGERPRINT,
        },
        "surface": surface,
    }


def _carryover_constant_name() -> str:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "migrate_catalog_kind_v2_live", ROOT / "scripts" / "migrate_catalog_kind_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return ("CARRYOVER_OVERRIDE_FIELDS"
            if hasattr(module, "CARRYOVER_OVERRIDE_FIELDS") else "WORDING_OVERRIDE_FIELDS")


def _git_blob(path: Path) -> Optional[str]:
    import subprocess
    try:
        out = subprocess.run(["git", "hash-object", str(path)], cwd=str(ROOT),
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def classify_op(op: Dict[str, Any], change_type_after: str, surface: Dict[str, Any]) -> Tuple[str, str]:
    """`native` when the generator consumes the change as authored; `gap` otherwise."""
    parts = _parts(op["path"])
    economic = set(surface["economic_fields"])
    if any(p in economic for p in parts):
        return "gap", "path touches an economic field; economics are never authorable"
    head = parts[0]
    if head in ENTRY_META:
        return "native", "manifest metadata"
    if head == "change_type":
        return ("native", "change type") if op.get("after") in VALID_CHANGE_TYPES else ("gap", "unknown change type")
    if head != "successors":
        return "gap", f"unknown path head {head!r}"
    if len(parts) == 2:
        return "native", "successor added or removed"
    field = parts[2]
    if field == "overrides":
        if len(parts) < 4:
            return "gap", "override needs a field"
        f = parts[3]
        if change_type_after == "split":
            return ("native", "split successor inherited-field override") if f in surface["inherited_fields"] \
                else ("gap", "split successors may override only inherited structural fields")
        return ("native", "carryover override admitted by the authoring surface") if f in surface["wording_override_fields"] \
            else ("gap", "field is outside the carryover override set this authoring surface admits; economics are never authorable")
    if change_type_after == "split":
        return ("native", "split successor identity field") if field in IDENTITY_FIELDS \
            else ("gap", "unknown successor field")
    return ("native", "carryover kind/atomic_claim") if field in ("kind", "atomic_claim") \
        else ("gap", "carryover successor field not consumed by the generator; use overrides")


def metadata_problems(before: Dict[str, Any], after: Dict[str, Any], ops: Sequence[Dict[str, Any]]) -> List[str]:
    probs = []
    ct = after.get("change_type")
    touched = {_parts(op["path"])[0] for op in ops}
    if ct in ("split", "retired"):
        if after.get("deprecated") is not True:
            probs.append(f"{ct} entry must set deprecated=true")
        if after.get("requires_re_resolution") is not True:
            probs.append(f"{ct} entry must set requires_re_resolution=true")
    if ct == "retired" and after.get("successors"):
        probs.append("retired entry must have no successors")
    kinds_before = [s.get("kind") for s in before.get("successors") or []]
    kinds_after = [s.get("kind") for s in after.get("successors") or []]
    kind_changed = kinds_before != kinds_after
    claims_changed = [s.get("atomic_claim") for s in before.get("successors") or []] != \
        [s.get("atomic_claim") for s in after.get("successors") or []]
    if kind_changed and after.get("requires_re_resolution") is not True:
        probs.append("a kind change requires requires_re_resolution=true")
    successor_shape = any(len(_parts(op["path"])) == 2 and _parts(op["path"])[0] == "successors" for op in ops)
    if successor_shape or "change_type" in touched or kind_changed or claims_changed:
        for meta in ("atomicity_rationale", "expected_effects"):
            if meta not in touched:
                probs.append(f"successor/kind/claim/change_type edits must also refresh {meta}")
    return probs


def catalog_diff(base: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    b = {it["id"]: it for it in base["items"]}
    c = {it["id"]: it for it in cand["items"]}
    added, removed = sorted(set(c) - set(b)), sorted(set(b) - set(c))
    item_diff = {}
    for item_id in sorted(set(b) & set(c)):
        fields = {f: {"before": b[item_id].get(f), "after": c[item_id].get(f)}
                  for f in sorted(set(b[item_id]) | set(c[item_id])) if b[item_id].get(f) != c[item_id].get(f)}
        if fields:
            item_diff[item_id] = fields
    kept = [i for i in [it["id"] for it in cand["items"]] if i in b]
    order_ok = kept == [i for i in [it["id"] for it in base["items"]] if i in c]
    return {"added": added, "removed": removed, "item_diff": item_diff, "order_ok": order_ok}


def dry_run(baseline: Dict[str, Any], ops: Sequence[Dict[str, Any]], surface: Dict[str, Any]) -> Dict[str, Any]:
    if not ops:
        return {"ok": False, "error": "no ops", "classification": [], "metadata_problems": []}
    gen, v1, decisions = baseline["gen"], baseline["v1"], baseline["decisions"]
    try:
        patched = apply_ops(decisions, ops)
    except ProposalError as exc:
        return {"ok": False, "error": str(exc), "classification": [], "metadata_problems": []}
    legacy_ids = sorted({op["legacy_id"] for op in ops})
    entries_after = {lid: _entry(patched, lid) for lid in legacy_ids}
    classification = [{"path": op["path"], "legacy_id": op["legacy_id"],
                       **dict(zip(("class", "reason"), classify_op(op, entries_after[op["legacy_id"]].get("change_type"), surface)))}
                      for op in ops]
    meta = []
    for lid in legacy_ids:
        meta.extend(f"{lid}: {p}" for p in metadata_problems(_entry(decisions, lid), entries_after[lid],
                                                              [op for op in ops if op["legacy_id"] == lid]))
    try:
        catalog, manifest = gen.generate(copy.deepcopy(v1), copy.deepcopy(patched))
    except SystemExit as exc:
        return {"ok": False, "error": str(exc), "classification": classification, "metadata_problems": meta}
    diff = catalog_diff(baseline["catalog"], catalog)
    if not diff["added"] and not diff["removed"] and not diff["item_diff"]:
        return {"ok": False, "error": "ops change nothing in the generated catalog", "classification": classification,
                "metadata_problems": meta, **diff}
    base_entries = {e["legacy_id"]: e for e in baseline["manifest"]["entries"]}
    cand_entries = {e["legacy_id"]: e for e in manifest["entries"]}
    manifest_diff = {lid: {"before": base_entries.get(lid), "after": cand_entries.get(lid)}
                     for lid in legacy_ids if base_entries.get(lid) != cand_entries.get(lid)}
    by_id = {it["id"]: it for it in catalog["items"]}
    added_detail = {}
    for item_id in diff["added"]:
        it = by_id[item_id]
        added_detail[item_id] = {"pricing_status": it.get("pricing_status"),
                                 "economics": {f: it.get(f) for f in surface["economic_fields"]},
                                 "repair_support_marker_routes": sorted(room for room, e in (it.get("package_affinity") or {}).items()
                                                                        if isinstance(e, dict) and e.get("repair_support_when_driven"))}
    cat_v = validate_issue_catalog(catalog)
    man_v = validate_migration_manifest(manifest, v1, catalog)
    return {"ok": cat_v.ok and man_v.ok and not meta, "error": None, "classification": classification,
            "metadata_problems": meta, **diff, "manifest_diff": manifest_diff, "added_detail": added_detail,
            "catalog_validation": {"errors": list(cat_v.errors), "warnings": list(cat_v.warnings)},
            "manifest_validation": {"errors": list(man_v.errors), "warnings": list(man_v.warnings)}}


# --------------------------------------------------------------------------- evidence bar and derivation

def adjudication_of(review: Optional[Dict[str, Any]], row_id: str) -> Optional[str]:
    if review is None:
        return None
    row = review["rows"].get(row_id)
    return row.get("adjudication") if row else "missing"


def unit_row_id(pid: str, entry: Dict[str, Any]) -> str:
    """The pinned review row an evidence unit is read from (the cluster's own row unless bound elsewhere)."""
    return entry.get("review_row_id") or f"{pid}:{entry['unit_key']}"


def control_source(pid: str, judgment: Dict[str, Any]) -> str:
    return (judgment.get("controls") or {}).get("review_source_cluster") or pid


def row_prefix(row_id: Any) -> str:
    return str(row_id).split(":", 1)[0]


def is_cap_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(PROPOSAL_ID) and value[len(PROPOSAL_ID):].isdigit() and len(value) == 7


def cap_number(value: str) -> int:
    return int(value[len(PROPOSAL_ID):])


def evidence_bar(cluster: Dict[str, Any], ev: Evidence, adjud: Dict[str, Optional[str]],
                 photo_status: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    j = cluster["judgment"]
    entries = j.get("units") or []
    support = [ev.units[e["unit_key"]] for e in entries if e["role"] == "support"]
    refutable = [ev.units[e["unit_key"]] for e in entries if e["role"] in ("support", "counterexample")]
    counted = [u for u in support if adjud.get(u["unit_key"]) == "supports_claim"]
    human = [u for u in counted if u["primary_method"] in HUMAN_METHODS and u["independent"]]
    corrob_gold = [u for u in counted if u["primary_method"] in HUMAN_METHODS and not u["independent"]]
    leads = [u for u in counted if u["primary_method"] == "model_judge"]

    def available(u: Dict[str, Any]) -> bool:
        return photo_status is None or photo_status.get(u["unit_key"]) == "available"

    refuted = [u for u in refutable if adjud.get(u["unit_key"]) == "refuted" and available(u)]
    reviewed_refutations = sum(1 for u in refuted if u["primary_method"] in HUMAN_METHODS)
    lead_refutations = sum(1 for u in refuted if u["primary_method"] == "model_judge")
    mixed_signal = bool(human or corrob_gold) and any(
        adjud.get(u["unit_key"]) == "refuted" and u["primary_method"] in HUMAN_METHODS for u in support)
    props = {u["property_key"] for u in human}
    human_photos = {(u["property_key"], pk) for u in human for pk in u["photo_keys"]}
    human_conds = {(u["property_key"], u.get("condition_id")) for u in human}
    corroboration = "none"
    for u in (leads + corrob_gold) if human else []:
        shares = (bool({(u["property_key"], pk) for pk in u["photo_keys"]} & human_photos)
                  or (u["property_key"], u.get("condition_id")) in human_conds)
        if not shares:
            corroboration = "independent"
        elif corroboration == "none":
            corroboration = "method"
    correlated = sum(1 for u in human if sum(1 for v in human if v["property_key"] == u["property_key"]) > 1)
    claim = (j.get("evidence_bar_claim") or {}).get("bar_met_by")
    demo = (j.get("proposal") or {}).get("structural_exception_demonstration")
    demo_ok = (isinstance(demo, dict) and bool(demo.get("fact")) and isinstance(demo.get("item_ids"), list)
               and bool(demo.get("item_ids")) and isinstance(demo.get("fields"), list) and bool(demo.get("fields")))
    missing_coverage = bool((j.get("coverage_gap") or {}).get("is_missing_coverage"))
    if len(human) >= 2:
        met = "two_independent_human"
    elif len(human) == 1 and corroboration == "independent":
        met = "one_human_plus_independent_corroboration"
    elif claim == "structural_exception" and demo_ok and not missing_coverage and len(human) + len(corrob_gold) >= 1:
        met = "structural_exception"
    else:
        met = "none"
    contextual = Counter(adjud.get(u["unit_key"]) or "pending" for u in support if adjud.get(u["unit_key"]) != "supports_claim")
    return {"support_units": len(support), "adjudicated_support": len(counted), "human_units": len(human),
            "corroborating_gold_units": len(corrob_gold), "lead_units": len(leads), "independent_support": len(human),
            "distinct_properties": len(props), "correlated_support": correlated, "corroboration_type": corroboration,
            "bar_met_by": met, "contextual_units": dict(sorted(contextual.items())),
            "claimed_bar_met_by": claim, "consistent_with_claim": claim == met if claim is not None else None,
            "reviewed_refutations": reviewed_refutations, "lead_refutations": lead_refutations,
            "mixed_signal": mixed_signal}


def bind_controls(ev: Evidence, cluster: Dict[str, Any],
                  review: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Family controls joined to their pinned review rows. A reshaped cluster reads the source
    cluster's packet rows and keeps only the cards that were actually reviewed there; the
    dropped card ids are returned so the omission is visible."""
    j, pid = cluster["judgment"], cluster["proposal_id"]
    src = control_source(pid, j)
    reshaped = src != pid or bool((j.get("cluster") or {}).get("reshape"))
    excluded = set((j.get("controls") or {}).get("excluded") or [])
    packet_rows = (review or {}).get("packet_rows") if review else None
    out: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for ctl in select_controls(ev, cluster):
        row_id = f"{src}:control:{ctl['card_id']}"
        if reshaped and packet_rows is not None:
            prow = packet_rows.get(row_id)
            if prow is None:
                dropped.append(ctl["card_id"])
                continue
            ctl["required"] = prow.get("priority") == "required" and ctl["card_id"] not in excluded
        ctl["review_row_id"] = row_id
        ctl["adjudication"] = adjudication_of(review, row_id)
        ctl["review_reason"] = (((review or {}).get("rows") or {}).get(row_id) or {}).get("reason") if review else None
        out.append(ctl)
    return out, dropped


def card_role(ev: Evidence, card_id: str) -> str:
    if card_id in ev.labels:
        cls = ev.labels[card_id].get("class_v1_1") or ""
        return {"supported_billed": "positive_use", "hard_false_billed": "hallucination"}.get(cls, cls or "label")
    lane = (ev.cases.get(card_id) or {}).get("lane") or ""
    if "halluc" in lane:
        return "hallucination"
    if "correct_rejection" in lane:
        return "correct_rejection"
    if "agreement" in lane:
        return "agreement"
    return lane or "case"


def family_cards(ev: Evidence, item_id: str) -> List[Tuple[str, str]]:
    """Every reviewed card the bundle knows for an item: the worklist family when it exists,
    otherwise the v1.1 labels and attribution cases resolved to the item."""
    w = ev.worklist.get(item_id)
    if w:
        return [(cid, role) for field, role in CONTROL_ROLES.items() for cid in (w.get(field) or [])]
    cards = [(l["card_id"], card_role(ev, l["card_id"])) for l in ev.labels.values() if l.get("catalog_item_id") == item_id]
    cards += [(cid, card_role(ev, cid)) for cid in (ev.by_item.get(item_id) or {}).get("cases") or []]
    seen: set = set()
    out = []
    for cid, role in cards:
        if cid not in seen:
            seen.add(cid)
            out.append((cid, role))
    return out


def target_item_controls(ev: Evidence, cluster: Dict[str, Any], review: Optional[Dict[str, Any]],
                         run: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Regression context for items an op touches without implicating them: every card the bundle
    holds for the item, with its pinned review row wherever some cluster's packet reviewed it."""
    j = cluster["judgment"]
    implicated = set(j.get("implicated_items") or [])
    touched = set((run or {}).get("item_diff") or {}) | set((run or {}).get("added") or []) | set((run or {}).get("removed") or [])
    touched |= set((j.get("proposal") or {}).get("target_items") or [])
    rows = ((review or {}).get("rows") or {}) if review else {}
    out = []
    for item_id in sorted(touched - implicated):
        for cid, role in family_cards(ev, item_id):
            src = ev.cases.get(cid) or ev.labels.get(cid) or {}
            truth = src.get("human_truth") if cid in ev.cases else {k: src.get(k) for k in ("claim", "work", "class_v1_1")}
            hits = sorted(rid for rid in rows if rid.endswith(f":control:{cid}"))
            out.append({"item_id": item_id, "card_id": cid, "role": role, "lane": (ev.cases.get(cid) or {}).get("lane"),
                        "property_key": src.get("property_key"), "human_truth": truth, "terra_verdict": src.get("terra_verdict"),
                        "review_row_id": hits[0] if hits else None, "reviewed_under": [row_prefix(h) for h in hits],
                        "adjudication": rows[hits[0]].get("adjudication") if hits else None, "reviewed": bool(hits)})
    return out


def derive_cluster(cluster: Dict[str, Any], ev: Evidence, baseline: Dict[str, Any], review: Optional[Dict[str, Any]],
                   photo_root: Path, cache: Dict[str, Any]) -> Dict[str, Any]:
    j = cluster["judgment"]
    pid = cluster["proposal_id"]
    units = []
    adjud: Dict[str, Optional[str]] = {}
    for entry in sorted(j.get("units") or [], key=lambda e: e["unit_key"]):
        u = ev.units[entry["unit_key"]]
        rids = [r["record_id"] for r in u["records"]]
        row_id = unit_row_id(pid, entry)
        a = adjudication_of(review, row_id)
        adjud[entry["unit_key"]] = a
        row = (review or {}).get("rows", {}).get(row_id) if review else None
        units.append({"unit_key": entry["unit_key"], "role": entry["role"], "review_row_id": row_id,
                      "transfer_note": entry.get("transfer_note"), "evidence_class": u["primary_method"],
                      "property_key": u["property_key"], "independent": u["independent"], "rules": u["rules"],
                      "photos": photo_refs(ev, entry["unit_key"], rids, photo_root, cache),
                      "photo_review": {"status": "reviewed" if a not in (None, "missing", "unavailable") else
                                       ("pending" if a is None else a), "adjudication": a,
                                       "reason": (row or {}).get("reason"), "what_image_shows": (row or {}).get("what_image_shows"),
                                       "lineage_note": (row or {}).get("lineage_note")},
                      "lineage": ev.lineage(entry["unit_key"], rids)})
    controls, dropped = bind_controls(ev, cluster, review)
    implicated = j.get("implicated_items") or []
    items = {i: ev.item_semantics(i) for i in implicated}
    family_members: set = set()
    frozen_neighbors: Dict[str, Dict[str, Any]] = {}
    per_item_counts = {}
    for i in implicated:
        fam = ev.families.get(i) or {}
        family_members.update(fam.get("family_members") or [])
        for n in fam.get("frozen_candidate_neighbors") or []:
            cur = frozen_neighbors.setdefault(n["item_id"], {"item_id": n["item_id"], "co_listed": 0, "best_rank": n["best_rank"]})
            cur["co_listed"] += n["co_listed"]
            cur["best_rank"] = min(cur["best_rank"], n["best_rank"])
        w = ev.worklist.get(i) or {}
        per_item_counts[i] = {k: len(w.get(k) or []) for k in ("positive_uses", "agreements", "correct_rejections",
                                                                 "notes", "hallucination_annotations")}
        per_item_counts[i]["migration"] = fam.get("migration")
    family_members -= set(implicated)
    family_members |= set((j.get("family") or {}).get("added") or [])
    family_members -= set((j.get("family") or {}).get("removed") or [])
    ops = fill_before(baseline["decisions"], (j.get("proposal") or {}).get("ops") or [])
    run = dry_run(baseline, ops, ev.surface) if ops else None
    retirement = {}
    if run and run.get("removed"):
        for item_id in run["removed"]:
            it = baseline["items"][item_id]
            retirement[item_id] = {"work_item_code": it.get("work_item_code"), "estimate": it.get("estimate"),
                                   "cost": it.get("cost"), "package_affinity": it.get("package_affinity"),
                                   "route_override": it.get("route_override"), "scene_groups": it.get("scene_groups")}
    retrieval_after = {}
    if run and run.get("ok"):
        for item_id, fields in run["item_diff"].items():
            retrieval_after[item_id] = {f: fields[f]["after"] for f in fields}
    if run and run.get("ok") and run["added"]:
        patched = apply_ops(baseline["decisions"], ops)
        cat, _ = baseline["gen"].generate(copy.deepcopy(baseline["v1"]), patched)
        by_id = {i["id"]: i for i in cat["items"]}
        for item_id in run["added"]:
            it = by_id[item_id]
            retrieval_after[item_id] = {**{f: it.get(f) for f in SEMANTIC_SNAPSHOT}, "claim_text": claim_text(it),
                                        "embed_source_text": embed_source_text(it)}
    targets = target_item_controls(ev, cluster, review, run)
    quarantined = any(items[i].get("trade_quarantined") for i in implicated)
    coverage = j.get("coverage_gap") or {}
    parent = coverage.get("split_parent")
    parent_ok = None
    if parent:
        parent_ok = parent in ev.migration["change_type"] and ev.migration["change_type"].get(parent) != "retired"
    bar = evidence_bar(cluster, ev, adjud, {u["unit_key"]: u["photos"]["status"] for u in units})
    return {"units": units, "controls": controls, "controls_dropped": dropped, "target_item_controls": targets,
            "per_item": per_item_counts,
            "item_semantics": items,
            "family_members": sorted(family_members), "frozen_candidate_neighbors": sorted(
                frozen_neighbors.values(), key=lambda n: (-n["co_listed"], n["best_rank"], n["item_id"])),
            "ops": ops, "dry_run": run, "retrieval_after": retrieval_after, "retirement_lost_behavior": retirement,
            "trade_quarantined": quarantined, "coverage_gap": {"declared": bool(coverage.get("is_missing_coverage")),
                                                               "split_parent": parent, "split_parent_legitimate": parent_ok},
            "evidence_bar": bar,
            "adjudication_counts": dict(sorted(Counter(a or "pending" for a in adjud.values()).items()))}


def derive(proposals: Dict[str, Any], ev: Evidence, baseline: Dict[str, Any], review: Optional[Dict[str, Any]],
           packet_on_disk: Optional[str], packet_fresh: Optional[str],
           known_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    out = copy.deepcopy(proposals)
    photo_root = Path(out["review"]["photo_root"])
    cache: Dict[str, Any] = {}
    for cluster in out["clusters"]:
        cluster["derived"] = derive_cluster(cluster, ev, baseline, review, photo_root, cache)
    triage_rows = list(out.get("coverage_triage") or [])
    present = {t.get("unit_key") for t in triage_rows}
    triage_rows += [{"unit_key": k, "judgment": {"triage_class": None, "ref": None, "note": None, "not_promoted_reason": None}}
                    for k in ev.coverage_units if k not in present]
    for row in triage_rows:
        unit_key = row.get("unit_key")
        if unit_key not in ev.units:
            row["derived"] = None
            continue
        u = ev.units[unit_key]
        rids = [r["record_id"] for r in u["records"]]
        lin = ev.lineage(unit_key, rids)
        a = adjudication_of(review, f"coverage:{unit_key}")
        gold = lin.get("gold") or {}
        row["derived"] = {"finding": gold.get("finding"), "matching_decision": gold.get("matching_decision"),
                          "attribution": gold.get("attribution"), "covering_item": gold.get("covering_item"),
                          "property_key": u["property_key"], "photo": f"{u['property_key']}/{u['photo_key']}",
                          "adjudication": a, "review_reason": ((review or {}).get("rows", {}).get(f"coverage:{unit_key}") or {}).get("reason") if review else None}
    out["coverage_triage"] = sorted(triage_rows, key=lambda t: str(t.get("unit_key")))
    coverage_map: Dict[str, List[str]] = {i: [] for i in ev.worklist}
    for cluster in out["clusters"]:
        for i in cluster["judgment"].get("implicated_items") or []:
            coverage_map.setdefault(i, []).append(cluster["proposal_id"])
    out["derived"] = {
        "baseline": {"proposal_baseline_commit": "e9a7dc3a54439fb341c1ae8cd665e5e572fcfaf6",
                     "generated_catalog_commit": ev.bundle["catalog_identity"]["proposal_baseline"].get("catalog_commit"),
                     "evidence_era": ev.bundle["catalog_identity"]["evidence_era"],
                     "evidence_sha256": EVIDENCE_SHA256, "evidence_fingerprint": EVIDENCE_FINGERPRINT,
                     "sources": baseline["hashes"], "bundle_git_head": ev.bundle["git"]["head"]},
        "review": {"packet_fingerprint_on_disk": packet_on_disk, "packet_fingerprint_regenerated": packet_fresh,
                   "packet_file_sha256": None,
                   "packet_drift": (packet_on_disk != packet_fresh) if packet_on_disk and packet_fresh else None,
                   "results_sha256": review["sha256"] if review else None, "results_path": review["path"] if review else None,
                   "reviewer": review.get("reviewer") if review else None,
                   "pinned": {"review_sha256": REVIEW_SHA256, "packet_sha256": PACKET_SHA256},
                   "pins_verified": bool(review) and review.get("sha256") == REVIEW_SHA256
                   and review.get("packet_sha256") == PACKET_SHA256},
        "id_ledger": id_ledger(out["clusters"], known_ids),
        "pass_2d_follow_ups": follow_up_register(out["clusters"]),
        "regression_register": regression_register(out["clusters"]),
        "worklist_coverage": {k: sorted(v) for k, v in sorted(coverage_map.items())},
        "counts": {"clusters": len(out["clusters"]),
                   "by_outcome": dict(sorted(Counter(c["judgment"].get("outcome_type") or "provisional" for c in out["clusters"]).items())),
                   "by_bar": dict(sorted(Counter(c["derived"]["evidence_bar"]["bar_met_by"] for c in out["clusters"]).items())),
                   "coverage_by_triage": dict(sorted(Counter(t["judgment"].get("triage_class") or "pending" for t in out["coverage_triage"]).items())),
                   "by_mismatch_class": dict(sorted(Counter((c["judgment"].get("cluster") or {}).get("mismatch_class") or "unset"
                                                            for c in out["clusters"]).items())),
                   "by_deferral_reason": dict(sorted(Counter(c["judgment"].get("deferral_reason") for c in out["clusters"]
                                                             if c["judgment"].get("deferral_reason")).items()))},
    }
    out["derived"]["validation"] = validate(out, ev, review, known_ids)
    out["clusters"].sort(key=lambda c: c["proposal_id"])
    return out


def id_ledger(clusters: Sequence[Dict[str, Any]], known_ids: Optional[Sequence[str]]) -> Dict[str, Any]:
    known = list(known_ids or [])
    ids = [c.get("proposal_id") for c in clusters]
    nums = [cap_number(k) for k in known if is_cap_id(k)]
    return {"known_ids": known, "max_known": (f"CAP-{max(nums):03d}" if nums else None),
            "new_ids": sorted(i for i in ids if i not in known)}


def follow_up_register(clusters: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Reachable-right-item selection failures, grouped by confusion family for the post-catalog Pass 2d task."""
    reg: Dict[str, List[Dict[str, Any]]] = {}
    for c in clusters:
        for f in c["judgment"].get("pass_2d_follow_ups") or []:
            if isinstance(f, dict):
                reg.setdefault(str(f.get("confusion_family")), []).append({"proposal_id": c["proposal_id"], **f})
    return {k: sorted(v, key=lambda x: (x["proposal_id"], str(x.get("case_ref")))) for k, v in sorted(reg.items())}


def regression_register(clusters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every hallucination and correct-rejection control, with the expectation each proposal states for it."""
    reg = []
    for c in clusters:
        d, j = c.get("derived") or {}, c["judgment"]
        expect = {r.get("card_id"): r for r in (j.get("proposal") or {}).get("regression_controls") or [] if isinstance(r, dict)}
        for ctl in (d.get("controls") or []) + (d.get("target_item_controls") or []):
            if ctl.get("role") in REGRESSION_ROLES:
                e = expect.get(ctl["card_id"]) or {}
                reg.append({"proposal_id": c["proposal_id"], "outcome": j.get("outcome_type"), "card_id": ctl["card_id"],
                            "role": ctl["role"], "item_id": ctl.get("item_id"), "adjudication": ctl.get("adjudication"),
                            "expected_after": e.get("expected_after"), "note": e.get("note")})
    return sorted(reg, key=lambda r: (r["proposal_id"], r["role"], r["card_id"]))


# --------------------------------------------------------------------------- validation

def validate(p: Dict[str, Any], ev: Evidence, review: Optional[Dict[str, Any]],
             known_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    errors: List[str] = []
    pending: List[str] = []
    provisional = review is None
    packet_rows = (review or {}).get("packet_rows") if review else None
    bound: Dict[str, List[str]] = {}      # review row id -> proposal ids that read it
    cov_bound: Dict[str, str] = {}        # coverage unit key -> proposal id that binds it

    def err(msg: str) -> None:
        errors.append(msg)

    for key in ("schema_version", "program", "session", "proposal_date", "policy", "review", "clusters", "coverage_triage"):
        if key not in p:
            err(f"missing top-level key {key}")
    if not Path(p.get("review", {}).get("photo_root", "")).is_dir():
        err("review.photo_root is not a directory")
    ids = [c.get("proposal_id") for c in p.get("clusters") or []]
    for pid, n in Counter(ids).items():
        if n > 1:
            err(f"duplicate proposal id {pid}")
        if not is_cap_id(pid):
            err(f"proposal id {pid!r} must match CAP-NNN")
    if ids != sorted(ids):
        err("clusters are not in proposal-id order")
    covered: set = set()
    for c in p.get("clusters") or []:
        pid = c.get("proposal_id")
        j = c.get("judgment") or {}
        d = c.get("derived") or {}
        for key in ("cluster", "implicated_items", "family", "units", "outcome_type", "diagnosis",
                    "alternative_stage_analysis", "proposal", "evidence_bar_claim", "absence_statements",
                    "confidence", "unresolved_questions", "coverage_gap"):
            if key not in j:
                err(f"{pid}: judgment missing {key}")
        if c.get("human_disposition") != {"disposition": None, "approved_diff": None, "conditions": [], "notes": ""}:
            err(f"{pid}: human_disposition must be blank")
        cl = j.get("cluster") or {}
        for key in ("physical_subject", "visible_state_or_mechanism", "claim_under_test", "merge_split_rationale"):
            if not cl.get(key):
                err(f"{pid}: cluster.{key} is empty")
        for i in j.get("implicated_items") or []:
            if i not in ev.items and i not in ev.families:
                err(f"{pid}: implicated item {i} is not in the catalog")
            covered.add(i)
        roles = Counter()
        for e in j.get("units") or []:
            if e.get("unit_key") not in ev.units:
                err(f"{pid}: unit {e.get('unit_key')} is not in the bundle")
            if e.get("role") not in UNIT_ROLES:
                err(f"{pid}: unit role {e.get('role')!r} invalid")
            roles[e.get("role")] += 1
        if roles["support"] == 0:
            err(f"{pid}: no support units")
        for cid in (j.get("controls") or {}).get("excluded") or []:
            if cid not in ev.cases and cid not in ev.labels:
                err(f"{pid}: excluded control {cid} unknown")

        # ---- review-row bindings, reshaping provenance, human inputs, follow-ups, regression controls
        reshape = cl.get("reshape")
        sources: List[str] = []
        if reshape is not None:
            if not isinstance(reshape, dict) or not isinstance(reshape.get("source_cluster_ids"), list) \
                    or not reshape.get("source_cluster_ids"):
                err(f"{pid}: cluster.reshape must be {{source_cluster_ids: [..], claim_relation, rationale}}")
            else:
                sources = [str(x) for x in reshape["source_cluster_ids"]]
                for src_id in sources:
                    if not is_cap_id(src_id):
                        err(f"{pid}: reshape source {src_id!r} must be a CAP-NNN id")
                    elif known_ids is not None and src_id not in known_ids and src_id not in ids:
                        err(f"{pid}: reshape source {src_id} is not a known cluster id")
                if reshape.get("claim_relation") not in RESHAPE_RELATIONS:
                    err(f"{pid}: reshape.claim_relation must be equivalent or narrower")
                if not reshape.get("rationale"):
                    err(f"{pid}: reshape.rationale is empty")
        for e in j.get("units") or []:
            unit_key = e.get("unit_key")
            extra = set(e) - set(UNIT_ENTRY_KEYS)
            if extra:
                err(f"{pid}: unit {unit_key} has unsupported keys {sorted(extra)}; adjudications come only from the pinned review")
            rid = unit_row_id(pid, e)
            if not isinstance(rid, str) or not rid.endswith(f":{unit_key}"):
                err(f"{pid}: review_row_id {rid!r} does not carry unit {unit_key}")
                continue
            prefix = row_prefix(rid)
            if prefix == COVERAGE_PREFIX:
                cov_bound[unit_key] = pid
                if not e.get("transfer_note"):
                    err(f"{pid}: coverage-bound unit {unit_key} needs a transfer_note")
            elif prefix != pid:
                if reshape is None or prefix not in sources:
                    err(f"{pid}: unit {unit_key} is bound to {rid} but cluster.reshape does not list {prefix} as a source")
                if not e.get("transfer_note"):
                    err(f"{pid}: transferred unit {unit_key} needs a transfer_note")
            if packet_rows is not None:
                prow = packet_rows.get(rid)
                if prow is None:
                    err(f"{pid}: review row {rid} is not in the pinned packet")
                elif prow.get("row_type") not in ("unit", "coverage") or prow.get("unit_key") != unit_key:
                    err(f"{pid}: review row {rid} is a {prow.get('row_type')} row for {prow.get('unit_key')}, not unit {unit_key}")
            bound.setdefault(rid, []).append(pid)
        src = control_source(pid, j)
        if src != pid and (reshape is None or src not in sources):
            err(f"{pid}: controls.review_source_cluster {src} is not a reshape source")
        if packet_rows is not None:
            for ctl in d.get("controls") or []:
                rid = ctl.get("review_row_id")
                prow = packet_rows.get(rid)
                if ctl.get("required") and prow is None:
                    err(f"{pid}: required control {ctl.get('card_id')} has no review row {rid} in the pinned packet")
                elif prow is not None and (prow.get("row_type") != "control" or prow.get("card_id") != ctl.get("card_id")):
                    err(f"{pid}: review row {rid} is not the control row for {ctl.get('card_id')}")
            if d.get("controls_dropped") and not d.get("controls"):
                reviewed_items = {pr.get("item_id") for pr in packet_rows.values() if pr.get("row_type") == "control"}
                recoverable = sorted(set(j.get("implicated_items") or []) & reviewed_items)
                if recoverable:
                    err(f"{pid}: no reviewed controls resolved under {src}, but the packet holds control rows for "
                        f"{recoverable}; check controls.review_source_cluster")
        human_inputs = j.get("human_inputs", [])
        if not isinstance(human_inputs, list):
            err(f"{pid}: human_inputs must be a list")
        else:
            for i, h in enumerate(human_inputs):
                if not isinstance(h, dict) or set(h) != set(HUMAN_INPUT_KEYS) or not all(isinstance(h[k], str) for k in HUMAN_INPUT_KEYS):
                    err(f"{pid}: human_inputs[{i}] must be {{ref, source, ruling, effect}} of strings")
                elif not h["source"]:
                    err(f"{pid}: human_inputs[{i}].source is empty")
        follow_ups = j.get("pass_2d_follow_ups", [])
        if not isinstance(follow_ups, list):
            err(f"{pid}: pass_2d_follow_ups must be a list")
        else:
            for i, f in enumerate(follow_ups):
                if not isinstance(f, dict) or set(f) != set(FOLLOW_UP_KEYS):
                    err(f"{pid}: pass_2d_follow_ups[{i}] must be {{case_ref, selected_item, better_item, better_rank, confusion_family, note}}")
                    continue
                if not f.get("case_ref") or not f.get("confusion_family"):
                    err(f"{pid}: pass_2d_follow_ups[{i}] needs case_ref and confusion_family")
                for k in ("selected_item", "better_item"):
                    if f.get(k) not in ev.items:
                        err(f"{pid}: pass_2d_follow_ups[{i}].{k} {f.get(k)!r} is not in the catalog")
                rank = f.get("better_rank")
                if rank is not None and (not isinstance(rank, int) or isinstance(rank, bool) or rank < 1):
                    err(f"{pid}: pass_2d_follow_ups[{i}].better_rank must be a positive integer or null")
                if rank is None and not f.get("note"):
                    err(f"{pid}: pass_2d_follow_ups[{i}] has no rank, so the note must say the better item was "
                        f"absent from the frozen candidate pool")
        all_control_cards = {x["card_id"] for x in (d.get("controls") or []) + (d.get("target_item_controls") or [])}
        regression_cards = {x["card_id"] for x in (d.get("controls") or []) + (d.get("target_item_controls") or [])
                            if x.get("role") in REGRESSION_ROLES}
        stated_rc: set = set()
        rcs = (j.get("proposal") or {}).get("regression_controls", [])
        if not isinstance(rcs, list):
            err(f"{pid}: proposal.regression_controls must be a list")
        else:
            for i, r in enumerate(rcs):
                if not isinstance(r, dict) or set(r) != {"card_id", "expected_after", "note"}:
                    err(f"{pid}: regression_controls[{i}] must be {{card_id, expected_after, note}}")
                    continue
                if r["card_id"] not in all_control_cards and r["card_id"] not in ev.cases and r["card_id"] not in ev.labels:
                    err(f"{pid}: regression_controls[{i}] names {r['card_id']}, which is not a control of this "
                        f"cluster nor any reviewed card in the evidence bundle")
                if r["expected_after"] not in CONTROL_EXPECTATIONS:
                    err(f"{pid}: regression_controls[{i}].expected_after {r['expected_after']!r} invalid")
                stated_rc.add(r["card_id"])

        outcome = j.get("outcome_type")
        status = c.get("status")
        if outcome is None:
            if status != "provisional":
                err(f"{pid}: status must be provisional while outcome_type is null")
            pending.append(f"{pid}: outcome pending")
            continue
        if outcome not in OUTCOMES:
            err(f"{pid}: outcome_type {outcome!r} invalid")
        if status != outcome:
            err(f"{pid}: status must mirror outcome_type")
        diag = j.get("diagnosis") or {}
        if diag.get("primary_cause") not in CAUSES:
            err(f"{pid}: diagnosis.primary_cause {diag.get('primary_cause')!r} invalid")
        if not diag.get("catalog_ownership_argument"):
            err(f"{pid}: diagnosis.catalog_ownership_argument is empty")
        if not isinstance(diag.get("rejected_alternative_owners"), list):
            err(f"{pid}: diagnosis.rejected_alternative_owners must be a list")
        asa = j.get("alternative_stage_analysis") or {}
        for q in STAGE_QUESTIONS:
            if not asa.get(q):
                err(f"{pid}: alternative_stage_analysis.{q} unanswered")
        for key in ("success_uses", "correct_rejections", "counterexamples"):
            if (j.get("absence_statements") or {}).get(key) is None:
                err(f"{pid}: absence_statements.{key} must be a statement or an explicit list note")
        if not d.get("controls") and not str((j.get("absence_statements") or {}).get("success_uses") or "").strip():
            err(f"{pid}: no reviewed controls exist; absence_statements.success_uses must say so explicitly")
        if j.get("confidence") not in ("low", "medium", "high"):
            err(f"{pid}: confidence invalid")
        bar = d.get("evidence_bar") or {}
        claim = (j.get("evidence_bar_claim") or {}).get("bar_met_by")
        if claim not in BAR:
            err(f"{pid}: evidence_bar_claim.bar_met_by invalid")
        if not (j.get("evidence_bar_claim") or {}).get("justification"):
            err(f"{pid}: evidence_bar_claim.justification is empty")

        # ---- operational equivalence, deferral reasons, structural-exception guard, native prerequisites
        mc = cl.get("mismatch_class")
        dims = cl.get("equivalence_dimensions")
        if mc not in MISMATCH_CLASSES:
            err(f"{pid}: cluster.mismatch_class {mc!r} invalid")
        if not isinstance(dims, list) or any(x not in EQUIVALENCE_DIMENSIONS for x in dims):
            err(f"{pid}: cluster.equivalence_dimensions must list dimensions from {list(EQUIVALENCE_DIMENSIONS)}")
        elif (mc == "operationally_consequential") != bool(dims):
            err(f"{pid}: equivalence_dimensions must be non-empty exactly when mismatch_class is operationally_consequential")
        reason = j.get("deferral_reason")
        if outcome == DEFERRED and reason not in DEFERRAL_REASONS:
            err(f"{pid}: deferred outcome needs deferral_reason in {list(DEFERRAL_REASONS)}")
        if outcome != DEFERRED and reason is not None:
            err(f"{pid}: deferral_reason is only for deferred outcomes")
        missing_coverage = bool((j.get("coverage_gap") or {}).get("is_missing_coverage"))
        if claim == "structural_exception":
            demo = (j.get("proposal") or {}).get("structural_exception_demonstration")
            if missing_coverage:
                err(f"{pid}: structural_exception may not be claimed for a missing-coverage cluster")
            if not (isinstance(demo, dict) and demo.get("fact") and isinstance(demo.get("item_ids"), list) and demo.get("item_ids")
                    and isinstance(demo.get("fields"), list) and demo.get("fields")):
                err(f"{pid}: proposal.structural_exception_demonstration must be {{fact: str, item_ids: [..], fields: [..]}}")
            else:
                for item_id in demo["item_ids"]:
                    if item_id not in ev.items:
                        err(f"{pid}: structural_exception_demonstration item {item_id} is not in the catalog")
                for field in demo["fields"]:
                    if field not in STRUCTURAL_FIELDS:
                        err(f"{pid}: structural_exception_demonstration field {field!r} is not a catalog semantic field")
        if outcome == "native_decisions_proposal":
            if mc != "operationally_consequential" and not missing_coverage:
                err(f"{pid}: native proposal needs mismatch_class operationally_consequential or a declared coverage gap")
            missing_rc = sorted(regression_cards - stated_rc)
            if missing_rc:
                err(f"{pid}: native proposal lacks regression_controls for {missing_rc}")

        if provisional:
            pending.append(f"{pid}: adjudication-dependent rules pending")
        else:
            human_support = [u for u in d.get("units") or [] if u["role"] == "support"
                             and u["evidence_class"] in HUMAN_METHODS
                             and u["photo_review"]["adjudication"] == "supports_claim" and u["photos"]["status"] == "available"]
            nonnative_ok = outcome in REFUTATION_BACKED and (bar.get("reviewed_refutations", 0) >= 1 or bool(human_support))
            if bar.get("bar_met_by") != claim:
                err(f"{pid}: evidence bar recomputed as {bar.get('bar_met_by')} but claimed {claim}")
            if bar.get("bar_met_by") == "none" and outcome not in (DEFERRED, "no_change") and not nonnative_ok:
                err(f"{pid}: bar not met; outcome must be deferred or no_change (or a refutation-backed "
                    f"non_catalog_action / migration_system_gap)")
            if bar.get("bar_met_by") == "none" and outcome != DEFERRED and missing_coverage:
                err(f"{pid}: declared coverage gap needs a met evidence bar")
            if bar.get("bar_met_by") == "two_independent_human" and bar.get("distinct_properties", 0) < 2 \
                    and not (j.get("evidence_bar_claim") or {}).get("correlation_note"):
                err(f"{pid}: same-property support needs evidence_bar_claim.correlation_note")
            if bar.get("mixed_signal") and not (j.get("evidence_bar_claim") or {}).get("conflict_note"):
                err(f"{pid}: mixed human signal (supports_claim and refuted) needs evidence_bar_claim.conflict_note")
            if outcome != DEFERRED:
                if not human_support and not nonnative_ok:
                    err(f"{pid}: non-deferred outcome without a reviewed supporting unit or reviewed refutation")
                for u in d.get("units") or []:
                    if u["role"] == "support" and u["photo_review"]["adjudication"] in (None, "missing"):
                        err(f"{pid}: support unit {u['unit_key']} has no review answer")
        if d.get("trade_quarantined") and outcome != "non_catalog_action":
            err(f"{pid}: product-quarantined trade must be non_catalog_action")
        cov = d.get("coverage_gap") or {}
        if cov.get("declared"):
            if cov.get("split_parent") and cov.get("split_parent_legitimate") is False:
                err(f"{pid}: coverage split parent is not a legitimate live legacy entry")
            if not cov.get("split_parent") and outcome not in ("migration_system_gap", DEFERRED, "no_change"):
                err(f"{pid}: missing coverage without a split parent must be migration_system_gap")
        run = d.get("dry_run")
        prop = j.get("proposal") or {}
        if outcome == "native_decisions_proposal":
            if not prop.get("ops"):
                err(f"{pid}: native proposal without ops")
            elif run is None or not run.get("ok"):
                err(f"{pid}: native proposal dry run failed: {(run or {}).get('error')}; {(run or {}).get('metadata_problems')}")
            else:
                gaps = [x for x in run["classification"] if x["class"] == "gap"]
                if gaps:
                    err(f"{pid}: native proposal has non-native ops: {[g['path'] for g in gaps]}")
                expected = prop.get("expected_item_changes") or {}
                actual = {i: sorted(f) for i, f in run["item_diff"].items()}
                for i in run["added"]:
                    actual[i] = ["<added>"]
                for i in run["removed"]:
                    actual[i] = ["<removed>"]
                if {k: sorted(v) for k, v in expected.items()} != actual:
                    err(f"{pid}: dry-run changes {actual} differ from expected_item_changes {expected}")
                if not run.get("order_ok"):
                    err(f"{pid}: item order changed")
                if run["added"] and not prop.get("economics_fit"):
                    err(f"{pid}: split adds items; economics_fit is required")
                if run["removed"] and not d.get("retirement_lost_behavior"):
                    err(f"{pid}: removal without lost-behavior snapshot")
            for key in ("semantic_before_after", "retrieval_effects", "package_route_effects", "expected_benefit", "regression_risk"):
                if not prop.get(key):
                    err(f"{pid}: proposal.{key} is empty")
        if outcome == "migration_system_gap":
            if prop.get("ops") and run and run.get("ok") and all(x["class"] == "native" for x in run["classification"]):
                err(f"{pid}: ops are fully native and succeed; this is not a migration gap")
            if not prop.get("gap_description"):
                err(f"{pid}: migration gap needs proposal.gap_description")
        if outcome in ("non_catalog_action", "no_change", DEFERRED) and prop.get("ops"):
            err(f"{pid}: {outcome} must not carry ops")
    for i in ev.worklist:
        if i not in covered:
            err(f"worklist item {i} is in no cluster")
    seen_cov = Counter(t.get("unit_key") for t in p.get("coverage_triage") or [])
    for unit_key in ev.coverage_units:
        if seen_cov[unit_key] != 1:
            err(f"coverage unit {unit_key} appears {seen_cov[unit_key]} times in coverage_triage")
    for t in p.get("coverage_triage") or []:
        tj = t.get("judgment") or {}
        if t.get("unit_key") not in ev.units:
            err(f"coverage_triage unit {t.get('unit_key')} unknown")
        if tj.get("triage_class") is None:
            pending.append(f"coverage {t.get('unit_key')}: triage pending")
        elif tj["triage_class"] not in TRIAGE:
            err(f"coverage {t.get('unit_key')}: triage class {tj['triage_class']!r} invalid")
        elif tj["triage_class"] == "attaches_to_cluster" and tj.get("ref") not in ids:
            err(f"coverage {t.get('unit_key')}: attaches_to_cluster needs a cluster ref")
        elif tj["triage_class"] == "covered_by_existing_item" and tj.get("ref") not in ev.items:
            err(f"coverage {t.get('unit_key')}: covered_by_existing_item needs an item ref")
        elif tj["triage_class"] == "coverage_gap_candidate" and not (tj.get("ref") in ids or tj.get("not_promoted_reason")):
            err(f"coverage {t.get('unit_key')}: gap candidate needs a cluster ref or not_promoted_reason")
    # ---- one reviewed row grounds one cluster; coverage promotion is consistent both ways; id ledger
    for rid, pids in sorted(bound.items()):
        if len(pids) > 1:
            err(f"review row {rid} is bound by {pids}; one reviewed row grounds one cluster")
    triage_by_unit = {t.get("unit_key"): (t.get("judgment") or {}) for t in p.get("coverage_triage") or []}
    for unit_key, pid in sorted(cov_bound.items()):
        tj = triage_by_unit.get(unit_key) or {}
        if tj.get("triage_class") not in ("attaches_to_cluster", "coverage_gap_candidate") or tj.get("ref") != pid:
            err(f"coverage {unit_key}: bound by {pid} but triage is {tj.get('triage_class')} ref {tj.get('ref')}")
    for unit_key, tj in sorted(triage_by_unit.items(), key=lambda kv: str(kv[0])):
        if tj.get("triage_class") in ("attaches_to_cluster", "coverage_gap_candidate") and tj.get("ref") in ids \
                and cov_bound.get(unit_key) != tj.get("ref"):
            err(f"coverage {unit_key}: triage refs {tj.get('ref')} but {tj.get('ref')} does not bind coverage:{unit_key}")
    if known_ids is not None:
        known = list(known_ids)
        for kid in known:
            if kid not in ids:
                err(f"ledger id {kid} is missing from clusters; ids are never removed")
        max_known = max((cap_number(k) for k in known if is_cap_id(k)), default=0)
        for pid in ids:
            if is_cap_id(pid) and pid not in known and cap_number(pid) <= max_known:
                err(f"proposal id {pid} is not in the id ledger and not above CAP-{max_known:03d}; "
                    f"new ids allocate from CAP-{max_known + 1:03d}")
    return {"ok": not errors, "mode": "provisional" if provisional else "final", "errors": errors, "pending": pending}


# --------------------------------------------------------------------------- rendering

def _md(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return ", ".join(_md(x) for x in v) if v else "—"
    if isinstance(v, dict):
        return "`" + canonical_json(v) + "`"
    s = str(v).replace("|", "\\|").replace("\n", " ")
    return s


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_md(x) for x in r) + " |")
    return out


def render_cluster(c: Dict[str, Any], full: bool) -> List[str]:
    j, d = c["judgment"], c["derived"]
    bar = d["evidence_bar"]
    cl = j["cluster"]
    head = (f"Status `{c['status']}` · outcome `{j.get('outcome_type')}` · confidence `{j.get('confidence')}` · "
            f"items {_md(['`' + i + '`' for i in j['implicated_items']])}")
    if j.get("deferral_reason"):
        head += f" · deferral `{j['deferral_reason']}`"
    L = [f"### {c['proposal_id']} — {c['title']}", "", head, "",
         f"**Claim under test.** {_md(cl.get('claim_under_test'))}", "",
         f"**Cluster.** subject: {_md(cl.get('physical_subject'))}; mechanism: {_md(cl.get('visible_state_or_mechanism'))}; "
         f"kinds: {_md(cl.get('upstream_kinds'))}; scenes: {_md(cl.get('scenes'))}; "
         f"commitments: {_md(cl.get('unsupported_commitment_types'))}. {_md(cl.get('merge_split_rationale'))}", ""]
    if cl.get("reshape"):
        r = cl["reshape"]
        L += [f"**Reshape.** from {_md(r.get('source_cluster_ids'))} ({_md(r.get('claim_relation'))}): {_md(r.get('rationale'))}", ""]
    if cl.get("mismatch_class"):
        L += [f"**Mismatch class.** `{cl['mismatch_class']}`; differing dimensions {_md(cl.get('equivalence_dimensions'))}.", ""]
    L += _table(("unit", "class", "role", "property", "photos", "adjudication", "review row", "Terra", "human truth"),
                ((u["unit_key"], u["evidence_class"], u["role"], u["property_key"],
                  ", ".join(p["photo_key"] for p in u["photos"]["photos"]) or u["photos"]["status"],
                  u["photo_review"]["adjudication"], u.get("review_row_id"),
                  (u["lineage"]["cases"][0]["terra_verdict"] if u["lineage"]["cases"] else
                   (u["lineage"]["leads"][0]["stored_verdict"] if u["lineage"]["leads"] else None)),
                  ((u["lineage"]["cases"][0].get("human_truth") or {}).get("claim") if u["lineage"]["cases"] else
                   ((u["lineage"].get("gold") or {}).get("matching_decision")))) for u in d["units"]))
    transferred = [u for u in d["units"] if u.get("transfer_note")]
    if transferred:
        L += ["", "**Transferred units.** " + " ".join(f"{u['unit_key']} ← `{u['review_row_id']}`: {_md(u['transfer_note'])}"
                                                      for u in transferred)]
    ebc = j.get("evidence_bar_claim") or {}
    L += ["", f"**Evidence bar.** recomputed `{bar['bar_met_by']}` (claimed `{bar['claimed_bar_met_by']}`): "
          f"{bar['human_units']} human/gold, {bar['lead_units']} lead, {bar['distinct_properties']} properties, "
          f"{bar['correlated_support']} correlated, corroboration `{bar['corroboration_type']}`, contextual {_md(bar['contextual_units'])}; "
          f"reviewed refutations {bar.get('reviewed_refutations')} (lead {bar.get('lead_refutations')}), mixed signal `{bar.get('mixed_signal')}`. "
          f"{_md(ebc.get('justification'))}"
          + (f" Correlation: {_md(ebc.get('correlation_note'))}" if ebc.get("correlation_note") else "")
          + (f" Conflict: {_md(ebc.get('conflict_note'))}" if ebc.get("conflict_note") else ""), ""]
    ctl = d["controls"]
    src = control_source(c["proposal_id"], j)
    L += [f"**Controls.** {len(ctl)} ({sum(1 for x in ctl if x['required'])} required"
          + (f", rows from {src}" if src != c["proposal_id"] else "")
          + f", missing {sum(1 for x in ctl if x.get('adjudication') == 'missing')}"
          + f", unreviewed family cards {len(d.get('controls_dropped') or [])}): "
          + (", ".join(f"{x['card_id']} {x['role']}→{x.get('adjudication')}" for x in ctl if x["required"]) or "none"), ""]
    if d.get("target_item_controls"):
        L += ["**Target-item controls.** " + ", ".join(
            f"{x['item_id']}:{x['card_id']} {x['role']}→{x.get('adjudication') if x.get('reviewed') else 'unreviewed'}"
            for x in d["target_item_controls"]), ""]
    L += [f"**Absence statements.** {_md(j.get('absence_statements'))}", ""]
    if j.get("human_inputs"):
        L += ["**Human inputs.**", ""]
        L += _table(("ref", "source", "ruling", "effect"),
                    ((h.get("ref"), h.get("source"), h.get("ruling"), h.get("effect")) for h in j["human_inputs"]))
        L += [""]
    if full:
        diag = j.get("diagnosis") or {}
        L += [f"**Diagnosis.** primary `{diag.get('primary_cause')}`; contributing {_md(diag.get('contributing_causes'))}. "
              f"{_md(diag.get('catalog_ownership_argument'))}", ""]
        L += _table(("rejected owner", "why"), ((x.get("owner"), x.get("why")) for x in diag.get("rejected_alternative_owners") or []))
        L += ["", "**Alternative-stage analysis.**", ""]
        L += _table(("question", "answer"), ((q, (j.get("alternative_stage_analysis") or {}).get(q)) for q in STAGE_QUESTIONS))
        prop = j.get("proposal") or {}
        run = d.get("dry_run")
        if d.get("ops"):
            L += ["", "**Decisions diff.**", ""]
            L += _table(("legacy id", "path", "op", "before", "after", "class"),
                        ((o["legacy_id"], o["path"], o["op"], o.get("before"), o.get("after"),
                          next((x["class"] for x in (run or {}).get("classification", []) if x["path"] == o["path"]), None)) for o in d["ops"]))
            if run:
                L += ["", f"Dry run: ok `{run.get('ok')}`; error {_md(run.get('error'))}; added {_md(run.get('added'))}; "
                      f"removed {_md(run.get('removed'))}; changed {_md(sorted((run.get('item_diff') or {}).keys()))}; "
                      f"order ok `{run.get('order_ok')}`; metadata {_md(run.get('metadata_problems'))}; "
                      f"validator errors {_md((run.get('catalog_validation') or {}).get('errors'))} / {_md((run.get('manifest_validation') or {}).get('errors'))}"]
        demo = prop.get("structural_exception_demonstration")
        if isinstance(demo, dict):
            L += ["", f"**Structural exception.** {_md(demo.get('fact'))} — items {_md(demo.get('item_ids'))}; fields {_md(demo.get('fields'))}."]
        if prop.get("regression_controls"):
            L += ["", "**Regression controls.**", ""]
            adj = {x["card_id"]: x for x in (d.get("controls") or []) + (d.get("target_item_controls") or [])}
            L += _table(("card", "role", "item", "adjudicated", "expected after", "note"),
                        ((r.get("card_id"), (adj.get(r.get("card_id")) or {}).get("role"), (adj.get(r.get("card_id")) or {}).get("item_id"),
                          (adj.get(r.get("card_id")) or {}).get("adjudication"), r.get("expected_after"), r.get("note"))
                         for r in prop["regression_controls"]))
        for key in ("semantic_before_after", "retrieval_effects", "economics_fit", "package_route_effects",
                    "product_policy", "expected_benefit", "regression_risk", "gap_description"):
            if prop.get(key):
                L += ["", f"**{key.replace('_', ' ').capitalize()}.** {_md(prop[key])}"]
        if j.get("pass_2d_follow_ups"):
            L += ["", "**Post-catalog Pass 2d follow-ups.**", ""]
            L += _table(("case", "selected", "better item", "rank", "family", "note"),
                        ((f.get("case_ref"), f.get("selected_item"), f.get("better_item"), f.get("better_rank"),
                          f.get("confusion_family"), f.get("note")) for f in j["pass_2d_follow_ups"]))
        L += ["", f"**Unresolved questions.** {_md(j.get('unresolved_questions'))}", "",
              "**Human disposition.** _blank_ (approved / approved_with_modification / rejected / deferred / reclassified_non_catalog)", ""]
    return L


def render(p: Dict[str, Any]) -> str:
    d = p["derived"]
    ledger = d.get("id_ledger") or {}
    L = [f"# Catalog audit proposals — Session 2 ({p['proposal_date']})", "",
         f"Schema {p['schema_version']} · evidence `{EVIDENCE_SHA256[:12]}…` (fingerprint `{EVIDENCE_FINGERPRINT[:12]}…`) · "
         f"proposal baseline `{d['baseline']['proposal_baseline_commit'][:7]}` (catalog 3.2 at `{(d['baseline']['generated_catalog_commit'] or '')[:7]}`) · "
         f"evidence era `{d['baseline']['evidence_era']['commit'][:7]}` (catalog 3.1, checkout sha `{d['baseline']['evidence_era']['sha256_crlf'][:12]}…`).", "",
         f"Review packet `{_md(d['review']['packet_file_sha256'])}` · results `{_md(d['review']['results_sha256'])}` "
         f"(reviewer {_md(d['review']['reviewer'])}; packet drift `{d['review']['packet_drift']}`; pins verified `{d['review'].get('pins_verified')}`).", "",
         f"Validation `{d['validation']['mode']}`: ok `{d['validation']['ok']}`, {len(d['validation']['errors'])} errors, "
         f"{len(d['validation']['pending'])} pending.", "",
         "Counts by outcome: " + _md(d["counts"]["by_outcome"]) + "; by evidence bar: " + _md(d["counts"]["by_bar"]) +
         "; by mismatch class: " + _md(d["counts"].get("by_mismatch_class")) + "; by deferral reason: " + _md(d["counts"].get("by_deferral_reason")) +
         "; coverage triage: " + _md(d["counts"]["coverage_by_triage"]) + ".", "",
         f"Id ledger: {len(ledger.get('known_ids') or [])} known (max {_md(ledger.get('max_known'))}); new {_md(ledger.get('new_ids'))}.", "",
         "Policy in force:", ""]
    L += [f"- **{k}.** {_md(v)}" for k, v in sorted(p["policy"].items())] + [""]
    groups = [("Native decisions proposals", "native_decisions_proposal"), ("Non-catalog actions", "non_catalog_action"),
              ("Migration-system gaps", "migration_system_gap")]
    for title, outcome in groups:
        L += [f"## {title}", ""]
        found = [c for c in p["clusters"] if c["judgment"].get("outcome_type") == outcome]
        L += sum((render_cluster(c, True) for c in found), []) or ["_none_", ""]
    L += ["## Appendix A — no-change clusters", ""]
    L += sum((render_cluster(c, True) for c in p["clusters"] if c["judgment"].get("outcome_type") == "no_change"), []) or ["_none_", ""]
    L += ["## Appendix B — deferred and provisional clusters", ""]
    L += sum((render_cluster(c, True) for c in p["clusters"] if c["judgment"].get("outcome_type") in (DEFERRED, None)), []) or ["_none_", ""]
    L += ["## Appendix C — coverage triage", ""]
    L += _table(("unit", "finding", "matching", "attribution", "adjudication", "triage", "ref", "note"),
                ((t["unit_key"], t["derived"]["finding"], t["derived"]["matching_decision"],
                  ((t["derived"].get("attribution") or {}).get("attribution"), (t["derived"].get("attribution") or {}).get("stage")),
                  t["derived"]["adjudication"], t["judgment"].get("triage_class"), t["judgment"].get("ref"),
                  t["judgment"].get("note") or t["judgment"].get("not_promoted_reason")) for t in p["coverage_triage"]))
    L += ["", "## Appendix D — source fingerprints", ""]
    L += _table(("source", "sha256"), sorted(d["baseline"]["sources"].items()))
    L += ["", f"Bundle git head `{d['baseline']['bundle_git_head']}`; evidence bundle sha256 `{EVIDENCE_SHA256}`; "
          f"fingerprint `{EVIDENCE_FINGERPRINT}`; packet `{_md(d['review']['packet_file_sha256'])}`; "
          f"results `{_md(d['review']['results_sha256'])}` at {_md(d['review']['results_path'])}; "
          f"pinned review `{REVIEW_SHA256}` / packet `{PACKET_SHA256}`.", "",
          "## Appendix E — item-family matrix", ""]
    L += _table(("proposal", "item", "kind", "migration", "pos. uses", "agree", "correct rej.", "notes", "2a halluc.", "family", "frozen neighbours"),
                ((c["proposal_id"], i, (c["derived"]["item_semantics"][i].get("proposal_baseline") or {}).get("kind"),
                  ((c["derived"]["per_item"][i].get("migration") or {}).get("change_type")),
                  c["derived"]["per_item"][i]["positive_uses"], c["derived"]["per_item"][i]["agreements"],
                  c["derived"]["per_item"][i]["correct_rejections"], c["derived"]["per_item"][i]["notes"],
                  c["derived"]["per_item"][i]["hallucination_annotations"], len(c["derived"]["family_members"]),
                  ", ".join(n["item_id"] for n in c["derived"]["frozen_candidate_neighbors"][:5]))
                 for c in p["clusters"] for i in c["judgment"]["implicated_items"]))
    L += ["", "## Appendix F — validation matrix", ""]
    L += _table(("proposal", "outcome", "mismatch", "bar (recomputed)", "bar (claimed)", "consistent", "adjudications", "refutations", "mixed", "dry run", "quarantined"),
                ((c["proposal_id"], c["judgment"].get("outcome_type"), (c["judgment"].get("cluster") or {}).get("mismatch_class"),
                  c["derived"]["evidence_bar"]["bar_met_by"],
                  c["derived"]["evidence_bar"]["claimed_bar_met_by"], c["derived"]["evidence_bar"]["consistent_with_claim"],
                  c["derived"]["adjudication_counts"],
                  f"{c['derived']['evidence_bar'].get('reviewed_refutations')}/{c['derived']['evidence_bar'].get('lead_refutations')}",
                  c["derived"]["evidence_bar"].get("mixed_signal"), (c["derived"].get("dry_run") or {}).get("ok"),
                  c["derived"]["trade_quarantined"]) for c in p["clusters"]))
    L += ["", "Errors: " + (_md(d["validation"]["errors"]) if d["validation"]["errors"] else "none") + ".",
          "Pending: " + (_md(d["validation"]["pending"]) if d["validation"]["pending"] else "none") + ".", ""]
    L += ["## Appendix G — post-catalog Pass 2d and retrieval worklist by confusion family", ""]
    fam = d.get("pass_2d_follow_ups") or {}
    if fam:
        for family, entries in fam.items():
            L += [f"### {family}", ""]
            L += _table(("proposal", "case", "selected", "better item", "rank", "note"),
                        ((e.get("proposal_id"), e.get("case_ref"), e.get("selected_item"), e.get("better_item"),
                          (e.get("better_rank") if e.get("better_rank") is not None else "not in pool"), e.get("note"))
                         for e in entries))
            L += [""]
    else:
        L += ["_none_", ""]
    L += ["## Appendix H — Pass 2a regression-control register", ""]
    reg = d.get("regression_register") or []
    L += (_table(("proposal", "outcome", "card", "role", "item", "adjudicated", "expected after", "note"),
                 ((r["proposal_id"], r["outcome"], r["card_id"], r["role"], r["item_id"], r["adjudication"], r["expected_after"], r["note"])
                  for r in reg)) if reg else ["_none_"]) + [""]
    return "\n".join(L)


# --------------------------------------------------------------------------- main

def proposal_md_path(p: Dict[str, Any]) -> Path:
    return ROOT / "docs" / f"PROPOSAL_catalog_audit_{p['proposal_date'].replace('-', '')}.md"


def build(proposals_path: Path = PROPOSALS_JSON, review_path: Optional[Path] = None,
          packet_path: Path = PACKET_JSON, known_ids: Optional[Sequence[str]] = KNOWN_IDS) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    rp = review_path or (REVIEW_JSON if REVIEW_JSON.is_file() else None)
    if rp is not None:
        verify_pins(rp, packet_path)
    bundle = load_bundle()
    baseline = load_baseline(bundle)
    ev = Evidence(bundle, latest_verdicts(LEDGER))
    if not proposals_path.is_file():
        fail(f"missing proposals file {proposals_path}")
    proposals = _read_json(proposals_path)
    photo_root = Path(proposals["review"]["photo_root"])
    packet = build_packet(proposals, ev, photo_root)
    packet_fresh = sha256_canonical(packet)
    packet_on_disk = sha256_canonical(_read_json(packet_path)) if packet_path.is_file() else None
    review = None
    if rp is not None:
        review = load_review(rp, packet_path)
    out = derive(proposals, ev, baseline, review, packet_on_disk, packet_fresh, known_ids=known_ids)
    if packet_path.is_file():
        out["derived"]["review"]["packet_file_sha256"] = sha256_file(packet_path)
    return out, packet, render(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="derive in memory and compare with disk; write nothing")
    parser.add_argument("--write-current-surface", action="store_true",
                        help="write the versioned current authoring-surface record and exit")
    parser.add_argument("--packet", action="store_true", help="write the photo-review packet (refused once the pinned review exists)")
    parser.add_argument("--review", type=Path, default=None, help="review results file (must be the pinned path)")
    args = parser.parse_args(argv)
    if args.write_current_surface:
        record = current_surface_record()
        atomic_json(CURRENT_SURFACE_JSON, record)
        print(f"wrote {_rel(CURRENT_SURFACE_JSON)} "
              f"carryover_override_fields={record['surface']['wording_override_fields']}")
        return 0
    if args.packet and REVIEW_JSON.is_file():
        fail(f"--packet refused: pinned review {_rel(REVIEW_JSON)} exists on disk; regenerating the packet would orphan it "
             f"(move the review deliberately first)")
    if args.review is not None and Path(args.review).resolve() != REVIEW_JSON.resolve():
        fail(f"--review must point at the pinned review {_rel(REVIEW_JSON)}; got {args.review}")
    watched = tuple(p for p in GUARDED + PINNED_INPUTS if not (args.packet and p == PACKET_JSON))
    before = guard_snapshot(watched)
    out, packet, md = build(review_path=args.review)
    md_path = proposal_md_path(out)
    v = out["derived"]["validation"]
    rc_ = 0
    if args.packet:
        atomic_json(PACKET_JSON, packet)
        print(f"wrote {PACKET_JSON} sha256 {sha256_file(PACKET_JSON)} rows {packet['counts']}")
    elif args.check:
        existing = _read_json(PROPOSALS_JSON) if PROPOSALS_JSON.is_file() else None
        same_json = existing is not None and canonical_json(existing) == canonical_json(out)
        same_md = md_path.is_file() and md_path.read_text(encoding="utf-8") == md
        print(f"proposals json {'matches' if same_json else 'DIFFERS'}; markdown {'matches' if same_md else 'DIFFERS'}; "
              f"validation {v['mode']} ok={v['ok']} errors={len(v['errors'])} pending={len(v['pending'])}")
        rc_ = 0 if (same_json and same_md and v["ok"]) else 1
    else:
        atomic_json(PROPOSALS_JSON, out)
        md_path.write_text(md, encoding="utf-8")
        print(f"wrote {PROPOSALS_JSON} and {md_path}; validation {v['mode']} ok={v['ok']} errors={len(v['errors'])} pending={len(v['pending'])}")
        for e in v["errors"][:40]:
            print("  error:", e)
        rc_ = 0 if v["ok"] else 1
    after = guard_snapshot(watched)
    if after != before:
        fail("guarded files changed during the run: " + ", ".join(sorted(k for k in before if before[k] != after[k])))
    return rc_


if __name__ == "__main__":
    raise SystemExit(main())
