"""Evidence foundation for the catalog audit (Session 1).

Freezes provenance, reconciles the evidence-era catalog (3.1, the state every
frozen run artifact hashes) against the proposal baseline (3.2 on disk), joins
every frozen human/gold/model source on run-scoped keys, seeds a neutral
worklist by mechanical rules, deduplicates it into independent evidence units,
and builds the mechanical part of item families. Writes

  reports/catalog_audit_evidence.json   the bundle Session 2 audits from
  reports/catalog_audit_evidence.md     compact rendering of the same dict

Nothing here judges a catalog item: every seed carries the rule that included
it and no evidence-bar verdict is computed. Zero provider calls, nothing
written outside reports/, and read-only git is the only thing consulted beyond
the pinned files (it materializes the evidence-era catalog). A frozen input
whose hash drifted fails the run before anything is built.

Run:
  .venv\\Scripts\\python.exe scripts\\build_catalog_audit_evidence.py [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.error_attribution_report import latest_verdicts  # noqa: E402
from tools import review_cards as rc  # noqa: E402
from tools.catalog_embeddings import CatalogEmbeddingsRetriever  # noqa: E402
from tools.catalog_validation import ECONOMIC_FIELDS  # noqa: E402
from tools.comparison_common import atomic_json, sha256_bytes, sha256_canonical, sha256_file  # noqa: E402
from tools.renovation_architecture.terra_review import _claim_text  # noqa: E402
from tools.renovation_estimate import product_quarantined_trade_buckets  # noqa: E402

SCHEMA_VERSION = 1
OUT_JSON = ROOT / "reports" / "catalog_audit_evidence.json"
OUT_MD = ROOT / "reports" / "catalog_audit_evidence.md"

# Every file the builder reads, by logical name (repo-relative, posix).
REL = {
    "queue": "reports/error_attribution_queue.json",
    "verdicts": "reports/error_attribution_verdicts.jsonl",
    "gold_cases": "reports/error_attribution_gold_cases.json",
    "labels": "reports/labels_v1_1.json",
    "review_queue": "reports/review_queue.json",
    "review_verdicts": "reports/review_verdicts.jsonl",
    "review_analysis": "reports/review_analysis.json",
    "scorecard": "reports/factorized_review_scorecard.json",
    "gold_reference": "benchmarks/pass2a-prompt/gold/reference.json",
    "catalog_v2": "tools/issue_catalog_kind_v2.json",
    "decisions": "tools/catalog_migrations/kind_v2_decisions.json",
    "catalog_v1": "tools/issue_catalog.json",
    "manifest": "tools/catalog_migrations/2.1_to_3.0.json",
    "generator": "scripts/migrate_catalog_kind_v2.py",
    "validator": "tools/catalog_validation.py",
    "findings": "docs/FINDINGS_catalog_3_2_deferred_issues.md",
    "factorized_manifest": "artifacts_canary/factorized_v1_20260831/manifest.json",
    "builder": "scripts/build_catalog_audit_evidence.py",
}

# Frozen tier: a mismatch is baseline drift and fails the run. Reconciling one
# means editing this table with the reason recorded in the session handoff.
FROZEN_SHA256 = {
    "queue": "b58dcca7c3779f2b0539f988077f865c6c1ca040d23a44646d89b04a398940df",
    "verdicts": "166e8bec20641c0a8fb0ca5ddbd6c042f31b828447304ccbbbd4d585cd2a894e",
    "gold_cases": "77a7b515c594eb862bfc9ffbd0073556bc8a44f2fed644610f2412cb3bd0954a",
    "labels": "7f9b03017195144195550074b49dbb0488ed3ad33fdde905ba4e5867dfdbe3bd",
    "review_queue": "8512b87c2af18d1104069c15a5eda977d5fe79eb736f9aedd976d174f89d318a",
    "review_verdicts": "0c7ca6a8b55d6dac69021a00e81315be1b15cef68ffd59fbe2973cb895a8d70f",
    "review_analysis": "3652ba3c3feb15800c064ffafd076d15cd9773de29c3d77c7b383fe26a259c9a",
    "scorecard": "60300ec5e957e42d2e443ab9493d0b9820b66498a2f57e3ac34a4479fcd9d103",
    "gold_reference": "253075132988ac2a1abbdb6e0815c1bd75e89d9ad982e745ddb9ce5a9f424543",
    "catalog_v2": "51bf7e263ff98109d657ef730b0ce1a705598101f09843eb288b1bdc3e0eaa54",
    "decisions": "47614d822bdd6b672d8596050b46839d04a8a6df0a04c18e052f4a63bd1903e8",
}

MISS_LANES = ("miss_label", "miss_v1only", "miss_gold")
HALLUC_LANES = ("halluc_label", "halluc_v1only", "halluc_gold_extra")

RULES = {
    "R1": "latest effective attribution is downstream at stage 2d",
    "R2": "queue lane appendix_misnamed (human: claim misnamed, work warranted)",
    "R3": "miss lane with latest attribution downstream at stage terra",
    "R4": "factorized verifier answered claim_accurate_as_written=no (model lead, not truth)",
    "R5": "gold finding with no matching v5 condition (gold miss candidate)",
    "R6": "gold finding the review judged outside the catalog (open coverage question)",
    "R7": "previously deferred catalog finding (backlog/constraint reference, not evidence)",
}
PRECEDENCE = {"case": 0, "gold_case": 1, "factorized_lead": 2, "gold_row": 3, "deferred_finding": 4}
METHOD = {"case": "human_review", "gold_case": "gold_reference", "factorized_lead": "model_judge",
          "gold_row": "gold_reference", "deferred_finding": "backlog"}

SEMANTIC_FIELDS = ("kind", "name", "severity", "atomic_claim", "description", "embed_text",
                   "support_any", "deny_any", "require_any", "scene_groups", "drop_if_generic",
                   "defaultHidden", "route_override", "trade_bucket", "tier", "scope", "category",
                   "display_class")
DERIVED_FIELDS = ("claim_text", "embed_source_text")
COMPARED_FIELDS = SEMANTIC_FIELDS + tuple(ECONOMIC_FIELDS) + DERIVED_FIELDS

RETRIEVAL_UNAVAILABLE = ("no offline embedding store exists; tools/catalog_embeddings.py builds vectors "
                         "at construction against the live sidecar, which this session may not call")


class EvidenceError(SystemExit):
    """Fail closed: drifted input, unresolvable baseline, or a seed lost in reconciliation."""


def fail_closed(problems: Sequence[str]) -> None:
    if problems:
        raise EvidenceError("build_catalog_audit_evidence: " + "; ".join(problems))


# --------------------------------------------------------------------------- keys

def runtime_key(source: Any, property_key: Any, run_id: Any, condition_id: Any) -> str:
    return f"runtime:{source}:{property_key}:{run_id}:{condition_id}"


def gold_key(property_key: Any, photo_key: Any, gold_id: Any) -> str:
    return f"gold:{property_key}:{photo_key}:{gold_id}"


def run_key(source: Any, property_key: Any, run_id: Any) -> str:
    return f"{source}:{property_key}:{run_id}"


def case_unit_key(case: Dict[str, Any]) -> Optional[str]:
    """Runtime evidence unit: (source, property_key, run_id, condition_id) or None."""
    claim = case.get("v5_claim") or {}
    ref = case.get("run_ref") or {}
    if not claim.get("condition_id"):
        return None
    return runtime_key(ref.get("source"), ref.get("property_key"), ref.get("run_id"), claim["condition_id"])


def gold_unit_key(case: Dict[str, Any]) -> str:
    truth = case.get("human_truth") or {}
    property_key, photo_key = str(truth.get("photo") or "/").split("/", 1)
    return gold_key(property_key, photo_key, truth.get("gold_id"))


# --------------------------------------------------------------------------- provenance

def verify_pins(actual: Dict[str, Optional[str]], expected: Dict[str, str]) -> List[str]:
    """Frozen-tier check: every expected name must be present and byte-identical."""
    problems = []
    for name, want in sorted(expected.items()):
        got = actual.get(name)
        if got is None:
            problems.append(f"{name}: missing")
        elif got != want:
            problems.append(f"{name}: sha256 {got} != pinned {want}")
    return problems


def to_crlf(blob: bytes) -> bytes:
    return re.sub(rb"(?<!\r)\n", b"\r\n", blob)


def resolve_catalog_identity(targets: Iterable[str], history: Sequence[Tuple[str, bytes]],
                             *, head_blob: bytes, on_disk: bytes) -> Dict[str, Any]:
    """Which git state of the generated catalog do the run artifacts hash?

    `history` is newest-first (commit, raw blob). Each blob is hashed as stored
    (LF) and as a CRLF checkout, because catalog_projection hashes the on-disk
    bytes and this checkout converts line endings. Exactly one distinct state
    must match or the evidence baseline is unresolvable."""
    targets = set(targets)
    candidates = []
    for commit, blob in history:
        try:
            version = json.loads(blob.decode("utf-8")).get("version")
        except (ValueError, UnicodeDecodeError):
            version = None
        lf, crlf = sha256_bytes(blob), sha256_bytes(to_crlf(blob))
        matched = [name for name, digest in (("lf", lf), ("crlf", crlf)) if digest in targets]
        candidates.append({"commit": commit, "version": version, "sha256_lf": lf,
                           "sha256_crlf": crlf, "matched_conventions": matched})
    hits = [c for c in candidates if c["matched_conventions"]]
    states = {(c["sha256_lf"], c["sha256_crlf"]) for c in hits}
    if len(states) != 1:
        fail_closed([f"artifact catalog_sha256 {sorted(targets)} matched {len(states)} distinct "
                     f"catalog states in git history (need exactly 1)"])
    era = hits[0]  # newest matching commit; duplicates (same blob) listed alongside
    on_disk_sha = sha256_bytes(on_disk)
    convention = "crlf" if b"\r\n" in on_disk else "lf"
    matches_head = on_disk_sha in (sha256_bytes(head_blob), sha256_bytes(to_crlf(head_blob)))
    return {
        "artifact_catalog_sha256": sorted(targets),
        "checkout_line_ending_convention": convention,
        "evidence_era": {"commit": era["commit"], "version": era["version"],
                         "sha256_lf": era["sha256_lf"], "sha256_crlf": era["sha256_crlf"],
                         "matched_convention": era["matched_conventions"][0],
                         "matching_commits": [c["commit"] for c in hits]},
        "proposal_baseline": {"catalog_commit": history[0][0] if history else None,
                              "version": candidates[0]["version"] if candidates else None,
                              "sha256_lf": sha256_bytes(head_blob),
                              "sha256_crlf": sha256_bytes(to_crlf(head_blob)),
                              "on_disk_sha256": on_disk_sha,
                              "working_tree_matches_head": matches_head},
        "history": candidates,
    }


# --------------------------------------------------------------------------- catalogs

def claim_text(item: Dict[str, Any]) -> str:
    """The claim Terra is shown for this item (atomic_claim subject + state)."""
    return _claim_text(item, str(item.get("id")))


def embed_source_text(item: Dict[str, Any]) -> str:
    # _catalog_text reads only its `it` argument; the unbound call keeps the
    # retriever (and its sidecar) out of an offline script.
    return CatalogEmbeddingsRetriever._catalog_text(None, item)  # type: ignore[arg-type]


def item_semantics(item: Dict[str, Any]) -> Dict[str, Any]:
    out = {field: item.get(field) for field in SEMANTIC_FIELDS + tuple(ECONOMIC_FIELDS)}
    out["claim_text"] = claim_text(item)
    out["embed_source_text"] = embed_source_text(item)
    out["pricing_status"] = item.get("pricing_status")
    return out


def _items(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {it["id"]: it for it in catalog.get("items") or [] if isinstance(it, dict) and it.get("id")}


def compare_catalogs(era: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Field-level 3.1 -> 3.2 comparison; the two headline booleans are computed, never assumed."""
    era_items, cur_items = _items(era), _items(current)
    era_ids, cur_ids = list(era_items), list(cur_items)
    shared = [i for i in cur_ids if i in era_items]
    counts: Counter = Counter()
    changed = []
    for item_id in shared:
        before, after = item_semantics(era_items[item_id]), item_semantics(cur_items[item_id])
        diff = {f: {"evidence_era": before[f], "proposal_baseline": after[f]}
                for f in COMPARED_FIELDS if before[f] != after[f]}
        if diff:
            counts.update(list(diff))
            changed.append({"id": item_id, "changes": diff})
    markers = []
    for item_id in shared:
        before = (era_items[item_id].get("package_affinity") or {})
        for room, entry in sorted((cur_items[item_id].get("package_affinity") or {}).items()):
            if isinstance(entry, dict) and entry.get("repair_support_when_driven") \
                    and not (before.get(room) or {}).get("repair_support_when_driven"):
                markers.append({"id": item_id, "room": room, "package_type": entry.get("package_type")})
    return {
        "fields_compared": list(COMPARED_FIELDS),
        "evidence_era": {"version": era.get("version"), "item_count": len(era_ids)},
        "proposal_baseline": {"version": current.get("version"), "item_count": len(cur_ids)},
        "added_ids": sorted(set(cur_ids) - set(era_ids)),
        "removed_ids": sorted(set(era_ids) - set(cur_ids)),
        "order_changed": [i for i in era_ids if i in cur_items] != shared,
        "trade_buckets_changed": (era.get("trade_buckets") or []) != (current.get("trade_buckets") or []),
        "per_field_change_counts": dict(sorted(counts.items())),
        "changed_items": changed,
        "atomic_claim_unchanged": counts.get("atomic_claim", 0) == 0 and set(cur_ids) == set(era_ids),
        "claim_text_unchanged": counts.get("claim_text", 0) == 0,
        "repair_support_markers_added": markers,
        "product_quarantined_trades": {"evidence_era": sorted(product_quarantined_trade_buckets(era)),
                                       "proposal_baseline": sorted(product_quarantined_trade_buckets(current))},
    }


def migration_index(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Parent/successor/sibling relationships; generated items carry no parent pointer."""
    parent_of: Dict[str, str] = {}
    successors: Dict[str, List[str]] = {}
    change_type: Dict[str, str] = {}
    legacy_kind: Dict[str, Any] = {}
    for entry in manifest.get("entries") or []:
        legacy = entry["legacy_id"]
        ids = [s["id"] for s in entry.get("successors") or []]
        successors[legacy] = ids
        change_type[legacy] = entry.get("change_type")
        legacy_kind[legacy] = entry.get("legacy_kind")
        for item_id in ids:
            parent_of[item_id] = legacy
    siblings = {item_id: [s for s in successors[legacy] if s != item_id]
                for item_id, legacy in parent_of.items()}
    return {"parent_of": dict(sorted(parent_of.items())), "successors": dict(sorted(successors.items())),
            "siblings": dict(sorted(siblings.items())), "change_type": dict(sorted(change_type.items())),
            "legacy_kind": dict(sorted(legacy_kind.items()))}


def extract_findings_refs(text: str, item_ids: Iterable[str]) -> List[Dict[str, Any]]:
    """`## N. title` sections of the deferred-findings doc; backticked tokens that
    are catalog item ids (either era) become item_refs, everything else stays a mention."""
    known = set(item_ids)
    sections: List[Dict[str, Any]] = []
    for line in text.splitlines():
        head = re.match(r"^## (\d+)\.\s+(.*)$", line)
        if head:
            sections.append({"section": int(head.group(1)), "title": head.group(2).strip(), "_text": line})
        elif sections:
            sections[-1]["_text"] += "\n" + line
    out = []
    for sec in sections:
        tokens = sorted(set(re.findall(r"`([a-z][a-z0-9_]*)`", sec.pop("_text"))))
        out.append({**sec, "mentions": tokens, "item_refs": [t for t in tokens if t in known]})
    return sorted(out, key=lambda s: s["section"])


# --------------------------------------------------------------------------- cases

def attribution_of(rec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rec:
        return None
    return {"attribution": rec.get("attribution"), "stage": rec.get("first_responsible_stage"),
            "confidence": rec.get("confidence"), "reviewer": rec.get("reviewer"),
            "revised": bool(rec.get("revision_of")), "gold_match": rec.get("gold_match")}


def case_resolved_items(case: Dict[str, Any]) -> Dict[str, Any]:
    """The authoritative item mapping: the projected condition's item plus every
    per-issue `p2d.resolved_item_id`. `p2c_surviving[].catalog_item_id` is never read."""
    claim = case.get("v5_claim") or {}
    issues = ((case.get("lineage") or {}).get("per_issue")) or []
    resolved = sorted({(i.get("p2d") or {}).get("resolved_item_id") for i in issues
                       if (i.get("p2d") or {}).get("resolved_item_id")})
    return {"condition_item": claim.get("catalog_item_id"), "issue_items": resolved}


def case_record(case: Dict[str, Any], verdict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ref, claim = case.get("run_ref") or {}, case.get("v5_claim") or {}
    lineage, truth = case.get("lineage") or {}, case.get("human_truth") or {}
    items = case_resolved_items(case)
    issues = sorted(({"issue_id": i.get("issue_id"), "photo_key": i.get("photo_key"),
                      "observation": i.get("observation"),
                      "resolved_item_id": (i.get("p2d") or {}).get("resolved_item_id"),
                      "resolution_path": (i.get("p2d") or {}).get("resolution_path"),
                      "shortcut_reason": (i.get("p2d") or {}).get("shortcut_reason"),
                      "p2b_join": (i.get("p2b_join") or {}).get("method"),
                      "p2e_status": i.get("p2e_status"),
                      "projected_condition_id": i.get("projected_condition_id")}
                     for i in lineage.get("per_issue") or []),
                    key=lambda r: (str(r["photo_key"]), str(r["issue_id"])))
    return {
        "case_id": case["case_id"], "lane": case.get("lane"), "attribute": bool(case.get("attribute")),
        "status": case.get("status"), "untraceable_reason": case.get("untraceable_reason"),
        "human_truth": {k: truth.get(k) for k in ("basis", "class_v1_1", "slug", "claim", "work",
                                                   "note", "arm", "v1_verdict")},
        "attribution": attribution_of(verdict),
        "source": ref.get("source"), "property_key": ref.get("property_key"), "run_id": ref.get("run_id"),
        "artifact_pinned": bool(ref.get("artifact_sha256")),
        "condition_id": claim.get("condition_id") or None,
        "condition_item": items["condition_item"], "issue_items": items["issue_items"],
        "catalog_kind": claim.get("catalog_kind"), "scene_group": claim.get("scene_group"),
        "estimate_unit_id": claim.get("estimate_unit_id"), "claim_text": claim.get("claim_text"),
        "terra_verdict": claim.get("terra_verdict"), "terra_rationale": claim.get("terra_rationale"),
        "disposition": claim.get("disposition"), "reason_code": claim.get("reason_code"),
        "accepted": claim.get("accepted"),
        "photo_keys": sorted(lineage.get("photo_keys") or []), "issues": issues,
        "unit_key": case_unit_key(case),
        "card_id": (case.get("source_ids") or {}).get("card_id"),
        "factorized_flags": sorted((case.get("context") or {}).get("in_factorized_disagreements") or []),
    }


def gold_case_record(case: Dict[str, Any], verdict: Optional[Dict[str, Any]],
                     row: Optional[Dict[str, Any]], cases_by_unit: Dict[str, str],
                     cases: Dict[str, Dict[str, Any]],
                     artifacts: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The covering rejected condition (when any) names the implicated item: through
    its queue case when one exists, else through the hash-verified run artifact."""
    ref, truth, claim = case.get("run_ref") or {}, case.get("human_truth") or {}, case.get("v5_claim") or {}
    property_key, photo_key = str(truth.get("photo") or "/").split("/", 1)
    covering = claim.get("covering_rejected_condition_id")
    covering_key = runtime_key(ref.get("source"), property_key, ref.get("run_id"), covering) if covering else None
    covering_case = cases_by_unit.get(covering_key) if covering_key else None
    covering_item, item_source = None, None
    if covering_case:
        covering_item, item_source = cases[covering_case].get("condition_item"), "queue_case"
    elif covering:
        art = (artifacts or {}).get(run_key(ref.get("source"), property_key, ref.get("run_id"))) or {}
        cond = (art.get("conditions") or {}).get(covering) if art.get("verified") else None
        if cond:
            covering_item, item_source = cond.get("catalog_item_id"), "artifact_observed_conditions"
    return {
        "case_id": case["case_id"], "lane": case.get("lane"), "attribute": bool(case.get("attribute")),
        "status": case.get("status"), "basis": truth.get("basis"),
        "gold_id": truth.get("gold_id"), "finding": truth.get("finding"), "photo": truth.get("photo"),
        "property_key": property_key, "photo_key": photo_key,
        "source": ref.get("source"), "run_id": ref.get("run_id"),
        "artifact_pinned": bool(ref.get("artifact_sha256")),
        "attribution": attribution_of(verdict),
        "matching_decision": (row or {}).get("decision"), "matching_note": (row or {}).get("note"),
        "covering_rejected_condition_id": covering,
        "covering_unit_key": covering_key, "covering_case_id": covering_case,
        "covering_item": covering_item, "covering_item_source": item_source,
        "unit_key": gold_unit_key(case),
    }


def index_cases(queue: Dict[str, Any], gold_doc: Dict[str, Any], latest: Dict[str, Dict[str, Any]],
                artifacts: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    cases = {c["case_id"]: case_record(c, latest.get(c["case_id"])) for c in queue.get("cases") or []}
    cases_by_unit = {c["unit_key"]: cid for cid, c in sorted(cases.items()) if c["unit_key"]}
    rows_by_case = {r["case_id"]: r for r in gold_doc.get("matching_table") or [] if r.get("case_id")}
    gold_cases = {c["case_id"]: gold_case_record(c, latest.get(c["case_id"]), rows_by_case.get(c["case_id"]),
                                                 cases_by_unit, cases, artifacts)
                  for c in gold_doc.get("cases") or []}
    by_item: Dict[str, Dict[str, List[str]]] = {}
    by_run: Dict[str, List[str]] = {}
    by_photo: Dict[str, Dict[str, List[str]]] = {}
    for cid, c in sorted(cases.items()):
        for item in sorted(set(c["issue_items"]) | ({c["condition_item"]} if c["condition_item"] else set())):
            by_item.setdefault(item, {"cases": [], "gold_cases": []})["cases"].append(cid)
        if c["run_id"]:
            by_run.setdefault(run_key(c["source"], c["property_key"], c["run_id"]), []).append(cid)
        for pk in c["photo_keys"]:
            by_photo.setdefault(f"{c['property_key']}/{pk}", {"cases": [], "gold_cases": []})["cases"].append(cid)
    for gid, g in sorted(gold_cases.items()):
        if g["covering_item"]:
            by_item.setdefault(g["covering_item"], {"cases": [], "gold_cases": []})["gold_cases"].append(gid)
        by_photo.setdefault(g["photo"], {"cases": [], "gold_cases": []})["gold_cases"].append(gid)
    return {"cases": cases, "gold_cases": gold_cases, "cases_by_unit": cases_by_unit,
            "by_item": dict(sorted(by_item.items())), "by_run": dict(sorted(by_run.items())),
            "by_photo": dict(sorted(by_photo.items())),
            "by_condition": {k: v for k, v in sorted(cases_by_unit.items())}}


# --------------------------------------------------------------------------- joins

def card_unit_key(card: Dict[str, Any]) -> Optional[str]:
    meta = card.get("meta") or {}
    if card.get("kind") != "condition" or not meta.get("condition_id"):
        return None
    return runtime_key(card.get("source"), card.get("property_key"), card.get("run_id"), meta["condition_id"])


def join_labels(labels: Dict[str, Dict[str, Any]], cards: Dict[str, Dict[str, Any]],
                cases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every v1.1 label joined to its exact review card (card id is the key on both sides)."""
    out = []
    for card_id, label in sorted(labels.items()):
        card = cards.get(card_id) or {}
        meta = card.get("meta") or {}
        photos = sorted({p.get("key") for strip in card.get("strips") or [] for p in strip.get("photos") or []
                         if p.get("key")})
        out.append({
            "card_id": card_id, "card_present": bool(card),
            "class_v1_1": label.get("class_v1_1"), "slug": label.get("slug"), "claim": label.get("claim"),
            "work": label.get("work"), "note": label.get("note"), "arm": label.get("arm"),
            "source": label.get("source"), "property_key": label.get("property_key"),
            "run_id": card.get("run_id"), "condition_id": label.get("condition_id"),
            "catalog_item_id": label.get("catalog_item_id"), "catalog_kind": label.get("catalog_kind"),
            "item_id_matches_card": (meta.get("catalog_item_id") == label.get("catalog_item_id")) if card else None,
            "card_claim": (card.get("claim") or {}).get("catalog_claim"),
            "card_photo_keys": photos, "terra_verdict": meta.get("terra_verdict"),
            "disposition": meta.get("disposition"), "accepted": meta.get("accepted"),
            "case_id": card_id if card_id in cases else None,
            "unit_key": card_unit_key(card) if card else None,
        })
    return out


def join_notes(notes: Sequence[Dict[str, Any]], cards: Dict[str, Dict[str, Any]],
               cases: Dict[str, Dict[str, Any]], labels: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for note in sorted(notes, key=lambda n: str(n.get("card_id"))):
        card_id = note.get("card_id")
        card = cards.get(card_id) or {}
        case = cases.get(card_id) or {}
        out.append({"card_id": card_id, "theme": note.get("theme"), "tag": note.get("tag"),
                    "title": note.get("title"), "verdict": note.get("verdict"), "note": note.get("note"),
                    "card_present": bool(card), "is_case": card_id in cases, "is_label": card_id in labels,
                    "catalog_item_id": case.get("condition_item") or (card.get("meta") or {}).get("catalog_item_id"),
                    "unit_key": case.get("unit_key") or (card_unit_key(card) if card else None)})
    return out


def join_gold_rows(rows: Sequence[Dict[str, Any]], gold_photos: Dict[str, Dict[str, Any]],
                   cases_by_unit: Dict[str, str]) -> List[Dict[str, Any]]:
    """Matching-table rows with run identity from the queue's gold photo, never from the property alone."""
    out = []
    for row in rows:
        photo = str(row.get("photo") or "/")
        property_key, photo_key = photo.split("/", 1)
        ref = (gold_photos.get(photo) or {}).get("run_ref") or {}
        link = row.get("matched_condition_id") or row.get("condition_id") or row.get("covering_rejected_condition_id")
        unit = runtime_key(ref.get("source"), property_key, ref.get("run_id"), link) if link and ref.get("run_id") else None
        gold_id = row.get("gold_id")
        out.append({"decision": row.get("decision"), "gold_id": gold_id, "finding": row.get("finding"),
                    "photo": photo, "property_key": property_key, "photo_key": photo_key,
                    "source": ref.get("source"), "run_id": ref.get("run_id"),
                    "matched_condition_id": row.get("matched_condition_id"),
                    "covering_rejected_condition_id": row.get("covering_rejected_condition_id"),
                    "gold_incomplete_condition_id": row.get("condition_id"),
                    "photo_supports_claim": row.get("photo_supports_claim"),
                    "case_id": row.get("case_id"), "already_cased": row.get("already_cased"),
                    "note": row.get("note"), "runtime_unit_key": unit,
                    "linked_case_id": cases_by_unit.get(unit) if unit else None,
                    "gold_unit_key": gold_key(property_key, photo_key, gold_id) if gold_id else None,
                    "role": "seed" if row.get("decision") in ("miss_candidate", "out_of_catalog") else "annotation"})
    return sorted(out, key=lambda r: (r["photo"], str(r["gold_id"] or ""), str(r["decision"]),
                                      str(r["gold_incomplete_condition_id"] or "")))


def join_factorized_leads(leads: Sequence[Dict[str, Any]], canary_runs: Dict[str, str],
                          artifacts: Dict[str, Dict[str, Any]], cases_by_unit: Dict[str, str],
                          labels_by_unit: Dict[str, str], cards_by_unit: Dict[str, str]) -> List[Dict[str, Any]]:
    """Leads carry (property_key, condition_id) only. Run identity comes from the
    factorized manifest (canary run_1) resolved per property through the queue; the
    item id comes from the hash-verified artifact. Anything unpinned stays unattached."""
    out = []
    for lead in sorted(leads, key=lambda r: (str(r.get("property_key")), str(r.get("condition_id")))):
        prop, cid = lead.get("property_key"), lead.get("condition_id")
        rec = {"record_id": f"lead:{prop}:{cid}", "property_key": prop, "condition_id": cid,
               "derived_class": lead.get("derived_class"), "observed_description": lead.get("observed_description"),
               "stored_verdict": lead.get("stored_verdict"), "source": "canary", "run_id": canary_runs.get(prop),
               "unit_key": None, "catalog_item_id": None, "item_source": None,
               "joined_case_id": None, "joined_card_id": None, "joined_label_id": None,
               "unattached_reason": None}
        run_id = rec["run_id"]
        art = artifacts.get(run_key("canary", prop, run_id)) if run_id else None
        if not run_id:
            rec["unattached_reason"] = "run_identity_unpinned"
        elif not art or not art.get("verified"):
            rec["unattached_reason"] = "artifact_unverified"
        elif cid not in (art.get("conditions") or {}):
            rec["unattached_reason"] = "condition_absent_from_pinned_run"
        else:
            unit = runtime_key("canary", prop, run_id, cid)
            rec.update(unit_key=unit, catalog_item_id=art["conditions"][cid].get("catalog_item_id"),
                       item_source="artifact_observed_conditions", joined_case_id=cases_by_unit.get(unit),
                       joined_card_id=cards_by_unit.get(unit), joined_label_id=labels_by_unit.get(unit))
        out.append(rec)
    return out


# --------------------------------------------------------------------------- artifacts

def artifact_conditions(art: Dict[str, Any]) -> Dict[str, Any]:
    """Run-artifact index the joins need: condition -> item, plus issue -> photo/observation."""
    res = rc.v5_result(art)
    if res is None:
        return {"verified": False, "conditions": {}, "issue_refs": {}, "idx": None}
    idx = rc.index_result(res)
    conditions = {cid: {"catalog_item_id": c.get("catalog_item_id"), "catalog_kind": c.get("catalog_kind"),
                        "scene_group": c.get("scene_group"), "issue_ids": list(c.get("issue_ids") or [])}
                  for cid, c in sorted(idx["conds"].items())}
    issue_refs: Dict[str, Dict[str, Any]] = {}
    for cid, ev in sorted(idx["evs"].items()):
        for ref in ev.get("evidence_refs") or []:
            if isinstance(ref, dict) and ref.get("issue_id"):
                issue_refs[ref["issue_id"]] = {"photo_key": ref.get("photo_key"),
                                               "observation": ref.get("observation"), "condition_id": cid}
    return {"conditions": conditions, "issue_refs": issue_refs, "idx": idx}


def resolved_item_detail(art: Dict[str, Any], photo_key: Any, issue_id: Any) -> Optional[Dict[str, Any]]:
    """Pass 2d's candidate list for one issue, verbatim order, rank = position."""
    photo = (art.get("photos") or {}).get(photo_key) or {}
    for rec in (photo.get("debug") or {}).get("resolved_items") or []:
        if isinstance(rec, dict) and rec.get("issue_id") == issue_id:
            return {"resolved_item_id": rec.get("resolved_item_id"), "resolution_path": rec.get("resolution_path"),
                    "routing_reason": rec.get("routing_reason"), "shortcut_reason": rec.get("shortcut_reason"),
                    "candidates": [{"rank": n + 1, "item_id": c.get("item_id"), "score": c.get("score")}
                                   for n, c in enumerate(rec.get("candidates") or []) if isinstance(c, dict)]}
    return None


def candidate_enrichment(issues: Sequence[Dict[str, Any]], art: Dict[str, Any]) -> Dict[str, Any]:
    """issue_id -> frozen 2d detail; `agrees_with_queue` cross-checks the queue's resolved id."""
    out = {}
    for issue in issues:
        detail = resolved_item_detail(art, issue.get("photo_key"), issue.get("issue_id"))
        if detail is None:
            out[str(issue.get("issue_id"))] = {"status": "unavailable", "reason": "issue not in artifact resolved_items"}
            continue
        detail["photo_key"] = issue.get("photo_key")
        detail["observation"] = issue.get("observation")
        queued = issue.get("resolved_item_id")
        detail["agrees_with_queue"] = (queued == detail["resolved_item_id"]) if queued is not None else None
        out[str(issue.get("issue_id"))] = detail
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- seeds and units

def _lane(items: Sequence[str], record_type: str, trade_of: Dict[str, str], quarantined: Iterable[str]) -> str:
    quarantined = set(quarantined)
    if items and all(trade_of.get(i) in quarantined for i in items):
        return "product_policy"
    if record_type == "deferred_finding":
        return "backlog"
    if not items:
        return "coverage_question" if record_type in ("gold_case", "gold_row") else "unattached"
    return "catalog_candidate"


def seed_records(cases: Dict[str, Dict[str, Any]], gold_cases: Dict[str, Dict[str, Any]],
                 leads: Sequence[Dict[str, Any]], gold_rows: Sequence[Dict[str, Any]],
                 findings: Sequence[Dict[str, Any]], trade_of: Dict[str, str],
                 quarantined: Iterable[str]) -> List[Dict[str, Any]]:
    """One record per source row; a row hit by several rules carries them all."""
    records: Dict[str, Dict[str, Any]] = {}

    def add(record_id: str, rule: str, base: Dict[str, Any]) -> None:
        rec = records.setdefault(record_id, {**base, "record_id": record_id, "rules": [], "reasons": []})
        rec["rules"].append(rule)
        rec["reasons"].append(RULES[rule])

    for cid, c in sorted(cases.items()):
        att = c.get("attribution") or {}
        items = sorted(set(c["issue_items"]) | ({c["condition_item"]} if c["condition_item"] else set()))
        base = {"record_type": "case", "method": METHOD["case"], "unit_key": c["unit_key"],
                "implicated_items": items, "property_key": c["property_key"], "source": c["source"],
                "case_lane": c["lane"], "attribution": att.get("attribution"), "stage": att.get("stage"),
                "unattached_reason": None if c["unit_key"] else "no_condition_in_case"}
        if att.get("attribution") == "downstream" and att.get("stage") == "2d":
            add(cid, "R1", base)
        if c["lane"] == "appendix_misnamed":
            add(cid, "R2", base)
        if c["lane"] in MISS_LANES and att.get("attribution") == "downstream" and att.get("stage") == "terra":
            add(cid, "R3", base)
    for gid, g in sorted(gold_cases.items()):
        att = g.get("attribution") or {}
        base = {"record_type": "gold_case", "method": METHOD["gold_case"], "unit_key": g["unit_key"],
                "implicated_items": [g["covering_item"]] if g.get("covering_item") else [],
                "property_key": g["property_key"], "source": g["source"], "case_lane": g["lane"],
                "attribution": att.get("attribution"), "stage": att.get("stage"), "unattached_reason": None}
        add(gid, "R5", base)
        if att.get("attribution") == "downstream" and att.get("stage") == "2d":
            add(gid, "R1", base)
        if g["lane"] in MISS_LANES and att.get("attribution") == "downstream" and att.get("stage") == "terra":
            add(gid, "R3", base)
    for lead in leads:
        base = {"record_type": "factorized_lead", "method": METHOD["factorized_lead"], "unit_key": lead["unit_key"],
                "implicated_items": [lead["catalog_item_id"]] if lead.get("catalog_item_id") else [],
                "property_key": lead["property_key"], "source": lead["source"], "case_lane": None,
                "attribution": None, "stage": None, "unattached_reason": lead.get("unattached_reason")}
        add(lead["record_id"], "R4", base)
    for row in gold_rows:
        if row.get("decision") != "out_of_catalog":
            continue
        base = {"record_type": "gold_row", "method": METHOD["gold_row"], "unit_key": row["gold_unit_key"],
                "implicated_items": [], "property_key": row["property_key"], "source": row["source"],
                "case_lane": None, "attribution": None, "stage": None, "unattached_reason": None}
        add(f"gold_row:{row['photo']}:{row['gold_id']}", "R6", base)
    for finding in findings:
        base = {"record_type": "deferred_finding", "method": METHOD["deferred_finding"], "unit_key": None,
                "implicated_items": list(finding.get("item_refs") or []), "property_key": None, "source": None,
                "case_lane": None, "attribution": None, "stage": None, "unattached_reason": "backlog_reference"}
        add(f"finding:{finding['section']}", "R7", base)
    out = []
    for rec in sorted(records.values(), key=lambda r: r["record_id"]):
        rec["rules"] = sorted(rec["rules"])
        rec["reasons"] = [RULES[r] for r in rec["rules"]]
        rec["lane"] = _lane(rec["implicated_items"], rec["record_type"], trade_of, quarantined)
        out.append(rec)
    return out


def assign_units(records: Sequence[Dict[str, Any]], cases: Dict[str, Dict[str, Any]],
                 gold_cases: Dict[str, Dict[str, Any]], trade_of: Dict[str, str],
                 quarantined: Iterable[str]) -> Dict[str, Any]:
    """Group records into evidence units by key; within a unit the primary record is
    independent and every other record is method corroboration or annotation."""
    units: Dict[str, Dict[str, Any]] = {}
    unattached = []
    for rec in sorted(records, key=lambda r: r["record_id"]):
        key = rec.get("unit_key")
        if not key:
            unattached.append({"record_id": rec["record_id"], "record_type": rec["record_type"],
                               "rules": rec["rules"], "reason": rec.get("unattached_reason") or "no_unit_key"})
            continue
        unit = units.get(key)
        if unit is None:
            parts = key.split(":")
            if parts[0] == "runtime":
                unit = {"unit_key": key, "unit_type": "runtime", "source": parts[1], "property_key": parts[2],
                        "run_id": parts[3], "condition_id": parts[4], "photo_key": None, "gold_id": None}
            else:
                unit = {"unit_key": key, "unit_type": "gold", "source": "canary", "property_key": parts[1],
                        "run_id": None, "condition_id": None, "photo_key": parts[2], "gold_id": parts[3]}
            unit.update(records=[], photo_keys=[], covering_unit_key=None, covering_case_id=None)
            units[key] = unit
        unit["records"].append({"record_id": rec["record_id"], "record_type": rec["record_type"],
                                "method": rec["method"], "rules": rec["rules"], "role": None})
        if rec["record_type"] == "case":
            unit["photo_keys"] = cases[rec["record_id"]]["photo_keys"]
        elif rec["record_type"] == "gold_case":
            g = gold_cases[rec["record_id"]]
            unit["photo_keys"] = [g["photo_key"]]
            unit["covering_unit_key"] = g.get("covering_unit_key")
            unit["covering_case_id"] = g.get("covering_case_id")
        elif rec["record_type"] == "gold_row":
            unit["photo_keys"] = [unit["photo_key"]]
    by_id = {r["record_id"]: r for r in records}
    for unit in units.values():
        unit["records"].sort(key=lambda r: (PRECEDENCE[r["record_type"]], r["record_id"]))
        primary = unit["records"][0]
        primary["role"] = "independent"
        for other in unit["records"][1:]:
            other["role"] = "method_corroboration" if other["method"] != primary["method"] else "annotation"
        items = sorted({i for r in unit["records"] for i in by_id[r["record_id"]]["implicated_items"]})
        unit.update(primary_record_id=primary["record_id"], primary_method=primary["method"],
                    rules=sorted({x for r in unit["records"] for x in r["rules"]}),
                    implicated_items=items, attached=bool(items),
                    unattached_reason=None if items else "no_implicated_item",
                    lane=_lane(items, primary["record_type"], trade_of, quarantined),
                    independent=True, corroborates=None, corroborated_by=[],
                    same_property_units=[], same_photo_units=[])
    return {"units": [units[k] for k in sorted(units)], "unattached": unattached}


def correlations(units: Sequence[Dict[str, Any]]) -> None:
    """Same-condition gold links downgrade to corroboration; same photo / property are flags."""
    by_key = {u["unit_key"]: u for u in units}
    for u in units:
        cov = u.get("covering_unit_key")
        if u["unit_type"] == "gold" and cov and cov in by_key:
            u["independent"], u["corroborates"] = False, cov
            by_key[cov]["corroborated_by"].append(u["unit_key"])
    photos: Dict[Tuple[str, str], List[str]] = {}
    props: Dict[str, List[str]] = {}
    for u in units:
        props.setdefault(u["property_key"], []).append(u["unit_key"])
        for pk in u["photo_keys"]:
            photos.setdefault((u["property_key"], pk), []).append(u["unit_key"])
    for u in units:
        u["corroborated_by"].sort()
        u["same_property_units"] = sorted(k for k in props[u["property_key"]] if k != u["unit_key"])
        u["same_photo_units"] = sorted({k for pk in u["photo_keys"] for k in photos[(u["property_key"], pk)]
                                        if k != u["unit_key"]})


def reconcile_seeds(records: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any]],
                    unattached: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Every record lands in exactly one unit or the unattached list; counts before and after."""
    in_units = Counter(r["record_id"] for u in units for r in u["records"])
    in_unattached = Counter(x["record_id"] for x in unattached)
    problems = []
    for rec in records:
        n = in_units[rec["record_id"]] + in_unattached[rec["record_id"]]
        if n != 1:
            problems.append(f"{rec['record_id']} reconciled {n} times")
    known = {r["record_id"] for r in records}
    for rid in sorted((set(in_units) | set(in_unattached)) - known):
        problems.append(f"{rid} reconciled but never seeded")
    fail_closed(problems)
    roles = Counter(r["role"] for u in units for r in u["records"])
    return {
        "before_dedup": {
            "records": len(records),
            "records_by_rule": dict(sorted(Counter(rule for r in records for rule in r["rules"]).items())),
            "records_by_type": dict(sorted(Counter(r["record_type"] for r in records).items())),
            "records_by_lane": dict(sorted(Counter(r["lane"] for r in records).items())),
            "rule_record_pairs": sum(len(r["rules"]) for r in records),
        },
        "after_dedup": {
            "units": len(units),
            "units_by_type": dict(sorted(Counter(u["unit_type"] for u in units).items())),
            "units_by_lane": dict(sorted(Counter(u["lane"] for u in units).items())),
            "units_by_source": dict(sorted(Counter(u["source"] for u in units).items())),
            "units_attached": sum(1 for u in units if u["attached"]),
            "independent_units": sum(1 for u in units if u["independent"]),
            "corroborating_units": sum(1 for u in units if not u["independent"]),
            "record_roles": dict(sorted(roles.items())),
            "unattached_records": len(unattached),
            "unattached_by_reason": dict(sorted(Counter(x["reason"] for x in unattached).items())),
        },
        "every_seed_reconciled_once": True,
    }


# --------------------------------------------------------------------------- families and worklist

def item_families(item_ids: Iterable[str], current: Dict[str, Any], era: Dict[str, Any],
                  migration: Dict[str, Any], enrichment: Dict[str, Dict[str, Any]],
                  units: Sequence[Dict[str, Any]], cases: Dict[str, Dict[str, Any]],
                  cases_by_unit: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    cur, old = _items(current), _items(era)
    by_wic: Dict[str, List[str]] = {}
    by_group: Dict[str, List[str]] = {}
    by_pkg: Dict[str, List[str]] = {}
    for item_id, it in cur.items():
        if it.get("work_item_code"):
            by_wic.setdefault(it["work_item_code"], []).append(item_id)
        if (it.get("estimate") or {}).get("group"):
            by_group.setdefault(it["estimate"]["group"], []).append(item_id)
        for entry in (it.get("package_affinity") or {}).values():
            if isinstance(entry, dict) and entry.get("package_type"):
                by_pkg.setdefault(entry["package_type"], []).append(item_id)
    out = {}
    for item_id in sorted(set(item_ids)):
        it = cur.get(item_id) or old.get(item_id) or {}
        legacy = migration["parent_of"].get(item_id)
        scenes = set(it.get("scene_groups") or [])
        same = sorted(o for o, other in cur.items() if o != item_id and other.get("kind") == it.get("kind")
                      and other.get("trade_bucket") == it.get("trade_bucket")
                      and scenes & set(other.get("scene_groups") or []))
        shared = {"work_item_code": sorted(o for o in by_wic.get(it.get("work_item_code") or "", []) if o != item_id),
                  "estimate_group": sorted(o for o in by_group.get((it.get("estimate") or {}).get("group") or "", [])
                                           if o != item_id),
                  "package_type": sorted({o for e in (it.get("package_affinity") or {}).values() if isinstance(e, dict)
                                          for o in by_pkg.get(e.get("package_type") or "", []) if o != item_id})}
        my_units = [u for u in units if item_id in u["implicated_items"]]
        neighbors: Dict[str, Dict[str, Any]] = {}
        observations = set()
        scenes_seen = set()
        for u in my_units:
            # Seed records plus the unit's own condition case (a lead may sit on a
            # non-seeded case whose frozen candidates are keyed by that case id).
            source_ids = {r["record_id"] for r in u["records"]}
            if cases_by_unit.get(u["unit_key"]):
                source_ids.add(cases_by_unit[u["unit_key"]])
            for rid in sorted(source_ids):
                case = cases.get(rid)
                if case:
                    scenes_seen.add(case.get("scene_group"))
                    observations.update(i["observation"] for i in case["issues"] if i.get("observation"))
                for detail in (enrichment.get(rid) or {}).values():
                    if detail.get("resolved_item_id") != item_id:
                        continue
                    for cand in detail.get("candidates") or []:
                        if cand["item_id"] == item_id:
                            continue
                        n = neighbors.setdefault(cand["item_id"], {"item_id": cand["item_id"], "co_listed": 0,
                                                                    "best_rank": cand["rank"],
                                                                    "score_min": cand["score"],
                                                                    "score_max": cand["score"]})
                        n["co_listed"] += 1
                        n["best_rank"] = min(n["best_rank"], cand["rank"])
                        if cand["score"] is not None:
                            n["score_min"] = min(n["score_min"], cand["score"]) if n["score_min"] is not None else cand["score"]
                            n["score_max"] = max(n["score_max"], cand["score"]) if n["score_max"] is not None else cand["score"]
        claim = it.get("atomic_claim") if isinstance(it.get("atomic_claim"), dict) else {}
        members = sorted(set(migration["siblings"].get(item_id, [])) | set(same)
                         | {o for ids in shared.values() for o in ids})
        out[item_id] = {
            "item_id": item_id, "in_proposal_baseline": item_id in cur, "in_evidence_era": item_id in old,
            "migration": {"legacy_id": legacy, "change_type": migration["change_type"].get(legacy) if legacy else None,
                          "legacy_kind": migration["legacy_kind"].get(legacy) if legacy else None,
                          "successors_of_parent": migration["successors"].get(legacy, []) if legacy else [],
                          "siblings": migration["siblings"].get(item_id, [])},
            "same_kind_trade_scene": same,
            "shared_economics": shared,
            "family_members": members,
            "frozen_candidate_neighbors": sorted(neighbors.values(),
                                                 key=lambda n: (-n["co_listed"], n["best_rank"], n["item_id"])),
            "current_retrieval_neighbors": {"status": "unavailable", "reason": RETRIEVAL_UNAVAILABLE},
            "reviewer_neighbors": {"added": [], "removed": [], "rationale": None},
            "observations": sorted(observations),
            "dimensions": {"kind": it.get("kind"), "trade_bucket": it.get("trade_bucket"),
                           "scene_groups": list(it.get("scene_groups") or []),
                           "scene_groups_seen": sorted(s for s in scenes_seen if s),
                           "claim_subject": claim.get("subject"), "claim_state": claim.get("state"),
                           "ontology_basis": claim.get("ontology_basis"),
                           "claim_text": {"evidence_era": claim_text(old[item_id]) if item_id in old else None,
                                          "proposal_baseline": claim_text(cur[item_id]) if item_id in cur else None}},
        }
    return out


def build_worklist(units: Sequence[Dict[str, Any]], families: Dict[str, Dict[str, Any]],
                   cases: Dict[str, Dict[str, Any]], labels_joined: Sequence[Dict[str, Any]],
                   notes_joined: Sequence[Dict[str, Any]], findings: Sequence[Dict[str, Any]],
                   trade_of: Dict[str, str], kind_of: Dict[str, str]) -> List[Dict[str, Any]]:
    """Per implicated item: counts only. The evidence bar is Session 2's call."""
    positive = {}
    for lab in labels_joined:
        if lab["class_v1_1"] == "supported_billed":
            positive.setdefault(lab["catalog_item_id"], []).append(lab["card_id"])
    agreements: Dict[str, List[str]] = {}
    rejections: Dict[str, List[str]] = {}
    hallucinations: Dict[str, List[str]] = {}
    for cid, c in sorted(cases.items()):
        item = c["condition_item"]
        if c["lane"] == "counted_agreement":
            agreements.setdefault(item, []).append(cid)
        elif c["lane"] == "counted_correct_rejection":
            rejections.setdefault(item, []).append(cid)
        elif c["lane"] in HALLUC_LANES and (c.get("attribution") or {}).get("attribution") == "pass_2a":
            hallucinations.setdefault(item, []).append(cid)
    notes: Dict[str, List[str]] = {}
    for note in notes_joined:
        notes.setdefault(note["catalog_item_id"], []).append(note["card_id"])
    refs: Dict[str, List[int]] = {}
    for f in findings:
        for item in f.get("item_refs") or []:
            refs.setdefault(item, []).append(f["section"])

    def count(table: Dict[str, List[str]], ids: Iterable[str]) -> int:
        return sum(len(table.get(i, [])) for i in ids)

    out = []
    for item_id, fam in sorted(families.items()):
        mine = [u for u in units if item_id in u["implicated_items"]]
        independent = [u for u in mine if u["independent"]]
        scope = [item_id] + fam["family_members"]
        out.append({
            "item_id": item_id, "lane": "product_policy" if any(u["lane"] == "product_policy" for u in mine)
            else "catalog_candidate", "kind": kind_of.get(item_id), "trade_bucket": trade_of.get(item_id),
            "unit_keys": [u["unit_key"] for u in mine],
            "record_ids": sorted({r["record_id"] for u in mine for r in u["records"]}),
            "rules": sorted({x for u in mine for x in u["rules"]}),
            "independent_units": len(independent), "corroborating_units": len(mine) - len(independent),
            "distinct_properties": len({u["property_key"] for u in independent}),
            "units_sharing_a_photo": sum(1 for u in independent if set(u["same_photo_units"]) & {v["unit_key"] for v in mine}),
            "positive_uses": positive.get(item_id, []), "agreements": agreements.get(item_id, []),
            "correct_rejections": rejections.get(item_id, []), "notes": notes.get(item_id, []),
            "hallucination_annotations": hallucinations.get(item_id, []),
            "deferred_finding_refs": refs.get(item_id, []),
            "family_size": len(fam["family_members"]),
            "family_counts": {"positive_uses": count(positive, scope), "agreements": count(agreements, scope),
                              "correct_rejections": count(rejections, scope), "notes": count(notes, scope)},
        })
    return out


# --------------------------------------------------------------------------- bundle

def fingerprint(bundle: Dict[str, Any]) -> str:
    """Content hash that ignores the git block, so parity survives HEAD moving."""
    return sha256_canonical({k: v for k, v in bundle.items() if k not in ("git", "fingerprint")})


def build_bundle(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Pure assembly from loaded inputs (see load_all for the keys)."""
    queue, gold_doc, latest = inp["queue"], inp["gold_cases"], inp["latest"]
    current, era = inp["catalog_current"], inp["catalog_era"]
    cur_items, era_items = _items(current), _items(era)
    trade_of = {i: it.get("trade_bucket") for i, it in {**era_items, **cur_items}.items()}
    kind_of = {i: it.get("kind") for i, it in {**era_items, **cur_items}.items()}
    quarantined = product_quarantined_trade_buckets(current)

    idx = index_cases(queue, gold_doc, latest, inp["artifacts"])
    cases, gold_cases = idx["cases"], idx["gold_cases"]
    cards = inp["cards"]
    labels_joined = join_labels(inp["labels"], cards, cases)
    notes_joined = join_notes(inp["notes"], cards, cases, inp["labels"])
    gold_photos = {g["gold_photo_id"]: g for g in queue.get("gold_photos") or []}
    gold_rows = join_gold_rows(gold_doc.get("matching_table") or [], gold_photos, idx["cases_by_unit"])
    canary_runs: Dict[str, str] = {}
    run_conflicts = []
    for rec in list(cases.values()) + list(gold_cases.values()):
        if rec["source"] == "canary" and rec["run_id"]:
            if canary_runs.setdefault(rec["property_key"], rec["run_id"]) != rec["run_id"]:
                run_conflicts.append(rec["property_key"])
    labels_by_unit = {lab["unit_key"]: lab["card_id"] for lab in labels_joined if lab["unit_key"]}
    cards_by_unit = {card_unit_key(c): cid for cid, c in sorted(cards.items()) if card_unit_key(c)}
    leads = join_factorized_leads(inp["leads"], canary_runs, inp["artifacts"], idx["cases_by_unit"],
                                  labels_by_unit, cards_by_unit)
    findings = extract_findings_refs(inp["findings_text"], set(cur_items) | set(era_items))

    # Frozen candidate detail, only from hash-verified artifacts named by the queue.
    enrichment: Dict[str, Dict[str, Any]] = {}
    enrichment_gold: Dict[str, Dict[str, Any]] = {}
    unavailable = []
    for cid, c in sorted(cases.items()):
        art = inp["artifacts"].get(run_key(c["source"], c["property_key"], c["run_id"])) if c["run_id"] else None
        if c["issues"] and art and art.get("verified"):
            enrichment[cid] = candidate_enrichment(c["issues"], art["artifact"])
        elif c["status"] != "untraceable":
            unavailable.append({"scope": "candidate_enrichment", "id": cid, "reason": "artifact not pinned or unverified"})
    for lead in leads:
        if not lead["unit_key"] or lead["joined_case_id"]:
            continue
        art = inp["artifacts"][run_key("canary", lead["property_key"], lead["run_id"])]
        issue_ids = art["conditions"][lead["condition_id"]].get("issue_ids") or []
        issues = [{"issue_id": iid, **art["issue_refs"].get(iid, {})} for iid in issue_ids]
        enrichment[lead["record_id"]] = candidate_enrichment(issues, art["artifact"])
    for gid, g in sorted(gold_photos.items()):
        ref = g.get("run_ref") or {}
        art = inp["artifacts"].get(run_key(ref.get("source"), g["property_key"], ref.get("run_id")))
        surviving = [{"issue_id": s.get("issue_id"), "photo_key": g["photo_key"], "observation": s.get("description"),
                      "resolved_item_id": None}
                     for s in ((g.get("lineage") or {}).get("p2c_surviving") or [])]
        if art and art.get("verified"):
            enrichment_gold[gid] = candidate_enrichment(surviving, art["artifact"])
        else:
            unavailable.append({"scope": "candidate_enrichment", "id": gid, "reason": "artifact not pinned or unverified"})
    for lead in leads:
        if lead["unattached_reason"]:
            unavailable.append({"scope": "factorized_lead", "id": lead["record_id"], "reason": lead["unattached_reason"]})
    for run in inp["run_artifacts"]:
        if run["status"] != "pinned":
            unavailable.append({"scope": "run_artifact", "id": run_key(run["source"], run["property_key"], run["run_id"]),
                                "reason": "queue stores no artifact path/hash for this run (orphan lane); not read"})
    unavailable.append({"scope": "current_retrieval_neighbors", "id": "*", "reason": RETRIEVAL_UNAVAILABLE})

    records = seed_records(cases, gold_cases, leads, gold_rows, findings, trade_of, quarantined)
    assigned = assign_units(records, cases, gold_cases, trade_of, quarantined)
    units, unattached = assigned["units"], assigned["unattached"]
    correlations(units)
    # Gold annotations (matched / gold_incomplete / already_cased rows) on seeded runtime units.
    by_key = {u["unit_key"]: u for u in units}
    for u in units:
        u["gold_annotations"] = []
    for row in gold_rows:
        if row["role"] == "annotation" and row["runtime_unit_key"] in by_key:
            by_key[row["runtime_unit_key"]]["gold_annotations"].append(
                {"decision": row["decision"], "gold_id": row["gold_id"], "photo": row["photo"]})
    reconciliation = reconcile_seeds(records, units, unattached)

    item_ids = {i for u in units for i in u["implicated_items"]}
    families = item_families(item_ids, current, era, inp["migration"], enrichment, units, cases, idx["cases_by_unit"])
    worklist = build_worklist(units, families, cases, labels_joined, notes_joined, findings, trade_of, kind_of)

    comparison = compare_catalogs(era, current)
    decision_counts = dict(sorted(Counter(e.get("change_type") for e in inp["decisions"].get("entries") or []).items()))

    checks = [
        {"check": "frozen_pins_match", "ok": True, "detail": "verified before build (fail-closed)"},
        {"check": "queue_lane_counts_match_header",
         "ok": dict(Counter(c["lane"] for c in cases.values())) == dict(queue.get("lane_counts") or {}),
         "detail": dict(sorted(Counter(c["lane"] for c in cases.values()).items()))},
        {"check": "run_artifacts_verified",
         "ok": all(r["verified"] for r in inp["run_artifacts"] if r["status"] == "pinned"),
         "detail": {"pinned": sum(1 for r in inp["run_artifacts"] if r["status"] == "pinned"),
                    "verified": sum(1 for r in inp["run_artifacts"] if r.get("verified"))}},
        {"check": "artifact_catalog_sha256_uniform",
         "ok": len({r["catalog_sha256"] for r in inp["run_artifacts"] if r["status"] == "pinned"}) == 1,
         "detail": sorted({str(r["catalog_sha256"]) for r in inp["run_artifacts"] if r["status"] == "pinned"})},
        {"check": "catalog_identity_resolved", "ok": bool(inp["catalog_identity"]["evidence_era"]["commit"]),
         "detail": inp["catalog_identity"]["evidence_era"]["commit"]},
        {"check": "working_tree_matches_head",
         "ok": inp["catalog_identity"]["proposal_baseline"]["working_tree_matches_head"], "detail": None},
        {"check": "canary_run_identity_unique", "ok": not run_conflicts, "detail": sorted(set(run_conflicts))},
        {"check": "decision_counts_total", "ok": sum(decision_counts.values()) == len(inp["decisions"].get("entries") or []),
         "detail": decision_counts},
        {"check": "issue_items_equal_condition_item",
         "ok": True, "detail": sorted(cid for cid, c in cases.items() if c["condition_item"]
                                      and set(c["issue_items"]) - {c["condition_item"]})},
        {"check": "label_item_matches_card", "ok": True,
         "detail": sorted(lab["card_id"] for lab in labels_joined if lab["item_id_matches_card"] is False)},
        {"check": "labels_without_card", "ok": True,
         "detail": sorted(lab["card_id"] for lab in labels_joined if not lab["card_present"])},
        {"check": "artifact_resolution_agrees_with_queue", "ok": True,
         "detail": sorted(f"{cid}:{iid}" for cid, e in enrichment.items() for iid, d in e.items()
                          if d.get("agrees_with_queue") is False)},
        {"check": "every_seed_reconciled_once", "ok": reconciliation["every_seed_reconciled_once"], "detail": None},
    ]
    informational = {"issue_items_equal_condition_item", "label_item_matches_card", "labels_without_card",
                     "artifact_resolution_agrees_with_queue"}
    fail_closed([f"validation failed: {c['check']} ({c['detail']})" for c in checks
                 if not c["ok"] and c["check"] not in informational])

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "builder": REL["builder"],
        "git": inp["git"],
        "sources": inp["sources"],
        "queue_input_pins": inp["queue_input_pins"],
        "run_artifacts": inp["run_artifacts"],
        "run_count_by_source": inp["run_count_by_source"],
        "catalog_identity": inp["catalog_identity"],
        "factorized_manifest": {k: inp["factorized_manifest"].get(k) for k in
                                ("catalog_sha256", "projection_fingerprint", "root", "prompt_version",
                                 "response_contract", "created_at", "dry_run")},
        "baseline_comparison": comparison,
        "decision_counts": decision_counts,
        "authoring_surface": inp["authoring_surface"],
        "migration": inp["migration"],
        "item_semantics": {"era": "proposal_baseline", "items": {i: item_semantics(it) for i, it in sorted(cur_items.items())}},
        "indexes": {"cases": cases, "gold_cases": gold_cases, "by_item": idx["by_item"], "by_run": idx["by_run"],
                    "by_photo": idx["by_photo"], "by_condition": idx["by_condition"]},
        "labels": labels_joined,
        "positive_uses": [lab["card_id"] for lab in labels_joined if lab["class_v1_1"] == "supported_billed"],
        "agreements": sorted(cid for cid, c in cases.items() if c["lane"] == "counted_agreement"),
        "correct_rejections": sorted(cid for cid, c in cases.items() if c["lane"] == "counted_correct_rejection"),
        "review_notes": notes_joined,
        "factorized_leads": leads,
        "gold": {"cases": gold_cases, "matching_rows": gold_rows},
        "candidate_enrichment": {"cases_and_leads": enrichment, "gold_photos": enrichment_gold},
        "seeds": {"rules": RULES, "records": records},
        "evidence_units": units,
        "unattached": unattached,
        "worklist": worklist,
        "families": families,
        "lanes": {
            "product_policy": [r["record_id"] for r in records if r["lane"] == "product_policy"],
            "coverage_questions": [r["record_id"] for r in records if r["lane"] == "coverage_question"],
            "deferred_findings": findings,
            "hallucination_annotations": sorted(cid for cid, c in cases.items() if c["lane"] in HALLUC_LANES
                                                and (c.get("attribution") or {}).get("attribution") == "pass_2a"),
            "product_quarantined_trades": sorted(quarantined),
        },
        "reconciliation": reconciliation,
        "validation": {"ok": all(c["ok"] for c in checks), "checks": checks},
        "unavailable": sorted(unavailable, key=lambda x: (x["scope"], x["id"])),
        "notes": [
            "Evidence-era (3.1) item semantics = item_semantics.items overlaid with baseline_comparison"
            ".changed_items[].changes[field].evidence_era; only differing fields are stored twice.",
            "run_artifacts[].artifact_path is the queue's absolute path, kept verbatim as frozen provenance.",
            "Seeds are mechanical inclusion rules, not diagnoses; worklist rows carry counts only.",
            "condition_id is run-scoped: every runtime key is (source, property_key, run_id, condition_id).",
            "gold_id is scoped per photo: every gold key is (property_key, photo_key, gold_id).",
        ],
    }
    bundle["fingerprint"] = fingerprint(bundle)
    return bundle


# --------------------------------------------------------------------------- loading (I/O)

def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git(*args: str) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=str(ROOT), check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        fail_closed([f"git {' '.join(args)} failed: {exc}"])
        raise


def load_git(paths: Sequence[str]) -> Dict[str, Any]:
    tracked = set(_git("ls-files", "--", *paths).decode("utf-8").split("\n"))
    return {
        "head": _git("rev-parse", "HEAD").decode("utf-8").strip(),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").decode("utf-8").strip(),
        "tracked_changes": sorted(l for l in _git("status", "--porcelain", "--untracked-files=no")
                                  .decode("utf-8").splitlines() if l.strip()),
        "tracked_inputs": {p: p in tracked for p in sorted(paths)},
        "note": "not part of the fingerprint; untracked files are deliberately not counted",
    }


def load_catalog_history(rel_path: str) -> List[Tuple[str, bytes]]:
    commits = _git("log", "--format=%H", "--", rel_path).decode("utf-8").split()
    return [(c, _git("show", f"{c}:{rel_path}")) for c in commits]


def load_authoring_surface() -> Dict[str, Any]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("migrate_catalog_kind_v2", ROOT / REL["generator"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return {"wording_override_fields": sorted(module.WORDING_OVERRIDE_FIELDS),
            "inherited_fields": list(module.INHERITED_FIELDS),
            "successor_required_fields": list(module.SUCCESSOR_REQUIRED_FIELDS),
            "economic_fields": list(ECONOMIC_FIELDS), "target_version": module.TARGET_VERSION}


def load_artifacts(queue: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Every run the queue references; pinned ones are re-hashed and indexed."""
    refs: Dict[str, Dict[str, Any]] = {}
    for kind, recs in (("cases", queue.get("cases") or []), ("gold_photos", queue.get("gold_photos") or [])):
        for rec in recs:
            ref = rec.get("run_ref") or {}
            key = run_key(ref.get("source"), ref.get("property_key"), ref.get("run_id"))
            entry = refs.setdefault(key, {"source": ref.get("source"), "property_key": ref.get("property_key"),
                                          "run_id": ref.get("run_id"), "artifact_path": ref.get("artifact_path"),
                                          "queue_sha256": ref.get("artifact_sha256"), "case_ids": [],
                                          "gold_photo_ids": []})
            if ref.get("artifact_path") and not entry["artifact_path"]:
                entry.update(artifact_path=ref["artifact_path"], queue_sha256=ref.get("artifact_sha256"))
            entry["case_ids" if kind == "cases" else "gold_photo_ids"].append(rec.get("case_id") or rec.get("gold_photo_id"))
    artifacts: Dict[str, Dict[str, Any]] = {}
    rows, problems = [], []
    for key, entry in sorted(refs.items()):
        row = {**entry, "case_ids": sorted(entry["case_ids"]), "gold_photo_ids": sorted(entry["gold_photo_ids"]),
               "status": "pinned" if entry["artifact_path"] else "unpinned", "current_sha256": None,
               "verified": False, "catalog_sha256": None, "catalog_version": None, "architecture_mode": None}
        if entry["artifact_path"]:
            path = Path(entry["artifact_path"])
            if not path.is_file():
                problems.append(f"{key}: artifact missing at {path}")
            else:
                row["current_sha256"] = sha256_file(path)
                row["verified"] = row["current_sha256"] == entry["queue_sha256"]
                if not row["verified"]:
                    problems.append(f"{key}: artifact sha256 {row['current_sha256']} != queue {entry['queue_sha256']}")
                art = _read_json(path)
                env = art.get("renovation_estimate_v5") if isinstance(art.get("renovation_estimate_v5"), dict) \
                    else (art.get("analysis_debug") or {}).get("renovation_estimate_v5") or {}
                prov = env.get("provenance") or {}
                row.update(catalog_sha256=prov.get("catalog_sha256"), catalog_version=prov.get("catalog_version"),
                           architecture_mode=prov.get("architecture_mode"))
                artifacts[key] = {"verified": row["verified"], "artifact": art, **artifact_conditions(art)}
        rows.append(row)
    return artifacts, rows, problems


def load_all() -> Dict[str, Any]:
    paths = {name: ROOT / rel for name, rel in REL.items()}
    shas = {name: (sha256_file(p) if p.is_file() else None) for name, p in paths.items()}
    fail_closed(verify_pins(shas, FROZEN_SHA256))
    queue = _read_json(paths["queue"])
    pins = {}
    for name, rec in sorted((queue.get("inputs") or {}).items()):
        rel = str(rec.get("path") or "").replace("\\", "/")
        current = sha256_file(ROOT / rel) if (ROOT / rel).is_file() else None
        pins[name] = {"path": rel, "queue_sha256": rec.get("sha256"), "current_sha256": current,
                      "match": current == rec.get("sha256")}
    fail_closed([f"queue-embedded pin {n}: {p['current_sha256']} != {p['queue_sha256']}"
                 for n, p in pins.items() if not p["match"]])
    artifacts, run_rows, problems = load_artifacts(queue)
    fail_closed(problems)
    counts: Dict[str, Dict[str, int]] = {}
    for row in run_rows:
        counts.setdefault(row["source"], {"pinned": 0, "unpinned": 0})[row["status"]] += 1
    history = load_catalog_history(REL["catalog_v2"])
    identity = resolve_catalog_identity(
        {r["catalog_sha256"] for r in run_rows if r["status"] == "pinned" and r["catalog_sha256"]},
        history, head_blob=_git("show", f"HEAD:{REL['catalog_v2']}"), on_disk=paths["catalog_v2"].read_bytes())
    fail_closed([] if identity["proposal_baseline"]["working_tree_matches_head"]
                else ["on-disk catalog differs from HEAD's blob (uncommitted catalog change)"])
    era_blob = dict(history)[identity["evidence_era"]["commit"]]
    analysis = _read_json(paths["review_analysis"])
    scorecard = _read_json(paths["scorecard"])
    return {
        "queue": queue, "gold_cases": _read_json(paths["gold_cases"]),
        "latest": latest_verdicts(paths["verdicts"]),
        "labels": _read_json(paths["labels"]).get("labels") or {},
        "cards": {c["card_id"]: c for c in _read_json(paths["review_queue"]).get("cards") or []},
        "notes": (analysis.get("qualitative") or {}).get("notes") or [],
        "leads": (scorecard.get("disagreements") or {}).get("misnamed_with_description") or [],
        "findings_text": paths["findings"].read_text(encoding="utf-8"),
        "factorized_manifest": _read_json(paths["factorized_manifest"]),
        "catalog_current": _read_json(paths["catalog_v2"]),
        "catalog_era": json.loads(era_blob.decode("utf-8")),
        "catalog_identity": identity,
        "manifest": _read_json(paths["manifest"]), "migration": migration_index(_read_json(paths["manifest"])),
        "decisions": _read_json(paths["decisions"]),
        "authoring_surface": load_authoring_surface(),
        "artifacts": artifacts, "run_artifacts": run_rows, "run_count_by_source": dict(sorted(counts.items())),
        "sources": [{"name": name, "path": REL[name], "tier": "frozen" if name in FROZEN_SHA256 else "recorded",
                     "sha256": shas[name], "expected_sha256": FROZEN_SHA256.get(name),
                     "match": (shas[name] == FROZEN_SHA256[name]) if name in FROZEN_SHA256 else None}
                    for name in sorted(REL)],
        "queue_input_pins": pins,
        "git": load_git(sorted(REL.values())),
    }


# --------------------------------------------------------------------------- rendering

def render_markdown(bundle: Dict[str, Any]) -> str:
    L: List[str] = []
    add = L.append
    ident, cmp_ = bundle["catalog_identity"], bundle["baseline_comparison"]
    add("# Catalog audit evidence bundle (Session 1)\n")
    add(f"Schema {bundle['schema_version']} · fingerprint `{bundle['fingerprint']}` · "
        f"starting commit `{bundle['git']['head']}` on `{bundle['git']['branch']}` (git block is not fingerprinted).\n")
    add("## Provenance\n")
    add("| Source | Tier | SHA-256 | Pinned |\n|---|---|---|---|")
    for s in bundle["sources"]:
        add(f"| `{s['path']}` | {s['tier']} | `{s['sha256']}` | {'match' if s['match'] else ('recorded' if s['match'] is None else 'MISMATCH')} |")
    add("")
    counts = bundle["run_count_by_source"]
    add(f"Run artifacts: {', '.join(f'{src} {c['pinned']} pinned / {c['unpinned']} unpinned' for src, c in counts.items())}; "
        f"all pinned artifacts re-hashed and verified; every pinned artifact records catalog_sha256 "
        f"`{', '.join(ident['artifact_catalog_sha256'])}`.\n")
    add("## Catalog identity\n")
    era, prop = ident["evidence_era"], ident["proposal_baseline"]
    add(f"- Evidence era: version {era['version']} at `{str(era['commit'])[:7]}` — blob (LF) `{era['sha256_lf']}`, "
        f"checkout (CRLF) `{era['sha256_crlf']}`, matched via {era['matched_convention']} "
        f"(checkout convention: {ident['checkout_line_ending_convention']}).")
    add(f"- Proposal baseline: version {prop['version']} at `{str(prop['catalog_commit'])[:7]}` — on-disk `{prop['on_disk_sha256']}`, "
        f"blob (LF) `{prop['sha256_lf']}`, working tree matches HEAD: {prop['working_tree_matches_head']}.")
    add(f"- Git history examined: {len(ident['history'])} commits of the generated catalog.\n")
    add("## Baseline comparison (3.1 → 3.2)\n")
    add(f"- Items: {cmp_['evidence_era']['item_count']} → {cmp_['proposal_baseline']['item_count']}; added {cmp_['added_ids']}, "
        f"removed {cmp_['removed_ids']}, order changed: {cmp_['order_changed']}, trade buckets changed: {cmp_['trade_buckets_changed']}.")
    add(f"- atomic_claim unchanged: **{cmp_['atomic_claim_unchanged']}**; claim_text unchanged: **{cmp_['claim_text_unchanged']}**.")
    add(f"- Per-field change counts: {cmp_['per_field_change_counts'] or 'none'}.")
    add(f"- Changed items ({len(cmp_['changed_items'])}): {', '.join('`' + c['id'] + '`' for c in cmp_['changed_items']) or 'none'}.")
    add(f"- Repair-support markers added by 3.2: {len(cmp_['repair_support_markers_added'])}; "
        f"product-quarantined trades: {cmp_['product_quarantined_trades']}.")
    add(f"- Decision counts: {bundle['decision_counts']}.\n")
    rec = bundle["reconciliation"]
    add("## Seeds and evidence units\n")
    add("| Rule | Records | Reason |\n|---|---:|---|")
    for rule, n in rec["before_dedup"]["records_by_rule"].items():
        add(f"| {rule} | {n} | {bundle['seeds']['rules'][rule]} |")
    add("")
    add(f"Records before dedup: {rec['before_dedup']['records']} ({rec['before_dedup']['rule_record_pairs']} rule/record pairs; "
        f"by type {rec['before_dedup']['records_by_type']}). After dedup: {rec['after_dedup']['units']} units "
        f"({rec['after_dedup']['units_by_type']}), {rec['after_dedup']['independent_units']} independent, "
        f"{rec['after_dedup']['corroborating_units']} corroborating, {rec['after_dedup']['units_attached']} attached to an item; "
        f"record roles {rec['after_dedup']['record_roles']}; unattached records {rec['after_dedup']['unattached_by_reason']}. "
        f"Every seed reconciled once: {rec['every_seed_reconciled_once']}.\n")
    add("## Worklist (counts only; no evidence-bar verdict)\n")
    add("| Item | Lane | Units | Indep. | Props | Corrob. | Pos. uses | Agree | Correct rej. | Notes | 2a halluc. | Family | Rules |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for w in bundle["worklist"]:
        add(f"| `{w['item_id']}` | {w['lane']} | {len(w['unit_keys'])} | {w['independent_units']} | {w['distinct_properties']} | "
            f"{w['corroborating_units']} | {len(w['positive_uses'])} | {len(w['agreements'])} | {len(w['correct_rejections'])} | "
            f"{len(w['notes'])} | {len(w['hallucination_annotations'])} | {w['family_size']} | {' '.join(w['rules'])} |")
    add("")
    add(f"Coverage questions (no implicated item): {len(bundle['lanes']['coverage_questions'])} records; "
        f"product-policy lane: {len(bundle['lanes']['product_policy'])} records; deferred findings: "
        f"{len(bundle['lanes']['deferred_findings'])} sections; hallucination annotations: "
        f"{len(bundle['lanes']['hallucination_annotations'])} cases.\n")
    add("## Unavailable data\n")
    for u in bundle["unavailable"]:
        add(f"- {u['scope']} `{u['id']}`: {u['reason']}")
    add("")
    add("## Validation\n")
    add("| Check | OK | Detail |\n|---|---|---|")
    for c in bundle["validation"]["checks"]:
        detail = json.dumps(c["detail"], sort_keys=True) if c["detail"] not in (None, [], {}) else ""
        add(f"| {c['check']} | {c['ok']} | {detail[:200]} |")
    add("")
    return "\n".join(L)


# --------------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--md", type=Path, default=OUT_MD)
    parser.add_argument("--check", action="store_true",
                        help="rebuild in memory and compare fingerprints with --out instead of writing")
    args = parser.parse_args(argv)

    bundle = build_bundle(load_all())
    if args.check:
        existing = _read_json(args.out) if args.out.is_file() else {}
        same = existing.get("fingerprint") == bundle["fingerprint"]
        print(f"fingerprint {'matches' if same else 'DIFFERS'}: {bundle['fingerprint']}")
        return 0 if same else 1
    atomic_json(args.out, bundle)
    args.md.write_text(render_markdown(bundle), encoding="utf-8")
    rec = bundle["reconciliation"]
    print(f"cases {len(bundle['indexes']['cases'])} + gold {len(bundle['indexes']['gold_cases'])}; "
          f"records {rec['before_dedup']['records']} -> units {rec['after_dedup']['units']}; "
          f"worklist items {len(bundle['worklist'])}; fingerprint {bundle['fingerprint']}")
    print(f"wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
