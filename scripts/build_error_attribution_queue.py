"""Case + lineage extractor for the Pass 2a vs. downstream error attribution audit.

Every prior analysis stops at "Terra was wrong" or "the claim was false". This
builds the one join nothing in the repo performs: from a human-labelled v5 error
back through condition projection, 2e, 2d, 2c and 2b to the raw Pass 2a prose
that produced it, so a reviewer can say *which stage* introduced the error.

Reads only frozen inputs (reports/labels_v1_1.json is the dedup-safe spine;
review_queue.json supplies run ids and blind meta; the frozen Pass 2a gold adds
a photo-level lane) plus the frozen run artifacts they point at. Writes

  reports/error_attribution_queue.json   one record per case, attribution null

Nothing here judges anything: it resolves each case to its exact frozen run and
pre-computes stage presence/absence so the review step judges only semantics.
The reviewer's verdicts land in reports/error_attribution_verdicts.jsonl and are
scored by scripts/error_attribution_report.py.

Zero provider calls, zero writes outside reports/. Never substitutes another run
for a missing one: condition_id is run-scoped (zero cross-run overlap), so a
case whose run is absent is `untraceable`, terminal.

Run:
  .venv\\Scripts\\python.exe scripts\\build_error_attribution_queue.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import review_cards as rc  # noqa: E402
from tools.comparison_common import atomic_json, sha256_bytes, sha256_file  # noqa: E402
from tools.pass_2f_artifact_inputs import photo_key_to_path  # noqa: E402

LABELS = ROOT / "reports" / "labels_v1_1.json"
QUEUE = ROOT / "reports" / "review_queue.json"
VERDICTS = ROOT / "reports" / "review_verdicts.jsonl"
ANALYSIS = ROOT / "reports" / "review_analysis.json"
GOLD = ROOT / "benchmarks" / "pass2a-prompt" / "gold" / "reference.json"
SCORECARD = ROOT / "reports" / "factorized_review_scorecard.json"
OUT = ROOT / "reports" / "error_attribution_queue.json"

SCHEMA_VERSION = 1

# Lane -> what the reviewer is being asked. Primary lanes carry the headline
# counts; appendix lanes are attributed too (Steven's call) but reported apart.
PRIMARY_LANES = ("miss_label", "miss_v1only", "miss_gold",
                 "halluc_label", "halluc_v1only", "halluc_gold_extra")
APPENDIX_LANES = ("appendix_misnamed", "appendix_trivial", "appendix_inconclusive")
COUNTED_ONLY_LANES = ("counted_correct_rejection", "counted_agreement", "counted_orphan")

MISS_CLASSES = ("dirB_recovery", "dirB_wording_recovery")
APPENDIX_CLASS_LANE = {"misnamed_billed": "appendix_misnamed",
                       "trivial_billed": "appendix_trivial",
                       "excluded": "appendix_inconclusive"}

STAGES = ("2b", "2c", "2d", "2e", "condition_projection", "terra")


def norm(text: Any) -> str:
    """Whitespace- and case-insensitive form, for the 2b join fallback."""
    return " ".join(str(text or "").split()).casefold()


# --------------------------------------------------------------------------- inputs

def load_inputs(*, labels_path: Path = LABELS, queue_path: Path = QUEUE,
                verdicts_path: Path = VERDICTS, gold_path: Path = GOLD,
                analysis_path: Path = ANALYSIS,
                scorecard_path: Path = SCORECARD) -> Dict[str, Any]:
    """Every frozen human-label source, plus the sha of each for the queue root."""
    labels_doc = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    queue_doc = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    gold_doc = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    analysis = (json.loads(Path(analysis_path).read_text(encoding="utf-8"))
                if Path(analysis_path).is_file() else {})
    scorecard = (json.loads(Path(scorecard_path).read_text(encoding="utf-8"))
                 if Path(scorecard_path).is_file() else {})
    inputs = {
        "labels": labels_doc.get("labels") or {},
        "cards": {c["card_id"]: c for c in queue_doc.get("cards") or []},
        "verdicts": rc.latest_verdicts(Path(verdicts_path)),
        "gold": gold_doc.get("photos") or {},
        "orphans": ((analysis.get("integrity") or {}).get("orphans") or []),
        "factorized": _factorized_flags(scorecard),
        "shas": {},
    }
    for name, path in (("labels_v1_1", labels_path), ("review_queue", queue_path),
                       ("review_verdicts", verdicts_path), ("gold_reference", gold_path),
                       ("review_analysis", analysis_path),
                       ("factorized_scorecard", scorecard_path)):
        p = Path(path)
        inputs["shas"][name] = {"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                                "sha256": sha256_file(p) if p.is_file() else None}
    return inputs


def _factorized_flags(scorecard: Dict[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    """(property_key, condition_id) -> factorized disagreement lists it appears in.

    Context only. These are model judgments, never a case selector."""
    out: Dict[Tuple[str, str], List[str]] = {}
    for name, rows in (scorecard.get("disagreements") or {}).items():
        for row in rows or []:
            key = (row.get("property_key"), row.get("condition_id"))
            if all(key):
                out.setdefault(key, []).append(name)
    return out


def load_listings() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """(source, property_key) -> {run_id, path, artifact, result, idx, paths}.

    Canary run_1 (18 properties, shadow v5) plus every production run carrying a
    complete v5 envelope (10 at the time of writing; the rest of the corpus is v4)."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    run1, _ = rc.load_canary(rc.CANARY_ROOT)
    listings: List[Tuple[str, str, Path, Dict[str, Any]]] = [
        ("canary", prop, path, art) for prop, (path, art) in sorted(run1.items())]
    for prop, _run_id, path, art in rc.iter_runs(rc.PROD_ROOT):
        listings.append(("production", prop, path, art))
    for source, prop, path, art in listings:
        res = rc.v5_result(art)
        if res is None:
            continue
        out[(source, prop)] = {"run_id": path.parent.name, "path": path, "artifact": art,
                               "result": res, "idx": rc.index_result(res),
                               "paths": photo_key_to_path(art)}
    return out


# --------------------------------------------------------------------------- lineage

def photo_lineage(art: Dict[str, Any], photo_key: str,
                  paths: Optional[Dict[str, Path]] = None) -> Dict[str, Any]:
    """Everything one photo contributed, stage by stage.

    2a is the whole freeform blob: there are no observation ids anywhere in 2a or
    2b, so provenance stops at the photo. 2c keeps no rationale for what it
    dropped, so the drop set is a set difference and nothing more."""
    photo = (art.get("photos") or {}).get(photo_key) or {}
    debug = photo.get("debug") or {}
    prose = ((photo.get("features") or {}).get("observations_freeform")) or ""
    bullets = [str(o.get("description") or "") for o
               in ((debug.get("observations_struct") or {}).get("observations") or [])
               if isinstance(o, dict)]
    surviving = [{"issue_id": i.get("issue_id"), "kind": i.get("kind"),
                  "catalog_item_id": i.get("catalogItemId"),
                  "description": i.get("description")}
                 for i in ((photo.get("issues") or {}).get("matched") or [])
                 if isinstance(i, dict)]
    kept = {s["description"] for s in surviving}
    p2e = (debug.get("passes") or {}).get("2e") or {}
    image_path = (paths or {}).get(photo_key)
    return {
        "photo_key": photo_key,
        "image_path": str(image_path) if image_path else None,
        "image_exists": bool(image_path and Path(image_path).is_file()),
        "scene": photo.get("scene"),
        "pass_states": photo.get("pass_states"),
        "p2a_prose": prose,
        "p2a_sha256": sha256_bytes(prose.encode("utf-8")) if prose else None,
        "p2a_present": bool(prose),
        "p2b_bullets": bullets,
        "p2c_surviving": surviving,
        # 2b bullets that no surviving issue carries: 2c's drop set, no reason recorded.
        "p2c_dropped": [b for b in bullets if b not in kept],
        "p2e": {"kept_issue_ids": p2e.get("kept_issue_ids") or [],
                "removed_count": int(p2e.get("removed_count") or 0),
                "suppressed_reason_counts": p2e.get("suppressed_reason_counts") or {}},
    }


def issue_lineage(art: Dict[str, Any], photo_key: str, issue_id: str,
                  observation: Optional[str], idx: Dict[str, Any]) -> Dict[str, Any]:
    """One evidence ref walked back to its 2b bullet.

    `observation` is the canonical issue description stamped at projection
    (tools/renovation_architecture/conditions.py:156-163), which is why the
    exact-string join to 2b works. A failed join is data, not an error."""
    photo = (art.get("photos") or {}).get(photo_key) or {}
    debug = photo.get("debug") or {}
    bullets = [str(o.get("description") or "") for o
               in ((debug.get("observations_struct") or {}).get("observations") or [])
               if isinstance(o, dict)]
    if observation in bullets:
        join = {"matched": True, "method": "exact", "bullet_index": bullets.index(observation)}
    else:
        lowered = [norm(b) for b in bullets]
        if norm(observation) in lowered:
            join = {"matched": True, "method": "normalized",
                    "bullet_index": lowered.index(norm(observation))}
        else:
            join = {"matched": False, "method": "none", "bullet_index": None}
    matched_ids = {i.get("issue_id") for i in ((photo.get("issues") or {}).get("matched") or [])
                   if isinstance(i, dict)}
    resolved = next((r for r in (debug.get("resolved_items") or [])
                     if isinstance(r, dict) and r.get("issue_id") == issue_id), None)
    p2e = (debug.get("passes") or {}).get("2e") or {}
    kept = p2e.get("kept_issue_ids") or []
    flat = next((f for f in (art.get("estimate_issues_flat") or [])
                 if isinstance(f, dict) and f.get("issue_id") == issue_id), None)
    return {
        "issue_id": issue_id,
        "photo_key": photo_key,
        "observation": observation,
        "p2b_join": join,
        "p2c_present": issue_id in matched_ids,
        "p2d": {"resolved_item_id": (resolved or {}).get("resolved_item_id"),
                "resolution_path": (resolved or {}).get("resolution_path"),
                "routing_reason": (resolved or {}).get("routing_reason"),
                "shortcut_reason": (resolved or {}).get("shortcut_reason"),
                "candidate_count": len((resolved or {}).get("candidates") or [])} if resolved
               else None,
        "p2e_status": ("kept" if issue_id in kept else "absent" if kept else "unknown"),
        # Enrichment only: conditions consume the product-filtered lane, so a miss
        # here is a lane note, never a broken walk.
        "flat_lane": "present" if flat else "missing",
        "projected_condition_id": (idx["cond_by_issue"].get(issue_id) or {}).get("condition_id"),
    }


def condition_lineage(listing: Dict[str, Any], condition_id: str) -> Dict[str, Any]:
    """Full backward walk for one condition, accepted or rejected alike.

    Anchored on the condition, not on `condition_dispositions[accepted_for_work]`:
    projection runs before Terra, so rejected (dirB) conditions carry identical
    lineage and an accepted-only walk would silently skip every miss case."""
    art, idx = listing["artifact"], listing["idx"]
    cond = idx["conds"].get(condition_id) or {}
    ev = idx["evs"].get(condition_id) or {}
    photo_keys = list(ev.get("photo_keys") or [])
    refs = [r for r in (ev.get("evidence_refs") or []) if isinstance(r, dict)]
    return {
        "photo_keys": photo_keys,
        "representative_photo_keys": list(ev.get("representative_photo_keys") or []),
        "issue_ids": list(cond.get("issue_ids") or []),
        "per_photo": {pk: photo_lineage(art, pk, listing["paths"]) for pk in photo_keys},
        "per_issue": [issue_lineage(art, r.get("photo_key"), r.get("issue_id"),
                                    r.get("observation"), idx)
                      for r in refs],
    }


# --------------------------------------------------------------------------- cases

def _human_truth_from_label(label: Dict[str, Any]) -> Dict[str, Any]:
    """v1.1 is the authority; v1 fields ride along for side-by-side only, and the
    re-tag answer is carried verbatim (it is validation, never a translation)."""
    return {"basis": "v1_1", "class_v1_1": label.get("class_v1_1"),
            "slug": label.get("slug"), "claim": label.get("claim"),
            "work": label.get("work"), "note": label.get("note"),
            "retag_answer": label.get("retag_answer"),
            "v1_verdict": label.get("v1_verdict"), "v1_class": label.get("v1_class"),
            "stored_terra_verdict": label.get("stored_terra_verdict"),
            "adjudication_card_id": label.get("adjudication_card_id"),
            "arm": label.get("arm")}


def _human_truth_from_v1(card: Dict[str, Any], verdict: str) -> Dict[str, Any]:
    """A card the v1.1 adjudication never re-asked: one collapsed label, no axes.

    Weaker basis, so the report caps these at medium confidence and reports them
    on their own row rather than pooling them with the adjudicated cohort."""
    return {"basis": "v1_only", "class_v1_1": None, "slug": None, "claim": None,
            "work": None, "note": (card.get("_v1_note") or None),
            "retag_answer": None, "v1_verdict": verdict, "v1_class": None,
            "stored_terra_verdict": (card.get("meta") or {}).get("terra_verdict"),
            "adjudication_card_id": None, "arm": None}


def _v5_claim(listing: Dict[str, Any], condition_id: str,
              claims: Dict[str, str]) -> Dict[str, Any]:
    idx = listing["idx"]
    cond = idx["conds"].get(condition_id) or {}
    rev = idx["revs"].get(condition_id) or {}
    disp = idx["disps"].get(condition_id) or {}
    item = cond.get("catalog_item_id")
    return {"condition_id": condition_id, "catalog_item_id": item,
            "catalog_kind": cond.get("catalog_kind"),
            "estimate_unit_id": cond.get("estimate_unit_id"),
            "scene_group": cond.get("scene_group"),
            "claim_text": claims.get(item, item),
            "terra_verdict": rev.get("verdict"), "terra_rationale": rev.get("rationale"),
            "disposition": disp.get("disposition"), "reason_code": disp.get("reason_code"),
            "accepted": rev.get("verdict") == "supported"
                        and disp.get("disposition") == "accepted_for_work"}


def condition_case(*, case_id: str, lane: str, card: Optional[Dict[str, Any]],
                   human_truth: Dict[str, Any], listings: Dict[Tuple[str, str], Dict[str, Any]],
                   source: str, property_key: str, condition_id: str,
                   claims: Dict[str, str], factorized: Dict[Tuple[str, str], List[str]],
                   attribute: bool = True) -> Dict[str, Any]:
    """One condition-anchored case, resolved to its exact frozen run or untraceable."""
    case: Dict[str, Any] = {
        "case_id": case_id,
        "lane": lane,
        "attribute": attribute,
        "source_ids": {"card_id": (card or {}).get("card_id"),
                       "adjudication_card_id": human_truth.get("adjudication_card_id"),
                       "gold_id": None, "gold_photo_key": None},
        "run_ref": {"source": source, "property_key": property_key,
                    "run_id": (card or {}).get("run_id"), "artifact_path": None,
                    "artifact_sha256": None},
        "human_truth": human_truth,
        "v5_claim": None,
        "lineage": None,
        "context": {"strata": (card or {}).get("strata") or [],
                    "direction": ((card or {}).get("meta") or {}).get("direction"),
                    "second_opinion": ((card or {}).get("meta") or {}).get("second_opinion"),
                    "photo_count": ((card or {}).get("meta") or {}).get("photo_count"),
                    "in_factorized_disagreements": factorized.get((property_key, condition_id)) or []},
        "mechanical_hints": {},
        "status": "pending",
        "untraceable_reason": None,
    }
    listing = listings.get((source, property_key))
    if listing is None:
        case["status"] = "untraceable"
        case["untraceable_reason"] = f"no complete-v5 run loaded for {source}/{property_key}"
        return case
    card_run = (card or {}).get("run_id")
    if card_run and card_run != listing["run_id"]:
        # condition_id is run-scoped: a different run is a different condition, never a substitute.
        case["status"] = "untraceable"
        case["untraceable_reason"] = (f"card names run {card_run}; the loaded run is "
                                      f"{listing['run_id']} (runs are not interchangeable)")
        return case
    case["run_ref"].update({"run_id": listing["run_id"],
                            "artifact_path": str(listing["path"]),
                            "artifact_sha256": sha256_file(listing["path"])})
    if condition_id not in listing["idx"]["conds"]:
        case["status"] = "untraceable"
        case["untraceable_reason"] = f"condition {condition_id} absent from run {listing['run_id']}"
        return case
    case["v5_claim"] = _v5_claim(listing, condition_id, claims)
    case["lineage"] = condition_lineage(listing, condition_id)
    case["mechanical_hints"] = _hints(case)
    return case


def _hints(case: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic observations, explicitly non-binding on the reviewer."""
    lin = case.get("lineage") or {}
    per_issue = lin.get("per_issue") or []
    joins = [i["p2b_join"]["method"] for i in per_issue]
    return {
        "condition_reached_terra": bool((case.get("v5_claim") or {}).get("terra_verdict")),
        "all_issues_joined_to_2b": bool(per_issue) and all(m != "none" for m in joins),
        "join_methods": sorted(set(joins)),
        "evidence_photo_count": len(lin.get("per_photo") or {}),
        "any_photo_missing_2a": any(not p.get("p2a_present")
                                    for p in (lin.get("per_photo") or {}).values()),
        "note": "mechanical only; the reviewer decides the stage",
    }


def gold_photo_record(*, listings: Dict[Tuple[str, str], Dict[str, Any]], property_key: str,
                      photo_key: str, findings: List[Dict[str, Any]],
                      claims: Dict[str, str]) -> Dict[str, Any]:
    """One gold photo: the frozen human findings beside every v5 condition on it.

    Matching gold findings to conditions is per-photo semantic work that needs the
    photo in hand, so this emits the evidence and leaves the mapping to review.
    Gold is photo-observation truth, not billable-condition truth (its own note
    keeps "technically-true conditions that are normal-for-context"), so a gold
    finding with no condition is only a miss once review says the catalog could
    have carried it."""
    listing = listings.get(("canary", property_key))
    rec: Dict[str, Any] = {
        "gold_photo_id": f"{property_key}/{photo_key}",
        "property_key": property_key, "photo_key": photo_key, "source": "canary",
        "gold_findings": [{"gold_id": f.get("gold_id"), "condition": f.get("condition")}
                          for f in findings],
        "run_ref": {"source": "canary", "property_key": property_key, "run_id": None,
                    "artifact_path": None, "artifact_sha256": None},
        "conditions_on_photo": [], "lineage": None,
        "status": "pending", "untraceable_reason": None,
    }
    if listing is None:
        rec.update(status="untraceable",
                   untraceable_reason=f"no canary run loaded for {property_key}")
        return rec
    rec["run_ref"].update({"run_id": listing["run_id"], "artifact_path": str(listing["path"]),
                           "artifact_sha256": sha256_file(listing["path"])})
    idx = listing["idx"]
    for ev in listing["result"].get("evidence_facts") or []:
        if photo_key not in (ev.get("photo_keys") or []):
            continue
        cid = ev.get("condition_id")
        claim = _v5_claim(listing, cid, claims)
        claim["photo_keys"] = list(ev.get("photo_keys") or [])
        claim["observations_on_this_photo"] = sorted(
            {str(r.get("observation")) for r in (ev.get("evidence_refs") or [])
             if isinstance(r, dict) and r.get("photo_key") == photo_key})
        rec["conditions_on_photo"].append(claim)
    rec["conditions_on_photo"].sort(key=lambda c: (not c["accepted"], c["catalog_item_id"] or ""))
    rec["lineage"] = photo_lineage(listing["artifact"], photo_key, listing["paths"])
    return rec


# --------------------------------------------------------------------------- selection

def select_cases(inputs: Dict[str, Any], listings: Dict[Tuple[str, str], Dict[str, Any]],
                 claims: Dict[str, str]) -> List[Dict[str, Any]]:
    """The deterministic cohort. Every rule is a property of the frozen labels."""
    labels, cards, verdicts = inputs["labels"], inputs["cards"], inputs["verdicts"]
    factorized = inputs["factorized"]
    cases: List[Dict[str, Any]] = []

    def add(card_id: str, lane: str, truth: Dict[str, Any], *, source: str,
            property_key: str, condition_id: str, attribute: bool = True) -> None:
        cases.append(condition_case(
            case_id=card_id, lane=lane, card=cards.get(card_id), human_truth=truth,
            listings=listings, source=source, property_key=property_key,
            condition_id=condition_id, claims=claims, factorized=factorized,
            attribute=attribute))

    # -- v1.1-labelled lanes -------------------------------------------------
    for card_id, label in sorted(labels.items()):
        klass = label.get("class_v1_1")
        if klass in MISS_CLASSES:
            lane = "miss_label"
        elif klass == "hard_false_billed" or label.get("claim") == "absent":
            lane = "halluc_label"
        elif klass in APPENDIX_CLASS_LANE:
            lane = APPENDIX_CLASS_LANE[klass]
        else:
            continue  # supported_billed: the human agreed with the pipeline
        add(card_id, lane, _human_truth_from_label(label), source=label.get("source"),
            property_key=label.get("property_key"), condition_id=label.get("condition_id"))

    # -- v1-only lanes: condition cards the v1.1 adjudication never re-asked ---
    for card_id, card in sorted(cards.items()):
        if card.get("kind") != "condition" or card_id in labels:
            continue
        rec = verdicts.get(card_id)
        if not rec:
            continue
        verdict, meta = rec.get("verdict"), card.get("meta") or {}
        accepted = bool(meta.get("accepted"))
        card = dict(card, _v1_note=rec.get("notes"))
        if verdict == "terra_claim_unsupported" and accepted:
            lane, attribute = "halluc_v1only", True      # billed a claim the human rejected
        elif verdict == "terra_claim_supported" and not accepted:
            lane, attribute = "miss_v1only", True        # a real condition the pipeline dropped
        elif verdict == "terra_evidence_inconclusive":
            lane, attribute = "appendix_inconclusive", True
        elif verdict == "terra_claim_unsupported" and not accepted:
            lane, attribute = "counted_correct_rejection", False
        elif verdict == "terra_claim_supported" and accepted:
            lane, attribute = "counted_agreement", False
        else:
            lane, attribute = "counted_agreement", False
        cases.append(condition_case(
            case_id=card_id, lane=lane, card=card,
            human_truth=_human_truth_from_v1(card, verdict), listings=listings,
            source=card.get("source"), property_key=card.get("property_key"),
            condition_id=meta.get("condition_id"), claims=claims, factorized=factorized,
            attribute=attribute))

    return cases


def orphan_cases(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Human judgments the published rates excluded (mostly thumbnail evidence).

    Recorded so the audit accounts for every verdict, never attributed: the
    reviewer cannot judge a claim on a photo that was excluded as unusable."""
    out = []
    for orphan in inputs["orphans"]:
        out.append({
            "case_id": orphan.get("card_id"), "lane": "counted_orphan", "attribute": False,
            "source_ids": {"card_id": orphan.get("card_id"), "adjudication_card_id": None,
                           "gold_id": None, "gold_photo_key": None},
            "run_ref": {"source": orphan.get("source"), "property_key": orphan.get("property_key"),
                        "run_id": orphan.get("run_id"), "artifact_path": None,
                        "artifact_sha256": None},
            "human_truth": {"basis": "v1_only", "v1_verdict": orphan.get("verdict"),
                            "class_v1_1": None, "slug": None, "claim": None, "work": None,
                            "note": None, "retag_answer": None, "v1_class": None,
                            "stored_terra_verdict": None, "adjudication_card_id": None,
                            "arm": None},
            "v5_claim": None, "lineage": None,
            "context": {"orphan_classification": orphan.get("classification"),
                        "strata": [], "direction": None, "second_opinion": None,
                        "photo_count": None, "in_factorized_disagreements": []},
            "mechanical_hints": {},
            "status": "untraceable",
            "untraceable_reason": f"orphan verdict ({orphan.get('classification')})",
        })
    return out


# --------------------------------------------------------------------------- build

def build_queue(inputs: Dict[str, Any], listings: Dict[Tuple[str, str], Dict[str, Any]],
                claims: Dict[str, str]) -> Dict[str, Any]:
    cases = select_cases(inputs, listings, claims) + orphan_cases(inputs)
    gold_photos = []
    for gold_key, findings in sorted((inputs["gold"] or {}).items()):
        property_key, photo_key = gold_key.split("/", 1)
        gold_photos.append(gold_photo_record(listings=listings, property_key=property_key,
                                             photo_key=photo_key, findings=findings,
                                             claims=claims))
    lanes: Dict[str, int] = {}
    for case in cases:
        lanes[case["lane"]] = lanes.get(case["lane"], 0) + 1
    joins: Dict[str, int] = {}
    for case in cases:
        for issue in ((case.get("lineage") or {}).get("per_issue") or []):
            method = issue["p2b_join"]["method"]
            joins[method] = joins.get(method, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs["shas"],
        "roots": {"canary": str(rc.CANARY_ROOT), "production": str(rc.PROD_ROOT),
                  "listings_loaded": len(listings)},
        "lane_counts": dict(sorted(lanes.items())),
        "attributable_cases": sum(1 for c in cases if c["attribute"]),
        "p2b_join_health": dict(sorted(joins.items())),
        "gold_photo_count": len(gold_photos),
        "gold_finding_count": sum(len(g["gold_findings"]) for g in gold_photos),
        "stages": list(STAGES),
        "notes": [
            "Attribution granularity stops at photo + 2b bullet: neither 2a nor 2b carries "
            "observation ids, so a sentence-level 2a excerpt is the reviewer's inference.",
            "2c records no rationale for what it dropped; p2c_dropped is a set difference.",
            "Gold is photo-observation truth, not billable-condition truth; an unmatched gold "
            "finding is a miss only once review judges the catalog could have carried it.",
        ],
        "cases": cases,
        "gold_photos": gold_photos,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    inputs = load_inputs()
    listings = load_listings()
    claims = rc.claim_texts()
    queue = build_queue(inputs, listings, claims)
    atomic_json(args.out, queue)

    print(f"listings loaded: {queue['roots']['listings_loaded']}")
    print(f"cases: {len(queue['cases'])} (attributable {queue['attributable_cases']})")
    for lane, n in queue["lane_counts"].items():
        print(f"  {lane}: {n}")
    print(f"2b join health: {queue['p2b_join_health']}")
    print(f"gold: {queue['gold_photo_count']} photos, {queue['gold_finding_count']} findings")
    untraceable = [c["case_id"] for c in queue["cases"] if c["status"] == "untraceable"]
    print(f"untraceable: {len(untraceable)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
