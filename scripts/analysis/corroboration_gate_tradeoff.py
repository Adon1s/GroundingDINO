"""Measure a two-distinct-view gate on the frozen reviewed cohort, offline.

No catalog mutation, model call, population precision/recall or price estimate.
--families accepts a comma-separated item list; the default is the union of
RESULT error-attribution section 9.4 and review_analysis section 3 by-item rows,
minus D1 trim and the retired CAP-007 parent. Old parent cards stay explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.terra_miss_mechanisms import (
    load_inputs, provenance, read, relative, table, write,
)
from tools import review_cards as rc
from tools.renovation_architecture.disposition import decide_disposition

DEFAULT_FAMILIES = tuple(sorted({
    "older_flooring_style", "cabinets_dated_style", "worn_or_stained_flooring",
    "peeling_or_discolored_paint", "hard_flooring_scratched_or_worn", "worn_or_stained_carpet",
    "baseboard_wear_scuffs", "floor_dirty_or_heavily_soiled", "bath_fixtures_stained_or_worn",
    "outdated_bathroom_finishes", "patio_or_porch_surface_wear", "wall_scuffs_marks_or_dents",
}))
TRIVIAL_ITEMS = {"older_flooring_style", "cabinets_dated_style", "dated_interior_doors"}
CLASSES = ("absent", "misnamed", "trivial", "supported_and_warranted", "inconclusive")


def label_class(label):
    if label["claim"] == "absent":
        return "absent"
    if label["claim"] == "misnamed":
        return "misnamed"
    if label["work"] == "trivial":
        return "trivial"
    if label["claim"] == "exact" and label["work"] == "warranted":
        return "supported_and_warranted"
    return "inconclusive"


def gate_outcome(verdict, views, selected, accepted):
    # Reuse production ordering: an unsupported/cannot_assess condition does
    # not become withheld. Report threshold failure independently of routing.
    if not selected:
        return False, "item_not_selected"
    disposition, reason = decide_disposition(verdict, "work", views, 2)
    if not accepted:
        return False, "already_not_billed:" + reason
    return disposition == "withheld", reason


def summarize(rows):
    out = {}
    for klass in CLASSES:
        group = [r for r in rows if r["label_class"] == klass]
        out[klass] = {"withheld": sum(r["newly_withheld"] for r in group),
                      "not_withheld": sum(not r["newly_withheld"] for r in group),
                      "below_two_views": sum(r["below_two_views"] for r in group),
                      "already_not_billed": sum(not r["accepted"] for r in group),
                      "cards": len(group)}
    return out


def run(families):
    cards, labels, _, artifacts = load_inputs()
    card_index = {c["card_id"]: c for c in cards}
    error_path = ROOT / "reports/error_attribution_queue.json"
    errors = read(error_path)["cases"]
    halluc = {c["case_id"]: c for c in errors
              if c["case_id"].startswith("rc_") and c["lane"].startswith("halluc_")}
    if len(halluc) != 12:
        raise ValueError(f"Expected 12 rc_ hallucination cases, got {len(halluc)}")
    current_path = ROOT / "tools/issue_catalog_kind_v2.json"
    current_items = {i["id"]: i for i in read(current_path)["items"]}
    unknown = set(families) - current_items.keys()
    if unknown:
        raise ValueError(f"Unknown or retired gate item(s): {sorted(unknown)}")
    reviewed_keys = set()
    rows = []
    for card_id in sorted(set(labels) | set(halluc)):
        c = card_index[card_id]
        label = labels.get(card_id)
        if label is None:
            label = halluc[card_id]["human_truth"]
        key = (c["source"], c["property_key"], c["run_id"])
        path, artifact = artifacts[key]
        idx = rc.index_result(rc.v5_result(artifact))
        cid = c["meta"]["condition_id"]
        condition, evidence, review = idx["conds"][cid], idx["evs"][cid], idx["revs"][cid]
        item = condition["catalog_item_id"]
        if card_id in labels and (labels[card_id]["condition_id"] != cid or labels[card_id]["catalog_item_id"] != item):
            raise ValueError(f"Label/condition join drift: {card_id}")
        views = evidence["distinct_view_count"]
        if not isinstance(views, int) or views < 0:
            raise ValueError(f"Missing distinct_view_count: {path} {cid}")
        accepted = idx["disps"][cid]["disposition"] == "accepted_for_work"
        selected = item in families
        withheld, reason = gate_outcome(review["verdict"], views, selected, accepted)
        row = {"card_id": card_id, "source": c["source"], "property_key": c["property_key"],
               "run_id": c["run_id"], "condition_id": cid, "catalog_item_id": item,
               "artifact_path": relative(path), "label_class": label_class(label),
               "claim": label["claim"], "work": label["work"], "label_v1_1": label,
               "cohorts": (["labels_v1_1"] if card_id in labels else []) + (["rc_hallucination"] if card_id in halluc else []),
               "distinct_view_count": views, "distinct_photo_count": evidence["distinct_photo_count"],
               "photo_keys": evidence["photo_keys"], "representative_photo_keys": evidence["representative_photo_keys"],
               "below_two_views": views < 2, "gate_selected": selected, "terra_verdict": review["verdict"],
               "accepted": accepted, "newly_withheld": withheld, "gate_reason": reason,
               "stored_disposition": idx["disps"][cid], "active_work_item_id": (idx["work_by_cond"].get(cid) or {}).get("work_item_id"),
               "policy_note": "D1: already no_action in 3.2" if item == "dated_interior_trim" else
                              "CAP-007 parent retired; successor requires live 2d, no inferred remap" if item == "dated_window_treatment_valance" else None}
        rows.append(row)
        reviewed_keys.add((*key, cid))
    unreviewed = []
    for (source, prop, run_id), (path, artifact) in sorted(artifacts.items()):
        if source != "canary":
            continue
        idx = rc.index_result(rc.v5_result(artifact))
        for cid, condition in sorted(idx["conds"].items()):
            evidence = idx["evs"][cid]
            if evidence["distinct_view_count"] != 1 or idx["disps"][cid]["disposition"] != "accepted_for_work":
                continue
            if (source, prop, run_id, cid) in reviewed_keys:
                continue
            unreviewed.append({"property_key": prop, "run_id": run_id, "condition_id": cid,
                               "catalog_item_id": condition["catalog_item_id"], "distinct_view_count": 1,
                               "photo_keys": evidence["photo_keys"], "selected": condition["catalog_item_id"] in families,
                               "artifact_path": relative(path)})
    by_item = defaultdict(list)
    for row in rows:
        by_item[row["catalog_item_id"]].append(row)
    # Reproduce the published six/six item-exclusion inventory separately;
    # it was not a measurement of a distinct-view gate.
    inventory = [r for r in rows if r["catalog_item_id"] in TRIVIAL_ITEMS and r["accepted"] and "labels_v1_1" in r["cohorts"]]
    source_paths = [p for p, _ in artifacts.values()] + [error_path, current_path, Path(__file__),
        Path(__file__).with_name("terra_miss_mechanisms.py")]
    source_paths += [ROOT / "reports" / n for n in ("review_queue.json", "review_verdicts.jsonl", "labels_v1_1.json")]
    return {"schema_version": 1, "model_calls": 0, "families": sorted(families), "min_photo_evidence": 2,
            "scope": "Frozen reviewed outcomes, not population precision/recall; distinct views measure coverage, not condition corroboration. Package and dollar effects require a replay.",
            "sources": provenance(source_paths), "coverage": {
                "labels_v1_1": sum("labels_v1_1" in r["cohorts"] for r in rows),
                "rc_hallucination": sum("rc_hallucination" in r["cohorts"] for r in rows),
                "unique_cards": len(rows), "unresolved": 0},
            "summary": summarize(rows),
            "by_source": {s: summarize([r for r in rows if r["source"] == s]) for s in ("canary", "production")},
            "by_item": {item: {"selected": item in families, "counts": summarize(rs)} for item, rs in sorted(by_item.items())},
            "three_item_blanket_exclusion_inventory": {"items": sorted(TRIVIAL_ITEMS),
                "counts": dict(Counter(r["label_class"] for r in inventory)), "card_ids": [r["card_id"] for r in inventory],
                "note": "All billed cards on these three items, irrespective of distinct views; not the gate's measured effect."},
            "rows": rows, "unlabelled_single_view_billed": unreviewed,
            "unlabelled_single_view_by_item": dict(sorted(Counter(r["catalog_item_id"] for r in unreviewed).items()))}


def markdown(doc):
    lines = ["# Two-view gate: reviewed-cohort tradeoff", "", doc["scope"], "",
             "Each cell is **newly withheld / not withheld**. Only currently accepted conditions can be newly withheld. All original cards remain in the JSON.", "",
             table(["Item", "Selected", *CLASSES], [(item, v["selected"], *[
                 f"{v['counts'][k]['withheld']} / {v['counts'][k]['not_withheld']}" for k in CLASSES])
                 for item, v in doc["by_item"].items()]), "",
             "## Totals", "", table(["Class", "Withheld", "Not withheld", "Already not billed"],
                 [(k, v["withheld"], v["not_withheld"], v["already_not_billed"]) for k, v in doc["summary"].items()]), "",
             "## Unlabelled single-view billed conditions in run_1", "",
             table(["Item", "Conditions", "Selected"], [(k, v, k in doc["families"]) for k, v in doc["unlabelled_single_view_by_item"].items()]), "",
             "## Three-item blanket-exclusion cross-check", "",
             str(doc["three_item_blanket_exclusion_inventory"]), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--out", type=Path, default=ROOT / "reports/corroboration_gate_tradeoff_20260910.json")
    args = parser.parse_args()
    doc = run(set(filter(None, (s.strip() for s in args.families.split(",")))))
    write(args.out, doc)
    args.out.with_suffix(".md").write_text(markdown(doc), encoding="utf-8", newline="\n")
    print(doc["coverage"])
    print(doc["summary"])


if __name__ == "__main__":
    main()
