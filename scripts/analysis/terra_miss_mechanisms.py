"""Offline Session A census and six-case evidence packet; never calls a model.

Run from any directory: python scripts/analysis/terra_miss_mechanisms.py
The lexical signals are screening proxies, not semantic or causal judgments.
Raw clauses and observations accompany every flag so it can be audited.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import review_cards as rc
from tools.renovation_architecture.terra_review import _claim_text, OBSERVATION_MAX_CHARS
from tools.renovation_architecture.evidence import compute_photo_identity

CANARY = ROOT / "artifacts_canary/renovation_session9_20260818"
PRODUCTION = Path("C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts")
FACTOR = ROOT / "artifacts_canary/factorized_v1_20260831"
SIX = {
    "rc_790b583fcbf0": "neutral_paint",
    "rc_905575ed8a3d": "porch_connection_vs_soffit_panels",
    "rc_a35f1b3a231e": "siding_and_trim",
    "rc_ace7463d6837": "shower_hardware_vs_enclosure",
    "rc_d0197df43080": "floor_wear_dispute",
    "rc_eace19540d2d": "cabinet_components",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def provenance(paths):
    return [{"path": relative(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
            for p in sorted(set(map(Path, paths)), key=relative)]


def write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def table(headers, rows):
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |"] +
                      ["| " + " | ".join(map(cell, r)) + " |" for r in rows])


def load_inputs():
    cards = read(ROOT / "reports/review_queue.json")["cards"]
    labels = read(ROOT / "reports/labels_v1_1.json")["labels"]
    verdicts = rc.latest_verdicts(ROOT / "reports/review_verdicts.jsonl")
    runs, _ = rc.load_canary(CANARY)
    if len(runs) != 18:
        raise ValueError(f"Expected 18 run_1 properties, got {len(runs)}")
    # Exact queue run, never newest-run substitution for reviewed production cards.
    artifacts = {}
    for prop, (path, artifact) in sorted(runs.items()):
        artifacts[("canary", prop, path.parent.name)] = (path, artifact)
    for card in cards:
        key = (card["source"], card["property_key"], card["run_id"])
        if key not in artifacts:
            base = PRODUCTION if key[0] == "production" else CANARY / "run_1/candidate"
            path = base / key[1] / key[2] / "photo_intel_debug.json"
            artifacts[key] = (path, read(path))
    return cards, labels, verdicts, artifacts


def condition_row(source, prop, path, artifact, cid, observables):
    result = rc.v5_result(artifact)
    if result is None:
        raise ValueError(f"Incomplete v5: {path}")
    idx = rc.index_result(result)
    condition, review, evidence = idx["conds"][cid], idx["revs"][cid], idx["evs"][cid]
    call = idx["calls"][review["terra_call_id"]]
    if cid not in call["condition_ids"] or review["request_fingerprint"] != call["request_fingerprint"]:
        raise ValueError(f"Broken Terra call join: {path} {cid}")
    batch_photos = sorted({p for c in call["condition_ids"] for p in idx["evs"][c]["representative_photo_keys"]})
    item = condition["catalog_item_id"]
    observable = observables[item]
    observations = sorted({e["observation"][:OBSERVATION_MAX_CHARS]
                           for e in evidence["evidence_refs"] if e.get("observation")})
    disp = idx["disps"][cid]
    return {
        "source": source, "property_key": prop, "run_id": path.parent.name,
        "artifact_path": relative(path), "condition_id": cid, "catalog_item_id": item,
        "catalog_kind": condition["catalog_kind"], "estimate_unit_id": condition["estimate_unit_id"],
        "claim": _claim_text(observable, item), "atomic_claim": observable.get("atomic_claim"),
        "observations": observations, "verdict": review["verdict"], "rationale": review["rationale"],
        "distinct_view_count": evidence["distinct_view_count"], "evidence_photo_keys": evidence["photo_keys"],
        "representative_photo_keys": evidence["representative_photo_keys"], "call_id": call["call_id"],
        "batch_conditions": len(call["condition_ids"]), "batch_photos": len(batch_photos),
        "batch_photo_keys": batch_photos, "request_fingerprint": call["request_fingerprint"],
        "call_budget_debited_tokens": call["budget_debited_tokens"],
        "disposition": disp, "active_work_item": idx["work_by_cond"].get(cid),
    }


# Explicit morphology families, independent of labels. Only words actually in
# the catalog claim are candidates; labels never drive phrase extraction.
TERM_FAMILIES = (
    "peel peeling peeled", "bubble bubbling bubbled", "age aged aging older tired",
    "scratch scratches scratched scratching", "scuff scuffs scuffed scuffing",
    "wear worn weathered weathering", "fade fading faded", "chalk chalking chalky",
    "discolor discoloration discolored", "stain stains stained staining",
    "crack cracks cracked cracking", "chip chips chipped chipping",
    "damage damaged damaging", "break breaks broken", "sag sagging sagged",
    "loose looseness loosened", "miss missing absent", "water", "vinyl", "linoleum",
    "hardwood", "wood wooden", "carpet carpeting", "tile tiled", "valance valances",
    "dated outdated", "uneven unevenness", "warp warped warping", "rot rotted rotting",
)
ALIASES = {word: family.split()[0] for family in TERM_FAMILIES for word in family.split()}
NEGATION = re.compile(r"\b(?:no|not|without|neither|cannot|can't|doesn't|isn't|aren't)\b", re.I)


def terms(text):
    return {ALIASES[w] for w in re.findall(r"[a-z]+", text.lower()) if w in ALIASES}


def signals(row):
    claim = terms(row["claim"])
    observed = terms(" ".join(row["observations"]))
    clauses = re.split(r"[.;!?]|\bbut\b|\bhowever\b", row["rationale"], flags=re.I)
    negated = []
    for clause in clauses:
        match = NEGATION.search(clause)
        if match:
            # Restrict to words after negation; never count the preceding subject.
            found = terms(clause[match.start():]) & claim
            if found:
                negated.append({"clause": clause.strip(), "claim_terms": sorted(found)})
    all_negated = {t for x in negated for t in x["claim_terms"]}
    absent = all_negated - observed
    shared = all_negated & observed
    return {"adjacent_any": bool(absent), "adjacent_only": bool(absent) and not shared,
            "adjacent_terms": sorted(absent), "shared_refuted_terms": sorted(shared),
            "negated_clauses": negated,
            "claim_observation_term_disjoint": bool(claim and observed and not claim & observed)}


def bucket(n, axis):
    if axis == "batch_conditions":
        return "1" if n == 1 else "2-4" if n <= 4 else "5-8" if n <= 8 else "9+"
    return "1" if n == 1 else "2-3" if n <= 3 else "4+"


def summarize(rows):
    return {"n": len(rows), "properties": len({r["property_key"] for r in rows}),
            "units": len({(r["property_key"], r["call_id"]) for r in rows}),
            "verdicts": dict(Counter(r["verdict"] for r in rows)),
            "human": dict(Counter(r.get("human_group", "unlabelled") for r in rows))}


def feature_tables(rows):
    out = {}
    for axis in ("adjacent_any", "adjacent_only", "claim_observation_term_disjoint",
                 "batch_conditions", "batch_photos"):
        groups = defaultdict(list)
        for r in rows:
            value = bucket(r[axis], axis) if axis.startswith("batch_") else str(r["signals"][axis]).lower()
            groups[value].append(r)
        out[axis] = {k: summarize(v) for k, v in sorted(groups.items())}
    return out


def historical_catalog():
    raw = subprocess.check_output(["git", "show", "07ee112:tools/issue_catalog_kind_v2.json"], cwd=ROOT)
    catalog = json.loads(raw)
    return {i["id"]: i for i in catalog["items"]}, {
        "ref": "07ee112:tools/issue_catalog_kind_v2.json", "version": catalog["version"],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def run():
    cards, labels, verdicts, artifacts = load_inputs()
    observables, catalog_source = historical_catalog()
    census, all_rows = [], {}
    sources = [p for p, _ in artifacts.values()]
    for (source, prop, run_id), (path, artifact) in sorted(artifacts.items()):
        result = rc.v5_result(artifact)
        if result is None:
            raise ValueError(f"Missing v5 at {path}")
        for review in result["condition_reviews"]:
            row = condition_row(source, prop, path, artifact, review["condition_id"], observables)
            row["signals"] = signals(row)
            all_rows[(source, prop, run_id, row["condition_id"])] = row
            if source == "canary":
                census.append(row)
    if len(census) != 1047:
        raise ValueError(f"Expected 1047 run_1 reviews, got {len(census)}")
    dirb, excluded = [], []
    for card in cards:
        if "dirB" not in card["strata"]:
            continue
        verdict = verdicts.get(card["card_id"], {}).get("verdict")
        if verdict not in ("terra_claim_supported", "terra_claim_unsupported"):
            excluded.append({"card_id": card["card_id"], "verdict": verdict,
                             "reason": "inconclusive human label; no binary truth control"})
            continue
        key = (card["source"], card["property_key"], card["run_id"], card["meta"]["condition_id"])
        row = dict(all_rows[key])
        row.update(card_id=card["card_id"], human_group="miss" if verdict == "terra_claim_supported" else "correct_rejection",
                   v1_verdict=verdict, label_v1_1=labels.get(card["card_id"]), six_case=card["card_id"] in SIX)
        dirb.append(row)
    if len(dirb) != 47:
        raise ValueError(f"Expected 47 binary dirB cards, got {len(dirb)}")
    error_queue = ROOT / "reports/error_attribution_queue.json"
    errors = {x["case_id"]: x for x in read(error_queue)["cases"]}
    retags = rc.latest_verdicts(ROOT / "reports/retag_verdicts.jsonl")
    adjudications = rc.latest_verdicts(ROOT / "reports/adjudication_verdicts.jsonl")
    attribution_path = ROOT / "reports/error_attribution_verdicts.jsonl"
    attributions = {r["case_id"]: r for r in (
        json.loads(line) for line in attribution_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )}
    six = []
    for row in sorted(dirb, key=lambda r: r["card_id"]):
        cid = row["card_id"]
        if cid not in SIX:
            continue
        c = errors[cid]
        matches = []
        for path in sorted((FACTOR / row["property_key"] / "units").glob("*.json")):
            unit = read(path)
            if row["condition_id"] in unit["condition_ids"]:
                matches.append((path, unit))
        if len(matches) != 1:
            raise ValueError(f"Expected one factorized unit for {cid}")
        path, unit = matches[0]
        sources.append(path)
        if unit["stored_request_fingerprint"] != row["request_fingerprint"] or not unit["fingerprint_match"]:
            raise ValueError(f"Six-case fingerprint mismatch: {cid}")
        from PIL import Image
        photos = []
        for photo_key, photo in sorted(c["lineage"]["per_photo"].items()):
            image_path = Path(photo["image_path"])
            sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            pixel_sha = compute_photo_identity(image_path).exact_sha256
            with Image.open(image_path) as im:
                size = list(im.size)
            photos.append({"key": photo_key, "path": relative(image_path), "size": size,
                           "encoded_file_sha256": sha, "normalized_rgb_sha256": pixel_sha,
                           "matches_factorized_pixels": pixel_sha == unit["image_sha256"].get(photo_key)})
        adj = adjudications[c["human_truth"]["adjudication_card_id"]]
        six.append({**row, "case_group": SIX[cid], "factorized_path": relative(path),
                    "factorized_review": unit["reviews"][row["condition_id"]],
                    "fingerprint_match": True, "photos": photos,
                    "human_review": verdicts[cid], "retag": retags[cid], "adjudication": adj,
                    "attribution_record": attributions[cid],
                    "weak_label": {"nonblind_reask": True, "notes_absent": not adj.get("notes"),
                                   "attribution_skeptic_checked": attributions[cid].get("skeptic_checked")},
                    "per_issue_lineage": c["lineage"]["per_issue"]})
    sources += [ROOT / "reports" / n for n in ("review_queue.json", "review_verdicts.jsonl", "labels_v1_1.json",
                                               "retag_verdicts.jsonl", "adjudication_verdicts.jsonl")]
    sources += [error_queue, attribution_path, ROOT / "tools/issue_catalog_kind_v2.json", Path(__file__)]
    return {"schema_version": 1, "model_calls": 0, "historical_catalog": catalog_source,
            "sources": provenance(sources), "lexical_method": {
                "term_families": TERM_FAMILIES, "negation_regex": NEGATION.pattern,
                "limits": "Lexical screening only; negation scope and synonyms are imperfect. An alternative term absent from an observation does not establish that Terra ignored the observed condition."
            }, "census_summary": summarize(census), "census_tables": feature_tables(census),
            "dirb_summary": summarize(dirb), "dirb_tables": feature_tables(dirb),
            "dirb_without_six_tables": feature_tables([r for r in dirb if not r["six_case"]]),
            "dirb_by_source": {s: feature_tables([r for r in dirb if r["source"] == s]) for s in ("canary", "production")},
            "inconclusive_dirb": excluded, "six_cases": six, "dirb_rows": dirb,
            "census_rows": sorted(census, key=lambda r: (r["property_key"], r["condition_id"]))}


def markdown(doc):
    lines = ["# Terra miss mechanism census", "", "Offline, zero model calls. Counts are cohort outcomes, not population recall or causal effects.",
             "", f"Run_1: {doc['census_summary']['n']} reviews. Binary dirB: {doc['dirb_summary']['n']} cards; four inconclusive cards retained separately.", ""]
    for name in ("dirb_tables", "dirb_without_six_tables"):
        lines += [f"## {name}", "", table(["Feature", "Group", "Cards", "Properties", "Units", "Human miss", "Correct rejection"],
            [(axis, group, v["n"], v["properties"], v["units"], v["human"].get("miss", 0), v["human"].get("correct_rejection", 0))
             for axis, groups in doc[name].items() for group, v in groups.items()]), ""]
    lines += ["## Six cases", "", table(["Card", "Item", "Group", "Conditions/photos in call", "Terra", "Factorized"],
        [(r["card_id"], r["catalog_item_id"], r["case_group"], f"{r['batch_conditions']}/{r['batch_photos']}",
          r["verdict"], r["factorized_review"]["derived_class"]) for r in doc["six_cases"]]), "",
        "The JSON includes every joined row, matched negation clause, raw human answer, source hash and six-case image fingerprint.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/terra_miss_mechanisms_20260910.json")
    args = parser.parse_args()
    doc = run()
    write(args.out, doc)
    args.out.with_suffix(".md").write_text(markdown(doc), encoding="utf-8", newline="\n")
    print(json.dumps({"census": doc["census_summary"], "dirb": doc["dirb_summary"], "out": relative(args.out)}))


if __name__ == "__main__":
    main()
