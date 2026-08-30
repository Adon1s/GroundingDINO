"""Package-outcome scoring on canonical physical rooms (v3 lineage).

Predictions are aligned to gold's canonical rooms by supporting-photo
overlap before any matching, so ordinal unit drift (bedroom_2 vs bedroom_3)
can no longer flip credit between physical rooms. Each (cell, property,
repeat) gets a full diagnostic record — family recall, signed tier distance,
component and exact-ID coverage, extras — with strict completion retained as
a secondary verdict that diagnostic_only gold targets can never fail.

Cap comparisons are paired by (property, repeat); Pass 2f disagreements on
equivalent evidence are attributed to 2f noise, never to the cap.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

from tools import benchmark_pass2a_packages as pkg
from tools.benchmark_pass2a import _load_json, jaccard


# ---------------------------------------------------------------------------
# Canonical-room alignment
# ---------------------------------------------------------------------------

def unit_room_family(unit_id: str) -> Optional[str]:
    """Room family from a predicted estimate-unit id (bedroom_2 -> bedroom,
    living_room_primary -> living, basement_primary -> utility). None when
    unknown — alignment then considers every canonical room and photo
    overlap alone decides."""
    import re
    from tools.benchmark_pass2a_reachability import ROOM_FAMILY_SCENE_GROUPS
    from tools.pipeline_common import SCENE_TO_GROUP_UI
    from tools.rehab_packages import _normalize_scene_to_room
    base = re.sub(r"_(primary|secondary(_\d+)?|\d+)$", "", str(unit_id or ""))
    group = SCENE_TO_GROUP_UI.get(base)
    family = "living" if group == "living_areas" else group
    if family in ROOM_FAMILY_SCENE_GROUPS:
        return family
    family = _normalize_scene_to_room(base)
    return family if family in ROOM_FAMILY_SCENE_GROUPS else None


def align_room_for(photo_keys: Set[str], family: Optional[str],
                   rooms: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Align one predicted unit to a canonical room by photo overlap.

    Candidates are restricted to the unit's room family when known; a unique
    positive best overlap aligns; ties and zero overlap stay unmatched with
    an explicit diagnostic."""
    candidates = [r for r in rooms
                  if family is None or r.get("room") == family]
    scored = [(len(photo_keys & set(r.get("photo_keys") or [])), r["room_id"])
              for r in candidates]
    positive = [(n, rid) for n, rid in scored if n > 0]
    if not positive:
        return {"status": "no_overlap", "room_id": None, "overlap": 0,
                "candidates": []}
    best = max(n for n, _ in positive)
    winners = sorted(rid for n, rid in positive if n == best)
    if len(winners) > 1:
        return {"status": "tie", "room_id": None, "overlap": best,
                "candidates": winners}
    return {"status": "aligned", "room_id": winners[0], "overlap": best,
            "candidates": winners}


def align_units_to_rooms(unit_photos: Dict[str, Set[str]],
                         rooms: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        unit: align_room_for(photos, unit_room_family(unit), rooms)
        for unit, photos in sorted(unit_photos.items())
    }


def _unit_photo_map(final: Dict[str, Any]) -> Dict[str, Set[str]]:
    return {
        str(u.get("estimate_unit_id")): set(u.get("photo_ids") or [])
        for u in final.get("estimate_units") or []
        if u.get("estimate_unit_id")
    }


# ---------------------------------------------------------------------------
# Tier distance (signed steps along the escalation chain)
# ---------------------------------------------------------------------------

def tier_distance(target: str, observed: Optional[str]) -> Optional[int]:
    """observed minus target in escalation steps: positive = over-tiered,
    negative = under-tiered, None = missing or on a disconnected chain."""
    if not observed:
        return None
    if observed == target:
        return 0
    from tools.rehab_packages import _PRICING_TIER_ESCALATION
    up = {low: spec[0] for low, spec in _PRICING_TIER_ESCALATION.items()}
    cur, steps = target, 0
    while cur in up:
        cur, steps = up[cur], steps + 1
        if cur == observed:
            return steps
    cur, steps = observed, 0
    while cur in up:
        cur, steps = up[cur], steps + 1
        if cur == target:
            return -steps
    return None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _match_expected(expected: Dict[str, Any], approved: List[Dict[str, Any]],
                    room_of_unit: Dict[str, Optional[str]],
                    adjacency: Dict[str, Set[str]]) -> Tuple[str, Optional[str], Optional[str]]:
    """(status, observed_profile, observed_unit) against room-aligned
    approved packages."""
    target = expected["target_pricing_profile"]
    for p in sorted(approved, key=lambda p: str(p.get("estimate_unit_id"))):
        if p.get("package_type") != expected["package_type"]:
            continue
        if room_of_unit.get(str(p.get("estimate_unit_id"))) != expected["room_id"]:
            continue
        profile = p.get("pricing_profile")
        unit = p.get("estimate_unit_id")
        if profile == target:
            return "matched_exact", profile, unit
        if expected.get("adjacent_acceptable", True) and \
                profile in adjacency.get(target, set()):
            return "matched_adjacent", profile, unit
        return "wrong_tier", profile, unit
    return "missing", None, None


def _component_of(cid: str, trade_by_id: Dict[str, str]) -> Optional[str]:
    from tools.rehab_packages import classify_component
    return classify_component({"catalog_item_id": cid,
                               "trade_bucket": trade_by_id.get(cid) or ""})


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_package_round(config: Dict[str, Any], manifest: Dict[str, Any],
                        round_label: str, gold: Dict[str, Any],
                        decisions: Dict[str, Any]) -> Dict[str, Any]:
    from tools import pipeline_config as cfg
    from tools.artifact_writers import load_issue_catalog

    variant_a, variant_b = pkg.package_round_variants(config, round_label)
    cells = pkg.cells_for_round(variant_a, variant_b)
    repeats = int(config["repeats"])
    gates = pkg.package_eval_config(config).get("gates") or {}
    pass_reps = int(gates.get("pass_reps_required", 2))
    adjacency = pkg.pricing_profile_adjacency()
    gold_sha = pkg.package_gold_sha(gold)
    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)
    trade_by_id = {i.get("id"): i.get("trade_bucket") or ""
                   for i in catalog.get("items") or []}

    outcomes: Dict[str, Any] = {}
    review_rows: Dict[str, Dict[str, Any]] = {}
    pending: Dict[str, Set[str]] = {"extras_pending": set(), "gold_gap": set(),
                                    "needs_rereview": set()}
    # target -> {"policy", "occurrences", "cells": {cell: [reps]}}
    distinct_missing: Dict[str, Dict[str, Any]] = {}

    def _record_missing(prop: str, kind: str, gid: str, policy: str,
                        cell: str, rep: int) -> None:
        row = distinct_missing.setdefault(f"{prop}|{kind}|{gid}", {
            "policy": policy, "occurrences": 0, "cells": {}})
        row["occurrences"] += 1
        row["cells"].setdefault(cell, []).append(rep)

    for cell in cells:
        per_prop: Dict[str, Any] = {}
        for prop in sorted(manifest["properties"]):
            entry = (gold.get("properties") or {}).get(prop) or {}
            rooms = entry.get("canonical_rooms") or []
            expected = entry.get("expected_packages") or []
            required = entry.get("required_work_items") or []
            gold_ids = pkg.package_gold_ids(gold, prop)
            # Accepted-sibling representations of a gold target are that
            # target, never extras.
            accepted_work_sigs = {
                f"{cid}@{w['room_id']}" for w in required
                for cid in [w["catalog_item_id"],
                            *(w.get("accepted_sibling_ids") or [])]}
            family_of = {e["package_type"]: pkg._package_type_room(e["package_type"])
                         for e in expected}
            reps: Dict[str, Any] = {}
            complete_reps = 0
            pending_reps = 0
            for rep in range(1, repeats + 1):
                cell_dir = pkg._cell_dir(cell, rep, prop)
                final_path = cell_dir / "final_estimate.json"
                if not final_path.is_file():
                    reps[str(rep)] = {"status": "not_evaluated"}
                    continue
                final = _load_json(final_path)
                approved = [p for p in final.get("packages") or []
                            if p.get("estimate_eligible")
                            and pkg._scoreable_package(p)]
                items = final.get("line_items") or []
                approved_ids = {p.get("package_id") for p in approved}
                evidence_ids: Set[str] = set()
                for p in approved:
                    evidence_ids.update(p.get("supporting_catalog_item_ids") or [])

                alignment = align_units_to_rooms(_unit_photo_map(final), rooms)
                room_of_unit = {u: a["room_id"] for u, a in alignment.items()}

                def _room_or_unit(unit: Optional[str]) -> str:
                    return room_of_unit.get(str(unit)) or str(unit)

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

                for p in approved:
                    unit = str(p.get("estimate_unit_id"))
                    sig = f"{p.get('package_type')}__{_room_or_unit(unit)}"
                    if sig in gold_ids["pkg"]:
                        continue
                    row_id = f"{prop}|pkg|{sig}"
                    if _judge_extra(row_id, "pkg") != "equivalent":
                        _collect_review_row(
                            review_rows, row_id, prop, "package", p, None,
                            cell, rep, alignment.get(unit))
                for li in items:
                    if li.get("package_id"):
                        continue
                    if not (li.get("cost_low") or li.get("cost_high")):
                        continue
                    unit = str(li.get("billable_estimate_unit_id"))
                    sig = f"{li.get('catalog_item_id')}@{_room_or_unit(unit)}"
                    if sig in gold_ids["item"] or sig in accepted_work_sigs:
                        continue
                    if li.get("catalog_item_id") in evidence_ids:
                        continue  # evidence of an approved package, never an extra
                    row_id = f"{prop}|item|{sig}"
                    if _judge_extra(row_id, "item") != "equivalent":
                        _collect_review_row(
                            review_rows, row_id, prop, "work_item", None, li,
                            cell, rep, alignment.get(unit))

                # Expected packages on canonical rooms.
                pkg_status: Dict[str, Any] = {}
                adjacent_used: List[str] = []
                strict_missing: List[str] = []
                family_hits: List[bool] = []
                for exp in expected:
                    gid = f"{exp['package_type']}__{exp['room_id']}"
                    policy = exp.get("policy") or "strict"
                    status, observed, observed_unit = _match_expected(
                        exp, approved, room_of_unit, adjacency)
                    if status == "missing" and gid in equivalent_credit["pkg"]:
                        status = "matched_equivalent"
                    family = family_of.get(exp["package_type"])
                    family_present = any(
                        room_of_unit.get(str(p.get("estimate_unit_id")))
                        == exp["room_id"]
                        and pkg._package_type_room(str(p.get("package_type")))
                        == family
                        for p in approved)
                    distance = tier_distance(exp["target_pricing_profile"],
                                             observed)
                    pkg_status[gid] = {
                        "status": status,
                        "observed_profile": observed,
                        "observed_unit": observed_unit,
                        "tier_distance": distance,
                        "abs_tier_distance": (abs(distance)
                                              if distance is not None else None),
                        "family_present": family_present,
                        "policy": policy,
                    }
                    if status == "matched_adjacent":
                        adjacent_used.append(gid)
                    if policy == "strict":
                        family_hits.append(family_present)
                        if status in ("wrong_tier", "missing"):
                            strict_missing.append(gid)
                    if status in ("wrong_tier", "missing"):
                        _record_missing(prop, "pkg", gid, policy, cell, rep)

                # Required work on canonical rooms (line item, package
                # support, sibling, or equivalent decision).
                work_status: Dict[str, Any] = {}
                for work in required:
                    gid = f"{work['catalog_item_id']}@{work['room_id']}"
                    policy = work.get("policy") or "strict"
                    accepted = [work["catalog_item_id"]] + list(
                        work.get("accepted_sibling_ids") or [])
                    route = None
                    matched_cid = None
                    for cid in accepted:
                        if any(li.get("catalog_item_id") == cid
                               and _room_or_unit(
                                   li.get("billable_estimate_unit_id"))
                               == work["room_id"]
                               and (not li.get("package_id")
                                    or li.get("package_id") in approved_ids)
                               for li in items):
                            route, matched_cid = "line_item", cid
                            break
                        if any(room_of_unit.get(str(p.get("estimate_unit_id")))
                               == work["room_id"]
                               and cid in (p.get("supporting_catalog_item_ids")
                                           or [])
                               for p in approved):
                            route, matched_cid = "package_support", cid
                            break
                    if route is None and gid in equivalent_credit["item"]:
                        route, matched_cid = "equivalent_decision", \
                            work["catalog_item_id"]
                    if route is None:
                        work_status[gid] = {"status": "missing", "route": None,
                                            "policy": policy}
                        if policy == "strict":
                            strict_missing.append(gid)
                        _record_missing(prop, "item", gid, policy, cell, rep)
                    else:
                        exact = matched_cid == work["catalog_item_id"]
                        work_status[gid] = {
                            "status": "satisfied" if exact else "satisfied_sibling",
                            "route": route,
                            "policy": policy,
                        }

                # Component coverage: did anything produced in the room carry
                # the same component class, regardless of exact catalog id?
                produced_pairs: Set[Tuple[str, Optional[str]]] = set()
                for li in items:
                    room = _room_or_unit(li.get("billable_estimate_unit_id"))
                    comp = _component_of(str(li.get("catalog_item_id")),
                                         trade_by_id)
                    if comp:
                        produced_pairs.add((room, comp))
                for p in approved:
                    room = _room_or_unit(p.get("estimate_unit_id"))
                    for cid in p.get("supporting_catalog_item_ids") or []:
                        comp = _component_of(str(cid), trade_by_id)
                        if comp:
                            produced_pairs.add((room, comp))
                target_pairs: Set[Tuple[str, str]] = set()
                for work in required:
                    if (work.get("policy") or "strict") != "strict":
                        continue
                    comp = _component_of(work["catalog_item_id"], trade_by_id)
                    if comp:
                        target_pairs.add((work["room_id"], comp))
                covered_pairs = {t for t in target_pairs if t in produced_pairs}
                strict_work = [w for w in required
                               if (w.get("policy") or "strict") == "strict"]
                exact_covered = sum(
                    1 for w in strict_work
                    if work_status[f"{w['catalog_item_id']}@{w['room_id']}"]
                    ["status"] == "satisfied")

                matched_signatures = sorted(
                    gid for gid, row in pkg_status.items()
                    if row["status"] in ("matched_exact", "matched_adjacent",
                                         "matched_equivalent"))

                strict_complete: Optional[bool]
                if extras_open:
                    strict_complete = None
                    pending_reps += 1
                else:
                    strict_complete = (not strict_missing
                                       and not extras_confirmed)
                    if strict_complete:
                        complete_reps += 1
                reps[str(rep)] = {
                    "status": ("pending_review" if strict_complete is None
                               else "complete" if strict_complete
                               else "incomplete"),
                    "strict_complete": strict_complete,
                    "alignment": alignment,
                    "expected_packages": pkg_status,
                    "package_family_recall": (
                        round(sum(family_hits) / len(family_hits), 3)
                        if family_hits else None),
                    "required_work": work_status,
                    "component_coverage": {
                        "covered": len(covered_pairs),
                        "total": len(target_pairs),
                        "missing_components": sorted(
                            f"{room}:{comp}"
                            for room, comp in target_pairs - covered_pairs),
                    },
                    "exact_id_coverage": {"covered": exact_covered,
                                          "total": len(strict_work)},
                    "strict_missing": sorted(strict_missing),
                    "adjacent_used": sorted(adjacent_used),
                    "extras_confirmed": sorted(set(extras_confirmed)),
                    "extras_open": sorted(set(extras_open)),
                    "final_midpoint": int((final.get("final_rehab") or {})
                                          .get("midpoint") or 0),
                    "matched_gold_packages": matched_signatures,
                }
            evaluated = [r for r in reps.values()
                         if r.get("status") != "not_evaluated"]
            per_prop[prop] = {
                "reps": reps,
                "passing_reps": complete_reps,
                # Pending review nulls only this property's verdict, and only
                # because ITS reps carry undecided rows.
                "passed": (None if (not evaluated or pending_reps)
                           else complete_reps >= pass_reps),
            }
        outcomes[cell] = per_prop

    pending_counts = {k: len(v) for k, v in pending.items()}
    blocked = sum(pending_counts.values()) > 0
    disagreements = _pass_2f_disagreements(config, manifest, gold, variant_b)
    cap_pairs = _cap_pairs(config, manifest, outcomes, variant_b)
    cap_experiment = _aggregate_cap_verdicts(
        config, manifest, outcomes, variant_b, gates, disagreements)
    status = ("blocked" if blocked else "final") if any(
        r.get("status") != "not_evaluated"
        for cell in outcomes.values() for prop in cell.values()
        for r in prop["reps"].values()) else "no_2f_results"
    return {
        "round": round_label,
        "status": status,
        "pending_review": pending_counts,
        "outcomes": outcomes,
        "cap_pairs": cap_pairs,
        "cap_experiment": cap_experiment,
        "pass_2f_disagreements": disagreements,
        "candidate_diagnostics": _candidate_diagnostics(
            config, manifest, gold, adjacency, variant_a, variant_b),
        "distinct_missing_targets_across_runs": {
            k: {**v, "cells": {c: sorted(r) for c, r in v["cells"].items()}}
            for k, v in sorted(distinct_missing.items())},
        "cost_deviation": _cost_deviation(gold, outcomes),
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
                        p: Optional[Dict[str, Any]],
                        item: Optional[Dict[str, Any]],
                        cell: str, rep: int,
                        alignment: Optional[Dict[str, Any]]) -> None:
    row = rows.setdefault(row_id, {
        "row_id": row_id, "property": prop, "outcome_kind": outcome_kind,
        "package_type": (p or {}).get("package_type", ""),
        "estimate_unit_id": ((p or {}).get("estimate_unit_id")
                             or (item or {}).get("billable_estimate_unit_id", "")),
        "room_id": (alignment or {}).get("room_id") or "",
        "alignment_status": (alignment or {}).get("status") or "no_units",
        "catalog_item_id": (item or {}).get("catalog_item_id", ""),
        "item_name": (item or {}).get("name", ""),
        "profiles": set(), "cost_low": 0, "cost_high": 0,
        "occurs": {}, "evidence_photos": set(),
        "evidence_summary": (p or {}).get("evidence_summary", ""),
    })
    if p:
        row["profiles"].add(p.get("pricing_profile") or "")
        row["cost_low"] = max(row["cost_low"], int(p.get("cost_low") or 0))
        row["cost_high"] = max(row["cost_high"], int(p.get("cost_high") or 0))
        row["evidence_photos"].update(p.get("review_photo_keys") or [])
    if item:
        row["cost_low"] = max(row["cost_low"], int(item.get("cost_low") or 0))
        row["cost_high"] = max(row["cost_high"], int(item.get("cost_high") or 0))
    row["occurs"].setdefault(cell, set()).add(rep)


# ---------------------------------------------------------------------------
# Paired cap reporting
# ---------------------------------------------------------------------------

def _rep_metrics(rep_row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    distances = [row["abs_tier_distance"]
                 for row in (rep_row.get("expected_packages") or {}).values()
                 if row.get("policy") == "strict"
                 and row.get("abs_tier_distance") is not None]
    comp = rep_row.get("component_coverage") or {}
    exact = rep_row.get("exact_id_coverage") or {}
    return {
        "package_family_recall": rep_row.get("package_family_recall"),
        "mean_abs_tier_distance": (round(statistics.mean(distances), 3)
                                   if distances else None),
        "component_coverage": (comp["covered"] / comp["total"]
                               if comp.get("total") else None),
        "exact_id_coverage": (exact["covered"] / exact["total"]
                              if exact.get("total") else None),
        "extras": len(rep_row.get("extras_confirmed") or [])
        + len(rep_row.get("extras_open") or []),
        "final_midpoint": rep_row.get("final_midpoint"),
    }


def _cap_bound(config: Dict[str, Any], manifest: Dict[str, Any],
               variant: str, prop: str, rep: int) -> Optional[bool]:
    """Did the 2d cap actually bind for this (property, repeat)?"""
    photos = [p for pk, p in pkg.manifest_photos(manifest) if pk == prop]
    from tools.benchmark_pass2a import _photo_ckpt_dir
    any_seen = False
    for photo in photos:
        ckpt_path = (_photo_ckpt_dir(
            pkg.variant_dir(variant) / f"rep{rep}" / prop)
            / f"{photo['photo_key']}.json")
        if not ckpt_path.is_file():
            continue
        any_seen = True
        kept, resolved = pkg._checkpoint_lanes(_load_json(ckpt_path))
        if len(kept) > len(resolved):
            return True
    return False if any_seen else None


def _cap_pairs(config: Dict[str, Any], manifest: Dict[str, Any],
               outcomes: Dict[str, Any], variant_b: str) -> Dict[str, Any]:
    """Primary cap analysis: capped vs all-retained paired by (property,
    repeat), with metric deltas (all_retained minus capped) and whether the
    cap actually bound. Never collapsed into a single verdict."""
    capped, allret = f"{variant_b}_cap25", f"{variant_b}_all_retained"
    repeats = int(config["repeats"])
    out: Dict[str, Any] = {}
    for prop in sorted(manifest["properties"]):
        rows: Dict[str, Any] = {}
        for rep in range(1, repeats + 1):
            c = ((outcomes.get(capped) or {}).get(prop) or {}) \
                .get("reps", {}).get(str(rep)) or {}
            a = ((outcomes.get(allret) or {}).get(prop) or {}) \
                .get("reps", {}).get(str(rep)) or {}
            if c.get("status") in (None, "not_evaluated") or \
                    a.get("status") in (None, "not_evaluated"):
                rows[str(rep)] = {"status": "unpaired"}
                continue
            cm, am = _rep_metrics(c), _rep_metrics(a)
            delta = {
                key: (round(am[key] - cm[key], 3)
                      if isinstance(am.get(key), (int, float))
                      and isinstance(cm.get(key), (int, float)) else None)
                for key in cm
            }
            rows[str(rep)] = {
                "status": "paired",
                "cap_bound": _cap_bound(config, manifest, variant_b, prop, rep),
                "capped": cm,
                "all_retained": am,
                "delta": delta,
                "strict": {"capped": c.get("strict_complete"),
                           "all_retained": a.get("strict_complete")},
            }
        out[prop] = rows
    return out


def _aggregate_cap_verdicts(config: Dict[str, Any], manifest: Dict[str, Any],
                            outcomes: Dict[str, Any], variant_b: str,
                            gates: Dict[str, Any],
                            disagreements: Dict[str, Any]) -> Dict[str, Any]:
    """SECONDARY summary only — the strict 2-of-3 verdict per property. The
    paired per-repeat deltas in cap_pairs are the primary signal."""
    capped, allret = f"{variant_b}_cap25", f"{variant_b}_all_retained"
    out: Dict[str, Any] = {"_note": "secondary aggregate; cap_pairs is primary"}
    for prop in sorted(manifest["properties"]):
        c = (outcomes.get(capped) or {}).get(prop) or {}
        a = (outcomes.get(allret) or {}).get(prop) or {}
        verdict = None
        if c.get("passed") is not None and a.get("passed") is not None:
            verdict = {
                (True, True): "cap_benign",
                (False, True): "cap_hurts",
                (True, False): "surplus_hurts",
                (False, False): "both_wrong",
            }[(bool(c["passed"]), bool(a["passed"]))]
        flags: List[str] = []
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
        noise = (disagreements.get(prop) or {}).get("count") or 0
        confidence = "normal"
        if verdict and verdict != "cap_benign" and noise:
            confidence = "reduced_by_2f_noise"
        out[prop] = {"verdict": verdict, "flags": flags,
                     "verdict_confidence": confidence}
    return out


# ---------------------------------------------------------------------------
# Pass 2f disagreement detection (noise vs cap effect)
# ---------------------------------------------------------------------------

def _candidate_identity(candidate: Dict[str, Any],
                        rooms: List[Dict[str, Any]]) -> Tuple[str, str]:
    """(package_type, canonical room or raw unit). Aligned by the candidate's
    review photos so unit renumbering cannot split identities."""
    ptype = str(candidate.get("package_type") or "")
    photos = set(candidate.get("review_photo_keys") or [])
    family = pkg._package_type_room(ptype)
    aligned = align_room_for(photos, family, rooms) if photos else None
    room = (aligned or {}).get("room_id") \
        or str(candidate.get("estimate_unit_id") or "")
    return ptype, room


def _pass_2f_disagreements(config: Dict[str, Any], manifest: Dict[str, Any],
                           gold: Dict[str, Any],
                           variant_b: str) -> Dict[str, Any]:
    """A capped/all-retained decision difference counts as 2f noise ONLY when
    the canonical package identity AND its candidate evidence (supporting
    issue ids + review photo keys) are equivalent. Different evidence is a
    genuine cap effect and is never flagged here."""
    capped, allret = f"{variant_b}_cap25", f"{variant_b}_all_retained"
    repeats = int(config["repeats"])
    out: Dict[str, Any] = {}
    for prop in sorted(manifest["properties"]):
        rooms = ((gold.get("properties") or {}).get(prop) or {}) \
            .get("canonical_rooms") or []
        records: List[Dict[str, Any]] = []
        for rep in range(1, repeats + 1):
            sides: Dict[str, Dict[Tuple[str, str], List[Dict[str, Any]]]] = {}
            ok = True
            for label, cell in (("capped", capped), ("all_retained", allret)):
                cell_dir = pkg._cell_dir(cell, rep, prop)
                if not ((cell_dir / "candidates.json").is_file()
                        and (cell_dir / "verifications.json").is_file()):
                    ok = False
                    break
                cands = [c for c in _load_json(
                    cell_dir / "candidates.json")["candidates"]
                    if pkg._candidate_in_scope(c)]
                vers = _load_json(cell_dir / "verifications.json")["verifications"]
                table: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
                for c in cands:
                    identity = _candidate_identity(c, rooms)
                    table.setdefault(identity, []).append({
                        "evidence": (
                            frozenset(c.get("supporting_catalog_item_ids") or []),
                            frozenset(c.get("review_photo_keys") or [])),
                        "status": (vers.get(str(c.get("package_id"))) or {})
                        .get("verification_status"),
                    })
                sides[label] = table
            if not ok:
                continue
            shared = set(sides["capped"]) & set(sides["all_retained"])
            for identity in sorted(shared):
                c_rows = sides["capped"][identity]
                a_rows = sides["all_retained"][identity]
                if len(c_rows) != 1 or len(a_rows) != 1:
                    continue  # ambiguous identity — never attribute noise
                c_row, a_row = c_rows[0], a_rows[0]
                if c_row["evidence"] != a_row["evidence"]:
                    continue  # different evidence — genuine cap effect
                c_status, a_status = c_row["status"], a_row["status"]
                if "confirmed_by_rule" in (c_status, a_status):
                    continue
                if c_status != a_status:
                    records.append({
                        "identity": f"{identity[0]}__{identity[1]}",
                        "rep": rep,
                        "capped_status": c_status,
                        "all_retained_status": a_status,
                    })
        out[prop] = {"count": len(records), "records": records}
    return out


# ---------------------------------------------------------------------------
# Candidate diagnostics + cost deviation (rescore-only, canonical rooms)
# ---------------------------------------------------------------------------

def _candidate_diagnostics(config: Dict[str, Any], manifest: Dict[str, Any],
                           gold: Dict[str, Any],
                           adjacency: Dict[str, Set[str]],
                           variant_a: str, variant_b: str) -> Dict[str, Any]:
    repeats = int(config["repeats"])
    out: Dict[str, Any] = {}
    for cell in pkg.cells_for_round(variant_a, variant_b):
        per_prop: Dict[str, Any] = {}
        for prop in sorted(manifest["properties"]):
            entry = (gold.get("properties") or {}).get(prop) or {}
            rooms = entry.get("canonical_rooms") or []
            expected = entry.get("expected_packages") or []
            tiers = {"exact": 0, "adjacent": 0, "wrong": 0, "missing": 0}
            recalls: List[float] = []
            extra_counts: List[int] = []
            id_sets: List[Set[str]] = []
            for rep in range(1, repeats + 1):
                path = pkg._cell_dir(cell, rep, prop) / "candidates.json"
                if not path.is_file():
                    continue
                candidates = [c for c in _load_json(path)["candidates"]
                              if pkg._candidate_in_scope(c)]
                room_of_unit = {
                    str(c.get("estimate_unit_id")):
                    _candidate_identity(c, rooms)[1]
                    for c in candidates}
                sigs = {f"{c.get('package_type')}__"
                        f"{_candidate_identity(c, rooms)[1]}"
                        for c in candidates}
                id_sets.append(sigs)
                gold_sigs = {f"{e['package_type']}__{e['room_id']}"
                             for e in expected}
                if gold_sigs:
                    recalls.append(len(sigs & gold_sigs) / len(gold_sigs))
                extra_counts.append(len(sigs - gold_sigs))
                for exp in expected:
                    status, _, _ = _match_expected(exp, candidates,
                                                   room_of_unit, adjacency)
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


def _cost_deviation(gold: Dict[str, Any],
                    outcomes: Dict[str, Any]) -> Dict[str, Any]:
    """Matched-package cost vs the gold's diagnostic range. Report-only."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for cell, per_prop in outcomes.items():
        for prop, data in per_prop.items():
            expected = ((gold.get("properties") or {}).get(prop) or {}) \
                .get("expected_packages") or []
            ranges = {f"{e['package_type']}__{e['room_id']}":
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
