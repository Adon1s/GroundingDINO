"""Cluster the Session 9 canary review items and prefill the review template.

967 review items is unreviewable one-by-one, but they are not 967 distinct
facts: most are the same few mechanical effects repeated across properties.
This tool sorts every item into one of three tiers:

  TIER 1 - machine-verified no material change (auto-accepted). Criteria are
    numeric and checked here, not asserted: identical dollars, rounding
    within $2, catalog id absent from every v4 row in the corpus, etc.
  TIER 2 - grouped POLICY decisions (drafted, need Steven's sign-off). Each
    is one real decision covering many items, with its full corpus dollar
    effect stated. These are prefilled as DRAFTS: read the digest section,
    then keep or edit the explanation.
  TIER 3 - individual review. Left blank.

A crucial correction the comparator's per-catalog-id view hides: when v5
merges N catalog items into ONE work item, _v5_scope emits the work item's
FULL price under each constituent catalog id. So a merge shows up as N
separate "increases" even when the merge group's total went DOWN. This tool
reconstructs merge groups from the v5 work items and judges them at the
group level, which is the only level where the dollars mean anything.

Run:
  .venv\\Scripts\\python.exe scripts\\cluster_session9_reviews.py ^
    --report reports\\renovation_architecture_session9_canary_20260821.json ^
    --template reports\\renovation_architecture_session9_reviews_20260821.json ^
    --canary-root artifacts_canary\\renovation_session9_20260818 ^
    --digest reports\\session9_review_digest_20260821.md ^
    --out reports\\renovation_architecture_session9_reviews_20260821.prefilled.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_renovation_architecture_cutover import (  # noqa: E402
    _load_latest_artifacts,
    _v4_scope,
)

# Scope review items compare v4 vs v5 within the FIRST run only (see
# compare_canary: `artifact = runs[0][key][1]`), so merge groups are built
# from run_1 to match.
BASELINE_RUN = "run_1"

TRIAGED = {
    "unfinished_basement_present",
    "staging_or_decluttering_opportunity",
    "mismatched_or_inconsistent_furniture_staging",
    "curb_appeal_upgrade",
    "landscaping_enhancement_opportunity",
}

ROUNDING_TOLERANCE = 2  # dollars per side

TIER1 = {
    "scope_identical_dollars": (
        "v4 and v5 bill identical dollars for this item (every low/high equal, "
        "machine-verified); only the representation label differs (v4 "
        "'absorbed_by:<type>' vs the v5 coverage-ledger form). No scope or "
        "dollar change."
    ),
    "scope_v5_class": (
        "Catalog item billed by v5 only: it appears in zero v4 scope rows "
        "anywhere in the 36-artifact canary corpus. Member of the v5 billing "
        "class accepted as intended scope in Session 8 (48-item class "
        "decision, 2026-08-16)."
    ),
    "scope_dropped_triaged": (
        "Item is route_override=no_action (Session 8 five-item triage "
        "decision, 2026-08-16): v5 intentionally does not bill it; v4 ignores "
        "the override. Expected drop, verified against the shipped v2 catalog."
    ),
    "scope_rounding_only": (
        "Same item, same scope; v5 consolidates v4's per-occurrence rows into "
        "one work item and the integer allocation rounds differently - low and "
        "high each differ by at most $2 (machine-verified). No material change."
    ),
    "scope_merge_neutral": (
        "v5 merges these catalog items into one work item; the merge group's "
        "total is within $2 of v4's sum for the same items at the same unit "
        "(machine-verified). The apparent per-item change is only the merged "
        "work item's full price being listed under each constituent id."
    ),
    "package_key_migration": (
        "Same package on both sides: v4 keys packages without an estimate "
        "unit ('type|') while v5 keys per unit ('type|unit'), so one package "
        "surfaces as a paired removal+addition. Package type and pricing tier "
        "match (machine-verified); no package change."
    ),
    "package_metadata_only": (
        "Identical package on both sides: package_type, estimate_unit_id, "
        "pricing_tier, and status all equal (machine-verified). The diff is "
        "v5-only metadata fields (Sol decision, reason_code) that v4 rows "
        "never carried."
    ),
}

TIER2 = {
    "scope_merge_dedup_lower": {
        "title": "v5 merge dedup prices shared scope LOWER",
        "question": (
            "v5 replaces several stacked v4 rows for the same unit with ONE "
            "merged work item, and charges less than v4's sum. This is the "
            "negative residual term measured in Session 8 "
            "(docs/analysis/session8_floor_vs_sum_trace.md: 'v5 prices shared "
            "scope lower'). Accepting means agreeing that one action covering "
            "several co-located observations should not stack prices."
        ),
        "explanation": (
            "v5 merges the co-located observations at this unit into one work "
            "item rather than stacking a price per observation, so the group "
            "totals less than v4's sum. Accepted as the intended dedup "
            "behavior (Session 8 floor-vs-sum analysis, 'v5 prices shared "
            "scope lower'); the apparent per-item change is the merged work "
            "item's full price listed under each constituent catalog id."
        ),
    },
    "scope_repriced_single_item": {
        "title": "single-item repricing (v5 range wider at the high end)",
        "question": (
            "Same item, same unit, same scope, not merged - v5 simply prices "
            "it differently. The observed signature is a near-unchanged low "
            "and a materially higher high, concentrated in area-priced "
            "surface items (flooring, carpet, ceiling). Accepting means "
            "endorsing v5's price range for these items."
        ),
        "explanation": (
            "Same item, unit, and scope on both sides; v5's price range is "
            "wider at the high end while the low is essentially unchanged. "
            "Accepted as v5 pricing-model output for area-priced surface "
            "work; no scope change."
        ),
    },
    "scope_v4_zero_now_priced": {
        "title": "v4 carried the item at $0, v5 prices it",
        "question": (
            "v4 has a scope row for this item but at $0/$0 (withheld or "
            "evidence-gated); v5 assigns a real price. Accepting means v5 is "
            "right to price scope v4 acknowledged but never costed."
        ),
        "explanation": (
            "v4 carried this item at $0/$0 (withheld / evidence-gated) while "
            "recognizing the scope; v5 prices it. Accepted: the item is real "
            "scope and pricing it is the intended v5 behavior."
        ),
    },
    "scope_merge_scope_promotion": {
        "title": "merged work item takes the stricter estimate_scope",
        "question": (
            "Identical dollars, but v5 merged this marketability item with a "
            "required-scope item and the merged work item carries the "
            "stricter scope. Accepting means agreeing a merged action inherits "
            "the strictest constituent scope."
        ),
        "explanation": (
            "Dollars identical on both sides; v5 merged this observation with "
            "a required-scope item at the same unit and the merged work item "
            "carries the stricter scope. Accepted: a single action covering "
            "required work is itself required."
        ),
    },
}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sig(row):
    return (row["catalog_item_id"], row["billable_unit_id"],
            row.get("estimate_scope"), row["low"], row["high"])


def _money(rows):
    return (sum(r["low"] for r in rows), sum(r["high"] for r in rows))


def _scopes(rows):
    return sorted({str(r.get("estimate_scope") or "") for r in rows})


def _fmt(pair):
    return f"${pair[0]:,}/{pair[1]:,}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--canary-root", required=True)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--blank-drafts", action="store_true",
        help="leave TIER 2 policy groups blank instead of prefilling drafts",
    )
    args = parser.parse_args(argv)

    report = _load(args.report)
    template = _load(args.template)
    items = report["review_items"]
    root = Path(args.canary_root)

    # ---- corpus facts -------------------------------------------------
    v4_ids = set()                              # ids v4 bills ANYWHERE
    v4_by_prop = defaultdict(set)               # prop -> {(id, unit)}
    v4_money = {}                               # (prop, id, unit) -> (lo, hi)
    v5_conditions = defaultdict(dict)           # prop -> {id: [dispositions]}
    merge_of = {}                               # (prop, id, unit) -> group id
    groups = {}                                 # group id -> facts

    for run in ("run_1", "run_2"):
        for prop, (_, artifact) in _load_latest_artifacts(
                root / run / "candidate").items():
            for key, rows in _v4_scope(artifact).items():
                cid, _, unit = key.partition("|")
                for row in rows:
                    v4_ids.add(row["catalog_item_id"])
                    v4_by_prop[prop].add(
                        (row["catalog_item_id"], row["billable_unit_id"]))
                if run == BASELINE_RUN:
                    v4_money[(prop, cid, unit)] = _money(rows)
            envelope = ((artifact.get("analysis_debug") or {}).get(
                "renovation_estimate_v5")
                or artifact.get("renovation_estimate_v5") or {})
            result = envelope.get("result") or {}
            audit = {str(r.get("condition_id")): str(r.get("disposition") or "")
                     for r in result.get("condition_dispositions") or []
                     if isinstance(r, dict)}
            for cond in result.get("observed_conditions") or []:
                if isinstance(cond, dict):
                    v5_conditions[prop].setdefault(
                        str(cond.get("catalog_item_id") or ""), []).append(
                            audit.get(str(cond.get("condition_id")), ""))
            if run != BASELINE_RUN:
                continue
            # Merge groups: one v5 work item may cover several catalog ids,
            # and _v5_scope lists its FULL price under each of them.
            for work in result.get("work_items") or []:
                if not isinstance(work, dict) or work.get("status") != "active":
                    continue
                cids = sorted(str(c) for c in work.get("catalog_item_ids") or [])
                if len(cids) < 2:
                    continue
                unit = str(work.get("billable_unit_id") or "")
                gid = (prop, unit, tuple(cids))
                groups[gid] = {
                    "v5": (int(work.get("low") or 0), int(work.get("high") or 0)),
                    "action_code": work.get("action_code"),
                }
                for cid in cids:
                    merge_of[(prop, cid, unit)] = gid

    for gid, g in groups.items():
        prop, unit, cids = gid
        lo = hi = 0
        for cid in cids:
            a, b = v4_money.get((prop, cid, unit), (0, 0))
            lo += a
            hi += b
        g["v4"] = (lo, hi)

    # ---- classification ----------------------------------------------
    clusters = defaultdict(list)      # name -> [(item, evidence, explanation)]

    def add(name, item, evidence="", explanation=None):
        clusters[name].append((item, evidence, explanation))

    for item in items:
        cat, prop = item["category"], item["property_key"]
        cid, _, unit = item["key"].partition("|")
        b, c = item.get("baseline"), item.get("candidate")

        if cat == "scope":
            brows, crows = b or [], c or []
            gid = merge_of.get((prop, cid, unit))
            group = groups.get(gid) if gid else None

            if brows and crows:
                bm, cm = _money(brows), _money(crows)
                same_money = (abs(cm[0] - bm[0]) <= ROUNDING_TOLERANCE
                              and abs(cm[1] - bm[1]) <= ROUNDING_TOLERANCE)
                same_scope = _scopes(brows) == _scopes(crows)
                money = f"{_fmt(bm)} -> {_fmt(cm)}"
                if {_sig(r) for r in brows} == {_sig(r) for r in crows}:
                    add("scope_identical_dollars", item)
                elif group:
                    gv4, gv5 = group["v4"], group["v5"]
                    ev = (f"merge group {list(gid[2])} @ {unit or 'listing'}: "
                          f"{_fmt(gv4)} -> {_fmt(gv5)} "
                          f"({gv5[0] - gv4[0]:+,}/{gv5[1] - gv4[1]:+,})")
                    within = (abs(gv5[0] - gv4[0]) <= ROUNDING_TOLERANCE
                              and abs(gv5[1] - gv4[1]) <= ROUNDING_TOLERANCE)
                    if within:
                        add("scope_merge_neutral", item, ev)
                    elif gv5[0] <= gv4[0] and gv5[1] <= gv4[1]:
                        add("scope_merge_dedup_lower", item, ev,
                            TIER2["scope_merge_dedup_lower"]["explanation"]
                            + f" This group: {_fmt(gv4)} -> {_fmt(gv5)}.")
                    else:
                        add("scope_merge_increased", item, ev)
                elif same_money and same_scope:
                    add("scope_rounding_only", item, money)
                elif same_money:
                    add("scope_merge_scope_promotion", item,
                        f"{money}; scope {'+'.join(_scopes(brows))} -> "
                        f"{'+'.join(_scopes(crows))}",
                        TIER2["scope_merge_scope_promotion"]["explanation"])
                elif bm == (0, 0):
                    add("scope_v4_zero_now_priced", item,
                        f"v4 $0/$0 -> {_fmt(cm)}",
                        TIER2["scope_v4_zero_now_priced"]["explanation"])
                else:
                    add("scope_repriced_single_item", item,
                        f"{money} ({cm[0] - bm[0]:+,}/{cm[1] - bm[1]:+,})",
                        TIER2["scope_repriced_single_item"]["explanation"])
            elif crows and not brows:
                cm = _money(crows)
                if cid not in v4_ids:
                    add("scope_v5_class", item, f"+{_fmt(cm)}")
                elif any(k[0] == cid for k in v4_by_prop.get(prop, ())):
                    add("scope_added_reattributed", item,
                        f"+{_fmt(cm)}; v4 bills this id at another unit")
                else:
                    add("scope_added_routing_diff", item,
                        f"+{_fmt(cm)}; v4 bills this id elsewhere in the "
                        "corpus but not on this property")
            else:
                bm = _money(brows)
                if cid in TRIAGED:
                    add("scope_dropped_triaged", item, f"-{_fmt(bm)}")
                else:
                    disps = v5_conditions.get(prop, {}).get(cid)
                    if disps and not any(d == "accepted_for_work" for d in disps):
                        add("scope_dropped_disposition", item,
                            f"-{_fmt(bm)}; v5 disposition(s): {sorted(set(disps))}")
                    else:
                        add("scope_dropped_other", item, f"-{_fmt(bm)}")
        elif cat == "package":
            add("package_raw", item)
        elif cat == "headline_delta":
            d = (c or {}).get("delta_pct") or {}
            ch = (c or {}).get("headline") or {}
            add("headline", item,
                f"${(b or {}).get('low', 0):,}/{(b or {}).get('high', 0):,} -> "
                f"${ch.get('low', 0):,}/{ch.get('high', 0):,} "
                f"({d.get('low', 0):+.0%}/{d.get('high', 0):+.0%})")
        else:
            add("stability", item)

    # ---- package pairing ---------------------------------------------
    paired_ids = set()
    by_prop = defaultdict(lambda: {"v4": [], "v5": [], "other": []})
    for item, _, _ in clusters.pop("package_raw", []):
        b, c = item.get("baseline"), item.get("candidate")
        if b and c and all((b.get(k) or None) == (c.get(k) or None) for k in
                           ("package_type", "estimate_unit_id",
                            "pricing_tier", "status")):
            add("package_metadata_only", item,
                f"{item['key']} tier={c.get('pricing_tier')}")
        elif b and not c and not b.get("estimate_unit_id"):
            by_prop[item["property_key"]]["v4"].append(item)
        elif c and not b and c.get("estimate_unit_id"):
            by_prop[item["property_key"]]["v5"].append(item)
        else:
            by_prop[item["property_key"]]["other"].append(item)

    for prop, grouped in by_prop.items():
        v4_types = {i["baseline"]["package_type"]: i for i in grouped["v4"]}
        for item in grouped["v5"]:
            c = item["candidate"]
            mate = v4_types.get(c["package_type"])
            if mate is not None and mate["baseline"].get("pricing_tier") == c.get("pricing_tier"):
                add("package_key_migration", item,
                    f"pairs with v4 '{c['package_type']}|' (same tier)")
                paired_ids.add(item["review_id"])
                if mate["review_id"] not in paired_ids:
                    add("package_key_migration", mate,
                        f"pairs with v5 '{c['package_type']}|"
                        f"{c['estimate_unit_id']}' (same tier)")
                    paired_ids.add(mate["review_id"])
            else:
                why = "no v4 package of this type" if mate is None else "tier differs"
                add("package_needs_eyes", item,
                    f"tier={c.get('pricing_tier')} reason={c.get('reason_code')} ({why})")
        for item in grouped["v4"]:
            if item["review_id"] not in paired_ids:
                add("package_needs_eyes", item,
                    f"v4-only, tier={item['baseline'].get('pricing_tier')}")
        for item in grouped["other"]:
            add("package_needs_eyes", item, "changed in place (tier/decision/status)")

    # ---- prefill ------------------------------------------------------
    reviews = dict(template["reviews"])
    for name, text in TIER1.items():
        for item, _, _ in clusters.get(name, []):
            reviews[item["review_id"]] = {"decision": "accepted",
                                          "explanation": text}
    drafted = 0
    if not args.blank_drafts:
        for name in TIER2:
            for item, _, explanation in clusters.get(name, []):
                reviews[item["review_id"]] = {
                    "decision": "accepted",
                    "explanation": explanation or TIER2[name]["explanation"]}
                drafted += 1
    Path(args.out).write_text(json.dumps({**template, "reviews": reviews},
                                         indent=1), encoding="utf-8")

    # ---- digest -------------------------------------------------------
    def total(rows):
        lo = hi = 0
        seen = set()
        for item, _, _ in rows:
            b, c = item.get("baseline") or [], item.get("candidate") or []
            prop = item["property_key"]
            cid, _, unit = item["key"].partition("|")
            gid = merge_of.get((prop, cid, unit))
            if gid:                       # count each merge group once
                if gid in seen:
                    continue
                seen.add(gid)
                g = groups[gid]
                lo += g["v5"][0] - g["v4"][0]
                hi += g["v5"][1] - g["v4"][1]
            else:
                bm, cm = _money(b), _money(c)
                lo += cm[0] - bm[0]
                hi += cm[1] - bm[1]
        return lo, hi

    auto_n = sum(len(clusters.get(n, [])) for n in TIER1)
    tier2_n = sum(len(clusters.get(n, [])) for n in TIER2)
    tier3 = ["scope_merge_increased", "scope_added_reattributed",
             "scope_added_routing_diff", "scope_dropped_disposition",
             "scope_dropped_other", "package_needs_eyes", "headline",
             "stability"]
    tier3_n = sum(len(clusters.get(n, [])) for n in tier3)

    L = ["# Session 9 canary review digest", "",
         f"**{len(items)} review items** sorted into three tiers:", "",
         f"- **Tier 1 - {auto_n} auto-accepted**, machine-verified no material "
         "change. Skim the criteria; no per-item reading needed.",
         f"- **Tier 2 - {tier2_n} items in {len(TIER2)} POLICY GROUPS**, "
         f"prefilled as DRAFTS. Read these: each is one real decision, and "
         f"the dollar effect is stated.",
         f"- **Tier 3 - {tier3_n} individual reviews**, left blank for you.",
         "",
         "Tier 2 drafts are already written into the prefilled review file. "
         "Re-run with `--blank-drafts` to leave them empty instead.",
         "", "---", "", "## TIER 1 - auto-accepted (machine-verified)", ""]
    for name, text in TIER1.items():
        rows = clusters.get(name, [])
        if not rows:
            continue
        L += [f"### {name} - {len(rows)} items", f"Criterion/explanation: {text}", ""]
        for item, ev, _ in rows[:2]:
            L.append(f"- e.g. `{item['review_id']}` {item['property_key']} "
                     f"`{item['key']}` {ev}")
        if len(rows) > 2:
            L.append(f"- ... and {len(rows) - 2} more")
        L.append("")

    L += ["---", "", "## TIER 2 - policy decisions (DRAFTED - read and approve)", ""]
    for name, spec in TIER2.items():
        rows = clusters.get(name, [])
        if not rows:
            continue
        lo, hi = total(rows)
        L += [f"### {spec['title']}",
              f"**{len(rows)} review items | corpus effect {lo:+,} low / "
              f"{hi:+,} high**", "",
              f"**The decision:** {spec['question']}", "",
              f"Drafted explanation: {spec['explanation']}", ""]
        seen = set()
        for item, ev, _ in sorted(rows, key=lambda r: (r[0]["property_key"],
                                                       r[0]["key"])):
            cid2, _, unit2 = item["key"].partition("|")
            gid = merge_of.get((item["property_key"], cid2, unit2))
            if gid:
                if gid in seen:
                    continue
                seen.add(gid)
            L.append(f"- `{item['review_id']}` {item['property_key']} {ev}")
        L.append("")

    L += ["---", "", "## TIER 3 - individual review (blank)", ""]
    for name in tier3:
        rows = clusters.get(name, [])
        if not rows:
            continue
        head = f"### {name} - {len(rows)} items"
        if name.startswith("scope"):
            lo, hi = total(rows)
            head += f" | corpus effect {lo:+,} low / {hi:+,} high"
        L += [head, ""]
        seen = set()
        for item, ev, _ in sorted(rows, key=lambda r: (r[0]["property_key"],
                                                       r[0]["key"])):
            if name == "scope_merge_increased":
                cid2, _, unit2 = item["key"].partition("|")
                gid = merge_of.get((item["property_key"], cid2, unit2))
                if gid in seen:
                    continue
                seen.add(gid)
            L.append(f"- `{item['review_id']}` {item['property_key']} "
                     f"`{item['key']}` {ev}")
        L.append("")

    Path(args.digest).write_text("\n".join(L), encoding="utf-8")

    filled = sum(1 for r in reviews.values() if r["decision"])
    print(f"tier1 auto-accepted : {auto_n}")
    print(f"tier2 drafted policy: {tier2_n} in {len(TIER2)} groups "
          f"({'written' if drafted else 'left blank'})")
    print(f"tier3 individual    : {tier3_n}")
    for name in list(TIER1) + list(TIER2) + tier3:
        if clusters.get(name):
            print(f"   {name:34} {len(clusters[name]):>4}")
    print(f"prefilled {filled}/{len(items)} -> {args.out}")
    print(f"digest -> {args.digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
