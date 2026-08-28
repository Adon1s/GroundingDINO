"""Session 8 offline analyses over the baseline replay outputs.

Two measurements, both consuming scripts/replay_renovation_architecture.py
--label baseline output (which already embeds the stored v4 final_rehab):

1. group-caps: simulate v4's GROUP_BUDGET_CAPS stack rules over the v5
   standalone work items. Two honesty buckets — (a) items v4 actually grouped
   (estimable: estimate block with tier high/medium; every such item carries
   an explicit group) get the full v4 stack rule; (b) v4-invisible/ungrouped
   items are reported separately and never capped, because v4 never priced
   them and blanket-applying the "other" fallback would fabricate a parity
   target. v4 caps unfactored sums and scales at the end
   (tools/renovation_estimate_v4.py:337-338), so the caps are scaled by each
   property's cost factor before comparing against v5's factored dollars.

2. floor-trace: decompose headline_v5 - headline_v4 per property into
   class-surfaced standalone scope, class absorbed-child allowances, package
   floor lift (effective above owned-child truth), and residual.

Usage:
  .venv\\Scripts\\python.exe scripts\\analyze_session8_canary.py \
      --replay artifacts_canary\\renovation_session6_20260816_02\\analysis_session8\\baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.renovation_estimate import GROUP_BUDGET_CAPS  # noqa: E402

CATALOG_PATH = REPO_ROOT / "tools" / "issue_catalog_kind_v2.json"
_STACK_PRECEDENCE = ("max_only", "group_cap", "sum")


def _catalog_meta() -> Dict[str, Dict[str, Any]]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    meta: Dict[str, Dict[str, Any]] = {}
    for item in catalog["items"]:
        estimate = item.get("estimate") or {}
        tier = estimate.get("estimate_tier", "minor")
        estimable = isinstance(item.get("estimate"), dict) and tier in ("high", "medium")
        meta[item["id"]] = {
            "v4_estimable": estimable,
            "group": estimate.get("group", "other"),
            "explicit_group": "group" in estimate,
            "stack_behavior": estimate.get("stack_behavior", "sum"),
            # The 48-item class: v4-invisible but v5-billable.
            "v5_class": not estimable
            and bool(item.get("cost") or item.get("work_item_code")),
        }
    return meta


def _dominant(behaviors: List[str]) -> str:
    for behavior in _STACK_PRECEDENCE:
        if behavior in behaviors:
            return behavior
    return "sum"


def group_caps_property(
    result: Mapping[str, Any], meta: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    factor = float(result["standalone_estimate"]["property_cost_factor"])
    active = [w for w in result["work_items"] if w["status"] == "active"]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    bucket_b = {"items": 0, "low": 0, "high": 0}
    mixed = 0
    for work in active:
        metas = [meta[cid] for cid in work["catalog_item_ids"]]
        estimable = [m for m in metas if m["v4_estimable"]]
        if not estimable:
            bucket_b["items"] += 1
            bucket_b["low"] += work["low"]
            bucket_b["high"] += work["high"]
            continue
        if len(estimable) != len(metas):
            mixed += 1
        member_groups = sorted({m["group"] for m in estimable})
        groups.setdefault(member_groups[0], []).append({
            "low": work["low"], "high": work["high"],
            "behaviors": [m["stack_behavior"] for m in estimable],
            "multi_group": len(member_groups) > 1,
        })
    per_group = {}
    removed_low = removed_high = 0
    for group in sorted(groups):
        members = groups[group]
        raw_low = sum(m["low"] for m in members)
        raw_high = sum(m["high"] for m in members)
        dominant = _dominant([b for m in members for b in m["behaviors"]])
        best = max(members, key=lambda m: m["high"])
        if dominant == "max_only":
            capped = (best["low"], best["high"])
        elif dominant == "group_cap":
            cap_low, cap_high = GROUP_BUDGET_CAPS.get(group, (200, 5_000))
            cap_low = int(round(cap_low * factor))
            cap_high = int(round(cap_high * factor))
            capped = (
                max(min(raw_low, cap_low), best["low"]),
                max(min(raw_high, cap_high), best["high"]),
            )
        else:
            capped = (raw_low, raw_high)
        per_group[group] = {
            "items": len(members),
            "stack_behavior": dominant,
            "raw": [raw_low, raw_high],
            "capped": list(capped),
            "removed": [raw_low - capped[0], raw_high - capped[1]],
            "multi_group_members": sum(1 for m in members if m["multi_group"]),
        }
        removed_low += raw_low - capped[0]
        removed_high += raw_high - capped[1]
    return {
        "factor": factor,
        "per_group": per_group,
        "removed": [removed_low, removed_high],
        "bucket_b_uncapped": bucket_b,
        "mixed_estimability_work_items": mixed,
    }


def floor_trace_property(
    result: Mapping[str, Any],
    v4: Mapping[str, Any],
    meta: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    work = {w["work_item_id"]: w for w in result["work_items"] if w["status"] == "active"}

    def _is_class(item: Mapping[str, Any]) -> bool:
        flags = [meta[cid]["v5_class"] for cid in item["catalog_item_ids"]]
        return all(flags)

    mixed = sum(
        1 for w in work.values()
        if any(meta[c]["v5_class"] for c in w["catalog_item_ids"]) and not _is_class(w)
    )
    class_standalone = [0, 0]
    for entry in result["coverage_ledger"]:
        if entry["representation"] == "standalone" and _is_class(work[entry["work_item_id"]]):
            class_standalone[0] += entry["low"]
            class_standalone[1] += entry["high"]
    floor_lift = [0, 0]
    class_absorbed = [0, 0]
    for app in result["package_applications"]:
        if app["status"] != "applied":
            continue
        owned = app["absorbed_work_item_ids"]
        child_low = sum(work[c]["low"] for c in owned)
        child_high = sum(work[c]["high"] for c in owned)
        floor_lift[0] += max(0, app["effective_low"] - child_low)
        floor_lift[1] += max(0, app["effective_high"] - child_high)
        for child in owned:
            if _is_class(work[child]):
                class_absorbed[0] += work[child]["low"]
                class_absorbed[1] += work[child]["high"]
    headline = result["totals"]["headline"]
    delta = [headline["low"] - v4["low"], headline["high"] - v4["high"]]
    residual = [
        delta[0] - class_standalone[0] - class_absorbed[0] - floor_lift[0],
        delta[1] - class_standalone[1] - class_absorbed[1] - floor_lift[1],
    ]
    return {
        "v4_headline": [v4["low"], v4["high"]],
        "v5_headline": [headline["low"], headline["high"]],
        "delta": delta,
        "class_standalone": class_standalone,
        "class_absorbed_child_allowances": class_absorbed,
        "package_floor_lift": floor_lift,
        "residual": residual,
        "mixed_class_work_items": mixed,
    }


def _sum_into(total: Dict[str, List[int]], row: Mapping[str, Any], keys) -> None:
    for key in keys:
        total.setdefault(key, [0, 0])
        total[key][0] += row[key][0]
        total[key][1] += row[key][1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out or args.replay.parent
    meta = _catalog_meta()

    caps_rows: Dict[str, Any] = {}
    trace_rows: Dict[str, Any] = {}
    for path in sorted(args.replay.glob("*.replay.json")):
        key = path.name.replace(".replay.json", "")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["result"]
        caps_rows[key] = group_caps_property(result, meta)
        trace_rows[key] = floor_trace_property(
            result, payload["v4_final_rehab"], meta
        )

    (out_dir / "group_caps.json").write_text(
        json.dumps(caps_rows, indent=1, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "floor_trace.json").write_text(
        json.dumps(trace_rows, indent=1, sort_keys=True), encoding="utf-8"
    )

    # ── group caps summary ──
    print("== group caps (bucket a: v4-estimable grouped items, caps factor-scaled) ==")
    corpus_removed = [0, 0]
    corpus_b = [0, 0]
    by_group: Dict[str, List[int]] = {}
    for key, row in caps_rows.items():
        corpus_removed[0] += row["removed"][0]
        corpus_removed[1] += row["removed"][1]
        corpus_b[0] += row["bucket_b_uncapped"]["low"]
        corpus_b[1] += row["bucket_b_uncapped"]["high"]
        if row["removed"] != [0, 0]:
            print(f"  {key}: removed {row['removed'][0]:,}/{row['removed'][1]:,}")
        for group, g in row["per_group"].items():
            entry = by_group.setdefault(group, [0, 0, 0])
            entry[0] += g["removed"][0]
            entry[1] += g["removed"][1]
            entry[2] += g["items"]
    print(f"  CORPUS removed by v4-style caps: {corpus_removed[0]:,}/{corpus_removed[1]:,}")
    print(f"  CORPUS bucket-b (v4-invisible/ungrouped, never capped): {corpus_b[0]:,}/{corpus_b[1]:,}")
    print("  removed by group (low/high/items):")
    for group, (low, high, n) in sorted(by_group.items(), key=lambda kv: -kv[1][1]):
        if (low, high) != (0, 0):
            print(f"    {group:15s} {low:8,} {high:9,}  ({n} items)")

    # ── floor trace summary ──
    print("== floor trace (v5 headline - v4 final_rehab decomposition) ==")
    corpus: Dict[str, List[int]] = {}
    keys = (
        "delta", "class_standalone", "class_absorbed_child_allowances",
        "package_floor_lift", "residual",
    )
    for key, row in trace_rows.items():
        _sum_into(corpus, row, keys)
    for key in ("redfin_11077450", "redfin_10806500", "redfin_10803207"):
        row = trace_rows[key]
        print(f"  {key}: delta {row['delta'][0]:+,}/{row['delta'][1]:+,} = "
              f"class_standalone {row['class_standalone'][0]:,}/{row['class_standalone'][1]:,} "
              f"+ class_absorbed {row['class_absorbed_child_allowances'][0]:,}/{row['class_absorbed_child_allowances'][1]:,} "
              f"+ floor_lift {row['package_floor_lift'][0]:,}/{row['package_floor_lift'][1]:,} "
              f"+ residual {row['residual'][0]:+,}/{row['residual'][1]:+,}")
    print("  CORPUS: " + " | ".join(
        f"{key} {corpus[key][0]:+,}/{corpus[key][1]:+,}" for key in keys
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
