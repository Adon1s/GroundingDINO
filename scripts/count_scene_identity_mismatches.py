"""One-off QP9-amendment measurement: how often a package candidate's
evidence photos carry scene assignments that conflict with the package room
(the P05 "indoor photo in an exterior package" mode; also the utility-room
bathroom surrogate). Counts only — NOT a pipeline stage; a rule is proposed
later only if the volume warrants it.

Scans the frozen canary run_1 plus post-cutover production runs
(>= 20260821_230000), excluding the two thumbnail-based listings. Evidence
photos are what Terra saw: children -> conditions -> representative photo
keys. A photo whose scene group is "other" is never counted as a conflict.

Run:
  .venv\\Scripts\\python.exe scripts\\count_scene_identity_mismatches.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.pipeline_common import normalize_scene_group  # noqa: E402
from tools.review_cards import (  # noqa: E402
    CANARY_ROOT,
    PROD_ROOT,
    iter_runs,
    load_canary,
    v5_result,
)

THUMBNAIL_LISTINGS = {"redfin_10965375", "redfin_10922002"}
ROOM_TO_GROUPS = {
    "kitchen": {"kitchen"},
    "bathroom": {"bathroom"},
    "bedroom": {"bedroom"},
    "living": {"living_areas"},
    "exterior": {"exterior"},
}
INTERIOR_GROUPS = {"kitchen", "bathroom", "bedroom", "living_areas", "utility"}


def _photo_groups(art: Mapping[str, Any]) -> Dict[str, str]:
    groups = {}
    for key, record in (art.get("photos") or {}).items():
        scene = (record or {}).get("scene") or {}
        groups[str(key)] = normalize_scene_group(scene.get("group"))
    return groups


def _mode(room: str, group: str) -> str:
    if room == "exterior" and group in INTERIOR_GROUPS:
        return "interior_photo_in_exterior_package"
    if room != "exterior" and group == "exterior":
        return "exterior_photo_in_interior_package"
    if room == "bathroom" and group == "utility":
        return "utility_photo_in_bathroom_package"
    return "cross_room"


def scan_artifact(
    prop: str, art: Mapping[str, Any], counts: Counter, examples: List[str]
) -> int:
    result = v5_result(art)
    if result is None:
        return 0
    photo_groups = _photo_groups(art)
    work = {str(w.get("work_item_id")): w
            for w in result.get("work_items") or []}
    evidence = {str(e.get("condition_id")): e
                for e in result.get("evidence_facts") or []}
    checked = 0
    for candidate in result.get("package_candidates") or []:
        room = str(candidate.get("room") or "")
        expected = ROOM_TO_GROUPS.get(room)
        if candidate.get("display_only") or expected is None:
            continue
        checked += 1
        mismatched = False
        for child in candidate.get("child_work_item_ids") or []:
            for condition_id in (work.get(str(child)) or {}).get("condition_ids") or []:
                fact = evidence.get(str(condition_id)) or {}
                for key in fact.get("representative_photo_keys") or []:
                    group = photo_groups.get(str(key), "other")
                    if group == "other" or group in expected:
                        continue
                    mismatched = True
                    mode = _mode(room, group)
                    counts[f"photos_{mode}"] += 1
                    if counts[f"photos_{mode}"] <= 3:
                        examples.append(
                            f"{prop} {candidate.get('package_type')}@"
                            f"{candidate.get('estimate_unit_id')}: {key} "
                            f"scene={group} ({mode})"
                        )
        counts["candidates_with_mismatch"] += int(mismatched)
    counts["candidates_checked"] += checked
    return checked


def main() -> int:
    report: Dict[str, Any] = {}
    run1, _ = load_canary(CANARY_ROOT)
    for source, artifacts in (
        ("canary_run_1", [(prop, art) for prop, (_, art) in sorted(run1.items())]),
        ("production", [
            (prop, art)
            for prop, _, _, art in iter_runs(PROD_ROOT)
            if prop not in THUMBNAIL_LISTINGS
        ]),
    ):
        counts: Counter = Counter()
        examples: List[str] = []
        listings = 0
        for prop, art in artifacts:
            listings += bool(scan_artifact(prop, art, counts, examples))
        report[source] = {
            "listings": listings,
            "counts": dict(sorted(counts.items())),
            "examples": examples,
        }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
