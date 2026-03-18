"""
Post-processing utilities for detections
-----------------------------------------
Provides NMS (Non-Maximum Suppression) and scene-aware filtering.
"""

from typing import List, Dict, Tuple, Optional


def _sanitize_box_xyxy(box):
    x1, y1, x2, y2 = map(float, box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def clamp_box_to_image(box, W, H):
    x1, y1, x2, y2 = _sanitize_box_xyxy(box)
    x1 = max(0.0, min(x1, W))
    x2 = max(0.0, min(x2, W))
    y1 = max(0.0, min(y1, H))
    y2 = max(0.0, min(y2, H))
    return x1, y1, x2, y2


def box_area(box):
    x1, y1, x2, y2 = _sanitize_box_xyxy(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_area_pct_of_image(box, W, H):
    a = box_area(clamp_box_to_image(box, W, H))
    return a / float(W * H) if W > 0 and H > 0 else 0.0


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = _sanitize_box_xyxy(a)
    bx1, by1, bx2, by2 = _sanitize_box_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def is_box_fully_contained(inner, outer, eps: float = 0.0) -> bool:
    """Return True if *inner* lies completely inside *outer* (within eps)."""
    ix1, iy1, ix2, iy2 = _sanitize_box_xyxy(inner)
    ox1, oy1, ox2, oy2 = _sanitize_box_xyxy(outer)
    return (
            ix1 >= ox1 - eps
            and iy1 >= oy1 - eps
            and ix2 <= ox2 + eps
            and iy2 <= oy2 + eps
    )


def roi_zone_rect(roi_hint, W, H):
    """
    Map 'top_left' | 'top_center' | 'top_right' |
        'mid_left' | 'center'      | 'mid_right' |
        'bottom_left' | 'bottom_center' | 'bottom_right'
    to pixel XYXY. Unknown -> whole image.
    """
    if roi_hint in ("center", "mid_center"):
        # middle cell of the 3x3 grid
        return (W / 3.0, H / 3.0, 2 * W / 3.0, 2 * H / 3.0)

    rows = {"top": (0.0, H / 3.0), "mid": (H / 3.0, 2 * H / 3.0), "bottom": (2 * H / 3.0, float(H))}
    cols = {"left": (0.0, W / 3.0), "center": (W / 3.0, 2 * W / 3.0), "right": (2 * W / 3.0, float(W))}
    try:
        row_key, col_key = roi_hint.split("_")
        y1, y2 = rows[row_key]
        x1, x2 = cols[col_key]
        return (x1, y1, x2, y2)
    except Exception:
        return (0.0, 0.0, float(W), float(H))


def roi_overlap_ratio(box, roi_hint, W, H):
    """
    Return fraction of the detection *area* that lies in the hinted zone [0..1].
    """
    bx = clamp_box_to_image(box, W, H)
    zx = roi_zone_rect(roi_hint, W, H)
    inter = intersection_area(bx, zx)
    det_a = box_area(bx)
    return inter / det_a if det_a > 0 else 0.0


def _zone_of_center(box, W, H):
    x1, y1, x2, y2 = _sanitize_box_xyxy(box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    col = "left" if cx < W / 3 else ("center" if cx < 2 * W / 3 else "right")
    row = "top" if cy < H / 3 else ("mid" if cy < 2 * H / 3 else "bottom")
    return f"{row}_{col}"


def roi_majority_zone(box, W, H):
    """
    Determine the zone that contains the majority of the object:
    compute overlap with all 9 zones and return the max-overlap zone label.
    """
    zones = [
        "top_left",
        "top_center",
        "top_right",
        "mid_left",
        "center",
        "mid_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ]
    best_zone, best_r = "center", -1.0
    for z in zones:
        r = roi_overlap_ratio(box, z, W, H)
        if r > best_r:
            best_r, best_zone = r, z
    return best_zone, best_r


def apply_roi_hint_bonus_overlap(
        dets,
        roi_hint,
        W,
        H,
        full_bonus=0.06,
        half_bonus=0.03,
        penalty=0.03,
        hi=0.40,
        lo=0.10,
        attach_debug=True,
):
    """
    - If detection overlaps hinted zone > hi: +full_bonus
    - If lo <= overlap < hi          : +half_bonus
    - If overlap < lo and centroid clearly opposite: -penalty
    Adds debug fields: img_frac, overlap_ratio, hint, adj, majority_zone, majority_r.
    """
    if not dets or not roi_hint or roi_hint == "unknown":
        if dets and attach_debug:
            for d in dets:
                try:
                    b = clamp_box_to_image(_to_xyxy(d), W, H)
                except Exception:
                    continue
                mz, mr = roi_majority_zone(b, W, H)
                d.setdefault("roi_debug", {})["majority_zone"] = mz
                d["roi_debug"]["majority_r"] = mr
                d["roi_debug"]["hint"] = roi_hint or "unknown"
                d["roi_debug"]["img_frac"] = box_area_pct_of_image(b, W, H)
        return dets

    for d in dets:
        try:
            x1, y1, x2, y2 = clamp_box_to_image(_to_xyxy(d), W, H)
        except Exception:
            if attach_debug:
                dbg = d.setdefault("roi_debug", {})
                dbg["hint"] = roi_hint
                dbg["adj"] = "parse_fail"
            continue
        box = (x1, y1, x2, y2)

        img_frac = box_area_pct_of_image(box, W, H)
        r = roi_overlap_ratio(box, roi_hint, W, H)
        mz, mr = roi_majority_zone(box, W, H)

        tag = "none"
        if r >= hi:
            d["score"] = float(d.get("score", 0.0)) + full_bonus
            tag = "full_bonus"
        elif r >= lo:
            d["score"] = float(d.get("score", 0.0)) + half_bonus
            tag = "half_bonus"
        else:
            cz = _zone_of_center(box, W, H)
            opp = (
                    ("top" in roi_hint and cz.startswith("bottom"))
                    or ("bottom" in roi_hint and cz.startswith("top"))
                    or (roi_hint.endswith("left") and cz.endswith("right"))
                    or (roi_hint.endswith("right") and cz.endswith("left"))
            )
            if opp:
                d["score"] = float(d.get("score", 0.0)) - penalty
                tag = "penalty"

        if attach_debug:
            dbg = d.setdefault("roi_debug", {})
            dbg["img_frac"] = img_frac
            dbg["overlap_ratio"] = r
            dbg["hint"] = roi_hint
            dbg["adj"] = tag
            dbg["majority_zone"] = mz
            dbg["majority_r"] = mr
    return dets


def _to_xyxy(det: Dict) -> Tuple[float, float, float, float]:
    """
    Normalize your box to [x1,y1,x2,y2].
    Supports det["box"] as either [x1,y1,x2,y2] or [x,y,w,h] (xywh).
    If your JSON uses different keys, adapt here.
    """
    box = det.get("box") or det.get("bbox") or det.get("bbox_xyxy")
    if box is None:
        raise ValueError("Detection missing 'box'/'bbox' field")
    if len(box) != 4:
        raise ValueError("Box must have 4 numbers")

    x1, y1, x2, y2 = box
    # If x2<x1 or y2<y1 we assume xywh and convert
    if x2 < x1 or y2 < y1:
        x, y, w, h = box
        return (x, y, x + w, y + h)
    return (x1, y1, x2, y2)


def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def class_aware_nms(dets: List[Dict],
                    per_class_iou: Optional[Dict[str, float]] = None,
                    default_iou: float = 0.3) -> List[Dict]:
    """
    Suppresses duplicates *within the same label* only.
    Each det dict must have at least: {'label': str, 'score': float, 'box': [4 numbers]}

    Args:
        dets: List of detection dictionaries
        per_class_iou: Optional dict mapping label -> IoU threshold
        default_iou: Default IoU threshold for classes not in per_class_iou

    Returns:
        Filtered list of detections with duplicates removed
    """
    buckets: Dict[str, List[Dict]] = {}
    for d in dets:
        buckets.setdefault(d["label"], []).append(d)

    survivors: List[Dict] = []
    for label, lst in buckets.items():
        thr = (per_class_iou or {}).get(label, default_iou)
        # sort high→low by score
        lst = sorted(lst, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        keep: List[Dict] = []
        while lst:
            best = lst.pop(0)
            keep.append(best)
            b_box = _to_xyxy(best)
            lst = [d for d in lst if _iou(b_box, _to_xyxy(d)) <= thr]
        survivors.extend(keep)

    # stable sort across classes by score (optional)
    survivors.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return survivors


# Optional: simple per-scene caps (keeps highest-score boxes)
DEFAULT_SCENE_CAPS = {
    "bathroom": {"sink": 1, "vanity": 1, "bathtub": 1, "toilet": 1, "faucet": 2},
    "kitchen": {"sink": 1, "refrigerator": 1, "range": 1, "oven": 1, "dishwasher": 1},
}


def enforce_scene_caps(dets: List[Dict], scene: str,
                       caps_map: Optional[Dict[str, Dict[str, int]]] = None) -> List[Dict]:
    """
    Enforce per-scene maximum counts for specific object classes.
    Keeps only the highest-scoring detections up to the cap.

    Args:
        dets: List of detection dictionaries (should be sorted by score)
        scene: Scene type (e.g., "bathroom", "kitchen")
        caps_map: Optional dict mapping scene -> {label: max_count}

    Returns:
        Filtered list of detections respecting scene caps
    """
    caps = (caps_map or DEFAULT_SCENE_CAPS).get(scene, {})
    if not caps:
        return dets
    out, counts = [], {}
    for d in sorted(dets, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        lbl = d["label"]
        if lbl in caps:
            counts[lbl] = counts.get(lbl, 0)
            if counts[lbl] >= caps[lbl]:
                continue
            counts[lbl] += 1
        out.append(d)
    return out


def drop_detections_inside_mirrors(
        dets: List[Dict],
        mirror_labels: Optional[List[str]] = None,
        containment_eps: float = 0.0,
) -> List[Dict]:
    """
    Remove any detection completely contained inside a mirror detection.

    Args:
        dets: List of detection dictionaries.
        mirror_labels: Labels that should be treated as mirrors.
        containment_eps: Optional slack (in pixels) for containment checks.

    Returns:
        Filtered detections.
    """

    if not dets:
        return dets

    label_set = {
        str(lbl).strip().lower() for lbl in (mirror_labels or ["mirror"]) if lbl
    }
    if not label_set:
        return dets

    box_map = {}
    parse_fail_ids = set()
    for det in dets:
        try:
            box_map[id(det)] = _to_xyxy(det)
        except Exception:
            parse_fail_ids.add(id(det))

    if not box_map:
        return dets

    mirror_entries = []
    for det in dets:
        det_id = id(det)
        if det_id not in box_map or det_id in parse_fail_ids:
            continue
        label = str(det.get("label") or "").strip().lower()
        if label in label_set:
            mirror_entries.append((det, box_map[det_id]))

    if not mirror_entries:
        return dets

    # Keep only the "outer" mirrors (drop mirrors contained inside bigger mirrors)
    mirror_entries.sort(key=lambda item: box_area(item[1]), reverse=True)
    active_mirrors = []
    dropped_mirror_ids = set()
    for det, box in mirror_entries:
        if any(
                is_box_fully_contained(box, keep_box, containment_eps)
                for _, keep_box in active_mirrors
        ):
            dropped_mirror_ids.add(id(det))
            continue
        active_mirrors.append((det, box))

    active_boxes = [box for _, box in active_mirrors]

    survivors: List[Dict] = []
    for det in dets:
        det_id = id(det)
        # Keep detections we could not parse safely
        if det_id not in box_map or det_id in parse_fail_ids:
            survivors.append(det)
            continue

        box = box_map[det_id]
        label = str(det.get("label") or "").strip().lower()

        if label in label_set:
            if det_id in dropped_mirror_ids:
                continue
            survivors.append(det)
            continue

        if any(is_box_fully_contained(box, m_box, containment_eps) for m_box in active_boxes):
            continue

        survivors.append(det)

    return survivors


def drop_nested_fixtures(
        dets: List[Dict],
        fixture_labels: Optional[List[str]] = None,
        containment_eps: float = 0.0,
) -> List[Dict]:
    """
    Remove fixture detections that are fully contained in larger fixtures.

    Only detections whose label is in *fixture_labels* are considered for
    dropping. Detections with other labels are returned unchanged.

    Args:
        dets: List of detection dictionaries.
        fixture_labels: Labels that should be treated as light fixtures.
        containment_eps: Optional slack (in pixels) for containment checks.

    Returns:
        Filtered detections.
    """

    if not dets:
        return dets

    label_set = {
        str(lbl).strip().lower()
        for lbl in (fixture_labels or ["light_fixture", "vanity_light", "ceiling_light"])
        if lbl
    }
    if not label_set:
        return dets

    box_map = {}
    parse_fail_ids = set()
    for det in dets:
        try:
            box_map[id(det)] = _to_xyxy(det)
        except Exception:
            parse_fail_ids.add(id(det))

    if not box_map:
        return dets

    fixture_entries = []
    for det in dets:
        det_id = id(det)
        if det_id not in box_map or det_id in parse_fail_ids:
            continue
        label = str(det.get("label") or "").strip().lower()
        if label in label_set:
            fixture_entries.append((det, box_map[det_id]))

    if not fixture_entries:
        return dets

    fixture_entries.sort(key=lambda item: box_area(item[1]), reverse=True)
    active_fixtures = []
    dropped_fixture_ids = set()
    for det, box in fixture_entries:
        if any(
                is_box_fully_contained(box, keep_box, containment_eps)
                for _, keep_box in active_fixtures
        ):
            dropped_fixture_ids.add(id(det))
            continue
        active_fixtures.append((det, box))

    survivors: List[Dict] = []
    for det in dets:
        det_id = id(det)
        if det_id not in box_map or det_id in parse_fail_ids:
            survivors.append(det)
            continue

        label = str(det.get("label") or "").strip().lower()
        if label in label_set and det_id in dropped_fixture_ids:
            continue

        survivors.append(det)

    return survivors


def apply_special_case_filters(
        dets: List[Dict],
        image_size: Optional[Tuple[int, int]] = None,
        config: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """Run any configured special-case filters over the detections."""

    if not dets or not config:
        return dets

    survivors = dets
    mirror_cfg = config.get("mirror_containment")
    if mirror_cfg and mirror_cfg.get("enabled", True):
        survivors = drop_detections_inside_mirrors(
            survivors,
            mirror_labels=mirror_cfg.get("mirror_labels") or ["mirror"],
            containment_eps=float(mirror_cfg.get("containment_eps", 0.0)),
        )

    fixture_cfg = config.get("fixture_collapse")
    if fixture_cfg and fixture_cfg.get("enabled", True):
        survivors = drop_nested_fixtures(
            survivors,
            fixture_labels=fixture_cfg.get("fixture_labels")
                           or ["light_fixture", "vanity_light", "ceiling_light"],
            containment_eps=float(fixture_cfg.get("containment_eps", 0.0)),
        )

    # Future special cases can be chained here using the same pattern.

    return survivors
