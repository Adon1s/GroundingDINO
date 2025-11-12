"""
Post-processing utilities for GroundingDINO detections
------------------------------------------------------
Provides NMS (Non-Maximum Suppression) and scene-aware filtering.
"""

from typing import List, Dict, Tuple, Optional


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
                    default_iou: float = 0.5) -> List[Dict]:
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