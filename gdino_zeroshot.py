#!/usr/bin/env python3
"""
gdino_zeroshot.py — One-file Grounding DINO runner (official IDEA-Research repo layout)

What it does
------------
- Ensures you have a config + checkpoint (local or auto-download from Hugging Face).
- Runs zero-shot detection on a single image with a text prompt.
- Saves:
  1) an annotated JPEG with boxes+labels
  2) a JSON with boxes (both normalized and absolute) + scores + phrases
  3) (optional) a binary mask PNG by filling the predicted boxes

Requirements
------------
- Run this **inside** the GroundingDINO repo after installing it:
    python -m venv .venv && source .venv/bin/activate        # (Linux/macOS)
    # or on Windows: py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -e .
    pip install huggingface_hub opencv-python

Usage
-----
Examples:
  python gdino_zeroshot.py \
    --image .asset/cat_dog.jpeg \
    --prompt "cat . dog ." \
    --out outputs/demo_run

  python gdino_zeroshot.py \
    --image /path/to/house.jpg \
    --prompt "front door . garage door . house numbers . mailbox ." \
    --box-threshold 0.25 --text-threshold 0.20 \
    --out outputs/house_test

Notes:
- Separate categories with a period and a trailing space, e.g. "front door . garage door ."
- If --weights / --config are not provided, the script looks for:
    weights/groundingdino_swint_ogc.pth
    groundingdino/config/GroundingDINO_SwinT_OGC.py
  If missing, it will try downloading from Hugging Face:
    repo: ShilongLiu/GroundingDINO
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import cv2

# GroundingDINO utilities (must be installed from the repo: `pip install -e .`)
from groundingdino.util.inference import load_model, load_image, predict, annotate

# Optional: auto-download config/weights from Hugging Face if missing
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


DEFAULT_REPO = "ShilongLiu/GroundingDINO"
DEFAULT_WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
# Try common names for the config on HF; will fall back to the local one if present
HF_CONFIG_CANDIDATES = [
    "GroundingDINO_SwinT_OGC.py",
    "GroundingDINO_SwinT_OGC.cfg.py",
]


def ensure_file(path: Path, *, candidate_filenames: List[str] | None = None, repo: str = DEFAULT_REPO) -> Path:
    """
    If `path` exists, return it. Otherwise, if `candidate_filenames` is provided, attempt to download
    each candidate from HF hub into `path.parent`. Return the first successful path.
    """
    if path.exists():
        return path

    if candidate_filenames is None or not HF_AVAILABLE:
        return path  # caller can check existence and error if needed

    path.parent.mkdir(parents=True, exist_ok=True)
    for name in candidate_filenames:
        try:
            cached = hf_hub_download(repo_id=repo, filename=name)
            dst = path.parent / name
            if Path(cached) != dst:
                os.makedirs(dst.parent, exist_ok=True)
                # Copy file contents (avoid shutil to keep imports minimal)
                with open(cached, "rb") as src, open(dst, "wb") as out:
                    out.write(src.read())
            return dst
        except Exception:
            continue
    return path  # none succeeded


def ensure_weights(path: Path, *, repo: str = DEFAULT_REPO, filename: str = DEFAULT_WEIGHTS_NAME) -> Path:
    if path.exists():
        return path
    if not HF_AVAILABLE:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cached = hf_hub_download(repo_id=repo, filename=filename)
        if Path(cached) != path:
            with open(cached, "rb") as src, open(path, "wb") as out:
                out.write(src.read())
    except Exception:
        pass
    return path


def boxes_norm_to_xyxy_abs(boxes_cxcywh: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert normalized (cx, cy, w, h) in [0,1] to absolute pixel (x0, y0, x1, y1).
    """
    cx = boxes_cxcywh[:, 0] * width
    cy = boxes_cxcywh[:, 1] * height
    bw = boxes_cxcywh[:, 2] * width
    bh = boxes_cxcywh[:, 3] * height
    x0 = cx - bw / 2.0
    y0 = cy - bh / 2.0
    x1 = cx + bw / 2.0
    y1 = cy + bh / 2.0
    xyxy = np.stack([x0, y0, x1, y1], axis=1)
    return xyxy


def save_mask_from_boxes(image_shape: Tuple[int, int, int], boxes_xyxy_abs: np.ndarray, out_path: Path) -> None:
    """
    Save a binary mask PNG where predicted boxes are filled with 255.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x0, y0, x1, y1) in boxes_xyxy_abs.astype(int):
        x0c = max(0, min(w, x0))
        y0c = max(0, min(h, y0))
        x1c = max(0, min(w, x1))
        y1c = max(0, min(h, y1))
        if x1c > x0c and y1c > y0c:
            mask[y0c:y1c, x0c:x1c] = 255
    cv2.imwrite(str(out_path), mask)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Grounding DINO runner (boxes + JSON [+ optional mask])")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--prompt", required=True, help='Text prompt; classes separated by periods, e.g. "door . window ."')
    parser.add_argument("--out", default="outputs/demo_run", help="Output directory")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Box threshold (lower => more boxes)")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text threshold (lower => more matches)")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU mode even if CUDA is available")
    parser.add_argument("--cuda-device", default=None, help='CUDA device id, e.g. "0"; sets CUDA_VISIBLE_DEVICES (ignored if --cpu-only)')
    parser.add_argument("--weights", default="weights/groundingdino_swint_ogc.pth", help="Path to checkpoint .pth")
    parser.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Path to config .py")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt Hugging Face downloads if files are missing")
    parser.add_argument("--save-mask", action="store_true", help="Also save a binary mask PNG (box-filled)")

    args = parser.parse_args()

    img_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.cpu_only:
        device = "cpu"
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        if args.cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure config & weights
    cfg_path = Path(args.config)
    w_path = Path(args.weights)

    if not args.no_download and not cfg_path.exists():
        cfg_path = ensure_file(cfg_path, candidate_filenames=HF_CONFIG_CANDIDATES)

    if not args.no_download and not w_path.exists():
        w_path = ensure_weights(w_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path} (set --config or allow auto-download)")
    if not w_path.exists():
        raise FileNotFoundError(f"Weights not found: {w_path} (set --weights or allow auto-download)")

    # Load model
    model = load_model(str(cfg_path), str(w_path))
    # Move to device if needed
    model = model.to(device)

    # Load image
    image_source, image = load_image(str(img_path))

    # Predict
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )

    # Annotate + save
    annotated = annotate(
        image_source=image_source.copy(),
        boxes=boxes,
        logits=logits,
        phrases=phrases
    )

    # Save annotated image
    annotated_name = out_dir / f"{img_path.stem}__gdino_annotated.jpg"
    cv2.imwrite(str(annotated_name), annotated[:, :, ::-1])  # BGR expected by cv2

    # Prepare JSON with boxes + scores + phrases
    h, w, _ = image_source.shape
    boxes_xyxy_abs = boxes_norm_to_xyxy_abs(boxes, w, h)
    results = []
    for i in range(len(phrases)):
        results.append({
            "phrase": str(phrases[i]),
            "score": float(torch.sigmoid(torch.tensor(logits[i])).item()),  # logits->prob approx
            "box_norm_cxcywh": [float(v) for v in boxes[i].tolist()],
            "box_abs_xyxy": [int(v) for v in boxes_xyxy_abs[i].tolist()],
        })

    json_path = out_dir / f"{img_path.stem}__gdino_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "image": str(img_path),
            "prompt": args.prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "device": device,
            "detections": results
        }, f, indent=2)

    # Optional mask
    if args.save_mask and len(results) > 0:
        mask_path = out_dir / f"{img_path.stem}__gdino_mask.png"
        save_mask_from_boxes(image_source.shape, boxes_xyxy_abs, mask_path)

    print(f"[OK] Saved:\n- {annotated_name}\n- {json_path}")
    if args.save_mask and len(results) > 0:
        print(f"- {mask_path}")


if __name__ == "__main__":
    main()
