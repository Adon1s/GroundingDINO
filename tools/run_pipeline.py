#!/usr/bin/env python3
"""
RealtorVision Analysis Pipeline Runner
---------------------------------------
Simple script to run the complete analysis pipeline for testing.

This version runs from the tools/ directory.

Usage (from tools/ directory):
    python run_pipeline.py --images path/to/image.jpg
    python run_pipeline.py --images img1.jpg img2.jpg img3.jpg
    python run_pipeline.py --images-dir path/to/folder
    python run_pipeline.py --images image.jpg --property-key test_001 --no-verify
    python run_pipeline.py --images-dir path/to/folder --no-summary

All parameters can be configured in pipeline_config.py
"""

import sys
import json
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

# Add tools directory to path if needed
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import configuration
try:
    from tools import pipeline_config as cfg
except ImportError:
    print("ERROR: pipeline_config.py not found!")
    print("Make sure pipeline_config.py is in the same directory as this script.")
    print(f"Looking in: {SCRIPT_DIR}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if cfg.DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def redraw_overlay(image_path: Path,
                   detections: List[Dict[str, Any]],
                   out_path: Path) -> None:
    """
    Draw bounding boxes + labels from the filtered detections onto an image.
    This makes pred.jpg match NMS / ROI outputs.
    """
    if not detections:
        return

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        box = det.get("bbox_xyxy") or det.get("bbox") or det.get("box")
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = map(float, box)
        label = str(det.get("label", "obj"))
        score = float(det.get("score", 0.0))
        text = f"{label}({score:.2f})"

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

        tb = draw.textbbox((0, 0), text, font=font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        text_bg = [x1, max(0, y1 - th), x1 + tw, y1]
        draw.rectangle(text_bg, fill="black")

        draw.text((x1, text_bg[1]), text, fill="white", font=font)

    img.save(out_path)


def _natural_key(p: Path):
    """Natural sort key (so photo_2.jpg comes before photo_10.jpg)."""
    s = p.name.lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def validate_environment():
    """Check that all required files exist and create output directories."""
    # Create artifacts directory if needed
    cfg.ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("✅ Environment validation passed")
    return True


def collect_images(args) -> list:
    """Collect image paths from arguments (with deduplication and natural sorting)."""
    candidates = []

    if args.images:
        # Individual image files specified
        candidates = [Path(p) for p in args.images]
    elif args.images_dir:
        # Directory specified
        img_dir = Path(args.images_dir).resolve()
        if not img_dir.exists():
            logger.error(f"Directory not found: {img_dir}")
            sys.exit(1)

        # Single pass - collect all files (we'll filter by suffix)
        candidates = [p for p in img_dir.iterdir() if p.is_file()]
    else:
        logger.error("Error: Must specify either --images or --images-dir")
        sys.exit(1)

    # Deduplicate and filter
    seen = set()
    unique = []
    removed_non_img = 0
    removed_artifacts = 0

    for p in candidates:
        rp = p.resolve()

        if not rp.exists():
            logger.error(f"Image not found: {rp}")
            sys.exit(1)

        # Check if it's an image file
        if rp.suffix.lower() not in IMG_EXTS:
            removed_non_img += 1
            continue

        # Keep artifacts directory files out (avoid processing previous outputs)
        try:
            if cfg.ARTIFACTS_ROOT.resolve() in rp.parents:
                removed_artifacts += 1
                continue
        except Exception:
            pass

        # Deduplicate
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rp)

    # Natural sort (so photo_2.jpg comes before photo_10.jpg)
    unique.sort(key=_natural_key)

    # Log what was filtered
    if removed_non_img > 0:
        logger.debug(f"Filtered {removed_non_img} non-image file(s)")
    if removed_artifacts > 0:
        logger.debug(f"Filtered {removed_artifacts} file(s) from artifacts directory")

    logger.info(f"📸 Found {len(unique)} unique image(s) (from {len(candidates)} candidates)")

    if not unique:
        logger.error("No images found!")
        sys.exit(1)

    return unique


def run_pipeline(property_key: str, images: list, args):
    """Run the complete analysis pipeline.

    TODO: Rewire to use the orchestrator-based pipeline (AutoAnalyzer removed).
    """
    box_thr = args.box_thr if args.box_thr is not None else cfg.BOX_THRESHOLD
    text_thr = args.text_thr if args.text_thr is not None else cfg.TEXT_THRESHOLD

    logger.info("=" * 70)
    logger.info("🚀 Starting RealtorVision Analysis Pipeline")
    logger.info("=" * 70)
    logger.info(f"Property: {property_key}")
    logger.info(f"Images: {len(images)}")
    logger.info(f"Box Threshold: {box_thr}")
    logger.info(f"Text Threshold: {text_thr}")
    logger.info("=" * 70)

    logger.error("AutoAnalyzer has been removed. This runner needs to be rewired to use the orchestrator pipeline.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run RealtorVision analysis pipeline on property images"
    )

    # Image input - both are optional flags now
    parser.add_argument(
        "--images",
        nargs="+",
        help="Path(s) to image file(s)"
    )
    parser.add_argument(
        "--images-dir",
        help="Directory containing images"
    )

    # Optional property key
    parser.add_argument(
        "--property-key",
        default=None,
        help="Property identifier (default: auto-generated)"
    )

    # Parameter overrides (optional - uses config.py defaults)
    parser.add_argument(
        "--box-thr",
        type=float,
        default=None,
        help=f"Box threshold (default: {cfg.BOX_THRESHOLD})"
    )
    parser.add_argument(
        "--text-thr",
        type=float,
        default=None,
        help=f"Text threshold (default: {cfg.TEXT_THRESHOLD})"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        default=False,
        help="Skip property summary generation"
    )

    # Output options
    parser.add_argument(
        "--output-json",
        help="Save JSON summary to specified path"
    )
    args = parser.parse_args()

    # Validate that at least one image source is provided
    if not args.images and not args.images_dir:
        parser.error("Must specify either --images or --images-dir")

    if args.images and args.images_dir:
        parser.error("Cannot specify both --images and --images-dir")

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Collect images
    images = collect_images(args)

    # Generate property key if not provided
    if args.property_key:
        property_key = args.property_key
    else:
        # Use first image filename as property key
        property_key = f"test_{images[0].stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run pipeline
    try:
        run_pipeline(property_key, images, args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()