#!/usr/bin/env python3
"""
GroundingDINO Analysis Pipeline Runner
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
    """Check that all required files exist."""
    required_paths = {
        "GDINO Config": cfg.GDINO_CONFIG,
        "GDINO Checkpoint": cfg.GDINO_CHECKPOINT,
        "GDINO Inference Script": cfg.GDINO_INFER_SCRIPT,
        "Scene Classifier": cfg.TOOLS_DIR / "scene_classifier.py",
        "Chip Verifier": cfg.TOOLS_DIR / "chip_verifier.py",
        "Auto Analyzer": cfg.TOOLS_DIR / "auto_analyzer.py",
    }

    missing = []
    for name, path in required_paths.items():
        if not path.exists():
            missing.append(f"  ❌ {name}: {path}")

    if missing:
        logger.error("Missing required files:")
        for m in missing:
            logger.error(m)
        logger.error("\nCheck your GroundingDINO installation and file paths.")
        return False

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
    """Run the complete analysis pipeline."""
    from auto_analyzer import AutoAnalyzer

    # Override config with command line args if provided
    skip_verify = args.no_verify if args.no_verify is not None else cfg.SKIP_VERIFICATION
    box_thr = args.box_thr if args.box_thr is not None else cfg.BOX_THRESHOLD
    text_thr = args.text_thr if args.text_thr is not None else cfg.TEXT_THRESHOLD
    skip_summary = getattr(args, 'no_summary', False)

    logger.info("=" * 70)
    logger.info("🚀 Starting GroundingDINO Analysis Pipeline")
    logger.info("=" * 70)
    logger.info(f"Property: {property_key}")
    logger.info(f"Images: {len(images)}")
    logger.info(f"Box Threshold: {box_thr}")
    logger.info(f"Text Threshold: {text_thr}")
    logger.info(f"Verification: {'SKIPPED' if skip_verify else 'ENABLED'}")
    logger.info(f"Property Summary: {'SKIPPED' if skip_summary else 'ENABLED'}")
    logger.info("=" * 70)

    try:
        analyzer = AutoAnalyzer(
            python_exe=sys.executable,
            artifacts_root=str(cfg.ARTIFACTS_ROOT),
            box_threshold=box_thr,
            text_threshold=text_thr,
            chip_margin=cfg.CHIP_MARGIN,
            max_keywords=cfg.MAX_KEYWORDS,
            skip_verification=skip_verify,
            debug=cfg.DEBUG_MODE
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Run analysis
    job = analyzer.analyze_property(
        property_key=property_key,
        images=images
    )

    # Print summary
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Job ID:           {job.job_id}")
    print(f"Property:         {job.property_key}")
    print(f"Images Processed: {len(job.results)}")
    print(f"Processing Time:  {job.total_processing_time:.1f} seconds")
    print(f"Artifacts:        {job.artifacts_dir}")

    # Scene breakdown
    scene_counts = {}
    for result in job.results:
        scene_counts[result.scene] = scene_counts.get(result.scene, 0) + 1

    if scene_counts:
        print("\n📊 Scenes Detected:")
        for scene, count in sorted(scene_counts.items()):
            print(f"  • {scene:20} {count}")

    # Detection summary
    total_detections = sum(r.detection_count for r in job.results)
    total_verified = sum(r.verified_count or 0 for r in job.results)

    print(f"\n🔍 Detection Summary:")
    print(f"  Total Detections:  {total_detections}")
    if not skip_verify:
        print(f"  Verified:          {total_verified}")
        if total_detections > 0:
            print(f"  Verification Rate: {(total_verified / total_detections) * 100:.1f}%")

    # Save outputs if configured
    artifacts_path = Path(job.artifacts_dir)

    # Save photo_intel (includes deterministic summary_v1 + property_summary.json)
    photo_intel_path = analyzer.save_photo_intel(
        job,
        artifacts_path / "photo_intel.json",
    )

    # Check if property summary was generated
    summary_path = artifacts_path / "property_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)

            print("\n" + "=" * 70)
            print("📋 PROPERTY SUMMARY")
            print("=" * 70)
            print(f"  Overall Condition:   {summary_data.get('overall_condition', 'N/A').upper()}")
            print(f"  Investment Verdict:  {summary_data.get('investment_verdict', 'N/A').replace('_', ' ').upper()}")
            print(f"  Renovation Scope:    {summary_data.get('renovation_scope', 'N/A')}")
            print(
                f"  Issues Found:        {summary_data.get('total_issues_found', 0)} across {summary_data.get('total_images_analyzed', 0)} images")

            # Show risk flags if any
            risk_flags = summary_data.get('risk_flags', [])
            if risk_flags:
                print(f"\n  ⚠️  Risk Flags ({len(risk_flags)}):")
                for flag in risk_flags[:3]:
                    print(f"     • {flag}")
                if len(risk_flags) > 3:
                    print(f"     ... and {len(risk_flags) - 3} more")

            # Show top priorities
            priorities = summary_data.get('renovation_priorities', [])
            if priorities:
                print(f"\n  🔧 Top Renovation Priorities:")
                for i, priority in enumerate(priorities[:3], 1):
                    print(f"     {i}. {priority}")
                if len(priorities) > 3:
                    print(f"     ... and {len(priorities) - 3} more")

            # Show investment rationale
            rationale = summary_data.get('investment_rationale', '')
            if rationale:
                print(f"\n  💡 Rationale: {rationale[:150]}{'...' if len(rationale) > 150 else ''}")

        except Exception as e:
            logger.warning(f"Could not display summary: {e}")

    if cfg.SAVE_JSON_SUMMARY or args.output_json:
        json_path = artifacts_path / "summary.json"
        if args.output_json:
            json_path = Path(args.output_json)

        summary = {
            "job_id": job.job_id,
            "property_key": job.property_key,
            "timestamp": job.timestamp,
            "artifacts_dir": job.artifacts_dir,
            "processing_time": job.total_processing_time,
            "parameters": job.parameters,
            "image_count": len(job.results),
            "photo_intel_path": str(photo_intel_path),
            "property_summary_path": str(summary_path) if summary_path.exists() else None,
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 JSON Summary: {json_path}")

    print(f"📁 Photo Intel:  {photo_intel_path}")

    if summary_path.exists():
        print(f"📋 Property Summary: {summary_path}")

    print("\n" + "=" * 70)

    return job


def main():
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO analysis pipeline on property images"
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
        "--no-verify",
        action="store_true",
        default=None,
        help="Skip chip verification (faster testing)"
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