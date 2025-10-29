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

All parameters can be configured in pipeline_config.py
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add tools directory to path if needed
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import configuration
try:
    import pipeline_config as cfg
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
    """Collect image paths from arguments."""
    images = []

    if args.images:
        # Individual image files specified
        images = [Path(p).resolve() for p in args.images]
    elif args.images_dir:
        # Directory specified
        img_dir = Path(args.images_dir).resolve()
        if not img_dir.exists():
            logger.error(f"Directory not found: {img_dir}")
            sys.exit(1)

        # Find all image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        for ext in extensions:
            images.extend(img_dir.glob(f"*{ext}"))
            images.extend(img_dir.glob(f"*{ext.upper()}"))
    else:
        logger.error("Error: Must specify either --images or --images-dir")
        sys.exit(1)

    # Validate all images exist
    for img in images:
        if not img.exists():
            logger.error(f"Image not found: {img}")
            sys.exit(1)

    if not images:
        logger.error("No images found!")
        sys.exit(1)

    logger.info(f"📸 Found {len(images)} image(s) to process")
    return images


def run_pipeline(property_key: str, images: list, args):
    """Run the complete analysis pipeline."""
    from auto_analyzer import AutoAnalyzer

    # Override config with command line args if provided
    skip_verify = args.no_verify if args.no_verify is not None else cfg.SKIP_VERIFICATION
    box_thr = args.box_thr if args.box_thr is not None else cfg.BOX_THRESHOLD
    text_thr = args.text_thr if args.text_thr is not None else cfg.TEXT_THRESHOLD

    logger.info("=" * 70)
    logger.info("🚀 Starting GroundingDINO Analysis Pipeline")
    logger.info("=" * 70)
    logger.info(f"Property: {property_key}")
    logger.info(f"Images: {len(images)}")
    logger.info(f"Box Threshold: {box_thr}")
    logger.info(f"Text Threshold: {text_thr}")
    logger.info(f"Verification: {'SKIPPED' if skip_verify else 'ENABLED'}")
    logger.info("=" * 70)

    try:
        analyzer = AutoAnalyzer(
            python_exe=sys.executable,
            artifacts_root=str(cfg.ARTIFACTS_ROOT),
            box_threshold=box_thr,
            text_threshold=text_thr,
            chip_margin=cfg.CHIP_MARGIN,
            max_keywords=cfg.MAX_KEYWORDS,
            include_common=cfg.INCLUDE_COMMON,
            include_conditions=cfg.INCLUDE_CONDITIONS,
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
            "results": [
                {
                    "image_path": r.image_path,
                    "scene": r.scene,
                    "scene_confidence": r.scene_confidence,
                    "keywords_used": r.keywords_used,
                    "detection_count": r.detection_count,
                    "verified_count": r.verified_count,
                    "output_dir": r.output_dir,
                    "processing_time": r.processing_time,
                    "error": r.error
                }
                for r in job.results
            ]
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 JSON Summary: {json_path}")

    if cfg.GENERATE_HTML_REPORT or args.html_report:
        html_path = artifacts_path / "report.html"
        analyzer.create_html_report(job, html_path)
        print(f"📄 HTML Report:  {html_path}")

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

    # Output options
    parser.add_argument(
        "--output-json",
        help="Save JSON summary to specified path"
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report"
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