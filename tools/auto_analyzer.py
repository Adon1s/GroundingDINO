#!/usr/bin/env python3
"""
Auto-Analyzer for RealtorVision
--------------------------------
Orchestrates the complete automated property image analysis pipeline:
1. Scene classification with keyword generation
2. GroundingDINO detection
3. Chip verification (optional)

Usage:
  python auto_analyzer.py --property-key redfin_12345 --images img1.jpg img2.jpg
  python auto_analyzer.py --property-key zillow_99 --images-dir ./property_photos/
  python auto_analyzer.py --property-key mls_88 --images img.jpg --no-verify

Environment variables (required):
  GDINO_PY            - Python executable for GroundingDINO venv
  GDINO_CONFIG        - Path to GroundingDINO config
  GDINO_CHECKPOINT    - Path to GroundingDINO weights
  GDINO_INFER_SCRIPT  - Path to GroundingDINO inference script (demo/inference_on_a_image.py)
  CHIP_VERIFIER_PY    - Path to chip_verifier.py
  ANALYZER_CLI        - Path to analyzer_cli.py
  ARTIFACTS_ROOT      - Output root directory
  LM_STUDIO_URL       - LM Studio endpoint
  LM_STUDIO_MODEL     - Model name

Optional:
  SCENE_CLASSIFIER_PY - Path to scene_classifier.py
"""

import os
import sys
import json
import time
import uuid
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# ── Console encoding (Windows safety) ─────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration from environment ──────────────────────────────────────────
# Core pipeline components
GDINO_PY = os.getenv("GDINO_PY", sys.executable)
GDINO_CONFIG = os.getenv("GDINO_CONFIG", "")
GDINO_CHECKPOINT = os.getenv("GDINO_CHECKPOINT", "")
GDINO_INFER_SCRIPT = os.getenv("GDINO_INFER_SCRIPT", "")
CHIP_VERIFIER_PY = os.getenv("CHIP_VERIFIER_PY", "")
ANALYZER_CLI = os.getenv("ANALYZER_CLI", "./analyzer_cli.py")
ARTIFACTS_ROOT = os.getenv("ARTIFACTS_ROOT", "./artifacts")

# Auto-analyzer components (assume in same directory if not specified)
SCRIPT_DIR = Path(__file__).parent
SCENE_CLASSIFIER_PY = os.getenv("SCENE_CLASSIFIER_PY", str(SCRIPT_DIR / "scene_classifier.py"))

# LM Studio config
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://192.168.86.143:1234")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "gemma-3-27b-it")

# Analysis parameters
DEFAULT_BOX_THR = 0.30
DEFAULT_TEXT_THR = 0.25
DEFAULT_CHIP_MARGIN = 0.15
DEFAULT_MAX_KEYWORDS = 25

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class ImageAnalysisRequest:
    """Request for analyzing a single image."""
    image_path: Path
    property_key: str
    scene: Optional[str] = None
    keywords: Optional[List[str]] = None
    grounding_prompt: Optional[str] = None


@dataclass
class ImageAnalysisResult:
    """Result from analyzing a single image."""
    image_path: str
    scene: str
    scene_confidence: float
    keywords_used: List[str]
    detection_count: int
    verified_count: Optional[int] = None
    output_dir: str = ""
    processing_time: float = 0.0
    error: Optional[str] = None


@dataclass
class PropertyAnalysisJob:
    """Complete property analysis job."""
    job_id: str
    property_key: str
    timestamp: str
    images: List[ImageAnalysisRequest]
    results: List[ImageAnalysisResult]
    artifacts_dir: str
    total_processing_time: float
    parameters: Dict[str, Any]


# ── Auto-Analyzer Class ──────────────────────────────────────────────────────
class AutoAnalyzer:
    def __init__(self,
                 python_exe: str = GDINO_PY,
                 artifacts_root: str = ARTIFACTS_ROOT,
                 box_threshold: float = DEFAULT_BOX_THR,
                 text_threshold: float = DEFAULT_TEXT_THR,
                 chip_margin: float = DEFAULT_CHIP_MARGIN,
                 max_keywords: int = DEFAULT_MAX_KEYWORDS,
                 include_common: bool = True,
                 include_conditions: bool = False,
                 skip_verification: bool = False,
                 debug: bool = False):

        self.python_exe = Path(python_exe)
        self.artifacts_root = Path(artifacts_root)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.chip_margin = chip_margin
        self.max_keywords = max_keywords
        self.include_common = include_common
        self.include_conditions = include_conditions
        self.skip_verification = skip_verification
        self.debug = debug

        # Validate required paths
        self._validate_environment()

    def _validate_environment(self):
        """Validate that required components are available."""
        required_paths = {
            "Python executable": self.python_exe,
            "Scene classifier": Path(SCENE_CLASSIFIER_PY),
            "Analyzer CLI": Path(ANALYZER_CLI),
            "GDINO config": Path(GDINO_CONFIG),
            "GDINO checkpoint": Path(GDINO_CHECKPOINT),
            "GDINO infer script": Path(GDINO_INFER_SCRIPT)
        }

        # Only require verifier if we're actually verifying
        if not self.skip_verification:
            required_paths["Chip verifier"] = Path(CHIP_VERIFIER_PY)

        missing = []
        for name, path in required_paths.items():
            if not path.exists():
                missing.append(f"{name}: {path}")

        if missing:
            error_msg = "Missing required components:\n" + "\n".join(missing)
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Create artifacts root if needed
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a command and capture output."""
        if self.debug:
            logger.debug(f"Running: {' '.join(str(c) for c in cmd)}")

        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = proc.communicate()
        return proc.returncode or 0, stdout, stderr

    def classify_scene(self, image_path: Path) -> Tuple[str, float, str, List[str], str]:
        """
        Classify the scene type and get keywords for detection.

        Returns:
            Tuple of (scene, confidence, reasoning, keywords, grounding_prompt)
        """
        logger.info(f"Classifying scene for: {image_path.name}")

        cmd = [
            self.python_exe,
            SCENE_CLASSIFIER_PY,
            str(image_path),
            "--model", LM_STUDIO_MODEL,
            "--lm-studio-url", LM_STUDIO_URL,
            "--max-keywords", str(self.max_keywords)
        ]

        if self.include_conditions:
            cmd.append("--include-conditions")

        if not self.include_common:
            cmd.append("--no-common")

        if self.debug:
            cmd.append("--debug")

        code, stdout, stderr = self._run_command(cmd)

        if code != 0:
            logger.error(f"Scene classification failed: {stderr}")
            return "unknown", 0.0, "Classification failed", [], ""

        try:
            result = json.loads(stdout)
            scene = result.get("scene", "unknown")
            conf = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")
            keywords = result.get("keywords", []) or []
            prompt = result.get("groundingdino_prompt", "")

            # Fallback: construct prompt from keywords if not provided
            if not prompt and keywords:
                prompt = ". ".join(keywords) + "."

            return scene, conf, reasoning, keywords, prompt

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse scene classification: {e}")
            return "unknown", 0.0, "Parse error", [], ""

    def run_detection_pipeline(self,
                               property_key: str,
                               images: List[Path],
                               text_prompt: str) -> Dict[str, Any]:
        """Run the GroundingDINO detection and verification pipeline."""
        logger.info(f"Running detection pipeline for {len(images)} image(s)")

        # Build analyzer_cli command
        cmd = [
                  self.python_exe,
                  ANALYZER_CLI,
                  "--property-key", property_key,
                  "--text-prompt", text_prompt,
                  "--box-thr", str(self.box_threshold),
                  "--text-thr", str(self.text_threshold),
                  "--chip-margin", str(self.chip_margin),

                  # ✅ Pass top-level artifacts root, not per-job dir
                  # analyzer_cli will create its own <root>/<property>/<job_id>/ structure
                  "--artifacts-root", str(self.artifacts_root),

                  # ✅ Pass explicit paths instead of relying on env vars
                  "--detect-script", GDINO_INFER_SCRIPT,
                  "--config", GDINO_CONFIG,
                  "--checkpoint", GDINO_CHECKPOINT,

                  # ✅ Pass LM Studio settings for verifier
                  "--lm-url", LM_STUDIO_URL,
                  "--lm-model", LM_STUDIO_MODEL,

                  "--images"
              ] + [str(img) for img in images]

        # Add optional flags
        cmd.extend(["--chip-quality", "--create-thumbnail"])

        # Only pass verifier when not skipping and path is set
        if not self.skip_verification and CHIP_VERIFIER_PY:
            cmd.extend(["--verifier", CHIP_VERIFIER_PY])
        else:
            cmd.append("--no-verify")

        code, stdout, stderr = self._run_command(cmd)

        if code != 0:
            logger.error(f"Detection pipeline failed (exit {code})")
            logger.error(f"STDERR (tail): {stderr[-2000:]}")
            logger.error(f"STDOUT (tail): {stdout[-2000:]}")
            return {"error": stderr, "code": code, "stdout": stdout}

        try:
            return json.loads(stdout)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse detection results: {e}")
            return {"error": str(e), "stdout": stdout}

    def analyze_image(self,
                      image_path: Path,
                      property_key: str) -> ImageAnalysisResult:
        """Analyze a single image through the complete pipeline."""
        t0 = time.time()

        try:
            # Step 1: Classify scene (now returns keywords & prompt)
            scene, scene_conf, reasoning, keywords, grounding_prompt = self.classify_scene(image_path)

            # Step 2: Run detection (single image)
            detection_result = self.run_detection_pipeline(
                property_key=property_key,
                images=[image_path],
                text_prompt=grounding_prompt
            )

            # Handle pipeline errors
            if "error" in detection_result:
                return ImageAnalysisResult(
                    image_path=str(image_path),
                    scene=scene,
                    scene_confidence=scene_conf,
                    keywords_used=keywords,
                    detection_count=0,
                    verified_count=None,
                    output_dir="",
                    processing_time=time.time() - t0,
                    error=f"Detection pipeline failed: {detection_result['error']}"
                )

            # Parse results
            detection_count = 0
            verified_count = None
            img_output_dir = ""

            if "results" in detection_result and detection_result["results"]:
                first_result = detection_result["results"][0]

                if "detection" in first_result:
                    det_data = first_result["detection"]
                    detections = det_data.get("detections", [])
                    detection_count = len(detections)

                if "verification" in first_result and first_result["verification"]:
                    ver_data = first_result["verification"]
                    summary = ver_data.get("summary", {})
                    verified_count = summary.get("valid", 0)

                # Handle both outputDir and output_dir keys
                img_output_dir = first_result.get("outputDir") or first_result.get("output_dir") or ""

            return ImageAnalysisResult(
                image_path=str(image_path),
                scene=scene,
                scene_confidence=scene_conf,
                keywords_used=keywords,
                detection_count=detection_count,
                verified_count=verified_count,
                output_dir=img_output_dir,
                processing_time=time.time() - t0
            )

        except Exception as e:
            logger.error(f"Failed to analyze {image_path}: {e}")
            return ImageAnalysisResult(
                image_path=str(image_path),
                scene="unknown",
                scene_confidence=0.0,
                keywords_used=[],
                detection_count=0,
                processing_time=time.time() - t0,
                error=str(e)
            )

    def analyze_property(self,
                         property_key: str,
                         images: List[Path]) -> PropertyAnalysisJob:
        """Analyze all images for a property."""
        job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir = self.artifacts_root / property_key / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting property analysis job: {job_id}")
        logger.info(f"Property: {property_key}")
        logger.info(f"Images: {len(images)}")
        logger.info(f"Output: {job_dir}")

        t0 = time.time()

        # Prepare requests
        requests = [
            ImageAnalysisRequest(
                image_path=img,
                property_key=property_key
            ) for img in images
        ]

        # Process each image
        results = []
        for i, req in enumerate(requests, 1):
            logger.info(f"\n[{i}/{len(requests)}] Processing: {req.image_path.name}")

            # Analyze image
            result = self.analyze_image(
                image_path=req.image_path,
                property_key=property_key
            )

            results.append(result)

            # Update request with discovered info
            req.scene = result.scene
            req.keywords = result.keywords_used

            # Log progress
            logger.info(f"  Scene: {result.scene} (confidence: {result.scene_confidence:.1%})")
            logger.info(f"  Keywords: {len(result.keywords_used)}")
            logger.info(f"  Detections: {result.detection_count}")
            if result.verified_count is not None:
                logger.info(f"  Verified: {result.verified_count}")

        # Create job summary
        job = PropertyAnalysisJob(
            job_id=job_id,
            property_key=property_key,
            timestamp=datetime.now().isoformat(),
            images=requests,
            results=results,
            artifacts_dir=str(job_dir),
            total_processing_time=time.time() - t0,
            parameters={
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "chip_margin": self.chip_margin,
                "max_keywords": self.max_keywords,
                "include_common": self.include_common,
                "include_conditions": self.include_conditions,
                "skip_verification": self.skip_verification
            }
        )

        # Save job summary
        self._save_job_summary(job, job_dir)

        return job

    def _save_job_summary(self, job: PropertyAnalysisJob, job_dir: Path):
        """Save job summary to JSON file."""
        summary = {
            "job_id": job.job_id,
            "property_key": job.property_key,
            "timestamp": job.timestamp,
            "total_images": len(job.images),
            "total_processing_time": job.total_processing_time,
            "parameters": job.parameters,
            "images": [
                {
                    "path": str(req.image_path),
                    "scene": req.scene,
                    "keywords_count": len(req.keywords) if req.keywords else 0
                } for req in job.images
            ],
            "results": [asdict(r) for r in job.results],
            "statistics": {
                "total_detections": sum(r.detection_count for r in job.results),
                "total_verified": sum(
                    r.verified_count or 0 for r in job.results) if not self.skip_verification else None,
                "scenes_identified": list(set(r.scene for r in job.results)),
                "avg_confidence": sum(r.scene_confidence for r in job.results) / len(job.results) if job.results else 0
            }
        }

        summary_file = job_dir / "job_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Job summary saved to: {summary_file}")

    def create_html_report(self, job: PropertyAnalysisJob, output_path: Path):
        """Create an HTML report for the analysis job."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Property Analysis Report - {job.property_key}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }}
        .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .image-result {{ border: 1px solid #bdc3c7; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .scene {{ font-weight: bold; color: #27ae60; }}
        .error {{ color: #e74c3c; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
    </style>
</head>
<body>
    <h1>Property Analysis Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Property Key:</strong> {job.property_key}</p>
        <p><strong>Job ID:</strong> {job.job_id}</p>
        <p><strong>Timestamp:</strong> {job.timestamp}</p>
        <p><strong>Total Images:</strong> {len(job.images)}</p>
        <p><strong>Processing Time:</strong> {job.total_processing_time:.1f} seconds</p>
    </div>

    <h2>Analysis Results</h2>
    <table>
        <tr>
            <th>Image</th>
            <th>Scene</th>
            <th>Confidence</th>
            <th>Keywords</th>
            <th>Detections</th>
            <th>Verified</th>
            <th>Time (s)</th>
        </tr>
"""

        for result in job.results:
            img_name = Path(result.image_path).name
            verified = result.verified_count if result.verified_count is not None else "N/A"
            error_class = ' class="error"' if result.error else ''
            keyword_count = len(result.keywords_used)

            html += f"""        <tr{error_class}>
            <td>{img_name}</td>
            <td class="scene">{result.scene}</td>
            <td>{result.scene_confidence:.1%}</td>
            <td>{keyword_count}</td>
            <td>{result.detection_count}</td>
            <td>{verified}</td>
            <td>{result.processing_time:.1f}</td>
        </tr>
"""

        html += """    </table>

    <h2>Parameters Used</h2>
    <ul>
"""

        for key, value in job.parameters.items():
            html += f"        <li><strong>{key}:</strong> {value}</li>\n"

        html += """    </ul>
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML report saved to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Automated property image analysis orchestrator"
    )

    # Required arguments
    parser.add_argument(
        "--property-key",
        required=True,
        help="Property identifier (e.g., redfin_12345)"
    )

    # Image input (one of these required)
    img_group = parser.add_mutually_exclusive_group(required=True)
    img_group.add_argument(
        "--images",
        nargs="+",
        help="Path(s) to image files"
    )
    img_group.add_argument(
        "--images-dir",
        help="Directory containing images"
    )

    # Detection parameters
    parser.add_argument(
        "--box-thr",
        type=float,
        default=DEFAULT_BOX_THR,
        help=f"Box threshold (default: {DEFAULT_BOX_THR})"
    )
    parser.add_argument(
        "--text-thr",
        type=float,
        default=DEFAULT_TEXT_THR,
        help=f"Text threshold (default: {DEFAULT_TEXT_THR})"
    )
    parser.add_argument(
        "--chip-margin",
        type=float,
        default=DEFAULT_CHIP_MARGIN,
        help=f"Chip margin (default: {DEFAULT_CHIP_MARGIN})"
    )

    # Keyword parameters
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=DEFAULT_MAX_KEYWORDS,
        help=f"Max keywords per scene (default: {DEFAULT_MAX_KEYWORDS})"
    )
    parser.add_argument(
        "--include-conditions",
        action="store_true",
        help="Include damage/condition detection keywords"
    )
    parser.add_argument(
        "--no-common",
        action="store_true",
        help="Exclude common objects like window, door, light"
    )

    # Pipeline options
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip chip verification step"
    )
    parser.add_argument(
        "--artifacts-root",
        default=ARTIFACTS_ROOT,
        help=f"Output root directory (default: {ARTIFACTS_ROOT})"
    )
    parser.add_argument(
        "--python-exe",
        default=GDINO_PY,
        help="Python executable for GroundingDINO"
    )

    # Output options
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report"
    )
    parser.add_argument(
        "--output-json",
        help="Save results to specified JSON file"
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Collect images
    images = []
    if args.images:
        images = [Path(p) for p in args.images]
    else:
        # Get all images from directory
        img_dir = Path(args.images_dir)
        if not img_dir.exists():
            logger.error(f"Directory not found: {img_dir}")
            sys.exit(1)

        # Common image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        for ext in extensions:
            images.extend(img_dir.glob(f"*{ext}"))
            images.extend(img_dir.glob(f"*{ext.upper()}"))

        if not images:
            logger.error(f"No images found in: {img_dir}")
            sys.exit(1)

    # Validate images exist
    for img in images:
        if not img.exists():
            logger.error(f"Image not found: {img}")
            sys.exit(1)

    logger.info(f"Found {len(images)} image(s) to process")

    # Create analyzer
    try:
        analyzer = AutoAnalyzer(
            python_exe=args.python_exe,
            artifacts_root=args.artifacts_root,
            box_threshold=args.box_thr,
            text_threshold=args.text_thr,
            chip_margin=args.chip_margin,
            max_keywords=args.max_keywords,
            include_common=not args.no_common,
            include_conditions=args.include_conditions,
            skip_verification=args.no_verify,
            debug=args.debug
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Run analysis
    logger.info("=" * 60)
    logger.info("Starting automated property analysis")
    logger.info("=" * 60)

    job = analyzer.analyze_property(
        property_key=args.property_key,
        images=images
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Job ID: {job.job_id}")
    print(f"Property: {job.property_key}")
    print(f"Images Processed: {len(job.results)}")
    print(f"Total Time: {job.total_processing_time:.1f} seconds")
    print(f"Artifacts: {job.artifacts_dir}")

    # Scene breakdown
    scene_counts = {}
    for result in job.results:
        scene_counts[result.scene] = scene_counts.get(result.scene, 0) + 1

    print("\nScenes Detected:")
    for scene, count in sorted(scene_counts.items()):
        print(f"  {scene}: {count}")

    # Detection summary
    total_detections = sum(r.detection_count for r in job.results)
    total_verified = sum(r.verified_count or 0 for r in job.results)

    print(f"\nTotal Detections: {total_detections}")
    if not args.no_verify:
        print(f"Total Verified: {total_verified}")

    # Save outputs
    if args.output_json:
        output_path = Path(args.output_json)
        summary = {
            "job_id": job.job_id,
            "property_key": job.property_key,
            "timestamp": job.timestamp,
            "artifacts_dir": job.artifacts_dir,
            "total_processing_time": job.total_processing_time,
            "parameters": job.parameters,
            "results": [asdict(r) for r in job.results]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")

    if args.html_report:
        report_path = Path(job.artifacts_dir) / "report.html"
        analyzer.create_html_report(job, report_path)
        print(f"HTML report saved to: {report_path}")

    # Print to stdout as JSON (for pipeline integration)
    output = {
        "success": True,
        "job_id": job.job_id,
        "property_key": job.property_key,
        "artifacts_dir": job.artifacts_dir,
        "total_images": len(job.results),
        "total_detections": total_detections,
        "total_verified": total_verified if not args.no_verify else None,
        "scenes": list(scene_counts.keys())
    }

    print("\n" + json.dumps(output, indent=2))


if __name__ == "__main__":
    main()