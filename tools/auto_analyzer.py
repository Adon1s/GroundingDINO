#!/usr/bin/env python3
"""
Auto-Analyzer for GroundingDINO Pipeline
-----------------------------------------
Orchestrates: Scene Classification → Detection → Verification
Uses settings from pipeline_config.py
"""

import os
import sys
import json
import time
import uuid
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Import configuration
try:
    import pipeline_config as cfg
except ImportError:
    print("ERROR: pipeline_config.py not found!")
    sys.exit(1)

# Console encoding safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Logging
logging.basicConfig(
    level=logging.DEBUG if cfg.DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Data Classes ─────────────────────────────────────────────────────────────
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
    results: List[ImageAnalysisResult]
    artifacts_dir: str
    total_processing_time: float
    parameters: Dict[str, Any]


# ── Auto-Analyzer Class ──────────────────────────────────────────────────────
class AutoAnalyzer:
    def __init__(self,
                 python_exe: str = sys.executable,
                 artifacts_root: str = None,
                 box_threshold: float = None,
                 text_threshold: float = None,
                 chip_margin: float = None,
                 max_keywords: int = None,
                 include_common: bool = None,
                 include_conditions: bool = None,
                 skip_verification: bool = None,
                 debug: bool = None):
        
        # Use config defaults if not specified
        self.python_exe = Path(python_exe)
        self.artifacts_root = Path(artifacts_root or cfg.ARTIFACTS_ROOT)
        self.box_threshold = box_threshold if box_threshold is not None else cfg.BOX_THRESHOLD
        self.text_threshold = text_threshold if text_threshold is not None else cfg.TEXT_THRESHOLD
        self.chip_margin = chip_margin if chip_margin is not None else cfg.CHIP_MARGIN
        self.max_keywords = max_keywords if max_keywords is not None else cfg.MAX_KEYWORDS
        self.include_common = include_common if include_common is not None else cfg.INCLUDE_COMMON
        self.include_conditions = include_conditions if include_conditions is not None else cfg.INCLUDE_CONDITIONS
        self.skip_verification = skip_verification if skip_verification is not None else cfg.SKIP_VERIFICATION
        self.debug = debug if debug is not None else cfg.DEBUG_MODE
        
        # Validate environment
        self._validate_environment()

    def _validate_environment(self):
        """Validate that required components are available."""
        required_paths = {
            "Python executable": self.python_exe,
            "Scene classifier": cfg.TOOLS_DIR / "scene_classifier.py",
            "GDINO config": cfg.GDINO_CONFIG,
            "GDINO checkpoint": cfg.GDINO_CHECKPOINT,
            "GDINO infer script": cfg.GDINO_INFER_SCRIPT
        }
        
        if not self.skip_verification:
            required_paths["Chip verifier"] = cfg.TOOLS_DIR / "chip_verifier.py"
        
        missing = []
        for name, path in required_paths.items():
            if not path.exists():
                missing.append(f"{name}: {path}")
        
        if missing:
            error_msg = "Missing required components:\n" + "\n".join(missing)
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Create artifacts root
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
        logger.info(f"Classifying scene: {image_path.name}")
        
        cmd = [
            self.python_exe,
            cfg.TOOLS_DIR / "scene_classifier.py",
            str(image_path),
            "--model", cfg.LM_STUDIO_MODEL,
            "--lm-studio-url", cfg.LM_STUDIO_URL,
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
            
            if not prompt and keywords:
                prompt = ". ".join(keywords) + "."
            
            return scene, conf, reasoning, keywords, prompt
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse scene classification: {e}")
            return "unknown", 0.0, "Parse error", [], ""

    def run_detection(self, 
                     image_path: Path,
                     output_dir: Path,
                     text_prompt: str) -> Dict[str, Any]:
        """Run GroundingDINO detection on a single image."""
        logger.info(f"Running detection: {image_path.name}")
        
        cmd = [
            self.python_exe,
            cfg.GDINO_INFER_SCRIPT,
            "--config_file", cfg.GDINO_CONFIG,
            "--checkpoint_path", cfg.GDINO_CHECKPOINT,
            "--image_path", str(image_path),
            "--text_prompt", text_prompt,
            "--output_dir", str(output_dir),
            "--box_threshold", str(self.box_threshold),
            "--text_threshold", str(self.text_threshold),
            "--extract-chips",
            "--chip-margin", str(self.chip_margin),
        ]
        
        if cfg.CPU_ONLY:
            cmd.append("--cpu-only")
        
        if cfg.COMPUTE_CHIP_QUALITY:
            cmd.append("--chip-quality")
        
        if cfg.CREATE_THUMBNAILS:
            cmd.extend(["--create-thumbnail", "--thumbnail-size", str(cfg.THUMBNAIL_SIZE)])
        
        code, stdout, stderr = self._run_command(cmd)
        
        if code != 0:
            logger.error(f"Detection failed: {stderr}")
            return {"error": stderr, "code": code}
        
        # Load results
        pred_json = output_dir / "pred.json"
        if pred_json.exists():
            with open(pred_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"error": "pred.json not produced"}

    def run_verification(self, output_dir: Path) -> Optional[Dict[str, Any]]:
        """Run chip verification on detection results."""
        if self.skip_verification:
            return {"skipped": True}
        
        logger.info(f"Verifying detections in: {output_dir.name}")
        
        cmd = [
            self.python_exe,
            cfg.TOOLS_DIR / "chip_verifier.py",
            str(output_dir),
            "--model", cfg.LM_STUDIO_MODEL,
            "--lm-studio-url", cfg.LM_STUDIO_URL,
            "--max-chips", str(cfg.MAX_CHIPS_PER_DETECTION),
        ]
        
        if self.debug:
            cmd.append("--debug")
        
        code, stdout, stderr = self._run_command(cmd)
        
        if code != 0:
            logger.error(f"Verification failed: {stderr}")
            return {"error": stderr, "code": code}
        
        # Load verification results
        ver_json = output_dir / "verification_results.json"
        if ver_json.exists():
            with open(ver_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None

    def analyze_image(self,
                     image_path: Path,
                     output_dir: Path) -> ImageAnalysisResult:
        """Analyze a single image through the complete pipeline."""
        t0 = time.time()
        
        try:
            # 1. Classify scene and get keywords
            scene, scene_conf, reasoning, keywords, prompt = self.classify_scene(image_path)
            
            logger.info(f"  Scene: {scene} (confidence: {scene_conf:.2f})")
            logger.info(f"  Keywords: {len(keywords)} objects")
            
            if not prompt:
                logger.warning(f"  No detection prompt for {scene}, skipping detection")
                return ImageAnalysisResult(
                    image_path=str(image_path),
                    scene=scene,
                    scene_confidence=scene_conf,
                    keywords_used=keywords,
                    detection_count=0,
                    output_dir=str(output_dir),
                    processing_time=time.time() - t0,
                    error="No detection prompt generated"
                )
            
            # 2. Run detection
            detection_result = self.run_detection(image_path, output_dir, prompt)
            
            if "error" in detection_result:
                return ImageAnalysisResult(
                    image_path=str(image_path),
                    scene=scene,
                    scene_confidence=scene_conf,
                    keywords_used=keywords,
                    detection_count=0,
                    output_dir=str(output_dir),
                    processing_time=time.time() - t0,
                    error=detection_result["error"]
                )
            
            detections = detection_result.get("detections", [])
            detection_count = len(detections)
            
            logger.info(f"  Detections: {detection_count}")
            
            # 3. Verify detections (optional)
            verified_count = None
            if not self.skip_verification and detection_count > 0:
                verification_result = self.run_verification(output_dir)
                if verification_result and "summary" in verification_result:
                    verified_count = verification_result["summary"].get("valid", 0)
                    logger.info(f"  Verified: {verified_count}/{detection_count}")
            
            return ImageAnalysisResult(
                image_path=str(image_path),
                scene=scene,
                scene_confidence=scene_conf,
                keywords_used=keywords,
                detection_count=detection_count,
                verified_count=verified_count,
                output_dir=str(output_dir),
                processing_time=time.time() - t0
            )
        
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            return ImageAnalysisResult(
                image_path=str(image_path),
                scene="unknown",
                scene_confidence=0.0,
                keywords_used=[],
                detection_count=0,
                output_dir=str(output_dir),
                processing_time=time.time() - t0,
                error=str(e)
            )

    def analyze_property(self,
                        property_key: str,
                        images: List[Path]) -> PropertyAnalysisJob:
        """Analyze all images for a property."""
        t0 = time.time()
        
        # Create job directory
        job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        job_dir = self.artifacts_root / property_key / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Property: {property_key}")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Images: {len(images)}")
        logger.info(f"{'='*60}\n")
        
        results = []
        
        for idx, image_path in enumerate(images):
            logger.info(f"\n[{idx+1}/{len(images)}] Processing: {image_path.name}")
            logger.info(f"{'-'*60}")
            
            output_dir = job_dir / f"img_{idx:03d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = self.analyze_image(image_path, output_dir)
            results.append(result)
        
        # Create job summary
        job = PropertyAnalysisJob(
            job_id=job_id,
            property_key=property_key,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                "skip_verification": self.skip_verification,
            }
        )
        
        return job

    def create_html_report(self, job: PropertyAnalysisJob, output_path: Path):
        """Generate a simple HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Analysis Report - {job.property_key}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>Property Analysis Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Property:</strong> {job.property_key}</p>
        <p><strong>Job ID:</strong> {job.job_id}</p>
        <p><strong>Timestamp:</strong> {job.timestamp}</p>
        <p><strong>Images Processed:</strong> {len(job.results)}</p>
        <p><strong>Total Time:</strong> {job.total_processing_time:.1f} seconds</p>
        <p><strong>Artifacts:</strong> {job.artifacts_dir}</p>
    </div>
    
    <h2>Parameters</h2>
    <ul>
"""
        
        for key, value in job.parameters.items():
            html += f"        <li><strong>{key}:</strong> {value}</li>\n"
        
        html += """    </ul>
    
    <h2>Results</h2>
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
            verified_text = str(result.verified_count) if result.verified_count is not None else "N/A"
            error_class = ' class="error"' if result.error else ''
            
            html += f"""        <tr{error_class}>
            <td>{img_name}</td>
            <td>{result.scene}</td>
            <td>{result.scene_confidence:.2f}</td>
            <td>{len(result.keywords_used)}</td>
            <td>{result.detection_count}</td>
            <td>{verified_text}</td>
            <td>{result.processing_time:.1f}</td>
        </tr>
"""
        
        html += """    </table>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML report saved to: {output_path}")
