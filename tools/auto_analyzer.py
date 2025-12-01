#!/usr/bin/env python3
"""
Auto-Analyzer for GroundingDINO Pipeline
-----------------------------------------
Orchestrates: Scene Classification → Detection → Verification → Property Summary
Uses settings from pipeline_config.py
"""

import os
import sys
import json
import time
import uuid
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from PIL import Image

from run_pipeline import redraw_overlay
from scene_classifier import load_issue_catalog

# Import configuration
try:
    import pipeline_config as cfg
except ImportError:
    print("ERROR: pipeline_config.py not found!")
    sys.exit(1)

# Import renovation cost table from dedicated module
try:
    from renovation_costs import RENOVATION_COST_TABLE
except ImportError:
    print("WARNING: renovation_costs.py not found, using empty cost table")
    RENOVATION_COST_TABLE = {}

# Import NMS postprocessing
try:
    from postprocess import (
        class_aware_nms,
        enforce_scene_caps,
        apply_roi_hint_bonus_overlap,
        apply_special_case_filters,
    )
except ImportError:
    print("ERROR: postprocess.py not found!")
    sys.exit(1)

# Import property summarizer (optional - graceful fallback if not available)
try:
    from property_summarizer import PropertySummarizer
except ImportError:
    PropertySummarizer = None
    print("INFO: property_summarizer.py not found, property summaries will be skipped")

# Console encoding safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Logging
logging.basicConfig(
    level=logging.DEBUG if cfg.DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

default_catalog_path = getattr(
    cfg, "PROJECT_ROOT", Path(__file__).resolve().parent.parent
) / "issue_catalog.json"
ISSUE_CATALOG_PATH = Path(getattr(cfg, "ISSUE_CATALOG_PATH", default_catalog_path))
ISSUE_CATALOG = load_issue_catalog(ISSUE_CATALOG_PATH)

SEVERITY_RANK = {
    "none": 0,
    "minor_repair": 1,
    "moderate_repair": 2,
    "full_replacement": 3,
}


# ── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class ImageAnalysisResult:
    """Result from analyzing a single image."""
    image_path: str
    scene: str
    scene_data: Optional[Dict[str, Any]]
    keywords_used: List[str]
    detection_count: int
    verified_count: Optional[int] = None
    output_dir: str = ""
    processing_time: float = 0.0
    error: Optional[str] = None
    scene_classifier: Optional[Dict[str, Any]] = None


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


def _worst_severity(current: str, new: str) -> str:
    if SEVERITY_RANK.get(new, 0) > SEVERITY_RANK.get(current, 0):
        return new
    return current


def _parse_catalog_flag(flag: Any) -> Tuple[str, str, str]:
    """Normalize catalog flag inputs from dicts or dataclass-like objects."""

    if isinstance(flag, dict):
        present = str(flag.get("present", "")).lower() or "uncertain"
        evidence = str(flag.get("evidence", ""))
        severity = str(flag.get("severity", "none")).lower() or "none"
    else:
        present = str(getattr(flag, "present", "uncertain") or "uncertain").lower()
        evidence = str(getattr(flag, "evidence", ""))
        severity = str(getattr(flag, "severity", "none") or "none").lower()

    if present != "yes":
        severity = "none"

    return present, severity, evidence


def build_renovation_needs(job: "PropertyAnalysisJob") -> Dict[str, Any]:
    """
    Aggregate per-photo catalog_flags + severity into a property-level
    renovation_needs structure.
    """

    issue_meta: Dict[str, Dict[str, str]] = {}
    for item in ISSUE_CATALOG.get("defect_issues", []) or []:
        if isinstance(item, dict) and item.get("id"):
            issue_meta[item["id"]] = {
                "name": item.get("name", item["id"]),
                "category": item.get("category", "unknown"),
            }
    for item in ISSUE_CATALOG.get("opportunity_flags", []) or []:
        if isinstance(item, dict) and item.get("id"):
            issue_meta[item["id"]] = {
                "name": item.get("name", item["id"]),
                "category": item.get("category", "unknown"),
            }

    aggregates: Dict[str, Dict[str, Any]] = {}
    totals_by_category: Dict[str, Dict[str, float]] = {}

    for result in job.results:
        scene_payload = (result.scene_classifier or result.scene_data) or {}
        if not isinstance(scene_payload, dict):
            continue

        flags = scene_payload.get("catalog_flags", {}) or {}
        photo_name = Path(result.image_path).name

        for issue_id, flag in flags.items():
            present, severity, evidence = _parse_catalog_flag(flag)
            if present != "yes":
                continue

            meta = issue_meta.get(issue_id, {"name": issue_id, "category": "unknown"})
            agg = aggregates.setdefault(issue_id, {
                "issue_id": issue_id,
                "name": meta["name"],
                "category": meta["category"],
                "worst_severity": "none",
                "occurrences": 0,
                "present_in_photos": [],
                "sample_evidence": "",
                "est_cost_low": 0.0,
                "est_cost_high": 0.0,
            })

            agg["occurrences"] += 1
            agg["present_in_photos"].append(photo_name)
            agg["worst_severity"] = _worst_severity(agg["worst_severity"], severity)

            if not agg["sample_evidence"] and evidence:
                agg["sample_evidence"] = evidence

            cost_cfg = RENOVATION_COST_TABLE.get(issue_id, {})
            sev_cost = cost_cfg.get(severity)
            if sev_cost:
                low, high = sev_cost
                agg["est_cost_low"] += low
                agg["est_cost_high"] += high

    for issue in aggregates.values():
        cat = issue.get("category", "unknown")
        cat_totals = totals_by_category.setdefault(cat, {
            "est_cost_low": 0.0,
            "est_cost_high": 0.0,
        })
        cat_totals["est_cost_low"] += issue["est_cost_low"]
        cat_totals["est_cost_high"] += issue["est_cost_high"]

    grand_low = sum(cat.get("est_cost_low", 0.0) for cat in totals_by_category.values())
    grand_high = sum(cat.get("est_cost_high", 0.0) for cat in totals_by_category.values())

    issues_sorted = sorted(
        aggregates.values(),
        key=lambda x: x.get("est_cost_high", 0.0),
        reverse=True,
    )

    return {
        "issues": issues_sorted,
        "totals_by_category": totals_by_category,
        "grand_total": {
            "est_cost_low": grand_low,
            "est_cost_high": grand_high,
        },
    }


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

    @staticmethod
    def _normalize_label_for_hint(label: Optional[str]) -> str:
        if not label:
            return ""
        s = str(label).strip().lower()
        s = s.replace("-", " ").replace("_", " ")
        return " ".join(s.split())

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

    def classify_scene(
            self, image_path: Path
    ) -> Tuple[str, str, List[str], str, List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
        """
        Classify the scene type and get keywords for detection.

        Returns:
            Tuple of (scene, reasoning, keywords, grounding_prompt, planner_targets,
            planner_hints, scene_payload)
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

        if self.debug:
            cmd.append("--debug")

        code, stdout, stderr = self._run_command(cmd)

        if code != 0:
            logger.error(f"Scene classification failed: {stderr}")
            empty_payload = self._scene_classifier_payload(
                None, scene_override="unknown", error="Classification failed"
            )
            return "unknown", "Classification failed", [], "", [], {}, empty_payload

        try:
            result = json.loads(stdout)
            scene = result.get("scene", "unknown")
            reasoning = result.get("reasoning", "")
            keywords = result.get("keywords", []) or []
            prompt = result.get("groundingdino_prompt", "")
            targets = result.get("targets", []) or []

            planner_hints: Dict[str, str] = {}
            for target in targets:
                hint = str(target.get("roi_hint", "unknown") or "unknown")
                label_norm = self._normalize_label_for_hint(target.get("label"))
                if label_norm:
                    planner_hints[label_norm] = hint
                for syn in (target.get("synonyms") or []):
                    syn_norm = self._normalize_label_for_hint(syn)
                    if syn_norm and syn_norm not in planner_hints:
                        planner_hints[syn_norm] = hint

            if not prompt and keywords:
                prompt = ". ".join(keywords) + "."

            # Persist the prompt in the scene result for downstream artifacts
            if not result.get("groundingdino_prompt"):
                result["groundingdino_prompt"] = prompt

            # Normalize optional collections so they are always present downstream
            result.setdefault("targets", targets)
            result.setdefault("gdino_terms", [])
            result.setdefault("keywords", keywords)
            result.setdefault("issues_natural_language", [])
            result.setdefault("catalog_flags", {})
            result.setdefault("issue_visual_anchors", [])

            scene_payload = self._scene_classifier_payload(
                result, scene_override=scene
            )
            keywords = scene_payload.get("keywords", keywords) or []

            return (
                scene,
                reasoning,
                keywords,
                prompt,
                targets,
                planner_hints,
                scene_payload,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse scene classification: {e}")
            empty_payload = self._scene_classifier_payload(
                None, scene_override="unknown", error="Parse error"
            )
            return "unknown", "Parse error", [], "", [], {}, empty_payload

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
            (
                scene,
                reasoning,
                keywords,
                prompt,
                planner_targets,
                planner_hints,
                scene_details,
            ) = self.classify_scene(image_path)

            logger.info(f"  Scene: {scene}")
            logger.info(f"  Keywords: {len(keywords)} objects")

            if not prompt:
                logger.warning(f"  No detection prompt for {scene}, skipping detection")
                return ImageAnalysisResult(
                    image_path=str(image_path),
                    scene=scene,
                    scene_data=scene_details,
                    scene_classifier=scene_details,
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
                    scene_data=scene_details,
                    scene_classifier=scene_details,
                    keywords_used=keywords,
                    detection_count=0,
                    output_dir=str(output_dir),
                    processing_time=time.time() - t0,
                    error=detection_result["error"]
                )

            detections = detection_result.get("detections", []) or []
            detection_result["planner_targets"] = planner_targets
            detection_result["planner_hints"] = planner_hints

            size_info = detection_result.get("size") or {}
            img_w = int(size_info.get("width") or 0)
            img_h = int(size_info.get("height") or 0)
            if (img_w <= 0 or img_h <= 0) and image_path.exists():
                try:
                    with Image.open(image_path) as pil_im:
                        img_w, img_h = pil_im.size
                except Exception:
                    pass

            if (
                    getattr(cfg, "ROI_HINTS_ENABLED", False)
                    and detections
                    and img_w > 0
                    and img_h > 0
            ):
                detections_by_class: Dict[str, List[Dict[str, Any]]] = {}
                unlabeled: List[Dict[str, Any]] = []
                for det in detections:
                    if det.get("score") is None:
                        det["score"] = 0.0
                    label_name = str(det.get("label") or "").strip()
                    if not label_name:
                        unlabeled.append(det)
                        continue
                    detections_by_class.setdefault(label_name, []).append(det)

                for label_name, dets in detections_by_class.items():
                    norm_label = self._normalize_label_for_hint(label_name)
                    roi_hint = planner_hints.get(norm_label, "unknown")
                    apply_roi_hint_bonus_overlap(
                        dets=dets,
                        roi_hint=roi_hint,
                        W=img_w,
                        H=img_h,
                        full_bonus=getattr(cfg, "ROI_FULL_BONUS", 0.06),
                        half_bonus=getattr(cfg, "ROI_HALF_BONUS", 0.03),
                        penalty=getattr(cfg, "ROI_PENALTY", 0.03),
                        hi=getattr(cfg, "ROI_OVERLAP_HI", 0.40),
                        lo=getattr(cfg, "ROI_OVERLAP_LO", 0.10),
                        attach_debug=True,
                    )

                if unlabeled:
                    apply_roi_hint_bonus_overlap(
                        dets=unlabeled,
                        roi_hint="unknown",
                        W=img_w,
                        H=img_h,
                        full_bonus=getattr(cfg, "ROI_FULL_BONUS", 0.06),
                        half_bonus=getattr(cfg, "ROI_HALF_BONUS", 0.03),
                        penalty=getattr(cfg, "ROI_PENALTY", 0.03),
                        hi=getattr(cfg, "ROI_OVERLAP_HI", 0.40),
                        lo=getattr(cfg, "ROI_OVERLAP_LO", 0.10),
                        attach_debug=True,
                    )

            # Sort by score for stable behavior
            detections.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)

            # Special-case filters (mirror containment, etc.)
            special_case_cfg = getattr(cfg, "SPECIAL_CASE_FILTERS", {})
            if detections and special_case_cfg:
                pre_case = len(detections)
                detections = apply_special_case_filters(
                    detections,
                    image_size=(img_w, img_h),
                    config=special_case_cfg,
                )
                if pre_case != len(detections):
                    logger.info(
                        f"  Special cases: {pre_case} → {len(detections)} detections"
                    )

            # --- NMS here (AFTER detection, BEFORE verification) ---
            if getattr(cfg, "USE_NMS", True) and detections:
                pre_nms_count = len(detections)
                detections = class_aware_nms(
                    detections,
                    per_class_iou=getattr(cfg, "NMS_PER_CLASS", None),
                    default_iou=getattr(cfg, "NMS_DEFAULT_IOU", 0.3),
                )
                logger.info(f"  NMS: {pre_nms_count} → {len(detections)} detections")

            # Optional: enforce per-scene caps (keeps top-K per class)
            if getattr(cfg, "USE_SCENE_CAPS", False) and detections:
                pre_cap_count = len(detections)
                detections = enforce_scene_caps(
                    detections,
                    scene=scene,
                    caps_map=getattr(cfg, "SCENE_CAPS", None),
                )
                if pre_cap_count != len(detections):
                    logger.info(f"  Scene caps: {pre_cap_count} → {len(detections)} detections")

            if (
                    getattr(cfg, "ROI_HINTS_ENABLED", False)
                    and detections
                    and logger.isEnabledFor(logging.DEBUG)
            ):
                for det in detections:
                    dbg = det.get("roi_debug", {})
                    logger.debug(
                        f"ROI [{det.get('label')}]: score={float(det.get('score', 0.0)):.3f} "
                        f"hint={dbg.get('hint')} adj={dbg.get('adj')} "
                        f"overlap={dbg.get('overlap_ratio', 0):.2f} "
                        f"maj={dbg.get('majority_zone')}({dbg.get('majority_r', 0):.2f}) "
                        f"img%={dbg.get('img_frac', 0):.3f}"
                    )

            # Write survivors back so downstream tools (e.g., chip_verifier) see the filtered set
            detection_result["detections"] = detections

            # Keep top-level counts consistent with filtered detections
            for k in ("count", "num_detections", "detections_count"):
                if k in detection_result:
                    detection_result[k] = len(detections)

            pred_json_path = output_dir / "pred.json"
            with open(pred_json_path, "w", encoding="utf-8") as f:
                json.dump(detection_result, f, indent=2, ensure_ascii=False)

            # (Optional) prune chips for detections that were dropped
            if getattr(cfg, "PRUNE_DROPPED_CHIPS", False):
                try:
                    chip_dir = Path(
                        (detection_result.get("chip_extraction", {}) or {}).get("chips_directory",
                                                                                str(output_dir / "chips"))
                    )
                    keep = set()
                    for d in detections:
                        fn = ((d.get("chip_info") or {}).get("filename"))
                        if fn:
                            keep.add(str((chip_dir / fn).resolve()))

                    if chip_dir.exists():
                        pruned = 0
                        for fp in chip_dir.glob("*"):
                            if keep and str(fp.resolve()) not in keep:
                                try:
                                    fp.unlink()
                                    pruned += 1
                                except Exception:
                                    pass
                        if pruned > 0:
                            logger.info(f"  Pruned {pruned} dropped chip file(s)")
                except Exception as e:
                    logger.warning(f"  Chip pruning failed: {e}")

            # Redraw overlay using filtered detections so pred.jpg matches NMS output
            try:
                nms_overlay = output_dir / "pred_nms.jpg"
                redraw_overlay(image_path, detections, nms_overlay)

                raw_overlay = output_dir / "pred.jpg"
                before_overlay = output_dir / "pred_before.jpg"
                if raw_overlay.exists():
                    try:
                        shutil.copy2(raw_overlay, before_overlay)
                    except Exception:
                        before_overlay.write_bytes(raw_overlay.read_bytes())

                nms_overlay.replace(raw_overlay)

                logger.info("  Overlay updated with NMS/ROI filtered detections")
            except Exception as e:
                logger.warning(f"  Failed to redraw overlay: {e}")

            # Finally, the filtered count
            detection_count = len(detections)

            # 3. Verify detections (optional)
            verified_count = None
            if not self.skip_verification and detection_count > 0:
                logger.info(f"  Verifying detections for {Path(image_path).name} in: {Path(output_dir).name}")
                verification_result = self.run_verification(output_dir)  # ← single call
                if verification_result and "summary" in verification_result:
                    verified_count = verification_result["summary"].get("valid", 0)
                    logger.info(f"  Verified: {verified_count}/{detection_count}")

            return ImageAnalysisResult(
                image_path=str(image_path),
                scene=scene,
                scene_data=scene_details,
                scene_classifier=scene_details,
                keywords_used=keywords,
                detection_count=detection_count,
                verified_count=verified_count,
                output_dir=str(output_dir),
                processing_time=time.time() - t0
            )

        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            fallback_scene = self._scene_classifier_payload(
                None, scene_override="unknown", error=str(e)
            )
            return ImageAnalysisResult(
                image_path=str(image_path),
                scene="unknown",
                scene_data=fallback_scene,
                scene_classifier=fallback_scene,
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

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Property: {property_key}")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Images: {len(images)}")
        logger.info(f"{'=' * 60}\n")

        results = []

        for idx, image_path in enumerate(images):
            logger.info(f"\n[{idx + 1}/{len(images)}] Processing: {image_path.name}")
            logger.info(f"{'-' * 60}")

            output_dir = job_dir / f"img_{idx:03d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"\n[{idx + 1}/{len(images)}] Processing: {image_path.name} → {output_dir.name}")

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
                "use_nms": getattr(cfg, "USE_NMS", True),
                "use_scene_caps": getattr(cfg, "USE_SCENE_CAPS", False),
            }
        )

        return job

    @staticmethod
    def _scene_classifier_payload(
            scene_data: Optional[Dict[str, Any]],
            scene_override: Optional[str] = None,
            error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize scene classifier output so all fields are present."""

        payload: Dict[str, Any] = dict(scene_data or {})

        if scene_override:
            payload.setdefault("scene", scene_override)

        payload.setdefault("scene", "unknown")
        payload.setdefault("is_staged", None)
        payload.setdefault("overall_impression", "")
        payload.setdefault("reasoning", "" if error is None else error)
        payload.setdefault("targets", [])
        payload.setdefault("gdino_terms", [])
        payload.setdefault("keywords", [])
        payload.setdefault("groundingdino_prompt", "")
        payload.setdefault("issues_natural_language", [])
        payload.setdefault("catalog_flags", {})
        payload.setdefault("issue_visual_anchors", [])
        payload.setdefault("processing_time", payload.get("processing_time"))
        payload.setdefault("prompt_version", payload.get("prompt_version", ""))
        payload.setdefault("scene_policy_version", payload.get("scene_policy_version", ""))
        payload.setdefault("image_summary", payload.get("image_summary", ""))
        payload.setdefault("image", payload.get("image"))

        if error and not payload.get("error"):
            payload["error"] = error
        payload.setdefault("error", payload.get("error"))

        return payload

    @staticmethod
    def _build_photo_entry(result: ImageAnalysisResult) -> Dict[str, Any]:
        """Create a serializable photo entry including scene classifier details."""
        scene_payload = AutoAnalyzer._scene_classifier_payload(result.scene_classifier or result.scene_data)

        return {
            "image_path": result.image_path,
            "scene": result.scene,
            "is_staged": scene_payload.get("is_staged"),
            "overall_impression": scene_payload.get("overall_impression", ""),
            "reasoning": scene_payload.get("reasoning", ""),
            "targets": scene_payload.get("targets", []),
            "gdino_terms": scene_payload.get("gdino_terms", []),
            "keywords": scene_payload.get("keywords", result.keywords_used),
            "groundingdino_prompt": scene_payload.get("groundingdino_prompt", ""),
            "issues_natural_language": scene_payload.get("issues_natural_language", []),
            "catalog_flags": scene_payload.get("catalog_flags", {}),
            "issue_visual_anchors": scene_payload.get("issue_visual_anchors", []),
            "scene_classifier": scene_payload,
            "keywords_used": result.keywords_used,
            "detection_count": result.detection_count,
            "verified_count": result.verified_count,
            "output_dir": result.output_dir,
            "processing_time": result.processing_time,
            "error": result.error,
        }

    @staticmethod
    def _pick_first_metadata(results: List[ImageAnalysisResult], key: str, default: str = "") -> str:
        for res in results:
            payload = AutoAnalyzer._scene_classifier_payload(res.scene_classifier or res.scene_data)
            val = payload.get(key)
            if isinstance(val, str) and val:
                return val
        return default

    def save_photo_intel(
            self,
            job: PropertyAnalysisJob,
            output_path: Optional[Path] = None,
            generate_summary: bool = True
    ) -> Path:
        """
        Persist per-photo intelligence (including scene classifier fields).

        Args:
            job: The completed PropertyAnalysisJob
            output_path: Where to save photo_intel.json
            generate_summary: If True, also generate property_summary.json

        Returns:
            Path to the saved photo_intel.json
        """
        created_at = datetime.utcnow().isoformat() + "Z"

        photos: Dict[str, Any] = {}
        for res in job.results:
            image_key = Path(res.image_path).name
            payload = self._scene_classifier_payload(res.scene_classifier or res.scene_data)
            photos[image_key] = {"image_path": res.image_path, **payload}

        photo_intel = {
            "run_id": job.job_id,
            "job_id": job.job_id,
            "property_key": job.property_key,
            "timestamp": job.timestamp,
            "created_at": created_at,
            "artifacts_dir": job.artifacts_dir,
            "model": getattr(cfg, "LM_STUDIO_MODEL", ""),
            "prompt_version": self._pick_first_metadata(job.results, "prompt_version", ""),
            "scene_policy_version": self._pick_first_metadata(job.results, "scene_policy_version", ""),
            "photos": photos,
        }

        try:
            photo_intel["renovation_needs"] = build_renovation_needs(job)
        except Exception as exc:
            logger.error(f"Failed to build renovation_needs: {exc}", exc_info=True)

        output_path = output_path or Path(job.artifacts_dir) / "photo_intel.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(photo_intel, f, indent=2, ensure_ascii=False)

        logger.info(f"Photo intel saved to: {output_path}")

        # Generate property summary if requested
        if generate_summary:
            summary_path = output_path.parent / "property_summary.json"
            self.generate_property_summary(job, photo_intel_path=output_path, output_path=summary_path)

        return output_path

    def generate_property_summary(
            self,
            job: PropertyAnalysisJob,
            photo_intel_path: Optional[Path] = None,
            output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Generate a property-level summary from the analysis results.

        This runs an additional VLM pass to aggregate all per-image issues
        into a coherent property assessment.

        Args:
            job: The completed PropertyAnalysisJob
            photo_intel_path: Path to photo_intel.json (if already saved)
            output_path: Where to save property_summary.json

        Returns:
            Path to the saved summary, or None if summarization failed/skipped
        """
        if PropertySummarizer is None:
            logger.warning("PropertySummarizer not available, skipping summary generation")
            return None

        # Check config flag
        if not getattr(cfg, "GENERATE_PROPERTY_SUMMARY", True):
            logger.info("Property summary generation disabled in config")
            return None

        logger.info(f"\n{'=' * 60}")
        logger.info("Generating Property Summary...")
        logger.info(f"{'=' * 60}")

        try:
            # Load photo_intel from file if provided, otherwise build from job
            if photo_intel_path and photo_intel_path.exists():
                with open(photo_intel_path, 'r', encoding='utf-8') as f:
                    photo_intel = json.load(f)
            else:
                # Build photo_intel structure from job results
                photos = {}
                for res in job.results:
                    image_key = Path(res.image_path).name
                    payload = self._scene_classifier_payload(res.scene_classifier or res.scene_data)
                    photos[image_key] = {"image_path": res.image_path, **payload}

                photo_intel = {
                    "property_key": job.property_key,
                    "job_id": job.job_id,
                    "timestamp": job.timestamp,
                    "artifacts_dir": job.artifacts_dir,
                    "photos": photos,
                }

                # Add renovation_needs if available
                try:
                    photo_intel["renovation_needs"] = build_renovation_needs(job)
                except Exception as exc:
                    logger.warning(f"Could not build renovation_needs: {exc}")

            # Create summarizer
            summarizer = PropertySummarizer(
                lm_studio_url=cfg.LM_STUDIO_URL,
                model_name=getattr(cfg, "SUMMARY_MODEL", cfg.LM_STUDIO_MODEL),
                debug=self.debug
            )

            # Generate summary
            summary = summarizer.summarize_property(photo_intel)

            # Save summary
            if output_path is None:
                output_path = Path(job.artifacts_dir) / "property_summary.json"

            PropertySummarizer.save_summary(summary, output_path)

            logger.info(f"Property summary saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate property summary: {e}", exc_info=True)
            return None

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