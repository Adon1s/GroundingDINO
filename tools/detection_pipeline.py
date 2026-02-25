# tools/detection_pipeline.py
"""
Detection backend runner + post-processing pipeline.

Encapsulates:
  - GroundingDINO and DINO-X backend invocation
  - ROI hint bonus application
  - Special-case filters, NMS, scene caps
  - pred.json rewrite, chip pruning, overlay redraw

Exposes one public entry point:
  run_detection_stage(...) -> (detection_result, detections)
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from tools.postprocess import (
    class_aware_nms,
    enforce_scene_caps,
    apply_roi_hint_bonus_overlap,
    apply_special_case_filters,
)
from tools.run_pipeline import redraw_overlay
from tools.pipeline_common import normalize_label_for_hint

logger = logging.getLogger(__name__)


def _run_command(cmd: List[str], *, debug: bool, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    import subprocess
    if debug:
        logger.debug("Running: %s", " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()
    return proc.returncode or 0, stdout, stderr


def run_detection_backend(
    *,
    cfg: Any,
    python_exe: Path,
    backend: str,
    image_path: Path,
    output_dir: Path,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    chip_margin: float,
    debug: bool,
    dinox_client: Any = None,
) -> Dict[str, Any]:
    if backend == "groundingdino":
        logger.info(f"Running GroundingDINO detection: {image_path.name}")
        cmd = [
            python_exe,
            cfg.GDINO_INFER_SCRIPT,
            "--config_file", cfg.GDINO_CONFIG,
            "--checkpoint_path", cfg.GDINO_CHECKPOINT,
            "--image_path", str(image_path),
            "--text_prompt", text_prompt,
            "--output_dir", str(output_dir),
            "--box_threshold", str(box_threshold),
            "--text_threshold", str(text_threshold),
            "--extract-chips",
            "--chip-margin", str(chip_margin),
        ]
        if cfg.CPU_ONLY:
            cmd.append("--cpu-only")
        if getattr(cfg, "COMPUTE_CHIP_QUALITY", False):
            cmd.append("--chip-quality")
        if getattr(cfg, "CREATE_THUMBNAILS", False):
            cmd.extend(["--create-thumbnail", "--thumbnail-size", str(cfg.THUMBNAIL_SIZE)])

        code, _, stderr = _run_command(cmd, debug=debug)
        if code != 0:
            logger.error(f"GroundingDINO detection failed: {stderr}")
            return {"error": stderr, "code": code}

        pred_json = output_dir / "pred.json"
        if pred_json.exists():
            return json.loads(pred_json.read_text(encoding="utf-8"))
        return {"error": "pred.json not produced"}

    if backend == "dinox":
        logger.info(f"Running DINO-X detection: {image_path.name}")
        dinox_script = getattr(cfg, "DINOX_INFER_SCRIPT", None)

        # Mode 1: local script
        if dinox_script and Path(dinox_script).exists():
            cmd = [
                python_exe,
                dinox_script,
                "--image_path", str(image_path),
                "--text_prompt", text_prompt,
                "--output_dir", str(output_dir),
                "--box_threshold", str(box_threshold),
                "--text_threshold", str(text_threshold),
            ]
            if getattr(cfg, "DINOX_EXTRACT_CHIPS", False):
                cmd.append("--extract-chips")
                cmd.extend(["--chip-margin", str(chip_margin)])
            if getattr(cfg, "DINOX_CPU_ONLY", cfg.CPU_ONLY):
                cmd.append("--cpu-only")

            code, _, stderr = _run_command(cmd, debug=debug)
            if code != 0:
                logger.error(f"DINO-X detection failed: {stderr}")
                return {"error": stderr, "code": code}

            pred_json = output_dir / "pred.json"
            if pred_json.exists():
                return json.loads(pred_json.read_text(encoding="utf-8"))
            return {"error": "pred.json not produced by DINO-X"}

        # Mode 2: API client
        if dinox_client:
            try:
                dinox_prompt = text_prompt.replace(". ", ".").replace(".", ".").strip(".")
                logger.info(f"  Using DINO-X API with prompt: {dinox_prompt}")

                result = dinox_client.detect(
                    image_path=image_path,
                    prompt=dinox_prompt,
                    bbox_threshold=box_threshold,
                    targets=["bbox"],
                )
                detections = [
                    {"label": o.category, "score": o.score, "box": o.bbox.to_list()}
                    for o in result.objects
                ]
                pred = {
                    "image": str(image_path),
                    "detections": detections,
                    "detection_count": len(detections),
                    "dinox_task_uuid": result.task_uuid,
                    "processing_time": result.processing_time,
                }
                try:
                    with Image.open(image_path) as im:
                        w, h = im.size
                        pred["size"] = {"width": w, "height": h}
                except Exception:
                    pass

                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "pred.json").write_text(
                    json.dumps(pred, indent=2), encoding="utf-8"
                )
                logger.info(f"  DINO-X detected {len(detections)} objects in {result.processing_time:.1f}s")
                return pred
            except Exception as e:
                logger.error(f"DINO-X API detection failed: {e}")
                return {"error": str(e)}

        # Mode 3: No option available
        import os
        api_token = (
            getattr(cfg, "DINOX_API_TOKEN", None)
            or getattr(cfg, "DINOX_API_KEY", None)
            or os.environ.get("DINOX_API_TOKEN")
            or os.environ.get("DINOX_API_KEY")
        )
        if api_token:
            return {"error": "DINOX_API_TOKEN is set but dinox_client.py is not available."}
        return {"error": "DINO-X backend selected but no detection method available."}

    return {"error": f"Unknown detection backend: {backend}"}


def run_detection_stage(
    *,
    cfg: Any,
    python_exe: Path,
    backend: str,
    image_path: Path,
    output_dir: Path,
    prompt: str,
    scene: str,
    planner_targets: List[Dict[str, Any]],
    planner_hints: Dict[str, str],
    box_threshold: float,
    text_threshold: float,
    chip_margin: float,
    debug: bool,
    dinox_client: Any = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Full detection + post-processing pipeline stage.

    Returns (detection_result, detections) where detection_result may contain
    an 'error' key on failure.
    """
    detection_result = run_detection_backend(
        cfg=cfg,
        python_exe=python_exe,
        backend=backend,
        image_path=image_path,
        output_dir=output_dir,
        text_prompt=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        chip_margin=chip_margin,
        debug=debug,
        dinox_client=dinox_client,
    )

    if "error" in detection_result:
        return detection_result, []

    detections = detection_result.get("detections", []) or []
    detection_result["planner_targets"] = planner_targets
    detection_result["planner_hints"] = planner_hints

    # image size
    size_info = detection_result.get("size") or {}
    img_w = int(size_info.get("width") or 0)
    img_h = int(size_info.get("height") or 0)
    if (img_w <= 0 or img_h <= 0) and image_path.exists():
        try:
            with Image.open(image_path) as pil_im:
                img_w, img_h = pil_im.size
        except Exception:
            pass

    # ROI hint bonus
    if (
        getattr(cfg, "ROI_HINTS_ENABLED", False)
        and planner_hints
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
            norm_label = normalize_label_for_hint(label_name)
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
            logger.info(f"  Special cases: {pre_case} -> {len(detections)} detections")

    # NMS (AFTER detection, BEFORE verification)
    if getattr(cfg, "USE_NMS", True) and detections:
        pre_nms_count = len(detections)
        detections = class_aware_nms(
            detections,
            per_class_iou=getattr(cfg, "NMS_PER_CLASS", None),
            default_iou=getattr(cfg, "NMS_DEFAULT_IOU", 0.3),
        )
        logger.info(f"  NMS: {pre_nms_count} -> {len(detections)} detections")

    # Optional: enforce per-scene caps
    if getattr(cfg, "USE_SCENE_CAPS", False) and detections:
        pre_cap_count = len(detections)
        detections = enforce_scene_caps(
            detections,
            scene=scene,
            caps_map=getattr(cfg, "SCENE_CAPS", None),
        )
        if pre_cap_count != len(detections):
            logger.info(f"  Scene caps: {pre_cap_count} -> {len(detections)} detections")

    # Debug ROI logging
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

    # Write survivors back
    detection_result["detections"] = detections

    # Update all count fields unconditionally to match filtered detections
    final_count = len(detections)
    for k in ("count", "num_detections", "detections_count", "detection_count"):
        detection_result[k] = final_count

    pred_json_path = output_dir / "pred.json"
    pred_json_path.write_text(
        json.dumps(detection_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # (Optional) prune chips for detections that were dropped
    if getattr(cfg, "PRUNE_DROPPED_CHIPS", False):
        try:
            chip_dir = Path(
                (detection_result.get("chip_extraction", {}) or {}).get(
                    "chips_directory", str(output_dir / "chips")
                )
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

    # Redraw overlay
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

    return detection_result, detections
