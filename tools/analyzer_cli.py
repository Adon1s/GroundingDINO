#!/usr/bin/env python3
"""
CLI entrypoint for running the GroundingDINO pipeline as an external tool.

Designed to be called from RealtorVision via child_process.spawn using the
GDINO_PY virtualenv interpreter.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# The full pipeline lives in the GroundingDINO repo. This CLI expects to be
# placed alongside the existing auto_analyzer.py / pipeline_config.py modules
# and reuses the AutoAnalyzer class exported there.
try:
    from auto_analyzer import AutoAnalyzer  # type: ignore
    import pipeline_config as cfg  # type: ignore
except Exception as exc:  # pragma: no cover - external dependency
    print(f"Failed to import pipeline modules: {exc}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)


PATH_OVERRIDE_KEYS = {
    "GDINO_CONFIG",
    "GDINO_CHECKPOINT",
    "GDINO_INFER_SCRIPT",
    "SCENE_CLASSIFIER_PY",
    "CHIP_VERIFIER_PY",
    "ANALYZER_CLI",
}

STRING_OVERRIDE_KEYS = {
    "LM_STUDIO_URL",
    "LM_STUDIO_MODEL",
}


def _apply_env_overrides() -> None:
    """Allow env vars to override key config values when called from Node."""
    # Filesystem path overrides – keep them as Path objects, not strings.
    for key in PATH_OVERRIDE_KEYS:
        val = os.environ.get(key)
        if val:
            resolved = Path(val).resolve()
            setattr(cfg, key, resolved)
            logger.debug("Override %s = %s", key, resolved)

    # Plain string overrides (no Path.resolve)
    for key in STRING_OVERRIDE_KEYS:
        val = os.environ.get(key)
        if val:
            setattr(cfg, key, val)
            logger.debug("Override %s = %s", key, val)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GroundingDINO auto analyzer")
    parser.add_argument("--property-key", required=True, dest="property_key")
    parser.add_argument("--images", required=True, nargs="+", help="Absolute image paths")
    parser.add_argument("--artifacts-root", required=True, dest="artifacts_root")
    parser.add_argument("--html-report", action="store_true", help="Write HTML report")
    parser.add_argument("--output-json", dest="output_json", help="Path to write JSON summary")
    parser.add_argument("--python-exe", dest="python_exe", default=sys.executable)
    parser.add_argument("--box-threshold", type=float)
    parser.add_argument("--text-threshold", type=float)
    parser.add_argument("--chip-margin", type=float)
    parser.add_argument("--max-keywords", type=int)
    parser.add_argument(
        "--include-common",
        action="store_const",
        const=True,
        default=None,
        help="Override config to include common objects",
    )
    parser.add_argument(
        "--include-conditions",
        action="store_const",
        const=True,
        default=None,
        help="Override config to include defect/condition keywords",
    )
    parser.add_argument("--no-verify", dest="skip_verification", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _build_summary(job: Any) -> Dict[str, Any]:
    total_detections = sum((res.detection_count or 0) for res in job.results)
    verified = [res for res in job.results if res.verified_count is not None]
    total_verified = sum(res.verified_count or 0 for res in verified)

    return {
        "success": True,
        "jobId": job.job_id,
        "property_key": job.property_key,
        "artifacts_dir": job.artifacts_dir,
        "total_detections": total_detections,
        "verified_detections": total_verified,
        "total_processing_time": job.total_processing_time,
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    _apply_env_overrides()

    images: List[Path] = [Path(img).resolve() for img in args.images]
    artifacts_root = Path(args.artifacts_root).resolve()

    analyzer = AutoAnalyzer(
        python_exe=args.python_exe,
        artifacts_root=artifacts_root,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        chip_margin=args.chip_margin,
        max_keywords=args.max_keywords,
        include_common=args.include_common,
        include_conditions=args.include_conditions,
        skip_verification=args.skip_verification,
        debug=args.debug,
    )

    job = analyzer.analyze_property(args.property_key, images)

    if args.html_report:
        report_path = artifacts_root / args.property_key / job.job_id / "report.html"
        analyzer.create_html_report(job, report_path)

    summary = _build_summary(job)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Last line on stdout must be JSON for the Node caller
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
