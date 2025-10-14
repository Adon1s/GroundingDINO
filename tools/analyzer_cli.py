#!/usr/bin/env python3
"""
Analyzer CLI
------------

Runs your *existing* GroundingDINO demo script (detect + chips) and then your
ChipVerifier over the produced folder. Aggregates results and prints a single JSON.

You can provide paths via environment variables or flags:

Env (preferred):
  GDINO_DETECT_SCRIPT   = path to GroundingDINO demo detect script
  GDINO_CONFIG          = path to GroundingDINO config .py
  GDINO_CHECKPOINT      = path to GroundingDINO weights .pth
  CHIP_VERIFIER_PY      = path to your ChipVerifier.py
  ARTIFACTS_ROOT        = output root directory (default: ./artifacts)
  LM_STUDIO_URL         = http://host:port for LM Studio (optional)
  LM_STUDIO_MODEL       = model id (optional, default gemma-3-27b-it)

Usage example (Windows PowerShell):
  C:\...\GroundingDINO\.venv\Scripts\python.exe analyzer_cli.py `
      --property-key redfin_123 `
      --images "C:\img\a.jpg" "C:\img\b.jpg" `
      --box-thr 0.30 --text-thr 0.25 --chip-margin 0.15 --chip-quality --thumbnail
"""

import os
import sys
import json
import uuid
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any


# ---------- helpers ----------

def eprint(*args):
    print(*args, file=sys.stderr)


def run_python(python_exe: Path, argv: List[str], cwd: Path | None = None, stdin: str | None = None) -> tuple[
    int, str, str]:
    """Run a python command, capture stdout/stderr, return (code, out, err)."""
    proc = subprocess.Popen(
        [str(python_exe)] + [str(a) for a in argv],
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.PIPE if stdin is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if stdin is not None and proc.stdin:
        proc.stdin.write(stdin)
        proc.stdin.close()
    out, err = proc.communicate()
    return proc.returncode or 0, out, err


def ensure_abs(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def must_exist(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Run GroundingDINO detect+chips and ChipVerifier, aggregate JSON.")
    parser.add_argument("--property-key", required=True, help="Canonical property key (e.g., redfin_12345678)")
    parser.add_argument("--images", nargs="+", required=True, help="One or more absolute image paths")
    parser.add_argument("--text-prompt", default="sofa. chair. television. painting. window. microwave. plant.",
                        help="Detection text prompt")
    parser.add_argument("--box-thr", type=float, default=0.30, help="Box threshold (default 0.30)")
    parser.add_argument("--text-thr", type=float, default=0.25, help="Text threshold (default 0.25)")
    parser.add_argument("--chip-margin", type=float, default=0.15, help="Chip margin percent (0.15 = 15%)")
    parser.add_argument("--chip-quality", action="store_true", help="Compute chip quality metrics")
    parser.add_argument("--thumbnail", action="store_true", help="Create thumbnail with masks")

    # Optional explicit paths (otherwise read from env)
    parser.add_argument("--detect-script", help="Path to GDINO demo detect script (override GDINO_DETECT_SCRIPT)")
    parser.add_argument("--config", help="Path to GDINO config .py (override GDINO_CONFIG)")
    parser.add_argument("--checkpoint", help="Path to GDINO weights .pth (override GDINO_CHECKPOINT)")
    parser.add_argument("--verifier", help="Path to ChipVerifier.py (override CHIP_VERIFIER_PY)")
    parser.add_argument("--artifacts-root", help="Output root (override ARTIFACTS_ROOT)")
    parser.add_argument("--lm-url", help="LM Studio URL (override LM_STUDIO_URL)")
    parser.add_argument("--lm-model", help="LM Studio model id (override LM_STUDIO_MODEL)")

    # If you run this from the GDINO venv, sys.executable is already that python.
    parser.add_argument("--python-exe", help="Python exe to use (default: sys.executable)")

    args = parser.parse_args()

    # Resolve config from env or args
    detect_script = ensure_abs(
        args.detect_script
        or os.getenv("GDINO_DETECT_SCRIPT")
        or os.getenv("GDINO_INFER_SCRIPT", "")
    )
    config_path = ensure_abs(args.config or os.getenv("GDINO_CONFIG", ""))
    checkpoint = ensure_abs(args.checkpoint or os.getenv("GDINO_CHECKPOINT", ""))
    verifier_py = ensure_abs(args.verifier or os.getenv("CHIP_VERIFIER_PY", ""))
    artifacts_root = ensure_abs(args.artifacts_root or os.getenv("ARTIFACTS_ROOT", "./artifacts"))

    lm_url = args.lm_url or os.getenv("LM_STUDIO_URL", "http://localhost:1234")
    lm_model = args.lm_model or os.getenv("LM_STUDIO_MODEL", "gemma-3-27b-it")

    # Python exe: default to current interpreter (recommended: run via GDINO venv python.exe)
    python_exe = ensure_abs(args.python_exe) if args.python_exe else Path(sys.executable).resolve()

    # Validate paths
    must_exist(python_exe, "Python")
    must_exist(detect_script, "GDINO detect script")
    must_exist(config_path, "GDINO config")
    must_exist(checkpoint, "GDINO checkpoint")
    must_exist(verifier_py, "ChipVerifier")

    # Validate images
    images = [ensure_abs(p) for p in args.images]
    for p in images:
        must_exist(p, "Image")

    # Prepare job dir
    job_id = timestamp_id()
    job_dir = artifacts_root / args.property_key / job_id
    (job_dir).mkdir(parents=True, exist_ok=True)

    # Orchestration
    per_image: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for idx, img in enumerate(images):
        out_dir = job_dir / f"img_{idx:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Run GroundingDINO demo detect script (extract chips, thresholds, thumbnail)
        gd_args = [
            str(detect_script),
            "--config_file", str(config_path),
            "--checkpoint_path", str(checkpoint),
            "--image_path", str(img),
            "--text_prompt", args.text_prompt,
            "--output_dir", str(out_dir),
            "--box_threshold", str(args.box_thr),
            "--text_threshold", str(args.text_thr),
            "--extract-chips",
            "--chip-margin", str(args.chip_margin),
        ]
        if args.chip_quality:
            gd_args.append("--chip-quality")
        if args.thumbnail:
            gd_args.extend(["--create-thumbnail", "--thumbnail-size", "384"])

        code, out, err = run_python(python_exe, gd_args)
        if code != 0:
            failures.append({
                "step": "detect",
                "image": str(img),
                "code": code,
                "stderr": err,
                "stdout": out,
            })
            continue

        pred_json = out_dir / "pred.json"
        if not pred_json.exists():
            failures.append({
                "step": "detect",
                "image": str(img),
                "code": code,
                "stderr": err,
                "stdout": out,
                "error": "pred.json not produced"
            })
            continue

        # 2) Run ChipVerifier on that output directory
        ver_args = [
            str(verifier_py),
            str(out_dir),
            "--lm-studio-url", lm_url,
            "--model", lm_model,
            "--max-chips", "3",
        ]
        v_code, v_out, v_err = run_python(python_exe, ver_args)
        if v_code != 0:
            failures.append({
                "step": "verify",
                "image": str(img),
                "code": v_code,
                "stderr": v_err,
                "stdout": v_out,
            })

        # Read produced JSONs
        try:
            with open(pred_json, "r", encoding="utf-8") as f:
                pred = json.load(f)
        except Exception as ex:
            pred = {"error": f"Failed to read pred.json: {ex}"}

        ver_json_path = out_dir / "verification_results.json"
        ver = None
        if ver_json_path.exists():
            try:
                with open(ver_json_path, "r", encoding="utf-8") as f:
                    ver = json.load(f)
            except Exception as ex:
                ver = {"error": f"Failed to read verification_results.json: {ex}"}

        per_image.append({
            "imagePath": str(img),
            "outputDir": str(out_dir),
            "detection": pred,
            "verification": ver,
        })

    # Build final payload
    result: Dict[str, Any] = {
        "jobId": job_id,
        "propertyKey": args.property_key,
        "prompt": args.text_prompt,
        "boxThreshold": args.box_thr,
        "textThreshold": args.text_thr,
        "chipMargin": args.chip_margin,
        "imagesProcessed": len(per_image),
        "results": per_image,
        "failures": failures,
        "artifactsRoot": str(job_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Print one JSON object to stdout
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        eprint(f"[FATAL] {e}")
        sys.exit(1)
