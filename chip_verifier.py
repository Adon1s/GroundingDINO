#!/usr/bin/env python3
"""
GroundingDINO Detection Chip Verifier (Generic)
----------------------------------------------
Purpose: Verify that each detection chip primarily and tightly contains the
object label that GroundingDINO predicted — nothing class‑specific.

Inputs (from a GroundingDINO run's output directory):
  - pred.json  (must include a top-level key "detections")
  - chips/chip_metadata.json (for mapping chips -> detections)
  - thumbnail_with_masks.jpg (optional overview for context)

Output:
  - verification_results.json  with a summary and per-detection verdicts

Notes:
  - Uses an OpenAI-compatible LM Studio endpoint that accepts image inputs via
    "image_url" (base64 data URLs). Adjust LM_STUDIO_URL and DEFAULT_MODEL.
  - Prompts are generic: "Does this chip contain the target label <X>? Is the
    crop tight and mostly the target?"
"""

import json
import base64
import requests
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from PIL import Image
import logging
import re

# ── Console encoding (Windows safety) ─────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ───────────────────────────────────────────────────────────────────
LM_STUDIO_URL = "http://192.168.86.143:1234"  # Change to your LM Studio URL
DEFAULT_MODEL = "gemma-3-27b-it"              # Use a vision-capable model

# Global consensus thresholds (class-agnostic)
VERIFY_THRESHOLDS = {
    "consensus_ratio": 0.60,   # fraction of chips that must be valid
    "avg_confidence": 0.60     # average model confidence across chips
}

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class ChipVerification:
    chip_path: str
    detection_idx: int
    label: str
    original_confidence: float
    is_valid: bool
    verification_confidence: float
    reasoning: str
    bbox_quality: Optional[str] = None  # "tight"|"acceptable"|"too_broad"|"wrong_object"
    object_fill_ratio: Optional[float] = None
    mislabel_if_any: Optional[str] = None
    processing_time: float = 0.0
    error: Optional[str] = None


@dataclass
class DetectionVerification:
    detection_idx: int
    label: str
    original_confidence: float
    bbox: List[int]
    chip_verifications: List[ChipVerification]
    final_verdict: str  # "VALID" | "INVALID" | "UNCERTAIN"
    consensus_confidence: float
    reasoning: str


# ── Verifier class ───────────────────────────────────────────────────────────
class ChipVerifier:
    def __init__(self, lm_studio_url: str = LM_STUDIO_URL, model_name: str = DEFAULT_MODEL,
                 batch_size: int = 3, debug: bool = False):
        self.lm_studio_url = lm_studio_url.rstrip('/')
        self.model_name = model_name
        self.batch_size = batch_size
        self.debug = debug

    # ── IO helpers ───────────────────────────────────────────────────────────
    def load_grounding_dino_output(self, output_dir: Path) -> Dict:
        output_dir = Path(output_dir)

        pred_json = output_dir / "pred.json"
        if not pred_json.exists():
            raise FileNotFoundError(f"No pred.json found in {output_dir}")
        with open(pred_json, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        chip_metadata_file = output_dir / "chips" / "chip_metadata.json"
        chip_metadata = []
        if chip_metadata_file.exists():
            with open(chip_metadata_file, 'r', encoding='utf-8') as f:
                chip_metadata = json.load(f)

        thumbnail_path = output_dir / "thumbnail_with_masks.jpg"
        thumbnail = thumbnail_path if thumbnail_path.exists() else None

        return {
            "predictions": pred_data,
            "chip_metadata": chip_metadata,
            "thumbnail": thumbnail,
            "chips_dir": output_dir / "chips"
        }

    @staticmethod
    def encode_image_to_b64(image_path: Path) -> str:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    # ── Prompting ────────────────────────────────────────────────────────────
    @staticmethod
    def build_generic_prompt(target_label: str, gd_conf: Optional[float]) -> str:
        label = target_label.strip()
        conf_txt = f" (detector score ~{gd_conf:.2f})" if isinstance(gd_conf, (int, float)) else ""
        return (
            "You are an expert visual verifier.\n\n"
            f"TASK: Given a candidate crop ('chip') and an overview thumbnail, determine whether the chip primarily and tightly contains an instance of the target label: '{label}'.{conf_txt}\n\n"
            "CRITERIA:\n"
            "1) The target is clearly present in the chip.\n"
            "2) The chip is tight: minimal unrelated background.\n"
            "3) No other object dominates the chip.\n\n"
            "RESPOND ONLY WITH JSON in this schema (no extra text):\n"
            "{\n"
            "  \"is_valid\": true/false,\n"
            "  \"confidence\": 0.0-1.0,\n"
            "  \"reasoning\": \"short explanation\",\n"
            "  \"bbox_quality\": \"tight|acceptable|too_broad|wrong_object\",\n"
            "  \"object_fill_ratio\": 0.0-1.0,\n"
            "  \"contains_target\": true/false,\n"
            "  \"mislabel_if_any\": \"what it likely is if not the target\"\n"
            "}"
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Try to robustly extract the first JSON object from a text response."""
        # Strip code fences if present
        if "```" in text:
            # Prefer fenced JSON if available
            m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
            if m:
                text = m.group(1)
            else:
                text = re.sub(r"^```[\w-]*|```$", "", text.strip())
        # Find first balanced { ... }
        brace_stack = []
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if not brace_stack:
                    start = i
                brace_stack.append('{')
            elif ch == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start is not None:
                        candidate = text[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            pass
        return None

    # ── Core verification ────────────────────────────────────────────────────
    def verify_single_chip(self, chip_path: Path, label: str, gd_conf: Optional[float],
                           thumbnail_path: Optional[Path]) -> ChipVerification:
        t0 = time.time()
        try:
            prompt = self.build_generic_prompt(label, gd_conf)
            content = [{"type": "text", "text": prompt}]

            # Main chip (required)
            chip_b64 = self.encode_image_to_b64(chip_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{chip_b64}"}
            })

            # Thumbnail context (optional)
            if thumbnail_path and thumbnail_path.exists():
                thumb_b64 = self.encode_image_to_b64(thumbnail_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{thumb_b64}"}
                })

            messages = [
                {"role": "system", "content": "You are a careful, literal verifier. Respond with valid JSON only."},
                {"role": "user", "content": content}
            ]

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 400,
                "stream": False
            }

            resp = requests.post(f"{self.lm_studio_url}/v1/chat/completions", json=payload, timeout=45)
            if resp.status_code != 200:
                raise RuntimeError(f"API error: HTTP {resp.status_code} - {resp.text[:200]}")

            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            parsed = self._extract_json(raw) or {}

            is_valid = bool(parsed.get("is_valid", False))
            vconf = float(parsed.get("confidence", 0.5))

            return ChipVerification(
                chip_path=str(chip_path),
                detection_idx=-1,
                label=label,
                original_confidence=float(gd_conf) if isinstance(gd_conf, (int, float)) else 0.0,
                is_valid=is_valid,
                verification_confidence=vconf,
                reasoning=str(parsed.get("reasoning", ""))[:500],
                bbox_quality=parsed.get("bbox_quality"),
                object_fill_ratio=float(parsed.get("object_fill_ratio", 0.0)) if parsed.get("object_fill_ratio") is not None else None,
                mislabel_if_any=parsed.get("mislabel_if_any"),
                processing_time=time.time() - t0
            )

        except Exception as e:
            logger.error(f"Error verifying chip {chip_path}: {e}")
            return ChipVerification(
                chip_path=str(chip_path),
                detection_idx=-1,
                label=label,
                original_confidence=float(gd_conf) if isinstance(gd_conf, (int, float)) else 0.0,
                is_valid=False,
                verification_confidence=0.0,
                reasoning="Verification failed",
                error=str(e),
                processing_time=time.time() - t0
            )

    def verify_batch(self, chips: List[Tuple[Path, str, Optional[float]]], thumbnail_path: Optional[Path]) -> List[ChipVerification]:
        results = []
        for chip_path, label, gd_conf in chips:
            if self.debug:
                logger.info(f"Verifying {chip_path.name}  label='{label}'  det_conf={gd_conf}")
            result = self.verify_single_chip(chip_path, label, gd_conf, thumbnail_path)
            results.append(result)
            time.sleep(0.4)  # avoid hammering the local API
        return results

    @staticmethod
    def aggregate_detection_results(detection_idx: int, label: str, original_conf: Optional[float],
                                    bbox_xyxy: List[int], verifications: List[ChipVerification]) -> DetectionVerification:
        if not verifications:
            return DetectionVerification(
                detection_idx=detection_idx,
                label=label,
                original_confidence=float(original_conf) if original_conf is not None else 0.0,
                bbox=bbox_xyxy,
                chip_verifications=[],
                final_verdict="INVALID",
                consensus_confidence=0.0,
                reasoning="No verification chips available"
            )

        valid_count = sum(1 for v in verifications if v.is_valid)
        valid_ratio = valid_count / len(verifications)
        avg_conf = float(np.mean([v.verification_confidence for v in verifications])) if verifications else 0.0

        if valid_ratio >= VERIFY_THRESHOLDS["consensus_ratio"] and avg_conf >= VERIFY_THRESHOLDS["avg_confidence"]:
            verdict = "VALID"
            reason = f"Consensus {valid_ratio:.0%} with avg confidence {avg_conf:.2f}"
        elif valid_ratio <= 0.25 or avg_conf < 0.40:
            verdict = "INVALID"
            reason = f"Too weak consensus ({valid_ratio:.0%}) and/or low avg confidence ({avg_conf:.2f})"
        else:
            verdict = "UNCERTAIN"
            reason = f"Mixed verification: {valid_count}/{len(verifications)} valid, avg confidence {avg_conf:.2f}"

        return DetectionVerification(
            detection_idx=detection_idx,
            label=label,
            original_confidence=float(original_conf) if original_conf is not None else 0.0,
            bbox=bbox_xyxy,
            chip_verifications=verifications,
            final_verdict=verdict,
            consensus_confidence=avg_conf,
            reasoning=reason
        )

    # ── Pipeline ─────────────────────────────────────────────────────────────
    def verify_detections(self, output_dir: str, max_chips_per_detection: int = 3) -> Dict:
        out_path = Path(output_dir)
        logger.info(f"Starting verification for {out_path}")

        gd = self.load_grounding_dino_output(out_path)
        preds = gd.get("predictions", {})
        detections = preds.get("detections", [])
        chip_metadata = gd.get("chip_metadata", [])
        thumbnail = gd.get("thumbnail")

        if not detections:
            logger.warning("No detections found in pred.json")
            return {"results": [], "summary": {}}

        # Map detection_idx -> chips
        chips_by_detection: Dict[int, List[dict]] = defaultdict(list)
        for info in chip_metadata:
            det_idx = int(info.get("detection_idx", -1))
            if det_idx < 0:
                continue
            chip_path = gd["chips_dir"] / info.get("filename", "")
            if chip_path.exists():
                entry = dict(info)
                entry["__path"] = chip_path
                chips_by_detection[det_idx].append(entry)

        all_results: List[DetectionVerification] = []
        counts = {"VALID": 0, "INVALID": 0, "UNCERTAIN": 0}

        for det_idx, det in enumerate(detections):
            label = str(det.get("label", ""))
            gd_conf = det.get("score", None)
            bbox_xyxy = det.get("bbox_xyxy", [])

            logger.info(f"\nVerifying detection {det_idx}: label='{label}'  det_conf={gd_conf}")

            chips = chips_by_detection.get(det_idx, [])
            if not chips:
                logger.warning(f"No chips found for detection {det_idx}")
                # still record an INVALID with no chips
                dv = self.aggregate_detection_results(det_idx, label, gd_conf, bbox_xyxy, [])
                all_results.append(dv)
                counts[dv.final_verdict] += 1
                continue

            # Limit number of chips per detection
            chips = chips[:max_chips_per_detection]

            to_verify: List[Tuple[Path, str, Optional[float]]] = []
            for c in chips:
                to_verify.append((c["__path"], label, gd_conf))

            verifs = self.verify_batch(to_verify, thumbnail)
            for v in verifs:
                v.detection_idx = det_idx

            dv = self.aggregate_detection_results(det_idx, label, gd_conf, bbox_xyxy, verifs)
            all_results.append(dv)
            counts[dv.final_verdict] += 1

            logger.info(f"  Verdict: {dv.final_verdict}  (avg conf: {dv.consensus_confidence:.2f})")
            logger.info(f"  Reason:  {dv.reasoning}")

        # Summary
        verified = len(all_results)
        valid = counts["VALID"]
        summary = {
            "total_detections": len(detections),
            "verified_detections": verified,
            "valid": valid,
            "invalid": counts["INVALID"],
            "uncertain": counts["UNCERTAIN"],
            "verification_rate": (valid / verified) if verified else 0.0,
            "thresholds": VERIFY_THRESHOLDS,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save results
        results_path = out_path / "verification_results.json"
        serializable = {
            "summary": summary,
            "results": [
                {
                    "detection_idx": r.detection_idx,
                    "label": r.label,
                    "original_confidence": r.original_confidence,
                    "bbox": r.bbox,
                    "final_verdict": r.final_verdict,
                    "consensus_confidence": r.consensus_confidence,
                    "reasoning": r.reasoning,
                    "chip_verifications": [
                        {
                            "chip_path": cv.chip_path,
                            "is_valid": cv.is_valid,
                            "confidence": cv.verification_confidence,
                            "reasoning": cv.reasoning,
                            "bbox_quality": cv.bbox_quality,
                            "object_fill_ratio": cv.object_fill_ratio,
                            "mislabel_if_any": cv.mislabel_if_any,
                        } for cv in r.chip_verifications
                    ]
                } for r in all_results
            ],
            "config": {
                "model": self.model_name,
                "max_chips_per_detection": max_chips_per_detection
            }
        }
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info("\n" + "=" * 60)
        logger.info("Verification Complete!")
        logger.info(f"Valid: {counts['VALID']}, Invalid: {counts['INVALID']}, Uncertain: {counts['UNCERTAIN']}")
        logger.info(f"Results saved to: {results_path}")

        return {"summary": summary, "results": all_results}


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify that chips contain their GroundingDINO labels (generic)")
    parser.add_argument("output_dir", help="Path to GroundingDINO output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name")
    parser.add_argument("--lm-studio-url", default=LM_STUDIO_URL, help="LM Studio API URL")
    parser.add_argument("--max-chips", type=int, default=3, help="Max chips per detection")
    parser.add_argument("--batch-size", type=int, default=3, help="(reserved) Batch size for API calls")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    verifier = ChipVerifier(
        lm_studio_url=args.lm_studio_url,
        model_name=args.model,
        batch_size=args.batch_size,
        debug=args.debug
    )

    results = verifier.verify_detections(output_dir=args.output_dir, max_chips_per_detection=args.max_chips)

    summary = results.get("summary", {})
    print("\n✅ Verification Summary:")
    print(f"  Total Detections: {summary.get('total_detections', 0)}")
    print(f"  Verified:         {summary.get('verified_detections', 0)}")
    print(f"  Valid:            {summary.get('valid', 0)}")
    print(f"  Invalid:          {summary.get('invalid', 0)}")
    print(f"  Uncertain:        {summary.get('uncertain', 0)}")
    print(f"  Verification Rate: {summary.get('verification_rate', 0.0):.1%}")


if __name__ == "__main__":
    main()
