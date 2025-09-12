#!/usr/bin/env python3
"""
GroundingDINO Detection Chip Verifier
Verifies object detection chips using LM Studio vision models
Compatible with the verifier cascade strategy
"""

import json
import base64
import requests
import time
import warnings
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

# Force UTF-8 output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

# Configuration
LM_STUDIO_URL = "http://192.168.86.143:1234"  # Adjust to your LM Studio URL
DEFAULT_MODEL = "gemma-3-27b-it"  # Adjust to your vision model

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Class-specific verification thresholds
CLASS_THRESHOLDS = {
    "water_stain": {"confidence": 0.65, "consensus": 0.6},
    "water stain": {"confidence": 0.65, "consensus": 0.6},
    "chipped_paint": {"confidence": 0.6, "consensus": 0.5},
    "chipped paint": {"confidence": 0.6, "consensus": 0.5},
    "crack": {"confidence": 0.7, "consensus": 0.6},
    "hole": {"confidence": 0.75, "consensus": 0.7},
    "mold": {"confidence": 0.7, "consensus": 0.6},
    "discoloration": {"confidence": 0.6, "consensus": 0.5},
    "front_door": {"confidence": 0.7, "consensus": 0.6},
    "front door": {"confidence": 0.7, "consensus": 0.6},
    "garage_door": {"confidence": 0.7, "consensus": 0.6},
    "garage door": {"confidence": 0.7, "consensus": 0.6},
    "window": {"confidence": 0.65, "consensus": 0.5},
    "roof_line": {"confidence": 0.6, "consensus": 0.5},
    "roof line": {"confidence": 0.6, "consensus": 0.5},
    "default": {"confidence": 0.65, "consensus": 0.5}
}

# Verification prompt templates per class type
VERIFICATION_PROMPTS = {
    "defect": """You are a building inspection expert verifying detected defects.

TASK: Verify if the bounding box TIGHTLY contains ONLY a {class_name} and nothing else significant.

CONTEXT:
- Detection confidence: {confidence:.2f}
- Location in image: {location_description}
- Chip taken with {margin_percent}% context margin

VISUAL CHARACTERISTICS OF {class_name}:
{class_characteristics}

CRITICAL VERIFICATION:
1. Does the bounding box contain PRIMARILY just the {class_name}?
2. Is the detection too broad (includes walls, other objects, large areas)?
3. Does the {class_name} fill at least 50% of the detection area?
4. Are there other significant objects wrongly included?

RESPOND WITH JSON:
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "bbox_quality": "tight/acceptable/too_broad/wrong_object",
  "object_fill_ratio": "estimated percentage the target object fills the bbox",
  "alternative_explanation": "What else it could be if not {class_name}"
}}""",

    "structural": """You are a construction expert verifying architectural features.

TASK: Verify if the bounding box SPECIFICALLY outlines ONLY a {class_name}.

DETECTION INFO:
- Confidence: {confidence:.2f}
- Location: {location_description}
- Context margin: {margin_percent}%

EXPECTED CHARACTERISTICS OF {class_name}:
{class_characteristics}

CRITICAL CHECKS:
1. Is the bounding box tightly focused on JUST the {class_name}?
2. Does it include too much surrounding area (walls, ground, sky)?
3. Would a human draw a similar box around just this {class_name}?
4. Does the {class_name} occupy most (>60%) of the detection area?

RESPOND WITH JSON:
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "bbox_quality": "tight/acceptable/too_broad/wrong_object",
  "contains_target": true/false,
  "detection_precision": "precise/acceptable/poor"
}}"""
}

# Class-specific visual characteristics
CLASS_CHARACTERISTICS = {
    "water_stain": """
- Irregular, organic shape with soft edges
- Discoloration (yellow, brown, or gray)
- Often has concentric rings or tide marks
- May show through paint/drywall
- Typically on ceilings or upper walls""",

    "chipped_paint": """
- Sharp, irregular edges
- Exposed substrate visible (wood, drywall, concrete)
- Color contrast between paint and substrate
- May show peeling or flaking edges
- Often near high-traffic or moisture areas""",

    "crack": """
- Linear or branching pattern
- Consistent width or tapering
- May be straight or jagged
- Shows depth/shadow
- Different from surface scratches""",

    "mold": """
- Dark spots or patches (black, green, gray)
- Fuzzy or spotty texture
- Often in clusters
- Common in damp areas
- May have irregular spreading pattern""",

    "front_door": """
- Rectangular shape with proper proportions
- Door hardware visible (handle, lock, hinges)
- Frame and threshold visible
- Usually has panels or solid surface
- Entry features nearby (porch, steps, lighting)""",

    "garage_door": """
- Wide rectangular shape (wider than tall)
- Horizontal panels or sections
- May show windows
- Driveway or concrete pad below
- Larger than regular doors""",

    "window": """
- Glass panes visible
- Frame structure (sash, mullions)
- Transparent or reflective surface
- Regular geometric shape
- May show interior/exterior contrast"""
}


@dataclass
class ChipVerification:
    """Single chip verification result"""
    chip_path: str
    detection_idx: int
    class_name: str
    original_confidence: float
    is_valid: bool
    verification_confidence: float
    reasoning: str
    visual_evidence: List[str]
    alternative_explanation: Optional[str] = None
    processing_time: float = 0.0
    error: Optional[str] = None


@dataclass
class DetectionVerification:
    """Aggregated verification for a detection"""
    detection_idx: int
    class_name: str
    original_confidence: float
    bbox: List[int]
    chip_verifications: List[ChipVerification]
    final_verdict: str  # "VALID", "INVALID", "UNCERTAIN"
    consensus_confidence: float
    reasoning: str


class ChipVerifier:
    def __init__(self,
                 lm_studio_url: str = LM_STUDIO_URL,
                 model_name: str = DEFAULT_MODEL,
                 batch_size: int = 3,
                 debug: bool = False):
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.batch_size = batch_size
        self.debug = debug

    def load_grounding_dino_output(self, output_dir: Path) -> Dict:
        """Load GroundingDINO output files"""
        output_dir = Path(output_dir)

        # Load main prediction file
        pred_json = output_dir / "pred.json"
        if not pred_json.exists():
            raise FileNotFoundError(f"No pred.json found in {output_dir}")

        with open(pred_json, 'r') as f:
            pred_data = json.load(f)

        # Load chip metadata if it exists
        chip_metadata_file = output_dir / "chips" / "chip_metadata.json"
        chip_metadata = []
        if chip_metadata_file.exists():
            with open(chip_metadata_file, 'r') as f:
                chip_metadata = json.load(f)

        # Load thumbnail if it exists
        thumbnail_path = output_dir / "thumbnail_with_masks.jpg"
        thumbnail = None
        if thumbnail_path.exists():
            thumbnail = thumbnail_path

        return {
            "predictions": pred_data,
            "chip_metadata": chip_metadata,
            "thumbnail": thumbnail,
            "chips_dir": output_dir / "chips"
        }

    def create_control_chip(self, original_image_path: Path,
                            detections: List[Dict],
                            target_idx: int) -> Optional[Image.Image]:
        """Extract a control chip from a clean area near the detection"""
        try:
            img = Image.open(original_image_path)
            W, H = img.size

            target = detections[target_idx]
            target_bbox = target['bbox_xyxy']
            tx0, ty0, tx1, ty1 = target_bbox

            # Find a clean area (no other detections)
            # Try areas around the target
            margin = int(min(W, H) * 0.1)  # 10% margin

            candidates = [
                # Left of target
                [max(0, tx0 - margin - (tx1 - tx0)), ty0,
                 max(0, tx0 - margin), ty1],
                # Right of target
                [min(W, tx1 + margin), ty0,
                 min(W, tx1 + margin + (tx1 - tx0)), ty1],
                # Above target
                [tx0, max(0, ty0 - margin - (ty1 - ty0)),
                 tx1, max(0, ty0 - margin)],
                # Below target
                [tx0, min(H, ty1 + margin),
                 tx1, min(H, ty1 + margin + (ty1 - ty0))]
            ]

            # Find first candidate that doesn't overlap with any detection
            for cx0, cy0, cx1, cy1 in candidates:
                if cx1 <= cx0 or cy1 <= cy0:
                    continue

                # Check for overlaps
                overlaps = False
                for det in detections:
                    dx0, dy0, dx1, dy1 = det['bbox_xyxy']
                    if not (cx1 < dx0 or cx0 > dx1 or cy1 < dy0 or cy0 > dy1):
                        overlaps = True
                        break

                if not overlaps:
                    # Valid control area found
                    control = img.crop((cx0, cy0, cx1, cy1))
                    return control

            return None

        except Exception as e:
            logger.error(f"Error creating control chip: {e}")
            return None

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def create_verification_prompt(self,
                                   class_name: str,
                                   confidence: float,
                                   location_description: str,
                                   margin_percent: float = 15) -> str:
        """Create class-specific verification prompt"""
        # Determine prompt type
        prompt_type = "defect" if class_name.lower() in [
            "water_stain", "water stain", "chipped_paint", "chipped paint",
            "crack", "hole", "mold", "discoloration"
        ] else "structural"

        # Get class characteristics
        chars_key = class_name.lower().replace("_", " ")
        characteristics = CLASS_CHARACTERISTICS.get(
            chars_key,
            "Standard visual characteristics for this object type"
        )

        prompt = VERIFICATION_PROMPTS[prompt_type].format(
            class_name=class_name,
            confidence=confidence,
            location_description=location_description,
            margin_percent=margin_percent,
            class_characteristics=characteristics
        )

        return prompt

    def verify_single_chip(self,
                           chip_path: Path,
                           class_name: str,
                           confidence: float,
                           thumbnail_path: Optional[Path] = None,
                           control_chip: Optional[Image.Image] = None) -> ChipVerification:
        """Verify a single chip using LM Studio"""
        start_time = time.time()

        try:
            # Build message content
            prompt = self.create_verification_prompt(
                class_name=class_name,
                confidence=confidence,
                location_description="See thumbnail for context",
                margin_percent=15
            )

            content = [{"type": "text", "text": prompt}]

            # Add main chip
            chip_b64 = self.encode_image(chip_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{chip_b64}"}
            })

            # Add thumbnail if available
            if thumbnail_path and thumbnail_path.exists():
                thumb_b64 = self.encode_image(thumbnail_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{thumb_b64}"}
                })

            # Add control chip if available
            if control_chip:
                import io
                buffer = io.BytesIO()
                control_chip.save(buffer, format='JPEG')
                control_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{control_b64}"}
                })

            # Prepare API request
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at visual verification of detected objects. Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.1,  # Low temperature for consistent verification
                "max_tokens": 500,
                "stream": False
            }

            # Make request
            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")

            # Parse response
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Try to parse JSON from response
            try:
                # Handle potential markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                result = json.loads(content.strip())

                return ChipVerification(
                    chip_path=str(chip_path),
                    detection_idx=-1,  # Will be set later
                    class_name=class_name,
                    original_confidence=confidence,
                    is_valid=result.get("is_valid", False),
                    verification_confidence=result.get("confidence", 0.5),
                    reasoning=result.get("reasoning", "No reasoning provided"),
                    visual_evidence=result.get("visual_evidence", []),
                    alternative_explanation=result.get("alternative_explanation"),
                    processing_time=time.time() - start_time
                )

            except json.JSONDecodeError:
                # Fallback: try to extract boolean from text
                content_lower = content.lower()
                is_valid = "valid" in content_lower and "invalid" not in content_lower

                return ChipVerification(
                    chip_path=str(chip_path),
                    detection_idx=-1,
                    class_name=class_name,
                    original_confidence=confidence,
                    is_valid=is_valid,
                    verification_confidence=0.5,
                    reasoning=content[:200],
                    visual_evidence=[],
                    processing_time=time.time() - start_time
                )

        except Exception as e:
            logger.error(f"Error verifying chip {chip_path}: {e}")
            return ChipVerification(
                chip_path=str(chip_path),
                detection_idx=-1,
                class_name=class_name,
                original_confidence=confidence,
                is_valid=False,
                verification_confidence=0.0,
                reasoning="Verification failed",
                visual_evidence=[],
                error=str(e),
                processing_time=time.time() - start_time
            )

    def verify_batch(self,
                     chips: List[Tuple[Path, str, float]],
                     thumbnail_path: Optional[Path] = None) -> List[ChipVerification]:
        """Verify a batch of chips"""
        results = []

        for chip_path, class_name, confidence in chips:
            if self.debug:
                logger.info(f"Verifying {chip_path.name} ({class_name}, conf={confidence:.2f})")

            result = self.verify_single_chip(
                chip_path=chip_path,
                class_name=class_name,
                confidence=confidence,
                thumbnail_path=thumbnail_path
            )
            results.append(result)

            # Small delay to avoid overwhelming the API
            time.sleep(0.5)

        return results

    def aggregate_detection_results(self,
                                    detection_idx: int,
                                    verifications: List[ChipVerification]) -> DetectionVerification:
        """Aggregate multiple chip verifications for a single detection"""
        if not verifications:
            return DetectionVerification(
                detection_idx=detection_idx,
                class_name="unknown",
                original_confidence=0.0,
                bbox=[],
                chip_verifications=[],
                final_verdict="INVALID",
                consensus_confidence=0.0,
                reasoning="No verifications available"
            )

        # Get class info from first verification
        class_name = verifications[0].class_name
        original_conf = verifications[0].original_confidence

        # Calculate consensus
        valid_count = sum(1 for v in verifications if v.is_valid)
        valid_ratio = valid_count / len(verifications)
        avg_confidence = np.mean([v.verification_confidence for v in verifications])

        # Get thresholds for this class
        thresholds = CLASS_THRESHOLDS.get(
            class_name.lower().replace("_", " "),
            CLASS_THRESHOLDS["default"]
        )

        # Determine verdict
        if valid_ratio >= thresholds["consensus"] and avg_confidence >= thresholds["confidence"]:
            verdict = "VALID"
            reasoning = f"Verified with {valid_count}/{len(verifications)} positive confirmations"
        elif valid_ratio < 0.3 or avg_confidence < 0.4:
            verdict = "INVALID"
            reasoning = f"Only {valid_count}/{len(verifications)} positive confirmations"
        else:
            verdict = "UNCERTAIN"
            reasoning = f"Mixed results: {valid_count}/{len(verifications)} positive, needs review"

        return DetectionVerification(
            detection_idx=detection_idx,
            class_name=class_name,
            original_confidence=original_conf,
            bbox=[],  # Will be filled from detection data
            chip_verifications=verifications,
            final_verdict=verdict,
            consensus_confidence=avg_confidence,
            reasoning=reasoning
        )

    def verify_detections(self,
                          output_dir: str,
                          original_image_path: Optional[str] = None,
                          max_chips_per_detection: int = 3) -> Dict:
        """Main verification pipeline"""
        output_dir = Path(output_dir)
        logger.info(f"Starting verification for {output_dir}")

        # Load GroundingDINO outputs
        gd_data = self.load_grounding_dino_output(output_dir)
        detections = gd_data["predictions"]["detections"]
        chip_metadata = gd_data["chip_metadata"]
        thumbnail = gd_data["thumbnail"]

        if not detections:
            logger.warning("No detections to verify")
            return {"results": [], "summary": {}}

        # Group chips by detection
        chips_by_detection = defaultdict(list)
        for chip_info in chip_metadata:
            det_idx = chip_info["detection_idx"]
            chip_path = gd_data["chips_dir"] / chip_info["filename"]
            if chip_path.exists():
                chips_by_detection[det_idx].append(chip_info)

        # Verify each detection
        all_results = []
        valid_count = 0
        invalid_count = 0
        uncertain_count = 0

        for det_idx, detection in enumerate(detections):
            logger.info(f"\nVerifying detection {det_idx}: {detection['label']}")

            # Get chips for this detection
            detection_chips = chips_by_detection.get(det_idx, [])
            if not detection_chips:
                logger.warning(f"No chips found for detection {det_idx}")
                continue

            # Limit chips per detection
            detection_chips = detection_chips[:max_chips_per_detection]

            # Prepare chip data
            chips_to_verify = []
            for chip_info in detection_chips:
                chip_path = gd_data["chips_dir"] / chip_info["filename"]
                chips_to_verify.append((
                    chip_path,
                    detection["label"].split("(")[0].strip(),  # Remove confidence from label
                    detection["score"] or 0.5
                ))

            # Verify chips
            verifications = self.verify_batch(chips_to_verify, thumbnail)

            # Update detection indices
            for i, v in enumerate(verifications):
                v.detection_idx = det_idx

            # Aggregate results
            detection_result = self.aggregate_detection_results(det_idx, verifications)
            detection_result.bbox = detection["bbox_xyxy"]

            all_results.append(detection_result)

            # Count verdicts
            if detection_result.final_verdict == "VALID":
                valid_count += 1
            elif detection_result.final_verdict == "INVALID":
                invalid_count += 1
            else:
                uncertain_count += 1

            # Log result
            logger.info(f"  Verdict: {detection_result.final_verdict} "
                        f"(confidence: {detection_result.consensus_confidence:.2f})")
            logger.info(f"  Reasoning: {detection_result.reasoning}")

        # Create summary
        summary = {
            "total_detections": len(detections),
            "verified_detections": len(all_results),
            "valid": valid_count,
            "invalid": invalid_count,
            "uncertain": uncertain_count,
            "verification_rate": valid_count / len(all_results) if all_results else 0,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save results
        output_file = output_dir / "verification_results.json"
        results_dict = {
            "summary": summary,
            "results": [self._detection_to_dict(r) for r in all_results],
            "config": {
                "model": self.model_name,
                "thresholds": CLASS_THRESHOLDS,
                "max_chips_per_detection": max_chips_per_detection
            }
        }

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Verification Complete!")
        logger.info(f"Valid: {valid_count}, Invalid: {invalid_count}, Uncertain: {uncertain_count}")
        logger.info(f"Results saved to: {output_file}")

        return results_dict

    def _detection_to_dict(self, detection: DetectionVerification) -> Dict:
        """Convert DetectionVerification to dictionary"""
        return {
            "detection_idx": detection.detection_idx,
            "class_name": detection.class_name,
            "original_confidence": detection.original_confidence,
            "bbox": detection.bbox,
            "final_verdict": detection.final_verdict,
            "consensus_confidence": detection.consensus_confidence,
            "reasoning": detection.reasoning,
            "chip_verifications": [
                {
                    "chip_path": v.chip_path,
                    "is_valid": v.is_valid,
                    "confidence": v.verification_confidence,
                    "reasoning": v.reasoning,
                    "visual_evidence": v.visual_evidence,
                    "alternative": v.alternative_explanation
                } for v in detection.chip_verifications
            ]
        }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Verify GroundingDINO detection chips")
    parser.add_argument("output_dir", help="Path to GroundingDINO output directory")
    parser.add_argument("--original-image", help="Path to original image (for control chips)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name")
    parser.add_argument("--lm-studio-url", default=LM_STUDIO_URL, help="LM Studio API URL")
    parser.add_argument("--max-chips", type=int, default=3, help="Max chips per detection")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size for API calls")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Create verifier
    verifier = ChipVerifier(
        lm_studio_url=args.lm_studio_url,
        model_name=args.model,
        batch_size=args.batch_size,
        debug=args.debug
    )

    # Run verification
    results = verifier.verify_detections(
        output_dir=args.output_dir,
        original_image_path=args.original_image,
        max_chips_per_detection=args.max_chips
    )

    # Print summary
    summary = results["summary"]
    print(f"\n✅ Verification Summary:")
    print(f"  Total Detections: {summary['total_detections']}")
    print(f"  Valid: {summary['valid']}")
    print(f"  Invalid: {summary['invalid']}")
    print(f"  Uncertain: {summary['uncertain']}")
    print(f"  Verification Rate: {summary['verification_rate']:.1%}")


if __name__ == "__main__":
    main()