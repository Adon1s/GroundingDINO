"""
DINO-X Detection Client
-----------------------
Client for the DINO-X object detection API from DeepDataSpace.

Supports two methods:
1. Official SDK (dds-cloudapi-sdk) - Recommended
2. Raw HTTP API - Fallback

API Documentation: https://api.deepdataspace.com/
SDK GitHub: https://github.com/deepdataspace/dds-cloudapi-sdk

Usage:
    client = DINOXClient(api_token="your_token")
    
    # Detect objects in an image
    result = client.detect(
        image_path=Path("photo.jpg"),
        prompt="crack.stain.damage.mold",
        bbox_threshold=0.25,
    )
    
    # Result contains bounding boxes and masks
    for obj in result.objects:
        print(f"{obj.category}: {obj.score:.2f} at {obj.bbox}")
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

logger = logging.getLogger(__name__)

# Try to import official SDK
try:
    from dds_cloudapi_sdk import Client, Config
    from dds_cloudapi_sdk.tasks.v2_task import V2Task, create_task_with_local_image_auto_resize
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.info("dds-cloudapi-sdk not installed, using HTTP API fallback")


# ═══════════════════════════════════════════════════════════════════════════════fopenai
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    """Bounding box coordinates [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    def to_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class DetectedObject:
    """Single detected object from DINO-X."""
    category: str
    score: float
    bbox: BoundingBox
    mask: Optional[Dict[str, Any]] = None  # COCO RLE format if requested
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "category": self.category,
            "score": self.score,
            "bbox": self.bbox.to_dict(),
        }
        if self.mask:
            result["mask"] = self.mask
        return result


@dataclass
class DetectionResult:
    """Complete detection result from DINO-X."""
    task_uuid: str
    status: str
    objects: List[DetectedObject] = field(default_factory=list)
    image_size: Optional[Tuple[int, int]] = None  # (width, height)
    processing_time: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    
    @property
    def object_count(self) -> int:
        return len(self.objects)
    
    def filter_by_category(self, category: str) -> List[DetectedObject]:
        """Get all objects of a specific category."""
        return [obj for obj in self.objects if obj.category == category]
    
    def filter_by_score(self, min_score: float) -> List[DetectedObject]:
        """Get all objects above a score threshold."""
        return [obj for obj in self.objects if obj.score >= min_score]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_uuid": self.task_uuid,
            "status": self.status,
            "object_count": self.object_count,
            "objects": [obj.to_dict() for obj in self.objects],
            "image_size": self.image_size,
            "processing_time": self.processing_time,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DINO-X Client
# ═══════════════════════════════════════════════════════════════════════════════

class DINOXClient:
    """
    Client for DINO-X object detection API.
    
    Uses official SDK when available, falls back to HTTP API.
    """
    
    # API endpoints
    BASE_URL = "https://api.deepdataspace.com"
    DETECTION_PATH = "/v2/task/dinox/detection"
    STATUS_PATH = "/v2/task_status"
    
    # Default model
    DEFAULT_MODEL = "DINO-X-1.0"
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        use_sdk: bool = True,
        poll_interval: float = 1.0,
        max_poll_time: float = 120.0,
        request_timeout: float = 10.0,
    ):
        """
        Initialize DINO-X client.
        
        Args:
            api_token: API token (or set DINOX_API_TOKEN env var)
            use_sdk: Use official SDK if available (default: True)
            poll_interval: Seconds between status polls (default: 1.0)
            max_poll_time: Maximum seconds to wait for task (default: 120.0)
            request_timeout: HTTP request timeout (default: 10.0)
        """
        self.api_token = api_token or os.environ.get("DINOX_API_TOKEN") or os.environ.get("DINOX_API_KEY")
        
        if not self.api_token:
            raise ValueError(
                "DINO-X API token required. Set DINOX_API_TOKEN environment variable "
                "or pass api_token parameter."
            )
        
        self.use_sdk = use_sdk and SDK_AVAILABLE
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.request_timeout = request_timeout
        
        # Initialize SDK client if available
        self._sdk_client = None
        if self.use_sdk:
            try:
                config = Config(self.api_token)
                self._sdk_client = Client(config)
                logger.info("Using DINO-X official SDK")
            except Exception as e:
                logger.warning(f"Failed to initialize SDK client: {e}, falling back to HTTP API")
                self.use_sdk = False
        
        if not self.use_sdk:
            logger.info("Using DINO-X HTTP API")
    
    def _image_to_base64(self, image_path: Path) -> str:
        """Convert image file to base64 string."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers with auth token."""
        return {
            "Content-Type": "application/json",
            "Token": self.api_token,
        }
    
    def _parse_detection_result(
        self,
        task_uuid: str,
        raw_result: Dict[str, Any],
        processing_time: float = 0.0,
    ) -> DetectionResult:
        """Parse raw API response into DetectionResult."""
        objects = []
        
        # Extract objects from result
        # The API returns objects in result.objects or result.predictions
        raw_objects = raw_result.get("objects", []) or raw_result.get("predictions", [])
        
        for obj in raw_objects:
            # Parse bounding box - API returns [x1, y1, x2, y2] or {"rect": [...]}
            bbox_data = obj.get("bbox") or obj.get("rect") or obj.get("box", [0, 0, 0, 0])
            if isinstance(bbox_data, dict):
                bbox_data = bbox_data.get("rect", [0, 0, 0, 0])
            
            bbox = BoundingBox(
                x1=float(bbox_data[0]),
                y1=float(bbox_data[1]),
                x2=float(bbox_data[2]),
                y2=float(bbox_data[3]),
            )
            
            # Parse category and score
            category = obj.get("category") or obj.get("label") or obj.get("class", "unknown")
            score = float(obj.get("score") or obj.get("confidence", 0.0))
            
            # Parse mask if present
            mask = obj.get("mask") or obj.get("segmentation")
            
            objects.append(DetectedObject(
                category=category,
                score=score,
                bbox=bbox,
                mask=mask,
            ))
        
        # Extract image size if available
        image_size = None
        if "image_size" in raw_result:
            size = raw_result["image_size"]
            image_size = (size.get("width", 0), size.get("height", 0))
        elif "width" in raw_result and "height" in raw_result:
            image_size = (raw_result["width"], raw_result["height"])
        
        return DetectionResult(
            task_uuid=task_uuid,
            status="success",
            objects=objects,
            image_size=image_size,
            processing_time=processing_time,
            raw_response=raw_result,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SDK-based detection
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_with_sdk(
        self,
        image_path: Path,
        prompt: str,
        model: str,
        targets: List[str],
        bbox_threshold: float,
        iou_threshold: float,
        mask_format: str,
    ) -> DetectionResult:
        """Run detection using official SDK."""
        start_time = time.time()
        
        api_body = {
            "model": model,
            "prompt": {
                "type": "text",
                "text": prompt,
            },
            "targets": targets,
            "bbox_threshold": bbox_threshold,
            "iou_threshold": iou_threshold,
        }
        
        if "mask" in targets:
            api_body["mask_format"] = mask_format
        
        # Create task with local image (auto-resized for faster processing)
        task = create_task_with_local_image_auto_resize(
            api_path=self.DETECTION_PATH,
            api_body_without_image=api_body,
            image_path=str(image_path),
        )
        
        # Set timeout
        task.set_request_timeout(int(self.request_timeout))
        
        # Run task (SDK handles polling)
        logger.debug(f"Running DINO-X detection via SDK for {image_path.name}")
        self._sdk_client.run_task(task)
        
        processing_time = time.time() - start_time
        
        # Parse result
        task_uuid = getattr(task, 'task_uuid', 'sdk_task')
        return self._parse_detection_result(
            task_uuid=task_uuid,
            raw_result=task.result,
            processing_time=processing_time,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HTTP API-based detection
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_with_http(
        self,
        image_path: Path,
        prompt: str,
        model: str,
        targets: List[str],
        bbox_threshold: float,
        iou_threshold: float,
        mask_format: str,
    ) -> DetectionResult:
        """Run detection using HTTP API."""
        start_time = time.time()
        headers = self._build_headers()
        
        # Step 1: Prepare request body
        api_body = {
            "model": model,
            "image": self._image_to_base64(image_path),
            "prompt": {
                "type": "text",
                "text": prompt,
            },
            "targets": targets,
            "bbox_threshold": bbox_threshold,
            "iou_threshold": iou_threshold,
        }
        
        if "mask" in targets:
            api_body["mask_format"] = mask_format
        
        # Step 2: Create task
        logger.debug(f"Creating DINO-X detection task for {image_path.name}")
        
        create_url = f"{self.BASE_URL}{self.DETECTION_PATH}"
        resp = requests.post(
            url=create_url,
            json=api_body,
            headers=headers,
            timeout=self.request_timeout,
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"DINO-X task creation failed: HTTP {resp.status_code} - {resp.text[:500]}")
        
        create_result = resp.json()
        
        if create_result.get("code") != 0:
            raise RuntimeError(f"DINO-X task creation failed: {create_result.get('msg', 'Unknown error')}")
        
        task_uuid = create_result["data"]["task_uuid"]
        logger.debug(f"Created DINO-X task: {task_uuid}")
        
        # Step 3: Poll for results
        status_url = f"{self.BASE_URL}{self.STATUS_PATH}/{task_uuid}"
        poll_start = time.time()
        
        while True:
            if time.time() - poll_start > self.max_poll_time:
                raise TimeoutError(f"DINO-X task {task_uuid} timed out after {self.max_poll_time}s")
            
            resp = requests.get(
                url=status_url,
                headers=headers,
                timeout=self.request_timeout,
            )
            
            if resp.status_code != 200:
                raise RuntimeError(f"DINO-X status check failed: HTTP {resp.status_code}")
            
            status_result = resp.json()
            status = status_result["data"]["status"]
            
            if status == "success":
                logger.debug(f"DINO-X task {task_uuid} completed successfully")
                break
            elif status == "failed":
                error = status_result["data"].get("error", "Unknown error")
                raise RuntimeError(f"DINO-X task {task_uuid} failed: {error}")
            elif status in ["waiting", "running"]:
                logger.debug(f"DINO-X task {task_uuid} status: {status}")
                time.sleep(self.poll_interval)
            else:
                raise RuntimeError(f"DINO-X task {task_uuid} unknown status: {status}")
        
        processing_time = time.time() - start_time
        
        # Parse result
        return self._parse_detection_result(
            task_uuid=task_uuid,
            raw_result=status_result["data"]["result"],
            processing_time=processing_time,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect(
        self,
        image_path: Union[str, Path],
        prompt: str,
        model: str = DEFAULT_MODEL,
        targets: Optional[List[str]] = None,
        bbox_threshold: float = 0.25,
        iou_threshold: float = 0.8,
        mask_format: str = "coco_rle",
    ) -> DetectionResult:
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to image file
            prompt: Detection prompt (dot-separated categories)
                    e.g., "crack.stain.damage.mold.water_damage"
            model: Model name (default: "DINO-X-1.0")
            targets: Output targets (default: ["bbox"])
                     Options: "bbox", "mask"
            bbox_threshold: Confidence threshold (default: 0.25)
            iou_threshold: NMS IoU threshold (default: 0.8)
            mask_format: Mask format if masks requested (default: "coco_rle")
        
        Returns:
            DetectionResult with detected objects
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if targets is None:
            targets = ["bbox"]
        
        logger.info(f"DINO-X detection: {image_path.name} | prompt='{prompt}' | threshold={bbox_threshold}")
        
        if self.use_sdk and self._sdk_client:
            return self._detect_with_sdk(
                image_path=image_path,
                prompt=prompt,
                model=model,
                targets=targets,
                bbox_threshold=bbox_threshold,
                iou_threshold=iou_threshold,
                mask_format=mask_format,
            )
        else:
            return self._detect_with_http(
                image_path=image_path,
                prompt=prompt,
                model=model,
                targets=targets,
                bbox_threshold=bbox_threshold,
                iou_threshold=iou_threshold,
                mask_format=mask_format,
            )
    
    def detect_issues(
        self,
        image_path: Union[str, Path],
        issue_categories: Optional[List[str]] = None,
        bbox_threshold: float = 0.25,
        include_masks: bool = False,
    ) -> DetectionResult:
        """
        Convenience method for detecting property issues.
        
        Args:
            image_path: Path to image file
            issue_categories: List of issue types to detect
                             (default: common property issues)
            bbox_threshold: Confidence threshold
            include_masks: Include segmentation masks
        
        Returns:
            DetectionResult with detected issues
        """
        if issue_categories is None:
            issue_categories = [
                "crack", "stain", "damage", "mold", "water_damage",
                "rust", "peeling_paint", "hole", "dent", "scratch",
                "discoloration", "wear", "rot", "corrosion", "leak",
            ]
        
        prompt = ".".join(issue_categories)
        targets = ["bbox", "mask"] if include_masks else ["bbox"]
        
        return self.detect(
            image_path=image_path,
            prompt=prompt,
            targets=targets,
            bbox_threshold=bbox_threshold,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════════

def create_dinox_client(
    api_token: Optional[str] = None,
    use_sdk: bool = True,
) -> DINOXClient:
    """
    Create a DINO-X client.
    
    Args:
        api_token: API token (or uses DINOX_API_TOKEN env var)
        use_sdk: Use official SDK if available
    
    Returns:
        Configured DINOXClient
    """
    return DINOXClient(api_token=api_token, use_sdk=use_sdk)


def create_dinox_client_from_config(config: Any) -> DINOXClient:
    """
    Create a DINO-X client from pipeline_config module.
    
    Args:
        config: pipeline_config module
    
    Returns:
        Configured DINOXClient
    """
    api_token = (
        getattr(config, "DINOX_API_TOKEN", None) or
        getattr(config, "DINOX_API_KEY", None) or
        os.environ.get("DINOX_API_TOKEN") or
        os.environ.get("DINOX_API_KEY")
    )
    
    use_sdk = getattr(config, "DINOX_USE_SDK", True)
    poll_interval = getattr(config, "DINOX_POLL_INTERVAL", 1.0)
    max_poll_time = getattr(config, "DINOX_MAX_POLL_TIME", 120.0)
    
    return DINOXClient(
        api_token=api_token,
        use_sdk=use_sdk,
        poll_interval=poll_interval,
        max_poll_time=max_poll_time,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI for testing
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DINO-X Detection CLI")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--prompt", default="crack.stain.damage.mold", help="Detection prompt")
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--token", help="API token (or set DINOX_API_TOKEN)")
    parser.add_argument("--no-sdk", action="store_true", help="Use HTTP API instead of SDK")
    parser.add_argument("--masks", action="store_true", help="Include segmentation masks")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create client
    client = DINOXClient(
        api_token=args.token,
        use_sdk=not args.no_sdk,
    )
    
    # Run detection
    targets = ["bbox", "mask"] if args.masks else ["bbox"]
    result = client.detect(
        image_path=Path(args.image),
        prompt=args.prompt,
        bbox_threshold=args.threshold,
        targets=targets,
    )
    
    # Print results
    print(f"\n{'=' * 60}")
    print(f"DINO-X Detection Results")
    print(f"{'=' * 60}")
    print(f"Task UUID: {result.task_uuid}")
    print(f"Status: {result.status}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print(f"Objects Found: {result.object_count}")
    print(f"{'=' * 60}")
    
    for i, obj in enumerate(result.objects, 1):
        print(f"\n[{i}] {obj.category}")
        print(f"    Score: {obj.score:.3f}")
        print(f"    BBox: ({obj.bbox.x1:.1f}, {obj.bbox.y1:.1f}) - ({obj.bbox.x2:.1f}, {obj.bbox.y2:.1f})")
        print(f"    Size: {obj.bbox.width:.1f} x {obj.bbox.height:.1f}")
    
    # Output JSON
    print(f"\n{'=' * 60}")
    print("JSON Output:")
    print(json.dumps(result.to_dict(), indent=2))
