#!/usr/bin/env python3
"""
Property Summarizer for RealtorVision Pipeline
-----------------------------------------------
Aggregates per-image issues and generates property-level summaries via VLM.

This script:
1. Collects all issues_natural_language from each image's scene classifier output
2. Groups issues by room/scene type to manage context size
3. Generates per-room-category summaries via VLM
4. Produces an overall property investment summary

Usage:
  python property_summarizer.py path/to/photo_intel.json
  python property_summarizer.py path/to/photo_intel.json --output summary.json
  python property_summarizer.py path/to/photo_intel.json --model qwen3-vl-30b --debug
"""

import os
import sys
import json
import time
import logging
import argparse
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Literal
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import requests

# Import VLMClient for unified LM Studio / OpenAI support
try:
    from vlm_client import VLMClient, create_vlm_client

    VLM_CLIENT_AVAILABLE = True
except ImportError:
    VLMClient = None
    create_vlm_client = None
    VLM_CLIENT_AVAILABLE = False

# Console encoding safety (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration ────────────────────────────────────────────────────────────
try:
    import pipeline_config as cfg

    LM_STUDIO_URL = cfg.LM_STUDIO_URL
    DEFAULT_MODEL = cfg.LM_STUDIO_MODEL
except ImportError:
    LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://100.102.92.1:1234")
    DEFAULT_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-vl-30b")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SUMMARY_VERSION = "v1"

# Scene type groupings for summarization
SCENE_GROUPS = {
    "kitchen": ["kitchen", "pantry"],
    "bathroom": ["bathroom"],
    "bedroom": ["bedroom", "closet"],
    "living_areas": ["living_room", "dining_room", "home_office", "hallway", "stairway"],
    "utility": ["laundry_room", "basement", "attic", "garage"],
    "exterior": [
        "exterior_front", "exterior_back", "exterior_side",
        "yard", "patio", "deck", "balcony", "driveway",
        "pool", "garden", "hvac"
    ],
    "other": ["floor_plan", "aerial_view", "street_view", "unknown"]
}

# Reverse lookup: scene -> group
SCENE_TO_GROUP = {}
for group_name, scenes in SCENE_GROUPS.items():
    for scene in scenes:
        SCENE_TO_GROUP[scene] = group_name


# ── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class IssueItem:
    """A single issue from a photo."""
    description: str
    rough_category: str
    location_hint: str
    source_image: str
    scene: str


@dataclass
class RoomGroupSummary:
    """Summary for a room group (e.g., all kitchens, all bathrooms)."""
    group_name: str
    scenes_included: List[str]
    image_count: int
    issue_count: int
    summary_text: str
    key_concerns: List[str]
    severity_assessment: str  # "minor", "moderate", "significant", "major"
    estimated_scope: str  # brief description of work needed


@dataclass
class PropertySummary:
    """Complete property-level summary."""
    property_key: str
    job_id: str
    created_at: str
    summary_version: str

    # Overall assessment
    overall_condition: str  # "excellent", "good", "fair", "poor"
    overall_summary: str  # 2-3 paragraph narrative
    investment_verdict: str  # "strong_buy", "buy", "hold", "caution", "avoid"
    investment_rationale: str

    # Renovation scope
    renovation_scope: str  # "cosmetic", "light", "moderate", "heavy", "gut"
    renovation_priorities: List[str]  # ordered list of what to address first

    # Risk assessment
    risk_flags: List[str]  # major concerns that could affect value/safety
    deferred_maintenance: List[str]  # items that need attention but aren't urgent

    # Room-by-room
    room_summaries: Dict[str, RoomGroupSummary]

    # Stats
    total_images_analyzed: int
    total_issues_found: int
    issues_by_category: Dict[str, int]

    # Processing info
    processing_time: float
    model_used: str
    error: Optional[str] = None


# ── Helper Functions ─────────────────────────────────────────────────────────
def _extract_json(text: Any) -> Optional[dict]:
    """
    Extract JSON from LLM response, handling markdown fences.

    Accepts Any type:
    - If already a dict, returns it directly
    - If a list, returns None (don't guess)
    - Otherwise, casts to str and attempts extraction

    Never throws on bad input; always returns dict or None.
    """
    # If already a dict, return directly
    if isinstance(text, dict):
        return text

    # If a list, return None (don't guess)
    if isinstance(text, list):
        return None

    # Handle None or empty
    if text is None:
        return None

    # Cast to string for all other types
    try:
        s = str(text).strip()
    except Exception:
        return None

    if not s:
        return None

    # Strip markdown code fences
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    # Try direct parse
    try:
        result = json.loads(s)
        # Ensure we return a dict, not other JSON types
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: find JSON object bounds
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(s[start:end + 1])
            if isinstance(result, dict):
                return result
            return None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    return None


def _call_vlm(
        prompt: str,
        model_name: str,
        lm_studio_url: str,
        timeout: int = 60
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call the VLM with a text-only prompt.
    Returns (response_text, error_message).
    """
    messages = [
        {
            "role": "system",
            "content": "You are a real estate investment analyst. Respond with valid JSON only."
        },
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4096,
        "stream": False
    }

    try:
        resp = requests.post(
            f"{lm_studio_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=timeout
        )

        if resp.status_code != 200:
            return None, f"API error: HTTP {resp.status_code}"

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content, None

    except requests.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, str(e)


# ── Prompt Builders ──────────────────────────────────────────────────────────
def build_room_group_prompt(
        group_name: str,
        issues: List[IssueItem],
        image_count: int
) -> str:
    """Build prompt for summarizing a room group's issues."""

    # Format issues as a list
    issues_text = ""
    for i, issue in enumerate(issues, 1):
        issues_text += f"{i}. [{issue.rough_category}] {issue.description}"
        if issue.location_hint:
            issues_text += f" (Location: {issue.location_hint})"
        issues_text += f" [from: {issue.source_image}]\n"

    return f"""You are analyzing issues found in the {group_name.upper()} areas of a property.

CONTEXT:
- Room group: {group_name}
- Number of images analyzed: {image_count}
- Total issues found: {len(issues)}

ISSUES FOUND:
{issues_text}

TASK: Summarize these issues into a coherent assessment for this room group.

Return a JSON object with this exact structure:
{{
    "summary_text": "<2-3 sentence summary of the overall condition of this area>",
    "key_concerns": ["<concern 1>", "<concern 2>", ...],
    "severity_assessment": "<one of: minor, moderate, significant, major>",
    "estimated_scope": "<brief description of work needed, e.g., 'cosmetic updates' or 'moderate repairs needed'>"
}}

SEVERITY GUIDELINES:
- minor: Cosmetic issues only, no functional problems
- moderate: Some repairs needed but nothing structural or urgent
- significant: Multiple issues requiring professional attention
- major: Serious problems affecting safety, structure, or habitability

Respond with JSON only, no additional text."""


def build_property_summary_prompt(
        property_key: str,
        room_summaries: Dict[str, RoomGroupSummary],
        renovation_needs: Optional[Dict[str, Any]],
        total_images: int,
        total_issues: int,
        issues_by_category: Dict[str, int]
) -> str:
    """Build prompt for the overall property summary."""

    # Format room summaries
    room_text = ""
    for group_name, summary in room_summaries.items():
        room_text += f"\n### {group_name.upper()} ({summary.image_count} images, {summary.issue_count} issues)\n"
        room_text += f"Severity: {summary.severity_assessment}\n"
        room_text += f"Summary: {summary.summary_text}\n"
        if summary.key_concerns:
            room_text += f"Key concerns: {', '.join(summary.key_concerns)}\n"
        room_text += f"Scope: {summary.estimated_scope}\n"

    # Format category breakdown
    category_text = ", ".join([f"{cat}: {count}" for cat, count in issues_by_category.items()])

    # Format cost estimates if available
    cost_text = ""
    if renovation_needs:
        grand_total = renovation_needs.get("grand_total", {})
        low = grand_total.get("est_cost_low", 0)
        high = grand_total.get("est_cost_high", 0)
        if low > 0 or high > 0:
            cost_text = f"\nEstimated renovation costs: ${low:,.0f} - ${high:,.0f}"

    return f"""You are a real estate investment analyst preparing a comprehensive property assessment.

PROPERTY: {property_key}

ANALYSIS OVERVIEW:
- Total images analyzed: {total_images}
- Total issues identified: {total_issues}
- Issues by category: {category_text}{cost_text}

ROOM-BY-ROOM SUMMARIES:
{room_text}

TASK: Create a comprehensive property investment summary.

Return a JSON object with this exact structure:
{{
    "overall_condition": "<one of: excellent, good, fair, poor>",
    "overall_summary": "<2-3 paragraph narrative describing the property's condition, highlighting both positives and negatives>",
    "investment_verdict": "<one of: strong_buy, buy, hold, caution, avoid>",
    "investment_rationale": "<1-2 sentences explaining the verdict>",
    "renovation_scope": "<one of: cosmetic, light, moderate, heavy, gut>",
    "renovation_priorities": ["<priority 1>", "<priority 2>", "<priority 3>", ...],
    "risk_flags": ["<any major concerns that could affect value or safety>"],
    "deferred_maintenance": ["<items needing attention but not urgent>"]
}}

CONDITION GUIDELINES:
- excellent: Move-in ready, minimal issues, updated finishes
- good: Well-maintained, minor cosmetic issues only
- fair: Functional but dated, multiple areas need updating
- poor: Significant repairs needed, major systems concerns

INVESTMENT VERDICT GUIDELINES:
- strong_buy: Excellent condition or high upside potential with manageable improvements
- buy: Good value, reasonable renovation scope
- hold: Average opportunity, renovation costs may eat into margins
- caution: Significant issues that require careful cost analysis
- avoid: Major structural/systems concerns or excessive renovation needs

RENOVATION SCOPE:
- cosmetic: Paint, flooring, fixtures only (<$15k typical)
- light: Cosmetic plus minor repairs ($15-40k typical)
- moderate: Kitchen/bath updates, some systems ($40-80k typical)
- heavy: Major remodel, multiple systems ($80-150k typical)
- gut: Complete renovation needed (>$150k typical)

Respond with JSON only, no additional text."""


# ── Main Summarizer Class ────────────────────────────────────────────────────
class PropertySummarizer:
    """Aggregates per-image issues and generates property-level summaries."""

    def __init__(
            self,
            lm_studio_url: Optional[str] = None,
            model_name: str = DEFAULT_MODEL,
            debug: bool = False,
            api_key: Optional[str] = None,
            provider: Literal["lmstudio", "openai"] = "lmstudio",
            max_tokens: int = 3000,
            temperature: float = 0.2,
            max_retries: int = 2,
            **kwargs,  # future-proof
    ):
        # Use provided URL or fall back to default for LM Studio
        self.lm_studio_url = (lm_studio_url or LM_STUDIO_URL).rstrip('/') if lm_studio_url else LM_STUDIO_URL.rstrip(
            '/')
        self.model_name = model_name
        self.debug = debug
        self.api_key = api_key
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        if debug:
            logger.setLevel(logging.DEBUG)

        # Create VLMClient if available
        self.vlm_client = None
        if VLM_CLIENT_AVAILABLE:
            self.vlm_client = VLMClient(
                default_timeout=120,
                default_max_tokens=max_tokens,
                default_temperature=temperature,
            )

        logger.info(f"PropertySummarizer initialized:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Provider: {self.provider}")
        if self.provider == "lmstudio":
            logger.info(f"  LM Studio URL: {self.lm_studio_url}")
        else:
            logger.info(f"  Using OpenAI API")

    def _call_vlm_unified(
            self,
            prompt: str,
            timeout: int = 60
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Call VLM with unified routing to LM Studio or OpenAI.
        Returns (response_text, error_message).
        """
        system_prompt = "You are a real estate investment analyst. Respond with valid JSON only."

        # Use VLMClient if available
        if self.vlm_client:
            try:
                if self.provider == "openai":
                    text = self.vlm_client.analyze_text_sync(
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        model=self.model_name,
                        api_key=self.api_key,
                        provider="openai",
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                else:
                    text = self.vlm_client.analyze_text_sync(
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        model=self.model_name,
                        url=self.lm_studio_url,
                        provider="lmstudio",
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                return text, None
            except Exception as e:
                return None, str(e)

        # Fallback to direct requests (LM Studio only)
        return _call_vlm(prompt, self.model_name, self.lm_studio_url, timeout)

    def _collect_issues_from_photo_intel(
            self,
            photo_intel: Dict[str, Any]
    ) -> Tuple[List[IssueItem], Dict[str, int]]:
        """
        Extract all *verified* issues from photo_intel.json.
        Returns (list of IssueItem, scene_image_counts).

        IMPORTANT:
        - Only uses `verified_issues`.
        - If there are no verified_issues, we treat that as "no issues",
          even if detected_issues/issues_natural_language exist.
        """
        issues: List[IssueItem] = []
        scene_counts: Dict[str, int] = defaultdict(int)

        photos = photo_intel.get("photos", {}) or {}

        for image_name, photo_data in photos.items():
            scene = photo_data.get("scene", "unknown")
            scene_counts[scene] += 1

            # ✅ Only trust verified_issues – detection-only issues are considered "not real"
            # Fallback to issues_natural_language if verified_issues isn't populated
            verified = photo_data.get("verified_issues")
            if not verified:
                verified = photo_data.get("issues_natural_language") or []
            if not isinstance(verified, list):
                continue

            for v in verified:
                if not isinstance(v, dict):
                    continue

                desc = v.get("description", "")
                if not desc:
                    continue

                # Some of your verified_issues don't have a category; give a sane default
                cat = v.get("rough_category") or "defect"
                loc = v.get("location", "") or v.get("location_hint", "")

                issues.append(IssueItem(
                    description=desc,
                    rough_category=cat,
                    location_hint=loc,
                    source_image=image_name,
                    scene=scene,
                ))

        return issues, dict(scene_counts)

    def _group_issues_by_room(
            self,
            issues: List[IssueItem]
    ) -> Dict[str, List[IssueItem]]:
        """Group issues by room type category."""
        grouped: Dict[str, List[IssueItem]] = defaultdict(list)

        for issue in issues:
            group = SCENE_TO_GROUP.get(issue.scene, "other")
            grouped[group].append(issue)

        return dict(grouped)

    def _summarize_room_group(
            self,
            group_name: str,
            issues: List[IssueItem],
            scene_counts: Dict[str, int]
    ) -> RoomGroupSummary:
        """Generate summary for a single room group."""

        # Count images in this group
        scenes_in_group = SCENE_GROUPS.get(group_name, [group_name])
        image_count = sum(scene_counts.get(s, 0) for s in scenes_in_group)
        scenes_found = [s for s in scenes_in_group if scene_counts.get(s, 0) > 0]

        if not issues:
            return RoomGroupSummary(
                group_name=group_name,
                scenes_included=scenes_found,
                image_count=image_count,
                issue_count=0,
                summary_text="No issues identified in this area.",
                key_concerns=[],
                severity_assessment="minor",
                estimated_scope="No work needed"
            )

        logger.info(f"  Summarizing {group_name}: {len(issues)} issues from {image_count} images")

        prompt = build_room_group_prompt(group_name, issues, image_count)

        response = None
        error = None
        parsed = None

        for attempt in range(1, self.max_retries + 1):
            response, error = self._call_vlm_unified(prompt)

            if error:
                logger.warning(
                    "Room group '%s' attempt %d/%d failed with VLM error: %s",
                    group_name, attempt, self.max_retries, error,
                )
                if attempt < self.max_retries:
                    time.sleep(1.0 * attempt)
                    continue
                break  # out of retries

            parsed = _extract_json(response)
            if isinstance(parsed, dict):
                break  # success

            logger.warning(
                "Room group '%s' attempt %d/%d: JSON parse failed. Raw response (truncated): %s",
                group_name,
                attempt,
                self.max_retries,
                str(response)[:500],
            )
            if attempt < self.max_retries:
                time.sleep(1.0 * attempt)
                continue
            break  # out of retries

        if error:
            # final failure: VLM error
            logger.error(f"  VLM error for {group_name} after {self.max_retries} attempts: {error}")
            return RoomGroupSummary(
                group_name=group_name,
                scenes_included=scenes_found,
                image_count=image_count,
                issue_count=len(issues),
                summary_text=f"Analysis failed: {error}",
                key_concerns=[issue.description[:100] for issue in issues[:3]],
                severity_assessment="unknown",
                estimated_scope="Unable to assess"
            )

        if isinstance(parsed, dict):
            return RoomGroupSummary(
                group_name=group_name,
                scenes_included=scenes_found,
                image_count=image_count,
                issue_count=len(issues),
                summary_text=parsed.get("summary_text", ""),
                key_concerns=parsed.get("key_concerns", []),
                severity_assessment=parsed.get("severity_assessment", "unknown"),
                estimated_scope=parsed.get("estimated_scope", "")
            )

        # Final fallback after all retries
        logger.warning(
            "Failed to parse JSON for room group '%s' after %d attempts. Last raw response (truncated): %s",
            group_name,
            self.max_retries,
            str(response)[:1000] if response is not None else "<no response>",
        )
        return RoomGroupSummary(
            group_name=group_name,
            scenes_included=scenes_found,
            image_count=image_count,
            issue_count=len(issues),
            summary_text="Multiple issues identified requiring attention.",
            key_concerns=[issue.description[:100] for issue in issues[:3]],
            severity_assessment="moderate",
            estimated_scope="Professional assessment recommended"
        )

    def summarize_property(
            self,
            photo_intel: Dict[str, Any]
    ) -> PropertySummary:
        """
        Generate complete property summary from photo_intel.json data.
        """
        t0 = time.time()

        property_key = photo_intel.get("property_key", "unknown")
        job_id = photo_intel.get("job_id", photo_intel.get("run_id", "unknown"))
        renovation_needs = photo_intel.get("renovation_needs")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Generating Property Summary: {property_key}")
        logger.info(f"{'=' * 60}")

        # Collect all issues
        issues, scene_counts = self._collect_issues_from_photo_intel(photo_intel)
        total_images = sum(scene_counts.values())

        logger.info(f"Collected {len(issues)} issues from {total_images} images")

        # Count issues by category
        issues_by_category: Dict[str, int] = defaultdict(int)
        for issue in issues:
            issues_by_category[issue.rough_category] += 1

        # Group and summarize by room type
        grouped_issues = self._group_issues_by_room(issues)
        room_summaries: Dict[str, RoomGroupSummary] = {}

        logger.info(f"\nSummarizing by room group:")
        for group_name in SCENE_GROUPS.keys():
            group_issues = grouped_issues.get(group_name, [])
            if group_issues or any(scene_counts.get(s, 0) > 0 for s in SCENE_GROUPS[group_name]):
                room_summaries[group_name] = self._summarize_room_group(
                    group_name, group_issues, scene_counts
                )
                time.sleep(0.5)  # Rate limiting between VLM calls

        # Generate overall property summary
        logger.info(f"\nGenerating overall property summary...")

        prompt = build_property_summary_prompt(
            property_key=property_key,
            room_summaries=room_summaries,
            renovation_needs=renovation_needs,
            total_images=total_images,
            total_issues=len(issues),
            issues_by_category=dict(issues_by_category)
        )

        # Defaults in case of failure
        overall_condition = "fair"
        overall_summary = "Unable to generate comprehensive summary."
        investment_verdict = "hold"
        investment_rationale = "Analysis incomplete."
        renovation_scope = "moderate"
        renovation_priorities = []
        risk_flags = []
        deferred_maintenance = []
        summary_error = None

        response = None
        error = None
        parsed = None

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "Calling VLM for property summary (attempt %d/%d)...",
                attempt, self.max_retries
            )
            response, error = self._call_vlm_unified(prompt, timeout=90)

            if error:
                logger.warning(
                    "Property summary attempt %d/%d failed with VLM error: %s",
                    attempt, self.max_retries, error,
                )
                if attempt < self.max_retries:
                    time.sleep(1.0 * attempt)
                    continue
                summary_error = str(error)
                break

            parsed = _extract_json(response)
            if isinstance(parsed, dict):
                break  # success

            logger.warning(
                "Property summary attempt %d/%d: JSON parse failed. Raw response (truncated): %s",
                attempt,
                self.max_retries,
                str(response)[:1000],
            )
            if attempt < self.max_retries:
                time.sleep(1.0 * attempt)
                continue
            summary_error = "JSON parse error"
            break

        if isinstance(parsed, dict):
            overall_condition = parsed.get("overall_condition", overall_condition)
            overall_summary = parsed.get("overall_summary", overall_summary)
            investment_verdict = parsed.get("investment_verdict", investment_verdict)
            investment_rationale = parsed.get("investment_rationale", investment_rationale)
            renovation_scope = parsed.get("renovation_scope", renovation_scope)
            renovation_priorities = parsed.get("renovation_priorities", [])
            risk_flags = parsed.get("risk_flags", [])
            deferred_maintenance = parsed.get("deferred_maintenance", [])
        else:
            if summary_error:
                logger.error("Property summary failed after %d attempts: %s", self.max_retries, summary_error)
            else:
                logger.warning(
                    "Property summary parse failed after %d attempts with no explicit error. Last response (truncated): %s",
                    self.max_retries,
                    str(response)[:1000] if response is not None else "<no response>",
                )
                summary_error = "JSON parse error"

        processing_time = time.time() - t0

        summary = PropertySummary(
            property_key=property_key,
            job_id=job_id,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            summary_version=SUMMARY_VERSION,
            overall_condition=overall_condition,
            overall_summary=overall_summary,
            investment_verdict=investment_verdict,
            investment_rationale=investment_rationale,
            renovation_scope=renovation_scope,
            renovation_priorities=renovation_priorities,
            risk_flags=risk_flags,
            deferred_maintenance=deferred_maintenance,
            room_summaries=room_summaries,
            total_images_analyzed=total_images,
            total_issues_found=len(issues),
            issues_by_category=dict(issues_by_category),
            processing_time=processing_time,
            model_used=self.model_name,
            error=summary_error
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Summary Complete")
        logger.info(f"  Condition: {overall_condition}")
        logger.info(f"  Verdict: {investment_verdict}")
        logger.info(f"  Scope: {renovation_scope}")
        logger.info(f"  Processing time: {processing_time:.1f}s")
        logger.info(f"{'=' * 60}")

        return summary

    def summarize_from_file(
            self,
            photo_intel_path: Path,
            output_path: Optional[Path] = None
    ) -> PropertySummary:
        """Load photo_intel.json and generate summary."""

        with open(photo_intel_path, 'r', encoding='utf-8') as f:
            photo_intel = json.load(f)

        summary = self.summarize_property(photo_intel)

        # Save if output path specified
        if output_path:
            self.save_summary(summary, output_path)

        return summary

    @staticmethod
    def save_summary(summary: PropertySummary, output_path: Path) -> None:
        """Save summary to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dicts
        data = asdict(summary)

        # Convert RoomGroupSummary objects
        room_summaries_dict = {}
        for group_name, room_summary in data["room_summaries"].items():
            if isinstance(room_summary, dict):
                room_summaries_dict[group_name] = room_summary
            else:
                room_summaries_dict[group_name] = asdict(room_summary)
        data["room_summaries"] = room_summaries_dict

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate property-level summary from photo analysis"
    )
    parser.add_argument(
        "photo_intel",
        help="Path to photo_intel.json file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for summary JSON (default: property_summary.json in same dir)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LM Studio model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--lm-studio-url",
        default=LM_STUDIO_URL,
        help=f"LM Studio API URL (default: {LM_STUDIO_URL})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    photo_intel_path = Path(args.photo_intel)
    if not photo_intel_path.exists():
        logger.error(f"File not found: {photo_intel_path}")
        sys.exit(1)

    # Default output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = photo_intel_path.parent / "property_summary.json"

    # Create summarizer and run
    summarizer = PropertySummarizer(
        lm_studio_url=args.lm_studio_url,
        model_name=args.model,
        debug=args.debug
    )

    try:
        summary = summarizer.summarize_from_file(photo_intel_path, output_path)

        # Print summary to stdout as well
        print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()