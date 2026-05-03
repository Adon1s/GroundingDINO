"""
Catalog Auditor — cloud-model judge-based catalog improvement tool.

Runs dual-model analysis (local + cloud) on property images, then uses
the cloud model as a judge to diagnose pipeline failures and propose
catalog improvements.

Supports OpenAI GPT and Google Gemini as cloud providers (--cloud-provider).

Usage:
    python tools/catalog_auditor.py --property redfin_126224899
    python tools/catalog_auditor.py --property redfin_126224899 --cloud-provider gemini
    python tools/catalog_auditor.py --property redfin_126224899 --dry-run --max-images 5
    python tools/catalog_auditor.py --property redfin_126224899 --mode full-scan
    python tools/catalog_auditor.py --property redfin_126224899 --resume-latest
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or tools/
# ---------------------------------------------------------------------------
_TOOLS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLS_DIR.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pipeline_config as cfg
from vlm_client import VLMClient, create_vlm_client, get_model_configs_from_pipeline_config, get_gemini_config_from_pipeline_config
from catalog_embeddings import CatalogEmbeddingsRetriever, MatchCandidate, build_guardrails_from_catalog
from scene_classifier_passes import (
    evaluate_kind_routing,
    prioritize_resolution_candidates,
    run_pass_1a_scene_type,
    run_pass_2a,
    run_pass_2b,
    run_pass_2c,
)
from llm_json import extract_json_object

logger = logging.getLogger("catalog_auditor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ARTIFACTS_DIR = Path("C:/Users/Steven/IntelliJProjects/realtorvision/artifacts")
DEFAULT_IMAGES_BASE = Path("C:/Users/Steven/IntelliJProjects/realtorvision/public/images/properties")

# Freeform output truncation cap (chars). Now that 2b/2c structuring uses
# the cloud structuring model (not local), context limits are much less of
# a concern — we only truncate as a sanity guard against runaway outputs.
# Truncation is paragraph-aware: we cut at the last paragraph break before
# the cap to avoid splitting an observation mid-sentence.
DEFAULT_FREEFORM_CAP = 12000


def _truncate_freeform(text: str, cap: int) -> Tuple[str, bool]:
    """
    Paragraph-aware truncation. Returns (truncated_text, was_truncated).

    If text is over `cap`, cut at the last paragraph break (\\n\\n) before
    `cap`, falling back to last newline, then last sentence end, then a
    hard cut as a last resort.
    """
    if cap <= 0 or len(text) <= cap:
        return text, False

    head = text[:cap]
    # Prefer paragraph break
    cut = head.rfind("\n\n")
    if cut < cap // 2:
        cut = head.rfind("\n")
    if cut < cap // 2:
        # Sentence-end fallback
        for sep in (". ", "! ", "? "):
            i = head.rfind(sep)
            if i > cut:
                cut = i + 1
    if cut < cap // 2:
        cut = cap  # hard cut

    return text[:cut].rstrip(), True


SCENE_TO_GROUP: Dict[str, str] = {
    "kitchen": "kitchen", "pantry": "kitchen",
    "bathroom": "bathroom",
    "bedroom": "bedroom", "closet": "bedroom",
    "living_room": "living_areas", "dining_room": "living_areas",
    "home_office": "living_areas", "hallway": "living_areas", "stairway": "living_areas",
    "laundry_room": "utility", "basement": "utility", "attic": "utility",
    "garage": "utility", "hvac": "utility",
    "exterior_front": "exterior", "exterior_back": "exterior", "exterior_side": "exterior",
    "yard": "exterior", "patio": "exterior", "deck": "exterior", "balcony": "exterior",
    "driveway": "exterior", "pool": "exterior", "garden": "exterior",
    "roof": "other", "other": "other", "unknown": "other",
    "floor_plan": "other", "aerial_view": "other", "street_view": "other",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ObservationRecord:
    """A single observation with full provenance tracking."""
    description: str
    source_model_2a: str           # "local", "openai", or "gemini"
    structured_by_model: str       # always "local"
    labeled_by_model: str          # always "local"
    label: str                     # defect_or_damage, upgrade_candidate, etc.
    kind: str                      # defect or upgrade
    retrieval_status: str          # strong, weak, unmatched
    best_match_id: Optional[str] = None
    best_match_score: float = 0.0
    top_candidates: List[Dict] = field(default_factory=list)
    failure_stage: str = "uncertain"
    # For local observations loaded from artifacts:
    was_suppressed: bool = False
    suppression_reason: Optional[str] = None
    catalog_item_id_from_pipeline: Optional[str] = None


@dataclass
class ImageResult:
    """Results for a single image across both models."""
    image_path: str
    property_id: str
    photo_key: str
    scene: str
    scene_group: str

    # Local model observations (from artifacts or fresh run)
    local_observations: List[ObservationRecord] = field(default_factory=list)
    # Cloud model observations (GPT or Gemini)
    cloud_observations: List[ObservationRecord] = field(default_factory=list)

    # Judge verdicts (Phase 3)
    judge_verdicts: List[Dict] = field(default_factory=list)
    missed_issues: List[Dict] = field(default_factory=list)

    # Raw data for debugging
    local_freeform: str = ""
    cloud_freeform: str = ""
    error: Optional[str] = None

    # Provenance — when the local artifacts (issues-only mode) were generated.
    # Empty for full-scan mode since local observations come from a fresh run.
    artifact_run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Image Discovery
# ---------------------------------------------------------------------------

def discover_images(
    artifacts_dir: Path,
    images_base: Path,
    mode: str,
    max_images: int,
    property_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scan artifacts and resolve image paths.

    Returns list of dicts:
        {property_id, photo_key, image_path, artifact_path, debug_data}
    """
    candidates = []

    if not artifacts_dir.exists():
        logger.error(f"Artifacts directory not found: {artifacts_dir}")
        return []

    for prop_dir in sorted(artifacts_dir.iterdir()):
        if not prop_dir.is_dir():
            continue
        property_id = prop_dir.name

        # Property filter
        if property_filter and property_id != property_filter:
            continue

        # Find latest run directory (most recent timestamp)
        run_dirs = sorted(
            [d for d in prop_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if not run_dirs:
            continue
        run_dir = run_dirs[0]

        debug_path = run_dir / "photo_intel_debug.json"
        if not debug_path.exists():
            continue

        try:
            with open(debug_path, "r", encoding="utf-8") as f:
                debug_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {debug_path}: {e}")
            continue

        photos = debug_data.get("photos", {})
        if not isinstance(photos, dict):
            continue

        for photo_key, photo_data in photos.items():
            image_path = images_base / property_id / photo_key
            if not image_path.exists():
                continue

            # Check mode filter and count issues
            issues = photo_data.get("issues", {})
            final_issues = issues.get("final", []) if isinstance(issues, dict) else []
            issue_count = len(final_issues)

            if mode == "issues-only" and issue_count == 0:
                continue

            candidates.append({
                "property_id": property_id,
                "photo_key": photo_key,
                "image_path": str(image_path),
                "debug_data": photo_data,
                "issue_count": issue_count,
                "artifact_run_id": run_dir.name,
            })

    # Sort by issue count descending — highest-issue images first
    candidates.sort(key=lambda c: c["issue_count"], reverse=True)

    # Apply max_images cap (only when needed)
    if max_images > 0 and len(candidates) > max_images:
        # When filtering to a single property, just take top N by issue count
        if property_filter:
            candidates = candidates[:max_images]
        else:
            # Multi-property: sample across properties, preferring high-issue images
            by_prop: Dict[str, List] = {}
            for c in candidates:
                by_prop.setdefault(c["property_id"], []).append(c)

            selected = []
            prop_ids = list(by_prop.keys())
            random.shuffle(prop_ids)

            # Round-robin across properties (each property's list is already sorted by issue_count)
            idx = 0
            while len(selected) < max_images and prop_ids:
                pid = prop_ids[idx % len(prop_ids)]
                items = by_prop[pid]
                if items:
                    selected.append(items.pop(0))  # take highest-issue image first
                if not items:
                    prop_ids.remove(pid)
                    if prop_ids:
                        idx = idx % len(prop_ids)
                else:
                    idx += 1
            candidates = selected

    logger.info(f"Discovered {len(candidates)} images ({mode} mode)")
    return candidates


# ---------------------------------------------------------------------------
# Phase 1+2: Observation Collection + Catalog Matching
# ---------------------------------------------------------------------------

def _label_to_kind(label: str) -> str:
    label = (label or "").strip().lower()
    if label == "upgrade_candidate":
        return "upgrade"
    return "defect"


def _classify_match(score: float, kind: str) -> str:
    threshold = cfg.EMBEDDINGS_THRESHOLD_DEFECT if kind == "defect" else cfg.EMBEDDINGS_THRESHOLD_OPPORTUNITY
    if score >= threshold:
        return "strong"
    elif score >= threshold - 0.10:
        return "weak"
    return "unmatched"


def _run_embedding_audit(
    labeled_forward: List[Dict],
    retriever: CatalogEmbeddingsRetriever,
    scene_group: str,
    source_model: str,
) -> List[ObservationRecord]:
    """Run embeddings retrieval on a set of observations and create ObservationRecords."""
    records = []
    for obs in labeled_forward:
        desc = obs.get("description", "").strip()
        if not desc:
            continue

        label = obs.get("label", "")
        kind = obs.get("kind") or _label_to_kind(label)
        routing = evaluate_kind_routing(desc, kind)
        allowed_kinds = set(routing.expanded_kinds) if routing.expanded_kinds else ({kind} if kind else None)
        requested_topk = 10 if allowed_kinds and len(allowed_kinds) > 1 else 5

        candidates = retriever.retrieve_candidates(
            desc,
            allowed_kinds=allowed_kinds,
            allowed_groups={scene_group},
            topk=requested_topk,
        )
        candidates = prioritize_resolution_candidates(
            [asdict(candidate) for candidate in candidates],
            widened_routing=len(routing.expanded_kinds) > 1,
        )
        candidates = [
            MatchCandidate(
                item_id=c["item_id"],
                name=c["name"],
                kind=c["kind"],
                trade_bucket=c["trade_bucket"],
                severity=c["severity"],
                description=c["description"],
                support_any=tuple(c["support_any"]),
                defaultHidden=bool(c["defaultHidden"]),
                drop_if_generic=bool(c["drop_if_generic"]),
                score=float(c["score"]),
                scene_groups=tuple(c["scene_groups"]),
            )
            for c in candidates[:5]
        ]

        best_score = candidates[0].score if candidates else 0.0
        best_id = candidates[0].item_id if candidates else None
        match_status = _classify_match(best_score, kind)

        # Heuristic failure_stage
        if match_status == "strong":
            failure_stage = "none"
        elif match_status == "weak":
            failure_stage = "retrieval"  # item probably exists but embed_text weak
        else:
            failure_stage = "catalog_gap"  # may be a real gap or bad observation

        records.append(ObservationRecord(
            description=desc,
            source_model_2a=source_model,
            structured_by_model="local",
            labeled_by_model="local",
            label=label,
            kind=kind,
            retrieval_status=match_status,
            best_match_id=best_id,
            best_match_score=best_score,
            top_candidates=[
                {"item_id": c.item_id, "name": c.name, "score": round(c.score, 4)}
                for c in candidates[:3]
            ],
            failure_stage=failure_stage,
        ))
    return records


def _load_local_observations_from_artifact(
    debug_data: Dict,
    retriever: CatalogEmbeddingsRetriever,
    scene_group: str,
) -> Tuple[List[ObservationRecord], str]:
    """
    Extract local model observations from artifact debug data.

    Returns (records, scene_id).
    """
    scene_info = debug_data.get("scene", {})
    scene_id = scene_info.get("id", "other") if isinstance(scene_info, dict) else "other"

    debug = debug_data.get("debug", {})
    labeled_forward = debug.get("labeled_forward", [])
    labeled_debug = debug.get("labeled_debug", [])
    resolved_items = debug.get("resolved_items", [])

    issues = debug_data.get("issues", {})
    final_issues = issues.get("final", []) if isinstance(issues, dict) else []
    matched_issues = issues.get("matched", []) if isinstance(issues, dict) else []

    # Build set of final issue descriptions for suppression tracking
    final_descs = {i.get("description", "").strip().lower() for i in final_issues}
    matched_descs = {i.get("description", "").strip().lower() for i in matched_issues}

    # Build resolution map: description -> catalogItemId
    resolution_map: Dict[str, str] = {}
    for ri in (resolved_items or []):
        if isinstance(ri, dict) and ri.get("resolved_item_id"):
            rdesc = ri.get("observation", "").strip().lower()
            if rdesc:
                resolution_map[rdesc] = ri["resolved_item_id"]

    # 2e telemetry for suppression reasons
    telem_2e = debug_data.get("_pass_2e_telemetry", {})

    records = []
    for obs in labeled_forward:
        if not isinstance(obs, dict):
            continue
        desc = obs.get("description", "").strip()
        if not desc:
            continue

        label = obs.get("label", "")
        kind = obs.get("kind") or _label_to_kind(label)
        catalog_id = resolution_map.get(desc.lower())

        # Check if this observation survived to final
        was_suppressed = desc.lower() not in final_descs
        suppression_reason = None
        if was_suppressed and desc.lower() in matched_descs:
            suppression_reason = "policy_gated"
        elif was_suppressed:
            suppression_reason = "removed_or_deduped"

        # Run embedding audit for this observation
        routing = evaluate_kind_routing(desc, kind)
        allowed_kinds = set(routing.expanded_kinds) if routing.expanded_kinds else ({kind} if kind else None)
        requested_topk = 10 if allowed_kinds and len(allowed_kinds) > 1 else 5
        candidates = retriever.retrieve_candidates(
            desc,
            allowed_kinds=allowed_kinds,
            allowed_groups={scene_group},
            topk=requested_topk,
        )
        candidates = prioritize_resolution_candidates(
            [asdict(candidate) for candidate in candidates],
            widened_routing=len(routing.expanded_kinds) > 1,
        )
        candidates = [
            MatchCandidate(
                item_id=c["item_id"],
                name=c["name"],
                kind=c["kind"],
                trade_bucket=c["trade_bucket"],
                severity=c["severity"],
                description=c["description"],
                support_any=tuple(c["support_any"]),
                defaultHidden=bool(c["defaultHidden"]),
                drop_if_generic=bool(c["drop_if_generic"]),
                score=float(c["score"]),
                scene_groups=tuple(c["scene_groups"]),
            )
            for c in candidates[:5]
        ]

        best_score = candidates[0].score if candidates else 0.0
        best_id = candidates[0].item_id if candidates else None
        match_status = _classify_match(best_score, kind)

        if match_status == "strong":
            failure_stage = "false_suppression" if was_suppressed else "none"
        elif match_status == "weak":
            failure_stage = "retrieval"
        else:
            failure_stage = "catalog_gap"

        records.append(ObservationRecord(
            description=desc,
            source_model_2a="local",
            structured_by_model="local",
            labeled_by_model="local",
            label=label,
            kind=kind,
            retrieval_status=match_status,
            best_match_id=catalog_id or best_id,
            best_match_score=best_score,
            top_candidates=[
                {"item_id": c.item_id, "name": c.name, "score": round(c.score, 4)}
                for c in candidates[:3]
            ],
            failure_stage=failure_stage,
            was_suppressed=was_suppressed,
            suppression_reason=suppression_reason,
            catalog_item_id_from_pipeline=catalog_id,
        ))

    return records, scene_id


async def process_single_image(
    image_info: Dict,
    vlm_client: VLMClient,
    retriever: CatalogEmbeddingsRetriever,
    local_config: Dict,
    cloud_config: Dict,
    structuring_config: Dict,
    mode: str,
) -> ImageResult:
    """Process a single image: collect observations from both models + embedding audit."""
    image_path = Path(image_info["image_path"])
    property_id = image_info["property_id"]
    photo_key = image_info["photo_key"]
    debug_data = image_info["debug_data"]

    result = ImageResult(
        image_path=str(image_path),
        property_id=property_id,
        photo_key=photo_key,
        scene="other",
        scene_group="other",
        artifact_run_id=image_info.get("artifact_run_id"),
    )

    try:
        # ── Load local model results from artifacts ──────────────────────
        local_records, scene_id = _load_local_observations_from_artifact(
            debug_data, retriever,
            SCENE_TO_GROUP.get(
                debug_data.get("scene", {}).get("id", "other") if isinstance(debug_data.get("scene"), dict) else "other",
                "other"
            ),
        )
        result.scene = scene_id
        result.scene_group = SCENE_TO_GROUP.get(scene_id, "other")
        result.local_observations = local_records

        # ── Run cloud model Pass 2a on the image ────────────────────────
        cloud_provider = cloud_config.get("provider", "openai")
        cloud_model = cloud_config.get("model", "unknown")
        cloud_2a = await run_pass_2a(image_path, vlm_client, cloud_config)
        result.cloud_freeform = cloud_2a.observations_freeform

        cloud_freeform = cloud_2a.observations_freeform.strip()
        if cloud_freeform:
            cloud_freeform, was_truncated = _truncate_freeform(
                cloud_freeform, DEFAULT_FREEFORM_CAP
            )
            if was_truncated:
                logger.warning(
                    f"  {cloud_provider}/{cloud_model} freeform output truncated to "
                    f"{len(cloud_freeform)} chars (paragraph-aware) for {photo_key}"
                )

            # Structure cloud observations through the structuring model
            # (cloud-grade by default — gpt-5.4-mini — to avoid local-model
            # mangling cloud freeform during 2b/2c).
            cloud_2b = await run_pass_2b(vlm_client, structuring_config, cloud_freeform)
            if cloud_2b.observations:
                cloud_2c = await run_pass_2c(vlm_client, structuring_config, cloud_2b.observations, scene_id)
                # Run embedding audit on cloud model's forward observations
                cloud_records = _run_embedding_audit(
                    cloud_2c.labeled_forward, retriever, result.scene_group, cloud_provider
                )
                result.cloud_observations = cloud_records

        logger.info(
            f"  {photo_key}: scene={scene_id}, "
            f"local_obs={len(result.local_observations)}, "
            f"cloud_obs({cloud_provider}/{cloud_model})={len(result.cloud_observations)}"
        )

    except Exception as e:
        logger.error(f"Error processing {photo_key}: {e}")
        result.error = str(e)

    return result


async def process_single_image_fullscan(
    image_info: Dict,
    vlm_client: VLMClient,
    retriever: CatalogEmbeddingsRetriever,
    local_config: Dict,
    cloud_config: Dict,
    structuring_config: Dict,
) -> ImageResult:
    """Full-scan mode: run both models from scratch."""
    image_path = Path(image_info["image_path"])
    property_id = image_info["property_id"]
    photo_key = image_info["photo_key"]

    result = ImageResult(
        image_path=str(image_path),
        property_id=property_id,
        photo_key=photo_key,
        scene="other",
        scene_group="other",
    )

    try:
        cloud_provider = cloud_config.get("provider", "openai")
        cloud_model = cloud_config.get("model", "unknown")

        # Pass 1a: Scene classification
        scene_result = await run_pass_1a_scene_type(image_path, vlm_client, local_config)
        result.scene = scene_result.scene
        result.scene_group = SCENE_TO_GROUP.get(scene_result.scene, "other")

        # Pass 2a: Run both models concurrently
        local_2a, cloud_2a = await asyncio.gather(
            run_pass_2a(image_path, vlm_client, local_config),
            run_pass_2a(image_path, vlm_client, cloud_config),
        )
        result.local_freeform = local_2a.observations_freeform
        result.cloud_freeform = cloud_2a.observations_freeform

        local_freeform_capped, local_truncated = _truncate_freeform(
            local_2a.observations_freeform, DEFAULT_FREEFORM_CAP
        )
        cloud_freeform_capped, cloud_truncated = _truncate_freeform(
            cloud_2a.observations_freeform, DEFAULT_FREEFORM_CAP
        )
        if local_truncated:
            logger.warning(f"  local freeform truncated to {len(local_freeform_capped)} chars for {photo_key}")
        if cloud_truncated:
            logger.warning(f"  cloud freeform truncated to {len(cloud_freeform_capped)} chars for {photo_key}")

        # Pass 2b+2c for both — use the structuring_config (cloud-grade)
        # so the local model doesn't mangle either side's freeform output.
        async def _structure_and_audit(freeform: str, source: str) -> List[ObservationRecord]:
            if not freeform.strip():
                return []
            r2b = await run_pass_2b(vlm_client, structuring_config, freeform)
            if not r2b.observations:
                return []
            r2c = await run_pass_2c(vlm_client, structuring_config, r2b.observations, result.scene)
            return _run_embedding_audit(r2c.labeled_forward, retriever, result.scene_group, source)

        local_records, cloud_records = await asyncio.gather(
            _structure_and_audit(local_freeform_capped, "local"),
            _structure_and_audit(cloud_freeform_capped, cloud_provider),
        )
        result.local_observations = local_records
        result.cloud_observations = cloud_records

        logger.info(
            f"  {photo_key}: scene={result.scene}, "
            f"local_obs={len(local_records)}, cloud_obs({cloud_provider}/{cloud_model})={len(cloud_records)}"
        )

    except Exception as e:
        logger.error(f"Error processing {photo_key}: {e}")
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Phase 3: Cloud Model Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert real estate photo analyst and issue catalog quality auditor.
Your job is to diagnose failures in an automated defect/upgrade detection pipeline.

You will see an image, a list of candidate observations (each with an opaque obs_id), and the catalog items they matched to.
The observations come from two different detection runs, but you are NOT told which observation came from which run — judge each on its own merits.

═══════════════════════════════════════════════════════════════════════
TWO KINDS OF JUDGMENT — DO NOT CONFUSE THEM
═══════════════════════════════════════════════════════════════════════

(A) IMAGE-GROUNDED judgments — look at the image:
    - is_visible_issue: is what THIS observation describes actually visible?
    - missed_issues: what real issues did NO observation catch?

(B) TEXT-ALIGNED judgments — compare TEXT to TEXT, do NOT re-scan the image:
    - is_matched_catalog_item_correct: does the matched catalog item's
      embed_text reasonably cover what THIS observation says?
    - suggested_embed_text_fix: if the match was wrong for THIS observation,
      what wording would have helped THIS observation match?
    - best_existing_catalog_item_id_if_any, needs_new_catalog_entry,
      suggested_catalog_improvement_type, failure_stage_override="retrieval":
      all derive from the text-to-text comparison above.

The most common failure mode in past audits: the judge looks at the image,
notices a different salient issue (a fan, a floor, an island), and then
judges the catalog match against THAT instead of against the observation
text. This corrupts the verdict. DO NOT do this.

═══════════════════════════════════════════════════════════════════════
LOCK-IN RULE
═══════════════════════════════════════════════════════════════════════

For each observation you will first write `observation_paraphrase` —
a one-sentence restatement of what THAT observation claims. Once written,
EVERY other field for that obs_id must be evaluated against that paraphrase.
Do not pivot mid-verdict to a different image element. The matched catalog
item does not need to be the best item for the photo overall — only a
reasonable fit for the observation as paraphrased.

═══════════════════════════════════════════════════════════════════════
ANTI-PATTERN EXAMPLES (do NOT do this)
═══════════════════════════════════════════════════════════════════════

Example 1 — drift to a different element:
  Observation: "The kitchen has light oak wood cabinets described as dated."
  Matched: outdated_kitchen_finishes
  WRONG verdict: is_matched_catalog_item_correct=false,
    notes="the island looks like a basic cabinet block, not heavy clutter"
  → The judge re-noticed the island. The match was correct for the
    observation (oak cabinets ARE outdated kitchen finishes).
  CORRECT verdict: observation_paraphrase="cabinets are dated oak",
    matched_item_intent="catalog item for outdated kitchen finishes",
    is_matched_catalog_item_correct=true.

Example 2 — drift to a different axis:
  Observation: "The kitchen has older brown tile flooring."
  Matched: outdated_kitchen_finishes
  WRONG verdict: notes="the small ceiling fixture is visible. The match
    should be dated_lighting_fixtures"
  → The judge swapped the floor observation for a lighting one. The match
    is reasonable for "older flooring" framed as a kitchen finish.
  CORRECT verdict: observation_paraphrase="older brown tile floor",
    matched_item_intent="outdated kitchen finishes (broad)",
    is_matched_catalog_item_correct=true (or specify a tighter
    floor-specific item in best_existing_catalog_item_id_if_any if one
    exists in the listed catalog items, but do NOT pivot to lighting).

═══════════════════════════════════════════════════════════════════════
PIPELINE STAGE DIAGNOSIS
═══════════════════════════════════════════════════════════════════════

For each observation, diagnose WHICH STEP in the pipeline failed (if any):
- observation: the run hallucinated or described something not visible
- structuring: the freeform text was reasonable but got mangled during JSON structuring
- labeling: the observation was real but got the wrong label (defect vs upgrade vs other)
- retrieval: the observation is valid AND a matching catalog item exists, but embedding retrieval missed it
- catalog_gap: the observation is valid but NO existing catalog item covers this issue
- false_suppression: the observation was valid and matched, but pipeline rules incorrectly suppressed it

If is_visible_issue is false, failure_stage_override MUST be "observation".
Use failure_stage_override="retrieval" ONLY when (a) is_visible_issue is true
AND (b) is_matched_catalog_item_correct is false AND (c) you can name a
specific best_existing_catalog_item_id_if_any from the listed catalog items.
A judge preference for a different image element is NOT a retrieval failure.

Be precise. Do not jump to "needs new catalog entry" when the real problem is bad embed_text on an existing item."""


def _build_judge_prompt(
    image_result: ImageResult,
    catalog: Dict,
) -> Tuple[str, Dict[str, Tuple[str, ObservationRecord]]]:
    """
    Build the user prompt for the judge, with observations blinded under opaque IDs.

    Returns (prompt, id_map) where id_map maps obs_id -> (source, ObservationRecord).
    """
    catalog_items_by_id = {}
    for item in catalog.get("items", []):
        catalog_items_by_id[item["id"]] = item

    lines = [f"## Scene: {image_result.scene} ({image_result.scene_group})"]

    # Build a single shuffled list of (obs_id, source, obs).
    # The judge does NOT see "source" — only an opaque id. We restore source post-parse.
    tagged: List[Tuple[str, ObservationRecord]] = []
    for obs in image_result.local_observations:
        tagged.append(("local", obs))
    for obs in image_result.cloud_observations:
        tagged.append(("cloud", obs))

    # Deterministic shuffle keyed on photo identity so a re-run produces the same order
    # (helps debugging) but the order conveys no source signal to the judge.
    rng = random.Random(f"{image_result.property_id}/{image_result.photo_key}")
    rng.shuffle(tagged)

    id_map: Dict[str, Tuple[str, ObservationRecord]] = {}
    for idx, (source, obs) in enumerate(tagged, 1):
        obs_id = f"obs_{idx:02d}"
        id_map[obs_id] = (source, obs)

    lines.append("\n## Observations:")
    if not tagged:
        lines.append("(none detected by any run)")
    else:
        for obs_id, (source, obs) in id_map.items():
            status = f"[{obs.retrieval_status}]"
            match_info = (
                f" → matched: {obs.best_match_id} (score={obs.best_match_score:.3f})"
                if obs.best_match_id else " → no match"
            )
            supp = ""
            if obs.was_suppressed:
                supp = f" [SUPPRESSED: {obs.suppression_reason}]"
            lines.append(
                f"  {obs_id}. {status} ({obs.label}/{obs.kind}) {obs.description}{match_info}{supp}"
            )

    # Relevant catalog snippets
    mentioned_ids = set()
    for _, obs in tagged:
        if obs.best_match_id:
            mentioned_ids.add(obs.best_match_id)
        for cand in obs.top_candidates:
            mentioned_ids.add(cand["item_id"])

    if mentioned_ids:
        lines.append("\n## Relevant Catalog Items:")
        for cid in sorted(mentioned_ids):
            item = catalog_items_by_id.get(cid)
            if item:
                lines.append(f"  - {cid}: \"{item.get('name', '')}\" (kind={item.get('kind', '')}, trade={item.get('trade_bucket', '')})")
                lines.append(f"    embed_text: \"{item.get('embed_text', '')}\"")
                support = item.get("support_any", [])
                require = item.get("require_any", [])
                deny = item.get("deny_any", [])
                if support:
                    lines.append(f"    support_any: {support}")
                if require:
                    lines.append(f"    require_any: {require}")
                if deny:
                    lines.append(f"    deny_any: {deny}")

    lines.append("""
## Your Task:
For each observation, fill the schema below. Reference observations by their obs_id.
Use the image ONLY for is_visible_issue and missed_issues. All catalog-match
fields are TEXT-to-TEXT comparisons against the observation as you paraphrased
it — NOT against whatever else you happen to notice in the image.

Per-field rules:
- observation_paraphrase: Restate THIS observation in your own words (≤20 words).
  Summarize the OBSERVATION TEXT, not the image. This is your anchor; every
  later field must be evaluated against it.
- matched_item_intent: One sentence on what the matched catalog item is meant
  to capture, based on its embed_text shown above. Null if no match.
- is_matched_catalog_item_correct: Does matched_item_intent reasonably cover
  observation_paraphrase? TEXT-to-TEXT only. The matched item does not need
  to be the best item for the photo overall — only a reasonable fit for THIS
  observation. Null if no match exists.
- best_existing_catalog_item_id_if_any: If a better item exists IN THE LISTED
  CATALOG ITEMS for THIS observation, name it. Null otherwise. Do NOT invent ids.
- suggested_embed_text_fix: Only fill if the matched item's embed_text failed
  to retrieve for THIS observation. Suggest text that would have helped THIS
  observation match. Do NOT suggest keywords for a different image element.
  Null when not applicable.
- failure_stage_override="retrieval": Use ONLY when is_visible_issue is true
  AND is_matched_catalog_item_correct is false AND you named a specific
  best_existing_catalog_item_id_if_any. Drift cases (you preferred a different
  image element) MUST NOT be coded as retrieval failures.
- notes: <100 words. MUST reference observation_paraphrase, not a different
  image element you noticed.

Return JSON:
{
  "observation_verdicts": [
    {
      "obs_id": "obs_01",
      "is_visible_issue": true/false,
      "is_observation_reasonable": true/false,
      "is_structured_label_reasonable": true/false,
      "observation_paraphrase": "1 sentence restating what THIS observation claims",
      "matched_item_intent": "1 sentence on what the matched catalog item captures, or null",
      "is_matched_catalog_item_correct": true/false/null,
      "best_existing_catalog_item_id_if_any": "item_id" or null,
      "needs_new_catalog_entry": true/false,
      "suggested_catalog_improvement_type": "embed_text"/"keywords"/"new_entry"/"none",
      "suggested_embed_text_fix": "..." or null,
      "failure_stage_override": "observation"/"structuring"/"labeling"/"retrieval"/"catalog_gap"/"false_suppression" or null,
      "notes": "brief reasoning grounded in observation_paraphrase, <100 words"
    }
  ],
  "missed_issues": [
    {
      "description": "issue visible in photo that no observation above caught",
      "kind": "defect" or "upgrade",
      "suggested_catalog_entry_name": "proposed name",
      "category": "cosmetic/moisture/structure/exterior/safety/systems/opportunity",
      "trade_bucket": "..."
    }
  ]
}""")

    return "\n".join(lines), id_map


async def run_judge(
    image_result: ImageResult,
    vlm_client: VLMClient,
    cloud_config: Dict,
    catalog: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    """Run the cloud model judge on a single image. Returns (verdicts, missed_issues)."""
    user_prompt, id_map = _build_judge_prompt(image_result, catalog)

    # Higher token limit for the judge response — complex images with many
    # observations can produce long verdicts. With blinding the schema is
    # also slightly larger so we give it more headroom.
    judge_config = {**cloud_config, "max_tokens": 8000}

    try:
        response = await vlm_client.analyze_image(
            image_path=Path(image_result.image_path),
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **judge_config,
        )

        try:
            result = extract_json_object(response or "")
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(
                f"  Judge response unparseable for {image_result.photo_key} "
                f"({len(response or '')} chars): {e}"
            )
            return [], []

        verdicts_raw = result.get("observation_verdicts", [])
        missed = result.get("missed_issues", [])

        if not isinstance(verdicts_raw, list):
            verdicts_raw = []
        if not isinstance(missed, list):
            missed = []

        # Restore source + description from the id_map so downstream consumers
        # (synthesis, reports) see them. Verdicts referencing unknown obs_ids
        # are dropped with a warning — judge hallucinated them.
        verdicts: List[Dict] = []
        unknown_ids = []
        for v in verdicts_raw:
            if not isinstance(v, dict):
                continue
            obs_id = v.get("obs_id")
            mapping = id_map.get(obs_id) if obs_id else None
            if mapping is None:
                unknown_ids.append(obs_id)
                continue
            source, obs = mapping
            v["source"] = source
            v["description"] = obs.description
            # If is_visible_issue is False and no override, force "observation"
            # so hallucinations land in the failure breakdown.
            if v.get("is_visible_issue") is False and not v.get("failure_stage_override"):
                v["failure_stage_override"] = "observation"
            verdicts.append(v)

        if unknown_ids:
            logger.warning(
                f"  Judge returned {len(unknown_ids)} verdict(s) with unknown obs_id "
                f"for {image_result.photo_key}: {unknown_ids[:5]}"
            )

        return verdicts, missed

    except Exception as e:
        err_str = str(e)
        # Propagate quota/billing errors so the caller can abort early
        if "insufficient_quota" in err_str or "billing" in err_str.lower() or "quota" in err_str.lower() or "ResourceExhausted" in err_str:
            raise
        logger.error(f"Judge error for {image_result.photo_key}: {e}")
        return [], []


# ---------------------------------------------------------------------------
# Phase 4: Synthesis
# ---------------------------------------------------------------------------

def _synthesize_report(
    image_results: List[ImageResult],
    catalog: Dict,
    retriever: CatalogEmbeddingsRetriever,
    models_info: Dict,
) -> Dict:
    """Aggregate all results into the final report."""
    import numpy as np

    # ── Summary stats ────────────────────────────────────────────────────
    total_local = sum(len(r.local_observations) for r in image_results)
    total_cloud = sum(len(r.cloud_observations) for r in image_results)
    total_missed = sum(len(r.missed_issues) for r in image_results)

    all_obs = []
    for r in image_results:
        all_obs.extend(r.local_observations)
        all_obs.extend(r.cloud_observations)

    strong = sum(1 for o in all_obs if o.retrieval_status == "strong")
    weak = sum(1 for o in all_obs if o.retrieval_status == "weak")
    unmatched = sum(1 for o in all_obs if o.retrieval_status == "unmatched")

    # ── Failure stage breakdown ──────────────────────────────────────────
    # Use judge overrides if available, otherwise heuristic
    stage_counts: Dict[str, List[Dict]] = {
        "observation_failures": [],
        "structuring_failures": [],
        "labeling_failures": [],
        "retrieval_failures": [],
        "catalog_gaps": [],
        "false_suppressions": [],
    }

    stage_map = {
        "observation": "observation_failures",
        "structuring": "structuring_failures",
        "labeling": "labeling_failures",
        "retrieval": "retrieval_failures",
        "catalog_gap": "catalog_gaps",
        "false_suppression": "false_suppressions",
    }

    all_verdicts = []
    for r in image_results:
        for v in r.judge_verdicts:
            all_verdicts.append(v)
            override = v.get("failure_stage_override")
            if override and override in stage_map:
                stage_counts[stage_map[override]].append({
                    "description": v.get("description", ""),
                    "source": v.get("source", ""),
                    "notes": v.get("notes", ""),
                    "image": r.photo_key,
                    "property": r.property_id,
                })

    # Also count heuristic stages for observations without judge verdicts
    judged_descs = {(v.get("source", ""), v.get("description", "")) for v in all_verdicts}
    for r in image_results:
        for obs in r.local_observations + r.cloud_observations:
            key = (obs.source_model_2a, obs.description)
            if key not in judged_descs and obs.failure_stage in stage_map:
                stage_counts[stage_map[obs.failure_stage]].append({
                    "description": obs.description,
                    "source": obs.source_model_2a,
                    "notes": f"heuristic (no judge verdict)",
                    "image": r.photo_key,
                    "property": r.property_id,
                })

    failure_breakdown = {}
    for stage_key, examples in stage_counts.items():
        failure_breakdown[stage_key] = {
            "count": len(examples),
            "examples": examples[:10],  # cap examples
        }

    # ── Catalog entry proposals ──────────────────────────────────────────
    # Collect unmatched observations where judge says needs_new_catalog_entry
    new_entry_candidates = []
    improve_existing_candidates = []

    for r in image_results:
        for v in r.judge_verdicts:
            if not v.get("is_visible_issue"):
                continue
            imp_type = v.get("suggested_catalog_improvement_type", "none")
            if imp_type == "new_entry" and v.get("needs_new_catalog_entry"):
                new_entry_candidates.append({
                    "description": v.get("description", ""),
                    "source": v.get("source", ""),
                    "property_id": r.property_id,
                    "scene": r.scene,
                    "scene_group": r.scene_group,
                    "notes": v.get("notes", ""),
                })
            elif imp_type in ("embed_text", "keywords"):
                improve_existing_candidates.append({
                    "description": v.get("description", ""),
                    "catalog_id": v.get("best_existing_catalog_item_id_if_any"),
                    "improvement_type": imp_type,
                    "suggested_fix": v.get("suggested_embed_text_fix"),
                    "source": v.get("source", ""),
                    "property_id": r.property_id,
                    "notes": v.get("notes", ""),
                })

    # Cluster new_entry_candidates by semantic similarity
    high_confidence = []
    candidate_entries = []

    # Promotion thresholds for clustered new-entry candidates.
    # The previous heuristic (>=2 props OR >=2 scenes) was too permissive:
    # any systematic judge bias trivially crosses 2 properties given a small
    # sample. We require evidence breadth across BOTH dimensions.
    HIGH_CONF_MIN_EVIDENCE = 3
    HIGH_CONF_MIN_PROPERTIES = 3
    CLUSTER_SIM_THRESHOLD = 0.72

    if new_entry_candidates and len(new_entry_candidates) >= 2:
        descs = [c["description"] for c in new_entry_candidates]
        embeddings = retriever._encode_queries(descs)

        n = len(descs)
        sim = embeddings @ embeddings.T
        visited = [False] * n
        clusters: List[List[int]] = []

        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j] and sim[i, j] > CLUSTER_SIM_THRESHOLD:
                    cluster.append(j)
                    visited[j] = True
            clusters.append(cluster)

        for cluster_idxs in clusters:
            members = [new_entry_candidates[i] for i in cluster_idxs]
            distinct_props = len(set(m["property_id"] for m in members))
            distinct_scenes = len(set(m["scene_group"] for m in members))

            entry = {
                "proposed_id": _slugify(members[0]["description"][:60]),
                "description": members[0]["description"],
                "evidence_count": len(members),
                "distinct_properties": distinct_props,
                "distinct_scenes": distinct_scenes,
                "evidence_observations": [m["description"] for m in members],
                "evidence_images": [f"{m['property_id']}" for m in members],
                "notes": members[0].get("notes", ""),
            }

            if (distinct_props >= HIGH_CONF_MIN_PROPERTIES
                    and len(members) >= HIGH_CONF_MIN_EVIDENCE):
                high_confidence.append(entry)
            else:
                entry["reason_for_candidate_tier"] = (
                    f"{len(members)} evidence, {distinct_props} property(s), "
                    f"{distinct_scenes} scene(s) "
                    f"(need >={HIGH_CONF_MIN_EVIDENCE} evidence and "
                    f">={HIGH_CONF_MIN_PROPERTIES} properties for high confidence)"
                )
                candidate_entries.append(entry)
    elif new_entry_candidates:
        # Single candidates
        for c in new_entry_candidates:
            candidate_entries.append({
                "proposed_id": _slugify(c["description"][:60]),
                "description": c["description"],
                "evidence_count": 1,
                "distinct_properties": 1,
                "distinct_scenes": 1,
                "evidence_observations": [c["description"]],
                "evidence_images": [c["property_id"]],
                "reason_for_candidate_tier": "Single occurrence",
                "notes": c.get("notes", ""),
            })

    # ── Proposed improvements to existing entries ────────────────────────
    improvements_by_id: Dict[str, List[Dict]] = {}
    for c in improve_existing_candidates:
        cid = c.get("catalog_id")
        if cid:
            improvements_by_id.setdefault(cid, []).append(c)

    proposed_improvements = []
    catalog_items_by_id = {item["id"]: item for item in catalog.get("items", [])}
    for cid, entries in improvements_by_id.items():
        item = catalog_items_by_id.get(cid, {})
        proposed_improvements.append({
            "catalog_id": cid,
            "improvement_type": entries[0]["improvement_type"],
            "current_embed_text": item.get("embed_text", ""),
            "current_keywords": item.get("support_any", []) + item.get("require_any", []),
            "suggested_fixes": [e.get("suggested_fix") for e in entries if e.get("suggested_fix")],
            "evidence_observations": [e["description"] for e in entries],
            "reason": entries[0].get("notes", ""),
        })

    # ── False positive patterns ──────────────────────────────────────────
    false_positives = []
    fp_descs: Dict[str, int] = {}
    for v in all_verdicts:
        if v.get("is_visible_issue") is False:
            desc = v.get("description", "")
            fp_descs[desc] = fp_descs.get(desc, 0) + 1

    for desc, count in sorted(fp_descs.items(), key=lambda x: -x[1]):
        if count >= 2:
            false_positives.append({
                "pattern": desc,
                "frequency": count,
                "recommendation": "Consider adding to Pass 2e deny phrases",
            })

    # ── Missed issue patterns ────────────────────────────────────────────
    missed_patterns: Dict[str, Dict] = {}
    for r in image_results:
        for m in r.missed_issues:
            desc = m.get("description", "")
            if desc not in missed_patterns:
                missed_patterns[desc] = {**m, "frequency": 0, "properties": []}
            missed_patterns[desc]["frequency"] += 1
            missed_patterns[desc]["properties"].append(r.property_id)

    missed_list = sorted(missed_patterns.values(), key=lambda x: -x["frequency"])

    # ── True/false positive counts from judge ────────────────────────────
    true_pos = sum(1 for v in all_verdicts if v.get("is_visible_issue") is True)
    false_pos = sum(1 for v in all_verdicts if v.get("is_visible_issue") is False)

    # ── Build report ─────────────────────────────────────────────────────
    per_image = []
    for r in image_results:
        per_image.append({
            "image_path": r.image_path,
            "property_id": r.property_id,
            "photo_key": r.photo_key,
            "scene": r.scene,
            "scene_group": r.scene_group,
            "artifact_run_id": r.artifact_run_id,
            "local_observations": [asdict(o) for o in r.local_observations],
            "cloud_observations": [asdict(o) for o in r.cloud_observations],
            "judge_verdicts": r.judge_verdicts,
            "missed_issues": r.missed_issues,
            "error": r.error,
        })

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "images_processed": len(image_results),
            "models": models_info,
            "thresholds": {
                "defect": cfg.EMBEDDINGS_THRESHOLD_DEFECT,
                "upgrade": cfg.EMBEDDINGS_THRESHOLD_OPPORTUNITY,
            },
        },
        "summary": {
            "total_local_obs": total_local,
            "total_cloud_obs": total_cloud,
            "true_positives": true_pos,
            "false_positives": false_pos,
            "missed_issues": total_missed,
            "strong_matches": strong,
            "weak_matches": weak,
            "unmatched": unmatched,
            "high_confidence_new_entries": len(high_confidence),
            "candidate_new_entries": len(candidate_entries),
            "proposed_improvements": len(proposed_improvements),
        },
        "failure_stage_breakdown": failure_breakdown,
        "per_image": per_image,
        "high_confidence_new_entries": high_confidence,
        "candidate_new_entries": candidate_entries,
        "proposed_improvements": proposed_improvements,
        "false_positive_patterns": false_positives,
        "missed_issue_patterns": missed_list[:20],
    }


def _slugify(text: str) -> str:
    """Convert text to a slug suitable for a catalog ID."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s_]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text[:50]


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

_AUDIT_OUTPUT_PREFIX = "catalog_audit"
_CHECKPOINT_SUFFIX = ".checkpoint.json"


def _sanitize_filename_token(value: str) -> str:
    """Return a filesystem-safe token derived from a property id."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    token = re.sub(r"_+", "_", token).strip("._-")
    if not token:
        raise ValueError("Cannot derive output filename because property_id is empty.")
    return token


def _derive_single_property_id(images: List[Dict[str, Any]]) -> str:
    """Return the one property_id represented by discovered images."""
    property_ids = sorted({
        str(img.get("property_id", "")).strip()
        for img in images
        if str(img.get("property_id", "")).strip()
    })

    if not property_ids:
        raise ValueError("Cannot derive output filename because no discovered image has property_id.")
    if len(property_ids) > 1:
        shown = ", ".join(property_ids[:10])
        extra = "" if len(property_ids) <= 10 else f", ... ({len(property_ids)} total)"
        raise ValueError(
            "Catalog auditor auto-naming requires exactly one property per run. "
            f"Found: {shown}{extra}. Re-run with --property <property_id>."
        )
    return property_ids[0]


def _audit_version_from_name(filename: str, property_id: str) -> Optional[int]:
    safe_property_id = _sanitize_filename_token(property_id)
    pattern = rf"^{re.escape(_AUDIT_OUTPUT_PREFIX)}_{re.escape(safe_property_id)}_v(?P<version>\d+)(?:_|\.|$)"
    match = re.match(pattern, filename)
    if not match:
        return None
    try:
        return int(match.group("version"))
    except ValueError:
        return None


def _next_audit_output_path(output_dir: Path, property_id: str) -> Tuple[Path, int]:
    """Create the next versioned output path for a property's audit report."""
    output_dir = Path(output_dir)
    safe_property_id = _sanitize_filename_token(property_id)
    highest_version = 0

    for candidate in output_dir.glob(f"{_AUDIT_OUTPUT_PREFIX}_{safe_property_id}_v*.json"):
        version = _audit_version_from_name(candidate.name, property_id)
        if version is not None:
            highest_version = max(highest_version, version)

    next_version = highest_version + 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_AUDIT_OUTPUT_PREFIX}_{safe_property_id}_v{next_version:03d}_{ts}.json"
    return output_dir / filename, next_version


def _latest_checkpoint_output_path(output_dir: Path, property_id: str) -> Optional[Path]:
    """Return the report path whose checkpoint is newest for this property."""
    output_dir = Path(output_dir)
    safe_property_id = _sanitize_filename_token(property_id)
    checkpoints = []

    for checkpoint in output_dir.glob(f"{_AUDIT_OUTPUT_PREFIX}_{safe_property_id}_v*{_CHECKPOINT_SUFFIX}"):
        version = _audit_version_from_name(checkpoint.name, property_id)
        if version is None:
            continue
        try:
            mtime = checkpoint.stat().st_mtime
        except OSError:
            continue
        checkpoints.append((mtime, version, checkpoint))

    if not checkpoints:
        return None

    _, _, latest_checkpoint = max(checkpoints, key=lambda item: (item[0], item[1]))
    report_name = latest_checkpoint.name[:-len(_CHECKPOINT_SUFFIX)] + ".json"
    return latest_checkpoint.with_name(report_name)


def _looks_like_output_file_path(value: str) -> bool:
    return Path(value).suffix.lower() in {".json", ".jsonl", ".txt", ".csv"}


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    output_alias = getattr(args, "output", None)
    output_dir = getattr(args, "output_dir", ".") or "."
    output_dir_is_default = str(output_dir) in ("", ".")

    if output_alias and not output_dir_is_default:
        raise ValueError("Use either --output-dir or the deprecated --output directory alias, not both.")

    chosen = output_alias or output_dir
    label = "--output" if output_alias else "--output-dir"
    if _looks_like_output_file_path(chosen):
        raise ValueError(
            f"{label} now expects an output directory, not a report filename: {chosen}. "
            "The catalog auditor derives report filenames from property_id automatically."
        )

    path = Path(chosen).expanduser()
    if path.exists() and not path.is_dir():
        raise ValueError(f"{label} must point to a directory, but this path is a file: {path}")
    return path


def _resolve_output_path(args: argparse.Namespace, images: List[Dict[str, Any]]) -> Tuple[Path, int]:
    """Resolve the truthful report path after image discovery."""
    property_id = _derive_single_property_id(images)
    output_dir = _resolve_output_dir(args)

    if getattr(args, "resume_latest", False):
        output_path = _latest_checkpoint_output_path(output_dir, property_id)
        if output_path is None:
            raise ValueError(
                f"--resume-latest was set, but no checkpoint was found for property "
                f"{property_id!r} in {output_dir}."
            )
        version = _audit_version_from_name(output_path.name, property_id) or 0
        return output_path, version

    return _next_audit_output_path(output_dir, property_id)


# ---------------------------------------------------------------------------
# Checkpoint support
# ---------------------------------------------------------------------------

def _checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(".checkpoint.json")


def _serialize_image_result(r: ImageResult) -> Dict:
    """Serialize a full ImageResult to a JSON-safe dict."""
    return {
        "image_path": r.image_path,
        "property_id": r.property_id,
        "photo_key": r.photo_key,
        "scene": r.scene,
        "scene_group": r.scene_group,
        "local_observations": [asdict(o) for o in r.local_observations],
        "cloud_observations": [asdict(o) for o in r.cloud_observations],
        "judge_verdicts": r.judge_verdicts,
        "missed_issues": r.missed_issues,
        "local_freeform": r.local_freeform,
        "cloud_freeform": r.cloud_freeform,
        "error": r.error,
        "artifact_run_id": r.artifact_run_id,
    }


def _deserialize_image_result(data: Dict) -> ImageResult:
    """Reconstruct an ImageResult (and nested ObservationRecords) from a dict."""
    def _to_obs(d: Dict) -> ObservationRecord:
        # Filter to known fields so a schema change in the dataclass doesn't crash on stale checkpoints.
        known = {f for f in ObservationRecord.__dataclass_fields__}
        return ObservationRecord(**{k: v for k, v in d.items() if k in known})

    return ImageResult(
        image_path=data.get("image_path", ""),
        property_id=data.get("property_id", ""),
        photo_key=data.get("photo_key", ""),
        scene=data.get("scene", "other"),
        scene_group=data.get("scene_group", "other"),
        local_observations=[_to_obs(o) for o in data.get("local_observations", [])],
        cloud_observations=[_to_obs(o) for o in data.get("cloud_observations", [])],
        judge_verdicts=data.get("judge_verdicts", []) or [],
        missed_issues=data.get("missed_issues", []) or [],
        local_freeform=data.get("local_freeform", "") or "",
        cloud_freeform=data.get("cloud_freeform", "") or "",
        error=data.get("error"),
        artifact_run_id=data.get("artifact_run_id"),
    )


def _load_checkpoint(output_path: Path) -> Dict[str, ImageResult]:
    """Load checkpoint and reconstruct full ImageResult objects keyed by property/photo."""
    ckpt_path = _checkpoint_path(output_path)
    if not ckpt_path.exists():
        return {}
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"Checkpoint at {ckpt_path} has unexpected shape; ignoring")
            return {}
        results = {}
        for key, payload in data.items():
            if isinstance(payload, dict):
                results[key] = _deserialize_image_result(payload)
        logger.info(f"Loaded checkpoint with {len(results)} completed images from {ckpt_path}")
        return results
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return {}


def _save_checkpoint(output_path: Path, completed: Dict[str, ImageResult]) -> None:
    """Atomically save the full checkpoint."""
    ckpt_path = _checkpoint_path(output_path)
    tmp = ckpt_path.with_suffix(".checkpoint.json.tmp")
    serialized = {k: _serialize_image_result(r) for k, r in completed.items()}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, default=str)
    os.replace(tmp, ckpt_path)


# ---------------------------------------------------------------------------
# Interactive review checkpoint
# ---------------------------------------------------------------------------

async def _prompt_review(
    phase: str,
    image_results: List[ImageResult],
    total: int,
    *,
    skipped_from_checkpoint: int = 0,
    mode_label: str = "",
) -> str:
    """
    Prompt the user mid-run to decide whether to continue, stop, or skip the
    current phase. Returns one of: "continue", "stop", "next".

    `phase` is "phase12", "judge", or "boundary". The header and stat lines
    are tailored to each. `mode_label` (e.g. "full-scan") is appended to the
    Phase 1+2 header for context.
    """
    successful = [r for r in image_results if not r.error]
    errors = sum(1 for r in image_results if r.error)
    local_obs = sum(len(r.local_observations) for r in successful)
    cloud_obs = sum(len(r.cloud_observations) for r in successful)
    verdicts = sum(len(r.judge_verdicts) for r in successful)
    missed = sum(len(r.missed_issues) for r in successful)

    if phase == "phase12":
        header = f"Review checkpoint — Phase 1+2{f' ({mode_label})' if mode_label else ''}"
        stats = (
            f"  Processed: {len(image_results)}/{total} images\n"
            f"  Successful: {len(successful)}     Errors: {errors}\n"
            f"  Local obs so far: {local_obs}     Cloud obs so far: {cloud_obs}"
        )
        next_action = "next phase (skip rest of Phase 1+2)"
    elif phase == "judge":
        header = "Review checkpoint — Phase 3 (judge)"
        stats = (
            f"  Judged: {len(image_results)}/{total} images\n"
            f"  Verdicts: {verdicts}     Missed-issues: {missed}\n"
            f"  Skipped from checkpoint: {skipped_from_checkpoint}"
        )
        next_action = "next phase (skip rest of Phase 3)"
    else:  # boundary
        header = "Review checkpoint — between Phase 1+2 and Phase 3"
        stats = (
            f"  Phase 1+2 complete: {len(successful)}/{total} successful, {errors} errors\n"
            f"  Local obs: {local_obs}     Cloud obs: {cloud_obs}"
        )
        next_action = "skip judging entirely (go straight to synthesis)"

    bar = "─" * 3
    prompt_text = (
        f"\n{bar} {header} {bar}\n"
        f"{stats}\n"
        f"[c]ontinue / [s]top + write report / [n]ext phase ({next_action})\n"
        "> "
    )

    # Loop until we get a valid response. asyncio.to_thread keeps the event
    # loop alive while we wait on stdin so any in-flight tasks don't stall.
    while True:
        try:
            raw = await asyncio.to_thread(input, prompt_text)
        except EOFError:
            # Non-interactive stdin — treat as continue so unattended runs
            # don't hang forever.
            logger.warning("  stdin EOF during review prompt — continuing")
            return "continue"
        choice = (raw or "").strip().lower()
        if choice in ("", "c", "continue", "y", "yes"):
            return "continue"
        if choice in ("s", "stop", "abort", "q", "quit"):
            return "stop"
        if choice in ("n", "next", "skip"):
            return "next"
        print(f"  Unrecognized input: {raw!r}. Enter c, s, or n.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace):
    start_time = time.time()

    # Load catalog
    logger.info(f"Loading catalog from {cfg.ISSUE_CATALOG_PATH}")
    with open(cfg.ISSUE_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Initialize components
    logger.info("Initializing VLM client and embeddings retriever...")
    vlm_client = create_vlm_client()
    local_config, gpt_config = get_model_configs_from_pipeline_config(cfg)

    # Select cloud provider
    if args.cloud_provider == "gemini":
        cloud_config = get_gemini_config_from_pipeline_config(cfg)
    else:
        cloud_config = gpt_config

    cloud_provider = cloud_config.get("provider", args.cloud_provider)
    cloud_model = cloud_config.get("model", "unknown")
    cloud_label = f"{cloud_provider}/{cloud_model}"

    # Build structuring config — used for Pass 2b/2c on freeform output.
    # We deliberately do NOT use the local model here: it can mangle cloud-
    # generated freeform during JSON structuring, which previously biased
    # the audit by misattributing structuring failures. Defaults to a
    # cheaper cloud model (gpt-5.4-mini) so we don't pay full cloud price
    # for what is mostly text-to-JSON conversion.
    if args.structuring_model:
        structuring_model = args.structuring_model
    elif cloud_provider == "openai":
        structuring_model = getattr(cfg, "GPT_PASS_2B_MODEL", None) or "gpt-5.4-mini"
    else:
        # Gemini: reuse the same cloud model since there's no configured mini.
        structuring_model = cloud_model

    structuring_config = {**cloud_config, "model": structuring_model}
    structuring_label = f"{cloud_provider}/{structuring_model}"

    # Apply local model override if specified
    if args.local_model:
        local_config["model"] = args.local_model
        logger.info(f"Local model overridden to: {args.local_model}")

    retriever = CatalogEmbeddingsRetriever(
        catalog,
        model_name=cfg.EMBEDDINGS_MODEL_NAME,
        device=cfg.EMBEDDINGS_DEVICE,
        trust_remote_code=getattr(cfg, "EMBEDDINGS_TRUST_REMOTE_CODE", False),
        default_topk=cfg.EMBEDDINGS_TOPK,
        guardrails=build_guardrails_from_catalog(catalog),
    )

    models_info = {
        "local_model": local_config.get("model", "unknown"),
        "cloud_model": cloud_model,
        "cloud_provider": cloud_provider,
        "structuring_model": structuring_model,
        "embeddings_model": cfg.EMBEDDINGS_MODEL_NAME,
    }
    logger.info(
        f"Models: local={models_info['local_model']}, cloud={cloud_label}, "
        f"structuring={structuring_label}"
    )

    # Discover images
    artifacts_dir = Path(args.artifacts_dir)
    images_base = Path(args.images_base)
    images = discover_images(artifacts_dir, images_base, args.mode, args.max_images, args.property)

    if not images:
        logger.error("No images found. Check --artifacts-dir and --images-base paths.")
        return

    try:
        run_property_id = _derive_single_property_id(images)
        output_path, output_version = _resolve_output_path(args, images)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(2) from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output report path: {output_path}")
    logger.info(f"Checkpoint path: {_checkpoint_path(output_path)}")
    if args.resume_latest:
        logger.info(f"Resuming latest checkpoint for property_id={run_property_id}")

    # Load checkpoint
    checkpoint = _load_checkpoint(output_path)

    # ── Phase 1+2: Process images ────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1+2: Processing {len(images)} images...")
    logger.info(f"{'='*60}")

    sem = asyncio.Semaphore(args.concurrency)
    # Checkpoint stores full ImageResult objects keyed by "property/photo".
    # Cached entries are passed through (Phase 1+2 skipped); the judge phase
    # below decides separately whether to re-judge based on whether the
    # cached result already has verdicts.
    completed: Dict[str, ImageResult] = dict(checkpoint)
    save_lock = asyncio.Lock()

    async def _process_with_sem(img_info: Dict) -> ImageResult:
        key = f"{img_info['property_id']}/{img_info['photo_key']}"
        cached = completed.get(key)
        if cached is not None and not cached.error:
            logger.info(f"  [cached] {key}")
            return cached

        async with sem:
            logger.info(f"  Processing {key}...")
            if args.mode == "full-scan":
                r = await process_single_image_fullscan(
                    img_info, vlm_client, retriever, local_config, cloud_config, structuring_config
                )
            else:
                r = await process_single_image(
                    img_info, vlm_client, retriever, local_config, cloud_config, structuring_config, args.mode
                )

        # Persist after each image so a crash doesn't lose work.
        if not r.error:
            async with save_lock:
                completed[key] = r
                try:
                    _save_checkpoint(output_path, completed)
                except Exception as e:
                    logger.warning(f"  Checkpoint save failed for {key}: {e}")
        return r

    # When review-every > 0 we batch-process so we can pause for user input
    # between batches. Concurrency within a batch is preserved by the sem.
    # When disabled (default 0), one giant batch == old gather behavior.
    review_every = max(0, int(args.review_every or 0))
    batch_size = review_every if review_every > 0 else len(images)
    image_results: List[ImageResult] = []
    stop_requested = False        # user said "stop + write report"
    skip_phase3 = False           # user said "skip Phase 3" at the boundary
    phase12_cut_short = False     # user said "next" mid-Phase 1+2 (just stop processing more)

    for batch_start in range(0, len(images), batch_size):
        batch = images[batch_start:batch_start + batch_size]
        batch_results = await asyncio.gather(*[_process_with_sem(img) for img in batch])
        image_results.extend(batch_results)

        is_last_batch = (batch_start + batch_size) >= len(images)
        if review_every > 0 and not is_last_batch:
            action = await _prompt_review(
                "phase12",
                image_results,
                total=len(images),
                mode_label=args.mode,
            )
            if action == "stop":
                logger.info("  User chose to stop after Phase 1+2 mid-phase prompt")
                stop_requested = True
                break
            if action == "next":
                logger.info("  User chose to skip rest of Phase 1+2 (will proceed to Phase 3)")
                phase12_cut_short = True
                break

    successful = [r for r in image_results if not r.error]
    logger.info(f"\nPhase 1+2 complete: {len(successful)}/{len(image_results)} images processed successfully")

    # Phase boundary prompt — only when review is on AND we're not already
    # stopping AND we didn't just say "skip rest of Phase 1+2" (which already
    # implies "proceed to Phase 3") AND there's something to judge.
    if (review_every > 0
            and not stop_requested
            and not phase12_cut_short
            and not args.dry_run
            and successful):
        action = await _prompt_review(
            "boundary",
            image_results,
            total=len(images),
        )
        if action == "stop":
            logger.info("  User chose to stop at Phase 1+2/3 boundary")
            stop_requested = True
        elif action == "next":
            logger.info("  User chose to skip Phase 3 entirely")
            skip_phase3 = True

    # ── Phase 3: Judge ───────────────────────────────────────────────────
    if not args.dry_run and not stop_requested and not skip_phase3:
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 3: Running {cloud_label} judge on {len(successful)} images...")
        logger.info(f"{'='*60}")

        judge_delay = args.judge_delay
        judged_count = 0
        skipped_count = 0
        # Tracks images we've actually fed to the judge this run, for the
        # mid-phase prompt cadence (cached entries don't count).
        new_judge_calls = 0
        for i, result in enumerate(successful, 1):
            key = f"{result.property_id}/{result.photo_key}"
            # Skip if cached checkpoint already has judge output for this image.
            if result.judge_verdicts or result.missed_issues:
                skipped_count += 1
                logger.info(f"  [{i}/{len(successful)}] [judge cached] {key}")
                continue

            logger.info(f"  [{i}/{len(successful)}] Judging {key}...")
            try:
                verdicts, missed = await run_judge(result, vlm_client, cloud_config, catalog)
                result.judge_verdicts = verdicts
                result.missed_issues = missed
                judged_count += 1
                new_judge_calls += 1
                logger.info(f"    → {len(verdicts)} verdicts, {len(missed)} missed issues")
                # Persist after each judge call so a crash mid-Phase-3 doesn't lose verdicts.
                completed[key] = result
                try:
                    _save_checkpoint(output_path, completed)
                except Exception as ckpt_err:
                    logger.warning(f"  Checkpoint save failed for {key}: {ckpt_err}")
            except Exception as e:
                err_str = str(e)
                if "insufficient_quota" in err_str or "billing" in err_str.lower() or "quota" in err_str.lower() or "ResourceExhausted" in err_str:
                    logger.error(f"  {cloud_label} quota exceeded — aborting judge phase. "
                                 f"({judged_count}/{len(successful)} images judged this run, "
                                 f"{skipped_count} from checkpoint)")
                    break
                logger.error(f"  Judge error for {result.photo_key}: {e}")

            # Mid-phase review prompt every N actual judge calls (cached
            # entries don't trigger it). Skip on the very last image.
            if (review_every > 0
                    and new_judge_calls > 0
                    and new_judge_calls % review_every == 0
                    and i < len(successful)):
                action = await _prompt_review(
                    "judge",
                    successful[:i],
                    total=len(successful),
                    skipped_from_checkpoint=skipped_count,
                )
                if action == "stop":
                    logger.info("  User chose to stop in Phase 3")
                    stop_requested = True
                    break
                if action == "next":
                    logger.info("  User chose to skip rest of Phase 3")
                    break

            # Pace requests to avoid rate limits
            if judge_delay > 0 and i < len(successful):
                await asyncio.sleep(judge_delay)

        if skipped_count:
            logger.info(f"  Phase 3: {judged_count} judged this run, {skipped_count} reused from checkpoint")
    elif args.dry_run:
        logger.info("\n[dry-run] Skipping Phase 3 (judge)")
    elif stop_requested:
        logger.info("\nSkipping Phase 3 (user stopped earlier)")
    elif skip_phase3:
        logger.info("\nSkipping Phase 3 (user chose to skip judging at boundary)")

    # ── Phase 4: Synthesis ───────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Phase 4: Synthesizing report...")
    logger.info(f"{'='*60}")

    report = _synthesize_report(successful, catalog, retriever, models_info)
    report["meta"].update({
        "property_id": run_property_id,
        "output_version": f"v{output_version:03d}" if output_version else None,
        "output_path": str(output_path),
        "checkpoint_path": str(_checkpoint_path(output_path)),
    })

    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start_time

    # Print summary
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"CATALOG AUDIT REPORT")
    print(f"{'='*60}")
    print(f"Property ID:             {report['meta']['property_id']}")
    print(f"Output version:          {report['meta']['output_version']}")
    print(f"Images processed:        {report['meta']['images_processed']}")
    print(f"Cloud provider:          {cloud_label}")
    print(f"Structuring model:       {structuring_label}")
    print(f"Local observations:      {s['total_local_obs']}")
    print(f"Cloud observations:      {s['total_cloud_obs']}")
    if not args.dry_run:
        print(f"True positives:          {s['true_positives']}")
        print(f"False positives:         {s['false_positives']}")
        print(f"Missed issues:           {s['missed_issues']}")
    print(f"Strong matches:          {s['strong_matches']}")
    print(f"Weak matches:            {s['weak_matches']}")
    print(f"Unmatched:               {s['unmatched']}")
    if not args.dry_run:
        print(f"\nFailure Stage Breakdown:")
        for stage, data in report["failure_stage_breakdown"].items():
            print(f"  {stage}: {data['count']}")
        print(f"\nHigh-confidence new entries:  {s['high_confidence_new_entries']}")
        print(f"Candidate new entries:       {s['candidate_new_entries']}")
        print(f"Proposed improvements:       {s['proposed_improvements']}")
    print(f"\nElapsed time:            {elapsed:.1f}s")
    print(f"Report saved to:         {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Catalog Auditor — cloud-model judge-based catalog improvement tool (supports OpenAI GPT and Google Gemini)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(DEFAULT_ARTIFACTS_DIR),
        help="Path to artifacts directory (default: %(default)s)",
    )
    parser.add_argument(
        "--images-base",
        default=str(DEFAULT_IMAGES_BASE),
        help="Base path for property images (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for auto-named reports (default: current directory)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Deprecated alias for --output-dir. Must be a directory; report "
            "filenames are derived from property_id automatically."
        ),
    )
    parser.add_argument(
        "--property",
        default=None,
        help="Filter to a specific property ID (e.g. redfin_126224899)",
    )
    parser.add_argument(
        "--cloud-provider",
        choices=["openai", "gemini"],
        default="openai",
        help="Cloud model provider for dual-model analysis and judge (default: %(default)s)",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Override the local LM Studio model name (e.g. google/gemma-4-26b-a4b)",
    )
    parser.add_argument(
        "--structuring-model",
        default=None,
        help=(
            "Cloud model used for Pass 2b/2c structuring of freeform output. "
            "Defaults to GPT_PASS_2B_MODEL or gpt-5.4-mini for OpenAI; same as "
            "--cloud-provider model for Gemini. Avoids using the local model "
            "for structuring (which can mangle cloud freeform)."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images to process (default: 20 for multi-property, all for single --property)",
    )
    parser.add_argument(
        "--mode",
        choices=["issues-only", "full-scan"],
        default="issues-only",
        help="issues-only: only images with existing detections. full-scan: all images (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Phase 3 cloud judge (only dual-model comparison + match scores)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent image processing tasks (default: %(default)s)",
    )
    parser.add_argument(
        "--judge-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between judge calls to avoid rate limits (default: %(default)s)",
    )
    parser.add_argument(
        "--review-every",
        type=int,
        default=0,
        help=(
            "Pause every N images and prompt the user (continue / stop / "
            "skip-to-next-phase). Also prompts at the boundary between "
            "Phase 1+2 and Phase 3. Default 0 disables prompting."
        ),
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the newest checkpoint for the discovered property in the output directory.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.output and args.output_dir not in (None, "."):
        parser.error("Use either --output-dir or the deprecated --output directory alias, not both.")
    for attr, label in (("output", "--output"), ("output_dir", "--output-dir")):
        value = getattr(args, attr, None)
        if value and _looks_like_output_file_path(value):
            parser.error(
                f"{label} now expects a directory, not a report filename: {value}. "
                "Report filenames are derived from property_id automatically."
            )

    # Default max_images: no cap for single property, 20 for multi-property scans
    if args.max_images is None:
        args.max_images = 0 if args.property else 20

    return args


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
