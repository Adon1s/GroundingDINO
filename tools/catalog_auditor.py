"""
Catalog Auditor — GPT-5.4 judge-based catalog improvement tool.

Runs dual-model analysis (local + GPT-5.4) on property images, then uses
GPT-5.4 as a judge to diagnose pipeline failures and propose catalog
improvements.

Usage:
    python tools/catalog_auditor.py --max-images 20
    python tools/catalog_auditor.py --dry-run --max-images 5
    python tools/catalog_auditor.py --mode full-scan --max-images 10
"""

import argparse
import asyncio
import json
import logging
import os
import random
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
from vlm_client import VLMClient, create_vlm_client, get_model_configs_from_pipeline_config
from catalog_embeddings import CatalogEmbeddingsRetriever, MatchCandidate
from scene_classifier_passes import (
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
    source_model_2a: str           # "local" or "gpt"
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
    # GPT model observations
    gpt_observations: List[ObservationRecord] = field(default_factory=list)

    # Judge verdicts (Phase 3)
    judge_verdicts: List[Dict] = field(default_factory=list)
    missed_issues: List[Dict] = field(default_factory=list)

    # Raw data for debugging
    local_freeform: str = ""
    gpt_freeform: str = ""
    error: Optional[str] = None


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

        candidates = retriever.retrieve_candidates(
            desc,
            allowed_kinds={kind},
            allowed_groups={scene_group},
            topk=5,
        )

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
        candidates = retriever.retrieve_candidates(
            desc,
            allowed_kinds={kind},
            allowed_groups={scene_group},
            topk=5,
        )

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
    gpt_config: Dict,
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

        # ── Run GPT-5.4 Pass 2a on the image ────────────────────────────
        gpt_2a = await run_pass_2a(image_path, vlm_client, gpt_config)
        result.gpt_freeform = gpt_2a.observations_freeform

        gpt_freeform = gpt_2a.observations_freeform.strip()
        if gpt_freeform:
            # Truncate if GPT output is excessively long (prevent context overflow on local model)
            MAX_FREEFORM_CHARS = 3000
            if len(gpt_freeform) > MAX_FREEFORM_CHARS:
                logger.warning(
                    f"  GPT freeform output truncated: {len(gpt_freeform)} → {MAX_FREEFORM_CHARS} chars"
                )
                gpt_freeform = gpt_freeform[:MAX_FREEFORM_CHARS]

            # Structure GPT observations through local model (2b + 2c)
            gpt_2b = await run_pass_2b(vlm_client, local_config, gpt_freeform)
            if gpt_2b.observations:
                gpt_2c = await run_pass_2c(vlm_client, local_config, gpt_2b.observations, scene_id)
                # Run embedding audit on GPT's forward observations
                gpt_records = _run_embedding_audit(
                    gpt_2c.labeled_forward, retriever, result.scene_group, "gpt"
                )
                result.gpt_observations = gpt_records

        logger.info(
            f"  {photo_key}: scene={scene_id}, "
            f"local_obs={len(result.local_observations)}, "
            f"gpt_obs={len(result.gpt_observations)}"
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
    gpt_config: Dict,
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
        # Pass 1a: Scene classification
        scene_result = await run_pass_1a_scene_type(image_path, vlm_client, local_config)
        result.scene = scene_result.scene
        result.scene_group = SCENE_TO_GROUP.get(scene_result.scene, "other")

        # Pass 2a: Run both models concurrently
        local_2a, gpt_2a = await asyncio.gather(
            run_pass_2a(image_path, vlm_client, local_config),
            run_pass_2a(image_path, vlm_client, gpt_config),
        )
        result.local_freeform = local_2a.observations_freeform
        result.gpt_freeform = gpt_2a.observations_freeform

        # Pass 2b+2c for both (using local model for structuring)
        async def _structure_and_audit(freeform: str, source: str) -> List[ObservationRecord]:
            if not freeform.strip():
                return []
            r2b = await run_pass_2b(vlm_client, local_config, freeform)
            if not r2b.observations:
                return []
            r2c = await run_pass_2c(vlm_client, local_config, r2b.observations, result.scene)
            return _run_embedding_audit(r2c.labeled_forward, retriever, result.scene_group, source)

        local_records, gpt_records = await asyncio.gather(
            _structure_and_audit(local_2a.observations_freeform, "local"),
            _structure_and_audit(gpt_2a.observations_freeform, "gpt"),
        )
        result.local_observations = local_records
        result.gpt_observations = gpt_records

        logger.info(
            f"  {photo_key}: scene={result.scene}, "
            f"local_obs={len(local_records)}, gpt_obs={len(gpt_records)}"
        )

    except Exception as e:
        logger.error(f"Error processing {photo_key}: {e}")
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Phase 3: GPT-5.4 Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert real estate photo analyst and issue catalog quality auditor.
Your job is to diagnose failures in an automated defect/upgrade detection pipeline.

You will see an image, observations from two models (local and GPT), and the catalog items they matched to.
For each observation, diagnose WHICH STEP in the pipeline failed (if any):
- observation: the model hallucinated or described something not visible
- structuring: the freeform text was reasonable but got mangled during JSON structuring
- labeling: the observation was real but got the wrong label (defect vs upgrade vs other)
- retrieval: the observation is valid AND a matching catalog item exists, but embedding retrieval missed it
- catalog_gap: the observation is valid but NO existing catalog item covers this issue
- false_suppression: the observation was valid and matched, but pipeline rules incorrectly suppressed it

Be precise. Do not jump to "needs new catalog entry" when the real problem is bad embed_text on an existing item."""


def _build_judge_prompt(image_result: ImageResult, catalog: Dict) -> str:
    """Build the user prompt for the judge, including only relevant catalog snippets."""
    catalog_items_by_id = {}
    for item in catalog.get("items", []):
        catalog_items_by_id[item["id"]] = item

    lines = [f"## Scene: {image_result.scene} ({image_result.scene_group})"]

    # Local model observations
    lines.append("\n## Local Model Observations:")
    if not image_result.local_observations:
        lines.append("(none detected)")
    for i, obs in enumerate(image_result.local_observations, 1):
        status = f"[{obs.retrieval_status}]"
        match_info = f" → matched: {obs.best_match_id} (score={obs.best_match_score:.3f})" if obs.best_match_id else " → no match"
        supp = ""
        if obs.was_suppressed:
            supp = f" [SUPPRESSED: {obs.suppression_reason}]"
        lines.append(f"  {i}. {status} ({obs.label}/{obs.kind}) {obs.description}{match_info}{supp}")

    # GPT model observations
    lines.append("\n## GPT Model Observations:")
    if not image_result.gpt_observations:
        lines.append("(none detected)")
    for i, obs in enumerate(image_result.gpt_observations, 1):
        status = f"[{obs.retrieval_status}]"
        match_info = f" → matched: {obs.best_match_id} (score={obs.best_match_score:.3f})" if obs.best_match_id else " → no match"
        lines.append(f"  {i}. {status} ({obs.label}/{obs.kind}) {obs.description}{match_info}")

    # Relevant catalog snippets
    mentioned_ids = set()
    all_obs = image_result.local_observations + image_result.gpt_observations
    for obs in all_obs:
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
                kw = item.get("keywords_allow", [])
                if kw:
                    lines.append(f"    keywords_allow: {kw}")

    lines.append("""
## Your Task:
Evaluate each observation from BOTH models against what you see in the image.

Return JSON:
{
  "observation_verdicts": [
    {
      "source": "local" or "gpt",
      "description": "the original observation text",
      "is_visible_issue": true/false,
      "is_observation_reasonable": true/false,
      "is_structured_label_reasonable": true/false,
      "is_matched_catalog_item_correct": true/false/null,
      "best_existing_catalog_item_id_if_any": "item_id" or null,
      "needs_new_catalog_entry": true/false,
      "suggested_catalog_improvement_type": "embed_text"/"keywords"/"new_entry"/"none",
      "suggested_embed_text_fix": "..." or null,
      "failure_stage_override": "observation"/"structuring"/"labeling"/"retrieval"/"catalog_gap"/"false_suppression" or null,
      "notes": "brief reasoning"
    }
  ],
  "missed_issues": [
    {
      "description": "issue visible in photo that neither model caught",
      "kind": "defect" or "upgrade",
      "suggested_catalog_entry_name": "proposed name",
      "category": "cosmetic/moisture/structure/exterior/safety/systems/opportunity",
      "trade_bucket": "..."
    }
  ]
}""")

    return "\n".join(lines)


async def run_judge(
    image_result: ImageResult,
    vlm_client: VLMClient,
    gpt_config: Dict,
    catalog: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    """Run the GPT-5.4 judge on a single image. Returns (verdicts, missed_issues)."""
    user_prompt = _build_judge_prompt(image_result, catalog)

    # Need higher token limit for the judge response — complex images with
    # many observations can produce very long verdicts
    judge_config = {**gpt_config, "max_tokens": 4000}

    try:
        response = await vlm_client.analyze_image(
            image_path=Path(image_result.image_path),
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **judge_config,
        )

        result = extract_json_object(response)

        # If extract_json_object fails, try repairing common truncation issues
        if result is None and response:
            try:
                # Try to find and fix truncated JSON — close open arrays/objects
                import re
                # Find the start of our JSON
                json_match = re.search(r'\{', response)
                if json_match:
                    raw = response[json_match.start():]
                    # Count unclosed brackets and braces
                    open_braces = raw.count('{') - raw.count('}')
                    open_brackets = raw.count('[') - raw.count(']')
                    # Try closing them
                    repaired = raw + ']' * open_brackets + '}' * open_braces
                    result = json.loads(repaired)
                    logger.info(f"  Judge JSON repaired for {image_result.photo_key}")
            except Exception:
                pass

        if result is None:
            logger.warning(f"Judge returned unparseable response for {image_result.photo_key} ({len(response or '')} chars)")
            return [], []

        verdicts = result.get("observation_verdicts", [])
        missed = result.get("missed_issues", [])

        if not isinstance(verdicts, list):
            verdicts = []
        if not isinstance(missed, list):
            missed = []

        return verdicts, missed

    except Exception as e:
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
    total_gpt = sum(len(r.gpt_observations) for r in image_results)
    total_missed = sum(len(r.missed_issues) for r in image_results)

    all_obs = []
    for r in image_results:
        all_obs.extend(r.local_observations)
        all_obs.extend(r.gpt_observations)

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
        for obs in r.local_observations + r.gpt_observations:
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

    if new_entry_candidates and len(new_entry_candidates) >= 2:
        descs = [c["description"] for c in new_entry_candidates]
        embeddings = retriever._encode_queries(descs)

        # Simple agglomerative clustering: cosine similarity > 0.7 = same cluster
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
                if not visited[j] and sim[i, j] > 0.7:
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

            if distinct_props >= 2 or distinct_scenes >= 2:
                high_confidence.append(entry)
            else:
                entry["reason_for_candidate_tier"] = (
                    f"Only {distinct_props} property(s), {distinct_scenes} scene(s)"
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
            "current_keywords": item.get("keywords_allow", []),
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
            "local_observations": [asdict(o) for o in r.local_observations],
            "gpt_observations": [asdict(o) for o in r.gpt_observations],
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
            "total_gpt_obs": total_gpt,
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
        "proposed_catalog_improvements": proposed_improvements,
        "false_positive_patterns": false_positives,
        "missed_issue_patterns": missed_list[:20],
    }


def _slugify(text: str) -> str:
    """Convert text to a slug suitable for a catalog ID."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s_]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text[:50]


# ---------------------------------------------------------------------------
# Checkpoint support
# ---------------------------------------------------------------------------

def _load_checkpoint(output_path: Path) -> Dict[str, ImageResult]:
    """Load checkpoint data if it exists."""
    ckpt_path = output_path.with_suffix(".checkpoint.json")
    if not ckpt_path.exists():
        return {}
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded checkpoint with {len(data)} completed images")
        return data
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return {}


def _save_checkpoint(output_path: Path, completed: Dict):
    """Save checkpoint after each image."""
    ckpt_path = output_path.with_suffix(".checkpoint.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(completed, f, indent=2, default=str)


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
    )

    models_info = {
        "local_model": local_config.get("model", "unknown"),
        "gpt_model": gpt_config.get("model", "unknown"),
        "embeddings_model": cfg.EMBEDDINGS_MODEL_NAME,
    }
    logger.info(f"Models: local={models_info['local_model']}, gpt={models_info['gpt_model']}")

    # Discover images
    artifacts_dir = Path(args.artifacts_dir)
    images_base = Path(args.images_base)
    images = discover_images(artifacts_dir, images_base, args.mode, args.max_images, args.property)

    if not images:
        logger.error("No images found. Check --artifacts-dir and --images-base paths.")
        return

    # Load checkpoint
    output_path = Path(args.output)
    checkpoint = _load_checkpoint(output_path)

    # ── Phase 1+2: Process images ────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1+2: Processing {len(images)} images...")
    logger.info(f"{'='*60}")

    sem = asyncio.Semaphore(args.concurrency)
    image_results: List[ImageResult] = []

    async def _process_with_sem(img_info: Dict) -> ImageResult:
        async with sem:
            key = f"{img_info['property_id']}/{img_info['photo_key']}"
            if key in checkpoint:
                logger.info(f"  [cached] {key}")
                # Reconstruct ImageResult from checkpoint
                # For now just skip and re-process
                pass

            logger.info(f"  Processing {key}...")
            if args.mode == "full-scan":
                return await process_single_image_fullscan(
                    img_info, vlm_client, retriever, local_config, gpt_config
                )
            else:
                return await process_single_image(
                    img_info, vlm_client, retriever, local_config, gpt_config, args.mode
                )

    tasks = [_process_with_sem(img) for img in images]
    image_results = await asyncio.gather(*tasks)

    # Save checkpoint
    completed_keys = {
        f"{r.property_id}/{r.photo_key}": True
        for r in image_results if not r.error
    }
    _save_checkpoint(output_path, completed_keys)

    successful = [r for r in image_results if not r.error]
    logger.info(f"\nPhase 1+2 complete: {len(successful)}/{len(image_results)} images processed successfully")

    # ── Phase 3: Judge ───────────────────────────────────────────────────
    if not args.dry_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 3: Running GPT-5.4 judge on {len(successful)} images...")
        logger.info(f"{'='*60}")

        for i, result in enumerate(successful, 1):
            logger.info(f"  [{i}/{len(successful)}] Judging {result.property_id}/{result.photo_key}...")
            verdicts, missed = await run_judge(result, vlm_client, gpt_config, catalog)
            result.judge_verdicts = verdicts
            result.missed_issues = missed
            logger.info(f"    → {len(verdicts)} verdicts, {len(missed)} missed issues")
    else:
        logger.info("\n[dry-run] Skipping Phase 3 (judge)")

    # ── Phase 4: Synthesis ───────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Phase 4: Synthesizing report...")
    logger.info(f"{'='*60}")

    report = _synthesize_report(successful, catalog, retriever, models_info)

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start_time

    # Print summary
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"CATALOG AUDIT REPORT")
    print(f"{'='*60}")
    print(f"Images processed:        {report['meta']['images_processed']}")
    print(f"Local observations:      {s['total_local_obs']}")
    print(f"GPT observations:        {s['total_gpt_obs']}")
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
        description="Catalog Auditor — GPT-5.4 judge-based catalog improvement tool",
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
        "--output", "-o",
        default=None,
        help="Output report path (default: catalog_audit_<timestamp>.json)",
    )
    parser.add_argument(
        "--property",
        default=None,
        help="Filter to a specific property ID (e.g. redfin_126224899)",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Override the local LM Studio model name (e.g. google/gemma-4-26b-a4b)",
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
        help="Skip Phase 3 GPT judge (only dual-model comparison + match scores)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent image processing tasks (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"catalog_audit_{ts}.json"

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
