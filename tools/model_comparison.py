"""
Model Comparison — head-to-head Gemma 4 vs Qwen 3.6 across pipeline passes.

Reframes "Gemma or Qwen?" as four separate skill questions (2a, 2b, 2c, 2d)
plus a coupled 2c+2d real-world-unit question. GPT-5.4 judges each skill
independently, so a model that's best at seeing can still lose at structuring.

The judge is BLINDED: it sees "Model A" / "Model B" only, never "Gemma" /
"Qwen". This reduces position and identity bias. --ab-order controls how
A/B is assigned; 'random' (default) is deterministic per (photo, skill) for
reproducibility.

Sequential hosting: load one model in LM Studio at a time. The script runs
Phase A with Gemma, prompts to swap, then Phase B with Qwen. GPT-5.4 fixtures
(Phase 0) and judges (Phase D) run without any local model loaded.

Usage:
    python tools/model_comparison.py --property redfin_126224899 --max-images 20
    python tools/model_comparison.py --property redfin_126224899 --resume
    python tools/model_comparison.py --property redfin_126224899 --skip-skills 2d_isolated

Bias-check workflow (cheap: reuses all cells, only re-runs Phase D):
    # 1) Initial run with forced gemma-first
    python tools/model_comparison.py --property X --ab-order gemma_first
    # 2) Re-judge the same cells with forced qwen-first
    python tools/model_comparison.py --property X --resume --force-rejudge \\
        --ab-order qwen_first
    # 3) Diff the two reports — large winner-flip rate = position bias is real.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from vlm_client import (
    VLMClient,
    create_vlm_client,
    get_model_configs_from_pipeline_config,
)
from catalog_embeddings import CatalogEmbeddingsRetriever, MatchCandidate, build_guardrails_from_catalog
from scene_classifier_passes import (
    evaluate_kind_routing,
    prioritize_resolution_candidates,
    run_pass_2a,
    run_pass_2b,
    run_pass_2c,
    run_pass_2d,
)
from llm_json import extract_json_object

logger = logging.getLogger("model_comparison")

# ---------------------------------------------------------------------------
# Constants (copied from catalog_auditor.py for consistency)
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

MAX_FREEFORM_CHARS = 3000  # truncation cap for freeform before feeding to 2b

# Skill keys — stable throughout code + output JSON
SKILLS = ("2a", "2b", "2c", "2d_isolated", "2c+2d_coupled")
MODEL_KEYS = ("gemma", "qwen")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TwoDRow:
    """One 2d decision, enriched with diagnostic context for post-hoc analysis."""
    observation: str
    kind: str                                    # defect | upgrade
    candidates: List[Dict[str, Any]]             # [{item_id, name, trade_bucket, score}, ...]
    chosen_id: Optional[str]
    chosen_null: bool
    # Coupled-mode only: the 2c output that fed this 2d call
    from_2c: Optional[Dict[str, str]] = None     # {label, kind}
    # Filled in by judge (Phase D) — never by the local model
    judge_correct_id: Optional[str] = None
    judge_correct_rank: Optional[int] = None     # 1-indexed, null if not in candidates
    judge_correct_in_candidates: Optional[bool] = None


@dataclass
class ModelCells:
    """All outputs for one local model (Gemma or Qwen) on one image."""
    pass_2a: Optional[str] = None                              # freeform text
    pass_2b: List[Dict[str, str]] = field(default_factory=list)  # [{description}]
    pass_2c: List[Dict[str, str]] = field(default_factory=list)  # [{description, label}]
    pass_2d_isolated: List[TwoDRow] = field(default_factory=list)
    pass_2c_2d_coupled: List[TwoDRow] = field(default_factory=list)


@dataclass
class FixtureCells:
    """GPT-5.4 fixture outputs used to isolate downstream passes."""
    pass_2a: Optional[str] = None
    pass_2b: List[Dict[str, str]] = field(default_factory=list)
    pass_2c: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ImageRecord:
    """All cells + judge verdicts for a single image."""
    property_id: str
    photo_key: str
    image_path: str
    scene: str = "other"
    scene_group: str = "other"

    fixture: FixtureCells = field(default_factory=FixtureCells)
    gemma: ModelCells = field(default_factory=ModelCells)
    qwen: ModelCells = field(default_factory=ModelCells)

    judge: Dict[str, Optional[Dict[str, Any]]] = field(
        default_factory=lambda: {k: None for k in SKILLS}
    )
    # Per-skill count of JSON parse failures encountered during judging (0..2).
    # Surfaced in the report meta so silent judge dropouts are visible.
    judge_parse_failures: Dict[str, int] = field(default_factory=dict)

    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_images(
    artifacts_dir: Path,
    images_base: Path,
    max_images: Optional[int],
    property_filter: Optional[str],
) -> List[Dict[str, Any]]:
    """Scan artifacts dir + resolve image paths. Returns list of image info dicts."""
    out: List[Dict[str, Any]] = []
    if not artifacts_dir.exists():
        logger.error(f"Artifacts directory not found: {artifacts_dir}")
        return out

    for prop_dir in sorted(artifacts_dir.iterdir()):
        if not prop_dir.is_dir():
            continue
        property_id = prop_dir.name
        if property_filter and property_id != property_filter:
            continue

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

            scene_info = photo_data.get("scene", {})
            scene_id = (
                scene_info.get("id", "other")
                if isinstance(scene_info, dict)
                else "other"
            )

            out.append({
                "property_id": property_id,
                "photo_key": photo_key,
                "image_path": str(image_path),
                "scene": scene_id,
                "scene_group": SCENE_TO_GROUP.get(scene_id, "other"),
            })

    if max_images and max_images > 0 and len(out) > max_images:
        out = out[:max_images]

    logger.info(f"Discovered {len(out)} images")
    return out


# ---------------------------------------------------------------------------
# Pass 2d candidate retrieval
# ---------------------------------------------------------------------------

def _label_to_kind(label: str) -> str:
    return "upgrade" if (label or "").strip().lower() == "upgrade_candidate" else "defect"


def _retrieve_candidates_for_obs(
    retriever: CatalogEmbeddingsRetriever,
    description: str,
    kind: str,
    scene_group: str,
    topk: int = 5,
) -> Tuple[List[Dict[str, Any]], Any]:
    """Run embeddings retrieval and format candidates as dicts suitable for run_pass_2d + storage."""
    routing = evaluate_kind_routing(description, kind)
    allowed_kinds = set(routing.expanded_kinds) if routing.expanded_kinds else ({kind} if kind else None)
    requested_topk = topk * 2 if allowed_kinds and len(allowed_kinds) > 1 else topk
    matches: List[MatchCandidate] = retriever.retrieve_candidates(
        description,
        allowed_kinds=allowed_kinds,
        allowed_groups={scene_group},
        topk=requested_topk,
    )
    candidates = [
        {
            "item_id": m.item_id,
            "name": m.name,
            "trade_bucket": m.trade_bucket,
            "kind": m.kind,
            "description": m.description,
            "support_any": list(m.support_any),
            "defaultHidden": m.defaultHidden,
            "drop_if_generic": m.drop_if_generic,
            "score": round(float(m.score), 4),
        }
        for m in matches
    ]
    candidates = prioritize_resolution_candidates(
        candidates,
        widened_routing=len(routing.expanded_kinds) > 1,
    )
    return candidates[:topk], routing


# ---------------------------------------------------------------------------
# Phase 0: GPT-5.4 fixture generation
# ---------------------------------------------------------------------------

async def build_fixture_for_image(
    image_info: Dict[str, Any],
    vlm_client: VLMClient,
    gpt_config: Dict[str, Any],
) -> FixtureCells:
    """Run Pass 2a → 2b → 2c on one image with GPT-5.4 to produce fixtures."""
    image_path = Path(image_info["image_path"])
    scene = image_info.get("scene", "other")

    r2a = await run_pass_2a(image_path, vlm_client, gpt_config)
    freeform = (r2a.observations_freeform or "").strip()
    if len(freeform) > MAX_FREEFORM_CHARS:
        freeform = freeform[:MAX_FREEFORM_CHARS]

    if not freeform:
        return FixtureCells(pass_2a="", pass_2b=[], pass_2c=[])

    r2b = await run_pass_2b(vlm_client, gpt_config, freeform)
    observations = r2b.observations or []
    if not observations:
        return FixtureCells(pass_2a=freeform, pass_2b=[], pass_2c=[])

    r2c = await run_pass_2c(vlm_client, gpt_config, observations, scene)
    labeled = r2c.labeled_debug or []

    return FixtureCells(
        pass_2a=freeform,
        pass_2b=observations,
        pass_2c=labeled,
    )


# ---------------------------------------------------------------------------
# Phases A & B: local model cells (all 5 skills)
# ---------------------------------------------------------------------------

async def run_local_model_cells(
    image_info: Dict[str, Any],
    fixture: FixtureCells,
    vlm_client: VLMClient,
    model_config: Dict[str, Any],
    retriever: CatalogEmbeddingsRetriever,
    skip_skills: set,
) -> ModelCells:
    """Run all 5 skill cells for one local model on one image."""
    image_path = Path(image_info["image_path"])
    scene = image_info.get("scene", "other")
    scene_group = image_info.get("scene_group", "other")

    cells = ModelCells()

    # ── 2a: seeing on raw image ──────────────────────────────────────────
    if "2a" not in skip_skills:
        r2a = await run_pass_2a(image_path, vlm_client, model_config)
        freeform = (r2a.observations_freeform or "").strip()
        if len(freeform) > MAX_FREEFORM_CHARS:
            freeform = freeform[:MAX_FREEFORM_CHARS]
        cells.pass_2a = freeform

    # ── 2b: structuring F_2a ─────────────────────────────────────────────
    if "2b" not in skip_skills and fixture.pass_2a:
        r2b = await run_pass_2b(vlm_client, model_config, fixture.pass_2a)
        cells.pass_2b = r2b.observations or []

    # ── 2c: labeling F_2b (isolated) ─────────────────────────────────────
    if "2c" not in skip_skills and fixture.pass_2b:
        r2c = await run_pass_2c(vlm_client, model_config, fixture.pass_2b, scene)
        cells.pass_2c = r2c.labeled_debug or []

    # ── 2d isolated: resolve F_2c against embeddings candidates ──────────
    if "2d_isolated" not in skip_skills and fixture.pass_2c:
        cells.pass_2d_isolated = await _run_2d_batch(
            labeled_rows=fixture.pass_2c,
            vlm_client=vlm_client,
            model_config=model_config,
            retriever=retriever,
            scene_group=scene_group,
            include_from_2c=False,
        )

    # ── 2c+2d coupled: use THIS model's own 2c output, feed into 2d ──────
    if "2c+2d_coupled" not in skip_skills and cells.pass_2c:
        cells.pass_2c_2d_coupled = await _run_2d_batch(
            labeled_rows=cells.pass_2c,
            vlm_client=vlm_client,
            model_config=model_config,
            retriever=retriever,
            scene_group=scene_group,
            include_from_2c=True,
        )

    return cells


async def _run_2d_batch(
    labeled_rows: List[Dict[str, str]],
    vlm_client: VLMClient,
    model_config: Dict[str, Any],
    retriever: CatalogEmbeddingsRetriever,
    scene_group: str,
    include_from_2c: bool,
) -> List[TwoDRow]:
    """Run Pass 2d on every labeled observation. One call per observation."""
    results: List[TwoDRow] = []

    # Only defect_or_damage and upgrade_candidate rows are forwarded to 2d.
    forward = [
        r for r in labeled_rows
        if r.get("label") in {"defect_or_damage", "upgrade_candidate"}
    ]

    for row in forward:
        desc = (row.get("description") or "").strip()
        if not desc:
            continue
        label = row.get("label", "")
        kind = _label_to_kind(label)

        candidates, kind_routing = _retrieve_candidates_for_obs(
            retriever=retriever,
            description=desc,
            kind=kind,
            scene_group=scene_group,
        )

        if not candidates:
            # No candidates → 2d cannot run, record the state
            results.append(TwoDRow(
                observation=desc,
                kind=kind,
                candidates=[],
                chosen_id=None,
                chosen_null=True,
                from_2c={"label": label, "kind": kind} if include_from_2c else None,
            ))
            continue

        try:
            r2d = await run_pass_2d(
                vlm_client=vlm_client,
                model_config=model_config,
                observation=desc,
                candidates=candidates,
                kind=kind,
                kind_routing=kind_routing,
            )
            chosen_id = r2d.resolved_item_id
        except Exception as e:
            logger.error(f"Pass 2d error for '{desc[:40]}...': {e}")
            chosen_id = None

        results.append(TwoDRow(
            observation=desc,
            kind=kind,
            candidates=candidates,
            chosen_id=chosen_id,
            chosen_null=(chosen_id is None),
            from_2c={"label": label, "kind": kind} if include_from_2c else None,
        ))

    return results


# ---------------------------------------------------------------------------
# Phase D: Judges (5 separate skill-specific prompts)
# ---------------------------------------------------------------------------

# ── A/B blinding: judge sees "Model A" / "Model B" instead of "Gemma" / "Qwen"
# to reduce position + identity bias. Mapping is per-image+skill (deterministic
# random by default, or forced by --ab-order). Verdicts are canonicalized back
# to gemma/qwen immediately after parsing so the rest of the pipeline is
# unchanged.

import hashlib as _hashlib


def _ab_mapping_for(
    ab_order: str,
    photo_key: str,
    skill: str,
) -> Dict[str, str]:
    """Return {'A': 'gemma'|'qwen', 'B': 'gemma'|'qwen'}.

    `random` is deterministic per (photo_key, skill) so reruns are reproducible
    and so the bias-check workflow (forced gemma_first vs forced qwen_first)
    produces clean comparisons.
    """
    if ab_order == "gemma_first":
        return {"A": "gemma", "B": "qwen"}
    if ab_order == "qwen_first":
        return {"A": "qwen", "B": "gemma"}
    # random: deterministic per (photo_key, skill)
    h = _hashlib.sha256(f"{photo_key}::{skill}".encode("utf-8")).hexdigest()
    if int(h[:8], 16) % 2 == 0:
        return {"A": "gemma", "B": "qwen"}
    return {"A": "qwen", "B": "gemma"}


def _canonicalize_verdict(
    verdict: Optional[Dict[str, Any]],
    ab_mapping: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Convert a judge verdict from A/B naming to gemma/qwen naming in place.

    Preserves `ab_mapping` on the verdict for audit. If the verdict is already
    in gemma/qwen form (e.g. legacy pre-A/B checkpoint), returns it unchanged.
    """
    if not isinstance(verdict, dict):
        return verdict

    # Legacy verdict: already has gemma/qwen keys, nothing to do.
    scores = verdict.get("scores")
    if isinstance(scores, dict) and ("gemma" in scores or "qwen" in scores):
        verdict.setdefault("ab_mapping", ab_mapping)
        return verdict

    # Map winner
    winner = verdict.get("winner")
    if winner in ("A", "B"):
        verdict["winner"] = ab_mapping.get(winner, winner)
    # "tie" passes through unchanged

    # Map scores
    if isinstance(scores, dict):
        new_scores: Dict[str, Any] = {}
        for ab, model in ab_mapping.items():
            if ab in scores:
                new_scores[model] = scores[ab]
        # Preserve any unexpected keys too (defensive)
        for k, v in scores.items():
            if k not in ("A", "B") and k not in new_scores:
                new_scores[k] = v
        verdict["scores"] = new_scores

    # Map top-level model-named lists (unique_good / hallucinations)
    for suffix in ("unique_good", "hallucinations"):
        for ab in ("A", "B"):
            src_key = f"model_{ab.lower()}_{suffix}"
            if src_key in verdict:
                model = ab_mapping.get(ab)
                if model:
                    verdict[f"{model}_{suffix}"] = verdict.pop(src_key)

    # Map per_observation entries
    per_obs = verdict.get("per_observation")
    if isinstance(per_obs, list):
        for entry in per_obs:
            if not isinstance(entry, dict):
                continue
            for ab in ("A", "B"):
                src_key = f"model_{ab.lower()}_correct"
                if src_key in entry:
                    model = ab_mapping.get(ab)
                    if model:
                        entry[f"{model}_correct"] = entry.pop(src_key)

    verdict["ab_mapping"] = ab_mapping
    return verdict


def _norm_text(s: str) -> str:
    """Cheap normalization for robust observation-text matching in backfill.

    Collapses whitespace, lowercases, strips trailing punctuation. Claude's
    and GPT's quote behavior is usually exact, but this catches the rare
    drift (stripped period, smart quotes, trailing space) without risking
    false matches the way fuzzy matching would.
    """
    if not s:
        return ""
    import re
    x = s.strip().lower()
    x = re.sub(r"\s+", " ", x)
    x = x.rstrip(".,;:!?'\"` ")
    return x


# System prompts per skill — all framed as expert real-estate analyst picking a winner.

_JUDGE_SYSTEM_2A = """\
You are an expert real estate photo analyst evaluating two anonymous vision models head-to-head.
The models are labeled only as "Model A" and "Model B" — you do not know their identities and
must not speculate about them.

You will see an image and two freeform observation lists from Model A and Model B.
Your task: decide which model *sees* the image better. Focus on:
- accuracy (each observation is actually visible)
- completeness (the model caught the important stuff)
- hallucination_penalty (fewer fabricated claims is better; score 5 = none, 0 = many)
- specificity (concrete and actionable beats vague)

Return a single JSON object — no prose, no markdown."""

_JUDGE_SYSTEM_2B = """\
You are an expert evaluating two anonymous models' ability to decompose freeform notes into
structured JSON. The models are labeled only as "Model A" and "Model B".

You will see the EXACT freeform input text given to both models, plus each model's JSON array of
observations. Neither model saw the image — this tests pure text-to-JSON faithfulness.

Score each on:
- faithfulness_to_input (did it only restate what the input said?)
- schema_adherence (each entry is a dict with a 'description' string, 5–25 words)
- decomposition_quality (distinct observations correctly split, no merging/dropping)
- no_speculation (no invented causes or consequences)

Return a single JSON object — no prose, no markdown."""

_JUDGE_SYSTEM_2C = """\
You are an expert evaluating two anonymous models' ability to label observations by taxonomy.
The models are labeled only as "Model A" and "Model B".

Labels: defect_or_damage, upgrade_candidate, good_condition, generic_presence, other.
You will see the shared input observation list and each model's labeled output.

Score each on:
- label_correctness (right label for each observation)
- forward_set_purity (the defect_or_damage + upgrade_candidate subset is correct, no leaks)
- no_dimension_leakage (dimension strings like "12'6 x 10'" should be labeled 'other', not defect)

Return a single JSON object — no prose, no markdown."""

_JUDGE_SYSTEM_2D_ISOLATED = """\
You are an expert evaluating two anonymous models' ability to resolve observations to catalog
item IDs. The models are labeled only as "Model A" and "Model B".

Both models received the EXACT SAME labeled observations and the same candidate list per
observation. This isolates the 'picking' skill from any upstream differences.

For each observation you will see:
- the observation text and its kind (defect/upgrade)
- the candidate list (item_id, name, trade_bucket, score)
- each model's chosen_id (or null)

For EACH observation, determine the correct pick (or null if no candidate fits) and score each
model on:
- pick_correctness (did they pick what you would?)
- no_hallucinated_ids (they must only use item_ids from the candidate list; null is allowed)
- null_when_appropriate (did they return null when no candidate truly fits?)

You MUST also, per observation, return `judge_correct_id` (or null if no candidate fits) — this
will be written back into the 2d row as ground truth for diagnostic purposes.

Return a single JSON object — no prose, no markdown."""

_JUDGE_SYSTEM_2D_COUPLED = """\
You are an expert evaluating two anonymous models on the COUPLED 2c+2d chain — each model's own
2c labels fed its own 2d candidate pick. This measures real-world chain quality. The models are
labeled only as "Model A" and "Model B".

For each observation you will see:
- the observation text
- each model's from_2c label/kind (which determined which pool 2d searched in)
- each model's candidate list and chosen_id

Because the two models can produce different labels in 2c, they can even search different
candidate pools — a legitimate failure mode. Score each on:
- end_to_end_resolution_quality (final pick quality)
- label_kind_consistency (was the 2c label sensible and did the kind pass through cleanly?)
- final_pick_correctness (did they pick the right item OR appropriately return null?)

For EACH observation, also return your judge_correct_id (null allowed). This will be written
back into the 2d row for diagnostics.

Return a single JSON object — no prose, no markdown."""

_JUDGE_SYSTEM_COMPREHENSIVE = """\
You are an expert real estate photo analyst performing an END-TO-END evaluation of two anonymous
local VLM pipelines. The models are labeled only as "Model A" and "Model B" — you do not know
their identities and must not speculate. You will see the photo AND each model's FINAL
forward-set — the real issues (defects + upgrade candidates) each model's chain produced after
2a→2b→2c.

Use the PHOTO as ground truth. This is the question you are answering:
  "Given this photo, which chain produced a more useful final issue list for a contractor?"

Score each on a 0–5 scale:
- accuracy          → every item in the forward-set is actually visible / defensible from the photo
- completeness      → the chain caught the issues that really matter in this room
- hallucination_penalty → 5 = no fabricated claims; 0 = many fabrications
- specificity       → concrete location, material, extent (beats "walls need work")

You MUST also list:
- model_a_unique_good  — items Model A caught that Model B missed AND are truly present
- model_b_unique_good  — same for Model B
- model_a_hallucinations — items Model A claimed that are NOT visible / NOT defensible
- model_b_hallucinations — same for Model B
- missed_by_both       — issues clearly visible in the photo that neither chain surfaced

Return a single JSON object — no prose, no markdown."""


def _verdict_schema_for(skill: str) -> str:
    """Describe the expected output JSON schema for each skill.

    Uses Model A / Model B naming — the judge is blinded to model identity.
    Verdicts get canonicalized back to gemma/qwen immediately after parsing.
    """
    if skill == "2a":
        return """\
{
  "skill": "2a",
  "winner": "A" | "B" | "tie",
  "scores": {
    "A": {"accuracy": 0-5, "completeness": 0-5, "hallucination_penalty": 0-5, "specificity": 0-5},
    "B": {"accuracy": 0-5, "completeness": 0-5, "hallucination_penalty": 0-5, "specificity": 0-5}
  },
  "model_a_unique_good": ["..."],
  "model_b_unique_good": ["..."],
  "model_a_hallucinations": ["..."],
  "model_b_hallucinations": ["..."],
  "missed_by_both": ["..."],
  "rationale": "..."
}"""
    if skill == "2b":
        return """\
{
  "skill": "2b",
  "winner": "A" | "B" | "tie",
  "scores": {
    "A": {"faithfulness_to_input": 0-5, "schema_adherence": 0-5, "decomposition_quality": 0-5, "no_speculation": 0-5},
    "B": {"faithfulness_to_input": 0-5, "schema_adherence": 0-5, "decomposition_quality": 0-5, "no_speculation": 0-5}
  },
  "rationale": "..."
}"""
    if skill == "2c":
        return """\
{
  "skill": "2c",
  "winner": "A" | "B" | "tie",
  "scores": {
    "A": {"label_correctness": 0-5, "forward_set_purity": 0-5, "no_dimension_leakage": 0-5},
    "B": {"label_correctness": 0-5, "forward_set_purity": 0-5, "no_dimension_leakage": 0-5}
  },
  "rationale": "..."
}"""
    if skill == "2d_isolated":
        return """\
{
  "skill": "2d_isolated",
  "winner": "A" | "B" | "tie",
  "scores": {
    "A": {"pick_correctness": 0-5, "no_hallucinated_ids": 0-5, "null_when_appropriate": 0-5},
    "B": {"pick_correctness": 0-5, "no_hallucinated_ids": 0-5, "null_when_appropriate": 0-5}
  },
  "per_observation": [
    {
      "observation": "...",
      "judge_correct_id": "..." | null,
      "model_a_correct": true | false,
      "model_b_correct": true | false
    }
  ],
  "rationale": "..."
}"""
    if skill == "2c+2d_coupled":
        return """\
{
  "skill": "2c+2d_coupled",
  "winner": "A" | "B" | "tie",
  "scores": {
    "A": {"end_to_end_resolution_quality": 0-5, "label_kind_consistency": 0-5, "final_pick_correctness": 0-5},
    "B": {"end_to_end_resolution_quality": 0-5, "label_kind_consistency": 0-5, "final_pick_correctness": 0-5}
  },
  "per_observation": [
    {
      "observation": "...",
      "judge_correct_id": "..." | null,
      "model_a_correct": true | false,
      "model_b_correct": true | false,
      "failure_attribution": "2c" | "2d" | "both" | "none"
    }
  ],
  "rationale": "..."
}"""
    if skill == "comprehensive":
        return """\
{
  "skill": "comprehensive",
  "winner": "A" | "B" | "tie",
  "scores": {
    "A": {"accuracy": 0-5, "completeness": 0-5, "hallucination_penalty": 0-5, "specificity": 0-5},
    "B": {"accuracy": 0-5, "completeness": 0-5, "hallucination_penalty": 0-5, "specificity": 0-5}
  },
  "model_a_unique_good": ["..."],
  "model_b_unique_good": ["..."],
  "model_a_hallucinations": ["..."],
  "model_b_hallucinations": ["..."],
  "missed_by_both": ["..."],
  "rationale": "..."
}"""
    raise ValueError(f"Unknown skill: {skill}")


# ── Per-skill user prompt builders ───────────────────────────────────────────
#
# Each builder takes an `ab_mapping` dict like {"A": "gemma", "B": "qwen"} and
# arranges the presentation so the judge sees Model A first, then Model B,
# with no reference to gemma/qwen. The mapping is then used to canonicalize
# the judge's A/B-keyed response back into gemma/qwen keys.


def _cells_for(record: ImageRecord, model_key: str) -> ModelCells:
    return record.gemma if model_key == "gemma" else record.qwen


def _build_user_prompt_2a(record: ImageRecord, ab_mapping: Dict[str, str]) -> str:
    a_cells = _cells_for(record, ab_mapping["A"])
    b_cells = _cells_for(record, ab_mapping["B"])
    return (
        f"## Scene: {record.scene}\n\n"
        f"## Model A freeform observations:\n{a_cells.pass_2a or '(empty)'}\n\n"
        f"## Model B freeform observations:\n{b_cells.pass_2a or '(empty)'}\n\n"
        f"## Your task:\nScore each model. Return JSON matching:\n{_verdict_schema_for('2a')}"
    )


def _build_user_prompt_2b(record: ImageRecord, ab_mapping: Dict[str, str]) -> str:
    a_cells = _cells_for(record, ab_mapping["A"])
    b_cells = _cells_for(record, ab_mapping["B"])
    return (
        f"## Shared input freeform (identical for both models):\n"
        f"{record.fixture.pass_2a or '(empty)'}\n\n"
        f"## Model A structured observations:\n{json.dumps(a_cells.pass_2b, indent=2)}\n\n"
        f"## Model B structured observations:\n{json.dumps(b_cells.pass_2b, indent=2)}\n\n"
        f"## Your task:\nScore each model. Return JSON matching:\n{_verdict_schema_for('2b')}"
    )


def _build_user_prompt_2c(record: ImageRecord, ab_mapping: Dict[str, str]) -> str:
    a_cells = _cells_for(record, ab_mapping["A"])
    b_cells = _cells_for(record, ab_mapping["B"])
    return (
        f"## Shared input observations:\n{json.dumps(record.fixture.pass_2b, indent=2)}\n\n"
        f"## Model A labeled output:\n{json.dumps(a_cells.pass_2c, indent=2)}\n\n"
        f"## Model B labeled output:\n{json.dumps(b_cells.pass_2c, indent=2)}\n\n"
        f"## Your task:\nScore each model. Return JSON matching:\n{_verdict_schema_for('2c')}"
    )


def _twod_row_to_prompt_dict(row: TwoDRow) -> Dict[str, Any]:
    """Serialize a TwoDRow for inclusion in a judge prompt (strip judge-backfill fields)."""
    d = {
        "observation": row.observation,
        "kind": row.kind,
        "candidates": row.candidates,
        "chosen_id": row.chosen_id,
        "chosen_null": row.chosen_null,
    }
    if row.from_2c:
        d["from_2c"] = row.from_2c
    return d


def _build_user_prompt_2d_isolated(record: ImageRecord, ab_mapping: Dict[str, str]) -> str:
    a_cells = _cells_for(record, ab_mapping["A"])
    b_cells = _cells_for(record, ab_mapping["B"])
    a_rows = [_twod_row_to_prompt_dict(r) for r in a_cells.pass_2d_isolated]
    b_rows = [_twod_row_to_prompt_dict(r) for r in b_cells.pass_2d_isolated]
    return (
        f"## Shared input labels (identical for both models):\n"
        f"{json.dumps(record.fixture.pass_2c, indent=2)}\n\n"
        f"## Model A 2d decisions:\n{json.dumps(a_rows, indent=2)}\n\n"
        f"## Model B 2d decisions:\n{json.dumps(b_rows, indent=2)}\n\n"
        f"## Your task:\nScore each model and, per observation, provide judge_correct_id. "
        f"Return JSON matching:\n{_verdict_schema_for('2d_isolated')}"
    )


def _build_user_prompt_2d_coupled(record: ImageRecord, ab_mapping: Dict[str, str]) -> str:
    a_cells = _cells_for(record, ab_mapping["A"])
    b_cells = _cells_for(record, ab_mapping["B"])
    a_rows = [_twod_row_to_prompt_dict(r) for r in a_cells.pass_2c_2d_coupled]
    b_rows = [_twod_row_to_prompt_dict(r) for r in b_cells.pass_2c_2d_coupled]
    return (
        f"## Shared source observations:\n"
        f"{json.dumps(record.fixture.pass_2b, indent=2)}\n\n"
        f"## Model A coupled 2c+2d rows (each model used its OWN 2c):\n"
        f"{json.dumps(a_rows, indent=2)}\n\n"
        f"## Model B coupled 2c+2d rows:\n{json.dumps(b_rows, indent=2)}\n\n"
        f"## Your task:\nScore each model on the coupled chain quality. Attribute each failure "
        f"to 2c, 2d, both, or none. Return JSON matching:\n{_verdict_schema_for('2c+2d_coupled')}"
    )


def _forward_set(cells_2c: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter a 2c labeled output down to the 'real issues' forward-set."""
    return [
        r for r in (cells_2c or [])
        if r.get("label") in {"defect_or_damage", "upgrade_candidate"}
    ]


def _build_user_prompt_comprehensive(record: ImageRecord, ab_mapping: Dict[str, str]) -> str:
    """End-to-end: image + both models' final forward-sets (after their 2a→2b→2c chains)."""
    a_cells = _cells_for(record, ab_mapping["A"])
    b_cells = _cells_for(record, ab_mapping["B"])
    a_forward = _forward_set(a_cells.pass_2c)
    b_forward = _forward_set(b_cells.pass_2c)
    return (
        f"## Scene: {record.scene} (group: {record.scene_group})\n\n"
        f"## Model A final forward-set ({len(a_forward)} items):\n"
        f"{json.dumps(a_forward, indent=2)}\n\n"
        f"## Model B final forward-set ({len(b_forward)} items):\n"
        f"{json.dumps(b_forward, indent=2)}\n\n"
        f"## Your task:\nUse the PHOTO as ground truth. Score each chain on what it produced. "
        f"Call out uniquely-good items, hallucinations, and anything visible that BOTH missed. "
        f"Return JSON matching:\n{_verdict_schema_for('comprehensive')}"
    )


_SKILL_PROMPT_BUILDERS = {
    "2a": (_JUDGE_SYSTEM_2A, _build_user_prompt_2a, True),          # needs image
    "2b": (_JUDGE_SYSTEM_2B, _build_user_prompt_2b, False),
    "2c": (_JUDGE_SYSTEM_2C, _build_user_prompt_2c, False),
    "2d_isolated": (_JUDGE_SYSTEM_2D_ISOLATED, _build_user_prompt_2d_isolated, False),
    "2c+2d_coupled": (_JUDGE_SYSTEM_2D_COUPLED, _build_user_prompt_2d_coupled, False),
    "comprehensive": (_JUDGE_SYSTEM_COMPREHENSIVE, _build_user_prompt_comprehensive, True),  # needs image
}

# The "comprehensive" skill is orthogonal to the per-pass SKILLS. Kept out of SKILLS so
# the existing per-pass aggregation/recommendation stays clean.
COMPREHENSIVE_SKILL = "comprehensive"


async def run_judge_for_skill(
    skill: str,
    record: ImageRecord,
    vlm_client: VLMClient,
    judge_config: Dict[str, Any],
    ab_order: str = "random",
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Run a single per-skill judge call. Returns (verdict | None, parse_failure_count).

    The judge is blinded: it sees "Model A" / "Model B" with their order
    determined by `ab_order` ('random' | 'gemma_first' | 'qwen_first').
    Verdicts are canonicalized back to gemma/qwen keys before return.

    On a JSON parse failure, retries once with a correction prompt. The parse
    failure count (0, 1, or 2) is returned so callers can surface it.
    """
    if skill not in _SKILL_PROMPT_BUILDERS:
        raise ValueError(f"Unknown skill: {skill}")

    system_prompt, prompt_builder, needs_image = _SKILL_PROMPT_BUILDERS[skill]
    ab_mapping = _ab_mapping_for(ab_order, record.photo_key, skill)
    user_prompt = prompt_builder(record, ab_mapping)

    judge_cfg = {**judge_config, "max_tokens": 4000}

    async def _call(prompt: str) -> str:
        if needs_image:
            return await vlm_client.analyze_image(
                image_path=Path(record.image_path),
                system_prompt=system_prompt,
                user_prompt=prompt,
                **judge_cfg,
            )
        return await vlm_client.analyze_text(
            system_prompt=system_prompt,
            user_prompt=prompt,
            **judge_cfg,
        )

    parse_failures = 0

    try:
        response = await _call(user_prompt)
    except Exception as e:
        err_str = str(e)
        if any(kw in err_str.lower() for kw in ("insufficient_quota", "billing", "quota", "resourceexhausted")):
            raise  # let caller abort
        logger.error(f"Judge error (skill={skill}) for {record.photo_key}: {e}")
        return None, parse_failures

    try:
        verdict = extract_json_object(response)
    except (ValueError, json.JSONDecodeError) as e:
        parse_failures += 1
        logger.warning(
            f"Judge unparseable response for {record.photo_key}/{skill} "
            f"(attempt 1): {e}. Retrying once with correction prompt."
        )
        retry_prompt = (
            "Your previous response was not parseable as a JSON object. "
            "Return ONLY a single valid JSON object — no prose, no markdown fences, "
            "no leading or trailing text. Reattempt the original task now:\n\n"
            f"{user_prompt}"
        )
        try:
            response = await _call(retry_prompt)
            verdict = extract_json_object(response)
        except Exception as e2:
            err_str = str(e2)
            if any(kw in err_str.lower() for kw in ("insufficient_quota", "billing", "quota", "resourceexhausted")):
                raise
            parse_failures += 1
            logger.error(
                f"Judge retry also failed for {record.photo_key}/{skill}: {e2}"
            )
            return None, parse_failures

    if not isinstance(verdict, dict):
        return None, parse_failures

    # Canonicalize A/B -> gemma/qwen so the rest of the pipeline is unchanged.
    verdict = _canonicalize_verdict(verdict, ab_mapping)
    return verdict, parse_failures


def backfill_2d_judge_into_rows(
    rows: List[TwoDRow],
    verdict: Optional[Dict[str, Any]],
) -> None:
    """After a 2d judge runs, write judge_correct_id + rank back into each TwoDRow.

    Matches verdict per_observation entries to rows by observation text.
    Tries exact-match first; if that fails, retries with whitespace/case/
    trailing-punctuation normalization. Fuzzy matching is intentionally avoided
    since it can silently match wrong observations — only cheap deterministic
    normalization is applied.
    """
    if not verdict:
        return
    per_obs = verdict.get("per_observation")
    if not isinstance(per_obs, list):
        return

    by_desc: Dict[str, Dict[str, Any]] = {}
    by_desc_norm: Dict[str, Dict[str, Any]] = {}
    for entry in per_obs:
        if not isinstance(entry, dict):
            continue
        desc = (entry.get("observation") or "").strip()
        if not desc:
            continue
        by_desc[desc] = entry
        by_desc_norm[_norm_text(desc)] = entry

    for row in rows:
        row_desc = row.observation.strip()
        entry = by_desc.get(row_desc)
        if entry is None:
            entry = by_desc_norm.get(_norm_text(row_desc))
        if not entry:
            continue
        correct_id = entry.get("judge_correct_id")
        row.judge_correct_id = correct_id if correct_id else None

        if row.judge_correct_id:
            rank: Optional[int] = None
            for i, cand in enumerate(row.candidates, start=1):
                if cand.get("item_id") == row.judge_correct_id:
                    rank = i
                    break
            row.judge_correct_rank = rank
            row.judge_correct_in_candidates = rank is not None
        else:
            # Judge says null is correct (no candidate fits)
            row.judge_correct_rank = None
            row.judge_correct_in_candidates = False


# ---------------------------------------------------------------------------
# Aggregation + recommendation
# ---------------------------------------------------------------------------

def _sum_scores(verdicts: List[Dict[str, Any]], model_key: str) -> Dict[str, float]:
    """Average the per-criterion scores across verdicts for one model."""
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for v in verdicts:
        if not v or not isinstance(v.get("scores"), dict):
            continue
        m_scores = v["scores"].get(model_key)
        if not isinstance(m_scores, dict):
            continue
        for k, val in m_scores.items():
            try:
                totals[k] = totals.get(k, 0.0) + float(val)
                counts[k] = counts.get(k, 0) + 1
            except (TypeError, ValueError):
                continue
    return {k: round(totals[k] / counts[k], 3) for k in totals if counts.get(k, 0) > 0}


def aggregate_per_skill(records: List[ImageRecord], skill: str) -> Dict[str, Any]:
    """Count wins/ties and average scores for one skill."""
    verdicts = [r.judge[skill] for r in records if r.judge.get(skill)]
    gemma_wins = sum(1 for v in verdicts if v.get("winner") == "gemma")
    qwen_wins = sum(1 for v in verdicts if v.get("winner") == "qwen")
    ties = sum(1 for v in verdicts if v.get("winner") == "tie")

    return {
        "judged_images": len(verdicts),
        "gemma_wins": gemma_wins,
        "qwen_wins": qwen_wins,
        "ties": ties,
        "avg_scores": {
            "gemma": _sum_scores(verdicts, "gemma"),
            "qwen": _sum_scores(verdicts, "qwen"),
        },
    }


def _pick_winner(agg: Dict[str, Any]) -> str:
    if agg.get("judged_images", 0) == 0:
        return "tie"
    g = agg.get("gemma_wins", 0)
    q = agg.get("qwen_wins", 0)
    if g > q:
        return "gemma"
    if q > g:
        return "qwen"
    return "tie"


def build_recommendation(aggregate: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Derive suggested config from per-skill aggregates.

    Two recommendations are produced:
    - suggested_config_per_pass: independent winner per pass. 2c and 2d are
      allowed to split across models — if the isolated data says Gemma is best
      at 2c and Qwen is best at 2d, that's what gets surfaced. The user can
      decide whether to actually test that gemma-2c -> qwen-2d combination.
    - suggested_config_coupled_2cd: treats 2c+2d as an atomic unit (same model
      for both), based on the coupled judge. Simpler to deploy.

    The flaw-check signal (does isolated agree with coupled?) is exposed via
    the raw best_2c / best_2d_isolated / best_2c+2d_coupled fields and the
    winners_split flag — so the caller can see disagreement without the
    recommendation logic having to hide it.
    """
    best_2a = _pick_winner(aggregate["2a"])
    best_2b = _pick_winner(aggregate["2b"])
    best_2c = _pick_winner(aggregate["2c"])
    best_2d_iso = _pick_winner(aggregate["2d_isolated"])
    best_2cd_coupled = _pick_winner(aggregate["2c+2d_coupled"])

    winners = {best_2a, best_2b, best_2c, best_2d_iso, best_2cd_coupled} - {"tie"}
    winners_split = len(winners) > 1

    # Per-pass: every pass chooses independently. Ties fall back to gemma
    # (a harmless default; the user sees the tie in best_* fields anyway).
    suggested_per_pass = {
        "2a": best_2a if best_2a != "tie" else "gemma",
        "2b": best_2b if best_2b != "tie" else "gemma",
        "2c": best_2c if best_2c != "tie" else "gemma",
        "2d": best_2d_iso if best_2d_iso != "tie" else "gemma",
    }

    coupled_choice = best_2cd_coupled if best_2cd_coupled != "tie" else "gemma"
    suggested_coupled = {
        "2a": best_2a if best_2a != "tie" else "gemma",
        "2b": best_2b if best_2b != "tie" else "gemma",
        "2c_and_2d": coupled_choice,
    }

    # Flag when the two recommendations diverge on 2c/2d so the caller knows
    # to inspect the isolated-vs-coupled disagreement.
    per_pass_vs_coupled_divergence = (
        suggested_per_pass["2c"] != suggested_coupled["2c_and_2d"]
        or suggested_per_pass["2d"] != suggested_coupled["2c_and_2d"]
    )

    confidence = "high" if not winners_split and len(winners) == 1 else (
        "medium" if len(winners) == 2 else "low"
    )

    rationale_parts = [
        "The user picks between suggested_config_per_pass (maximally optimized, may use different "
        "models for different passes, possibly including a gemma-2c -> qwen-2d split) and "
        "suggested_config_coupled_2cd (treats 2c+2d as atomic, simpler to deploy) based on "
        "operational preference.",
    ]
    if per_pass_vs_coupled_divergence:
        rationale_parts.append(
            "NOTE: per-pass and coupled recommendations disagree on 2c/2d. This is a judge/flaw "
            "signal — the isolated 2c/2d judgments point one way, the coupled judgment points "
            "another. Inspect winners_split and the raw best_* fields before choosing."
        )

    return {
        "best_2a": best_2a,
        "best_2b": best_2b,
        "best_2c": best_2c,
        "best_2d_isolated": best_2d_iso,
        "best_2c+2d_coupled": best_2cd_coupled,
        "winners_split": winners_split,
        "per_pass_vs_coupled_divergence": per_pass_vs_coupled_divergence,
        "suggested_config_per_pass": suggested_per_pass,
        "suggested_config_coupled_2cd": suggested_coupled,
        "confidence": confidence,
        "rationale": " ".join(rationale_parts),
    }


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

def _record_to_dict(r: ImageRecord) -> Dict[str, Any]:
    return {
        "property_id": r.property_id,
        "photo_key": r.photo_key,
        "image_path": r.image_path,
        "scene": r.scene,
        "scene_group": r.scene_group,
        "fixture": asdict(r.fixture),
        "gemma": {
            "pass_2a": r.gemma.pass_2a,
            "pass_2b": r.gemma.pass_2b,
            "pass_2c": r.gemma.pass_2c,
            "pass_2d_isolated": [asdict(x) for x in r.gemma.pass_2d_isolated],
            "pass_2c_2d_coupled": [asdict(x) for x in r.gemma.pass_2c_2d_coupled],
        },
        "qwen": {
            "pass_2a": r.qwen.pass_2a,
            "pass_2b": r.qwen.pass_2b,
            "pass_2c": r.qwen.pass_2c,
            "pass_2d_isolated": [asdict(x) for x in r.qwen.pass_2d_isolated],
            "pass_2c_2d_coupled": [asdict(x) for x in r.qwen.pass_2c_2d_coupled],
        },
        "judge": r.judge,
        "judge_parse_failures": r.judge_parse_failures,
        "error": r.error,
    }


def _twod_rows_from_dicts(raw: Any) -> List[TwoDRow]:
    if not isinstance(raw, list):
        return []
    out: List[TwoDRow] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        out.append(TwoDRow(
            observation=entry.get("observation", ""),
            kind=entry.get("kind", "defect"),
            candidates=entry.get("candidates") or [],
            chosen_id=entry.get("chosen_id"),
            chosen_null=bool(entry.get("chosen_null", False)),
            from_2c=entry.get("from_2c"),
            judge_correct_id=entry.get("judge_correct_id"),
            judge_correct_rank=entry.get("judge_correct_rank"),
            judge_correct_in_candidates=entry.get("judge_correct_in_candidates"),
        ))
    return out


def _record_from_dict(d: Dict[str, Any]) -> ImageRecord:
    r = ImageRecord(
        property_id=d.get("property_id", ""),
        photo_key=d.get("photo_key", ""),
        image_path=d.get("image_path", ""),
        scene=d.get("scene", "other"),
        scene_group=d.get("scene_group", "other"),
    )
    fx = d.get("fixture") or {}
    r.fixture = FixtureCells(
        pass_2a=fx.get("pass_2a"),
        pass_2b=fx.get("pass_2b") or [],
        pass_2c=fx.get("pass_2c") or [],
    )
    for key in ("gemma", "qwen"):
        src = d.get(key) or {}
        target = ModelCells(
            pass_2a=src.get("pass_2a"),
            pass_2b=src.get("pass_2b") or [],
            pass_2c=src.get("pass_2c") or [],
            pass_2d_isolated=_twod_rows_from_dicts(src.get("pass_2d_isolated")),
            pass_2c_2d_coupled=_twod_rows_from_dicts(src.get("pass_2c_2d_coupled")),
        )
        setattr(r, key, target)
    r.judge = d.get("judge") or {k: None for k in SKILLS}
    r.judge_parse_failures = d.get("judge_parse_failures") or {}
    r.error = d.get("error")
    return r


def load_checkpoint(output_path: Path) -> Dict[str, ImageRecord]:
    ckpt = output_path.with_suffix(".checkpoint.json")
    if not ckpt.exists():
        return {}
    try:
        with open(ckpt, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {ckpt}: {e}")
        return {}

    out: Dict[str, ImageRecord] = {}
    for key, entry in (data or {}).items():
        if isinstance(entry, dict):
            out[key] = _record_from_dict(entry)
    logger.info(f"Loaded checkpoint with {len(out)} image records")
    return out


def save_checkpoint(output_path: Path, records: Dict[str, ImageRecord]) -> None:
    ckpt = output_path.with_suffix(".checkpoint.json")
    payload = {key: _record_to_dict(r) for key, r in records.items()}
    with open(ckpt, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _phase_complete(record: ImageRecord, phase: str, skip_skills: Optional[set] = None) -> bool:
    """Check whether a given phase has data for this record (used by --resume).

    Uses AND semantics: if the phase had an error, or if a required cell
    wasn't populated, it's incomplete and should re-run. Empty outputs
    (pass_2a="", pass_2b=[]) are treated as 'completed with empty result'
    when no error was recorded — this is a legitimate model behavior.

    `skip_skills` narrows which cells count as 'required'; if not provided,
    all non-skipped cells are assumed required.
    """
    skip_skills = skip_skills or set()

    if phase == "fixture":
        # Fixture is "done" even if empty (valid case: empty 2a → empty 2b/2c)
        return record.fixture.pass_2a is not None

    if phase in ("gemma", "qwen"):
        # If this phase errored mid-run, it's incomplete regardless of cell content.
        if record.error and record.error.startswith(f"{phase}:"):
            return False
        cells = record.gemma if phase == "gemma" else record.qwen
        # 2a is the only cell that distinguishes None (never ran) from "" (ran, empty).
        # If 2a wasn't skipped and is None, the phase clearly didn't complete.
        if "2a" not in skip_skills and cells.pass_2a is None:
            return False
        # 2b/2c/2d_isolated/2c+2d_coupled default to empty lists, so we cannot
        # distinguish "never ran" from "ran with empty result" at the cell level.
        # The error flag above is the authoritative signal for mid-phase failures.
        # This means a silent empty cell will look complete — acceptable because
        # any real failure sets record.error and flips this to False.
        return True

    if phase == "judge":
        skills_expected = [s for s in SKILLS if s not in skip_skills]
        return all(record.judge.get(s) is not None for s in skills_expected)

    raise ValueError(f"Unknown phase: {phase}")


# ---------------------------------------------------------------------------
# User prompts (model swap between Phase A and Phase B)
# ---------------------------------------------------------------------------

def _prompt_swap_model(target_model: str) -> None:
    """Ask user to swap LM Studio to the target model, then wait for Enter."""
    print("")
    print("=" * 60)
    print(f"  Swap LM Studio to: {target_model}")
    print("  Load the model in LM Studio, then press Enter to continue.")
    print("  (Ctrl-C to abort; progress is saved in the checkpoint.)")
    print("=" * 60)
    try:
        input("  Press Enter when ready... ")
    except EOFError:
        pass


# ---------------------------------------------------------------------------
# Phase runners (concurrency: Semaphore + gather, one task per image)
# ---------------------------------------------------------------------------

class _AbortPhase(Exception):
    """Raised inside a worker to cancel the rest of the current phase (e.g. quota)."""


async def _confirm_continue_or_exit(completed: int, total: int, phase_label: str) -> None:
    """Interactive safeguard: pause and wait for Enter before continuing.
    Ctrl-C cleanly stops with exit code 0 (checkpoint already saved)."""
    print("\n" + "─" * 60)
    print(f"  SAFEGUARD: {phase_label} -- {completed}/{total} images processed.")
    print("  Press Enter to continue, or Ctrl-C to stop (progress is saved).")
    print("─" * 60)
    try:
        await asyncio.to_thread(input)
    except (KeyboardInterrupt, EOFError):
        logger.info("User stopped via safeguard. Checkpoint saved; you can --resume later.")
        raise SystemExit(0)


async def _run_phase_0(
    images: List[Dict[str, Any]],
    records: Dict[str, ImageRecord],
    vlm_client: VLMClient,
    judge_config: Dict[str, Any],
    output_path: Path,
    save_ckpt,
    concurrency: int,
    confirm_every: int = 0,
) -> None:
    """Phase 0 worker pool. Each image runs Pass 2a→2b→2c with GPT-5.4 for fixtures.
    If `confirm_every > 0`, pauses for user confirmation after each batch of that many images."""
    sem = asyncio.Semaphore(max(1, concurrency))
    n = len(images)

    async def _worker(i: int, img: Dict[str, Any]) -> None:
        async with sem:
            key = f"{img['property_id']}/{img['photo_key']}"
            rec = records[key]
            if _phase_complete(rec, "fixture"):
                logger.info(f"  [{i}/{n}] [cached] {key}")
                return
            logger.info(f"  [{i}/{n}] fixture {key}")
            try:
                rec.fixture = await build_fixture_for_image(img, vlm_client, judge_config)
            except Exception as e:
                err_str = str(e)
                if any(kw in err_str.lower() for kw in ("quota", "billing", "resourceexhausted")):
                    logger.error(f"GPT-5.4 quota exceeded — aborting phase. ({e})")
                    raise _AbortPhase(str(e))
                logger.error(f"Fixture error for {key}: {e}")
                rec.error = f"fixture: {e}"
        await save_ckpt()

    batch_size = confirm_every if confirm_every > 0 else n
    completed = 0
    try:
        for start in range(0, n, batch_size):
            batch = list(enumerate(images, 1))[start:start + batch_size]
            tasks = [_worker(i, img) for i, img in batch]
            await asyncio.gather(*tasks)
            completed += len(batch)
            if completed < n and confirm_every > 0:
                await _confirm_continue_or_exit(completed, n, "Phase 0 (GPT-5.4 fixtures)")
    except _AbortPhase as e:
        logger.error(f"Phase 0 aborted: {e}")
        await save_ckpt()
        raise SystemExit(2)


async def _run_phase_local(
    phase_name: str,               # "gemma" or "qwen" (used for log lines)
    model_cells_attr: str,         # "gemma" or "qwen" (ImageRecord attribute name)
    model_config: Dict[str, Any],
    images: List[Dict[str, Any]],
    records: Dict[str, ImageRecord],
    vlm_client: VLMClient,
    retriever: CatalogEmbeddingsRetriever,
    skip_skills: set,
    save_ckpt,
    concurrency: int,
) -> None:
    """Phases A/B worker pool. Each image runs all 5 local-model skill cells."""
    sem = asyncio.Semaphore(max(1, concurrency))
    n = len(images)

    async def _worker(i: int, img: Dict[str, Any]) -> None:
        async with sem:
            key = f"{img['property_id']}/{img['photo_key']}"
            rec = records[key]
            if _phase_complete(rec, phase_name, skip_skills):
                logger.info(f"  [{i}/{n}] [cached] {key}")
                return
            logger.info(f"  [{i}/{n}] {phase_name}  {key}")
            try:
                cells = await run_local_model_cells(
                    img, rec.fixture, vlm_client, model_config, retriever, skip_skills,
                )
                setattr(rec, model_cells_attr, cells)
            except Exception as e:
                logger.error(f"{phase_name} phase error for {key}: {e}")
                rec.error = f"{phase_name}: {e}"
        await save_ckpt()

    tasks = [_worker(i, img) for i, img in enumerate(images, 1)]
    await asyncio.gather(*tasks)


async def _run_phase_judge(
    records: Dict[str, ImageRecord],
    vlm_client: VLMClient,
    judge_config: Dict[str, Any],
    skip_skills: set,
    judge_delay: float,
    save_ckpt,
    concurrency: int,
    include_comprehensive: bool = False,
    confirm_every: int = 0,
    ab_order: str = "random",
) -> None:
    """Phase D worker pool. Each image runs its per-skill judge calls sequentially
    within its worker; concurrency is across images.

    If `include_comprehensive` is True, the 'comprehensive' skill is appended to the
    active skills (image + both models' final forward-sets, end-to-end judgment).

    If `confirm_every > 0`, pauses for user confirmation after each batch.

    `ab_order` ('random' | 'gemma_first' | 'qwen_first') controls how the judge
    sees Model A vs Model B. Use forced orders for bias-check workflows.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    active_skills = [s for s in SKILLS if s not in skip_skills]
    if include_comprehensive and COMPREHENSIVE_SKILL not in skip_skills:
        active_skills.append(COMPREHENSIVE_SKILL)

    records_list = list(records.items())
    n = len(records_list)

    async def _worker(i: int, key: str, rec: ImageRecord) -> None:
        async with sem:
            # Local "done" check that accounts for comprehensive (which isn't in SKILLS).
            if all(rec.judge.get(s) is not None for s in active_skills):
                logger.info(f"  [{i}/{n}] [cached] {key}")
                return
            if rec.error:
                logger.info(f"  [{i}/{n}] [skip error] {key} ({rec.error})")
                return

            for skill in active_skills:
                if rec.judge.get(skill) is not None:
                    continue
                logger.info(f"  [{i}/{n}] judge/{skill} {key}")
                try:
                    verdict, parse_failures = await run_judge_for_skill(
                        skill, rec, vlm_client, judge_config, ab_order=ab_order,
                    )
                except Exception as e:
                    err_str = str(e)
                    if any(kw in err_str.lower() for kw in ("quota", "billing", "resourceexhausted")):
                        logger.error(f"  Judge aborted (quota) for {key}/{skill}: {e}")
                        raise _AbortPhase(str(e))
                    logger.error(f"  Judge error {key}/{skill}: {e}")
                    continue
                rec.judge[skill] = verdict
                if parse_failures > 0:
                    rec.judge_parse_failures[skill] = parse_failures

                # Backfill judge ground-truth into 2d rows
                if skill == "2d_isolated":
                    backfill_2d_judge_into_rows(rec.gemma.pass_2d_isolated, verdict)
                    backfill_2d_judge_into_rows(rec.qwen.pass_2d_isolated, verdict)
                elif skill == "2c+2d_coupled":
                    backfill_2d_judge_into_rows(rec.gemma.pass_2c_2d_coupled, verdict)
                    backfill_2d_judge_into_rows(rec.qwen.pass_2c_2d_coupled, verdict)

                if judge_delay > 0:
                    await asyncio.sleep(judge_delay)
        await save_ckpt()

    batch_size = confirm_every if confirm_every > 0 else n
    completed = 0
    try:
        for start in range(0, n, batch_size):
            batch = list(enumerate(records_list, 1))[start:start + batch_size]
            tasks = [_worker(i, key, rec) for i, (key, rec) in batch]
            await asyncio.gather(*tasks)
            completed += len(batch)
            if completed < n and confirm_every > 0:
                await _confirm_continue_or_exit(completed, n, "Phase D (Judge)")
    except _AbortPhase as e:
        logger.error(f"Phase D aborted: {e}")
        await save_ckpt()
        raise SystemExit(2)


async def async_main(args: argparse.Namespace) -> None:
    start_time = time.time()

    # ── Resolve model configs ────────────────────────────────────────────
    qwen_cfg, gpt_cfg = get_model_configs_from_pipeline_config(cfg)

    gemma_config = {
        "model": args.gemma_model,
        "url": args.lm_studio_url,
        "provider": "lmstudio",
    }
    qwen_config = {
        "model": args.qwen_model,
        "url": args.lm_studio_url,
        "provider": "lmstudio",
    }

    judge_model = args.judge_model or gpt_cfg.get("model", "gpt-5.4")
    judge_config = {
        "model": judge_model,
        "api_key": gpt_cfg.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
        "provider": "openai",
    }

    logger.info(f"Gemma model:  {gemma_config['model']}")
    logger.info(f"Qwen model:   {qwen_config['model']}")
    logger.info(f"Judge model:  {judge_config['model']}")
    logger.info(f"LM Studio URL: {gemma_config['url']}")

    # ── Discover images ──────────────────────────────────────────────────
    images = discover_images(
        artifacts_dir=Path(args.artifacts_dir),
        images_base=Path(args.images_base),
        max_images=args.max_images,
        property_filter=args.property,
    )
    if not images:
        logger.error("No images found.")
        return

    # ── Setup clients + retriever ────────────────────────────────────────
    vlm_client = create_vlm_client()

    logger.info(f"Loading catalog from {cfg.ISSUE_CATALOG_PATH}")
    with open(cfg.ISSUE_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    logger.info("Building catalog embeddings retriever...")
    retriever = CatalogEmbeddingsRetriever(
        catalog,
        model_name=cfg.EMBEDDINGS_MODEL_NAME,
        device=cfg.EMBEDDINGS_DEVICE,
        trust_remote_code=getattr(cfg, "EMBEDDINGS_TRUST_REMOTE_CODE", False),
        default_topk=cfg.EMBEDDINGS_TOPK,
        guardrails=build_guardrails_from_catalog(catalog),
    )

    # ── Load checkpoint / initialize records ─────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_checkpoint(output_path) if args.resume else {}
    records: Dict[str, ImageRecord] = {}
    for img in images:
        key = f"{img['property_id']}/{img['photo_key']}"
        if key in existing:
            records[key] = existing[key]
        else:
            records[key] = ImageRecord(
                property_id=img["property_id"],
                photo_key=img["photo_key"],
                image_path=img["image_path"],
                scene=img["scene"],
                scene_group=img["scene_group"],
            )

    # --force-rejudge: wipe saved verdicts from checkpoint so Phase D re-runs.
    # This is the core of the bias-check workflow: rerun only Phase D with a
    # different --ab-order (e.g. --ab-order qwen_first) against the same cells.
    if args.force_rejudge:
        cleared = 0
        for rec in records.values():
            for s in list(rec.judge.keys()):
                if rec.judge[s] is not None:
                    cleared += 1
                rec.judge[s] = None
            # Also clear any parse-failure counts from the prior judge run.
            rec.judge_parse_failures = {}
            # Also clear any judge-backfilled fields on 2d rows so they reflect
            # the new run, not the prior one.
            for row in rec.gemma.pass_2d_isolated + rec.qwen.pass_2d_isolated \
                    + rec.gemma.pass_2c_2d_coupled + rec.qwen.pass_2c_2d_coupled:
                row.judge_correct_id = None
                row.judge_correct_rank = None
                row.judge_correct_in_candidates = None
        logger.info(f"--force-rejudge: cleared {cleared} prior judge verdicts across "
                    f"{len(records)} records; Phase D will re-run.")

    skip_skills: set = set(args.skip_skills)

    # ── Shared checkpoint lock (defensive: json.dump under asyncio is atomic,
    #    but the lock protects against future refactors + clarifies intent) ──
    ckpt_lock = asyncio.Lock()

    async def _save_ckpt() -> None:
        async with ckpt_lock:
            save_checkpoint(output_path, records)

    # ── Phase 0: GPT-5.4 fixtures (concurrency=args.concurrency) ─────────
    print("\n" + "=" * 60)
    print(f"Phase 0: GPT-5.4 fixture generation (concurrency={args.concurrency})")
    if args.confirm_every > 0:
        print(f"  SAFEGUARD: will pause for confirmation every {args.confirm_every} images.")
    print("=" * 60)
    await _run_phase_0(
        images=images,
        records=records,
        vlm_client=vlm_client,
        judge_config=judge_config,
        output_path=output_path,
        save_ckpt=_save_ckpt,
        concurrency=args.concurrency,
        confirm_every=args.confirm_every,
    )

    # ── Phase A: Gemma (concurrency=args.local_concurrency) ──────────────
    _phase_needed = any(not _phase_complete(records[k], "gemma", skip_skills) for k in records)
    if _phase_needed and not args.skip_gemma:
        _prompt_swap_model(gemma_config["model"])
        print("\n" + "=" * 60)
        print(f"Phase A: Gemma ({gemma_config['model']}) "
              f"(local-concurrency={args.local_concurrency})")
        print("=" * 60)
        await _run_phase_local(
            phase_name="gemma",
            model_cells_attr="gemma",
            model_config=gemma_config,
            images=images,
            records=records,
            vlm_client=vlm_client,
            retriever=retriever,
            skip_skills=skip_skills,
            save_ckpt=_save_ckpt,
            concurrency=args.local_concurrency,
        )

    # ── Phase B: Qwen (concurrency=args.local_concurrency) ───────────────
    _phase_needed = any(not _phase_complete(records[k], "qwen", skip_skills) for k in records)
    if _phase_needed and not args.skip_qwen:
        _prompt_swap_model(qwen_config["model"])
        print("\n" + "=" * 60)
        print(f"Phase B: Qwen ({qwen_config['model']}) "
              f"(local-concurrency={args.local_concurrency})")
        print("=" * 60)
        await _run_phase_local(
            phase_name="qwen",
            model_cells_attr="qwen",
            model_config=qwen_config,
            images=images,
            records=records,
            vlm_client=vlm_client,
            retriever=retriever,
            skip_skills=skip_skills,
            save_ckpt=_save_ckpt,
            concurrency=args.local_concurrency,
        )

    # ── Phase D: Judge (concurrency=args.concurrency) ────────────────────
    if not args.skip_judge:
        print("\n" + "=" * 60)
        print(f"Phase D: Judge ({judge_config['model']}) "
              f"(concurrency={args.concurrency})")
        print(f"  Judge is BLINDED: models shown as 'Model A' / 'Model B' "
              f"(ab_order={args.ab_order})")
        if args.comprehensive_judge:
            print("  Comprehensive end-to-end judge: ENABLED (+1 judge call per image)")
        if args.confirm_every > 0:
            print(f"  SAFEGUARD: will pause for confirmation every {args.confirm_every} images.")
        print("=" * 60)
        await _run_phase_judge(
            records=records,
            vlm_client=vlm_client,
            judge_config=judge_config,
            skip_skills=skip_skills,
            judge_delay=args.judge_delay,
            save_ckpt=_save_ckpt,
            concurrency=args.concurrency,
            include_comprehensive=args.comprehensive_judge,
            confirm_every=args.confirm_every,
            ab_order=args.ab_order,
        )

    # ── Aggregate + write final report ───────────────────────────────────
    records_list = list(records.values())
    aggregate = {skill: aggregate_per_skill(records_list, skill) for skill in SKILLS}
    recommendation = build_recommendation(aggregate)

    # Comprehensive aggregate is optional and kept separate from per-pass aggregates.
    comprehensive_agg = None
    if args.comprehensive_judge:
        comprehensive_agg = aggregate_per_skill(records_list, COMPREHENSIVE_SKILL)

    # Roll up judge parse failures across images, grouped by skill.
    parse_failures_by_skill: Dict[str, int] = {}
    for rec in records_list:
        for skill, cnt in (rec.judge_parse_failures or {}).items():
            try:
                parse_failures_by_skill[skill] = parse_failures_by_skill.get(skill, 0) + int(cnt)
            except (TypeError, ValueError):
                continue
    parse_failures_total = sum(parse_failures_by_skill.values())

    # Images where a judge call returned None (verdict is missing from aggregation).
    # This is the "silent dropout" counter — images lost to unrecoverable judge errors.
    missing_verdicts_by_skill: Dict[str, int] = {}
    active_skills_all = list(SKILLS) + ([COMPREHENSIVE_SKILL] if args.comprehensive_judge else [])
    for skill in active_skills_all:
        if skill in skip_skills:
            continue
        missing = sum(1 for r in records_list if r.judge.get(skill) is None and not r.error)
        if missing:
            missing_verdicts_by_skill[skill] = missing

    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "property_id": args.property,
            "n_images": len(records_list),
            "gemma_model": gemma_config["model"],
            "qwen_model": qwen_config["model"],
            "judge_model": judge_config["model"],
            "fixture_source": judge_config["model"],
            "skills_tested": [s for s in SKILLS if s not in skip_skills],
            "comprehensive_judge": args.comprehensive_judge,
            "ab_order": args.ab_order,
            "force_rejudge": bool(args.force_rejudge),
            "judge_parse_failures_total": parse_failures_total,
            "judge_parse_failures_by_skill": parse_failures_by_skill,
            "missing_verdicts_by_skill": missing_verdicts_by_skill,
        },
        "images": [_record_to_dict(r) for r in records_list],
        "aggregate": aggregate,
        "recommendation": recommendation,
    }
    if comprehensive_agg is not None:
        report["comprehensive"] = comprehensive_agg

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Images:          {len(records_list)}")
    for skill in SKILLS:
        if skill in skip_skills:
            continue
        agg = aggregate[skill]
        print(f"  {skill:20s} gemma={agg['gemma_wins']} qwen={agg['qwen_wins']} ties={agg['ties']} (judged {agg['judged_images']})")
    if comprehensive_agg is not None:
        print(f"  {'comprehensive':20s} gemma={comprehensive_agg['gemma_wins']} "
              f"qwen={comprehensive_agg['qwen_wins']} ties={comprehensive_agg['ties']} "
              f"(judged {comprehensive_agg['judged_images']})")
    print("")
    print(f"Best 2a:             {recommendation['best_2a']}")
    print(f"Best 2b:             {recommendation['best_2b']}")
    print(f"Best 2c:             {recommendation['best_2c']}")
    print(f"Best 2d (isolated):  {recommendation['best_2d_isolated']}")
    print(f"Best 2c+2d coupled:  {recommendation['best_2c+2d_coupled']}")
    if comprehensive_agg is not None:
        print(f"Best comprehensive:  {_pick_winner(comprehensive_agg)}")
    print(f"Winners split:       {recommendation['winners_split']}")
    print(f"Confidence:          {recommendation['confidence']}")
    print(f"AB order:            {args.ab_order}")
    if parse_failures_total:
        print(f"Judge parse retries: {parse_failures_total} "
              f"(by skill: {parse_failures_by_skill})")
    if missing_verdicts_by_skill:
        print(f"Missing verdicts:    {missing_verdicts_by_skill} "
              f"(silent judge dropouts — review report.json)")
    print(f"Report saved to:     {output_path}")
    print(f"Elapsed:             {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model comparison — head-to-head Gemma 4 vs Qwen 3.6 across pipeline passes.",
    )
    parser.add_argument(
        "--property",
        default=None,
        help="Filter to a single property ID (e.g. redfin_126224899).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(DEFAULT_ARTIFACTS_DIR),
        help="Path to artifacts directory (default: %(default)s).",
    )
    parser.add_argument(
        "--images-base",
        default=str(DEFAULT_IMAGES_BASE),
        help="Base path for property images (default: %(default)s).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: model_comparison_<property>_<timestamp>.json).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap on images processed.",
    )
    parser.add_argument(
        "--gemma-model",
        default=os.environ.get("GEMMA_MODEL", "unsloth/gemma-4-26b-a4b-it"),
        help="Gemma model id in LM Studio (env: GEMMA_MODEL).",
    )
    parser.add_argument(
        "--qwen-model",
        default=os.environ.get("QWEN_MODEL", "qwen/qwen3.6-27b"),
        help="Qwen model id in LM Studio (env: QWEN_MODEL).",
    )
    parser.add_argument(
        "--lm-studio-url",
        default=cfg.LM_STUDIO_URL,
        help=f"LM Studio URL (default: %(default)s).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model (default: cfg.GPT_MODEL).",
    )
    parser.add_argument(
        "--judge-delay",
        type=float,
        default=2.0,
        help="Seconds between judge calls (rate limiting). Default: 2.0",
    )
    parser.add_argument(
        "--skip-skills",
        nargs="*",
        default=[],
        choices=list(SKILLS),
        help="Skills to skip entirely (cells + judge).",
    )
    parser.add_argument("--skip-gemma", action="store_true", help="Skip Phase A (Gemma).")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Phase B (Qwen).")
    parser.add_argument("--skip-judge", action="store_true", help="Skip Phase D (Judge).")
    parser.add_argument(
        "--comprehensive-judge",
        action="store_true",
        help="Add an end-to-end 6th judge call per image that sees the IMAGE plus both "
             "models' final forward-sets (after full 2a->2b->2c chains). Captures "
             "'who missed what that is visible in the photo' more directly than per-pass judges.",
    )
    parser.add_argument(
        "--confirm-every",
        type=int,
        default=0,
        metavar="N",
        help="Safeguard against runaway API spend: after every N images are processed in "
             "an API phase (Phase 0 and Phase D), pause and require Enter to continue. "
             "Ctrl-C cleanly stops (checkpoint preserved; --resume picks up). "
             "Default 0 = no pause. Typical value: 20.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if present.")
    parser.add_argument(
        "--ab-order",
        choices=("random", "gemma_first", "qwen_first"),
        default="random",
        help="How the judge sees Model A vs Model B (always blinded — 'gemma'/'qwen' never "
             "appear in judge prompts). 'random' assigns a deterministic random order per "
             "(photo, skill) so reruns are reproducible. Use 'gemma_first' or 'qwen_first' "
             "for bias-check runs: run once with one forced order, then --resume "
             "--force-rejudge --ab-order <opposite> to re-judge the same cells with reversed "
             "positions. Large disagreement between the two = position bias is real.",
    )
    parser.add_argument(
        "--force-rejudge",
        action="store_true",
        help="On --resume, wipe existing judge verdicts so Phase D re-runs from scratch "
             "(cells/fixtures are preserved). Intended for the bias-check workflow: pair "
             "with a new --ab-order to re-judge the same cells with swapped positions.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent images for API phases (0 and D). Default: 3. "
             "Mirrors catalog_auditor's default.",
    )
    parser.add_argument(
        "--local-concurrency",
        type=int,
        default=3,
        help="Max concurrent images for local LM Studio phases (A and B). Default: 3. "
             "Lower to 1 if LM Studio KV-cache-corrupts under load on your setup.",
    )

    args = parser.parse_args()

    if not args.output:
        stem = args.property or "all"
        # On --resume without an explicit --output, reuse the most recent
        # checkpoint matching this property stem. Otherwise every resumed run
        # gets a fresh timestamp and load_checkpoint finds nothing.
        if args.resume:
            prefix = f"model_comparison_{stem}_"
            cwd = Path(".")
            candidates = sorted(
                cwd.glob(f"{prefix}*.checkpoint.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                args.output = str(candidates[0]).replace(".checkpoint.json", ".json")
                logger.info(f"--resume: reusing checkpoint {candidates[0]}")
            else:
                logger.warning(
                    f"--resume set but no checkpoint found matching {prefix}*.checkpoint.json "
                    f"in cwd; starting fresh with a new timestamped output."
                )
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                args.output = f"model_comparison_{stem}_{ts}.json"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"model_comparison_{stem}_{ts}.json"

    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
