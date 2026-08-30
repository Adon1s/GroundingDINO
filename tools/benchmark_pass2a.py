"""Pass 2a prompt-ablation benchmark: resumable, dataset-independent.

Measures whether inventory-framed Pass 2a wording beats the salience-framed
production prompt on run-to-run stability, per
docs/HANDOFF_pass2a_variance_ablation.md. Measurement only — nothing here
ships a prompt.

Stages (each resumable, each fingerprint-guarded):

  init         build benchmarks/pass2a-prompt/manifest.json from the frozen
               source artifacts (image hashes, scenes, property metadata,
               frozen Pass 2a captures)
  attribution  replay 2b->2c->2d->pre-2f costing k times per property from the
               FROZEN 2a captures; evaluate the material-flip stop gate
  scene-capture  freeze Terra Pass 1a scenes once per photo (v3 lineage);
               --check validates provenance/routing without any VLM call
  run          k full repeats per photo for one prompt variant, replaying the
               frozen scene capture (pass_1a_frozen_scene) into runs/v3/
  legacy-rescore  offline rescore of the frozen v2 artifacts with the
               repaired projection + canonical-room scorer (no VLM calls);
               output confined to runs/legacy_v2_rescore/
  match        blinded TEXT-ONLY alignment of every generated claim to the
               human gold, one call per variant-repeat per photo
  review       export the match to one editable CSV / import the decisions back
  repin-gold   acknowledge an edit to the human gold (invalidates match stages)
  report       stability metrics plus the helped/tied/hurt verdict per property
  judge        LEGACY, archived: the image-based gpt-5.6-sol round that the
               match/review stages replace. Kept only to read old artifacts.

Run from the pass2a_prompt_bench worktree (functional baseline cdefc2b):

  .venv/Scripts/python.exe tools/benchmark_pass2a.py init
  .venv/Scripts/python.exe tools/benchmark_pass2a.py attribution
  .venv/Scripts/python.exe tools/benchmark_pass2a.py run --variant baseline
  .venv/Scripts/python.exe tools/benchmark_pass2a.py match --round baseline_vs_checklist
  .venv/Scripts/python.exe tools/benchmark_pass2a.py review export --round baseline_vs_checklist
  .venv/Scripts/python.exe tools/benchmark_pass2a.py review import --round baseline_vs_checklist --csv <edited.csv>
  .venv/Scripts/python.exe tools/benchmark_pass2a.py report

Scoring counts human gold conditions covered minus human-confirmed unsupported
additions. A prompt is never judged on unsupported *rate*, which rewards
terseness; downstream filtering and catalog resolution are diagnostic only.

Dollar figures produced here are "pre-2f, all packages assumed confirmed"
benchmark totals, NOT production headlines: Pass 2f never runs, and every
inferred package is force-confirmed so package-member line items price
instead of being zeroed by the not_run gate (see
tools/rehab_packages.py apply_package_verifications_to_candidates).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import logging
import os
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks" / "pass2a-prompt"
MANIFEST_PATH = BENCH_DIR / "manifest.json"
PROMPTS_PATH = BENCH_DIR / "prompts.json"
CONFIG_PATH = BENCH_DIR / "config.json"
GOLD_PATH = BENCH_DIR / "gold" / "reference.json"
RUNS_DIR = BENCH_DIR / "runs"
# v3 artifact lineage: controlled Terra Pass 1a via frozen scene replay.
# Everything under runs/v3/ is produced by the repaired harness; the flat
# runs/* stage dirs are the frozen v2 lineage (read-only legacy evidence).
V3_DIR = RUNS_DIR / "v3"

logger = logging.getLogger("benchmark_pass2a")

_ENV_READY = False


def _setup_env() -> None:
    """Match the canary candidate-side environment BEFORE tools imports.

    KIND_ONTOLOGY_VERSION drives cfg.ISSUE_CATALOG_PATH (v2 catalog) and
    cfg.PIPELINE_MODE (publish); stale .env ARTIFACTS_ROOT / ISSUE_CATALOG_PATH
    must not leak in (same discipline as scripts/run_kind_canary.py).
    """
    global _ENV_READY
    if _ENV_READY:
        return
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    os.environ.pop("ARTIFACTS_ROOT", None)
    os.environ.pop("ISSUE_CATALOG_PATH", None)
    os.environ["KIND_ONTOLOGY_VERSION"] = "observation_kind_v2"
    # The gold and the paid Pass 2f artifacts are v4/2f-bound; an inherited
    # `new` mode would silently add v5 work to every benchmark photo
    # (tools/artifact_writers.py branches on it), so pin rather than trust
    # the default. RENOVATION_TERRA_USAGE_ROOT is deliberately NOT set:
    # benchmark live runs debit the shared production Terra/Sol daily
    # ledgers (Steven, 2026-08-30).
    os.environ["RENOVATION_ARCHITECTURE_MODE"] = "current"
    # Per-pass output caps must be exported BEFORE tools.pipeline_config is
    # imported (it snapshots the env at import time). Inventory-framed 2a
    # wording overflows the production 2000-token cap.
    if CONFIG_PATH.is_file():
        caps = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get(
            "openai_max_output_tokens") or {}
        for pass_key, cap in caps.items():
            os.environ[f"OPENAI_PASS_{pass_key.upper()}_MAX_TOKENS"] = str(int(cap))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _ENV_READY = True


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    from tools.comparison_common import atomic_json
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Config / prompts / manifest
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    return _load_json(CONFIG_PATH)


def load_prompts() -> Dict[str, Dict[str, str]]:
    """Return {variant: {"text": ..., "sha256": ...}} with hashes recomputed."""
    from tools.comparison_common import sha256_bytes
    raw = _load_json(PROMPTS_PATH)
    out: Dict[str, Dict[str, str]] = {}
    for name, entry in raw.items():
        text = entry["text"] if isinstance(entry, dict) else str(entry)
        out[name] = {"text": text, "sha256": sha256_bytes(text.encode("utf-8"))}
    return out


def load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise SystemExit("manifest.json missing — run the init stage first")
    return _load_json(MANIFEST_PATH)


def manifest_photos(manifest: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Flatten to [(property_key, photo_record), ...] in stable order."""
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for prop_key in sorted(manifest["properties"]):
        for photo in manifest["properties"][prop_key]["photos"]:
            rows.append((prop_key, photo))
    return rows


def _manifest_hashes(manifest: Dict[str, Any], field: str) -> Dict[str, str]:
    return {
        f"{prop}/{photo['photo_key']}": photo[field]
        for prop, photo in manifest_photos(manifest)
    }


def compute_fingerprint(
    manifest: Dict[str, Any],
    config: Dict[str, Any],
    prompt_sha: Optional[str],
    *,
    scene_capture_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Everything that must be identical for a resume to be valid.

    scene_capture_sha pins the frozen Pass 1a scene capture (v3 lineage);
    None marks a legacy-v2-style run without controlled scenes. Together with
    pass_2c_prompt_version and image_detail it makes v2 and v3 artifacts
    mutually non-resumable.
    """
    from tools.comparison_common import sha256_canonical, sha256_file
    from tools.scene_classifier_passes import PASS_2C_PROMPT_VERSION
    from tools import pipeline_config as cfg
    return {
        "git_head": _git_head(),
        "image_hashes_sha": sha256_canonical(
            _manifest_hashes(manifest, "image_sha256")),
        "frozen_2a_sha": sha256_canonical(
            _manifest_hashes(manifest, "frozen_2a_sha256")),
        "scene_capture_sha": scene_capture_sha,
        "prompt_sha256": prompt_sha,
        "model_overrides": config["model_overrides"],
        "reasoning_efforts": config["reasoning_efforts"],
        "pass_toggles": config["pass_toggles"],
        "openai_max_output_tokens": config.get("openai_max_output_tokens") or {},
        "catalog_sha256": sha256_file(Path(cfg.ISSUE_CATALOG_PATH)),
        "embeddings_model": cfg.EMBEDDINGS_MODEL_NAME,
        "pass_2c_prompt_version": PASS_2C_PROMPT_VERSION,
        "image_detail": "original",
        "pipeline_mode": "publish",
        "repeats": config["repeats"],
    }


def guard_fingerprint(stage_dir: Path, fingerprint: Dict[str, Any]) -> None:
    """First run writes the fingerprint; a resume must match it exactly."""
    fp_path = stage_dir / "fingerprint.json"
    if fp_path.is_file():
        stored = _load_json(fp_path)
        if stored != fingerprint:
            diffs = sorted(
                k for k in set(stored) | set(fingerprint)
                if stored.get(k) != fingerprint.get(k)
            )
            raise SystemExit(
                f"resume rejected: fingerprint mismatch in {stage_dir} "
                f"(differs on: {diffs}). Start a fresh stage directory or "
                "restore the original code/config."
            )
        return
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(fp_path, fingerprint)


# ---------------------------------------------------------------------------
# init — build the manifest from the frozen source artifacts
# ---------------------------------------------------------------------------

def _latest_run_dir(prop_dir: Path) -> Path:
    runs = sorted(
        (d for d in prop_dir.iterdir()
         if d.is_dir() and (d / "photo_intel_debug.json").is_file()),
        key=lambda d: d.name, reverse=True,
    )
    if not runs:
        raise SystemExit(f"no run with photo_intel_debug.json under {prop_dir}")
    return runs[0]


def stage_init(config: Dict[str, Any]) -> None:
    from tools.comparison_common import sha256_bytes, sha256_file
    images_root = Path(config["images_root"])
    source_root = Path(config["source_artifacts_root"])
    properties: Dict[str, Any] = {}
    for prop_key in config["properties"]:
        run_dir = _latest_run_dir(source_root / prop_key)
        artifact = _load_json(run_dir / "photo_intel_debug.json")
        photos_map = artifact.get("photos") or {}
        photos: List[Dict[str, Any]] = []
        for photo_key in sorted(photos_map):
            record = photos_map[photo_key]
            frozen = ((record.get("features") or {}).get("observations_freeform") or "").strip()
            if not frozen:
                raise SystemExit(
                    f"{prop_key}/{photo_key}: no observations_freeform in "
                    f"{run_dir} — frozen 2a capture is required"
                )
            image_path = images_root / prop_key / photo_key
            if not image_path.is_file():
                raise SystemExit(f"image missing on disk: {image_path}")
            photos.append({
                "photo_key": photo_key,
                "image_sha256": sha256_file(image_path),
                "scene": ((record.get("scene") or {}).get("id")) or "unknown",
                "frozen_2a": frozen,
                "frozen_2a_sha256": sha256_bytes(frozen.encode("utf-8")),
            })
        properties[prop_key] = {
            "photos": photos,
            "property_metadata": artifact.get("property_metadata") or {},
            "source_run_dir": str(run_dir),
            "source_catalog_version": artifact.get("catalog_version"),
        }
    manifest = {
        "benchmark": "pass2a-prompt",
        "created_at": _utcnow(),
        "images_root": str(images_root),
        "properties": properties,
        "photo_count": sum(len(p["photos"]) for p in properties.values()),
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"manifest written: {manifest['photo_count']} photos across "
          f"{len(properties)} properties -> {MANIFEST_PATH}")


# ---------------------------------------------------------------------------
# Pipeline context (catalog, embeddings provider, orchestrator, client)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PipelineContext:
    cfg: Any
    catalog: Dict[str, Any]
    orchestrator: Any
    vlm_client: Any
    gpt5_config: Dict[str, Any]
    options_base: Any  # SceneClassifierRunOptions


def preflight_embeddings() -> bool:
    """Real-POST probe; /health lies when the GPU device is lost."""
    import urllib.request
    from tools import pipeline_config as cfg
    url = f"{cfg.EMBEDDINGS_BASE_URL.rstrip('/')}/embeddings"
    payload = {"input": ["preflight: worn roof shingles"],
               "model": cfg.EMBEDDINGS_MODEL_NAME}
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
        dim = len((body.get("data") or [{}])[0].get("embedding") or [])
        if dim <= 0:
            print(f"preflight: embeddings returned no vector from {url}",
                  file=sys.stderr)
            return False
        print(f"preflight: embeddings OK ({cfg.EMBEDDINGS_MODEL_NAME}, dim={dim})")
        return True
    except Exception as exc:  # noqa: BLE001 — any failure means do not start
        print(f"preflight: embeddings unreachable at {url}: {exc}\n"
              "Restart the sidecar (a dead device still answers /health 200).",
              file=sys.stderr)
        return False


def build_pipeline_context(config: Dict[str, Any]) -> PipelineContext:
    from tools import pipeline_config as cfg
    from tools.artifact_writers import load_issue_catalog
    from tools.catalog_embeddings import build_candidate_provider
    from tools.scene_classifier_orchestrator import create_orchestrator_from_config
    from tools.vlm_client import create_vlm_client, get_model_configs_from_pipeline_config
    from tools.pass_config import SceneClassifierRunOptions

    catalog = load_issue_catalog(cfg.ISSUE_CATALOG_PATH)
    candidate_provider = build_candidate_provider(catalog)
    _, gpt5_config = get_model_configs_from_pipeline_config(cfg)
    vlm_client = create_vlm_client()
    orchestrator = create_orchestrator_from_config(
        cfg,
        candidate_provider=candidate_provider,
        catalog_items=catalog.get("items"),
        vlm_client=vlm_client,
    )
    options_base = SceneClassifierRunOptions.from_analysis_profile(
        analysis_profile="standard",
        toggles=config["pass_toggles"],
        model_overrides=config["model_overrides"],
        reasoning_efforts=config["reasoning_efforts"],
        pipeline_mode="publish",
    )
    return PipelineContext(
        cfg=cfg, catalog=catalog, orchestrator=orchestrator,
        vlm_client=vlm_client, gpt5_config=gpt5_config,
        options_base=options_base,
    )


# ---------------------------------------------------------------------------
# Property run: photos through the orchestrator, artifact via write_photo_intel
# ---------------------------------------------------------------------------

def _photo_ckpt_dir(prop_out_dir: Path) -> Path:
    return prop_out_dir / ".photos"


def _load_photo_checkpoints(
    ckpt_dir: Path, image_paths: List[Path],
) -> Dict[int, Any]:
    """Return {index: ImageResult} for photos already completed."""
    from tools.analyzer_cli import ImageResult
    cached: Dict[int, Any] = {}
    if not ckpt_dir.is_dir():
        return cached
    for idx, image_path in enumerate(image_paths):
        path = ckpt_dir / f"{image_path.name}.json"
        if not path.is_file():
            continue
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("image_path") != str(image_path):
            continue
        cached[idx] = ImageResult(
            image_path=str(image_path),
            scene_data=payload.get("scene_data"),
            scene=payload.get("scene", "unknown"),
            processing_time=float(payload.get("processing_time") or 0.0),
        )
    return cached


async def run_property_once(
    ctx: PipelineContext,
    manifest: Dict[str, Any],
    property_key: str,
    out_dir: Path,
    job_id: str,
    *,
    variant_prompt: Optional[str] = None,
    frozen: bool = False,
    scenes: Optional[Dict[str, str]] = None,
    concurrency: int = 3,
) -> Path:
    """One repeat of one property. Checkpoints after every photo; skips
    photos already checkpointed; returns the photo_intel.json path.

    scenes ({photo_key: scene}) replays a frozen Pass 1a scene capture via the
    pass_1a_frozen_scene orchestrator hook — no scene-classification VLM call
    is made and every repeat shares identical room assignments."""
    import time as _time
    from tools.analyzer_cli import ImageResult, PropertyAnalysisJob
    from tools.artifact_writers import write_photo_intel
    from tools.photo_pass_runner import run_photo_passes

    prop = manifest["properties"][property_key]
    images_root = Path(manifest["images_root"])
    image_paths = [images_root / property_key / p["photo_key"] for p in prop["photos"]]
    frozen_by_key = {p["photo_key"]: p["frozen_2a"] for p in prop["photos"]}

    job_dir = out_dir / property_key / job_id
    artifact_path = job_dir / "photo_intel.json"
    if artifact_path.is_file():
        return artifact_path
    job_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = _photo_ckpt_dir(out_dir / property_key)
    cached = _load_photo_checkpoints(ckpt_dir, image_paths)
    if cached:
        logger.info("%s/%s: resuming, %d/%d photos checkpointed",
                    property_key, job_id, len(cached), len(image_paths))

    started = _time.perf_counter()

    if scenes is not None:
        missing_scenes = [p.name for p in image_paths if p.name not in scenes]
        if missing_scenes:
            raise SystemExit(
                f"{property_key}: scene capture has no entry for "
                f"{missing_scenes} — rerun scene-capture before this stage"
            )

    async def _analyze_one(idx: int, image_path: Path) -> Any:
        meta: Dict[str, Any] = dict(
            run_id=job_id, photo_key=image_path.name, property_key=property_key,
        )
        if scenes is not None:
            meta["pass_1a_frozen_scene"] = scenes[image_path.name]
        if frozen:
            meta["pass_2a_frozen_freeform"] = frozen_by_key[image_path.name]
        elif variant_prompt is not None:
            meta["pass_2a_user_prompt"] = variant_prompt
        options = ctx.options_base.with_meta(**meta)
        t0 = _time.perf_counter()
        analysis = await ctx.orchestrator.analyze_image(
            image_path=image_path, options=options,
        )
        elapsed = _time.perf_counter() - t0
        result = ImageResult(
            image_path=str(image_path),
            scene_data=analysis.to_dict(),
            scene=analysis.scene or "unknown",
            processing_time=elapsed,
        )
        return result

    def _make_failed(idx: int, image_path: Path, exc: Exception) -> Any:
        logger.error("%s failed: %s", image_path.name, exc)
        return ImageResult(image_path=str(image_path), scene="unknown", error=str(exc))

    def _make_aborted(idx: int, image_path: Path) -> Any:
        return ImageResult(image_path=str(image_path), scene="unknown",
                           error="aborted", error_kind="aborted")

    def _on_result(idx: int, img_result: Any) -> None:
        if not img_result.error:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            target = ckpt_dir / f"{Path(img_result.image_path).name}.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(dataclasses.asdict(img_result), default=str,
                           ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(target)

    outcome = await run_photo_passes(
        image_paths,
        concurrency=concurrency,
        analyze_one=_analyze_one,
        make_failed=_make_failed,
        make_aborted=_make_aborted,
        cached_results=cached,
        on_cached=lambda idx, r: None,
        on_result=_on_result,
        fail_fast=True,
    )
    failed = [r for r in outcome.results
              if getattr(r, "error", None) and getattr(r, "error_kind", None) != "aborted"]
    if failed:
        raise RuntimeError(
            f"{property_key}/{job_id}: {len(failed)} photos failed "
            f"({failed[0].image_path}: {failed[0].error}); photo checkpoints "
            "kept — rerun to resume."
        )

    job = PropertyAnalysisJob(
        property_key=property_key,
        job_id=job_id,
        artifacts_dir=str(job_dir),
        timestamp=_utcnow(),
        results=list(outcome.results),
        total_processing_time=_time.perf_counter() - started,
        property_metadata=prop["property_metadata"] or None,
    )
    write_photo_intel(
        cfg=ctx.cfg,
        job=job,
        detection_backend="dinox",
        analysis_profile="standard",
        use_pass_architecture=True,
        pass_toggles=ctx.options_base.toggles.to_dict(),
        model_overrides=dict(load_config()["model_overrides"]),
        gpt_config=ctx.gpt5_config,
        issue_catalog=ctx.catalog,
        vlm_client=ctx.vlm_client,
        reasoning_efforts=dict(load_config()["reasoning_efforts"]),
        dependency_status={"embeddings": "ready"},
    )
    # Photo checkpoints are inputs to judging — keep them (unlike the
    # analyzer, which clears its checkpoints on success).
    return artifact_path


# ---------------------------------------------------------------------------
# Pre-2f benchmark totals (all packages force-confirmed)
# ---------------------------------------------------------------------------

def compute_pre2f_totals(artifact: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Two-phase v4: infer packages (probe), confirm every package id, recompute.

    Without this, no-2f runs zero every package-member line item
    (verification_status not_run -> is_valid_detection False), hiding exactly
    the cost-bearing flips the benchmark measures. Package inference is
    deterministic, so the probe's package ids match the recompute's.
    """
    from tools.renovation_estimate_v4 import compute_renovation_estimate_v4

    issues_flat = (
        artifact.get("product_estimate_issues_flat")
        if artifact.get("estimate_issues_flat")
        else artifact.get("product_issues_flat")
    ) or []
    photos = artifact.get("photos") or {}
    metadata = artifact.get("property_metadata") or None

    probe = compute_renovation_estimate_v4(
        issues_flat, catalog, photos, property_metadata=metadata,
    )
    package_ids = sorted({
        str(p.get("package_id"))
        for p in (probe.get("package_candidates") or [])
        if isinstance(p, dict) and p.get("package_id")
    })
    verifications = {
        pid: {"package_id": pid, "verification_status": "confirmed"}
        for pid in package_ids
    }
    v4 = compute_renovation_estimate_v4(
        issues_flat, catalog, photos, property_metadata=metadata,
        package_verifications=verifications,
    )

    line_items: List[Dict[str, Any]] = []
    for group in v4.get("groups") or []:
        for li in group.get("line_items") or []:
            # Tri-state: None means 2f never reviewed the line (standalone
            # work) — only an explicit False rejection excludes it.
            if li.get("is_valid_detection") is False:
                continue
            line_items.append({
                "catalog_item_id": li.get("catalog_item_id"),
                "name": li.get("name"),
                "cost_low": int(li.get("cost_low") or 0),
                "cost_high": int(li.get("cost_high") or 0),
                "package_id": li.get("package_id"),
                "trade_bucket": li.get("trade_bucket"),
            })
    packages = [
        {
            "package_id": p.get("package_id"),
            "package_type": p.get("package_type"),
            "cost_low": int(p.get("cost_low") or 0),
            "cost_high": int(p.get("cost_high") or 0),
        }
        for p in (v4.get("packages") or []) if isinstance(p, dict)
    ]
    final = v4.get("final_rehab") or {}
    low = int(final.get("low") or 0)
    high = int(final.get("high") or 0)
    return {
        "label": "pre_2f_all_packages_assumed_confirmed",
        "final_rehab": {
            "low": low, "high": high,
            "midpoint": int(final.get("midpoint") or round((low + high) / 2)),
        },
        "line_items": line_items,
        "packages": packages,
        "confirmed_package_count": len(package_ids),
    }


def totals_path_for(artifact_path: Path) -> Path:
    return artifact_path.parent / "benchmark_totals.json"


def ensure_totals(artifact_path: Path, catalog: Dict[str, Any]) -> Dict[str, Any]:
    out = totals_path_for(artifact_path)
    if out.is_file():
        return _load_json(out)
    totals = compute_pre2f_totals(_load_json(artifact_path), catalog)
    _write_json(out, totals)
    return totals


# ---------------------------------------------------------------------------
# Matrix runner shared by attribution + run stages
# ---------------------------------------------------------------------------

def stage_dir_for(stage_label: str) -> Path:
    return RUNS_DIR / stage_label


def v3_stage_dir(stage_label: str) -> Path:
    """v3-lineage stage directory (controlled Terra Pass 1a, frozen scenes)."""
    return V3_DIR / stage_label


# Stage-lineage routing (mirrors the V3_DIR/LEGACY_* split in
# benchmark_pass2a_packages): v3 stages resolve under runs/v3/, everything
# else stays on the flat legacy layout. "attribution" is frozen v2 evidence
# read only as a gate by `run`; "judge_*" is the archived image-based Sol
# round, superseded by the text-only match stage — both deliberately flat.
V3_STAGE_PREFIXES = ("variant_", "match_", "review_")


def resolve_stage_dir(stage_label: str) -> Path:
    if stage_label.startswith(V3_STAGE_PREFIXES):
        return v3_stage_dir(stage_label)
    return stage_dir_for(stage_label)


def rep_job_id(stage_label: str, rep: int) -> str:
    return f"{stage_label}_rep{rep}"


def run_matrix(
    config: Dict[str, Any],
    manifest: Dict[str, Any],
    stage_label: str,
    *,
    variant_prompt: Optional[str] = None,
    prompt_sha: Optional[str] = None,
    frozen: bool = False,
    skip_preflight: bool = False,
    stage_dir: Optional[Path] = None,
    scenes_by_prop: Optional[Dict[str, Dict[str, str]]] = None,
    scene_capture_sha: Optional[str] = None,
) -> Dict[str, Dict[int, Path]]:
    """Run k repeats of every manifest property. Returns
    {property_key: {rep: artifact_path}}. Fully resumable.

    stage_dir overrides the flat runs/<label> layout (v3 lineage);
    scenes_by_prop + scene_capture_sha replay a frozen Pass 1a capture."""
    if not skip_preflight and not preflight_embeddings():
        raise SystemExit(2)
    stage_dir = stage_dir if stage_dir is not None else resolve_stage_dir(stage_label)
    fingerprint = compute_fingerprint(
        manifest, config, prompt_sha, scene_capture_sha=scene_capture_sha)
    guard_fingerprint(stage_dir, fingerprint)

    ctx = build_pipeline_context(config)
    repeats = int(config["repeats"])
    artifacts: Dict[str, Dict[int, Path]] = {}
    for rep in range(1, repeats + 1):
        rep_dir = stage_dir / f"rep{rep}"
        for prop_key in sorted(manifest["properties"]):
            print(f"[{stage_label}] rep {rep}/{repeats} {prop_key} ...")
            path = asyncio.run(run_property_once(
                ctx, manifest, prop_key, rep_dir,
                rep_job_id(stage_label, rep),
                variant_prompt=variant_prompt, frozen=frozen,
                scenes=(scenes_by_prop or {}).get(prop_key),
                concurrency=int(config.get("photo_concurrency", 3)),
            ))
            ensure_totals(path, ctx.catalog)
            artifacts.setdefault(prop_key, {})[rep] = path
    return artifacts


def find_stage_artifacts(stage_label: str, manifest: Dict[str, Any],
                         repeats: int) -> Dict[str, Dict[int, Path]]:
    """Locate already-produced artifacts for a stage without running anything."""
    out: Dict[str, Dict[int, Path]] = {}
    for rep in range(1, repeats + 1):
        for prop_key in sorted(manifest["properties"]):
            path = (resolve_stage_dir(stage_label) / f"rep{rep}" / prop_key
                    / rep_job_id(stage_label, rep) / "photo_intel.json")
            if path.is_file():
                out.setdefault(prop_key, {})[rep] = path
    return out


# ---------------------------------------------------------------------------
# Attribution stop gate
# ---------------------------------------------------------------------------

def _priced_ids_by_rep(totals_by_rep: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, int]]:
    """{rep: {catalog_item_id: summed line midpoint}} for priced line items."""
    out: Dict[int, Dict[str, int]] = {}
    for rep, totals in totals_by_rep.items():
        per_id: Dict[str, int] = {}
        for li in totals.get("line_items") or []:
            if (li.get("cost_high") or 0) <= 0 and (li.get("cost_low") or 0) <= 0:
                continue
            cid = str(li.get("catalog_item_id") or "")
            if not cid:
                continue
            mid = round(((li.get("cost_low") or 0) + (li.get("cost_high") or 0)) / 2)
            per_id[cid] = per_id.get(cid, 0) + mid
        out[rep] = per_id
    return out


def evaluate_attribution_gate(
    totals_by_prop: Dict[str, Dict[int, Dict[str, Any]]],
    *,
    flip_abs_usd: float,
    flip_pct: float,
) -> Dict[str, Any]:
    """FAIL when a priced id present in only a strict subset of repeats has a
    line midpoint >= flip_abs_usd, or >= flip_pct% of the property's median
    pre-2f midpoint. That means downstream (2b/2c) instability alone flips
    material cost, and prompt-tuning 2a cannot fix it."""
    properties: Dict[str, Any] = {}
    failing_flips: List[Dict[str, Any]] = []
    for prop_key, totals_by_rep in totals_by_prop.items():
        reps = sorted(totals_by_rep)
        priced = _priced_ids_by_rep(totals_by_rep)
        midpoints = [totals_by_rep[r]["final_rehab"]["midpoint"] for r in reps]
        median_mid = statistics.median(midpoints) if midpoints else 0
        all_ids = sorted({cid for per in priced.values() for cid in per})
        flips = []
        for cid in all_ids:
            present = [r for r in reps if cid in priced[r]]
            if 0 < len(present) < len(reps):
                contribution = round(
                    sum(priced[r][cid] for r in present) / len(present)
                )
                pct = (100.0 * contribution / median_mid) if median_mid else 0.0
                material = (
                    contribution >= flip_abs_usd or pct >= flip_pct
                )
                row = {
                    "catalog_item_id": cid,
                    "present_in_reps": present,
                    "absent_in_reps": [r for r in reps if r not in present],
                    "line_midpoint_usd": contribution,
                    "pct_of_property_midpoint": round(pct, 1),
                    "material": material,
                }
                flips.append(row)
                if material:
                    failing_flips.append({"property_key": prop_key, **row})
        properties[prop_key] = {
            "midpoints_by_rep": {str(r): totals_by_rep[r]["final_rehab"] for r in reps},
            "median_midpoint": median_mid,
            "midpoint_spread": (max(midpoints) - min(midpoints)) if midpoints else 0,
            "flips": flips,
        }
    return {
        "gate": "attribution_material_flip",
        "thresholds": {"flip_abs_usd": flip_abs_usd, "flip_pct": flip_pct},
        "passed": not failing_flips,
        "failing_flips": failing_flips,
        "properties": properties,
    }


def stage_attribution(config: Dict[str, Any], manifest: Dict[str, Any],
                      skip_preflight: bool = False) -> Dict[str, Any]:
    artifacts = run_matrix(
        config, manifest, "attribution", frozen=True,
        prompt_sha=None, skip_preflight=skip_preflight,
    )
    totals_by_prop = {
        prop: {rep: _load_json(totals_path_for(path))
               for rep, path in by_rep.items()}
        for prop, by_rep in artifacts.items()
    }
    gate = evaluate_attribution_gate(
        totals_by_prop,
        flip_abs_usd=float(config["gates"]["attribution_flip_abs_usd"]),
        flip_pct=float(config["gates"]["attribution_flip_pct"]),
    )
    gate["evaluated_at"] = _utcnow()
    _write_json(resolve_stage_dir("attribution") / "gate.json", gate)
    verdict = "PASSED" if gate["passed"] else "FAILED"
    print(f"\nattribution gate: {verdict}")
    for flip in gate["failing_flips"]:
        print(f"  MATERIAL FLIP {flip['property_key']} {flip['catalog_item_id']}: "
              f"${flip['line_midpoint_usd']:,} "
              f"({flip['pct_of_property_midpoint']}% of midpoint), "
              f"present only in reps {flip['present_in_reps']}")
    if not gate["passed"]:
        print("\nVariance lives downstream of Pass 2a (2b/2c extraction). "
              "Do NOT run prompt variants; report to Steven.")
    return gate


# ---------------------------------------------------------------------------
# Judge (gpt-5.6-sol, blinded A/B, one image call per photo)
# ---------------------------------------------------------------------------

HALLUCINATION_CATEGORIES = [
    "unsupported_moisture_or_mold",
    "structural_failure",
    "hidden_system_condition",
    "absence_inferred_from_nonvisibility",
    "safety_hazard",
    "causal_inference",
    "repair_or_inspection_advice",
    "other_unsupported",
    "none",
]

JUDGE_SYSTEM_PROMPT = (
    "You are a meticulous renovation-photo evidence auditor. You are shown one "
    "property photo and the claims that two anonymized prompt variants (A and B) "
    "produced about it across several independent repeats. Judge every claim "
    "strictly against what is VISIBLE in this photo. A claim is 'supported' only "
    "if the photo shows direct visible evidence for it; 'unsupported' if the "
    "photo cannot support it (including hidden systems, causes, or anything "
    "outside the frame); 'uncertain' if visibility is genuinely ambiguous. "
    "Never reward or penalize verbosity. Judge claims before comparing prompts."
)

GOLD_SYSTEM_PROMPT = (
    "You match renovation condition statements. For each gold condition, decide "
    "for prompt A and prompt B separately whether ANY of that prompt's claims "
    "(across all repeats) states the same visible condition. Paraphrase counts; "
    "a different condition, surface, or component does not."
)


def _judge_schema() -> Dict[str, Any]:
    claim = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "enum": ["A", "B"]},
            "repeat": {"type": "integer"},
            "claim_index": {"type": "integer"},
            "label": {"type": "string",
                      "enum": ["supported", "unsupported", "uncertain"]},
            "visible_evidence": {"type": "string"},
            "hallucination_category": {"type": "string",
                                       "enum": HALLUCINATION_CATEGORIES},
            "cost_bearing": {"type": "boolean"},
        },
        "required": ["prompt", "repeat", "claim_index", "label",
                     "visible_evidence", "hallucination_category", "cost_bearing"],
        "additionalProperties": False,
    }
    per_prompt = {
        "type": "object",
        "properties": {
            "coverage": {"type": "string"},
            "missing_visible_conditions": {"type": "array",
                                           "items": {"type": "string"}},
            "verbosity_bias": {"type": "string"},
            "systematic_patterns": {"type": "string"},
        },
        "required": ["coverage", "missing_visible_conditions",
                     "verbosity_bias", "systematic_patterns"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "claims": {"type": "array", "items": claim},
            "per_prompt": {
                "type": "object",
                "properties": {"A": per_prompt, "B": per_prompt},
                "required": ["A", "B"],
                "additionalProperties": False,
            },
        },
        "required": ["claims", "per_prompt"],
        "additionalProperties": False,
    }


def _gold_schema() -> Dict[str, Any]:
    row = {
        "type": "object",
        "properties": {
            "gold_id": {"type": "string"},
            "covered_by_A": {"type": "boolean"},
            "covered_by_B": {"type": "boolean"},
        },
        "required": ["gold_id", "covered_by_A", "covered_by_B"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"conditions": {"type": "array", "items": row}},
        "required": ["conditions"],
        "additionalProperties": False,
    }


def load_photo_repeat_records(
    stage_label: str, rep: int, property_key: str, photo_key: str,
) -> Dict[str, Any]:
    """Compact per-repeat record from the per-photo checkpoint + totals."""
    prop_dir = resolve_stage_dir(stage_label) / f"rep{rep}" / property_key
    ckpt = _load_json(_photo_ckpt_dir(prop_dir) / f"{photo_key}.json")
    scene_data = ckpt.get("scene_data") or {}
    claims = [
        (o.get("description") or "").strip()
        for o in ((scene_data.get("observations_struct") or {}).get("observations") or [])
        if (o.get("description") or "").strip()
    ]
    kept = [
        {"description": o.get("description"), "kind": o.get("kind"),
         "issue_id": o.get("issue_id")}
        for o in (scene_data.get("observations") or [])
    ]
    excluded = [
        {"description": o.get("description"), "reason": o.get("reason")}
        for o in (scene_data.get("excluded_observations") or [])
    ]
    resolved_ids = sorted({
        str(r.get("resolved_item_id"))
        for r in (scene_data.get("resolved_items") or [])
        if r.get("resolved_item_id")
    })
    resolved_by_issue = {
        str(r.get("issue_id")): r.get("resolved_item_id")
        for r in (scene_data.get("resolved_items") or [])
        if r.get("issue_id")
    }
    debug = scene_data.get("debug") or {}
    skipped_by_text = {
        (row.get("observation") or "").strip(): row.get("skipped_reason")
        for row in (debug.get("pass_2d_per_observation") or [])
        if row.get("skipped_reason")
    }
    totals = _load_json(totals_path_for(
        prop_dir / rep_job_id(stage_label, rep) / "photo_intel.json"
    ))
    priced_property_ids = {
        str(li.get("catalog_item_id")): li for li in totals.get("line_items") or []
    }
    priced = [
        {"catalog_item_id": cid,
         "cost_low": priced_property_ids[cid]["cost_low"],
         "cost_high": priced_property_ids[cid]["cost_high"]}
        for cid in resolved_ids if cid in priced_property_ids
    ]
    return {"claims": claims, "kept": kept, "excluded": excluded,
            "resolved_ids": resolved_ids, "priced": priced,
            "resolved_by_issue": resolved_by_issue,
            "total_resolve_count": int(
                (debug.get("pass_2d_gate") or {}).get("total_resolve_count") or 0),
            "skipped_by_text": skipped_by_text}


# ---------------------------------------------------------------------------
# Claim lineage: what the pipeline did with each atomic 2b claim
# ---------------------------------------------------------------------------

LINEAGE_STATUSES = (
    "resolved",                  # 2d returned a catalog id
    "retained_unresolved",       # 2d ran and declined to map it
    "resolution_skipped",        # 2d had nothing to offer it (no candidates)
    "resolution_not_attempted",  # past max_resolve_per_image — never offered
    "filtered_2c",               # dropped before resolution
    "unknown_lane",              # defensive: text in no lane
)

# A linkage miss means the resolver saw the observation and produced no catalog
# id. Truncated observations were never offered, so they are not misses.
LINKAGE_MISS_STATUSES = ("retained_unresolved", "resolution_skipped")


def _norm_text(value: Optional[str]) -> str:
    return " ".join((value or "").split()).casefold()


def classify_claim_lineage(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One pipeline status per atomic 2b claim, in claim order.

    Claims are consumed against the kept/excluded lanes by text — 2c neither
    reorders nor rewrites descriptions, and kept+excluded reproduces the claim
    multiset exactly. Kept claims are then placed by their Pass 2d outcome:
    the resolver only ever sees observations[:max_resolve_per_image], so kept
    claims past that cut were never offered and must not read as linkage misses.
    """
    kept_by_text: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, obs in enumerate(record.get("kept") or []):
        kept_by_text.setdefault(
            _norm_text(obs.get("description")), []).append((idx, obs))
    excluded_by_text: Dict[str, List[Dict[str, Any]]] = {}
    for obs in record.get("excluded") or []:
        excluded_by_text.setdefault(
            _norm_text(obs.get("description")), []).append(obs)

    resolved_by_issue = record.get("resolved_by_issue") or {}
    skipped_by_text = {_norm_text(k): v
                       for k, v in (record.get("skipped_by_text") or {}).items()}
    cut = int(record.get("total_resolve_count") or 0)

    rows: List[Dict[str, Any]] = []
    for claim_index, text in enumerate(record.get("claims") or []):
        norm = _norm_text(text)
        row: Dict[str, Any] = {"claim_index": claim_index, "text": text,
                               "resolved_item_id": None, "detail": None}
        if kept_by_text.get(norm):
            kept_index, obs = kept_by_text[norm].pop(0)
            row["kind"] = obs.get("kind")
            issue_id = obs.get("issue_id")
            if issue_id in resolved_by_issue:
                item_id = resolved_by_issue[issue_id]
                row["status"] = "resolved" if item_id else "retained_unresolved"
                row["resolved_item_id"] = item_id
            elif kept_index >= cut:
                row["status"] = "resolution_not_attempted"
                row["detail"] = f"beyond the pass 2d cut of {cut}"
            else:
                row["status"] = "resolution_skipped"
                row["detail"] = skipped_by_text.get(norm) or "no_resolver_row"
        elif excluded_by_text.get(norm):
            row["status"] = "filtered_2c"
            row["detail"] = excluded_by_text[norm].pop(0).get("reason")
        else:
            row["status"] = "unknown_lane"
        rows.append(row)
    return rows


def _judge_user_prompt(photo_payload: Dict[str, Any]) -> str:
    return (
        "Grade every atomic claim below against the attached photo, then "
        "summarize each prompt.\n\n"
        "For each claim return: prompt (A/B), repeat, claim_index, label "
        "(supported/unsupported/uncertain), a short visible_evidence "
        "explanation, hallucination_category ('none' when supported), and "
        "cost_bearing (true when the claim maps to a priced renovation line "
        "item in its repeat's 'priced' list).\n\n"
        "Then, per prompt: coverage of the visible renovation-relevant "
        "conditions, missing_visible_conditions the prompt failed to mention "
        "in ANY repeat, verbosity_bias, and systematic_patterns.\n\n"
        + json.dumps(photo_payload, ensure_ascii=False, indent=1)
    )


def _judge_config(config: Dict[str, Any], gpt5_config: Dict[str, Any],
                  schema: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
    judge = config["judge"]
    return {
        **gpt5_config,
        "provider": "openai",
        "model": judge["model"],
        "max_tokens": int(judge.get("max_tokens", 16000)),
        "reasoning_effort": judge.get("reasoning_effort", "medium"),
        "response_json_schema": schema,
        "response_schema_name": schema_name,
        "analysis_pass": f"pass2a-benchmark judge ({schema_name})",
    }


async def judge_photo(
    ctx: PipelineContext,
    config: Dict[str, Any],
    manifest: Dict[str, Any],
    round_label: str,
    variant_a: str,
    variant_b: str,
    property_key: str,
    photo: Dict[str, Any],
    gold_conditions: List[Dict[str, str]],
) -> Dict[str, Any]:
    from tools.llm_json import extract_json_object
    photo_key = photo["photo_key"]
    repeats = int(config["repeats"])

    # Blinding: deterministic per (round, photo) so resume reproduces it.
    rng = random.Random(f"{round_label}|{property_key}|{photo_key}")
    a_is_first_variant = rng.random() < 0.5
    mapping = ({"A": variant_a, "B": variant_b} if a_is_first_variant
               else {"A": variant_b, "B": variant_a})

    payload: Dict[str, Any] = {"photo": photo_key, "prompts": {}}
    for letter in ("A", "B"):
        stage_label = f"variant_{mapping[letter]}"
        reps = []
        for rep in range(1, repeats + 1):
            record = load_photo_repeat_records(
                stage_label, rep, property_key, photo_key)
            reps.append({
                "repeat": rep,
                "claims": [
                    {"claim_index": i, "text": t}
                    for i, t in enumerate(record["claims"])
                ],
                "kept_after_classification": record["kept"],
                "excluded": record["excluded"],
                "resolved_catalog_ids": record["resolved_ids"],
                "priced": record["priced"],
            })
        payload["prompts"][letter] = reps

    image_path = Path(manifest["images_root"]) / property_key / photo_key
    judge_cfg = _judge_config(config, ctx.gpt5_config, _judge_schema(),
                              "pass2a_claim_judgment_v1")
    response = await ctx.vlm_client.analyze_image(
        image_path=image_path,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=_judge_user_prompt(payload),
        **judge_cfg,
    )
    verdict = extract_json_object(response)

    gold_result = None
    if gold_conditions:
        gold_payload = {
            "gold_conditions": gold_conditions,
            "prompts": {
                letter: [
                    {"repeat": rep["repeat"],
                     "claims": [c["text"] for c in rep["claims"]]}
                    for rep in payload["prompts"][letter]
                ]
                for letter in ("A", "B")
            },
        }
        gold_cfg = _judge_config(config, ctx.gpt5_config, _gold_schema(),
                                 "pass2a_gold_coverage_v1")
        gold_response = await ctx.vlm_client.analyze_text(
            system_prompt=GOLD_SYSTEM_PROMPT,
            user_prompt=json.dumps(gold_payload, ensure_ascii=False, indent=1),
            **gold_cfg,
        )
        gold_result = extract_json_object(gold_response)

    return {
        "property_key": property_key,
        "photo_key": photo_key,
        "blinding": mapping,       # concealed mapping, stored for de-aliasing
        "payload_prompts": payload["prompts"],
        "verdict": verdict,
        "gold_coverage": gold_result,
        "judged_at": _utcnow(),
    }


def stage_judge(config: Dict[str, Any], manifest: Dict[str, Any],
                round_label: str) -> Path:
    variant_a, _, variant_b = round_label.partition("_vs_")
    if not variant_a or not variant_b:
        raise SystemExit("--round must look like baseline_vs_checklist")
    prompts = load_prompts()
    for v in (variant_a, variant_b):
        if v not in prompts:
            raise SystemExit(f"unknown variant in round: {v}")
        found = find_stage_artifacts(f"variant_{v}", manifest, int(config["repeats"]))
        expected = len(manifest["properties"]) * int(config["repeats"])
        have = sum(len(r) for r in found.values())
        if have != expected:
            raise SystemExit(
                f"variant_{v} incomplete: {have}/{expected} property-repeats "
                "done — finish the run stage first"
            )

    stage_dir = resolve_stage_dir(f"judge_{round_label}")
    fingerprint = compute_fingerprint(
        manifest, config,
        prompt_sha=f"{prompts[variant_a]['sha256']}|{prompts[variant_b]['sha256']}",
    )
    guard_fingerprint(stage_dir, fingerprint)

    gold = _load_json(GOLD_PATH) if GOLD_PATH.is_file() else {"photos": {}}
    ctx = build_pipeline_context(config)

    out_path = stage_dir / "judgments.json"
    judgments: Dict[str, Any] = (
        _load_json(out_path) if out_path.is_file() else {}
    )
    for prop_key, photo in manifest_photos(manifest):
        key = f"{prop_key}/{photo['photo_key']}"
        if key in judgments:
            continue
        print(f"[judge {round_label}] {key} ...")
        gold_conditions = gold.get("photos", {}).get(key) or []
        judgments[key] = asyncio.run(judge_photo(
            ctx, config, manifest, round_label, variant_a, variant_b,
            prop_key, photo, gold_conditions,
        ))
        _write_json(out_path, judgments)  # checkpoint after every Sol call
    print(f"judged {len(judgments)} photos -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Match — blinded, text-only alignment of generated claims to the human gold
# ---------------------------------------------------------------------------

MATCH_DECISIONS = ("match", "no_match", "ambiguous")

MATCHER_SYSTEM_PROMPT = (
    "You align generated renovation observations with a human-authored list of "
    "gold conditions for one photo. You are never shown the photo: judge the "
    "text alone, and treat the gold list as the only ground truth.\n"
    "Answer 'match', with one or more gold_ids, ONLY when the observation "
    "states the same component AND the same condition as those gold "
    "conditions, in the same location when the gold condition names one, at a "
    "materially similar severity. Paraphrase counts in full: wording, "
    "ordering, terminology, and any catalog or product naming are irrelevant.\n"
    "Answer 'ambiguous' when an observation covers a gold condition but also "
    "asserts a material extra claim the gold does not state, or when you "
    "genuinely cannot decide. A compound observation carrying an unsupported "
    "extra condition is 'ambiguous', never 'match'.\n"
    "Answer 'no_match' when no gold condition describes it.\n"
    "Several observations may match one gold condition, and one observation "
    "may match several. Return exactly one row per observation, and give a "
    "one-sentence explanation for every row."
)

MATCHER_USER_TEMPLATE = (
    "Gold conditions and generated observations for one photo follow. Return "
    "exactly one row per observation, keyed by its claim_index.\n\n{payload}"
)


def _matcher_schema() -> Dict[str, Any]:
    row = {
        "type": "object",
        "properties": {
            "claim_index": {"type": "integer"},
            "decision": {"type": "string", "enum": list(MATCH_DECISIONS)},
            "gold_ids": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "string"},
        },
        "required": ["claim_index", "decision", "gold_ids", "explanation"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": row}},
        "required": ["rows"],
        "additionalProperties": False,
    }


def _matcher_call_config(config: Dict[str, Any], gpt5_config: Dict[str, Any],
                         schema: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
    matcher = config.get("matcher") or {}
    provider = matcher.get("provider") or "openai"
    call = {
        "provider": provider,
        "model": matcher["model"],
        "max_tokens": int(matcher.get("max_tokens", 8000)),
        "analysis_pass": f"pass2a-benchmark matcher ({schema_name})",
    }
    if provider == "openai":
        return {
            **gpt5_config, **call,
            "reasoning_effort": matcher.get("reasoning_effort", "low"),
            "response_json_schema": schema,
            "response_schema_name": schema_name,
        }
    url = matcher.get("url")
    if not url:
        from tools import pipeline_config as cfg
        url = cfg.LM_STUDIO_URL
    return {**call, "url": url}


def validate_matcher_rows(rows: Any, n_claims: int,
                          gold_ids: List[str]) -> List[Dict[str, Any]]:
    """Enforce the matcher contract harness-side, identically for every provider.

    Only the OpenAI path of VLMClient.analyze_text forwards a response schema to
    the server, so this — not the provider — is what makes Terra and a local
    Qwen produce the same rows.
    """
    if not isinstance(rows, list):
        raise ValueError("matcher payload has no 'rows' list")
    allowed = set(gold_ids)
    seen: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"row is not an object: {row!r}")
        try:
            idx = int(row["claim_index"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"row has no usable claim_index: {row!r}")
        if not 0 <= idx < n_claims:
            raise ValueError(f"claim_index {idx} outside 0..{n_claims - 1}")
        if idx in seen:
            raise ValueError(f"duplicate claim_index {idx}")
        decision = row.get("decision")
        if decision not in MATCH_DECISIONS:
            raise ValueError(f"claim {idx}: bad decision {decision!r}")
        ids = [str(g) for g in (row.get("gold_ids") or [])]
        unknown = sorted(set(ids) - allowed)
        if unknown:
            raise ValueError(f"claim {idx}: gold_ids not on this photo: {unknown}")
        if decision == "match" and not ids:
            raise ValueError(f"claim {idx}: 'match' carries no gold_ids")
        if decision == "no_match" and ids:
            raise ValueError(f"claim {idx}: 'no_match' must carry no gold_ids")
        seen[idx] = {"claim_index": idx, "decision": decision,
                     "gold_ids": sorted(dict.fromkeys(ids)),
                     "explanation": str(row.get("explanation") or "")}
    missing = [i for i in range(n_claims) if i not in seen]
    if missing:
        raise ValueError(f"matcher skipped claim_index {missing[:10]}"
                         + (" ..." if len(missing) > 10 else ""))
    return [seen[i] for i in range(n_claims)]


def compute_match_fingerprint(
    manifest: Dict[str, Any], config: Dict[str, Any], round_label: str,
    variant_a: str, variant_b: str,
) -> Dict[str, Any]:
    """Resume identity for the match stage.

    Binds the matcher itself (model, prompt, schema), the human gold, and the
    generation runs being scored — the variant fingerprints already carry their
    git head, model overrides and token caps. Deliberately excludes this file's
    git head: harness edits must not orphan paid matcher calls.
    """
    from tools.comparison_common import sha256_bytes, sha256_canonical, sha256_file
    prompts = load_prompts()
    matcher = config.get("matcher") or {}
    run_fps: Dict[str, Optional[str]] = {}
    for variant in (variant_a, variant_b):
        fp_path = resolve_stage_dir(f"variant_{variant}") / "fingerprint.json"
        run_fps[variant] = (sha256_canonical(_load_json(fp_path))
                            if fp_path.is_file() else None)
    return {
        "stage": f"match_{round_label}",
        "matcher_provider": matcher.get("provider") or "openai",
        "matcher_model": matcher.get("model"),
        "matcher_url": matcher.get("url"),
        "matcher_reasoning_effort": matcher.get("reasoning_effort"),
        "matcher_prompt_sha256": sha256_bytes(
            f"{MATCHER_SYSTEM_PROMPT}\x00{MATCHER_USER_TEMPLATE}".encode("utf-8")),
        "matcher_schema_sha256": sha256_canonical(_matcher_schema()),
        "gold_sha256": sha256_file(GOLD_PATH),
        "image_hashes_sha": sha256_canonical(
            _manifest_hashes(manifest, "image_sha256")),
        "frozen_2a_sha": sha256_canonical(
            _manifest_hashes(manifest, "frozen_2a_sha256")),
        "variant_prompt_shas": {v: prompts[v]["sha256"]
                                for v in (variant_a, variant_b)},
        "variant_run_fingerprint_shas": run_fps,
        "repeats": int(config["repeats"]),
    }


def match_blinding(round_label: str, property_key: str, photo_key: str,
                   variant_a: str, variant_b: str) -> Dict[str, str]:
    """Deterministic per (round, photo) so a resume reproduces the mapping."""
    rng = random.Random(f"match|{round_label}|{property_key}|{photo_key}")
    if rng.random() < 0.5:
        return {"A": variant_a, "B": variant_b}
    return {"A": variant_b, "B": variant_a}


def matcher_payload(photo: Dict[str, Any], gold_rows: List[Dict[str, str]],
                    claims: List[str]) -> Dict[str, Any]:
    """Everything the matcher sees. No image, no catalog ids, no pipeline lanes."""
    return {
        "photo": photo["photo_key"],
        "scene": photo.get("scene"),
        "gold_conditions": [{"gold_id": g["gold_id"], "condition": g["condition"]}
                            for g in gold_rows],
        "claims": [{"claim_index": i, "text": t} for i, t in enumerate(claims)],
    }


def build_matcher_client(config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Text client only — matching never touches the pipeline or embeddings."""
    from tools import pipeline_config as cfg
    from tools.vlm_client import (create_vlm_client,
                                  get_model_configs_from_pipeline_config)
    _, gpt5_config = get_model_configs_from_pipeline_config(cfg)
    return create_vlm_client(), gpt5_config


async def run_matcher_call(
    vlm_client: Any, gpt5_config: Dict[str, Any], config: Dict[str, Any],
    photo: Dict[str, Any], gold_rows: List[Dict[str, str]], claims: List[str],
) -> List[Dict[str, Any]]:
    """One blinded text-only call: every claim of one variant-repeat."""
    from tools.llm_json import extract_json_object
    schema = _matcher_schema()
    call_cfg = _matcher_call_config(config, gpt5_config, schema,
                                    "pass2a_gold_match_v1")
    user_prompt = MATCHER_USER_TEMPLATE.format(payload=json.dumps(
        matcher_payload(photo, gold_rows, claims), ensure_ascii=False, indent=1))
    if call_cfg.get("provider") != "openai":
        # analyze_text forwards response_json_schema on the OpenAI path only,
        # so non-OpenAI providers get the contract in-band instead.
        user_prompt += ("\n\nReply with ONLY a JSON object matching this schema:\n"
                        + json.dumps(schema, ensure_ascii=False))
    gold_ids = [str(g["gold_id"]) for g in gold_rows]
    last_error: Optional[Exception] = None
    for _ in range(2):
        response = await vlm_client.analyze_text(
            system_prompt=MATCHER_SYSTEM_PROMPT, user_prompt=user_prompt,
            **call_cfg,
        )
        try:
            parsed = extract_json_object(response)
            return validate_matcher_rows(
                (parsed or {}).get("rows"), len(claims), gold_ids)
        except (ValueError, AttributeError, TypeError) as exc:  # noqa: PERF203
            last_error = exc
            logger.warning("matcher response rejected, retrying: %s", exc)
    raise RuntimeError(f"matcher returned an unusable payload: {last_error}")


def match_photos_dir(round_label: str) -> Path:
    return resolve_stage_dir(f"match_{round_label}") / "photos"


def load_match_artifacts(round_label: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    photos_dir = match_photos_dir(round_label)
    if not photos_dir.is_dir():
        return out
    for path in sorted(photos_dir.glob("*.json")):
        art = _load_json(path)
        out[f"{art['property_key']}/{art['photo_key']}"] = art
    return out


def parse_round_label(round_label: str, prompts: Dict[str, Any]) -> Tuple[str, str]:
    variant_a, _, variant_b = round_label.partition("_vs_")
    if not variant_a or not variant_b:
        raise SystemExit("--round must look like baseline_vs_checklist")
    for variant in (variant_a, variant_b):
        if variant not in prompts:
            raise SystemExit(f"unknown variant in round: {variant}")
    return variant_a, variant_b


def load_gold() -> Dict[str, Any]:
    if not GOLD_PATH.is_file():
        raise SystemExit(f"human gold missing: {GOLD_PATH}")
    return _load_json(GOLD_PATH)


def gold_ids_by_photo(gold: Dict[str, Any]) -> Dict[str, set]:
    return {key: {str(g["gold_id"]) for g in rows}
            for key, rows in (gold.get("photos") or {}).items()}


def _require_pinned_gold(manifest: Dict[str, Any]) -> str:
    from tools.comparison_common import sha256_file
    actual = sha256_file(GOLD_PATH)
    pinned = manifest.get("gold_sha256")
    if pinned and pinned != actual:
        raise SystemExit(
            f"gold/reference.json changed since the manifest was pinned "
            f"({pinned[:12]}... -> {actual[:12]}...). Run the repin-gold stage "
            "to acknowledge the edit, then rerun match.")
    return actual


def _require_complete_variants(config: Dict[str, Any], manifest: Dict[str, Any],
                               variants: Tuple[str, ...]) -> None:
    repeats = int(config["repeats"])
    expected = len(manifest["properties"]) * repeats
    for variant in variants:
        found = find_stage_artifacts(f"variant_{variant}", manifest, repeats)
        have = sum(len(r) for r in found.values())
        if have != expected:
            raise SystemExit(
                f"variant_{variant} incomplete: {have}/{expected} "
                "property-repeats done — finish the run stage first")


def stage_match(config: Dict[str, Any], manifest: Dict[str, Any],
                round_label: str, *, dry_run: bool = False) -> Path:
    variant_a, variant_b = parse_round_label(round_label, load_prompts())
    _require_complete_variants(config, manifest, (variant_a, variant_b))
    gold = load_gold()
    _require_pinned_gold(manifest)
    repeats = int(config["repeats"])
    by_photo = gold.get("photos") or {}

    if dry_run:
        return _match_dry_run(config, manifest, gold, round_label,
                              variant_a, variant_b)

    stage_dir = resolve_stage_dir(f"match_{round_label}")
    guard_fingerprint(stage_dir, compute_match_fingerprint(
        manifest, config, round_label, variant_a, variant_b))
    vlm_client, gpt5_config = build_matcher_client(config)

    made = skipped = 0
    for prop_key, photo in manifest_photos(manifest):
        photo_key = photo["photo_key"]
        key = f"{prop_key}/{photo_key}"
        gold_rows = by_photo.get(key) or []
        out_path = match_photos_dir(round_label) / f"{prop_key}__{photo_key}.json"
        artifact = _load_json(out_path) if out_path.is_file() else {}
        mapping = match_blinding(round_label, prop_key, photo_key,
                                 variant_a, variant_b)
        artifact.update({
            "property_key": prop_key, "photo_key": photo_key,
            "blinding": mapping,
            "gold_ids": [str(g["gold_id"]) for g in gold_rows],
        })
        calls = artifact.setdefault("calls", {})
        for letter in ("A", "B"):
            for rep in range(1, repeats + 1):
                call_key = f"{letter}|rep{rep}"
                if call_key in calls:
                    skipped += 1
                    continue
                record = load_photo_repeat_records(
                    f"variant_{mapping[letter]}", rep, prop_key, photo_key)
                claims = record["claims"]
                print(f"[match {round_label}] {key} {call_key} "
                      f"({len(claims)} claims, {len(gold_rows)} gold) ...")
                if gold_rows and claims:
                    rows = asyncio.run(run_matcher_call(
                        vlm_client, gpt5_config, config, photo, gold_rows, claims))
                else:
                    rows = [{"claim_index": i, "decision": "no_match",
                             "gold_ids": [],
                             "explanation": "no gold conditions for this photo"}
                            for i in range(len(claims))]
                calls[call_key] = {"rows": rows, "matched_at": _utcnow()}
                _write_json(out_path, artifact)  # checkpoint after every call
                made += 1
        if not out_path.is_file():
            _write_json(out_path, artifact)
    print(f"match {round_label}: {made} new calls, {skipped} resumed "
          f"-> {match_photos_dir(round_label)}")
    return match_photos_dir(round_label)


def _match_dry_run(config: Dict[str, Any], manifest: Dict[str, Any],
                   gold: Dict[str, Any], round_label: str,
                   variant_a: str, variant_b: str) -> Path:
    """Build every payload and classify every lane without calling a model."""
    repeats = int(config["repeats"])
    by_photo = gold.get("photos") or {}
    summary: Dict[str, Any] = {"round": round_label, "generated_at": _utcnow(),
                               "variants": {}}
    for variant in (variant_a, variant_b):
        lanes: Dict[str, int] = {}
        claims_total = photo_repeats = truncated = payload_max = 0
        for prop_key, photo in manifest_photos(manifest):
            for rep in range(1, repeats + 1):
                record = load_photo_repeat_records(
                    f"variant_{variant}", rep, prop_key, photo["photo_key"])
                gold_rows = by_photo.get(f"{prop_key}/{photo['photo_key']}") or []
                matcher_payload(photo, gold_rows, record["claims"])  # shape check
                for row in classify_claim_lineage(record):
                    lanes[row["status"]] = lanes.get(row["status"], 0) + 1
                photo_repeats += 1
                claims_total += len(record["claims"])
                payload_max = max(payload_max, len(record["claims"]))
                if len(record["kept"]) > int(record["total_resolve_count"]):
                    truncated += 1
        summary["variants"][variant] = {
            "photo_repeats": photo_repeats, "claims_total": claims_total,
            "max_claims_in_one_call": payload_max,
            "photo_repeats_truncated_by_2d_cap": truncated,
            "lanes": dict(sorted(lanes.items())),
        }
    summary["photos_with_gold"] = sum(
        1 for prop_key, photo in manifest_photos(manifest)
        if by_photo.get(f"{prop_key}/{photo['photo_key']}"))
    summary["photos_total"] = len(manifest_photos(manifest))
    summary["planned_calls"] = 2 * repeats * summary["photos_total"]
    out = resolve_stage_dir(f"match_{round_label}") / "dry_run.json"
    _write_json(out, summary)
    print(json.dumps(summary, indent=2))
    print(f"\ndry run only - no model calls made. -> {out}")
    return out


# ---------------------------------------------------------------------------
# Review — one editable CSV, human decisions keyed by a rematch-stable row id
# ---------------------------------------------------------------------------

HUMAN_DECISIONS = ("match", "unsupported", "exclude", "gold_gap")

# States that must be adjudicated before a round can be reported.
BLOCKING_DECISIONS = ("pending", "gold_gap", "needs_rereview")

REVIEW_COLUMNS = [
    "row_id", "property", "photo", "variant", "repeat", "claim_index",
    "observation", "pipeline_status", "resolved_item_id",
    "proposed_match", "proposed_gold_ids", "matcher_explanation", "audit",
    "human_decision", "corrected_gold_ids", "critical", "reviewer_note",
]


def review_dir_for(round_label: str) -> Path:
    return v3_stage_dir(f"review_{round_label}")


def decisions_path_for(round_label: str) -> Path:
    return review_dir_for(round_label) / "decisions.json"


def load_decisions(round_label: str) -> Dict[str, Any]:
    path = decisions_path_for(round_label)
    return (_load_json(path).get("decisions") or {}) if path.is_file() else {}


def review_row_id(variant: str, rep: int, property_key: str, photo_key: str,
                  claim_index: int) -> str:
    """Stable across rematches: generation is frozen, so claim order is too."""
    return f"{variant}|rep{rep}|{property_key}/{photo_key}|c{claim_index}"


def parse_review_row_id(row_id: str) -> Tuple[str, int, str, int]:
    variant, _, rest = row_id.partition("|")
    rep_token, _, rest = rest.partition("|")
    photo_ref, _, claim_token = rest.partition("|")
    return variant, int(rep_token[3:]), photo_ref, int(claim_token[1:])


def _split_ids(value: Optional[str]) -> List[str]:
    tokens = (value or "").replace(",", ";").split(";")
    return sorted(dict.fromkeys(t.strip() for t in tokens if t.strip()))


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("x", "y", "yes", "true", "1")


def effective_decision(matcher_row: Dict[str, Any],
                       decision: Optional[Dict[str, Any]],
                       known_gold: set,
                       gold_sha: Optional[str]) -> Tuple[str, List[str]]:
    """A human decision wins unless the gold moved underneath it."""
    if decision:
        ids = [str(g) for g in (decision.get("gold_ids") or [])]
        moved = (
            (decision.get("decision") == "match" and not set(ids) <= known_gold)
            or (decision.get("decision") == "gold_gap"
                and decision.get("gold_sha256") != gold_sha)
        )
        return ("needs_rereview" if moved else decision["decision"]), ids
    if matcher_row["decision"] == "match":
        return "match", list(matcher_row["gold_ids"])
    return "pending", list(matcher_row["gold_ids"])


def build_review_rows(config: Dict[str, Any], manifest: Dict[str, Any],
                      round_label: str, gold: Dict[str, Any],
                      decisions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The single assembly point behind export, import and the report."""
    from tools.comparison_common import sha256_file
    repeats = int(config["repeats"])
    by_photo_ids = gold_ids_by_photo(gold)
    gold_sha = sha256_file(GOLD_PATH) if GOLD_PATH.is_file() else None
    artifacts = load_match_artifacts(round_label)
    rows: List[Dict[str, Any]] = []
    for prop_key, photo in manifest_photos(manifest):
        photo_key = photo["photo_key"]
        key = f"{prop_key}/{photo_key}"
        artifact = artifacts.get(key)
        if not artifact:
            continue
        known_gold = by_photo_ids.get(key, set())
        for letter, variant in artifact["blinding"].items():
            for rep in range(1, repeats + 1):
                call = (artifact.get("calls") or {}).get(f"{letter}|rep{rep}")
                if not call:
                    continue
                lineage = {r["claim_index"]: r for r in classify_claim_lineage(
                    load_photo_repeat_records(
                        f"variant_{variant}", rep, prop_key, photo_key))}
                for matcher_row in call["rows"]:
                    index = matcher_row["claim_index"]
                    lane = lineage.get(index) or {}
                    row_id = review_row_id(variant, rep, prop_key, photo_key, index)
                    decision = decisions.get(row_id)
                    eff, eff_ids = effective_decision(
                        matcher_row, decision, known_gold, gold_sha)
                    rows.append({
                        "row_id": row_id,
                        "property": prop_key,
                        "photo": photo_key,
                        "variant": variant,
                        "repeat": rep,
                        "claim_index": index,
                        "observation": lane.get("text", ""),
                        "pipeline_status": lane.get("status", "unknown_lane"),
                        "resolved_item_id": lane.get("resolved_item_id") or "",
                        "proposed_match": matcher_row["decision"],
                        "proposed_gold_ids": ";".join(matcher_row["gold_ids"]),
                        "matcher_explanation": matcher_row.get("explanation", ""),
                        "audit": "",
                        "human_decision": (decision or {}).get("decision", ""),
                        "corrected_gold_ids": ";".join(
                            (decision or {}).get("gold_ids") or []),
                        "critical": "x" if (decision or {}).get("critical") else "",
                        "reviewer_note": (decision or {}).get("note", ""),
                        "effective_decision": eff,
                        "effective_gold_ids": eff_ids,
                        "critical_flag": bool((decision or {}).get("critical")),
                    })
    return rows


def _audit_flags(rows: List[Dict[str, Any]], round_label: str,
                 rate: float) -> set:
    """Seeded, non-blocking spot-check sample over unreviewed auto-matches."""
    auto = sorted(r["row_id"] for r in rows
                  if r["effective_decision"] == "match" and not r["human_decision"])
    if not auto or rate <= 0:
        return set()
    count = max(1, round(len(auto) * float(rate)))
    return set(random.Random(f"audit|{round_label}").sample(
        auto, min(count, len(auto))))


def review_export(config: Dict[str, Any], manifest: Dict[str, Any],
                  round_label: str, csv_path: Optional[str] = None) -> Path:
    gold = load_gold()
    rows = build_review_rows(config, manifest, round_label, gold,
                             load_decisions(round_label))
    if not rows:
        raise SystemExit(f"no match artifacts for {round_label} — run match first")
    audit = _audit_flags(rows, round_label,
                         float((config.get("gates") or {}).get(
                             "supported_audit_rate", 0.1)))
    ordered = sorted(rows, key=lambda r: (
        0 if r["effective_decision"] in BLOCKING_DECISIONS else 1,
        r["property"], r["photo"], r["observation"], r["variant"], r["repeat"]))
    out = Path(csv_path) if csv_path else review_dir_for(round_label) / "review.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow({**row,
                             "audit": "audit" if row["row_id"] in audit else ""})
    pending = sum(1 for r in ordered
                  if r["effective_decision"] in BLOCKING_DECISIONS)
    print(f"review export: {len(ordered)} rows ({pending} needing adjudication, "
          f"{len(audit)} audit spot-checks) -> {out}")
    return out


def review_import(config: Dict[str, Any], manifest: Dict[str, Any],
                  round_label: str, csv_path: str) -> Path:
    """Read an edited CSV back into decisions.json.

    A blank human_decision clears any stored decision for that row, so a
    reviewer can take one back. A decision then carries to every other
    undecided row repeating the same observation text on the same photo —
    generation repeats regenerate near-identical claims, and the same sentence
    about the same photo cannot warrant two different verdicts.
    """
    from tools.comparison_common import sha256_file
    gold = load_gold()
    by_photo_ids = gold_ids_by_photo(gold)
    gold_sha = sha256_file(GOLD_PATH)
    rows_by_id = {r["row_id"]: r for r in build_review_rows(
        config, manifest, round_label, gold, {})}

    errors: List[str] = []
    explicit: Dict[str, Any] = {}
    cleared: List[str] = []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        for lineno, raw in enumerate(csv.DictReader(handle), start=2):
            row_id = (raw.get("row_id") or "").strip()
            if not row_id:
                continue
            if row_id not in rows_by_id:
                errors.append(f"line {lineno}: unknown row_id {row_id!r}")
                continue
            decision = (raw.get("human_decision") or "").strip().lower()
            if not decision:
                cleared.append(row_id)
                continue
            if decision not in HUMAN_DECISIONS:
                errors.append(
                    f"line {lineno}: human_decision {decision!r} not one of "
                    f"{list(HUMAN_DECISIONS)}")
                continue
            _, _, photo_ref, _ = parse_review_row_id(row_id)
            ids = _split_ids(raw.get("corrected_gold_ids"))
            if decision == "match" and not ids:
                ids = _split_ids(raw.get("proposed_gold_ids"))
            unknown = sorted(set(ids) - by_photo_ids.get(photo_ref, set()))
            if unknown:
                errors.append(
                    f"line {lineno}: gold ids {unknown} are not on {photo_ref}")
                continue
            if decision == "match" and not ids:
                errors.append(f"line {lineno}: 'match' needs at least one gold id")
                continue
            explicit[row_id] = {
                "decision": decision, "gold_ids": ids,
                "critical": _truthy(raw.get("critical")),
                "note": (raw.get("reviewer_note") or "").strip(),
                "gold_sha256": gold_sha, "imported_at": _utcnow(),
            }
    if errors:
        raise SystemExit("review import rejected:\n  " + "\n  ".join(errors))

    merged = load_decisions(round_label)
    for row_id in cleared:
        merged.pop(row_id, None)
    merged.update(explicit)
    signatures = {
        (rows_by_id[row_id]["property"], rows_by_id[row_id]["photo"],
         _norm_text(rows_by_id[row_id]["observation"])): row_id
        for row_id in explicit
    }
    propagated = 0
    for row_id, row in rows_by_id.items():
        if row_id in merged:
            continue
        source = signatures.get(
            (row["property"], row["photo"], _norm_text(row["observation"])))
        if source:
            merged[row_id] = {**explicit[source], "propagated_from": source}
            propagated += 1

    out = decisions_path_for(round_label)
    _write_json(out, {"round": round_label, "updated_at": _utcnow(),
                      "decisions": merged})
    print(f"review import: {len(explicit)} decisions, {propagated} propagated to "
          f"repeated observations, {len(cleared)} rows left undecided -> {out}")
    return out


def stage_repin_gold(manifest: Dict[str, Any]) -> str:
    """Acknowledge an edit to the human gold; invalidates every match stage."""
    from tools.comparison_common import sha256_file
    actual = sha256_file(GOLD_PATH)
    previous = manifest.get("gold_sha256")
    manifest["gold_sha256"] = actual
    manifest["gold_frozen_at"] = _utcnow()
    _write_json(MANIFEST_PATH, manifest)
    print(f"gold pin: {previous} -> {actual}")
    print("Existing match stages are now stale; rerun match for every round.")
    return actual


# ---------------------------------------------------------------------------
# Metrics + report
# ---------------------------------------------------------------------------

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def variant_stability_metrics(
    stage_label: str, manifest: Dict[str, Any], repeats: int,
) -> Dict[str, Any]:
    """Deterministic (judge-free) per-variant metrics."""
    per_property: Dict[str, Any] = {}
    for prop_key in sorted(manifest["properties"]):
        id_sets: Dict[int, set] = {}
        obs_counts: Dict[int, int] = {}
        excl_counts: Dict[int, int] = {}
        totals_by_rep: Dict[int, Dict[str, Any]] = {}
        for rep in range(1, repeats + 1):
            prop_dir = resolve_stage_dir(stage_label) / f"rep{rep}" / prop_key
            ids: set = set()
            n_obs = n_excl = 0
            for photo in manifest["properties"][prop_key]["photos"]:
                ckpt = _load_json(
                    _photo_ckpt_dir(prop_dir) / f"{photo['photo_key']}.json")
                sd = ckpt.get("scene_data") or {}
                n_obs += len(((sd.get("observations_struct") or {})
                              .get("observations")) or [])
                n_excl += len(sd.get("excluded_observations") or [])
                ids.update(
                    str(r.get("resolved_item_id"))
                    for r in (sd.get("resolved_items") or [])
                    if r.get("resolved_item_id")
                )
            id_sets[rep] = ids
            obs_counts[rep] = n_obs
            excl_counts[rep] = n_excl
            totals_by_rep[rep] = _load_json(totals_path_for(
                prop_dir / rep_job_id(stage_label, rep) / "photo_intel.json"))
        reps = sorted(id_sets)
        pairs = [(a, b) for i, a in enumerate(reps) for b in reps[i + 1:]]
        jaccards = [jaccard(id_sets[a], id_sets[b]) for a, b in pairs]
        midpoints = [totals_by_rep[r]["final_rehab"]["midpoint"] for r in reps]
        priced = _priced_ids_by_rep(totals_by_rep)
        all_priced = sorted({cid for per in priced.values() for cid in per})
        partial = [
            {
                "catalog_item_id": cid,
                "present_in_reps": [r for r in reps if cid in priced[r]],
                "mean_line_midpoint_usd": round(statistics.mean(
                    [priced[r][cid] for r in reps if cid in priced[r]])),
            }
            for cid in all_priced
            if 0 < sum(1 for r in reps if cid in priced[r]) < len(reps)
        ]
        per_property[prop_key] = {
            "resolved_id_jaccard_mean": round(statistics.mean(jaccards), 3) if jaccards else None,
            "resolved_id_jaccard_pairs": {f"{a}v{b}": round(j, 3)
                                          for (a, b), j in zip(pairs, jaccards)},
            "midpoints_by_rep": {str(r): totals_by_rep[r]["final_rehab"] for r in reps},
            "midpoint_spread_usd": max(midpoints) - min(midpoints) if midpoints else 0,
            "median_midpoint_usd": statistics.median(midpoints) if midpoints else 0,
            "observation_counts_by_rep": obs_counts,
            "excluded_counts_by_rep": excl_counts,
            "cost_bearing_partial_ids": partial,
            "priced_ids_by_rep": {str(r): sorted(priced[r]) for r in reps},
        }
    means = [p["resolved_id_jaccard_mean"] for p in per_property.values()
             if p["resolved_id_jaccard_mean"] is not None]
    spreads = [p["midpoint_spread_usd"] for p in per_property.values()]
    return {
        "stage": stage_label,
        "per_property": per_property,
        "mean_resolved_id_jaccard": round(statistics.mean(means), 3) if means else None,
        "median_midpoint_spread_usd": statistics.median(spreads) if spreads else 0,
    }


def judge_metrics(judgments: Dict[str, Any], repeats: int) -> Dict[str, Any]:
    """De-aliased claim-level rates per variant, plus the review queue."""
    per_variant: Dict[str, Dict[str, Any]] = {}
    review_queue: List[Dict[str, Any]] = []
    critical: Dict[str, Dict[str, List[int]]] = {}  # variant -> photo -> reps
    gold_cov: Dict[str, List[float]] = {}

    for key, j in judgments.items():
        mapping = j["blinding"]  # {"A": variant, "B": variant}
        claim_totals = {v: {"supported": 0, "unsupported": 0, "uncertain": 0}
                        for v in mapping.values()}
        for claim in (j.get("verdict") or {}).get("claims") or []:
            variant = mapping.get(claim.get("prompt"))
            if variant is None:
                continue
            label = claim.get("label")
            if label in claim_totals[variant]:
                claim_totals[variant][label] += 1
            cat = claim.get("hallucination_category")
            is_critical = (
                label == "unsupported"
                and cat in ("unsupported_moisture_or_mold", "structural_failure",
                            "hidden_system_condition", "safety_hazard")
            )
            if is_critical:
                critical.setdefault(variant, {}).setdefault(key, []).append(
                    int(claim.get("repeat") or 0))
            if label in ("unsupported", "uncertain"):
                review_queue.append({
                    "photo": key, "variant": variant, **claim,
                    "reason": "judge_flagged",
                })
        for variant, counts in claim_totals.items():
            agg = per_variant.setdefault(variant, {
                "supported": 0, "unsupported": 0, "uncertain": 0})
            for k, v in counts.items():
                agg[k] += v

        gold = j.get("gold_coverage")
        if gold and (gold.get("conditions") or []):
            rows = gold["conditions"]
            for letter in ("A", "B"):
                variant = mapping[letter]
                covered = sum(1 for r in rows if r.get(f"covered_by_{letter}"))
                gold_cov.setdefault(variant, []).append(covered / len(rows))

    out: Dict[str, Any] = {"per_variant": {}, "review_queue": review_queue}
    for variant, counts in per_variant.items():
        total = sum(counts.values()) or 1
        crit = critical.get(variant, {})
        repeated_critical = {
            photo: sorted(set(reps)) for photo, reps in crit.items()
            if len(set(reps)) >= 2
        }
        recalls = gold_cov.get(variant)
        out["per_variant"][variant] = {
            "claims_total": sum(counts.values()),
            "supported": counts["supported"],
            "unsupported": counts["unsupported"],
            "uncertain": counts["uncertain"],
            "unsupported_rate_pct": round(100 * counts["unsupported"] / total, 2),
            "uncertain_rate_pct": round(100 * counts["uncertain"] / total, 2),
            "critical_hallucination_photos": {
                photo: sorted(set(reps)) for photo, reps in crit.items()},
            "critical_in_2plus_repeats": repeated_critical,
            "supported_recall_vs_gold_pct": (
                round(100 * statistics.mean(recalls), 1) if recalls else None),
        }
    return out


def evaluate_candidate_gates(
    baseline_stab: Dict[str, Any],
    candidate_stab: Dict[str, Any],
    jm: Dict[str, Any],
    baseline_variant: str,
    candidate_variant: str,
    gates: Dict[str, Any],
) -> Dict[str, Any]:
    b = jm["per_variant"].get(baseline_variant, {})
    c = jm["per_variant"].get(candidate_variant, {})
    reasons_reject: List[str] = []
    flags: List[str] = []

    new_critical = {
        photo: reps
        for photo, reps in (c.get("critical_in_2plus_repeats") or {}).items()
        if photo not in (b.get("critical_hallucination_photos") or {})
    }
    if new_critical:
        reasons_reject.append(
            f"new critical hallucination in >=2/3 repeats: {sorted(new_critical)}")

    db_rate = (c.get("unsupported_rate_pct") or 0) - (b.get("unsupported_rate_pct") or 0)
    dclaims = (c.get("unsupported") or 0) - (b.get("unsupported") or 0)
    if db_rate > float(gates["unsupported_rate_pp"]) and \
            dclaims >= int(gates["unsupported_min_claims"]):
        reasons_reject.append(
            f"unsupported rate +{db_rate:.1f}pp with {dclaims} more claims")

    br, cr = b.get("supported_recall_vs_gold_pct"), c.get("supported_recall_vs_gold_pct")
    if br is not None and cr is not None and (br - cr) > float(gates["recall_drop_pp"]):
        reasons_reject.append(f"supported recall fell {br - cr:.1f}pp")

    b_mids = [p["median_midpoint_usd"] for p in baseline_stab["per_property"].values()]
    c_mids = [p["median_midpoint_usd"] for p in candidate_stab["per_property"].values()]
    b_med, c_med = statistics.median(b_mids or [0]), statistics.median(c_mids or [0])
    mid_shift_pct = (100.0 * (c_med - b_med) / b_med) if b_med else 0.0
    if abs(mid_shift_pct) > float(gates["midpoint_shift_pct"]):
        flags.append(
            f"median pre-2f midpoint shifted {mid_shift_pct:+.1f}% — requires "
            "supported new observations to explain it (Steven review; "
            "auto-reject only if unexplained)")

    d_jaccard = None
    if candidate_stab["mean_resolved_id_jaccard"] is not None and \
            baseline_stab["mean_resolved_id_jaccard"] is not None:
        d_jaccard = round(candidate_stab["mean_resolved_id_jaccard"]
                          - baseline_stab["mean_resolved_id_jaccard"], 3)
    b_spread = baseline_stab["median_midpoint_spread_usd"]
    c_spread = candidate_stab["median_midpoint_spread_usd"]
    spread_reduction_pct = (
        round(100.0 * (b_spread - c_spread) / b_spread, 1) if b_spread else None)

    improves = (
        (d_jaccard is not None and d_jaccard >= float(gates["jaccard_gain"]))
        or (spread_reduction_pct is not None
            and spread_reduction_pct >= float(gates["spread_reduction_pct"]))
    )
    verdict = (
        "reject" if reasons_reject
        else ("advance" if improves else "no_improvement")
    )
    return {
        "baseline": baseline_variant,
        "candidate": candidate_variant,
        "verdict": verdict,
        "reject_reasons": reasons_reject,
        "flags_for_steven": flags,
        "delta_jaccard": d_jaccard,
        "spread_reduction_pct": spread_reduction_pct,
        "midpoint_shift_pct": round(mid_shift_pct, 1),
        "unsupported_rate_pp_delta": round(db_rate, 2),
    }


def property_verdict(
    base_nets: List[int], cand_nets: List[int], cand_has_critical: bool,
    base_spread: float, cand_spread: float, tie_pct: float = 25.0,
) -> str:
    """helped / tied / hurt for one property.

    A human-confirmed critical unsupported claim loses the property outright.
    Otherwise the median net observation score decides, and only an exact tie
    falls through to midpoint spread.
    """
    if cand_has_critical:
        return "hurt"
    base = statistics.median(base_nets) if base_nets else 0
    cand = statistics.median(cand_nets) if cand_nets else 0
    if cand > base:
        return "helped"
    if cand < base:
        return "hurt"
    if not base_spread and not cand_spread:
        return "tied"
    if not base_spread:
        return "hurt"      # candidate introduced spread where there was none
    if not cand_spread:
        return "helped"
    reduction_pct = 100.0 * (base_spread - cand_spread) / base_spread
    if reduction_pct >= tie_pct:
        return "helped"
    if reduction_pct <= -tie_pct:
        return "hurt"
    return "tied"


def score_match_round(
    rows: List[Dict[str, Any]], stabilities: Dict[str, Any],
    baseline_variant: str, candidate_variant: str, tie_pct: float = 25.0,
) -> Dict[str, Any]:
    """Net-observation scoring: gold coverage earned minus noise added."""
    cells: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    pending: Dict[str, int] = {k: 0 for k in BLOCKING_DECISIONS}
    critical_findings: List[Dict[str, Any]] = []
    for row in rows:
        cell = cells.setdefault(
            (row["variant"], row["property"], row["repeat"]),
            {"gold": set(), "unsupported": 0, "linkage_misses": 0})
        decision = row["effective_decision"]
        if decision in pending:
            pending[decision] += 1
        elif decision == "match":
            cell["gold"].update((row["photo"], g)
                                for g in row["effective_gold_ids"])
            if row["pipeline_status"] in LINKAGE_MISS_STATUSES:
                cell["linkage_misses"] += 1
        elif decision == "unsupported":
            cell["unsupported"] += 1
            if row["critical_flag"]:
                critical_findings.append({
                    "variant": row["variant"], "property": row["property"],
                    "photo": row["photo"], "repeat": row["repeat"],
                    "observation": row["observation"],
                    "pipeline_status": row["pipeline_status"],
                    "reviewer_note": row["reviewer_note"],
                })
    blocked = sum(pending.values()) > 0

    def _variant_property(variant: str, prop: str) -> Dict[str, Any]:
        reps = sorted(r for (v, p, r) in cells if v == variant and p == prop)
        by_rep = {r: cells[(variant, prop, r)] for r in reps}
        matches = {r: len(by_rep[r]["gold"]) for r in reps}
        unsupported = {r: by_rep[r]["unsupported"] for r in reps}
        nets = [matches[r] - unsupported[r] for r in reps]
        spread = ((stabilities.get(variant) or {}).get("per_property", {})
                  .get(prop, {}).get("midpoint_spread_usd", 0))
        return {
            "gold_matches_by_rep": {str(r): matches[r] for r in reps},
            "unsupported_by_rep": {str(r): unsupported[r] for r in reps},
            "net_by_rep": {str(r): matches[r] - unsupported[r] for r in reps},
            "median_gold_matches": statistics.median(
                [matches[r] for r in reps]) if reps else 0,
            "median_unsupported": statistics.median(
                [unsupported[r] for r in reps]) if reps else 0,
            "median_net": statistics.median(nets) if nets else 0,
            "catalog_linkage_misses": sum(by_rep[r]["linkage_misses"] for r in reps),
            "midpoint_spread_usd": spread,
            "_nets": nets,
        }

    properties: Dict[str, Any] = {}
    overall = {"helped": 0, "tied": 0, "hurt": 0}
    for prop in sorted({p for (_, p, _) in cells}):
        base = _variant_property(baseline_variant, prop)
        cand = _variant_property(candidate_variant, prop)
        has_critical = any(f["variant"] == candidate_variant and
                           f["property"] == prop for f in critical_findings)
        verdict = None
        if not blocked:
            verdict = property_verdict(
                base.pop("_nets"), cand.pop("_nets"), has_critical,
                base["midpoint_spread_usd"], cand["midpoint_spread_usd"], tie_pct)
            overall[verdict] += 1
        base.pop("_nets", None)
        cand.pop("_nets", None)
        properties[prop] = {"baseline": base, "candidate": cand,
                            "verdict": verdict}

    b_spread = (stabilities.get(baseline_variant) or {}).get(
        "median_midpoint_spread_usd", 0)
    c_spread = (stabilities.get(candidate_variant) or {}).get(
        "median_midpoint_spread_usd", 0)
    return {
        "baseline": baseline_variant,
        "candidate": candidate_variant,
        "status": "blocked" if blocked else "final",
        "pending_review": pending,
        "rows_total": len(rows),
        "overall": None if blocked else overall,
        "properties": properties,
        "critical_findings": critical_findings,
        "spread_reduction_pct": (round(100.0 * (b_spread - c_spread) / b_spread, 1)
                                 if b_spread else None),
        "tie_breaker_pct": tie_pct,
    }


def stage_report(config: Dict[str, Any], manifest: Dict[str, Any]) -> Path:
    repeats = int(config["repeats"])
    report: Dict[str, Any] = {"generated_at": _utcnow(), "stages": {}}

    gate_path = resolve_stage_dir("attribution") / "gate.json"
    if gate_path.is_file():
        report["attribution_gate"] = _load_json(gate_path)

    stabilities: Dict[str, Dict[str, Any]] = {}
    for variant in load_prompts():
        stage_label = f"variant_{variant}"
        found = find_stage_artifacts(stage_label, manifest, repeats)
        expected = len(manifest["properties"]) * repeats
        have = sum(len(r) for r in found.values())
        if have == expected:
            stabilities[variant] = variant_stability_metrics(
                stage_label, manifest, repeats)
    if "attribution" not in stabilities and \
            (resolve_stage_dir("attribution") / "rep1").is_dir():
        found = find_stage_artifacts("attribution", manifest, repeats)
        if sum(len(r) for r in found.values()) == len(manifest["properties"]) * repeats:
            stabilities["__frozen_2a_attribution__"] = variant_stability_metrics(
                "attribution", manifest, repeats)
    report["stability"] = stabilities

    report["rounds"] = {}
    # Archived image-based Sol rounds live on the flat legacy layout.
    for round_dir in sorted(RUNS_DIR.glob("judge_*")):
        judgments_path = round_dir / "judgments.json"
        if not judgments_path.is_file():
            continue
        round_label = round_dir.name[len("judge_"):]
        judgments = _load_json(judgments_path)
        jm = judge_metrics(judgments, repeats)
        baseline_variant, _, candidate_variant = round_label.partition("_vs_")
        round_report: Dict[str, Any] = {"judge_metrics": jm}
        if baseline_variant in stabilities and candidate_variant in stabilities:
            round_report["gates"] = evaluate_candidate_gates(
                stabilities[baseline_variant], stabilities[candidate_variant],
                jm, baseline_variant, candidate_variant, config["gates"],
            )
        # Supported-audit sample (seeded) for human review
        rng = random.Random(f"audit|{round_label}")
        supported: List[Dict[str, Any]] = []
        for key, j in judgments.items():
            for claim in (j.get("verdict") or {}).get("claims") or []:
                if claim.get("label") == "supported":
                    supported.append({"photo": key,
                                      "variant": j["blinding"].get(claim.get("prompt")),
                                      **claim})
        sample_n = max(1, round(len(supported)
                                * float(config["gates"]["supported_audit_rate"])))
        round_report["supported_audit_sample"] = rng.sample(
            supported, min(sample_n, len(supported)))
        report["rounds"][round_label] = round_report

    from tools import benchmark_pass2a_packages as pkg_bench
    pkg_report = pkg_bench.report_contribution(config, manifest,
                                               "baseline_vs_checklist")
    if pkg_report is not None:
        report["package_eval"] = pkg_report

    report["match_rounds"] = {}
    gold = _load_json(GOLD_PATH) if GOLD_PATH.is_file() else {"photos": {}}
    tie_pct = float((config.get("gates") or {}).get("match_tie_breaker_pct", 25.0))
    for round_dir in sorted(V3_DIR.glob("match_*")):
        round_label = round_dir.name[len("match_"):]
        if not (round_dir / "photos").is_dir():
            continue
        baseline_variant, _, candidate_variant = round_label.partition("_vs_")
        rows = build_review_rows(config, manifest, round_label, gold,
                                 load_decisions(round_label))
        if not rows:
            continue
        report["match_rounds"][round_label] = score_match_round(
            rows, stabilities, baseline_variant, candidate_variant, tie_pct)

    out_json = V3_DIR / "report.json"
    _write_json(out_json, report)
    _write_report_md(V3_DIR / "report.md", report, config)
    print(f"report -> {out_json} and {V3_DIR / 'report.md'}")
    return out_json


def _write_report_md(path: Path, report: Dict[str, Any],
                     config: Dict[str, Any]) -> None:
    lines: List[str] = ["# Pass 2a prompt ablation — benchmark report", ""]
    lines.append(f"Generated {report['generated_at']}. Dollar figures are "
                 "**pre-2f, all packages assumed confirmed** benchmark totals, "
                 "not production headlines.")
    lines.append("")
    gate = report.get("attribution_gate")
    if gate:
        lines.append(f"## Attribution gate: "
                     f"{'PASSED' if gate['passed'] else 'FAILED'}")
        for flip in gate.get("failing_flips") or []:
            lines.append(f"- MATERIAL FLIP `{flip['catalog_item_id']}` "
                         f"({flip['property_key']}): "
                         f"${flip['line_midpoint_usd']:,} "
                         f"({flip['pct_of_property_midpoint']}%), reps "
                         f"{flip['present_in_reps']}")
        for prop, row in (gate.get("properties") or {}).items():
            lines.append(f"- {prop}: median midpoint "
                         f"${row['median_midpoint']:,}, spread "
                         f"${row['midpoint_spread']:,}, "
                         f"{len(row['flips'])} partial priced ids")
        lines.append("")
    for variant, stab in (report.get("stability") or {}).items():
        lines.append(f"## Stability — {variant}")
        lines.append(f"- mean resolved-ID Jaccard: "
                     f"{stab['mean_resolved_id_jaccard']}")
        lines.append(f"- median midpoint spread: "
                     f"${stab['median_midpoint_spread_usd']:,}")
        for prop, row in stab["per_property"].items():
            lines.append(f"  - {prop}: Jaccard {row['resolved_id_jaccard_mean']}, "
                         f"spread ${row['midpoint_spread_usd']:,}, "
                         f"{len(row['cost_bearing_partial_ids'])} partial priced ids")
        lines.append("")
    if report.get("package_eval"):
        from tools import benchmark_pass2a_packages as pkg_bench
        lines.extend(pkg_bench.package_report_md_lines(report["package_eval"]))

    for round_label, mr in (report.get("match_rounds") or {}).items():
        lines.append(f"## Match round — {round_label}")
        lines.append(f"Baseline `{mr['baseline']}` vs candidate "
                     f"`{mr['candidate']}`, scored against the human gold.")
        lines.append("")
        if mr["status"] == "blocked":
            pending = ", ".join(f"{n} {k}" for k, n in mr["pending_review"].items()
                                if n)
            lines.append(f"**Not final — {pending} still to adjudicate.** "
                         "Export the review CSV, decide every flagged row, "
                         "import it, then rerun the report.")
            lines.append("")
        else:
            overall = mr["overall"]
            lines.append(f"**{overall['helped']} helped, {overall['tied']} tied, "
                         f"{overall['hurt']} hurt** across "
                         f"{len(mr['properties'])} properties.")
            lines.append("")
        lines.append("| property | gold matches | unsupported additions | "
                     "linkage misses | midpoint spread | verdict |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for prop, row in mr["properties"].items():
            base, cand = row["baseline"], row["candidate"]
            lines.append(
                f"| {prop} | {base['median_gold_matches']:g} → "
                f"{cand['median_gold_matches']:g} | "
                f"{base['median_unsupported']:g} → "
                f"{cand['median_unsupported']:g} | "
                f"{base['catalog_linkage_misses']} → "
                f"{cand['catalog_linkage_misses']} | "
                f"${base['midpoint_spread_usd']:,} → "
                f"${cand['midpoint_spread_usd']:,} | "
                f"{row['verdict'] or 'pending'} |")
        lines.append("")
        if mr["critical_findings"]:
            lines.append("### Critical findings")
            for finding in mr["critical_findings"]:
                lines.append(
                    f"- `{finding['variant']}` {finding['property']}/"
                    f"{finding['photo']} rep{finding['repeat']}: "
                    f"{finding['observation']}"
                    + (f" — {finding['reviewer_note']}"
                       if finding["reviewer_note"] else ""))
            lines.append("")
        lines.append(f"Median counts are per repeat. Row-level detail lives in "
                     f"`runs/review_{round_label}/` and "
                     f"`runs/match_{round_label}/`, not here.")
        lines.append("")

    for round_label, rr in (report.get("rounds") or {}).items():
        lines.append(f"## Judge round (archived — superseded by match) "
                     f"— {round_label}")
        for variant, m in rr["judge_metrics"]["per_variant"].items():
            lines.append(
                f"- {variant}: {m['claims_total']} claims, "
                f"unsupported {m['unsupported_rate_pct']}%, "
                f"uncertain {m['uncertain_rate_pct']}%, "
                f"recall vs gold {m['supported_recall_vs_gold_pct']}%, "
                f"critical(2+ reps) {len(m['critical_in_2plus_repeats'])}")
        gates = rr.get("gates")
        if gates:
            lines.append(f"- **verdict: {gates['verdict']}** "
                         f"(dJaccard {gates['delta_jaccard']}, spread "
                         f"-{gates['spread_reduction_pct']}%, midpoint shift "
                         f"{gates['midpoint_shift_pct']}%)")
            for r in gates["reject_reasons"]:
                lines.append(f"  - REJECT: {r}")
            for f in gates["flags_for_steven"]:
                lines.append(f"  - FLAG: {f}")
        queue = rr["judge_metrics"]["review_queue"]
        lines.append(f"- human review queue: {len(queue)} judge-flagged claims "
                     f"+ {len(rr['supported_audit_sample'])} supported-audit "
                     "samples (see report.json)")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("init")
    p_attr = sub.add_parser("attribution")
    p_attr.add_argument("--skip-preflight", action="store_true")
    p_scenes = sub.add_parser(
        "scene-capture",
        help="freeze Terra Pass 1a scenes once per photo (v3 lineage)")
    p_scenes.add_argument("--check", action="store_true",
                          help="validate the existing capture, no VLM call")
    p_run = sub.add_parser("run")
    p_run.add_argument("--variant", required=True)
    p_run.add_argument("--skip-preflight", action="store_true")
    p_judge = sub.add_parser("judge", help="legacy image-based Sol round")
    p_judge.add_argument("--round", required=True,
                         help="e.g. baseline_vs_checklist")
    p_match = sub.add_parser("match")
    p_match.add_argument("--round", required=True,
                         help="e.g. baseline_vs_checklist")
    p_match.add_argument("--dry-run", action="store_true",
                         help="build every payload and lane classification "
                              "without calling a model")
    p_review = sub.add_parser("review")
    p_review.add_argument("action", choices=["export", "import"])
    p_review.add_argument("--round", required=True)
    p_review.add_argument("--csv", default=None,
                          help="required for import; defaults to "
                               "runs/review_<round>/review.csv for export")
    sub.add_parser("repin-gold")
    p_pgold = sub.add_parser("package-gold",
                             help="package-outcome gold authoring")
    p_pgold.add_argument("action", choices=["template", "check"])
    p_peval = sub.add_parser("package-eval",
                             help="package-outcome evaluation stages")
    p_peval.add_argument("--round", required=True,
                         help="e.g. baseline_vs_checklist")
    p_peval.add_argument("--stage", dest="stage_name", default="all",
                         choices=["tail", "cells", "2f", "score", "all"])
    p_peval.add_argument("--dry-run", action="store_true",
                         help="extras-scale report; refuses the 2f stage")
    p_peval.add_argument("--budget", type=int, default=None,
                         help="stop after N VLM package-verification calls")
    p_peval.add_argument("--skip-preflight", action="store_true")
    p_prev = sub.add_parser("package-review")
    p_prev.add_argument("action", choices=["export", "import"])
    p_prev.add_argument("--round", required=True)
    p_prev.add_argument("--csv", default=None)
    p_legacy = sub.add_parser(
        "legacy-rescore",
        help="offline rescore of the frozen v2 artifacts (no VLM calls)")
    p_legacy.add_argument("--round", required=True,
                          help="e.g. baseline_vs_checklist")
    sub.add_parser("report")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    _setup_env()
    config = load_config()

    if args.stage == "init":
        stage_init(config)
        return 0

    manifest = load_manifest()
    if args.stage == "attribution":
        gate = stage_attribution(config, manifest,
                                 skip_preflight=args.skip_preflight)
        return 0 if gate["passed"] else 3
    if args.stage == "scene-capture":
        from tools import benchmark_pass2a_scenes as scenes_mod
        if args.check:
            scenes_mod.check_scene_capture(config, manifest)
        else:
            scenes_mod.stage_scene_capture(config, manifest)
        return 0
    if args.stage == "run":
        prompts = load_prompts()
        if args.variant not in prompts:
            raise SystemExit(f"unknown variant {args.variant!r}; "
                             f"choices: {sorted(prompts)}")
        gate_path = resolve_stage_dir("attribution") / "gate.json"
        if not gate_path.is_file():
            raise SystemExit("attribution stage has not been evaluated — run it first")
        if not _load_json(gate_path)["passed"]:
            raise SystemExit("attribution gate FAILED — prompt variants are "
                             "off the table until 2b/2c instability is addressed")
        from tools import benchmark_pass2a_scenes as scenes_mod
        capture = scenes_mod.load_scene_capture(config, manifest)
        run_matrix(
            config, manifest, f"variant_{args.variant}",
            variant_prompt=prompts[args.variant]["text"],
            prompt_sha=prompts[args.variant]["sha256"],
            skip_preflight=args.skip_preflight,
            stage_dir=v3_stage_dir(f"variant_{args.variant}"),
            scenes_by_prop=scenes_mod.scenes_by_property(capture),
            scene_capture_sha=capture["capture_sha256"],
        )
        return 0
    if args.stage == "judge":
        stage_judge(config, manifest, args.round)
        return 0
    if args.stage == "match":
        stage_match(config, manifest, args.round, dry_run=args.dry_run)
        return 0
    if args.stage == "review":
        if args.action == "export":
            review_export(config, manifest, args.round, args.csv)
        else:
            if not args.csv:
                raise SystemExit("review import needs --csv <edited file>")
            review_import(config, manifest, args.round, args.csv)
        return 0
    if args.stage == "repin-gold":
        stage_repin_gold(manifest)
        return 0
    if args.stage == "legacy-rescore":
        from tools import benchmark_pass2a_legacy as legacy
        legacy.stage_legacy_rescore(config, manifest, args.round)
        return 0
    if args.stage == "package-gold":
        from tools import benchmark_pass2a_packages as pkg
        if args.action == "template":
            pkg.package_gold_template(config, manifest)
        else:
            pkg.package_gold_check(config, manifest)
        return 0
    if args.stage == "package-eval":
        from tools import benchmark_pass2a_packages as pkg
        stages = (["tail", "cells", "2f"] if args.stage_name == "all"
                  else [args.stage_name])
        if "tail" in stages:
            pkg.stage_package_tail(config, manifest, args.round,
                                   skip_preflight=args.skip_preflight)
        if "cells" in stages:
            pkg.stage_package_cells(config, manifest, args.round)
        if args.dry_run:
            pkg.stage_package_2f(config, manifest, args.round, dry_run=True)
            return 0
        if "2f" in stages:
            if args.stage_name == "all" and not pkg.PACKAGE_GOLD_PATH.is_file():
                print("package gold not authored yet — stopping before the "
                      "paid 2f stage. Run package-gold template, author the "
                      "gold, then package-eval --stage 2f.")
                return 0
            pkg.stage_package_2f(config, manifest, args.round,
                                 budget=args.budget)
        if "score" in stages:
            scored = pkg.score_package_round(
                config, manifest, args.round, pkg.load_package_gold(),
                pkg.load_package_decisions())
            scored.pop("_review_rows", None)
            _write_json(pkg.EVAL_DIR / "scores.json", scored)
            print(f"scores -> {pkg.EVAL_DIR / 'scores.json'} "
                  f"(status: {scored['status']})")
        return 0
    if args.stage == "package-review":
        from tools import benchmark_pass2a_packages as pkg
        if args.action == "export":
            pkg.package_review_export(config, manifest, args.round, args.csv)
        else:
            if not args.csv:
                raise SystemExit("package-review import needs --csv")
            pkg.package_review_import(config, manifest, args.round, args.csv)
        return 0
    if args.stage == "report":
        stage_report(config, manifest)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
