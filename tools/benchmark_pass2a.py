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
  run          k full repeats per photo for one prompt variant
  judge        one blinded gpt-5.6-sol call per photo grading every repeat's
               atomic claims, plus one text-only gold-coverage call
  report       metrics, decision gates, and the human-review queue

Run from the pass2a_prompt_bench worktree (functional baseline cdefc2b):

  .venv/Scripts/python.exe tools/benchmark_pass2a.py init
  .venv/Scripts/python.exe tools/benchmark_pass2a.py attribution
  .venv/Scripts/python.exe tools/benchmark_pass2a.py run --variant baseline
  .venv/Scripts/python.exe tools/benchmark_pass2a.py judge --round baseline_vs_checklist
  .venv/Scripts/python.exe tools/benchmark_pass2a.py report

Dollar figures produced here are "pre-2f, all packages assumed confirmed"
benchmark totals, NOT production headlines: Pass 2f never runs, and every
inferred package is force-confirmed so package-member line items price
instead of being zeroed by the not_run gate (see
tools/rehab_packages.py apply_package_verifications_to_candidates).
"""
from __future__ import annotations

import argparse
import asyncio
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


def compute_fingerprint(
    manifest: Dict[str, Any],
    config: Dict[str, Any],
    prompt_sha: Optional[str],
) -> Dict[str, Any]:
    """Everything that must be identical for a resume to be valid."""
    from tools.comparison_common import sha256_canonical, sha256_file
    from tools import pipeline_config as cfg
    image_hashes = {
        f"{prop}/{photo['photo_key']}": photo["image_sha256"]
        for prop, photo in manifest_photos(manifest)
    }
    frozen_hashes = {
        f"{prop}/{photo['photo_key']}": photo["frozen_2a_sha256"]
        for prop, photo in manifest_photos(manifest)
    }
    return {
        "git_head": _git_head(),
        "image_hashes_sha": sha256_canonical(image_hashes),
        "frozen_2a_sha": sha256_canonical(frozen_hashes),
        "prompt_sha256": prompt_sha,
        "model_overrides": config["model_overrides"],
        "reasoning_efforts": config["reasoning_efforts"],
        "pass_toggles": config["pass_toggles"],
        "catalog_sha256": sha256_file(Path(cfg.ISSUE_CATALOG_PATH)),
        "embeddings_model": cfg.EMBEDDINGS_MODEL_NAME,
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
    concurrency: int = 3,
) -> Path:
    """One repeat of one property. Checkpoints after every photo; skips
    photos already checkpointed; returns the photo_intel.json path."""
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

    async def _analyze_one(idx: int, image_path: Path) -> Any:
        meta: Dict[str, Any] = dict(
            run_id=job_id, photo_key=image_path.name, property_key=property_key,
        )
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
            if not li.get("is_valid_detection"):
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
) -> Dict[str, Dict[int, Path]]:
    """Run k repeats of every manifest property. Returns
    {property_key: {rep: artifact_path}}. Fully resumable."""
    if not skip_preflight and not preflight_embeddings():
        raise SystemExit(2)
    stage_dir = stage_dir_for(stage_label)
    fingerprint = compute_fingerprint(manifest, config, prompt_sha)
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
            path = (stage_dir_for(stage_label) / f"rep{rep}" / prop_key
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
    _write_json(stage_dir_for("attribution") / "gate.json", gate)
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
    prop_dir = stage_dir_for(stage_label) / f"rep{rep}" / property_key
    ckpt = _load_json(_photo_ckpt_dir(prop_dir) / f"{photo_key}.json")
    scene_data = ckpt.get("scene_data") or {}
    claims = [
        (o.get("description") or "").strip()
        for o in ((scene_data.get("observations_struct") or {}).get("observations") or [])
        if (o.get("description") or "").strip()
    ]
    kept = [
        {"description": o.get("description"), "kind": o.get("kind")}
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
            "resolved_ids": resolved_ids, "priced": priced}


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

    stage_dir = stage_dir_for(f"judge_{round_label}")
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
            prop_dir = stage_dir_for(stage_label) / f"rep{rep}" / prop_key
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


def stage_report(config: Dict[str, Any], manifest: Dict[str, Any]) -> Path:
    repeats = int(config["repeats"])
    report: Dict[str, Any] = {"generated_at": _utcnow(), "stages": {}}

    gate_path = stage_dir_for("attribution") / "gate.json"
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
            (stage_dir_for("attribution") / "rep1").is_dir():
        found = find_stage_artifacts("attribution", manifest, repeats)
        if sum(len(r) for r in found.values()) == len(manifest["properties"]) * repeats:
            stabilities["__frozen_2a_attribution__"] = variant_stability_metrics(
                "attribution", manifest, repeats)
    report["stability"] = stabilities

    report["rounds"] = {}
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

    out_json = RUNS_DIR / "report.json"
    _write_json(out_json, report)
    _write_report_md(RUNS_DIR / "report.md", report, config)
    print(f"report -> {out_json} and {RUNS_DIR / 'report.md'}")
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
    for round_label, rr in (report.get("rounds") or {}).items():
        lines.append(f"## Judge round — {round_label}")
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
    p_run = sub.add_parser("run")
    p_run.add_argument("--variant", required=True)
    p_run.add_argument("--skip-preflight", action="store_true")
    p_judge = sub.add_parser("judge")
    p_judge.add_argument("--round", required=True,
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
    if args.stage == "run":
        prompts = load_prompts()
        if args.variant not in prompts:
            raise SystemExit(f"unknown variant {args.variant!r}; "
                             f"choices: {sorted(prompts)}")
        gate_path = stage_dir_for("attribution") / "gate.json"
        if not gate_path.is_file():
            raise SystemExit("attribution stage has not been evaluated — run it first")
        if not _load_json(gate_path)["passed"]:
            raise SystemExit("attribution gate FAILED — prompt variants are "
                             "off the table until 2b/2c instability is addressed")
        run_matrix(
            config, manifest, f"variant_{args.variant}",
            variant_prompt=prompts[args.variant]["text"],
            prompt_sha=prompts[args.variant]["sha256"],
            skip_preflight=args.skip_preflight,
        )
        return 0
    if args.stage == "judge":
        stage_judge(config, manifest, args.round)
        return 0
    if args.stage == "report":
        stage_report(config, manifest)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
