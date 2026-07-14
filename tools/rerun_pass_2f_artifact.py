"""Replay Pass 2f against a saved photo_intel.json artifact.

This helper avoids rerunning the earlier image-analysis passes. It rebuilds
estimate candidates from the persisted artifact inputs, runs the existing
Pass 2f batch reviewer, recomputes renovation estimates, and patches the
artifact in place after writing a timestamped backup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import pipeline_config as cfg
from tools.artifact_writers import (
    _strip_pass_2f_audit_rationale,
    load_issue_catalog,
)
from tools.backfill_reno_v4 import _resolve_property_metadata_from_artifact
from tools.renovation_estimate_v4 import compute_renovation_estimate_v4
from tools.rehab_packages import infer_package_candidates
from tools.vlm_client import create_vlm_client, get_model_configs_from_pipeline_config
from tools.pass_2f_artifact_inputs import (
    photo_key_to_path as _photo_key_to_path,
    prepare_replay_inputs as _prepare_replay,
    resolve_artifact_path as _resolve_artifact_path,
    select_issues_flat as _select_issues_flat,
)




def _env_pass_2f_model() -> Optional[str]:
    return os.environ.get("OPENAI_PASS_2F_MODEL") or os.environ.get("OPENAI_PASS2F_MODEL")


def _resolve_model_config(
    *,
    provider: str,
    model_override: Optional[str],
    cfg_module: Any,
    require_api_key: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Return (model_config, source, error_reason)."""
    local_config, premium_config = get_model_configs_from_pipeline_config(cfg_module)
    if provider == "premium":
        model_config = dict(premium_config or {})
        source = "premium_default"
        env_model = _env_pass_2f_model()
        if env_model:
            model_config["model"] = env_model
            source = "env_override"
        if model_override:
            model_config["model"] = model_override
            source = "cli_override"
        if not model_config.get("model"):
            return None, None, "missing premium model config"
        if require_api_key and not model_config.get("api_key"):
            return None, None, "missing OpenAI API key for premium provider"
        model_config["provider"] = "openai"
        return model_config, source, None

    if provider == "local":
        model_config = dict(local_config or {})
        source = "standard_default"
        if model_override:
            model_config["model"] = model_override
            source = "cli_override"
        if not model_config.get("model"):
            return None, None, "missing local model config"
        model_config["provider"] = model_config.get("provider") or "lmstudio"
        return model_config, source, None

    return None, None, f"unsupported provider: {provider}"


def _pass_2f_counts(v4_estimate: Any) -> Dict[str, Any]:
    trace = v4_estimate.get("pass_2f_trace") if isinstance(v4_estimate, dict) else {}
    if not isinstance(trace, dict):
        trace = {}
    return {
        "package_candidate_count": int(trace.get("candidate_count") or 0),
        "attempted_count": int(trace.get("attempted_count") or 0),
        "confirmed_count": int(trace.get("confirmed_count") or 0),
        "rejected_count": int(trace.get("rejected_count") or 0),
        "uncertain_count": int(trace.get("uncertain_count") or 0),
        "no_image_count": int(trace.get("no_image_count") or 0),
    }


def _raw_total(estimate: Any) -> Dict[str, int]:
    if not isinstance(estimate, dict):
        return {"low": 0, "high": 0}
    for key in ("raw_totals", "primary_estimate", "final_rehab"):
        value = estimate.get(key)
        if isinstance(value, dict):
            return {
                "low": int(value.get("low") or 0),
                "high": int(value.get("high") or 0),
            }
    return {"low": 0, "high": 0}


def _model_routing_entry(
    *,
    provider: str,
    model_config: Dict[str, Any],
    source: str,
) -> Dict[str, str]:
    return {
        "pass": "2f",
        "model_family": "gpt5" if provider == "premium" else "qwen",
        "model": str(model_config.get("model") or ""),
        "source": source,
    }


def _replace_pass_2f_routing(
    artifact: Dict[str, Any],
    entry: Dict[str, str],
) -> None:
    routing = artifact.get("model_routing")
    if not isinstance(routing, list):
        artifact["model_routing"] = [entry]
        return
    artifact["model_routing"] = [
        item for item in routing
        if not (isinstance(item, dict) and item.get("pass") == "2f")
    ]
    artifact["model_routing"].append(entry)


def _write_patched_artifact(
    artifact_path: Path,
    artifact: Dict[str, Any],
    *,
    backup_dir: Path,
) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{artifact_path.stem}.pre_2f_replay_{stamp}{artifact_path.suffix}"
    shutil.copy2(artifact_path, backup_path)

    tmp_path = artifact_path.parent / (artifact_path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)
        tmp_path.replace(artifact_path)
    except OSError:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return backup_path


def _debug_note_path(artifact_path: Path) -> Optional[Path]:
    debug_path = artifact_path.parent / "photo_intel_debug.json"
    return debug_path if debug_path.is_file() else None




def replay_pass_2f_artifact(
    artifact_path: Path,
    catalog: Dict[str, Any],
    *,
    provider: str = "premium",
    model_override: Optional[str] = None,
    dry_run: bool = False,
    backup_dir: Optional[Path] = None,
    cfg_module: Any = cfg,
    vlm_client: Any = None,
) -> Dict[str, Any]:
    """Replay Pass 2f and patch a saved artifact.

    Returns a status dictionary and never raises for expected user-facing
    failures.
    """
    try:
        with artifact_path.open("r", encoding="utf-8") as f:
            artifact = json.load(f)
    except json.JSONDecodeError as e:
        return {"status": "error", "reason": f"invalid JSON: {e}"}
    except OSError as e:
        return {"status": "error", "reason": f"cannot read file: {e}"}

    if not isinstance(artifact, dict):
        return {"status": "error", "reason": "artifact root must be a JSON object"}

    issues_flat, candidates, photo_map = _prepare_replay(artifact, catalog)
    if issues_flat is None:
        return {
            "status": "error",
            "reason": "artifact has no usable estimate_issues_flat or issues_flat",
        }

    package_candidates = infer_package_candidates(candidates, [], catalog)
    if not package_candidates:
        return {
            "status": "skip_no_eligible",
            "candidate_count": len(candidates),
            "package_candidate_count": 0,
        }

    model_config, model_source, model_error = _resolve_model_config(
        provider=provider,
        model_override=model_override,
        cfg_module=cfg_module,
        require_api_key=not dry_run and provider == "premium",
    )
    if model_error:
        return {"status": "error", "reason": model_error}
    assert model_config is not None
    assert model_source is not None

    old_total = _raw_total(artifact.get("renovation_estimate_v4"))
    if dry_run:
        return {
            "status": "dry_run",
            "provider": provider,
            "model": model_config.get("model"),
            "model_source": model_source,
            "candidate_count": len(candidates),
            "package_candidate_count": len(package_candidates),
            "photo_path_count": len(photo_map),
            "old_total": old_total,
            "debug_note_path": str(_debug_note_path(artifact_path) or ""),
        }

    client = vlm_client if vlm_client is not None else create_vlm_client()

    try:
        property_metadata = _resolve_property_metadata_from_artifact(artifact)
        v4_est = compute_renovation_estimate_v4(
            issues_flat=issues_flat,
            issue_catalog=catalog,
            photos=artifact.get("photos") or {},
            property_metadata=property_metadata or None,
            pass_2f_vlm_client=client,
            pass_2f_model_config=model_config,
            photo_key_to_path=photo_map,
            pass_2f_provider=provider,
        )
    except Exception as e:
        return {"status": "error", "reason": f"estimate recomputation failed: {e}"}

    patched = dict(artifact)
    patched["renovation_estimate_v4"] = v4_est
    _replace_pass_2f_routing(
        patched,
        _model_routing_entry(
            provider=provider,
            model_config=model_config,
            source=model_source,
        ),
    )
    _strip_pass_2f_audit_rationale(patched.get("renovation_estimate_v4"))

    try:
        backup_path = _write_patched_artifact(
            artifact_path,
            patched,
            backup_dir=backup_dir or artifact_path.parent,
        )
    except OSError as e:
        return {"status": "error", "reason": f"write failed: {e}"}

    counts = _pass_2f_counts(v4_est)
    return {
        "status": "updated",
        "provider": provider,
        "model": model_config.get("model"),
        "model_source": model_source,
        "candidate_count": len(candidates),
        **counts,
        "old_total": old_total,
        "new_total": _raw_total(v4_est),
        "backup_path": str(backup_path),
        "debug_note_path": str(_debug_note_path(artifact_path) or ""),
    }


def _format_money_range(total: Dict[str, int]) -> str:
    return f"${int(total.get('low') or 0):,}-${int(total.get('high') or 0):,}"


def _format_result(artifact_path: Path, result: Dict[str, Any]) -> str:
    status = result.get("status")
    if status == "dry_run":
        note = (
            f"; debug unchanged: {result['debug_note_path']}"
            if result.get("debug_note_path")
            else ""
        )
        return (
            f"dry_run: {artifact_path} | provider={result['provider']} "
            f"model={result['model']} source={result['model_source']} "
            f"candidates={result['candidate_count']} "
            f"package_candidates={result['package_candidate_count']} "
            f"photo_paths={result['photo_path_count']} "
            f"old_total={_format_money_range(result['old_total'])}{note}"
        )
    if status == "skip_no_eligible":
        return (
            f"skip_no_eligible: {artifact_path} | "
            f"candidates={result.get('candidate_count', 0)} package_candidates=0"
        )
    if status == "updated":
        note = (
            f"; debug unchanged: {result['debug_note_path']}"
            if result.get("debug_note_path")
            else ""
        )
        return (
            f"updated: {artifact_path} | provider={result['provider']} "
            f"model={result['model']} source={result['model_source']} "
            f"package_candidates={result['package_candidate_count']} "
            f"attempted={result['attempted_count']} confirmed={result['confirmed_count']} "
            f"rejected={result['rejected_count']} uncertain={result['uncertain_count']} "
            f"old_total={_format_money_range(result['old_total'])} "
            f"new_total={_format_money_range(result['new_total'])} "
            f"backup={result['backup_path']}{note}"
        )
    return f"error: {artifact_path} - {result.get('reason', 'unknown error')}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rerun only Pass 2f against an existing photo_intel artifact.",
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to photo_intel.json or a run directory containing it.",
    )
    parser.add_argument(
        "--catalog",
        help="Path to issue catalog JSON. Defaults to pipeline_config.ISSUE_CATALOG_PATH.",
    )
    parser.add_argument(
        "--provider",
        choices=("premium", "local"),
        default="premium",
        help="Model provider for Pass 2f replay.",
    )
    parser.add_argument(
        "--model",
        help="Override the selected Pass 2f model name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print candidate and model summary without running 2f or writing.",
    )
    parser.add_argument(
        "--backup-dir",
        help="Directory for the timestamped backup. Defaults to the artifact directory.",
    )
    args = parser.parse_args(argv)

    try:
        artifact_path = _resolve_artifact_path(Path(args.run))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    catalog_path = Path(args.catalog) if args.catalog else Path(cfg.ISSUE_CATALOG_PATH)
    if not catalog_path.is_file():
        print(f"error: catalog not found: {catalog_path}", file=sys.stderr)
        return 1

    try:
        catalog = load_issue_catalog(catalog_path)
    except Exception as e:
        print(f"error: failed to load catalog {catalog_path}: {e}", file=sys.stderr)
        return 1

    result = replay_pass_2f_artifact(
        artifact_path,
        catalog,
        provider=args.provider,
        model_override=args.model,
        dry_run=args.dry_run,
        backup_dir=Path(args.backup_dir) if args.backup_dir else None,
    )
    line = _format_result(artifact_path, result)
    if result.get("status") == "error":
        print(line, file=sys.stderr)
        return 1
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
