"""Run the frozen Session 6 renovation-architecture shadow canary.

This is a thin coordinator around scripts/run_kind_canary.py. It reuses the
approved 18-property manifest, snapshots byte-identical images/metadata plus
code/catalog/policy/model routing, and runs two isolated shadow replicas. Each
artifact contains legacy v4 and private v5 results produced from the same
upstream issue lane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Direct `python scripts/...` runs do not put the repo root on sys.path,
    # and the freeze records the contracts/policy versions from tools/.
    sys.path.insert(0, str(ROOT))
MANIFEST_PATH = ROOT / "configs" / "kind_ontology_canary_manifest.json"
MODEL_MAP_PATH = ROOT / "benchmarks" / "configs" / "kind_canary_model_map.json"
CONFIG_PATH = ROOT / "configs" / "renovation_architecture_cutover.json"
CANARY_DRIVER = ROOT / "scripts" / "run_kind_canary.py"
CATALOG_PATH = ROOT / "tools" / "issue_catalog_kind_v2.json"
IMAGES_ROOT = Path(
    "C:/Users/Steven/IntelliJProjects/renointel-prod/public/images/properties"
)
ARTIFACT_CORPUS = Path(
    "C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SAFE_MODEL_ENV_KEYS = (
    "OPENAI_MODEL",
    "RENOVATION_TERRA_MODEL",
    "RENOVATION_TERRA_MAX_OUTPUT_TOKENS",
    "RENOVATION_SOL_MODEL",
    "RENOVATION_SOL_MAX_OUTPUT_TOKENS",
    # Local routing matters too: pass 1a runs on LM Studio, so a different
    # host/model changes upstream observations and must be part of the freeze.
    "LM_STUDIO_URL",
    "LM_STUDIO_MODEL",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )


def _dotenv_values() -> Dict[str, str]:
    values: Dict[str, str] = {}
    path = ROOT / ".env"
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _safe_model_environment() -> Dict[str, str]:
    dotenv = _dotenv_values()
    values = {
        key: str(os.environ.get(key, dotenv.get(key, "")))
        for key in SAFE_MODEL_ENV_KEYS
    }
    values["effective_terra_model"] = (
        values["RENOVATION_TERRA_MODEL"] or values["OPENAI_MODEL"]
    )
    values["effective_sol_model"] = (
        values["RENOVATION_SOL_MODEL"] or values["OPENAI_MODEL"]
    )
    return values


def _latest_metadata(property_key: str) -> Dict[str, Any]:
    """Stored listing facts (price/beds/baths/sqft/area ppsf) from the newest
    corpus artifact — the same-inputs discipline the canary depends on.

    These live at the artifact ROOT under `property_metadata`; `property` holds
    a flattened projection and `property.metadata` is null in current
    artifacts, so reading that would silently price every canary listing with
    no metadata at all.
    """
    root = ARTIFACT_CORPUS / property_key
    if not root.is_dir():
        return {}
    paths = sorted(root.glob("*/photo_intel.json"), reverse=True)
    for path in paths:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = artifact.get("property_metadata")
        if not isinstance(metadata, dict) or not metadata:
            # Pre-metadata-wiring artifacts kept it under property.metadata.
            metadata = (artifact.get("property") or {}).get("metadata")
        if isinstance(metadata, dict) and metadata:
            return metadata
    return {}


def _code_hashes() -> Dict[str, str]:
    paths = list((ROOT / "tools" / "renovation_architecture").glob("*.py"))
    paths.extend(
        ROOT / relative
        for relative in (
            "tools/artifact_writers.py",
            "tools/pipeline_config.py",
            "tools/publication_gate.py",
            "tools/rehab_packages.py",
            "tools/renovation_estimate_v4.py",
            "scripts/run_kind_canary.py",
            "scripts/run_renovation_architecture_canary.py",
        )
    )
    return {
        path.relative_to(ROOT).as_posix(): _sha256_file(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def build_freeze(
    *, manifest_path: Path = MANIFEST_PATH, model_map_path: Path = MODEL_MAP_PATH
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("properties") or []
    if manifest.get("status") != "frozen" or len(rows) != 18:
        raise RuntimeError(
            "Session 6 requires the approved frozen 18-property canary manifest"
        )

    from tools.renovation_architecture.contracts import (
        CONTRACTS_SCHEMA_VERSION,
        ENVELOPE_SCHEMA_VERSION,
        POLICY_VERSIONS,
    )

    properties: Dict[str, Any] = {}
    for row in rows:
        key = str(row["property_key"])
        image_root = IMAGES_ROOT / key
        images = sorted(
            path
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise RuntimeError(f"{key} has no canary images at {image_root}")
        metadata = _latest_metadata(key)
        properties[key] = {
            "stratum": row.get("stratum"),
            "images": [
                {"path": str(path.resolve()), "sha256": _sha256_file(path)}
                for path in images
            ],
            "property_metadata": metadata,
            "property_metadata_sha256": _canonical_sha256(metadata),
        }

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "config_sha256": _sha256_file(CONFIG_PATH),
        "model_map_path": model_map_path.relative_to(ROOT).as_posix(),
        "model_map_sha256": _sha256_file(model_map_path),
        "catalog_path": CATALOG_PATH.relative_to(ROOT).as_posix(),
        "catalog_sha256": _sha256_file(CATALOG_PATH),
        "git_commit": commit,
        "code_files": _code_hashes(),
        "contracts_schema_version": CONTRACTS_SCHEMA_VERSION,
        "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        "policy_versions": dict(POLICY_VERSIONS),
        "model_environment": _safe_model_environment(),
        "properties": properties,
    }
    payload["freeze_sha256"] = _canonical_sha256(payload)
    return payload


def _write_or_verify_freeze(path: Path, freeze: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != freeze:
            raise RuntimeError(
                f"existing freeze differs from current inputs/code: {path}; "
                "use a new output root rather than mixing canary inputs"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--model-map", type=Path, default=MODEL_MAP_PATH)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--only", help="run one property in each replica (retry helper)")
    parser.add_argument(
        "--replicate", type=int, choices=(1, 2), default=None,
        help="run ONE replica now against the same freeze (budget staging). The "
             "release gate still needs both replicas before the comparison.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--terra-stop-margin", type=int, default=None,
        help="passed to the driver: stop cleanly (rc=3) before starting a "
             "property once the remaining daily Terra budget drops below this "
             "(driver default 500,000). Set it here rather than mid-rerun — "
             "the input freeze hashes these scripts.",
    )
    args = parser.parse_args(argv)
    if args.replicates != 2:
        parser.error("the approved Session 6 stability gate requires exactly 2 replicas")

    out = args.out.resolve()
    freeze_path = out / "input_freeze.json"
    freeze = build_freeze(
        manifest_path=args.manifest.resolve(),
        model_map_path=args.model_map.resolve(),
    )
    _write_or_verify_freeze(freeze_path, freeze)

    env = dict(os.environ)
    env["KIND_ONTOLOGY_VERSION"] = "observation_kind_v2"
    env["RENOVATION_ARCHITECTURE_MODE"] = "shadow"
    # Both replicas write to isolated artifacts roots but must debit ONE daily
    # Terra budget, so the 2.5M ceiling is enforced across the whole canary
    # rather than per replica.
    env["RENOVATION_TERRA_USAGE_ROOT"] = str(out)
    # Session 9 opt-in: meter EVERY OpenAI call (upstream scene passes
    # included) against the shared daily ledgers, and fail properties closed
    # at the ceiling instead of silently continuing past the free quota.
    env["RENOVATION_VLM_BUDGET_GUARD"] = "1"
    from tools.pipeline_config import resolve_renovation_terra_daily_ceiling

    terra_daily_ceiling = resolve_renovation_terra_daily_ceiling(
        env.get("RENOVATION_TERRA_DAILY_TOKEN_CEILING")
    )
    failures = []
    numbers = (
        [args.replicate] if args.replicate else list(range(1, args.replicates + 1))
    )
    for number in numbers:
        replicate_root = out / f"run_{number}"
        command = [
            sys.executable,
            str(CANARY_DRIVER),
            "--side", "candidate",
            "--build-root", str(ROOT),
            "--out", str(replicate_root),
            "--manifest", str(args.manifest.resolve()),
            "--model-map", str(args.model_map.resolve()),
            "--input-freeze", str(freeze_path),
        ]
        if args.only:
            command.extend(["--only", args.only])
        if args.skip_preflight:
            command.append("--skip-preflight")
        if args.terra_stop_margin is not None:
            command.extend(["--terra-stop-margin", str(args.terra_stop_margin)])
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode == 3:
            # Clean budget stop from the inner driver's pre-property gate:
            # nothing failed, the daily Terra ledger is simply near the
            # ceiling. Resume tomorrow with the same command.
            print(
                json.dumps(
                    {
                        "success": False,
                        "budget_stop": True,
                        "stopped_in_replicate": number,
                        "terra_usage_root": str(out),
                        "terra_daily_token_ceiling": terra_daily_ceiling,
                        "resume": "re-run the same command after the UTC day "
                                  "rolls over; completed properties are skipped",
                    },
                    indent=2,
                )
            )
            return 3
        if result.returncode:
            failures.append({"replicate": number, "returncode": result.returncode})
    if failures:
        print(json.dumps({"success": False, "failures": failures}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "success": True,
                "freeze": str(freeze_path),
                "freeze_sha256": freeze["freeze_sha256"],
                "replicates_run": numbers,
                "replicates_required": args.replicates,
                "terra_usage_root": str(out),
                "budget_guard": True,
                "terra_daily_token_ceiling": terra_daily_ceiling,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
