"""Drive the Task 4A kind-ontology canary: run every manifest property through
one build ("side") against the same stored images and metadata.

Run once per side, then compare:

  # baseline: pinned legacy build (worktree at a1972cf), legacy behavior
  .venv/Scripts/python.exe scripts/run_kind_canary.py --side baseline \
      --build-root ../rv-legacy-a1972cf --out artifacts_canary

  # candidate: this branch under observation_kind_v2 (publish mode)
  .venv/Scripts/python.exe scripts/run_kind_canary.py --side candidate \
      --build-root . --out artifacts_canary

  .venv/Scripts/python.exe tools/compare_kind_cutover.py \
      --baseline artifacts_canary/baseline --candidate artifacts_canary/candidate \
      --report reports/kind_cutover_canary_<date>.json

Both sides default to the same model map (benchmarks/configs/kind_canary_model_map.json)
so model choice is controlled and only the code/ontology differs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES_ROOT = Path("C:/Users/Steven/IntelliJProjects/renointel-prod/public/images/properties")
ARTIFACT_CORPUS = Path("C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts")
MANIFEST_PATH = ROOT / "configs" / "kind_ontology_canary_manifest.json"
DEFAULT_MODEL_MAP = ROOT / "benchmarks" / "configs" / "kind_canary_model_map.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _latest_run_artifact(property_key: str) -> Path | None:
    prop_dir = ARTIFACT_CORPUS / property_key
    if not prop_dir.is_dir():
        return None
    runs = sorted(
        (d for d in prop_dir.iterdir() if d.is_dir() and (d / "photo_intel.json").is_file()),
        key=lambda d: d.name, reverse=True,
    )
    return runs[0] / "photo_intel.json" if runs else None


def _stored_property_metadata(property_key: str) -> dict:
    """Same-inputs discipline: metadata comes from the stored baseline
    artifact, not a fresh scrape."""
    artifact_path = _latest_run_artifact(property_key)
    if artifact_path is None:
        return {}
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    prop = artifact.get("property")
    if isinstance(prop, dict):
        metadata = prop.get("metadata")
        if isinstance(metadata, dict) and metadata:
            return metadata
    return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--build-root", type=Path, required=True,
                        help="repo root of the build to run (worktree for baseline, . for candidate)")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts_canary")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--model-map", type=Path, default=DEFAULT_MODEL_MAP,
                        help="model map JSON applied to BOTH sides so models are controlled")
    parser.add_argument("--only", help="run a single property_key (retry helper)")
    parser.add_argument("--allow-proposed", action="store_true",
                        help="run against a manifest that is not frozen yet")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" and not args.allow_proposed:
        print("error: manifest status is not 'frozen'; get approval or pass --allow-proposed",
              file=sys.stderr)
        return 2

    routing = json.loads(args.model_map.read_text(encoding="utf-8"))
    model_map = routing.get("model_map") or {}
    reasoning_map = routing.get("reasoning_map") or {}
    build_root = args.build_root.resolve()
    analyzer = build_root / "tools" / "analyzer_cli.py"
    python_exe = build_root / ".venv" / "Scripts" / "python.exe"
    if not python_exe.is_file():
        python_exe = ROOT / ".venv" / "Scripts" / "python.exe"
    out_root = (args.out / args.side).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if args.side == "candidate":
        env["KIND_ONTOLOGY_VERSION"] = "observation_kind_v2"
        env.pop("ISSUE_CATALOG_PATH", None)
    else:
        # The pinned legacy build has no selector; make sure this shell's env
        # cannot leak one into a future build either.
        env.pop("KIND_ONTOLOGY_VERSION", None)

    failures = []
    rows = [row for row in manifest["properties"]
            if not args.only or row["property_key"] == args.only]
    for index, row in enumerate(rows, start=1):
        key = row["property_key"]
        if (out_root / key).is_dir() and any((out_root / key).iterdir()):
            print(f"[{index}/{len(rows)}] {key}: already has output, skipping (delete to re-run)")
            continue
        images = sorted(
            str(p) for p in (IMAGES_ROOT / key).iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            print(f"[{index}/{len(rows)}] {key}: NO IMAGES on disk — skipping", file=sys.stderr)
            failures.append(key)
            continue
        metadata = _stored_property_metadata(key)
        cmd = [
            str(python_exe), str(analyzer),
            "--property-key", key,
            "--images", *images,
            "--artifacts-root", str(out_root),
            "--model-map", json.dumps(model_map),
        ]
        if reasoning_map:
            cmd += ["--reasoning-map", json.dumps(reasoning_map)]
        if metadata:
            cmd += ["--property-metadata-json", json.dumps(metadata)]
        print(f"[{index}/{len(rows)}] {key}: {len(images)} images ({row['stratum']})")
        result = subprocess.run(cmd, cwd=str(build_root), env=env)
        if result.returncode != 0:
            print(f"[{index}/{len(rows)}] {key}: FAILED rc={result.returncode}", file=sys.stderr)
            failures.append(key)

    print(f"\n{args.side}: {len(rows) - len(failures)}/{len(rows)} completed"
          + (f"; failures: {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
