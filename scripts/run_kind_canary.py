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
import hashlib
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


def _preflight_embeddings() -> bool:
    """Real-POST probe of the embeddings sidecar with the CONFIGURED model.

    /health returns 200 while the GPU device is lost, and a wrong model name
    can also answer 200 — so probe exactly what Pass 2d will ask for.
    """
    import urllib.error
    import urllib.request

    sys.path.insert(0, str(ROOT))
    from tools import pipeline_config as cfg

    url = f"{cfg.EMBEDDINGS_BASE_URL.rstrip('/')}/embeddings"
    payload = {"input": ["preflight: worn roof shingles"], "model": cfg.EMBEDDINGS_MODEL_NAME}
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
        dim = len((body.get("data") or [{}])[0].get("embedding") or [])
        if dim <= 0:
            print(f"preflight: embeddings returned no vector from {url}", file=sys.stderr)
            return False
        print(f"preflight: embeddings OK ({cfg.EMBEDDINGS_MODEL_NAME}, dim={dim})")
        return True
    except urllib.error.HTTPError as exc:
        print(f"preflight: embeddings HTTP {exc.code} from {url}: "
              f"{exc.read()[:200].decode(errors='replace')}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - any failure means do not start
        print(f"preflight: embeddings unreachable at {url}: {exc}", file=sys.stderr)
    print("Restart the embeddings sidecar before running the canary "
          "(a dead device still answers /health with 200).", file=sys.stderr)
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_images(property_key: str, entry: dict) -> list[str] | None:
    """Frozen image paths, or None when the freeze no longer matches disk.

    A canary replica is only comparable if every replica saw byte-identical
    input, so a changed/missing image fails the property instead of silently
    running on whatever is on disk now.
    """
    rows = entry.get("images") or []
    if not rows:
        print(f"{property_key}: freeze lists no images", file=sys.stderr)
        return None
    images: list[str] = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            print(f"{property_key}: frozen image is gone: {path}", file=sys.stderr)
            return None
        expected = row.get("sha256")
        if expected:
            actual = _sha256_file(path)
            if actual != expected:
                print(
                    f"{property_key}: frozen image changed on disk: {path}\n"
                    f"  expected sha256 {expected}\n  found    sha256 {actual}",
                    file=sys.stderr,
                )
                return None
        images.append(str(path))
    return images


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
    artifact, not a fresh scrape.

    The listing facts live at the artifact ROOT under `property_metadata`;
    `property.metadata` is null in current artifacts (the `property` block
    holds a flattened projection instead), so it is only a legacy fallback.
    """
    artifact_path = _latest_run_artifact(property_key)
    if artifact_path is None:
        return {}
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    metadata = artifact.get("property_metadata")
    if isinstance(metadata, dict) and metadata:
        return metadata
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
    parser.add_argument("--input-freeze", type=Path, default=None,
                        help="frozen-input JSON: run from its recorded image paths "
                             "(SHA-256 verified) and stored property metadata "
                             "instead of discovering both from disk")
    parser.add_argument("--allow-proposed", action="store_true",
                        help="run against a manifest that is not frozen yet")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="skip the embeddings probe (only when 2d is intentionally off)")
    args = parser.parse_args(argv)

    if not args.skip_preflight and not _preflight_embeddings():
        return 2

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" and not args.allow_proposed:
        print("error: manifest status is not 'frozen'; get approval or pass --allow-proposed",
              file=sys.stderr)
        return 2

    frozen_properties: dict | None = None
    if args.input_freeze is not None:
        freeze = json.loads(args.input_freeze.read_text(encoding="utf-8"))
        frozen_properties = freeze.get("properties") or {}

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
    # The analyzer CLI reads os.environ (the Node bridge normally injects
    # .env); for direct runs, bridge the backend .env ourselves without
    # overriding anything already set in the shell.
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env.setdefault(key.strip(), value.strip())
    # ARTIFACTS_ROOT in .env is stale and must not misdirect output; the CLI's
    # --artifacts-root argument is authoritative, but drop it anyway.
    env.pop("ARTIFACTS_ROOT", None)
    env.pop("ISSUE_CATALOG_PATH", None)
    if args.side == "candidate":
        env["KIND_ONTOLOGY_VERSION"] = "observation_kind_v2"
    else:
        # The pinned legacy build has no selector; make sure this shell's env
        # cannot leak one into a future build either.
        env.pop("KIND_ONTOLOGY_VERSION", None)

    failures = []
    rows = [row for row in manifest["properties"]
            if not args.only or row["property_key"] == args.only]
    for index, row in enumerate(rows, start=1):
        key = row["property_key"]
        # Skip only on a REAL artifact. A failed run still leaves an empty run
        # directory behind, and treating that as "done" would silently drop the
        # property from the canary.
        if any((out_root / key).glob("*/photo_intel.json")):
            print(f"[{index}/{len(rows)}] {key}: already has an artifact, skipping (delete to re-run)")
            continue
        if frozen_properties is not None:
            entry = frozen_properties.get(key)
            if not isinstance(entry, dict):
                print(f"[{index}/{len(rows)}] {key}: manifest/freeze drift — not in the "
                      "input freeze", file=sys.stderr)
                failures.append(key)
                continue
            frozen = _frozen_images(key, entry)
            if frozen is None:
                print(f"[{index}/{len(rows)}] {key}: FROZEN INPUT MISMATCH — skipping",
                      file=sys.stderr)
                failures.append(key)
                continue
            images = frozen
            metadata = entry.get("property_metadata") or {}
        else:
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
