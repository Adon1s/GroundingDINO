"""Session 6 persistent-worker smoke: start analyzer_server.py in a given
architecture mode, wait for readiness, run one listing, verify the artifact.

Usage:
  python worker_smoke.py --mode new     --out <artifacts_root> [--property redfin_10806500]
  python worker_smoke.py --mode current --out <artifacts_root>
  python worker_smoke.py --mode new --out <dir> \
      --dotenv C:/Users/Steven/IntelliJProjects/renointel-prod/.env --routing production

Exercises exactly what the runbook requires: a FRESH worker process per mode
(configuration is startup-time), readiness gating, one end-to-end listing. It
prints the artifact's photo_intel_path on success; verify the artifact with
scripts/verify_renovation_artifact.py (runbook §3/§4 checks).

--dotenv   which .env seeds the worker env (setdefault: the calling shell's
           exports still win). Default: the backend repo .env (canary posture).
           Pass the frontend's renointel-prod/.env to reproduce the production
           worker environment (that is what `npm run worker` loads).
--routing  canary     = the frozen canary model map (2a/2b/2c/2f -> gpt-5.6-terra,
                        low/low/low/medium, 2d local), as Session 6 ran it;
           production = what the FE sends under `npm run worker -- --premium`
                        (1a/2a/2b/2c/2d -> OPENAI_MODEL at effort none,
                        2f -> OPENAI_PASS_2F_PRIORITY_MODEL at medium,
                        modelRoutingProfile standard, detectionBackend rv, no
                        concurrency override). Model names come from the loaded env.
--local-2d           = mirror the production worker's --premium --local-2d flag;
                       retain local Qwen for catalog resolution.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve()
while ROOT.name and not (ROOT / "tools" / "analyzer_server.py").is_file():
    if ROOT.parent == ROOT:
        break
    ROOT = ROOT.parent

IMAGES_ROOT = Path("C:/Users/Steven/IntelliJProjects/renointel-prod/public/images/properties")
ARTIFACT_CORPUS = Path("C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
# Same routing as the frozen canary model map (benchmarks/configs/kind_canary_model_map.json).
MODEL_OVERRIDES = {"2a": "gpt-5.6-terra", "2b": "gpt-5.6-terra",
                   "2c": "gpt-5.6-terra", "2f": "gpt-5.6-terra"}
REASONING_EFFORTS = {"2a": "low", "2b": "low", "2c": "low", "2f": "medium"}


def _stored_metadata(property_key: str) -> dict:
    root = ARTIFACT_CORPUS / property_key
    if not root.is_dir():
        return {}
    for path in sorted(root.glob("*/photo_intel.json"), reverse=True):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = artifact.get("property_metadata")
        if isinstance(metadata, dict) and metadata:
            return metadata
    return {}


def _dotenv(path: Path) -> dict:
    """Minimal KEY=VALUE loader (no `export`, no inline comments, no quotes
    beyond a surrounding pair) — sufficient for the two .env files it is used
    on (backend repo .env, renointel-prod/.env)."""
    values = {}
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"')
    return values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--mode", required=True, choices=("current", "shadow", "new"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--property", default="redfin_10806500")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dotenv", type=Path, default=None,
                        help=".env to seed the worker env (default: <repo-root>/.env)")
    parser.add_argument("--routing", choices=("canary", "production"), default="canary",
                        help="canary = frozen canary model map; production = what the FE "
                             "sends under `npm run worker -- --premium`")
    parser.add_argument('--local-2d', action='store_true',
                        help='Mirror the production worker local Pass 2d routing flag')
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dotenv_path = (args.dotenv or (repo / ".env")).resolve()

    images = sorted(
        str(p) for p in (IMAGES_ROOT / args.property).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"no images for {args.property}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    if not dotenv_path.is_file():
        print(f"[smoke] --dotenv file not found: {dotenv_path}", file=sys.stderr)
        return 2
    for key, value in _dotenv(dotenv_path).items():
        env.setdefault(key, value)
    env.pop("ARTIFACTS_ROOT", None)
    env.pop("ISSUE_CATALOG_PATH", None)
    # Rollback per the runbook changes ONLY the architecture mode: the kind
    # ontology selector stays where production has it, so the rollback smoke
    # proves the estimator toggles independently of the catalog selector.
    env["KIND_ONTOLOGY_VERSION"] = "observation_kind_v2"
    env["RENOVATION_ARCHITECTURE_MODE"] = args.mode
    # Make the effective (non-secret) worker env visible: setdefault means a
    # value already exported in the calling shell silently beats the .env file.
    print(f"[smoke] dotenv={dotenv_path} routing={args.routing} mode={args.mode}")
    for key in ("OPENAI_MODEL", "OPENAI_PASS_2F_PRIORITY_MODEL", "LM_STUDIO_URL",
                "LM_STUDIO_MODEL", "RENOVATION_TERRA_MODEL", "RENOVATION_SOL_MODEL",
                "RENOVATION_VLM_BUDGET_GUARD", "RENOVATION_TERRA_USAGE_ROOT",
                "PASS_2D_TEMPERATURE",
                "KIND_ONTOLOGY_VERSION", "RENOVATION_ARCHITECTURE_MODE"):
        print(f"[smoke]   {key}={env.get(key, '')!r}")

    if args.routing == "production":
        upstream = (env.get("OPENAI_MODEL") or "").strip()
        pass_2f = (env.get("OPENAI_PASS_2F_PRIORITY_MODEL") or "").strip()
        if not upstream or not pass_2f:
            print("[smoke] --routing production needs OPENAI_MODEL and "
                  "OPENAI_PASS_2F_PRIORITY_MODEL in the loaded env", file=sys.stderr)
            return 2
        model_overrides = {k: upstream for k in ("1a", "2a", "2b", "2c", "2d")}
        model_overrides["2f"] = pass_2f
        reasoning_efforts = {k: "none" for k in ("1a", "2a", "2b", "2c", "2d")}
        reasoning_efforts["2f"] = "medium"
        if args.local_2d:
            model_overrides.pop('2d', None)
            reasoning_efforts.pop('2d', None)
        routing_fields = {"modelRoutingProfile": "standard", "detectionBackend": "rv"}
    else:
        model_overrides = dict(MODEL_OVERRIDES)
        reasoning_efforts = dict(REASONING_EFFORTS)
        routing_fields = {"modelRoutingProfile": "premium", "concurrency": 4}
    print(f"[smoke]   modelOverrides={model_overrides}")
    print(f"[smoke]   reasoningEfforts={reasoning_efforts}")

    proc = subprocess.Popen(
        [str(repo / ".venv" / "Scripts" / "python.exe"),
         str(repo / "tools" / "analyzer_server.py")],
        cwd=str(repo), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    stderr_tail: list[str] = []

    def _drain():
        for line in proc.stderr:
            stderr_tail.append(line.rstrip())
            del stderr_tail[:-40]

    threading.Thread(target=_drain, daemon=True).start()

    state = {"ready": False, "done": False, "error": None}
    timer = threading.Timer(args.timeout, proc.kill)
    timer.start()
    try:
        # 1. readiness
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "ready":
                state["ready"] = True
                print(f"[smoke] worker ready in mode={args.mode}")
                break
            if msg.get("type") == "error":
                state["error"] = msg
                print(f"[smoke] STARTUP REFUSED: {json.dumps(msg)}")
                break
        if not state["ready"]:
            print("[smoke] worker never signalled ready")
            print("\n".join(stderr_tail[-15:]))
            return 1

        # 2. one listing
        request = {
            "type": "job",
            "jobId": f"smoke_{args.mode}",
            "runId": f"smoke_{args.mode}_run",
            "propertyKey": args.property,
            "images": images,
            "artifactsRoot": str(out),
            "analysisProfile": "standard",
            "modelOverrides": model_overrides,
            "reasoningEfforts": reasoning_efforts,
            "propertyMetadata": _stored_metadata(args.property),
            **routing_fields,
        }
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        print(f"[smoke] submitted {args.property} ({len(images)} images)")

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type")
            if kind == "progress":
                continue
            if kind == "error":
                state["error"] = msg
                print(f"[smoke] JOB ERROR: {json.dumps(msg)[:600]}")
            if kind == "result":
                print(f"[smoke] result: {json.dumps(msg)[:300]}")
                state["photo_intel_path"] = msg.get("photo_intel_path")
                state["success"] = msg.get("success")
            if kind == "job_done":
                state["done"] = True
                break
    finally:
        timer.cancel()
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()

    if state["error"] and not state["done"]:
        print("\n".join(stderr_tail[-15:]))
        return 1
    photo_intel_path = state.get("photo_intel_path")
    print(f"[smoke] job_done={state['done']} success={state.get('success')}")
    print(f"[smoke] photo_intel_path={photo_intel_path}")
    if not state["done"]:
        return 1
    if state.get("success") is False or not photo_intel_path:
        # job_done without a published artifact (e.g. new-mode
        # AuthoritativeEstimateIncomplete) is a failed smoke, not a pass.
        print("\n".join(stderr_tail[-15:]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
