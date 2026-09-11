"""Catalog-audit Session 6, Part A: cost-gated live Stage A validation (evidence only).

Replays the 66 pinned cases of reports/catalog_audit_live_experiment_manifest.json from their
FROZEN Pass 2c observations through the live pipeline tail -- Pass 2d resolution, condition
projection, Terra verification, disposition/routing, standalone pricing -- in two arms whose only
difference is the generated catalog. Pass 1a/2a/2b/2c/2e stay frozen: every observation is read
from the stored artifact named in the manifest, at its pinned sha256.

This file produces EVIDENCE, never a verdict. It applies no rubric, writes no recommendation and
no handoff; those belong to Part B, which reads the frozen bundle. The split is deliberate: the
session that spends the tokens is not the session that decides whether the spend was worth it.

Subcommands (only `run` may contact a provider):
  verify   every manifest invariant, plus the budget gate; exits non-zero before any client
  plan     dry run: target units, request bytes, reservations, call count, budget headroom
  run      --arm {baseline,candidate}: live Pass 2d (LM Studio) + Terra (OpenAI), per property
  compare  join both arms, compute the manifest's aggregates as numbers, interpret nothing
  freeze   write reports/catalog_audit_live_validation.json

Arm selection is by --arm-root, pre-parsed out of argv and placed on sys.path before `tools` is
imported, so the arm supplies both its code and its catalog. Every input path resolves against the
main checkout, never the working directory: 17 of the pinned artifacts are absolute paths into the
production tree and 49 are repo-relative, and the candidate worktree holds neither set.

Cost discipline: one Terra ledger shared by both arms at the run root, an explicit pre-call check
against the authorized figure, per-unit records written after every call, and resume keyed on the
request fingerprint. The production .checkpoints stores are never read or written -- a fingerprint
match there would silently republish a stored verdict and measure nothing.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]

BASELINE_COMMIT = "9afe0fa5a0490a856abed507dccc4a02ed1de24c"
CANDIDATE_COMMIT = "6e67eaa83887cf4d218100dcd060c07d3bd30522"
CANDIDATE_ROOT_DEFAULT = Path("C:/Users/Steven/PycharmProjects/rv-catalog-audit-s4")
PROD_ARTIFACTS_ROOT = Path("C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts")
PARENT_ID = "dated_window_treatment_valance"
BLINDS_ID = "window_blinds_basic_or_plain"
FABRIC_ID = "dated_window_valance_or_curtains"
WINDOWS_ID = "dated_or_older_windows"
TOP_K = 8  # the production effective value; the retriever's own default is 5

MANIFEST_PATH = MAIN_ROOT / "reports" / "catalog_audit_live_experiment_manifest.json"
SESSION5_FILE_SHA256 = "e1f3b7f1f967266e64c71dedb6f93cef2e329cc02dca78799cfec4a678a4a42f"
SESSION5_MANIFEST_SHA256 = "8530cca8b9df8a3b0c65c087002a3255c6d261676cd3abc02588894f97e8cbf0"
RUN_ROOT = MAIN_ROOT / "artifacts_canary" / "catalog_audit_session6_20260907"
BUNDLE_PATH = MAIN_ROOT / "reports" / "catalog_audit_live_validation.json"

# Product-lane rows owned by the deprecated parent that the manifest does not pin. Each is a whole
# condition of its own, none sits in a target unit, and each is a stale id under the candidate
# catalog -- so both arms drop them and the exclusion is reported rather than silently absorbed.
UNPINNED_PARENT_ISSUE_IDS = frozenset({
    "98f322926d11c81e",   # redfin_125970550  "Window trim and window treatment are dated."
    "cc67016e45fdbcd9",   # redfin_80925528   "Window treatments and hardware appear older."
    "62681b053d83cb31",   # redfin_10949071   "The window treatments are dated."
    "3e92a07468c0ba6c",   # redfin_80917686   "The blinds appear utilitarian."
})

# Terra's measured self-disagreement on identical inputs (16/250 canary replica flips, Wilson 95%
# [4.0%, 10.1%]). Recorded so the bundle carries it; Part A never applies it to a judgement.
TERRA_REPLICA_FLIP_RATE = 0.064

PROVIDER_MODULES = ("tools.vlm_client", "tools.openai_client")
TERRA_MODULE = "tools.renovation_architecture.terra_review"
SOL_MODULE = "tools.renovation_architecture.sol_review"


class HarnessError(RuntimeError):
    """Operator-facing failure: drift, a missing input, or a broken invariant."""


def fail(msg: str):
    raise HarnessError(msg)


def _arm_root_from_argv(argv: Sequence[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--arm-root" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if arg.startswith("--arm-root="):
            return Path(arg.split("=", 1)[1]).resolve()
    return MAIN_ROOT


def _opt_from_argv(argv: Sequence[str], flag: str) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


ARM_ROOT = _arm_root_from_argv(sys.argv[1:])
if str(ARM_ROOT) in sys.path:
    sys.path.remove(str(ARM_ROOT))
sys.path.insert(0, str(ARM_ROOT))

# The manifest pins LM_STUDIO_URL as part of its environment snapshot. That address is a transport
# route to the Pass 2d model, not an experimental variable: the routing identity is
# lmstudio/<LM_STUDIO_MODEL>, which is checked separately and must not move. When the pinned address
# is unreachable, --lm-studio-url substitutes another route TO THE SAME SERVER, identically in both
# arms, and the substitution is recorded as a validated deviation. Set before pipeline_config is
# imported, because it resolves the constant at import time; the .env loader uses setdefault, so an
# explicit value here wins.
LM_STUDIO_URL_PINNED: Optional[str] = None
LM_STUDIO_URL_OVERRIDE = _opt_from_argv(sys.argv[1:], "--lm-studio-url")
if LM_STUDIO_URL_OVERRIDE:
    os.environ["LM_STUDIO_URL"] = LM_STUDIO_URL_OVERRIDE

import tools  # noqa: E402
from tools.model_comparison_config import load_dotenv_without_override  # noqa: E402

# tools.pipeline_config resolves the model, key and ceiling constants at IMPORT time, so the .env
# has to land first or the run would resolve a different Pass 2d URL and a different Terra model
# than the manifest pins. Exported process variables still win (the loader uses setdefault). The
# main checkout's .env is the operator's single environment; the worktree carries none.
load_dotenv_without_override(MAIN_ROOT / ".env")

import tools.pipeline_config as cfg  # noqa: E402
from tools.comparison_common import sha256_bytes, sha256_canonical, sha256_file  # noqa: E402

if Path(tools.__file__).resolve().parent != (ARM_ROOT / "tools"):
    fail(f"imported tools from {tools.__file__}, expected {ARM_ROOT / 'tools'}")


def assert_not_imported(*modules: str) -> None:
    loaded = sorted(m for m in modules if m in sys.modules)
    if loaded:
        fail(f"modules that must not be loaded in this subcommand are imported: {loaded}")


# --------------------------------------------------------------------------- small helpers

def rel(path: Path, root: Path = MAIN_ROOT) -> str:
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)
    return sha256_bytes(blob)


def git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if out.returncode != 0:
        fail(f"git {' '.join(args)} in {root} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def git_blob(root: Path, path: Path) -> Optional[str]:
    out = subprocess.run(["git", "-C", str(root), "hash-object", str(path)], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_input(path_str: str) -> Path:
    """Manifest paths mix absolute (production tree) and repo-relative (canary tree). Both resolve
    against the main checkout: the candidate worktree holds neither set."""
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (MAIN_ROOT / p).resolve()


def load_manifest() -> Dict[str, Any]:
    m = read_json(MANIFEST_PATH)
    probe = {k: v for k, v in m.items() if k != "manifest_sha256"}
    if sha256_canonical(probe) != m.get("manifest_sha256"):
        fail("manifest self-hash mismatch: manifest_sha256 does not match its own content")
    return m


def manifest_content_proof(m: Dict[str, Any]) -> Dict[str, Any]:
    """Prove only the budget block moved since Session 5: restore the null block, and the canonical
    hash of everything else must come back to the Session 5 value."""
    back = json.loads(json.dumps(m))
    back["budget"] = {
        "authorized_at": None, "authorized_by": None,
        "reference_costs": m["budget"]["reference_costs"],
        "rule": m["budget"]["rule"],
        "stage_a_authorized_tokens": None, "stage_b_authorized_tokens": None,
    }
    restored = sha256_canonical({k: v for k, v in back.items() if k != "manifest_sha256"})
    return {"restored_session5_manifest_sha256": restored, "expected": SESSION5_MANIFEST_SHA256,
            "ok": restored == SESSION5_MANIFEST_SHA256}


def cases_by_run(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One group per property run, ordered findings -> controls -> affected, then manifest order."""
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for idx, case in enumerate(m["cases"]):
        key = (case["source"], case["property_key"], case["run_id"])
        g = groups.setdefault(key, {
            "source": case["source"], "property_key": case["property_key"], "run_id": case["run_id"],
            "artifact_path": case["artifact_path"], "artifact_sha256": case["artifact_sha256"],
            "cases": [], "first_index": idx,
        })
        g["cases"].append(case)

    def rank(g: Dict[str, Any]) -> Tuple[int, int]:
        roles = {c["role"] for c in g["cases"]}
        tier = 0 if "finding" in roles else (1 if "control" in roles else 2)
        return (tier, g["first_index"])
    return sorted(groups.values(), key=rank)


def product_lane(art: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The lane production feeds to the v5 seam: product_estimate_issues_flat when the raw canonical
    estimate lane is non-empty, else product_issues_flat (tools/artifact_writers.py:1045-1050).
    Reading estimate_issues_flat instead would build ~17% more conditions than the run being
    replayed ever had, and would pay Terra for them."""
    lane = art.get("product_estimate_issues_flat") if art.get("estimate_issues_flat") \
        else art.get("product_issues_flat")
    return list(lane or [])


# --------------------------------------------------------------------------- identity records

def prompt_identity(root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    passes = (root / "tools" / "scene_classifier_passes.py").read_text(encoding="utf-8")
    for name in ("PASS_2B_PROMPT_VERSION", "PASS_2C_PROMPT_VERSION",
                 "PASS_2D_PROMPT_VERSION", "PASS_2F_PROMPT_VERSION"):
        mo = re.search(rf'^{name}\s*=\s*["\']([^"\']+)["\']', passes, re.M)
        if mo:
            out[name] = mo.group(1)
    contracts = (root / "tools" / "renovation_architecture" / "contracts.py").read_text(encoding="utf-8")
    for name in ("TERRA_REVIEW_PROMPT_VERSION", "SOL_REVIEW_PROMPT_VERSION",
                 "TERRA_REVIEW_REASONING_EFFORT", "SOL_REVIEW_REASONING_EFFORT",
                 "REQUIRED_CATALOG_VERSION", "REQUIRED_CATALOG_ONTOLOGY"):
        mo = re.search(rf'^{name}\s*=\s*["\']([^"\']+)["\']', contracts, re.M)
        if mo:
            out[name] = mo.group(1)
    for label, relpath in (("terra_review_module", "tools/renovation_architecture/terra_review.py"),
                           ("sol_review_module", "tools/renovation_architecture/sol_review.py")):
        p = root / relpath
        if p.is_file():
            out[f"{label}_blob"] = git_blob(root, p)
    return out


def arm_projection(root: Path) -> Dict[str, Any]:
    """The arm's catalog loaded exactly as production loads it, plus its projection."""
    from tools.artifact_writers import load_issue_catalog
    from tools.renovation_architecture.catalog_projection import build_renovation_catalog_projection
    catalog_path = root / "tools" / "issue_catalog_kind_v2.json"
    catalog = load_issue_catalog(catalog_path)
    if not catalog.get("items"):
        fail(f"catalog at {catalog_path} loaded empty (load_issue_catalog swallows a missing file)")
    projection = build_renovation_catalog_projection(catalog, catalog_path=catalog_path)
    return {"catalog": catalog, "projection": projection, "catalog_path": catalog_path}


def probe_sidecar() -> Dict[str, Any]:
    """A real POST: /health has returned 200 while every embeddings call failed."""
    base = str(cfg.EMBEDDINGS_BASE_URL).rstrip("/")
    rec: Dict[str, Any] = {"base_url": base, "model": cfg.EMBEDDINGS_MODEL_NAME,
                           "expected_dimension": int(cfg.EMBEDDINGS_DIMENSION)}
    payload = json.dumps({"input": ["preflight: worn roof shingles"],
                          "model": cfg.EMBEDDINGS_MODEL_NAME}).encode("utf-8")
    req = urllib.request.Request(f"{base}/embeddings", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return rec
    rec["latency_sec"] = round(time.time() - started, 3)
    vec = ((body.get("data") or [{}])[0]).get("embedding") or []
    rec["dimension"] = len(vec)
    rec["served_model"] = body.get("model")
    rec["ok"] = (len(vec) == int(cfg.EMBEDDINGS_DIMENSION)
                 and body.get("model") == cfg.EMBEDDINGS_MODEL_NAME)
    if vec:
        rec["probe_vector_sha256"] = sha256_canonical([round(float(x), 6) for x in vec])
    return rec


def probe_lmstudio() -> Dict[str, Any]:
    """Pass 2d runs on the local model; a dead server must stop the run before any paid call."""
    base = str(cfg.LM_STUDIO_URL).rstrip("/")
    url = base if base.endswith("/v1") else f"{base}/v1"
    rec: Dict[str, Any] = {"base_url": base, "expected_model": cfg.LM_STUDIO_MODEL}
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return rec
    served = [d.get("id") for d in (body.get("data") or []) if isinstance(d, dict)]
    rec["served_models"] = served
    rec["ok"] = cfg.LM_STUDIO_MODEL in served
    if not rec["ok"]:
        rec["error"] = f"{cfg.LM_STUDIO_MODEL!r} not served; got {served}"
    return rec


# --------------------------------------------------------------------------- verify

def _check(checks: List[Dict[str, Any]], cid: str, name: str, ok: bool, detail: Any = None) -> bool:
    checks.append({"id": cid, "name": name, "result": "pass" if ok else "FAIL", "detail": detail})
    return ok


def verify(m: Dict[str, Any], *, candidate_root: Path, require_live: bool) -> Dict[str, Any]:
    """Every invariant the manifest lists, plus the budget gate. Contacts no provider."""
    checks: List[Dict[str, Any]] = []

    for label, root, expected in (("baseline", MAIN_ROOT, BASELINE_COMMIT),
                                  ("candidate", candidate_root, CANDIDATE_COMMIT)):
        head = git(root, "rev-parse", "HEAD")
        _check(checks, f"I1.{label}.head", f"{label} HEAD == {expected[:7]}", head == expected,
               {"actual": head, "expected": expected})
        dirty = git(root, "status", "--porcelain", "-uno")
        _check(checks, f"I2.{label}.clean", f"{label} arm tracked-clean", dirty == "",
               {"porcelain_tracked": dirty})

    for label, root in (("baseline", MAIN_ROOT), ("candidate", candidate_root)):
        want = m["arms"][label]
        ctx = arm_projection(root)
        proj = ctx["projection"]
        _check(checks, f"I3.{label}.catalog_sha", f"{label} catalog sha256",
               proj["catalog_sha256"] == want["catalog_sha256"],
               {"actual": proj["catalog_sha256"], "expected": want["catalog_sha256"]})
        _check(checks, f"I4.{label}.fingerprint", f"{label} projection fingerprint",
               proj["fingerprint"] == want["projection_fingerprint"],
               {"actual": proj["fingerprint"], "expected": want["projection_fingerprint"]})
        _check(checks, f"I5.{label}.items", f"{label} item count == {want['item_count']}",
               len(ctx["catalog"]["items"]) == want["item_count"],
               {"actual": len(ctx["catalog"]["items"])})
        _check(checks, f"I6.{label}.version", f"{label} catalog version {want['catalog_version']}",
               ctx["catalog"].get("version") == want["catalog_version"],
               {"actual": ctx["catalog"].get("version")})

    for label, root in (("baseline", MAIN_ROOT), ("candidate", candidate_root)):
        got = prompt_identity(root)
        diff = {k: {"actual": got.get(k), "expected": v}
                for k, v in m["prompt_identity"].items() if got.get(k) != v}
        _check(checks, f"I7.{label}.prompts", f"{label} prompt identity matches the manifest",
               not diff, diff or got)

    for label, root in (("baseline", MAIN_ROOT), ("candidate", candidate_root)):
        bad = {}
        for relpath, want_blob in m["code_hashes"].items():
            got_blob = git(root, "rev-parse", f"HEAD:{relpath}")
            if got_blob != want_blob:
                bad[relpath] = {"actual": got_blob, "expected": want_blob}
        _check(checks, f"I8.{label}.code", f"{label} code blobs match the manifest", not bad, bad)

    mm_path = MAIN_ROOT / m["model_map"]["path"]
    _check(checks, "I9.model_map", "model map sha256",
           sha256_file(mm_path) == m["model_map"]["sha256"], {"actual": sha256_file(mm_path)})
    env_want = m["model_map"]["environment"]
    env_bad = {k: {"actual": os.environ.get(k), "expected": v} for k, v in env_want.items()
               if k != "LM_STUDIO_URL" and (os.environ.get(k) or None) != (v or None)}
    _check(checks, "I10.model_env", "model names in the environment match the manifest",
           not env_bad, env_bad)
    pinned_url = env_want.get("LM_STUDIO_URL")
    used_url = os.environ.get("LM_STUDIO_URL")
    substituted = used_url != pinned_url
    _check(checks, "I10b.lmstudio_url",
           "Pass 2d transport address: pinned, or an explicitly substituted route to the same model",
           (not substituted) or bool(LM_STUDIO_URL_OVERRIDE),
           {"pinned": pinned_url, "effective": used_url, "substituted": substituted,
            "explicit_override": LM_STUDIO_URL_OVERRIDE,
            "note": "the routing identity is lmstudio/<LM_STUDIO_MODEL>, checked by I26b; the "
                    "address is transport and is applied identically to both arms"})

    _check(checks, "I11.ontology", "KIND_ONTOLOGY_VERSION == observation_kind_v2",
           cfg.KIND_ONTOLOGY_VERSION == "observation_kind_v2", {"actual": cfg.KIND_ONTOLOGY_VERSION})
    _check(checks, "I12.catalog_path_env", "ISSUE_CATALOG_PATH unset",
           "ISSUE_CATALOG_PATH" not in os.environ, {"actual": os.environ.get("ISSUE_CATALOG_PATH")})
    _check(checks, "I13.budget_guard",
           "RENOVATION_VLM_BUDGET_GUARD off (this harness meters Terra itself)",
           not getattr(cfg, "RENOVATION_VLM_BUDGET_GUARD", False))
    _check(checks, "I14.usage_root",
           "RENOVATION_TERRA_USAGE_ROOT unset (the run pins its own ledger)",
           not (os.environ.get("RENOVATION_TERRA_USAGE_ROOT") or "").strip())

    emb = m["embeddings"]
    _check(checks, "I15.shortcut", "shortcut thresholds match the manifest",
           float(cfg.PASS_2D_SHORTCUT_MIN_SCORE) == emb["shortcut_min_score"]
           and float(cfg.PASS_2D_SHORTCUT_MIN_MARGIN) == emb["shortcut_min_margin"],
           {"score": cfg.PASS_2D_SHORTCUT_MIN_SCORE, "margin": cfg.PASS_2D_SHORTCUT_MIN_MARGIN})
    _check(checks, "I16.embed_cfg", "embeddings backend/url/model/dimension match the manifest",
           str(cfg.EMBEDDINGS_BACKEND) == emb["backend"]
           and str(cfg.EMBEDDINGS_BASE_URL) == emb["base_url"]
           and str(cfg.EMBEDDINGS_MODEL_NAME) == emb["model"]
           and int(cfg.EMBEDDINGS_DIMENSION) == emb["dimension"],
           {"backend": cfg.EMBEDDINGS_BACKEND, "base_url": cfg.EMBEDDINGS_BASE_URL,
            "model": cfg.EMBEDDINGS_MODEL_NAME, "dimension": cfg.EMBEDDINGS_DIMENSION})
    _check(checks, "I16b.top_k", f"Pass 2d top-K is the production value {TOP_K}", TOP_K == 8)

    groups = cases_by_run(m)
    art_bad: Dict[str, Any] = {}
    lane_bad: Dict[str, Any] = {}
    scene_bad: Dict[str, Any] = {}
    photo_bad: Dict[str, Any] = {}
    unpinned: Dict[str, Any] = {}
    lane_counts: Dict[str, int] = {}
    pinned_ids = {c["issue_id"] for c in m["cases"]}
    for g in groups:
        path = resolve_input(g["artifact_path"])
        if not path.is_file():
            art_bad[g["artifact_path"]] = "missing"
            continue
        got_sha = sha256_file(path)
        if got_sha != g["artifact_sha256"]:
            art_bad[g["artifact_path"]] = {"actual": got_sha, "expected": g["artifact_sha256"]}
            continue
        art = read_json(path)
        lane = product_lane(art)
        lane_counts[g["property_key"]] = len(lane)
        by_issue = {r.get("issue_id"): r for r in lane}
        photos = art.get("photos") or {}
        for c in g["cases"]:
            row = by_issue.get(c["issue_id"])
            if row is None:
                lane_bad[c["case_id"]] = "issue_id absent from the product lane"
            elif row.get("catalog_item_id") != c["frozen_resolution"]["resolved_item_id"]:
                lane_bad[c["case_id"]] = {"lane_owner": row.get("catalog_item_id"),
                                          "frozen_owner": c["frozen_resolution"]["resolved_item_id"]}
            stored_group = ((photos.get(c["photo_key"]) or {}).get("scene") or {}).get("group")
            if stored_group != c["frozen_pass_2c"]["scene_group"]:
                scene_bad[c["case_id"]] = {"artifact": stored_group,
                                           "manifest": c["frozen_pass_2c"]["scene_group"]}
        for row in lane:
            if row.get("catalog_item_id") == PARENT_ID and row.get("issue_id") not in pinned_ids:
                unpinned[row["issue_id"]] = {"property_key": g["property_key"],
                                             "photo_key": row.get("photo_key"),
                                             "observation": row.get("description")}
    _check(checks, "I17.artifacts", f"{len(groups)} stored artifacts at their pinned sha256",
           not art_bad, art_bad)
    _check(checks, "I18.lane", "every pinned case is in the product lane with its frozen owner",
           not lane_bad, lane_bad)
    _check(checks, "I19.scene", "every pinned case's scene group matches the artifact",
           not scene_bad, scene_bad)

    for c in m["cases"]:
        p = Path(c["photo"]["path"])
        if not p.is_file():
            photo_bad[c["case_id"]] = "missing"
        elif p.stat().st_size != c["photo"]["bytes"] or sha256_file(p) != c["photo"]["sha256"]:
            photo_bad[c["case_id"]] = "hash or size drift"
    _check(checks, "I20.photos", f"all {len(m['cases'])} pinned photos unchanged", not photo_bad, photo_bad)
    _check(checks, "I21.exclusions", "the unpinned parent-owned rows are exactly the four expected",
           set(unpinned) == set(UNPINNED_PARENT_ISSUE_IDS), unpinned)

    b = m["budget"]
    _check(checks, "I22.budget", "stage A authorization filled in by a human",
           isinstance(b.get("stage_a_authorized_tokens"), int) and b["stage_a_authorized_tokens"] > 0
           and bool(b.get("authorized_by")) and bool(b.get("authorized_at")),
           {"stage_a_authorized_tokens": b.get("stage_a_authorized_tokens"),
            "authorized_by": b.get("authorized_by"), "authorized_at": b.get("authorized_at")})
    _check(checks, "I23.stage_b", "stage B still unauthorized (Part A must not run a canary)",
           b.get("stage_b_authorized_tokens") is None, {"actual": b.get("stage_b_authorized_tokens")})
    proof = manifest_content_proof(m)
    _check(checks, "I24.manifest_proof",
           "only the budget block changed since Session 5 (restoring it reproduces 8530cca8...)",
           proof["ok"], proof)

    sidecar = probe_sidecar()
    _check(checks, "I25.sidecar", "embeddings sidecar answers a real POST at the pinned dimension",
           bool(sidecar.get("ok")), sidecar)
    lms = probe_lmstudio()
    ok_lms = bool(lms.get("ok"))
    checks.append({"id": "I26.lmstudio", "name": "LM Studio serves the pinned Pass 2d model",
                   "result": "pass" if ok_lms else ("FAIL" if require_live else "deferred"),
                   "detail": lms})
    _check(checks, "I26b.pass2d_route",
           "Pass 2d model identity is the manifest's LM_STUDIO_MODEL",
           str(cfg.LM_STUDIO_MODEL) == env_want.get("LM_STUDIO_MODEL"),
           {"actual": cfg.LM_STUDIO_MODEL, "expected": env_want.get("LM_STUDIO_MODEL")})

    failures = [c for c in checks if c["result"] == "FAIL"]
    return {"generated_at_utc": utc_now(), "checks": checks,
            "result": "pass" if not failures else "FAIL", "failures": failures,
            "sidecar": sidecar, "lmstudio": lms, "lane_row_counts": lane_counts,
            "unpinned_parent_rows": unpinned,
            "lm_studio": {"pinned_url": env_want.get("LM_STUDIO_URL"),
                          "effective_url": os.environ.get("LM_STUDIO_URL"),
                          "explicit_override": LM_STUDIO_URL_OVERRIDE,
                          "model": cfg.LM_STUDIO_MODEL, "served_models": lms.get("served_models")},
            "environment": {"python": sys.version.split()[0], "executable": sys.executable,
                            "arm_root": str(ARM_ROOT), "script_sha256": sha256_file(SCRIPT_PATH)}}


# --------------------------------------------------------------------------- pass 2d (live, local)

PASS_2D_ATTEMPTS = 3
PASS_2D_RETRY_PAUSE_SEC = 20


async def _resolve_rows(rows: List[Dict[str, Any]], *, provider, vlm_client, model_config,
                        photos: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Live Pass 2d for the pinned rows of one property, through the production resolution path.

    A transport failure is retried, never recorded as an outcome: "the model could not be reached"
    and "the observation resolved to nothing" are different facts, and only the second is evidence.
    A semantic failure (an out-of-ontology kind, an off-kind candidate) is fail-closed by design and
    is not retried."""
    from tools.scene_classifier_orchestrator import resolve_observation_against_catalog
    from tools.scene_classifier_passes import PassExecutionError
    out: List[Dict[str, Any]] = []
    for row in rows:
        case = row["case"]
        frozen = case["frozen_pass_2c"]
        photo = photos.get(case["photo_key"]) or {}
        image_path = ((photo.get("photo") or {}).get("image_path")) or case["photo"]["path"]
        scene = ((photo.get("scene") or {}).get("id")) or frozen["scene_group"]
        observation = {
            "description": frozen["observation"], "kind": frozen["kind"],
            "issue_id": case["issue_id"], "scene_group": frozen["scene_group"],
            "source_photo_key": case["photo_key"],
        }
        rec: Dict[str, Any] = {"case_id": case["case_id"], "issue_id": case["issue_id"],
                               "observation": frozen["observation"], "kind": frozen["kind"],
                               "scene": scene, "scene_group": frozen["scene_group"]}
        before = dict(getattr(vlm_client, "usage_stats", {}) or {})
        started = time.time()
        attempts: List[str] = []
        for attempt in range(1, PASS_2D_ATTEMPTS + 1):
            try:
                resolved_row, debug_row, _ = await resolve_observation_against_catalog(
                    vlm_client=vlm_client, model_config=model_config, candidate_provider=provider,
                    observation=observation,
                    base_context={"scene": scene, "scene_group": frozen["scene_group"]},
                    top_k=TOP_K, source_image_path=str(image_path),
                )
                cands = (resolved_row or {}).get("candidates") or []
                scores = [c.get("score") for c in cands if isinstance(c.get("score"), (int, float))]
                rec.update({
                    "status": "resolved" if (resolved_row or {}).get("resolved_item_id") else "unresolved",
                    "resolved_item_id": (resolved_row or {}).get("resolved_item_id"),
                    "resolution_path": debug_row.get("resolution_path"),
                    "shortcut_reason": debug_row.get("shortcut_reason"),
                    "routing_reason": debug_row.get("routing_reason"),
                    "skipped_reason": debug_row.get("skipped_reason"),
                    "candidate_count": debug_row.get("candidate_count"),
                    "candidate_ids": [c.get("item_id") for c in cands],
                    "candidate_scores": [round(float(sc), 12) for sc in scores],
                    "top_two_margin": round(float(scores[0] - scores[1]), 12) if len(scores) > 1 else None,
                })
                break
            except PassExecutionError as exc:
                stage = getattr(exc, "stage", None)
                attempts.append(f"attempt {attempt}: {stage}/{getattr(exc, 'code', None)}: {exc}"[:300])
                rec.update({"status": "pass_error", "resolved_item_id": None,
                            "pass_error_code": getattr(exc, "code", None),
                            "pass_error_stage": stage, "error": str(exc)[:600]})
                if stage != "request" or attempt == PASS_2D_ATTEMPTS:
                    break
                print(f"    Pass 2d transport failure, retrying ({attempt}/{PASS_2D_ATTEMPTS})",
                      flush=True)
                await asyncio.sleep(PASS_2D_RETRY_PAUSE_SEC)
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"attempt {attempt}: {type(exc).__name__}: {exc}"[:300])
                rec.update({"status": "error", "resolved_item_id": None,
                            "error": f"{type(exc).__name__}: {exc}"[:600]})
                if attempt == PASS_2D_ATTEMPTS:
                    break
                await asyncio.sleep(PASS_2D_RETRY_PAUSE_SEC)
        if attempts:
            rec["attempts"] = attempts
        after = dict(getattr(vlm_client, "usage_stats", {}) or {})
        rec["duration_sec"] = round(time.time() - started, 2)
        rec["local_usage"] = {k: (after.get(k, 0) or 0) - (before.get(k, 0) or 0)
                              for k in ("input_tokens", "output_tokens", "total_tokens", "calls")}
        out.append(rec)
    return out


# --------------------------------------------------------------------------- the run

_TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
                 "budget_debited_tokens")


def _unit_path(out_dir: Path, estimate_unit_id: str) -> Path:
    digest = hashlib.sha256(estimate_unit_id.encode("utf-8")).hexdigest()[:16]
    return out_dir / "units" / f"terra_unit_{digest}.json"


def _warm_pass_2d(vlm_client, model_config: Dict[str, Any]) -> None:
    """One throwaway call before the first property. The local model shares a GPU with whatever else
    is on the box; learning that on case 1 of 66 is cheaper than learning it on case 40, and this
    runs before any paid Terra call."""
    started = time.time()
    try:
        asyncio.run(vlm_client.analyze_text(system_prompt="Reply with the single word: ok.",
                                            user_prompt="ok", **model_config))
    except Exception as exc:  # noqa: BLE001
        fail(f"Pass 2d warm-up failed after {time.time() - started:.0f}s: {type(exc).__name__}: {exc}. "
             f"The local model at {cfg.LM_STUDIO_URL} is unreachable, cold or contended for GPU. "
             f"No provider call was made.")
    print(f"  Pass 2d warm-up ok in {time.time() - started:.1f}s "
          f"({model_config.get('provider')}/{model_config.get('model')})", flush=True)


def _guard_out_root(out_root: Path) -> None:
    """The run root must not overlap the frozen canary tree or the production artifacts root: both
    are read-only evidence for this program."""
    out = out_root.resolve()
    protected = [MAIN_ROOT / "artifacts_canary" / "renovation_session9_20260818",
                 MAIN_ROOT / "artifacts_canary" / "renovation_session6_20260816_02",
                 PROD_ARTIFACTS_ROOT]
    for p in protected:
        p = Path(p).resolve()
        if out == p or p in out.parents or out in p.parents:
            fail(f"run root {out} overlaps the protected root {p}")


def run_arm(m: Dict[str, Any], *, arm: str, out_root: Path, limit_properties: Optional[int],
            dry_run: bool) -> Dict[str, Any]:
    """Live Stage A for one arm: local Pass 2d, then Terra per target unit, then the deterministic
    disposition and standalone-pricing tail. Metered, resumable, and stoppable at the budget."""
    assert_not_imported(SOL_MODULE)
    _guard_out_root(out_root)
    want_arm = m["arms"][arm]
    if not dry_run and str(ARM_ROOT) != str(Path(want_arm["checkout"]).resolve()):
        fail(f"--arm-root {ARM_ROOT} is not the {arm} checkout {want_arm['checkout']}")

    ctx = arm_projection(ARM_ROOT)
    projection, catalog = ctx["projection"], ctx["catalog"]
    if not dry_run:
        if projection["catalog_sha256"] != want_arm["catalog_sha256"]:
            fail(f"{arm} catalog sha drift: {projection['catalog_sha256']}")
        if projection["fingerprint"] != want_arm["projection_fingerprint"]:
            fail(f"{arm} projection fingerprint drift: {projection['fingerprint']}")
    observables = projection["observables"]
    terminal_routes = projection["terminal_routes"]

    authorized = int(m["budget"]["stage_a_authorized_tokens"])
    terra_model = cfg.RENOVATION_TERRA_MODEL
    max_out = int(cfg.RENOVATION_TERRA_MAX_OUTPUT_TOKENS)

    from tools.catalog_embeddings import build_candidate_provider
    from tools.pass_2f_artifact_inputs import photo_key_to_path
    from tools.pass_config import SceneClassifierRunOptions, get_model_config_for_pass
    from tools.renovation_architecture.conditions import build_observed_conditions
    from tools.renovation_architecture.contracts import (CONDITION_DISPOSITION_POLICY_VERSION,
                                                         CONTRACTS_SCHEMA_VERSION,
                                                         TERRA_REVIEW_REASONING_EFFORT)
    from tools.renovation_architecture.disposition import decide_disposition
    from tools.renovation_architecture.evidence import build_evidence_facts, build_photo_identity_index
    from tools.renovation_architecture.ids import make_disposition_id, make_estimate_id
    from tools.renovation_architecture.terra_review import build_unit_request
    from tools.renovation_architecture.usage_guard import (TerraDailyBudgetExceeded, TerraUsageLedger,
                                                           estimate_reservation_tokens)
    from tools.renovation_architecture.validators import validate_condition_review_result
    from tools.renovation_architecture.work_items import derive_standalone_estimate
    from tools.scene_classifier_passes import PassExecutionError

    source_run_id = f"catalog_audit_s6_{arm}"
    ledger = TerraUsageLedger(out_root, daily_ceiling=authorized + 50_000, usage_root_override=None)

    vlm_client = None
    model_config_2d = None
    review_unit_fresh = None
    if not dry_run:
        from tools.renovation_architecture.review_pipeline import _review_unit_fresh as review_unit_fresh
        from tools.vlm_client import create_vlm_client, get_model_configs_from_pipeline_config
        if not cfg.OPENAI_API_KEY:
            fail("OPENAI_API_KEY is not resolved")
        if terra_model != m["model_map"]["environment"]["RENOVATION_TERRA_MODEL"]:
            fail(f"resolved Terra model {terra_model!r} differs from the manifest")
        qwen_config, gpt5_config = get_model_configs_from_pipeline_config(cfg)
        model_config_2d = get_model_config_for_pass("2d", SceneClassifierRunOptions(premium=True),
                                                    qwen_config, gpt5_config)
        if (model_config_2d.get("provider") != "lmstudio"
                or model_config_2d.get("model") != cfg.LM_STUDIO_MODEL):
            fail(f"Pass 2d route drifted off the local model: "
                 f"{model_config_2d.get('provider')}/{model_config_2d.get('model')}")
        vlm_client = create_vlm_client()
        vlm_client.reset_usage_stats()
        _warm_pass_2d(vlm_client, model_config_2d)

    provider = None
    groups = cases_by_run(m)
    if limit_properties:
        groups = groups[:limit_properties]

    run_record: Dict[str, Any] = {
        "schema_version": "catalog-audit-live-validation-arm-v1",
        "arm": arm, "arm_root": str(ARM_ROOT), "dry_run": dry_run, "started_utc": utc_now(),
        "catalog_sha256": projection["catalog_sha256"],
        "projection_fingerprint": projection["fingerprint"],
        "projection_version": projection.get("version"),
        "source_run_id": source_run_id, "terra_model": terra_model,
        "terra_max_output_tokens": max_out, "terra_reasoning_effort": TERRA_REVIEW_REASONING_EFFORT,
        "pass_2d_route": model_config_2d and {k: model_config_2d.get(k) for k in ("provider", "model")},
        "authorized_tokens": authorized, "ledger_path": str(ledger.path),
        "properties": [], "cases": [], "units": [], "excluded_rows": [], "stopped": None,
    }
    spent_before = 0 if dry_run else ledger.spent_today()
    stop = None

    for g in groups:
        if stop:
            break
        art_path = resolve_input(g["artifact_path"])
        got_sha = sha256_file(art_path)
        if got_sha != g["artifact_sha256"]:
            fail(f"artifact drift at {art_path}: {got_sha}")
        art = read_json(art_path)
        lane = product_lane(art)
        photos = art.get("photos") or {}
        property_metadata = art.get("property_metadata")
        paths = photo_key_to_path(art)
        prop_dir = out_root / arm / g["property_key"] / g["run_id"]
        prop_dir.mkdir(parents=True, exist_ok=True)

        # ---- Pass 2d, live, for this run's pinned rows only. Every other row stays frozen.
        by_issue = {r.get("issue_id"): r for r in lane}
        rows = [{"case": c, "lane_row": by_issue.get(c["issue_id"])} for c in g["cases"]]
        cache_path = prop_dir / "pass_2d.json"
        cached = read_json(cache_path) if cache_path.is_file() else None
        cache_key = sha256_canonical({
            "catalog": projection["catalog_sha256"], "top_k": TOP_K,
            "rows": [{"i": r["case"]["issue_id"], "o": r["case"]["frozen_pass_2c"]["observation"],
                      "k": r["case"]["frozen_pass_2c"]["kind"],
                      "s": r["case"]["frozen_pass_2c"]["scene_group"]} for r in rows]})
        if cached and cached.get("cache_key") == cache_key:
            resolutions = cached["resolutions"]
            pass2d_source = "resumed"
        elif dry_run:
            resolutions = [{"case_id": r["case"]["case_id"], "issue_id": r["case"]["issue_id"],
                            "status": "dry_run",
                            "resolved_item_id": r["case"]["frozen_resolution"]["resolved_item_id"]}
                           for r in rows]
            pass2d_source = "dry_run"
        else:
            if provider is None:
                provider = build_candidate_provider(catalog)
            resolutions = asyncio.run(_resolve_rows(rows, provider=provider, vlm_client=vlm_client,
                                                    model_config=model_config_2d, photos=photos))
            pass2d_source = "live"
            broken = [r for r in resolutions if r["status"] in ("pass_error", "error")]
            if broken:
                # Never cache a transport failure and never carry one into the lane. A case with no
                # resolution reads downstream as "reached no catalog item", which is one of the very
                # outcomes this experiment measures -- so an unreachable model would manufacture a
                # finding. Stop instead, loudly, with nothing cached.
                write_json(prop_dir / "pass_2d_failed.json",
                           {"arm": arm, "property_key": g["property_key"], "resolutions": resolutions})
                fail(f"{g['property_key']}: {len(broken)} of {len(resolutions)} Pass 2d calls failed "
                     f"after {PASS_2D_ATTEMPTS} attempts: "
                     f"{(broken[0].get('error') or '')[:160]}. Nothing cached; fix the Pass 2d "
                     f"endpoint and re-run this arm -- finished units resume from disk.")
            write_json(cache_path, {"cache_key": cache_key, "arm": arm,
                                    "catalog_sha256": projection["catalog_sha256"],
                                    "resolutions": resolutions})
        res_by_issue = {r["issue_id"]: r for r in resolutions}

        # ---- rebuild the lane: pinned rows take their live resolution, the four unpinned go
        replay_lane: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []
        for row in lane:
            iid = row.get("issue_id")
            if iid in UNPINNED_PARENT_ISSUE_IDS:
                excluded.append({"issue_id": iid, "property_key": g["property_key"],
                                 "photo_key": row.get("photo_key"),
                                 "observation": row.get("description"),
                                 "frozen_owner": row.get("catalog_item_id"),
                                 "reason": "parent-owned row outside the pinned case set; a stale id "
                                           "under the candidate catalog, dropped in BOTH arms"})
                continue
            new = dict(row)
            live = res_by_issue.get(iid)
            if live is not None:
                rid = live.get("resolved_item_id")
                new["catalog_item_id"] = rid
                new["catalog_item_kind"] = (observables.get(rid) or {}).get("kind") if rid else None
            replay_lane.append(new)
        run_record["excluded_rows"].extend(excluded)

        # ---- condition projection under this arm's catalog
        estimate_id = make_estimate_id(property_key=g["property_key"], source_run_id=source_run_id,
                                       catalog_sha256=projection["catalog_sha256"],
                                       projection_fingerprint=projection["fingerprint"])
        drafts = build_observed_conditions(issues_flat=replay_lane, photos=photos,
                                           property_metadata=property_metadata,
                                           projection=projection, estimate_id=estimate_id)
        pinned_ids = {c["issue_id"] for c in g["cases"]}
        target_units = sorted({d.condition.estimate_unit_id for d in drafts
                               if pinned_ids & set(d.condition.issue_ids)})
        unit_drafts = {u: [d for d in drafts if d.condition.estimate_unit_id == u] for u in target_units}

        # ---- evidence for every condition in the target units (Terra judges whole units)
        needed = sorted({ref["photo_key"] for u in target_units for d in unit_drafts[u]
                         for ref in d.evidence_refs})
        identity_index = build_photo_identity_index(needed, paths)
        pairs_by_unit = {
            u: [(d, build_evidence_facts(d, identity_index=identity_index, observables=observables,
                                         estimate_id=estimate_id)) for d in unit_drafts[u]]
            for u in target_units
        }

        terra_calls: List[Dict[str, Any]] = []
        reviews: List[Dict[str, Any]] = []
        prop_stop = None
        for unit_id in target_units:
            pairs = pairs_by_unit[unit_id]
            request = build_unit_request(
                estimate_unit_id=unit_id, unit_pairs=pairs, observables=observables,
                photo_key_to_path=paths, identity_index=identity_index,
                projection_fingerprint=projection["fingerprint"], model=terra_model,
                reasoning_effort=TERRA_REVIEW_REASONING_EFFORT, max_output_tokens=max_out,
            )
            reservation = estimate_reservation_tokens(max_output_tokens=max_out,
                                                      request_bytes=request.request_bytes,
                                                      image_count=len(request.image_paths))
            upath = _unit_path(prop_dir, unit_id)
            prior = read_json(upath) if upath.is_file() else None
            if (prior and prior.get("request_fingerprint") == request.request_fingerprint
                    and prior.get("status") == "reviewed"):
                terra_calls.append(prior["terra_call"])
                reviews.extend(prior["reviews"])
                run_record["units"].append({
                    "arm": arm, "property_key": g["property_key"], "estimate_unit_id": unit_id,
                    "condition_ids": prior["condition_ids"],
                    "target_condition_ids": prior.get("target_condition_ids"),
                    "request_fingerprint": prior["request_fingerprint"],
                    "photo_keys": prior["photo_keys"], "request_bytes": prior["request_bytes"],
                    "estimated_reservation_tokens": prior.get("estimated_reservation_tokens"),
                    "status": "resumed",
                    "total_tokens": prior["terra_call"]["total_tokens"],
                    "budget_debited_tokens": prior["terra_call"]["budget_debited_tokens"]})
                continue
            unit_rec: Dict[str, Any] = {
                "schema_version": "catalog-audit-live-validation-unit-v1",
                "arm": arm, "property_key": g["property_key"], "run_id": g["run_id"],
                "estimate_unit_id": unit_id, "estimate_id": estimate_id,
                "condition_ids": sorted(request.condition_ids),
                "target_condition_ids": sorted(d.condition.condition_id for d, _ in pairs
                                               if pinned_ids & set(d.condition.issue_ids)),
                "request_fingerprint": request.request_fingerprint,
                "photo_keys": list(request.photo_keys),
                "image_sha256": {k: identity_index[k].exact_sha256 for k in request.photo_keys},
                "request_bytes": request.request_bytes,
                "estimated_reservation_tokens": reservation,
                "system_prompt_sha256": sha256_bytes(request.system_prompt.encode("utf-8")),
                "user_prompt": request.user_prompt, "model": terra_model,
                "max_output_tokens": max_out, "reasoning_effort": TERRA_REVIEW_REASONING_EFFORT,
            }
            if dry_run:
                unit_rec["status"] = "planned"
                run_record["units"].append(unit_rec)
                continue
            spent = ledger.spent_today()
            if spent + reservation > authorized:
                prop_stop = {"reason": "stopped_at_authorization", "spent_today": spent,
                             "reservation": reservation, "authorized": authorized,
                             "at_unit": unit_id, "property_key": g["property_key"]}
                unit_rec["status"] = "skipped_budget"
                write_json(upath, unit_rec)
                run_record["units"].append(unit_rec)
                break
            try:
                call, unit_reviews = review_unit_fresh(
                    request=request, vlm_client=vlm_client, api_key=cfg.OPENAI_API_KEY,
                    terra_model=terra_model, terra_max_output_tokens=max_out, ledger=ledger,
                    property_key=g["property_key"], source_run_id=source_run_id,
                    estimate_id=estimate_id)
            except TerraDailyBudgetExceeded as exc:
                prop_stop = {"reason": "ledger_denied", "detail": str(exc)[:400], "at_unit": unit_id}
                unit_rec["status"] = "denied"
                write_json(upath, unit_rec)
                run_record["units"].append(unit_rec)
                break
            except PassExecutionError as exc:
                unit_rec.update({"status": "pass_error", "error": str(exc)[:600],
                                 "error_code": getattr(exc, "code", None)})
                write_json(upath, unit_rec)
                run_record["units"].append(unit_rec)
                prop_stop = {"reason": "terra_pass_error", "detail": str(exc)[:400], "at_unit": unit_id}
                break
            unit_rec.update({"status": "reviewed", "terra_call": call, "reviews": unit_reviews})
            write_json(upath, unit_rec)
            run_record["units"].append({
                "arm": arm, "property_key": g["property_key"], "estimate_unit_id": unit_id,
                "condition_ids": unit_rec["condition_ids"],
                "target_condition_ids": unit_rec["target_condition_ids"],
                "request_fingerprint": unit_rec["request_fingerprint"],
                "photo_keys": unit_rec["photo_keys"], "request_bytes": unit_rec["request_bytes"],
                "estimated_reservation_tokens": reservation, "status": "reviewed",
                "total_tokens": call["total_tokens"],
                "budget_debited_tokens": call["budget_debited_tokens"]})
            terra_calls.append(call)
            reviews.extend(unit_reviews)
            print(f"  [{arm}] {g['property_key']} {unit_id}: {len(unit_reviews)} conditions, "
                  f"{call['total_tokens']} tokens, ledger {ledger.spent_today():,}/{authorized:,}",
                  flush=True)

        # ---- dispositions and standalone pricing over the units actually reviewed
        reviewed_units = sorted({c["estimate_unit_id"] for c in terra_calls})
        reviews_by_condition = {r["condition_id"]: r for r in reviews}
        dispositions: List[Dict[str, Any]] = []
        for unit_id in reviewed_units:
            for draft, evidence in pairs_by_unit[unit_id]:
                cond = draft.condition
                review = reviews_by_condition[cond.condition_id]
                route = terminal_routes[cond.catalog_item_id]["route"]
                disposition, reason_code = decide_disposition(
                    review["verdict"], route, evidence.distinct_view_count,
                    evidence.min_photo_evidence_required)
                dispositions.append({
                    "disposition_id": make_disposition_id(estimate_id=estimate_id,
                                                          condition_id=cond.condition_id),
                    "schema_version": CONTRACTS_SCHEMA_VERSION,
                    "condition_id": cond.condition_id, "review_id": review["review_id"],
                    "disposition": disposition, "reason_code": reason_code,
                    "policy_version": CONDITION_DISPOSITION_POLICY_VERSION,
                    "evidence_id": evidence.evidence_id, "terminal_route": route})

        reviewed_pairs = [p for u in reviewed_units for p in pairs_by_unit[u]]
        review_result = None
        standalone = None
        billing: Dict[str, Dict[str, int]] = {}
        if reviewed_pairs:
            unit_usage = [{"schema_version": CONTRACTS_SCHEMA_VERSION,
                           "estimate_unit_id": c["estimate_unit_id"], "call_ids": [c["call_id"]],
                           **{n: c[n] for n in _TOKEN_FIELDS}} for c in terra_calls]
            listing_usage = {"schema_version": CONTRACTS_SCHEMA_VERSION, "call_count": len(terra_calls),
                             **{n: sum(u[n] for u in unit_usage) for n in _TOKEN_FIELDS}}
            review_result = {
                "observed_conditions": [d.condition.to_dict() for d, _ in reviewed_pairs],
                "evidence_facts": [e.to_dict() for _, e in reviewed_pairs],
                "condition_reviews": reviews, "condition_dispositions": dispositions,
                "terra_calls": terra_calls, "terra_unit_usage": unit_usage,
                "terra_listing_usage": listing_usage}
            v = validate_condition_review_result(review_result, estimate_id=estimate_id)
            if not v.ok:
                fail(f"{g['property_key']}: assembled review result invalid: {'; '.join(v.errors[:5])}")
            standalone = derive_standalone_estimate(review_result=review_result, projection=projection,
                                                    property_metadata=property_metadata,
                                                    estimate_id=estimate_id)
            for item in standalone["work_items"]:
                if item.get("status") != "active":
                    continue
                for cid in item.get("condition_ids") or []:
                    bucket = billing.setdefault(cid, {"low": 0, "high": 0})
                    bucket["low"] += int(item.get("low") or 0)
                    bucket["high"] += int(item.get("high") or 0)

        # ---- per-case rows
        cond_by_issue: Dict[str, Any] = {}
        for d in drafts:
            for iid in d.condition.issue_ids:
                cond_by_issue[iid] = d
        disp_by_condition = {d["condition_id"]: d for d in dispositions}
        for c in g["cases"]:
            live = res_by_issue.get(c["issue_id"], {})
            draft = cond_by_issue.get(c["issue_id"])
            cond = draft.condition if draft else None
            review = reviews_by_condition.get(cond.condition_id) if cond else None
            disp = disp_by_condition.get(cond.condition_id) if cond else None
            bill = billing.get(cond.condition_id) if cond else None
            run_record["cases"].append({
                "case_id": c["case_id"], "arm": arm, "role": c["role"],
                "census_bucket": c["census_bucket"], "review_card_id": c["review_card_id"],
                "expected_after": c["expected_after"],
                "expected_after_reason_class": c["expected_after_reason_class"],
                "manifest_condition_id": c["condition_id"], "property_key": c["property_key"],
                "run_id": c["run_id"], "photo_key": c["photo_key"], "issue_id": c["issue_id"],
                "observation": c["frozen_pass_2c"]["observation"], "kind": c["frozen_pass_2c"]["kind"],
                "scene_group": c["frozen_pass_2c"]["scene_group"],
                "frozen_resolved_item_id": c["frozen_resolution"]["resolved_item_id"],
                "frozen_resolution_path": c["frozen_resolution"]["resolution_path"],
                "offline_prediction": c["offline_prediction"],
                "pass_2d_status": live.get("status"),
                "resolved_item_id": live.get("resolved_item_id"),
                "resolution_path": live.get("resolution_path"),
                "shortcut_reason": live.get("shortcut_reason"),
                "top_two_margin": live.get("top_two_margin"),
                "candidate_ids": live.get("candidate_ids"),
                "pass_2d_error": live.get("error"),
                "condition_id": cond.condition_id if cond else None,
                "condition_catalog_item_id": cond.catalog_item_id if cond else None,
                "estimate_unit_id": cond.estimate_unit_id if cond else None,
                "condition_issue_ids": list(cond.issue_ids) if cond else None,
                "condition_is_shared": bool(cond and len(cond.issue_ids) > 1),
                "condition_reviewed": bool(review),
                "terra_verdict": review.get("verdict") if review else None,
                "terra_rationale": review.get("rationale") if review else None,
                "disposition": disp.get("disposition") if disp else None,
                "reason_code": disp.get("reason_code") if disp else None,
                "route": disp.get("terminal_route") if disp else (
                    terminal_routes.get(cond.catalog_item_id, {}).get("route") if cond else None),
                "billed_low": (bill or {}).get("low"), "billed_high": (bill or {}).get("high")})

        # ---- neighbouring conditions inside the reviewed units (error-migration context)
        neighbours = []
        for unit_id in reviewed_units:
            for draft, evidence in pairs_by_unit[unit_id]:
                cond = draft.condition
                if pinned_ids & set(cond.issue_ids):
                    continue
                review = reviews_by_condition.get(cond.condition_id)
                disp = disp_by_condition.get(cond.condition_id)
                neighbours.append({
                    "arm": arm, "property_key": g["property_key"], "estimate_unit_id": unit_id,
                    "condition_id": cond.condition_id, "catalog_item_id": cond.catalog_item_id,
                    "issue_ids": list(cond.issue_ids),
                    "terra_verdict": review.get("verdict") if review else None,
                    "disposition": disp.get("disposition") if disp else None,
                    "terminal_route": disp.get("terminal_route") if disp else None,
                    "billed_low": (billing.get(cond.condition_id) or {}).get("low"),
                    "billed_high": (billing.get(cond.condition_id) or {}).get("high")})

        prop_rec = {
            "property_key": g["property_key"], "run_id": g["run_id"], "source": g["source"],
            "artifact_path": g["artifact_path"], "artifact_sha256": g["artifact_sha256"],
            "estimate_id": estimate_id, "pass_2d_source": pass2d_source,
            "lane_rows": len(lane), "replay_lane_rows": len(replay_lane),
            "excluded_rows": [e["issue_id"] for e in excluded],
            "conditions_built": len(drafts), "target_units": target_units,
            "target_unit_conditions": {u: len(unit_drafts[u]) for u in target_units},
            "reviewed_units": reviewed_units, "conditions_reviewed": len(reviewed_pairs),
            "neighbour_conditions": neighbours,
            "standalone_headline": (standalone or {}).get("standalone_estimate", {}).get("headline"),
            "property_cost_factor": (standalone or {}).get("standalone_estimate", {}).get(
                "property_cost_factor"),
            "property_cost_factor_audit": (standalone or {}).get("standalone_estimate", {}).get(
                "property_cost_factor_audit"),
            "stop": prop_stop}
        write_json(prop_dir / "property_record.json",
                   {**prop_rec, "review_result": review_result,
                    "standalone_estimate": (standalone or {}).get("standalone_estimate"),
                    "work_items": (standalone or {}).get("work_items")})
        run_record["properties"].append(prop_rec)
        if prop_stop:
            stop = prop_stop

    spent_after = 0 if dry_run else ledger.spent_today()
    run_record["stopped"] = stop
    run_record["finished_utc"] = utc_now()
    run_record["spend"] = {
        "ledger_spent_today_before": spent_before, "ledger_spent_today_after": spent_after,
        "ledger_delta": spent_after - spent_before,
        "terra_calls": sum(1 for u in run_record["units"] if u.get("status") == "reviewed"),
        "terra_calls_resumed": sum(1 for u in run_record["units"] if u.get("status") == "resumed"),
        "terra_total_tokens": sum(u.get("total_tokens") or 0 for u in run_record["units"]),
        "authorized_tokens": authorized}
    if vlm_client is not None:
        run_record["vlm_usage_stats"] = dict(getattr(vlm_client, "usage_stats", {}) or {})
    out = out_root / arm / "arm_record.json"
    sha = write_json(out, run_record)
    print(f"[{arm}] {len(run_record['properties'])} properties, "
          f"{run_record['spend']['terra_calls']} Terra calls, "
          f"{run_record['spend']['terra_total_tokens']:,} tokens "
          f"(ledger {spent_after:,}/{authorized:,}); wrote {rel(out)} sha256={sha[:16]}")
    return run_record


# --------------------------------------------------------------------------- compare (mechanical)

def compare(m: Dict[str, Any], out_root: Path) -> Dict[str, Any]:
    """Join both arms and count. No rubric, no verdict: Part B judges this evidence."""
    arms = {}
    for arm in ("baseline", "candidate"):
        p = out_root / arm / "arm_record.json"
        if not p.is_file():
            fail(f"missing arm record: {p}")
        arms[arm] = read_json(p)
    cases_by_arm = {a: {c["case_id"]: c for c in arms[a]["cases"]} for a in arms}
    manifest_cases = {c["case_id"]: c for c in m["cases"]}
    successors = {BLINDS_ID, FABRIC_ID}

    rows: List[Dict[str, Any]] = []
    for cid, mc in manifest_cases.items():
        b = cases_by_arm["baseline"].get(cid)
        c = cases_by_arm["candidate"].get(cid)
        row: Dict[str, Any] = {
            "case_id": cid, "role": mc["role"], "census_bucket": mc["census_bucket"],
            "review_card_id": mc["review_card_id"], "expected_after": mc["expected_after"],
            "expected_after_reason_class": mc["expected_after_reason_class"],
            "property_key": mc["property_key"], "photo_key": mc["photo_key"],
            "observation": mc["frozen_pass_2c"]["observation"],
            "frozen_owner": mc["frozen_resolution"]["resolved_item_id"],
            "frozen_resolution_path": mc["frozen_resolution"]["resolution_path"],
            "offline_prediction": mc["offline_prediction"], "executed": bool(b and c)}
        keys = ("resolved_item_id", "resolution_path", "shortcut_reason", "top_two_margin",
                "condition_id", "condition_catalog_item_id", "estimate_unit_id",
                "condition_is_shared", "terra_verdict", "terra_rationale", "disposition",
                "reason_code", "route", "billed_low", "billed_high", "pass_2d_status")
        for label, rec in (("baseline", b), ("candidate", c)):
            row[label] = None if rec is None else {k: rec.get(k) for k in keys}
        if b and c:
            row["resolved_changed"] = b["resolved_item_id"] != c["resolved_item_id"]
            row["verdict_changed"] = b["terra_verdict"] != c["terra_verdict"]
            row["disposition_changed"] = b["disposition"] != c["disposition"]
            row["route_changed"] = b["route"] != c["route"]
            row["billing_changed"] = ((b.get("billed_low"), b.get("billed_high"))
                                      != (c.get("billed_low"), c.get("billed_high")))
            row["baseline_2d_matches_frozen"] = b["resolved_item_id"] == row["frozen_owner"]
            offline = mc["offline_prediction"]["candidate_shortcut"]
            row["candidate_matches_offline_shortcut"] = (
                c["resolved_item_id"] == offline if offline else None)
            cand_item = c.get("condition_catalog_item_id")
            row["reaches_successor"] = cand_item in successors
            row["hijack_candidate"] = bool(
                cand_item and cand_item not in successors and cand_item != row["frozen_owner"]
                and cand_item != offline)
        rows.append(row)

    executed = [r for r in rows if r["executed"]]
    n = len(executed)

    def count(pred) -> int:
        return sum(1 for r in executed if pred(r))

    controls = [r for r in executed if r["role"] == "control"]
    control_table = [{
        "case_id": r["case_id"], "review_card_id": r["review_card_id"],
        "expected_after": r["expected_after"],
        "expected_after_reason_class": r["expected_after_reason_class"],
        "observation": r["observation"],
        "baseline_verdict": r["baseline"]["terra_verdict"],
        "candidate_verdict": r["candidate"]["terra_verdict"],
        "baseline_disposition": r["baseline"]["disposition"],
        "candidate_disposition": r["candidate"]["disposition"],
        "baseline_item": r["baseline"]["condition_catalog_item_id"],
        "candidate_item": r["candidate"]["condition_catalog_item_id"],
        "verdict_changed": r["verdict_changed"]} for r in controls]

    findings = {
        "F-REACH-oc1_93a2e3f3d299a7a1": [r for r in executed
                                         if r["case_id"].startswith("production:redfin_80916010:")],
        "F-SHORTCUT-fee5271eaff08de5": [r for r in executed
                                        if r["case_id"].endswith(":fee5271eaff08de5")]}
    verdict_flips = [r for r in executed if r["verdict_changed"]]

    aggregates = {
        "cases_pinned": len(rows), "cases_executed": n,
        "cases_not_executed": [r["case_id"] for r in rows if not r["executed"]],
        "reachability": {
            "reaches_blinds_successor": count(
                lambda r: r["candidate"]["condition_catalog_item_id"] == BLINDS_ID),
            "reaches_fabric_successor": count(
                lambda r: r["candidate"]["condition_catalog_item_id"] == FABRIC_ID),
            "reaches_neither_successor": count(
                lambda r: r["candidate"]["condition_catalog_item_id"] not in successors),
            "candidate_items": _tally(executed, lambda r: r["candidate"]["condition_catalog_item_id"]),
            "baseline_items": _tally(executed, lambda r: r["baseline"]["condition_catalog_item_id"])},
        "resolution": {
            "resolved_changed_between_arms": count(lambda r: r["resolved_changed"]),
            "baseline_2d_reproduces_frozen": count(lambda r: r["baseline_2d_matches_frozen"]),
            "baseline_2d_drift": [
                {"case_id": r["case_id"], "frozen": r["frozen_owner"],
                 "live": r["baseline"]["resolved_item_id"],
                 "frozen_path": r["frozen_resolution_path"],
                 "live_path": r["baseline"]["resolution_path"]}
                for r in executed if not r["baseline_2d_matches_frozen"]],
            "candidate_matches_offline_shortcut": count(
                lambda r: r["candidate_matches_offline_shortcut"] is True),
            "candidate_diverges_from_offline_shortcut": [
                {"case_id": r["case_id"], "offline": r["offline_prediction"]["candidate_shortcut"],
                 "live": r["candidate"]["resolved_item_id"]}
                for r in executed if r["candidate_matches_offline_shortcut"] is False],
            "paths_baseline": _tally(executed, lambda r: r["baseline"]["resolution_path"]),
            "paths_candidate": _tally(executed, lambda r: r["candidate"]["resolution_path"])},
        "terra": {
            "verdict_changed": len(verdict_flips),
            "verdict_changes": [
                {"case_id": r["case_id"], "baseline": r["baseline"]["terra_verdict"],
                 "candidate": r["candidate"]["terra_verdict"],
                 "baseline_item": r["baseline"]["condition_catalog_item_id"],
                 "candidate_item": r["candidate"]["condition_catalog_item_id"]}
                for r in verdict_flips],
            "verdicts_baseline": _tally(executed, lambda r: r["baseline"]["terra_verdict"]),
            "verdicts_candidate": _tally(executed, lambda r: r["candidate"]["terra_verdict"]),
            "replica_noise_expected_flips": round(TERRA_REPLICA_FLIP_RATE * n, 2) if n else 0,
            "replica_flip_rate_of_record": TERRA_REPLICA_FLIP_RATE},
        "disposition": {
            "disposition_changed": count(lambda r: r["disposition_changed"]),
            "route_changed": count(lambda r: r["route_changed"]),
            "dispositions_baseline": _tally(executed, lambda r: r["baseline"]["disposition"]),
            "dispositions_candidate": _tally(executed, lambda r: r["candidate"]["disposition"]),
            "routes_baseline": _tally(executed, lambda r: r["baseline"]["route"]),
            "routes_candidate": _tally(executed, lambda r: r["candidate"]["route"])},
        "billing": {
            "billing_changed": count(lambda r: r["billing_changed"]),
            "candidate_zero_dollar_no_action": count(
                lambda r: r["candidate"]["route"] == "no_action" and not (r["candidate"]["billed_low"] or 0)),
            "baseline_billed_total": {
                "low": sum(r["baseline"]["billed_low"] or 0 for r in executed),
                "high": sum(r["baseline"]["billed_high"] or 0 for r in executed)},
            "candidate_billed_total": {
                "low": sum(r["candidate"]["billed_low"] or 0 for r in executed),
                "high": sum(r["candidate"]["billed_high"] or 0 for r in executed)},
            "note": "standalone billing over the reviewed units only; never a property headline"},
        "hijacking": {
            "candidate_rows_on_a_third_item": [
                {"case_id": r["case_id"], "item": r["candidate"]["condition_catalog_item_id"],
                 "frozen_owner": r["frozen_owner"], "observation": r["observation"],
                 "verdict": r["candidate"]["terra_verdict"], "route": r["candidate"]["route"],
                 "billed_low": r["candidate"]["billed_low"]}
                for r in executed if r["hijack_candidate"]]},
        "shared_conditions": {
            "candidate_rows_in_a_shared_condition": [
                {"case_id": r["case_id"], "item": r["candidate"]["condition_catalog_item_id"],
                 "condition_id": r["candidate"]["condition_id"]}
                for r in executed if r["candidate"]["condition_is_shared"]]},
        "census_reconciliation": {
            "pinned_rows_by_bucket": _tally(rows, lambda r: r["census_bucket"]),
            "manifest_offline_expectations": m["metrics"]["offline_expectations_to_confirm"],
            "note": "the manifest's 48 / 28 / 10 / 1 is the 25-artifact evidence-era corpus; the "
                    "pinned Stage A set is 66 rows in 44 conditions across 22 artifacts"},
        "controls": control_table,
        "material_findings": {
            k: [{"case_id": r["case_id"], "observation": r["observation"],
                 "offline_prediction": r["offline_prediction"],
                 "baseline": r["baseline"], "candidate": r["candidate"]} for r in v]
            for k, v in findings.items()},
        "neighbours": {
            a: {"conditions": len(arms[a].get("properties", []) and
                                 [x for p in arms[a]["properties"] for x in p["neighbour_conditions"]]),
                "verdicts": _tally([x for p in arms[a]["properties"] for x in p["neighbour_conditions"]],
                                   lambda x: x["terra_verdict"])} for a in arms},
    }

    crosswalk = [{
        "manifest_condition_id": mc["condition_id"], "case_id": cid, "issue_id": mc["issue_id"],
        "property_key": mc["property_key"],
        "baseline": None if not cases_by_arm["baseline"].get(cid) else {
            k: cases_by_arm["baseline"][cid].get(k)
            for k in ("condition_id", "condition_catalog_item_id", "estimate_unit_id")},
        "candidate": None if not cases_by_arm["candidate"].get(cid) else {
            k: cases_by_arm["candidate"][cid].get(k)
            for k in ("condition_id", "condition_catalog_item_id", "estimate_unit_id")},
    } for cid, mc in manifest_cases.items()]

    return {"generated_at_utc": utc_now(), "rows": rows, "aggregates": aggregates,
            "crosswalk": crosswalk,
            "note": "Part A computes numbers and row lists only. No rubric is applied and no "
                    "recommendation is derived; that is Part B's work on this frozen evidence."}


def _tally(items, key) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        try:
            k = key(it)
        except (TypeError, KeyError):
            k = None
        out[str(k)] = out.get(str(k), 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- freeze

def freeze(m: Dict[str, Any], out_root: Path, *, verify_record: Dict[str, Any]) -> Dict[str, Any]:
    arms = {a: read_json(out_root / a / "arm_record.json") for a in ("baseline", "candidate")}
    cmp_rec = compare(m, out_root)
    # The dry run writes under _plan/; it is planning evidence, not run evidence, and must not be
    # listed among the artifacts Part B re-hashes.
    def _run_files(pattern: str):
        return sorted(p for p in out_root.rglob(pattern) if "_plan" not in p.parts)
    unit_files = _run_files("terra_unit_*.json")
    prop_files = _run_files("property_record.json")
    pass2d_files = _run_files("pass_2d.json")
    bundle = {
        "schema_version": "catalog-audit-live-validation-v1",
        "program": "catalog_audit", "session": 6, "part": "A",
        "status": "awaiting_evaluation", "generated_at_utc": utc_now(),
        "produced_by": "scripts/catalog_audit_live_validation.py",
        "script_sha256": sha256_file(SCRIPT_PATH),
        "what_this_is": (
            "Stage A evidence only. Part A ran the experiment and recorded what happened; it applies "
            "no rubric and states no recommendation. Part B reads this bundle, re-verifies its "
            "hashes, inspects the per-case evidence, and decides."),
        "manifest": {
            "path": rel(MANIFEST_PATH), "file_sha256": sha256_file(MANIFEST_PATH),
            "manifest_sha256": m["manifest_sha256"],
            "session5_file_sha256": SESSION5_FILE_SHA256,
            "session5_manifest_sha256": SESSION5_MANIFEST_SHA256,
            "content_proof": manifest_content_proof(m), "budget": m["budget"],
            "status_field_note": "the manifest's own `status` still reads "
                                 "awaiting_human_cost_authorization: only the budget block was "
                                 "edited, and a stale-but-cautious status fails safe"},
        "arms": {a: {k: arms[a][k] for k in
                     ("arm", "arm_root", "catalog_sha256", "projection_fingerprint",
                      "projection_version", "source_run_id", "terra_model", "terra_max_output_tokens",
                      "terra_reasoning_effort", "pass_2d_route", "started_utc", "finished_utc",
                      "spend", "stopped")} for a in arms},
        "invariants": verify_record,
        "per_case": cmp_rec["rows"], "aggregates": cmp_rec["aggregates"],
        "crosswalk": cmp_rec["crosswalk"],
        "crosswalk_note": (
            "the manifest's condition ids are 3.1-era: condition_id hashes estimate_id, which hashes "
            "the catalog sha and projection fingerprint, so none of the 44 reproduces under either "
            "3.2 arm. Join by case_id / issue_id and use this crosswalk for the manifest's own ids."),
        "per_property": {a: arms[a]["properties"] for a in arms},
        "per_unit": {a: arms[a]["units"] for a in arms},
        "excluded_rows": arms["baseline"]["excluded_rows"],
        "raw_artifacts": {
            "run_root": rel(out_root),
            "arm_records": {a: {"path": rel(out_root / a / "arm_record.json"),
                                "sha256": sha256_file(out_root / a / "arm_record.json")} for a in arms},
            "terra_ledger": rel(out_root / ".renovation_architecture" / "terra_usage.sqlite3"),
            "unit_records": [{"path": rel(p), "sha256": sha256_file(p)} for p in unit_files],
            "property_records": [{"path": rel(p), "sha256": sha256_file(p)} for p in prop_files],
            "pass_2d_records": [{"path": rel(p), "sha256": sha256_file(p)} for p in pass2d_files],
            "authorization_proof": rel(out_root / "provenance" / "authorization_proof.json")},
        "deviations": [
            {"id": "D1", "what": "a purpose-built harness rather than an existing driver",
             "why": "no entry point replays a stored Pass 2c into a live Pass 2d: the canary drivers "
                    "re-run 2a/2b/2c, replay_renovation_architecture is provider-free by "
                    "construction, and redecide_renovation_architecture refuses on catalog drift "
                    "(every stored artifact is catalog 3.1 / projection v2)",
             "impact": "one untracked script and its tests; no production module changed"},
            {"id": "D2", "what": "Pass 2d re-run for the 66 pinned rows only",
             "why": "the manifest freezes 2a/2b/2c and pins the case set; every other row keeps its "
                    "stored resolution, and all of those ids exist in both 3.2 catalogs (checked)",
             "impact": "the replay differs from the stored run only where the experiment intends"},
            {"id": "D3", "what": "four parent-owned rows dropped from the replay lane in both arms",
             "why": "they sit outside the pinned case set and are stale ids under the candidate "
                    "catalog, which the projection fails closed on",
             "impact": "no target unit is affected; each is its own condition. See excluded_rows"},
            {"id": "D4", "what": "whole target units reviewed, not only the pinned conditions",
             "why": "Terra judges a unit; sending a subset would change the photo set, the prompt "
                    "numbering and the request fingerprint, and would measure a different question",
             "impact": "neighbouring conditions are reviewed too, and are reported as migration "
                       "context rather than as cases"},
            {"id": "D5", "what": "billing is a standalone estimate over the reviewed units",
             "why": "nothing in the repository emits a per-condition billed_amount; distinct-unit "
                    "pricing and max-envelope dedup make a whole-property figure unavailable without "
                    "running whole properties",
             "impact": "dollar figures compare the two arms and are never a property headline"}],
        "not_done_in_part_a": [
            "no rubric applied and no recommendation stated (Part B)",
            "no RESULT document and no HANDOFF_SESSION_6 (Part B)",
            "no Sol call, no package candidates, no v4/v5 totals: Stage A reviews conditions only",
            "no Stage B canary: it remains unauthorized",
            "no benchmark gold slice edited (S5-4 deferred to its own task)",
            "nothing committed, nothing published"],
        "noise_floor": {
            "terra_replica_flip_rate": TERRA_REPLICA_FLIP_RATE, "wilson_95": [0.040, 0.101],
            "source": "16/250 canary replica flips (reports/review_analysis.md section 6)",
            "pass_2d_sampling": f"temperature {cfg.PASS_2D_TEMPERATURE}, no seed: the LLM resolution path is not "
                                "deterministic, so a baseline resolution may differ from the frozen "
                                "one without any catalog cause"},
    }
    sha = write_json(BUNDLE_PATH, bundle)
    print(f"wrote {rel(BUNDLE_PATH)} sha256={sha}")
    return {"bundle": bundle, "sha256": sha}



# --------------------------------------------------------------------------- replicate (repeat stage)

def _repeat_authorization(m: Dict[str, Any]) -> Tuple[int, int]:
    """(stage A figure, repeat figure). Both must be human-filled; the ledger is cumulative for the
    UTC day, so the effective ceiling for the repeat is their sum."""
    b = m["budget"]
    a = b.get("stage_a_authorized_tokens")
    r = b.get("stage_a_repeat_authorized_tokens")
    if not (isinstance(a, int) and a > 0 and isinstance(r, int) and r > 0
            and b.get("stage_a_repeat_authorized_by") and b.get("stage_a_repeat_authorized_at")):
        fail("repeat authorization missing from the manifest budget block; refusing to make any call")
    return a, r


def replicate_arm(m: Dict[str, Any], *, arm: str, out_root: Path, pass_2d_replicas: int,
                  terra_replica: bool, limit_properties: Optional[int]) -> Dict[str, Any]:
    """The repeat stage for one arm. Stage A records are read, never written."""
    assert_not_imported(SOL_MODULE)
    _guard_out_root(out_root)
    want_arm = m["arms"][arm]
    if str(ARM_ROOT) != str(Path(want_arm["checkout"]).resolve()):
        fail(f"--arm-root {ARM_ROOT} is not the {arm} checkout {want_arm['checkout']}")
    ctx = arm_projection(ARM_ROOT)
    projection, catalog = ctx["projection"], ctx["catalog"]
    if projection["catalog_sha256"] != want_arm["catalog_sha256"]:
        fail(f"{arm} catalog sha drift: {projection['catalog_sha256']}")
    if projection["fingerprint"] != want_arm["projection_fingerprint"]:
        fail(f"{arm} projection fingerprint drift: {projection['fingerprint']}")
    observables = projection["observables"]
    terminal_routes = projection["terminal_routes"]
    stage_a_auth, repeat_auth = _repeat_authorization(m)
    ceiling = stage_a_auth + repeat_auth
    terra_model = cfg.RENOVATION_TERRA_MODEL
    max_out = int(cfg.RENOVATION_TERRA_MAX_OUTPUT_TOKENS)

    from tools.catalog_embeddings import build_candidate_provider
    from tools.pass_2f_artifact_inputs import photo_key_to_path
    from tools.pass_config import SceneClassifierRunOptions, get_model_config_for_pass
    from tools.renovation_architecture.conditions import build_observed_conditions
    from tools.renovation_architecture.contracts import TERRA_REVIEW_REASONING_EFFORT
    from tools.renovation_architecture.disposition import decide_disposition
    from tools.renovation_architecture.evidence import build_evidence_facts, build_photo_identity_index
    from tools.renovation_architecture.ids import make_estimate_id
    from tools.renovation_architecture.review_pipeline import _review_unit_fresh as review_unit_fresh
    from tools.renovation_architecture.terra_review import build_unit_request
    from tools.renovation_architecture.usage_guard import (TerraDailyBudgetExceeded, TerraUsageLedger,
                                                           estimate_reservation_tokens)
    from tools.scene_classifier_passes import PassExecutionError
    from tools.vlm_client import create_vlm_client, get_model_configs_from_pipeline_config

    if not cfg.OPENAI_API_KEY:
        fail("OPENAI_API_KEY is not resolved")
    qwen_config, gpt5_config = get_model_configs_from_pipeline_config(cfg)
    model_config_2d = get_model_config_for_pass("2d", SceneClassifierRunOptions(premium=True),
                                                qwen_config, gpt5_config)
    if model_config_2d.get("provider") != "lmstudio" or model_config_2d.get("model") != cfg.LM_STUDIO_MODEL:
        fail("Pass 2d route drifted off the local model")
    vlm_client = create_vlm_client()
    vlm_client.reset_usage_stats()
    if pass_2d_replicas:
        _warm_pass_2d(vlm_client, model_config_2d)
    ledger = TerraUsageLedger(out_root, daily_ceiling=ceiling + 50_000, usage_root_override=None)
    source_run_id = f"catalog_audit_s6_{arm}"
    provider = None
    groups = cases_by_run(m)
    if limit_properties:
        groups = groups[:limit_properties]

    rec: Dict[str, Any] = {
        "schema_version": "catalog-audit-live-validation-replica-v1", "arm": arm,
        "started_utc": utc_now(), "catalog_sha256": projection["catalog_sha256"],
        "projection_fingerprint": projection["fingerprint"], "pass_2d_replicas": pass_2d_replicas,
        "terra_replica": terra_replica, "ceiling_tokens": ceiling,
        "pass_2d": [], "terra": [], "properties": [], "stopped": None,
    }
    spent_before = ledger.spent_today()
    stop = None
    for g in groups:
        if stop:
            break
        art_path = resolve_input(g["artifact_path"])
        if sha256_file(art_path) != g["artifact_sha256"]:
            fail(f"artifact drift at {art_path}")
        art = read_json(art_path)
        lane = product_lane(art)
        photos = art.get("photos") or {}
        property_metadata = art.get("property_metadata")
        paths = photo_key_to_path(art)
        prop_dir = out_root / arm / g["property_key"] / g["run_id"]
        stage_a = read_json(prop_dir / "pass_2d.json")
        by_issue = {r.get("issue_id"): r for r in lane}
        rows = [{"case": c, "lane_row": by_issue.get(c["issue_id"])} for c in g["cases"]]
        expected_key = sha256_canonical({
            "catalog": projection["catalog_sha256"], "top_k": TOP_K,
            "rows": [{"i": r["case"]["issue_id"], "o": r["case"]["frozen_pass_2c"]["observation"],
                      "k": r["case"]["frozen_pass_2c"]["kind"],
                      "s": r["case"]["frozen_pass_2c"]["scene_group"]} for r in rows]})
        if stage_a.get("cache_key") != expected_key:
            fail(f"{g['property_key']}: Stage A pass_2d record does not match the pinned rows")

        # ---- Pass 2d replicas (local). Each replica file is final once written.
        prop_2d = []
        for k in range(1, pass_2d_replicas + 1):
            rpath = prop_dir / f"pass_2d_replica_{k}.json"
            if rpath.is_file() and read_json(rpath).get("cache_key") == expected_key:
                prop_2d.append(read_json(rpath))
                continue
            if provider is None:
                provider = build_candidate_provider(catalog)
            res = asyncio.run(_resolve_rows(rows, provider=provider, vlm_client=vlm_client,
                                            model_config=model_config_2d, photos=photos))
            broken = [r for r in res if r["status"] in ("pass_error", "error")]
            if broken:
                fail(f"{g['property_key']} replica {k}: {len(broken)} Pass 2d failures; nothing cached")
            payload = {"cache_key": expected_key, "arm": arm, "replica": k, "resolutions": res}
            write_json(rpath, payload)
            prop_2d.append(payload)
        for r in rows:
            iid = r["case"]["issue_id"]
            a0 = next(x for x in stage_a["resolutions"] if x["issue_id"] == iid)
            reps = [next(x for x in p_["resolutions"] if x["issue_id"] == iid) for p_ in prop_2d]
            rec["pass_2d"].append({
                "case_id": r["case"]["case_id"], "arm": arm, "issue_id": iid,
                "observation": r["case"]["frozen_pass_2c"]["observation"],
                "stage_a": {"resolved_item_id": a0.get("resolved_item_id"), "path": a0.get("resolution_path"),
                            "margin": a0.get("top_two_margin")},
                "replicas": [{"resolved_item_id": x.get("resolved_item_id"), "path": x.get("resolution_path"),
                              "margin": x.get("top_two_margin")} for x in reps]})

        # ---- Terra replica: identical conditions and units to Stage A, fresh calls
        prop_terra = []
        prop_stop = None
        if terra_replica:
            res_by_issue = {r["issue_id"]: r for r in stage_a["resolutions"]}
            replay_lane = []
            for row in lane:
                iid = row.get("issue_id")
                if iid in UNPINNED_PARENT_ISSUE_IDS:
                    continue
                new = dict(row)
                live = res_by_issue.get(iid)
                if live is not None:
                    rid = live.get("resolved_item_id")
                    new["catalog_item_id"] = rid
                    new["catalog_item_kind"] = (observables.get(rid) or {}).get("kind") if rid else None
                replay_lane.append(new)
            estimate_id = make_estimate_id(property_key=g["property_key"], source_run_id=source_run_id,
                                           catalog_sha256=projection["catalog_sha256"],
                                           projection_fingerprint=projection["fingerprint"])
            drafts = build_observed_conditions(issues_flat=replay_lane, photos=photos,
                                               property_metadata=property_metadata,
                                               projection=projection, estimate_id=estimate_id)
            pinned_ids = {c["issue_id"] for c in g["cases"]}
            target_units = sorted({d.condition.estimate_unit_id for d in drafts
                                   if pinned_ids & set(d.condition.issue_ids)})
            unit_drafts = {u: [d for d in drafts if d.condition.estimate_unit_id == u] for u in target_units}
            needed = sorted({ref["photo_key"] for u in target_units for d in unit_drafts[u]
                             for ref in d.evidence_refs})
            identity_index = build_photo_identity_index(needed, paths)
            for unit_id in target_units:
                pairs = [(d, build_evidence_facts(d, identity_index=identity_index, observables=observables,
                                                  estimate_id=estimate_id)) for d in unit_drafts[unit_id]]
                request = build_unit_request(
                    estimate_unit_id=unit_id, unit_pairs=pairs, observables=observables,
                    photo_key_to_path=paths, identity_index=identity_index,
                    projection_fingerprint=projection["fingerprint"], model=terra_model,
                    reasoning_effort=TERRA_REVIEW_REASONING_EFFORT, max_output_tokens=max_out)
                a_unit_path = _unit_path(prop_dir, unit_id)
                a_unit = read_json(a_unit_path) if a_unit_path.is_file() else None
                same_prompt = bool(a_unit and a_unit.get("request_fingerprint") == request.request_fingerprint
                                   and a_unit.get("status") == "reviewed")
                if not same_prompt:
                    fail(f"{g['property_key']} {unit_id}: rebuilt request fingerprint differs from Stage A's; "
                         f"the replica would not be a same-prompt control")
                rpath = prop_dir / "units_replica" / a_unit_path.name
                prior = read_json(rpath) if rpath.is_file() else None
                if prior and prior.get("request_fingerprint") == request.request_fingerprint \
                        and prior.get("status") == "reviewed":
                    unit_rec = prior
                else:
                    reservation = estimate_reservation_tokens(max_output_tokens=max_out,
                                                              request_bytes=request.request_bytes,
                                                              image_count=len(request.image_paths))
                    spent = ledger.spent_today()
                    if spent + reservation > ceiling:
                        prop_stop = {"reason": "stopped_at_authorization", "spent_today": spent,
                                     "reservation": reservation, "ceiling": ceiling, "at_unit": unit_id}
                        break
                    try:
                        call, reviews = review_unit_fresh(
                            request=request, vlm_client=vlm_client, api_key=cfg.OPENAI_API_KEY,
                            terra_model=terra_model, terra_max_output_tokens=max_out, ledger=ledger,
                            property_key=g["property_key"], source_run_id=source_run_id + "_replica",
                            estimate_id=estimate_id)
                    except TerraDailyBudgetExceeded as exc:
                        prop_stop = {"reason": "ledger_denied", "detail": str(exc)[:300], "at_unit": unit_id}
                        break
                    except PassExecutionError as exc:
                        prop_stop = {"reason": "terra_pass_error", "detail": str(exc)[:300], "at_unit": unit_id}
                        break
                    unit_rec = {"schema_version": "catalog-audit-live-validation-unit-replica-v1",
                                "arm": arm, "property_key": g["property_key"], "estimate_unit_id": unit_id,
                                "request_fingerprint": request.request_fingerprint,
                                "stage_a_unit": rel(a_unit_path), "status": "reviewed",
                                "terra_call": call, "reviews": reviews}
                    write_json(rpath, unit_rec)
                    print(f"  [{arm} replica] {g['property_key']} {unit_id}: {len(reviews)} conditions, "
                          f"{call['total_tokens']} tokens, ledger {ledger.spent_today():,}/{ceiling:,}", flush=True)
                a_by_cond = {r["condition_id"]: r for r in a_unit["reviews"]}
                cond_meta = {d.condition.condition_id: d.condition for d in unit_drafts[unit_id]}
                for rv in unit_rec["reviews"]:
                    cid = rv["condition_id"]
                    cond = cond_meta[cid]
                    is_target = bool(pinned_ids & set(cond.issue_ids))
                    route = terminal_routes[cond.catalog_item_id]["route"]
                    ev = next(e for d, e in pairs if d.condition.condition_id == cid)
                    disp_a = decide_disposition(a_by_cond[cid]["verdict"], route, ev.distinct_view_count,
                                                ev.min_photo_evidence_required)[0]
                    disp_r = decide_disposition(rv["verdict"], route, ev.distinct_view_count,
                                                ev.min_photo_evidence_required)[0]
                    prop_terra.append({
                        "arm": arm, "property_key": g["property_key"], "estimate_unit_id": unit_id,
                        "condition_id": cid, "catalog_item_id": cond.catalog_item_id,
                        "issue_ids": list(cond.issue_ids), "is_target": is_target, "route": route,
                        "stage_a_verdict": a_by_cond[cid]["verdict"], "replica_verdict": rv["verdict"],
                        "stage_a_disposition": disp_a, "replica_disposition": disp_r,
                        "replica_rationale": rv.get("rationale"), "flipped": a_by_cond[cid]["verdict"] != rv["verdict"]})
        rec["terra"].extend(prop_terra)
        rec["properties"].append({"property_key": g["property_key"], "pass_2d_replicas_done": len(prop_2d),
                                  "terra_conditions_compared": len(prop_terra), "stop": prop_stop})
        if prop_stop:
            stop = prop_stop
    spent_after = ledger.spent_today()
    rec["stopped"] = stop
    rec["finished_utc"] = utc_now()
    rec["spend"] = {"ledger_spent_today_before": spent_before, "ledger_spent_today_after": spent_after,
                    "ledger_delta": spent_after - spent_before, "ceiling_tokens": ceiling,
                    "terra_replica_calls": len({(t["property_key"], t["estimate_unit_id"]) for t in rec["terra"]})}
    rec["vlm_usage_stats"] = dict(getattr(vlm_client, "usage_stats", {}) or {})
    out = out_root / arm / "replica_record.json"
    sha = write_json(out, rec)
    print(f"[{arm} replica] {len(rec['properties'])} properties, {len(rec['pass_2d'])} rows x {pass_2d_replicas} "
          f"2d replicas, {rec['spend']['terra_replica_calls']} Terra units, ledger delta "
          f"{rec['spend']['ledger_delta']:,}; wrote {rel(out)} sha256={sha[:16]}")
    return rec


def compare_replica(m: Dict[str, Any], out_root: Path) -> Dict[str, Any]:
    """Mechanical: Pass 2d stability per row and same-prompt Terra flip rates per arm."""
    recs = {a: read_json(out_root / a / "replica_record.json") for a in ("baseline", "candidate")}
    successors = {BLINDS_ID, FABRIC_ID}
    rows_out = []
    manifest_cases = {c["case_id"]: c for c in m["cases"]}
    for cid, mc in manifest_cases.items():
        entry: Dict[str, Any] = {"case_id": cid, "role": mc["role"], "review_card_id": mc["review_card_id"],
                                 "observation": mc["frozen_pass_2c"]["observation"]}
        for a in recs:
            r = next((x for x in recs[a]["pass_2d"] if x["case_id"] == cid), None)
            if r is None:
                entry[a] = None
                continue
            items = [r["stage_a"]["resolved_item_id"]] + [x["resolved_item_id"] for x in r["replicas"]]
            entry[a] = {"stage_a": r["stage_a"]["resolved_item_id"],
                        "distribution": _tally(items, lambda x: x),
                        "n": len(items), "agree_with_stage_a": sum(1 for x in items[1:] if x == items[0]),
                        "landed_on_trim": sum(1 for x in items if x == "dated_interior_trim"),
                        "landed_on_billable_third": sum(1 for x in items if x and x not in successors
                                                         and x != mc["frozen_resolution"]["resolved_item_id"]
                                                         and x not in (WINDOWS_ID, "dated_overall_decor_style")),
                        "paths": _tally([r["stage_a"]["path"]] + [x["path"] for x in r["replicas"]], lambda x: x)}
        rows_out.append(entry)

    def terra_stats(a: str, only_target: Optional[bool]) -> Dict[str, Any]:
        t = [x for x in recs[a]["terra"] if only_target is None or x["is_target"] == only_target]
        n = len(t)
        flips = [x for x in t if x["flipped"]]
        return {"n": n, "flips": len(flips), "rate": round(len(flips) / n, 4) if n else None,
                "disposition_changes": sum(1 for x in t if x["stage_a_disposition"] != x["replica_disposition"]),
                "flip_pairs": _tally(flips, lambda x: f"{x['stage_a_verdict']}->{x['replica_verdict']}")}

    trim_rows = [r for r in rows_out if any(r[a] and r[a]["landed_on_trim"] for a in recs if r.get(a))]
    s61 = {
        "candidate_rows_ever_on_trim": [{"case_id": r["case_id"], "observation": r["observation"],
                                         "candidate": r["candidate"], "baseline": r["baseline"]}
                                        for r in rows_out if r.get("candidate") and r["candidate"]["landed_on_trim"]],
        "baseline_rows_ever_on_trim": [{"case_id": r["case_id"], "observation": r["observation"],
                                        "baseline": r["baseline"]}
                                       for r in rows_out if r.get("baseline") and r["baseline"]["landed_on_trim"]],
        "candidate_trim_landings_total": sum(r["candidate"]["landed_on_trim"] for r in rows_out if r.get("candidate")),
        "baseline_trim_landings_total": sum(r["baseline"]["landed_on_trim"] for r in rows_out if r.get("baseline")),
        "resolutions_per_arm": sum(r["candidate"]["n"] for r in rows_out if r.get("candidate")),
    }
    stability = {a: {"rows": sum(1 for r in rows_out if r.get(a)),
                     "rows_fully_stable": sum(1 for r in rows_out if r.get(a) and r[a]["agree_with_stage_a"] == r[a]["n"] - 1),
                     "replica_resolutions": sum(r[a]["n"] - 1 for r in rows_out if r.get(a)),
                     "replica_resolutions_agreeing": sum(r[a]["agree_with_stage_a"] for r in rows_out if r.get(a))}
                 for a in recs}
    for a in stability:
        st = stability[a]
        st["agreement_rate"] = round(st["replica_resolutions_agreeing"] / st["replica_resolutions"], 4) if st["replica_resolutions"] else None
    trim_flip_terra = [x for a in recs for x in recs[a]["terra"] if x["catalog_item_id"] == "dated_interior_trim" and x["is_target"]]
    controls_terra = []
    for r in rows_out:
        if r["role"] != "control":
            continue
        for a in recs:
            hits = [x for x in recs[a]["terra"] if x["is_target"] and any(iid == manifest_cases[r["case_id"]]["issue_id"] for iid in x["issue_ids"])]
            for x in hits:
                controls_terra.append({"arm": a, "review_card_id": r["review_card_id"], "item": x["catalog_item_id"],
                                       "stage_a": x["stage_a_verdict"], "replica": x["replica_verdict"]})
    return {"generated_at_utc": utc_now(), "pass_2d_rows": rows_out, "pass_2d_stability": stability,
            "terra_same_prompt": {a: {"all": terra_stats(a, None), "targets": terra_stats(a, True),
                                      "neighbours": terra_stats(a, False)} for a in recs},
            "s6_1": s61, "trim_conditions_terra_replica": trim_flip_terra, "controls_terra_replica": controls_terra,
            "spend": {a: recs[a]["spend"] for a in recs},
            "note": "numbers only; interpretation is Part B's"}


# --------------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-root", default=str(MAIN_ROOT))
    ap.add_argument("--candidate-root", default=str(CANDIDATE_ROOT_DEFAULT))
    ap.add_argument("--out-root", default=str(RUN_ROOT))
    ap.add_argument("--lm-studio-url", default=None,
                    help="substitute route to the Pass 2d model server when the manifest's pinned "
                         "address is unreachable; recorded as a validated deviation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    r.add_argument("--limit-properties", type=int, default=None)
    sub.add_parser("compare")
    sub.add_parser("freeze")
    rp = sub.add_parser("replicate")
    rp.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    rp.add_argument("--pass-2d-replicas", type=int, default=5)
    rp.add_argument("--terra-replica", action="store_true")
    rp.add_argument("--limit-properties", type=int, default=None)
    sub.add_parser("compare-replica")
    args = ap.parse_args(argv)

    m = load_manifest()
    out_root = Path(args.out_root).resolve()
    candidate_root = Path(args.candidate_root).resolve()

    if args.cmd == "verify":
        rec = verify(m, candidate_root=candidate_root, require_live=False)
        write_json(out_root / "verify.json", rec)
        for c in rec["checks"]:
            mark = {"pass": "ok   ", "FAIL": "FAIL ", "deferred": "defer"}[c["result"]]
            print(f"  {mark} {c['id']:<22} {c['name']}")
            if c["result"] != "pass":
                print(f"         detail: {json.dumps(c['detail'], default=str)[:400]}")
        print(f"verify: {rec['result']} ({len(rec['checks'])} checks, {len(rec['failures'])} failures)")
        return 0 if rec["result"] == "pass" else 2

    if args.cmd == "plan":
        rec = run_arm(m, arm="baseline", out_root=out_root / "_plan", limit_properties=None,
                      dry_run=True)
        units = [u for u in rec["units"] if u.get("status") == "planned"]
        total_res = sum(u["estimated_reservation_tokens"] for u in units)
        conds = sum(len(u["condition_ids"]) for u in units)
        print(f"plan: {len(rec['properties'])} properties, {len(units)} target units per arm, "
              f"{conds} conditions per arm")
        print(f"      request bytes {sum(u['request_bytes'] for u in units):,}; "
              f"reservations {total_res:,}/arm, {2 * total_res:,} both arms")
        print(f"      authorization {m['budget']['stage_a_authorized_tokens']:,}; "
              f"at the stored ~5k/call actual, both arms ~{2 * len(units) * 5000:,}")
        return 0

    if args.cmd == "run":
        rec = verify(m, candidate_root=candidate_root, require_live=True)
        write_json(out_root / f"verify_{args.arm}.json", rec)
        if rec["result"] != "pass":
            for c in rec["failures"]:
                print(f"  FAIL {c['id']} {c['name']}: {json.dumps(c['detail'], default=str)[:300]}")
            fail("verify failed; refusing to make any provider call")
        run_arm(m, arm=args.arm, out_root=out_root, limit_properties=args.limit_properties,
                dry_run=False)
        return 0

    if args.cmd == "compare":
        assert_not_imported(*PROVIDER_MODULES, SOL_MODULE)
        rec = compare(m, out_root)
        sha = write_json(out_root / "compare.json", rec)
        print(json.dumps(rec["aggregates"], indent=2, default=str)[:6000])
        print(f"wrote compare.json sha256={sha[:16]}")
        return 0

    if args.cmd == "replicate":
        rec = verify(m, candidate_root=candidate_root, require_live=True)
        write_json(out_root / f"verify_replica_{args.arm}.json", rec)
        if rec["result"] != "pass":
            for c in rec["failures"]:
                print(f"  FAIL {c['id']} {c['name']}: {json.dumps(c['detail'], default=str)[:300]}")
            fail("verify failed; refusing to make any provider call")
        replicate_arm(m, arm=args.arm, out_root=out_root, pass_2d_replicas=args.pass_2d_replicas,
                      terra_replica=args.terra_replica, limit_properties=args.limit_properties)
        return 0

    if args.cmd == "compare-replica":
        assert_not_imported(*PROVIDER_MODULES, SOL_MODULE)
        rec = compare_replica(m, out_root)
        sha = write_json(out_root / "compare_replica.json", rec)
        print(json.dumps({k: rec[k] for k in ("pass_2d_stability", "terra_same_prompt", "s6_1", "spend")},
                         indent=1, default=str)[:6000])
        print(f"wrote compare_replica.json sha256={sha[:16]}")
        return 0

    if args.cmd == "freeze":
        assert_not_imported(*PROVIDER_MODULES, SOL_MODULE)
        vpath = out_root / "verify_candidate.json"
        vrec = read_json(vpath) if vpath.is_file() else read_json(out_root / "verify.json")
        freeze(m, out_root, verify_record=vrec)
        return 0
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HarnessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
