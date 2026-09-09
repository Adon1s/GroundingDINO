"""Catalog-audit Session 5: Tier 1 deterministic and Tier 2 provider-free retrieval validation.

Validates the approved CAP-007 candidate (window-treatment split) against the clean baseline.
Two arms, never one: the baseline is the main checkout at 9afe0fa, the candidate is the
worktree at 6e67eaa. Nothing here writes to either arm's tracked files; every output goes to
the main checkout's reports/ and docs/ by absolute path.

Provider-free by construction. The only network endpoint used is the embeddings sidecar
(EMBEDDINGS_BASE_URL). No VLM/Terra/Sol client module may be imported -- asserted at startup --
so this cannot spend provider tokens even by accident.

Subcommands:
  tier1     arm identity, regeneration parity, validators, approved-vs-actual diff, economics/routes
  corpus    the frozen Pass 2c observation set for replay, from the 25 pinned evidence-era artifacts
  probe     real POST to the embeddings sidecar (a 200 from /health is not sufficient)
  snapshot  per-arm retrieval replay: kind/scene filtering, top-K, guardrails, lexical shortcut
  compare   join both snapshots, evaluate the Tier 2 rules, emit the tier12 report
  manifest  the pinned live-experiment manifest, only if every hard rule passed

The arm whose code and catalog are used is chosen by --arm-root, which is pre-parsed out of
argv and put on sys.path before `tools` is imported: `python scripts/catalog_audit_validation.py
snapshot --arm-root C:/.../rv-catalog-audit-s4`. tier1 always runs its comparisons from the
baseline arm's code, reading the candidate's data by explicit path, so the only variable is data.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]

# Pins identify WHICH candidate is under validation. They were hardcoded for
# CAP-007, which left the harness unrunnable against any later candidate (S7-3).
# They are now defaults: --manifest rebinds them, so the same check bodies run
# against a different candidate without being rewritten. Omitting --manifest
# reproduces the Session 5/6 behaviour exactly.
DEFAULT_PINS: Dict[str, Any] = {
    "baseline_commit": "9afe0fa5a0490a856abed507dccc4a02ed1de24c",
    "candidate_commit": "6e67eaa83887cf4d218100dcd060c07d3bd30522",
    "approvals_relpath": "reports/catalog_audit_approvals.json",
    "approvals_sha256": "a725c89af131e94b04867f4f5a226bda495278a8ba28fdd7162ed6188d76c871",
    "approved_proposal_id": "CAP-007",
    "approved_op_count": 7,
    "parent_id": "dated_window_treatment_valance",
    "blinds_id": "window_blinds_basic_or_plain",
    "fabric_id": "dated_window_valance_or_curtains",
    "windows_id": "dated_or_older_windows",
    "expected_diff_files": [
        "tests/test_catalog_cap007_window_treatment_split.py", "tests/test_catalog_kind_v2.py",
        "tests/test_kind_ontology_guards.py", "tests/test_renovation_architecture_catalog.py",
        "tools/catalog_migrations/2.1_to_3.0.json", "tools/catalog_migrations/2.1_to_3.0_audit.md",
        "tools/catalog_migrations/kind_v2_decisions.json", "tools/issue_catalog_kind_v2.json"],
    "expected_hashes": {
        "baseline": {"tools/catalog_migrations/kind_v2_decisions.json":
                     "47614d822bdd6b672d8596050b46839d04a8a6df0a04c18e052f4a63bd1903e8",
                     "tools/issue_catalog_kind_v2.json":
                     "51bf7e263ff98109d657ef730b0ce1a705598101f09843eb288b1bdc3e0eaa54"},
        "candidate": {"tools/catalog_migrations/kind_v2_decisions.json":
                      "2e49a82f9961c2623342ec84743a0997299fd50f0dfcc7939196029cf19b9c76",
                      "tools/issue_catalog_kind_v2.json":
                      "43d8147ee8abd4f4df51bcf88a83ba99d834c415b9d5d28295fa0a33f3457acf",
                      "tools/catalog_migrations/2.1_to_3.0.json":
                      "6b944b83fee7b350bdc7183d3def8f5849d3b97ca4d017c7dcc4efcd64936a18",
                      "tools/catalog_migrations/2.1_to_3.0_audit.md":
                      "cac2750a20799b837293c04b7afd098fb071f8326c1f9edcc41e1dc1ed0e8a81"}},
}
PINS: Dict[str, Any] = dict(DEFAULT_PINS)

BASELINE_COMMIT = PINS["baseline_commit"]
CANDIDATE_COMMIT = PINS["candidate_commit"]
APPROVALS_SHA256 = PINS["approvals_sha256"]
APPROVED_PROPOSAL_ID = PINS["approved_proposal_id"]
PARENT_ID = PINS["parent_id"]
BLINDS_ID = PINS["blinds_id"]
FABRIC_ID = PINS["fabric_id"]
WINDOWS_ID = PINS["windows_id"]


def apply_pins(manifest_path: Optional[str]) -> None:
    """Rebind the pin globals from a manifest. Unknown keys fail loudly rather
    than silently leaving a stale pin in force."""
    global PINS, BASELINE_COMMIT, CANDIDATE_COMMIT, APPROVALS_SHA256
    global APPROVED_PROPOSAL_ID, PARENT_ID, BLINDS_ID, FABRIC_ID, WINDOWS_ID
    if not manifest_path:
        return
    loaded = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    pins = loaded.get("pins", loaded)
    unknown = set(pins) - set(DEFAULT_PINS)
    if unknown:
        fail("pin manifest has unknown keys " + repr(sorted(unknown))
             + "; known keys are " + repr(sorted(DEFAULT_PINS)))
    PINS = {**DEFAULT_PINS, **pins}
    BASELINE_COMMIT = PINS["baseline_commit"]
    CANDIDATE_COMMIT = PINS["candidate_commit"]
    APPROVALS_SHA256 = PINS["approvals_sha256"]
    APPROVED_PROPOSAL_ID = PINS["approved_proposal_id"]
    PARENT_ID = PINS["parent_id"]
    BLINDS_ID = PINS["blinds_id"]
    FABRIC_ID = PINS["fabric_id"]
    WINDOWS_ID = PINS["windows_id"]
SCORE_EPSILON = 1e-5  # float32 matmul noise bound; thresholds sit 3 orders of magnitude above it
TOP_K = 8  # production effective value (scene_classifier_orchestrator context key), not the retriever default of 5

# Provider client modules that must never be imported by this process.
FORBIDDEN_MODULES = (
    "tools.vlm_client",
    "tools.openai_client",
    "tools.renovation_architecture.terra_review",
    "tools.renovation_architecture.sol_review",
    "tools.renovation_architecture.factorized_review",
)


class ValidationError(RuntimeError):
    """Operator-facing failure: drift, missing input, or a broken invariant."""


def fail(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    raise ValidationError(msg)


def _arm_root_from_argv(argv: Sequence[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--arm-root" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if arg.startswith("--arm-root="):
            return Path(arg.split("=", 1)[1]).resolve()
    return MAIN_ROOT


ARM_ROOT = _arm_root_from_argv(sys.argv[1:])
if str(ARM_ROOT) in sys.path:
    sys.path.remove(str(ARM_ROOT))
sys.path.insert(0, str(ARM_ROOT))

import tools  # noqa: E402
import tools.pipeline_config as cfg  # noqa: E402
from tools.comparison_common import canonical_json, sha256_bytes, sha256_canonical, sha256_file  # noqa: E402

if Path(tools.__file__).resolve().parent != (ARM_ROOT / "tools"):
    fail(f"imported tools from {tools.__file__}, expected {ARM_ROOT / 'tools'}")


def assert_no_provider_imports() -> None:
    loaded = sorted(m for m in FORBIDDEN_MODULES if m in sys.modules)
    if loaded:
        fail(f"provider client modules imported: {loaded}")


def rel(path: Path, root: Path = MAIN_ROOT) -> str:
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> str:
    """Write indented JSON with stable key order; return the file's sha256."""
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


# --------------------------------------------------------------------------- arm identity

RUNTIME_MODULES = (
    "tools/catalog_embeddings.py",
    "tools/scene_classifier_passes.py",
    "tools/scene_classifier_orchestrator.py",
    "tools/pipeline_config.py",
    "tools/pipeline_common.py",
    "tools/catalog_validation.py",
    "tools/renovation_architecture/catalog_projection.py",
    "tools/renovation_estimate.py",
    "tools/observation_kinds.py",
    "scripts/migrate_catalog_kind_v2.py",
)

CATALOG_ARTIFACTS = (
    "tools/catalog_migrations/kind_v2_decisions.json",
    "tools/issue_catalog_kind_v2.json",
    "tools/catalog_migrations/2.1_to_3.0.json",
    "tools/catalog_migrations/2.1_to_3.0_audit.md",
    "tools/issue_catalog.json",
)


def prompt_identity(root: Path) -> Dict[str, Any]:
    """Prompt versions and shas, read as source text so no arm's modules need importing twice."""
    out: Dict[str, Any] = {}
    passes = (root / "tools" / "scene_classifier_passes.py").read_text(encoding="utf-8")
    for name in ("PASS_2B_PROMPT_VERSION", "PASS_2B_PROMPT_SHA256", "PASS_2C_PROMPT_VERSION",
                 "PASS_2C_PROMPT_SHA256", "PASS_2D_PROMPT_VERSION", "PASS_2D_PROMPT_SHA256",
                 "PASS_2F_PROMPT_VERSION"):
        m = re.search(rf'^{name}\s*=\s*["\']([^"\']+)["\']', passes, re.M)
        if m:
            out[name] = m.group(1)
    contracts = (root / "tools" / "renovation_architecture" / "contracts.py").read_text(encoding="utf-8")
    for name in ("TERRA_REVIEW_PROMPT_VERSION", "SOL_REVIEW_PROMPT_VERSION",
                 "TERRA_REVIEW_REASONING_EFFORT", "SOL_REVIEW_REASONING_EFFORT",
                 "REQUIRED_CATALOG_VERSION", "REQUIRED_CATALOG_ONTOLOGY"):
        m = re.search(rf'^{name}\s*=\s*["\']([^"\']+)["\']', contracts, re.M)
        if m:
            out[name] = m.group(1)
    # git blob, not the on-disk hash: 92 tracked files are checked out LF in the main
    # checkout and CRLF in the worktree, so an on-disk comparison reports line endings
    # rather than content. Blob identity is what "same code in both arms" means here.
    for label, relpath in (("terra_review_module", "tools/renovation_architecture/terra_review.py"),
                           ("sol_review_module", "tools/renovation_architecture/sol_review.py")):
        path = root / relpath
        if path.is_file():
            out[f"{label}_blob"] = git_blob(root, path)
    return out


def arm_record(root: Path, *, label: str, expected_commit: Optional[str] = None) -> Dict[str, Any]:
    """Everything that identifies one arm: commit, file hashes, catalog identity, runtime code."""
    root = Path(root).resolve()
    head = git(root, "rev-parse", "HEAD")
    if expected_commit and head != expected_commit:
        fail(f"{label} arm HEAD {head} != expected {expected_commit}")
    catalog_path = root / "tools" / "issue_catalog_kind_v2.json"
    catalog = read_json(catalog_path)
    files: Dict[str, Dict[str, Optional[str]]] = {}
    for relpath in CATALOG_ARTIFACTS + RUNTIME_MODULES + ("tools/catalog_validation.py",):
        path = root / relpath
        if path.is_file():
            files[relpath] = {"sha256_on_disk": sha256_file(path), "git_blob": git_blob(root, path)}
    kinds: Dict[str, int] = {}
    for item in catalog["items"]:
        kinds[item.get("kind", "?")] = kinds.get(item.get("kind", "?"), 0) + 1
    return {
        "label": label,
        "root": str(root),
        "git": {
            "head": head,
            "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "porcelain_tracked": git(root, "status", "--porcelain", "-uno"),
        },
        "catalog": {
            "path": rel(catalog_path, root),
            "sha256": sha256_file(catalog_path),
            "version": catalog.get("version"),
            "ontology_version": catalog.get("ontology_version"),
            "publication_status": catalog.get("publication_status"),
            "item_count": len(catalog["items"]),
            "kind_counts": dict(sorted(kinds.items())),
        },
        "files": files,
        "prompts": prompt_identity(root),
    }


def environment_record() -> Dict[str, Any]:
    keys = ("KIND_ONTOLOGY_VERSION", "ISSUE_CATALOG_PATH", "EMBEDDINGS_BACKEND", "EMBEDDINGS_BASE_URL",
            "EMBEDDINGS_MODEL_NAME", "PASS_2D_SHORTCUT_MIN_SCORE", "PASS_2D_SHORTCUT_MIN_MARGIN",
            "RENOVATION_ARCHITECTURE_MODE")
    req = MAIN_ROOT / "requirements.txt"
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "requirements_sha256": sha256_file(req) if req.is_file() else None,
        "env_set": {k: os.environ.get(k) for k in keys if k in os.environ},
        "effective": {
            "KIND_ONTOLOGY_VERSION": cfg.KIND_ONTOLOGY_VERSION,
            "ISSUE_CATALOG_PATH": str(cfg.ISSUE_CATALOG_PATH),
            "PIPELINE_MODE": cfg.PIPELINE_MODE,
            "EMBEDDINGS_BACKEND": cfg.EMBEDDINGS_BACKEND,
            "EMBEDDINGS_BASE_URL": cfg.EMBEDDINGS_BASE_URL,
            "EMBEDDINGS_MODEL_NAME": cfg.EMBEDDINGS_MODEL_NAME,
            "EMBEDDINGS_DIMENSION": cfg.EMBEDDINGS_DIMENSION,
            "PASS_2D_SHORTCUT_MIN_SCORE": cfg.PASS_2D_SHORTCUT_MIN_SCORE,
            "PASS_2D_SHORTCUT_MIN_MARGIN": cfg.PASS_2D_SHORTCUT_MIN_MARGIN,
        },
        "script_sha256": sha256_file(SCRIPT_PATH),
        "arm_root": str(ARM_ROOT),
    }


# --------------------------------------------------------------------------- tier 1

def _serialise_decisions(obj: Any) -> bytes:
    """The generator's own serialisation, CRLF, as the program pins it."""
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    return text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")


def _load_renderer():
    """The Session 2 renderer, imported from the BASELINE arm only (its pins are baseline pins)."""
    if str(MAIN_ROOT) not in sys.path:
        sys.path.insert(0, str(MAIN_ROOT))
    import importlib
    return importlib.import_module("scripts.render_catalog_audit_proposals")


def tier1(candidate_root: Path, out_dir: Path) -> Dict[str, Any]:
    assert_no_provider_imports()
    checks: List[Dict[str, Any]] = []

    def check(cid: str, name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"id": cid, "name": name, "result": "pass" if ok else "FAIL", "detail": detail})

    candidate_root = Path(candidate_root).resolve()
    baseline = arm_record(MAIN_ROOT, label="baseline", expected_commit=BASELINE_COMMIT)
    candidate = arm_record(candidate_root, label="candidate", expected_commit=CANDIDATE_COMMIT)

    # 1.1 arm identity ------------------------------------------------------
    check("1.1a", "baseline arm at 9afe0fa, tracked-clean",
          baseline["git"]["head"] == BASELINE_COMMIT and baseline["git"]["porcelain_tracked"] == "",
          {"head": baseline["git"]["head"], "porcelain": baseline["git"]["porcelain_tracked"]})
    check("1.1b", "candidate arm at 6e67eaa, tracked-clean",
          candidate["git"]["head"] == CANDIDATE_COMMIT and candidate["git"]["porcelain_tracked"] == "",
          {"head": candidate["git"]["head"], "porcelain": candidate["git"]["porcelain_tracked"]})
    expected_hashes = PINS["expected_hashes"]
    pin_problems = []
    for arm, rec in (("baseline", baseline), ("candidate", candidate)):
        for relpath, want in expected_hashes[arm].items():
            got = rec["files"].get(relpath, {}).get("sha256_on_disk")
            if got != want:
                pin_problems.append(f"{arm}:{relpath} {got} != {want}")
    check("1.1c", "handoff hash table reproduced on disk (CRLF)", not pin_problems, pin_problems)

    shared_code = {r: (baseline["files"].get(r, {}).get("git_blob"),
                       candidate["files"].get(r, {}).get("git_blob")) for r in RUNTIME_MODULES}
    code_drift = {r: v for r, v in shared_code.items() if v[0] != v[1]}
    check("1.1d", "runtime code and generator identical across arms (git blob)", not code_drift, code_drift)
    ondisk_drift = {r: [baseline["files"].get(r, {}).get("sha256_on_disk"),
                        candidate["files"].get(r, {}).get("sha256_on_disk")]
                    for r in RUNTIME_MODULES
                    if baseline["files"].get(r, {}).get("sha256_on_disk")
                    != candidate["files"].get(r, {}).get("sha256_on_disk")}
    check("1.1d2", "runtime modules also byte-identical on disk (no line-ending split in the replay path)",
          not ondisk_drift, ondisk_drift)
    check("1.1e", "prompt versions and shas identical across arms",
          baseline["prompts"] == candidate["prompts"],
          {"baseline": baseline["prompts"], "candidate": candidate["prompts"]})

    diff_files = sorted(git(candidate_root, "diff", "--name-only", BASELINE_COMMIT, CANDIDATE_COMMIT).splitlines())
    expected_diff = sorted(PINS["expected_diff_files"])
    check("1.1f", "candidate commit touches exactly the " + str(len(expected_diff)) + " expected files",
          diff_files == expected_diff, {"actual": diff_files, "expected": expected_diff})

    # Arm-isolation inventory: every tracked file whose bytes differ between the arms and
    # is not part of the approved diff. All of these are line-ending-only (git blobs equal);
    # the check is that content never differs, and the count is reported so a future session
    # can tell a fresh CRLF checkout from real drift.
    tracked = git(MAIN_ROOT, "ls-tree", "-r", "--name-only", BASELINE_COMMIT).splitlines()
    approved = set(expected_diff)
    content_drift, ending_only = [], []
    for relpath in tracked:
        if relpath in approved:
            continue
        a, b = MAIN_ROOT / relpath, candidate_root / relpath
        if not (a.is_file() and b.is_file()):
            continue
        if sha256_file(a) == sha256_file(b):
            continue
        if git_blob(MAIN_ROOT, a) == git_blob(candidate_root, b):
            ending_only.append(relpath)
        else:
            content_drift.append(relpath)
    check("1.1i", "no tracked file outside the approved diff differs in content between the arms",
          not content_drift, {"content_drift": content_drift,
                              "line_ending_only_count": len(ending_only),
                              "line_ending_only_sample": ending_only[:5]})


    # approvals manifest pins ----------------------------------------------
    approvals_path = MAIN_ROOT / PINS["approvals_relpath"]
    approvals_sha = sha256_file(approvals_path)
    approvals = read_json(approvals_path)
    against = approvals["approved_against"]
    pin_checks = {
        "approvals_sha256": (approvals_sha, APPROVALS_SHA256),
        "git_head_at_review": (against["git_head_at_review"], BASELINE_COMMIT),
        "decisions_sha256": (baseline["files"]["tools/catalog_migrations/kind_v2_decisions.json"]["sha256_on_disk"],
                             against["decisions_sha256"]),
        "generated_catalog_sha256": (baseline["files"]["tools/issue_catalog_kind_v2.json"]["sha256_on_disk"],
                                     against["generated_catalog_sha256"]),
        "generator_sha256": (baseline["files"]["scripts/migrate_catalog_kind_v2.py"]["sha256_on_disk"],
                             against["generator_sha256"]),
        "validator_sha256": (baseline["files"]["tools/catalog_validation.py"]["sha256_on_disk"],
                             against["validator_sha256"]),
        "evidence_sha256": (sha256_file(MAIN_ROOT / "reports" / "catalog_audit_evidence.json"),
                            against["evidence_sha256"]),
        "proposals_sha256": (sha256_file(MAIN_ROOT / "reports" / "catalog_audit_proposals.json"),
                             against["proposals_sha256"]),
        "adversarial_review_sha256": (sha256_file(MAIN_ROOT / "reports" / "catalog_audit_adversarial_review.json"),
                                      against["adversarial_review_sha256"]),
        "redraft_sha256": (sha256_file(MAIN_ROOT / "reports" / "catalog_audit_redraft.json"),
                           against["redraft_sha256"]),
        "renewed_review_sha256": (sha256_file(MAIN_ROOT / "reports" / "catalog_audit_renewed_review.json"),
                                  against["renewed_review_sha256"]),
        "gate_decision_record_sha256": (sha256_file(MAIN_ROOT / "reports" / "catalog_audit_gate_decisions.json"),
                                        against["gate_decision_record_sha256"]),
        "live_check_results_sha256": (sha256_file(MAIN_ROOT / "reports" / "catalog_audit_live_check_results.json"),
                                      against["live_check_results_sha256"]),
    }
    pin_bad = {k: v for k, v in pin_checks.items() if v[0] != v[1]}
    check("1.1g", "every approvals `approved_against` pin verified", not pin_bad, pin_bad)

    cap007 = next(d for d in approvals["dispositions"] if d["proposal_id"] == APPROVED_PROPOSAL_ID)
    ops = cap007["approved_diff"]["ops"]
    others = [d for d in approvals["dispositions"] if d["proposal_id"] != APPROVED_PROPOSAL_ID]
    check("1.1h", "only " + APPROVED_PROPOSAL_ID + " carries an approved diff",
          all(d["approved_diff"] is None for d in others) and len(ops) == PINS["approved_op_count"],
          {"op_count": len(ops), "expected_op_count": PINS["approved_op_count"],
           "other_records": len(others)})

    return {"checks": checks, "baseline": baseline, "candidate": candidate,
            "arm_isolation": {"line_ending_only_files": sorted(ending_only),
                              "content_drift_files": sorted(content_drift)},
            "approvals": {"path": rel(approvals_path), "sha256": approvals_sha,
                          "ops": ops, "cap007": cap007}}


def tier1_reconcile(candidate_root: Path, ops: Sequence[Dict[str, Any]],
                    cap007: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Approved-vs-actual, validators, economics and routes.

    Runs from the baseline arm's code throughout: the renderer's loaders pin baseline hashes
    and abort in the worktree, and using one code path over two data sets keeps the only
    variable the catalog itself. Candidate files are read by explicit path.
    """
    checks: List[Dict[str, Any]] = []
    detail_out: Dict[str, Any] = {}

    def check(cid: str, name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"id": cid, "name": name, "result": "pass" if ok else "FAIL", "detail": detail})

    s2 = _load_renderer()
    from tools.catalog_validation import ECONOMIC_FIELDS, validate_issue_catalog, validate_migration_manifest
    from tools.renovation_architecture.catalog_projection import build_renovation_catalog_projection

    # 1.3 baseline reproduction (in memory; writes nothing)
    bundle = s2.load_bundle()
    base = s2.load_baseline(bundle)
    check("1.3", "baseline regenerates byte-identically in memory (load_baseline)", True,
          {"items": len(base["catalog"]["items"])})

    cand_catalog_path = candidate_root / "tools" / "issue_catalog_kind_v2.json"
    cand_decisions_path = candidate_root / "tools" / "catalog_migrations" / "kind_v2_decisions.json"
    cand_manifest_path = candidate_root / "tools" / "catalog_migrations" / "2.1_to_3.0.json"
    cand_catalog = read_json(cand_catalog_path)
    cand_decisions = read_json(cand_decisions_path)
    cand_manifest = read_json(cand_manifest_path)

    # 1.5 approved-vs-actual
    patched = s2.apply_ops(base["decisions"], ops)
    check("1.5a", "apply_ops(baseline, approved ops) == candidate decisions (parsed)",
          patched == cand_decisions,
          None if patched == cand_decisions else "decision trees differ")
    check("1.5b", "and byte-identical under the generator serialisation (CRLF)",
          _serialise_decisions(patched) == cand_decisions_path.read_bytes(),
          {"regenerated_sha256": sha256_bytes(_serialise_decisions(patched)),
           "on_disk_sha256": sha256_file(cand_decisions_path)})

    gen_catalog, gen_manifest = base["gen"].generate(copy.deepcopy(base["v1"]), copy.deepcopy(patched))
    check("1.5c", "generate(v1, patched) == candidate catalog", gen_catalog == cand_catalog)
    check("1.5d", "generate(v1, patched) == candidate migration manifest", gen_manifest == cand_manifest)

    diff = s2.catalog_diff(base["catalog"], gen_catalog)
    expected = cap007["approved_diff"]["expected_generated_changes"]
    actual_diff = {"added_items": diff["added"], "removed_items": diff["removed"],
                   "item_diff": diff["item_diff"], "order_ok": diff["order_ok"]}
    expected_cmp = {k: expected[k] for k in ("added_items", "removed_items", "item_diff", "order_ok")}
    check("1.5e", "generated diff equals manifest expected_generated_changes",
          actual_diff == expected_cmp, {"actual": actual_diff, "expected": expected_cmp})
    detail_out["catalog_diff"] = actual_diff

    surface = bundle["authoring_surface"]
    dry = s2.dry_run(base, ops, surface)
    approved_dry = cap007["approved_diff"]["dry_run_at_approval"]
    dry_cmp = {
        "ok": dry["ok"],
        "classification": [[c["path"], c["class"]] for c in dry["classification"]],
        "metadata_problems": dry["metadata_problems"],
        "catalog_validation_errors": dry["catalog_validation"]["errors"],
        "manifest_validation_errors": dry["manifest_validation"]["errors"],
        "added_detail": dry["added_detail"],
    }
    approved_cmp = {
        "ok": approved_dry["ok"],
        "classification": [list(c) for c in approved_dry["classification"]],
        "metadata_problems": approved_dry["metadata_problems"],
        "catalog_validation_errors": approved_dry["catalog_validation_errors"],
        "manifest_validation_errors": approved_dry["manifest_validation_errors"],
        "added_detail": approved_dry["added_detail"],
    }
    check("1.5f", "dry_run reproduces dry_run_at_approval exactly",
          dry_cmp == approved_cmp, {"actual": dry_cmp} if dry_cmp != approved_cmp else None)
    check("1.5g", "manifest diff keys equal approved manifest_diff_keys",
          sorted(dry.get("manifest_diff", {})) == sorted(expected["manifest_diff_keys"]),
          {"actual": sorted(dry.get("manifest_diff", {}))})

    cand_items = {it["id"]: it for it in cand_catalog["items"]}
    authored_problems = []
    for op in ops:
        if not op["path"].startswith("/successors/"):
            continue
        spec = op["after"]
        item = cand_items.get(spec["id"])
        if item is None:
            authored_problems.append(f"{spec['id']} missing from the candidate catalog")
            continue
        for field in ("kind", "severity", "name", "description", "embed_text", "support_any", "atomic_claim"):
            if field in spec and item.get(field) != spec[field]:
                authored_problems.append(f"{spec['id']}.{field} differs from the approved op")
        for field, value in (spec.get("overrides") or {}).items():
            if item.get(field) != value:
                authored_problems.append(f"{spec['id']}.{field} override differs from the approved op")
    check("1.5h", "both successors authored surfaces landed verbatim", not authored_problems, authored_problems)

    b_list = base["decisions"]["decisions"] if "decisions" in base["decisions"] else base["decisions"]["entries"]
    c_list = cand_decisions["decisions"] if "decisions" in cand_decisions else cand_decisions["entries"]
    unrelated_bad = [b["legacy_id"] for b, c in zip(b_list, c_list)
                     if b["legacy_id"] != PARENT_ID and b != c]
    check("1.5i", "all unrelated decision entries byte-identical, order and count unchanged",
          not unrelated_bad and len(b_list) == len(c_list) == 107
          and [e["legacy_id"] for e in b_list] == [e["legacy_id"] for e in c_list],
          {"changed": unrelated_bad, "counts": [len(b_list), len(c_list)]})

    # 1.4 explicit v2 validation
    v_base = validate_issue_catalog(base["catalog"])
    v_cand = validate_issue_catalog(cand_catalog)
    m_base = validate_migration_manifest(base["manifest"], base["v1"], base["catalog"])
    m_cand = validate_migration_manifest(cand_manifest, base["v1"], cand_catalog)
    check("1.4a", "validate_issue_catalog: zero errors in both arms",
          not v_base.errors and not v_cand.errors,
          {"baseline": list(v_base.errors), "candidate": list(v_cand.errors)})
    check("1.4b", "validate_migration_manifest: zero errors in both arms",
          not m_base.errors and not m_cand.errors,
          {"baseline": list(m_base.errors), "candidate": list(m_cand.errors)})
    check("1.4c", "validator warning sets identical across arms",
          sorted(v_base.warnings) == sorted(v_cand.warnings),
          {"only_baseline": sorted(set(v_base.warnings) - set(v_cand.warnings)),
           "only_candidate": sorted(set(v_cand.warnings) - set(v_base.warnings))})

    # 1.6 economics, routes, packages
    base_items = {it["id"]: it for it in base["catalog"]["items"]}
    shared = sorted(set(base_items) & set(cand_items))
    econ_bad = {i: {f: [base_items[i].get(f), cand_items[i].get(f)] for f in ECONOMIC_FIELDS
                    if base_items[i].get(f) != cand_items[i].get(f)}
                for i in shared if any(base_items[i].get(f) != cand_items[i].get(f) for f in ECONOMIC_FIELDS)}
    check("1.6a", f"economic fields identical on all {len(shared)} shared items", not econ_bad, econ_bad)

    added_detail = cap007["approved_diff"]["dry_run_at_approval"]["added_detail"]
    succ_bad = {}
    for item_id, det in added_detail.items():
        item = cand_items[item_id]
        got = {f: item.get(f) for f in ECONOMIC_FIELDS}
        if got != det["economics"] or item.get("pricing_status") != det["pricing_status"]:
            succ_bad[item_id] = {"economics": got, "pricing_status": item.get("pricing_status")}
    check("1.6b", "successors carry the approved inherited economics verbatim", not succ_bad, succ_bad)

    proj_base = build_renovation_catalog_projection(
        base["catalog"], catalog_path=MAIN_ROOT / "tools" / "issue_catalog_kind_v2.json")
    proj_cand = build_renovation_catalog_projection(cand_catalog, catalog_path=cand_catalog_path)
    routes_base, routes_cand = proj_base["terminal_routes"], proj_cand["terminal_routes"]
    route_changes = {i: {"before": routes_base.get(i), "after": routes_cand.get(i)}
                     for i in sorted(set(routes_base) | set(routes_cand))
                     if routes_base.get(i) != routes_cand.get(i)}
    expected_routes = {
        PARENT_ID: {"before": {"route": "work", "reason_code": "work_default"}, "after": None},
        BLINDS_ID: {"before": None, "after": {"route": "no_action", "reason_code": "route_override_no_action"}},
        FABRIC_ID: {"before": None, "after": {"route": "work", "reason_code": "work_default"}},
    }
    check("1.6c", "terminal routes change only for the parent and its two successors",
          route_changes == expected_routes, {"actual": route_changes})
    check("1.6d", "route counts move 128 to 129 with no_action gaining exactly one",
          proj_cand["route_counts"].get("no_action") == proj_base["route_counts"].get("no_action", 0) + 1
          and sum(proj_cand["route_counts"].values()) == sum(proj_base["route_counts"].values()) + 1,
          {"baseline": proj_base["route_counts"], "candidate": proj_cand["route_counts"]})
    wp_base, wp_cand = proj_base["work_policy"], proj_cand["work_policy"]
    wp_changes = {i: {"before": wp_base.get(i), "after": wp_cand.get(i)}
                  for i in sorted(set(wp_base) | set(wp_cand)) if wp_base.get(i) != wp_cand.get(i)}
    check("1.6e", "work policy: parent replaced by the fabric successor on identical terms, blinds absent",
          sorted(wp_changes) == sorted([PARENT_ID, FABRIC_ID])
          and wp_changes.get(PARENT_ID, {}).get("before") == wp_changes.get(FABRIC_ID, {}).get("after")
          and BLINDS_ID not in wp_cand,
          {"changed": sorted(wp_changes), "blinds_in_work_policy": BLINDS_ID in wp_cand,
           "terms_match": wp_changes.get(PARENT_ID, {}).get("before") == wp_changes.get(FABRIC_ID, {}).get("after")})
    fr_base = proj_base["package_policy"]["flat_roles"]
    fr_cand = proj_cand["package_policy"]["flat_roles"]
    check("1.6f", "flat package roles gain exactly one; both successors inherit 'ignore'",
          len(fr_cand) == len(fr_base) + 1 and fr_cand.get(BLINDS_ID) == "ignore"
          and fr_cand.get(FABRIC_ID) == "ignore",
          {"counts": [len(fr_base), len(fr_cand)]})
    detail_out["projection"] = {
        "fingerprint": {"baseline": proj_base["fingerprint"], "candidate": proj_cand["fingerprint"]},
        "catalog_sha256": {"baseline": proj_base["catalog_sha256"], "candidate": proj_cand["catalog_sha256"]},
        "route_counts": {"baseline": proj_base["route_counts"], "candidate": proj_cand["route_counts"]},
        "catalog_version": {"baseline": proj_base["catalog_version"], "candidate": proj_cand["catalog_version"]},
    }
    check("1.6g", "projection fingerprint changes (CCF-8: no stored Terra checkpoint survives)",
          proj_base["fingerprint"] != proj_cand["fingerprint"], detail_out["projection"]["fingerprint"])
    check("1.6h", "catalog version string is unchanged, so live runs must be identified by catalog_sha256",
          proj_base["catalog_version"] == proj_cand["catalog_version"] == "3.2",
          detail_out["projection"]["catalog_version"])

    # 1.7 identity and order
    base_order = [it["id"] for it in base["catalog"]["items"]]
    cand_order = [it["id"] for it in cand_catalog["items"]]
    parent_index = base_order.index(PARENT_ID)
    check("1.7a", "128 to 129 items; the successors occupy the parent position",
          len(base_order) == 128 and len(cand_order) == 129
          and cand_order[parent_index:parent_index + 2] == [BLINDS_ID, FABRIC_ID],
          {"parent_index": parent_index, "at_index": cand_order[parent_index:parent_index + 2]})
    check("1.7b", "every other item keeps its relative order",
          [i for i in cand_order if i in set(base_order)] == [i for i in base_order if i in set(cand_order)])

    detail_out["dry_run"] = dry_cmp
    detail_out["successor_economics"] = {i: {f: cand_items[i].get(f) for f in ECONOMIC_FIELDS}
                                         for i in (BLINDS_ID, FABRIC_ID)}
    return checks, detail_out


def regeneration_parity(candidate_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """1.2 double generation in the candidate worktree, then prove it left the tree untouched.

    The generator is the only thing this session runs inside the worktree, and only because
    byte parity has to be demonstrated against the arm that actually ships the files.
    """
    checks: List[Dict[str, Any]] = []
    targets = ["tools/issue_catalog_kind_v2.json",
               "tools/catalog_migrations/2.1_to_3.0.json",
               "tools/catalog_migrations/2.1_to_3.0_audit.md"]
    before = {t: sha256_file(candidate_root / t) for t in targets}
    runs = []
    for n in (1, 2):
        proc = subprocess.run([sys.executable, str(candidate_root / "scripts" / "migrate_catalog_kind_v2.py")],
                              cwd=str(candidate_root), capture_output=True, text=True)
        if proc.returncode != 0:
            fail(f"generator run {n} failed in the candidate arm: {proc.stderr.strip()[-800:]}")
        runs.append({"run": n, "stdout_tail": proc.stdout.strip().splitlines()[-3:],
                     "hashes": {t: sha256_file(candidate_root / t) for t in targets}})
    porcelain = git(candidate_root, "status", "--porcelain", "-uno")
    checks.append({"id": "1.2a", "name": "double generation is byte-identical",
                   "result": "pass" if runs[0]["hashes"] == runs[1]["hashes"] else "FAIL",
                   "detail": {"run1": runs[0]["hashes"], "run2": runs[1]["hashes"]}})
    checks.append({"id": "1.2b", "name": "regeneration reproduces the committed artifacts exactly",
                   "result": "pass" if runs[1]["hashes"] == before else "FAIL",
                   "detail": {"before": before, "after": runs[1]["hashes"]}})
    checks.append({"id": "1.2c", "name": "candidate worktree still tracked-clean after regeneration",
                   "result": "pass" if porcelain == "" else "FAIL", "detail": porcelain})
    return checks, {"runs": runs, "before": before}


def cmd_tier1(args: argparse.Namespace) -> int:
    candidate_root = Path(args.candidate_root).resolve()
    out_path = Path(args.out).resolve()
    identity = tier1(candidate_root, out_path.parent)
    checks = list(identity["checks"])
    regen_checks, regen_detail = regeneration_parity(candidate_root)
    checks.extend(regen_checks)
    recon_checks, recon_detail = tier1_reconcile(candidate_root, identity["approvals"]["ops"],
                                                 identity["approvals"]["cap007"])
    checks.extend(recon_checks)
    failures = [c for c in checks if c["result"] != "pass"]
    payload = {
        "schema_version": "catalog-audit-tier1-v1",
        "program": "catalog_audit",
        "session": 5,
        "tier": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_record(),
        "arms": {"baseline": identity["baseline"], "candidate": identity["candidate"]},
        "approvals": {"path": identity["approvals"]["path"], "sha256": identity["approvals"]["sha256"],
                      "proposal_id": APPROVED_PROPOSAL_ID, "op_count": len(identity["approvals"]["ops"])},
        "checks": checks,
        "evidence": {**recon_detail, "regeneration": regen_detail,
                     "arm_isolation": identity["arm_isolation"]},
        "result": "pass" if not failures else "FAIL",
        "failures": [c["id"] for c in failures],
    }
    sha = write_json(out_path, payload)
    print(f"tier1: {payload['result']}  ({len(checks) - len(failures)}/{len(checks)} checks passed)")
    for c in failures:
        print(f"  FAIL {c['id']} {c['name']}")
    print(f"wrote {rel(out_path)} sha256={sha}")
    return 0 if not failures else 1


# --------------------------------------------------------------------------- tier 2: corpus

# Stems that can reach either successor's gate, plus the neighbouring subjects the split
# competes with. Deliberately wider than the successors' own term lists: the point is to
# catch rows the change could touch, including ones the gates then reject.
REPLAY_STEMS = ("blind", "shade", "curtain", "drape", "drap", "fabric", "valance", "cornice",
                "swag", "sheer", "window treatment", "window covering", "window")

BLINDS_STEMS = ("blind", "roller shade", "window shade")
FABRIC_BARE_STEMS = ("curtain", "drape", "fabric")


def _artifact_rows(art: Dict[str, Any], *, source: str, property_key: str, run_id: str,
                   artifact_path: str, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every Pass 2d row in one stored artifact, with the scene group its photo carries.

    The retrieval kind is `original_kind`: `resolved_kind` is the post-resolution value and
    the row has no `kind` key at all, so reading the wrong one silently produces
    `invalid_kind` routing and an empty replay.
    """
    rows: List[Dict[str, Any]] = []
    issue_refs = conditions.get("issue_refs") or {}
    cond_index = conditions.get("conditions") or {}
    for photo_key, photo in sorted((art.get("photos") or {}).items()):
        if not isinstance(photo, dict):
            continue
        scene_group = str(((photo.get("scene") or {}).get("group")) or "").strip().lower() or "unknown"
        for rec in (photo.get("debug") or {}).get("resolved_items") or []:
            if not isinstance(rec, dict):
                continue
            issue_id = rec.get("issue_id")
            ref = issue_refs.get(issue_id) or {}
            condition_id = ref.get("condition_id")
            cond = cond_index.get(condition_id) or {}
            rows.append({
                "row_id": f"{source}:{property_key}:{run_id}:{photo_key}:{issue_id}",
                "source": source, "property_key": property_key, "run_id": run_id,
                "artifact_path": artifact_path, "photo_key": photo_key, "issue_id": issue_id,
                "observation": rec.get("description") or "",
                "kind": rec.get("original_kind"),
                "resolved_kind": rec.get("resolved_kind"),
                "scene_group": scene_group,
                "frozen": {
                    "resolved_item_id": rec.get("resolved_item_id"),
                    "resolution_path": rec.get("resolution_path"),
                    "routing_reason": rec.get("routing_reason"),
                    "shortcut_reason": rec.get("shortcut_reason"),
                    "candidates": [{"rank": n + 1, "item_id": c.get("item_id"), "score": c.get("score")}
                                   for n, c in enumerate(rec.get("candidates") or []) if isinstance(c, dict)],
                },
                "condition_id": condition_id,
                "condition_item_id": cond.get("catalog_item_id"),
                "condition_scene_group": cond.get("scene_group"),
            })
    return rows


def _frozen_shortcut_selftest(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Replay the frozen candidate lists through today's shortcut function.

    If this disagrees with what the artifacts recorded, the thresholds or the negation
    patterns have moved since the evidence era and nothing downstream is comparable.
    """
    from tools.scene_classifier_passes import _resolve_candidate_via_lexical_shortcut
    mismatches, considered = [], 0
    for row in rows:
        cands = row["frozen"]["candidates"]
        if not cands or not row["kind"]:
            continue
        considered += 1
        # the stored rows carry only id/score; the shortcut needs the full candidate dicts
        rid, _kind, reason = _resolve_candidate_via_lexical_shortcut(
            row["observation"], row["_frozen_candidate_dicts"], kind=row["kind"])
        fired = rid is not None
        was = row["frozen"]["resolution_path"] == "lexical_shortcut"
        if fired != was or (fired and reason != row["frozen"]["shortcut_reason"]) or (
                fired and rid != row["frozen"]["resolved_item_id"]):
            mismatches.append({"row_id": row["row_id"], "replayed": [rid, reason],
                               "frozen": [row["frozen"]["resolved_item_id"], row["frozen"]["shortcut_reason"],
                                          row["frozen"]["resolution_path"]]})
    return {"rows_considered": considered, "mismatches": mismatches,
            "result": "pass" if not mismatches else "FAIL"}


def cmd_corpus(args: argparse.Namespace) -> int:
    assert_no_provider_imports()
    if ARM_ROOT != MAIN_ROOT:
        fail("corpus must be built from the baseline arm (the evidence bundle pins baseline hashes)")
    import importlib
    if str(MAIN_ROOT) not in sys.path:
        sys.path.insert(0, str(MAIN_ROOT))
    builder = importlib.import_module("scripts.build_catalog_audit_evidence")

    evidence_path = MAIN_ROOT / "reports" / "catalog_audit_evidence.json"
    evidence = read_json(evidence_path)
    redraft = read_json(MAIN_ROOT / "reports" / "catalog_audit_redraft.json")
    review_queue = read_json(MAIN_ROOT / "reports" / "review_queue.json")

    artifacts, rows, problems = [], [], []
    for entry in evidence["run_artifacts"]:
        path = entry.get("artifact_path")
        if not path:
            artifacts.append({**{k: entry[k] for k in ("source", "property_key", "run_id", "status")},
                              "readable": False, "reason": "no artifact_path in the pinned bundle"})
            continue
        p = Path(path)
        actual = sha256_file(p)
        if actual != entry["current_sha256"]:
            problems.append(f"{path}: sha256 {actual} != pinned {entry['current_sha256']}")
            continue
        art = read_json(p)
        conds = builder.artifact_conditions(art)
        got = _artifact_rows(art, source=entry["source"], property_key=entry["property_key"],
                             run_id=entry["run_id"], artifact_path=path, conditions=conds)
        # keep the full candidate dicts for the shortcut self-test, out of the written corpus
        for row in got:
            photo = (art.get("photos") or {}).get(row["photo_key"]) or {}
            for rec in (photo.get("debug") or {}).get("resolved_items") or []:
                if isinstance(rec, dict) and rec.get("issue_id") == row["issue_id"]:
                    row["_frozen_candidate_dicts"] = [c for c in (rec.get("candidates") or [])
                                                      if isinstance(c, dict)]
                    break
            row.setdefault("_frozen_candidate_dicts", [])
        rows.extend(got)
        artifacts.append({"source": entry["source"], "property_key": entry["property_key"],
                          "run_id": entry["run_id"], "status": entry["status"], "readable": True,
                          "artifact_path": path, "sha256": actual,
                          "catalog_version": entry.get("catalog_version"),
                          "catalog_sha256": entry.get("catalog_sha256"), "rows": len(got)})
    if problems:
        fail("pinned run artifacts drifted: " + "; ".join(problems))

    selftest = _frozen_shortcut_selftest(rows)

    # --- selection --------------------------------------------------------
    parent_rows = [r for r in rows if r["frozen"]["resolved_item_id"] == PARENT_ID]
    parent_candidate_rows = [r for r in rows
                             if any(c["item_id"] == PARENT_ID for c in r["frozen"]["candidates"])]
    stem_rows = [r for r in rows
                 if any(s in (r["observation"] or "").lower() for s in REPLAY_STEMS)]
    modernization_rows = [r for r in rows if r["kind"] == "modernization"]
    selected_ids = {r["row_id"] for r in parent_rows + parent_candidate_rows + stem_rows + modernization_rows}

    # named cases: review cards give property/run/condition, the condition joins back to rows
    cards = {c["card_id"]: c for c in review_queue["cards"]}
    named: Dict[str, Any] = {}
    for card_id in ("rc_7c5b9f9bb3aa", "rc_429ba6851d09", "rc_4f4b93c40ef3", "rc_ffad088fa5fe",
                    "rc_64f996e3a886", "rc_a22a241b3bf0", "rc_8afb476fa69f"):
        card = cards.get(card_id)
        if card is None:
            named[card_id] = {"status": "card_not_found"}
            continue
        meta = card.get("meta") or {}
        cond_id = meta.get("condition_id")
        hits = [r["row_id"] for r in rows if r["condition_id"] == cond_id]
        named[card_id] = {"condition_id": cond_id, "property_key": meta.get("property_key"),
                          "run_id": meta.get("run_id"), "catalog_item_id": meta.get("catalog_item_id"),
                          "scene_group": meta.get("scene_group"), "row_ids": hits,
                          "status": "resolved" if hits else "no_row"}
        selected_ids.update(hits)

    census = ((redraft.get("cap007") or {}).get("withdrawn_billings_census") or {})
    for key in ("condition_ids_moving", "condition_ids_surviving", "condition_ids_neither"):
        for cond_id in census.get(key, []):
            hits = [r["row_id"] for r in rows if r["condition_id"] == cond_id]
            selected_ids.update(hits)
            named.setdefault(cond_id, {"condition_id": cond_id, "census_bucket": key,
                                       "row_ids": hits, "status": "resolved" if hits else "no_row"})

    selected = [r for r in rows if r["row_id"] in selected_ids]
    for row in selected:
        row.pop("_frozen_candidate_dicts", None)
    for row in rows:
        row.pop("_frozen_candidate_dicts", None)

    # --- census reconciliation from the artifacts themselves ---------------
    parent_conditions = sorted({r["condition_id"] for r in rows
                                if r["condition_item_id"] == PARENT_ID and r["condition_id"]})
    reconciliation = {
        "parent_conditions_in_artifacts": len(parent_conditions),
        "redraft_conditions_total": census.get("conditions_total"),
        "redraft_moving": census.get("moving_to_no_action"),
        "redraft_surviving": census.get("surviving_on_fabric_successor"),
        "redraft_neither": census.get("reaching_neither"),
        "census_ids_found_in_artifacts": {
            key: sum(1 for cid in census.get(key, []) if cid in set(parent_conditions))
            for key in ("condition_ids_moving", "condition_ids_surviving", "condition_ids_neither")},
    }

    payload = {
        "schema_version": "catalog-audit-replay-corpus-v1",
        "program": "catalog_audit", "session": 5,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence": {"path": rel(evidence_path), "sha256": sha256_file(evidence_path),
                     "fingerprint": evidence.get("fingerprint")},
        "script_sha256": sha256_file(SCRIPT_PATH),
        "artifacts": artifacts,
        "totals": {"artifacts_readable": sum(1 for a in artifacts if a["readable"]),
                   "rows_total": len(rows), "rows_selected": len(selected),
                   "rows_modernization": len(modernization_rows),
                   "rows_stem": len(stem_rows),
                   "rows_frozen_owner_is_parent": len(parent_rows),
                   "rows_parent_in_candidates": len(parent_candidate_rows)},
        "frozen_shortcut_selftest": selftest,
        "census_reconciliation": reconciliation,
        "named_cases": named,
        "rows": selected,
    }
    out = Path(args.out).resolve()
    sha = write_json(out, payload)
    print(f"corpus: {payload['totals']['rows_selected']} rows selected of {len(rows)} "
          f"({payload['totals']['artifacts_readable']} artifacts)")
    print(f"  frozen shortcut self-test: {selftest['result']} "
          f"({selftest['rows_considered']} rows, {len(selftest['mismatches'])} mismatches)")
    print(f"  parent conditions in artifacts: {reconciliation['parent_conditions_in_artifacts']} "
          f"(redraft says {reconciliation['redraft_conditions_total']})")
    unresolved = sorted(k for k, v in named.items() if v.get("status") != "resolved")
    if unresolved:
        print(f"  unresolved named cases: {len(unresolved)} -> {unresolved[:6]}")
    print(f"wrote {rel(out)} sha256={sha}")
    return 0 if selftest["result"] == "pass" else 1


# --------------------------------------------------------------------------- tier 2: sidecar

def probe_sidecar() -> Dict[str, Any]:
    """A real POST, because /health has returned 200 while every embedding call failed.

    Probes with the *configured* model name, so a server answering under a different alias
    is caught too. Records the server's own /props so the vector cache can refuse to reuse
    vectors produced by a different build or model.
    """
    import urllib.error
    import urllib.request

    base = str(cfg.EMBEDDINGS_BASE_URL).rstrip("/")
    record: Dict[str, Any] = {"base_url": base, "model": cfg.EMBEDDINGS_MODEL_NAME,
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
        record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        fail(f"embeddings sidecar POST failed: {record['error']}")
    record["latency_sec"] = round(time.time() - started, 3)
    vec = ((body.get("data") or [{}])[0]).get("embedding") or []
    record["dimension"] = len(vec)
    record["served_model"] = body.get("model")
    if len(vec) != int(cfg.EMBEDDINGS_DIMENSION):
        record["ok"] = False
        fail(f"embeddings sidecar returned dimension {len(vec)}, expected {cfg.EMBEDDINGS_DIMENSION}")
    props: Optional[Dict[str, Any]] = None
    try:
        with urllib.request.urlopen(f"{base.rsplit('/v1', 1)[0]}/props", timeout=30) as resp:
            props = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - informational only
        props = None
    if isinstance(props, dict):
        record["server"] = {k: props.get(k) for k in ("model_path", "build_info", "n_ctx") if k in props}
    record["ok"] = True
    record["probe_vector_sha256"] = sha256_canonical([round(float(x), 6) for x in vec])
    return record


class CachingEncoder:
    """Memoises embeddings by exact text so both arms score identical strings identically.

    The catalog is embedded in batches of 32, so inserting one item shifts every later
    item's batch position and can move scores by a few ULP. That is real production noise,
    but it is not the effect under test: the cache removes it as a confound and the noise
    probe measures it separately.
    """

    def __init__(self, inner: Any, cache_path: Path, header: Dict[str, Any]):
        import numpy as np
        self._np = np
        self._inner = inner
        self._path = Path(cache_path)
        self._header = header
        self.dimension = int(getattr(inner, "dimension", cfg.EMBEDDINGS_DIMENSION))
        self._store: Dict[str, List[float]] = {}
        self.stats = {"hits": 0, "misses": 0, "loaded": 0}
        if self._path.is_file():
            blob = read_json(self._path)
            if blob.get("header") == header:
                self._store = blob.get("vectors") or {}
                self.stats["loaded"] = len(self._store)
            else:
                fail(f"vector cache header mismatch at {self._path}: the sidecar or model changed")

    def encode(self, texts: Sequence[str]) -> Any:
        texts = list(texts)
        missing = [t for t in dict.fromkeys(texts) if t not in self._store]
        for start in range(0, len(missing), 32):
            batch = missing[start:start + 32]
            vecs = self._inner.encode(batch)
            for text, vec in zip(batch, vecs):
                self._store[text] = [float(x) for x in vec]
        self.stats["misses"] += len(missing)
        self.stats["hits"] += len(texts) - len(missing)
        return self._np.asarray([self._store[t] for t in texts], dtype="float32")

    def save(self) -> str:
        return write_json(self._path, {"header": self._header, "vectors": self._store})

    def sentinel_ok(self, sentinel: str) -> bool:
        """Re-embed a fixed sentence and require the cached vector back, to catch a
        server that answers under the same alias with different weights."""
        cached = self._store.get(sentinel)
        fresh = [float(x) for x in self._inner.encode([sentinel])[0]]
        if cached is None:
            self._store[sentinel] = fresh
            return True
        dot = sum(a * b for a, b in zip(cached, fresh))
        return dot >= 0.9999


# --------------------------------------------------------------------------- tier 2: snapshot

def _guardrail_trace(item: Dict[str, Any], text_lower: str) -> Dict[str, Any]:
    """Why one candidate would be dropped, in the retriever's order: deny first, then require."""
    from tools.pipeline_common import term_matches
    deny_hits = [t for t in (item.get("deny_any") or []) if term_matches(str(t).lower(), text_lower)]
    require = [str(t).lower() for t in (item.get("require_any") or [])]
    require_hits = [t for t in require if term_matches(t, text_lower)]
    if deny_hits:
        return {"dropped": True, "role": "deny_any", "terms": deny_hits}
    if require and not require_hits:
        return {"dropped": True, "role": "require_any", "terms": []}
    return {"dropped": False, "role": None, "terms": require_hits}


def cmd_snapshot(args: argparse.Namespace) -> int:
    assert_no_provider_imports()
    if cfg.KIND_ONTOLOGY_VERSION != "observation_kind_v2":
        fail("snapshot requires KIND_ONTOLOGY_VERSION=observation_kind_v2 in the environment "
             "(it is resolved at import, so it must be set before the interpreter starts)")
    if os.environ.get("ISSUE_CATALOG_PATH"):
        fail("ISSUE_CATALOG_PATH must not be set: under observation_kind_v2 the selector owns the catalog")
    catalog_path = Path(cfg.ISSUE_CATALOG_PATH).resolve()
    if catalog_path != (ARM_ROOT / "tools" / "issue_catalog_kind_v2.json"):
        fail(f"selector resolved {catalog_path}, expected the {args.arm} arm's own catalog")
    head = git(ARM_ROOT, "rev-parse", "HEAD")
    expected_head = BASELINE_COMMIT if args.arm == "baseline" else CANDIDATE_COMMIT
    if head != expected_head:
        fail(f"{args.arm} arm HEAD {head} != {expected_head}")
    if git(ARM_ROOT, "status", "--porcelain", "-uno") != "":
        fail(f"{args.arm} arm has tracked modifications; snapshots must come from a clean arm")

    from tools.catalog_embeddings import (CatalogEmbeddingsRetriever, build_guardrails_from_catalog,
                                          make_candidate_provider)
    from tools.scene_classifier_passes import (_analyze_visible_condition_signal,
                                               _resolve_candidate_via_lexical_shortcut,
                                               evaluate_kind_routing, is_generic_resolution_candidate)

    catalog = read_json(catalog_path)
    items = catalog["items"]
    probe = probe_sidecar()

    header = {"model": cfg.EMBEDDINGS_MODEL_NAME, "dimension": int(cfg.EMBEDDINGS_DIMENSION),
              "base_url": str(cfg.EMBEDDINGS_BASE_URL), "server": probe.get("server")}
    from tools.catalog_embeddings import OpenAICompatibleEncoder
    inner = OpenAICompatibleEncoder(model_name=cfg.EMBEDDINGS_MODEL_NAME,
                                    base_url=str(cfg.EMBEDDINGS_BASE_URL),
                                    dimension=int(cfg.EMBEDDINGS_DIMENSION))
    cache = CachingEncoder(inner, Path(args.cache), header)
    sentinel = "sentinel: the blinds are old."
    if not cache.sentinel_ok(sentinel):
        fail("vector cache sentinel mismatch: the sidecar is serving different weights than the cached run")
    cache_sha_before = sha256_canonical(sorted(cache._store)) if cache._store else None

    guardrails = build_guardrails_from_catalog(catalog)
    retriever = CatalogEmbeddingsRetriever(
        catalog, model_name=cfg.EMBEDDINGS_MODEL_NAME, device=cfg.EMBEDDINGS_DEVICE,
        default_topk=TOP_K, guardrails=guardrails, backend=cfg.EMBEDDINGS_BACKEND,
        base_url=str(cfg.EMBEDDINGS_BASE_URL), embedding_dimension=int(cfg.EMBEDDINGS_DIMENSION),
        encoder=cache)
    if len(retriever._items) != len(items):
        fail(f"retriever indexed {len(retriever._items)} of {len(items)} items "
             "(an item without a kind is dropped silently)")
    provider = make_candidate_provider(retriever)
    by_id = {it["id"]: it for it in items}

    corpus = read_json(Path(args.corpus))
    rows = corpus["rows"]
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        kind, text = row.get("kind"), row.get("observation") or ""
        routing = evaluate_kind_routing(text, kind or "")
        allowed = list(routing.expanded_kinds)
        context = {"kind": kind, "allowed_kinds": allowed, "scene_group": row.get("scene_group"),
                   "top_k_candidates": TOP_K}
        post = provider(text, context)[:TOP_K] if allowed else []
        retriever.guardrails = {}
        try:
            pre = provider(text, context)[:TOP_K] if allowed else []
        finally:
            retriever.guardrails = guardrails
        text_lower = re.sub(r"\s+", " ", text.strip()).lower()
        pre_ids = [c.get("item_id") for c in pre]
        post_ids = [c.get("item_id") for c in post]
        drops = {}
        for cid in pre_ids:
            if cid not in post_ids:
                drops[cid] = _guardrail_trace(by_id.get(cid, {}), text_lower)
        rid, rkind, reason = _resolve_candidate_via_lexical_shortcut(text, post, kind=kind or "")
        pre_rid, _pk, pre_reason = _resolve_candidate_via_lexical_shortcut(text, pre, kind=kind or "")
        _c, _cond, negated = _analyze_visible_condition_signal(text)
        top = post[0] if post else None
        second = post[1] if len(post) > 1 else None
        out_rows.append({
            "row_id": row["row_id"],
            "kind": kind, "scene_group": row.get("scene_group"),
            "routing_reason": routing.reason,
            "pre_guardrail": [{"rank": n + 1, "item_id": c.get("item_id"), "score": c.get("score")}
                              for n, c in enumerate(pre)],
            "post_guardrail": [{"rank": n + 1, "item_id": c.get("item_id"), "score": c.get("score")}
                               for n, c in enumerate(post)],
            "guardrail_drops": drops,
            "shortcut": {"resolved_item_id": rid, "resolved_kind": rkind, "reason": reason,
                         "top_score": float(top.get("score")) if top else None,
                         "second_score": float(second.get("score")) if second else 0.0,
                         "margin": (float(top.get("score")) - (float(second.get("score")) if second else 0.0))
                                   if top else None,
                         "negation_blocked": bool(negated),
                         "top_is_generic": bool(is_generic_resolution_candidate(top)) if top else None},
            "shortcut_pre_guardrail": {"resolved_item_id": pre_rid, "reason": pre_reason},
        })

    # item-item neighbourhoods, restricted to the pool an observation could actually see
    import numpy as np
    focus = [PARENT_ID, BLINDS_ID, FABRIC_ID, WINDOWS_ID]
    idx_of = {m.item_id: n for n, m in enumerate(retriever._items)}
    mat = retriever._mat
    neighbourhoods: Dict[str, Any] = {}
    for item_id in focus:
        if item_id not in idx_of:
            continue
        me = by_id[item_id]
        my_groups = set(me.get("scene_groups") or [])
        sims = mat @ mat[idx_of[item_id]]
        ranked = []
        for other, n in idx_of.items():
            if other == item_id:
                continue
            o = by_id[other]
            same_kind = o.get("kind") == me.get("kind")
            shared = (not my_groups) or (not o.get("scene_groups")) or bool(my_groups & set(o.get("scene_groups") or []))
            ranked.append({"item_id": other, "cosine": round(float(sims[n]), 6),
                           "same_kind": same_kind, "shared_scene": shared,
                           "route_override": o.get("route_override")})
        ranked.sort(key=lambda r: -r["cosine"])
        neighbourhoods[item_id] = {
            "top10_any": ranked[:10],
            "top10_same_kind_shared_scene": [r for r in ranked if r["same_kind"] and r["shared_scene"]][:10],
        }
    pairs = {}
    for a in focus:
        for b in focus:
            if a < b and a in idx_of and b in idx_of:
                pairs[f"{a}|{b}"] = round(float(mat[idx_of[a]] @ mat[idx_of[b]]), 6)

    cache_sha_after = sha256_canonical(sorted(cache._store))
    cache_file_sha = cache.save()
    payload = {
        "schema_version": "catalog-audit-replay-snapshot-v1",
        "program": "catalog_audit", "session": 5, "arm": args.arm,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm_root": str(ARM_ROOT), "git_head": head,
        "catalog": {"path": str(catalog_path), "sha256": sha256_file(catalog_path),
                    "version": catalog.get("version"), "item_count": len(items),
                    "indexed_items": len(retriever._items)},
        "corpus": {"path": rel(Path(args.corpus)), "sha256": sha256_file(Path(args.corpus)),
                   "rows": len(rows)},
        "environment": environment_record(),
        "probe": probe,
        "top_k": TOP_K,
        "cache": {"path": str(Path(args.cache)), "header": header, "stats": cache.stats,
                  "texts_before": cache.stats["loaded"], "texts_after": len(cache._store),
                  "key_set_sha256_before": cache_sha_before, "key_set_sha256_after": cache_sha_after,
                  "file_sha256": cache_file_sha},
        "script_sha256": sha256_file(SCRIPT_PATH),
        "neighbourhoods": neighbourhoods,
        "focus_pair_cosines": pairs,
        "rows": out_rows,
    }
    out = Path(args.out).resolve()
    sha = write_json(out, payload)
    fired = sum(1 for r in out_rows if r["shortcut"]["resolved_item_id"])
    print(f"snapshot[{args.arm}]: {len(out_rows)} rows, {fired} shortcut fires, "
          f"cache {cache.stats['hits']} hits / {cache.stats['misses']} misses")
    print(f"wrote {rel(out)} sha256={sha}")
    return 0


# --------------------------------------------------------------------------- tier 2: compare

def _gate_outcome(item: Optional[Dict[str, Any]], row: Dict[str, Any]) -> Dict[str, Any]:
    """Would this item be retrievable for this observation, ignoring rank?

    Reproduces the runtime order exactly: kind first (strict, fail-closed), then scene group,
    then deny_any, then require_any. This is the same gate the redraft census and leak test
    used, so their numbers can be checked rather than trusted.
    """
    from tools.pipeline_common import term_matches
    if item is None:
        return {"eligible": False, "blocked_by": "absent"}
    if item.get("kind") != row.get("kind"):
        return {"eligible": False, "blocked_by": "kind"}
    groups = item.get("scene_groups")
    if groups and row.get("scene_group") not in set(groups):
        return {"eligible": False, "blocked_by": "scene"}
    text_lower = re.sub(r"\s+", " ", (row.get("observation") or "").strip()).lower()
    deny = [t for t in (item.get("deny_any") or []) if term_matches(str(t).lower(), text_lower)]
    if deny:
        return {"eligible": False, "blocked_by": "deny_any", "terms": deny}
    require = [str(t).lower() for t in (item.get("require_any") or [])]
    if require:
        hits = [t for t in require if term_matches(t, text_lower)]
        if not hits:
            return {"eligible": False, "blocked_by": "require_any"}
        return {"eligible": True, "blocked_by": None, "require_hits": hits}
    return {"eligible": True, "blocked_by": None}


def _support_hits(item: Optional[Dict[str, Any]], text: str) -> List[str]:
    from tools.pipeline_common import term_matches
    if item is None:
        return []
    low = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return [str(t) for t in (item.get("support_any") or []) if term_matches(str(t).lower(), low)]


def _rank_of(entries: Sequence[Dict[str, Any]], item_id: str) -> Optional[int]:
    for e in entries:
        if e["item_id"] == item_id:
            return e["rank"]
    return None


def _score_of(entries: Sequence[Dict[str, Any]], item_id: str) -> Optional[float]:
    for e in entries:
        if e["item_id"] == item_id:
            return e["score"]
    return None


def cmd_compare(args: argparse.Namespace) -> int:
    assert_no_provider_imports()
    corpus = read_json(Path(args.corpus))
    base_snap = read_json(Path(args.baseline))
    cand_snap = read_json(Path(args.candidate))
    tier1_report = read_json(Path(args.tier1)) if Path(args.tier1).is_file() else {}
    if base_snap["corpus"]["sha256"] != cand_snap["corpus"]["sha256"]:
        fail("the two snapshots were built from different corpora")
    if base_snap["arm"] != "baseline" or cand_snap["arm"] != "candidate":
        fail("snapshot arms are mislabelled")

    rows = {r["row_id"]: r for r in corpus["rows"]}
    b = {r["row_id"]: r for r in base_snap["rows"]}
    c = {r["row_id"]: r for r in cand_snap["rows"]}
    base_catalog = {it["id"]: it for it in read_json(Path(base_snap["catalog"]["path"]))["items"]}
    cand_catalog = {it["id"]: it for it in read_json(Path(cand_snap["catalog"]["path"]))["items"]}

    redraft = read_json(MAIN_ROOT / "reports" / "catalog_audit_redraft.json")
    census = (redraft.get("cap007") or {}).get("withdrawn_billings_census") or {}
    leak_test = (redraft.get("cap007") or {}).get("leak_test") or []

    hard: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}

    def rule(rid: str, name: str, ok: bool, detail: Any = None) -> None:
        hard.append({"id": rid, "name": name, "result": "pass" if ok else "FAIL", "detail": detail})

    # --- score stability: identical texts must score identically for shared items ------
    score_drift = []
    for row_id, br in b.items():
        cr = c[row_id]
        bs = {e["item_id"]: e["score"] for e in br["post_guardrail"]}
        cs = {e["item_id"]: e["score"] for e in cr["post_guardrail"]}
        for item_id in set(bs) & set(cs):
            if bs[item_id] != cs[item_id]:
                score_drift.append({"row_id": row_id, "item_id": item_id,
                                    "delta": abs(bs[item_id] - cs[item_id])})
    max_drift = max((d["delta"] for d in score_drift), default=0.0)
    # The shared vector cache makes both arms embed identical text to identical vectors, so
    # any residual difference is float32 accumulation noise from multiplying a 129-row matrix
    # instead of a 128-row one. That noise is real in production too; what matters is that it
    # stays far below the shortcut thresholds, and that no decision sits inside the band.
    rule("T2-0", "shared items score within float32 noise across arms (cache holds; drift is not semantic)",
         max_drift < SCORE_EPSILON, {"rows_with_drift": len(score_drift), "max_delta": max_drift,
                                     "epsilon": SCORE_EPSILON})
    # A decision sitting near a threshold is only a validation problem if the two arms
    # disagreed about it and the gap is small enough that noise, not the split, could
    # explain the disagreement. Knife-edge rows that both arms decided the same way are
    # recorded as fragility, not counted as drift.
    fragile = []
    for row_id, cr in c.items():
        sc, bs = cr["shortcut"], b[row_id]["shortcut"]
        if sc["resolved_item_id"] == bs["resolved_item_id"]:
            continue
        top, margin = sc.get("top_score"), sc.get("margin")
        if top is None or margin is None:
            continue
        if (abs(top - float(cfg.PASS_2D_SHORTCUT_MIN_SCORE)) < SCORE_EPSILON
                or abs(margin - float(cfg.PASS_2D_SHORTCUT_MIN_MARGIN)) < SCORE_EPSILON):
            fragile.append({"row_id": row_id, "observation": rows[row_id]["observation"],
                            "top_score": top, "margin": margin,
                            "baseline_shortcut": bs["resolved_item_id"],
                            "candidate_shortcut": sc["resolved_item_id"]})
    rule("T2-0b", "no shortcut decision that differs between arms is inside the float32 noise band",
         not fragile, fragile[:10])
    # decisions close to a threshold are not wrong, but they are the ones a future embedding
    # or catalog change will flip first, so they are named rather than buried
    near_band = [{"row_id": rid, "observation": rows[rid]["observation"],
                  "top_score": cr["shortcut"]["top_score"], "margin": cr["shortcut"]["margin"],
                  "shortcut": cr["shortcut"]["resolved_item_id"]}
                 for rid, cr in c.items()
                 if cr["shortcut"]["margin"] is not None
                 and abs(cr["shortcut"]["margin"] - float(cfg.PASS_2D_SHORTCUT_MIN_MARGIN)) < 0.002]
    metrics["score_stability"] = {"rows_with_drift": len(score_drift), "max_delta": max_drift,
                                  "epsilon": SCORE_EPSILON, "inside_noise_band": fragile,
                                  "within_0.002_of_the_margin_threshold": near_band}

    # --- kind purity -------------------------------------------------------------------
    impure = []
    for snap, cat, label in ((b, base_catalog, "baseline"), (c, cand_catalog, "candidate")):
        for row_id, r in snap.items():
            kind = rows[row_id].get("kind")
            for e in r["post_guardrail"]:
                if cat.get(e["item_id"], {}).get("kind") != kind:
                    impure.append({"arm": label, "row_id": row_id, "item_id": e["item_id"]})
    rule("T2-P", "kind filtering stays strict in both arms", not impure, impure[:10])

    # --- gate re-derivation, census reconciliation -------------------------------------
    per_row_gates: Dict[str, Any] = {}
    for row_id, row in rows.items():
        per_row_gates[row_id] = {
            "parent": _gate_outcome(base_catalog.get(PARENT_ID), row),
            "blinds": _gate_outcome(cand_catalog.get(BLINDS_ID), row),
            "fabric": _gate_outcome(cand_catalog.get(FABRIC_ID), row),
        }

    control_condition_ids = {v.get("condition_id") for k, v in corpus["named_cases"].items()
                             if isinstance(v, dict) and k.startswith("rc_")}

    by_condition: Dict[str, List[str]] = {}
    for row_id, row in rows.items():
        if row.get("condition_item_id") == PARENT_ID and row.get("condition_id"):
            by_condition.setdefault(row["condition_id"], []).append(row_id)

    predicted: Dict[str, str] = {}
    for cond_id, row_ids in by_condition.items():
        reaches_fabric = any(per_row_gates[r]["fabric"]["eligible"] for r in row_ids)
        reaches_blinds = any(per_row_gates[r]["blinds"]["eligible"] for r in row_ids)
        if reaches_fabric:
            predicted[cond_id] = "surviving"
        elif reaches_blinds:
            predicted[cond_id] = "moving"
        else:
            predicted[cond_id] = "neither"
    census_expected = {cid: "moving" for cid in census.get("condition_ids_moving", [])}
    census_expected.update({cid: "surviving" for cid in census.get("condition_ids_surviving", [])})
    census_expected.update({cid: "neither" for cid in census.get("condition_ids_neither", [])})
    census_mismatch = {cid: {"replayed": predicted.get(cid), "census": want}
                       for cid, want in census_expected.items() if predicted.get(cid) != want}
    rule("T2-A", "approved census reproduced from the artifacts by the production gates",
         not census_mismatch and len(by_condition) == census.get("conditions_total"),
         {"conditions": len(by_condition), "expected": census.get("conditions_total"),
          "mismatches": census_mismatch,
          "replayed_counts": {k: sum(1 for v in predicted.values() if v == k)
                              for k in ("moving", "surviving", "neither")}})

    # reachability in the ranked list, not only at the gate
    # The census is a statement about the term gates. Retrieval adds ranking, so a condition
    # can be gate-eligible for a successor that never enters the candidate list. This audit
    # reports both, per condition, and names every exception rather than averaging them away.
    reach_rows, reach_exceptions = [], []
    for cond_id, row_ids in sorted(by_condition.items()):
        want = predicted[cond_id]
        target = {"moving": BLINDS_ID, "surviving": FABRIC_ID, "neither": None}[want]
        ranks = {r: {"blinds": _rank_of(c[r]["post_guardrail"], BLINDS_ID),
                     "fabric": _rank_of(c[r]["post_guardrail"], FABRIC_ID),
                     "parent_baseline": _rank_of(b[r]["post_guardrail"], PARENT_ID)} for r in row_ids}
        if target is None:
            reached = any(v["blinds"] or v["fabric"] for v in ranks.values())
            ok = not reached
        else:
            key = "blinds" if want == "moving" else "fabric"
            ok = any(v[key] for v in ranks.values())
        entry = {"condition_id": cond_id, "predicted_bucket": want, "target": target,
                 "reachable_in_topk": ok, "rows": [
                     {"row_id": r, "observation": rows[r]["observation"], "kind": rows[r]["kind"],
                      "scene_group": rows[r]["scene_group"], "ranks": ranks[r],
                      "gate_blinds": per_row_gates[r]["blinds"], "gate_fabric": per_row_gates[r]["fabric"],
                      "candidate_top3": [e["item_id"] for e in c[r]["post_guardrail"][:3]],
                      "candidate_shortcut": c[r]["shortcut"]["resolved_item_id"]}
                     for r in row_ids]}
        reach_rows.append(entry)
        if not ok:
            reach_exceptions.append(entry)
    metrics["reachability_audit"] = {
        "conditions": len(reach_rows),
        "reachable": sum(1 for e in reach_rows if e["reachable_in_topk"]),
        "exceptions": reach_exceptions,
        "by_bucket": {bucket: {
            "total": sum(1 for e in reach_rows if e["predicted_bucket"] == bucket),
            "reachable": sum(1 for e in reach_rows if e["predicted_bucket"] == bucket
                             and e["reachable_in_topk"])}
            for bucket in ("moving", "surviving", "neither")},
        "conditions_table": reach_rows,
    }
    # The hard rule is about reviewed evidence: a condition carrying a human review card must
    # not lose its predicted successor. Unreviewed exceptions are reported as findings.
    reviewed_exceptions = [e for e in reach_exceptions if e["condition_id"] in control_condition_ids]
    rule("T2-A2", "no human-reviewed condition loses its predicted successor from the candidate list",
         not reviewed_exceptions, {"reviewed_exceptions": reviewed_exceptions,
                                   "unreviewed_exceptions": [e["condition_id"] for e in reach_exceptions]})

    # --- preservation: what the baseline arm found must survive -------------------------
    displaced = []
    for row_id, br in b.items():
        cr = c[row_id]
        b_ids = [e["item_id"] for e in br["post_guardrail"]]
        c_entries = cr["post_guardrail"]
        c_ids = [e["item_id"] for e in c_entries]
        for e in br["post_guardrail"]:
            if e["item_id"] == PARENT_ID or e["item_id"] in c_ids:
                continue
            # Count entrants in the PRE-guardrail list: retrieval takes top-K first and applies
            # guardrails afterwards, so a successor that enters the raw top-K and is then denied
            # still consumed the slot. Counting only survivors would report those rows as
            # displacement when they are the retriever's documented ordering behaviour.
            pre_entries = cr["pre_guardrail"]
            # rank the item by where it sat in the BASELINE raw top-K, since that is the
            # position a new item has to beat to evict it
            base_pre_rank = _rank_of(br["pre_guardrail"], e["item_id"]) or e["rank"]
            base_pre_ids = [x["item_id"] for x in br["pre_guardrail"]]
            # compare by score, not by rank: ranks renumber as items enter, so an entrant that
            # scores above this item can sit at a numerically larger rank and still evict it
            own_score = _score_of(br["pre_guardrail"], e["item_id"])
            entrants = sum(1 for x in pre_entries
                           if x["item_id"] not in base_pre_ids and own_score is not None
                           and x["score"] > own_score)
            displaced.append({"row_id": row_id, "item_id": e["item_id"], "baseline_rank": e["rank"],
                              "baseline_raw_rank": base_pre_rank,
                              "frozen_owner": rows[row_id]["frozen"]["resolved_item_id"],
                              "entrants_above": entrants,
                              "explained_by_insertion": base_pre_rank + entrants > TOP_K})
    deep_displaced = [d for d in displaced if not d["explained_by_insertion"]]
    rule("T2-B", "no item other than the parent leaves the ranked list except by tail insertion",
         not deep_displaced,
         {"total_displaced": len(displaced),
          "explained_by_insertion": len(displaced) - len(deep_displaced),
          "unexplained": deep_displaced[:10]})
    # How often a denied successor consumes a retrieval slot. This is existing retriever
    # behaviour (guardrails run after top-K), but the split makes it fire more often because
    # two items now compete for the slot one item used to hold.
    slot_consumed = []
    for row_id, cr in c.items():
        pre_ids = [e["item_id"] for e in cr["pre_guardrail"]]
        post_ids = [e["item_id"] for e in cr["post_guardrail"]]
        denied = [i for i in (BLINDS_ID, FABRIC_ID) if i in pre_ids and i not in post_ids]
        if not denied:
            continue
        lost = [e["item_id"] for e in b[row_id]["post_guardrail"]
                if e["item_id"] != PARENT_ID and e["item_id"] not in post_ids]
        if lost:
            slot_consumed.append({"row_id": row_id, "denied": denied, "pushed_out": lost,
                                  "observation": rows[row_id]["observation"]})
    metrics["denied_successor_slot_consumption"] = {
        "rows": len(slot_consumed), "sample": slot_consumed[:15],
        "note": "guardrails apply after top-K, so a denied successor still occupies a candidate slot",
    }

    shortcut_changes = []
    for row_id, br in b.items():
        cr = c[row_id]
        bid = br["shortcut"]["resolved_item_id"]
        cid = cr["shortcut"]["resolved_item_id"]
        if bid == cid:
            continue
        shortcut_changes.append({
            "row_id": row_id, "baseline": bid, "candidate": cid,
            "baseline_reason": br["shortcut"]["reason"], "candidate_reason": cr["shortcut"]["reason"],
            "frozen_owner": rows[row_id]["frozen"]["resolved_item_id"],
            "frozen_path": rows[row_id]["frozen"]["resolution_path"],
            "observation": rows[row_id]["observation"], "kind": rows[row_id]["kind"],
            "scene_group": rows[row_id]["scene_group"],
        })
    # Three kinds of change, and only one of them is a concern. Landing on a successor is the
    # point of the split. Landing on the item the frozen run already chose is convergence.
    # Landing on a third item the frozen run did not choose is the case to look at, because
    # removing the parent can lift a neighbour past the margin gate and pre-empt the LLM.
    for chg in shortcut_changes:
        if chg["candidate"] in (BLINDS_ID, FABRIC_ID):
            chg["class"] = "onto_successor"
        elif chg["candidate"] is None:
            chg["class"] = "shortcut_lost"
        elif chg["candidate"] == chg["frozen_owner"]:
            chg["class"] = "onto_frozen_owner"
        else:
            chg["class"] = "onto_third_item"
    third = [s for s in shortcut_changes if s["class"] == "onto_third_item"]
    # only the human-reviewed cards count as reviewed evidence here; the census condition ids
    # are the affected population, not adjudicated cases
    control_conditions = {v.get("condition_id") for k, v in corpus["named_cases"].items()
                          if isinstance(v, dict) and k.startswith("rc_")}
    third_on_reviewed = [s for s in third
                         if rows[s["row_id"]].get("condition_id") in control_conditions]
    rule("T2-C", "no reviewed case is diverted onto a third item by a newly opened shortcut",
         not third_on_reviewed, third_on_reviewed[:10])
    metrics["shortcut_changes"] = {
        "total": len(shortcut_changes),
        "by_class": {k: sum(1 for s in shortcut_changes if s["class"] == k)
                     for k in ("onto_successor", "shortcut_lost", "onto_frozen_owner", "onto_third_item")},
        "onto_third_item": third,
        "rows": shortcut_changes,
    }

    # hijack: a shortcut that newly fires onto a successor where the frozen run resolved elsewhere
    hijacks = [s for s in shortcut_changes
               if s["candidate"] in (BLINDS_ID, FABRIC_ID)
               and s["frozen_owner"] not in (None, PARENT_ID, BLINDS_ID, FABRIC_ID)]
    rule("T2-D", "no observation owned by another item is hijacked by a successor shortcut",
         not hijacks, hijacks[:10])

    # --- named controls -----------------------------------------------------------------
    named = corpus["named_cases"]
    control_rows: Dict[str, Any] = {}
    for card_id in ("rc_7c5b9f9bb3aa", "rc_429ba6851d09", "rc_4f4b93c40ef3", "rc_ffad088fa5fe",
                    "rc_64f996e3a886", "rc_a22a241b3bf0", "rc_8afb476fa69f"):
        entry = named.get(card_id) or {}
        for row_id in entry.get("row_ids", []):
            row = rows[row_id]
            control_rows[card_id] = {
                "row_id": row_id, "observation": row["observation"], "kind": row["kind"],
                "scene_group": row["scene_group"], "frozen_owner": row["frozen"]["resolved_item_id"],
                "frozen_path": row["frozen"]["resolution_path"],
                "baseline_top": [e["item_id"] for e in b[row_id]["post_guardrail"][:3]],
                "candidate_top": [e["item_id"] for e in c[row_id]["post_guardrail"][:3]],
                "baseline_shortcut": b[row_id]["shortcut"]["resolved_item_id"],
                "candidate_shortcut": c[row_id]["shortcut"]["resolved_item_id"],
                "candidate_shortcut_reason": c[row_id]["shortcut"]["reason"],
                "blinds_gate": per_row_gates[row_id]["blinds"],
                "fabric_gate": per_row_gates[row_id]["fabric"],
                "blinds_rank": _rank_of(c[row_id]["post_guardrail"], BLINDS_ID),
                "fabric_rank": _rank_of(c[row_id]["post_guardrail"], FABRIC_ID),
                "windows_rank_baseline": _rank_of(b[row_id]["post_guardrail"], WINDOWS_ID),
                "windows_rank_candidate": _rank_of(c[row_id]["post_guardrail"], WINDOWS_ID),
            }
    metrics["controls"] = control_rows
    # the one reviewed positive use must still reach a successor
    pos = control_rows.get("rc_7c5b9f9bb3aa") or {}
    rule("T2-E", "the single reviewed positive use still reaches a successor (Q-1 moves it to no_action, "
                 "it must not become unreachable)",
         bool(pos.get("blinds_rank") or pos.get("fabric_rank")), pos)
    # the windows-item correct rejection must be untouched
    win = control_rows.get("rc_64f996e3a886") or {}
    rule("T2-F", "the dated_or_older_windows correct rejection is unchanged by the split",
         win.get("baseline_top") == win.get("candidate_top")
         and win.get("baseline_shortcut") == win.get("candidate_shortcut"), win)

    # --- blinds shortcut fire rate ------------------------------------------------------
    blinds_pop = [rid for rid, row in rows.items()
                  if row.get("kind") == "modernization"
                  and per_row_gates[rid]["blinds"]["eligible"]
                  and any(s in (row.get("observation") or "").lower() for s in BLINDS_STEMS)]
    blinds_fired = [rid for rid in blinds_pop if c[rid]["shortcut"]["resolved_item_id"] == BLINDS_ID]
    base_parent_fired = [rid for rid in blinds_pop if b[rid]["shortcut"]["resolved_item_id"] == PARENT_ID]
    excluded_kind = [rid for rid, row in rows.items()
                     if row.get("kind") != "modernization"
                     and any(s in (row.get("observation") or "").lower() for s in BLINDS_STEMS)]
    metrics["blinds_shortcut"] = {
        "denominator_definition": "modernization AND scene-eligible AND passes the blinds gate AND names a blind/shade",
        "denominator": len(blinds_pop),
        "candidate_fires_onto_blinds": len(blinds_fired),
        "baseline_fires_onto_parent": len(base_parent_fired),
        "candidate_fire_rate": round(len(blinds_fired) / len(blinds_pop), 4) if blinds_pop else None,
        "baseline_fire_rate": round(len(base_parent_fired) / len(blinds_pop), 4) if blinds_pop else None,
        "non_modernization_blinds_rows_unreachable_by_design": len(excluded_kind),
        "rows": [{"row_id": rid, "observation": rows[rid]["observation"],
                  "scene_group": rows[rid]["scene_group"],
                  "frozen_owner": rows[rid]["frozen"]["resolved_item_id"],
                  "baseline_shortcut": b[rid]["shortcut"]["resolved_item_id"],
                  "candidate_shortcut": c[rid]["shortcut"]["resolved_item_id"],
                  "candidate_reason": c[rid]["shortcut"]["reason"],
                  "blinds_rank": _rank_of(c[rid]["post_guardrail"], BLINDS_ID),
                  "top_score": c[rid]["shortcut"]["top_score"],
                  "margin": c[rid]["shortcut"]["margin"]} for rid in blinds_pop],
    }
    # Why the fire rate moved: the presence-only successor carries no datedness language, so on
    # "the blinds are dated" bullets it scores closer to dated_or_older_windows than the parent
    # did, and the top-two margin collapses toward the 0.03 gate.
    margins = []
    for rid in blinds_pop:
        bm, cm = b[rid]["shortcut"]["margin"], c[rid]["shortcut"]["margin"]
        if bm is None or cm is None:
            continue
        margins.append({"row_id": rid, "observation": rows[rid]["observation"],
                        "baseline_margin": round(bm, 5), "candidate_margin": round(cm, 5),
                        "baseline_top": b[rid]["shortcut"]["top_score"],
                        "candidate_top": c[rid]["shortcut"]["top_score"],
                        "crossed_below_threshold": bm >= float(cfg.PASS_2D_SHORTCUT_MIN_MARGIN)
                                                   > cm})
    crossed = [m for m in margins if m["crossed_below_threshold"]]
    metrics["blinds_margin_collapse"] = {
        "rows": len(margins),
        "median_baseline_margin": round(sorted(m["baseline_margin"] for m in margins)[len(margins) // 2], 5)
        if margins else None,
        "median_candidate_margin": round(sorted(m["candidate_margin"] for m in margins)[len(margins) // 2], 5)
        if margins else None,
        "rows_that_fell_below_the_margin_gate": len(crossed),
        "crossed": crossed,
        "explanation": ("the blinds successor's presence-only embed_text drops the datedness language the "
                        "parent carried, so on dated/older blinds bullets it sits closer to "
                        "dated_or_older_windows and the top-two margin shrinks toward the 0.03 gate"),
    }

    misfires = [rid for rid in blinds_fired
                if rows[rid]["frozen"]["resolved_item_id"] not in (None, PARENT_ID)
                and rows[rid]["frozen"]["resolution_path"] == "lexical_shortcut"]
    rule("T2-G", "the blinds shortcut never fires on an observation another item resolved by shortcut",
         not misfires, [{"row_id": r, "frozen_owner": rows[r]["frozen"]["resolved_item_id"],
                         "observation": rows[r]["observation"]} for r in misfires][:10])

    # --- fabric bare-stem surface --------------------------------------------------------
    fabric_rows = [rid for rid, row in rows.items()
                   if any(s in (row.get("observation") or "").lower() for s in FABRIC_BARE_STEMS)]
    fabric_table = []
    for rid in fabric_rows:
        row = rows[rid]
        gate = per_row_gates[rid]["fabric"]
        fabric_table.append({
            "row_id": rid, "observation": row["observation"], "kind": row["kind"],
            "scene_group": row["scene_group"], "frozen_owner": row["frozen"]["resolved_item_id"],
            "frozen_path": row["frozen"]["resolution_path"],
            "gate": gate.get("blocked_by") or "eligible",
            "gate_terms": gate.get("terms") or gate.get("require_hits"),
            "fabric_rank": _rank_of(c[rid]["post_guardrail"], FABRIC_ID),
            "support_hits": _support_hits(cand_catalog.get(FABRIC_ID), row["observation"]),
            "baseline_shortcut": b[rid]["shortcut"]["resolved_item_id"],
            "candidate_shortcut": c[rid]["shortcut"]["resolved_item_id"],
            "shortcut_onto_fabric": c[rid]["shortcut"]["resolved_item_id"] == FABRIC_ID,
        })
    fabric_hijacks = [f for f in fabric_table
                      if f["shortcut_onto_fabric"]
                      and f["frozen_owner"] not in (None, PARENT_ID)
                      and f["frozen_owner"] != FABRIC_ID]
    metrics["fabric_bare_stems"] = {
        "stems": list(FABRIC_BARE_STEMS), "rows": len(fabric_table),
        "reaching_fabric_gate": sum(1 for f in fabric_table if f["gate"] == "eligible"),
        "shortcut_onto_fabric": sum(1 for f in fabric_table if f["shortcut_onto_fabric"]),
        "blocked_by": {k: sum(1 for f in fabric_table if f["gate"] == k)
                       for k in sorted({f["gate"] for f in fabric_table})},
        "table": fabric_table,
    }
    rule("T2-H", "no bare-stem row owned by another item is taken by the fabric successor's shortcut",
         not fabric_hijacks, fabric_hijacks[:10])

    # --- windows neighbour ----------------------------------------------------------------
    windows_rows = [rid for rid, row in rows.items()
                    if row["frozen"]["resolved_item_id"] == WINDOWS_ID
                    or _rank_of(b[rid]["post_guardrail"], WINDOWS_ID)
                    or _rank_of(c[rid]["post_guardrail"], WINDOWS_ID)]
    windows_table = []
    for rid in windows_rows:
        row = rows[rid]
        wb, wc = _rank_of(b[rid]["post_guardrail"], WINDOWS_ID), _rank_of(c[rid]["post_guardrail"], WINDOWS_ID)
        windows_table.append({
            "row_id": rid, "observation": row["observation"], "kind": row["kind"],
            "scene_group": row["scene_group"], "frozen_owner": row["frozen"]["resolved_item_id"],
            "windows_rank_baseline": wb, "windows_rank_candidate": wc,
            "windows_score": _score_of(c[rid]["post_guardrail"], WINDOWS_ID),
            "blinds_rank": _rank_of(c[rid]["post_guardrail"], BLINDS_ID),
            "blinds_score": _score_of(c[rid]["post_guardrail"], BLINDS_ID),
            "baseline_shortcut": b[rid]["shortcut"]["resolved_item_id"],
            "candidate_shortcut": c[rid]["shortcut"]["resolved_item_id"],
            "scene_eligible_for_successors": row["scene_group"] in {"kitchen", "bathroom", "bedroom",
                                                                    "living_areas", "utility"},
        })
    # Falling off the tail of a row the windows item never owned is insertion arithmetic.
    # A regression is losing a row it owned, or one where it stood near the top.
    windows_regressions = [w for w in windows_table
                           if w["windows_rank_baseline"] is not None and w["windows_rank_candidate"] is None
                           and (w["frozen_owner"] == WINDOWS_ID or w["windows_rank_baseline"] <= 3)]
    metrics["windows_neighbour"] = {
        "rows": len(windows_table),
        "windows_dropped_from_topk": len(windows_regressions),
        "outranked_by_blinds": sum(1 for w in windows_table
                                   if w["blinds_rank"] and w["windows_rank_candidate"]
                                   and w["blinds_rank"] < w["windows_rank_candidate"]),
        "shortcut_moved_windows_to_blinds": [w for w in windows_table
                                             if w["baseline_shortcut"] == WINDOWS_ID
                                             and w["candidate_shortcut"] == BLINDS_ID],
        "table": windows_table,
    }
    rule("T2-I", "dated_or_older_windows keeps every ranked position it held in the baseline arm",
         not windows_regressions, {"regressions": windows_regressions[:10],
                                   "tail_only_dropouts": sum(
                                       1 for w in windows_table
                                       if w["windows_rank_baseline"] is not None
                                       and w["windows_rank_candidate"] is None
                                       and w["frozen_owner"] != WINDOWS_ID
                                       and w["windows_rank_baseline"] > 3)})

    # --- ambiguity, measured pre-guardrail where co-presence is possible ------------------
    ambiguous = []
    for rid, cr in c.items():
        pre_ids = [e["item_id"] for e in cr["pre_guardrail"]]
        if BLINDS_ID in pre_ids and FABRIC_ID in pre_ids:
            sb = _score_of(cr["pre_guardrail"], BLINDS_ID)
            sf = _score_of(cr["pre_guardrail"], FABRIC_ID)
            if sb is not None and sf is not None and abs(sb - sf) < 0.03:
                ambiguous.append({"row_id": rid, "observation": rows[rid]["observation"],
                                  "blinds": sb, "fabric": sf, "delta": round(abs(sb - sf), 6),
                                  "shortcut": cr["shortcut"]["resolved_item_id"]})
    neither = [rid for rid, row in rows.items()
               if row.get("condition_item_id") == PARENT_ID
               and not per_row_gates[rid]["blinds"]["eligible"]
               and not per_row_gates[rid]["fabric"]["eligible"]]
    metrics["ambiguity"] = {"both_successors_within_margin_pre_guardrail": len(ambiguous),
                            "rows": ambiguous[:40],
                            "parent_rows_reaching_neither_successor": len(neither),
                            "neither_row_ids": neither}

    # margin inflation: dropping a successor can open a shortcut onto whatever survives
    inflation = [{"row_id": rid, "pre": cr["shortcut_pre_guardrail"]["resolved_item_id"],
                  "post": cr["shortcut"]["resolved_item_id"],
                  "observation": rows[rid]["observation"]}
                 for rid, cr in c.items()
                 if cr["shortcut_pre_guardrail"]["resolved_item_id"] != cr["shortcut"]["resolved_item_id"]]
    base_inflation = [rid for rid, br in b.items()
                      if br["shortcut_pre_guardrail"]["resolved_item_id"] != br["shortcut"]["resolved_item_id"]]
    metrics["margin_inflation"] = {"candidate_rows": len(inflation), "baseline_rows": len(base_inflation),
                                   "rows": inflation[:20]}

    # --- leak-test parity ------------------------------------------------------------------
    leak_rows, leak_mismatch = [], []
    for entry in leak_test:
        text = entry.get("bullet") or ""
        kinds = entry.get("pinned_kind") or []
        groups = entry.get("pinned_scene_groups") or []
        if isinstance(kinds, str):
            kinds = [kinds]
        if isinstance(groups, str):
            groups = [groups]
        # the recorded gate flags are term-level (deny/require only); reaches_* additionally
        # requires the kind and scene gates, so both are replayed separately
        term_row = {"observation": text, "kind": cand_catalog[BLINDS_ID]["kind"],
                    "scene_group": (cand_catalog[BLINDS_ID].get("scene_groups") or [None])[0]}
        gb_term = _gate_outcome(cand_catalog.get(BLINDS_ID), term_row)
        term_row_f = dict(term_row, kind=cand_catalog[FABRIC_ID]["kind"])
        gf_term = _gate_outcome(cand_catalog.get(FABRIC_ID), term_row_f)
        reaches_fabric = any(
            _gate_outcome(cand_catalog.get(FABRIC_ID),
                          {"observation": text, "kind": k, "scene_group": g})["eligible"]
            for k in (kinds or [None]) for g in (groups or [None]))
        reaches_blinds = any(
            _gate_outcome(cand_catalog.get(BLINDS_ID),
                          {"observation": text, "kind": k, "scene_group": g})["eligible"]
            for k in (kinds or [None]) for g in (groups or [None]))
        replayed = {"passes_blinds_gate": gb_term["eligible"], "passes_fabric_gate": gf_term["eligible"],
                    "reaches_fabric_successor": reaches_fabric, "reaches_blinds_successor": reaches_blinds,
                    "shortcut_surface_fabric": bool(_support_hits(cand_catalog.get(FABRIC_ID), text)),
                    "shortcut_surface_blinds": bool(_support_hits(cand_catalog.get(BLINDS_ID), text))}
        recorded = {k: entry[k] for k in
                    ("passes_blinds_gate", "passes_fabric_gate", "reaches_fabric_successor",
                     "shortcut_surface_fabric", "shortcut_surface_blinds") if k in entry}
        bad = {k: {"recorded": v, "replayed": replayed.get(k)}
               for k, v in recorded.items() if replayed.get(k) != v}
        row = {"bullet": text, "pinned_kind": kinds, "pinned_scene_groups": groups,
               "recorded": recorded, "replayed": replayed, "mismatch": bad or None}
        leak_rows.append(row)
        if bad:
            leak_mismatch.append(row)
    rule("T2-J", "the approved leak test reproduces field for field under the production gates",
         not leak_mismatch, {"rows": len(leak_rows), "mismatches": leak_mismatch[:10]})
    metrics["leak_test"] = {"rows": leak_rows}

    # --- neighbourhoods --------------------------------------------------------------------
    metrics["neighbourhoods"] = {"baseline": base_snap["neighbourhoods"],
                                 "candidate": cand_snap["neighbourhoods"],
                                 "focus_pair_cosines": {"baseline": base_snap["focus_pair_cosines"],
                                                        "candidate": cand_snap["focus_pair_cosines"]}}
    presentation_only = [i for i, it in cand_catalog.items() if it.get("route_override") == "no_action"]
    close_siblings = {}
    for item_id in (BLINDS_ID, FABRIC_ID):
        nb = cand_snap["neighbourhoods"].get(item_id, {}).get("top10_same_kind_shared_scene", [])
        close_siblings[item_id] = [n for n in nb if n["item_id"] in presentation_only or n["item_id"] == WINDOWS_ID]
    metrics["neighbourhoods"]["flagged_siblings"] = close_siblings

    # --- findings: true statements the rules do not capture, written out rather than averaged --
    findings: List[Dict[str, Any]] = []
    audit = metrics["reachability_audit"]
    for exc in audit["exceptions"]:
        reviewed = exc["condition_id"] in control_condition_ids
        findings.append({
            "id": f"F-REACH-{exc['condition_id']}",
            "severity": "material" if exc["predicted_bucket"] == "surviving" else "expected",
            "condition_id": exc["condition_id"],
            "predicted_bucket": exc["predicted_bucket"],
            "human_reviewed": reviewed,
            "summary": (f"the approval predicts this condition lands on {exc['target']}, but that item "
                        f"is not in the candidate top-{TOP_K} for any of its observations"),
            "observations": [{"observation": r["observation"], "scene_group": r["scene_group"],
                              "candidate_top3": r["candidate_top3"],
                              "candidate_shortcut": r["candidate_shortcut"],
                              "gate_blinds": r["gate_blinds"], "gate_fabric": r["gate_fabric"],
                              "parent_baseline_rank": r["ranks"]["parent_baseline"]}
                             for r in exc["rows"]],
        })
    for chg in metrics["shortcut_changes"]["onto_third_item"]:
        findings.append({
            "id": f"F-SHORTCUT-{chg['row_id'].rsplit(':', 1)[-1]}",
            "severity": "material" if chg["candidate"] not in (WINDOWS_ID,) else "expected",
            "summary": (f"removing the parent opened a lexical shortcut onto {chg['candidate']}, "
                        f"which the frozen run did not choose; the LLM no longer sees this row"),
            "observation": chg["observation"], "scene_group": chg["scene_group"],
            "frozen_owner": chg["frozen_owner"], "frozen_path": chg["frozen_path"],
            "candidate_shortcut": chg["candidate"], "row_id": chg["row_id"],
        })
    if metrics["denied_successor_slot_consumption"]["rows"]:
        findings.append({
            "id": "F-SLOT-CONSUMPTION",
            "severity": "informational",
            "summary": (f"on {metrics['denied_successor_slot_consumption']['rows']} rows a successor "
                        f"entered the raw top-{TOP_K} and was then denied by a guardrail, so the slot "
                        "was spent and a lower-scoring item never became a candidate. This is existing "
                        "retriever behaviour (guardrails run after top-K), amplified because two items "
                        "now compete where one stood."),
        })
    if metrics["score_stability"]["within_0.002_of_the_margin_threshold"]:
        findings.append({
            "id": "F-MARGIN-FRAGILITY",
            "severity": "informational",
            "summary": (f"{len(metrics['score_stability']['within_0.002_of_the_margin_threshold'])} rows "
                        "decide the lexical shortcut within 0.002 of the margin threshold; these flip "
                        "first under any future embedding or wording change."),
        })
    metrics["findings"] = findings

    failures = [r for r in hard if r["result"] != "pass"]
    payload = {
        "schema_version": "catalog-audit-validation-tier12-v1",
        "program": "catalog_audit", "session": 5,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "corpus": {"path": rel(Path(args.corpus)), "sha256": sha256_file(Path(args.corpus))},
            "snapshot_baseline": {"path": rel(Path(args.baseline)), "sha256": sha256_file(Path(args.baseline))},
            "snapshot_candidate": {"path": rel(Path(args.candidate)), "sha256": sha256_file(Path(args.candidate))},
            "tier1": {"path": rel(Path(args.tier1)), "sha256": sha256_file(Path(args.tier1))
                      if Path(args.tier1).is_file() else None},
        },
        "arms": {"baseline": {"git_head": base_snap["git_head"], "catalog": base_snap["catalog"]},
                 "candidate": {"git_head": cand_snap["git_head"], "catalog": cand_snap["catalog"]}},
        "probe": cand_snap["probe"],
        "tier1_result": tier1_report.get("result"),
        "tier1_checks": tier1_report.get("checks"),
        "tier2_rules": hard,
        "tier2_metrics": metrics,
        "result": "pass" if not failures and tier1_report.get("result") == "pass" else "FAIL",
        "tier2_result": "pass" if not failures else "FAIL",
        "findings": findings,
        "findings_summary": {sev: sum(1 for f in findings if f["severity"] == sev)
                             for sev in ("material", "expected", "informational")},
        "failures": [r["id"] for r in failures],
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    out = Path(args.out).resolve()
    sha = write_json(out, payload)
    print(f"tier2: {payload['tier2_result']} ({len(hard) - len(failures)}/{len(hard)} rules passed); "
          f"overall {payload['result']}")
    for r in failures:
        print(f"  FAIL {r['id']} {r['name']}")
    print(f"  blinds fire rate: baseline {metrics['blinds_shortcut']['baseline_fire_rate']} -> "
          f"candidate {metrics['blinds_shortcut']['candidate_fire_rate']} "
          f"(n={metrics['blinds_shortcut']['denominator']})")
    print(f"  shortcut changes: {metrics['shortcut_changes']['total']} "
          f"{metrics['shortcut_changes']['by_class']}")
    print(f"  reachability: {metrics['reachability_audit']['reachable']}/"
          f"{metrics['reachability_audit']['conditions']} parent conditions reach their predicted successor")
    for f in findings:
        if f["severity"] == "material":
            print(f"  FINDING [{f['severity']}] {f['id']}: {f['summary'][:110]}")
    print(f"wrote {rel(out)} sha256={sha}")
    return 0 if not failures else 1


# --------------------------------------------------------------------------- stale-id inventory

def stale_id_inventory(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Every stored artifact that still names the deprecated parent (CCF-8, S4-2).

    There is no runtime alias and backfill_kind_v2 is a no-op at the target version, so each of
    these fails hard on any offline recompute or resumed run against the candidate catalog. The
    inventory exists so Session 6 replays them rather than discovering them one ValueError at a
    time, and so nobody is tempted to edit a frozen artifact to make a red test green.
    """
    roots: Dict[str, Dict[str, Any]] = {}
    total = 0
    canary = MAIN_ROOT / "artifacts_canary"
    needle = PARENT_ID.encode("utf-8")
    if canary.is_dir():
        for path in canary.rglob("photo_intel_debug.json"):
            try:
                if needle not in path.read_bytes():
                    continue
            except OSError:
                continue
            total += 1
            root = path.relative_to(canary).parts[0]
            entry = roots.setdefault(root, {"files": 0, "properties": set()})
            entry["files"] += 1
            parts = path.relative_to(canary).parts
            if len(parts) > 1:
                entry["properties"].add(parts[1])
    pinned = []
    for row in evidence.get("run_artifacts", []):
        path = row.get("artifact_path")
        if not path or not Path(path).is_file():
            continue
        if needle in Path(path).read_bytes():
            pinned.append({"source": row["source"], "property_key": row["property_key"],
                           "run_id": row["run_id"], "sha256": row["current_sha256"],
                           "catalog_version": row.get("catalog_version"),
                           "artifact_path": rel(Path(path))})
    return {
        "note": ("stored artifacts naming the deprecated parent; each fails hard on offline recompute "
                 "against the candidate catalog (tools/renovation_estimate.py stale-id guard). The fix "
                 "is a replay from frozen Pass 2c, never an alias and never an edit to a stored artifact."),
        "canary_roots": {k: {"files": v["files"], "properties": len(v["properties"])}
                         for k, v in sorted(roots.items())},
        "canary_files_total": total,
        "pinned_evidence_runs": pinned,
        "pinned_evidence_run_count": len(pinned),
        "known_failing_test": ("tests/test_benchmark_pass2a.py::test_compute_pre2f_totals_force_confirms_packages "
                               "-> artifacts_canary/candidate/redfin_11000447/20260808_085243_19c4a4d0"),
    }


# --------------------------------------------------------------------------- benchmark gold drafts

def gold_case_drafts(tier12: Dict[str, Any], corpus: Dict[str, Any]) -> Dict[str, Any]:
    """Two draft benchmark cases, one per successor, grounded in the replay rather than invented.

    Session 4 left `test_every_split_successor_is_gold_somewhere` red: the split created two items
    the catalog-resolution benchmark has never scored. The slices are frozen and fingerprinted and
    committed results pin those fingerprints, so this session drafts the cases and hands them to
    Session 6 to add and re-freeze alongside a live benchmark run rather than editing frozen data.

    Each draft is chosen from observations that actually resolve to the intended successor in the
    candidate replay, so the case is evidence rather than an assumption about how retrieval behaves.
    """
    rows = {r["row_id"]: r for r in corpus["rows"]}
    picks: Dict[str, Any] = {}
    blinds_rows = tier12["tier2_metrics"]["blinds_shortcut"]["rows"]
    for row in blinds_rows:
        if row["candidate_shortcut"] != BLINDS_ID or row["blinds_rank"] != 1:
            continue
        src = rows[row["row_id"]]
        if src["scene_group"] not in ("bedroom", "living_areas"):
            continue
        picks[BLINDS_ID] = (src, row)
        break
    for row in tier12["tier2_metrics"]["fabric_bare_stems"]["table"]:
        if not row["shortcut_onto_fabric"] or row["fabric_rank"] != 1:
            continue
        src = rows[row["row_id"]]
        if src["scene_group"] not in ("bedroom", "living_areas", "kitchen"):
            continue
        picks[FABRIC_ID] = (src, row)
        break

    drafts = []
    for n, (item_id, key) in enumerate(((BLINDS_ID, "blinds"), (FABRIC_ID, "fabric")), start=1):
        if item_id not in picks:
            drafts.append({"resolved_id": item_id, "status": "no_grounded_row_found"})
            continue
        src, replay = picks[item_id]
        drafts.append({
            "case_id": f"res-mod-cap007-{key}",
            "family": "modernization_misc",
            "paired_group": "cap007-window-treatment-split",
            "case_type": "resolution",
            "scene": {"bedroom": "bedroom", "living_areas": "living_room",
                      "kitchen": "kitchen"}.get(src["scene_group"], src["scene_group"]),
            "scene_group": src["scene_group"],
            "input": src["observation"],
            "kind": src["kind"],
            "gold": {"resolved_id": item_id, "acceptable_candidate_ids": [item_id]},
            "legacy": None,
            "_provenance": {
                "row_id": src["row_id"], "property_key": src["property_key"], "run_id": src["run_id"],
                "photo_key": src["photo_key"], "artifact_sha256": next(
                    (a["sha256"] for a in corpus["artifacts"]
                     if a.get("readable") and a.get("run_id") == src["run_id"]), None),
                "candidate_rank_1": True,
                "resolves_by": replay.get("candidate_reason") or "support_phrase_hit",
                "frozen_owner": src["frozen"]["resolved_item_id"],
            },
        })
    return {
        "owner": "Session 6 (add to a slice and re-freeze with a live benchmark run)",
        "why_not_here": ("benchmarks/catalog-resolution-v2 slices are frozen and fingerprinted, committed "
                         "results pin those fingerprints, and --gates refuses a non-frozen slice; editing "
                         "them is evaluation content, outside a validation session's authority"),
        "target_file_suggestion": "benchmarks/catalog-resolution-v2/cases_holdout.json",
        "unblocks": "tests/test_catalog_resolution_benchmark.py::test_every_split_successor_is_gold_somewhere",
        "schema_checks": ("case_type resolution requires an in-ontology kind, a non-null gold.resolved_id and "
                          "a non-empty acceptable_candidate_ids; test_shipped_cases_reference_real_catalog_items "
                          "additionally requires catalog[gold].kind == case.kind and "
                          "case.scene_group in catalog[gold].scene_groups (both hold for these drafts)"),
        "drafts": drafts,
    }


# --------------------------------------------------------------------------- live experiment manifest

SAFE_MODEL_ENV_KEYS = ("OPENAI_MODEL", "GPT_MODEL", "RENOVATION_TERRA_MODEL",
                       "RENOVATION_TERRA_MAX_OUTPUT_TOKENS", "RENOVATION_SOL_MODEL",
                       "RENOVATION_SOL_MAX_OUTPUT_TOKENS", "LM_STUDIO_URL", "LM_STUDIO_MODEL",
                       "RENOVATION_TERRA_DAILY_TOKEN_CEILING")

MANIFEST_CODE_FILES = (
    "tools/scene_classifier_passes.py", "tools/scene_classifier_orchestrator.py",
    "tools/catalog_embeddings.py", "tools/pipeline_common.py", "tools/pipeline_config.py",
    "tools/catalog_validation.py", "tools/renovation_estimate.py", "tools/observation_kinds.py",
    "scripts/migrate_catalog_kind_v2.py",
)


def _dotenv_model_map(root: Path) -> Dict[str, Optional[str]]:
    """Model identity by name only. Values come from .env, which also holds an API key, so the
    manifest records the routing keys and never the file."""
    out: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in SAFE_MODEL_ENV_KEYS}
    env_path = root / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in SAFE_MODEL_ENV_KEYS and not out.get(key):
                out[key] = value.strip().strip('"').strip("'")
    return out


def _photo_reference(root: Optional[Path], property_key: str, photo_key: str) -> Dict[str, Any]:
    """Pin the image itself, not just its name.

    Session 6 replays from frozen Pass 2c, so it does not re-run image analysis - but Terra
    verification is shown the photo, so a changed or missing image would silently change the
    experiment. Recording the hash makes that detectable instead of invisible.
    """
    if root is None or not photo_key:
        return {"photo_key": photo_key, "sha256": None, "status": "photo_root_not_supplied"}
    path = root / property_key / photo_key
    if not path.is_file():
        return {"photo_key": photo_key, "sha256": None, "status": "not_found"}
    return {"photo_key": photo_key, "path": path.as_posix(), "sha256": sha256_file(path),
            "bytes": path.stat().st_size, "status": "pinned"}


def cmd_manifest(args: argparse.Namespace) -> int:
    assert_no_provider_imports()
    tier12 = read_json(Path(args.tier12))
    if tier12.get("result") != "pass":
        fail("refusing to build a live manifest: the Tier 1/2 report does not record a pass")
    corpus = read_json(Path(args.corpus))
    evidence = read_json(MAIN_ROOT / "reports" / "catalog_audit_evidence.json")
    approvals = read_json(MAIN_ROOT / "reports" / "catalog_audit_approvals.json")
    cap007 = next(d for d in approvals["dispositions"] if d["proposal_id"] == APPROVED_PROPOSAL_ID)
    candidate_root = Path(args.candidate_root).resolve()
    rows = {r["row_id"]: r for r in corpus["rows"]}
    cand_snap = read_json(Path(args.candidate_snapshot))
    base_snap = read_json(Path(args.baseline_snapshot))
    cand_rows = {r["row_id"]: r for r in cand_snap["rows"]}
    base_rows = {r["row_id"]: r for r in base_snap["rows"]}

    # --- the case set: every row Session 6 must rerun, frozen at Pass 2c -------------------
    census = (read_json(MAIN_ROOT / "reports" / "catalog_audit_redraft.json")
              .get("cap007", {}).get("withdrawn_billings_census", {}))
    buckets = {}
    for key, label in (("condition_ids_moving", "moving_to_no_action"),
                       ("condition_ids_surviving", "surviving_on_fabric"),
                       ("condition_ids_neither", "reaching_neither")):
        for cid in census.get(key, []):
            buckets[cid] = label
    control_of = {}
    for card_id, entry in corpus["named_cases"].items():
        if card_id.startswith("rc_") and isinstance(entry, dict):
            for rid in entry.get("row_ids", []):
                control_of[rid] = card_id
    ledger = {row.get("card_id"): row for row
              in (read_json(MAIN_ROOT / "reports" / "catalog_audit_redraft.json")
                  .get("cap007", {}).get("regression_ledger", []) or [])
              if row.get("card_id")}
    photo_root = Path(args.photo_root) if args.photo_root else None

    artifact_sha = {a["run_id"]: a.get("sha256") for a in corpus["artifacts"] if a.get("readable")}
    cases = []
    for row_id, row in sorted(rows.items()):
        cond = row.get("condition_id")
        bucket = buckets.get(cond)
        card = control_of.get(row_id)
        is_finding = any(row_id == f.get("row_id") or
                         any(o.get("observation") == row["observation"] for o in f.get("observations", []))
                         for f in tier12.get("findings", []) if f["severity"] == "material")
        if not (bucket or card or is_finding):
            continue
        cases.append({
            "case_id": row_id,
            "role": ("control" if card else "affected") if not is_finding else "finding",
            "review_card_id": card,
            "census_bucket": bucket,
            "condition_id": cond,
            "property_key": row["property_key"], "run_id": row["run_id"], "source": row["source"],
            "artifact_path": rel(Path(row["artifact_path"])),
            "artifact_sha256": artifact_sha.get(row["run_id"]),
            "photo_key": row["photo_key"], "issue_id": row["issue_id"],
            "frozen_pass_2c": {"observation": row["observation"], "kind": row["kind"],
                               "scene_group": row["scene_group"]},
            "frozen_resolution": row["frozen"],
            "offline_prediction": {
                "baseline_shortcut": base_rows[row_id]["shortcut"]["resolved_item_id"],
                "candidate_shortcut": cand_rows[row_id]["shortcut"]["resolved_item_id"],
                "candidate_top3": [e["item_id"] for e in cand_rows[row_id]["post_guardrail"][:3]],
            },
            "photo": _photo_reference(photo_root, row["property_key"], row["photo_key"]),
            "expected_after": (ledger.get(card) or {}).get("expected_after"),
            "expected_after_reason_class": (ledger.get(card) or {}).get("reason_class"),
        })

    manifest: Dict[str, Any] = {
        "schema_version": "catalog-audit-live-experiment-manifest-v1",
        "program": "catalog_audit", "produced_by_session": 5,
        "consumed_by_session": 6,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "awaiting_human_cost_authorization",

        "arms": {
            "baseline": {
                "role": "control", "git_commit": BASELINE_COMMIT, "branch": "terra_factorized_verifier",
                "checkout": str(MAIN_ROOT),
                "catalog_sha256": base_snap["catalog"]["sha256"],
                "catalog_version": base_snap["catalog"]["version"],
                "item_count": base_snap["catalog"]["item_count"],
                "projection_fingerprint": tier12["tier1_checks"] and
                    next((c["detail"]["baseline"] for c in tier12["tier1_checks"]
                          if c["id"] == "1.6g"), None),
            },
            "candidate": {
                "role": "candidate", "git_commit": CANDIDATE_COMMIT, "branch": "catalog_audit_session4",
                "checkout": str(candidate_root),
                "catalog_sha256": cand_snap["catalog"]["sha256"],
                "catalog_version": cand_snap["catalog"]["version"],
                "item_count": cand_snap["catalog"]["item_count"],
                "projection_fingerprint": next((c["detail"]["candidate"] for c in tier12["tier1_checks"]
                                                if c["id"] == "1.6g"), None),
            },
            "identity_rule": ("both arms carry catalog version '3.2'; the version string cannot tell them "
                              "apart, so every live run must be identified by catalog_sha256 and the "
                              "projection fingerprint above"),
        },

        "approved": {
            "proposal_ids": [APPROVED_PROPOSAL_ID],
            "approvals_path": "reports/catalog_audit_approvals.json",
            "approvals_sha256": APPROVALS_SHA256,
            "disposition": cap007["disposition"],
            "conditions": cap007["conditions"],
            "added_items": [BLINDS_ID, FABRIC_ID],
            "removed_items": [PARENT_ID],
        },

        "inputs_pinned": {
            "evidence": {"path": "reports/catalog_audit_evidence.json",
                         "sha256": sha256_file(MAIN_ROOT / "reports" / "catalog_audit_evidence.json"),
                         "fingerprint": evidence.get("fingerprint")},
            "proposals": {"path": "reports/catalog_audit_proposals.json",
                          "sha256": sha256_file(MAIN_ROOT / "reports" / "catalog_audit_proposals.json")},
            "redraft": {"path": "reports/catalog_audit_redraft.json",
                        "sha256": sha256_file(MAIN_ROOT / "reports" / "catalog_audit_redraft.json")},
            "tier12_report": {"path": rel(Path(args.tier12)), "sha256": sha256_file(Path(args.tier12))},
            "replay_corpus": {"path": rel(Path(args.corpus)), "sha256": sha256_file(Path(args.corpus))},
            "snapshot_baseline": {"path": rel(Path(args.baseline_snapshot)),
                                  "sha256": sha256_file(Path(args.baseline_snapshot))},
            "snapshot_candidate": {"path": rel(Path(args.candidate_snapshot)),
                                   "sha256": sha256_file(Path(args.candidate_snapshot))},
            "validation_script": {"path": rel(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        },

        "frozen_inputs_rule": ("Pass 2a, 2b and 2c stay frozen: every case below replays from the stored "
                               "artifact named in it, at the pinned sha256. Session 6 may not re-run 2a/2b/2c, "
                               "change any prompt, change model routing, or add or drop a case. Any such change "
                               "invalidates this manifest and requires a new one."),

        "model_map": {
            "path": "benchmarks/configs/kind_canary_model_map.json",
            "sha256": sha256_file(MAIN_ROOT / "benchmarks" / "configs" / "kind_canary_model_map.json")
            if (MAIN_ROOT / "benchmarks" / "configs" / "kind_canary_model_map.json").is_file() else None,
            "environment": _dotenv_model_map(MAIN_ROOT),
            "rule": "identical on both arms; only the catalog artifacts differ",
        },
        "prompt_identity": next((c["detail"]["baseline"] for c in tier12["tier1_checks"]
                                 if c["id"] == "1.1e"), None) or {},
        "code_hashes": {f: git_blob(MAIN_ROOT, MAIN_ROOT / f) for f in MANIFEST_CODE_FILES},
        "embeddings": {
            "backend": cfg.EMBEDDINGS_BACKEND, "base_url": str(cfg.EMBEDDINGS_BASE_URL),
            "model": cfg.EMBEDDINGS_MODEL_NAME, "dimension": int(cfg.EMBEDDINGS_DIMENSION),
            "top_k_candidates": TOP_K,
            "shortcut_min_score": float(cfg.PASS_2D_SHORTCUT_MIN_SCORE),
            "shortcut_min_margin": float(cfg.PASS_2D_SHORTCUT_MIN_MARGIN),
            "probe": cand_snap["probe"],
            "rule": "probe with a real POST before any run; a 200 from /health is not readiness",
        },

        "stages_to_rerun": [
            {"stage": "pass_2d", "from": "frozen Pass 2c observations in each case below",
             "note": "resolution only; the candidate set and shortcut are already measured offline here"},
            {"stage": "condition_projection", "note": "rebuild from the arm's own catalog projection"},
            {"stage": "terra_verification", "note": "no stored checkpoint survives: both successors carry new "
                                                    "atomic_claims, so the projection fingerprint changed (CCF-8)"},
            {"stage": "disposition_and_routing"},
            {"stage": "v5_estimate_packages_totals",
             "note": "package and dollar deltas are reported for context, never scored as the primary result"},
        ],
        "stages_frozen": ["pass_1a", "pass_2a", "pass_2b", "pass_2c", "pass_2e"],

        "metrics": {
            "primary": ["resolved_catalog_item_id per case", "terra_verdict and rationale",
                        "final disposition and accepted/rejected state", "terminal route and reason_code",
                        "error migration into neighbouring items or stages"],
            "secondary": ["package membership and totals (context only, never the primary result)",
                          "count of no_action blinds conditions rendering at zero dollars (A-4)"],
            "comparison_schema": {
                "per_case": ["case_id", "arm", "resolved_item_id", "candidate_ids", "terra_verdict",
                             "disposition", "route", "reason_code", "billed_amount"],
                "aggregate": ["reachability", "successful_use_preservation", "correct_rejection_preservation",
                              "neighbour_hijacking", "conditions_moving_to_no_action",
                              "conditions_surviving_on_fabric", "conditions_reaching_neither"],
            },
            "offline_expectations_to_confirm": {
                "conditions_total": census.get("conditions_total"),
                "moving_to_no_action": census.get("moving_to_no_action"),
                "surviving_on_fabric_successor": census.get("surviving_on_fabric_successor"),
                "reaching_neither": census.get("reaching_neither"),
                "reachability_measured_offline": tier12["tier2_metrics"]["reachability_audit"]["by_bucket"],
                "blinds_shortcut_fire_rate": {
                    "baseline": tier12["tier2_metrics"]["blinds_shortcut"]["baseline_fire_rate"],
                    "candidate": tier12["tier2_metrics"]["blinds_shortcut"]["candidate_fire_rate"],
                    "denominator": tier12["tier2_metrics"]["blinds_shortcut"]["denominator"]},
            },
            "unmeasured_by_design": ["the three predicted Terra flips to supported on the presence-only claim "
                                     "(rc_4f4b93c40ef3, rc_ffad088fa5fe, rc_a22a241b3bf0): unbillable on every path"],
        },

        "ordering": {
            "stage_a": "targeted cases below, grouped by property and run; findings first, then controls, "
                       "then the remaining affected conditions",
            "stage_b": "the full frozen 18-property canary, ONLY if Stage A passes and its budget is "
                       "authorized separately",
            "full_canary_authorization_required": True,
            "full_canary_manifest": "configs/kind_ontology_canary_manifest.json",
        },

        "budget": {
            "stage_a_authorized_tokens": None,
            "stage_b_authorized_tokens": None,
            "authorized_by": None,
            "authorized_at": None,
            "reference_costs": {
                "full_18_property_replica": "~3.98M tokens (historical)",
                "single_condition_terra_call": "~3.2k tokens (gate live check: 23 calls, 72,756 tokens)",
                "stage_a_case_count": len(cases),
            },
            "rule": "Session 6 must not make a provider call until a human fills these fields in.",
        },

        "stop_conditions": [
            "targeted behaviour does not improve as predicted",
            "a reviewed success or correct rejection regresses",
            "a new false positive or mapping hijack appears",
            "the result sits inside the noise band without an authorized repeat",
            "an experiment invariant changes (catalog sha, projection fingerprint, prompt, model, case set)",
            "the authorized budget boundary is reached",
        ],

        "invariants_to_verify_before_any_call": [
            f"baseline HEAD == {BASELINE_COMMIT} and candidate HEAD == {CANDIDATE_COMMIT}",
            "both arms tracked-clean",
            "catalog sha256 and projection fingerprint match the arms block above",
            "prompt versions and shas match prompt_identity above",
            "code_hashes match (git blob, so line endings do not matter)",
            "the embeddings sidecar answers a real POST at dimension 1024",
            "ISSUE_CATALOG_PATH is unset; KIND_ONTOLOGY_VERSION=observation_kind_v2",
        ],

        "carried_findings": tier12.get("findings", []),
        "stale_id_inventory": stale_id_inventory(evidence),
        "benchmark_gold_drafts": gold_case_drafts(tier12, corpus),
        "cases": cases,
        "case_count": len(cases),
        "photo_set": {
            "root": str(photo_root) if photo_root else None,
            "pinned": sum(1 for c in cases if c["photo"]["status"] == "pinned"),
            "unresolved": sorted({f"{c['property_key']}/{c['photo']['photo_key']}"
                                  for c in cases if c["photo"]["status"] != "pinned"}),
        },
    }
    manifest["manifest_sha256"] = sha256_canonical(manifest)
    out = Path(args.out).resolve()
    sha = write_json(out, manifest)
    print(f"manifest: {len(cases)} targeted cases, "
          f"{len(manifest['carried_findings'])} carried findings, budget fields left null")
    print(f"  photos pinned: {manifest['photo_set']['pinned']}/{len(cases)}"
          + (f" (unresolved: {len(manifest['photo_set']['unresolved'])})"
             if manifest['photo_set']['unresolved'] else ""))
    print(f"  stale-id inventory: {manifest['stale_id_inventory']['canary_files_total']} stored artifacts "
          f"name the deprecated parent")
    print(f"wrote {rel(out)} sha256={sha}")
    return 0


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm-root", default=str(MAIN_ROOT),
                   help="checkout whose tools/ and catalog are used (pre-parsed onto sys.path)")
    p.add_argument("--manifest", default=None,
                   help="pin manifest identifying the candidate under validation; "
                        "omit for the Session 5/6 CAP-007 pins")
    sub = p.add_subparsers(dest="command", required=True)

    t1 = sub.add_parser("tier1", help="deterministic validation across both arms")
    t1.add_argument("--candidate-root", required=True)
    t1.add_argument("--out", default=str(MAIN_ROOT / "reports" / "catalog_audit_tier1.json"))
    t1.set_defaults(func=cmd_tier1)

    co = sub.add_parser("corpus", help="frozen Pass 2c observation set for the retrieval replay")
    co.add_argument("--out", default=str(MAIN_ROOT / "reports" / "catalog_audit_replay_corpus.json"))
    co.set_defaults(func=cmd_corpus)

    pr = sub.add_parser("probe", help="real POST to the embeddings sidecar")
    pr.set_defaults(func=cmd_probe)

    sn = sub.add_parser("snapshot", help="per-arm retrieval replay")
    sn.add_argument("--arm", required=True, choices=("baseline", "candidate"))
    sn.add_argument("--corpus", default=str(MAIN_ROOT / "reports" / "catalog_audit_replay_corpus.json"))
    sn.add_argument("--cache", default=str(Path(os.environ.get("TEMP", ".")) / "catalog_audit_vector_cache.json"))
    sn.add_argument("--out", required=True)
    sn.set_defaults(func=cmd_snapshot)

    cp = sub.add_parser("compare", help="join both snapshots and evaluate the Tier 2 rules")
    cp.add_argument("--corpus", default=str(MAIN_ROOT / "reports" / "catalog_audit_replay_corpus.json"))
    cp.add_argument("--baseline", default=str(MAIN_ROOT / "reports" / "catalog_audit_replay_snapshot_baseline.json"))
    cp.add_argument("--candidate", default=str(MAIN_ROOT / "reports" / "catalog_audit_replay_snapshot_candidate.json"))
    cp.add_argument("--tier1", default=str(MAIN_ROOT / "reports" / "catalog_audit_tier1.json"))
    cp.add_argument("--out", default=str(MAIN_ROOT / "reports" / "catalog_audit_validation_tier12.json"))
    cp.set_defaults(func=cmd_compare)

    mf = sub.add_parser("manifest", help="pinned live-experiment manifest for Session 6")
    mf.add_argument("--candidate-root", required=True)
    mf.add_argument("--tier12", default=str(MAIN_ROOT / "reports" / "catalog_audit_validation_tier12.json"))
    mf.add_argument("--corpus", default=str(MAIN_ROOT / "reports" / "catalog_audit_replay_corpus.json"))
    mf.add_argument("--baseline-snapshot",
                    default=str(MAIN_ROOT / "reports" / "catalog_audit_replay_snapshot_baseline.json"))
    mf.add_argument("--candidate-snapshot",
                    default=str(MAIN_ROOT / "reports" / "catalog_audit_replay_snapshot_candidate.json"))
    mf.add_argument("--photo-root",
                    default="C:/Users/Steven/IntelliJProjects/renointel-prod/public/images/properties",
                    help="review photo root, hashed into the manifest so the image set is pinned too")
    mf.add_argument("--out", default=str(MAIN_ROOT / "reports" / "catalog_audit_live_experiment_manifest.json"))
    mf.set_defaults(func=cmd_manifest)

    return p


def cmd_probe(args: argparse.Namespace) -> int:
    record = probe_sidecar()
    print(json.dumps(record, indent=2))
    return 0 if record.get("ok") else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    apply_pins(getattr(args, "manifest", None))
    try:
        return int(args.func(args))
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
