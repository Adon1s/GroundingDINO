"""Verify one analyzer artifact against the cutover runbook (Session 6/9 §3/§4).

Usage:
  python scripts/verify_renovation_artifact.py --artifact <run dir or photo_intel.json> \
      --expect-mode {new,current,shadow} [--expect-terra-model M] [--expect-sol-model M]

Read-only. Needs NO environment: the publication gate and the envelope
validator are payload-only (they never consult pipeline_config), so the same
command verifies a production artifact, a smoke artifact, or a canary artifact.

Checks (both files: photo_intel.json = slim, photo_intel_debug.json = full):
  - root ontology/catalog stamps (observation-kind-v2 / 3.1);
  - tools.publication_gate.validate_publication_payload on the FULL payload (what
    the writer validated before writing) and on the slim one;
  - renovation_estimate_v4 present (version renovation_estimate_v4);
  - mode-specific placement of renovation_estimate_v5:
      new     -> at the root (both files), complete, validate_envelope ok,
                 provenance.architecture_mode == new, no private copy;
      current -> absent everywhere;
      shadow  -> only under analysis_debug in the debug file, complete;
  - prints the v5 Terra/Sol call models (optional asserts) and model_routing.
Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.publication_gate import validate_publication_payload  # noqa: E402
from tools.renovation_architecture.contracts import (  # noqa: E402
    REQUIRED_CATALOG_ONTOLOGY,
    REQUIRED_CATALOG_VERSION,
    REQUIRED_KIND_ONTOLOGY_SELECTOR,
    SHADOW_DEBUG_KEY,
)
from tools.renovation_architecture.validators import validate_envelope  # noqa: E402

V4_KEY = "renovation_estimate_v4"
Check = Tuple[bool, str, str]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog_for(stamp: Optional[str]) -> Tuple[Path, Dict[str, Any]]:
    name = "issue_catalog_kind_v2.json" if stamp == REQUIRED_CATALOG_ONTOLOGY else "issue_catalog.json"
    path = ROOT / "tools" / name
    return path, _load(path)


def _private_v5(debug: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not isinstance(debug, dict):
        return None
    analysis_debug = debug.get("analysis_debug")
    return analysis_debug.get(SHADOW_DEBUG_KEY) if isinstance(analysis_debug, dict) else None


def _check_envelope(checks: List[Check], env: Any, expected_mode: str,
                    expect_terra: Optional[str], expect_sol: Optional[str]) -> None:
    where = f"{SHADOW_DEBUG_KEY} ({expected_mode})"
    if not isinstance(env, dict):
        checks.append((False, f"{where} is an object", type(env).__name__))
        return
    validation = validate_envelope(env)
    checks.append((validation.ok, f"{where} validate_envelope ok",
                   "; ".join(validation.errors[:5])))
    checks.append((env.get("state") == "complete", f"{where} state == complete", repr(env.get("state"))))
    prov = env.get("provenance") if isinstance(env.get("provenance"), dict) else {}
    checks.append((prov.get("architecture_mode") == expected_mode,
                   f"{where} provenance.architecture_mode == {expected_mode}",
                   repr(prov.get("architecture_mode"))))
    checks.append((str(prov.get("catalog_version")) == REQUIRED_CATALOG_VERSION,
                   f"{where} provenance.catalog_version == {REQUIRED_CATALOG_VERSION}",
                   repr(prov.get("catalog_version"))))
    checks.append((prov.get("kind_ontology_selector") == REQUIRED_KIND_ONTOLOGY_SELECTOR,
                   f"{where} provenance.kind_ontology_selector == {REQUIRED_KIND_ONTOLOGY_SELECTOR}",
                   repr(prov.get("kind_ontology_selector"))))
    result = env.get("result") if isinstance(env.get("result"), dict) else {}
    terra_models = sorted({str(c.get("model")) for c in (result.get("terra_calls") or []) if isinstance(c, dict)})
    sol_models = sorted({str(c.get("model")) for c in (result.get("sol_calls") or []) if isinstance(c, dict)})
    print(f"  v5 terra_calls: n={len(result.get('terra_calls') or [])} models={terra_models}")
    print(f"  v5 sol_calls:   n={len(result.get('sol_calls') or [])} models={sol_models}")
    if expect_terra:
        checks.append((terra_models == [expect_terra], f"{where} terra model == {expect_terra}", repr(terra_models)))
    if expect_sol:
        checks.append((sol_models == [expect_sol], f"{where} sol model == {expect_sol}", repr(sol_models)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True, type=Path,
                        help="run directory or its photo_intel.json")
    parser.add_argument("--expect-mode", required=True, choices=("new", "current", "shadow"))
    parser.add_argument("--expect-terra-model", default=None)
    parser.add_argument("--expect-sol-model", default=None)
    args = parser.parse_args(argv)

    slim_path = args.artifact / "photo_intel.json" if args.artifact.is_dir() else args.artifact
    debug_path = slim_path.with_name("photo_intel_debug.json")
    if not slim_path.is_file():
        print(f"FAIL: {slim_path} not found")
        return 1
    slim = _load(slim_path)
    debug = _load(debug_path) if debug_path.is_file() else None

    checks: List[Check] = []
    checks.append((debug is not None, "photo_intel_debug.json present next to photo_intel.json", str(debug_path)))

    stamp = slim.get("ontology_version")
    catalog_path, catalog = _catalog_for(stamp)
    checks.append((stamp == REQUIRED_CATALOG_ONTOLOGY, f"root ontology_version == {REQUIRED_CATALOG_ONTOLOGY}", repr(stamp)))
    checks.append((str(slim.get("catalog_version")) == REQUIRED_CATALOG_VERSION,
                   f"root catalog_version == {REQUIRED_CATALOG_VERSION}", repr(slim.get("catalog_version"))))

    for label, payload in (("debug/full", debug), ("slim", slim)):
        if payload is None:
            continue
        try:
            validate_publication_payload(payload, catalog)
            checks.append((True, f"publication gate ({label}) against {catalog_path.name}", ""))
        except Exception as exc:  # noqa: BLE001 - the gate raises RuntimeError with the reason
            checks.append((False, f"publication gate ({label}) against {catalog_path.name}", str(exc)[:400]))

    v4 = slim.get(V4_KEY)
    checks.append((isinstance(v4, dict) and v4.get("version") == V4_KEY,
                   f"{V4_KEY} present (version {V4_KEY})",
                   repr(v4.get("version")) if isinstance(v4, dict) else repr(v4)))

    root_slim = slim.get(SHADOW_DEBUG_KEY)
    root_debug = debug.get(SHADOW_DEBUG_KEY) if isinstance(debug, dict) else None
    private = _private_v5(debug)
    mode = args.expect_mode
    if mode == "new":
        checks.append((isinstance(root_debug, dict), f"root {SHADOW_DEBUG_KEY} present (debug/full)", ""))
        checks.append((isinstance(root_slim, dict), f"root {SHADOW_DEBUG_KEY} present (slim)", ""))
        checks.append((private is None, f"no private analysis_debug.{SHADOW_DEBUG_KEY}", ""))
        _check_envelope(checks, root_debug if root_debug is not None else root_slim, "new",
                        args.expect_terra_model, args.expect_sol_model)
    elif mode == "current":
        checks.append((root_debug is None and root_slim is None, f"no root {SHADOW_DEBUG_KEY} (either file)", ""))
        checks.append((private is None, f"no private analysis_debug.{SHADOW_DEBUG_KEY}", ""))
    else:  # shadow
        checks.append((root_debug is None and root_slim is None, f"no root {SHADOW_DEBUG_KEY} (either file)", ""))
        checks.append((isinstance(private, dict), f"private analysis_debug.{SHADOW_DEBUG_KEY} present (debug/full)", ""))
        if isinstance(private, dict):
            _check_envelope(checks, private, "shadow", args.expect_terra_model, args.expect_sol_model)

    routing = slim.get("model_routing") or []
    if isinstance(routing, list) and routing:
        print("  model_routing:")
        for entry in routing:
            if isinstance(entry, dict):
                print(f"    {entry.get('pass')}: {entry.get('model')} "
                      f"effort={entry.get('reasoning_effort')} source={entry.get('source')}")

    failed = [c for c in checks if not c[0]]
    for ok, label, detail in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
    print(f"{'PASS' if not failed else 'FAIL'}: {len(checks) - len(failed)}/{len(checks)} checks, "
          f"mode={mode}, artifact={slim_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
