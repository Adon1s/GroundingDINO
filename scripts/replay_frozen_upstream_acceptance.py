"""Frozen-observation acceptance through the production orchestrator and writer.

The zero-cost `prove` command must succeed before live execution is permitted.
It includes prior null selections and compares every lane and Terra fingerprint.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import asdict
import hashlib
import json
import logging
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace, FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None

from scripts.analysis.terra_miss_mechanisms import CANARY, PRODUCTION, read, write, relative, provenance
from tools import pipeline_config as cfg, review_cards as rc
from tools.pass_2f_artifact_inputs import photo_key_to_path
from tools.scene_classifier_orchestrator import SceneClassifierOrchestrator
from tools.pass_config import SceneClassifierRunOptions, PassToggles
from tools.artifact_writers import write_photo_intel
from tools.renovation_architecture.runtime import initialize_renovation_architecture

LANES = ("issues_flat", "estimate_issues_flat", "product_issues_flat", "product_estimate_issues_flat")
DEFAULT_OUT = ROOT / "artifacts_canary/backend_acceptance_20260910"
CATALOG_SHA = "787964013d083368403524c610df54cc9858a8f6c1a1e5c9535c7204c1e67a2f"


class FrozenInputGap(ValueError):
    pass


class NoModelCalls:
    def __getattr__(self, name):
        if name.startswith("analyze_"):
            raise AssertionError(f"model call forbidden in Tier 0: {name}")
        raise AttributeError(name)


def guard_out_root(path):
    path = Path(path).resolve()
    for protected in (CANARY.resolve(), PRODUCTION.resolve()):
        if path == protected or path.is_relative_to(protected) or protected.is_relative_to(path):
            raise ValueError(f"refusing protected/ancestor output root: {path}")
    if path == ROOT or not path.is_relative_to(ROOT / "artifacts_canary"):
        raise ValueError("acceptance outputs must be isolated under artifacts_canary")
    return path


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def reconstruct_photo(photo, photo_key):
    """Recover the full 2c lane, checking independent persisted counts and ids.

    2e survivors, removals and suppressed samples are the input spine. Resolved
    records supply original order and text, and include LLM null selections.
    Any unpreserved row stops the run; no invented or silently omitted inputs.
    """
    issues = photo["issues"]
    telemetry = photo["_pass_2e_telemetry"]
    debug = photo["debug"]
    union = {}
    for lane in (issues.get("matched", []), issues.get("final", []),
                 issues.get("removed", []), telemetry.get("suppressed_samples", [])):
        for row in lane:
            if not row.get("issue_id"):
                raise FrozenInputGap(f"{photo_key}: row has no issue_id")
            union.setdefault(row["issue_id"], row)
    expected = telemetry["input_count"]
    if len(union) != expected:
        raise FrozenInputGap(f"{photo_key}: recoverable {len(union)} of {expected} 2c observations")
    resolutions = debug.get("resolved_items", [])
    ordered_ids = [r["issue_id"] for r in resolutions]
    if len(ordered_ids) != len(set(ordered_ids)) or not set(ordered_ids) <= union.keys():
        raise FrozenInputGap(f"{photo_key}: resolution ids do not join uniquely to 2e inputs")
    ordered_ids += [iid for iid in union if iid not in set(ordered_ids)]
    by_id = {r["issue_id"]: r for r in resolutions}
    observations = []
    for iid in ordered_ids:
        row = copy.deepcopy(union[iid])
        for field in ("catalogItemId", "catalogItemKind", "source_photo_id", "removed_reason", "suppressed_reason", "reason"):
            row.pop(field, None)
        row.setdefault("source_photo_key", photo_key)
        if iid in by_id:
            row["description"] = by_id[iid]["description"]
            row["kind"] = by_id[iid]["original_kind"]
        if not row.get("description") or row.get("kind") not in ("defect", "degradation", "modernization"):
            raise FrozenInputGap(f"{photo_key}/{iid}: description or kind not preserved")
        observations.append(row)
    return {"observations": observations, "excluded": debug.get("excluded_observations", [])}, {
        iid: by_id.get(iid) for iid in ordered_ids
    }


def frozen_meta(photo, key, run_id, *, freeze_2d):
    observations, resolutions = reconstruct_photo(photo, key)
    meta = {"run_id": run_id, "photo_key": key,
            "pass_1a_frozen_scene": photo["scene"]["id"],
            "pass_2a_frozen_freeform": photo["features"]["observations_freeform"],
            "pass_2b_frozen_struct": photo["debug"]["observations_struct"],
            "pass_2c_frozen_observations": observations}
    if freeze_2d:
        meta["pass_2d_frozen_resolutions"] = resolutions
    return meta


async def replay_photos(artifact, *, catalog, orchestrator, run_id, freeze_2d, checkpoint_dir=None):
    # SimpleNamespace satisfies the same ImageResult protocol as the worker.
    results, records = [], []
    paths = photo_key_to_path(artifact)
    for key, photo in artifact["photos"].items():
        meta = frozen_meta(photo, key, run_id, freeze_2d=freeze_2d)
        options = SceneClassifierRunOptions(
            pipeline_mode="publish", toggles=PassToggles(pass_2f=False), meta=meta,
        )
        fingerprint = hashlib.sha256(canonical_bytes({'meta': meta, 'catalog': catalog,
            'model': orchestrator.qwen_config, 'temperature': 0.1})).hexdigest()
        ckpt = Path(checkpoint_dir) / (key + '.json') if checkpoint_dir else None
        if ckpt and ckpt.exists():
            saved = read(ckpt)
            if saved['input_fingerprint'] != fingerprint:
                raise ValueError(f'{key}: checkpoint input drift')
            data, record = saved['analysis'], saved['record']
        else:
            analysis = await orchestrator.analyze_image(image_path=paths[key], options=options)
            data = analysis.to_dict()
            record = {"photo_key": key, "observations": analysis.observations,
                      "resolutions": analysis.resolved_items,
                      "per_observation": analysis.debug.get("pass_2d_per_observation", [])}
            if ckpt:
                from tools.artifact_writers import _write_json_atomic
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                _write_json_atomic(ckpt, {'input_fingerprint': fingerprint, 'analysis': data, 'record': record})
                print(f"Local 2d {artifact['property']['property_key']} {key}: {len(record['observations'])} observations", flush=True)
        results.append(SimpleNamespace(image_path=str(paths[key]), scene_data=data,
                                      scene_classifier=None, scene=data.get('scene'), error=None, processing_time=0))
        records.append(record)
    return results, records


def write_property(artifact, *, results, out, catalog, client, writer=write_photo_intel):
    prop = artifact["property"]["property_key"]
    run_id = artifact["run"]["run_id"]
    job = SimpleNamespace(property_key=prop, job_id=run_id, artifacts_dir=str(out),
                          timestamp=artifact["run"].get("timestamp"), results=results,
                          total_processing_time=0, property_metadata=artifact.get("property_metadata"))
    out.mkdir(parents=True, exist_ok=True)
    return writer(cfg=cfg, job=job, detection_backend="dinox", analysis_profile="standard",
        use_pass_architecture=True, pass_toggles=PassToggles(pass_2f=False).to_dict(),
        model_overrides={}, gpt_config={}, issue_catalog=catalog, vlm_client=client,
        dependency_status={"embeddings": "frozen" if isinstance(client, NoModelCalls) else "ready"})


def historical_projection_builder():
    """Load the original production projection/validator privately for Tier 0.

    Current production correctly refuses 3.1. Never relax its version gates.
    These immutable git sources are used only to rebuild historical requests.
    """
    modules = {}
    for name, path in (("contracts", "tools/renovation_architecture/contracts.py"),
                       ("validation", "tools/catalog_validation.py"),
                       ("projection", "tools/renovation_architecture/catalog_projection.py")):
        source = subprocess.check_output(["git", "show", f"07ee112:{path}"], cwd=ROOT)
        module_name = f"_acceptance_historical_{name}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(ROOT / path)
        sys.modules[module_name] = module
        exec(compile(source, f"07ee112:{path}", "exec"), module.__dict__)
        modules[name] = module
    projection = modules["projection"]
    projection.REQUIRED_CATALOG_VERSION = modules["contracts"].REQUIRED_CATALOG_VERSION
    projection.PROJECTION_VERSION = modules["contracts"].PROJECTION_VERSION
    projection.validate_issue_catalog = modules["validation"].validate_issue_catalog
    return projection.build_renovation_catalog_projection


def historical_manifest_writer(out):
    """Same production writer, privately bound to the historical manifest gate.

    Today's gate must reject the retired CAP-007 parent in real output. The
    historical proof must validate it under the manifest that originally ran.
    No production module globals are changed and no validation is skipped.
    """
    source = subprocess.check_output(["git", "show", "07ee112:tools/publication_gate.py"], cwd=ROOT)
    namespace = {"__file__": str(ROOT / "tools/publication_gate.py")}
    exec(compile(source, "07ee112:tools/publication_gate.py", "exec"), namespace)
    manifest_path = out / "historical_migration_manifest.json"
    manifest_path.write_bytes(subprocess.check_output(
        ["git", "show", "07ee112:tools/catalog_migrations/2.1_to_3.0.json"], cwd=ROOT))
    namespace["_MANIFEST_PATH"] = manifest_path
    writer = FunctionType(write_photo_intel.__code__, {
        **write_photo_intel.__globals__, "validate_publication_payload": namespace["validate_publication_payload"]
    }, write_photo_intel.__name__, write_photo_intel.__defaults__)
    writer.__kwdefaults__ = write_photo_intel.__kwdefaults__
    return writer


def project_requests(artifact, catalog, catalog_path, *, projection_builder=None):
    from tools.renovation_architecture.catalog_projection import build_renovation_catalog_projection
    from tools.renovation_architecture.conditions import build_observed_conditions
    from tools.renovation_architecture.evidence import build_photo_identity_index, build_evidence_facts
    from tools.renovation_architecture.ids import make_estimate_id
    from tools.renovation_architecture.terra_review import build_unit_request
    projection = (projection_builder or build_renovation_catalog_projection)(catalog, catalog_path=catalog_path)
    estimate_id = make_estimate_id(property_key=artifact["property"]["property_key"],
        source_run_id=artifact["run"]["run_id"], catalog_sha256=projection["catalog_sha256"],
        projection_fingerprint=projection["fingerprint"])
    lane = artifact["product_estimate_issues_flat"] if artifact["estimate_issues_flat"] else artifact["product_issues_flat"]
    drafts = build_observed_conditions(issues_flat=lane, photos=artifact["photos"],
        property_metadata=artifact.get("property_metadata"), projection=projection, estimate_id=estimate_id)
    paths = photo_key_to_path(artifact)
    needed = sorted({ref["photo_key"] for d in drafts for ref in d.evidence_refs})
    identities = build_photo_identity_index(needed, paths)
    observables = projection["observables"]
    pairs = [(d, build_evidence_facts(d, identity_index=identities, observables=observables, estimate_id=estimate_id)) for d in drafts]
    requests = []
    for unit in sorted({d.condition.estimate_unit_id for d in drafts}):
        request = build_unit_request(estimate_unit_id=unit,
            unit_pairs=[p for p in pairs if p[0].condition.estimate_unit_id == unit],
            observables=observables, photo_key_to_path=paths, identity_index=identities,
            projection_fingerprint=projection["fingerprint"], model="gpt-5.6-terra",
            reasoning_effort="medium", max_output_tokens=8192)
        requests.append({"unit": unit, "condition_ids": list(request.condition_ids),
                         "fingerprint": request.request_fingerprint,
                         "request_bytes": request.request_bytes, "photo_count": len(request.photo_keys)})
    return {"observed_conditions": [asdict(d.condition) for d in drafts],
            "evidence_facts": [asdict(e) for _, e in pairs], "requests": requests}


def diff_values(left, right, path="", limit=8):
    if left == right:
        return []
    if type(left) != type(right):
        return [{"path": path, "expected": left, "actual": right}]
    out = []
    if isinstance(left, dict):
        for key in sorted(left.keys() | right.keys()):
            out.extend(diff_values(left.get(key), right.get(key), f"{path}/{key}", limit))
            if len(out) >= limit:
                break
    elif isinstance(left, list):
        if len(left) != len(right):
            out.append({"path": path + "/length", "expected": len(left), "actual": len(right)})
        for i, (a, b) in enumerate(zip(left, right)):
            out.extend(diff_values(a, b, f"{path}/{i}", limit))
            if len(out) >= limit:
                break
    else:
        out.append({"path": path, "expected": left, "actual": right})
    return out[:limit]


def prove(out_root, properties=None):
    out_root = guard_out_root(out_root)
    raw = subprocess.check_output(["git", "show", "07ee112:tools/issue_catalog_kind_v2.json"], cwd=ROOT)
    frozen_sha = read(CANARY / "input_freeze.json")["catalog_sha256"]
    if hashlib.sha256(raw).hexdigest() != frozen_sha:
        # Git stores LF; the pinned Windows source file used CRLF.
        raw = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if hashlib.sha256(raw).hexdigest() != frozen_sha:
        raise FrozenInputGap("historical catalog bytes do not match input_freeze SHA")
    catalog = json.loads(raw)
    catalog_path = out_root / "tier0/catalog_3_1.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve historical file bytes: the catalog sha participates in ids.
    catalog_path.write_bytes(raw)
    initialize_renovation_architecture(mode="current", catalog=catalog, catalog_path=catalog_path,
                                      kind_ontology_version="observation_kind_v2")
    client = NoModelCalls()
    orchestrator = SceneClassifierOrchestrator(qwen_config={"provider": "lmstudio", "model": "frozen"},
        gpt5_config={}, vlm_client=client, catalog_items=catalog["items"], max_resolve_per_image=10000)
    runs, _ = rc.load_canary(CANARY)
    report = {"schema_version": 1, "model_calls": 0, "properties": [],
              "catalog_sha256": hashlib.sha256(raw).hexdigest()}
    historical_builder = historical_projection_builder()
    historical_writer = historical_manifest_writer(catalog_path.parent)
    for prop, (source, artifact) in sorted(runs.items()):
        if properties and prop not in properties:
            continue
        row = {"property_key": prop, "source": provenance([source])[0]}
        try:
            results, records = asyncio.run(replay_photos(artifact, catalog=catalog, orchestrator=orchestrator,
                run_id=artifact["run"]["run_id"], freeze_2d=True))
            dest = out_root / "tier0" / prop / artifact["run"]["run_id"]
            write_property(artifact, results=results, out=dest, catalog=catalog, client=client, writer=historical_writer)
            replay = read(dest / "photo_intel_debug.json")
            write(dest / "resolution_record.json", records)
            row["lanes"] = {lane: {"equal": canonical_bytes(artifact[lane]) == canonical_bytes(replay[lane]),
                                   "diff": diff_values(artifact[lane], replay[lane])} for lane in LANES}
            if all(v["equal"] for v in row["lanes"].values()):
                rebuilt = project_requests(replay, catalog, catalog_path, projection_builder=historical_builder)
                stored = rc.v5_result(artifact)
                for key in ("observed_conditions", "evidence_facts"):
                    a = sorted(stored[key], key=lambda r: r["condition_id"])
                    b = sorted(rebuilt[key], key=lambda r: r["condition_id"])
                    # Dataclass tuples serialize as lists; compare serialized values.
                    row[key] = {"equal": canonical_bytes(a) == canonical_bytes(b),
                                "diff": diff_values(a, json.loads(canonical_bytes(b)))}
                old = {r["estimate_unit_id"]: r["request_fingerprint"] for r in stored["terra_calls"]}
                new = {r["unit"]: r["fingerprint"] for r in rebuilt["requests"]}
                row["terra_fingerprints"] = {"equal": old == new, "diff": diff_values(old, new)}
            row["passed"] = all(v["equal"] for v in row["lanes"].values()) and all(
                row.get(k, {}).get("equal", False) for k in ("observed_conditions", "evidence_facts", "terra_fingerprints"))
            row["observation_count"] = sum(len(r["observations"]) for r in records)
            row["null_resolutions"] = sum(not r.get("resolved_item_id") for record in records for r in record["resolutions"])
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        report["properties"].append(row)
        print(f"Tier 0 {prop}: {'PASS' if row['passed'] else row.get('error', 'DIFF')}", flush=True)
    report["passed"] = len(report["properties"]) == 18 and all(r["passed"] for r in report["properties"])
    write(out_root / "tier0_proof.json", report)
    return report


PINNED_MODELS = {'2d': 'unsloth/qwen3.6-27b@q6_k', 'terra': 'gpt-5.6-terra', 'sol': 'gpt-5.6-sol'}
LOCAL_MODEL_URL = 'http://127.0.0.1:1234'
CATALOG_PATH = ROOT / 'tools/issue_catalog_kind_v2.json'


def runtime_sources():
    paths = list((ROOT / 'tools').rglob('*.py'))
    paths += [Path(__file__)]
    return provenance(p for p in paths if p.is_file())


def verify_inputs(out_root):
    out_root = guard_out_root(out_root)
    runs, _ = rc.load_canary(CANARY)
    proof = read(out_root / 'tier0_proof.json')
    problems = []
    if len(runs) != 18 or not proof.get('passed') or len(proof.get('properties', [])) != 18:
        problems.append('complete 18-property Tier 0 proof required')
    expected = {r['property_key']: r['source'] for r in proof.get('properties', [])}
    for prop, (path, artifact) in runs.items():
        if expected.get(prop) != provenance([path])[0]:
            problems.append(f'{prop}: source differs from Tier 0 proof')
        for key, photo in artifact['photos'].items():
            reconstruct_photo(photo, key)
    catalog_sha = hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()
    if catalog_sha != CATALOG_SHA:
        problems.append('current catalog differs from approved 3.2 checkpoint')
    freeze = read(CANARY / 'input_freeze.json')
    images = []
    for prop in sorted(runs):
        for row in freeze['properties'][prop]['images']:
            path = Path(row['path'])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                problems.append(f'{prop}: image drift: {path.name}')
            images.append(row)
    report = {'passed': not problems, 'problems': problems, 'properties': len(runs),
        'images': len(images), 'catalog_sha256': catalog_sha,
        'sources': provenance([path for path, _ in runs.values()]),
        'image_freeze': provenance([CANARY / 'input_freeze.json'])[0], 'runtime_sources': runtime_sources()}
    write(out_root / 'verification.json', report)
    return report


def plan(out_root):
    out_root = guard_out_root(out_root)
    verified = verify_inputs(out_root)
    if not verified['passed']:
        return verified
    runs, second = rc.load_canary(CANARY)
    properties = []
    for prop, (_, artifact) in sorted(runs.items()):
        historical = [rc.v5_result(artifact), rc.v5_result(second[prop][1])]
        terra = max(v['terra_listing_usage']['budget_debited_tokens'] for v in historical)
        sol = max(v['sol_listing_usage']['total_tokens'] for v in historical)
        properties.append({'property_key': prop, 'source_run_id': artifact['run']['run_id'],
            'terra_forecast': int(terra * 1.05 + 0.999), 'sol_forecast': int(sol * 1.05 + 0.999),
            'source_terra_calls': len(historical[0]['terra_calls']),
            'source_request_bytes': sum(r.get('request_bytes', 0) for r in historical[0]['terra_calls'])})
    manifest = {'schema_version': 1, 'approval': 'Steven approved the recommendations in this task.',
        'models': PINNED_MODELS, 'local_model_url': LOCAL_MODEL_URL,
        'embeddings': {k: getattr(cfg, k) for k in ('EMBEDDINGS_MODEL_NAME', 'EMBEDDINGS_BASE_URL', 'EMBEDDINGS_BACKEND', 'EMBEDDINGS_DIMENSION', 'EMBEDDINGS_TOPK')},
        'pass_2d_temperature': 0.1, 'corroboration_gate': 'off',
        'freeze_passes': ['1a', '2a', '2b', '2c'], 'live_passes': ['2d', '2e', 'Terra', 'Sol'],
        'batch_terra_ceiling': 2_000_000, 'sol_daily_ceiling': 250_000,
        'usage_root': str(out_root / 'usage'), 'replicas': 2, 'different_utc_days': True,
        'verification': verified, 'properties': properties,
        'terra_forecast_per_replica': sum(p['terra_forecast'] for p in properties),
        'sol_forecast_per_replica': sum(p['sol_forecast'] for p in properties),
        'forecast_method': 'Per property max of stored two replicas, plus 5%; fresh local 2d request counts checked before paid calls.',
        'reserved_batch_2': 'Unspent; no experimental authority granted.'}
    identity = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    dest = out_root / 'manifest.json'
    if dest.exists():
        previous = read(dest)
        if previous['identity'] != identity:
            raise ValueError('manifest drift: use a separately reviewed plan; never overwrite an existing batch boundary')
        return previous
    manifest.update(identity=identity, batch_start=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    passed=manifest['terra_forecast_per_replica'] <= 1_000_000)
    write(dest, manifest)
    return manifest


def load_manifest(out_root):
    manifest = read(out_root / 'manifest.json')
    verified = verify_inputs(out_root)
    if not verified['passed'] or verified != manifest['verification']:
        raise ValueError('pinned inputs/code changed; no live calls permitted')
    if not manifest['passed']:
        raise ValueError('forecast over 1M/replica: replica-2 selection must be recorded before live execution')
    return manifest


def configure(manifest, *, mode):
    cfg.RENOVATION_ARCHITECTURE_MODE = mode
    cfg.RENOVATION_TERRA_MODEL = PINNED_MODELS['terra']
    cfg.RENOVATION_SOL_MODEL = PINNED_MODELS['sol']
    cfg.RENOVATION_TERRA_USAGE_ROOT = manifest['usage_root']
    cfg.RENOVATION_VLM_BUDGET_GUARD = True
    cfg.RENOVATION_SOL_DAILY_TOKEN_CEILING = 250_000
    cfg.PASS_2D_TEMPERATURE = 0.1
    cfg.LM_STUDIO_MODEL = PINNED_MODELS['2d']
    cfg.LM_STUDIO_URL = manifest['local_model_url']
    for key, value in manifest['embeddings'].items():
        setattr(cfg, key, value)
    os.environ['RENOVATION_TERRA_BATCH_START'] = manifest['batch_start']
    os.environ['RENOVATION_TERRA_BATCH_TOKEN_CEILING'] = str(manifest['batch_terra_ceiling'])
    from tools.renovation_architecture.usage_guard import TerraUsageLedger
    spent = TerraUsageLedger(Path(manifest['usage_root'])).spent_since(manifest['batch_start'])
    cfg.RENOVATION_TERRA_DAILY_TOKEN_CEILING = min(2_500_000, manifest['batch_terra_ceiling'] - spent)
    catalog = read(CATALOG_PATH)
    initialize_renovation_architecture(mode=mode, catalog=catalog, catalog_path=CATALOG_PATH,
        kind_ontology_version='observation_kind_v2', terra_model=PINNED_MODELS['terra'], sol_model=PINNED_MODELS['sol'])
    return catalog


def prepare(out_root, replica, properties=None):
    """Local-only stage. No paid client is passed to the artifact writer."""
    from tools.catalog_embeddings import build_candidate_provider
    from tools.vlm_client import create_vlm_client
    out_root = guard_out_root(out_root)
    manifest = load_manifest(out_root)
    catalog = configure(manifest, mode='current')
    provider = build_candidate_provider(catalog)  # real embeddings POST preflight
    client = create_vlm_client()
    orchestrator = SceneClassifierOrchestrator(
        qwen_config={'provider': 'lmstudio', 'url': cfg.LM_STUDIO_URL,
                     'model': PINNED_MODELS['2d'], 'temperature': 0.1},
        gpt5_config={}, vlm_client=client, candidate_provider=provider,
        catalog_items=catalog['items'], max_resolve_per_image=10000)
    runs, _ = rc.load_canary(CANARY)
    for prop, (_, artifact) in sorted(runs.items()):
        if properties and prop not in properties:
            continue
        dest = out_root / f'replica_{replica}' / prop / artifact['run']['run_id']
        results, records = asyncio.run(replay_photos(artifact, catalog=catalog, orchestrator=orchestrator,
            run_id=artifact['run']['run_id'], freeze_2d=False, checkpoint_dir=dest / '.photos'))
        write_property(artifact, results=results, out=dest / 'prepared', catalog=catalog, client=NoModelCalls())
        from tools.artifact_writers import _write_json_atomic
        _write_json_atomic(dest / 'resolution_record.json', records)
        projected = project_requests(read(dest / 'prepared/photo_intel_debug.json'), catalog, CATALOG_PATH)
        write(dest / 'prepared_requests.json', projected)
        write(dest / 'prepared_complete.json', {'manifest_identity': manifest['identity'],
            'files': provenance([dest / 'resolution_record.json', dest / 'prepared_requests.json',
                                 dest / 'prepared/photo_intel_debug.json',
                                 *sorted((dest / '.photos').glob('*.json'))])})
    return {'passed': True, 'paid_calls': 0}


def assert_worker_idle():
    """Fail closed if the same-day production-worker precondition is unknown."""
    import psutil
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process.info['pid'] == os.getpid():
                continue
            name = (process.info['name'] or '').lower()
            if not any(s in name for s in ('python', 'node')):
                continue
            command = ' '.join(process.info['cmdline'] or []).lower()
            if any(s in command for s in ('analyzer_server.py', 'worker_server.py', 'worker_main.py', 'run_worker_smoke.py', 'analyze_property.py')):
                raise RuntimeError(f'Production worker must be idle: relevant process pid={process.info["pid"]}')
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            raise RuntimeError('Cannot establish production worker idle; no paid calls permitted')


def run_replica(out_root, replica, properties=None):
    from tools.vlm_client import create_vlm_client
    from tools.renovation_architecture.usage_guard import TerraUsageLedger, SolUsageLedger
    from scripts.verify_renovation_artifact import main as verify_artifact
    out_root = guard_out_root(out_root)
    manifest = load_manifest(out_root)
    assert_worker_idle()
    day = datetime.now(timezone.utc).date().isoformat()
    previous_path = out_root / f'replica_{3-replica}' / 'execution.json'
    if previous_path.exists() and day in read(previous_path)['utc_days']:
        raise RuntimeError('The two replicas must execute on different UTC days for the Sol quota')
    catalog = configure(manifest, mode='new')
    if not cfg.OPENAI_API_KEY:
        raise RuntimeError('Missing OpenAI credentials')
    terra = TerraUsageLedger(Path(manifest['usage_root']))
    sol = SolUsageLedger(Path(manifest['usage_root']))
    execution_path = out_root / f'replica_{replica}' / 'execution.json'
    execution = read(execution_path) if execution_path.exists() else {'utc_days': [], 'properties': []}
    if day not in execution['utc_days']:
        execution['utc_days'].append(day)
    write(execution_path, execution)
    runs, _ = rc.load_canary(CANARY)
    client = create_vlm_client()
    for forecast in manifest['properties']:
        prop = forecast['property_key']
        if properties and prop not in properties:
            continue
        artifact = runs[prop][1]
        dest = out_root / f'replica_{replica}' / prop / artifact['run']['run_id']
        prepared = read(dest / 'prepared_complete.json')
        if prepared['manifest_identity'] != manifest['identity'] or any(
            provenance([ROOT / f['path']])[0] != f for f in prepared['files']):
            raise ValueError(f'{prop}: prepared checkpoint drift')
        if (dest / 'complete.json').exists():
            complete = read(dest / 'complete.json')
            if complete['manifest_identity'] != manifest['identity'] or complete['artifact'] != provenance([dest / 'photo_intel_debug.json'])[0]:
                raise ValueError(f'{prop}: completed artifact drift')
            continue
        projected = read(dest / 'prepared_requests.json')
        # New unit count increases invalidate the historical cost forecast.
        factor = max(1, len(projected['requests']) / max(1, forecast['source_terra_calls']))
        expected = int(forecast['terra_forecast'] * factor)
        if terra.spent_since(manifest['batch_start']) + expected > manifest['batch_terra_ceiling']:
            raise RuntimeError(f'{prop}: forecast would cross shared Terra batch cap')
        if sol.spent_today() + forecast['sol_forecast'] > 250_000:
            raise RuntimeError(f'{prop}: forecast would cross daily Sol cap')
        results = []
        for key, path in photo_key_to_path(artifact).items():
            saved = read(dest / '.photos' / (key + '.json'))
            data = saved['analysis']
            results.append(SimpleNamespace(image_path=str(path), scene_data=data,
                scene_classifier=None, scene=data.get('scene'), error=None, processing_time=0))
        write_property(artifact, results=results, out=dest, catalog=catalog, client=client)
        if verify_artifact(['--artifact', str(dest), '--expect-mode', 'new',
                '--expect-terra-model', PINNED_MODELS['terra'], '--expect-sol-model', PINNED_MODELS['sol']]):
            raise RuntimeError(f'{prop}: production artifact verifier failed')
        write(dest / 'complete.json', {'manifest_identity': manifest['identity'],
            'artifact': provenance([dest / 'photo_intel_debug.json'])[0], 'utc_day': day})
        execution['properties'].append(prop)
        write(execution_path, execution)
    return {'passed': True, 'completed_properties': execution['properties']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prove", "verify", "plan", "prepare", "run", "score"])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--properties", nargs="*")
    parser.add_argument('--replica', type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    if args.command == 'prove':
        report = prove(args.out_root, args.properties)
    elif args.command == 'verify':
        report = verify_inputs(args.out_root)
    elif args.command == 'plan':
        report = plan(args.out_root)
    elif args.command == 'prepare':
        report = prepare(args.out_root, args.replica, args.properties)
    elif args.command == 'run':
        report = run_replica(args.out_root, args.replica, args.properties)
    else:
        from scripts.analysis.backend_acceptance_score import score
        report = score(guard_out_root(args.out_root))
    print(json.dumps({k: v for k, v in report.items() if k in ('passed', 'problems',
        'terra_forecast_per_replica', 'sol_forecast_per_replica', 'completed_properties')}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
