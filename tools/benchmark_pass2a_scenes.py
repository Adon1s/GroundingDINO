"""Frozen Pass 1a scene capture for the v3 benchmark lineage.

Runs Terra scene classification exactly once per manifest photo and freezes
the result (runs/v3/scene_capture/scenes.json). Variant runs replay the
captured scenes through the pass_1a_frozen_scene orchestrator hook, so every
variant/repeat shares identical scene and physical-room assignments, and the
per-photo provenance proves the route was the configured explicit OpenAI
override — never a silent local-model fallback.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from tools.benchmark_pass2a import (
    V3_DIR,
    _load_json,
    _utcnow,
    _write_json,
    manifest_photos,
)

SCENES_DIR = V3_DIR / "scene_capture"

REQUIRED_PROVIDER = "openai"
REQUIRED_SOURCE = "explicit_override"


def required_override(config: Dict[str, Any]) -> str:
    override = (config.get("model_overrides") or {}).get("1a")
    if not override:
        raise SystemExit(
            "scene capture requires an explicit '1a' entry in "
            "config.json model_overrides (the Terra routing contract)"
        )
    return str(override)


def _capture_core(capture: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The provenance subset the fingerprint hash covers (timestamps and
    free-text reasoning excluded so a resumed capture hashes identically)."""
    return {
        key: {
            "scene": rec.get("scene"),
            "image_sha256": rec.get("image_sha256"),
            "provider": rec.get("provider"),
            "model": rec.get("model"),
            "reasoning_effort": rec.get("reasoning_effort"),
        }
        for key, rec in sorted((capture.get("photos") or {}).items())
    }


def scene_capture_sha(capture: Dict[str, Any]) -> str:
    from tools.comparison_common import sha256_canonical
    return sha256_canonical(_capture_core(capture))


def assert_capture_routing(capture: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Variant-execution abort gate: every captured Pass 1a route must be the
    configured explicit OpenAI override."""
    override = required_override(config)
    bad = []
    for key, rec in sorted((capture.get("photos") or {}).items()):
        route = (rec.get("provider"), rec.get("model"), rec.get("source"))
        if route != (REQUIRED_PROVIDER, override, REQUIRED_SOURCE):
            bad.append(
                f"{key}: provider={rec.get('provider')!r} "
                f"model={rec.get('model')!r} source={rec.get('source')!r}"
            )
    if bad:
        raise SystemExit(
            f"scene capture routing check failed (need provider={REQUIRED_PROVIDER}, "
            f"model={override}, source={REQUIRED_SOURCE}):\n  " + "\n  ".join(bad)
        )


def load_scene_capture(config: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Load + fully re-validate scenes.json (photo set, image hashes, sha,
    routing). Every caller that replays scenes goes through this gate."""
    path = SCENES_DIR / "scenes.json"
    if not path.is_file():
        raise SystemExit(f"scene capture missing: {path} — run scene-capture first")
    capture = _load_json(path)
    photos = capture.get("photos") or {}
    expected = {
        f"{prop}/{photo['photo_key']}": photo["image_sha256"]
        for prop, photo in manifest_photos(manifest)
    }
    if set(photos) != set(expected):
        raise SystemExit(
            "scene capture photo set differs from manifest "
            f"(captured {len(photos)}, manifest {len(expected)}) — re-run scene-capture"
        )
    stale = sorted(
        key for key, rec in photos.items()
        if rec.get("image_sha256") != expected[key]
    )
    if stale:
        raise SystemExit(f"scene capture image hashes differ from manifest: {stale}")
    if capture.get("capture_sha256") != scene_capture_sha(capture):
        raise SystemExit(
            "scene capture sha mismatch — scenes.json was edited; re-run scene-capture"
        )
    assert_capture_routing(capture, config)
    return capture


def scenes_by_property(capture: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """{property_key: {photo_key: scene}} for run_matrix replay."""
    out: Dict[str, Dict[str, str]] = {}
    for key, rec in (capture.get("photos") or {}).items():
        prop, _, photo_key = key.partition("/")
        out.setdefault(prop, {})[photo_key] = rec["scene"]
    return out


def stage_scene_capture(config: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Capture Terra Pass 1a once per photo. Resumable per photo; rebuilding
    scenes.json from complete checkpoints makes no VLM calls."""
    from tools import pipeline_config as cfg
    from tools.comparison_common import sha256_file
    from tools.pass_config import SceneClassifierRunOptions, get_model_config_for_pass
    from tools.pipeline_common import normalize_scene_id
    from tools.scene_classifier_passes import run_pass_1a_scene_type
    from tools.vlm_client import create_vlm_client, get_model_configs_from_pipeline_config

    override = required_override(config)
    options = SceneClassifierRunOptions.from_analysis_profile(
        analysis_profile="standard",
        toggles=config["pass_toggles"],
        model_overrides=config["model_overrides"],
        reasoning_efforts=config["reasoning_efforts"],
        pipeline_mode="publish",
    )
    qwen_config, gpt5_config = get_model_configs_from_pipeline_config(cfg)
    model_config = get_model_config_for_pass("1a", options, qwen_config, gpt5_config)
    # Same rule as SceneClassifierOrchestrator._record_model_routing: the
    # presence of the override is what makes the route "explicit_override".
    source = (
        REQUIRED_SOURCE
        if options.model_overrides and options.model_overrides["1a"]
        else "standard_default"
    )
    route = {
        "provider": model_config.get("provider"),
        "model": model_config.get("model"),
        "reasoning_effort": model_config.get("reasoning_effort"),
        "source": source,
    }
    if (route["provider"], route["model"], route["source"]) != (
            REQUIRED_PROVIDER, override, REQUIRED_SOURCE):
        raise SystemExit(
            f"scene capture refused: resolved 1a route {route} is not the "
            f"explicit OpenAI override ({override})"
        )

    vlm_client = None  # created lazily so a fully-checkpointed rerun stays offline
    images_root = Path(manifest["images_root"])
    records: Dict[str, Dict[str, Any]] = {}
    for prop, photo in manifest_photos(manifest):
        key = f"{prop}/{photo['photo_key']}"
        ckpt = SCENES_DIR / prop / f"{photo['photo_key']}.json"
        if ckpt.is_file():
            rec = _load_json(ckpt)
            if rec.get("image_sha256") == photo["image_sha256"]:
                records[key] = rec
                continue
        image_path = images_root / prop / photo["photo_key"]
        digest = sha256_file(image_path)
        if digest != photo["image_sha256"]:
            raise SystemExit(
                f"{key}: image on disk ({digest[:12]}...) differs from the "
                "manifest hash — refusing to capture a drifted photo set"
            )
        if vlm_client is None:
            vlm_client = create_vlm_client()
        result = asyncio.run(run_pass_1a_scene_type(
            image_path=image_path,
            vlm_client=vlm_client,
            model_config=model_config,
        ))
        rec = {
            "property": prop,
            "photo_key": photo["photo_key"],
            "scene": normalize_scene_id(result.scene),
            "reasoning": result.reasoning or "",
            "image_sha256": digest,
            **route,
            "captured_at": _utcnow(),
        }
        _write_json(ckpt, rec)
        records[key] = rec
        print(f"[scene-capture] {key}: {rec['scene']}")

    capture = {
        "benchmark": "pass2a-prompt",
        "pass": "1a",
        "captured_at": _utcnow(),
        "photos": records,
    }
    capture["capture_sha256"] = scene_capture_sha(capture)
    assert_capture_routing(capture, config)
    _write_json(SCENES_DIR / "scenes.json", capture)
    print(f"scene capture complete: {len(records)} photos, "
          f"sha {capture['capture_sha256'][:16]}... -> {SCENES_DIR / 'scenes.json'}")
    return capture


def check_scene_capture(config: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    capture = load_scene_capture(config, manifest)
    print(f"scene capture OK: {len(capture['photos'])} photos, "
          f"sha {capture['capture_sha256'][:16]}...")
    for key, rec in sorted(capture["photos"].items()):
        print(f"  {key}: {rec['scene']} "
              f"[{rec['provider']}/{rec['model']}/{rec.get('reasoning_effort')}]")


# ---------------------------------------------------------------------------
# Frozen-scene consistency check. NOT yet wired into any stage (2026-08-30
# port note): the intended caller is a future v3 package-evaluation round,
# at the point where it loads one property's artifacts across cells. It must
# never run inside the legacy v2 rescore — those artifacts predate the scene
# capture, so their live Pass 1a scenes can legitimately differ.
# ---------------------------------------------------------------------------

def artifact_scene_map(artifact: Dict[str, Any]) -> Dict[str, str]:
    photos = artifact.get("photos") or {}
    return {
        key: (((rec.get("scene") or {}).get("id")) or "unknown")
        for key, rec in photos.items()
    }


def assert_frozen_scene_consistency(
    capture: Dict[str, Any],
    property_key: str,
    artifacts_by_label: Dict[str, Dict[str, Any]],
) -> None:
    """Every artifact must carry exactly the captured scenes, and all
    artifacts must agree on the photo -> physical-room mapping
    (build_room_surrogates is deterministic given identical scenes)."""
    from tools.room_surrogates import build_room_surrogates

    frozen = {
        key.partition("/")[2]: rec["scene"]
        for key, rec in (capture.get("photos") or {}).items()
        if key.partition("/")[0] == property_key
    }
    problems = []
    surrogate_maps: Dict[str, Dict[str, str]] = {}
    for label, artifact in sorted(artifacts_by_label.items()):
        for photo_key, scene in sorted(artifact_scene_map(artifact).items()):
            want = frozen.get(photo_key)
            if want is not None and scene != want:
                problems.append(
                    f"{label}/{photo_key}: scene {scene!r} != captured {want!r}")
        surro = build_room_surrogates(artifact.get("photos") or {})
        surrogate_maps[label] = surro.get("photo_key_to_room_surrogate_id") or {}
    if surrogate_maps:
        base_label = min(surrogate_maps)
        base = surrogate_maps[base_label]
        for label, mapping in sorted(surrogate_maps.items()):
            if mapping != base:
                diff = sorted(
                    k for k in set(mapping) | set(base)
                    if mapping.get(k) != base.get(k)
                )
                problems.append(
                    f"{label}: room mapping differs from {base_label} on {diff}")
    if problems:
        raise SystemExit(
            f"frozen scene consistency check failed for {property_key}:\n  "
            + "\n  ".join(problems)
        )
