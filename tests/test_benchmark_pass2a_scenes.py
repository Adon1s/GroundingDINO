"""Unit tests for the frozen Pass 1a scene-capture stage (no API calls).

Covers: capture resume/idempotence, capture-sha stability, the Terra
routing abort gate, manifest image-hash drift refusal, load-time
re-validation of an edited capture, and the frozen-scene / physical-room
consistency preflight.
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_pass2a as bench
from tools import benchmark_pass2a_scenes as scenes_mod
from tools.comparison_common import sha256_file


CONFIG = {
    "model_overrides": {"1a": "gpt-5.6-terra", "2a": "gpt-5.6-terra"},
    "reasoning_efforts": {"1a": "low", "2a": "low"},
    "pass_toggles": {"2f": False},
}

GPT5_CONFIG = {"url": "http://localhost:1", "model": "gpt-5.6-base",
               "api_key": "k", "provider": "openai"}
QWEN_CONFIG = {"url": "http://localhost:1", "model": "qwen",
               "provider": "lmstudio"}


@pytest.fixture
def scene_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(scenes_mod, "SCENES_DIR", tmp_path / "runs" / "v3" / "scene_capture")
    return tmp_path


def _manifest_with_images(tmp_path, scenes=("bedroom", "kitchen")):
    images_root = tmp_path / "img"
    photos = []
    for i, _scene in enumerate(scenes, start=1):
        photo_key = f"photo_{i:03d}.jpg"
        image_path = images_root / "prop" / photo_key
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"image-bytes-{i}".encode())
        photos.append({"photo_key": photo_key,
                       "image_sha256": sha256_file(image_path),
                       "frozen_2a_sha256": f"fh{i}"})
    return {
        "images_root": str(images_root),
        "properties": {"prop": {"photos": photos, "property_metadata": {}}},
    }


@pytest.fixture
def capture_stubs(monkeypatch):
    """Route resolution + Pass 1a stubbed; records every VLM invocation."""
    calls = []

    async def fake_run_pass_1a(image_path, vlm_client, model_config):
        from tools.scene_classifier_passes import Pass1aResult
        calls.append(str(image_path))
        scene = "bedroom" if "photo_001" in str(image_path) else "kitchen"
        return Pass1aResult(scene=scene, reasoning="stubbed")

    import tools.scene_classifier_passes as passes
    import tools.vlm_client as vlm
    monkeypatch.setattr(passes, "run_pass_1a_scene_type", fake_run_pass_1a)
    monkeypatch.setattr(vlm, "create_vlm_client", lambda: SimpleNamespace())
    monkeypatch.setattr(vlm, "get_model_configs_from_pipeline_config",
                        lambda cfg: (QWEN_CONFIG, GPT5_CONFIG))
    return calls


def test_capture_runs_once_per_photo_and_resumes_offline(scene_tree, capture_stubs):
    manifest = _manifest_with_images(scene_tree)
    capture = scenes_mod.stage_scene_capture(CONFIG, manifest)
    assert len(capture_stubs) == 2
    assert capture["photos"]["prop/photo_001.jpg"]["scene"] == "bedroom"
    rec = capture["photos"]["prop/photo_002.jpg"]
    assert (rec["provider"], rec["model"], rec["source"]) == (
        "openai", "gpt-5.6-terra", "explicit_override")
    assert rec["reasoning_effort"] == "low"
    sha_first = capture["capture_sha256"]

    # Second run: fully checkpointed — zero VLM calls, identical sha.
    capture2 = scenes_mod.stage_scene_capture(CONFIG, manifest)
    assert len(capture_stubs) == 2
    assert capture2["capture_sha256"] == sha_first

    # And the finalized capture loads through the full validation gate.
    loaded = scenes_mod.load_scene_capture(CONFIG, manifest)
    assert loaded["capture_sha256"] == sha_first


def test_capture_requires_explicit_1a_override(scene_tree, capture_stubs):
    manifest = _manifest_with_images(scene_tree)
    config = {**CONFIG, "model_overrides": {"2a": "gpt-5.6-terra"}}
    with pytest.raises(SystemExit, match="explicit '1a'"):
        scenes_mod.stage_scene_capture(config, manifest)


def test_capture_refuses_local_route(scene_tree, capture_stubs, monkeypatch):
    import tools.pass_config as pass_config
    monkeypatch.setattr(pass_config, "get_model_config_for_pass",
                        lambda *a, **k: dict(QWEN_CONFIG))
    manifest = _manifest_with_images(scene_tree)
    with pytest.raises(SystemExit, match="not the explicit OpenAI override"):
        scenes_mod.stage_scene_capture(CONFIG, manifest)
    assert capture_stubs == []  # aborted before any VLM call


def test_capture_refuses_drifted_image(scene_tree, capture_stubs):
    manifest = _manifest_with_images(scene_tree)
    manifest["properties"]["prop"]["photos"][0]["image_sha256"] = "not-the-disk-hash"
    with pytest.raises(SystemExit, match="differs from the manifest hash"):
        scenes_mod.stage_scene_capture(CONFIG, manifest)


def test_load_rejects_edited_capture(scene_tree, capture_stubs):
    manifest = _manifest_with_images(scene_tree)
    scenes_mod.stage_scene_capture(CONFIG, manifest)
    path = scenes_mod.SCENES_DIR / "scenes.json"

    # Hand-edited scene -> sha mismatch.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["photos"]["prop/photo_001.jpg"]["scene"] = "living_room"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="sha mismatch"):
        scenes_mod.load_scene_capture(CONFIG, manifest)

    # Sha recomputed to hide the edit, but the route is not the override:
    # the routing gate still aborts.
    payload["photos"]["prop/photo_001.jpg"].update(
        model="qwen", provider="lmstudio", source="standard_default")
    payload["capture_sha256"] = scenes_mod.scene_capture_sha(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="routing check failed"):
        scenes_mod.load_scene_capture(CONFIG, manifest)


def test_load_rejects_manifest_drift(scene_tree, capture_stubs):
    manifest = _manifest_with_images(scene_tree)
    scenes_mod.stage_scene_capture(CONFIG, manifest)
    drifted = json.loads(json.dumps(manifest))
    drifted["properties"]["prop"]["photos"][1]["image_sha256"] = "new-image-hash"
    with pytest.raises(SystemExit, match="image hashes differ"):
        scenes_mod.load_scene_capture(CONFIG, drifted)
    missing = json.loads(json.dumps(manifest))
    missing["properties"]["prop"]["photos"].pop()
    with pytest.raises(SystemExit, match="photo set differs"):
        scenes_mod.load_scene_capture(CONFIG, missing)


def test_scenes_by_property_shape(scene_tree, capture_stubs):
    manifest = _manifest_with_images(scene_tree)
    capture = scenes_mod.stage_scene_capture(CONFIG, manifest)
    assert scenes_mod.scenes_by_property(capture) == {
        "prop": {"photo_001.jpg": "bedroom", "photo_002.jpg": "kitchen"}}


# ---------------------------------------------------------------------------
# Frozen-scene / physical-room consistency preflight
# ---------------------------------------------------------------------------

def _artifact(scene_by_photo):
    return {"photos": {
        key: {"scene": {"id": scene}, "photo": {"index": i}}
        for i, (key, scene) in enumerate(sorted(scene_by_photo.items()), start=1)
    }}


def _capture_for(scene_by_photo):
    return {"photos": {
        f"prop/{key}": {"scene": scene}
        for key, scene in scene_by_photo.items()
    }}


def test_frozen_scene_consistency_passes_when_identical():
    scenes = {"photo_001.jpg": "bedroom", "photo_002.jpg": "kitchen",
              "photo_003.jpg": "bedroom"}
    scenes_mod.assert_frozen_scene_consistency(
        _capture_for(scenes), "prop",
        {"cell_a": _artifact(scenes), "cell_b": _artifact(scenes)},
    )


def test_frozen_scene_consistency_flags_scene_drift():
    scenes = {"photo_001.jpg": "bedroom", "photo_002.jpg": "kitchen"}
    drifted = {"photo_001.jpg": "living_room", "photo_002.jpg": "kitchen"}
    with pytest.raises(SystemExit) as excinfo:
        scenes_mod.assert_frozen_scene_consistency(
            _capture_for(scenes), "prop",
            {"cell_a": _artifact(scenes), "cell_b": _artifact(drifted)},
        )
    message = str(excinfo.value)
    assert "cell_b/photo_001.jpg" in message
    assert "living_room" in message
    # Room drift is reported too: the bedroom ordinal chain changed.
    assert "room mapping differs" in message
