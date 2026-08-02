import asyncio
import json
import logging
from pathlib import Path
import pytest
from types import SimpleNamespace

from tools import vlm_client as vlm_module
from tools.scene_classifier_passes import (
    PASS_2F_KITCHEN_USER_PROMPT,
    PASS_2F_PROMPT_VERSION,
    PASS_2F_ROOM_PROMPTS,
    run_pass_2f,
)
from tools.vlm_client import VLMClient


class _FakeOpenAIResponses:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text=json.dumps({"ok": True}), usage=None)


class _FakeOpenAIClient:
    def __init__(self):
        self.responses = _FakeOpenAIResponses()


def test_openai_analyze_images_uses_one_text_block_then_multiple_images():
    tmp_dir = Path("tests") / "_tmp_package_pass_2f"
    tmp_dir.mkdir(exist_ok=True)
    img1 = tmp_dir / "kitchen1.jpg"
    img2 = tmp_dir / "kitchen2.jpg"
    img1.write_bytes(b"image-one")
    img2.write_bytes(b"image-two")
    fake = _FakeOpenAIClient()
    client = VLMClient()
    client._get_openai_client = lambda api_key=None: fake

    try:
        result = asyncio.run(client.analyze_images(
            image_paths=[img1, img2],
            system_prompt="system",
            user_prompt="Analyze these kitchen photos together.",
            model="gpt-5.5",
            provider="openai",
            max_tokens=200,
        ))
    finally:
        for path in (img1, img2):
            if path.exists():
                path.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    assert json.loads(result) == {"ok": True}
    request = fake.responses.request
    user_content = request["input"][1]["content"]
    assert user_content[0] == {
        "type": "input_text",
        "text": "Analyze these kitchen photos together.",
    }
    assert [block["type"] for block in user_content[1:]] == ["input_image", "input_image"]
    assert all(block["image_url"].startswith("data:image/jpeg;base64,") for block in user_content[1:])
    assert client.usage_stats["calls"] == 1
    assert client.usage_stats["metered_calls"] == 0



class _FakeLMStudioResponse:
    status_code = 200
    text = "ok"

    def json(self):
        return {
            "choices": [{
                "message": {"content": json.dumps({"ok": True})},
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
        }


def test_lmstudio_analyze_images_sends_every_image_in_one_request(
    tmp_path,
    monkeypatch,
):
    img1 = tmp_path / "kitchen1.jpg"
    img2 = tmp_path / "kitchen2.jpg"
    img1.write_bytes(b"image-one")
    img2.write_bytes(b"image-two")
    captured = {}

    def fake_post(url, *, json, headers, timeout):
        captured.update({
            "url": url,
            "json": json,
            "headers": headers,
            "timeout": timeout,
        })
        return _FakeLMStudioResponse()

    monkeypatch.setattr(vlm_module.requests, "post", fake_post)
    client = VLMClient()
    result = asyncio.run(client.analyze_images(
        image_paths=[img1, img2],
        system_prompt="system",
        user_prompt="Analyze these kitchen photos together.",
        model="local-qwen",
        provider="lmstudio",
        url="http://localhost:1234",
        timeout=30,
        max_tokens=200,
    ))

    assert json.loads(result) == {"ok": True}
    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    content = captured["json"]["messages"][1]["content"]
    assert content[0] == {
        "type": "text",
        "text": "Analyze these kitchen photos together.",
    }
    assert [block["type"] for block in content[1:]] == [
        "image_url",
        "image_url",
    ]
    assert all(
        block["image_url"]["url"].startswith("data:image/jpeg;base64,")
        for block in content[1:]
    )
    assert client.usage_stats["input_tokens"] == 12
    assert client.usage_stats["output_tokens"] == 3
    assert client.usage_stats["total_tokens"] == 15
    assert client.usage_stats["attempted_calls"] == 1
    assert client.usage_stats["calls"] == 1
    assert client.usage_stats["failed_calls"] == 0
    assert client.usage_stats["metered_calls"] == 1
    assert client.usage_stats["api_duration_sec"] >= 0

def test_analyze_text_log_includes_analysis_pass(caplog):
    client = VLMClient()

    with caplog.at_level(logging.INFO, logger="tools.vlm_client"):
        try:
            asyncio.run(client.analyze_text(
                system_prompt="system",
                user_prompt="user",
                model="test-model",
                provider="lmstudio",
                analysis_pass="Pass 2b (observations JSON)",
            ))
        except ValueError as exc:
            assert "URL required for LM Studio provider" in str(exc)

    assert "Analyzing text for Pass 2b (observations JSON) with lmstudio/test-model" in caplog.text


class _PackageVLM:
    def __init__(self):
        self.calls = []

    async def analyze_images(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({
            "verification_status": "confirmed",
            "confirmed_issue_ids": ["issue_1"],
            "rejected_issue_ids": [],
            "evidence_summary": "Visible dated cabinets and counters.",
        })


class _BathroomPackageVLM:
    def __init__(self):
        self.calls = []

    async def analyze_images(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({
            "verification_status": "confirmed",
            "confirmed_issue_ids": ["issue_1"],
            "rejected_issue_ids": [],
            "evidence_summary": "Visible dated tile and vanity.",
            "visible_room_count": "multiple_rooms",
            "visible_room_count_evidence": "Tile color and vanity style conflict.",
        })


def test_package_pass_2f_returns_visual_truth_fields_only():
    tmp_dir = Path("tests") / "_tmp_package_pass_2f"
    tmp_dir.mkdir(exist_ok=True)
    img = tmp_dir / "kitchen.jpg"
    img.write_bytes(b"image")
    vlm = _PackageVLM()

    try:
        result = asyncio.run(run_pass_2f(
            image_paths=[img],
            vlm_client=vlm,
            model_config={"model": "gpt-5.5", "provider": "openai"},
            room="kitchen",
            package_id="kitchen_modernization__kitchen_1",
            package_type="kitchen_modernization",
            evidence_items=[{
                "catalog_item_id": "outdated_kitchen_finishes",
                "issue_ids": ["issue_1"],
                "observations": ["dated cabinets"],
            }],
        ))
    finally:
        if img.exists():
            img.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    prompt = vlm.calls[0]["user_prompt"]
    assert vlm.calls[0]["analysis_pass"] == "Pass 2f (package verification)"
    assert "pricing_posture" not in prompt
    assert "cost_low" not in prompt
    assert "cost_high" not in prompt
    assert result.verification_status == "confirmed"
    assert result.confirmed_issue_ids == ["issue_1"]


def test_bathroom_pass_2f_adds_room_count_without_changing_kitchen_schema():
    assert "visible_room_count" not in PASS_2F_KITCHEN_USER_PROMPT

    tmp_dir = Path("tests") / "_tmp_package_pass_2f"
    tmp_dir.mkdir(exist_ok=True)
    img = tmp_dir / "bathroom.jpg"
    img.write_bytes(b"image")
    vlm = _BathroomPackageVLM()

    try:
        result = asyncio.run(run_pass_2f(
            image_paths=[img],
            vlm_client=vlm,
            model_config={"model": "gpt-5.5", "provider": "openai"},
            room="bathroom",
            package_id="bathroom_modernization__bathroom_primary",
            package_type="bathroom_modernization",
            evidence_items=[{
                "catalog_item_id": "outdated_bathroom_finishes",
                "issue_ids": ["issue_1"],
                "observations": ["dated tile"],
            }],
        ))
    finally:
        if img.exists():
            img.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    prompt = vlm.calls[0]["user_prompt"]
    assert "visible_room_count" in prompt
    assert result.verification_status == "confirmed"
    assert result.visible_room_count == "multiple_rooms"
    assert result.visible_room_count_evidence == "Tile color and vanity style conflict."


class _RoomPackageVLM:
    def __init__(self):
        self.calls = []

    async def analyze_images(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({
            "verification_status": "confirmed",
            "confirmed_issue_ids": ["issue_1"],
            "rejected_issue_ids": [],
            "evidence_summary": "Visible worn carpet and a dated fixture.",
        })


def test_bedroom_and_living_pass_2f_use_standard_schema_no_room_count():
    tmp_dir = Path("tests") / "_tmp_package_pass_2f"
    tmp_dir.mkdir(exist_ok=True)
    img = tmp_dir / "room.jpg"
    img.write_bytes(b"image")

    cases = [
        ("bedroom", "bedroom_modernization", "bedroom_modernization__bedroom_1"),
        ("living", "living_modernization", "living_modernization__living_primary"),
    ]
    try:
        for room, package_type, package_id in cases:
            vlm = _RoomPackageVLM()
            result = asyncio.run(run_pass_2f(
                image_paths=[img],
                vlm_client=vlm,
                model_config={"model": "gpt-5.5", "provider": "openai"},
                room=room,
                package_id=package_id,
                package_type=package_type,
                evidence_items=[{
                    "catalog_item_id": "worn_carpet" if room == "bedroom" else "dated_living_finishes",
                    "issue_ids": ["issue_1"],
                    "observations": ["worn carpet"],
                }],
            ))
            prompt = vlm.calls[0]["user_prompt"]
            # Visual-truth only: no pricing, and the standard schema (no room-count telemetry).
            assert "cost_low" not in prompt
            assert "pricing_posture" not in prompt
            assert "visible_room_count" not in prompt
            assert result.verification_status == "confirmed"
            assert result.confirmed_issue_ids == ["issue_1"]
            assert result.visible_room_count == "unclear"
    finally:
        if img.exists():
            img.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

def test_exterior_pass_2f_is_registered_and_self_contained():
    tmp_dir = Path("tests") / "_tmp_package_pass_2f"
    tmp_dir.mkdir(exist_ok=True)
    img = tmp_dir / "elevation.jpg"
    img.write_bytes(b"image")

    try:
        assert "exterior" in PASS_2F_ROOM_PROMPTS, (
            "without a registered prompt an exterior package dies 'uncertain' "
            "and is dropped at finalize"
        )
        vlm = _RoomPackageVLM()
        result = asyncio.run(run_pass_2f(
            image_paths=[img],
            vlm_client=vlm,
            model_config={"model": "gpt-5.5", "provider": "openai"},
            room="exterior",
            package_id="exterior_repair__exterior_primary",
            package_type="exterior_repair",
            evidence_items=[{
                "catalog_item_id": "damaged_or_rotted_siding_or_trim",
                "issue_ids": ["issue_1"],
                "observations": ["rotted siding"],
            }],
        ))
        system_prompt = vlm.calls[0]["system_prompt"]
        user_prompt = vlm.calls[0]["user_prompt"]

        # Visual-truth only, standard schema (no room-count telemetry).
        assert "cost_low" not in user_prompt
        assert "pricing_posture" not in user_prompt
        assert "visible_room_count" not in user_prompt
        assert result.verification_status == "confirmed"
        assert result.confirmed_issue_ids == ["issue_1"]

        # Self-contained: no cross-room vocabulary bleeding attention.
        lowered = system_prompt.lower()
        for foreign in ("kitchen", "bathroom", "bedroom", "vanity", "cabinet",
                        "backsplash", "carpet"):
            assert foreign not in lowered, f"exterior prompt mentions {foreign!r}"
        # Out-of-scope evidence must not be suggested to the model.
        for out_of_scope in ("shingle", "roof", "gutter", "landscap", "lawn", "driveway"):
            assert out_of_scope not in lowered, (
                f"exterior prompt primes {out_of_scope!r}, which this package does not cover"
            )
        assert "siding" in lowered and "soffit" in lowered
    finally:
        if img.exists():
            img.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def test_pass_2f_prompt_version_unchanged_by_new_rooms():
    # The 2f evidence-revival path gates on this exact string; bumping it
    # silently disables revival. Adding a room changes only the sha.
    assert PASS_2F_PROMPT_VERSION == "pass_2f_package_v2"


def test_job_usage_attributes_concurrent_calls_to_canonical_passes():
    client = VLMClient()

    async def metered_call(delay, input_tokens, output_tokens):
        await asyncio.sleep(delay)
        client._record_call()
        client._record_usage(input_tokens, output_tokens, input_tokens + output_tokens)
        return "ok"

    async def run_calls():
        await asyncio.gather(
            client._run_with_telemetry(
                "Pass 1a (scene classification)",
                metered_call(0.01, 10, 2),
            ),
            client._run_with_telemetry(
                "Pass 2f (package verification)",
                metered_call(0.005, 20, 3),
            ),
        )

    asyncio.run(run_calls())

    assert client.usage_stats["attempted_calls"] == 2
    assert client.usage_stats["calls"] == 2
    assert client.usage_stats["metered_calls"] == 2
    assert client.usage_stats["total_tokens"] == 35
    assert client.usage_stats["per_pass"]["1a"]["total_tokens"] == 12
    assert client.usage_stats["per_pass"]["2f"]["total_tokens"] == 23
    assert client.usage_stats["per_pass"]["1a"]["api_duration_sec"] > 0
    assert client.usage_stats["per_pass"]["2f"]["api_duration_sec"] > 0


def test_job_usage_records_failed_logical_provider_call():
    client = VLMClient()

    async def fail():
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(client._run_with_telemetry("Pass 2a", fail()))

    assert client.usage_stats["attempted_calls"] == 1
    assert client.usage_stats["calls"] == 0
    assert client.usage_stats["failed_calls"] == 1
    assert client.usage_stats["per_pass"]["2a"]["failed_calls"] == 1
