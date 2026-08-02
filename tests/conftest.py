"""
Shared test fixtures.

Exists mainly to give one OpenAI fake with the response shapes the pass-failure
contract depends on: incomplete (token truncation), refusal, and empty output.
Before this, each test module hand-rolled its own fake -- with mutually
incompatible `_get_openai_client` signatures -- and none of them covered those
three paths, which are exactly the ones that used to be swallowed into empty
results.

No pytest.ini exists, so nothing here may rely on markers or plugins. Async
tests use bare asyncio.run(), matching the existing convention.
"""
import json
from types import SimpleNamespace

import pytest


def openai_response(
    *,
    text=None,
    status="completed",
    incomplete_reason=None,
    refusal=None,
    response_id="resp_test",
):
    """
    Build a fake OpenAI Responses object.

    Mirrors the attribute shape VLMClient._extract_openai_output_text reads:
    .status, .incomplete_details.reason, .output_text, .output[].content[].
    """
    output = []
    if refusal is not None:
        output = [SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal=refusal)])]
    return SimpleNamespace(
        id=response_id,
        status=status,
        incomplete_details=(
            SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
        ),
        output_text=text,
        output=output,
        usage=None,
    )


class FakeOpenAI:
    """
    Stand-in for the OpenAI SDK client.

    `requests` records every payload passed to responses.create, so tests can
    assert on the token cap and reasoning effort that actually went on the wire.
    """

    def __init__(self, *, responses=None, raises=None):
        self._queued = list(responses or [])
        self._raises = raises
        self.requests = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self._raises is not None:
            raise self._raises
        if self._queued:
            return self._queued.pop(0)
        return openai_response(text=json.dumps({"ok": True}))

    @property
    def last_request(self):
        return self.requests[-1] if self.requests else None

    def attach(self, client):
        """Install onto a VLMClient, tolerating either call signature."""
        client._get_openai_client = lambda *args, **kwargs: self
        return self


@pytest.fixture
def fake_openai():
    """Factory: fake_openai(responses=[...]) or fake_openai(raises=Exception(...))."""
    def _make(*, responses=None, raises=None):
        return FakeOpenAI(responses=responses, raises=raises)
    return _make


@pytest.fixture
def openai_config():
    """A minimal model_config that the resolver treats as OpenAI-bound."""
    return {"provider": "openai", "model": "gpt-5.4-mini", "api_key": "test-key"}


@pytest.fixture
def qwen_config():
    """A local-provider config, which OpenAI policy must leave untouched."""
    return {"provider": "lmstudio", "model": "qwen3.6-27b", "url": "http://localhost:1234"}


@pytest.fixture
def clean_token_env(monkeypatch):
    """Remove token-cap env vars so table defaults are what's under test."""
    for name in (
        "OPENAI_DEFAULT_MAX_TOKENS",
        "OPENAI_PASS_1A_MAX_TOKENS",
        "OPENAI_PASS_2A_MAX_TOKENS",
        "OPENAI_PASS_2D_MAX_TOKENS",
        "OPENAI_PASS_2F_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Benchmark dataset fixtures
# ---------------------------------------------------------------------------
# Shared by the benchmark schema/dataset/authoring tests. Real PNG bytes rather
# than empty files, because photo integrity checks hash and size-check them.

_PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def tiny_png(red: int, green: int, blue: int) -> bytes:
    """A valid 1x1 PNG. Distinct colors give distinct hashes."""
    import struct
    import zlib

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    pixels = chunk(b"IDAT", zlib.compress(b"\x00" + bytes([red, green, blue])))
    return _PNG_HEADER + header + pixels + chunk(b"IEND", b"")


class BenchmarkDatasetBuilder:
    """Builds a minimal on-disk dataset that passes validation."""

    def __init__(self, root):
        from pathlib import Path

        self.root = Path(root)
        self.dataset_version = "renovation-v1"
        self.path = self.root / self.dataset_version
        (self.path / "listings").mkdir(parents=True, exist_ok=True)
        self.write_manifest([])

    # -- manifest ---------------------------------------------------------
    def write_manifest(self, listings, *, state="draft", **extra):
        from tools.comparison_common import atomic_json

        manifest = {
            "schema_version": 1,
            "dataset_id": "renovation",
            "dataset_version": self.dataset_version,
            "state": state,
            "listings": listings,
            **extra,
        }
        atomic_json(self.path / "manifest.json", manifest)
        return manifest

    def read_manifest(self):
        return json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))

    # -- listing ----------------------------------------------------------
    def add_listing(self, listing_id="hsv-001", *, photo_count=3, tier="gold",
                    slices=("gold-development",), duplicate_last=False):
        """Write photos + metadata.json and register the listing."""
        from tools.comparison_common import atomic_json, sha256_file

        photos_dir = self.path / "listings" / listing_id / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)

        photos = []
        for index in range(photo_count):
            name = f"photo_{index + 1:03d}.png"
            # With duplicate_last, the final photo is byte-identical to the first
            # so duplicate-hash detection has something real to find.
            is_dupe = duplicate_last and index == photo_count - 1 and photo_count >= 2
            color = (0, 20, 30) if is_dupe else (10 * index, 20, 30)
            (photos_dir / name).write_bytes(tiny_png(*color))
            photos.append({
                "order": index + 1,
                "filename": name,
                "sha256": sha256_file(photos_dir / name),
                "byte_size": (photos_dir / name).stat().st_size,
            })

        listing = {
            "schema_version": 1,
            "listing_id": listing_id,
            "dataset_version": self.dataset_version,
            "original_metadata": {},
            "asking_price": 185000,
            "sqft": 1450,
            "beds": 3,
            "baths": 2,
            "location": "Huntsville, AL 35801",
            "market_inputs": {"area_ppsf": 118.0, "source": "manual", "sample_size": 12},
            "photos": photos,
        }
        atomic_json(self.path / "listings" / listing_id / "metadata.json", listing)

        entries = self.read_manifest()["listings"]
        entries = [e for e in entries if e.get("id") != listing_id]
        entries.append({
            "id": listing_id,
            "tier": tier,
            "slices": sorted(slices),
            "metadata_path": f"listings/{listing_id}/metadata.json",
            "reference_path": f"listings/{listing_id}/reference.json",
        })
        entries.sort(key=lambda e: e["id"])
        self.write_manifest(entries)
        return listing

    def listing_path(self, listing_id="hsv-001"):
        return self.path / "listings" / listing_id

    def photo(self, listing_id="hsv-001", name="photo_001.png"):
        return self.listing_path(listing_id) / "photos" / name

    def write_reference(self, reference, listing_id="hsv-001"):
        from tools.comparison_common import atomic_json

        atomic_json(self.listing_path(listing_id) / "reference.json", reference)

    def write_draft(self, text, listing_id="hsv-001"):
        path = self.listing_path(listing_id) / "reference.draft.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


@pytest.fixture
def dataset_builder(tmp_path):
    return BenchmarkDatasetBuilder(tmp_path / "datasets")


@pytest.fixture(scope="session")
def issue_catalog():
    """The real catalog. Vocabulary tests assert against real ids on purpose:
    a snapshot built from a synthetic catalog would not catch drift in the one
    that actually ships."""
    from pathlib import Path

    from tools import pipeline_config as cfg
    from tools.artifact_writers import load_issue_catalog

    return load_issue_catalog(Path(cfg.ISSUE_CATALOG_PATH))


@pytest.fixture(scope="session")
def frozen_vocabulary(issue_catalog):
    from tools.benchmarking import vocabulary as vocab_mod

    return vocab_mod.snapshot(issue_catalog)
