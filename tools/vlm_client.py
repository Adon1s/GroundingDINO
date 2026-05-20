"""
VLM Client for Scene Classification Pipeline
---------------------------------------------
Provides a unified interface for calling Vision Language Models.

Supports:
- LM Studio (local Qwen, etc.) - uses requests for chat/completions
- OpenAI API (GPT-4o, GPT-5) - uses official OpenAI Python SDK
- Google Gemini API - uses google-genai SDK

Usage:
    client = VLMClient()

    # Image analysis with LM Studio (base64 encoding)
    result = await client.analyze_image(
        image_path=Path('/path/to/image.jpg'),
        system_prompt="You are a real estate analyst.",
        user_prompt="What do you see?",
        url="http://localhost:1234",
        model="qwen-vl-7b",
        provider="lmstudio",
    )

    # Image analysis with OpenAI (file upload via SDK)
    result = await client.analyze_image(
        image_path=Path('/path/to/image.jpg'),
        system_prompt="You are a real estate analyst.",
        user_prompt="What do you see?",
        model="gpt-4o",
        api_key="sk-...",
        provider="openai",
    )

Requirements:
    pip install openai requests
"""

import asyncio
import base64
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import requests

# Import official OpenAI SDK
try:
    from openai import OpenAI

    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_SDK_AVAILABLE = False
    print("WARNING: openai package not installed. Install with: pip install openai")

try:
    from google import genai
    from google.genai import types as genai_types

    GEMINI_SDK_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GEMINI_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

# Provider types
ProviderType = Literal["openai", "lmstudio", "gemini", "auto"]


class VLMClient:
    """
    Unified client for Vision Language Model calls.

    Handles both local (LM Studio) and cloud (OpenAI) endpoints.
    Uses official OpenAI SDK for OpenAI calls.
    """

    def __init__(
            self,
            default_timeout: int = 120,
            default_max_tokens: int = 4096,
            default_temperature: float = 0.2,
    ):
        self.default_timeout = default_timeout
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

        # Cache for OpenAI clients (keyed by api_key)
        self._openai_clients: Dict[str, Any] = {}

        # Cache for uploaded file IDs (avoid re-uploading same file)
        self._file_cache: Dict[str, str] = {}

        # Cache for Gemini clients (keyed by api_key)
        self._gemini_clients: Dict[str, Any] = {}

        # Token usage accumulator — shared across provider calls.
        # Calls run in run_in_executor threads, so guard with a lock.
        self._usage_lock = threading.Lock()
        self.usage_stats: Dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    def reset_usage_stats(self) -> None:
        """Zero out the accumulated token counters. Call between jobs."""
        with self._usage_lock:
            self.usage_stats["input_tokens"] = 0
            self.usage_stats["output_tokens"] = 0
            self.usage_stats["total_tokens"] = 0
            self.usage_stats["calls"] = 0

    def _record_usage(
            self,
            input_tokens: Optional[int],
            output_tokens: Optional[int],
            total_tokens: Optional[int],
    ) -> None:
        """Add a single call's token usage to the running totals. Tolerant of None/missing fields."""
        try:
            i = int(input_tokens) if input_tokens is not None else 0
            o = int(output_tokens) if output_tokens is not None else 0
            t = int(total_tokens) if total_tokens is not None else (i + o)
        except (TypeError, ValueError):
            return
        with self._usage_lock:
            self.usage_stats["input_tokens"] += i
            self.usage_stats["output_tokens"] += o
            self.usage_stats["total_tokens"] += t
            self.usage_stats["calls"] += 1

    def _detect_provider(self, url: Optional[str], api_key: Optional[str]) -> ProviderType:
        """
        Auto-detect provider.
        - If URL looks local → LM Studio
        - Else if key exists → OpenAI
        - Else → LM Studio
        """
        key = api_key or os.environ.get("OPENAI_API_KEY")
        u = (url or "").lower().strip()

        # Strong LM Studio signals
        is_local = any(x in u for x in ("localhost", "127.0.0.1", "0.0.0.0", "169.254.", ".local"))

        if url and is_local:
            return "lmstudio"
        if key:
            return "openai"
        return "lmstudio"

    def _get_openai_client(self, api_key: Optional[str] = None) -> Any:
        """Get or create an OpenAI client. If api_key is None, rely on env + OpenAI()."""
        if not OPENAI_SDK_AVAILABLE:
            raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")

        # Read optional base URL overrides (ONLY if explicitly set)
        base_url = (
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE")
                or ""
        ).strip()

        if base_url:
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
        else:
            base_url = None  # <-- important: do not force a default

        key_part = api_key or "__env__"
        base_part = base_url or "__default__"
        cache_key = f"{key_part}:{base_part}"

        if cache_key not in self._openai_clients:
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url

            # This will be OpenAI() when kwargs is empty
            self._openai_clients[cache_key] = OpenAI(**kwargs)

        return self._openai_clients[cache_key]

    def _encode_image_base64(self, image_path: Path) -> Tuple[str, str]:
        """Encode image to base64 and determine media type."""
        suffix = image_path.suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
        }
        media_type = media_types.get(suffix, 'image/jpeg')

        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')

        return data, media_type

    # ═══════════════════════════════════════════════════════════════════════════
    # OpenAI SDK Methods (using official openai package)
    # ═══════════════════════════════════════════════════════════════════════════

    def _upload_file_to_openai(
            self,
            client: Any,
            file_path: Path,
    ) -> str:
        """
        Upload a file to OpenAI using the SDK and return the file ID.

        Uses caching to avoid re-uploading the same file.
        """
        cache_key = f"{file_path}:{file_path.stat().st_mtime}"

        if cache_key in self._file_cache:
            logger.debug(f"Using cached file ID for {file_path.name}")
            return self._file_cache[cache_key]

        logger.debug(f"Uploading file to OpenAI: {file_path.name}")

        # ⭐ DEBUG: Confirm the actual base URL the SDK is using
        logger.debug(f"OpenAI base_url in use: {getattr(client, 'base_url', 'unknown')}")

        # Use official SDK to upload file
        with open(file_path, 'rb') as f:
            file_response = client.files.create(
                file=f,
                purpose="user_data"
            )

        file_id = file_response.id
        self._file_cache[cache_key] = file_id

        logger.debug(f"Uploaded file {file_path.name} -> {file_id}")
        return file_id

    def _openai_text_config(
            self,
            response_json_schema: Optional[Dict[str, Any]],
            response_schema_name: Optional[str],
            verbosity: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Build Responses API text config for strict structured output."""
        text_config: Dict[str, Any] = {}
        if response_json_schema:
            text_config["format"] = {
                "type": "json_schema",
                "name": response_schema_name or "structured_response",
                "schema": response_json_schema,
                "strict": True,
            }
        if verbosity:
            text_config["verbosity"] = verbosity
        return text_config or None

    def _extract_openai_output_text(self, response: Any) -> str:
        """Return visible output text or raise a useful error for empty/incomplete responses."""
        status = getattr(response, "status", None)
        response_id = getattr(response, "id", None)
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details is not None else None
            raise RuntimeError(
                f"OpenAI response incomplete"
                f"{f' ({reason})' if reason else ''}"
                f"{f' [response_id={response_id}]' if response_id else ''}"
            )

        output_text = (getattr(response, "output_text", None) or "").strip()
        if output_text:
            return output_text

        refusal_texts: list[str] = []
        fallback_texts: list[str] = []
        for item in getattr(response, "output", None) or []:
            for content in getattr(item, "content", None) or []:
                content_type = getattr(content, "type", None)
                text = getattr(content, "text", None) or getattr(content, "refusal", None)
                if not text:
                    continue
                if content_type == "refusal":
                    refusal_texts.append(str(text))
                elif content_type in {"output_text", "text"}:
                    fallback_texts.append(str(text))

        if refusal_texts:
            snippet = " ".join(refusal_texts)[:300]
            raise RuntimeError(
                f"OpenAI model refusal: {snippet}"
                f"{f' [response_id={response_id}]' if response_id else ''}"
            )
        if fallback_texts:
            return "\n".join(fallback_texts).strip()

        raise RuntimeError(
            f"OpenAI response had no output_text"
            f"{f' (status={status})' if status else ''}"
            f"{f' [response_id={response_id}]' if response_id else ''}"
        )


    async def _analyze_image_openai(
            self,
            image_path: Path,
            system_prompt: str,
            user_prompt: str,
            model: str,
            api_key: Optional[str],
            max_tokens: int,
            response_json_schema: Optional[Dict[str, Any]] = None,
            response_schema_name: Optional[str] = None,
            reasoning_effort: Optional[str] = None,
            verbosity: Optional[str] = None,
    ) -> str:
        client = self._get_openai_client(api_key)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not image_path.is_file():
            raise ValueError(f"Not a file: {image_path}")

        logger.debug(f"OpenAI image path OK: {image_path.resolve()}")
        logger.debug(f"OpenAI image size bytes: {image_path.stat().st_size}")

        # Base64 encode
        image_data, media_type = self._encode_image_base64(image_path)

        # Doc-aligned format
        data_url = f"data:{media_type};base64,{image_data}"
        text_config = self._openai_text_config(
            response_json_schema,
            response_schema_name,
            verbosity if str(model).lower().startswith("gpt-5") else None,
        )

        def _call():
            request: Dict[str, Any] = {
                "model": model,
                "input": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                # optional knob from your docs:
                                # "detail": "low",
                            },
                        ],
                    },
                ],
                "max_output_tokens": max_tokens,
            }
            if text_config:
                request["text"] = text_config
            if reasoning_effort and str(model).lower().startswith("gpt-5"):
                request["reasoning"] = {"effort": reasoning_effort}
            return client.responses.create(**request)

        response = await asyncio.get_event_loop().run_in_executor(None, _call)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._record_usage(
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
                getattr(usage, "total_tokens", None),
            )
        return self._extract_openai_output_text(response)

    async def _analyze_images_openai(
            self,
            image_paths: List[Path],
            system_prompt: str,
            user_prompt: str,
            model: str,
            api_key: Optional[str],
            max_tokens: int,
            response_json_schema: Optional[Dict[str, Any]] = None,
            response_schema_name: Optional[str] = None,
            reasoning_effort: Optional[str] = None,
            verbosity: Optional[str] = None,
    ) -> str:
        client = self._get_openai_client(api_key)

        if not image_paths:
            raise ValueError("At least one image path is required")

        content: List[Dict[str, Any]] = [
            {"type": "input_text", "text": user_prompt},
        ]
        for image_path in image_paths:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not image_path.is_file():
                raise ValueError(f"Not a file: {image_path}")
            logger.debug(f"OpenAI multi-image path OK: {image_path.resolve()}")
            image_data, media_type = self._encode_image_base64(image_path)
            content.append({
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{image_data}",
            })

        text_config = self._openai_text_config(
            response_json_schema,
            response_schema_name,
            verbosity if str(model).lower().startswith("gpt-5") else None,
        )

        def _call():
            request: Dict[str, Any] = {
                "model": model,
                "input": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                "max_output_tokens": max_tokens,
            }
            if text_config:
                request["text"] = text_config
            if reasoning_effort and str(model).lower().startswith("gpt-5"):
                request["reasoning"] = {"effort": reasoning_effort}
            return client.responses.create(**request)

        response = await asyncio.get_event_loop().run_in_executor(None, _call)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._record_usage(
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
                getattr(usage, "total_tokens", None),
            )
        return self._extract_openai_output_text(response)

    async def _analyze_text_openai(
            self,
            system_prompt: str,
            user_prompt: str,
            model: str,
            api_key: Optional[str],
            max_tokens: int,
            response_json_schema: Optional[Dict[str, Any]] = None,
            response_schema_name: Optional[str] = None,
            reasoning_effort: Optional[str] = None,
            verbosity: Optional[str] = None,
    ) -> str:
        """
        Text-only analysis using OpenAI's Responses API via official SDK.
        """
        client = self._get_openai_client(api_key)

        logger.debug(f"Calling OpenAI Responses API (text): {model}")
        text_config = self._openai_text_config(
            response_json_schema,
            response_schema_name,
            verbosity if str(model).lower().startswith("gpt-5") else None,
        )

        def _call():
            request: Dict[str, Any] = {
                "model": model,
                "input": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "max_output_tokens": max_tokens,
            }
            if text_config:
                request["text"] = text_config
            if reasoning_effort and str(model).lower().startswith("gpt-5"):
                request["reasoning"] = {"effort": reasoning_effort}
            return client.responses.create(**request)

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            _call,
        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            self._record_usage(
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
                getattr(usage, "total_tokens", None),
            )
        return self._extract_openai_output_text(response)

    # ═══════════════════════════════════════════════════════════════════════════
    # Google Gemini Methods (using google-genai)
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_gemini_client(self, api_key: Optional[str] = None) -> Any:
        """Get or create a Gemini client. If api_key is None, rely on GEMINI_API_KEY env."""
        if not GEMINI_SDK_AVAILABLE:
            raise RuntimeError("google-genai SDK not installed. Run: pip install google-genai")

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass api_key.")

        if key not in self._gemini_clients:
            self._gemini_clients[key] = genai.Client(api_key=key)

        return self._gemini_clients[key]

    async def _analyze_image_gemini(
            self,
            image_path: Path,
            system_prompt: str,
            user_prompt: str,
            model: str,
            api_key: Optional[str],
            max_tokens: int,
    ) -> str:
        """Analyze image using Google Gemini's generate_content API."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        client = self._get_gemini_client(api_key)

        # Read image bytes and determine mime type
        image_data, media_type = self._encode_image_base64(image_path)
        image_bytes = base64.b64decode(image_data)

        logger.debug(f"Calling Gemini ({model}) for image: {image_path.name}")

        def _call():
            return client.models.generate_content(
                model=model,
                contents=[
                    genai_types.Part.from_bytes(data=image_bytes, mime_type=media_type),
                    user_prompt,
                ],
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                ),
            )

        response = await asyncio.get_event_loop().run_in_executor(None, _call)
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self._record_usage(
                getattr(usage, "prompt_token_count", None),
                getattr(usage, "candidates_token_count", None),
                getattr(usage, "total_token_count", None),
            )
        return response.text

    async def _analyze_text_gemini(
            self,
            system_prompt: str,
            user_prompt: str,
            model: str,
            api_key: Optional[str],
            max_tokens: int,
    ) -> str:
        """Text-only analysis using Google Gemini."""
        client = self._get_gemini_client(api_key)

        logger.debug(f"Calling Gemini ({model}) for text analysis")

        def _call():
            return client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                ),
            )

        response = await asyncio.get_event_loop().run_in_executor(None, _call)
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self._record_usage(
                getattr(usage, "prompt_token_count", None),
                getattr(usage, "candidates_token_count", None),
                getattr(usage, "total_token_count", None),
            )
        return response.text

    # ═══════════════════════════════════════════════════════════════════════════
    # LM Studio Methods (using requests)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _analyze_image_lmstudio(
            self,
            image_path: Path,
            system_prompt: str,
            user_prompt: str,
            url: str,
            model: str,
            timeout: int,
            max_tokens: int,
            temperature: float,
    ) -> str:
        """
        Analyze image using LM Studio's chat/completions endpoint.

        Uses base64 image encoding inline with the user message.
        """
        # Encode image to base64
        image_data, media_type = self._encode_image_base64(image_path)
        data_url = f"data:{media_type};base64,{image_data}"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        endpoint = f"{url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        logger.debug(f"Calling LM Studio: {endpoint} with model {model}")

        # Run blocking request in executor
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        )

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(f"LM Studio API error: HTTP {response.status_code} - {error_text}")
            raise RuntimeError(f"LM Studio API error: HTTP {response.status_code}")

        data = response.json()

        if "choices" not in data:
            logger.error(f"LM Studio response missing 'choices': {data}")
            raise RuntimeError(f"Unexpected LM Studio response format from LM Studio")

        usage = data.get("usage") or {}
        if usage:
            self._record_usage(
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
            )
        return data["choices"][0]["message"]["content"]

    async def _analyze_text_lmstudio(
            self,
            system_prompt: str,
            user_prompt: str,
            url: str,
            model: str,
            timeout: int,
            max_tokens: int,
            temperature: float,
    ) -> str:
        """
        Text-only analysis using LM Studio's chat/completions endpoint.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        endpoint = f"{url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        logger.debug(f"Calling LM Studio (text): {endpoint} with model {model}")

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        )

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(f"LM Studio API error: HTTP {response.status_code} - {error_text}")
            raise RuntimeError(f"LM Studio API error: HTTP {response.status_code}")

        data = response.json()

        if "choices" not in data:
            logger.error(f"LM Studio response missing 'choices': {data}")
            raise RuntimeError(f"Unexpected LM Studio response format from LM Studio")

        usage = data.get("usage") or {}
        if usage:
            self._record_usage(
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
            )
        return data["choices"][0]["message"]["content"]

    # ═══════════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════════

    async def analyze_image(
            self,
            image_path: Path,
            system_prompt: str,
            user_prompt: str,
            model: str,
            url: Optional[str] = None,
            api_key: Optional[str] = None,
            provider: ProviderType = "auto",
            timeout: Optional[int] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            **kwargs,
    ) -> str:
        """
        Analyze an image with a VLM.

        Args:
            image_path: Path to the image file
            system_prompt: System message for the model
            user_prompt: User message/question about the image
            model: Model name (e.g., "gpt-4o", "qwen-vl-7b")
            url: API endpoint URL (required for LM Studio)
            api_key: API key (required for OpenAI)
            provider: "openai", "lmstudio", or "auto" (default: auto-detect)
            timeout: Request timeout in seconds
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters (ignored)

        Returns:
            Model response as string (typically JSON)
        """
        # Accept either name from callers (orchestrator may provide max_output_tokens)
        if max_tokens is None:
            mo = kwargs.get("max_output_tokens")
            if mo is not None:
                try:
                    max_tokens = int(mo)
                except Exception:
                    pass
        response_json_schema = kwargs.get("response_json_schema") or kwargs.get("json_schema")
        response_schema_name = kwargs.get("response_schema_name") or kwargs.get("json_schema_name")
        reasoning_effort = kwargs.get("reasoning_effort")
        verbosity = kwargs.get("verbosity")

        timeout = timeout or self.default_timeout
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        # Auto-detect provider if needed
        if provider == "auto":
            provider = self._detect_provider(url, api_key)

        logger.info(f"Analyzing image {image_path.name} with {provider}/{model}")

        if provider == "openai":
            # If api_key is provided explicitly, use it.
            # Otherwise rely on OpenAI() reading OPENAI_API_KEY from env.
            return await self._analyze_image_openai(
                image_path=image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=api_key,  # <-- pass through as Optional
                max_tokens=max_tokens,
                response_json_schema=response_json_schema,
                response_schema_name=response_schema_name,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            )

        elif provider == "gemini":
            return await self._analyze_image_gemini(
                image_path=image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
            )

        else:  # lmstudio
            if not url:
                raise ValueError("URL required for LM Studio provider")

            return await self._analyze_image_lmstudio(
                image_path=image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                url=url,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    async def analyze_images(
            self,
            image_paths: List[Path],
            system_prompt: str,
            user_prompt: str,
            model: str,
            url: Optional[str] = None,
            api_key: Optional[str] = None,
            provider: ProviderType = "auto",
            timeout: Optional[int] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            **kwargs,
    ) -> str:
        """
        Analyze multiple images together with a VLM.

        OpenAI uses the Responses multi-image structure: one user content array
        containing a single input_text block followed by one input_image block
        per image. Other providers fall back to the first image so existing
        local workflows keep running, but package verification should use
        OpenAI/premium for true multi-image review.
        """
        if max_tokens is None:
            mo = kwargs.get("max_output_tokens")
            if mo is not None:
                try:
                    max_tokens = int(mo)
                except Exception:
                    pass
        response_json_schema = kwargs.get("response_json_schema") or kwargs.get("json_schema")
        response_schema_name = kwargs.get("response_schema_name") or kwargs.get("json_schema_name")
        reasoning_effort = kwargs.get("reasoning_effort")
        verbosity = kwargs.get("verbosity")

        image_paths = [Path(p) for p in (image_paths or [])]
        if not image_paths:
            raise ValueError("At least one image path is required")

        timeout = timeout or self.default_timeout
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        if provider == "auto":
            provider = self._detect_provider(url, api_key)

        logger.info("Analyzing %d images with %s/%s", len(image_paths), provider, model)

        if provider == "openai":
            return await self._analyze_images_openai(
                image_paths=image_paths,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                response_json_schema=response_json_schema,
                response_schema_name=response_schema_name,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            )

        # Backward-compatible fallback for providers that do not currently have
        # a native multi-image helper in this client.
        fallback_prompt = user_prompt
        if len(image_paths) > 1:
            fallback_prompt = (
                user_prompt
                + "\n\nNote: this provider path can only review one image; "
                "review the supplied representative image only."
            )
        return await self.analyze_image(
            image_path=image_paths[0],
            system_prompt=system_prompt,
            user_prompt=fallback_prompt,
            model=model,
            url=url,
            api_key=api_key,
            provider=provider,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def analyze_text(
            self,
            system_prompt: str,
            user_prompt: str,
            model: str,
            url: Optional[str] = None,
            api_key: Optional[str] = None,
            provider: ProviderType = "auto",
            timeout: Optional[int] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            **kwargs,
    ) -> str:
        """
        Text-only analysis (no image).

        Args:
            system_prompt: System message for the model
            user_prompt: User message/question
            model: Model name
            url: API endpoint URL (required for LM Studio)
            api_key: API key (required for OpenAI)
            provider: "openai", "lmstudio", or "auto"
            timeout: Request timeout in seconds
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters (ignored)

        Returns:
            Model response as string
        """
        # Accept either name from callers (orchestrator may provide max_output_tokens)
        if max_tokens is None:
            mo = kwargs.get("max_output_tokens")
            if mo is not None:
                try:
                    max_tokens = int(mo)
                except Exception:
                    pass
        response_json_schema = kwargs.get("response_json_schema") or kwargs.get("json_schema")
        response_schema_name = kwargs.get("response_schema_name") or kwargs.get("json_schema_name")
        reasoning_effort = kwargs.get("reasoning_effort")
        verbosity = kwargs.get("verbosity")

        timeout = timeout or self.default_timeout
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        # Auto-detect provider if needed
        if provider == "auto":
            provider = self._detect_provider(url, api_key)

        logger.info(f"Analyzing text with {provider}/{model}")

        if provider == "openai":
            # If api_key is provided explicitly, use it.
            # Otherwise rely on OpenAI() reading OPENAI_API_KEY from env.
            return await self._analyze_text_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=api_key,  # <-- pass through as Optional
                max_tokens=max_tokens,
                response_json_schema=response_json_schema,
                response_schema_name=response_schema_name,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            )

        elif provider == "gemini":
            return await self._analyze_text_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
            )

        else:  # lmstudio
            if not url:
                raise ValueError("URL required for LM Studio provider")

            return await self._analyze_text_lmstudio(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                url=url,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    # Sync wrappers for non-async code
    def analyze_image_sync(self, **kwargs) -> str:
        """Synchronous wrapper for analyze_image."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.analyze_image(**kwargs))
        finally:
            loop.close()

    def analyze_images_sync(self, **kwargs) -> str:
        """Synchronous wrapper for analyze_images."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.analyze_images(**kwargs))
        finally:
            loop.close()

    def analyze_text_sync(self, **kwargs) -> str:
        """Synchronous wrapper for analyze_text."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.analyze_text(**kwargs))
        finally:
            loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════════

def create_vlm_client(timeout: int = 120) -> VLMClient:
    """Create a VLMClient with default settings."""
    return VLMClient(default_timeout=timeout)


def get_model_configs_from_pipeline_config(cfg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract Qwen and GPT configs from pipeline_config module.

    Returns:
        (qwen_config, gpt_config) dicts ready for VLMClient
    """
    qwen_config = {
        'url': getattr(cfg, 'LM_STUDIO_URL', 'http://localhost:1234'),
        'model': getattr(cfg, 'LM_STUDIO_MODEL', os.environ.get('LM_STUDIO_MODEL', '')),
        'provider': 'lmstudio',
    }

    # Support multiple naming conventions for GPT model
    # Priority: GPT_MODEL > GPT5_MODEL > OPENAI_MODEL > default
    gpt_model = (
            getattr(cfg, 'GPT_MODEL', None) or
            getattr(cfg, 'GPT5_MODEL', None) or
            getattr(cfg, 'OPENAI_MODEL', None) or
            os.environ.get('GPT_MODEL') or
            os.environ.get('GPT5_MODEL') or
            os.environ.get('OPENAI_MODEL') or
            'gpt-5.4'
    )

    # Get API key from config or environment
    api_key = (
            getattr(cfg, 'OPENAI_API_KEY', None) or
            os.environ.get('OPENAI_API_KEY', '')
    )

    gpt_config = {
        'model': gpt_model,
        'api_key': api_key,
        'provider': 'openai',
    }

    return qwen_config, gpt_config


def get_gemini_config_from_pipeline_config(cfg: Any) -> Dict[str, Any]:
    """
    Extract Gemini cloud model config from pipeline_config module.

    Returns:
        Config dict ready for VLMClient with provider='gemini'
    """
    gemini_model = (
            getattr(cfg, 'GEMINI_MODEL', None) or
            os.environ.get('GEMINI_MODEL') or
            'gemini-3.1-pro-preview'
    )

    api_key = (
            getattr(cfg, 'GEMINI_API_KEY', None) or
            os.environ.get('GEMINI_API_KEY', '')
    )

    return {
        'model': gemini_model,
        'api_key': api_key,
        'provider': 'gemini',
    }


def get_pass_specific_gpt_config(cfg: Any, pass_key: str) -> Dict[str, Any]:
    """
    Get GPT config for a specific pass, allowing per-pass model overrides.

    Args:
        cfg: Pipeline config module
        pass_key: Pass identifier ('1b', '2a', '4', etc.)

    Returns:
        Config dict for the specific pass
    """
    _, base_gpt_config = get_model_configs_from_pipeline_config(cfg)

    # Check for pass-specific model override
    pass_model_key = f'GPT_PASS_{pass_key.upper()}_MODEL'
    pass_model = getattr(cfg, pass_model_key, None)

    if pass_model:
        return {**base_gpt_config, 'model': pass_model}

    return base_gpt_config
