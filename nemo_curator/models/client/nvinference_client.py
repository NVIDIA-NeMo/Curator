"""Thin NVIDIA Inference API client helpers.

This module intentionally contains only low-level utilities:
- Load `NVINFERENCE_API_KEY` from the environment (warn + raise if missing)
- Create an OpenAI-compatible client pointed at NVIDIA Inference API
- Collect streamed chat completion output into a single string
"""

import os
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = Any


def get_nvinference_api_key(env_var: str = "NVINFERENCE_API_KEY") -> str:
    """Load NVIDIA Inference API key from env.

    Raises:
        RuntimeError: If env var is missing/empty.
    """
    value = os.environ.get(env_var, "").strip()
    if not value:
        logger.warning(f"{env_var} is not set; OCR verification cannot run.")
        msg = f"{env_var} is not set"
        raise RuntimeError(msg)
    return value


def create_openai_client(*, api_key: str, base_url: str = "https://inference-api.nvidia.com") -> OpenAI:
    """Create an OpenAI-compatible client configured for NVIDIA Inference API.

    Note: Return type is `Any` to avoid importing `openai` at import-time for typing.
    """
    from openai import OpenAI  # local import to keep module thin

    return OpenAI(base_url=base_url, api_key=api_key)


def stream_chat_completion_text(  # noqa: PLR0913
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, object]],
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    extra_headers: dict[str, str] | None = None,
) -> str:
    """Run a streamed chat completion and collect output text.

    Mirrors the streaming collection pattern in `try_inference_nostream.py`.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
        extra_headers=extra_headers,
    )

    parts: list[str] = []
    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            parts.append(delta)
    return "".join(parts)
