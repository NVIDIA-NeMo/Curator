# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP helpers for the hosted TTS / ASR / LLM backends.

Uses ``httpx`` (a dependency of the ``openai`` package that NeMo Curator already requires), so the
backends add no new dependency. Only transient failures are retried: connection errors, HTTP 5xx
and 429.
"""

import os
import time

import httpx
from loguru import logger

_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_SERVER_ERROR = 500
_HTTP_CLIENT_ERROR = 400


def api_key(env_var: str) -> str:
    """Read an API key from ``env_var``; raise a clear error when it is missing."""
    value = os.environ.get(env_var)
    if not value:
        msg = f"Set the {env_var} environment variable to the API key of this endpoint."
        raise RuntimeError(msg)
    return value


def auth_header(env_var: str | None) -> dict[str, str]:
    """``Authorization: Bearer`` header from ``env_var`` when it is set; empty for keyless endpoints.

    Self-hosted NIMs take no key, hosted deployments usually do; this lets one backend serve both.
    """
    value = os.environ.get(env_var) if env_var else None
    return {"Authorization": f"Bearer {value}"} if value else {}


def build_url(base_url: str, route: str, model: str | None = None) -> str:
    """Join ``base_url`` and ``route``; ``route`` may contain a ``{model}`` placeholder.

    A NIM served directly uses its documented route unchanged. A gateway that fronts several NIMs
    often exposes each one under a per-model path; give that path as ``route`` with ``{model}``
    where the model name goes, and set ``model``.
    """
    if "{model}" in route:
        if not model:
            msg = f"route {route!r} has a {{model}} placeholder but no model was given"
            raise ValueError(msg)
        route = route.replace("{model}", model)
    return f"{base_url.rstrip('/')}/{route.lstrip('/')}"


def _is_transient(response: httpx.Response) -> bool:
    return response.status_code >= _HTTP_SERVER_ERROR or response.status_code == _HTTP_TOO_MANY_REQUESTS


def post_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 90.0,
    max_retries: int = 5,
    backoff_s: float = 2.0,
    **request_kwargs: object,
) -> httpx.Response:
    """POST ``url`` and return the response, retrying transient failures with linear backoff.

    ``request_kwargs`` are passed to :func:`httpx.post` (``files=``, ``data=``, ``json=`` ...).
    Non-transient HTTP errors (4xx other than 429) raise immediately.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.post(url, headers=headers or {}, timeout=timeout, **request_kwargs)
        except httpx.HTTPError as exc:
            last_error = exc
        else:
            if response.status_code < _HTTP_CLIENT_ERROR:
                return response
            detail = f"HTTP {response.status_code}: {response.text[:300]}"
            if not _is_transient(response):
                msg = f"POST {url} failed: {detail}"
                raise RuntimeError(msg)
            last_error = RuntimeError(detail)
        if attempt < max_retries:
            wait = backoff_s * attempt
            logger.warning(f"POST {url} failed ({attempt}/{max_retries}): {last_error}; retrying in {wait:.1f}s")
            time.sleep(wait)
    msg = f"POST {url} failed after {max_retries} attempts: {last_error}"
    raise RuntimeError(msg)


def require_wav(content: bytes, context: str) -> None:
    """Raise if ``content`` is not a RIFF/WAV payload (endpoints return JSON errors as 200 sometimes)."""
    if content[:4] != b"RIFF":
        msg = f"{context} did not return WAV audio: {content[:120]!r}"
        raise RuntimeError(msg)
