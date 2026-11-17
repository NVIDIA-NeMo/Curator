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

"""NVIDIA Build API client: chat-completion media descriptions and text embeddings."""

from __future__ import annotations

import base64
import io
import time
import wave
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from pathlib import Path

    from omnifuse_tutorial.config import EEEConfig

AUDIO_INLINE_PREVIEW_BYTES = 160 * 1024
RETRYABLE_STATUS = {500, 502, 503, 504}
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
}


class NvidiaApi:
    def __init__(self, eee: EEEConfig, timeout: int = 120):
        if eee.needs_api_key:
            raise ValueError("NVIDIA API key required. Set NV_BUILD_API_KEY in .env or eee.nvidia_api_key.")
        self.base_url, self.embedding_url = eee.base_urls
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if eee.api_key:
            self.headers["Authorization"] = f"Bearer {eee.api_key}"
        self.nim_embeddings = False
        self.models = {
            "image": eee.nvidia_image_describer_model,
            "audio": eee.nvidia_audio_describer_model,
            "video": eee.nvidia_video_describer_model,
        }
        self.embedding_model, self.timeout = eee.nvidia_embedding_model, timeout

    def embed(self, texts: list[str], input_type: str, batch_size: int = 1) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            response = self._post(
                f"{self.embedding_url}/embeddings", self.headers, self._embed_payload(chunk, input_type)
            )
            # NIM rejects the OpenAI-style payload; vLLM silently ignores input_type, so its prefix goes in the text.
            if not self.nim_embeddings and response.status_code == 400 and "input_type" in response.text:
                self.nim_embeddings = True
                response = self._post(
                    f"{self.embedding_url}/embeddings", self.headers, self._embed_payload(chunk, input_type)
                )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: int(item.get("index", 0)))
            if len(data) != len(chunk):
                raise RuntimeError(f"NVIDIA embeddings returned {len(data)} vectors for {len(chunk)} texts")
            vectors += [[float(value) for value in item["embedding"]] for item in data]
        return vectors

    def _embed_payload(self, chunk: list[str], input_type: str) -> dict[str, Any]:
        payload = {"model": self.embedding_model, "encoding_format": "float"}
        if self.nim_embeddings:
            return {**payload, "input": chunk, "input_type": input_type, "truncate": "END"}
        return {**payload, "input": [f"{input_type}: {text}" for text in chunk]}

    def describe(self, path: Path, modality: str, prompt: str) -> str:
        """Describe an image, audio, or video file sent inline as base64."""
        model, url = self.models[modality], f"{self.base_url}/chat/completions"
        mime = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
        if modality == "audio":
            return self._describe_audio(path, model, prompt, url)
        # shortcut: no NVCF asset upload (that API is retired), so media over the inline size limit fails; resize first
        kind, encoded = f"{modality}_url", base64.b64encode(path.read_bytes()).decode("utf-8")
        content = [
            {"type": "text", "text": prompt},
            {kind: {"url": f"data:{mime};base64,{encoded}"}, "type": kind},
        ]
        response = self._post(url, self.headers, _chat(model, [{"role": "user", "content": content}]))
        return _response_text(self._resolve(response, self.headers, model, url).json(), model, url)

    def _describe_audio(self, path: Path, model: str, prompt: str, url: str) -> str:
        """Audio chat models: inline audio, then an inline opening preview if the full file is too large."""
        payload = _chat(model, [{"role": "user", "content": _inline_audio(path, prompt)}])
        response = self._post(url, self.headers, payload)
        if response.status_code == 413 or (
            response.status_code == 400
            and any(hint in response.text.lower() for hint in ("payload", "too large", "request size", "audio_url"))
        ):
            preview = _inline_audio(path, prompt, preview=True)
            response = self._post(url, self.headers, {**payload, "messages": [{"role": "user", "content": preview}]})
        return _response_text(self._resolve(response, self.headers, model, url).json(), model, url)

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any], attempts: int = 3) -> Any:
        for attempt in range(attempts):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.Timeout:
                if attempt == attempts - 1:
                    raise
            else:
                degraded = response.status_code in {400, 503} and "DEGRADED function" in response.text
                if not (degraded or response.status_code in RETRYABLE_STATUS) or attempt == attempts - 1:
                    break
            time.sleep(2 * (attempt + 1))
        return response

    def _resolve(self, response: Any, headers: dict[str, str], model: str, url: str) -> Any:
        """Raise on HTTP errors and poll 202 (accepted) responses until the result is ready."""
        if response.status_code != 202:
            _raise_for_status(response, model, url)
            return response
        headers_in = response.headers
        request_id = headers_in.get("NVCF-REQID") or headers_in.get("x-request-id") or headers_in.get("requestId")
        if not request_id:
            try:
                request_id = _first_string(response.json(), ("requestId", "request_id", "id"))
            except ValueError:
                request_id = None
        if not request_id:
            raise RuntimeError(f"NVIDIA API returned 202 without requestId for model {model}: {response.text[:1000]}")
        status_url = f"{self.base_url}/status/{request_id}"
        deadline = time.monotonic() + max(self.timeout, 1)
        while (response := requests.get(status_url, headers=headers, timeout=self.timeout)).status_code == 202:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"NVIDIA API polling timed out for model {model} request {request_id}")
            time.sleep(2)
        _raise_for_status(response, model, status_url)
        return response


def _chat(model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2,
        "stream": False,
        # The hosted reasoning describer ignores /no_think and otherwise spends max_tokens thinking, leaving no content.
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _inline_audio(path: Path, prompt: str, preview: bool = False) -> list[dict[str, Any]]:
    if preview:
        prompt = (
            "Only the opening segment is attached because the full audio exceeds the current inline payload limit. "
            + prompt
        )
    encoded = base64.b64encode(_audio_preview(path) if preview else path.read_bytes()).decode("utf-8")
    return [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": encoded, "format": _audio_format(path)}},
    ]


def _audio_preview(path: Path) -> bytes:
    """Re-headered opening frames of a WAV file, or the leading bytes of anything else."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as source:
                params = source.getparams()
                frame_size = max(1, params.nchannels * params.sampwidth)
                frames = source.readframes(
                    max(1, min(params.nframes, (AUDIO_INLINE_PREVIEW_BYTES - 4096) // frame_size))
                )
            output = io.BytesIO()
            with wave.open(output, "wb") as target:
                target.setparams(params._replace(nframes=0))
                target.writeframes(frames)
            return output.getvalue()
        except (EOFError, OSError, wave.Error):
            pass
    return path.read_bytes()[:AUDIO_INLINE_PREVIEW_BYTES]


def _audio_format(path: Path) -> str:
    audio_format = path.suffix.lower().lstrip(".")
    audio_format = "mp3" if audio_format == "mpeg" else audio_format
    if audio_format not in {"wav", "mp3"}:
        raise ValueError(f"NVIDIA audio descriptions support wav/mp3 only, got {path.suffix} for {path}")
    return audio_format


def _raise_for_status(response: Any, model: str, url: str) -> None:
    if response.status_code >= 400:
        raise RuntimeError(
            f"NVIDIA API request failed: model={model} url={url} status={response.status_code} body={response.text[:1000]}"
        )


def _response_text(payload: Any, model: str, url: str) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    content = choices[0].get("message", {}).get("content") if choices else None
    if isinstance(content, str) and content.strip():
        return content.strip()
    raise RuntimeError(f"NVIDIA API response missing text content: model={model} url={url} body={str(payload)[:1000]}")


def _first_string(payload: Any, keys: tuple[str, ...]) -> str | None:
    values = [payload.get(key) for key in keys] if isinstance(payload, dict) else []
    return next((value for value in values if isinstance(value, str) and value), None)
