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

"""Embedding backends for the Expert Embedding Engine."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import re
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol


class EEEBackend(Protocol):
    def embed_raw(self, record: dict[str, Any], expert: str) -> list[float]: ...

    def embed_annotation(self, record: dict[str, Any], expert: str) -> list[float]: ...

    def embed_query(self, query: str, expert: str = "text-based") -> list[float]: ...


BackendName = Literal["hybrid", "local", "api"]
BackendFactory = Callable[[Any, Any], EEEBackend]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
SUPPORTED_EXPERTS = {"text-based", "fusion", "e2e"}
PHI4_MULTIMODAL_MODEL = "microsoft/phi-4-multimodal-instruct"
GEMMA_3N_E4B_MODEL = "google/gemma-3n-e4b-it"
AUDIO_URL_CHAT_MODELS = {PHI4_MULTIMODAL_MODEL, GEMMA_3N_E4B_MODEL}
NVCF_ASSET_UPLOAD_THRESHOLD_BYTES = 180 * 1024
AUDIO_INLINE_PREVIEW_BYTES = 160 * 1024
NVCF_ASSET_BASE_URL = "https://api.nvcf.nvidia.com/v2/nvcf"
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


class LocalEEEBackend:
    """Full local EEE backend: text expert, LanguageBind fusion, Omni-Embed e2e."""

    def __init__(self, config: Any | None = None, runtime: Any | None = None, embedding_dim: int = 2048):
        from omnifuse_tutorial.eee.local_models import FullLocalEEEBackend

        self._backend = FullLocalEEEBackend(config=config, runtime=runtime, embedding_dim=embedding_dim)

    def embed_raw(self, record: dict[str, Any], expert: str) -> list[float]:
        return self._backend.embed_raw(record, expert)

    def embed_annotation(self, record: dict[str, Any], expert: str) -> list[float]:
        return self._backend.embed_annotation(record, expert)

    def embed_query(self, query: str, expert: str = "text-based") -> list[float]:
        return self._backend.embed_query(query, expert)

    def unload(self) -> None:
        self._backend.unload()


class NvidiaApiEEEBackend:
    """NVIDIA API backend aligned with EmbedSim's API text expert path."""

    def __init__(
        self,
        embedding_dim: int = 2048,
        api_key: str | None = None,
        api_base_url: str = "https://integrate.api.nvidia.com/v1",
        text_model: str = "nvidia/nemotron-nano-12b-v2-vl",
        image_model: str = "nvidia/nemotron-nano-12b-v2-vl",
        video_model: str = "nvidia/nemotron-nano-12b-v2-vl",
        audio_model: str = GEMMA_3N_E4B_MODEL,
        embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2",
        timeout: int = 120,
        batch_size: int = 4,
    ):
        self.embedding_dim = embedding_dim
        self.api_key = api_key or os.environ.get("NV_BUILD_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key required. Set eee.nvidia_api_key or NV_BUILD_API_KEY.")
        self.api_base_url = os.environ.get("NVIDIA_API_BASE_URL", api_base_url).rstrip("/")
        self.text_model = text_model
        self.image_model = image_model
        self.video_model = video_model
        self.audio_model = audio_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.batch_size = max(1, min(int(batch_size), 16))

    def embed_raw(self, record: dict[str, Any], expert: str) -> list[float]:
        _validate_expert(expert)
        if expert == "text-based":
            text = self._describe_raw(record)
        else:
            # The sibling API toggle is primarily for the text expert. For the
            # other experts, preserve distinct expert spaces by embedding a
            # modality-aware textual representation through the API encoder.
            text = _raw_feature_text(record)
        return self._embed_text(text, expert)

    def embed_annotation(self, record: dict[str, Any], expert: str) -> list[float]:
        _validate_expert(expert)
        annotation = _text_or_empty(record.get("sns_annotation")) or _text_or_empty(record.get("annotation"))
        return self._embed_text(annotation, expert)

    def embed_query(self, query: str, expert: str = "text-based") -> list[float]:
        _validate_expert(expert)
        return self._embed_text(query, expert)

    def _describe_raw(self, record: dict[str, Any]) -> str:
        modality = str(record.get("modality") or "text")
        raw_text = _text_or_empty(record.get("sns_raw_text")) or _text_or_empty(record.get("raw_text"))
        if raw_text:
            return raw_text
        if modality == "text":
            return _read_text_path(record.get("raw_path")) or _raw_feature_text(record)

        raw_path = _path_or_none(record.get("raw_path"))
        if raw_path is None or not raw_path.exists():
            return _raw_feature_text(record)

        prompt = _prompt_for_modality(modality)
        if modality == "image":
            return self._describe_file(raw_path, self.image_model, "image_url", prompt)
        if modality == "audio":
            return self._describe_file(raw_path, self.audio_model, "input_audio", prompt)
        if modality == "video":
            return self._describe_file(raw_path, self.video_model, "video_url", prompt)
        return _raw_feature_text(record)

    def _describe_file(self, path: Path, model: str, content_type: str, prompt: str) -> str:
        return describe_file_with_nvidia_api(
            path=path,
            model=model,
            content_type=content_type,
            prompt=prompt,
            api_base_url=self.api_base_url,
            headers=self._headers(),
            timeout=self.timeout,
        )

    def _embed_text(self, text: str, expert: str) -> list[float]:
        import requests

        response = requests.post(
            f"{self.api_base_url}/embeddings",
            headers=self._headers(),
            json={
                "model": self.embedding_model,
                "input": [text],
                "input_type": "query" if expert == "text-based" else "passage",
                "encoding_format": "float",
                "truncate": "END",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        return _resize_and_normalize([float(item) for item in vector], self.embedding_dim)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }


class HybridEEEBackend:
    """API-first EEE backend with local fusion and true multimodal experts.

    The text-based expert uses NVIDIA API descriptions and text embeddings.
    The fusion and e2e experts stay local because the current tutorial code
    uses LanguageBind and Omni-Embed-Nemotron for those actual expert roles.
    """

    def __init__(self, config: Any, runtime: Any | None = None):
        self.api = NvidiaApiEEEBackend(
            embedding_dim=int(getattr(config, "embedding_dim", 2048)),
            api_key=getattr(config, "nvidia_api_key", None),
            api_base_url=getattr(config, "nvidia_api_base_url", "https://integrate.api.nvidia.com/v1"),
            text_model=getattr(config, "nvidia_text_describer_model", "nvidia/nemotron-nano-12b-v2-vl"),
            image_model=getattr(config, "nvidia_image_describer_model", "nvidia/nemotron-nano-12b-v2-vl"),
            video_model=getattr(config, "nvidia_video_describer_model", "nvidia/nemotron-nano-12b-v2-vl"),
            audio_model=getattr(config, "nvidia_audio_describer_model", GEMMA_3N_E4B_MODEL),
            embedding_model=getattr(config, "nvidia_embedding_model", "nvidia/llama-nemotron-embed-1b-v2"),
            batch_size=int(getattr(config, "batch_size", 4)),
        )
        self.local = LocalEEEBackend(
            config=config,
            runtime=runtime,
            embedding_dim=int(getattr(config, "embedding_dim", 2048)),
        )

    def embed_raw(self, record: dict[str, Any], expert: str) -> list[float]:
        expert = _validate_expert(expert)
        if expert == "text-based":
            return self.api.embed_raw(record, expert)
        return self.local.embed_raw(record, expert)

    def embed_annotation(self, record: dict[str, Any], expert: str) -> list[float]:
        expert = _validate_expert(expert)
        if expert == "text-based":
            return self.api.embed_annotation(record, expert)
        return self.local.embed_annotation(record, expert)

    def embed_query(self, query: str, expert: str = "text-based") -> list[float]:
        expert = _validate_expert(expert)
        if expert == "text-based":
            return self.api.embed_query(query, expert)
        return self.local.embed_query(query, expert)

    def describe_record(self, record: dict[str, Any]) -> str:
        return self.api._describe_raw(record)

    def unload(self) -> None:
        self.local.unload()


def describe_file_with_nvidia_api(
    *,
    path: Path,
    model: str,
    content_type: str,
    prompt: str,
    api_base_url: str,
    headers: dict[str, str],
    timeout: int,
) -> str:
    if content_type == "input_audio" and model in AUDIO_URL_CHAT_MODELS:
        return _describe_audio_url_chat_file(path, model, prompt, api_base_url, headers, timeout)
    return _describe_chat_completion_file(path, model, content_type, prompt, api_base_url, headers, timeout)


def _describe_chat_completion_file(
    path: Path,
    model: str,
    content_type: str,
    prompt: str,
    api_base_url: str,
    headers: dict[str, str],
    timeout: int,
) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    mime = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    if content_type == "input_audio":
        media_content = {"type": "input_audio", "input_audio": {"data": encoded, "format": _audio_format(path)}}
    else:
        media_content = {content_type: {"url": f"data:{mime};base64,{encoded}"}, "type": content_type}
    url = f"{api_base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, media_content]}],
        "max_tokens": 512,
        "temperature": 0.2,
        "stream": False,
    }
    response = _post_nvidia_json_with_retries(url, headers, payload, timeout)
    response = _resolve_nvidia_response(response, api_base_url, headers, timeout, model, url)
    return _response_text(response.json(), model, url)


def _describe_audio_url_chat_file(
    path: Path,
    model: str,
    prompt: str,
    api_base_url: str,
    headers: dict[str, str],
    timeout: int,
) -> str:
    audio_format = _audio_format(path)
    mime = "audio/wav" if audio_format == "wav" else "audio/mpeg"
    request_headers = dict(headers)
    if path.stat().st_size > NVCF_ASSET_UPLOAD_THRESHOLD_BYTES:
        asset_id = _upload_nvcf_asset(path, mime, headers, timeout)
        request_headers["NVCF-INPUT-ASSET-REFERENCES"] = asset_id
        content: str | list[dict[str, Any]] = (
            f'{prompt}\n<audio src="data:audio/{audio_format};asset_id,{asset_id}" />'
        )
    else:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        content = [
            {"type": "text", "text": prompt},
            {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
    url = f"{api_base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "temperature": 0.2,
        "stream": False,
    }
    response = _post_nvidia_json_with_retries(url, request_headers, payload, timeout)
    if _is_missing_nvcf_asset_response(response):
        content = _inline_audio_content(path, mime, prompt, preview=True)
        payload = {**payload, "messages": [{"role": "user", "content": content}]}
        response = _post_nvidia_json_with_retries(url, headers, payload, timeout)
    response = _resolve_nvidia_response(response, api_base_url, headers, timeout, model, url)
    return _response_text(response.json(), model, url)


def _upload_nvcf_asset(path: Path, mime: str, headers: dict[str, str], timeout: int) -> str:
    import requests

    description = f"omni-fuse tutorial audio asset {path.name}"
    asset_base_url = os.environ.get("NVIDIA_NVCF_ASSET_BASE_URL", NVCF_ASSET_BASE_URL).rstrip("/")
    create_url = f"{asset_base_url}/assets"
    response = requests.post(
        create_url,
        headers=headers,
        json={"contentType": mime, "description": description},
        timeout=timeout,
    )
    _raise_for_nvidia_response(response, "nvcf-asset", create_url)
    payload = response.json()
    asset_id = _first_string(payload, ("assetId", "asset_id", "id"))
    upload_url = _first_string(payload, ("uploadUrl", "upload_url", "uploadURL", "url"))
    if not asset_id or not upload_url:
        raise RuntimeError(f"NVIDIA NVCF asset response missing assetId/uploadUrl: {_safe_payload(payload)}")
    with path.open("rb") as handle:
        upload_response = requests.put(
            upload_url,
            data=handle,
            headers={
                "Content-Type": mime,
                "x-amz-meta-nvcf-asset-description": description,
            },
            timeout=timeout,
        )
    if upload_response.status_code >= 400:
        raise RuntimeError(
            f"NVIDIA NVCF asset upload failed: status={upload_response.status_code} body={upload_response.text[:1000]}"
        )
    return asset_id


def _inline_audio_content(path: Path, mime: str, prompt: str, preview: bool = False) -> list[dict[str, Any]]:
    if preview:
        prompt = (
            "Only the opening segment is attached because the full audio exceeds the current inline payload limit. "
            + prompt
        )
        audio_bytes = _audio_preview_bytes(path)
    else:
        audio_bytes = path.read_bytes()
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    return [
        {"type": "text", "text": prompt},
        {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{encoded}"}},
    ]


def _audio_preview_bytes(path: Path) -> bytes:
    if path.suffix.lower() != ".wav":
        return path.read_bytes()[:AUDIO_INLINE_PREVIEW_BYTES]
    try:
        with wave.open(str(path), "rb") as source:
            params = source.getparams()
            frame_size = max(1, params.nchannels * params.sampwidth)
            max_frames = max(1, min(params.nframes, (AUDIO_INLINE_PREVIEW_BYTES - 4096) // frame_size))
            frames = source.readframes(max_frames)
        output = io.BytesIO()
        with wave.open(output, "wb") as target:
            target.setnchannels(params.nchannels)
            target.setsampwidth(params.sampwidth)
            target.setframerate(params.framerate)
            target.setcomptype(params.comptype, params.compname)
            target.writeframes(frames)
        return output.getvalue()
    except (EOFError, OSError, wave.Error):
        return path.read_bytes()[:AUDIO_INLINE_PREVIEW_BYTES]


def _post_nvidia_json_with_retries(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
    attempts: int = 3,
) -> Any:
    import requests

    response = None
    for attempt in range(attempts):
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if not _is_retryable_nvidia_response(response) or attempt == attempts - 1:
            return response
        time.sleep(2 * (attempt + 1))
    return response


def _is_retryable_nvidia_response(response: Any) -> bool:
    return _is_degraded_response(response) or response.status_code in {500, 502, 503, 504}


def _is_degraded_response(response: Any) -> bool:
    return response.status_code in {400, 503} and "DEGRADED function" in response.text


def _is_missing_nvcf_asset_response(response: Any) -> bool:
    return response.status_code == 400 and "not found in nvcf_assets" in response.text


def _resolve_nvidia_response(
    response: Any,
    api_base_url: str,
    headers: dict[str, str],
    timeout: int,
    model: str,
    url: str,
) -> Any:
    import requests

    if response.status_code != 202:
        _raise_for_nvidia_response(response, model, url)
        return response

    request_id = _request_id(response)
    if not request_id:
        raise RuntimeError(f"NVIDIA API returned 202 without requestId for model {model}: {response.text[:1000]}")

    status_url = f"{api_base_url}/status/{request_id}"
    deadline = time.monotonic() + max(timeout, 1)
    while True:
        poll_response = requests.get(status_url, headers=headers, timeout=timeout)
        if poll_response.status_code == 202:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"NVIDIA API polling timed out for model {model} request {request_id}")
            time.sleep(2)
            continue
        _raise_for_nvidia_response(poll_response, model, status_url)
        return poll_response


def _raise_for_nvidia_response(response: Any, model: str, url: str) -> None:
    if response.status_code < 400:
        return
    raise RuntimeError(
        f"NVIDIA API request failed: model={model} url={url} status={response.status_code} body={response.text[:1000]}"
    )


def _request_id(response: Any) -> str | None:
    for header in ("NVCF-REQID", "x-request-id", "X-Request-Id", "requestId"):
        value = response.headers.get(header)
        if value:
            return str(value)
    try:
        payload = response.json()
    except ValueError:
        return None
    return _first_string(payload, ("requestId", "request_id", "id"))


def _audio_format(path: Path) -> str:
    audio_format = path.suffix.lower().lstrip(".")
    if audio_format == "mpeg":
        audio_format = "mp3"
    if audio_format not in {"wav", "mp3"}:
        raise ValueError(f"NVIDIA audio descriptions support wav/mp3 only, got {path.suffix} for {path}")
    return audio_format


def _response_text(payload: Any, model: str, url: str) -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                text = _content_text(content)
                if text:
                    return text
        for key in ("content", "text", "output", "response"):
            text = _content_text(payload.get(key))
            if text:
                return text
        outputs = payload.get("outputs")
        if isinstance(outputs, list) and outputs:
            text = _content_text(outputs[0])
            if text:
                return text
    raise RuntimeError(
        f"NVIDIA API response missing text content: model={model} url={url} body={_safe_payload(payload)}"
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()
    if isinstance(content, dict):
        value = content.get("text") or content.get("content")
        if isinstance(value, str):
            return value.strip()
    return ""


def _first_string(payload: Any, keys: tuple[str, ...]) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _safe_payload(payload: Any) -> str:
    text = str(payload)
    return text[:1000]


def backend_factory(config_or_name: Any, runtime: Any | None = None) -> EEEBackend:
    name = _backend_name(config_or_name)
    if name == "hybrid":
        if isinstance(config_or_name, str):
            raise ValueError("Hybrid EEE backend requires a full EEE config")
        return HybridEEEBackend(config_or_name, runtime)
    if name == "local":
        return LocalEEEBackend(
            config=config_or_name if not isinstance(config_or_name, str) else None,
            runtime=runtime,
            embedding_dim=int(getattr(config_or_name, "embedding_dim", 2048)),
        )
    if name == "api":
        return NvidiaApiEEEBackend(
            embedding_dim=int(getattr(config_or_name, "embedding_dim", 2048)),
            api_key=getattr(config_or_name, "nvidia_api_key", None),
            api_base_url=getattr(config_or_name, "nvidia_api_base_url", "https://integrate.api.nvidia.com/v1"),
            text_model=getattr(config_or_name, "nvidia_text_describer_model", "nvidia/nemotron-nano-12b-v2-vl"),
            image_model=getattr(config_or_name, "nvidia_image_describer_model", "nvidia/nemotron-nano-12b-v2-vl"),
            video_model=getattr(config_or_name, "nvidia_video_describer_model", "nvidia/nemotron-nano-12b-v2-vl"),
            audio_model=getattr(config_or_name, "nvidia_audio_describer_model", GEMMA_3N_E4B_MODEL),
            embedding_model=getattr(
                config_or_name,
                "nvidia_embedding_model",
                "nvidia/llama-nemotron-embed-1b-v2",
            ),
            batch_size=int(getattr(config_or_name, "batch_size", 4)),
        )
    raise ValueError(f"Unsupported EEE backend: {name}")


def _backend_name(config_or_name: Any) -> str:
    if isinstance(config_or_name, str):
        return config_or_name
    return str(getattr(config_or_name, "backend", "hybrid"))


def _validate_expert(expert: str) -> str:
    if expert not in SUPPORTED_EXPERTS:
        raise ValueError(f"Unsupported EEE expert: {expert}")
    return expert


def _describe_raw(record: dict[str, Any]) -> str:
    modality = str(record.get("modality") or "text")
    raw_text = _text_or_empty(record.get("sns_raw_text")) or _text_or_empty(record.get("raw_text"))
    if raw_text:
        return raw_text
    if modality == "text":
        return _read_text_path(record.get("raw_path")) or _raw_feature_text(record)
    return " ".join([_prompt_for_modality(modality), _raw_feature_text(record)])


def _raw_feature_text(record: dict[str, Any]) -> str:
    raw_path = _path_or_none(record.get("raw_path"))
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    parts = [
        f"modality:{record.get('modality')}",
        f"pool:{record.get('pool')}",
        f"path:{raw_path.name if raw_path else record.get('raw_path')}",
    ]
    if raw_path:
        parts.append(_file_fingerprint(raw_path))
        parts.extend(_path_tokens(raw_path))
    for key in sorted(metadata):
        value = metadata[key]
        if value is not None and value != "":
            parts.append(f"{key}:{value}")
    raw_text = _text_or_empty(record.get("sns_raw_text")) or _text_or_empty(record.get("raw_text"))
    if raw_text:
        parts.append(raw_text)
    return " ".join(parts)


def _prompt_for_modality(modality: str) -> str:
    if modality == "image":
        return "Describe this image in detail."
    if modality == "audio":
        return "Transcribe and describe this audio. What sounds do you hear?"
    if modality == "video":
        return "Describe what happens in this video in detail."
    return "Describe this content in detail."


def _file_fingerprint(path: Path) -> str:
    try:
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(65536))
        return f"file:{path.suffix.lower()} size:{stat.st_size} sha256:{digest.hexdigest()[:24]}"
    except OSError:
        return f"file:{path.suffix.lower()} unreadable"


def _path_tokens(path: Path) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", path.stem)


def _read_text_path(value: Any) -> str:
    path = _path_or_none(value)
    if not path or path.suffix.lower() not in TEXT_EXTENSIONS:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def _path_or_none(value: Any) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    try:
        path = Path(value)
    except OSError:
        return None
    return path


def _text_or_empty(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _resize_and_normalize(vector: list[float], dim: int) -> list[float]:
    if len(vector) < dim:
        vector = vector + [0.0] * (dim - len(vector))
    elif len(vector) > dim:
        vector = vector[:dim]
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0:
        return vector
    return [item / norm for item in vector]
