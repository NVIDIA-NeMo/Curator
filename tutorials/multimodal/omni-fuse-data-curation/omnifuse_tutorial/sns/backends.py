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

"""SNS similarity and description backends."""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any, Protocol

from omnifuse_tutorial.data.io import cosine_similarity
from omnifuse_tutorial.eee.backends import (
    GEMMA_3N_E4B_MODEL,
    _post_nvidia_json_with_retries,
    describe_file_with_nvidia_api,
)


class SNSBackend(Protocol):
    def image_text(self, image_data: Any, text: str) -> float: ...

    def audio_text(self, audio_data: Any, text: str) -> float: ...

    def video_text(self, video_data: Any, text: str) -> float: ...

    def text_text(self, left: Any, right: Any, dim: int = 2048) -> float: ...

    def text_text_matrix(self, texts_a: list[str], texts_b: list[str], batch_size: int = 16) -> Any: ...

    def describe_record(self, record: dict[str, Any]) -> str: ...

    def forward_media(self, record: dict[str, Any], annotation: str) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def unload(self) -> None: ...


class LocalSNSBackend:
    """Full local SNS backend using Omni-Embed similarity and forward extractors."""

    def __init__(self, sns_config: Any | None = None, eee_config: Any | None = None, runtime: Any | None = None):
        if sns_config is None:
            raise ValueError("LocalSNSBackend requires a full SNS config")
        self.sns_config = sns_config
        self.eee_config = eee_config
        self.embedding_dim = int(getattr(eee_config, "embedding_dim", 2048))
        from omnifuse_tutorial.eee.local_models import (
            FullLocalEEEBackend,
            OmniEmbedNemotronRuntime,
            resolve_device,
            resolve_offline_mode,
        )
        from omnifuse_tutorial.sns.full_forward import ForwardModelStore

        self.device = resolve_device(runtime)
        self.offline_mode = resolve_offline_mode(runtime)
        self.omni_model = str(getattr(sns_config, "nvidia_model", "nvidia/omni-embed-nemotron-3b"))
        self._omni_runtime_cls = OmniEmbedNemotronRuntime
        self._omni: Any | None = None
        self._text_embedding_cache: dict[str, list[float]] = {}
        self._description_backend = FullLocalEEEBackend(
            config=eee_config, runtime=runtime, embedding_dim=self.embedding_dim
        )
        self._forward_models = ForwardModelStore(sns_config, self.device, self.offline_mode)
        self.require_forward_models = bool(getattr(sns_config, "require_forward_models", True))

    def image_text(self, image_data: Any, text: str) -> float:
        return cosine_similarity(self._embed_media(image_data, "image"), self._embed_text(text))

    def audio_text(self, audio_data: Any, text: str) -> float:
        return cosine_similarity(self._embed_media(audio_data, "audio"), self._embed_text(text))

    def video_text(self, video_data: Any, text: str) -> float:
        return cosine_similarity(self._embed_media(video_data, "video"), self._embed_text(text))

    def text_text(self, left: Any, right: Any, dim: int = 2048) -> float:
        left = _text_or_none(left) or ""
        right = _text_or_none(right) or ""
        if not left or not right:
            return 0.0
        return cosine_similarity(self._embed_text(left), self._embed_text(right))

    def text_text_matrix(self, texts_a: list[str], texts_b: list[str], batch_size: int = 16) -> Any:
        import numpy as np

        return np.array([[self.text_text(left, right) for right in texts_b] for left in texts_a], dtype=np.float32)

    def describe_record(self, record: dict[str, Any]) -> str:
        return self._description_backend.describe_record(record)

    def forward_media(self, record: dict[str, Any], annotation: str) -> tuple[dict[str, Any], dict[str, Any]]:
        from omnifuse_tutorial.sns.full_forward import (
            forward_extract_audio,
            forward_extract_image,
            forward_extract_video,
        )

        modality = str(record.get("modality") or "")
        raw_value = record.get("raw_path")
        try:
            if modality == "image":
                candidate, extraction = forward_extract_image(raw_value, annotation, self.sns_config)
            elif modality == "audio":
                candidate, extraction = forward_extract_audio(
                    raw_value, annotation, self.sns_config, self._forward_models
                )
            elif modality == "video":
                candidate, extraction = forward_extract_video(
                    raw_value, annotation, self.sns_config, self._forward_models
                )
            else:
                return record, {
                    "direction": "forward",
                    "accepted": False,
                    "reason": "unsupported_media_modality",
                    "modality": modality,
                }
        except RuntimeError as exc:
            if self.require_forward_models:
                raise
            return record, {
                "direction": "forward",
                "accepted": False,
                "reason": "forward_model_unavailable",
                "modality": modality,
                "error": str(exc),
            }

        if not _changed_media(raw_value, candidate):
            return record, {
                "direction": "forward",
                "accepted": False,
                "reason": extraction.get("reason", "no_media_nucleus"),
                "modality": modality,
                "extraction": _jsonable(extraction),
            }

        original_sim = self._raw_annotation_similarity(raw_value, modality, annotation)
        candidate_sim = self._raw_annotation_similarity(candidate, modality, annotation)
        mi_threshold = float(getattr(self.sns_config, "mi_ratio", 0.95)) * max(
            original_sim,
            float(getattr(self.sns_config, "mi_eps", 0.05)),
        )
        if candidate_sim < mi_threshold:
            return record, {
                "direction": "forward",
                "accepted": False,
                "reason": "mi_gate_failed",
                "modality": modality,
                "original_similarity": original_sim,
                "candidate_similarity": candidate_sim,
                "mi_threshold": mi_threshold,
                "extraction": _jsonable(extraction),
            }

        updated = dict(record)
        updated["raw_path"] = str(candidate)
        updated["sns_raw_text"] = None
        return updated, {
            "direction": "forward",
            "accepted": True,
            "reason": extraction.get("reason", "media_nucleus_extracted"),
            "modality": modality,
            "original_similarity": original_sim,
            "candidate_similarity": candidate_sim,
            "mi_threshold": mi_threshold,
            "output_path": str(candidate),
            "extraction": _jsonable(extraction),
        }

    def raw_annotation_similarity(self, raw_value: Any, modality: str, annotation: str) -> float:
        return self._raw_annotation_similarity(raw_value, modality, annotation)

    def unload(self) -> None:
        self._text_embedding_cache.clear()
        self._description_backend.unload()
        self._forward_models.unload()
        if self._omni is not None:
            self._omni.unload()
            self._omni = None

    def _raw_annotation_similarity(self, raw_value: Any, modality: str, annotation: str) -> float:
        if modality == "text":
            return self.text_text(raw_value, annotation, self.embedding_dim)
        if modality == "image":
            return self.image_text(raw_value, annotation)
        if modality == "audio":
            return self.audio_text(raw_value, annotation)
        if modality == "video":
            return self.video_text(raw_value, annotation)
        return 0.0

    def _embed_text(self, text: str) -> list[float]:
        if text not in self._text_embedding_cache:
            self._text_embedding_cache[text] = self._resize_vector(self._ensure_omni().encode_text(text))
        return self._text_embedding_cache[text]

    def _embed_media(self, value: Any, modality: str) -> list[float]:
        omni = self._ensure_omni()
        if modality == "image":
            return self._resize_vector(omni.encode_image(value))
        if modality == "audio":
            return self._resize_vector(omni.encode_audio(value))
        if modality == "video":
            return self._resize_vector(omni.encode_video(value))
        raise ValueError(f"Unsupported media modality: {modality}")

    def _ensure_omni(self) -> Any:
        if self._omni is None:
            self._omni = self._omni_runtime_cls(self.omni_model, self.device, self.offline_mode)
        return self._omni

    def _resize_vector(self, vector: Any) -> list[float]:
        from omnifuse_tutorial.eee.local_models import _resize_and_normalize

        return _resize_and_normalize(vector, self.embedding_dim)


class NvidiaApiSNSBackend:
    """NVIDIA API-backed SNS similarity/describer backend.

    This backend provides the model-backed backward extraction path and a
    conservative media forward path that writes a generated text nucleus to
    `sns_raw_text` when the media description is sufficiently aligned. It does
    not crop pixels or cut media segments; local Grounding-DINO/DETR-style
    extraction still requires those optional model stacks.
    """

    def __init__(
        self,
        sns_config: Any,
        eee_config: Any,
        timeout: int = 120,
    ):
        self.sns_config = sns_config
        self.embedding_dim = int(getattr(eee_config, "embedding_dim", 2048))
        self.api_key = (
            getattr(eee_config, "nvidia_api_key", None)
            or os.environ.get("NV_BUILD_API_KEY")
            or os.environ.get("NVIDIA_API_KEY")
        )
        if not self.api_key:
            raise ValueError("NVIDIA API key required. Set NV_BUILD_API_KEY in .env or the environment.")
        base_url = getattr(eee_config, "nvidia_api_base_url", "https://integrate.api.nvidia.com/v1")
        self.api_base_url = os.environ.get("NVIDIA_API_BASE_URL", base_url).rstrip("/")
        self.text_model = getattr(eee_config, "nvidia_text_describer_model", "nvidia/nemotron-nano-12b-v2-vl")
        self.image_model = getattr(eee_config, "nvidia_image_describer_model", "nvidia/nemotron-nano-12b-v2-vl")
        self.video_model = getattr(eee_config, "nvidia_video_describer_model", "nvidia/nemotron-nano-12b-v2-vl")
        self.audio_model = getattr(eee_config, "nvidia_audio_describer_model", GEMMA_3N_E4B_MODEL)
        self.embedding_model = getattr(
            eee_config,
            "nvidia_embedding_model",
            "nvidia/llama-nemotron-embed-1b-v2",
        )
        self.timeout = timeout
        self._embedding_cache: dict[str, list[float]] = {}
        self._description_cache: dict[str, str] = {}

    def image_text(self, image_data: Any, text: str) -> float:
        return self.text_text(self._describe_media_value(image_data, "image"), text)

    def audio_text(self, audio_data: Any, text: str) -> float:
        return self.text_text(self._describe_media_value(audio_data, "audio"), text)

    def video_text(self, video_data: Any, text: str) -> float:
        return self.text_text(self._describe_media_value(video_data, "video"), text)

    def text_text(self, left: Any, right: Any, dim: int = 2048) -> float:
        left = _text_or_none(left) or ""
        right = _text_or_none(right) or ""
        if not left or not right:
            return 0.0
        return cosine_similarity(self._embed_text(left), self._embed_text(right))

    def text_text_matrix(self, texts_a: list[str], texts_b: list[str], batch_size: int = 16) -> Any:
        import numpy as np

        return np.array(
            [[self.text_text(left, right) for right in texts_b] for left in texts_a],
            dtype=np.float32,
        )

    def describe_record(self, record: dict[str, Any]) -> str:
        raw_text = _text_or_none(record.get("sns_raw_text")) or _text_or_none(record.get("raw_text"))
        if raw_text:
            return raw_text
        return self._describe_media_value(record.get("raw_path"), str(record.get("modality") or "text"))

    def forward_media(self, record: dict[str, Any], annotation: str) -> tuple[dict[str, Any], dict[str, Any]]:
        modality = str(record.get("modality") or "")
        description = self.describe_record(record)
        similarity = self.text_text(description, annotation)
        threshold = _forward_threshold(self.sns_config, modality)
        if similarity < threshold:
            return record, {
                "direction": "forward",
                "accepted": False,
                "reason": "media_description_below_threshold",
                "modality": modality,
                "similarity": similarity,
                "threshold": threshold,
            }
        updated = dict(record)
        updated["sns_raw_text"] = description
        return updated, {
            "direction": "forward",
            "accepted": True,
            "reason": "media_description_nucleus_api_backend",
            "modality": modality,
            "similarity": similarity,
            "threshold": threshold,
        }

    def unload(self) -> None:
        self._embedding_cache.clear()
        self._description_cache.clear()

    def _describe_media_value(self, value: Any, modality: str) -> str:
        if modality == "text":
            return _text_or_none(value) or ""
        path = _path_or_none(value)
        if path is None or not path.exists():
            return _describe_media(value, modality)
        cache_key = f"{modality}:{path}"
        if cache_key in self._description_cache:
            return self._description_cache[cache_key]
        prompt = _prompt_for_modality(modality)
        if modality == "image":
            description = self._describe_file(path, self.image_model, "image_url", prompt)
        elif modality == "audio":
            description = self._describe_file(path, self.audio_model, "input_audio", prompt)
        elif modality == "video":
            description = self._describe_file(path, self.video_model, "video_url", prompt)
        else:
            description = _describe_media(path, modality)
        self._description_cache[cache_key] = description
        return description

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

    def _embed_text(self, text: str) -> list[float]:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        response = _post_nvidia_json_with_retries(
            url=f"{self.api_base_url}/embeddings",
            headers=self._headers(),
            payload={
                "model": self.embedding_model,
                "input": [text],
                "input_type": "passage",
                "encoding_format": "float",
                "truncate": "END",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        vector = [float(item) for item in response.json()["data"][0]["embedding"]]
        vector = _resize_and_normalize(vector, self.embedding_dim)
        self._embedding_cache[text] = vector
        return vector

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }


class HybridSNSBackend:
    """API-first SNS backend with local forward extraction and multimodal scoring.

    Backward extraction and text-text decisions use NVIDIA API descriptions and
    text embeddings. Image, audio, and video forward extraction stay local
    because those steps require Grounding-DINO, AM-DETR, and CG-DETR.
    """

    def __init__(self, sns_config: Any, eee_config: Any, runtime: Any | None = None):
        self.api = NvidiaApiSNSBackend(sns_config, eee_config)
        self.local = LocalSNSBackend(sns_config, eee_config, runtime)
        self.embedding_dim = int(getattr(eee_config, "embedding_dim", 2048))

    def image_text(self, image_data: Any, text: str) -> float:
        return self.local.image_text(image_data, text)

    def audio_text(self, audio_data: Any, text: str) -> float:
        return self.local.audio_text(audio_data, text)

    def video_text(self, video_data: Any, text: str) -> float:
        return self.local.video_text(video_data, text)

    def text_text(self, left: Any, right: Any, dim: int = 2048) -> float:
        return self.api.text_text(left, right, dim)

    def text_text_matrix(self, texts_a: list[str], texts_b: list[str], batch_size: int = 16) -> Any:
        return self.api.text_text_matrix(texts_a, texts_b, batch_size)

    def describe_record(self, record: dict[str, Any]) -> str:
        return self.api.describe_record(record)

    def forward_media(self, record: dict[str, Any], annotation: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.local.forward_media(record, annotation)

    def raw_annotation_similarity(self, raw_value: Any, modality: str, annotation: str) -> float:
        if modality == "text":
            return self.api.text_text(raw_value, annotation, self.embedding_dim)
        return self.local.raw_annotation_similarity(raw_value, modality, annotation)

    def unload(self) -> None:
        self.api.unload()
        self.local.unload()


def backend_factory(
    config_or_name: Any = "local", eee_config: Any | None = None, runtime: Any | None = None
) -> SNSBackend:
    name = _sns_backend_name(config_or_name, eee_config)
    if name == "hybrid":
        if isinstance(config_or_name, str):
            raise ValueError("Hybrid SNS backend requires a full SNS config")
        return HybridSNSBackend(config_or_name, eee_config, runtime)
    if name == "local":
        return LocalSNSBackend(config_or_name if not isinstance(config_or_name, str) else None, eee_config, runtime)
    if name == "api":
        return NvidiaApiSNSBackend(config_or_name, eee_config)
    raise ValueError(f"Unsupported SNS backend: {name}")


def _sns_backend_name(config_or_name: Any, eee_config: Any | None) -> str:
    if isinstance(config_or_name, str):
        name = config_or_name
    else:
        name = str(getattr(config_or_name, "backend", "auto"))
    if name == "auto":
        return str(getattr(eee_config, "backend", "hybrid"))
    if name in {"hybrid", "local", "api"}:
        return name
    raise ValueError(f"Unsupported SNS backend: {name}")


def _text_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _path_or_none(value: Any) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    try:
        return Path(value)
    except OSError:
        return None


def _changed_media(original: Any, candidate: Any) -> bool:
    if candidate is None:
        return False
    return str(original) != str(candidate)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _path_tokens(path: Path) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", path.stem)


def _describe_media(value: Any, modality: str) -> str:
    if isinstance(value, dict):
        file_path = value.get("file_path") or value.get("path") or value.get("raw_path")
        if file_path:
            path = _path_or_none(file_path)
            if path:
                return f"modality {modality} {' '.join(_path_tokens(path))} {_file_fingerprint(path)}"
    path = _path_or_none(value)
    if path:
        return f"modality {modality} {' '.join(_path_tokens(path))} {_file_fingerprint(path)}"
    return f"modality {modality} {value!r}"


def _file_fingerprint(path: Path) -> str:
    try:
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(65536))
        return f"{path.suffix.lower()} {stat.st_size} {digest.hexdigest()[:24]}"
    except OSError:
        return f"{path.suffix.lower()} unreadable"


def _prompt_for_modality(modality: str) -> str:
    if modality == "image":
        return "Describe the annotation-relevant visual content in this image in detail."
    if modality == "audio":
        return "Transcribe and describe the annotation-relevant sounds in this audio in detail."
    if modality == "video":
        return "Describe the annotation-relevant events and actions in this video in detail."
    return "Describe this content in detail."


def _forward_threshold(config: Any, modality: str) -> float:
    if modality == "image":
        return float(getattr(config, "tau_forward_image", 0.30))
    if modality == "audio":
        return float(getattr(config, "tau_forward_audio", 0.25))
    if modality == "video":
        return float(getattr(config, "tau_forward_video", 0.20))
    return float(getattr(config, "tau_forward_text", 0.30))


def _resize_and_normalize(vector: list[float], dim: int) -> list[float]:
    if len(vector) < dim:
        vector = vector + [0.0] * (dim - len(vector))
    elif len(vector) > dim:
        vector = vector[:dim]
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0:
        return vector
    return [item / norm for item in vector]
