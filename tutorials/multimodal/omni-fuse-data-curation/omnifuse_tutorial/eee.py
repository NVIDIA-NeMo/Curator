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

"""Expert Embedding Engine: text-based, fusion (LanguageBind), and e2e (Omni-Embed) experts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from omnifuse_tutorial.data import annotation_text, fingerprint, raw_text, unit
from omnifuse_tutorial.models import LanguageBind, OmniEmbed, resolve_device
from omnifuse_tutorial.nvidia_api import NvidiaApi

if TYPE_CHECKING:
    from omnifuse_tutorial.config import EEEConfig, RuntimeConfig

EEE_PROMPTS = {
    "image": "Describe this image in detail.",
    "audio": "Transcribe and describe this audio. What sounds do you hear?",
    "video": "Describe what happens in this video in detail.",
}


class EEEBackend:
    """The text-based expert embeds NVIDIA API descriptions with the API text encoder; fusion and e2e run locally."""

    def __init__(self, eee: EEEConfig, runtime: RuntimeConfig):
        self.eee, self.dim, self.api = eee, eee.embedding_dim, NvidiaApi(eee)
        self.device = resolve_device(runtime)
        self.omni = OmniEmbed(eee.nvidia_multimodal_model, self.device)
        self.languagebind = self.adapter = None

    def embed_raw(self, record: dict[str, Any], expert: str) -> np.ndarray:
        if expert == "text-based":
            return self.embed_query(self._describe(record), expert)
        modality = record["modality"]
        value = raw_value(record, modality)
        if modality == "text":
            return self.embed_query(value, expert)
        if expert == "fusion":
            return self._adapt(self._languagebind().encode_media(value, modality))
        return unit(self.omni.encode_media(value, modality), self.dim)

    def embed_annotation(self, record: dict[str, Any], expert: str) -> np.ndarray:
        return self.embed_query(annotation_text(record), expert)

    def embed_query(self, text: str, expert: str = "text-based") -> np.ndarray:
        if expert == "text-based":
            return unit(self.api.embed([text], "query")[0], self.dim, np.float64)
        if expert == "fusion":
            return self._adapt(self._languagebind().encode_text(text))
        return unit(self.omni.encode_texts([text], batch_size=1)[0], self.dim)

    def _describe(self, record: dict[str, Any]) -> str:
        if text := raw_text(record):
            return text
        modality = record.get("modality") or "text"
        path = Path(record["raw_path"]) if record.get("raw_path") else None
        if modality not in EEE_PROMPTS or path is None or not path.exists():
            return feature_text(record)
        return self.api.describe(path, modality, EEE_PROMPTS[modality])

    def _languagebind(self) -> LanguageBind:
        if self.languagebind is None:
            self.languagebind = LanguageBind(self.device)
        return self.languagebind

    def _adapt(self, vector: np.ndarray) -> np.ndarray:
        """Project LanguageBind features to `embedding_dim` with a fixed-seed random matrix."""
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            if self.adapter is None or self.adapter.shape[0] != vector.shape[0]:
                weights = np.random.default_rng(13).standard_normal((vector.shape[0], self.dim)).astype(np.float32)
                self.adapter = weights / np.sqrt(max(1, vector.shape[0]))
            vector = (vector[None, :] @ self.adapter)[0]
        return unit(vector, self.dim)


def raw_value(record: dict[str, Any], modality: str) -> str:
    """Raw text for text records (from the record or its file); otherwise the media path."""
    if modality == "text":
        if text := raw_text(record):
            return text
        path = Path(record["raw_path"]) if record.get("raw_path") else None
        if path and path.exists():
            return path.read_text(encoding="utf-8").strip()
    return str(record["raw_path"]) if record.get("raw_path") else str(record.get("raw_text") or "")


def feature_text(record: dict[str, Any]) -> str:
    """Modality-aware text stand-in for raw media: modality, pool, path tokens, file fingerprint, and metadata."""
    path = Path(record["raw_path"]) if isinstance(record.get("raw_path"), str) else None
    parts = [
        f"modality:{record.get('modality')}",
        f"pool:{record.get('pool')}",
        f"path:{path.name if path else record.get('raw_path')}",
    ]
    if path:
        parts += [fingerprint(path), *re.findall(r"[a-zA-Z0-9]+", path.stem)]
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    parts += [f"{key}:{value}" for key, value in sorted(metadata.items()) if value is not None and value != ""]
    return " ".join(parts + ([text] if (text := raw_text(record)) else []))
