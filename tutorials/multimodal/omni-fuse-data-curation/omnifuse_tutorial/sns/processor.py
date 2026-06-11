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

"""Lightweight SNS processor with explicit model-backend extension points."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from omnifuse_tutorial.config.models import SNSConfig
from omnifuse_tutorial.sns.backends import LocalSNSBackend, SNSBackend

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class SNSProcessor:
    config: SNSConfig
    embedding_dim: int = 64
    backend: SNSBackend = field(default_factory=LocalSNSBackend)

    def process_record(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.config.enabled:
            output = dict(record)
            output["sns_raw_text"] = record.get("raw_text")
            output["sns_annotation"] = record.get("annotation")
            manifest = self._manifest(record, output, enabled=False, accepted=False, reason="disabled")
            return output, manifest

        output = dict(record)
        output["sns_raw_text"] = _text_or_none(record.get("raw_text"))
        output["sns_annotation"] = _text_or_none(record.get("annotation")) or ""
        decisions: list[dict[str, Any]] = []

        if self.config.direction in {"forward", "bidirectional"}:
            output, decision = self._forward(output)
            decisions.append(decision)
        if self.config.direction in {"backward", "bidirectional"}:
            output, decision = self._backward(output)
            decisions.append(decision)

        accepted = any(item.get("accepted") for item in decisions)
        manifest = self._manifest(record, output, enabled=True, accepted=accepted, reason="processed")
        manifest["decisions"] = decisions
        return output, manifest

    def _forward(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        modality = record["modality"]
        annotation = _text_or_none(record.get("sns_annotation")) or _text_or_none(record.get("annotation")) or ""
        if modality != "text":
            return self.backend.forward_media(record, annotation)

        raw_text = _text_or_none(record.get("raw_text")) or ""
        sentences = _sentences(raw_text)
        tau = self.config.tau_forward_text
        components = self._annotation_components(annotation)
        kept = [
            sentence
            for sentence in sentences
            if components
            and max(self.backend.text_text(sentence, component, self.embedding_dim) for component in components) >= tau
        ]
        if not kept:
            return record, {"direction": "forward", "accepted": False, "reason": "no_sentence_above_threshold"}

        candidate = " ".join(kept)
        if not self._passes_mi_gate("text", candidate, annotation, raw_text, annotation):
            return record, {"direction": "forward", "accepted": False, "reason": "mi_gate_failed"}

        updated = dict(record)
        updated["sns_raw_text"] = candidate
        return updated, {
            "direction": "forward",
            "accepted": True,
            "reason": "text_sentences_kept",
            "kept_sentences": len(kept),
            "total_sentences": len(sentences),
            "annotation_components": components,
        }

    def _backward(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        raw_description = (
            _text_or_none(record.get("sns_raw_text"))
            or _text_or_none(record.get("raw_text"))
            or _text_or_none(metadata.get("raw_description"))
        )
        if not raw_description:
            raw_description = self.backend.describe_record(record)
        annotation = _text_or_none(record.get("sns_annotation")) or _text_or_none(record.get("annotation")) or ""
        sentences = _sentences(annotation)
        kept = [
            sentence
            for sentence in sentences
            if self.backend.text_text(sentence, raw_description, self.embedding_dim) >= self.config.tau_backward
        ]
        if not kept:
            return record, {
                "direction": "backward",
                "accepted": False,
                "reason": "no_annotation_sentence_above_threshold",
            }

        candidate = " ".join(kept)
        modality = str(record.get("modality") or "text")
        raw_value = self._raw_value_for_similarity(record, modality)
        if not self._passes_mi_gate(modality, raw_value, candidate, raw_value, annotation):
            return record, {"direction": "backward", "accepted": False, "reason": "mi_gate_failed"}

        updated = dict(record)
        updated["sns_annotation"] = candidate
        return updated, {
            "direction": "backward",
            "accepted": True,
            "reason": "annotation_sentences_kept",
            "kept_sentences": len(kept),
            "total_sentences": len(sentences),
        }

    def _passes_mi_gate(
        self,
        modality: str,
        candidate_raw: Any,
        candidate_annotation: str,
        original_raw: Any,
        original_annotation: str,
    ) -> bool:
        original_sim = max(
            self._raw_annotation_similarity(original_raw, modality, original_annotation),
            self.config.mi_eps,
        )
        candidate_sim = self._raw_annotation_similarity(candidate_raw, modality, candidate_annotation)
        return candidate_sim >= self.config.mi_ratio * original_sim

    def _raw_annotation_similarity(self, raw_value: Any, modality: str, annotation: str) -> float:
        method = getattr(self.backend, "raw_annotation_similarity", None)
        if callable(method):
            return float(method(raw_value, modality, annotation))
        if modality == "image":
            return self.backend.image_text(raw_value, annotation)
        if modality == "audio":
            return self.backend.audio_text(raw_value, annotation)
        if modality == "video":
            return self.backend.video_text(raw_value, annotation)
        return self.backend.text_text(raw_value, annotation, self.embedding_dim)

    @staticmethod
    def _raw_value_for_similarity(record: dict[str, Any], modality: str) -> Any:
        if modality == "text":
            return (
                _text_or_none(record.get("sns_raw_text"))
                or _text_or_none(record.get("raw_text"))
                or _text_or_none(record.get("raw_path"))
                or ""
            )
        return record.get("raw_path") or ""

    def _annotation_components(self, annotation: str) -> list[str]:
        if not self.config.use_ann_components:
            return [annotation] if annotation.strip() else []
        try:
            from omnifuse_tutorial.sns.full_forward import annotation_components

            return annotation_components(annotation, self.config)
        except RuntimeError:
            if self.config.require_forward_models:
                raise
            return [annotation] if annotation.strip() else []

    def _manifest(
        self,
        original: dict[str, Any],
        output: dict[str, Any],
        enabled: bool,
        accepted: bool,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "pair_id": original["pair_id"],
            "pool": original["pool"],
            "modality": original["modality"],
            "enabled": enabled,
            "direction": self.config.direction,
            "accepted": accepted,
            "reason": reason,
            "original_raw_path": original.get("raw_path"),
            "original_annotation": original.get("annotation"),
            "sns_raw_text": output.get("sns_raw_text"),
            "sns_annotation": output.get("sns_annotation"),
            "thresholds": {
                "mi_ratio": self.config.mi_ratio,
                "mi_eps": self.config.mi_eps,
                "tau_forward_text": self.config.tau_forward_text,
                "tau_forward_image": self.config.tau_forward_image,
                "tau_forward_video": self.config.tau_forward_video,
                "tau_forward_audio": self.config.tau_forward_audio,
                "tau_backward": self.config.tau_backward,
            },
        }


def _sentences(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in SENTENCE_RE.split(_text_or_none(text) or "")]
    return [chunk for chunk in chunks if chunk]


def _text_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None
