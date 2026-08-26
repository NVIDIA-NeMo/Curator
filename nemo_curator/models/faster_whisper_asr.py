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

"""Faster-Whisper implementation of the shared ASR adapter."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nemo_curator.models.asr.base import ASRResult

_TARGET_SAMPLE_RATE = 16_000
_LANGUAGE_ALIASES = {
    "fil": "tl",
    "in": "id",
    "iw": "he",
    "ji": "yi",
    "jv": "jw",
    "nb": "no",
}


def _faster_whisper_stack() -> tuple[type, Any]:
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
    except ImportError as error:
        msg = "Faster-Whisper ASR requires the 'faster-whisper' package"
        raise ImportError(msg) from error
    return WhisperModel, download_model


def _whisper_model_class() -> type:
    return _faster_whisper_stack()[0]


def _download_whisper_model(model_id: str, revision: str | None) -> None:
    _faster_whisper_stack()[1](model_id, revision=revision)


@dataclass
class FasterWhisperASR:
    """Run Faster-Whisper on mono 16 kHz waveforms prepared by ``ASRStage``."""

    model_id: str = "large-v3"
    revision: str | None = None
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    without_timestamps: bool = True
    cpu_compute_type: str = "int8"
    _model: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_id:
            msg = "FasterWhisperASR.model_id must be non-empty"
            raise ValueError(msg)

    def download_weights_on_node(self) -> None:
        """Populate Faster-Whisper's node-local cache without allocating a GPU."""
        _download_whisper_model(self.model_id, self.revision)

    def load_model(self, *, num_gpus: int) -> None:
        if self._model is not None:
            return
        if num_gpus < 0:
            msg = "num_gpus must be non-negative"
            raise ValueError(msg)

        device = "cuda" if num_gpus else "cpu"
        compute_type = self.compute_type if num_gpus else self.cpu_compute_type
        self._model = _whisper_model_class()(
            self.model_id,
            device=device,
            compute_type=compute_type,
            revision=self.revision,
        )

    def unload_model(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        if self._model is None:
            msg = "FasterWhisperASR is not initialized; call load_model() first"
            raise RuntimeError(msg)

        results: list[ASRResult] = []
        for item in items:
            waveform = np.asarray(item.get("waveform"), dtype=np.float32)
            if waveform.size == 0:
                results.append(ASRResult(text="", skipped=True, skip_reason="empty_audio"))
                continue
            if waveform.ndim != 1:
                msg = f"ASRStage must provide a mono 1-D waveform, got shape {waveform.shape}"
                raise ValueError(msg)
            sample_rate = int(item.get("sample_rate") or 0)
            if sample_rate != _TARGET_SAMPLE_RATE:
                msg = f"ASRStage must provide {_TARGET_SAMPLE_RATE} Hz audio; received {sample_rate} Hz"
                raise ValueError(msg)

            raw_language = str(item.get("language_code") or "").strip().lower()
            language = _LANGUAGE_ALIASES.get(raw_language, raw_language) or None
            segments, _ = self._model.transcribe(
                np.ascontiguousarray(waveform),
                language=language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                without_timestamps=self.without_timestamps,
            )
            text = " ".join(segment.text for segment in segments).strip()
            results.append(ASRResult(text=text, extras={"language_code": language}))
        return results
