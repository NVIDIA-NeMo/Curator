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

"""Model-only adapter for Faster-Whisper ASR."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch
import torchaudio.functional as torchaudio_functional
from loguru import logger

from nemo_curator.models.base import ModelInterface

_TARGET_SAMPLE_RATE = 16000


class FasterWhisperASR(ModelInterface):
    """Own Faster-Whisper lifecycle and waveform-to-text inference."""

    def __init__(  # noqa: PLR0913
        self,
        model_size_or_path: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        vad_filter: bool = True,
        without_timestamps: bool = True,
    ) -> None:
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.without_timestamps = without_timestamps
        self._model: Any | None = None
        self._resolved_device = device

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_size_or_path]

    def setup(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            msg = "Faster-Whisper ASR requires the 'faster-whisper' package"
            raise ImportError(msg) from error

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif resolved_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; Faster-Whisper is falling back to CPU")
            resolved_device = "cpu"

        compute_type = self.compute_type
        if resolved_device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        self._resolved_device = resolved_device
        self._model = WhisperModel(
            self.model_size_or_path,
            device=resolved_device,
            compute_type=compute_type,
        )

    def teardown(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _prepare_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        if tensor.ndim > 1:
            tensor = tensor.mean(dim=-1)
        tensor = tensor.unsqueeze(0)
        if int(sample_rate) != _TARGET_SAMPLE_RATE:
            tensor = torchaudio_functional.resample(
                tensor,
                orig_freq=int(sample_rate),
                new_freq=_TARGET_SAMPLE_RATE,
            )
        return tensor.squeeze(0).contiguous().numpy()

    def generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        language_codes: list[str],
    ) -> tuple[list[str], list[str]]:
        if self._model is None:
            msg = "FasterWhisperASR.setup() must be called before generate()"
            raise RuntimeError(msg)
        if not (len(waveforms) == len(sample_rates) == len(language_codes)):
            msg = "waveforms, sample_rates, and language_codes must have equal lengths"
            raise ValueError(msg)

        texts: list[str] = []
        for waveform, sample_rate, language in zip(
            waveforms,
            sample_rates,
            language_codes,
            strict=True,
        ):
            if np.asarray(waveform).size == 0:
                texts.append("")
                continue
            audio = self._prepare_waveform(waveform, sample_rate)
            segments, _ = self._model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                without_timestamps=self.without_timestamps,
            )
            texts.append(" ".join(segment.text for segment in segments).strip())
        return texts, list(language_codes)
