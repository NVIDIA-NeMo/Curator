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

"""Faster-Whisper ASR for in-memory mono waveforms (language forced per sample).

Uses https://github.com/SYSTRAN/faster-whisper
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch
import torchaudio.functional as F_ta

from loguru import logger

from nemo_curator.models.base import ModelInterface

_TARGET_SR = 16000


class FasterWhisperASR(ModelInterface):
    """Batched-style API over sequential faster-whisper ``transcribe`` calls."""

    def __init__(
        self,
        model_size_or_path: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        vad_filter: bool = True,
        without_timestamps: bool = True,
    ):
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.without_timestamps = without_timestamps
        self._model: Any = None
        self._resolved_device: str = device

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_size_or_path]

    def setup(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            msg = "faster-whisper is required for FasterWhisperASR. Install: pip install faster-whisper"
            raise ImportError(msg) from e

        resolved = self.device
        if resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        elif resolved == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for FasterWhisper but unavailable; using CPU.")
            resolved = "cpu"

        compute_type = self.compute_type
        if resolved == "cpu" and compute_type == "float16":
            compute_type = "int8"

        self._resolved_device = resolved
        logger.info(
            f"Loading FasterWhisper model={self.model_size_or_path} "
            f"device={resolved} compute_type={compute_type}"
        )
        self._model = WhisperModel(
            self.model_size_or_path,
            device=resolved,
            compute_type=compute_type,
        )
        logger.info("FasterWhisper model loaded")

    def teardown(self) -> None:
        del self._model
        self._model = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    def generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        whisper_language_codes: list[str],
    ) -> tuple[list[str], list[str]]:
        """Transcribe with a fixed Whisper ``language`` code per sample (ISO-style, e.g. ``tl``, ``fa``)."""
        if self._model is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        texts: list[str] = []
        langs_out: list[str] = []

        transcribe_kw: dict[str, Any] = {
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "without_timestamps": self.without_timestamps,
        }

        for w, sr, wlang in zip(waveforms, sample_rates, whisper_language_codes, strict=True):
            if w.size == 0:
                texts.append("")
                langs_out.append(wlang)
                continue

            wav = torch.from_numpy(np.asarray(w, dtype=np.float32))
            if wav.ndim > 1:
                wav = wav.mean(dim=-1)
            wav = wav.unsqueeze(0)
            if int(sr) != _TARGET_SR:
                wav = F_ta.resample(wav, orig_freq=int(sr), new_freq=_TARGET_SR)
            audio_fp32 = wav.squeeze(0).contiguous().numpy()

            segments, _info = self._model.transcribe(audio_fp32, language=wlang, **transcribe_kw)
            parts = [seg.text for seg in segments]
            texts.append(" ".join(parts).strip())
            langs_out.append(wlang)

        return texts, langs_out
