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

"""TitaNet speaker embeddings for Granary segments (shared by speaker clustering and context selection)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from nemo_curator.stages.audio.tts.audio_io import SegmentAudioError, TarAudioConfig, load_segment_audio

DEFAULT_TITANET_MODEL = "nvidia/speakerverification_en_titanet_large"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


@dataclass
class TitaNetEmbedder:
    """Batched TitaNet (``EncDecSpeakerLabelModel``) embeddings of segment audio.

    ``model_path`` (a local ``.nemo``) takes precedence over ``model_name`` (``from_pretrained``). A pre-loaded
    model may be injected through ``model`` (tests, or sharing one instance between stages).
    """

    model_name: str = DEFAULT_TITANET_MODEL
    model_path: str | None = None
    batch_size: int = 32
    device: str = "cuda"
    model: Any | None = None
    sample_rate: int = 16000
    _device: Any = field(default=None, init=False, repr=False)

    def download(self) -> None:
        if self.model_path is None and self.model is None:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel

            EncDecSpeakerLabelModel.from_pretrained(self.model_name, return_model_file=True)

    def load(self) -> None:
        import torch

        self._device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        if self.model is None:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel

            if self.model_path:
                self.model = EncDecSpeakerLabelModel.restore_from(self.model_path, map_location=self._device)
            else:
                self.model = EncDecSpeakerLabelModel.from_pretrained(self.model_name, map_location=self._device)
            self.model = self.model.to(self._device)
        self.model.eval()
        preprocessor = getattr(self.model, "preprocessor", None)
        self.sample_rate = int(getattr(preprocessor, "_sample_rate", self.sample_rate))

    def embed_waveforms(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Embed equal-rate mono waveforms (zero-padded per batch); returns raw embeddings ``[N, D]``."""
        import torch

        outputs: list[np.ndarray] = []
        for start in range(0, len(waveforms), self.batch_size):
            batch = waveforms[start : start + self.batch_size]
            lengths = torch.tensor([len(w) for w in batch], dtype=torch.long)
            audio = torch.zeros(len(batch), int(lengths.max()), dtype=torch.float32)
            for row, wave in enumerate(batch):
                audio[row, : len(wave)] = torch.from_numpy(np.asarray(wave, dtype=np.float32))
            with torch.no_grad():
                _, embeddings = self.model.forward(
                    input_signal=audio.to(self._device), input_signal_length=lengths.to(self._device)
                )
            outputs.append(embeddings.detach().cpu().to(torch.float32).numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)

    def embed_segments(
        self, segments: list[dict[str, Any]], audio_config: TarAudioConfig | None = None
    ) -> tuple[list[np.ndarray | None], list[str | None]]:
        """Embed segments; returns per-segment embeddings (``None`` on failure) and error codes.

        A failing batch marks all its segments ``embedding_failed`` (reference behaviour).
        """
        embeddings: list[np.ndarray | None] = [None] * len(segments)
        errors: list[str | None] = [None] * len(segments)
        pending: list[tuple[int, np.ndarray]] = []

        def flush() -> None:
            if not pending:
                return
            try:
                vectors = self.embed_waveforms([wave for _, wave in pending])
                for (index, _), vector in zip(pending, vectors, strict=True):
                    embeddings[index] = vector
            except Exception as exc:  # noqa: BLE001
                logger.warning("TitaNet embedding batch failed: {}", exc)
                for index, _ in pending:
                    errors[index] = "embedding_failed"
            pending.clear()

        for index, segment in enumerate(segments):
            try:
                wave, _ = load_segment_audio(segment, self.sample_rate, audio_config)
            except SegmentAudioError as exc:
                errors[index] = exc.code
                continue
            pending.append((index, wave))
            if len(pending) >= self.batch_size:
                flush()
        flush()
        return embeddings, errors
