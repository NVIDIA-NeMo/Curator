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

"""WhisperX (pyannote segmentation) implementation of the VAD adapter contract."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .base import VADResult, VADSegment, _validate_detection_options

if TYPE_CHECKING:
    import numpy as np


def load_whisperx_vad_model(
    device: str,
    *,
    vad_onset: float,
    vad_offset: float,
    hf_token: str | None = None,
) -> Any:  # noqa: ANN401
    """Load WhisperX's pyannote VAD model on ``device`` (downloads weights on first use)."""
    import torch
    from whisperx.vads.pyannote import load_vad_model

    if device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA device requested but not available. Set device='cpu' to run without GPU."
        raise RuntimeError(msg)
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"
    try:
        return load_vad_model(torch.device(device), token=hf_token, vad_onset=vad_onset, vad_offset=vad_offset)
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous


def whisperx_vad_segments(
    model: Any,  # noqa: ANN401
    audio: np.ndarray,
    sample_rate: int,
    *,
    merge_max_length: float,
    vad_onset: float,
) -> list[dict[str, Any]]:
    """Run WhisperX VAD on channel-first ``audio`` and merge chunks up to ``merge_max_length`` seconds."""
    import torch
    from whisperx.vads.pyannote import Pyannote

    raw = model({"waveform": torch.from_numpy(audio), "sample_rate": sample_rate})
    return Pyannote.merge_chunks(raw, merge_max_length, onset=vad_onset)


@dataclass
class WhisperXVADAdapter:
    """Detect speech with WhisperX's pyannote VAD model.

    ``threshold`` is the WhisperX onset, ``max_duration_sec`` the chunk-merge
    length, and ``min_duration_sec`` drops merged chunks shorter than it.
    ``min_interval_ms`` and ``speech_pad_ms`` belong to the shared VAD stage
    contract; WhisperX has no equivalent and ignores them.
    """

    threshold: float = 0.5
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0
    min_interval_ms: int = 500
    speech_pad_ms: int = 300
    vad_offset: float = 0.363
    hf_token: str | None = field(default=None, repr=False)

    _model: Any = field(default=None, init=False, repr=False)
    _device: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_detection_options(
            threshold=self.threshold,
            min_duration_sec=self.min_duration_sec,
            max_duration_sec=self.max_duration_sec,
            min_interval_ms=self.min_interval_ms,
            speech_pad_ms=self.speech_pad_ms,
            owner=type(self).__name__,
        )

    def _load(self, device: str) -> Any:  # noqa: ANN401
        return load_whisperx_vad_model(
            device, vad_onset=self.threshold, vad_offset=self.vad_offset, hf_token=self.hf_token
        )

    def download_weights_on_node(self) -> None:
        """Cache the VAD weights by loading the model once on CPU."""
        self._load("cpu")

    def load_model(self, *, num_gpus: int) -> None:
        """Load the VAD model on CUDA when ``num_gpus > 0``, else on CPU."""
        if self._model is not None:
            return
        self._device = "cuda" if num_gpus > 0 else "cpu"
        self._model = self._load(self._device)

    def unload_model(self) -> None:
        """Release the model and any CUDA cache it occupied."""
        used_cuda = self._device == "cuda"
        self._model = None
        self._device = None
        if used_cuda:
            import torch

            torch.cuda.empty_cache()

    def detect_batch(self, items: list[dict[str, Any]]) -> list[VADResult]:
        """Detect speech per mono item, preserving input order."""
        if self._model is None:
            msg = "WhisperX VAD model is not loaded; call load_model() before detect_batch()"
            raise RuntimeError(msg)
        import numpy as np
        import torch

        results: list[VADResult] = []
        for item in items:
            try:
                audio = np.ascontiguousarray(np.asarray(item["waveform"], dtype=np.float32)[None, :])
                chunks = whisperx_vad_segments(
                    self._model,
                    audio,
                    int(item["sample_rate"]),
                    merge_max_length=self.max_duration_sec,
                    vad_onset=self.threshold,
                )
                results.append(
                    VADResult(
                        segments=[
                            VADSegment(start=float(chunk["start"]), end=float(chunk["end"]))
                            for chunk in chunks
                            if float(chunk["end"]) - float(chunk["start"]) >= self.min_duration_sec
                        ]
                    )
                )
            except (MemoryError, torch.cuda.OutOfMemoryError):
                raise
            except Exception as exc:  # noqa: BLE001
                results.append(VADResult(segments=[], error=f"{type(exc).__name__}: {exc}"))
        return results
