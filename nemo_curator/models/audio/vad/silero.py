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

"""Silero TorchScript and ONNX implementations of the VAD adapter contract."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal

from .base import VADResult, VADSegment

if TYPE_CHECKING:
    import torch


SILERO_SUPPORTED_SAMPLE_RATES = frozenset({8000, 16000, 32000, 48000, 64000, 96000})
SILERO_TARGET_SAMPLE_RATE = 16000
_CHANNEL_FIRST_DIMENSIONS = 2


def _validate_num_gpus(num_gpus: object, *, owner: str, allow_gpu: bool) -> int:
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, Integral):
        msg = f"{owner}.load_model num_gpus must be an integer, got {num_gpus!r}"
        raise TypeError(msg)
    count = int(num_gpus)
    maximum = 1 if allow_gpu else 0
    if not 0 <= count <= maximum:
        expectation = "0 or 1" if allow_gpu else "0"
        msg = f"{owner}.load_model num_gpus must be {expectation}, got {count}"
        raise ValueError(msg)
    return count


def _validate_detection_options(  # noqa: PLR0913
    *,
    threshold: float,
    min_duration_sec: float,
    max_duration_sec: float,
    min_interval_ms: int,
    speech_pad_ms: int,
    owner: str,
) -> None:
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        msg = f"{owner}.threshold must be finite and in [0, 1], got {threshold!r}"
        raise ValueError(msg)
    minimum = float(min_duration_sec)
    maximum = float(max_duration_sec)
    if not math.isfinite(minimum) or minimum < 0:
        msg = f"{owner}.min_duration_sec must be finite and non-negative, got {min_duration_sec!r}"
        raise ValueError(msg)
    if not maximum > minimum:
        msg = f"{owner}.max_duration_sec must be greater than min_duration_sec, got {max_duration_sec!r}"
        raise ValueError(msg)
    for name, value in (("min_interval_ms", min_interval_ms), ("speech_pad_ms", speech_pad_ms)):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            msg = f"{owner}.{name} must be a non-negative integer, got {value!r}"
            raise ValueError(msg)


def _prepare_waveform(
    item: dict[str, Any],
    *,
    device: object,
) -> tuple[torch.Tensor, int]:
    """Prepare one stage-normalized mono item with Silero's rate rules."""
    import torch

    if "waveform" not in item:
        msg = "Silero VAD item is missing 'waveform'"
        raise KeyError(msg)
    if "sample_rate" not in item:
        msg = "Silero VAD item is missing 'sample_rate'"
        raise KeyError(msg)

    sample_rate_value = item["sample_rate"]
    if isinstance(sample_rate_value, bool) or not isinstance(sample_rate_value, Integral):
        msg = f"Silero VAD sample_rate must be an integer, got {sample_rate_value!r}"
        raise TypeError(msg)
    sample_rate = int(sample_rate_value)
    if sample_rate <= 0:
        msg = f"Silero VAD sample_rate must be positive, got {sample_rate}"
        raise ValueError(msg)

    waveform = torch.as_tensor(item["waveform"], dtype=torch.float32).detach()
    if waveform.ndim == _CHANNEL_FIRST_DIMENSIONS and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if waveform.ndim != 1:
        msg = f"Silero VAD expects a mono 1-D waveform, got shape {tuple(waveform.shape)}"
        raise ValueError(msg)
    waveform = waveform.contiguous()

    needs_resample = sample_rate not in SILERO_SUPPORTED_SAMPLE_RATES
    if needs_resample:
        import torchaudio

        waveform_cpu = waveform.cpu().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=SILERO_TARGET_SAMPLE_RATE,
        )
        waveform = resampler(waveform_cpu).squeeze(0).contiguous()
        sample_rate = SILERO_TARGET_SAMPLE_RATE

    return waveform.to(device=device, dtype=torch.float32, non_blocking=True), sample_rate


def _timestamps_to_result(timestamps: list[dict[str, Any]], sample_rate: int) -> VADResult:
    return VADResult(
        segments=[
            VADSegment(
                start=float(timestamp["start"]) / sample_rate,
                end=float(timestamp["end"]) / sample_rate,
            )
            for timestamp in timestamps
        ]
    )


@dataclass
class SileroVADAdapter:
    """Run the official bundled Silero model through TorchScript or ONNX.

    The official helper accepts 8 kHz, 16 kHz, and integer multiples of
    16 kHz.  Those rates are passed through exactly so the helper performs its
    registered decimation behavior.  Other rates are resampled to 16 kHz.
    """

    backend: Literal["torch", "onnx"] = "torch"
    threshold: float = 0.5
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0
    min_interval_ms: int = 500
    speech_pad_ms: int = 300

    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.backend not in {"torch", "onnx"}:
            msg = f"Unsupported Silero backend {self.backend!r}; expected 'torch' or 'onnx'"
            raise ValueError(msg)
        _validate_detection_options(
            threshold=self.threshold,
            min_duration_sec=self.min_duration_sec,
            max_duration_sec=self.max_duration_sec,
            min_interval_ms=self.min_interval_ms,
            speech_pad_ms=self.speech_pad_ms,
            owner=type(self).__name__,
        )

    def download_weights_on_node(self) -> None:
        """No-op: the official wheel contains both model files."""

    def load_model(self, *, num_gpus: int) -> None:
        """Load the selected official runtime into worker-local state."""
        import torch

        gpu_count = _validate_num_gpus(
            num_gpus,
            owner=type(self).__name__,
            allow_gpu=self.backend == "torch",
        )
        if self._model is not None:
            return
        if gpu_count and not torch.cuda.is_available():
            msg = "Silero TorchScript GPU inference requested, but CUDA is unavailable"
            raise RuntimeError(msg)

        from silero_vad import load_silero_vad

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sampling rate is a multiple of 16000")
            model = load_silero_vad(onnx=True, opset_version=16) if self.backend == "onnx" else load_silero_vad()

        self._device = torch.device("cuda" if gpu_count else "cpu")
        if self.backend == "torch" and gpu_count:
            model = model.to(self._device)
        self._model = model

    def unload_model(self) -> None:
        """Release the model and any CUDA allocator cache it occupied."""
        used_cuda = getattr(self._device, "type", None) == "cuda"
        self._model = None
        self._device = None
        if used_cuda:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def detect_batch(self, items: list[dict[str, Any]]) -> list[VADResult]:
        """Detect speech in a ragged batch while preserving input order."""
        if not items:
            return []
        if self._model is None or self._device is None:
            msg = "Silero VAD model is not loaded; call load_model() before detect_batch()"
            raise RuntimeError(msg)

        import torch
        from silero_vad import get_speech_timestamps

        results: list[VADResult] = []
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sampling rate is a multiple of 16000")
            for item in items:
                waveform, sample_rate = _prepare_waveform(
                    item,
                    device=self._device,
                )
                try:
                    timestamps = get_speech_timestamps(
                        waveform,
                        self._model,
                        sampling_rate=sample_rate,
                        threshold=self.threshold,
                        min_speech_duration_ms=self.min_duration_sec * 1000,
                        max_speech_duration_s=self.max_duration_sec,
                        min_silence_duration_ms=self.min_interval_ms,
                        speech_pad_ms=self.speech_pad_ms,
                    )
                    results.append(_timestamps_to_result(timestamps, sample_rate))
                except Exception as exc:  # noqa: BLE001
                    results.append(VADResult(segments=[], error=f"{type(exc).__name__}: {exc}"))
        return results
