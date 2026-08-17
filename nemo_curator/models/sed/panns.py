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

"""PANNs CNN14 implementation of the shared sound-event adapter.

The Curator stage supplies mono float32 waveforms at ``sample_rate``. This
adapter selects a checkpoint-compatible CNN14 variant, pads one ragged batch,
runs one model call, and packages each row as ``SEDResult``.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from nemo_curator.models.sed.base import SEDResult


@dataclass
class PANNsSEDAdapter:
    """Run an AudioSet-pretrained PANNs CNN14 checkpoint.

    ``model_type`` must be one of the checkpoint names exposed by
    ``nemo_curator.models.sed.SUPPORTED_MODEL_TYPES``. The frontend arguments
    must match the checkpoint. The default configuration produces 527-class
    output at 50 frames per second.
    """

    checkpoint_path: str
    sample_rate: int = 16000
    model_type: str = "Cnn14_DecisionLevelMax"
    window_size: int = 1024
    hop_size: int = 320
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 14000
    classes_num: int = 527
    pad_short_segments: bool = True
    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.checkpoint_path:
            msg = "checkpoint_path is required for PANNsSEDAdapter"
            raise ValueError(msg)
        for field_name in ("sample_rate", "window_size", "hop_size", "mel_bins", "classes_num"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                msg = f"PANNsSEDAdapter.{field_name} must be a positive integer, got {value!r}"
                raise ValueError(msg)

    def load_model(self, *, num_gpus: int) -> None:
        """Load checkpoint weights on CPU first, then place the model."""
        if self._model is not None:
            return
        if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus not in {0, 1}:
            msg = f"PANNsSEDAdapter supports zero or one physical GPU, got {num_gpus!r}"
            raise ValueError(msg)

        import torch

        from nemo_curator.models.sed import get_model_class

        model_cls = get_model_class(self.model_type)
        use_cuda = num_gpus == 1 and torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")
        model = model_cls(
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            hop_size=self.hop_size,
            mel_bins=self.mel_bins,
            fmin=self.fmin,
            fmax=self.fmax,
            classes_num=self.classes_num,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.to(self._device)
        model.eval()
        self._model = model
        logger.info("Loaded {} from {} on {}", self.model_type, self.checkpoint_path, self._device)

    def unload_model(self) -> None:
        """Release the model and any reclaimable CUDA cache state."""
        self._model = None
        self._device = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: {}", exc)

    def _pad_to_rectangle(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Zero-pad a ragged batch to the CNN14 minimum and longest row."""
        min_input = max(self.window_size, self.hop_size * 32)
        if self.pad_short_segments:
            waveforms = [
                np.pad(waveform, (0, min_input - waveform.size)) if waveform.size < min_input else waveform
                for waveform in waveforms
            ]

        max_len = max(waveform.size for waveform in waveforms)
        padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
        for index, waveform in enumerate(waveforms):
            padded[index, : waveform.size] = waveform
        return padded

    @staticmethod
    def _waveform(item: dict[str, Any]) -> np.ndarray:
        waveform = np.ascontiguousarray(item.get("waveform"), dtype=np.float32)
        if waveform.ndim != 1:
            msg = f"SEDInferenceStage must provide a mono 1-D waveform, got shape {waveform.shape}"
            raise ValueError(msg)
        if waveform.size == 0:
            msg = "SEDInferenceStage must provide a non-empty waveform"
            raise ValueError(msg)
        return waveform

    def infer_batch(self, items: list[dict[str, Any]]) -> list[SEDResult]:
        """Run one checkpoint-compatible CNN14 call for the prepared batch."""
        if not items:
            return []
        if self._model is None or self._device is None:
            msg = "PANNsSEDAdapter model is not loaded"
            raise RuntimeError(msg)

        import torch

        waveforms = [self._waveform(item) for item in items]
        padded = self._pad_to_rectangle(waveforms)
        tensor = torch.from_numpy(padded).to(self._device)
        with torch.no_grad():
            output = self._model(tensor)

        framewise = output["framewise_output"].cpu().numpy()
        if framewise.shape[0] != len(waveforms):
            msg = f"PANNs model returned {framewise.shape[0]} rows for {len(waveforms)} waveforms"
            raise RuntimeError(msg)

        fps = float(self.sample_rate) / self.hop_size
        results: list[SEDResult] = []
        for waveform, row in zip(waveforms, framewise, strict=True):
            valid_frames = min(int(np.ceil(waveform.size / self.hop_size)), row.shape[0])
            results.append(
                SEDResult(
                    framewise_output=row,
                    fps=fps,
                    valid_frames=valid_frames,
                    original_num_samples=waveform.size,
                )
            )
        logger.info(
            "PANNs SED batch: processed {} waveforms (max_samples={}, fps={:.1f})",
            len(waveforms),
            padded.shape[1],
            fps,
        )
        return results
