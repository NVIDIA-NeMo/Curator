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

"""NeMo AmberNet implementation of the audio-LID adapter."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from .base import (
    AudioLIDResult,
    _validate_sample_rate,
    _validate_single_device_gpu_count,
    _waveform_from_item,
)

if TYPE_CHECKING:
    import numpy as np

_DEFAULT_MODEL_NAME = "langid_ambernet"
_LOGITS_RANK = 2


def _nemo_asr_module() -> Any:  # noqa: ANN401
    """Import NeMo ASR only when AmberNet is selected."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        msg = "AmberNetLIDAdapter requires an audio extra: uv sync --extra audio_cpu or --extra audio_cuda12"
        raise ImportError(msg) from exc
    return nemo_asr


def _config_value(config: object, key: str) -> object:
    if config is None:
        return None
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, None)
    return getattr(config, key, None)


def _model_labels(model: object, class_count: int) -> list[str]:
    config = getattr(model, "cfg", None)
    train_ds = _config_value(config, "train_ds")
    labels = _config_value(train_ds, "labels") or _config_value(config, "labels")
    if labels is None:
        return [str(index) for index in range(class_count)]
    normalized = [str(label) for label in labels]
    if len(normalized) != class_count:
        msg = f"AmberNet exposes {len(normalized)} labels for {class_count} output classes"
        raise RuntimeError(msg)
    return normalized


@dataclass
class AmberNetLIDAdapter:
    """Identify speech with NeMo's pretrained ``langid_ambernet`` model."""

    model_name: str = _DEFAULT_MODEL_NAME
    sample_rate: int = 16_000
    _model: Any = field(default=None, init=False, repr=False)
    _device: torch.device | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.sample_rate = _validate_sample_rate(self.sample_rate, owner=type(self).__name__)
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            msg = "AmberNetLIDAdapter.model_name must be a non-empty string"
            raise ValueError(msg)

    def download_weights_on_node(self) -> None:
        """Ask NeMo to populate its checkpoint cache without loading a model."""
        _nemo_asr_module().models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=self.model_name,
            return_model_file=True,
        )

    def load_model(self, *, num_gpus: int) -> None:
        """Load one worker-local AmberNet model on CPU or CUDA."""
        if self._model is not None:
            return
        gpu_count = _validate_single_device_gpu_count(num_gpus, owner=type(self).__name__)
        if gpu_count and not torch.cuda.is_available():
            msg = "AmberNetLIDAdapter received num_gpus=1, but CUDA is not available"
            raise RuntimeError(msg)

        self._device = torch.device("cuda" if gpu_count else "cpu")
        model = _nemo_asr_module().models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=self.model_name,
            map_location=self._device,
        )
        model.to(self._device)
        model.eval()
        self._model = model
        logger.info("Loaded AmberNet LID model {} on {}", self.model_name, self._device)

    def unload_model(self) -> None:
        """Release worker-local model state and reclaim CUDA cache."""
        self._model = None
        self._device = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: {}", exc)

    def identify_batch(self, items: list[dict[str, Any]]) -> list[AudioLIDResult]:
        """Classify a ragged waveform batch while preserving input order."""
        if not items:
            return []
        if self._model is None or self._device is None:
            msg = "AmberNetLIDAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)

        results = [AudioLIDResult("", 0.0) for _ in items]
        valid_indices: list[int] = []
        waveforms: list[np.ndarray] = []
        for index, item in enumerate(items):
            waveform = _waveform_from_item(item, owner=type(self).__name__)
            if waveform.size:
                valid_indices.append(index)
                waveforms.append(waveform)
        if not waveforms:
            return results

        lengths = torch.tensor([waveform.size for waveform in waveforms], dtype=torch.long, device=self._device)
        padded = torch.zeros((len(waveforms), int(lengths.max().item())), dtype=torch.float32)
        for row, waveform in enumerate(waveforms):
            padded[row, : waveform.size] = torch.from_numpy(waveform)
        padded = padded.to(self._device)

        with torch.inference_mode():
            output = self._model.forward(input_signal=padded, input_signal_length=lengths)
            logits = output[0] if isinstance(output, tuple) else output
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predicted_indices = probabilities.max(dim=-1)

        if logits.ndim != _LOGITS_RANK or logits.shape[0] != len(valid_indices):
            msg = f"AmberNet returned logits shaped {tuple(logits.shape)} for {len(valid_indices)} inputs"
            raise RuntimeError(msg)
        labels = _model_labels(self._model, int(logits.shape[-1]))
        for item_index, prediction, confidence in zip(
            valid_indices,
            predicted_indices.detach().cpu().tolist(),
            confidences.detach().cpu().tolist(),
            strict=True,
        ):
            results[item_index] = AudioLIDResult(labels[int(prediction)], float(confidence))
        return results
