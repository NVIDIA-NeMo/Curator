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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from nemo_curator.models.sed import get_model_class
from nemo_curator.models.sed.base import SEDResult

_DEFAULT_MODEL_TYPE = "Cnn14_DecisionLevelMax"
_DEFAULT_CHECKPOINT_FILENAME = "Cnn14_DecisionLevelMax_mAP=0.385.pth"
_DEFAULT_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
_DEFAULT_CHECKPOINT_CONFIG: dict[str, int] = {
    "sample_rate": 32000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 527,
}


@dataclass
class PANNsSEDAdapter:
    """Run an AudioSet-pretrained PANNs CNN14 checkpoint.

    ``model_type`` must be one of the checkpoint names exposed by
    ``nemo_curator.models.sed.SUPPORTED_MODEL_TYPES``. The frontend arguments
    must match the checkpoint. With no ``checkpoint_path``, the official
    DecisionLevelMax checkpoint is downloaded from the upstream PANNs Zenodo
    release and cached. The default configuration produces 527-class output
    at 100 frames per second.
    """

    checkpoint_path: str | None = None
    sample_rate: int = _DEFAULT_CHECKPOINT_CONFIG["sample_rate"]
    model_type: str = _DEFAULT_MODEL_TYPE
    window_size: int = _DEFAULT_CHECKPOINT_CONFIG["window_size"]
    hop_size: int = _DEFAULT_CHECKPOINT_CONFIG["hop_size"]
    mel_bins: int = _DEFAULT_CHECKPOINT_CONFIG["mel_bins"]
    fmin: int = _DEFAULT_CHECKPOINT_CONFIG["fmin"]
    fmax: int = _DEFAULT_CHECKPOINT_CONFIG["fmax"]
    classes_num: int = _DEFAULT_CHECKPOINT_CONFIG["classes_num"]
    pad_short_segments: bool = True
    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for field_name in ("window_size", "hop_size", "mel_bins", "classes_num"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                msg = f"PANNsSEDAdapter.{field_name} must be a positive integer, got {value!r}"
                raise ValueError(msg)

    def _validate_default_checkpoint_config(self) -> None:
        """Reject incompatible settings before resolving the registered default.

        For example, ``PANNsSEDAdapter(sample_rate=16_000).download_weights_on_node()``
        reaches this validation and raises because the registered checkpoint uses
        its published 32 kHz frontend. Supplying ``checkpoint_path`` selects
        user-owned weights and bypasses this default-checkpoint validation.
        """
        if self.model_type != _DEFAULT_MODEL_TYPE:
            msg = (
                f"No automatic PANNs checkpoint is registered for model_type={self.model_type!r}; "
                "provide checkpoint_path for this model"
            )
            raise ValueError(msg)

        mismatches = {
            name: (getattr(self, name), expected)
            for name, expected in _DEFAULT_CHECKPOINT_CONFIG.items()
            if getattr(self, name) != expected
        }
        if mismatches:
            details = ", ".join(
                f"{name}={actual!r} (expected {expected!r})" for name, (actual, expected) in mismatches.items()
            )
            msg = (
                f"The automatic {_DEFAULT_CHECKPOINT_FILENAME} checkpoint requires its published frontend: "
                f"{details}. Provide a compatible checkpoint_path for custom settings."
            )
            raise ValueError(msg)

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load local weights or resolve the registered default through PyTorch Hub."""
        if self.checkpoint_path is not None:
            return torch.load(Path(self.checkpoint_path).expanduser(), map_location="cpu", weights_only=True)

        self._validate_default_checkpoint_config()
        return torch.hub.load_state_dict_from_url(
            _DEFAULT_CHECKPOINT_URL,
            map_location="cpu",
            progress=False,
            file_name=_DEFAULT_CHECKPOINT_FILENAME,
            weights_only=True,
        )

    def download_weights_on_node(self) -> None:
        """Populate PyTorch Hub's cache for the registered default checkpoint."""
        if self.checkpoint_path is None:
            self._load_checkpoint()

    def load_model(self, *, num_gpus: int) -> None:
        """Load checkpoint weights on CPU first, then place the model."""
        if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus not in {0, 1}:
            msg = f"PANNsSEDAdapter supports zero or one physical GPU, got {num_gpus!r}"
            raise ValueError(msg)

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
        checkpoint = self._load_checkpoint()
        model.load_state_dict(checkpoint["model"])
        model.to(self._device)
        model.eval()
        self._model = model
        checkpoint_source = self.checkpoint_path or _DEFAULT_CHECKPOINT_URL
        logger.info("Loaded {} from {} on {}", self.model_type, checkpoint_source, self._device)

    def unload_model(self) -> None:
        """Release the model and any reclaimable CUDA cache state."""
        self._model = None
        self._device = None
        gc.collect()
        try:
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

    def infer_batch(self, items: list[dict[str, Any]]) -> list[SEDResult]:
        """Run one checkpoint-compatible CNN14 call for the prepared batch."""
        waveforms = [item["waveform"] for item in items]
        padded = self._pad_to_rectangle(waveforms)
        tensor = torch.from_numpy(padded).to(self._device)
        with torch.no_grad():
            output = self._model(tensor)

        framewise = output["framewise_output"].cpu().numpy()
        fps = float(self.sample_rate) / self.hop_size
        results: list[SEDResult] = []
        for waveform, row in zip(  # noqa: B905 - CNN14 preserves input batch cardinality
            waveforms, framewise
        ):
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
