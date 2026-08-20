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
from numbers import Integral
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
    must match the checkpoint. ``checkpoint_path`` is the single checkpoint
    location: an existing file is loaded directly, while a missing file path
    is populated from the upstream PANNs Zenodo release. With no path, that
    same existing-or-missing behavior applies to the standard PyTorch Hub
    checkpoint file. The default configuration produces 527-class output at
    100 frames per second.
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
        reaches this validation when its resolved checkpoint file is missing and
        raises because the registered checkpoint uses its published 32 kHz
        frontend. Any existing resolved file, including one in the standard
        PyTorch Hub cache, bypasses this download-time validation.
        """
        if self.model_type != _DEFAULT_MODEL_TYPE:
            msg = (
                f"No automatic PANNs checkpoint is registered for model_type={self.model_type!r}; "
                "provide checkpoint_path to an existing checkpoint for this model"
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
                f"{details}. Provide checkpoint_path to an existing compatible checkpoint for custom settings."
            )
            raise ValueError(msg)

    def _resolve_checkpoint_path(self) -> Path:
        """Return the configured file or the standard PyTorch Hub checkpoint file."""
        checkpoint_path = (
            Path(torch.hub.get_dir()) / "checkpoints" / _DEFAULT_CHECKPOINT_FILENAME
            if self.checkpoint_path is None
            else Path(self.checkpoint_path).expanduser()
        )
        if checkpoint_path.is_dir():
            msg = f"checkpoint_path must include a checkpoint filename, got directory {checkpoint_path}"
            raise IsADirectoryError(msg)
        return checkpoint_path

    def _download_default_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Download or reuse the registered default at the requested file location."""
        self._validate_default_checkpoint_config()
        return torch.hub.load_state_dict_from_url(
            _DEFAULT_CHECKPOINT_URL,
            model_dir=str(checkpoint_path.parent),
            map_location="cpu",
            progress=False,
            file_name=checkpoint_path.name,
            weights_only=True,
        )

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load an existing checkpoint or download the registered default."""
        checkpoint_path = self._resolve_checkpoint_path()
        if checkpoint_path.is_file():
            return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return self._download_default_checkpoint(checkpoint_path)

    def download_weights_on_node(self) -> None:
        """Ensure the checkpoint exists without constructing or loading the model."""
        checkpoint_path = self._resolve_checkpoint_path()
        if not checkpoint_path.is_file():
            self._download_default_checkpoint(checkpoint_path)

    def load_model(self, *, num_gpus: int) -> None:
        """Load one single-device model on CPU or CUDA.

        Like the direct audio GPU stages, zero selects CPU and any positive
        worker GPU allocation selects CUDA. PANNs remains single-device even
        when a worker reserves more than one GPU.
        """
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, Integral) or num_gpus < 0:
            msg = f"PANNsSEDAdapter requires a non-negative integer num_gpus, got {num_gpus!r}"
            raise ValueError(msg)
        if num_gpus > 0 and not torch.cuda.is_available():
            msg = f"PANNsSEDAdapter received num_gpus={num_gpus}, but CUDA is not available"
            raise RuntimeError(msg)

        model_cls = get_model_class(self.model_type)
        self._device = torch.device("cuda" if num_gpus > 0 else "cpu")
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
        checkpoint_source = self._resolve_checkpoint_path()
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
