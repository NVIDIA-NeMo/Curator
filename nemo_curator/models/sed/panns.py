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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from filelock import FileLock
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


def _default_cache_dir() -> Path:
    """Return the platform cache location used for downloaded PANNs weights."""
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
    return cache_root / "nemo_curator" / "panns"


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
    cache_dir: str | None = None
    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for field_name in ("window_size", "hop_size", "mel_bins", "classes_num"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                msg = f"PANNsSEDAdapter.{field_name} must be a positive integer, got {value!r}"
                raise ValueError(msg)

    def _validate_default_checkpoint_config(self) -> None:
        """Reject automatic weights when the configured frontend is incompatible."""
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

    def _cached_checkpoint_path(self) -> Path:
        cache_dir = Path(self.cache_dir).expanduser() if self.cache_dir else _default_cache_dir()
        return cache_dir / _DEFAULT_CHECKPOINT_FILENAME

    def _download_default_checkpoint(self, checkpoint_path: Path) -> Path:
        """Download the official checkpoint into the managed cache."""
        logger.info("Downloading PANNs SED checkpoint from {} to {}", _DEFAULT_CHECKPOINT_URL, checkpoint_path)
        try:
            torch.hub.download_url_to_file(
                _DEFAULT_CHECKPOINT_URL,
                str(checkpoint_path),
                progress=False,
            )
        except Exception as exc:
            msg = (
                f"Failed to download PANNs checkpoint for {self.model_type} "
                f"from {_DEFAULT_CHECKPOINT_URL} to {checkpoint_path}"
            )
            raise RuntimeError(msg) from exc
        return checkpoint_path

    def _resolve_checkpoint_path(self) -> Path:
        """Resolve a strict local override or the managed cached default."""
        if self.checkpoint_path is not None:
            checkpoint_path = Path(self.checkpoint_path).expanduser()
            if not checkpoint_path.is_file():
                msg = f"PANNs checkpoint_path does not point to a file: {checkpoint_path}"
                raise FileNotFoundError(msg)
            return checkpoint_path.resolve()

        self._validate_default_checkpoint_config()
        checkpoint_path = self._cached_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(f"{checkpoint_path}.lock")
        with FileLock(lock_path):
            if checkpoint_path.is_file():
                logger.info("Using cached PANNs SED checkpoint at {}", checkpoint_path)
                return checkpoint_path
            elif checkpoint_path.exists():
                msg = f"PANNs checkpoint cache target exists but is not a file: {checkpoint_path}"
                raise RuntimeError(msg)

            return self._download_default_checkpoint(checkpoint_path)

    def download_weights_on_node(self) -> None:
        """Warm or validate the checkpoint cache without constructing the model."""
        self._resolve_checkpoint_path()

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
        checkpoint_path = self._resolve_checkpoint_path()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.to(self._device)
        model.eval()
        self._model = model
        logger.info("Loaded {} from {} on {}", self.model_type, checkpoint_path, self._device)

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
