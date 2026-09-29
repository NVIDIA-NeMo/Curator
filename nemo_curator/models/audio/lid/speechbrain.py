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

"""SpeechBrain VoxLingua107 implementation of the audio-LID adapter."""

from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from .base import (
    AudioLIDResult,
    _validate_sample_rate,
    _validate_single_device_gpu_count,
    _waveform_from_item,
)

_DEFAULT_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"


def _encoder_classifier_class() -> type:
    """Import SpeechBrain only when an adapter actually needs it."""
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError as exc:
        msg = "SpeechBrainLIDAdapter requires an audio extra: uv sync --extra audio_cpu or --extra audio_cuda12"
        raise ImportError(msg) from exc
    return EncoderClassifier


def _snapshot_download(**kwargs: object) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        msg = "SpeechBrainLIDAdapter requires huggingface-hub: uv sync --extra audio_cpu or --extra audio_cuda12"
        raise ImportError(msg) from exc
    return str(snapshot_download(**kwargs))


def _normalize_language(raw: object) -> str:
    """Keep the ISO code from SpeechBrain labels such as ``ta: Tamil``."""
    while isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == 1:
        raw = raw[0]
    text = str(raw or "").strip()
    return text.split(":", 1)[0].strip().lower() if text else ""


@dataclass
class SpeechBrainLIDAdapter:
    """Identify speech with SpeechBrain's VoxLingua107 ECAPA-TDNN model."""

    source: str = _DEFAULT_SOURCE
    sample_rate: int = 16_000
    revision: str | None = None
    cache_dir: str | None = None
    savedir: str | None = None
    _classifier: Any = field(default=None, init=False, repr=False)
    _device: torch.device | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.sample_rate = _validate_sample_rate(self.sample_rate, owner=type(self).__name__)
        if not isinstance(self.source, str) or not self.source.strip():
            msg = "SpeechBrainLIDAdapter.source must be a non-empty string"
            raise ValueError(msg)

    def download_weights_on_node(self) -> None:
        """Populate the Hugging Face cache without constructing a classifier."""
        source_path = Path(self.source).expanduser()
        if source_path.exists():
            return
        kwargs: dict[str, Any] = {"repo_id": self.source}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        _snapshot_download(**kwargs)

    def _load_source(self) -> str:
        """Resolve explicit cache/revision settings to an immutable snapshot."""
        source_path = Path(self.source).expanduser()
        if source_path.exists() or (self.revision is None and self.cache_dir is None):
            return str(source_path) if source_path.exists() else self.source
        kwargs: dict[str, Any] = {"repo_id": self.source, "local_files_only": True}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        return _snapshot_download(**kwargs)

    def load_model(self, *, num_gpus: int) -> None:
        """Load one worker-local classifier on CPU or its allocated GPU."""
        if self._classifier is not None:
            return
        gpu_count = _validate_single_device_gpu_count(num_gpus, owner=type(self).__name__)
        if gpu_count and not torch.cuda.is_available():
            msg = "SpeechBrainLIDAdapter received num_gpus=1, but CUDA is not available"
            raise RuntimeError(msg)

        self._device = torch.device("cuda" if gpu_count else "cpu")
        cache_root = Path(self.savedir or Path(tempfile.gettempdir()) / "speechbrain_langid").expanduser()
        actor_savedir = cache_root / f"actor_{os.getpid()}"
        self._classifier = _encoder_classifier_class().from_hparams(
            source=self._load_source(),
            savedir=str(actor_savedir),
            run_opts={"device": str(self._device)},
        )
        logger.info("Loaded SpeechBrain LID source {} on {}", self.source, self._device)

    def unload_model(self) -> None:
        """Release worker-local classifier state and reclaim CUDA cache."""
        self._classifier = None
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
        if self._classifier is None:
            msg = "SpeechBrainLIDAdapter is not initialized; call load_model() first"
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

        max_samples = max(waveform.size for waveform in waveforms)
        padded = torch.zeros((len(waveforms), max_samples), dtype=torch.float32)
        for row, waveform in enumerate(waveforms):
            padded[row, : waveform.size] = torch.from_numpy(waveform)
        relative_lengths = torch.tensor(
            [waveform.size / max_samples for waveform in waveforms],
            dtype=torch.float32,
        )

        with torch.inference_mode():
            _probabilities, scores, _indices, labels = self._classifier.classify_batch(padded, relative_lengths)
        confidences = torch.exp(torch.as_tensor(scores)).reshape(-1).detach().cpu().tolist()
        if len(confidences) != len(valid_indices) or len(labels) != len(valid_indices):
            msg = (
                "SpeechBrain classifier returned "
                f"{len(confidences)} scores and {len(labels)} labels for {len(valid_indices)} inputs"
            )
            raise RuntimeError(msg)
        for item_index, label, confidence in zip(valid_indices, labels, confidences, strict=True):
            results[item_index] = AudioLIDResult(_normalize_language(label), float(confidence))
        return results
