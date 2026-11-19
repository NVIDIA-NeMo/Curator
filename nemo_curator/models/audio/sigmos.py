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

"""SIGMOS (Microsoft SIG-Challenge ONNX) quality-scoring adapter.

The adapter owns weight download, ONNX session lifecycle and inference.
``SIGMOSFilterStage`` owns task I/O and the threshold decision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
import torch
from loguru import logger

from nemo_curator.models.audio.third_party.sigmos.sigmos import build_sigmos_model

if TYPE_CHECKING:
    import numpy as np

_SIGMOS_MODEL_URL = (
    "https://github.com/microsoft/SIG-Challenge/raw/main/"
    "ICASSP2024/sigmos/model-sigmos_1697718653_41d092e8-epo-200.onnx"
)
_SIGMOS_MODEL_FILENAME = "model-sigmos_1697718653_41d092e8-epo-200.onnx"
_DEFAULT_MODEL_DIR = str(Path.home() / ".cache" / "nemo_curator" / "sigmos_model")


def download_sigmos_model(model_dir: str) -> str:
    """Download the SIGMOS ONNX model into ``model_dir`` once; return the validated file path."""
    weights_path = str(Path(model_dir) / _SIGMOS_MODEL_FILENAME)

    if not os.path.exists(weights_path):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading SIGMOS model from {_SIGMOS_MODEL_URL}")
        response = requests.get(_SIGMOS_MODEL_URL, timeout=120)
        response.raise_for_status()

        with open(weights_path, "wb") as f:
            f.write(response.content)

        logger.info(f"SIGMOS model saved to {weights_path}")

    if not os.path.isfile(weights_path) or os.path.getsize(weights_path) == 0:
        msg = (
            f"SIGMOS model file is missing or empty at {weights_path}. "
            f"Download manually from {_SIGMOS_MODEL_URL} "
            f"and place it at {weights_path}"
        )
        raise RuntimeError(msg)

    return weights_path


@dataclass
class SIGMOSAdapter:
    """Score one mono waveform with the SIGMOS ONNX model.

    ``model_path`` points at a local ONNX file and skips the download;
    otherwise the model is downloaded into ``model_dir``.
    """

    model_dir: str = _DEFAULT_MODEL_DIR
    model_path: str | None = None
    _model: Any = field(default=None, init=False, repr=False)

    def download_weights_on_node(self) -> None:
        """Download the default model; an explicit ``model_path`` needs no download."""
        if self.model_path is None:
            download_sigmos_model(self.model_dir)

    def _resolve_model_path(self) -> str:
        """Resolve the ONNX model path: model_path override → model_dir download."""
        if self.model_path is not None:
            if os.path.isfile(self.model_path) and os.path.getsize(self.model_path) > 0:
                return self.model_path
            msg = f"SIGMOS model_path does not exist, is empty, or is not a file: {self.model_path!r}"
            raise FileNotFoundError(msg)
        return download_sigmos_model(self.model_dir)

    def load_model(self, *, num_gpus: int) -> None:
        """Open the ONNX session on the current CUDA device, or on CPU."""
        if self._model is not None:
            return
        from nemo_curator.utils.gpu_utils import ensure_cudnn_loaded

        ensure_cudnn_loaded()
        resolved_path = self._resolve_model_path()
        if num_gpus > 0 and torch.cuda.is_available():
            device_id = int(torch.cuda.current_device())
            self._model = build_sigmos_model(force_cpu=False, device_id=device_id, model_path=resolved_path)
        else:
            self._model = build_sigmos_model(force_cpu=True, model_path=resolved_path)
        logger.info("SIGMOS model loaded successfully")

    def unload_model(self) -> None:
        """Release the ONNX session and CUDA cache."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Return the raw SIGMOS score mapping (``MOS_NOISE``, ``MOS_OVRL``, ...)."""
        if self._model is None:
            msg = "SIGMOSAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)
        return self._model.run(audio=audio, sr=sr)
