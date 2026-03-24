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

"""
SIGMOS pipeline: in-memory MOS prediction for SIGMOSFilterStage.

Only predict_audio_mos(audio_data, sample_rate, model_path) is used by the stage.
"""

import os
import sys
from typing import Any

import numpy as np
import torch

try:
    from .third_party.sigmos.sigmos import build_sigmos_model
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, "third_party"))
    from sigmos.sigmos import build_sigmos_model

_MODEL_CACHE: dict[str, Any] = {}


class _SIGMOSPipeline:
    """Internal: model cache + predict_audio. Used only by predict_audio_mos."""

    def __init__(self, model_path: str | None = None) -> None:
        model_key = f"_{model_path}" if model_path else ""

        if torch.cuda.is_available():
            device_id = int(torch.cuda.current_device())
            cache_key = f"gpu_{device_id}{model_key}"
        else:
            cache_key = f"cpu{model_key}"

        if cache_key not in _MODEL_CACHE:
            if torch.cuda.is_available():
                _MODEL_CACHE[cache_key] = build_sigmos_model(
                    force_cpu=False,
                    device_id=device_id,
                    model_path=model_path,
                )
            else:
                _MODEL_CACHE[cache_key] = build_sigmos_model(
                    force_cpu=True,
                    model_path=model_path,
                )
        self.model = _MODEL_CACHE[cache_key]

    def predict_audio(self, audio_data: np.ndarray, sample_rate: int) -> dict[str, float]:
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)
        try:
            return self.model.run(audio=audio_data, sr=sample_rate)
        except Exception as e:
            raise RuntimeError(f"Failed to predict MOS for audio: {e}") from e


def predict_audio_mos(
    audio_data: np.ndarray,
    sample_rate: int,
    model_path: str | None = None,
) -> dict[str, float]:
    """Predict MOS for in-memory audio. Used by SIGMOSFilterStage."""
    pipeline = _SIGMOSPipeline(model_path=model_path)
    return pipeline.predict_audio(audio_data, sample_rate)
