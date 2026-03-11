#!/usr/bin/env python3
"""
SIGMOS pipeline: in-memory MOS prediction for SIGMOSFilterStage.

Only predict_audio_mos(audio_data, sample_rate, gpu_id, config) is used by the stage.
"""

import os
import sys
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

try:
    from .third_party.sigmos.sigmos import build_sigmos_model
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, "third_party"))
    from sigmos.sigmos import build_sigmos_model

_MODEL_CACHE: Dict[str, Any] = {}


def _get_config_val(
    config: Optional[Union[Dict[str, Any], Any]],
    key: str,
    default: Any = None,
) -> Any:
    if config is None:
        return default
    if isinstance(config, dict) and "sigmos" in config and isinstance(config["sigmos"], dict):
        val = config["sigmos"].get(key)
        if val is not None:
            return val
    if isinstance(config, dict):
        val = config.get(f"sigmos_{key}") or config.get(key)
        if val is not None:
            return val
    if hasattr(config, f"sigmos_{key}"):
        return getattr(config, f"sigmos_{key}")
    if hasattr(config, key):
        return getattr(config, key)
    return default


class _SIGMOSPipeline:
    """Internal: model cache + predict_audio. Used only by predict_audio_mos."""

    def __init__(
        self,
        gpu_id: int = 0,
        config: Optional[Union[Dict[str, Any], Any]] = None,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        use_gpu = _get_config_val(config, "use_gpu")
        if use_gpu is None:
            use_gpu = True
        if isinstance(config, dict):
            use_gpu = config.get("use_gpu", use_gpu)
        elif config is not None:
            use_gpu = getattr(config, "use_gpu", use_gpu)
        self.force_cpu = not use_gpu

        model_path = _get_config_val(config, "model_path", None)
        model_key = f"_{model_path}" if model_path else ""
        cache_key = f"cpu{model_key}" if self.force_cpu else f"gpu_{gpu_id}{model_key}"

        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = build_sigmos_model(
                force_cpu=self.force_cpu,
                device_id=gpu_id,
                model_path=model_path,
            )
        self.model = _MODEL_CACHE[cache_key]

    def predict_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)
        try:
            return self.model.run(audio=audio_data, sr=sample_rate)
        except Exception as e:
            raise RuntimeError(f"Failed to predict MOS for audio: {e}") from e


def predict_audio_mos(
    audio_data: np.ndarray,
    sample_rate: int,
    gpu_id: int = 0,
    config: Optional[Union[Dict[str, Any], Any]] = None,
) -> Dict[str, float]:
    """Predict MOS for in-memory audio. Used by SIGMOSFilterStage."""
    pipeline = _SIGMOSPipeline(gpu_id=gpu_id, config=config)
    return pipeline.predict_audio(audio_data, sample_rate)
