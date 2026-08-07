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

"""Qwen3-ASR vLLM implementation of the shared ASR adapter.

This uses the same ``Qwen3ASRModel.LLM`` construction and vLLM engine settings
as the nkoluguri reference. ``ASRStage`` owns mono conversion and resampling;
the adapter hands one prepared batch to one ``transcribe`` call and maps
results back to ``ASRResult`` positions.
"""

from __future__ import annotations

import gc
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from loguru import logger

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.utils.vllm_utils import merge_vllm_kwargs

_DEFAULT_QWEN3_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"

# Qwen's audio processor needs >=200 samples for STFT padding. 1600 samples
# (100 ms at 16 kHz) is a conservative floor that also matches the Qwen-Omni
# preprocessing path.
_MIN_SAMPLES = 1600


def _qwen_asr_model_cls() -> Any:  # noqa: ANN401
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        msg = "QwenASRAdapter requires the audio_cuda12 and vllm extras: uv sync --extra audio_cuda12 --extra vllm"
        raise ImportError(msg) from exc
    return Qwen3ASRModel


@dataclass
class QwenASRAdapter:
    """Run vLLM-backed Qwen3-ASR over Curator waveform items.

    Every valid item in one adapter call goes to a single ``transcribe`` call,
    so the caller's batch boundary is the model's batch boundary.
    ``max_inference_batch_size`` is the library's own internal cap and is passed
    through at construction.

    ``revision`` is accepted to satisfy the shared adapter constructor and is
    forwarded to both weight prefetch and the vLLM model loader.

    ``vllm_kwargs`` exposes additional engine settings, following the existing
    Qwen-Omni adapter convention. Adapter-owned settings cannot be overridden
    through this mapping. Its default is empty, so normal construction exactly
    matches the nkoluguri reference engine arguments.
    """

    model_id: str = _DEFAULT_QWEN3_ASR_MODEL
    revision: str | None = None
    gpu_memory_utilization: float = 0.7
    max_new_tokens: int = 4096
    max_inference_batch_size: int = 128
    vllm_kwargs: dict[str, Any] = field(default_factory=dict)
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_id:
            msg = "QwenASRAdapter.model_id must be non-empty"
            raise ValueError(msg)
        if not 0.0 < float(self.gpu_memory_utilization) <= 1.0:
            msg = f"QwenASRAdapter.gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            raise ValueError(msg)
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 0
        ):
            msg = f"QwenASRAdapter.max_new_tokens must be a positive integer, got {self.max_new_tokens!r}"
            raise ValueError(msg)
        if (
            not isinstance(self.max_inference_batch_size, int)
            or isinstance(self.max_inference_batch_size, bool)
            or self.max_inference_batch_size <= 0
        ):
            msg = (
                "QwenASRAdapter.max_inference_batch_size must be a positive integer, "
                f"got {self.max_inference_batch_size!r}"
            )
            raise ValueError(msg)
        self.vllm_kwargs = deepcopy(dict(self.vllm_kwargs))

    def _model_owned_vllm_kwargs(self) -> dict[str, Any]:
        """Return the qwen-asr constructor arguments owned by this adapter."""
        return {
            "model": self.model_id,
            "revision": self.revision,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "trust_remote_code": True,
            "enforce_eager": True,
            "enable_prefix_caching": True,
            "prefix_caching_hash_algo": "xxhash",
        }

    @classmethod
    def download_weights_on_node(cls, model_id: str, revision: str | None = None) -> None:
        """Populate the local Hugging Face cache without allocating a GPU."""
        snapshot_download(model_id, revision=revision)

    def load_model(self, *, num_gpus: int) -> None:
        """Load one worker-local Qwen3-ASR model through its vLLM backend."""
        if self._model is not None:
            return
        if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus != 1:
            msg = f"QwenASRAdapter requires exactly one integer GPU, got {num_gpus!r}"
            raise ValueError(msg)

        logger.info(
            "Loading QwenASRAdapter model={} gpu_mem={} max_new_tokens={} max_batch={}",
            self.model_id,
            self.gpu_memory_utilization,
            self.max_new_tokens,
            self.max_inference_batch_size,
        )
        model_kwargs = merge_vllm_kwargs(
            self.vllm_kwargs,
            self._model_owned_vllm_kwargs(),
            owner_description="adapter-owned arguments",
        )
        if model_kwargs["revision"] is None:
            del model_kwargs["revision"]
        try:
            self._model = _qwen_asr_model_cls().LLM(**model_kwargs)
        except Exception:
            self.unload_model()
            raise
        logger.info("QwenASRAdapter ready ({})", self.model_id)

    def unload_model(self) -> None:
        """Release the worker-local model and CUDA cache state."""
        self._model = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: {}", exc)

    @staticmethod
    def _waveform(item: dict[str, Any]) -> np.ndarray:
        waveform = np.asarray(item.get("waveform"), dtype=np.float32)
        if waveform.ndim != 1:
            msg = f"ASRStage must provide a mono 1-D waveform, got shape {waveform.shape}"
            raise ValueError(msg)
        return waveform

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe one adapter call while preserving input order."""
        if not items:
            return []

        valid_indices: list[int] = []
        audio_inputs: list[tuple[np.ndarray, int]] = []
        languages: list[str | None] = []
        for index, item in enumerate(items):
            waveform = self._waveform(item)
            source_rate = int(item.get("sample_rate") or 0)
            if waveform.size < _MIN_SAMPLES or source_rate <= 0:
                continue
            valid_indices.append(index)
            audio_inputs.append((waveform, source_rate))
            languages.append(item.get("language"))

        results = [ASRResult(text="", skipped=True) for _ in items]
        if not audio_inputs:
            logger.warning(
                "QwenASRAdapter: all {} items were shorter than {} samples or lacked a sample rate",
                len(items),
                _MIN_SAMPLES,
            )
            return results
        if len(audio_inputs) < len(items):
            logger.warning(
                "QwenASRAdapter: skipping {}/{} items shorter than {} samples",
                len(items) - len(audio_inputs),
                len(items),
                _MIN_SAMPLES,
            )

        outputs = self._model.transcribe(audio=audio_inputs, language=languages)

        outputs = list(outputs or [])
        if len(outputs) != len(valid_indices):
            msg = f"Qwen3-ASR returned {len(outputs)} transcriptions for {len(valid_indices)} valid inputs"
            raise RuntimeError(msg)

        for index, output in zip(valid_indices, outputs, strict=True):
            text = getattr(output, "text", output)
            text = "" if text is None else str(text)
            detected_language = getattr(output, "language", "") or ""
            results[index] = ASRResult(
                text=text,
                skipped=not text.strip(),
                extras={"detected_language": str(detected_language)} if detected_language else {},
            )
        return results
