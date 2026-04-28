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

"""Qwen3-ASR model wrapper for in-process vLLM inference.

Uses the ``qwen_asr`` library which wraps vLLM internally and exposes a
high-level ``transcribe()`` API that accepts in-memory numpy waveforms.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.models.base import ModelInterface

if TYPE_CHECKING:
    import numpy as np

_QWEN3_ASR_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"


class QwenASR(ModelInterface):
    """Qwen3-ASR model via the ``qwen_asr`` library with vLLM backend.

    Audio is accepted as in-memory numpy arrays (mono, any sample rate).
    The ``qwen_asr`` library handles resampling to 16 kHz, chunking long
    audio, and batched vLLM inference internally.
    """

    def __init__(
        self,
        model_id: str = _QWEN3_ASR_MODEL_ID,
        gpu_memory_utilization: float = 0.7,
        max_new_tokens: int = 4096,
        max_inference_batch_size: int = 128,
    ):
        self.model_id = model_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_new_tokens = max_new_tokens
        self.max_inference_batch_size = max_inference_batch_size

        self._model: Any = None

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_id]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_transformers_compat() -> None:
        """Patch transformers.check_model_inputs for qwen-asr compatibility.

        Newer transformers changed check_model_inputs from a decorator factory
        (called with parentheses) to a plain decorator. The qwen-asr package
        uses the old ``@check_model_inputs()`` syntax which breaks on newer
        versions. This wraps it to accept both styles.
        """
        try:
            import transformers
            original = getattr(transformers, "check_model_inputs", None)
            if original is None:
                return
            import inspect
            sig = inspect.signature(original)
            params = list(sig.parameters.values())
            if params and params[0].name == "func":
                def compat_check_model_inputs(*args):  # noqa: ANN202
                    if args and callable(args[0]):
                        return original(args[0])
                    return original
                transformers.check_model_inputs = compat_check_model_inputs
        except Exception:  # noqa: BLE001, S110
            pass

    def setup(self) -> None:
        self._patch_transformers_compat()

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            msg = "qwen_asr is required for QwenASR. Install it: pip install qwen-asr[vllm]"
            raise ImportError(msg) from None

        logger.info(
            f"Loading QwenASR model={self.model_id}  "
            f"gpu_mem={self.gpu_memory_utilization}  "
            f"max_new_tokens={self.max_new_tokens}  "
            f"max_batch={self.max_inference_batch_size}"
        )

        self._model = Qwen3ASRModel.LLM(
            model=self.model_id,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_inference_batch_size=self.max_inference_batch_size,
            max_new_tokens=self.max_new_tokens,
            trust_remote_code=True,
            enforce_eager=True,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="xxhash",
        )

        logger.info("QwenASR model loaded")

    def teardown(self) -> None:
        del self._model
        self._model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        contexts: list[str] | None = None,
        languages: list[str | None] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Run batched ASR inference on in-memory audio waveforms.

        Args:
            waveforms: List of 1-D mono numpy float32 arrays.
            sample_rates: Corresponding sample rates for each waveform.
            contexts: Optional per-sample instruction strings for
                ``with_instruction`` mode.
            languages: Per-sample language names (e.g. ``"English"``).

        Returns:
            ``(texts, languages)`` -- transcribed text and detected
            language for each input.
        """
        if self._model is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        n = len(waveforms)
        valid_indices = [i for i, w in enumerate(waveforms) if w.size > 0]

        if not valid_indices:
            logger.warning(f"All {n} audio samples are empty, returning empty predictions")
            return [""] * n, [""] * n

        if len(valid_indices) < n:
            logger.warning(f"Skipping {n - len(valid_indices)}/{n} empty waveforms in QwenASR batch")

        valid_waveforms = [waveforms[i] for i in valid_indices]
        valid_rates = [sample_rates[i] for i in valid_indices]
        valid_langs = [languages[i] for i in valid_indices] if languages else None
        valid_contexts = [contexts[i] for i in valid_indices] if contexts else None

        audio_inputs: list[tuple[np.ndarray, int]] = list(
            zip(valid_waveforms, valid_rates, strict=True)
        )

        kwargs: dict[str, Any] = {
            "audio": audio_inputs,
            "language": valid_langs,
        }
        if valid_contexts is not None:
            kwargs["context"] = valid_contexts

        results = self._model.transcribe(**kwargs)

        texts: list[str] = [""] * n
        detected_langs: list[str] = [""] * n
        for idx, r in zip(valid_indices, results, strict=True):
            texts[idx] = getattr(r, "text", str(r))
            detected_langs[idx] = getattr(r, "language", "")

        return texts, detected_langs
