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

"""Adapter for an Indic Canary model exported as a TensorRT-LLM engine."""

from __future__ import annotations

import gc
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from nemo_curator.models.asr.base import ASRResult

_TARGET_SAMPLE_RATE = 16_000
_DEFAULT_MAX_DURATION_SEC = 40.0
_DEFAULT_MIN_DURATION_SEC = 0.5
_MIN_DURATION_SAMPLES = 400
_REQUIRED_ENGINE_FILES = (
    "encoder/encoder.plan",
    "encoder/config.json",
    "decoder/config.json",
    "decoder/rank0.engine",
    "decoder/vocab.json",
    "preprocessor/config.json",
    "preprocessor/mel_basis.pt",
)


class IndicCanaryTRTLLMASR:
    """Run static-batch Indic Canary inference from a prebuilt engine directory."""

    def __init__(  # noqa: PLR0913
        self,
        engine_dir: str,
        *,
        num_beams: int = 4,
        max_new_tokens: int = 374,
        pnc: bool = False,
        max_duration_sec: float = _DEFAULT_MAX_DURATION_SEC,
        min_duration_sec: float = _DEFAULT_MIN_DURATION_SEC,
        kv_cache_free_gpu_memory_fraction: float = 0.2,
        cross_kv_cache_fraction: float = 0.2,
    ) -> None:
        if not engine_dir:
            msg = "IndicCanaryTRTLLMASR.engine_dir must point at a prebuilt engine directory"
            raise ValueError(msg)
        if num_beams < 1 or max_new_tokens < 1:
            msg = "num_beams and max_new_tokens must both be at least 1"
            raise ValueError(msg)
        if max_duration_sec <= 0 or min_duration_sec <= 0:
            msg = "max_duration_sec and min_duration_sec must both be positive"
            raise ValueError(msg)
        if not 0 < kv_cache_free_gpu_memory_fraction < 1:
            msg = "kv_cache_free_gpu_memory_fraction must be between 0 and 1"
            raise ValueError(msg)
        if not 0 < cross_kv_cache_fraction < 1:
            msg = "cross_kv_cache_fraction must be between 0 and 1"
            raise ValueError(msg)

        self.model_id = engine_dir
        self.engine_dir = engine_dir
        self.num_beams = int(num_beams)
        self.max_new_tokens = int(max_new_tokens)
        self.pnc = bool(pnc)
        self.max_duration_sec = float(max_duration_sec)
        self.min_duration_sec = min(float(min_duration_sec), self.max_duration_sec)
        self.max_samples = int(self.max_duration_sec * _TARGET_SAMPLE_RATE)
        self.min_samples = int(self.min_duration_sec * _TARGET_SAMPLE_RATE)
        self.kv_cache_free_gpu_memory_fraction = float(kv_cache_free_gpu_memory_fraction)
        self.cross_kv_cache_fraction = float(cross_kv_cache_fraction)
        self._model: Any = None

    def download_weights_on_node(self) -> None:
        """Validate local engine artifacts without allocating GPU state."""
        root = Path(self.engine_dir)
        missing = [
            str(root / relative_path)
            for relative_path in _REQUIRED_ENGINE_FILES
            if not (root / relative_path).is_file()
        ]
        if missing:
            msg = f"engine_dir {self.engine_dir!r} is missing required file(s): {missing}"
            raise FileNotFoundError(msg)

    def load_model(self, *, num_gpus: int) -> None:
        """Load the TensorRT-LLM runtime on its one required GPU."""
        if self._model is not None:
            return
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, Integral) or num_gpus != 1:
            msg = f"IndicCanaryTRTLLMASR requires exactly one GPU, got {num_gpus!r}"
            raise ValueError(msg)
        self.download_weights_on_node()
        from nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime import CanaryTRTLLM

        logger.info("Loading Indic Canary TensorRT-LLM engine from {}", self.engine_dir)
        self._model = CanaryTRTLLM(
            self.engine_dir,
            device="cuda:0",
            kv_cache_free_gpu_memory_fraction=self.kv_cache_free_gpu_memory_fraction,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
        )

    def unload_model(self) -> None:
        """Release the engine and its CUDA allocations."""
        self._model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    def _normalize_language(self, language: str) -> str | None:
        tokenizer = self._model.tokenizer
        candidates = [language]
        if "-" in language:
            candidates.append(language.split("-", maxsplit=1)[0])
        supported_languages = set(getattr(tokenizer, "langs", ()))
        supports_prompt_language = getattr(tokenizer, "supports_prompt_language", None)
        for candidate in candidates:
            if (
                callable(supports_prompt_language) and supports_prompt_language(candidate)
            ) or candidate in supported_languages:
                return candidate
        return None

    def _prompt_config(self, language: str) -> dict[str, object]:
        return {
            "task": "transcribe",
            "pnc": self.pnc,
            "source_language": language,
            "target_language": language,
            "itn": False,
            "romanized": False,
            "timestamp": False,
            "diarize": False,
        }

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe supported rows and preserve their original positions."""
        if not items:
            return []
        if self._model is None:
            msg = "IndicCanaryTRTLLMASR is not initialized; call load_model() first"
            raise RuntimeError(msg)

        import torch

        from nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime import pad_or_trim

        results = [ASRResult(text="") for _ in items]
        prepared: list[Any] = []
        durations: list[int] = []
        normalized_languages: list[str] = []
        valid_indices: list[int] = []
        truncated: list[bool] = []
        audio_durations: list[float] = []

        for index, item in enumerate(items):
            language = str(item.get("language_code") or "").strip().lower()
            normalized_language = self._normalize_language(language)
            if normalized_language is None:
                results[index] = ASRResult(
                    text="",
                    skipped=True,
                    skip_reason="language_not_supported",
                    unsupported_language=language or None,
                    extras={"language_code": language, "language_unsupported": True},
                )
                continue

            waveform = np.asarray(item.get("waveform"), dtype=np.float32)
            if waveform.ndim != 1:
                msg = f"ASRStage must provide a mono 1-D waveform, got shape {waveform.shape}"
                raise ValueError(msg)
            sample_rate = int(item.get("sample_rate") or 0)
            if sample_rate != _TARGET_SAMPLE_RATE:
                msg = f"ASRStage must provide {_TARGET_SAMPLE_RATE} Hz audio; received {sample_rate} Hz"
                raise ValueError(msg)
            clipped = waveform[: self.max_samples]
            prepared.append(torch.from_numpy(np.ascontiguousarray(clipped)))
            durations.append(min(max(int(clipped.size), _MIN_DURATION_SAMPLES), self.max_samples))
            normalized_languages.append(normalized_language)
            valid_indices.append(index)
            truncated.append(waveform.size > self.max_samples)
            audio_durations.append(float(waveform.size) / _TARGET_SAMPLE_RATE)

        if not prepared:
            return results

        pad_length = max([self.min_samples, *[int(waveform.shape[0]) for waveform in prepared]])
        padded = [pad_or_trim(waveform, pad_length) for waveform in prepared]
        predictions = self._model.process_batch(
            padded,
            [min(duration, pad_length) for duration in durations],
            [self._prompt_config(language) for language in normalized_languages],
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
        )
        if len(predictions) != len(prepared):
            msg = f"Indic Canary returned {len(predictions)} transcriptions for {len(prepared)} inputs"
            raise RuntimeError(msg)
        for valid_position, prediction in enumerate(predictions):
            item_index = valid_indices[valid_position]
            results[item_index] = ASRResult(
                text=str(prediction),
                extras={
                    "language_code": normalized_languages[valid_position],
                    "truncated": truncated[valid_position],
                    "audio_duration_sec": audio_durations[valid_position],
                },
            )

        return results


__all__ = ["IndicCanaryTRTLLMASR"]
