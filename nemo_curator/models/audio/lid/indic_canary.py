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

"""Indic Canary TensorRT-LLM implementation of the audio-LID adapter."""

from __future__ import annotations

import gc
import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from .base import (
    AudioLIDResult,
    _validate_sample_rate,
    _validate_single_device_gpu_count,
    _waveform_from_item,
)

_TARGET_SAMPLE_RATE = 16_000
_MIN_MODEL_SAMPLES = 400
_DEFAULT_RUNTIME_CLASS_PATH = "nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime.CanaryTRTLLM"
_ENGINE_FILES = (
    "encoder/encoder.plan",
    "encoder/config.json",
    "decoder/rank0.engine",
    "decoder/config.json",
    "decoder/vocab.json",
    "preprocessor/config.json",
    "preprocessor/mel_basis.pt",
)
_NON_LANGUAGE_SPECIAL_TOKENS = frozenset({"pnc", "itn"})
_LANGUAGE_CODE_LENGTHS = frozenset({2, 3})
_MAX_LANGUAGE_CODE_PARTS = 2
_COUNTRY_CODE_LENGTH = 2


def _validate_engine_dir(engine_dir: str | Path, owner: str = "IndicCanaryLIDAdapter") -> Path:
    """Validate the portable engine artifacts required by the runtime."""
    if not str(engine_dir):
        msg = f"{owner} requires engine_dir to point at a prebuilt TensorRT-LLM engine"
        raise ValueError(msg)
    root = Path(engine_dir).expanduser()
    missing = [str(root / relative_path) for relative_path in _ENGINE_FILES if not (root / relative_path).is_file()]
    if missing:
        msg = f"engine_dir {str(root)!r} is missing required file(s): {missing}"
        raise FileNotFoundError(msg)
    return root


def _language_code_from_special_token(token: str) -> str | None:
    """Return a plausible ISO language code from one Canary special token."""
    if not token.startswith("<|") or not token.endswith("|>"):
        return None
    code = token[2:-2]
    if code.lower() in _NON_LANGUAGE_SPECIAL_TOKENS:
        return None
    parts = code.split("-")
    if len(parts) not in {1, _MAX_LANGUAGE_CODE_PARTS} or not all(part.isalpha() for part in parts):
        return None
    if len(parts[0]) not in _LANGUAGE_CODE_LENGTHS:
        return None
    if len(parts) == _MAX_LANGUAGE_CODE_PARTS and len(parts[1]) != _COUNTRY_CODE_LENGTH:
        return None
    return code.lower()


def _normalize_candidate_languages(candidate_languages: list[str] | None) -> frozenset[str] | None:
    if candidate_languages is None:
        return None
    return frozenset(str(language).strip().lower() for language in candidate_languages if str(language).strip())


def _resolve_runtime_class(path: str) -> type:
    module_name, separator, class_name = path.rpartition(".")
    if not separator or not module_name or not class_name:
        msg = f"IndicCanaryLIDAdapter.runtime_class_path must be a fully qualified class path, got {path!r}"
        raise ValueError(msg)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        msg = (
            "Indic Canary LID requires its TensorRT-LLM runtime dependencies. "
            "Use the tested isolated/container environment from the metadata-extraction deployment."
        )
        raise ImportError(msg) from exc
    runtime_class = getattr(module, class_name, None)
    if not isinstance(runtime_class, type):
        msg = f"Indic Canary runtime class not found: {path}"
        raise TypeError(msg)
    return runtime_class


def _pad_or_trim(audio: torch.Tensor, length: int) -> torch.Tensor:
    """Pad or clip a one-dimensional waveform without importing the runtime."""
    if audio.numel() >= length:
        return audio[:length]
    return torch.nn.functional.pad(audio, (0, length - audio.numel()))


@dataclass
class IndicCanaryLIDAdapter:
    """Identify language tokens emitted by a prebuilt Indic Canary engine."""

    engine_dir: str = ""
    sample_rate: int = _TARGET_SAMPLE_RATE
    runtime_class_path: str = _DEFAULT_RUNTIME_CLASS_PATH
    runtime_class: type | None = field(default=None, repr=False)
    num_beams: int = 1
    max_new_tokens: int = 1
    prompt_text: str | None = None
    max_duration_sec: float = 40.0
    # The reference prebuilt engine profiles at least 32 encoder frames. A
    # one-second floor matches the reference stage and keeps direct adapter use
    # comfortably inside that profile instead of relying on a wrapping stage to
    # reject sub-profile clips first.
    min_duration_sec: float = 1.0
    candidate_languages: list[str] | None = None
    model_batch_size: int = 32
    kv_cache_free_gpu_memory_fraction: float = 0.2
    cross_kv_cache_fraction: float = 0.2
    _model: Any = field(default=None, init=False, repr=False)
    _language_by_token_id: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.sample_rate = _validate_sample_rate(self.sample_rate, owner=type(self).__name__)
        if self.sample_rate != _TARGET_SAMPLE_RATE:
            msg = f"IndicCanaryLIDAdapter requires sample_rate={_TARGET_SAMPLE_RATE}, got {self.sample_rate}"
            raise ValueError(msg)
        for field_name in ("num_beams", "max_new_tokens", "model_batch_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                msg = f"IndicCanaryLIDAdapter.{field_name} must be a positive integer, got {value!r}"
                raise ValueError(msg)
        for field_name in ("max_duration_sec", "min_duration_sec"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                msg = f"IndicCanaryLIDAdapter.{field_name} must be a finite non-negative number, got {value!r}"
                raise ValueError(msg)
        if self.max_duration_sec <= 0 or self.min_duration_sec > self.max_duration_sec:
            msg = "IndicCanaryLIDAdapter duration bounds require 0 <= min_duration_sec <= max_duration_sec"
            raise ValueError(msg)
        for field_name in ("kv_cache_free_gpu_memory_fraction", "cross_kv_cache_fraction"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                msg = f"IndicCanaryLIDAdapter.{field_name} must be in [0, 1], got {value!r}"
                raise ValueError(msg)
        if not isinstance(self.runtime_class_path, str) or not self.runtime_class_path.strip():
            msg = "IndicCanaryLIDAdapter.runtime_class_path must be a non-empty string"
            raise ValueError(msg)

    @property
    def max_samples(self) -> int:
        return int(self.max_duration_sec * self.sample_rate)

    @property
    def min_samples(self) -> int:
        return int(self.min_duration_sec * self.sample_rate)

    def download_weights_on_node(self) -> None:
        """Validate the prebuilt bundle; this backend has no provider download."""
        _validate_engine_dir(self.engine_dir, type(self).__name__)

    def load_model(self, *, num_gpus: int) -> None:
        """Load one GPU-only TensorRT-LLM runtime lazily inside the worker."""
        if self._model is not None:
            return
        _validate_single_device_gpu_count(num_gpus, owner=type(self).__name__, gpu_required=True)
        if not torch.cuda.is_available():
            msg = "IndicCanaryLIDAdapter requires CUDA, but CUDA is not available"
            raise RuntimeError(msg)
        engine_dir = _validate_engine_dir(self.engine_dir, type(self).__name__)
        runtime_class = self.runtime_class or _resolve_runtime_class(self.runtime_class_path)
        self._model = runtime_class(
            engine_dir,
            device="cuda:0",
            kv_cache_free_gpu_memory_fraction=self.kv_cache_free_gpu_memory_fraction,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
        )
        self._language_by_token_id = self._collect_language_token_ids()
        if not self._language_by_token_id:
            msg = "Indic Canary tokenizer has no language special tokens matching candidate_languages"
            raise RuntimeError(msg)
        logger.info(
            "Loaded Indic Canary LID engine {} with {} candidate language tokens",
            engine_dir,
            len(self._language_by_token_id),
        )

    def unload_model(self) -> None:
        """Release the TensorRT-LLM runtime and CUDA cache state."""
        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
        self._model = None
        self._language_by_token_id = {}
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: {}", exc)

    def _collect_language_token_ids(self) -> dict[int, str]:
        candidates = _normalize_candidate_languages(self.candidate_languages)
        token_map = getattr(self._model.tokenizer, "id_to_token", {})
        languages: dict[int, str] = {}
        for token_id, token in token_map.items():
            code = _language_code_from_special_token(str(token))
            if code is not None and (candidates is None or code in candidates):
                languages[int(token_id)] = code
        return languages

    def _lid_prompt_ids(self) -> list[int]:
        tokenizer = self._model.tokenizer
        if self.prompt_text:
            prompt = self.prompt_text
        elif tokenizer.prompt_format == "canary2":
            prompt = "<|startofcontext|> <|startoftranscript|> <|emo:undefined|>"
        else:
            prompt = "<|startoftranscript|>"
        return [int(token_id) for token_id in tokenizer.encode(prompt)]

    @staticmethod
    def _generated_part(sequence: list[int], prompt_ids: list[int]) -> list[int]:
        return sequence[len(prompt_ids) :] if sequence[: len(prompt_ids)] == prompt_ids else sequence

    def _parse_language(self, sequence: list[int], prompt_ids: list[int]) -> AudioLIDResult:
        tokenizer = self._model.tokenizer
        eos_id = getattr(tokenizer, "eos_id", None)
        pad_id = getattr(tokenizer, "pad_id", None)
        normalized_sequence = [int(token_id) for token_id in sequence]
        for token_id in self._generated_part(normalized_sequence, prompt_ids):
            if token_id == eos_id:
                break
            if token_id == pad_id:
                continue
            language = self._language_by_token_id.get(token_id)
            if language is not None:
                # The current runtime does not expose generation logits.
                return AudioLIDResult(language, 1.0)
        return AudioLIDResult("", 0.0)

    def _identify_waveforms(self, waveforms: list[torch.Tensor]) -> list[AudioLIDResult]:
        prompt_ids = self._lid_prompt_ids()
        decoder_max_input = int(getattr(self._model.decoder, "max_input_len", 0))
        if len(prompt_ids) > decoder_max_input:
            msg = f"Indic Canary LID prompt length {len(prompt_ids)} exceeds decoder max_input_len {decoder_max_input}"
            raise ValueError(msg)

        runtime_batch_size = max(1, int(self._model.max_batch_size))
        batch_size = min(self.model_batch_size, runtime_batch_size)
        results: list[AudioLIDResult] = []
        for start in range(0, len(waveforms), batch_size):
            chunk = waveforms[start : start + batch_size]
            lengths = [min(waveform.numel(), self.max_samples) for waveform in chunk]
            pad_length = min(max(lengths), self.max_samples)
            padded = [_pad_or_trim(waveform, pad_length) for waveform in chunk]
            feature_lengths = [min(max(length, _MIN_MODEL_SAMPLES), pad_length) for length in lengths]

            decoder_input_ids = torch.tensor(prompt_ids, dtype=torch.int64).repeat(len(chunk), 1)
            decoder_input_ids = decoder_input_ids.to(self._model.device)
            stream = torch.cuda.current_stream("cuda")
            mel, mel_lengths = self._model.preprocessor.get_feats(padded, feature_lengths)
            encoded, encoded_lengths = self._model.encoder.infer(mel, mel_lengths, stream)
            output_ids = self._model.decoder.generate(
                decoder_input_ids,
                encoded,
                encoded_lengths,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
            if len(output_ids) != len(chunk):
                msg = f"Indic Canary returned {len(output_ids)} predictions for {len(chunk)} inputs"
                raise RuntimeError(msg)
            for row in output_ids:
                sequence = row[0] if row and isinstance(row[0], (list, tuple)) else row
                results.append(self._parse_language(sequence, prompt_ids))
        return results

    def identify_batch(self, items: list[dict[str, Any]]) -> list[AudioLIDResult]:
        """Run Indic Canary LID while preserving empty and short input rows."""
        if not items:
            return []
        if self._model is None:
            msg = "IndicCanaryLIDAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)

        results = [AudioLIDResult("", 0.0) for _ in items]
        valid_indices: list[int] = []
        waveforms: list[torch.Tensor] = []
        for index, item in enumerate(items):
            waveform = _waveform_from_item(item, owner=type(self).__name__)
            if waveform.size >= self.min_samples:
                valid_indices.append(index)
                waveforms.append(torch.from_numpy(waveform[: self.max_samples]))
        if not waveforms:
            return results

        predictions = self._identify_waveforms(waveforms)
        if len(predictions) != len(valid_indices):
            msg = f"Indic Canary returned {len(predictions)} predictions for {len(valid_indices)} inputs"
            raise RuntimeError(msg)
        for item_index, prediction in zip(valid_indices, predictions, strict=True):
            results[item_index] = prediction
        return results
