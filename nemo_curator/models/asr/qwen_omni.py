# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-Omni ASR adapter using in-process vLLM."""

from __future__ import annotations

import gc
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import snapshot_download
from loguru import logger

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.utils.vllm_utils import create_vllm_llm

if TYPE_CHECKING:
    import numpy as np

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    process_mm_info = None  # type: ignore[assignment,misc]

try:
    from transformers import Qwen3OmniMoeProcessor
except ImportError:
    Qwen3OmniMoeProcessor = None  # type: ignore[assignment,misc]

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]


def _require_qwen_omni_stack(*, context: str) -> None:
    """Raise a single ImportError listing missing Qwen-Omni dependencies."""
    missing: list[str] = []
    if SamplingParams is None:
        missing.append("vllm")
    if process_mm_info is None:
        missing.append("qwen-omni-utils")
    if Qwen3OmniMoeProcessor is None:
        missing.append("transformers (Qwen3OmniMoeProcessor)")
    if missing:
        msg = (
            f"QwenOmniASRAdapter {context} requires the audio_cuda12 and vllm extras. "
            f"Missing: {', '.join(missing)}. Install with: "
            "uv sync --extra audio_cuda12 --extra vllm"
        )
        raise ImportError(msg)


_QWEN3_OMNI_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
_QWEN_OMNI_SAMPLE_RATE = 16000
_MIN_QWEN_AUDIO_SAMPLES = 1600
_PROMPT_CONTENT_ORDERS = frozenset({"text_audio", "audio_text"})
_RESERVED_VLLM_KWARGS = frozenset({"model", "revision", "tensor_parallel_size"})


def _default_vllm_kwargs() -> dict[str, Any]:
    """Return Qwen-Omni defaults forwarded to Curator's shared vLLM helper."""
    return {
        "max_model_len": 32768,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.95,
        "dtype": "auto",
        "trust_remote_code": True,
        "enable_prefix_caching": True,
        "prefix_caching_hash_algo": "xxhash",
        # TODO: Re-evaluate with vLLM 0.24, which includes https://github.com/vllm-project/vllm/pull/44264.
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 2},
        "seed": 1234,
    }


def _default_sampling_kwargs() -> dict[str, Any]:
    """Return Qwen-Omni defaults forwarded to vLLM ``SamplingParams``."""
    return {
        "temperature": 0.0,
        "top_k": 1,
        "repetition_penalty": 1.0,
    }


@dataclass
class QwenOmniASRAdapter:
    """Qwen3-Omni in-process vLLM adapter (thinker-only path).

    Stages construct adapters via
    ``cls(model_id=..., revision=..., **adapter_kwargs)``, so the fields below
    can be supplied from the YAML ``adapter_kwargs``.

    Notable Args:
        prompt_text / *_file: User prompt; ``{language}`` is interpolated
            per-item when the stage supplies a language. ``*_file`` variants
            load text from a UTF-8 file at ``__post_init__`` time.
        en_prompt_text / en_prompt_file: override used when language is
            ``"English"``.
        system_prompt / *_file: optional system message.
        prompt_content_order: order of text and audio blocks in each user
            message. ``audio_text`` matches Qwen's official ASR cookbook.
        max_output_tokens: maximum transcription tokens. Kept separate so the
            adapter remains the only source of ``SamplingParams.max_tokens``.
        vllm_kwargs: engine settings forwarded to Curator's shared
            ``create_vllm_llm`` helper. ``model``, ``revision``, and
            ``tensor_parallel_size`` are stage-owned and cannot be overridden.
        sampling_kwargs: settings forwarded to vLLM ``SamplingParams``.
            ``max_tokens`` is adapter-owned and cannot be overridden.
    """

    model_id: str = _QWEN3_OMNI_MODEL_ID
    revision: str | None = None

    prompt_text: str = "Transcribe the audio."
    prompt_file: str | None = None
    en_prompt_text: str | None = None
    en_prompt_file: str | None = None
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    prompt_content_order: str = "text_audio"
    max_output_tokens: int = 256
    vllm_kwargs: dict[str, Any] = field(default_factory=_default_vllm_kwargs)
    sampling_kwargs: dict[str, Any] = field(default_factory=_default_sampling_kwargs)

    def __post_init__(self) -> None:
        self.prompt_text = self._load_text(self.prompt_text, self.prompt_file) or ""
        self.en_prompt_text = self._load_text(self.en_prompt_text, self.en_prompt_file)
        self.system_prompt = self._load_text(self.system_prompt, self.system_prompt_file)

        if self.max_output_tokens <= 0:
            msg = "max_output_tokens must be positive"
            raise ValueError(msg)
        if self.prompt_content_order not in _PROMPT_CONTENT_ORDERS:
            msg = (
                "prompt_content_order must be one of "
                f"{sorted(_PROMPT_CONTENT_ORDERS)}, got {self.prompt_content_order!r}"
            )
            raise ValueError(msg)
        self.vllm_kwargs = deepcopy(dict(self.vllm_kwargs))
        self.sampling_kwargs = deepcopy(dict(self.sampling_kwargs))
        reserved_vllm_kwargs = sorted(_RESERVED_VLLM_KWARGS.intersection(self.vllm_kwargs))
        if reserved_vllm_kwargs:
            msg = f"vllm_kwargs cannot override stage-owned arguments: {', '.join(reserved_vllm_kwargs)}"
            raise ValueError(msg)
        if "max_tokens" in self.sampling_kwargs:
            msg = "sampling_kwargs cannot override adapter-owned max_tokens; use max_output_tokens"
            raise ValueError(msg)

        self._processor: Any = None
        self._llm: Any = None
        self._sampling_params: Any = None

    @staticmethod
    def _load_text(text: str | None, file_path: str | None) -> str | None:
        if file_path:
            path = Path(file_path)
            if not path.exists():
                msg = f"QwenOmniASRAdapter prompt file not found: {path}"
                raise FileNotFoundError(msg)
            return path.read_text(encoding="utf-8").strip()
        return text

    @classmethod
    def download_weights_on_node(cls, model_id: str, revision: str | None = None) -> None:
        """Cache the model snapshot on local disk without touching the GPU."""
        kwargs: dict[str, Any] = {}
        if revision is not None:
            kwargs["revision"] = revision
        snapshot_download(model_id, **kwargs)

    def load_model(self, *, num_gpus: int) -> None:
        if self._llm is not None:
            return
        if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus <= 0:
            msg = f"QwenOmniASRAdapter requires a positive integer num_gpus, got {num_gpus!r}"
            raise ValueError(msg)
        _require_qwen_omni_stack(context="load_model()")

        configured_max_model_len = self.vllm_kwargs.get("max_model_len")
        configured_max_num_seqs = self.vllm_kwargs.get("max_num_seqs")
        configured_batched_tokens = self.vllm_kwargs.get("max_num_batched_tokens")
        logger.info(
            f"Loading QwenOmni model={self.model_id}  tp={num_gpus}  "
            f"max_model_len={configured_max_model_len}  max_num_seqs={configured_max_num_seqs}"
            + (
                f"  max_num_batched_tokens={configured_batched_tokens}"
                if configured_batched_tokens is not None
                else ""
            )
            + (f"  revision={self.revision}" if self.revision is not None else "")
        )

        engine_kwargs = dict(self.vllm_kwargs)
        engine_kwargs["tensor_parallel_size"] = num_gpus
        if self.revision is not None:
            engine_kwargs["revision"] = self.revision

        try:
            proc_kwargs: dict[str, Any] = {}
            if self.revision is not None:
                proc_kwargs["revision"] = self.revision
            sampling_kwargs = dict(self.sampling_kwargs)
            sampling_kwargs["max_tokens"] = self.max_output_tokens
            self._llm = create_vllm_llm(self.model_id, **engine_kwargs)
            self._sampling_params = SamplingParams(**sampling_kwargs)
            self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id, **proc_kwargs)
        except Exception:
            self.unload_model()
            raise

    def unload_model(self) -> None:
        self._processor = None
        self._llm = None
        self._sampling_params = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: {}", exc)

    def _generate(self, prompts: list[Any]) -> list[Any]:
        if self._llm is None or self._sampling_params is None:
            msg = "vLLM engine not initialized. Call load_model() first."
            raise RuntimeError(msg)
        try:
            return self._llm.generate(prompts, sampling_params=self._sampling_params, use_tqdm=False)
        except (RuntimeError, TypeError, ValueError) as exc:
            msg = f"Error generating text: {exc}"
            raise RuntimeError(msg) from exc

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Run batched inference over per-task dicts.

        Skipped items (empty / unprocessable waveforms) round-trip as
        ``ASRResult(text="", skipped=True)`` to preserve ordering.
        """
        if not items:
            return []
        for index, item in enumerate(items):
            sample_rate = item.get("sample_rate")
            if sample_rate != _QWEN_OMNI_SAMPLE_RATE:
                msg = (
                    f"QwenOmniASRAdapter requires {_QWEN_OMNI_SAMPLE_RATE} Hz audio, "
                    f"but batch item {index} was decoded at {sample_rate!r} Hz"
                )
                raise ValueError(msg)
        waveforms = [it["waveform"] for it in items]
        languages = [it.get("language") for it in items]
        pred_texts, skipped_indices = self._run_inference(waveforms, languages)
        return [
            ASRResult(
                text=pred,
                skipped=(i in skipped_indices),
            )
            for i, pred in enumerate(pred_texts)
        ]

    # Input preparation

    def _resolve_prompt(self, template: str, language: str | None) -> str:
        result = template
        if language and "{language}" in result:
            result = result.replace("{language}", language)
        return result

    def _get_prompt_text(self, language: str | None) -> str:
        if language == "English" and self.en_prompt_text:
            return self._resolve_prompt(self.en_prompt_text, language)
        return self._resolve_prompt(self.prompt_text, language)

    def _build_audio_prompt_messages(
        self,
        waveform: np.ndarray,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        prompt = self._get_prompt_text(language)
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            sys_prompt = self._resolve_prompt(self.system_prompt, language)
            messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
        text_content = {"type": "text", "text": prompt}
        audio_content = {"type": "audio", "audio": waveform}
        content = (
            [audio_content, text_content]
            if self.prompt_content_order == "audio_text"
            else [text_content, audio_content]
        )
        messages.append({"role": "user", "content": content})
        return messages

    def _build_messages(
        self,
        waveform: np.ndarray,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._build_audio_prompt_messages(waveform, language)

    def _pack_vllm_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Render chat ``messages`` into a vLLM request dict."""
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs: dict[str, Any] = {
            "prompt": text,
            "multi_modal_data": {},
            "mm_processor_kwargs": {"use_audio_in_video": False},
        }
        if audios is not None:
            inputs["multi_modal_data"]["audio"] = audios
        if images is not None:
            inputs["multi_modal_data"]["image"] = images
        if videos is not None:
            inputs["multi_modal_data"]["video"] = videos
        return inputs

    def _prepare_single(
        self,
        waveform: np.ndarray,
        language: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            if waveform.size == 0:
                logger.warning("Skipping empty waveform")
                return None
            if waveform.size < _MIN_QWEN_AUDIO_SAMPLES:
                logger.warning("Skipping too-short waveform ({} samples)", waveform.size)
                return None
            messages = self._build_messages(waveform, language)
            inputs = self._pack_vllm_inputs(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to preprocess audio, skipping (waveform shape={}): {}",
                getattr(waveform, "shape", None),
                exc,
            )
            return None

        return inputs

    def _prepare_batch(
        self,
        waveforms: list[np.ndarray],
        languages: list[str | None] | None = None,
    ) -> list[dict[str, Any] | None]:
        langs = languages or [None] * len(waveforms)
        return [self._prepare_single(waveform, language) for waveform, language in zip(waveforms, langs, strict=True)]

    @staticmethod
    def _first_output_text(output: Any) -> str:  # noqa: ANN401
        sequences = getattr(output, "outputs", None) or []
        if not sequences:
            return ""
        return (getattr(sequences[0], "text", "") or "").strip()

    def _infer_batch(
        self,
        inputs: list[dict[str, Any]],
        indices: list[int],
        n: int,
    ) -> list[str]:
        """Run one vLLM batch and scatter its texts back to input order.

        ``indices[k]`` is the position in the length-``n`` batch that
        ``inputs[k]`` came from.
        """
        outputs = self._generate(inputs)
        texts: list[str] = [""] * n
        # strict=True: a count mismatch means a broken engine contract; fail
        # loud rather than silently emit empty text with skipped=False.
        for idx, out in zip(indices, outputs, strict=True):
            texts[idx] = self._first_output_text(out)
        return texts

    def _run_inference(
        self,
        waveforms: list[np.ndarray],
        languages: list[str | None] | None = None,
    ) -> tuple[list[str], set[int]]:
        """Run batched inference on in-memory waveforms."""
        n = len(waveforms)

        prepared = self._prepare_batch(waveforms, languages)
        valid_indices = [i for i, p in enumerate(prepared) if p is not None]
        valid_inputs = [p for p in prepared if p is not None]
        skipped_indices = set(range(n)) - set(valid_indices)

        if not valid_inputs:
            logger.warning(f"All {n} audio samples in batch failed preprocessing")
            return [""] * n, skipped_indices

        if len(valid_inputs) < n:
            logger.warning(f"Skipped {n - len(valid_inputs)}/{n} corrupt audio samples")

        pred_texts = self._infer_batch(valid_inputs, valid_indices, n)
        empty_output_indices = {i for i in valid_indices if not pred_texts[i]}
        if empty_output_indices:
            skipped_indices.update(empty_output_indices)
            logger.warning(
                "Skipping {}/{} audio samples with empty vLLM output",
                len(empty_output_indices),
                len(valid_indices),
            )

        return pred_texts, skipped_indices
