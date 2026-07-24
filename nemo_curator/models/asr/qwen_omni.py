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
from dataclasses import dataclass
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


def _require_audio_vllm_stack(*, context: str) -> None:
    """Raise a single ImportError listing missing audio-vLLM dependencies."""
    missing: list[str] = []
    if SamplingParams is None:
        missing.append("vllm")
    if process_mm_info is None:
        missing.append("qwen-omni-utils")
    if Qwen3OmniMoeProcessor is None:
        missing.append("transformers (Qwen3OmniMoeProcessor)")
    if missing:
        msg = (
            f"QwenOmniASRAdapter {context} requires the audio_vllm extra. "
            f"Missing: {', '.join(missing)}. Install with: uv sync --extra audio_vllm"
        )
        raise ImportError(msg)


_QWEN3_OMNI_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
_MIN_QWEN_AUDIO_SAMPLES = 1600
_PROMPT_CONTENT_ORDERS = frozenset({"text_audio", "audio_text"})
_FOLLOWUP_PROMPT_DEFAULT = (
    "Now listen to the audio again and add any false starts, filler words "
    "and preserve colloquial words (like lemme, gonna, wanna, etc) as is "
    "spoken in the audio."
)


@dataclass
class QwenOmniASRAdapter:
    """Qwen3-Omni in-process vLLM adapter (thinker-only path).

    Stages construct adapters via
    ``cls(model_id=..., revision=..., **adapter_kwargs)``, so every field
    below is a keyword-only knob settable from the YAML ``adapter_kwargs``.

    Notable Args (most are plain vLLM/sampling knobs):
        prompt_text / *_file: Turn-1 user prompt; ``{language}`` and
            ``{transcript}`` are interpolated per-item when the stage supplies
            language and reference text values. ``*_file`` variants load text
            from a UTF-8 file at ``__post_init__`` time.
        en_prompt_text / en_prompt_file: override used when language is
            ``"English"``.
        followup_prompt / *_file: when set, enables Turn-2 inference.
        system_prompt / *_file: optional system message for both turns.
        prompt_content_order: order of text and audio blocks in each user
            message. ``text_audio`` preserves reference-adapter behavior;
            ``audio_text`` matches Qwen's official ASR cookbook.
        enable_prefix_caching: default ``True`` since prompts repeat across
            requests; disable for highly variable prompts.
        limit_mm_per_prompt_audio: per-request audio cap. This adapter attaches
            one audio item in both single-turn and two-turn modes, so ``1`` is
            sufficient for either. The default ``2`` is a permissive adapter
            default. Image/video multimodal caps are passed as ``1`` for
            Qwen/vLLM compatibility even though ASR requests only attach audio.
        max_num_batched_tokens: optional vLLM scheduler/encoder-cache budget.
            Long single audio items can exceed the default multimodal encoder
            cache even when ``max_model_len`` is large enough; set this to at
            least the observed audio feature length for 40-50 minute probes.
        seed: exposed so reproducibility / bit-exactness tests can override.
    """

    model_id: str = _QWEN3_OMNI_MODEL_ID
    revision: str | None = None

    prompt_text: str = "Transcribe the audio."
    prompt_file: str | None = None
    en_prompt_text: str | None = None
    en_prompt_file: str | None = None
    followup_prompt: str | None = None
    followup_prompt_file: str | None = None
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    prompt_content_order: str = "text_audio"
    max_model_len: int = 32768
    max_num_batched_tokens: int | None = None
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.95
    max_output_tokens: int = 256
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int = 1
    repetition_penalty: float = 1.0
    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "xxhash"
    limit_mm_per_prompt_audio: int = 2
    seed: int = 1234

    def __post_init__(self) -> None:
        self.prompt_text = self._load_text(self.prompt_text, self.prompt_file) or ""
        self.en_prompt_text = self._load_text(self.en_prompt_text, self.en_prompt_file)
        self.followup_prompt = self._load_text(self.followup_prompt, self.followup_prompt_file)
        self.system_prompt = self._load_text(self.system_prompt, self.system_prompt_file)

        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens <= 0:
            msg = "max_num_batched_tokens must be positive when set"
            raise ValueError(msg)
        if self.max_output_tokens <= 0:
            msg = "max_output_tokens must be positive"
            raise ValueError(msg)
        if self.prompt_content_order not in _PROMPT_CONTENT_ORDERS:
            msg = (
                "prompt_content_order must be one of "
                f"{sorted(_PROMPT_CONTENT_ORDERS)}, got {self.prompt_content_order!r}"
            )
            raise ValueError(msg)
        if self.top_p is not None and not 0.0 < float(self.top_p) <= 1.0:
            msg = f"top_p must be in (0, 1] when set, got {self.top_p}"
            raise ValueError(msg)
        if self.repetition_penalty <= 0.0:
            msg = f"repetition_penalty must be greater than zero, got {self.repetition_penalty}"
            raise ValueError(msg)
        if self.limit_mm_per_prompt_audio <= 0:
            msg = "limit_mm_per_prompt_audio must be positive"
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
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        """Cache the model snapshot on local disk without touching the GPU."""
        kwargs: dict[str, Any] = {}
        if revision is not None:
            kwargs["revision"] = revision
        snapshot_download(model_id, **kwargs)

    def setup(self, *, num_gpus: int) -> None:
        if self._llm is not None:
            return
        if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus <= 0:
            msg = f"QwenOmniASRAdapter requires a positive integer num_gpus, got {num_gpus!r}"
            raise ValueError(msg)
        _require_audio_vllm_stack(context="setup()")

        logger.info(
            f"Loading QwenOmni model={self.model_id}  tp={num_gpus}  "
            f"max_model_len={self.max_model_len}  max_num_seqs={self.max_num_seqs}"
            + (
                f"  max_num_batched_tokens={self.max_num_batched_tokens}"
                if self.max_num_batched_tokens is not None
                else ""
            )
            + (f"  revision={self.revision}" if self.revision is not None else "")
        )

        engine_kwargs: dict[str, Any] = {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": num_gpus,
            "max_model_len": self.max_model_len,
            "seed": int(self.seed),
            "enable_prefix_caching": bool(self.enable_prefix_caching),
            "prefix_caching_hash_algo": str(self.prefix_caching_hash_algo),
        }
        if self.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = int(self.max_num_batched_tokens)
        if self.revision is not None:
            engine_kwargs["revision"] = self.revision

        try:
            proc_kwargs: dict[str, Any] = {}
            if self.revision is not None:
                proc_kwargs["revision"] = self.revision
            sampling_kwargs: dict[str, Any] = {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "max_tokens": self.max_output_tokens,
            }
            if self.top_p is not None:
                sampling_kwargs["top_p"] = float(self.top_p)
            if self.repetition_penalty != 1.0:
                sampling_kwargs["repetition_penalty"] = float(self.repetition_penalty)
            self._llm = create_vllm_llm(
                self.model_id,
                max_num_seqs=self.max_num_seqs,
                dtype="auto",
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 1, "video": 1, "audio": int(self.limit_mm_per_prompt_audio)},
                **engine_kwargs,
            )
            self._sampling_params = SamplingParams(**sampling_kwargs)
            self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id, **proc_kwargs)
        except Exception:
            self.teardown()
            raise

    def teardown(self) -> None:
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
            msg = "vLLM engine not initialized. Call setup() first."
            raise RuntimeError(msg)
        try:
            return self._llm.generate(prompts, sampling_params=self._sampling_params, use_tqdm=False)
        except (RuntimeError, TypeError, ValueError) as exc:
            msg = f"Error generating text: {exc}"
            raise RuntimeError(msg) from exc

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Run batched two-turn inference over per-task dicts.

        Skipped items (empty / unprocessable waveforms) round-trip as
        ``ASRResult(text="", skipped=True)`` to preserve ordering.
        """
        if not items:
            return []
        waveforms = [it["waveform"] for it in items]
        languages = [it.get("language") for it in items]
        reference_texts = [it.get("reference_text") for it in items]
        pred_texts, disfl_texts, skipped_indices = self._run_two_turn(
            waveforms,
            languages,
            reference_texts,
        )
        has_t2 = bool(self.followup_prompt)
        return [
            ASRResult(
                text=pred,
                secondary_text=(disfl if has_t2 else None),
                skipped=(i in skipped_indices),
            )
            for i, (pred, disfl) in enumerate(zip(pred_texts, disfl_texts, strict=True))
        ]

    # Input preparation

    def _resolve_prompt(self, template: str, language: str | None, reference_text: str | None = None) -> str:
        result = template
        if language and "{language}" in result:
            result = result.replace("{language}", language)
        if reference_text is None and "{transcript}" in result:
            msg = "Prompt template contains {transcript}, but no reference text was provided"
            raise ValueError(msg)
        if reference_text is not None:
            result = result.replace("{transcript}", reference_text)
        return result

    def _get_prompt_text(self, language: str | None, reference_text: str | None = None) -> str:
        if language == "English" and self.en_prompt_text:
            return self._resolve_prompt(self.en_prompt_text, language, reference_text)
        return self._resolve_prompt(self.prompt_text, language, reference_text)

    def _build_audio_prompt_messages(
        self,
        waveform: np.ndarray,
        language: str | None = None,
        reference_text: str | None = None,
    ) -> list[dict[str, Any]]:
        prompt = self._get_prompt_text(language, reference_text)
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            sys_prompt = self._resolve_prompt(self.system_prompt, language, reference_text)
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
        reference_text: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._build_audio_prompt_messages(waveform, language, reference_text)

    def _build_turn2_messages(
        self,
        waveform: np.ndarray,
        pred_text: str,
        language: str | None = None,
        reference_text: str | None = None,
    ) -> list[dict[str, Any]]:
        followup = self._resolve_prompt(self.followup_prompt or _FOLLOWUP_PROMPT_DEFAULT, language, reference_text)
        messages = self._build_audio_prompt_messages(waveform, language, reference_text)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": pred_text}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": followup},
                ],
            }
        )
        return messages

    def _pack_vllm_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Render chat ``messages`` into a vLLM request dict (shared by both turns)."""
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
        reference_text: str | None = None,
    ) -> tuple[dict[str, Any], np.ndarray] | None:
        try:
            if waveform.size == 0:
                logger.warning("Skipping empty waveform")
                return None
            if waveform.size < _MIN_QWEN_AUDIO_SAMPLES:
                logger.warning("Skipping too-short waveform ({} samples)", waveform.size)
                return None
            messages = self._build_messages(waveform, language, reference_text)
            inputs = self._pack_vllm_inputs(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to preprocess audio, skipping (waveform shape={}): {}",
                getattr(waveform, "shape", None),
                exc,
            )
            return None

        return inputs, waveform

    def _prepare_batch(
        self,
        waveforms: list[np.ndarray],
        languages: list[str | None] | None = None,
        reference_texts: list[str | None] | None = None,
    ) -> list[tuple[dict[str, Any], np.ndarray] | None]:
        langs = languages or [None] * len(waveforms)
        refs = reference_texts or [None] * len(waveforms)
        return [
            self._prepare_single(waveform, language, reference_text)
            for waveform, language, reference_text in zip(waveforms, langs, refs, strict=True)
        ]

    def _prepare_turn2_single(
        self,
        waveform: np.ndarray,
        pred_text: str,
        language: str | None = None,
        reference_text: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            messages = self._build_turn2_messages(waveform, pred_text, language, reference_text)
            inputs = self._pack_vllm_inputs(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to preprocess Turn 2 audio (shape={}): {}",
                getattr(waveform, "shape", None),
                exc,
            )
            return None

        return inputs

    def _prepare_turn2_batch(
        self,
        waveforms: list[np.ndarray],
        pred_texts: list[str],
        languages: list[str | None] | None = None,
        reference_texts: list[str | None] | None = None,
    ) -> list[dict[str, Any] | None]:
        langs = languages or [None] * len(waveforms)
        refs = reference_texts or [None] * len(waveforms)
        return [
            self._prepare_turn2_single(w, pt, lang, ref)
            for w, pt, lang, ref in zip(waveforms, pred_texts, langs, refs, strict=True)
        ]

    @staticmethod
    def _first_output_text(output: Any) -> str:  # noqa: ANN401
        sequences = getattr(output, "outputs", None) or []
        if not sequences:
            return ""
        return (getattr(sequences[0], "text", "") or "").strip()

    def _infer_turn(
        self,
        inputs: list[dict[str, Any]],
        indices: list[int],
        n: int,
    ) -> list[str]:
        """Run one vLLM turn and scatter its texts back to input order.

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

    def _run_two_turn(
        self,
        waveforms: list[np.ndarray],
        languages: list[str | None] | None = None,
        reference_texts: list[str | None] | None = None,
    ) -> tuple[list[str], list[str], set[int]]:
        """Run batched two-turn inference on in-memory waveforms.

        Returns ``(pred_texts, disfluency_texts, skipped_indices)``.
        ``disfluency_texts`` is all empty strings when ``followup_prompt``
        is not set.
        """
        n = len(waveforms)

        # -- Turn 1 ----------------------------------------------------------
        prepared = self._prepare_batch(waveforms, languages, reference_texts)
        valid_indices = [i for i, p in enumerate(prepared) if p is not None]
        valid_inputs = [prepared[i][0] for i in valid_indices]
        prepared_waveforms: dict[int, np.ndarray] = {i: prepared[i][1] for i in valid_indices}
        skipped_indices = set(range(n)) - set(valid_indices)

        if not valid_inputs:
            logger.warning(f"All {n} audio samples in batch failed preprocessing")
            return [""] * n, [""] * n, skipped_indices

        if len(valid_inputs) < n:
            logger.warning(f"Skipped {n - len(valid_inputs)}/{n} corrupt audio samples")

        pred_texts = self._infer_turn(valid_inputs, valid_indices, n)
        empty_output_indices = {i for i in valid_indices if not pred_texts[i]}
        if empty_output_indices:
            skipped_indices.update(empty_output_indices)
            logger.warning(
                "Skipping {}/{} audio samples with empty Turn 1 vLLM output",
                len(empty_output_indices),
                len(valid_indices),
            )

        # -- Turn 2 (disfluency refinement) -----------------------------------
        if not self.followup_prompt:
            return pred_texts, [""] * n, skipped_indices

        t2_indices = [i for i in valid_indices if i not in skipped_indices and pred_texts[i]]
        if not t2_indices:
            return pred_texts, [""] * n, skipped_indices

        langs = languages or [None] * n
        refs = reference_texts or [None] * n
        t2_prepared = self._prepare_turn2_batch(
            [prepared_waveforms[i] for i in t2_indices],
            [pred_texts[i] for i in t2_indices],
            [langs[i] for i in t2_indices],
            [refs[i] for i in t2_indices],
        )

        t2_valid = [(i, p) for i, p in zip(t2_indices, t2_prepared, strict=False) if p is not None]
        if not t2_valid:
            logger.warning("All Turn 2 samples failed preprocessing")
            return pred_texts, [""] * n, skipped_indices

        t2_valid_indices = [i for i, _ in t2_valid]
        t2_inputs = [p for _, p in t2_valid]
        disfluency_texts = self._infer_turn(t2_inputs, t2_valid_indices, n)

        return pred_texts, disfluency_texts, skipped_indices
