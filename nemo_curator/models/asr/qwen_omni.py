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

"""Qwen3-Omni ASR adapter (in-process vLLM).

Implements the :class:`~nemo_curator.models.asr.ASRAdapter` protocol on the
in-process vLLM thinker-only path. Two-turn (Turn-1 transcribe, Turn-2
disfluency/refinement) when ``followup_prompt`` is set; single-turn otherwise.

Engine plumbing is inherited from
:class:`nemo_curator.models.vllm_model.VLLMBase`; this module adds the
Qwen-Omni surface (multimodal preprocessing, prompt construction, prep thread
pool, adapter protocol methods). Both turns share ``_infer_turn`` and
``_pack_vllm_inputs``, differing only in prompt and output list.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download
from loguru import logger

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.models.vllm_model import VLLM_AVAILABLE, VLLMBase
from nemo_curator.utils.gpu_utils import get_gpu_count

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    process_mm_info = None  # type: ignore[assignment,misc]

try:
    from transformers import Qwen3OmniMoeProcessor
except ImportError:
    Qwen3OmniMoeProcessor = None  # type: ignore[assignment,misc]


def _require_audio_qwen_stack(*, context: str) -> None:
    """Raise a single ImportError listing missing audio_qwen-only deps."""
    missing: list[str] = []
    if not VLLM_AVAILABLE:
        missing.append("vllm")
    if process_mm_info is None:
        missing.append("qwen-omni-utils")
    if Qwen3OmniMoeProcessor is None:
        missing.append("transformers (Qwen3OmniMoeProcessor)")
    if missing:
        msg = (
            f"QwenOmniASRAdapter {context} requires the audio_qwen extra. "
            f"Missing: {', '.join(missing)}. Install with: uv sync --extra audio_qwen"
        )
        raise ImportError(msg)


_QWEN3_OMNI_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
_QWEN_SAMPLE_RATE = 16000
_MIN_QWEN_AUDIO_SAMPLES = 1600
_WAVEFORM_2D_NDIM = 2
_FOLLOWUP_PROMPT_DEFAULT = (
    "Now listen to the audio again and add any false starts, filler words "
    "and preserve colloquial words (like lemme, gonna, wanna, etc) as is "
    "spoken in the audio."
)


@dataclass
class QwenOmniASRAdapter(VLLMBase):
    """Qwen3-Omni in-process vLLM adapter (thinker-only path).

    Stages construct adapters via
    ``cls(model_id=..., revision=..., **adapter_kwargs)``, so every field
    below is a keyword-only knob settable from the YAML ``adapter_kwargs``.

    Resource expectations:
        * ~40 GB VRAM for Qwen3-Omni-30B-A3B (FP8): one A100-80GB or two
          A100-40GB with ``tensor_parallel_size=2``.
        * ~50-80 audio-seconds/GPU-second on A100-80GB at ``batch_size=32``.
        * ~15 GB cached weights on first run (HuggingFace Hub).

    Notable Args (most are plain vLLM/sampling knobs):
        prompt_text / *_file: Turn-1 user prompt; ``{language}`` and
            ``{transcript}`` are interpolated per-item when the stage supplies
            language and reference text values. ``*_file`` variants load text
            from a UTF-8 file at ``__post_init__`` time.
        en_prompt_text / en_prompt_file: override used when language is
            ``"English"``.
        followup_prompt / *_file: when set, enables Turn-2 inference.
        system_prompt / *_file: optional system message for both turns.
        tensor_parallel_size: ``None`` -> auto-detect from visible GPUs.
        enable_prefix_caching: default ``True`` since prompts repeat across
            requests; disable for highly variable prompts.
        limit_mm_per_prompt_audio: per-prompt audio cap; ``2`` covers the
            two-turn flow, ``1`` for strictly single-turn. This audio adapter
            passes image/video multimodal caps as ``1`` for Qwen/vLLM
            compatibility even though ASR requests only attach audio payloads.
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
    max_model_len: int = 32768
    max_num_batched_tokens: int | None = None
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.95
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 1
    prep_workers: int = 8

    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "xxhash"
    limit_mm_per_prompt_audio: int = 2
    seed: int = 1234

    last_metrics: dict[str, float] = field(default_factory=dict)

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
        if self.limit_mm_per_prompt_audio <= 0:
            msg = "limit_mm_per_prompt_audio must be positive"
            raise ValueError(msg)

        self._processor: Any = None
        self._prep_pool: ThreadPoolExecutor | None = None

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

    def setup(self) -> None:
        if self._llm is not None:
            return
        _require_audio_qwen_stack(context="setup()")

        tp_size = self.tensor_parallel_size or get_gpu_count()
        logger.info(
            f"Loading QwenOmni model={self.model_id}  tp={tp_size}  "
            f"max_model_len={self.max_model_len}  max_num_seqs={self.max_num_seqs}"
            + (
                f"  max_num_batched_tokens={self.max_num_batched_tokens}"
                if self.max_num_batched_tokens is not None
                else ""
            )
            + (f"  revision={self.revision}" if self.revision is not None else "")
        )

        model_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "trust_remote_code": True,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": tp_size,
            "limit_mm_per_prompt": {"image": 1, "video": 1, "audio": int(self.limit_mm_per_prompt_audio)},
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "seed": int(self.seed),
            "enable_prefix_caching": bool(self.enable_prefix_caching),
            "prefix_caching_hash_algo": str(self.prefix_caching_hash_algo),
        }
        if self.max_num_batched_tokens is not None:
            model_kwargs["max_num_batched_tokens"] = int(self.max_num_batched_tokens)
        if self.revision is not None:
            model_kwargs["revision"] = self.revision

        sampling_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_tokens": self.max_output_tokens,
        }

        try:
            self._init_engine(model_kwargs, sampling_kwargs)

            proc_kwargs: dict[str, Any] = {}
            if self.revision is not None:
                proc_kwargs["revision"] = self.revision
            self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id, **proc_kwargs)
            self._prep_pool = ThreadPoolExecutor(max_workers=self.prep_workers)
        except Exception:
            self.teardown()
            raise

    def teardown(self) -> None:
        if self._prep_pool is not None:
            self._prep_pool.shutdown(wait=False)
            self._prep_pool = None
        self._processor = None
        self._cleanup_gpu()

    def estimate_item_cost(self, item: dict[str, Any]) -> float | None:
        """Return an optional scheduler cost for one prepared ASR item."""

        for key in ("estimated_vram_units", "estimated_encoder_tokens", "audio_seconds"):
            value = item.get(key)
            if value is not None:
                return float(value)
        return None

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Run batched two-turn inference over per-task dicts.

        Skipped items (empty / unprocessable waveforms) round-trip as
        ``ASRResult(text="", skipped=True)`` to preserve ordering.
        """
        if not items:
            return []
        waveforms = [it["waveform"] for it in items]
        sample_rates = [it["sample_rate"] for it in items]
        languages = [it.get("language") for it in items]
        reference_texts = [it.get("reference_text") for it in items]
        pred_texts, disfl_texts, skipped_indices = self._run_two_turn(
            waveforms,
            sample_rates,
            languages,
            reference_texts,
        )
        has_t2 = bool(self.followup_prompt)
        return [
            ASRResult(
                text=pred,
                secondary_text=(disfl if has_t2 else None),
                skipped=(i in skipped_indices),
                model_id=self.model_id,
            )
            for i, (pred, disfl) in enumerate(zip(pred_texts, disfl_texts, strict=True))
        ]

    # Input preparation

    @staticmethod
    def _to_mono_numpy_1d(waveform: object) -> np.ndarray:
        """Normalize Curator waveform objects to Qwen's 1-D mono numpy input."""
        if waveform is None:
            return np.asarray([], dtype=np.float32)
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        arr = np.asarray(waveform, dtype=np.float32)
        if arr.size == 0:
            return arr.reshape(0)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 1:
            return np.ascontiguousarray(arr)

        squeezed = np.squeeze(arr)
        if squeezed.ndim == 1:
            return np.ascontiguousarray(squeezed.astype(np.float32, copy=False))
        if squeezed.ndim == _WAVEFORM_2D_NDIM:
            # Curator's canonical waveform is channels-first (C, T). If an
            # adapter caller supplies channel-last (T, C), average over the
            # smaller channel-looking axis.
            axis = 0 if squeezed.shape[0] <= squeezed.shape[1] else 1
            return np.ascontiguousarray(squeezed.mean(axis=axis).astype(np.float32, copy=False))

        msg = f"Expected 1-D or 2-D waveform, got shape {arr.shape}"
        raise ValueError(msg)

    @staticmethod
    def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int = _QWEN_SAMPLE_RATE) -> np.ndarray:
        if orig_sr == target_sr:
            return waveform
        import librosa

        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

    def _resolve_prompt(self, template: str, language: str | None, reference_text: str | None = None) -> str:
        result = template
        if language and "{language}" in result:
            result = result.replace("{language}", language)
        if reference_text is not None and "{transcript}" in result:
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
            sys_prompt = self._resolve_prompt(self.system_prompt, language)
            messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": waveform},
                    {"type": "text", "text": prompt},
                ],
            }
        )
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
        waveform: object,
        sample_rate: int,
        language: str | None = None,
        reference_text: str | None = None,
    ) -> tuple[dict[str, Any], np.ndarray] | None:
        try:
            waveform_1d = self._to_mono_numpy_1d(waveform)
            if waveform_1d.size == 0:
                logger.warning("Skipping empty waveform")
                return None
            if waveform_1d.size < _MIN_QWEN_AUDIO_SAMPLES:
                logger.warning("Skipping too-short waveform ({} samples)", waveform_1d.size)
                return None
            waveform_16k = self._resample(waveform_1d, sample_rate)
            messages = self._build_messages(waveform_16k, language, reference_text)
            inputs = self._pack_vllm_inputs(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to preprocess audio, skipping (waveform shape={}, sr={}): {}",
                getattr(waveform, "shape", None),
                sample_rate,
                exc,
            )
            return None

        return inputs, waveform_16k

    def _prepare_batch(
        self,
        waveforms: list[object],
        sample_rates: list[int],
        languages: list[str | None] | None = None,
        reference_texts: list[str | None] | None = None,
    ) -> list[tuple[dict[str, Any], np.ndarray] | None]:
        langs = languages or [None] * len(waveforms)
        refs = reference_texts or [None] * len(waveforms)
        if self._prep_pool is None:
            return [
                self._prepare_single(w, sr, lang, ref)
                for w, sr, lang, ref in zip(waveforms, sample_rates, langs, refs, strict=False)
            ]
        return list(self._prep_pool.map(self._prepare_single, waveforms, sample_rates, langs, refs))

    def _prepare_turn2_single(
        self,
        waveform_16k: np.ndarray,
        pred_text: str,
        language: str | None = None,
        reference_text: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            messages = self._build_turn2_messages(waveform_16k, pred_text, language, reference_text)
            inputs = self._pack_vllm_inputs(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to preprocess Turn 2 audio (shape={}): {}",
                getattr(waveform_16k, "shape", None),
                exc,
            )
            return None

        return inputs

    def _prepare_turn2_batch(
        self,
        waveforms_16k: list[np.ndarray],
        pred_texts: list[str],
        languages: list[str | None] | None = None,
        reference_texts: list[str | None] | None = None,
    ) -> list[dict[str, Any] | None]:
        langs = languages or [None] * len(waveforms_16k)
        refs = reference_texts or [None] * len(waveforms_16k)
        if self._prep_pool is None:
            return [
                self._prepare_turn2_single(w, pt, lang, ref)
                for w, pt, lang, ref in zip(waveforms_16k, pred_texts, langs, refs, strict=False)
            ]
        return list(self._prep_pool.map(self._prepare_turn2_single, waveforms_16k, pred_texts, langs, refs))

    @staticmethod
    def _count_output_tokens(outputs: list[Any]) -> float:
        total = 0.0
        for output in outputs:
            sequences = getattr(output, "outputs", None) or []
            if not sequences:
                continue
            token_ids = getattr(sequences[0], "token_ids", None)
            if token_ids is not None:
                total += float(len(token_ids))
        return total

    @staticmethod
    def _first_output_text(output: Any) -> str:  # noqa: ANN401
        sequences = getattr(output, "outputs", None) or []
        if not sequences:
            return ""
        return (getattr(sequences[0], "text", "") or "").strip()

    def _output_hit_token_limit(self, output: Any) -> bool:  # noqa: ANN401
        """Return whether vLLM stopped the first sequence at the token cap."""
        sequences = getattr(output, "outputs", None) or []
        if not sequences:
            return False
        sequence = sequences[0]
        finish_reason = getattr(sequence, "finish_reason", None)
        if finish_reason == "length":
            return True
        token_ids = getattr(sequence, "token_ids", None)
        return finish_reason is None and token_ids is not None and len(token_ids) >= self.max_output_tokens

    def _infer_turn(
        self,
        inputs: list[dict[str, Any]],
        indices: list[int],
        n: int,
    ) -> tuple[list[str], float, float]:
        """Run one vLLM turn and scatter its texts back to input order.

        ``indices[k]`` is the position in the length-``n`` batch that
        ``inputs[k]`` came from. Returns
        ``(texts_of_len_n, generation_time_s, output_token_count)``.
        """
        t0 = time.perf_counter()
        outputs = self._generate(inputs)
        generation_time_s = time.perf_counter() - t0
        output_tokens = self._count_output_tokens(outputs)
        truncated_indices = [
            idx for idx, output in zip(indices, outputs, strict=True) if self._output_hit_token_limit(output)
        ]
        if truncated_indices:
            msg = (
                f"Qwen ASR output reached max_output_tokens={self.max_output_tokens} for batch positions "
                f"{truncated_indices}; refusing to emit an incomplete transcript. Reduce "
                "ASRStage.max_inference_duration_s or increase adapter max_output_tokens."
            )
            raise RuntimeError(msg)
        texts: list[str] = [""] * n
        # strict=True: a count mismatch means a broken engine contract; fail
        # loud rather than silently emit empty text with skipped=False.
        for idx, out in zip(indices, outputs, strict=True):
            texts[idx] = self._first_output_text(out)
        return texts, generation_time_s, output_tokens

    def _run_vllm_turn(
        self,
        inputs: list[dict[str, Any]],
        indices: list[int],
        n: int,
        metrics: dict[str, float],
        turn_name: str,
    ) -> list[str]:
        texts, generation_s, output_tokens = self._infer_turn(inputs, indices, n)
        metrics[f"{turn_name}_generation_time_s"] = generation_s
        metrics[f"{turn_name}_output_tokens"] = output_tokens
        metrics["output_tokens"] += output_tokens
        return texts

    def _run_two_turn(
        self,
        waveforms: list[object],
        sample_rates: list[int],
        languages: list[str | None] | None = None,
        reference_texts: list[str | None] | None = None,
    ) -> tuple[list[str], list[str], set[int]]:
        """Run batched two-turn inference on in-memory waveforms.

        Returns ``(pred_texts, disfluency_texts, skipped_indices)``.
        ``disfluency_texts`` is all empty strings when ``followup_prompt``
        is not set.
        """
        n = len(waveforms)
        # audio_duration_s / waveform_bytes are deliberately omitted: the stage
        # (ASRStage.assemble) owns those canonical, adapter-agnostic counters.
        metrics: dict[str, float] = {
            "utterances_input": float(n),
            "turn1_prep_time_s": 0.0,
            "turn1_generation_time_s": 0.0,
            "turn2_prep_time_s": 0.0,
            "turn2_generation_time_s": 0.0,
            "turn1_valid_inputs": 0.0,
            "turn2_valid_inputs": 0.0,
            "utterances_skipped_preprocess": 0.0,
            "utterances_skipped_empty_output": 0.0,
            "output_tokens": 0.0,
            "turn1_output_tokens": 0.0,
            "turn2_output_tokens": 0.0,
        }
        self.last_metrics = metrics

        # -- Turn 1 ----------------------------------------------------------
        prep_t0 = time.perf_counter()
        prepared = self._prepare_batch(waveforms, sample_rates, languages, reference_texts)
        metrics["turn1_prep_time_s"] = time.perf_counter() - prep_t0
        valid_indices = [i for i, p in enumerate(prepared) if p is not None]
        valid_inputs = [prepared[i][0] for i in valid_indices]
        waveforms_16k: dict[int, np.ndarray] = {i: prepared[i][1] for i in valid_indices}
        skipped_indices = set(range(n)) - set(valid_indices)
        metrics["turn1_valid_inputs"] = float(len(valid_inputs))
        metrics["utterances_skipped_preprocess"] = float(len(skipped_indices))

        if not valid_inputs:
            logger.warning(f"All {n} audio samples in batch failed preprocessing")
            return [""] * n, [""] * n, skipped_indices

        if len(valid_inputs) < n:
            logger.warning(f"Skipped {n - len(valid_inputs)}/{n} corrupt audio samples")

        pred_texts = self._run_vllm_turn(valid_inputs, valid_indices, n, metrics, "turn1")
        empty_output_indices = {i for i in valid_indices if not pred_texts[i]}
        if empty_output_indices:
            skipped_indices.update(empty_output_indices)
            metrics["utterances_skipped_empty_output"] = float(len(empty_output_indices))
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
        t2_prep_t0 = time.perf_counter()
        t2_prepared = self._prepare_turn2_batch(
            [waveforms_16k[i] for i in t2_indices],
            [pred_texts[i] for i in t2_indices],
            [langs[i] for i in t2_indices],
            [refs[i] for i in t2_indices],
        )
        metrics["turn2_prep_time_s"] = time.perf_counter() - t2_prep_t0

        t2_valid = [(i, p) for i, p in zip(t2_indices, t2_prepared, strict=False) if p is not None]
        metrics["turn2_valid_inputs"] = float(len(t2_valid))
        if not t2_valid:
            logger.warning("All Turn 2 samples failed preprocessing")
            return pred_texts, [""] * n, skipped_indices

        t2_valid_indices = [i for i, _ in t2_valid]
        t2_inputs = [p for _, p in t2_valid]
        disfluency_texts = self._run_vllm_turn(t2_inputs, t2_valid_indices, n, metrics, "turn2")

        return pred_texts, disfluency_texts, skipped_indices
