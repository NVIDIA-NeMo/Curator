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

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np  # noqa: TC002 — used at runtime, not just type hints
from loguru import logger

from nemo_curator.models.base import ModelInterface
from nemo_curator.utils.gpu_utils import get_gpu_count

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:  # type: ignore[no-redef]
        pass

    class SamplingParams:  # type: ignore[no-redef]
        pass


_QWEN3_OMNI_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
_QWEN_SAMPLE_RATE = 16000
_DEFAULT_MAX_RETRIES = 3


class QwenOmni(ModelInterface):
    """Qwen3-Omni multimodal model via vLLM for audio-conditioned text generation.

    Uses the *thinker-only* path: audio waveforms go in, text comes out.

    Audio is accepted as in-memory numpy arrays (mono, any sample rate).
    Arrays are resampled to 16 kHz before being passed to
    ``qwen_omni_utils.process_mm_info`` which natively accepts
    ``np.ndarray`` inputs — no temporary files are created.

    Performance knobs (learned from video/text/interleaved modalities):

    - ``fp8``: halves KV-cache memory, larger effective batches.
    - ``enforce_eager``: skip CUDA-graph compilation for faster cold start.
    - ``max_num_batched_tokens``: controls vLLM's internal chunked-prefill.
    - ``mm_cache_gb``: cache preprocessed multimodal tokens across requests.
    - ``max_retries``: auto-reset the vLLM engine on transient CUDA errors.

    Future optimisation opportunity -- **async prep/inference overlap**:

    The current ``generate()`` flow is sequential:
    ``prep_batch -> inference_t1 -> prep_turn2 -> inference_t2``.  Each
    phase blocks on the previous.  The ``ThreadPoolExecutor`` parallelises
    *within* a prep phase, but prep and inference never overlap across
    batches.  A double-buffering pattern -- where batch N+1's audio
    preprocessing runs on CPU while batch N's inference runs on GPU --
    could hide prep latency entirely and close the remaining throughput
    gap observed between raw vLLM (600 hrs/hr) and Curator in-process
    (468 hrs/hr -> ~515 with vLLM arg tuning) on YODAS 8xGPU benchmarks.
    This requires changes at the stage/executor level (e.g. an async
    generator or prefetch queue feeding ``process_batch``).
    """

    def __init__(  # noqa: PLR0913
        self,
        model_id: str = _QWEN3_OMNI_MODEL_ID,
        prompt_text: str = "Transcribe the audio.",
        followup_prompt: str = "Now listen to the audio again and add any false starts, filler words and preserve colloquial words (like lemme, gonna, wanna, etc) as is spoken in the audio.",
        system_prompt: str | None = None,
        max_model_len: int = 32768,
        max_num_seqs: int = 32,
        gpu_memory_utilization: float = 0.95,
        tensor_parallel_size: int | None = None,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 1,
        prep_workers: int = 8,
        fp8: bool = False,
        enforce_eager: bool = False,
        max_num_batched_tokens: int | None = None,
        mm_cache_gb: float = 4.0,
        disable_log_stats: bool = True,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        inference_chunk_size: int | None = None,
    ):
        self.model_id = model_id
        self.prompt_text = prompt_text
        self.followup_prompt = followup_prompt
        self.system_prompt = system_prompt
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.prep_workers = prep_workers
        self.fp8 = fp8
        self.enforce_eager = enforce_eager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.mm_cache_gb = mm_cache_gb
        self.disable_log_stats = disable_log_stats
        self.max_retries = max_retries
        self.inference_chunk_size = inference_chunk_size

        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None
        self._processor: Any = None
        self._prep_pool: ThreadPoolExecutor | None = None
        self._tp_size: int = 0

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_id]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_llm_kwargs(self) -> dict[str, Any]:
        """Centralised vLLM ``LLM(...)`` keyword dict — reused by setup and reset."""
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "trust_remote_code": True,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self._tp_size,
            "limit_mm_per_prompt": {"image": 1, "video": 1, "audio": 2},
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "enforce_eager": self.enforce_eager,
            "seed": 1234,
        }
        if self.fp8:
            kwargs["quantization"] = "fp8"
        if self.max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.mm_cache_gb >= 0:
            kwargs["mm_processor_cache_gb"] = self.mm_cache_gb
        if self.disable_log_stats:
            kwargs["disable_log_stats"] = True
        return kwargs

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vLLM is required for QwenOmni. Install it: pip install vllm"
            raise ImportError(msg)

        self._tp_size = self.tensor_parallel_size or get_gpu_count()

        logger.info(
            "Loading QwenOmni model=%s  tp=%d  max_model_len=%d  max_num_seqs=%d  fp8=%s  eager=%s",
            self.model_id,
            self._tp_size,
            self.max_model_len,
            self.max_num_seqs,
            self.fp8,
            self.enforce_eager,
        )

        # SECURITY: trust_remote_code=True is required by Qwen3-Omni's custom
        # modeling code on HuggingFace.  Changing model_id to an untrusted
        # repository will execute that repository's Python code in the worker.
        self._llm = LLM(**self._build_llm_kwargs())

        from transformers import Qwen3OmniMoeProcessor

        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id)

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            max_tokens=self.max_output_tokens,
        )

        self._prep_pool = ThreadPoolExecutor(max_workers=self.prep_workers)

        logger.info("QwenOmni model loaded")

    def _reset_engine(self) -> None:
        """Tear down and reinitialise the vLLM engine after a transient failure.

        Mirrors the ``_reset_vllm`` pattern from
        ``NemotronParseInferenceStage`` (interleaved modality).
        """
        logger.warning("Resetting vLLM engine after inference failure")
        try:
            del self._llm
            self._llm = None
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass
        self._llm = LLM(**self._build_llm_kwargs())

    def teardown(self) -> None:
        if self._prep_pool is not None:
            self._prep_pool.shutdown(wait=False)
            self._prep_pool = None
        del self._llm
        self._llm = None
        self._processor = None
        self._sampling_params = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int = _QWEN_SAMPLE_RATE) -> np.ndarray:
        """Resample waveform to target sample rate if needed."""
        if orig_sr == target_sr:
            return waveform
        import librosa

        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

    def _build_messages(self, waveform: np.ndarray) -> list[dict[str, Any]]:
        """Build Turn 1 chat messages with an in-memory waveform (numpy array at 16 kHz)."""
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_text},
                    {"type": "audio", "audio": waveform},
                ],
            }
        )
        return messages

    def _build_turn2_messages(self, waveform: np.ndarray, pred_text: str) -> list[dict[str, Any]]:
        """Build Turn 2 messages: full Turn 1 conversation history + follow-up prompt."""
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_text},
                    {"type": "audio", "audio": waveform},
                ],
            }
        )
        messages.append({"role": "assistant", "content": [{"type": "text", "text": pred_text}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.followup_prompt},
                ],
            }
        )
        return messages

    def _prepare_single(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> tuple[dict[str, Any], np.ndarray] | None:
        from qwen_omni_utils import process_mm_info

        try:
            waveform_16k = self._resample(waveform, sample_rate)
            messages = self._build_messages(waveform_16k)
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to preprocess audio, skipping (waveform shape=%s, sr=%d)", waveform.shape, sample_rate
            )
            return None

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
        return inputs, waveform_16k

    def _prepare_batch(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
    ) -> list[tuple[dict[str, Any], np.ndarray] | None]:
        if self._prep_pool is None:
            return [self._prepare_single(w, sr) for w, sr in zip(waveforms, sample_rates, strict=True)]
        return list(self._prep_pool.map(self._prepare_single, waveforms, sample_rates))

    def _prepare_turn2_single(
        self,
        waveform_16k: np.ndarray,
        pred_text: str,
    ) -> dict[str, Any] | None:
        from qwen_omni_utils import process_mm_info

        try:
            messages = self._build_turn2_messages(waveform_16k, pred_text)
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to preprocess Turn 2 audio (shape=%s)", waveform_16k.shape)
            return None

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

    def _prepare_turn2_batch(
        self,
        waveforms_16k: list[np.ndarray],
        pred_texts: list[str],
    ) -> list[dict[str, Any] | None]:
        if self._prep_pool is None:
            return [self._prepare_turn2_single(w, pt) for w, pt in zip(waveforms_16k, pred_texts, strict=True)]
        return list(self._prep_pool.map(self._prepare_turn2_single, waveforms_16k, pred_texts))

    # ------------------------------------------------------------------
    # Inference with retry (adapted from NemotronParse interleaved)
    # ------------------------------------------------------------------

    def _generate_with_retry(self, inputs: list[dict[str, Any]]) -> list[Any]:
        """Call ``_llm.generate`` with automatic engine-reset retry on failure.

        For large-scale pipelines (millions of hours), transient CUDA OOM or
        driver errors should not kill the entire job.  The interleaved
        ``NemotronParseInferenceStage`` proved this pattern essential.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._llm.generate(inputs, sampling_params=self._sampling_params, use_tqdm=False)
            except Exception as exc:  # noqa: PERF203
                logger.warning("vLLM inference failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    self._reset_engine()
                else:
                    raise
        return []  # unreachable but satisfies type checker

    def _chunked_generate(self, inputs: list[dict[str, Any]]) -> list[Any]:
        """Optionally chunk large batches to avoid vLLM OOM.

        Mirrors the ``split_by_chunk_size`` pattern from video
        ``QwenVL.generate``.  When ``inference_chunk_size`` is ``None``
        (the default), all inputs are sent in a single call — vLLM's own
        continuous batching handles scheduling.
        """
        if self.inference_chunk_size is None or len(inputs) <= self.inference_chunk_size:
            return self._generate_with_retry(inputs)

        all_outputs: list[Any] = []
        for start in range(0, len(inputs), self.inference_chunk_size):
            chunk = inputs[start : start + self.inference_chunk_size]
            all_outputs.extend(self._generate_with_retry(chunk))
        return all_outputs

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
    ) -> tuple[list[str], list[str], dict[str, float]]:
        """Run batched two-turn inference on in-memory audio waveforms.

        Turn 1 transcribes using ``prompt_text``.  If ``followup_prompt``
        is set, Turn 2 re-listens with the full conversation history and
        a follow-up prompt (e.g. to add disfluencies / filler words).

        Args:
            waveforms: List of 1-D mono numpy float32 arrays.
            sample_rates: Corresponding sample rates for each waveform.

        Returns:
            ``(pred_texts, disfluency_texts, metrics)`` — one string per
            input for each turn, plus a timing-metrics dict.
            ``disfluency_texts`` is all empty strings when
            ``followup_prompt`` is ``None``.
        """
        if self._llm is None or self._sampling_params is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        n = len(waveforms)
        metrics: dict[str, float] = {"batch_size": float(n)}

        # -- Turn 1: preprocess -----------------------------------------------
        t0 = time.perf_counter()
        prepared = self._prepare_batch(waveforms, sample_rates)
        metrics["t1_prep_s"] = time.perf_counter() - t0

        valid_indices = [i for i, p in enumerate(prepared) if p is not None]
        valid_inputs = [prepared[i][0] for i in valid_indices]
        waveforms_16k: dict[int, np.ndarray] = {i: prepared[i][1] for i in valid_indices}

        del prepared

        if not valid_inputs:
            logger.warning("All %d audio samples in batch failed preprocessing", n)
            return [""] * n, [""] * n, metrics

        if len(valid_inputs) < n:
            logger.warning("Skipped %d/%d corrupt audio samples", n - len(valid_inputs), n)
        metrics["t1_valid_count"] = float(len(valid_inputs))

        # -- Turn 1: inference ------------------------------------------------
        t0 = time.perf_counter()
        t1_outputs = self._chunked_generate(valid_inputs)
        metrics["t1_inference_s"] = time.perf_counter() - t0

        del valid_inputs

        pred_texts: list[str] = [""] * n
        for idx, out in zip(valid_indices, t1_outputs, strict=True):
            pred_texts[idx] = out.outputs[0].text.strip()

        del t1_outputs

        # -- Turn 2 (disfluency refinement) -----------------------------------
        if not self.followup_prompt:
            return pred_texts, [""] * n, metrics

        t2_indices = [i for i in valid_indices if pred_texts[i]]
        if not t2_indices:
            return pred_texts, [""] * n, metrics

        t0 = time.perf_counter()
        t2_prepared = self._prepare_turn2_batch(
            [waveforms_16k[i] for i in t2_indices],
            [pred_texts[i] for i in t2_indices],
        )
        metrics["t2_prep_s"] = time.perf_counter() - t0

        del waveforms_16k

        t2_valid = [(i, p) for i, p in zip(t2_indices, t2_prepared, strict=True) if p is not None]
        del t2_prepared

        if not t2_valid:
            logger.warning("All Turn 2 samples failed preprocessing")
            return pred_texts, [""] * n, metrics

        metrics["t2_valid_count"] = float(len(t2_valid))

        t0 = time.perf_counter()
        t2_outputs = self._chunked_generate([p for _, p in t2_valid])
        metrics["t2_inference_s"] = time.perf_counter() - t0

        disfluency_texts: list[str] = [""] * n
        for (idx, _), out in zip(t2_valid, t2_outputs, strict=True):
            disfluency_texts[idx] = out.outputs[0].text.strip()

        return pred_texts, disfluency_texts, metrics
