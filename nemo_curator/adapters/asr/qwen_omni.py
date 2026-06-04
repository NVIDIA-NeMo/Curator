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

Implements the :class:`~nemo_curator.adapters.asr.ASRAdapter` protocol on
top of the in-process vLLM thinker-only path. Two-turn output (Turn-1
transcribe, Turn-2 disfluency/refinement) is supported when
``followup_prompt`` is set; single-turn otherwise.

Core inference logic (vLLM engine setup, ``qwen_omni_utils.process_mm_info``
preprocessing, Turn-1/Turn-2 prompt construction, batched preprocessing
thread pool) is preserved verbatim from the pre-split ``QwenOmni`` model
wrapper - this module only re-houses that code inside the adapter
protocol and adds ``prefetch_weights`` + ``transcribe_batch`` thin
wrappers around the existing ``generate`` flow.
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import snapshot_download
from loguru import logger

from nemo_curator.adapters.asr.base import ASRResult
from nemo_curator.utils.gpu_utils import get_gpu_count

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
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:  # type: ignore[no-redef]
        pass

    class SamplingParams:  # type: ignore[no-redef]
        pass


def _require_audio_cuda12_stack(*, context: str) -> None:
    """Raise a single ImportError listing missing audio_cuda12-only deps."""
    missing: list[str] = []
    if not VLLM_AVAILABLE:
        missing.append("vllm")
    if process_mm_info is None:
        missing.append("qwen-omni-utils")
    if Qwen3OmniMoeProcessor is None:
        missing.append("transformers (Qwen3OmniMoeProcessor)")
    if missing:
        msg = (
            f"QwenOmniASRAdapter {context} requires the audio_cuda12 extra. "
            f"Missing: {', '.join(missing)}. Install with: uv sync --extra audio_cuda12"
        )
        raise ImportError(msg)


_QWEN3_OMNI_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
_QWEN_SAMPLE_RATE = 16000
_FOLLOWUP_PROMPT_DEFAULT = (
    "Now listen to the audio again and add any false starts, filler words "
    "and preserve colloquial words (like lemme, gonna, wanna, etc) as is "
    "spoken in the audio."
)


@dataclass
class QwenOmniASRAdapter:
    """Qwen3-Omni in-process vLLM adapter (thinker-only path).

    Stages instantiate adapters as
    via ``cls(model_id=..., revision=..., **adapter_kwargs)``, so every
    field below is a keyword-only knob the YAML's ``adapter_kwargs`` block
    can set.

    Resource expectations:
        * ~40 GB VRAM for Qwen3-Omni-30B-A3B (FP8). One A100-80GB or two
          A100-40GB with ``tensor_parallel_size=2``.
        * ~50-80 audio-seconds/GPU-second on A100-80GB with
          ``batch_size=32`` and ``max_num_seqs=32``.
        * ~15 GB cached weights on first run (HuggingFace Hub).

    Args:
        model_id: HuggingFace model identifier. Defaults to the published
            Qwen3-Omni-30B-A3B Instruct checkpoint.
        revision: Optional HuggingFace revision to pin.
        prompt_text: User prompt sent alongside the audio for Turn-1.
            ``{language}`` is interpolated per-item from the stage-supplied
            language name.
        prompt_file: Optional path to a UTF-8 text file whose contents
            replace ``prompt_text`` when present. Resolved at
            ``__post_init__`` time.
        en_prompt_text / en_prompt_file: English-specific prompt override
            (used when the per-item language resolves to ``"English"``).
        followup_prompt / followup_prompt_file: When set, enables Turn-2
            inference. ``{language}`` is interpolated.
        system_prompt / system_prompt_file: Optional system message
            prepended to both turns.
        max_model_len: vLLM context length.
        max_num_seqs: vLLM max concurrent sequences.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        tensor_parallel_size: TP world size. ``None`` -> auto-detect from
            visible GPUs on the worker.
        max_output_tokens: Max generated tokens per turn.
        temperature: Sampling temperature (0.0 = greedy).
        top_k: Top-k sampling.
        prep_workers: Thread-pool size for parallel audio preprocessing.
        enable_prefix_caching: vLLM prefix-cache toggle. Default ``True``
            because system / user / follow-up prompts repeat across requests;
            disable for deployments with highly variable prompts.
        prefix_caching_hash_algo: Backing hash algorithm for the prefix
            cache. ``"xxhash"`` matches the doc default; vLLM also accepts
            ``"sha256"``.
        limit_mm_per_prompt_audio: Per-prompt audio-token cap for vLLM's
            multi-modal limiter. Default ``2`` matches the doc and the
            default (enough for the two-turn flow).
            Set to ``1`` for strictly single-turn deployments.
        seed: vLLM scheduler / sampling seed. Exposed so reproducibility
            tests (and follow-up bit-exactness checks) can override it
            from YAML.
    """

    # Universal adapter constructor fields (forwarded by ASRStage).
    model_id: str = _QWEN3_OMNI_MODEL_ID
    revision: str | None = None

    # Qwen-Omni-specific knobs (flow in via adapter_kwargs).
    prompt_text: str = "Transcribe the audio."
    prompt_file: str | None = None
    en_prompt_text: str | None = None
    en_prompt_file: str | None = None
    followup_prompt: str | None = None
    followup_prompt_file: str | None = None
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    max_model_len: int = 32768
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.95
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 1
    prep_workers: int = 8

    # vLLM knobs (set via adapter_kwargs in YAML).
    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "xxhash"
    limit_mm_per_prompt_audio: int = 2
    seed: int = 1234

    # Per-batch state - reset by transcribe_batch, surfaced to the stage
    # for _log_metrics merging.
    last_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Resolve any file-backed prompts into their text form once at
        # construction so the per-batch path stays cheap.
        self.prompt_text = self._load_text(self.prompt_text, self.prompt_file) or ""
        self.en_prompt_text = self._load_text(self.en_prompt_text, self.en_prompt_file)
        self.followup_prompt = self._load_text(self.followup_prompt, self.followup_prompt_file)
        self.system_prompt = self._load_text(self.system_prompt, self.system_prompt_file)

        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None
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

    # ------------------------------------------------------------------
    # ASRAdapter protocol surface
    # ------------------------------------------------------------------

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
        _require_audio_cuda12_stack(context="setup()")

        setup_t0 = time.perf_counter()
        tp_size = self.tensor_parallel_size or get_gpu_count()

        logger.info(
            f"Loading QwenOmni model={self.model_id}  tp={tp_size}  "
            f"max_model_len={self.max_model_len}  max_num_seqs={self.max_num_seqs}"
            + (f"  revision={self.revision}" if self.revision is not None else "")
        )

        llm_kwargs: dict[str, Any] = {}
        if self.revision is not None:
            llm_kwargs["revision"] = self.revision
        self._llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp_size,
            limit_mm_per_prompt={"image": 1, "video": 1, "audio": int(self.limit_mm_per_prompt_audio)},
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            seed=int(self.seed),
            enable_prefix_caching=bool(self.enable_prefix_caching),
            prefix_caching_hash_algo=str(self.prefix_caching_hash_algo),
            **llm_kwargs,
        )

        proc_kwargs: dict[str, Any] = {}
        if self.revision is not None:
            proc_kwargs["revision"] = self.revision
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id, **proc_kwargs)

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            max_tokens=self.max_output_tokens,
        )

        self._prep_pool = ThreadPoolExecutor(max_workers=self.prep_workers)

        logger.info("QwenOmni model loaded in {:.3f}s", time.perf_counter() - setup_t0)

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
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: {}", e)

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Run batched Qwen-Omni inference over a batch of per-task dicts.

        Unpacks the per-task dicts the stage assembles, dispatches into
        the underlying two-turn generation, and packs each output into
        an :class:`ASRResult`. Skipped items (empty / unprocessable
        waveforms) round-trip as ``ASRResult(text="", skipped=True)``.
        """
        if not items:
            return []
        waveforms = [it["waveform"] for it in items]
        sample_rates = [it["sample_rate"] for it in items]
        languages = [it.get("language") for it in items]
        pred_texts, disfl_texts, skipped_indices = self._generate(waveforms, sample_rates, languages)
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

    # ------------------------------------------------------------------
    # Input preparation - logic preserved from pre-split QwenOmni
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int = _QWEN_SAMPLE_RATE) -> np.ndarray:
        if orig_sr == target_sr:
            return waveform
        import librosa

        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

    def _resolve_prompt(self, template: str, language: str | None) -> str:
        if language and "{language}" in template:
            return template.replace("{language}", language)
        return template

    def _get_prompt_text(self, language: str | None) -> str:
        if language == "English" and self.en_prompt_text:
            return self.en_prompt_text
        return self._resolve_prompt(self.prompt_text, language)

    def _build_messages(self, waveform: np.ndarray, language: str | None = None) -> list[dict[str, Any]]:
        prompt = self._get_prompt_text(language)
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            sys_prompt = self._resolve_prompt(self.system_prompt, language)
            messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": waveform},
            ],
        })
        return messages

    def _build_turn2_messages(
        self, waveform: np.ndarray, pred_text: str, language: str | None = None,
    ) -> list[dict[str, Any]]:
        prompt = self._get_prompt_text(language)
        followup = self._resolve_prompt(self.followup_prompt or _FOLLOWUP_PROMPT_DEFAULT, language)
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            sys_prompt = self._resolve_prompt(self.system_prompt, language)
            messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": waveform},
            ],
        })
        messages.append({"role": "assistant", "content": [{"type": "text", "text": pred_text}]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": followup},
            ],
        })
        return messages

    def _prepare_single(
        self, waveform: np.ndarray, sample_rate: int, language: str | None = None,
    ) -> tuple[dict[str, Any], np.ndarray] | None:
        if waveform is None or waveform.size == 0:
            logger.warning("Skipping empty waveform")
            return None

        try:
            waveform_16k = self._resample(waveform, sample_rate)
            messages = self._build_messages(waveform_16k, language)
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to preprocess audio, skipping (waveform shape={}, sr={}): {}",
                getattr(waveform, "shape", None),
                sample_rate,
                exc,
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
        languages: list[str | None] | None = None,
    ) -> list[tuple[dict[str, Any], np.ndarray] | None]:
        langs = languages or [None] * len(waveforms)
        if self._prep_pool is None:
            return [
                self._prepare_single(w, sr, lang)
                for w, sr, lang in zip(waveforms, sample_rates, langs, strict=False)
            ]
        return list(self._prep_pool.map(self._prepare_single, waveforms, sample_rates, langs))

    def _prepare_turn2_single(
        self, waveform_16k: np.ndarray, pred_text: str, language: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            messages = self._build_turn2_messages(waveform_16k, pred_text, language)
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to preprocess Turn 2 audio (shape={}): {}",
                getattr(waveform_16k, "shape", None),
                exc,
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
        return inputs

    def _prepare_turn2_batch(
        self,
        waveforms_16k: list[np.ndarray],
        pred_texts: list[str],
        languages: list[str | None] | None = None,
    ) -> list[dict[str, Any] | None]:
        langs = languages or [None] * len(waveforms_16k)
        if self._prep_pool is None:
            return [
                self._prepare_turn2_single(w, pt, lang)
                for w, pt, lang in zip(waveforms_16k, pred_texts, langs, strict=False)
            ]
        return list(self._prep_pool.map(self._prepare_turn2_single, waveforms_16k, pred_texts, langs))

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
    def _first_output_text(output: Any) -> str:
        sequences = getattr(output, "outputs", None) or []
        if not sequences:
            return ""
        return (getattr(sequences[0], "text", "") or "").strip()

    # ------------------------------------------------------------------
    # Two-turn generation - logic preserved from pre-split QwenOmni
    # ------------------------------------------------------------------

    def _generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        languages: list[str | None] | None = None,
    ) -> tuple[list[str], list[str], set[int]]:
        """Run batched two-turn inference on in-memory waveforms.

        Returns ``(pred_texts, disfluency_texts, skipped_indices)``.
        ``disfluency_texts`` is all empty strings when ``followup_prompt``
        is not set.
        """
        if self._llm is None or self._sampling_params is None:
            msg = "Adapter not initialized. Call setup() first."
            raise RuntimeError(msg)

        n = len(waveforms)
        metrics: dict[str, float] = {
            "utterances_input": float(n),
            "audio_duration_s": sum(
                float(w.shape[0]) / float(sr)
                for w, sr in zip(waveforms, sample_rates, strict=False)
                if sr and w is not None and getattr(w, "size", 0) > 0
            ),
            "waveform_bytes": sum(
                float(getattr(w, "nbytes", 0))
                for w in waveforms
                if w is not None
            ),
            "turn1_prep_time_s": 0.0,
            "turn1_generation_time_s": 0.0,
            "turn2_prep_time_s": 0.0,
            "turn2_generation_time_s": 0.0,
            "turn1_valid_inputs": 0.0,
            "turn2_valid_inputs": 0.0,
            "utterances_skipped_preprocess": 0.0,
            "output_tokens": 0.0,
            "turn1_output_tokens": 0.0,
            "turn2_output_tokens": 0.0,
        }
        self.last_metrics = metrics

        # -- Turn 1 ----------------------------------------------------------
        prep_t0 = time.perf_counter()
        prepared = self._prepare_batch(waveforms, sample_rates, languages)
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

        t1_t0 = time.perf_counter()
        t1_outputs = self._llm.generate(valid_inputs, sampling_params=self._sampling_params, use_tqdm=False)
        metrics["turn1_generation_time_s"] = time.perf_counter() - t1_t0
        metrics["turn1_output_tokens"] = self._count_output_tokens(t1_outputs)
        metrics["output_tokens"] += metrics["turn1_output_tokens"]

        pred_texts: list[str] = [""] * n
        for idx, out in zip(valid_indices, t1_outputs, strict=False):
            pred_texts[idx] = self._first_output_text(out)

        # -- Turn 2 (disfluency refinement) -----------------------------------
        if not self.followup_prompt:
            return pred_texts, [""] * n, skipped_indices

        t2_indices = [i for i in valid_indices if pred_texts[i]]
        if not t2_indices:
            return pred_texts, [""] * n, skipped_indices

        langs = languages or [None] * n
        t2_prep_t0 = time.perf_counter()
        t2_prepared = self._prepare_turn2_batch(
            [waveforms_16k[i] for i in t2_indices],
            [pred_texts[i] for i in t2_indices],
            [langs[i] for i in t2_indices],
        )
        metrics["turn2_prep_time_s"] = time.perf_counter() - t2_prep_t0

        t2_valid = [(i, p) for i, p in zip(t2_indices, t2_prepared, strict=False) if p is not None]
        metrics["turn2_valid_inputs"] = float(len(t2_valid))
        if not t2_valid:
            logger.warning("All Turn 2 samples failed preprocessing")
            return pred_texts, [""] * n, skipped_indices

        t2_t0 = time.perf_counter()
        t2_outputs = self._llm.generate(
            [p for _, p in t2_valid], sampling_params=self._sampling_params, use_tqdm=False,
        )
        metrics["turn2_generation_time_s"] = time.perf_counter() - t2_t0
        metrics["turn2_output_tokens"] = self._count_output_tokens(t2_outputs)
        metrics["output_tokens"] += metrics["turn2_output_tokens"]

        disfluency_texts: list[str] = [""] * n
        for (idx, _), out in zip(t2_valid, t2_outputs, strict=False):
            disfluency_texts[idx] = self._first_output_text(out)

        return pred_texts, disfluency_texts, skipped_indices
