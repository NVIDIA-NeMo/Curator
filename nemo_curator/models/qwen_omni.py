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
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.models.base import ModelInterface
from nemo_curator.utils.gpu_utils import get_gpu_count

if TYPE_CHECKING:
    import numpy as np

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


class QwenOmni(ModelInterface):
    """Qwen3-Omni multimodal model via vLLM for audio-conditioned text generation.

    Uses the *thinker-only* path: audio waveforms go in, text comes out.

    Audio is accepted as in-memory numpy arrays (mono, any sample rate).
    Arrays are resampled to 16 kHz before being passed to
    ``qwen_omni_utils.process_mm_info`` which natively accepts
    ``np.ndarray`` inputs — no temporary files are created.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_id: str = _QWEN3_OMNI_MODEL_ID,
        prompt_text: str = "Transcribe the audio.",
        en_prompt_text: str | None = None,
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
    ):
        self.model_id = model_id
        self.prompt_text = prompt_text
        self.en_prompt_text = en_prompt_text
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

        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None
        self._processor: Any = None
        self._prep_pool: ThreadPoolExecutor | None = None

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_id]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vLLM is required for QwenOmni. Install it: pip install vllm"
            raise ImportError(msg)

        tp_size = self.tensor_parallel_size or get_gpu_count()

        logger.info(
            f"Loading QwenOmni model={self.model_id}  tp={tp_size}  max_model_len={self.max_model_len}  max_num_seqs={self.max_num_seqs}"
        )

        self._llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp_size,
            limit_mm_per_prompt={"image": 1, "video": 1, "audio": 2},
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            seed=1234,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="xxhash",
        )

        from transformers import Qwen3OmniMoeProcessor

        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id)

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            max_tokens=self.max_output_tokens,
        )

        self._prep_pool = ThreadPoolExecutor(max_workers=self.prep_workers)

        logger.info("QwenOmni model loaded")

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
        except Exception as e:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: {}", e)

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

    def _resolve_prompt(self, template: str, language: str | None) -> str:
        """Replace ``{language}`` placeholder if *language* is provided."""
        if language and "{language}" in template:
            return template.replace("{language}", language)
        return template

    def _get_prompt_text(self, language: str | None) -> str:
        """Return the EN-specific prompt for English, otherwise the default prompt."""
        if language and language == "English" and self.en_prompt_text:
            return self.en_prompt_text
        return self._resolve_prompt(self.prompt_text, language)

    def _build_messages(self, waveform: np.ndarray, language: str | None = None) -> list[dict[str, Any]]:
        """Build Turn 1 chat messages with an in-memory waveform (numpy array at 16 kHz)."""
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

    def _build_turn2_messages(self, waveform: np.ndarray, pred_text: str, language: str | None = None) -> list[dict[str, Any]]:
        """Build Turn 2 messages: full Turn 1 conversation history + follow-up prompt."""
        prompt = self._get_prompt_text(language)
        followup = self._resolve_prompt(self.followup_prompt, language)
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
        from qwen_omni_utils import process_mm_info

        if waveform is None or waveform.size == 0:
            logger.warning("Skipping empty waveform")
            return None

        try:
            waveform_16k = self._resample(waveform, sample_rate)
            messages = self._build_messages(waveform_16k, language)
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        except Exception:  # noqa: BLE001
            logger.warning(f"Failed to preprocess audio, skipping (waveform shape={waveform.shape}, sr={sample_rate})")
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
            return [self._prepare_single(w, sr, lang) for w, sr, lang in zip(waveforms, sample_rates, langs, strict=False)]
        return list(self._prep_pool.map(self._prepare_single, waveforms, sample_rates, langs))

    def _prepare_turn2_single(
        self, waveform_16k: np.ndarray, pred_text: str, language: str | None = None,
    ) -> dict[str, Any] | None:
        from qwen_omni_utils import process_mm_info

        try:
            messages = self._build_turn2_messages(waveform_16k, pred_text, language)
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        except Exception:  # noqa: BLE001
            logger.warning(f"Failed to preprocess Turn 2 audio (shape={waveform_16k.shape})")
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

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        languages: list[str | None] | None = None,
    ) -> tuple[list[str], list[str], set[int]]:
        """Run batched two-turn inference on in-memory audio waveforms.

        Turn 1 transcribes using ``prompt_text``.  If ``followup_prompt``
        is set, Turn 2 re-listens with the full conversation history and
        a follow-up prompt (e.g. to add disfluencies / filler words).

        Prompts may contain a ``{language}`` placeholder which is replaced
        per-sample with the corresponding entry from *languages* (e.g.
        ``"English"``).  When *languages* is ``None`` or an entry is
        ``None``, prompts are used as-is.

        Args:
            waveforms: List of 1-D mono numpy float32 arrays.
            sample_rates: Corresponding sample rates for each waveform.
            languages: Optional per-sample language names for prompt
                interpolation.

        Returns:
            ``(pred_texts, disfluency_texts, skipped_indices)`` — one string
            per input for each turn, plus a set of indices that were skipped
            due to empty/corrupt audio.  ``disfluency_texts`` is all empty
            strings when ``followup_prompt`` is ``None``.
        """
        if self._llm is None or self._sampling_params is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        n = len(waveforms)

        # -- Turn 1 ----------------------------------------------------------
        prepared = self._prepare_batch(waveforms, sample_rates, languages)
        valid_indices = [i for i, p in enumerate(prepared) if p is not None]
        valid_inputs = [prepared[i][0] for i in valid_indices]
        waveforms_16k: dict[int, np.ndarray] = {i: prepared[i][1] for i in valid_indices}
        skipped_indices = set(range(n)) - set(valid_indices)

        if not valid_inputs:
            logger.warning(f"All {n} audio samples in batch failed preprocessing")
            return [""] * n, [""] * n, skipped_indices

        if len(valid_inputs) < n:
            logger.warning(f"Skipped {n - len(valid_inputs)}/{n} corrupt audio samples")

        t1_outputs = self._llm.generate(valid_inputs, sampling_params=self._sampling_params, use_tqdm=False)

        pred_texts: list[str] = [""] * n
        for idx, out in zip(valid_indices, t1_outputs, strict=False):
            pred_texts[idx] = out.outputs[0].text.strip()

        # -- Turn 2 (disfluency refinement) -----------------------------------
        if not self.followup_prompt:
            return pred_texts, [""] * n, skipped_indices

        t2_indices = [i for i in valid_indices if pred_texts[i]]
        if not t2_indices:
            return pred_texts, [""] * n, skipped_indices

        langs = languages or [None] * n
        t2_prepared = self._prepare_turn2_batch(
            [waveforms_16k[i] for i in t2_indices],
            [pred_texts[i] for i in t2_indices],
            [langs[i] for i in t2_indices],
        )

        t2_valid = [(i, p) for i, p in zip(t2_indices, t2_prepared, strict=False) if p is not None]
        if not t2_valid:
            logger.warning("All Turn 2 samples failed preprocessing")
            return pred_texts, [""] * n, skipped_indices

        t2_outputs = self._llm.generate(
            [p for _, p in t2_valid], sampling_params=self._sampling_params, use_tqdm=False,
        )

        disfluency_texts: list[str] = [""] * n
        for (idx, _), out in zip(t2_valid, t2_outputs, strict=False):
            disfluency_texts[idx] = out.outputs[0].text.strip()

        return pred_texts, disfluency_texts, skipped_indices
