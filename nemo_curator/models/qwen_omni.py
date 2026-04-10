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
from typing import Any

import numpy as np
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
            "Loading QwenOmni model=%s  tp=%d  max_model_len=%d  max_num_seqs=%d",
            self.model_id, tp_size, self.max_model_len, self.max_num_seqs,
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
        except Exception:  # noqa: BLE001
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
        """Build chat messages with an in-memory waveform (numpy array at 16 kHz)."""
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt_text},
                {"type": "audio", "audio": waveform},
            ],
        })
        return messages

    def _prepare_single(self, waveform: np.ndarray, sample_rate: int) -> dict[str, Any] | None:
        from qwen_omni_utils import process_mm_info

        try:
            waveform_16k = self._resample(waveform, sample_rate)
            messages = self._build_messages(waveform_16k)
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        except Exception:
            logger.warning("Failed to preprocess audio, skipping (waveform shape=%s, sr=%d)", waveform.shape, sample_rate)
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

    def _prepare_batch(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
    ) -> list[dict[str, Any] | None]:
        if self._prep_pool is None:
            return [self._prepare_single(w, sr) for w, sr in zip(waveforms, sample_rates)]
        return list(self._prep_pool.map(self._prepare_single, waveforms, sample_rates))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
    ) -> list[str]:
        """Run batched inference on in-memory audio waveforms.

        Args:
            waveforms: List of 1-D mono numpy float32 arrays.
            sample_rates: Corresponding sample rates for each waveform.

        Returns:
            One predicted text string per input.  Entries that fail
            preprocessing get an empty string.
        """
        if self._llm is None or self._sampling_params is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        prepared = self._prepare_batch(waveforms, sample_rates)

        valid_indices = [i for i, p in enumerate(prepared) if p is not None]
        valid_inputs = [prepared[i] for i in valid_indices]

        if not valid_inputs:
            logger.warning("All %d audio samples in batch failed preprocessing", len(waveforms))
            return [""] * len(waveforms)

        if len(valid_inputs) < len(waveforms):
            logger.warning("Skipped %d/%d corrupt audio samples", len(waveforms) - len(valid_inputs), len(waveforms))

        outputs = self._llm.generate(valid_inputs, sampling_params=self._sampling_params, use_tqdm=False)

        results = [""] * len(waveforms)
        for idx, out in zip(valid_indices, outputs):
            results[idx] = out.outputs[0].text.strip()
        return results
