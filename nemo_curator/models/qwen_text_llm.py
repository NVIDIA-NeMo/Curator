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
from typing import Any

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


_QWEN_TEXT_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"


class QwenTextLLM(ModelInterface):
    """Text-only Qwen LLM via vLLM for two-step PnC restoration.

    Step 1 – Completeness check: asks the model whether a given text
    is a complete sentence.  If the answer is "no", the original text
    is returned unchanged.

    Step 2 – PnC restoration: sends the text with a user-supplied
    prompt that instructs the model to restore punctuation and
    capitalisation.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_id: str = _QWEN_TEXT_MODEL_ID,
        completeness_prompt: str = (
            "Is the following text a complete sentence? Answer only 'yes' or 'no'.\n\nText: {text}"
        ),
        pnc_prompt: str = (
            "Restore proper punctuation and capitalization to the following text. "
            "Output only the corrected text, nothing else.\n\nText: {text}"
        ),
        system_prompt: str | None = None,
        max_model_len: int = 4096,
        max_num_seqs: int = 16,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = 1,
    ):
        self.model_id = model_id
        self.completeness_prompt = completeness_prompt
        self.pnc_prompt = pnc_prompt
        self.system_prompt = system_prompt
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_k = top_k

        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None
        self._short_sampling_params: SamplingParams | None = None

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_id]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vLLM is required for QwenTextLLM. Install it: pip install vllm"
            raise ImportError(msg)

        tp_size = self.tensor_parallel_size or get_gpu_count()

        logger.info(
            "Loading QwenTextLLM model={}  tp={}  max_model_len={}  max_num_seqs={}",
            self.model_id, tp_size, self.max_model_len, self.max_num_seqs,
        )

        self._llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp_size,
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            seed=1234,
        )

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            max_tokens=self.max_output_tokens,
        )

        self._short_sampling_params = SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=8,
        )

        logger.info("QwenTextLLM model loaded")

    def teardown(self) -> None:
        del self._llm
        self._llm = None
        self._sampling_params = None
        self._short_sampling_params = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_chat(self, user_text: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    @staticmethod
    def _is_yes(answer: str) -> bool:
        return answer.strip().lower().startswith("yes")

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _format_prompt(self, user_text: str) -> str:
        """Format a user message into a chat prompt string via tokenizer.

        Disables Qwen3's "thinking" mode so the model answers directly.
        """
        messages = self._build_chat(user_text)
        try:
            tokenizer = self._llm.get_tokenizer()
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except Exception:
            return user_text

    def _prepare_single(self, text: str, prompt_template: str) -> dict[str, Any] | None:
        """Build a single vLLM input dict, returning None on failure."""
        try:
            user_text = prompt_template.format(text=text)
            return {"prompt": self._format_prompt(user_text)}
        except Exception:
            logger.warning("Failed to prepare prompt for text (len={}), skipping", len(text))
            return None

    def _prepare_batch(
        self,
        texts: list[str],
        indices: list[int],
        prompt_template: str,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        """Prepare a batch, filtering out items that fail preprocessing.

        Returns:
            ``(valid_indices, valid_inputs)`` — parallel lists of the
            original indices and their corresponding vLLM input dicts.
        """
        valid_indices: list[int] = []
        valid_inputs: list[dict[str, Any]] = []
        for i in indices:
            inp = self._prepare_single(texts[i], prompt_template)
            if inp is not None:
                valid_indices.append(i)
                valid_inputs.append(inp)

        if len(valid_inputs) < len(indices):
            logger.warning(
                "Skipped {}/{} texts that failed preprocessing",
                len(indices) - len(valid_inputs), len(indices),
            )
        return valid_indices, valid_inputs

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        texts: list[str],
    ) -> tuple[list[bool], list[str]]:
        """Run batched two-step PnC restoration on text inputs.

        Step 1: completeness check — ask if each text is a complete sentence.
        Step 2: PnC restoration — for texts that *are* complete, ask the model
                 to restore punctuation and capitalisation.

        Args:
            texts: List of cleaned text strings.

        Returns:
            ``(is_complete, pnc_texts)`` where:
            - ``is_complete[i]`` is True if text *i* was deemed a complete sentence.
            - ``pnc_texts[i]`` contains the PnC-restored text when complete,
              or the original ``texts[i]`` when incomplete / empty.
        """
        if self._llm is None or self._sampling_params is None or self._short_sampling_params is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        n = len(texts)
        is_complete: list[bool] = [False] * n
        pnc_texts: list[str] = list(texts)

        # -- Step 1: completeness check ------------------------------------
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not non_empty_indices:
            logger.warning("All {} texts in batch are empty", n)
            return is_complete, pnc_texts

        s1_valid_indices, s1_valid_inputs = self._prepare_batch(
            texts, non_empty_indices, self.completeness_prompt,
        )

        if not s1_valid_inputs:
            logger.warning("All {} texts in batch failed Step 1 preprocessing", n)
            return is_complete, pnc_texts

        step1_outputs = self._llm.generate(
            s1_valid_inputs, sampling_params=self._short_sampling_params, use_tqdm=False,
        )

        complete_indices: list[int] = []
        sample_answers: list[str] = []
        for idx, out in zip(s1_valid_indices, step1_outputs):
            answer = out.outputs[0].text.strip()
            if len(sample_answers) < 5:
                sample_answers.append(repr(answer))
            if self._is_yes(answer):
                is_complete[idx] = True
                complete_indices.append(idx)

        logger.info(
            "PnC completeness check: {}/{} complete, {}/{} incomplete, {} empty",
            len(complete_indices),
            len(s1_valid_indices),
            len(s1_valid_indices) - len(complete_indices),
            len(s1_valid_indices),
            n - len(non_empty_indices),
        )

        # -- Step 2: PnC restoration for complete sentences ----------------
        if not complete_indices:
            return is_complete, pnc_texts

        s2_valid_indices, s2_valid_inputs = self._prepare_batch(
            texts, complete_indices, self.pnc_prompt,
        )

        if not s2_valid_inputs:
            logger.warning("All Step 2 texts failed preprocessing")
            return is_complete, pnc_texts

        step2_outputs = self._llm.generate(
            s2_valid_inputs, sampling_params=self._sampling_params, use_tqdm=False,
        )

        for idx, out in zip(s2_valid_indices, step2_outputs):
            restored = out.outputs[0].text.strip()
            if restored:
                pnc_texts[idx] = restored

        logger.info("PnC restoration: restored {} texts", len(s2_valid_indices))
        return is_complete, pnc_texts
