# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import Any

import torch
from loguru import logger
from transformers import AutoConfig

from nemo_curator.models.base import ModelInterface

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:
        pass

    class SamplingParams:
        pass


def _get_max_model_len_from_config(model: str) -> int | None:
    """
    Try to get max model length from HuggingFace AutoConfig.

    Args:
        model: Model identifier (e.g., "microsoft/phi-4")

    Returns:
        Max model length if found, None otherwise.
    """
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    max_len = (
        getattr(config, "max_position_embeddings", None)
        or getattr(config, "n_positions", None)
        or getattr(config, "max_sequence_length", None)
    )
    if max_len:
        logger.info(f"Auto-detected max_model_len={max_len} for {model}")

    return max_len


def _get_gpu_count() -> int:
    """
    Get number of available CUDA GPUs as a power of 2.

    Many models require tensor parallelism to use power-of-2 GPU counts.
    This returns the largest power of 2 <= available GPU count.

    Returns:
        Power of 2 GPU count, minimum 1.
    """
    count = torch.cuda.device_count()
    tp_size = 2 ** int(math.log2(count)) if count >= 2 else 1  # noqa: PLR2004
    logger.info(
        f"Detected {count} GPU(s), using tensor_parallel_size={tp_size}"
    )
    return tp_size


class VLLMModel(ModelInterface):
    """Generic vLLM language model wrapper for text generation."""

    def __init__(  # noqa: PLR0913
        self,
        model: str,
        max_model_len: int | None = None,
        tensor_parallel_size: int | None = None,
        max_num_batched_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int | None = None,
        cache_dir: str | None = None,
    ):
        """
        Initialize the vLLM model wrapper.

        Args:
            model: Model identifier (e.g., "microsoft/phi-4")
            max_model_len: Maximum model context length. If not specified,
                will be auto-detected from HuggingFace AutoConfig.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
                If not specified, auto-detects available GPUs.
            max_num_batched_tokens: Maximum tokens per batch. Defaults to
                4096.
            temperature: Sampling temperature. Defaults to 0.7.
            top_p: Top-p sampling parameter. Defaults to 0.8.
            top_k: Top-k sampling parameter. Defaults to 20.
            min_p: Min-p sampling parameter (for Qwen3). Defaults to 0.0.
            max_tokens: Maximum tokens to generate. Defaults to None.
            cache_dir: Cache directory for model weights. Defaults to None.
        """
        self.model = model
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None
        self._final_max_model_len: int | None = None
        self._is_qwen3: bool = False

    def model_id_names(self) -> list[str]:
        """Return the model identifier."""
        return [self.model]

    def setup(self) -> None:
        """Set up the vLLM model and sampling parameters."""
        if not VLLM_AVAILABLE:
            msg = (
                "vLLM is required for VLLMModel. "
                "Please install it: pip install vllm"
            )
            raise ImportError(msg)

        # Fetch max_model_len from user param or auto-detect from HuggingFace AutoConfig
        if self.max_model_len is not None:
            final_max_model_len = self.max_model_len
        else:
            final_max_model_len = _get_max_model_len_from_config(self.model)

        # Set tensor_parallel_size as user param or auto-detect from GPU count
        final_tp_size = self.tensor_parallel_size if self.tensor_parallel_size is not None else _get_gpu_count()

        # Set max_num_batched_tokens as user param or use default
        final_max_batched = self.max_num_batched_tokens

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "enforce_eager": False,
            "trust_remote_code": True,
            "tensor_parallel_size": final_tp_size,
            "max_num_batched_tokens": final_max_batched,
        }

        if final_max_model_len is not None:
            llm_kwargs["max_model_len"] = final_max_model_len

        if self.cache_dir is not None:
            llm_kwargs["download_dir"] = self.cache_dir

        logger.info(
            f"Initializing vLLM with: model={self.model}, "
            f"max_model_len={final_max_model_len}, "
            f"tensor_parallel_size={final_tp_size}, "
            f"max_num_batched_tokens={final_max_batched}"
        )

        self._llm = LLM(**llm_kwargs)
        self._final_max_model_len = final_max_model_len

        max_gen_tokens = (
            self.max_tokens
            if self.max_tokens is not None
            else final_max_model_len
        )
        is_qwen3 = "Qwen3" in self.model or "qwen3" in self.model.lower()

        sampling_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": max_gen_tokens,
        }

        if is_qwen3:
            sampling_kwargs.update(
                {
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "min_p": self.min_p,
                }
            )
        else:
            sampling_kwargs["top_p"] = self.top_p

        self._sampling_params = SamplingParams(**sampling_kwargs)
        self._is_qwen3 = is_qwen3

    def generate(
        self,
        prompts: list[str] | list[list[dict[str, str]]],
    ) -> list[str]:
        """
        Generate text from prompts.

        Args:
            prompts: List of prompt strings or list of message dicts
                (for chat template).

        Returns:
            List of generated text strings.

        Raises:
            RuntimeError: If the model is not set up or generation fails.
        """
        if self._llm is None or self._sampling_params is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        try:
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )
            return [
                out.outputs[0].text if out.outputs else ""
                for out in outputs
            ]
        except Exception as e:
            msg = f"Error generating text: {e}"
            raise RuntimeError(msg) from e

    def get_tokenizer(self) -> Any:  # noqa: ANN401
        """Get the tokenizer from the LLM instance."""
        if self._llm is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self._llm.get_tokenizer()
