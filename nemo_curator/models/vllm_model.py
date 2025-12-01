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

from typing import Any, NamedTuple

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:
        pass

    class SamplingParams:
        pass

from nemo_curator.models.base import ModelInterface


class ModelSpec(NamedTuple):
    """Model specification for vLLM configuration."""

    name: str
    max_model_len: int
    tensor_parallel_size: int
    max_num_batched_tokens: int
    batch_size: int

_MODELS: dict[str, ModelSpec] = {
    "microsoft/phi-4": ModelSpec("microsoft/phi-4", 16_384, 1, 8_192, 128),
    "google/gemma-3-12b-it": ModelSpec("google/gemma-3-12b-it", 131_072, 1, 8_192, 128),
    "google/gemma-3-27b-it": ModelSpec("google/gemma-3-27b-it", 131_072, 2, 4096, 64),
    "google/gemma-3-27b-it-32k": ModelSpec("google/gemma-3-27b-it", 32_768, 2, 4096, 64),
    "Qwen/Qwen2-7B-Instruct": ModelSpec("Qwen/Qwen2-7B-Instruct", 131_072, 1, 8_192, 128),
    "Qwen/Qwen2-72B-Instruct": ModelSpec("Qwen/Qwen2-72B-Instruct", 131_072, 4, 4096, 64),
    "mistralai/Mistral-7B-Instruct-v0.3": ModelSpec(
        "mistralai/Mistral-7B-Instruct-v0.3", 32768, 1, 8_192, 128
    ),
    "deepseek-ai/DeepSeek-V3": ModelSpec("deepseek-ai/DeepSeek-V3", 163840, 8, 4096, 64),
    "Qwen/Qwen2.5-32B-Instruct": ModelSpec("Qwen/Qwen2.5-32B-Instruct", 32_768, 8, 4096, 64),
    "Qwen/Qwen2.5-72B-Instruct": ModelSpec("Qwen/Qwen2.5-72B-Instruct", 131_072, 8, 4096, 64),
    "Qwen/Qwen3-30B-A3B": ModelSpec("Qwen/Qwen3-30B-A3B", 32_768, 2, 8_192, 64),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ModelSpec(
        "Qwen/Qwen3-30B-A3B-Instruct-2507", 262144, 4, 4096, 32
    ),
    "openai/gpt-oss-20b": ModelSpec("openai/gpt-oss-20b", 131_072, 1, 8_192, 128),
    "openai/gpt-oss-120b": ModelSpec("openai/gpt-oss-120b", 131_072, 4, 8_192, 64),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": ModelSpec(
        "Qwen/Qwen3-Coder-30B-A3B-Instruct", 262144, 4, 8_192, 256
    ),
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": ModelSpec(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2", 131_072, 1, 8_192, 128
    ),
}


class VLLMModel(ModelInterface):
    """Generic vLLM language model wrapper for text generation."""

    def __init__(  # noqa: PLR0913
        self,
        model: str,
        max_model_len: int | None = None,
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
            max_model_len: Maximum model context length. If not specified, vLLM will auto-detect.
            temperature: Sampling temperature. Defaults to 0.7.
            top_p: Top-p sampling parameter. Defaults to 0.8.
            top_k: Top-k sampling parameter. Defaults to 20.
            min_p: Min-p sampling parameter (for Qwen3). Defaults to 0.0.
            max_tokens: Maximum tokens to generate. Defaults to None.
            cache_dir: Cache directory for model weights. Defaults to None.
        """
        self.model = model
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None

    def model_id_names(self) -> list[str]:
        """Return the model identifier."""
        return [self.model]

    def setup(self) -> None:
        """Set up the vLLM model and sampling parameters."""
        if not VLLM_AVAILABLE:
            msg = "vLLM is required for VLLMModel. Please install it: pip install vllm"
            raise ImportError(msg)

        model_spec = _MODELS.get(self.model)

        if model_spec:
            final_max_model_len = (
                min(self.max_model_len, model_spec.max_model_len)
                if self.max_model_len is not None
                else model_spec.max_model_len
            )
        else:
            final_max_model_len = self.max_model_len

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "enforce_eager": False,
            "trust_remote_code": True,
        }

        if final_max_model_len is not None:
            llm_kwargs["max_model_len"] = final_max_model_len

        if model_spec:
            llm_kwargs["tensor_parallel_size"] = model_spec.tensor_parallel_size
            llm_kwargs["max_num_batched_tokens"] = model_spec.max_num_batched_tokens

        self._llm = LLM(**llm_kwargs)
        self._final_max_model_len = final_max_model_len

        max_gen_tokens = (
            self.max_tokens
            if self.max_tokens is not None
            else (model_spec.max_model_len if model_spec else final_max_model_len)
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

    def generate(self, prompts: list[str] | list[list[dict[str, str]]]) -> list[str]:
        """
        Generate text from prompts.

        Args:
            prompts: List of prompt strings or list of message dicts (for chat template).

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
            return [out.outputs[0].text for out in outputs]
        except Exception as e:
            msg = f"Error generating text: {e}"
            raise RuntimeError(msg) from e

    def get_tokenizer(self) -> Any:  # noqa: ANN401
        """Get the tokenizer from the LLM instance."""
        if self._llm is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self._llm.get_tokenizer()
