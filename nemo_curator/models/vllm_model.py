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

"""vLLM model wrappers.

- :class:`VLLMBase` - shared engine management (creation, generation, GPU
  cleanup). ``_generate`` accepts text prompts *or* multimodal prompt dicts
  so audio/vision adapters reuse the same plumbing.
- :class:`VLLMModel` - generic text-generation :class:`ModelInterface`.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import torch
from loguru import logger

from nemo_curator.models.base import ModelInterface
from nemo_curator.utils.gpu_utils import get_gpu_count, get_max_model_len_from_config

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:
        pass

    class SamplingParams:
        pass


class VLLMBase:
    """Shared vLLM engine management for text and multimodal generation.

    Holds the loaded ``LLM`` engine and ``SamplingParams`` and exposes
    protected helpers for engine creation, generation, and GPU cleanup. Not
    for direct instantiation. ``_generate`` returns raw ``RequestOutput``
    objects so callers can read text *and* token-level metadata.
    """

    _llm: LLM | None = None
    _sampling_params: SamplingParams | None = None

    def _init_engine(self, model_kwargs: dict[str, Any], sampling_kwargs: dict[str, Any]) -> None:
        """Create the vLLM ``LLM`` engine and ``SamplingParams``.

        Args forward to ``vllm.LLM`` / ``vllm.SamplingParams``. Raises
        ``RuntimeError`` if engine construction fails.
        """
        start_time = time.perf_counter()
        try:
            self._llm = LLM(**model_kwargs)
        except Exception as e:
            msg = f"vLLM model loading failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
        logger.info("vLLM engine loaded in {:.3f}s", time.perf_counter() - start_time)
        self._sampling_params = SamplingParams(**sampling_kwargs)

    def _generate(self, prompts: list, *, use_tqdm: bool = False) -> list:
        """Run generation and return raw ``RequestOutput`` objects, one per prompt.

        ``prompts`` are text strings or multimodal prompt dicts. Raises
        ``RuntimeError`` if the engine is uninitialized or generation fails.
        """
        if self._llm is None or self._sampling_params is None:
            msg = "vLLM engine not initialized. Call setup() first."
            raise RuntimeError(msg)
        try:
            return self._llm.generate(prompts, sampling_params=self._sampling_params, use_tqdm=use_tqdm)
        except (RuntimeError, ValueError, TypeError) as e:
            msg = f"Error generating text: {e}"
            raise RuntimeError(msg) from e

    def _cleanup_gpu(self) -> None:
        """Release the engine and GPU memory.

        vLLM owns its tensor-parallel process group, so we do not call
        ``torch.distributed.destroy_process_group()`` here: that destroys the
        default/global group and would corrupt any other component (another
        stage, Ray primitives) sharing it in this process.
        """
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._sampling_params = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: {}", e)


class VLLMModel(VLLMBase, ModelInterface):
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
        """Initialize the vLLM model wrapper.

        Args:
            model: Model identifier (e.g., "microsoft/phi-4").
            max_model_len: Context length; auto-detected from HF AutoConfig
                when ``None``.
            tensor_parallel_size: TP GPU count; auto-detected when ``None``.
            min_p: Min-p sampling (Qwen3 only).
            cache_dir: Model weight cache directory.
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
        self._final_max_model_len: int | None = None
        self._is_qwen3: bool = False

    @property
    def model_id_names(self) -> list[str]:
        """Return the model identifier."""
        return [self.model]

    def setup(self) -> None:
        """Set up the vLLM model and sampling parameters."""
        if not VLLM_AVAILABLE:
            msg = "vLLM is required for VLLMModel. Please install it: pip install vllm"
            raise ImportError(msg)

        if self.max_model_len is not None:
            final_max_model_len = self.max_model_len
        else:
            final_max_model_len = get_max_model_len_from_config(self.model, cache_dir=self.cache_dir)

        final_tp_size = self.tensor_parallel_size if self.tensor_parallel_size is not None else get_gpu_count()

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

        max_gen_tokens = self.max_tokens if self.max_tokens is not None else final_max_model_len
        if max_gen_tokens is None:
            logger.warning(
                "max_tokens is None and max_model_len could not be auto-detected. "
                "vLLM will use its default (typically 16 tokens), which may be too few."
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

        self._init_engine(llm_kwargs, sampling_kwargs)
        self._final_max_model_len = final_max_model_len
        self._is_qwen3 = is_qwen3

    def generate(
        self,
        prompts: list[str],
    ) -> list[str]:
        """Generate text from prompt strings (or chat message dicts).

        Raises ``RuntimeError`` if the model is not set up or generation fails.
        """
        outputs = self._generate(prompts)
        return [out.outputs[0].text if out.outputs else "" for out in outputs]

    def get_tokenizer(self) -> Any:  # noqa: ANN401
        """Get the tokenizer from the LLM instance."""
        if self._llm is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self._llm.get_tokenizer()
