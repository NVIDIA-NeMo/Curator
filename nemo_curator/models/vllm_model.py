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

"""vLLM model wrappers for text generation.

Provides:
- :class:`VLLMModel` — generic vLLM wrapper implementing :class:`ModelInterface`.
- :class:`VLLMInference` — chat-template-aware helper with prompt formatting,
  batched generation, and GPU cleanup (used by audio PNC stages).
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any

import torch
import torch.distributed as dist
import yaml
from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoTokenizer

from nemo_curator.models.base import ModelInterface
from nemo_curator.utils.gpu_utils import get_gpu_count, get_max_model_len_from_config

try:
    os.environ.setdefault("VLLM_USE_V1", "0")
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:
        pass

    class SamplingParams:
        pass


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

    @property
    def model_id_names(self) -> list[str]:
        """Return the model identifier."""
        return [self.model]

    def setup(self) -> None:
        """Set up the vLLM model and sampling parameters."""
        if not VLLM_AVAILABLE:
            msg = "vLLM is required for VLLMModel. Please install it: pip install vllm"
            raise ImportError(msg)

        # Fetch max_model_len from user param or auto-detect from HuggingFace AutoConfig
        if self.max_model_len is not None:
            final_max_model_len = self.max_model_len
        else:
            final_max_model_len = get_max_model_len_from_config(self.model, cache_dir=self.cache_dir)

        # Set tensor_parallel_size as user param or auto-detect from GPU count
        final_tp_size = self.tensor_parallel_size if self.tensor_parallel_size is not None else get_gpu_count()

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

        self._sampling_params = SamplingParams(**sampling_kwargs)
        self._is_qwen3 = is_qwen3

    def generate(
        self,
        prompts: list[str],
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
            return [out.outputs[0].text if out.outputs else "" for out in outputs]
        except (RuntimeError, ValueError, TypeError) as e:
            msg = f"Error generating text: {e}"
            raise RuntimeError(msg) from e

    def get_tokenizer(self) -> Any:  # noqa: ANN401
        """Get the tokenizer from the LLM instance."""
        if self._llm is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self._llm.get_tokenizer()


class VLLMInference:
    """Chat-template-aware vLLM helper with prompt formatting and GPU cleanup.

    This is a plain (non-stage) class that wraps a vLLM ``LLM`` engine with
    chat-template prompt formatting, model loading, batched generation, and
    GPU cleanup.  Stage subclasses (e.g. ``PNCwithvLLMInferenceStage``) compose
    or inherit from it to get inference capabilities while plugging into the
    Curator pipeline lifecycle.

    Supports three prompt configuration modes:
    - a static prompt template (``prompt``)
    - a per-entry field containing the prompt (``prompt_field``)
    - a YAML file containing the prompt structure (``prompt_file``)
    """

    def __init__(  # noqa: PLR0913
        self,
        prompt: dict[str, str] | None = None,
        prompt_field: str | None = None,
        prompt_file: str | None = None,
        generation_field: str = "generation",
        model: dict[str, Any] | None = None,
        inference: dict[str, Any] | None = None,
        apply_chat_template: dict[str, Any] | None = None,
        use_chat_api: bool = False,
        device: str = "cpu",
    ):
        if model is None:
            model = {}
        if inference is None:
            inference = {}
        if apply_chat_template is None:
            apply_chat_template = {}

        self.generation_field = generation_field
        self.prompt = prompt
        self.prompt_field = prompt_field
        self.device = device

        prompt_args_counter = sum(
            [
                prompt is not None,
                prompt_field is not None,
                prompt_file is not None,
            ]
        )
        if prompt_args_counter < 1:
            msg = "One of `prompt`, `prompt_field` or `prompt_file` should be provided."
            raise ValueError(msg)
        if prompt_args_counter > 1:
            err: list[str] = []
            if prompt:
                err.append(f"`prompt` ({prompt})")
            if prompt_field:
                err.append(f"`prompt_field` ({prompt_field})")
            if prompt_file:
                err.append(f"`prompt_file` ({prompt_file})")
            msg = f"Found more than one prompt values: {', '.join(err)}."
            raise ValueError(msg)

        if prompt_file:
            self.prompt = self._read_prompt_file(prompt_file)

        self.model_params = model
        self.inference_params = inference
        self.chat_template_params = apply_chat_template
        self.use_chat_api = use_chat_api or "tokenizer_mode" in model

        self.sampling_params: SamplingParams | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.llm: LLM | None = None

    def setup_on_node(self) -> None:
        """Download model weights and tokenizer to the local cache.

        Called once per node before workers start.  Only downloads artifacts
        (no GPU memory used) so that :meth:`setup` can load instantly.
        """
        model_name = self.model_params.get("model", "")
        if model_name:
            snapshot_download(repo_id=model_name)
            AutoTokenizer.from_pretrained(model_name)

    def setup(self) -> None:
        """Instantiate the vLLM engine on the target device.

        Called once per worker.  Detects CUDA availability, then loads the
        model.  If :meth:`setup_on_node` was not called first, the tokenizer
        and sampling params are created here as well.
        """
        if self.device == "cuda" and not torch.cuda.is_available():
            msg = "CUDA is not available, but CUDA device is requested"
            raise RuntimeError(msg)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if self.tokenizer is None:
            model_name = self.model_params.get("model")
            if not model_name:
                msg = "'model' key is required in model_params but was not provided."
                raise ValueError(msg)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.sampling_params is None:
            self.sampling_params = SamplingParams(**self.inference_params)

        if self.llm is None:
            self.load_model()

    @staticmethod
    def _read_prompt_file(prompt_filepath: str) -> dict:
        """Read a YAML file with a chat-style prompt template."""
        with open(prompt_filepath) as f:
            return yaml.safe_load(f)

    def get_entry_prompt(self, data_entry: dict) -> str | list[dict]:
        """Format the prompt for a single data entry using the chat template."""
        prompt = self.prompt
        if self.prompt_field:
            prompt = data_entry.get(self.prompt_field)
            if prompt is None:
                logger.warning(f"prompt_field '{self.prompt_field}' missing from entry; skipping.")
                return []

        try:
            entry_chat = [{"role": role, "content": prompt[role].format(**data_entry)} for role in prompt]
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error formatting prompt template: {e}")
            return []

        try:
            return self.tokenizer.apply_chat_template(entry_chat, **self.chat_template_params)
        except (TypeError, ValueError, KeyError, RuntimeError) as e:
            logger.error(f"Error applying chat template: {e}")
            if self.use_chat_api:
                return entry_chat
            return []

    def load_model(self) -> None:
        """Instantiate the ``vllm.LLM`` engine."""
        os.environ["VLLM_USE_V1"] = "0"
        start_time = time.time()
        try:
            self.llm = LLM(**self.model_params)
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            msg = f"Model loading failed: {e}"
            raise RuntimeError(msg) from e

        logger.info(f"Time taken to load model: {round((time.time() - start_time) / 60, 2)} minutes")

    def process_batch(self, entry_prompts: list) -> list:
        """Run a single generation call on a batch of prompts."""
        try:
            if self.use_chat_api:
                return self.llm.chat(entry_prompts, self.sampling_params)
            return self.llm.generate(entry_prompts, self.sampling_params)
        except Exception as e:
            logger.error(f"Failed to generate outputs: {e}")
            msg = f"Generation failed: {e}"
            raise RuntimeError(msg) from e

    def process_entry_prompts(self, entry_prompts: list, batch_size: int = 10000) -> list:
        """Generate in batches, then clean up."""
        if self.llm is None:
            msg = "VLLMInference.setup() must be called before process_entry_prompts()."
            raise RuntimeError(msg)
        start_time = time.time()
        outputs: list = []
        for i in range(0, len(entry_prompts), batch_size):
            batch = entry_prompts[i : i + batch_size]
            outputs.extend(self.process_batch(batch))
        logger.info(f"Time taken to generate outputs: {round((time.time() - start_time) / 60, 2)} minutes")
        logger.info("Cleaning up!")
        self.clean_up()
        return outputs

    def clean_up(self) -> None:
        """Release GPU memory occupied by the vLLM engine."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
