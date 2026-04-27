# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Reusable vLLM inference helper class.

Provides :class:`VLLMInference`, a plain (non-stage) class that wraps a vLLM
``LLM`` engine with chat-template prompt formatting, model loading, batched
generation, and GPU cleanup.

This class is **not** a Curator ``ProcessingStage``.  Stage subclasses (e.g.
:class:`PNCwithvLLMInferenceStage`) compose or inherit from it to get
inference capabilities while plugging into the Curator pipeline lifecycle.

Supports three prompt configuration modes:

- a static prompt template (``prompt``)
- a per-entry field containing the prompt (``prompt_field``)
- a YAML file containing the prompt structure (``prompt_file``)
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

try:
    # vLLM v1 engine uses multiprocessing that conflicts with Ray actors.
    # Force the classic (v0) engine to avoid core process spawning issues.
    os.environ.setdefault("VLLM_USE_V1", "0")
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


class VLLMInference:
    """Plain helper that manages a vLLM engine for text generation.

    This mirrors the ``vLLMInference`` class from *generic-sdp* so that the
    same prompt-formatting and inference logic can be reused across different
    Curator stages without coupling to the ``ProcessingStage`` hierarchy.

    Args:
        prompt:             Static YAML-style chat prompt (dict of role→content).
        prompt_field:       Per-entry key that holds the prompt template.
        prompt_file:        Path to a YAML file with the prompt structure.
        generation_field:   Output key for the generated text.
        model:              Kwargs forwarded to ``vllm.LLM()``.
        inference:          Kwargs forwarded to ``vllm.SamplingParams()``.
        apply_chat_template: Kwargs forwarded to ``tokenizer.apply_chat_template()``.
        use_chat_api:       When *True*, use ``llm.chat()`` instead of
                            ``llm.generate()``.  Defaults to *False*.
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

        self.device: str | None = None
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
        self.device = "cuda"
        if not torch.cuda.is_available():
            self.device = "cpu"
            logger.warning("CUDA is not available, using CPU")
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_params["model"])
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
            prompt = data_entry[self.prompt_field]

        entry_chat = [{"role": role, "content": prompt[role].format(**data_entry)} for role in prompt]

        try:
            return self.tokenizer.apply_chat_template(entry_chat, **self.chat_template_params)
        except (TypeError, ValueError, KeyError, RuntimeError) as e:
            logger.error(f"Error applying chat template: {e}")
            return entry_chat

    def load_model(self) -> None:
        """Instantiate the ``vllm.LLM`` engine."""
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
        """Generate in batches, then clean up.

        If :meth:`setup` has not been called yet, it is called automatically.
        """
        if self.llm is None:
            self.setup()
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
        if self.device is not None and self.device != "cpu":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
