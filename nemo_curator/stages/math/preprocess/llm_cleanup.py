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

from typing import Any

import pandas as pd

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    # Create dummy classes for type hints when vllm is not available
    class LLM:  # noqa: PLW0602
        pass

    class SamplingParams:  # noqa: PLW0602
        pass

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.models.utils import format_name_with_suffix
from nemo_curator.tasks import DocumentBatch


class LLMCleanupStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    LLM-based text cleanup stage using vLLM for distributed inference.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        text_field: str = "text",
        output_field: str = "cleaned_text",
        max_model_len: int | None = None,
        classification: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int | None = None,
        cache_dir: str | None = None,
        filter_by_n_tokens: bool = False,
        n_tokens_field: str = "n_tokens",
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.text_field = text_field
        self.output_field = output_field
        self.max_model_len = max_model_len
        self.classification = classification
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.filter_by_n_tokens = filter_by_n_tokens
        self.n_tokens_field = n_tokens_field
        self._llm = None
        self._sampling_params = None
        self._resources = Resources(cpus=1, gpus=1)
        self._name = format_name_with_suffix(self.model, suffix="_llm_cleanup")

    def inputs(self) -> tuple[list[str], list[str]]:
        input_fields = [self.text_field]
        if self.filter_by_n_tokens:
            input_fields.append(self.n_tokens_field)
        return ["data"], input_fields

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.classification:
            return ["data"], ["label"]
        return ["data"], [self.output_field]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if not VLLM_AVAILABLE:
            msg = "vLLM is required for LLMCleanupStage. Please install it: pip install vllm"
            raise ImportError(msg)

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "enforce_eager": False,
            "trust_remote_code": True,
        }
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len
        if self.cache_dir is not None:
            llm_kwargs["cache_dir"] = self.cache_dir

        self._llm = LLM(**llm_kwargs)

        max_gen_tokens = self.max_tokens if self.max_tokens is not None else self.max_model_len
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

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        if self._llm is None or self._sampling_params is None:
            msg = "LLM not initialized. Call setup() first."
            raise RuntimeError(msg)

        if self.filter_by_n_tokens and self.n_tokens_field in df.columns:
            if self.max_model_len is None:
                msg = "filter_by_n_tokens requires max_model_len to be set"
                raise ValueError(msg)
            threshold = int(0.8 * self.max_model_len)
            df = df[df[self.n_tokens_field] < threshold].copy()
            if len(df) == 0:
                return DocumentBatch(
                    task_id=batch.task_id,
                    dataset_name=batch.dataset_name,
                    data=pd.DataFrame(),
                    _metadata=batch._metadata,
                    _stage_perf=batch._stage_perf,
                )

        if self.n_tokens_field in df.columns:
            df = df.sort_values(by=self.n_tokens_field, kind="stable", ignore_index=True)

        prompts = []
        for _, row in df.iterrows():
            text = str(row[self.text_field]) if pd.notna(row[self.text_field]) else ""
            prompt = self.system_prompt.format(text=text)
            prompts.append(prompt)

        try:
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )
            generated_texts = [out.outputs[0].text for out in outputs]
        except Exception as e:
            msg = f"Error generating text for batch: {e}"
            raise RuntimeError(msg) from e

        output_df = df.copy()

        if self.classification:
            output_df["label"] = generated_texts
            if self.text_field in output_df.columns:
                output_df = output_df.drop(columns=[self.text_field])
        else:
            output_df[self.output_field] = generated_texts

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=output_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
