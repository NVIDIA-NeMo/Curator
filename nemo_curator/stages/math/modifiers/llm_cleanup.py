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

import pandas as pd

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.models.vllm_model import VLLMModel
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.models.utils import format_name_with_suffix
from nemo_curator.tasks import DocumentBatch


class LLMCleanupStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    LLM-based text cleanup stage using vLLM for distributed inference.

    This stage uses a VLLMModel wrapper to generate cleaned text from input prompts.
    It handles filtering, sorting, prompt formatting, and output field management.
    """

    def __init__(
        self,
        model: str | VLLMModel,
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
        """
        Initialize the LLM cleanup stage.

        Args:
            model: Model identifier string (e.g., "microsoft/phi-4") or VLLMModel instance.
            system_prompt: Prompt template string with {text} placeholder.
            text_field: Name of the input text field. Defaults to "text".
            output_field: Name of the output field. Defaults to "cleaned_text".
            max_model_len: Maximum model context length. If not specified, vLLM will auto-detect.
            classification: If True, output to "label" field instead of output_field. Defaults to False.
            temperature: Sampling temperature. Defaults to 0.7.
            top_p: Top-p sampling parameter. Defaults to 0.8.
            top_k: Top-k sampling parameter. Defaults to 20.
            min_p: Min-p sampling parameter (for Qwen3). Defaults to 0.0.
            max_tokens: Maximum tokens to generate. Defaults to None.
            cache_dir: Cache directory for model weights. Defaults to None.
            filter_by_n_tokens: Filter chunks by n_tokens field. Defaults to False.
            n_tokens_field: Name of the n_tokens field. Defaults to "n_tokens".
        """
        if isinstance(model, VLLMModel):
            self._model = model
            model_name = model.model
        else:
            self._model = VLLMModel(
                model=model,
                max_model_len=max_model_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
            )
            model_name = model

        self.system_prompt = system_prompt
        self.text_field = text_field
        self.output_field = output_field
        self.max_model_len = max_model_len
        self.classification = classification
        self.filter_by_n_tokens = filter_by_n_tokens
        self.n_tokens_field = n_tokens_field
        self._resources = Resources(cpus=1, gpus=1)
        self._name = format_name_with_suffix(model_name, suffix="_llm_cleanup")

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
        """Set up the model wrapper."""
        self._model.setup()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

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

        generated_texts = self._model.generate(prompts)

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
