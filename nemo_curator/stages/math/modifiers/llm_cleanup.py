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

from collections import defaultdict

import pandas as pd
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
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

    def __init__(  # noqa: PLR0913
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
            n_tokens_field: Name of the n_tokens field. Defaults to "n_tokens".
        """
        self._model_kwargs = {
            "model": model if isinstance(model, str) else model.model,
            "max_model_len": max_model_len,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "cache_dir": cache_dir,
        }
        if isinstance(model, VLLMModel):
            self._model = model
            self.model_name = model.model
        else:
            self._model = None
            self.model_name = model

        self.system_prompt = system_prompt
        self.text_field = text_field
        self.output_field = output_field
        self.max_model_len = max_model_len
        self.classification = classification
        self.n_tokens_field = n_tokens_field
        self.resources = Resources(cpus=1.0, gpus=1.0)
        self.name = format_name_with_suffix(self.model_name, suffix="_llm_cleanup")
        self._final_max_model_len = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.classification:
            return ["data"], ["label"]
        return ["data"], [self.output_field]

    def _initialize_model(self) -> None:
        """Create and initialize the VLLMModel."""
        if self._model is None:
            self._model = VLLMModel(**self._model_kwargs)
        self._model.setup()
        if hasattr(self._model, "_final_max_model_len"):
            self._final_max_model_len = self._model._final_max_model_len
        else:
            self._final_max_model_len = self.max_model_len

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Download weights and initialize vLLM once per node to avoid torch.compile race conditions."""
        cache_dir = self._model_kwargs.get("cache_dir") if self._model is None else self._model.cache_dir

        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self.model_name, cache_dir=cache_dir, local_files_only=False)
        self._initialize_model()

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Load tokenizer per worker. Falls back to full init if setup_on_node was not called."""
        if self._model is None or not hasattr(self._model, "_llm") or self._model._llm is None:
            self._initialize_model()
        self._tokenizer = self._model.get_tokenizer()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        if self.n_tokens_field in df.columns:
            final_max_model_len = (
                self._final_max_model_len if self._final_max_model_len is not None else self.max_model_len
            )
            if final_max_model_len is None:
                msg = "max_model_len must be set when processing chunked data (n_tokens field present)"
                raise ValueError(msg)
            threshold = int(0.8 * final_max_model_len)
            df = df[df[self.n_tokens_field] < threshold].copy()
            if len(df) == 0:
                return DocumentBatch(
                    task_id=batch.task_id,
                    dataset_name=batch.dataset_name,
                    data=pd.DataFrame(columns=df.columns),
                    _metadata=batch._metadata,
                    _stage_perf=batch._stage_perf,
                )
            df = df.sort_values(by=self.n_tokens_field, kind="stable", ignore_index=True)
            df = df.drop(columns=[self.n_tokens_field])

        # Qwen3 supports /no_think as an inline prompt switch to disable chain-of-thought.
        # Qwen3.5+ dropped this; use enable_thinking=False in chat template instead.
        is_qwen3_family = "Qwen3" in self.model_name or "qwen3" in self.model_name.lower()
        is_qwen3_only = is_qwen3_family and "Qwen3." not in self.model_name and "qwen3." not in self.model_name.lower()

        prompts = []
        for _, row in df.iterrows():
            text = str(row[self.text_field]) if pd.notna(row[self.text_field]) else ""
            user_prompt = self.system_prompt.format_map(defaultdict(str, text=text))

            if is_qwen3_only:
                user_prompt = user_prompt + " /no_think"
                system_content = " /no_think"
            else:
                system_content = ""

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ]

            if self._tokenizer is not None:
                try:
                    formatted_prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **({"enable_thinking": False} if is_qwen3_family else {}),
                    )
                    prompts.append(formatted_prompt)
                except (AttributeError, ValueError, TypeError, KeyError) as e:
                    logger.warning(f"apply_chat_template failed, using raw prompt: {e}")
                    prompts.append(user_prompt)
            else:
                prompts.append(user_prompt)

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
