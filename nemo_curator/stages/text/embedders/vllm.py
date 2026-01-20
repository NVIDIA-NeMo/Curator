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

import time
from typing import TYPE_CHECKING, Any

from huggingface_hub import snapshot_download
from vllm import LLM

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.models.utils import format_name_with_suffix
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from transformers import AutoTokenizer


class VLLMEmbeddingModelStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        vllm_init_kwargs: dict[str, Any] | None = None,
        text_field: str = "text",
        pretokenize: bool = False,
        embedding_field: str = "embeddings",
        max_chars: int | None = None,
        cache_dir: str | None = None,
        hf_token: str | None = None,
        verbose: bool = False,
    ):
        self.model_identifier = model_identifier
        self.vllm_init_kwargs = vllm_init_kwargs or {}

        self.text_field = text_field
        self.pretokenize = pretokenize
        self.embedding_field = embedding_field
        self.max_chars = max_chars

        self.cache_dir = cache_dir
        self.hf_token = hf_token

        self.verbose = verbose
        # after setup
        self.model: None | LLM = None
        self.tokenizer: None | AutoTokenizer = None
        # stage setup
        self.resources = Resources(
            cpus=1,
            gpus=1,
        )
        self.name = format_name_with_suffix(model_identifier, suffix="_vllm")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field, self.embedding_field]

    def _initialize_vllm(self) -> None:
        vllm_init_kwargs = self.vllm_init_kwargs.copy()
        # set defaults here
        if "enforce_eager" not in vllm_init_kwargs:
            vllm_init_kwargs["enforce_eager"] = False
        if "runner" not in vllm_init_kwargs:
            vllm_init_kwargs["runner"] = "pooling"
        if "model_impl" not in vllm_init_kwargs:
            # TODO: Once transformers is bumped to 5.0 then we should also support transformers
            vllm_init_kwargs["model_impl"] = "vllm"
        if self.cache_dir is not None and "download_dir" not in vllm_init_kwargs:
            vllm_init_kwargs["download_dir"] = self.cache_dir

        # Reduce verbosity when not in verbose mode
        if not self.verbose and "disable_log_stats" not in vllm_init_kwargs:
            vllm_init_kwargs["disable_log_stats"] = True

        self.model = LLM(model=self.model_identifier, **vllm_init_kwargs)

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if not self.verbose:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()

        snapshot_download(self.model_identifier, cache_dir=self.cache_dir, token=self.hf_token, local_files_only=False)
        self._initialize_vllm()

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self.model is None:
            self._initialize_vllm()
        if self.pretokenize:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                local_files_only=False,
            )

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        if self.max_chars is not None:
            df[self.text_field] = df[self.text_field].str.slice(0, self.max_chars)
        input_data = df[self.text_field].tolist()
        metrics = {}

        if self.pretokenize:
            from vllm.inputs import TokensPrompt

            if self.tokenizer is None:
                msg = (
                    "Tokenizer is not initialized. Please call setup() before processing or set pretokenize to False."
                )
                raise ValueError(msg)

            t0 = time.perf_counter()
            max_model_len = self.model.model_config.max_model_len
            tokenized_data = self.tokenizer.batch_encode_plus(input_data, truncation=True, max_length=max_model_len)
            input_data = [TokensPrompt(prompt_token_ids=ids) for ids in tokenized_data.input_ids]
            metrics["tokenization_time"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        vllm_output = self.model.embed(input_data, truncate_prompt_tokens=-1, use_tqdm=self.verbose)
        metrics["vllm_embedding_time"] = time.perf_counter() - t0

        df[self.embedding_field] = [e.outputs.embedding for e in vllm_output]

        self._log_metrics(metrics)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
