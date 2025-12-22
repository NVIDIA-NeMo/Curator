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

import gc
import time

import vllm
from loguru import logger

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.stages.text.models.utils import format_name_with_suffix
from nemo_curator.tasks import DocumentBatch


class VLLMEmbeddingModelStage(ModelStage):
    """VLLM-based model stage that produces embeddings.

    Unlike HuggingFace-based embedding stage, this submits all texts to VLLM
    at once and lets VLLM handle the batching internally for optimal performance.

    Args:
        model_identifier: The identifier of the Hugging Face model compatible with VLLM.
        text_field: The field in the DataFrame containing the text to embed.
        embedding_field: The field name for the output embeddings.
        hf_token: Hugging Face token for downloading the model, if needed.
        gpu_memory_utilization: Fraction of GPU memory to use for VLLM.
        num_cpus: Number of CPUs to use for the model.
        num_gpus: Number of GPUs to use for the model.

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        text_field: str = "text",
        embedding_field: str = "embeddings",
        hf_token: str | None = None,
        gpu_memory_utilization: float = 0.5,
        num_cpus: int = 8,
        num_gpus: int = 1,
    ):
        # Initialize base ModelStage - VLLM doesn't use tokenized input or batching
        super().__init__(
            model_identifier=model_identifier,
            hf_token=hf_token,
            has_seq_order=False,  # VLLM preserves order internally
        )
        if num_gpus > 1:
            msg = "Tensor parallelism is not supported for VLLM yet. If you need it, please open an issue on GitHub."
            raise ValueError(msg)

        # Override name and resources for VLLM-specific configuration
        self.name = format_name_with_suffix(model_identifier, suffix="_vllm_model")
        self.resources = Resources(cpus=num_cpus, gpus=num_gpus)

        self.text_field = text_field
        self.embedding_field = embedding_field
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.embedding_field]

    def _setup(self, local_files_only: bool = True) -> None:
        """Load the VLLM model for inference."""

        if self.model is None:
            start_time = time.time()
            self.model = vllm.LLM(
                model=self.model_identifier,
                runner="pooling",
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=max(1, int(self.num_gpus)),
                trust_remote_code=not local_files_only,
            )

            end_time = time.time()
            logger.info(f"VLLM model loaded in {end_time - start_time} seconds")
        else:
            logger.info("VLLM model already loaded")

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Process all texts in a single VLLM call.

        VLLM handles batching internally, so we submit all texts at once
        for optimal throughput.
        """
        df_cpu = batch.to_pandas()

        # Get all texts and submit to VLLM at once
        texts = df_cpu[self.text_field].tolist()
        outputs: list[vllm.EmbeddingRequestOutput] = self.model.embed(texts, use_tqdm=False)
        # Extract embeddings from outputs
        embeddings = [out.outputs.embedding for out in outputs]
        df_cpu = df_cpu.assign(**{self.embedding_field: embeddings})

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df_cpu,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def teardown(self) -> None:
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        gc.collect()
