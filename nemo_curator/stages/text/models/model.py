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
import os
from collections.abc import Generator
from typing import Literal

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from .utils import ATTENTION_MASK_COLUMN, INPUT_ID_COLUMN, SEQ_ORDER_COLUMN, clip_tokens, format_name_with_suffix


class ModelStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Base class for Hugging Face model inference.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        cache_dir: The Hugging Face cache directory. Defaults to None.
        hf_token: Hugging Face token for downloading the model, if needed. Defaults to None.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        has_seq_order: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        padding_side: The side to pad the input tokens. Defaults to "right".
        unpack_inference_batch: Whether to unpack the inference batch with **kwargs. Defaults to False.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        cache_dir: str | None = None,
        hf_token: str | None = None,
        model_inference_batch_size: int = 256,
        has_seq_order: bool = True,
        padding_side: Literal["left", "right"] = "right",
        unpack_inference_batch: bool = False,
        autocast: bool = True,
    ):
        self._name = format_name_with_suffix(model_identifier, suffix="_model")
        # Assume that the model can fit on a single GPU
        self._resources = Resources(cpus=1, gpus=1)

        self.model_identifier = model_identifier
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.model_inference_batch_size = model_inference_batch_size
        self.has_seq_order = has_seq_order
        self.padding_side = padding_side
        self.unpack_inference_batch = unpack_inference_batch
        self.autocast = autocast

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN] + ([SEQ_ORDER_COLUMN] if self.has_seq_order else [])

    def outputs(self) -> tuple[list[str], list[str]]:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        try:
            snapshot_download(
                repo_id=self.model_identifier,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                local_files_only=False,
            )

            _setup_function = getattr(self, "_setup", None)
            if callable(_setup_function):
                _setup_function(local_files_only=False)
            else:
                logger.warning(f"Subclass {self.__class__.__name__} does not implement _setup method")

        except Exception as e:
            msg = f"Failed to download {self.model_identifier}"
            raise RuntimeError(msg) from e

    def setup(self, _: WorkerMetadata | None = None) -> None:
        _setup_function = getattr(self, "_setup", None)
        if callable(_setup_function):
            # We use the _setup function to ensure that everything needed for the model is downloaded and loaded properly
            _setup_function(local_files_only=True)
        else:
            msg = "Subclasses must implement this method"
            raise NotImplementedError(msg)

    def yield_next_batch(self, df: pd.DataFrame) -> Generator[dict[str, torch.Tensor]]:
        """
        Yields a generator of model inputs for the next batch.
        We only move the batch to the GPU to reduce the memory overhead.

        Args:
            df (pd.DataFrame): The Pandas DataFrame (with input_ids and attention_mask) to process.

        Yields:
            Generator[dict[str, torch.Tensor]]: A generator of model inputs for the next batch.

        """
        for i in range(0, len(df), self.model_inference_batch_size):
            yield clip_tokens(
                {
                    INPUT_ID_COLUMN: torch.tensor(
                        df[INPUT_ID_COLUMN][i : i + self.model_inference_batch_size].tolist()
                    ).to(self.model.device),
                    ATTENTION_MASK_COLUMN: torch.tensor(
                        df[ATTENTION_MASK_COLUMN][i : i + self.model_inference_batch_size].tolist()
                    ).to(self.model.device),
                },
                padding_side=self.padding_side,
            )

    def process_model_output(
        self, outputs: torch.Tensor, model_input_batch: dict[str, torch.Tensor] | None = None
    ) -> dict[str, np.ndarray] | torch.Tensor:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def collect_outputs(self, processed_outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        result = {}
        for key in processed_outputs[0]:
            result[key] = np.concatenate([out[key] for out in processed_outputs], axis=0)
        return result

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def _model_forward(self, model_input_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.unpack_inference_batch:
            return self.model(**model_input_batch)
        else:
            return self.model(model_input_batch)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df_cpu = batch.to_pandas()

        processed_outputs = []
        for model_input_batch in self.yield_next_batch(df_cpu):
            # Forward pass
            with torch.no_grad():
                if self.autocast:
                    with torch.autocast(device_type="cuda"):
                        outputs = self._model_forward(model_input_batch)
                else:
                    outputs = self._model_forward(model_input_batch)

            processed_output = self.process_model_output(outputs, model_input_batch)
            del model_input_batch
            processed_outputs.append(processed_output)

        # Collect all outputs
        collected_output = self.collect_outputs(processed_outputs)

        # Create output Pandas DataFrame
        df_cpu = self.create_output_dataframe(df_cpu, collected_output)

        # Sort by seq_order to preserve original order from tokenizer
        if self.has_seq_order:
            df_cpu = df_cpu.sort_values(by=SEQ_ORDER_COLUMN, ignore_index=True).drop(columns=[SEQ_ORDER_COLUMN])

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df_cpu,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def teardown(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()
