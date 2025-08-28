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

import os
from functools import lru_cache
from typing import Any, Literal

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

from .utils import (
    ATTENTION_MASK_COLUMN,
    INPUT_ID_COLUMN,
    SEQ_ORDER_COLUMN,
    TOKEN_LENGTH_COLUMN,
    format_name_with_suffix,
)


class TokenizerStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Tokenizer stage for Hugging Face models.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        cache_dir: The Hugging Face cache directory. Defaults to None.
        hf_token: Hugging Face token for downloading the model, if needed. Defaults to None.
        text_field: The name of the text field in the input data. Defaults to "text".
        max_chars: Limits the total number of characters that can be fed to the tokenizer.
            If None, text will not be truncated. Defaults to None.
        max_seq_length: Limits the total sequence returned by the tokenizer so that it has a maximum length.
            If None, the tokenizer's model_max_length is used. Defaults to None.
        padding_side: The side to pad the input tokens. Defaults to "right".
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        unk_token: If True, set the pad_token to the tokenizer's unk_token. Defaults to False.

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        cache_dir: str | None = None,
        hf_token: str | None = None,
        text_field: str = "text",
        max_chars: int | None = None,
        max_seq_length: int | None = None,
        padding_side: Literal["left", "right"] = "right",
        sort_by_length: bool = True,
        unk_token: bool = False,
    ):
        self._name = format_name_with_suffix(model_identifier, suffix="_tokenizer")

        self.model_identifier = model_identifier
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.text_field = text_field
        self.max_chars = max_chars
        self.max_seq_length = max_seq_length
        self.padding_side = padding_side
        self.sort_by_length = sort_by_length
        self.unk_token = unk_token

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field, INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN] + (
            [SEQ_ORDER_COLUMN] if self.sort_by_length else []
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_actor_stage": True}

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        try:
            snapshot_download(
                repo_id=self.model_identifier,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                local_files_only=False,
            )
            self._setup(local_files_only=False)
        except Exception as e:
            msg = f"Failed to download {self.model_identifier}"
            raise RuntimeError(msg) from e

    @lru_cache(maxsize=1)  # noqa: B019
    def load_cfg(self, local_files_only: bool = True) -> AutoConfig:
        return AutoConfig.from_pretrained(
            self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only
        )

    # We use the _setup function to ensure that everything needed for the tokenizer is downloaded and loaded properly
    def _setup(self, local_files_only: bool = True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_identifier,
            padding_side=self.padding_side,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        if self.unk_token:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        if self.max_seq_length is None:
            self.max_seq_length = self.tokenizer.model_max_length

            # Guard against the HF bug
            # which sets max_seq_length to max(int) for some models
            if self.max_seq_length > 1e5:  # noqa: PLR2004
                self.max_seq_length = self.load_cfg(local_files_only=local_files_only).max_position_embeddings

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._setup(local_files_only=True)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        if self.max_chars is not None and self.max_chars > 0:
            df[self.text_field] = df[self.text_field].str.slice(0, self.max_chars)

        with torch.no_grad():
            tokens = self.tokenizer.batch_encode_plus(
                df[self.text_field].tolist(),
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
            )

        output = df.copy()
        output[INPUT_ID_COLUMN] = tokens.input_ids.tolist()
        output[ATTENTION_MASK_COLUMN] = tokens.attention_mask.tolist()

        if self.sort_by_length:
            # Add column to preserve original order
            output[SEQ_ORDER_COLUMN] = np.arange(len(df))
            output[TOKEN_LENGTH_COLUMN] = tokens.attention_mask.sum(axis=1)
            output = output.sort_values(by=TOKEN_LENGTH_COLUMN, kind="stable", ignore_index=True).drop(
                columns=[TOKEN_LENGTH_COLUMN]
            )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=output,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
