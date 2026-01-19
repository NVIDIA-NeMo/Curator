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

import struct
import uuid
from dataclasses import dataclass, field
from typing import BinaryIO

import numpy as np
from loguru import logger
from transformers import AutoTokenizer

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.file_utils import FILETYPE_TO_DEFAULT_EXTENSIONS

from .base import BaseWriter
from .utils import batched

_INDEX_HEADER = b"MMIDIDX\x00\x00"


@dataclass
class MegatronTokenizerWriter(BaseWriter):
    """Writer that writes a DocumentBatch to Megatron ready tokenized files."""

    model_identifier: str | None = None
    cache_dir: str | None = None
    hf_token: str | None = None
    text_field: str = "text"
    tokenization_batch_size: int = 1000  # Renamed from batch_size to avoid shadowing ProcessingStage.batch_size
    append_eod: bool = False

    # Disable the inherited fields attribute
    fields: list[str] | None = field(default=None, init=False, repr=False)

    name: str = "megatron_tokenizer_writer"
    file_extension: list[str] = field(default_factory=lambda: FILETYPE_TO_DEFAULT_EXTENSIONS["megatron"])

    def __post_init__(self):
        if self.model_identifier is None:
            msg = "model_identifier is required and must be provided"
            raise ValueError(msg)
        super().__post_init__()

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        try:
            # download the relevant tokenizer files once
            _ = AutoTokenizer.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
        except Exception as e:
            msg = f"Failed to download {self.model_identifier}"
            raise RuntimeError(msg) from e

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_identifier,
            cache_dir=self.cache_dir,
            local_files_only=True,
        )

    def process(self, task: DocumentBatch) -> FileGroupTask:
        sequence_lengths: list[int] = []
        # Get source files from metadata for deterministic naming
        if source_files := task._metadata.get("source_files"):
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            logger.warning("The task does not have source_files in metadata, using UUID for base filename")
            filename = uuid.uuid4().hex

        file_prefix = self.fs.sep.join([self._fs_path, filename])
        for file_extension in self.file_extension:
            file_path = file_prefix + file_extension
            if self.fs.exists(file_path):
                logger.debug(f"File {file_path} already exists, overwriting it")

        token_size = (
            -1
            if self.tokenizer.vocab_size is None
            else (4 if self.tokenizer.vocab_size > np.iinfo(np.uint16).max + 1 else 2)
        )
        if token_size == -1:
            logger.warning("tokenizer.vocab_size is not set, assuming 4 bytes per token (vocab_size > 65536)")
            token_size = 4
        token_dtype = np.int32 if token_size == 4 else np.uint16  # noqa: PLR2004
        token_dtype_code = (
            4 if token_size == 4 else 8  # noqa: PLR2004
        )  # NOTE(asolergi-nv): Megatron needs this dtype code in the .idx file | https://github.com/NVIDIA/Megatron-LM/blob/64cbae55ac85cd73fbadbc3c0d715c8123c5e13b/megatron/core/datasets/indexed_dataset.py#L41

        eod_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1
        if eod_token_id == -1:
            logger.warning("tokenizer.eos_token_id is not set, disabling append_eod")
            self.append_eod = False

        num_docs = task.num_items

        df = task.to_pandas()

        try:
            with self.fs.open(file_prefix + ".bin", "wb") as bin_file:
                for batch in batched(df[self.text_field], self.tokenization_batch_size):
                    tokens_batch = self.tokenizer.batch_encode_plus(
                        batch,
                        padding=False,
                        truncation=False,
                        add_special_tokens=False,
                        return_token_type_ids=False,
                        return_attention_mask=False,
                    ).input_ids
                    self.write_data(bin_file, token_dtype, eod_token_id, tokens_batch, sequence_lengths)
        except Exception as e:
            logger.error(f"Error while writing tokens to {file_prefix}: {e}")
            if self.fs.exists(file_prefix + ".bin"):
                self.fs.remove(file_prefix + ".bin")
            raise

        self.write_idx_data(file_prefix, token_size, token_dtype_code, sequence_lengths)

        logger.debug(f"Written batch to {file_prefix} with {num_docs} documents ({sum(sequence_lengths)} tokens)")

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[file_prefix + file_extension for file_extension in self.file_extension],
            _metadata={
                **task._metadata,
                "format": "megatron",
                "file_prefix": file_prefix,
                "num_tokens": sum(sequence_lengths),
                "token_size": token_size,
                "eod_token_id": eod_token_id,
            },
            _stage_perf=task._stage_perf,
        )

    def write_data(
        self,
        bin_file: BinaryIO,
        token_dtype: np.dtype,
        eod_token_id: int,
        tokens_batch: list[list[int]],
        sequence_lengths: list[int],
    ) -> None:
        """Write tokens to the .bin file
        Args:
            tokens_batch (list[list[int]]): The batch of tokens to write
        """
        if self.append_eod:
            tokens_batch = [[*tokens, eod_token_id] for tokens in tokens_batch]
        sequence_lengths.extend([len(tokens) for tokens in tokens_batch])
        tokens_batch = np.concatenate([np.array(tokens, dtype=token_dtype) for tokens in tokens_batch])
        bin_file.write(tokens_batch.tobytes(order="C"))

    def write_idx_data(
        self, file_prefix: str, token_size: int, token_dtype_code: int, sequence_lengths: list[int]
    ) -> None:
        """Write the .idx file data"""

        # Save .idx file
        # This file has:
        ## 9 Bytes from the _INDEX_HEADER
        ## 8 Bytes from the version (Just a "1")
        ## 1 Byte from the token_dtype_code
        ## 8 Bytes from the number of sequences
        ## 8 Bytes from the number of documents
        ## 8 Bytes from the initial document index
        ## 20 Bytes for every sequence/document:
        ### - 4 Bytes from the sequence length
        ### - 8 bytes from the sequence offset
        ### - 8 Bytes from the document index
        # So, if the .bin contains tokens from 35000 text sequences/documents, the .idx will have
        # 9+8+1+8+8+8+20*35000 = 700042 Bytes
        try:
            with self.fs.open(file_prefix + ".idx", "wb") as idx_file:
                # Index Header
                idx_file.write(_INDEX_HEADER)
                # Version
                idx_file.write(struct.pack("<Q", 1))
                # Numeric code for the DType
                idx_file.write(struct.pack("<B", token_dtype_code))

                # Number of sequences in the dataset
                sequence_count = len(sequence_lengths)
                idx_file.write(struct.pack("<Q", sequence_count))

                document_indices = np.arange(len(sequence_lengths) + 1, dtype=np.int64)
                # Number of documents in the dataset
                document_count = len(document_indices)
                idx_file.write(struct.pack("<Q", document_count))

                # Number of tokens per sequence
                sequence_lengths = np.array(sequence_lengths, dtype=np.int32)
                idx_file.write(sequence_lengths.tobytes(order="C"))

                # Byte offsets for all sequences
                sequence_pointers = np.array(self._sequence_pointers(sequence_lengths, token_size), dtype=np.int64)
                idx_file.write(sequence_pointers.tobytes(order="C"))

                # Sequence indices marking the end of each document
                idx_file.write(document_indices.tobytes(order="C"))
        except Exception as e:
            logger.error(f"Error while writing idx data to {file_prefix}: {e}")
            if self.fs.exists(file_prefix + ".idx"):
                self.fs.remove(file_prefix + ".idx")
            raise

    @staticmethod
    def _sequence_pointers(sequence_lengths: list[int], token_size: int) -> list[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (list[int]): The length of each sequence
            token_size (int): The size of each token in bytes
        Returns:
            list[int]: The pointer to the beginning of each sequence
        """
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * token_size
        return list_ptr
