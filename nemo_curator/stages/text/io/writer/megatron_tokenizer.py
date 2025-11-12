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

import numpy as np
from loguru import logger

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_COLUMN, INPUT_ID_COLUMN, TOKEN_LENGTH_COLUMN
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.file_utils import FILETYPE_TO_DEFAULT_EXTENSIONS

from .base import BaseWriter

_INDEX_HEADER = b"MMIDIDX\x00\x00"


@dataclass
class MegatronTokenWriterStage(BaseWriter):
    """Stage that writes a DocumentBatch to a Megatron tokenizer index file."""

    append_eod: bool = False
    file_extension: list[str] = field(default_factory=lambda: FILETYPE_TO_DEFAULT_EXTENSIONS["megatron"])
    _document_indices: list[int] = field(default_factory=lambda: [0])
    _sequence_lengths: list[int] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(self._sequence_lengths)

    @property
    def num_documents(self) -> int:
        return len(self._document_indices)

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

    def process(self, task: DocumentBatch) -> FileGroupTask:
        """Process a DocumentBatch and write to files.

        Args:
            task (DocumentBatch): DocumentBatch containing data to write

        Returns:
            FileGroupTask: Task containing paths to written files
        """
        # Get source files from metadata for deterministic naming
        if source_files := task._metadata.get("source_files"):
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            logger.warning("The task does not have source_files in metadata, using UUID for base filename")
            filename = uuid.uuid4().hex

        token_size = task._metadata.get("token_size", -1)
        if token_size == -1:
            logger.warning("tokenizer.vocab_size is not set, assuming 4 bytes per token (vocab_size > 65536)")
            token_size = 4
        eod_token_id = task._metadata.get("eod_token_id", -1)
        if eod_token_id == -1:
            logger.warning("tokenizer.eos_token_id is not set, disabling append_eod")
            self.append_eod = False
        file_prefix = self.fs.sep.join([self._fs_path, filename])
        for file_extension in self.file_extension:
            file_path = file_prefix + file_extension
            if self.fs.exists(file_path):
                logger.debug(f"File {file_path} already exists, overwriting it")

        self.write_data(task, file_prefix, token_size, eod_token_id)
        logger.debug(f"Written {self.num_tokens} tokens ({self.num_documents} documents) to {file_prefix}")

        # Create FileGroupTask with written files
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[file_prefix + file_extension for file_extension in self.file_extension],
            _metadata={
                **task._metadata,
                "format": "megatron",
                "file_prefix": file_prefix,
            },
            _stage_perf=task._stage_perf,
        )

    def write_data(self, task: DocumentBatch, file_prefix: str, token_size: int, eod_token_id: int) -> None:
        """Write data to Megatron tokenizer index file."""

        token_dtype = np.int32 if token_size == 4 else np.uint16  # noqa: PLR2004
        token_dtype_code = (
            4 if token_size == 4 else 8  # noqa: PLR2004
        )  # NOTE(asolergi-nv): From https://github.com/NVIDIA/Megatron-LM/blob/d6979d6cceb0007eec7c8960738f4dc0276bb540/megatron/core/datasets/indexed_dataset.py#L49-L59

        df = task.to_pandas()  # Convert to pandas DataFrame if needed # TODO(asolergi-nv): Check with curator if needed or if we should do this after computing the lengths of each document

        # Compute length of each sample with the attention mask
        df[TOKEN_LENGTH_COLUMN] = df[ATTENTION_MASK_COLUMN].apply(np.sum)
        # Drop attention mask
        df = df.drop(columns=[ATTENTION_MASK_COLUMN])  # TODO(asolergi-nv): Optional
        # Write tokens to disk

        """
        for document_tokens in df[INPUT_ID_COLUMN].tolist():
            self._sequence_lengths.append(len(document_tokens))
            self._document_indices.append(len(self._sequence_lengths))
        """
        with open(file_prefix + ".bin", "wb") as f:
            for document_tokens in df[INPUT_ID_COLUMN].tolist():
                if self.append_eod:
                    document_tokens.append(eod_token_id)
                f.write(
                    np.array(document_tokens, dtype=token_dtype).tobytes(order="C")
                )  # TODO(asolergi-nv): Check if document_tokens is already a np array + which dtype
                self._sequence_lengths.append(len(document_tokens))
                self._document_indices.append(len(self._sequence_lengths))

        # Write index file to disk
        # Save .idx file
        # This file has:
        ## 9 Bytes from the _INDEX_HEADER
        ## 8 Byte of metadata (Just a "1")
        ## 1 Byte from the token_dtype_code
        ## 8 Bytes from the number of sequences
        ## 8 Bytes from the number of documents
        ## 8 Bytes from the initial document index
        ## 20 Bytes for every sequence/document
        ### 4 Bytes from the sequence length
        ### 8 bytes from the sequence offset
        ### 8 Bytes from the document index
        # So, if the .bin contains tokens from 35000 text sequences/documents, the .idx will have
        # 9+8+1+8+8+8+20*35000 = 700042 Bytes
        with open(file_prefix + ".idx", "wb") as f:
            # Index Header
            f.write(_INDEX_HEADER)
            # Version
            f.write(struct.pack("<Q", 1))
            # Numeric code for the DType
            f.write(struct.pack("<B", token_dtype_code))

            sequence_pointers = self._sequence_pointers(self._sequence_lengths, token_size)

            # Number of sequences in the dataset
            sequence_count = len(self._sequence_lengths)
            f.write(struct.pack("<Q", sequence_count))

            # Number of documents in the dataset
            document_count = len(self._document_indices)
            f.write(struct.pack("<Q", document_count))

            # Number of tokens per sequence
            sequence_lengths = np.array(self._sequence_lengths, dtype=np.int32)
            f.write(sequence_lengths.tobytes(order="C"))
            del sequence_lengths

            # Byte offsets for all sequences
            sequence_pointers = np.array(sequence_pointers, dtype=np.int64)
            f.write(sequence_pointers.tobytes(order="C"))
            del sequence_pointers

            # Sequence indices marking the end of each document
            document_indices = np.array(self._document_indices, dtype=np.int64)
            f.write(document_indices.tobytes(order="C"))


@dataclass
class MegatronTokenizerWriter(CompositeStage[DocumentBatch, FileGroupTask]):
    """Writer that writes a DocumentBatch to a Megatron tokenizer index file."""

    output_dir: str
    # TokenizerStage arguments
    model_identifier: str
    cache_dir: str | None = None
    hf_token: str | None = None
    text_field: str = "text"
    max_seq_length: int | None = None  # TODO(asolergi-nv): This should be infinite
    unk_token: bool = False
    # MegatronTokenWriterStage arguments
    append_eod: bool = False

    _name: str = "megatron_tokenizer_writer"

    def __post_init__(self) -> None:
        super().__init__()
        self.stages = [
            TokenizerStage(
                model_identifier=self.model_identifier,
                cache_dir=self.cache_dir,
                hf_token=self.hf_token,
                text_field=self.text_field,
                max_seq_length=self.max_seq_length,
                sort_by_length=False,
                unk_token=self.unk_token,
            ),
            MegatronTokenWriterStage(self.output_dir, append_eod=self.append_eod),
        ]

    def decompose(self) -> list[ProcessingStage]:
        return self.stages

    def get_description(self) -> str:
        """Get a description of this composite stage."""

        return "Tokenize and write to Megatron tokenizer index files"
