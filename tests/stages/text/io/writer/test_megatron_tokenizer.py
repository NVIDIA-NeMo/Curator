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
import struct
import time
import uuid
from typing import Any
from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
import pytest

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.stages.text.io.writer import MegatronTokenizerWriter
from nemo_curator.stages.text.io.writer.megatron_tokenizer import _INDEX_HEADER
from nemo_curator.tasks import DocumentBatch


class MockTokenizerOutput:
    def __init__(self, input_ids: list[list[int]], attention_mask: list[list[int]]) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask


@pytest.fixture
def mock_tokenizer() -> Mock:
    tokenizer = Mock()
    tokenizer.vocab_size = 2**12
    tokenizer.eos_token_id = 1

    def mock_batch_encode_plus(texts: list[str], **kwargs: Any) -> MockTokenizerOutput:  # noqa: ANN401 ARG001
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []

        for text in texts:
            # Simulate tokenization: longer texts get more tokens
            token_count = len(text.split())

            # Create input IDs
            ids = [*range(1000, 1000 + token_count)]
            # Create attention mask
            mask = [1] * token_count

            input_ids.append(ids)
            attention_masks.append(mask)

        return MockTokenizerOutput(input_ids, attention_masks)

    tokenizer.batch_encode_plus = mock_batch_encode_plus
    return tokenizer


@pytest.fixture(autouse=True)
def setup_mocks(mock_tokenizer: Mock):
    with (
        patch("nemo_curator.stages.text.io.writer.megatron_tokenizer.AutoTokenizer") as mock_auto_tokenizer,
        patch("nemo_curator.stages.text.io.writer.megatron_tokenizer.snapshot_download") as mock_snapshot_download,
    ):
        # Setup AutoTokenizer mock
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # snapshot_download doesn't need special setup, just needs to not fail

        yield {
            "auto_tokenizer": mock_auto_tokenizer,
            "snapshot_download": mock_snapshot_download,
        }


def test_mocks_are_working_automatically(tmpdir: str):
    # This test can create a MegatronTokenizerWriter and call setup() without any issues
    # because the setup_mocks fixture is automatically applied due to autouse=True
    output_dir = os.path.join(tmpdir, "megatron_tokenizer")
    stage = MegatronTokenizerWriter(path=output_dir, model_identifier="test/model")

    # This would fail without the mocks being active
    stage.setup()

    # Verify the tokenizer was mocked correctly
    assert stage.tokenizer is not None
    assert stage.tokenizer.vocab_size is not None
    assert stage.tokenizer.eos_token_id is not None
    assert hasattr(stage.tokenizer, "batch_encode_plus")


class TestMegatronTokenizerWriter:
    """Test suite for MegatronTokenizerWriter with different data types."""

    @pytest.mark.parametrize("document_batch", ["pandas", "pyarrow"], indirect=True)
    @pytest.mark.parametrize("consistent_filename", [True, False])
    def test_megatron_tokenizer_writer(
        self,
        document_batch: DocumentBatch,
        consistent_filename: bool,
        tmpdir: str,
    ):
        """Test MegatronTokenizerWriter with different data types."""
        # Create writer with specific output directory for this test
        output_dir = os.path.join(tmpdir, f"megatron_tokenizer_{document_batch.task_id}")
        writer = MegatronTokenizerWriter(path=output_dir, model_identifier="test/model")

        # Setup
        writer.setup()
        assert writer.name == "megatron_tokenizer_writer"

        # Process
        with (
            mock.patch.object(
                writer_utils, "get_deterministic_hash", return_value="_TEST_FILE_HASH"
            ) as mock_get_deterministic_hash,
            mock.patch.object(uuid, "uuid4", return_value=mock.Mock(hex="_TEST_FILE_HASH")) as mock_uuid4,
        ):
            if consistent_filename:
                source_files = [f"file_{i}.jsonl" for i in range(len(document_batch.data))]
                document_batch._metadata["source_files"] = source_files
            result = writer.process(document_batch)

            if consistent_filename:
                assert mock_get_deterministic_hash.call_count == 1
                # Verify get_deterministic_hash was called with correct arguments
                mock_get_deterministic_hash.assert_called_once_with(source_files, document_batch.task_id)
                # because we call it once for task, and that should be the only one
                assert mock_uuid4.call_count <= 1
            else:
                assert mock_get_deterministic_hash.call_count == 0
                # because we call it once for task, and once for the filename
                assert mock_uuid4.call_count == 2

        # Verify file was created
        assert result.task_id == document_batch.task_id  # Task ID should match input
        assert len(result.data) == 2
        assert result._metadata["format"] == "megatron"
        # assert previous keys from document_batch are present
        assert result._metadata["dummy_key"] == "dummy_value"
        # Verify stage_perf is properly handled
        # The stage should preserve all existing stage performance entries
        assert len(result._stage_perf) >= len(document_batch._stage_perf)

        # All original stage performance entries should be preserved
        for original_perf in document_batch._stage_perf:
            assert original_perf in result._stage_perf, "Original stage performance should be preserved"

        # Verify file extensions
        assert result.data[0].endswith(".bin"), "Bin file should have .bin extension"
        assert result.data[1].endswith(".idx"), "Index file should have .idx extension"

        for file_path in result.data:
            assert "_TEST_FILE_HASH" in file_path, f"File path should contain hash: {file_path}"
            assert os.path.exists(file_path), f"Output file should exist: {file_path}"
            assert os.path.getsize(file_path) > 0, "Output file should not be empty"

    @pytest.mark.parametrize("large_vocab_size", [True, False])
    @pytest.mark.parametrize("append_eod", [True, False])
    def test_megatron_tokenizer_writer_file_prefix(
        self,
        pandas_document_batch: DocumentBatch,
        large_vocab_size: bool,
        append_eod: bool,
        tmpdir: str,
        setup_mocks: dict[str, Mock],
    ):
        """Test that MegatronTokenizerWriter files prefixes are correct."""

        if large_vocab_size:
            setup_mocks["auto_tokenizer"].from_pretrained.return_value.vocab_size = 2**17

        # Create writer with specific output directory for this test
        output_dir = os.path.join(
            tmpdir, f"megatron_{pandas_document_batch.task_id}{'_append_eod' if append_eod else ''}"
        )
        writer = MegatronTokenizerWriter(path=output_dir, model_identifier="test/model", append_eod=append_eod)

        # Setup
        writer.setup()

        # Process
        result = writer.process(pandas_document_batch)

        token_dtype_code = 4 if result._metadata["token_size"] == 4 else 8

        with open(result.data[1], "rb") as f:
            # Check the header
            header = f.read(9)
            assert header == _INDEX_HEADER, f"bad header, {_INDEX_HEADER} expected, {header} found"

            # Check the version
            version = struct.unpack("<Q", f.read(8))[0]
            assert version == 1, f"bad version, 1 expected, {version} found"

            # Check the dtype code
            code = struct.unpack("<B", f.read(1))[0]
            assert code == token_dtype_code, f"bad dtype code, {token_dtype_code} expected, {code} found"

            # Check the sequence count
            sequence_count = struct.unpack("<Q", f.read(8))[0]
            assert sequence_count == pandas_document_batch.num_items, (
                f"bad sequence count, {pandas_document_batch.num_items} expected, {sequence_count} found"
            )

            # Check the document count
            document_count = struct.unpack("<Q", f.read(8))[0]
            assert document_count == pandas_document_batch.num_items + 1, (
                f"bad document count, {pandas_document_batch.num_items + 1} expected, {document_count} found"
            )

            # Get the offset
            offset = f.tell()

        idx_buffer_mmap = np.memmap(result.data[1], mode="r", order="C")
        idx_buffer = memoryview(idx_buffer_mmap.data)

        sequence_lengths = np.frombuffer(idx_buffer, dtype=np.int32, count=sequence_count, offset=offset)

        sequence_pointers = np.frombuffer(
            idx_buffer,
            dtype=np.int64,
            count=sequence_count,
            offset=offset + sequence_lengths.nbytes,
        )

        document_indices = np.frombuffer(
            idx_buffer,
            dtype=np.int64,
            count=document_count,
            offset=offset + sequence_lengths.nbytes + sequence_pointers.nbytes,
        )
        # Check the sequence lengths & sequence count
        assert sequence_lengths.shape[0] == sequence_count, (
            f"bad sequence lengths, {sequence_count} expected, {sequence_lengths.shape[0]} found"
        )
        assert sequence_lengths.shape[0] == document_indices[-1], (
            f"bad document indices, {sequence_lengths.shape[0]} expected, {document_indices[-1]} found"
        )
        # Check the number of tokens
        assert (
            os.path.getsize(result.data[0]) // result._metadata["token_size"]
            == result._metadata["num_tokens"]
            == sum(sequence_lengths)
        ), (
            f"Number of tokens mismatch: Expected {os.path.getsize(result.data[0]) // result._metadata['token_size']} received {result._metadata['num_tokens']} and {sum(sequence_lengths)}"
        )

    @pytest.mark.parametrize("large_vocab_size", [True, False])
    def test_megatron_tokenizer_writer_append_eod(
        self,
        pandas_document_batch: DocumentBatch,
        large_vocab_size: bool,
        tmpdir: str,
        setup_mocks: dict[str, Mock],
    ):
        """Test that MegatronTokenizerWriter appends EOD token when append_eod is True."""

        if large_vocab_size:
            setup_mocks["auto_tokenizer"].from_pretrained.return_value.vocab_size = 2**17

        # Create writer with specific output directory for this test
        output_dir_append_eod = os.path.join(tmpdir, f"megatron_{pandas_document_batch.task_id}_append_eod")
        output_dir_no_append_eod = os.path.join(tmpdir, f"megatron_{pandas_document_batch.task_id}_no_append_eod")
        writer_append_eod = MegatronTokenizerWriter(
            path=output_dir_append_eod, model_identifier="test/model", append_eod=True
        )
        writer_no_append_eod = MegatronTokenizerWriter(
            path=output_dir_no_append_eod, model_identifier="test/model", append_eod=False
        )

        # Setup
        writer_append_eod.setup()
        writer_no_append_eod.setup()

        # Process
        result1 = writer_append_eod.process(pandas_document_batch)
        result2 = writer_no_append_eod.process(pandas_document_batch)

        # Check for EOD Tokens in the bin files
        bin_buffer_mmap_append_eod = np.memmap(result1.data[0], mode="r", order="C")
        bin_buffer_append_eod = memoryview(bin_buffer_mmap_append_eod.data)
        tokens = np.frombuffer(
            bin_buffer_append_eod,
            dtype=np.int32 if large_vocab_size else np.uint16,
            count=result1._metadata["num_tokens"],
            offset=0,
        )
        assert (tokens == result1._metadata["eod_token_id"]).sum() == pandas_document_batch.num_items, (
            f"EOD token id ({result1._metadata['eod_token_id']}) should appear exactly {pandas_document_batch.num_items} times in the token array, "
            f"but found {(tokens == result1._metadata['eod_token_id']).sum()}"
        )

        bin_buffer_mmap = np.memmap(result2.data[0], mode="r", order="C")
        bin_buffer = memoryview(bin_buffer_mmap.data)
        tokens2 = np.frombuffer(
            bin_buffer,
            dtype=np.int32 if large_vocab_size else np.uint16,
            count=result2._metadata["num_tokens"],
            offset=0,
        )
        assert (tokens2 == result2._metadata["eod_token_id"]).sum() == 0, (
            f"EOD token id ({result2._metadata['eod_token_id']}) should not appear in token array when append_eod is False, "
            f"but found {(tokens2 == result2._metadata['eod_token_id']).sum()}"
        )

        # We are appending 1 EOD token per document and we need to account for the token size
        eod_tokens_bytes_size = pandas_document_batch.num_items * result2._metadata["token_size"]
        assert os.path.getsize(result1.data[0]) - os.path.getsize(result2.data[0]) == eod_tokens_bytes_size, (
            f"File size difference should be equal to the number of documents * token size, got {os.path.getsize(result1.data[0]) - os.path.getsize(result2.data[0])} and the expected size is {eod_tokens_bytes_size}"
        )
        assert os.path.getsize(result1.data[1]) == os.path.getsize(result2.data[1]), (
            f"Index file sizes should be equal independent of append_eod since we have the same number of documents, got {os.path.getsize(result1.data[1])} and {os.path.getsize(result2.data[1])}"
        )

    @pytest.mark.parametrize("tokenization_batch_size", [1, 2, 3])
    def test_megatron_tokenizer_writer_tokenization_batch_size(
        self,
        pandas_document_batch: DocumentBatch,
        tokenization_batch_size: int,
        tmpdir: str,
    ):
        """Test that MegatronTokenizerWriter is not affected by the batch size."""

        # Create writer with specific output directory for this test
        output_dir_default_tokenization_batch_size = os.path.join(
            tmpdir, f"megatron_{pandas_document_batch.task_id}_default_tokenization_batch_size"
        )
        output_dir_modified_tokenization_batch_size = os.path.join(
            tmpdir, f"megatron_{pandas_document_batch.task_id}_modified_tokenization_batch_size"
        )
        writer_default_tokenization_batch_size = MegatronTokenizerWriter(
            path=output_dir_default_tokenization_batch_size, model_identifier="test/model"
        )
        writer_modified_tokenization_batch_size = MegatronTokenizerWriter(
            path=output_dir_modified_tokenization_batch_size,
            model_identifier="test/model",
            tokenization_batch_size=tokenization_batch_size,
        )

        # Setup
        writer_default_tokenization_batch_size.setup()
        writer_modified_tokenization_batch_size.setup()

        # Process
        result1 = writer_default_tokenization_batch_size.process(pandas_document_batch)
        result2 = writer_modified_tokenization_batch_size.process(pandas_document_batch)

        assert os.path.getsize(result1.data[0]) == os.path.getsize(result2.data[0]), (
            f"File sizes should be equal independent of batch size, got {os.path.getsize(result1.data[0])} and {os.path.getsize(result2.data[0])}"
        )
        assert os.path.getsize(result1.data[1]) == os.path.getsize(result2.data[1]), (
            f"Index file sizes should be equal independent of batch size, got {os.path.getsize(result1.data[1])} and {os.path.getsize(result2.data[1])}"
        )

    def test_megatron_tokenizer_writer_overwrite_mode(self, pandas_document_batch: DocumentBatch, tmpdir: str):
        """Overwrite mode should remove existing dir contents and recreate the directory."""
        output_dir = os.path.join(tmpdir, "megatron_overwrite")
        os.makedirs(output_dir, exist_ok=True)
        dummy_file = os.path.join(output_dir, "dummy.txt")
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write("to be removed")

        # Sanity preconditions
        assert os.path.isdir(output_dir)
        assert os.path.exists(dummy_file)

        writer = MegatronTokenizerWriter(path=output_dir, model_identifier="test/model", mode="overwrite")
        writer.setup()
        _result = writer.process(pandas_document_batch)

        # Directory should exist; dummy file should be removed by overwrite
        assert os.path.isdir(output_dir)
        assert not os.path.exists(dummy_file)

        # Exactly two megatron output files are expected
        files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith((".bin", ".idx"))])
        assert len(files) == 2
        # Verify file extensions
        assert files[0].endswith(".bin"), "Bin file should have .bin extension"
        assert files[1].endswith(".idx"), "Index file should have .idx extension"

    @pytest.mark.parametrize("consistent_filename", [True, False])
    def test_megatron_tokenizer_writer_overwrites_existing_file(
        self,
        pandas_document_batch: DocumentBatch,
        consistent_filename: bool,
        tmpdir: str,
    ):
        """Test that MegatronTokenizerWriter overwrites existing files when writing to the same path."""

        # Create writer with specific output directory for this test
        output_dir = os.path.join(tmpdir, f"megatron_{pandas_document_batch.task_id}")
        writer = MegatronTokenizerWriter(path=output_dir, model_identifier="test/model")

        # Setup
        writer.setup()

        # Process
        if consistent_filename:
            source_files = [f"file_{i}.jsonl" for i in range(len(pandas_document_batch.data))]
            pandas_document_batch._metadata["source_files"] = source_files
        # We write once
        result1 = writer.process(pandas_document_batch)
        filesize_1, file_modification_time_1 = os.path.getsize(result1.data[0]), os.path.getmtime(result1.data[0])
        time.sleep(0.01)
        # Then we overwrite it
        result2 = writer.process(pandas_document_batch)
        filesize_2, file_modification_time_2 = os.path.getsize(result2.data[0]), os.path.getmtime(result2.data[0])

        if consistent_filename:
            assert result1.data[0] == result2.data[0], "File paths should be the same, since it'll be a hash"
            assert result1.data[1] == result2.data[1], "File paths should be the same, since it'll be a hash"
        else:
            assert result1.data[0] != result2.data[0], "File paths should be different, since it'll be a uuid"
            assert result1.data[1] != result2.data[1], "File paths should be different, since it'll be a uuid"
            # When using UUIDs, files are different, so no overwrite occurs

        assert filesize_1 == filesize_2, "File size should be the same when written twice"
        assert file_modification_time_1 < file_modification_time_2, (
            "File modification time should be newer than the first write"
        )

        assert os.path.getsize(result1.data[0]) == os.path.getsize(result2.data[0]), (
            f"File sizes should be equal, got {os.path.getsize(result1.data[0])} and {os.path.getsize(result2.data[0])}"
        )
        assert os.path.getsize(result1.data[1]) == os.path.getsize(result2.data[1]), (
            f"File sizes should be equal, got {os.path.getsize(result1.data[1])} and {os.path.getsize(result2.data[1])}"
        )
