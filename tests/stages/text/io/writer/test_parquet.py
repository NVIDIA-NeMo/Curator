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
import time
import uuid
from unittest import mock

import pandas as pd
import pytest

from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.io.writer import utils as writer_utils
from nemo_curator.tasks import DocumentBatch


class TestParquetWriter:
    """Test suite for ParquetWriter with different data types."""

    @pytest.mark.parametrize("document_batch", ["pandas", "pyarrow"], indirect=True)
    @pytest.mark.parametrize("consistent_filename", [True, False])
    def test_parquet_writer(
        self,
        document_batch: DocumentBatch,
        consistent_filename: bool,
        tmpdir: str,
    ):
        """Test ParquetWriter with different data types."""
        # Create writer with specific output directory for this test
        output_dir = os.path.join(tmpdir, f"parquet_{document_batch.task_id}")
        writer = ParquetWriter(path=output_dir)

        # Setup
        writer.setup()
        assert writer.name == "parquet_writer"

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
        assert len(result.data) == 1
        assert result._metadata["format"] == "parquet"
        # assert previous keys from document_batch are present
        assert result._metadata["dummy_key"] == "dummy_value"
        # Verify stage_perf is properly handled
        # The stage should preserve all existing stage performance entries
        assert len(result._stage_perf) >= len(document_batch._stage_perf)

        # All original stage performance entries should be preserved
        for original_perf in document_batch._stage_perf:
            assert original_perf in result._stage_perf, "Original stage performance should be preserved"

        file_path = result.data[0]
        assert "_TEST_FILE_HASH" in file_path, f"File path should contain hash: {file_path}"
        assert os.path.exists(file_path), f"Output file should exist: {file_path}"
        assert os.path.getsize(file_path) > 0, "Output file should not be empty"

        # Verify file extension and content
        assert file_path.endswith(".parquet"), "Parquet files should have .parquet extension"
        df = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(df, document_batch.to_pandas())

    @pytest.mark.parametrize("document_batch", ["pandas"], indirect=True)
    def test_parquet_writer_overwrite_mode(self, document_batch: DocumentBatch, tmpdir: str):
        """Overwrite mode should remove existing dir contents and recreate the directory."""
        output_dir = os.path.join(tmpdir, "parquet_overwrite")
        os.makedirs(output_dir, exist_ok=True)
        dummy_file = os.path.join(output_dir, "dummy.txt")
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write("to be removed")

        # Sanity preconditions
        assert os.path.isdir(output_dir)
        assert os.path.exists(dummy_file)

        writer = ParquetWriter(path=output_dir, mode="overwrite")
        writer.setup()
        result = writer.process(document_batch)

        # Directory should exist; dummy file should be removed by overwrite
        assert os.path.isdir(output_dir)
        assert not os.path.exists(dummy_file)

        # Exactly one parquet output file is expected
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".parquet")]
        assert len(files) == 1
        assert result.data == files

        df = pd.read_parquet(files[0])
        pd.testing.assert_frame_equal(df, document_batch.to_pandas())

    def test_parquet_writer_with_columns_subset(self, pandas_document_batch: DocumentBatch, tmpdir: str):
        """Only selected columns should be written when columns are provided."""
        output_dir = os.path.join(tmpdir, "parquet_columns_subset")
        writer = ParquetWriter(path=output_dir, fields=["text", "score"])  # keep only subset

        writer.setup()
        result = writer.process(pandas_document_batch)

        # Verify file content only contains selected columns
        file_path = result.data[0]
        df = pd.read_parquet(file_path)
        expected = pandas_document_batch.to_pandas()[["text", "score"]]
        pd.testing.assert_frame_equal(df, expected)

    def test_parquet_writer_with_custom_options(self, pandas_document_batch: DocumentBatch, tmpdir: str):
        """Test ParquetWriter with custom formatting options."""
        output_dir = os.path.join(tmpdir, "parquet_custom")
        writer = ParquetWriter(path=output_dir, write_kwargs={"compression": "gzip", "engine": "pyarrow"})

        writer.setup()
        result = writer.process(pandas_document_batch)

        # Verify file was created with custom options
        file_path = result.data[0]
        assert os.path.exists(file_path)
        df = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(df, pandas_document_batch.to_pandas())

        # Verify task_id and stage_perf are preserved
        assert result.task_id == pandas_document_batch.task_id

        # Verify stage_perf is properly handled
        assert len(result._stage_perf) >= len(pandas_document_batch._stage_perf)
        for original_perf in pandas_document_batch._stage_perf:
            assert original_perf in result._stage_perf, "Original stage performance should be preserved"

    def test_parquet_writer_with_write_kwargs_override(self, pandas_document_batch: DocumentBatch, tmpdir: str):
        """Test that write_kwargs can override default parameters."""
        output_dir = os.path.join(tmpdir, "parquet_override")
        writer = ParquetWriter(
            path=output_dir,
            write_kwargs={"index": True, "compression": "lz4"},  # Override defaults
        )

        writer.setup()
        result = writer.process(pandas_document_batch)

        # Verify file was created - will include index due to override
        file_path = result.data[0]
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) > 0

        # Verify task_id and stage_perf are preserved
        assert result.task_id == pandas_document_batch.task_id

        # Verify stage_perf is properly handled
        assert len(result._stage_perf) >= len(pandas_document_batch._stage_perf)
        for original_perf in pandas_document_batch._stage_perf:
            assert original_perf in result._stage_perf, "Original stage performance should be preserved"

    def test_parquet_writer_with_custom_file_extension(self, pandas_document_batch: DocumentBatch, tmpdir: str):
        """Test ParquetWriter with custom file extension."""
        output_dir = os.path.join(tmpdir, "parquet_custom_ext")
        writer = ParquetWriter(
            path=output_dir,
            file_extension="pq",  # Use custom extension
        )

        writer.setup()
        result = writer.process(pandas_document_batch)

        # Verify file was created with custom extension
        file_path = result.data[0]
        assert os.path.exists(file_path), f"Output file should exist: {file_path}"
        assert os.path.getsize(file_path) > 0, "Output file should not be empty"

        # Verify the file has the custom extension
        assert file_path.endswith(".pq"), "File should have .pq extension when file_extension is set to 'pq'"

        # Verify content is still readable as Parquet
        df = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(df, pandas_document_batch.to_pandas())

        # Verify task_id and stage_perf are preserved
        assert result.task_id == pandas_document_batch.task_id

        # Verify stage_perf is properly handled
        assert len(result._stage_perf) >= len(pandas_document_batch._stage_perf)
        for original_perf in pandas_document_batch._stage_perf:
            assert original_perf in result._stage_perf, "Original stage performance should be preserved"

    @pytest.mark.parametrize("consistent_filename", [True, False])
    def test_jsonl_writer_overwrites_existing_file(
        self,
        pandas_document_batch: DocumentBatch,
        consistent_filename: bool,
        tmpdir: str,
    ):
        """Test that ParquetWriter overwrites existing files when writing to the same path."""
        # Create writer with specific output directory for this test
        output_dir = os.path.join(tmpdir, f"jsonl_{pandas_document_batch.task_id}")
        writer = ParquetWriter(path=output_dir)

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
            assert result1.data[0] == result2.data[0], "File path should be the same, since it'll be a hash"
        else:
            assert result1.data[0] != result2.data[0], "File path should be different, since it'll be a uuid"
            # When using UUIDs, files are different, so no overwrite occurs

        assert filesize_1 == filesize_2, "File size should be the same when written twice"
        assert file_modification_time_1 < file_modification_time_2, (
            "File modification time should be newer than the first write"
        )

        pd.testing.assert_frame_equal(pd.read_parquet(result1.data[0]), pd.read_parquet(result2.data[0]))
