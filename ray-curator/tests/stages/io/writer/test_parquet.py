"""Test suite for ParquetWriter."""

import os
import uuid
from unittest import mock

import pandas as pd
import pytest

import ray_curator.stages.io.writer.utils as writer_utils
from ray_curator.stages.io.writer import ParquetWriter
from ray_curator.tasks import DocumentBatch


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
        writer = ParquetWriter(output_dir=output_dir)

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
                assert mock_uuid4.call_count == 0
            else:
                assert mock_get_deterministic_hash.call_count == 0
                assert mock_uuid4.call_count == 1

        # Verify file was created
        assert result.task_id == document_batch.task_id  # Task ID should match input
        assert len(result.data) == 1
        assert result._metadata["output_dir"] == output_dir
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

    def test_parquet_writer_with_custom_options(self, pandas_document_batch: DocumentBatch, tmpdir: str):
        """Test ParquetWriter with custom formatting options."""
        output_dir = os.path.join(tmpdir, "parquet_custom")
        writer = ParquetWriter(output_dir=output_dir, parquet_kwargs={"compression": "gzip", "engine": "pyarrow"})

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

    def test_parquet_writer_with_parquet_kwargs_override(self, pandas_document_batch: DocumentBatch, tmpdir: str):
        """Test that parquet_kwargs can override default parameters."""
        output_dir = os.path.join(tmpdir, "parquet_override")
        writer = ParquetWriter(
            output_dir=output_dir,
            parquet_kwargs={"index": True, "compression": "lz4"},  # Override defaults
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
            output_dir=output_dir,
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
