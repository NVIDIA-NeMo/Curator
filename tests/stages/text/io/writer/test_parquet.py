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
from pathlib import Path
from unittest import mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.core import split_protocol

from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.io.writer import base as writer_base
from nemo_curator.tasks import DocumentBatch
from tests.stages.text.io.utils import normalize_string_dtypes


class TestParquetWriter:
    """Test suite for ParquetWriter with different data types."""

    def test_arrow_table_preserves_schema_and_writer_options(self, tmp_path: Path) -> None:
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("text", pa.large_string()),
                pa.field("embeddings", pa.list_(pa.float32())),
                pa.field("nested", pa.list_(pa.struct([("score", pa.float32())]))),
            ],
            metadata={b"source": b"test"},
        )
        table = pa.Table.from_arrays(
            [
                pa.array([1, 2], type=pa.int64()),
                pa.array(["first", None], type=pa.large_string()),
                pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32())),
                pa.array([[{"score": 1.5}], None], type=schema.field("nested").type),
            ],
            schema=schema,
        )
        batch = DocumentBatch(dataset_name="test", data=table)
        batch.to_pandas = mock.Mock(side_effect=AssertionError("Arrow input must not convert to pandas"))
        writer = ParquetWriter(
            path=str(tmp_path),
            write_kwargs={
                "compression": "zstd",
                "compression_level": 3,
                "row_group_size": 1,
                "use_compliant_nested_type": False,
            },
        )
        writer.setup()

        output_file = Path(writer.process(batch).data[0])
        result = pq.read_table(output_file)

        assert result.equals(table)
        assert result.schema.equals(schema, check_metadata=True)
        assert pq.ParquetFile(output_file).metadata.num_row_groups == 2
        assert all(pq.ParquetFile(output_file).metadata.row_group(i).column(0).compression == "ZSTD" for i in range(2))

    def test_arrow_table_applies_field_selection(self, tmp_path: Path) -> None:
        table = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "embeddings": pa.array([[1.0], [2.0]], type=pa.list_(pa.float32())),
                "unused": ["a", "b"],
            }
        )
        writer = ParquetWriter(path=str(tmp_path), fields=["embeddings", "id"])
        writer.setup()

        result = pq.read_table(writer.process(DocumentBatch(dataset_name="test", data=table)).data[0])

        assert result.column_names == ["embeddings", "id"]
        assert result.equals(table.select(["embeddings", "id"]))

    def test_arrow_table_uses_pandas_for_index(self, tmp_path: Path) -> None:
        table = pa.table({"id": [1], "text": ["first"]})
        batch = DocumentBatch(dataset_name="test", data=table)
        dataframe = table.to_pandas()
        batch.to_pandas = mock.Mock(return_value=dataframe)
        writer = ParquetWriter(path=str(tmp_path), write_kwargs={"index": True})
        writer.setup()

        output_file = writer.process(batch).data[0]

        batch.to_pandas.assert_called_once_with()
        assert "__index_level_0__" in pq.read_table(output_file).column_names

    @pytest.mark.parametrize(
        "write_kwargs",
        [{"engine": "fastparquet"}, {"index": True}, {"partition_cols": []}],
    )
    def test_arrow_table_uses_pandas_for_unsupported_options(
        self, tmp_path: Path, write_kwargs: dict[str, object]
    ) -> None:
        writer = ParquetWriter(path=str(tmp_path), write_kwargs=write_kwargs)

        assert writer._write_arrow(pa.table({"id": [1]}), str(tmp_path / "unused.parquet")) is False

    def test_arrow_table_writes_through_configured_fsspec_filesystem(self) -> None:
        table = pa.table({"text": pa.array(["first", "second"], type=pa.large_string())})
        batch = DocumentBatch(dataset_name="test", data=table)
        batch.to_pandas = mock.Mock(side_effect=AssertionError("Arrow input must not convert to pandas"))
        writer = ParquetWriter(
            path="memory://nmcur-316/output",
            write_kwargs={"storage_options": {"skip_instance_cache": True}},
        )
        writer.setup()

        output_file = writer.process(batch).data[0]
        _, fs_path = split_protocol(output_file)
        result = pq.read_table(fs_path, filesystem=writer.fs)

        assert result.equals(table)
        assert result.schema.equals(table.schema)

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
                writer_base, "get_deterministic_hash", return_value="_TEST_FILE_HASH"
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
                # consistent path uses the content hash for the filename; uuid is unused
                assert mock_uuid4.call_count == 0
            else:
                assert mock_get_deterministic_hash.call_count == 0
                # non-consistent path uses a single uuid for the filename
                assert mock_uuid4.call_count == 1

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
        if isinstance(document_batch.data, pa.Table):
            assert pq.read_table(file_path).equals(document_batch.data)
        else:
            df = pd.read_parquet(file_path)
            pd.testing.assert_frame_equal(
                normalize_string_dtypes(df),
                normalize_string_dtypes(document_batch.to_pandas()),
            )

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

    @pytest.mark.parametrize(
        "path",
        [
            "s3://test-bucket/output",
            "/local/path",
        ],
    )
    def test_parquet_writer_write_data_path_protocol_handling(self, pandas_document_batch: DocumentBatch, path: str):
        """Test that write_data is called with correct protocol handling for cloud and local paths."""
        with mock.patch.object(writer_base, "check_output_mode", return_value=None):
            writer = ParquetWriter(path=path)
            writer.setup()

        with (
            mock.patch.object(writer.fs, "exists", return_value=False),
            mock.patch.object(writer, "write_data") as mock_write_data,
        ):
            writer.process(pandas_document_batch)

            mock_write_data.assert_called_once()
            file_path = mock_write_data.call_args[0][1]
            assert file_path.startswith(path), f"Path should start with {path}"
