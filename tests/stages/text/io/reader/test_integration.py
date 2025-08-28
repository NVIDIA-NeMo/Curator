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

import contextlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pytest
from packaging.version import Version

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
    create_id_generator_actor,
    kill_id_generator_actor,
)
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.tasks import DocumentBatch

ray_version = Version(importlib.metadata.version("ray"))
# TODO: Some Ray Data tests are skipped because they require ray >= 2.49.0 see https://github.com/ray-project/ray/issues/54841


# File format configurations
FILE_FORMAT_CONFIGS = {
    "jsonl": {
        "extension": ".jsonl",
        "reader_class": JsonlReader,
        "writer_func": "_write_jsonl_file",
    },
}


def _write_jsonl_file(file_path: Path, file_data: list[dict[str, Any]]) -> None:
    """Write data to a JSONL file."""
    with open(file_path, "w") as f:
        for record in file_data:
            f.write(json.dumps(record) + "\n")


def _write_parquet_file(file_path: Path, file_data: list[dict[str, Any]]) -> None:
    """Write data to a Parquet file (placeholder for future implementation)."""
    df = pd.DataFrame(file_data)
    df.to_parquet(file_path, index=False)


def create_test_files(
    input_dir: Path,
    file_format: Literal["jsonl"] = "jsonl",  # Can extend to include "parquet" in future
    num_files: int = 100,
    records_per_file: int = 2,
) -> pd.DataFrame:
    """Create test files in the specified format and return the expected combined DataFrame.

    Args:
        input_dir: Directory to create test files in
        file_format: Format of files to create ("jsonl", "parquet", etc.)
        num_files: Number of files to create
        records_per_file: Number of records per file

    Returns:
        DataFrame containing all the test data
    """
    if file_format not in FILE_FORMAT_CONFIGS:
        msg = f"Unsupported file format: {file_format}. Supported formats: {list(FILE_FORMAT_CONFIGS.keys())}"
        raise ValueError(msg)

    input_dir.mkdir(parents=True, exist_ok=True)
    config = FILE_FORMAT_CONFIGS[file_format]

    all_data = []
    for file_idx in range(num_files):
        file_data = []
        for record_idx in range(records_per_file):
            record = {
                "text": f"Document {file_idx}-{record_idx}",
                "category": f"category_{file_idx}",
                "score": 0.9 + file_idx * 0.01 + record_idx * 0.001,
                "metadata": {"file_id": file_idx, "record_id": record_idx},
            }
            file_data.append(record)
            all_data.append(record)

        # Write file in the specified format
        file_path = input_dir / f"test_file_{file_idx}{config['extension']}"
        writer_func = globals()[config["writer_func"]]
        writer_func(file_path, file_data)

    return pd.DataFrame(all_data)


def create_reader_pipeline(
    input_dir: Path,
    file_format: Literal["jsonl"] = "jsonl",  # Can extend to include "parquet" in future
    generate_ids: bool = False,
    assign_ids: bool = False,
) -> Pipeline:
    """Create a pipeline with the appropriate reader for the specified file format.

    Args:
        input_dir: Directory containing input files
        file_format: Format of files to read ("jsonl", "parquet", etc.)
        generate_ids: Whether to generate IDs for documents
        assign_ids: Whether to assign IDs to documents

    Returns:
        Pipeline configured with the appropriate reader
    """
    if file_format not in FILE_FORMAT_CONFIGS:
        msg = f"Unsupported file format: {file_format}. Supported formats: {list(FILE_FORMAT_CONFIGS.keys())}"
        raise ValueError(msg)

    pipeline = Pipeline(name="reader_integration_test")
    config = FILE_FORMAT_CONFIGS[file_format]
    reader_class = config["reader_class"]

    reader = reader_class(
        file_paths=str(input_dir),  # Pass directory path, not glob pattern
        files_per_partition=1,
        _generate_ids=generate_ids,
        _assign_ids=assign_ids,
    )

    pipeline.add_stage(reader)
    return pipeline


@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(
            ((RayDataExecutor, {}), "jsonl"),
            id="ray_data_jsonl",
            marks=pytest.mark.skipif(ray_version < Version("2.49.0"), reason="Requires ray >= 2.49"),
        ),
        pytest.param(((XennaExecutor, {"execution_mode": "streaming"}), "jsonl"), id="xenna_streaming_jsonl"),
    ],
    indirect=True,
)
class TestReaderIntegrationWithoutIdGenerator:
    """Integration tests for readers without ID generation."""

    # Class attributes for shared test data
    backend_cls: type | None = None
    config: dict[str, Any] | None = None
    input_dir: Path | None = None
    expected_df: pd.DataFrame | None = None
    output_tasks: list[DocumentBatch] | None = None

    @pytest.fixture
    def test_config(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> "TestReaderIntegrationWithoutIdGenerator":
        """Set up test environment and execute pipeline."""
        # Get parameters from the combined parametrize
        (backend_cls, config), file_format = request.param

        self.backend_cls = backend_cls
        self.config = config
        self.file_format = file_format

        # Create test data
        tmp_path = tmp_path_factory.mktemp(f"reader_test_no_ids_{file_format}")
        self.input_dir = tmp_path / "input"
        self.expected_df = create_test_files(self.input_dir, file_format=file_format)

        # Create and run pipeline
        pipeline = create_reader_pipeline(
            self.input_dir, file_format=file_format, generate_ids=False, assign_ids=False
        )
        executor = backend_cls(config)

        # Execute pipeline
        self.output_tasks = pipeline.run(executor)

        return self

    def test_output_task_count_and_types(self, test_config: "TestReaderIntegrationWithoutIdGenerator"):
        """Test that output tasks have correct count and types."""
        assert test_config.output_tasks is not None
        assert len(test_config.output_tasks) > 0
        assert all(isinstance(task, DocumentBatch) for task in test_config.output_tasks)

    def test_dataframe_content_equality(self, test_config: "TestReaderIntegrationWithoutIdGenerator"):
        """Test that output DataFrame matches input data (without ID columns)."""
        assert test_config.output_tasks is not None
        assert test_config.expected_df is not None

        # Combine all output DataFrames
        combined_df = pd.concat([task.to_pandas() for task in test_config.output_tasks], ignore_index=True)

        # Should not have ID column
        assert CURATOR_DEDUP_ID_STR not in combined_df.columns

        # Sort both DataFrames for comparison (order might differ due to parallel processing)
        expected_sorted = test_config.expected_df.sort_values("text").reset_index(drop=True)
        actual_sorted = combined_df.sort_values("text").reset_index(drop=True)

        # Compare DataFrames
        pd.testing.assert_frame_equal(expected_sorted, actual_sorted)

    def test_total_record_count(self, test_config: "TestReaderIntegrationWithoutIdGenerator"):
        """Test that total number of records matches expected."""
        assert test_config.output_tasks is not None
        assert test_config.expected_df is not None

        total_records = sum(len(task.to_pandas()) for task in test_config.output_tasks)
        assert total_records == len(test_config.expected_df)


@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(
            ((RayDataExecutor, {}), "jsonl"),
            id="ray_data_jsonl",
            marks=pytest.mark.skipif(ray_version < Version("2.49.0"), reason="Requires ray >= 2.49"),
        ),
        pytest.param(((XennaExecutor, {"execution_mode": "streaming"}), "jsonl"), id="xenna_streaming_jsonl"),
        # Future formats can be added here:
    ],
    indirect=True,
)
class TestReaderIntegrationWithIdGenerator:
    """Integration tests for readers with ID generation."""

    # Class attributes for shared test data
    backend_cls: type | None = None
    config: dict[str, Any] | None = None
    input_dir: Path | None = None
    expected_df: pd.DataFrame | None = None
    generate_output_tasks: list[DocumentBatch] | None = None
    assign_output_tasks: list[DocumentBatch] | None = None

    @pytest.fixture
    def test_config(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> "TestReaderIntegrationWithIdGenerator":
        """Set up test environment with ID generator and execute pipelines."""
        # Get parameters from the combined parametrize
        (backend_cls, config), file_format = request.param

        self.backend_cls = backend_cls
        self.config = config
        self.file_format = file_format

        # Create test data
        tmp_path = tmp_path_factory.mktemp(f"reader_test_with_ids_{file_format}")
        self.input_dir = tmp_path / "input"
        self.expected_df = create_test_files(self.input_dir, file_format=file_format)

        # Create ID generator actor
        create_id_generator_actor()

        try:
            # Run pipeline with ID generation
            generate_pipeline = create_reader_pipeline(
                self.input_dir, file_format=file_format, generate_ids=True, assign_ids=False
            )
            executor1 = backend_cls(config)
            self.generate_output_tasks = generate_pipeline.run(executor1)

            # Run pipeline with ID assignment
            assign_pipeline = create_reader_pipeline(
                self.input_dir, file_format=file_format, generate_ids=False, assign_ids=True
            )
            executor2 = backend_cls(config)
            self.assign_output_tasks = assign_pipeline.run(executor2)

            yield self

        finally:
            # Cleanup ID generator actor
            with contextlib.suppress(Exception):
                kill_id_generator_actor()

    def test_id_generation_pipeline(self, test_config: "TestReaderIntegrationWithIdGenerator"):
        """Test that ID generation pipeline produces correct IDs."""
        assert test_config.generate_output_tasks is not None

        # Combine all output DataFrames
        combined_df = pd.concat([task.to_pandas() for task in test_config.generate_output_tasks], ignore_index=True)

        # Should have ID column
        assert CURATOR_DEDUP_ID_STR in combined_df.columns

        # IDs should be unique and sequential
        ids = sorted(combined_df[CURATOR_DEDUP_ID_STR].tolist())
        expected_ids = list(range(len(combined_df)))
        assert ids == expected_ids

    def test_id_assignment_pipeline(self, test_config: "TestReaderIntegrationWithIdGenerator"):
        """Test that ID assignment pipeline produces same IDs as generation."""
        assert test_config.assign_output_tasks is not None
        assert test_config.generate_output_tasks is not None

        # Get DataFrames from both pipelines
        generate_df = pd.concat([task.to_pandas() for task in test_config.generate_output_tasks], ignore_index=True)
        assign_df = pd.concat([task.to_pandas() for task in test_config.assign_output_tasks], ignore_index=True)

        # Both should have ID columns
        assert CURATOR_DEDUP_ID_STR in generate_df.columns
        assert CURATOR_DEDUP_ID_STR in assign_df.columns

        # Sort both by text to ensure same order for comparison
        generate_df_sorted = generate_df.sort_values("text").reset_index(drop=True)
        assign_df_sorted = assign_df.sort_values("text").reset_index(drop=True)

        # IDs should be identical
        assert generate_df_sorted[CURATOR_DEDUP_ID_STR].tolist() == assign_df_sorted[CURATOR_DEDUP_ID_STR].tolist()

    def test_dataframe_content_equality_with_ids(self, test_config: "TestReaderIntegrationWithIdGenerator"):
        """Test that output DataFrame matches input data (excluding ID column)."""
        assert test_config.generate_output_tasks is not None
        assert test_config.expected_df is not None

        # Get DataFrame from generation pipeline
        combined_df = pd.concat([task.to_pandas() for task in test_config.generate_output_tasks], ignore_index=True)

        # Remove ID column for comparison
        content_df = combined_df.drop(columns=[CURATOR_DEDUP_ID_STR])

        # Sort both DataFrames for comparison
        expected_sorted = test_config.expected_df.sort_values("text").reset_index(drop=True)
        actual_sorted = content_df.sort_values("text").reset_index(drop=True)

        # Compare DataFrames
        pd.testing.assert_frame_equal(expected_sorted, actual_sorted)

    def test_total_record_count_with_ids(self, test_config: "TestReaderIntegrationWithIdGenerator"):
        """Test that total number of records matches expected."""
        assert test_config.generate_output_tasks is not None
        assert test_config.assign_output_tasks is not None
        assert test_config.expected_df is not None

        generate_total = sum(len(task.to_pandas()) for task in test_config.generate_output_tasks)
        assign_total = sum(len(task.to_pandas()) for task in test_config.assign_output_tasks)
        expected_total = len(test_config.expected_df)

        assert generate_total == expected_total
        assert assign_total == expected_total
