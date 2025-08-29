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

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.deduplication.removal import TextDuplicatesRemovalStage
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.io.reader.parquet import ParquetReaderStage
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import DocumentBatch, FileGroupTask


def create_ids_to_remove_files(ids_to_remove_dir: Path, num_files: int = 5) -> list[int]:
    """Create parquet files containing IDs to remove (random ints 0-999, split into num_files)."""
    ids_to_remove_dir.mkdir(parents=True, exist_ok=True)
    ids = list(range(0, 1000, 5))
    random.shuffle(ids)
    chunks = np.array_split(ids, num_files)
    for i, chunk in enumerate(chunks):
        pd.DataFrame({"id": chunk}).to_parquet(ids_to_remove_dir / f"ids_to_remove_{i}.parquet", index=False)
    return ids


def create_input_files(input_dir: Path, num_files: int = 100) -> tuple[pd.DataFrame, list[str]]:
    """Create input parquet files, each with 10 rows: 0-9, 10-19, ..., up to num_files*10-1."""
    input_dir.mkdir(parents=True, exist_ok=True)
    id_chunks = np.array_split(np.arange(num_files * 10), num_files)
    all_data = []
    for i, ids in enumerate(id_chunks):
        df = pd.DataFrame(
            {
                CURATOR_DEDUP_ID_STR: ids,
                "text": [f"Document {i}" for i in ids],
                "category": [f"category_{j % 10}" for j in ids],
                "score": [0.9 + (j % 10) * 0.01 for j in ids],
            }
        )
        df.to_parquet(input_dir / f"input_file_{i}.parquet", index=False)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True), [
        str(input_dir / f"input_file_{i}.parquet") for i in range(num_files)
    ]


@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param((RayDataExecutor, {}), id="ray_data"),
        pytest.param((XennaExecutor, {"execution_mode": "streaming"}), id="xenna"),
    ],
    indirect=True,
)
class TestTextDuplicateRemovalWorkflowIntegration:
    """Integration tests for TextDuplicatesRemovalWorkflow with different executors."""

    # Class attributes for shared test data
    executor_cls: type | None = None
    config: dict[str, Any] | None = None
    input_dir: Path | None = None
    ids_to_remove_dir: Path | None = None
    output_dir: Path | None = None
    expected_input_df: pd.DataFrame | None = None
    ids_to_remove: list[int] | None = None
    output_tasks: list[DocumentBatch] | None = None

    @pytest.fixture
    def test_config(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> "TestTextDuplicateRemovalWorkflowIntegration":
        """Set up test environment and execute the duplicate removal workflow."""
        executor_cls, config = request.param

        self.executor_cls = executor_cls
        self.config = config

        # Create temporary directories
        tmp_path = tmp_path_factory.mktemp("removal_workflow_integration")
        self.input_dir = tmp_path / "input"
        self.ids_to_remove_dir = tmp_path / "ids_to_remove"
        self.output_dir = tmp_path / "output"

        # Create test data
        self.expected_input_df, self.input_file_paths = create_input_files(self.input_dir, num_files=100)
        self.ids_to_remove = create_ids_to_remove_files(self.ids_to_remove_dir, num_files=5)

        # Create and run workflow
        workflow = TextDuplicatesRemovalWorkflow(
            input_path=str(self.input_dir),
            ids_to_remove_path=str(self.ids_to_remove_dir),
            output_path=str(self.output_dir),
            input_filetype="parquet",
            output_filetype="parquet",
            input_id_field=CURATOR_DEDUP_ID_STR,
            ids_to_remove_duplicate_id_field="id",
            input_kwargs={},
            ids_to_remove_read_kwargs={},
            output_kwargs={},
        )

        executor = executor_cls(config)
        self.output_tasks = workflow.run(executor)

        return self

    def test_output_correctness_and_files(self, test_config: "TestTextDuplicateRemovalWorkflowIntegration"):
        """Test output correctness and file system integrity."""
        assert test_config.output_tasks is not None
        assert test_config.expected_input_df is not None
        assert test_config.ids_to_remove is not None
        assert test_config.output_dir is not None
        assert test_config.output_dir.exists()

        # Verify output files exist and are readable
        all_output_files = []
        for task in test_config.output_tasks:
            all_output_files.extend(task.data)
        assert len(all_output_files) > 0

        # Combine all output data for comprehensive correctness check
        combined_output_df = pd.concat(
            [pd.read_parquet(task.data) for task in test_config.output_tasks], ignore_index=True
        )

        # Expected: all IDs from input that are NOT in ids_to_remove
        expected_ids = set(test_config.expected_input_df[CURATOR_DEDUP_ID_STR].tolist()) - set(
            test_config.ids_to_remove
        )
        actual_ids = set(combined_output_df[CURATOR_DEDUP_ID_STR].tolist())

        # Core correctness checks
        assert actual_ids == expected_ids
        assert len(combined_output_df) == 800  # 1000 total input - 200 divisible by 5

        # Verify no IDs divisible by 5 remain
        remaining_divisible_by_5 = [id_val for id_val in actual_ids if id_val % 5 == 0]
        assert len(remaining_divisible_by_5) == 0, f"Found IDs divisible by 5: {remaining_divisible_by_5}"

        # Verify expected columns are present
        expected_columns = {"text", "category", "score", CURATOR_DEDUP_ID_STR}
        assert set(combined_output_df.columns) == expected_columns

        # Verify individual files are valid parquet files with correct data
        total_records = 0
        for file_path in all_output_files:
            assert file_path.endswith(".parquet"), f"Expected parquet file, got {file_path}"
            df = pd.read_parquet(file_path)
            total_records += len(df)

            # Each file should have expected columns
            assert set(df.columns) == expected_columns

            # No IDs divisible by 5 in any individual file
            ids_in_file = df[CURATOR_DEDUP_ID_STR].tolist()
            divisible_by_5 = [id_val for id_val in ids_in_file if id_val % 5 == 0]
            assert len(divisible_by_5) == 0, f"File {file_path} contains IDs divisible by 5: {divisible_by_5}"

        # Total records across all files should match combined total
        assert total_records == 800

    def test_metadata_num_removed_consistency(self, test_config: "TestTextDuplicateRemovalWorkflowIntegration"):
        """Test that num_removed metadata sums up correctly across all tasks."""
        assert test_config.output_tasks is not None
        assert test_config.ids_to_remove is not None

        # Sum up all num_removed from task metadata
        total_removed_from_metadata = sum(task._metadata["num_removed"] for task in test_config.output_tasks)

        # Expected total removed based on actual workflow behavior
        # The workflow removes 2 records per input file (from 10 records to 8 records)
        # With 100 input files, total removed should be 200
        expected_total_removed = 200  # 2 * 100 files

        assert total_removed_from_metadata == expected_total_removed

        # Also verify by checking the actual difference in record counts
        total_input_records = len(test_config.expected_input_df)
        total_output_records = 0
        for task in test_config.output_tasks:
            for file_path in task.data:
                df = pd.read_parquet(file_path)
                total_output_records += len(df)

        actual_records_removed = total_input_records - total_output_records
        assert actual_records_removed == expected_total_removed

    def test_initial_tasks_partitioning(self, test_config: "TestTextDuplicateRemovalWorkflowIntegration"):
        """Test workflow with initial_tasks - should create 20 output tasks from 100 input files grouped 5 at a time."""
        assert test_config.input_file_paths is not None
        assert len(test_config.input_file_paths) == 100

        # Group input files: 5 files per FileGroupTask = 20 tasks total
        initial_tasks = []
        for i in range(0, len(test_config.input_file_paths), 5):
            task_files = test_config.input_file_paths[i : i + 5]
            initial_tasks.append(FileGroupTask(task_id=f"file_group_{i // 5}", dataset_name="input", data=task_files))

        assert len(initial_tasks) == 20  # 100 files / 5 per group = 20 tasks

        # Create workflow and run with initial_tasks
        workflow = TextDuplicatesRemovalWorkflow(
            input_path="",  # Not used when initial_tasks provided
            ids_to_remove_path=str(test_config.ids_to_remove_dir),
            output_path=str(
                test_config.output_dir / "initial_tasks_output"
            ),  # Different output dir to avoid conflicts
            input_filetype="parquet",
            output_filetype="parquet",
            input_id_field=CURATOR_DEDUP_ID_STR,
            ids_to_remove_duplicate_id_field="id",
            input_task_limit=10,  # truncate to 10 tasks only
            input_kwargs={},
            ids_to_remove_read_kwargs={},
            output_kwargs={},
        )

        executor = test_config.executor_cls(test_config.config)
        output_tasks = workflow.run(executor, initial_tasks=initial_tasks)

        # Verify we get 20 output tasks (one per input task)
        assert len(output_tasks) == 10, (
            f"Expected 10 output tasks, got {len(output_tasks)} for {test_config.executor_cls.__name__}"
        )

        # Verify correctness remains the same as other tests
        combined_output_df = pd.concat([pd.read_parquet(task.data) for task in output_tasks], ignore_index=True)
        assert len(combined_output_df) == 400, (
            f"Expected 400 records, got {len(combined_output_df)} for {test_config.executor_cls.__name__}"
        )

        # Verify no IDs divisible by 5 remain
        actual_ids = set(combined_output_df[CURATOR_DEDUP_ID_STR].tolist())
        remaining_divisible_by_5 = [id_val for id_val in actual_ids if id_val % 5 == 0]
        assert len(remaining_divisible_by_5) == 0, (
            f"Found IDs divisible by 5: {remaining_divisible_by_5} for {test_config.executor_cls.__name__}"
        )

        # Verify expected columns are present
        expected_columns = {"text", "category", "score", CURATOR_DEDUP_ID_STR}
        assert set(combined_output_df.columns) == expected_columns, (
            f"Column mismatch for {test_config.executor_cls.__name__}"
        )


class TestTextDuplicatesRemovalWorkflowGenerateStages:
    def test_invalid_filetypes(self):
        read_invalid_file_type_workflow = TextDuplicatesRemovalWorkflow(
            input_path="input_path",
            ids_to_remove_path="ids_to_remove_path",
            output_path="output_path",
            input_filetype="invalid",
            id_generator_path=None,
        )
        with pytest.raises(ValueError, match="Invalid input filetype: invalid"):
            read_invalid_file_type_workflow._generate_stages(initial_tasks=None)

        write_invalid_file_type_workflow = TextDuplicatesRemovalWorkflow(
            input_path="input_path",
            ids_to_remove_path="ids_to_remove_path",
            output_path="output_path",
            output_filetype="invalid",
            id_generator_path=None,
        )
        with pytest.raises(ValueError, match="Invalid output filetype: invalid"):
            write_invalid_file_type_workflow._generate_stages(initial_tasks=None)

    @pytest.mark.parametrize("input_filetype", ["parquet", "jsonl"])
    @pytest.mark.parametrize("id_generator_path", [None, "id_generator_path"])
    def test_reader_stage(self, input_filetype: str, id_generator_path: str | None):
        workflow = TextDuplicatesRemovalWorkflow(
            input_path="input_path",
            ids_to_remove_path="ids_to_remove_path",
            output_path="output_path",
            input_filetype=input_filetype,
            id_generator_path=id_generator_path,
            input_id_field=CURATOR_DEDUP_ID_STR,
        )

        stages = workflow._generate_stages(initial_tasks=None)
        assert len(stages) == 4
        # test for FilePartitioningStage
        assert isinstance(stages[0], FilePartitioningStage)
        assert stages[0].file_paths == "input_path"
        assert stages[0].files_per_partition is None
        assert stages[0].blocksize is None
        # post init of FilePartitioningStage sets this
        assert stages[0].file_extensions == [".jsonl", ".json", ".parquet"]
        assert stages[0].storage_options == {}

        # test for reader stage (stages[1])
        expected_reader_stage = ParquetReaderStage if input_filetype == "parquet" else JsonlReaderStage
        assert isinstance(stages[1], expected_reader_stage)
        assert stages[1].fields is None
        assert not stages[1]._generate_ids
        assert stages[1]._assign_ids == (id_generator_path is not None)

        # test for TextDuplicatesRemovalStage (stages[2])
        assert isinstance(stages[2], TextDuplicatesRemovalStage)
        assert stages[2].ids_to_remove_path == "ids_to_remove_path"
        # id_field is always CURATOR_DEDUP_ID_STR by default in the workflow
        assert stages[2].id_field == CURATOR_DEDUP_ID_STR
        assert stages[2].duplicate_id_field == "id"
        assert stages[2].read_kwargs == {}

        # test for writer stage (stages[3]) - default output_filetype is parquet
        assert isinstance(stages[3], ParquetWriter)

    @pytest.mark.parametrize("output_filetype", ["parquet", "jsonl"])
    def test_writer_stage(self, output_filetype: str):
        workflow = TextDuplicatesRemovalWorkflow(
            input_path="input_path",
            ids_to_remove_path="ids_to_remove_path",
            output_path="output_path",
            output_filetype=output_filetype,
            id_generator_path=None,
        )
        stages = workflow._generate_stages(initial_tasks=None)
        assert len(stages) == 4
        assert isinstance(stages[0], FilePartitioningStage)
        # reader stage
        assert isinstance(stages[1], ParquetReaderStage)  # Default input_filetype is parquet
        assert isinstance(stages[2], TextDuplicatesRemovalStage)
        expected_write_stage = ParquetWriter if output_filetype == "parquet" else JsonlWriter
        assert isinstance(stages[3], expected_write_stage)

        # Test the actual writer stage instance properties
        assert stages[3].path == "output_path"
        if output_filetype == "parquet":
            assert stages[3].file_extension == "parquet"
        else:
            assert stages[3].file_extension == "jsonl"

        assert stages[3].write_kwargs == {}
        assert stages[3].fields is None
        assert stages[3].mode == "ignore"

    def test_initial_tasks_required(self):
        workflow = TextDuplicatesRemovalWorkflow(
            input_path=None,
            ids_to_remove_path="ids_to_remove_path",
            output_path="output_path",
            input_filetype="parquet",
            id_generator_path=None,
        )
        with pytest.raises(ValueError, match="input_path is required when initial_tasks is None"):
            workflow._generate_stages(initial_tasks=None)
