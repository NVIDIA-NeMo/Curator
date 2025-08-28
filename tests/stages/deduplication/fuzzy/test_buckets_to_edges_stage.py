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
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.tasks import FileGroupTask

BucketsToEdgesStage = pytest.importorskip(
    "nemo_curator.stages.deduplication.fuzzy.buckets_to_edges"
).BucketsToEdgesStage


@pytest.fixture
def sample_bucket_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample bucket data that mimics LSH output with overlapping document IDs."""
    data1 = pd.DataFrame(
        {
            "_bucket_id": ["bucket_1", "bucket_2", "bucket_3", "bucket_4"],
            CURATOR_DEDUP_ID_STR: [
                [0, 1, 2],
                [1, 3, 4],
                [5],
                [2, 6, 7, 8],
            ],
        }
    )

    data2 = pd.DataFrame(
        {
            "_bucket_id": ["bucket_5", "bucket_6", "bucket_7"],
            CURATOR_DEDUP_ID_STR: [
                [3, 4, 10, 11],
                [7, 12, 13],
                [11, 15, 16],
            ],
        }
    )

    return data1, data2


@pytest.fixture
def sample_files(tmp_path: Path, sample_bucket_data: tuple[pd.DataFrame, pd.DataFrame]) -> list[str]:
    """Create sample parquet files with bucket data."""
    data1, data2 = sample_bucket_data

    file1 = tmp_path / "buckets1.parquet"
    file2 = tmp_path / "buckets2.parquet"

    table1 = pa.Table.from_pandas(data1)
    table2 = pa.Table.from_pandas(data2)

    pq.write_table(table1, file1)
    pq.write_table(table2, file2)

    return [str(file1), str(file2)]


@pytest.fixture
def input_task(sample_files: list[str]) -> FileGroupTask:
    """Create a FileGroupTask from sample files."""
    return FileGroupTask(
        task_id="test_task",
        dataset_name="test_buckets",
        data=sample_files,
        _metadata={"batch_id": 0, "total_batches": 1},
    )


# Marking as GPU so that they don't get skiped on GPU CI runs
@pytest.mark.gpu
class TestBucketsToEdgesStage:
    """Test suite for BucketsToEdgesStage ProcessingStage."""

    def test_basic_edge_creation(self, input_task: FileGroupTask, tmp_path: Path) -> None:
        """Test basic edge creation from bucket data."""
        stage = BucketsToEdgesStage(
            output_path=str(tmp_path / "output"),
            doc_id_field=CURATOR_DEDUP_ID_STR,
        )

        # Process the task
        output_task = stage.process(input_task)

        # Verify output task structure
        assert isinstance(output_task, FileGroupTask)
        assert len(output_task.data) == 1
        assert output_task.dataset_name == "test_buckets_edges"
        assert "storage_options" in output_task._metadata

        # Verify output file exists
        output_file = output_task.data[0]
        assert os.path.exists(output_file)

        result_df = pd.read_parquet(output_file)

        assert f"{CURATOR_DEDUP_ID_STR}_x" in result_df.columns
        assert f"{CURATOR_DEDUP_ID_STR}_y" in result_df.columns
        assert len(result_df.columns) == 2

        expected_edges = [
            (0, 1),
            (1, 2),
            (1, 3),
            (3, 4),
            (2, 6),
            (6, 7),
            (7, 8),
            (3, 4),
            (4, 10),
            (10, 11),
            (7, 12),
            (12, 13),
            (11, 15),
            (15, 16),
        ]

        assert len(result_df) == len(expected_edges)

        edges_list = list(
            zip(result_df[f"{CURATOR_DEDUP_ID_STR}_x"], result_df[f"{CURATOR_DEDUP_ID_STR}_y"], strict=False)
        )
        for edge in expected_edges:
            assert edge in edges_list

    def test_custom_column_name(self, tmp_path: Path) -> None:
        """Test edge creation with custom document ID column."""
        # Create sample data with custom column
        custom_data = pd.DataFrame(
            {
                "_bucket_id": ["bucket_a", "bucket_b", "bucket_c"],
                "custom_doc_id": [
                    ["doc_1", "doc_2", "doc_3"],
                    ["doc_4", "doc_5"],
                    ["doc_6"],
                ],
            }
        )

        file = tmp_path / "buckets_custom.parquet"
        table = pa.Table.from_pandas(custom_data)
        pq.write_table(table, file)

        input_task = FileGroupTask(
            task_id="test_task_custom",
            dataset_name="test_buckets_custom",
            data=[str(file)],
            _metadata={"batch_id": 0, "total_batches": 1},
        )

        stage = BucketsToEdgesStage(
            output_path=str(tmp_path / "output"),
            doc_id_field="custom_doc_id",
        )

        output_task = stage.process(input_task)

        output_file = output_task.data[0]
        assert os.path.exists(output_file)

        result_df = pd.read_parquet(output_file)

        assert "custom_doc_id_x" in result_df.columns
        assert "custom_doc_id_y" in result_df.columns

        expected_edges = [
            ("doc_1", "doc_2"),
            ("doc_2", "doc_3"),
            ("doc_4", "doc_5"),
        ]

        assert len(result_df) == len(expected_edges)

        actual_edges = set(zip(result_df["custom_doc_id_x"], result_df["custom_doc_id_y"], strict=False))
        expected_edges_set = set(expected_edges)

        assert actual_edges == expected_edges_set

    def test_empty_input_handling(self, tmp_path: Path) -> None:
        """Test handling of empty input files."""
        empty_data = pd.DataFrame({"_bucket_id": [], CURATOR_DEDUP_ID_STR: []})

        input_file = tmp_path / "empty_buckets.parquet"
        table = pa.Table.from_pandas(empty_data)
        pq.write_table(table, input_file)

        input_task = FileGroupTask(
            task_id="empty_test",
            dataset_name="empty_buckets",
            data=[str(input_file)],
            _metadata={},
        )

        stage = BucketsToEdgesStage(
            output_path=str(tmp_path / "output"),
            doc_id_field=CURATOR_DEDUP_ID_STR,
        )

        output_task = stage.process(input_task)

        result_df = pd.read_parquet(output_task.data[0])
        assert len(result_df) == 0
        assert f"{CURATOR_DEDUP_ID_STR}_x" in result_df.columns
        assert f"{CURATOR_DEDUP_ID_STR}_y" in result_df.columns

    def test_single_document_buckets(self, tmp_path: Path) -> None:
        """Test handling of buckets with only single documents (no edges)."""
        single_doc_data = pd.DataFrame(
            {
                "_bucket_id": ["bucket_1", "bucket_2", "bucket_3"],
                CURATOR_DEDUP_ID_STR: [
                    [100],
                    [200],
                    [300],
                ],
            }
        )

        input_file = tmp_path / "single_doc_buckets.parquet"
        table = pa.Table.from_pandas(single_doc_data)
        pq.write_table(table, input_file)

        input_task = FileGroupTask(
            task_id="single_doc_test",
            dataset_name="single_doc_buckets",
            data=[str(input_file)],
            _metadata={},
        )

        stage = BucketsToEdgesStage(
            output_path=str(tmp_path / "output"),
            doc_id_field=CURATOR_DEDUP_ID_STR,
        )

        output_task = stage.process(input_task)

        result_df = pd.read_parquet(output_task.data[0])
        assert len(result_df) == 0

    def test_large_buckets(self, tmp_path: Path) -> None:
        """Test handling of large buckets with many documents."""
        large_bucket_data = pd.DataFrame(
            {
                "_bucket_id": ["large_bucket", "small_bucket"],
                CURATOR_DEDUP_ID_STR: [
                    list(range(100)),
                    [1000, 1001],
                ],
            }
        )

        input_file = tmp_path / "large_buckets.parquet"
        table = pa.Table.from_pandas(large_bucket_data)
        pq.write_table(table, input_file)

        input_task = FileGroupTask(
            task_id="large_bucket_test",
            dataset_name="large_buckets",
            data=[str(input_file)],
            _metadata={},
        )

        stage = BucketsToEdgesStage(
            output_path=str(tmp_path / "output"),
            doc_id_field=CURATOR_DEDUP_ID_STR,
        )

        output_task = stage.process(input_task)

        result_df = pd.read_parquet(output_task.data[0])

        assert len(result_df) == 100

        edges_set = set(
            zip(result_df[f"{CURATOR_DEDUP_ID_STR}_x"], result_df[f"{CURATOR_DEDUP_ID_STR}_y"], strict=False)
        )
        assert (0, 1) in edges_set
        assert (98, 99) in edges_set
        assert (1000, 1001) in edges_set

    def test_output_directory_cleanup(self, input_task: FileGroupTask, tmp_path: Path) -> None:
        """Test that existing output directory is cleaned up."""
        output_dir = tmp_path / "output"

        existing_dir = output_dir / "BucketsToEdgesStage"
        existing_dir.mkdir(parents=True)
        existing_file = existing_dir / "existing.txt"
        existing_file.write_text("This should be deleted")

        assert existing_file.exists()

        stage = BucketsToEdgesStage(
            output_path=str(output_dir),
            doc_id_field=CURATOR_DEDUP_ID_STR,
        )

        assert not existing_file.exists()

        output_task = stage.process(input_task)
        assert os.path.exists(output_task.data[0])
