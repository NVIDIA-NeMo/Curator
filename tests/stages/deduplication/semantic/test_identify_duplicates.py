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

# ruff: noqa: E402
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from nemo_curator.tasks import FileGroupTask

cudf = pytest.importorskip("cudf")
cuml = pytest.importorskip("cuml")
cp = pytest.importorskip("cupy")


from nemo_curator.stages.deduplication.semantic.identify_duplicates import IdentifyDuplicatesStage


class TestIdentifyDuplicatesStage:
    """Test cases for IdentifyDuplicatesStage."""

    def create_test_similarity_file(self, file_path: str, data: dict) -> None:
        """Create a test similarity file with the given data."""
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)

    def verify_output_file(
        self, output_file: str, expected_count: int, expected_row_groups: int | None = None
    ) -> None:
        """Verify output file has correct content, sorting, and row groups."""
        assert os.path.exists(output_file)

        # Read and verify results
        result_df = pd.read_parquet(output_file)
        assert len(result_df) == expected_count

        # Verify IDs are sorted
        if expected_count > 0:
            assert result_df["id"].tolist() == sorted(result_df["id"].tolist())

            parquet_file = pq.ParquetFile(output_file)
            num_row_groups = parquet_file.num_row_groups

            # If expected_row_groups is not specified, calculate it based on data size
            if expected_row_groups is None:
                # Calculate expected row groups: max(1, num_rows // 10)
                # This matches the logic in IdentifyDuplicatesStage
                expected_row_groups = max(1, expected_count // 10)

            assert num_row_groups == expected_row_groups, (
                f"Expected {expected_row_groups} row groups, got {num_row_groups}"
            )

    def test_identify_duplicates_various_cases(self, tmp_path: Path) -> None:
        """Test basic functionality, edge cases, and different epsilon values in one comprehensive test."""
        output_dir = tmp_path / "identify_duplicates_output"
        output_dir.mkdir()

        # Test 1: Empty input handling
        stage = IdentifyDuplicatesStage(output_path=str(output_dir), eps=0.1, verbose=True)
        result_tasks = stage.process_batch([])
        assert len(result_tasks) == 0, "Empty input should return empty result"

        # Test 2: Single item cluster (should create empty result)
        single_file = tmp_path / "single_item.parquet"
        single_data = {
            "id": ["doc1"],
            "max_id": ["doc1"],
            "cosine_sim_score": [0.0],  # Self-similarity is 0
        }
        self.create_test_similarity_file(str(single_file), single_data)

        task_single = FileGroupTask(task_id="single", dataset_name="test", data=[str(single_file)])
        result_single = stage.process_batch([task_single])
        assert len(result_single) == 1
        result_df = pd.read_parquet(result_single[0].data[0])
        assert len(result_df) == 0, "Single item should create empty result"

        # Test 3: No similar items (all below threshold)
        no_similar_file = tmp_path / "no_similar.parquet"
        no_similar_data = {
            "id": ["doc1", "doc2", "doc3"],
            "max_id": ["doc2", "doc3", "doc1"],
            "cosine_sim_score": [0.5, 0.6, 0.7],  # All below 0.9 threshold
        }
        self.create_test_similarity_file(str(no_similar_file), no_similar_data)

        task_no_similar = FileGroupTask(task_id="no_similar", dataset_name="test", data=[str(no_similar_file)])
        result_no_similar = stage.process_batch([task_no_similar])
        assert len(result_no_similar) == 1
        result_df = pd.read_parquet(result_no_similar[0].data[0])
        assert len(result_df) == 0, "No items should meet similarity threshold"

        # Test 4: Different epsilon values
        eps_test_file = tmp_path / "eps_test.parquet"
        eps_test_data = {
            "id": ["doc1", "doc2", "doc3", "doc4"],
            "max_id": ["doc2", "doc1", "doc4", "doc3"],
            "cosine_sim_score": [0.98, 0.98, 0.85, 0.85],
        }
        self.create_test_similarity_file(str(eps_test_file), eps_test_data)

        # Strict epsilon (0.01) - threshold = 0.99, should get 0 results
        stage_strict = IdentifyDuplicatesStage(output_path=str(output_dir), eps=0.01, verbose=True)
        task_eps = FileGroupTask(task_id="eps_test", dataset_name="test", data=[str(eps_test_file)])
        result_strict = stage_strict.process_batch([task_eps])
        result_df = pd.read_parquet(result_strict[0].data[0])
        assert len(result_df) == 0, "Strict epsilon should return 0 results"

        # Permissive epsilon (0.2) - threshold = 0.8, should get 4 results
        stage_permissive = IdentifyDuplicatesStage(output_path=str(output_dir), eps=0.2, verbose=True)
        result_permissive = stage_permissive.process_batch([task_eps])
        # With 4 items, we can't have 10 row groups (max(1, 4//10) = 1)
        # So we'll just verify the content without checking row groups
        output_file = result_permissive[0].data[0]
        assert os.path.exists(output_file)
        result_df = pd.read_parquet(output_file)
        assert len(result_df) == 4
        assert result_df["id"].tolist() == sorted(result_df["id"].tolist())

        # Test 5: Basic functionality with mixed similarity scores
        basic_file = tmp_path / "basic_functionality.parquet"
        basic_data = {
            "id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "max_id": ["doc2", "doc1", "doc4", "doc3", "doc1"],
            "cosine_sim_score": [0.95, 0.95, 0.85, 0.85, 0.75],  # Some above/below threshold
        }
        self.create_test_similarity_file(str(basic_file), basic_data)

        task_basic = FileGroupTask(task_id="basic", dataset_name="test", data=[str(basic_file)])
        result_basic = stage.process_batch([task_basic])
        assert len(result_basic) == 1
        output_file = result_basic[0].data[0]

        # Verify content, sorting, and row groups
        # With 2 rows, we expect max(1, 2//10) = 1 sized row group, but since each row group can only have 1 row,
        # we'll get 2 row groups total
        self.verify_output_file(output_file, 2, expected_row_groups=2)

        # Verify metadata
        assert "num_removed" in result_basic[0]._metadata
        assert result_basic[0]._metadata["num_removed"] == 2

    def test_identify_duplicates_stage_batch_processing(self, tmp_path: Path) -> None:
        """Test batch processing of multiple clusters."""
        # Create test data for multiple clusters
        cluster_files = []
        for cluster_id in range(10):
            input_file = tmp_path / f"cluster_{cluster_id}.parquet"
            test_data = {
                "id": [f"doc{cluster_id}_1", f"doc{cluster_id}_2", f"doc{cluster_id}_3"],
                "max_id": [f"doc{cluster_id}_2", f"doc{cluster_id}_1", f"doc{cluster_id}_1"],
                "cosine_sim_score": [0.95, 0.95, 0.85],  # First two above threshold
            }
            self.create_test_similarity_file(str(input_file), test_data)
            cluster_files.append(str(input_file))

        output_dir = tmp_path / "identify_duplicates_output"
        output_dir.mkdir()
        stage = IdentifyDuplicatesStage(
            output_path=str(output_dir),
            eps=0.1,  # threshold = 0.9
            verbose=True,
        )

        # Create tasks for each cluster
        tasks = []
        for i, file_path in enumerate(cluster_files):
            task = FileGroupTask(
                task_id=f"test_batch_{i}",
                dataset_name="test",
                data=[file_path],
            )
            tasks.append(task)

        # Process batch
        result_tasks = stage.process_batch(tasks)
        assert len(result_tasks) == 1  # Should combine into one output task

        # Check combined results with verification
        # 10 clusters * 2 items above threshold each = 20 total
        # With 20 rows and _num_row_groups_hint=10: row_group_size = max(1, 20//10) = 2
        # This creates 20//2 = 10 row groups
        self.verify_output_file(result_tasks[0].data[0], 20, expected_row_groups=10)

        # Check metadata
        assert "num_removed" in result_tasks[0]._metadata
        assert result_tasks[0]._metadata["num_removed"] == 20

    def test_identify_duplicates_stage_custom_row_groups(self, tmp_path: Path) -> None:
        """Test custom row group configuration."""
        input_file = tmp_path / "cluster_custom_row_groups.parquet"
        test_data = {
            "id": [f"doc{i}" for i in range(50)],  # 50 documents
            "max_id": [f"doc{(i + 1) % 50}" for i in range(50)],
            "cosine_sim_score": [0.95] * 50,  # All above threshold
        }
        self.create_test_similarity_file(str(input_file), test_data)

        output_dir = tmp_path / "identify_duplicates_output"
        output_dir.mkdir()
        # Test with custom row group hint
        stage = IdentifyDuplicatesStage(
            output_path=str(output_dir),
            eps=0.1,
            _num_row_groups_hint=5,  # Should create 5 row groups
            verbose=True,
        )

        task = FileGroupTask(
            task_id="test_custom_row_groups",
            dataset_name="test",
            data=[str(input_file)],
        )

        result_tasks = stage.process_batch([task])
        output_file = result_tasks[0].data[0]

        # Verify content and sorting
        # With 50 rows and _num_row_groups_hint=5: row_group_size = max(1, 50//5) = 10
        # This creates 50//10 = 5 row groups
        self.verify_output_file(output_file, 50, expected_row_groups=5)
