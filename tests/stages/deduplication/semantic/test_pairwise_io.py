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
from pathlib import Path

import pytest

cudf = pytest.importorskip("cudf")
cupy = pytest.importorskip("cupy")

from nemo_curator.stages.deduplication.semantic.pairwise_io import ClusterWiseFilePartitioningStage
from nemo_curator.tasks import FileGroupTask, _EmptyTask


@pytest.mark.gpu  # TODO : Remove this once we figure out how to import semantic on CPU
class TestClusterWiseFilePartitioningStage:
    """Test cases for ClusterWiseFilePartitioningStage."""

    def test_process_finds_all_centroid_files(self, tmp_path: Path):
        """Test that process method finds all files in centroid directories."""

        # Create centroid directories with parquet files
        centroid_0_dir = tmp_path / "centroid=0"
        centroid_1_dir = tmp_path / "centroid=1"
        centroid_2_dir = tmp_path / "centroid=2"

        centroid_0_dir.mkdir()
        centroid_1_dir.mkdir()
        centroid_2_dir.mkdir()

        # Create some parquet files in each centroid
        (centroid_0_dir / "file1.parquet").write_text("data1")
        (centroid_0_dir / "file2.parquet").write_text("data2")
        (centroid_1_dir / "file3.parquet").write_text("data3")
        (centroid_2_dir / "file4.parquet").write_text("data4")
        (centroid_2_dir / "file5.parquet").write_text("data5")

        # Create some non-centroid directories
        (tmp_path / "other_dir").mkdir()
        (tmp_path / "other_dir" / "file6.parquet").write_text("data6")
        # Write some file at tmp_path
        (tmp_path / "file7.parquet").write_text("data7")
        (tmp_path / "file8.jsonl").write_text("data8")

        # Test the stage
        stage = ClusterWiseFilePartitioningStage(str(tmp_path))
        stage.setup()

        empty_task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        result = stage.process(empty_task)

        # Should create 3 FileGroupTasks for 3 centroids
        assert len(result) == 3
        assert all(isinstance(task, FileGroupTask) for task in result)

        # Sort by centroid_id for consistent testing
        result.sort(key=lambda x: x._metadata["centroid_id"])

        # Check each task
        assert result[0].task_id == "pairwise_centroid_0"
        assert result[0]._metadata == {"centroid_id": 0, "filetype": "parquet"}
        assert result[0].data == [str(centroid_0_dir / "file1.parquet"), str(centroid_0_dir / "file2.parquet")]

        assert result[1].task_id == "pairwise_centroid_1"
        assert result[1]._metadata == {"centroid_id": 1, "filetype": "parquet"}
        assert result[1].data == [str(centroid_1_dir / "file3.parquet")]

        assert result[2].task_id == "pairwise_centroid_2"
        assert result[2]._metadata == {"centroid_id": 2, "filetype": "parquet"}
        assert result[2].data == [str(centroid_2_dir / "file4.parquet"), str(centroid_2_dir / "file5.parquet")]
