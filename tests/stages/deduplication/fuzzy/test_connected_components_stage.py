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
from typing import TYPE_CHECKING

import pandas as pd
import pytest

cudf = pytest.importorskip("cudf", reason="ConnectedComponentsStage tests require cudf")
cugraph = pytest.importorskip("cugraph", reason="ConnectedComponentsStage tests require cugraph")

from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.fuzzy.connected_components import ConnectedComponentsStage
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.tasks import FileGroupTask

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_edge_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample edge data that mimics buckets_to_edges output."""
    data1 = pd.DataFrame(
        {
            f"{CURATOR_DEDUP_ID_STR}_x": [10, 11, 12, 13, 14, 15, 11],
            f"{CURATOR_DEDUP_ID_STR}_y": [11, 12, 13, 120, 15, 110, 12],
        }
    )

    data2 = pd.DataFrame(
        {
            f"{CURATOR_DEDUP_ID_STR}_x": [14, 15, 16, 17, 18, 19, 110],
            f"{CURATOR_DEDUP_ID_STR}_y": [15, 16, 17, 18, 19, 110, 111],
        }
    )
    return data1, data2


@pytest.fixture
def sample_files(tmp_path: "Path", sample_edge_data: tuple[pd.DataFrame, pd.DataFrame]) -> list[str]:
    """Create sample parquet files with edge data."""
    data1, data2 = sample_edge_data

    input_dir = tmp_path / "input_edges"
    input_dir.mkdir(exist_ok=True)

    file1 = str(input_dir / "edges_batch_0.parquet")
    file2 = str(input_dir / "edges_batch_1.parquet")

    data1.to_parquet(file1, index=False)
    data2.to_parquet(file2, index=False)

    return [file1, file2]


@pytest.fixture
def input_tasks(sample_files: list[str]) -> list[FileGroupTask]:
    """Create FileGroupTasks with multiple files."""
    return [
        FileGroupTask(
            dataset_name="test_edges",
            task_id="edge_group_0",
            data=[sample_files[0]],
        ),
        FileGroupTask(
            dataset_name="test_edges",
            task_id="edge_group_1",
            data=[sample_files[1]],
        ),
    ]


@pytest.mark.gpu
class TestConnectedComponentsStage:
    """Test suite for ConnectedComponentsStage ProcessingStage."""

    @pytest.mark.usefixtures("shared_ray_client")
    def test_single_file_group_processing(self, tmp_path: "Path", input_tasks: list[FileGroupTask]) -> None:
        """Test processing a single file group task using Ray cluster."""
        output_dir = str(tmp_path / "cc_output_single")

        pipeline = Pipeline(name="test_connected_components_single")
        pipeline.add_stage(
            ConnectedComponentsStage(
                output_path=output_dir,
            )
        )
        executor = RayActorPoolExecutor()
        output_tasks = pipeline.run(executor=executor, initial_tasks=input_tasks[:1])

        assert len(output_tasks) == 1
        output_task = output_tasks[0]

        output_file = output_task.data[0]
        assert os.path.exists(output_file)

        result_df = pd.read_parquet(output_file)
        grouped_df = result_df.groupby("_duplicate_group_id")[CURATOR_DEDUP_ID_STR].agg(list)
        assert len(result_df) == 8
        assert len(grouped_df) == 2
        found_docs = {tuple(sorted(group)) for group in grouped_df.tolist()}
        assert found_docs == {(10, 11, 12, 13, 120), (14, 15, 110)}

    @pytest.mark.usefixtures("shared_ray_client")
    def test_multiple_file_groups_processing(self, tmp_path: "Path", input_tasks: list[FileGroupTask]) -> None:
        """Test processing multiple file group tasks using Ray cluster."""
        output_dir = str(tmp_path / "cc_output_multiple")

        pipeline = Pipeline(name="test_connected_components_multiple")
        pipeline.add_stage(
            ConnectedComponentsStage(
                output_path=output_dir,
            )
        )

        executor = RayActorPoolExecutor()
        output_tasks = pipeline.run(executor=executor, initial_tasks=input_tasks)

        assert len(output_tasks) == 2

        all_results = []
        for output_task in output_tasks:
            assert len(output_task.data) == 1
            output_files = output_task.data
            result_df = pd.read_parquet(output_files)
            all_results.append(result_df)

        result_df = pd.concat(all_results, ignore_index=True)
        assert len(result_df) == 13
        grouped_df = result_df.groupby("_duplicate_group_id")[CURATOR_DEDUP_ID_STR].agg(list)
        found_docs = {tuple(sorted(group)) for group in grouped_df.tolist()}
        assert found_docs == {(10, 11, 12, 13, 120), (14, 15, 16, 17, 18, 19, 110, 111)}

    def test_output_directory_cleanup(self, tmp_path: "Path") -> None:
        """Test that output directory is cleaned up if it exists."""
        output_dir = tmp_path / "cc_output_cleanup"
        output_dir.mkdir(exist_ok=True)

        dummy_file = output_dir / "ConnectedComponentsStage" / "dummy.txt"
        dummy_file.parent.mkdir(exist_ok=True)
        dummy_file.write_text("dummy content")

        assert dummy_file.exists()

        pipeline = Pipeline(name="test_cleanup")
        pipeline.add_stage(
            ConnectedComponentsStage(
                output_path=str(output_dir),
            )
        )
        assert not dummy_file.exists()

    def test_process_not_implemented(self, tmp_path: "Path", input_tasks: list[FileGroupTask]) -> None:
        """Test that process method raises NotImplementedError."""
        stage = ConnectedComponentsStage(
            output_path=str(tmp_path / "output"),
        )

        with pytest.raises(NotImplementedError, match="only support process batch"):
            stage.process(input_tasks[0])
