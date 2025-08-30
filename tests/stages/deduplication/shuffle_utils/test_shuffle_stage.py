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

import pytest

cudf = pytest.importorskip("cudf", reason="ShuffleStage tests require cudf")

from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.shuffle_utils.stage import ShuffleStage
from nemo_curator.tasks import FileGroupTask


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestShuffleStage:
    @pytest.fixture(autouse=True)
    def test_data(self, tmp_path: Path) -> list[FileGroupTask]:
        """Create test data with multiple files and overlapping categories for shuffling."""
        tasks = []

        df1 = cudf.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "category": ["A", "B", "A", "C", "B"],
                "value": [10, 20, 30, 40, 20],
                "text": [f"file1_text_{i}" for i in range(5)],
            }
        )

        df2 = cudf.DataFrame(
            {
                "id": [6, 7, 8, 9, 10],
                "category": ["B", "C", "D", "B", "A"],
                "value": [20, 40, 80, 90, 30],
                "text": [f"file2_text_{i}" for i in range(5)],
            }
        )

        df3 = cudf.DataFrame(
            {
                "id": [11, 12, 13, 14, 15],
                "category": ["A", "C", "D", "A", "B"],
                "value": [10, 40, 80, 30, 20],
                "text": [f"file3_text_{i}" for i in range(5)],
            }
        )

        for i, df in enumerate([df1, df2, df3]):
            test_file = os.path.join(tmp_path, f"test_data_{i}.parquet")
            df.to_parquet(test_file)

            tasks.append(
                FileGroupTask(
                    task_id=f"test_data_{i}",
                    dataset_name="test_dataset",
                    data=[test_file],
                    _metadata={
                        "partition_index": i,
                        "total_partitions": 3,
                        "source_files": [test_file],
                    },
                )
            )

        return tasks

    @pytest.fixture(autouse=True)
    def test_data_df(self, test_data: list[FileGroupTask]) -> cudf.DataFrame:
        """Create a dataframe from the test data."""
        return cudf.concat([cudf.read_parquet(task.data) for task in test_data], ignore_index=True)

    @pytest.mark.parametrize(
        ("shuffle_on", "total_nparts"),
        [
            (["category"], None),
            (["category"], 1),
            (["category", "value"], 3),
            # num_output_parititons > num_input_partitions
            (["category", "value"], 16),
        ],
    )
    def test_shuffle(
        self,
        test_data: list[FileGroupTask],
        test_data_df: cudf.DataFrame,
        shuffle_on: list[str],
        total_nparts: int | None,
        tmp_path: Path,
    ) -> None:
        """Test shuffling data on a single column."""

        shuffle_stage = ShuffleStage(
            shuffle_on=shuffle_on,
            total_nparts=total_nparts,
            output_path=str(tmp_path / "shuffle_output"),
        )

        pipeline = Pipeline(name="test_shuffle_single", stages=[shuffle_stage])
        executor = RayActorPoolExecutor()

        result_tasks = pipeline.run(executor, initial_tasks=test_data)

        expected_output_partitions = total_nparts if total_nparts is not None else len(test_data)
        assert len(result_tasks) == expected_output_partitions

        unique_keys_per_partition = {}
        for partition_id, task in enumerate(result_tasks):
            files = task.data if isinstance(task.data, list) else [task.data]
            for file in files:
                assert os.path.exists(file), f"File {file} does not exist"
                df = cudf.read_parquet(file)
                if len(df) > 0:
                    unique_keys_per_partition[partition_id] = len(df.drop_duplicates(subset=shuffle_on))
        total_unique_keys = sum(list(unique_keys_per_partition.values()))
        expected_unique_keys = len(test_data_df.drop_duplicates(subset=shuffle_on))
        assert total_unique_keys == expected_unique_keys, (
            f"Total unique keys {total_unique_keys} does not match expected {expected_unique_keys}"
        )

    def test_overwrite_existing_output_dir(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that ShuffleStage overwrites existing output directory."""
        output_path = str(tmp_path / "shuffle_overwrite")

        os.makedirs(output_path, exist_ok=True)
        dummy_file = os.path.join(output_path, "dummy_file.txt")
        with open(dummy_file, "w") as f:
            f.write("This file should be deleted")

        sub_dir = os.path.join(output_path, "old_partition")
        os.makedirs(sub_dir, exist_ok=True)
        old_file = os.path.join(sub_dir, "old_data.parquet")
        with open(old_file, "w") as f:
            f.write("Old data")

        assert os.path.exists(dummy_file), "Dummy file should exist before creating new ShuffleStage"
        assert os.path.exists(old_file), "Old file should exist before creating new ShuffleStage"

        ShuffleStage(
            shuffle_on=["id"],
            total_nparts=2,
            output_path=output_path,
        )

        assert not os.path.exists(dummy_file), "Dummy file should have been deleted during overwrite"
        assert not os.path.exists(old_file), "Old file should have been deleted during overwrite"
        assert not os.path.exists(sub_dir), "Old subdirectory should have been deleted during overwrite"

        assert os.path.exists(output_path), "Output directory should exist"

    def test_actor_not_initialized(
        self,
        test_data: list[FileGroupTask],
        tmp_path: Path,
    ) -> None:
        """Test that proper error is raised when actor object is not initialized."""
        shuffle_stage = ShuffleStage(
            shuffle_on=["id"],
            total_nparts=2,
            output_path=str(tmp_path / "shuffle_no_actor"),
        )

        with pytest.raises(RuntimeError, match="Actor object not initialized"):
            shuffle_stage.read_and_insert(test_data[0])

        with pytest.raises(RuntimeError, match="Actor object not initialized"):
            shuffle_stage.insert_finished()

        with pytest.raises(RuntimeError, match="Actor object not initialized"):
            shuffle_stage.extract_and_write()

        with pytest.raises(RuntimeError, match="Actor object not initialized"):
            shuffle_stage.teardown()

        with pytest.raises(NotImplementedError, match="ShufflerStage does not support the process method"):
            shuffle_stage.process(test_data[0])

        shuffle_stage._actor_obj = "not_an_actor"
        with pytest.raises(RuntimeError, match="Actor object not initialized"):
            shuffle_stage.read_and_insert(test_data[0])
