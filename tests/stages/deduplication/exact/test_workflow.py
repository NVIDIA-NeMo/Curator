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
from typing import Literal

import pytest

cudf = pytest.importorskip("cudf")
import numpy as np
import pandas as pd

from nemo_curator.stages.deduplication.exact.workflow import ID_GENERATOR_OUTPUT_FILENAME, ExactDeduplicationWorkflow
from nemo_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
    IdGeneratorBase,
    create_id_generator_actor,
    kill_id_generator_actor,
)
from nemo_curator.tasks import FileGroupTask


def get_original_df_with_curator_ids(
    id_generator_path: str, tasks: list[FileGroupTask], filetype: Literal["parquet", "jsonl"]
) -> cudf.DataFrame:
    """Get mapping from curator IDs to original IDs using IDGeneratorActor.

    Args:
        files: List of parquet files that were processed
        id_column: Name of the original ID column in the data

    Returns:
        Dictionary mapping curator_dedup_id -> original_id
    """
    id_generator = IdGeneratorBase.from_disk(id_generator_path)
    dfs = []
    for task in tasks:
        min_id, max_id = id_generator.get_batch_range(task.data, None)
        df = cudf.read_parquet(task.data) if filetype == "parquet" else cudf.read_json(task.data, lines=True)
        df[CURATOR_DEDUP_ID_STR] = np.arange(min_id, max_id + 1)
        dfs.append(df)

    return cudf.concat(dfs)


@pytest.fixture
def exact_dedup_data_parquet(tmp_path: Path) -> list[FileGroupTask]:
    df1 = pd.DataFrame({"id": [1, 2, 300], "text": ["Small String", "Large String", "Medium String"]})
    df2 = pd.DataFrame({"id": [4, -1], "text": ["Large String", "Small String"]})

    file1 = tmp_path / "data_part1.parquet"
    file2 = tmp_path / "data_part2.parquet"

    df1.to_parquet(file1)
    df2.to_parquet(file2)

    return [
        FileGroupTask(
            task_id="exact_dedup_0",
            dataset_name="exact_dedup_dataset",
            data=[str(file1)],
            _metadata={
                "partition_index": 0,
                "total_partitions": 2,
                "source_files": [str(file1)],
            },
        ),
        FileGroupTask(
            task_id="exact_dedup_1",
            dataset_name="exact_dedup_dataset",
            data=[str(file2)],
            _metadata={
                "partition_index": 1,
                "total_partitions": 2,
                "source_files": [str(file2)],
            },
        ),
    ]


@pytest.fixture
def exact_no_dedup_data_jsonl(tmp_path: Path) -> list[FileGroupTask]:
    df = pd.DataFrame({"id": [1, 2, 300], "content": ["abc", "aba", "abb"]})

    file1 = tmp_path / "no_dedup_data.jsonl"
    df.to_json(file1, orient="records", lines=True)

    return [
        FileGroupTask(
            task_id="no_dedup_0",
            dataset_name="no_dedup_dataset",
            data=[str(file1)],
            _metadata={
                "partition_index": 0,
                "total_partitions": 1,
                "source_files": [str(file1)],
            },
        ),
    ]


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestExactDuplicatesWorkflow:
    @pytest.mark.parametrize("assign_id", [True, False])
    def test_dup(self, exact_dedup_data_parquet: list[FileGroupTask], tmpdir: Path, assign_id: bool) -> None:
        workflow = ExactDeduplicationWorkflow(
            output_path=str(tmpdir),
            input_filetype="parquet",
            assign_id=assign_id,
            id_field="id" if not assign_id else None,
            text_field="text",
            perform_removal=False,
        )
        workflow.run(initial_tasks=exact_dedup_data_parquet)

        original_df_with_curator_ids = (
            get_original_df_with_curator_ids(
                id_generator_path=str(tmpdir / ID_GENERATOR_OUTPUT_FILENAME),
                tasks=exact_dedup_data_parquet,
                filetype="parquet",
            )
            if assign_id
            else None
        )

        removal_ids_df = cudf.read_parquet(tmpdir / "ExactDuplicateIds")
        removal_ids_df = (
            removal_ids_df.merge(original_df_with_curator_ids, on=CURATOR_DEDUP_ID_STR, how="left")
            if assign_id
            else removal_ids_df
        )
        removal_ids = set(removal_ids_df.id.to_arrow().to_pylist())
        duplicate_docs = [{1, -1}, {2, 4}]
        # For every duplicate group assert that 1 document was not removed
        assert all(len(expected_group - removal_ids) == 1 for expected_group in duplicate_docs)

    def test_no_dedup(self, exact_no_dedup_data_jsonl: list[FileGroupTask], tmpdir: Path) -> None:
        workflow = ExactDeduplicationWorkflow(
            output_path=str(tmpdir),
            input_filetype="jsonl",
            assign_id=True,
            text_field="content",
            perform_removal=False,
            input_path=str(tmpdir),
        )
        workflow.run(initial_tasks=exact_no_dedup_data_jsonl)

        removal_ids_df = cudf.read_parquet(tmpdir / "ExactDuplicateIds")
        assert len(removal_ids_df) == 0

    def test_bad_inputs(self, tmpdir: Path) -> None:
        with pytest.raises(NotImplementedError, match="Removal is not implemented"):
            # Removal is not implemented yet
            ExactDeduplicationWorkflow(
                input_path="/dummy",
                output_path=str(tmpdir),
                perform_removal=True,
            )

        workflow = ExactDeduplicationWorkflow(
            input_path="/dummy",
            output_path=str(tmpdir),
            assign_id=True,
        )
        create_id_generator_actor()
        with pytest.raises(RuntimeError, match="An existing id generator actor was found"):
            workflow.run()
        kill_id_generator_actor()

        workflow = ExactDeduplicationWorkflow(
            output_path=str(tmpdir),
        )
        with pytest.raises(
            ValueError, match="input_path to the dataset must be provided if initial_tasks are not provided"
        ):
            workflow.run(initial_tasks=None)
