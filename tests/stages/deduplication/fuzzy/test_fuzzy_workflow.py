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

from nemo_curator.stages.deduplication.fuzzy.identify_duplicates import DUPLICATE_IDS_SUBDIR
from nemo_curator.stages.deduplication.fuzzy.utils import (
    CURATOR_FUZZY_DUPLICATE_GROUP_FIELD,
)
from nemo_curator.stages.deduplication.fuzzy.workflow import (
    ID_GENERATOR_OUTPUT_FILENAME,
    FuzzyDeduplicationWorkflow,
)
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


def create_fuzzy_dedup_test_data() -> pd.DataFrame:
    """Create common test data for fuzzy dedup tests."""
    return pd.DataFrame(
        {
            "id": [1, 2, 300, 4, -1],
            "text": [
                "A test string",
                "A different test string",
                "A different object",
                "The quick brown fox jumps over the lazy dog",
                "The quick black cat jumps over the lazy dog",
            ],
            "content": [
                "A test string",
                "A different test string",
                "A different object",
                "The quick brown fox jumps over the lazy dog",
                "The quick black cat jumps over the lazy dog",
            ],
        }
    )


@pytest.fixture
def fuzzy_dedup_data_jsonl(tmp_path: Path) -> list[FileGroupTask]:
    """Create test data with duplicates and return file paths and FileGroupTasks."""
    df = create_fuzzy_dedup_test_data()

    file1 = tmp_path / "part1.jsonl"
    file2 = tmp_path / "part2.jsonl"

    df.iloc[:3].to_json(file1, orient="records", lines=True)
    df.iloc[3:].to_json(file2, orient="records", lines=True)

    files = [str(file1), str(file2)]
    return [
        FileGroupTask(
            task_id="file_group_0", dataset_name="test_dataset", data=files, _metadata={"source_files": files}
        )
    ]


@pytest.fixture
def fuzzy_dedup_data_parquet(tmp_path: Path) -> list[FileGroupTask]:
    """Create test data with duplicates and return file paths and FileGroupTasks."""
    df = create_fuzzy_dedup_test_data()

    file1 = tmp_path / "part1.parquet"
    file2 = tmp_path / "part2.parquet"

    df.iloc[:3].to_parquet(file1)
    df.iloc[3:].to_parquet(file2)

    files = [str(file1), str(file2)]
    return [
        FileGroupTask(
            task_id="file_group_0", dataset_name="test_dataset", data=files, _metadata={"source_files": files}
        )
    ]


@pytest.fixture
def no_duplicates_fuzzy_dedup_data(tmp_path: Path) -> list[FileGroupTask]:
    """Create test data with no duplicates."""
    data = {
        "id": [1, 2, 3, 4],
        "text": [
            "A test string",
            "Very different thing",
            "Something completely else that doesn't match",
            "The quick black cat jumps over the lazy dog",
        ],
    }
    df = pd.DataFrame(data)

    # Split into 2 files
    file1 = tmp_path / "part1.parquet"
    file2 = tmp_path / "part2.parquet"

    df.iloc[:2].to_parquet(file1)
    df.iloc[2:].to_parquet(file2)

    files = [str(file1), str(file2)]
    return [
        FileGroupTask(
            task_id="file_group_0", dataset_name="test_dataset", data=files, _metadata={"source_files": files}
        )
    ]


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestFuzzyDuplicates:
    @pytest.mark.parametrize("use_64_bit_hash", [False, True])
    @pytest.mark.parametrize(
        ("num_bands", "text_field", "duplicate_docs", "filetype"),
        [
            (5, "text", [[4, -1], [1, 2, 300]], "parquet"),
            (10, "content", [[4, -1], [1, 2, 300]], "jsonl"),
        ],
    )
    def test_fuzzy_dedup(  # noqa: PLR0913
        self,
        request: pytest.FixtureRequest,
        filetype: str,
        use_64_bit_hash: bool,
        num_bands: int,
        text_field: str,
        duplicate_docs: list[list[int]],
        tmp_path: Path,
    ) -> None:
        tasks = request.getfixturevalue(f"fuzzy_dedup_data_{filetype}")
        cache_path = tmp_path / "cache"
        output_path = tmp_path / "output"
        cache_path.mkdir(exist_ok=True)

        workflow = FuzzyDeduplicationWorkflow(
            cache_path=str(cache_path),
            output_path=str(output_path),
            input_filetype=filetype,
            text_field=text_field,
            perform_removal=False,
            seed=42,
            char_ngrams=5,
            num_bands=num_bands,
            minhashes_per_band=1,
            use_64_bit_hash=use_64_bit_hash,
            bands_per_iteration=5,
        )

        workflow.run(initial_tasks=tasks)

        # Verify the duplicate groups found match expected
        connected_components_df = cudf.read_parquet(cache_path / "ConnectedComponentsStage")
        original_df_with_curator_ids = get_original_df_with_curator_ids(
            output_path / ID_GENERATOR_OUTPUT_FILENAME, tasks, filetype
        )
        connected_components_df = connected_components_df.merge(
            original_df_with_curator_ids, on=CURATOR_DEDUP_ID_STR, how="left"
        )
        connected_components_df = connected_components_df[
            connected_components_df[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD].duplicated(keep=False)
        ]
        result_df = connected_components_df.groupby(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD).id.agg(list)
        result_df = result_df.list.sort_values()
        result_df = result_df.sort_values().to_arrow().to_pylist()
        assert all(
            set(got_group) == set(expected_group)
            for got_group, expected_group in zip(result_df, duplicate_docs, strict=False)
        )

        removal_ids_df = cudf.read_parquet(output_path / DUPLICATE_IDS_SUBDIR)
        removal_ids_df = removal_ids_df.merge(original_df_with_curator_ids, on=CURATOR_DEDUP_ID_STR, how="left")
        removal_ids = set(removal_ids_df.id.to_arrow().to_pylist())
        # For every duplicate group assert that 1 document was not removed
        assert all(len(set(expected_group) - removal_ids) == 1 for expected_group in duplicate_docs)

    def test_fuzzy_dedup_no_duplicates(
        self,
        no_duplicates_fuzzy_dedup_data: list[FileGroupTask],
        tmp_path: Path,
    ) -> None:
        tasks = no_duplicates_fuzzy_dedup_data
        cache_path = tmp_path / "cache"
        output_path = tmp_path / "output"
        cache_path.mkdir(exist_ok=True)

        workflow = FuzzyDeduplicationWorkflow(
            cache_path=str(cache_path),
            output_path=str(output_path),
            input_filetype="parquet",
            text_field="text",
            perform_removal=False,
            seed=42,
            char_ngrams=5,
            num_bands=10,
            minhashes_per_band=1,
            use_64_bit_hash=False,
            bands_per_iteration=10,
        )

        workflow.run(initial_tasks=tasks)

        assert not (cache_path / "ConnectedComponentsStage").exists()
        assert not (cache_path / "BucketsToEdgesStage").exists()
        assert not (output_path / DUPLICATE_IDS_SUBDIR).exists()

        lsh_df = cudf.read_parquet(cache_path / "LSHStage")
        assert len(lsh_df) == 0

    def test_bad_inputs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="bands_per_iteration must be between"):
            # bands_per_iteration must be between 1 and num_bands
            FuzzyDeduplicationWorkflow(
                input_path="/dummy",
                cache_path=str(tmp_path),
                output_path=str(tmp_path),
                bands_per_iteration=0,
            )

        with pytest.raises(ValueError, match="bands_per_iteration must be between"):
            # bands_per_iteration cannot exceed num_bands
            FuzzyDeduplicationWorkflow(
                input_path="/dummy",
                cache_path=str(tmp_path),
                output_path=str(tmp_path),
                num_bands=5,
                bands_per_iteration=10,
            )

        with pytest.raises(NotImplementedError, match="Removal is not implemented"):
            # Removal is not implemented yet
            FuzzyDeduplicationWorkflow(
                input_path="/dummy",
                cache_path=str(tmp_path),
                output_path=str(tmp_path),
                perform_removal=True,
            )

        workflow = FuzzyDeduplicationWorkflow(
            input_path="/dummy",
            cache_path=str(tmp_path),
            output_path=str(tmp_path),
        )
        create_id_generator_actor()
        with pytest.raises(RuntimeError, match="An existing id generator actor was found"):
            workflow.run()
        kill_id_generator_actor()

        workflow = FuzzyDeduplicationWorkflow(
            cache_path=str(tmp_path),
            output_path=str(tmp_path),
        )
        with pytest.raises(
            ValueError, match="input_path to the dataset must be provided if initial_tasks are not provided"
        ):
            workflow.run(initial_tasks=None)
