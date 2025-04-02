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

from typing import Literal

import pandas as pd
import pytest
from dask import dataframe as dd

from nemo_curator.utils.duplicates_removal import remove_duplicates


@pytest.fixture()
def ids():
    # Dataset has id a0...a9, b0...b9, c0...c9, d0...d9
    l = [f"{group}{i}" for group in ["a", "b", "c", "d"] for i in range(10)]
    return l


@pytest.fixture
def sample_data(ids):
    df = pd.DataFrame(
        {
            "id": ids,
            "text": [f"text for {_id}" for _id in ids],
        }
    )
    return dd.from_pandas(df, npartitions=4)


@pytest.fixture
def duplicate_data(ids):
    # In each group we want to keep only the first occurrence (e.g. a1, b1, c1, d1)
    df = pd.DataFrame([{"id": _id, "group": _id[0]} for _id in ids])
    return dd.from_pandas(df, npartitions=2)


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("perform_shuffle", [False, True])
def test_remove_duplicates_basic(
    backend: Literal["cudf", "pandas"],
    perform_shuffle: bool,
    sample_data: dd.DataFrame,
    duplicate_data: dd.DataFrame,
):
    if perform_shuffle:
        # We shuffle the data to make sure that duplicates are not in the same partition
        duplicate_data = duplicate_data.sample(frac=1).reset_index(drop=True)

    sample_data = sample_data.to_backend(backend)
    duplicate_data = duplicate_data.to_backend(backend)

    # Test basic duplicate removal functionality
    result = remove_duplicates(
        left=sample_data,
        duplicates=duplicate_data,
        id_field="id",
        group_field="group",
        perform_shuffle=perform_shuffle,
    ).to_backend("pandas")

    result = result.compute()

    assert list(result.columns) == ["id", "text"]
    assert len(result) == 4
    # It's not guaranteed that we'll have a0, b0, c0, d0 in the result
    # So we should check the first character
    assert set(result["id"].apply(lambda x: x[0]).tolist()) == set(["a", "b", "c", "d"])


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("perform_shuffle", [False, True])
def test_remove_duplicates_all_duplicates(
    backend: Literal["cudf", "pandas"],
    perform_shuffle: bool,
    ids: list[str],
    sample_data: dd.DataFrame,
):

    duplicates = dd.from_pandas(
        pd.DataFrame({"id": ids, "group": [1] * len(ids)}), npartitions=2
    )
    sample_data = sample_data.to_backend(backend)
    duplicates = duplicates.to_backend(backend)

    result = remove_duplicates(
        left=sample_data,
        duplicates=duplicates,
        id_field="id",
        group_field="group",
        perform_shuffle=perform_shuffle,
    ).to_backend("pandas")

    assert list(result.columns) == ["id", "text"]
    result = result.compute()
    if perform_shuffle:
        assert len(result) == 1
    else:
        # If we don't shuffle, and both partitions have the same group
        # in both partitions we'd be left with 1 row after "deduplication"
        # and after the left-anti join we'd be left with 2 rows
        assert len(result) == 2


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("perform_shuffle", [False, True])
def test_not_remove_duplicates_unique(
    backend: Literal["cudf", "pandas"],
    perform_shuffle: bool,
    ids: list[str],
    sample_data: dd.DataFrame,
):
    # We create a dataset where first 30 ids are in one group
    # Next 9 ids are in distinct groups
    # And last id is not mentioned in duplicates

    duplicates = dd.from_pandas(
        pd.DataFrame(
            {
                "id": ids[:30] + ids[30:39],
                "group": ["group0"] * 30 + [f"group{i}" for i in range(1, 10)],
            }
        ),
        npartitions=2,
    )
    sample_data = sample_data.to_backend(backend)
    duplicates = duplicates.to_backend(backend)
    if perform_shuffle:
        # We shuffle the data to make sure that duplicates are not in the same partition
        duplicates = duplicates.sample(frac=1, random_state=42).reset_index(drop=True)

    result = remove_duplicates(
        left=sample_data,
        duplicates=duplicates,
        id_field="id",
        group_field="group",
        perform_shuffle=perform_shuffle,
    ).to_backend("pandas")

    result = result.compute()
    assert list(result.columns) == ["id", "text"]
    if perform_shuffle:
        # Since we've performed a shuffle, we know groups are collacated and there are 3 groups
        # 1.  1 row from the first group of 30
        # 2.  9 rows from the 9 distinct groups
        # 3. And 1 row from the last group which is not included in set of duplicates
        assert len(result) == 1 + 9 + 1
        # The last 10 ids should be in the result, there would be one more from the first 30
        assert set(ids[30:]).issubset(set(result["id"].tolist()))
    else:
        # If we don't shuffle, we'de be left with 2 partitions both having rows from group 1
        assert len(result) == 2 + 9 + 1


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("left_npartitions", [2, 3])
@pytest.mark.parametrize("right_npartitions", [2, 3])
def test_remove_duplicates_repartition(
    backend: Literal["cudf", "pandas"],
    left_npartitions: int,
    right_npartitions: int,
):
    # Create sample dataframes with specific partition counts
    df1 = dd.from_pandas(
        pd.DataFrame({"id": ["a1", "a2", "a3"], "text": ["text1", "text2", "text3"]}),
        npartitions=left_npartitions,
    )  # dataset with 2 partitions

    duplicates = dd.from_pandas(
        pd.DataFrame(
            {"id": ["a1", "a2", "a3"], "group": ["group1", "group1", "group1"]}
        ),
        npartitions=right_npartitions,
    )  # duplicates dataset with 3 partitions
    df1 = df1.to_backend(backend)
    duplicates = duplicates.to_backend(backend)

    # Test that it raises ValueError when right npartitions are greater than left npartitions
    output = remove_duplicates(
        left=df1,
        duplicates=duplicates,
        id_field="id",
        group_field="group",
    )
    output_dask_graph_keys = set(
        k[0].rsplit("-", 1)[0] for k in output.optimize().__dask_graph__().keys()
    )
    assert "broadcastjoin" in output_dask_graph_keys
    if left_npartitions < right_npartitions:
        assert "repartitiontofewer" in output_dask_graph_keys
    else:
        assert "repartitiontofewer" not in output_dask_graph_keys
