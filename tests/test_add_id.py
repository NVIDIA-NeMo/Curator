# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import dask.dataframe as dd
import pandas as pd
import pytest

import nemo_curator
from nemo_curator.datasets import DocumentDataset


def list_to_dataset(documents, col_name="text", npartitions=2):
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentDataset(dd.from_pandas(pdf, npartitions=npartitions))


@pytest.fixture
def single_partition_dataset():
    return list_to_dataset(
        ["First", "Second", "Third", "Fourth", "Fifth"], npartitions=1
    )


@pytest.fixture
def two_partition_dataset():
    return list_to_dataset(
        ["First", "Second", "Third", "Fourth", "Fifth"], npartitions=2
    )


class TestPrepareTaskData:
    def test_basic_id(self, single_partition_dataset):
        id_field = "id"
        add_id = nemo_curator.AddId(id_field)
        id_dataset = add_id(single_partition_dataset)
        actual_ids = id_dataset.df[id_field].compute()
        expected_ids = pd.Series(
            [
                "doc_id-0000000000",
                "doc_id-0000000001",
                "doc_id-0000000002",
                "doc_id-0000000003",
                "doc_id-0000000004",
            ]
        )

        assert all(
            expected_ids == actual_ids
        ), f"Expected: {expected_ids}, got: {actual_ids}"

    def test_two_partitions(self, two_partition_dataset):
        id_field = "id"
        add_id = nemo_curator.AddId(id_field)
        id_dataset = add_id(two_partition_dataset)
        actual_ids = id_dataset.df[id_field].compute()
        expected_ids = pd.Series(
            [
                "doc_id-0000000000",
                "doc_id-0000000001",
                "doc_id-0000000002",
                "doc_id-0000000003",
                "doc_id-0000000004",
            ]
        )

        assert all(
            expected_ids == actual_ids
        ), f"Expected: {expected_ids}, got: {actual_ids}"

    def test_id_prefix(self, two_partition_dataset):
        id_field = "id"
        id_prefix = "my_id"
        add_id = nemo_curator.AddId(id_field, id_prefix=id_prefix)
        id_dataset = add_id(two_partition_dataset)
        actual_ids = id_dataset.df[id_field].compute()
        expected_ids = pd.Series(
            [
                f"{id_prefix}-0000000000",
                f"{id_prefix}-0000000001",
                f"{id_prefix}-0000000002",
                f"{id_prefix}-0000000003",
                f"{id_prefix}-0000000004",
            ]
        )

        assert all(
            expected_ids == actual_ids
        ), f"Expected: {expected_ids}, got: {actual_ids}"

    def test_start_index(self, two_partition_dataset):
        id_field = "id"
        start_index = 13
        add_id = nemo_curator.AddId(id_field, start_index=start_index)
        id_dataset = add_id(two_partition_dataset)
        actual_ids = id_dataset.df[id_field].compute()
        expected_ids = pd.Series(
            [
                "doc_id-0000000013",
                "doc_id-0000000014",
                "doc_id-0000000015",
                "doc_id-0000000016",
                "doc_id-0000000017",
            ]
        )

        assert all(
            expected_ids == actual_ids
        ), f"Expected: {expected_ids}, got: {actual_ids}"
