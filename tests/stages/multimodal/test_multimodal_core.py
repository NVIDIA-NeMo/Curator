# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import json

import pandas as pd
import pyarrow as pa
import pytest

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.tasks import MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA


@pytest.fixture
def single_row_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "metadata_source": json.dumps(
                    {
                        "source_id": "doc.pdf",
                        "source_shard": "shard-00000.tar",
                        "content_path": "/tmp/shard-00000.tar",
                        "content_key": "s1.json",
                    }
                ),
                "metadata_json": None,
                "materialize_error": None,
            }
        ],
        schema=MULTIMODAL_SCHEMA,
    )


@pytest.fixture
def single_row_task(single_row_table: pa.Table) -> MultiBatchTask:
    return MultiBatchTask(task_id="t1", dataset_name="d1", data=single_row_table)


def test_to_pandas_keeps_arrow_dtypes(single_row_task: MultiBatchTask) -> None:
    df = single_row_task.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert str(df.dtypes["sample_id"]).endswith("[pyarrow]")
    assert str(df.dtypes["position"]).endswith("[pyarrow]")


def test_with_parsed_source_columns(single_row_task: MultiBatchTask) -> None:
    df = single_row_task.with_parsed_source_columns()
    assert df.loc[0, "_src_source_id"] == "doc.pdf"
    assert df.loc[0, "_src_source_shard"] == "shard-00000.tar"
    assert df.loc[0, "_src_content_path"] == "/tmp/shard-00000.tar"
    assert df.loc[0, "_src_content_key"] == "s1.json"


def test_split_table_keeps_group_intact() -> None:
    table = pa.Table.from_pylist(
        [
            {"sample_id": "a", "position": 0, "value": "x" * 20},
            {"sample_id": "a", "position": 1, "value": "y" * 20},
            {"sample_id": "b", "position": 0, "value": "z" * 20},
            {"sample_id": "b", "position": 1, "value": "w" * 20},
        ]
    )

    splits = split_table_by_group_max_bytes(table, "sample_id", max_batch_bytes=120)
    assert len(splits) == 2

    first_groups = set(splits[0]["sample_id"].to_pylist())
    second_groups = set(splits[1]["sample_id"].to_pylist())
    assert len(first_groups) == 1
    assert len(second_groups) == 1
    assert first_groups != second_groups
