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

from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

_SAMPLE_ROW = {
    "sample_id": "s1",
    "position": 0,
    "modality": "text",
    "content_type": "text/plain",
    "text_content": "hello",
    "binary_content": None,
    "source_ref": None,
    "materialize_error": None,
}


def _make_batch(data: pa.Table | pd.DataFrame) -> InterleavedBatch:
    return InterleavedBatch(task_id="t", dataset_name="d", data=data)


# --- to_pyarrow ---


@pytest.mark.parametrize("data_type", ["pyarrow", "pandas"])
def test_to_pyarrow(data_type: str) -> None:
    if data_type == "pyarrow":
        data = pa.Table.from_pylist([_SAMPLE_ROW], schema=INTERLEAVED_SCHEMA)
    else:
        data = pd.DataFrame([_SAMPLE_ROW])
    result = _make_batch(data).to_pyarrow()
    assert isinstance(result, pa.Table)


def test_to_pyarrow_invalid_type() -> None:
    task = _make_batch(pa.Table.from_pylist([_SAMPLE_ROW], schema=INTERLEAVED_SCHEMA))
    object.__setattr__(task, "data", [1, 2, 3])
    with pytest.raises(TypeError, match="Cannot convert"):
        task.to_pyarrow()


# --- to_pandas ---


def test_to_pandas_invalid_type() -> None:
    task = _make_batch(pa.Table.from_pylist([_SAMPLE_ROW], schema=INTERLEAVED_SCHEMA))
    object.__setattr__(task, "data", [1, 2, 3])
    with pytest.raises(TypeError, match="Cannot convert"):
        task.to_pandas()


# --- get_columns ---


@pytest.mark.parametrize("data_type", ["pyarrow", "pandas"])
def test_get_columns(data_type: str) -> None:
    if data_type == "pyarrow":
        data = pa.Table.from_pylist([_SAMPLE_ROW], schema=INTERLEAVED_SCHEMA)
    else:
        data = pd.DataFrame([_SAMPLE_ROW])
    cols = _make_batch(data).get_columns()
    assert isinstance(cols, list)
    assert "sample_id" in cols
    assert "modality" in cols


def test_get_columns_invalid_type() -> None:
    task = _make_batch(pa.Table.from_pylist([_SAMPLE_ROW], schema=INTERLEAVED_SCHEMA))
    object.__setattr__(task, "data", [1, 2, 3])
    with pytest.raises(TypeError, match="Unsupported data type"):
        task.get_columns()


# --- validate ---


@pytest.mark.parametrize(
    ("data_factory", "expected"),
    [
        pytest.param(
            lambda: pa.Table.from_pylist([_SAMPLE_ROW], schema=INTERLEAVED_SCHEMA),
            True,
            id="valid_task",
        ),
        pytest.param(
            lambda: pa.Table.from_pylist([], schema=INTERLEAVED_SCHEMA),
            False,
            id="empty_task",
        ),
        pytest.param(
            lambda: pa.table({"sample_id": ["s1"], "position": [0]}),
            False,
            id="missing_required_columns",
        ),
    ],
)
def test_validate(data_factory: object, expected: bool) -> None:
    task = _make_batch(data_factory())
    assert task.validate() is expected


# --- add_rows / delete_rows ---


def test_add_rows_and_delete_rows_not_implemented() -> None:
    task = _make_batch(pa.Table.from_pylist([_SAMPLE_ROW], schema=INTERLEAVED_SCHEMA))
    with pytest.raises(NotImplementedError):
        task.add_rows(pd.DataFrame([_SAMPLE_ROW]))
    with pytest.raises(NotImplementedError):
        task.delete_rows(pd.Series([True]))


# --- parse_source_ref edge cases ---


def test_parse_source_ref_non_dict_raises() -> None:
    with pytest.raises(TypeError, match="source_ref must decode to a JSON object"):
        InterleavedBatch.parse_source_ref("[1, 2]")


def test_parse_source_ref_with_frame_index() -> None:
    ref = json.dumps({"path": "/a.tar", "member": "m.jpg", "byte_offset": 10, "byte_size": 20, "frame_index": 5})
    parsed = InterleavedBatch.parse_source_ref(ref)
    assert parsed["path"] == "/a.tar"
    assert parsed["member"] == "m.jpg"
    assert parsed["byte_offset"] == 10
    assert parsed["byte_size"] == 20
    assert parsed["frame_index"] == 5


# --- build_source_ref ---


@pytest.mark.parametrize(
    ("frame_index", "key_present"),
    [
        pytest.param(3, True, id="with_frame_index"),
        pytest.param(None, False, id="without_frame_index"),
    ],
)
def test_build_source_ref_frame_index(frame_index: int | None, key_present: bool) -> None:
    ref_str = InterleavedBatch.build_source_ref(
        path="/a.tar",
        member="m.jpg",
        byte_offset=10,
        byte_size=20,
        frame_index=frame_index,
    )
    parsed = json.loads(ref_str)
    assert ("frame_index" in parsed) is key_present
    if key_present:
        assert parsed["frame_index"] == frame_index
