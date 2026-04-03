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

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.utils.validation_utils import (
    resolve_storage_options,
    validate_and_project_source_fields,
)
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA


def _make_task(metadata: dict | None = None) -> InterleavedBatch:
    table = pa.Table.from_pylist([], schema=INTERLEAVED_SCHEMA)
    return InterleavedBatch(task_id="t", dataset_name="d", data=table, _metadata=metadata or {})


# --- resolve_storage_options ---


@pytest.mark.parametrize(
    ("task_metadata", "io_kwargs", "expected"),
    [
        pytest.param(None, None, {}, id="none_none"),
        pytest.param(
            {"source_storage_options": {"key": "s3secret"}},
            None,
            {"key": "s3secret"},
            id="task_metadata",
        ),
        pytest.param(
            {},
            {"storage_options": {"k": "v"}},
            {"k": "v"},
            id="io_kwargs_fallback",
        ),
        pytest.param(
            {},
            {"storage_options": "not-a-dict"},
            {},
            id="non_dict_storage_options",
        ),
    ],
)
def test_resolve_storage_options(
    task_metadata: dict | None,
    io_kwargs: dict | None,
    expected: dict,
) -> None:
    task = _make_task(task_metadata) if task_metadata is not None else None
    assert resolve_storage_options(task=task, io_kwargs=io_kwargs) == expected


# --- validate_and_project_source_fields ---


@pytest.mark.parametrize(
    ("sample", "fields", "excluded", "expected"),
    [
        pytest.param(
            {"x": 1, "y": 2, "z": 3},
            None,
            {"x"},
            {"y": 2, "z": 3},
            id="fields_none_excludes",
        ),
        pytest.param(
            {"a": {"nested": True}},
            None,
            set(),
            {"a": json.dumps({"nested": True}, ensure_ascii=True)},
            id="dict_value_serialized",
        ),
        pytest.param(
            {"a": [1, 2]},
            None,
            set(),
            {"a": json.dumps([1, 2], ensure_ascii=True)},
            id="list_value_serialized",
        ),
        pytest.param(
            {"a": "hello"},
            None,
            set(),
            {"a": "hello"},
            id="scalar_passthrough",
        ),
        pytest.param(
            {"a": 1},
            ("missing_key",),
            set(),
            {"missing_key": None},
            id="missing_fills_none",
        ),
    ],
)
def test_validate_and_project_source_fields(
    sample: dict,
    fields: tuple[str, ...] | None,
    excluded: set[str],
    expected: dict,
) -> None:
    assert validate_and_project_source_fields(sample, fields, excluded) == expected


def test_validate_and_project_reserved_field_raises() -> None:
    with pytest.raises(ValueError, match="fields contains reserved keys"):
        validate_and_project_source_fields({"reserved": 1}, ("reserved",), {"reserved"})
