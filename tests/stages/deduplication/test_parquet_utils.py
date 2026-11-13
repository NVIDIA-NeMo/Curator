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

from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.deduplication.parquet_utils import (
    has_page_indexed_delta_byte_array,
    read_parquet_with_pyarrow,
)


def _write_parquet(path: Path, *, page_index: bool, include_id: bool = True) -> None:
    values = {"text": ["hello", "parquet"]}
    if include_id:
        values = {"id": [1, 2], **values}
    pq.write_table(
        pa.table(values),
        path,
        use_dictionary=False,
        column_encoding={"text": "DELTA_BYTE_ARRAY"},
        write_page_index=page_index,
    )


def test_detects_only_selected_columns_with_both_page_indexes(tmp_path: Path) -> None:
    indexed = tmp_path / "indexed.parquet"
    no_index = tmp_path / "no-index.parquet"
    _write_parquet(indexed, page_index=True)
    _write_parquet(no_index, page_index=False)

    metadata = pq.ParquetFile(indexed).metadata.row_group(0).column(1)
    assert "DELTA_BYTE_ARRAY" in metadata.encodings
    assert metadata.has_column_index
    assert metadata.has_offset_index

    assert has_page_indexed_delta_byte_array(str(indexed))
    assert not has_page_indexed_delta_byte_array(str(indexed), columns=["id"])
    assert not has_page_indexed_delta_byte_array(str(no_index))


def test_arrow_read_preserves_strings_and_column_projection(tmp_path: Path) -> None:
    indexed = tmp_path / "indexed.parquet"
    _write_parquet(indexed, page_index=True)

    table = read_parquet_with_pyarrow(str(indexed), columns=["text"], allow_mismatched_pq_schemas=True)

    assert table.column_names == ["text"]
    assert table.column("text").to_pylist() == ["hello", "parquet"]


def test_arrow_read_preserves_filters(tmp_path: Path) -> None:
    indexed = tmp_path / "indexed.parquet"
    _write_parquet(indexed, page_index=True)

    table = read_parquet_with_pyarrow(
        str(indexed),
        columns=["text"],
        filters=[("id", "==", 2)],
        allow_mismatched_pq_schemas=True,
    )

    assert table.column("text").to_pylist() == ["parquet"]


def test_arrow_read_handles_a_mixed_file_list_in_input_order(tmp_path: Path) -> None:
    indexed = tmp_path / "indexed.parquet"
    no_index = tmp_path / "no-index.parquet"
    _write_parquet(indexed, page_index=True, include_id=False)
    _write_parquet(no_index, page_index=False)

    table = read_parquet_with_pyarrow(
        [str(indexed), str(no_index)],
        columns=["text", "id"],
        allow_mismatched_pq_schemas=True,
    )

    assert table.column("text").to_pylist() == ["hello", "parquet", "hello", "parquet"]
    assert table.column("id").to_pylist() == [None, None, 1, 2]


@pytest.mark.parametrize("filesystem_kind", ["none", "fsspec", "pyarrow"])
def test_arrow_read_ignores_conflicting_unselected_columns(tmp_path: Path, filesystem_kind: str) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.table({"text": ["hello"], "payload": [{"name": "a"}], "id": [1], "extra": [1]}), first)
    pq.write_table(pa.table({"text": ["world"], "payload": [{"name": "b"}], "id": [2], "extra": ["x"]}), second)

    filesystem = {"none": None, "fsspec": fsspec.filesystem("file"), "pyarrow": pafs.LocalFileSystem()}[
        filesystem_kind
    ]

    table = read_parquet_with_pyarrow(
        [str(first), str(second)],
        columns=["text", "payload.name"],
        filters=[("id", "==", 2)],
        filesystem=filesystem,
        allow_mismatched_pq_schemas=True,
    )

    assert table.column("text").to_pylist() == ["world"]
    assert table.column("name").to_pylist() == ["b"]


@pytest.mark.parametrize("storage_options", [{}, {"skip_instance_cache": True}])
def test_local_storage_options_are_used_for_footer_inspection(
    tmp_path: Path, storage_options: dict[str, bool]
) -> None:
    indexed = tmp_path / "indexed.parquet"
    _write_parquet(indexed, page_index=True)

    assert has_page_indexed_delta_byte_array(str(indexed), storage_options=storage_options)


def test_provided_filesystem_is_used_for_inspection_and_reading(tmp_path: Path) -> None:
    indexed = tmp_path / "indexed.parquet"
    _write_parquet(indexed, page_index=True)
    filesystem = fsspec.filesystem("file", skip_instance_cache=True)

    assert has_page_indexed_delta_byte_array(str(indexed), filesystem=filesystem)
    table = read_parquet_with_pyarrow(str(indexed), filesystem=filesystem, columns=["text"])

    assert table.column("text").to_pylist() == ["hello", "parquet"]


def test_pyarrow_filesystem_is_used_for_inspection_and_reading(tmp_path: Path) -> None:
    indexed = tmp_path / "indexed.parquet"
    _write_parquet(indexed, page_index=True)
    filesystem = pafs.LocalFileSystem()

    assert has_page_indexed_delta_byte_array(str(indexed), filesystem=filesystem)
    table = read_parquet_with_pyarrow(str(indexed), filesystem=filesystem, columns=["text"])

    assert table.column("text").to_pylist() == ["hello", "parquet"]
