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

import pyarrow as pa
import pytest

from nemo_curator.stages.text.io.reader.base import BaseReader
from nemo_curator.stages.text.io.reader.lance import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
    LANCE_ROWID_COLUMN,
    LancePartitioningStage,
    LanceReader,
    LanceReaderStage,
)
from nemo_curator.tasks import EmptyTask

pytest.importorskip("lance")


def _write_lance_dataset(path: Path, *, enable_stable_row_ids: bool = False) -> None:
    import lance

    table = pa.table(
        {
            "snapshot_id": ["CC-MAIN-2025-26", "CC-MAIN-2025-18", "CC-MAIN-2025-26", "CC-MAIN-2025-26"],
            "url": ["https://a.example", "https://b.example", "https://c.example", "https://d.example"],
            "text": ["alpha one", "beta two", "gamma three", "delta four"],
            "content_zlib": lance.blob_array([b"html-a", b"html-b", b"html-c", b"html-d"]),
        },
        schema=pa.schema(
            [
                pa.field("snapshot_id", pa.string()),
                pa.field("url", pa.string()),
                pa.field("text", pa.string()),
                lance.blob_field("content_zlib"),
            ]
        ),
    )
    lance.write_dataset(
        table,
        str(path),
        mode="create",
        max_rows_per_file=2,
        max_rows_per_group=2,
        data_storage_version="2.2",
        enable_stable_row_ids=enable_stable_row_ids,
    )


def test_lance_reader_partitions_filters_blobs_and_metadata(tmp_path: Path):
    dataset_path = tmp_path / "docs.lance"
    _write_lance_dataset(dataset_path)
    read_tasks = LancePartitioningStage(path=str(dataset_path), fragments_per_partition=1).process(EmptyTask)

    assert issubclass(LanceReaderStage, BaseReader)
    assert len(read_tasks) == 2
    assert read_tasks[0].dataset_name == str(dataset_path)
    assert {fragment_id for task in read_tasks for fragment_id in task.data} == {0, 1}
    assert read_tasks[0].get_deterministic_id() != read_tasks[1].get_deterministic_id()

    reader = LanceReaderStage(
        path=str(dataset_path),
        fields=["snapshot_id", "url", "content_zlib"],
        read_kwargs={"filter": "snapshot_id = 'CC-MAIN-2025-26'", "scanner_options": {"batch_size": 2}},
    )
    batches = [batch for task in read_tasks if (batch := reader.process(task))]

    seen_fragments: set[int] = set()
    for batch in batches:
        table = batch.to_pyarrow()
        assert "schema" in batch._metadata["lance"]
        assert batch._metadata["lance"]["has_stable_row_ids"] is False
        assert LANCE_ROWID_COLUMN in table.column_names
        assert LANCE_ROWADDR_COLUMN in table.column_names
        assert LANCE_FRAGID_COLUMN in table.column_names
        assert table.schema.field("content_zlib").type.extension_name == "lance.blob.v2"
        fragids = {int(value) for value in table[LANCE_FRAGID_COLUMN].combine_chunks().to_pylist()}
        assert seen_fragments.isdisjoint(fragids)
        seen_fragments.update(fragids)
    assert seen_fragments == {0, 1}


def test_lance_reader_exposes_stable_row_ids(tmp_path: Path):
    dataset_path = tmp_path / "stable.lance"
    _write_lance_dataset(dataset_path, enable_stable_row_ids=True)
    task = LancePartitioningStage(path=str(dataset_path)).process(EmptyTask)[0]

    batch = LanceReaderStage(path=str(dataset_path), fields=["url"]).process(task)
    table = batch.to_pyarrow()

    assert batch._metadata["lance"]["has_stable_row_ids"] is True
    assert table[LANCE_ROWID_COLUMN].null_count == 0
    assert len(set(table[LANCE_ROWID_COLUMN].to_pylist())) == table.num_rows


def test_lance_reader_validates_requested_fragments(tmp_path: Path):
    dataset_path = tmp_path / "docs.lance"
    _write_lance_dataset(dataset_path)

    tasks = LancePartitioningStage(path=str(dataset_path), fragments_per_partition=1, fragment_ids=[1, 0, 1]).process(
        EmptyTask
    )
    assert [task.data for task in tasks] == [[0], [1]]

    with pytest.raises(ValueError, match="requested fragment ids"):
        LancePartitioningStage(path=str(dataset_path), fragment_ids=[999]).process(EmptyTask)


def test_lance_reader_columns_empty_filters_and_fields_override(tmp_path: Path):
    dataset_path = tmp_path / "docs.lance"
    _write_lance_dataset(dataset_path)
    task = LancePartitioningStage(path=str(dataset_path)).process(EmptyTask)[0]

    batch = LanceReaderStage(
        path=str(dataset_path), read_kwargs={"columns": ["url"]}, include_lance_metadata=False
    ).process(task)
    assert batch.to_pyarrow().column_names == ["url"]

    empty_batch = LanceReaderStage(path=str(dataset_path), read_kwargs={"filter": "snapshot_id = 'missing'"}).process(
        task
    )
    empty_table = empty_batch.to_pyarrow()
    assert empty_table.num_rows == 0
    assert LANCE_ROWID_COLUMN in empty_table.column_names
    assert LANCE_ROWADDR_COLUMN in empty_table.column_names
    assert LANCE_FRAGID_COLUMN in empty_table.column_names

    _, reader_stage = LanceReader(
        path="example.lance", fields=["a", "b"], read_kwargs={"columns": ["ignored"]}
    ).decompose()
    assert reader_stage.fields == ["a", "b"]
    assert reader_stage.include_lance_metadata is True


def test_lance_reader_uses_partition_version(tmp_path: Path):
    import lance

    dataset_path = tmp_path / "docs.lance"
    lance.write_dataset(pa.table({"text": ["old"]}), str(dataset_path), mode="create", max_rows_per_file=1)
    task = LancePartitioningStage(path=str(dataset_path)).process(EmptyTask)[0]
    lance.write_dataset(pa.table({"text": ["new"]}), str(dataset_path), mode="overwrite", max_rows_per_file=1)

    batch = LanceReaderStage(path=str(dataset_path), fields=["text"], include_lance_metadata=False).process(task)

    assert batch.to_pyarrow()["text"].to_pylist() == ["old"]


def test_lance_reader_rejects_conflicting_version(tmp_path: Path):
    import lance

    dataset_path = tmp_path / "docs.lance"
    lance.write_dataset(pa.table({"text": ["old"]}), str(dataset_path), mode="create", max_rows_per_file=1)
    task = LancePartitioningStage(path=str(dataset_path)).process(EmptyTask)[0]
    lance.write_dataset(pa.table({"text": ["new"]}), str(dataset_path), mode="overwrite", max_rows_per_file=1)
    latest_version = lance.dataset(str(dataset_path)).version

    with pytest.raises(ValueError, match="version mismatch"):
        LanceReaderStage(
            path=str(dataset_path),
            fields=["text"],
            read_kwargs={"version": latest_version},
            include_lance_metadata=False,
        ).process(task)
