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
from pathlib import Path

import pyarrow as pa
import pytest

from nemo_curator.stages.text.io.reader.lance import (
    LancePartitioningStage,
    LanceReaderStage,
)
from nemo_curator.stages.text.io.writer import (
    LanceWriter,
    commit_lance_checkpoint,
)
from nemo_curator.tasks import DocumentBatch, EmptyTask

pytest.importorskip("lance")
pytest.importorskip("lance_ray")


def _blob_schema(extra_fields: list[pa.Field] | None = None) -> pa.Schema:
    import lance

    fields = [
        pa.field("id", pa.int64()),
        pa.field("url", pa.string()),
        pa.field("text", pa.string()),
        lance.blob_field("content_zlib"),
    ]
    fields.extend(extra_fields or [])
    return pa.schema(fields)


def _blob_table() -> pa.Table:
    import lance

    return pa.table(
        {
            "id": [1, 2, 3, 4],
            "url": ["https://a.example", "https://b.example", "https://c.example", "https://d.example"],
            "text": ["alpha one", "beta two", "gamma three", "delta four"],
            "content_zlib": lance.blob_array([b"html-a", b"html-b", b"html-c", b"html-d"]),
        },
        schema=_blob_schema(),
    )


def _write_source_dataset(path: Path) -> None:
    import lance

    lance.write_dataset(
        _blob_table(), str(path), mode="create", max_rows_per_file=2, max_rows_per_group=2, data_storage_version="2.2"
    )


def _assert_blob_dataset(path: Path, version: int) -> None:
    import lance

    dataset = lance.dataset(str(path), version=version)
    assert dataset.count_rows() == 4
    assert dataset.schema.field("content_zlib").type.extension_name == "lance.blob.v2"
    blobs = dataset.read_blobs("content_zlib", indices=[0, 1, 2, 3], preserve_order=True)
    assert sorted(payload for _, payload in blobs) == [b"html-a", b"html-b", b"html-c", b"html-d"]


def test_lance_writer_checkpoint_commit_retry_and_blobs(tmp_path: Path):
    output_path = tmp_path / "out.lance"
    commit_path = tmp_path / "writer_commit"
    batch = DocumentBatch(dataset_name="docs", data=_blob_table())
    batch._set_task_id("0", "task")
    writer = LanceWriter(
        path=str(output_path),
        commit_path=str(commit_path),
        schema=_blob_schema(),
        mode="overwrite",
        write_kwargs={"max_rows_per_file": 2, "max_rows_per_group": 2, "data_storage_version": "2.2"},
    )

    writer.process(batch)
    writer.process(batch)

    records = [json.loads(path.read_text()) for path in (commit_path / "records").glob("*.json")]
    assert {(record["task_id"], record["fragment_index"]) for record in records} == {("0_task", 0), ("0_task", 1)}
    version = commit_lance_checkpoint(str(output_path), str(commit_path))
    assert commit_lance_checkpoint(str(output_path), str(commit_path)) == version
    _assert_blob_dataset(output_path, version)


def test_lance_writer_enables_stable_row_ids(tmp_path: Path):
    import lance

    output_path = tmp_path / "stable.lance"
    commit_path = tmp_path / "stable_commit"
    batch = DocumentBatch(dataset_name="docs", data=_blob_table())
    batch._set_task_id("0", "stable")

    LanceWriter(
        path=str(output_path),
        commit_path=str(commit_path),
        schema=_blob_schema(),
        enable_stable_row_ids=True,
        write_kwargs={"max_rows_per_file": 2, "data_storage_version": "2.2"},
    ).process(batch)

    version = commit_lance_checkpoint(str(output_path), str(commit_path))
    dataset = lance.dataset(str(output_path), version=version)
    assert dataset.has_stable_row_ids


def test_lance_writer_preserves_reader_blob_columns_without_explicit_schema(tmp_path: Path):
    source_path = tmp_path / "source.lance"
    output_path = tmp_path / "out.lance"
    commit_path = tmp_path / "writer_commit"
    _write_source_dataset(source_path)
    read_task = LancePartitioningStage(path=str(source_path), fragments_per_partition=2).process(EmptyTask)[0]
    batch = LanceReaderStage(path=str(source_path), fields=["id", "url", "text", "content_zlib"]).process(read_task)

    LanceWriter(
        path=str(output_path),
        commit_path=str(commit_path),
        mode="overwrite",
        write_kwargs={"data_storage_version": "2.2"},
    ).process(batch)

    _assert_blob_dataset(output_path, commit_lance_checkpoint(str(output_path), str(commit_path)))


def test_lance_writer_allows_empty_batches(tmp_path: Path):
    batch = DocumentBatch(dataset_name="docs", data=pa.table({"text": pa.array([], type=pa.string())}))
    batch._set_task_id("0", "empty")

    result = LanceWriter(
        path=str(tmp_path / "out.lance"),
        commit_path=str(tmp_path / "writer_commit"),
        schema=batch.to_pyarrow().schema,
    ).process(batch)

    assert result.data == []
