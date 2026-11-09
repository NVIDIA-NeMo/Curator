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
from unittest.mock import patch

import lance
import pyarrow as pa
import pytest
from lance_ray import LanceFragmentCommitter

from nemo_curator.stages.text.io.reader.lance import (
    LancePartitioningStage,
    LanceReaderStage,
)
from nemo_curator.stages.text.io.writer import (
    LanceWriter,
    commit_lance_checkpoint,
)
from nemo_curator.tasks import DocumentBatch, EmptyTask


def _blob_schema(extra_fields: list[pa.Field] | None = None) -> pa.Schema:
    fields = [
        pa.field("id", pa.int64()),
        pa.field("url", pa.string()),
        pa.field("text", pa.string()),
        lance.blob_field("content_zlib"),
    ]
    fields.extend(extra_fields or [])
    return pa.schema(fields)


def _blob_table() -> pa.Table:
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
    lance.write_dataset(
        _blob_table(), str(path), mode="create", max_rows_per_file=2, max_rows_per_group=2, data_storage_version="2.2"
    )


def _assert_blob_dataset(path: Path, version: int) -> None:
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

    record_paths = writer.process(batch).data
    # A retry writes fresh fragments but reuses the deterministic checkpoint record paths.
    retry_write = writer.process(batch)
    assert retry_write.data == record_paths
    assert len(record_paths) == 1

    records = [json.loads(path.read_text()) for path in (commit_path / "records").glob("*.json")]
    assert len(records) == len(record_paths)
    assert records[0]["task_id"] == "0_task"
    assert len(records[0]["fragments"]) == 2
    assert all(isinstance(fragment, dict) for fragment in records[0]["fragments"])
    version = commit_lance_checkpoint(str(output_path), str(commit_path))
    with patch("nemo_curator.stages.text.io.writer.lance.logger") as mock_logger:
        assert commit_lance_checkpoint(str(output_path), str(commit_path)) == version
    mock_logger.warning.assert_called_once_with(
        f"Lance checkpoint {commit_path} already committed as version {version}; skipping"
    )
    assert json.loads((commit_path / "_COMMITTED").read_text()) == {
        "dataset_path": str(output_path),
        "version": version,
    }
    with pytest.raises(ValueError, match="belongs to"):
        commit_lance_checkpoint(str(tmp_path / "other.lance"), str(commit_path))
    _assert_blob_dataset(output_path, version)


def test_lance_writer_commit_error_identifies_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_path = tmp_path / "out.lance"
    commit_path = tmp_path / "writer_commit"
    schema = pa.schema([pa.field("text", pa.string())])
    writer = LanceWriter(path=str(output_path), commit_path=str(commit_path), schema=schema, mode="overwrite")
    for task_id in ("task-a", "task-b"):
        batch = DocumentBatch(dataset_name="docs", data=pa.table({"text": [task_id]}, schema=schema))
        batch._set_task_id("0", task_id)
        writer.process(batch)

    def fail_commit(_self: object, _write_result: object) -> None:
        msg = "commit failed"
        raise RuntimeError(msg)

    monkeypatch.setattr("lance_ray.LanceFragmentCommitter.on_write_complete", fail_commit)
    with pytest.raises(RuntimeError, match="commit failed") as exc_info:
        commit_lance_checkpoint(str(output_path), str(commit_path))

    assert exc_info.value.__notes__ == [
        "Lance commit includes fragments from Curator task IDs ['0_task-a', '0_task-b']"
    ]


def test_lance_writer_recovers_append_after_marker_failure(tmp_path: Path):
    output_path = tmp_path / "out.lance"
    commit_path = tmp_path / "writer_commit"
    schema = pa.schema([pa.field("text", pa.string())])
    lance.write_dataset(pa.table({"text": ["existing"]}, schema=schema), str(output_path), mode="create")
    batch = DocumentBatch(dataset_name="docs", data=pa.table({"text": ["new"]}, schema=schema))
    batch._set_task_id("0", "append")
    LanceWriter(path=str(output_path), commit_path=str(commit_path), schema=schema, mode="append").process(batch)

    with (
        patch(
            "nemo_curator.stages.text.io.writer.lance.write_json_file",
            side_effect=RuntimeError("marker write failed"),
        ),
        pytest.raises(RuntimeError, match="marker write failed"),
    ):
        commit_lance_checkpoint(str(output_path), str(commit_path))

    published = lance.dataset(str(output_path))
    assert published.count_rows() == 2
    assert not (commit_path / "_COMMITTED").exists()
    assert commit_lance_checkpoint(str(output_path), str(commit_path)) == published.version
    retried = lance.dataset(str(output_path))
    assert retried.version == published.version
    assert retried.count_rows() == 2


def test_lance_writer_records_own_version_after_concurrent_append(tmp_path: Path):
    output_path = tmp_path / "out.lance"
    commit_path = tmp_path / "writer_commit"
    schema = pa.schema([pa.field("text", pa.string())])
    lance.write_dataset(pa.table({"text": ["existing"]}, schema=schema), str(output_path), mode="create")
    batch = DocumentBatch(dataset_name="docs", data=pa.table({"text": ["ours"]}, schema=schema))
    batch._set_task_id("0", "append")
    LanceWriter(path=str(output_path), commit_path=str(commit_path), schema=schema, mode="append").process(batch)

    on_write_complete = LanceFragmentCommitter.on_write_complete

    def commit_then_append(committer: LanceFragmentCommitter, write_result: object) -> None:
        on_write_complete(committer, write_result)
        lance.write_dataset(pa.table({"text": ["concurrent"]}, schema=schema), str(output_path), mode="append")

    with patch.object(LanceFragmentCommitter, "on_write_complete", commit_then_append):
        committed_version = commit_lance_checkpoint(str(output_path), str(commit_path))

    latest = lance.dataset(str(output_path))
    assert committed_version == latest.version - 1
    assert json.loads((commit_path / "_COMMITTED").read_text())["version"] == committed_version
    assert lance.dataset(str(output_path), version=committed_version).count_rows() == 2
    assert latest.count_rows() == 3


def test_lance_writer_creates_dataset(tmp_path: Path):
    output_path = tmp_path / "created.lance"
    commit_path = tmp_path / "create_commit"
    batch = DocumentBatch(dataset_name="docs", data=_blob_table())
    batch._set_task_id("0", "create")

    LanceWriter(
        path=str(output_path),
        commit_path=str(commit_path),
        schema=_blob_schema(),
        write_kwargs={"max_rows_per_file": 2, "data_storage_version": "2.2"},
    ).process(batch)

    _assert_blob_dataset(output_path, commit_lance_checkpoint(str(output_path), str(commit_path)))


def test_lance_writer_preserves_reader_blob_columns_without_explicit_schema(tmp_path: Path):
    source_path = tmp_path / "source.lance"
    output_path = tmp_path / "out.lance"
    commit_path = tmp_path / "writer_commit"
    _write_source_dataset(source_path)
    read_task = LancePartitioningStage(path=str(source_path), fragments_per_partition=2).process(EmptyTask)[0]
    batch = LanceReaderStage(fields=["id", "url", "text", "content_zlib"]).process(read_task)

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
