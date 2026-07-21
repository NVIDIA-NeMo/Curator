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

from nemo_curator.stages.interleaved.lance import InterleavedLanceReader, InterleavedLanceReaderStage
from nemo_curator.stages.text.io.reader.lance import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
    LANCE_ROWID_COLUMN,
    LancePartitioningStage,
    LanceReadTask,
)
from nemo_curator.tasks import EmptyTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

lance = pytest.importorskip("lance")


def _row(sample_id: str, position: int, modality: str, text: str | None = None) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "position": position,
        "modality": modality,
        "content_type": "text/plain" if modality == "text" else None,
        "text_content": text,
        "binary_content": None,
        "source_ref": None,
        "materialize_error": None,
    }


def _write_interleaved_dataset(path: Path, rows: list[dict[str, object]], *, max_rows_per_group: int = 2) -> None:
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    lance.write_dataset(
        table,
        str(path),
        mode="create",
        max_rows_per_file=len(rows),
        max_rows_per_group=max_rows_per_group,
    )


def _nullable_sample_id_schema() -> pa.Schema:
    return pa.schema([pa.field("sample_id", pa.string(), nullable=True), *list(INTERLEAVED_SCHEMA)[1:]])


def _single_fragment_task(dataset_path: Path) -> LanceReadTask:
    return LancePartitioningStage(path=str(dataset_path), fragments_per_partition=1).process(EmptyTask())[0]


def _tables(result: InterleavedBatch | list[InterleavedBatch]) -> list[pa.Table]:
    batches = result if isinstance(result, list) else [result]
    return [batch.to_pyarrow() for batch in batches]


def test_interleaved_lance_reader_validates_required_fields() -> None:
    with pytest.raises(ValueError, match="omit required columns"):
        InterleavedLanceReader(path="example.lance", fields=["sample_id"]).decompose()


def test_interleaved_lance_reader_decomposes() -> None:
    partitioner, reader = InterleavedLanceReader(
        path="example.lance",
        fields=list(INTERLEAVED_SCHEMA.names),
        max_batch_bytes=256 * 1024 * 1024,
        max_batch_rows=1024,
        streaming_read=True,
        max_output_rows=10_000,
        fragments_per_partition=2,
        fragment_ids=[1],
    ).decompose()

    assert partitioner.fragments_per_partition == 2
    assert partitioner.fragment_ids == [1]
    assert reader.fields == list(INTERLEAVED_SCHEMA.names)
    assert reader.max_batch_bytes == 256 * 1024 * 1024
    assert reader.max_batch_rows == 1024
    assert reader.streaming_read is True
    assert reader.max_output_rows == 10_000
    assert reader.include_lance_metadata is True


@pytest.mark.parametrize(
    "field",
    ["max_batch_bytes", "max_batch_rows", "max_output_rows"],
)
def test_interleaved_lance_reader_rejects_non_positive_limits(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be > 0"):
        InterleavedLanceReaderStage(**{field: 0})


def test_interleaved_lance_reader_splits_without_splitting_sample_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "interleaved.lance"
    rows = [
        _row("doc-a", 0, "text", "a0"),
        _row("doc-a", 1, "image"),
        _row("doc-b", 0, "text", "b0"),
        _row("doc-b", 1, "image"),
        _row("doc-c", 0, "text", "c0"),
        _row("doc-c", 1, "image"),
    ]
    _write_interleaved_dataset(dataset_path, rows)
    task = _single_fragment_task(dataset_path)

    result = InterleavedLanceReaderStage(
        fields=list(INTERLEAVED_SCHEMA.names),
        max_batch_rows=3,
        include_lance_metadata=False,
    ).process(task)

    tables = _tables(result)
    assert [table["sample_id"].combine_chunks().to_pylist() for table in tables] == [
        ["doc-a", "doc-a"],
        ["doc-b", "doc-b"],
        ["doc-c", "doc-c"],
    ]


def test_interleaved_lance_reader_rejects_null_sample_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "null-sample-id.lance"
    row = _row("doc-a", 0, "text", "a0")
    row["sample_id"] = None
    table = pa.Table.from_pylist([row], schema=_nullable_sample_id_schema())
    lance.write_dataset(table, str(dataset_path), mode="create")
    task = _single_fragment_task(dataset_path)

    with pytest.raises(ValueError, match="contains null values"):
        InterleavedLanceReaderStage(
            fields=list(INTERLEAVED_SCHEMA.names),
            include_lance_metadata=False,
        ).process(task)


def test_interleaved_lance_reader_streaming_carries_sample_across_scanner_batches(tmp_path: Path) -> None:
    dataset_path = tmp_path / "streaming.lance"
    rows = [
        _row("doc-a", 0, "text", "a0"),
        _row("doc-a", 1, "text", "a1"),
        _row("doc-a", 2, "image"),
        _row("doc-b", 0, "text", "b0"),
        _row("doc-b", 1, "image"),
    ]
    _write_interleaved_dataset(dataset_path, rows, max_rows_per_group=1)
    task = _single_fragment_task(dataset_path)

    result = InterleavedLanceReaderStage(
        fields=list(INTERLEAVED_SCHEMA.names),
        read_kwargs={"scanner_options": {"batch_size": 1}},
        streaming_read=True,
        max_batch_rows=3,
        include_lance_metadata=False,
    ).process(task)

    tables = _tables(result)
    assert [table["sample_id"].combine_chunks().to_pylist() for table in tables] == [
        ["doc-a", "doc-a", "doc-a"],
        ["doc-b", "doc-b"],
    ]


def test_interleaved_lance_reader_streaming_respects_max_output_rows_without_partial_sample(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "limited.lance"
    rows = [
        _row("doc-a", 0, "text", "a0"),
        _row("doc-a", 1, "image"),
        _row("doc-b", 0, "text", "b0"),
        _row("doc-b", 1, "image"),
    ]
    _write_interleaved_dataset(dataset_path, rows, max_rows_per_group=1)
    task = _single_fragment_task(dataset_path)

    result = InterleavedLanceReaderStage(
        fields=list(INTERLEAVED_SCHEMA.names),
        read_kwargs={"scanner_options": {"batch_size": 1}},
        streaming_read=True,
        max_output_rows=3,
        include_lance_metadata=False,
    ).process(task)

    tables = _tables(result)
    assert len(tables) == 1
    assert tables[0]["sample_id"].combine_chunks().to_pylist() == ["doc-a", "doc-a"]


def test_interleaved_lance_reader_non_streaming_records_stopped_early_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "limited-non-streaming.lance"
    rows = [
        _row("doc-a", 0, "text", "a0"),
        _row("doc-a", 1, "image"),
        _row("doc-b", 0, "text", "b0"),
        _row("doc-b", 1, "image"),
    ]
    _write_interleaved_dataset(dataset_path, rows)
    task = _single_fragment_task(dataset_path)

    result = InterleavedLanceReaderStage(
        fields=list(INTERLEAVED_SCHEMA.names),
        max_output_rows=3,
        include_lance_metadata=False,
    ).process(task)

    assert result._metadata["lance"]["streaming_read"] is False
    assert result._metadata["lance"]["stopped_early"] is True
    assert _tables(result)[0]["sample_id"].combine_chunks().to_pylist() == ["doc-a", "doc-a"]


def test_interleaved_lance_reader_adds_lance_metadata_columns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "metadata.lance"
    rows = [_row("doc-a", 0, "text", "a0"), _row("doc-a", 1, "image")]
    _write_interleaved_dataset(dataset_path, rows)
    task = _single_fragment_task(dataset_path)

    result = InterleavedLanceReaderStage(fields=list(INTERLEAVED_SCHEMA.names)).process(task)
    table = _tables(result)[0]

    assert LANCE_ROWID_COLUMN in table.column_names
    assert LANCE_ROWADDR_COLUMN in table.column_names
    assert LANCE_FRAGID_COLUMN in table.column_names
    assert result._metadata["lance"]["version"] == task.version
    assert result._metadata["lance"]["fragment_ids"] == task.data
    assert result._metadata["lance"]["has_stable_row_ids"] is False


def test_interleaved_lance_reader_streaming_rejects_blob_columns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "blob.lance"
    schema = INTERLEAVED_SCHEMA.append(lance.blob_field("payload"))
    row = _row("doc-a", 0, "text", "a0")
    table = pa.Table.from_pylist([row], schema=INTERLEAVED_SCHEMA).append_column(
        "payload",
        lance.blob_array([b"payload"]),
    )
    lance.write_dataset(
        table.cast(schema),
        str(dataset_path),
        mode="create",
        data_storage_version="2.2",
    )
    task = _single_fragment_task(dataset_path)

    with pytest.raises(NotImplementedError, match=r"does not support lance\.blob\.v2"):
        InterleavedLanceReaderStage(fields=list(schema.names), streaming_read=True).process(task)


def test_interleaved_lance_reader_streaming_honors_top_level_version_read_kwarg(tmp_path: Path) -> None:
    dataset_path = tmp_path / "streaming-version.lance"
    rows = [_row("doc-a", 0, "text", "a0"), _row("doc-a", 1, "image")]
    _write_interleaved_dataset(dataset_path, rows)
    version = lance.dataset(str(dataset_path)).version
    partitioner, reader = InterleavedLanceReader(
        path=str(dataset_path),
        fields=list(INTERLEAVED_SCHEMA.names),
        read_kwargs={"version": version, "scanner_options": {"batch_size": 1}},
        streaming_read=True,
        include_lance_metadata=False,
    ).decompose()
    task = partitioner.process(EmptyTask())[0]

    result = reader.process(task)

    assert _tables(result)[0]["sample_id"].combine_chunks().to_pylist() == ["doc-a", "doc-a"]
    assert result._metadata["lance"]["version"] == version
