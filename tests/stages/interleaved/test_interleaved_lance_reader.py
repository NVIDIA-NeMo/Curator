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
from nemo_curator.utils.performance_utils import StagePerfStats

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


def test_interleaved_lance_reader_stage_reports_missing_required_fields() -> None:
    fields = ["sample_id"]
    missing = sorted(InterleavedBatch.REQUIRED_COLUMNS - set(fields))

    with pytest.raises(ValueError, match="omit required columns") as exc_info:
        InterleavedLanceReaderStage(fields=fields)

    error = str(exc_info.value)
    assert "omit required columns" in error
    assert all(field in error for field in missing)


def test_interleaved_lance_reader_decomposes() -> None:
    partitioner, reader = InterleavedLanceReader(
        path="example.lance",
        fields=list(INTERLEAVED_SCHEMA.names),
        max_batch_bytes=256 * 1024 * 1024,
        max_batch_rows=1024,
        fragments_per_partition=2,
        fragment_ids=[1],
    ).decompose()

    assert partitioner.fragments_per_partition == 2
    assert partitioner.fragment_ids == [1]
    assert reader.fields == list(INTERLEAVED_SCHEMA.names)
    assert reader.max_batch_bytes == 256 * 1024 * 1024
    assert reader.max_batch_rows == 1024
    assert reader.include_lance_metadata is True


@pytest.mark.parametrize(
    "field",
    ["max_batch_bytes", "max_batch_rows"],
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
    task._stage_perf = [StagePerfStats(stage_name="upstream")]

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
    batches = result if isinstance(result, list) else [result]
    assert all(batch._stage_perf == task._stage_perf for batch in batches)
    assert all(batch._stage_perf is not task._stage_perf for batch in batches)
    assert len({id(batch._stage_perf) for batch in batches}) == len(batches)


def test_interleaved_lance_reader_rejects_null_sample_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "null-sample-id.lance"
    row = _row("doc-a", 0, "text", "a0")
    row["sample_id"] = None
    table = pa.Table.from_pylist([row], schema=_nullable_sample_id_schema())
    lance.write_dataset(table, str(dataset_path), mode="create")
    task = _single_fragment_task(dataset_path)

    with pytest.raises(ValueError, match="null values"):
        InterleavedLanceReaderStage(
            fields=list(INTERLEAVED_SCHEMA.names),
            include_lance_metadata=False,
        ).process(task)


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


def test_interleaved_lance_reader_honors_top_level_version_read_kwarg(tmp_path: Path) -> None:
    dataset_path = tmp_path / "version.lance"
    old_rows = [_row("doc-old", 0, "text", "old"), _row("doc-old", 1, "image")]
    _write_interleaved_dataset(dataset_path, old_rows)
    old_version = lance.dataset(str(dataset_path)).version
    new_rows = [_row("doc-new", 0, "text", "new"), _row("doc-new", 1, "image")]
    lance.write_dataset(
        pa.Table.from_pylist(new_rows, schema=INTERLEAVED_SCHEMA),
        str(dataset_path),
        mode="overwrite",
        max_rows_per_file=len(new_rows),
    )
    partitioner, reader = InterleavedLanceReader(
        path=str(dataset_path),
        fields=list(INTERLEAVED_SCHEMA.names),
        read_kwargs={"version": old_version, "scanner_options": {"batch_size": 1}},
        include_lance_metadata=False,
    ).decompose()
    task = partitioner.process(EmptyTask())[0]

    result = reader.process(task)

    assert task.version == old_version
    assert _tables(result)[0]["sample_id"].combine_chunks().to_pylist() == ["doc-old", "doc-old"]
    assert result._metadata["lance"]["version"] == old_version
