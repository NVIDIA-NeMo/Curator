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
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved.io.readers.parquet import InterleavedParquetReaderStage
from nemo_curator.stages.interleaved.utils.schema import reconcile_schema
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

from .conftest import make_row


def _write_synthetic_parquet(path: Path, num_samples: int = 3) -> str:
    rows = []
    for i in range(num_samples):
        sid = f"sample_{i}"
        rows.append(make_row(sample_id=sid, position=-1, modality="metadata"))
        rows.append(make_row(sample_id=sid, position=0, modality="text", text_content=f"Hello from sample {i}"))
        rows.append(make_row(sample_id=sid, position=1, modality="image"))
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    file_path = str(path / "test.parquet")
    pq.write_table(table, file_path)
    return file_path


def test_read_basic(tmp_path: Path) -> None:
    pq_path = _write_synthetic_parquet(tmp_path)
    task = FileGroupTask(task_id="t0", dataset_name="test", data=[pq_path], _metadata={"source_files": [pq_path]})
    result = InterleavedParquetReaderStage().process(task)
    assert isinstance(result, InterleavedBatch)
    assert result.count() == 9
    assert result.count(modality="metadata") == 3
    assert result.count(modality="text") == 3
    assert result.count(modality="image") == 3


def test_read_with_fields_subset(tmp_path: Path) -> None:
    pq_path = _write_synthetic_parquet(tmp_path)
    task = FileGroupTask(task_id="t0", dataset_name="test", data=[pq_path], _metadata={"source_files": [pq_path]})
    result = InterleavedParquetReaderStage(fields=["sample_id", "position", "modality"]).process(task)
    assert isinstance(result, InterleavedBatch)
    cols = set(result.to_pyarrow().column_names)
    assert {"sample_id", "position", "modality"} <= cols


def test_read_missing_columns_filled_as_null(tmp_path: Path) -> None:
    pq_path = _write_synthetic_parquet(tmp_path)
    task = FileGroupTask(task_id="t0", dataset_name="test", data=[pq_path], _metadata={"source_files": [pq_path]})
    result = InterleavedParquetReaderStage(fields=["sample_id", "position", "modality", "nonexistent_col"]).process(
        task
    )
    table = result.to_pyarrow()
    assert "nonexistent_col" in table.column_names
    assert table.column("nonexistent_col").null_count == table.num_rows


def test_reconcile_schema_large_string_compat() -> None:
    inferred = pa.schema(
        [
            pa.field("sample_id", pa.large_string()),
            pa.field("position", pa.int32()),
            pa.field("modality", pa.large_string()),
            pa.field("extra_col", pa.float64()),
        ]
    )
    reconciled = reconcile_schema(inferred)
    assert reconciled.field("sample_id").type == pa.large_string()
    assert reconciled.field("modality").type == pa.large_string()
    assert reconciled.field("position").type == pa.int32()
    assert reconciled.field("extra_col").type == pa.float64()


def test_reconcile_schema_preserves_small_types() -> None:
    inferred = pa.schema(
        [
            pa.field("sample_id", pa.string()),
            pa.field("position", pa.int32()),
            pa.field("modality", pa.string()),
        ]
    )
    reconciled = reconcile_schema(inferred)
    assert reconciled.field("sample_id").type == pa.string()
    assert reconciled.field("position").type == pa.int32()


def test_read_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pq1 = _write_synthetic_parquet(tmp_path / "a", num_samples=2)
    pq2 = _write_synthetic_parquet(tmp_path / "b", num_samples=3)
    task = FileGroupTask(task_id="t0", dataset_name="test", data=[pq1, pq2], _metadata={"source_files": [pq1, pq2]})
    result = InterleavedParquetReaderStage().process(task)
    assert isinstance(result, InterleavedBatch)
    assert result.count() == 15


@pytest.mark.parametrize(
    ("schema", "schema_overrides", "expected_cols"),
    [
        pytest.param(
            pa.schema(
                [
                    pa.field("sample_id", pa.string()),
                    pa.field("position", pa.int32()),
                    pa.field("modality", pa.string()),
                ]
            ),
            None,
            ["sample_id", "position", "modality"],
            id="explicit_schema_drops_extra_cols",
        ),
        pytest.param(
            None,
            {"sample_id": pa.large_string()},
            None,  # all INTERLEAVED_SCHEMA cols present; check sample_id type instead
            id="schema_overrides_applied",
        ),
    ],
)
def test_explicit_schema_aligns_table(
    tmp_path: Path,
    schema: pa.Schema | None,
    schema_overrides: dict | None,
    expected_cols: list[str] | None,
) -> None:
    pq_path = _write_synthetic_parquet(tmp_path)
    task = FileGroupTask(task_id="t0", dataset_name="test", data=[pq_path], _metadata={"source_files": [pq_path]})
    result = InterleavedParquetReaderStage(schema=schema, schema_overrides=schema_overrides).process(task)
    assert isinstance(result, InterleavedBatch)
    table = result.to_pyarrow()
    if expected_cols is not None:
        assert table.column_names == expected_cols
    else:
        assert table.schema.field("sample_id").type == pa.large_string()
