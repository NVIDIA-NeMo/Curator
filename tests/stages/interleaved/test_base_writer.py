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

"""Tests for BaseInterleavedWriter: on_materialize_error modes, _write_dataframe,
and _align_output schema alignment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved.io.writers.base import BaseInterleavedWriter
from nemo_curator.stages.interleaved.io.writers.tabular import InterleavedParquetWriterStage
from nemo_curator.tasks import InterleavedBatch

_BASE_ROW: dict[str, Any] = {
    "sample_id": "s1",
    "position": 0,
    "modality": "text",
    "content_type": "text/plain",
    "text_content": "hello",
    "binary_content": None,
    "source_ref": None,
}


def _make_task(rows: list[dict[str, Any]]) -> InterleavedBatch:
    return InterleavedBatch(
        task_id="base_writer_test",
        dataset_name="test",
        data=pd.DataFrame(rows),
        _metadata={"source_files": ["x.tar"]},
    )


def _writer(tmp_path: Path, on_error: str = "error") -> InterleavedParquetWriterStage:
    return InterleavedParquetWriterStage(
        path=str(tmp_path / "out"),
        materialize_on_write=False,
        mode="overwrite",
        on_materialize_error=on_error,
    )


# ---------------------------------------------------------------------------
# on_materialize_error modes
# ---------------------------------------------------------------------------


def test_on_materialize_error_error_raises(tmp_path: Path) -> None:
    """on_materialize_error='error' (default) raises RuntimeError on first error."""
    rows = [{**_BASE_ROW, "materialize_error": "404 Not Found"}]
    with pytest.raises(RuntimeError, match="Materialization failed"):
        _writer(tmp_path, on_error="error").process(_make_task(rows))


def test_on_materialize_error_warn_keeps_all_rows(tmp_path: Path) -> None:
    """on_materialize_error='warn' keeps all rows, including errored ones."""
    rows = [
        {**_BASE_ROW, "position": 0, "materialize_error": "some error"},
        {**_BASE_ROW, "position": 1, "text_content": "world", "materialize_error": None},
    ]
    result = _writer(tmp_path, on_error="warn").process(_make_task(rows))
    assert pq.read_table(result.data[0]).num_rows == 2


def test_on_materialize_error_drop_row_removes_errored_rows(tmp_path: Path) -> None:
    """on_materialize_error='drop_row' removes only rows with errors."""
    rows = [
        {**_BASE_ROW, "position": 0, "materialize_error": "fetch failed"},
        {**_BASE_ROW, "position": 1, "text_content": "good", "materialize_error": None},
        {**_BASE_ROW, "position": 2, "text_content": "also good", "materialize_error": None},
    ]
    result = _writer(tmp_path, on_error="drop_row").process(_make_task(rows))
    assert pq.read_table(result.data[0]).num_rows == 2  # position-0 row dropped


def test_on_materialize_error_drop_sample_removes_entire_sample(tmp_path: Path) -> None:
    """on_materialize_error='drop_sample' removes ALL rows for samples with any error."""
    rows = [
        # s1: has one error → both s1 rows dropped
        {**_BASE_ROW, "sample_id": "s1", "position": 0, "materialize_error": "timeout"},
        {**_BASE_ROW, "sample_id": "s1", "position": 1, "text_content": "meta", "materialize_error": None},
        # s2: no errors → kept
        {**_BASE_ROW, "sample_id": "s2", "position": 0, "text_content": "good", "materialize_error": None},
    ]
    result = _writer(tmp_path, on_error="drop_sample").process(_make_task(rows))
    written = pq.read_table(result.data[0]).to_pandas()
    assert len(written) == 1
    assert "s1" not in written["sample_id"].tolist()
    assert "s2" in written["sample_id"].tolist()


def test_no_materialize_error_column_returns_all_rows(tmp_path: Path) -> None:
    """If materialize_error column is absent, all rows returned unchanged (early-exit path)."""
    rows = [{**_BASE_ROW}]  # no materialize_error key → column absent from DataFrame
    result = _writer(tmp_path, on_error="error").process(_make_task(rows))
    assert pq.read_table(result.data[0]).num_rows == 1


def test_all_null_materialize_error_returns_all_rows(tmp_path: Path) -> None:
    """materialize_error column present but entirely None → no rows dropped."""
    rows = [{**_BASE_ROW, "materialize_error": None}]
    result = _writer(tmp_path, on_error="error").process(_make_task(rows))
    assert pq.read_table(result.data[0]).num_rows == 1


# ---------------------------------------------------------------------------
# _write_dataframe: NotImplementedError on abstract base
# ---------------------------------------------------------------------------


def test_base_write_dataframe_raises_not_implemented(tmp_path: Path) -> None:
    """BaseInterleavedWriter._write_dataframe() raises NotImplementedError."""

    @dataclass
    class _StubWriter(BaseInterleavedWriter):
        file_extension: str = "stub"
        name: str = "stub_writer"

    writer = _StubWriter(path=str(tmp_path / "out"), mode="overwrite")
    with pytest.raises(NotImplementedError, match="_write_dataframe"):
        writer._write_dataframe(pd.DataFrame(), "dummy.stub", {})


# ---------------------------------------------------------------------------
# _align_output: with and without schema
# ---------------------------------------------------------------------------


def test_align_output_with_schema_drops_extra_columns(tmp_path: Path) -> None:
    """_align_output with schema= set drops columns not in the schema."""
    target_schema = pa.schema(
        [
            pa.field("sample_id", pa.string()),
            pa.field("position", pa.int32()),
            pa.field("modality", pa.string()),
        ]
    )
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "out"),
        materialize_on_write=False,
        mode="overwrite",
        schema=target_schema,
    )
    df = pd.DataFrame([{"sample_id": "s1", "position": 0, "modality": "text", "content_type": "text/plain"}])
    result = writer._align_output(df)
    assert "content_type" not in result.columns
    assert "sample_id" in result.columns


def test_align_output_without_schema_preserves_extra_columns(tmp_path: Path) -> None:
    """_align_output with schema=None (default) preserves passthrough columns."""
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "out"),
        materialize_on_write=False,
        mode="overwrite",
        schema=None,
    )
    df = pd.DataFrame([{"sample_id": "s1", "position": 0, "modality": "text", "my_custom_column": "keep_me"}])
    result = writer._align_output(df)
    assert "my_custom_column" in result.columns
    assert result["my_custom_column"].iloc[0] == "keep_me"


# ---------------------------------------------------------------------------
# process(): source_files determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metadata",
    [
        {},  # source_files key absent
        {"source_files": []},  # present but empty — matches text BaseWriter behaviour
    ],
    ids=["absent", "empty_list"],
)
def test_process_no_source_files_uses_uuid(tmp_path: Path, metadata: dict) -> None:
    """source_files absent or empty → UUID fallback (non-deterministic), matching text BaseWriter."""
    rows = [{**_BASE_ROW}]
    task = InterleavedBatch(
        task_id="no-source",
        dataset_name="test",
        data=pd.DataFrame(rows),
        _metadata=metadata,
    )
    writer1 = InterleavedParquetWriterStage(
        path=str(tmp_path / "out1"),
        materialize_on_write=False,
        mode="overwrite",
    )
    writer2 = InterleavedParquetWriterStage(
        path=str(tmp_path / "out2"),
        materialize_on_write=False,
        mode="overwrite",
    )
    result1 = writer1.process(task)
    result2 = writer2.process(task)

    name1 = Path(result1.data[0]).name
    name2 = Path(result2.data[0]).name
    # UUIDs are random — the two names should differ (astronomically unlikely to collide)
    assert name1 != name2, f"source_files={metadata.get('source_files')!r} should produce different UUIDs each call"


def test_base_writer_inputs_and_outputs(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    task_attrs, data_attrs = writer.inputs()
    assert task_attrs == ["data"]
    assert data_attrs == []

    task_attrs, data_attrs = writer.outputs()
    assert task_attrs == ["data"]
    assert data_attrs == []
