# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pandas as pd
import pyarrow as pa
import pytest

from nemo_curator.stages.text.io.reader.parquet import ParquetReader, ParquetReaderStage
from nemo_curator.tasks.document import DocumentBatch
from nemo_curator.tasks.file_group import FileGroupTask


@pytest.fixture
def sample_parquet_files(tmp_path: Path) -> list[str]:
    """Create multiple Parquet files for testing."""
    files = []
    for i in range(3):
        file_path = tmp_path / f"test_{i}.parquet"
        # Create records with different ranges to ensure variety
        records = _sample_records(start=i * 2, n=2)
        _write_parquet_file(file_path, records)
        files.append(str(file_path))
    return files


@pytest.fixture
def parquet_file_group_tasks(sample_parquet_files: list[str]) -> list[FileGroupTask]:
    """Create multiple FileGroupTasks for parquet files."""
    return [
        FileGroupTask(task_id=f"task_{i}", dataset_name="test_dataset", data=[file_path], _metadata={})
        for i, file_path in enumerate(sample_parquet_files)
    ]


def _write_parquet_file(file_path: Path, records: list[dict]) -> None:
    # Use pandas to write a Parquet file
    df = pd.DataFrame(records)
    df.to_parquet(file_path, index=False)


def _sample_records(start: int = 0, n: int = 2) -> list[dict]:
    return [
        {
            "text": f"doc_{start + i}",
            "category": f"cat_{(start + i) % 3}",
            "score": float(start + i),
        }
        for i in range(n)
    ]


def _make_file_group_task(files: list[str]) -> FileGroupTask:
    return FileGroupTask(
        task_id="fg1",
        dataset_name="ds",
        data=files,
        reader_config={},
        _metadata={"source_files": files},
    )


def test_parquet_reader_stage_pandas_reads_and_concatenates(sample_parquet_files: list[str]):
    # Use the first two files from the fixture
    task = _make_file_group_task(sample_parquet_files[:2])
    stage = ParquetReaderStage(fields=None)

    out = stage.process(task)
    assert isinstance(out, DocumentBatch)

    df = out.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # 2 files * 2 records each = 4 records
    assert {"text", "category", "score"}.issubset(set(df.columns))


class TestParquetReaderStorageOptionsAndColumns:
    def test_columns_selection(self, tmp_path: Path):
        f = tmp_path / "a.parquet"
        _write_parquet_file(f, _sample_records(0, 3))
        task = _make_file_group_task([str(f)])
        stage = ParquetReaderStage(fields=["text"])  # select one column
        out = stage.process(task)
        df = out.to_pandas()
        assert list(df.columns) == ["text"]
        assert len(df) == 3

    def test_storage_options_via_read_kwargs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        f = tmp_path / "a.parquet"
        _write_parquet_file(f, _sample_records(0, 1))
        # Reader should use read_kwargs storage options
        task = _make_file_group_task([str(f)])
        stage = ParquetReaderStage(read_kwargs={"storage_options": {"auto_mkdir": True}})

        seen: dict[str, object] = {}

        def fake_read_parquet(_path: object, *_args: object, **kwargs: object) -> pd.DataFrame:
            seen["storage_options"] = kwargs.get("storage_options") if isinstance(kwargs, dict) else None
            return pd.DataFrame(_sample_records(0, 1))

        monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

        out = stage.process(task)
        assert seen["storage_options"] == {"auto_mkdir": True}
        df = out.to_pandas()
        assert len(df) == 1


def test_parquet_reader_stage_pandas_errors_when_some_columns_missing(tmp_path: Path):
    # Prepare files with known columns
    f = tmp_path / "a.parquet"
    _write_parquet_file(f, _sample_records(0, 3))

    task = _make_file_group_task([str(f)])
    stage = ParquetReaderStage(fields=["text", "does_not_exist"])

    with pytest.raises(pa.lib.ArrowInvalid):
        _ = stage.process(task)


def test_parquet_reader_stage_pandas_raises_when_all_columns_missing(tmp_path: Path):
    f = tmp_path / "a.parquet"
    _write_parquet_file(f, _sample_records(0, 2))

    task = _make_file_group_task([str(f)])
    stage = ParquetReaderStage(fields=["missing_only"])

    with pytest.raises(pa.lib.ArrowInvalid):
        _ = stage.process(task)


def test_parquet_reader_stage_pyarrow_reads_and_concatenates(tmp_path: Path):
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    _write_parquet_file(f1, _sample_records(0, 1))
    _write_parquet_file(f2, _sample_records(1, 2))

    task = _make_file_group_task([str(f1), str(f2)])
    stage = ParquetReaderStage(read_kwargs={"engine": "pyarrow"}, fields=None)

    out = stage.process(task)
    table = out.to_pyarrow()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    assert {"text", "category", "score"}.issubset(set(table.column_names))


def test_parquet_reader_stage_pyarrow_errors_when_some_columns_missing(tmp_path: Path):
    f = tmp_path / "a.parquet"
    _write_parquet_file(f, _sample_records(0, 4))

    task = _make_file_group_task([str(f)])
    stage = ParquetReaderStage(read_kwargs={"engine": "pyarrow"}, fields=["category", "not_there"])

    with pytest.raises(pa.lib.ArrowInvalid):
        _ = stage.process(task)


def test_base_reader_outputs_reflect_columns():
    stage = ParquetReaderStage(fields=["a", "b"])
    inputs, outputs = stage.inputs(), stage.outputs()
    assert inputs == ([], [])
    assert outputs == (["data"], ["a", "b"])


def test_parquet_reader_decompose_configuration(tmp_path: Path):
    reader = ParquetReader(
        file_paths=str(tmp_path),
        files_per_partition=3,
        fields=["text", "score"],
        read_kwargs={"engine": "pyarrow", "storage_options": {"anon": True}},
    )
    stages = reader.decompose()
    assert len(stages) == 2

    # First stage: FilePartitioningStage with parquet extension filter
    first = stages[0]
    from nemo_curator.stages.file_partitioning import FilePartitioningStage

    assert isinstance(first, FilePartitioningStage)
    assert first.file_extensions == [".parquet"]
    assert first.file_paths == str(tmp_path)
    assert first.files_per_partition == 3
    assert first.storage_options == {"anon": True}

    # Second stage: ParquetReaderStage config propagated (includes storage_options)
    second = stages[1]
    assert isinstance(second, ParquetReaderStage)
    assert second.fields == ["text", "score"]
    assert second.read_kwargs == {"engine": "pyarrow", "storage_options": {"anon": True}}


def test_parquet_reader_get_description():
    reader1 = ParquetReader(file_paths="s3://bucket/path", files_per_partition=5, fields=["text"])
    desc1 = reader1.get_description()
    assert "Read Parquet files from s3://bucket/path" in desc1
    assert "with 5 files per partition" in desc1
    assert "reading columns: ['text']" in desc1

    reader2 = ParquetReader(file_paths="/data", blocksize="128MB")
    desc2 = reader2.get_description()
    assert "Read Parquet files from /data" in desc2
    assert "with target blocksize 128MB" in desc2


def test_parquet_reader_non_document_task_type_not_supported():
    reader = ParquetReader(file_paths="/data", task_type="image")  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="Converting DocumentBatch to image is not supported yet"):
        _ = reader.decompose()


def test_parquet_reader_with_file_group_tasks_fixture(parquet_file_group_tasks: list[FileGroupTask]):
    """Demonstrate usage of parquet_file_group_tasks fixture."""
    stage = ParquetReaderStage(read_kwargs={}, fields=None)

    all_results = []
    for task in parquet_file_group_tasks:
        result = stage.process(task)
        df = result.to_pandas()
        assert len(df) == 2  # Each task has 1 file with 2 records
        assert {"text", "category", "score"}.issubset(set(df.columns))
        all_results.append(df)

    # Verify we processed all 3 tasks (files)
    assert len(all_results) == 3

    # Verify each file has unique records based on the start offset
    for i, df in enumerate(all_results):
        expected_texts = [f"doc_{i * 2}", f"doc_{i * 2 + 1}"]
        actual_texts = df["text"].tolist()
        assert actual_texts == expected_texts
