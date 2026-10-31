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
from types import SimpleNamespace
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.deduplication.shuffle_utils.partitioned_exact_filter import (
    PartitionedExactFilterStage,
)
from nemo_curator.tasks import FileGroupTask


def make_stage(tmp_path: Path, **kwargs) -> PartitionedExactFilterStage:
    reference_path = tmp_path / "reference"
    reference_path.mkdir(exist_ok=True)
    return PartitionedExactFilterStage(
        key_fields=kwargs.pop("key_fields", "key"),
        reference_path=str(reference_path),
        output_path=str(tmp_path / "output"),
        total_partitions=kwargs.pop("total_partitions", 2),
        **kwargs,
    )


@pytest.mark.parametrize("key_fields", [[], "", ["key", "key"], ["key", ""]])
def test_rejects_invalid_key_fields(tmp_path: Path, key_fields: str | list[str]) -> None:
    with pytest.raises(ValueError, match="key_fields"):
        make_stage(tmp_path, key_fields=key_fields)


@pytest.mark.parametrize("total_partitions", [0, -1, True, 1.5])
def test_rejects_invalid_partition_count(tmp_path: Path, total_partitions: int) -> None:
    with pytest.raises(ValueError, match="total_partitions"):
        make_stage(tmp_path, total_partitions=total_partitions)


@pytest.mark.parametrize("kwarg", ["columns", "filters", "row_groups", "nrows", "skip_rows"])
@pytest.mark.parametrize("side", ["left", "reference"])
def test_rejects_row_subsetting_read_kwargs(tmp_path: Path, kwarg: str, side: str) -> None:
    kwargs = {"read_kwargs" if side == "left" else "reference_read_kwargs": {kwarg: object()}}
    with pytest.raises(ValueError, match=kwarg):
        make_stage(tmp_path, **kwargs)


@pytest.mark.parametrize(
    "kwarg",
    ["index", "metadata_file_path", "partition_cols", "partition_file_name", "partition_offsets"],
)
def test_rejects_non_single_file_write_kwargs(tmp_path: Path, kwarg: str) -> None:
    with pytest.raises(ValueError, match=kwarg):
        make_stage(tmp_path, write_kwargs={kwarg: object()})


def test_rejects_invalid_filter_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"leftsemi.*leftanti"):
        make_stage(tmp_path, mode="inner")


def test_rejects_destructive_output_root(tmp_path: Path) -> None:
    reference_path = tmp_path / "partitions"
    reference_path.mkdir()
    with pytest.raises(ValueError, match="distinct from reference_path"):
        PartitionedExactFilterStage(
            key_fields="key",
            reference_path=str(reference_path),
            output_path=str(reference_path),
            total_partitions=1,
        )


def test_rejects_destructive_output_root_alias(tmp_path: Path) -> None:
    reference_path = tmp_path / "partitions"
    reference_path.mkdir()
    alias_path = tmp_path / "partition-alias"
    alias_path.symlink_to(reference_path, target_is_directory=True)
    with pytest.raises(ValueError, match="distinct from reference_path"):
        PartitionedExactFilterStage(
            key_fields="key",
            reference_path=str(reference_path),
            output_path=str(alias_path),
            total_partitions=1,
        )


@pytest.mark.parametrize(
    ("metadata", "exception", "message"),
    [
        ({"total_partitions": 2}, TypeError, "partition_index"),
        ({"partition_index": True, "total_partitions": 2}, TypeError, "partition_index"),
        ({"partition_index": -1, "total_partitions": 2}, ValueError, "outside"),
        ({"partition_index": 2, "total_partitions": 2}, ValueError, "outside"),
        ({"partition_index": 0, "total_partitions": 3}, ValueError, "does not match"),
    ],
)
def test_requires_complete_shuffle_partition_metadata(
    tmp_path: Path,
    metadata: dict[str, int],
    exception: type[Exception],
    message: str,
) -> None:
    stage = make_stage(tmp_path)
    task = FileGroupTask(dataset_name="left", data=["left.parquet"], _metadata=metadata)
    with pytest.raises(exception, match=message):
        stage.process(task)


def test_requires_matching_reference_partition(tmp_path: Path) -> None:
    stage = make_stage(tmp_path)
    task = FileGroupTask(
        dataset_name="left",
        data=["left.parquet"],
        _metadata={"partition_index": 1, "total_partitions": 2},
    )
    with pytest.raises(FileNotFoundError, match=r"part\.1\.parquet"):
        stage.process(task)


def test_rejects_left_input_overwrite(tmp_path: Path) -> None:
    stage = make_stage(tmp_path)
    task = FileGroupTask(
        dataset_name="left",
        data=[str(tmp_path / "output" / "part.0.parquet")],
        _metadata={"partition_index": 0, "total_partitions": 2},
    )
    with pytest.raises(ValueError, match="overwrite its left input"):
        stage.process(task)


def test_rejects_left_input_overwrite_alias(tmp_path: Path) -> None:
    stage = make_stage(tmp_path)
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(tmp_path / "output", target_is_directory=True)
    task = FileGroupTask(
        dataset_name="left",
        data=[str(output_alias / "." / "part.0.parquet")],
        _metadata={"partition_index": 0, "total_partitions": 2},
    )
    with pytest.raises(ValueError, match="overwrite its left input"):
        stage.process(task)


@pytest.mark.parametrize(("mode", "output_rows"), [("leftanti", 3), ("leftsemi", 2)])
def test_process_preserves_stage_perf_without_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    output_rows: int,
) -> None:
    class FakeFrame:
        columns: ClassVar[list[str]] = ["key", "value"]

        def __init__(self, rows: int) -> None:
            self.rows = rows
            self.write: tuple[str, bool] | None = None

        def __len__(self) -> int:
            return self.rows

        def merge(self, other: "FakeFrame", *, how: str, on: list[str]) -> "FakeFrame":
            assert len(other) == 2
            assert how == mode
            assert on == ["key"]
            return FakeFrame(output_rows)

        def to_parquet(self, path: str, *, index: bool, **kwargs) -> None:
            assert not kwargs
            self.write = (path, index)

    left = FakeFrame(5)
    reference = FakeFrame(2)

    def read_parquet(path: str | list[str], **kwargs) -> FakeFrame:
        if isinstance(path, list):
            assert kwargs == {}
            return left
        assert kwargs == {"columns": ["key"]}
        return reference

    monkeypatch.setitem(__import__("sys").modules, "cudf", SimpleNamespace(read_parquet=read_parquet))
    stage = make_stage(tmp_path, total_partitions=1, mode=mode)
    pq.write_table(pa.table({"key": pa.array([], type=pa.string())}), tmp_path / "reference" / "part.0.parquet")
    task = FileGroupTask(
        dataset_name="left",
        data=[str(tmp_path / "left.parquet")],
        _metadata={"partition_index": 0, "total_partitions": 1},
        _stage_perf=["upstream-stage"],
    )

    result = stage.process(task)

    assert result._stage_perf == task._stage_perf
    assert result._metadata["input_rows"] == 5
    assert result._metadata["reference_rows"] == 2
    assert result._metadata["output_rows"] == output_rows


def test_reference_schema_error_precedes_cudf_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def read_parquet(_path: str | list[str], **_kwargs) -> None:
        pytest.fail("cuDF data read should not run after schema validation fails")

    monkeypatch.setitem(__import__("sys").modules, "cudf", SimpleNamespace(read_parquet=read_parquet))
    stage = make_stage(tmp_path, total_partitions=1)
    pq.write_table(pa.table({"other": pa.array([], type=pa.string())}), tmp_path / "reference" / "part.0.parquet")
    task = FileGroupTask(
        dataset_name="left",
        data=[str(tmp_path / "left.parquet")],
        _metadata={"partition_index": 0, "total_partitions": 1},
    )

    with pytest.raises(ValueError, match="reference partition is missing exact key columns"):
        stage.process(task)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("mode", "expected_ids"),
    [
        ("leftsemi", [1, 2, 4]),
        ("leftanti", [0, 3]),
    ],
)
def test_exact_filter_preserves_left_multiplicity(
    tmp_path: Path,
    mode: str,
    expected_ids: list[int],
) -> None:
    import cudf

    reference_path = tmp_path / "reference"
    reference_path.mkdir()
    left_path = tmp_path / "left.parquet"
    cudf.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "key": ["a", "b", "b", "c", "d"],
            "value": [10, 20, 21, 30, 40],
        }
    ).to_parquet(left_path)
    cudf.DataFrame({"key": ["b", "d", "d"]}).to_parquet(reference_path / "part.0.parquet")

    stage = PartitionedExactFilterStage(
        key_fields="key",
        reference_path=str(reference_path),
        output_path=str(tmp_path / "output"),
        total_partitions=1,
        mode=mode,
    )
    task = FileGroupTask(
        dataset_name="left",
        data=[str(left_path)],
        _metadata={"partition_index": 0, "total_partitions": 1},
        _stage_perf=["upstream-stage"],
    )
    result = stage.process(task)

    output = cudf.read_parquet(result.data[0]).sort_values("id")
    assert output["id"].to_arrow().to_pylist() == expected_ids
    assert list(output.columns) == ["id", "key", "value"]
    assert result._metadata["input_rows"] == 5
    assert result._metadata["reference_rows"] == 3
    assert result._metadata["output_rows"] == len(expected_ids)
    assert result._stage_perf == task._stage_perf
