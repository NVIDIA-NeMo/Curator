# modality: text

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

from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.tasks import FileGroupTask

pytestmark = pytest.mark.gpu


@pytest.mark.usefixtures("shared_ray_client")
@pytest.mark.parametrize("groupby", [["key"], ["key", "version"]])
def test_global_aggregation(tmp_path: Path, groupby: list[str]) -> None:
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.deduplication.shuffle_utils.aggregation import GroupByAggregation

    schema = pa.schema(
        [
            ("key", pa.string()),
            ("value", pa.int64()),
            ("source", pa.string()),
            ("version", pa.string()),
            ("unused", pa.list_(pa.int64())),
        ]
    )
    rows = [
        {"key": key, "value": value, "source": source, "version": version, "unused": [1]}
        for key, value, source, version in [
            ("a", 2**31, "s1", "v1"),
            ("a", 3, "s1", "v1"),
            ("a", None, "s1", "v2"),
            ("a", -1, "s2", "v1"),
            ("a", 0, None, "v1"),
            ("a", 4, "s3", None),
            ("A", 5, "s1", "v1"),
            ("", None, None, None),
            (None, 6, "s1", "v2"),
            (None, 6, "s1", "v2"),
        ]
    ]
    tasks = []
    for index, shard in enumerate([rows[::2], rows[1::2], []]):
        path = tmp_path / f"input-{index}.parquet"
        pq.write_table(pa.Table.from_pylist(shard, schema=schema), path)
        tasks.append(FileGroupTask(dataset_name="records", data=[str(path)]))
    stage = GroupByAggregation(
        groupby=groupby,
        aggregations={
            "rows": ("key", "size"),
            "nonnull": ("value", "count"),
            "total": ("value", "sum"),
            "low": ("value", "min"),
            "high": ("value", "max"),
            "values": ("source", "collect"),
            "distinct": ("source", "unique"),
            "records": (["source", "version"], "collect"),
            "sources": (["source", "version"], "unique"),
        },
        output_path=str(tmp_path / "output"),
        total_nparts=16,
        read_kwargs={"use_pandas_metadata": False},
        write_kwargs={"row_group_size_rows": 1},
    ).with_(num_workers=2)
    results = Pipeline(name="groupby", stages=[stage]).run(RayActorPoolExecutor(), initial_tasks=tasks)
    paths = [path for task in results for path in task.data]
    records = pq.read_table(paths).to_pylist()
    expected = defaultdict(list)
    for row in rows:
        expected[tuple(row[field] for field in groupby)].append(row)
    assert len(records) == len(expected)
    assert sum(task._metadata["num_input_rows"] for task in results) == len(rows)
    assert sum(task._metadata["num_output_rows"] for task in results) == len(expected)
    for row in records:
        group = expected[tuple(row[field] for field in groupby)]
        numbers = [item["value"] for item in group if item["value"] is not None]
        assert row["rows"] == len(group)
        assert row["nonnull"] == len(numbers)
        assert row["total"] == (sum(numbers) if numbers else None)
        assert row["low"] == (min(numbers) if numbers else None)
        assert row["high"] == (max(numbers) if numbers else None)
        assert Counter(row["values"]) == Counter(item["source"] for item in group)
        assert Counter(row["distinct"]) == Counter({item["source"] for item in group})
        pairs = [(item["source"], item["version"]) for item in group]
        assert Counter((item["source"], item["version"]) for item in row["records"]) == Counter(pairs)
        assert Counter((item["source"], item["version"]) for item in row["sources"]) == Counter(set(pairs))
    for path in paths:
        assert pq.read_schema(path) == pq.read_schema(paths[0])
        metadata = pq.read_metadata(path)
        assert all(metadata.row_group(i).num_rows <= 1 for i in range(metadata.num_row_groups))


@pytest.mark.usefixtures("shared_ray_client")
def test_idle_worker_metadata_and_projected_filter(tmp_path: Path) -> None:
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.deduplication.shuffle_utils.aggregation import GroupByAggregation

    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"key": ["a", "a", "b", "b"], "keep": [True, False, True, True]}), path)
    stage = GroupByAggregation(
        groupby=["key"],
        aggregations={"rows": ("key", "size")},
        output_path=str(tmp_path / "output"),
        total_nparts=4,
        read_kwargs={"filters": [("keep", "==", True)]},
    ).with_(num_workers=2)
    results = Pipeline(name="groupby", stages=[stage]).run(
        RayActorPoolExecutor(), initial_tasks=[FileGroupTask(dataset_name="records", data=[str(path)])]
    )
    assert len(results) == 4
    assert {task.dataset_name for task in results} == {"records_GroupByAggregation"}
    table = pq.read_table([path for task in results for path in task.data])
    assert table.column_names == ["key", "rows"]
    assert sorted(table.to_pylist(), key=lambda row: row["key"]) == [{"key": "a", "rows": 1}, {"key": "b", "rows": 2}]


def test_empty_and_null_records(tmp_path: Path) -> None:
    import cudf

    from nemo_curator.stages.deduplication.shuffle_utils.aggregation import GroupByAggregation

    stage = GroupByAggregation(
        ["key"], {"rows": ("key", "size"), "records": (["source", "version"], "unique")}, str(tmp_path / "out")
    )
    frame = cudf.DataFrame({field: cudf.Series([None, None], dtype="str") for field in ["key", "source", "version"]})
    frame.index = [10, 20]
    assert stage._aggregate(frame).to_arrow().to_pylist() == [
        {"key": None, "rows": 2, "records": [{"source": None, "version": None}]}
    ]
    empty = stage._aggregate(frame.head(0))
    assert len(empty) == 0
    empty.to_parquet(tmp_path / "empty.parquet", index=False, force_nullable_schema=True)
    stage._aggregate(frame).to_parquet(tmp_path / "nulls.parquet", index=False, force_nullable_schema=True)
    assert pq.read_schema(tmp_path / "empty.parquet") == pq.read_schema(tmp_path / "nulls.parquet")


def test_unsupported_dtype_is_not_silently_omitted(tmp_path: Path) -> None:
    import cudf

    from nemo_curator.stages.deduplication.shuffle_utils.aggregation import GroupByAggregation

    stage = GroupByAggregation(["key"], {"rows": ("key", "size"), "invalid": ("value", "sum")}, str(tmp_path / "out"))
    with pytest.raises(TypeError):
        stage._aggregate(cudf.DataFrame({"key": ["a"], "value": ["text"]}))


@pytest.mark.parametrize(
    ("groupby", "aggregations", "kwargs"),
    [
        ([], {"rows": ("key", "size")}, {}),
        (["key", "key"], {"rows": ("key", "size")}, {}),
        (["key"], {}, {}),
        (["key"], {"key": ("key", "size")}, {}),
        (["key"], {"rows": ("key", "mean")}, {}),
        (["key"], {"rows": (["a", "b"], "sum")}, {}),
        (["key"], {"rows": ([], "collect")}, {}),
        (["key"], {"rows": ("key", "size")}, {"total_nparts": 0}),
    ],
)
def test_invalid_options_preserve_output(tmp_path: Path, groupby: list, aggregations: dict, kwargs: dict) -> None:
    from nemo_curator.stages.deduplication.shuffle_utils.aggregation import GroupByAggregation

    marker = tmp_path / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match=r"must|Unsupported|supported"):
        GroupByAggregation(groupby, aggregations, str(tmp_path), **kwargs)
    assert marker.read_text() == "keep"
