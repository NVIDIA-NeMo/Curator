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

import cudf
import pylibcudf as plc

from nemo_curator.stages.deduplication.shuffle_utils.rapidsmpf_shuffler import pylibcudf_to_cudf_dataframe
from nemo_curator.stages.deduplication.shuffle_utils.stage import ShuffleStage
from nemo_curator.tasks import FileGroupTask


class GroupByAggregation(ShuffleStage):
    """Group exact keys across Parquet files using the native GPU shuffle.

    Run with ``RayActorPoolExecutor``. ``groupby`` selects scalar key columns;
    null keys are retained. ``aggregations`` maps output names to
    ``(input_column, operation)`` pairs. Operations are ``sum``, ``count``
    (non-null values), ``size`` (all rows), ``min``, ``max``, ``collect``
    (all values), and ``unique`` (distinct values). Collections retain nulls
    and have unspecified order. Sum/min/max ignore nulls and return null for
    all-null groups. Numeric overflow follows cuDF semantics.

    For ``collect`` or ``unique``, an input may instead be a list of scalar
    columns. These are collected together as structs, preserving associations
    and records with null fields. Input schemas must match across files.

    Additional keyword arguments are forwarded to ``ShuffleStage``, including
    partition, memory, and I/O options. Read columns are projected from the
    grouping and aggregation fields. ``output_path`` must be a dedicated
    directory: construction overwrites it, as in ``ShuffleStage``.

    Each input task and output partition must fit GPU memory, including the
    aggregation working set. More partitions cannot split a single hot key.
    This stage does not support checkpoint resumption.

    Examples
    --------
    GroupByAggregation(
        groupby=["key"],
        aggregations={
            "rows": ("key", "size"),
            "total": ("value", "sum"),
            "sources": (["source", "version"], "unique"),
        },
        output_path="grouped",
        total_nparts=128,
    )
    """

    name = "GroupByAggregation"

    def __init__(
        self,
        groupby: list[str],
        aggregations: dict[str, tuple[str | list[str], str]],
        output_path: str,
        **shuffle_kwargs: object,
    ) -> None:
        if not groupby or len(set(groupby)) != len(groupby) or any(not field for field in groupby):
            msg = "groupby must contain distinct, nonempty column names."
            raise ValueError(msg)
        if not aggregations or any(not name or name in groupby for name in aggregations):
            msg = "Aggregation output names must be nonempty and separate from grouping columns."
            raise ValueError(msg)
        columns = list(groupby)
        for source, operation in aggregations.values():
            if operation not in {"sum", "count", "size", "min", "max", "collect", "unique"}:
                msg = f"Unsupported aggregation: {operation}"
                raise ValueError(msg)
            fields = [source] if isinstance(source, str) else source
            if not fields or any(not field for field in fields) or len(set(fields)) != len(fields):
                msg = "Aggregation inputs must contain distinct, nonempty column names."
                raise ValueError(msg)
            if not isinstance(source, str) and operation not in {"collect", "unique"}:
                msg = "Multiple input columns are supported only for collect and unique."
                raise ValueError(msg)
            columns.extend(fields)
        if shuffle_kwargs.get("total_nparts") is not None and shuffle_kwargs["total_nparts"] < 1:
            msg = "total_nparts must be positive."
            raise ValueError(msg)
        self.aggregations = aggregations
        self.output_columns = list(dict.fromkeys(columns))
        self.dataset_name = ""
        read_kwargs = dict(shuffle_kwargs.pop("read_kwargs", None) or {})
        read_kwargs["columns"] = self.output_columns
        super().__init__(shuffle_on=groupby, output_path=output_path, read_kwargs=read_kwargs, **shuffle_kwargs)

    def _aggregate(self, frame: cudf.DataFrame) -> cudf.DataFrame:
        if any(isinstance(dtype, (cudf.StructDtype, cudf.ListDtype)) for dtype in frame.dtypes):
            msg = "Input columns must be scalar; select flat fields to collect together."
            raise TypeError(msg)
        values = frame[self.shuffle_on].copy(deep=False)
        operations = {}
        for name, (source, operation) in self.aggregations.items():
            if isinstance(source, str):
                values[name] = (
                    cudf.Series(1, index=frame.index, dtype="int64") if operation == "size" else frame[source]
                )
            else:
                children, _ = frame[source].to_pylibcudf()
                # to_struct can inherit child null masks; the record itself must remain valid.
                column = plc.Column(plc.DataType(plc.TypeId.STRUCT), len(frame), None, None, 0, 0, children.columns())
                names = [(name, [(field, []) for field in source])]
                table = plc.io.TableWithMetadata(plc.Table([column]), names)
                records = cudf.DataFrame.from_pylibcudf(table)[name]
                records.index = frame.index
                values[name] = records
            operations[name] = {"size": "sum", "collect": list, "unique": list}.get(operation, operation)
        result = values.groupby(self.shuffle_on, dropna=False, sort=False).agg(operations)
        if set(result.columns) != set(self.aggregations):
            msg = "cuDF did not produce every requested aggregation; check input types and operations."
            raise TypeError(msg)
        result = result.reset_index()
        for name, (_, operation) in self.aggregations.items():
            if operation == "unique":
                result[name] = result[name].list.unique()
        return result

    def extract_and_write(self) -> list[FileGroupTask]:
        self._check_actor_obj()
        tasks = []
        # Keep nested nullability consistent across empty and nonempty partitions.
        write_kwargs = {"index": False, "force_nullable_schema": True, **self.write_kwargs}
        for partition_id, partition in self._actor_obj.extract():
            frame = pylibcudf_to_cudf_dataframe(partition, self.output_columns)
            result = self._aggregate(frame)
            output_file = f"{self.output_path}/part.{partition_id}.parquet"
            result.to_parquet(output_file, **write_kwargs)
            tasks.append(
                FileGroupTask(
                    dataset_name=f"{self.dataset_name}_{self.name}",
                    data=[output_file],
                    _metadata={
                        "partition_index": partition_id,
                        "num_input_rows": len(frame),
                        "num_output_rows": len(result),
                        "output_columns": list(result.columns),
                    },
                )
            )
        return tasks
