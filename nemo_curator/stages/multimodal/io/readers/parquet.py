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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow as pa

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.stages.multimodal.io.readers.base import BaseMultimodalReader
from nemo_curator.tasks import FileGroupTask, MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA


@dataclass
class MultimodalParquetReaderStage(BaseMultimodalReader):
    """Read parquet files into MultiBatchTask(s) with the same row format as WebdatasetReaderStage."""

    fields: list[str] | None = None
    max_batch_bytes: int | None = None
    name: str = "multimodal_parquet_reader"

    def _read_parquet(self, paths: list[str]) -> pd.DataFrame:
        read_kwargs = dict(self.read_kwargs)
        if self.fields is not None:
            read_kwargs["columns"] = self.fields
        if "engine" not in read_kwargs:
            read_kwargs["engine"] = "pyarrow"
        if "dtype_backend" not in read_kwargs:
            read_kwargs["dtype_backend"] = "pyarrow"
        return pd.concat(
            (pd.read_parquet(p, **read_kwargs) for p in paths),
            ignore_index=True,
        )

    def process(self, task: FileGroupTask) -> MultiBatchTask | list[MultiBatchTask]:
        if not task.data:
            return MultiBatchTask(
                task_id=f"{task.task_id}_processed",
                dataset_name=task.dataset_name,
                data=pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA),
                _metadata=dict(task._metadata),
                _stage_perf=task._stage_perf,
            )
        df = self._read_parquet(task.data)
        if df.empty:
            return MultiBatchTask(
                task_id=f"{task.task_id}_processed",
                dataset_name=task.dataset_name,
                data=pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA),
                _metadata=dict(task._metadata),
                _stage_perf=task._stage_perf,
            )
        storage_options = self.read_kwargs.get("storage_options") or {}
        if self.max_batch_bytes is not None and self.max_batch_bytes > 0:
            table = pa.Table.from_pandas(df, preserve_index=False)
            if "sample_id" not in table.column_names:
                batches = [table]
            else:
                batches = split_table_by_group_max_bytes(
                    table, "sample_id", self.max_batch_bytes
                )
        else:
            batches = [df]
        out: list[MultiBatchTask] = []
        for idx, data in enumerate(batches):
            task_id = (
                f"{task.task_id}_processed"
                if len(batches) == 1
                else f"{task.task_id}_processed_{idx:05d}"
            )
            metadata = dict(task._metadata)
            if storage_options:
                metadata["source_storage_options"] = storage_options
            out.append(
                MultiBatchTask(
                    task_id=task_id,
                    dataset_name=task.dataset_name,
                    data=data,
                    _metadata=metadata,
                    _stage_perf=task._stage_perf,
                )
            )
        return out[0] if len(out) == 1 else out


__all__ = ["MultimodalParquetReaderStage"]
