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

"""Lance reader for row-wise interleaved multimodal datasets."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nemo_curator.core.utils import split_table_by_group
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.interleaved.utils.schema import align_interleaved_table
from nemo_curator.stages.text.io.reader.lance import LancePartitioningStage, LanceReaderStage
from nemo_curator.tasks import EmptyTask, InterleavedBatch, LanceReadTask

if TYPE_CHECKING:
    from nemo_curator.stages.text.io.reader.base import ReaderOutput


_GROUP_COLUMN = "sample_id"


def _validate_positive_optional(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        msg = f"{name} must be > 0, got {value}"
        raise ValueError(msg)


@dataclass
class InterleavedLanceReaderStage(LanceReaderStage):
    """Read Lance fragments into validated ``InterleavedBatch`` objects."""

    max_batch_bytes: int | None = None
    max_batch_rows: int | None = None
    name: str = "interleaved_lance_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fields is not None:
            missing = sorted(InterleavedBatch.REQUIRED_COLUMNS - set(self.fields))
            if missing:
                msg = f"Interleaved Lance fields omit required columns: {missing}"
                raise ValueError(msg)
        _validate_positive_optional("max_batch_bytes", self.max_batch_bytes)
        _validate_positive_optional("max_batch_rows", self.max_batch_rows)

    def process(self, task: LanceReadTask) -> InterleavedBatch | list[InterleavedBatch]:
        started = time.perf_counter()
        output: ReaderOutput = self.read_task(task, dict(self.read_kwargs or {}), self.fields)
        table = align_interleaved_table(output.data)
        self._validate_result(task, table)
        splits = split_table_by_group(
            table,
            _GROUP_COLUMN,
            max_batch_bytes=self.max_batch_bytes,
            max_batch_rows=self.max_batch_rows,
        )
        output_rows = sum(split.num_rows for split in splits)
        metadata = output.metadata if output.metadata is not None else task._metadata

        batches = [
            InterleavedBatch(
                dataset_name=task.dataset_name,
                data=split,
                _metadata=metadata,
                _stage_perf=list(task._stage_perf),
            )
            for split in splits
        ]
        for batch in batches:
            if batch.to_pyarrow().num_rows and not batch.validate():
                msg = f"Lance fragment task {task.task_id} is not a valid InterleavedBatch"
                raise ValueError(msg)

        self._log_metrics(
            {
                "reader_process_seconds": time.perf_counter() - started,
                "reader_output_splits": float(len(splits)),
                "reader_output_rows": float(output_rows),
                "reader_output_bytes": float(sum(split.nbytes for split in splits)),
            }
        )
        if not batches:
            return []
        return batches if len(batches) > 1 else batches[0]


@dataclass
class InterleavedLanceReader(CompositeStage[EmptyTask, InterleavedBatch]):
    """Partition and read a Lance dataset as row-wise interleaved batches."""

    path: str
    fragments_per_partition: int = 1
    fields: list[str] | None = None
    max_batch_bytes: int | None = None
    max_batch_rows: int | None = None
    read_kwargs: dict[str, Any] | None = None
    include_lance_metadata: bool = True
    fragment_ids: list[int] | None = None
    name: str = "interleaved_lance_reader"

    def __post_init__(self) -> None:
        super().__init__()
        self.read_kwargs = {} if self.read_kwargs is None else dict(self.read_kwargs)

    def decompose(self) -> list[ProcessingStage]:
        return [
            LancePartitioningStage(
                path=self.path,
                fragments_per_partition=self.fragments_per_partition,
                fragment_ids=self.fragment_ids,
                read_kwargs=self.read_kwargs,
            ),
            InterleavedLanceReaderStage(
                fields=self.fields,
                max_batch_bytes=self.max_batch_bytes,
                max_batch_rows=self.max_batch_rows,
                read_kwargs=self.read_kwargs,
                include_lance_metadata=self.include_lance_metadata,
            ),
        ]
