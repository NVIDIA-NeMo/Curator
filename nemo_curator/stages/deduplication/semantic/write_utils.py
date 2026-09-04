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

import itertools
import math
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock
from typing import TYPE_CHECKING, Protocol

import cupy as cp
import numpy as np
from cudf.io.parquet import ParquetDatasetWriter, ParquetWriter
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

if TYPE_CHECKING:
    import cudf


class PartitionedParquetWriter(Protocol):
    batch_count: int
    batch_rows: int
    batch_bytes: int

    @property
    def group_count(self) -> int: ...

    def write_table(self, frame: "cudf.DataFrame") -> None: ...

    def close(self) -> None: ...


def local_path(path: str, storage_options: dict | None) -> str | None:
    fs, resolved = url_to_fs(path, **(storage_options or {}))
    return resolved if isinstance(fs, LocalFileSystem) else None


def interval_wall_time(intervals: list[tuple[float, float]]) -> float:
    intervals = sorted(intervals)
    if not intervals:
        return 0.0
    wall_time = 0.0
    start, stop = intervals[0]
    for next_start, next_stop in intervals[1:]:
        if next_start > stop:
            wall_time += stop - start
            start, stop = next_start, next_stop
        else:
            stop = max(stop, next_stop)
    return wall_time + stop - start


class RollingParquetWriter:
    """Keep each partition's list child below cuDF's column-size limit."""

    # TODO(https://github.com/NVIDIA/cudf/issues/23378): Remove logical rollover
    # once the fix is available in our cuDF pin.

    def __init__(
        self,
        create_writer: Callable[[int], ParquetDatasetWriter],
        n_partitions: int,
        max_rows_per_partition: int,
        partition_column: str,
    ) -> None:
        self._create_writer = create_writer
        self._partition_column = partition_column
        self._writers: dict[int, ParquetDatasetWriter] = {}
        self._generation = np.zeros(n_partitions, dtype=np.int64)
        self._rows = np.zeros(n_partitions, dtype=np.int64)
        self._target = max_rows_per_partition
        self.batch_count = 0
        self.batch_rows = 0
        self.batch_bytes = 0

    def _writer(self, generation: int) -> ParquetDatasetWriter:
        if generation not in self._writers:
            self._writers[generation] = self._create_writer(generation)
        return self._writers[generation]

    def write_table(self, frame: "cudf.DataFrame") -> None:
        counts = cp.asnumpy(cp.bincount(frame[self._partition_column].values, minlength=len(self._generation)))
        if counts.max(initial=0) > self._target:
            slices = int(np.ceil(counts.max() / self._target))
            rows = int(np.ceil(len(frame) / slices))
            for start in range(0, len(frame), rows):
                self.write_table(frame.iloc[start : start + rows])
            return

        self.batch_count += 1
        self.batch_rows = max(self.batch_rows, len(frame))
        self.batch_bytes = max(self.batch_bytes, int(frame.memory_usage(deep=True).sum()))

        present = counts > 0
        roll = present & (self._rows > 0) & (self._rows + counts > self._target)
        self._generation[roll] += 1
        self._rows[roll] = 0
        generations = np.unique(self._generation[present])
        if len(generations) == 1:
            self._writer(int(generations[0])).write_table(frame)
        else:
            row_generations = cp.asarray(self._generation)[frame[self._partition_column].values]
            for generation in generations:
                self._writer(int(generation)).write_table(frame[row_generations == generation])
        self._rows += counts

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

    @property
    def group_count(self) -> int:
        return len(self._writers)


class LocalPartitionedParquetWriter:
    """Write bounded centroid ranges without cuDF's full-frame partition copy."""

    _PARTITION_COLUMN = "centroid"
    WRITE_LANES = 3

    # TODO(https://github.com/NVIDIA/cudf/issues/23502): Use cuDF's partitioned
    # writer once it can bound the grouped copy and encoder buffers itself.

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        file_name_prefix: str,
        n_partitions: int,
        max_rows_per_partition: int,
        write_kwargs: dict,
        memory_budget: "WriteMemoryBudget",
    ) -> None:
        self._output_path = output_path
        self._file_name_prefix = file_name_prefix
        self._n_partitions = n_partitions
        self._max_rows_per_partition = max_rows_per_partition
        self._write_kwargs = write_kwargs
        self._memory_budget = memory_budget
        self._ranges: list[tuple[int, int]] = []
        self._writers: list[ParquetWriter | None] = []
        self._generations: list[int] = []
        self._rows = np.zeros(n_partitions, dtype=np.int64)
        self.batch_count = 0
        self.batch_rows = 0
        self.batch_bytes = 0

    def _initialize_ranges(self, counts: np.ndarray, rows_per_batch: int) -> None:
        requested = min(self._n_partitions, math.ceil(counts.sum() / rows_per_batch))
        cumulative = np.cumsum(counts)
        boundaries = [0]
        for target in np.linspace(0, counts.sum(), requested + 1, dtype=np.int64)[1:-1]:
            boundaries.append(int(np.searchsorted(cumulative, target, side="right")))
        boundaries.append(self._n_partitions)
        boundaries = list(dict.fromkeys(boundaries))
        self._ranges = list(itertools.pairwise(boundaries))
        self._writers = [None] * len(self._ranges)
        self._generations = [0] * len(self._ranges)

    def _create_writer(self, range_index: int) -> ParquetWriter:
        first, last = self._ranges[range_index]
        generation = self._generations[range_index]
        paths = []
        for partition in range(first, last):
            directory = os.path.join(self._output_path, f"{self._PARTITION_COLUMN}={partition}")
            os.makedirs(directory, exist_ok=True)
            paths.append(os.path.join(directory, f"{self._file_name_prefix}_{generation}.parquet"))
        return ParquetWriter(paths, index=False, **self._write_kwargs)

    def write_table(self, frame: "cudf.DataFrame") -> None:
        labels = frame[self._PARTITION_COLUMN].values
        counts = cp.asnumpy(cp.bincount(labels, minlength=self._n_partitions))
        frame_bytes = int(frame.memory_usage(deep=True).sum())
        with self._memory_budget.claim(len(frame), frame_bytes) as rows_per_batch:
            if not self._ranges:
                self._initialize_ranges(counts, rows_per_batch)

            order = labels.argsort()
            offsets = np.concatenate(([0], np.cumsum(counts)))
            for range_index, (first, last) in enumerate(self._ranges):
                range_counts = counts[first:last]
                if range_counts.max(initial=0) == 0:
                    continue
                if np.any(
                    (self._rows[first:last] > 0)
                    & (self._rows[first:last] + range_counts > self._max_rows_per_partition)
                ):
                    if self._writers[range_index] is not None:
                        self._writers[range_index].close()
                    self._writers[range_index] = None
                    self._generations[range_index] += 1
                    self._rows[first:last] = 0

                writer = self._writers[range_index]
                if writer is None:
                    writer = self._writers[range_index] = self._create_writer(range_index)
                range_start, range_stop = int(offsets[first]), int(offsets[last])
                for start in range(range_start, range_stop, rows_per_batch):
                    stop = min(start + rows_per_batch, range_stop)
                    batch = frame.take(order[start:stop]).drop(columns=self._PARTITION_COLUMN)
                    partition_info = [
                        (
                            max(int(offsets[partition]), start) - start,
                            max(0, min(int(offsets[partition + 1]), stop) - max(int(offsets[partition]), start)),
                        )
                        for partition in range(first, last)
                    ]
                    writer.write_table(batch, partition_info)
                    self.batch_count += 1
                    self.batch_rows = max(self.batch_rows, len(batch))
                    self.batch_bytes = max(self.batch_bytes, frame_bytes * len(batch) // len(frame))
                self._rows[first:last] += range_counts

    def close(self) -> None:
        for writer in self._writers:
            if writer is not None:
                writer.close()

    @property
    def group_count(self) -> int:
        return len(self._ranges)


class WriteMemoryBudget:
    """Coordinate transient device memory across concurrent Parquet writers."""

    # TODO(https://github.com/NVIDIA/cudf/issues/23502): Replace this upper
    # bound when cuDF exposes or internally bounds partitioned-write memory.
    _BYTES_PER_INPUT_BYTE = 5

    def __init__(self, max_lanes: int) -> None:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        # Writing should not need a larger device allocation than the data it
        # is draining. When fit already occupies most of the GPU, free memory
        # naturally becomes the tighter bound.
        self._capacity = min(free_bytes, total_bytes - free_bytes)
        self._max_lanes = max_lanes
        self._reserved = 0
        self._active = 0
        self._lock = Lock()

    @contextmanager
    def claim(self, frame_rows: int, frame_bytes: int) -> Iterator[int]:
        with self._lock:
            free_bytes = cp.cuda.runtime.memGetInfo()[0]
            remaining = min(self._capacity - self._reserved, free_bytes - self._reserved)
            lanes_left = self._max_lanes - self._active
            available = max(1, remaining // max(1, lanes_left))
            # The input size comes from the complete runtime schema, including
            # variable-width metadata. The multiplier also covers the grouped
            # table, encoder buffers, and state retained by persistent writers.
            multiplier = self._BYTES_PER_INPUT_BYTE
            rows = max(1, min(frame_rows, frame_rows * available // max(1, multiplier * frame_bytes)))
            reservation = max(1, multiplier * frame_bytes * rows // frame_rows)
            if reservation > remaining:
                msg = f"Insufficient GPU memory for one Parquet row: need {reservation} bytes, have {remaining}"
                raise MemoryError(msg)
            self._reserved += reservation
            self._active += 1
        try:
            yield rows
        finally:
            with self._lock:
                self._reserved -= reservation
                self._active -= 1


class _WriterLane:
    def __init__(self, create_writer: Callable[[], PartitionedParquetWriter], lane_index: int) -> None:
        self._create_writer = create_writer
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"parquet-writer-{lane_index}")
        self._future: Future[None] | None = None
        self._writer: PartitionedParquetWriter | None = None
        self.intervals: list[tuple[float, float]] = []

    def _write(self, frame: "cudf.DataFrame") -> None:
        started = time.perf_counter()
        try:
            if self._writer is None:
                self._writer = self._create_writer()
            self._writer.write_table(frame)
        finally:
            self.intervals.append((started, time.perf_counter()))

    def submit(self, frame: "cudf.DataFrame") -> None:
        if self._future is not None:
            msg = "Cannot submit to a busy Parquet writer lane"
            raise RuntimeError(msg)
        self._future = self._executor.submit(self._write, frame)

    def is_available(self) -> bool:
        if self._future is None:
            return True
        if not self._future.done():
            return False
        self.wait()
        return True

    def wait(self) -> None:
        if self._future is not None:
            try:
                self._future.result()
            finally:
                self._future = None

    def close(self) -> None:
        error: Exception | None = None
        try:
            self.wait()
        except Exception as exc:  # noqa: BLE001
            error = exc
        try:
            if self._writer is not None:
                self._executor.submit(self._writer.close).result()
        finally:
            self._executor.shutdown()
        if error is not None:
            raise error


class ConcurrentParquetWriters:
    """Keep a small number of persistent Parquet writers busy concurrently."""

    def __init__(
        self,
        create_writer: Callable[[int], PartitionedParquetWriter],
        max_lanes: int = 1,
    ) -> None:
        self._create_writer = create_writer
        self._max_lanes = max_lanes
        self._lanes: list[_WriterLane] = []

    def _new_lane(self) -> _WriterLane:
        lane_index = len(self._lanes)
        lane = _WriterLane(
            lambda lane_index=lane_index: self._create_writer(lane_index),
            lane_index,
        )
        self._lanes.append(lane)
        return lane

    def submit(self, frame: "cudf.DataFrame") -> None:
        if not len(frame):
            return

        for lane in self._lanes:
            if lane.is_available():
                lane.submit(frame)
                return

        if len(self._lanes) < self._max_lanes:
            self._new_lane().submit(frame)
            return

        # Reusing the oldest lane bounds the number of resident input frames.
        lane = self._lanes[0]
        lane.wait()
        lane.submit(frame)

    def flush(self) -> None:
        for lane in self._lanes:
            lane.wait()

    def close(self) -> None:
        first_error: Exception | None = None
        for lane in self._lanes:
            try:
                lane.close()
            except Exception as exc:  # noqa: BLE001
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    @property
    def write_work_time(self) -> float:
        return sum(stop - start for lane in self._lanes for start, stop in lane.intervals)

    @property
    def write_wall_time(self) -> float:
        return interval_wall_time([interval for lane in self._lanes for interval in lane.intervals])

    @property
    def lane_count(self) -> int:
        return len(self._lanes)

    @property
    def batch_rows(self) -> int:
        return max((lane._writer.batch_rows for lane in self._lanes if lane._writer is not None), default=0)

    @property
    def batch_bytes(self) -> int:
        return max((lane._writer.batch_bytes for lane in self._lanes if lane._writer is not None), default=0)

    @property
    def batch_count(self) -> int:
        return sum(lane._writer.batch_count for lane in self._lanes if lane._writer is not None)

    @property
    def group_count(self) -> int:
        return sum(lane._writer.group_count for lane in self._lanes if lane._writer is not None)
