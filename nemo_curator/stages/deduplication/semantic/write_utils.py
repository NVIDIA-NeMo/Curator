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

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cudf.io.parquet import ParquetDatasetWriter

if TYPE_CHECKING:
    import cudf


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


class _WriterLane:
    def __init__(self, create_writer: Callable[[], RollingParquetWriter], lane_index: int) -> None:
        self._create_writer = create_writer
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"parquet-writer-{lane_index}")
        self._future: Future[None] | None = None
        self._writer: RollingParquetWriter | None = None
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
    """Overlap one whole-frame Parquet write with preparation of the next frame."""

    def __init__(
        self,
        create_writer: Callable[[int], RollingParquetWriter],
    ) -> None:
        self._create_writer = create_writer
        self._lanes: list[_WriterLane] = []
        self._batch_rows = 0
        self._batch_bytes = 0

    @staticmethod
    def _frame_bytes(frame: "cudf.DataFrame") -> int:
        return int(frame.memory_usage(deep=True).sum())

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
        self.flush()
        self._batch_rows = max(self._batch_rows, len(frame))
        self._batch_bytes = max(self._batch_bytes, self._frame_bytes(frame))
        (self._lanes[0] if self._lanes else self._new_lane()).submit(frame)

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
        return self._batch_rows

    @property
    def batch_bytes(self) -> int:
        return self._batch_bytes
