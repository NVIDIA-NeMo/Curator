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

from threading import Event, Lock

import pytest


class _RecordingWriter:
    def __init__(self) -> None:
        self.frames = []
        self.closed = False

    def write_table(self, frame: object) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


@pytest.mark.gpu
def test_rolling_writer_uses_native_incremental_writer_without_rollover() -> None:
    import cudf

    from nemo_curator.stages.deduplication.semantic.write_utils import RollingParquetWriter

    created = []

    def create_writer(generation: int) -> _RecordingWriter:
        writer = _RecordingWriter()
        created.append((generation, writer))
        return writer

    writer = RollingParquetWriter(
        create_writer,
        n_partitions=2,
        max_rows_per_partition=10,
        partition_column="centroid",
    )
    frame = cudf.DataFrame({"centroid": [0, 1], "value": [1, 2]})

    writer.write_table(frame)
    writer.close()

    assert len(created) == 1
    assert created[0][0] == 0
    assert created[0][1].frames == [frame]
    assert created[0][1].closed


@pytest.mark.gpu
def test_rolling_writer_limits_rows_per_partition() -> None:
    import cudf

    from nemo_curator.stages.deduplication.semantic.write_utils import RollingParquetWriter

    created = []

    def create_writer(generation: int) -> _RecordingWriter:
        writer = _RecordingWriter()
        created.append((generation, writer))
        return writer

    writer = RollingParquetWriter(
        create_writer,
        n_partitions=2,
        max_rows_per_partition=2,
        partition_column="centroid",
    )
    writer.write_table(cudf.DataFrame({"centroid": [0, 0, 1], "value": [1, 2, 3]}))
    writer.write_table(cudf.DataFrame({"centroid": [0, 1], "value": [4, 5]}))
    writer.close()

    assert [generation for generation, _ in created] == [0, 1]
    for _, created_writer in created:
        partition_counts = cudf.concat(created_writer.frames).groupby("centroid").size()
        assert partition_counts.max() <= 2


@pytest.mark.gpu
def test_concurrent_writer_hands_frame_to_background_thread() -> None:
    import cudf

    from nemo_curator.stages.deduplication.semantic.write_utils import ConcurrentParquetWriters

    lock = Lock()
    release = Event()
    writers = []

    def create_writer(lane_index: int) -> _RecordingWriter:
        class BlockingWriter(_RecordingWriter):
            def write_table(self, frame: object) -> None:
                release.wait(timeout=5)
                super().write_table(frame)

        writer = BlockingWriter()
        with lock:
            writers.append((lane_index, writer))
        return writer

    concurrent = ConcurrentParquetWriters(create_writer)
    concurrent.submit(cudf.DataFrame({"value": [0, 1, 2, 3]}))
    release.set()
    concurrent.close()

    assert [lane for lane, _ in writers] == [0]
    assert 0 < concurrent.write_wall_time <= concurrent.write_work_time
    values = [frame["value"].iloc[0] for _, writer in writers for frame in writer.frames]
    assert values == [0]
    assert all(writer.closed for _, writer in writers)


@pytest.mark.gpu
def test_concurrent_writer_preserves_variable_width_metadata() -> None:
    import cudf

    from nemo_curator.stages.deduplication.semantic.write_utils import ConcurrentParquetWriters

    writers = []

    def create_writer(_: int) -> _RecordingWriter:
        writer = _RecordingWriter()
        writers.append(writer)
        return writer

    frame = cudf.DataFrame({"value": range(8), "metadata": ["x" * size for size in range(8)]})
    concurrent = ConcurrentParquetWriters(create_writer)
    concurrent.submit(frame)
    concurrent.close()

    written = cudf.concat([part for writer in writers for part in writer.frames]).sort_values("value")
    assert written.to_pandas().reset_index(drop=True).equals(frame.to_pandas())
