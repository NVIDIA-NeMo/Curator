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

from pathlib import Path
from threading import Barrier, Event, Lock

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
    # The first call is itself too large for centroid 0; the second call also
    # checks that each centroid continues in the right generation afterward.
    writer.write_table(cudf.DataFrame({"centroid": [0, 0, 0, 0, 0, 1], "value": range(6)}))
    writer.write_table(cudf.DataFrame({"centroid": [0, 1], "value": [6, 7]}))
    writer.close()

    assert [generation for generation, _ in created] == [0, 1, 2]
    for _, created_writer in created:
        partition_counts = cudf.concat(created_writer.frames).groupby("centroid").size()
        assert partition_counts.max() <= 2


@pytest.mark.gpu
def test_concurrent_writer_uses_persistent_lanes_for_overlapping_writes() -> None:
    import cudf

    from nemo_curator.stages.deduplication.semantic.write_utils import ConcurrentParquetWriters

    lock = Lock()
    release = Event()
    started = Barrier(3)
    writers = []

    def create_writer(lane_index: int) -> _RecordingWriter:
        class BlockingWriter(_RecordingWriter):
            def write_table(self, frame: object) -> None:
                started.wait(timeout=5)
                release.wait(timeout=5)
                super().write_table(frame)

        writer = BlockingWriter()
        with lock:
            writers.append((lane_index, writer))
        return writer

    concurrent = ConcurrentParquetWriters(create_writer, max_lanes=2)
    frame = cudf.DataFrame({"value": [0, 1, 2, 3]})
    concurrent.submit(frame)
    concurrent.submit(frame + 4)
    started.wait(timeout=5)

    # Neither write can finish yet, so reaching two writers proves the second
    # submission did not quietly serialize behind the first one.
    assert [lane for lane, _ in writers] == [0, 1]
    release.set()
    concurrent.close()

    assert 0 < concurrent.write_wall_time <= concurrent.write_work_time
    values = [frame["value"].iloc[0] for _, writer in writers for frame in writer.frames]
    assert values == [0, 4]
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


@pytest.mark.gpu
def test_local_partitioned_writer_preserves_rows_across_bounded_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cudf

    from nemo_curator.stages.deduplication.semantic.write_utils import LocalPartitionedParquetWriter

    # Keep the batches deliberately tiny here. This exercises the same persistent
    # multi-sink Parquet writer used by KMeans while making both centroid ranges
    # cross a write boundary in a small test.
    monkeypatch.setattr(LocalPartitionedParquetWriter, "_rows_per_batch", lambda *_: 2)
    writer = LocalPartitionedParquetWriter(
        output_path=str(tmp_path),
        file_name_prefix="part",
        n_partitions=2,
        max_rows_per_partition=3,
        write_kwargs={},
    )
    first = cudf.DataFrame(
        {
            "centroid": [1, 0, 1, 0, 1, 0],
            "value": [0, 1, 2, 3, 4, 5],
            "metadata": ["", "x", "yy", "zzz", "w" * 4, "v" * 5],
        }
    )
    second = first.copy()
    second["value"] += len(first)
    expected = cudf.concat([first, second], ignore_index=True)

    writer.write_table(first)
    writer.write_table(second)
    writer.close()

    actual = cudf.read_parquet(str(tmp_path)).sort_values("value").reset_index(drop=True)
    actual["centroid"] = actual["centroid"].astype(expected["centroid"].dtype)
    assert actual.to_pandas().equals(expected[actual.columns].to_pandas())
    assert writer.batch_count > 1
    assert writer.group_count == 2
