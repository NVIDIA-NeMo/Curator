# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task
from nemo_curator.utils.performance_utils import StagePerfStats
from nemo_curator.utils.stage_perf_collector import (
    _StagePerfSpool,
    record_stage_perf,
    start_stage_perf_collector,
    stop_stage_perf_collector,
)


class _Stage(ProcessingStage):
    name = "stage"

    def process(self, task: Task) -> Task:
        return task


def test_record_stage_perf_is_noop_without_collector() -> None:
    assert (
        record_stage_perf(
            _Stage(),
            StagePerfStats(stage_name="stage"),
            attached_to_output=False,
        )
        is False
    )


def test_record_stage_perf_waits_for_collector_ack() -> None:
    stage = _Stage()
    stage._curator_stage_perf_collector_name = "collector"
    collector = MagicMock()
    record_ref = collector.record.remote.return_value

    with (
        patch("nemo_curator.utils.stage_perf_collector.ray.get_actor", return_value=collector) as get_actor,
        patch("nemo_curator.utils.stage_perf_collector.ray.get") as ray_get,
    ):
        assert (
            record_stage_perf(
                stage,
                StagePerfStats(stage_name="stage"),
                attached_to_output=True,
            )
            is True
        )

    get_actor.assert_called_once_with("collector")
    collector.record.remote.assert_called_once()
    ray_get.assert_called_once_with(record_ref)


@pytest.mark.usefixtures("shared_ray_client")
def test_collector_returns_disk_backed_record_store() -> None:
    stage = _Stage()
    stage.extended_performance_metrics = True
    collector = start_stage_perf_collector([stage])
    assert collector is not None

    assert record_stage_perf(
        stage,
        StagePerfStats(stage_name="stage", invocation_id="invocation-1"),
        attached_to_output=False,
    )
    record_store = stop_stage_perf_collector(collector, [stage])

    assert len(record_store) == 1
    assert record_store.path
    assert [record.invocation_id for record in record_store] == ["invocation-1"]
    record_store.cleanup()


def test_high_cardinality_spool_has_bounded_memory(tmp_path: Path) -> None:
    spool = _StagePerfSpool(str(tmp_path / "records.jsonl"))
    record = StagePerfStats(
        stage_name="ASR",
        invocation_id="invocation",
        process_time=1.0,
        custom_metrics={"audio_duration_s": 12.0},
    )

    tracemalloc.start()
    try:
        for _ in range(1_000):
            spool.record(record)
        baseline_bytes, _ = tracemalloc.get_traced_memory()
        for _ in range(49_000):
            spool.record(record)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    path, record_count = spool.finish()

    assert record_count == 50_000
    assert current_bytes - baseline_bytes < 512 * 1024
    assert peak_bytes < 2 * 1024 * 1024
    with Path(path).open(encoding="utf-8") as records_file:
        assert sum(1 for _ in records_file) == record_count
