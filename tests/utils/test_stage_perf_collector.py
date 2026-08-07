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

import gc
import subprocess
import sys
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import ray

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task
from nemo_curator.utils.performance_utils import StagePerfStats
from nemo_curator.utils.stage_perf_collector import (
    COLLECTOR_ACTOR_ATTR,
    COLLECTOR_NAME_ATTR,
    COLLECTOR_PENDING_ATTR,
    COLLECTOR_REQUIRED_ATTR,
    PerformanceRecordStore,
    _StagePerfCollector,
    _StagePerfCollectorHandle,
    _StagePerfSpool,
    record_stage_perf,
    start_stage_perf_collector,
    stop_stage_perf_collector,
)


class _Stage(ProcessingStage):
    name = "stage"

    def process(self, task: Task) -> Task:
        return task


class _ReportStage(_Stage):
    def requests_performance_records(self) -> bool:
        return True


def test_record_stage_perf_is_noop_without_collector() -> None:
    assert (
        record_stage_perf(
            _Stage(),
            StagePerfStats(stage_name="stage"),
        )
        is False
    )


def test_required_record_stage_perf_acknowledges_every_publication() -> None:
    stage = _Stage()
    collector = MagicMock()
    record_refs = [object(), object(), object()]
    collector.record.remote.side_effect = record_refs
    setattr(stage, COLLECTOR_ACTOR_ATTR, collector)
    setattr(stage, COLLECTOR_PENDING_ATTR, [])
    setattr(stage, COLLECTOR_REQUIRED_ATTR, True)

    with patch("nemo_curator.utils.stage_perf_collector.ray.get") as ray_get:
        for _ in record_refs:
            assert record_stage_perf(stage, StagePerfStats(stage_name="stage"))

    assert [call.args[0] for call in ray_get.call_args_list] == record_refs
    assert getattr(stage, COLLECTOR_PENDING_ATTR) == []


def test_required_publish_failure_is_not_silenced() -> None:
    stage = _Stage()
    collector = MagicMock()
    record_ref = object()
    collector.record.remote.return_value = record_ref
    setattr(stage, COLLECTOR_ACTOR_ATTR, collector)
    setattr(stage, COLLECTOR_PENDING_ATTR, [])
    setattr(stage, COLLECTOR_REQUIRED_ATTR, True)

    with (
        patch("nemo_curator.utils.stage_perf_collector.ray.get", side_effect=OSError("publish failed")) as ray_get,
        pytest.raises(RuntimeError, match="Required stage performance publication failed"),
    ):
        record_stage_perf(stage, StagePerfStats(stage_name="stage"))

    ray_get.assert_called_once_with(record_ref)


def test_required_finish_failure_is_not_silenced(tmp_path: Path) -> None:
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    spool_path = spool_dir / "records.jsonl"
    spool_path.write_text("", encoding="utf-8")
    actor = MagicMock()
    finish_ref = object()
    actor.finish.remote.return_value = finish_ref
    handle = _StagePerfCollectorHandle(actor=actor, spool_path=str(spool_path), report_required=True)

    with (
        patch("nemo_curator.utils.stage_perf_collector.ray.get", side_effect=OSError("finish failed")) as ray_get,
        patch("nemo_curator.utils.stage_perf_collector.ray.kill"),
        pytest.raises(RuntimeError, match="Required stage performance collector finish failed"),
    ):
        stop_stage_perf_collector(handle, [_Stage()], raise_on_failure=True)

    ray_get.assert_called_once_with(finish_ref)
    assert not spool_path.exists()


def test_required_dropped_record_failure_is_not_silenced(tmp_path: Path) -> None:
    spool_path = tmp_path / "records.jsonl"
    spool_path.write_text("", encoding="utf-8")
    actor = MagicMock()
    finish_ref = object()
    actor.finish.remote.return_value = finish_ref
    handle = _StagePerfCollectorHandle(actor=actor, spool_path=str(spool_path), report_required=True)

    with (
        patch(
            "nemo_curator.utils.stage_perf_collector.ray.get",
            return_value=(str(spool_path), 0, 1, ["OSError: spool failed"]),
        ) as ray_get,
        patch("nemo_curator.utils.stage_perf_collector.ray.kill"),
        pytest.raises(RuntimeError, match="dropped 1 record"),
    ):
        stop_stage_perf_collector(handle, [_Stage()], raise_on_failure=True)

    ray_get.assert_called_once_with(finish_ref)
    assert not spool_path.exists()


def test_collector_start_failure_cleans_created_spool(tmp_path: Path) -> None:
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    spool_path = spool_dir / "records.jsonl"
    stage = _ReportStage()
    runtime_context = MagicMock()
    runtime_context.get_node_id.return_value = "a" * 56
    actor = MagicMock()
    actor_builder = MagicMock()
    actor_builder.remote.return_value = actor

    with (
        patch("nemo_curator.utils.stage_perf_collector._new_spool_path", return_value=str(spool_path)),
        patch("nemo_curator.utils.stage_perf_collector.ray.get_runtime_context", return_value=runtime_context),
        patch.object(_StagePerfCollector, "options", return_value=actor_builder),
        patch("nemo_curator.utils.stage_perf_collector.ray.get", side_effect=OSError("ready failed")),
        patch("nemo_curator.utils.stage_perf_collector.ray.kill") as ray_kill,
        pytest.raises(OSError, match="ready failed"),
    ):
        start_stage_perf_collector([stage])

    ray_kill.assert_called_once_with(actor, no_restart=True)
    assert not spool_path.exists()
    assert not spool_dir.exists()


def test_stop_collector_clears_stage_routing_and_kills_actor(tmp_path: Path) -> None:
    spool_path = tmp_path / "records.jsonl"
    spool_path.write_text("", encoding="utf-8")
    actor = MagicMock()
    finish_ref = object()
    actor.finish.remote.return_value = finish_ref
    handle = _StagePerfCollectorHandle(actor=actor, spool_path=str(spool_path), report_required=True)
    stage = _Stage()
    for attr_name, value in (
        (COLLECTOR_NAME_ATTR, "collector"),
        (COLLECTOR_ACTOR_ATTR, actor),
        (COLLECTOR_PENDING_ATTR, []),
        (COLLECTOR_REQUIRED_ATTR, True),
    ):
        setattr(stage, attr_name, value)

    with (
        patch(
            "nemo_curator.utils.stage_perf_collector.ray.get",
            return_value=(str(spool_path), 0, 0, []),
        ),
        patch("nemo_curator.utils.stage_perf_collector.ray.kill") as ray_kill,
    ):
        record_store = stop_stage_perf_collector(handle, [stage], raise_on_failure=True)

    assert all(
        not hasattr(stage, attr_name)
        for attr_name in (
            COLLECTOR_NAME_ATTR,
            COLLECTOR_ACTOR_ATTR,
            COLLECTOR_PENDING_ATTR,
            COLLECTOR_REQUIRED_ATTR,
        )
    )
    ray_kill.assert_called_once_with(actor, no_restart=True)
    record_store.cleanup()


@pytest.mark.usefixtures("shared_ray_client")
def test_collector_returns_disk_backed_record_store() -> None:
    stage = _ReportStage()
    collector = start_stage_perf_collector([stage])
    assert collector is not None

    assert record_stage_perf(
        stage,
        StagePerfStats(stage_name="stage", invocation_id="invocation-1"),
    )
    record_store = stop_stage_perf_collector(collector, [stage])

    assert len(record_store) == 1
    assert record_store.path
    assert [record.invocation_id for record in record_store] == ["invocation-1"]
    record_store.cleanup()


@pytest.mark.usefixtures("shared_ray_client")
def test_required_publications_from_multiple_submitters_are_complete() -> None:
    stages = [_ReportStage() for _ in range(4)]
    collector = start_stage_perf_collector(stages)

    @ray.remote
    def publish_records(stage: _ReportStage, producer_index: int) -> None:
        for record_index in range(7):
            record_stage_perf(
                stage,
                StagePerfStats(
                    stage_name="stage",
                    invocation_id=f"producer-{producer_index}-record-{record_index}",
                ),
            )

    ray.get([publish_records.remote(stage, producer_index) for producer_index, stage in enumerate(stages)])
    record_store = stop_stage_perf_collector(collector, stages, raise_on_failure=True)

    expected_ids = {
        f"producer-{producer_index}-record-{record_index}" for producer_index in range(4) for record_index in range(7)
    }
    assert len(record_store) == len(expected_ids)
    assert {record.invocation_id for record in record_store} == expected_ids
    record_store.cleanup()


def test_record_store_cleans_spool_when_last_owner_is_released() -> None:
    record_store = PerformanceRecordStore.from_records([StagePerfStats(stage_name="stage")])
    spool_path = Path(record_store.path)

    assert spool_path.is_file()
    del record_store
    gc.collect()

    assert not spool_path.exists()
    assert not spool_path.parent.exists()


def test_record_store_context_manager_preserves_iteration_until_close() -> None:
    with PerformanceRecordStore.from_records(
        [StagePerfStats(stage_name="stage", invocation_id="invocation-1")]
    ) as record_store:
        spool_path = Path(record_store.path)
        assert [record.invocation_id for record in record_store] == ["invocation-1"]
        assert [record.invocation_id for record in record_store] == ["invocation-1"]

    assert not spool_path.exists()
    assert not spool_path.parent.exists()
    assert record_store.path == ""
    assert len(record_store) == 0


def test_record_store_cleans_spool_at_normal_process_exit(tmp_path: Path) -> None:
    path_record = tmp_path / "spool-path.txt"
    script = """
from pathlib import Path
import sys

from nemo_curator.utils.performance_utils import StagePerfStats
from nemo_curator.utils.stage_perf_collector import PerformanceRecordStore

store = PerformanceRecordStore.from_records([StagePerfStats(stage_name="stage")])
Path(sys.argv[1]).write_text(store.path, encoding="utf-8")
"""

    subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(path_record)],
        check=True,
    )
    spool_path = Path(path_record.read_text(encoding="utf-8"))

    assert not spool_path.exists()
    assert not spool_path.parent.exists()


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
