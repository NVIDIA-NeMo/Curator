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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import ray

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task
from nemo_curator.utils.stage_perf_collector import (
    COLLECTOR_ACTOR_ATTR,
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


def _perf_record(invocation_id: str = "") -> dict[str, object]:
    return {
        "stage_name": "stage",
        "invocation_id": invocation_id,
        "process_time": 0.0,
        "actor_idle_time": 0.0,
        "num_items_processed": 0,
        "custom_metrics": {},
    }


def test_publication_waits_for_its_actor_acknowledgement() -> None:
    stage = _Stage()
    collector = MagicMock()
    record_ref = object()
    collector.record.remote.return_value = record_ref
    setattr(stage, COLLECTOR_ACTOR_ATTR, collector)

    with patch("nemo_curator.utils.stage_perf_collector.ray.get") as ray_get:
        record_stage_perf(stage, _perf_record())

    ray_get.assert_called_once_with(record_ref)


def test_failed_publication_poisons_the_collector_and_raises() -> None:
    stage = _Stage()
    collector = MagicMock()
    record_ref, fail_ref = object(), object()
    collector.record.remote.return_value = record_ref
    collector.fail.remote.return_value = fail_ref
    setattr(stage, COLLECTOR_ACTOR_ATTR, collector)

    with (
        patch(
            "nemo_curator.utils.stage_perf_collector.ray.get",
            side_effect=[OSError("acknowledgement lost"), RuntimeError("poisoned")],
        ) as ray_get,
        pytest.raises(RuntimeError, match="Required stage performance publication failed"),
    ):
        record_stage_perf(stage, _perf_record())

    assert [call.args[0] for call in ray_get.call_args_list] == [record_ref, fail_ref]
    collector.fail.remote.assert_called_once_with("OSError: acknowledgement lost")


def test_start_failure_cleans_the_spool(tmp_path: Path) -> None:
    spool_path = tmp_path / "spool" / "records.jsonl"
    stage = _Stage()
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
    assert not spool_path.parent.exists()


def test_finish_failure_cleans_the_spool(tmp_path: Path) -> None:
    spool_path = tmp_path / "spool" / "records.jsonl"
    spool_path.parent.mkdir()
    spool_path.write_text("", encoding="utf-8")
    actor = MagicMock()
    finish_ref = object()
    actor.finish.remote.return_value = finish_ref
    handle = _StagePerfCollectorHandle(actor=actor, spool_path=str(spool_path))

    with (
        patch("nemo_curator.utils.stage_perf_collector.ray.get", side_effect=OSError("finish failed")),
        patch("nemo_curator.utils.stage_perf_collector.ray.kill"),
        pytest.raises(OSError, match="finish failed"),
    ):
        stop_stage_perf_collector(handle, [_Stage()])

    assert not spool_path.parent.exists()


@pytest.mark.usefixtures("shared_ray_client")
def test_publications_from_multiple_submitters_are_complete() -> None:
    stages = [_Stage() for _ in range(4)]
    collector = start_stage_perf_collector(stages)

    @ray.remote
    def publish_records(stage: _Stage, producer_index: int) -> None:
        for record_index in range(7):
            record_stage_perf(stage, _perf_record(f"{producer_index}-{record_index}"))

    ray.get([publish_records.remote(stage, index) for index, stage in enumerate(stages)])
    record_store = stop_stage_perf_collector(collector, stages)
    expected_ids = {f"{producer}-{record}" for producer in range(4) for record in range(7)}

    assert {record["invocation_id"] for record in record_store} == expected_ids
    assert all(not hasattr(stage, COLLECTOR_ACTOR_ATTR) for stage in stages)
    spool_path = Path(record_store.path)
    record_store.cleanup()
    assert not spool_path.parent.exists()


@pytest.mark.usefixtures("shared_ray_client")
def test_actor_serialization_failure_is_terminal() -> None:
    stage = _Stage()
    collector = start_stage_perf_collector([stage])

    with pytest.raises(RuntimeError, match="Required stage performance publication failed"):
        record_stage_perf(stage, {"not-json-serializable": object()})
    with pytest.raises(ray.exceptions.RayTaskError, match="Stage performance collector is poisoned"):
        stop_stage_perf_collector(collector, [stage])

    assert not Path(collector.spool_path).exists()


def test_explicit_failure_prevents_a_partial_store(tmp_path: Path) -> None:
    spool_path = tmp_path / "records.jsonl"
    spool = _StagePerfSpool(str(spool_path))
    spool.record({"invocation_id": "written"})

    with pytest.raises(RuntimeError, match="poisoned: acknowledgement lost"):
        spool.fail("acknowledgement lost")
    with pytest.raises(RuntimeError, match="poisoned: acknowledgement lost"):
        spool.finish()
    spool_path.unlink()
