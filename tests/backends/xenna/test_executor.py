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

from typing import Any

import pytest
from cosmos_xenna.utils.verbosity import VerbosityLevel

from nemo_curator.backends.xenna.executor import XennaExecutor
from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.stages.audio.io.audio_file_reader import AudioFileReaderStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, Task


class _PassthroughStage(ProcessingStage[AudioTask, AudioTask]):
    name = "passthrough"
    resources = Resources(cpus=1.0)
    batch_size = 2

    def process(self, task: AudioTask) -> AudioTask:
        return task


class _CentralizedStage(_PassthroughStage):
    name = "centralized"

    def __init__(self) -> None:
        self.batch_policy = BatchPolicy(
            buckets_sec=[0, 30, 1200],
            max_items_per_batch_by_bucket=[2, 1, 1],
            max_audio_sec_per_batch=None,
        )

    def build_prebucketed_tasks(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return list(tasks)

    def scheduler_task_cost(self, task: AudioTask) -> float:
        return float(task.data.get("duration", 0.0))

    def assemble_prebucketed_task_results(
        self,
        _tasks: list[AudioTask],
        processed_tasks: list[AudioTask],
    ) -> list[AudioTask]:
        return processed_tasks


class _WorkerSizedStage(_PassthroughStage):
    name = "worker_sized"

    def __init__(self, workers: int | None = None, stage_spec: dict[str, Any] | None = None) -> None:
        self._workers = workers
        self._stage_spec = stage_spec or {}

    def num_workers(self) -> int | None:
        return self._workers

    def xenna_stage_spec(self) -> dict[str, Any]:
        return dict(self._stage_spec)


def test_xenna_executor_keeps_centralized_stage_inside_one_pipeline(monkeypatch) -> None:  # noqa: ANN001
    executor = XennaExecutor()
    stages: list[ProcessingStage[Any, Any]] = [
        _PassthroughStage(),
        _CentralizedStage(),
        _PassthroughStage(),
    ]
    initial_tasks = [AudioTask(data={"duration": 5.0})]
    calls: list[tuple[list[ProcessingStage[Any, Any]], list[Task]]] = []

    def fake_run_xenna_pipeline(
        stages_arg: list[ProcessingStage[Any, Any]],
        initial_tasks_arg: list[Task],
    ) -> list[Task]:
        calls.append((stages_arg, initial_tasks_arg))
        return initial_tasks_arg

    monkeypatch.setattr(executor, "_run_xenna_pipeline", fake_run_xenna_pipeline)

    out = executor.execute(stages, initial_tasks)

    assert out == initial_tasks
    assert calls == [(stages, initial_tasks)]


def test_xenna_verbosity_none_uses_default() -> None:
    executor = XennaExecutor(config={"actor_pool_verbosity_level": None})

    assert executor._get_verbosity_config("actor_pool_verbosity_level") is VerbosityLevel.INFO


def test_xenna_verbosity_bad_string_has_helpful_error() -> None:
    executor = XennaExecutor(config={"actor_pool_verbosity_level": "loud"})

    with pytest.raises(ValueError, match="Invalid Xenna verbosity config actor_pool_verbosity_level='loud'"):
        executor._get_verbosity_config("actor_pool_verbosity_level")


def test_xenna_stage_spec_falls_back_to_stage_num_workers() -> None:
    stage_spec = XennaExecutor()._build_stage_spec(_WorkerSizedStage(workers=3))

    assert stage_spec.num_workers == 3
    assert stage_spec.num_workers_per_node is None


def test_real_audio_stages_use_main_worker_sizing_contract(tmp_path) -> None:  # noqa: ANN001
    executor = XennaExecutor()

    asr_spec = executor._build_stage_spec(
        ASRStage(
            adapter_target="tests.fake.Adapter",
            model_id="fake-model",
            xenna_num_workers=2,
        )
    )
    reader_spec = executor._build_stage_spec(AudioFileReaderStage(xenna_num_workers=3))
    writer_spec = executor._build_stage_spec(ManifestWriterStage(output_path=str(tmp_path / "out.jsonl")))

    assert asr_spec.num_workers == 2
    assert asr_spec.num_workers_per_node is None
    assert reader_spec.num_workers == 3
    assert reader_spec.num_workers_per_node is None
    assert writer_spec.num_workers == 1
    assert writer_spec.num_workers_per_node is None


def test_xenna_stage_spec_num_workers_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"Use num_workers\(\) instead"):
        XennaExecutor()._build_stage_spec(_WorkerSizedStage(stage_spec={"num_workers": 4}))


def test_xenna_num_workers_per_node_is_rejected_with_stage_num_workers() -> None:
    with pytest.raises(ValueError, match=r"num_workers\(\).*num_workers_per_node"):
        XennaExecutor()._build_stage_spec(_WorkerSizedStage(workers=3, stage_spec={"num_workers_per_node": 2}))


def test_xenna_num_workers_per_node_is_rejected_with_legacy_num_workers() -> None:
    stage = _WorkerSizedStage(stage_spec={"num_workers": 4, "num_workers_per_node": 2})

    with pytest.raises(ValueError, match=r"Use num_workers\(\) instead"):
        XennaExecutor()._build_stage_spec(stage)


def test_xenna_rejects_conflicting_cluster_worker_counts() -> None:
    stage = _WorkerSizedStage(workers=3, stage_spec={"num_workers": 4})

    with pytest.raises(ValueError, match=r"Use num_workers\(\) instead"):
        XennaExecutor()._build_stage_spec(stage)
