# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import Callable
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from ray.data import ActorPoolStrategy

from nemo_curator.backends.ray_data.adapter import RayDataStageAdapter
from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.backends.ray_data.utils import (
    coerce_batch_tasks,
    get_actor_compute_strategy_for_stage,
)
from nemo_curator.backends.utils import RayStageSpecKeys, get_available_cpu_gpu_resources
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask
from tests.backends.test_utils import reset_head_node_cache  # noqa: F401


class _PreplannedEchoStage(ProcessingStage[AudioTask, AudioTask]):
    name = "preplanned_echo"
    resources = Resources(cpus=1.0)
    batch_size = 99

    def process(self, task: AudioTask) -> AudioTask:
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            task.data["seen_batch_size"] = len(tasks)
        return tasks


class _CentralizedPlanningStage(ProcessingStage[AudioTask, AudioTask]):
    name = "centralized_planning"
    resources = Resources(cpus=1.0)
    batch_size = 2

    def __init__(self) -> None:
        self.batch_policy = BatchPolicy(
            buckets_sec=[0, 30, 1200],
            max_items_per_batch_by_bucket=[2, 1, 1],
            max_audio_sec_per_batch=None,
        )

    def process(self, task: AudioTask) -> AudioTask:
        return task

    def build_prebucketed_tasks(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return list(tasks)

    def scheduler_task_cost(self, task: AudioTask) -> float:
        return float(task.data["duration"])

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            task.data["processed_batch_size"] = len(tasks)
        return tasks

    def assemble_prebucketed_task_results(
        self,
        tasks: list[AudioTask],
        _processed_tasks: list[AudioTask],
    ) -> list[AudioTask]:
        return list(tasks)


class _FakeDataset:
    def __init__(self, sample_items: list[object] | None = None) -> None:
        self.repartition_calls: list[tuple[tuple, dict]] = []
        self.map_batches_calls: list[tuple[object, dict]] = []
        self.sample_output: dict | None = None
        self.sample_items = sample_items

    def repartition(self, *args, **kwargs) -> "_FakeDataset":
        self.repartition_calls.append((args, kwargs))
        return self

    def map_batches(self, fn: Callable[[dict[str, object]], dict[str, object]], **kwargs) -> "_FakeDataset":
        self.map_batches_calls.append((fn, kwargs))
        sample_items = self.sample_items
        if sample_items is None:
            first = AudioTask(data={"duration": 5.0})
            second = AudioTask(data={"duration": 600.0})
            sample_items = [first, second]
        self.sample_output = fn({"item": sample_items})
        return self


class TestGetAvailableCpuGpuResources:
    # TODO: Move this to tests/backends/test_utils.py
    """Test class for utility functions in ray_data backend."""

    def test_get_available_cpu_gpu_resources_conftest(self, shared_ray_client: None):
        """Test get_available_cpu_gpu_resources function."""
        cpus, gpus = get_available_cpu_gpu_resources()
        assert cpus == 11
        # GPU count depends on local hardware and whether GPU tests are selected.
        assert gpus in [0.0, 1.0, 2.0]

    @pytest.mark.usefixtures("reset_head_node_cache")
    def test_get_resources_with_ignore_head_node(
        self,
        shared_ray_client: None,
    ):
        """ignore_head_node=True skips the head node; running on the head node, resources are 0."""
        cpus_without_head, gpus_without_head = get_available_cpu_gpu_resources(ignore_head_node=True)
        assert cpus_without_head == 0
        assert gpus_without_head == 0

    @patch("ray.available_resources", return_value={"CPU": 4.0, "node:10.0.0.1": 1.0, "memory": 1000000000})
    def test_get_available_cpu_gpu_resources_mock_no_gpus(self, mock_available_resources: MagicMock):
        """Test get_available_cpu_gpu_resources when no GPUs available."""
        assert get_available_cpu_gpu_resources() == (4.0, 0)
        mock_available_resources.assert_called_once()

    @patch("ray.available_resources", return_value={"node:10.0.0.1": 1.0, "memory": 1000000000})
    def test_get_available_cpu_gpu_resources_mock_no_resources(self, mock_available_resources: MagicMock):
        assert get_available_cpu_gpu_resources() == (0, 0)
        mock_available_resources.assert_called_once()


class TestGetActorComputeStrategyForStage:
    """Test class for Ray Data compute strategy construction."""

    @pytest.mark.parametrize(
        ("num_workers", "ray_stage_spec", "expected", "expected_warning"),
        [
            (4, {}, ActorPoolStrategy(size=4), None),
            (0, {}, ActorPoolStrategy(min_size=1, max_size=None), None),
            (-1, {}, ActorPoolStrategy(min_size=1, max_size=None), None),
            (None, {}, ActorPoolStrategy(min_size=1, max_size=None), None),
            (
                None,
                {
                    RayStageSpecKeys.MIN_WORKERS: 2,
                    RayStageSpecKeys.MAX_WORKERS: 8,
                    RayStageSpecKeys.INITIAL_WORKERS: 4,
                },
                ActorPoolStrategy(min_size=2, max_size=8, initial_size=4),
                None,
            ),
            (
                3,
                {
                    RayStageSpecKeys.MIN_WORKERS: 1,
                    RayStageSpecKeys.MAX_WORKERS: 8,
                    RayStageSpecKeys.INITIAL_WORKERS: 2,
                },
                ActorPoolStrategy(size=3),
                "uses num_workers=3",
            ),
        ],
    )
    def test_actor_compute_strategy(
        self,
        num_workers: int | None,
        ray_stage_spec: dict[str, object],
        expected: ActorPoolStrategy,
        expected_warning: str | None,
    ) -> None:
        mock_stage = Mock(num_workers=lambda: num_workers, ray_stage_spec=lambda: ray_stage_spec)
        mock_stage.name = "stage"

        with patch("nemo_curator.backends.ray_data.utils.logger.warning") as mock_warning:
            assert get_actor_compute_strategy_for_stage(mock_stage) == expected

        if expected_warning is None:
            mock_warning.assert_not_called()
        else:
            mock_warning.assert_called_once()
            assert expected_warning in mock_warning.call_args.args[0]

    def test_actor_compute_strategy_rejects_invalid_sizing(self) -> None:
        mock_stage = Mock(
            num_workers=lambda: None,
            ray_stage_spec=lambda: {
                RayStageSpecKeys.MIN_WORKERS: 1,
                RayStageSpecKeys.MAX_WORKERS: 4,
                RayStageSpecKeys.INITIAL_WORKERS: 10,
            },
        )
        mock_stage.name = "stage"

        with pytest.raises(ValueError, match="Invalid Ray Data actor pool sizing for stage stage"):
            get_actor_compute_strategy_for_stage(mock_stage)


class TestCoerceBatchTasks:
    def test_coerce_batch_tasks_from_numpy_object_array(self) -> None:
        sentinel = object()
        batch = np.array([sentinel], dtype=object)
        assert coerce_batch_tasks(batch) == [sentinel]

    def test_coerce_batch_tasks_empty(self) -> None:
        assert coerce_batch_tasks([]) == []
        assert coerce_batch_tasks(np.array([], dtype=object)) == []
        assert coerce_batch_tasks(None) == []


def test_ray_data_adapter_passes_backend_batch_to_stage_process_batch() -> None:
    dataset = _FakeDataset()
    adapter = RayDataStageAdapter(_CentralizedPlanningStage())

    out = adapter.process_dataset(dataset)

    assert out is dataset
    assert dataset.repartition_calls == []
    assert len(dataset.map_batches_calls) == 1
    assert dataset.map_batches_calls[0][1]["batch_size"] == 2
    assert dataset.sample_output is not None
    processed_durations = [task.data["duration"] for task in dataset.sample_output["item"]]
    processed_batch_sizes = [task.data["processed_batch_size"] for task in dataset.sample_output["item"]]
    assert processed_durations == [5.0, 600.0]
    assert processed_batch_sizes == [2, 2]


def test_ray_data_executor_keeps_centralized_stage_in_ray_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = RayDataExecutor(ignore_head_node=True)
    stage = _CentralizedPlanningStage()
    input_dataset = object()
    output_dataset = object()
    calls: dict[str, object] = {}

    class FakeRayDataStageAdapter:
        def __init__(self, stage_arg: ProcessingStage) -> None:
            calls["stage"] = stage_arg

        def process_dataset(self, dataset_arg: object) -> object:
            calls["dataset"] = dataset_arg
            return output_dataset

    monkeypatch.setattr(
        executor,
        "_dataset_to_tasks",
        Mock(side_effect=AssertionError("centralized stages should not materialize Ray Data datasets")),
    )
    monkeypatch.setattr(
        executor,
        "_tasks_to_dataset",
        Mock(side_effect=AssertionError("centralized stages should not rebuild Ray Data datasets")),
    )
    monkeypatch.setattr("nemo_curator.backends.ray_data.executor.RayDataStageAdapter", FakeRayDataStageAdapter)

    out = executor._process_stage_dataset(stage, input_dataset)

    assert out is output_dataset
    assert calls == {"stage": stage, "dataset": input_dataset}


def test_ray_data_executor_keeps_noncentral_stage_in_ray_data(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = RayDataExecutor(ignore_head_node=True)
    stage = _PreplannedEchoStage()
    input_dataset = object()
    output_dataset = object()
    calls: dict[str, object] = {}

    class FakeRayDataStageAdapter:
        def __init__(self, stage_arg: ProcessingStage) -> None:
            calls["stage"] = stage_arg

        def process_dataset(self, dataset_arg: object) -> object:
            calls["dataset"] = dataset_arg
            return output_dataset

    monkeypatch.setattr("nemo_curator.backends.ray_data.executor.RayDataStageAdapter", FakeRayDataStageAdapter)

    out = executor._process_stage_dataset(stage, input_dataset)

    assert out is output_dataset
    assert calls == {"stage": stage, "dataset": input_dataset}
