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

from nemo_curator.backends.base import SchedulerReadyTaskBatch
from nemo_curator.backends.ray_data.adapter import RayDataStageAdapter, create_task_from_stage
from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.backends.ray_data.utils import (
    calculate_concurrency_for_actors_for_stage,
    coerce_batch_tasks,
    get_available_cpu_gpu_resources,
)
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


class _PrebatchPlanningStage(ProcessingStage[AudioTask, AudioTask]):
    name = "prebatch_planning"
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

    def batch_task_cost(self, task: AudioTask) -> float:
        return float(task.data["duration"])


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


class TestCalculateConcurrencyForActorsForStage:
    """Test class for calculate_concurrency_for_actors_for_stage function."""

    @patch("nemo_curator.backends.ray_data.utils.get_available_cpu_gpu_resources")
    def test_calculate_concurrency_explicit_num_workers(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when num_workers is explicitly set."""
        mock_stage = Mock(num_workers=lambda: 4, resources=Resources(cpus=2.0, gpus=0.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == 4
        mock_get_resources.assert_not_called()

    @patch("nemo_curator.backends.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_explicit_num_workers_zero_or_negative(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when num_workers is explicitly set to 0 or negative."""
        mock_stage = Mock(num_workers=lambda: 0, resources=Resources(cpus=2.0, gpus=0.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_cpu_only_constraint(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with CPU-only constraint."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=0.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 4.0))
    def test_calculate_concurrency_gpu_only_constraint(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with GPU-only constraint."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=0.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 4.0))
    def test_calculate_concurrency_both_cpu_gpu_constraints(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with both CPU and GPU constraints."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.ray_data.utils.get_available_cpu_gpu_resources", return_value=(4.0, 8.0))
    def test_calculate_concurrency_cpu_more_limiting(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when CPU is more limiting than GPU."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 2)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.ray_data.utils.get_available_cpu_gpu_resources", return_value=(16.0, 2.0))
    def test_calculate_concurrency_gpu_more_limiting(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when GPU is more limiting than CPU."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 2)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_no_resource_requirements(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when stage has no resource requirements."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=0.0, gpus=0.0))
        with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
            calculate_concurrency_for_actors_for_stage(mock_stage)

        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.ray_data.utils.get_available_cpu_gpu_resources", return_value=(1.0, 0.0))
    def test_calculate_concurrency_insufficient_resources(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when there are insufficient resources."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=4.0, gpus=2.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 0)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_fractional_resources(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with fractional resource requirements."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=0.5, gpus=0.25))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 8)
        mock_get_resources.assert_called_once()


class TestCoerceBatchTasks:
    def test_coerce_batch_tasks_from_numpy_object_array(self) -> None:
        sentinel = object()
        batch = np.array([sentinel], dtype=object)
        assert coerce_batch_tasks(batch) == [sentinel]

    def test_coerce_batch_tasks_empty(self) -> None:
        assert coerce_batch_tasks([]) == []
        assert coerce_batch_tasks(np.array([], dtype=object)) == []
        assert coerce_batch_tasks(None) == []


def test_preplanned_ray_data_task_adapter_preserves_planned_batches() -> None:
    first = AudioTask(data={})
    second = AudioTask(data={})
    third = AudioTask(data={})
    stage_map_fn = create_task_from_stage(_PreplannedEchoStage(), preplanned_batches=True)

    out = stage_map_fn({"item": [[first, second], [third]]})

    assert out["item"] == [first, second, third]
    assert first.data["seen_batch_size"] == 2
    assert second.data["seen_batch_size"] == 2
    assert third.data["seen_batch_size"] == 1


def test_ray_data_prebatch_planning_uses_distributed_windows() -> None:
    dataset = _FakeDataset()
    adapter = RayDataStageAdapter(_PrebatchPlanningStage())

    out = adapter._prebatch_dataset(dataset)

    assert out is dataset
    assert dataset.repartition_calls == []
    assert dataset.map_batches_calls[0][1]["batch_size"] == 4
    assert dataset.sample_output is not None
    planned_durations = [[task.data["duration"] for task in row] for row in dataset.sample_output["item"]]
    assert planned_durations == [[600.0], [5.0]]


def test_ray_data_scheduler_ready_stage_preserves_ready_rows() -> None:
    first = AudioTask(data={"duration": 5.0})
    second = AudioTask(data={"duration": 600.0})
    dataset = _FakeDataset(
        [
            SchedulerReadyTaskBatch(tasks=[first]),
            SchedulerReadyTaskBatch(tasks=[second]),
        ]
    )
    adapter = RayDataStageAdapter(_CentralizedPlanningStage())

    out = adapter.process_scheduler_ready_dataset(dataset)

    assert out is dataset
    assert dataset.repartition_calls == []
    assert len(dataset.map_batches_calls) == 1
    assert dataset.map_batches_calls[0][1]["batch_size"] == 1
    assert dataset.sample_output is not None
    processed_durations = [task.data["duration"] for task in dataset.sample_output["item"]]
    processed_batch_sizes = [task.data["processed_batch_size"] for task in dataset.sample_output["item"]]
    assert processed_durations == [5.0, 600.0]
    assert processed_batch_sizes == [1, 1]


def test_ray_data_adapter_processes_centralized_parent_windows_in_ray_data() -> None:
    dataset = _FakeDataset()
    adapter = RayDataStageAdapter(_CentralizedPlanningStage())

    out = adapter.process_dataset(dataset)

    assert out is dataset
    assert dataset.repartition_calls == []
    assert len(dataset.map_batches_calls) == 1
    assert dataset.map_batches_calls[0][1]["batch_size"] == 4
    assert dataset.sample_output is not None
    processed_durations = [task.data["duration"] for task in dataset.sample_output["item"]]
    processed_batch_sizes = [task.data["processed_batch_size"] for task in dataset.sample_output["item"]]
    assert processed_durations == [5.0, 600.0]
    assert processed_batch_sizes == [1, 1]


def test_ray_data_executor_globalizes_centralized_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = RayDataExecutor(ignore_head_node=True)
    stage = _CentralizedPlanningStage()
    input_dataset = object()
    ready_dataset = object()
    processed_dataset = object()
    output_dataset = object()
    first = AudioTask(data={"duration": 5.0})
    second = AudioTask(data={"duration": 600.0})
    calls: dict[str, object] = {}

    class FakeRayDataStageAdapter:
        def __init__(self, stage_arg: ProcessingStage) -> None:
            calls["stage"] = stage_arg

        def process_scheduler_ready_dataset(self, dataset_arg: object, ignore_head_node: bool) -> object:
            calls["ready_dataset"] = dataset_arg
            calls["ignore_head_node"] = ignore_head_node
            return processed_dataset

    dataset_to_tasks = Mock(side_effect=[[first, second], [second, first]])
    tasks_to_dataset = Mock(return_value=output_dataset)
    monkeypatch.setattr(
        executor,
        "_dataset_to_tasks",
        dataset_to_tasks,
    )
    monkeypatch.setattr(
        executor,
        "_tasks_to_dataset",
        tasks_to_dataset,
    )
    from_items = Mock(return_value=ready_dataset)
    monkeypatch.setattr("ray.data.from_items", from_items)
    monkeypatch.setattr("nemo_curator.backends.ray_data.executor.RayDataStageAdapter", FakeRayDataStageAdapter)

    out = executor._process_stage_dataset(stage, input_dataset)

    assert out is output_dataset
    dataset_to_tasks.assert_any_call(input_dataset)
    dataset_to_tasks.assert_any_call(processed_dataset)
    tasks_to_dataset.assert_called_once_with([first, second])
    ready_batches = from_items.call_args.args[0]
    assert [batch.tasks for batch in ready_batches] == [[second], [first]]
    assert calls == {
        "stage": stage,
        "ready_dataset": ready_dataset,
        "ignore_head_node": True,
    }


def test_ray_data_executor_keeps_noncentral_stage_in_ray_data(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = RayDataExecutor(ignore_head_node=True)
    stage = _PreplannedEchoStage()
    input_dataset = object()
    output_dataset = object()
    calls: dict[str, object] = {}

    class FakeRayDataStageAdapter:
        def __init__(self, stage_arg: ProcessingStage) -> None:
            calls["stage"] = stage_arg

        def process_dataset(self, dataset_arg: object, ignore_head_node: bool) -> object:
            calls["dataset"] = dataset_arg
            calls["ignore_head_node"] = ignore_head_node
            return output_dataset

    monkeypatch.setattr("nemo_curator.backends.ray_data.executor.RayDataStageAdapter", FakeRayDataStageAdapter)

    out = executor._process_stage_dataset(stage, input_dataset)

    assert out is output_dataset
    assert calls == {
        "stage": stage,
        "dataset": input_dataset,
        "ignore_head_node": True,
    }
